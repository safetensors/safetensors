import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
from safetensors import (
    TensorSpec,
    deserialize,
    mps_load_safetensors as _mps_load_safetensors,
    safe_open,
    serialize,
    serialize_file,
)


def storage_ptr(tensor: torch.Tensor) -> int:
    try:
        return tensor.untyped_storage().data_ptr()
    except Exception:
        # Fallback for torch==1.10
        try:
            return tensor.storage().data_ptr()
        except NotImplementedError:
            # Fallback for meta storage
            return 0


def _end_ptr(tensor: torch.Tensor) -> int:
    if tensor.nelement():
        stop = tensor.view(-1)[-1].data_ptr() + _SIZE[tensor.dtype]
    else:
        stop = tensor.data_ptr()
    return stop


def storage_size(tensor: torch.Tensor) -> int:
    try:
        return tensor.untyped_storage().nbytes()
    except AttributeError:
        # Fallback for torch==1.10
        try:
            return tensor.storage().size() * _SIZE[tensor.dtype]
        except NotImplementedError:
            # Fallback for meta storage
            # On torch >=2.0 this is the tensor size
            return tensor.nelement() * _SIZE[tensor.dtype]


def _filter_shared_not_shared(
    tensors: List[Set[str]], state_dict: Dict[str, torch.Tensor]
) -> List[Set[str]]:
    filtered_tensors = []
    for shared in tensors:
        if len(shared) < 2:
            filtered_tensors.append(shared)
            continue

        areas = []
        for name in shared:
            tensor = state_dict[name]
            areas.append((tensor.data_ptr(), _end_ptr(tensor), name))
        areas.sort()

        _, last_stop, last_name = areas[0]
        filtered_tensors.append({last_name})
        for start, stop, name in areas[1:]:
            if start >= last_stop:
                filtered_tensors.append({name})
            else:
                filtered_tensors[-1].add(name)
            last_stop = stop

    return filtered_tensors


def _find_shared_tensors(state_dict: Dict[str, torch.Tensor]) -> List[Set[str]]:
    tensors = defaultdict(set)
    for k, v in state_dict.items():
        if (
            v.device != torch.device("meta")
            and storage_ptr(v) != 0
            and storage_size(v) != 0
        ):
            # Need to add device as key because of multiple GPU.
            tensors[(v.device, storage_ptr(v), storage_size(v))].add(k)
    tensors = list(sorted(tensors.values()))
    tensors = _filter_shared_not_shared(tensors, state_dict)
    return tensors


def _is_complete(tensor: torch.Tensor) -> bool:
    return tensor.data_ptr() == storage_ptr(tensor) and tensor.nelement() * _SIZE[
        tensor.dtype
    ] == storage_size(tensor)


def _remove_duplicate_names(
    state_dict: Dict[str, torch.Tensor],
    *,
    preferred_names: Optional[List[str]] = None,
    discard_names: Optional[List[str]] = None,
) -> Dict[str, List[str]]:
    if preferred_names is None:
        preferred_names = []
    preferred_names = set(preferred_names)
    if discard_names is None:
        discard_names = []
    discard_names = set(discard_names)

    shareds = _find_shared_tensors(state_dict)
    to_remove = defaultdict(list)
    for shared in shareds:
        complete_names = set(
            [name for name in shared if _is_complete(state_dict[name])]
        )
        if not complete_names:
            raise RuntimeError(
                "Error while trying to find names to remove to save state dict, but found no suitable name to keep"
                f" for saving amongst: {shared}. None is covering the entire storage.Refusing to save/load the model"
                " since you could be storing much more memory than needed. Please refer to"
                " https://huggingface.co/docs/safetensors/torch_shared_tensors for more information. Or open an"
                " issue."
            )

        keep_name = sorted(list(complete_names))[0]

        # Mechanism to preferentially select keys to keep
        # coming from the on-disk file to allow
        # loading models saved with a different choice
        # of keep_name
        preferred = complete_names.difference(discard_names)
        if preferred:
            keep_name = sorted(list(preferred))[0]

        if preferred_names:
            preferred = preferred_names.intersection(complete_names)
            if preferred:
                keep_name = sorted(list(preferred))[0]
        for name in sorted(shared):
            if name != keep_name:
                to_remove[keep_name].append(name)
    return to_remove


def save_model(
    model: torch.nn.Module,
    filename: str,
    metadata: Optional[Dict[str, str]] = None,
    force_contiguous: bool = True,
):
    """
    Saves a given torch model to specified filename.
    This method exists specifically to avoid tensor sharing issues which are
    not allowed in `safetensors`. [More information on tensor sharing](../torch_shared_tensors)

    Args:
        model (`torch.nn.Module`):
            The model to save on disk.
        filename (`str`):
            The filename location to save the file
        metadata (`Dict[str, str]`, *optional*):
            Extra information to save along with the file.
            Some metadata will be added for each dropped tensors.
            This information will not be enough to recover the entire
            shared structure but might help understanding things
        force_contiguous (`boolean`, *optional*, defaults to True):
            Forcing the state_dict to be saved as contiguous tensors.
            This has no effect on the correctness of the model, but it
            could potentially change performance if the layout of the tensor
            was chosen specifically for that reason.
    """
    state_dict = model.state_dict()
    to_removes = _remove_duplicate_names(state_dict)

    for kept_name, to_remove_group in to_removes.items():
        for to_remove in to_remove_group:
            if metadata is None:
                metadata = {}

            if to_remove not in metadata:
                # Do not override user data
                metadata[to_remove] = kept_name
            del state_dict[to_remove]
    if force_contiguous:
        state_dict = {k: v.contiguous() for k, v in state_dict.items()}
    try:
        save_file(state_dict, filename, metadata=metadata)
    except ValueError as e:
        msg = str(e)
        msg += " Or use save_model(..., force_contiguous=True), read the docs for potential caveats."
        raise ValueError(msg)


def load_model(
    model: torch.nn.Module,
    filename: Union[str, os.PathLike],
    strict: bool = True,
    device: Union[str, int] = "cpu",
) -> Tuple[List[str], List[str]]:
    """
    Loads a given filename onto a torch model.
    This method exists specifically to avoid tensor sharing issues which are
    not allowed in `safetensors`. [More information on tensor sharing](../torch_shared_tensors)

    Args:
        model (`torch.nn.Module`):
            The model to load onto.
        filename (`str`, or `os.PathLike`):
            The filename location to load the file from.
        strict (`bool`, *optional*, defaults to True):
            Whether to fail if you're missing keys or having unexpected ones.
            When false, the function simply returns missing and unexpected names.
        device (`Union[str, int]`, *optional*, defaults to `cpu`):
            The device where the tensors need to be located after load.
            available options are all regular torch device locations.

    Returns:
        `(missing, unexpected): (List[str], List[str])`
            `missing` are names in the model which were not modified during loading
            `unexpected` are names that are on the file, but weren't used during
            the load.
    """
    state_dict = load_file(filename, device=device)
    model_state_dict = model.state_dict()
    to_removes = _remove_duplicate_names(
        model_state_dict, preferred_names=state_dict.keys()
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    missing = set(missing)
    for to_remove_group in to_removes.values():
        for to_remove in to_remove_group:
            if to_remove not in missing:
                unexpected.append(to_remove)
            else:
                missing.remove(to_remove)
    if strict and (missing or unexpected):
        missing_keys = ", ".join([f'"{k}"' for k in sorted(missing)])
        unexpected_keys = ", ".join([f'"{k}"' for k in sorted(unexpected)])
        error = f"Error(s) in loading state_dict for {model.__class__.__name__}:"
        if missing:
            error += f"\n    Missing key(s) in state_dict: {missing_keys}"
        if unexpected:
            error += f"\n    Unexpected key(s) in state_dict: {unexpected_keys}"
        raise RuntimeError(error)
    return missing, unexpected


def save(
    tensors: Dict[str, torch.Tensor], metadata: Optional[Dict[str, str]] = None
) -> bytes:
    """
    Saves a dictionary of tensors into raw bytes in safetensors format.

    Args:
        tensors (`Dict[str, torch.Tensor]`):
            The incoming tensors. Tensors need to be contiguous and dense.
        metadata (`Dict[str, str]`, *optional*, defaults to `None`):
            Optional text only metadata you might want to save in your header.
            For instance it can be useful to specify more about the underlying
            tensors. This is purely informative and does not affect tensor loading.

    Returns:
        `bytes`: The raw bytes representing the format

    Example:

    ```python
    from safetensors.torch import save
    import torch

    tensors = {"embedding": torch.zeros((512, 1024)), "attention": torch.zeros((256, 256))}
    byte_data = save(tensors)
    ```
    """
    keep_references_alive = []  # to avoid garbage collection of temporary numpy arrays while we write to disk
    serialized = serialize(
        _flatten_as_ptr(tensors, keep_references_alive), metadata=metadata
    )
    result = bytes(serialized)
    return result


def save_file(
    tensors: Dict[str, torch.Tensor],
    filename: Union[str, os.PathLike],
    metadata: Optional[Dict[str, str]] = None,
):
    """
    Saves a dictionary of tensors into `filename` in safetensors format.
    There is no mechanism in place to prevent the caller from modifying the data while a file save occurs,
    please be wary when calling `save_file` and modifying tensors referenced in the `tensors` dict concurrently;
    it may lead to corrupted files.

    Args:
        tensors (`Dict[str, torch.Tensor]`):
            The incoming tensors. Tensors need to be contiguous and dense.
        filename (`str`, or `os.PathLike`)):
            The filename we're saving into.
        metadata (`Dict[str, str]`, *optional*, defaults to `None`):
            Optional text only metadata you might want to save in your header.
            For instance it can be useful to specify more about the underlying
            tensors. This is purely informative and does not affect tensor loading.

    Returns:
        `None`

    Example:

    ```python
    from safetensors.torch import save_file
    import torch

    tensors = {"embedding": torch.zeros((512, 1024)), "attention": torch.zeros((256, 256))}
    save_file(tensors, "model.safetensors")
    ```
    """
    keep_references_alive = []  # to avoid garbage collection of temporary numpy arrays while we write to disk
    serialize_file(
        _flatten_as_ptr(tensors, keep_references_alive), filename, metadata=metadata
    )


def _is_mps_device(device: Union[str, int, torch.device]) -> bool:
    if isinstance(device, torch.device):
        return device.type == "mps" and device.index in (None, 0)
    if isinstance(device, str):
        return device == "mps" or device == "mps:0"
    return False


def load_file(
    filename: Union[str, os.PathLike], device: Union[str, int] = "cpu"
) -> Dict[str, torch.Tensor]:
    """
    Loads a safetensors file into torch format.

    Args:
        filename (`str`, or `os.PathLike`):
            The name of the file which contains the tensors
        device (`Union[str, int]`, *optional*, defaults to `cpu`):
            The device where the tensors need to be located after load.
            available options are all regular torch device locations.

    Returns:
        `Dict[str, torch.Tensor]`: dictionary that contains name as key, value as `torch.Tensor`

    Example:

    ```python
    from safetensors.torch import load_file

    file_path = "./my_folder/bert.safetensors"
    loaded = load_file(file_path)
    ```
    """
    # Fast path: parallel pread straight into MPS buffers via host-alias storage
    # (pytorch/pytorch#179190, PR #180961). Falls through on stock PyTorch.
    if (
        _is_mps_device(device)
        and _mps_load_safetensors is not None
        and hasattr(torch, "mps")
        and hasattr(torch.mps, "_host_alias_storage")
    ):
        return _mps_load_safetensors(os.fspath(filename))
    result = {}
    with safe_open(filename, framework="pt", device=device) as f:
        for k in f.offset_keys():
            result[k] = f.get_tensor(k)
    return result


def load(data: bytes) -> Dict[str, torch.Tensor]:
    """
    Loads a safetensors file into torch format from pure bytes.

    Args:
        data (`bytes`):
            The content of a safetensors file

    Returns:
        `Dict[str, torch.Tensor]`: dictionary that contains name as key, value as `torch.Tensor` on cpu

    Example:

    ```python
    from safetensors.torch import load

    file_path = "./my_folder/bert.safetensors"
    with open(file_path, "rb") as f:
        data = f.read()

    loaded = load(data)
    ```
    """
    flat = deserialize(data)
    return _view2torch(flat)


# torch.float8 formats require 2.1; we do not support these dtypes on earlier versions
_float8_e4m3fn = getattr(torch, "float8_e4m3fn", None)
_float8_e4m3fnuz = getattr(torch, "float8_e4m3fnuz", None)
_float8_e5m2 = getattr(torch, "float8_e5m2", None)
_float8_e5m2fnuz = getattr(torch, "float8_e5m2fnuz", None)
_float8_e8m0 = getattr(torch, "float8_e8m0fnu", None)
_float4_e2m1_x2 = getattr(torch, "float4_e2m1fn_x2", None)

_SIZE = {
    torch.int64: 8,
    torch.float32: 4,
    torch.int32: 4,
    torch.bfloat16: 2,
    torch.float16: 2,
    torch.int16: 2,
    torch.uint8: 1,
    torch.int8: 1,
    torch.bool: 1,
    torch.float64: 8,
    torch.complex64: 8,
    _float8_e4m3fn: 1,
    _float8_e4m3fnuz: 1,
    _float8_e5m2: 1,
    _float8_e5m2fnuz: 1,
    _float8_e8m0: 1,
    _float4_e2m1_x2: 1,
}

if hasattr(torch, "uint64"):  # Torch 2.3.0+
    _SIZE.update(
        {
            torch.uint64: 8,
            torch.uint32: 4,
            torch.uint16: 2,
        }
    )

_TYPES = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
    "F8_E4M3": _float8_e4m3fn,
    "F8_E4M3FNUZ": _float8_e4m3fnuz,
    "F8_E5M2": _float8_e5m2,
    "F8_E5M2FNUZ": _float8_e5m2fnuz,
    "C64": torch.complex64,
}

if hasattr(torch, "uint64"):  # Torch 2.3.0+
    _TYPES.update(
        {
            "U64": torch.uint64,
            "U32": torch.uint32,
            "U16": torch.uint16,
        }
    )


def _getdtype(dtype_str: str) -> torch.dtype:
    return _TYPES[dtype_str]


def _view2torch(safeview) -> Dict[str, torch.Tensor]:
    result = {}
    for k, v in safeview:
        dtype = _getdtype(v["dtype"])
        if len(v["data"]) == 0:
            # Workaround because frombuffer doesn't accept zero-size tensors
            assert any(x == 0 for x in v["shape"])
            arr = torch.empty(v["shape"], dtype=dtype)
        else:
            arr = torch.frombuffer(v["data"], dtype=dtype).reshape(v["shape"])
        if sys.byteorder == "big":
            arr = torch.from_numpy(arr.numpy().byteswap(inplace=False))
        result[k] = arr

    return result


def _to_ndarray(tensor: torch.Tensor):
    if tensor.device.type != "cpu":
        # Moving tensor to cpu before saving
        tensor = tensor.to("cpu")

    import ctypes

    import numpy as np

    # When shape is empty (scalar), np.prod returns a float
    # we need a int for the following calculations
    length = int(np.prod(tensor.shape).item())
    bytes_per_item = _SIZE[tensor.dtype]

    total_bytes = length * bytes_per_item

    ptr = tensor.data_ptr()
    if ptr == 0:
        return np.empty(
            0
        ), 0  # XXX: bogus value we don't really care if we return a tensor here
    newptr = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_ubyte))
    data = np.ctypeslib.as_array(newptr, (total_bytes,))  # no internal copy
    if sys.byteorder == "big":
        NPDTYPES = {
            torch.int64: np.int64,
            torch.float32: np.float32,
            torch.int32: np.int32,
            # XXX: This is ok because both have the same width
            torch.bfloat16: np.float16,
            torch.float16: np.float16,
            torch.int16: np.int16,
            torch.uint8: np.uint8,
            torch.int8: np.int8,
            torch.bool: bool,
            torch.float64: np.float64,
            # XXX: This is ok because both have the same width and byteswap is a no-op anyway
            _float8_e4m3fn: np.uint8,
            _float8_e4m3fnuz: np.uint8,
            _float8_e5m2: np.uint8,
            _float8_e5m2fnuz: np.uint8,
            _float8_e8m0: np.uint8,
            _float4_e2m1_x2: np.uint8,
            torch.complex64: np.complex64,
        }
        npdtype = NPDTYPES[tensor.dtype]
        # Not in place as that would potentially modify a live running model
        data = data.view(npdtype).byteswap(inplace=False)
    return data, tensor


def _evaluate_tensors_for_save(tensors: Dict[str, torch.Tensor]) -> None:
    if not isinstance(tensors, dict):
        raise ValueError(
            f"Expected a dict of [str, torch.Tensor] but received {type(tensors)}"
        )

    sparse_tensors = []
    for k, v in tensors.items():
        if not isinstance(v, torch.Tensor):
            raise ValueError(
                f"Key `{k}` is invalid, expected torch.Tensor but received {type(v)}"
            )

        if v.layout != torch.strided:
            sparse_tensors.append(k)

    if sparse_tensors:
        raise ValueError(
            f"You are trying to save a sparse tensors: `{sparse_tensors}` which this library does not support."
            " You can make it a dense tensor before saving with `.to_dense()` but be aware this might"
            " make a much larger file than needed."
        )

    shared_pointers = _find_shared_tensors(tensors)
    failing = []
    for names in shared_pointers:
        if len(names) > 1:
            failing.append(names)

    if failing:
        raise RuntimeError(
            f"""
            Some tensors share memory, this will lead to duplicate memory on disk and potential differences when loading them again: {failing}.
            A potential way to correctly save your model is to use `save_model`.
            More information at https://huggingface.co/docs/safetensors/torch_shared_tensors
            """
        )


def _flatten_as_ptr(
    tensors: Dict[str, torch.Tensor], keep_alive_buffer: List
) -> Dict[str, Dict[str, Any]]:
    _evaluate_tensors_for_save(tensors)
    flattened = {}
    for k, v in tensors.items():
        # XXX: doing this check later on instead of in _evaluate_tensors_for_save
        # since on old versions of torch, SparseTensorImpl do not implement is_contiguous
        # and we do the sparsity check in _evaluate_tensors_for_save.
        if not v.is_contiguous():
            raise ValueError(
                f"You are trying to save a non contiguous tensor: `{k}` which is not allowed. It either means you"
                " are trying to save tensors which are reference of each other in which case it's recommended to save"
                " only the full tensors, and reslice at load time, or simply call `.contiguous()` on your tensor to"
                " pack it before saving."
            )
        arr, tensor_ref = _to_ndarray(v)
        keep_alive_buffer.append((arr, tensor_ref))
        flattened[k] = TensorSpec(
            dtype=str(v.dtype).split(".")[-1],
            shape=v.shape,
            data_ptr=arr.ctypes.data,
            data_len=arr.nbytes,
        )
    return flattened
