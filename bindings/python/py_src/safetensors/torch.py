import itertools
import math
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
from safetensors import (
    TensorSpec,
    deserialize,
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
        stop = (
            tensor.data_ptr()
            + (
                sum(
                    (shape - 1) * max(stride, 0)
                    for shape, stride in zip(tensor.shape, tensor.stride())
                )
                + 1
            )
            * _SIZE.get(tensor.dtype, tensor.element_size())
        )
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


def _ceil_div(n: int, d: int) -> int:
    return -(-n // d)


def _ext_gcd(a: int, b: int) -> Tuple[int, int, int]:
    x0, x1, y0, y1 = 1, 0, 0, 1
    while b != 0:
        q = a // b
        a, b = b, a % b
        x0, x1 = x1, x0 - q * x1
        y0, y1 = y1, y0 - q * y1
    return a, x0, y0


def _can_flatten_to_1d(shape: Tuple[int, ...], stride: Tuple[int, ...]) -> Tuple[bool, int, int]:
    if 0 in shape:
        return True, 0, 0
    dims = [(s, st) for s, st in zip(shape, stride) if s > 1]
    if not dims:
        return True, 1, 0
    if any(st <= 0 for _, st in dims):
        return False, 0, 0
    dims.sort(key=lambda x: x[1])
    base_step = dims[0][1]
    expected_stride = base_step
    total_len = 1
    for s, st in dims:
        if st != expected_stride:
            return False, 0, 0
        expected_stride *= s
        total_len *= s
    return True, total_len, base_step


def _solve_1d_diophantine_point(S1: int, n1: int, S2: int, n2: int, k: int) -> bool:
    g, x0, y0 = _ext_gcd(S1, S2)
    if k % g != 0:
        return False
    factor = k // g
    i0 = x0 * factor
    j0 = -y0 * factor

    step_i = S2 // g
    step_j = S1 // g

    t_min_i = _ceil_div(-i0, step_i)
    t_max_i = (n1 - 1 - i0) // step_i

    t_min_j = _ceil_div(-j0, step_j)
    t_max_j = (n2 - 1 - j0) // step_j

    t_min = max(t_min_i, t_min_j)
    t_max = min(t_max_i, t_max_j)

    return t_min <= t_max


def _solve_1d_byte_overlap(
    o1: int, n1: int, s1: int, w1: int,
    o2: int, n2: int, s2: int, w2: int,
) -> bool:
    if n1 == 0 or n2 == 0:
        return False

    S1 = s1 * w1
    S2 = s2 * w2

    if n1 == 1 or S1 == 0:
        if n2 == 1 or S2 == 0:
            return max(o1, o2) < min(o1 + w1, o2 + w2)
        L = (o1 - o2) - w2 + 1
        R = (o1 - o2) + w1 - 1
        j_min = max(0, _ceil_div(L, S2))
        j_max = min(n2 - 1, R // S2)
        return j_min <= j_max

    if n2 == 1 or S2 == 0:
        L = (o2 - o1) - w1 + 1
        R = (o2 - o1) + w2 - 1
        i_min = max(0, _ceil_div(L, S1))
        i_max = min(n1 - 1, R // S1)
        return i_min <= i_max

    max_b1 = o1 + (n1 - 1) * S1 + w1
    max_b2 = o2 + (n2 - 1) * S2 + w2
    if max_b1 <= o2 or max_b2 <= o1:
        return False

    delta = o2 - o1
    L = delta - w1 + 1
    R = delta + w2 - 1

    g = math.gcd(S1, S2)
    m_min = _ceil_div(L, g)
    m_max = R // g
    if m_min > m_max:
        return False

    for m in range(m_min, m_max + 1):
        k = m * g
        if _solve_1d_diophantine_point(S1, n1, S2, n2, k):
            return True

    return False


def _multidim_byte_gcd_reject(
    o1: int, shape1: Tuple[int, ...], strides1: Tuple[int, ...], w1: int,
    o2: int, shape2: Tuple[int, ...], strides2: Tuple[int, ...], w2: int,
) -> bool:
    S1 = [st * w1 for s, st in zip(shape1, strides1) if s > 1 and st > 0]
    S2 = [st * w2 for s, st in zip(shape2, strides2) if s > 1 and st > 0]

    all_strides = S1 + S2
    if not all_strides:
        return False

    g = all_strides[0]
    for s in all_strides[1:]:
        g = math.gcd(g, s)

    delta = o2 - o1
    L = delta - w1 + 1
    R = delta + w2 - 1

    m_min = _ceil_div(L, g)
    m_max = R // g
    return m_min > m_max


def _tensors_overlap(t1: torch.Tensor, t2: torch.Tensor, max_exact_numel: int = 5000) -> bool:
    if t1.device == torch.device("meta") or t2.device == torch.device("meta"):
        return False
    if t1.device != t2.device or storage_ptr(t1) != storage_ptr(t2):
        return False
    if t1.numel() == 0 or t2.numel() == 0:
        return False

    w1 = _SIZE.get(t1.dtype, t1.element_size())
    w2 = _SIZE.get(t2.dtype, t2.element_size())
    o1 = t1.data_ptr()
    o2 = t2.data_ptr()

    # Tier 1: Bounding-box rejection
    stop1 = o1 + (sum((s - 1) * max(st, 0) for s, st in zip(t1.shape, t1.stride())) + 1) * w1
    stop2 = o2 + (sum((s - 1) * max(st, 0) for s, st in zip(t2.shape, t2.stride())) + 1) * w2
    if stop1 <= o2 or stop2 <= o1:
        return False

    # Tier 2: Multidimensional byte GCD rejection
    if _multidim_byte_gcd_reject(o1, t1.shape, t1.stride(), w1, o2, t2.shape, t2.stride(), w2):
        return False

    # Tier 3: 1D Diophantine solver (including legitimately flattened multidimensional tensors)
    can1, n1, s1 = _can_flatten_to_1d(t1.shape, t1.stride())
    can2, n2, s2 = _can_flatten_to_1d(t2.shape, t2.stride())
    if can1 and can2:
        return _solve_1d_byte_overlap(o1, n1, s1, w1, o2, n2, s2, w2)

    # Tier 4: Exact element byte-address checking for small bounded tensors
    if min(t1.numel(), t2.numel()) <= max_exact_numel:
        if t1.numel() <= t2.numel():
            small_t, small_o, small_w = t1, o1, w1
            large_t, large_o, large_w = t2, o2, w2
        else:
            small_t, small_o, small_w = t2, o2, w2
            large_t, large_o, large_w = t1, o1, w1

        small_bytes = set()
        for idx in itertools.product(*[range(s) for s in small_t.shape]):
            elem_addr = small_o + sum(i * st for i, st in zip(idx, small_t.stride())) * small_w
            for b in range(elem_addr, elem_addr + small_w):
                small_bytes.add(b)

        for idx in itertools.product(*[range(s) for s in large_t.shape]):
            elem_addr = large_o + sum(i * st for i, st in zip(idx, large_t.stride())) * large_w
            for b in range(elem_addr, elem_addr + large_w):
                if b in small_bytes:
                    return True
        return False

    # Tier 5: Conservative fallback for large unresolved multidimensional layouts
    return True


def _filter_shared_not_shared(
    tensors: List[Set[str]], state_dict: Dict[str, torch.Tensor]
) -> List[Set[str]]:
    filtered_tensors = []
    for shared in tensors:
        if len(shared) < 2:
            filtered_tensors.append(shared)
            continue

        names = sorted(list(shared))
        parent = {name: name for name in names}

        def find(x: str) -> str:
            path = []
            while parent[x] != x:
                path.append(x)
                x = parent[x]
            for p in path:
                parent[p] = x
            return x

        def union(x: str, y: str) -> None:
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[ry] = rx

        for i in range(len(names)):
            t1 = state_dict[names[i]]
            for j in range(i + 1, len(names)):
                t2 = state_dict[names[j]]
                if _tensors_overlap(t1, t2):
                    union(names[i], names[j])

        components = defaultdict(set)
        for name in names:
            components[find(name)].add(name)

        filtered_tensors.extend(sorted(components.values(), key=lambda s: sorted(list(s))[0]))

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


def _is_canonical_dense_stride(shape: Tuple[int, ...], stride: Tuple[int, ...]) -> bool:
    if len(shape) != len(stride):
        return False
    if any(st < 0 for st in stride):
        return False
    if 0 in shape:
        return True
    pairs = [(s, st) for s, st in zip(shape, stride) if s > 1]
    if not pairs:
        return True
    if any(st <= 0 for _, st in pairs):
        return False
    pairs.sort(key=lambda x: x[1])
    if pairs[0][1] != 1:
        return False
    expected_stride = 1
    for i in range(len(pairs) - 1):
        expected_stride *= pairs[i][0]
        if pairs[i + 1][1] != expected_stride:
            return False
    return True


def _is_complete(tensor: torch.Tensor) -> bool:
    return (
        tensor.data_ptr() == storage_ptr(tensor)
        and tensor.nelement() * _SIZE[tensor.dtype] == storage_size(tensor)
        and _is_canonical_dense_stride(tensor.shape, tensor.stride())
    )


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

    storage_counts = defaultdict(int)
    for v in state_dict.values():
        if (
            v.device != torch.device("meta")
            and storage_ptr(v) != 0
            and storage_size(v) != 0
        ):
            storage_counts[(v.device, storage_ptr(v), storage_size(v))] += 1

    shareds = _find_shared_tensors(state_dict)
    to_remove = defaultdict(list)
    for shared in shareds:
        complete_names = set(
            [name for name in shared if _is_complete(state_dict[name])]
        )
        if not complete_names:
            if len(shared) == 1:
                name = list(shared)[0]
                t = state_dict[name]
                # A singleton group is safe to clone into independent contiguous storage ONLY IF:
                # 1. It does NOT use zero-stride broadcasting (e.g. expand), AND
                # 2. It was partitioned from a multi-tensor shared storage group where the grouping
                #    algorithm established that no other relevant tensor overlaps it.
                has_zero_stride = any(st == 0 for s, st in zip(t.shape, t.stride()) if s > 1)
                was_partitioned = storage_counts.get((t.device, storage_ptr(t), storage_size(t)), 0) > 1

                if not has_zero_stride and was_partitioned:
                    state_dict[name] = t.clone()
                    complete_names = {name}
                else:
                    raise RuntimeError(
                        "Error while trying to find names to remove to save state dict, but found no suitable name to keep"
                        f" for saving amongst: {shared}. None is covering the entire storage.Refusing to save/load the model"
                        " since you could be storing much more memory than needed. Please refer to"
                        " https://huggingface.co/docs/safetensors/torch_shared_tensors for more information. Or open an"
                        " issue."
                    )
            else:
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
    *,
    backend: str = "mmap",
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
        backend (`str`, *optional*, defaults to `"mmap"`):
            Storage backend used to serve tensor bytes. `"mmap"` (default)
            and `"pread"` uses `pread(2)` to read tensor bytes.

    Returns:
        `(missing, unexpected): (List[str], List[str])`
            `missing` are names in the model which were not modified during loading
            `unexpected` are names that are on the file, but weren't used during
            the load.
    """
    state_dict = load_file(filename, device=device, backend=backend)
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


def load_file(
    filename: Union[str, os.PathLike],
    device: Union[str, int] = "cpu",
    *,
    backend: str = "mmap",
) -> Dict[str, torch.Tensor]:
    """
    Loads a safetensors file into torch format.

    Args:
        filename (`str`, or `os.PathLike`):
            The name of the file which contains the tensors
        device (`Union[str, int]`, *optional*, defaults to `cpu`):
            The device where the tensors need to be located after load.
            available options are all regular torch device locations.
        backend (`str`, *optional*, defaults to `"mmap"`):
            Storage backend used to serve tensor bytes. `"mmap"` (default)
            and `"pread"` uses `pread(2)` to read tensor bytes.

    Returns:
        `Dict[str, torch.Tensor]`: dictionary that contains name as key, value as `torch.Tensor`

    Example:

    ```python
    from safetensors.torch import load_file

    file_path = "./my_folder/bert.safetensors"
    loaded = load_file(file_path)
    ```
    """
    with safe_open(filename, framework="pt", device=device, backend=backend) as f:
        return f.get_tensors()


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
