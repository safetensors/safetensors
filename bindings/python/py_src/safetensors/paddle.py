import os
import sys
from typing import Any, Dict, List, Optional, Union

import numpy as np
import paddle

from safetensors import TensorSpec, numpy, deserialize, safe_open, serialize, serialize_file


def save(
    tensors: Dict[str, paddle.Tensor], metadata: Optional[Dict[str, str]] = None
) -> bytes:
    """
    Saves a dictionary of tensors into raw bytes in safetensors format.

    Args:
        tensors (`Dict[str, paddle.Tensor]`):
            The incoming tensors. Tensors need to be contiguous and dense.
        metadata (`Dict[str, str]`, *optional*, defaults to `None`):
            Optional text only metadata you might want to save in your header.
            For instance it can be useful to specify more about the underlying
            tensors. This is purely informative and does not affect tensor loading.

    Returns:
        `bytes`: The raw bytes representing the format

    Example:

    ```python
    from safetensors.paddle import save
    import paddle

    tensors = {"embedding": paddle.zeros((512, 1024)), "attention": paddle.zeros((256, 256))}
    byte_data = save(tensors)
    ```
    """
    keep_references_alive = []
    serialized = serialize(_flatten(tensors, keep_references_alive), metadata=metadata)
    result = bytes(serialized)
    return result


def save_file(
    tensors: Dict[str, paddle.Tensor],
    filename: Union[str, os.PathLike],
    metadata: Optional[Dict[str, str]] = None,
) -> None:
    """
    Saves a dictionary of tensors into raw bytes in safetensors format.

    Args:
        tensors (`Dict[str, paddle.Tensor]`):
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
    from safetensors.paddle import save_file
    import paddle

    tensors = {"embedding": paddle.zeros((512, 1024)), "attention": paddle.zeros((256, 256))}
    save_file(tensors, "model.safetensors")
    ```
    """
    keep_references_alive = []
    serialize_file(
        _flatten(tensors, keep_references_alive), filename, metadata=metadata
    )


def load(data: bytes, device: str = "cpu") -> Dict[str, paddle.Tensor]:
    """
    Loads a safetensors file into paddle format from pure bytes.

    Args:
        data (`bytes`):
            The content of a safetensors file

    Returns:
        `Dict[str, paddle.Tensor]`: dictionary that contains name as key, value as `paddle.Tensor` on cpu

    Example:

    ```python
    from safetensors.paddle import load

    file_path = "./my_folder/bert.safetensors"
    with open(file_path, "rb") as f:
        data = f.read()

    loaded = load(data)
    ```
    """
    if paddle.__version__ >= "3.2.0":
        flat = deserialize(data)
        return _view2paddle(flat, device)
    else:
        flat = numpy.load(data)
        return _np2paddle(flat, device)


def load_file(
    filename: Union[str, os.PathLike], device="cpu"
) -> Dict[str, paddle.Tensor]:
    """
    Loads a safetensors file into paddle format.

    Args:
        filename (`str`, or `os.PathLike`)):
            The name of the file which contains the tensors
        device (`Union[Dict[str, any], str]`, *optional*, defaults to `cpu`):
            The device where the tensors need to be located after load.
            available options are all regular paddle device locations

    Returns:
        `Dict[str, paddle.Tensor]`: dictionary that contains name as key, value as `paddle.Tensor`

    Example:

    ```python
    from safetensors.paddle import load_file

    file_path = "./my_folder/bert.safetensors"
    loaded = load_file(file_path)
    ```
    """
    result = {}
    if paddle.__version__ >= "3.2.0":
        with safe_open(filename, framework="paddle", device=device) as f:
            for k in f.offset_keys():
                result[k] = f.get_tensor(k)
    else:
        flat = numpy.load_file(filename)
        result = _np2paddle(flat, device)
    return result


def _np2paddle(
    numpy_dict: Dict[str, np.ndarray], device: str = "cpu"
) -> Dict[str, paddle.Tensor]:
    for k, v in numpy_dict.items():
        numpy_dict[k] = paddle.to_tensor(v, place=device)
    return numpy_dict


def _paddle2np(paddle_dict: Dict[str, paddle.Tensor]) -> Dict[str, np.array]:
    for k, v in paddle_dict.items():
        paddle_dict[k] = v.detach().cpu().numpy()
    return paddle_dict


_SIZE = {
    paddle.int64: 8,
    paddle.float32: 4,
    paddle.int32: 4,
    paddle.bfloat16: 2,
    paddle.float16: 2,
    paddle.int16: 2,
    paddle.uint8: 1,
    paddle.int8: 1,
    paddle.bool: 1,
    paddle.float64: 8,
    paddle.float8_e4m3fn: 1,
    paddle.float8_e5m2: 1,
    paddle.complex64: 8,
    # XXX: These are not supported yet in paddle
    # paddle.uint64: 8,
    # paddle.uint32: 4,
    # paddle.uint16: 2,
    # paddle.float8_e8m0: 1,
    # paddle.float4_e2m1_x2: 1,
}

_TYPES = {
    "F64": paddle.float64,
    "F32": paddle.float32,
    "F16": paddle.float16,
    "BF16": paddle.bfloat16,
    "I64": paddle.int64,
    "I32": paddle.int32,
    "I16": paddle.int16,
    "I8": paddle.int8,
    "U8": paddle.uint8,
    "BOOL": paddle.bool,
    "F8_E4M3": paddle.float8_e4m3fn,
    "F8_E5M2": paddle.float8_e5m2,
}

NPDTYPES = {
    paddle.int64: np.int64,
    paddle.float32: np.float32,
    paddle.int32: np.int32,
    # XXX: This is ok because both have the same width
    paddle.bfloat16: np.float16,
    paddle.float16: np.float16,
    paddle.int16: np.int16,
    paddle.uint8: np.uint8,
    paddle.int8: np.int8,
    paddle.bool: bool,
    paddle.float64: np.float64,
    # XXX: This is ok because both have the same width and byteswap is a no-op anyway
    paddle.float8_e4m3fn: np.uint8,
    paddle.float8_e5m2: np.uint8,
}


def _getdtype(dtype_str: str) -> paddle.dtype:
    return _TYPES[dtype_str]


def _view2paddle(safeview, device) -> Dict[str, paddle.Tensor]:
    result = {}
    for k, v in safeview:
        dtype = _getdtype(v["dtype"])
        if len(v["data"]) == 0:
            # Workaround because frombuffer doesn't accept zero-size tensors
            assert any(x == 0 for x in v["shape"])
            arr = paddle.empty(v["shape"], dtype=dtype)
        else:
            arr = paddle.base.core.frombuffer(v["data"], dtype).reshape(v["shape"])
            if device != "cpu":
                arr = arr.to(device)
        if sys.byteorder == "big":
            arr = paddle.to_tensor(arr.numpy().byteswap(inplace=False), place=device)
        result[k] = arr

    return result


def _to_ndarray(tensor: paddle.Tensor, name: str):
    if not tensor.is_contiguous():
        raise ValueError(
            f"You are trying to save a non contiguous tensor: `{name}` which is not allowed. It either means you"
            " are trying to save tensors which are reference of each other in which case it's recommended to save"
            " only the full tensors, and reslice at load time, or simply call `.contiguous()` on your tensor to"
            " pack it before saving."
        )
    if not tensor.place.is_cpu_place():
        # Moving tensor to cpu before saving
        tensor = tensor.cpu()

    import ctypes

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
        npdtype = NPDTYPES[tensor.dtype]
        # Not in place as that would potentially modify a live running model
        data = data.view(npdtype).byteswap(inplace=False)
    return data, tensor


def _flatten(
    tensors: Dict[str, paddle.Tensor], keep_alive_buffer: List
) -> Dict[str, Dict[str, Any]]:
    if not isinstance(tensors, dict):
        raise ValueError(
            f"Expected a dict of [str, paddle.Tensor] but received {type(tensors)}"
        )

    for k, v in tensors.items():
        if not isinstance(v, paddle.Tensor):
            raise ValueError(
                f"Key `{k}` is invalid, expected paddle.Tensor but received {type(v)}"
            )

    flattened = {}
    for k, v in tensors.items():
        arr, tensor_ref = _to_ndarray(v, k)
        keep_alive_buffer.append((arr, tensor_ref))
        flattened[k] = TensorSpec(
            dtype=str(v.dtype).split(".")[-1],
            shape=v.shape,
            data_ptr=arr.ctypes.data,
            data_len=arr.nbytes,
        )
    return flattened
