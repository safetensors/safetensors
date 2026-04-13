import os
import sys
from typing import Dict, List, Optional, Union

import numpy as np

from safetensors import deserialize, safe_open, serialize, serialize_file


def _flatten(
    tensor_dict: Dict[str, np.ndarray], keep_alive_buffer: List
) -> Dict[str, Dict]:
    flattened = {}
    for k, v in tensor_dict.items():
        tensor = v
        if not _is_little_endian(tensor):
            tensor = tensor.byteswap(inplace=False)
            keep_alive_buffer.append(tensor)
        flattened[k] = {
            "dtype": tensor.dtype.name,
            "shape": tensor.shape,
            "data_ptr": tensor.ctypes.data,
            "data_len": tensor.nbytes,
        }
    return flattened


def save(
    tensor_dict: Dict[str, np.ndarray],
    metadata: Optional[Dict[str, str]] = None,
    max_header_size: Optional[int] = None,
) -> bytes:
    """
    Saves a dictionary of tensors into raw bytes in safetensors format.

    Args:
        tensor_dict (`Dict[str, np.ndarray]`):
            The incoming tensors. Tensors need to be contiguous and dense.
        metadata (`Dict[str, str]`, *optional*, defaults to `None`):
            Optional text only metadata you might want to save in your header.
            For instance it can be useful to specify more about the underlying
            tensors. This is purely informative and does not affect tensor loading.
        max_header_size (`int`, *optional*):
            Maximum allowed header size in bytes. Defaults to 100MB.

    Returns:
        `bytes`: The raw bytes representing the format

    Example:

    ```python
    from safetensors.numpy import save
    import numpy as np

    tensors = {"embedding": np.zeros((512, 1024)), "attention": np.zeros((256, 256))}
    byte_data = save(tensors)
    ```
    """
    keep_alive_buffer = []  # to keep byteswapped tensors alive
    serialized = serialize(
        _flatten(tensor_dict, keep_alive_buffer),
        metadata=metadata,
        max_header_size=max_header_size,
    )
    result = bytes(serialized)
    return result


def save_file(
    tensor_dict: Dict[str, np.ndarray],
    filename: Union[str, os.PathLike],
    metadata: Optional[Dict[str, str]] = None,
    max_header_size: Optional[int] = None,
) -> None:
    """
    Saves a dictionary of tensors into raw bytes in safetensors format.

    Args:
        tensor_dict (`Dict[str, np.ndarray]`):
            The incoming tensors. Tensors need to be contiguous and dense.
        filename (`str`, or `os.PathLike`)):
            The filename we're saving into.
        metadata (`Dict[str, str]`, *optional*, defaults to `None`):
            Optional text only metadata you might want to save in your header.
            For instance it can be useful to specify more about the underlying
            tensors. This is purely informative and does not affect tensor loading.
        max_header_size (`int`, *optional*):
            Maximum allowed header size in bytes. Defaults to 100MB.

    Returns:
        `None`

    Example:

    ```python
    from safetensors.numpy import save_file
    import numpy as np

    tensors = {"embedding": np.zeros((512, 1024)), "attention": np.zeros((256, 256))}
    save_file(tensors, "model.safetensors")
    ```
    """
    keep_alive_buffer = []  # to keep byteswapped tensors alive
    serialize_file(
        _flatten(tensor_dict, keep_alive_buffer),
        filename,
        metadata=metadata,
        max_header_size=max_header_size,
    )


def load(
    data: bytes, max_header_size: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    Loads a safetensors file into numpy format from pure bytes.

    Args:
        data (`bytes`):
            The content of a safetensors file
        max_header_size (`int`, *optional*):
            Maximum allowed header size in bytes. Defaults to 100MB.

    Returns:
        `Dict[str, np.ndarray]`: dictionary that contains name as key, value as `np.ndarray` on cpu

    Example:

    ```python
    from safetensors.numpy import load

    file_path = "./my_folder/bert.safetensors"
    with open(file_path, "rb") as f:
        data = f.read()

    loaded = load(data)
    ```
    """
    flat = deserialize(data, max_header_size=max_header_size)
    return _view2np(flat)


def load_file(
    filename: Union[str, os.PathLike], max_header_size: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    Loads a safetensors file into numpy format.

    Args:
        filename (`str`, or `os.PathLike`)):
            The name of the file which contains the tensors
        max_header_size (`int`, *optional*):
            Maximum allowed header size in bytes. Defaults to 100MB.

    Returns:
        `Dict[str, np.ndarray]`: dictionary that contains name as key, value as `np.ndarray`

    Example:

    ```python
    from safetensors.numpy import load_file

    file_path = "./my_folder/bert.safetensors"
    loaded = load_file(file_path)
    ```
    """
    result = {}
    with safe_open(filename, framework="np", max_header_size=max_header_size) as f:
        for k in f.offset_keys():
            result[k] = f.get_tensor(k)
    return result


_TYPES = {
    "F64": np.float64,
    "F32": np.float32,
    "F16": np.float16,
    "I64": np.int64,
    "U64": np.uint64,
    "I32": np.int32,
    "U32": np.uint32,
    "I16": np.int16,
    "U16": np.uint16,
    "I8": np.int8,
    "U8": np.uint8,
    "BOOL": bool,
    "C64": np.complex64,
}


def _getdtype(dtype_str: str) -> np.dtype:
    return _TYPES[dtype_str]


def _view2np(safeview) -> Dict[str, np.ndarray]:
    result = {}
    for k, v in safeview:
        dtype = _getdtype(v["dtype"])
        arr = np.frombuffer(v["data"], dtype=dtype).reshape(v["shape"])
        result[k] = arr
    return result


def _is_little_endian(tensor: np.ndarray) -> bool:
    byteorder = tensor.dtype.byteorder
    if byteorder == "=":
        if sys.byteorder == "little":
            return True
        else:
            return False
    elif byteorder == "|":
        return True
    elif byteorder == "<":
        return True
    elif byteorder == ">":
        return False
    raise ValueError(f"Unexpected byte order {byteorder}")
