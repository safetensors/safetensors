import os
import sys
from typing import Dict, Optional, Union

import numpy as np

from safetensors import (
    FileLayout,
    TensorSpec,
    deserialize,
    safe_open,
    serialize,
    serialize_file,
)


def _flatten(tensor_dict: Dict[str, np.ndarray]) -> Dict[str, TensorSpec]:
    flattened = {}
    for k, v in tensor_dict.items():
        # C order and little-endian, in one copy at most
        tensor = np.ascontiguousarray(v, dtype=v.dtype.newbyteorder("<"))
        flattened[k] = TensorSpec(
            dtype=v.dtype.name,
            shape=v.shape,
            data=tensor.reshape(-1).view(np.uint8),
        )
    return flattened


def save(
    tensor_dict: Dict[str, np.ndarray], metadata: Optional[Dict[str, str]] = None
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
    serialized = serialize(_flatten(tensor_dict), metadata=metadata)
    result = bytes(serialized)
    return result


def save_file(
    tensor_dict: Dict[str, np.ndarray],
    filename: Union[str, os.PathLike],
    metadata: Optional[Dict[str, str]] = None,
    *,
    direct: Optional[bool] = None,
) -> FileLayout:
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
        direct (`bool`, *optional*):
            Whether to write the file directly to the disk, bypassing the page
            cache. `None`, the default, does so for a file larger than 16 MiB;
            `True` for any file; `False` for none. Filesystems that do not
            support direct writes always go through the cache. Linux only.

    Returns:
        `FileLayout`: where the file holds each tensor: `data_offset`, the start of
        the data section, and `tensors`, each tensor's dtype, shape and offsets
        within that section, by name, in file order.

    Example:

    ```python
    from safetensors.numpy import save_file
    import numpy as np

    tensors = {"embedding": np.zeros((512, 1024)), "attention": np.zeros((256, 256))}
    save_file(tensors, "model.safetensors")
    ```
    """
    return serialize_file(
        _flatten(tensor_dict), filename, metadata=metadata, direct=direct
    )


def load(data: bytes) -> Dict[str, np.ndarray]:
    """
    Loads a safetensors file into numpy format from pure bytes.

    Args:
        data (`bytes`):
            The content of a safetensors file

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
    flat = deserialize(data)
    return _view2np(flat)


def load_file(
    filename: Union[str, os.PathLike], *, backend: str = "mmap"
) -> Dict[str, np.ndarray]:
    """
    Loads a safetensors file into numpy format.

    Args:
        filename (`str`, or `os.PathLike`)):
            The name of the file which contains the tensors
        backend (`str`, *optional*, defaults to `"mmap"`):
            Storage backend used to serve tensor bytes. `"mmap"` (default)
            and `"pread"` uses `pread(2)` to read tensor bytes.

    Returns:
        `Dict[str, np.ndarray]`: dictionary that contains name as key, value as `np.ndarray`

    Example:

    ```python
    from safetensors.numpy import load_file

    file_path = "./my_folder/bert.safetensors"
    loaded = load_file(file_path)
    ```
    """
    with safe_open(filename, framework="np", backend=backend) as f:
        return f.get_tensors()


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
        if sys.byteorder == "big":
            arr = arr.byteswap(inplace=False)
        result[k] = arr
    return result
