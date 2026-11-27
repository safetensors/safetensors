import os
from typing import Dict, Optional, Union

import numpy as np

import mlx.core as mx
from safetensors import (
    FileLayout,
    TensorSpec,
    deserialize,
    numpy,
    safe_open,
    serialize,
    serialize_file,
)


def save(
    tensors: Dict[str, mx.array], metadata: Optional[Dict[str, str]] = None
) -> bytes:
    """
    Saves a dictionary of tensors into raw bytes in safetensors format.

    Args:
        tensors (`Dict[str, mx.array]`):
            The incoming tensors. Tensors need to be contiguous and dense.
        metadata (`Dict[str, str]`, *optional*, defaults to `None`):
            Optional text only metadata you might want to save in your header.
            For instance it can be useful to specify more about the underlying
            tensors. This is purely informative and does not affect tensor loading.

    Returns:
        `bytes`: The raw bytes representing the format

    Example:

    ```python
    from safetensors.mlx import save
    import mlx.core as mx

    tensors = {"embedding": mx.zeros((512, 1024)), "attention": mx.zeros((256, 256))}
    byte_data = save(tensors)
    ```
    """
    return bytes(serialize(_flatten(tensors), metadata=metadata))


def save_file(
    tensors: Dict[str, mx.array],
    filename: Union[str, os.PathLike],
    metadata: Optional[Dict[str, str]] = None,
    *,
    direct: Optional[bool] = None,
) -> FileLayout:
    """
    Saves a dictionary of tensors into raw bytes in safetensors format.

    Args:
        tensors (`Dict[str, mx.array]`):
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
    from safetensors.mlx import save_file
    import mlx.core as mx

    tensors = {"embedding": mx.zeros((512, 1024)), "attention": mx.zeros((256, 256))}
    save_file(tensors, "model.safetensors")
    ```
    """
    return serialize_file(_flatten(tensors), filename, metadata=metadata, direct=direct)


def load(data: bytes) -> Dict[str, mx.array]:
    """
    Loads a safetensors file into MLX format from pure bytes.

    Args:
        data (`bytes`):
            The content of a safetensors file

    Returns:
        `Dict[str, mx.array]`: dictionary that contains name as key, value as `mx.array`

    Example:

    ```python
    from safetensors.mlx import load

    file_path = "./my_folder/bert.safetensors"
    with open(file_path, "rb") as f:
        data = f.read()

    loaded = load(data)
    ```
    """
    return _view2mx(deserialize(data))


def load_file(
    filename: Union[str, os.PathLike], *, backend: str = "mmap"
) -> Dict[str, mx.array]:
    """
    Loads a safetensors file into MLX format.

    Args:
        filename (`str`, or `os.PathLike`)):
            The name of the file which contains the tensors
        backend (`str`, *optional*, defaults to `"mmap"`):
            Storage backend used to serve tensor bytes. `"mmap"` (default)
            and `"pread"` uses `pread(2)` to read tensor bytes. On macOS, a
            file larger than 2 MiB is read in parallel with either.

    Returns:
        `Dict[str, mx.array]`: dictionary that contains name as key, value as `mx.array`

    Example:

    ```python
    from safetensors.flax import load_file

    file_path = "./my_folder/bert.safetensors"
    loaded = load_file(file_path)
    ```
    """
    with safe_open(filename, framework="mlx", backend=backend) as f:
        return f.get_tensors()


def _flatten(tensors: Dict[str, mx.array]) -> Dict[str, TensorSpec]:
    flattened = {}
    for k, v in tensors.items():
        # numpy has no bfloat16: the bytes cross as uint16
        bf16 = isinstance(v, mx.array) and v.dtype == mx.bfloat16
        array = np.asarray(v.view(mx.uint16) if bf16 else v)
        # C order and little-endian, in one copy at most
        array = np.asarray(array, dtype=array.dtype.newbyteorder("<"), order="C")
        flattened[k] = TensorSpec(
            dtype="bfloat16" if bf16 else array.dtype.name,
            shape=array.shape,
            data=array.reshape(-1).view(np.uint8),
        )
    return flattened


def _view2mx(safeview) -> Dict[str, mx.array]:
    result = {}
    for k, v in safeview:
        bf16 = v["dtype"] == "BF16"
        if bf16:
            v = {**v, "dtype": "U16"}
        array = mx.array(numpy._view2np([(k, v)])[k])
        result[k] = array.view(mx.bfloat16) if bf16 else array
    return result
