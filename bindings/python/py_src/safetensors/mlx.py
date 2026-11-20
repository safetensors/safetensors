import os
from typing import Dict, Optional, Union

import numpy as np

import mlx.core as mx
from safetensors import numpy, safe_open


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
    np_tensors = _mx2np(tensors)
    return numpy.save(np_tensors, metadata=metadata)


def save_file(
    tensors: Dict[str, mx.array],
    filename: Union[str, os.PathLike],
    metadata: Optional[Dict[str, str]] = None,
) -> None:
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

    Returns:
        `None`

    Example:

    ```python
    from safetensors.mlx import save_file
    import mlx.core as mx

    tensors = {"embedding": mx.zeros((512, 1024)), "attention": mx.zeros((256, 256))}
    save_file(tensors, "model.safetensors")
    ```
    """
    np_tensors = _mx2np(tensors)
    return numpy.save_file(np_tensors, filename, metadata=metadata)


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
    flat = numpy.load(data)
    return _np2mx(flat)


def load_file(
    filename: Union[str, os.PathLike],
    *,
    device: Optional[str] = None,
    backend: str = "mmap",
) -> Dict[str, mx.array]:
    """
    Loads a safetensors file into MLX format.

    Args:
        filename (`str`, or `os.PathLike`)):
            The name of the file which contains the tensors
        device (`str`, *optional*):
            The device where the tensors need to be located after load.
            `"mps"` loads them onto the GPU: on Apple silicon with
            `mlx >= 0.32` this aliases a shared `MTLBuffer` per tensor via
            DLPack with no extra copy (older mlx or unsupported dtypes
            gracefully fall back to the copying CPU path).
        backend (`str`, *optional*, defaults to `"mmap"`):
            Storage backend used to serve tensor bytes. `"mmap"` (default)
            and `"pread"` uses `pread(2)` to read tensor bytes.

    Returns:
        `Dict[str, mx.array]`: dictionary that contains name as key, value as `mx.array`

    Example:

    ```python
    from safetensors.mlx import load_file

    file_path = "./my_folder/bert.safetensors"
    loaded = load_file(file_path)
    ```
    """
    with safe_open(filename, framework="mlx", device=device, backend=backend) as f:
        return f.get_tensors()


def _np2mx(numpy_dict: Dict[str, np.ndarray]) -> Dict[str, mx.array]:
    for k, v in numpy_dict.items():
        numpy_dict[k] = mx.array(v)
    return numpy_dict


def _mx2np(mx_dict: Dict[str, mx.array]) -> Dict[str, np.array]:
    new_dict = {}
    for k, v in mx_dict.items():
        new_dict[k] = np.asarray(v)
    return new_dict
