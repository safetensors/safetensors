import os
from typing import Dict, Optional, Union

import numpy as np

import mlx.core as mx
from safetensors import numpy, safe_open


def save(
    tensors: Dict[str, mx.array],
    metadata: Optional[Dict[str, str]] = None,
    max_header_size: Optional[int] = None,
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
        max_header_size (`int`, *optional*):
            Maximum allowed header size in bytes. Defaults to 100MB.

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
    return numpy.save(np_tensors, metadata=metadata, max_header_size=max_header_size)


def save_file(
    tensors: Dict[str, mx.array],
    filename: Union[str, os.PathLike],
    metadata: Optional[Dict[str, str]] = None,
    max_header_size: Optional[int] = None,
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
        max_header_size (`int`, *optional*):
            Maximum allowed header size in bytes. Defaults to 100MB.

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
    return numpy.save_file(
        np_tensors, filename, metadata=metadata, max_header_size=max_header_size
    )


def load(data: bytes, max_header_size: Optional[int] = None) -> Dict[str, mx.array]:
    """
    Loads a safetensors file into MLX format from pure bytes.

    Args:
        data (`bytes`):
            The content of a safetensors file
        max_header_size (`int`, *optional*):
            Maximum allowed header size in bytes. Defaults to 100MB.

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
    flat = numpy.load(data, max_header_size=max_header_size)
    return _np2mx(flat)


def load_file(
    filename: Union[str, os.PathLike], max_header_size: Optional[int] = None
) -> Dict[str, mx.array]:
    """
    Loads a safetensors file into MLX format.

    Args:
        filename (`str`, or `os.PathLike`)):
            The name of the file which contains the tensors
        max_header_size (`int`, *optional*):
            Maximum allowed header size in bytes. Defaults to 100MB.

    Returns:
        `Dict[str, mx.array]`: dictionary that contains name as key, value as `mx.array`

    Example:

    ```python
    from safetensors.flax import load_file

    file_path = "./my_folder/bert.safetensors"
    loaded = load_file(file_path)
    ```
    """
    result = {}
    with safe_open(filename, framework="mlx", max_header_size=max_header_size) as f:
        for k in f.offset_keys():
            result[k] = f.get_tensor(k)
    return result


def _np2mx(numpy_dict: Dict[str, np.ndarray]) -> Dict[str, mx.array]:
    for k, v in numpy_dict.items():
        numpy_dict[k] = mx.array(v)
    return numpy_dict


def _mx2np(mx_dict: Dict[str, mx.array]) -> Dict[str, np.array]:
    new_dict = {}
    for k, v in mx_dict.items():
        new_dict[k] = np.asarray(v)
    return new_dict
