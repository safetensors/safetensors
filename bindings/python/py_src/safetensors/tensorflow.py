import os
from typing import Dict, Optional, Union

import numpy as np
import tensorflow as tf

from safetensors import FileLayout, numpy, safe_open


def save(
    tensors: Dict[str, tf.Tensor], metadata: Optional[Dict[str, str]] = None
) -> bytes:
    """
    Saves a dictionary of tensors into raw bytes in safetensors format.

    Args:
        tensors (`Dict[str, tf.Tensor]`):
            The incoming tensors. Tensors need to be contiguous and dense.
        metadata (`Dict[str, str]`, *optional*, defaults to `None`):
            Optional text only metadata you might want to save in your header.
            For instance it can be useful to specify more about the underlying
            tensors. This is purely informative and does not affect tensor loading.

    Returns:
        `bytes`: The raw bytes representing the format

    Example:

    ```python
    from safetensors.tensorflow import save
    import tensorflow as tf

    tensors = {"embedding": tf.zeros((512, 1024)), "attention": tf.zeros((256, 256))}
    byte_data = save(tensors)
    ```
    """
    np_tensors = _tf2np(tensors)
    return numpy.save(np_tensors, metadata=metadata)


def save_file(
    tensors: Dict[str, tf.Tensor],
    filename: Union[str, os.PathLike],
    metadata: Optional[Dict[str, str]] = None,
    *,
    direct: Optional[bool] = None,
) -> FileLayout:
    """
    Saves a dictionary of tensors into raw bytes in safetensors format.

    Args:
        tensors (`Dict[str, tf.Tensor]`):
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
    from safetensors.tensorflow import save_file
    import tensorflow as tf

    tensors = {"embedding": tf.zeros((512, 1024)), "attention": tf.zeros((256, 256))}
    save_file(tensors, "model.safetensors")
    ```
    """
    np_tensors = _tf2np(tensors)
    return numpy.save_file(np_tensors, filename, metadata=metadata, direct=direct)


def load(data: bytes) -> Dict[str, tf.Tensor]:
    """
    Loads a safetensors file into tensorflow format from pure bytes.

    Args:
        data (`bytes`):
            The content of a safetensors file

    Returns:
        `Dict[str, tf.Tensor]`: dictionary that contains name as key, value as `tf.Tensor` on cpu

    Example:

    ```python
    from safetensors.tensorflow import load

    file_path = "./my_folder/bert.safetensors"
    with open(file_path, "rb") as f:
        data = f.read()

    loaded = load(data)
    ```
    """
    flat = numpy.load(data)
    return _np2tf(flat)


def load_file(
    filename: Union[str, os.PathLike], *, backend: str = "mmap"
) -> Dict[str, tf.Tensor]:
    """
    Loads a safetensors file into tensorflow format.

    Args:
        filename (`str`, or `os.PathLike`)):
            The name of the file which contains the tensors
        backend (`str`, *optional*, defaults to `"mmap"`):
            Storage backend used to serve tensor bytes. `"mmap"` (default)
            and `"pread"` uses `pread(2)` to read tensor bytes.

    Returns:
        `Dict[str, tf.Tensor]`: dictionary that contains name as key, value as `tf.Tensor`

    Example:

    ```python
    from safetensors.tensorflow import load_file

    file_path = "./my_folder/bert.safetensors"
    loaded = load_file(file_path)
    ```
    """
    with safe_open(filename, framework="tf", backend=backend) as f:
        return f.get_tensors()


def _np2tf(numpy_dict: Dict[str, np.ndarray]) -> Dict[str, tf.Tensor]:
    for k, v in numpy_dict.items():
        numpy_dict[k] = tf.convert_to_tensor(v)
    return numpy_dict


def _tf2np(tf_dict: Dict[str, tf.Tensor]) -> Dict[str, np.array]:
    for k, v in tf_dict.items():
        tf_dict[k] = v.numpy()
    return tf_dict
