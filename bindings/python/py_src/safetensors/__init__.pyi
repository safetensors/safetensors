# Generated content — partially. The structure and docstrings are produced by
# `python stub.py`. The following are hand-edited additions that must be
# re-applied after each regeneration:
#   - module-level imports (`os`, `typing`)
#   - `__version__: str`
#   - type annotations on `TensorSpec` / `serialize` / `serialize_file`
#
# TODO: once we upgrade pyo3 to >= 0.28 (assuming `experimental-inspect`,
# replace `stub.py` with a dedicated `tools/stub-gen` binary using
# `pyo3-introspection`, mirroring how `huggingface/tokenizers` does it (see
# #1928). That generator emits typed stubs directly from Rust
# signatures — no hand-editing, no drift.
import os
from typing import Dict, List, Optional, Sequence, Union

__version__: str

@staticmethod
def deserialize(bytes):
    """
    Opens a safetensors lazily and returns tensors as asked

    Args:
        data (`bytes`):
            The byte content of a file

    Returns:
        (`List[str, Dict[str, Dict[str, any]]]`):
            The deserialized content is like:
                [("tensor_name", {"shape": [2, 3], "dtype": "F32", "data": b"\0\0.." }), (...)]
    """
    pass

@staticmethod
def serialize(
    tensor_dict: Dict[str, TensorSpec],
    metadata: Optional[Dict[str, str]] = None,
) -> bytes:
    """
    Serializes raw data.

    NOTE: the caller is required to ensure any pointer passed via `TensorSpec.data_ptr` is valid
    and stays alive for the duration of the serialization.
    We will remove the need for the caller to hold references themselves when we drop support for
    python versions prior to 3.11 where the `PyBuffer` API is available.
    Creating a `PyBuffer` will enable us to hold a reference to each passed in data array,
    increasing its ref count preventing the gc from collecting it while we serialize.

    Args:
        tensor_dict (`Dict[str, TensorSpec]`):
            Mapping of tensor name to its `TensorSpec`, e.g.:
                {"tensor_name": TensorSpec(dtype="float32", shape=[2, 3], data_ptr=1234, data_len=24)}
        metadata (`Dict[str, str]`, *optional*):
            The optional purely text annotations

    Returns:
        (`bytes`):
            The serialized content.
    """
    pass

@staticmethod
def serialize_file(
    tensor_dict: Dict[str, TensorSpec],
    filename: Union[str, "os.PathLike[str]"],
    metadata: Optional[Dict[str, str]] = None,
) -> None:
    """
    Serializes raw data into file.

    NOTE: the caller is required to ensure any pointer passed via `TensorSpec.data_ptr` is valid
    and stays alive for the duration of the serialization.
    We will remove the need for the caller to hold references themselves when we drop support for
    python versions prior to 3.11 where the `PyBuffer` API is available.
    Creating a `PyBuffer` will enable us to hold a reference to each passed in data array,
    increasing its ref count preventing the gc from collecting it while we serialize.

    Args:
        tensor_dict (`Dict[str, TensorSpec]`):
            Mapping of tensor name to its `TensorSpec`, e.g.:
                {"tensor_name": TensorSpec(dtype="float32", shape=[2, 3], data_ptr=1234, data_len=24)}
        filename (`str`, or `os.PathLike`):
            The name of the file to write into.
        metadata (`Dict[str, str]`, *optional*):
            The optional purely text annotations

    Returns:
        (`NoneType`):
            On success return None
    """
    pass

class TensorSpec:
    """
    Describes a single tensor passed to [`serialize`] / [`serialize_file`].

    Constructed from Python as `TensorSpec(dtype, shape, data_ptr, data_len)`.
    The dtype string is validated at construction; an unknown dtype raises
    immediately rather than failing deep inside the serializer.

    `shape` is the logical (header) shape — the number of elements along each
    axis as recorded in the safetensors header. For packed dtypes like
    `float4_e2m1fn_x2` (two F4 values per byte), callers may pass the storage
    shape reported by their framework (e.g. `torch.Size`); the constructor
    transparently doubles the last dimension so `spec.shape` always reflects
    the logical element count.

    SAFETY: `data_ptr` is a raw memory address. The caller must ensure the
    underlying buffer stays alive for the duration of every `serialize` /
    `serialize_file` call that consumes this spec.
    """
    def __init__(
        self,
        dtype: str,
        shape: Sequence[int],
        data_ptr: int,
        data_len: int,
    ) -> None:
        pass

    @property
    def data_len(self) -> int:
        """
        The length of the tensor's buffer in bytes.
        """
        pass

    @property
    def data_ptr(self) -> int:
        """
        The raw memory address of the tensor's contiguous buffer.
        """
        pass

    @property
    def dtype(self) -> str:
        """
        The tensor's dtype as its safetensors format code (e.g. `"F32"`, `"BF16"`,
        `"F8_E5M2FNUZ"`). This is the identifier written into the safetensors
        header, not the Python constructor-style name (`"float32"` etc.).
        """
        pass

    @property
    def shape(self) -> List[int]:
        """
        The tensor's logical shape — the element-count shape recorded in the
        safetensors header. For packed dtypes like `float4_e2m1fn_x2`, this is
        the last-dim-doubled version of whatever was passed to the constructor.
        """
        pass

class safe_open:
    """
    Opens a safetensors lazily and returns tensors as asked

    Args:
        filename (`str`, or `os.PathLike`):
            The filename to open

        framework (`str`):
            The framework you want you tensors in. Supported values:
            `pt`, `tf`, `flax`, `numpy`.

        device (`str`, defaults to `"cpu"`):
            The device on which you want the tensors.
    """
    def __init__(self, filename, framework, device=...):
        pass

    def __enter__(self):
        """
        Start the context manager
        """
        pass

    def __exit__(self, _exc_type, _exc_value, _traceback):
        """
        Exits the context manager
        """
        pass

    def get_slice(self, name):
        """
        Returns a full slice view object

        Args:
            name (`str`):
                The name of the tensor you want

        Returns:
            (`PySafeSlice`):
                A dummy object you can slice into to get a real tensor
        Example:
        ```python
        from safetensors import safe_open

        with safe_open("model.safetensors", framework="pt", device=0) as f:
            tensor_part = f.get_slice("embedding")[:, ::8]

        ```
        """
        pass

    def get_tensor(self, name):
        """
        Returns a full tensor

        Args:
            name (`str`):
                The name of the tensor you want

        Returns:
            (`Tensor`):
                The tensor in the framework you opened the file for.

        Example:
        ```python
        from safetensors import safe_open

        with safe_open("model.safetensors", framework="pt", device=0) as f:
            tensor = f.get_tensor("embedding")

        ```
        """
        pass

    def keys(self):
        """
        Returns the names of the tensors in the file.

        Returns:
            (`List[str]`):
                The name of the tensors contained in that file
        """
        pass

    def metadata(self):
        """
        Return the special non tensor information in the header

        Returns:
            (`Dict[str, str]`):
                The freeform metadata.
        """
        pass

    def offset_keys(self):
        """
        Returns the names of the tensors in the file, ordered by offset.

        Returns:
            (`List[str]`):
                The name of the tensors contained in that file
        """
        pass

class SafetensorError(Exception):
    """
    Custom Python Exception for Safetensor errors.
    """
    def add_note(self, object, /):
        """
        Exception.add_note(note) --
            add a note to the exception
        """
        pass

    def with_traceback(self, object, /):
        """
        Exception.with_traceback(tb) --
            set self.__traceback__ to tb and return self.
        """
        pass
