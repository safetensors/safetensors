# Generated content — partially. The structure and docstrings are produced by
# `python stub.py`. The following are hand-edited additions that must be
# re-applied after each regeneration:
#   - module-level imports (`os`, `typing`)
#   - `__version__: str`
#   - type annotations on `TensorSpec` / `serialize` / `serialize_file`
#   - the `prefetch` method on `safe_open` and the `PrefetchLoader` class
#   - the `PrefetchPlan` alias and the `TensorMeta` / `PySafeSlice` classes
#
# TODO: once we upgrade pyo3 to >= 0.28, replace `stub.py` with a dedicated
# `tools/stub-gen` binary using `pyo3-introspection`,
# mirroring how `huggingface/tokenizers` does it (see PR #1928).
# That generator emits typed stubs directly from Rust
# signatures — no hand-editing, no drift.
import os
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

PrefetchPlan = Dict[str, Optional[slice]]

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
    immediately rather than failing further inside the serializer.

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
        *,
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

class TensorMeta:
    """A tensor's header entry, as written in the file. No data is read to build it."""

    dtype: str
    """safetensors dtype name, e.g. `"F32"`"""
    shape: List[int]
    data_offsets: Tuple[int, int]
    """`(start, end)` of the tensor's bytes within the data section"""

class PySafeSlice:
    """Lazy view of one tensor returned by `safe_open.get_slice`: reads only the
    bytes an indexing expression asks for at indexation time (hence "lazy view")."""

    def get_shape(self) -> List[int]:
        pass
    def get_dtype(self) -> str:
        pass
    def __getitem__(self, index: Any) -> Any:
        pass

class PrefetchLoader:
    """
    The tensors of one file loading to a CUDA device in the background, returned by
    `safe_open.prefetch`. Each tensor is handed out once, by `take` or by iterating, as a
    zero-copy view of device memory. An allocation's memory is freed once every tensor in it
    has been taken and dropped; allocations with untaken tensors are held until `close`.

    Tensors are ready on any stream when handed out. The free is fenced on the CUDA stream that was
    current at hand-out; a consumer that reads a tensor on another stream must synchronize it before
    dropping the last reference.
    """

    def names(self) -> List[str]:
        """The names this loader hands out, in file order."""
        pass
    def __len__(self) -> int:
        pass
    def take(self, name: str) -> Any:
        """
        Takes `name`'s tensor, waiting if its bytes have not landed yet. Each tensor can be
        taken once, by `take` or by iterating the loader; asking again raises. Rows planned
        as a slice come back as that slice.
        """
        pass
    def __iter__(self) -> Iterator[Tuple[str, Any]]:
        """`(name, tensor)` pairs as their bytes land (unspecified order); tensors already taken are skipped."""
        pass
    def close(self) -> None:
        """Stops the background load and releases every tensor not taken yet; tensors already handed out keep their memory."""
        pass
    def __enter__(self) -> "PrefetchLoader":
        pass
    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
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

        backend (`str`, *keyword-only*, defaults to `"mmap"`):
            Storage backend used to serve tensor bytes. `"mmap"` (the default)
            memory-maps the file; `"pread"` reads tensor bytes with `pread(2)`.
            On Apple-silicon MPS, prefer `"pread"`: it reads straight into the
            shared `MTLBuffer` (1x model memory, no page-cache duplication) and
            loads a full model several times faster than `"mmap"`.
    """
    def __init__(
        self,
        filename,
        framework,
        device=...,
        *,
        backend: str = "mmap",
    ):
        pass

    def prefetch(
        self,
        plan: Optional[PrefetchPlan] = None,
        *,
        device: Optional[Union[str, int]] = None,
        threads: int = 8,
    ) -> "PrefetchLoader":
        """
        Start loading the file's tensors to a CUDA device in the background.

        Returns a `PrefetchLoader` that hands each tensor out once, by `take(name)` or by
        iterating it as `(name, tensor)` pairs in readiness order, as zero-copy views of
        device memory. The handle itself is unchanged: its header queries and `get_slice`
        keep working, and several loaders can be started from one handle (for instance one
        per device). Requires `framework="pt"`; works with either backend. Each loader runs
        `threads` reader threads; all loaders in the process share one pinned staging pool.

        Args:
            plan (`Dict[str, Optional[slice]]`, *optional*):
                Which tensors to load: keys are tensor names, values `None` for
                the whole tensor or a step-1 `slice` along its first dimension.
                Tensors absent from the plan are not loaded by this loader.
                `None` (the default) loads every tensor whole.
            device (`str` or `int`, *optional*):
                The CUDA device to load to; defaults to the handle's `device`.
            threads (`int`, defaults to 8):
                Reader threads for this loader.

        Raises if a planned tensor has a dtype torch cannot represent (F6), if a
        slice has a step other than 1, or if a sliced tensor is 0-d.
        """
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

    def get_slice(self, name: str) -> PySafeSlice:
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

    def get_tensors(self):
        """
        Returns every tensor in the file as a dict keyed by name.

        Equivalent to iterating `offset_keys()` and calling `get_tensor` on
        each, but specific `framework` + `device` combinations take an internal
        fast path. On Apple-silicon MPS with PyTorch and the `"pread"` backend,
        it bulk-allocates shared `MTLBuffer`s, fills them with parallel
        `pread(2)`, and hands them to torch via DLPack with no extra copy.

        Returns:
            (`Dict[str, Tensor]`):
                A dict of all tensors in the file.

        Example:
        ```python
        from safetensors import safe_open

        with safe_open("model.safetensors", framework="pt", device="mps", backend="pread") as f:
            state_dict = f.get_tensors()

        ```
        """
        pass

    def get_tensor_meta(self, name: str) -> TensorMeta:
        """
        Returns a tensor's header entry without reading any data. Works on
        every backend.

        Args:
            name (`str`):
                The name of the tensor
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
