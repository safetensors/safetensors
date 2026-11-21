# Generated content — partially. The structure and docstrings are produced by
# `python stub.py`. The following are hand-edited additions that must be
# re-applied after each regeneration:
#   - module-level imports (`os`, `typing`)
#   - `__version__: str`
#   - type annotations on `TensorSpec` / `serialize` / `serialize_file`
#   - the `prefetch` method on `safe_open` and the `PrefetchLoader` class
#   - the `PrefetchPlan` alias and the `FileLayout` / `TensorMeta` / `PySafeSlice` classes
#
# TODO: once we upgrade pyo3 to >= 0.28, replace `stub.py` with a dedicated
# `tools/stub-gen` binary using `pyo3-introspection`,
# mirroring how `huggingface/tokenizers` does it (see PR #1928).
# That generator emits typed stubs directly from Rust
# signatures — no hand-editing, no drift.
import os
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

PrefetchPlan = Dict[str, Optional[Any]]

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

    Args:
        tensor_dict (`Dict[str, TensorSpec]`):
            Mapping of tensor name to its `TensorSpec`, e.g.:
                {"tensor_name": TensorSpec(dtype="float32", shape=[2, 3], data=array)}
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
    *,
    direct: Optional[bool] = None,
) -> FileLayout:
    """
    Serializes raw data into file.

    On Linux, a tensor on a CUDA device (`TensorSpec.cuda`) is read from the
    device: work that writes it must be complete before the call.

    Args:
        tensor_dict (`Dict[str, TensorSpec]`):
            Mapping of tensor name to its `TensorSpec`, e.g.:
                {"tensor_name": TensorSpec(dtype="float32", shape=[2, 3], data=array)}
        filename (`str`, or `os.PathLike`):
            The name of the file to write into.
        metadata (`Dict[str, str]`, *optional*):
            The optional purely text annotations
        direct (`bool`, *optional*):
            Whether to write the file directly to the disk, bypassing the page
            cache. `None`, the default, does so for a file larger than 16 MiB;
            `True` for any file; `False` for none. Filesystems that do not
            support direct writes always go through the cache. Linux only.

    Returns:
        (`FileLayout`):
            Where the file holds each tensor
    """
    pass

class TensorSpec:
    """
    Describes a single tensor passed to [`serialize`] / [`serialize_file`].

    Constructed from Python as `TensorSpec(dtype, shape, data)`.
    The dtype string is validated at construction; an unknown dtype raises
    immediately rather than failing further inside the serializer.

    `shape` is the logical (header) shape — the number of elements along each
    axis as recorded in the safetensors header. For packed dtypes like
    `float4_e2m1fn_x2` (two F4 values per byte), callers may pass the storage
    shape reported by their framework (e.g. `torch.Size`); the constructor
    transparently doubles the last dimension so `spec.shape` always reflects
    the logical element count.

    `data` is an object exposing the tensor's bytes through the buffer protocol
    (e.g. a numpy array, `bytes` or a `memoryview`): C-contiguous, in
    little-endian order, as many bytes as `dtype` and `shape` take. The spec
    holds the buffer, which keeps the memory alive until the spec is dropped.

    `TensorSpec.cuda` describes a tensor on a CUDA device: only
    `serialize_file`, on Linux, reads such a spec.
    """
    def __init__(
        self,
        *,
        dtype: str,
        shape: Sequence[int],
        data: Any,
    ) -> None:
        pass

    @staticmethod
    def cuda(
        *,
        dtype: str,
        shape: Sequence[int],
        owner: Any,
        data_ptr: int,
        device: int,
    ) -> "TensorSpec":
        """
        Describes a tensor on CUDA device `device`: `data_ptr` is the device
        address of its bytes, as many as `dtype` and `shape` take, and the spec
        holds `owner` to keep them alive.

        SAFETY: `data_ptr` is a raw device address. The caller must ensure it
        is valid for as long as `owner` is alive.
        """
        pass

    @property
    def data_len(self) -> int:
        """
        The length of the tensor's buffer in bytes.
        """
        pass

    @property
    def device(self) -> Optional[int]:
        """
        The index of the CUDA device the tensor is on, `None` for host memory.
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

class FileLayout:
    """Where a file written by `serialize_file` holds its tensors."""

    data_offset: int
    """Where the data section starts in the file"""
    tensors: Dict[str, "TensorMeta"]
    """Each tensor's header entry by name, in file order"""

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
    The tensors of one file loading in the background, to a CUDA or MPS device or to numpy or
    MLX arrays, returned by `safe_open.prefetch`. Each tensor is handed out once, by `take` or by iterating; tensors not
    taken are released at `close`.

    On CUDA, each one is a tensor torch allocated on the stream current at `prefetch`, so torch
    accounts for it and frees it when dropped. Tensors are ready on any stream when handed out.
    Their memory is kept for the stream current at hand-out too; reading one on yet another
    stream needs `record_stream`, as for any torch tensor.

    On MPS, each one wraps a Metal buffer allocated by safetensors, as with `get_tensor`.
    """

    def names(self) -> List[str]:
        """The names this loader hands out, in file order."""
        pass
    def __len__(self) -> int:
        pass
    def take(self, name: str) -> Any:
        """
        Takes `name`'s tensor, waiting if its bytes have not landed yet. Each tensor can be
        taken once, by `take` or by iterating the loader; asking again raises. A tensor
        planned as a slice comes back as that slice.
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

class ReadRequest:
    """
    One read of a `Reader`, or several started together.
    """

    def wait(self) -> None:
        """Returns once the bytes are in `out`. Raises if the read failed."""
        pass
    def done(self) -> bool:
        """Whether `wait` would return at once."""
        pass

class Reader:
    """
    Reads parts of the tensors of one or several safetensors files into tensors the caller
    gives, in the background. Not available on Windows.

    Experimental: the API may change or be removed in a later release, without a
    deprecation period.

    A reader is meant to live as long as its files are read from: it keeps them open and
    keeps its threads. Each `read` returns at once with a `ReadRequest`, and the bytes are in
    `out` once it was waited. `safe_open.prefetch` is the way to load a file, or a planned
    part of it, into new tensors.

    Args:
        filenames (`str`, `os.PathLike` or a list of those):
            The files to read from. A tensor name appears in one of them only.
        threads (`int`, *keyword-only*, defaults to 8):
            How many reads run at once.
        direct (`bool`, *keyword-only*, defaults to `False`):
            Whether reads skip the page cache, on the filesystems that allow it: faster for
            files that are not cached, and they stay uncached.
    """
    def __init__(
        self,
        filenames: Union[str, os.PathLike, Sequence[Union[str, os.PathLike]]],
        *,
        threads: int = 8,
        direct: bool = False,
    ):
        pass
    def read(self, name: str, index: Any, out: Any) -> ReadRequest:
        """
        Starts reading `name[index]` into `out`, and returns at once.

        Args:
            name (`str`):
                The tensor to read from.
            index (*optional*):
                Which part: any index `get_slice` takes whose bytes are contiguous in the
                file, such as a row or a range of rows. `None` reads the whole tensor.
            out (`torch.Tensor`):
                Where the bytes go: a contiguous tensor with the dtype and shape of
                `name[index]`, on the CPU, on a CUDA device or on MPS (torch 2.12 or
                later). Nothing may read or write it, on its device either, until the
                request was waited.

        Returns:
            (`ReadRequest`):
                The read, to wait on.
        """
        pass
    def read_many(self, reads: Sequence[Tuple[str, Any, Any]]) -> ReadRequest:
        """
        Starts several reads as one request, which is done once all of them are: one call
        and one wait, however many reads.

        Args:
            reads (a list of `(name, index, out)`):
                The reads, each as for `read`.

        Returns:
            (`ReadRequest`):
                The reads, to wait on together.
        """
        pass
    def close(self) -> None:
        """Waits for the reads in flight and stops the reader's threads."""
        pass
    def __enter__(self) -> "Reader":
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
            On Apple-silicon MPS, `get_tensor` with `"pread"` reads straight into
            the tensor's Metal buffer.
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
        threads: int = 16,
        direct: Optional[bool] = None,
    ) -> "PrefetchLoader":
        """
        Start loading the file's tensors in the background: with torch to a CUDA device, or
        to MPS on Apple silicon; with numpy or MLX to the CPU.

        Returns a `PrefetchLoader` that hands each tensor out once, by `take(name)` or by
        iterating it as `(name, tensor)` pairs in readiness order, as tensors allocated
        as the load advances, at most about 1 GiB ahead of the reads. The handle itself is
        unchanged: its header queries and `get_slice` keep working, and several loaders can be
        started from one handle (for instance one per device). Requires `framework="pt"`, `"np"`
        or `"mlx"`; works with either backend. Each loader runs `threads` reader threads; all CUDA loaders in
        the process share one pinned staging pool.

        On CUDA, torch allocates the tensors on the stream current at the call, and their
        copies wait for the work queued on that stream when they are allocated. To load
        independently of that work, call `prefetch` under a side stream
        (`with torch.cuda.stream(side):`); torch then keeps a separate cache for it.

        On MPS, which needs torch 2.9 or later, each tensor wraps a Metal buffer allocated
        by safetensors, as with `get_tensor`, and reads go through the page cache.

        With numpy on Linux and macOS, and with MLX on macOS, each tensor is an array the
        framework allocates, which the file is read into. MLX arrays are all allocated at
        the call.

        Args:
            plan (`Dict[str, Optional[Any]]`, *optional*):
                Which tensors to load, and which part of each: keys are tensor
                names, values `None` for the whole tensor or any index `get_slice`
                takes (an int, a slice, `...`, or a tuple of those). In a tuple, a
                dimension's entry can also be a list of ints and slices, kept in
                order along that dimension (e.g. the two halves of a fused
                projection); an int in a list keeps its dimension, as in numpy.
                Tensors absent from the plan are not loaded by this loader.
                `None` (the default) loads every tensor whole.
            device (`str` or `int`, *optional*):
                The CUDA device to load to, or `"mps"`; defaults to the handle's `device`.
            threads (`int`, defaults to 16):
                Reader threads for this loader.
            direct (`bool`, *optional*):
                Whether to read the file directly from the disk, bypassing the page
                cache. Direct reads are faster on fast disks but leave the file
                uncached for later loads. `None` (the default) reads the parts already
                cached from the cache and the rest directly; `True` reads everything
                directly; `False` reads everything through the cache. Filesystems that
                do not support direct reads always go through the cache. On macOS, `None`
                reads through the cache too, and `True` still takes from the cache what it
                holds.

        Raises if a planned tensor has a dtype the installed torch cannot represent
        (F6, F8_E8M0 before torch 2.7, F4 before 2.8), if a slice has a step other
        than 1, if the entries of a dimension overlap or are out of order, if a part
        is too fragmented to load, or if torch can't allocate the first 1 GiB.
        A later allocation failure ends the load: `take` and iteration raise, and the
        tensors not taken are released at once; after any other failure they are
        released at `close`.
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
        fast path. On a CUDA device with PyTorch, and on Apple-silicon MPS with
        torch 2.9 or later, it loads through `prefetch`. On the CPU with PyTorch
        and the `"pread"` backend, or with numpy and either backend, on Linux and
        macOS, it reads a file larger than 2 MiB in parallel. So it does with MLX on macOS.

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
