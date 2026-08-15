# Generated content — partially. The structure and docstrings are produced by
# `python stub.py`. The following are hand-edited additions that must be
# re-applied after each regeneration:
#   - module-level imports (`os`, `typing`)
#   - `__version__: str`
#   - type annotations on `TensorSpec` / `serialize` / `serialize_file`
#
# TODO: once we upgrade pyo3 to >= 0.28, replace `stub.py` with a dedicated
# `tools/stub-gen` binary using `pyo3-introspection`,
# mirroring how `huggingface/tokenizers` does it (see PR #1928).
# That generator emits typed stubs directly from Rust
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

def configure_cuda_loading(
    *,
    chunk_mb: Optional[int] = None,
    workers: Optional[int] = None,
    slabs: Optional[int] = None,
    inflight_mb: Optional[int] = None,
):
    """
    Configure the CUDA loading engine (chunk/pinned-slab size in MiB, shared
    engine worker-thread count, pinned slab ring depth, opportunistic
    read-ahead budget in MiB).

    One-shot: call it before the first CUDA `safe_open(..., backend="pread")`
    load in the process; the engine otherwise initializes lazily with the
    defaults below. Calling it after the engine initialized (or twice)
    raises `SafetensorError`. Omitted arguments keep their default, so
    setting one knob does not pin the others.

    Defaults: `chunk_mb=16`, `workers=24`, `slabs=24`. `inflight_mb` bounds
    the tensor bytes allocated ahead of consumption across every file being
    loaded at once; left unset it is derived at engine init as one
    sixteenth of the loading device's free memory, clamped to
    `[512, 8192]` MiB.
    """
    pass

def _engine_inflight_mb() -> int:
    """
    The read-ahead budget the CUDA loading engine is running with, in MiB:
    the configured `inflight_mb`, or the free-memory-derived default.

    Internal instrumentation. Reading it initializes the budget, freezing
    the derived value, so call it only after (or instead of) a load.
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

        backend (`str`, *keyword-only*, defaults to `"mmap"`):
            Storage backend used to serve tensor bytes. `"mmap"` (the default)
            memory-maps the file; `"pread"` reads tensor bytes with `pread(2)`.
            On Apple-silicon MPS, prefer `"pread"`: it reads straight into the
            shared `MTLBuffer` (1x model memory, no page-cache duplication) and
            loads a full model several times faster than `"mmap"`.
            On CUDA devices with PyTorch, `"pread"` makes `get_tensors()`
            bulk-load the whole file: chunked parallel `pread(2)` through a
            process-global ring of reusable pinned slabs and
            `cudaMemcpyAsync` to the device, served by a shared engine
            worker pool (see `configure_cuda_loading` for the tunables).

        prefetch (`bool`, *keyword-only*, defaults to `False`):
            CUDA + PyTorch + `backend="pread"` only. Starts loading the whole
            tensor-data section at open: reads and H2D copies proceed fully
            in the background (no GIL involvement) into per-tensor
            stream-ordered CUDA allocations (`cudaMallocAsync`), handed to
            torch via DLPack at delivery. `get_tensor` then blocks only
            until that tensor's bytes are resident, `tensor_stream()` yields
            pairs in completion order as they become ready, and
            `get_tensors()` drains the same machinery. Read-ahead across any
            number of prefetching files is bounded by the in-flight budget
            (`configure_cuda_loading(inflight_mb=...)`, otherwise free
            device memory / 16 clamped to `[512, 8192]`MiB):
            allocated-but-unconsumed tensor bytes; at the cap, prefetching
            idles until tensors are consumed. Tensor memory lives outside torch's caching
            allocator (visible in `torch.cuda.mem_get_info`, not in
            `torch.cuda.memory_allocated`) and is freed when the tensor
            drops. Each tensor is delivered once; asking again re-reads it
            from disk.
    """
    def __init__(self, filename, framework, device=..., *, backend: str = "mmap", prefetch: bool = False):
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

    def _forced_pushes(self) -> int:
        """
        Work items this handle has pushed onto the loading engine's forced
        (consumer-demanded) lane; `0` without `prefetch=True`.

        Internal instrumentation: consuming through `tensor_stream` takes
        tensors in the order the engine finishes them, so a plain drain
        never demands work and this stays `0`. Random-access `get_tensor`,
        and a stream starved of budget by other open files, move it.
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
        On CUDA with PyTorch and the `"pread"` backend, the file is loaded
        through a ring of reusable pinned slabs straight into owned
        per-tensor allocations, in completion order (which is also the dict
        insertion order); `prefetch=True` merely starts that work at open.

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

    def tensor_stream(self):
        """
        Returns an iterator of `(name, tensor)` pairs.

        With `prefetch=True`, each pair is yielded as soon as its bytes are
        resident on the device, so per-tensor consumer work overlaps the
        remaining I/O. The yield order is therefore **completion order, not
        offset order**, and varies from run to run: waiting for a particular
        tensor while later ones are already resident is exactly the
        head-of-line stall this iterator exists to avoid. Every tensor is
        yielded exactly once, so `dict(f.tensor_stream())` is unaffected;
        use `get_tensor` when a specific tensor is needed next.

        Without prefetch it is equivalent to `get_tensor` over
        `offset_keys()`, in offset order.

        Returns:
            (`Iterator[Tuple[str, Tensor]]`):
                Every tensor exactly once, in completion order under
                `prefetch=True` and offset order otherwise.

        Example:
        ```python
        from safetensors import safe_open

        with safe_open("model.safetensors", framework="pt", device="cuda:0",
                       backend="pread", prefetch=True) as f:
            for name, tensor in f.tensor_stream():
                ...

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
