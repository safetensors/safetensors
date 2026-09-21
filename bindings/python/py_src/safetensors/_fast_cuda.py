"""Opt-in CUDA fast path for safe_open (torch only).

Enabled by setting the environment variable ``SAFETENSORS_FAST_CUDA=1``.

The default ``safe_open(device="cuda")`` path in the Rust bindings copies from
a memory-mapped file region directly into GPU memory. Each cudaMemcpy serializes
on kernel page faults for the underlying mmap pages, preventing overlap between
disk I/O and GPU transfer.

This wrapper instead:

1. Reads tensors into host memory via the existing CPU path (leveraging the
   OS page cache and sequential readahead).
2. Applies ``madvise(MADV_SEQUENTIAL)`` to the file so the kernel prefetches
   aggressively during the initial read.
3. Pins the CPU tensor and issues an async ``.to(cuda, non_blocking=True)`` on
   a dedicated CUDA stream, synchronized when the ``safe_open`` context exits.

See https://github.com/huggingface/safetensors/issues/729 for the motivating
benchmarks.

Trade-offs:
- Peak memory increases by the size of the largest tensor (briefly held in
  both pinned CPU RAM and GPU VRAM during transfer).
- Caller must use ``safe_open`` as a context manager so the stream is
  synchronized before tensors are consumed.
- Applies only when framework=="pt" and device is a CUDA device; other cases
  fall through to the Rust implementation unchanged.
"""

from __future__ import annotations

import mmap
import os
from typing import Any, Union


def _is_cuda_device(device: Union[str, int, Any]) -> bool:
    if isinstance(device, int):
        return True
    if isinstance(device, str):
        return device.startswith("cuda")
    # torch.device("cuda") — duck-typed check without importing torch.
    return getattr(device, "type", None) == "cuda"


def fast_cuda_enabled(framework: str, device: Union[str, int, Any]) -> bool:
    """Return True if the fast-CUDA path should be used for this open."""
    if os.environ.get("SAFETENSORS_FAST_CUDA", "0") != "1":
        return False
    if framework != "pt":
        return False
    return _is_cuda_device(device)


def _apply_madvise_sequential(filename: str) -> None:
    """Hint the kernel to prefetch the file sequentially. Best-effort."""
    try:
        fd = os.open(filename, os.O_RDONLY)
        try:
            size = os.fstat(fd).st_size
            if size == 0:
                return
            mm = mmap.mmap(fd, size, access=mmap.ACCESS_READ)
            try:
                if hasattr(mm, "madvise") and hasattr(mmap, "MADV_SEQUENTIAL"):
                    mm.madvise(mmap.MADV_SEQUENTIAL)
            finally:
                mm.close()
        finally:
            os.close(fd)
    except OSError:
        # madvise is a hint; failure is never fatal.
        pass


class FastCudaSafeOpen:
    """Python wrapper that routes CUDA reads through CPU + pin + async transfer.

    Mirrors the surface of ``safetensors._safetensors_rust.safe_open`` so it
    can be substituted transparently inside a ``with`` block.
    """

    def __init__(
        self,
        filename: Union[str, os.PathLike],
        framework: str,
        device: Union[str, int, Any] = "cpu",
    ) -> None:
        # Deferred import so safetensors remains usable without torch installed
        # for callers that never hit this path.
        import torch

        from ._safetensors_rust import safe_open as _rust_safe_open

        self._device = device if not isinstance(device, int) else f"cuda:{device}"
        self._inner = _rust_safe_open(str(filename), framework=framework, device="cpu")
        self._stream = torch.cuda.Stream()
        _apply_madvise_sequential(str(filename))

    def __enter__(self) -> "FastCudaSafeOpen":
        self._inner.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._stream.synchronize()
        return self._inner.__exit__(exc_type, exc_value, traceback)

    def keys(self):
        return self._inner.keys()

    def offset_keys(self):
        return self._inner.offset_keys()

    def metadata(self):
        return self._inner.metadata()

    def get_slice(self, name: str):
        # Slicing is a lazy operation against the mmap; we do not accelerate it.
        return self._inner.get_slice(name)

    def get_tensor(self, name: str):
        import torch

        cpu_tensor = self._inner.get_tensor(name)
        pinned = cpu_tensor.pin_memory()
        del cpu_tensor
        with torch.cuda.stream(self._stream):
            return pinned.to(self._device, non_blocking=True)
