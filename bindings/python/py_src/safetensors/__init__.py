# Re-export this
from ._safetensors_rust import (  # noqa: F401
    SafetensorError,
    __version__,
    deserialize,
    _safe_open_handle,
    serialize,
    serialize_file,
)
from ._safetensors_rust import safe_open as _rust_safe_open


def safe_open(filename, framework, device="cpu"):
    """Open a safetensors file lazily.

    Dispatches to the Rust implementation by default. When the environment
    variable ``SAFETENSORS_FAST_CUDA=1`` is set and the caller requests a
    torch tensor on a CUDA device, a Python wrapper routes reads through
    CPU + pinned memory + async transfer on a dedicated stream. See
    ``safetensors._fast_cuda`` for details.
    """
    from ._fast_cuda import FastCudaSafeOpen, fast_cuda_enabled

    if fast_cuda_enabled(framework, device):
        return FastCudaSafeOpen(filename, framework=framework, device=device)
    return _rust_safe_open(filename, framework=framework, device=device)
