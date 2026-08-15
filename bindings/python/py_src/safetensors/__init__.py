# Re-export this
from ._safetensors_rust import (  # noqa: F401
    configure_cuda_loading,
    _engine_inflight_mb,
    SafetensorError,
    TensorSpec,
    __version__,
    deserialize,
    safe_open,
    _safe_open_handle,
    serialize,
    serialize_file,
)
