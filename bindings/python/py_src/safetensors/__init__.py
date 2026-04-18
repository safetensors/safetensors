# Re-export this
from ._safetensors_rust import (  # noqa: F401
    SafetensorError,
    TensorSpec,
    __version__,
    deserialize,
    safe_open,
    _safe_open_handle,
    serialize,
    serialize_file,
)
