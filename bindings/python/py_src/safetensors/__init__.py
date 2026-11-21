# Re-export this
import sys

from ._safetensors_rust import (  # noqa: F401
    FileLayout,
    PrefetchLoader,
    PySafeSlice,
    SafetensorError,
    TensorMeta,
    TensorSpec,
    __version__,
    deserialize,
    safe_open,
    _safe_open_handle,
    serialize,
    serialize_file,
)

if sys.platform != "win32":
    from ._safetensors_rust import ReadRequest, Reader  # noqa: F401
