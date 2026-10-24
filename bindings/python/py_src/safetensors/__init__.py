# Re-export this
from ._safetensors_rust import (  # noqa: F401
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


def _reject_metadata_tensor_name(tensor_dict) -> None:
    """Refuse a tensor named ``__metadata__``.

    The header uses that key for the freeform metadata map. Saving a tensor under
    the same name writes a duplicate JSON field, and the file then fails to load.
    """
    if tensor_dict is not None and "__metadata__" in tensor_dict:
        raise ValueError(
            "Cannot save a tensor named '__metadata__'. That name is reserved for the header metadata map, "
            "and a file that uses it for both cannot be loaded."
        )
