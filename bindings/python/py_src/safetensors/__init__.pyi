# Generated content DO NOT EDIT
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
def serialize(tensor_dict, metadata=None):
    """
    Serializes raw data.

    NOTE: the caller is required to ensure any pointer passed via `data_ptr` is valid and will live
    long enough for the duration of the serialization.
    We will remove the need for the caller to hold references themselves when we drop support for
    python versions prior to 3.11 where the `PyBuffer` API is available.
    Creating a `PyBuffer` will enable us to hold a reference to each passed in data array,
    increasing its ref count preventing the gc from collecting it while we serialize.

    Args:
        tensor_dict (`Dict[str, Dict[Any]]`):
            The tensor dict is like:
                {"tensor_name": {"dtype": "F32", "shape": [2, 3], "data_ptr": 1234, "data_len": 24}}
        metadata (`Dict[str, str]`, *optional*):
            The optional purely text annotations

    Returns:
        (`bytes`):
            The serialized content.
    """
    pass

@staticmethod
def serialize_file(tensor_dict, filename, metadata=None):
    """
    Serializes raw data into file.

    NOTE: the caller is required to ensure any pointer passed via `data_ptr` is valid and will live
    long enough for the duration of the serialization.
    We will remove the need for the caller to hold references themselves when we drop support for
    python versions prior to 3.11 where the `PyBuffer` API is available.
    Creating a `PyBuffer` will enable us to hold a reference to each passed in data array,
    increasing its ref count preventing the gc from collecting it while we serialize.

    Args:
        tensor_dict (`Dict[str, Dict[Any]]`):
            The tensor dict is like:
                {"tensor_name": {"dtype": "F32", "shape": [2, 3], "data_ptr": 1234, "data_len": 24}}
        filename (`str`, or `os.PathLike`):
            The name of the file to write into.
        metadata (`Dict[str, str]`, *optional*):
            The optional purely text annotations

    Returns:
        (`NoneType`):
            On success return None
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
