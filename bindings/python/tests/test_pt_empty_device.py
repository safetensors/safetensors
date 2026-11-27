import contextlib
import json
import struct

import pytest
import torch

from safetensors.torch import load, load_file


@contextlib.contextmanager
def _meta_default(mode):
    if mode == "context":
        with torch.device("meta"):
            yield
    else:
        previous = torch.get_default_device()
        try:
            torch.set_default_device("meta")
            yield
        finally:
            torch.set_default_device(previous)


@pytest.mark.parametrize("mode", ["context", "global"])
@pytest.mark.parametrize("backend", ["bytes", "mmap", "pread"])
@pytest.mark.parametrize("shape", [(0,), (2, 0, 3)])
@pytest.mark.parametrize(
    "code, dtype",
    [
        ("F32", torch.float32),
        ("BF16", torch.bfloat16),
        ("I64", torch.int64),
        ("BOOL", torch.bool),
    ],
)
def test_empty_tensors_load_on_cpu(tmp_path, mode, backend, shape, code, dtype):
    # Build the checkpoint independently of the serializer: an I32 control
    # tensor followed by an empty tensor with the requested dtype and shape.
    header = json.dumps(
        {
            "full": {"dtype": "I32", "shape": [3], "data_offsets": [0, 12]},
            "empty": {"dtype": code, "shape": shape, "data_offsets": [12, 12]},
        }
    ).encode()
    header += b" " * (-len(header) % 8)
    data = struct.pack("<Q", len(header)) + header + struct.pack("<iii", 1, -2, 3)
    filename = tmp_path / "empty.safetensors"
    filename.write_bytes(data)

    with _meta_default(mode):
        if backend == "bytes":
            tensors = load(data)
        else:
            tensors = load_file(filename, device="cpu", backend=backend)

    assert tensors["empty"].device == torch.device("cpu")
    assert tensors["empty"].dtype == dtype
    assert tuple(tensors["empty"].shape) == shape
    torch.testing.assert_close(
        tensors["full"], torch.tensor([1, -2, 3], dtype=torch.int32, device="cpu")
    )
