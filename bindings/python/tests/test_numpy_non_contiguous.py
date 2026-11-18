import numpy as np
from safetensors.numpy import load, load_file, save, save_file


def test_save_preserves_non_contiguous_arrays():
    values = np.arange(24, dtype=np.float32).reshape(3, 8)
    tensors = {
        "fortran": np.asfortranarray(values),
        "strided": values[:, ::2],
        "big_endian": np.asfortranarray(values).astype(">f4"),
    }

    loaded = load(save(tensors))

    for name, tensor in tensors.items():
        np.testing.assert_array_equal(loaded[name], tensor)


def test_save_file_preserves_non_contiguous_arrays(tmp_path):
    values = np.arange(24, dtype=np.float32).reshape(3, 8)
    tensors = {
        "fortran": np.asfortranarray(values),
        "strided": values[:, ::2],
        "big_endian": np.asfortranarray(values).astype(">f4"),
    }

    path = tmp_path / "non-contiguous.safetensors"
    save_file(tensors, path)
    loaded = load_file(path)

    for name, tensor in tensors.items():
        np.testing.assert_array_equal(loaded[name], tensor)
