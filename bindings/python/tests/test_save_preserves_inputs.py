import importlib

import numpy as np
import pytest


@pytest.mark.parametrize("framework", ["tensorflow", "flax"])
@pytest.mark.parametrize("as_file", [False, True])
def test_save_preserves_inputs(framework, as_file, tmp_path):
    if framework == "tensorflow":
        tf = pytest.importorskip("tensorflow")
        convert = tf.convert_to_tensor
    else:
        jnp = pytest.importorskip("jax.numpy")
        convert = jnp.asarray
    serialization = importlib.import_module(f"safetensors.{framework}")
    tensors = {
        "weights": convert(np.array([[1.0, -2.0]], dtype=np.float32)),
        "empty": convert(np.zeros((2, 0), dtype=np.float32)),
    }
    original = tensors.copy()
    for _ in range(2):
        if as_file:
            filename = tmp_path / "checkpoint.safetensors"
            serialization.save_file(tensors, filename)
            reloaded = serialization.load_file(filename)
        else:
            reloaded = serialization.load(serialization.save(tensors))

        assert tensors.keys() == original.keys() == reloaded.keys()
        for name, tensor in original.items():
            assert tensors[name] is tensor
            np.testing.assert_array_equal(
                np.asarray(reloaded[name]), np.asarray(tensor)
            )
