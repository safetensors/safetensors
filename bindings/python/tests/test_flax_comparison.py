import platform
import unittest
import sys


if platform.system() != "Windows":
    # This platform is not supported, we don't want to crash on import
    # This test will be skipped anyway.
    import jax.numpy as jnp
    from jax import random
    from flax.serialization import msgpack_restore, msgpack_serialize
    from safetensors import safe_open
    from safetensors.flax import load_file, save_file


# Jax doesn't not exist on Windows
@unittest.skipIf(platform.system() == "Windows", "Flax is not available on Windows")
class LoadTestCase(unittest.TestCase):
    def setUp(self):
        key = random.key(0)
        data = {
            "test": random.normal(key, (1024, 1024), dtype=jnp.float32),
            "test2": random.normal(key, (1024, 1024), dtype=jnp.float16),
            "test3": random.normal(key, (1024, 1024), dtype=jnp.bfloat16),
            "test4": random.normal(key, (1024, 1024), dtype=jnp.complex64),
        }
        self.flax_filename = "./tests/data/flax_load.msgpack"
        self.sf_filename = "./tests/data/flax_load.safetensors"

        serialized = msgpack_serialize(data)
        with open(self.flax_filename, "wb") as f:
            f.write(serialized)

        save_file(data, self.sf_filename)

    def test_zero_sized(self):
        data = {
            "test": jnp.zeros((2, 0), dtype=jnp.float32),
        }
        local = "./tests/data/out_safe_flat_mmap_small2.safetensors"
        save_file(data.copy(), local)
        reloaded = load_file(local)
        # Empty tensor != empty tensor on numpy, so comparing shapes
        # instead
        self.assertEqual(data["test"].shape, reloaded["test"].shape)

    def test_deserialization_safe(self):
        weights = load_file(self.sf_filename)

        with open(self.flax_filename, "rb") as f:
            data = f.read()
        flax_weights = msgpack_restore(data)

        for k, v in weights.items():
            tv = flax_weights[k]
            self.assertTrue(jnp.allclose(v, tv))

    def test_deserialization_safe_open(self):
        weights = {}
        with safe_open(self.sf_filename, framework="flax") as f:
            for k in f.keys():
                weights[k] = f.get_tensor(k)

        with open(self.flax_filename, "rb") as f:
            data = f.read()
        flax_weights = msgpack_restore(data)

        for k, v in weights.items():
            tv = flax_weights[k]
            self.assertTrue(jnp.allclose(v, tv))

    def test_loading_without_ml_dtype(self):
        # This does not work as we cannot unload
        # modules, copy this into its own file to test.
        # https://github.com/safetensors/safetensors/issues/598
        sys.modules.pop("ml_dtypes", None)
        with safe_open(self.sf_filename, framework="flax") as f:
            f.get_tensor("test3")
