import platform
import unittest
from unittest import mock

import numpy as np


HAS_MLX = False
if platform.system() == "Darwin":
    # This platform is not supported, we don't want to crash on import
    # This test will be skipped anyway.
    try:
        import mlx.core as mx

        HAS_MLX = True
    except ImportError:
        pass
    if HAS_MLX:
        from safetensors import safe_open
        from safetensors.mlx import load, load_file, save, save_file


# MLX only exists on Mac
@unittest.skipIf(platform.system() != "Darwin", "Mlx is not available on non Mac")
@unittest.skipIf(not HAS_MLX, "Mlx is not available.")
class LoadTestCase(unittest.TestCase):
    def setUp(self):
        data = {
            "test": mx.random.uniform(shape=(1024, 1024), dtype=mx.float32),
            "test2": mx.random.uniform(shape=(1024, 1024), dtype=mx.float32),
            "test3": mx.random.uniform(shape=(1024, 1024), dtype=mx.float32),
            "test4": mx.random.uniform(shape=(1024, 1024), dtype=mx.float32).astype(
                mx.complex64
            ),
        }
        self.mlx_filename = "./tests/data/mlx_load.npz"
        self.sf_filename = "./tests/data/mlx_load.safetensors"

        mx.savez(self.mlx_filename, **data)
        save_file(data, self.sf_filename)

    def test_zero_sized(self):
        data = {
            "test": mx.zeros((2, 0), dtype=mx.float32),
        }
        local = "./tests/data/out_safe_flat_mmap_small2.safetensors"
        save_file(data.copy(), local)
        reloaded = load_file(local)
        # Empty tensor != empty tensor on numpy, so comparing shapes
        # instead
        self.assertEqual(data["test"].shape, reloaded["test"].shape)

    def test_deserialization_safe(self):
        weights = load_file(self.sf_filename)

        mlx_weights = mx.load(self.mlx_filename)

        for k, v in weights.items():
            tv = mlx_weights[k]
            self.assertTrue(mx.allclose(v, tv))

    def test_bfloat16(self):
        data = {"a": mx.arange(-8, 8, dtype=mx.float32).astype(mx.bfloat16)}
        local = "./tests/data/mlx_bfloat16.safetensors"
        save_file(data, local)
        for loaded in [load_file(local), load(save(data)), mx.load(local)]:
            self.assertEqual(loaded["a"].dtype, mx.bfloat16)
            self.assertTrue(mx.array_equal(loaded["a"], data["a"]))

    def test_save_strided(self):
        bf16 = mx.arange(10, dtype=mx.float32).astype(mx.bfloat16)
        data = {
            "a": mx.arange(10)[::2],
            "b": mx.arange(12).reshape(3, 4)[:, ::2],
            "c": bf16[::2],
            "d": mx.arange(6).reshape(2, 3).T,
        }
        loaded = load(save(data))
        for k, v in data.items():
            self.assertEqual(loaded[k].dtype, v.dtype)
            self.assertTrue(mx.array_equal(loaded[k], v))

    def test_save_big_endian(self):
        loaded = load(save({"a": np.arange(4, dtype=">i4")}))
        self.assertEqual(loaded["a"].dtype, mx.int32)
        self.assertTrue(mx.array_equal(loaded["a"], mx.arange(4)))

    def test_load_file_reads_every_shape(self):
        # a scalar, an empty tensor, and one of several 16 MiB chunks: the file
        # is large enough to be read in parallel
        data = {
            "scalar": mx.array(3, dtype=mx.int64),
            "empty": mx.zeros((2, 0), dtype=mx.float32),
            "bool": mx.arange(11) % 3 == 0,
            "bf16": mx.arange(-8, 8, dtype=mx.float32).astype(mx.bfloat16),
            "large": mx.arange(10 << 20, dtype=mx.float32),
        }
        local = "./tests/data/mlx_shapes.safetensors"
        save_file(data, local)
        for backend in ["mmap", "pread"]:
            with mock.patch.object(mx, "zeros", wraps=mx.zeros) as zeros:
                loaded = load_file(local, backend=backend)
            # the parallel read allocates the arrays on the CPU stream
            self.assertEqual(zeros.call_count, len(data))
            for call in zeros.call_args_list:
                self.assertEqual(call.kwargs["stream"], mx.cpu)
            self.assertEqual(loaded.keys(), data.keys())
            for k, v in data.items():
                self.assertEqual(loaded[k].dtype, v.dtype)
                self.assertEqual(loaded[k].shape, v.shape)
                self.assertTrue(mx.array_equal(loaded[k], v))

    def test_deserialization_safe_open(self):
        weights = {}
        with safe_open(self.sf_filename, framework="mlx") as f:
            for k in f.keys():
                weights[k] = f.get_tensor(k)

        mlx_weights = mx.load(self.mlx_filename)

        for k, v in weights.items():
            tv = mlx_weights[k]
            self.assertTrue(mx.allclose(v, tv))
