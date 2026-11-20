import platform
import unittest


HAS_MLX = False
MLX_VERSION = None
if platform.system() == "Darwin":
    # This platform is not supported, we don't want to crash on import
    # This test will be skipped anyway.
    try:
        import mlx.core as mx

        HAS_MLX = True
        MLX_VERSION = getattr(mx, "__version__", None)
    except ImportError:
        pass
    if HAS_MLX:
        from safetensors import safe_open
        from safetensors.mlx import load_file, save_file


def _mlx_at_least(major: int, minor: int) -> bool:
    if MLX_VERSION is None:
        return False
    parts = MLX_VERSION.split(".")
    try:
        return (int(parts[0]), int(parts[1])) >= (major, minor)
    except (ValueError, IndexError):
        return False


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
            # This doesn't work because bfloat16 is not implemented
            # with similar workarounds as jax/tensorflow.
            # https://github.com/ml-explore/mlx/issues/1296
            # "test4": mx.random.uniform(shape=(1024, 1024), dtype=mx.bfloat16),
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

    def test_deserialization_safe_open(self):
        weights = {}
        with safe_open(self.sf_filename, framework="mlx") as f:
            for k in f.keys():
                weights[k] = f.get_tensor(k)

        mlx_weights = mx.load(self.mlx_filename)

        for k, v in weights.items():
            tv = mlx_weights[k]
            self.assertTrue(mx.allclose(v, tv))


@unittest.skipIf(platform.system() != "Darwin", "Mlx is not available on non Mac")
@unittest.skipIf(not HAS_MLX, "Mlx is not available.")
@unittest.skipIf(
    not (HAS_MLX and mx.metal.is_available()),
    "Metal is not available.",
)
@unittest.skipIf(
    not _mlx_at_least(0, 32),
    "Metal DLPack zero-copy requires mlx >= 0.32",
)
class LoadMpsTestCase(unittest.TestCase):
    def setUp(self):
        self.data = {
            "test": mx.random.uniform(shape=(256, 256), dtype=mx.float32),
            "test2": mx.ones((64,), dtype=mx.int32),
            "test3": mx.zeros((16, 8), dtype=mx.float16),
        }
        self.local = "./tests/data/out_safe_mlx_mps.safetensors"
        save_file(self.data, self.local)

    def test_mps(self):
        for backend in ("mmap", "pread"):
            with self.subTest(backend=backend):
                reloaded = load_file(self.local, device="mps", backend=backend)
                for k, v in self.data.items():
                    self.assertTrue(mx.allclose(v, reloaded[k]))

    def test_mps_safe_open(self):
        for backend in ("mmap", "pread"):
            with self.subTest(backend=backend):
                weights = {}
                with safe_open(
                    self.local, framework="mlx", device="mps", backend=backend
                ) as f:
                    for k in f.keys():
                        weights[k] = f.get_tensor(k)
                for k, v in self.data.items():
                    self.assertTrue(mx.allclose(v, weights[k]))

    def test_mps_zero_sized(self):
        data = {"test": mx.zeros((2, 0), dtype=mx.float32)}
        local = "./tests/data/out_safe_mlx_mps_empty.safetensors"
        save_file(data, local)
        reloaded = load_file(local, device="mps")
        # Empty tensor != empty tensor on numpy, so comparing shapes instead
        self.assertEqual(data["test"].shape, reloaded["test"].shape)

    def test_mps_slice(self):
        for backend in ("mmap", "pread"):
            with self.subTest(backend=backend):
                with safe_open(
                    self.local, framework="mlx", device="mps", backend=backend
                ) as f:
                    full = f.get_slice("test")[...]
                    self.assertTrue(mx.allclose(full, self.data["test"]))
                    sub = f.get_slice("test")[0:4, :8]
                    self.assertTrue(mx.allclose(sub, self.data["test"][0:4, :8]))

    def test_mps_f64_fallback(self):
        # MLX has no GPU float64: the loader must gracefully fall back to
        # the copying path instead of failing on the DLPack gate.
        import numpy as np

        from safetensors._safetensors_rust import TensorSpec, serialize

        raw = np.array([1.5, -2.5, 3.25, 0.0], dtype=np.float64)
        spec = TensorSpec(
            dtype="float64", shape=[2, 2], data_ptr=raw.ctypes.data, data_len=raw.nbytes
        )
        local = "./tests/data/out_safe_mlx_f64.safetensors"
        with open(local, "wb") as f:
            f.write(serialize({"f64": spec}))

        reloaded = load_file(local, device="mps")
        # mx has no float64: values are preserved but downcast to float32
        self.assertTrue(mx.allclose(reloaded["f64"], mx.array([[1.5, -2.5], [3.25, 0.0]])))
