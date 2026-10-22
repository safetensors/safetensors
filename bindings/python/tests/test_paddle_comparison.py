import unittest
import numpy as np

from safetensors import safe_open


try:
    import paddle
    from safetensors.paddle import load_file, save_file, save, load

    HAS_PADDLE = True
except ImportError:
    HAS_PADDLE = False


@unittest.skipIf(not HAS_PADDLE, "Paddle is not available")
class SafeTestCase(unittest.TestCase):
    def setUp(self):
        data = {
            "test": paddle.zeros((1024, 1024), dtype=paddle.float32),
            "test2": paddle.zeros((1024, 1024), dtype=paddle.float32),
            "test3": paddle.zeros((1024, 1024), dtype=paddle.float32),
            "test4": paddle.zeros((1024, 1024), dtype=paddle.complex64),
        }
        self.paddle_filename = "./tests/data/paddle_load.pdparams"
        self.sf_filename = "./tests/data/paddle_load.safetensors"

        paddle.save(data, self.paddle_filename)
        save_file(data, self.sf_filename)

    @unittest.expectedFailure
    def test_zero_sized(self):
        # This fails because paddle wants initialized tensor before
        # sending to numpy
        data = {
            "test": paddle.zeros((2, 0), dtype=paddle.float32),
        }
        local = "./tests/data/out_safe_paddle_mmap_small2.safetensors"
        save_file(data, local)
        reloaded = load_file(local)
        self.assertTrue(paddle.equal(data["test"], reloaded["test"]))

    def test_deserialization_safe(self):
        weights = load_file(self.sf_filename)

        paddle_weights = paddle.load(self.paddle_filename)
        for k, v in weights.items():
            tv = paddle_weights[k]
            self.assertTrue(np.allclose(v, tv))


@unittest.skipIf(not HAS_PADDLE, "Paddle is not available")
class WithOpenCase(unittest.TestCase):
    def test_paddle_tensor_cpu(self):
        A = paddle.randn((10, 5))
        tensors = {
            "a": A,
        }
        save_file(tensors, "./tensor_paddle.safetensors")

        # Now loading cpu
        with safe_open(
            "./tensor_paddle.safetensors", framework="paddle", device="cpu"
        ) as f:
            tensor = f.get_tensor("a")
            self.assertEqual(list(tensor.shape), [10, 5])
            assert paddle.allclose(tensor, A).item()
            assert not tensor.place.is_gpu_place()

    def test_paddle_tensor_gpu(self):
        A = paddle.randn((10, 5))
        tensors = {
            "a": A,
        }
        save_file(tensors, "./tensor_paddle.safetensors")
        # Now loading gpu
        with safe_open(
            "./tensor_paddle.safetensors", framework="paddle", device="cuda"
        ) as f:
            tensor = f.get_tensor("a")
            self.assertEqual(list(tensor.shape), [10, 5])
            assert paddle.allclose(tensor, A).item()
            assert tensor.place.is_gpu_place()

    def test_paddle_slice_cpu(self):
        A = paddle.randn((10, 5))
        tensors = {
            "a": A,
        }
        save_file(tensors, "./slice_paddle.safetensors")

        # Now loading
        with safe_open(
            "./slice_paddle.safetensors", framework="paddle", device="cpu"
        ) as f:
            slice_ = f.get_slice("a")
            tensor = slice_[:]
            self.assertEqual(list(tensor.shape), [10, 5])
            assert paddle.allclose(tensor, A).item()
            assert not tensor.place.is_gpu_place()

            tensor = slice_[tuple()]
            self.assertEqual(list(tensor.shape), [10, 5])
            assert paddle.allclose(tensor, A).item()
            assert not tensor.place.is_gpu_place()

            tensor = slice_[:2]
            self.assertEqual(list(tensor.shape), [2, 5])
            assert paddle.allclose(tensor, A[:2]).item()
            assert not tensor.place.is_gpu_place()

            tensor = slice_[:, :2]
            self.assertEqual(list(tensor.shape), [10, 2])
            assert paddle.allclose(tensor, A[:, :2]).item()
            assert not tensor.place.is_gpu_place()

            tensor = slice_[0, :2]
            self.assertEqual(list(tensor.shape), [2])
            assert paddle.allclose(tensor, A[0, :2]).item()
            assert not tensor.place.is_gpu_place()

            tensor = slice_[2:, 0]
            self.assertEqual(list(tensor.shape), [8])
            assert paddle.allclose(tensor, A[2:, 0]).item()
            assert not tensor.place.is_gpu_place()

            tensor = slice_[2:, 1]
            self.assertEqual(list(tensor.shape), [8])
            assert paddle.allclose(tensor, A[2:, 1]).item()
            assert not tensor.place.is_gpu_place()

            tensor = slice_[2:, -1]
            self.assertEqual(list(tensor.shape), [8])
            assert paddle.allclose(tensor, A[2:, -1]).item()
            assert not tensor.place.is_gpu_place()

            tensor = slice_[list()]
            self.assertEqual(list(tensor.shape), [0, 5])
            assert paddle.allclose(tensor, A[list()]).item()
            assert not tensor.place.is_gpu_place()

    def test_paddle_slice_gpu(self):
        A = paddle.randn((10, 5))
        tensors = {
            "a": A,
        }
        save_file(tensors, "./slice_paddle.safetensors")

        # Now loading
        with safe_open(
            "./slice_paddle.safetensors", framework="paddle", device="cuda"
        ) as f:
            slice_ = f.get_slice("a")
            tensor = slice_[:]
            self.assertEqual(list(tensor.shape), [10, 5])
            assert paddle.allclose(tensor, A).item()
            assert tensor.place.is_gpu_place()

            tensor = slice_[tuple()]
            self.assertEqual(list(tensor.shape), [10, 5])
            assert paddle.allclose(tensor, A).item()
            assert tensor.place.is_gpu_place()

            tensor = slice_[:2]
            self.assertEqual(list(tensor.shape), [2, 5])
            assert paddle.allclose(tensor, A[:2]).item()
            assert tensor.place.is_gpu_place()

            tensor = slice_[:, :2]
            self.assertEqual(list(tensor.shape), [10, 2])
            assert paddle.allclose(tensor, A[:, :2]).item()
            assert tensor.place.is_gpu_place()

            tensor = slice_[0, :2]
            self.assertEqual(list(tensor.shape), [2])
            assert paddle.allclose(tensor, A[0, :2]).item()
            assert tensor.place.is_gpu_place()

            tensor = slice_[2:, 0]
            self.assertEqual(list(tensor.shape), [8])
            assert paddle.allclose(tensor, A[2:, 0]).item()
            assert tensor.place.is_gpu_place()

            tensor = slice_[2:, 1]
            self.assertEqual(list(tensor.shape), [8])
            assert paddle.allclose(tensor, A[2:, 1]).item()
            assert tensor.place.is_gpu_place()

            tensor = slice_[2:, -1]
            self.assertEqual(list(tensor.shape), [8])
            assert paddle.allclose(tensor, A[2:, -1]).item()
            assert tensor.place.is_gpu_place()

            tensor = slice_[list()]
            self.assertEqual(list(tensor.shape), [0, 5])
            assert paddle.allclose(tensor, A[list()]).item()
            assert tensor.place.is_gpu_place()


@unittest.skipIf(not HAS_PADDLE, "Paddle is not available")
class SaveLoadCase(unittest.TestCase):
    def test_complex64_in_memory(self):
        values = np.array(
            [[1.25 + 2.5j, -3.25 - 4.5j], [0j, 7 - 0.125j]], dtype=np.complex64
        )
        for array in (values, values.reshape(-1), values[0, 0], values[:0]):
            with self.subTest(shape=array.shape):
                tensor = paddle.to_tensor(array, dtype=paddle.complex64)
                reloaded = load(save({"test": tensor}))["test"]
                self.assertEqual(reloaded.dtype, paddle.complex64)
                self.assertEqual(tuple(reloaded.shape), array.shape)
                np.testing.assert_array_equal(reloaded.numpy(), array)

    def test_in_memory(self):
        data = {
            "test": paddle.zeros((2, 2), dtype=paddle.float32),
        }
        binary = save(data)
        self.assertEqual(
            binary,
            # Spaces are for forcing the alignment.
            b'@\x00\x00\x00\x00\x00\x00\x00{"test":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}}   '
            b" \x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",
        )
        reloaded = load(binary)
        out = paddle.equal(data["test"], reloaded["test"])
        self.assertTrue(paddle.all(out))

    def test_save_load_cpu(self):
        if paddle.__version__ >= "3.2.0":
            self.dtype = paddle.bfloat16
        else:
            self.dtype = paddle.float32
        data = {
            "test": paddle.randn((2, 2), dtype=self.dtype),
        }
        self.sf_filename = "./tests/data/paddle_save_load_cpu.safetensors"
        save_file(data, self.sf_filename)
        reloaded = load(open(self.sf_filename, "rb").read())
        out = paddle.equal(data["test"], reloaded["test"])
        self.assertTrue(paddle.all(out))

    def test_odd_dtype(self):
        if paddle.__version__ >= "3.2.0":
            data = {
                "test1": paddle.randn((2, 2), dtype=paddle.bfloat16),
                "test2": paddle.randn((2, 2), dtype=paddle.float32),
            }
            self.sf_filename = "./tests/data/paddle_save_load_type.safetensors"
            save_file(data, self.sf_filename)
            reloaded = load_file(self.sf_filename)
            self.assertTrue(paddle.all(paddle.equal(data["test1"], reloaded["test1"])))
            self.assertEqual(reloaded["test1"].dtype, paddle.bfloat16)
            self.assertTrue(paddle.all(paddle.equal(data["test2"], reloaded["test2"])))
            self.assertEqual(reloaded["test2"].dtype, paddle.float32)
