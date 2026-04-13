import sys
import unittest

import torch

from safetensors import safe_open
from safetensors.torch import load, load_file, save, save_file


try:
    import torch_npu  # noqa

    npu_present = True
except Exception:
    npu_present = False


class TorchTestCase(unittest.TestCase):
    def test_serialization(self):
        data = torch.zeros((2, 2), dtype=torch.int32)
        out = save({"test": data})

        self.assertEqual(
            out,
            b'@\x00\x00\x00\x00\x00\x00\x00{"test":{"dtype":"I32","shape":[2,2],"data_offsets":[0,16]}}   '
            b" \x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",
        )

        save_file({"test": data}, "serialization.safetensors")
        out = open("serialization.safetensors", "rb").read()
        self.assertEqual(
            out,
            b'@\x00\x00\x00\x00\x00\x00\x00{"test":{"dtype":"I32","shape":[2,2],"data_offsets":[0,16]}}   '
            b" \x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",
        )

        data[1, 1] = 1
        out = save({"test": data})

        self.assertEqual(
            out,
            b'@\x00\x00\x00\x00\x00\x00\x00{"test":{"dtype":"I32","shape":[2,2],"data_offsets":[0,16]}}   '
            b" \x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00",
        )
        save_file({"test": data}, "serialization.safetensors")
        out = open("serialization.safetensors", "rb").read()
        self.assertEqual(
            out,
            b'@\x00\x00\x00\x00\x00\x00\x00{"test":{"dtype":"I32","shape":[2,2],"data_offsets":[0,16]}}   '
            b" \x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00",
        )

        data = torch.ones((2, 2), dtype=torch.bfloat16)
        data[0, 0] = 2.25
        out = save({"test": data})
        self.assertEqual(
            out,
            b'@\x00\x00\x00\x00\x00\x00\x00{"test":{"dtype":"BF16","shape":[2,2],"data_offsets":[0,8]}}    \x10@\x80?\x80?\x80?',
        )

    def test_odd_dtype(self):
        data = {
            "test": torch.randn((2, 2), dtype=torch.bfloat16),
            "test2": torch.randn((2, 2), dtype=torch.float16),
            "test3": torch.zeros((2, 2), dtype=torch.bool),
            "test4": torch.zeros((2, 2), dtype=torch.complex64),
        }

        # Modify bool to have both values.
        data["test3"][0, 0] = True
        local = "./tests/data/out_safe_pt_mmap_small.safetensors"

        save_file(data, local)
        reloaded = load_file(local)
        self.assertTrue(torch.equal(data["test"], reloaded["test"]))
        self.assertTrue(torch.equal(data["test2"], reloaded["test2"]))
        self.assertTrue(torch.equal(data["test3"], reloaded["test3"]))
        self.assertTrue(torch.equal(data["test4"], reloaded["test4"]))

    def test_complex(self):
        # Test complex separately. Each value consists of two numbers
        # and we want to validate that the representation is the same
        # across platforms.
        data = torch.zeros((2, 2), dtype=torch.complex64)
        out = save({"test": data})

        self.assertEqual(
            out,
            b'@\x00\x00\x00\x00\x00\x00\x00{"test":{"dtype":"C64","shape":[2,2],"data_offsets":[0,32]}}    '
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",
        )

        real = torch.tensor([-1.0])
        imag = torch.tensor([1.0])
        data[1][1] = torch.complex(real, imag)
        out = save({"test": data})

        self.assertEqual(
            out,
            b'@\x00\x00\x00\x00\x00\x00\x00{"test":{"dtype":"C64","shape":[2,2],"data_offsets":[0,32]}}    '
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x80\xbf\x00\x00\x80?",
        )

    def test_odd_dtype_fp8(self):
        if not hasattr(torch, "float8"):
            return  # torch.float8 requires 2.1

        data = {
            "test1": torch.tensor([-0.5], dtype=torch.float8_e4m3fn),
            "test2": torch.tensor([-0.5], dtype=torch.float8_e5m2),
        }
        local = "./tests/data/out_safe_pt_mmap_small.safetensors"

        save_file(data, local)
        reloaded = load_file(local)
        # note: PyTorch doesn't implement torch.equal for float8 so we just compare the single element
        self.assertEqual(reloaded["test1"].dtype, torch.float8_e4m3fn)
        self.assertEqual(reloaded["test1"].item(), -0.5)
        self.assertEqual(reloaded["test2"].dtype, torch.float8_e5m2)
        self.assertEqual(reloaded["test2"].item(), -0.5)

    def test_odd_dtype_fp8_fnuz(self):
        if Version(torch.__version__) < Version("2.1"):
            return  # torch.float8 requires 2.1

        if not hasattr(torch, "float8_e4m3fnuz"):
            return  # fnuz dtypes not available in this torch version

        data = {
            "test1": torch.tensor([0.5], dtype=torch.float8_e4m3fnuz),
            "test2": torch.tensor([0.5], dtype=torch.float8_e5m2fnuz),
        }
        local = "./tests/data/out_safe_pt_mmap_small_fnuz.safetensors"

        save_file(data, local)
        reloaded = load_file(local)
        self.assertEqual(reloaded["test1"].dtype, torch.float8_e4m3fnuz)
        self.assertEqual(reloaded["test1"].item(), 0.5)
        self.assertEqual(reloaded["test2"].dtype, torch.float8_e5m2fnuz)
        self.assertEqual(reloaded["test2"].item(), 0.5)

    def test_odd_dtype_fp4(self):
        if not hasattr(torch, "float4"):
            return  # torch.float4 requires 2.8

        test1 = torch.tensor([0.0], dtype=torch.float8_e8m0fnu)
        test2 = torch.empty(2, 2, device="cpu", dtype=torch.float4_e2m1fn_x2)
        data = {
            "test1": test1,
            "test2": test2,
        }
        local = "./tests/data/out_safe_pt_mmap_fp4.safetensors"

        save_file(data, local)
        reloaded = load_file(local)
        # note: PyTorch doesn't implement torch.equal for float8 so we just compare the single element
        self.assertEqual(reloaded["test1"].dtype, torch.float8_e8m0fnu)
        self.assertEqual(reloaded["test1"], test1)
        self.assertEqual(reloaded["test2"].dtype, torch.float4_e2m1fn_x2)
        # TODO RuntimeError: "eq_cpu" not implemented for 'Float4_e2m1fn_x2'
        # self.assertEqual(reloaded["test2"], test2)

    def test_zero_sized(self):
        data = {
            "test": torch.zeros((2, 0), dtype=torch.float),
        }
        local = "./tests/data/out_safe_pt_mmap_small2.safetensors"
        save_file(data, local)
        reloaded = load_file(local)
        self.assertTrue(torch.equal(data["test"], reloaded["test"]))
        reloaded = load(open(local, "rb").read())
        self.assertTrue(torch.equal(data["test"], reloaded["test"]))

    def test_multiple_zero_sized(self):
        data = {
            "test": torch.zeros((2, 0), dtype=torch.float),
            "test2": torch.zeros((2, 0), dtype=torch.float),
        }
        local = "./tests/data/out_safe_pt_mmap_small3.safetensors"
        save_file(data, local)
        reloaded = load_file(local)
        self.assertTrue(torch.equal(data["test"], reloaded["test"]))
        self.assertTrue(torch.equal(data["test2"], reloaded["test2"]))

    def test_disjoint_tensors_shared_storage(self):
        A = torch.zeros((10, 10))
        data = {
            "test": A[1:],
            "test2": A[:1],
        }
        local = "./tests/data/out_safe_pt_mmap_small4.safetensors"
        save_file(data, local)

    def test_meta_tensor(self):
        A = torch.zeros((10, 10), device=torch.device("meta"))
        data = {
            "test": A,
        }
        local = "./tests/data/out_safe_pt_mmap_small5.safetensors"
        with self.assertRaises(RuntimeError) as ex:
            save_file(data, local)
        self.assertIn("Cannot copy out of meta tensor", str(ex.exception))

    def test_in_memory(self):
        data = {
            "test": torch.zeros((2, 2), dtype=torch.float32),
        }
        binary = save(data)
        self.assertEqual(
            binary,
            # Spaces are for forcing the alignment.
            b'@\x00\x00\x00\x00\x00\x00\x00{"test":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}}   '
            b" \x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",
        )
        reloaded = load(binary)
        self.assertTrue(torch.equal(data["test"], reloaded["test"]))

    @unittest.skipIf(not torch.cuda.is_available(), "Cuda is not available")
    def test_gpu(self):
        data = {
            "test": torch.arange(4).view((2, 2)).to("cuda:0"),
        }
        local = "./tests/data/out_safe_pt_mmap_small.safetensors"
        save_file(data, local)
        reloaded = load_file(local)
        self.assertTrue(torch.equal(torch.arange(4).view((2, 2)), reloaded["test"]))

    @unittest.skipIf(not torch.cuda.is_available(), "Cuda is not available")
    def test_gpu_default_device(self):
        data = {
            "test": torch.arange(4).view((2, 2)).to("cuda:0"),
        }
        local = "./tests/data/out_safe_pt_mmap_small.safetensors"
        save_file(data, local)
        with torch.device("cuda:0"):
            reloaded = load_file(local)
            assert reloaded["test"].device == torch.device("cpu")
            reloaded = load_file(local, device="cuda:0")
            assert reloaded["test"].device == torch.device("cuda:0")
            self.assertTrue(torch.equal(torch.arange(4).view((2, 2)), reloaded["test"]))

    @unittest.skipIf(not npu_present, "Npu is not available")
    def test_npu(self):
        data = {
            "test1": torch.zeros((2, 2), dtype=torch.float32).to("npu:0"),
            "test2": torch.zeros((2, 2), dtype=torch.float16).to("npu:0"),
        }
        local = "./tests/data/out_safe_pt_mmap_small_npu.safetensors"
        save_file(data, local)

        reloaded = load_file(local, device="npu:0")
        for k, v in reloaded.items():
            self.assertTrue(torch.allclose(data[k], reloaded[k]))

    def test_hpu(self):
        # must be run to load torch with Intel Gaudi bindings
        try:
            import habana_frameworks.torch.core as htcore  # noqa: F401
        except ImportError:
            self.skipTest("HPU is not available")

        data = {
            "test1": torch.zeros((2, 2), dtype=torch.float32).to("hpu"),
            "test2": torch.zeros((2, 2), dtype=torch.float16).to("hpu"),
        }
        local = "./tests/data/out_safe_pt_mmap_small_hpu.safetensors"
        save_file(data, local)

        reloaded = load_file(local, device="hpu")
        for k, v in reloaded.items():
            self.assertTrue(torch.allclose(data[k], reloaded[k]))

    @unittest.skipIf(not torch.cuda.is_available(), "Cuda is not available")
    def test_anonymous_accelerator(self):
        data = {
            "test1": torch.zeros((2, 2), dtype=torch.float32).to(device=0),
            "test2": torch.zeros((2, 2), dtype=torch.float16).to(device=0),
        }
        local = "./tests/data/out_safe_pt_mmap_small_anonymous.safetensors"
        save_file(data, local)

        reloaded = load_file(local, device=0)
        for k, v in reloaded.items():
            self.assertTrue(torch.allclose(data[k], reloaded[k]))

    def test_sparse(self):
        data = {"test": torch.sparse_coo_tensor(size=(2, 3))}
        local = "./tests/data/out_safe_pt_sparse.safetensors"
        with self.assertRaises(ValueError) as ctx:
            save_file(data, local)
        self.assertEqual(
            str(ctx.exception),
            "You are trying to save a sparse tensors: `['test']` which this library does not support. You can make it"
            " a dense tensor before saving with `.to_dense()` but be aware this might make a much larger file than"
            " needed.",
        )

    def test_bogus(self):
        data = {"test": {"some": "thing"}}
        local = "./tests/data/out_safe_pt_sparse.safetensors"
        with self.assertRaises(ValueError) as ctx:
            save_file(data, local)
        self.assertEqual(
            str(ctx.exception),
            "Key `test` is invalid, expected torch.Tensor but received <class 'dict'>",
        )

        with self.assertRaises(ValueError) as ctx:
            save_file("notadict", local)
        self.assertEqual(
            str(ctx.exception),
            "Expected a dict of [str, torch.Tensor] but received <class 'str'>",
        )


class LoadTestCase(unittest.TestCase):
    def setUp(self):
        data = {
            "test": torch.zeros((1024, 1024), dtype=torch.float32),
            "test2": torch.zeros((1024, 1024), dtype=torch.float32),
            "test3": torch.zeros((1024, 1024), dtype=torch.float32),
        }
        self.pt_filename = "./tests/data/pt_load.pt"
        self.sf_filename = "./tests/data/pt_load.safetensors"

        with open(self.pt_filename, "wb") as f:
            torch.save(data, f)

        save_file(data, self.sf_filename)

    def test_deserialization_safe(self):
        tweights = torch.load(self.pt_filename)
        weights = load_file(self.sf_filename)

        for k, v in weights.items():
            tv = tweights[k]
            self.assertTrue(torch.allclose(v, tv))
            self.assertEqual(v.device, torch.device("cpu"))

    @unittest.skipIf(not torch.cuda.is_available(), "Cuda is not available")
    def test_deserialization_device(self):
        with torch.device("cuda:0"):
            weights = load_file(self.sf_filename)
            self.assertEqual(weights["test"].device, torch.device("cpu"))

        torch.set_default_device(torch.device("cuda:0"))
        weights = load_file(self.sf_filename)
        self.assertEqual(weights["test"].device, torch.device("cpu"))
        torch.set_default_device(torch.device("cpu"))

        torch.set_default_device(torch.device(0))
        weights = load_file(self.sf_filename)
        self.assertEqual(weights["test"].device, torch.device("cpu"))
        torch.set_default_device(torch.device("cpu"))

    @unittest.skipIf(not torch.cuda.is_available(), "Cuda is not available")
    def test_deserialization_safe_gpu(self):
        # First time to hit disk
        tweights = torch.load(self.pt_filename, map_location="cuda:0")

        load_file(self.sf_filename, device=0)
        weights = load_file(self.sf_filename, device="cuda:0")

        for k, v in weights.items():
            tv = tweights[k]
            self.assertTrue(torch.allclose(v, tv))
            self.assertEqual(v.device, torch.device("cuda:0"))

    @unittest.skipIf(not torch.cuda.is_available(), "Cuda is not available")
    def test_deserialization_safe_gpu_slice(self):
        weights = {}
        with safe_open(self.sf_filename, framework="pt", device="cuda:0") as f:
            for k in f.keys():
                weights[k] = f.get_slice(k)[:1]
        tweights = torch.load(self.pt_filename, map_location="cuda:0")
        tweights = {k: v[:1] for k, v in tweights.items()}
        for k, v in weights.items():
            tv = tweights[k]
            self.assertTrue(torch.allclose(v, tv))
            self.assertEqual(v.device, torch.device("cuda:0"))

    @unittest.skipIf(torch.cuda.device_count() < 2, "Only 1 device available")
    def test_deserialization_safe_device_1(self):
        load_file(self.sf_filename, device=1)
        weights = load_file(self.sf_filename, device="cuda:1")
        tweights = torch.load(self.pt_filename, map_location="cuda:1")
        for k, v in weights.items():
            tv = tweights[k]
            self.assertTrue(torch.allclose(v, tv))
            self.assertEqual(v.device, torch.device("cuda:1"))


class SliceTestCase(unittest.TestCase):
    def setUp(self):
        self.tensor = torch.arange(6, dtype=torch.float32).reshape((1, 2, 3))
        self.data = {"test": self.tensor}
        self.local = "./tests/data/out_safe_pt_mmap_slice.safetensors"
        # Need to copy since that call mutates the tensors to numpy
        save_file(self.data.copy(), self.local)

    def test_cannot_serialize_a_non_contiguous_tensor(self):
        tensor = torch.arange(6, dtype=torch.float32).reshape((1, 2, 3))
        x = tensor[:, :, 1]
        data = {"test": x}
        self.assertFalse(
            x.is_contiguous(),
        )
        with self.assertRaises(ValueError):
            save_file(data, "./tests/data/out.safetensors")

    def test_cannot_serialize_shared(self):
        A = torch.arange(6, dtype=torch.float32).reshape((2, 3))
        B = A[:1]
        data = {"A": A, "B": B}
        with self.assertRaises(RuntimeError):
            save_file(data, "./tests/data/out.safetensors")

        B = A[1:]
        data = {"A": A, "B": B}
        with self.assertRaises(RuntimeError):
            save_file(data, "./tests/data/out.safetensors")

    def test_deserialization_slice(self):
        with safe_open(self.local, framework="pt") as f:
            _slice = f.get_slice("test")
            self.assertEqual(_slice.get_shape(), [1, 2, 3])
            self.assertEqual(_slice.get_dtype(), "F32")
            tensor = _slice[:, :, 1:2]

        self.assertTrue(torch.equal(tensor, torch.Tensor([[[1.0], [4.0]]])))
        self.assertTrue(torch.equal(tensor, self.tensor[:, :, 1:2]))

        buffer = tensor.numpy()
        if sys.byteorder == "big":
            buffer.byteswap(inplace=True)
        buffer = buffer.tobytes()
        self.assertEqual(
            buffer,
            b"\x00\x00\x80?\x00\x00\x80@",
        )

    def test_deserialization_metadata(self):
        with safe_open(self.local, framework="pt") as f:
            metadata = f.metadata()
        self.assertEqual(metadata, None)

        # Save another one *with* metadata
        tensor = torch.arange(6, dtype=torch.float32).reshape((1, 2, 3))
        data = {"test": tensor}
        local = "./tests/data/out_safe_pt_mmap2.safetensors"
        # Need to copy since that call mutates the tensors to numpy
        save_file(data, local, metadata={"Something": "more"})
        with safe_open(local, framework="pt") as f:
            metadata = f.metadata()
        self.assertEqual(metadata, {"Something": "more"})
