import os
import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np
import torch

from safetensors import SafetensorError, TensorSpec, safe_open, serialize
from safetensors.numpy import load, load_file, save, save_file
from safetensors.torch import _find_shared_tensors
from safetensors.torch import load_file as load_file_pt
from safetensors.torch import save_file as save_file_pt
from safetensors.torch import storage_ptr, storage_size


class TestCase(unittest.TestCase):
    def test_serialization(self):
        data = np.zeros((2, 2), dtype=np.int32)
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

    def test_deserialization(self):
        serialized = b"""<\x00\x00\x00\x00\x00\x00\x00{"test":{"dtype":"I32","shape":[2,2],"data_offsets":[0,16]}}\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"""

        out = load(serialized)
        self.assertEqual(list(out.keys()), ["test"])
        np.testing.assert_array_equal(out["test"], np.zeros((2, 2), dtype=np.int32))

    def test_deserialization_metadata(self):
        serialized = (
            b'f\x00\x00\x00\x00\x00\x00\x00{"__metadata__":{"framework":"pt"},"test1":{"dtype":"I32","shape":[2,2],"data_offsets":[0,16]}}'
            b"       \x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
        )

        with tempfile.NamedTemporaryFile() as f:
            f.write(serialized)
            f.seek(0)

            with safe_open(f.name, framework="np") as g:
                self.assertEqual(g.metadata(), {"framework": "pt"})

    def test_serialization_order_invariant(self):
        data = np.zeros((2, 2), dtype=np.int32)
        out1 = save({"test1": data, "test2": data})
        out2 = save({"test2": data, "test1": data})
        self.assertEqual(out1, out2)

    def test_serialization_forces_alignment(self):
        data = np.zeros((2, 2), dtype=np.int32)
        data2 = np.zeros((2, 2), dtype=np.float16)
        out1 = save({"test1": data, "test2": data2})
        out2 = save({"test2": data2, "test1": data})
        self.assertEqual(out1, out2)
        self.assertEqual(
            out1,
            b'\x80\x00\x00\x00\x00\x00\x00\x00{"test1":{"dtype":"I32","shape":[2,2],"data_offsets":[0,16]},"test2":{"dtype":"F16","shape":[2,2],"data_offsets":[16,24]}}'
            b"      \x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",
        )
        self.assertEqual(out1[8:].index(b"\x00") + 8, 136)
        self.assertEqual((out1[8:].index(b"\x00") + 8) % 8, 0)

    def test_serialization_metadata(self):
        data = np.zeros((2, 2), dtype=np.int32)
        out1 = save({"test1": data}, metadata={"framework": "pt"})
        self.assertEqual(
            out1,
            b'`\x00\x00\x00\x00\x00\x00\x00{"__metadata__":{"framework":"pt"},"test1":{"dtype":"I32","shape":[2,2],"data_offsets":[0,16]}}'
            b" \x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",
        )
        self.assertEqual(out1[8:].index(b"\x00") + 8, 104)
        self.assertEqual((out1[8:].index(b"\x00") + 8) % 8, 0)

    def test_serialization_no_big_endian(self):
        # Big endian tensor
        data = np.zeros((2, 2), dtype=">i4")
        out1 = save({"test1": data}, metadata={"framework": "pt"})
        self.assertEqual(
            out1,
            b'`\x00\x00\x00\x00\x00\x00\x00{"__metadata__":{"framework":"pt"},"test1":{"dtype":"I32","shape":[2,2],"data_offsets":[0,16]}}'
            b" \x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",
        )
        self.assertEqual(out1[8:].index(b"\x00") + 8, 104)
        self.assertEqual((out1[8:].index(b"\x00") + 8) % 8, 0)

    def test_accept_path(self):
        tensors = {
            "a": torch.zeros((2, 2)),
            "b": torch.zeros((2, 3), dtype=torch.uint8),
        }
        filename = f"./out_{threading.get_ident()}.safetensors"
        save_file_pt(tensors, Path(filename))
        load_file_pt(Path(filename))
        os.remove(Path(filename))

    def test_pt_sf_save_model_overlapping_storage(self):
        m = torch.randn(10)
        n = torch.empty([], dtype=m.dtype, device=m.device)
        element_size = torch.finfo(m.dtype).bits // 8
        try:
            smaller_storage = m.untyped_storage()[: 4 * element_size]
        except Exception:
            try:
                # Fallback for torch>=1.13
                smaller_storage = m.storage().untyped()[: 4 * element_size]
            except Exception:
                try:
                    # Fallback for torch>=1.11
                    smaller_storage = m.storage()._untyped()[: 4 * element_size]
                except Exception:
                    # Fallback for torch==1.10
                    smaller_storage = m.storage()[:4]

        n.set_(source=smaller_storage)

        # Check that we can have tensors with storage that have the same `data_ptr` but not the same storage size
        self.assertEqual(storage_ptr(n), storage_ptr(m))
        self.assertNotEqual(storage_size(n), storage_size(m))
        self.assertEqual(storage_size(n), 4 * element_size)
        self.assertEqual(storage_size(m), 10 * element_size)

        shared_tensors = _find_shared_tensors({"m": m, "n": n})
        self.assertEqual(shared_tensors, [{"m"}, {"n"}])


class WindowsTestCase(unittest.TestCase):
    def test_get_correctly_dropped(self):
        tensors = {
            "a": torch.zeros((2, 2)),
            "b": torch.zeros((2, 3), dtype=torch.uint8),
        }
        save_file_pt(tensors, "./out_windows.safetensors")
        with safe_open("./out_windows.safetensors", framework="pt") as f:
            pass

        with self.assertRaises(SafetensorError):
            print(f.keys())

        with open("./out_windows.safetensors", "w") as g:
            g.write("something")


class ErrorsTestCase(unittest.TestCase):
    def test_file_not_found(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            with safe_open("notafile", framework="pt"):
                pass
        self.assertEqual(str(ctx.exception), "No such file or directory: notafile")


class ReadmeTestCase(unittest.TestCase):
    def assertTensorEqual(self, tensors1, tensors2, equality_fn):
        self.assertEqual(tensors1.keys(), tensors2.keys(), "tensor keys don't match")

        for k, v1 in tensors1.items():
            v2 = tensors2[k]

            self.assertTrue(equality_fn(v1, v2), f"{k} tensors are different")

    def test_numpy_example(self):
        tensors = {"a": np.zeros((2, 2)), "b": np.zeros((2, 3), dtype=np.uint8)}

        save_file(tensors, "./out_np.safetensors")
        out = save(tensors)

        # Now loading
        loaded = load_file("./out_np.safetensors")
        self.assertTensorEqual(tensors, loaded, np.allclose)

        loaded = load(out)
        self.assertTensorEqual(tensors, loaded, np.allclose)

    def test_numpy_bool(self):
        tensors = {"a": np.asarray(False)}

        save_file(tensors, "./out_bool.safetensors")
        out = save(tensors)

        # Now loading
        loaded = load_file("./out_bool.safetensors")
        self.assertTensorEqual(tensors, loaded, np.allclose)

        loaded = load(out)
        self.assertTensorEqual(tensors, loaded, np.allclose)

    def test_torch_example(self):
        tensors = {
            "a": torch.randn((2, 2)),
            "b": torch.randint(0, 128, (2, 3), dtype=torch.uint8),
        }
        # Saving modifies the tensors to type numpy, so we must copy for the
        # test to be correct.
        tensors2 = tensors.copy()

        filename = f"./out_pt_{threading.get_ident()}.safetensors"
        save_file_pt(tensors, filename)

        # Now loading
        loaded = load_file_pt(filename)
        self.assertTensorEqual(tensors2, loaded, torch.allclose)

        # XXX: On Windows, if we write to a file that is currently open with mmap, we get "error 1224" (ERROR_USER_MAPPED_FILE)
        # which occurs when trying to write to a file that has an active memory-mapped section.
        # To avoid this, we delete the loaded object before saving again.
        del loaded

        save_file_pt(tensors, filename)

        loaded = load_file_pt(filename)
        self.assertTensorEqual(tensors2, loaded, torch.allclose)

    def test_exception(self):
        # Unknown dtype is rejected by TensorSpec's constructor.
        with self.assertRaises(SafetensorError):
            TensorSpec(dtype="nonsense", shape=[1], data_ptr=0, data_len=0)

        # Raw dicts are no longer accepted by `serialize` (API moved to TensorSpec in 0.8.0).
        with self.assertRaises(TypeError):
            serialize(
                {
                    "test": {
                        "dtype": "float32",
                        "shape": [1],
                        "data_ptr": 0,
                        "data_len": 0,
                    }
                }
            )

    def test_f4_shape_last_dim_overflow(self):
        # The F4 branch doubles the last shape dim (storage -> logical). If the
        # last dim > usize::MAX / 2, the multiplication must error rather than
        # wrap silently (release) or panic (debug).
        huge = 2**63 + 1
        with self.assertRaises(SafetensorError):
            TensorSpec(
                dtype="float4_e2m1fn_x2",
                shape=[huge],
                data_ptr=0,
                data_len=1,
            )

    def test_torch_slice(self):
        A = torch.randn((10, 5))
        tensors = {
            "a": A,
        }
        ident = threading.get_ident()
        save_file_pt(tensors, f"./slice_{ident}.safetensors")

        # Now loading
        with safe_open(
            f"./slice_{ident}.safetensors", framework="pt", device="cpu"
        ) as f:
            slice_ = f.get_slice("a")
            tensor = slice_[:]
            self.assertEqual(list(tensor.shape), [10, 5])
            torch.testing.assert_close(tensor, A)

            tensor = slice_[tuple()]
            self.assertEqual(list(tensor.shape), [10, 5])
            torch.testing.assert_close(tensor, A)

            tensor = slice_[:2]
            self.assertEqual(list(tensor.shape), [2, 5])
            torch.testing.assert_close(tensor, A[:2])

            tensor = slice_[:, :2]
            self.assertEqual(list(tensor.shape), [10, 2])
            torch.testing.assert_close(tensor, A[:, :2])

            tensor = slice_[0, :2]
            self.assertEqual(list(tensor.shape), [2])
            torch.testing.assert_close(tensor, A[0, :2])

            tensor = slice_[2:, 0]
            self.assertEqual(list(tensor.shape), [8])
            torch.testing.assert_close(tensor, A[2:, 0])

            tensor = slice_[2:, 1]
            self.assertEqual(list(tensor.shape), [8])
            torch.testing.assert_close(tensor, A[2:, 1])

            tensor = slice_[2:, -1]
            self.assertEqual(list(tensor.shape), [8])
            torch.testing.assert_close(tensor, A[2:, -1])

            tensor = slice_[list()]
            self.assertEqual(list(tensor.shape), [0, 5])
            torch.testing.assert_close(tensor, A[list()])

    def test_numpy_slice(self):
        A = np.random.rand(10, 5)
        tensors = {
            "a": A,
        }
        filename = f"./slice_{threading.get_ident()}.safetensors"
        save_file(tensors, filename)

        # Now loading
        with safe_open(filename, framework="np", device="cpu") as f:
            slice_ = f.get_slice("a")
            tensor = slice_[:]
            self.assertEqual(list(tensor.shape), [10, 5])
            self.assertTrue(np.allclose(tensor, A))

            tensor = slice_[tuple()]
            self.assertEqual(list(tensor.shape), [10, 5])
            self.assertTrue(np.allclose(tensor, A))

            tensor = slice_[:2]
            self.assertEqual(list(tensor.shape), [2, 5])
            self.assertTrue(np.allclose(tensor, A[:2]))

            tensor = slice_[:, :2]
            self.assertEqual(list(tensor.shape), [10, 2])
            self.assertTrue(np.allclose(tensor, A[:, :2]))

            tensor = slice_[0, :2]
            self.assertEqual(list(tensor.shape), [2])
            self.assertTrue(np.allclose(tensor, A[0, :2]))

            tensor = slice_[2:, 0]
            self.assertEqual(list(tensor.shape), [8])
            self.assertTrue(np.allclose(tensor, A[2:, 0]))

            tensor = slice_[2:, 1]
            self.assertEqual(list(tensor.shape), [8])
            self.assertTrue(np.allclose(tensor, A[2:, 1]))

            tensor = slice_[2:, -1]
            self.assertEqual(list(tensor.shape), [8])
            self.assertTrue(np.allclose(tensor, A[2:, -1]))

            tensor = slice_[2:, -5]
            self.assertEqual(list(tensor.shape), [8])
            self.assertTrue(np.allclose(tensor, A[2:, -5]))

            tensor = slice_[list()]
            self.assertEqual(list(tensor.shape), [0, 5])
            self.assertTrue(np.allclose(tensor, A[list()]))

            with self.assertRaises(SafetensorError) as cm:
                tensor = slice_[2:, -6]
            self.assertEqual(
                str(cm.exception), "Invalid index -6 for dimension 1 of size 5"
            )

            with self.assertRaises(SafetensorError) as cm:
                tensor = slice_[[0, 1]]
            self.assertEqual(str(cm.exception), "Non empty lists are not implemented")

            with self.assertRaises(SafetensorError) as cm:
                tensor = slice_[2:, 20]
            self.assertEqual(
                str(cm.exception),
                "Error during slicing [2:, 20] with shape [10, 5]: index 20 out of bounds for tensor dimension #1 of size 5",
            )

            with self.assertRaises(SafetensorError) as cm:
                tensor = slice_[:20]
            self.assertEqual(
                str(cm.exception),
                "Error during slicing [:20] with shape [10, 5]: index 19 out of bounds for tensor dimension #0 of size 10",
            )

            with self.assertRaises(SafetensorError) as cm:
                tensor = slice_[:, :20]
            self.assertEqual(
                str(cm.exception),
                "Error during slicing [:, :20] with shape [10, 5]: index 19 out of bounds for tensor dimension #1 of size 5",
            )
