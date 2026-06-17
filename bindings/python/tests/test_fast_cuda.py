import os
import tempfile
import unittest

import torch

from safetensors import _rust_safe_open, safe_open
from safetensors._fast_cuda import FastCudaSafeOpen, fast_cuda_enabled
from safetensors.torch import save_file


class FastCudaDispatcherTest(unittest.TestCase):
    """fast_cuda_enabled gates the fast path correctly."""

    def setUp(self):
        self._prev = os.environ.pop("SAFETENSORS_FAST_CUDA", None)

    def tearDown(self):
        if self._prev is None:
            os.environ.pop("SAFETENSORS_FAST_CUDA", None)
        else:
            os.environ["SAFETENSORS_FAST_CUDA"] = self._prev

    def test_disabled_by_default(self):
        self.assertFalse(fast_cuda_enabled("pt", "cuda"))
        self.assertFalse(fast_cuda_enabled("pt", "cuda:0"))
        self.assertFalse(fast_cuda_enabled("pt", 0))

    def test_enabled_only_for_pt_and_cuda(self):
        os.environ["SAFETENSORS_FAST_CUDA"] = "1"
        self.assertTrue(fast_cuda_enabled("pt", "cuda"))
        self.assertTrue(fast_cuda_enabled("pt", "cuda:0"))
        self.assertTrue(fast_cuda_enabled("pt", 0))
        self.assertFalse(fast_cuda_enabled("pt", "cpu"))
        self.assertFalse(fast_cuda_enabled("tf", "cuda"))
        self.assertFalse(fast_cuda_enabled("jax", "cuda"))
        self.assertFalse(fast_cuda_enabled("numpy", "cuda"))

    def test_non_one_values_disable(self):
        os.environ["SAFETENSORS_FAST_CUDA"] = "0"
        self.assertFalse(fast_cuda_enabled("pt", "cuda"))
        os.environ["SAFETENSORS_FAST_CUDA"] = "true"  # only "1" enables
        self.assertFalse(fast_cuda_enabled("pt", "cuda"))

    def test_dispatcher_returns_rust_class_when_disabled(self):
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
            path = tmp.name
        try:
            save_file({"a": torch.zeros(4)}, path)
            with safe_open(path, framework="pt", device="cpu") as f:
                self.assertIsInstance(f, _rust_safe_open)
        finally:
            os.unlink(path)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for this test")
class FastCudaRoundTripTest(unittest.TestCase):
    """With the env var set and CUDA available, tensors round-trip bitwise."""

    def setUp(self):
        self._prev = os.environ.pop("SAFETENSORS_FAST_CUDA", None)

    def tearDown(self):
        if self._prev is None:
            os.environ.pop("SAFETENSORS_FAST_CUDA", None)
        else:
            os.environ["SAFETENSORS_FAST_CUDA"] = self._prev

    def test_fast_path_matches_rust_path(self):
        torch.manual_seed(0)
        tensors = {
            "a": torch.randn(64, 64, dtype=torch.float32),
            "b": torch.randint(0, 100, (32,), dtype=torch.int32),
            "c": torch.randn(8, dtype=torch.float16),
        }
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
            path = tmp.name
        try:
            save_file(tensors, path)

            baseline = {}
            with _rust_safe_open(path, framework="pt", device="cuda") as f:
                for k in f.keys():
                    baseline[k] = f.get_tensor(k).cpu()

            os.environ["SAFETENSORS_FAST_CUDA"] = "1"
            with safe_open(path, framework="pt", device="cuda") as f:
                self.assertIsInstance(f, FastCudaSafeOpen)
                fast = {k: f.get_tensor(k) for k in f.keys()}

            for k, v in fast.items():
                self.assertEqual(v.device.type, "cuda")
                self.assertTrue(torch.equal(v.cpu(), baseline[k]))
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
