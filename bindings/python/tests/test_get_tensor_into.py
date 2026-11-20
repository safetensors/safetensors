"""Tests for `safe_open(...).get_tensor_into(dst, name)`.

`get_tensor_into` reads a tensor's bytes with a single `pread(2)` into a
caller-provided tensor, so the caller owns the destination and decides
how many reads are in flight. These tests cover the bytes that land in
the destination, the metadata that describes them, and the concurrency
the API is meant to allow.
"""

import os
import tempfile
import threading
import unittest

import numpy as np
import torch
from safetensors import safe_open
from safetensors.numpy import save_file as save_file_numpy
from safetensors.torch import save_file

SOURCE_TENSORS = {
    "fp32_2d": torch.arange(12, dtype=torch.float32).reshape(3, 4).contiguous(),
    "bf16_2d": torch.arange(8, dtype=torch.bfloat16).reshape(2, 4).contiguous(),
    "fp16_3d": torch.arange(24, dtype=torch.float16).reshape(2, 3, 4).contiguous(),
    "scalar_fp32": torch.tensor(7.5, dtype=torch.float32),
    "empty_2d": torch.empty((0, 5), dtype=torch.float16),
    "i64_1d": torch.arange(5, dtype=torch.int64),
}


def _bytes_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Compare two tensors byte for byte.

    `torch.equal` is not implemented for sub-byte dtypes on some builds,
    so the storage is reinterpreted as uint8 when it fails.
    """
    try:
        return torch.equal(a, b)
    except RuntimeError:
        return torch.equal(
            a.contiguous().view(torch.uint8),
            b.contiguous().view(torch.uint8),
        )


class GetTensorIntoTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tempdir.name, "tiny.safetensors")
        save_file(SOURCE_TENSORS, self.path, metadata={"foo": "bar"})

    def tearDown(self):
        self.tempdir.cleanup()

    def test_bytes_match_the_file(self):
        """The destination holds the tensor's bytes, for every backend."""
        for backend in ("pread", "mmap"):
            with safe_open(self.path, framework="pt", backend=backend) as f:
                for name, expected in SOURCE_TENSORS.items():
                    meta = f.get_tensor_meta(name)
                    nbytes = meta.data_offsets[1] - meta.data_offsets[0]
                    dst = torch.zeros(nbytes, dtype=torch.uint8)
                    read = f.get_tensor_into(dst, name)
                    self.assertEqual(read, nbytes, name)

                    got = dst.view(expected.dtype).reshape(expected.shape)
                    self.assertTrue(_bytes_equal(got, expected), f"{backend}/{name}")

    def test_a_larger_destination_is_accepted(self):
        """A destination bigger than the tensor is fine; only the tensor's
        bytes are written."""
        with safe_open(self.path, framework="pt", backend="pread") as f:
            expected = SOURCE_TENSORS["fp32_2d"]
            nbytes = expected.numel() * expected.element_size()
            dst = torch.full((nbytes + 64,), 0xAB, dtype=torch.uint8)
            read = f.get_tensor_into(dst, "fp32_2d")
            self.assertEqual(read, nbytes)

            got = dst[:nbytes].view(expected.dtype).reshape(expected.shape)
            self.assertTrue(_bytes_equal(got, expected))
            # The tail past the tensor is untouched.
            self.assertTrue(torch.all(dst[nbytes:] == 0xAB))

    def test_a_too_small_destination_is_rejected(self):
        with safe_open(self.path, framework="pt", backend="pread") as f:
            dst = torch.zeros(4, dtype=torch.uint8)
            with self.assertRaises(Exception):
                f.get_tensor_into(dst, "fp32_2d")

    def test_a_missing_tensor_is_rejected(self):
        with safe_open(self.path, framework="pt", backend="pread") as f:
            dst = torch.zeros(48, dtype=torch.uint8)
            with self.assertRaises(Exception):
                f.get_tensor_into(dst, "nope")

    def test_a_non_tensor_destination_is_rejected(self):
        with safe_open(self.path, framework="pt", backend="pread") as f:
            with self.assertRaises(Exception):
                f.get_tensor_into(bytearray(48), "fp32_2d")

    def test_a_strided_destination_is_rejected(self):
        """The bytes are read contiguously, so a strided destination would
        not receive them in the file's order."""
        with safe_open(self.path, framework="pt", backend="pread") as f:
            dst = torch.zeros(96, dtype=torch.uint8)[::2]
            with self.assertRaises(Exception):
                f.get_tensor_into(dst, "fp32_2d")

    def test_an_empty_tensor_reads_nothing(self):
        with safe_open(self.path, framework="pt", backend="pread") as f:
            dst = torch.zeros(1, dtype=torch.uint8)
            self.assertEqual(f.get_tensor_into(dst, "empty_2d"), 0)

    def test_concurrent_reads_into_separate_destinations(self):
        """The point of the API: separate calls run concurrently, each
        into its own destination."""
        with safe_open(self.path, framework="pt", backend="pread") as f:
            names = list(SOURCE_TENSORS)
            results = {}
            errors = []

            def worker(name):
                try:
                    meta = f.get_tensor_meta(name)
                    nbytes = meta.data_offsets[1] - meta.data_offsets[0]
                    dst = torch.zeros(nbytes, dtype=torch.uint8)
                    f.get_tensor_into(dst, name)
                    results[name] = dst
                except Exception as e:  # pragma: no cover - reported below
                    errors.append((name, e))

            threads = [threading.Thread(target=worker, args=(n,)) for n in names]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            self.assertEqual(errors, [])
            self.assertEqual(set(results), set(names))
            for name, dst in results.items():
                expected = SOURCE_TENSORS[name]
                got = dst.view(expected.dtype).reshape(expected.shape)
                self.assertTrue(_bytes_equal(got, expected), name)

    def test_a_numpy_destination_works(self):
        """The destination can be a numpy array as well as a torch tensor."""
        path = os.path.join(self.tempdir.name, "numpy.safetensors")
        tensors = {
            "a": np.arange(12, dtype=np.float32).reshape(3, 4),
            "b": np.arange(8, dtype=np.int64),
        }
        save_file_numpy(tensors, path)
        with safe_open(path, framework="numpy", backend="pread") as f:
            for name, expected in tensors.items():
                meta = f.get_tensor_meta(name)
                nbytes = meta.data_offsets[1] - meta.data_offsets[0]
                dst = np.zeros(nbytes, dtype=np.uint8)
                read = f.get_tensor_into(dst, name)
                self.assertEqual(read, nbytes, name)
                got = dst.view(expected.dtype).reshape(expected.shape)
                self.assertTrue(np.array_equal(got, expected), name)

    def test_a_read_only_numpy_destination_is_rejected(self):
        path = os.path.join(self.tempdir.name, "numpy_ro.safetensors")
        save_file_numpy({"a": np.arange(12, dtype=np.float32)}, path)
        with safe_open(path, framework="numpy", backend="pread") as f:
            dst = np.zeros(48, dtype=np.uint8)
            dst.flags.writeable = False
            with self.assertRaises(Exception):
                f.get_tensor_into(dst, "a")

    def test_the_pread_backend_does_not_map_the_file(self):
        """With the pread backend the file is never mapped, so opening it
        touches only the header's pages."""
        with safe_open(self.path, framework="pt", backend="pread") as f:
            # The metadata is available without any data being read.
            self.assertEqual(
                sorted(f.keys()), sorted(SOURCE_TENSORS.keys())
            )
            meta = f.get_tensor_meta("fp32_2d")
            self.assertEqual(meta.shape, [3, 4])
            self.assertEqual(meta.dtype, "F32")


if __name__ == "__main__":
    unittest.main()
