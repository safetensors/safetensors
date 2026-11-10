import os
import tempfile
import unittest

import torch

from safetensors.torch import load, load_file, save


class EmptyTensorDefaultDeviceTestCase(unittest.TestCase):
    """Regression tests for #899: empty tensors must be deserialized on
    CPU even when a non-CPU default device is active. Non-empty tensors
    are already forced on CPU by ``torch.frombuffer``.
    """

    def setUp(self):
        self.default_device = torch.get_default_device()
        torch.set_default_device("meta")

    def tearDown(self):
        torch.set_default_device(self.default_device)

    def test_load_empty_tensor(self):
        data = {
            "empty": torch.zeros((0, 10), dtype=torch.float32, device="cpu"),
            "dense": torch.arange(5, dtype=torch.float32, device="cpu"),
        }
        serialized = save(data)

        weights = load(serialized)
        self.assertEqual(weights["empty"].shape, (0, 10))
        self.assertEqual(weights["empty"].dtype, torch.float32)
        self.assertEqual(weights["empty"].device, torch.device("cpu"))
        self.assertEqual(weights["dense"].shape, (5,))
        self.assertEqual(weights["dense"].device, torch.device("cpu"))

    def test_load_file_empty_tensor(self):
        data = {
            "empty": torch.zeros((0, 10), dtype=torch.float32, device="cpu"),
            "dense": torch.arange(5, dtype=torch.float32, device="cpu"),
        }
        serialized = save(data)

        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
            tmp.write(serialized)
            filename = tmp.name
        try:
            weights = load_file(filename)
        finally:
            os.remove(filename)

        self.assertEqual(weights["empty"].shape, (0, 10))
        self.assertEqual(weights["empty"].dtype, torch.float32)
        self.assertEqual(weights["empty"].device, torch.device("cpu"))
        self.assertEqual(weights["dense"].shape, (5,))
        self.assertEqual(weights["dense"].device, torch.device("cpu"))
