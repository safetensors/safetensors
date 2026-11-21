import os
import tempfile
import unittest

import torch

import safetensors
from safetensors.torch import save_file

MPS = torch.backends.mps.is_available() and tuple(
    int(n) for n in torch.__version__.split(".")[:2]
) >= (2, 12)
DEVICES = (
    ["cpu"]
    + (["cuda:0"] if torch.cuda.is_available() else [])
    + (["mps"] if MPS else [])
)


@unittest.skipUnless(hasattr(safetensors, "Reader"), "no Reader on this platform")
class ReaderTests(unittest.TestCase):
    """`Reader.read(name, index, out)`: parts of tensors read into tensors the caller gives."""

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.bank = torch.arange(64 * 1000, dtype=torch.float32).reshape(64, 1000)
        self.halves = torch.arange(8 * 3 * 5, dtype=torch.float16).reshape(8, 3, 5)
        self.other = torch.arange(1000, dtype=torch.int64)
        self.paths = [
            os.path.join(self.tempdir.name, name)
            for name in ("a.safetensors", "b.safetensors")
        ]
        save_file({"bank": self.bank, "halves": self.halves}, self.paths[0])
        save_file({"other": self.other}, self.paths[1])

    def tearDown(self):
        self.tempdir.cleanup()

    def _empty(self, like, device):
        out = torch.empty(like.shape, dtype=like.dtype, device=device)
        if device == "mps":
            torch.mps.synchronize()
        return out

    def test_rows_land_in_slots(self):
        rows = [3, 60, 17, 0]
        for device in DEVICES:
            with safetensors.Reader(self.paths, threads=4) as reader:
                slots = self._empty(self.bank[:8], device)
                requests = [
                    reader.read("bank", slice(row, row + 1), slots[i : i + 1])
                    for i, row in enumerate(rows)
                ]
                for request in requests:
                    request.wait()
                    self.assertTrue(request.done())
                self.assertTrue(
                    torch.equal(slots[: len(rows)].cpu(), self.bank[rows]), device
                )

    def test_indexes_and_files(self):
        for device in DEVICES:
            with safetensors.Reader(self.paths) as reader:
                whole = self._empty(self.other, device)
                reader.read("other", None, whole).wait()
                self.assertTrue(torch.equal(whole.cpu(), self.other), device)
                part = self._empty(self.halves[4:6], device)
                reader.read("halves", slice(4, 6), part).wait()
                self.assertTrue(torch.equal(part.cpu(), self.halves[4:6]), device)
                row = self._empty(self.bank[7], device)
                reader.read("bank", 7, row).wait()
                self.assertTrue(torch.equal(row.cpu(), self.bank[7]), device)
                nothing = self._empty(self.bank[5:5], device)
                reader.read("bank", slice(5, 5), nothing).wait()

    def test_a_read_spanning_several_chunks(self):
        # larger than one 16 MiB staging slab
        big = torch.arange(5 << 20, dtype=torch.int64)
        path = os.path.join(self.tempdir.name, "big.safetensors")
        save_file({"big": big}, path)
        for device in DEVICES:
            with safetensors.Reader(path) as reader:
                out = self._empty(big, device)
                reader.read("big", None, out).wait()
                self.assertTrue(torch.equal(out.cpu(), big), device)

    def test_a_dropped_request_still_lands(self):
        for device in DEVICES:
            with safetensors.Reader(self.paths) as reader:
                out = self._empty(self.bank[9], device)
                reader.read("bank", 9, out)
            self.assertTrue(torch.equal(out.cpu(), self.bank[9]), device)

    def test_rejects_what_it_cannot_read(self):
        with safetensors.Reader(self.paths[0]) as reader:
            row = torch.empty(1000)
            with self.assertRaisesRegex(Exception, "no tensor nope"):
                reader.read("nope", None, row)
            with self.assertRaisesRegex(Exception, "not contiguous in the file"):
                reader.read("bank", (slice(None), slice(0, 4)), torch.empty((64, 4)))
            with self.assertRaisesRegex(Exception, "shape"):
                reader.read("bank", 0, torch.empty(999))
            with self.assertRaisesRegex(Exception, "F32"):
                reader.read("bank", 0, torch.empty(1000, dtype=torch.float64))
            with self.assertRaisesRegex(Exception, "contiguous"):
                reader.read("bank", slice(0, 2), torch.empty((1000, 2)).T)
        with self.assertRaisesRegex(Exception, "closed"):
            reader.read("bank", 0, row)
        with self.assertRaisesRegex(Exception, "more than one file"):
            safetensors.Reader([self.paths[0], self.paths[0]])
        with self.assertRaises(FileNotFoundError):
            safetensors.Reader(os.path.join(self.tempdir.name, "missing.safetensors"))
