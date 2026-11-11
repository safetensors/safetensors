import json
import os
import resource
import struct
import tempfile
import unittest

import torch

from safetensors import safe_open
from safetensors.torch import save_file


def minflt() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_minflt


class ZeroCopyTestCase(unittest.TestCase):
    """get_tensor should describe a tensor, not copy it.

    The Storage::Torch branch used to build the tensor with
    `torch.asarray`, which walks the storage and touches every page. That
    turns describing a shard into copying it: the copy costs a fault per
    4 KiB page and holds the GIL for the whole time, so python threads
    cannot parallelize it. It now builds a view with `set_`, which
    aliases the storage without touching a byte.
    """

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, "model.safetensors")
        # Big enough that the pages dominate the fault count: a copy
        # would cost one fault per 4 KiB page, a view almost none.
        self.tensors = {
            "f32": torch.randn(1024, 1024),
            "bf16": torch.randn(2048, 1024).bfloat16(),
            "u8": torch.randint(0, 255, (1 << 20,), dtype=torch.uint8),
            "i64": torch.randint(-1000, 1000, (1 << 14,), dtype=torch.int64),
            "scalar": torch.tensor(3.5),
        }
        save_file(self.tensors, self.path)

    def tearDown(self):
        self.dir.cleanup()

    def test_values_match(self):
        with safe_open(self.path, framework="pt") as f:
            for name, expected in self.tensors.items():
                got = f.get_tensor(name)
                self.assertEqual(got.dtype, expected.dtype, name)
                self.assertEqual(got.shape, expected.shape, name)
                self.assertTrue(torch.equal(got, expected), name)

    def test_is_a_view(self):
        """The tensor must alias the file's mapping, not own a copy."""
        with safe_open(self.path, framework="pt") as f:
            maps = []
            with open("/proc/self/maps") as mf:
                for line in mf:
                    if self.path in line:
                        lo, hi = line.split()[0].split("-")
                        maps.append((int(lo, 16), int(hi, 16)))
            if not maps:
                self.skipTest("no mapping for the file on this platform")
            for name in self.tensors:
                t = f.get_tensor(name)
                self.assertTrue(
                    any(lo <= t.data_ptr() < hi for lo, hi in maps),
                    f"{name} is not a view over the file",
                )

    def test_does_not_touch_the_pages(self):
        """Describing the file must not fault in its pages."""
        size = os.path.getsize(self.path)
        with open(self.path, "rb") as fh:
            (hlen,) = struct.unpack("<Q", fh.read(8))
            data_start = 8 + hlen
        npages = (size - data_start) // 4096

        with safe_open(self.path, framework="pt") as f:
            keys = list(f.keys())
            before = minflt()
            tensors = [f.get_tensor(k) for k in keys]
            faults = minflt() - before

        # The header is a few KiB, so a handful of faults is expected; a
        # copy of the data would cost one per page.
        self.assertLess(
            faults,
            max(16, npages // 10),
            f"{faults} faults for {npages} pages: the data was copied",
        )
        del tensors

    def test_reads_the_right_bytes(self):
        """A view must read the bytes that are in the file."""
        with safe_open(self.path, framework="pt") as f:
            with open(self.path, "rb") as fh:
                (hlen,) = struct.unpack("<Q", fh.read(8))
                header = json.loads(fh.read(hlen))
                data_start = 8 + hlen
                for name, t in f.get_tensors().items():
                    s, e = header[name]["data_offsets"]
                    fh.seek(data_start + s)
                    raw = fh.read(e - s)
                    got = (
                        t.contiguous()
                        .reshape(-1)
                        .view(torch.uint8)
                        .numpy()
                        .tobytes()
                    )
                    self.assertEqual(got, raw, name)


if __name__ == "__main__":
    unittest.main()
