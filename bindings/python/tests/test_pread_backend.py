"""Tests for `safe_open(..., backend="pread")`.

The `pread` backend serves each tensor via `pread(2)` instead of mmap'ing
the file, dropping each host buffer immediately after the device transfer so
cumulative host residency stays bounded at one tensor.
"""

import gc
import json
import os
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

import numpy as np
import torch
from safetensors import SafetensorError, safe_open
from safetensors.torch import load_file as load_file_pt
from safetensors.torch import load_model, save_file, save_model

SOURCE_TENSORS = {
    "fp32_2d": torch.arange(12, dtype=torch.float32).reshape(3, 4).contiguous(),
    "bf16_2d": torch.arange(8, dtype=torch.bfloat16).reshape(2, 4).contiguous(),
    "fp16_3d": torch.arange(24, dtype=torch.float16).reshape(2, 3, 4).contiguous(),
    "scalar_fp32": torch.tensor(7.5, dtype=torch.float32),
    "empty_2d": torch.empty((0, 5), dtype=torch.float16),
    "i64_1d": torch.arange(5, dtype=torch.int64),
}

if hasattr(torch, "float8_e4m3fn"):
    SOURCE_TENSORS["fp8_e4m3fn"] = torch.zeros(8, dtype=torch.float8_e4m3fn)


MIB, GIB = 1 << 20, 1 << 30


def _pattern(i, n):
    # bytes of tensor i: depend on the tensor and the position
    return ((torch.arange(n, device="cuda:0") * 7 + i * 31) % 251).to(torch.uint8)


def _write_u8(path, sizes, fill=True):
    # U8 tensors t0..tN of the given byte sizes, filled with `_pattern` or sparse
    header, offset = {}, 0
    for i, n in enumerate(sizes):
        header[f"t{i}"] = {
            "dtype": "U8",
            "shape": [n],
            "data_offsets": [offset, offset + n],
        }
        offset += n
    raw = json.dumps(header).encode()
    raw += b" " * (-len(raw) % 8)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(raw)))
        fh.write(raw)
        for i, n in enumerate(sizes):
            if fill and n:
                fh.write(_pattern(i, n).cpu().numpy().tobytes())
        fh.truncate(8 + len(raw) + offset)


def _allocator_threads():
    # prefetch's allocator threads, by the 15 bytes of their name Linux keeps
    count = 0
    for tid in os.listdir("/proc/self/task"):
        with open(f"/proc/self/task/{tid}/comm") as fh:
            count += fh.read().startswith("safetensors-pre")
    return count


def _matmul_seconds(x):
    # how long one `x @ x` takes on the current stream
    x @ x
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(5):
        x @ x
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / 5


def _tensors_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    # torch.equal is not implemented for sub-byte / fp8 dtypes on some builds,
    # so reinterpret the underlying storage as uint8 and compare bytes.
    try:
        return torch.equal(a, b)
    except RuntimeError:
        return torch.equal(
            a.contiguous().view(torch.uint8),
            b.contiguous().view(torch.uint8),
        )


class PreadBackendTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tempdir.name, "tiny.safetensors")
        save_file(SOURCE_TENSORS, self.path, metadata={"foo": "bar"})

    def tearDown(self):
        self.tempdir.cleanup()

    def _assert_state_dict_equal(self, sd):
        self.assertEqual(set(sd.keys()), set(SOURCE_TENSORS.keys()))
        for k, expected in SOURCE_TENSORS.items():
            actual = sd[k]
            self.assertEqual(actual.dtype, expected.dtype, k)
            self.assertEqual(tuple(actual.shape), tuple(expected.shape), k)
            if expected.numel() > 0:
                self.assertTrue(_tensors_equal(actual.cpu(), expected.cpu()), k)

    def test_safe_open_round_trip(self):
        with safe_open(self.path, framework="pt", device="cpu", backend="pread") as f:
            self.assertEqual(f.metadata(), {"foo": "bar"})
            sd = {k: f.get_tensor(k) for k in f.keys()}
        self._assert_state_dict_equal(sd)

    def test_get_tensors_round_trip(self):
        with safe_open(self.path, framework="pt", device="cpu", backend="pread") as f:
            sd = f.get_tensors()
        self._assert_state_dict_equal(sd)

    def test_load_file_round_trip(self):
        sd = load_file_pt(self.path, backend="pread")
        self._assert_state_dict_equal(sd)

    def test_default_backend_unchanged(self):
        # mmap and pread must produce identical bytes.
        with safe_open(self.path, framework="pt", device="cpu") as f:
            sd_mmap = f.get_tensors()
        with safe_open(self.path, framework="pt", device="cpu", backend="pread") as f:
            sd_pread = f.get_tensors()
        self.assertEqual(set(sd_mmap.keys()), set(sd_pread.keys()))
        for k in sd_mmap:
            a, b = sd_mmap[k], sd_pread[k]
            self.assertEqual(a.dtype, b.dtype, k)
            self.assertEqual(tuple(a.shape), tuple(b.shape), k)
            if a.numel() > 0:
                self.assertTrue(_tensors_equal(a.cpu(), b.cpu()), k)

    def test_load_file_reads_cold_and_cached(self):
        # tensors across the 16 MiB chunks the file is read in, at odd offsets,
        # one empty, read with nothing cached then from the page cache
        sizes = [3, 17 * MIB + 5, 0, 1001, 40 * MIB + 3]
        tensors = {
            f"t{i}": ((torch.arange(n) * 7 + i * 31) % 251).to(torch.uint8)
            for i, n in enumerate(sizes)
        }
        path = os.path.join(self.tempdir.name, "chunks.safetensors")
        save_file(tensors, path)
        # dropping the file from the page cache takes posix_fadvise
        for cold in [True, False] if hasattr(os, "posix_fadvise") else [False]:
            if cold:
                fd = os.open(path, os.O_RDONLY)
                os.fsync(fd)
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                os.close(fd)
            got = load_file_pt(path, backend="pread")
            self.assertEqual(list(got), list(tensors))
            for k, v in tensors.items():
                self.assertTrue(torch.equal(got[k], v), f"{k}, cold {cold}")

    def test_get_slice(self):
        with safe_open(self.path, framework="pt", device="cpu", backend="pread") as f:
            slice_obj = f.get_slice("fp32_2d")
            self.assertEqual(list(slice_obj.get_shape()), [3, 4])
            sub = slice_obj[:, 1:3]
        expected = SOURCE_TENSORS["fp32_2d"][:, 1:3]
        self.assertEqual(sub.dtype, expected.dtype)
        self.assertEqual(tuple(sub.shape), tuple(expected.shape))
        self.assertTrue(torch.equal(sub, expected))

    def test_load_model(self):
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = torch.nn.Linear(4, 3)

        src = Tiny()
        path = os.path.join(self.tempdir.name, "tiny_model.safetensors")
        save_model(src, path)

        dst = Tiny()
        self.assertFalse(torch.equal(src.lin.weight, dst.lin.weight))
        load_model(dst, path, backend="pread")
        self.assertTrue(torch.equal(src.lin.weight, dst.lin.weight))
        self.assertTrue(torch.equal(src.lin.bias, dst.lin.bias))

    def test_invalid_backend_string_raises(self):
        with self.assertRaises(Exception):
            safe_open(self.path, framework="pt", device="cpu", backend="not_a_backend")

    def test_truncated_header_is_rejected(self):
        bad_path = os.path.join(self.tempdir.name, "truncated.safetensors")
        with open(self.path, "rb") as src, open(bad_path, "wb") as dst:
            head = src.read(8)
            dst.write(head)
            n = struct.unpack("<Q", head)[0]
            dst.write(src.read(n // 2))
        with (
            self.assertRaises(Exception),
            safe_open(bad_path, framework="pt", device="cpu", backend="pread") as f,
        ):
            _ = f.get_tensors()

    def test_data_offset_overflow_is_rejected(self):
        bad_path = os.path.join(self.tempdir.name, "lying.safetensors")
        header = (
            b'{"liar":{"dtype":"F32","shape":[1024,1024],"data_offsets":[0,4194304]}}'
        )
        with open(bad_path, "wb") as f:
            f.write(struct.pack("<Q", len(header)))
            f.write(header)
            f.write(b"\x00")
        with (
            self.assertRaises(Exception),
            safe_open(bad_path, framework="pt", device="cpu", backend="pread") as f,
        ):
            _ = f.get_tensors()

    def test_prefetch_works_from_any_backend(self):
        # the backend only decides how get_slice reads; a loader opens its own reads
        with safe_open(self.path, framework="pt", device="cpu", backend="mmap") as f:
            with self.assertRaisesRegex(Exception, "devices, not cpu"):
                f.prefetch()

    def test_prefetch_requires_pt(self):
        with safe_open(self.path, framework="numpy", backend="pread") as f:
            with self.assertRaisesRegex(Exception, 'framework="pt"'):
                f.prefetch()

    def test_prefetch_requires_cuda(self):
        with safe_open(self.path, framework="pt", device="cpu", backend="pread") as f:
            with self.assertRaisesRegex(Exception, "cuda"):
                f.prefetch()

    def test_prefetch_plan_rejects_bad_entries(self):
        # plan parsing happens before any device work, so these run without a GPU
        with safe_open(
            self.path, framework="pt", device="cuda:0", backend="pread"
        ) as f:
            with self.assertRaisesRegex(Exception, "does not contain"):
                f.prefetch({"nope": None})
            with self.assertRaisesRegex(Exception, "step 1"):
                f.prefetch({"fp32_2d": slice(0, 3, 2)})
            with self.assertRaisesRegex(Exception, "step 1"):
                f.prefetch({"fp32_2d": (slice(None), slice(0, 4, 2))})
            with self.assertRaisesRegex(
                Exception, "more slicing indexes than dimensions"
            ):
                f.prefetch({"scalar_fp32": slice(0, 1)})
            with self.assertRaisesRegex(Exception, "out of bounds"):
                f.prefetch({"fp32_2d": 3})  # an int selects a row, as with get_slice
            with self.assertRaisesRegex(Exception, "Unsupported slice index"):
                f.prefetch({"fp32_2d": "rows"})

    def test_prefetch_rejects_unsupported_dtype(self):
        # F6 has no torch dtype: refuse at prefetch, before any device memory moves.
        path = os.path.join(self.tempdir.name, "f6.safetensors")
        header = b'{"x":{"dtype":"F6_E2M3","shape":[4],"data_offsets":[0,3]}}'
        with open(path, "wb") as fh:
            fh.write(struct.pack("<Q", len(header)))
            fh.write(header)
            fh.write(bytes([0, 0, 0]))
        with safe_open(path, framework="pt", device="cuda:0", backend="pread") as f:
            with self.assertRaisesRegex(Exception, "F6_E2M3"):
                f.prefetch()

    def test_prefetch_rejects_a_dtype_this_torch_lacks(self):
        # stands in for an older torch (the package allows 2.6, float8_e8m0fnu came with 2.7):
        # refused while planning, by name, before anything is allocated
        path = os.path.join(self.tempdir.name, "e8m0.safetensors")
        header = b'{"x":{"dtype":"F8_E8M0","shape":[4],"data_offsets":[0,4]}}'
        with open(path, "wb") as fh:
            fh.write(struct.pack("<Q", len(header)))
            fh.write(header)
            fh.write(bytes(4))
        dtype = getattr(torch, "float8_e8m0fnu", None)
        if dtype is not None:
            del torch.float8_e8m0fnu
            self.addCleanup(setattr, torch, "float8_e8m0fnu", dtype)
        with safe_open(path, framework="pt", device="cuda:0", backend="pread") as f:
            with self.assertRaisesRegex(SafetensorError, "F8_E8M0"):
                f.prefetch()

    def test_get_tensor_meta_matches_header(self):
        with open(self.path, "rb") as fh:
            header = json.loads(fh.read(struct.unpack("<Q", fh.read(8))[0]))
        with safe_open(self.path, framework="pt", device="cpu", backend="pread") as f:
            for name, entry in header.items():
                if name == "__metadata__":
                    continue
                meta = f.get_tensor_meta(name)
                self.assertEqual(meta.dtype, entry["dtype"])
                self.assertEqual(meta.shape, entry["shape"])
                self.assertEqual(list(meta.data_offsets), entry["data_offsets"])
            with self.assertRaisesRegex(Exception, "does not contain"):
                f.get_tensor_meta("nope")

    def test_prefetch_device_overrides_the_handle(self):
        # the device check runs before any CUDA call: a cuda handle asked to load to cpu must refuse
        with safe_open(
            self.path, framework="pt", device="cuda:0", backend="pread"
        ) as f:
            with self.assertRaisesRegex(Exception, "devices, not cpu"):
                f.prefetch(device="cpu")

    def test_prefetch_loader_is_exported(self):
        from safetensors import PrefetchLoader  # noqa: F401

    def test_numpy_framework(self):
        np_path = os.path.join(self.tempdir.name, "np.safetensors")
        from safetensors.numpy import save_file as save_np

        np_data = {
            "a": np.arange(6, dtype=np.float32).reshape(2, 3),
            "b": np.arange(8, dtype=np.int64).reshape(4, 2),
        }
        save_np(np_data, np_path)

        with safe_open(np_path, framework="numpy", backend="pread") as f:
            for k, expected in np_data.items():
                got = f.get_tensor(k)
                self.assertEqual(got.dtype, expected.dtype, k)
                self.assertEqual(got.shape, expected.shape, k)
                np.testing.assert_array_equal(got, expected)


@unittest.skipUnless(
    torch.backends.mps.is_available()
    and tuple(int(n) for n in torch.__version__.split(".")[:2]) >= (2, 9),
    "needs MPS and torch 2.9",
)
class PrefetchMpsTests(unittest.TestCase):
    """`prefetch` to MPS: tensors on Metal buffers the loader fills."""

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tempdir.name, "tiny.safetensors")
        save_file(SOURCE_TENSORS, self.path)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_tensors_match_source(self):
        for backend in ("mmap", "pread"):
            with safe_open(
                self.path, framework="pt", device="mps", backend=backend
            ) as f:
                with f.prefetch() as loader:
                    loaded = dict(loader)
            self.assertEqual(set(loaded), set(SOURCE_TENSORS))
            for name, expected in SOURCE_TENSORS.items():
                self.assertEqual(loaded[name].device.type, "mps", name)
                self.assertEqual(loaded[name].dtype, expected.dtype, name)
                self.assertEqual(loaded[name].shape, expected.shape, name)
                self.assertTrue(_tensors_equal(loaded[name].cpu(), expected), name)

    def test_plan_rows_match_source(self):
        plan = {
            "fp32_2d": slice(1, 3),
            "fp16_3d": (slice(None), [0, 2]),
            "i64_1d": None,
        }
        with safe_open(self.path, framework="pt") as f:
            with f.prefetch(plan, device="mps") as loader:
                rows = loader.take("fp32_2d")
                with self.assertRaisesRegex(Exception, "already delivered"):
                    loader.take("fp32_2d")
                with self.assertRaisesRegex(Exception, "not in the prefetch plan"):
                    loader.take("bf16_2d")
                rest = dict(loader)
        self.assertTrue(torch.equal(rows.cpu(), SOURCE_TENSORS["fp32_2d"][1:3]))
        self.assertEqual(set(rest), {"fp16_3d", "i64_1d"})
        self.assertTrue(
            torch.equal(rest["fp16_3d"].cpu(), SOURCE_TENSORS["fp16_3d"][:, [0, 2]])
        )
        self.assertTrue(torch.equal(rest["i64_1d"].cpu(), SOURCE_TENSORS["i64_1d"]))

    def test_tensors_larger_than_a_chunk(self):
        # several 16 MiB chunks per tensor, and chunks that span two tensors
        big = os.path.join(self.tempdir.name, "big.safetensors")
        source = {
            f"t{i}": torch.arange((5 + i) << 20, dtype=torch.float32) + i
            for i in range(4)
        }
        save_file(source, big)
        with safe_open(big, framework="pt", device="mps") as f:
            with f.prefetch(threads=4) as loader:
                loaded = dict(loader)
        for name, expected in source.items():
            self.assertTrue(torch.equal(loaded[name].cpu(), expected), name)

    def test_closing_releases_the_tensors_not_taken(self):
        with safe_open(self.path, framework="pt", device="mps") as f:
            loader = f.prefetch()
            taken = loader.take("fp32_2d")
            loader.close()
            with self.assertRaisesRegex(Exception, "closed"):
                loader.take("i64_1d")
        self.assertTrue(torch.equal(taken.cpu(), SOURCE_TENSORS["fp32_2d"]))


@unittest.skipIf(not torch.cuda.is_available(), "Cuda is not available")
class PrefetchCudaTests(unittest.TestCase):
    """`safe_open(...).prefetch(...)`: a `PrefetchLoader` reading the file into
    tensors torch allocates, in the background, each handed out once."""

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tempdir.name, "tiny.safetensors")
        save_file(SOURCE_TENSORS, self.path, metadata={"foo": "bar"})

    def tearDown(self):
        self.tempdir.cleanup()

    def _open(self, path=None, plan=None, **kwargs):
        # the default (mmap, cpu) handle: the loader is told its device
        self.handle = safe_open(path or self.path, framework="pt")
        return self.handle.prefetch(plan, device="cuda:0", **kwargs)

    def test_plan_rows_match_source(self):
        plan = {"fp32_2d": slice(1, 3), "i64_1d": slice(2, 5), "fp16_3d": None}
        with self._open(plan=plan) as loader:
            self.assertEqual(set(loader.names()), set(plan))
            self.assertEqual(len(loader), 3)
            self.assertTrue(
                torch.equal(
                    loader.take("fp32_2d").cpu(), SOURCE_TENSORS["fp32_2d"][1:3]
                )
            )
            self.assertTrue(
                torch.equal(loader.take("i64_1d").cpu(), SOURCE_TENSORS["i64_1d"][2:5])
            )
            with self.assertRaisesRegex(Exception, "not in the prefetch plan"):
                loader.take("bf16_2d")
            streamed = dict(loader)
        self.assertEqual(set(streamed), {"fp16_3d"})  # the two others were taken
        self.assertTrue(
            torch.equal(streamed["fp16_3d"].cpu(), SOURCE_TENSORS["fp16_3d"])
        )

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "needs two CUDA devices")
    def test_non_default_device(self):
        # the calling thread is on device 0 at the call; every copy must still land on cuda:1,
        # whether the device comes from the handle or from the prefetch call
        torch.cuda.set_device(0)
        with (
            safe_open(self.path, framework="pt", device="cuda:1", backend="pread") as f,
            f.prefetch() as loader,
        ):
            first = loader.take("fp32_2d")
            streamed = dict(loader)
        streamed["fp32_2d"] = first
        for t in streamed.values():
            self.assertEqual(t.device, torch.device("cuda:1"))
        self._assert_matches_source(streamed)
        with safe_open(self.path, framework="pt") as f:
            with f.prefetch(device="cuda:1") as loader:
                streamed = dict(loader)
        for t in streamed.values():
            self.assertEqual(t.device, torch.device("cuda:1"))
        self._assert_matches_source(streamed)

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "needs two CUDA devices")
    def test_slabs_move_between_devices(self):
        # pinned slabs are process-wide; their last-copy events rebind per device
        with safe_open(self.path, framework="pt") as f:
            for device in ("cuda:0", "cuda:1", "cuda:0"):
                with f.prefetch(device=device) as loader:
                    streamed = dict(loader)
                for t in streamed.values():
                    self.assertEqual(t.device, torch.device(device))
                self._assert_matches_source(streamed)

    def test_plan_gap_is_not_allocated(self):
        # a, b, c of 64 MiB each; the plan skips b: torch holds 128 MiB, not 192
        mib = 2**20
        path = os.path.join(self.tempdir.name, "gap.safetensors")
        save_file(
            {k: torch.zeros(16 * mib, dtype=torch.float32) for k in ("a", "b", "c")},
            path,
        )
        before = torch.cuda.memory_allocated(0)
        with self._open(path, plan={"a": None, "c": None}) as loader:
            held = dict(loader)
            used = torch.cuda.memory_allocated(0) - before
        self.assertEqual(set(held), {"a", "c"})
        self.assertEqual(used, 128 * mib)

    def test_plan_rejects_out_of_range_rows(self):
        # Python would clamp slice(0, 10) on 3 rows to 3 rows; a plan is explicit
        with safe_open(self.path, framework="pt") as f:
            with self.assertRaisesRegex(Exception, "out of bounds"):
                f.prefetch({"fp32_2d": slice(0, 10)}, device="cuda:0")
            with self.assertRaisesRegex(Exception, "invalid prefetch plan slice"):
                f.prefetch({"fp32_2d": slice(0, 3, 0)}, device="cuda:0")
            with self.assertRaisesRegex(Exception, "selects no tensors"):
                f.prefetch({}, device="cuda:0")

    def test_plan_slices_match_source(self):
        # a plan entry means what indexing `get_slice` means, plus lists per dimension
        plan = {
            "fp32_2d": (slice(None), slice(1, 3)),
            "fp16_3d": (1, ..., slice(1, 3)),  # an int drops its dimension
            "i64_1d": ([slice(0, 2), slice(3, 5)],),
            "bf16_2d": ([0, 1], [0, slice(2, 4)]),  # lists may mix ints and slices
            "scalar_fp32": None,
        }
        with self._open(plan=plan) as loader:
            got = {name: loader.take(name) for name in plan}
        src = SOURCE_TENSORS
        with safe_open(self.path, framework="pt") as f:
            expected = {
                "fp32_2d": f.get_slice("fp32_2d")[:, 1:3],
                "fp16_3d": f.get_slice("fp16_3d")[1, ..., 1:3],
            }
        expected |= {
            "i64_1d": torch.cat([src["i64_1d"][0:2], src["i64_1d"][3:5]]),
            "bf16_2d": src["bf16_2d"][0:2][:, [0, 2, 3]],
            "scalar_fp32": src["scalar_fp32"],
        }
        for name, want in expected.items():
            self.assertTrue(got[name].is_contiguous(), name)
            self.assertEqual(tuple(got[name].shape), tuple(want.shape), name)
            self.assertTrue(torch.equal(got[name].cpu(), want), name)

    def test_plan_int_in_a_list_keeps_its_dimension(self):
        with self._open(plan={"fp16_3d": ([1], slice(None), slice(1, 3))}) as loader:
            got = loader.take("fp16_3d")
        self.assertTrue(torch.equal(got.cpu(), SOURCE_TENSORS["fp16_3d"][1:2, :, 1:3]))

    def test_slices_across_chunks(self):
        # chunks are 16 MiB: in a 64 MiB tensor, a column slice is cut by chunk edges
        # mid-row, and so are the parts of a row list
        path = os.path.join(self.tempdir.name, "big.safetensors")
        w = torch.arange(2048 * 8192, dtype=torch.float32).reshape(2048, 8192)
        save_file({"w": w}, path)
        with safe_open(path, framework="pt") as f:
            with f.prefetch(
                {"w": (slice(None), slice(100, 1124))}, device="cuda:0"
            ) as loader:
                cols = loader.take("w")
            with f.prefetch(
                {"w": ([slice(0, 256), slice(1792, 2048)],)}, device="cuda:0"
            ) as loader:
                rows = loader.take("w")
        self.assertTrue(torch.equal(cols.cpu(), w[:, 100:1124]))
        self.assertTrue(torch.equal(rows.cpu(), torch.cat([w[0:256], w[1792:2048]])))

    def test_plan_slices_allocate_their_bytes(self):
        # 16 MiB tensors; a quarter of the columns of one and of the rows of the other
        path = os.path.join(self.tempdir.name, "quarters.safetensors")
        save_file({k: torch.zeros(1024, 4096) for k in ("a", "b")}, path)
        plan = {"a": (slice(None), slice(0, 1024)), "b": slice(0, 256)}
        before = torch.cuda.memory_allocated(0)
        with self._open(path, plan=plan) as loader:
            held = dict(loader)
            used = torch.cuda.memory_allocated(0) - before
        self.assertEqual(
            {k: tuple(t.shape) for k, t in held.items()},
            {"a": (1024, 1024), "b": (256, 4096)},
        )
        self.assertEqual(used, 8 * MIB)

    def test_plan_rejects_bad_slices(self):
        with safe_open(self.path, framework="pt") as f:
            with self.assertRaisesRegex(
                Exception, "more slicing indexes than dimensions"
            ):
                f.prefetch(
                    {"fp32_2d": ([slice(None)], slice(None), slice(None))},
                    device="cuda:0",
                )
            with self.assertRaisesRegex(Exception, "overlap or are out of order"):
                f.prefetch({"fp32_2d": ([slice(2, 3), slice(0, 1)],)}, device="cuda:0")
            with self.assertRaisesRegex(Exception, "Unsupported slice index"):
                f.prefetch({"fp32_2d": (["a"],)}, device="cuda:0")
            with self.assertRaisesRegex(Exception, "ints, slices or lists"):
                f.prefetch({"fp32_2d": ([slice(0, 1)], ...)}, device="cuda:0")
            with self.assertRaisesRegex(Exception, "step 1"):
                f.prefetch({"fp32_2d": ([slice(0, 3, 2)],)}, device="cuda:0")
            with self.assertRaisesRegex(Exception, "out of bounds"):
                f.prefetch({"fp32_2d": (slice(None), slice(0, 9))}, device="cuda:0")

    def test_plan_rejects_too_fragmented_slices(self):
        path = os.path.join(self.tempdir.name, "wide.safetensors")
        save_file({"w": torch.zeros(2, 65536)}, path)
        every_other_column = [slice(i, i + 1) for i in range(0, 65536, 2)]
        with safe_open(path, framework="pt") as f:
            with self.assertRaisesRegex(
                Exception, "load the tensor whole and index it on the device"
            ):
                f.prefetch({"w": (slice(None), every_other_column)}, device="cuda:0")

    def test_torch_accounts_prefetched_memory(self):
        # torch allocates every tensor at prefetch and frees each one on drop; untaken
        # tensors go at close, and `empty_cache` hands it all back to the device
        mib = 2**20
        path = os.path.join(self.tempdir.name, "accounted.safetensors")
        save_file(
            {k: torch.zeros(16 * mib, dtype=torch.float32) for k in ("a", "b", "c")},
            path,
        )
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated(0)
        reserved = torch.cuda.memory_reserved(0)
        with self._open(path) as loader:
            a = loader.take("a")
            self.assertEqual(torch.cuda.memory_allocated(0) - allocated, 192 * mib)
            del a
            self.assertEqual(torch.cuda.memory_allocated(0) - allocated, 128 * mib)
        self.assertEqual(torch.cuda.memory_allocated(0), allocated)
        torch.cuda.empty_cache()
        self.assertEqual(torch.cuda.memory_reserved(0), reserved)

    def test_one_device_allocation_per_load(self):
        # 16 tensors over 10 MiB would be 16 cudaMallocs; the warm-up makes it one
        mib = 2**20
        path = os.path.join(self.tempdir.name, "many.safetensors")
        save_file({f"t{i}": torch.zeros(4 * mib) for i in range(16)}, path)
        torch.cuda.empty_cache()
        before = torch.cuda.memory_stats(0)["segment.all.allocated"]
        with self._open(path) as loader:
            held = dict(loader)
            segments = torch.cuda.memory_stats(0)["segment.all.allocated"] - before
        self.assertEqual(len(held), 16)
        self.assertLessEqual(segments, 1)

    def test_warm_pieces_are_bounded(self):
        # 12 x 64 MiB warm up as 3 pieces of 256 MiB: a tensor kept pins one piece
        path = os.path.join(self.tempdir.name, "pieces.safetensors")
        _write_u8(path, [64 * MIB] * 12, fill=False)
        gc.collect()
        torch.cuda.empty_cache()
        segments = torch.cuda.memory_stats(0)["segment.all.allocated"]
        reserved = torch.cuda.memory_reserved(0)
        with self._open(path) as loader:
            sd = dict(loader)
        self.assertEqual(
            torch.cuda.memory_stats(0)["segment.all.allocated"] - segments, 3
        )
        kept = sd["t5"]
        del sd
        torch.cuda.empty_cache()
        self.assertLessEqual(torch.cuda.memory_reserved(0) - reserved, 256 * MIB)
        del kept

    def test_no_warm_up_past_free_memory(self):
        # the tensors fit in torch's cache, their warm block not on the device: torch
        # would flush the cache to retry it
        path = os.path.join(self.tempdir.name, "full.safetensors")
        _write_u8(path, [16 * MIB] * 8, fill=False)
        gc.collect()
        torch.cuda.empty_cache()
        cached = [
            torch.empty(16 * MIB, dtype=torch.uint8, device="cuda:0") for _ in range(8)
        ]
        del cached
        free = torch.cuda.mem_get_info(0)[0]
        hog = torch.empty(free - 64 * MIB, dtype=torch.uint8, device="cuda:0")
        retries = torch.cuda.memory_stats(0)["num_alloc_retries"]
        with self._open(path) as loader:
            sd = dict(loader)
        self.assertEqual(torch.cuda.memory_stats(0)["num_alloc_retries"], retries)
        self.assertEqual(len(sd), 8)
        del sd, hog

    def test_reload_on_a_used_cache(self):
        # one warm block per load would reserve 1280 MiB here
        a = os.path.join(self.tempdir.name, "a.safetensors")
        b = os.path.join(self.tempdir.name, "b.safetensors")
        _write_u8(a, [64 * MIB] * 8, fill=False)
        _write_u8(b, [64 * MIB] * 12, fill=False)
        gc.collect()
        torch.cuda.empty_cache()
        reserved = torch.cuda.memory_reserved(0)
        with self._open(a) as loader:
            sd = dict(loader)
        del sd
        with self._open(b) as loader:
            sd = dict(loader)
        self.assertLessEqual(torch.cuda.memory_reserved(0) - reserved, GIB)

    def _in_a_fresh_process(self, check, env=None):
        # torch reads its allocator settings once per process
        path = os.path.join(self.tempdir.name, "fresh.safetensors")
        _write_u8(path, [16 * MIB] * 16, fill=False)
        script = (
            "import sys, torch\n"
            "from safetensors import safe_open\n"
            "with safe_open(sys.argv[1], framework='pt') as f:\n"
            "    with f.prefetch(device='cuda:0') as loader:\n"
            "        sd = dict(loader)\n" + check
        )
        subprocess.run([sys.executable, "-c", script, path], check=True, env=env)

    def test_warm_up_in_a_fresh_process(self):
        self._in_a_fresh_process(
            "n = torch.cuda.memory_stats(0)['segment.all.allocated']\n"
            "assert n <= 2, n\n"
        )

    def test_max_split_size(self):
        conf = "max_split_size_mb:64"
        self._in_a_fresh_process(
            "gap = torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)\n"
            "assert gap < 16 * 2**20, gap\n",
            dict(os.environ, PYTORCH_CUDA_ALLOC_CONF=conf, PYTORCH_ALLOC_CONF=conf),
        )

    def test_plan_empty_rows(self):
        with self._open(plan={"fp32_2d": slice(2, 2)}) as loader:
            t = loader.take("fp32_2d")
        self.assertEqual(tuple(t.shape), (0, 4))
        self.assertEqual(t.dtype, torch.float32)
        with self._open(plan={"fp16_3d": (slice(None), slice(1, 1))}) as loader:
            t = loader.take("fp16_3d")
        self.assertEqual(tuple(t.shape), (2, 0, 4))

    def test_two_loaders_from_one_handle(self):
        # one handle, several loaders (one per device in practice): the handle is unchanged
        with safe_open(self.path, framework="pt") as f:
            with f.prefetch({"fp32_2d": None}, device="cuda:0") as first:
                with f.prefetch({"bf16_2d": None}, device="cuda:0") as second:
                    self.assertTrue(
                        torch.equal(
                            first.take("fp32_2d").cpu(), SOURCE_TENSORS["fp32_2d"]
                        )
                    )
                    self.assertTrue(
                        torch.equal(
                            second.take("bf16_2d").cpu(), SOURCE_TENSORS["bf16_2d"]
                        )
                    )
                    # the handle's own reads keep working next to the loaders
                    self.assertTrue(
                        torch.equal(
                            f.get_slice("fp16_3d")[:1], SOURCE_TENSORS["fp16_3d"][:1]
                        )
                    )
                    self.assertTrue(
                        torch.equal(f.get_tensor("fp32_2d"), SOURCE_TENSORS["fp32_2d"])
                    )
                    self.assertEqual(f.get_tensor_meta("fp32_2d").shape, [3, 4])
                    self.assertEqual(set(f.get_tensors()), set(SOURCE_TENSORS))

    def test_loader_outlives_handle(self):
        # the loader owns its file reference and workers: the handle can go first
        with safe_open(self.path, framework="pt") as f:
            loader = f.prefetch(device="cuda:0")
        first = loader.take("fp32_2d")
        streamed = dict(loader)
        streamed["fp32_2d"] = first
        self._assert_matches_source(streamed)
        loader = safe_open(self.path, framework="pt").prefetch(
            device="cuda:0"
        )  # handle collected at once
        self._assert_matches_source(dict(loader))

    def test_int_device(self):
        # torch spells cuda:N as a bare N, on the handle and on the call
        with safe_open(self.path, framework="pt", device=0) as f:
            with f.prefetch() as loader:
                self._assert_matches_source(dict(loader))
        with safe_open(self.path, framework="pt") as f:
            with f.prefetch(device=0) as loader:
                self._assert_matches_source(dict(loader))

    def test_names_len_and_repeated_iteration(self):
        with self._open() as loader:
            self.assertEqual(loader.names(), self.handle.offset_keys())  # file order
            self.assertEqual(len(loader), len(SOURCE_TENSORS))
            first = dict(loader)
            second = dict(loader)  # a second pass finds nothing left
        self.assertEqual(set(first), set(SOURCE_TENSORS))
        self.assertEqual(second, {})

    def test_take_after_close_raises(self):
        loader = self._open()
        loader.close()
        with self.assertRaisesRegex(Exception, "closed"):
            loader.take("fp32_2d")

    def test_prefetch_on_closed_handle_raises(self):
        f = safe_open(self.path, framework="pt")
        f.__exit__(None, None, None)
        with self.assertRaisesRegex(Exception, "closed"):
            f.prefetch(device="cuda:0")

    def test_handle_exit_while_another_thread_reads(self):
        # the read in flight finishes on its own reference; the next call finds the handle closed
        size = 256 * 2**20
        path = os.path.join(self.tempdir.name, "big.safetensors")
        save_file({"t": torch.zeros(size, dtype=torch.uint8)}, path)
        f = safe_open(path, framework="pt", device="cuda:0", backend="pread")
        started = threading.Event()

        def read():
            started.set()
            return f.get_tensor("t")

        with ThreadPoolExecutor(1) as pool:
            fut = pool.submit(read)
            started.wait()
            time.sleep(0.01)
            f.__exit__(None, None, None)
        self.assertEqual(fut.result().numel(), size)
        with self.assertRaisesRegex(SafetensorError, "closed"):
            f.get_tensor("t")

    def test_threads(self):
        # readers are capped at one per chunk: tiny files spawn few, absurd requests spawn none extra
        for threads in (2, 10_000):
            with self._open(threads=threads) as loader:
                sd = dict(loader)
            self._assert_matches_source(sd)

    @unittest.skipUnless(
        hasattr(torch, "float4_e2m1fn_x2"), "float4_e2m1fn_x2 requires torch 2.8"
    )
    def test_plan_rows_of_packed_fp4(self):
        # F4 shape [4, 8] packs two elements per byte: 4 bytes per row
        path = os.path.join(self.tempdir.name, "fp4.safetensors")
        header = b'{"x":{"dtype":"F4","shape":[4,8],"data_offsets":[0,16]}}'
        raw = bytes(range(16))
        with open(path, "wb") as fh:
            fh.write(struct.pack("<Q", len(header)))
            fh.write(header)
            fh.write(raw)
        with self._open(path, plan={"x": slice(1, 3)}) as loader:
            x = loader.take("x")
        self.assertEqual(x.dtype, torch.float4_e2m1fn_x2)
        self.assertEqual(tuple(x.shape), (2, 4))
        self.assertEqual(x.view(torch.uint8).cpu().flatten().tolist(), list(raw[4:12]))
        # two F4 elements per byte: a column slice must cut on whole bytes
        with self._open(path, plan={"x": (slice(None), slice(2, 6))}) as loader:
            cols = loader.take("x")
        self.assertEqual(tuple(cols.shape), (4, 2))
        self.assertEqual(
            cols.view(torch.uint8).cpu().tolist(),
            [list(raw[r * 4 + 1 : r * 4 + 3]) for r in range(4)],
        )
        with self.assertRaisesRegex(Exception, "byte boundary"):
            self._open(path, plan={"x": (slice(None), slice(1, 5))})

    def _assert_matches_source(self, sd):
        self.assertEqual(set(sd.keys()), set(SOURCE_TENSORS.keys()))
        for k, expected in SOURCE_TENSORS.items():
            got = sd[k]
            self.assertEqual(got.device.type, "cuda", k)
            self.assertEqual(got.dtype, expected.dtype, k)
            self.assertEqual(tuple(got.shape), tuple(expected.shape), k)
            if expected.numel() > 0:
                self.assertTrue(_tensors_equal(got.cpu(), expected), k)

    def test_iteration_matches_source(self):
        with self._open() as loader:
            sd = dict(loader)
        self._assert_matches_source(sd)

    def test_take_matches_source(self):
        with self._open() as loader:
            sd = {k: loader.take(k) for k in loader.names()}
        self._assert_matches_source(sd)

    def test_take_once(self):
        with self._open() as loader:
            loader.take("fp32_2d")
            with self.assertRaisesRegex(Exception, "already delivered"):
                loader.take("fp32_2d")

    def test_iteration_skips_taken(self):
        with self._open() as loader:
            loader.take("fp32_2d")
            names = {name for name, _ in loader}
        self.assertEqual(names, set(SOURCE_TENSORS.keys()) - {"fp32_2d"})

    def test_views_outlive_loader(self):
        # Delivered tensors own their memory: reading after close is valid.
        with self._open() as loader:
            sd = dict(loader)
        torch.cuda.synchronize()
        self._assert_matches_source(sd)

    def test_tensors_own_their_storage(self):
        # one storage per tensor, none a view of a shared block: save_file takes them
        with self._open() as loader:
            sd = dict(loader)
        for name, tensor in sd.items():
            self.assertEqual(tensor.untyped_storage().nbytes(), tensor.nbytes, name)
        out = os.path.join(self.tempdir.name, "resaved.safetensors")
        save_file(sd, out)
        self._assert_matches_source(load_file_pt(out, device="cuda:0"))

    def test_load_file_to_cuda(self):
        # through prefetch, onto the handle's device, from either backend
        for i in range(min(2, torch.cuda.device_count())):
            for backend in ("mmap", "pread"):
                sd = load_file_pt(self.path, device=f"cuda:{i}", backend=backend)
                self._assert_matches_source(sd)
                self.assertTrue(all(t.device.index == i for t in sd.values()))

    def test_take_from_threads(self):
        # how transformers calls it: takes from a thread pool
        with self._open() as loader:
            names = loader.names()
            with ThreadPoolExecutor(4) as pool:
                sd = dict(zip(names, pool.map(loader.take, names)))
        self._assert_matches_source(sd)

    def test_take_on_another_stream(self):
        # handed out on a stream other than the allocating one: recorded on that stream
        side = torch.cuda.Stream(device=0)
        with (
            self._open() as loader,
            mock.patch.object(torch.Tensor, "record_stream") as record,
        ):
            a = loader.take("fp32_2d")
            record.assert_not_called()
            with torch.cuda.stream(side):
                b = loader.take("i64_1d")
            record.assert_called_once_with(side)
        self.assertTrue(torch.equal(a.cpu(), SOURCE_TENSORS["fp32_2d"]))
        self.assertTrue(torch.equal(b.cpu(), SOURCE_TENSORS["i64_1d"]))

    def test_prefetch_on_a_side_stream(self):
        # allocated on a side stream, handed out on the default one: every tensor is
        # recorded for real, the empty and the 0-d one included
        side = torch.cuda.Stream(device=0)
        with torch.cuda.stream(side):
            loader = self._open()
        with loader:
            sd = dict(loader)
        torch.cuda.synchronize()
        self._assert_matches_source(sd)

    def test_waits_for_the_allocating_stream(self):
        # in deterministic mode torch.empty fills new tensors with a kernel on the
        # allocating stream, queued here behind about 1 s of matmuls: a copy that
        # didn't wait for the fill would be overwritten
        path = os.path.join(self.tempdir.name, "pattern.safetensors")
        sizes = [64 * 1024 + i for i in range(16)]
        _write_u8(path, sizes)
        x = torch.randn(16384, 16384, device="cuda:0")
        n = max(2, round(1.0 / _matmul_seconds(x)))
        # the switch, the fill kernel and the staging pool are slow the first time
        torch.use_deterministic_algorithms(True)
        try:
            with self._open(path) as loader:
                dict(loader)
        finally:
            torch.use_deterministic_algorithms(False)
        torch.cuda.synchronize()
        for _ in range(n):
            x @ x  # before the switch: cuBLAS refuses deterministic mode
        torch.use_deterministic_algorithms(True)
        try:
            loader = self._open(path)
        finally:
            torch.use_deterministic_algorithms(False)
        self.assertFalse(
            torch.cuda.current_stream(0).query(), "the allocating stream is idle"
        )
        with loader:
            sd = dict(loader)
        torch.cuda.synchronize()
        for i, size in enumerate(sizes):
            self.assertTrue(torch.equal(sd[f"t{i}"], _pattern(i, size)), f"t{i}")

    def test_close_mid_stream(self):
        loader = self._open()
        stream = iter(loader)
        next(stream)
        loader.close()
        with self.assertRaisesRegex(Exception, "closed"):
            for _ in stream:
                pass

    def test_truncated_data_raises(self):
        bad_path = os.path.join(self.tempdir.name, "short.safetensors")
        header = b'{"t":{"dtype":"F32","shape":[1024],"data_offsets":[0,4096]}}'
        with open(bad_path, "wb") as fh:
            fh.write(struct.pack("<Q", len(header)))
            fh.write(header)
            fh.write(b"\x00" * 100)
        with self.assertRaises(Exception), self._open(bad_path) as loader:
            loader.take("t")

    @unittest.skipUnless(sys.platform == "linux", "reads /proc/self/maps")
    def test_reuses_framework_cuda_runtime(self):
        # The engine must attach to the libcudart torch already loaded, never
        # bring up a second runtime next to it.
        def mapped_cudarts():
            with open("/proc/self/maps") as maps:
                return {line.split()[-1] for line in maps if "libcudart.so" in line}

        torch.zeros(1, device="cuda:0")
        before = mapped_cudarts()
        self.assertTrue(before)
        with self._open() as loader:
            loader.take(loader.names()[0])
        self.assertEqual(mapped_cudarts(), before)

    @unittest.skipUnless(
        hasattr(torch, "float8_e5m2fnuz"), "torch lacks float8_e5m2fnuz"
    )
    def test_fnuz_dtype_view(self):
        path = os.path.join(self.tempdir.name, "fnuz.safetensors")
        header = b'{"x":{"dtype":"F8_E5M2FNUZ","shape":[4],"data_offsets":[0,4]}}'
        raw = bytes([1, 2, 3, 4])
        with open(path, "wb") as fh:
            fh.write(struct.pack("<Q", len(header)))
            fh.write(header)
            fh.write(raw)
        with self._open(path) as loader:
            x = loader.take("x")
        self.assertEqual(x.dtype, torch.float8_e5m2fnuz)
        self.assertEqual(x.view(torch.uint8).cpu().tolist(), list(raw))

    def test_misaligned_offsets(self):
        # save_file keeps tensors aligned; hand-build a file where an F32 tensor
        # sits at byte offset 3: it still lands in an aligned torch tensor.
        bad_path = os.path.join(self.tempdir.name, "misaligned.safetensors")
        header = (
            b'{"u8":{"dtype":"U8","shape":[3],"data_offsets":[0,3]},'
            b'"f32":{"dtype":"F32","shape":[4],"data_offsets":[3,19]}}'
        )
        with open(bad_path, "wb") as fh:
            fh.write(struct.pack("<Q", len(header)))
            fh.write(header)
            fh.write(bytes([1, 2, 3]))
            fh.write(struct.pack("<4f", 0.0, 1.0, 2.0, 3.0))
        with self._open(bad_path) as loader:
            f32 = loader.take("f32")
            u8 = loader.take("u8")
        self.assertTrue(torch.equal(f32.cpu(), torch.arange(4, dtype=torch.float32)))
        self.assertTrue(
            torch.equal(u8.cpu(), torch.tensor([1, 2, 3], dtype=torch.uint8))
        )

    @unittest.skipUnless(hasattr(os, "posix_fadvise"), "evicts the file from the cache")
    def test_direct_reads_match_cpu_load(self):
        # data at an odd offset, file length not a multiple of 4 KiB, read uncached
        path = os.path.join(self.tempdir.name, "odd.safetensors")
        header = json.dumps(
            {
                "w": {
                    "dtype": "I32",
                    "shape": [300, 1000],
                    "data_offsets": [0, 1200000],
                },
                "tail": {
                    "dtype": "U8",
                    "shape": [777],
                    "data_offsets": [1200000, 1200777],
                },
            }
        ).encode()
        header += b" " * (1 - len(header) % 2)
        data = torch.randint(0, 256, (1200777,), dtype=torch.uint8).numpy().tobytes()
        with open(path, "wb") as fh:
            fh.write(struct.pack("<Q", len(header)) + header + data)
            os.fsync(fh.fileno())
        cpu = load_file_pt(path, backend="pread")
        plan = {"w": (slice(5, 290), [slice(3, 40), slice(500, 777)]), "tail": None}
        w = cpu["w"][5:290]
        sliced = {"w": torch.cat([w[:, 3:40], w[:, 500:777]], 1), "tail": cpu["tail"]}
        for direct in (None, True, False):
            for p, want in ((None, cpu), (plan, sliced)):
                fd = os.open(path, os.O_RDONLY)
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                os.close(fd)
                with (
                    safe_open(path, framework="pt") as f,
                    f.prefetch(p, device="cuda:0", direct=direct) as loader,
                ):
                    got = dict(loader)
                for name, t in want.items():
                    self.assertTrue(torch.equal(got[name].cpu(), t), (direct, name))


@unittest.skipIf(not torch.cuda.is_available(), "Cuda is not available")
class PrefetchAllocationTests(unittest.TestCase):
    """Plans over the 1 GiB torch allocates at `prefetch`: the rest is allocated
    in the background as the load advances."""

    # odd sizes so chunks carry several tensors and batch ends fall inside chunks;
    # zero-size and small tensors in every batch
    SIZES = [
        n
        for k in range(200)
        for n in (
            (k % 7 + 1) * 3 * MIB + 17 * k,
            0 if k % 9 == 0 else 4096 + k,
            1 + k % 3,
        )
    ] + [0, 5, 0]

    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.TemporaryDirectory()
        cls.path = os.path.join(cls.tempdir.name, "pattern.safetensors")
        _write_u8(cls.path, cls.SIZES)

    @classmethod
    def tearDownClass(cls):
        cls.tempdir.cleanup()

    def _check(self, sd):
        self.assertEqual(len(sd), len(self.SIZES))
        for i, n in enumerate(self.SIZES):
            t = sd[f"t{i}"]
            self.assertEqual(t.numel(), n, i)
            if n:
                self.assertTrue(torch.equal(t, _pattern(i, n)), f"t{i} ({n} bytes)")

    def _fill_device_at(self, extra):
        # torch can allocate `extra` bytes more on cuda:0
        gc.collect()
        torch.cuda.empty_cache()
        before = torch.cuda.memory_allocated(0)
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction((before + extra) / total, 0)
        self.addCleanup(torch.cuda.set_per_process_memory_fraction, 1.0, 0)
        return before

    def test_bytes_land_across_batches(self):
        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.mem_get_info(0)[0] < 9 * GIB:
            self.skipTest("needs 9 GiB free on cuda:0")
        for threads in (1, 8, 32):
            for how in ("iterate", "take", "take reversed"):
                with self.subTest(threads=threads, how=how):
                    # torch's cache holds 0xFF where the tensors go: a missed copy shows
                    torch.full((6 * GIB,), 255, dtype=torch.uint8, device="cuda:0")
                    with (
                        safe_open(self.path, framework="pt") as f,
                        f.prefetch(device="cuda:0", threads=threads) as loader,
                    ):
                        names = loader.names()
                        if how == "iterate":
                            sd = dict(loader)
                        else:
                            order = names[::-1] if how == "take reversed" else names
                            sd = {n: loader.take(n) for n in order}
                    torch.cuda.synchronize()
                    self._check(sd)
                    del sd

    def test_adjacent_destinations_land(self):
        # torch packs small tensors in 512-byte blocks: 512-multiples sit back to back,
        # odd sizes leave holes; with zero sizes, plan gaps, odd row slices, large-pool
        # tensors, and spans across chunks and batches in between
        path = os.path.join(self.tempdir.name, "adjacent.safetensors")
        sizes = [
            n
            for k in range(3000)
            for n in (512 * (k % 4 + 1), 1 + k % 700, 0 if k % 13 == 0 else 1024)
            + ((5 * MIB + 512, 3 * MIB) if k % 500 == 0 else ())
        ]
        _write_u8(path, sizes)
        plan = {f"t{i}": None for i in range(len(sizes)) if i % 10 != 7}
        plan.update(
            {
                f"t{i}": slice(3, sizes[i] - 1)
                for i in range(5, len(sizes), 101)
                if sizes[i] > 4
            }
        )
        for threads in (1, 8):
            with self.subTest(threads=threads):
                gc.collect()
                torch.cuda.empty_cache()
                # 0xFF in torch's cache, small and large pools alike
                dirty = [
                    torch.full((n,), 255, dtype=torch.uint8, device="cuda:0")
                    for n in [MIB] * 64 + [GIB]
                ]
                del dirty
                with (
                    safe_open(path, framework="pt") as f,
                    f.prefetch(plan, device="cuda:0", threads=threads) as loader,
                ):
                    sd = dict(loader)
                torch.cuda.synchronize()
                self.assertEqual(set(sd), set(plan))
                for name, rows in plan.items():
                    i = int(name[1:])
                    expected = _pattern(i, sizes[i])[rows or slice(None)]
                    self.assertTrue(
                        torch.equal(sd[name], expected), f"{name} ({sizes[i]} bytes)"
                    )
                del sd

    def test_allocator_thread_uses_the_callers_stream(self):
        # every block holding a tensor comes from the stream current at prefetch,
        # including the background ones; hand-out on that stream records nothing
        side = torch.cuda.Stream(device=0)
        gc.collect()
        torch.cuda.empty_cache()
        with torch.cuda.stream(side):
            with (
                safe_open(self.path, framework="pt") as f,
                f.prefetch(device="cuda:0") as loader,
            ):
                with mock.patch.object(torch.Tensor, "record_stream") as record:
                    sd = dict(loader)
                record.assert_not_called()
        ptrs = {t.data_ptr() for t in sd.values() if t.numel()}
        streams = set()
        for seg in torch.cuda.memory_snapshot():
            addr = seg["address"]
            if any(addr <= p < addr + seg["total_size"] for p in ptrs):
                streams.add(seg["stream"])
        self.assertEqual(streams, {side.cuda_stream})
        torch.cuda.synchronize()
        self._check(sd)

    def test_side_stream_load_ignores_queued_work(self):
        # started under a side stream, with at least 4 s of matmuls queued on the default
        # stream as the tensors are taken: the takes return long before that work drains
        path = os.path.join(self.tempdir.name, "side.safetensors")
        _write_u8(path, [64 * MIB] * 32, fill=False)  # 2 GiB: past the first batch
        x = torch.randn(16384, 16384, device="cuda:0")
        per_take = -int(-4.0 // (32 * _matmul_seconds(x)))  # rounded up
        side = torch.cuda.Stream(device=0)
        with torch.cuda.stream(side):
            loader = safe_open(path, framework="pt").prefetch(device="cuda:0")
        with loader:
            dict(loader)  # the allocator's blocks for the side stream
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.cuda.stream(side):
            loader = safe_open(path, framework="pt").prefetch(device="cuda:0")
        with loader:
            taken = []
            for name in loader.names():
                taken.append(loader.take(name))
                for _ in range(per_take):
                    x @ x
            takes = time.perf_counter() - start
        torch.cuda.synchronize()
        drained = time.perf_counter() - start
        self.assertLess(
            takes, drained / 4, f"takes {takes:.2f} s, work drained {drained:.2f} s"
        )

    def test_footprint_follows_the_load(self):
        gc.collect()
        before = torch.cuda.memory_allocated(0)
        with (
            safe_open(self.path, framework="pt") as f,
            f.prefetch(device="cuda:0", threads=1) as loader,
        ):
            first = torch.cuda.memory_allocated(0) - before
            sd = dict(loader)
            # torch's blocks, or more where it doesn't split off a small remainder
            rounded = sum(-(-n // 512) * 512 for n in self.SIZES)
            self.assertGreaterEqual(torch.cuda.memory_allocated(0) - before, rounded)
        self.assertGreaterEqual(first, GIB)
        # the allocator thread may have topped up already, far from all of it
        self.assertLess(first, 2 * GIB)
        del sd
        gc.collect()
        self.assertEqual(torch.cuda.memory_allocated(0), before)

    def test_prefetch_after_the_exit_hook_raises(self):
        # a loader started once the `atexit` hook ran would keep its allocator
        # thread into finalization
        path = os.path.join(self.tempdir.name, "exit.safetensors")
        _write_u8(path, [MIB])
        script = (
            "import atexit, sys, torch\n"
            "from safetensors import safe_open\n"
            "f = safe_open(sys.argv[1], framework='pt')\n"
            "f.prefetch(device='cuda:0').close()\n"
            "atexit._run_exitfuncs()\n"
            "try:\n"
            "    f.prefetch(device='cuda:0')\n"
            "except Exception as e:\n"
            "    assert 'interpreter exits' in str(e), e\n"
            "else:\n"
            "    raise AssertionError('prefetch started')\n"
        )
        subprocess.run([sys.executable, "-c", script, path], check=True)

    def test_first_batch_must_fit(self):
        path = os.path.join(self.tempdir.name, "first.safetensors")
        _write_u8(path, [64 * MIB] * 64, fill=False)
        before = self._fill_device_at(512 * MIB)
        with self.assertRaisesRegex(SafetensorError, "cannot allocate") as failure:
            safe_open(path, framework="pt").prefetch(device="cuda:0")
        self.assertIsInstance(failure.exception.__cause__, torch.OutOfMemoryError)
        self.assertEqual(torch.cuda.memory_allocated(0), before)

    def test_allocation_failure_ends_the_load(self):
        # the device fills after the first batch: tensors taken before stay, later
        # takes raise with torch's error as the cause once the loader released the
        # tensors not taken, without waiting for close
        path = os.path.join(self.tempdir.name, "fill.safetensors")
        _write_u8(path, [64 * MIB] * 64, fill=False)
        before = self._fill_device_at(2 * GIB)
        loader = safe_open(path, framework="pt").prefetch(device="cuda:0")
        taken = []
        with self.assertRaisesRegex(SafetensorError, "cannot allocate") as failure:
            for name in loader.names():
                taken.append(loader.take(name))
        self.assertIsInstance(failure.exception.__cause__, torch.OutOfMemoryError)
        self.assertGreater(len(taken), 0)
        held = before + len(taken) * 64 * MIB
        self.assertEqual(torch.cuda.memory_allocated(0), held)
        with self.assertRaises(SafetensorError) as again:
            dict(loader)
        self.assertIsInstance(again.exception.__cause__, torch.OutOfMemoryError)
        loader.close()
        self.assertEqual(torch.cuda.memory_allocated(0), held)

    def test_dropped_loader_releases_its_tensors(self):
        # dropped mid-load without `close`: the allocator thread stops and every
        # tensor not taken goes
        path = os.path.join(self.tempdir.name, "dropped.safetensors")
        _write_u8(path, [64 * MIB] * 64, fill=False)
        gc.collect()
        before, running = torch.cuda.memory_allocated(0), _allocator_threads()
        loader = safe_open(path, framework="pt").prefetch(device="cuda:0", threads=1)
        self.assertLess(torch.cuda.memory_allocated(0) - before, 4 * GIB)
        del loader
        gc.collect()
        self.assertEqual(_allocator_threads(), running)
        self.assertEqual(torch.cuda.memory_allocated(0), before)

    def _race(self, call, close):
        # runs `call` in a thread, and `close` here once the thread is inside it
        started = threading.Event()

        def run():
            started.set()
            return call()

        with ThreadPoolExecutor(1) as pool:
            fut = pool.submit(run)
            started.wait()
            time.sleep(0.1)
            close()
        return fut

    def test_close_while_another_thread_takes(self):
        # `t511` sits 32 GiB into a hole: no host loads that far in the 0.1 s before
        # the close, so the take can't succeed (only ~1 GiB ahead is allocated)
        path = os.path.join(self.tempdir.name, "slow.safetensors")
        _write_u8(path, [64 * MIB] * 512, fill=False)
        closers = {
            "close": lambda loader: loader.close(),
            "__exit__": lambda loader: loader.__exit__(None, None, None),
        }
        for name, close in closers.items():
            with self.subTest(closer=name):
                gc.collect()
                before = torch.cuda.memory_allocated(0)
                loader = safe_open(path, framework="pt").prefetch(
                    device="cuda:0", threads=1
                )
                fut = self._race(lambda: loader.take("t511"), lambda: close(loader))
                with self.assertRaisesRegex(SafetensorError, "closed"):
                    fut.result()
                loader.close()
                loader.__exit__(None, None, None)
                del fut
                self.assertEqual(torch.cuda.memory_allocated(0), before)


if __name__ == "__main__":
    unittest.main()
