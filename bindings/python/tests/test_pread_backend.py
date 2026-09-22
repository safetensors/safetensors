"""Tests for `safe_open(..., backend="pread")`.

The `pread` backend serves each tensor via `pread(2)` instead of mmap'ing
the file, dropping each host buffer immediately after the device transfer so
cumulative host residency stays bounded at one tensor.
"""

import json
import os
import struct
import sys
import tempfile
import unittest

import numpy as np
import torch
from safetensors import safe_open
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
            with self.assertRaisesRegex(Exception, "cuda devices, not cpu"):
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
            with self.assertRaisesRegex(Exception, "0-d"):
                f.prefetch({"scalar_fp32": slice(0, 1)})
            with self.assertRaisesRegex(Exception, "None or a slice"):
                f.prefetch({"fp32_2d": 3})

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
            with self.assertRaisesRegex(Exception, "cuda devices, not cpu"):
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


@unittest.skipIf(not torch.cuda.is_available(), "Cuda is not available")
class PrefetchCudaTests(unittest.TestCase):
    """`safe_open(...).prefetch(...)`: a `PrefetchLoader` reading the file into
    device memory in the background, tensors handed out once as zero-copy views."""

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
        # a, b, c of 64 MiB each; the plan skips b: the device must hold ~128 MiB, not 192
        mib = 2**20
        path = os.path.join(self.tempdir.name, "gap.safetensors")
        save_file(
            {k: torch.zeros(16 * mib, dtype=torch.float32) for k in ("a", "b", "c")},
            path,
        )
        torch.cuda.synchronize()
        free_before, _ = torch.cuda.mem_get_info(0)
        with self._open(path, plan={"a": None, "c": None}) as loader:
            held = dict(loader)
            torch.cuda.synchronize()
            free_after, _ = torch.cuda.mem_get_info(0)
        self.assertEqual(set(held), {"a", "c"})
        used = free_before - free_after
        self.assertGreaterEqual(used, 120 * mib, f"{used / mib:.0f} MiB")
        self.assertLess(
            used, 176 * mib, f"{used / mib:.0f} MiB resident, gap was allocated"
        )

    def test_plan_rejects_out_of_range_rows(self):
        # Python would clamp slice(0, 10) on 3 rows to 3 rows; a plan is explicit
        with safe_open(self.path, framework="pt") as f:
            with self.assertRaisesRegex(Exception, "out of range"):
                f.prefetch({"fp32_2d": slice(0, 10)}, device="cuda:0")
            with self.assertRaisesRegex(Exception, "invalid prefetch plan slice"):
                f.prefetch({"fp32_2d": slice(0, 3, 0)}, device="cuda:0")
            with self.assertRaisesRegex(Exception, "selects no tensors"):
                f.prefetch({}, device="cuda:0")
            with f.prefetch({"fp32_2d": slice(-3, 3)}, device="cuda:0") as loader:
                self.assertEqual(tuple(loader.take("fp32_2d").shape), (3, 4))

    def test_delivered_allocations_are_released_before_close(self):
        # a loader kept open must not pin shards whose tensors were all handed out and dropped:
        # loading the same 192 MiB file twice with both loaders open costs about one file
        mib = 2**20
        path = os.path.join(self.tempdir.name, "twice.safetensors")
        save_file(
            {k: torch.zeros(16 * mib, dtype=torch.float32) for k in ("a", "b", "c")},
            path,
        )
        torch.cuda.synchronize()
        free_before, _ = torch.cuda.mem_get_info(0)
        loaders = []
        with safe_open(path, framework="pt") as f:
            for _ in range(2):
                loader = f.prefetch(device="cuda:0")
                loaders.append(loader)
                tensors = dict(loader)
                self.assertEqual(set(tensors), {"a", "b", "c"})
                del tensors
                torch.cuda.synchronize()  # the frees are issued at drop; let them complete
            free_after, _ = torch.cuda.mem_get_info(0)
        for loader in loaders:
            loader.close()
        used = free_before - free_after
        self.assertGreaterEqual(
            used, 150 * mib, f"{used / mib:.0f} MiB: nothing stayed resident"
        )
        self.assertLess(
            used, 300 * mib, f"{used / mib:.0f} MiB resident with both loaders open"
        )

    def test_plan_empty_rows(self):
        with self._open(plan={"fp32_2d": slice(2, 2)}) as loader:
            t = loader.take("fp32_2d")
        self.assertEqual(tuple(t.shape), (0, 4))
        self.assertEqual(t.dtype, torch.float32)

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
        # sits at byte offset 3 to exercise the realignment path.
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


if __name__ == "__main__":
    unittest.main()
