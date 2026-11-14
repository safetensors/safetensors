import copy
import os
import tempfile
import unittest

import torch

from safetensors import safe_open
from safetensors.torch import (
    _end_ptr,
    _find_shared_tensors,
    _is_canonical_dense_stride,
    _is_complete,
    _remove_duplicate_names,
    _tensors_overlap,
    load_model,
    save_file,
    save_model,
)


class OnesModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(4, 4)
        self.a.weight = torch.nn.Parameter(torch.ones((4, 4)))
        self.a.bias = torch.nn.Parameter(torch.ones((4,)))
        self.b = self.a


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(100, 100)
        self.b = self.a


class NonContiguousModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(100, 100)
        A = torch.zeros((100, 100))
        A = A.transpose(0, 1)
        self.a.weight = torch.nn.Parameter(A)


class CopyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(100, 100)
        self.b = copy.deepcopy(self.a)


class NoSharedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(100, 100)
        self.b = torch.nn.Linear(100, 100)


class TorchModelTestCase(unittest.TestCase):
    def test_is_complete(self):
        A = torch.zeros((3, 3))
        self.assertTrue(_is_complete(A))

        B = A[:1, :]
        self.assertFalse(_is_complete(B))

        # Covers the whole storage but with holes
        C = A[::2, :]
        self.assertFalse(_is_complete(C))

        D = torch.zeros((2, 2), device=torch.device("meta"))
        self.assertTrue(_is_complete(D))

    def test_is_complete_canonical_dense_layouts(self):
        # Direct predicate validation on canonical vs non-canonical strides
        self.assertTrue(_is_canonical_dense_stride((3, 3), (3, 1)))
        self.assertFalse(_is_canonical_dense_stride((2, 2), (2, 0)))
        self.assertFalse(_is_canonical_dense_stride((2,), (2, 1)))

        # 1. Ordinary contiguous tensor is complete
        # Shape (3, 3) with standard strides (3, 1) covers elements 0..8
        A = torch.zeros((3, 3))
        self.assertTrue(_is_complete(A))

        # 2. Transposed dense tensor is complete
        # Shape (4, 3) with strides (1, 4) bijectively covers elements 0..11
        A_t = torch.zeros((3, 4)).t()
        self.assertTrue(_is_complete(A_t))

        # 3. Permuted dense tensor is complete
        # Shape (4, 2, 3) with strides (1, 12, 4) bijectively covers elements 0..23
        A_p = torch.zeros((2, 3, 4)).permute(2, 0, 1)
        self.assertTrue(_is_complete(A_p))

        # 4. Expanded zero-stride view is incomplete (bugfix: prevents silent omission)
        # raw[::2] has 2 elements; expand(2, 2) has stride (2, 0), touching only offsets 0 and 2
        raw4 = torch.tensor([1.0, 2.0, 3.0, 4.0])
        A_exp = raw4[::2].unsqueeze(1).expand(2, 2)
        self.assertFalse(_is_complete(A_exp))

        # 5. Overlapping positive-stride as_strided view is incomplete
        # Coordinates (0, 1) and (1, 0) both evaluate to offset 1; offset 3 is omitted
        raw5 = torch.arange(4, dtype=torch.float32)
        A_pos = torch.as_strided(raw5, (2, 2), (1, 1))
        self.assertFalse(_is_complete(A_pos))

        # 6. Gapped view is incomplete
        # Shape (2, 4) with strides (8, 1) leaves rows 1 and 3 omitted from 16-element buffer
        base6 = torch.zeros((4, 4))
        A_gap = base6[::2, :]
        self.assertFalse(_is_complete(A_gap))

        # 7. Nonzero storage offset view is incomplete
        # Offset 1 on 10-element buffer omits index 0
        base7 = torch.arange(10, dtype=torch.float32)
        A_off = base7[1:]
        self.assertFalse(_is_complete(A_off))

        # 8. Empty view backed by nonempty storage is incomplete
        # nelement == 0 but storage_size == 40 bytes; cannot own the nonempty storage
        base8 = torch.arange(10, dtype=torch.float32)
        A_empty_slice = base8[:0]
        self.assertFalse(_is_complete(A_empty_slice))

        # 9. Scalar tensor is complete
        # 0-D scalar with shape () and stride () addresses offset 0
        A_scalar = torch.tensor(42.0)
        self.assertTrue(_is_complete(A_scalar))

        # 10. Singleton dimensions preserve correct dense classification
        # Shape (2, 1, 3) with arbitrary stride on singleton dim 1 bijectively spans 6 elements
        A_single = torch.zeros((2, 1, 3))
        self.assertTrue(_is_complete(A_single))

        # 11. Storage bounds are respected
        # Sub-matrix (2, 2) on 3x4 buffer covers only 4 elements of 12
        base11 = torch.zeros((3, 4))
        A_sub = base11[:2, :2]
        self.assertFalse(_is_complete(A_sub))

        # 12. Meta device tensor preserves supported behavior without storage access
        A_meta = torch.zeros((2, 2), device=torch.device("meta"))
        self.assertTrue(_is_complete(A_meta))

    def test_save_model_rejects_expanded_zero_stride_checkpoint_omission(self):
        # Demonstrates that an expanded zero-stride view cannot claim storage completeness
        # and cause valid sibling parameters to be silently omitted from the checkpoint.
        class ExpandedSharedModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                raw = torch.tensor([10.0, 20.0, 30.0, 40.0], dtype=torch.float32)
                self.expand_view = torch.nn.Parameter(raw[0:1].expand(4))
                self.slice_mid = torch.nn.Parameter(raw[1:2])

        model = ExpandedSharedModel()
        with tempfile.TemporaryDirectory() as tmp:
            filename = os.path.join(tmp, "model_expanded.safetensors")
            with self.assertRaises(RuntimeError) as ctx:
                save_model(model, filename)
            self.assertIn("None is covering the entire storage", str(ctx.exception))

        # Also assert that when a dense base tensor is present alongside an expanded view,
        # the base tensor is correctly selected as keep_name even if the expanded tensor
        # sorts alphabetically earlier.
        class ExpandedWithBaseModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                raw = torch.tensor([10.0, 20.0, 30.0, 40.0], dtype=torch.float32)
                self.a_expanded = torch.nn.Parameter(raw[0:1].expand(4))
                self.b_base = torch.nn.Parameter(raw)

        model_with_base = ExpandedWithBaseModel()
        to_removes = _remove_duplicate_names(model_with_base.state_dict())
        self.assertIn("b_base", to_removes)
        self.assertEqual(to_removes["b_base"], ["a_expanded"])

    def test_find_shared_tensors(self):
        A = torch.zeros((3, 3))
        B = A[:1, :]

        self.assertEqual(_find_shared_tensors({"A": A, "B": B}), [{"A", "B"}])
        self.assertEqual(_find_shared_tensors({"A": A}), [{"A"}])
        self.assertEqual(_find_shared_tensors({"B": B}), [{"B"}])

        C = torch.zeros((2, 2), device=torch.device("meta"))
        D = C[:1]
        # Meta device is not shared
        self.assertEqual(_find_shared_tensors({"C": C, "D": D}), [])
        self.assertEqual(_find_shared_tensors({"C": C}), [])
        self.assertEqual(_find_shared_tensors({"D": D}), [])

    def test_find_shared_non_shared_tensors(self):
        A = torch.zeros((4,))
        B = A[:2]
        C = A[2:]
        # Shared storage but do not overlap
        self.assertEqual(_find_shared_tensors({"B": B, "C": C}), [{"B"}, {"C"}])

        B = A[:2]
        C = A[1:]
        # Shared storage but *do* overlap
        self.assertEqual(_find_shared_tensors({"B": B, "C": C}), [{"B", "C"}])

        B = A[:2]
        C = A[2:]
        D = A[:1]
        # Shared storage but *do* overlap
        self.assertEqual(
            _find_shared_tensors({"B": B, "C": C, "D": D}), [{"B", "D"}, {"C"}]
        )

    def test_end_ptr(self):
        A = torch.zeros((4,))
        start = A.data_ptr()
        end = _end_ptr(A)
        self.assertEqual(end - start, 16)
        B = torch.zeros((16,))
        A = B[::4]
        start = A.data_ptr()
        end = _end_ptr(A)
        # Jump 3 times 16 byes (the stride of B)
        # Then add the size of the datapoint 4 bytes
        self.assertEqual(end - start, 16 * 3 + 4)

        # FLOAT16
        A = torch.zeros((4,), dtype=torch.float16)
        start = A.data_ptr()
        end = _end_ptr(A)
        self.assertEqual(end - start, 8)
        B = torch.zeros((16,), dtype=torch.float16)
        A = B[::4]
        start = A.data_ptr()
        end = _end_ptr(A)
        # Jump 3 times 8 bytes (the stride of B)
        # Then add the size of the datapoint 4 bytes
        self.assertEqual(end - start, 8 * 3 + 2)

    def test_remove_duplicate_names(self):
        A = torch.zeros((3, 3))
        B = A[:1, :]

        self.assertEqual(_remove_duplicate_names({"A": A, "B": B}), {"A": ["B"]})
        self.assertEqual(
            _remove_duplicate_names({"A": A, "B": B, "C": A}), {"A": ["B", "C"]}
        )
        with self.assertRaises(RuntimeError):
            self.assertEqual(_remove_duplicate_names({"B": B}), [])

    def test_failure(self):
        model = Model()
        with self.assertRaises(RuntimeError):
            save_file(model.state_dict(), "tmp.safetensors")

    # def test_workaround_refuse(self):
    #     model = Model()
    #     A = torch.zeros((1000, 10))
    #     a = A[:100, :]
    #     model.a.weight = torch.nn.Parameter(a)
    #     with self.assertRaises(RuntimeError) as ctx:
    #         save_model(model, "tmp4.safetensors")
    #     self.assertIn(".Refusing to save/load the model since you could be storing much more memory than needed.", str(ctx.exception))

    def test_save(self):
        # Just testing the actual saved file to make sure we're ok on big endian
        model = OnesModel()
        save_model(model, "tmp_ones.safetensors")
        with safe_open("tmp_ones.safetensors", framework="pt") as f:
            self.assertEqual(f.metadata(), {"b.bias": "a.bias", "b.weight": "a.weight"})

        # 192 hardcoded to skip the header, metadata order is random.
        self.assertEqual(
            open("tmp_ones.safetensors", "rb").read()[192:],
            b"""\x00\x00\x80?\x00\x00\x80?\x00\x00\x80?\x00\x00\x80?\x00\x00\x80?\x00\x00\x80?\x00\x00\x80?\x00\x00\x80?\x00\x00\x80?\x00\x00\x80?\x00\x00\x80?\x00\x00\x80?\x00\x00\x80?\x00\x00\x80?\x00\x00\x80?\x00\x00\x80?\x00\x00\x80?\x00\x00\x80?\x00\x00\x80?\x00\x00\x80?""",
        )

        model2 = OnesModel()
        load_model(model2, "tmp_ones.safetensors")

        state_dict = model.state_dict()
        for k, v in model2.state_dict().items():
            torch.testing.assert_close(v, state_dict[k])

    def test_workaround(self):
        model = Model()
        save_model(model, "tmp.safetensors")
        with safe_open("tmp.safetensors", framework="pt") as f:
            self.assertEqual(f.metadata(), {"b.bias": "a.bias", "b.weight": "a.weight"})

        model2 = Model()
        load_model(model2, "tmp.safetensors")

        state_dict = model.state_dict()
        for k, v in model2.state_dict().items():
            torch.testing.assert_close(v, state_dict[k])

    def test_workaround_works_with_different_on_file_names(self):
        model = Model()
        state_dict = model.state_dict()
        state_dict.pop("a.weight")
        state_dict.pop("a.bias")
        save_file(state_dict, "tmp.safetensors")

        model2 = Model()
        load_model(model2, "tmp.safetensors")

        state_dict = model.state_dict()
        for k, v in model2.state_dict().items():
            torch.testing.assert_close(v, state_dict[k])

    def test_workaround_non_contiguous(self):
        model = NonContiguousModel()

        with self.assertRaises(ValueError) as ctx:
            save_model(model, "tmp_c.safetensors", force_contiguous=False)
        self.assertIn("use save_model(..., force_contiguous=True)", str(ctx.exception))
        save_model(model, "tmp_c.safetensors", force_contiguous=True)

        model2 = NonContiguousModel()
        load_model(model2, "tmp_c.safetensors")

        state_dict = model.state_dict()
        for k, v in model2.state_dict().items():
            torch.testing.assert_close(v, state_dict[k])

    def test_workaround_copy(self):
        model = CopyModel()
        self.assertEqual(
            _find_shared_tensors(model.state_dict()),
            [{"a.weight"}, {"a.bias"}, {"b.weight"}, {"b.bias"}],
        )
        save_model(model, "tmp.safetensors")

        model2 = CopyModel()
        load_model(model2, "tmp.safetensors")

        state_dict = model.state_dict()
        for k, v in model2.state_dict().items():
            torch.testing.assert_close(v, state_dict[k])

    def test_difference_with_torch(self):
        model = Model()
        torch.save(model.state_dict(), "tmp2.bin")

        model2 = NoSharedModel()
        # This passes on torch.
        # The tensors are shared on disk, they are *not* shared within the model
        # The model happily loads the tensors, and ends up *not* sharing the tensors by.
        # doing copies
        self.assertEqual(
            _find_shared_tensors(model2.state_dict()),
            [{"a.weight"}, {"a.bias"}, {"b.weight"}, {"b.bias"}],
        )
        model2.load_state_dict(torch.load("tmp2.bin"))
        self.assertEqual(
            _find_shared_tensors(model2.state_dict()),
            [{"a.weight"}, {"a.bias"}, {"b.weight"}, {"b.bias"}],
        )

        # However safetensors cannot save those, so we cannot
        # reload the saved file with the different model
        save_model(model, "tmp2.safetensors")
        with self.assertRaises(RuntimeError) as ctx:
            load_model(model2, "tmp2.safetensors")
        self.assertIn(
            """Missing key(s) in state_dict: "b.bias", "b.weight""", str(ctx.exception)
        )

    def test_difference_torch_odd(self):
        model = NoSharedModel()
        a = model.a.weight
        b = model.b.weight
        self.assertNotEqual(a.data_ptr(), b.data_ptr())
        torch.save(model.state_dict(), "tmp3.bin")

        model2 = Model()
        self.assertEqual(
            _find_shared_tensors(model2.state_dict()),
            [{"a.weight", "b.weight"}, {"b.bias", "a.bias"}],
        )
        # Torch will affect either `b` or `a` to the shared tensor in the `model2`
        model2.load_state_dict(torch.load("tmp3.bin"))

        # XXX: model2 uses only the B weight not the A weight anymore.
        self.assertFalse(torch.allclose(model2.a.weight, model.a.weight))
        torch.testing.assert_close(model2.a.weight, model.b.weight)
        self.assertEqual(
            _find_shared_tensors(model2.state_dict()),
            [{"a.weight", "b.weight"}, {"b.bias", "a.bias"}],
        )

        # Everything is saved as-is
        save_model(model, "tmp3.safetensors")
        # safetensors will yell that there were 2 tensors on disk, while
        # the models expects only 1 tensor since both are shared.
        with self.assertRaises(RuntimeError) as ctx:
            load_model(model2, "tmp3.safetensors")
        # Safetensors properly warns the user that some ke
        self.assertIn(
            """Unexpected key(s) in state_dict: "b.bias", "b.weight""",
            str(ctx.exception),
        )


    # --- STAGE 2 TESTS (Safetensors #926) ---

    def test_tensors_overlap_1d_cases(self):
        raw = torch.arange(20, dtype=torch.float32)

        # 1. Interleaved 1D slices are disjoint
        even = raw[::2]
        odd = raw[1::2]
        self.assertFalse(_tensors_overlap(even, odd))

        # 2. Partially overlapping strided slices overlap (at indices 0, 6, 12, 18)
        p2 = raw[::2]
        p3 = raw[::3]
        self.assertTrue(_tensors_overlap(p2, p3))

        # 3. Identical views overlap
        self.assertTrue(_tensors_overlap(even, even))

        # 4. Disjoint contiguous blocks
        b1 = raw[:8]
        b2 = raw[8:16]
        self.assertFalse(_tensors_overlap(b1, b2))

        # 5. Overlapping contiguous blocks
        b3 = raw[4:12]
        self.assertTrue(_tensors_overlap(b1, b3))

        # 6. Expanded zero-stride disjoint vs overlapping
        exp = raw[0:1].expand(8)  # addresses element 0 only
        slice_disjoint = raw[2:6]
        slice_overlap = raw[0:4]
        self.assertFalse(_tensors_overlap(exp, slice_disjoint))
        self.assertTrue(_tensors_overlap(exp, slice_overlap))

        # 7. Empty tensors never overlap
        empty = raw[:0]
        self.assertFalse(_tensors_overlap(empty, raw))
        self.assertFalse(_tensors_overlap(empty, empty))

        # 8. Nonzero storage offset
        off1 = raw[1:5]
        off2 = raw[5:9]
        self.assertFalse(_tensors_overlap(off1, off2))

        # 9. Large stride views
        large_stride1 = raw[0:2:10]  # indices 0, 10
        large_stride2 = raw[1:2:10]  # indices 1, 11
        self.assertFalse(_tensors_overlap(large_stride1, large_stride2))

    def test_tensors_overlap_multidimensional_cases(self):
        M = torch.arange(100, dtype=torch.float32).view(10, 10)

        # 1. 2D Interleaved rows are disjoint
        r_even = M[::2, :]
        r_odd = M[1::2, :]
        self.assertFalse(_tensors_overlap(r_even, r_odd))

        # 2. 2D Interleaved cols are disjoint
        c_even = M[:, ::2]
        c_odd = M[:, 1::2]
        self.assertFalse(_tensors_overlap(c_even, c_odd))

        # 3. Overlapping 2D submatrices
        s1 = M[:6, :6]
        s2 = M[4:8, 4:8]
        self.assertTrue(_tensors_overlap(s1, s2))

        # 4. Disjoint 2D submatrices
        s_disjoint = M[6:10, 6:10]
        self.assertFalse(_tensors_overlap(s1, s_disjoint))

        # 5. Diagonal vs Anti-diagonal on even dimension
        d1 = M.diagonal()
        d2 = M.flip(0).diagonal()
        self.assertFalse(_tensors_overlap(d1, d2))

        # 6. Singleton dimension preserving layout
        s_dim1 = M[::2, :].unsqueeze(1)
        s_dim2 = M[1::2, :].unsqueeze(1)
        self.assertFalse(_tensors_overlap(s_dim1, s_dim2))

    def test_shared_storage_grouping_interleaved_and_transitive(self):
        raw = torch.arange(30, dtype=torch.float32)

        # 1. Interleaved 1D slices separate into singletons
        sd_1d = {"even": raw[::2], "odd": raw[1::2]}
        self.assertEqual(_find_shared_tensors(sd_1d), [{"even"}, {"odd"}])

        # 2. Interleaved 2D rows separate into singletons
        M = raw[:25].view(5, 5)
        sd_2d = {"r_even": M[::2, :], "r_odd": M[1::2, :]}
        self.assertEqual(_find_shared_tensors(sd_2d), [{"r_even"}, {"r_odd"}])

        # 3. Partially overlapping strided views remain grouped
        sd_part = {"p2": raw[::2], "p3": raw[::3]}
        self.assertEqual(_find_shared_tensors(sd_part), [{"p2", "p3"}])

        # 4. Transitive overlap: A overlaps B, B overlaps C, A does not overlap C
        # A=[0:10], B=[5:15], C=[10:20], D=[25:30]
        sd_trans = {
            "A": raw[0:10],
            "B": raw[5:15],
            "C": raw[10:20],
            "D": raw[25:30],
        }
        res_trans = _find_shared_tensors(sd_trans)
        self.assertEqual(res_trans, [{"A", "B", "C"}, {"D"}])

    def test_save_load_interleaved_1d_model_round_trip(self):
        class Interleaved1DModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                base = torch.arange(20, dtype=torch.float32)
                self.even = torch.nn.Parameter(base[::2])
                self.odd = torch.nn.Parameter(base[1::2])

        model = Interleaved1DModel()
        with tempfile.TemporaryDirectory() as tmp:
            filename = os.path.join(tmp, "model_1d_interleaved.safetensors")
            save_model(model, filename)

            model2 = Interleaved1DModel()
            model2.even.data.zero_()
            model2.odd.data.zero_()
            load_model(model2, filename)

            torch.testing.assert_close(model2.even, model.even)
            torch.testing.assert_close(model2.odd, model.odd)

    def test_save_load_interleaved_2d_model_round_trip(self):
        class Interleaved2DModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                M = torch.arange(100, dtype=torch.float32).view(10, 10)
                self.r_even = torch.nn.Parameter(M[::2, :])
                self.r_odd = torch.nn.Parameter(M[1::2, :])

        model = Interleaved2DModel()
        with tempfile.TemporaryDirectory() as tmp:
            filename = os.path.join(tmp, "model_2d_interleaved.safetensors")
            save_model(model, filename)

            model2 = Interleaved2DModel()
            model2.r_even.data.zero_()
            model2.r_odd.data.zero_()
            load_model(model2, filename)

            torch.testing.assert_close(model2.r_even, model.r_even)
            torch.testing.assert_close(model2.r_odd, model.r_odd)

    def test_save_model_rejects_overlapping_views_without_base(self):
        class OverlappingModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                base = torch.arange(20, dtype=torch.float32)
                self.p2 = torch.nn.Parameter(base[::2])
                self.p3 = torch.nn.Parameter(base[::3])

        model = OverlappingModel()
        with tempfile.TemporaryDirectory() as tmp:
            filename = os.path.join(tmp, "model_overlap.safetensors")
            with self.assertRaises(RuntimeError) as ctx:
                save_model(model, filename)
            self.assertIn("None is covering the entire storage", str(ctx.exception))

    def test_complete_base_with_interleaved_views_deduplication(self):
        class BaseWithInterleavedModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                raw = torch.arange(20, dtype=torch.float32)
                self.base = torch.nn.Parameter(raw)
                self.even = torch.nn.Parameter(raw[::2])
                self.odd = torch.nn.Parameter(raw[1::2])

        model = BaseWithInterleavedModel()
        with tempfile.TemporaryDirectory() as tmp:
            filename = os.path.join(tmp, "model_base_interleaved.safetensors")
            save_model(model, filename)

            with safe_open(filename, framework="pt") as f:
                keys = set(f.keys())
                self.assertEqual(keys, {"base"})
                metadata = f.metadata()
                self.assertEqual(metadata.get("even"), "base")
                self.assertEqual(metadata.get("odd"), "base")
