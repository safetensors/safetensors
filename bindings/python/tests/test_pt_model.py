import copy
import unittest

import torch

from safetensors import safe_open
from safetensors.torch import (
    _end_ptr,
    _find_shared_tensors,
    _is_complete,
    _remove_duplicate_names,
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

    def test_save_does_not_mutate_metadata(self):
        model = OnesModel()
        metadata = {"format": "pt"}

        save_model(model, "tmp_ones.safetensors", metadata=metadata)

        self.assertEqual(metadata, {"format": "pt"})
        with safe_open("tmp_ones.safetensors", framework="pt") as f:
            self.assertEqual(
                f.metadata(),
                {
                    "format": "pt",
                    "b.bias": "a.bias",
                    "b.weight": "a.weight",
                },
            )

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
