"""Round-trip + determinism tests for the Llama-shaped fixture generator.

These tests guard the structural-shape coverage gap: the rest of the suite
exercises empty headers and very small synthetic tensors, this file exercises
the 201-tensor Llama-3.2 layout end-to-end via the canonical reader. The
fixture machinery itself is also under test here so that downstream tests can
take the seeded determinism as a hard guarantee.
"""

import hashlib
import os
import tempfile
import unittest

import numpy as np

from safetensors import safe_open
from safetensors.numpy import load_file

from tests.fixtures.llama_generator import (
    DTYPE_BYTES,
    LlamaShape,
    build_entries,
    build_header,
    write_fixture,
)


def _read_header_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        prefix = f.read(8)
        n = int.from_bytes(prefix, "little", signed=False)
        return f.read(n)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class LlamaFixtureStructureTestCase(unittest.TestCase):
    def test_default_shape_produces_201_tensors(self):
        # The TinyLlama-1.1B / Llama-3.2-1B tensor inventory: 1 embedding
        # + 22 layers x 9 tensors/layer + 1 final norm + 1 LM head = 201.
        entries = build_entries(LlamaShape())
        self.assertEqual(len(entries), 201)

    def test_offsets_are_contiguous_and_non_overlapping(self):
        entries = build_entries(LlamaShape())
        prev_end = 0
        for e in entries:
            start, end = e.data_offsets
            self.assertEqual(start, prev_end)
            self.assertEqual(end - start, e.nbytes())
            self.assertGreater(end, start)
            prev_end = end

    def test_header_dtype_coverage_includes_real_model_set(self):
        # Even though the default uses F32 across the board, the generator
        # must accept the full Llama-relevant dtype set.
        for dt in ("F16", "BF16", "F32", "I32", "I64"):
            shape = LlamaShape(weight_dtype=dt, norm_dtype=dt)
            entries = build_entries(shape)
            for e in entries:
                self.assertEqual(e.dtype, dt)
                self.assertIn(e.dtype, DTYPE_BYTES)


class LlamaFixtureDeterminismTestCase(unittest.TestCase):
    def test_same_seed_produces_byte_identical_output(self):
        with tempfile.TemporaryDirectory() as d:
            a = os.path.join(d, "a.safetensors")
            b = os.path.join(d, "b.safetensors")
            # Use a smaller shape so the bytes are cheap to hash.
            shape = LlamaShape(
                vocab_size=1024,
                hidden_size=64,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_kv_heads=1,
                head_dim=32,
                intermediate_size=128,
            )
            write_fixture(a, shape=shape, seed=42)
            write_fixture(b, shape=shape, seed=42)
            self.assertEqual(_sha256(a), _sha256(b))

    def test_different_seed_changes_data_segment(self):
        with tempfile.TemporaryDirectory() as d:
            a = os.path.join(d, "a.safetensors")
            b = os.path.join(d, "b.safetensors")
            shape = LlamaShape(
                vocab_size=1024,
                hidden_size=64,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_kv_heads=1,
                head_dim=32,
                intermediate_size=128,
            )
            write_fixture(a, shape=shape, seed=1)
            write_fixture(b, shape=shape, seed=2)
            # The headers are identical because shapes match; the data
            # segments diverge.
            self.assertEqual(_read_header_bytes(a), _read_header_bytes(b))
            self.assertNotEqual(_sha256(a), _sha256(b))


class LlamaFixtureRoundTripTestCase(unittest.TestCase):
    def test_canonical_reader_accepts_full_201_tensor_fixture(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "llama.safetensors")
            # Default shape but with a much smaller vocab to keep the test
            # fast; structural coverage is what we want, not throughput.
            shape = LlamaShape(vocab_size=512, hidden_size=64)
            summary = write_fixture(path, shape=shape, seed=0)
            self.assertEqual(summary["n_tensors"], 201)

            loaded = load_file(path)
            self.assertEqual(len(loaded), 201)
            self.assertIn("model.embed_tokens.weight", loaded)
            self.assertIn("lm_head.weight", loaded)
            self.assertIn("model.layers.0.self_attn.q_proj.weight", loaded)
            self.assertIn("model.layers.21.post_attention_layernorm.weight", loaded)

            # Round-trip the embedding shape via the canonical reader.
            emb = loaded["model.embed_tokens.weight"]
            self.assertEqual(emb.shape, (512, 64))
            self.assertEqual(emb.dtype, np.float32)

    def test_canonical_reader_round_trips_data_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "llama.safetensors")
            shape = LlamaShape(
                vocab_size=256,
                hidden_size=32,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_kv_heads=1,
                head_dim=16,
                intermediate_size=64,
            )
            write_fixture(path, shape=shape, seed=7)

            header_bytes = _read_header_bytes(path)
            with open(path, "rb") as f:
                f.seek(8 + len(header_bytes))
                raw_tail = f.read()

            # Pull the embedding span from the raw tail and compare against
            # what the canonical reader exposes.
            entries = build_entries(shape)
            emb_entry = entries[0]
            self.assertEqual(emb_entry.name, "model.embed_tokens.weight")
            start, end = emb_entry.data_offsets
            raw_embed = raw_tail[start:end]

            loaded = load_file(path)
            read_embed = loaded["model.embed_tokens.weight"]
            self.assertEqual(read_embed.tobytes(), raw_embed)

    def test_metadata_round_trips_via_safe_open(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "llama.safetensors")
            shape = LlamaShape(
                vocab_size=128,
                hidden_size=32,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_kv_heads=1,
                head_dim=16,
                intermediate_size=64,
            )
            write_fixture(path, shape=shape, seed=0)

            with safe_open(path, framework="np") as f:
                meta = f.metadata()
            self.assertEqual(meta["model_type"], "llama")
            self.assertEqual(meta["num_hidden_layers"], "2")
            self.assertEqual(meta["num_kv_heads"], "1")


class LlamaFixtureHeaderJsonTestCase(unittest.TestCase):
    def test_header_json_is_pure_ascii_and_well_formed(self):
        shape = LlamaShape(
            vocab_size=128,
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_kv_heads=1,
            head_dim=16,
            intermediate_size=64,
        )
        entries = build_entries(shape)
        header_json, total_data = build_header(entries, shape)
        # ASCII keeps the fixture portable across editors / CI workspaces.
        header_json.decode("ascii")
        self.assertGreater(total_data, 0)
        self.assertTrue(header_json.startswith(b"{"))
        self.assertTrue(header_json.endswith(b"}"))


if __name__ == "__main__":
    unittest.main()
