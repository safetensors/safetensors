"""
Regression tests for Null-Byte Injection Vulnerability
=======================================================
Issue: #748 / Huntr 8317f258-7731-4e13-8aa7-ae2d2630c155

An attacker can embed a hidden tensor whose name contains a null byte
(e.g. "weights\\x00.hidden").  Python's json library and Rust's serde_json
both treat \\x00 as valid string content, but C-string-based security
scanners truncate at \\x00 – making the hidden portion invisible to them
while it still executes at the Python / Rust runtime level.

The same attack can be mounted via the __metadata__ dictionary: even if
tensor names are validated, embedding \\x00 in a metadata key or value
can confuse downstream C-string consumers that inspect that field.

The Rust fix in safetensors/src/tensor.rs now rejects any tensor name
containing \\x00 in both the serialisation (write) and deserialisation
(read) paths, returning an InvalidTensorName error.  __metadata__
keys/values are similarly rejected with InvalidMetadata.

These tests verify that the rejection is correctly surfaced to Python
callers as a SafetensorError.
"""

import json
import struct
import tempfile
import unittest

import numpy as np

from safetensors import SafetensorError
from safetensors.numpy import load, save, save_file


class NullByteInTensorNameTestCase(unittest.TestCase):
    # ------------------------------------------------------------------
    # Serialisation path (write) – save() / save_file()
    # ------------------------------------------------------------------

    def test_save_rejects_null_byte_in_tensor_name(self):
        """save() must raise SafetensorError for names containing \\x00."""
        data = np.array([1, 2, 3], dtype=np.float32)
        with self.assertRaises(SafetensorError) as ctx:
            save({"test\x00_tensor": data})
        err = str(ctx.exception).lower()
        self.assertTrue("null byte" in err, f"Unexpected message: {ctx.exception}")

    def test_save_file_rejects_null_byte_in_tensor_name(self):
        """save_file() must raise SafetensorError for names containing \\x00."""
        data = np.array([1.0, 2.0], dtype=np.float32)
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            tmp_path = f.name
        with self.assertRaises(SafetensorError):
            save_file({"weights\x00.hidden": data}, tmp_path)

    def test_save_rejects_null_byte_only_name(self):
        data = np.zeros((2,), dtype=np.int32)
        with self.assertRaises(SafetensorError):
            save({"\x00": data})

    def test_save_rejects_null_byte_at_start(self):
        data = np.zeros((2,), dtype=np.int32)
        with self.assertRaises(SafetensorError):
            save({"\x00hidden_tensor": data})

    def test_save_rejects_null_byte_at_end(self):
        data = np.zeros((2,), dtype=np.int32)
        with self.assertRaises(SafetensorError):
            save({"tensor\x00": data})

    def test_save_accepts_normal_tensor_name(self):
        """Sanity-check: ordinary names must continue to work."""
        data = np.array([1, 2, 3], dtype=np.float32)
        try:
            result = save({"normal_tensor": data})
        except SafetensorError as e:
            self.fail(f"save() raised SafetensorError unexpectedly: {e}")
        self.assertIsInstance(result, bytes)

    def test_multiple_tensors_one_bad_name_is_rejected(self):
        data = np.zeros((2,), dtype=np.float32)
        with self.assertRaises(SafetensorError):
            save({"good_tensor": data, "bad\x00tensor": data})

    # ------------------------------------------------------------------
    # Deserialisation path (read) – load()
    # ------------------------------------------------------------------

    def _craft_safetensors_with_null_tensor_name(self) -> bytes:
        """Craft a raw safetensors buffer with a null byte inside a tensor name."""
        tensor_name = "weights\x00hidden"
        header_dict = {
            tensor_name: {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}
        }
        header_json = json.dumps(header_dict, separators=(",", ":")).encode("utf-8")
        remainder = len(header_json) % 8
        if remainder:
            header_json += b" " * (8 - remainder)
        return struct.pack("<Q", len(header_json)) + header_json + b"\x00\x00\x00\x00"

    def test_load_rejects_null_byte_in_tensor_name(self):
        crafted = self._craft_safetensors_with_null_tensor_name()
        with self.assertRaises(SafetensorError) as ctx:
            load(crafted)
        err = str(ctx.exception).lower()
        self.assertTrue(
            "null byte" in err or "invalid" in err,
            f"Unexpected error: {ctx.exception}",
        )


class NullByteInMetadataTestCase(unittest.TestCase):
    """
    Phase 2 addition: verify __metadata__ key/value null-byte bypass is blocked.
    """

    # ------------------------------------------------------------------
    # Serialisation path – metadata with null byte in KEY
    # ------------------------------------------------------------------

    def test_save_file_rejects_null_byte_in_metadata_key(self):
        """save_file() must raise when a __metadata__ key contains \\x00."""
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            tmp_path = f.name
        with self.assertRaises(SafetensorError) as ctx:
            save_file({"a": data}, tmp_path, metadata={"frame\x00work": "pt"})
        err = str(ctx.exception).lower()
        self.assertTrue(
            "null byte" in err or "invalid" in err,
            f"Unexpected message: {ctx.exception}",
        )

    def test_save_file_rejects_null_byte_in_metadata_value(self):
        """save_file() must raise when a __metadata__ value contains \\x00."""
        data = np.array([1.0], dtype=np.float32)
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            tmp_path = f.name
        with self.assertRaises(SafetensorError) as ctx:
            save_file({"a": data}, tmp_path, metadata={"framework": "pt\x00injected"})
        err = str(ctx.exception).lower()
        self.assertTrue(
            "null byte" in err or "invalid" in err,
            f"Unexpected message: {ctx.exception}",
        )

    def test_save_accepts_clean_metadata(self):
        """Sanity-check: clean metadata must continue to work."""
        data = np.array([1.0], dtype=np.float32)
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            tmp_path = f.name
        try:
            save_file({"a": data}, tmp_path, metadata={"framework": "pt"})
        except SafetensorError as e:
            self.fail(f"save_file() raised unexpectedly with clean metadata: {e}")

    # ------------------------------------------------------------------
    # Deserialisation path – crafted buffer with null in __metadata__
    # ------------------------------------------------------------------

    def _craft_safetensors_with_null_metadata_key(self) -> bytes:
        """
        Build a raw safetensors buffer whose __metadata__ dict contains a
        null byte in a key.  Python json serialises \x00 as \\u0000 in JSON,
        which serde_json accepts, so only the explicit Rust guard blocks this.
        """
        header_dict = {
            "__metadata__": {"frame\x00work": "pt"},
            "a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
        }
        header_json = json.dumps(header_dict, separators=(",", ":")).encode("utf-8")
        remainder = len(header_json) % 8
        if remainder:
            header_json += b" " * (8 - remainder)
        return struct.pack("<Q", len(header_json)) + header_json + b"\x00\x00\x00\x00"

    def _craft_safetensors_with_null_metadata_value(self) -> bytes:
        header_dict = {
            "__metadata__": {"framework": "pt\x00injected"},
            "a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
        }
        header_json = json.dumps(header_dict, separators=(",", ":")).encode("utf-8")
        remainder = len(header_json) % 8
        if remainder:
            header_json += b" " * (8 - remainder)
        return struct.pack("<Q", len(header_json)) + header_json + b"\x00\x00\x00\x00"

    def test_load_rejects_null_byte_in_metadata_key(self):
        """load() must raise for a crafted buffer with \\x00 in a metadata key."""
        crafted = self._craft_safetensors_with_null_metadata_key()
        with self.assertRaises(SafetensorError) as ctx:
            load(crafted)
        err = str(ctx.exception).lower()
        self.assertTrue(
            "null byte" in err or "invalid" in err,
            f"Unexpected error: {ctx.exception}",
        )

    def test_load_rejects_null_byte_in_metadata_value(self):
        """load() must raise for a crafted buffer with \\x00 in a metadata value."""
        crafted = self._craft_safetensors_with_null_metadata_value()
        with self.assertRaises(SafetensorError) as ctx:
            load(crafted)
        err = str(ctx.exception).lower()
        self.assertTrue(
            "null byte" in err or "invalid" in err,
            f"Unexpected error: {ctx.exception}",
        )

    def test_load_does_not_panic_on_null_metadata(self):
        """
        The library must raise a structured error, never panic/abort, when
        encountering null bytes in metadata.
        """
        crafted = self._craft_safetensors_with_null_metadata_key()
        try:
            load(crafted)
            self.fail("Expected SafetensorError but got nothing")
        except SafetensorError:
            pass  # correct – structured error, no panic
        except Exception as e:
            self.fail(f"Expected SafetensorError but got {type(e).__name__}: {e}")


if __name__ == "__main__":
    unittest.main()
