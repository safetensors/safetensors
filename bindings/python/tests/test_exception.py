import pickle
import unittest

from safetensors import SafetensorError


class SafetensorErrorTestCase(unittest.TestCase):
    def test_module_matches_installed_extension(self):
        # SafetensorError must be discoverable by pickle, which looks the class
        # up via `import_module(cls.__module__)`. The Rust extension is
        # installed at `safetensors._safetensors_rust`, so __module__ must match.
        self.assertEqual(SafetensorError.__module__, "safetensors._safetensors_rust")

    def test_is_picklable_roundtrip(self):
        original = SafetensorError("boom")
        data = pickle.dumps(original)
        restored = pickle.loads(data)
        self.assertIsInstance(restored, SafetensorError)
        self.assertEqual(restored.args, ("boom",))


if __name__ == "__main__":
    unittest.main()
