import os
import tempfile
import unittest

import numpy as np

from safetensors import safe_open
from safetensors.numpy import save_file


class FileOpenTestCase(unittest.TestCase):
    def test_missing_file(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            with safe_open("notafile", framework="np"):
                pass
        self.assertEqual(str(ctx.exception), "No such file or directory: notafile")

    def test_permission_denied(self):
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            self.skipTest("root bypasses file mode bits")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.safetensors")
            save_file({"w": np.array([1.0], dtype=np.float32)}, path)
            os.chmod(path, 0)
            with self.assertRaises(PermissionError) as ctx:
                with safe_open(path, framework="np"):
                    pass
        self.assertIn(path, str(ctx.exception))
