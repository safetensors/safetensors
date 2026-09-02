import os
import stat
import tempfile
import unittest
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from safetensors import safe_open


class ErrorsTestCase(unittest.TestCase):
    @unittest.skipIf(
        os.name == "nt" or getattr(os, "geteuid", lambda: 0)() == 0,
        "POSIX file permissions require a non-root user",
    )
    def test_permission_denied(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.safetensors"
            save_file({"test": np.zeros(1, dtype=np.float32)}, path)
            original_mode = stat.S_IMODE(path.stat().st_mode)
            path.chmod(0)
            try:
                with (
                    self.assertRaises(PermissionError),
                    safe_open(path, framework="np"),
                ):
                    pass
            finally:
                path.chmod(original_mode)
