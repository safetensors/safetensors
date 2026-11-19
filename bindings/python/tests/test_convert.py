import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import convert


class ConvertTestCase(TestCase):
    def test_convert_multi_propagates_revision(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            names = [
                "pytorch_model-00001-of-00002.bin",
                "pytorch_model-00002-of-00002.bin",
            ]
            for i, name in enumerate(names):
                torch.save({f"weight_{i}": torch.tensor([float(i)])}, root / name)
            index = "pytorch_model.bin.index.json"
            (root / index).write_text(
                json.dumps(
                    {
                        "weight_map": {
                            f"weight_{i}": name for i, name in enumerate(names)
                        }
                    }
                )
            )

            calls = []

            def download(*args, filename, **kwargs):
                calls.append((filename, kwargs.get("revision")))
                return str(root / filename)

            with patch.object(convert, "hf_hub_download", side_effect=download):
                operations, errors = convert.convert_multi(
                    "local/test-model",
                    revision="v1.2.3",
                    folder=tmp,
                    token=None,
                    discard_names=[],
                )

            self.assertEqual(errors, [])
            self.assertEqual(len(operations), 3)
            # Verify index download received the revision
            index_calls = [c for c in calls if c[0] == "pytorch_model.bin.index.json"]
            self.assertEqual(len(index_calls), 1)
            self.assertEqual(index_calls[0][1], "v1.2.3")

            # Verify every shard download received the revision
            shard_calls = [c for c in calls if c[0] in names]
            self.assertEqual(len(shard_calls), 2)
            for shard_filename, shard_rev in shard_calls:
                self.assertEqual(shard_rev, "v1.2.3")

    def test_convert_multi_default_revision_is_none(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            names = ["pytorch_model-00001-of-00001.bin"]
            torch.save({"weight_0": torch.tensor([0.0])}, root / names[0])
            index = "pytorch_model.bin.index.json"
            (root / index).write_text(
                json.dumps({"weight_map": {"weight_0": names[0]}})
            )

            calls = []

            def download(*args, filename, **kwargs):
                calls.append((filename, kwargs.get("revision")))
                return str(root / filename)

            with patch.object(convert, "hf_hub_download", side_effect=download):
                operations, errors = convert.convert_multi(
                    "local/test-model",
                    folder=tmp,
                    token=None,
                    discard_names=[],
                )

            self.assertEqual(errors, [])
            for filename, rev in calls:
                self.assertIsNone(rev)
