import json
import tempfile
import unittest
from pathlib import Path

import torch

from examples.simBenchmarks.LIBERO.eval_files.fastwam_parity.stats import (
    normalize_minmax,
    unnormalize_minmax,
)
from examples.simBenchmarks.LIBERO.eval_files.fastwam_parity.tensor_compare import (
    compare_tensors,
    save_tensor_record,
    validate_manifest,
    write_manifest,
)


class FastWAMParityHelperTest(unittest.TestCase):
    def test_compare_tensors_reports_error_and_first_mismatch(self):
        official = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        starvla = torch.tensor([[1.0, 2.0], [3.5, 4.0]])

        result = compare_tensors(official, starvla)

        self.assertFalse(result["torch_equal"])
        self.assertEqual(result["first_mismatch"]["index"], [1, 0])
        self.assertEqual(result["first_mismatch"]["official"], 3.0)
        self.assertEqual(result["first_mismatch"]["starvla"], 3.5)
        self.assertEqual(result["max_abs_error"], 0.5)
        self.assertGreater(result["relative_l2_error"], 0.0)

    def test_dump_record_and_manifest_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            record = save_tensor_record("stage/a", torch.arange(4), tmp_path)
            manifest = write_manifest([record], tmp_path / "manifest.json", {"kind": "unit"})

            validate_manifest(manifest)
            loaded = json.loads((tmp_path / "manifest.json").read_text())
            validate_manifest(loaded)
            self.assertEqual(loaded["records"][0]["stage"], "stage/a")
            self.assertTrue(Path(loaded["records"][0]["path"]).is_file())

    def test_manifest_validation_rejects_missing_tensor_keys(self):
        with self.assertRaisesRegex(ValueError, "missing keys"):
            validate_manifest({"schema_version": 1, "records": [{"stage": "x"}]})

    def test_minmax_normalization_round_trip(self):
        stats = {
            "action": {
                "default": {
                    "global_min": [-2.0, 0.0],
                    "global_max": [2.0, 4.0],
                }
            }
        }
        value = torch.tensor([[0.0, 2.0], [2.0, 4.0]])

        normalized = normalize_minmax(value, stats, "action")
        restored = unnormalize_minmax(normalized, stats, "action")

        self.assertTrue(torch.allclose(restored, value))


if __name__ == "__main__":
    unittest.main()

