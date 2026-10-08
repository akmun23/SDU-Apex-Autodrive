from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from score_yaw_teacher_final_holdout import _admit_final_runs


class FinalHoldoutAdmissionTest(unittest.TestCase):
    def _dataset(self, split: str) -> Path:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        (root / "openplane_dynamics.npz").touch()
        row = {
            "run_id": "holdout_r01",
            "effective_split": split,
            "aborted": False,
            "reason": "schedule complete",
            "clean_stream_and_collision_gate": True,
            "timing_faults": 0,
            "collisions": [0],
            "quality_failures": [],
            "whole_bag_quality_failures": [],
        }
        (root / "manifest.json").write_text(
            json.dumps({"runs": [row]}), encoding="utf-8")
        return root

    def test_only_explicit_clean_final_test_runs_are_admitted(self) -> None:
        captures = _admit_final_runs(self._dataset("final_test"),
                                     ["holdout_r01"])
        self.assertEqual(len(captures), 1)
        self.assertEqual(captures[0].split, "final_test")

    def test_validation_run_is_not_opened_by_final_scorer(self) -> None:
        with self.assertRaisesRegex(ValueError, "final_test runs only"):
            _admit_final_runs(self._dataset("validation"), ["holdout_r01"])


if __name__ == "__main__":
    unittest.main()
