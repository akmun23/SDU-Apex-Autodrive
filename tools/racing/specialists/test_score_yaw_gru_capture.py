from __future__ import annotations

import unittest

import numpy as np

from tools.racing.specialists.score_yaw_gru_capture import (
    _json_safe,
    _validate_report_for_score,
)


def _report(status: str = "running") -> dict[str, object]:
    return {
        "status": status,
        "best_epoch": 70,
        "training_runs": ["train-r01"],
        "validation_runs": ["selection-r02"],
    }


class CheckpointReportAdmissionTest(unittest.TestCase):
    def test_running_checkpoint_requires_explicit_provisional_opt_in(self) -> None:
        with self.assertRaisesRegex(ValueError, "allow-running-checkpoint"):
            _validate_report_for_score(_report(), "unseen-r03", False)

        self.assertTrue(_validate_report_for_score(
            _report(), "unseen-r03", True))

    def test_complete_checkpoint_is_not_provisional(self) -> None:
        self.assertFalse(_validate_report_for_score(
            _report("complete"), "unseen-r03", False))

    def test_training_and_checkpoint_selection_runs_are_never_scored(self) -> None:
        for run_id in ("train-r01", "selection-r02"):
            with self.subTest(run_id=run_id):
                with self.assertRaisesRegex(ValueError, "checkpoint-selection run"):
                    _validate_report_for_score(
                        _report("complete"), run_id, True)

    def test_running_checkpoint_without_saved_best_epoch_is_rejected(self) -> None:
        report = _report()
        report["best_epoch"] = None
        with self.assertRaisesRegex(ValueError, "allow-running-checkpoint"):
            _validate_report_for_score(report, "unseen-r03", True)

    def test_numpy_scalars_and_arrays_are_json_safe(self) -> None:
        self.assertEqual(_json_safe({"n": np.int64(7), "x": np.float32(0.5),
                                     "values": np.asarray([1, 2])}),
                         {"n": 7, "x": 0.5, "values": [1, 2]})

if __name__ == "__main__":
    unittest.main()
