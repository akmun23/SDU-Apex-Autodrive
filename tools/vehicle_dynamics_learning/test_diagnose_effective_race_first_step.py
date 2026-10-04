from __future__ import annotations

import unittest

import numpy as np

from tools.vehicle_dynamics_learning.diagnose_effective_race_first_step import (
    _command_rows,
    _summarize,
    _transition_indices,
)


class FirstStepDiagnosticTests(unittest.TestCase):
    def test_command_offset_maps_target_row_to_tested_alignment(self):
        source_rows = np.asarray((100, 101))
        np.testing.assert_array_equal(
            _command_rows(source_rows, -1), np.asarray((100, 101)))
        np.testing.assert_array_equal(
            _command_rows(source_rows, 0), np.asarray((101, 102)))

    def test_command_offset_rejects_unsupported_alignment(self):
        with self.assertRaises(ValueError):
            _command_rows(np.asarray((100,)), 1)

    def test_transition_rows_stay_inside_selected_sequence_bounds(self):
        data = {
            "splits": np.asarray(("validation", "train")),
            "bounds": np.asarray(((0, 84), (84, 170))),
            "seq_run": np.asarray((0, 1)),
        }
        rows, runs = _transition_indices(data, {"validation"})
        np.testing.assert_array_equal(rows, np.arange(79, 83))
        np.testing.assert_array_equal(runs, np.zeros(4, dtype=np.int32))

    def test_macro_rmse_weights_whole_runs_not_frame_counts(self):
        values = np.asarray((1.0, 1.0, 1.0, 3.0))
        run_index = np.asarray((0, 0, 0, 1))
        summary = _summarize(
            values, run_index, np.asarray(("run-a", "run-b")),
            np.ones(4, dtype=bool))
        self.assertEqual(summary["transition_count"], 4)
        self.assertEqual(summary["independent_whole_runs"], 2)
        self.assertAlmostEqual(summary["macro_run_rmse"], 2.0)
        self.assertAlmostEqual(
            summary["macro_run_bias_prediction_minus_truth"], 2.0)

    def test_validity_mask_applies_only_to_selected_metric_support(self):
        values = np.asarray((1.0, 3.0, 5.0))
        run_index = np.asarray((0, 0, 1))
        summary = _summarize(
            values, run_index, np.asarray(("run-a", "run-b")),
            np.ones(3, dtype=bool), np.asarray((False, True, False)))
        self.assertEqual(summary["transition_count"], 1)
        self.assertEqual(summary["independent_whole_runs"], 1)
        self.assertEqual(summary["macro_run_rmse"], 3.0)


if __name__ == "__main__":
    unittest.main()
