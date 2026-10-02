from __future__ import annotations

import unittest

import numpy as np

from tools.vehicle_dynamics_learning.score_rssm_free_running_capture import (
    EXPECTED_PHASES,
    _errors,
    _group_expected_phase_fragments,
    _merge_contiguous_sequences,
    _pose_errors,
    _scoring_fragments,
    _time_bin_summary,
    _wrap_angle,
)


class RssmFreeRunScoreTest(unittest.TestCase):
    def test_heading_errors_are_wrapped_across_pi(self) -> None:
        error = _wrap_angle(np.asarray([np.pi + 0.1]))
        np.testing.assert_allclose(error, [-np.pi + 0.1], atol=1e-12)

        position, heading = _pose_errors(
            np.asarray([[1.0, 2.0, -np.pi + 0.1]]),
            np.asarray([[1.0, 2.0, np.pi - 0.1]]))
        np.testing.assert_allclose(position, [[0.0, 0.0]])
        np.testing.assert_allclose(heading, [0.2], atol=1e-12)

    def test_error_metrics_are_physical_rmse_bias_and_absolute_p95(self) -> None:
        result = _errors(np.asarray([[1.0], [3.0]]), np.asarray([[0.0], [1.0]]))
        self.assertAlmostEqual(result["rmse"][0], np.sqrt(2.5))
        self.assertAlmostEqual(result["bias"][0], 1.5)
        self.assertAlmostEqual(result["absolute_p95"][0], 1.95)

    def test_time_bins_cover_full_rollout_and_keep_final_partial_bin(self) -> None:
        state_error = np.zeros((5, 7), dtype=np.float64)
        state_error[:, 0] = np.arange(1.0, 6.0)
        position_error = np.column_stack((state_error[:, 0],
                                          np.zeros(5)))
        heading_error = state_error[:, 0] * 0.1
        speed_error = state_error[:, 0] * 0.2

        result = _time_bin_summary(
            state_error, position_error, heading_error, speed_error,
            bin_width_s=0.05)

        self.assertEqual(len(result["bins"]), 3)
        self.assertEqual([row["sample_count"] for row in result["bins"]],
                         [2, 2, 1])
        self.assertAlmostEqual(result["bins"][0]["elapsed_start_s"], 0.0)
        self.assertAlmostEqual(result["bins"][0]["elapsed_end_s"], 0.05)
        self.assertAlmostEqual(result["bins"][-1]["elapsed_end_s"], 0.125)
        self.assertAlmostEqual(
            result["peak_absolute_error_by_state"]["u_rear_mps"]["elapsed_s"],
            0.125)

    def test_only_packet_contiguous_same_condition_fragments_are_rejoined(self) -> None:
        data = {
            "bounds": np.asarray([[0, 3], [3, 6], [6, 9]], dtype=np.int64),
            "seq_run": np.asarray([0, 0, 0], dtype=np.int32),
            "sequence_condition_id": np.asarray([0, 0, 0], dtype=np.int32),
            "condition_labels": np.asarray(["same"]),
            "packet_sequence": np.asarray([10, 11, 12, 13, 14, 15,
                                            20, 21, 22], dtype=np.int64),
        }

        groups = _merge_contiguous_sequences(data, 0)

        self.assertEqual(groups, [(0, 6, "same"), (6, 9, "same")])

    def test_missing_packet_keeps_two_fragments_for_one_scheduled_probe(self) -> None:
        label = sorted(EXPECTED_PHASES)[0]
        groups = [(index * 10, index * 10 + 8, phase)
                  for index, phase in enumerate(sorted(EXPECTED_PHASES))]
        groups.append((1000, 1008, label))

        fragments = _group_expected_phase_fragments(groups)

        self.assertEqual(set(fragments), EXPECTED_PHASES)
        self.assertEqual(len(fragments[label]), 2)

    def test_continuous_run_scoring_preserves_each_packet_gap_segment(self) -> None:
        groups = [(10, 30, "mixed_run"), (40, 60, "mixed_run")]
        self.assertEqual(
            _scoring_fragments(groups, continuous_run=True),
            [("mixed_run", 0, 10, 30), ("mixed_run", 1, 40, 60)])


if __name__ == "__main__":
    unittest.main()
