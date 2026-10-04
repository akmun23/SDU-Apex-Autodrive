from __future__ import annotations

import unittest

import numpy as np

from tools.vehicle_dynamics_learning.diagnose_effective_race_residuals import (
    _command_rows,
    _practice_windows,
    _source_roll_states,
)


class CommandAlignmentTests(unittest.TestCase):
    def test_preceding_and_target_row_alignments(self) -> None:
        sources = np.asarray([[10, 11], [20, 21]], dtype=np.int64)
        np.testing.assert_array_equal(
            _command_rows(sources, -1), [[10, 11], [20, 21]])
        np.testing.assert_array_equal(
            _command_rows(sources, 0), [[11, 12], [21, 22]])

    def test_rejects_unsupported_offsets(self) -> None:
        with self.assertRaises(ValueError):
            _command_rows(np.asarray([0]), 1)

    def test_internal_roll_is_aligned_to_transition_source(self) -> None:
        initial = np.asarray([[0.1, -0.2]])
        future = np.asarray([[[0.2, -0.1], [0.3, 0.0], [0.4, 0.1]]])
        np.testing.assert_array_equal(
            _source_roll_states(initial, future),
            [[[0.1, -0.2], [0.2, -0.1], [0.3, 0.0]]])

    def test_practice_window_can_cross_verified_packet_boundary(self) -> None:
        data = {
            "run_ids": np.asarray(["practice_r01"]),
            "splits": np.asarray(["unseen_practice"]),
            "seq_run": np.asarray([0, 0]),
            "bounds": np.asarray([[0, 40], [40, 100]]),
            "sequence_reset_index": np.asarray([3, 3]),
            "packet_sequence": np.arange(100),
            "dt_s": np.full(100, 0.025),
        }
        benchmark = {"windows": [{
            "run_id": "practice_r01",
            "global_start_index": 80,
            "categories": ["turn"],
        }]}
        rows = _practice_windows(benchmark, data, horizon_steps=10)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["start"], 80)

    def test_practice_window_rejects_unverified_packet_boundary(self) -> None:
        data = {
            "run_ids": np.asarray(["practice_r01"]),
            "splits": np.asarray(["unseen_practice"]),
            "seq_run": np.asarray([0, 0]),
            "bounds": np.asarray([[0, 40], [40, 100]]),
            "sequence_reset_index": np.asarray([3, 3]),
            "packet_sequence": np.concatenate((np.arange(40), np.arange(41, 101))),
            "dt_s": np.full(100, 0.025),
        }
        benchmark = {"windows": [{
            "run_id": "practice_r01",
            "global_start_index": 80,
            "categories": ["turn"],
        }]}
        self.assertEqual(_practice_windows(benchmark, data, 10), [])


if __name__ == "__main__":
    unittest.main()
