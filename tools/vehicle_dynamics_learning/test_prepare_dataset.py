"""Focused tests for reset-safe fixed-step dataset assembly."""

import unittest

import numpy as np

from tools.vehicle_dynamics_learning.prepare_dataset import (
    _coalesce_contiguous_sequences,
)


def _sequence(label: str, packet_ids: list[int], receipts_ns: list[int]) -> tuple:
    count = len(packet_ids)
    return (
        label,
        np.zeros((count, 9), dtype=np.float32),
        np.zeros((count, 10), dtype=np.float32),
        np.ones(count, dtype=bool),
        np.zeros((count, 4), dtype=np.float32),
        np.ones(count, dtype=bool),
        np.full(count, 0.025, dtype=np.float32),
        np.asarray(packet_ids, dtype=np.int64),
        np.asarray(receipts_ns, dtype=np.int64),
        np.zeros((count, 3), dtype=np.float32),
        np.zeros((count, 3), dtype=np.float32),
        np.zeros(count, dtype=np.int32),
        np.zeros((count, 13), dtype=np.float32),
        np.zeros((count, 3), dtype=np.float32),
    )


class ResetBoundaryTests(unittest.TestCase):
    def test_reset_edge_splits_consecutive_packet_ids(self):
        merged = _coalesce_contiguous_sequences(
            [
                _sequence("condition_a", [10, 11], [100, 125]),
                _sequence("condition_b", [12, 13], [200, 225]),
            ],
            reset_epoch_starts_ns=[150],
        )

        self.assertEqual([len(sequence[1]) for sequence in merged], [2, 2])
        self.assertEqual([sequence[0] for sequence in merged],
                         ["condition_a", "condition_b"])

    def test_contiguous_phases_join_without_a_reset(self):
        merged = _coalesce_contiguous_sequences(
            [
                _sequence("condition_a", [10, 11], [100, 125]),
                _sequence("condition_b", [12, 13], [150, 175]),
            ],
            reset_epoch_starts_ns=[],
        )

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0][0], "continuous_run_mixed_conditions")
        self.assertEqual(len(merged[0][1]), 4)

    def test_packet_gap_still_splits_without_reset_metadata(self):
        merged = _coalesce_contiguous_sequences(
            [
                _sequence("condition_a", [10, 11], [100, 125]),
                _sequence("condition_b", [13, 14], [150, 175]),
            ],
        )

        self.assertEqual([len(sequence[1]) for sequence in merged], [2, 2])


if __name__ == "__main__":
    unittest.main()
