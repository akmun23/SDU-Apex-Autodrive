"""Focused tests for reset-safe fixed-step dataset assembly."""

import unittest
from types import SimpleNamespace

import numpy as np

from tools.vehicle_dynamics_learning.prepare_dataset import (
    _coalesce_contiguous_sequences,
    _quality,
    _split_for_name,
)
from tools import evaluate_open_plane_body_dynamics as body


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


class RunSplitTests(unittest.TestCase):
    def test_swerve_validation_runs_cannot_default_to_training(self):
        self.assertEqual(
            _split_for_name("openplane_swerve_throttle_slew_validation_r01_20261006"),
            "validation")
        self.assertEqual(
            _split_for_name("openplane_swerve_throttle_slew_train_r01_20261006"),
            "train")


class FixedPacketQualityTests(unittest.TestCase):
    @staticmethod
    def _capture(packet_ids: list[int], total_samples: int | None = None):
        stats = {
            topic: (39.95, 25.9, 174.0 if topic not in body.COMMAND_STREAM_TOPICS else 31.0)
            for topic in body.STREAM_TOPICS
        }
        return SimpleNamespace(
            aborted=False,
            collision_count_start=0,
            collision_count_end=0,
            timing_faults=0,
            packet_sequence_matched_samples=len(packet_ids),
            packet_sequence_total_samples=(
                len(packet_ids) if total_samples is None else total_samples),
            phase_stream_stats=stats,
            stream_stats=stats,
            sequences=[[SimpleNamespace(packet_sequence=value)
                        for value in packet_ids]],
        )

    def test_receipt_jitter_is_accepted_only_with_explicit_contiguous_packet_clock(self):
        capture = self._capture(list(range(100, 110)))

        self.assertFalse(_quality(capture)[0])
        self.assertEqual(_quality(capture, fixed_packet_timebase=True), (True, []))

    def test_fixed_packet_clock_does_not_hide_packet_loss(self):
        capture = self._capture([100, 101, 103, 104])

        accepted, failures = _quality(capture, fixed_packet_timebase=True)

        self.assertFalse(accepted)
        self.assertIn("simulator_packet_sequence_not_contiguous", failures)

    def test_unmatched_edge_samples_do_not_invent_an_internal_time_gap(self):
        capture = self._capture(list(range(100, 110)), total_samples=12)

        accepted, failures = _quality(capture, fixed_packet_timebase=True)

        self.assertTrue(accepted)
        self.assertEqual(failures, [])

    def test_command_receive_gap_gate_remains_active(self):
        capture = self._capture(list(range(100, 110)))
        topic = next(iter(body.COMMAND_STREAM_TOPICS))
        capture.phase_stream_stats[topic] = (39.95, 25.9, 121.0)

        accepted, failures = _quality(capture, fixed_packet_timebase=True)

        self.assertFalse(accepted)
        self.assertIn(f"stream_quality:{topic}", failures)


if __name__ == "__main__":
    unittest.main()
