from __future__ import annotations

import unittest

import numpy as np

from tools.race_domain_dynamic_coupled_plan import build_dynamic_coupled_plan
from tools.vehicle_dynamics_learning.build_replacement_teacher_dataset import (
    _condition_labels,
    _frame_source_sequence_index,
    _validate_streams,
)


class ReplacementTeacherDatasetTest(unittest.TestCase):
    def test_source_index_never_crosses_or_skips_sequence_bounds(self):
        actual = _frame_source_sequence_index(
            np.asarray([[0, 3], [3, 5], [5, 8]], dtype=np.int64), 8)
        np.testing.assert_array_equal(actual, [0, 0, 0, 1, 1, 2, 2, 2])
        with self.assertRaisesRegex(ValueError, "contiguous"):
            _frame_source_sequence_index(
                np.asarray([[0, 3], [4, 8]], dtype=np.int64), 8)

    def test_condition_labels_follow_seeded_condition_and_reset_order(self):
        seed = 20261005
        resets = np.arange(1, 10, dtype=np.int32)
        actual = _condition_labels(seed, resets)
        expected = [f"{row.condition_id}/reset_epoch_{reset:02d}"
                    for row, reset in zip(build_dynamic_coupled_plan(seed), resets)]
        self.assertEqual(actual, expected)
        self.assertEqual(len(set(actual)), 9)
        with self.assertRaisesRegex(ValueError, "sequence/reset"):
            _condition_labels(seed, np.arange(0, 9, dtype=np.int32))

    def test_stream_gate_checks_full_sensor_and_actuator_set(self):
        streams = {
            topic: {"hz": 39.9, "gap_p95_ms": 27.0, "gap_max_ms": 50.0}
            for topic in (
                "/autodrive/roboracer_1/bridge_packet_timing",
                "/autodrive/roboracer_1/imu",
                "/autodrive/roboracer_1/left_encoder",
                "/autodrive/roboracer_1/odom",
                "/autodrive/roboracer_1/right_encoder",
                "/autodrive/roboracer_1/steering",
                "/autodrive/roboracer_1/steering_command",
                "/autodrive/roboracer_1/throttle",
                "/autodrive/roboracer_1/throttle_command",
            )
        }
        _validate_streams({"streams": streams})
        del streams["/autodrive/roboracer_1/imu"]
        with self.assertRaisesRegex(ValueError, "stream set"):
            _validate_streams({"streams": streams})


if __name__ == "__main__":
    unittest.main()
