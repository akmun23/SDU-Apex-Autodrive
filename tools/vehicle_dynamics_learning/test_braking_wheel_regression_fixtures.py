from __future__ import annotations

import unittest

import numpy as np

from tools.vehicle_dynamics_learning.build_braking_wheel_regression_fixtures import (
    _candidate_events,
)


class BrakingWheelRegressionFixturesTest(unittest.TestCase):
    def test_finds_zero_throttle_onset_and_later_release_with_history(self):
        frames = np.zeros((180, 9), dtype=np.float32)
        frames[:, 0] = 5.0
        frames[:, 7] = -0.1
        frames[:, 8] = 0.4
        frames[120:150, 8] = 0.0
        rigid = np.zeros((180, 13), dtype=np.float32)
        rigid[:, 7] = 5.0
        rigid[119, 7] = 11.074
        bounds = np.asarray([[0, 180]], dtype=np.int64)
        data = {
            "run_ids": np.asarray(["openplane_race_domain_moderate_braking_train_r01"]),
            "run_splits": np.asarray(["train"]),
            "frames": frames,
            "simulator_rigid_state": rigid,
            "sequence_bounds": bounds,
            "sequence_run_index": np.asarray([0], dtype=np.int32),
            "sequence_labels": np.asarray(["race_domain_moderate_braking"]),
            "packet_sequence": np.arange(180, dtype=np.int64),
            "dt_s": np.full(180, 0.025, dtype=np.float32),
        }
        events = _candidate_events(
            data, "openplane_race_domain_moderate_braking_train_r01")
        event = next(row for row in events
                     if row["fixture_name"] == "high_speed_low_steer")
        self.assertEqual(event["brake_command_index"], 120)
        self.assertEqual(event["brake_release_command_index"], 150)
        self.assertEqual(event["brake_initial_state_index"], 119)
        self.assertEqual(event["brake_history_start_index"], 40)

    def test_rejects_a_packet_gap_in_selected_event(self):
        frames = np.zeros((180, 9), dtype=np.float32)
        frames[:, 0] = 5.0
        frames[:, 7] = -0.1
        frames[:, 8] = 0.4
        frames[120:150, 8] = 0.0
        rigid = np.zeros((180, 13), dtype=np.float32)
        rigid[:, 7] = 5.0
        rigid[119, 7] = 11.074
        packet = np.arange(180, dtype=np.int64)
        packet[130:] += 1
        data = {
            "run_ids": np.asarray(["openplane_race_domain_moderate_braking_train_r01"]),
            "run_splits": np.asarray(["train"]),
            "frames": frames,
            "simulator_rigid_state": rigid,
            "sequence_bounds": np.asarray([[0, 180]], dtype=np.int64),
            "sequence_run_index": np.asarray([0], dtype=np.int32),
            "sequence_labels": np.asarray(["race_domain_moderate_braking"]),
            "packet_sequence": packet,
            "dt_s": np.full(180, 0.025, dtype=np.float32),
        }
        with self.assertRaisesRegex(ValueError, "packet gap"):
            _candidate_events(
                data, "openplane_race_domain_moderate_braking_train_r01")


if __name__ == "__main__":
    unittest.main()
