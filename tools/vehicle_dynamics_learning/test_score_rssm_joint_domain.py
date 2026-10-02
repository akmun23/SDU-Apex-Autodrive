from __future__ import annotations

import unittest

import numpy as np

from tools.vehicle_dynamics_learning.score_rssm_joint_domain import (
    _bin,
    _validation_windows,
)


class JointDomainScoreTest(unittest.TestCase):
    def test_joint_bins_keep_speed_and_steering_distinct(self) -> None:
        self.assertEqual(_bin(8.0, (0.0, 5.0, 7.0, 9.0, 12.000001), "mps"),
                         "7-9mps")
        self.assertEqual(
            _bin(0.45, (0.0, 0.1, 0.2, 0.3, 0.4, 0.5240001), "rad"),
            "0.4-0.524rad")
        self.assertIsNone(_bin(12.1, (0.0, 5.0, 7.0, 9.0, 12.000001), "mps"))

    def test_validation_windows_are_deterministic_and_run_grouped(self) -> None:
        count = 20
        frames = np.zeros((count, 9), dtype=np.float32)
        frames[:, 3] = 0.25
        frames[:, 5:7] = 6.0
        rigid = np.zeros((count, 15), dtype=np.float32)
        rigid[:, 7] = 6.0
        rigid[:, 8] = 0.1
        pose = np.zeros((count, 3), dtype=np.float32)
        targets = np.zeros((count, 10), dtype=np.float32)
        targets[:, 0] = 6.0
        data = {
            "bounds": np.asarray([[0, count]], dtype=np.int64),
            "seq_run": np.asarray([0], dtype=np.int32),
            "splits": np.asarray(["validation"]),
            "frames": frames,
            "simulator_rigid_state": rigid,
            "simulator_pose_xyyaw": pose,
        }

        first = _validation_windows(data, targets, 3, 4, 2, 2)
        second = _validation_windows(data, targets, 3, 4, 2, 2)

        key = ("5-7mps", "0.2-0.3rad", 0)
        self.assertIn(key, first)
        self.assertEqual(first, second)
        self.assertEqual(len(first[key]), 2)
        self.assertTrue(all(future_start - start == 3
                            for start, future_start in first[key]))

    def test_training_runs_are_not_returned(self) -> None:
        count = 20
        data = {
            "bounds": np.asarray([[0, count]], dtype=np.int64),
            "seq_run": np.asarray([0], dtype=np.int32),
            "splits": np.asarray(["train"]),
            "frames": np.zeros((count, 9), dtype=np.float32),
            "simulator_rigid_state": np.zeros((count, 15), dtype=np.float32),
            "simulator_pose_xyyaw": np.zeros((count, 3), dtype=np.float32),
        }
        targets = np.zeros((count, 10), dtype=np.float32)
        self.assertEqual(_validation_windows(data, targets, 3, 4, 1, 4), {})

if __name__ == "__main__":
    unittest.main()
