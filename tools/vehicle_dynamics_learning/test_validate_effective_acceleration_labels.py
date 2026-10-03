"""Focused numerical tests for WP8 acceleration transforms."""

from __future__ import annotations

import unittest

import numpy as np

from tools.vehicle_dynamics_learning.validate_effective_acceleration_labels import (
    _make_candidates,
    _rotate_xyzw,
)
from tools.vehicle_dynamics_learning.effective_race_teacher import (
    fit_actuator_dynamics,
)


class EffectiveAccelerationLabelTest(unittest.TestCase):
    def test_quaternion_inverse_recovers_body_vector(self):
        angle = 0.73
        q = np.asarray([[0.0, 0.0, np.sin(angle / 2), np.cos(angle / 2)]])
        body = np.asarray([[1.7, -3.2, 0.4]])
        world = _rotate_xyzw(q, body)
        recovered = _rotate_xyzw(q, world, inverse=True)
        np.testing.assert_allclose(recovered, body, rtol=0.0, atol=1e-12)

    def test_candidate_set_exposes_frame_and_sample_alignment(self):
        angle = 0.41
        quaternion = np.asarray([
            [0.0, 0.0, np.sin(angle / 2), np.cos(angle / 2)],
            [0.0, 0.0, np.sin(angle / 2), np.cos(angle / 2)],
        ])
        rigid = np.zeros((2, 13), dtype=np.float64)
        rigid[:, 3:7] = quaternion
        body_acc = np.asarray([[2.0, -1.0, 0.0], [4.0, 3.0, 0.0]])
        world_acc = _rotate_xyzw(quaternion, body_acc)
        candidates = _make_candidates(rigid, world_acc, np.asarray([0]))
        np.testing.assert_allclose(
            candidates["quaternion_inverse_start_sample"], body_acc[:1, :2],
            rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(
            candidates["quaternion_inverse_interval_mean"],
            np.mean(body_acc[:, :2], axis=0, keepdims=True),
            rtol=0.0, atol=1e-12)
        self.assertIn("unrotated_start_sample", candidates)
        self.assertIn("unrotated_end_sample", candidates)

    def test_actuator_fit_uses_balanced_sequence_weighting(self):
        rng = np.random.default_rng(7721)
        lengths = (120, 1400)
        frame_blocks = []
        bounds = []
        cursor = 0
        for length in lengths:
            block = np.zeros((length, 9), dtype=np.float64)
            for feedback_col, command_col in ((3, 7), (4, 8)):
                command = rng.choice((0.0, 0.2, 0.5, 0.8), size=length)
                feedback = np.zeros(length, dtype=np.float64)
                for index in range(length - 1):
                    feedback[index + 1] = feedback[index] + 0.4 * (
                        command[index] - feedback[index])
                block[:, command_col] = command
                block[:, feedback_col] = feedback
            frame_blocks.append(block)
            bounds.append((cursor, cursor + length))
            cursor += length
        data = {
            "frames": np.concatenate(frame_blocks),
            "bounds": np.asarray(bounds, dtype=np.int64),
            "seq_run": np.asarray([0, 1], dtype=np.int32),
            "splits": np.asarray(["train", "train"]),
            "run_ids": np.asarray(["short", "long"]),
            "training_families": np.asarray(["open_plane", "open_plane"]),
            "run_families": np.asarray(["open_plane", "open_plane"]),
            "sequence_condition_id": np.asarray([0, 1], dtype=np.int32),
            "training_family_names": np.asarray(["open_plane"]),
            "training_family_probabilities": np.asarray([1.0]),
        }
        fit = fit_actuator_dynamics(data)
        self.assertEqual(fit.steering.delay_steps, 0)
        self.assertEqual(fit.throttle.delay_steps, 0)
        self.assertAlmostEqual(fit.steering.alpha, 0.4, places=5)
        self.assertAlmostEqual(fit.throttle.alpha, 0.4, places=5)


if __name__ == "__main__":
    unittest.main()
