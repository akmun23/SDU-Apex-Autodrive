from __future__ import annotations

import unittest

import numpy as np

from tools.vehicle_dynamics_learning.train_nssm import _torch
from tools.vehicle_dynamics_learning.train_rssm_teacher import _rssm_model
from tools.vehicle_dynamics_learning.train_rssm_teacher import (
    COM_X_M,
    _batch_loss,
    _integrate_pose_from_com_states,
    _local_pose_targets,
    _pose_aware_selection_score,
)


class RssmTeacherTest(unittest.TestCase):
    def test_pose_targets_are_in_initial_body_frame_and_wrap_heading(self):
        poses = np.asarray([
            [3.0, -2.0, np.pi - 0.02],
            [3.0 - np.sin(np.pi - 0.02),
             -2.0 + np.cos(np.pi - 0.02), -np.pi + 0.03],
        ])

        local = _local_pose_targets(poses)

        np.testing.assert_allclose(local[0, :2], [0.0, 1.0], atol=1e-6)
        self.assertAlmostEqual(float(local[0, 2]), 0.05, places=6)

    def test_pose_aware_checkpoint_score_is_run_and_horizon_macro(self):
        validation = {
            "checkpoint_selection_score": 0.2,
            "per_run": {
                "run-a": {"horizons": {
                    "0.25s": {"position_xy_rmse_m": [0.25, 0.0],
                               "heading_rmse_rad": 0.1},
                    "0.75s": {"position_xy_rmse_m": [0.25, 0.0],
                               "heading_rmse_rad": 0.1},
                    "2s": {"position_xy_rmse_m": [0.25, 0.0],
                           "heading_rmse_rad": 0.1}}},
                "run-b": {"horizons": {
                    "0.25s": {"position_xy_rmse_m": [0.0, 0.0],
                               "heading_rmse_rad": 0.0}}},
            },
        }

        score, pose_score = _pose_aware_selection_score(validation, 0.1)

        self.assertAlmostEqual(pose_score, 0.75)
        self.assertAlmostEqual(score, 0.275)

    def test_com_state_pose_integration_matches_rear_axle_circle(self):
        torch, _ = _torch()
        dt = 0.025
        u, yaw_rate = 2.0, 0.5
        state = torch.tensor([u, COM_X_M * yaw_rate, yaw_rate,
                              0.0, 0.0, 0.0, 0.0])
        initial = state[None, :]
        predicted = state[None, None, :].expand(1, 80, 7).clone()

        pose = _integrate_pose_from_com_states(
            torch, predicted, initial, np.zeros(7, dtype=np.float32),
            np.ones(7, dtype=np.float32))
        angle = yaw_rate * dt * 80
        radius = u / yaw_rate
        expected = np.asarray((radius * np.sin(angle),
                               radius * (1.0 - np.cos(angle)), angle))

        np.testing.assert_allclose(pose[0, -1].numpy(), expected, atol=1e-4)

    def test_pose_loss_backpropagates_through_deterministic_prior_rollout(self):
        torch, nn = _torch()
        torch.set_num_threads(1)
        model_type = _rssm_model(
            torch, nn, 32, 8, np.zeros(9, dtype=np.float32),
            np.ones(9, dtype=np.float32), np.zeros(10, dtype=np.float32),
            np.ones(10, dtype=np.float32))
        model = model_type()
        context = torch.zeros((2, 40, 9), dtype=torch.float32)
        commands = torch.zeros((2, 80, 2), dtype=torch.float32)
        labels = torch.zeros((2, 80, 10), dtype=torch.float32)
        initial = torch.zeros((2, 7), dtype=torch.float32)
        target_pose = torch.zeros((2, 80, 3), dtype=torch.float32)

        loss, parts = _batch_loss(
            torch, nn, model, context, commands, labels, initial,
            kl_weight=0.01, free_rollout_weight=1.0,
            acceleration_weight=0.05, pose_targets=target_pose,
            pose_loss_weight=0.1, y_mean=np.zeros(10, dtype=np.float32),
            y_scale=np.ones(10, dtype=np.float32))
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(parts["integrated_pose"]))
        self.assertTrue(any(parameter.grad is not None
                            and torch.isfinite(parameter.grad).all()
                            for parameter in model.parameters()))

    def test_prior_rollout_predicts_state_and_acceleration(self) -> None:
        torch, nn = _torch()
        torch.set_num_threads(1)
        mean = np.zeros(9, dtype=np.float32)
        scale = np.ones(9, dtype=np.float32)
        target_mean = np.zeros(10, dtype=np.float32)
        target_scale = np.ones(10, dtype=np.float32)
        model_type = _rssm_model(
            torch, nn, 32, 8, mean, scale, target_mean, target_scale)
        model = model_type()
        context = torch.zeros((3, 40, 9), dtype=torch.float32)
        commands = torch.zeros((3, 80, 2), dtype=torch.float32)

        prediction = model(context, commands)

        self.assertEqual(tuple(prediction.shape), (3, 80, 10))
        self.assertTrue(torch.isfinite(prediction).all())

    def test_prior_free_rollout_does_not_require_future_labels(self) -> None:
        torch, nn = _torch()
        torch.set_num_threads(1)
        mean = np.zeros(9, dtype=np.float32)
        scale = np.ones(9, dtype=np.float32)
        target_mean = np.zeros(10, dtype=np.float32)
        target_scale = np.ones(10, dtype=np.float32)
        model_type = _rssm_model(
            torch, nn, 32, 8, mean, scale, target_mean, target_scale)
        model = model_type().eval()
        context = torch.randn((2, 40, 9), dtype=torch.float32)
        commands = torch.randn((2, 80, 2), dtype=torch.float32)

        with torch.no_grad():
            first = model(context, commands)
            second = model(context, commands)

        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.isfinite(first).all())


if __name__ == "__main__":
    unittest.main()
