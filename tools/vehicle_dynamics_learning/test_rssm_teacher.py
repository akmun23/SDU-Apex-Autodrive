from __future__ import annotations

import unittest

import numpy as np

from tools.vehicle_dynamics_learning.train_nssm import _torch
from tools.vehicle_dynamics_learning.train_rssm_teacher import _rssm_model


class RssmTeacherTest(unittest.TestCase):
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
