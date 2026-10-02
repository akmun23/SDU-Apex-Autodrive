"""Mathematical contract tests for the offline hybrid plant."""

from __future__ import annotations

import unittest

import numpy as np

from tools.vehicle_dynamics_learning.four_wheel_greybox import (
    WHEEL_RADIUS_M,
    _physical_model,
)
from tools.vehicle_dynamics_learning.hybrid_race_teacher import (
    HISTORY_FEATURE_SIZE,
    HISTORY_STEPS,
    hybrid_model_type,
)
from tools.vehicle_dynamics_learning.train_nssm import _torch


class HybridRaceTeacherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.torch, cls.nn = _torch()

    def _model(self):
        nominal_type = _physical_model(self.torch, self.nn, True)
        nominal = nominal_type()
        model_type = hybrid_model_type(
            self.torch, self.nn, nominal,
            np.zeros(HISTORY_FEATURE_SIZE, dtype=np.float32),
            np.ones(HISTORY_FEATURE_SIZE, dtype=np.float32),
        )
        model = model_type()
        model.eval()
        return nominal, model

    def test_zero_residual_is_exact_nominal_transition(self):
        torch = self.torch
        nominal, model = self._model()
        state = torch.zeros(2, 21)
        state[:, 0] = torch.tensor((2.0, 7.0))
        state[:, 3:7] = state[:, 0:1] / WHEEL_RADIUS_M
        state[:, 17] = torch.tensor((0.0, 0.15))
        state[:, 18] = torch.tensor((0.35, 0.55))
        state[:, 19] = state[:, 17]
        state[:, 20] = state[:, 18]
        command = torch.tensor(((0.1, 0.4), (-0.2, 0.6)))
        latent = torch.zeros(2, model.latent_size)

        expected_state, expected_acceleration = nominal.step(state, command)
        actual_state, _, actual_acceleration, _, residual = model.step(
            state, latent, command)

        self.assertTrue(torch.equal(residual, torch.zeros_like(residual)))
        self.assertTrue(torch.allclose(actual_state, expected_state,
                                       rtol=0.0, atol=1.0e-7))
        self.assertTrue(torch.allclose(actual_acceleration,
                                       expected_acceleration,
                                       rtol=0.0, atol=1.0e-7))

    def test_causal_rollout_has_expected_shapes_and_finite_values(self):
        torch = self.torch
        _, model = self._model()
        initial = torch.zeros(2, 21)
        initial[:, 0] = torch.tensor((1.0, 5.0))
        initial[:, 3:7] = initial[:, 0:1] / WHEEL_RADIUS_M
        history = torch.zeros(2, HISTORY_STEPS, HISTORY_FEATURE_SIZE)
        commands = torch.zeros(2, 12, 2)
        commands[:, :, 1] = 0.4

        state, acceleration, wheel_acceleration, residual = model.rollout(
            initial, history, commands)

        self.assertEqual(tuple(state.shape), (2, 12, 21))
        self.assertEqual(tuple(acceleration.shape), (2, 12, 3))
        self.assertEqual(tuple(wheel_acceleration.shape), (2, 12, 2))
        self.assertEqual(tuple(residual.shape), (2, 12, 5))
        for values in (state, acceleration, wheel_acceleration, residual):
            self.assertTrue(torch.isfinite(values).all())


if __name__ == "__main__":
    unittest.main()
