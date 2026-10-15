from __future__ import annotations

import unittest

import numpy as np

from tools.vehicle_dynamics_learning.four_wheel_greybox import (
    GRAVITY_MPS2,
    MASS_KG,
    WHEELBASE_M,
    _physical_model,
)
from tools.vehicle_dynamics_learning.train_nssm import _torch


class FourWheelGreyboxTest(unittest.TestCase):
    def test_published_tire_curve_landmarks_and_static_load_balance(self) -> None:
        torch, nn = _torch()
        model_type = _physical_model(torch, nn)
        model = model_type()
        values = model._curve(
            torch.tensor([0.0, 0.15, 0.25]), 0.15, 0.72, 0.25, 0.464,
            torch.tensor(1.0))
        np.testing.assert_allclose(values.numpy(), [0.0, 0.72, 0.464],
                                   atol=1.0e-6)
        parameters = model.physical_parameters()
        loads = model._wheel_loads(
            torch.zeros(1), torch.zeros(1), parameters)
        self.assertAlmostEqual(float(loads.detach().sum()), MASS_KG * GRAVITY_MPS2,
                               places=5)

    def test_ackermann_and_zero_slip_body_derivatives(self) -> None:
        torch, nn = _torch()
        torch.set_num_threads(1)
        model_type = _physical_model(torch, nn)
        model = model_type()
        positive = model._ackermann(torch.tensor([0.2]))
        self.assertGreater(float(positive[0, 0]), float(positive[0, 1]))
        negative = model._ackermann(torch.tensor([-0.2]))
        self.assertTrue(np.isclose(float(positive[0, 0]),
                                   -float(negative[0, 1])))
        state = torch.zeros((1, 17))
        state[:, 0] = 2.0
        state[:, 3:7] = 2.0 / 0.059
        state[:, 15:17] = 0.0
        derivative, acceleration = model._forces_and_derivatives(
            state, torch.zeros(1), torch.zeros(1))
        self.assertTrue(torch.isfinite(derivative).all())
        self.assertTrue(torch.isfinite(acceleration).all())
        self.assertEqual(tuple(derivative.shape), (1, 17))
        self.assertEqual(tuple(acceleration.shape), (1, 3))
        self.assertAlmostEqual(WHEELBASE_M, 0.324, places=6)

    def test_tire_force_slips_match_documented_equations(self) -> None:
        torch, nn = _torch()
        model = _physical_model(torch, nn)()
        longitudinal, lateral = model._tire_slips(
            torch.tensor([[12.0]]), torch.tensor([[8.0]]),
            torch.tensor([[3.0]]))
        self.assertAlmostEqual(float(longitudinal[0, 0]), 0.5, places=6)
        self.assertAlmostEqual(float(lateral[0, 0]), 3.0 / 8.0, places=6)

    def test_tire_force_slips_remain_finite_at_zero_contact_speed(self) -> None:
        torch, nn = _torch()
        model = _physical_model(torch, nn)()
        longitudinal, lateral = model._tire_slips(
            torch.tensor([[0.6]]), torch.tensor([[0.0]]),
            torch.tensor([[0.3]]))
        self.assertAlmostEqual(float(longitudinal[0, 0]), 1.2, places=6)
        self.assertAlmostEqual(float(lateral[0, 0]), 0.6, places=6)

    def test_exact_zero_throttle_brake_cannot_reverse_wheel_speed(self) -> None:
        torch, nn = _torch()
        model = _physical_model(torch, nn, exact_unity_actuators=True)()
        state = torch.zeros((1, 21), dtype=torch.float32)
        state[0, 3:7] = torch.tensor([100.0, -100.0, 0.01, -0.01])
        following, _ = model.step(state, torch.zeros((1, 2)))
        np.testing.assert_allclose(following[0, 3:7].detach().numpy(),
                                   np.zeros(4), atol=1.0e-6)

    def test_exact_unity_cawd_applies_428_nm_as_107_nm_per_wheel(self) -> None:
        torch, nn = _torch()
        model = _physical_model(torch, nn, exact_unity_actuators=True)()
        state = torch.zeros((1, 17), dtype=torch.float32)
        derivative, _ = model._forces_and_derivatives(
            state, torch.zeros(1), torch.ones(1))
        wheel_inertia = model.physical_parameters()["wheel_inertia_kgm2"]
        expected = 107.0 / wheel_inertia
        np.testing.assert_allclose(
            derivative[0, 3:7].detach().numpy(),
            np.full(4, float(expected.detach())), rtol=1e-5, atol=1e-2)

    def test_short_recursive_rollout_is_finite(self) -> None:
        torch, nn = _torch()
        torch.set_num_threads(1)
        model_type = _physical_model(torch, nn)
        model = model_type()
        initial = torch.zeros((2, 21), dtype=torch.float32)
        initial[:, 0] = 1.0
        initial[:, 3:7] = 1.0 / 0.059
        inputs = torch.zeros((2, 4, 2), dtype=torch.float32)
        inputs[:, :, 0] = 0.1
        inputs[:, :, 1] = 0.2
        state, acceleration = model.rollout(initial, inputs)
        self.assertEqual(tuple(state.shape), (2, 4, 21))
        self.assertEqual(tuple(acceleration.shape), (2, 4, 3))
        self.assertTrue(torch.isfinite(state).all())
        self.assertTrue(torch.isfinite(acceleration).all())
        loss = state.square().mean() + acceleration.square().mean()
        loss.backward()
        self.assertIsNotNone(model.raw_parameters.grad)
        self.assertTrue(torch.isfinite(model.raw_parameters.grad).all())

    def test_actuators_use_commands_and_respect_slew(self) -> None:
        torch, nn = _torch()
        model = _physical_model(torch, nn)()
        current = torch.zeros((1, 2))
        command = torch.tensor([[0.5, 1.0]])
        state = torch.zeros((1, 21))
        state[:, 17:19] = current
        first, _ = model.step(state, command)
        self.assertAlmostEqual(float(first[0, 17].detach()), 0.0, places=6)
        self.assertAlmostEqual(float(first[0, 18].detach()), 0.0, places=6)
        second, _ = model.step(first, command)
        self.assertGreater(float(second[0, 17].detach()), 0.0)
        self.assertGreater(float(second[0, 18].detach()), 0.0)
        self.assertLessEqual(float(second[0, 17].detach()), 8.0 * 0.025 + 1e-6)
        self.assertLessEqual(float(second[0, 18].detach()), 20.0 * 0.025 + 1e-6)


if __name__ == "__main__":
    unittest.main()
