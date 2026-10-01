from __future__ import annotations

import unittest

import numpy as np

from tools.vehicle_dynamics_learning.train_nssm import _torch
from tools.vehicle_dynamics_learning.offline_plant import (
    HistoricalGruPlant,
    RssmTeacherPlant,
)
from tools.vehicle_dynamics_learning.train_rssm_teacher import _rssm_model


class _Cell:
    hidden_size = 1

    def __call__(self, feature, hidden):
        return hidden + feature[:, :1]


class _EchoModel:
    def __init__(self, torch):
        self.torch = torch
        self.cell = _Cell()
        self._parameter = torch.nn.Parameter(torch.zeros(()))

    def parameters(self):
        return iter((self._parameter,))

    def advance(self, feature, hidden, dt):
        # Hold the predicted plant state; the test checks causal command wiring.
        return feature[:, :7], hidden + dt[:, None]


class OfflinePlantTest(unittest.TestCase):
    def test_reset_and_command_only_step(self) -> None:
        torch, _ = _torch()
        torch.set_num_threads(1)
        plant = HistoricalGruPlant(
            torch, [_EchoModel(torch), _EchoModel(torch)],
            np.zeros(9, dtype=np.float32), np.ones(9, dtype=np.float32), 3)
        history = np.zeros((3, 9), dtype=np.float32)
        history[:, 0] = 2.0
        history[:, 1] = 0.2
        history[:, 2] = 0.5
        history[:, 3:7] = [0.1, 0.4, 1.8, 2.1]
        history[:, 7:9] = [0.1, 0.4]
        initial_pose = np.asarray([1.0, 2.0, 0.3], dtype=np.float32)
        initial = plant.reset(history, initial_pose)
        np.testing.assert_allclose(
            initial.state, np.concatenate((initial_pose, history[-1, :7])))
        self.assertEqual(plant.get_support(), None)

        output = plant.step(0.3, 0.5)

        np.testing.assert_allclose(output.state[3:], history[-1, :7])
        self.assertGreater(output.state[0], initial_pose[0])
        self.assertEqual(output.uncertainty.shape, (10,))
        np.testing.assert_array_equal(output.uncertainty, np.zeros(10))
        self.assertEqual(plant.get_support(), None)

    def test_requires_reset_and_rejects_bad_time_step(self) -> None:
        torch, _ = _torch()
        plant = HistoricalGruPlant(
            torch, [_EchoModel(torch)], np.zeros(9, dtype=np.float32),
            np.ones(9, dtype=np.float32), 2)
        with self.assertRaises(RuntimeError):
            plant.step(0.0, 0.0)
        plant.reset(np.zeros((2, 9), dtype=np.float32))
        with self.assertRaises(ValueError):
            plant.step(0.0, 0.0, 0.0)

    def test_rssm_adapter_steps_from_prior_only(self) -> None:
        torch, nn = _torch()
        torch.set_num_threads(1)
        x_mean = np.zeros(9, dtype=np.float32)
        x_scale = np.ones(9, dtype=np.float32)
        y_mean = np.zeros(10, dtype=np.float32)
        y_scale = np.ones(10, dtype=np.float32)
        model_type = _rssm_model(
            torch, nn, 16, 4, x_mean, x_scale, y_mean, y_scale)
        model = model_type().eval()
        plant = RssmTeacherPlant(
            torch, [model], x_mean, x_scale, y_mean, y_scale, 40)
        history = np.zeros((40, 9), dtype=np.float32)
        history[-1, :3] = [2.0, 0.2, 1.0]
        history[-1, 3:9] = [0.1, 0.4, 1.8, 2.1, 0.2, 0.5]

        initial = plant.reset(history)
        self.assertAlmostEqual(initial.state[4], 0.2)
        output = plant.step(0.15, 0.45)

        self.assertEqual(output.state.shape, (10,))
        self.assertEqual(output.uncertainty.shape, (10,))
        self.assertTrue(np.isfinite(output.state).all())
        self.assertTrue(np.isfinite(output.uncertainty).all())
        self.assertIsNone(plant.get_support())


if __name__ == "__main__":
    unittest.main()
