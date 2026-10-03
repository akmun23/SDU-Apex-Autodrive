"""Check symmetry augmentation preserves the state/pose equations."""

import unittest

import numpy as np

from tools.vehicle_dynamics_learning.train_effective_race_teacher import (
    _random_reflection_augmentation,
    _reflect_training_batch,
)


def _batch(state_size=9, history_state_size=9, extra_width=3):
    history = np.arange(2 * 4 * (history_state_size + 2 + extra_width),
                        dtype=np.float32).reshape(
                            2, 4, history_state_size + 2 + extra_width)
    state = np.arange(2 * 3 * state_size, dtype=np.float32).reshape(
        2, 3, state_size)
    initial = state[:, 0].copy()
    delayed = np.arange(4, dtype=np.float32).reshape(2, 2)
    commands = np.arange(12, dtype=np.float32).reshape(2, 3, 2)
    acceleration = np.arange(30, dtype=np.float32).reshape(2, 3, 5)
    poses = np.asarray([
        [[2.0, 1.0, 0.4], [2.2, 1.1, 0.5], [2.5, 1.4, 0.7], [2.7, 1.8, 0.9]],
        [[-1.0, 3.0, -0.8], [-0.8, 2.8, -0.7], [-0.5, 2.3, -0.4], [-0.1, 1.9, -0.2]],
    ], dtype=np.float32)
    return (history, initial, delayed, commands, state[:, 1:], acceleration, poses)


class TurnReflectionTrainingTest(unittest.TestCase):
    def test_reflection_is_an_involution_and_keeps_initial_pose(self):
        batch = _batch()
        reflected = _reflect_training_batch(batch, history_state_size=9)
        twice = _reflect_training_batch(reflected, history_state_size=9)
        for original, restored in zip(batch, twice):
            np.testing.assert_allclose(restored, original, rtol=0.0, atol=2e-6)
        np.testing.assert_allclose(reflected[-1][:, 0], batch[-1][:, 0],
                                   rtol=0.0, atol=1e-6)

    def test_roll_residual_history_and_encoder_sides_are_reflected(self):
        batch = _batch(state_size=9, history_state_size=7, extra_width=3)
        reflected = _reflect_training_batch(batch, history_state_size=7)
        history = reflected[0]
        # Roll is intentionally omitted from the seven-state residual history;
        # wheel-rate history is left/right swapped, validity is unchanged.
        start = 9
        np.testing.assert_array_equal(history[..., start], batch[0][..., start + 1])
        np.testing.assert_array_equal(history[..., start + 1], batch[0][..., start])
        np.testing.assert_array_equal(history[..., start + 2], batch[0][..., start + 2])

    def test_random_augmentation_preserves_batch_size_and_can_mirror_all(self):
        batch = _batch(state_size=7, history_state_size=7, extra_width=0)
        rng = np.random.default_rng(2)
        augmented = _random_reflection_augmentation(batch, 7, rng)
        expected = _reflect_training_batch(batch, 7)
        for actual, mirrored in zip(augmented, expected):
            np.testing.assert_array_equal(actual, mirrored)
            self.assertEqual(len(actual), len(batch[1]))


if __name__ == "__main__":
    unittest.main()
