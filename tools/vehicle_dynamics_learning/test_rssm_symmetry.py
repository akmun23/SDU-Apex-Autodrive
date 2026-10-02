from __future__ import annotations

import unittest

import numpy as np

from tools.vehicle_dynamics_learning.train_rssm_teacher import (
    _mirror_normalized_commands,
    _mirror_normalized_features,
    _mirror_normalized_state,
    _mirror_normalized_targets,
)


class RssmSymmetryTest(unittest.TestCase):
    def test_reflection_signs_and_wheel_swap_are_applied_in_physical_units(self):
        feature_mean = np.asarray([0.3, 0.1, -0.2, 0.05, 0.4,
                                   0.2, -0.1, 0.02, 0.35], dtype=np.float32)
        feature_scale = np.asarray([2.0, 0.5, 1.2, 0.4, 0.3,
                                    3.0, 2.0, 0.4, 0.3], dtype=np.float32)
        features = np.asarray([[4.0, 0.2, 0.4, 0.3, 0.5,
                                8.0, 9.0, 0.35, 0.6]], dtype=np.float32)
        normalized_features = (features - feature_mean) / feature_scale
        mirrored_features = _mirror_normalized_features(
            normalized_features, feature_mean, feature_scale)
        physical_mirrored_features = (
            mirrored_features * feature_scale + feature_mean)
        np.testing.assert_allclose(
            physical_mirrored_features,
            [[4.0, -0.2, -0.4, -0.3, 0.5, 9.0, 8.0, -0.35, 0.6]],
            atol=1e-6)
        np.testing.assert_allclose(
            _mirror_normalized_features(
                mirrored_features, feature_mean, feature_scale),
            normalized_features, atol=1e-6)

    def test_state_and_acceleration_reflection(self):
        state_mean = np.asarray([0.2, -0.1, 0.3, 0.05, 0.4, 1.0, 1.5],
                                dtype=np.float32)
        state_scale = np.asarray([2.0, 0.7, 1.2, 0.4, 0.3, 3.0, 2.0],
                                 dtype=np.float32)
        state = np.asarray([[4.0, 0.3, 0.4, 0.3, 0.5, 8.0, 9.0]],
                           dtype=np.float32)
        normalized_state = (state - state_mean) / state_scale
        reflected_state = _mirror_normalized_state(
            normalized_state, state_mean, state_scale)
        np.testing.assert_allclose(
            reflected_state * state_scale + state_mean,
            [[4.0, -0.3, -0.4, -0.3, 0.5, 9.0, 8.0]], atol=1e-6)

        target_mean = np.r_[state_mean, [0.2, -0.3, 0.1]].astype(np.float32)
        target_scale = np.r_[state_scale, [2.0, 3.0, 0.8]].astype(np.float32)
        target = np.asarray([[4.0, 0.3, 0.4, 0.3, 0.5,
                              8.0, 9.0, 1.2, 2.1, 0.5]], dtype=np.float32)
        normalized_target = (target - target_mean) / target_scale
        reflected_target = _mirror_normalized_targets(
            normalized_target, target_mean, target_scale)
        np.testing.assert_allclose(
            reflected_target * target_scale + target_mean,
            [[4.0, -0.3, -0.4, -0.3, 0.5,
              9.0, 8.0, 1.2, -2.1, -0.5]], atol=1e-6)
        np.testing.assert_allclose(
            _mirror_normalized_targets(
                reflected_target, target_mean, target_scale),
            normalized_target, atol=1e-6)

    def test_steering_command_reflection_preserves_throttle(self):
        mean = np.asarray([0.1, 0.4], dtype=np.float32)
        scale = np.asarray([0.5, 0.2], dtype=np.float32)
        commands = np.asarray([[0.3, 0.6]], dtype=np.float32)
        normalized = (commands - mean) / scale
        reflected = _mirror_normalized_commands(normalized, mean, scale)
        np.testing.assert_allclose(reflected * scale + mean,
                                   [[-0.3, 0.6]], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
