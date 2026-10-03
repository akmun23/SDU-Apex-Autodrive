"""Mathematical checks for left/right reflection transforms."""

import unittest

import numpy as np

from tools.vehicle_dynamics_learning.diagnose_turn_reflection_symmetry import (
    _reflect_features,
    _reflect_state_delta,
)


class TurnReflectionSymmetryTest(unittest.TestCase):
    def test_feature_reflection_changes_signs_and_swaps_rear_wheels(self):
        source = np.asarray([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
                              8.0, 9.0, 10.0, 11.0]])
        expected = np.asarray([[1.0, -2.0, -3.0, -4.0, 5.0, 7.0, 6.0,
                                -8.0, 9.0, -10.0, 11.0]])
        np.testing.assert_array_equal(_reflect_features(source), expected)

    def test_state_delta_reflection_negates_lateral_yaw_steering_and_swaps_wheels(self):
        source = np.asarray([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]])
        expected = np.asarray([[1.0, -2.0, -3.0, -4.0, 5.0, 7.0, 6.0]])
        np.testing.assert_array_equal(_reflect_state_delta(source), expected)

    def test_invalid_shapes_rejected(self):
        with self.assertRaises(ValueError):
            _reflect_features(np.zeros((2, 10)))
        with self.assertRaises(ValueError):
            _reflect_state_delta(np.zeros((2, 6)))


if __name__ == "__main__":
    unittest.main()
