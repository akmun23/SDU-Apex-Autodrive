import unittest

import numpy as np

from fit_regime_yaw_step_response import _equilibrium


class EquilibriumSurfaceTests(unittest.TestCase):
    def setUp(self):
        self.table = {
            (speed, steering): steering * (2.0 if speed == 4.5 else 3.0)
            for speed in (4.5, 6.5)
            for steering in (-0.10, -0.05, 0.0, 0.05, 0.10)
        }

    def test_negative_steering_preserves_measured_yaw_sign(self):
        values = _equilibrium(
            np.asarray((5.5, 5.5)), np.asarray((-0.05, 0.05)), self.table)
        self.assertLess(values[0], 0.0)
        self.assertGreater(values[1], 0.0)
        np.testing.assert_allclose(values, (-0.125, 0.125), atol=1e-10)

    def test_surface_does_not_extrapolate_beyond_steering_support(self):
        value = _equilibrium(np.asarray((5.5,)), np.asarray((0.15,)), self.table)
        self.assertTrue(np.isnan(value[0]))


if __name__ == "__main__":
    unittest.main()
