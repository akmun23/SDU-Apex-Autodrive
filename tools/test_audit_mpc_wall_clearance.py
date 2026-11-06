import math
import unittest
import numpy as np
from tools.audit_mpc_wall_clearance import sample_trajectory, vehicle_pose


class WallClearanceAuditMathTest(unittest.TestCase):
    def setUp(self):
        self.line = np.array([
            [0, 1., 2., math.pi - .02, 0, 2, 0, 1, 1],
            [1, 2., 2., -math.pi + .02, 0, 2, 0, 1, 1],
            [2, 2., 3., math.pi / 2, 0, 2, 0, 1, 1],
        ], dtype=float)

    def test_heading_interpolation_uses_shortest_angle(self):
        _, _, yaw = sample_trajectory(self.line, 3., .5)
        self.assertAlmostEqual(yaw, math.pi, places=7)

    def test_rear_axle_uses_normal_not_tangent(self):
        x, y, h = vehicle_pose(self.line, 3., 0., .2, .1)
        self.assertAlmostEqual(x, 1. - .2 * math.sin(math.pi - .02))
        self.assertAlmostEqual(y, 2. + .2 * math.cos(math.pi - .02))
        self.assertAlmostEqual(h, math.pi - .02 + .1)

    def test_closed_wrap(self):
        a = sample_trajectory(self.line, 3., .4)
        b = sample_trajectory(self.line, 3., 3.4)
        self.assertTrue(np.allclose(a, b))


if __name__ == "__main__":
    unittest.main()
