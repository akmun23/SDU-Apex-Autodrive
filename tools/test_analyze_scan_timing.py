import unittest
import numpy as np
from tools.analyze_scan_timing import endpoint_losses


class ScanTimingTests(unittest.TestCase):
    def test_map_pixel_centers_and_exact_pose(self):
        # Vertical wall at x=2.5, map lower-left corner (0, 0), one-metre cells.
        field = np.tile(np.array([2., 1., 0., 1., 2.]), (5, 1))
        poses = np.array([[.5, .5, 0.], [.4, .5, 0.]])
        losses = endpoint_losses(field, 1., [0., 0., 0.], poses,
                                 np.array([2.]), np.array([0.]), (0., 0.))
        self.assertAlmostEqual(losses[0], 0.)
        self.assertAlmostEqual(losses[1], .01)

    def test_rotated_laser_extrinsic(self):
        field = np.tile(np.array([2., 1., 0., 1., 2.])[:, None], (1, 5))
        losses = endpoint_losses(field, 1., [0., 0., 0.],
            np.array([[.5, .5, np.pi/2]]), np.array([1.]), np.array([0.]), (1., 0.))
        self.assertAlmostEqual(losses[0], 0.)


if __name__ == '__main__': unittest.main()
