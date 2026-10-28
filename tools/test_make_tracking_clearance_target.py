import unittest
import numpy as np
from tools.make_tracking_clearance_target import smooth_offsets, path_geometry


class TrackingClearanceTargetTests(unittest.TestCase):
    def test_sufficient_corridor_does_not_move_path(self):
        offsets = smooth_offsets(np.ones(40), np.ones(40), .25, .06, .6, .1)
        np.testing.assert_array_equal(offsets, np.zeros(40))

    def test_moves_away_from_correct_wall_with_periodic_smoothness(self):
        left, right = np.ones(80), np.ones(80)
        left[0] = .2
        offset = smooth_offsets(left, right, .25, .06, .6, .1)
        self.assertAlmostEqual(offset[0], -.05, places=7)
        self.assertLess(offset[-1], 0)
        np.testing.assert_allclose(offset[1:8], offset[-1:-8:-1], atol=1.e-7)
        mirrored = smooth_offsets(right, left, .25, .06, .6, .1)
        np.testing.assert_allclose(mirrored, -offset, atol=1.e-7)

    def test_rejects_impossible_reserve(self):
        with self.assertRaises(ValueError):
            smooth_offsets(np.full(40, .2), np.full(40, .2), .25, .06, .6, .1)

    def test_closed_circle_geometry_has_correct_curvature(self):
        angle = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        xy = 2 * np.column_stack((np.cos(angle), np.sin(angle)))
        segments, heading, curvature = path_geometry(xy)
        np.testing.assert_allclose(curvature, .5, atol=1.e-12)
        self.assertTrue(np.all(segments > 0))
        np.testing.assert_allclose(np.diff(heading), 2*np.pi/100, atol=1.e-12)


if __name__ == '__main__':
    unittest.main()
