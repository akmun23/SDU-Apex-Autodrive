import unittest
import numpy as np
from tools.make_tracking_clearance_target import (
    smooth_offsets, path_geometry, smooth_geometry, target_speed_profile)


class TrackingClearanceTargetTests(unittest.TestCase):
    def test_speed_scaling_preserves_geometry_and_scales_acceleration_squared(self):
        speed = np.array([3., 4., 6., 5.])
        segments = np.array([.05, .07, .03, .08])
        u, a = target_speed_profile(speed, segments)
        np.testing.assert_array_equal(u, speed)
        scaled_u, scaled_a = target_speed_profile(speed, segments, 1.03)
        np.testing.assert_allclose(scaled_u, 1.03 * u)
        np.testing.assert_allclose(scaled_a, 1.03**2 * a)
        for invalid in [0., -.1, float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                target_speed_profile(speed, segments, invalid)
        with self.assertRaises(ValueError):
            target_speed_profile(speed, [-1., .1, .1, .1])

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

    def test_geometry_regularization_matches_periodic_circle_eigenvalue(self):
        angle = np.linspace(0., 2*np.pi, 80, endpoint=False)
        xy = 2*np.column_stack((np.cos(angle), np.sin(angle)))
        h = np.linalg.norm(xy[1] - xy[0])
        eigenvalue = (2 - 2*np.cos(2*np.pi/len(xy))) / h**2
        scale = 1 / (1 + .1**4 * eigenvalue**2)
        actual = smooth_geometry(xy, .1, .02)
        np.testing.assert_allclose(actual, scale*xy, atol=2.e-12)
        np.testing.assert_array_equal(smooth_geometry(xy, 0., .02), xy)

    def test_geometry_smoothing_is_rigid_transform_and_seam_invariant(self):
        angle = np.linspace(0., 2*np.pi, 71, endpoint=False)**1.05
        angle *= (2*np.pi - .06)/angle[-1]
        xy = np.column_stack((2*np.cos(angle), np.sin(angle)))
        actual = smooth_geometry(xy, .08, .05)
        rotation = np.array([[.6, -.8], [.8, .6]])
        shifted = xy @ rotation.T + [100., -40.]
        np.testing.assert_allclose(smooth_geometry(shifted, .08, .05),
            actual @ rotation.T + [100., -40.], atol=2.e-12)
        np.testing.assert_allclose(smooth_geometry(np.roll(xy, 17, axis=0), .08, .05),
            np.roll(actual, 17, axis=0), atol=2.e-12)

    def test_geometry_smoothing_reduces_small_alternating_curvature_noise(self):
        angle = np.linspace(0., 2*np.pi, 160, endpoint=False)
        radius = 2 + .001*(-1.)**np.arange(len(angle))
        xy = radius[:, None]*np.column_stack((np.cos(angle), np.sin(angle)))
        actual = smooth_geometry(xy, .1, .02)
        before = path_geometry(xy)[2]
        after = path_geometry(actual)[2]
        self.assertLess(np.std(after), .1*np.std(before))
        self.assertLess(np.max(np.linalg.norm(actual-xy, axis=1)), .002)
        with self.assertRaisesRegex(ValueError, 'displacement'):
            smooth_geometry(xy, .1, .0001)
        for bad in [-.1, float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                smooth_geometry(xy, bad, .02)


if __name__ == '__main__':
    unittest.main()
