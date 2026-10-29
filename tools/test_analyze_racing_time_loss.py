import math
import unittest
import numpy as np
from tools.analyze_racing_time_loss import (account, clipped_samples, time_densities,
                                          truth_path_rows, summarize_truth_path)


class RacingTimeLossTests(unittest.TestCase):
    def test_density_identity_with_sideslip_and_outside_path(self):
        for u, v, epsi, ey, curve, uref in [(5., .2, .1, .1, -.5, 6.),
                                           (3., -.1, -.2, -.1, .8, 3.5)]:
            terms = time_densities(u, v, epsi, ey, curve, uref)
            exact = (1-curve*ey)/(u*math.cos(epsi)-v*math.sin(epsi)) - 1/uref
            self.assertAlmostEqual(float(sum(terms)), exact, places=14)

    def test_uniform_run_matches_exact_time_loss_across_seam(self):
        t = np.linspace(0., 5., 201)
        samples = np.column_stack([t, 3.+4*t, np.full_like(t, 4.),
                                   np.zeros((len(t), 3))])
        result = account(samples, np.array([0., 2., 5., 8.]),
                         np.full(4, 5.), np.zeros(4), 10., 1.)
        total = result['bins'].sum(axis=0)
        np.testing.assert_allclose(total, [5., 4., 1., 0., 0., 0., 20.], atol=1.e-12)
        self.assertEqual(result['reference_midpoint_integral_s'], 2.)

    def test_backwards_projection_is_reported_not_dropped(self):
        samples = np.array([[0, 4, 4, 0, 0, 0], [.1, 3.99, 4, 0, 0, 0],
                            [.2, 4.8, 4, 0, 0, 0]], dtype=float)
        result = account(samples, np.array([0., 2., 5., 8.]),
                         np.full(4, 4.), np.zeros(4), 10., 1.)
        self.assertEqual(result['backward_progress_intervals'], 1)
        self.assertAlmostEqual(result['bins'][:, 0].sum(), .2)

    def test_window_clipping_interpolates_both_endpoints(self):
        rows = np.array([[0, 0, 3, 0, 0, 0], [1, 3, 3, 0, 0, 0],
                         [2, 6, 3, 0, 0, 0]], dtype=float)
        clipped = clipped_samples(rows, .25, 1.75)
        np.testing.assert_allclose(clipped[:, :2], [[.25, .75], [1., 3.], [1.75, 5.25]])
        with self.assertRaises(ValueError): clipped_samples(rows, -1., 1.)

    def test_invalid_motion_is_rejected(self):
        for values in [(0., 0., 0., 0., 0., 5.),
                       (5., 0., math.pi, 0., 0., 5.),
                       (5., 0., 0., 2., 1., 5.)]:
            with self.assertRaises(ValueError): time_densities(*values)

    def test_truth_projection_uses_exact_epoch_not_nearby_pose(self):
        ref = np.array([[0., 0., 0., 0.], [2., 2., 0., math.pi/2],
                        [4., 2., 2., math.pi], [6., 0., 2., -math.pi/2]])
        gt = dict(stamp_ns=100, bag_time=1., x=1., y=.1, yaw=0., u=4., r=0.)
        poses = [dict(stamp_ns=99, x=100., y=100.),
                 dict(stamp_ns=100, x=.95, y=.12)]
        rows, length = truth_path_rows([gt], poses, [dict(bag_time=0., value=1)], ref)
        self.assertEqual(length, 8.)
        self.assertAlmostEqual(rows[0]['s_m'], 1.)
        self.assertAlmostEqual(rows[0]['map_forward_error_m'], -.05)
        self.assertAlmostEqual(rows[0]['map_lateral_error_m'], .02)
        report = summarize_truth_path(rows, length, 1)
        self.assertEqual(report['whole_run']['samples'], 1)
        rows, _ = truth_path_rows([gt], poses[:1], [], ref)
        self.assertNotIn('map_forward_error_m', rows[0])
        self.assertEqual(rows[0]['lap'], -1)
        with self.assertRaisesRegex(ValueError, 'no truth samples'):
            summarize_truth_path(rows, length, 1)


if __name__ == '__main__': unittest.main()
