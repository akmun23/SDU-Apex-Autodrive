"""Mathematical checks for race-track projection and Frenet sign conventions."""

from __future__ import annotations

import math
import unittest

from tools.racing.track_projection import TrackProjection, wrap_angle


class TrackProjectionTests(unittest.TestCase):
    def test_straight_signed_offset_and_heading_wrap(self) -> None:
        track = TrackProjection([0.0, 1.0, 2.0], [0.0, 0.0, 0.0], closed=False)
        left = track.project(0.75, 0.2, math.pi - 0.02)
        right = track.project(0.75, -0.2, -math.pi + 0.02)
        self.assertAlmostEqual(left.lateral_offset_m, 0.2)
        self.assertAlmostEqual(right.lateral_offset_m, -0.2)
        self.assertAlmostEqual(left.s_m, 0.75)
        self.assertAlmostEqual(abs(left.heading_error_rad), math.pi - 0.02)
        self.assertAlmostEqual(right.heading_error_rad, -math.pi + 0.02)
        self.assertAlmostEqual(wrap_angle(2.0 * math.pi + 0.1), 0.1)

    def test_circle_projection_and_loop_wrap(self) -> None:
        count = 512
        radius = 2.0
        angles = [2.0 * math.pi * i / count for i in range(count)]
        track = TrackProjection(
            [radius * math.cos(a) for a in angles],
            [radius * math.sin(a) for a in angles],
            closed=True,
        )
        angle = 1.1
        query_radius = radius + 0.3
        result = track.project(
            query_radius * math.cos(angle), query_radius * math.sin(angle), angle + math.pi / 2
        )
        # For a counter-clockwise circle, the left normal points inward.
        self.assertAlmostEqual(result.lateral_offset_m, -0.3, delta=0.001)
        self.assertAlmostEqual(result.heading_error_rad, 0.0, delta=0.002)
        self.assertLess(result.s_m, track.total_length)
        self.assertGreater(result.s_m, 0.0)

        near_wrap = track.project(radius * 1.01, -0.001, -math.pi / 2)
        self.assertLess(min(near_wrap.s_m, track.total_length - near_wrap.s_m), 0.01)

    def test_synthetic_spline_and_nearest_segment_continuity(self) -> None:
        xs = [i / 100.0 for i in range(-100, 101)]
        ys = [0.15 * x * x for x in xs]
        track = TrackProjection(xs, ys, closed=False)
        index = 147
        dx = xs[index + 1] - xs[index]
        dy = ys[index + 1] - ys[index]
        heading = math.atan2(dy, dx)
        normal = (-math.sin(heading), math.cos(heading))
        qx = xs[index] + 0.4 * dx + 0.2 * normal[0]
        qy = ys[index] + 0.4 * dy + 0.2 * normal[1]
        local = track.project(
            qx, qy, heading, previous_segment=index, local_search_radius=2
        )
        global_result = track.project(qx, qy, heading)
        self.assertAlmostEqual(local.lateral_offset_m, 0.2, delta=1e-9)
        self.assertEqual(local.segment_index, index)
        self.assertAlmostEqual(local.s_m, global_result.s_m, delta=1e-9)


if __name__ == "__main__":
    unittest.main()
