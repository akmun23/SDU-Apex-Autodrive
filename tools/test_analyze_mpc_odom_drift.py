"""Focused tests for the exact-epoch odometry drift report."""

from __future__ import annotations

import math
import unittest

from tools import analyze_mpc_odom_drift as drift


class ExactEpochOdomDriftTest(unittest.TestCase):
    def test_exact_join_does_not_round_a_one_nanosecond_mismatch(self) -> None:
        truth = [{"stamp_ns": 1_000_000_000}, {"stamp_ns": 2_000_000_000}]
        odom = [{"stamp_ns": 1_000_000_000}, {"stamp_ns": 2_000_000_001}]
        pairs, counts = drift.exact_epoch_pairs(truth, odom)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(counts["truth_unmatched"], 1)
        self.assertEqual(counts["odom_unmatched"], 1)

    def test_exact_join_rejects_duplicate_epochs(self) -> None:
        rows = [{"stamp_ns": 5}, {"stamp_ns": 5}]
        with self.assertRaisesRegex(ValueError, "duplicate exact timestamp"):
            drift.exact_epoch_pairs(rows, [{"stamp_ns": 5}])

    def test_exact_join_sorts_reordered_rows_and_counts_a_missing_epoch(self) -> None:
        truth = [{"stamp_ns": 30}, {"stamp_ns": 10}]
        odom = [{"stamp_ns": 20}, {"stamp_ns": 30}, {"stamp_ns": 10}]

        pairs, counts = drift.exact_epoch_pairs(truth, odom)

        self.assertEqual(
            [(gt["stamp_ns"], estimate["stamp_ns"]) for gt, estimate in pairs],
            [(10, 10), (30, 30)])
        self.assertEqual(counts["truth_unmatched"], 0)
        self.assertEqual(counts["odom_unmatched"], 1)

    def test_midpoint_and_substep_integrate_straight_motion(self) -> None:
        rows = [{
            "stamp_ns": index * 25_000_000,
            "u": 4.0,
            "v": 0.0,
            "r": 0.0,
            "yaw": 0.0,
        } for index in range(41)]
        midpoint = drift.integrate_twist(rows, "odom")
        substep = drift.integrate_twist(rows, "odom", substep_s=0.001)
        self.assertAlmostEqual(midpoint[-1][0], 4.0, places=10)
        self.assertAlmostEqual(midpoint[-1][1], 0.0, places=10)
        self.assertAlmostEqual(substep[-1][0], 4.0, places=10)
        self.assertAlmostEqual(substep[-1][1], 0.0, places=10)

    def test_1ms_integration_tracks_constant_curvature(self) -> None:
        speed = 2.0
        yaw_rate = 1.0
        dt = 0.025
        rows = []
        for index in range(41):
            yaw = index * dt * yaw_rate
            rows.append({
                "stamp_ns": index * 25_000_000,
                "u": speed,
                # Make COM lateral velocity correspond to zero rear-axle v.
                "v": yaw_rate * drift.COM_TO_REAR_AXLE_M,
                "r": yaw_rate,
                "yaw": yaw,
            })
        result = drift.integrate_twist(rows, "gt", substep_s=0.001)[-1]
        total_yaw = 40 * dt * yaw_rate
        radius = speed / yaw_rate
        expected = (radius * __import__("math").sin(total_yaw),
                    radius * (1 - __import__("math").cos(total_yaw)))
        self.assertAlmostEqual(result[0], expected[0], delta=2.0e-5)
        self.assertAlmostEqual(result[1], expected[1], delta=2.0e-5)

    def test_reversing_yaw_rate_integrates_continuously(self) -> None:
        dt_s = 0.005
        rows = []
        for index in range(201):
            time_s = index * dt_s
            yaw = time_s if time_s <= 0.5 else 1.0 - time_s
            yaw_rate = 1.0 if time_s <= 0.5 else -1.0
            rows.append({
                "stamp_ns": index * 5_000_000,
                "u": 2.0,
                "v": 0.0,
                "r": yaw_rate,
                "yaw": yaw,
            })

        x, y = drift.integrate_twist(rows, "odom")[-1]

        self.assertAlmostEqual(x, 4.0 * math.sin(0.5), delta=1.0e-5)
        self.assertAlmostEqual(y, 4.0 * (1.0 - math.cos(0.5)), delta=1.0e-5)

    def test_yaw_seam_uses_short_wrapped_rotation(self) -> None:
        rows = [
            {"stamp_ns": 0, "u": 1.0, "v": 0.0, "r": 0.0,
             "yaw": math.pi - 0.01},
            {"stamp_ns": 25_000_000, "u": 1.0, "v": 0.0, "r": 0.0,
             "yaw": -math.pi + 0.01},
        ]

        x, y = drift.integrate_twist(rows, "odom")[-1]

        self.assertGreater(x, 0.0249)
        self.assertGreater(y, 0.0)
        self.assertLess(y, 0.0003)

    def test_com_to_rear_velocity_conversion_matches_high_slip_rear_twist(self) -> None:
        dt_s = 0.025
        gt_rows = []
        rear_rows = []
        for index in range(41):
            time_s = index * dt_s
            yaw_rate = 2.0
            v_com = 1.2
            v_rear = v_com - yaw_rate * drift.COM_TO_REAR_AXLE_M
            yaw = yaw_rate * time_s
            stamp_ns = index * 25_000_000
            gt_rows.append({
                "stamp_ns": stamp_ns, "u": 2.0, "v": v_com,
                "r": yaw_rate, "yaw": yaw,
            })
            rear_rows.append({
                "stamp_ns": stamp_ns, "u": 2.0, "v": v_rear,
                "r": yaw_rate, "yaw": yaw,
            })

        gt_path = drift.integrate_twist(gt_rows, "gt")
        rear_path = drift.integrate_twist(rear_rows, "odom")

        self.assertGreater(abs(v_rear), 0.5)
        for (gt_x, gt_y), (rear_x, rear_y) in zip(gt_path, rear_path):
            self.assertAlmostEqual(gt_x, rear_x, places=12)
            self.assertAlmostEqual(gt_y, rear_y, places=12)

    def test_jittered_and_missed_packets_use_elapsed_time(self) -> None:
        # The 47 ms gap stands in for one missed nominal 25 ms sample; the
        # integration must use stamped elapsed time, not a fixed packet dt.
        rows = [
            {"stamp_ns": 0, "u": 3.5, "v": 0.0, "r": 0.0, "yaw": 0.0},
            {"stamp_ns": 23_000_000, "u": 3.5, "v": 0.0, "r": 0.0,
             "yaw": 0.0},
            {"stamp_ns": 70_000_000, "u": 3.5, "v": 0.0, "r": 0.0,
             "yaw": 0.0},
            {"stamp_ns": 96_000_000, "u": 3.5, "v": 0.0, "r": 0.0,
             "yaw": 0.0},
        ]

        x, y = drift.integrate_twist(rows, "odom")[-1]

        self.assertAlmostEqual(x, 3.5 * 0.096, places=12)
        self.assertAlmostEqual(y, 0.0, places=12)

    def test_initial_pose_alignment_removes_only_fixed_rigid_transform(self) -> None:
        yaw = 0.7
        local = (1.25, -0.4)
        world = (3.0 + math.cos(yaw) * local[0] - math.sin(yaw) * local[1],
                 -2.0 + math.sin(yaw) * local[0] + math.cos(yaw) * local[1])
        aligned = drift.rotate_to_initial_heading(*world, 3.0, -2.0, yaw)
        self.assertAlmostEqual(aligned[0], local[0], places=12)
        self.assertAlmostEqual(aligned[1], local[1], places=12)


if __name__ == "__main__":
    unittest.main()
