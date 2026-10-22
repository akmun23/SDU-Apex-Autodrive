"""Regression tests for the source-exact Unity steering phase diagnostic."""

from __future__ import annotations

import math
import unittest

from tools import evaluate_mpc_bag as evaluator


class UnitySteeringSourceDiagnosticTest(unittest.TestCase):
    def test_normal_motion_uses_serialized_steering_rate(self) -> None:
        angle = evaluator._unity_vehicle_steering_angle_after(
            0.0, 0.5, 0.025)
        expected = math.radians(183.345993 * 0.025)
        self.assertAlmostEqual(angle, expected, places=10)

    def test_source_target_crossing_can_snap_when_unwinding(self) -> None:
        # This reproduces VehicleController.Steer's branch/clamp behavior;
        # it intentionally is not a symmetric rate-limited-servo result.
        angle = evaluator._unity_vehicle_steering_angle_after(
            0.4, 0.1, 0.001)
        self.assertAlmostEqual(
            angle, 0.1 * evaluator.MAX_STEERING_RAD, places=10)
        self.assertGreater(abs(angle - 0.4),
                           math.radians(183.345993 * 0.001))

    def test_command_is_clamped_like_autonomoussteering_setter(self) -> None:
        low = evaluator._unity_vehicle_steering_angle_after(
            0.0, -2.0, 0.025)
        high = evaluator._unity_vehicle_steering_angle_after(
            0.0, -1.0, 0.025)
        self.assertAlmostEqual(low, high, places=12)

    def test_nonfinite_or_nonpositive_interval_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            evaluator._unity_vehicle_steering_angle_after(0.0, 0.0, 0.0)
        with self.assertRaises(ValueError):
            evaluator._unity_vehicle_steering_angle_after(
                math.nan, 0.0, 0.025)


if __name__ == "__main__":
    unittest.main()
