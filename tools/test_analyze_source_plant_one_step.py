"""Tests for the development-only source-plant one-step report helpers."""

from __future__ import annotations

import math
import unittest

from tools import analyze_source_plant_one_step as report


class SourcePlantOneStepReportTest(unittest.TestCase):
    def test_metric_ignores_nonfinite_values_and_reports_tail(self) -> None:
        actual = report._metric([1.0, -2.0, math.nan, math.inf])
        self.assertEqual(actual["n"], 2)
        self.assertEqual(actual["bias"], -0.5)
        self.assertAlmostEqual(actual["rmse"], math.sqrt(2.5))
        self.assertEqual(actual["p95_abs"], 1.0)
        self.assertEqual(actual["max_abs"], 2.0)

    def test_regime_summary_keeps_sample_count_and_velocity_channels_separate(self) -> None:
        rows = [
            {"u_error_mps": -0.1, "production_odom_u_error_mps": -0.08,
             "initial_source_u_error_mps": -0.07,
             "v_com_error_mps": 0.02,
             "yaw_rate_error_radps": 0.03, "speed_mps": 2.0},
            {"u_error_mps": -0.3, "production_odom_u_error_mps": -0.25,
             "initial_source_u_error_mps": -0.11,
             "v_com_error_mps": -0.02,
             "yaw_rate_error_radps": -0.01, "speed_mps": 3.0},
        ]
        result = report._groups(
            rows, [("all", lambda row: row["speed_mps"] >= 2.0),
                   ("none", lambda row: row["speed_mps"] > 4.0)])
        self.assertEqual(result["all"]["count"], 2)
        self.assertAlmostEqual(
            result["all"]["forward_velocity_error_mps"]["bias"], -0.2)
        self.assertAlmostEqual(
            result["all"]["initial_source_state_forward_velocity_error_mps"]["bias"],
            -0.09)
        self.assertAlmostEqual(
            result["all"]["lateral_velocity_error_mps"]["rmse"], 0.02)
        self.assertEqual(result["none"]["count"], 0)
        self.assertIsNone(
            result["none"]["yaw_rate_error_radps"]["rmse"])

    def test_alternate_command_scores_are_separate_and_ignore_missing_feedback(self) -> None:
        rows = [
            {"u_error_mps": 0.10,
             "production_odom_u_error_mps": 0.08,
             "slot1_u_error_mps": 0.03,
             "next_feedback_u_error_mps": -0.01,
             "oracle_motion_u_error_mps": 0.02,
             "imu_trapezoid_from_source_u_error_mps": 0.04,
             "imu_trapezoid_from_GT_u_error_mps": -0.01,
             "v_com_error_mps": 0.0,
             "yaw_rate_error_radps": 0.0},
            {"u_error_mps": 0.12,
             "production_odom_u_error_mps": 0.09,
             "slot1_u_error_mps": 0.04,
             "next_feedback_u_error_mps": math.nan,
             "oracle_motion_u_error_mps": math.nan,
             "imu_trapezoid_from_source_u_error_mps": 0.05,
             "imu_trapezoid_from_GT_u_error_mps": 0.03,
             "v_com_error_mps": 0.0,
             "yaw_rate_error_radps": 0.0},
        ]
        result = report._groups(rows, [("all", lambda _row: True)])
        self.assertAlmostEqual(
            result["all"]["forward_velocity_error_mps"]["rmse"],
            math.sqrt((0.10**2 + 0.12**2) / 2))
        self.assertAlmostEqual(
            result["all"]["slot1_forward_velocity_error_mps"]["rmse"],
            math.sqrt((0.03**2 + 0.04**2) / 2))
        self.assertEqual(
            result["all"]["paired_previous_queue_slot"]["n"], 2)
        self.assertEqual(
            result["all"]["next_feedback_forward_velocity_error_mps"]["n"],
            1)
        self.assertEqual(
            result["all"]["paired_next_epoch_feedback_oracle"]["n"], 1)
        self.assertEqual(
            result["all"]["paired_same_epoch_GT_motion_state_oracle"]["n"],
            1)
        self.assertEqual(
            result["all"]["paired_IMU_trapezoid_from_GT_start_oracle"]["n"],
            2)

    def test_paired_alternative_comparison_uses_same_samples(self) -> None:
        rows = [
            {"u_error_mps": 0.10, "slot1_u_error_mps": 0.02,
             "next_feedback_u_error_mps": -0.01},
            {"u_error_mps": -0.10, "slot1_u_error_mps": -0.20,
             "next_feedback_u_error_mps": math.nan},
        ]
        slot1 = report._paired_alternative(rows, "slot1_u_error_mps")
        feedback = report._paired_alternative(
            rows, "next_feedback_u_error_mps")
        self.assertEqual(slot1["n"], 2)
        self.assertEqual(slot1["alternative_lower_abs_error"], 1)
        self.assertEqual(slot1["alternative_higher_abs_error"], 1)
        self.assertAlmostEqual(slot1["alternative_rmse_mps"],
                               math.sqrt((0.02**2 + 0.20**2) / 2))
        self.assertEqual(feedback["n"], 1)
        self.assertEqual(feedback["alternative_lower_abs_error"], 1)

    def test_correlation_handles_finite_pairs_and_constant_signals(self) -> None:
        result = report._correlation(
            [{"x": 1.0, "y": 2.0}, {"x": 2.0, "y": 4.0},
             {"x": math.nan, "y": 5.0}], "x", "y")
        self.assertEqual(result["n"], 2)
        self.assertAlmostEqual(result["pearson_r"], 1.0)
        constant = report._correlation(
            [{"x": 1.0, "y": 2.0}, {"x": 1.0, "y": 4.0}], "x", "y")
        self.assertIsNone(constant["pearson_r"])

    def test_elapsed_summary_separates_inherited_error_from_interval_growth(self) -> None:
        rows = [
            {"elapsed_from_first_epoch_s": 0.10,
             "initial_source_u_error_mps": -0.01,
             "u_error_mps": -0.05,
             "one_step_transition_error_mps": -0.04,
             "max_abs_source_rear_longitudinal_slip": 0.8,
             "rear_wheel_body_speed_mismatch_proxy_mps": 4.0,
             "queue_throttle_used": 0.2,
             "queue_throttle_delta": 0.1,
             "throttle_feedback_next_epoch": 0.2,
             "bridge_sent_throttle_start_epoch": 0.2},
            {"elapsed_from_first_epoch_s": 0.30,
             "initial_source_u_error_mps": -0.05,
             "u_error_mps": -0.051,
             "one_step_transition_error_mps": -0.001,
             "max_abs_source_rear_longitudinal_slip": 0.1,
             "rear_wheel_body_speed_mismatch_proxy_mps": 0.2,
             "queue_throttle_used": 0.3,
             "queue_throttle_delta": 0.0,
             "throttle_feedback_next_epoch": 0.3,
             "bridge_sent_throttle_start_epoch": 0.3},
        ]
        result = report._elapsed_time_summary(rows)
        first = result["0.00_0.25s"]
        second = result["0.25_0.50s"]
        self.assertEqual(first["count"], 1)
        self.assertAlmostEqual(
            first["initial_source_state_u_error_mps"]["bias"], -0.01)
        self.assertAlmostEqual(
            first["one_step_next_u_error_mps"]["bias"], -0.05)
        self.assertAlmostEqual(
            first["one_step_transition_error_mps"]["bias"], -0.04)
        self.assertEqual(second["count"], 1)
        self.assertAlmostEqual(
            second["one_step_transition_error_mps"]["bias"], -0.001)
        self.assertAlmostEqual(
            second["next_feedback_minus_queue_throttle"]["rmse"], 0.0)
        self.assertEqual(result["5.00s_plus"]["count"], 0)


if __name__ == "__main__":
    unittest.main()
