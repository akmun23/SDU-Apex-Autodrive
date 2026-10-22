"""Unit tests for the Unity fixed-step rollout analysis helpers."""

from __future__ import annotations

import math
import unittest

from tools import analyze_unity_fixedstep_rollout as analysis


class UnityFixedStepRolloutTest(unittest.TestCase):
    def test_inverse_rotation_recovers_local_forward_from_world_velocity(self) -> None:
        root_half = math.sqrt(0.5)
        # Unity yaw +90 degrees maps local +Z forward onto world +X.
        u, v, r = analysis.unity_world_to_body(
            (0.0, root_half, 0.0, root_half),
            (4.0, 0.0, 0.0),
            (0.0, -1.25, 0.0),
        )
        self.assertAlmostEqual(u, 4.0, places=12)
        self.assertAlmostEqual(v, 0.0, places=12)
        self.assertAlmostEqual(r, 1.25, places=12)

    def test_local_lateral_axis_sign_matches_ros_left(self) -> None:
        u, v, r = analysis.unity_world_to_body(
            (0.0, 0.0, 0.0, 1.0),
            (-0.5, 0.0, 3.0),
            (0.0, 0.2, 0.0),
        )
        self.assertAlmostEqual(u, 3.0)
        self.assertAlmostEqual(v, 0.5)
        self.assertAlmostEqual(r, -0.2)

    def test_transition_residual_removes_initial_state_error(self) -> None:
        # Model starts 0.2 m/s low; if both model and truth gain 0.5 m/s,
        # the transition itself is exact despite a 0.2 endpoint error.
        self.assertAlmostEqual(
            analysis.transition_residual(2.3, 1.8, 2.5, 2.0), 0.0)

    def test_zero_quaternion_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "nonzero"):
            analysis.unity_world_to_body((0.0, 0.0, 0.0, 0.0),
                                         (0.0, 0.0, 0.0),
                                         (0.0, 0.0, 0.0))

    def test_packet_interval_summary_keeps_step_count_arms_separate(self) -> None:
        rows = [
            {
                "actual_fixed_steps": 24,
                "held_25_u_absolute_error": 0.0,
                "held_actual_steps_u_absolute_error": 0.0,
                "trace_actual_steps_u_absolute_error": 0.0,
                "held_25_v_com_absolute_error": 0.0,
                "held_actual_steps_v_com_absolute_error": 0.0,
                "trace_actual_steps_v_com_absolute_error": 0.0,
                "held_25_yaw_rate_absolute_error": 0.0,
                "held_actual_steps_yaw_rate_absolute_error": 0.0,
                "trace_actual_steps_yaw_rate_absolute_error": 0.0,
                "held_25_u_transition_error": 0.2,
                "held_actual_steps_u_transition_error": 0.1,
                "trace_actual_steps_u_transition_error": 0.05,
                "held_25_v_com_transition_error": 0.0,
                "held_actual_steps_v_com_transition_error": 0.0,
                "trace_actual_steps_v_com_transition_error": 0.0,
                "held_25_yaw_rate_transition_error": 0.0,
                "held_actual_steps_yaw_rate_transition_error": 0.0,
                "trace_actual_steps_yaw_rate_transition_error": 0.0,
            }
        ]
        metrics = analysis._packet_interval_metrics(rows)
        self.assertEqual(metrics["count"], 1)
        self.assertAlmostEqual(
            metrics["u"]["held_25"]["transition_error"]["rmse"], 0.2)
        self.assertAlmostEqual(
            metrics["u"]["held_actual_steps"]["transition_error"]["rmse"],
            0.1)
        self.assertAlmostEqual(
            metrics["u"]["trace_actual_steps"]["transition_error"]["rmse"],
            0.05)
        self.assertEqual(metrics["by_actual_fixed_step_count"]["24"]["count"], 1)

    def test_command_epoch_summary_counts_application_offset(self) -> None:
        rows = [
            {
                "start_trace_command_sequence": 8,
                "event_command_sequence": 9,
                "event_command_apply_offset_steps": 1,
                "end_trace_command_sequence": 9,
                "requested_received_throttle_error": 0.0,
                "requested_received_steering_error": 0.0,
                "drive_torque_per_normalized_throttle_nm": 428.0,
            },
            {
                "start_trace_command_sequence": 9,
                "event_command_sequence": 10,
                "event_command_apply_offset_steps": 0,
                "end_trace_command_sequence": 10,
                "requested_received_throttle_error": 0.0,
                "requested_received_steering_error": 0.0,
                "drive_torque_per_normalized_throttle_nm": 428.0,
            },
        ]
        metrics = analysis._command_epoch_metrics(rows)
        self.assertEqual(
            metrics["trace_command_sequence_at_snapshot_minus_event_sequence"],
            {"-1": 2})
        self.assertEqual(
            metrics["first_fixedupdate_offset_for_event_command"],
            {"0": 1, "1": 1})
        self.assertEqual(
            metrics["trace_command_sequence_at_next_snapshot_minus_event_sequence"],
            {"0": 2})
        self.assertAlmostEqual(
            metrics["positive_drive_torque_per_normalized_throttle_nm"]["mean"],
            428.0)

    def test_fixedstep_metrics_separate_absolute_and_transition_error(self) -> None:
        row = {
            "packet_throttle_net_delta": -0.2,
            "packet_start_steering_rad": 0.02,
            "model_u_absolute_error": -0.12,
            "model_u_transition_error": 0.0005,
            "model_v_com_absolute_error": 0.01,
            "model_v_com_transition_error": -0.0002,
            "model_yaw_rate_absolute_error": 0.03,
            "model_yaw_rate_transition_error": 0.001,
        }
        for wheel in analysis.WHEEL_RPM_FIELDS:
            row[f"model_{wheel}_omega_phase_error_radps"] = 0.25

        summary = analysis._fixedstep_metrics([row])
        self.assertEqual(summary["count"], 1)
        near_straight = summary["groups"]["near_straight"]
        self.assertEqual(near_straight["count"], 1)
        self.assertAlmostEqual(near_straight["u"]["absolute_error"]["rmse"],
                               0.12)
        self.assertAlmostEqual(near_straight["u"]["transition_error"]["rmse"],
                               0.0005)
        self.assertAlmostEqual(
            summary["groups"]["throttle_falling"][
                "wheel_omega_phase_comparison_radps"]["rl_rpm"]["rmse"],
            0.25)


if __name__ == "__main__":
    unittest.main()
