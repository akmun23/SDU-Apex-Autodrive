#!/usr/bin/env python3
"""Focused regression tests for exact-epoch MPC report export."""

from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from tools import evaluate_mpc_bag as evaluator
from tools import export_mpc_run_report as report


class MpcRunReportTest(unittest.TestCase):
    def test_publication_and_history_timings_are_separate(self) -> None:
        row = report._cycle_row(3., 0., {
            "middleware_publish_us": 53.2, "command_history_us": .7,
            "end_to_end_control_ms": 19.1}, {}, {})
        self.assertEqual(row["middleware_publish_us"], 53.2)
        self.assertEqual(row["command_history_us"], .7)
        self.assertEqual(row["callback_to_publish_return_ms"], 19.1)
        row = report._cycle_row(3., 0., {"middleware_publish_us": float('nan')}, {}, {})
        self.assertIsNone(row["middleware_publish_us"])
        self.assertIsNone(row["command_history_us"])

    def test_identical_lap_durations_are_separate_completion_events(self) -> None:
        counts = [{"bag_time": 0, "value": 0},
                  {"bag_time": 1, "value": 1},
                  {"bag_time": 2, "value": 1},
                  {"bag_time": 6.5, "value": 2}]
        durations = [{"bag_time": 0, "value": 0},
                     {"bag_time": 1.001, "value": 5.5},
                     {"bag_time": 2, "value": 5.5},
                     {"bag_time": 6.501, "value": 5.5}]
        self.assertEqual(report._distinct_lap_times(durations, None, counts),
                         [5.5, 5.5])
        self.assertEqual(report._distinct_lap_times(durations, 6.5, counts),
                         [5.5])

    def test_counter_join_rejects_stale_duration_and_counter_reset(self) -> None:
        counts = [{"bag_time": 0, "value": 2},
                  {"bag_time": 1, "value": 0},
                  {"bag_time": 2, "value": 1}]
        durations = [{"bag_time": 0, "value": 5.0},
                     {"bag_time": 1.999, "value": 5.0},
                     {"bag_time": 2.025, "value": 6.0}]
        self.assertEqual(report._distinct_lap_times(durations, None, counts),
                         [6.0])
        self.assertEqual(report._distinct_lap_times(durations[:2], None, counts),
                         [])

    def test_diagnostic_sidecar_preserves_candidate_nominal_and_plan_payloads(self) -> None:
        payload = {
            "status": "accepted_candidate_prefix",
            "action_source": "revalidated_candidate_prefix",
            "command_published": True,
            "source_stamp_ns": 123,
            "control_ros_stamp_ns": 124,
            "candidate_command_stamp_ns": 125,
            "issued_command_stamp_ns": 126,
            "candidate_controls": [[0.1, 0.2], [0.3, 0.4]],
            "nominal_controls": [[-0.1, 0.5], [-0.2, 0.6]],
            "certified_plan_snapshot": {"valid": True, "cursor": 7},
            "prior_plan_revalidation": {"attempted": True, "exact_valid": False},
        }
        rows = [
            {"bag_time": 2.0, "json": payload},
            {"bag_time": 1.0, "json": {"status": "waiting"}},
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "diagnostics.jsonl"
            count = report.write_diagnostic_sequence_sidecar(output, rows)
            records = [json.loads(line) for line in output.read_text().splitlines()]

        self.assertEqual(count, 2)
        self.assertEqual(records[0]["bag_receipt_epoch_s"], 1.0)
        self.assertEqual(records[1]["candidate_command_stamp_ns"], 125)
        self.assertEqual(records[1]["issued_command_stamp_ns"], 126)
        self.assertEqual(records[1]["payload"]["candidate_controls"],
                         payload["candidate_controls"])
        self.assertEqual(records[1]["payload"]["nominal_controls"],
                         payload["nominal_controls"])
        self.assertEqual(records[1]["payload"]["certified_plan_snapshot"],
                         payload["certified_plan_snapshot"])

    def test_drive_command_ros_stamp_keeps_exact_nanoseconds(self) -> None:
        expected = 1_792_611_522_369_873_606
        message = SimpleNamespace(header=SimpleNamespace(
            stamp=SimpleNamespace(sec=expected // 1_000_000_000,
                                  nanosec=expected % 1_000_000_000)))
        exact = evaluator.stamp_nanoseconds(message)
        indexed = report._drive_command_by_stamp([
            {"stamp": exact * 1e-9, "stamp_ns": exact,
             "steering": 0.1, "throttle": 0.3},
        ])
        self.assertEqual(exact, expected)
        self.assertIn(expected, indexed)

    def test_solver_termination_categories_remain_distinct(self) -> None:
        self.assertEqual(report._normalized_solver_category("converged"),
                         "optimal")
        self.assertEqual(report._normalized_solver_category("iteration_cap"),
                         "iteration_cap")
        self.assertEqual(report._normalized_solver_category("time_budget"),
                         "admm_time_budget")
        self.assertEqual(
            report._normalized_solver_category("deadline_before_admm"),
            "qp_build_deadline_skip")
        self.assertEqual(
            report._normalized_solver_category("numerical_or_solver_error"),
            "numerical_or_solver_error")
        self.assertEqual(report._normalized_solver_category("not_run"),
                         "not_run")
        self.assertEqual(report._normalized_solver_category({"bad": "value"}),
                         "unknown")

    def test_waiting_for_localization_is_known_not_run(self) -> None:
        row = report._cycle_row(
            1.0, 1.0,
            {"status": "waiting_for_localization", "solver": {"admm_status": -1}},
            {}, {},
        )
        self.assertEqual(row["solver_stop_reason"], "not_run")
        self.assertEqual(report._normalized_solver_category(
            row["solver_stop_reason"]), "not_run")

    def test_rejection_and_prefix_failure_details_are_exported(self) -> None:
        payload = {
            "status": "rejected_nonlinear_rollout",
            "reason": "accepted-plan prefix failed exact delayed-command rollout/corridor validation",
            "input_failure_reason": "none",
            "nonlinear_failure_reason": "corridor",
            "nonlinear_failure_stage": 8,
            "nonlinear_failure": {"stage": 8, "margin_m": 0.02},
            "candidate_prefix_revalidation": {
                "attempted": True, "exact_valid": False,
                "failure_stage": 3, "validation_status": 5,
                "validation_horizon": 3, "timing_rejected": False,
                "solver_converged": False,
                "solver_iteration_limited": True,
            },
            "prior_plan_revalidation": {
                "attempted": True, "exact_valid": False,
                "failure_stage": 2, "validation_status": 5,
                "validation_horizon": 3, "timing_rejected": False,
            },
            "full_prior_plan_suffix_revalidation": {
                "attempted": True, "exact_valid": False,
                "failure_stage": 11, "validation_status": 5,
                "validation_horizon": 27, "timing_rejected": False,
                "validation_us": 610.5,
            },
            "quality_aware_prefix_selection_on_rejection": True,
            "quality_aware_prefix_selection": {
                "attempted": True, "choice": "candidate",
                "candidate_valid": True, "prior_valid": True,
                "candidate_objective": 12.5, "prior_objective": 13.0,
                "candidate_minimum_corridor_slack_m": 0.08,
                "prior_minimum_corridor_slack_m": 0.07,
                "validation_us": 214.0,
                "candidate_prefix_controls": [[0.1, 0.4], [0.2, 0.3]],
                "prior_prefix_controls": [[-0.1, 0.2], [-0.2, 0.2]],
            },
            "solver": {"admm_status": 1},
        }
        row = report._cycle_row(3.0, 0.0, payload, {}, {})
        self.assertEqual(row["cycle_reason"], payload["reason"])
        self.assertEqual(row["nonlinear_failure_reason"], "corridor")
        self.assertEqual(row["nonlinear_failure_stage"], 8)
        self.assertEqual(row["nonlinear_failure_detail"],
                         {"stage": 8, "margin_m": 0.02})
        self.assertTrue(row["candidate_prefix_validation_attempted"])
        self.assertFalse(row["candidate_prefix_exact_valid"])
        self.assertEqual(row["candidate_prefix_failure_stage"], 3)
        self.assertEqual(row["candidate_prefix_validation_horizon"], 3)
        self.assertFalse(row["prior_plan_exact_valid"])
        self.assertEqual(row["prior_plan_failure_stage"], 2)
        self.assertTrue(row["full_prior_suffix_attempted"])
        self.assertFalse(row["full_prior_suffix_exact_valid"])
        self.assertEqual(row["full_prior_suffix_failure_stage"], 11)
        self.assertEqual(row["full_prior_suffix_validation_horizon"], 27)
        self.assertEqual(row["full_prior_suffix_validation_us"], 610.5)
        self.assertTrue(row["quality_prefix_policy_enabled"])
        self.assertTrue(row["quality_prefix_attempted"])
        self.assertEqual(row["quality_prefix_choice"], "candidate")
        self.assertTrue(row["quality_candidate_prefix_exact_valid"])
        self.assertTrue(row["quality_prior_prefix_exact_valid"])
        self.assertEqual(row["quality_candidate_prefix_objective"], 12.5)
        self.assertEqual(row["quality_prior_prefix_objective"], 13.0)
        self.assertEqual(row["quality_candidate_prefix_min_slack_m"], 0.08)
        self.assertEqual(row["quality_prior_prefix_min_slack_m"], 0.07)
        self.assertEqual(row["quality_prefix_validation_us"], 214.0)
        self.assertEqual(
            row["quality_candidate_prefix_controls_json"],
            "[[0.1,0.4],[0.2,0.3]]")
        self.assertEqual(
            row["quality_prior_prefix_controls_json"],
            "[[-0.1,0.2],[-0.2,0.2]]")

        compact = report._cycle_row(
            4.0, 0.0,
            {"status": "rejected", "reason": "prefix validation failed",
             "solver": {"admm_status": -1}}, {}, {})
        self.assertEqual(compact["cycle_reason"], "prefix validation failed")
        self.assertIsNone(compact["candidate_prefix_failure_stage"])

    def test_failure_reason_counts_use_diagnostic_fields(self) -> None:
        diagnostic = {"bag_time": 1.0, "json": {
            "status": "rejected_nonlinear_rollout",
            "reason": "no exact-feasible prefix",
            "nonlinear_failure_reason": "corridor",
            "nonlinear_failure_stage": 8,
            "candidate_prefix_revalidation": {
                "attempted": True, "exact_valid": False,
                "failure_stage": 3,
            },
            "solver": {"admm_status": -1},
        }}
        data = {
            evaluator.DIAGNOSTICS: [diagnostic],
            evaluator.CANDIDATE_DIAGNOSTICS: [],
            evaluator.BRIDGE_TIMING: [], evaluator.DRIVE_COMMAND: [],
            evaluator.LAST_LAP: [], evaluator.COLLISION: [],
            evaluator.BRIDGE_FAULT: [], evaluator.GT: [],
        }
        result, _ = report.make_report(
            Path.cwd(), Path("missing_bag"), data, None, None, 5.0,
            "live_closed_loop")
        self.assertEqual(result["summary"]["cycle_reason_counts"],
                         {"no exact-feasible prefix": 1})
        self.assertEqual(
            result["summary"]["nonlinear_failure_reason_counts"],
            {"corridor": 1})
        self.assertEqual(
            result["summary"]["candidate_prefix_failure_stage_counts"],
            {"3": 1})

    def test_quality_prefix_selection_is_summarized(self) -> None:
        data = {
            evaluator.DIAGNOSTICS: [
                {"bag_time": 0.5, "json": {
                    "status": "accepted_prior_plan",
                    "quality_aware_prefix_selection_on_rejection": True,
                    "quality_aware_prefix_selection": {
                        "attempted": True, "choice": "retained_plan",
                        "candidate_valid": True, "prior_valid": True,
                        "validation_us": 200.0,
                    },
                    "solver": {"admm_status": 1},
                }},
                {"bag_time": 0.6, "json": {
                    "status": "accepted_candidate_prefix",
                    "quality_aware_prefix_selection_on_rejection": True,
                    "quality_aware_prefix_selection": {
                        "attempted": True, "choice": "candidate",
                        "candidate_valid": True, "prior_valid": False,
                        "validation_us": 250.0,
                    },
                    "solver": {"admm_status": 1},
                }},
                {"bag_time": 0.7, "json": {
                    "status": "accepted_optimal",
                    "quality_aware_prefix_selection_on_rejection": True,
                    "quality_aware_prefix_selection": {"attempted": False},
                    "solver": {"admm_status": 0},
                }},
            ],
            evaluator.CANDIDATE_DIAGNOSTICS: [],
            evaluator.BRIDGE_TIMING: [], evaluator.DRIVE_COMMAND: [],
            evaluator.LAST_LAP: [], evaluator.COLLISION: [],
            evaluator.BRIDGE_FAULT: [], evaluator.GT: [],
        }
        result, cycles = report.make_report(
            Path.cwd(), Path("missing_bag"), data, None, None, 5.0,
            "live_closed_loop")
        self.assertEqual(len(cycles), 3)
        summary = result["summary"]
        self.assertTrue(summary["quality_prefix_policy_enabled"])
        self.assertEqual(summary["quality_prefix_attempt_count"], 2)
        self.assertEqual(summary["quality_prefix_choice_counts"], {
            "retained_plan": 1, "candidate": 1})
        self.assertEqual(summary["quality_prefix_both_exact_valid_count"], 1)
        self.assertEqual(summary["quality_prefix_validation_us"]["p50"], 200.0)

    def test_joins_only_exact_bridge_and_command_stamps(self) -> None:
        bridge = report._bridge_by_request_stamp([
            {"json": {"bridge_request_ros_stamp_ns": 1_000_000_000,
                      "request_sequence": 17,
                      "bridge_receive_ros_stamp_ns": 1_002_000_000}},
        ])
        commands = report._drive_command_by_stamp([
            {"stamp": 2.0, "steering": 0.2, "throttle": 0.4},
        ])
        payload = {
            "control_time_prediction": {"plant_source_stamp_ns": 1_000_000_000},
            "candidate_command_stamp_ns": 2_000_000_000,
            "issued_command_stamp_ns": 2_000_000_000,
            "status": "accepted_optimal", "command_published": True,
            "first_action": [0.2, 0.4],
            "previous_diagnostics_timing": {
                "queue_enqueue_wall_us": 12.0,
                "json_build_wall_us": 25.0,
            },
            "diagnostic_async_publisher": {
                "published_frames": 10,
                "replaced_frames": 2,
                "publish_failures": 0,
                "lower_priority": True,
                "last_serialization_wall_us": 30.0,
            },
            "solver": {
                "admm_status": 0, "time_budget_exhausted": False,
                "jacobian_threads_used": 2,
                "qp_build_us": 3500.0, "jacobian_wall_us": 8000.0,
                "jacobian_thread_cpu_us": 30000.0,
                "admm_us": 1500.0, "exact_rollout_us": 900.0,
                "jacobian_worker_cpu_us": [20.0, 10.0],
                "jacobian_worker_stage_wall_us": [5.0, 7.0],
                "jacobian_worker_stage_count": [2, 2],
                "jacobian_worker_os_tid": [101, 102],
            },
            "state": [0.1, -0.2, 5.0, 0.1, 0.3],
        }
        row = report._cycle_row(3.0, 0.0, payload, bridge, commands)
        self.assertEqual(row["bridge_request_sequence"], 17)
        self.assertEqual(row["scheduled_bridge_request_sequence"], None)
        self.assertEqual(row["actual_published_command_steering_rad"], 0.2)
        self.assertTrue(row["published_steering_message_matches_candidate"])
        self.assertIsNone(row["actual_unity_effect_epoch"])
        self.assertEqual(row["previous_diagnostic_queue_enqueue_wall_us"], 12.0)
        self.assertEqual(row["diagnostic_worker_replaced_frames"], 2)
        self.assertEqual(row["async_last_worker_serialization_wall_us"], 30.0)
        self.assertEqual(row["jacobian_slowest_worker_cpu_index"], 0)
        self.assertEqual(row["jacobian_slowest_worker_cpu_tid"], 101)
        self.assertEqual(row["jacobian_slowest_worker_cpu_us"], 20.0)
        self.assertEqual(row["jacobian_slowest_worker_wall_index"], 1)
        self.assertEqual(row["jacobian_slowest_worker_wall_tid"], 102)
        self.assertEqual(row["jacobian_slowest_worker_wall_us"], 7.0)
        self.assertEqual(row["qp_build_ms"], 3.5)
        self.assertEqual(row["jacobian_wall_ms"], 8.0)
        self.assertEqual(row["jacobian_worker_cpu_sum_ms"], 30.0)
        self.assertEqual(row["admm_ms"], 1.5)
        self.assertEqual(row["exact_rollout_ms"], 0.9)

        # ROS command payloads are float32; accept representational noise but
        # do not hide a meaningful output mismatch.
        commands[2_000_000_000]["steering"] = 0.2 + 2.0e-8
        row = report._cycle_row(3.0, 0.0, payload, bridge, commands)
        self.assertTrue(row["published_steering_message_matches_candidate"])
        commands[2_000_000_000]["steering"] = 0.2 + 1.0e-7
        row = report._cycle_row(3.0, 0.0, payload, bridge, commands)
        self.assertFalse(row["published_steering_message_matches_candidate"])

        # A one-nanosecond difference must remain explicitly unmatched.
        payload["candidate_command_stamp_ns"] = 2_000_000_001
        payload["issued_command_stamp_ns"] = 2_000_000_001
        row = report._cycle_row(3.0, 0.0, payload, bridge, commands)
        self.assertIsNone(row["actual_published_command_steering_rad"])
        self.assertIsNone(row["published_steering_message_matches_candidate"])

    def test_scores_only_through_first_collision(self) -> None:
        def diagnostic(stamp_ns: int, bag_time: float, progress: float):
            return {"bag_time": bag_time, "json": {
                "control_time_prediction": {"plant_source_stamp_ns": stamp_ns},
                "candidate_command_stamp_ns": stamp_ns + 25_000_000,
                "status": "accepted_candidate_prefix",
                "command_published": True,
                "progress_m": progress, "state": [0.2, 0.1, 4.0, 0.0, 0.2],
                "first_action": [0.1, 0.3], "nominal_first_action": [0.0, 0.3],
                "publication_gate_elapsed_ms": 12.0,
                "end_to_end_cycle_budget_ms": 20.0,
                "solver": {"admm_status": 1, "time_budget_exhausted": True,
                           "iterations": 100, "primal_residual": 0.4,
                           "dual_residual": 0.2,
                           "jacobian_threads_used": 2,
                           "jacobian_wall_us": 8000.0,
                           "jacobian_thread_cpu_us": 30000.0,
                           "jacobian_worker_cpu_us": [10000.0, 20000.0],
                           "jacobian_worker_stage_wall_us": [5000.0, 7000.0],
                           "jacobian_worker_stage_count": [2, 2],
                           "jacobian_worker_os_tid": [101, 102],
                           "rho_change_count": 3, "rho_start": 7.0,
                           "rho_final": 14.0, "rho_u_start": 7.0,
                           "rho_u_final": 14.0, "factorization_count": 4,
                           "regularization_count": 1,
                           "max_regularization": 1.0e-5},
            }}

        data = {
            evaluator.DIAGNOSTICS: [diagnostic(1_000_000_000, 0.5, 5.0),
                                   diagnostic(2_000_000_000, 1.5, 10.0)],
            evaluator.CANDIDATE_DIAGNOSTICS: [],
            evaluator.BRIDGE_TIMING: [], evaluator.DRIVE_COMMAND: [],
            evaluator.LAST_LAP: [
                {"bag_time": 0.0, "value": 0.0},
                {"bag_time": 1.0, "value": 6.1},
                {"bag_time": 1.1, "value": 6.1},
                {"bag_time": 2.0, "value": 6.0},
                {"bag_time": 2.1, "value": 6.0}],
            evaluator.COLLISION: [{"bag_time": 0.0, "value": 0},
                                  {"bag_time": 1.0, "value": 1}],
            evaluator.BRIDGE_FAULT: [], evaluator.GT: [],
        }
        result, cycles = report.make_report(
            Path.cwd(), Path("missing_bag"), data, None, None, 5.0,
            "live_closed_loop")
        self.assertEqual(len(cycles), 2)
        self.assertEqual(sum(row["score_eligible"] for row in cycles), 1)
        self.assertEqual(result["summary"]["cycles_scored"], 1)
        self.assertEqual(result["summary"]["short_prefix_accepts"], 1)
        self.assertEqual(result["summary"]["full_horizon_exact_accepts"], 0)
        self.assertEqual(
            result["summary"]["first_collision_bag_receipt_epoch_s"], 1.0)
        self.assertEqual(result["summary"]["first_collision_from_start_s"], 1.0)
        self.assertEqual(
            result["summary"]["scored_through_first_fault_from_start_s"], 1.0)
        self.assertEqual(result["summary"]["score_cutoff_reason"], "collision")
        self.assertEqual(cycles[0]["bag_time_from_start_s"], 0.5)
        self.assertEqual(result["experiment_regime"], "live_closed_loop")
        self.assertNotIn("cycles", result)
        self.assertEqual(result["cycle_detail_format"],
                         "companion CSV; one row per MPC diagnostic cycle")
        self.assertEqual(result["summary"]["lap_time_topic_samples"], 5)
        self.assertEqual(result["summary"]["recorded_lap_time_s"]["count"], 1)
        self.assertEqual(result["summary"]["recorded_lap_times_s"], [6.1])
        self.assertEqual(result["summary"]["jacobian_profile"]["wall_us"]["max"],
                         8000.0)
        self.assertEqual(
            result["summary"]["jacobian_profile"]["sum_worker_cpu_us"]["max"],
            30000.0)
        self.assertEqual(
            result["summary"]["solver_profile"]["rho_changes"]["max"], 3.0)
        self.assertEqual(
            result["summary"]["solver_profile"]["factorizations"]["max"], 4.0)
        self.assertNotIn("raw_diagnostic", cycles[0])

    def test_rejection_and_bridge_fault_cutoffs_truncate_lap_scores(self) -> None:
        def diagnostic(bag_time: float, status: str, published: bool):
            return {"bag_time": bag_time, "json": {
                "status": status, "command_published": published,
                "control_time_prediction": {"plant_source_stamp_ns": 1},
                "candidate_command_stamp_ns": 2,
                "issued_command_stamp_ns": 2,
                "first_action": [0.0, 0.2], "state": [0.0, 0.0, 1.0],
                "solver": {"admm_status": -1},
            }}

        common = {
            evaluator.CANDIDATE_DIAGNOSTICS: [],
            evaluator.BRIDGE_TIMING: [], evaluator.DRIVE_COMMAND: [],
            evaluator.COLLISION: [], evaluator.GT: [], evaluator.LAP: [],
            evaluator.LAST_LAP: [
                {"bag_time": 0.1, "value": 0.0},
                {"bag_time": 0.8, "value": 6.0},
                {"bag_time": 1.2, "value": 6.2}],
        }
        for rejected_status in ("rejected_input", "rejected_solver",
                                "rejected_nonlinear_rollout"):
            data = dict(common)
            data[evaluator.DIAGNOSTICS] = [
                diagnostic(0.5, "accepted_optimal", True),
                diagnostic(1.0, rejected_status, False),
                diagnostic(1.5, "accepted_optimal", True)]
            result, rows = report.make_report(
                Path.cwd(), Path("missing_bag"), data, None, None, 5.0)
            self.assertEqual(
                result["summary"]["first_unpublished_mpc_rejection_from_start_s"],
                0.9)
            self.assertEqual(
                result["summary"]["scored_through_first_fault_from_start_s"],
                0.9)
            self.assertEqual(
                result["summary"]["score_cutoff_reason"],
                "unpublished_mpc_rejection")
            self.assertEqual(sum(row["score_eligible"] for row in rows), 2)
            self.assertEqual(result["summary"]["recorded_lap_time_s"]["count"], 1)
            self.assertEqual(result["summary"]["recorded_lap_times_s"], [6.0])

        data = dict(common)
        data[evaluator.DIAGNOSTICS] = [
            diagnostic(0.5, "accepted_optimal", True),
            diagnostic(1.5, "accepted_optimal", True)]
        data[evaluator.BRIDGE_FAULT] = [
            {"bag_time": 0.2, "value": "bridge timeout"}]
        result, rows = report.make_report(
            Path.cwd(), Path("missing_bag"), data, None, None, 5.0)
        self.assertEqual(result["summary"]["first_bridge_fault_from_start_s"], 0.1)
        self.assertEqual(result["summary"]["score_cutoff_reason"], "bridge_fault")
        self.assertEqual(sum(row["score_eligible"] for row in rows), 0)
        self.assertIsNone(result["summary"]["recorded_lap_time_s"])

    def test_fault_cutoff_uses_bag_epoch_and_reports_elapsed_time(self) -> None:
        epoch = 1_700_000_000.0
        data = {
            evaluator.DIAGNOSTICS: [
                {"bag_time": epoch + 0.5, "json": {
                    "status": "accepted_optimal", "command_published": True,
                    "control_time_prediction": {"plant_source_stamp_ns": 1},
                    "first_action": [0.0, 0.3], "state": [0.0, 0.0, 1.0],
                    "solver": {"admm_status": 0},
                }},
                {"bag_time": epoch + 1.0, "json": {
                    "status": "rejected_solver", "command_published": False,
                    "control_time_prediction": {"plant_source_stamp_ns": 2},
                    "first_action": [0.0, 0.3], "state": [0.0, 0.0, 1.0],
                    "solver": {"admm_status": -1},
                }},
            ],
            evaluator.CANDIDATE_DIAGNOSTICS: [],
            evaluator.BRIDGE_TIMING: [], evaluator.DRIVE_COMMAND: [],
            evaluator.COLLISION: [{"bag_time": epoch, "value": 0}],
            evaluator.BRIDGE_FAULT: [],
            evaluator.GT: [{"bag_time": epoch}],
            evaluator.LAP: [],
            evaluator.LAST_LAP: [
                {"bag_time": epoch, "value": 0.0},
                {"bag_time": epoch + 0.8, "value": 6.0},
                {"bag_time": epoch + 1.2, "value": 6.2},
            ],
        }
        result, rows = report.make_report(
            Path.cwd(), Path("missing_bag"), data, None, None, 5.0)
        summary = result["summary"]
        self.assertEqual(summary["score_cutoff_reason"],
                         "unpublished_mpc_rejection")
        self.assertEqual(summary["scored_through_first_fault_epoch_s"],
                         epoch + 1.0)
        self.assertAlmostEqual(
            summary["scored_through_first_fault_from_start_s"], 1.0, places=5)
        self.assertEqual(summary["recorded_lap_time_s"]["count"], 1)
        self.assertEqual(sum(row["score_eligible"] for row in rows), 2)


if __name__ == "__main__":
    unittest.main()
