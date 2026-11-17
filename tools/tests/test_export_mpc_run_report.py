#!/usr/bin/env python3
"""Focused regression tests for exact-epoch MPC report export."""

from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from tools.analysis import evaluate_mpc_bag as evaluator
from tools.analysis import export_mpc_run_report as report


class MpcRunReportTest(unittest.TestCase):
    def test_fixed_commands_are_not_scored_as_unexecuted_plans(self):
        rows = [dict(xy_m=.02, path_max_xy_m=.03, end_u_error_mps=.01,
                     end_v_error_mps=.002, end_r_error_radps=.02),
                dict(xy_m=.04, path_max_xy_m=.06, progress_m=float('nan'), end_u_error_mps=-.01,
                     end_v_error_mps=-.002, end_r_error_radps=-.02)]
        result = report.summarize_fixed_command_rollout(rows)
        self.assertEqual(result['windows'], 2)
        self.assertAlmostEqual(result['endpoint_xy_rmse_m'], (.001)**.5)
        self.assertEqual(result['full_path_xy_max_m'], .06)
        self.assertEqual(result['paths_over_5cm'], 1)
        self.assertIsNone(result['worst_window']['progress_m'])
        json.dumps(result, allow_nan=False)
        self.assertIn('Not an unexecuted MPC-plan test', result['interpretation'])
        self.assertEqual(report.summarize_fixed_command_rollout([])['status'], 'unavailable')

    def test_lap_tracking_uses_counter_boundaries_not_wall_time_guess(self) -> None:
        laps = [dict(bag_time=10., value=0), dict(bag_time=20., value=1),
                dict(bag_time=30., value=2)]
        def cycle(t, error):
            return dict(bag_receipt_epoch_s=t, score_eligible=True, e_y_m=error,
                        e_psi_rad=error/2, speed_error_mps=error/4,
                        callback_to_gate_ms=7., status="accepted_feasible")
        result = report._per_lap_tracking(laps, [cycle(11., .1),cycle(12., -.1),
                                                   cycle(21., .02),cycle(22., .02),
                                                   cycle(31., .2)])
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["official_lap_index"], 1)
        self.assertEqual(result[1]["official_lap_index"], 2)
        self.assertAlmostEqual(result[0]["cross_track_rmse_m"], .1)
        self.assertAlmostEqual(result[1]["cross_track_rmse_m"], .02)
        self.assertEqual(result[1]["diagnostic_cycles"], 2)


    def test_publication_gate_and_emission_timings_are_separate(self) -> None:
        row = report._cycle_row(3., 0., {
            "publication_gate_elapsed_ms": 18.0,
            "end_to_end_control_ms": 19.1}, {}, {})
        self.assertEqual(row["callback_to_gate_ms"], 18.0)
        self.assertEqual(row["callback_to_publish_return_ms"], 19.1)
        row = report._cycle_row(3., 0., {"publication_gate_elapsed_ms": float('nan')}, {}, {})
        self.assertIsNone(row["callback_to_gate_ms"])

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
        self.assertEqual(report._distinct_lap_times(durations), [])

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

    def test_diagnostic_sidecar_preserves_candidate_nominal_and_source_controls(self) -> None:
        payload = {
            "status": "accepted_feasible",
            "command_published": True,
            "source_stamp_ns": 123,
            "control_ros_stamp_ns": 124,
            "candidate_command_stamp_ns": 125,
            "issued_command_stamp_ns": 126,
            "candidate_controls": [[0.1, 0.2], [0.3, 0.4]],
            "nominal_controls": [[-0.1, 0.5], [-0.2, 0.6]],
            "source_optimized_controls": [[0.2, 0.3], [0.4, 0.5]],
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
        self.assertEqual(records[1]["payload"]["source_optimized_controls"],
                         payload["source_optimized_controls"])

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

    def test_waiting_for_localization_is_known_not_run(self) -> None:
        row = report._cycle_row(
            1.0, 1.0,
            {"status": "waiting_for_localization"},
            {}, {},
        )
        self.assertEqual(row["solver_stop_reason"], "not_run")

    def test_failure_reason_counts_use_diagnostic_fields(self) -> None:
        diagnostic = {"bag_time": 1.0, "json": {
            "status": "rejected_nonlinear_rollout",
            "reason": "source_sqp_unqualified",
            "nonlinear_failure_reason": "corridor",
            "nonlinear_failure_stage": 8,
            "solver": {"termination": "primal_infeasible"},
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
                         {"source_sqp_unqualified": 1})
        self.assertEqual(
            result["summary"]["nonlinear_failure_reason_counts"],
            {"corridor": 1})
        self.assertEqual(result["summary"]["solver_termination_counts"],
                         {"primal_infeasible": 1})

    def test_joins_only_exact_bridge_and_command_stamps(self) -> None:
        bridge = report._bridge_by_request_stamp([
            {"json": {"bridge_request_ros_stamp_ns": 1_000_000_000,
                      "request_sequence": 17,
                      "bridge_receive_ros_stamp_ns": 1_002_000_000}},
        ])
        commands = report._drive_command_by_stamp([
            {"stamp": 2.0, "stamp_ns": 2_000_000_000, "steering": 0.2, "throttle": 0.4},
        ])
        payload = {
            "control_time_prediction": {"plant_source_stamp_ns": 1_000_000_000},
            "candidate_command_stamp_ns": 2_000_000_000,
            "issued_command_stamp_ns": 2_000_000_000,
            "status": "accepted_feasible", "command_published": True,
            "first_action": [0.2, 0.4],
            "solve_us": 12000.0, "nominal_build_us": 500.0,
            "sqp_us": 11000.0, "validation_us": 500.0,
            "solver": {
                "termination": "feasible_solution_available",
                "sqp_iterations_attempted": 4, "sqp_iterations_accepted": 3,
                "source_exact_nlp_source_trials": 8,
            },
            "state": [0.1, -0.2, 5.0, 0.1, 0.3],
        }
        row = report._cycle_row(3.0, 0.0, payload, bridge, commands)
        self.assertEqual(row["bridge_request_sequence"], 17)
        self.assertEqual(row["scheduled_bridge_request_sequence"], None)
        self.assertEqual(row["actual_published_command_steering_rad"], 0.2)
        self.assertTrue(row["published_steering_message_matches_candidate"])
        self.assertIsNone(row["actual_unity_effect_epoch"])
        self.assertEqual(row["node_solve_ms"], 12.0)
        self.assertEqual(row["nominal_build_ms"], .5)
        self.assertEqual(row["sqp_ms"], 11.0)
        self.assertEqual(row["nonlinear_validation_ms"], .5)
        self.assertEqual(row["sqp_iterations_attempted"], 4)
        self.assertEqual(row["source_trials"], 8)

        # Accept numeric representation noise but
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
                "status": "accepted_feasible",
                "command_published": True,
                "progress_m": progress, "state": [0.2, 0.1, 4.0, 0.0, 0.2],
                "first_action": [0.1, 0.3], "nominal_first_action": [0.0, 0.3],
                "publication_gate_elapsed_ms": 12.0,
                "end_to_end_cycle_budget_ms": 20.0,
                "solver": {"termination": "feasible_solution_available",
                           "sqp_iterations_attempted": 4},
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
            evaluator.LAP: [{"bag_time": 0.0, "value": 0},
                            {"bag_time": 1.0, "value": 1},
                            {"bag_time": 2.0, "value": 2}],
            evaluator.BRIDGE_FAULT: [], evaluator.GT: [],
        }
        result, cycles = report.make_report(
            Path.cwd(), Path("missing_bag"), data, None, None, 5.0,
            "live_closed_loop")
        self.assertEqual(len(cycles), 2)
        self.assertEqual(sum(row["score_eligible"] for row in cycles), 1)
        self.assertEqual(result["summary"]["cycles_scored"], 1)
        self.assertEqual(result["summary"]["full_horizon_exact_accepts"], 1)
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
        self.assertEqual(result["summary"]["solver_termination_counts"],
                         {"feasible_solution_available": 1})
        self.assertNotIn("raw_diagnostic", cycles[0])

    def test_rejection_and_bridge_fault_cutoffs_truncate_lap_scores(self) -> None:
        def diagnostic(bag_time: float, status: str, published: bool):
            return {"bag_time": bag_time, "json": {
                "status": status, "command_published": published,
                "control_time_prediction": {"plant_source_stamp_ns": 1},
                "candidate_command_stamp_ns": 2,
                "issued_command_stamp_ns": 2,
                "first_action": [0.0, 0.2], "state": [0.0, 0.0, 1.0],
                "solver": {"termination": "invalid_model"},
            }}

        common = {
            evaluator.CANDIDATE_DIAGNOSTICS: [],
            evaluator.BRIDGE_TIMING: [], evaluator.DRIVE_COMMAND: [],
            evaluator.COLLISION: [], evaluator.GT: [],
            evaluator.LAP: [{"bag_time": 0.1, "value": 0},
                            {"bag_time": 0.8, "value": 1},
                            {"bag_time": 1.2, "value": 2}],
            evaluator.LAST_LAP: [
                {"bag_time": 0.1, "value": 0.0},
                {"bag_time": 0.8, "value": 6.0},
                {"bag_time": 1.2, "value": 6.2}],
        }
        for rejected_status in ("rejected_input", "rejected_solver",
                                "rejected_nonlinear_rollout"):
            data = dict(common)
            data[evaluator.DIAGNOSTICS] = [
                diagnostic(0.5, "accepted_feasible", True),
                diagnostic(1.0, rejected_status, False),
                diagnostic(1.5, "accepted_feasible", True)]
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
            diagnostic(0.5, "accepted_feasible", True),
            diagnostic(1.5, "accepted_feasible", True)]
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
                    "status": "accepted_feasible", "command_published": True,
                    "control_time_prediction": {"plant_source_stamp_ns": 1},
                    "first_action": [0.0, 0.3], "state": [0.0, 0.0, 1.0],
                    "solver": {"termination": "feasible_solution_available"},
                }},
                {"bag_time": epoch + 1.0, "json": {
                    "status": "rejected_solver", "command_published": False,
                    "control_time_prediction": {"plant_source_stamp_ns": 2},
                    "first_action": [0.0, 0.3], "state": [0.0, 0.0, 1.0],
                    "solver": {"termination": "invalid_model"},
                }},
            ],
            evaluator.CANDIDATE_DIAGNOSTICS: [],
            evaluator.BRIDGE_TIMING: [], evaluator.DRIVE_COMMAND: [],
            evaluator.COLLISION: [{"bag_time": epoch, "value": 0}],
            evaluator.BRIDGE_FAULT: [],
            evaluator.GT: [{"bag_time": epoch}],
            evaluator.LAP: [{"bag_time": epoch, "value": 0},
                            {"bag_time": epoch + 0.8, "value": 1},
                            {"bag_time": epoch + 1.2, "value": 2}],
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
