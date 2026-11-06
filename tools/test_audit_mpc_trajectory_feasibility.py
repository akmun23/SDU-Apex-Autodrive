import csv
import io
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import audit_mpc_trajectory_feasibility as audit


class TrajectoryAuditTests(unittest.TestCase):
    def setUp(self):
        self.points = [
            {"s": 0.0, "x": 1.0, "y": 0.0, "heading": math.pi / 2,
             "curvature": 1.0, "speed": 2.0, "acceleration": 0.0,
             "left": 0.8, "right": 0.6},
            {"s": math.pi / 2, "x": 0.0, "y": 1.0, "heading": math.pi,
             "curvature": 1.0, "speed": 2.0, "acceleration": 0.0,
             "left": 0.8, "right": 0.6},
            {"s": math.pi, "x": -1.0, "y": 0.0, "heading": 3 * math.pi / 2,
             "curvature": 1.0, "speed": 2.0, "acceleration": 0.0,
             "left": 0.8, "right": 0.6},
            {"s": 3 * math.pi / 2, "x": 0.0, "y": -1.0, "heading": 2 * math.pi,
             "curvature": 1.0, "speed": 2.0, "acceleration": 0.0,
             "left": 0.8, "right": 0.6},
        ]
        self.length = 2 * math.pi

    def test_periodic_interpolation_closes_to_first_sample(self):
        at_start = audit.sample_path(self.points, self.length, 0.0)
        at_end = audit.sample_path(self.points, self.length, self.length)
        self.assertAlmostEqual(at_start["x"], at_end["x"])
        self.assertAlmostEqual(at_start["y"], at_end["y"])
        self.assertAlmostEqual(at_start["heading"], at_end["heading"])
        self.assertAlmostEqual(at_start["curvature"], at_end["curvature"])

    def test_heading_aware_footprint_grows_with_yaw_error(self):
        left, right = audit.footprint_extents(0.2)
        self.assertGreater(left, right)
        self.assertEqual(audit.footprint_extents(-0.2), (right, left))
        self.assertEqual(audit.footprint_extents(0.0), (0.1365, 0.1365))

    def test_footprint_support_matches_all_four_rotated_corners(self):
        for step in range(-180, 181):
            angle = math.radians(step)
            lateral = [x * math.sin(angle) + y * math.cos(angle)
                       for x in (-audit.REAR_OVERHANG_M, audit.REAR_AXLE_TO_FRONT_M)
                       for y in (-audit.VEHICLE_WIDTH_M / 2, audit.VEHICLE_WIDTH_M / 2)]
            left, right = audit.footprint_extents(angle)
            self.assertAlmostEqual(left, max(lateral), places=12)
            self.assertAlmostEqual(right, -min(lateral), places=12)

    def test_geometry_reports_arc_bins_and_physical_clearance(self):
        summary, bins = audit.geometry_audit(
            self.points, self.length, 0.5, 0.15, 0.001)
        self.assertEqual(summary["point_count"], 4)
        self.assertTrue(bins)
        self.assertAlmostEqual(
            summary["minimum_centerline_corridor_slack_after_footprint_m"],
            0.6 - 0.15 - 0.1365, places=8)
        self.assertTrue(summary["geometric_centerline_feasible_at_requested_clearance"])

    def test_runtime_corridor_profile_is_read_explicitly(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mpc.yaml"
            path.write_text(
                "corridor_margin_m: 0.05 # physical wall buffer\n"
                "first_prediction_corridor_margin_m: 0.04\n"
                "nonlinear_corridor_tolerance_m: 0.005\n"
                "corridor_preview_halfwidth_m: 0.10\n")
            margins = audit.read_runtime_corridor_profile(path)
        self.assertEqual(margins, {
            "corridor_margin_m": 0.05,
            "first_prediction_corridor_margin_m": 0.04,
            "nonlinear_corridor_tolerance_m": 0.005,
            "corridor_preview_halfwidth_m": 0.10,
        })

    def test_runtime_objective_profile_reads_production_cost_weights(self):
        profile = audit.read_runtime_objective_profile(
            Path("f1tenth_mpc/config/mpc_competition.yaml"))
        self.assertEqual(profile["weight_e_y"], 4000.0)
        self.assertEqual(profile["weight_steering_slew"], 2000.0)
        self.assertEqual(profile["terminal_multiplier"], 3.0)

    def test_corridor_preview_includes_internal_knots_and_wraps_at_seam(self):
        points = [dict(point) for point in self.points]
        points[1]["left"] = 0.2
        left, right = audit.sample_corridor_bounds(
            points, self.length, 0.0, 2.0)
        self.assertAlmostEqual(left, 0.2)
        self.assertAlmostEqual(right, 0.6)

        points[0]["right"] = 0.15
        left, right = audit.sample_corridor_bounds(
            points, self.length, self.length - 0.05, 0.2)
        self.assertAlmostEqual(right, 0.15)

    def test_trajectory_loader_skips_comment_header_and_checks_periodicity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "line.csv"
            path.write_text(
                "# s_m,x_m,y_m,psi_rad,kappa_radpm,velocity_mps,"
                "acceleration_mps2,d_left_m,d_right_m\n"
                "0,1,0,1.57079632679,1,2,0,.8,.6\n"
                "1.57079632679,0,1,3.14159265359,1,2,0,.8,.6\n"
                "3.14159265359,-1,0,4.71238898038,1,2,0,.8,.6\n"
                "4.71238898038,0,-1,6.28318530718,1,2,0,.8,.6\n")
            points, length = audit.load_trajectory(path)
        self.assertEqual(len(points), 4)
        self.assertAlmostEqual(length, 1.5 * math.pi + math.sqrt(2.0), places=8)

    def test_terminal_predecessors_select_full_horizon_plans_only(self):
        def row(status, controls=True, hidden=True):
            return {
                "status": status,
                "nominal_controls": [[0.0, 0.2]] * (30 if controls else 29),
                "candidate_controls": [[0.0, 0.2]] * (30 if controls else 29),
                "plant_internal_state": [0.0] * (22 if hidden else 21),
            }
        rows = [row("accepted_optimal") for _ in range(4)]
        rows += [row("accepted_candidate_prefix") for _ in range(3)]
        rows += [row("rejected_nonlinear_rollout")]
        selected = audit.select_terminal_predecessors(rows, 4)
        self.assertEqual(len(selected), 4)
        self.assertTrue(all(item["status"] == "accepted_optimal"
                            for item in selected))

    def test_terminal_prefix_sequence_is_kept_separate_from_accepted_plans(self):
        prefix = {
            "status": "accepted_candidate_prefix",
            "candidate_controls": [[0.0, 0.2]] * 30,
            "plant_internal_state": [0.0] * 22,
        }
        rows = [{"status": "accepted_optimal"}, prefix,
                {"status": "rejected_nonlinear_rollout"}]
        self.assertIs(audit.select_last_candidate_prefix_before_terminal_failure(rows),
                      prefix)

    def test_nearest_regime_selection_reports_speed_and_curvature_distance(self):
        rows = []
        for stamp, speed, curvature in ((1.0, 4.0, -0.8),
                                        (2.0, 3.0, 0.9),
                                        (2.1, 4.1, -0.82)):
            rows.append({
                "status": "accepted_optimal",
                "bag_time": stamp,
                "state": [0.0, 0.0, speed, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.2],
                "path_curvature_per_m": curvature,
                "candidate_controls": [[0.0, 0.2]] * 30,
                "plant_internal_state": [0.0] * 22,
            })
        selected = audit.select_nearest_regime(rows, 2, 4.0, 0.8)
        self.assertEqual(len(selected), 2)
        self.assertEqual(selected[0]["bag_time"], 1.0)
        self.assertAlmostEqual(
            selected[0]["regime_match"]["absolute_curvature_error_per_m"], 0.0)

    def test_source_state_reorders_mpc_hidden_state_to_unity_abi(self):
        diag = {
            "state": [0.1, -0.2, 3.0, 0.4, 0.5, 0.01, 0.02, 0.03, 0.4, 0.5],
            "plant_internal_state": list(map(float, range(22))),
            "plant_state_flags": [1, 1, 1],
        }
        state, e_y, suspension_initialized = audit._plant_state_from_diag(diag)
        self.assertEqual(len(state), 26)
        self.assertAlmostEqual(state[0], 3.0)
        self.assertAlmostEqual(state[1], 0.4 + 0.5 * audit.COM_FORWARD_FROM_REAR_M)
        self.assertEqual(state[3:13], list(map(float, range(10))))
        self.assertAlmostEqual(state[13], 0.03)
        self.assertEqual(state[14:26], list(map(float, range(10, 22))))
        self.assertAlmostEqual(e_y, 0.1)
        self.assertEqual(suspension_initialized, 1)

    def test_source_rollout_uses_current_optimizer_command_at_each_stage(self):
        self.assertEqual(audit.MPC_COMMAND_DELAY_STEPS, 0)
        self.assertEqual(audit.source_rollout_command([0.23, 0.41]),
                         (0.23, 0.41))
        with patch.object(audit, "MPC_COMMAND_DELAY_STEPS", 1):
            with self.assertRaisesRegex(RuntimeError, "command chronology"):
                audit.source_rollout_command([0.23, 0.41])

    def test_fixed_throttle_sweep_rejects_unavailable_brake_mode(self):
        audit.validate_fixed_throttle_grid([0.001, 0.01, 0.1, 1.0])
        for values in ([0.0], [-0.1], [1.01], [math.nan], [math.inf]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                audit.validate_fixed_throttle_grid(list(values))

    def test_blended_continuation_uses_shared_prefix_and_holds_prior_tail(self):
        candidate = [[float(i), 0.8] for i in range(30)]
        prior = [[-float(i), 0.2] for i in range(2)]
        blended = audit.blend_candidate_and_prior_controls(
            candidate, prior, 0.5)
        self.assertEqual(len(blended), 30)
        self.assertEqual(blended[0], [0.0, 0.5])
        self.assertEqual(blended[1], [0.0, 0.5])
        self.assertEqual(blended[2], [0.5, 0.5])
        self.assertEqual(blended[-1], [14.0, 0.5])

    def test_dynamic_feasibility_bins_report_tested_witnesses_only(self):
        comparisons = [
            {"progress_m": 1.2, "blended_continuation_trials": [
                {"candidate_weight": 0.25, "full_horizon_source_valid": True,
                 "minimum_corridor_slack_m": 0.04},
                {"candidate_weight": 0.50, "full_horizon_source_valid": True,
                 "minimum_corridor_slack_m": -0.002},
                {"candidate_weight": 0.75, "full_horizon_source_valid": False,
                 "minimum_corridor_slack_m": -0.1},
            ]},
            {"progress_m": 1.4, "blended_continuation_trials": [
                {"candidate_weight": 0.25, "full_horizon_source_valid": True,
                 "minimum_corridor_slack_m": 0.01},
            ]},
            {"progress_m": 2.1, "blended_continuation_trials": [
                {"candidate_weight": 0.50, "full_horizon_source_valid": True,
                 "minimum_corridor_slack_m": -0.001},
            ]},
        ]
        bins = audit.summarize_dynamic_feasibility_by_progress(
            comparisons, 1.0)
        self.assertEqual(len(bins), 2)
        self.assertEqual(bins[0]["saved_state_count"], 2)
        self.assertEqual(bins[0]["source_valid_trial_count"], 3)
        self.assertEqual(bins[0]["strict_positive_slack_witness_count"], 2)
        self.assertEqual(bins[0]["by_candidate_weight_valid_counts"], {
            "0.25": 2, "0.50": 1, "0.75": 0,
        })
        self.assertEqual(
            bins[0]["classification"],
            "tested_source_plant_feasible_witness_found")
        self.assertEqual(bins[1]["strict_positive_slack_witness_count"], 0)
        self.assertEqual(
            bins[1]["classification"],
            "no_positive_witness_in_tested_blends_not_global_infeasibility")

    def test_dynamic_feasibility_bins_require_positive_width(self):
        with self.assertRaises(ValueError):
            audit.summarize_dynamic_feasibility_by_progress([], 0.0)

    def test_blend_weight_grid_is_sorted_and_rejects_invalid_values(self):
        self.assertEqual(audit.validate_blend_weight_grid([0.75, 0.25, 0.5]),
                         (0.25, 0.5, 0.75))
        for values in ([], [-0.1], [1.01], [math.nan], [0.5, 0.5]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                audit.validate_blend_weight_grid(list(values))

    def test_blend_endpoints_reproduce_candidate_and_held_prior(self):
        candidate = [[0.1, 0.4]] * 30
        prior = [[-0.2, 0.3], [0.2, 0.7]]
        all_candidate = audit.blend_candidate_and_prior_controls(
            candidate, prior, 1.0)
        all_prior = audit.blend_candidate_and_prior_controls(
            candidate, prior, 0.0)
        self.assertEqual(all_candidate, candidate)
        self.assertEqual(all_prior[:2], prior)
        self.assertEqual(all_prior[2:], [prior[-1]] * 28)

    def test_blend_rejects_invalid_horizon_weight_or_control(self):
        candidate = [[0.0, 0.2]] * 30
        prior = [[0.0, 0.2]]
        for weight in (-0.1, 1.1, math.nan, math.inf):
            with self.subTest(weight=weight), self.assertRaises(ValueError):
                audit.blend_candidate_and_prior_controls(
                    candidate, prior, weight)
        with self.assertRaises(ValueError):
            audit.blend_candidate_and_prior_controls(candidate[:29], prior, 0.5)
        with self.assertRaises(ValueError):
            audit.blend_candidate_and_prior_controls(
                candidate, [[0.0]], 0.5)

    def test_candidate_prefix_snapshot_cursor_is_post_publication(self):
        prior_controls = [[float(index), 0.2] for index in range(30)]
        candidate_controls = [[-1.0, 0.3] for _ in range(30)]
        captured_prior_sequences = []
        captured_prefix_sequences = []
        comparison_stream = io.StringIO()
        comparison_writer = csv.writer(comparison_stream)
        comparison_writer.writerow(audit.CANDIDATE_PRIOR_FIELDS)

        def fake_rollout(diag, *args, **kwargs):
            controls = diag["candidate_controls"]
            if kwargs.get("scenario_label") == "retained_certified_plan_suffix":
                captured_prior_sequences.append(controls)
            if kwargs.get("scenario_label") in {
                    "candidate_75ms_prefix", "prior_75ms_prefix"}:
                captured_prefix_sequences.append(
                    (kwargs["scenario_label"], controls))
            return {
                "first_corridor_violation_stage": None,
                "min_corridor_slack_m": 0.1,
                "source_plant_replay_steps": len(controls),
            }

        def record(status):
            return {
                "bag_time": 10.0,
                "status": status,
                "progress_m": 12.0,
                "consecutive_candidate_prefix_actions": 1,
                "candidate_controls": candidate_controls,
                "state": [0.0] * 10,
                "certified_plan_snapshot": {
                    "valid": True,
                    "horizon": 30,
                    "cursor": 4,
                    "controls": prior_controls,
                },
            }

        with patch.object(audit, "_source_rollout", side_effect=fake_rollout):
            candidate_rows = audit.compare_candidate_and_prior_plan(
                [record("accepted_candidate_prefix")], self.points,
                self.length, object(), csv.writer(io.StringIO()),
                comparison_writer, 0.05, 0.05, 0.1, 0.005)
            rejected_rows = audit.compare_candidate_and_prior_plan(
                [record("rejected_nonlinear_rollout")], self.points,
                self.length, object(), csv.writer(io.StringIO()),
                comparison_writer, 0.05, 0.05, 0.1, 0.005)

        self.assertEqual(captured_prior_sequences[0][0], prior_controls[4])
        self.assertEqual(captured_prior_sequences[1][0], prior_controls[5])
        self.assertEqual(captured_prefix_sequences[0][1], candidate_controls[:3])
        self.assertEqual(captured_prefix_sequences[1][1], prior_controls[4:7])
        self.assertEqual(captured_prefix_sequences[2][1], candidate_controls[:3])
        self.assertEqual(captured_prefix_sequences[3][1], prior_controls[5:8])

        csv_rows = list(csv.DictReader(io.StringIO(comparison_stream.getvalue())))
        self.assertEqual(len(csv_rows), 2)
        self.assertEqual(csv_rows[0]["prior_snapshot_source"], "recorded")
        self.assertEqual(csv_rows[0]["candidate_prefix_chain_length"], "1")
        self.assertEqual(float(csv_rows[0]["progress_m"]), 12.0)
        self.assertEqual(float(csv_rows[0]["prior_action_steering_rad"]), 4.0)
        self.assertEqual(csv_rows[1]["prior_snapshot_source"], "recorded")

    def test_retained_plan_history_reconstructs_published_cursor(self):
        controls = [[index * 0.001, 0.2] for index in range(30)]
        candidate = [[-index * 0.001, 0.3] for index in range(30)]
        rows, summary = audit.reconstruct_certified_plan_history([
            {
                "status": "accepted_optimal",
                "command_published": True,
                "candidate_controls": controls,
                "certified_plan_snapshot": {
                    "valid": True, "horizon": 30, "cursor": 0,
                    "controls": controls,
                },
            },
            {
                "status": "accepted_candidate_prefix",
                "command_published": True,
                "candidate_controls": candidate,
                "certified_plan_snapshot": {
                    "valid": True, "horizon": 30, "cursor": 1,
                    "controls": controls,
                },
            },
            {
                "status": "rejected_nonlinear_rollout",
                "command_published": False,
                "candidate_controls": candidate,
            },
        ])
        self.assertEqual(summary["recorded_snapshot_count"], 2)
        self.assertEqual(summary["reconstruction_matches_recorded_count"], 2)
        self.assertEqual(summary["reconstruction_mismatch_count"], 0)
        self.assertEqual(
            rows[2]["reconstructed_certified_plan_snapshot"]["cursor"], 1)
        self.assertEqual(
            rows[2]["reconstructed_certified_plan_snapshot"]["controls"],
            controls)


if __name__ == "__main__":
    unittest.main()
