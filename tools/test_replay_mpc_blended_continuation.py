"""Tests for the offline source-plant continuation seed replay utility."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from tools import replay_mpc_blended_continuation as replay


class BlendedContinuationReplayTests(unittest.TestCase):
    def test_direct_replay_weights_match_production_profile_order(self):
        self.assertEqual(replay.REPLAY_ARGUMENTS[-12:], [
            "4000", "1500", "7.5", "60", "25", "5", "7.8125",
            "3", "5", "0.1", "2000", "100",
        ])

    def test_replay_arguments_make_backtracking_explicit_and_bounded(self):
        baseline = replay.replay_arguments()
        experimental = replay.replay_arguments(3)
        self.assertEqual(baseline[replay.REPLAY_LINE_SEARCH_ARGUMENT_INDEX], "0")
        self.assertEqual(experimental[replay.REPLAY_LINE_SEARCH_ARGUMENT_INDEX], "3")
        for invalid in (-1, 4, True, 1.5):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                replay.replay_arguments(invalid)

    def test_load_feasible_blends_requires_full_horizon_and_positive_slack(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "blends.csv"
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=[
                    "full_horizon_source_valid", "command_bounds_valid",
                    "source_steps", "minimum_corridor_slack_m", "progress_m",
                    "controls_json", "bag_time_s", "candidate_weight",
                    "speed_error_rmse_mps", "objective_total",
                ])
                writer.writeheader()
                controls = [[0.0, 0.001]] * 30
                base = {
                    "full_horizon_source_valid": "True",
                    "command_bounds_valid": "True",
                    "source_steps": "30",
                    "progress_m": "5.0",
                    "controls_json": json.dumps(controls),
                    "bag_time_s": "1.0",
                    "candidate_weight": "0.25",
                    "speed_error_rmse_mps": "0.3",
                }
                writer.writerow({**base, "minimum_corridor_slack_m": "0.01",
                                 "objective_total": "10"})
                writer.writerow({**base, "minimum_corridor_slack_m": "-0.001",
                                 "objective_total": "1"})
                writer.writerow({**base, "source_steps": "29",
                                 "minimum_corridor_slack_m": "0.2",
                                 "objective_total": "2"})
                writer.writerow({**base, "candidate_weight": "0.50",
                                 "minimum_corridor_slack_m": "0.02",
                                 "objective_total": "12"})
            result = replay.load_feasible_blends(path)
            filtered = replay.load_feasible_blends(path, 0.50)
        self.assertEqual(list(result), [5.0])
        self.assertEqual(result[5.0]["objective_total"], 10.0)
        self.assertEqual(filtered[5.0]["candidate_weight"], 0.5)
        self.assertEqual(filtered[5.0]["objective_total"], 12.0)

    def test_load_feasible_blends_accepts_minimum_positive_throttle(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "minimum_drive.csv"
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=[
                    "full_horizon_source_valid", "command_bounds_valid",
                    "source_steps", "minimum_corridor_slack_m", "progress_m",
                    "controls_json", "bag_time_s", "candidate_weight",
                    "speed_error_rmse_mps", "objective_total",
                ])
                writer.writeheader()
                writer.writerow({
                    "full_horizon_source_valid": "True",
                    "command_bounds_valid": "True",
                    "source_steps": "30",
                    "minimum_corridor_slack_m": "0.02",
                    "progress_m": "1.0",
                    "controls_json": json.dumps([[0.0, 0.001]] * 30),
                    "bag_time_s": "1.0",
                    "candidate_weight": "0.25",
                    "speed_error_rmse_mps": "0.1",
                    "objective_total": "1.0",
                })
            self.assertIn(1.0, replay.load_feasible_blends(path))

    def test_select_states_matches_progress_and_requires_a_match(self):
        states = [[10.0, 5.0] + [0.0] * 36,
                  [11.0, 9.0] + [0.0] * 36]
        blends = {5.00001: {"progress_m": 5.00001},
                  12.0: {"progress_m": 12.0}}
        selected = replay.select_matched_states(states, blends)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0][0][0], 10.0)
        with self.assertRaises(ValueError):
            replay.select_matched_states(states[1:], blends)

    def test_quality_selector_accounts_for_unseeded_saved_states(self):
        states = [[100.0, 1.0] + [0.0] * 36,
                  [200.0, 2.0] + [0.0] * 36,
                  [300.0, 3.0] + [0.0] * 36]
        blend = {
            "source_steps": 30,
            "minimum_corridor_slack_m": 0.08,
            "objective_total": 12.0,
        }
        selected = [(1, states[1], blend)]
        cold = [
            {"status": "6", "nonlinear_objective": "inf",
             "min_corridor_slack_m": "-inf", "cycle_us": "1"},
            {"status": "0", "nonlinear_objective": "15",
             "min_corridor_slack_m": "0.2", "cycle_us": "2"},
            {"status": "6", "nonlinear_objective": "inf",
             "min_corridor_slack_m": "-inf", "cycle_us": "3"},
        ]
        seeded = [{"status": "6", "nonlinear_objective": "inf",
                   "min_corridor_slack_m": "-inf", "cycle_us": "4"}]
        rows = replay.build_quality_selector_rows(states, selected, cold, seeded)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["selected_action"],
                         "no_exact_valid_full_horizon_action")
        self.assertEqual(rows[1]["selected_action"],
                         "retain_exact_valid_source_seed")
        self.assertEqual(rows[1]["selected_objective"], 12.0)
        self.assertEqual(rows[1]["same_cycle_guard_action"],
                         "retain_exact_valid_source_seed")
        self.assertEqual(rows[1]["same_cycle_guard_objective"], 12.0)
        self.assertEqual(rows[2]["selected_action"],
                         "no_exact_valid_full_horizon_action")
        summary = replay.summarize_quality_selector(rows, cold, seeded)
        comparison = summary[
            "selected_vs_ordinary_cold_on_jointly_valid_states"]
        self.assertEqual(comparison["states"], 1)
        self.assertEqual(comparison["higher_objective_count"], 0)

    def test_quality_selector_uses_candidate_only_when_valid_and_no_worse(self):
        state = [100.0, 2.0] + [0.0] * 36
        selected = [(0, state, {
            "source_steps": 30,
            "minimum_corridor_slack_m": 0.08,
            "objective_total": 12.0,
        })]
        cold = [{"status": "6", "nonlinear_objective": "inf",
                 "min_corridor_slack_m": "-inf", "cycle_us": "1"}]
        seeded = [{"status": "8", "nonlinear_objective": "10",
                   "min_corridor_slack_m": "0.04", "cycle_us": "2"}]
        rows = replay.build_quality_selector_rows([state], selected,
                                                  cold, seeded)
        self.assertEqual(rows[0]["selected_action"],
                         "mpc_candidate_from_source_seed")
        self.assertEqual(rows[0]["selected_slack_m"], 0.04)
        self.assertEqual(rows[0]["same_cycle_guard_action"],
                         "accept_seeded_mpc_candidate")

    def test_quality_selector_includes_cold_result_for_seeded_state(self):
        state = [100.0, 2.0] + [0.0] * 36
        selected = [(0, state, {
            "source_steps": 30,
            "minimum_corridor_slack_m": 0.08,
            "objective_total": 12.0,
        })]
        cold = [{"status": "0", "nonlinear_objective": "3.5",
                 "min_corridor_slack_m": "0.12", "cycle_us": "1"}]
        seeded = [{"status": "1", "nonlinear_objective": "67",
                   "min_corridor_slack_m": "0.18", "cycle_us": "2"}]
        rows = replay.build_quality_selector_rows([state], selected,
                                                  cold, seeded)
        self.assertEqual(rows[0]["selected_action"],
                         "ordinary_exact_valid_mpc_candidate")
        self.assertEqual(rows[0]["selected_objective"], 3.5)
        self.assertEqual(rows[0]["selected_slack_m"], 0.12)
        self.assertEqual(rows[0]["same_cycle_guard_action"],
                         "retain_exact_valid_source_seed")
        summary = replay.summarize_quality_selector(
            rows, cold, seeded)["same_cycle_source_nominal_guard"]
        self.assertEqual(summary["choice_counts"],
                         {"retain_exact_valid_source_seed": 1})
        self.assertEqual(
            summary["comparison_to_cold_on_jointly_valid_states"][
                "higher_objective_count"], 1)

    def test_quality_selector_retains_source_seed_if_it_is_best(self):
        state = [100.0, 2.0] + [0.0] * 36
        selected = [(0, state, {
            "source_steps": 30,
            "minimum_corridor_slack_m": 0.08,
            "objective_total": 2.0,
        })]
        cold = [{"status": "0", "nonlinear_objective": "3.5",
                 "min_corridor_slack_m": "0.12", "cycle_us": "1"}]
        seeded = [{"status": "1", "nonlinear_objective": "67",
                   "min_corridor_slack_m": "0.18", "cycle_us": "2"}]
        rows = replay.build_quality_selector_rows([state], selected,
                                                  cold, seeded)
        self.assertEqual(rows[0]["selected_action"],
                         "retain_exact_valid_source_seed")
        self.assertEqual(rows[0]["selected_objective"], 2.0)
        self.assertEqual(rows[0]["selected_slack_m"], 0.08)
        self.assertEqual(rows[0]["same_cycle_guard_action"],
                         "retain_exact_valid_source_seed")

    def test_replay_summary_counts_only_exact_full_horizon_statuses(self):
        rows = [
            {"status": "0", "cycle_us": "1000", "nonlinear_objective": "2"},
            {"status": "1", "cycle_us": "3000", "nonlinear_objective": "4"},
            {"status": "6", "cycle_us": "5000", "nonlinear_objective": "inf"},
        ]
        summary = replay.summarize_replay(rows)
        self.assertEqual(summary["exact_full_horizon_accepts"], 2)
        self.assertEqual(summary["status_counts"], {"0": 1, "1": 1, "6": 1})
        self.assertEqual(summary["max_cycle_ms"], 5.0)
        self.assertEqual(summary["median_accepted_nonlinear_objective"], 2.0)


if __name__ == "__main__":
    unittest.main()
