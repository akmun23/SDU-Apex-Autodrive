"""Focused tests for matching full-path source-rollout failures to controls."""

from __future__ import annotations

import math
import unittest

from tools import evaluate_mpc_bag as evaluator


def _window(stamp: int, *, error: float, speed: float = 3.0,
            steering: float = 0.02, progress: float = 10.0,
            throttle_delta: float = 0.2) -> dict[str, object]:
    return {
        "start_stamp_ns": stamp,
        "path_max_xy_m": error,
        "start_speed_mps": speed,
        "start_steering_rad": steering,
        "progress_m": progress,
        "start_queue_throttle_delta": throttle_delta,
        "command_trace": [],
    }


class SourceRolloutCaseIndexTest(unittest.TestCase):
    def test_initial_motion_oracle_changes_only_requested_states(self) -> None:
        state = [1.0, 2.0, 3.0, 4.0]
        truth = {"u": 10.0, "v": 20.0, "r": 30.0}
        corrected = evaluator.apply_initial_motion_oracle(
            state, truth, {"u"})
        self.assertEqual(corrected, [10.0, 2.0, 3.0, 4.0])
        self.assertEqual(state, [1.0, 2.0, 3.0, 4.0])

    def test_initial_motion_oracle_rejects_unknown_state(self) -> None:
        with self.assertRaisesRegex(ValueError, "u/v/r"):
            evaluator.apply_initial_motion_oracle(
                [0.0, 0.0, 0.0], {"u": 1.0}, {"wheel"})

    def test_failure_gets_unique_control_in_same_supported_regime(self) -> None:
        failure = _window(1_000_000_000, error=0.08)
        controls = [
            _window(2_000_000_000, error=0.04, speed=3.2,
                    steering=0.04, progress=12.0),
            _window(3_000_000_000, error=0.049, speed=4.1,
                    steering=0.04, progress=12.0),
        ]
        result = evaluator._build_source_rollout_case_index(
            [failure, *controls], {"bag_path": "missing-test-bag"})
        self.assertEqual(result["summary"]["full_path_over_5cm"], 1)
        self.assertEqual(
            result["summary"]["matched_distinct_nonfailure_controls"], 1)
        pair = result["pairs"][0]
        self.assertEqual(pair["failure_start_stamp_ns"], 1_000_000_000)
        self.assertEqual(pair["control_start_stamp_ns"], 2_000_000_000)
        self.assertEqual(pair["match_class"], "strict")

    def test_no_match_when_command_transition_or_regime_disagrees(self) -> None:
        failure = _window(1_000_000_000, error=0.08)
        wrong_transition = _window(2_000_000_000, error=0.02,
                                   throttle_delta=-0.2)
        wrong_regime = _window(3_000_000_000, error=0.02, speed=6.0)
        result = evaluator._build_source_rollout_case_index(
            [failure, wrong_transition, wrong_regime], {})
        self.assertEqual(
            result["summary"]["matched_distinct_nonfailure_controls"], 0)
        self.assertEqual(
            result["summary"]["unmatched_failure_start_stamps_ns"],
            [1_000_000_000])

    def test_nonfinite_progress_does_not_break_strict_matching(self) -> None:
        failure = _window(1_000_000_000, error=0.08, progress=math.nan)
        control = _window(2_000_000_000, error=0.02, progress=math.nan)
        result = evaluator._build_source_rollout_case_index(
            [failure, control], {})
        self.assertEqual(
            result["summary"]["strict_matches"], 1)
        self.assertIsNone(result["pairs"][0]["feature_delta"]["progress_m"])

    def test_truth_pose_twist_discontinuity_is_not_scored_as_model_failure(self) -> None:
        contaminated = _window(1_000_000_000, error=0.30)
        contaminated["truth_pose_twist_closure_max_m"] = 0.34
        clean_failure = _window(2_000_000_000, error=0.08)
        clean_failure["truth_pose_twist_closure_max_m"] = 0.01
        clean_control = _window(3_000_000_000, error=0.02)
        clean_control["truth_pose_twist_closure_max_m"] = 0.01
        result = evaluator._build_source_rollout_case_index(
            [contaminated, clean_failure, clean_control], {})
        summary = result["summary"]
        self.assertEqual(summary["raw_full_path_over_5cm"], 2)
        self.assertEqual(summary["truth_pose_twist_closure_invalid_windows"], 1)
        self.assertEqual(summary["clean_full_path_over_5cm"], 1)
        self.assertEqual(len(result["pairs"]), 1)
        self.assertEqual(result["pairs"][0]["failure_start_stamp_ns"],
                         2_000_000_000)


if __name__ == "__main__":
    unittest.main()
