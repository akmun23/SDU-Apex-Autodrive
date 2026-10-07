import json
import tempfile
import unittest
from pathlib import Path

from tools.racing.build_empirical_throttle_response_model import (
    ALL_METRICS,
    METRICS,
    WINDOWED_METRICS,
    build_model,
    lookup_supported_effect,
    recommend_throttle_profile,
)


def _row(run_id, split, cell_index, wheel_effect, accel_effect):
    conditions = [
        (0.133333, 0.12, "left"),
        (0.266667, 0.20, "right"),
    ]
    rate, steering, turn = conditions[cell_index]
    row = {
        "run_id": run_id,
        "split": split,
        "valid": True,
        "ramp_speed_governor_ticks": 0,
        "step_speed_governor_ticks": 0,
        "speed_target_mps": 9.0,
        "throttle_delta_direction": "up",
        "throttle_delta_norm": 0.04,
        "throttle_rise_rate_norm_per_sec": rate,
        "abs_steering_command_rad": steering,
        "turn_direction": turn,
        METRICS["wheel_residual_mps"]: wheel_effect,
        METRICS["wheel_slip_ratio_proxy"]: 0.01 * wheel_effect,
        METRICS["longitudinal_accel_mps2"]: accel_effect,
    }
    for metric_field in ALL_METRICS.values():
        row.setdefault(metric_field, 0.0)
    return row


class BuildEmpiricalThrottleResponseModelTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def _analysis(self, name, rows, runs=None):
        path = self.root / name
        if runs is None:
            runs = [{
                "run_id": run_id,
                "aborted": False,
                "collision_count_start_end": [0, 0],
                "bridge_timing_faults": 0,
                "reset_recovery_pass": True,
                "streams_meet_40hz_receive_gate": True,
                "encoder_source_stamp_match_fraction": {"left": 1.0, "right": 1.0},
            } for run_id in sorted({row["run_id"] for row in rows})]
        path.write_text(json.dumps({"paired_results": rows, "runs": runs}),
                        encoding="utf-8")
        return path

    def _model(self):
        training = self._analysis("train.json", [
            _row("train-a", "train", 0, 1.0, -0.10),
            _row("train-a", "train", 1, -5.0, 0.50),
            _row("train-b", "train", 0, 1.2, -0.12),
            _row("train-b", "train", 1, -5.2, 0.52),
        ])
        validation = self._analysis("validation.json", [
            _row("validation-a", "validation", 0, 1.1, -0.11),
            _row("validation-a", "validation", 1, -5.1, 0.51),
            _row("validation-b", "validation", 0, 0.9, -0.09),
            _row("validation-b", "validation", 1, -4.9, 0.49),
        ])
        return build_model([training], [validation])

    def test_lookup_uses_training_only_and_rejects_unsupported_cells(self):
        model = self._model()
        effect = lookup_supported_effect(
            model,
            speed_target_mps=9.0,
            throttle_delta_direction="up",
            throttle_delta_norm=0.04,
            throttle_rise_rate_norm_per_sec=0.133333,
            abs_steering_command_rad=0.12,
            turn_direction="left",
        )
        self.assertAlmostEqual(effect["wheel_residual_mps"], 1.1)
        self.assertAlmostEqual(effect["wheel_slip_ratio_proxy"], 0.011)
        self.assertAlmostEqual(effect["longitudinal_accel_mps2"], -0.11)
        self.assertIn("wheel_residual_early_mps", effect)
        unsupported = lookup_supported_effect(
            model,
            speed_target_mps=9.0,
            throttle_delta_direction="up",
            throttle_delta_norm=0.04,
            throttle_rise_rate_norm_per_sec=0.133333,
            abs_steering_command_rad=0.15,
            turn_direction="left",
        )
        self.assertIsNone(unsupported)
        self.assertEqual(model["training_capture_count"], 2)
        self.assertEqual(model["validation_capture_count"], 2)
        self.assertEqual(model["training_cross_validation"]["fold_count"], 2)
        actions = [cell["throttle_action_evidence"]
                   for cell in model["support_limited_lookup"]["cells"]]
        self.assertEqual(actions, ["ramp_dominates_step", "step_dominates_ramp"])
        self.assertEqual(recommend_throttle_profile(
            model,
            speed_target_mps=9.0,
            throttle_delta_direction="up",
            throttle_delta_norm=0.04,
            throttle_rise_rate_norm_per_sec=0.133333,
            abs_steering_command_rad=0.12,
            turn_direction="left",
        ), "ramp")
        self.assertEqual(recommend_throttle_profile(
            model,
            speed_target_mps=9.0,
            throttle_delta_direction="up",
            throttle_delta_norm=0.04,
            throttle_rise_rate_norm_per_sec=0.266667,
            abs_steering_command_rad=0.20,
            turn_direction="right",
        ), "step")

    def test_validation_is_scored_separately_from_fitted_lookup(self):
        model = self._model()
        validation = model["held_out_validation"]
        self.assertEqual(validation["method"],
                         "independent whole-capture validation; validation not fitted")
        self.assertEqual(len(validation["per_run"]), 2)
        wheel = validation["summary"]["wheel_residual_mps"]
        self.assertAlmostEqual(wheel["macro_run_exact_rmse_common_support"], 0.1)
        self.assertGreater(wheel["macro_run_coarse_rmse_common_support"], 1.0)
        cell = model["support_limited_lookup"]["cells"][0]
        self.assertEqual(cell["independent_training_run_count"], 2)
        action_check = validation["dominance_suggestion_check"]
        self.assertEqual(action_check["direction_matches"], 4)
        self.assertEqual(action_check["direction_mismatches"], 0)

    def test_action_is_not_recommended_from_one_validation_capture(self):
        training = self._analysis("train-one-cell.json", [
            _row("train-a", "train", 0, 1.0, -0.10),
            _row("train-b", "train", 0, 1.2, -0.12),
        ])
        validation = self._analysis("validation-one-cell.json", [
            _row("validation-a", "validation", 0, 1.1, -0.11),
        ])
        model = build_model([training], [validation])
        self.assertIsNone(recommend_throttle_profile(
            model,
            speed_target_mps=9.0,
            throttle_delta_direction="up",
            throttle_delta_norm=0.04,
            throttle_rise_rate_norm_per_sec=0.133333,
            abs_steering_command_rad=0.12,
            turn_direction="left",
        ))

    def test_conflicting_slip_ratio_sign_blocks_action_recommendation(self):
        train_rows = [
            _row("train-a", "train", 0, 1.0, -0.10),
            _row("train-b", "train", 0, 1.2, -0.12),
        ]
        validation_rows = [
            _row("validation-a", "validation", 0, 1.1, -0.11),
            _row("validation-b", "validation", 0, 0.9, -0.09),
        ]
        slip_field = METRICS["wheel_slip_ratio_proxy"]
        for row in (*train_rows, *validation_rows):
            row[slip_field] = -0.01
        model = build_model(
            [self._analysis("train-slip-conflict.json", train_rows)],
            [self._analysis("validation-slip-conflict.json", validation_rows)],
        )
        cell = model["support_limited_lookup"]["cells"][0]
        self.assertEqual(cell["throttle_action_evidence"],
                         "tradeoff_or_run_inconsistent")
        self.assertIsNone(recommend_throttle_profile(
            model,
            speed_target_mps=9.0,
            throttle_delta_direction="up",
            throttle_delta_norm=0.04,
            throttle_rise_rate_norm_per_sec=0.133333,
            abs_steering_command_rad=0.12,
            turn_direction="left",
        ))

    def test_left_right_pooling_combines_mirror_turns_within_each_run(self):
        train_rows = []
        validation_rows = []
        for run_id, split, destination, left_scale, right_scale in (
            ("train-a", "train", train_rows, 1.0, 3.0),
            ("train-b", "train", train_rows, 1.2, 2.8),
            ("validation-a", "validation", validation_rows, 1.1, 2.9),
            ("validation-b", "validation", validation_rows, 0.9, 3.1),
        ):
            left = _row(run_id, split, 0, left_scale, -0.1 * left_scale)
            right = _row(run_id, split, 0, right_scale, -0.1 * right_scale)
            right["turn_direction"] = "left" if left["turn_direction"] == "right" else "right"
            destination.extend((left, right))
        model = build_model(
            [self._analysis("mirror-train.json", train_rows)],
            [self._analysis("mirror-validation.json", validation_rows)],
            assume_left_right_symmetry=True,
        )
        self.assertTrue(model["assume_left_right_symmetry"])
        self.assertEqual(model["training_pair_count"], 4)
        self.assertEqual(len(model["support_limited_lookup"]["cells"]), 1)
        cell = model["support_limited_lookup"]["cells"][0]
        self.assertEqual(cell["turn_direction"], "both")
        self.assertAlmostEqual(
            cell["effects"]["wheel_residual_mps"]["step_minus_ramp_mean"],
            2.0,
        )
        for turn in ("left", "right"):
            effect = lookup_supported_effect(
                model,
                speed_target_mps=9.0,
                throttle_delta_direction="up",
                throttle_delta_norm=0.04,
                throttle_rise_rate_norm_per_sec=0.133333,
                abs_steering_command_rad=0.12,
                turn_direction=turn,
            )
            self.assertAlmostEqual(effect["wheel_residual_mps"], 2.0)
            self.assertEqual(recommend_throttle_profile(
                model,
                speed_target_mps=9.0,
                throttle_delta_direction="up",
                throttle_delta_norm=0.04,
                throttle_rise_rate_norm_per_sec=0.133333,
                abs_steering_command_rad=0.12,
                turn_direction=turn,
            ), "ramp")

    def test_legacy_rows_without_window_metrics_keep_integrated_support(self):
        train_rows = [
            _row("train-a", "train", 0, 1.0, -0.10),
            _row("train-b", "train", 0, 1.2, -0.12),
        ]
        validation_rows = [
            _row("validation-a", "validation", 0, 1.1, -0.11),
            _row("validation-b", "validation", 0, 0.9, -0.09),
        ]
        window_fields = tuple(WINDOWED_METRICS.values())
        for row in (*train_rows, *validation_rows):
            for field in window_fields:
                row.pop(field)
        model = build_model(
            [self._analysis("legacy-train.json", train_rows)],
            [self._analysis("legacy-validation.json", validation_rows)],
        )
        effect = lookup_supported_effect(
            model,
            speed_target_mps=9.0,
            throttle_delta_direction="up",
            throttle_delta_norm=0.04,
            throttle_rise_rate_norm_per_sec=0.133333,
            abs_steering_command_rad=0.12,
            turn_direction="left",
        )
        self.assertAlmostEqual(effect["wheel_residual_mps"], 1.1)
        self.assertNotIn("wheel_residual_early_mps", effect)

    def test_sealed_rows_are_rejected_even_if_invalid(self):
        training = self._analysis("sealed.json", [
            _row("train-a", "train", 0, 1.0, 0.10),
            {**_row("sealed", "final_test", 0, 99.0, 99.0), "valid": False},
        ])
        validation = self._analysis("validation-only.json", [
            _row("validation-a", "validation", 0, 1.1, 0.11),
        ])
        with self.assertRaisesRegex(ValueError, "sealed split"):
            build_model([training], [validation])

    def test_train_validation_capture_id_overlap_is_rejected(self):
        training = self._analysis("train-overlap.json", [
            _row("same-run", "train", 0, 1.0, 0.10),
        ])
        validation = self._analysis("validation-overlap.json", [
            _row("same-run", "validation", 0, 1.1, 0.11),
        ])
        with self.assertRaisesRegex(ValueError, "leakage"):
            build_model([training], [validation])

    def test_capture_that_misses_40hz_gate_is_rejected(self):
        training = self._analysis("bad-timing.json", [
            _row("train-a", "train", 0, 1.0, -0.10),
        ], runs=[{
            "run_id": "train-a",
            "aborted": False,
            "collision_count_start_end": [0, 0],
            "bridge_timing_faults": 0,
            "reset_recovery_pass": True,
            "streams_meet_40hz_receive_gate": False,
            "encoder_source_stamp_match_fraction": {"left": 1.0, "right": 1.0},
        }])
        validation = self._analysis("good-validation.json", [
            _row("validation-a", "validation", 0, 1.1, -0.11),
        ])
        with self.assertRaisesRegex(ValueError, "40hz_stream_gate_failed"):
            build_model([training], [validation])


if __name__ == "__main__":
    unittest.main()
