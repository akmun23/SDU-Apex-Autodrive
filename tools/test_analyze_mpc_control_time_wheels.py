import math
import unittest

from tools import analyze_mpc_control_time_wheels as wheels
from tools import evaluate_mpc_bag as evaluator


def _encoder_rows(omega_at_one_second, acceleration, stamps):
    return [{
        "stamp": stamp,
        "angle": (omega_at_one_second * (stamp - 1.0) +
                  0.5 * acceleration * (stamp - 1.0) ** 2),
    } for stamp in stamps]


class ControlTimeWheelAnalysisTests(unittest.TestCase):
    def test_candidate_epoch_matches_preceding_fixed_physics_interval(self):
        stamps = [0.95, 0.975, 1.0, 1.025, 1.05, 1.075]
        target_ns = 1_025_000_000
        self.assertEqual(len(wheels._encoder_rows({
            wheels.ENCODER_TOPICS["left"]: _encoder_rows(95.0, 400.0, stamps),
        })["left"][0]), len(stamps))
        left_interval = wheels._nearest_encoder_interval(
            stamps, _encoder_rows(95.0, 400.0, stamps), target_ns * 1.0e-9)
        right_interval = wheels._nearest_encoder_interval(
            stamps, _encoder_rows(115.0, 400.0, stamps), target_ns * 1.0e-9)
        self.assertIsNotNone(left_interval)
        self.assertIsNotNone(right_interval)
        self.assertAlmostEqual(left_interval[1], 100.0, places=7)
        self.assertAlmostEqual(right_interval[1], 120.0, places=7)
        self.assertAlmostEqual(left_interval[2], 0.0)

    def test_offset_scan_distinguishes_adjacent_epochs(self):
        stamps = [0.925, 0.95, 0.975, 1.0, 1.025, 1.05, 1.075]
        diag = {
            "candidate_command_stamp_ns": 1_025_000_000,
            "source_stamp_ns": 975_000_000,
            "plant_internal_state": [0.0, 0.0, 100.0, 120.0] + [0.0] * 18,
            "state": [0.0, 0.0, 4.0, 0.1, 0.0, 0.0, 0.0, 0.0],
            "control_time_prediction": {"age_s": 0.05},
            "first_action": [0.05, 0.4],
            "progress_m": 12.0,
            "status": "accepted_optimal",
        }
        bag_data = {
            evaluator.DIAGNOSTICS: [{"json": diag, "bag_time": 1.0}],
            wheels.ENCODER_TOPICS["left"]:
                _encoder_rows(95.0, 400.0, stamps),
            wheels.ENCODER_TOPICS["right"]:
                _encoder_rows(115.0, 400.0, stamps),
        }
        report, rows = wheels.analyze(bag_data, [-1, 0, 1])
        self.assertEqual(report["matched_control_epochs"], 1)
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["candidate_command_lead_ms"], 50.0)
        self.assertAlmostEqual(rows[0]["predicted_plant_age_ms"], 50.0)
        self.assertAlmostEqual(rows[0]["rear_left_error_radps"], 0.0, places=7)
        self.assertAlmostEqual(rows[0]["rear_right_error_radps"], 0.0, places=7)
        offsets = {row["packet_offset"]: row for row in report["offset_scan"]}
        self.assertEqual(offsets[0]["sample_count"], 2)
        self.assertAlmostEqual(offsets[0]["rmse_radps"], 0.0, places=7)
        self.assertAlmostEqual(offsets[-1]["rmse_radps"], 10.0, places=7)
        self.assertAlmostEqual(offsets[1]["rmse_radps"], 10.0, places=7)
        self.assertEqual(report["no_future_runtime_input"], True)

    def test_missing_encoder_epoch_or_bad_interval_is_not_matched(self):
        stamps = [1.0, 1.04]
        rows = _encoder_rows(20.0, 0.0, stamps)
        self.assertIsNone(wheels._nearest_encoder_interval(
            stamps, rows, 1.04))
        self.assertIsNone(wheels._nearest_encoder_interval(
            stamps, rows, 1.035))

    def test_percentile_and_empty_metrics_are_well_defined(self):
        self.assertIsNone(wheels._percentile([], 0.95))
        self.assertIsNone(wheels._stats([])["rmse_radps"])
        self.assertEqual(wheels._percentile([1.0, 2.0, 3.0], 0.95), 3.0)
        self.assertTrue(math.isfinite(wheels._stats([1.0, -1.0])["rmse_radps"]))


if __name__ == "__main__":
    unittest.main()
