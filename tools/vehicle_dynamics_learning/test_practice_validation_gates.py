from __future__ import annotations

import unittest

from tools.vehicle_dynamics_learning.evaluate_sensor_observer_practice import (
    _validate_lap_count_transitions,
)
from tools.vehicle_dynamics_learning.prepare_dataset import _split_for_name
from tools.vehicle_dynamics_learning.validate_practice_capture import (
    _classify_active_packet_rows,
    _sensor_ready_index,
)


class PracticeValidationGateTest(unittest.TestCase):
    def test_six_lap_gate_accepts_only_exact_zero_through_six(self):
        _validate_lap_count_transitions(list(range(7)), 6)
        for counts in (list(range(13)), [0, 1, 2, 4, 5, 6],
                       [0, 1, 2, 1, 2, 3, 4, 5, 6]):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                _validate_lap_count_transitions(counts, 6)

    def test_legacy_twelve_lap_gate_contract_is_unchanged(self):
        _validate_lap_count_transitions(list(range(13)), 12)
        with self.assertRaises(ValueError):
            _validate_lap_count_transitions(list(range(7)), 12)

    def test_fresh_practice_validation_names_cannot_default_to_training(self):
        self.assertEqual(_split_for_name("practice_model_validation_r01"),
                         "validation")
        self.assertEqual(_split_for_name("practice_model_validation_r02"),
                         "validation")
        self.assertEqual(_split_for_name("practice_model_train_r01"), "train")

    def test_only_leading_unaligned_bridge_startup_is_outside_scored_interval(self):
        rows = [(100, None, 0.008), (125, None, 0.007),
                (150, 4, 0.006), (175, 5, 0.005), (200, 6, 0.004)]
        report = _classify_active_packet_rows(rows, 90)
        self.assertEqual(report["matched_samples"], 3)
        self.assertEqual(report["fraction"], 1.0)
        self.assertEqual(report["first_last_packet_id"], [4, 6])
        self.assertEqual(
            report["pre_packet_startup"]["samples_excluded_from_scoring"], 2)

    def test_packet_loss_after_bridge_stream_start_is_not_excluded(self):
        for rows in (
                [(100, 4, 1.0), (125, None, 1.0), (150, 6, 1.0)],
                [(100, 4, 1.0), (125, 6, 1.0)],
                [(100, None, 1.0), (125, None, 1.0)]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                _classify_active_packet_rows(rows, 90)

    def test_only_short_encoder_history_initialization_may_precede_sensor_ready(self):
        self.assertEqual(_sensor_ready_index(
            [False, True, True], [0, 25_000_000, 50_000_000]), 1)
        for valid, stamps in (
                ([False, False, True], [0, 80_000_000, 160_000_000]),
                ([True, False, True], [0, 25_000_000, 50_000_000]),
                ([False, False], [0, 25_000_000])):
            with self.subTest(valid=valid), self.assertRaises(ValueError):
                _sensor_ready_index(valid, stamps)


if __name__ == "__main__":
    unittest.main()
