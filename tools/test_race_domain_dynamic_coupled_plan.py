#!/usr/bin/env python3
"""Plan-level checks for the WP3 dynamic-coupled campaign."""

from __future__ import annotations

import unittest

from race_domain_dynamic_coupled_plan import (
    DYNAMIC_STEERING_FREQUENCIES_HZ,
    FRONTIER_11MPS_ANGLES_RAD,
    FRONTIER_SWEEP_ANGLES_RAD,
    PRBS_MIN_DWELL_S,
    build_dynamic_coupled_plan,
    plan_as_dicts,
)


class DynamicCoupledConditionPlanTest(unittest.TestCase):
    def test_plan_covers_handoff_speeds_and_three_bands(self) -> None:
        conditions = build_dynamic_coupled_plan(20261002)
        self.assertEqual(len(conditions), 9)
        self.assertEqual(
            sorted(row.target_speed_mps for row in conditions),
            [5.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.5, 10.5, 11.1],
        )
        expected_band_counts = {"5-7": 2, "7-9": 4, "9-11.2": 3}
        self.assertEqual(
            {band: sum(row.speed_band == band for row in conditions)
             for band in expected_band_counts},
            expected_band_counts,
        )
        for row in conditions:
            low, high = map(float, row.speed_band.split("-"))
            self.assertGreaterEqual(row.target_speed_mps, low)
            self.assertLess(row.target_speed_mps, high)
            self.assertIn(row.first_turn_sign, (-1, 1))
            self.assertGreater(row.max_steering_rad, 0.0)
            self.assertLessEqual(row.max_steering_rad, 0.5236)

    def test_frontier_levels_reach_both_sides_but_respect_11mps_limit(self) -> None:
        self.assertEqual(FRONTIER_SWEEP_ANGLES_RAD,
                         (0.04, 0.08, 0.12, 0.16, 0.20))
        self.assertEqual(FRONTIER_11MPS_ANGLES_RAD,
                         (0.04, 0.08, 0.12, 0.16, 0.18))

    def test_steering_envelope_uses_recorded_frontier_support(self) -> None:
        expected = {
            5.0: 0.30, 6.5: 0.20, 7.0: 0.20, 7.5: 0.5236,
            8.0: 0.10, 8.5: 0.10, 9.5: 0.20, 10.5: 0.20,
            11.1: 0.18,
        }
        for condition in build_dynamic_coupled_plan(20261002):
            self.assertEqual(condition.max_steering_rad,
                             expected[condition.target_speed_mps])

    def test_seeded_waveform_parameters_are_reproducible_and_bounded(self) -> None:
        first = build_dynamic_coupled_plan(20261002)
        self.assertEqual(first, build_dynamic_coupled_plan(20261002))
        self.assertNotEqual(first, build_dynamic_coupled_plan(20261003))
        for condition in first:
            self.assertIn(condition.triangle_frequency_hz,
                          DYNAMIC_STEERING_FREQUENCIES_HZ)
            self.assertEqual(len(condition.multisine_phase_rad), 4)
            self.assertTrue(all(0.0 <= value < 6.283185307179586
                                for value in condition.multisine_phase_rad))
            levels = condition.prbs_levels_normalized
            self.assertEqual(len(levels), 12)
            self.assertEqual(len(set(levels)), 12)
            self.assertTrue(all(-1.0 < value < 1.0 for value in levels))
            self.assertEqual(PRBS_MIN_DWELL_S, 0.50)

    def test_json_plan_records_seeded_waveform_details(self) -> None:
        rows = plan_as_dicts(20261002)
        self.assertEqual(len({row["condition_id"] for row in rows}), 9)
        self.assertEqual(
            {row["target_speed_mps"] for row in rows},
            {5.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.5, 10.5, 11.1},
        )
        self.assertTrue(all("multisine_phase_rad" in row
                            and "prbs_levels_normalized" in row
                            for row in rows))


if __name__ == "__main__":
    unittest.main()
