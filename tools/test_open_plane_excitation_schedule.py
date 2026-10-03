#!/usr/bin/env python3
"""Focused schedule checks for the recursive high-steering capture."""

from __future__ import annotations

import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from open_plane_excitation import (
    DYNAMIC_COUPLED_PROFILES,
    _phase_steering_command,
    _slew_probe_command,
    build_schedule,
)
from race_domain_dynamic_coupled_plan import (
    DYNAMIC_STEERING_FREQUENCIES_HZ,
    FRONTIER_11MPS_ANGLES_RAD,
    FRONTIER_SWEEP_ANGLES_RAD,
    PRBS_MIN_DWELL_S,
    build_dynamic_coupled_plan,
)


class DynamicSteeringScheduleTest(unittest.TestCase):
    def test_dynamic_steering_covers_both_signs_and_all_speed_blocks(self) -> None:
        phases = build_schedule(17037, "race_domain_dynamic_steering")
        probes = [phase for phase in phases
                  if phase.label.startswith("dynamic_v")]
        self.assertEqual(len(probes), 36)
        self.assertEqual(
            sorted({phase.speed_target_mps for phase in probes}),
            [4.5, 6.5, 7.5],
        )
        for speed in (4.5, 6.5, 7.5):
            local = [phase for phase in probes
                     if phase.speed_target_mps == speed]
            steering = [phase.steering_rad for phase in local]
            self.assertEqual(len(local), 12)
            self.assertEqual(sorted(abs(value) for value in steering),
                             sorted((0.0, 0.0, 0.15, 0.15, 0.30, 0.30,
                                     0.42, 0.42, 0.50, 0.50,
                                     0.5236, 0.5236)))
            self.assertTrue(all(phase.duration_s == 1.25 for phase in local))
            self.assertTrue(all(phase.validate_samples for phase in local))
            self.assertTrue(all(not phase.validate_speed for phase in local))
            self.assertTrue(all(not phase.validate_steering for phase in local))

    def test_seed_randomizes_turn_direction_but_schedule_is_reproducible(self) -> None:
        def trace(seed: int) -> list[float]:
            return [phase.steering_rad for phase in build_schedule(
                seed, "race_domain_dynamic_steering")
                    if phase.label.startswith("dynamic_v4.5")]

        self.assertEqual(trace(17037), trace(17037))
        self.assertNotEqual(trace(17037), trace(17042))

    def test_speed_approach_precedes_each_dynamic_block(self) -> None:
        phases = build_schedule(17037, "race_domain_dynamic_steering")
        for speed in (4.5, 6.5, 7.5):
            approach_index = next(
                index for index, phase in enumerate(phases)
                if phase.label == f"approach_{speed:.1f}mps")
            self.assertTrue(phases[approach_index].reach_speed_target)
            following = phases[approach_index + 1]
            self.assertEqual(following.label, f"settle_{speed:.1f}mps")
            self.assertEqual(following.speed_target_mps, speed)
            self.assertFalse(following.reach_speed_target)


class HighSpeedSteeringFrontierScheduleTest(unittest.TestCase):
    def test_frontier_probes_are_matched_and_cover_both_turn_signs(self) -> None:
        phases = build_schedule(17041, "race_domain_steering_frontier")
        probes = [phase for phase in phases
                  if phase.label.startswith("frontier_")]
        self.assertEqual(len(probes), 61)
        self.assertEqual(len(phases), 122)
        self.assertTrue(all(phase.settle_before_probe for phase in probes))
        self.assertTrue(all(phase.validate_samples for phase in probes))
        self.assertTrue(all(not phase.validate_speed for phase in probes))
        self.assertTrue(all(phase.duration_s == 3.0 for phase in probes))
        for index, phase in enumerate(phases):
            if not phase.label.startswith("frontier_"):
                continue
            approach = phases[index - 1]
            self.assertEqual(approach.label, f"approach_{phase.label}")
            self.assertEqual(approach.speed_target_mps,
                             phase.speed_target_mps)
            self.assertEqual(approach.duration_s, 6.0)
            self.assertEqual(approach.throttle_mode, "race_domain_approach")
            self.assertTrue(approach.reach_speed_target)
        for speed in (9.5, 10.5, 11.1):
            local = [phase for phase in probes
                     if phase.speed_target_mps == speed]
            self.assertEqual(local[0].steering_rad, 0.0)
            self.assertEqual(len(local), 21 if speed < 11.0 else 19)
            magnitudes = sorted({abs(phase.steering_rad) for phase in local
                                 if phase.steering_rad != 0.0})
            self.assertEqual(magnitudes,
                             [0.02 * index for index in range(1, 11)]
                             if speed < 11.0 else
                             [0.02 * index for index in range(1, 10)])
            for magnitude in magnitudes:
                self.assertEqual(sum(phase.steering_rad > 0
                                     and abs(phase.steering_rad - magnitude) < 1e-9
                                     for phase in local), 1)
                self.assertEqual(sum(phase.steering_rad < 0
                                     and abs(phase.steering_rad + magnitude) < 1e-9
                                     for phase in local), 1)

    def test_frontier_speed_approach_precedes_each_settled_grid(self) -> None:
        phases = build_schedule(17041, "race_domain_steering_frontier")
        for speed in (9.5, 10.5, 11.1):
            probes = [phase for phase in phases
                      if phase.label.startswith("frontier_")
                      and phase.speed_target_mps == speed]
            self.assertTrue(probes)
            for probe in probes:
                index = phases.index(probe)
                approach = phases[index - 1]
                self.assertEqual(approach.label, f"approach_{probe.label}")
                self.assertEqual(approach.speed_target_mps, speed)
                self.assertTrue(approach.reach_speed_target)
            self.assertEqual(approach.throttle_mode,
                             "race_domain_approach")


class DynamicCoupledScheduleTest(unittest.TestCase):
    def test_profiles_cover_handoff_speeds_and_reset_each_condition(self) -> None:
        seed = 20261002
        conditions = build_dynamic_coupled_plan(seed)
        for profile in DYNAMIC_COUPLED_PROFILES:
            with self.subTest(profile=profile):
                phases = build_schedule(seed, profile)
                self.assertEqual(len(phases), 84)
                approaches = [phase for phase in phases
                              if phase.label.startswith("approach_coupled_")]
                self.assertEqual(len(approaches), len(conditions))
                self.assertEqual(
                    [phase.condition_pair_id for phase in approaches],
                    [condition.condition_id for condition in conditions],
                )
                for condition in conditions:
                    local = [phase for phase in phases
                             if phase.condition_pair_id == condition.condition_id]
                    dynamic_count = 6 if condition.speed_band == "9-11.2" else 8
                    self.assertEqual(len(local), dynamic_count + 2)
                    self.assertEqual(local[0].speed_target_mps,
                                     condition.target_speed_mps)
                    self.assertTrue(local[0].reach_speed_target)
                    self.assertEqual(local[0].throttle_mode,
                                     "race_domain_approach")
                    self.assertTrue(local[1].label.startswith("settle_coupled_"))
                    probes = local[2:]
                    self.assertEqual(len(probes), dynamic_count)
                    self.assertTrue(all(phase.validate_samples for phase in probes))
                    self.assertTrue(all(not phase.validate_speed
                                        and not phase.validate_steering
                                        for phase in probes))
                    ramps = [phase for phase in probes
                             if phase.throttle_mode == "slew_probe"]
                    self.assertEqual(
                        len(ramps), 2 if condition.speed_band == "9-11.2" else 3)
                    for phase in ramps:
                        self.assertEqual(phase.throttle_profile, "ramp")
                        self.assertGreaterEqual(phase.throttle_start_norm, 0.0)
                        self.assertLessEqual(phase.throttle_end_norm, 0.5)
                        start = phase.throttle_start_norm
                        end = phase.throttle_end_norm
                        self.assertAlmostEqual(
                            _slew_probe_command(phase, 0.0), start)
                        self.assertAlmostEqual(
                            _slew_probe_command(
                                phase, phase.throttle_stimulus_delay_s),
                            start)
                        self.assertAlmostEqual(
                            _slew_probe_command(
                                phase,
                                phase.throttle_stimulus_delay_s
                                + phase.throttle_ramp_duration_s / 2),
                            (start + end) / 2)
                        self.assertAlmostEqual(
                            _slew_probe_command(
                                phase,
                                phase.throttle_stimulus_delay_s
                                + phase.throttle_ramp_duration_s),
                            end)
                    brake = next(phase for phase in probes
                                 if phase.label.endswith("steering_active_brake"))
                    self.assertEqual(brake.throttle_mode, "fixed")
                    self.assertEqual(brake.throttle_norm, 0.0)
                    self.assertTrue(any(phase.steering_profile == "waypoints"
                                        for phase in probes))

    def test_band_a_uses_requested_frequencies_and_seeded_prbs(self) -> None:
        phases = build_schedule(20261002, DYNAMIC_COUPLED_PROFILES[0])
        for condition in build_dynamic_coupled_plan(20261002):
            if condition.speed_band != "5-7":
                continue
            local = [phase for phase in phases
                     if phase.condition_pair_id == condition.condition_id]
            multisine = next(phase for phase in local
                             if phase.steering_profile == "multisine")
            self.assertEqual(multisine.steering_frequencies_hz,
                             DYNAMIC_STEERING_FREQUENCIES_HZ)
            self.assertEqual(len(multisine.steering_phases_rad), 4)
            triangle = next(phase for phase in local
                            if phase.label.endswith("triangular_steering"))
            self.assertIn(triangle.steering_frequency_hz,
                          DYNAMIC_STEERING_FREQUENCIES_HZ)
            prbs = next(phase for phase in local
                        if phase.label.endswith("random_piecewise_steering"))
            self.assertEqual(prbs.steering_dwell_s, PRBS_MIN_DWELL_S)
            self.assertEqual(len(prbs.steering_prbs_levels_normalized), 12)

    def test_band_b_exact_speeds_use_measured_limits(self) -> None:
        expected = {7.0: 0.20, 7.5: 0.5236, 8.0: 0.10, 8.5: 0.10}
        conditions = [row for row in build_dynamic_coupled_plan(20261002)
                      if row.speed_band == "7-9"]
        self.assertEqual({row.target_speed_mps for row in conditions},
                         set(expected))
        self.assertEqual({row.target_speed_mps: row.max_steering_rad
                          for row in conditions}, expected)

    def test_band_c_sequences_cross_frontier_angles_in_both_signs(self) -> None:
        phases = build_schedule(20261002, DYNAMIC_COUPLED_PROFILES[0])
        conditions = [row for row in build_dynamic_coupled_plan(20261002)
                      if row.speed_band == "9-11.2"]
        for condition in conditions:
            local = [phase for phase in phases
                     if phase.condition_pair_id == condition.condition_id]
            sweep = next(phase for phase in local
                         if phase.label.endswith("frontier_sweep_both_signs"))
            expected_angles = (FRONTIER_11MPS_ANGLES_RAD
                               if condition.target_speed_mps == 11.1
                               else FRONTIER_SWEEP_ANGLES_RAD)
            values = {value for _, value in sweep.steering_waypoints}
            for angle in expected_angles:
                self.assertIn(angle, values)
                self.assertIn(-angle, values)
            for step in range(int(sweep.duration_s * 40) + 1):
                self.assertLessEqual(
                    abs(_phase_steering_command(sweep, step / 40.0)),
                    condition.max_steering_rad + 1e-9)
            mixed = next(phase for phase in local
                         if phase.label.endswith("frontier_mixed_order"))
            mixed_values = [value for _, value in mixed.steering_waypoints]
            self.assertTrue(0.12 in mixed_values or -0.12 in mixed_values)
            self.assertTrue(0.04 in mixed_values or -0.04 in mixed_values)

    def test_seed_changes_plan_and_all_profiles_remain_inside_steering_limit(self) -> None:
        first = build_schedule(20261002, DYNAMIC_COUPLED_PROFILES[0])
        second = build_schedule(20261003, DYNAMIC_COUPLED_PROFILES[0])
        trace = lambda rows: [phase.condition_pair_id for phase in rows
                              if phase.label.startswith("approach_coupled_")]
        self.assertNotEqual(trace(first), trace(second))
        self.assertEqual(first, build_schedule(20261002, DYNAMIC_COUPLED_PROFILES[0]))
        for phase in first:
            if phase.steering_profile is None:
                continue
            for step in range(int(phase.duration_s * 40) + 1):
                self.assertLessEqual(
                    abs(_phase_steering_command(phase, step / 40.0)),
                    0.5236 + 1e-9)

    def test_seed_changes_condition_order_and_signed_first_turns(self) -> None:
        first = build_schedule(20261002, DYNAMIC_COUPLED_PROFILES[0])
        second = build_schedule(20261003, DYNAMIC_COUPLED_PROFILES[0])
        trace = lambda rows: [phase.condition_pair_id for phase in rows
                              if phase.label.startswith("approach_coupled_")]
        self.assertNotEqual(trace(first), trace(second))
        self.assertEqual(first, build_schedule(20261002, DYNAMIC_COUPLED_PROFILES[0]))

    def test_profile_phase_duration_and_reset_budget_fit_documented_timeout(self):
        seed = 20261002
        phases = build_schedule(seed, DYNAMIC_COUPLED_PROFILES[0])
        duration_s = sum(phase.duration_s for phase in phases)
        reset_budget_s = len(build_dynamic_coupled_plan(seed)) * (0.90 + 4.0)
        self.assertLess(duration_s + reset_budget_s + 5.0, 600.0)


if __name__ == "__main__":
    unittest.main()
