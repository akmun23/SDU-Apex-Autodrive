#!/usr/bin/env python3
"""Focused schedule checks for the recursive high-steering capture."""

from __future__ import annotations

import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from open_plane_excitation import (
    DYNAMIC_COUPLED_PROFILES,
    EXPERIMENT_PROFILE_CHOICES,
    LOW_SPEED_TRANSIENT_PROFILE,
    LOW_SPEED_TRANSIENT_SPEEDS_MPS,
    LOW_SPEED_TRANSIENT_STEERING_RAD,
    YAW_TRANSIENT_PROFILE,
    YAW_TRANSIENT_SPEEDS_MPS,
    YAW_TRANSIENT_STEERING_RAD,
    YAW_ATLAS_INTERPOLATION_PROFILE,
    YAW_ATLAS_INTERPOLATION_POINTS,
    YAW_ATLAS_OFFGRID_FINAL_PROFILE,
    YAW_ATLAS_OFFGRID_FINAL_POINTS,
    YAW_ATLAS_EXTRATREES_FINAL_PROFILE,
    YAW_ATLAS_EXTRATREES_FINAL_POINTS,
    YAW_ATLAS_EXTRATREES_HIGHSTEER_FINAL_PROFILE,
    YAW_ATLAS_EXTRATREES_HIGHSTEER_FINAL_POINTS,
    YAW_HIGHSTEER_SPEED_SURFACE_TRAIN_PROFILE,
    YAW_HIGHSTEER_SPEED_SURFACE_TRAIN_POINTS,
    YAW_HIGHSTEER_SPEED_SURFACE_TRAIN_REPEATS,
    YAW_ATLAS_INTERPOLATION_REPEATS,
    YAW_FULLBAND_GAPFILL_PROFILE,
    YAW_FULLBAND_GAPFILL_POINTS,
    YAW_FULLBAND_GAPFILL_REPEATS,
    YAW_UNWIND_THROTTLE_PROFILE,
    YAW_UNWIND_THROTTLE_SPEED_MPS,
    YAW_UNWIND_THROTTLE_STEERING_RAD,
    YAW_UNWIND_THROTTLE_START_NORM,
    YAW_UNWIND_THROTTLE_END_NORM,
    YAW_UNWIND_THROTTLE_REPEATS,
    YAW_UNWIND_THROTTLE_MODES,
    YAW_FRONTIER_THROTTLE_SLEW_PROFILE,
    YAW_FRONTIER_THROTTLE_SLEW_SPEED_STEERING_RAD,
    YAW_FRONTIER_LOWANGLE_UNWIND_PROFILE,
    YAW_FRONTIER_LOWANGLE_UNWIND_REPEATS,
    YAW_FRONTIER_LOWANGLE_UNWIND_SPEED_STEERING_RAD,
    YAW_LOW_ANGLE_RATE_PROFILE,
    YAW_LOW_ANGLE_RATE_SPEEDS_MPS,
    YAW_LOW_ANGLE_RATE_STEERING_RAD,
    YAW_LOW_ANGLE_RATE_MODES,
    YAW_LOW_ANGLE_RATE_REPEATS,
    YAW_MISMATCH_TRANSITION_PROFILE,
    YAW_MISMATCH_TRANSITION_REPEATS,
    YAW_MISMATCH_ONSET_SPEED_MPS,
    YAW_MISMATCH_ONSET_STEERING_RAD,
    YAW_MISMATCH_REVERSAL_SPEED_MPS,
    YAW_MISMATCH_REVERSAL_STEERING_RAD,
    YAW_CELL_MISMATCH_TRANSITION_PROFILE,
    YAW_CELL_MISMATCH_STEP_REPEATS,
    YAW_CELL_MISMATCH_RAMP_REPEATS,
    YAW_CELL_MISMATCH_SPEED_MPS,
    YAW_CELL_MISMATCH_STEERING_RAD,
    YAW_CELL_MISMATCH_COMMAND_RAD,
    YAW_CELL_MISMATCH_RAMP_S,
    YAW_ERROR_STEERING_EVENT_PROFILE,
    YAW_ERROR_STEERING_EVENT_MODES,
    YAW_ERROR_STEERING_EVENT_DELAYS_S,
    YAW_ERROR_COMMAND_GAP_PROFILE,
    YAW_ERROR_COMMAND_GAP_LOW_SPEED_PROFILE,
    YAW_ERROR_MIDSPEED_STEERING_PROFILE,
    YAW_ERROR_MIDSPEED_STEERING_SPEEDS_MPS,
    YAW_ERROR_MIDSPEED_STEERING_ANGLES_RAD,
    YAW_ERROR_HIGHSPEED_STEERING_PROFILE,
    YAW_ERROR_HIGHSPEED_STEERING_POINTS,
    YAW_ERROR_LOWSPEED_STEERING_PROFILE,
    YAW_ERROR_LOWSPEED_STEERING_POINTS,
    YAW_ERROR_COMMAND_GAP_MODES,
    YAW_ERROR_COMMAND_GAP_DELAYS_S,
    YAW_ERROR_COMMAND_GAP_POINTS,
    YAW_ERROR_COMMAND_GAP_LOW_SPEED_POINTS,
    YAW_ERROR_WHEELSPIN_PROFILE,
    YAW_ERROR_WHEELSPIN_POINTS,
    YAW_ERROR_WHEELSPIN_MODES,
    YAW_ERROR_LOWSPEED_WHEELSPIN_PROFILE,
    YAW_ERROR_LOWSPEED_WHEELSPIN_POINTS,
    YAW_ERROR_LOWSPEED_WHEELSPIN_DELTAS,
    YAW_LOW_SPEED_THROTTLE_CALIBRATION,
    _low_speed_wheelspin_feedforward,
    YAW_ERROR_MIDSPEED_THROTTLE_PROFILE,
    YAW_ERROR_MIDSPEED_THROTTLE_POINTS,
    YAW_ERROR_MIDSPEED_THROTTLE_DELTAS,
    YAW_ERROR_MIDSPEED_THROTTLE_DIRECTIONS,
    YAW_ERROR_RESIDUAL_GRID_PROFILE,
    YAW_ERROR_RESIDUAL_GRID_POINTS,
    YAW_ERROR_LOWSPEED_HIGHSTEER_PROFILE,
    YAW_ERROR_LOWSPEED_HIGHSTEER_POINTS,
    YAW_ERROR_HIGHSTEER_REVERSAL_PROFILE,
    YAW_ERROR_HIGHSTEER_REVERSAL_POINTS,
    YAW_ERROR_HIGHSTEER_REVERSAL_MODES,
    YAW_ERROR_CRAWL_THROTTLE_CALIBRATION_PROFILE,
    YAW_ERROR_CRAWL_THROTTLE_LEVELS,
    YAW_ERROR_CRAWL_THROTTLE_STEERING_RAD,
    YAW_ERROR_CRAWL_THROTTLE_REPEATS,
    YAW_ERROR_CRAWL_THROTTLE_HOLD_S,
    YAW_ERROR_CRAWL_STEERING_PROFILE,
    YAW_ERROR_CRAWL_STEERING_POINTS,
    YAW_ERROR_CRAWL_FINE_PROFILE,
    YAW_ERROR_CRAWL_FINE_POINTS,
    YAW_ERROR_SUBCRAWL_STEERING_PROFILE,
    YAW_ERROR_SUBCRAWL_STEERING_POINTS,
    YAW_TRANSIENT_PROFILES,
    SWERVE_THROTTLE_SLEW_PROFILES,
    SWERVE_THROTTLE_SLEW_FRONTIER_PROFILE,
    SWERVE_THROTTLE_SLEW_FRONTIER_SPEED_STEERING_RAD,
    SWERVE_THROTTLE_SLEW_11MPS_REPLICATION_PROFILE,
    SWERVE_THROTTLE_SLEW_11MPS_REPLICATION_SPEED_STEERING_RAD,
    SWERVE_THROTTLE_SLEW_MODERATE_PROFILE,
    SWERVE_THROTTLE_SLEW_MODERATE_SPEED_STEERING_RAD,
    SWERVE_THROTTLE_SLEW_LOWSTEER_PROFILE,
    SWERVE_THROTTLE_SLEW_LOWSTEER_SPEED_STEERING_RAD,
    SWERVE_THROTTLE_RATE_SWEEP_PROFILE,
    SWERVE_THROTTLE_RATE_SWEEP_HIGHSTEER_PROFILE,
    SWERVE_THROTTLE_RATE_FACTORIAL_PROFILE,
    SWERVE_THROTTLE_RATE_FACTORIAL_4P5_PROFILE,
    SWERVE_THROTTLE_RATE_FACTORIAL_4P5_SPEED_MPS,
    SWERVE_THROTTLE_RATE_FACTORIAL_6P5_PROFILE,
    SWERVE_THROTTLE_RATE_FACTORIAL_6P5_SPEED_MPS,
    SWERVE_THROTTLE_RATE_FACTORIAL_7P5_PROFILE,
    SWERVE_THROTTLE_RATE_FACTORIAL_7P5_SPEED_MPS,
    SWERVE_THROTTLE_RATE_FRONTIER_UP_PROFILE,
    SWERVE_THROTTLE_RATE_FRONTIER_UP_TRAIN_PROFILE,
    SWERVE_THROTTLE_RATE_FRONTIER_UP_VALIDATION_PROFILE,
    SWERVE_THROTTLE_RATE_FRONTIER_UP_SPEED_STEERING_RAD,
    SWERVE_THROTTLE_RATE_FRONTIER_UP_DELTA_NORM,
    SWERVE_THROTTLE_RATE_FRONTIER_UP_RAMP_DURATIONS_S,
    SWERVE_THROTTLE_RATE_RACE_DOMAIN_TRAIN_PROFILE,
    SWERVE_THROTTLE_RATE_RACE_DOMAIN_VALIDATION_PROFILE,
    SWERVE_THROTTLE_RATE_RACE_DOMAIN_STEERING_RAD,
    SWERVE_THROTTLE_RATE_RACE_DOMAIN_DELTAS_NORM,
    SWERVE_THROTTLE_RATE_RACE_DOMAIN_RAMP_DURATIONS_S,
    SWERVE_THROTTLE_RATE_RACE_DOMAIN_SPEEDS_MPS,
    SWERVE_THROTTLE_RATE_FACTORIAL_DELTAS_NORM,
    SWERVE_THROTTLE_RATE_FACTORIAL_RATES_NORM_PER_SEC,
    SWERVE_THROTTLE_RATE_SWEEP_SPEED_MPS,
    SWERVE_THROTTLE_RATE_SWEEP_STEERING_RAD,
    SWERVE_THROTTLE_RATE_SWEEP_RAMP_DURATIONS_S,
    SWERVE_THROTTLE_SLEW_SPEED_STEERING_RAD,
    SWERVE_THROTTLE_SLEW_APPROACH_S,
    SIM_RESET_HOLD_SEC,
    SIM_RESET_TIMEOUT_SEC,
    PROBE_START_TIMEOUT_SEC,
    _phase_steering_command,
    _is_yaw_transient_approach,
    _slew_probe_command,
    _slew_probe_command,
    build_schedule,
)
from analyze_throttle_slew_pairs import (
    SWERVE_EXPECTED_PAIRS_BY_PROFILE,
    SWERVE_EXPECTED_RESETS_BY_PROFILE,
    SWERVE_PROFILES,
    _average_rear_contact_speeds_by_sequence,
    _mean_window,
    _pair_match,
    _response_surface_rows,
    _rigid_body_acceleration_by_sequence,
    _sample_values,
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


class LowSpeedHighSteeringTransientScheduleTest(unittest.TestCase):
    def test_full_signed_speed_angle_matrix_has_matched_reset_approaches(self) -> None:
        phases = build_schedule(202610071, LOW_SPEED_TRANSIENT_PROFILE)
        manoeuvres = [phase for phase in phases
                      if phase.label.startswith("lowdyn_")]
        approaches = [phase for phase in phases
                      if phase.label.startswith("approach_lowdyn_")]
        self.assertEqual(len(manoeuvres), 24)
        self.assertEqual(len(approaches), 24)
        expected = {
            (speed, angle, sign)
            for speed in LOW_SPEED_TRANSIENT_SPEEDS_MPS
            for angle in LOW_SPEED_TRANSIENT_STEERING_RAD
            for sign in (-1, 1)
        }
        observed = set()
        for index, phase in enumerate(phases):
            if not phase.label.startswith("lowdyn_"):
                continue
            condition = phase.condition_pair_id
            self.assertIsNotNone(condition)
            speed, angle, sign = condition.split("_")
            observed.add((float(speed[1:]), float(angle[1:]), int(sign[4:])))
            self.assertEqual(phases[index - 2].label,
                             f"approach_lowdyn_{condition}")
            self.assertEqual(phases[index - 1].label,
                             f"settle_lowdyn_{condition}")
            self.assertTrue(phases[index - 2].reach_speed_target)
            self.assertEqual(phase.duration_s, 1.50)
            self.assertTrue(phase.validate_samples)
            self.assertFalse(phase.validate_speed)
            self.assertFalse(phase.validate_steering)
            self.assertEqual(phase.steering_waypoints[0], (0.0, 0.0))
            self.assertEqual(phase.steering_waypoints[-1], (1.45, 0.0))
        self.assertEqual(observed, expected)

    def test_seed_randomizes_order_without_changing_coverage(self) -> None:
        def labels(seed: int) -> list[str]:
            return [phase.label for phase in build_schedule(
                seed, LOW_SPEED_TRANSIENT_PROFILE)
                    if phase.label.startswith("lowdyn_")]

        first = labels(202610071)
        second = labels(202610072)
        self.assertEqual(set(first), set(second))
        self.assertNotEqual(first, second)


class YawTransitionScheduleTest(unittest.TestCase):
    def test_targeted_speed_steering_matrix_has_matched_reset_approaches(self) -> None:
        phases = build_schedule(202610071, YAW_TRANSIENT_PROFILE)
        manoeuvres = [phase for phase in phases
                      if phase.label.startswith("yawdyn_")]
        approaches = [phase for phase in phases
                      if phase.label.startswith("approach_yawdyn_")]
        self.assertEqual(len(manoeuvres), 18)
        self.assertEqual(len(approaches), 18)
        expected = {
            (speed, angle, sign)
            for speed in YAW_TRANSIENT_SPEEDS_MPS
            for angle in YAW_TRANSIENT_STEERING_RAD
            for sign in (-1, 1)
        }
        observed = set()
        for index, phase in enumerate(phases):
            if not phase.label.startswith("yawdyn_"):
                continue
            condition = phase.condition_pair_id
            self.assertIsNotNone(condition)
            speed, angle, sign = condition.split("_")
            observed.add((float(speed[1:]), float(angle[1:]), int(sign[4:])))
            self.assertEqual(phases[index - 2].label,
                             f"approach_yawdyn_{condition}")
            self.assertEqual(phases[index - 1].label,
                             f"settle_yawdyn_{condition}")
            self.assertTrue(phases[index - 2].reach_speed_target)
            self.assertEqual(phase.duration_s, 2.75)
            self.assertTrue(phase.validate_samples)
            self.assertFalse(phase.validate_speed)
            self.assertFalse(phase.validate_steering)
            self.assertEqual(phase.steering_waypoints[0], (0.0, 0.0))
            self.assertEqual(phase.steering_waypoints[-1], (2.60, 0.0))
        self.assertEqual(observed, expected)

    def test_seed_randomizes_order_without_changing_coverage(self) -> None:
        def labels(seed: int) -> list[str]:
            return [phase.label for phase in build_schedule(
                seed, YAW_TRANSIENT_PROFILE)
                    if phase.label.startswith("yawdyn_")]

        first = labels(202610071)
        second = labels(202610072)
        self.assertEqual(set(first), set(second))
        self.assertNotEqual(first, second)


class YawCrawlSpeedValidationScheduleTest(unittest.TestCase):
    def test_crawl_probe_speed_is_a_required_quality_gate(self) -> None:
        for profile, prefix, expected_count in (
                (YAW_ERROR_CRAWL_STEERING_PROFILE,
                 "probe_yawerr_crawl_", 96),
                (YAW_ERROR_CRAWL_FINE_PROFILE,
                 "probe_yawerr_crawl_fine_", 384)):
            with self.subTest(profile=profile):
                phases = build_schedule(202610084, profile)
                probes = [phase for phase in phases
                          if phase.label.startswith(prefix)]
                self.assertEqual(len(probes), expected_count)
                self.assertTrue(all(phase.validate_samples for phase in probes))
                self.assertTrue(all(phase.validate_speed for phase in probes))
                self.assertTrue(all(not phase.validate_steering
                                    for phase in probes))

    def test_low_throttle_calibration_resets_each_measured_condition(self) -> None:
        seed = 202610089
        phases = build_schedule(
            seed, YAW_ERROR_CRAWL_THROTTLE_CALIBRATION_PROFILE)
        probes = [phase for phase in phases
                  if phase.label.startswith(
                      "probe_yawerr_crawl_throttle_")]
        expected = {
            (throttle, steering, repetition)
            for throttle in YAW_ERROR_CRAWL_THROTTLE_LEVELS
            for steering in YAW_ERROR_CRAWL_THROTTLE_STEERING_RAD
            for repetition in range(1, YAW_ERROR_CRAWL_THROTTLE_REPEATS + 1)
        }
        observed = set()
        for index, phase in enumerate(phases):
            if not phase.label.startswith(
                    "probe_yawerr_crawl_throttle_"):
                continue
            approach, settle = phases[index - 2:index]
            self.assertTrue(_is_yaw_transient_approach(approach.label))
            self.assertTrue(approach.reach_speed_target)
            self.assertEqual(approach.throttle_mode, "fixed")
            self.assertEqual(approach.throttle_norm, 0.0)
            self.assertEqual(settle.throttle_norm, 0.0)
            self.assertEqual(settle.steering_rad, phase.steering_rad)
            self.assertEqual(phase.duration_s,
                             YAW_ERROR_CRAWL_THROTTLE_HOLD_S)
            self.assertEqual(phase.throttle_mode, "fixed")
            self.assertTrue(phase.validate_samples)
            self.assertFalse(phase.validate_speed)
            self.assertFalse(phase.validate_steering)
            self.assertIsNotNone(phase.condition_pair_id)
            throttle_text, angle_text, repeat_text = (
                phase.condition_pair_id.split("_"))
            observed.add((
                int(throttle_text[1:]) / 100.0,
                float(angle_text[1:]),
                int(repeat_text[1:]),
            ))
        self.assertEqual(len(probes), 70)
        self.assertEqual(len(phases), 3 * len(probes))
        self.assertEqual(observed, expected)
        repeated = build_schedule(
            seed, YAW_ERROR_CRAWL_THROTTLE_CALIBRATION_PROFILE)
        self.assertEqual([p.label for p in phases],
                         [p.label for p in repeated])
        other_seed = build_schedule(
            seed + 1, YAW_ERROR_CRAWL_THROTTLE_CALIBRATION_PROFILE)
        self.assertNotEqual([p.label for p in phases],
                            [p.label for p in other_seed])
        required_s = (
            sum(phase.duration_s for phase in phases)
            + len(probes) * (SIM_RESET_HOLD_SEC + SIM_RESET_TIMEOUT_SEC)
            + 5.0)
        self.assertLess(required_s, 1200.0)


class YawAtlasInterpolationScheduleTest(unittest.TestCase):
    def test_off_grid_points_are_reset_isolated_and_repeated_both_directions(self) -> None:
        phases = build_schedule(202610071, YAW_ATLAS_INTERPOLATION_PROFILE)
        probes = [phase for phase in phases
                  if phase.label.startswith("atlas_")]
        approaches = [phase for phase in phases
                      if phase.label.startswith("approach_atlas_")]
        expected = {
            (repeat, speed, angle, sign)
            for repeat in range(1, YAW_ATLAS_INTERPOLATION_REPEATS + 1)
            for speed, angle in YAW_ATLAS_INTERPOLATION_POINTS
            for sign in (-1, 1)
        }
        self.assertEqual(len(probes), len(expected))
        self.assertEqual(len(approaches), len(expected))
        observed = set()
        for index, phase in enumerate(phases):
            if not phase.label.startswith("atlas_"):
                continue
            condition = phase.condition_pair_id
            self.assertIsNotNone(condition)
            repeat, speed, angle, sign = condition.split("_")
            observed.add((int(repeat[1:]), float(speed[1:]),
                          float(angle[1:]), int(sign[4:])))
            self.assertEqual(phases[index - 2].label,
                             f"approach_atlas_{condition}")
            self.assertEqual(phases[index - 1].label,
                             f"settle_atlas_{condition}")
            self.assertTrue(phases[index - 2].reach_speed_target)
            self.assertTrue(_is_yaw_transient_approach(
                phases[index - 2].label))
            self.assertTrue(phase.validate_samples)
            self.assertFalse(phase.validate_speed)
            self.assertFalse(phase.validate_steering)
            self.assertEqual(phase.steering_waypoints[0], (0.0, 0.0))
            self.assertEqual(phase.steering_waypoints[-1], (2.75, 0.0))
        self.assertEqual(observed, expected)

    def test_seed_randomizes_condition_order_without_changing_the_matrix(self) -> None:
        def labels(seed: int) -> list[str]:
            return [phase.label for phase in build_schedule(
                seed, YAW_ATLAS_INTERPOLATION_PROFILE)
                    if phase.label.startswith("atlas_")]

        first = labels(202610071)
        second = labels(202610072)
        self.assertEqual(set(first), set(second))
        self.assertNotEqual(first, second)

    def test_final_offgrid_points_are_distinct_reset_isolated_and_both_signed(self) -> None:
        self.assertTrue(set(YAW_ATLAS_OFFGRID_FINAL_POINTS).isdisjoint(
            YAW_ATLAS_INTERPOLATION_POINTS))
        phases = build_schedule(202610073, YAW_ATLAS_OFFGRID_FINAL_PROFILE)
        probes = [phase for phase in phases
                  if phase.label.startswith("atlas_")]
        expected = {
            (repeat, speed, angle, sign)
            for repeat in range(1, YAW_ATLAS_INTERPOLATION_REPEATS + 1)
            for speed, angle in YAW_ATLAS_OFFGRID_FINAL_POINTS
            for sign in (-1, 1)
        }
        observed = set()
        for index, phase in enumerate(phases):
            if not phase.label.startswith("atlas_"):
                continue
            repeat, speed, angle, sign = phase.condition_pair_id.split("_")
            observed.add((int(repeat[1:]), float(speed[1:]),
                          float(angle[1:]), int(sign[4:])))
            self.assertEqual(phases[index - 2].label,
                             f"approach_atlas_{phase.condition_pair_id}")
            self.assertEqual(phases[index - 1].label,
                             f"settle_atlas_{phase.condition_pair_id}")
            self.assertTrue(phases[index - 2].reach_speed_target)
            self.assertTrue(phase.validate_samples)
            self.assertFalse(phase.validate_speed)
            self.assertFalse(phase.validate_steering)
        self.assertEqual(len(probes), 24)
        self.assertEqual(observed, expected)

    def test_extratrees_final_points_are_new_and_reset_isolated(self) -> None:
        points = set(YAW_ATLAS_EXTRATREES_FINAL_POINTS)
        self.assertEqual(len(points), 6)
        self.assertTrue(points.isdisjoint(YAW_ATLAS_INTERPOLATION_POINTS))
        self.assertTrue(points.isdisjoint(YAW_ATLAS_OFFGRID_FINAL_POINTS))
        phases = build_schedule(202610077, YAW_ATLAS_EXTRATREES_FINAL_PROFILE)
        probes = [phase for phase in phases
                  if phase.label.startswith("atlas_")]
        expected = {
            (repeat, speed, angle, sign)
            for repeat in range(1, YAW_ATLAS_INTERPOLATION_REPEATS + 1)
            for speed, angle in YAW_ATLAS_EXTRATREES_FINAL_POINTS
            for sign in (-1, 1)
        }
        observed = set()
        for index, phase in enumerate(phases):
            if not phase.label.startswith("atlas_"):
                continue
            repeat, speed, angle, sign = phase.condition_pair_id.split("_")
            observed.add((int(repeat[1:]), float(speed[1:]),
                          float(angle[1:]), int(sign[4:])))
            self.assertTrue(phase.validate_samples)
            self.assertFalse(phase.validate_speed)
            self.assertFalse(phase.validate_steering)
            self.assertEqual(phases[index - 2].label,
                             f"approach_atlas_{phase.condition_pair_id}")
            self.assertEqual(phases[index - 1].label,
                             f"settle_atlas_{phase.condition_pair_id}")
            self.assertTrue(phases[index - 2].reach_speed_target)
        self.assertEqual(len(probes), 24)
        self.assertEqual(observed, expected)

    def test_extratrees_highsteer_final_points_are_supported_and_offgrid(self) -> None:
        points = set(YAW_ATLAS_EXTRATREES_HIGHSTEER_FINAL_POINTS)
        self.assertEqual(len(points), 4)
        self.assertTrue(all(8.0 < speed < 10.0 and 0.30 <= angle <= 0.43
                            for speed, angle in points))
        phases = build_schedule(
            202610078, YAW_ATLAS_EXTRATREES_HIGHSTEER_FINAL_PROFILE)
        probes = [phase for phase in phases
                  if phase.label.startswith("atlas_")]
        expected = {
            (repeat, speed, angle, sign)
            for repeat in range(1, YAW_ATLAS_INTERPOLATION_REPEATS + 1)
            for speed, angle in YAW_ATLAS_EXTRATREES_HIGHSTEER_FINAL_POINTS
            for sign in (-1, 1)
        }
        observed = set()
        for phase in probes:
            repeat, speed, angle, sign = phase.condition_pair_id.split("_")
            observed.add((int(repeat[1:]), float(speed[1:]),
                          float(angle[1:]), int(sign[4:])))
            self.assertTrue(phase.validate_samples)
            self.assertFalse(phase.validate_speed)
            self.assertFalse(phase.validate_steering)
        self.assertEqual(len(probes), 16)
        self.assertEqual(observed, expected)

    def test_highsteer_speed_surface_train_covers_full_steering_edge(self) -> None:
        points = set(YAW_HIGHSTEER_SPEED_SURFACE_TRAIN_POINTS)
        self.assertEqual(len(points), 12)
        self.assertEqual({angle for _, angle in points},
                         {0.350, 0.425, 0.475, 0.500})
        self.assertEqual({speed for speed, _ in points},
                         {8.25, 8.75, 9.25})
        phases = build_schedule(
            202610079, YAW_HIGHSTEER_SPEED_SURFACE_TRAIN_PROFILE)
        probes = [phase for phase in phases
                  if phase.label.startswith("atlas_")]
        expected = {
            (repeat, speed, angle, sign)
            for repeat in range(1, YAW_HIGHSTEER_SPEED_SURFACE_TRAIN_REPEATS + 1)
            for speed, angle in points
            for sign in (-1, 1)
        }
        observed = set()
        for index, phase in enumerate(phases):
            if not phase.label.startswith("atlas_"):
                continue
            repeat, speed, angle, sign = phase.condition_pair_id.split("_")
            observed.add((int(repeat[1:]), float(speed[1:]),
                          float(angle[1:]), int(sign[4:])))
            self.assertTrue(phase.validate_samples)
            self.assertFalse(phase.validate_speed)
            self.assertFalse(phase.validate_steering)
            self.assertEqual(phases[index - 2].label,
                             f"approach_atlas_{phase.condition_pair_id}")
            self.assertEqual(phases[index - 1].label,
                             f"settle_atlas_{phase.condition_pair_id}")
            self.assertTrue(phases[index - 2].reach_speed_target)
        self.assertEqual(len(probes), 48)
        self.assertEqual(observed, expected)


class YawLowAngleRateScheduleTest(unittest.TestCase):
    def test_reset_isolated_pair_matrix_covers_step_ramp_both_signs(self) -> None:
        speed = 4.25
        phases = build_schedule(202610071, YAW_LOW_ANGLE_RATE_PROFILE, speed)
        probes = [phase for phase in phases if phase.label.startswith("lowyaw_")]
        approaches = [phase for phase in phases
                      if phase.label.startswith("approach_lowyaw_")]
        expected = {
            (repeat, speed, angle, mode, sign)
            for repeat in range(1, YAW_LOW_ANGLE_RATE_REPEATS + 1)
            for angle in YAW_LOW_ANGLE_RATE_STEERING_RAD
            for mode, _ in YAW_LOW_ANGLE_RATE_MODES
            for sign in (-1, 1)
        }
        self.assertEqual(len(probes), len(expected))
        self.assertEqual(len(approaches), len(expected))
        observed = set()
        for index, phase in enumerate(phases):
            if not phase.label.startswith("lowyaw_"):
                continue
            self.assertTrue(phase.condition_pair_id)
            repeat, speed_token, angle_token, rate_token, turn_token = (
                phase.condition_pair_id.split("_"))
            condition = (int(repeat[1:]), float(speed_token[1:]),
                         float(angle_token[1:]), rate_token[4:],
                         int(turn_token[4:]))
            observed.add(condition)
            self.assertEqual(phases[index - 2].label,
                             f"approach_lowyaw_{phase.condition_pair_id}")
            self.assertEqual(phases[index - 1].label,
                             f"settle_lowyaw_{phase.condition_pair_id}")
            self.assertTrue(phases[index - 2].reach_speed_target)
            self.assertEqual(phase.duration_s, 2.75)
            self.assertTrue(phase.validate_samples)
            self.assertFalse(phase.validate_speed)
            self.assertFalse(phase.validate_steering)
            first_turn_duration = phase.steering_waypoints[1][0]
            self.assertEqual(first_turn_duration,
                             0.025 if rate_token == "ratestep" else 0.30)
            self.assertEqual(phase.steering_waypoints[0], (0.0, 0.0))
            self.assertEqual(phase.steering_waypoints[-1], (2.75, 0.0))
        self.assertEqual(observed, expected)

    def test_speed_blocks_are_explicit_and_invalid_speed_is_rejected(self) -> None:
        self.assertIn(4.25, YAW_LOW_ANGLE_RATE_SPEEDS_MPS)
        for speed in (4.25, 6.25, 8.25, 10.25):
            self.assertEqual(len(build_schedule(
                202610071, YAW_LOW_ANGLE_RATE_PROFILE, speed)), 96)
        with self.assertRaises(ValueError):
            build_schedule(202610071, YAW_LOW_ANGLE_RATE_PROFILE, 5.25)


class YawFullbandGapfillScheduleTest(unittest.TestCase):
    def test_gapfill_uses_exact_supported_cell_centers_and_randomized_pairs(self) -> None:
        phases = build_schedule(202610074, YAW_FULLBAND_GAPFILL_PROFILE)
        probes = [phase for phase in phases
                  if phase.label.startswith("yawgap_")]
        expected = {
            (repeat, speed, angle, sign)
            for repeat in range(1, YAW_FULLBAND_GAPFILL_REPEATS + 1)
            for speed, angle in YAW_FULLBAND_GAPFILL_POINTS
            for sign in (-1, 1)
        }
        observed = set()
        for index, phase in enumerate(phases):
            if not phase.label.startswith("yawgap_"):
                continue
            repeat, speed, angle, sign = phase.condition_pair_id.split("_")
            observed.add((int(repeat[1:]), float(speed[1:]),
                          float(angle[1:]), int(sign[4:])))
            self.assertEqual(phases[index - 2].label,
                             f"approach_yawgap_{phase.condition_pair_id}")
            self.assertEqual(phases[index - 1].label,
                             f"settle_yawgap_{phase.condition_pair_id}")
            self.assertTrue(phases[index - 2].reach_speed_target)
            self.assertEqual(phase.steering_amplitude_rad, float(angle[1:]))
            self.assertEqual(phase.steering_waypoints[0], (0.0, 0.0))
            self.assertEqual(phase.steering_waypoints[-1], (2.75, 0.0))
            self.assertTrue(phase.validate_samples)
            self.assertFalse(phase.validate_speed)
            self.assertFalse(phase.validate_steering)
        self.assertEqual(len(probes), len(expected))
        self.assertEqual(observed, expected)

    def test_order_changes_by_seed_without_changing_gapfill_matrix(self) -> None:
        def labels(seed: int) -> list[str]:
            return [phase.label for phase in build_schedule(
                seed, YAW_FULLBAND_GAPFILL_PROFILE)
                    if phase.label.startswith("yawgap_")]

        first, second = labels(202610074), labels(202610075)
        self.assertEqual(set(first), set(second))
        self.assertNotEqual(first, second)


class YawUnwindThrottleScheduleTest(unittest.TestCase):
    def test_step_ramp_pairs_cover_both_steering_directions_and_repeats(self) -> None:
        phases = build_schedule(202610076, YAW_UNWIND_THROTTLE_PROFILE)
        probes = [phase for phase in phases
                  if phase.label.startswith("yawbrake_")]
        expected = {
            (repeat, mode, sign)
            for repeat in range(1, YAW_UNWIND_THROTTLE_REPEATS + 1)
            for mode in YAW_UNWIND_THROTTLE_MODES
            for sign in (-1, 1)
        }
        observed = set()
        self.assertEqual(len(probes), len(expected))
        self.assertEqual(len(phases), 3 * len(expected))
        for index, phase in enumerate(phases):
            if not phase.label.startswith("yawbrake_"):
                continue
            repeat_token, mode, turn_token = phase.condition_pair_id.split("_")
            observed.add((int(repeat_token[1:]), mode, int(turn_token[4:])))
            self.assertEqual(phases[index - 2].label,
                             f"approach_yawbrake_{phase.condition_pair_id}")
            self.assertEqual(phases[index - 1].label,
                             f"settle_yawbrake_{phase.condition_pair_id}")
            self.assertTrue(phases[index - 2].reach_speed_target)
            self.assertEqual(phase.speed_target_mps,
                             YAW_UNWIND_THROTTLE_SPEED_MPS)
            self.assertEqual(phase.duration_s, 4.0)
            self.assertTrue(phase.validate_samples)
            self.assertFalse(phase.validate_speed)
            self.assertFalse(phase.validate_steering)
            self.assertTrue(phase.settle_before_probe)
            self.assertTrue(phase.probe_race_domain)
            self.assertEqual(phase.throttle_mode, "slew_probe")
            self.assertEqual(phase.throttle_start_norm,
                             YAW_UNWIND_THROTTLE_START_NORM)
            self.assertEqual(phase.throttle_end_norm,
                             YAW_UNWIND_THROTTLE_END_NORM)
            self.assertEqual(phase.throttle_profile, mode)
            self.assertEqual(phase.throttle_stimulus_delay_s, 0.70)
            self.assertEqual(phase.throttle_ramp_duration_s,
                             0.30 if mode == "ramp" else 0.0)
            self.assertEqual(phase.steering_amplitude_rad,
                             YAW_UNWIND_THROTTLE_STEERING_RAD)
            self.assertEqual(phase.steering_waypoints[0], (0.0, 0.0))
            self.assertEqual(phase.steering_waypoints[-1], (4.0, 0.0))
        self.assertEqual(observed, expected)

    def test_pair_order_is_randomized_by_seed(self) -> None:
        def labels(seed: int) -> list[str]:
            return [phase.label for phase in build_schedule(
                seed, YAW_UNWIND_THROTTLE_PROFILE)
                    if phase.label.startswith("yawbrake_")]

        first = labels(202610076)
        second = labels(202610077)
        self.assertEqual(set(first), set(second))
        self.assertNotEqual(first, second)

    def test_step_and_ramp_apply_the_same_final_throttle(self) -> None:
        phases = [phase for phase in build_schedule(
            202610076, YAW_UNWIND_THROTTLE_PROFILE)
                  if phase.label.startswith("yawbrake_")]
        step = next(phase for phase in phases if phase.throttle_profile == "step")
        ramp = next(phase for phase in phases if phase.throttle_profile == "ramp")
        self.assertEqual(_slew_probe_command(step, 0.675),
                         YAW_UNWIND_THROTTLE_START_NORM)
        self.assertEqual(_slew_probe_command(step, 0.725),
                         YAW_UNWIND_THROTTLE_END_NORM)
        self.assertEqual(_slew_probe_command(ramp, 0.675),
                         YAW_UNWIND_THROTTLE_START_NORM)
        self.assertAlmostEqual(
            _slew_probe_command(ramp, 0.85),
            0.5 * (YAW_UNWIND_THROTTLE_START_NORM
                   + YAW_UNWIND_THROTTLE_END_NORM),
        )
        self.assertEqual(_slew_probe_command(ramp, 1.0),
                         YAW_UNWIND_THROTTLE_END_NORM)


class YawCommandMismatchScheduleTest(unittest.TestCase):
    def test_reset_isolated_step_ramp_onset_and_full_steer_reversal(self) -> None:
        phases = build_schedule(202610078, YAW_MISMATCH_TRANSITION_PROFILE)
        probes = [phase for phase in phases if phase.label.startswith("yawmis_")]
        self.assertEqual(len(probes), 16)
        self.assertEqual(len(phases), 3 * len(probes))
        paired_modes = {}
        for index, phase in enumerate(phases):
            if not phase.label.startswith("yawmis_"):
                continue
            tokens = phase.condition_pair_id.split("_")
            repeat, manoeuvre, speed_token, angle_token, mode, turn = tokens
            speed = YAW_MISMATCH_REVERSAL_SPEED_MPS if manoeuvre == "reversal" else YAW_MISMATCH_ONSET_SPEED_MPS
            angle = YAW_MISMATCH_REVERSAL_STEERING_RAD if manoeuvre == "reversal" else YAW_MISMATCH_ONSET_STEERING_RAD
            self.assertAlmostEqual(float(speed_token[1:]), speed)
            self.assertAlmostEqual(float(angle_token[1:]), angle)
            self.assertEqual(phase.speed_target_mps, speed)
            self.assertEqual(phases[index - 2].label,
                             f"approach_yawmis_{phase.condition_pair_id}")
            self.assertEqual(phases[index - 1].label,
                             f"settle_yawmis_{phase.condition_pair_id}")
            self.assertTrue(phases[index - 2].reach_speed_target)
            self.assertTrue(_is_yaw_transient_approach(phases[index - 2].label))
            self.assertEqual(phases[index - 2].duration_s, 10.0)
            self.assertEqual(phases[index - 1].duration_s, 0.75)
            self.assertTrue(phase.validate_samples)
            self.assertFalse(phase.validate_speed)
            self.assertFalse(phase.validate_steering)
            self.assertEqual(phase.steering_profile, "waypoints")
            self.assertAlmostEqual(phase.steering_amplitude_rad, angle)
            paired_modes.setdefault((repeat, manoeuvre, speed_token,
                                     angle_token, turn), set()).add(mode)
            if manoeuvre == "onset":
                sign = float(turn[4:])
                target = sign * angle
                if mode == "step":
                    self.assertEqual(phase.duration_s, 2.0)
                    self.assertEqual(_phase_steering_command(phase, 0.225), target)
                else:
                    self.assertAlmostEqual(
                        _phase_steering_command(phase, 0.35), 0.5 * target)
            else:
                sign = float(turn[4:])
                self.assertEqual(phase.duration_s, 2.3)
                self.assertAlmostEqual(
                    _phase_steering_command(phase, 0.70), sign * angle)
                if mode == "step":
                    self.assertEqual(_phase_steering_command(phase, 0.725),
                                     -sign * angle)
                else:
                    self.assertAlmostEqual(
                        _phase_steering_command(phase, 0.85), 0.0)
        expected_pair_count = YAW_MISMATCH_TRANSITION_REPEATS * 2 * 2
        self.assertEqual(len(paired_modes), expected_pair_count)
        self.assertTrue(all(modes == {"step", "ramp"}
                            for modes in paired_modes.values()))

    def test_seed_randomizes_order_without_losing_paired_conditions(self) -> None:
        def labels(seed: int) -> list[str]:
            return [phase.condition_pair_id for phase in build_schedule(
                seed, YAW_MISMATCH_TRANSITION_PROFILE)
                    if phase.label.startswith("yawmis_")]

        first, second = labels(202610078), labels(202610079)
        self.assertEqual(set(first), set(second))
        self.assertNotEqual(first, second)


class YawFittedCellMismatchScheduleTest(unittest.TestCase):
    def test_targets_the_heldout_mismatch_inside_the_fitted_cell(self) -> None:
        phases = build_schedule(202610079,
                                YAW_CELL_MISMATCH_TRANSITION_PROFILE)
        probes = [p for p in phases if p.label.startswith("yawmisgap_")]
        expected = {
            (repeat, mode, sign)
            for mode in ("step", "ramp")
            for repeat in range(
                1,
                (YAW_CELL_MISMATCH_STEP_REPEATS if mode == "step"
                 else YAW_CELL_MISMATCH_RAMP_REPEATS) + 1,
            )
            for sign in (-1.0, 1.0)
        }
        self.assertEqual(len(probes), len(expected))
        self.assertEqual(len(phases), 3 * len(expected))
        observed = set()
        for index, phase in enumerate(phases):
            if not phase.label.startswith("yawmisgap_"):
                continue
            repeat_token, mode, speed_token, angle_token, turn_token = (
                phase.condition_pair_id.split("_"))
            repeat = int(repeat_token[1:])
            sign = float(turn_token[4:])
            observed.add((repeat, mode, sign))
            self.assertAlmostEqual(float(speed_token[1:]),
                                   YAW_CELL_MISMATCH_SPEED_MPS)
            self.assertAlmostEqual(float(angle_token[1:]),
                                   YAW_CELL_MISMATCH_STEERING_RAD)
            self.assertEqual(phase.speed_target_mps,
                             YAW_CELL_MISMATCH_SPEED_MPS)
            self.assertEqual(phase.duration_s, 1.5)
            self.assertEqual(phases[index - 2].label,
                             f"approach_yawmisgap_{phase.condition_pair_id}")
            self.assertEqual(phases[index - 1].label,
                             f"settle_yawmisgap_{phase.condition_pair_id}")
            self.assertTrue(phases[index - 2].reach_speed_target)
            self.assertTrue(_is_yaw_transient_approach(phases[index - 2].label))
            self.assertEqual(phases[index - 2].duration_s, 10.0)
            self.assertEqual(phases[index - 1].duration_s, 0.75)
            self.assertEqual(phases[index - 1].steering_rad,
                             sign * YAW_CELL_MISMATCH_STEERING_RAD)
            self.assertTrue(phase.validate_samples)
            self.assertFalse(phase.validate_speed)
            self.assertFalse(phase.validate_steering)
            start = sign * YAW_CELL_MISMATCH_STEERING_RAD
            target = -sign * YAW_CELL_MISMATCH_COMMAND_RAD
            self.assertAlmostEqual(target - start, -sign * 0.1184, places=4)
            self.assertAlmostEqual(_phase_steering_command(phase, 0.20), start)
            if mode == "step":
                self.assertAlmostEqual(
                    _phase_steering_command(phase, 0.225), target)
            else:
                midpoint = 0.5 * (start + target)
                self.assertAlmostEqual(
                    _phase_steering_command(
                        phase, 0.20 + 0.5 * YAW_CELL_MISMATCH_RAMP_S),
                    midpoint)
                self.assertAlmostEqual(
                    _phase_steering_command(
                        phase, 0.20 + YAW_CELL_MISMATCH_RAMP_S), target)
        self.assertEqual(observed, expected)

    def test_step_ramp_pairing_is_seeded_and_randomized(self) -> None:
        def labels(seed: int) -> list[str]:
            return [p.condition_pair_id for p in build_schedule(
                seed, YAW_CELL_MISMATCH_TRANSITION_PROFILE)
                    if p.label.startswith("yawmisgap_")]

        first = labels(202610079)
        self.assertEqual(first, labels(202610079))
        self.assertNotEqual(first, labels(202610080))


class YawErrorSupportGapfillScheduleTest(unittest.TestCase):
    def test_every_yaw_transient_schedule_is_available_from_cli(self) -> None:
        self.assertTrue(set(YAW_TRANSIENT_PROFILES)
                        <= set(EXPERIMENT_PROFILE_CHOICES))

    def test_steering_event_grid_pairs_step_and_ramp_at_two_event_ages(self) -> None:
        phases = build_schedule(202610071, YAW_ERROR_STEERING_EVENT_PROFILE)
        probes = [p for p in phases
                  if p.label.startswith("probe_yawerr_steer_")]
        self.assertEqual(len(probes), 32)
        self.assertEqual(len(phases), 3 * len(probes))
        paired: dict[str, set[str]] = {}
        for index, phase in enumerate(phases):
            if not phase.label.startswith("probe_yawerr_steer_"):
                continue
            pair_id = phase.condition_pair_id
            self.assertIsNotNone(pair_id)
            mode = "step" if "_step" in phase.label else "ramp"
            paired.setdefault(pair_id, set()).add(mode)
            approach, settle = phases[index - 2:index]
            self.assertTrue(_is_yaw_transient_approach(approach.label))
            self.assertTrue(approach.reach_speed_target)
            self.assertEqual(settle.steering_rad,
                             phase.steering_waypoints[0][1])
            self.assertTrue(phase.validate_samples)
            self.assertFalse(phase.validate_speed)
            self.assertFalse(phase.validate_steering)
            delay = float(pair_id.split("delay", 1)[1])
            target = phase.steering_waypoints[2][1]
            ramp_end = delay + (0.025 if mode == "step" else 0.30)
            self.assertAlmostEqual(
                _phase_steering_command(phase, ramp_end), target)
        self.assertEqual(len(paired), 16)
        self.assertTrue(all(modes == {"step", "ramp"}
                            for modes in paired.values()))
        self.assertEqual(set(YAW_ERROR_STEERING_EVENT_DELAYS_S), {0.25, 0.75})
        required_s = (
            sum(p.duration_s for p in phases)
            + sum(_is_yaw_transient_approach(p.label) for p in phases)
            * (SIM_RESET_HOLD_SEC + SIM_RESET_TIMEOUT_SEC) + 5.0)
        self.assertLess(required_s, 1200.0)

    def test_highspeed_transition_grid_stays_inside_measured_frontier(self) -> None:
        phases = build_schedule(202610076,
                                YAW_ERROR_HIGHSPEED_STEERING_PROFILE)
        probes = [p for p in phases
                  if p.label.startswith("probe_yawerr_high_")]
        self.assertEqual(len(probes), 64)
        self.assertEqual(len(phases), 3 * len(probes))
        paired: dict[str, set[str]] = {}
        observed = set()
        for index, phase in enumerate(phases):
            if not phase.label.startswith("probe_yawerr_high_"):
                continue
            pair_id = phase.condition_pair_id
            self.assertIsNotNone(pair_id)
            mode = "step" if "_step" in phase.label else "ramp"
            paired.setdefault(pair_id, set()).add(mode)
            approach, settle = phases[index - 2:index]
            self.assertTrue(_is_yaw_transient_approach(approach.label))
            self.assertTrue(approach.reach_speed_target)
            self.assertEqual(phase.speed_target_mps,
                             approach.speed_target_mps)
            family, speed_text, angle_text, turn_text, delay_text = (
                pair_id.split("_", 4))
            speed = float(speed_text[1:])
            angle = float(angle_text[1:])
            sign = float(turn_text[4:])
            delay = float(delay_text.removeprefix("delay"))
            observed.add((family, speed, angle, sign, delay))
            self.assertIn((speed, angle), YAW_ERROR_HIGHSPEED_STEERING_POINTS)
            self.assertEqual(settle.steering_rad,
                             phase.steering_waypoints[0][1])
            start = sign * angle if family == "unwind" else 0.0
            target = 0.0 if family == "unwind" else sign * angle
            self.assertAlmostEqual(phase.steering_waypoints[0][1], start)
            ramp_end = delay + (0.025 if mode == "step" else 0.30)
            self.assertAlmostEqual(_phase_steering_command(phase, ramp_end),
                                   target)
        self.assertEqual(len(paired), 32)
        self.assertTrue(all(modes == {"step", "ramp"}
                            for modes in paired.values()))
        expected = {
            (family, speed, angle, sign, delay)
            for family in ("onset", "unwind")
            for speed, angle in YAW_ERROR_HIGHSPEED_STEERING_POINTS
            for sign in (-1.0, 1.0)
            for delay in YAW_ERROR_COMMAND_GAP_DELAYS_S
        }
        self.assertEqual(observed, expected)
        required_s = (
            sum(p.duration_s for p in phases)
            + sum(_is_yaw_transient_approach(p.label) for p in phases)
            * (SIM_RESET_HOLD_SEC + SIM_RESET_TIMEOUT_SEC) + 5.0)
        self.assertLess(required_s, 1200.0)

    def test_lowspeed_highsteer_grid_pairs_both_turn_directions(self) -> None:
        phases = build_schedule(202610077,
                                YAW_ERROR_LOWSPEED_STEERING_PROFILE)
        probes = [p for p in phases
                  if p.label.startswith("probe_yawerr_low_")]
        self.assertEqual(len(probes), 64)
        self.assertEqual(len(phases), 3 * len(probes))
        paired: dict[str, set[str]] = {}
        observed = set()
        for index, phase in enumerate(phases):
            if not phase.label.startswith("probe_yawerr_low_"):
                continue
            pair_id = phase.condition_pair_id
            self.assertIsNotNone(pair_id)
            mode = "step" if "_step" in phase.label else "ramp"
            paired.setdefault(pair_id, set()).add(mode)
            approach, settle = phases[index - 2:index]
            self.assertTrue(_is_yaw_transient_approach(approach.label))
            self.assertTrue(approach.reach_speed_target)
            family, speed_text, angle_text, turn_text, delay_text = (
                pair_id.split("_", 4))
            speed = float(speed_text[1:])
            angle = float(angle_text[1:])
            sign = float(turn_text[4:])
            delay = float(delay_text.removeprefix("delay"))
            observed.add((family, speed, angle, sign, delay))
            self.assertIn((speed, angle), YAW_ERROR_LOWSPEED_STEERING_POINTS)
            start = sign * angle if family == "reversal" else 0.0
            target = -start if family == "reversal" else sign * angle
            self.assertAlmostEqual(settle.steering_rad, start)
            self.assertAlmostEqual(phase.steering_waypoints[0][1], start)
            self.assertAlmostEqual(phase.steering_waypoints[-2][1], target)
            ramp_end = delay + (0.025 if mode == "step" else 0.30)
            self.assertAlmostEqual(_phase_steering_command(phase, ramp_end),
                                   target)
        self.assertEqual(len(paired), 32)
        self.assertTrue(all(modes == {"step", "ramp"}
                            for modes in paired.values()))
        expected = {
            (family, speed, angle, sign, delay)
            for family in ("onset", "reversal")
            for speed, angle in YAW_ERROR_LOWSPEED_STEERING_POINTS
            for sign in (-1.0, 1.0)
            for delay in YAW_ERROR_COMMAND_GAP_DELAYS_S
        }
        self.assertEqual(observed, expected)
        required_s = (
            sum(p.duration_s for p in phases)
            + sum(_is_yaw_transient_approach(p.label) for p in phases)
            * (SIM_RESET_HOLD_SEC + SIM_RESET_TIMEOUT_SEC) + 5.0)
        self.assertLess(required_s, 1200.0)

    def test_small_angle_command_gap_grid_brackets_speed_and_angle(self) -> None:
        profiles = (
            (YAW_ERROR_COMMAND_GAP_PROFILE, YAW_ERROR_COMMAND_GAP_POINTS),
            (YAW_ERROR_COMMAND_GAP_LOW_SPEED_PROFILE,
             YAW_ERROR_COMMAND_GAP_LOW_SPEED_POINTS),
        )
        for profile, points in profiles:
            with self.subTest(profile=profile):
                phases = build_schedule(202610072, profile)
                probes = [p for p in phases
                          if p.label.startswith("probe_yawerr_gap_")]
                self.assertEqual(len(probes), len(points) * 8)
                self.assertEqual(len(phases), 3 * len(probes))
                paired: dict[str, set[str]] = {}
                for index, phase in enumerate(phases):
                    if not phase.label.startswith("probe_yawerr_gap_"):
                        continue
                    pair_id = phase.condition_pair_id
                    mode = ("step" if phase.label.endswith("_step")
                            else "ramp")
                    paired.setdefault(pair_id, set()).add(mode)
                    approach, settle = phases[index - 2:index]
                    self.assertTrue(_is_yaw_transient_approach(
                        approach.label))
                    self.assertEqual(phase.speed_target_mps,
                                     approach.speed_target_mps)
                    self.assertAlmostEqual(settle.steering_rad,
                                           phase.steering_waypoints[0][1])
                    sign = 1.0 if "turn+1" in pair_id else -1.0
                    delay = float(pair_id.split("delay", 1)[1])
                    target = -sign * YAW_CELL_MISMATCH_COMMAND_RAD
                    self.assertAlmostEqual(
                        phase.steering_waypoints[-1][1], target)
                    end = delay + (0.025 if mode == "step" else 0.30)
                    self.assertAlmostEqual(
                        _phase_steering_command(phase, end), target)
                self.assertEqual(len(paired), len(points) * 4)
                self.assertTrue(all(modes == {"step", "ramp"}
                                    for modes in paired.values()))
                self.assertEqual(set(points), set(
                    YAW_ERROR_COMMAND_GAP_POINTS
                    if profile == YAW_ERROR_COMMAND_GAP_PROFILE
                    else YAW_ERROR_COMMAND_GAP_LOW_SPEED_POINTS))
        self.assertEqual(set(YAW_ERROR_COMMAND_GAP_DELAYS_S), {0.25, 0.75})

    def test_midspeed_steering_transition_grid_pairs_onset_and_reversal(self) -> None:
        phases = build_schedule(202610075, YAW_ERROR_MIDSPEED_STEERING_PROFILE)
        probes = [p for p in phases
                  if p.label.startswith("probe_yawerr_mid_")]
        self.assertEqual(len(probes), 64)
        self.assertEqual(len(phases), 3 * len(probes))
        paired: dict[str, set[str]] = {}
        observed_cells = set()
        for index, phase in enumerate(phases):
            if not phase.label.startswith("probe_yawerr_mid_"):
                continue
            pair_id = phase.condition_pair_id
            self.assertIsNotNone(pair_id)
            mode = "step" if "_step" in phase.label else "ramp"
            paired.setdefault(pair_id, set()).add(mode)
            approach, settle = phases[index - 2:index]
            self.assertTrue(_is_yaw_transient_approach(approach.label))
            self.assertTrue(approach.reach_speed_target)
            self.assertEqual(approach.speed_target_mps,
                             phase.speed_target_mps)
            self.assertEqual(settle.steering_rad,
                             phase.steering_waypoints[0][1])
            self.assertTrue(phase.validate_samples)
            self.assertFalse(phase.validate_speed)
            self.assertFalse(phase.validate_steering)
            family, speed_text, angle_text, turn_text, delay_text = (
                pair_id.split("_", 4))
            speed = float(speed_text[1:])
            angle = float(angle_text[1:])
            sign = float(turn_text[4:])
            delay = float(delay_text.removeprefix("delay"))
            observed_cells.add((family, speed, angle, sign, delay))
            self.assertIn(speed, YAW_ERROR_MIDSPEED_STEERING_SPEEDS_MPS)
            self.assertIn(angle, YAW_ERROR_MIDSPEED_STEERING_ANGLES_RAD)
            start = sign * angle if family == "reversal" else 0.0
            target = -start if family == "reversal" else sign * angle
            self.assertAlmostEqual(phase.steering_waypoints[0][1], start)
            ramp_end = delay + (0.025 if mode == "step" else 0.30)
            self.assertAlmostEqual(_phase_steering_command(phase, ramp_end),
                                   target)
        self.assertEqual(len(paired), 32)
        self.assertTrue(all(modes == {"step", "ramp"}
                            for modes in paired.values()))
        expected_cells = {
            (family, speed, angle, sign, delay)
            for family in ("onset", "reversal")
            for speed in YAW_ERROR_MIDSPEED_STEERING_SPEEDS_MPS
            for angle in YAW_ERROR_MIDSPEED_STEERING_ANGLES_RAD
            for sign in (-1.0, 1.0)
            for delay in YAW_ERROR_COMMAND_GAP_DELAYS_S
        }
        self.assertEqual(observed_cells, expected_cells)
        required_s = (
            sum(p.duration_s for p in phases)
            + sum(_is_yaw_transient_approach(p.label) for p in phases)
            * (SIM_RESET_HOLD_SEC + SIM_RESET_TIMEOUT_SEC) + 5.0)
        self.assertLess(required_s, 1200.0)

    def test_wheelspin_grid_pairs_throttle_steps_and_ramps_in_both_turns(self) -> None:
        phases = build_schedule(202610073, YAW_ERROR_WHEELSPIN_PROFILE)
        probes = [p for p in phases
                  if p.label.startswith("probe_yawerr_wheel_")]
        self.assertEqual(len(probes), 48)
        self.assertEqual(len(phases), 2 * len(probes))
        paired: dict[str, set[str]] = {}
        for index, phase in enumerate(phases):
            if not phase.label.startswith("probe_yawerr_wheel_"):
                continue
            approach = phases[index - 1]
            self.assertTrue(_is_yaw_transient_approach(approach.label))
            self.assertTrue(phase.settle_before_probe)
            self.assertEqual(phase.throttle_mode, "slew_probe")
            self.assertTrue(phase.probe_race_domain)
            self.assertTrue(phase.validate_samples)
            pair_id = phase.condition_pair_id
            mode = phase.throttle_profile
            paired.setdefault(pair_id, set()).add(mode)
            start = phase.throttle_start_norm
            end = phase.throttle_end_norm
            self.assertGreaterEqual(start, 0.0)
            self.assertLessEqual(start, 0.5)
            self.assertGreaterEqual(end, 0.0)
            self.assertLessEqual(end, 0.5)
            if mode == "step":
                self.assertEqual(_slew_probe_command(phase, 0.80), start)
                self.assertEqual(_slew_probe_command(phase, 0.85), end)
            else:
                self.assertAlmostEqual(
                    _slew_probe_command(phase, 0.95),
                    0.5 * (start + end))
                self.assertAlmostEqual(
                    _slew_probe_command(phase, 1.10), end)
        self.assertEqual(len(paired), 24)
        self.assertTrue(all(modes == {"step", "ramp"}
                            for modes in paired.values()))
        self.assertEqual(set(YAW_ERROR_WHEELSPIN_POINTS), {
            (8.5, 0.06), (8.5, 0.10), (9.5, 0.06), (9.5, 0.10),
            (11.0, 0.06), (11.0, 0.10),
        })
        required_s = (
            sum(p.duration_s for p in phases)
            + sum(_is_yaw_transient_approach(p.label) for p in phases)
            * (SIM_RESET_HOLD_SEC + SIM_RESET_TIMEOUT_SEC)
            + sum(p.settle_before_probe for p in phases)
            * PROBE_START_TIMEOUT_SEC + 5.0)
        self.assertLess(required_s, 1200.0)

    def test_residual_grid_covers_every_audited_cell_with_transition_pairs(self) -> None:
        phases = build_schedule(202610081, YAW_ERROR_RESIDUAL_GRID_PROFILE)
        probes = [p for p in phases
                  if p.label.startswith("probe_yawerr_residual_")]
        self.assertEqual(len(probes), 160)
        self.assertEqual(len(phases), 3 * len(probes))
        expected = {
            (family, speed, angle, sign, delay, mode)
            for family in ("onset", "reversal")
            for speed, angle in YAW_ERROR_RESIDUAL_GRID_POINTS
            for sign in (-1.0, 1.0)
            for delay in YAW_ERROR_COMMAND_GAP_DELAYS_S
            for mode, _ in YAW_ERROR_COMMAND_GAP_MODES
        }
        observed = set()
        paired: dict[str, set[str]] = {}
        for index, phase in enumerate(phases):
            if not phase.label.startswith("probe_yawerr_residual_"):
                continue
            pair_id = phase.condition_pair_id
            self.assertIsNotNone(pair_id)
            family, speed_token, angle_token, turn_token, delay_token = (
                pair_id.split("_", 4))
            mode = "step" if "_step" in phase.label else "ramp"
            speed = float(speed_token[1:])
            angle = float(angle_token[1:])
            sign = float(turn_token[4:])
            delay = float(delay_token.removeprefix("delay"))
            observed.add((family, speed, angle, sign, delay, mode))
            paired.setdefault(pair_id, set()).add(mode)
            approach, settle = phases[index - 2:index]
            self.assertTrue(_is_yaw_transient_approach(approach.label))
            self.assertTrue(approach.reach_speed_target)
            self.assertEqual(approach.speed_target_mps, speed)
            self.assertEqual(phase.speed_target_mps, speed)
            self.assertEqual(settle.steering_rad,
                             phase.steering_waypoints[0][1])
            self.assertTrue(phase.validate_samples)
            self.assertFalse(phase.validate_speed)
            self.assertFalse(phase.validate_steering)
            start = sign * angle if family == "reversal" else 0.0
            target = -start if family == "reversal" else sign * angle
            self.assertAlmostEqual(phase.steering_waypoints[0][1], start)
            ramp_end = delay + (0.025 if mode == "step" else 0.30)
            self.assertAlmostEqual(_phase_steering_command(phase, ramp_end),
                                   target)
        self.assertEqual(observed, expected)
        self.assertEqual(len(paired), 80)
        self.assertTrue(all(modes == {"step", "ramp"}
                            for modes in paired.values()))
        required_s = (
            sum(p.duration_s for p in phases)
            + sum(_is_yaw_transient_approach(p.label) for p in phases)
            * (SIM_RESET_HOLD_SEC + SIM_RESET_TIMEOUT_SEC) + 5.0)
        self.assertLess(required_s, 3600.0)

    def test_lowspeed_highsteer_replication_adds_independent_transition_coverage(self) -> None:
        phases = build_schedule(
            202610084, YAW_ERROR_LOWSPEED_HIGHSTEER_PROFILE)
        probes = [p for p in phases
                  if p.label.startswith("probe_yawerr_lowhigh_")]
        self.assertEqual(len(probes), 144)
        self.assertEqual(len(phases), 3 * len(probes))
        expected = {
            (family, speed, angle, sign, delay, mode)
            for family in ("onset", "reversal")
            for speed, angle in YAW_ERROR_LOWSPEED_HIGHSTEER_POINTS
            for sign in (-1.0, 1.0)
            for delay in YAW_ERROR_COMMAND_GAP_DELAYS_S
            for mode, _ in YAW_ERROR_COMMAND_GAP_MODES
        }
        observed = set()
        paired: dict[str, set[str]] = {}
        for index, phase in enumerate(phases):
            if not phase.label.startswith("probe_yawerr_lowhigh_"):
                continue
            pair_id = phase.condition_pair_id
            self.assertIsNotNone(pair_id)
            family, speed_token, angle_token, turn_token, delay_token = (
                pair_id.split("_", 4))
            mode = "step" if "_step" in phase.label else "ramp"
            speed = float(speed_token[1:])
            angle = float(angle_token[1:])
            sign = float(turn_token[4:])
            delay = float(delay_token.removeprefix("delay"))
            observed.add((family, speed, angle, sign, delay, mode))
            paired.setdefault(pair_id, set()).add(mode)
            approach, settle = phases[index - 2:index]
            self.assertTrue(_is_yaw_transient_approach(approach.label))
            self.assertTrue(approach.reach_speed_target)
            self.assertEqual(approach.speed_target_mps, speed)
            self.assertEqual(phase.speed_target_mps, speed)
            self.assertEqual(settle.steering_rad,
                             phase.steering_waypoints[0][1])
            self.assertTrue(phase.validate_samples)
            self.assertFalse(phase.validate_speed)
            self.assertFalse(phase.validate_steering)
            start = sign * angle if family == "reversal" else 0.0
            target = -start if family == "reversal" else sign * angle
            self.assertAlmostEqual(phase.steering_waypoints[0][1], start)
            ramp_end = delay + (0.025 if mode == "step" else 0.30)
            self.assertAlmostEqual(_phase_steering_command(phase, ramp_end),
                                   target)
        self.assertEqual(observed, expected)
        self.assertEqual(len(paired), 72)
        self.assertTrue(all(modes == {"step", "ramp"}
                            for modes in paired.values()))
        required_s = (
            sum(p.duration_s for p in phases)
            + sum(_is_yaw_transient_approach(p.label) for p in phases)
            * (SIM_RESET_HOLD_SEC + SIM_RESET_TIMEOUT_SEC) + 5.0)
        self.assertLess(required_s, 3600.0)

    def test_highsteer_reversal_gapfill_covers_event_timing_and_both_directions(self) -> None:
        phases = build_schedule(
            202610088, YAW_ERROR_HIGHSTEER_REVERSAL_PROFILE)
        probes = [phase for phase in phases
                  if phase.label.startswith(
                      "probe_yawerr_highsteer_reversal_")]
        self.assertEqual(len(probes), 216)
        self.assertEqual(len(phases), 3 * len(probes))
        expected = {
            (family, speed, angle, sign, delay, mode, ramp_s)
            for family in ("onset", "unwind", "reversal")
            for speed, angle in YAW_ERROR_HIGHSTEER_REVERSAL_POINTS
            for sign in (-1.0, 1.0)
            for delay in (0.25, 0.75)
            for mode, ramp_s in YAW_ERROR_HIGHSTEER_REVERSAL_MODES
        }
        observed = set()
        for index, phase in enumerate(phases):
            if not phase.label.startswith(
                    "probe_yawerr_highsteer_reversal_"):
                continue
            approach, settle = phases[index - 2:index]
            self.assertTrue(_is_yaw_transient_approach(approach.label))
            self.assertTrue(approach.reach_speed_target)
            self.assertTrue(phase.validate_samples)
            self.assertTrue(phase.validate_speed)
            self.assertFalse(phase.validate_steering)
            self.assertEqual(phase.speed_target_mps,
                             approach.speed_target_mps)
            pair_id = phase.condition_pair_id
            self.assertIsNotNone(pair_id)
            family, speed_text, angle_text, turn_text, delay_text = (
                pair_id.split("_", 4))
            speed = float(speed_text[1:])
            angle = float(angle_text[1:])
            sign = float(turn_text[4:])
            delay = float(delay_text.removeprefix("delay"))
            self.assertIn((speed, angle),
                          YAW_ERROR_HIGHSTEER_REVERSAL_POINTS)
            start = sign * angle if family != "onset" else 0.0
            target = (0.0 if family == "unwind" else
                      -start if family == "reversal" else sign * angle)
            self.assertAlmostEqual(settle.steering_rad, start)
            self.assertAlmostEqual(phase.steering_waypoints[0][1], start)
            self.assertAlmostEqual(phase.steering_waypoints[-2][1], target)
            mode, ramp_s = next(
                (mode, ramp_s) for mode, ramp_s
                in YAW_ERROR_HIGHSTEER_REVERSAL_MODES
                if f"_{mode}{ramp_s:.3f}s" in phase.label)
            transition_end = delay + (0.025 if mode == "step" else ramp_s)
            self.assertAlmostEqual(
                _phase_steering_command(phase, transition_end), target)
            observed.add((family, speed, angle, sign, delay, mode, ramp_s))
        self.assertEqual(observed, expected)
        self.assertEqual(
            len({phase.condition_pair_id for phase in probes}), 72)
        required_s = (
            sum(phase.duration_s for phase in phases)
            + sum(_is_yaw_transient_approach(phase.label)
                  for phase in phases)
            * (SIM_RESET_HOLD_SEC + SIM_RESET_TIMEOUT_SEC) + 5.0)
        self.assertGreater(required_s, 3600.0)
        self.assertLess(required_s, 6000.0)

    def test_crawl_steering_gapfill_covers_sparse_low_speed_bins(self) -> None:
        phases = build_schedule(202610087, YAW_ERROR_CRAWL_STEERING_PROFILE)
        probes = [p for p in phases
                  if p.label.startswith("probe_yawerr_crawl_")]
        self.assertEqual(len(probes), 96)
        self.assertEqual(len(phases), 3 * len(probes))
        expected = {
            (family, speed, angle, sign, delay, mode)
            for family in ("onset", "reversal")
            for speed, angle in YAW_ERROR_CRAWL_STEERING_POINTS
            for sign in (-1.0, 1.0)
            for delay in YAW_ERROR_COMMAND_GAP_DELAYS_S
            for mode, _ in YAW_ERROR_COMMAND_GAP_MODES
        }
        observed = set()
        paired: dict[str, set[str]] = {}
        for index, phase in enumerate(phases):
            if not phase.label.startswith("probe_yawerr_crawl_"):
                continue
            pair_id = phase.condition_pair_id
            self.assertIsNotNone(pair_id)
            family, speed_token, angle_token, turn_token, delay_token = (
                pair_id.split("_", 4))
            mode = "step" if "_step" in phase.label else "ramp"
            speed = float(speed_token[1:])
            angle = float(angle_token[1:])
            sign = float(turn_token[4:])
            delay = float(delay_token.removeprefix("delay"))
            observed.add((family, speed, angle, sign, delay, mode))
            paired.setdefault(pair_id, set()).add(mode)
            approach, settle = phases[index - 2:index]
            self.assertTrue(_is_yaw_transient_approach(approach.label))
            self.assertTrue(approach.reach_speed_target)
            self.assertEqual(approach.speed_target_mps, speed)
            self.assertEqual(phase.speed_target_mps, speed)
            self.assertEqual(settle.steering_rad,
                             phase.steering_waypoints[0][1])
            self.assertTrue(phase.validate_samples)
            self.assertFalse(phase.validate_speed)
            self.assertFalse(phase.validate_steering)
            start = sign * angle if family == "reversal" else 0.0
            target = -start if family == "reversal" else sign * angle
            self.assertAlmostEqual(phase.steering_waypoints[0][1], start)
            ramp_end = delay + (0.025 if mode == "step" else 0.30)
            self.assertAlmostEqual(_phase_steering_command(phase, ramp_end),
                                   target)
        self.assertEqual(observed, expected)
        self.assertEqual(len(paired), 48)
        self.assertTrue(all(modes == {"step", "ramp"}
                            for modes in paired.values()))
        required_s = (
            sum(p.duration_s for p in phases)
            + sum(_is_yaw_transient_approach(p.label) for p in phases)
            * (SIM_RESET_HOLD_SEC + SIM_RESET_TIMEOUT_SEC) + 5.0)
        self.assertLess(required_s, 3600.0)

    def test_crawl_fine_gapfill_covers_sparse_low_speed_rectangle(self) -> None:
        phases = build_schedule(202610088, YAW_ERROR_CRAWL_FINE_PROFILE)
        probes = [p for p in phases
                  if p.label.startswith("probe_yawerr_crawl_fine_")]
        self.assertEqual(len(probes), 384)
        self.assertEqual(len(phases), 3 * len(probes))
        expected = {
            (family, speed, angle, sign, delay, mode)
            for family in ("onset", "reversal")
            for speed, angle in YAW_ERROR_CRAWL_FINE_POINTS
            for sign in (-1.0, 1.0)
            for delay in YAW_ERROR_COMMAND_GAP_DELAYS_S
            for mode, _ in YAW_ERROR_COMMAND_GAP_MODES
        }
        observed = set()
        paired: dict[str, set[str]] = {}
        for index, phase in enumerate(phases):
            if not phase.label.startswith("probe_yawerr_crawl_fine_"):
                continue
            pair_id = phase.condition_pair_id
            self.assertIsNotNone(pair_id)
            family, speed_token, angle_token, turn_token, delay_token = (
                pair_id.split("_", 4))
            mode = "step" if "_step" in phase.label else "ramp"
            speed = float(speed_token[1:])
            angle = float(angle_token[1:])
            sign = float(turn_token[4:])
            delay = float(delay_token.removeprefix("delay"))
            observed.add((family, speed, angle, sign, delay, mode))
            paired.setdefault(pair_id, set()).add(mode)
            approach, settle = phases[index - 2:index]
            self.assertTrue(_is_yaw_transient_approach(approach.label))
            self.assertTrue(approach.reach_speed_target)
            self.assertEqual(approach.speed_target_mps, speed)
            self.assertEqual(phase.speed_target_mps, speed)
            self.assertEqual(settle.steering_rad,
                             phase.steering_waypoints[0][1])
            self.assertTrue(phase.validate_samples)
            self.assertFalse(phase.validate_speed)
            self.assertFalse(phase.validate_steering)
        self.assertEqual(observed, expected)
        self.assertEqual(len(paired), 192)
        self.assertTrue(all(modes == {"step", "ramp"}
                            for modes in paired.values()))
        required_s = (
            sum(p.duration_s for p in phases)
            + sum(_is_yaw_transient_approach(p.label) for p in phases)
            * (SIM_RESET_HOLD_SEC + SIM_RESET_TIMEOUT_SEC) + 5.0)
        self.assertLess(required_s, 9000.0)

    def test_low_speed_wheelspin_gapfill_pairs_step_and_ramp_at_same_endpoint(self) -> None:
        for speed_mps, throttle_norm in YAW_LOW_SPEED_THROTTLE_CALIBRATION:
            self.assertAlmostEqual(
                _low_speed_wheelspin_feedforward(speed_mps), throttle_norm)
        self.assertAlmostEqual(
            _low_speed_wheelspin_feedforward(0.75), 0.03075, places=3)

        phases = build_schedule(
            202610089, YAW_ERROR_LOWSPEED_WHEELSPIN_PROFILE)
        probes = [phase for phase in phases
                  if phase.label.startswith("probe_yawerr_lowwheel_")]
        self.assertEqual(len(probes), 144)
        self.assertEqual(len(phases), 2 * len(probes))
        expected = {
            (speed, angle, sign, delta, mode)
            for speed, angle in YAW_ERROR_LOWSPEED_WHEELSPIN_POINTS
            for sign in (-1.0, 1.0)
            for delta in YAW_ERROR_LOWSPEED_WHEELSPIN_DELTAS
            for mode, _ in YAW_ERROR_WHEELSPIN_MODES
        }
        observed = set()
        paired: dict[str, set[str]] = {}
        for index, phase in enumerate(phases):
            if not phase.label.startswith("probe_yawerr_lowwheel_"):
                continue
            self.assertGreater(index, 0)
            approach = phases[index - 1]
            self.assertTrue(approach.label.startswith(
                "approach_yawerr_lowwheel_"))
            self.assertTrue(approach.reach_speed_target)
            self.assertEqual(approach.speed_target_mps,
                             phase.speed_target_mps)
            self.assertTrue(phase.settle_before_probe)
            self.assertTrue(phase.probe_race_domain)
            self.assertTrue(phase.validate_samples)
            self.assertFalse(phase.validate_speed)
            self.assertFalse(phase.validate_steering)
            pair_id = phase.condition_pair_id
            self.assertIsNotNone(pair_id)
            speed_token, angle_token, turn_token, delta_token = pair_id.split("_")
            speed = float(speed_token.removeprefix("v"))
            angle = float(angle_token.removeprefix("a"))
            sign = float(turn_token.removeprefix("turn"))
            delta = float(delta_token.removeprefix("up"))
            mode = phase.throttle_profile
            self.assertIn(mode, {"step", "ramp"})
            self.assertAlmostEqual(phase.speed_target_mps, speed)
            self.assertAlmostEqual(
                phase.throttle_start_norm,
                _low_speed_wheelspin_feedforward(speed))
            self.assertAlmostEqual(phase.steering_waypoints[2][1], sign * angle)
            self.assertAlmostEqual(
                phase.throttle_end_norm - phase.throttle_start_norm, delta)
            self.assertAlmostEqual(
                phase.throttle_ramp_duration_s,
                0.30 if mode == "ramp" else 0.0)
            observed.add((speed, angle, sign, delta, mode))
            paired.setdefault(pair_id, set()).add(mode)
        self.assertEqual(observed, expected)
        self.assertEqual(len(paired), 72)
        self.assertTrue(all(modes == {"step", "ramp"}
                            for modes in paired.values()))
        required_s = (
            sum(phase.duration_s for phase in phases)
            + sum(phase.label.startswith("approach_yawerr_lowwheel_")
                  for phase in phases)
            * (SIM_RESET_HOLD_SEC + SIM_RESET_TIMEOUT_SEC) + 5.0)
        self.assertLess(required_s, 3600.0)

    def test_subcrawl_steering_gapfill_uses_measured_speed_and_full_events(self) -> None:
        phases = build_schedule(
            202610088, YAW_ERROR_SUBCRAWL_STEERING_PROFILE)
        probes = [phase for phase in phases
                  if phase.label.startswith("probe_yawerr_subcrawl_")]
        self.assertEqual(len(probes), 144)
        self.assertEqual(len(phases), 3 * len(probes))
        self.assertIn(YAW_ERROR_SUBCRAWL_STEERING_PROFILE,
                      EXPERIMENT_PROFILE_CHOICES)
        self.assertIn(YAW_ERROR_SUBCRAWL_STEERING_PROFILE,
                      YAW_TRANSIENT_PROFILES)
        for speed, throttle in ((0.244, 0.01), (0.489, 0.02)):
            self.assertAlmostEqual(
                _low_speed_wheelspin_feedforward(speed), throttle)

        expected = {
            (family, speed, angle, sign, delay, mode)
            for family in ("onset", "unwind", "reversal")
            for speed, angle in YAW_ERROR_SUBCRAWL_STEERING_POINTS
            for sign in (-1.0, 1.0)
            for delay in (0.25, 0.75)
            for mode in ("step", "ramp")
        }
        observed = set()
        paired: dict[str, set[str]] = {}
        for index, phase in enumerate(phases):
            if not phase.label.startswith("probe_yawerr_subcrawl_"):
                continue
            pair_id = phase.condition_pair_id
            self.assertIsNotNone(pair_id)
            family, speed_token, angle_token, turn_token, delay_token = (
                pair_id.split("_", 4))
            mode = "step" if "_step" in phase.label else "ramp"
            speed = float(speed_token.removeprefix("v"))
            angle = float(angle_token.removeprefix("a"))
            sign = float(turn_token.removeprefix("turn"))
            delay = float(delay_token.removeprefix("delay"))
            observed.add((family, speed, angle, sign, delay, mode))
            paired.setdefault(pair_id, set()).add(mode)

            approach, settle = phases[index - 2:index]
            self.assertTrue(approach.label.startswith(
                "approach_yawerr_subcrawl_"))
            self.assertTrue(approach.reach_speed_target)
            self.assertAlmostEqual(approach.speed_target_mps, speed)
            self.assertAlmostEqual(phase.speed_target_mps, speed)
            self.assertEqual(settle.steering_rad,
                             phase.steering_waypoints[0][1])
            self.assertTrue(phase.validate_samples)
            self.assertTrue(phase.validate_speed)
            self.assertFalse(phase.validate_steering)
            self.assertEqual(phase.throttle_mode, "race_domain_hold")

        self.assertEqual(observed, expected)
        self.assertEqual(len(paired), 72)
        self.assertTrue(all(modes == {"step", "ramp"}
                            for modes in paired.values()))
        required_s = (
            sum(phase.duration_s for phase in phases)
            + sum(_is_yaw_transient_approach(phase.label) for phase in phases)
            * (SIM_RESET_HOLD_SEC + SIM_RESET_TIMEOUT_SEC) + 5.0)
        self.assertLess(required_s, 3600.0)

    def test_mid_speed_throttle_gapfill_pairs_both_polarities(self) -> None:
        phases = build_schedule(
            202610081, YAW_ERROR_MIDSPEED_THROTTLE_PROFILE)
        probes = [phase for phase in phases
                  if phase.label.startswith("probe_yawerr_midwheel_")]
        self.assertEqual(len(probes), 112)
        self.assertEqual(len(phases), 2 * len(probes))
        expected = {
            (speed, angle, sign, direction, delta, mode)
            for speed, angle in YAW_ERROR_MIDSPEED_THROTTLE_POINTS
            for sign in (-1.0, 1.0)
            for direction in YAW_ERROR_MIDSPEED_THROTTLE_DIRECTIONS
            for delta in YAW_ERROR_MIDSPEED_THROTTLE_DELTAS
            for mode, _ in YAW_ERROR_WHEELSPIN_MODES
        }
        observed = set()
        paired: dict[str, set[str]] = {}
        for index, phase in enumerate(phases):
            if not phase.label.startswith("probe_yawerr_midwheel_"):
                continue
            approach = phases[index - 1]
            self.assertTrue(approach.label.startswith(
                "approach_yawerr_midwheel_"))
            self.assertTrue(approach.reach_speed_target)
            self.assertEqual(approach.speed_target_mps,
                             phase.speed_target_mps)
            self.assertTrue(phase.settle_before_probe)
            self.assertTrue(phase.probe_race_domain)
            self.assertTrue(phase.validate_samples)
            self.assertFalse(phase.validate_speed)
            self.assertFalse(phase.validate_steering)
            pair_id = phase.condition_pair_id
            self.assertIsNotNone(pair_id)
            speed_token, angle_token, turn_token, direction, delta_token = (
                pair_id.split("_"))
            speed = float(speed_token.removeprefix("v"))
            angle = float(angle_token.removeprefix("a"))
            sign = float(turn_token.removeprefix("turn"))
            delta = float(delta_token.removeprefix("d"))
            signed_delta = delta if direction == "up" else -delta
            mode = phase.throttle_profile
            self.assertIn(mode, {"step", "ramp"})
            self.assertAlmostEqual(phase.speed_target_mps, speed)
            self.assertAlmostEqual(phase.steering_waypoints[2][1],
                                   sign * angle)
            self.assertAlmostEqual(
                phase.throttle_end_norm - phase.throttle_start_norm,
                signed_delta)
            self.assertGreaterEqual(phase.throttle_start_norm, 0.0)
            self.assertLessEqual(phase.throttle_start_norm, 0.5)
            self.assertGreaterEqual(phase.throttle_end_norm, 0.0)
            self.assertLessEqual(phase.throttle_end_norm, 0.5)
            self.assertAlmostEqual(
                phase.throttle_ramp_duration_s,
                0.30 if mode == "ramp" else 0.0)
            observed.add((speed, angle, sign, direction, delta, mode))
            paired.setdefault(pair_id, set()).add(mode)
        self.assertEqual(observed, expected)
        self.assertEqual(len(paired), 56)
        self.assertTrue(all(modes == {"step", "ramp"}
                            for modes in paired.values()))
        required_s = (
            sum(phase.duration_s for phase in phases)
            + sum(phase.label.startswith("approach_yawerr_midwheel_")
                  for phase in phases)
            * (SIM_RESET_HOLD_SEC + SIM_RESET_TIMEOUT_SEC) + 5.0)
        self.assertLess(required_s, 3600.0)

    def test_yaw_capture_schedules_are_seeded_and_shuffle_trials(self) -> None:
        for profile in (YAW_ERROR_STEERING_EVENT_PROFILE,
                        YAW_ERROR_COMMAND_GAP_PROFILE,
                        YAW_ERROR_WHEELSPIN_PROFILE,
                        YAW_ERROR_LOWSPEED_WHEELSPIN_PROFILE,
                        YAW_ERROR_MIDSPEED_THROTTLE_PROFILE,
                        YAW_ERROR_RESIDUAL_GRID_PROFILE,
                        YAW_ERROR_LOWSPEED_HIGHSTEER_PROFILE,
                        YAW_ERROR_CRAWL_STEERING_PROFILE,
                        YAW_ERROR_CRAWL_FINE_PROFILE):
            def condition_order(seed: int) -> list[str]:
                return [p.condition_pair_id for p in build_schedule(seed, profile)
                        if p.label.startswith("probe_yawerr_")]

            first = condition_order(202610074)
            self.assertEqual(first, condition_order(202610074))
            self.assertNotEqual(first, condition_order(202610075))


class YawFrontierThrottleSlewScheduleTest(unittest.TestCase):
    def test_frontier_capture_matches_the_held_out_9p5mps_matrix(self) -> None:
        phases = build_schedule(202610071, YAW_FRONTIER_THROTTLE_SLEW_PROFILE)
        approaches = [p for p in phases
                      if p.label.startswith("approach_swerve_pair_")]
        probes = [p for p in phases if p.label.startswith("swerve_slew_")]
        expected = {
            (speed, angle, turn)
            for speed, angles in YAW_FRONTIER_THROTTLE_SLEW_SPEED_STEERING_RAD
            for angle in angles
            for turn in (-1.0, 1.0)
        }
        observed = set()
        self.assertEqual(len(approaches), 6)
        self.assertEqual(len(probes), 12)
        self.assertEqual(len(phases), 18)
        for phase in probes:
            pair_id = phase.condition_pair_id
            pair = [p for p in probes if p.condition_pair_id == pair_id]
            self.assertEqual({p.throttle_profile for p in pair}, {"step", "ramp"})
            self.assertTrue(phase.settle_before_probe)
            self.assertTrue(phase.validate_samples)
            self.assertFalse(phase.validate_speed)
            self.assertFalse(phase.validate_steering)
            self.assertEqual(phase.duration_s, 1.85)
            self.assertEqual(phase.throttle_stimulus_delay_s, 0.60)
            self.assertEqual(phase.throttle_ramp_duration_s, 0.30)
            self.assertAlmostEqual(
                phase.throttle_start_norm - phase.throttle_end_norm, 0.08)
            if phase.throttle_profile == "step":
                self.assertEqual(_slew_probe_command(phase, 0.626),
                                 phase.throttle_end_norm)
            else:
                self.assertAlmostEqual(
                    _slew_probe_command(phase, 0.75),
                    0.5 * (phase.throttle_start_norm
                           + phase.throttle_end_norm))
            self.assertTrue(any(a.condition_pair_id == pair_id
                                and a.reach_speed_target
                                and a.speed_target_mps == 9.5
                                for a in approaches))
            turn = 1.0 if phase.steering_waypoints[1][1] > 0.0 else -1.0
            observed.add((phase.speed_target_mps,
                          phase.steering_amplitude_rad, turn))
            self.assertEqual(phase.steering_waypoints, (
                (0.00, 0.00), (0.30, turn * phase.steering_amplitude_rad),
                (0.60, turn * phase.steering_amplitude_rad), (0.95, 0.00),
                (1.25, -turn * phase.steering_amplitude_rad),
                (1.60, -turn * phase.steering_amplitude_rad), (1.85, 0.00),
            ))
        self.assertEqual(observed, expected)

    def test_frontier_profile_order_is_seeded_and_randomized(self) -> None:
        def signature(seed: int):
            return [(p.condition_pair_id, p.throttle_profile)
                    for p in build_schedule(
                        seed, YAW_FRONTIER_THROTTLE_SLEW_PROFILE)
                    if p.label.startswith("swerve_slew_")]

        self.assertEqual(signature(202610071), signature(202610071))
        self.assertNotEqual(signature(202610071), signature(202610072))


class YawFrontierLowangleUnwindScheduleTest(unittest.TestCase):
    def test_repeats_and_targets_the_missing_speed_steering_phase_cell(self) -> None:
        phases = build_schedule(202610081,
                                YAW_FRONTIER_LOWANGLE_UNWIND_PROFILE)
        approaches = [p for p in phases
                      if p.label.startswith("approach_swerve_pair_")]
        probes = [p for p in phases if p.label.startswith("swerve_slew_")]
        expected_pairs = {
            (speed, angle, turn, repeat)
            for speed, angles in YAW_FRONTIER_LOWANGLE_UNWIND_SPEED_STEERING_RAD
            for angle in angles
            for turn in (-1.0, 1.0)
            for repeat in range(1, YAW_FRONTIER_LOWANGLE_UNWIND_REPEATS + 1)
        }
        self.assertEqual(len(approaches), len(expected_pairs))
        self.assertEqual(len(probes), 2 * len(expected_pairs))
        self.assertEqual(len(phases), 3 * len(expected_pairs))
        self.assertEqual(len({p.condition_pair_id for p in approaches}),
                         len(expected_pairs))
        for phase in probes:
            pair_id = phase.condition_pair_id
            pair = [p for p in probes if p.condition_pair_id == pair_id]
            self.assertEqual({p.throttle_profile for p in pair}, {"step", "ramp"})
            self.assertTrue(any(a.condition_pair_id == pair_id
                                and a.reach_speed_target
                                and a.speed_target_mps == 9.5
                                for a in approaches))
            self.assertEqual(phase.duration_s, 1.85)
            self.assertEqual(phase.throttle_stimulus_delay_s, 0.60)
            self.assertEqual(phase.throttle_ramp_duration_s, 0.30)
            self.assertAlmostEqual(
                phase.throttle_start_norm - phase.throttle_end_norm, 0.08)
            turn = 1.0 if phase.steering_waypoints[1][1] > 0.0 else -1.0
            self.assertEqual(phase.steering_waypoints, (
                (0.00, 0.00),
                (0.30, turn * phase.steering_amplitude_rad),
                (0.60, turn * phase.steering_amplitude_rad),
                (0.70, turn * 0.030),
                (0.80, turn * 0.015),
                (0.90, 0.00),
                (1.15, 0.00),
                (1.40, -turn * phase.steering_amplitude_rad),
                (1.60, -turn * phase.steering_amplitude_rad),
                (1.85, 0.00),
            ))
            # The measured step enters 8.5–9.0 m/s near this part of the
            # maneuver; the low-angle segment now spans several 25 ms ticks.
            self.assertAlmostEqual(
                _phase_steering_command(phase, 0.725), turn * 0.02625)
            self.assertAlmostEqual(
                _phase_steering_command(phase, 0.750), turn * 0.02250)
            self.assertAlmostEqual(
                _phase_steering_command(phase, 0.775), turn * 0.01875)

    def test_repeated_lowangle_capture_order_is_seeded(self) -> None:
        def signature(seed: int):
            return [(p.condition_pair_id, p.throttle_profile)
                    for p in build_schedule(
                        seed, YAW_FRONTIER_LOWANGLE_UNWIND_PROFILE)
                    if p.label.startswith("swerve_slew_")]

        self.assertEqual(signature(202610081), signature(202610081))
        self.assertNotEqual(signature(202610081), signature(202610082))


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


class SwerveThrottleSlewScheduleTest(unittest.TestCase):
    def test_rate_factorial_crosses_delta_and_rate_independently(self):
        profiles = (
            (SWERVE_THROTTLE_RATE_FACTORIAL_4P5_PROFILE,
             SWERVE_THROTTLE_RATE_FACTORIAL_4P5_SPEED_MPS),
            (SWERVE_THROTTLE_RATE_FACTORIAL_PROFILE, 8.0),
            (SWERVE_THROTTLE_RATE_FACTORIAL_6P5_PROFILE,
             SWERVE_THROTTLE_RATE_FACTORIAL_6P5_SPEED_MPS),
            (SWERVE_THROTTLE_RATE_FACTORIAL_7P5_PROFILE,
             SWERVE_THROTTLE_RATE_FACTORIAL_7P5_SPEED_MPS),
        )
        for profile, expected_speed in profiles:
            with self.subTest(profile=profile):
                phases = build_schedule(20261010, profile)
                approaches = [phase for phase in phases
                              if phase.label.startswith("approach_swerve_pair_")]
                probes = [phase for phase in phases
                          if phase.label.startswith("swerve_slew_")]
                self.assertEqual(len(approaches), 48)
                self.assertEqual(len(probes), 48)
                self.assertEqual({phase.speed_target_mps for phase in probes},
                                 {expected_speed})
                groups = {}
                for phase in probes:
                    groups.setdefault(phase.condition_pair_id, []).append(phase)
                self.assertEqual(len(groups), 24)
                actual = set()
                for pair in groups.values():
                    self.assertEqual({phase.throttle_profile for phase in pair},
                                     {"ramp", "step"})
                    ramp = next(phase for phase in pair
                                if phase.throttle_profile == "ramp")
                    turn = 1.0 if ramp.steering_waypoints[1][1] > 0 else -1.0
                    measured_rate = (
                        ramp.throttle_end_norm - ramp.throttle_start_norm
                    ) / ramp.throttle_ramp_duration_s
                    actual.add((ramp.steering_amplitude_rad, turn,
                                round(ramp.throttle_end_norm
                                      - ramp.throttle_start_norm, 3),
                                round(measured_rate, 3)))
                expected = {
                    (angle, turn, delta, rate)
                    for angle in SWERVE_THROTTLE_RATE_SWEEP_STEERING_RAD
                    for turn in (-1.0, 1.0)
                    for delta in SWERVE_THROTTLE_RATE_FACTORIAL_DELTAS_NORM
                    for rate in (round(value, 3) for value in
                                 SWERVE_THROTTLE_RATE_FACTORIAL_RATES_NORM_PER_SEC)
                }
                self.assertEqual(actual, expected)
                self.assertEqual(SWERVE_EXPECTED_PAIRS_BY_PROFILE[profile], 24)
                self.assertEqual(SWERVE_EXPECTED_RESETS_BY_PROFILE[profile], 48)
                self.assertEqual(SWERVE_PROFILES[profile], "validation")

    def test_highsteer_rate_sweep_isolates_governor_clean_large_step_cells(self):
        phases = build_schedule(
            20261009, SWERVE_THROTTLE_RATE_SWEEP_HIGHSTEER_PROFILE,
            throttle_rate_sweep_delta_norm=0.16)
        approaches = [phase for phase in phases
                      if phase.label.startswith("approach_swerve_pair_")]
        probes = [phase for phase in phases
                  if phase.label.startswith("swerve_slew_")]
        self.assertEqual(len(approaches), 8)
        self.assertEqual(len(probes), 8)
        groups = {}
        for phase in probes:
            groups.setdefault(phase.condition_pair_id, []).append(phase)
        self.assertEqual(len(groups), 4)
        self.assertEqual(
            {(pair[0].steering_amplitude_rad,
              1.0 if pair[0].steering_waypoints[1][1] > 0 else -1.0,
              pair[0].throttle_ramp_duration_s)
             for pair in groups.values()},
            {(0.42, turn, duration)
             for turn in (-1.0, 1.0) for duration in (0.15, 0.30)})
        self.assertTrue(all({phase.throttle_profile for phase in pair}
                            == {"ramp", "step"}
                            for pair in groups.values()))
        self.assertEqual(SWERVE_EXPECTED_PAIRS_BY_PROFILE[
            SWERVE_THROTTLE_RATE_SWEEP_HIGHSTEER_PROFILE], 4)
        self.assertEqual(SWERVE_EXPECTED_RESETS_BY_PROFILE[
            SWERVE_THROTTLE_RATE_SWEEP_HIGHSTEER_PROFILE], 8)
        self.assertEqual(SWERVE_PROFILES[
            SWERVE_THROTTLE_RATE_SWEEP_HIGHSTEER_PROFILE], "validation")

    def test_highsteer_rate_sweep_pairs_three_ramps_with_reset_matched_steps(self):
        seed = 20261008
        phases = build_schedule(seed, SWERVE_THROTTLE_RATE_SWEEP_PROFILE)
        approaches = [phase for phase in phases
                      if phase.label.startswith("approach_swerve_pair_")]
        probes = [phase for phase in phases
                  if phase.label.startswith("swerve_slew_")]
        self.assertEqual(len(approaches), 24)
        self.assertEqual(len(probes), 24)

        expected = {
            (SWERVE_THROTTLE_RATE_SWEEP_SPEED_MPS, angle, turn, ramp_duration)
            for angle in SWERVE_THROTTLE_RATE_SWEEP_STEERING_RAD
            for turn in (-1.0, 1.0)
            for ramp_duration in SWERVE_THROTTLE_RATE_SWEEP_RAMP_DURATIONS_S
        }
        grouped = {}
        for phase in probes:
            grouped.setdefault(phase.condition_pair_id, []).append(phase)
        self.assertEqual(len(grouped), 12)
        actual = set()
        for pair_id, pair in grouped.items():
            self.assertEqual(len(pair), 2)
            self.assertEqual({phase.throttle_profile for phase in pair},
                             {"ramp", "step"})
            ramp = next(phase for phase in pair
                        if phase.throttle_profile == "ramp")
            step = next(phase for phase in pair
                        if phase.throttle_profile == "step")
            self.assertEqual(ramp.steering_waypoints, step.steering_waypoints)
            self.assertEqual(ramp.speed_target_mps,
                             SWERVE_THROTTLE_RATE_SWEEP_SPEED_MPS)
            self.assertTrue(all(phase.settle_before_probe for phase in pair))
            turn = 1.0 if ramp.steering_waypoints[1][1] > 0 else -1.0
            actual.add((ramp.speed_target_mps, ramp.steering_amplitude_rad,
                        turn, ramp.throttle_ramp_duration_s))
            for phase in pair:
                matching = [approach for approach in approaches
                            if approach.condition_pair_id == pair_id
                            and approach.label.endswith(
                                f"_{phase.throttle_profile}")]
                self.assertEqual(len(matching), 1)
                self.assertTrue(matching[0].reach_speed_target)
            self.assertAlmostEqual(
                _slew_probe_command(
                    ramp, ramp.throttle_stimulus_delay_s
                    + ramp.throttle_ramp_duration_s),
                ramp.throttle_end_norm,
            )
        self.assertEqual(actual, expected)
        self.assertEqual(SWERVE_EXPECTED_PAIRS_BY_PROFILE[
            SWERVE_THROTTLE_RATE_SWEEP_PROFILE], 12)
        self.assertEqual(SWERVE_EXPECTED_RESETS_BY_PROFILE[
            SWERVE_THROTTLE_RATE_SWEEP_PROFILE], 24)
        self.assertEqual(SWERVE_PROFILES[SWERVE_THROTTLE_RATE_SWEEP_PROFILE],
                         "validation")

        larger_delta = build_schedule(
            seed, SWERVE_THROTTLE_RATE_SWEEP_PROFILE,
            throttle_rate_sweep_delta_norm=0.16)
        larger_ramp = next(
            phase for phase in larger_delta
            if phase.label.startswith("swerve_slew_")
            and phase.throttle_profile == "ramp")
        self.assertAlmostEqual(
            larger_ramp.throttle_end_norm - larger_ramp.throttle_start_norm,
            0.16)
        self.assertLessEqual(larger_ramp.throttle_end_norm, 0.50)
        with self.assertRaises(ValueError):
            build_schedule(seed, SWERVE_THROTTLE_RATE_SWEEP_PROFILE,
                           throttle_rate_sweep_delta_norm=0.18)

    def test_low_steering_validation_covers_practice_line_cells(self):
        seed = 20261008
        profile = SWERVE_THROTTLE_SLEW_LOWSTEER_PROFILE
        phases = build_schedule(seed, profile)
        approaches = [phase for phase in phases
                      if phase.label.startswith("approach_swerve_pair_")]
        probes = [phase for phase in phases
                  if phase.label.startswith("swerve_slew_")]
        self.assertEqual(len(approaches), 48)
        self.assertEqual(len(probes), 48)
        expected = {
            (speed, angle, turn, throttle_direction)
            for speed, angles in SWERVE_THROTTLE_SLEW_LOWSTEER_SPEED_STEERING_RAD
            for angle in angles
            for turn in (-1.0, 1.0)
            for throttle_direction in (-1.0, 1.0)
        }
        grouped = {}
        for phase in probes:
            grouped.setdefault(phase.condition_pair_id, []).append(phase)
        actual = set()
        for pair_id, pair in grouped.items():
            self.assertEqual(len(pair), 2)
            self.assertEqual({phase.throttle_profile for phase in pair},
                             {"ramp", "step"})
            ramp = next(phase for phase in pair
                        if phase.throttle_profile == "ramp")
            step = next(phase for phase in pair
                        if phase.throttle_profile == "step")
            self.assertEqual(ramp.steering_waypoints, step.steering_waypoints)
            self.assertTrue(all(phase.settle_before_probe for phase in pair))
            for phase in pair:
                matching_approaches = [approach for approach in approaches
                                       if approach.condition_pair_id == pair_id
                                       and approach.label.endswith(
                                           f"_{phase.throttle_profile}")]
                self.assertEqual(len(matching_approaches), 1)
                self.assertTrue(matching_approaches[0].reach_speed_target)
            turn = 1.0 if ramp.steering_waypoints[1][1] > 0 else -1.0
            throttle_direction = (
                1.0 if ramp.throttle_end_norm > ramp.throttle_start_norm else -1.0)
            actual.add((ramp.speed_target_mps, ramp.steering_amplitude_rad,
                        turn, throttle_direction))
            self.assertTrue(any(phase.condition_pair_id == pair_id
                                for phase in approaches))
        self.assertEqual(actual, expected)
        self.assertEqual(SWERVE_EXPECTED_PAIRS_BY_PROFILE[profile], 24)
        self.assertEqual(SWERVE_EXPECTED_RESETS_BY_PROFILE[profile], 48)
        self.assertEqual(SWERVE_PROFILES[profile], "validation")
        required = (
            sum(phase.duration_s for phase in phases)
            + sum(phase.settle_before_probe for phase in phases)
            * PROBE_START_TIMEOUT_SEC
            + len(approaches) * (SIM_RESET_HOLD_SEC + SIM_RESET_TIMEOUT_SEC)
            + 5.0
        )
        self.assertLess(required, 1200.0)

    def test_moderate_steering_validation_fills_missing_paired_cells(self):
        seed = 20261007
        phases = build_schedule(seed, SWERVE_THROTTLE_SLEW_MODERATE_PROFILE)
        approaches = [phase for phase in phases
                      if phase.label.startswith("approach_swerve_pair_")]
        probes = [phase for phase in phases
                  if phase.label.startswith("swerve_slew_")]
        self.assertEqual(len(approaches), 24)
        self.assertEqual(len(probes), 48)
        self.assertEqual(
            {phase.speed_target_mps for phase in approaches},
            {speed for speed, _angles
             in SWERVE_THROTTLE_SLEW_MODERATE_SPEED_STEERING_RAD},
        )
        grouped = {}
        for phase in probes:
            grouped.setdefault(phase.condition_pair_id, []).append(phase)
        expected = {
            (speed, angle, turn, throttle_direction)
            for speed, angles in SWERVE_THROTTLE_SLEW_MODERATE_SPEED_STEERING_RAD
            for angle in angles
            for turn in (-1.0, 1.0)
            for throttle_direction in (-1.0, 1.0)
        }
        actual = set()
        for pair_id, pair in grouped.items():
            self.assertEqual(len(pair), 2)
            self.assertEqual({phase.throttle_profile for phase in pair},
                             {"ramp", "step"})
            ramp = next(phase for phase in pair
                        if phase.throttle_profile == "ramp")
            step = next(phase for phase in pair
                        if phase.throttle_profile == "step")
            self.assertEqual(ramp.steering_waypoints, step.steering_waypoints)
            self.assertTrue(all(phase.settle_before_probe for phase in pair))
            speed = ramp.speed_target_mps
            angle = ramp.steering_amplitude_rad
            turn = 1.0 if ramp.steering_waypoints[1][1] > 0 else -1.0
            throttle_direction = (
                1.0 if ramp.throttle_end_norm > ramp.throttle_start_norm else -1.0)
            actual.add((speed, angle, turn, throttle_direction))
            self.assertTrue(any(
                approach.condition_pair_id == pair_id
                for approach in approaches))
        self.assertEqual(actual, expected)
        required = (
            sum(phase.duration_s for phase in phases)
            + sum(phase.settle_before_probe for phase in phases)
            * PROBE_START_TIMEOUT_SEC
            + len(approaches) * (SIM_RESET_HOLD_SEC + SIM_RESET_TIMEOUT_SEC)
            + 5.0
        )
        self.assertLess(required, 1200.0)

    def test_train_and_validation_pair_the_full_measured_swerve_domain(self):
        seed = 20261006
        for profile in SWERVE_THROTTLE_SLEW_PROFILES:
            with self.subTest(profile=profile):
                phases = build_schedule(seed, profile)
                approaches = [phase for phase in phases
                              if phase.label.startswith("approach_swerve_pair_")]
                probes = [phase for phase in phases
                          if phase.label.startswith("swerve_slew_")]
                self.assertEqual(len(approaches), 32)
                self.assertEqual(len(probes), 64)
                self.assertEqual(
                    {phase.speed_target_mps for phase in approaches},
                    {speed for speed, _angles
                     in SWERVE_THROTTLE_SLEW_SPEED_STEERING_RAD},
                )
                grouped = {}
                for phase in probes:
                    grouped.setdefault(phase.condition_pair_id, []).append(phase)
                self.assertEqual(len(grouped), 32)
                expected = {
                    (speed, angle, turn, throttle_direction)
                    for speed, angles in SWERVE_THROTTLE_SLEW_SPEED_STEERING_RAD
                    for angle in angles
                    for turn in (-1.0, 1.0)
                    for throttle_direction in (
                        (-1.0,) if speed >= 9.5 else (-1.0, 1.0))
                }
                actual = set()
                for pair_id, pair in grouped.items():
                    self.assertEqual({phase.throttle_profile for phase in pair},
                                     {"ramp", "step"})
                    ramp = next(phase for phase in pair
                                if phase.throttle_profile == "ramp")
                    step = next(phase for phase in pair
                                if phase.throttle_profile == "step")
                    self.assertEqual(ramp.steering_waypoints,
                                     step.steering_waypoints)
                    self.assertTrue(all(phase.settle_before_probe
                                        for phase in pair))
                    self.assertTrue(all(phase.validate_samples
                                        and not phase.validate_speed
                                        and not phase.validate_steering
                                        for phase in pair))
                    self.assertEqual(ramp.duration_s, 1.85)
                    self.assertEqual(ramp.throttle_start_norm,
                                     step.throttle_start_norm)
                    self.assertEqual(ramp.throttle_end_norm,
                                     step.throttle_end_norm)
                    self.assertLessEqual(abs(
                        ramp.throttle_end_norm-ramp.throttle_start_norm),
                        0.08 + 1e-12)
                    self.assertAlmostEqual(
                        _slew_probe_command(ramp, 0.75),
                        (ramp.throttle_start_norm+ramp.throttle_end_norm)/2)
                    self.assertEqual(
                        _slew_probe_command(step, 0.75),
                        step.throttle_end_norm)
                    speed = ramp.speed_target_mps
                    angle = ramp.steering_amplitude_rad
                    turn = 1.0 if ramp.steering_waypoints[1][1] > 0 else -1.0
                    throttle_direction = (
                        1.0 if ramp.throttle_end_norm > ramp.throttle_start_norm
                        else -1.0)
                    actual.add((speed, angle, turn, throttle_direction))
                    self.assertTrue(pair_id in next(
                        approach.label for approach in approaches
                        if approach.condition_pair_id == pair_id))
                    for phase in pair:
                        for tick in range(int(phase.duration_s * 40) + 1):
                            self.assertLessEqual(
                                abs(_phase_steering_command(phase, tick/40.0)),
                                angle + 1e-9)
                self.assertEqual(actual, expected)

    def test_each_matched_condition_is_spawn_isolated_and_budgeted(self):
        phases = build_schedule(17061, SWERVE_THROTTLE_SLEW_PROFILES[0])
        approaches = [phase for phase in phases
                      if phase.label.startswith("approach_swerve_pair_")]
        self.assertEqual(len(approaches), 32)
        self.assertTrue(all(phase.reach_speed_target
                            and phase.throttle_mode == "race_domain_approach"
                            and phase.duration_s == SWERVE_THROTTLE_SLEW_APPROACH_S[
                                phase.speed_target_mps]
                            for phase in approaches))
        required = (
            sum(phase.duration_s for phase in phases)
            + sum(phase.settle_before_probe for phase in phases)
            * PROBE_START_TIMEOUT_SEC
            + len(approaches) * (SIM_RESET_HOLD_SEC + SIM_RESET_TIMEOUT_SEC)
            + 5.0
        )
        self.assertLess(required, 1200.0)
        self.assertGreater(required, 750.0)

    def test_seed_randomizes_pair_order_and_step_ramp_order_reproducibly(self):
        profile = SWERVE_THROTTLE_SLEW_PROFILES[0]
        first = build_schedule(17071, profile)
        repeat = build_schedule(17071, profile)
        other = build_schedule(17072, profile)
        self.assertEqual(first, repeat)
        self.assertNotEqual(first, other)
        first_shapes = [phase.throttle_profile for phase in first
                        if phase.label.startswith("swerve_slew_")]
        other_shapes = [phase.throttle_profile for phase in other
                        if phase.label.startswith("swerve_slew_")]
        self.assertNotEqual(first_shapes, other_shapes)

    def test_high_speed_frontier_extension_covers_each_steer_turn_pair(self):
        seed = 202610107
        phases = build_schedule(seed, SWERVE_THROTTLE_SLEW_FRONTIER_PROFILE)
        approaches = [phase for phase in phases
                      if phase.label.startswith("approach_swerve_pair_")]
        probes = [phase for phase in phases
                  if phase.label.startswith("swerve_slew_")]
        self.assertEqual(len(approaches), 18)
        self.assertEqual(len(probes), 36)
        self.assertEqual(
            {phase.speed_target_mps for phase in approaches},
            {speed for speed, _angles
             in SWERVE_THROTTLE_SLEW_FRONTIER_SPEED_STEERING_RAD},
        )
        grouped = {}
        for phase in probes:
            grouped.setdefault(phase.condition_pair_id, []).append(phase)
        expected = {
            (speed, angle, turn)
            for speed, angles in SWERVE_THROTTLE_SLEW_FRONTIER_SPEED_STEERING_RAD
            for angle in angles
            for turn in (-1.0, 1.0)
        }
        actual = set()
        for pair_id, pair in grouped.items():
            self.assertEqual({phase.throttle_profile for phase in pair},
                             {"ramp", "step"})
            self.assertTrue(all(phase.settle_before_probe
                                and phase.validate_samples
                                and not phase.validate_speed
                                for phase in pair))
            self.assertTrue(pair_id in next(
                approach.label for approach in approaches
                if approach.condition_pair_id == pair_id))
            ramp = next(phase for phase in pair
                        if phase.throttle_profile == "ramp")
            turn = (1.0 if ramp.steering_waypoints[1][1] > 0 else -1.0)
            actual.add((ramp.speed_target_mps,
                        ramp.steering_amplitude_rad, turn))
            self.assertLess(ramp.throttle_end_norm,
                            ramp.throttle_start_norm)
        self.assertEqual(actual, expected)
        required = (
            sum(phase.duration_s for phase in phases)
            + sum(phase.settle_before_probe for phase in phases)
            * PROBE_START_TIMEOUT_SEC
            + len(approaches) * (SIM_RESET_HOLD_SEC + SIM_RESET_TIMEOUT_SEC)
            + 5.0
        )
        self.assertLess(required, 1200.0)
        self.assertEqual(
            SWERVE_EXPECTED_PAIRS_BY_PROFILE[
                SWERVE_THROTTLE_SLEW_FRONTIER_PROFILE], 18)
        self.assertEqual(SWERVE_PROFILES[
            SWERVE_THROTTLE_SLEW_FRONTIER_PROFILE], "validation")

    def test_frontier_extension_order_is_seeded_and_has_random_ramp_step_order(self):
        profile = SWERVE_THROTTLE_SLEW_FRONTIER_PROFILE
        first = build_schedule(202610107, profile)
        repeat = build_schedule(202610107, profile)
        other = build_schedule(202610108, profile)
        self.assertEqual(first, repeat)
        self.assertNotEqual(first, other)
        first_shapes = [phase.throttle_profile for phase in first
                        if phase.label.startswith("swerve_slew_")]
        other_shapes = [phase.throttle_profile for phase in other
                        if phase.label.startswith("swerve_slew_")]
        self.assertNotEqual(first_shapes, other_shapes)

    def test_11mps_replication_is_limited_to_ambiguous_frontier_cells(self):
        profile = SWERVE_THROTTLE_SLEW_11MPS_REPLICATION_PROFILE
        phases = build_schedule(202610207, profile)
        approaches = [phase for phase in phases
                      if phase.label.startswith("approach_swerve_pair_")]
        probes = [phase for phase in phases
                  if phase.label.startswith("swerve_slew_")]
        expected = {
            (11.1, angle, turn)
            for _speed, angles
            in SWERVE_THROTTLE_SLEW_11MPS_REPLICATION_SPEED_STEERING_RAD
            for angle in angles
            for turn in (-1.0, 1.0)
        }
        actual = set()
        self.assertEqual(len(approaches), 6)
        self.assertEqual(len(probes), 12)
        for pair_id in {phase.condition_pair_id for phase in probes}:
            pair = [phase for phase in probes
                    if phase.condition_pair_id == pair_id]
            self.assertEqual({phase.throttle_profile for phase in pair},
                             {"ramp", "step"})
            ramp = next(phase for phase in pair
                        if phase.throttle_profile == "ramp")
            self.assertEqual(ramp.speed_target_mps, 11.1)
            self.assertLess(ramp.throttle_end_norm,
                            ramp.throttle_start_norm)
            turn = (1.0 if ramp.steering_waypoints[1][1] > 0 else -1.0)
            actual.add((ramp.speed_target_mps,
                        ramp.steering_amplitude_rad, turn))
        self.assertEqual(actual, expected)
        self.assertEqual(SWERVE_EXPECTED_PAIRS_BY_PROFILE[profile], 6)
        self.assertEqual(SWERVE_PROFILES[profile], "validation")

    def test_11mps_replication_order_is_seeded_and_randomized(self):
        profile = SWERVE_THROTTLE_SLEW_11MPS_REPLICATION_PROFILE
        first = build_schedule(202610207, profile)
        self.assertEqual(first, build_schedule(202610207, profile))
        self.assertNotEqual(first, build_schedule(202610208, profile))


class FrontierThrottleUpSwerveScheduleTest(unittest.TestCase):
    def test_positive_throttle_frontier_covers_only_the_missing_cells(self):
        for profile in (SWERVE_THROTTLE_RATE_FRONTIER_UP_TRAIN_PROFILE,
                        SWERVE_THROTTLE_RATE_FRONTIER_UP_VALIDATION_PROFILE):
            with self.subTest(profile=profile):
                self._assert_profile_schedule(profile)

    def _assert_profile_schedule(self, profile):
        phases = build_schedule(20261007, profile)
        approaches = [phase for phase in phases
                      if phase.label.startswith("approach_swerve_pair_")]
        probes = [phase for phase in phases
                  if phase.label.startswith("swerve_slew_")]
        self.assertEqual(len(approaches), 32)
        self.assertEqual(len(probes), 32)

        grouped = {}
        for phase in probes:
            grouped.setdefault(phase.condition_pair_id, []).append(phase)
        self.assertEqual(len(grouped), 16)
        expected = {
            (speed, angle, turn, duration)
            for speed, angles in SWERVE_THROTTLE_RATE_FRONTIER_UP_SPEED_STEERING_RAD
            for angle in angles
            for turn in (-1.0, 1.0)
            for duration in SWERVE_THROTTLE_RATE_FRONTIER_UP_RAMP_DURATIONS_S
        }
        actual = set()
        for pair_id, pair in grouped.items():
            self.assertEqual(len(pair), 2)
            self.assertEqual({phase.throttle_profile for phase in pair},
                             {"ramp", "step"})
            ramp = next(phase for phase in pair
                        if phase.throttle_profile == "ramp")
            step = next(phase for phase in pair
                        if phase.throttle_profile == "step")
            self.assertEqual(ramp.speed_target_mps, step.speed_target_mps)
            self.assertEqual(ramp.steering_waypoints, step.steering_waypoints)
            self.assertEqual(ramp.throttle_start_norm,
                             step.throttle_start_norm)
            self.assertEqual(ramp.throttle_end_norm, step.throttle_end_norm)
            self.assertAlmostEqual(
                ramp.throttle_end_norm - ramp.throttle_start_norm,
                SWERVE_THROTTLE_RATE_FRONTIER_UP_DELTA_NORM)
            self.assertTrue(ramp.settle_before_probe)
            turn = 1.0 if ramp.steering_waypoints[1][1] > 0 else -1.0
            actual.add((ramp.speed_target_mps,
                        ramp.steering_amplitude_rad, turn,
                        ramp.throttle_ramp_duration_s))
            for member in pair:
                matched_approach = [phase for phase in approaches
                                    if phase.condition_pair_id == pair_id
                                    and phase.label.endswith(
                                        f"_{member.throttle_profile}")]
                self.assertEqual(len(matched_approach), 1)
                self.assertTrue(matched_approach[0].reach_speed_target)
        self.assertEqual(actual, expected)
        self.assertEqual(SWERVE_EXPECTED_PAIRS_BY_PROFILE[profile], 16)
        self.assertEqual(SWERVE_EXPECTED_RESETS_BY_PROFILE[profile], 32)
        self.assertEqual(SWERVE_PROFILES[profile],
                         "train" if profile.endswith("_train")
                         else "validation")

    def test_seed_randomizes_condition_and_ramp_step_order(self):
        profile = SWERVE_THROTTLE_RATE_FRONTIER_UP_PROFILE
        first = build_schedule(20261007, profile)
        self.assertEqual(first, build_schedule(20261007, profile))
        self.assertNotEqual(first, build_schedule(20261008, profile))
        first_shapes = [phase.throttle_profile for phase in first
                        if phase.label.startswith("swerve_slew_")]
        other_shapes = [phase.throttle_profile for phase in build_schedule(
            20261008, profile) if phase.label.startswith("swerve_slew_")]
        self.assertNotEqual(first_shapes, other_shapes)


class RaceDomainThrottleRateScheduleTest(unittest.TestCase):
    def test_race_domain_factorial_is_complete_matched_and_within_timeout(self):
        for profile in (SWERVE_THROTTLE_RATE_RACE_DOMAIN_TRAIN_PROFILE,
                        SWERVE_THROTTLE_RATE_RACE_DOMAIN_VALIDATION_PROFILE):
            for speed in SWERVE_THROTTLE_RATE_RACE_DOMAIN_SPEEDS_MPS:
                with self.subTest(profile=profile, speed=speed):
                    phases = build_schedule(20261008, profile, speed)
                    approaches = [phase for phase in phases
                                  if phase.label.startswith(
                                      "approach_swerve_pair_")]
                    probes = [phase for phase in phases
                              if phase.label.startswith("swerve_slew_")]
                    self.assertEqual(len(approaches), 48)
                    self.assertEqual(len(probes), 48)
                    grouped = {}
                    for phase in probes:
                        grouped.setdefault(phase.condition_pair_id, []).append(phase)
                    expected = {
                        (angle, turn, delta, duration)
                        for angle in SWERVE_THROTTLE_RATE_RACE_DOMAIN_STEERING_RAD
                        for turn in (-1.0, 1.0)
                        for delta in SWERVE_THROTTLE_RATE_RACE_DOMAIN_DELTAS_NORM
                        for duration in SWERVE_THROTTLE_RATE_RACE_DOMAIN_RAMP_DURATIONS_S
                    }
                    actual = set()
                    for pair_id, pair in grouped.items():
                        self.assertEqual(len(pair), 2)
                        self.assertEqual({phase.throttle_profile for phase in pair},
                                         {"ramp", "step"})
                        ramp = next(phase for phase in pair
                                    if phase.throttle_profile == "ramp")
                        step = next(phase for phase in pair
                                    if phase.throttle_profile == "step")
                        self.assertEqual(ramp.speed_target_mps, speed)
                        self.assertEqual(ramp.steering_waypoints,
                                         step.steering_waypoints)
                        self.assertEqual(ramp.throttle_start_norm,
                                         step.throttle_start_norm)
                        self.assertEqual(ramp.throttle_end_norm,
                                         step.throttle_end_norm)
                        self.assertTrue(ramp.settle_before_probe)
                        self.assertTrue(step.settle_before_probe)
                        turn = (1.0 if ramp.steering_waypoints[1][1] > 0
                                else -1.0)
                        delta = ramp.throttle_end_norm - ramp.throttle_start_norm
                        actual.add((ramp.steering_amplitude_rad, turn,
                                    round(delta, 3),
                                    ramp.throttle_ramp_duration_s))
                        for member in pair:
                            matching = [phase for phase in approaches
                                        if phase.condition_pair_id == pair_id
                                        and phase.label.endswith(
                                            f"_{member.throttle_profile}")]
                            self.assertEqual(len(matching), 1)
                            self.assertTrue(matching[0].reach_speed_target)
                    self.assertEqual(actual, expected)
                    self.assertEqual(len(grouped), 24)
                    self.assertEqual(
                        SWERVE_EXPECTED_PAIRS_BY_PROFILE[profile], 24)
                    self.assertEqual(
                        SWERVE_EXPECTED_RESETS_BY_PROFILE[profile], 48)
                    self.assertEqual(
                        SWERVE_PROFILES[profile],
                        "train" if profile.endswith("_train") else "validation")
                    required = (
                        sum(phase.duration_s for phase in phases)
                        + sum(phase.settle_before_probe for phase in phases)
                        * PROBE_START_TIMEOUT_SEC
                        + len(approaches)
                        * (SIM_RESET_HOLD_SEC + SIM_RESET_TIMEOUT_SEC)
                        + 5.0)
                    self.assertLess(required, 1200.0)

    def test_seed_randomizes_order_and_invalid_speed_is_rejected(self):
        profile = SWERVE_THROTTLE_RATE_RACE_DOMAIN_TRAIN_PROFILE
        first = build_schedule(20261008, profile, 6.5)
        self.assertEqual(first, build_schedule(20261008, profile, 6.5))
        self.assertNotEqual(first, build_schedule(20261009, profile, 6.5))
        with self.assertRaisesRegex(ValueError, "speed must be one of"):
            build_schedule(20261008, profile, 5.5)


class SwerveThrottleSlewPairAnalysisTest(unittest.TestCase):
    def test_whole_response_score_is_a_time_window_mean(self):
        samples = [
            (0.10, {"residual": 0.10}),
            (0.35, {"residual": 0.20}),
            (0.70, {"residual": 0.40}),
            (1.00, {"residual": 9.00}),
        ]
        mean, count = _mean_window(samples, (0.10, 1.00))
        self.assertEqual(count, 3)
        self.assertAlmostEqual(mean["residual"], (0.10 + 0.20 + 0.40) / 3)

    @staticmethod
    def _phase(profile: str, steering_commands: tuple[float, float, float]):
        windows = ("0.10_0.35s", "0.35_0.65s", "0.65_1.00s")
        return {
            "profile": profile,
            "phase_valid": True,
            "command_profile_pass": True,
            "feedback_profile_pass": True,
            "fixed_packet_timebase_pass": True,
            "steering_profile": "waypoints",
            "steering_waypoints": [[0.0, 0.0], [0.3, 0.42], [0.6, 0.42],
                                   [0.95, 0.0], [1.25, -0.42], [1.6, -0.42],
                                   [1.85, 0.0]],
            "steering_command_rad": 0.0,
            "throttle_start_norm": 0.25,
            "throttle_end_norm": 0.33 if profile == "ramp" else 0.33,
            "stimulus_state": {
                "speed_mps": 6.5,
                "steering_feedback_rad": 0.31,
                "steering_command_rad": 0.42,
                "vy_mps": 0.2,
                "yaw_rate_rps": 0.5,
                "throttle_feedback_norm": 0.25,
            },
            "pre_window": {"median": {
                "common_rear_wheel_residual_mps": 0.0,
                "rear_wheel_residual_asymmetry_mps": 0.0,
            }},
            "post_windows": {
                window: {"median": {"steering_command_rad": command}}
                for window, command in zip(windows, steering_commands)
            },
        }

    def test_matched_swerve_accepts_expected_nonzero_steering_at_stimulus(self):
        ramp = self._phase("ramp", (0.42, 0.21, -0.42))
        step = self._phase("step", (0.42, 0.21, -0.42))
        matched, _differences, _pre_differences, failures = _pair_match(ramp, step)
        self.assertTrue(matched, failures)

    def test_pair_rejects_a_different_steering_command_trace(self):
        ramp = self._phase("ramp", (0.42, 0.21, -0.42))
        step = self._phase("step", (0.30, 0.21, -0.42))
        matched, _differences, _pre_differences, failures = _pair_match(ramp, step)
        self.assertFalse(matched)
        self.assertIn("steering_command_waveform_mismatch_0.10_0.35s", failures)

    def test_pair_rejects_unmatched_initial_rear_wheel_speed_residual(self):
        ramp = self._phase("ramp", (0.42, 0.21, -0.42))
        step = self._phase("step", (0.42, 0.21, -0.42))
        step["pre_window"]["median"]["common_rear_wheel_residual_mps"] = 0.90

        matched, _differences, pre_differences, failures = _pair_match(ramp, step)

        self.assertFalse(matched)
        self.assertAlmostEqual(pre_differences["left_rear_wheel_residual_mps"], 0.90)
        self.assertAlmostEqual(pre_differences["right_rear_wheel_residual_mps"], 0.90)
        self.assertIn("pre_left_rear_wheel_residual_mps_mismatch", failures)
        self.assertIn("pre_right_rear_wheel_residual_mps_mismatch", failures)

    def test_response_surface_uses_runs_as_uncertainty_units(self):
        keys = {
            "condition_pair_id": "cell-1",
            "capture_profile": "factorial",
            "speed_target_mps": 8.0,
            "throttle_delta_direction": "up",
            "throttle_delta_norm": 0.08,
            "throttle_rise_rate_norm_per_sec": 0.267,
            "abs_steering_command_rad": 0.42,
            "turn_direction": "left",
        }
        residual_metric = (
            "step_minus_ramp_mean_abs_rear_wheel_residual_mps_change_0.10_1.00s")
        proxy_metric = (
            "step_minus_ramp_mean_abs_rear_wheel_longitudinal_slip_ratio_proxy_change_0.10_1.00s")
        acceleration_metric = (
            "step_minus_ramp_rigid_body_longitudinal_acceleration_mps2_change_0.10_1.00s")
        pairs = []
        for run_id, residual in (("r1", 0.1), ("r1", 0.3), ("r2", 0.5)):
            pairs.append({
                **keys, "run_id": run_id, "valid": True,
                residual_metric: residual, proxy_metric: residual / 10,
                acceleration_metric: -residual,
            })
        pairs.append({**keys, "run_id": "bad", "valid": False})

        [row] = _response_surface_rows(pairs)

        self.assertEqual(row["valid_pair_count"], 3)
        self.assertEqual(row["run_count"], 2)
        self.assertAlmostEqual(
            row["wheel_residual_step_minus_ramp_mps_mean"], 0.35)
        self.assertAlmostEqual(
            row["wheel_residual_step_minus_ramp_mps_run_min"], 0.2)
        self.assertAlmostEqual(
            row["wheel_residual_step_minus_ramp_mps_run_max"], 0.5)

    def test_imu_roll_and_roll_rate_are_extracted_as_features(self):
        from types import SimpleNamespace

        sample = SimpleNamespace(
            state=(6.5, 0.1, 0.2),
            rear_wheel_surface_mps=(6.5, 6.5),
            actuators=(0.3, 0.3, 0.3, 0.6),
            imu_roll_pitch_rad=(0.04, -0.02),
            imu_roll_pitch_rate_rps=(0.30, -0.01),
            imu_acceleration_mps2=None,
        )

        values = _sample_values(sample, (6.0, 6.0))

        self.assertAlmostEqual(values["imu_roll_rad"], 0.04)
        self.assertAlmostEqual(values["abs_imu_roll_rad"], 0.04)
        self.assertAlmostEqual(values["imu_roll_rate_rps"], 0.30)
        self.assertAlmostEqual(values["abs_imu_roll_rate_rps"], 0.30)

    def test_rigid_acceleration_uses_rotating_body_frame_terms(self):
        from types import SimpleNamespace

        com_x = 0.15532
        samples = []
        for sequence in range(3):
            time_s = 0.025 * sequence
            u = 1.0 + 2.0 * time_s
            v_com = 0.2 + 3.0 * time_s
            yaw_rate = 0.5 + time_s
            v_rear = v_com - com_x * yaw_rate
            sample = SimpleNamespace(state=(u, v_rear, yaw_rate))
            samples.append((sequence, sample))
        acceleration = _rigid_body_acceleration_by_sequence(samples)[1]
        self.assertAlmostEqual(
            acceleration["rigid_body_longitudinal_acceleration_mps2"],
            2.0 - 0.525 * 0.275)
        self.assertAlmostEqual(
            acceleration["rigid_body_lateral_acceleration_mps2"],
            3.0 + 0.525 * 1.05)
        self.assertAlmostEqual(
            acceleration["rigid_body_yaw_acceleration_rps2"], 1.0)

    def test_encoder_speed_reference_averages_matching_four_packet_window(self):
        from types import SimpleNamespace

        samples = [
            (sequence, SimpleNamespace(
                state=(1.0 + 4.0 * (0.025 * sequence), 0.0, 0.0)))
            for sequence in range(5)
        ]
        averages = _average_rear_contact_speeds_by_sequence(samples)
        self.assertEqual(sorted(averages), [4])
        self.assertAlmostEqual(averages[4][0], 1.2)
        self.assertAlmostEqual(averages[4][1], 1.2)

    def test_encoder_speed_reference_does_not_bridge_packet_gaps(self):
        from types import SimpleNamespace

        samples = [
            (sequence, SimpleNamespace(state=(2.0, 0.0, 0.0)))
            for sequence in (0, 1, 3, 4)
        ]
        self.assertEqual(
            _average_rear_contact_speeds_by_sequence(samples), {})


if __name__ == "__main__":
    unittest.main()
