from __future__ import annotations

import unittest

from tools.race_domain_experiment_plan import (
    RACE_DOMAIN_GOVERNOR_MPS,
    BLOCK_DURATION_S,
    RACE_DOMAIN_HARD_LIMIT_MPS,
    RACE_DOMAIN_SPEEDS_MPS,
    build_race_domain_plan,
    build_race_domain_boundary_plan,
    build_race_domain_moderate_braking_plan,
    plan_duration_s,
    race_domain_feedforward,
    race_domain_moderate_steering_limit,
    steering_limit_for_speed,
)


class RaceDomainExperimentPlanTest(unittest.TestCase):
    def test_plan_is_seeded_full_spectrum_and_inside_guard_envelope(self):
        first = build_race_domain_plan(17001)
        repeat = build_race_domain_plan(17001)
        other = build_race_domain_plan(17002)
        self.assertEqual(first, repeat)
        self.assertNotEqual(first, other)
        self.assertEqual(len(first), 65)
        self.assertAlmostEqual(plan_duration_s(first), 260.0)
        self.assertTrue(all(block.duration_s == BLOCK_DURATION_S for block in first))
        self.assertLess(max(block.target_speed_mps for block in first),
                        RACE_DOMAIN_HARD_LIMIT_MPS)

        for speed in RACE_DOMAIN_SPEEDS_MPS:
            grid = [block for block in first
                    if block.label.startswith(f"grid_{speed:g}_")]
            self.assertEqual(len(grid), 7)
            self.assertLessEqual(
                max(abs(block.steering_rad) for block in grid),
                steering_limit_for_speed(speed))
            self.assertTrue(any(block.steering_rad < 0.0 for block in grid))
            self.assertTrue(any(block.steering_rad > 0.0 for block in grid))

        self.assertEqual(sum(block.label.startswith("corner_exit_")
                             for block in first), 4)
        self.assertEqual(sum(block.label.startswith("brake_transition_")
                             for block in first), 5)

    def test_steering_envelope_tightens_with_speed(self):
        limits = [steering_limit_for_speed(speed)
                  for speed in (2.0, 4.0, 6.0, 8.0, 10.0)]
        self.assertEqual(limits, sorted(limits, reverse=True))

    def test_boundary_revision_increases_coverage_without_crossing_governor(self):
        plan = build_race_domain_plan(17004, boundary_speed_mps=11.1)
        high_grid = [block for block in plan
                     if block.label.startswith("grid_11.1_")]
        self.assertEqual(len(high_grid), 7)
        self.assertTrue(all(block.target_speed_mps == 11.1
                            for block in high_grid))
        self.assertLess(max(block.target_speed_mps for block in plan),
                        RACE_DOMAIN_GOVERNOR_MPS)

    def test_boundary_feedforward_uses_measured_throttle_surface_anchors(self):
        self.assertAlmostEqual(race_domain_feedforward(8.0, 0.328), 0.328)
        self.assertAlmostEqual(race_domain_feedforward(9.66, 0.34), 0.40)
        self.assertAlmostEqual(race_domain_feedforward(10.82, 0.34), 0.45)
        self.assertLessEqual(race_domain_feedforward(11.9, 0.34), 0.5)

    def test_combined_boundary_plan_has_paired_braking_and_both_turn_directions(self):
        first = build_race_domain_boundary_plan(17020)
        self.assertEqual(first, build_race_domain_boundary_plan(17020))
        self.assertNotEqual(first, build_race_domain_boundary_plan(17021))
        self.assertEqual(len(first), 51)
        self.assertEqual(plan_duration_s(first), 204.0)
        self.assertTrue(all(block.target_speed_mps <= 11.1 for block in first))
        self.assertLess(max(block.target_speed_mps for block in first),
                        RACE_DOMAIN_HARD_LIMIT_MPS)
        for speed, maximum in ((9.5, 0.09), (10.5, 0.09), (11.1, 0.05)):
            rows = [block for block in first
                    if block.label.startswith(f"boundary_grid_{speed:g}_")]
            self.assertTrue(rows)
            self.assertLessEqual(max(abs(block.steering_rad) for block in rows),
                                 maximum)
            self.assertTrue(any(block.steering_rad < 0.0 for block in rows))
            self.assertTrue(any(block.steering_rad > 0.0 for block in rows))
            amplitudes = [abs(block.steering_rad) for block in rows]
            self.assertEqual(amplitudes, sorted(amplitudes))
        self.assertEqual(sum(block.label.startswith("brake_approach_")
                             for block in first), 8)
        self.assertEqual(sum(block.label.startswith("brake_pulse_")
                             for block in first), 8)
        approaches = [index for index, block in enumerate(first)
                      if block.label.startswith("brake_approach_")]
        for index in approaches:
            approach, pulse = first[index:index + 2]
            self.assertTrue(pulse.label.startswith("brake_pulse_"))
            self.assertEqual(approach.steering_rad, pulse.steering_rad)
            self.assertGreater(approach.target_speed_mps,
                               pulse.target_speed_mps)
        self.assertTrue(any(block.label == "brake_test_reposition_low"
                            and block.target_speed_mps == 3.0 for block in first))
        self.assertTrue(any(block.label == "corner_exit_reposition_low"
                            and block.target_speed_mps == 3.0 for block in first))
        self.assertTrue(any(block.label.startswith("corner_exit_L_")
                            for block in first))
        self.assertTrue(any(block.label.startswith("corner_exit_R_")
                            for block in first))

    def test_moderate_braking_plan_steps_through_turn_in_cut_brake_and_release(self):
        first = build_race_domain_moderate_braking_plan(17022)
        self.assertEqual(first, build_race_domain_moderate_braking_plan(17022))
        self.assertNotEqual(first, build_race_domain_moderate_braking_plan(17023))
        self.assertEqual(len(first), 64)
        self.assertEqual(plan_duration_s(first), 384.0)
        self.assertTrue(all(block.duration_s == 6.0 for block in first))
        self.assertLessEqual(max(block.target_speed_mps for block in first), 11.1)
        self.assertEqual(race_domain_moderate_steering_limit(9.5), 0.14)
        self.assertEqual(race_domain_moderate_steering_limit(10.7), 0.14)
        self.assertEqual(race_domain_moderate_steering_limit(11.0), 0.12)
        self.assertEqual(race_domain_moderate_steering_limit(8.0), 0.20)

        for speed, maximum, count in ((9.5, 0.14, 6),
                                      (10.5, 0.14, 6),
                                      (11.1, 0.12, 4)):
            grid = [block for block in first
                    if block.label.startswith(f"boundary_grid_{speed:g}_")]
            self.assertEqual(len(grid), count)
            self.assertEqual(max(abs(block.steering_rad) for block in grid),
                             maximum)
            self.assertTrue(any(block.steering_rad < 0.0 for block in grid))
            self.assertTrue(any(block.steering_rad > 0.0 for block in grid))

        for index in range(6):
            group = [block for block in first
                     if block.label.startswith((
                         f"transition_reposition_{index}_",
                         f"transition_approach_{index}_",
                         f"turn_in_{index}_",
                         f"throttle_reduce_{index}_",
                         f"brake_onset_{index}_",
                         f"brake_release_{index}_"))]
            self.assertEqual(len(group), 6)
            reposition, approach, turn_in, throttle_reduce, brake, release = group
            self.assertEqual(reposition.target_speed_mps, 3.0)
            self.assertEqual(approach.steering_rad, 0.0)
            self.assertEqual(turn_in.target_speed_mps, approach.target_speed_mps)
            self.assertEqual(turn_in.steering_rad, throttle_reduce.steering_rad)
            self.assertGreater(turn_in.target_speed_mps,
                               throttle_reduce.target_speed_mps)
            self.assertEqual(brake.target_speed_mps, 3.0)
            self.assertEqual(brake.steering_rad, turn_in.steering_rad)
            self.assertEqual(release.target_speed_mps, 3.0)
            self.assertEqual(release.steering_rad, 0.0)
        self.assertEqual(sum(block.label.startswith("corner_exit_L_")
                             for block in first), 4)
        self.assertEqual(sum(block.label.startswith("corner_exit_R_")
                             for block in first), 4)


if __name__ == "__main__":
    unittest.main()
