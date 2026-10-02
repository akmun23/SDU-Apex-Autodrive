#!/usr/bin/env python3
"""Focused schedule checks for the recursive high-steering capture."""

from __future__ import annotations

import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from open_plane_excitation import build_schedule


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


if __name__ == "__main__":
    unittest.main()
