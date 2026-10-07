from __future__ import annotations

import unittest
from pathlib import Path

from tools.racing.offline.controller.production_actuator import ProductionActuator
from sdu_apex_autodrive.sdu_apex_autodrive.speed_controller import LongitudinalMode


REPO = Path(__file__).resolve().parents[4]


class ProductionActuatorTest(unittest.TestCase):
    def test_production_yaml_and_speed_controller(self):
        config = REPO / "sdu_apex_autodrive/config/actuator_interface.yaml"
        actuator = ProductionActuator(config)
        actuator.observe(1_000_000_000, 0.0, 0.0)
        first = actuator.tick(0.20, 4.0)
        self.assertAlmostEqual(first.steering_normalized, 0.20 / 0.5236)
        self.assertGreaterEqual(first.throttle_normalized, 0.0)
        self.assertLessEqual(first.throttle_normalized, 1.0)
        actuator.observe(1_025_000_000, 5.0, 0.0)
        braking = actuator.tick(0.0, 1.0)
        self.assertEqual(braking.mode, LongitudinalMode.BRAKE)
        self.assertEqual(braking.throttle_normalized, 0.0)

    def test_throttle_rise_condition_is_scoped_to_measured_regime(self):
        config = REPO / "sdu_apex_autodrive/config/actuator_interface.yaml"
        base = ProductionActuator(config)
        candidate = ProductionActuator(
            config,
            REPO / "config/racing/throttle_slew_moderate_candidate.yaml")
        for actuator in (base, candidate):
            actuator.observe(1_000_000_000, 5.0, 0.0)
        base_active = base.tick(0.16, 6.0)
        candidate_active = candidate.tick(0.16, 6.0)
        self.assertGreater(base_active.throttle_normalized, 0.0)
        self.assertAlmostEqual(
            candidate_active.throttle_normalized,
            0.267 * 0.025,
            places=6)

        base.reset()
        candidate.reset()
        for actuator in (base, candidate):
            actuator.observe(2_000_000_000, 5.0, 0.0)
        outside_regime = candidate.tick(0.10, 6.0)
        baseline_outside_regime = base.tick(0.10, 6.0)
        self.assertEqual(outside_regime, baseline_outside_regime)

        candidate.reset()
        candidate.speed_controller.last_output = 0.50
        candidate.speed_controller._active_throttle_rise_rate_per_sec = 0.267
        self.assertAlmostEqual(
            candidate.speed_controller._slew_to(0.0, 0.025), 0.25)

    def test_lowsteer_candidate_uses_measured_steering_boundary_only(self):
        config = REPO / "sdu_apex_autodrive/config/actuator_interface.yaml"
        base = ProductionActuator(config)
        candidate = ProductionActuator(
            config,
            REPO / "config/racing/throttle_slew_lowsteer_candidate.yaml")
        for actuator in (base, candidate):
            actuator.observe(1_000_000_000, 5.0, 0.0)

        capped = candidate.tick(0.08, 6.0)
        baseline = base.tick(0.08, 6.0)
        self.assertAlmostEqual(capped.throttle_normalized, 0.267 * 0.025,
                               places=6)
        self.assertGreater(baseline.throttle_normalized,
                           capped.throttle_normalized)

        base.reset()
        candidate.reset()
        for actuator in (base, candidate):
            actuator.observe(2_000_000_000, 5.0, 0.0)
        below_gate = candidate.tick(0.079, 6.0)
        baseline_below_gate = base.tick(0.079, 6.0)
        self.assertEqual(below_gate, baseline_below_gate)

    def test_highspeed_lowsteer_candidate_only_caps_the_supported_speed_band(self):
        config = REPO / "sdu_apex_autodrive/config/actuator_interface.yaml"
        base = ProductionActuator(config)
        candidate = ProductionActuator(
            config,
            REPO / "config/racing/throttle_slew_highspeed_lowsteer_candidate.yaml")
        for actuator in (base, candidate):
            actuator.observe(1_000_000_000, 6.5, 0.0)

        highspeed_gate = candidate.tick(0.08, 7.0)
        baseline_highspeed = base.tick(0.08, 7.0)
        self.assertAlmostEqual(highspeed_gate.throttle_normalized,
                               0.267 * 0.025, places=6)
        self.assertGreater(baseline_highspeed.throttle_normalized,
                           highspeed_gate.throttle_normalized)

        for measured_speed, steering in ((5.99, 0.08), (6.5, 0.079)):
            base.reset()
            candidate.reset()
            for actuator in (base, candidate):
                actuator.observe(2_000_000_000, measured_speed, 0.0)
            candidate_outside = candidate.tick(steering, 7.0)
            baseline_outside = base.tick(steering, 7.0)
            self.assertEqual(candidate_outside, baseline_outside)

    def test_fast_ramp_candidate_is_scoped_to_the_replicated_cell(self):
        config = REPO / "sdu_apex_autodrive/config/actuator_interface.yaml"
        base = ProductionActuator(config)
        candidate = ProductionActuator(
            config,
            REPO / "config/racing/throttle_slew_6p5_lowsteer_fast_ramp_candidate.yaml")
        for actuator in (base, candidate):
            actuator.observe(1_000_000_000, 6.5, 0.0)

        active = candidate.tick(0.08, 7.0)
        baseline = base.tick(0.08, 7.0)
        self.assertAlmostEqual(active.throttle_normalized, 0.533 * 0.025,
                               places=6)
        self.assertGreater(baseline.throttle_normalized,
                           active.throttle_normalized)

        for measured_speed, steering in (
                (6.29, 0.08), (6.71, 0.08), (6.5, 0.069), (6.5, 0.091)):
            base.reset()
            candidate.reset()
            for actuator in (base, candidate):
                actuator.observe(2_000_000_000, measured_speed, 0.0)
            candidate_outside = candidate.tick(steering, measured_speed + 0.5)
            baseline_outside = base.tick(steering, measured_speed + 0.5)
            self.assertEqual(candidate_outside, baseline_outside)


if __name__ == "__main__":
    unittest.main()
