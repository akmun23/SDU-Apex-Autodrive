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
            actuator.speed_controller.last_output = 0.325

        active = candidate.speed_controller._drive(
            0.405, 0.025, 6.5034, 0.08, True)
        baseline = base.speed_controller._drive(
            0.405, 0.025, 6.5034, 0.08, True)
        self.assertAlmostEqual(
            active.throttle_normalized, 0.325 + 0.5333333333333333 * 0.025)
        self.assertAlmostEqual(baseline.throttle_normalized, 0.405)

        for measured_speed, steering in (
                (6.50, 0.08), (6.51, 0.08),
                (6.5034, 0.0740), (6.5034, 0.0802)):
            base.speed_controller.reset()
            candidate.speed_controller.reset()
            base.speed_controller.last_output = 0.325
            candidate.speed_controller.last_output = 0.325
            candidate_outside = candidate.speed_controller._drive(
                0.405, 0.025, measured_speed, steering, True)
            baseline_outside = base.speed_controller._drive(
                0.405, 0.025, measured_speed, steering, True)
            self.assertEqual(candidate_outside, baseline_outside)

    def test_fast_ramp_candidate_requires_the_measured_throttle_increment(self):
        config = REPO / "sdu_apex_autodrive/config/actuator_interface.yaml"
        overlay = REPO / "config/racing/throttle_slew_6p5_lowsteer_fast_ramp_candidate.yaml"
        candidate = ProductionActuator(config, overlay)
        controller = candidate.speed_controller
        self.assertTrue(controller.config.throttle_rise_event_enabled)

        controller.last_output = 0.325
        matching = controller._drive(0.405, 0.025, 6.5034, 0.08, True)
        self.assertAlmostEqual(
            matching.throttle_normalized,
            0.325 + 0.5333333333333333 * 0.025)
        self.assertAlmostEqual(controller._throttle_rise_event_target, 0.405)

        # The intervention is latched. It does not disappear when the next
        # 40 Hz steering/speed sample moves just outside the entry cell.
        for _ in range(5):
            matching = controller._drive(0.405, 0.025, 6.51, 0.20, True)
        self.assertAlmostEqual(matching.throttle_normalized, 0.405)
        self.assertIsNone(controller._throttle_rise_event_target)

    def test_fast_ramp_candidate_does_not_match_other_actions_or_stale_state(self):
        config = REPO / "sdu_apex_autodrive/config/actuator_interface.yaml"
        overlay = REPO / "config/racing/throttle_slew_6p5_lowsteer_fast_ramp_candidate.yaml"
        cases = (
            (0.365, 6.5034, 0.08, True),  # +0.04 increment, not +0.08
            (0.405, 6.2, 0.08, True),
            (0.405, 6.5034, 0.10, True),
            (0.405, 6.5034, 0.08, False),
        )
        for desired, speed, steering, fresh in cases:
            with self.subTest(desired=desired, speed=speed,
                              steering=steering, fresh=fresh):
                candidate = ProductionActuator(config, overlay)
                controller = candidate.speed_controller
                controller.last_output = 0.325
                command = controller._drive(
                    desired, 0.025, speed, steering, fresh)
                self.assertIsNone(controller._throttle_rise_event_target)
                self.assertAlmostEqual(command.throttle_normalized, desired)

    def test_fast_ramp_candidate_latches_small_changes_and_aborts_below_output(self):
        config = REPO / "sdu_apex_autodrive/config/actuator_interface.yaml"
        overlay = REPO / "config/racing/throttle_slew_6p5_lowsteer_fast_ramp_candidate.yaml"
        candidate = ProductionActuator(config, overlay)
        controller = candidate.speed_controller
        controller.last_output = 0.325
        controller._drive(0.405, 0.025, 6.5034, 0.08, True)
        latched = controller._drive(0.350, 0.025, 6.5034, 0.08, True)
        self.assertGreater(latched.throttle_normalized, 0.3383)
        self.assertAlmostEqual(controller._throttle_rise_event_target, 0.405)

        command = controller._drive(0.300, 0.025, 6.5034, 0.08, True)
        self.assertAlmostEqual(command.throttle_normalized, 0.300)
        self.assertIsNone(controller._throttle_rise_event_target)


if __name__ == "__main__":
    unittest.main()
