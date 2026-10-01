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


if __name__ == "__main__":
    unittest.main()

