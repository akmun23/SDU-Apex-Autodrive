from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.racing.offline.localization.build_production_odom import build
from tools.racing.offline.localization.production_odom import ProductionOdometry


REPO = Path(__file__).resolve().parents[4]


class ProductionOdometryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory(prefix="production-odom-test-")
        cls.library = Path(cls.temp.name) / "libproduction_odom.so"
        build(REPO, cls.library)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def test_production_config_loads_and_packet_assembly_is_synchronized(self):
        config = REPO / "f1tenth_localization/config/sensor_odometry.yaml"
        with ProductionOdometry(self.library, config) as odom:
            stamp = 1_000_000_000
            odom.add_steering_feedback(stamp - 1_000_000, 0.12)
            odom.add_throttle_feedback(stamp - 1_000_000, 0.14)
            odom.add_left_encoder(stamp, 0.0, receive_ns=stamp)
            self.assertIsNone(odom.snapshot())
            odom.add_right_encoder(stamp, 0.0, receive_ns=stamp)
            self.assertIsNone(odom.snapshot())
            odom.add_imu(stamp, 0.0, 0.0, 0.0, 0.0, receive_ns=stamp)
            self.assertIsNone(odom.snapshot())
            odom.advance_time(stamp + 1_000_000)
            first = odom.snapshot()
            self.assertIsNotNone(first)
            self.assertEqual(first.valid, 1.0)
            self.assertEqual(first.stamp_s, 1.0)
            self.assertEqual(first.dt_s, 0.0)
            self.assertEqual(first.actuator_feedback_valid, 1.0)
            self.assertAlmostEqual(first.steering_feedback_rad, 0.12)
            self.assertAlmostEqual(first.throttle_feedback, 0.14)

            # The production observer uses a fixed 25 ms physics clock even
            # when the synchronized sensor packet's source timestamps jitter.
            stamp += 17_000_000
            angle = 0.5 * 2.0 * 0.025 / 0.059
            odom.add_left_encoder(stamp, angle, receive_ns=stamp)
            odom.add_right_encoder(stamp, angle, receive_ns=stamp)
            odom.add_imu(stamp, 2.0, 0.0, 0.1, 0.0025, receive_ns=stamp)
            odom.advance_time(stamp + 1_000_000)
            second = odom.snapshot()
            self.assertIsNotNone(second)
            self.assertAlmostEqual(second.stamp_s, 1.025, places=12)
            self.assertAlmostEqual(second.dt_s, 0.025, places=9)
            self.assertTrue(np.isfinite(second.as_array()).all())
            self.assertEqual(second.yaw_rate_radps, 0.1)
            self.assertEqual(second.timing_degraded, 0.0)

            odom.reset()
            self.assertIsNone(odom.snapshot())

    def test_actuator_feedback_expires_after_production_freshness_window(self):
        config = REPO / "f1tenth_localization/config/sensor_odometry.yaml"
        with ProductionOdometry(self.library, config) as odom:
            base = 2_000_000_000
            odom.add_steering_feedback(base, 0.2)
            odom.add_throttle_feedback(base, 0.2)
            for source_stamp, receive_stamp in (
                (base + 1_000_000, base + 1_000_000),
                (base + 130_000_000, base + 130_000_000),
            ):
                odom.add_left_encoder(source_stamp, 0.0, receive_ns=receive_stamp)
                odom.add_right_encoder(source_stamp, 0.0, receive_ns=receive_stamp)
                odom.add_imu(source_stamp, 0.0, 0.0, 0.0, 0.0,
                             receive_ns=receive_stamp)
                odom.advance_time(receive_stamp + 1_000_000)
            self.assertEqual(odom.snapshot().actuator_feedback_valid, 0.0)


if __name__ == "__main__":
    unittest.main()
