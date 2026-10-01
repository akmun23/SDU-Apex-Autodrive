from __future__ import annotations

import unittest

import numpy as np

from tools.racing.offline.sensors import SyntheticImu, SyntheticRearEncoders


class SyntheticSensorTest(unittest.TestCase):
    def test_rear_encoders_integrate_and_quantize_with_published_resolution(self):
        sensor = SyntheticRearEncoders()
        first = sensor.step(1.0, 2.0, 0.025)
        expected_left_count = int(np.rint(1.0 * 0.025 / 0.059
                                          * 1920 / (2.0 * np.pi)))
        expected_right_count = int(np.rint(2.0 * 0.025 / 0.059
                                           * 1920 / (2.0 * np.pi)))
        self.assertEqual(first.left_count, expected_left_count)
        self.assertEqual(first.right_count, expected_right_count)
        self.assertAlmostEqual(first.left_angle_rad,
                               2.0 * np.pi * expected_left_count / 1920)
        self.assertAlmostEqual(first.right_angle_rad,
                               2.0 * np.pi * expected_right_count / 1920)
        self.assertEqual(first.stamp_s, 0.025)
        second = sensor.step(0.0, 0.0, 0.025)
        self.assertEqual(second.left_count, first.left_count)
        self.assertEqual(second.right_count, first.right_count)

    def test_imu_com_acceleration_maps_back_to_rear_axle(self):
        sensor = SyntheticImu(seed=1)
        sensor.reset(0.0, np.asarray([1.0, 0.0, 0.0]),
                     np.asarray([0.0, 0.0, 0.0]))
        sample = sensor.step(np.asarray([2.0, 0.0, 0.0]),
                             np.asarray([0.0375, 0.0, 0.0]), 0.025)
        self.assertAlmostEqual(sample.linear_acceleration_x_mps2, 40.0)
        self.assertAlmostEqual(sample.linear_acceleration_y_mps2, 0.0)
        self.assertEqual(sample.orientation_yaw_rad, 0.0)

    def test_imu_lateral_com_offset_tracks_angular_acceleration(self):
        sensor = SyntheticImu(seed=2)
        sensor.reset(0.0, np.asarray([2.0, 0.0, 0.0]),
                     np.asarray([0.0, 0.0, 0.0]))
        sample = sensor.step(np.asarray([2.0, 0.0, 1.0]),
                             np.asarray([0.05, 0.0, 0.0125]), 0.025)
        self.assertAlmostEqual(sample.linear_acceleration_x_mps2,
                               -0.25 * 0.15532, places=5)
        self.assertAlmostEqual(sample.linear_acceleration_y_mps2,
                               1.0 + 0.15532 / 0.025, places=5)


if __name__ == "__main__":
    unittest.main()
