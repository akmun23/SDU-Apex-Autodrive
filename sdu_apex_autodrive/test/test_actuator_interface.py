import unittest

import rclpy
from sdu_apex_msgs.msg import DriveCommand

from sdu_apex_autodrive.actuator_interface import (
    MIN_DRIVE_THROTTLE,
    ActuatorInterface,
)


class ActuatorInterfaceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.node = ActuatorInterface()
        self.outputs = []
        self.node._publish = lambda steering, throttle: self.outputs.append(
            (steering, throttle))

    def tearDown(self):
        self.node.destroy_node()

    def test_valid_command_is_forwarded_immediately_and_once(self):
        command = DriveCommand()
        command.steering_angle_rad = 0.25
        command.throttle_normalized = 0.4

        self.node._on_command(command)

        self.assertEqual(len(self.outputs), 1)
        self.assertAlmostEqual(self.outputs[0][0], 0.25 / self.node.max_steering)
        self.assertAlmostEqual(self.outputs[0][1], 0.4)

    def test_float32_rounding_at_minimum_drive_is_accepted(self):
        command = DriveCommand()
        command.steering_angle_rad = 0.0
        command.throttle_normalized = MIN_DRIVE_THROTTLE

        self.node._on_command(command)

        self.assertEqual(len(self.outputs), 1)
        self.assertEqual(self.outputs[0][0], 0.0)
        self.assertAlmostEqual(self.outputs[0][1], MIN_DRIVE_THROTTLE, places=7)

    def test_zero_throttle_is_rejected_without_substitute_output(self):
        command = DriveCommand()
        command.steering_angle_rad = -0.1
        command.throttle_normalized = 0.0
        self.node._on_command(command)
        self.assertEqual(self.outputs, [])

    def test_invalid_command_does_not_publish_fallback(self):
        command = DriveCommand()
        command.steering_angle_rad = 0.1
        command.throttle_normalized = MIN_DRIVE_THROTTLE - 0.001

        self.node._on_command(command)

        self.assertEqual(self.outputs, [])


if __name__ == "__main__":
    unittest.main()
