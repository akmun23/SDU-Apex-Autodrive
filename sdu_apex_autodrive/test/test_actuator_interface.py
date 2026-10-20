import time
import unittest

import rclpy
from nav_msgs.msg import Odometry
from sdu_apex_msgs.msg import DriveCommand
from std_msgs.msg import Bool

from sdu_apex_autodrive.actuator_interface import ActuatorInterface


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

    def test_valid_command_is_forwarded_without_waiting_for_watchdog_tick(self):
        odom = Odometry()
        self.node._on_odom(odom)
        command = DriveCommand()
        command.steering_angle_rad = 0.25
        command.throttle_normalized = 0.4

        self.node._on_command(command)

        self.assertEqual(len(self.outputs), 1)
        self.assertAlmostEqual(self.outputs[0][0], 0.25 / self.node.max_steering)
        self.assertAlmostEqual(self.outputs[0][1], 0.4)
        self.node._tick()
        self.assertEqual(len(self.outputs), 1)

    def test_stale_odom_blocks_command_until_watchdog_recovers(self):
        command = DriveCommand()
        command.steering_angle_rad = -0.1
        command.throttle_normalized = 0.3
        self.node._on_command(command)
        self.assertEqual(self.outputs[-1], (0.0, 0.0))

        self.node._on_odom(Odometry())
        self.node._tick()
        self.assertAlmostEqual(self.outputs[-1][0], -0.1 / self.node.max_steering)
        self.assertAlmostEqual(self.outputs[-1][1], 0.3)

        self.node.odom_received_at = time.monotonic() - self.node.odom_timeout - 0.01
        self.node._tick()
        self.assertEqual(self.outputs[-1], (0.0, 0.0))

    def test_external_stop_publishes_neutral_immediately(self):
        self.node._on_odom(Odometry())
        command = DriveCommand()
        command.throttle_normalized = 0.6
        self.node._on_command(command)
        count_before_stop = len(self.outputs)

        stop = Bool()
        stop.data = True
        self.node._on_external_stop(stop)

        self.assertEqual(len(self.outputs), count_before_stop + 1)
        self.assertEqual(self.outputs[-1], (0.0, 0.0))
        self.assertFalse(self.node.output_active)


if __name__ == "__main__":
    unittest.main()
