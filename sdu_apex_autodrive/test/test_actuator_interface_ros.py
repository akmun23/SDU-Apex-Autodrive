import time

from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
import pytest
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Float32

from sdu_apex_autodrive.actuator_interface import ActuatorInterface


def _spin_for(executor: SingleThreadedExecutor, duration_sec: float) -> None:
    deadline = time.monotonic() + duration_sec
    while time.monotonic() < deadline:
        executor.spin_once(timeout_sec=0.01)


def test_valid_inputs_drive_then_command_watchdog_publishes_neutral():
    rclpy.init(args=[
        '--ros-args',
        '-p', 'input_topic:=/test/actuator/cmd',
        '-p', 'odom_topic:=/test/actuator/odom',
        '-p', 'steering_topic:=/test/actuator/steering',
        '-p', 'throttle_topic:=/test/actuator/throttle',
        '-p', 'command_timeout_sec:=0.12',
        '-p', 'odom_timeout_sec:=0.12',
        '-p', 'publish_rate_hz:=100.0',
    ])
    interface = ActuatorInterface()
    peer = Node('actuator_interface_test_peer')
    executor = SingleThreadedExecutor()
    executor.add_node(interface)
    executor.add_node(peer)

    commands = peer.create_publisher(
        AckermannDriveStamped, '/test/actuator/cmd', 1)
    odometry = peer.create_publisher(Odometry, '/test/actuator/odom', 1)
    steering_values = []
    throttle_values = []
    peer.create_subscription(
        Float32,
        '/test/actuator/steering',
        lambda message: steering_values.append(message.data),
        10,
    )
    peer.create_subscription(
        Float32,
        '/test/actuator/throttle',
        lambda message: throttle_values.append(message.data),
        10,
    )

    command = AckermannDriveStamped()
    command.drive.steering_angle = 0.2618
    command.drive.speed = 1.0
    odom = Odometry()
    odom.twist.twist.linear.x = 0.5

    try:
        deadline = time.monotonic() + 0.35
        while time.monotonic() < deadline:
            commands.publish(command)
            odometry.publish(odom)
            executor.spin_once(timeout_sec=0.01)

        assert any(value == pytest.approx(0.5, abs=1e-5)
                   for value in steering_values)
        assert any(value > 0.0 for value in throttle_values)

        steering_count = len(steering_values)
        throttle_count = len(throttle_values)
        _spin_for(executor, 0.25)
        assert steering_values[steering_count:]
        assert throttle_values[throttle_count:]
        assert steering_values[-1] == pytest.approx(0.0)
        assert throttle_values[-1] == pytest.approx(0.0)
    finally:
        executor.remove_node(peer)
        executor.remove_node(interface)
        peer.destroy_node()
        interface.destroy_node()
        executor.shutdown()
        if rclpy.ok():
            rclpy.shutdown()
