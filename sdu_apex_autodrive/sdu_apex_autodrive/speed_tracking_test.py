"""Bounded closed-loop target-speed step/ramp test."""

import csv
from datetime import datetime, timezone
import math
from pathlib import Path

from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Int32

from .data_recorder import CSV_FIELDS


class SpeedTrackingTest(Node):
    def __init__(self) -> None:
        super().__init__('speed_tracking_test')
        self.declare_parameter('mode', 'speed_steps')
        self.declare_parameter('output_dir', '/workspace/src/autodrive_artifacts/calibration/raw')
        self.declare_parameter('speed_sequence_mps', [0.0, 0.5, 1.0, 1.5, 2.0, 1.0, 0.0])
        self.declare_parameter('hold_sec', 4.0)
        self.declare_parameter('sample_rate_hz', 50.0)
        self.declare_parameter('maximum_test_speed_mps', 3.0)
        self.declare_parameter('maximum_total_duration_sec', 90.0)
        self.declare_parameter('telemetry_timeout_sec', 0.5)
        self.declare_parameter('command_topic', '/cmd/controller')

        self._mode = str(self.get_parameter('mode').value)
        sequence = [float(value) for value in self.get_parameter('speed_sequence_mps').value]
        maximum_speed = float(self.get_parameter('maximum_test_speed_mps').value)
        if not sequence or not all(math.isfinite(value) for value in sequence):
            raise ValueError('speed sequence must contain finite values')
        if any(value < 0.0 or value > maximum_speed for value in sequence):
            raise ValueError('speed sequence exceeds configured test speed')
        self._sequence = sequence
        self._hold = max(0.1, float(self.get_parameter('hold_sec').value))
        self._maximum_speed = maximum_speed
        self._maximum_duration = float(self.get_parameter('maximum_total_duration_sec').value)
        self._telemetry_timeout = float(self.get_parameter('telemetry_timeout_sec').value)

        output_dir = Path(str(self.get_parameter('output_dir').value))
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        self._csv_path = output_dir / ('%s_%s.csv' % (self._mode, stamp))
        self._stream = self._csv_path.open('w', newline='', encoding='utf-8')
        self._writer = csv.DictWriter(self._stream, fieldnames=CSV_FIELDS)
        self._writer.writeheader()
        self._command_pub = self.create_publisher(
            AckermannDriveStamped,
            str(self.get_parameter('command_topic').value),
            10,
        )
        self.create_subscription(Odometry, '/autodrive/roboracer_1/odom', self._on_odom, 10)
        self.create_subscription(Float32, '/autodrive/roboracer_1/throttle_command', lambda msg: self._set('throttle_command', msg.data), 10)
        self.create_subscription(Float32, '/autodrive/roboracer_1/throttle', lambda msg: self._set('throttle_feedback', msg.data), 10)
        self.create_subscription(Float32, '/autodrive/roboracer_1/steering', lambda msg: self._set('steering_feedback', msg.data), 10)
        self.create_subscription(Int32, '/autodrive/roboracer_1/collision_count', self._on_collision, 10)
        self._state = {field: math.nan for field in CSV_FIELDS}
        self._state.update({
            'target_accel_mps2': 0.0,
            'steering_command': 0.0,
            'collision_count': 0,
        })
        self._start = self.get_clock().now()
        self._phase_start = self._start
        self._phase_index = 0
        self._last_odom_time = None
        self._collision_baseline = None
        self._finished = False
        rate = max(1.0, float(self.get_parameter('sample_rate_hz').value))
        self._timer = self.create_timer(1.0 / rate, self._tick)
        self.get_logger().info('Closed-loop speed test: %s' % self._csv_path)

    def _set(self, name, value) -> None:
        self._state[name] = value

    def _on_odom(self, message: Odometry) -> None:
        vx = message.twist.twist.linear.x
        vy = message.twist.twist.linear.y
        self._state.update({
            'speed_mps': math.hypot(vx, vy),
            'vx_mps': vx,
            'vy_mps': vy,
            'yaw_rate_radps': message.twist.twist.angular.z,
            'x_world_m': message.pose.pose.position.x,
            'y_world_m': message.pose.pose.position.y,
        })
        self._last_odom_time = self.get_clock().now()

    def _on_collision(self, message: Int32) -> None:
        self._state['collision_count'] = message.data
        if self._collision_baseline is None:
            self._collision_baseline = message.data

    def _publish_command(self, speed: float) -> None:
        message = AckermannDriveStamped()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = 'roboracer_1'
        message.drive.speed = float(speed)
        message.drive.steering_angle = 0.0
        message.drive.acceleration = 0.0
        self._command_pub.publish(message)

    def _finish(self, reason: str, failed: bool) -> None:
        if self._finished:
            return
        self._finished = True
        for _ in range(3):
            self._publish_command(0.0)
        self._stream.flush()
        self.get_logger().info('%s: %s; data=%s' % (
            'ABORT' if failed else 'COMPLETE', reason, self._csv_path))
        rclpy.shutdown(context=self.context)

    def _tick(self) -> None:
        now = self.get_clock().now()
        elapsed = (now - self._start).nanoseconds / 1e9
        if elapsed > self._maximum_duration:
            self._finish('maximum duration', True)
            return
        if self._last_odom_time is None:
            self._publish_command(0.0)
            if elapsed > self._telemetry_timeout:
                self._finish('odometry unavailable', True)
            return
        if (now - self._last_odom_time).nanoseconds / 1e9 > self._telemetry_timeout:
            self._finish('odometry timeout', True)
            return
        if float(self._state['speed_mps']) > self._maximum_speed:
            self._finish('maximum speed exceeded', True)
            return
        if (
            self._collision_baseline is not None
            and int(self._state['collision_count']) > self._collision_baseline
        ):
            self._finish('collision detected', True)
            return
        if self._phase_index >= len(self._sequence):
            self._finish('sequence complete', False)
            return
        if (now - self._phase_start).nanoseconds / 1e9 >= self._hold:
            self._phase_index += 1
            self._phase_start = now
            return

        target = self._sequence[self._phase_index]
        if self._mode == 'speed_ramp' and self._phase_index + 1 < len(self._sequence):
            fraction = min(1.0, (now - self._phase_start).nanoseconds / 1e9 / self._hold)
            target += fraction * (self._sequence[self._phase_index + 1] - target)
        self._publish_command(target)
        self._state['time_s'] = elapsed
        self._state['test_phase'] = '%s_%02d' % (self._mode, self._phase_index)
        self._state['target_speed_mps'] = target
        self._writer.writerow(self._state)
        self._stream.flush()

    def destroy_node(self) -> bool:
        if not self._finished and rclpy.ok(context=self.context):
            for _ in range(3):
                self._publish_command(0.0)
        if not self._stream.closed:
            self._stream.flush()
            self._stream.close()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SpeedTrackingTest()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
