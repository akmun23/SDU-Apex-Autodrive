"""Fixed-rate AutoDRIVE telemetry CSV recorder."""

import csv
from datetime import datetime, timezone
import math
from pathlib import Path

from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32, Int32


CSV_FIELDS = (
    'time_s', 'test_phase', 'target_speed_mps', 'target_accel_mps2',
    'throttle_command', 'throttle_feedback', 'steering_command',
    'steering_feedback', 'speed_mps', 'vx_mps', 'vy_mps', 'ax_mps2',
    'ay_mps2', 'yaw_rate_radps', 'x_world_m', 'y_world_m',
    'collision_count',
)


class DataRecorder(Node):
    """Record the common calibration schema; missing inputs remain NaN."""

    def __init__(self) -> None:
        super().__init__('autodrive_data_recorder')
        self.declare_parameter('output_dir', '/workspace/src/autodrive_artifacts/calibration/raw')
        self.declare_parameter('file_prefix', 'sensor_record')
        self.declare_parameter('sample_rate_hz', 50.0)
        self.declare_parameter('duration_sec', 0.0)
        self.declare_parameter('command_topic', '/cmd/controller')
        output_dir = Path(str(self.get_parameter('output_dir').value))
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        prefix = str(self.get_parameter('file_prefix').value)
        self.output_path = output_dir / ('%s_%s.csv' % (prefix, stamp))
        self._stream = self.output_path.open('w', newline='', encoding='utf-8')
        self._writer = csv.DictWriter(self._stream, fieldnames=CSV_FIELDS)
        self._writer.writeheader()
        self._state = {field: math.nan for field in CSV_FIELDS}
        self._state['test_phase'] = 'record'
        self._state['collision_count'] = 0
        self._start = self.get_clock().now()
        self._duration = max(0.0, float(self.get_parameter('duration_sec').value))

        self.create_subscription(Odometry, '/autodrive/roboracer_1/odom', self._on_odom, 10)
        self.create_subscription(Imu, '/autodrive/roboracer_1/imu', self._on_imu, 10)
        self.create_subscription(Float32, '/autodrive/roboracer_1/throttle_command', lambda msg: self._set('throttle_command', msg.data), 10)
        self.create_subscription(Float32, '/autodrive/roboracer_1/throttle', lambda msg: self._set('throttle_feedback', msg.data), 10)
        self.create_subscription(Float32, '/autodrive/roboracer_1/steering_command', lambda msg: self._set('steering_command', msg.data), 10)
        self.create_subscription(Float32, '/autodrive/roboracer_1/steering', lambda msg: self._set('steering_feedback', msg.data), 10)
        self.create_subscription(Int32, '/autodrive/roboracer_1/collision_count', lambda msg: self._set('collision_count', msg.data), 10)
        self.create_subscription(AckermannDriveStamped, str(self.get_parameter('command_topic').value), self._on_command, 10)
        rate = max(1.0, float(self.get_parameter('sample_rate_hz').value))
        self._timer = self.create_timer(1.0 / rate, self._sample)
        self.get_logger().info('Recording telemetry: %s' % self.output_path)

    def _set(self, name: str, value) -> None:
        self._state[name] = value

    def _on_command(self, message: AckermannDriveStamped) -> None:
        self._state['target_speed_mps'] = message.drive.speed
        self._state['target_accel_mps2'] = message.drive.acceleration

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

    def _on_imu(self, message: Imu) -> None:
        self._state['ax_mps2'] = message.linear_acceleration.x
        self._state['ay_mps2'] = message.linear_acceleration.y

    def _sample(self) -> None:
        elapsed = (self.get_clock().now() - self._start).nanoseconds / 1e9
        self._state['time_s'] = elapsed
        self._writer.writerow(self._state)
        self._stream.flush()
        if self._duration > 0.0 and elapsed >= self._duration:
            self.get_logger().info('Recording complete: %s' % self.output_path)
            rclpy.shutdown(context=self.context)

    def destroy_node(self) -> bool:
        if not self._stream.closed:
            self._stream.flush()
            self._stream.close()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DataRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
