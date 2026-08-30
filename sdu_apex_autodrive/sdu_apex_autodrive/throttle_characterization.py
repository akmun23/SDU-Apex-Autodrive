"""Deterministic raw-throttle characterization with owned safety limits."""

import csv
from datetime import datetime, timezone
import math
from pathlib import Path
import subprocess

from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32, Int32

from .data_recorder import CSV_FIELDS


class ThrottleCharacterization(Node):
    """Sole native-command publisher for bounded raw throttle tests."""

    def __init__(self) -> None:
        super().__init__('throttle_characterization')
        self.declare_parameter('mode', 'throttle_sweep')
        self.declare_parameter('output_dir', '/workspace/src/autodrive_artifacts/calibration/raw')
        self.declare_parameter('throttle_sequence', [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07])
        self.declare_parameter('hold_sec', 3.0)
        self.declare_parameter('zero_settle_sec', 2.0)
        self.declare_parameter('sample_rate_hz', 50.0)
        self.declare_parameter('maximum_test_speed_mps', 3.0)
        self.declare_parameter('maximum_throttle', 0.10)
        self.declare_parameter('maximum_total_duration_sec', 120.0)
        self.declare_parameter('telemetry_timeout_sec', 0.5)

        self._mode = str(self.get_parameter('mode').value)
        sequence = [float(value) for value in self.get_parameter('throttle_sequence').value]
        self._maximum_throttle = float(self.get_parameter('maximum_throttle').value)
        if not sequence or not all(math.isfinite(value) for value in sequence):
            raise ValueError('throttle sequence must contain finite values')
        if any(value < 0.0 or value > self._maximum_throttle for value in sequence):
            raise ValueError('throttle sequence exceeds configured bounds')
        hold = max(0.1, float(self.get_parameter('hold_sec').value))
        settle = max(0.1, float(self.get_parameter('zero_settle_sec').value))
        self._phases = []
        if self._mode == 'zero_throttle_decel':
            throttle = max(sequence)
            self._phases = [
                ('settle', 0.0, settle),
                ('accelerate', throttle, hold),
                ('zero_throttle_decel', 0.0, 2.0 * hold),
            ]
        elif self._mode in ('throttle_sweep', 'throttle_steps'):
            for index, throttle in enumerate(sequence):
                self._phases.append(('settle_%02d' % index, 0.0, settle))
                self._phases.append(('throttle_%0.3f' % throttle, throttle, hold))
            self._phases.append(('final_zero', 0.0, settle))
        else:
            raise ValueError('unsupported raw throttle mode: %s' % self._mode)

        output_dir = Path(str(self.get_parameter('output_dir').value))
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        self._csv_path = output_dir / ('%s_%s.csv' % (self._mode, stamp))
        self._yaml_path = self._csv_path.with_suffix('.yaml')
        self._stream = self._csv_path.open('w', newline='', encoding='utf-8')
        self._writer = csv.DictWriter(self._stream, fieldnames=CSV_FIELDS)
        self._writer.writeheader()

        self._steering_pub = self.create_publisher(
            Float32, '/autodrive/roboracer_1/steering_command', 10)
        self._throttle_pub = self.create_publisher(
            Float32, '/autodrive/roboracer_1/throttle_command', 10)
        self.create_subscription(Odometry, '/autodrive/roboracer_1/odom', self._on_odom, 10)
        self.create_subscription(Imu, '/autodrive/roboracer_1/imu', self._on_imu, 10)
        self.create_subscription(Float32, '/autodrive/roboracer_1/throttle', self._on_throttle_feedback, 10)
        self.create_subscription(Float32, '/autodrive/roboracer_1/steering', self._on_steering_feedback, 10)
        self.create_subscription(Int32, '/autodrive/roboracer_1/collision_count', self._on_collision, 10)

        self._state = {field: math.nan for field in CSV_FIELDS}
        self._state.update({
            'target_speed_mps': 0.0,
            'target_accel_mps2': 0.0,
            'steering_command': 0.0,
            'collision_count': 0,
        })
        self._start = self.get_clock().now()
        self._phase_start = self._start
        self._phase_index = 0
        self._last_odom_time = None
        self._collision_baseline = None
        self._maximum_speed = float(self.get_parameter('maximum_test_speed_mps').value)
        self._maximum_duration = float(self.get_parameter('maximum_total_duration_sec').value)
        self._telemetry_timeout = float(self.get_parameter('telemetry_timeout_sec').value)
        self._finished = False
        rate = max(1.0, float(self.get_parameter('sample_rate_hz').value))
        self._timer = self.create_timer(1.0 / rate, self._tick)
        self._write_metadata(sequence)
        self.get_logger().info('Raw throttle test owns actuator topics: %s' % self._csv_path)

    def _write_metadata(self, sequence) -> None:
        try:
            commit = subprocess.run(
                ['git', '-C', '/workspace/src', 'rev-parse', 'HEAD'],
                check=False, capture_output=True, text=True, timeout=2).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            commit = ''
        self._yaml_path.write_text(
            'mode: %s\nsequence: %s\nmaximum_throttle: %.9g\n'
            'maximum_test_speed_mps: %.9g\ngit_commit: %s\n' % (
                self._mode,
                sequence,
                self._maximum_throttle,
                self._maximum_speed,
                commit or 'unknown',
            ),
            encoding='utf-8',
        )

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

    def _on_imu(self, message: Imu) -> None:
        self._state['ax_mps2'] = message.linear_acceleration.x
        self._state['ay_mps2'] = message.linear_acceleration.y

    def _on_throttle_feedback(self, message: Float32) -> None:
        self._state['throttle_feedback'] = message.data

    def _on_steering_feedback(self, message: Float32) -> None:
        self._state['steering_feedback'] = message.data

    def _on_collision(self, message: Int32) -> None:
        self._state['collision_count'] = message.data
        if self._collision_baseline is None:
            self._collision_baseline = message.data

    def _neutral(self) -> None:
        self._steering_pub.publish(Float32(data=0.0))
        self._throttle_pub.publish(Float32(data=0.0))

    def _finish(self, reason: str, failed: bool) -> None:
        if self._finished:
            return
        self._finished = True
        for _ in range(3):
            self._neutral()
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
            self._neutral()
            if elapsed > self._telemetry_timeout:
                self._finish('odometry unavailable', True)
            return
        odom_age = (now - self._last_odom_time).nanoseconds / 1e9
        if odom_age > self._telemetry_timeout:
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
        if self._phase_index >= len(self._phases):
            self._finish('sequence complete', False)
            return

        phase, throttle, duration = self._phases[self._phase_index]
        phase_elapsed = (now - self._phase_start).nanoseconds / 1e9
        if phase_elapsed >= duration:
            self._phase_index += 1
            self._phase_start = now
            return
        self._steering_pub.publish(Float32(data=0.0))
        self._throttle_pub.publish(Float32(data=float(throttle)))
        self._state['time_s'] = elapsed
        self._state['test_phase'] = phase
        self._state['throttle_command'] = throttle
        self._writer.writerow(self._state)
        self._stream.flush()

    def destroy_node(self) -> bool:
        if not self._finished and rclpy.ok(context=self.context):
            for _ in range(3):
                self._neutral()
        if not self._stream.closed:
            self._stream.flush()
            self._stream.close()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ThrottleCharacterization()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
