"""Stationary normalized-steering step characterization."""

import csv
from datetime import datetime, timezone
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32


class SteeringCharacterization(Node):
    def __init__(self) -> None:
        super().__init__('steering_characterization')
        self.declare_parameter('output_dir', '/workspace/src/autodrive_artifacts/calibration/raw')
        self.declare_parameter('steering_sequence', [0.0, -0.25, 0.0, 0.25, 0.0, -0.5, 0.0, 0.5, 0.0])
        self.declare_parameter('hold_sec', 1.0)
        self.declare_parameter('maximum_steering_command', 0.5)
        sequence = [float(value) for value in self.get_parameter('steering_sequence').value]
        maximum = float(self.get_parameter('maximum_steering_command').value)
        if not sequence or any(abs(value) > maximum for value in sequence):
            raise ValueError('steering sequence exceeds configured maximum')
        self._sequence = sequence
        self._hold = max(0.1, float(self.get_parameter('hold_sec').value))
        output_dir = Path(str(self.get_parameter('output_dir').value))
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        self._path = output_dir / ('steering_steps_%s.csv' % stamp)
        self._stream = self._path.open('w', newline='', encoding='utf-8')
        self._writer = csv.writer(self._stream)
        self._writer.writerow(('time_s', 'phase', 'steering_command', 'steering_feedback_rad'))
        self._steering_pub = self.create_publisher(Float32, '/autodrive/roboracer_1/steering_command', 10)
        self._throttle_pub = self.create_publisher(Float32, '/autodrive/roboracer_1/throttle_command', 10)
        self.create_subscription(Float32, '/autodrive/roboracer_1/steering', self._on_feedback, 10)
        self._feedback = float('nan')
        self._start = self.get_clock().now()
        self._phase_start = self._start
        self._index = 0
        self._finished = False
        self._timer = self.create_timer(0.02, self._tick)

    def _on_feedback(self, message: Float32) -> None:
        self._feedback = message.data

    def _neutral(self) -> None:
        self._steering_pub.publish(Float32(data=0.0))
        self._throttle_pub.publish(Float32(data=0.0))

    def _tick(self) -> None:
        now = self.get_clock().now()
        if self._index >= len(self._sequence):
            self._finished = True
            for _ in range(3):
                self._neutral()
            self._stream.flush()
            self.get_logger().info('Steering test complete: %s' % self._path)
            rclpy.shutdown(context=self.context)
            return
        if (now - self._phase_start).nanoseconds / 1e9 >= self._hold:
            self._index += 1
            self._phase_start = now
            return
        command = self._sequence[self._index]
        self._throttle_pub.publish(Float32(data=0.0))
        self._steering_pub.publish(Float32(data=command))
        elapsed = (now - self._start).nanoseconds / 1e9
        self._writer.writerow((elapsed, self._index, command, self._feedback))
        self._stream.flush()

    def destroy_node(self) -> bool:
        if not self._finished and rclpy.ok(context=self.context):
            for _ in range(3):
                self._neutral()
        if not self._stream.closed:
            self._stream.close()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SteeringCharacterization()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
