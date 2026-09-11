"""Record causal model-identification events without feeding simulator truth back.

The recorder keeps bridge/Unity timing diagnostics in the same event stream as
the allowed sensor and actuator-feedback messages, but never subscribes to
simulator pose, collision, lap, or other ground-truth topics.  It is intended
to be the first reproducible MPC/odometry experiment artifact while the
current speed-target actuator path is still being identified.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import statistics
import time
from typing import Any, Iterable

import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu, JointState, LaserScan
from std_msgs.msg import Float32, Float64, Float64MultiArray, Int32, String


BRIDGE_TIMING_TOPIC = "/autodrive/roboracer_1/bridge_packet_timing"
EVENT_FIELDS = (
    "event_index", "arrival_monotonic_ns", "topic", "message_type",
    "header_stamp_ns", "simulation_time_s", "payload_json",
)
SOURCE_SENSOR_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=100,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
)

EVENT_FILE_NAMES = {
    "bridge_requests.csv": (BRIDGE_TIMING_TOPIC,),
    "simulator_packets.csv": (BRIDGE_TIMING_TOPIC,),
    "imu.csv": ("/autodrive/roboracer_1/imu",),
    "encoders.csv": (
        "/autodrive/roboracer_1/left_encoder",
        "/autodrive/roboracer_1/right_encoder",
    ),
    "actuator_feedback.csv": (
        "/autodrive/roboracer_1/steering",
        "/autodrive/roboracer_1/throttle",
    ),
    "runtime_state.csv": (
        "/odom", "/ekf_odom", "/amcl_pose", "/current_map_pose",
    ),
}


def _header_stamp_ns(message: Any) -> int | None:
    header = getattr(message, "header", None)
    if header is None:
        return None
    value = int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)
    return value if value > 0 else None


def _finite_payload(values: dict[str, Any]) -> dict[str, Any]:
    """Keep JSON output deterministic and avoid serializing ROS objects."""
    return {key: value for key, value in values.items() if value is not None}


def _timing_report(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    timing = []
    for row in rows:
        if row.get("topic") != BRIDGE_TIMING_TOPIC:
            continue
        try:
            timing.append(json.loads(row["payload_json"]))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue

    source_times = [
        float(row["simulation_time_s"])
        for row in timing
        if row.get("simulation_time_s") is not None
    ]
    source_dt = [b - a for a, b in zip(source_times, source_times[1:])]
    physics_steps = [
        int(row["simulation_physics_step"])
        for row in timing
        if row.get("simulation_physics_step") is not None
    ]
    physics_dt = [b - a for a, b in zip(physics_steps, physics_steps[1:])]
    request_sequences = [
        int(row["request_sequence"])
        for row in timing
        if row.get("request_sequence") is not None
    ]
    applied_lag = [
        int(row["request_sequence"]) - int(row["applied_command_sequence"])
        for row in timing
        if row.get("request_sequence") is not None and
        row.get("applied_command_sequence") is not None
    ]

    def summary(values: list[float | int]) -> dict[str, float | int | None]:
        if not values:
            return {"count": 0, "min": None, "mean": None, "max": None}
        return {
            "count": len(values),
            "min": min(values),
            "mean": statistics.fmean(values),
            "max": max(values),
        }

    duplicate_or_reverse = sum(value <= 0.0 for value in source_dt)
    gaps_over_30ms = sum(value > 0.030 for value in source_dt)
    sequence_gaps = sum(
        b - a != 1 for a, b in zip(request_sequences, request_sequences[1:]))
    return {
        "schema_version": 1,
        "bridge_packet_count": len(timing),
        "source_time_s": summary(source_dt),
        "physics_step_delta": summary(physics_dt),
        "request_sequence": {
            "count": len(request_sequences),
            "first": request_sequences[0] if request_sequences else None,
            "last": request_sequences[-1] if request_sequences else None,
            "gaps": sequence_gaps,
        },
        "request_to_applied_command_lag": summary(applied_lag),
        "source_duplicate_or_reverse_count": duplicate_or_reverse,
        "source_gaps_over_30ms": gaps_over_30ms,
        "physics_step_present_count": len(physics_steps),
        "applied_command_present_count": sum(
            row.get("applied_command_sequence") is not None for row in timing),
    }


class ModelIdTimingRecorder(Node):
    def __init__(self) -> None:
        super().__init__("model_id_timing_recorder")
        self.declare_parameter("output_dir", "")
        self.declare_parameter("run_name", "")
        self.declare_parameter("experiment_mode", "")
        self.declare_parameter("duration_sec", 0.0)
        self.declare_parameter("qos_depth", 200)

        output_dir = str(self.get_parameter("output_dir").value).strip()
        if not output_dir:
            output_dir = "/workspace/src/sdu_apex_autodrive/artifacts/model_id"
        run_name = str(self.get_parameter("run_name").value).strip()
        if not run_name:
            run_name = time.strftime("model_id_timing_test_%Y%m%d_%H%M%S")
        self.run_dir = Path(output_dir) / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_mode = str(
            self.get_parameter("experiment_mode").value).strip()
        self.duration_sec = max(0.0, float(self.get_parameter("duration_sec").value))
        depth = max(10, int(self.get_parameter("qos_depth").value))
        self.start_monotonic_ns = time.monotonic_ns()
        self.event_index = 0
        self.rows: list[dict[str, Any]] = []
        self.stream = (self.run_dir / "events.csv").open(
            "w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.stream, fieldnames=EVENT_FIELDS)
        self.writer.writeheader()

        sensor_qos = SOURCE_SENSOR_QOS
        self.create_subscription(
            String, BRIDGE_TIMING_TOPIC,
            lambda message: self._record_bridge_timing(message), depth)
        self.create_subscription(
            Imu, "/autodrive/roboracer_1/imu",
            lambda message: self._record_ros("/autodrive/roboracer_1/imu", message), sensor_qos)
        self.create_subscription(
            JointState, "/autodrive/roboracer_1/left_encoder",
            lambda message: self._record_ros("/autodrive/roboracer_1/left_encoder", message), sensor_qos)
        self.create_subscription(
            JointState, "/autodrive/roboracer_1/right_encoder",
            lambda message: self._record_ros("/autodrive/roboracer_1/right_encoder", message), sensor_qos)
        self.create_subscription(
            Odometry, "/odom",
            lambda message: self._record_ros("/odom", message), depth)
        self.create_subscription(
            Odometry, "/ekf_odom",
            lambda message: self._record_ros("/ekf_odom", message), depth)
        self.create_subscription(
            PoseWithCovarianceStamped, "/amcl_pose",
            lambda message: self._record_ros("/amcl_pose", message), depth)
        self.create_subscription(
            PoseWithCovarianceStamped, "/current_map_pose",
            lambda message: self._record_ros("/current_map_pose", message), depth)
        self.create_subscription(
            Float32, "/autodrive/roboracer_1/steering",
            lambda message: self._record_ros("/autodrive/roboracer_1/steering", message), depth)
        self.create_subscription(
            Float32, "/autodrive/roboracer_1/throttle",
            lambda message: self._record_ros("/autodrive/roboracer_1/throttle", message), depth)
        self.create_subscription(
            AckermannDriveStamped, "/cmd/speed",
            lambda message: self._record_ros("/cmd/speed", message), depth)
        self.create_subscription(
            LaserScan, "/autodrive/roboracer_1/lidar",
            lambda message: self._record_ros("/autodrive/roboracer_1/lidar", message), sensor_qos)
        for topic in (
                "/odom/diagnostics", "/amcl_localization_health",
                "/amcl_scan_alignment", "/amcl_gpu_timing",
                "/amcl_kld_diagnostics"):
            self.create_subscription(
                Float64MultiArray, topic,
                lambda message, topic=topic: self._record_ros(topic, message), depth)
        self.create_subscription(
            Float64, "/amcl_timing",
            lambda message: self._record_ros("/amcl_timing", message), depth)
        self.create_subscription(
            Int32, "/amcl_particle_count",
            lambda message: self._record_ros("/amcl_particle_count", message), depth)
        self.stop_timer = self.create_timer(0.1, self._check_duration)
        self._write_manifest()
        self.get_logger().info(f"Recording causal model-ID events to {self.run_dir}")

    def _write_manifest(self) -> None:
        manifest = {
            "schema_version": 1,
            "run_dir": str(self.run_dir),
            "created_wall_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "ground_truth_subscriptions": [],
            "timing_topic": BRIDGE_TIMING_TOPIC,
            "event_file": "events.csv",
            "dataset_files": [
                *EVENT_FILE_NAMES,
                "gt_odom.csv",
                "experiment_schedule.csv",
            ],
            "ground_truth_file_status": (
                "packet_embedded_offline_diagnostic; no ground-truth ROS topic "
                "subscription and no runtime controller consumer"
            ),
            "source_time_policy": (
                "ROS header stamps are recorded as supplied; bridge simulation_time_s "
                "is recorded separately and is not substituted into ROS headers."
            ),
        }
        (self.run_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _record_bridge_timing(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError, json.JSONDecodeError):
            payload = {"raw": message.data}
        simulation_time = payload.get("simulation_time_s")
        self._record(
            BRIDGE_TIMING_TOPIC, "std_msgs/String", None, simulation_time, payload)

    def _record_ros(self, topic: str, message: Any) -> None:
        payload: dict[str, Any]
        if isinstance(message, Imu):
            payload = _finite_payload({
                "ax_mps2": float(message.linear_acceleration.x),
                "ay_mps2": float(message.linear_acceleration.y),
                "az_mps2": float(message.linear_acceleration.z),
                "angular_velocity_x_radps": float(message.angular_velocity.x),
                "angular_velocity_y_radps": float(message.angular_velocity.y),
                "yaw_rate_radps": float(message.angular_velocity.z),
                "orientation_x": float(message.orientation.x),
                "orientation_y": float(message.orientation.y),
                "orientation_z": float(message.orientation.z),
                "orientation_w": float(message.orientation.w),
            })
        elif isinstance(message, JointState):
            payload = _finite_payload({
                "position": list(message.position),
                "velocity": list(message.velocity),
            })
        elif isinstance(message, Odometry):
            payload = _finite_payload({
                "x_m": float(message.pose.pose.position.x),
                "y_m": float(message.pose.pose.position.y),
                "yaw_rate_radps": float(message.twist.twist.angular.z),
                "speed_mps": float(message.twist.twist.linear.x),
                "lateral_speed_mps": float(message.twist.twist.linear.y),
            })
        elif isinstance(message, PoseWithCovarianceStamped):
            payload = _finite_payload({
                "x_m": float(message.pose.pose.position.x),
                "y_m": float(message.pose.pose.position.y),
            })
        elif isinstance(message, Float32):
            payload = {"value": float(message.data)}
        elif isinstance(message, AckermannDriveStamped):
            payload = _finite_payload({
                "steering_angle_rad": float(message.drive.steering_angle),
                "speed_mps": float(message.drive.speed),
                "acceleration_mps2": float(message.drive.acceleration),
            })
        elif isinstance(message, LaserScan):
            finite_ranges = [
                float(value) for value in message.ranges
                if value == value and value not in (float("inf"), float("-inf"))
            ]
            payload = _finite_payload({
                "range_count": len(message.ranges),
                "finite_range_count": len(finite_ranges),
                "min_range_m": min(finite_ranges) if finite_ranges else None,
                "max_range_m": max(finite_ranges) if finite_ranges else None,
                "angle_min_rad": float(message.angle_min),
                "angle_increment_rad": float(message.angle_increment),
            })
        elif isinstance(message, Float64MultiArray):
            payload = {"data": [float(value) for value in message.data]}
        elif isinstance(message, Float64):
            payload = {"value": float(message.data)}
        elif isinstance(message, Int32):
            payload = {"value": int(message.data)}
        else:
            payload = {"repr": repr(message)}
        self._record(topic, type(message).__name__, _header_stamp_ns(message), None, payload)

    def _record(
            self,
            topic: str,
            message_type: str,
            header_stamp_ns: int | None,
            simulation_time_s: float | None,
            payload: dict[str, Any]) -> None:
        row = {
            "event_index": self.event_index,
            "arrival_monotonic_ns": time.monotonic_ns(),
            "topic": topic,
            "message_type": message_type,
            "header_stamp_ns": header_stamp_ns,
            "simulation_time_s": simulation_time_s,
            "payload_json": json.dumps(payload, sort_keys=True, separators=(",", ":")),
        }
        self.event_index += 1
        self.rows.append(row)
        self.writer.writerow(row)
        self.stream.flush()

    def _check_duration(self) -> None:
        if self.duration_sec > 0.0 and (
                time.monotonic_ns() - self.start_monotonic_ns) / 1e9 >= self.duration_sec:
            self.get_logger().info("Model-ID recording duration reached")
            rclpy.shutdown()

    def close(self) -> None:
        self._write_dataset_files()
        report = _timing_report(self.rows)
        report["event_count"] = len(self.rows)
        report["duration_wall_s"] = (
            time.monotonic_ns() - self.start_monotonic_ns) / 1e9
        (self.run_dir / "timing_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self.stream.close()

    def _write_dataset_files(self) -> None:
        """Materialize the handoff's analysis files from the event log.

        These are deliberately thin, lossless topic partitions: the canonical
        event stream remains ``events.csv`` and payloads stay JSON so adding a
        field to a message does not silently change the tabular schema.
        Runtime ground truth is intentionally not subscribed to, so its file
        is a status marker rather than fabricated data.
        """
        timing_rows = self._timing_rows()
        self._write_timing_rows(self.run_dir / "bridge_requests.csv", timing_rows)
        self._write_timing_rows(self.run_dir / "simulator_packets.csv", timing_rows)
        self._write_ground_truth_rows(self.run_dir / "gt_odom.csv", timing_rows)

        for file_name, topics in EVENT_FILE_NAMES.items():
            if file_name in {"bridge_requests.csv", "simulator_packets.csv"}:
                continue
            rows = [row for row in self.rows if row["topic"] in topics]
            self._write_rows(self.run_dir / file_name, rows)
        self._write_rows(
            self.run_dir / "experiment_schedule.csv",
            [{
                "phase": "configured_capture_window",
                "mode": self.experiment_mode,
                "duration_sec": self.duration_sec,
                "source": "model_identification.launch.py",
                "status": "recorded",
            }],
            fieldnames=("phase", "mode", "duration_sec", "source", "status"),
        )

    @staticmethod
    def _write_rows(
            path: Path,
            rows: list[dict[str, Any]],
            fieldnames: tuple[str, ...] = EVENT_FIELDS) -> None:
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    def _timing_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for event in self.rows:
            if event["topic"] != BRIDGE_TIMING_TOPIC:
                continue
            try:
                payload = json.loads(event["payload_json"])
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            row = {
                "run_id": self.run_dir.name,
                "event_index": event["event_index"],
                "bridge_arrival_monotonic_ns": payload.get(
                    "bridge_arrival_monotonic_ns", event["arrival_monotonic_ns"]),
                "request_monotonic_ns": payload.get("request_monotonic_ns"),
                "request_sequence": payload.get("request_sequence"),
                "packet_sequence": payload.get("packet_sequence"),
                "telemetry_sequence": payload.get("telemetry_sequence"),
                "simulation_time_s": payload.get("simulation_time_s"),
                "simulation_physics_step": payload.get("simulation_physics_step"),
                "simulation_render_frame": payload.get("simulation_render_frame"),
                "sent_throttle_norm": payload.get("sent_throttle_norm"),
                "sent_steering_norm": payload.get("sent_steering_norm"),
                "sent_reset": payload.get("sent_reset"),
                "applied_command_sequence": payload.get("applied_command_sequence"),
                "commanded_throttle_norm": payload.get("commanded_throttle_norm"),
                "commanded_steering_norm": payload.get("commanded_steering_norm"),
                "applied_throttle_norm": payload.get("applied_throttle_norm"),
                "applied_steering_norm": payload.get("applied_steering_norm"),
                "response_received": True,
                "associated_packet_sequence": payload.get("packet_sequence"),
            }
            row.update({
                key: value for key, value in payload.items()
                if key.startswith("simulator_")
            })
            rows.append(row)
        return rows

    @staticmethod
    def _write_timing_rows(path: Path, rows: list[dict[str, Any]]) -> None:
        base_fields = (
            "run_id", "event_index", "request_sequence", "request_monotonic_ns",
            "packet_sequence", "associated_packet_sequence", "response_received",
            "bridge_arrival_monotonic_ns", "telemetry_sequence", "simulation_time_s",
            "simulation_physics_step", "simulation_render_frame", "sent_throttle_norm",
            "sent_steering_norm", "sent_reset", "commanded_throttle_norm",
            "commanded_steering_norm", "applied_command_sequence",
            "applied_throttle_norm", "applied_steering_norm",
        )
        extra_fields = sorted({key for row in rows for key in row
                               if key.startswith("simulator_")})
        ModelIdTimingRecorder._write_rows(path, rows, base_fields + tuple(extra_fields))

    @staticmethod
    def _write_ground_truth_rows(path: Path, rows: list[dict[str, Any]]) -> None:
        truth_keys = (
            "simulator_position_x", "simulator_position_y", "simulator_position_z",
            "simulator_orientation_quaternion_x",
            "simulator_orientation_quaternion_y",
            "simulator_orientation_quaternion_z",
            "simulator_orientation_quaternion_w",
            "simulator_orientation_euler_x", "simulator_orientation_euler_y",
            "simulator_orientation_euler_z", "simulator_linear_velocity_x",
            "simulator_linear_velocity_y", "simulator_linear_velocity_z",
            "simulator_angular_velocity_x", "simulator_angular_velocity_y",
            "simulator_angular_velocity_z", "simulator_linear_acceleration_x",
            "simulator_linear_acceleration_y", "simulator_linear_acceleration_z",
            "simulator_encoder_angles_left", "simulator_encoder_angles_right",
        )
        selected = [row for row in rows if any(row.get(key) is not None for key in truth_keys)]
        if not selected:
            ModelIdTimingRecorder._write_rows(
                path,
                [{"status": "not_recorded", "reason": "no simulator packet fields present"}],
                ("status", "reason"),
            )
            return
        fields = (
            "run_id", "packet_sequence", "request_sequence", "telemetry_sequence",
            "simulation_time_s", "simulation_physics_step", "simulation_render_frame",
            *truth_keys,
        )
        ModelIdTimingRecorder._write_rows(path, selected, fields)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ModelIdTimingRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
