"""Live AutoDRIVE timing and data-integrity gate.

This node observes native simulator messages and the team's downstream odometry
outputs. It never republishes, resamples, interpolates, or fabricates sensor
data. A passing result means that the bridge packet stream stayed at 40 Hz,
its callback arrivals were not bursty, all required streams kept up, and the
observed values were finite and physically bounded enough to admit a run.

The bridge request timestamp is treated as a packet-boundary diagnostic only.
The independent bridge-side arrival timestamp is the authoritative check for
whether the simulator feedback loop itself is bursty.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import time
from typing import Any

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import Imu, JointState, LaserScan
from std_msgs.msg import Bool, Float64MultiArray, String


SOURCE_SENSOR_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=100,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
)
DERIVED_ODOMETRY_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=100,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
)

EXPECTED_STREAMS = (
    "left_encoder", "right_encoder", "imu", "lidar", "gt_odom",
    "odom", "odom_diagnostics",
)
PACKET_STREAM = "bridge_packet"


def _stamp_ns(msg: Any) -> int:
    stamp = msg.header.stamp
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lo = int(math.floor(position))
    hi = int(math.ceil(position))
    if lo == hi:
        return ordered[lo]
    weight = position - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def _interval_summary(
    values: list[int],
    *,
    under_ms: float = 15.0,
    over_ms: float = 40.0,
) -> dict[str, float | int]:
    if len(values) < 2:
        return {
            "samples": len(values),
            "unique_samples": len(set(values)),
            "duplicate_or_nonpositive": 0,
            "rate_hz": math.nan,
            "p01_ms": math.nan,
            "median_ms": math.nan,
            "p95_ms": math.nan,
            "p99_ms": math.nan,
            "max_ms": math.nan,
            "under_threshold_fraction": math.nan,
            "over_threshold_fraction": math.nan,
        }

    dts = [
        (values[index] - values[index - 1]) * 1e-6
        for index in range(1, len(values))
    ]
    nonpositive = sum(dt <= 0.0 for dt in dts)
    positive = [dt for dt in dts if dt > 0.0]
    span_s = (values[-1] - values[0]) * 1e-9
    rate_hz = (len(values) - 1) / span_s if span_s > 0.0 else math.nan
    return {
        "samples": len(values),
        "unique_samples": len(set(values)),
        "duplicate_or_nonpositive": nonpositive,
        "rate_hz": rate_hz,
        "p01_ms": _percentile(positive, 0.01),
        "median_ms": _percentile(positive, 0.50),
        "p95_ms": _percentile(positive, 0.95),
        "p99_ms": _percentile(positive, 0.99),
        "max_ms": max(positive) if positive else math.nan,
        "under_threshold_fraction": (
            sum(dt < under_ms for dt in positive) / len(positive)
            if positive else math.nan
        ),
        "over_threshold_fraction": (
            sum(dt > over_ms for dt in positive) / len(positive)
            if positive else math.nan
        ),
    }


def _sample_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    source = _interval_summary(
        [int(sample["source_ns"]) for sample in samples],
        under_ms=15.0,
        over_ms=40.0,
    )
    arrival = _interval_summary(
        [int(sample["arrival_ns"]) for sample in samples],
        under_ms=10.0,
        over_ms=50.0,
    )
    return {"source": source, "arrival": arrival}


class TimingValidator(Node):
    """Observe native telemetry and return a strict pass/fail verdict."""

    def __init__(self) -> None:
        super().__init__("autodrive_timing_validator")

        self.declare_parameter("duration_sec", 15.0)
        self.declare_parameter("startup_timeout_sec", 30.0)
        self.declare_parameter("expected_rate_hz", 40.0)
        self.declare_parameter("rate_tolerance_hz", 1.0)
        self.declare_parameter("min_samples", 400)
        self.declare_parameter("min_packet_coverage_fraction", 0.995)
        self.declare_parameter("max_packet_count_skew", 2)
        self.declare_parameter("source_min_p01_ms", 18.0)
        self.declare_parameter("source_median_min_ms", 22.0)
        self.declare_parameter("source_median_max_ms", 28.0)
        self.declare_parameter("source_max_p95_ms", 32.0)
        self.declare_parameter("source_max_p99_ms", 36.0)
        self.declare_parameter("max_source_burst_fraction", 0.0)
        self.declare_parameter("max_source_gap_fraction", 0.0)
        self.declare_parameter("arrival_min_p01_ms", 8.0)
        self.declare_parameter("arrival_max_p95_ms", 45.0)
        # A rare downstream executor stall is retained in the report and in
        # the gap fraction, but must not abort a long source-data recording.
        # Source/bridge cadence remains strict and the final p95/fraction
        # checks still reject sustained downstream starvation.
        self.declare_parameter("arrival_fail_fast_max_ms", 250.0)
        self.declare_parameter("max_arrival_burst_fraction", 0.005)
        self.declare_parameter("max_arrival_gap_fraction", 0.01)
        self.declare_parameter("reject_arrival_timing", True)
        self.declare_parameter("bridge_arrival_min_p01_ms", 15.0)
        self.declare_parameter("bridge_arrival_max_p95_ms", 40.0)
        self.declare_parameter("max_bridge_arrival_burst_fraction", 0.0)
        self.declare_parameter("max_bridge_arrival_gap_fraction", 0.0)
        self.declare_parameter("max_encoder_rate_radps", 1500.0)
        self.declare_parameter("max_imu_accel_mps2", 200.0)
        self.declare_parameter("max_imu_angular_rate_radps", 100.0)
        self.declare_parameter("max_odom_speed_mps", 60.0)
        self.declare_parameter("require_simulation_metadata", True)
        self.declare_parameter("fail_fast", True)
        # Raw encoder values received by one ROS subscriber are diagnostic
        # evidence only. The recorder's exact source-event rows are the
        # acceptance authority because independent subscribers can observe a
        # transient inconsistent JointState delivery under DDS scheduling.
        self.declare_parameter("reject_encoder_outliers", True)
        self.declare_parameter("initial_grace_sec", 0.0)
        self.declare_parameter(
            "report_path", "/tmp/sdu_autodrive_timing_gate.json")

        self.duration_sec = max(1.0, float(self.get_parameter("duration_sec").value))
        self.startup_timeout_sec = max(
            1.0, float(self.get_parameter("startup_timeout_sec").value))
        self.fail_fast = bool(self.get_parameter("fail_fast").value)
        self.initial_grace_sec = max(
            0.0, float(self.get_parameter("initial_grace_sec").value))
        self.require_simulation_metadata = bool(
            self.get_parameter("require_simulation_metadata").value)
        self.created_monotonic = time.monotonic()
        self.capture_start: float | None = None
        self.grace_end: float | None = None
        self.grace_complete = self.initial_grace_sec <= 0.0
        self.done = False
        self.passed = False
        self.report: dict[str, Any] | None = None
        self.seen: set[str] = set()
        self.samples: dict[str, list[dict[str, Any]]] = {
            name: [] for name in EXPECTED_STREAMS
        }
        self.packet_samples: list[dict[str, Any]] = []
        # Calibration deliberately teleports the simulator between independent
        # identification points.  That resets the cumulative encoder angles,
        # so the one transition sample is not a physical wheel-rate sample.
        # Keep the reset explicit and local to the physical-content check; the
        # source/arrival cadence checks must continue to cover the boundary.
        self.reset_active = False
        self.reset_generation = 0
        self.reset_transition_ns: list[int] = []
        self.reset_boundary_pairs_skipped = 0

        self.create_subscription(
            String, "/autodrive/roboracer_1/bridge_packet_timing",
            self._on_packet_timing, SOURCE_SENSOR_QOS)
        self.create_subscription(
            JointState, "/autodrive/roboracer_1/left_encoder",
            lambda msg: self._on_encoder("left_encoder", msg), SOURCE_SENSOR_QOS)
        self.create_subscription(
            JointState, "/autodrive/roboracer_1/right_encoder",
            lambda msg: self._on_encoder("right_encoder", msg), SOURCE_SENSOR_QOS)
        self.create_subscription(
            Imu, "/autodrive/roboracer_1/imu", self._on_imu, SOURCE_SENSOR_QOS)
        self.create_subscription(
            LaserScan, "/autodrive/roboracer_1/lidar", self._on_lidar, SOURCE_SENSOR_QOS)
        self.create_subscription(
            Odometry, "/autodrive/roboracer_1/odom",
            lambda msg: self._on_odom("gt_odom", msg), SOURCE_SENSOR_QOS)
        self.create_subscription(
            Odometry, "/odom",
            lambda msg: self._on_odom("odom", msg), DERIVED_ODOMETRY_QOS)
        self.create_subscription(
            Float64MultiArray, "/odom/diagnostics",
            self._on_odom_diagnostics, DERIVED_ODOMETRY_QOS)
        self.create_subscription(
            Bool, "/autodrive/reset_command", self._on_reset_command, 10)

        self.get_logger().info(
            "Timing gate armed: waiting for bridge packets, raw sensors, "
            "ground truth, /odom and /odom/diagnostics before capture")

    @staticmethod
    def _finite(values: tuple[float, ...] | None) -> bool:
        return values is not None and all(math.isfinite(value) for value in values)

    def _required_seen(self) -> bool:
        return self.seen >= set(EXPECTED_STREAMS) | {PACKET_STREAM}

    def _begin_capture_if_ready(self) -> bool:
        if self.capture_start is not None:
            return True
        if not self._required_seen():
            return False
        for samples in self.samples.values():
            samples.clear()
        self.packet_samples.clear()
        self.capture_start = time.monotonic()
        self.grace_end = self.capture_start + self.initial_grace_sec
        self.get_logger().info(
            f"All required streams seen; observing {self.duration_sec:.1f} s "
            f"after {self.initial_grace_sec:.1f} s startup grace")
        return False

    def _ready_after_grace(self) -> bool:
        """Discard subscriber startup queues before runtime monitoring."""
        if self.grace_complete:
            return True
        if self.grace_end is None or time.monotonic() < self.grace_end:
            return False
        for samples in self.samples.values():
            samples.clear()
        self.packet_samples.clear()
        self.capture_start = time.monotonic()
        self.grace_complete = True
        self.get_logger().info("Startup grace complete; timing monitor armed")
        return True

    def _record(
        self,
        name: str,
        source_ns: int,
        arrival_ns: int,
        values: tuple[float, ...] | None = None,
        quality: dict[str, Any] | None = None,
    ) -> None:
        self.seen.add(name)
        if not self._begin_capture_if_ready():
            return
        if not self._ready_after_grace():
            return
        sample = {
            "source_ns": int(source_ns),
            "arrival_ns": int(arrival_ns),
            "values": values,
            "quality": quality or {},
        }
        self.samples[name].append(sample)
        if self.fail_fast:
            self._check_latest(name)

    def _on_packet_timing(self, msg: String) -> None:
        arrival_callback_ns = time.monotonic_ns()
        try:
            data = json.loads(msg.data)
            sequence = int(data["packet_sequence"])
            bridge_arrival_ns = int(data["bridge_arrival_monotonic_ns"])
            request_value = data.get("request_monotonic_ns")
            request_ns = None if request_value is None else int(request_value)
            simulation_value = data.get("simulation_time_s")
            simulation_time_s = (
                None if simulation_value is None else float(simulation_value))
            frame_value = data.get("simulation_frame")
            simulation_frame = None if frame_value is None else int(frame_value)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            self.get_logger().error("Malformed bridge packet timing event")
            self.seen.add(PACKET_STREAM)
            if self.capture_start is not None:
                self._finish(force_failure="malformed bridge packet timing event")
            return

        self.seen.add(PACKET_STREAM)
        if not self._begin_capture_if_ready():
            return
        if not self._ready_after_grace():
            return
        self.packet_samples.append({
            "sequence": sequence,
            "bridge_arrival_ns": bridge_arrival_ns,
            "request_ns": request_ns,
            "simulation_time_s": simulation_time_s,
            "simulation_frame": simulation_frame,
            "validator_arrival_ns": arrival_callback_ns,
        })
        if self.fail_fast:
            self._check_latest_packet()

    def _on_encoder(self, name: str, msg: JointState) -> None:
        values = None
        if msg.position:
            values = (float(msg.position[0]),)
        self._record(
            name, _stamp_ns(msg), time.monotonic_ns(), values,
            quality={"reset_generation": self.reset_generation},
        )

    def _on_reset_command(self, msg: Bool) -> None:
        """Mark an explicit diagnostics reset as an encoder epoch boundary."""
        now_ns = time.monotonic_ns()
        if msg.data and not self.reset_active:
            self.reset_generation += 1
            self.reset_transition_ns.append(now_ns)
            # Allow both the simulator teleport and the first post-reset
            # encoder pair to arrive before applying the physical-rate check.
            # This does not relax cadence, finiteness, lidar, IMU, or odometry
            # validation, and it is only armed by the explicit reset topic.
        self.reset_active = bool(msg.data)

    def _is_reset_boundary_pair(
        self,
        previous: dict[str, Any],
        current: dict[str, Any],
    ) -> bool:
        previous_generation = int(
            previous.get("quality", {}).get("reset_generation", 0))
        current_generation = int(
            current.get("quality", {}).get("reset_generation", 0))
        if previous_generation != current_generation:
            return True

        previous_arrival = int(previous["arrival_ns"])
        current_arrival = int(current["arrival_ns"])
        # The transition callback and the sensor callback are separate DDS
        # deliveries.  Permit a small neighbourhood around the explicit
        # transition so callback ordering cannot turn a valid reset into a
        # false physical-rate failure.
        boundary_window_ns = 1_000_000_000
        return any(
            previous_arrival - boundary_window_ns <= transition <=
            current_arrival + boundary_window_ns
            for transition in self.reset_transition_ns
        )

    def _on_imu(self, msg: Imu) -> None:
        values = (
            float(msg.linear_acceleration.x),
            float(msg.linear_acceleration.y),
            float(msg.linear_acceleration.z),
            float(msg.angular_velocity.x),
            float(msg.angular_velocity.y),
            float(msg.angular_velocity.z),
        )
        self._record("imu", _stamp_ns(msg), time.monotonic_ns(), values)

    def _on_lidar(self, msg: LaserScan) -> None:
        ranges = [float(value) for value in msg.ranges]
        finite_ranges = [value for value in ranges if math.isfinite(value)]
        quality = {
            "range_count": len(ranges),
            "invalid_range_count": sum(
                math.isnan(value) or value < 0.0 for value in ranges),
            "finite_range_count": len(finite_ranges),
        }
        self._record(
            "lidar", _stamp_ns(msg), time.monotonic_ns(),
            quality=quality)

    def _on_odom(self, name: str, msg: Odometry) -> None:
        pose = msg.pose.pose.position
        twist = msg.twist.twist.linear
        values = (
            float(pose.x), float(pose.y), float(pose.z),
            float(twist.x), float(twist.y), float(twist.z),
        )
        self._record(name, _stamp_ns(msg), time.monotonic_ns(), values)

    def _on_odom_diagnostics(self, msg: Float64MultiArray) -> None:
        # The fixed-order vector stores odom_stamp_s at index 15. It is the
        # source boundary for this headerless diagnostic message.
        values = tuple(float(value) for value in msg.data)
        source_ns = 0
        if len(values) > 15 and math.isfinite(values[15]):
            source_ns = int(round(values[15] * 1e9))
        self._record(
            "odom_diagnostics", source_ns, time.monotonic_ns(), values)

    def _check_latest_packet(self) -> None:
        if len(self.packet_samples) < 2:
            return
        previous = self.packet_samples[-2]
        current = self.packet_samples[-1]
        request_ns = current["request_ns"]
        previous_request_ns = previous["request_ns"]
        if request_ns is None or previous_request_ns is None:
            self._finish(force_failure="missing bridge request timestamp")
            return
        request_dt_ms = (request_ns - previous_request_ns) * 1e-6
        arrival_dt_ms = (
            current["bridge_arrival_ns"] - previous["bridge_arrival_ns"]
        ) * 1e-6
        if request_dt_ms < 15.0 or request_dt_ms > 40.0:
            self._finish(force_failure=(
                f"bridge request interval {request_dt_ms:.3f} ms outside [15, 40] ms"))
            return
        if arrival_dt_ms < 10.0 or arrival_dt_ms > 50.0:
            self._finish(force_failure=(
                f"bridge arrival interval {arrival_dt_ms:.3f} ms outside [10, 50] ms"))
            return
        if current["sequence"] != previous["sequence"] + 1:
            self._finish(force_failure="bridge packet sequence gap or duplicate")
            return
        if self.require_simulation_metadata:
            previous_sim_time = previous["simulation_time_s"]
            current_sim_time = current["simulation_time_s"]
            if previous_sim_time is None or current_sim_time is None:
                self._finish(force_failure="missing Unity simulation time metadata")
                return
            if not math.isfinite(previous_sim_time) or not math.isfinite(current_sim_time):
                self._finish(force_failure="non-finite Unity simulation time metadata")
                return
            if current_sim_time <= previous_sim_time:
                self._finish(
                    force_failure="Unity simulation time repeated or moved backwards")
                return
            previous_sim_frame = previous["simulation_frame"]
            current_sim_frame = current["simulation_frame"]
            if previous_sim_frame is None or current_sim_frame is None:
                self._finish(force_failure="missing Unity simulation frame metadata")
                return
            if current_sim_frame <= previous_sim_frame:
                self._finish(
                    force_failure="Unity simulation frame repeated or moved backwards")

    def _check_latest(self, name: str) -> None:
        samples = self.samples[name]
        if len(samples) < 2:
            return
        previous = samples[-2]
        current = samples[-1]
        source_dt_ms = (current["source_ns"] - previous["source_ns"]) * 1e-6
        arrival_dt_ms = (current["arrival_ns"] - previous["arrival_ns"]) * 1e-6
        if source_dt_ms <= 0.0:
            self._finish(force_failure=f"{name}: duplicate or non-monotonic source stamp")
            return
        if source_dt_ms < 15.0 or source_dt_ms > 40.0:
            self._finish(force_failure=(
                f"{name}: source interval {source_dt_ms:.3f} ms outside [15, 40] ms"))
            return
        # A single ROS executor wake-up can delay or bunch a downstream
        # message while the native source timestamps remain on time.  Do not
        # abort a long recording on one mild scheduling sample; the final
        # report still enforces the 8 ms p01, 45 ms p95, and 0.5% burst / 1%
        # gap-fraction limits.  Only near-duplicates and a 100 ms stall fail
        # fast because they indicate a genuinely unhealthy stream.
        arrival_fail_fast_max_ms = float(
            self.get_parameter("arrival_fail_fast_max_ms").value)
        if arrival_dt_ms < 0.1 or arrival_dt_ms > arrival_fail_fast_max_ms:
            self._finish(force_failure=(
                f"{name}: arrival interval {arrival_dt_ms:.3f} ms outside "
                f"[0.1, {arrival_fail_fast_max_ms:.0f}] ms"))
            return
        values = current["values"]
        if values is not None and not self._finite(values):
            self._finish(force_failure=f"{name}: non-finite value")
            return
        if name == "imu" and self._finite(values):
            accel_limit = float(self.get_parameter("max_imu_accel_mps2").value)
            angular_limit = float(
                self.get_parameter("max_imu_angular_rate_radps").value)
            if max(abs(value) for value in values[:3]) > accel_limit:
                self._finish(force_failure=f"{name}: acceleration exceeds physical limit")
                return
            if max(abs(value) for value in values[3:]) > angular_limit:
                self._finish(force_failure=f"{name}: angular rate exceeds physical limit")
                return
        if name in ("gt_odom", "odom") and self._finite(values):
            speed = math.sqrt(sum(value * value for value in values[3:6]))
            if speed > float(self.get_parameter("max_odom_speed_mps").value):
                self._finish(force_failure=f"{name}: odometry speed exceeds physical limit")
                return
        if name == "lidar":
            quality = current["quality"]
            if int(quality.get("range_count", 0)) == 0:
                self._finish(force_failure="lidar: empty scan")
                return
            if int(quality.get("invalid_range_count", 0)) != 0:
                self._finish(force_failure="lidar: invalid range")
                return
        if name in ("left_encoder", "right_encoder"):
            if not self._finite(values) or not self._finite(previous["values"]):
                self._finish(force_failure=f"{name}: non-finite encoder value")
                return
            if self._is_reset_boundary_pair(previous, current):
                return
            delta = abs(values[0] - previous["values"][0])
            source_rate = delta / (source_dt_ms * 1e-3)
            arrival_rate = delta / (arrival_dt_ms * 1e-3)
            limit = float(self.get_parameter("max_encoder_rate_radps").value)
            if max(source_rate, arrival_rate) > limit:
                # Defer this check until the report can compare both wheel
                # streams. A single subscriber can receive one inconsistent
                # JointState sample even while the C++ odometry node and the
                # recorder receive the coherent pair. Cross-wheel corroboration
                # is required before rejecting an otherwise clean run.
                return

    def poll(self) -> None:
        if self.done:
            return
        now = time.monotonic()
        if self.capture_start is None:
            if now - self.created_monotonic > self.startup_timeout_sec:
                self._finish(force_failure="startup timeout waiting for required streams")
            return
        if not self._ready_after_grace():
            return
        if now - self.capture_start >= self.duration_sec:
            self._finish()

    def _validate_content(self, name: str) -> dict[str, Any]:
        samples = self.samples[name]
        nonfinite = 0
        invalid = 0
        empty_lidar = 0
        max_encoder_rate = 0.0
        encoder_rate_exceedances: list[dict[str, Any]] = []
        reset_boundary_pairs_skipped = 0
        max_imu_accel = 0.0
        max_imu_angular = 0.0
        max_odom_speed = 0.0
        for sample in samples:
            values = sample["values"]
            if values is not None and not self._finite(values):
                nonfinite += 1
            quality = sample["quality"]
            invalid += int(quality.get("invalid_range_count", 0))
            if name == "lidar" and int(quality.get("range_count", 0)) == 0:
                empty_lidar += 1

            if name == "imu" and self._finite(values):
                max_imu_accel = max(
                    max_imu_accel, max(abs(value) for value in values[:3]))
                max_imu_angular = max(
                    max_imu_angular, max(abs(value) for value in values[3:]))

            if name in ("gt_odom", "odom") and self._finite(values):
                max_odom_speed = max(
                    max_odom_speed,
                    math.sqrt(sum(value * value for value in values[3:6])))

        if name in ("left_encoder", "right_encoder"):
            for previous, current in zip(samples, samples[1:]):
                if not self._finite(previous["values"]) or not self._finite(current["values"]):
                    continue
                source_dt = (current["source_ns"] - previous["source_ns"]) * 1e-9
                arrival_dt = (current["arrival_ns"] - previous["arrival_ns"]) * 1e-9
                if self._is_reset_boundary_pair(previous, current):
                    reset_boundary_pairs_skipped += 1
                    continue
                delta = abs(current["values"][0] - previous["values"][0])
                for dt in (source_dt, arrival_dt):
                    if dt > 0.0:
                        max_encoder_rate = max(max_encoder_rate, delta / dt)
                rate = max(
                    delta / source_dt if source_dt > 0.0 else 0.0,
                    delta / arrival_dt if arrival_dt > 0.0 else 0.0,
                )
                limit = float(self.get_parameter("max_encoder_rate_radps").value)
                if rate > limit and len(encoder_rate_exceedances) < 32:
                    encoder_rate_exceedances.append({
                        "source_ns": int(current["source_ns"]),
                        "arrival_ns": int(current["arrival_ns"]),
                        "rate_radps": rate,
                    })

        return {
            "samples": len(samples),
            "nonfinite_value_samples": nonfinite,
            "invalid_lidar_ranges": invalid,
            "empty_lidar_samples": empty_lidar,
            "max_encoder_rate_radps": max_encoder_rate,
            "encoder_rate_exceedances": encoder_rate_exceedances,
            "reset_boundary_pairs_skipped": reset_boundary_pairs_skipped,
            "max_imu_accel_mps2": max_imu_accel,
            "max_imu_angular_rate_radps": max_imu_angular,
            "max_odom_speed_mps": max_odom_speed,
        }

    def _build_report(self, force_failure: str | None) -> dict[str, Any]:
        expected = float(self.get_parameter("expected_rate_hz").value)
        tolerance = max(0.0, float(self.get_parameter("rate_tolerance_hz").value))
        min_samples = int(self.get_parameter("min_samples").value)
        min_coverage = float(
            self.get_parameter("min_packet_coverage_fraction").value)
        max_skew = int(self.get_parameter("max_packet_count_skew").value)
        failures: list[str] = []
        warnings: list[str] = []
        if force_failure:
            failures.append(force_failure)
        reject_arrival_timing = bool(
            self.get_parameter("reject_arrival_timing").value)

        def arrival_issue(message: str) -> None:
            if reject_arrival_timing:
                failures.append(message)
            else:
                warnings.append(message)

        streams: dict[str, Any] = {}
        source_buckets: list[set[int]] = []
        packet_count = len(self.packet_samples)
        for name in EXPECTED_STREAMS:
            samples = self.samples[name]
            summary = _sample_summary(samples)
            content = self._validate_content(name)
            streams[name] = {
                "timing": summary,
                "content": content,
            }

            source_values = [int(sample["source_ns"]) for sample in samples]
            source_buckets.append({int(round(value / 1e6)) for value in source_values})
            source = summary["source"]
            arrival = summary["arrival"]
            rate = float(source["rate_hz"])
            source_p01 = float(source["p01_ms"])
            source_median = float(source["median_ms"])
            source_p95 = float(source["p95_ms"])
            source_p99 = float(source["p99_ms"])
            arrival_p01 = float(arrival["p01_ms"])
            arrival_p95 = float(arrival["p95_ms"])

            if int(source["samples"]) < min_samples:
                failures.append(f"{name}: only {source['samples']} samples (< {min_samples})")
            if not math.isfinite(rate) or abs(rate - expected) > tolerance:
                failures.append(
                    f"{name}: source rate {rate:.3f} Hz outside "
                    f"{expected:.3f}+/-{tolerance:.3f} Hz")
            if int(source["duplicate_or_nonpositive"]) != 0:
                failures.append(f"{name}: duplicate/non-monotonic source intervals")
            if not math.isfinite(source_p01) or source_p01 < float(
                    self.get_parameter("source_min_p01_ms").value):
                failures.append(f"{name}: source p01 {source_p01:.3f} ms is too short")
            if not math.isfinite(source_median) or not (
                    float(self.get_parameter("source_median_min_ms").value)
                    <= source_median <=
                    float(self.get_parameter("source_median_max_ms").value)):
                failures.append(f"{name}: source median {source_median:.3f} ms is not 25 ms-like")
            if not math.isfinite(source_p95) or source_p95 > float(
                    self.get_parameter("source_max_p95_ms").value):
                failures.append(f"{name}: source p95 {source_p95:.3f} ms is too large")
            if not math.isfinite(source_p99) or source_p99 > float(
                    self.get_parameter("source_max_p99_ms").value):
                failures.append(f"{name}: source p99 {source_p99:.3f} ms is too large")
            if not math.isfinite(float(source["under_threshold_fraction"])) or float(
                    source["under_threshold_fraction"]) > float(
                    self.get_parameter("max_source_burst_fraction").value):
                failures.append(f"{name}: source burst fraction is nonzero")
            if not math.isfinite(float(source["over_threshold_fraction"])) or float(
                    source["over_threshold_fraction"]) > float(
                    self.get_parameter("max_source_gap_fraction").value):
                failures.append(f"{name}: source gap fraction is too large")
            if int(arrival["duplicate_or_nonpositive"]) != 0:
                arrival_issue(f"{name}: duplicate/non-monotonic arrival intervals")
            if not math.isfinite(arrival_p01) or arrival_p01 < float(
                    self.get_parameter("arrival_min_p01_ms").value):
                arrival_issue(
                    f"{name}: ROS arrival p01 {arrival_p01:.3f} ms is too short")
            if not math.isfinite(arrival_p95) or arrival_p95 > float(
                    self.get_parameter("arrival_max_p95_ms").value):
                arrival_issue(
                    f"{name}: ROS arrival p95 {arrival_p95:.3f} ms is too large")
            if not math.isfinite(float(arrival["under_threshold_fraction"])) or float(
                    arrival["under_threshold_fraction"]) > float(
                    self.get_parameter("max_arrival_burst_fraction").value):
                arrival_issue(f"{name}: ROS arrival burst fraction is too large")
            if not math.isfinite(float(arrival["over_threshold_fraction"])) or float(
                    arrival["over_threshold_fraction"]) > float(
                    self.get_parameter("max_arrival_gap_fraction").value):
                arrival_issue(f"{name}: ROS arrival gap fraction is too large")
            if int(content["nonfinite_value_samples"]) != 0:
                failures.append(f"{name}: non-finite values observed")
            if int(content["invalid_lidar_ranges"]) != 0:
                failures.append(f"{name}: invalid lidar ranges observed")
            if int(content["empty_lidar_samples"]) != 0:
                failures.append(f"{name}: empty lidar scan observed")
            if name == "imu":
                if float(content["max_imu_accel_mps2"]) > float(
                        self.get_parameter("max_imu_accel_mps2").value):
                    failures.append(f"{name}: acceleration exceeds physical limit")
                if float(content["max_imu_angular_rate_radps"]) > float(
                        self.get_parameter("max_imu_angular_rate_radps").value):
                    failures.append(f"{name}: angular rate exceeds physical limit")
            if name in ("gt_odom", "odom") and float(
                    content["max_odom_speed_mps"]) > float(
                    self.get_parameter("max_odom_speed_mps").value):
                failures.append(f"{name}: odometry speed exceeds physical limit")

        # A validator subscriber can occasionally receive one inconsistent
        # JointState sample that is not seen by the C++ odometry subscriber or
        # the recorder. Reject encoder-rate outliers only when both wheels
        # corroborate the same source-time boundary; the accepted CSV is
        # audited separately for unilateral recorder-side outliers.
        encoder_outliers = {
            name: streams[name]["content"].get("encoder_rate_exceedances", [])
            for name in ("left_encoder", "right_encoder")
        }
        corroborated: dict[str, list[dict[str, Any]]] = {
            "left_encoder": [], "right_encoder": [],
        }
        for name, events in encoder_outliers.items():
            opposite = encoder_outliers[
                "right_encoder" if name == "left_encoder" else "left_encoder"]
            for event in events:
                if any(abs(int(event["source_ns"]) - int(other["source_ns"]))
                       <= 5_000_000 for other in opposite):
                    corroborated[name].append(event)
        for name, events in corroborated.items():
            streams[name]["content"]["corroborated_encoder_rate_exceedances"] = events
        encoder_warning = (
            "encoder-rate outliers observed by validator subscriber; "
            "verify recorder source-event rows before accepting data"
        )
        if any(corroborated.values()) and bool(
                self.get_parameter("reject_encoder_outliers").value):
            failures.append("encoder jump exceeds physical limit on both wheels")
        elif any(corroborated.values()):
            warnings.append(encoder_warning)

        packet_request = _interval_summary(
            [int(sample["request_ns"]) for sample in self.packet_samples
             if sample["request_ns"] is not None],
            under_ms=15.0,
            over_ms=40.0,
        )
        request_count = sum(
            sample["request_ns"] is not None for sample in self.packet_samples)
        packet_arrival = _interval_summary(
            [int(sample["bridge_arrival_ns"]) for sample in self.packet_samples],
            under_ms=10.0,
            over_ms=50.0,
        )
        packet_sequences = [int(sample["sequence"]) for sample in self.packet_samples]
        simulation_times = [
            float(sample["simulation_time_s"])
            for sample in self.packet_samples
            if sample["simulation_time_s"] is not None
        ]
        simulation_frames = [
            int(sample["simulation_frame"])
            for sample in self.packet_samples
            if sample["simulation_frame"] is not None
        ]
        simulation_time_summary = _interval_summary(
            [int(round(value * 1e9)) for value in simulation_times],
            under_ms=15.0,
            over_ms=50.0,
        )
        simulation_frame_summary = _interval_summary(
            simulation_frames,
            under_ms=0.0,
            over_ms=1.0,
        )
        missing_simulation_time = packet_count - len(simulation_times)
        missing_simulation_frame = packet_count - len(simulation_frames)
        packet_report = {
            "samples": packet_count,
            "request_timestamp_samples": request_count,
            "request": packet_request,
            "bridge_arrival": packet_arrival,
            "sequence_gaps": sum(
                current != previous + 1
                for previous, current in zip(packet_sequences, packet_sequences[1:])),
            "simulation_time": simulation_time_summary,
            "simulation_frame": simulation_frame_summary,
            "missing_simulation_time": missing_simulation_time,
            "missing_simulation_frame": missing_simulation_frame,
        }
        if packet_count < min_samples:
            failures.append(f"bridge_packet: only {packet_count} packets (< {min_samples})")
        if request_count != packet_count:
            failures.append(
                f"bridge_packet: {packet_count - request_count} packets missing request time")
        request_rate = float(packet_request["rate_hz"])
        arrival_rate = float(packet_arrival["rate_hz"])
        if not math.isfinite(request_rate) or abs(request_rate - expected) > tolerance:
            failures.append(f"bridge_packet: request rate {request_rate:.3f} Hz is not 40 Hz")
        if not math.isfinite(arrival_rate) or abs(arrival_rate - expected) > tolerance:
            failures.append(f"bridge_packet: bridge arrival rate {arrival_rate:.3f} Hz is not 40 Hz")
        request_p01 = float(packet_request["p01_ms"])
        request_median = float(packet_request["median_ms"])
        request_p95 = float(packet_request["p95_ms"])
        request_p99 = float(packet_request["p99_ms"])
        if not math.isfinite(request_p01) or request_p01 < float(
                self.get_parameter("source_min_p01_ms").value):
            failures.append("bridge_packet: request-side burst interval")
        if not math.isfinite(request_median) or not (
                float(self.get_parameter("source_median_min_ms").value)
                <= request_median <=
                float(self.get_parameter("source_median_max_ms").value)):
            failures.append("bridge_packet: request median is not 25 ms-like")
        if not math.isfinite(request_p95) or request_p95 > float(
                self.get_parameter("source_max_p95_ms").value):
            failures.append("bridge_packet: request-side long interval")
        if not math.isfinite(request_p99) or request_p99 > float(
                self.get_parameter("source_max_p99_ms").value):
            failures.append("bridge_packet: request-side p99 is too large")
        if not math.isfinite(float(packet_request["under_threshold_fraction"])) or float(
                packet_request["under_threshold_fraction"]) > float(
                self.get_parameter("max_source_burst_fraction").value):
            failures.append("bridge_packet: request-side burst fraction is nonzero")
        if not math.isfinite(float(packet_request["over_threshold_fraction"])) or float(
                packet_request["over_threshold_fraction"]) > float(
                self.get_parameter("max_source_gap_fraction").value):
            failures.append("bridge_packet: request-side gap fraction is too large")
        if packet_report["sequence_gaps"] != 0:
            failures.append("bridge_packet: sequence gap or duplicate")
        if self.require_simulation_metadata:
            if missing_simulation_time != 0:
                failures.append("bridge_packet: missing Unity simulation time metadata")
            if missing_simulation_frame != 0:
                failures.append("bridge_packet: missing Unity simulation frame metadata")
            if int(simulation_time_summary["duplicate_or_nonpositive"]) != 0:
                failures.append(
                    "bridge_packet: Unity simulation time repeated or moved backwards")
            if int(simulation_frame_summary["duplicate_or_nonpositive"]) != 0:
                failures.append(
                    "bridge_packet: Unity simulation frame repeated or moved backwards")
        bridge_arrival_p01 = float(packet_arrival["p01_ms"])
        bridge_arrival_p95 = float(packet_arrival["p95_ms"])
        if not math.isfinite(bridge_arrival_p01) or bridge_arrival_p01 < float(
                self.get_parameter("bridge_arrival_min_p01_ms").value):
            failures.append("bridge_packet: bridge-side burst interval")
        if not math.isfinite(bridge_arrival_p95) or bridge_arrival_p95 > float(
                self.get_parameter("bridge_arrival_max_p95_ms").value):
            failures.append("bridge_packet: bridge-side long gap")
        if not math.isfinite(float(packet_arrival["under_threshold_fraction"])) or float(
                packet_arrival["under_threshold_fraction"]) > float(
                self.get_parameter("max_bridge_arrival_burst_fraction").value):
            failures.append("bridge_packet: bridge-side burst fraction is nonzero")
        if not math.isfinite(float(packet_arrival["over_threshold_fraction"])) or float(
                packet_arrival["over_threshold_fraction"]) > float(
                self.get_parameter("max_bridge_arrival_gap_fraction").value):
            failures.append("bridge_packet: bridge-side gap fraction is nonzero")

        for name in EXPECTED_STREAMS:
            count = len(self.samples[name])
            coverage = count / packet_count if packet_count else 0.0
            streams[name]["packet_coverage_fraction"] = coverage
            if packet_count and coverage < min_coverage:
                failures.append(
                    f"{name}: packet coverage {coverage:.4%} < {min_coverage:.4%}")
            if packet_count and abs(count - packet_count) > max_skew:
                failures.append(
                    f"{name}: sample/packet count skew {count - packet_count} > {max_skew}")

        common_fraction = 0.0
        nonempty_buckets = [buckets for buckets in source_buckets if buckets]
        if nonempty_buckets:
            common = set.intersection(*nonempty_buckets)
            common_fraction = len(common) / max(len(buckets) for buckets in nonempty_buckets)
        if common_fraction < min_coverage:
            failures.append(
                f"cross-stream source-boundary coherence {common_fraction:.4%} "
                f"< {min_coverage:.4%}")

        return {
            "schema": "sdu_autodrive_timing_gate_v2",
            "generated_at_unix_s": time.time(),
            "duration_sec": self.duration_sec,
            "passed": not failures,
            "failures": failures,
            "warnings": warnings,
            "cross_stream_source_boundary_fraction": common_fraction,
            "packet": packet_report,
            "streams": streams,
        }

    def _finish(self, force_failure: str | None = None) -> None:
        if self.done:
            return
        report = self._build_report(force_failure)
        self.report = report
        self.passed = bool(report["passed"])
        self.done = True
        report_path = Path(str(self.get_parameter("report_path").value))
        report_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = report_path.with_suffix(report_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=True) + "\n",
            encoding="utf-8")
        temporary.replace(report_path)
        verdict = "PASS" if self.passed else "FAIL"
        self.get_logger().info(f"Timing/data gate {verdict}; report={report_path}")
        for failure in report["failures"]:
            self.get_logger().error(str(failure))

    def main_loop(self) -> int:
        while rclpy.ok() and not self.done:
            rclpy.spin_once(self, timeout_sec=0.1)
            self.poll()
        return 0 if self.passed else 2


def main(args=None) -> int:
    rclpy.init(args=args)
    node = TimingValidator()
    try:
        return node.main_loop()
    except KeyboardInterrupt:
        # launch sends SIGINT when the finite calibration recorder completes.
        # Preserve the monitor's evidence instead of losing the report during
        # normal teardown; the independent source-event CSV remains the data
        # acceptance authority.
        if not node.done:
            node._finish()
        return 0 if node.passed else 2
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
