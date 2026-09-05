"""Live timing gate for AutoDRIVE telemetry.

The validator observes native sensor messages only. It never republishes,
resamples, interpolates, or fabricates sensor data. A passing exit status means
that the observed source timestamps, callback arrival cadence, and cross-sensor
packet stamps are coherent enough to admit a calibration/model run.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import time
from typing import Any

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import Imu, JointState, LaserScan


SENSOR_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=200,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
)


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


def _interval_summary(samples: list[tuple[int, float]]) -> dict[str, float | int]:
    if len(samples) < 2:
        return {
            "samples": len(samples),
            "rate_hz": math.nan,
            "non_monotonic": 0,
            "p01_ms": math.nan,
            "median_ms": math.nan,
            "p95_ms": math.nan,
            "under_15ms_fraction": math.nan,
            "over_40ms_fraction": math.nan,
        }

    source_dts = [
        (samples[i][0] - samples[i - 1][0]) * 1e-6
        for i in range(1, len(samples))
    ]
    non_monotonic = sum(dt <= 0.0 for dt in source_dts)
    positive = [dt for dt in source_dts if dt > 0.0]
    span_s = (samples[-1][0] - samples[0][0]) * 1e-9
    rate_hz = (len(samples) - 1) / span_s if span_s > 0.0 else math.nan
    return {
        "samples": len(samples),
        "rate_hz": rate_hz,
        "non_monotonic": non_monotonic,
        "p01_ms": _percentile(positive, 0.01),
        "median_ms": _percentile(positive, 0.50),
        "p95_ms": _percentile(positive, 0.95),
        "under_15ms_fraction": (
            sum(dt < 15.0 for dt in positive) / len(positive) if positive else math.nan
        ),
        "over_40ms_fraction": (
            sum(dt > 40.0 for dt in positive) / len(positive) if positive else math.nan
        ),
    }


def _arrival_summary(samples: list[tuple[int, float]]) -> dict[str, float | int]:
    if len(samples) < 2:
        return {
            "non_monotonic": 0,
            "p01_ms": math.nan,
            "median_ms": math.nan,
            "p95_ms": math.nan,
            "under_10ms_fraction": math.nan,
            "over_45ms_fraction": math.nan,
        }
    dts = [
        (samples[i][1] - samples[i - 1][1]) * 1000.0
        for i in range(1, len(samples))
    ]
    non_monotonic = sum(dt <= 0.0 for dt in dts)
    positive = [dt for dt in dts if dt > 0.0]
    return {
        "non_monotonic": non_monotonic,
        "p01_ms": _percentile(positive, 0.01),
        "median_ms": _percentile(positive, 0.50),
        "p95_ms": _percentile(positive, 0.95),
        "under_10ms_fraction": (
            sum(dt < 10.0 for dt in positive) / len(positive) if positive else math.nan
        ),
        "over_45ms_fraction": (
            sum(dt > 45.0 for dt in positive) / len(positive) if positive else math.nan
        ),
    }


class TimingValidator(Node):
    """Observe AutoDRIVE sensor timing and return pass/fail without driving."""

    STREAMS = ("left_encoder", "right_encoder", "imu", "lidar")

    def __init__(self) -> None:
        super().__init__("autodrive_timing_validator")
        self.declare_parameter("duration_sec", 15.0)
        self.declare_parameter("startup_timeout_sec", 30.0)
        self.declare_parameter("expected_rate_hz", 40.0)
        self.declare_parameter("rate_tolerance_hz", 1.0)
        self.declare_parameter("min_samples", 400)
        self.declare_parameter("source_min_p01_ms", 18.0)
        self.declare_parameter("source_median_min_ms", 22.0)
        self.declare_parameter("source_median_max_ms", 28.0)
        self.declare_parameter("source_max_p95_ms", 32.0)
        self.declare_parameter("max_source_burst_fraction", 0.005)
        self.declare_parameter("max_source_gap_fraction", 0.010)
        self.declare_parameter("arrival_min_p01_ms", 10.0)
        self.declare_parameter("arrival_max_p95_ms", 45.0)
        self.declare_parameter("max_arrival_burst_fraction", 0.010)
        self.declare_parameter("max_arrival_gap_fraction", 0.020)
        self.declare_parameter("min_cross_sensor_stamp_fraction", 0.99)
        self.declare_parameter(
            "report_path", "/tmp/sdu_autodrive_timing_report.json")

        self.duration_sec = max(1.0, float(self.get_parameter("duration_sec").value))
        self.startup_timeout_sec = max(
            1.0, float(self.get_parameter("startup_timeout_sec").value))
        self.samples: dict[str, list[tuple[int, float]]] = {
            name: [] for name in self.STREAMS
        }
        self.seen: set[str] = set()
        self.created_monotonic = time.monotonic()
        self.capture_start: float | None = None
        self.done = False
        self.passed = False
        self.report: dict[str, Any] | None = None

        self.create_subscription(
            JointState, "/autodrive/roboracer_1/left_encoder",
            lambda msg: self._record("left_encoder", msg), SENSOR_QOS)
        self.create_subscription(
            JointState, "/autodrive/roboracer_1/right_encoder",
            lambda msg: self._record("right_encoder", msg), SENSOR_QOS)
        self.create_subscription(
            Imu, "/autodrive/roboracer_1/imu",
            lambda msg: self._record("imu", msg), SENSOR_QOS)
        self.create_subscription(
            LaserScan, "/autodrive/roboracer_1/lidar",
            lambda msg: self._record("lidar", msg), SENSOR_QOS)

        self.get_logger().info(
            "Timing gate armed: waiting for left/right encoder, IMU and LiDAR; "
            f"then observing {self.duration_sec:.1f} s of native telemetry")

    def _record(self, name: str, msg: Any) -> None:
        arrival = time.monotonic()
        self.seen.add(name)
        if self.capture_start is None:
            if self.seen == set(self.STREAMS):
                for stream in self.samples.values():
                    stream.clear()
                self.capture_start = arrival
                self.get_logger().info("All required streams seen; timing capture started")
            return
        self.samples[name].append((_stamp_ns(msg), arrival))

    def poll(self) -> None:
        if self.done:
            return
        now = time.monotonic()
        if self.capture_start is None:
            if now - self.created_monotonic > self.startup_timeout_sec:
                self._finish(force_failure="startup timeout waiting for required streams")
            return
        if now - self.capture_start >= self.duration_sec:
            self._finish()

    def _finish(self, force_failure: str | None = None) -> None:
        if self.done:
            return
        report = self._build_report(force_failure)
        self.report = report
        self.passed = bool(report["passed"])
        self.done = True
        report_path = Path(str(self.get_parameter("report_path").value))
        report_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = report_path.with_suffix(report_path.suffix + ".tmp")
        tmp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(report_path)
        verdict = "PASS" if self.passed else "FAIL"
        self.get_logger().info(f"Timing gate {verdict}; report={report_path}")
        for failure in report["failures"]:
            self.get_logger().error(failure)

    def _build_report(self, force_failure: str | None) -> dict[str, Any]:
        expected = float(self.get_parameter("expected_rate_hz").value)
        tolerance = max(0.0, float(self.get_parameter("rate_tolerance_hz").value))
        min_samples = int(self.get_parameter("min_samples").value)
        source_min_p01 = float(self.get_parameter("source_min_p01_ms").value)
        source_med_min = float(self.get_parameter("source_median_min_ms").value)
        source_med_max = float(self.get_parameter("source_median_max_ms").value)
        source_max_p95 = float(self.get_parameter("source_max_p95_ms").value)
        max_source_burst = float(
            self.get_parameter("max_source_burst_fraction").value)
        max_source_gap = float(self.get_parameter("max_source_gap_fraction").value)
        arrival_min_p01 = float(self.get_parameter("arrival_min_p01_ms").value)
        arrival_max_p95 = float(self.get_parameter("arrival_max_p95_ms").value)
        max_arrival_burst = float(
            self.get_parameter("max_arrival_burst_fraction").value)
        max_arrival_gap = float(
            self.get_parameter("max_arrival_gap_fraction").value)
        min_coherence = float(
            self.get_parameter("min_cross_sensor_stamp_fraction").value)

        failures: list[str] = []
        if force_failure:
            failures.append(force_failure)

        streams: dict[str, Any] = {}
        for name in self.STREAMS:
            source = _interval_summary(self.samples[name])
            arrival = _arrival_summary(self.samples[name])
            streams[name] = {"source": source, "arrival": arrival}

            def fail(condition: bool, text: str) -> None:
                if condition:
                    failures.append(f"{name}: {text}")

            fail(int(source["samples"]) < min_samples,
                 f"only {source['samples']} samples (< {min_samples})")
            rate = float(source["rate_hz"])
            fail(not math.isfinite(rate) or abs(rate - expected) > tolerance,
                 f"source rate {rate:.3f} Hz outside {expected:.3f}±{tolerance:.3f} Hz")
            fail(int(source["non_monotonic"]) != 0,
                 f"{source['non_monotonic']} non-monotonic source intervals")
            p01 = float(source["p01_ms"])
            median = float(source["median_ms"])
            p95 = float(source["p95_ms"])
            fail(not math.isfinite(p01) or p01 < source_min_p01,
                 f"source p01 {p01:.3f} ms < {source_min_p01:.3f} ms")
            fail(not math.isfinite(median) or not source_med_min <= median <= source_med_max,
                 f"source median {median:.3f} ms outside [{source_med_min:.3f}, {source_med_max:.3f}] ms")
            fail(not math.isfinite(p95) or p95 > source_max_p95,
                 f"source p95 {p95:.3f} ms > {source_max_p95:.3f} ms")
            source_burst = float(source["under_15ms_fraction"])
            source_gap = float(source["over_40ms_fraction"])
            fail(not math.isfinite(source_burst) or source_burst > max_source_burst,
                 f"source burst fraction {source_burst:.4%} > {max_source_burst:.4%}")
            fail(not math.isfinite(source_gap) or source_gap > max_source_gap,
                 f"source gap fraction {source_gap:.4%} > {max_source_gap:.4%}")

            fail(int(arrival["non_monotonic"]) != 0,
                 f"{arrival['non_monotonic']} non-monotonic arrival intervals")
            ap01 = float(arrival["p01_ms"])
            ap95 = float(arrival["p95_ms"])
            arrival_burst = float(arrival["under_10ms_fraction"])
            arrival_gap = float(arrival["over_45ms_fraction"])
            fail(not math.isfinite(ap01) or ap01 < arrival_min_p01,
                 f"arrival p01 {ap01:.3f} ms < {arrival_min_p01:.3f} ms")
            fail(not math.isfinite(ap95) or ap95 > arrival_max_p95,
                 f"arrival p95 {ap95:.3f} ms > {arrival_max_p95:.3f} ms")
            fail(not math.isfinite(arrival_burst) or arrival_burst > max_arrival_burst,
                 f"arrival burst fraction {arrival_burst:.4%} > {max_arrival_burst:.4%}")
            fail(not math.isfinite(arrival_gap) or arrival_gap > max_arrival_gap,
                 f"arrival gap fraction {arrival_gap:.4%} > {max_arrival_gap:.4%}")

        stamp_sets = [set(stamp for stamp, _ in self.samples[name]) for name in self.STREAMS]
        common = set.intersection(*stamp_sets) if stamp_sets else set()
        max_count = max((len(values) for values in stamp_sets), default=0)
        coherence = len(common) / max_count if max_count else 0.0
        if coherence < min_coherence:
            failures.append(
                "cross-sensor source-stamp coherence "
                f"{coherence:.4%} < {min_coherence:.4%}")

        return {
            "schema": "sdu_autodrive_timing_gate_v1",
            "generated_at_unix_s": time.time(),
            "duration_sec": self.duration_sec,
            "passed": not failures,
            "failures": failures,
            "cross_sensor_common_stamp_fraction": coherence,
            "thresholds": {
                "expected_rate_hz": expected,
                "rate_tolerance_hz": tolerance,
                "min_samples": min_samples,
                "source_min_p01_ms": source_min_p01,
                "source_median_min_ms": source_med_min,
                "source_median_max_ms": source_med_max,
                "source_max_p95_ms": source_max_p95,
                "max_source_burst_fraction": max_source_burst,
                "max_source_gap_fraction": max_source_gap,
                "arrival_min_p01_ms": arrival_min_p01,
                "arrival_max_p95_ms": arrival_max_p95,
                "max_arrival_burst_fraction": max_arrival_burst,
                "max_arrival_gap_fraction": max_arrival_gap,
                "min_cross_sensor_stamp_fraction": min_coherence,
            },
            "streams": streams,
        }


def main(args=None) -> int:
    rclpy.init(args=args)
    node = TimingValidator()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
            node.poll()
        return 0 if node.passed else 2
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
