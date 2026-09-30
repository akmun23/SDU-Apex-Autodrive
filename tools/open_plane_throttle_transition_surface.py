#!/usr/bin/env python3
"""Reset-isolated throttle transition sweep for the Explore simulator.

Development only. The simulator's built-in reset returns the car to its spawn
after every throttle condition. No speed or distance limit is applied to the
measured interval; collisions and bridge timing faults still stop the run.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import signal
import time
from pathlib import Path

import rclpy
from nav_msgs.msg import Odometry
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import Bool, Float32, Int32, String


ODOM = "/autodrive/roboracer_1/odom"
THROTTLE_FEEDBACK = "/autodrive/roboracer_1/throttle"
STEERING_FEEDBACK = "/autodrive/roboracer_1/steering"
COLLISIONS = "/autodrive/roboracer_1/collision_count"
TIMING_FAULT = "/autodrive/roboracer_1/bridge_timing_fault"
THROTTLE_COMMAND = "/autodrive/roboracer_1/throttle_command"
STEERING_COMMAND = "/autodrive/roboracer_1/steering_command"
RESET_COMMAND = "/autodrive/reset_command"
PHASE = "/open_plane_experiment/phase"
RATE_HZ = 40.0
RESET_HOLD_S = 0.90
RESET_TIMEOUT_S = 30.0
ODOM_TIMEOUT_S = 0.25
COLLISION_TIMEOUT_S = 0.50
SPAWN_POSITION_TOLERANCE_M = 0.25
BASELINE_DWELL_S = 4.0
RESPONSE_DWELL_S = 8.0
THROTTLE_TARGET_STEP_PERCENT = 5
THROTTLE_BASELINE_STEP_PERCENT = 10
THROTTLE_STEP_DELTAS_PERCENT = (5, 10, 20, 40)
THROTTLE_REPEAT_COUNT = 2
STEERING_LIMIT_RAD = 0.5236
STEERING_FEEDBACK_TOLERANCE_RAD = 0.02
THROTTLE_FEEDBACK_TOLERANCE = 0.005


def _condition_key(steering: float, throttle_start: float,
                   throttle_end: float, replicate: int) -> tuple[int, int, int, int]:
    return (
        round(steering * 10_000),
        round(throttle_start * 100),
        round(throttle_end * 100),
        int(replicate),
    )


class ThrottleTransitionExperiment:
    def __init__(self, seed: int, steering_angles_rad: list[float],
                 target_step_percent: int, baseline_step_percent: int,
                 repeat_count: int,
                 all_pairs: list[tuple[float, float, float, int]] | None = None,
                 pairs: list[tuple[float, float, float, int]] | None = None,
                 resume_metadata: dict[str, object] | None = None) -> None:
        self.seed = seed
        self.steering_angles_rad = steering_angles_rad
        self.target_step_percent = target_step_percent
        self.baseline_step_percent = baseline_step_percent
        self.repeat_count = repeat_count
        self.all_pairs = (all_pairs if all_pairs is not None else
                          self._make_pairs(seed, steering_angles_rad,
                                           target_step_percent,
                                           baseline_step_percent, repeat_count))
        self.pairs = self.all_pairs if pairs is None else pairs
        if not self.pairs:
            raise ValueError("no throttle conditions remain to run")
        self.resume_metadata = resume_metadata or {}

        self.node = rclpy.create_node("open_plane_throttle_transition_surface")
        self.throttle_pub = self.node.create_publisher(Float32, THROTTLE_COMMAND, 1)
        self.steering_pub = self.node.create_publisher(Float32, STEERING_COMMAND, 1)
        self.reset_pub = self.node.create_publisher(Bool, RESET_COMMAND, 1)
        self.phase_pub = self.node.create_publisher(String, PHASE, 10)
        sensor_qos = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        self.node.create_subscription(Odometry, ODOM, self._on_odom, sensor_qos)
        self.node.create_subscription(
            Float32, THROTTLE_FEEDBACK, self._on_throttle, sensor_qos)
        self.node.create_subscription(
            Float32, STEERING_FEEDBACK, self._on_steering, sensor_qos)
        self.node.create_subscription(Int32, COLLISIONS, self._on_collision, sensor_qos)
        self.node.create_subscription(Bool, TIMING_FAULT, self._on_timing_fault, 10)
        self.node.create_timer(1.0 / RATE_HZ, self._tick)

        self.replicate_index = 1
        self.design = {
            "name": "stratified_5pct_throttle_step_response",
            "target_step_percent": target_step_percent,
            "baseline_step_percent": baseline_step_percent,
            "step_deltas_percent": list(THROTTLE_STEP_DELTAS_PERCENT),
            "repeat_count": repeat_count,
            "baseline_dwell_s": BASELINE_DWELL_S,
            "response_dwell_s": RESPONSE_DWELL_S,
        }
        self.started_at = time.monotonic()
        self.state = "waiting"
        self.state_started_at = self.started_at
        self.reset_released_at: float | None = None
        self.reset_stable_since: float | None = None
        self.reset_reason = ""
        self.resume_pair_index = 0
        self.spawn_xy: tuple[float, float] | None = None
        self.position_xy: tuple[float, float] | None = None
        self.speed_mps: float | None = None
        self.vx_mps: float | None = None
        self.vy_mps: float | None = None
        self.yaw_rate_rps: float | None = None
        self.tilt_rad: float | None = None
        self.last_odom_at: float | None = None
        self.phase_odom_receipts: list[float] = []
        self.phase_max_speed_mps = 0.0
        self.phase_max_tilt_rad = 0.0
        self.phase_max_displacement_m = 0.0
        self.phase_started_at: float | None = None
        self.target_started_at: float | None = None
        self.response_observation_s: float | None = None
        self.phase_index = -1
        self.phase_label = ""
        self.phase_steering_rad = 0.0
        self.throttle_start_norm = 0.0
        self.throttle_end_norm = 0.0
        self.phase_start_state: dict[str, float] = {}
        self.baseline_state: dict[str, float] = {}
        self.throttle_feedback: float | None = None
        self.steering_feedback: float | None = None
        self.last_collision_at: float | None = None
        self.collision_count: int | None = None
        self.valid_pairs = 0
        self.invalid_pairs = 0
        self.pairs_attempted = 0
        self.resets_requested = 0
        self.resets_recovered = 0
        self.stream_quality_failure_count = 0
        self.stream_quality_failure_examples: list[str] = []
        self.last_heartbeat_at = self.started_at
        self.done = False
        self.aborted = False
        self.reason = ""

    @staticmethod
    def _make_pairs(seed: int, steering_angles_rad: list[float],
                    target_step_percent: int, baseline_step_percent: int,
                    repeat_count: int
                    ) -> list[tuple[float, float, float, int]]:
        transitions = {
            (0, end)
            for end in range(target_step_percent, 101, target_step_percent)
        }
        for start in range(baseline_step_percent, 100,
                           baseline_step_percent):
            for delta in THROTTLE_STEP_DELTAS_PERCENT:
                end = start + delta
                if end <= 100:
                    transitions.add((start, end))
            transitions.add((start, 100))

        pairs = [
            (steering, start / 100.0, end / 100.0, replicate)
            for replicate in range(1, repeat_count + 1)
            for steering in steering_angles_rad
            for start, end in sorted(transitions)
        ]
        random.Random(seed).shuffle(pairs)
        return pairs

    @staticmethod
    def _select_resume_pairs(
            all_pairs: list[tuple[float, float, float, int]],
            resume_analysis: Path, seed: int,
            steering_angles_rad: list[float], target_step_percent: int,
            baseline_step_percent: int, repeat_count: int
            ) -> tuple[list[tuple[float, float, float, int]], dict[str, object]]:
        try:
            source = resume_analysis.resolve(strict=True)
            previous = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read resume analysis {resume_analysis}: {exc}") from exc

        if previous.get("profile") != "throttle_transition_surface":
            raise ValueError("resume analysis is not a throttle-transition surface")
        design = previous.get("design")
        if not isinstance(design, dict) or design.get("name") != (
                "stratified_5pct_throttle_step_response"):
            raise ValueError("resume analysis has an unknown throttle design")
        if int(previous.get("experiment_end", {}).get("seed", -1)) != seed:
            raise ValueError("resume seed does not match the source analysis")
        if (int(design.get("target_step_percent", -1)) != target_step_percent
                or int(design.get("baseline_step_percent", -1))
                != baseline_step_percent
                or int(design.get("repeat_count", -1)) != repeat_count
                or [round(float(value) * 10_000) for value in
                    design.get("steering_angles_rad", [])]
                != [round(value * 10_000) for value in steering_angles_rad]
                or int(design.get("total_conditions", -1)) != len(all_pairs)):
            raise ValueError("resume analysis does not match the requested design")

        counts = previous.get("condition_counts", {})
        if (int(counts.get("expected", -1)) != len(all_pairs)
                or int(counts.get("duplicate", -1)) != 0
                or int(counts.get("unexpected", -1)) != 0):
            raise ValueError("source analysis has inconsistent condition coverage")

        rows = previous.get("conditions")
        if not isinstance(rows, list):
            raise ValueError("source analysis has no per-condition results")
        usable_keys: set[tuple[int, int, int, int]] = set()
        for row in rows:
            if row.get("usable_for_response_fit") is not True:
                continue
            key = _condition_key(
                float(row["steering_command_rad"]),
                float(row["throttle_start_norm"]),
                float(row["throttle_end_norm"]),
                int(row["replicate_index"]))
            if key in usable_keys:
                raise ValueError("source analysis repeats a fit-usable condition")
            usable_keys.add(key)

        full_keys = {_condition_key(*pair) for pair in all_pairs}
        if not usable_keys.issubset(full_keys):
            raise ValueError("source analysis contains conditions outside this design")
        remaining = [pair for pair in all_pairs
                     if _condition_key(*pair) not in usable_keys]
        if not remaining:
            raise ValueError("source analysis already covers every usable condition")
        metadata: dict[str, object] = {
            "resume_mode": "supplement",
            "source_analysis": str(source),
            "source_run_id": source.parent.name,
            "source_usable_conditions_skipped": len(usable_keys),
            "supplement_conditions_scheduled": len(remaining),
        }
        return remaining, metadata

    def _design_metadata(self) -> dict[str, object]:
        unique_transitions = {
            (round(start * 100), round(end * 100))
            for _, start, end, _ in self.all_pairs
        }
        design: dict[str, object] = {
            **self.design,
            "steering_angles_rad": self.steering_angles_rad,
            "unique_transitions_per_steering": len(unique_transitions),
            "total_conditions": len(self.all_pairs),
            "run_condition_count": len(self.pairs),
            "scheduled_conditions": [
                {
                    "steering_key_1e4_rad": round(steering * 10_000),
                    "throttle_start_percent": round(start * 100),
                    "throttle_end_percent": round(end * 100),
                    "replicate_index": replicate,
                }
                for steering, start, end, replicate in self.pairs
            ],
        }
        design.update(self.resume_metadata)
        return design

    def _publish_event(self, payload: dict[str, object]) -> None:
        payload.setdefault("wall_time_ns", time.time_ns())
        self.phase_pub.publish(String(data=json.dumps(
            payload, separators=(",", ":"), sort_keys=True)))

    def _command(self, throttle: float, steering: float = 0.0) -> None:
        self.throttle_pub.publish(Float32(data=max(0.0, min(1.0, throttle))))
        steering_rad = max(-STEERING_LIMIT_RAD,
                           min(STEERING_LIMIT_RAD, steering))
        self.steering_pub.publish(Float32(data=steering_rad / STEERING_LIMIT_RAD))

    def _on_odom(self, msg: Odometry) -> None:
        vx = float(msg.twist.twist.linear.x)
        vy = float(msg.twist.twist.linear.y)
        yaw = float(msg.twist.twist.angular.z)
        speed = math.hypot(vx, vy)
        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)
        q = msg.pose.pose.orientation
        tilt_cosine = 1.0 - 2.0 * (float(q.x) ** 2 + float(q.y) ** 2)
        tilt = math.acos(max(-1.0, min(1.0, tilt_cosine)))
        if not all(map(math.isfinite, (vx, vy, yaw, speed, x, y, tilt))):
            return
        now = time.monotonic()
        self.vx_mps = vx
        self.vy_mps = vy
        self.yaw_rate_rps = yaw
        self.speed_mps = speed
        self.position_xy = (x, y)
        self.tilt_rad = tilt
        self.last_odom_at = now
        if self.phase_started_at is not None and self.state in (
                "start_command", "target_command"):
            self.phase_odom_receipts.append(now)
            self.phase_max_speed_mps = max(self.phase_max_speed_mps, speed)
            self.phase_max_tilt_rad = max(self.phase_max_tilt_rad, tilt)
            if self.spawn_xy is not None:
                self.phase_max_displacement_m = max(
                    self.phase_max_displacement_m,
                    math.dist(self.spawn_xy, (x, y)),
                )

    def _on_throttle(self, msg: Float32) -> None:
        value = float(msg.data)
        if not math.isfinite(value):
            return
        self.throttle_feedback = value

    def _on_steering(self, msg: Float32) -> None:
        value = float(msg.data)
        if math.isfinite(value):
            self.steering_feedback = value

    def _on_collision(self, msg: Int32) -> None:
        self.last_collision_at = time.monotonic()
        self.collision_count = int(msg.data)
        if self.collision_count != 0:
            self._finish(f"collision_count={self.collision_count}", aborted=True)

    def _on_timing_fault(self, msg: Bool) -> None:
        if bool(msg.data):
            self._finish("bridge timing fault", aborted=True)

    def _current_state(self) -> dict[str, float]:
        return {
            "speed_mps": float(self.speed_mps or 0.0),
            "vx_mps": float(self.vx_mps or 0.0),
            "vy_mps": float(self.vy_mps or 0.0),
            "yaw_rate_rps": float(self.yaw_rate_rps or 0.0),
            "throttle_feedback_norm": float(self.throttle_feedback or 0.0),
            "steering_feedback_rad": float(self.steering_feedback or 0.0),
            "tilt_deg": math.degrees(self.tilt_rad or 0.0),
            "x_m": float(self.position_xy[0]) if self.position_xy else 0.0,
            "y_m": float(self.position_xy[1]) if self.position_xy else 0.0,
        }

    def _start_pair(self, pair_index: int) -> None:
        if pair_index >= len(self.pairs):
            self._finish("all throttle transitions complete", aborted=False)
            return
        self.phase_index = pair_index
        (self.phase_steering_rad, self.throttle_start_norm,
         self.throttle_end_norm, self.replicate_index) = self.pairs[pair_index]
        steering_millirad = int(round(abs(self.phase_steering_rad) * 1000.0))
        steering_tag = (
            f"{'p' if self.phase_steering_rad >= 0.0 else 'n'}{steering_millirad:04d}")
        self.phase_label = (
            f"throttle_r{self.replicate_index}_{steering_tag}_"
            f"{int(round(self.throttle_start_norm * 100)):03d}_"
            f"{int(round(self.throttle_end_norm * 100)):03d}")
        self.pairs_attempted += 1
        self.phase_started_at = time.monotonic()
        self.state = "start_command"
        self.state_started_at = self.phase_started_at
        self.target_started_at = None
        self.response_observation_s = None
        self.phase_start_state = self._current_state()
        self.baseline_state = {}
        self.phase_odom_receipts.clear()
        self.phase_max_speed_mps = float(self.speed_mps or 0.0)
        self.phase_max_tilt_rad = float(self.tilt_rad or 0.0)
        self.phase_max_displacement_m = 0.0
        self._publish_event({
            "event": "phase_start",
            "profile": "throttle_transition_surface",
            "seed": self.seed,
            "phase_index": pair_index,
            "phase_count": len(self.pairs),
            "replicate_index": self.replicate_index,
            "repeat_count": self.repeat_count,
            "label": self.phase_label,
            "target_speed_mps": 0.0,
            "steering_command_rad": self.phase_steering_rad,
            "throttle_mode": "transition_surface",
            "throttle_start_norm": self.throttle_start_norm,
            "throttle_end_norm": self.throttle_end_norm,
            "initial_state": self.phase_start_state,
            "monotonic_ns": time.monotonic_ns(),
        })
        if (pair_index + 1) % 25 == 1:
            self.node.get_logger().info(
                f"starting pair {pair_index + 1}/{len(self.pairs)}: "
                f"replicate={self.replicate_index}/{self.repeat_count}, "
                f"steer={self.phase_steering_rad:+.4f} rad, "
                f"{self.throttle_start_norm:.2f}->{self.throttle_end_norm:.2f}")

    def _close_pair(self, *, valid: bool, reason: str = "") -> None:
        if self.phase_started_at is None:
            return
        now = time.monotonic()
        phase_times = self.phase_odom_receipts
        gaps = [b - a for a, b in zip(phase_times, phase_times[1:]) if b > a]
        duration = sum(gaps)
        rate_hz = len(gaps) / duration if duration > 0.0 else None
        sorted_gaps = sorted(gaps)
        gap_p95 = (
            sorted_gaps[math.ceil(0.95 * len(sorted_gaps)) - 1] * 1000.0
            if sorted_gaps else None
        )
        gap_max_ms = sorted_gaps[-1] * 1000.0 if sorted_gaps else None
        stream_ok = (
            rate_hz is not None and rate_hz >= 38.0
            and gap_p95 is not None and gap_p95 <= 35.0
            and gap_max_ms is not None and gap_max_ms <= 60.0
        )
        stream_failures = [] if stream_ok else ["odom_stream_cadence"]
        if stream_failures:
            self.stream_quality_failure_count += 1
            if len(self.stream_quality_failure_examples) < 20:
                self.stream_quality_failure_examples.append(
                    f"{self.phase_label}:odom_stream_cadence")
        end_state = self._current_state()
        start_feedback = self.baseline_state.get("throttle_feedback_norm")
        end_feedback = end_state["throttle_feedback_norm"]
        start_feedback_error = (
            abs(float(start_feedback) - self.throttle_start_norm)
            if start_feedback is not None else None)
        end_feedback_error = abs(end_feedback - self.throttle_end_norm)
        feedback_tracking_ok = (
            start_feedback_error is not None
            and start_feedback_error <= THROTTLE_FEEDBACK_TOLERANCE
            and end_feedback_error <= THROTTLE_FEEDBACK_TOLERANCE)
        steering_feedback_error = (
            abs(end_state["steering_feedback_rad"] - self.phase_steering_rad)
            if self.steering_feedback is not None else None)
        steering_tracking_ok = (
            steering_feedback_error is not None
            and steering_feedback_error <= STEERING_FEEDBACK_TOLERANCE_RAD)
        result = {
            "event": "phase_end",
            "profile": "throttle_transition_surface",
            "seed": self.seed,
            "phase_index": self.phase_index,
            "replicate_index": self.replicate_index,
            "label": self.phase_label,
            "status": "complete" if valid else "incomplete",
            "reason": reason,
            "valid": valid,
            "quality_failures": (stream_failures if valid else
                                 [reason or "incomplete", *stream_failures]),
            "throttle_start_norm": self.throttle_start_norm,
            "throttle_end_norm": self.throttle_end_norm,
            "steering_command_rad": self.phase_steering_rad,
            "baseline_state": self.baseline_state,
            "end_state": end_state,
            "samples": len(phase_times),
            "duration_s": now - self.phase_started_at,
            "baseline_dwell_s": BASELINE_DWELL_S,
            "response_observation_s": self.response_observation_s,
            "measured_speed_max_mps": self.phase_max_speed_mps,
            "measured_tilt_max_deg": math.degrees(self.phase_max_tilt_rad),
            "maximum_displacement_from_spawn_m": self.phase_max_displacement_m,
            "start_throttle_feedback_abs_error_norm": start_feedback_error,
            "end_throttle_feedback_abs_error_norm": end_feedback_error,
            "throttle_feedback_tracking_ok": feedback_tracking_ok,
            "steering_feedback_abs_error_rad": steering_feedback_error,
            "steering_feedback_tracking_ok": steering_tracking_ok,
            "odom_stream_ok": stream_ok,
            "odom_gap_max_ms_during_pair": gap_max_ms,
            "usable_for_command_response_fit": (
                valid and feedback_tracking_ok and steering_tracking_ok
                and stream_ok),
            "odom_rate_hz_during_pair": rate_hz,
            "odom_gap_p95_ms_during_pair": gap_p95,
            "monotonic_ns": time.monotonic_ns(),
        }
        self._publish_event(result)
        self.phase_started_at = None
        self.phase_index = -1
        if valid:
            self.valid_pairs += 1
        else:
            self.invalid_pairs += 1

    def _begin_reset(self, now: float, reason: str, resume_pair_index: int) -> None:
        self.reset_reason = reason
        self.resume_pair_index = resume_pair_index
        self.reset_released_at = None
        self.reset_stable_since = None
        self.state = "reset_hold"
        self.resets_requested += 1
        self.state_started_at = now
        self._command(0.0)
        self._publish_event({
            "event": "sim_reset_start",
            "profile": "throttle_transition_surface",
            "reason": reason,
            "completed_pairs": self.valid_pairs,
            "invalid_pairs": self.invalid_pairs,
            "pairs_attempted": self.pairs_attempted,
            "position_xy": self.position_xy,
            "speed_mps": self.speed_mps,
            "monotonic_ns": time.monotonic_ns(),
        })

    def _reset_state_ready(self, now: float) -> bool:
        state_ready = (
            self.reset_released_at is not None
            and self.last_odom_at is not None
            and self.last_odom_at >= self.reset_released_at
            and self.speed_mps is not None and self.speed_mps <= 0.20
            and self.throttle_feedback is not None
            and abs(self.throttle_feedback) <= 0.02
            and self.steering_feedback is not None
            and abs(self.steering_feedback) <= 0.02
        )
        if not state_ready:
            return False
        if self.spawn_xy is None:
            return self.position_xy is not None
        return (
            self.position_xy is not None
            and math.dist(self.spawn_xy, self.position_xy)
            <= SPAWN_POSITION_TOLERANCE_M
        )

    def _finish(self, reason: str, *, aborted: bool) -> None:
        if self.done:
            return
        if self.phase_started_at is not None:
            self._close_pair(valid=False, reason=reason)
        self.done = True
        self.aborted = aborted
        self.reason = reason
        self._command(0.0)
        self.reset_pub.publish(Bool(data=False))
        self._publish_event({
            "event": "experiment_end",
            "profile": "throttle_transition_surface",
            "seed": self.seed,
            "phase_count": len(self.pairs),
            "completed_pairs": self.valid_pairs,
            "invalid_pairs": self.invalid_pairs,
            "pairs_attempted": self.pairs_attempted,
            "resets_requested": self.resets_requested,
            "resets_recovered": self.resets_recovered,
            "steering_angles_rad": self.steering_angles_rad,
            "design": self._design_metadata(),
            "quality_failures": (
                [] if self.stream_quality_failure_count == 0 else
                [f"odom_stream_cadence in {self.stream_quality_failure_count} pairs; "
                 f"examples={self.stream_quality_failure_examples}"]),
            "reason": reason,
            "aborted": aborted,
            "reset_recovered": self.resets_requested == self.resets_recovered,
            "monotonic_ns": time.monotonic_ns(),
        })
        self.node.get_logger().info(
            f"finished: reason={reason}, aborted={aborted}, "
            f"completed_pairs={self.valid_pairs}, invalid_pairs={self.invalid_pairs}")

    def _tick(self) -> None:
        if self.done:
            return
        now = time.monotonic()
        if now - self.last_heartbeat_at >= 60.0:
            self.last_heartbeat_at = now
            self.node.get_logger().info(
                f"heartbeat: state={self.state}, pair={self.phase_index + 1}/"
                f"{len(self.pairs)}, steer={self.phase_steering_rad:+.4f} rad, "
                f"speed={float(self.speed_mps or 0.0):.2f} m/s, "
                f"throttle_feedback={float(self.throttle_feedback or 0.0):.3f}, "
                f"displacement={self.phase_max_displacement_m:.1f} m, "
                f"resets={self.resets_recovered}/{self.resets_requested}")
        if self.state == "waiting":
            ready = (
                self.last_odom_at is not None
                and self.last_collision_at is not None
                and self.throttle_feedback is not None
                and self.steering_feedback is not None
                and self.throttle_pub.get_subscription_count() > 0
                and self.steering_pub.get_subscription_count() > 0
                and self.reset_pub.get_subscription_count() > 0
            )
            if not ready:
                return
            for topic in (THROTTLE_COMMAND, STEERING_COMMAND):
                if len(self.node.get_publishers_info_by_topic(topic)) > 1:
                    self._finish(f"conflicting command publishers on {topic}",
                                 aborted=True)
                    return
            if self.collision_count != 0:
                self._finish(f"unsafe initial collision count={self.collision_count}",
                             aborted=True)
                return
            self._begin_reset(now, "initial reset to spawn", 0)

        if self.state not in ("reset_hold", "reset_wait"):
            if self.last_odom_at is None or now - self.last_odom_at > ODOM_TIMEOUT_S:
                if self.phase_started_at is not None:
                    failed_index = self.phase_index
                    self._close_pair(valid=False, reason="odometry lost; reset requested")
                    self._begin_reset(now, "odometry lost during transition",
                                      failed_index + 1)
                elif self.state != "waiting":
                    self._begin_reset(now, "odometry lost before transition", self.resume_pair_index)
                return
            if (self.last_collision_at is None or
                    now - self.last_collision_at > COLLISION_TIMEOUT_S):
                self._finish("collision telemetry timeout", aborted=True)
                return

        if (self.phase_started_at is not None and self.tilt_rad is not None
                and self.tilt_rad >= math.radians(60.0)):
            failed_index = self.phase_index
            self._close_pair(valid=False, reason="vehicle rollover during probe")
            self._begin_reset(now, "vehicle rollover during probe", failed_index + 1)
            return

        if self.state == "reset_hold":
            self._command(0.0)
            self.reset_pub.publish(Bool(data=True))
            if now - self.state_started_at >= RESET_HOLD_S:
                self.reset_pub.publish(Bool(data=False))
                self.state = "reset_wait"
                self.state_started_at = now
                self.reset_released_at = now
                self.reset_stable_since = None
                self._publish_event({
                    "event": "sim_reset_release",
                    "profile": "throttle_transition_surface",
                    "reset_command": False,
                    "monotonic_ns": time.monotonic_ns(),
                })
            return

        if self.state == "reset_wait":
            self._command(0.0)
            if now - self.state_started_at > RESET_TIMEOUT_S:
                self._finish("simulator did not return to stable spawn after reset",
                             aborted=True)
                return
            if self._reset_state_ready(now):
                if self.spawn_xy is None and self.position_xy is not None:
                    self.spawn_xy = self.position_xy
                if self.reset_stable_since is None:
                    self.reset_stable_since = now
                elif now - self.reset_stable_since >= 0.50:
                    distance = math.dist(self.spawn_xy, self.position_xy)
                    self._publish_event({
                        "event": "sim_reset_recovered",
                        "profile": "throttle_transition_surface",
                        "reason": self.reset_reason,
                        "position_error_m": distance,
                        "position_xy": self.position_xy,
                        "speed_mps": self.speed_mps,
                        "monotonic_ns": time.monotonic_ns(),
                    })
                    self.resets_recovered += 1
                    self._start_pair(self.resume_pair_index)
                    return
            else:
                self.reset_stable_since = None
            return

        if self.state == "start_command":
            self._command(self.throttle_start_norm, self.phase_steering_rad)
            if now - self.state_started_at >= BASELINE_DWELL_S:
                self.baseline_state = self._current_state()
                self._publish_event({
                    "event": "throttle_slew_stimulus",
                    "profile": "throttle_transition_surface",
                    "phase_index": self.phase_index,
                    "replicate_index": self.replicate_index,
                    "label": self.phase_label,
                    "throttle_profile": "step",
                    "throttle_start_norm": self.throttle_start_norm,
                    "throttle_end_norm": self.throttle_end_norm,
                    "steering_command_rad": self.phase_steering_rad,
                    "baseline_observation_s": BASELINE_DWELL_S,
                    "baseline_state": self.baseline_state,
                    "speed_mps": self.speed_mps,
                    "vx_mps": self.vx_mps,
                    "vy_mps": self.vy_mps,
                    "yaw_rate_rps": self.yaw_rate_rps,
                    "throttle_feedback_norm": self.throttle_feedback,
                    "monotonic_ns": time.monotonic_ns(),
                })
                self.state = "target_command"
                self.state_started_at = now
                self.target_started_at = now
            return

        if self.state == "target_command":
            self._command(self.throttle_end_norm, self.phase_steering_rad)
            if (self.target_started_at is not None
                    and now - self.target_started_at >= RESPONSE_DWELL_S):
                completed_index = self.phase_index
                self.response_observation_s = now - self.target_started_at
                self._close_pair(valid=True)
                self._begin_reset(now, "fixed response observation complete",
                                  completed_index + 1)
                if (completed_index + 1) % 25 == 0:
                    self.node.get_logger().info(
                        f"progress: {completed_index + 1}/{len(self.pairs)} "
                        f"pairs; valid={self.valid_pairs}, invalid={self.invalid_pairs}; "
                        f"last={self.throttle_start_norm:.2f}->"
                        f"{self.throttle_end_norm:.2f}; speed="
                        f"{self.speed_mps:.2f}m/s, "
                        f"displacement={self.phase_max_displacement_m:.1f}m")
            return


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument(
        "--steering-angles-rad",
        default="0.0",
        help="comma-separated signed steering commands in radians; defaults to straight",
    )
    parser.add_argument("--target-step-percent", type=int,
                        default=THROTTLE_TARGET_STEP_PERCENT)
    parser.add_argument("--baseline-step-percent", type=int,
                        default=THROTTLE_BASELINE_STEP_PERCENT)
    parser.add_argument("--repeat-count", type=int,
                        default=THROTTLE_REPEAT_COUNT)
    parser.add_argument(
        "--resume-analysis", type=Path,
        help="schedule only conditions not marked fit-usable in a prior analysis",
    )
    args = parser.parse_args()
    try:
        steering_angles = [float(value.strip())
                           for value in args.steering_angles_rad.split(",")]
    except ValueError:
        parser.error("steering angles must be comma-separated finite radians")
    if (not steering_angles
            or not all(math.isfinite(value) and
                       abs(value) <= STEERING_LIMIT_RAD
                       for value in steering_angles)
            or len({round(value * 1000.0) for value in steering_angles})
            != len(steering_angles)):
        parser.error(
            f"steering angles must be unique, finite, and within +/-{STEERING_LIMIT_RAD} rad")

    if (args.target_step_percent <= 0
            or 100 % args.target_step_percent != 0
            or args.baseline_step_percent <= 0
            or 100 % args.baseline_step_percent != 0
            or args.baseline_step_percent % args.target_step_percent != 0
            or args.repeat_count <= 0):
        parser.error("throttle steps must divide 100%, baseline step must be a "
                     "multiple of target step, and repeat count must be positive")

    all_pairs = ThrottleTransitionExperiment._make_pairs(
        args.seed, steering_angles, args.target_step_percent,
        args.baseline_step_percent, args.repeat_count)
    pairs = all_pairs
    resume_metadata: dict[str, object] = {}
    if args.resume_analysis is not None:
        try:
            pairs, resume_metadata = ThrottleTransitionExperiment._select_resume_pairs(
                all_pairs, args.resume_analysis, args.seed, steering_angles,
                args.target_step_percent, args.baseline_step_percent,
                args.repeat_count)
        except (ValueError, KeyError, TypeError) as exc:
            parser.error(str(exc))

    def request_stop(_signum: int, _frame: object) -> None:
        raise KeyboardInterrupt

    # This process is launched in the background by the container runner, so
    # explicitly replace the inherited SIGINT-ignore disposition.
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    experiment = ThrottleTransitionExperiment(
        args.seed, steering_angles, args.target_step_percent,
        args.baseline_step_percent, args.repeat_count, all_pairs=all_pairs,
        pairs=pairs, resume_metadata=resume_metadata)
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(experiment.node)
    try:
        while rclpy.ok() and not experiment.done:
            executor.spin_once(timeout_sec=0.1)
    except KeyboardInterrupt:
        experiment._finish("interrupted", aborted=True)
    finally:
        executor.shutdown()
        experiment.node.destroy_node()
        rclpy.try_shutdown()
    return 1 if experiment.aborted else 0


if __name__ == "__main__":
    raise SystemExit(main())
