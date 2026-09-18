#!/usr/bin/env python3
"""Blind recursive score of the MPC longitudinal-response candidate.

Only source-stamped /current_map_pose, /odom and /cmd/speed enter the replay.
The simulator packet file is read solely as an offline truth target. The
baseline treats the rate-limited target-speed slew as achieved acceleration;
the candidate uses the fitted AutoDRIVE body-speed response compiled into
f1tenth_mpc/include/mpc_types.h.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any


TOPIC_ODOM = "/odom"
TOPIC_POSE = "/current_map_pose"
TOPIC_COMMAND = "/cmd/speed"
TOPIC_PACKET = "/autodrive/roboracer_1/bridge_packet_timing"
HORIZONS = (4, 5, 10, 20, 30)
ORIGIN_STRIDE = 5
MAX_INPUT_AGE_S = 0.040
MAX_TARGET_SPEED_MPS = 16.0
MAX_TARGET_ACCEL_MPS2 = 3.0
MAX_TARGET_BRAKE_MPS2 = 8.0


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def finite(value: str, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite {label}")
    return number


def payload(row: dict[str, str]) -> dict[str, Any]:
    return json.loads(row["payload_json"])


def source_time_offset(events: list[dict[str, str]]) -> dict[str, float]:
    odom = [row for row in events if row["topic"] == TOPIC_ODOM]
    packets = [row for row in events if row["topic"] == TOPIC_PACKET]
    if len(odom) < 100 or len(odom) != len(packets):
        raise ValueError(
            f"need one source-time packet per /odom event; got {len(packets)} and {len(odom)}")
    offsets: list[float] = []
    max_pair_delay_s = 0.0
    for pose_event, packet_event in zip(odom, packets):
        stamp_ns = int(pose_event["header_stamp_ns"])
        source_s = finite(packet_event["simulation_time_s"], "packet source time")
        offsets.append(source_s - stamp_ns * 1.0e-9)
        pair_delay = abs(
            int(pose_event["arrival_monotonic_ns"]) -
            int(packet_event["arrival_monotonic_ns"])) * 1.0e-9
        max_pair_delay_s = max(max_pair_delay_s, pair_delay)
    offset = statistics.median(offsets)
    spread = max(abs(value - offset) for value in offsets)
    if spread > 2.0e-6 or max_pair_delay_s > 0.100:
        raise ValueError(
            f"source-time join failed: offset spread={spread:.3g}s, "
            f"max paired callback separation={max_pair_delay_s:.3g}s")
    return {
        "source_time_offset_s": offset,
        "paired_packets": len(offsets),
        "offset_max_abs_residual_s": spread,
        "max_paired_callback_separation_s": max_pair_delay_s,
    }


def source_events(
    events: list[dict[str, str]], offset_s: float, topic: str,
) -> tuple[list[float], list[dict[str, Any]]]:
    selected: list[tuple[float, dict[str, Any]]] = []
    for event in events:
        if event["topic"] != topic or not event["header_stamp_ns"]:
            continue
        stamp = int(event["header_stamp_ns"]) * 1.0e-9 + offset_s
        selected.append((stamp, payload(event)))
    selected.sort(key=lambda value: value[0])
    times = [item[0] for item in selected]
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError(f"{topic} source stamps are not strictly increasing")
    return times, [item[1] for item in selected]


def parse_model_constants(header: Path) -> dict[str, float]:
    source = header.read_text(encoding="utf-8")
    names = (
        "MPC_YAW_RATE_RESPONSE_TIME_CONSTANT_SECONDS",
        "MPC_YAW_RATE_STEERING_GAIN_PER_M",
        "MPC_LONGITUDINAL_RESPONSE_BIAS_MPS2",
        "MPC_LONGITUDINAL_SPEED_COEFF_PER_S",
        "MPC_LONGITUDINAL_TARGET_ERROR_GAIN_PER_S",
        "MPC_LONGITUDINAL_TARGET_RATE_COEFF",
        "MPC_LONGITUDINAL_ACCEL_LIMIT_MPS2",
        "MPC_LONGITUDINAL_BRAKE_DECEL_INTERCEPT_MPS2",
        "MPC_LONGITUDINAL_BRAKE_DECEL_SLOPE_S_INV",
    )
    result: dict[str, float] = {}
    for name in names:
        match = re.search(
            rf"^#define\s+{name}\s+\(?\s*(-?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)f?\s*\)?",
            source, re.MULTILINE)
        if not match:
            raise ValueError(f"could not read {name} from {header}")
        result[name] = float(match.group(1))
    return result


def nearest_payload(
    times: list[float], rows: list[dict[str, Any]], query_s: float,
    label: str, *, strict_before: bool = False,
) -> tuple[dict[str, Any], float]:
    index = (bisect.bisect_left(times, query_s) if strict_before else
             bisect.bisect_right(times, query_s)) - 1
    if index < 0:
        raise ValueError(f"no prior {label} at source time {query_s:.6f}")
    age = query_s - times[index]
    if age < -1.0e-7 or age > MAX_INPUT_AGE_S:
        raise ValueError(f"stale {label}: age={age:.6f}s")
    return rows[index], age


def clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def next_state(
    state: dict[str, float], command: dict[str, Any], previous_target: float,
    dt: float, constants: dict[str, float], response_model: bool,
) -> dict[str, float]:
    target_command = clamp(float(command["speed_mps"]), 0.0, MAX_TARGET_SPEED_MPS)
    target_rate = clamp(
        (target_command - previous_target) / dt,
        -MAX_TARGET_BRAKE_MPS2, MAX_TARGET_ACCEL_MPS2)
    target_next = clamp(
        previous_target + target_rate * dt, 0.0, MAX_TARGET_SPEED_MPS)
    if response_model:
        target_mid = 0.5 * (previous_target + target_next)
        acceleration = (
            constants["MPC_LONGITUDINAL_RESPONSE_BIAS_MPS2"] +
            constants["MPC_LONGITUDINAL_SPEED_COEFF_PER_S"] * state["u"] +
            constants["MPC_LONGITUDINAL_TARGET_ERROR_GAIN_PER_S"] *
            (target_mid - state["u"]) +
            constants["MPC_LONGITUDINAL_TARGET_RATE_COEFF"] * target_rate)
        brake_limit = (
            constants["MPC_LONGITUDINAL_BRAKE_DECEL_INTERCEPT_MPS2"] +
            constants["MPC_LONGITUDINAL_BRAKE_DECEL_SLOPE_S_INV"] * state["u"])
        acceleration = clamp(
            acceleration, -brake_limit,
            constants["MPC_LONGITUDINAL_ACCEL_LIMIT_MPS2"])
        u_next = clamp(state["u"] + acceleration * dt, 0.0, MAX_TARGET_SPEED_MPS)
    else:
        # Pre-candidate assumption: requested target slew is achieved instantly.
        u_next = clamp(state["u"] + target_rate * dt, 0.0, MAX_TARGET_SPEED_MPS)

    steering = clamp(float(command["steering_angle_rad"]), -math.pi / 6.0, math.pi / 6.0)
    u_mid = 0.5 * (state["u"] + u_next)
    retention = math.exp(
        -dt / constants["MPC_YAW_RATE_RESPONSE_TIME_CONSTANT_SECONDS"])
    steady_yaw_rate = (
        u_mid * math.tan(steering) *
        constants["MPC_YAW_RATE_STEERING_GAIN_PER_M"])
    yaw_rate_next = retention * state["r"] + (1.0 - retention) * steady_yaw_rate
    r_mid = 0.5 * (state["r"] + yaw_rate_next)
    heading_mid = state["yaw"] + 0.5 * dt * r_mid
    lateral_velocity = state["v"]

    return {
        "x": state["x"] + dt * (
            u_mid * math.cos(heading_mid) - lateral_velocity * math.sin(heading_mid)),
        "y": state["y"] + dt * (
            u_mid * math.sin(heading_mid) + lateral_velocity * math.cos(heading_mid)),
        "yaw": wrap(state["yaw"] + dt * r_mid),
        "u": u_next,
        "v": lateral_velocity,
        "r": yaw_rate_next,
        "target": target_next,
    }


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def metric(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"samples": 0, "signed_mean": None, "mae": None,
                "abs_p95": None, "rmse": None}
    return {
        "samples": len(values),
        "signed_mean": statistics.fmean(values),
        "mae": statistics.fmean(abs(value) for value in values),
        "abs_p95": percentile([abs(value) for value in values], 0.95),
        "rmse": math.sqrt(statistics.fmean(value * value for value in values)),
    }


def score(
    trace_dir: Path, header: Path, *, split_name: str,
) -> dict[str, Any]:
    events = read_csv(trace_dir / "events.csv")
    packets = read_csv(trace_dir / "simulator_packets.csv")
    if len(packets) < 100:
        raise ValueError(f"insufficient packet truth in {trace_dir}")
    alignment = source_time_offset(events)
    offset = alignment["source_time_offset_s"]
    odom_t, odom = source_events(events, offset, TOPIC_ODOM)
    pose_t, pose = source_events(events, offset, TOPIC_POSE)
    command_t, commands = source_events(events, offset, TOPIC_COMMAND)
    constants = parse_model_constants(header)
    packet_times = [finite(row["simulation_time_s"], "packet source time") for row in packets]
    if any(b <= a for a, b in zip(packet_times, packet_times[1:])):
        raise ValueError("packet source-time sequence is not strictly increasing")
    dts = [b - a for a, b in zip(packet_times, packet_times[1:])]
    if max(dts) > 0.100:
        raise ValueError("source packet gap above 100 ms; refusing recursive score")

    scores: dict[str, dict[str, dict[str, list[float]]]] = {
        name: {
            str(horizon): {
                key: [] for key in (
                    "longitudinal_m", "lateral_m", "euclidean_m", "yaw_rad",
                    "body_speed_mps", "horizon_duration_s")
            }
            for horizon in HORIZONS
        }
        for name in ("slew_as_acceleration_baseline", "identified_speed_response")
    }
    stale_rejections = 0
    score_origins: dict[str, int] = {str(horizon): 0 for horizon in HORIZONS}
    final_origin = len(packets) - max(HORIZONS) - 1
    for origin in range(0, final_origin + 1, ORIGIN_STRIDE):
        t0 = packet_times[origin]
        try:
            state_odom, _ = nearest_payload(odom_t, odom, t0, "odom")
            state_pose, _ = nearest_payload(pose_t, pose, t0, "current_map_pose")
            prior_command, _ = nearest_payload(
                command_t, commands, t0, "prior speed command", strict_before=True)
        except ValueError:
            stale_rejections += 1
            continue

        initial = {
            "x": float(state_pose["x_m"]),
            "y": float(state_pose["y_m"]),
            "yaw": float(state_pose["yaw_rad"]),
            "u": max(0.0, float(state_odom["speed_mps"])),
            "v": float(state_odom["lateral_speed_mps"]),
            "r": float(state_odom["yaw_rate_radps"]),
            "target": clamp(
                float(prior_command["speed_mps"]), 0.0, MAX_TARGET_SPEED_MPS),
        }
        predictions = {
            "slew_as_acceleration_baseline": dict(initial),
            "identified_speed_response": dict(initial),
        }
        targets = {name: initial["target"] for name in predictions}
        valid_horizons = set(HORIZONS)
        for step in range(max(HORIZONS)):
            packet_index = origin + step
            interval_start = packet_times[packet_index]
            interval_end = packet_times[packet_index + 1]
            dt = interval_end - interval_start
            command_index = bisect.bisect_right(command_t, interval_start) - 1
            if command_index < 0 or interval_start - command_t[command_index] > MAX_INPUT_AGE_S:
                valid_horizons.clear()
                break
            command = commands[command_index]
            for model_name, is_candidate in (
                ("slew_as_acceleration_baseline", False),
                ("identified_speed_response", True),
            ):
                predictions[model_name] = next_state(
                    predictions[model_name], command, targets[model_name], dt,
                    constants, is_candidate)
                targets[model_name] = predictions[model_name]["target"]

            horizon = step + 1
            if horizon not in HORIZONS:
                continue
            target_packet = packets[origin + horizon]
            target_x = finite(target_packet["simulator_position_x"], "truth x")
            target_y = finite(target_packet["simulator_position_y"], "truth y")
            qx = finite(target_packet["simulator_orientation_quaternion_x"], "truth qx")
            qy = finite(target_packet["simulator_orientation_quaternion_y"], "truth qy")
            qz = finite(target_packet["simulator_orientation_quaternion_z"], "truth qz")
            qw = finite(target_packet["simulator_orientation_quaternion_w"], "truth qw")
            truth_yaw = math.atan2(
                2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
            for model_name, prediction in predictions.items():
                dx, dy = prediction["x"] - target_x, prediction["y"] - target_y
                along = math.cos(truth_yaw) * dx + math.sin(truth_yaw) * dy
                lateral = -math.sin(truth_yaw) * dx + math.cos(truth_yaw) * dy
                values = scores[model_name][str(horizon)]
                values["longitudinal_m"].append(along)
                values["lateral_m"].append(lateral)
                values["euclidean_m"].append(math.hypot(dx, dy))
                values["yaw_rad"].append(wrap(prediction["yaw"] - truth_yaw))
                values["body_speed_mps"].append(
                    prediction["u"] - finite(target_packet["simulator_linear_velocity_x"], "truth u"))
                values["horizon_duration_s"].append(
                    sum(dts[origin:origin + horizon]))
            score_origins[str(horizon)] += 1

    report: dict[str, Any] = {
        "split": split_name,
        "trace": str(trace_dir),
        "model_constants_from_runtime_header": constants,
        "source_alignment": alignment,
        "source_dt_s": {
            "median": statistics.median(dts),
            "p95": percentile(dts, 0.95),
            "max": max(dts),
        },
        "origin_stride_packets": ORIGIN_STRIDE,
        "stale_or_unavailable_origins_rejected": stale_rejections,
        "truth_usage": "offline target only; no future truth enters recursive rollout",
        "runtime_inputs_used": ["/current_map_pose", "/odom", "source-stamped /cmd/speed"],
        "horizons": {},
    }
    for horizon in HORIZONS:
        key = str(horizon)
        report["horizons"][key] = {
            "steps": horizon,
            "samples_each_model": score_origins[key],
            "duration_s_mean": statistics.fmean(
                scores["identified_speed_response"][key]["horizon_duration_s"])
                if score_origins[key] else None,
            "models": {
                name: {
                    metric_name: metric(values)
                    for metric_name, values in metric_values.items()
                    if metric_name != "horizon_duration_s"
                }
                for name, model_values in scores.items()
                for metric_values in (model_values[key],)
            },
        }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--header", type=Path, default=Path(
        "f1tenth_mpc/include/mpc_types.h"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "score_contract": {
            "primary": "recursive truth-heading longitudinal/lateral/yaw prediction at N5",
            "diagnostics": list(HORIZONS),
            "euclidean_position": "secondary summary only",
            "development": "coefficient-development capture; no refitting is done by this scorer",
            "holdout": "untouched capture used as the promotion/rejection score",
        },
        "development": score(args.development, args.header, split_name="development"),
        "holdout": score(args.holdout, args.header, split_name="holdout"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
