#!/usr/bin/env python3
"""Prepare and score a deterministic odometry observer replay.

The input stream contains only the legal sensor topics recorded by the dev
stack. Simulator packet fields are used only by the scoring command and are
never written into the observer input. The tool is intentionally offline: it
does not publish ROS messages, edit the simulator, or promote an estimator.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import statistics
from pathlib import Path


def _load_diagnostics(run_dir: Path) -> list[list[float]]:
    values: list[list[float]] = []
    with (run_dir / "events.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row.get("topic") != "/odom/diagnostics":
                continue
            values.append(json.loads(row["payload_json"])["data"])
    return values


def _load_sensor_maps(run_dir: Path) -> tuple[
        dict[int, dict[str, float]], dict[int, float], dict[int, float]]:
    imu: dict[int, dict[str, float]] = {}
    left: dict[int, float] = {}
    right: dict[int, float] = {}
    with (run_dir / "imu.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            imu[int(row["header_stamp_ns"])] = json.loads(row["payload_json"])
    with (run_dir / "encoders.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            stamp = int(row["header_stamp_ns"])
            value = float(json.loads(row["payload_json"])["position"][0])
            if "left_encoder" in row["topic"]:
                left[stamp] = value
            else:
                right[stamp] = value
    return imu, left, right


def prepare_input(run_dir: Path, output: Path) -> int:
    diagnostics = _load_diagnostics(run_dir)
    imu, left, right = _load_sensor_maps(run_dir)
    stamps = sorted(imu)

    def nearest(stamp_s: float) -> int:
        target = stamp_s * 1.0e9
        index = bisect.bisect_left(stamps, target)
        candidates = []
        if index < len(stamps):
            candidates.append(stamps[index])
        if index:
            candidates.append(stamps[index - 1])
        return min(candidates, key=lambda value: abs(value - target))

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "stamp_s", "left_angle_rad", "right_angle_rad", "ax_mps2",
            "ay_mps2", "yaw_rate_radps", "yaw_rad",
        ])
        for data in diagnostics:
            stamp = nearest(float(data[1]))
            message = imu[stamp]
            writer.writerow([
                data[1], left[stamp], right[stamp], message["ax_mps2"],
                message["ay_mps2"], message["yaw_rate_radps"], data[20],
            ])
    return len(diagnostics)


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def _metrics(
    packets: list[dict[str, str]],
    diagnostics: list[list[float]],
    estimates: list[dict[str, str]],
) -> dict[str, object]:
    if not (len(packets) == len(diagnostics) == len(estimates)):
        raise ValueError("packet, diagnostics, and replay lengths differ")
    x0 = float(packets[0]["simulator_position_x"])
    y0 = float(packets[0]["simulator_position_y"])
    yaw0 = _wrap(float(packets[0]["simulator_orientation_euler_z"]) - 2.0 * math.pi)
    position_errors: list[float] = []
    yaw_errors: list[float] = []
    lateral_errors: list[float] = []
    longitudinal_errors: list[float] = []
    speed_bins: dict[int, list[float]] = {}
    for packet, diagnostic, estimate in zip(packets, diagnostics, estimates):
        local_x = float(estimate["x_m"])
        local_y = float(estimate["y_m"])
        cosine = math.cos(yaw0)
        sine = math.sin(yaw0)
        world_x = x0 + cosine * local_x - sine * local_y
        world_y = y0 + sine * local_x + cosine * local_y
        truth_x = float(packet["simulator_position_x"])
        truth_y = float(packet["simulator_position_y"])
        position_error = math.hypot(world_x - truth_x, world_y - truth_y)
        position_errors.append(position_error)
        estimate_yaw = _wrap(yaw0 + float(
            estimate.get("yaw_rad", diagnostic[20])))
        truth_yaw = _wrap(
            float(packet["simulator_orientation_euler_z"]) - 2.0 * math.pi)
        yaw_errors.append(abs(_wrap(estimate_yaw - truth_yaw)))
        truth_u = float(packet["simulator_linear_velocity_x"])
        truth_v = float(packet["simulator_linear_velocity_y"])
        longitudinal_errors.append(float(estimate["body_u_mps"]) - truth_u)
        lateral_errors.append(float(estimate["body_v_mps"]) - truth_v)
        speed = math.hypot(truth_u, truth_v)
        speed_bins.setdefault(min(7, int(speed)), []).append(position_error)
    return {
        "n": len(position_errors),
        "position_median_m": statistics.median(position_errors),
        "position_p95_m": _p95(position_errors),
        "position_max_m": max(position_errors),
        "yaw_p95_rad": _p95(yaw_errors),
        "body_u_bias_mps": statistics.mean(longitudinal_errors),
        "body_v_bias_mps": statistics.mean(lateral_errors),
        "body_v_mae_mps": statistics.mean(abs(value) for value in lateral_errors),
        "body_v_p95_abs_mps": _p95([abs(value) for value in lateral_errors]),
        "position_p95_by_speed_bin_m": {
            str(key): _p95(values) for key, values in sorted(speed_bins.items())
        },
    }


def score_replay(run_dir: Path, replay_csv: Path, output_json: Path) -> None:
    packets = list(csv.DictReader((run_dir / "simulator_packets.csv").open(
        newline="", encoding="utf-8")))
    diagnostics = _load_diagnostics(run_dir)
    replay = list(csv.DictReader(replay_csv.open(newline="", encoding="utf-8")))
    baseline = [
        {"x_m": data[18], "y_m": data[19], "body_u_mps": data[7],
         "body_v_mps": data[8]}
        for data in diagnostics
    ]
    report = {
        "status": "offline_replay_only",
        "simulator_behavior_modified": False,
        "candidate_model": "lateral_velocity_yaw_rate_model_v1",
        "fit_source": (
            "four prior clean track replays plus fresh live baseline; "
            "truth only for offline scoring"
        ),
        "baseline": _metrics(packets, diagnostics, baseline),
        "candidate": _metrics(packets, diagnostics, replay),
        "candidate_runtime_inputs": ["mapped wheel speed", "IMU yaw rate"],
        "candidate_parameters": {
            "lateral_velocity_yaw_rate_gain_m": 0.167,
            "lateral_velocity_speed_yaw_rate_gain_s": -0.0063,
            "lateral_velocity_max_mps": 0.35,
        },
        "promotion": (
            "requires fresh live /odom EKF AMCL validation before MPC or "
            "final odom promotion"
        ),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("run_dir", type=Path)
    prepare.add_argument("output", type=Path)
    score = subparsers.add_parser("score")
    score.add_argument("run_dir", type=Path)
    score.add_argument("replay_csv", type=Path)
    score.add_argument("output_json", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        print(prepare_input(args.run_dir, args.output))
    else:
        score_replay(args.run_dir, args.replay_csv, args.output_json)


if __name__ == "__main__":
    main()
