#!/usr/bin/env python3
"""Fit and score the dev-side source-time MPC dynamic observer.

The observer receives only signals that are available on the dev branch:
source-stamped odometry speed, IMU gyro/lateral acceleration, steering command
and applied throttle. Simulator packet velocity is loaded in a separate
offline scoring path and is never passed to the observer update.

This tool is intentionally an offline identification/validation tool. It does
not publish ROS messages, edit simulator files, or promote the observer to
runtime authority.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np


TOPICS = {
    "imu": "/autodrive/roboracer_1/imu",
    "odom": "/odom",
}
FEATURE_NAMES = ("u_mps", "r_radps", "steering_rad", "ay_mps2", "bias")


def _number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _nearest(rows: list[tuple[float, dict[str, object]]], stamp: float) -> dict[str, object] | None:
    if not rows:
        return None
    row = min(rows, key=lambda item: abs(item[0] - stamp))
    return row[1] if abs(row[0] - stamp) <= 0.0015 else None


def _load_run(path: Path) -> list[dict[str, float]]:
    with (path / "bridge_requests.csv").open(newline="", encoding="utf-8") as stream:
        packets = list(csv.DictReader(stream))
    topic_rows: dict[str, list[tuple[float, dict[str, object]]]] = {}
    with (path / "events.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            topic = row.get("topic", "")
            if topic not in TOPICS.values():
                continue
            stamp_ns = _number(row.get("header_stamp_ns"))
            if stamp_ns is None:
                continue
            try:
                payload = json.loads(row["payload_json"])
            except (KeyError, json.JSONDecodeError):
                continue
            topic_rows.setdefault(topic, []).append((stamp_ns / 1.0e9, payload))
    for rows in topic_rows.values():
        rows.sort()
    imu_rows = topic_rows.get(TOPICS["imu"], [])
    if not packets or not imu_rows:
        raise ValueError(f"{path} lacks bridge packets or source-stamped IMU")
    first_source = _number(packets[0].get("simulation_time_s"))
    if first_source is None:
        raise ValueError(f"{path} has no finite source time")
    source_origin = imu_rows[0][0] - first_source

    samples: list[dict[str, float]] = []
    for packet in packets:
        source_time = _number(packet.get("simulation_time_s"))
        truth_u = _number(packet.get("simulator_linear_velocity_x"))
        truth_v = _number(packet.get("simulator_linear_velocity_y"))
        truth_r = _number(packet.get("simulator_angular_velocity_z"))
        if None in (source_time, truth_u, truth_v, truth_r):
            continue
        stamp = source_origin + source_time
        imu = _nearest(imu_rows, stamp)
        odom = _nearest(topic_rows.get(TOPICS["odom"], []), stamp)
        if imu is None or odom is None:
            continue
        values = {
            "stamp_s": stamp,
            "u_mps": _number(odom.get("speed_mps")),
            "r_radps": _number(imu.get("yaw_rate_radps")),
            "ax_mps2": _number(imu.get("ax_mps2")),
            "ay_mps2": _number(imu.get("ay_mps2")),
            # /cmd/speed is controller-time stamped rather than source-time
            # stamped. Use the source-associated applied steering command
            # from bridge timing until a typed feedback message is added.
            "steering_rad": (
                _number(packet.get("applied_steering_norm")) *
                0.5235987756
            ) if _number(packet.get("applied_steering_norm")) is not None else None,
            # The public throttle Float32 has no header. The bridge timing
            # record carries the source-associated applied command and is the
            # legal runtime feedback boundary used by the observer adapter.
            "applied_throttle_norm": _number(packet.get("applied_throttle_norm")),
            "truth_u_mps": truth_u,
            "truth_v_mps": truth_v,
            "truth_r_radps": truth_r,
        }
        if any(value is None for value in values.values()):
            continue
        samples.append({key: float(value) for key, value in values.items()})
    if len(samples) < 10:
        raise ValueError(f"{path} produced too few aligned observer samples")
    return samples


def _features(sample: dict[str, float]) -> list[float]:
    return [
        sample["u_mps"], sample["r_radps"], sample["steering_rad"],
        sample["ay_mps2"], 1.0,
    ]


def _observer(samples: list[dict[str, float]], coefficients: np.ndarray) -> dict[str, object]:
    v_hat = 0.0
    u_hat = None
    outputs: list[dict[str, float]] = []
    previous_stamp = None
    for sample in samples:
        target_v = float(np.dot(_features(sample), coefficients))
        target_v = max(-1.5, min(1.5, target_v))
        if previous_stamp is None:
            u_hat = sample["u_mps"]
            v_hat = target_v
            dt = 0.0
        else:
            dt = sample["stamp_s"] - previous_stamp
            if 0.015 <= dt <= 0.035:
                assert u_hat is not None
                u_prediction = u_hat + dt * sample["ax_mps2"]
                u_hat = max(0.0, min(20.0,
                    u_prediction + 0.20 * (sample["u_mps"] - u_prediction)))
                alpha = 1.0 - math.exp(-dt / 0.010)
                v_hat += alpha * (target_v - v_hat)
        previous_stamp = sample["stamp_s"]
        outputs.append({
            "u_hat_mps": u_hat if u_hat is not None else sample["u_mps"],
            "v_hat_mps": v_hat,
            "r_hat_radps": sample["r_radps"],
            "dt_s": dt,
        })

    def metric(key: str, truth_key: str) -> dict[str, float | None]:
        errors = [row[key] - sample[truth_key] for row, sample in zip(outputs, samples)]
        absolute = np.abs(np.asarray(errors, dtype=float))
        return {
            "mae": float(np.mean(absolute)),
            "rmse": float(np.sqrt(np.mean(np.square(errors)))),
            "p50_abs": float(np.quantile(absolute, 0.50)),
            "p95_abs": float(np.quantile(absolute, 0.95)),
            "max_abs": float(np.max(absolute)),
        }

    return {
        "samples": len(samples),
        "source_dt_s": {
            "min": min(row["dt_s"] for row in outputs[1:]),
            "p95": float(np.quantile([row["dt_s"] for row in outputs[1:]], 0.95)),
            "max": max(row["dt_s"] for row in outputs[1:]),
        },
        "u": metric("u_hat_mps", "truth_u_mps"),
        "v": metric("v_hat_mps", "truth_v_mps"),
        "r": metric("r_hat_radps", "truth_r_radps"),
    }


def run(
    root: Path,
    train_names: list[str],
    validation_names: list[str],
    output_json: Path | None,
) -> dict[str, object]:
    train_runs = {name: _load_run(root / name) for name in train_names}
    validation_runs = {name: _load_run(root / name) for name in validation_names}
    train_matrix = np.asarray([
        _features(sample) for samples in train_runs.values() for sample in samples
    ], dtype=float)
    train_target = np.asarray([
        sample["truth_v_mps"] for samples in train_runs.values() for sample in samples
    ], dtype=float)
    coefficients, _, _, _ = np.linalg.lstsq(train_matrix, train_target, rcond=None)
    report: dict[str, object] = {
        "status": "offline_mpc_observer_prototype",
        "runtime_truth_input": False,
        "simulator_behavior_modified": False,
        "observer_state": ["u", "v", "r", "delta", "q_drive"],
        "legal_runtime_inputs": [
            "source-stamped /odom speed",
            "source-stamped IMU yaw rate and lateral acceleration",
            "source-associated applied steering command (typed feedback adapter pending)",
            "applied throttle",
        ],
        "training_runs": train_names,
        "validation_runs": validation_names,
        "fit_features": FEATURE_NAMES,
        "v_coefficients": {
            name: float(value) for name, value in zip(FEATURE_NAMES, coefficients)
        },
        "validation": {
            name: _observer(samples, coefficients)
            for name, samples in validation_runs.items()
        },
        "training": {
            name: _observer(samples, coefficients)
            for name, samples in train_runs.items()
        },
        "promotion": {
            "u_accepted": False,
            "v_accepted": False,
            "r_accepted": False,
            "reason": "single track family and no open-scene cross-validation; observer remains prototype",
        },
    }
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--train", nargs="+", required=True)
    parser.add_argument("--validation", nargs="+", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run(args.root, args.train, args.validation, args.output_json), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
