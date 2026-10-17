#!/usr/bin/env python3
"""Replay packet-aligned actuator feedback through the production PhysX plant.

This is an offline diagnostic: it compares the plant's predicted rear-wheel
slip and wheel/body speed gap with encoder-versus-ground-truth error in an
existing production-Odom replay CSV. Ground truth is read only for scoring.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INITIAL = ROOT / "f1tenth_localization/config/physx_odom_reset_initial.json"
DEFAULT_SHAPES = ROOT / "f1tenth_localization/config/physx_odom_vehicle_shapes.json"
PACKET_S = 0.025
PHYSICS_DT_S = 0.001
PHYSICS_STEPS_PER_PACKET = 25
WHEELBASE_M = 0.324
TRACK_M = 0.236
WHEEL_RADIUS_M = 0.059
MOTOR_TORQUE_NM = 428.0


def ackermann(center_angle_deg: float) -> tuple[float, float]:
    tangent = math.tan(math.radians(center_angle_deg))
    numerator = 2.0 * WHEELBASE_M * tangent
    left = math.atan(numerator / (2.0 * WHEELBASE_M + TRACK_M * tangent))
    right = math.atan(numerator / (2.0 * WHEELBASE_M - TRACK_M * tangent))
    return left, right


def rotate_world_velocity_to_body(
        quaternion_xyzw: list[float], vector_xyz: list[float]) -> np.ndarray:
    qv = np.asarray(quaternion_xyzw[:3], dtype=np.float64)
    vector = np.asarray(vector_xyz, dtype=np.float64)
    w = float(quaternion_xyzw[3])
    return vector - 2.0 * w * np.cross(qv, vector) + 2.0 * np.cross(
        qv, np.cross(qv, vector))


def _step_request(
        process: subprocess.Popen[str], index: int, center_angle_deg: float,
        throttle: float) -> dict[str, Any]:
    assert process.stdin is not None and process.stdout is not None
    front_left, front_right = ackermann(center_angle_deg)
    drive_torque = 0.0 if throttle == 0.0 else MOTOR_TORQUE_NM * throttle / 4.0
    brake_torque = MOTOR_TORQUE_NM if throttle == 0.0 else 0.0
    wheel_commands = [
        {"steer_angle_rad": front_left,
         "drive_torque_nm": drive_torque, "brake_torque_nm": brake_torque},
        {"steer_angle_rad": front_right,
         "drive_torque_nm": drive_torque, "brake_torque_nm": brake_torque},
        {"steer_angle_rad": 0.0,
         "drive_torque_nm": drive_torque, "brake_torque_nm": brake_torque},
        {"steer_angle_rad": 0.0,
         "drive_torque_nm": drive_torque, "brake_torque_nm": brake_torque},
    ]
    request = {
        "op": "step",
        "capture_row": index,
        "time_s": index * PHYSICS_DT_S,
        "wheel_commands": wheel_commands,
    }
    process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
    process.stdin.flush()
    line = process.stdout.readline()
    if not line:
        raise RuntimeError(f"PhysX server stopped at physics step {index}")
    return json.loads(line)


def _correlation(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 3 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _summary(values: np.ndarray) -> dict[str, float | int]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {"count": 0}
    return {
        "count": int(len(finite)),
        "median": float(np.median(finite)),
        "p90": float(np.percentile(finite, 90)),
        "p95": float(np.percentile(finite, 95)),
        "max": float(np.max(finite)),
    }


def replay(csv_path: Path, binary: Path, initial: Path,
           shapes: Path, output_csv: Path) -> dict[str, Any]:
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    required = {
        "source_stamp_s", "steering_feedback_rad", "throttle_feedback",
        "truth_u_mps", "wheel_mapped_mps", "wheel_burst_rejected",
        "wheel_update_used",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"input replay CSV must include {sorted(required)}")

    with tempfile.TemporaryDirectory(prefix="physx_wheel_slip_") as temp_dir:
        initial_path = Path(temp_dir) / "initial.json"
        initial_data = json.loads(initial.read_text(encoding="utf-8"))
        initial_data["mode"] = "interactive_recursive_server"
        initial_path.write_text(json.dumps(initial_data, separators=(",", ":")),
                                encoding="utf-8")
        process = subprocess.Popen(
            [str(binary.resolve()), "--server", str(initial_path),
             str(shapes.resolve())],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None,
            text=True, bufsize=1)
        try:
            # Match the production runtime's reset-state settling interval.
            for index in range(1000):
                _step_request(process, index, 0.0, 0.0)

            center_angle_deg = 0.0
            predicted: list[dict[str, float]] = []
            for packet_index, row in enumerate(rows):
                target_angle_deg = -math.degrees(
                    float(row["steering_feedback_rad"]))
                throttle = float(row["throttle_feedback"])
                state: dict[str, Any] | None = None
                for substep in range(PHYSICS_STEPS_PER_PACKET):
                    fraction = (substep + 1) / PHYSICS_STEPS_PER_PACKET
                    angle = center_angle_deg + fraction * (
                        target_angle_deg - center_angle_deg)
                    physics_index = 1000 + packet_index * PHYSICS_STEPS_PER_PACKET + substep
                    state = _step_request(process, physics_index, angle, throttle)
                center_angle_deg = target_angle_deg
                assert state is not None
                body_velocity = rotate_world_velocity_to_body(
                    state["rotation_world_xyzw"],
                    state["linear_velocity_world_mps"])
                wheels = state["wheels"]
                rear_wheel_speed = np.asarray([
                    float(wheels[2]["omega_radps"]) * WHEEL_RADIUS_M,
                    float(wheels[3]["omega_radps"]) * WHEEL_RADIUS_M,
                ])
                rear_slip = np.asarray(state["wheel_longitudinal_slip"][2:4],
                                       dtype=np.float64)
                predicted.append({
                    "source_stamp_s": float(row["source_stamp_s"]),
                    "predicted_body_u_mps": float(body_velocity[2]),
                    "predicted_body_v_mps": float(-body_velocity[0]),
                    "predicted_rear_wheel_speed_mps": float(np.mean(rear_wheel_speed)),
                    "predicted_rear_wheel_body_gap_mps": float(
                        np.mean(rear_wheel_speed) - body_velocity[2]),
                    "predicted_rear_max_abs_longitudinal_slip": float(
                        np.max(np.abs(rear_slip))),
                    "truth_u_mps": float(row["truth_u_mps"]),
                    "encoder_minus_truth_mps": (
                        float(row["wheel_mapped_mps"]) - float(row["truth_u_mps"])),
                    "wheel_update_used": float(row["wheel_update_used"]),
                    "wheel_burst_rejected": float(row["wheel_burst_rejected"]),
                    "throttle_feedback": throttle,
                })

            assert process.stdin is not None
            process.stdin.write('{"op":"shutdown"}\n')
            process.stdin.flush()
            process.stdin.close()
            return_code = process.wait(timeout=30)
            if return_code != 0:
                raise RuntimeError(f"PhysX server exited with status {return_code}")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    columns = list(predicted[0])
    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(predicted)

    frame = {key: np.asarray([row[key] for row in predicted], dtype=np.float64)
             for key in columns}
    actual_gap = np.abs(frame["encoder_minus_truth_mps"])
    model_gap = np.abs(frame["predicted_rear_wheel_body_gap_mps"])
    model_slip = frame["predicted_rear_max_abs_longitudinal_slip"]
    report: dict[str, Any] = {
        "input_csv": str(csv_path.resolve()),
        "output_csv": str(output_csv.resolve()),
        "samples": len(rows),
        "physics_substeps_per_packet": PHYSICS_STEPS_PER_PACKET,
        "reset_settle_substeps": 1000,
        "predicted_body_u_rmse_mps": float(np.sqrt(np.mean(
            (frame["predicted_body_u_mps"] - frame["truth_u_mps"]) ** 2))),
        "predicted_rear_wheel_body_gap_abs_mps": _summary(model_gap),
        "predicted_rear_max_abs_longitudinal_slip": _summary(model_slip),
        "encoder_truth_gap_by_prediction_lag": {},
        "burst_only_encoder_truth_gap_by_prediction_lag": {},
        "accepted_wheel_encoder_truth_gap_by_prediction_lag": {},
        "throttle_fall_events": {},
    }
    masks = {
        "all": np.ones(len(rows), dtype=bool),
        "burst_only": frame["wheel_burst_rejected"] > 0.5,
        "accepted_wheel": frame["wheel_update_used"] > 0.5,
    }
    for label, mask in masks.items():
        target = report[{
            "all": "encoder_truth_gap_by_prediction_lag",
            "burst_only": "burst_only_encoder_truth_gap_by_prediction_lag",
            "accepted_wheel": "accepted_wheel_encoder_truth_gap_by_prediction_lag",
        }[label]]
        for lag in (0, 1, 2):
            indices = np.flatnonzero(mask)
            indices = indices[indices + lag < len(rows)]
            pred_gap = model_gap[indices]
            pred_slip = model_slip[indices]
            actual = actual_gap[indices + lag]
            target[str(lag)] = {
                "samples": int(len(indices)),
                "wheel_body_gap_abs_pearson": _correlation(pred_gap, actual),
                "tire_slip_abs_pearson": _correlation(pred_slip, actual),
                "wheel_body_gap_abs_median_mps": float(np.median(pred_gap))
                if len(indices) else None,
                "encoder_truth_gap_abs_median_mps": float(np.median(actual))
                if len(indices) else None,
            }

    throttle = frame["throttle_feedback"]
    changes = np.diff(throttle, prepend=throttle[0])
    for label, event_indices in (
            ("fall_gt_0p05", np.flatnonzero(changes < -0.05)),
            ("rise_gt_0p05", np.flatnonzero(changes > 0.05))):
        events: dict[str, Any] = {"count": int(len(event_indices)), "lags": {}}
        for lag in (0, 1, 2):
            indices = event_indices[event_indices + lag < len(rows)]
            actual = actual_gap[indices + lag]
            events["lags"][str(lag)] = {
                "model_gap_abs_median_mps": float(np.median(model_gap[indices]))
                if len(indices) else None,
                "tire_slip_abs_median": float(np.median(model_slip[indices]))
                if len(indices) else None,
                "encoder_truth_gap_abs_median_mps": float(np.median(actual))
                if len(indices) else None,
            }
        report["throttle_fall_events"][label] = events
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--initial", type=Path, default=DEFAULT_INITIAL)
    parser.add_argument("--shapes", type=Path, default=DEFAULT_SHAPES)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = replay(args.csv, args.binary, args.initial, args.shapes, args.output)
    encoded = json.dumps(report, indent=2)
    print(encoded)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
