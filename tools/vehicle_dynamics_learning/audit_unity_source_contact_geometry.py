#!/usr/bin/env python3
"""Compare the 3-D plant's source geometry to captured Unity WheelHit frames."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tools.vehicle_dynamics_learning.analyze_unity_suspension_load import (
    WHEEL_TRANSFORM,
)
from tools.vehicle_dynamics_learning.replay_unity_source_3d_plant import (
    WHEELS,
    WHEEL_KEYS,
    WHEEL_RADIUS_M,
    _rotate,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CAPTURES = (
    ROOT / "live_runs/unity_exact_binary_audit_20261012/"
    "dynamics_source_parity_20261012_r07/"
    "wheel_dynamics_unity_vehicle_model_parity_source_20261012_r07.csv",
    ROOT / "live_runs/unity_exact_binary_audit_20261012/"
    "dynamics_source_exact_replay_20261012_r11/"
    "wheel_dynamics_unity_vehicle_model_parity_source_exact_replay_20261012_r11.csv",
    ROOT / "live_runs/unity_mechanics_dynamic_validation_20261013_r12g/"
    "unity_source_capture/"
    "wheel_dynamics_unity_mechanics_dynamic_validation_r12g.csv",
)
SPEED_BINS = ((1.0, 3.0), (3.0, 6.0), (6.0, 9.0), (9.0, 12.1))
STEERING_BINS = ((0.0, 0.1), (0.1, 0.25), (0.25, 0.4), (0.4, 0.55))


def _metadata(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.startswith("# "):
                break
            key, value = line[2:].strip().split("=", 1)
            values[key] = value
    return values


def _rotate_rows(q: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    qv = q[:, :3]
    twice = 2.0 * np.cross(qv, vectors)
    return vectors + q[:, 3, None] * twice + np.cross(qv, twice)


def _summary(values: np.ndarray) -> dict[str, float | int]:
    if not values.size:
        return {"n": 0, "rmse": float("nan"), "p95_abs": float("nan")}
    return {
        "n": int(values.size),
        "rmse": float(np.sqrt(np.mean(values**2))),
        "p95_abs": float(np.quantile(np.abs(values), 0.95)),
    }


def _capture(path: Path, chunksize: int) -> dict[str, Any]:
    metadata = _metadata(path)
    com = np.asarray([float(value) for value in
                      metadata["center_of_mass_body"].split(",")])
    columns = ["pos_world_x_m", "pos_world_y_m", "pos_world_z_m",
               "rot_world_x", "rot_world_y", "rot_world_z", "rot_world_w",
               "vel_body_x_mps", "vel_body_y_mps", "vel_body_z_mps"]
    for wheel in WHEELS:
        columns.extend(f"{wheel}_{field}" for field in (
            "grounded", "steer_deg", "contact_body_x", "contact_body_y",
            "contact_body_z", "contact_forward_body_x",
            "contact_forward_body_y", "contact_forward_body_z",
            "contact_sideways_body_x", "contact_sideways_body_y",
            "contact_sideways_body_z"))
    accum: dict[str, dict[str, list[np.ndarray]]] = {
        wheel: {key: [] for key in (
            "speed", "steering", "recorded_steer_forward_angle_deg",
            "recorded_steer_side_angle_deg",
            "negated_steer_forward_angle_deg",
            "negated_steer_side_angle_deg",
            "contact_error_m", "contact_dx_m", "contact_dy_m",
            "contact_dz_m")} for wheel in WHEELS}
    row_count = 0
    for frame in pd.read_csv(path, comment="#", usecols=columns,
                             chunksize=chunksize):
        row_count += len(frame)
        q = frame[["rot_world_x", "rot_world_y", "rot_world_z",
                   "rot_world_w"]].to_numpy(dtype=np.float64)
        q /= np.maximum(np.linalg.norm(q, axis=1)[:, None], 1e-12)
        inverse = q.copy()
        inverse[:, :3] *= -1.0
        normal = _rotate_rows(
            inverse, np.broadcast_to((0.0, 1.0, 0.0), (len(frame), 3)))
        body_position = frame[["pos_world_x_m", "pos_world_y_m",
                               "pos_world_z_m"]].to_numpy(dtype=np.float64)
        velocity = frame[["vel_body_x_mps", "vel_body_y_mps",
                          "vel_body_z_mps"]].to_numpy(dtype=np.float64)
        speed = np.linalg.norm(velocity, axis=1)

        for wheel in WHEELS:
            mount, static_q_tuple = WHEEL_TRANSFORM[wheel]
            static_q = np.asarray(static_q_tuple, dtype=np.float64)
            collider_center = _rotate(
                static_q, np.asarray((0.0, 0.025, 0.0)))
            mount_body = np.asarray(mount, dtype=np.float64) + collider_center
            down_body = _rotate(static_q, np.asarray((0.0, -1.0, 0.0)))
            mount_world = body_position + _rotate_rows(
                q, np.broadcast_to(mount_body, (len(frame), 3)))
            down_world = _rotate_rows(
                q, np.broadcast_to(down_body, (len(frame), 3)))
            travel = ((WHEEL_RADIUS_M - mount_world[:, 1])
                      / np.where(np.abs(down_world[:, 1]) > 1e-6,
                                 down_world[:, 1], np.nan))
            predicted_contact = (
                mount_body + travel[:, None] * down_body
                - WHEEL_RADIUS_M * normal)
            observed_contact = frame[[
                f"{wheel}_contact_body_x", f"{wheel}_contact_body_y",
                f"{wheel}_contact_body_z"]].to_numpy(dtype=np.float64)
            contact_delta = predicted_contact - observed_contact
            contact_error = np.linalg.norm(contact_delta, axis=1)

            captured_steer = np.deg2rad(
                frame[f"{wheel}_steer_deg"].to_numpy(dtype=np.float64))
            # _initial_state() negates captured angles when recovering center
            # steering; _plant_derivative() negates the Ackermann angles again.
            # Thus the source plant's final physical wheel angles have the same
            # sign as the captured WheelCollider steer angles.
            steer = captured_steer
            local_forward = np.stack((np.sin(steer), np.zeros_like(steer),
                                      np.cos(steer)), axis=1)
            wheel_forward = np.broadcast_to(static_q, (len(frame), 4)).copy()
            wheel_forward = _rotate_rows(wheel_forward, local_forward)
            wheel_forward -= normal * np.sum(
                wheel_forward * normal, axis=1)[:, None]
            wheel_forward /= np.maximum(
                np.linalg.norm(wheel_forward, axis=1)[:, None], 1e-12)
            wheel_side = np.cross(normal, wheel_forward)
            wheel_side /= np.maximum(
                np.linalg.norm(wheel_side, axis=1)[:, None], 1e-12)
            observed_forward = frame[[
                f"{wheel}_contact_forward_body_x",
                f"{wheel}_contact_forward_body_y",
                f"{wheel}_contact_forward_body_z"]].to_numpy(dtype=np.float64)
            observed_side = frame[[
                f"{wheel}_contact_sideways_body_x",
                f"{wheel}_contact_sideways_body_y",
                f"{wheel}_contact_sideways_body_z"]].to_numpy(dtype=np.float64)
            forward_angle = np.rad2deg(np.arccos(np.clip(
                np.sum(wheel_forward * observed_forward, axis=1), -1.0, 1.0)))
            side_angle = np.rad2deg(np.arccos(np.clip(
                np.sum(wheel_side * observed_side, axis=1), -1.0, 1.0)))
            negated_local_forward = np.stack((
                np.sin(-captured_steer), np.zeros_like(captured_steer),
                np.cos(-captured_steer)), axis=1)
            negated_forward = _rotate_rows(
                np.broadcast_to(static_q, (len(frame), 4)).copy(),
                negated_local_forward)
            negated_forward -= normal * np.sum(
                negated_forward * normal, axis=1)[:, None]
            negated_forward /= np.maximum(
                np.linalg.norm(negated_forward, axis=1)[:, None], 1e-12)
            negated_side = np.cross(normal, negated_forward)
            negated_side /= np.maximum(
                np.linalg.norm(negated_side, axis=1)[:, None], 1e-12)
            negated_forward_angle = np.rad2deg(np.arccos(np.clip(
                np.sum(negated_forward * observed_forward, axis=1), -1.0, 1.0)))
            negated_side_angle = np.rad2deg(np.arccos(np.clip(
                np.sum(negated_side * observed_side, axis=1), -1.0, 1.0)))
            valid = ((frame[f"{wheel}_grounded"].to_numpy() > 0.5)
                     & (speed > 1.0) & np.isfinite(contact_error)
                     & np.isfinite(forward_angle) & np.isfinite(side_angle))
            accum[wheel]["speed"].append(speed[valid])
            accum[wheel]["steering"].append(np.abs(steer[valid]))
            accum[wheel]["recorded_steer_forward_angle_deg"].append(
                forward_angle[valid])
            accum[wheel]["recorded_steer_side_angle_deg"].append(
                side_angle[valid])
            accum[wheel]["negated_steer_forward_angle_deg"].append(
                negated_forward_angle[valid])
            accum[wheel]["negated_steer_side_angle_deg"].append(
                negated_side_angle[valid])
            accum[wheel]["contact_error_m"].append(contact_error[valid])
            for axis, column in enumerate(("contact_dx_m", "contact_dy_m",
                                           "contact_dz_m")):
                accum[wheel][column].append(contact_delta[valid, axis])

    wheel_report = {}
    for wheel, fields in accum.items():
        values = {key: np.concatenate(parts) for key, parts in fields.items()}
        valid_results = {}
        for metric in ("recorded_steer_forward_angle_deg",
                       "recorded_steer_side_angle_deg",
                       "negated_steer_forward_angle_deg",
                       "negated_steer_side_angle_deg",
                       "contact_error_m"):
            v = values[metric]
            valid_results[metric] = {
                "overall": _summary(v),
                "by_speed": {
                    f"{low:g}-{high:g}_mps": _summary(
                        v[(values["speed"] >= low)
                          & (values["speed"] < high)])
                    for low, high in SPEED_BINS
                },
                "by_abs_steering": {
                    f"{low:g}-{high:g}_rad": _summary(
                        v[(values["steering"] >= low)
                          & (values["steering"] < high)])
                    for low, high in STEERING_BINS
                },
            }
        contact_delta = np.stack((values["contact_dx_m"],
                                  values["contact_dy_m"],
                                  values["contact_dz_m"]), axis=1)
        valid_results["contact_error_vector_body_m"] = {
            "bias_xyz": np.mean(contact_delta, axis=0).tolist(),
            "rmse_xyz": np.sqrt(np.mean(contact_delta**2, axis=0)).tolist(),
        }
        wheel_report[wheel] = valid_results
    return {
        "capture": str(path.relative_to(ROOT)),
        "rows": row_count,
        "method": "reconstruct WheelCollider contact point and projected forward/side axes from serialized wheel transforms, Rigidbody pose, flat ground y=0, wheel radius, and captured steering; compare with Unity WheelHit.point/forwardDir/sidewaysDir",
        "com_body_m": com.tolist(),
        "per_wheel": wheel_report,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", nargs="*", type=Path,
                        default=DEFAULT_CAPTURES)
    parser.add_argument("--chunksize", type=int, default=25000)
    parser.add_argument("--output", type=Path, default=ROOT / (
        "live_runs/unity_exact_binary_audit_20261012/"
        "source_contact_geometry_audit_20261013.json"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    reports = [_capture(path, args.chunksize) for path in args.captures]
    payload = {
        "status": "offline Unity-source geometry validation; no runtime or plant default changed",
        "captures": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    for capture in reports:
        print(capture["capture"])
        for wheel, result in capture["per_wheel"].items():
            fwd = result["recorded_steer_forward_angle_deg"]["overall"]
            side = result["recorded_steer_side_angle_deg"]["overall"]
            alt_fwd = result["negated_steer_forward_angle_deg"]["overall"]
            point = result["contact_error_m"]["overall"]
            print(wheel, "forward-axis", round(fwd["rmse"], 5),
                  "negated-steer", round(alt_fwd["rmse"], 5),
                  "side-axis", round(side["rmse"], 5),
                  "contact-point", round(point["rmse"], 6))
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
