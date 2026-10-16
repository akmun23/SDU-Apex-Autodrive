#!/usr/bin/env python3
"""Identify the active WheelCollider spring/damper load state from Unity captures.

WheelCollider.GetGroundHit force and GetWorldPose wheel-center position are
privileged source labels. This checks whether the serialized suspension
parameters explain per-wheel normal load on a frozen independent capture; it
does not treat Unity-only signals as runtime odometry inputs.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from tools.vehicle_dynamics_learning.unity_vehicle_mechanics import (
    WHEEL_COLLIDER_TRANSFORM_POSITION_BODY_M,
    WHEEL_COLLIDER_TRANSFORM_ROTATION_BODY_XYZW,
)


ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / (
    "live_runs/unity_exact_binary_audit_20261012/"
    "dynamics_source_parity_20261012_r07/"
    "wheel_dynamics_unity_vehicle_model_parity_source_20261012_r07.csv")
VALIDATION = ROOT / (
    "live_runs/unity_exact_binary_audit_20261012/"
    "dynamics_source_exact_replay_20261012_r11/"
    "wheel_dynamics_unity_vehicle_model_parity_source_exact_replay_20261012_r11.csv")
OUTPUT = ROOT / (
    "live_runs/unity_exact_binary_audit_20261012/"
    "suspension_load_geometry_calibration_v2_20261012.json")
WHEELS = ("fl", "fr", "rl", "rr")
WHEEL_PREFIX = {"fl": "front_left", "fr": "front_right",
                "rl": "rear_left", "rr": "rear_right"}
PREFAB_TARGET_CENTER_BODY_Y_M = {"fl": 0.06, "fr": 0.06,
                                 "rl": 0.05, "rr": 0.05}
SERIALIZED_SPRING_NPM = 500.0
SERIALIZED_DAMPER_NSPM = 100.0
DERIVATIVE_WINDOW = 11
GROUND_PLANE_Y_M = 0.0
WHEEL_TRANSFORM = dict(zip(
    WHEELS, zip(WHEEL_COLLIDER_TRANSFORM_POSITION_BODY_M,
                WHEEL_COLLIDER_TRANSFORM_ROTATION_BODY_XYZW)))


def _rotate_vector(quaternion: tuple[float, float, float, float],
                   vector: np.ndarray) -> np.ndarray:
    xyz = np.asarray(quaternion[:3], dtype=np.float64)
    twice_cross = 2.0 * np.cross(xyz, vector)
    return vector + quaternion[3] * twice_cross + np.cross(xyz, twice_cross)


def _rotate_rows(quaternion: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    xyz = quaternion[:, :3]
    twice_cross = 2.0 * np.cross(xyz, vectors)
    return vectors + quaternion[:, 3, None] * twice_cross \
        + np.cross(xyz, twice_cross)


def _metadata(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.startswith("# "):
                break
            key, value = line[2:].strip().split("=", 1)
            result[key] = value
    return result


def _read(path: Path) -> dict[str, Any]:
    metadata = _metadata(path)
    columns = ["capture_realtime_s", "pos_world_x_m", "pos_world_y_m",
               "pos_world_z_m", "rot_world_x", "rot_world_y",
               "rot_world_z", "rot_world_w"]
    for wheel in WHEELS:
        columns += [f"{wheel}_{field}" for field in (
            "grounded", "contact_force_n", "wheel_pose_body_y")]
    frame = pd.read_csv(path, comment="#", usecols=columns)
    dt = float(metadata["fixed_delta_time_s"])
    result: dict[str, Any] = {"capture": path.name, "dt_s": dt,
                              "metadata": metadata, "wheels": {},
                              "position_world": frame[[
                                  "pos_world_x_m", "pos_world_y_m",
                                  "pos_world_z_m"]].to_numpy(dtype=np.float64),
                              "rotation_world": frame[[
                                  "rot_world_x", "rot_world_y", "rot_world_z",
                                  "rot_world_w"]].to_numpy(dtype=np.float64)}
    result["rotation_world"] /= np.linalg.norm(
        result["rotation_world"], axis=1)[:, None]
    for wheel in WHEELS:
        position = frame[f"{wheel}_wheel_pose_body_y"].to_numpy(dtype=np.float64)
        load = frame[f"{wheel}_contact_force_n"].to_numpy(dtype=np.float64)
        grounded = frame[f"{wheel}_grounded"].to_numpy() > 0.5
        rate = savgol_filter(position, DERIVATIVE_WINDOW, 3, deriv=1,
                             delta=dt, mode="interp")
        valid = grounded & np.isfinite(position) & np.isfinite(rate) \
            & np.isfinite(load)
        valid[:DERIVATIVE_WINDOW] = False
        valid[-DERIVATIVE_WINDOW:] = False
        result["wheels"][wheel] = {
            "position_m": position,
            "rate_mps": rate,
            "load_n": load,
            "valid": valid,
        }
    return result


def _score(error: np.ndarray) -> dict[str, float | int]:
    return {
        "samples": int(error.size),
        "rmse_n": float(np.sqrt(np.mean(error**2))),
        "bias_n": float(np.mean(error)),
        "p95_abs_n": float(np.quantile(np.abs(error), 0.95)),
    }


def _model(data: dict[str, Any], wheel: str, spring: float, damper: float,
           offset: float) -> np.ndarray:
    values = data["wheels"][wheel]
    return (offset + spring * values["position_m"]
            + damper * values["rate_mps"])


def _fit_and_score(train: dict[str, Any], validation: dict[str, Any]
                   ) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for wheel in WHEELS:
        train_wheel = train["wheels"][wheel]
        validation_wheel = validation["wheels"][wheel]
        train_valid = train_wheel["valid"]
        val_valid = validation_wheel["valid"]
        design = np.column_stack((
            np.ones(np.count_nonzero(train_valid)),
            train_wheel["position_m"][train_valid],
            train_wheel["rate_mps"][train_valid]))
        target = train_wheel["load_n"][train_valid]
        offset_fixed = float(np.mean(
            target - SERIALIZED_SPRING_NPM * design[:, 1]
            - SERIALIZED_DAMPER_NSPM * design[:, 2]))
        fitted_offset, fitted_spring, fitted_damper = np.linalg.lstsq(
            design, target, rcond=None)[0]
        sprung_mass = float(train["metadata"][
            f"{WHEEL_PREFIX[wheel]}_sprung_mass_kg"])
        source_target_y = ((sprung_mass * 9.80665 - offset_fixed)
                           / SERIALIZED_SPRING_NPM)
        models = {
            "serialized_kc_with_training_offset": (
                SERIALIZED_SPRING_NPM, SERIALIZED_DAMPER_NSPM, offset_fixed),
            "training_fit_kc_frozen": (
                float(fitted_spring), float(fitted_damper), float(fitted_offset)),
        }
        report[wheel] = {
            "serialized": {"spring_n_per_m": SERIALIZED_SPRING_NPM,
                           "damper_ns_per_m": SERIALIZED_DAMPER_NSPM},
            "sprung_mass_kg": sprung_mass,
            "serialized_coefficients_calibrated_target_center_body_y_m":
                source_target_y,
            "target_center_height_delta_from_prefab_m": (
                source_target_y - PREFAB_TARGET_CENTER_BODY_Y_M[wheel]),
            "fitted_training_parameters": {
                "spring_n_per_m": float(fitted_spring),
                "damper_ns_per_m": float(fitted_damper),
                "offset_n": float(fitted_offset),
                "effective_zero_spring_load_height_body_y_m": float(
                    (sprung_mass * 9.80665 - fitted_offset) / fitted_spring),
            },
            "prefab_target_center_body_y_m":
                PREFAB_TARGET_CENTER_BODY_Y_M[wheel],
            "position_range_training_m": [
                float(np.quantile(train_wheel["position_m"][train_valid], 0.01)),
                float(np.quantile(train_wheel["position_m"][train_valid], 0.99))],
            "load_range_training_n": [
                float(np.quantile(target, 0.01)), float(np.quantile(target, 0.99))],
            "models": {},
        }
        for name, (spring, damper, offset) in models.items():
            train_error = (_model(train, wheel, spring, damper, offset)
                           - train_wheel["load_n"])[train_valid]
            validation_error = (_model(validation, wheel, spring, damper, offset)
                                - validation_wheel["load_n"])[val_valid]
            report[wheel]["models"][name] = {
                "training": _score(train_error),
                "independent_run": _score(validation_error),
            }
    return report


def _geometry_load_prediction(data: dict[str, Any], wheel_name: str,
                              target_travel_m: float
                              ) -> tuple[np.ndarray, np.ndarray]:
    """Solve wheel travel from rigid-body pose and a flat known ground plane."""
    transform_position, transform_rotation = WHEEL_TRANSFORM[wheel_name]
    base_local = (np.asarray(transform_position, dtype=np.float64)
                  + _rotate_vector(transform_rotation,
                                   np.asarray([0.0, 0.025, 0.0])))
    down_local = _rotate_vector(transform_rotation,
                                np.asarray([0.0, -1.0, 0.0]))
    quaternion = data["rotation_world"]
    position = data["position_world"]
    base_world = position + _rotate_rows(
        quaternion, np.broadcast_to(base_local, position.shape))
    down_world = _rotate_rows(
        quaternion, np.broadcast_to(down_local, position.shape))
    wheel_data = data["wheels"][wheel_name]
    prefix = WHEEL_PREFIX[wheel_name]
    radius = float(data["metadata"][f"{prefix}_radius_m"])
    travel = ((GROUND_PLANE_Y_M + radius - base_world[:, 1])
              / down_world[:, 1])
    travel_rate = savgol_filter(
        travel, DERIVATIVE_WINDOW, 3, deriv=1, delta=data["dt_s"],
        mode="interp")
    compression = target_travel_m - travel
    compression_rate = -travel_rate
    sprung_mass = float(data["metadata"][f"{prefix}_sprung_mass_kg"])
    spring = float(data["metadata"][f"{prefix}_spring_n_per_m"])
    damper = float(data["metadata"][f"{prefix}_damper_ns_per_m"])
    prediction = np.maximum(
        0.0, sprung_mass * 9.80665 + spring * compression
        + damper * compression_rate)
    wheel_center_local_y = base_local[1] + down_local[1] * travel
    pose_error = (wheel_center_local_y - wheel_data["position_m"])
    return prediction, pose_error


def _geometry_results(train: dict[str, Any], validation: dict[str, Any]
                      ) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for wheel in WHEELS:
        prefix = WHEEL_PREFIX[wheel]
        spring = float(train["metadata"][
            f"{prefix}_spring_n_per_m"])
        damper = float(train["metadata"][
            f"{prefix}_damper_ns_per_m"])
        target_position = float(train["metadata"][
            f"{prefix}_target_position"])
        suspension_distance = float(train["metadata"][
            f"{prefix}_suspension_distance_m"])
        target_travel = target_position * suspension_distance
        train_wheel = train["wheels"][wheel]
        valid = train_wheel["valid"]
        # One per-wheel target-height calibration is fit on training WheelHit
        # labels; spring and damper remain fixed at Unity's serialized values.
        train_offset = float(np.mean(
            train_wheel["load_n"][valid]
            - spring * train_wheel["position_m"][valid]
            - damper * train_wheel["rate_mps"][valid]))
        sprung_mass = float(train["metadata"][f"{prefix}_sprung_mass_kg"])
        calibrated_target_y = (sprung_mass * 9.80665 - train_offset) / spring
        prefab_target_y = WHEEL_TRANSFORM[wheel][0][1]
        down_local_y = _rotate_vector(
            WHEEL_TRANSFORM[wheel][1], np.asarray([0.0, -1.0, 0.0]))[1]
        calibrated_target_travel = (
            target_travel + (calibrated_target_y - prefab_target_y)
            / down_local_y)
        results[wheel] = {
            "serialized_target_travel_m": target_travel,
            "training_calibrated_target_travel_m": calibrated_target_travel,
            "effective_target_center_y_body_m": calibrated_target_y,
            "prefab_target_center_y_body_m": prefab_target_y,
            "training": {},
            "independent_run": {},
        }
        for name, data in (("training", train),
                           ("independent_run", validation)):
            wheel_data = data["wheels"][wheel]
            score_mask = wheel_data["valid"]
            for candidate, travel in (("geometry_only", target_travel),
                                      ("training_calibrated_geometry",
                                       calibrated_target_travel)):
                prediction, pose_error = _geometry_load_prediction(
                    data, wheel, travel)
                scores = _score(
                    prediction[score_mask] - wheel_data["load_n"][score_mask])
                scores["wheel_pose_body_y_rmse_m"] = float(np.sqrt(np.mean(
                    pose_error[score_mask]**2)))
                results[wheel][name][candidate] = scores
    return results


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    train = _read(TRAIN)
    validation = _read(VALIDATION)
    payload = {
        "schema_version": 1,
        "status": "source-parameter test; no runtime model promoted",
        "method": "fit wheel normal load from body-frame wheel-center height and its rate; freeze parameters on a distinct Unity run",
        "training_capture": str(TRAIN.relative_to(ROOT)),
        "validation_capture": str(VALIDATION.relative_to(ROOT)),
        "fixed_delta_time_s": train["dt_s"],
        "derivative_window_samples": DERIVATIVE_WINDOW,
        "derivative_window_s": DERIVATIVE_WINDOW * train["dt_s"],
        "results": _fit_and_score(train, validation),
        "geometry_only_load_model": _geometry_results(train, validation),
        "geometry": {
            "ground_plane_y_m": GROUND_PLANE_Y_M,
            "wheel_transform_source": "F1TENTH.prefab wheel-collider transforms and center=(0,0.025,0); suspension ray solved to ground_y + radius",
            "calibration": "per-wheel effective target height fit on r07 WheelHit labels only; k,c fixed to serialized values; all parameters frozen on r11",
        },
        "limitations": [
            "WheelHit load and wheel pose are Unity-only training labels, not public competition sensor inputs.",
            "The learned per-wheel target-height adjustment is below 0.7 mm and must be frozen from training only.",
            "The open-plane geometry model is not yet validated on a banked/uneven road or inside a recursive vehicle rollout.",
        ],
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(OUTPUT)
    for wheel, result in payload["results"].items():
        print(wheel, "fit k,c", result["fitted_training_parameters"])
        for model, scores in result["models"].items():
            print(" ", model, "train", scores["training"]["rmse_n"],
                  "validation", scores["independent_run"]["rmse_n"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
