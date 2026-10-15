#!/usr/bin/env python3
"""Identify the longitudinal WheelHit slip law from Unity source captures.

Fit only on r07, then score unchanged on r11. Rigidbody state, contact-point
velocity, wheel RPM, and WheelHit slip are privileged Unity-only signals; this
identifies the slip equation, not a deployable odometry measurement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import least_squares


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN = ROOT / (
    "live_runs/unity_exact_binary_audit_20261012/"
    "dynamics_source_parity_20261012_r07/"
    "wheel_dynamics_unity_vehicle_model_parity_source_20261012_r07.csv")
DEFAULT_VALIDATION = ROOT / (
    "live_runs/unity_exact_binary_audit_20261012/"
    "dynamics_source_exact_replay_20261012_r11/"
    "wheel_dynamics_unity_vehicle_model_parity_source_exact_replay_20261012_r11.csv")
WHEELS = ("fl", "fr", "rl", "rr")
WHEEL_PREFIX = {"fl": "front_left", "fr": "front_right",
                "rl": "rear_left", "rr": "rear_right"}
SPEED_BINS = ((0.2, 1.0), (1.0, 3.0), (3.0, 6.0),
              (6.0, 9.0), (9.0, 12.1))


def _metadata(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.startswith("# "):
                break
            key, value = line[2:].strip().split("=", 1)
            result[key] = value
    return result


def _collect(path: Path, chunksize: int) -> dict[str, Any]:
    metadata = _metadata(path)
    output = {wheel: {name: [] for name in
                      ("speed", "steering", "contact_u", "surface", "slip")}
              for wheel in WHEELS}
    columns = ["vel_body_x_mps", "vel_body_y_mps", "vel_body_z_mps",
               "omega_body_x_rps", "omega_body_y_rps", "omega_body_z_rps"]
    for wheel in WHEELS:
        columns += [f"{wheel}_{field}" for field in (
            "grounded", "rpm", "forward_slip", "steer_deg",
            "contact_body_x", "contact_body_y", "contact_body_z",
            "contact_forward_body_x", "contact_forward_body_y",
            "contact_forward_body_z")]
    com = np.asarray([float(value) for value in
                      metadata["center_of_mass_body"].split(",")])
    for frame in pd.read_csv(path, comment="#", usecols=columns,
                             chunksize=chunksize):
        velocity = frame[["vel_body_x_mps", "vel_body_y_mps",
                          "vel_body_z_mps"]].to_numpy(dtype=np.float64)
        omega = frame[["omega_body_x_rps", "omega_body_y_rps",
                       "omega_body_z_rps"]].to_numpy(dtype=np.float64)
        speed = np.linalg.norm(velocity, axis=1)
        for wheel in WHEELS:
            grounded = frame[f"{wheel}_grounded"].to_numpy() > 0.5
            contact = frame[[f"{wheel}_contact_body_{axis}"
                             for axis in ("x", "y", "z")]].to_numpy(
                                 dtype=np.float64)
            forward = frame[[f"{wheel}_contact_forward_body_{axis}"
                             for axis in ("x", "y", "z")]].to_numpy(
                                 dtype=np.float64)
            point_velocity = velocity + np.cross(omega, contact - com)
            contact_u = np.einsum("ij,ij->i", point_velocity, forward)
            prefix = WHEEL_PREFIX[wheel]
            radius = float(metadata[f"{prefix}_radius_m"])
            surface = (frame[f"{wheel}_rpm"].to_numpy(dtype=np.float64)
                       * (2.0 * np.pi / 60.0) * radius)
            steer = np.abs(np.deg2rad(
                frame[f"{wheel}_steer_deg"].to_numpy(dtype=np.float64)))
            slip = frame[f"{wheel}_forward_slip"].to_numpy(dtype=np.float64)
            valid = (grounded & (speed >= SPEED_BINS[0][0])
                     & (speed < SPEED_BINS[-1][1])
                     & np.isfinite(contact_u) & np.isfinite(surface)
                     & np.isfinite(slip))
            for name, values in (("speed", speed), ("steering", steer),
                                 ("contact_u", contact_u),
                                 ("surface", surface), ("slip", slip)):
                output[wheel][name].append(values[valid])
    return {wheel: {name: np.concatenate(parts)
                    for name, parts in values.items()}
            for wheel, values in output.items()}


def _predict(model: str, params: np.ndarray, surface: np.ndarray,
             contact_u: np.ndarray) -> np.ndarray:
    radius_scale, regularizer = params
    numerator = radius_scale * surface - contact_u
    if model == "additive_speed_regularization":
        denominator = np.abs(contact_u) + regularizer
    elif model == "max_speed_floor":
        denominator = np.maximum(np.abs(contact_u), regularizer)
    elif model == "wheel_surface_floor":
        denominator = np.maximum(np.abs(radius_scale * surface), regularizer)
    elif model == "max_speed_plus_regularizer":
        denominator = (np.maximum(np.abs(radius_scale * surface),
                                  np.abs(contact_u)) + regularizer)
    else:
        raise ValueError(f"unknown slip law: {model}")
    return numerator / denominator


def _score(data: dict[str, np.ndarray], prediction: np.ndarray
           ) -> dict[str, Any]:
    error = prediction - data["slip"]
    speed = data["speed"]
    result: dict[str, Any] = {
        "samples": int(error.size),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "bias": float(np.mean(error)),
        "absolute_p95": float(np.quantile(np.abs(error), 0.95)),
        "speed_bins": {},
    }
    for low, high in SPEED_BINS:
        mask = (speed >= low) & (speed < high)
        result["speed_bins"][f"{low:g}-{high:g}mps"] = {
            "samples": int(mask.sum()),
            "rmse": (float(np.sqrt(np.mean(error[mask]**2)))
                     if mask.any() else None),
        }
    for label, low, high in (("low", 0.0, 0.15), ("medium", 0.15, 0.35),
                             ("high", 0.35, 0.55)):
        mask = (data["steering"] >= low) & (data["steering"] < high)
        result.setdefault("steering_bins", {})[label] = {
            "samples": int(mask.sum()),
            "rmse": (float(np.sqrt(np.mean(error[mask]**2)))
                     if mask.any() else None),
        }
    return result


def fit(train_path: Path, validation_path: Path, stride: int,
        chunksize: int) -> dict[str, Any]:
    train = _collect(train_path, chunksize)
    validation = _collect(validation_path, chunksize)
    laws = {}
    for law in ("additive_speed_regularization", "max_speed_floor",
                "wheel_surface_floor"):
        wheel_results = {}
        for wheel in WHEELS:
            train_data = train[wheel]
            fit_domain = train_data["speed"] >= 1.0
            fit_rows = np.flatnonzero(fit_domain)
            indexes = fit_rows[::stride]

            def residual(parameters: np.ndarray) -> np.ndarray:
                return (_predict(law, parameters,
                                 train_data["surface"][indexes],
                                 train_data["contact_u"][indexes])
                        - train_data["slip"][indexes])

            result = least_squares(
                residual, x0=np.asarray([1.0, 1.0]),
                bounds=([0.8, 0.01], [1.2, 5.0]),
                loss="soft_l1", f_scale=0.005, max_nfev=200)
            params = result.x
            wheel_results[wheel] = {
                "fitted_radius_scale": float(params[0]),
                "fitted_regularizer_mps": float(params[1]),
                "fit_samples": int(len(indexes)),
                "fit_speed_domain_mps": [1.0, 12.1],
                "training": _score(train_data, _predict(
                    law, params, train_data["surface"],
                    train_data["contact_u"])),
                "held_out": _score(validation[wheel], _predict(
                    law, params, validation[wheel]["surface"],
                    validation[wheel]["contact_u"])),
                "optimizer_status": int(result.status),
            }
        laws[law] = wheel_results

    baseline = {}
    for wheel in WHEELS:
        train_data = train[wheel]
        validation_data = validation[wheel]
        baseline[wheel] = {
            "training": _score(train_data, _predict(
                "additive_speed_regularization", np.asarray([1.0, 1.0]),
                train_data["surface"], train_data["contact_u"])),
            "held_out": _score(validation_data, _predict(
                "additive_speed_regularization", np.asarray([1.0, 1.0]),
                validation_data["surface"], validation_data["contact_u"])),
        }
    pooled_train = {
        name: np.concatenate([train[wheel][name] for wheel in WHEELS])
        for name in train[WHEELS[0]]
    }
    pooled_validation = {
        name: np.concatenate([validation[wheel][name] for wheel in WHEELS])
        for name in validation[WHEELS[0]]
    }
    low_speed_train = np.flatnonzero(pooled_train["speed"] >= 0.2)
    shared_indexes = low_speed_train[::stride]

    def shared_residual(parameters: np.ndarray) -> np.ndarray:
        selected = {name: values[shared_indexes]
                    for name, values in pooled_train.items()}
        return (_predict("max_speed_plus_regularizer", parameters,
                         selected["surface"], selected["contact_u"])
                - selected["slip"])

    shared_fit = least_squares(
        shared_residual, x0=np.asarray([1.0, 0.1]),
        bounds=([0.8, 0.0], [1.2, 2.0]),
        loss="soft_l1", f_scale=0.005, max_nfev=200)
    shared_parameters = shared_fit.x
    shared_candidate = {
        "model": "(R*omega - v_contact) / (max(abs(R*omega), abs(v_contact)) + epsilon)",
        "fitted_radius_scale": float(shared_parameters[0]),
        "fitted_speed_regularizer_mps": float(shared_parameters[1]),
        "fit_samples": int(shared_indexes.size),
        "fit_speed_domain_mps": [0.2, 12.1],
        "training": _score(
            pooled_train, _predict("max_speed_plus_regularizer",
                                   shared_parameters,
                                   pooled_train["surface"],
                                   pooled_train["contact_u"])),
        "held_out": _score(
            pooled_validation, _predict("max_speed_plus_regularizer",
                                        shared_parameters,
                                        pooled_validation["surface"],
                                        pooled_validation["contact_u"])),
        "optimizer_status": int(shared_fit.status),
    }
    return {
        "schema_version": 1,
        "status": "research-only identification; no runtime odometry change",
        "train_capture": str(train_path.relative_to(ROOT)),
        "held_out_capture": str(validation_path.relative_to(ROOT)),
        "train_run_fit_only": True,
        "inputs": "WheelHit.forwardSlip labels; GT contact-point velocity; WheelCollider RPM",
        "fit_method": (
            "per-wheel legacy candidates fit above 1 m/s; shared normalized "
            "candidate fit on every Nth pooled grounded sample from 0.2 m/s"),
        "stride": stride,
        "baseline_additive_one_mps": baseline,
        "laws": laws,
        "shared_max_speed_plus_regularizer_candidate": shared_candidate,
        "interpretation": (
            "Compare fitted law error by speed and steering. A good slip fit "
            "does not establish an odometry observer; runtime must estimate "
            "body velocity and wheel contact state from permitted sensors."),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--output", type=Path, default=ROOT / (
        "live_runs/unity_exact_binary_audit_20261012/"
        "longitudinal_slip_law_fit_20261012.json"))
    parser.add_argument("--stride", type=int, default=25)
    parser.add_argument("--chunksize", type=int, default=25000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    report = fit(args.train, args.validation, args.stride, args.chunksize)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    for law, wheels in report["laws"].items():
        print(law)
        for wheel, result in wheels.items():
            print(f"  {wheel}: train={result['training']['rmse']:.6g} "
                  f"heldout={result['held_out']['rmse']:.6g} "
                  f"radius={result['fitted_radius_scale']:.6f} "
                  f"regularizer={result['fitted_regularizer_mps']:.6g}")
    shared = report["shared_max_speed_plus_regularizer_candidate"]
    print("shared_max_speed_plus_regularizer",
          "radius", shared["fitted_radius_scale"],
          "epsilon", shared["fitted_speed_regularizer_mps"],
          "train_rmse", shared["training"]["rmse"],
          "heldout_rmse", shared["held_out"]["rmse"])
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
