#!/usr/bin/env python3
"""Attribute one-step structured-plant residuals by operating condition.

This is an offline diagnostic.  It uses the recorded next state to measure
derivative residuals, but never feeds that state back into the plant or any
runtime estimator.  The command inputs and current state are the only model
inputs.  Leave-one-run-out regressions are deliberately small and are used to
identify a missing physical mechanism, not to create a production correction
term automatically.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from structured_vehicle_plant import (  # noqa: E402
    MAX_STEERING_RAD,
    MIN_SLIP_SPEED_MPS,
    PlantParameters,
    _steering_next,
    _wheel_next,
    step,
)


STATE_FIELDS = ("u", "v", "r", "wheel", "steering")
REQUIRED_FIELDS = {
    "simulation_time_k_s", "dt_sim_s", "u_k_mps", "v_k_mps", "r_k_radps", "x_k_m",
    "y_k_m", "yaw_k_rad", "applied_throttle_norm_k1",
    "applied_steering_rad_k1", "simulator_feedback_steering_rad_k1",
    "wheel_speed_mps_k1", "segment_id", "u_k1_mps", "v_k1_mps",
    "r_k1_radps",
}


def _stats(values: Iterable[float]) -> dict[str, Any]:
    data = np.asarray([float(v) for v in values if math.isfinite(float(v))])
    if data.size == 0:
        return {"count": 0, "mae": None, "p95": None, "bias": None,
                "rmse": None}
    return {
        "count": int(data.size),
        "mae": float(np.mean(np.abs(data))),
        "p95": float(np.percentile(np.abs(data), 95)),
        "bias": float(np.mean(data)),
        "rmse": float(np.sqrt(np.mean(data * data))),
    }


def _read_rows(root: Path, names: list[str]) -> dict[str, list[dict[str, float]]]:
    runs: dict[str, list[dict[str, float]]] = {}
    for name in names:
        path = root / name / "assembled" / "model_transition_v4.csv"
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            missing = sorted(REQUIRED_FIELDS.difference(reader.fieldnames or ()))
            if missing:
                raise ValueError(f"{path} is missing fields: {missing}")
            rows: list[dict[str, float]] = []
            for raw in reader:
                row = {field: float(raw[field]) for field in REQUIRED_FIELDS}
                row["segment_id"] = int(round(row["segment_id"]))
                rows.append(row)
        runs[name] = rows
    return runs


def _resolve_parameters(lateral_report: Path, longitudinal_report: Path,
                        lateral_model: str, longitudinal_model: str,
                        damping_profile: str) -> PlantParameters:
    lateral = json.loads(lateral_report.read_text(encoding="utf-8"))
    longitudinal = json.loads(longitudinal_report.read_text(encoding="utf-8"))
    candidates = lateral.get("candidate_comparison", {})
    if lateral_model not in candidates:
        raise ValueError(f"unknown lateral model {lateral_model!r}")
    models = longitudinal.get("models", {})
    if longitudinal_model not in models:
        raise ValueError(f"unknown longitudinal model {longitudinal_model!r}")
    lateral_values = candidates[lateral_model]["parameters"]["parameters"]
    longitudinal_values = models[longitudinal_model]["parameters"]
    # This tool is report-driven by design: the caller selected these exact
    # candidate reports and the residual attribution must evaluate those
    # coefficients.  Canonical native replay resolves its defaults from the
    # manifest in replay_native_plant.py; silently switching this diagnostic
    # back to the manifest when the model names happen to match would hide a
    # newly fitted candidate (and can make stale parameters look current).
    parameters = PlantParameters.from_longitudinal_parameters(
        longitudinal_values).with_lateral(lateral_values)
    steering_dynamics = lateral.get("steering_integration", {}).get(
        "dynamics_kind", "rate_limited")
    parameters = PlantParameters(
        **{**parameters.__dict__,
           "steering_dynamics_kind": str(steering_dynamics)})
    if damping_profile == "unity_measured":
        overrides = {
            "coast_speed_drag_n_per_mps": 0.0,
            "linear_damping_per_s": 0.273,
            "angular_damping_per_s": 0.1,
        }
    elif damping_profile == "fitted":
        overrides = {"linear_damping_per_s": 0.0,
                     "angular_damping_per_s": 0.0}
    else:
        raise ValueError(f"unknown damping profile {damping_profile!r}")
    return PlantParameters(**{**parameters.__dict__, **overrides})


def _residual_rows(run_name: str, rows: list[dict[str, float]],
                   parameters: PlantParameters) -> list[dict[str, float | str]]:
    output: list[dict[str, float | str]] = []
    previous: dict[str, float] | None = None
    for row in rows:
        if previous is None or int(previous["segment_id"]) != int(row["segment_id"]):
            previous = row
            continue
        dt = row["dt_sim_s"]
        if not math.isfinite(dt) or dt <= 0.0:
            previous = row
            continue
        wheel = previous["wheel_speed_mps_k1"]
        steering = previous["simulator_feedback_steering_rad_k1"]
        throttle = max(0.0, min(1.0, row["applied_throttle_norm_k1"]))
        if not all(math.isfinite(value) for value in (wheel, steering, throttle)):
            previous = row
            continue

        steering_target = row["applied_steering_rad_k1"]
        steering_next = _steering_next(
            steering, steering_target, dt, parameters)
        wheel_next = _wheel_next(
            row["u_k_mps"], wheel, throttle, dt, parameters)
        # Score the same discrete transition used by the recursive plant.
        # Evaluating the force derivative only at ``steering`` (the beginning
        # of the sample) is wrong for a 40 Hz steering ramp: a reversal can
        # move through zero steering during this transition.  That old
        # approximation produced artificial 40--50 rad/s^2 yaw residuals.
        # ``step`` integrates the measured source dt with the same steering
        # ramp and wheel interpolation as the candidate plant.
        model_next = step(
            np.asarray([
                row["x_k_m"], row["y_k_m"], row["yaw_k_rad"],
                row["u_k_mps"], row["v_k_mps"], row["r_k_radps"],
                steering, wheel,
            ], dtype=float),
            steering_target / MAX_STEERING_RAD,
            throttle,
            dt,
            parameters,
        )
        model_u_dot = (model_next[3] - row["u_k_mps"]) / dt
        model_v_dot = (model_next[4] - row["v_k_mps"]) / dt
        model_r_dot = (model_next[5] - row["r_k_radps"]) / dt
        model_wheel_dot = (wheel_next - wheel) / dt
        model_steering_dot = (steering_next - steering) / dt
        measured = {
            "du_mps2": (row["u_k1_mps"] - row["u_k_mps"]) / dt,
            "dv_mps2": (row["v_k1_mps"] - row["v_k_mps"]) / dt,
            "dr_radps2": (row["r_k1_radps"] - row["r_k_radps"]) / dt,
            "wheel_dot_mps2": (row["wheel_speed_mps_k1"] - wheel) / dt,
            "steering_dot_radps": (
                row["simulator_feedback_steering_rad_k1"] - steering) / dt,
        }
        safe_u = math.copysign(
            max(abs(row["u_k_mps"]), MIN_SLIP_SPEED_MPS),
            row["u_k_mps"] or 1.0)
        alpha_front = steering - math.atan2(
            row["v_k_mps"] + parameters.lf_m * row["r_k_radps"], safe_u)
        alpha_rear = -math.atan2(
            row["v_k_mps"] - parameters.lr_m * row["r_k_radps"], safe_u)
        output.append({
            "run": run_name,
            "source_time_s": row["simulation_time_k_s"],
            "segment_id": int(row["segment_id"]),
            "dt_s": dt,
            "u_mps": row["u_k_mps"],
            "v_mps": row["v_k_mps"],
            "r_radps": row["r_k_radps"],
            "wheel_mps": wheel,
            "steering_rad": steering,
            "throttle_norm": throttle,
            "steering_target_rad": steering_target,
            "steering_rate_radps": measured["steering_dot_radps"],
            "model_steering_rate_radps": model_steering_dot,
            "wheel_rate_mps2": measured["wheel_dot_mps2"],
            "model_wheel_rate_mps2": model_wheel_dot,
            "front_slip_angle_rad": alpha_front,
            "rear_slip_angle_rad": alpha_rear,
            "measured_du_mps2": measured["du_mps2"],
            "model_du_mps2": model_u_dot,
            "residual_du_mps2": measured["du_mps2"] - model_u_dot,
            "measured_dv_mps2": measured["dv_mps2"],
            "model_dv_mps2": model_v_dot,
            "residual_dv_mps2": measured["dv_mps2"] - model_v_dot,
            "measured_dr_radps2": measured["dr_radps2"],
            "model_dr_radps2": model_r_dot,
            "residual_dr_radps2": measured["dr_radps2"] - model_r_dot,
        })
        previous = row
    return output


def _condition(value: float, bins: list[float]) -> str:
    for lower, upper in zip(bins, bins[1:]):
        if lower <= value < upper:
            return f"{lower:g}_{upper:g}"
    return f"{bins[-1]:g}_plus"


def _conditioned(rows: list[dict[str, float | str]], field: str,
                 bins: list[float], residual: str) -> dict[str, Any]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        value = (abs(float(row[field]))
                 if field in {"r_radps", "steering_rad", "steering_rate_radps"}
                 else float(row[field]))
        grouped.setdefault(_condition(value, bins), []).append(float(row[residual]))
    return {key: _stats(values) for key, values in sorted(grouped.items())}


FEATURES = {
    "r": lambda row: float(row["r_radps"]),
    "u_times_r": lambda row: float(row["u_mps"]) * float(row["r_radps"]),
    "abs_delta": lambda row: abs(float(row["steering_rad"])),
    "steering_rate": lambda row: float(row["steering_rate_radps"]),
    "u": lambda row: float(row["u_mps"]),
    "v": lambda row: float(row["v_mps"]),
    "throttle": lambda row: float(row["throttle_norm"]),
    "wheel_u": lambda row: float(row["wheel_mps"]) - float(row["u_mps"]),
    "front_slip": lambda row: float(row["front_slip_angle_rad"]),
    "rear_slip": lambda row: float(row["rear_slip_angle_rad"]),
    "throttle_abs_delta": lambda row: float(row["throttle_norm"]) * abs(float(row["steering_rad"])),
}


def _regressions(rows_by_run: dict[str, list[dict[str, float | str]]],
                 residual: str) -> dict[str, Any]:
    names = list(rows_by_run)
    baseline: list[float] = []
    for rows in rows_by_run.values():
        baseline.extend(float(row[residual]) for row in rows)
    result: dict[str, Any] = {"baseline_all_runs": _stats(baseline), "terms": {}}
    for name, function in FEATURES.items():
        folds: list[dict[str, Any]] = []
        for held_out in names:
            train = [row for run, rows in rows_by_run.items() if run != held_out for row in rows]
            test = rows_by_run[held_out]
            if len(train) < 10 or not test:
                continue
            x_train = np.asarray([function(row) for row in train], dtype=float)
            y_train = np.asarray([float(row[residual]) for row in train], dtype=float)
            x_test = np.asarray([function(row) for row in test], dtype=float)
            y_test = np.asarray([float(row[residual]) for row in test], dtype=float)
            design_train = np.column_stack((np.ones(len(x_train)), x_train))
            design_test = np.column_stack((np.ones(len(x_test)), x_test))
            coefficients, _, _, _ = np.linalg.lstsq(
                design_train, y_train, rcond=None)
            prediction = design_test @ coefficients
            folds.append({
                "held_out_run": held_out,
                "train_samples": int(len(train)),
                "test_samples": int(len(test)),
                "baseline_rmse": float(np.sqrt(np.mean(y_test * y_test))),
                "augmented_rmse": float(np.sqrt(np.mean((y_test - prediction) ** 2))),
                "coefficients": [float(value) for value in coefficients],
            })
        improvements = [
            (fold["baseline_rmse"] - fold["augmented_rmse"]) /
            max(fold["baseline_rmse"], 1.0e-12)
            for fold in folds
        ]
        result["terms"][name] = {
            "folds": folds,
            "mean_relative_rmse_improvement": (
                float(np.mean(improvements)) if improvements else None),
            "all_folds_improve": bool(improvements) and all(value > 0.0 for value in improvements),
        }
    return result


def analyze(accepted_root: Path, run_names: list[str], lateral_report: Path,
            longitudinal_report: Path, output: Path, residual_csv: Path,
            lateral_model: str, longitudinal_model: str,
            damping_profile: str) -> dict[str, Any]:
    parameters = _resolve_parameters(
        lateral_report, longitudinal_report, lateral_model,
        longitudinal_model, damping_profile)
    runs = _read_rows(accepted_root, run_names)
    residual_runs = {
        name: _residual_rows(name, rows, parameters)
        for name, rows in runs.items()
    }
    all_rows = [row for rows in residual_runs.values() for row in rows]
    fields = list(all_rows[0]) if all_rows else []
    residual_csv.parent.mkdir(parents=True, exist_ok=True)
    with residual_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)

    conditioning = {}
    for residual in ("residual_du_mps2", "residual_dv_mps2", "residual_dr_radps2"):
        conditioning[residual] = {
            "speed_mps": _conditioned(all_rows, "u_mps", [0.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0], residual),
            "abs_r_radps": _conditioned(all_rows, "r_radps", [0.0, 0.25, 0.5, 1.0, 2.0], residual),
            "abs_steering_rad": _conditioned(all_rows, "steering_rad", [0.0, 0.05, 0.15, 0.3, 0.6], residual),
            "throttle_norm": _conditioned(all_rows, "throttle_norm", [0.0, 0.01, 0.2, 0.5, 0.8, 1.0], residual),
            "abs_steering_rate_radps": _conditioned(all_rows, "steering_rate_radps", [0.0, 0.5, 1.5, 3.0, 8.0], residual),
        }
    report = {
        "schema_version": 1,
        "status": "offline_residual_attribution_not_runtime_validated",
        "ground_truth_use": "offline_derivative_residual_diagnostic_only",
        "run_names": run_names,
        "transition_count": len(all_rows),
        "plant_parameters": {
            "lateral_model": lateral_model,
            "longitudinal_model": longitudinal_model,
            "damping_profile": damping_profile,
            "resolved": {key: value for key, value in parameters.__dict__.items()
                          if key != "wheel_coefficients"},
            "wheel_coefficients": list(parameters.wheel_coefficients),
        },
        "residual_csv": str(residual_csv),
        "derivative_residuals": {
            key: _stats(float(row[key]) for row in all_rows)
            for key in ("residual_du_mps2", "residual_dv_mps2", "residual_dr_radps2")
        },
        "conditioned": conditioning,
        "leave_one_run_out_regression": {
            residual: _regressions(residual_runs, residual)
            for residual in ("residual_du_mps2", "residual_dv_mps2", "residual_dr_radps2")
        },
        "promotion": {
            "runtime_parameters_changed": False,
            "candidate_terms_require_held_out_improvement": True,
            "future_ground_truth_used_for_prediction": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted-root", type=Path, required=True)
    parser.add_argument("--run-names", required=True)
    parser.add_argument("--lateral-report", type=Path, required=True)
    parser.add_argument("--longitudinal-report", type=Path, required=True)
    parser.add_argument("--lateral-model", default="Y2_tanh_fixed_iz")
    parser.add_argument("--longitudinal-model", default="wheel_continuous")
    parser.add_argument("--damping-profile", choices=("unity_measured", "fitted"),
                        default="unity_measured")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--residual-csv", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(
        args.accepted_root,
        [value.strip() for value in args.run_names.split(",") if value.strip()],
        args.lateral_report, args.longitudinal_report, args.output,
        args.residual_csv, args.lateral_model, args.longitudinal_model,
        args.damping_profile)
    print(json.dumps({
        "output": str(args.output),
        "residual_csv": str(args.residual_csv),
        "transition_count": report["transition_count"],
        "derivative_residuals": report["derivative_residuals"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
