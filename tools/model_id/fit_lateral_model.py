#!/usr/bin/env python3
"""Fit and recursively score structured lateral vehicle candidates.

The fitter compares a kinematic baseline, a saturated linear dynamic bicycle,
and a low-parameter tanh tire model.  Steering is integrated as the measured
3.2 rad/s rate-limited ramp.  All recursive rollouts use only the initialized
state, recorded commands, and source ``dt``; simulator truth is used only for
offline fitting and final error scoring.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parent))
from structured_vehicle_plant import (  # noqa: E402
    IZ_KGM2,
    LF_M,
    LR_M,
    MAX_STEERING_RAD,
    PlantParameters,
    lateral_body_step,
    step,
)


MIN_DT_S = 0.015
MAX_DT_S = 0.035
HORIZONS_S = (0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00)


def _horizon_key(horizon: float) -> str:
    if abs(horizon - 0.025) < 1.0e-9:
        return "0.025s"
    return f"{horizon:.2f}s"


def _read_runs(root: Path, names: Sequence[str]) -> dict[str, list[dict[str, float]]]:
    required = {
        "dt_sim_s", "x_k_m", "y_k_m", "yaw_k_rad", "u_k_mps",
        "v_k_mps", "r_k_radps", "applied_throttle_norm_k1",
        "applied_steering_rad_k1", "simulator_feedback_steering_rad_k1",
        "wheel_speed_mps_k1", "segment_id", "x_k1_m", "y_k1_m",
        "yaw_k1_rad", "u_k1_mps", "v_k1_mps", "r_k1_radps",
    }
    result: dict[str, list[dict[str, float]]] = {}
    for name in names:
        path = root / name / "assembled" / "model_transition_v4.csv"
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            missing = sorted(required.difference(reader.fieldnames or ()))
            if missing:
                raise ValueError(f"{path} is missing fields: {missing}")
            rows: list[dict[str, float]] = []
            previous_segment: int | None = None
            previous_wheel = math.nan
            previous_feedback = math.nan
            for raw in reader:
                row = {
                    field: float(raw[field]) for field in required
                }
                if not all(math.isfinite(value) for value in row.values()):
                    raise ValueError(f"{path} contains non-finite transition data")
                dt = row["dt_sim_s"]
                if not MIN_DT_S <= dt <= MAX_DT_S:
                    raise ValueError(f"{path} contains dt outside source contract: {dt}")
                row["segment_id"] = int(round(row["segment_id"]))
                row["wheel_k_mps"] = (
                    previous_wheel if previous_segment == row["segment_id"] else math.nan)
                row["delta_k_rad"] = (
                    previous_feedback if previous_segment == row["segment_id"] else math.nan)
                rows.append(row)
                previous_segment = int(row["segment_id"])
                previous_wheel = row["wheel_speed_mps_k1"]
                previous_feedback = row["simulator_feedback_steering_rad_k1"]
        result[name] = rows
    return result


def _examples(runs: dict[str, list[dict[str, float]]]) -> list[dict[str, float]]:
    return [
        row for rows in runs.values() for row in rows
        if math.isfinite(row["delta_k_rad"])
    ]


def _candidate_parameters(kind: str, values: np.ndarray) -> dict[str, float | str]:
    if kind == "linear_saturated":
        return {
            "tire_model": kind,
            "cf_n_per_rad": float(values[0]),
            "cr_n_per_rad": float(values[1]),
            "iz_kgm2": float(values[2]),
            "df_n": 11.50,
            "dr_n": 10.60,
        }
    if kind == "tanh":
        return {
            "tire_model": kind,
            "cf_n_per_rad": float(values[0]),
            "cr_n_per_rad": float(values[1]),
            "df_n": float(values[2]),
            "dr_n": float(values[3]),
            "iz_kgm2": float(values[4]),
        }
    raise ValueError(f"unknown candidate: {kind}")


def _predict_lateral(row: dict[str, float], values: np.ndarray,
                     kind: str) -> tuple[float, float]:
    parameters = PlantParameters().with_lateral(
        _candidate_parameters(kind, values))
    return lateral_body_step(
        row["u_k_mps"], row["v_k_mps"], row["r_k_radps"],
        row["delta_k_rad"], row["applied_steering_rad_k1"],
        row["dt_sim_s"], parameters)


def _fit_candidate(
    rows: Sequence[dict[str, float]], kind: str,
    fixed_iz_kgm2: float | None = None,
) -> dict[str, Any]:
    if fixed_iz_kgm2 is not None and (
        not math.isfinite(fixed_iz_kgm2) or fixed_iz_kgm2 <= 0.0
    ):
        raise ValueError("fixed yaw inertia must be finite and positive")
    if kind == "linear_saturated":
        if fixed_iz_kgm2 is None:
            initial = np.asarray([800.0, 800.0, 0.035], dtype=float)
            lower = np.asarray([1.0, 1.0, 0.003])
            upper = np.asarray([5000.0, 5000.0, 0.080])
        else:
            initial = np.asarray([800.0, 800.0], dtype=float)
            lower = np.asarray([1.0, 1.0])
            upper = np.asarray([5000.0, 5000.0])
    else:
        if fixed_iz_kgm2 is None:
            initial = np.asarray([800.0, 800.0, 11.50, 10.60, 0.035], dtype=float)
            lower = np.asarray([1.0, 1.0, 2.0, 2.0, 0.003])
            upper = np.asarray([5000.0, 5000.0, 30.0, 30.0, 0.080])
        else:
            initial = np.asarray([800.0, 800.0, 11.50, 10.60], dtype=float)
            lower = np.asarray([1.0, 1.0, 2.0, 2.0])
            upper = np.asarray([5000.0, 5000.0, 30.0, 30.0])

    def parameter_values(values: np.ndarray) -> np.ndarray:
        if fixed_iz_kgm2 is None:
            return values
        if kind == "linear_saturated":
            return np.asarray([values[0], values[1], fixed_iz_kgm2])
        return np.asarray([
            values[0], values[1], values[2], values[3], fixed_iz_kgm2])

    def residual(values: np.ndarray) -> np.ndarray:
        output = np.empty(2 * len(rows), dtype=float)
        candidate_values = parameter_values(values)
        for index, row in enumerate(rows):
            predicted_v, predicted_r = _predict_lateral(
                row, candidate_values, kind)
            output[2 * index] = (predicted_v - row["v_k1_mps"]) / 0.05
            output[2 * index + 1] = (predicted_r - row["r_k1_radps"]) / 0.10
        return output

    result = least_squares(
        residual, initial, bounds=(lower, upper), loss="soft_l1",
        f_scale=1.0, x_scale="jac", max_nfev=160, verbose=0)
    return {
        "parameters": _candidate_parameters(kind, parameter_values(result.x)),
        "optimizer": {
            "success": bool(result.success),
            "message": str(result.message),
            "nfev": int(result.nfev),
            "cost": float(result.cost),
        },
    }


def _stats(values: Iterable[float]) -> dict[str, float | int | None]:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if finite.size == 0:
        return {"count": 0, "mae": None, "median": None, "p90": None,
                "p95": None, "p99": None, "max": None, "bias": None,
                "rmse": None}
    return {
        "count": int(finite.size),
        "mae": float(np.mean(np.abs(finite))),
        "median": float(np.median(np.abs(finite))),
        "p90": float(np.percentile(np.abs(finite), 90)),
        "p95": float(np.percentile(np.abs(finite), 95)),
        "p99": float(np.percentile(np.abs(finite), 99)),
        "max": float(np.max(np.abs(finite))),
        "bias": float(np.mean(finite)),
        "rmse": float(np.sqrt(np.mean(finite * finite))),
    }


def _one_step_scores(rows: Sequence[dict[str, float]], values: np.ndarray,
                     kind: str) -> dict[str, Any]:
    v_errors: list[float] = []
    r_errors: list[float] = []
    for row in rows:
        predicted_v, predicted_r = _predict_lateral(row, values, kind)
        v_errors.append(predicted_v - row["v_k1_mps"])
        r_errors.append(predicted_r - row["r_k1_radps"])
    return {"v_mps": _stats(v_errors), "r_radps": _stats(r_errors)}


def _kinematic_step(state: np.ndarray, steering_target_norm: float,
                    throttle_norm: float, dt: float,
                    parameters: PlantParameters) -> np.ndarray:
    """Kinematic baseline with the same actuator and wheel-state semantics."""
    result = state.copy()
    target = max(-1.0, min(1.0, steering_target_norm)) * parameters.max_steering_rad
    delta_end = state[6] + max(
        -parameters.steering_rate_radps * dt,
        min(parameters.steering_rate_radps * dt, target - state[6]))
    delta_mid = 0.5 * (state[6] + delta_end)
    speed = max(0.0, state[3])
    yaw_rate = speed * math.tan(delta_mid) / (parameters.lf_m + parameters.lr_m)
    result[0] += dt * speed * math.cos(state[2])
    result[1] += dt * speed * math.sin(state[2])
    result[2] += dt * yaw_rate
    result[4] = 0.0
    result[5] = yaw_rate
    result[6] = delta_end
    return result


def _error_vector(predicted: np.ndarray, row: dict[str, float]) -> dict[str, float]:
    heading = (predicted[2] - row["yaw_k1_rad"] + math.pi) % (2.0 * math.pi) - math.pi
    return {
        "x_m": predicted[0] - row["x_k1_m"],
        "y_m": predicted[1] - row["y_k1_m"],
        "position_m": math.hypot(predicted[0] - row["x_k1_m"],
                                  predicted[1] - row["y_k1_m"]),
        "heading_rad": heading,
        "u_mps": predicted[3] - row["u_k1_mps"],
        "v_mps": predicted[4] - row["v_k1_mps"],
        "r_radps": predicted[5] - row["r_k1_radps"],
        "steering_rad": predicted[6] - row["simulator_feedback_steering_rad_k1"],
        "wheel_mps": predicted[7] - row["wheel_speed_mps_k1"],
    }


def _recursive_scores(runs: dict[str, list[dict[str, float]]],
                      parameters: PlantParameters, kinematic: bool) -> dict[str, Any]:
    fields = ("x_m", "y_m", "position_m", "heading_rad", "u_mps", "v_mps",
              "r_radps", "steering_rad", "wheel_mps")
    output: dict[str, Any] = {}
    for horizon in HORIZONS_S:
        errors = {field: [] for field in fields}
        for rows in runs.values():
            for origin in range(len(rows)):
                first = rows[origin]
                if (not math.isfinite(first["wheel_k_mps"]) or
                        not math.isfinite(first["delta_k_rad"])):
                    continue
                state = np.asarray([
                    first["x_k_m"], first["y_k_m"], first["yaw_k_rad"],
                    first["u_k_mps"], first["v_k_mps"], first["r_k_radps"],
                    first["delta_k_rad"], first["wheel_k_mps"],
                ], dtype=float)
                segment = int(first["segment_id"])
                elapsed = 0.0
                index = origin
                while index < len(rows) and elapsed < horizon - 1.0e-10:
                    row = rows[index]
                    if int(row["segment_id"]) != segment:
                        break
                    steering_norm = row["applied_steering_rad_k1"] / MAX_STEERING_RAD
                    if kinematic:
                        state = _kinematic_step(
                            state, steering_norm,
                            row["applied_throttle_norm_k1"], row["dt_sim_s"],
                            parameters)
                    else:
                        state = step(
                            state, steering_norm,
                            row["applied_throttle_norm_k1"], row["dt_sim_s"],
                            parameters)
                    elapsed += row["dt_sim_s"]
                    index += 1
                if index == origin or elapsed < horizon - 1.0e-10:
                    continue
                row = rows[index - 1]
                for field, value in _error_vector(state, row).items():
                    errors[field].append(value)
        output[_horizon_key(horizon)] = {
            field: _stats(values) for field, values in errors.items()
        }
    return output


def _regime(row: dict[str, float]) -> str:
    speed = abs(row["u_k_mps"])
    steering = abs(row["applied_steering_rad_k1"])
    if speed < 0.5:
        return "low_speed"
    if steering < 0.01 and abs(row["applied_throttle_norm_k1"]) < 0.01:
        return "straight_coast"
    if steering < 0.01:
        return "straight_acceleration"
    if speed < 3.0:
        return "low_speed_corner"
    if steering * speed > 1.0:
        return "high_lateral_demand"
    return "cornering"


def _write_regime_csv(path: Path, rows: Sequence[dict[str, float]],
                      values: np.ndarray, kind: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["regime", "speed_mps", "steering_rad", "v_error_mps", "r_error_radps"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            predicted_v, predicted_r = _predict_lateral(row, values, kind)
            writer.writerow({
                "regime": _regime(row),
                "speed_mps": f"{row['u_k_mps']:.9g}",
                "steering_rad": f"{row['applied_steering_rad_k1']:.9g}",
                "v_error_mps": f"{predicted_v - row['v_k1_mps']:.9g}",
                "r_error_radps": f"{predicted_r - row['r_k1_radps']:.9g}",
            })


def fit(root: Path, train_names: Sequence[str], validation_names: Sequence[str],
        longitudinal_model: Path, output: Path, residual_csv: Path,
        fixed_iz_kgm2: float | None = None) -> dict[str, Any]:
    if set(train_names).intersection(validation_names):
        raise ValueError("training and validation runs overlap")
    train = _read_runs(root, train_names)
    validation = _read_runs(root, validation_names)
    train_rows = _examples(train)
    validation_rows = _examples(validation)
    if len(train_rows) < 100 or len(validation_rows) < 100:
        raise ValueError("insufficient lateral fitting transitions")

    longitudinal_report = json.loads(longitudinal_model.read_text(encoding="utf-8"))
    longitudinal = longitudinal_report["models"]["wheel_dynamic"]["parameters"]
    base_parameters = PlantParameters.from_longitudinal_parameters(longitudinal)
    candidates: dict[str, Any] = {
        "Y0_kinematic_bicycle": {
            "parameters": {"model": "kinematic_bicycle"},
            "train_recursive": _recursive_scores(train, base_parameters, True),
            "validation_recursive": _recursive_scores(validation, base_parameters, True),
        }
    }
    fitted: dict[str, tuple[np.ndarray, dict[str, Any]]] = {}
    candidate_suffix = "_fixed_iz" if fixed_iz_kgm2 is not None else ""
    for kind, base_name in (("linear_saturated", "Y1_linear_saturated"),
                            ("tanh", "Y2_tanh")):
        name = base_name + candidate_suffix
        fit_result = _fit_candidate(train_rows, kind, fixed_iz_kgm2)
        values = np.asarray([
            fit_result["parameters"][key]
            for key in (("cf_n_per_rad", "cr_n_per_rad", "iz_kgm2")
                        if kind == "linear_saturated" else
                        ("cf_n_per_rad", "cr_n_per_rad", "df_n", "dr_n",
                         "iz_kgm2"))
        ], dtype=float)
        fitted[kind] = (values, fit_result)
        parameters = base_parameters.with_lateral(fit_result["parameters"])
        candidates[name] = {
            "parameters": fit_result,
            "train_one_step": _one_step_scores(train_rows, values, kind),
            "validation_one_step": _one_step_scores(validation_rows, values, kind),
            "train_recursive": _recursive_scores(train, parameters, False),
            "validation_recursive": _recursive_scores(validation, parameters, False),
        }

    selected_name = (
        "Y2_tanh_fixed_iz"
        if fixed_iz_kgm2 is not None else "Y2_tanh")
    selected_kind = "tanh"
    selected_values, _ = fitted[selected_kind]
    _write_regime_csv(residual_csv, validation_rows, selected_values, selected_kind)
    report: dict[str, Any] = {
        "schema_version": 2,
        "status": "structured_lateral_candidate_not_runtime_validated",
        "ground_truth_use": "offline_identification_and_scoring_only",
        "recursive_prediction_uses_future_gt": False,
        "train_runs": list(train_names),
        "validation_runs": list(validation_names),
        "train_transition_count": len(train_rows),
        "validation_transition_count": len(validation_rows),
        "fixed_iz_kgm2": fixed_iz_kgm2,
        "state_definition": ["x_m", "y_m", "yaw_rad", "u_mps", "v_mps",
                              "r_radps", "steering_rad", "wheel_speed_mps"],
        "input_definition": ["steering_target_norm", "throttle_norm"],
        "steering_integration": {
            "rate_radps": 3.2,
            "substep_s": 0.002,
            "endpoint_target_semantics": "applied_physical_steering_rad",
        },
        "longitudinal_source_report": str(longitudinal_model),
        "candidate_comparison": candidates,
        "residual_by_regime_csv": str(residual_csv),
        "native_parity": {"status": "not_run"},
        "acceptance": {
            "two_percent_target": True,
            "required_horizons_s": list(HORIZONS_S),
            "recursive_open_loop_required": True,
            "repeatability_floor_required": True,
            "native_replay_required": True,
            "blind_run_required_after_model_freeze": True,
            "production_mpc_parameters_updated": False,
            "runtime_ground_truth_consumed": False,
        },
        "selected_for_next_full_plant_replay": selected_name,
        "next_action": (
            f"Use the causal {selected_name} candidate for composed Python plant "
            "replay, then port the same equations to native C and compare "
            "numerical parity."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return report


def _names(value: str) -> list[str]:
    names = [item.strip() for item in value.split(",") if item.strip()]
    if not names:
        raise ValueError("run list cannot be empty")
    return names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accepted-root", type=Path, required=True)
    parser.add_argument("--train-runs", required=True)
    parser.add_argument("--validation-runs", required=True)
    parser.add_argument("--longitudinal-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--residual-csv", type=Path, required=True)
    parser.add_argument(
        "--fixed-iz-kgm2", type=float, default=None,
        help="keep yaw inertia fixed during fitting instead of optimizing it")
    args = parser.parse_args()
    report = fit(
        args.accepted_root, _names(args.train_runs), _names(args.validation_runs),
        args.longitudinal_model, args.output, args.residual_csv,
        args.fixed_iz_kgm2)
    print(json.dumps({
        "output": str(args.output),
        "status": report["status"],
        "candidates": list(report["candidate_comparison"]),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
