#!/usr/bin/env python3
"""Fit and recursively score structured lateral vehicle candidates.

The fitter compares a kinematic baseline, a saturated linear dynamic bicycle,
and a low-parameter tanh tire model.  Steering follows the explicitly selected
measured transition contract (rate-limited or instantaneous).  All recursive
rollouts use only the initialized state, recorded commands, and source ``dt``;
simulator truth is used only for offline fitting and final error scoring.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
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
    _steering_next,
    lateral_body_step,
    step,
)


MIN_DT_S = 0.015
MAX_DT_S = 0.035
HORIZONS_S = (0.025, 0.05, 0.10, 0.25, 0.50, 0.75)
RECURSIVE_LATERAL_HORIZONS_S = (0.25, 0.50, 0.75)


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
    if kind == "speed_combined_tanh":
        return {
            "tire_model": kind,
            "cf_n_per_rad": float(values[0]),
            "cr_n_per_rad": float(values[1]),
            "df_n": float(values[2]),
            "dr_n": float(values[3]),
            "lateral_speed_stiffness_gain": float(values[4]),
            "lateral_speed_peak_gain": float(values[5]),
            "combined_slip_gain": float(values[6]),
            "iz_kgm2": float(values[7]),
        }
    raise ValueError(f"unknown candidate: {kind}")


def _predict_lateral(row: dict[str, float], values: np.ndarray,
                     kind: str,
                     steering_dynamics_kind: str = "rate_limited") -> tuple[float, float]:
    parameters = PlantParameters().with_lateral(
        _candidate_parameters(kind, values))
    parameters = replace(
        parameters, steering_dynamics_kind=steering_dynamics_kind)
    return lateral_body_step(
        row["u_k_mps"], row["v_k_mps"], row["r_k_radps"],
        row["delta_k_rad"], row["applied_steering_rad_k1"],
        row["dt_sim_s"], parameters, wheel=row["wheel_k_mps"],
        throttle=row["applied_throttle_norm_k1"])


def _recursive_lateral_windows(
        runs: dict[str, list[dict[str, float]]],
        origin_stride: int = 10,
        min_speed_mps: float = 1.0,
        max_speed_mps: float = 16.0,
        max_abs_steering_rad: float = 0.45,
        max_abs_lateral_accel_mps2: float = 14.0,
        maximum_horizon_s: float = 0.75,
) -> list[list[dict[str, float]]]:
    """Select contiguous, physically reachable turn windows for fitting.

    The lateral fit is deliberately not driven by arbitrary full-lock,
    full-speed commands.  This selector keeps the speed envelope at the
    project limit and rejects windows whose measured lateral acceleration is
    outside the intended raceline regime.  Measured ``u`` and wheel speed are
    used only as offline conditioning variables while identifying lateral
    dynamics; recursive validation still uses the complete causal plant.
    """
    if origin_stride < 1:
        raise ValueError("recursive origin stride must be positive")
    if min_speed_mps < 0.0 or max_speed_mps <= min_speed_mps:
        raise ValueError("invalid recursive speed envelope")
    if max_abs_steering_rad <= 0.0 or max_abs_lateral_accel_mps2 <= 0.0:
        raise ValueError("recursive lateral limits must be positive")
    needed = max(1, int(math.ceil(maximum_horizon_s / MIN_DT_S)) + 2)
    windows: list[list[dict[str, float]]] = []
    for rows in runs.values():
        for origin in range(1, len(rows) - needed, origin_stride):
            first = rows[origin]
            if (not math.isfinite(first["wheel_k_mps"]) or
                    not math.isfinite(first["delta_k_rad"])):
                continue
            window = rows[origin:origin + needed]
            if any(
                    int(row["segment_id"]) != int(first["segment_id"]) or
                    not (min_speed_mps <= row["u_k_mps"] <= max_speed_mps) or
                    abs(row["applied_steering_rad_k1"]) > max_abs_steering_rad or
                    abs(row["u_k_mps"] * row["r_k_radps"]) >
                    max_abs_lateral_accel_mps2 or
                    not math.isfinite(row["wheel_k_mps"])
                    for row in window):
                continue
            windows.append(window)
    if not windows:
        raise ValueError("recursive lateral regime selected no windows")
    return windows


def _recursive_lateral_errors(
        windows: Sequence[Sequence[dict[str, float]]],
        values: np.ndarray,
        kind: str,
        steering_dynamics_kind: str,
        v_scale: float = 0.05,
        r_scale: float = 0.05,
) -> np.ndarray:
    """Return causal v/r residuals at multi-step lateral horizons."""
    parameters = replace(
        PlantParameters().with_lateral(_candidate_parameters(kind, values)),
        steering_dynamics_kind=steering_dynamics_kind)
    errors: list[float] = []
    for rows in windows:
        current_v = rows[0]["v_k_mps"]
        current_r = rows[0]["r_k_radps"]
        current_delta = rows[0]["delta_k_rad"]
        elapsed = 0.0
        horizon_index = 0
        for row in rows:
            current_v, current_r = lateral_body_step(
                row["u_k_mps"], current_v, current_r, current_delta,
                row["applied_steering_rad_k1"], row["dt_sim_s"],
                parameters, wheel=row["wheel_k_mps"],
                throttle=row["applied_throttle_norm_k1"])
            current_delta = _steering_next(
                current_delta, row["applied_steering_rad_k1"],
                row["dt_sim_s"], parameters)
            elapsed += row["dt_sim_s"]
            while (horizon_index < len(RECURSIVE_LATERAL_HORIZONS_S) and
                   elapsed + 1.0e-9 >=
                   RECURSIVE_LATERAL_HORIZONS_S[horizon_index]):
                errors.extend([
                    (current_v - row["v_k1_mps"]) / v_scale,
                    (current_r - row["r_k1_radps"]) / r_scale,
                ])
                horizon_index += 1
            if horizon_index == len(RECURSIVE_LATERAL_HORIZONS_S):
                break
    return np.asarray(errors, dtype=float)


def _fit_candidate(
    rows: Sequence[dict[str, float]], kind: str,
    fixed_iz_kgm2: float | None = None,
    steering_dynamics_kind: str = "rate_limited",
    fit_objective: str = "one_step",
    recursive_windows: Sequence[Sequence[dict[str, float]]] | None = None,
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
    elif kind == "tanh":
        if fixed_iz_kgm2 is None:
            initial = np.asarray([800.0, 800.0, 11.50, 10.60, 0.035], dtype=float)
            lower = np.asarray([1.0, 1.0, 2.0, 2.0, 0.003])
            upper = np.asarray([5000.0, 5000.0, 30.0, 30.0, 0.080])
        else:
            initial = np.asarray([800.0, 800.0, 11.50, 10.60], dtype=float)
            lower = np.asarray([1.0, 1.0, 2.0, 2.0])
            upper = np.asarray([5000.0, 5000.0, 30.0, 30.0])
    elif kind == "speed_combined_tanh":
        if fixed_iz_kgm2 is None:
            initial = np.asarray(
                [3000.0, 5000.0, 13.0, 15.0, 0.0, 0.0, 1.0, 0.035],
                dtype=float)
            lower = np.asarray(
                [1.0, 1.0, 2.0, 2.0, -1.0, -1.0, 0.0, 0.003])
            upper = np.asarray(
                [10000.0, 10000.0, 40.0, 40.0, 2.0, 2.0, 20.0, 0.080])
        else:
            initial = np.asarray(
                [3000.0, 5000.0, 13.0, 15.0, 0.0, 0.0, 1.0],
                dtype=float)
            lower = np.asarray([1.0, 1.0, 2.0, 2.0, -1.0, -1.0, 0.0])
            upper = np.asarray(
                [10000.0, 10000.0, 40.0, 40.0, 2.0, 2.0, 20.0])
    else:
        raise ValueError(f"unknown candidate: {kind}")

    def parameter_values(values: np.ndarray) -> np.ndarray:
        if fixed_iz_kgm2 is None:
            return values
        if kind == "linear_saturated":
            return np.asarray([values[0], values[1], fixed_iz_kgm2])
        if kind == "tanh":
            return np.asarray([
            values[0], values[1], values[2], values[3], fixed_iz_kgm2])
        return np.asarray([
            values[0], values[1], values[2], values[3], values[4],
            values[5], values[6], fixed_iz_kgm2])

    def residual(values: np.ndarray) -> np.ndarray:
        candidate_values = parameter_values(values)
        if fit_objective == "recursive_raceline":
            if recursive_windows is None:
                raise ValueError("recursive fit objective has no windows")
            return _recursive_lateral_errors(
                recursive_windows, candidate_values, kind,
                steering_dynamics_kind)
        if fit_objective != "one_step":
            raise ValueError(f"unsupported lateral fit objective: {fit_objective}")
        output = np.empty(2 * len(rows), dtype=float)
        for index, row in enumerate(rows):
            predicted_v, predicted_r = _predict_lateral(
                row, candidate_values, kind, steering_dynamics_kind)
            output[2 * index] = (predicted_v - row["v_k1_mps"]) / 0.05
            output[2 * index + 1] = (predicted_r - row["r_k1_radps"]) / 0.10
        return output

    result = least_squares(
        residual, initial, bounds=(lower, upper), loss="soft_l1",
        f_scale=1.0, x_scale="jac",
        max_nfev=120 if fit_objective == "recursive_raceline" else 160,
        verbose=0)
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
                     kind: str,
                     steering_dynamics_kind: str = "rate_limited") -> dict[str, Any]:
    v_errors: list[float] = []
    r_errors: list[float] = []
    for row in rows:
        predicted_v, predicted_r = _predict_lateral(
            row, values, kind, steering_dynamics_kind)
        v_errors.append(predicted_v - row["v_k1_mps"])
        r_errors.append(predicted_r - row["r_k1_radps"])
    return {"v_mps": _stats(v_errors), "r_radps": _stats(r_errors)}


def _kinematic_step(state: np.ndarray, steering_target_norm: float,
                    throttle_norm: float, dt: float,
                    parameters: PlantParameters) -> np.ndarray:
    """Kinematic baseline with the same actuator and wheel-state semantics."""
    result = state.copy()
    target = max(-1.0, min(1.0, steering_target_norm)) * parameters.max_steering_rad
    delta_end = _steering_next(state[6], target, dt, parameters)
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
                      values: np.ndarray, kind: str,
                      steering_dynamics_kind: str = "rate_limited") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["regime", "speed_mps", "steering_rad", "v_error_mps", "r_error_radps"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            predicted_v, predicted_r = _predict_lateral(
                row, values, kind, steering_dynamics_kind)
            writer.writerow({
                "regime": _regime(row),
                "speed_mps": f"{row['u_k_mps']:.9g}",
                "steering_rad": f"{row['applied_steering_rad_k1']:.9g}",
                "v_error_mps": f"{predicted_v - row['v_k1_mps']:.9g}",
                "r_error_radps": f"{predicted_r - row['r_k1_radps']:.9g}",
            })


def fit(root: Path, train_names: Sequence[str], validation_names: Sequence[str],
        longitudinal_model: Path, output: Path, residual_csv: Path,
        fixed_iz_kgm2: float | None = None,
        steering_dynamics_kind: str = "rate_limited",
        candidate_kinds: Sequence[str] | None = None,
        include_train_recursive: bool = True,
        longitudinal_model_key: str = "wheel_continuous",
        fit_objective: str = "one_step",
        recursive_origin_stride: int = 10,
        recursive_min_speed_mps: float = 1.0,
        recursive_max_speed_mps: float = 16.0,
        recursive_max_abs_steering_rad: float = 0.45,
        recursive_max_abs_lateral_accel_mps2: float = 14.0) -> dict[str, Any]:
    if steering_dynamics_kind not in {"rate_limited", "instantaneous"}:
        raise ValueError(
            f"unsupported steering dynamics: {steering_dynamics_kind}")
    if fit_objective not in {"one_step", "recursive_raceline"}:
        raise ValueError(f"unsupported lateral fit objective: {fit_objective}")
    all_candidate_kinds = (
        "linear_saturated", "tanh", "speed_combined_tanh")
    selected_candidate_kinds = tuple(
        candidate_kinds if candidate_kinds is not None else all_candidate_kinds)
    unknown = set(selected_candidate_kinds).difference(all_candidate_kinds)
    if unknown:
        raise ValueError(f"unsupported candidate kinds: {sorted(unknown)}")
    if "tanh" not in selected_candidate_kinds:
        raise ValueError("the Y2 tanh candidate must be included")
    if set(train_names).intersection(validation_names):
        raise ValueError("training and validation runs overlap")
    train = _read_runs(root, train_names)
    validation = _read_runs(root, validation_names)
    train_rows = _examples(train)
    validation_rows = _examples(validation)
    if len(train_rows) < 100 or len(validation_rows) < 100:
        raise ValueError("insufficient lateral fitting transitions")
    recursive_windows = None
    if fit_objective == "recursive_raceline":
        recursive_windows = _recursive_lateral_windows(
            train, origin_stride=recursive_origin_stride,
            min_speed_mps=recursive_min_speed_mps,
            max_speed_mps=recursive_max_speed_mps,
            max_abs_steering_rad=recursive_max_abs_steering_rad,
            max_abs_lateral_accel_mps2=recursive_max_abs_lateral_accel_mps2)

    longitudinal_report = json.loads(longitudinal_model.read_text(encoding="utf-8"))
    try:
        longitudinal = longitudinal_report["models"][longitudinal_model_key][
            "parameters"]
    except KeyError as exc:
        available = sorted(longitudinal_report.get("models", {}).keys())
        raise ValueError(
            f"longitudinal model {longitudinal_model_key!r} is unavailable; "
            f"available models: {available}") from exc
    base_parameters = replace(
        PlantParameters.from_longitudinal_parameters(longitudinal),
        steering_dynamics_kind=steering_dynamics_kind)
    candidates: dict[str, Any] = {
        "Y0_kinematic_bicycle": {
            "parameters": {"model": "kinematic_bicycle"},
            "train_recursive": _recursive_scores(train, base_parameters, True),
            "validation_recursive": _recursive_scores(validation, base_parameters, True),
        }
    }
    fitted: dict[str, tuple[np.ndarray, dict[str, Any]]] = {}
    candidate_suffix = "_fixed_iz" if fixed_iz_kgm2 is not None else ""
    candidate_specs = (("linear_saturated", "Y1_linear_saturated"),
                       ("tanh", "Y2_tanh"),
                       ("speed_combined_tanh", "Y3_speed_combined_tanh"))
    for kind, base_name in candidate_specs:
        if kind not in selected_candidate_kinds:
            continue
        name = base_name + candidate_suffix
        fit_result = _fit_candidate(
            train_rows, kind, fixed_iz_kgm2, steering_dynamics_kind,
            fit_objective, recursive_windows)
        parameter_keys = {
            "linear_saturated": ("cf_n_per_rad", "cr_n_per_rad", "iz_kgm2"),
            "tanh": ("cf_n_per_rad", "cr_n_per_rad", "df_n", "dr_n",
                     "iz_kgm2"),
            "speed_combined_tanh": (
                "cf_n_per_rad", "cr_n_per_rad", "df_n", "dr_n",
                "lateral_speed_stiffness_gain", "lateral_speed_peak_gain",
                "combined_slip_gain", "iz_kgm2"),
        }[kind]
        values = np.asarray([
            fit_result["parameters"][key] for key in parameter_keys
        ], dtype=float)
        fitted[kind] = (values, fit_result)
        parameters = base_parameters.with_lateral(fit_result["parameters"])
        candidates[name] = {
            "parameters": fit_result,
            "train_one_step": _one_step_scores(
                train_rows, values, kind, steering_dynamics_kind),
            "validation_one_step": _one_step_scores(
                validation_rows, values, kind, steering_dynamics_kind),
            "train_recursive": (
                _recursive_scores(train, parameters, False)
                if include_train_recursive else {"status": "skipped"}),
            "validation_recursive": _recursive_scores(validation, parameters, False),
        }

    selected_name = (
        "Y2_tanh_fixed_iz"
        if fixed_iz_kgm2 is not None else "Y2_tanh")
    selected_kind = "tanh"
    selected_values, _ = fitted[selected_kind]
    _write_regime_csv(
        residual_csv, validation_rows, selected_values, selected_kind,
        steering_dynamics_kind)
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
            "rate_radps": 3.2 if steering_dynamics_kind == "rate_limited" else None,
            "dynamics_kind": steering_dynamics_kind,
            "substep_s": 0.002,
            "endpoint_target_semantics": "applied_physical_steering_rad",
        },
        "longitudinal_source_report": str(longitudinal_model),
        "longitudinal_source_model_key": longitudinal_model_key,
        "fit_objective": fit_objective,
        "recursive_lateral_regime": ({
            "window_count": len(recursive_windows),
            "origin_stride": recursive_origin_stride,
            "min_speed_mps": recursive_min_speed_mps,
            "max_speed_mps": recursive_max_speed_mps,
            "max_abs_steering_rad": recursive_max_abs_steering_rad,
            "max_abs_lateral_accel_mps2": recursive_max_abs_lateral_accel_mps2,
            "horizons_s": list(RECURSIVE_LATERAL_HORIZONS_S),
            "measured_longitudinal_state_use": "offline_fit_only",
        } if recursive_windows is not None else None),
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
        "candidate_kinds_evaluated": [
            "kinematic_bicycle", *selected_candidate_kinds],
        "train_recursive_scoring": (
            "included" if include_train_recursive else "skipped_for_targeted_holdout"),
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
    parser.add_argument(
        "--steering-dynamics", choices=("rate_limited", "instantaneous"),
        default="rate_limited",
        help="identified applied-steering transition used by the offline plant")
    parser.add_argument(
        "--candidate-kinds", default="linear_saturated,tanh,speed_combined_tanh",
        help="comma-separated candidate kinds to evaluate")
    parser.add_argument(
        "--skip-train-recursive", action="store_true",
        help="skip exhaustive training recursive scoring for a holdout-only run")
    parser.add_argument(
        "--longitudinal-model-key", default="wheel_continuous",
        help=("model entry to use from the longitudinal report; the default "
              "matches the canonical continuous-wheel candidate"))
    parser.add_argument(
        "--fit-objective", choices=("one_step", "recursive_raceline"),
        default="one_step",
        help="optimize one-step dynamics or multi-step reachable-turn errors")
    parser.add_argument("--recursive-origin-stride", type=int, default=10)
    parser.add_argument("--recursive-min-speed-mps", type=float, default=1.0)
    parser.add_argument("--recursive-max-speed-mps", type=float, default=16.0)
    parser.add_argument(
        "--recursive-max-abs-steering-rad", type=float, default=0.45)
    parser.add_argument(
        "--recursive-max-abs-lateral-accel-mps2", type=float, default=14.0)
    args = parser.parse_args()
    report = fit(
        args.accepted_root, _names(args.train_runs), _names(args.validation_runs),
        args.longitudinal_model, args.output, args.residual_csv,
        args.fixed_iz_kgm2, args.steering_dynamics,
        _names(args.candidate_kinds), not args.skip_train_recursive,
        args.longitudinal_model_key, args.fit_objective,
        args.recursive_origin_stride, args.recursive_min_speed_mps,
        args.recursive_max_speed_mps, args.recursive_max_abs_steering_rad,
        args.recursive_max_abs_lateral_accel_mps2)
    print(json.dumps({
        "output": str(args.output),
        "status": report["status"],
        "candidates": list(report["candidate_comparison"]),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
