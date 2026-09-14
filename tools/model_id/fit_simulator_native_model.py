#!/usr/bin/env python3
"""Fit and score the simulator-native MPC plant candidates offline.

This is the first comparison in the current model-fitting strategy.  It
keeps the documented wheel-friction shape and compares multiple contact
representations:

* ``VS1``: one virtual front/rear contact per axle;
* ``VS2``: four contacts with Ackermann steering and wheel kinematics;
* ``VS2.5``: four-contact normalized body force and yaw-moment basis.
* ``VS2.75``: split front/rear lateral and geometric yaw-moment bases.

The dynamic coefficients are *effective simulator gains*.  They are not
reported as physical cornering stiffness or yaw inertia.  In particular, the
candidate deliberately does not invent an ``I_z`` when the Unity diagnostic
dump is unavailable.

All simulator truth in this program is offline fitting/scoring data.  The
recursive evaluator starts from a measured origin and then propagates only
the candidate state plus recorded applied controls and source timing.  Future
measured states are never fed back into a rollout.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from simulator_native_model import (  # noqa: E402
    MAX_STEERING_RAD,
    NativeModelParameters,
    _effective_coordinates,
    _moment_basis_coordinates,
    _split_moment_basis_coordinates,
    f1tenth_prefab_parameters,
    step,
)


MIN_DT_S = 0.015
MAX_DT_S = 0.035
HORIZONS_S = (0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00)
MIXED_FIT_HORIZONS_S = (0.10, 0.25, 0.50, 1.00)
MIXED_FIT_STATE_SCALES = np.asarray(
    (0.50, 0.50, 0.20, 0.35, 0.60, 1.00), dtype=float)
MIXED_FIT_ONE_STEP_SCALES = np.asarray(
    (0.10, 0.10, 0.05, 0.30, 0.30, 0.75), dtype=float)
EFFECTIVE_GAIN_FIELDS = (
    "effective_longitudinal_gain_mps2",
    "effective_drag_linear_per_s",
    "effective_drag_quadratic_per_m",
    "effective_lateral_front_mps2",
    "effective_lateral_rear_mps2",
    "effective_yaw_front_per_s2",
    "effective_yaw_rear_per_s2",
)
MOMENT_BASIS_GAIN_FIELDS = (
    "effective_force_x_gain_mps2",
    "effective_drag_linear_per_s",
    "effective_drag_quadratic_per_m",
    "effective_force_y_gain_mps2",
    "effective_yaw_moment_gain_per_s2",
    "effective_yaw_damping_per_s",
)
SPLIT_MOMENT_BASIS_GAIN_FIELDS = (
    "effective_force_x_gain_mps2",
    "effective_drag_linear_per_s",
    "effective_drag_quadratic_per_m",
    "effective_force_y_front_gain_mps2",
    "effective_force_y_rear_gain_mps2",
    "effective_yaw_lateral_front_gain_per_s2",
    "effective_yaw_lateral_rear_gain_per_s2",
    "effective_yaw_longitudinal_gain_per_s2",
)
STATE_FIELDS = ("x_m", "y_m", "position_m", "heading_rad", "u_mps",
                "v_mps", "r_radps", "steering_rad", "wheel_mps",
                "wheel_left_mps", "wheel_right_mps")


def _gain_fields(contact_model: str) -> tuple[str, ...]:
    if contact_model == "moment_basis":
        return MOMENT_BASIS_GAIN_FIELDS
    if contact_model == "moment_basis_split":
        return SPLIT_MOMENT_BASIS_GAIN_FIELDS
    return EFFECTIVE_GAIN_FIELDS


def _horizon_key(horizon: float) -> str:
    if abs(horizon - 0.025) < 1.0e-9:
        return "0.025s"
    return f"{horizon:.2f}s"


def _stats(values: Iterable[float]) -> dict[str, float | int | None]:
    finite = np.asarray([value for value in values if math.isfinite(value)],
                        dtype=float)
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


def _read_runs(root: Path, names: Sequence[str]) -> dict[str, list[dict[str, float]]]:
    required = {
        "dt_sim_s", "x_k_m", "y_k_m", "yaw_k_rad", "u_k_mps",
        "v_k_mps", "r_k_radps", "applied_throttle_norm_k1",
        "applied_steering_rad_k1", "simulator_feedback_steering_rad_k1",
        "wheel_speed_mps_k1", "segment_id", "x_k1_m", "y_k1_m",
        "yaw_k1_rad", "u_k1_mps", "v_k1_mps", "r_k1_radps",
        "encoder_left_rad_k1", "encoder_right_rad_k1",
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
            previous_left_speed = math.nan
            previous_right_speed = math.nan
            previous_left_angle = math.nan
            previous_right_angle = math.nan
            previous_feedback = math.nan
            for raw in reader:
                row = {field: float(raw[field]) for field in required}
                if not all(math.isfinite(value) for value in row.values()):
                    raise ValueError(f"{path} contains non-finite transition data")
                dt = row["dt_sim_s"]
                if not MIN_DT_S <= dt <= MAX_DT_S:
                    raise ValueError(f"{path} contains dt outside source contract: {dt}")
                segment = int(round(row["segment_id"]))
                row["segment_id"] = float(segment)
                row["wheel_k_mps"] = (
                    previous_wheel if previous_segment == segment else math.nan)
                row["wheel_left_k_mps"] = (
                    previous_left_speed if previous_segment == segment else math.nan)
                row["wheel_right_k_mps"] = (
                    previous_right_speed if previous_segment == segment else math.nan)
                row["wheel_left_speed_mps_k1"] = math.nan
                row["wheel_right_speed_mps_k1"] = math.nan
                if (previous_segment == segment and
                        math.isfinite(previous_left_angle) and
                        math.isfinite(previous_right_angle)):
                    dt = row["dt_sim_s"]
                    row["wheel_left_speed_mps_k1"] = (
                        0.059 * (row["encoder_left_rad_k1"] -
                                 previous_left_angle) / dt)
                    row["wheel_right_speed_mps_k1"] = (
                        0.059 * (row["encoder_right_rad_k1"] -
                                 previous_right_angle) / dt)
                # The assembled table starts at the first post-transition
                # packet, so its first encoder derivative has no predecessor
                # row.  Preserve the already-derived mean speed only for this
                # boundary sample; never use it to replace valid side data.
                if not math.isfinite(row["wheel_left_speed_mps_k1"]):
                    row["wheel_left_speed_mps_k1"] = row["wheel_speed_mps_k1"]
                if not math.isfinite(row["wheel_right_speed_mps_k1"]):
                    row["wheel_right_speed_mps_k1"] = row["wheel_speed_mps_k1"]
                row["delta_k_rad"] = (
                    previous_feedback if previous_segment == segment else math.nan)
                rows.append(row)
                previous_segment = segment
                previous_wheel = row["wheel_speed_mps_k1"]
                previous_left_speed = row["wheel_left_speed_mps_k1"]
                previous_right_speed = row["wheel_right_speed_mps_k1"]
                previous_left_angle = row["encoder_left_rad_k1"]
                previous_right_angle = row["encoder_right_rad_k1"]
                previous_feedback = row["simulator_feedback_steering_rad_k1"]
        result[name] = rows
    return result


def _fit_examples(runs: dict[str, list[dict[str, float]]],
                  contact_model: str,
                  parameters: NativeModelParameters) -> dict[str, np.ndarray | int]:
    """Build causal feature matrices and derivative targets.

    Midpoint state values are used only as a numerical discretization of the
    measured transition.  The row's k+1 state is not used as a rollout input;
    it is used here solely to form the offline derivative target.
    """
    q_values: list[tuple[float, ...]] = []
    longitudinal_targets: list[float] = []
    lateral_targets: list[float] = []
    yaw_targets: list[float] = []
    yaw_rates: list[float] = []
    longitudinal_speeds: list[float] = []
    usable_rows = 0
    for rows in runs.values():
        for row in rows:
            if not math.isfinite(row["wheel_k_mps"]):
                continue
            state = np.asarray([
                row["x_k_m"], row["y_k_m"], row["yaw_k_rad"],
                row["u_k_mps"], row["v_k_mps"], row["r_k_radps"],
                row["delta_k_rad"], row["wheel_left_k_mps"],
                row["wheel_right_k_mps"],
            ], dtype=float)
            state_mid = state.copy()
            state_mid[3] = 0.5 * (row["u_k_mps"] + row["u_k1_mps"])
            state_mid[4] = 0.5 * (row["v_k_mps"] + row["v_k1_mps"])
            state_mid[5] = 0.5 * (row["r_k_radps"] + row["r_k1_radps"])
            state_mid[6] = 0.5 * (row["delta_k_rad"] +
                                   row["simulator_feedback_steering_rad_k1"])
            state_mid[7] = 0.5 * (
                row["wheel_left_k_mps"] + row["wheel_left_speed_mps_k1"])
            state_mid[8] = 0.5 * (
                row["wheel_right_k_mps"] + row["wheel_right_speed_mps_k1"])
            if contact_model in ("moment_basis", "moment_basis_split"):
                if contact_model == "moment_basis_split":
                    coordinates = _split_moment_basis_coordinates(
                        state_mid, state_mid[6], parameters)
                else:
                    coordinates = _moment_basis_coordinates(
                        state_mid, state_mid[6], parameters)
            else:
                coordinates = _effective_coordinates(
                    state_mid, state_mid[6], parameters)
            dt = row["dt_sim_s"]
            u_mid = state_mid[3]
            v_mid = state_mid[4]
            r_mid = state_mid[5]
            q_values.append(coordinates)
            longitudinal_targets.append(
                (row["u_k1_mps"] - row["u_k_mps"]) / dt - r_mid * v_mid)
            lateral_targets.append(
                (row["v_k1_mps"] - row["v_k_mps"]) / dt + r_mid * u_mid)
            yaw_targets.append((row["r_k1_radps"] - row["r_k_radps"]) / dt)
            yaw_rates.append(r_mid)
            longitudinal_speeds.append(u_mid)
            usable_rows += 1

    if usable_rows < 100:
        raise ValueError(f"only {usable_rows} causal fitting transitions available")
    q = np.asarray(q_values, dtype=float)
    u = np.asarray(longitudinal_speeds, dtype=float)
    return {
        "coordinates": q,
        "longitudinal_features": np.column_stack((q[:, 0], u, u * np.abs(u))),
        "longitudinal_target": np.asarray(longitudinal_targets, dtype=float),
        "lateral_features": q[:, 1:3],
        "lateral_target": np.asarray(lateral_targets, dtype=float),
        "yaw_features": (
            np.column_stack((q[:, 2], -np.asarray(yaw_rates)))
            if contact_model == "moment_basis" else
            np.asarray(q[:, 3:6], dtype=float)
            if contact_model == "moment_basis_split" else
            np.column_stack((q[:, 1], -q[:, 2]))),
        "yaw_target": np.asarray(yaw_targets, dtype=float),
        "samples": usable_rows,
    }


def _robust_linear_fit(features: np.ndarray, target: np.ndarray,
                       bounds: tuple[np.ndarray, np.ndarray] | None = None,
                       sample_weights: np.ndarray | None = None,
                       huber_scale: float = 1.0
                       ) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit a small linear regression with Huber-like IRLS weighting."""
    if features.ndim != 2 or target.ndim != 1 or len(features) != len(target):
        raise ValueError("feature and target shapes do not agree")
    if not np.all(np.isfinite(features)) or not np.all(np.isfinite(target)):
        raise ValueError("non-finite fit data")
    if not math.isfinite(huber_scale) or huber_scale <= 0.0:
        raise ValueError("huber_scale must be positive and finite")
    if sample_weights is None:
        base_weights = np.ones(len(target), dtype=float)
    else:
        base_weights = np.asarray(sample_weights, dtype=float)
        if base_weights.shape != (len(target),) or not np.all(np.isfinite(base_weights)):
            raise ValueError("sample weights have the wrong shape or are non-finite")
        if np.any(base_weights <= 0.0):
            raise ValueError("sample weights must be positive")
    scale = np.maximum(np.sqrt(np.mean(features * features, axis=0)), 1.0e-8)
    scaled = features / scale
    weights = base_weights.copy()
    theta_scaled = np.zeros(features.shape[1], dtype=float)
    for _ in range(8):
        weighted = scaled * weights[:, None]
        if bounds is None:
            lhs = scaled.T @ weighted + 1.0e-8 * np.eye(scaled.shape[1])
            rhs = scaled.T @ (weights * target)
            theta_scaled = np.linalg.solve(lhs, rhs)
        else:
            from scipy.optimize import lsq_linear
            lower, upper = bounds
            scaled_bounds = (lower * scale, upper * scale)
            result = lsq_linear(
                scaled * np.sqrt(weights)[:, None],
                target * np.sqrt(weights),
                bounds=scaled_bounds, lsmr_tol="auto", max_iter=200)
            if not result.success:
                raise RuntimeError(f"bounded regression failed: {result.message}")
            theta_scaled = result.x
        residual = target - scaled @ theta_scaled
        robust_weights = np.minimum(
            1.0, huber_scale / np.maximum(np.abs(residual), huber_scale))
        weights = base_weights * robust_weights
    coefficients = theta_scaled / scale
    residual = target - features @ coefficients
    singular_values = np.linalg.svd(scaled, compute_uv=False)
    rank = int(np.linalg.matrix_rank(scaled))
    condition = (float(singular_values[0] / singular_values[-1])
                 if singular_values[-1] > 1.0e-12 else math.inf)
    return coefficients, {
        "samples": int(len(target)),
        "rank": rank,
        "columns": int(features.shape[1]),
        "condition_number_scaled": condition,
        "column_scale": [float(value) for value in scale],
        "residual": _stats(residual),
        "fit_method": "causal_derivative_robust_irls",
        "bounds_applied": bounds is not None,
        "huber_scale": huber_scale,
        "sample_weight_range": [float(np.min(base_weights)),
                                 float(np.max(base_weights))],
    }


def _fit_parameters(runs: dict[str, list[dict[str, float]]],
                    contact_model: str,
                    base_parameters: NativeModelParameters | None = None
                    ) -> tuple[NativeModelParameters, dict[str, Any]]:
    base = (base_parameters if base_parameters is not None else
            NativeModelParameters(contact_model=contact_model))
    if base.contact_model != contact_model:
        base = replace(base, contact_model=contact_model)
    examples = _fit_examples(runs, contact_model, base)
    if contact_model == "moment_basis_split":
        coordinates = np.asarray(examples["coordinates"], dtype=float)
        longitudinal_q = coordinates[:, 0]
        lateral_q = coordinates[:, 1:3]
        yaw_q = coordinates[:, 3:6]
        longitudinal_weights = 0.10 + 0.90 * np.minimum(
            1.0, np.abs(longitudinal_q) / 0.05)
        lateral_weights = 0.10 + 0.90 * np.minimum(
            1.0, np.linalg.norm(lateral_q, axis=1) / 0.05)
        yaw_weights = 0.10 + 0.90 * np.minimum(
            1.0, np.linalg.norm(yaw_q, axis=1) / 0.01)
        longitudinal, longitudinal_fit = _robust_linear_fit(
            examples["longitudinal_features"],
            examples["longitudinal_target"],
            (np.asarray([0.0, -np.inf, -np.inf]),
             np.asarray([np.inf, 0.0, 0.0])),
            longitudinal_weights, huber_scale=1.0)
        lateral, lateral_fit = _robust_linear_fit(
            lateral_q, examples["lateral_target"],
            (np.zeros(2, dtype=float), np.full(2, np.inf, dtype=float)),
            lateral_weights, huber_scale=0.5)
        yaw, yaw_fit = _robust_linear_fit(
            yaw_q, examples["yaw_target"],
            (np.zeros(3, dtype=float), np.full(3, np.inf, dtype=float)),
            yaw_weights, huber_scale=7.0)
        values = {
            "effective_force_x_gain_mps2": float(longitudinal[0]),
            "effective_drag_linear_per_s": float(-longitudinal[1]),
            "effective_drag_quadratic_per_m": float(-longitudinal[2]),
            "effective_force_y_front_gain_mps2": float(lateral[0]),
            "effective_force_y_rear_gain_mps2": float(lateral[1]),
            "effective_yaw_lateral_front_gain_per_s2": float(yaw[0]),
            "effective_yaw_lateral_rear_gain_per_s2": float(yaw[1]),
            "effective_yaw_longitudinal_gain_per_s2": float(yaw[2]),
        }
        fitted = replace(
            base,
            effective_force_x_gain_mps2=values[
                "effective_force_x_gain_mps2"],
            effective_drag_linear_per_s=values[
                "effective_drag_linear_per_s"],
            effective_drag_quadratic_per_m=values[
                "effective_drag_quadratic_per_m"],
            effective_force_y_front_gain_mps2=values[
                "effective_force_y_front_gain_mps2"],
            effective_force_y_rear_gain_mps2=values[
                "effective_force_y_rear_gain_mps2"],
            effective_yaw_lateral_front_gain_per_s2=values[
                "effective_yaw_lateral_front_gain_per_s2"],
            effective_yaw_lateral_rear_gain_per_s2=values[
                "effective_yaw_lateral_rear_gain_per_s2"],
            effective_yaw_longitudinal_gain_per_s2=values[
                "effective_yaw_longitudinal_gain_per_s2"],
            parameter_provenance=(
                f"{base.parameter_provenance};"
                "effective_gains_fit:moment_basis_split"),
        )
        return fitted, {
            "contact_model": contact_model,
            "parameter_provenance": fitted.parameter_provenance,
            "effective_gains": values,
            "longitudinal_regression": longitudinal_fit,
            "lateral_regression": lateral_fit,
            "yaw_regression": yaw_fit,
        }
    if contact_model == "moment_basis":
        coordinates = np.asarray(examples["coordinates"], dtype=float)
        longitudinal_q = coordinates[:, 0]
        lateral_q = coordinates[:, 1]
        yaw_q = coordinates[:, 2]
        longitudinal_weights = 0.10 + 0.90 * np.minimum(
            1.0, np.abs(longitudinal_q) / 0.05)
        lateral_weights = 0.10 + 0.90 * np.minimum(
            1.0, np.abs(lateral_q) / 0.05)
        yaw_weights = 0.10 + 0.90 * np.minimum(
            1.0, np.abs(yaw_q) / 0.01)
        longitudinal, longitudinal_fit = _robust_linear_fit(
            examples["longitudinal_features"],
            examples["longitudinal_target"],
            (np.asarray([0.0, -np.inf, -np.inf]),
             np.asarray([np.inf, 0.0, 0.0])),
            longitudinal_weights, huber_scale=1.0)
        lateral, lateral_fit = _robust_linear_fit(
            lateral_q[:, None], examples["lateral_target"],
            (np.zeros(1, dtype=float), np.full(1, np.inf, dtype=float)),
            lateral_weights, huber_scale=0.5)
        yaw, yaw_fit = _robust_linear_fit(
            np.asarray(examples["yaw_features"], dtype=float),
            examples["yaw_target"],
            (np.zeros(2, dtype=float), np.full(2, np.inf, dtype=float)),
            yaw_weights, huber_scale=7.0)
        values = {
            "effective_force_x_gain_mps2": float(longitudinal[0]),
            "effective_drag_linear_per_s": float(-longitudinal[1]),
            "effective_drag_quadratic_per_m": float(-longitudinal[2]),
            "effective_force_y_gain_mps2": float(lateral[0]),
            "effective_yaw_moment_gain_per_s2": float(yaw[0]),
            "effective_yaw_damping_per_s": float(yaw[1]),
        }
        fitted = replace(
            base,
            effective_force_x_gain_mps2=values[
                "effective_force_x_gain_mps2"],
            effective_drag_linear_per_s=values["effective_drag_linear_per_s"],
            effective_drag_quadratic_per_m=values[
                "effective_drag_quadratic_per_m"],
            effective_force_y_gain_mps2=values[
                "effective_force_y_gain_mps2"],
            effective_yaw_moment_gain_per_s2=values[
                "effective_yaw_moment_gain_per_s2"],
            effective_yaw_damping_per_s=values[
                "effective_yaw_damping_per_s"],
            parameter_provenance=(
                f"{base.parameter_provenance};"
                "effective_gains_fit:moment_basis"),
        )
        return fitted, {
            "contact_model": contact_model,
            "parameter_provenance": fitted.parameter_provenance,
            "effective_gains": values,
            "longitudinal_regression": longitudinal_fit,
            "lateral_regression": lateral_fit,
            "yaw_regression": yaw_fit,
        }
    # These bounds preserve the sign implied by the documented force
    # coordinates while keeping the coefficients explicitly effective.
    inf = np.full(3, np.inf, dtype=float)
    q = np.asarray(examples["lateral_features"], dtype=float)
    longitudinal_q = np.asarray(
        examples["longitudinal_features"][:, 0], dtype=float)
    # Most accepted transitions are straight-line or stationary samples. A
    # small floor keeps those samples in the equilibrium fit, while the
    # lateral/yaw channels give informative turning transitions comparable
    # influence instead of fitting almost entirely to zero derivatives.
    longitudinal_weights = 0.10 + 0.90 * np.minimum(
        1.0, np.abs(longitudinal_q) / 0.05)
    lateral_weights = 0.10 + 0.90 * np.minimum(
        1.0, np.hypot(q[:, 0], q[:, 1]) / 0.05)
    yaw_weights = lateral_weights.copy()
    longitudinal, longitudinal_fit = _robust_linear_fit(
        examples["longitudinal_features"], examples["longitudinal_target"],
        (np.asarray([0.0, -np.inf, -np.inf]), np.asarray([np.inf, 0.0, 0.0])),
        longitudinal_weights, huber_scale=1.0)
    lateral, lateral_fit = _robust_linear_fit(
        examples["lateral_features"], examples["lateral_target"],
        (np.zeros(2, dtype=float), np.full(2, np.inf, dtype=float)),
        lateral_weights, huber_scale=0.5)
    yaw, yaw_fit = _robust_linear_fit(
        examples["yaw_features"], examples["yaw_target"],
        (np.zeros(2, dtype=float), inf[:2]), yaw_weights, huber_scale=7.0)
    values = {
        "effective_longitudinal_gain_mps2": float(longitudinal[0]),
        "effective_drag_linear_per_s": float(-longitudinal[1]),
        "effective_drag_quadratic_per_m": float(-longitudinal[2]),
        "effective_lateral_front_mps2": float(lateral[0]),
        "effective_lateral_rear_mps2": float(lateral[1]),
        "effective_yaw_front_per_s2": float(yaw[0]),
        "effective_yaw_rear_per_s2": float(yaw[1]),
    }
    # Preserve every fixed structural setting from the selected source
    # profile.  Reconstructing NativeModelParameters from only the effective
    # gains would silently revert geometry, friction curves, and the pose
    # reference point to guide defaults.
    fitted = replace(
        base,
        effective_longitudinal_gain_mps2=values[
            "effective_longitudinal_gain_mps2"],
        effective_drag_linear_per_s=values["effective_drag_linear_per_s"],
        effective_drag_quadratic_per_m=values[
            "effective_drag_quadratic_per_m"],
        effective_lateral_front_mps2=values["effective_lateral_front_mps2"],
        effective_lateral_rear_mps2=values["effective_lateral_rear_mps2"],
        effective_yaw_front_per_s2=values["effective_yaw_front_per_s2"],
        effective_yaw_rear_per_s2=values["effective_yaw_rear_per_s2"],
        parameter_provenance=(
            f"{base.parameter_provenance};"
            f"effective_gains_fit:{contact_model}"),
    )
    return fitted, {
        "contact_model": contact_model,
        "parameter_provenance": fitted.parameter_provenance,
        "effective_gains": values,
        "longitudinal_regression": longitudinal_fit,
        "lateral_regression": lateral_fit,
        "yaw_regression": yaw_fit,
    }


def _mixed_fit_origins(
        runs: dict[str, list[dict[str, float]]],
        max_origins_per_run: int) -> list[tuple[list[dict[str, float]], int]]:
    """Select causal training origins for the recursive gain fit.

    The derivative fit already uses all eligible transitions.  The recursive
    stage uses a smaller, deterministic subset so each optimizer evaluation
    remains practical while retaining every dynamic training regime.
    """
    origins: list[tuple[list[dict[str, float]], int]] = []
    for rows in runs.values():
        eligible = [index for index, row in enumerate(rows)
                    if math.isfinite(row["wheel_k_mps"])]
        # Uniform sampling alone routinely misses the one or two transitions
        # where the steering actuator crosses through zero. Preserve those
        # causal events first, then fill the remaining budget uniformly. The
        # selection uses only origin state and applied input; no future target
        # state is used to construct a rollout.
        event_indices = [
            index for index in eligible
            if abs(rows[index]["applied_steering_rad_k1"] -
                   rows[index]["delta_k_rad"]) > 0.02 or
            abs(rows[index]["r_k_radps"]) > 0.75 or
            abs(rows[index]["v_k_mps"]) > 0.15]
        if len(event_indices) > max_origins_per_run:
            event_indices = [
                event_indices[int(position)] for position in np.linspace(
                    0, len(event_indices) - 1, max_origins_per_run,
                    dtype=int)]
        selected = list(dict.fromkeys(event_indices))
        remaining = max_origins_per_run - len(selected)
        if remaining > 0:
            uniform = _score_origins(rows, max_origins_per_run)
            selected.extend(index for index in uniform if index not in selected)
            selected = selected[:max_origins_per_run]
        for index in sorted(selected):
            row = rows[index]
            if (row["u_k_mps"] > 0.8 or
                    abs(row["applied_steering_rad_k1"]) > 0.03):
                origins.append((rows, index))
    if not origins:
        raise ValueError("mixed recursive fit found no dynamic training origins")
    return origins


def _mixed_fit_residual(
        gains: np.ndarray,
        base: NativeModelParameters,
        origins: Sequence[tuple[list[dict[str, float]], int]],
        gain_fields: tuple[str, ...] | None = None) -> np.ndarray:
    """Return one-step plus recursive residuals for robust gain fitting."""
    fields = gain_fields if gain_fields is not None else _gain_fields(
        base.contact_model)
    parameters = replace(base, **dict(zip(fields, gains)))
    residuals: list[float] = []
    for rows, origin in origins:
        first = rows[origin]
        state = _state_from_row(first)
        one_step = step(
            state,
            first["applied_steering_rad_k1"] / parameters.steering_limit_rad,
            first["applied_throttle_norm_k1"], first["dt_sim_s"], parameters)
        one_step_heading = ((one_step[2] - first["yaw_k1_rad"] + math.pi) %
                            (2.0 * math.pi) - math.pi)
        one_step_values = np.asarray((
            one_step[0] - first["x_k1_m"],
            one_step[1] - first["y_k1_m"],
            one_step_heading,
            one_step[3] - first["u_k1_mps"],
            one_step[4] - first["v_k1_mps"],
            one_step[5] - first["r_k1_radps"],
        ), dtype=float)
        residuals.extend(
            (0.25 * one_step_values / MIXED_FIT_ONE_STEP_SCALES).tolist())

        segment = int(first["segment_id"])
        state = _state_from_row(first)
        elapsed = 0.0
        index = origin
        horizon_index = 0
        while index < len(rows) and horizon_index < len(MIXED_FIT_HORIZONS_S):
            row = rows[index]
            if int(row["segment_id"]) != segment:
                break
            state = step(
                state,
                row["applied_steering_rad_k1"] /
                parameters.steering_limit_rad,
                row["applied_throttle_norm_k1"], row["dt_sim_s"], parameters)
            elapsed += row["dt_sim_s"]
            index += 1
            while (horizon_index < len(MIXED_FIT_HORIZONS_S) and
                   elapsed >= MIXED_FIT_HORIZONS_S[horizon_index] - 1.0e-10):
                target = rows[index - 1]
                heading = ((state[2] - target["yaw_k1_rad"] + math.pi) %
                           (2.0 * math.pi) - math.pi)
                values = np.asarray((
                    state[0] - target["x_k1_m"],
                    state[1] - target["y_k1_m"],
                    heading,
                    state[3] - target["u_k1_mps"],
                    state[4] - target["v_k1_mps"],
                    state[5] - target["r_k1_radps"],
                ), dtype=float)
                residuals.extend(
                    (values / MIXED_FIT_STATE_SCALES).tolist())
                horizon_index += 1
    return np.asarray(residuals, dtype=float)


def _fit_mixed_parameters(
        runs: dict[str, list[dict[str, float]]],
        contact_model: str,
        base: NativeModelParameters,
        max_origins_per_run: int,
        max_nfev: int) -> tuple[NativeModelParameters, dict[str, Any]]:
    """Fit effective gains against local and recursive training behavior."""
    initial, initial_report = _fit_parameters(runs, contact_model, base)
    origins = _mixed_fit_origins(runs, max_origins_per_run)
    gain_fields = _gain_fields(contact_model)
    initial_gains = np.asarray(
        [getattr(initial, field) for field in gain_fields], dtype=float)
    initial_residual = _mixed_fit_residual(
        initial_gains, base, origins, gain_fields)
    lower = np.zeros(len(gain_fields), dtype=float)
    upper = np.asarray(
        (20.0, 2.0, 1.0, 20.0, 100.0, 20.0)
        if contact_model == "moment_basis" else
        (20.0, 2.0, 1.0, 20.0, 20.0, 100.0, 100.0, 100.0)
        if contact_model == "moment_basis_split" else
        (20.0, 2.0, 1.0, 20.0, 20.0, 20.0, 20.0), dtype=float)
    from scipy.optimize import least_squares
    result = least_squares(
        lambda gains: _mixed_fit_residual(gains, base, origins, gain_fields),
        initial_gains, bounds=(lower, upper), loss="soft_l1", f_scale=1.0,
        x_scale=np.maximum(np.abs(initial_gains), 0.1), diff_step=0.01,
        max_nfev=max_nfev)
    fitted = replace(
        base,
        **dict(zip(gain_fields, result.x)),
        parameter_provenance=(
            f"{base.parameter_provenance};"
            f"effective_gains_fit:mixed_recursive:{contact_model}"),
    )
    report = {
        **initial_report,
        "fit_method": "mixed_one_step_recursive_huber",
        "mixed_fit_horizons_s": list(MIXED_FIT_HORIZONS_S),
        "mixed_fit_state_scales": MIXED_FIT_STATE_SCALES.tolist(),
        "mixed_fit_one_step_weight": 0.25,
        "mixed_fit_origins": len(origins),
        "mixed_fit_max_origins_per_run": max_origins_per_run,
        "mixed_fit_origin_selection": (
            "event_preserving:steering_rate_or_high_yaw_or_lateral_state_then_uniform"),
        "mixed_fit_gain_fields": list(gain_fields),
        "mixed_fit_optimizer": {
            "loss": "soft_l1",
            "f_scale": 1.0,
            "bounds": [lower.tolist(), upper.tolist()],
            "diff_step": 0.01,
            "max_nfev": max_nfev,
            "nfev": int(result.nfev),
            "status": int(result.status),
            "message": result.message,
            "cost": float(result.cost),
            "optimality": float(result.optimality),
            "active_mask": result.active_mask.tolist(),
            "initial_rms": float(np.sqrt(np.mean(initial_residual ** 2))),
            "final_rms": float(np.sqrt(np.mean(result.fun ** 2))),
        },
        "effective_gains": {
            field: float(getattr(fitted, field)) for field in gain_fields
        },
        "parameter_provenance": fitted.parameter_provenance,
    }
    return fitted, report


def _state_from_row(row: dict[str, float]) -> np.ndarray:
    return np.asarray([
        row["x_k_m"], row["y_k_m"], row["yaw_k_rad"], row["u_k_mps"],
        row["v_k_mps"], row["r_k_radps"], row["delta_k_rad"],
        row["wheel_left_k_mps"], row["wheel_right_k_mps"],
    ], dtype=float)


def _error_vector(predicted: np.ndarray, row: dict[str, float]) -> dict[str, float]:
    heading = ((predicted[2] - row["yaw_k1_rad"] + math.pi) %
               (2.0 * math.pi) - math.pi)
    position = math.hypot(predicted[0] - row["x_k1_m"],
                          predicted[1] - row["y_k1_m"])
    return {
        "x_m": predicted[0] - row["x_k1_m"],
        "y_m": predicted[1] - row["y_k1_m"],
        "position_m": position,
        "heading_rad": heading,
        "u_mps": predicted[3] - row["u_k1_mps"],
        "v_mps": predicted[4] - row["v_k1_mps"],
        "r_radps": predicted[5] - row["r_k1_radps"],
        "steering_rad": (predicted[6] -
                          row["simulator_feedback_steering_rad_k1"]),
        "wheel_mps": (0.5 * (predicted[7] + predicted[8]) -
                      row["wheel_speed_mps_k1"]),
        "wheel_left_mps": predicted[7] - row["wheel_left_speed_mps_k1"],
        "wheel_right_mps": predicted[8] - row["wheel_right_speed_mps_k1"],
    }


def _one_step_scores(runs: dict[str, list[dict[str, float]]],
                     parameters: NativeModelParameters) -> dict[str, Any]:
    errors = {field: [] for field in STATE_FIELDS}
    for rows in runs.values():
        for row in rows:
            if not math.isfinite(row["wheel_k_mps"]):
                continue
            predicted = step(
                _state_from_row(row),
                row["applied_steering_rad_k1"] / parameters.steering_limit_rad,
                row["applied_throttle_norm_k1"], row["dt_sim_s"], parameters)
            for field, value in _error_vector(predicted, row).items():
                errors[field].append(value)
    return {field: _stats(values) for field, values in errors.items()}


def _score_origins(rows: Sequence[dict[str, float]], max_origins: int
                   ) -> list[int]:
    eligible = [index for index, row in enumerate(rows)
                if math.isfinite(row["wheel_k_mps"])]
    if len(eligible) <= max_origins:
        return eligible
    positions = np.linspace(0, len(eligible) - 1, max_origins, dtype=int)
    return [eligible[int(position)] for position in positions]


def _recursive_scores(runs: dict[str, list[dict[str, float]]],
                      parameters: NativeModelParameters,
                      max_origins_per_run: int) -> dict[str, Any]:
    # Horizons are nested.  Propagating each origin once to the longest
    # horizon and sampling the same causal state at all requested horizons is
    # mathematically identical to replaying each horizon independently, but
    # avoids repeating the expensive 2 ms native-model substeps nine times.
    errors_by_horizon = {
        horizon: {field: [] for field in STATE_FIELDS}
        for horizon in HORIZONS_S
    }
    origins_used: dict[str, int] = {}
    for name, rows in runs.items():
        origins = _score_origins(rows, max_origins_per_run)
        origins_used[name] = len(origins)
        for origin in origins:
            first = rows[origin]
            segment = int(first["segment_id"])
            state = _state_from_row(first)
            elapsed = 0.0
            index = origin
            horizon_index = 0
            while (index < len(rows) and horizon_index < len(HORIZONS_S) and
                   elapsed < HORIZONS_S[-1] - 1.0e-10):
                row = rows[index]
                if int(row["segment_id"]) != segment:
                    break
                state = step(
                    state,
                    row["applied_steering_rad_k1"] /
                    parameters.steering_limit_rad,
                    row["applied_throttle_norm_k1"], row["dt_sim_s"],
                    parameters)
                elapsed += row["dt_sim_s"]
                index += 1
                while (horizon_index < len(HORIZONS_S) and
                       elapsed >= HORIZONS_S[horizon_index] - 1.0e-10):
                    target = rows[index - 1]
                    errors = errors_by_horizon[HORIZONS_S[horizon_index]]
                    for field, value in _error_vector(state, target).items():
                        errors[field].append(value)
                    horizon_index += 1
    output = {
        _horizon_key(horizon): {
            field: _stats(values) for field, values in errors.items()
        }
        for horizon, errors in errors_by_horizon.items()
    }
    output["evaluation"] = {
        "max_origins_per_run": max_origins_per_run,
        "origins_used_per_run": origins_used,
        "target_horizon_semantics": "first source transition sum >= requested horizon",
    }
    return output


def _candidate_report(train: dict[str, list[dict[str, float]]],
                      validation: dict[str, list[dict[str, float]]],
                      contact_model: str, max_origins_per_run: int,
                      base_parameters: NativeModelParameters | None = None,
                      fit_method: str = "derivative",
                      mixed_fit_origins_per_run: int = 35,
                      mixed_fit_max_nfev: int = 18
                      ) -> dict[str, Any]:
    base = (base_parameters if base_parameters is not None else
            NativeModelParameters(contact_model=contact_model))
    if fit_method == "derivative":
        parameters, fit_report = _fit_parameters(train, contact_model, base)
    elif fit_method == "mixed_recursive":
        parameters, fit_report = _fit_mixed_parameters(
            train, contact_model, base, mixed_fit_origins_per_run,
            mixed_fit_max_nfev)
    else:
        raise ValueError(f"unsupported fit method: {fit_method}")
    return {
        "parameters": fit_report,
        "train_one_step": _one_step_scores(train, parameters),
        "validation_one_step": _one_step_scores(validation, parameters),
        "train_recursive": _recursive_scores(
            train, parameters, max_origins_per_run),
        "validation_recursive": _recursive_scores(
            validation, parameters, max_origins_per_run),
    }


def _profile_factory(parameter_profile: str):
    if parameter_profile == "guide":
        return lambda model: NativeModelParameters(contact_model=model)
    if parameter_profile == "f1tenth_prefab":
        return f1tenth_prefab_parameters
    raise ValueError(f"unsupported parameter profile: {parameter_profile}")


def score_existing_report(
        root: Path, train_names: Sequence[str], validation_names: Sequence[str],
        source_report: Path, output: Path,
        max_origins_per_run: int) -> dict[str, Any]:
    """Replay fixed gains from a prior fit at a larger scoring population.

    This separates expensive optimizer work from the full validation replay.
    The source report is read only; no fitted values are refit or silently
    replaced.  It is useful for checking a promising screen at the complete
    deterministic-origin budget before any acceptance decision.
    """
    parent = json.loads(source_report.read_text(encoding="utf-8"))
    parameter_profile = str(parent["parameter_profile"])
    profile_factory = _profile_factory(parameter_profile)
    train = _read_runs(root, train_names)
    validation = _read_runs(root, validation_names)
    candidates: dict[str, Any] = {}
    for name, parent_candidate in parent["candidate_comparison"].items():
        parent_parameters = parent_candidate["parameters"]
        contact_model = str(parent_parameters["contact_model"])
        base = profile_factory(contact_model)
        gains = {
            field: float(parent_parameters["effective_gains"].get(field, 0.0))
            for field in _gain_fields(contact_model)
        }
        parameters = replace(
            base, **gains,
            parameter_provenance=(
                f"{base.parameter_provenance};"
                f"effective_gains_replayed:{source_report}"))
        fit_report = {
            "contact_model": contact_model,
            "effective_gains": gains,
            "fit_method": "fixed_replay_from_report",
            "parameter_profile": parameter_profile,
            "parameter_provenance": parameters.parameter_provenance,
            "source_report": str(source_report),
        }
        candidates[name] = {
            "parameters": fit_report,
            "train_one_step": _one_step_scores(train, parameters),
            "validation_one_step": _one_step_scores(validation, parameters),
            "train_recursive": _recursive_scores(
                train, parameters, max_origins_per_run),
            "validation_recursive": _recursive_scores(
                validation, parameters, max_origins_per_run),
        }
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "simulator_native_effective_candidates_not_accepted",
        "strategy_reference": (
            "SDU_Apex_Virtual_Simulator_Model_Fitting_Strategy_2026-09-13.md"),
        "parameter_profile": parameter_profile,
        "fit_method": "fixed_replay_from_report",
        "source_report": str(source_report),
        "ground_truth_use": "offline_identification_and_scoring_only",
        "recursive_prediction_uses_future_gt": False,
        "train_runs": list(train_names),
        "validation_runs": list(validation_names),
        "state_definition": ["x_m", "y_m", "yaw_rad", "u_mps", "v_mps",
                              "r_radps", "steering_rad",
                              "wheel_left_speed_mps", "wheel_right_speed_mps"],
        "wheel_speed_semantics": (
            "wheel_left_speed_mps and wheel_right_speed_mps are encoder-derived "
            "wheel surface speeds, already equal to wheel_radius_m times "
            "angular rate; slip code must not multiply them by wheel_radius_m "
            "again. wheel_speed_mps is their recorded mean."),
        "input_definition": ["applied_steering_rad", "applied_throttle_norm"],
        "documented_slip_definition": parent["documented_slip_definition"],
        "parameter_policy": parent["parameter_policy"],
        "candidate_comparison": candidates,
        "evaluation": {
            "required_horizons_s": list(HORIZONS_S),
            "max_origins_per_run": max_origins_per_run,
            "native_c_parity": "not run",
            "blind_track_run": "not run",
            "production_mpc_updated": False,
            "runtime_ground_truth_consumed": False,
        },
        "selection": {
            "selected_candidate": None,
            "reason": (
                "This is a fixed-parameter replay only. Selection still requires "
                "native parity, observer replay, and blind track acceptance."),
            "next_action": (
                "Review full-horizon validation, then acquire diagnostic dump and "
                "run native parity before production MPC work."),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return report


def fit(root: Path, train_names: Sequence[str], validation_names: Sequence[str],
        output: Path, max_origins_per_run: int,
        parameter_profile: str = "guide",
        fit_method: str = "derivative",
        mixed_fit_origins_per_run: int = 35,
        mixed_fit_max_nfev: int = 18,
        candidate_models: Sequence[str] | None = None) -> dict[str, Any]:
    overlap = set(train_names).intersection(validation_names)
    if overlap:
        raise ValueError(f"training and validation runs overlap: {sorted(overlap)}")
    train = _read_runs(root, train_names)
    validation = _read_runs(root, validation_names)
    profile_factory = _profile_factory(parameter_profile)
    definitions = {
        "vs1": (
            "VS1_documented_slip_spline_axle_effective", "axle"),
        "vs2": (
            "VS2_documented_slip_spline_four_wheel_effective", "four_wheel"),
        "vs25": (
            "VS2_5_documented_slip_spline_four_contact_moment_basis",
            "moment_basis"),
        "vs275": (
            "VS2_75_documented_slip_spline_split_contact_moment_basis",
            "moment_basis_split"),
    }
    selected_models = tuple(candidate_models or definitions)
    unknown = sorted(set(selected_models).difference(definitions))
    if unknown:
        raise ValueError(f"unsupported candidate models: {unknown}")
    candidates: dict[str, Any] = {}
    for model_key in selected_models:
        name, contact_model = definitions[model_key]
        candidates[name] = _candidate_report(
            train, validation, contact_model, max_origins_per_run,
            profile_factory(contact_model), fit_method,
            mixed_fit_origins_per_run, mixed_fit_max_nfev)
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "simulator_native_effective_candidates_not_accepted",
        "strategy_reference": "SDU_Apex_Virtual_Simulator_Model_Fitting_Strategy_2026-09-13.md",
        "parameter_profile": parameter_profile,
        "fit_method": fit_method,
        "candidate_models": list(selected_models),
        "ground_truth_use": "offline_identification_and_scoring_only",
        "recursive_prediction_uses_future_gt": False,
        "train_runs": list(train_names),
        "validation_runs": list(validation_names),
        "state_definition": ["x_m", "y_m", "yaw_rad", "u_mps", "v_mps",
                              "r_radps", "steering_rad",
                              "wheel_left_speed_mps", "wheel_right_speed_mps"],
        "wheel_speed_semantics": (
            "wheel_left_speed_mps and wheel_right_speed_mps are encoder-derived "
            "wheel surface speeds, already equal to wheel_radius_m times "
            "angular rate; slip code must not multiply them by wheel_radius_m "
            "again. wheel_speed_mps is their recorded mean."),
        "input_definition": ["applied_steering_rad", "applied_throttle_norm"],
        "documented_slip_definition": {
            "longitudinal": "(rw*omega-vx)/max(abs(vx),0.25)",
            "lateral": "vy/abs(vx), represented by tire-frame contact velocity",
            "curve_source": "public guide breakpoints with documented two-piece cubic Hermite surrogate",
            "exact_unity_curve_claimed": False,
        },
        "parameter_policy": {
            "yaw_inertia_fitted": False,
            "yaw_inertia_source": "not used; effective yaw gains absorb unresolved simulator structure",
            "effective_gain_provenance_required": True,
            "official_guide_structure_until_diagnostic_dump": True,
        },
        "candidate_comparison": candidates,
        "evaluation": {
            "required_horizons_s": list(HORIZONS_S),
            "max_origins_per_run": max_origins_per_run,
            "native_c_parity": "not run",
            "blind_track_run": "not run",
            "production_mpc_updated": False,
            "runtime_ground_truth_consumed": False,
        },
        "selection": {
            "selected_candidate": None,
            "reason": "Selection requires review of recursive validation errors, native parity, and blind replay; neither candidate is promoted by this fit alone.",
            "next_action": "Acquire and analyze Unity diagnostic dump, then compare effective candidates against dumped structure before native C and production MPC work.",
        },
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-origins-per-run", type=int, default=1500)
    parser.add_argument(
        "--parameter-profile", choices=("guide", "f1tenth_prefab"),
        default="guide",
        help="fixed offline structure used before effective-gain fitting")
    parser.add_argument(
        "--fit-method", choices=("derivative", "mixed_recursive", "fixed_replay"),
        default="derivative",
        help="effective-gain fitting objective")
    parser.add_argument(
        "--source-report", type=Path,
        help="prior report to replay when --fit-method=fixed_replay")
    parser.add_argument("--mixed-fit-origins-per-run", type=int, default=35)
    parser.add_argument("--mixed-fit-max-nfev", type=int, default=18)
    parser.add_argument(
        "--candidates", default="vs1,vs2,vs25",
        help="comma-separated candidates: vs1, vs2, vs25, vs275")
    args = parser.parse_args()
    if args.max_origins_per_run < 1:
        raise ValueError("--max-origins-per-run must be positive")
    train_names = _names(args.train_runs)
    validation_names = _names(args.validation_runs)
    if args.fit_method == "fixed_replay":
        if args.source_report is None:
            raise ValueError("--source-report is required for fixed_replay")
        report = score_existing_report(
            args.accepted_root, train_names, validation_names,
            args.source_report, args.output, args.max_origins_per_run)
    else:
        if args.source_report is not None:
            raise ValueError("--source-report is only valid for fixed_replay")
        report = fit(
            args.accepted_root, train_names, validation_names,
            args.output, args.max_origins_per_run, args.parameter_profile,
            args.fit_method, args.mixed_fit_origins_per_run,
            args.mixed_fit_max_nfev, _names(args.candidates))
    print(json.dumps({
        "output": str(args.output),
        "status": report["status"],
        "candidates": list(report["candidate_comparison"]),
        "selected_candidate": report["selection"]["selected_candidate"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
