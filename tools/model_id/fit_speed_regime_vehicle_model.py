#!/usr/bin/env python3
"""Fit and score an offline, raceline-relevant speed-regime vehicle model.

The model is deliberately separate from the production MPC.  It uses only
the canonical ``model_transition_v4.csv`` transition tables and keeps the
simulator ground truth on the identification/scoring side of the boundary.

The candidate contains:

* an explicit effective-steering state with a fitted first-order lag;
* smoothly blended low/mid/high lateral tire parameters;
* a combined-slip lateral force term using wheel/body longitudinal slip;
* smoothly blended longitudinal drive-force and slip-gain profiles; and
* the measured Unity linear damping exactly once, with no fitted drag term.

The transition loader also preserves the packet timing contract: the raw
``commanded_steering_norm_k1`` value belongs to the post-step packet, so the
steering command consumed by the preceding transition is exposed as
``steering_target_norm_k`` and used for causal replay.

The three profile bands are 2--8, 8--12, and 12--16 m/s.  Samples outside
the 16 m/s project limit, unrealistic steering, or excessive lateral demand
are excluded from fitting and recursive acceptance scoring.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.optimize import least_squares

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from structured_vehicle_plant import (  # noqa: E402
    GRAVITY_MPS2,
    MASS_KG,
    MAX_STEERING_RAD,
    PlantParameters,
    _steering_next,
    _speed_regime_value,
    lateral_body_step,
    step,
)
from build_raceline_operating_envelope import (  # noqa: E402
    DEFAULT_RACELINE,
    _read_raceline,
)
from score_raceline_model import (  # noqa: E402
    _raceline_arrays,
    frenet_error,
    operating_class,
)


MAX_SPEED_MPS = 16.0
MAX_ABS_STEERING_RAD = 0.45
# The raceline-relevant data collected so far stays below approximately
# 12 m/s^2 lateral acceleration.  The steering-product envelope removes
# combinations such as near-full-lock at 14--16 m/s that are not present on
# the usable raceline.  These are identification/scoring filters, not vehicle
# or simulator limits.
MAX_ABS_LATERAL_ACCEL_MPS2 = 12.0
MAX_STEERING_RATE_RADPS = 3.2
MAX_SPEED_STEERING_PRODUCT_MPS_RAD = 3.5
# This is an optimizer guard used for model-identification diagnostics.  It is
# not a claim about the simulator tire law.  A fitted peak reaching this value
# means the data/model combination is asking for more capacity than the
# current search allows, so callers should repeat the fit with a larger value
# and compare held-out prediction before accepting the change.
DEFAULT_LATERAL_PEAK_UPPER_N = 60.0
REGIME_NAMES = ("low_2_8", "mid_8_12", "high_12_16")
REGIME_BOUNDS = ((2.0, 8.0), (8.0, 12.0), (12.0, 16.0))
# The active model horizon is 30 commands at 40 Hz = .75 s.  Longer horizons
# are intentionally not part of this fitter.
HORIZONS_S = (0.10, 0.25, 0.50, 0.75)
LATERAL_FIT_HORIZONS_S = (0.25, 0.50, 0.75)
RUN_WINDOW_RE = re.compile(r"^(.*)@(\d+):(\d+)$")


def _raceline_context(path: Path = DEFAULT_RACELINE) -> dict[str, Any]:
    rows = _read_raceline(path)
    speed_width = 1.0
    curvature_width = 0.025
    core = {
        (math.floor(row["vx_mps"] / speed_width),
         math.floor(abs(row["kappa_radpm"]) / curvature_width))
        for row in rows
    }
    guard = {
        (speed + speed_offset, curvature + curvature_offset)
        for speed, curvature in core
        for speed_offset in (-1, 0, 1)
        for curvature_offset in (-1, 0, 1)
        if speed + speed_offset >= 0 and curvature + curvature_offset >= 0
    }
    return {
        "path": str(path),
        "raceline": _raceline_arrays(rows),
        "core_bins": core,
        "guard_bins": guard,
        "speed_bin_width_mps": speed_width,
        "curvature_bin_width_radpm": curvature_width,
    }


def _stats(values: Iterable[float]) -> dict[str, float | int | None]:
    finite = np.asarray([value for value in values if math.isfinite(value)],
                        dtype=float)
    if finite.size == 0:
        return {"count": 0, "mae": None, "p50": None, "p95": None,
                "p99": None, "max": None, "bias": None, "rmse": None}
    return {
        "count": int(finite.size),
        "mae": float(np.mean(np.abs(finite))),
        "p50": float(np.percentile(np.abs(finite), 50)),
        "p95": float(np.percentile(np.abs(finite), 95)),
        "p99": float(np.percentile(np.abs(finite), 99)),
        "max": float(np.max(np.abs(finite))),
        "bias": float(np.mean(finite)),
        "rmse": float(np.sqrt(np.mean(finite * finite))),
    }


def _raceline_envelope_ok(row: dict[str, float]) -> bool:
    """Return whether a source transition belongs to the usable envelope."""
    speed = abs(row["u_k_mps"])
    target_delta = _steering_command(row) * MAX_STEERING_RAD
    if not (0.0 <= speed <= MAX_SPEED_MPS):
        return False
    if abs(target_delta) > MAX_ABS_STEERING_RAD:
        return False
    if speed * abs(target_delta) > MAX_SPEED_STEERING_PRODUCT_MPS_RAD:
        return False
    if abs(speed * row["r_k_radps"]) > MAX_ABS_LATERAL_ACCEL_MPS2:
        return False
    if math.isfinite(row.get("delta_k_rad", math.nan)):
        steering_rate = abs(
            (target_delta - row["delta_k_rad"]) / row["dt_sim_s"])
        if steering_rate > MAX_STEERING_RATE_RADPS + 1.0e-9:
            return False
    return True


def _lateral_excited(row: dict[str, float]) -> bool:
    target_delta = _steering_command(row) * MAX_STEERING_RAD
    return (abs(target_delta) >= 0.01 or
            abs(row["u_k_mps"] * row["r_k_radps"]) >= 0.5)


def _parse_run_spec(spec: str) -> tuple[Path, int | None, int | None]:
    match = RUN_WINDOW_RE.match(spec)
    if match is None:
        return Path(spec), None, None
    start = int(match.group(2))
    end = int(match.group(3))
    if end <= start:
        raise ValueError(f"invalid run window: {spec}")
    return Path(match.group(1)), start, end


def _read_run(spec: str) -> list[dict[str, float]]:
    path, start, end = _parse_run_spec(spec)
    csv_path = path / "assembled" / "model_transition_v4.csv"
    required = {
        "dt_sim_s", "x_k_m", "y_k_m", "yaw_k_rad", "u_k_mps",
        "v_k_mps", "r_k_radps", "commanded_steering_norm_k1",
        "applied_steering_rad_k1", "applied_throttle_norm_k1",
        "simulator_feedback_steering_rad_k1",
        "wheel_speed_mps_k1", "x_k1_m", "y_k1_m", "yaw_k1_rad",
        "u_k1_mps", "v_k1_mps", "r_k1_radps", "segment_id",
    }
    with csv_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = sorted(required.difference(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"{csv_path} is missing fields: {missing}")
        rows: list[dict[str, float]] = []
        previous_segment: int | None = None
        previous_wheel = math.nan
        previous_feedback = math.nan
        previous_commanded_steering = math.nan
        for raw in reader:
            row = {
                field: float(raw[field]) for field in required
                if field != "segment_id"
            }
            row["segment_id"] = int(round(float(raw["segment_id"])))
            # The packet is sampled after the simulator step.  Its command is
            # the command issued for the next transition; the previous
            # packet's command was consumed by the transition represented by
            # this row.  Keep the raw k1 command untouched and expose the
            # consumed command explicitly for causal replay.
            row["steering_target_norm_k"] = (
                previous_commanded_steering
                if previous_segment == row["segment_id"] else math.nan)
            row["wheel_k_mps"] = (
                previous_wheel if previous_segment == row["segment_id"] else math.nan)
            row["delta_k_rad"] = (
                previous_feedback if previous_segment == row["segment_id"] else math.nan)
            rows.append(row)
            previous_segment = row["segment_id"]
            previous_wheel = row["wheel_speed_mps_k1"]
            previous_feedback = row["simulator_feedback_steering_rad_k1"]
            previous_commanded_steering = row["commanded_steering_norm_k1"]
    if start is not None:
        rows = rows[start:end]
    if len(rows) < 20:
        raise ValueError(f"run window is too short for fitting: {spec}")
    return rows


def _load_runs(specs: Sequence[str]) -> dict[str, list[dict[str, float]]]:
    if not specs:
        raise ValueError("at least one run is required")
    return {spec: _read_run(spec) for spec in specs}


def _steering_command(row: dict[str, float]) -> float:
    """Return the command consumed by the source transition.

    ``commanded_steering_norm_k1`` is retained for provenance, but is the
    command present in the post-step packet.  New assembled runs provide the
    derived previous-packet command; synthetic/unit-test rows may not, so
    they retain the historical current-command fallback.
    """
    consumed = row.get("steering_target_norm_k", math.nan)
    if math.isfinite(consumed):
        return consumed
    return row["commanded_steering_norm_k1"]


def _fit_steering_lag(rows: Sequence[dict[str, float]]) -> dict[str, Any]:
    selected = [
        row for row in rows
        if math.isfinite(row["delta_k_rad"])
        and 0.015 <= row["dt_sim_s"] <= 0.035
        and math.isfinite(_steering_command(row))
    ]
    if len(selected) < 20:
        raise ValueError("insufficient steering transitions for lag fit")

    def residual(values: np.ndarray) -> np.ndarray:
        tau = float(values[0])
        errors = []
        for row in selected:
            target = max(-1.0, min(1.0, _steering_command(row))) * \
                MAX_STEERING_RAD
            alpha = 1.0 - math.exp(-row["dt_sim_s"] / tau)
            prediction = row["delta_k_rad"] + alpha * (target - row["delta_k_rad"])
            errors.append(prediction - row["simulator_feedback_steering_rad_k1"])
        return np.asarray(errors, dtype=float)

    result = least_squares(
        residual, np.asarray([0.010]), bounds=(np.asarray([0.0001]),
                                               np.asarray([0.250])),
        loss="soft_l1", f_scale=0.002, max_nfev=100)
    errors = residual(result.x)
    return {
        "kind": "first_order_effective_steering",
        "time_constant_s": float(result.x[0]),
        "samples": len(selected),
        "residual_rad": _stats(errors),
        "optimizer": {
            "success": bool(result.success),
            "message": str(result.message),
            "nfev": int(result.nfev),
            "cost": float(result.cost),
        },
    }


def _fit_steering_rate_residual(
        rows: Sequence[dict[str, float]],
        parameters: PlantParameters) -> dict[str, Any]:
    """Fit a small causal residual for a moving steering state.

    The structured tire law is evaluated first with zero residual gains.  A
    transition-rate term is then fitted to the one-step v/r residuals.  The
    term is deliberately limited to steering transitions and is not allowed
    to alter the steady-state cornering law.  This is an offline candidate
    only; it must pass the blind recursive gates before any runtime use.
    """
    zero_gain_parameters = PlantParameters(
        **{
            **parameters.__dict__,
            "steering_rate_force_gain_n_per_radps": 0.0,
            "steering_rate_moment_gain_nm_per_radps": 0.0,
        })
    selected: list[tuple[dict[str, float], float, float, float]] = []
    for row in rows:
        if (not math.isfinite(row["delta_k_rad"]) or
                not math.isfinite(row["wheel_k_mps"]) or
                not (2.0 <= row["u_k_mps"] <= MAX_SPEED_MPS) or
                not _raceline_envelope_ok(row) or
                not _lateral_excited(row)):
            continue
        target = (max(-1.0, min(1.0, _steering_command(row))) *
                  MAX_STEERING_RAD)
        delta_end = _steering_next(
            row["delta_k_rad"], target, row["dt_sim_s"], zero_gain_parameters)
        steering_rate = (delta_end - row["delta_k_rad"]) / row["dt_sim_s"]
        if abs(steering_rate) < 0.05:
            continue
        predicted_v, predicted_r = lateral_body_step(
            row["u_k_mps"], row["v_k_mps"], row["r_k_radps"],
            row["delta_k_rad"], target, row["dt_sim_s"],
            zero_gain_parameters, wheel=row["wheel_k_mps"],
            throttle=row["applied_throttle_norm_k1"])
        selected.append((
            row,
            (row["v_k1_mps"] - predicted_v),
            (row["r_k1_radps"] - predicted_r),
            steering_rate,
        ))
    if len(selected) < 50:
        raise ValueError(
            "insufficient steering-transition residual samples: "
            f"{len(selected)}")

    def fit_channel(channel: int, scale: float, upper: float) -> Any:
        def exact_residual(values: np.ndarray) -> np.ndarray:
            gain = float(values[0])
            output = []
            for row, v_error, r_error, steering_rate in selected:
                observed_error = v_error if channel == 0 else r_error
                state_scale = parameters.mass_kg if channel == 0 else parameters.iz_kgm2
                predicted_error = (gain * steering_rate * row["dt_sim_s"] /
                                   state_scale)
                output.append((predicted_error - observed_error) / scale)
            output.append(0.02 * gain / upper)
            return np.asarray(output, dtype=float)

        result = least_squares(
            exact_residual, np.asarray([0.0]),
            bounds=(np.asarray([-upper]), np.asarray([upper])),
            loss="soft_l1", f_scale=1.0, max_nfev=120)
        return result, exact_residual(result.x)

    force_result, force_residual = fit_channel(0, 0.05, 20.0)
    moment_result, moment_residual = fit_channel(1, 0.10, 5.0)
    return {
        "samples": len(selected),
        "force_gain_n_per_radps": float(force_result.x[0]),
        "moment_gain_nm_per_radps": float(moment_result.x[0]),
        "residual_v_mps": _stats(force_residual[:-1]),
        "residual_r_radps": _stats(moment_residual[:-1]),
        "optimizer": {
            "force_success": bool(force_result.success),
            "moment_success": bool(moment_result.success),
            "force_cost": float(force_result.cost),
            "moment_cost": float(moment_result.cost),
        },
    }


def _lateral_rows(rows: Sequence[dict[str, float]], low: float,
                  high: float,
                  context: dict[str, Any] | None = None) -> list[dict[str, float]]:
    selected = []
    for row in rows:
        speed = row["u_k_mps"]
        target_delta = _steering_command(row) * MAX_STEERING_RAD
        if not (low <= speed <= high):
            continue
        if (not math.isfinite(row["wheel_k_mps"]) or
                not math.isfinite(row["delta_k_rad"]) or
                not _raceline_envelope_ok(row) or
                not _lateral_excited(row)):
            continue
        if (context is not None and operating_class(
                row["x_k_m"], row["y_k_m"], row["u_k_mps"],
                context["raceline"], context["core_bins"],
                context["guard_bins"], context["speed_bin_width_mps"],
                context["curvature_bin_width_radpm"], MAX_SPEED_MPS)
                not in {"core", "guard"}):
            continue
        selected.append(row)
    return selected


def _lateral_windows(runs: dict[str, list[dict[str, float]]],
                     low: float, high: float,
                     maximum_windows: int = 160,
                     context: dict[str, Any] | None = None
                     ) -> list[list[dict[str, float]]]:
    """Select contiguous causal turn windows for the MPC-relevant fit."""
    windows: list[list[dict[str, float]]] = []
    for rows in runs.values():
        for origin in range(0, len(rows), 8):
            first = rows[origin]
            if (not math.isfinite(first["delta_k_rad"]) or
                    not math.isfinite(first["wheel_k_mps"])):
                continue
            segment = int(first["segment_id"])
            window: list[dict[str, float]] = []
            elapsed = 0.0
            index = origin
            while index < len(rows) and elapsed < max(LATERAL_FIT_HORIZONS_S):
                row = rows[index]
                target_delta = _steering_command(row) * MAX_STEERING_RAD
                if (int(row["segment_id"]) != segment or
                        not (low <= row["u_k_mps"] <= high) or
                        not math.isfinite(row["wheel_k_mps"]) or
                        not _raceline_envelope_ok(row)):
                    window = []
                    break
                if (context is not None and operating_class(
                        row["x_k_m"], row["y_k_m"], row["u_k_mps"],
                        context["raceline"], context["core_bins"],
                        context["guard_bins"], context["speed_bin_width_mps"],
                        context["curvature_bin_width_radpm"], MAX_SPEED_MPS)
                        not in {"core", "guard"}):
                    window = []
                    break
                window.append(row)
                elapsed += row["dt_sim_s"]
                index += 1
            has_turn_excitation = any(_lateral_excited(item) for item in window)
            if (window and has_turn_excitation and
                    elapsed >= max(LATERAL_FIT_HORIZONS_S) - 1.0e-10):
                windows.append(window)
                if len(windows) >= maximum_windows:
                    return windows
    return windows


def _lateral_parameters(base: PlantParameters, values: np.ndarray,
                        steering_dynamics_kind: str,
                        tau_s: float) -> PlantParameters:
    return PlantParameters(
        **{
            **base.__dict__,
            "steering_dynamics_kind": steering_dynamics_kind,
            "steering_lag_time_constant_s": float(tau_s),
            "cf_n_per_rad": float(values[0]),
            "cr_n_per_rad": float(values[1]),
            "df_n": float(values[2]),
            "dr_n": float(values[3]),
            "combined_slip_gain": float(values[4]),
            "tire_model": "regime_speed_combined_tanh",
        })


def _fit_lateral_regime(rows: Sequence[dict[str, float]],
                        base: PlantParameters, low: float, high: float,
                        steering_dynamics_kind: str,
                        tau_s: float,
                        runs: dict[str, list[dict[str, float]]],
                        context: dict[str, Any] | None = None,
                        lateral_peak_upper_n: float =
                        DEFAULT_LATERAL_PEAK_UPPER_N) -> dict[str, Any]:
    if not math.isfinite(lateral_peak_upper_n) or lateral_peak_upper_n <= 2.0:
        raise ValueError(
            "lateral_peak_upper_n must be finite and greater than 2 N; "
            "use a large finite value for an effectively unbounded diagnostic")
    selected = _lateral_rows(rows, low, high, context)
    if len(selected) < 80:
        raise ValueError(
            f"insufficient lateral rows in {low:g}--{high:g} m/s: {len(selected)}")
    windows = _lateral_windows(runs, low, high, context=context)
    if len(windows) < 8:
        raise ValueError(
            f"insufficient contiguous lateral windows in {low:g}--{high:g} m/s: "
            f"{len(windows)}")
    initial = np.asarray([
        base.cf_n_per_rad, base.cr_n_per_rad, base.df_n, base.dr_n,
        max(0.0, base.combined_slip_gain),
    ], dtype=float)
    lower = np.asarray([100.0, 100.0, 2.0, 2.0, 0.0])
    upper = np.asarray([
        10000.0, 10000.0, lateral_peak_upper_n, lateral_peak_upper_n, 20.0])

    def residual(values: np.ndarray) -> np.ndarray:
        parameters = _lateral_parameters(
            base, values, steering_dynamics_kind, tau_s)
        output: list[float] = []
        for window in windows:
            current_v = window[0]["v_k_mps"]
            current_r = window[0]["r_k_radps"]
            current_delta = window[0]["delta_k_rad"]
            current_yaw = window[0]["yaw_k_rad"]
            elapsed = 0.0
            horizon_index = 0
            for row in window:
                previous_r = current_r
                current_v, current_r = lateral_body_step(
                    row["u_k_mps"], current_v, current_r, current_delta,
                    _steering_command(row) * MAX_STEERING_RAD,
                    row["dt_sim_s"], parameters,
                    wheel=row["wheel_k_mps"],
                    throttle=row["applied_throttle_norm_k1"])
                current_yaw += 0.5 * (previous_r + current_r) * row["dt_sim_s"]
                current_delta = _steering_next(
                    current_delta,
                    _steering_command(row) * MAX_STEERING_RAD,
                    row["dt_sim_s"], parameters)
                elapsed += row["dt_sim_s"]
                while (horizon_index < len(LATERAL_FIT_HORIZONS_S) and
                       elapsed + 1.0e-9 >=
                       LATERAL_FIT_HORIZONS_S[horizon_index]):
                    heading_error = ((current_yaw - row["yaw_k1_rad"] +
                                      math.pi) % (2.0 * math.pi) - math.pi)
                    output.extend([
                        (current_v - row["v_k1_mps"]) / 0.10,
                        (current_r - row["r_k1_radps"]) / 0.20,
                        heading_error / 0.10,
                    ])
                    horizon_index += 1
                if horizon_index == len(LATERAL_FIT_HORIZONS_S):
                    break
        # Retain a small one-step term so the profile does not trade all local
        # accuracy for a good mean recursive score.
        for row in selected[::8]:
            predicted_v, predicted_r = lateral_body_step(
                row["u_k_mps"], row["v_k_mps"], row["r_k_radps"],
                row["delta_k_rad"],
                _steering_command(row) * MAX_STEERING_RAD,
                row["dt_sim_s"], parameters,
                wheel=row["wheel_k_mps"],
                throttle=row["applied_throttle_norm_k1"])
            output.extend([
                0.25 * (predicted_v - row["v_k1_mps"]) / 0.05,
                0.25 * (predicted_r - row["r_k1_radps"]) / 0.10,
            ])
        # Combined-slip is observable only through the lateral data's
        # longitudinal-slip excitation. Keep it conservative when that
        # excitation is weak instead of allowing it to absorb tire mismatch.
        output.append(0.05 * float(values[4]))
        return np.asarray(output, dtype=float)

    result = least_squares(
        residual, initial, bounds=(lower, upper), loss="soft_l1",
        f_scale=1.0, x_scale="jac", max_nfev=180)
    parameters = _lateral_parameters(
        base, result.x, steering_dynamics_kind, tau_s)
    one_step_v: list[float] = []
    one_step_r: list[float] = []
    for row in selected:
        predicted_v, predicted_r = lateral_body_step(
            row["u_k_mps"], row["v_k_mps"], row["r_k_radps"],
            row["delta_k_rad"],
            _steering_command(row) * MAX_STEERING_RAD,
            row["dt_sim_s"], parameters,
            wheel=row["wheel_k_mps"],
            throttle=row["applied_throttle_norm_k1"])
        one_step_v.append(predicted_v - row["v_k1_mps"])
        one_step_r.append(predicted_r - row["r_k1_radps"])
    return {
        "speed_range_mps": [low, high],
        "fit_filter": "core_guard" if context is not None else "physical_envelope",
        "lateral_peak_upper_n": float(lateral_peak_upper_n),
        "samples": len(selected),
        "recursive_windows": len(windows),
        "parameters": {
            "cf_n_per_rad": float(result.x[0]),
            "cr_n_per_rad": float(result.x[1]),
            "df_n": float(result.x[2]),
            "dr_n": float(result.x[3]),
            "combined_slip_gain": float(result.x[4]),
            "tire_model": "regime_speed_combined_tanh",
        },
        "force_capacity_diagnostics": {
            "df_plus_dr_n": float(result.x[2] + result.x[3]),
            "equivalent_mu_without_downforce": float(
                (result.x[2] + result.x[3]) / (MASS_KG * GRAVITY_MPS2)),
            "peak_at_upper_bound": bool(
                result.x[2] >= lateral_peak_upper_n - 1.0e-6 or
                result.x[3] >= lateral_peak_upper_n - 1.0e-6),
            "interpretation": (
                "diagnostic capacity ratio only; simulator tire normal-load, "
                "friction, and downforce behavior are not inferred from this "
                "ratio")
        },
        "one_step_residual": {
            "v_mps": _stats(one_step_v),
            "r_radps": _stats(one_step_r),
        },
        "optimizer": {
            "success": bool(result.success),
            "message": str(result.message),
            "nfev": int(result.nfev),
            "cost": float(result.cost),
        },
    }


def _longitudinal_rows(rows: Sequence[dict[str, float]]) -> list[dict[str, float]]:
    selected = []
    for row in rows:
        speed = 0.5 * (row["u_k_mps"] + row["u_k1_mps"])
        if not (0.5 <= speed <= MAX_SPEED_MPS):
            continue
        if (abs(row["applied_steering_rad_k1"]) > 0.01 or
                row["applied_throttle_norm_k1"] <= 0.005 or
                not math.isfinite(row["wheel_k_mps"])):
            continue
        selected.append(row)
    return selected


def _fit_longitudinal_profile(rows: Sequence[dict[str, float]],
                              base: PlantParameters) -> dict[str, Any]:
    selected = _longitudinal_rows(rows)
    if len(selected) < 100:
        raise ValueError(f"insufficient powered straight rows: {len(selected)}")
    speeds = np.asarray([
        0.5 * (row["u_k_mps"] + row["u_k1_mps"]) for row in selected])
    slips = np.asarray([
        0.5 * (row["wheel_k_mps"] + row["wheel_speed_mps_k1"]) -
        0.5 * (row["u_k_mps"] + row["u_k1_mps"]) for row in selected])
    targets = np.asarray([
        MASS_KG * ((row["u_k1_mps"] - row["u_k_mps"]) /
                   row["dt_sim_s"] + row["r_k_radps"] * row["v_k_mps"] +
                   base.linear_damping_per_s * speed)
        for row, speed in zip(selected, speeds)])
    initial = np.asarray([
        base.force_max_n, base.force_max_n, base.force_max_n,
        base.slip_gain_per_mps, base.slip_gain_per_mps,
        base.slip_gain_per_mps,
    ])
    lower = np.asarray([1.0, 1.0, 1.0, 0.02, 0.02, 0.02])
    upper = np.asarray([60.0, 60.0, 60.0, 10.0, 10.0, 10.0])

    def prediction(values: np.ndarray) -> np.ndarray:
        force_values = tuple(float(value) for value in values[:3])
        gain_values = tuple(float(value) for value in values[3:])
        force = np.asarray([
            _speed_regime_value(force_values, speed,
                                base.regime_transition_width_mps)
            for speed in speeds])
        gain = np.asarray([
            _speed_regime_value(gain_values, speed,
                                base.regime_transition_width_mps)
            for speed in speeds])
        return force * np.tanh(gain * slips)

    def residual(values: np.ndarray) -> np.ndarray:
        predicted = prediction(values)
        regularization = np.asarray([
            0.15 * (values[1] - values[0]),
            0.15 * (values[2] - values[1]),
            0.15 * (values[4] - values[3]),
            0.15 * (values[5] - values[4]),
        ])
        return np.concatenate(((predicted - targets) / MASS_KG,
                               regularization))

    result = least_squares(
        residual, initial, bounds=(lower, upper), loss="soft_l1",
        f_scale=1.0, x_scale="jac", max_nfev=220)
    errors = prediction(result.x) - targets
    return {
        "samples": len(selected),
        "parameters": {
            "force_max_regimes_n": [float(value) for value in result.x[:3]],
            "slip_gain_regimes_per_mps": [float(value) for value in result.x[3:]],
            "coast_speed_drag_n_per_mps": 0.0,
            "linear_damping_per_s": base.linear_damping_per_s,
        },
        "one_step_force_residual_n": _stats(errors),
        "observed_force_n": _stats(targets),
        "optimizer": {
            "success": bool(result.success),
            "message": str(result.message),
            "nfev": int(result.nfev),
            "cost": float(result.cost),
        },
    }


def _coast_diagnostic(rows: Sequence[dict[str, float]],
                      base: PlantParameters) -> dict[str, Any]:
    values: list[float] = []
    speeds: list[float] = []
    for row in rows:
        if (abs(row["applied_steering_rad_k1"]) > 0.01 or
                row["applied_throttle_norm_k1"] > 0.005 or
                not (1.0 <= row["u_k_mps"] <= MAX_SPEED_MPS)):
            continue
        observed_force = MASS_KG * (
            (row["u_k1_mps"] - row["u_k_mps"]) / row["dt_sim_s"] +
            row["r_k_radps"] * row["v_k_mps"] +
            base.linear_damping_per_s * row["u_k_mps"])
        values.append(observed_force)
        speeds.append(row["u_k_mps"])
    return {
        "samples": len(values),
        "residual_force_after_measured_drag_n": _stats(values),
        "speed_mps": _stats(speeds),
        "decision": "no_residual_drag_added; zero_throttle_is_active_brake",
    }


def _profile_parameters(base: PlantParameters,
                        steering_dynamics_kind: str,
                        steering_tau_s: float,
                        lateral: Sequence[dict[str, Any]],
                        longitudinal: dict[str, Any],
                        steering_rate_residual: dict[str, Any] | None = None
                        ) -> PlantParameters:
    lateral_parameters = [entry["parameters"] for entry in lateral]
    return PlantParameters(
        **{
            **base.__dict__,
            "steering_dynamics_kind": steering_dynamics_kind,
            "steering_lag_time_constant_s": steering_tau_s,
            "tire_model": "regime_speed_combined_tanh",
            "lateral_cf_regimes_n_per_rad": tuple(
                float(entry["cf_n_per_rad"]) for entry in lateral_parameters),
            "lateral_cr_regimes_n_per_rad": tuple(
                float(entry["cr_n_per_rad"]) for entry in lateral_parameters),
            "lateral_df_regimes_n": tuple(
                float(entry["df_n"]) for entry in lateral_parameters),
            "lateral_dr_regimes_n": tuple(
                float(entry["dr_n"]) for entry in lateral_parameters),
            "combined_slip_gain": float(np.mean([
                entry["combined_slip_gain"] for entry in lateral_parameters])),
            "steering_rate_force_gain_n_per_radps": float(
                (steering_rate_residual or {}).get(
                    "force_gain_n_per_radps", 0.0)),
            "steering_rate_moment_gain_nm_per_radps": float(
                (steering_rate_residual or {}).get(
                    "moment_gain_nm_per_radps", 0.0)),
            "force_max_regimes_n": tuple(
                float(value) for value in longitudinal["parameters"][
                    "force_max_regimes_n"]),
            "slip_gain_regimes_per_mps": tuple(
                float(value) for value in longitudinal["parameters"][
                    "slip_gain_regimes_per_mps"]),
            "coast_speed_drag_n_per_mps": 0.0,
        })


def _score(runs: dict[str, list[dict[str, float]]],
           parameters: PlantParameters, origin_stride: int = 1,
           context: dict[str, Any] | None = None) -> dict[str, Any]:
    if origin_stride < 1:
        raise ValueError("origin_stride must be positive")
    context = context or _raceline_context()
    raceline = context["raceline"]
    fields = (
        "position_m", "e_cross_m", "e_along_m", "heading_rad",
        "e_heading_rad", "u_mps", "v_mps", "r_radps",
        "corridor_margin_m", "normalized_corridor_error",
    )

    def summarize(values: dict[str, list[float]]) -> dict[str, Any]:
        return {field: _stats(values[field]) for field in fields}

    output: dict[str, Any] = {}
    for horizon in HORIZONS_S:
        errors = {field: [] for field in fields}
        class_errors = {
            "core": {field: [] for field in fields},
            "guard": {field: [] for field in fields},
        }
        per_run: dict[str, dict[str, list[float]]] = {}
        per_segment: dict[str, dict[str, list[float]]] = {}
        skipped_out_of_envelope = 0
        failure_counts = {
            "invalid_transition": 0,
            "nonfinite_state": 0,
            "unstable_state_bound": 0,
            "incomplete_rollout": 0,
        }
        attempted = 0
        attempted_by_class = {"core": 0, "guard": 0}
        for run_name, rows in runs.items():
            for origin in range(0, len(rows), origin_stride):
                first = rows[origin]
                first_class = operating_class(
                    first["x_k_m"], first["y_k_m"], first["u_k_mps"],
                    raceline, context["core_bins"], context["guard_bins"],
                    context["speed_bin_width_mps"],
                    context["curvature_bin_width_radpm"], MAX_SPEED_MPS)
                if (not math.isfinite(first["wheel_k_mps"]) or
                        not math.isfinite(first["delta_k_rad"]) or
                        first_class == "stress"):
                    continue
                attempted += 1
                attempted_by_class[first_class] += 1
                state = np.asarray([
                    first["x_k_m"], first["y_k_m"], first["yaw_k_rad"],
                    first["u_k_mps"], first["v_k_mps"], first["r_k_radps"],
                    first["delta_k_rad"], first["wheel_k_mps"],
                ], dtype=float)
                segment = int(first["segment_id"])
                elapsed = 0.0
                index = origin
                invalid = False
                failure_reason: str | None = None
                while index < len(rows) and elapsed < horizon - 1.0e-10:
                    row = rows[index]
                    if (int(row["segment_id"]) != segment or
                            operating_class(
                                row["x_k_m"], row["y_k_m"], row["u_k_mps"],
                                raceline, context["core_bins"],
                                context["guard_bins"],
                                context["speed_bin_width_mps"],
                                context["curvature_bin_width_radpm"],
                                MAX_SPEED_MPS) == "stress" or
                            not (0.0 <= row["u_k1_mps"] <= MAX_SPEED_MPS)):
                        invalid = True
                        failure_reason = "invalid_transition"
                        break
                    state = step(
                        state, _steering_command(row),
                        row["applied_throttle_norm_k1"], row["dt_sim_s"],
                        parameters)
                    if not np.all(np.isfinite(state)):
                        invalid = True
                        failure_reason = "nonfinite_state"
                        break
                    if (state[3] < -1.0e-9 or state[3] > MAX_SPEED_MPS + 1.0e-9 or
                            abs(state[4]) > 50.0 or abs(state[5]) > 100.0):
                        invalid = True
                        failure_reason = "unstable_state_bound"
                        break
                    elapsed += row["dt_sim_s"]
                    index += 1
                if invalid:
                    skipped_out_of_envelope += 1
                    failure_counts[failure_reason or "invalid_transition"] += 1
                    continue
                if index == origin or elapsed < horizon - 1.0e-10:
                    failure_counts["incomplete_rollout"] += 1
                    continue
                truth = rows[index - 1]
                frenet = frenet_error(
                    state[0], state[1], state[2], truth["x_k1_m"],
                    truth["y_k1_m"], truth["yaw_k1_rad"], raceline)
                values = {
                    "position_m": float(frenet["position_m"]),
                    "e_cross_m": float(frenet["e_cross_m"]),
                    "e_along_m": float(frenet["e_along_m"]),
                    "heading_rad": float(frenet["e_heading_rad"]),
                    "e_heading_rad": float(frenet["e_heading_rad"]),
                    "u_mps": float(state[3] - truth["u_k1_mps"]),
                    "v_mps": float(state[4] - truth["v_k1_mps"]),
                    "r_radps": float(state[5] - truth["r_k1_radps"]),
                    "corridor_margin_m": float(frenet["corridor_margin_m"]),
                    "normalized_corridor_error": float(
                        frenet["normalized_corridor_error"]),
                }
                for field, value in values.items():
                    errors[field].append(value)
                    class_errors[first_class][field].append(value)
                run_bucket = per_run.setdefault(
                    run_name, {field: [] for field in fields})
                segment_bucket = per_segment.setdefault(
                    str(segment), {field: [] for field in fields})
                for field, value in values.items():
                    run_bucket[field].append(value)
                    segment_bucket[field].append(value)
        output[f"{horizon:.2f}s"] = {
            **summarize(errors),
            "core": summarize(class_errors["core"]),
            "guard": summarize(class_errors["guard"]),
            "per_run": {
                name: summarize(values) for name, values in sorted(per_run.items())
            },
            "per_segment": {
                name: summarize(values)
                for name, values in sorted(per_segment.items())
            },
        }
        output[f"{horizon:.2f}s"]["evaluation"] = {
            "attempted_origins": attempted,
            "attempted_origins_by_class": attempted_by_class,
            "valid_endpoint_count": len(errors["position_m"]),
            "skipped_out_of_raceline_core_guard": skipped_out_of_envelope,
            "failure_counts": failure_counts,
            "rollout_rejection_count": sum(failure_counts.values()),
            "stability_failure_count": (
                failure_counts["nonfinite_state"] +
                failure_counts["unstable_state_bound"]),
            "origin_stride": origin_stride,
            "primary_metric": "e_cross_m",
            "euclidean_position_is_diagnostic": True,
        }
    return output


def fit(train_specs: Sequence[str], validation_specs: Sequence[str],
        output: Path, steering_dynamics_kind: str = "instantaneous",
        lateral_specs: dict[str, Sequence[str]] | None = None,
        score_origin_stride: int = 1,
        lateral_profile_kind: str = "fitted",
        fit_steering_rate_residual: bool = False,
        raceline_path: Path = DEFAULT_RACELINE,
        lateral_peak_upper_n: float = DEFAULT_LATERAL_PEAK_UPPER_N) -> dict[str, Any]:
    if steering_dynamics_kind not in {"instantaneous", "first_order", "rate_limited"}:
        raise ValueError(f"unsupported steering dynamics: {steering_dynamics_kind}")
    if lateral_profile_kind not in {"fitted", "canonical"}:
        raise ValueError(
            f"unsupported lateral profile kind: {lateral_profile_kind}")
    train_runs = _load_runs(train_specs)
    validation_runs = _load_runs(validation_specs)
    envelope_context = _raceline_context(raceline_path)
    train_rows = [row for rows in train_runs.values() for row in rows]
    base = PlantParameters.from_manifest()
    steering = _fit_steering_lag(train_rows)
    tau_s = float(steering["time_constant_s"])
    if lateral_specs is None:
        lateral_specs = {name: tuple(train_specs) for name in REGIME_NAMES}
    lateral: list[dict[str, Any]] = []
    for name, (low, high) in zip(REGIME_NAMES, REGIME_BOUNDS):
        regime_specs = tuple(lateral_specs.get(name, train_specs))
        regime_runs = _load_runs(regime_specs)
        regime_rows = [row for rows in regime_runs.values() for row in rows]
        # The archive contains enough contiguous occupied data for the
        # 2--8 m/s regime, so constrain that fit to CORE/GUARD.  The 8--12
        # archive has only sparse occupied transitions and no complete
        # 0.75-second windows; retain its physical-envelope fit until a clean
        # higher-speed track campaign supplies identifiable windows.  The
        # separate 12--16 m/s fit is diagnostic because the current raceline
        # does not reach that speed.
        regime_context = envelope_context if high <= 8.0 else None
        fit_report = _fit_lateral_regime(
            regime_rows, base, low, high, steering_dynamics_kind, tau_s,
            regime_runs, context=regime_context,
            lateral_peak_upper_n=lateral_peak_upper_n)
        fit_report["regime"] = name
        fit_report["train_specs"] = list(regime_specs)
        lateral.append(fit_report)
    longitudinal = _fit_longitudinal_profile(train_rows, base)
    if lateral_profile_kind == "canonical":
        profile_lateral = [{
            "parameters": {
                "cf_n_per_rad": base.cf_n_per_rad,
                "cr_n_per_rad": base.cr_n_per_rad,
                "df_n": base.df_n,
                "dr_n": base.dr_n,
                "combined_slip_gain": 0.0,
            },
        }] * len(REGIME_NAMES)
    else:
        profile_lateral = lateral
    provisional_profile = _profile_parameters(
        base, steering_dynamics_kind, tau_s, profile_lateral, longitudinal)
    steering_rate_residual = (
        _fit_steering_rate_residual(train_rows, provisional_profile)
        if fit_steering_rate_residual else None)
    profile = _profile_parameters(
        base, steering_dynamics_kind, tau_s, profile_lateral, longitudinal,
        steering_rate_residual)
    baseline = replace_for_baseline(base)
    train_scores = _score(
        train_runs, profile, score_origin_stride, envelope_context)
    validation_scores = _score(
        validation_runs, profile, score_origin_stride, envelope_context)
    baseline_validation_scores = _score(
        validation_runs, baseline, score_origin_stride, envelope_context)
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "offline_speed_regime_candidate_not_runtime_validated",
        "ground_truth_use": "offline_identification_and_scoring_only",
        "production_mpc_parameters_updated": False,
        "simulator_modified": False,
        "speed_limit_mps": MAX_SPEED_MPS,
        "lateral_peak_upper_n": float(lateral_peak_upper_n),
        "profile_speed_centers_mps": [5.0, 10.0, 14.0],
        "profile_transition_width_mps": base.regime_transition_width_mps,
        "train_specs": list(train_specs),
        "validation_specs": list(validation_specs),
        "lateral_train_specs": {
            name: list(specs) for name, specs in lateral_specs.items()
        },
        "train_transition_count": len(train_rows),
        "validation_transition_count": sum(len(rows) for rows in validation_runs.values()),
        "score_origin_stride": score_origin_stride,
        "primary_horizon_s": 0.75,
        "primary_horizon_commands": 30,
        "primary_horizon_command_dt_s": 0.025,
        "steering_input_contract": {
            "raw_packet_field": "commanded_steering_norm_k1",
            "transition_input_field": "steering_target_norm_k",
            "transition_input_definition": "previous_packet_command",
            "post_step_raw_field_is_not_used_as_transition_input": True,
        },
        "operating_envelope": {
            "raceline_file": envelope_context["path"],
            "score_classes": ["core", "guard"],
            "stress_in_promotion_percentiles": False,
            "speed_bin_width_mps": envelope_context["speed_bin_width_mps"],
            "curvature_bin_width_radpm": envelope_context[
                "curvature_bin_width_radpm"],
            "core_bin_count": len(envelope_context["core_bins"]),
            "guard_bin_count": len(envelope_context["guard_bins"]),
            "legacy_rectangular_filter": {
                "speed_mps": [2.0, 16.0],
                "max_abs_steering_rad": MAX_ABS_STEERING_RAD,
                "max_abs_lateral_acceleration_mps2": MAX_ABS_LATERAL_ACCEL_MPS2,
                "max_steering_rate_radps": MAX_STEERING_RATE_RADPS,
                "max_speed_steering_product_mps_rad":
                    MAX_SPEED_STEERING_PRODUCT_MPS_RAD,
                "used_for_promotion": False,
            },
        },
        "steering": steering,
        "steering_dynamics_kind_selected": steering_dynamics_kind,
        "lateral_profile_selection": lateral_profile_kind,
        "steering_rate_residual": steering_rate_residual or {
            "status": "not_fitted",
        },
        "lateral_regimes": lateral,
        "longitudinal": longitudinal,
        "coast_diagnostic": _coast_diagnostic(train_rows, base),
        "damping": {
            "unity_linear_damping_per_s": base.linear_damping_per_s,
            "additional_fitted_drag_n_per_mps": 0.0,
            "angular_damping_per_s": base.angular_damping_per_s,
        },
        "scores": {
            "train_profile": train_scores,
            "validation_profile": validation_scores,
            "validation_canonical_scalar_baseline": baseline_validation_scores,
        },
        "candidate_parameters": {
            "steering_dynamics_kind": steering_dynamics_kind,
            "steering_lag_time_constant_s": profile.steering_lag_time_constant_s,
            "lateral_cf_regimes_n_per_rad": profile.lateral_cf_regimes_n_per_rad,
            "lateral_cr_regimes_n_per_rad": profile.lateral_cr_regimes_n_per_rad,
            "lateral_df_regimes_n": profile.lateral_df_regimes_n,
            "lateral_dr_regimes_n": profile.lateral_dr_regimes_n,
            "combined_slip_gain": profile.combined_slip_gain,
            "steering_rate_force_gain_n_per_radps": (
                profile.steering_rate_force_gain_n_per_radps),
            "steering_rate_moment_gain_nm_per_radps": (
                profile.steering_rate_moment_gain_nm_per_radps),
            "force_max_regimes_n": profile.force_max_regimes_n,
            "slip_gain_regimes_per_mps": profile.slip_gain_regimes_per_mps,
            "coast_speed_drag_n_per_mps": profile.coast_speed_drag_n_per_mps,
        },
        "acceptance": {
            "native_parity": False,
            "blind_holdout": False,
            "live_runtime_validation": False,
            "production_mpc_migration": False,
        },
        "next_action": (
            "Use the 0.75 s validation score (30 commands at 40 Hz) as the "
            "MPC-relevant gate; do not migrate before independent blind "
            "validation."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return report


def replace_for_baseline(base: PlantParameters) -> PlantParameters:
    """Return the unchanged scalar plant with the same causal input contract."""
    return PlantParameters(
        **{
            **base.__dict__,
            "steering_dynamics_kind": "rate_limited",
            "steering_lag_time_constant_s": 0.0,
            "tire_model": "tanh",
            "force_max_regimes_n": None,
            "slip_gain_regimes_per_mps": None,
            "lateral_cf_regimes_n_per_rad": None,
            "lateral_cr_regimes_n_per_rad": None,
            "lateral_df_regimes_n": None,
            "lateral_dr_regimes_n": None,
            "steering_rate_force_gain_n_per_radps": 0.0,
            "steering_rate_moment_gain_nm_per_radps": 0.0,
        })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-run", action="append", required=True,
                        help="run directory, optionally suffixed with @start:end")
    parser.add_argument("--validation-run", action="append", required=True,
                        help="held-out run directory, optionally suffixed with @start:end")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raceline", type=Path, default=DEFAULT_RACELINE,
                        help="production raceline used for CORE/GUARD scoring")
    parser.add_argument(
        "--steering-dynamics", choices=("instantaneous", "first_order", "rate_limited"),
        default="instantaneous",
        help=("effective-steering state used in the causal rollout; the lag "
              "fit is always reported, but instantaneous is the default when "
              "the source does not identify a lag robustly"))
    parser.add_argument(
        "--lateral-run", action="append", default=[],
        help=("regime-specific lateral training spec in the form "
              "low_2_8=RUN, mid_8_12=RUN, or high_12_16=RUN; may be repeated"))
    parser.add_argument(
        "--score-origin-stride", type=int, default=1,
        help="evaluate every Nth origin; keep 1 for the final acceptance run")
    parser.add_argument(
        "--lateral-profile", choices=("fitted", "canonical"), default="fitted",
        help=("select the regime-fitted lateral profile or retain the canonical "
              "scalar lateral law while evaluating the longitudinal candidate"))
    parser.add_argument(
        "--fit-steering-rate-residual", action="store_true",
        help=("fit the bounded offline steering-transition force/moment residual "
              "using only causal one-step transitions"))
    parser.add_argument(
        "--lateral-peak-upper-n", type=float,
        default=DEFAULT_LATERAL_PEAK_UPPER_N,
        help=("diagnostic optimizer upper bound for per-axle lateral tire "
              "peak force; increase this to test whether the default bound "
              "is active, but compare held-out prediction before accepting"))
    args = parser.parse_args()
    lateral_specs: dict[str, list[str]] = {}
    for entry in args.lateral_run:
        if "=" not in entry:
            raise ValueError("--lateral-run must use REGIME=RUN syntax")
        regime, spec = entry.split("=", 1)
        if regime not in REGIME_NAMES or not spec:
            raise ValueError(
                f"--lateral-run regime must be one of {REGIME_NAMES}")
        lateral_specs.setdefault(regime, []).append(spec)
    report = fit(
        args.train_run, args.validation_run, args.output,
        steering_dynamics_kind=args.steering_dynamics,
        lateral_specs=lateral_specs or None,
        score_origin_stride=args.score_origin_stride,
        lateral_profile_kind=args.lateral_profile,
        fit_steering_rate_residual=args.fit_steering_rate_residual,
        raceline_path=args.raceline,
        lateral_peak_upper_n=args.lateral_peak_upper_n)
    print(json.dumps({
        "output": str(args.output),
        "status": report["status"],
        "steering_lag_time_constant_s": report["steering"][
            "time_constant_s"],
        "validation_0.75s": report["scores"]["validation_profile"]["0.75s"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
