#!/usr/bin/env python3
"""Fit and score an offline vehicle-model candidate from accepted runs.

This tool deliberately stops before changing the production MPC model.  It
fits an interpretable, smooth transition model to the accepted source-time
tables and scores it with open-loop recursive rollouts.  Simulator ground
truth is read only by this offline tool; it is never published or passed to a
runtime node.

The fitted plant boundary is explicit:

    [applied throttle, applied steering angle] -> [u, v, yaw rate]

Steering-command-to-feedback dynamics are fitted separately from bridge
request tables.  Keeping that separation is important because the current
production command path contains a software speed controller and because the
transition assembler aligns the applied command carried by packet k+1 with
the state transition k -> k+1.

The dynamic channels are fitted as direct one-step transitions of the form

    z[k+1] - z[k] = dt[k] * phi(z[k], input[k]) @ theta

where ``phi`` is a low-order, symmetry-aware polynomial basis.  Position and
yaw use exact planar constant-twist integration; only body dynamic states are
identified.  This is a candidate suitable for comparison with the existing
BachelorProject bicycle model, not an accepted runtime model.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Iterable, Sequence

import numpy as np


MAX_STEERING_RAD = 0.5236
MIN_DT_S = 0.015
MAX_DT_S = 0.035
HORIZONS_S = (0.05, 0.10, 0.25, 0.50, 1.00, 1.50, 2.00)
STATE_NAMES = ("x_m", "y_m", "yaw_rad", "u_mps", "v_mps", "r_radps")


def _finite(raw: str, field: str, path: Path, row_number: int) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}:{row_number}: invalid {field}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{path}:{row_number}: non-finite {field}")
    return value


def _read_csv(path: Path, required: Iterable[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or ())
        missing = sorted(set(required).difference(fields))
        if missing:
            raise ValueError(f"{path} is missing required fields: {missing}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path} contains no rows")
    return rows


def _read_transitions(run_dir: Path) -> list[dict[str, float]]:
    required = {
        "simulation_time_k_s", "simulation_time_k1_s", "dt_sim_s",
        "x_k_m", "y_k_m", "yaw_k_rad", "u_k_mps", "v_k_mps",
        "r_k_radps", "applied_throttle_norm_k1",
        "applied_steering_rad_k1",
        "simulator_feedback_steering_rad_k1", "reset_epoch", "segment_id",
        "x_k1_m", "y_k1_m", "yaw_k1_rad", "u_k1_mps", "v_k1_mps",
        "r_k1_radps",
    }
    path = run_dir / "assembled" / "model_transition_v4.csv"
    raw = _read_csv(path, required)
    rows: list[dict[str, float]] = []
    for number, row in enumerate(raw, start=2):
        parsed = {
            field: _finite(value, field, path, number)
            for field, value in row.items()
            if field in required
        }
        dt = parsed["dt_sim_s"]
        if not MIN_DT_S <= dt <= MAX_DT_S:
            raise ValueError(
                f"{run_dir}: transition dt {dt} is outside [{MIN_DT_S}, {MAX_DT_S}]")
        rows.append(parsed)
    # The recorder intentionally includes a short post-reset settling period
    # before the first excitation. Those rows are valid telemetry but contain
    # startup residuals that are not representative of a commanded plant
    # transition. Keep the rest of the run, including all zero-throttle coast
    # segments after the first excitation.
    first_excitation = next((index for index, row in enumerate(rows)
                             if abs(row["applied_throttle_norm_k1"]) > 1.0e-8 or
                             abs(row["applied_steering_rad_k1"]) > 1.0e-8), None)
    if first_excitation is None:
        return rows
    return rows[first_excitation:]


def _read_bridge_feedback(run_dir: Path) -> list[dict[str, float]]:
    """Read one row per source physics step for actuator identification."""
    path = run_dir / "bridge_requests.csv"
    required = {
        "simulation_physics_step", "simulation_time_s",
        "commanded_steering_norm", "simulator_feedback_steering_norm",
    }
    raw = _read_csv(path, required)
    result: list[dict[str, float]] = []
    seen_steps: set[int] = set()
    for number, row in enumerate(raw, start=2):
        step = int(_finite(row["simulation_physics_step"],
                           "simulation_physics_step", path, number))
        if step in seen_steps:
            continue
        seen_steps.add(step)
        result.append({
            "step": float(step),
            "time": _finite(row["simulation_time_s"], "simulation_time_s",
                             path, number),
            "command": _finite(row["commanded_steering_norm"],
                                "commanded_steering_norm", path, number),
            # Despite the historical field name, the values in the accepted
            # tables are physical steering radians (e.g. 0.0262 for 0.05 of
            # the 0.5236 rad steering limit).  Preserve the source name in
            # the report and make the unit conversion explicit below.
            "feedback_rad": _finite(
                row["simulator_feedback_steering_norm"],
                "simulator_feedback_steering_norm", path, number),
        })
    result.sort(key=lambda row: row["step"])
    if len(result) < 3:
        raise ValueError(f"{path} has too few unique source steps")
    return result


def _basis(row: dict[str, float], channel: str) -> tuple[list[str], np.ndarray]:
    """Return a bounded, interpretable basis for one derivative channel."""
    u = row["u_k_mps"]
    v = row["v_k_mps"]
    r = row["r_k_radps"]
    delta = row["applied_steering_rad_k1"]
    throttle = row["applied_throttle_norm_k1"]

    if channel == "u":
        # rv and lateral-state terms are retained because the body-frame
        # longitudinal derivative is not the world-frame acceleration.
        names = (
            "bias", "u", "u_squared", "u_cubed", "throttle",
            "throttle_times_u", "throttle_times_u_squared",
            "throttle_squared", "steering_squared",
            "steering_squared_times_u", "v_times_r", "v_squared",
            "r_squared",
        )
        values = (
            1.0, u, u * u, u * u * u, throttle, throttle * u,
            throttle * u * u, throttle * throttle, delta * delta,
            delta * delta * u, v * r, v * v, r * r,
        )
    elif channel in ("v", "r"):
        # This form has no free lateral drift: with v=r=delta=0 the lateral
        # channels remain at equilibrium. Coefficients vary smoothly with u.
        # Odd steering terms preserve left/right antisymmetry where the data
        # supports it, while delta*abs(delta) allows a mild saturation slope.
        names = (
            "v", "v_times_u", "v_times_u_squared",
            "r", "r_times_u", "r_times_u_squared",
            "steering", "steering_times_u", "steering_times_u_squared",
            "steering_times_abs_steering",
            "steering_times_abs_steering_times_u",
        )
        steering_abs = abs(delta)
        values = (
            v, v * u, v * u * u, r, r * u, r * u * u,
            delta, delta * u, delta * u * u, delta * steering_abs,
            delta * steering_abs * u,
        )
    else:
        raise ValueError(f"unknown model channel: {channel}")
    return list(names), np.asarray(values, dtype=float)


def _fit_robust_transition(
        rows: Sequence[dict[str, float]], channel: str,
        ridge: float = 1.0e-5) -> dict[str, object]:
    names, first = _basis(rows[0], channel)
    features = np.vstack([
        _basis(row, channel)[1] * row["dt_sim_s"] for row in rows
    ])
    key0 = {"u": "u_k_mps", "v": "v_k_mps", "r": "r_k_radps"}[channel]
    key1 = {"u": "u_k1_mps", "v": "v_k1_mps", "r": "r_k1_radps"}[channel]
    targets = np.asarray([row[key1] - row[key0] for row in rows], dtype=float)

    # Column scaling makes the small ridge penalty meaningful across the
    # throttle, velocity, and steering terms. Three Huber reweighting passes
    # stop collision/reset-adjacent outliers from dominating a smooth model.
    scale = np.maximum(np.sqrt(np.mean(features * features, axis=0)), 1.0e-8)
    scaled = features / scale
    weights = np.ones(len(rows), dtype=float)
    coefficient_scaled = np.zeros(len(names), dtype=float)
    for _ in range(3):
        weighted = scaled * weights[:, None]
        lhs = scaled.T @ weighted + ridge * np.eye(len(names))
        rhs = scaled.T @ (weights * targets)
        coefficient_scaled = np.linalg.solve(lhs, rhs)
        residuals = targets - scaled @ coefficient_scaled
        robust_scale = 1.4826 * float(np.median(np.abs(residuals)))
        robust_scale = max(robust_scale, 1.0e-5)
        weights = np.minimum(1.0, 1.5 * robust_scale / np.maximum(
            np.abs(residuals), 1.0e-12))

    coefficients = coefficient_scaled / scale
    fitted = features @ coefficients
    residuals = targets - fitted
    return {
        "channel": channel,
        "feature_names": names,
        "coefficients": [float(value) for value in coefficients],
        "samples": len(rows),
        "transition_error": _stats(
            np.abs(residuals), np.abs(targets)),
        "fit_method": "direct_transition_robust_ridge",
        "ridge": ridge,
        "column_scale": [float(value) for value in scale],
    }


def _stats(errors: Iterable[float], actual: Iterable[float]) -> dict[str, object]:
    pairs = sorted(
        (float(abs(error)), float(abs(value)))
        for error, value in zip(errors, actual)
        if math.isfinite(error) and math.isfinite(value))
    values = [error for error, _ in pairs]
    if not pairs:
        return {"count": 0, "mae": None, "p95": None, "max": None,
                "relative_p95": None, "relative_eligible_count": 0}
    relative = sorted(
        error / actual_value
        for error, actual_value in pairs
        if actual_value >= 1.0)
    index = min(len(values) - 1, int(math.ceil(0.95 * len(values))) - 1)
    relative_index = (min(len(relative) - 1,
                          int(math.ceil(0.95 * len(relative))) - 1)
                      if relative else None)
    return {
        "count": len(values),
        "mae": float(np.mean(values)),
        "p95": values[index],
        "max": values[-1],
        "relative_eligible_count": len(relative),
        "relative_p95": (relative[relative_index]
                          if relative_index is not None else None),
    }


def _channel_prediction(row: dict[str, float], channel: str,
                        model: dict[str, object]) -> float:
    _, values = _basis(row, channel)
    coefficients = np.asarray(model["coefficients"], dtype=float)
    return float(row["dt_sim_s"] * values @ coefficients)


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _pose_step(state: np.ndarray, dt: float) -> tuple[float, float, float]:
    """Integrate planar pose for a constant body twist over dt."""
    x, y, yaw, u, v, r = state
    angle = r * dt
    if abs(r) < 1.0e-8:
        body_dx = u * dt
        body_dy = v * dt
    else:
        body_dx = (u * math.sin(angle) + v * (math.cos(angle) - 1.0)) / r
        body_dy = (u * (1.0 - math.cos(angle)) + v * math.sin(angle)) / r
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    return (
        x + cos_yaw * body_dx - sin_yaw * body_dy,
        y + sin_yaw * body_dx + cos_yaw * body_dy,
        yaw + angle,
    )


def _predict(state: np.ndarray, row: dict[str, float], models: dict[str, object]) -> np.ndarray:
    pose = _pose_step(state, row["dt_sim_s"])
    next_state = np.asarray([
        pose[0], pose[1], pose[2],
        state[3] + _channel_prediction(row, "u", models["u"]),
        state[4] + _channel_prediction(row, "v", models["v"]),
        state[5] + _channel_prediction(row, "r", models["r"]),
    ], dtype=float)
    return next_state


def _state_errors(predicted: np.ndarray, truth: dict[str, float]) -> dict[str, float]:
    return {
        "position_m": math.hypot(predicted[0] - truth["x_k1_m"],
                                  predicted[1] - truth["y_k1_m"]),
        "heading_rad": abs(_wrap(predicted[2] - truth["yaw_k1_rad"])),
        "u_mps": abs(predicted[3] - truth["u_k1_mps"]),
        "v_mps": abs(predicted[4] - truth["v_k1_mps"]),
        "yaw_rate_radps": abs(predicted[5] - truth["r_k1_radps"]),
    }


def _one_step_scores(
        runs: dict[str, list[dict[str, float]]],
        models: dict[str, object]) -> dict[str, object]:
    errors: dict[str, list[float]] = {key: [] for key in (
        "position_m", "heading_rad", "u_mps", "v_mps", "yaw_rate_radps")}
    actual: dict[str, list[float]] = {key: [] for key in errors}
    actual_keys = {
        "position_m": "u_k1_mps", "heading_rad": "yaw_k1_rad",
        "u_mps": "u_k1_mps", "v_mps": "v_k1_mps",
        "yaw_rate_radps": "r_k1_radps",
    }
    for rows in runs.values():
        for row in rows:
            state = np.asarray([
                row["x_k_m"], row["y_k_m"], row["yaw_k_rad"],
                row["u_k_mps"], row["v_k_mps"], row["r_k_radps"],
            ], dtype=float)
            row_errors = _state_errors(_predict(state, row, models), row)
            for key, value in row_errors.items():
                errors[key].append(value)
                actual[key].append(row[actual_keys[key]])
    return {key: _stats(errors[key], actual[key]) for key in errors}


def _recursive_scores(
        runs: dict[str, list[dict[str, float]]],
        models: dict[str, object]) -> dict[str, object]:
    output: dict[str, object] = {}
    fields = ("position_m", "heading_rad", "u_mps", "v_mps", "yaw_rate_radps")
    truth_keys = {
        "position_m": "u_k1_mps", "heading_rad": "yaw_k1_rad",
        "u_mps": "u_k1_mps", "v_mps": "v_k1_mps",
        "yaw_rate_radps": "r_k1_radps",
    }
    for horizon in HORIZONS_S:
        errors: dict[str, list[float]] = {key: [] for key in fields}
        actual: dict[str, list[float]] = {key: [] for key in fields}
        for rows in runs.values():
            for origin in range(len(rows)):
                state = np.asarray([
                    rows[origin]["x_k_m"], rows[origin]["y_k_m"],
                    rows[origin]["yaw_k_rad"], rows[origin]["u_k_mps"],
                    rows[origin]["v_k_mps"], rows[origin]["r_k_radps"],
                ], dtype=float)
                elapsed = 0.0
                index = origin
                origin_segment = int(rows[origin]["segment_id"])
                crossed_segment_boundary = False
                while index < len(rows) and elapsed < horizon - 1.0e-10:
                    if int(rows[index]["segment_id"]) != origin_segment:
                        crossed_segment_boundary = True
                        break
                    state = _predict(state, rows[index], models)
                    elapsed += rows[index]["dt_sim_s"]
                    index += 1
                if (index == origin or index > len(rows) or
                        crossed_segment_boundary or elapsed < horizon - 1.0e-10):
                    continue
                # The last transition has produced the state represented by
                # row index-1's k+1 fields.
                truth = rows[index - 1]
                row_errors = _state_errors(state, truth)
                for key, value in row_errors.items():
                    errors[key].append(value)
                    actual[key].append(truth[truth_keys[key]])
        output[f"{horizon:.2f}s"] = {
            key: _stats(errors[key], actual[key]) for key in fields
        }
    return output


def _fit_steering_actuator(
        bridge_runs: dict[str, list[dict[str, float]]]) -> dict[str, object]:
    """Fit the smallest measured steering feedback model.

    The candidate is a rate-limited first-order actuator. ``lag_steps`` is
    searched explicitly because bridge request and simulator application are
    not assumed to be simultaneous.
    """
    best: dict[str, object] | None = None
    # 0.005 rad/s resolution is finer than the source steering feedback
    # quantization and keeps fitting quick even with all accepted runs.
    rates = np.linspace(0.5, 8.0, 1501)
    for lag_steps in range(4):
        pairs = _actuator_pairs(bridge_runs, lag_steps)
        if not pairs:
            continue
        previous = np.asarray([pair[0] for pair in pairs])
        target = np.asarray([pair[1] for pair in pairs])
        actual = np.asarray([pair[2] for pair in pairs])
        dt = np.asarray([pair[3] for pair in pairs])
        gap = target - previous
        # Keep the grid search bounded in memory.  A fully vectorized
        # rate-by-sample matrix is unnecessarily large for the combined
        # accepted runs.
        best_index = 0
        best_mae = float("inf")
        for index, rate in enumerate(rates):
            prediction = previous + np.clip(gap, -rate * dt, rate * dt)
            mae = float(np.mean(np.abs(prediction - actual)))
            if mae < best_mae:
                best_mae = mae
                best_index = index
        candidate = {
            "rate_radps": float(rates[best_index]),
            "lag_steps": lag_steps,
            "samples": len(pairs),
            "mae_rad": best_mae,
            "p95_rad": float(np.percentile(np.abs(
                previous + np.clip(gap, -rates[best_index] * dt,
                                   rates[best_index] * dt) - actual), 95)),
        }
        if best is None or candidate["mae_rad"] < best["mae_rad"]:
            best = candidate
    if best is None:
        raise ValueError("no valid steering actuator transitions")
    return {
        "state": "actual_steering_rad",
        "input": "commanded_steering_norm",
        "feedback_source_field": "simulator_feedback_steering_norm",
        "feedback_source_unit_interpreted_as": "physical_radians",
        "steering_limit_rad": MAX_STEERING_RAD,
        "model": "rate_limited_first_order",
        **best,
        "status": "candidate_not_runtime_validated",
    }


def _actuator_pairs(
        bridge_runs: dict[str, list[dict[str, float]]],
        lag_steps: int) -> list[tuple[float, float, float, float]]:
    pairs: list[tuple[float, float, float, float]] = []
    for rows in bridge_runs.values():
        for index in range(1, len(rows)):
            command_index = index - 1 - lag_steps
            if command_index < 0:
                continue
            previous = rows[index - 1]
            current = rows[index]
            dt = current["time"] - previous["time"]
            if not MIN_DT_S <= dt <= MAX_DT_S:
                continue
            pairs.append((previous["feedback_rad"],
                          rows[command_index]["command"] * MAX_STEERING_RAD,
                          current["feedback_rad"], dt))
    return pairs


def _score_steering_actuator(
        model: dict[str, object],
        bridge_runs: dict[str, list[dict[str, float]]]) -> dict[str, object]:
    pairs = _actuator_pairs(bridge_runs, int(model["lag_steps"]))
    rate = float(model["rate_radps"])
    errors: list[float] = []
    for previous, target, actual, dt in pairs:
        prediction = previous + max(-rate * dt, min(rate * dt, target - previous))
        errors.append(abs(prediction - actual))
    return {
        "samples": len(errors),
        "mae_rad": float(np.mean(errors)) if errors else None,
        "p95_rad": float(np.percentile(errors, 95)) if errors else None,
        "max_rad": max(errors, default=None),
    }


def _run_names(value: str) -> list[str]:
    names = [item.strip() for item in value.split(",") if item.strip()]
    if not names:
        raise ValueError("run list cannot be empty")
    return names


def _load_runs(root: Path, names: Sequence[str], need_bridge: bool) -> tuple[
        dict[str, list[dict[str, float]]], dict[str, list[dict[str, float]]]]:
    transitions: dict[str, list[dict[str, float]]] = {}
    bridge: dict[str, list[dict[str, float]]] = {}
    for name in names:
        run_dir = root / name
        if not run_dir.is_dir():
            raise ValueError(f"accepted run does not exist: {run_dir}")
        transitions[name] = _read_transitions(run_dir)
        if need_bridge:
            bridge[name] = _read_bridge_feedback(run_dir)
    return transitions, bridge


def fit(root: Path, train_names: Sequence[str], validation_names: Sequence[str],
        output: Path) -> dict[str, object]:
    overlap = sorted(set(train_names).intersection(validation_names))
    if overlap:
        raise ValueError(f"training and validation runs overlap: {overlap}")
    train, train_bridge = _load_runs(root, train_names, need_bridge=True)
    validation, validation_bridge = _load_runs(
        root, validation_names, need_bridge=True)
    train_rows = [row for rows in train.values() for row in rows]
    if len(train_rows) < 100:
        raise ValueError("too few training transitions")
    models = {
        channel: _fit_robust_transition(train_rows, channel)
        for channel in ("u", "v", "r")
    }
    actuator = _fit_steering_actuator(train_bridge)
    actuator["training_score"] = _score_steering_actuator(actuator, train_bridge)
    actuator["validation_score"] = _score_steering_actuator(
        actuator, validation_bridge)

    report: dict[str, object] = {
        "schema_version": 1,
        "status": "candidate_not_runtime_validated",
        "ground_truth_use": "offline_identification_and_scoring_only",
        "train_runs": list(train_names),
        "validation_runs": list(validation_names),
        "train_transition_count": len(train_rows),
        "validation_transition_count": sum(len(rows) for rows in validation.values()),
        "timing_contract": {
            "source_time_used_for_dt": True,
            "source_dt_window_s": [MIN_DT_S, MAX_DT_S],
            "raw_rows_upsampled": False,
        },
        "plant_boundary": {
            "inputs": ["applied_throttle_norm", "applied_steering_rad"],
            "states": list(STATE_NAMES),
            "dynamic_state_update": "identified_body_u_v_yaw_rate_transition",
            "pose_update": "exact_planar_constant_body_twist",
            "steering_actuator_is_separate": True,
        },
        "model": {
            "kind": "continuous_derivative_basis_used_as_direct_transition",
            "channels": models,
            "steering_actuator": actuator,
        },
        "training_scores": {
            "one_step": _one_step_scores(train, models),
            "recursive": _recursive_scores(train, models),
        },
        "validation_scores": {
            "one_step": _one_step_scores(validation, models),
            "recursive": _recursive_scores(validation, models),
        },
        "acceptance": {
            "two_percent_target": True,
            "required_horizons_s": list(HORIZONS_S),
            "recursive_open_loop_required": True,
            "repeatability_floor_required": True,
            "blind_run_required_after_model_freeze": True,
            "production_mpc_parameters_updated": False,
            "runtime_ground_truth_consumed": False,
        },
        "next_action": (
            "Use validation residuals to choose the next structured/discrete "
            "model revision, then collect a new untouched blind track-domain "
            "run before changing the C MPC model."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted-root", type=Path, required=True)
    parser.add_argument("--train-runs", required=True,
                        help="comma-separated accepted run directory names")
    parser.add_argument("--validation-runs", required=True,
                        help="comma-separated whole-run holdout names")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = fit(args.accepted_root, _run_names(args.train_runs),
                 _run_names(args.validation_runs), args.output)
    print(json.dumps({
        "output": str(args.output),
        "status": report["status"],
        "train_transition_count": report["train_transition_count"],
        "validation_transition_count": report["validation_transition_count"],
        "validation_recursive": report["validation_scores"]["recursive"],
        "steering_actuator": report["model"]["steering_actuator"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
