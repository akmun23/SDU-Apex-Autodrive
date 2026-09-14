#!/usr/bin/env python3
"""Compare offline longitudinal plant hypotheses on source-aligned data.

This tool is intentionally offline-only.  It compares a direct throttle
surface, an equilibrium-speed servo, and a wheel-slip force model using the
canonical ``model_transition_v4.csv`` tables.  It does not alter the native
MPC, runtime odometry, or simulator.

The recursive score is a causal longitudinal subsystem score.  It does not
use future measured body speed or wheel speed.  A separate diagnostic score
can condition the body acceleration on recorded lateral ``r*v`` coupling,
but that score is explicitly not full-vehicle MPC acceptance.  The
wheel-slip candidate uses measured encoder-derived wheel speed only to fit and
initialize the offline subsystem state.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


# Unity diagnostic anchor.  The previous 3.314 kg value belonged to the
# obsolete replay candidate and must not be reused for new fits.
MASS_KG = 3.470
MIN_DT_S = 0.015
MAX_DT_S = 0.035
HORIZONS_S = (0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00)


def _horizon_key(horizon: float) -> str:
    """Use labels that retain the 25 ms source-step horizon exactly."""
    if abs(horizon - 0.025) < 1.0e-9:
        return "0.025s"
    return f"{horizon:.2f}s"


def _read_rows(root: Path, names: Sequence[str]) -> dict[str, list[dict[str, float]]]:
    result: dict[str, list[dict[str, float]]] = {}
    required = {
        "dt_sim_s", "u_k_mps", "v_k_mps", "r_k_radps",
        "applied_throttle_norm_k1", "applied_steering_rad_k1",
        "u_k1_mps", "segment_id", "wheel_speed_mps_k1",
    }
    for name in names:
        path = root / name / "assembled" / "model_transition_v4.csv"
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            fields = set(reader.fieldnames or ())
            missing = sorted(required.difference(fields))
            if missing:
                raise ValueError(f"{path} is missing fields: {missing}")
            rows: list[dict[str, float]] = []
            previous_segment: int | None = None
            previous_wheel: float | None = None
            for raw in reader:
                row = {
                    key: (float(raw[key])
                          if raw[key] not in (None, "") else math.nan)
                    for key in required
                }
                row["segment_id"] = int(round(row["segment_id"]))
                row["wheel_k_mps"] = (
                    previous_wheel
                    if previous_segment == row["segment_id"] else math.nan)
                rows.append(row)
                previous_segment = row["segment_id"]
                previous_wheel = row.get("wheel_speed_mps_k1", math.nan)
            result[name] = rows
    return result


def _run_groups(runs: dict[str, list[dict[str, float]]]) -> dict[tuple[str, int], list[dict[str, float]]]:
    groups: dict[tuple[str, int], list[dict[str, float]]] = {}
    for name, rows in runs.items():
        for row in rows:
            groups.setdefault((name, int(row["segment_id"])), []).append(row)
    return groups


def _straight(row: dict[str, float]) -> bool:
    return abs(row["applied_steering_rad_k1"]) <= 0.01


def _constant_throttle(rows: Sequence[dict[str, float]]) -> bool:
    values = np.asarray([row["applied_throttle_norm_k1"] for row in rows])
    return len(values) >= 10 and float(np.ptp(values)) <= 1.0e-5


def _fit_equilibrium_map(runs: dict[str, list[dict[str, float]]]) -> dict[str, object]:
    points: dict[float, list[float]] = {}
    # Identification campaigns contain a throttle step followed by a coast
    # tail before the next reset, so a whole segment is not constant throttle.
    # Group the source-aligned steady-speed observations by their applied
    # throttle instead, using the upper tail of each group to avoid launch
    # transients. Moving zero-throttle coast is deliberately excluded.
    for row in (row for rows in runs.values() for row in rows):
        throttle = row["applied_throttle_norm_k1"]
        if throttle > 0.005 and _straight(row) and row["u_k1_mps"] >= 0.0:
            points.setdefault(round(throttle, 6), []).append(row["u_k1_mps"])
    point_counts = {str(throttle): len(speeds)
                    for throttle, speeds in points.items()}
    for throttle, speeds in list(points.items()):
        if len(speeds) >= 8:
            points[throttle] = [float(np.percentile(speeds, 95))]
    # Zero throttle is a defined physical equilibrium even though moving
    # coast segments are not equilibrium observations.
    throttle_values = [0.0]
    speed_values = [0.0]
    for throttle in sorted(points):
        throttle_values.append(throttle)
        speed_values.append(float(np.median(points[throttle])))
    if len(throttle_values) < 3:
        raise ValueError("insufficient constant-throttle equilibrium segments")
    # A single transient-only throttle level can otherwise create a
    # nonphysical decrease in the inverse map (the 0.70 sample is one such
    # case in the retained campaign). Preserve the measured points but enforce
    # the known forward-actuator monotonicity before interpolation.
    speed_values = list(np.maximum.accumulate(np.asarray(speed_values)))
    return {
        "throttle_norm": throttle_values,
        "u_inf_mps": speed_values,
        "point_counts": point_counts,
        "fit_source": "source_aligned_95th_percentile_by_applied_throttle",
    }


def _map_value(model_map: dict[str, object], throttle: float) -> float:
    throttle_values = np.asarray(model_map["throttle_norm"], dtype=float)
    speed_values = np.asarray(model_map["u_inf_mps"], dtype=float)
    return float(np.interp(throttle, throttle_values, speed_values))


def _direct_features(row: dict[str, float], u: float) -> np.ndarray:
    throttle = row["applied_throttle_norm_k1"]
    delta = row["applied_steering_rad_k1"]
    return np.asarray([
        1.0, u, u * u, throttle, throttle * u, throttle * throttle,
        delta * delta,
    ], dtype=float)


def _observed_acceleration(row: dict[str, float]) -> float:
    return ((row["u_k1_mps"] - row["u_k_mps"]) / row["dt_sim_s"] +
            row["r_k_radps"] * row["v_k_mps"])


def _fit_direct_surface(rows: Sequence[dict[str, float]]) -> dict[str, object]:
    selected = [row for row in rows if _straight(row)]
    features = np.vstack([_direct_features(row, row["u_k_mps"])
                          for row in selected])
    target = np.asarray([_observed_acceleration(row) for row in selected])
    ridge = 1.0e-6 * np.eye(features.shape[1])
    coefficients = np.linalg.solve(features.T @ features + ridge,
                                   features.T @ target)
    return {
        "kind": "direct_empirical_throttle_surface",
        "feature_names": ["bias", "u", "u_squared", "throttle",
                           "throttle_times_u", "throttle_squared",
                           "steering_squared"],
        "coefficients": [float(value) for value in coefficients],
        "fit_samples": len(selected),
    }


def _direct_acceleration(model: dict[str, object], row: dict[str, float], u: float) -> float:
    features = _direct_features(row, u)
    return float(features @ np.asarray(model["coefficients"], dtype=float))


def _fit_servo_gain(rows: Sequence[dict[str, float]], model_map: dict[str, object]) -> dict[str, object]:
    selected = [row for row in rows if _straight(row)]
    terms = []
    targets = []
    for row in selected:
        error = _map_value(model_map, row["applied_throttle_norm_k1"]) - row["u_k_mps"]
        if abs(error) >= 0.10:
            terms.append(error)
            targets.append(_observed_acceleration(row))
    if not terms:
        raise ValueError("insufficient equilibrium-speed error for servo fit")
    term_array = np.asarray(terms)
    target_array = np.asarray(targets)
    gain = max(0.0, float(term_array @ target_array /
                          max(term_array @ term_array, 1.0e-12)))
    return {
        "kind": "equilibrium_speed_servo",
        "gain_per_second": gain,
        "equilibrium_map": model_map,
        "fit_samples": len(terms),
    }


def _servo_acceleration(model: dict[str, object], row: dict[str, float], u: float) -> float:
    target = _map_value(model["equilibrium_map"], row["applied_throttle_norm_k1"])
    return float(model["gain_per_second"]) * (target - u)


def _fit_wheel_force(rows: Sequence[dict[str, float]], model_map: dict[str, object]) -> dict[str, object]:
    selected = [
        row for row in rows
        if _straight(row) and math.isfinite(row["wheel_k_mps"]) and
        math.isfinite(row["wheel_speed_mps_k1"])
    ]
    if len(selected) < 100:
        raise ValueError("insufficient encoder-derived wheel transitions")
    slip = np.asarray([
        0.5 * (row["wheel_k_mps"] + row["wheel_speed_mps_k1"]) -
        0.5 * (row["u_k_mps"] + row["u_k1_mps"])
        for row in selected
    ])
    speed = np.asarray([
        0.5 * (row["u_k_mps"] + row["u_k1_mps"])
        for row in selected
    ])
    force = np.asarray([MASS_KG * _observed_acceleration(row)
                        for row in selected])
    best: tuple[float, float, float, float] | None = None
    for k_x in np.geomspace(0.05, 100.0, 1200):
        design = np.column_stack((np.tanh(k_x * slip), -speed))
        coefficients, *_ = np.linalg.lstsq(design, force, rcond=None)
        if coefficients[0] <= 0.0 or coefficients[1] < 0.0:
            continue
        residual = force - design @ coefficients
        mae = float(np.mean(np.abs(residual)))
        candidate = (mae, k_x, float(coefficients[0]), float(coefficients[1]))
        if best is None or candidate[0] < best[0]:
            best = candidate
    if best is None:
        raise ValueError("wheel-force fit produced no positive force parameters")
    _, k_x, f_max, c_v = best
    return {
        "kind": "wheel_slip_force",
        "mass_kg": MASS_KG,
        "force_max_n": f_max,
        "slip_gain_per_mps": k_x,
        "coast_speed_drag_n_per_mps": c_v,
        "equilibrium_map": model_map,
        "fit_samples": len(selected),
        "fit_force_mae_n": best[0],
    }


def _fit_wheel_time_constant(rows: Sequence[dict[str, float]], model_map: dict[str, object]) -> float:
    selected = [
        row for row in rows
        if _straight(row) and math.isfinite(row["wheel_k_mps"]) and
        math.isfinite(row["wheel_speed_mps_k1"])
    ]
    best_tau = 0.1
    best_mae = float("inf")
    for tau in np.geomspace(0.005, 2.0, 400):
        errors = []
        for row in selected:
            target = _map_value(model_map, row["applied_throttle_norm_k1"])
            fraction = min(1.0, row["dt_sim_s"] / tau)
            prediction = row["wheel_k_mps"] + fraction * (target - row["wheel_k_mps"])
            errors.append(abs(prediction - row["wheel_speed_mps_k1"]))
        mae = float(np.mean(errors)) if errors else float("inf")
        if mae < best_mae:
            best_mae = mae
            best_tau = float(tau)
    return best_tau


def _wheel_features(row: dict[str, float], wheel: float,
                    body_speed: float) -> np.ndarray:
    throttle = row["applied_throttle_norm_k1"]
    return np.asarray([
        1.0, wheel, throttle, wheel * throttle,
        throttle * throttle, body_speed,
    ], dtype=float)


def _fit_wheel_dynamics(rows: Sequence[dict[str, float]]) -> dict[str, object]:
    selected = [
        row for row in rows
        if _straight(row) and math.isfinite(row["wheel_k_mps"]) and
        math.isfinite(row["wheel_speed_mps_k1"])
    ]
    features = np.vstack([_wheel_features(row, row["wheel_k_mps"],
                                          row["u_k_mps"])
                          for row in selected])
    target = np.asarray([row["wheel_speed_mps_k1"] for row in selected])
    ridge = 1.0e-6 * np.eye(features.shape[1])
    coefficients = np.linalg.solve(features.T @ features + ridge,
                                   features.T @ target)
    errors = np.abs(features @ coefficients - target)
    return {
        "kind": "identified_discrete_wheel_speed_state",
        "feature_names": ["bias", "wheel_speed", "throttle",
                           "wheel_speed_times_throttle", "throttle_squared",
                           "body_speed"],
        "coefficients": [float(value) for value in coefficients],
        "fit_samples": len(selected),
        "fit_wheel_speed_mae_mps": float(np.mean(errors)),
        "fit_wheel_speed_p95_mps": float(np.percentile(errors, 95)),
    }


def _wheel_acceleration(model: dict[str, object], row: dict[str, float],
                        u: float, wheel: float,
                        lateral_coupling: float) -> tuple[float, float]:
    target = _map_value(model["equilibrium_map"], row["applied_throttle_norm_k1"])
    fraction = min(1.0, row["dt_sim_s"] / float(model["wheel_time_constant_s"]))
    wheel_next = wheel + fraction * (target - wheel)
    slip = 0.5 * (wheel + wheel_next) - u
    force = (float(model["force_max_n"]) *
             math.tanh(float(model["slip_gain_per_mps"]) * slip) -
             float(model["coast_speed_drag_n_per_mps"]) * u)
    return force / MASS_KG + lateral_coupling, wheel_next


def _wheel_acceleration_dynamic(model: dict[str, object], row: dict[str, float],
                                u: float, wheel: float,
                                lateral_coupling: float) -> tuple[float, float]:
    wheel_coefficients = np.asarray(model["wheel_dynamics"]["coefficients"],
                                    dtype=float)
    wheel_next = max(0.0, float(_wheel_features(row, wheel, u) @
                                wheel_coefficients))
    slip = 0.5 * (wheel + wheel_next) - u
    force = (float(model["force_max_n"]) *
             math.tanh(float(model["slip_gain_per_mps"]) * slip) -
             float(model["coast_speed_drag_n_per_mps"]) * u)
    return force / MASS_KG + lateral_coupling, wheel_next


def _predict_acceleration(model_name: str, model: dict[str, object],
                          row: dict[str, float], u: float,
                          wheel: float | None,
                          lateral_coupling: float = 0.0
                          ) -> tuple[float, float | None]:
    if model_name == "direct":
        return _direct_acceleration(model, row, u), wheel
    if model_name == "servo":
        return _servo_acceleration(model, row, u), wheel
    if wheel is None or not math.isfinite(wheel):
        return math.nan, wheel
    if model_name == "wheel_dynamic":
        return _wheel_acceleration_dynamic(model, row, u, wheel,
                                           lateral_coupling)
    return _wheel_acceleration(model, row, u, wheel, lateral_coupling)


def _score_one_step(runs: dict[str, list[dict[str, float]]], model_name: str,
                    model: dict[str, object]) -> dict[str, object]:
    errors: list[float] = []
    actual: list[float] = []
    for rows in runs.values():
        for row in rows:
            wheel = row["wheel_k_mps"] if math.isfinite(row["wheel_k_mps"]) else None
            prediction, _ = _predict_acceleration(
                model_name, model, row, row["u_k_mps"], wheel,
                lateral_coupling=row["r_k_radps"] * row["v_k_mps"])
            if math.isfinite(prediction):
                errors.append(abs(prediction - _observed_acceleration(row)))
                actual.append(abs(_observed_acceleration(row)))
    return _stats(errors, actual)


def _score_recursive(runs: dict[str, list[dict[str, float]]], model_name: str,
                     model: dict[str, object]) -> dict[str, object]:
    output: dict[str, object] = {}
    for horizon in HORIZONS_S:
        errors: list[float] = []
        actual: list[float] = []
        for rows in runs.values():
            for origin, origin_row in enumerate(rows):
                wheel = (origin_row["wheel_k_mps"]
                         if math.isfinite(origin_row["wheel_k_mps"]) else None)
                if model_name in {"wheel", "wheel_dynamic"} and wheel is None:
                    continue
                elapsed = 0.0
                index = origin
                segment = int(origin_row["segment_id"])
                while index < len(rows) and elapsed < horizon - 1.0e-10:
                    row = rows[index]
                    if int(row["segment_id"]) != segment:
                        break
                    current_u = (origin_row["u_k_mps"]
                                 if index == origin else predicted_u)
                    # This strict recursive score intentionally uses no
                    # recorded future r/v. The full plant owns that coupling;
                    # using it here would make this a GT-conditioned score.
                    acceleration, wheel = _predict_acceleration(
                        model_name, model, row, current_u, wheel,
                        lateral_coupling=0.0)
                    if not math.isfinite(acceleration):
                        break
                    predicted_u = max(0.0, current_u + row["dt_sim_s"] * acceleration)
                    elapsed += row["dt_sim_s"]
                    index += 1
                if elapsed < horizon - 1.0e-10 or index == origin:
                    continue
                truth = rows[index - 1]
                errors.append(abs(predicted_u - truth["u_k1_mps"]))
                actual.append(abs(truth["u_k1_mps"]))
        output[_horizon_key(horizon)] = _stats(errors, actual)
    return output


def _stats(errors: Iterable[float], actual: Iterable[float]) -> dict[str, object]:
    pairs = sorted((float(error), float(value))
                   for error, value in zip(errors, actual)
                   if math.isfinite(error) and math.isfinite(value))
    values = [pair[0] for pair in pairs]
    if not values:
        return {"count": 0, "mae_mps2_or_mps": None, "p95": None,
                "max": None, "relative_p95": None}
    relative = sorted(error / value for error, value in pairs if value >= 1.0)
    p95_index = min(len(values) - 1, int(math.ceil(0.95 * len(values))) - 1)
    relative_index = (min(len(relative) - 1,
                          int(math.ceil(0.95 * len(relative))) - 1)
                      if relative else None)
    return {
        "count": len(values),
        "mae_mps2_or_mps": float(np.mean(values)),
        "p95": values[p95_index],
        "max": values[-1],
        "relative_p95": relative[relative_index] if relative_index is not None else None,
    }


def _names(value: str) -> list[str]:
    names = [item.strip() for item in value.split(",") if item.strip()]
    if not names:
        raise ValueError("run list cannot be empty")
    return names


def fit(root: Path, train_names: Sequence[str], validation_names: Sequence[str],
        output: Path) -> dict[str, object]:
    if set(train_names).intersection(validation_names):
        raise ValueError("train and validation runs overlap")
    train = _read_rows(root, train_names)
    validation = _read_rows(root, validation_names)
    train_rows = [row for rows in train.values() for row in rows]
    model_map = _fit_equilibrium_map(train)
    direct = _fit_direct_surface(train_rows)
    servo = _fit_servo_gain(train_rows, model_map)
    wheel = _fit_wheel_force(train_rows, model_map)
    wheel["wheel_time_constant_s"] = _fit_wheel_time_constant(train_rows, model_map)
    wheel_dynamic = dict(wheel)
    wheel_dynamic["kind"] = "wheel_slip_force_with_discrete_wheel_state"
    wheel_dynamic.pop("wheel_time_constant_s", None)
    wheel_dynamic["wheel_dynamics"] = _fit_wheel_dynamics(train_rows)

    models = {
        "direct": direct,
        "servo": servo,
        "wheel": wheel,
        "wheel_dynamic": wheel_dynamic,
    }
    report: dict[str, object] = {
        "schema_version": 2,
        "status": "offline_benchmark_not_runtime_validated",
        "ground_truth_use": "offline_identification_and_scoring_only",
        "train_runs": list(train_names),
        "validation_runs": list(validation_names),
        "train_transition_count": len(train_rows),
        "validation_transition_count": sum(len(rows) for rows in validation.values()),
        "recursive_score_boundary": "segment_id_hard_stop",
        "recursive_score_scope": "longitudinal_u_causal_zero_lateral_coupling",
        "recursive_prediction_uses_future_gt": False,
        "one_step_prediction_uses_future_gt": False,
        "measured_state_use": (
            "origin_u_and_origin_encoder_wheel_speed_only; measured states "
            "are used for one-step fitting/scoring, not recursive propagation"
        ),
        "models": {},
        "next_action": (
            "Use the best longitudinal candidate only as an input to native "
            "recursive replay; do not copy parameters into MPC until full "
            "vehicle blind validation passes."
        ),
    }
    for name, model in models.items():
        report["models"][name] = {
            "parameters": model,
            "train_one_step_acceleration": _score_one_step(train, name, model),
            "validation_one_step_acceleration": _score_one_step(
                validation, name, model),
            "train_recursive_u": _score_recursive(train, name, model),
            "validation_recursive_u": _score_recursive(validation, name, model),
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted-root", type=Path, required=True)
    parser.add_argument("--train-runs", required=True)
    parser.add_argument("--validation-runs", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = fit(args.accepted_root, _names(args.train_runs),
                 _names(args.validation_runs), args.output)
    print(json.dumps({
        "output": str(args.output),
        "status": report["status"],
        "models": {
            name: {
                "validation_one_step": value["validation_one_step_acceleration"],
                "validation_recursive_u_0.50s": value[
                    "validation_recursive_u"]["0.50s"],
            }
            for name, value in report["models"].items()
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
