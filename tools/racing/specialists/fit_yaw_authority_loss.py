#!/usr/bin/env python3
"""Fit and whole-capture score the measured low-speed yaw-authority specialist.

Only complete train captures are used to fit the speed/steering equilibrium
map and phase response constants.  The two complete validation captures are
opened only after those values are frozen.  Test/final-test archives are not
read.  Truth signals are labels only; this is an offline identification and
score utility, not a runtime sensor path.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.interpolate import PchipInterpolator


ROOT = Path(__file__).resolve().parents[3]
DT_S = 0.025
SPEED_KNOTS = (2.5, 3.0, 3.5)
MEASURED_STEERING_KNOTS = (0.30, 0.35, 0.42)
SURFACE_STEERING_KNOTS = (0.15, 0.20, 0.21, 0.22, 0.23,
                          0.25, 0.30, 0.35, 0.42, 0.50)
SUPPORT_SPEED_MPS = (2.5, 3.5)
SUPPORT_STEERING_RAD = (0.30, 0.42)
STEERING_BLEND_START_RAD = 0.29
STEERING_BLEND_FULL_START_RAD = 0.30
STEERING_BLEND_FULL_END_RAD = 0.42
STEERING_BLEND_END_RAD = 0.45
PHASE_RATE_THRESHOLD_RADPS = 0.05
HOLD_RATE_FADEOUT_RADPS = 0.15
LEGACY_TAU_S = 0.015
LEGACY_GAIN_PER_M = 2.95
LEGACY_GAIN_REDUCTION_PER_RAD = 35.6
LEGACY_GAIN_START_RAD = 0.41
LEGACY_GAIN_END_RAD = 0.46

TRAIN_SOURCES = (
    ("y1_targeted_retry", "train",
     "live_runs/racing_model_diagnostics_20261007/y1_low_speed_train_r01/openplane_dynamics.npz"),
    ("race_train_r02", "train",
     "live_runs/derived_dynamics_learning_20260928/race_domain_train_runs_20261001_r02/openplane_dynamics.npz"),
    ("race_train_r03", "train",
     "live_runs/derived_dynamics_learning_20260928/race_domain_train_runs_20261001_r03/openplane_dynamics.npz"),
)
VALIDATION_SOURCES = (
    ("race_validation_r04", "validation",
     "live_runs/derived_dynamics_learning_20260928/race_domain_validation_runs_20261001_r04/openplane_dynamics.npz"),
    ("race_validation_r05", "validation",
     "live_runs/derived_dynamics_learning_20260928/race_domain_validation_runs_20261001_r05/openplane_dynamics.npz"),
)
BOUNDARY_TRAIN_SURFACE = "f1tenth_mpc/config/yaw_response_surface.csv"


def _load_source(name: str, split: str, relative_path: str
                 ) -> dict[str, Any]:
    path = ROOT / relative_path
    manifest_path = path.with_name("manifest.json")
    if not path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"{name}: missing NPZ or whole-run manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest.get("runs", [])
    if not rows:
        raise ValueError(f"{name}: manifest has no run provenance")
    for run in rows:
        if (run.get("effective_split") != split or run.get("aborted")
                or run.get("reason") != "schedule complete"
                or not run.get("clean_stream_and_collision_gate")
                or any(int(value) != 0 for value in run.get("collisions", []))
                or int(run.get("timing_faults", -1)) != 0
                or run.get("quality_failures")
                or run.get("whole_bag_quality_failures")):
            raise ValueError(f"{name}: run failed its frozen {split} quality gate")
    with np.load(path, allow_pickle=False) as archive:
        data = {key: np.asarray(archive[key]) for key in archive.files}
    required = ("frames", "simulator_rigid_state", "sequence_bounds", "dt_s",
                "run_ids", "run_splits")
    if any(key not in data for key in required):
        raise ValueError(f"{name}: incomplete dynamics archive")
    if (len(data["run_ids"]) != 1 or str(data["run_splits"][0]) != split
            or not np.allclose(data["dt_s"], DT_S, rtol=0.0, atol=1.0e-7)
            or data["frames"].shape[1] != 9
            or data["simulator_rigid_state"].shape[1] != 13
            or not np.isfinite(data["frames"]).all()
            or not np.isfinite(data["simulator_rigid_state"]).all()):
        raise ValueError(f"{name}: split, cadence, or signal schema mismatch")
    if split not in ("train", "validation"):
        raise ValueError(f"{name}: test/final-test data are not allowed")
    data["source_name"] = name
    data["split"] = split
    data["path"] = relative_path
    return data


def _series(data: dict[str, Any]) -> dict[str, np.ndarray]:
    frames = data["frames"].astype(np.float64, copy=False)
    rigid = data["simulator_rigid_state"].astype(np.float64, copy=False)
    speed = np.hypot(rigid[:, 7], rigid[:, 8])
    steering = frames[:, 3]
    yaw = rigid[:, 12]
    steering_rate = np.full(len(steering), np.nan, dtype=np.float64)
    for begin_raw, end_raw in data["sequence_bounds"]:
        begin, end = int(begin_raw), int(end_raw)
        steering_rate[begin + 1:end] = np.diff(steering[begin:end]) / DT_S
    return {"speed": speed, "steering": steering, "yaw": yaw,
            "steering_rate": steering_rate}


def _legacy_equilibrium(speed: np.ndarray | float,
                        steering: np.ndarray | float) -> np.ndarray:
    speed_array = np.asarray(speed, dtype=np.float64)
    angle = np.asarray(steering, dtype=np.float64)
    reduction = LEGACY_GAIN_REDUCTION_PER_RAD * np.clip(
        np.abs(angle) - LEGACY_GAIN_START_RAD, 0.0,
        LEGACY_GAIN_END_RAD - LEGACY_GAIN_START_RAD)
    gain = LEGACY_GAIN_PER_M - reduction
    return speed_array * np.tan(angle) * gain


def _phase(steering: np.ndarray, steering_rate: np.ndarray) -> np.ndarray:
    magnitude_rate = np.sign(steering) * steering_rate
    labels = np.full(len(steering), "hold", dtype=object)
    labels[magnitude_rate > PHASE_RATE_THRESHOLD_RADPS] = "turn_in"
    labels[magnitude_rate < -PHASE_RATE_THRESHOLD_RADPS] = "unwind"
    return labels


def _estimate_equilibrium(training: list[dict[str, Any]]) -> tuple[
        dict[tuple[float, int], np.ndarray], list[dict[str, Any]]]:
    curves: dict[tuple[float, int], np.ndarray] = {}
    cell_report: list[dict[str, Any]] = []
    for speed_anchor in SPEED_KNOTS:
        for sign in (-1, 1):
            estimates = []
            for steering_magnitude in MEASURED_STEERING_KNOTS:
                run_medians: list[tuple[str, float, int]] = []
                for data in training:
                    values = _series(data)
                    mask = (
                        (np.abs(values["speed"] - speed_anchor) <= 0.18)
                        & (np.abs(values["steering"] - sign * steering_magnitude) <= 0.012)
                        & (np.abs(values["steering_rate"]) <= 0.10)
                    )
                    selected = sign * values["yaw"][mask]
                    if len(selected) >= 8:
                        run_medians.append((data["source_name"],
                                            float(np.median(selected)),
                                            int(len(selected))))
                if not run_medians:
                    raise ValueError(
                        f"unsupported training cell u={speed_anchor}, "
                        f"steer={sign * steering_magnitude}")
                # Equal weight per independent run avoids a long capture
                # overwhelming the deliberately reset-isolated source.
                response = float(np.median([row[1] for row in run_medians]))
                if response <= 0.0:
                    raise ValueError("signed equilibrium response changed direction")
                estimates.append(response)
                cell_report.append({
                    "speed_anchor_mps": speed_anchor,
                    "turn_sign": sign,
                    "steering_magnitude_rad": steering_magnitude,
                    "signed_yaw_rate_abs_rps": response,
                    "run_support": [
                        {"source": source, "median_rps": median, "samples": count}
                        for source, median, count in run_medians],
                })
            curves[(speed_anchor, sign)] = np.asarray(estimates, dtype=np.float64)
    return curves, cell_report


def _equilibrium(speed: np.ndarray, steering: np.ndarray,
                 curves: dict[tuple[float, int], np.ndarray]) -> np.ndarray:
    speed = np.asarray(speed, dtype=np.float64)
    steering = np.asarray(steering, dtype=np.float64)
    out = _legacy_equilibrium(speed, steering)
    supported = ((speed >= SUPPORT_SPEED_MPS[0])
                 & (speed <= SUPPORT_SPEED_MPS[1])
                 & (np.abs(steering) >= SUPPORT_STEERING_RAD[0])
                 & (np.abs(steering) <= SUPPORT_STEERING_RAD[1]))
    if not np.any(supported):
        return out
    angle_grid = np.asarray(MEASURED_STEERING_KNOTS, dtype=np.float64)
    sign = np.where(steering < 0.0, -1, 1)
    for direction in (-1, 1):
        indices = np.flatnonzero(supported & (sign == direction))
        if not len(indices):
            continue
        abs_angle = np.abs(steering[indices])
        anchor_rates = np.asarray([
            PchipInterpolator(angle_grid, curves[(anchor, direction)],
                              extrapolate=False)(abs_angle)
            for anchor in SPEED_KNOTS], dtype=np.float64)
        local_speed = speed[indices]
        above_first = local_speed >= SPEED_KNOTS[1]
        lower = np.where(above_first, 1, 0)
        fraction = ((local_speed - np.asarray(SPEED_KNOTS)[lower]) /
                    (np.asarray(SPEED_KNOTS)[lower + 1] -
                     np.asarray(SPEED_KNOTS)[lower]))
        low_rate = anchor_rates[lower, np.arange(len(indices))]
        high_rate = anchor_rates[lower + 1, np.arange(len(indices))]
        out[indices] = direction * (low_rate + fraction * (high_rate - low_rate))
    return out


def _phase_taus(training: list[dict[str, Any]],
                curves: dict[tuple[float, int], np.ndarray]
                ) -> tuple[dict[str, float], dict[str, Any], dict[str, float]]:
    phases: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {
        "turn_in": [], "hold": [], "unwind": []}
    counts: dict[str, int] = {key: 0 for key in phases}
    for data in training:
        values = _series(data)
        labels = _phase(values["steering"], values["steering_rate"])
        for begin_raw, end_raw in data["sequence_bounds"]:
            begin, end = int(begin_raw), int(end_raw)
            for index in range(begin, end - 1):
                speed = float(values["speed"][index])
                steering = float(values["steering"][index])
                if not (SUPPORT_SPEED_MPS[0] <= speed <= SUPPORT_SPEED_MPS[1]
                        and SUPPORT_STEERING_RAD[0] <= abs(steering)
                        <= SUPPORT_STEERING_RAD[1]):
                    continue
                phase = str(labels[index])
                eq = float(_equilibrium(np.asarray([speed]),
                                        np.asarray([steering]), curves)[0])
                phases[phase].append((np.asarray([values["yaw"][index]]),
                                      np.asarray([eq]),
                                      np.asarray([values["yaw"][index + 1]])))
                counts[phase] += 1

    taus: dict[str, float] = {}
    diagnostics: dict[str, Any] = {}
    pooled_current: list[np.ndarray] = []
    pooled_equilibrium: list[np.ndarray] = []
    pooled_following: list[np.ndarray] = []
    for name, rows in phases.items():
        if len(rows) < 20:
            raise ValueError(f"only {len(rows)} training transitions for phase {name}")
        current = np.concatenate([row[0] for row in rows])
        equilibrium = np.concatenate([row[1] for row in rows])
        following = np.concatenate([row[2] for row in rows])
        pooled_current.append(current)
        pooled_equilibrium.append(equilibrium)
        pooled_following.append(following)

        def objective(log_tau: float) -> float:
            tau = math.exp(log_tau)
            retention = math.exp(-DT_S / tau)
            prediction = retention * current + (1.0 - retention) * equilibrium
            return float(np.mean((prediction - following) ** 2))

        result = minimize_scalar(objective,
            bounds=(math.log(0.010), math.log(0.75)), method="bounded",
            options={"xatol": 1.0e-8})
        tau = math.exp(float(result.x))
        lower_bound = tau <= 0.0105
        upper_bound = tau >= 0.72
        taus[name] = tau
        diagnostics[name] = {
            "samples": counts[name],
            "tau_s": tau,
            "one_step_rmse_radps": math.sqrt(float(result.fun)),
            "hit_fit_bound": bool(lower_bound or upper_bound),
            "fit_bound": "lower" if lower_bound else "upper" if upper_bound else None,
            "data_volume_warning": (
                "sparse phase; do not promote this phase constant alone"
                if counts[name] < 100 else None),
        }

    current = np.concatenate(pooled_current)
    equilibrium = np.concatenate(pooled_equilibrium)
    following = np.concatenate(pooled_following)

    def pooled_objective(log_tau: float) -> float:
        retention = math.exp(-DT_S / math.exp(log_tau))
        return float(np.mean((retention * current +
                              (1.0 - retention) * equilibrium - following) ** 2))

    pooled_result = minimize_scalar(pooled_objective,
        bounds=(math.log(0.010), math.log(0.75)), method="bounded",
        options={"xatol": 1.0e-8})
    pooled_tau = math.exp(float(pooled_result.x))
    return taus, diagnostics, {"tau_s": pooled_tau,
        "samples": int(len(current)),
        "one_step_rmse_radps": math.sqrt(float(pooled_result.fun))}


def _boundary_training_table() -> dict[str, Any]:
    """Read only the already-frozen train-only low-speed response table."""
    path = ROOT / BOUNDARY_TRAIN_SURFACE
    lines = path.read_text(encoding="utf-8").splitlines()
    source_note = " ".join(line for line in lines if line.startswith("#"))
    if "repetitions 1-2 only" not in source_note or "4.5 m/s is held out" not in source_note:
        raise ValueError("low-speed response table is missing its train-only provenance")
    rows: dict[tuple[float, int, float], float] = {}
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(
            line for line in stream if line and not line.lstrip().startswith("#"))
        required = {"speed_knot_mps", "turn_sign", "steering_rad",
                    "yaw_rate_abs_rps"}
        if set(reader.fieldnames or ()) != required | {"demand_q"}:
            raise ValueError("unexpected low-speed response table schema")
        for row in reader:
            key = (float(row["speed_knot_mps"]), int(row["turn_sign"]),
                   float(row["steering_rad"]))
            value = float(row["yaw_rate_abs_rps"])
            if key in rows or not math.isfinite(value) or value <= 0.0:
                raise ValueError("duplicate or invalid low-speed training response")
            rows[key] = value
    angles = (0.15, 0.20, 0.21, 0.22, 0.23, 0.25, 0.30, 0.35, 0.42)
    speeds = sorted({key[0] for key in rows})
    if len(rows) != len(speeds) * 2 * 10:
        raise ValueError("low-speed training response table is incomplete")
    boundary_rows = []
    for speed in speeds:
        for angle in angles:
            for sign in (-1, 1):
                boundary_rows.append({
                    "speed_knot_mps": speed,
                    "steering_rad": sign * angle,
                    "turn_sign": sign,
                    "yaw_rate_abs_rps": rows[(speed, sign, angle)],
                })
    return {
        "source": BOUNDARY_TRAIN_SURFACE,
        "training_only_provenance": source_note,
        "speed_knots_mps": speeds,
        "response_by_angle_and_direction": boundary_rows,
        "boundary_observation": (
            "Training responses vary below 0.30 rad and the response knee shifts "
            "with speed; a universal 0.30-rad step is not supported."),
    }


def _validation_boundary_cells(validation: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for data in validation:
        values = _series(data)
        for angle in (0.15, 0.20, 0.21, 0.22, 0.23, 0.25, 0.30, 0.35, 0.42):
            for sign in (-1, 1):
                mask = (
                    (values["speed"] >= 3.35) & (values["speed"] <= 3.65)
                    & (np.abs(values["steering"] - sign * angle) <= 0.004)
                    & (np.abs(values["steering_rate"]) <= 0.10)
                )
                selected = sign * values["yaw"][mask]
                rows.append({
                    "run_id": data["source_name"],
                    "speed_band_mps": [3.35, 3.65],
                    "steering_rad": sign * angle,
                    "samples": int(len(selected)),
                    "signed_yaw_median_rps": (
                        float(np.median(selected)) if len(selected) else None),
                })
    return rows


def _smoothstep(value: np.ndarray, low: float, high: float) -> np.ndarray:
    t = np.clip((np.asarray(value, dtype=np.float64) - low) / (high - low),
                0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _runtime_surface_weight(speed: np.ndarray, steering: np.ndarray,
                            steering_rate: np.ndarray) -> np.ndarray:
    q = np.maximum(speed, 0.0) * np.abs(np.tan(steering))
    q_weight = _smoothstep(q, 0.30, 0.70)
    magnitude = np.abs(steering)
    steering_weight = (
        _smoothstep(magnitude, STEERING_BLEND_START_RAD,
                    STEERING_BLEND_FULL_START_RAD)
        * (1.0 - _smoothstep(magnitude, STEERING_BLEND_FULL_END_RAD,
                             STEERING_BLEND_END_RAD)))
    speed_weight = (
        _smoothstep(speed, 2.40, 2.50)
        * (1.0 - _smoothstep(speed, 3.50, 3.60)))
    hold_weight = 1.0 - _smoothstep(
        np.abs(steering_rate), PHASE_RATE_THRESHOLD_RADPS,
        HOLD_RATE_FADEOUT_RADPS)
    return q_weight * steering_weight * speed_weight * hold_weight


def _predict_candidate(current_yaw: np.ndarray,
                       legacy_equilibrium: np.ndarray,
                       candidate_equilibrium: np.ndarray,
                       speed: np.ndarray, steering: np.ndarray,
                       steering_rate: np.ndarray,
                       dt: np.ndarray, hold_tau_s: float) -> tuple[
                           np.ndarray, np.ndarray]:
    weight = _runtime_surface_weight(speed, steering, steering_rate)
    equilibrium = legacy_equilibrium + weight * (
        candidate_equilibrium - legacy_equilibrium)
    tau = LEGACY_TAU_S + weight * (hold_tau_s - LEGACY_TAU_S)
    retention = np.exp(-dt / tau)
    return (retention * current_yaw + (1.0 - retention) * equilibrium,
            equilibrium)


def _metrics(prediction: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    error = prediction - truth
    return {
        "samples": int(len(error)),
        "rmse_radps": float(np.sqrt(np.mean(error * error))),
        "mae_radps": float(np.mean(np.abs(error))),
        "bias_radps": float(np.mean(error)),
        "wrong_direction_count": int(np.sum((prediction * truth) < 0.0)),
    }


def _write_surface(path: Path, curves: dict[tuple[float, int], np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        stream.write("# Training-only Y1 response map; values are rad/s.\n")
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("speed_knot_mps", "turn_sign", "steering_rad",
                         "demand_q", "yaw_rate_abs_rps"))
        for speed in SPEED_KNOTS:
            for sign in (-1, 1):
                fitted = curves[(speed, sign)]
                interp = PchipInterpolator(MEASURED_STEERING_KNOTS, fitted,
                                           extrapolate=False)
                for angle in SURFACE_STEERING_KNOTS:
                    if MEASURED_STEERING_KNOTS[0] <= angle <= MEASURED_STEERING_KNOTS[-1]:
                        rate = float(interp(angle))
                    else:
                        rate = abs(float(_legacy_equilibrium(speed, sign * angle)))
                    writer.writerow((f"{speed:.6f}", sign, f"{angle:.6f}",
                                     f"{speed * math.tan(angle):.8f}",
                                     f"{max(rate, 1.0e-5):.8f}"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT /
        "live_runs/racing_model_diagnostics_20261007/y1_authority_loss_fit")
    args = parser.parse_args()

    training = [_load_source(*source) for source in TRAIN_SOURCES]
    curves, cell_report = _estimate_equilibrium(training)
    taus, tau_report, pooled_tau = _phase_taus(training, curves)
    # Only after the fitted values above are fixed are whole-run validation
    # captures loaded. They are never used to alter the map or phase constants.
    validation = [_load_source(*source) for source in VALIDATION_SOURCES]
    boundary_train = _boundary_training_table()
    boundary_validation = _validation_boundary_cells(validation)

    run_reports: list[dict[str, Any]] = []
    grouped: dict[tuple[str, int], dict[str, list[np.ndarray]]] = {}
    for data in validation:
        values = _series(data)
        labels = np.full(len(values["steering"]), "hold", dtype=object)
        for begin_raw, end_raw in data["sequence_bounds"]:
            begin, end = int(begin_raw), int(end_raw)
            labels[begin:end] = _phase(values["steering"][begin:end],
                                       values["steering_rate"][begin:end])
        supported = (
            (values["speed"] >= SUPPORT_SPEED_MPS[0])
            & (values["speed"] <= SUPPORT_SPEED_MPS[1])
            & (np.abs(values["steering"]) >= SUPPORT_STEERING_RAD[0])
            & (np.abs(values["steering"]) <= SUPPORT_STEERING_RAD[1])
            & np.isfinite(values["steering_rate"])
        )
        adjacent = np.zeros(len(values["speed"]), dtype=bool)
        for begin_raw, end_raw in data["sequence_bounds"]:
            begin, end = int(begin_raw), int(end_raw)
            adjacent[begin:end - 1] = True
        indices = np.flatnonzero(supported[:-1] & supported[1:] & adjacent[:-1])
        if not len(indices):
            raise ValueError(f"{data['source_name']}: no held-out Y1 support")
        speed = values["speed"][indices]
        steering = values["steering"][indices]
        yaw = values["yaw"][indices]
        truth_next = values["yaw"][indices + 1]
        dt = np.full(len(indices), DT_S, dtype=np.float64)
        phase = labels[indices]
        baseline_eq = _legacy_equilibrium(speed, steering)
        baseline = yaw + (1.0 - math.exp(-DT_S / LEGACY_TAU_S)) * (baseline_eq - yaw)
        candidate_eq = _equilibrium(speed, steering, curves)
        hold_tau = float(taus["hold"])
        candidate, gated_equilibrium = _predict_candidate(
            yaw, baseline_eq, candidate_eq, speed, steering,
            values["steering_rate"][indices], dt, hold_tau)
        map_only = yaw + (1.0 - math.exp(-DT_S / LEGACY_TAU_S)) * (
            gated_equilibrium - yaw)

        report = {
            "run_id": data["source_name"],
            "split": data["split"],
            "all_run_samples": int(len(values["speed"])),
            "supported_transition_samples": int(len(indices)),
            "speed_range_mps": [float(np.min(speed)), float(np.max(speed))],
            "abs_steering_range_rad": [
                float(np.min(np.abs(steering))),
                float(np.max(np.abs(steering)))],
            "baseline": _metrics(baseline, truth_next),
            "candidate_equilibrium_with_legacy_tau": _metrics(map_only, truth_next),
            "candidate": _metrics(candidate, truth_next),
            "candidate_error_reduction_fraction": float(
                1.0 - np.sqrt(np.mean((candidate - truth_next) ** 2)) /
                np.sqrt(np.mean((baseline - truth_next) ** 2))),
            "by_turn_sign": {},
            "by_phase": {},
        }
        for sign in (-1, 1):
            mask = np.sign(steering) == sign
            if np.any(mask):
                report["by_turn_sign"][str(sign)] = {
                    "baseline": _metrics(baseline[mask], truth_next[mask]),
                    "candidate": _metrics(candidate[mask], truth_next[mask]),
                }
        for phase_name in ("turn_in", "hold", "unwind"):
            mask = phase == phase_name
            if np.any(mask):
                report["by_phase"][phase_name] = {
                    "baseline": _metrics(baseline[mask], truth_next[mask]),
                    "candidate_equilibrium_with_legacy_tau": _metrics(
                        map_only[mask], truth_next[mask]),
                    "candidate_phase_tau": _metrics(candidate[mask], truth_next[mask]),
                }
        run_reports.append(report)
        for sign in (-1, 1):
            mask = np.sign(steering) == sign
            if np.any(mask):
                grouped[(data["source_name"], sign)] = {
                    "truth": [truth_next[mask]],
                    "baseline": [baseline[mask]],
                    "candidate": [candidate[mask]],
                }

    # A 25 ms model score is the directly identified quantity. Multi-step
    # recursive accuracy is intentionally not inferred from teacher-forced
    # future feedback; it is reserved for production-model replay and track A/B.
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    surface_path = output / "yaw_response_surface_y1.csv"
    _write_surface(surface_path, curves)
    report = {
        "study": "Y1 low-speed high-steering front-authority response",
        "train_sources": [
            {"run_id": data["source_name"], "path": data["path"],
             "split": data["split"]} for data in training],
        "validation_sources": [
            {"run_id": data["source_name"], "path": data["path"],
             "split": data["split"]} for data in validation],
        "forbidden_data_policy": "No test/final-test archive was opened.",
        "model": {
            "equilibrium": "separate signed PCHIP steering curves at 2.5/3.0/3.5 m/s; linear interpolation only between these speed anchors",
            "phase": "empirical equilibrium map and fitted hold tau are gated by current physical steering-rate magnitude",
            "transition": "r_next = r_eq + (r - r_eq) * exp(-dt/tau); map and hold-tau deltas share the measured support and hold-rate gate",
            "runtime_hold_tau_s": hold_tau,
            "runtime_turn_in_unwind_tau": "legacy 0.015 s; sparse fitted transient constants are diagnostic only",
            "runtime_rate_gate_radps": {
                "full_through": PHASE_RATE_THRESHOLD_RADPS,
                "zero_at": HOLD_RATE_FADEOUT_RADPS,
            },
            "runtime_steering_gate_rad": {
                "start": STEERING_BLEND_START_RAD,
                "full_start": STEERING_BLEND_FULL_START_RAD,
                "full_end": STEERING_BLEND_FULL_END_RAD,
                "end": STEERING_BLEND_END_RAD,
            },
            "runtime_full_weight_support": {
                "speed_mps": list(SUPPORT_SPEED_MPS),
                "abs_steering_rad": list(SUPPORT_STEERING_RAD),
            },
            "runtime_taper_margins": {
                "speed_mps": [2.40, 2.50, 3.50, 3.60],
                "abs_steering_rad": [0.29, 0.30, 0.42, 0.45],
            },
            "outside_support": "production legacy transition; no specialist response below the 0.29-rad steering gate or outside the speed/steering/rate gates",
        },
        "phase_definition": {
            "input": "current and immediately previous physical steering feedback",
            "threshold_radps": PHASE_RATE_THRESHOLD_RADPS,
            "turn_in": "signed steering magnitude increasing",
            "unwind": "signed steering magnitude decreasing",
            "hold": "absolute magnitude rate within threshold",
        },
        "phase_fit_training_only": tau_report,
        "phase_constants_runtime_policy": (
            "only the hold tau is used; turn-in and unwind tau fits have sparse training support and remain diagnostic"),
        "pooled_tau_training_only": pooled_tau,
        "equilibrium_cells_training_only": cell_report,
        "lower_steering_boundary": {
            "training_only_response_table": boundary_train,
            "independent_validation_cells_near_3p5mps": boundary_validation,
            "interpretation": (
                "The below-0.30-rad response is not a constant plateau. "
                "The training-only table has a speed-dependent, non-monotone "
                "response knee; validation support is dense at 0.15/0.30/0.42 "
                "but sparse near 0.20-0.25 and asymmetric by direction. "
                "The model remains exactly legacy through 0.29 rad, uses only "
                "a narrow 0.29-0.30 continuity taper, and is full-weight only "
                "over the directly validated 0.30-0.42-rad interval. The taper "
                "is not claimed as learned response below 0.30 rad."),
        },
        "validation_by_whole_run": run_reports,
        "validation_totals_not_weighted_as_independent_runs": {
            "captures": len(run_reports),
            "mean_baseline_rmse_radps": float(np.mean([
                row["baseline"]["rmse_radps"] for row in run_reports])),
            "mean_candidate_rmse_radps": float(np.mean([
                row["candidate"]["rmse_radps"] for row in run_reports])),
        },
        "limitations": [
            "One-step held-out yaw prediction is not recursive full-lap accuracy.",
            "Validation support is concentrated near the upper Y1 speed anchor; no capability is claimed outside the reported sample range.",
            "The generated CSV uses legacy values at unsupported steering knots; only a separate runtime support gate may activate the fitted region.",
            "Track transfer and MPC convergence are not established by this offline score.",
        ],
        "generated_surface_csv": str(surface_path.relative_to(ROOT)),
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Y1 low-speed yaw-authority fit — 2026-10-07",
        "",
        "The equilibrium map and phase constants were fit only from the three listed train captures. "
        "The independent whole-run validation captures were opened afterward. No test/final-test data were opened.",
        "",
        "| Validation run | Samples | Baseline yaw RMSE | Candidate yaw RMSE | Reduction | Wrong signs |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in run_reports:
        lines.append(
            f"| {row['run_id']} | {row['supported_transition_samples']} | "
            f"{row['baseline']['rmse_radps']:.4f} | "
            f"{row['candidate']['rmse_radps']:.4f} | "
            f"{100.0 * row['candidate_error_reduction_fraction']:.1f}% | "
            f"{row['candidate']['wrong_direction_count']} |")
    lines.extend(("", "## Training-fitted phase time constants", ""))
    for name, row in tau_report.items():
        bound = f" (hit {row['fit_bound']} fit bound)" if row["hit_fit_bound"] else ""
        lines.append(f"- `{name}`: {row['tau_s']:.4f} s, "
                     f"{row['one_step_rmse_radps']:.4f} rad/s training RMSE, "
                     f"n={row['samples']}{bound}.")
    lines.append(
        f"- Runtime uses the hold tau ({taus['hold']:.4f} s) only. "
        f"The map and hold-tau correction are both full through "
        f"{PHASE_RATE_THRESHOLD_RADPS:.2f} rad/s steering rate and fade to "
        f"legacy by {HOLD_RATE_FADEOUT_RADPS:.2f} rad/s; the sparse turn-in/"
        "unwind fits are not activated.")
    lines.extend(("", "The reported scores are 25 ms one-step predictions. They do not establish recursive lap accuracy, "
                  "MPC convergence, or track transfer. See `report.json` for run-level and direction-level details, "
                  "source provenance, fitted cell support, and limitations.", ""))
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "output_dir": str(output),
        "surface": str(surface_path),
        "tau_by_phase": tau_report,
        "validation": [{
            "run": row["run_id"],
            "n": row["supported_transition_samples"],
            "baseline_rmse": row["baseline"]["rmse_radps"],
            "candidate_rmse": row["candidate"]["rmse_radps"],
            "wrong_direction_count": row["candidate"]["wrong_direction_count"],
        } for row in run_reports],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
