#!/usr/bin/env python3
"""Fit and score a speed/steering/phase-specific yaw-response atlas.

This is an offline identification utility. Simulator truth is used only as the
label. The transient model is conditioned on the measured steering feedback
sequence so it isolates the yaw response from the separate steering actuator
model; that feedback sequence is an oracle input in the rollout scores and is
not evidence of end-to-end MPC accuracy.

Only explicit train captures are fitted. Validation captures are read as
whole-run holdouts. Mixed-split archives and test/final-test data are excluded.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DT_S = 0.025
SPEED_KNOTS_MPS = (4.5, 6.5, 7.5)
STEERING_KNOTS_RAD = (0.0, 0.15, 0.30, 0.42, 0.50, 0.524)
SPEED_SUPPORT_TOLERANCE_MPS = 0.28
STEERING_SUPPORT_TOLERANCE_RAD = 0.085
MIN_REGIME_TRANSITIONS = 10

DYNAMIC_TRAIN = (
    "dynamic_steering_train",
    "live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/"
    "dynamic_steering_r02_continuous_train_20261002/openplane_dynamics.npz",
    "train",
)
DYNAMIC_VALIDATION = (
    "dynamic_steering_validation",
    "live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/"
    "dynamic_steering_validation_r01_continuous_20261002/openplane_dynamics.npz",
    "validation",
)
HIGHSTEER_TRAIN = tuple(
    (f"highsteer_train_r0{run}",
     f"live_runs/openplane_subnet_highsteer_transients_train_r0{run}_20261004/"
     "source_continuous_v1/openplane_dynamics.npz", "train")
    for run in (2, 3)
)
HIGHSTEER_VALIDATION = tuple(
    (f"highsteer_validation_r0{run}",
     f"live_runs/openplane_subnet_highsteer_transients_validation_r0{run}_20261004/"
     "source_continuous_v1/openplane_dynamics.npz", "validation")
    for run in (1, 2, 3)
)

# These two parents were aborted only by their emergency speed cutoff. The
# loader below returns only individually valid, reset-isolated phases; no
# invalid/unscored phase or parent-run continuation is used for fitting.
PHASE_TRAIN_BAGS = (
    ("isolated_surface_train_r01",
     "live_runs/openplane_isolated_highspeed_surface_20260927/run/run_0.db3"),
    ("crossfactor_train_r02",
     "live_runs/openplane_highspeed_crossfactor_20260927_02/run/run_0.db3"),
)


@dataclass(frozen=True)
class Series:
    name: str
    split: str
    path: str
    speed: np.ndarray
    steering: np.ndarray
    yaw: np.ndarray
    bounds: np.ndarray
    steering_command: np.ndarray | None = None


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def _load_npz(name: str, relative_path: str, expected_split: str) -> Series:
    path = ROOT / relative_path
    manifest_path = path.with_name("manifest.json")
    if not path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"{name}: missing capture or manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    runs = manifest.get("runs", [])
    if len(runs) != 1:
        raise ValueError(f"{name}: expected one whole-run manifest record")
    row = runs[0]
    if (row.get("effective_split") != expected_split or row.get("aborted")
            or row.get("reason") != "schedule complete"
            or not row.get("clean_stream_and_collision_gate")
            or any(int(value) != 0 for value in row.get("collisions", []))
            or int(row.get("timing_faults", -1)) != 0
            or row.get("quality_failures")
            or row.get("whole_bag_quality_failures")):
        raise ValueError(f"{name}: failed frozen {expected_split} quality gate")
    with np.load(path, allow_pickle=False) as archive:
        required = ("frames", "simulator_rigid_state", "sequence_bounds",
                    "run_ids", "run_splits", "dt_s", "packet_sequence")
        missing = [key for key in required if key not in archive.files]
        if missing:
            raise ValueError(f"{name}: missing arrays {missing}")
        if (len(archive["run_ids"]) != 1
                or str(archive["run_splits"][0]) != expected_split
                or not np.all(np.asarray(archive["dt_s"]) == np.float32(DT_S))):
            raise ValueError(f"{name}: mixed split or non-25ms archive")
        frames = np.asarray(archive["frames"], dtype=np.float64)
        rigid = np.asarray(archive["simulator_rigid_state"], dtype=np.float64)
        bounds = np.asarray(archive["sequence_bounds"], dtype=np.int64)
        packets = np.asarray(archive["packet_sequence"], dtype=np.int64)
    if (frames.ndim != 2 or frames.shape[1] != 9
            or rigid.shape != (len(frames), 13)
            or not np.isfinite(frames).all()
            or not np.isfinite(rigid).all()):
        raise ValueError(f"{name}: invalid motion-state arrays")
    for begin, end in bounds:
        if (begin < 0 or end > len(frames) or end - begin < 3
                or np.any(np.diff(packets[begin:end]) != 1)):
            raise ValueError(f"{name}: invalid bounds or packet discontinuity")
    return Series(
        name=name, split=expected_split, path=relative_path,
        speed=np.hypot(rigid[:, 7], rigid[:, 8]),
        steering=frames[:, 3], yaw=rigid[:, 12], bounds=bounds,
        steering_command=frames[:, 7],
    )


def _load_valid_phase_train_bags() -> tuple[list[Series], list[dict[str, Any]]]:
    # Import the repository's phase-validating bag reader; it returns only
    # valid marked phases by default and preserves reset boundaries.
    from tools.evaluate_open_plane_body_dynamics import load_capture

    sequences: list[Series] = []
    audit: list[dict[str, Any]] = []
    for name, relative_path in PHASE_TRAIN_BAGS:
        capture = load_capture(ROOT / relative_path)
        if (capture.invalid_phase_count != 0 or capture.timing_faults != 0
                or capture.collision_count_start != capture.collision_count_end
                or capture.collision_count_start != 0
                or capture.valid_phase_count != len(capture.sequences)
                or not capture.sequences):
            raise ValueError(f"{name}: phase-level training quality gate failed")
        audit.append({
            "name": name,
            "path": relative_path,
            "parent_run_aborted": capture.aborted,
            "parent_run_reason": capture.reason,
            "valid_phases_used": capture.valid_phase_count,
            "invalid_phases_excluded": capture.invalid_phase_count,
            "unscored_phases_excluded": capture.unscored_phase_count,
            "timing_faults": capture.timing_faults,
            "collisions_start_end": [capture.collision_count_start,
                                      capture.collision_count_end],
            "use_policy": "valid reset-isolated phases only; aborted parent is not a whole-run validation capture",
        })
        for index, samples in enumerate(capture.sequences):
            if len(samples) < 24:
                continue
            rigid = [sample.simulator_rigid_state for sample in samples]
            if any(state is None or len(state) != 13 for state in rigid):
                continue
            # Both speed and yaw labels must come from simulator rigid-body
            # truth. The observer/odom state is not a fitting target or gate.
            speed = np.asarray([math.hypot(state[7], state[8])
                                for state in rigid], dtype=np.float64)
            steering = np.asarray([sample.actuators[0] for sample in samples],
                                  dtype=np.float64)
            yaw = np.asarray([state[12] for state in rigid], dtype=np.float64)
            steering_command = np.asarray([sample.actuators[3]
                                           for sample in samples],
                                          dtype=np.float64)
            sequences.append(Series(
                name=f"{name}_phase_{index:03d}", split="train",
                path=relative_path, speed=speed, steering=steering, yaw=yaw,
                bounds=np.asarray([[0, len(samples)]], dtype=np.int64),
                steering_command=steering_command,
            ))
    return sequences, audit


def _training_manifest() -> tuple[list[Series], list[Series], list[Series],
                                  list[dict[str, Any]]]:
    dynamic_train = _load_npz(*DYNAMIC_TRAIN)
    dynamic_validation = _load_npz(*DYNAMIC_VALIDATION)
    highsteer_train = [_load_npz(*source) for source in HIGHSTEER_TRAIN]
    highsteer_validation = [_load_npz(*source)
                            for source in HIGHSTEER_VALIDATION]
    phase_sequences, phase_audit = _load_valid_phase_train_bags()
    training = [dynamic_train, *highsteer_train]
    validation = [dynamic_validation, *highsteer_validation]
    return training, validation, phase_sequences, phase_audit


def _cell(value: float) -> float:
    # Steering test points are multiples of 0.005 rad; this keeps adjacent
    # measured conditions distinct (e.g. 0.20, 0.21, 0.22, 0.23).
    return round(float(value) * 200.0) / 200.0


def _collect_equilibrium(training: list[Series],
                         phase_sequences: list[Series]
                         ) -> tuple[dict[tuple[float, float], float],
                                    list[dict[str, Any]]]:
    by_source_cell: dict[tuple[str, float, float], list[float]] = defaultdict(list)
    source_count: dict[str, int] = defaultdict(int)

    def add_samples(series: Series, sample_indices: np.ndarray,
                    speed_anchor: float) -> None:
        for index in sample_indices:
            steering = float(series.steering[index])
            if abs(steering) < 0.02:
                signed_cell = 0.0
            else:
                signed_cell = math.copysign(_cell(abs(steering)), steering)
            by_source_cell[(series.name, speed_anchor, signed_cell)].append(
                float(series.yaw[index]))

    for series in training:
        derivative = np.zeros_like(series.steering)
        for begin, end in series.bounds:
            derivative[begin + 1:end] = np.diff(series.steering[begin:end]) / DT_S
        for speed_anchor in SPEED_KNOTS_MPS:
            speed_mask = np.abs(series.speed - speed_anchor) <= 0.18
            for signed_target in np.unique(np.asarray([
                    math.copysign(_cell(abs(value)), value)
                    if abs(value) >= 0.02 else 0.0
                    for value in series.steering])):
                mask = (speed_mask
                        & (np.abs(series.steering - signed_target) <= 0.0035)
                        & (np.abs(derivative) <= 0.10))
                indices = np.flatnonzero(mask)
                if len(indices) >= 6:
                    add_samples(series, indices, speed_anchor)
                    source_count[series.name] += 1

    # The aborted parent bags contribute only validated, reset-isolated hold
    # phases. Use a trailing stable window and exclude phases with a steering
    # movement or changing speed; this does not turn parent runs into clean runs.
    phase_window = 20
    for series in phase_sequences:
        begin, end = map(int, series.bounds[0])
        lo = max(begin, end - phase_window)
        delta = series.steering[lo:end]
        speed = series.speed[lo:end]
        if (len(delta) < 12 or np.ptp(delta) > 0.008
                or np.std(speed) > 0.10):
            continue
        measured_speed = float(np.median(speed))
        speed_anchor = min(SPEED_KNOTS_MPS,
                           key=lambda value: abs(value - measured_speed))
        if abs(measured_speed - speed_anchor) > 0.18:
            continue
        steering = float(np.median(delta))
        signed_cell = (0.0 if abs(steering) < 0.02 else
                       math.copysign(_cell(abs(steering)), steering))
        key = (series.name.rsplit("_phase_", 1)[0], speed_anchor, signed_cell)
        by_source_cell[key].extend(series.yaw[lo:end].tolist())
        source_count[key[0]] += 1

    # Equal source weight: repeated samples in one long capture cannot swamp
    # independent reset phases or another clean training capture.
    source_medians = {key: float(np.median(values))
                      for key, values in by_source_cell.items() if values}
    grouped: dict[tuple[float, float], list[tuple[str, float]]] = defaultdict(list)
    for (source, speed, steering), value in source_medians.items():
        grouped[(speed, steering)].append((source, value))
    table = {key: float(np.median([value for _, value in rows]))
             for key, rows in grouped.items()}
    report = []
    for (speed, steering), rows in sorted(grouped.items()):
        responses = np.asarray([value for _, value in rows])
        report.append({
            "speed_anchor_mps": speed,
            "steering_feedback_rad": steering,
            "yaw_rate_rps": table[(speed, steering)],
            "source_count": len(rows),
            "source_names": [name for name, _ in rows],
            "source_medians_rps": responses.tolist(),
            "source_spread_rps": float(np.ptp(responses)) if len(responses) > 1 else 0.0,
        })
    return table, report


def _equilibrium(speed: np.ndarray | float, steering: np.ndarray | float,
                 table: dict[tuple[float, float], float]) -> np.ndarray:
    speeds = np.atleast_1d(np.asarray(speed, dtype=np.float64))
    deltas = np.atleast_1d(np.asarray(steering, dtype=np.float64))
    if speeds.shape != deltas.shape:
        raise ValueError("speed and steering shape mismatch")
    out = np.full(speeds.shape, np.nan, dtype=np.float64)
    knots = np.asarray(sorted(set(k[0] for k in table)), dtype=np.float64)
    if len(knots) < 2:
        return out
    # The source holds are centered on nominal speed targets and the actual
    # clean samples vary by up to about 0.1 m/s. Treat that measured endpoint
    # band as the anchor itself; do not extrapolate a speed slope beyond it.
    endpoint_band_mps = 0.18
    for index, (local_speed, local_delta) in enumerate(zip(speeds, deltas)):
        if (local_speed < knots[0] - endpoint_band_mps
                or local_speed > knots[-1] + endpoint_band_mps):
            continue
        supported_speed = float(np.clip(local_speed, knots[0], knots[-1]))
        sign = -1.0 if local_delta < 0.0 else 1.0
        angle = float(local_delta)
        per_speed: dict[float, float] = {}
        for speed_knot in knots:
            rows = sorted((delta, value) for (cell_speed, delta), value in table.items()
                          if cell_speed == speed_knot and delta * sign >= -1.0e-9)
            if len(rows) < 2:
                continue
            x = np.asarray([delta for delta, _ in rows], dtype=np.float64)
            y = np.asarray([value for _, value in rows], dtype=np.float64)
            if sign < 0:
                # Reflect only the steering coordinate for interpolation.
                # The measured yaw values are already signed; reflecting them
                # again makes every negative-steering equilibrium positive.
                x = -x
                order = np.argsort(x)
                x, y = x[order], y[order]
            elif sign > 0:
                order = np.argsort(x)
                x, y = x[order], y[order]
            query = abs(angle)
            if query < x[0] or query > x[-1]:
                continue
            per_speed[float(speed_knot)] = float(PchipInterpolator(
                x, y, extrapolate=False)(query))
        if len(per_speed) < 2:
            continue
        local_knots = np.asarray(sorted(per_speed), dtype=np.float64)
        if (supported_speed < local_knots[0]
                or supported_speed > local_knots[-1]):
            continue
        out[index] = float(np.interp(
            supported_speed, local_knots,
            np.asarray([per_speed[value] for value in local_knots])))
    return out


def _features(series: Series,
              table: dict[tuple[float, float], float]
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    q = _equilibrium(series.speed, series.steering, table)
    speed_index = np.argmin(
        np.abs(series.speed[:, None] - np.asarray(SPEED_KNOTS_MPS)[None, :]),
        axis=1)
    steer_index = np.argmin(
        np.abs(np.abs(series.steering[:, None]) -
               np.asarray(STEERING_KNOTS_RAD)[None, :]), axis=1)
    magnitude_rate = np.zeros_like(series.steering)
    for begin, end in series.bounds:
        magnitude_rate[begin + 1:end] = np.diff(
            np.abs(series.steering[begin:end])) / DT_S
    phase = np.where(magnitude_rate > 0.10, 1,
                     np.where(magnitude_rate < -0.10, -1, 0))
    return q, speed_index, steer_index, phase, magnitude_rate


def _causal_actuator_proxy(series: Series) -> Series:
    """Build one-step steering inputs from current feedback and past commands.

    Captured command k-1 is the input available to predict feedback k+1.
    The 3.2 rad/s physical rate limit is the production model's configured
    limit. Speed at the target row is shifted from the current observation so
    the yaw fit does not consume future ground-truth speed either.
    """
    if series.steering_command is None:
        raise ValueError(f"{series.name}: missing steering command history")
    speed = np.full_like(series.speed, np.nan)
    steering = np.full_like(series.steering, np.nan)
    max_step = 3.2 * DT_S
    for begin_raw, end_raw in series.bounds:
        begin, end = int(begin_raw), int(end_raw)
        # target row i is predicted from the measured state at i-1 and the
        # command at i-2 (one command sample of actuator/bridge delay).
        for target in range(begin + 2, end):
            current = target - 1
            delayed_command = float(series.steering_command[target - 2])
            speed[target] = series.speed[current]
            steering[target] = series.steering[current] + float(np.clip(
                delayed_command - series.steering[current],
                -max_step, max_step))
    return Series(
        name=series.name + "_causal_inputs", split=series.split,
        path=series.path, speed=speed, steering=steering, yaw=series.yaw,
        bounds=series.bounds, steering_command=series.steering_command,
    )


def _score_steering_actuator(series: Series) -> dict[str, Any]:
    """Score the command-to-wheel-angle step using whole capture sequences."""
    if series.steering_command is None:
        return {"samples": 0, "reason": "steering command unavailable"}
    errors_all: list[float] = []
    errors_after_warmup: list[float] = []
    def angle_metrics(errors: list[float]) -> dict[str, Any]:
        values = np.asarray(errors, dtype=np.float64)
        if not len(values):
            return {"samples": 0}
        absolute = np.abs(values)
        return {
            "samples": int(len(values)),
            "rmse_rad": float(np.sqrt(np.mean(values * values))),
            "mae_rad": float(np.mean(absolute)),
            "p95_abs_rad": float(np.quantile(absolute, 0.95)),
            "p99_abs_rad": float(np.quantile(absolute, 0.99)),
            "max_abs_rad": float(np.max(absolute)),
            "fraction_abs_error_below_0p005_rad": float(
                np.mean(absolute < 0.005)),
        }

    by_sequence = []
    max_step = 3.2 * DT_S
    for sequence_index, (begin_raw, end_raw) in enumerate(series.bounds):
        begin, end = int(begin_raw), int(end_raw)
        local: list[float] = []
        for target in range(begin + 2, end):
            current = target - 1
            delayed_command = float(series.steering_command[target - 2])
            prediction = series.steering[current] + float(np.clip(
                delayed_command - series.steering[current],
                -max_step, max_step))
            error = prediction - float(series.steering[target])
            errors_all.append(error)
            local.append(error)
            if target - begin >= 4:
                errors_after_warmup.append(error)
        if local:
            by_sequence.append({
                "sequence_index": sequence_index,
                "samples": len(local),
                **angle_metrics(local),
            })
    return {
        "model": "delta_next = delta_current + clip(command[k-1] - delta_current, +/- 3.2*0.025 rad)",
        "all_transitions": angle_metrics(errors_all),
        "after_first_three_reset_samples": angle_metrics(errors_after_warmup),
        "by_sequence": by_sequence,
    }


def _inside_support(speed: float, steering: float, speed_index: int,
                    steer_index: int) -> bool:
    return (abs(speed - SPEED_KNOTS_MPS[speed_index]) <= SPEED_SUPPORT_TOLERANCE_MPS
            and abs(abs(steering) - STEERING_KNOTS_RAD[steer_index])
            <= STEERING_SUPPORT_TOLERANCE_RAD)


def _fit_models(training: list[Series], table: dict[tuple[float, float], float]
                ) -> tuple[dict[tuple[int, int, int], np.ndarray],
                           dict[tuple[int, int, int], int]]:
    samples: dict[tuple[int, int, int], list[tuple[np.ndarray, float]]] = defaultdict(list)
    for series in training:
        q, speed_index, steer_index, phase, _ = _features(series, table)
        for begin, end in series.bounds:
            for index in range(max(1, int(begin) + 1), int(end) - 1):
                target = index + 1
                if not np.isfinite(q[index - 1:target + 1]).all():
                    continue
                if abs(series.speed[target] - series.speed[index]) > 0.35:
                    continue
                si, di = int(speed_index[target]), int(steer_index[target])
                if not _inside_support(series.speed[target], series.steering[target], si, di):
                    continue
                key = (si, di, int(phase[target]))
                # Local two-pole response, with the new equilibrium as the
                # forcing target. The first two coefficients define a stable
                # AR(2) pair; the latter two encode the input step history.
                x = np.asarray((
                    series.yaw[index] - q[target],
                    series.yaw[index - 1] - q[target],
                    q[index] - q[target],
                    q[index - 1] - q[target],
                ), dtype=np.float64)
                y = float(series.yaw[target] - q[target])
                if np.isfinite(x).all() and math.isfinite(y):
                    samples[key].append((x, y))

    models: dict[tuple[int, int, int], np.ndarray] = {}
    counts: dict[tuple[int, int, int], int] = {}
    for key, rows in samples.items():
        counts[key] = len(rows)
        if len(rows) < MIN_REGIME_TRANSITIONS:
            continue
        x = np.stack([row[0] for row in rows])
        y = np.asarray([row[1] for row in rows])
        coefficient = np.linalg.solve(
            x.T @ x + 1.0e-5 * np.eye(x.shape[1]), x.T @ y)
        stable = (abs(coefficient[1]) < 0.995
                  and coefficient[0] + coefficient[1] < 0.995
                  and -coefficient[0] + coefficient[1] < 0.995)
        if not stable:
            constraints = [
                {"type": "ineq", "fun": lambda c: 0.995 - c[1]},
                {"type": "ineq", "fun": lambda c: 0.995 + c[1]},
                {"type": "ineq", "fun": lambda c: 0.995 - c[0] - c[1]},
                {"type": "ineq", "fun": lambda c: 0.995 + c[0] - c[1]},
            ]
            objective = lambda c: float(np.sum((x @ c - y) ** 2)
                                        + 1.0e-5 * np.sum(c * c))
            result = minimize(objective, coefficient, method="SLSQP",
                              constraints=constraints,
                              options={"maxiter": 300, "ftol": 1.0e-11})
            if not result.success:
                continue
            coefficient = np.asarray(result.x, dtype=np.float64)
        models[key] = coefficient
    return models, counts


def _fit_first_order(training: list[Series],
                     table: dict[tuple[float, float], float]
                     ) -> tuple[dict[tuple[int, int, int], np.ndarray],
                                dict[tuple[int, int, int], int]]:
    samples: dict[tuple[int, int, int], list[tuple[float, float]]] = defaultdict(list)
    for series in training:
        q, speed_index, steer_index, phase, _ = _features(series, table)
        for begin, end in series.bounds:
            for index in range(max(1, int(begin) + 1), int(end) - 1):
                target = index + 1
                if not np.isfinite(q[index:target + 1]).all():
                    continue
                if abs(series.speed[target] - series.speed[index]) > 0.35:
                    continue
                si, di = int(speed_index[target]), int(steer_index[target])
                if not _inside_support(series.speed[target], series.steering[target], si, di):
                    continue
                key = (si, di, int(phase[target]))
                samples[key].append((
                    float(series.yaw[index] - q[target]),
                    float(series.yaw[target] - q[target]),
                ))

    models: dict[tuple[int, int, int], np.ndarray] = {}
    counts: dict[tuple[int, int, int], int] = {}
    for key, rows in samples.items():
        counts[key] = len(rows)
        if len(rows) < MIN_REGIME_TRANSITIONS:
            continue
        x = np.asarray([row[0] for row in rows], dtype=np.float64)
        y = np.asarray([row[1] for row in rows], dtype=np.float64)
        raw_retention = float(np.dot(x, y) / (np.dot(x, x) + 1.0e-10))
        # A first-order relaxation cannot have an unstable or alternating
        # pole. If data demand that behavior, the second-order candidate must
        # explain it instead of allowing a fragile one-pole fit.
        models[key] = np.asarray([np.clip(raw_retention, 0.0, 0.995)])
    return models, counts


def _fit_rollout_first_order(
        training: list[Series], table: dict[tuple[float, float], float],
        initial: dict[tuple[int, int, int], np.ndarray]
        ) -> tuple[dict[tuple[int, int, int], np.ndarray], dict[str, Any]]:
    """Retune local one-pole retentions against training-only free rollouts."""
    keys = sorted(initial)
    key_index = {key: index for index, key in enumerate(keys)}
    initial_values = np.asarray([initial[key][0] for key in keys], dtype=np.float64)
    prepared = []
    for series in training:
        q, speed_index, steer_index, phase, _ = _features(series, table)
        prepared.append((series, q, speed_index, steer_index, phase))

    horizons = (1, 4, 10, 20)
    starts = 8  # 0.20 s spacing prevents dense overlapping windows dominating.

    def rollout_losses(retentions: np.ndarray) -> dict[int, list[float]]:
        losses: dict[int, list[float]] = {horizon: [] for horizon in horizons}
        for series, q, speed_index, steer_index, phase in prepared:
            for begin_raw, end_raw in series.bounds:
                begin, end = int(begin_raw), int(end_raw)
                for start in range(max(begin + 1, 1), end - 1, starts):
                    predicted = series.yaw.copy()
                    for target in range(start + 1, min(end, start + max(horizons) + 1)):
                        current = target - 1
                        if not np.isfinite(q[current:target + 1]).all():
                            break
                        si, di = int(speed_index[target]), int(steer_index[target])
                        if not _inside_support(series.speed[target],
                                              series.steering[target], si, di):
                            break
                        key = (si, di, int(phase[target]))
                        parameter = key_index.get(key)
                        if parameter is None:
                            break
                        value = float(q[target] + retentions[parameter] *
                                      (predicted[current] - q[target]))
                        if not math.isfinite(value) or abs(value) > 20.0:
                            break
                        predicted[target] = value
                        horizon = target - start
                        if horizon in losses:
                            error = value - float(series.yaw[target])
                            losses[horizon].append(error * error)
        return losses

    def objective(retentions: np.ndarray) -> float:
        losses = rollout_losses(retentions)
        per_horizon = [float(np.mean(values)) for values in losses.values()
                       if values]
        return float(np.mean(per_horizon)) if per_horizon else math.inf

    before = objective(initial_values)
    result = minimize(objective, initial_values, method="L-BFGS-B",
                      bounds=[(0.0, 0.995)] * len(keys),
                      options={"maxiter": 250, "ftol": 1.0e-10,
                               "maxls": 30})
    if not result.success and not np.isfinite(result.fun):
        raise RuntimeError(f"multi-step local response fit failed: {result.message}")
    after = objective(np.asarray(result.x, dtype=np.float64))
    fitted = {key: np.asarray([float(result.x[index])])
              for index, key in enumerate(keys)}
    return fitted, {
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "training_rollout_objective_before": before,
        "training_rollout_objective_after": after,
        "retention_parameters": len(keys),
        "start_spacing_steps": starts,
        "optimized_horizons_steps": list(horizons),
    }


def _legacy_equilibrium(speed: float, steering: float) -> float:
    magnitude = abs(steering)
    reduction = 35.6 * min(max(magnitude - 0.41, 0.0), 0.05)
    return speed * math.tan(steering) * (2.95 - reduction)


def _sample_metrics(errors: list[float]) -> dict[str, Any]:
    values = np.asarray(errors, dtype=np.float64)
    if not len(values):
        return {"samples": 0}
    absolute = np.abs(values)
    return {
        "samples": int(len(values)),
        "rmse_radps": float(np.sqrt(np.mean(values * values))),
        "mae_radps": float(np.mean(absolute)),
        "p95_abs_radps": float(np.quantile(absolute, 0.95)),
        "p99_abs_radps": float(np.quantile(absolute, 0.99)),
        "max_abs_radps": float(np.max(absolute)),
        "fraction_abs_error_below_0p1": float(np.mean(absolute < 0.1)),
    }


def _score_run(series: Series, table: dict[tuple[float, float], float],
               models: dict[tuple[int, int, int], np.ndarray],
               include_recursive: bool = True,
               ) -> dict[str, Any]:
    q, speed_index, steer_index, phase, _ = _features(series, table)
    records: list[tuple[int, tuple[int, int, int], float, float]] = []
    legacy_errors: list[float] = []
    one_step_errors: list[float] = []
    cell_errors: dict[tuple[int, int, int], list[float]] = defaultdict(list)
    cell_legacy_errors: dict[tuple[int, int, int], list[float]] = defaultdict(list)
    eligible = 0
    unsupported = 0
    worst_samples: list[dict[str, Any]] = []
    for begin, end in series.bounds:
        for index in range(max(int(begin) + 1, 1), int(end) - 1):
            target = index + 1
            if not (np.isfinite(q[index - 1:target + 1]).all()
                    and np.isfinite(series.yaw[index - 1:target + 1]).all()):
                continue
            si, di = int(speed_index[target]), int(steer_index[target])
            if not _inside_support(series.speed[target], series.steering[target], si, di):
                continue
            eligible += 1
            key = (si, di, int(phase[target]))
            coefficient = models.get(key)
            if coefficient is None:
                unsupported += 1
                continue
            if len(coefficient) == 1:
                prediction = float(q[target] + coefficient[0] *
                                   (series.yaw[index] - q[target]))
            else:
                x = np.asarray((
                    series.yaw[index] - q[target],
                    series.yaw[index - 1] - q[target],
                    q[index] - q[target],
                    q[index - 1] - q[target],
                ))
                prediction = float(q[target] + coefficient @ x)
            error = prediction - float(series.yaw[target])
            legacy_q = _legacy_equilibrium(float(series.speed[target]),
                                           float(series.steering[target]))
            retention = math.exp(-DT_S / 0.015)
            legacy_prediction = (retention * float(series.yaw[index])
                                 + (1.0 - retention) * legacy_q)
            legacy_error = legacy_prediction - float(series.yaw[target])
            one_step_errors.append(error)
            legacy_errors.append(legacy_error)
            cell_errors[key].append(error)
            cell_legacy_errors[key].append(legacy_error)
            records.append((index, key, prediction, float(series.yaw[target])))
            worst_samples.append({
                "target_index": target,
                "speed_mps": float(series.speed[target]),
                "steering_feedback_rad": float(series.steering[target]),
                "steering_magnitude_rate_radps": float(
                    (abs(series.steering[target]) -
                     abs(series.steering[index])) / DT_S),
                "yaw_truth_radps": float(series.yaw[target]),
                "yaw_prediction_radps": prediction,
                "yaw_error_radps": error,
                "regime_key": [si, di, int(phase[target])],
            })

    # The legacy recursive score conditions on recorded future steering; it
    # isolates yaw dynamics but is not a causal steering/vehicle rollout.
    horizons: dict[int, list[float]] = defaultdict(list)
    for begin, end in series.bounds if include_recursive else ():
        begin, end = int(begin), int(end)
        for start in range(max(begin + 1, 1), end - 1):
            predicted_yaw = series.yaw.copy()
            for index in range(start, end - 1):
                target = index + 1
                if not np.isfinite(q[index - 1:target + 1]).all():
                    break
                si, di = int(speed_index[target]), int(steer_index[target])
                if not _inside_support(series.speed[target], series.steering[target], si, di):
                    break
                coefficient = models.get((si, di, int(phase[target])))
                if coefficient is None:
                    break
                if len(coefficient) == 1:
                    value = float(q[target] + coefficient[0] *
                                  (predicted_yaw[index] - q[target]))
                else:
                    x = np.asarray((
                        predicted_yaw[index] - q[target],
                        predicted_yaw[index - 1] - q[target],
                        q[index] - q[target],
                        q[index - 1] - q[target],
                    ))
                    value = float(q[target] + coefficient @ x)
                if not math.isfinite(value) or abs(value) > 20.0:
                    break
                horizons[target - start].append(value - float(series.yaw[target]))
                predicted_yaw[target] = value

    cells = []
    for key in sorted(cell_errors):
        cells.append({
            "speed_anchor_mps": SPEED_KNOTS_MPS[key[0]],
            "steering_anchor_rad": STEERING_KNOTS_RAD[key[1]],
            "transition_phase": {-1: "unwind", 0: "hold", 1: "turn_in"}[key[2]],
            "candidate": _sample_metrics(cell_errors[key]),
            "production_legacy": _sample_metrics(cell_legacy_errors[key]),
        })
    return {
        "name": series.name,
        "split": series.split,
        "source": series.path,
        "eligible_transitions_with_equilibrium_support": eligible,
        "eligible_but_no_transient_fit": unsupported,
        "transient_model_coverage_fraction": (
            len(one_step_errors) / eligible if eligible else 0.0),
        "production_legacy": _sample_metrics(legacy_errors),
        "regime_specific_second_order": _sample_metrics(one_step_errors),
        "by_regime": cells,
        "worst_one_step_samples": sorted(
            worst_samples, key=lambda row: abs(row["yaw_error_radps"]),
            reverse=True)[:20],
        "recursive_oracle_steering_rollout": {
            str(horizon): _sample_metrics(errors)
            for horizon, errors in sorted(horizons.items())
            if horizon in (1, 4, 10, 20, 40)
        },
    }


def _coefficient_rows(models: dict[tuple[int, int, int], np.ndarray],
                     counts: dict[tuple[int, int, int], int],
                     model_order: int,
                     ) -> list[dict[str, Any]]:
    rows = []
    for key, coefficient in sorted(models.items()):
        a1 = coefficient[0]
        a2 = coefficient[1] if model_order == 2 else 0.0
        discriminant = a1 * a1 + 4.0 * a2
        roots: list[complex]
        if discriminant >= 0.0:
            root = math.sqrt(discriminant)
            roots = [complex((a1 + root) / 2.0), complex((a1 - root) / 2.0)]
        else:
            root = math.sqrt(-discriminant) / 2.0
            roots = [complex(a1 / 2.0, root), complex(a1 / 2.0, -root)]
        rows.append({
            "model_order": model_order,
            "speed_anchor_mps": SPEED_KNOTS_MPS[key[0]],
            "steering_anchor_rad": STEERING_KNOTS_RAD[key[1]],
            "transition_phase": {-1: "unwind", 0: "hold", 1: "turn_in"}[key[2]],
            "training_transition_count": counts[key],
            "a1": float(a1), "a2": float(a2),
            "b0": float(coefficient[2]) if model_order == 2 else 0.0,
            "b1": float(coefficient[3]) if model_order == 2 else 0.0,
            "pole_1_real": float(roots[0].real),
            "pole_1_imag": float(roots[0].imag),
            "pole_2_real": float(roots[1].real),
            "pole_2_imag": float(roots[1].imag),
            "max_pole_magnitude": float(max(abs(root) for root in roots)),
        })
    return rows


def run(output: Path, causal_only: bool = False) -> dict[str, Any]:
    training, validation, phase_sequences, phase_audit = _training_manifest()
    table, equilibrium_rows = _collect_equilibrium(training, phase_sequences)
    if causal_only:
        causal_training = [
            _causal_actuator_proxy(series)
            for series in [*training, *phase_sequences]
        ]
        causal_models, causal_counts = _fit_first_order(causal_training, table)
        reports = []
        for series in validation:
            causal_series = _causal_actuator_proxy(series)
            score = _score_run(
                causal_series, table, causal_models, include_recursive=False)
            reports.append({
                "name": series.name,
                "split": series.split,
                "source": series.path,
                "input_policy": (
                    "current GT speed and current GT yaw as present-time state; "
                    "next steering predicted from current feedback, delayed "
                    "command, and the production 3.2 rad/s rate limit"
                ),
                "actuator_prediction": _score_steering_actuator(series),
                "causal_yaw_prediction": score[
                    "regime_specific_second_order"],
                "legacy_yaw_baseline_same_causal_inputs": score[
                    "production_legacy"],
                "coverage_fraction": score[
                    "transient_model_coverage_fraction"],
                "by_regime": score["by_regime"],
                "worst_one_step_samples": score["worst_one_step_samples"],
            })
        causal_rows = [
            {"speed_anchor_mps": SPEED_KNOTS_MPS[key[0]],
             "steering_anchor_rad": STEERING_KNOTS_RAD[key[1]],
             "transition_phase": {-1: "unwind", 0: "hold", 1: "turn_in"}[key[2]],
             "training_transitions": causal_counts[key],
             "retention": float(value[0])}
            for key, value in sorted(causal_models.items())
        ]
        report = {
            "title": "Causal regime-local yaw response: command-driven steering input",
            "sample_period_s": DT_S,
            "train_sources": [
                {"name": series.name, "path": series.path,
                 "split": series.split, "sequences": int(len(series.bounds)),
                 "samples": int(len(series.speed))}
                for series in [*training, *phase_sequences]
            ],
            "validation_sources": [
                {"name": series.name, "path": series.path,
                 "split": series.split, "sequences": int(len(series.bounds)),
                 "samples": int(len(series.speed))}
                for series in validation
            ],
            "phase_train_bag_audit": phase_audit,
            "equilibrium_surface": {
                "construction": (
                    "training-only measured steady-response cells; PCHIP in "
                    "signed steering and linear interpolation between the "
                    "three supported speed anchors"
                ),
                "cell_count": len(equilibrium_rows),
                "cells": equilibrium_rows,
            },
            "causal_model": {
                "axes": ["speed anchor", "steering anchor", "turn-in/hold/unwind"],
                "equation": "r[k+1] = q(speed[k], delta_hat[k+1]) + alpha_regime*(r[k]-q(speed[k], delta_hat[k+1]))",
                "steering_equation": "delta_hat[k+1] = delta[k] + clip(command[k-1]-delta[k], +/- 3.2*0.025 rad)",
                "regimes_fitted": len(causal_models),
                "minimum_transitions_per_regime": MIN_REGIME_TRANSITIONS,
                "coefficients": causal_rows,
                "recursive_validation": "not performed; one-step scores only",
            },
            "validation": reports,
            "interpretation_limits": [
                "Simulator truth is used only for offline yaw labels and present-time speed/yaw in this component-isolation score.",
                "The next steering angle is causal and predicted from command history; future measured steering feedback and future speed are not used.",
                "This is not a recursive rollout, full 0-12 m/s model, MPC integration, or vehicle closed-loop validation.",
                "No test/final-test partition or mixed-split archive is read.",
                "The reset-isolated source bags contribute only individually valid training phases.",
            ],
        }
        output.mkdir(parents=True, exist_ok=True)
        _write_json(output / "causal_fit_validation_report.json", report)
        print(f"artifact: {output / 'causal_fit_validation_report.json'}")
        print(f"equilibrium cells: {len(equilibrium_rows)}; causal regimes: {len(causal_models)}")
        for row in reports:
            print(row["name"], "actuator", row["actuator_prediction"][
                "after_first_three_reset_samples"], "causal yaw",
                row["causal_yaw_prediction"], "legacy",
                row["legacy_yaw_baseline_same_causal_inputs"])
        return report

    first_order_models, first_order_counts = _fit_first_order(training, table)
    rollout_models, rollout_fit = _fit_rollout_first_order(
        training, table, first_order_models)
    second_order_models, second_order_counts = _fit_models(training, table)
    causal_training = [
        _causal_actuator_proxy(series)
        for series in [*training, *phase_sequences]
    ]
    causal_validation = [_causal_actuator_proxy(series) for series in validation]
    causal_models, causal_counts = _fit_first_order(causal_training, table)
    reports = []
    for series, causal_series in zip(validation, causal_validation):
        first_order = _score_run(series, table, first_order_models)
        rollout_fit_report = _score_run(series, table, rollout_models)
        second_order = _score_run(series, table, second_order_models)
        causal_response = _score_run(
            causal_series, table, causal_models, include_recursive=False)
        reports.append({
            "name": series.name,
            "split": series.split,
            "source": series.path,
            "eligible_transitions_with_equilibrium_support":
                second_order["eligible_transitions_with_equilibrium_support"],
            "eligible_but_no_first_order_fit":
                first_order["eligible_but_no_transient_fit"],
            "eligible_but_no_second_order_fit":
                second_order["eligible_but_no_transient_fit"],
            "first_order_coverage_fraction":
                first_order["transient_model_coverage_fraction"],
            "second_order_coverage_fraction":
                second_order["transient_model_coverage_fraction"],
            "production_legacy": second_order["production_legacy"],
            "regime_specific_first_order":
                first_order["regime_specific_second_order"],
            "first_order_by_regime": first_order["by_regime"],
            "first_order_recursive_oracle_steering_rollout":
                first_order["recursive_oracle_steering_rollout"],
            "regime_specific_rollout_fitted_first_order":
                rollout_fit_report["regime_specific_second_order"],
            "rollout_fitted_first_order_by_regime":
                rollout_fit_report["by_regime"],
            "rollout_fitted_first_order_recursive_oracle_steering_rollout":
                rollout_fit_report["recursive_oracle_steering_rollout"],
            "regime_specific_second_order":
                second_order["regime_specific_second_order"],
            "second_order_by_regime": second_order["by_regime"],
            "second_order_recursive_oracle_steering_rollout":
                second_order["recursive_oracle_steering_rollout"],
            "causal_command_driven_yaw_one_step": {
                "input_policy": (
                    "current GT speed and current GT yaw as present-time state; "
                    "next steering predicted from measured current steering, "
                    "past steering command, and the production 3.2 rad/s rate limit; "
                    "no future measured speed or steering feedback"
                ),
                "actuator_prediction": _score_steering_actuator(series),
                "yaw_prediction": causal_response[
                    "regime_specific_second_order"],
                "by_regime": causal_response["by_regime"],
                "worst_one_step_samples": causal_response[
                    "worst_one_step_samples"],
            },
        })
    first_order_rows = _coefficient_rows(
        first_order_models, first_order_counts, 1)
    rollout_fit_rows = _coefficient_rows(
        rollout_models, first_order_counts, 1)
    for row in rollout_fit_rows:
        row["model_order"] = "rollout_fitted_first_order"
    second_order_rows = _coefficient_rows(
        second_order_models, second_order_counts, 2)
    model_rows = first_order_rows + rollout_fit_rows + second_order_rows
    run_metrics = []
    for report in reports:
        for kind in ("production_legacy", "regime_specific_first_order",
                     "regime_specific_rollout_fitted_first_order",
                     "regime_specific_second_order"):
            metric = report[kind]
            if metric.get("samples", 0):
                run_metrics.append({
                    "run": report["name"], "model": kind,
                    "samples": metric["samples"],
                    "rmse_radps": metric["rmse_radps"],
                    "p95_abs_radps": metric["p95_abs_radps"],
                    "max_abs_radps": metric["max_abs_radps"],
                    "fraction_abs_error_below_0p1": metric[
                        "fraction_abs_error_below_0p1"],
                })
    report = {
        "title": "Speed/steering/phase-specific yaw step-response fit",
        "created_utc_note": "Use current machine time when publishing progress.",
        "sample_period_s": DT_S,
        "train_sources": [
            {"name": series.name, "path": series.path, "split": series.split,
             "sequences": int(len(series.bounds)), "samples": int(len(series.speed))}
            for series in training
        ],
        "validation_sources": [
            {"name": series.name, "path": series.path, "split": series.split,
             "sequences": int(len(series.bounds)), "samples": int(len(series.speed))}
            for series in validation
        ],
        "phase_train_bag_audit": phase_audit,
        "equilibrium_surface": {
            "construction": "run/phase-level medians on stable measured steering; PCHIP in signed steering and linear interpolation only between supported speed knots",
            "cell_count": len(equilibrium_rows),
            "cells": equilibrium_rows,
        },
        "transient_model": {
            "equation": "e[k+1]=a1*(r[k]-q[k+1])+a2*(r[k-1]-q[k+1])+b0*(q[k]-q[k+1])+b1*(q[k-1]-q[k+1]); rhat[k+1]=q[k+1]+e[k+1]",
            "regime_axes": ["speed anchor", "absolute steering anchor", "turn-in/hold/unwind"],
            "stable_poles_enforced": True,
            "minimum_training_transitions_per_regime": MIN_REGIME_TRANSITIONS,
            "first_order_fitted_regimes": len(first_order_models),
            "rollout_fitted_first_order_regimes": len(rollout_models),
            "second_order_fitted_regimes": len(second_order_models),
            "rollout_fit": rollout_fit,
            "first_order_coefficients": first_order_rows,
            "rollout_fitted_first_order_coefficients": rollout_fit_rows,
            "coefficients": model_rows,
        },
        "causal_command_driven_model": {
            "equation": "first-order local yaw relaxation toward the empirical equilibrium surface, conditioned on speed/steering/turn phase; future wheel angle is predicted from delayed command and current measured feedback",
            "training_regimes": len(causal_models),
            "minimum_training_transitions_per_regime": MIN_REGIME_TRANSITIONS,
            "regime_axes": ["speed anchor", "absolute steering anchor", "turn-in/hold/unwind"],
            "training_transition_counts": [
                {"speed_anchor_mps": SPEED_KNOTS_MPS[key[0]],
                 "steering_anchor_rad": STEERING_KNOTS_RAD[key[1]],
                 "transition_phase": {-1: "unwind", 0: "hold", 1: "turn_in"}[key[2]],
                 "transitions": count,
                 "fitted": key in causal_models}
                for key, count in sorted(causal_counts.items())
            ],
            "actuator_prediction": "delta[k+1] = delta[k] + clip(command[k-1] - delta[k], +/- 3.2*0.025 rad)",
            "recursive_validation": "not claimed; this score is one-step and teacher-forced only on present-time GT speed/yaw and current steering feedback",
        },
        "validation": reports,
        "run_level_metrics": run_metrics,
        "interpretation_limits": [
            "GT yaw is an offline label only.",
            "The legacy one-step and recursive yaw-response scores receive recorded target steering feedback and target speed; they are oracle-input component scores, not causal predictions.",
            "The new causal command-driven score predicts the next steering angle from current feedback and delayed command, but remains a one-step yaw component score conditioned on present-time ground-truth speed and yaw. It is not yet an MPC rollout or a full 0-12 m/s model.",
            "No test/final-test partition or mixed-split archive is read.",
            "The fit is supported only where the measured speed/steering/phase atlas has training transitions; unsupported samples are reported, not filled by a global fallback.",
            "The reset-isolated source bags were parent-aborted by an emergency speed cutoff; only their individually valid phases are admitted as training examples.",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "fit_validation_report.json", report)
    with (output / "regime_coefficients.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(model_rows[0]) if model_rows else [])
        if model_rows:
            writer.writeheader()
            writer.writerows(model_rows)
    with (output / "equilibrium_response_cells.csv").open("w", newline="", encoding="utf-8") as stream:
        columns = ("speed_anchor_mps", "steering_feedback_rad", "yaw_rate_rps",
                   "source_count", "source_names", "source_medians_rps",
                   "source_spread_rps")
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in equilibrium_rows:
            writer.writerow({**row,
                             "source_names": ";".join(row["source_names"]),
                             "source_medians_rps": ";".join(
                                 f"{x:.9g}" for x in row["source_medians_rps"])})
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "live_runs/racing_model_diagnostics_20261007/"
                "regime_yaw_step_response_causal_v2",
        help="workspace path for the fitted artifact and whole-run scores",
    )
    parser.add_argument(
        "--causal-only", action="store_true",
        help=("fit/score the causal command-driven one-step yaw model only; "
              "skip the known-slow retrospective recursive optimizer"),
    )
    args = parser.parse_args()
    report = run(args.output, causal_only=args.causal_only)
    print(f"artifact: {args.output}")
    print(f"equilibrium cells: {report['equilibrium_surface']['cell_count']}")
    if args.causal_only:
        print("causal yaw regimes:", report["causal_model"]["regimes_fitted"])
        for run_report in report["validation"]:
            print(run_report["name"],
                  "causal", run_report["causal_yaw_prediction"],
                  "legacy", run_report[
                      "legacy_yaw_baseline_same_causal_inputs"],
                  "coverage", run_report["coverage_fraction"])
    else:
        print("transient regimes:",
              report["transient_model"]["first_order_fitted_regimes"],
              "first-order,",
              report["transient_model"]["second_order_fitted_regimes"],
              "second-order")
        for run_report in report["validation"]:
            print(run_report["name"],
                  "legacy", run_report["production_legacy"],
                  "first-order", run_report["regime_specific_first_order"],
                  "rollout-fit", run_report[
                      "regime_specific_rollout_fitted_first_order"],
                  "second-order", run_report["regime_specific_second_order"],
                  "coverage",
                  run_report["first_order_coverage_fraction"],
                  run_report["second_order_coverage_fraction"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
