#!/usr/bin/env python3
"""Fit and whole-capture score an empirical high-speed yaw-response surface.

Only the explicitly listed training captures below are used to fit the table.
The separate validation captures are opened only after the table is frozen in
memory. This script does not inspect sealed test/final-test partitions, launch
the simulator, or modify production configuration.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
DT_S = 0.025
YAW_TAU_S = 0.015
YAW_GAIN_PER_M = 2.95
STEER_GAIN_DROP_PER_RAD = 35.6
STEER_GAIN_DROP_START_RAD = 0.41
STEER_GAIN_DROP_WIDTH_RAD = 0.05
STEERING_KNOTS = (0.15, 0.20, 0.21, 0.22, 0.23,
                  0.25, 0.30, 0.35, 0.42, 0.50)
SPEED_KNOTS = (4.98502375, 6.5, 7.5)
BLEND_Q_START = 0.78
BLEND_Q_END = 0.90
SPEED_BLEND_MARGIN_MPS = 0.10

TRAINING_SOURCES = (
    ("race_train_r02", "train",
     "live_runs/derived_dynamics_learning_20260928/race_domain_train_runs_20261001_r02/openplane_dynamics.npz"),
    ("race_train_r03", "train",
     "live_runs/derived_dynamics_learning_20260928/race_domain_train_runs_20261001_r03/openplane_dynamics.npz"),
    ("highsteer_train_r02", "train",
     "live_runs/openplane_subnet_highsteer_transients_train_r02_20261004/source_continuous_v1/openplane_dynamics.npz"),
    ("highsteer_train_r03", "train",
     "live_runs/openplane_subnet_highsteer_transients_train_r03_20261004/source_continuous_v1/openplane_dynamics.npz"),
)

VALIDATION_SOURCES = (
    ("race_validation_r04", "validation",
     "live_runs/derived_dynamics_learning_20260928/race_domain_validation_runs_20261001_r04/openplane_dynamics.npz"),
    ("race_validation_r05", "validation",
     "live_runs/derived_dynamics_learning_20260928/race_domain_validation_runs_20261001_r05/openplane_dynamics.npz"),
    ("highsteer_validation_r01", "validation",
     "live_runs/openplane_subnet_highsteer_transients_validation_r01_20261004/source_continuous_v1/openplane_dynamics.npz"),
    ("highsteer_validation_r02", "validation",
     "live_runs/openplane_subnet_highsteer_transients_validation_r02_20261004/source_continuous_v1/openplane_dynamics.npz"),
    ("highsteer_validation_r03", "validation",
     "live_runs/openplane_subnet_highsteer_transients_validation_r03_20261004/source_continuous_v1/openplane_dynamics.npz"),
)

PRACTICE_DIAGNOSTIC_SOURCES = (
    ("practice_steerclamp_r03", "live_runs/practice_q078090_steerclamp_r03_20261006/run/run_0.db3"),
    ("practice_heading20_r04", "live_runs/practice_q078090_heading20_r04_20261006/run/run_0.db3"),
    ("practice_lowfade060_r01", "live_runs/practice_yaw_lowfade060_closedloop_r01_20261006/run/run_0.db3"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_capture(name: str, split: str, relative_path: str
                  ) -> tuple[dict[str, np.ndarray], dict[str, Any], Path]:
    path = (ROOT / relative_path).resolve()
    manifest_path = path.with_name("manifest.json")
    if not path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"missing source or manifest for {name}: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest.get("runs", [])
    if len(rows) != 1:
        raise ValueError(f"{name} must contain exactly one whole run")
    run = rows[0]
    if (run.get("effective_split") != split or run.get("aborted")
            or run.get("reason") != "schedule complete"
            or not run.get("clean_stream_and_collision_gate")
            or any(int(value) != 0 for value in run.get("collisions", []))
            or int(run.get("timing_faults", -1)) != 0
            or run.get("quality_failures")
            or run.get("whole_bag_quality_failures")):
        raise ValueError(f"{name} did not pass the frozen {split} quality gate")
    if any(float(stats.get("hz", 0.0)) < 38.0
           for stats in run.get("streams", {}).values()):
        raise ValueError(f"{name} has a stream below 38 Hz")
    with np.load(path, allow_pickle=False) as archive:
        data = {key: np.asarray(archive[key]) for key in archive.files}
    required = ("frames", "simulator_rigid_state", "sequence_bounds", "dt_s",
                "run_ids", "run_splits")
    if any(key not in data for key in required):
        raise ValueError(f"{name} is missing required plant labels")
    if (len(data["run_ids"]) != 1
            or str(data["run_splits"][0]) != split
            or not np.allclose(data["dt_s"], DT_S, rtol=0.0, atol=1.0e-7)
            or data["frames"].shape[1] != 9
            or data["simulator_rigid_state"].shape[1] != 13):
        raise ValueError(f"{name} archive split, cadence, or schema mismatch")
    if not (np.isfinite(data["frames"]).all()
            and np.isfinite(data["simulator_rigid_state"]).all()):
        raise ValueError(f"{name} has non-finite state/input labels")
    return data, run, path


def _pchip_slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    h = np.diff(x)
    d = np.diff(y) / h
    slopes = np.zeros_like(y)

    def endpoint(h0: float, h1: float, d0: float, d1: float) -> float:
        result = ((2.0 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
        if result * d0 <= 0.0:
            return 0.0
        if d0 * d1 < 0.0 and abs(result) > 3.0 * abs(d0):
            return 3.0 * d0
        return result

    slopes[0] = endpoint(h[0], h[1], d[0], d[1])
    slopes[-1] = endpoint(h[-1], h[-2], d[-1], d[-2])
    for index in range(1, len(x) - 1):
        left, right = d[index - 1], d[index]
        if left * right <= 0.0:
            slopes[index] = 0.0
        else:
            w_left = 2.0 * h[index] + h[index - 1]
            w_right = h[index] + 2.0 * h[index - 1]
            slopes[index] = (w_left + w_right) / (
                w_left / left + w_right / right)
    return slopes


def _pchip_value(x: np.ndarray, y: np.ndarray, query: float) -> float:
    if query <= x[0]:
        return float(y[0])
    if query >= x[-1]:
        return float(y[-1])
    slopes = _pchip_slopes(x, y)
    index = int(np.searchsorted(x, query, side="left") - 1)
    width = x[index + 1] - x[index]
    t = (query - x[index]) / width
    t2, t3 = t * t, t * t * t
    return float(
        (2 * t3 - 3 * t2 + 1) * y[index]
        + (t3 - 2 * t2 + t) * width * slopes[index]
        + (-2 * t3 + 3 * t2) * y[index + 1]
        + (t3 - t2) * width * slopes[index + 1])


def _stable_rows(data: dict[str, np.ndarray], speed: tuple[float, float],
                 steering: float, turn_sign: int) -> np.ndarray:
    frames = data["frames"].astype(np.float64, copy=False)
    rigid = data["simulator_rigid_state"].astype(np.float64, copy=False)
    rate = np.zeros(len(frames), dtype=np.float64)
    for begin_raw, end_raw in data["sequence_bounds"]:
        begin, end = int(begin_raw), int(end_raw)
        rate[begin + 1:end] = np.diff(frames[begin:end, 3]) / DT_S
    speed_mps = np.hypot(rigid[:, 7], rigid[:, 8])
    angle = frames[:, 3]
    return np.flatnonzero(
        (speed_mps >= speed[0]) & (speed_mps <= speed[1])
        & (np.abs(angle - turn_sign * steering) <= 0.012)
        & (np.abs(rate) <= 0.5))


def _fit_point(sources: list[tuple[str, dict[str, np.ndarray]]],
               speed_band: tuple[float, float], steering: float,
               turn_sign: int) -> tuple[float, float, dict[str, Any]]:
    per_run: dict[str, dict[str, float | int]] = {}
    for name, data in sources:
        indices = _stable_rows(data, speed_band, steering, turn_sign)
        if len(indices) < 20:
            continue
        frames = data["frames"].astype(np.float64, copy=False)
        rigid = data["simulator_rigid_state"].astype(np.float64, copy=False)
        speed = np.hypot(rigid[indices, 7], rigid[indices, 8])
        per_run[name] = {
            "samples": int(len(indices)),
            "speed_median_mps": float(np.median(speed)),
            "q_median": float(np.median(speed * np.abs(np.tan(frames[indices, 3])))),
            "signed_yaw_rate_median_rps": float(
                np.median(turn_sign * rigid[indices, 12])),
        }
    if not per_run:
        raise ValueError(
            f"no training support at speed {speed_band}, steering {steering}, "
            f"turn sign {turn_sign}")
    rates = np.asarray([row["signed_yaw_rate_median_rps"]
                        for row in per_run.values()], dtype=np.float64)
    q_values = np.asarray([row["q_median"] for row in per_run.values()],
                          dtype=np.float64)
    rate = float(np.median(rates))
    q = float(np.median(q_values))
    if not math.isfinite(rate) or rate <= 0.0:
        raise ValueError("training response is not a positive signed yaw response")
    return q, rate, per_run


def _load_existing_low_speed_surface() -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    path = ROOT / "f1tenth_mpc/config/yaw_response_surface.csv"
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(
            line for line in stream if not line.lstrip().startswith("#")))
    if len(rows) != 60:
        raise ValueError("existing low-speed reference surface must remain 60 rows")
    speeds = sorted({float(row["speed_knot_mps"]) for row in rows})
    if len(speeds) != 3:
        raise ValueError("existing low-speed surface must have three speed knots")
    last_speed_rows = [row for row in rows
                       if abs(float(row["speed_knot_mps"]) - speeds[-1]) < 1.0e-6]
    positive = [row for row in last_speed_rows if int(row["turn_sign"]) == 1]
    if tuple(float(row["steering_rad"]) for row in positive) != STEERING_KNOTS:
        raise ValueError("existing low-speed surface steering grid changed")
    rate = np.asarray([float(row["yaw_rate_abs_rps"]) for row in positive])
    return (np.asarray(speeds, dtype=np.float64), rate,
            {"path": str(path.relative_to(ROOT)), "sha256": _sha256(path),
             "speed_knot_mps": speeds,
             "fitted_speed_knot_mps": speeds[-1]})


def _fit_surface(training: list[tuple[str, dict[str, np.ndarray]]]
                  ) -> tuple[np.ndarray, list[dict[str, Any]]]:
    old_speeds, rate5, old_source = _load_existing_low_speed_surface()
    if abs(old_speeds[-1] - SPEED_KNOTS[0]) > 0.02:
        raise ValueError("high-speed candidate must join the current surface at ~5 m/s")
    rates_by_speed: list[np.ndarray] = [rate5]
    diagnostics: list[dict[str, Any]] = [{
        "speed_knot_mps": float(SPEED_KNOTS[0]),
        "source": "existing frozen 3-5 m/s development surface",
        "response_abs_rps_by_steering_knot": rate5.tolist(),
    }]

    measured_65: list[tuple[float, float, dict[str, Any]]] = []
    for angle, band in ((0.05, (6.20, 6.80)),
                        (0.10, (6.20, 6.80)),
                        (0.20, (6.20, 6.80))):
        q_by_sign = []
        rate_by_sign = []
        details = {}
        for sign in (-1, 1):
            q, response, per_run = _fit_point(training, band, angle, sign)
            q_by_sign.append(q)
            rate_by_sign.append(response)
            details["left" if sign < 0 else "right"] = per_run
        measured_65.append((float(np.mean(q_by_sign)),
                            float(np.mean(rate_by_sign)), details))
    measured_65.sort(key=lambda value: value[0])
    q65 = np.asarray([row[0] for row in measured_65])
    y65 = np.asarray([row[1] for row in measured_65])

    measured_75: list[tuple[float, float, dict[str, Any]]] = []
    for angle, band, source_subset in (
            (0.10, (7.55, 8.20), training[:2]),
            (0.30, (7.20, 7.90), training[2:]),
            (0.42, (7.20, 7.90), training[2:])):
        q_by_sign = []
        rate_by_sign = []
        details = {}
        for sign in (-1, 1):
            q, response, per_run = _fit_point(source_subset, band, angle, sign)
            q_by_sign.append(q)
            rate_by_sign.append(response)
            details["left" if sign < 0 else "right"] = per_run
        measured_75.append((float(np.mean(q_by_sign)),
                            float(np.mean(rate_by_sign)), details))
    measured_75.sort(key=lambda value: value[0])
    q75 = np.asarray([row[0] for row in measured_75])
    y75 = np.asarray([row[1] for row in measured_75])

    row75 = []
    for angle in STEERING_KNOTS:
        query_q = SPEED_KNOTS[2] * math.tan(angle)
        row75.append(_pchip_value(q75, y75, query_q))
    row75 = np.asarray(row75)

    row65 = []
    for angle in STEERING_KNOTS:
        query_q = SPEED_KNOTS[1] * math.tan(angle)
        if query_q <= q65[-1]:
            response = _pchip_value(q65, y65, query_q)
        else:
            low = _pchip_value(SPEED_KNOTS[0] * np.tan(np.asarray(STEERING_KNOTS)),
                               rate5, query_q)
            high = _pchip_value(q75, y75, query_q)
            speed_fraction = ((SPEED_KNOTS[1] - SPEED_KNOTS[0]) /
                              (SPEED_KNOTS[2] - SPEED_KNOTS[0]))
            response = low + speed_fraction * (high - low)
        row65.append(response)
    row65 = np.asarray(row65)
    rates_by_speed.extend((row65, row75))
    diagnostics.extend((
        {"speed_knot_mps": float(SPEED_KNOTS[1]),
         "direct_training_points": [
             {"q": float(q), "response_abs_rps": float(y),
              "per_run": detail}
             for q, y, detail in measured_65],
         "response_abs_rps_by_steering_knot": row65.tolist(),
         "high_q_rule": "linear-in-speed bridge between 5.0 and 7.5 m/s",
         "existing_low_speed_source": old_source},
        {"speed_knot_mps": float(SPEED_KNOTS[2]),
         "direct_training_points": [
             {"q": float(q), "response_abs_rps": float(y),
              "per_run": detail}
             for q, y, detail in measured_75],
         "response_abs_rps_by_steering_knot": row75.tolist(),
         "unsupported_above_0p42_rad": True},
    ))
    if not np.isfinite(np.asarray(rates_by_speed)).all() or np.any(
            np.asarray(rates_by_speed) <= 0.0):
        raise ValueError("fitted high-speed response surface is invalid")
    return np.asarray(rates_by_speed), diagnostics


def _legacy_rate(speed: float, steering: float) -> float:
    gain = YAW_GAIN_PER_M - STEER_GAIN_DROP_PER_RAD * min(
        max(abs(steering) - STEER_GAIN_DROP_START_RAD, 0.0),
        STEER_GAIN_DROP_WIDTH_RAD)
    return speed * math.tan(steering) * gain


def _support_weight(speed: float, knot_speeds: np.ndarray,
                    low_fadeout_mps: float,
                    high_fadein_mps: float) -> float:
    if low_fadeout_mps <= 0.0 and high_fadein_mps <= 0.0:
        return 1.0
    if len(knot_speeds) < 3:
        raise ValueError("support-gated surface needs three speed knots")
    low_t = (float(np.clip((speed - knot_speeds[0]) / low_fadeout_mps,
                           0.0, 1.0)) if low_fadeout_mps > 0.0 else 0.0)
    high_t = (float(np.clip(
        (speed - (knot_speeds[2] - high_fadein_mps)) / high_fadein_mps,
        0.0, 1.0)) if high_fadein_mps > 0.0 else 0.0)
    low_weight = 1.0 - low_t * low_t * (3.0 - 2.0 * low_t)
    high_weight = high_t * high_t * (3.0 - 2.0 * high_t)
    return low_weight + high_weight


def _surface_rate(speed: float, steering: float, table: dict[str, Any],
                  blend_q_start: float, blend_q_end: float,
                  margin: float,
                  low_speed_margin: float | None = None,
                  low_support_fadeout_mps: float = 0.0,
                  high_support_fadein_mps: float = 0.0) -> float:
    input_speed = float(speed)
    legacy = _legacy_rate(speed, steering)
    if steering == 0.0:
        return legacy
    q = max(speed, 0.0) * abs(math.tan(steering))
    sign = -1.0 if steering < 0.0 else 1.0
    direction = 0 if steering < 0.0 else 1
    knot_speeds = table["speed_mps"]
    speed_rows = [
        _pchip_value(table["q"][index, direction],
                     table["rate"][index, direction], q)
        for index in range(len(knot_speeds))
    ]
    query_speed = float(np.clip(input_speed, knot_speeds[0], knot_speeds[-1]))
    interval = int(np.clip(np.searchsorted(knot_speeds, query_speed, side="left") - 1,
                            0, len(knot_speeds) - 2))
    fraction = ((query_speed - knot_speeds[interval]) /
                (knot_speeds[interval + 1] - knot_speeds[interval]))
    empirical_abs = speed_rows[interval] + fraction * (
        speed_rows[interval + 1] - speed_rows[interval])
    q_t = float(np.clip((q - blend_q_start) / (blend_q_end - blend_q_start), 0.0, 1.0))
    q_blend = q_t * q_t * (3.0 - 2.0 * q_t)
    low_margin = margin if low_speed_margin is None else low_speed_margin
    if low_margin > 0.0:
        enter_t = float(np.clip(
            (input_speed - (knot_speeds[0] - low_margin)) / low_margin,
            0.0, 1.0))
        enter = enter_t * enter_t * (3.0 - 2.0 * enter_t)
    else:
        enter = 1.0
    if margin > 0.0:
        leave_t = float(np.clip(((knot_speeds[-1] + margin) - input_speed) / margin,
                                0.0, 1.0))
        leave = leave_t * leave_t * (3.0 - 2.0 * leave_t)
    else:
        leave = 1.0
    speed_blend = enter * leave * _support_weight(
        input_speed, knot_speeds, low_support_fadeout_mps,
        high_support_fadein_mps)
    return legacy + q_blend * speed_blend * (sign * empirical_abs - legacy)


def _score_capture(name: str, data: dict[str, np.ndarray],
                   baseline: dict[str, Any], candidate: dict[str, Any],
                   candidate_low_speed_margin: float = SPEED_BLEND_MARGIN_MPS,
                   support_fadeout_mps: float = 0.0,
                   support_fadein_mps: float = 0.0,
                   ) -> dict[str, Any]:
    frames = data["frames"].astype(np.float64, copy=False)
    rigid = data["simulator_rigid_state"].astype(np.float64, copy=False)
    speed = np.hypot(rigid[:, 7], rigid[:, 8])
    current_r = rigid[:, 12]
    next_r = np.full(len(current_r), np.nan, dtype=np.float64)
    valid = np.zeros(len(current_r), dtype=bool)
    for begin_raw, end_raw in data["sequence_bounds"]:
        begin, end = int(begin_raw), int(end_raw)
        if end - begin > 1:
            next_r[begin:end - 1] = current_r[begin + 1:end]
            valid[begin:end - 1] = True
    keep = valid & np.isfinite(next_r) & (speed >= 2.0)
    indices = np.flatnonzero(keep)
    retention = math.exp(-DT_S / YAW_TAU_S)
    legacy_errors = []
    baseline_errors = []
    candidate_errors = []
    ungated_candidate_errors = []
    high_errors = {"baseline": [], "candidate": []}
    unsupported_errors = {"baseline": [], "candidate": []}
    low_speed_errors = {"baseline": [], "candidate": []}
    mid_speed_errors = {
        "legacy": [], "baseline": [], "ungated_candidate": [],
        "candidate": []}
    for index in indices:
        steering = float(frames[index, 3])
        legacy_next = (retention * current_r[index]
                       + (1.0 - retention) * _legacy_rate(speed[index], steering))
        baseline_next = (retention * current_r[index]
                         + (1.0 - retention) * _surface_rate(
                             speed[index], steering, baseline, BLEND_Q_START,
                             BLEND_Q_END, SPEED_BLEND_MARGIN_MPS))
        candidate_next = (retention * current_r[index]
                          + (1.0 - retention) * _surface_rate(
                              speed[index], steering, candidate, BLEND_Q_START,
                              BLEND_Q_END, SPEED_BLEND_MARGIN_MPS,
                              candidate_low_speed_margin,
                              support_fadeout_mps, support_fadein_mps))
        ungated_candidate_next = (retention * current_r[index]
                                  + (1.0 - retention) * _surface_rate(
                                      speed[index], steering, candidate,
                                      BLEND_Q_START, BLEND_Q_END,
                                      SPEED_BLEND_MARGIN_MPS,
                                      candidate_low_speed_margin))
        legacy_error = legacy_next - next_r[index]
        baseline_error = baseline_next - next_r[index]
        candidate_error = candidate_next - next_r[index]
        ungated_candidate_error = ungated_candidate_next - next_r[index]
        legacy_errors.append(legacy_error)
        baseline_errors.append(baseline_error)
        candidate_errors.append(candidate_error)
        ungated_candidate_errors.append(ungated_candidate_error)
        q = speed[index] * abs(math.tan(steering))
        if (baseline["speed_mps"][0] + support_fadeout_mps < speed[index]
                < candidate["speed_mps"][2] - support_fadein_mps
                and q >= BLEND_Q_START and abs(steering) <= 0.42):
            mid_speed_errors["legacy"].append(legacy_error)
            mid_speed_errors["baseline"].append(baseline_error)
            mid_speed_errors["ungated_candidate"].append(ungated_candidate_error)
            mid_speed_errors["candidate"].append(candidate_error)
        if (SPEED_KNOTS[0] - SPEED_BLEND_MARGIN_MPS <= speed[index]
                <= SPEED_KNOTS[-1] and q >= BLEND_Q_START
                and abs(steering) <= 0.42):
            high_errors["baseline"].append(baseline_error)
            high_errors["candidate"].append(candidate_error)
        if (SPEED_KNOTS[0] - SPEED_BLEND_MARGIN_MPS <= speed[index]
                <= SPEED_KNOTS[-1] and q >= BLEND_Q_START
                and abs(steering) > 0.42):
            unsupported_errors["baseline"].append(baseline_error)
            unsupported_errors["candidate"].append(candidate_error)
        if (speed[index] >= baseline["speed_mps"][0] - candidate_low_speed_margin
                and speed[index] < baseline["speed_mps"][0]
                and q >= BLEND_Q_START and abs(steering) <= 0.42):
            low_speed_errors["baseline"].append(baseline_error)
            low_speed_errors["candidate"].append(candidate_error)

    def metric(values: list[float]) -> dict[str, float | int | None]:
        array = np.asarray(values, dtype=np.float64)
        if not len(array):
            return {"samples": 0, "rmse_rps": None, "mae_rps": None,
                    "bias_rps": None, "p95_abs_rps": None}
        return {"samples": int(len(array)),
                "rmse_rps": float(np.sqrt(np.mean(array ** 2))),
                "mae_rps": float(np.mean(np.abs(array))),
                "bias_rps": float(np.mean(array)),
                "p95_abs_rps": float(np.quantile(np.abs(array), 0.95))}

    return {
        "all_valid_samples": metric(candidate_errors),
        "all_valid_baseline_surface": metric(baseline_errors),
        "all_valid_legacy_no_surface": metric(legacy_errors),
        "high_speed_high_demand": {
            "definition": (
                "4.885 <= speed <= 7.5 m/s, abs(steering) <= 0.42 rad, "
                "and speed*abs(tan(steering)) >= 0.78"),
            "baseline_surface": metric(high_errors["baseline"]),
            "candidate": metric(high_errors["candidate"]),
        },
        "low_speed_high_demand": {
            "definition": (
                f"{baseline['speed_mps'][0] - candidate_low_speed_margin:.3f} <= speed < "
                f"{baseline['speed_mps'][0]:.3f} m/s, abs(steering) <= 0.42 rad, "
                f"and speed*abs(tan(steering)) >= {BLEND_Q_START:.2f}"),
            "baseline_surface": metric(low_speed_errors["baseline"]),
            "candidate": metric(low_speed_errors["candidate"]),
        },
        "unsupported_middle_speed_high_demand": {
            "definition": (
                "between the measured low-speed band and high-speed support "
                "fade-in; q >= 0.78 and |steering| <= 0.42 rad"),
            "legacy_no_surface": metric(mid_speed_errors["legacy"]),
            "baseline_surface": metric(mid_speed_errors["baseline"]),
            "candidate_without_support_gate": metric(
                mid_speed_errors["ungated_candidate"]),
            "candidate_with_support_gate": metric(mid_speed_errors["candidate"]),
        },
        "unsupported_high_steering_in_surface_speed_band": {
            "definition": (
                "4.885 <= speed <= 7.5 m/s, abs(steering) > 0.42 rad, "
                "and speed*abs(tan(steering)) >= 0.78"),
            "samples": len(unsupported_errors["baseline"]),
            "legacy_rmse_rps": (
                float(np.sqrt(np.mean(np.square(unsupported_errors["baseline"]))))
                if unsupported_errors["baseline"] else None),
            "candidate_rmse_rps": (
                float(np.sqrt(np.mean(np.square(unsupported_errors["candidate"]))))
                if unsupported_errors["candidate"] else None),
        },
        "all_valid_candidate": metric(candidate_errors),
        "all_valid_candidate_without_support_gate": metric(
            ungated_candidate_errors),
        "all_valid_baseline_surface": metric(baseline_errors),
    }


def _run_bootstrap(validation_rows: dict[str, dict[str, Any]],
                   repeats: int = 10000) -> dict[str, Any]:
    names = sorted(validation_rows)
    deltas = np.asarray([
        validation_rows[name]["high_speed_high_demand"]["candidate"]["rmse_rps"]
        - validation_rows[name]["high_speed_high_demand"]["baseline_surface"]["rmse_rps"]
        for name in names], dtype=np.float64)
    rng = np.random.default_rng(20261006)
    draws = deltas[rng.integers(0, len(deltas), (repeats, len(deltas)))].mean(axis=1)
    return {"independent_run_count": len(names),
            "per_run_candidate_minus_baseline_rmse_rps": dict(zip(names, deltas.tolist())),
            "macro_run_mean_candidate_minus_baseline_rmse_rps": float(np.mean(deltas)),
            "run_cluster_bootstrap_95pct_ci": np.quantile(draws, (0.025, 0.975)).tolist()}


def _first_collision_receipt_ns(bag_path: Path) -> int | None:
    import sqlite3

    from tools.evaluate_open_plane_body_dynamics import analysis

    connection = sqlite3.connect(bag_path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        topics = analysis._topic_map(connection)
        previous = 0
        for receipt_ns, message in analysis._messages(
                connection, topics, analysis.COLLISIONS):
            count = int(message.data)
            if count > previous:
                return int(receipt_ns)
            previous = count
    finally:
        connection.close()
    return None


def _score_practice_bag(name: str, path: Path,
                        baseline: dict[str, Any], candidate: dict[str, Any],
                        candidate_low_speed_margin: float = SPEED_BLEND_MARGIN_MPS,
                        support_fadeout_mps: float = 0.0,
                        support_fadein_mps: float = 0.0,
                        ) -> dict[str, Any]:
    from tools.evaluate_open_plane_body_dynamics import load_capture

    collision_receipt_ns = _first_collision_receipt_ns(path)
    capture = load_capture(path, include_nonvalid_phases=True,
                           continuous_phased_run=True)
    retention = math.exp(-DT_S / YAW_TAU_S)
    errors = {"baseline": [], "candidate": []}
    ungated_candidate_errors = []
    moving_errors = {"baseline": [], "candidate": []}
    high = {"baseline": [], "candidate": []}
    mid_speed = {
        "legacy": [], "baseline": [], "ungated_candidate": [],
        "candidate": []}
    operating = []
    for sequence in capture.sequences:
        for current, following in zip(sequence, sequence[1:]):
            if (current.packet_sequence < 0
                    or following.packet_sequence != current.packet_sequence + 1
                    or (collision_receipt_ns is not None
                        and following.receipt_ns >= collision_receipt_ns)
                    or current.simulator_rigid_state is None
                    or following.simulator_rigid_state is None):
                continue
            rigid = np.asarray(current.simulator_rigid_state, dtype=np.float64)
            rigid_next = np.asarray(following.simulator_rigid_state, dtype=np.float64)
            if not (np.isfinite(rigid).all() and np.isfinite(rigid_next).all()):
                continue
            speed = float(np.hypot(rigid[7], rigid[8]))
            steering = float(current.actuators[0])
            yaw_rate = float(rigid[12])
            target = float(rigid_next[12])
            q = speed * abs(math.tan(steering))
            baseline_next = (retention * yaw_rate + (1.0 - retention)
                             * _surface_rate(
                                 speed, steering, baseline, BLEND_Q_START,
                                 BLEND_Q_END, SPEED_BLEND_MARGIN_MPS))
            candidate_next = (retention * yaw_rate + (1.0 - retention)
                              * _surface_rate(
                                  speed, steering, candidate, BLEND_Q_START,
                                  BLEND_Q_END, SPEED_BLEND_MARGIN_MPS,
                                  candidate_low_speed_margin,
                                  support_fadeout_mps, support_fadein_mps))
            ungated_candidate_next = (retention * yaw_rate + (1.0 - retention)
                                      * _surface_rate(
                                          speed, steering, candidate,
                                          BLEND_Q_START, BLEND_Q_END,
                                          SPEED_BLEND_MARGIN_MPS,
                                          candidate_low_speed_margin))
            old_error = baseline_next - target
            new_error = candidate_next - target
            ungated_candidate_error = ungated_candidate_next - target
            legacy_error = ((retention * yaw_rate + (1.0 - retention)
                             * _legacy_rate(speed, steering)) - target)
            errors["baseline"].append(old_error)
            errors["candidate"].append(new_error)
            ungated_candidate_errors.append(ungated_candidate_error)
            if speed >= 2.0:
                moving_errors["baseline"].append(old_error)
                moving_errors["candidate"].append(new_error)
            if (SPEED_KNOTS[0] - SPEED_BLEND_MARGIN_MPS <= speed
                    <= SPEED_KNOTS[-1] and q >= BLEND_Q_START
                    and abs(steering) <= 0.42):
                high["baseline"].append(old_error)
                high["candidate"].append(new_error)
            if (baseline["speed_mps"][0] + support_fadeout_mps < speed
                    < candidate["speed_mps"][2] - support_fadein_mps
                    and q >= BLEND_Q_START and abs(steering) <= 0.42):
                mid_speed["baseline"].append(old_error)
                mid_speed["legacy"].append(legacy_error)
                mid_speed["ungated_candidate"].append(ungated_candidate_error)
                mid_speed["candidate"].append(new_error)
            operating.append((speed, steering, q))

    def metric(values: list[float]) -> dict[str, float | int | None]:
        array = np.asarray(values, dtype=np.float64)
        if not len(array):
            return {"samples": 0, "rmse_rps": None, "mae_rps": None,
                    "bias_rps": None, "p95_abs_rps": None}
        return {"samples": int(len(array)),
                "rmse_rps": float(np.sqrt(np.mean(array ** 2))),
                "mae_rps": float(np.mean(np.abs(array))),
                "bias_rps": float(np.mean(array)),
                "p95_abs_rps": float(np.quantile(np.abs(array), 0.95))}

    op = np.asarray(operating, dtype=np.float64)
    return {
        "bag": str(path.relative_to(ROOT)),
        "bag_sha256": _sha256(path),
        "collision_count_start": int(capture.collision_count_start),
        "collision_count_end": int(capture.collision_count_end),
        "first_collision_receipt_ns": collision_receipt_ns,
        "post_collision_samples_excluded": collision_receipt_ns is not None,
        "sequence_count": len(capture.sequences),
        "scored_consecutive_pairs": len(errors["baseline"]),
        "whole_preimpact_baseline_surface": metric(errors["baseline"]),
        "whole_preimpact_candidate": metric(errors["candidate"]),
        "whole_preimpact_candidate_without_support_gate": metric(
            ungated_candidate_errors),
        "moving_speed_ge_2mps": {
            "baseline_surface": metric(moving_errors["baseline"]),
            "candidate": metric(moving_errors["candidate"]),
            "fraction_of_scored_pairs": (
                len(moving_errors["baseline"]) / len(errors["baseline"])
                if errors["baseline"] else 0.0),
        },
        "supported_high_speed_high_demand": {
            "definition": (
                "4.885 <= speed <= 7.5 m/s, abs(steering) <= 0.42 rad, "
                "and speed*abs(tan(steering)) >= 0.78"),
            "baseline_surface": metric(high["baseline"]),
            "candidate": metric(high["candidate"]),
        },
        "unsupported_middle_speed_high_demand": {
            "legacy_no_surface": metric(mid_speed["legacy"]),
            "baseline_surface": metric(mid_speed["baseline"]),
            "candidate_without_support_gate": metric(
                mid_speed["ungated_candidate"]),
            "candidate_with_support_gate": metric(mid_speed["candidate"]),
        },
        "operating_range": ({
            "all_pairs": {
                "speed_mps_p05_p50_p95": np.quantile(
                    op[:, 0], (0.05, 0.5, 0.95)).tolist(),
                "abs_steering_rad_p05_p50_p95": np.quantile(
                    np.abs(op[:, 1]), (0.05, 0.5, 0.95)).tolist(),
                "demand_q_p05_p50_p95": np.quantile(
                    op[:, 2], (0.05, 0.5, 0.95)).tolist(),
            },
            "moving_speed_ge_2mps": ({
                "speed_mps_p05_p50_p95": np.quantile(
                    op[op[:, 0] >= 2.0, 0], (0.05, 0.5, 0.95)).tolist(),
                "abs_steering_rad_p05_p50_p95": np.quantile(
                    np.abs(op[op[:, 0] >= 2.0, 1]), (0.05, 0.5, 0.95)).tolist(),
                "demand_q_p05_p50_p95": np.quantile(
                    op[op[:, 0] >= 2.0, 2], (0.05, 0.5, 0.95)).tolist(),
            } if np.any(op[:, 0] >= 2.0) else None),
        } if len(op) else None),
        "role": "post-fit practice transfer diagnostic only; not used to fit/select table",
    }


def _write_surface(path: Path, rates: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        stream.write(
            "# High-speed development yaw surface; fitted only from the training captures listed in its report.\n"
            "# Empirical response is held-out simulator yaw-rate magnitude in rad/s.\n")
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("speed_knot_mps", "turn_sign", "steering_rad",
                         "demand_q", "yaw_rate_abs_rps"))
        for speed_index, speed in enumerate(SPEED_KNOTS):
            for turn_sign in (-1, 1):
                for knot_index, steering in enumerate(STEERING_KNOTS):
                    q = speed * math.tan(steering)
                    response = float(rates[speed_index, knot_index])
                    writer.writerow((f"{speed:.8f}", turn_sign,
                                     f"{steering:.8f}", f"{q:.8f}",
                                     f"{response:.8f}"))


def _write_combined_surface(path: Path, highspeed_path: Path) -> None:
    """Keep all existing low-speed samples and append only new speed knots."""
    base_path = ROOT / "f1tenth_mpc/config/yaw_response_surface.csv"
    with base_path.open("r", encoding="utf-8", newline="") as stream:
        base_lines = [line for line in stream if line.strip()]
    with highspeed_path.open("r", encoding="utf-8", newline="") as stream:
        high_rows = list(csv.DictReader(
            line for line in stream if not line.lstrip().startswith("#")))
    base_rows = list(csv.DictReader(
        line for line in base_lines if not line.lstrip().startswith("#")))
    base_max = max(float(row["speed_knot_mps"]) for row in base_rows)
    appended = [row for row in high_rows
                if float(row["speed_knot_mps"]) > base_max + 1.0e-4]
    expected_append = 2 * len(STEERING_KNOTS) * (len(SPEED_KNOTS) - 1)
    if len(base_rows) != 3 * 2 * len(STEERING_KNOTS) or len(appended) != expected_append:
        raise ValueError("combined yaw surface source knot count is unexpected")
    with path.open("w", encoding="utf-8", newline="") as stream:
        stream.write(
            "# Combined development yaw surface: frozen existing low-speed rows plus fitted high-speed knots.\n"
            "# Existing rows are copied unchanged; added rows come from the fit report.\n")
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("speed_knot_mps", "turn_sign", "steering_rad",
                         "demand_q", "yaw_rate_abs_rps"))
        for row in base_rows + appended:
            writer.writerow(tuple(row[key] for key in (
                "speed_knot_mps", "turn_sign", "steering_rad", "demand_q",
                "yaw_rate_abs_rps")))


def _read_surface_table(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(
            line for line in stream if not line.lstrip().startswith("#")))
    rows_per_speed = 2 * len(STEERING_KNOTS)
    if len(rows) % rows_per_speed or not 3 <= len(rows) // rows_per_speed <= 5:
        raise ValueError(f"invalid yaw-surface row count: {path}")
    speed_count = len(rows) // rows_per_speed
    speeds = np.zeros(speed_count, dtype=np.float64)
    q_values = np.zeros((speed_count, 2, len(STEERING_KNOTS)), dtype=np.float64)
    rate_values = np.zeros_like(q_values)
    for speed_index in range(speed_count):
        for direction_index, expected_sign in enumerate((-1, 1)):
            start = (speed_index * 2 + direction_index) * len(STEERING_KNOTS)
            block = rows[start:start + len(STEERING_KNOTS)]
            knot_speed = float(block[0]["speed_knot_mps"])
            for knot_index, (expected_steering, row) in enumerate(
                    zip(STEERING_KNOTS, block)):
                row_speed = float(row["speed_knot_mps"])
                steering = float(row["steering_rad"])
                demand = float(row["demand_q"])
                response = float(row["yaw_rate_abs_rps"])
                if (int(row["turn_sign"]) != expected_sign
                        or abs(steering - expected_steering) > 1.0e-5
                        or abs(row_speed - knot_speed) > 1.0e-5
                        or not all(math.isfinite(value) and value > 0.0
                                   for value in (demand, response))):
                    raise ValueError(f"invalid yaw-surface rows: {path}")
                q_values[speed_index, direction_index, knot_index] = demand
                rate_values[speed_index, direction_index, knot_index] = response
            if direction_index == 0:
                speeds[speed_index] = knot_speed
            elif abs(speeds[speed_index] - knot_speed) > 1.0e-5:
                raise ValueError(f"directional speed knots do not match: {path}")
    if not np.all(np.diff(speeds) > 0.0):
        raise ValueError(f"yaw-surface speed knots are not increasing: {path}")
    return {"speed_mps": speeds, "q": q_values, "rate": rate_values}


def run(output_dir: Path,
        low_speed_blend_margin_mps: float = SPEED_BLEND_MARGIN_MPS,
        support_fadeout_mps: float = 0.0,
        support_fadein_mps: float = 0.0,
        ) -> dict[str, Any]:
    if (not math.isfinite(low_speed_blend_margin_mps)
            or low_speed_blend_margin_mps < 0.0
            or not math.isfinite(support_fadeout_mps)
            or support_fadeout_mps < 0.0
            or not math.isfinite(support_fadein_mps)
            or support_fadein_mps < 0.0
            or ((support_fadeout_mps > 0.0) != (support_fadein_mps > 0.0))):
        raise ValueError("yaw-surface blend and support margins are invalid")
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    training = []
    training_provenance = []
    for name, split, relative in TRAINING_SOURCES:
        data, run, path = _load_capture(name, split, relative)
        training.append((name, data))
        training_provenance.append({
            "name": name, "split": split, "run_id": run["run_id"],
            "path": str(path.relative_to(ROOT)), "sha256": _sha256(path),
        })
    rates, fit_details = _fit_surface(training)

    baseline_path = ROOT / "f1tenth_mpc/config/yaw_response_surface.csv"
    baseline_table = _read_surface_table(baseline_path)
    if (support_fadeout_mps > 0.0 and
            support_fadeout_mps + support_fadein_mps >=
            baseline_table["speed_mps"][2] - baseline_table["speed_mps"][0]):
        raise ValueError("support fades overlap the measured low/high-speed bands")

    table_path = output_dir / "yaw_response_surface_highspeed.csv"
    _write_surface(table_path, rates)
    combined_path = output_dir / "yaw_response_surface_combined.csv"
    _write_combined_surface(combined_path, table_path)
    candidate_table = _read_surface_table(combined_path)
    # Score only after the exact serialized runtime candidate has been read back.
    validation_rows = {}
    validation_provenance = []
    for name, split, relative in VALIDATION_SOURCES:
        data, run, path = _load_capture(name, split, relative)
        validation_rows[name] = _score_capture(
            name, data, baseline_table, candidate_table,
            low_speed_blend_margin_mps, support_fadeout_mps,
            support_fadein_mps)
        validation_provenance.append({
            "name": name, "split": split, "run_id": run["run_id"],
            "path": str(path.relative_to(ROOT)), "sha256": _sha256(path),
        })
    practice_diagnostics = {}
    for name, relative in PRACTICE_DIAGNOSTIC_SOURCES:
        path = (ROOT / relative).resolve()
        practice_diagnostics[name] = _score_practice_bag(
            name, path, baseline_table, candidate_table,
            low_speed_blend_margin_mps, support_fadeout_mps,
            support_fadein_mps)
    low_speed_validation = [
        row["low_speed_high_demand"] for row in validation_rows.values()
        if row["low_speed_high_demand"]["candidate"]["samples"] > 0
    ]
    report = {
        "study": "training-only high-speed yaw surface with untouched whole-capture validation",
        "model": {
            "type": "piecewise-PCHIP-in-demand-q, linear-between-speed-knots",
            "speed_knots_mps": list(SPEED_KNOTS),
            "steering_knots_rad": list(STEERING_KNOTS),
            "q_blend_start": BLEND_Q_START,
            "q_blend_end": BLEND_Q_END,
            "speed_blend_margin_mps": SPEED_BLEND_MARGIN_MPS,
            "low_speed_blend_margin_mps": low_speed_blend_margin_mps,
            "low_speed_support_fadeout_mps": support_fadeout_mps,
            "high_speed_support_fadein_mps": support_fadein_mps,
            "low_speed_support_ends_at_mps": float(
                candidate_table["speed_mps"][0]),
            "high_speed_support_starts_at_mps": float(
                candidate_table["speed_mps"][2]),
            "yaw_time_constant_s": YAW_TAU_S,
            "candidate_surface_sha256": _sha256(table_path),
            "surface_csv": str(table_path.relative_to(ROOT)),
            "combined_surface_sha256": _sha256(combined_path),
            "combined_surface_csv": str(combined_path.relative_to(ROOT)),
            "baseline_surface_sha256": _sha256(baseline_path),
            "baseline_surface_csv": str(baseline_path.relative_to(ROOT)),
        },
        "training_sources": training_provenance,
        "validation_sources": validation_provenance,
        "fit_details": fit_details,
        "validation_by_whole_run": validation_rows,
        "practice_transfer_diagnostic_only": practice_diagnostics,
        "high_speed_high_demand_run_cluster_bootstrap": _run_bootstrap(validation_rows),
        "validation_gate": {
            "passes_high_speed_transfer": all(
                row["high_speed_high_demand"]["candidate"]["rmse_rps"]
                < row["high_speed_high_demand"]["baseline_surface"]["rmse_rps"]
                for row in validation_rows.values()),
            "passes_full_run_non_regression": all(
                row["all_valid_candidate"]["rmse_rps"]
                <= row["all_valid_baseline_surface"]["rmse_rps"] + 0.02
                for row in validation_rows.values()),
            "passes_low_speed_high_demand_transfer": (
                len(low_speed_validation) >= 3 and all(
                    row["candidate"]["rmse_rps"] <
                    row["baseline_surface"]["rmse_rps"]
                    for row in low_speed_validation)),
            "passes_unsupported_middle_speed_legacy_parity": all(
                row["unsupported_middle_speed_high_demand"][
                    "candidate_with_support_gate"]["samples"] ==
                row["unsupported_middle_speed_high_demand"][
                    "legacy_no_surface"]["samples"]
                and row["unsupported_middle_speed_high_demand"][
                    "candidate_with_support_gate"]["rmse_rps"] ==
                row["unsupported_middle_speed_high_demand"][
                    "legacy_no_surface"]["rmse_rps"]
                for row in validation_rows.values()),
            "candidate_is_not_production_integrated": True,
            "validated_speed_interval_mps": [
                SPEED_KNOTS[0] - SPEED_BLEND_MARGIN_MPS, SPEED_KNOTS[-1]],
            "low_speed_response_interval_mps": [
                baseline_table["speed_mps"][0] - low_speed_blend_margin_mps,
                baseline_table["speed_mps"][0]],
            "unsupported_middle_speed_interval_mps": [
                baseline_table["speed_mps"][0] + support_fadeout_mps,
                baseline_table["speed_mps"][2] - support_fadein_mps],
            "unvalidated_above_speed_mps": SPEED_KNOTS[-1],
            "unvalidated_above_steering_rad": 0.42,
        },
        "interpretation": (
            "A passing one-step response score is necessary but not sufficient for closed-loop use. "
            "This report does not certify the MPC rollout, optimizer convergence, or lap feasibility."),
    }
    report_path = output_dir / "fit_and_validation_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=(
        ROOT / "live_runs/raceline_candidates/highspeed_yaw_surface_combined_r02_20261006"))
    parser.add_argument("--low-speed-blend-margin-mps", type=float,
                        default=SPEED_BLEND_MARGIN_MPS,
                        help="lower-speed fade width; upper-speed fade remains fixed")
    parser.add_argument("--support-fadeout-mps", type=float, default=0.0,
                        help="fade surface correction out above the measured low-speed knot")
    parser.add_argument("--support-fadein-mps", type=float, default=0.0,
                        help="fade surface correction in before the measured high-speed knot")
    args = parser.parse_args()
    report = run(args.output_dir, args.low_speed_blend_margin_mps,
                 args.support_fadeout_mps, args.support_fadein_mps)
    print(json.dumps({
        "surface_csv": report["model"]["combined_surface_csv"],
        "validation_gate": report["validation_gate"],
        "run_cluster_bootstrap": report[
            "high_speed_high_demand_run_cluster_bootstrap"],
    }, indent=2))
    for name, row in report["validation_by_whole_run"].items():
        old = row["high_speed_high_demand"]["baseline_surface"]["rmse_rps"]
        new = row["high_speed_high_demand"]["candidate"]["rmse_rps"]
        print(f"{name}: high-speed/high-demand yaw RMSE {old:.4f} -> {new:.4f} rad/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
