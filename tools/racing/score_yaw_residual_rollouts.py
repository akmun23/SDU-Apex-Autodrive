#!/usr/bin/env python3
"""Recursively score an offline yaw residual against the production C plant.

Each rollout starts from one measured P0 state. Thereafter it consumes only the
recorded MPC control-rate sequence and map curvature at its own predicted
progress. Future simulator states are used only as scoring labels.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import ctypes
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from tools.racing.offline.controller.production_mpc import ProductionMpc


ROOT = Path(__file__).resolve().parents[2]
DT_S = 0.025
HORIZONS = {0.1: 4, 0.25: 10, 0.5: 20, 0.75: 30}
CHANNELS = ("u_mps", "v_mps", "r_radps", "e_y_m", "e_psi_rad")


class ModelState(ctypes.Structure):
    _fields_ = [(name, ctypes.c_float) for name in (
        "e_y", "e_psi", "u", "v", "r", "target_speed",
        "steering_command", "delayed_steering_command_1",
        "delayed_steering_command_2", "actual_steering_angle")]


class ModelControl(ctypes.Structure):
    _fields_ = [("steering_rate", ctypes.c_float),
                ("target_speed_rate", ctypes.c_float)]


class StageResult(ctypes.Structure):
    _fields_ = [("next", ModelState), ("delta_s_m", ctypes.c_float),
                ("body_accel_mps2", ctypes.c_float),
                ("branch_flags", ctypes.c_uint), ("valid", ctypes.c_int)]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def value(row: dict[str, str], key: str) -> float | None:
    raw = row.get(key, "")
    if raw in (None, ""):
        return None
    result = float(raw)
    return result if math.isfinite(result) else None


def clone_state(state: ModelState) -> ModelState:
    return ModelState(*(getattr(state, name) for name, _ in state._fields_))


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def interpolate_truth(rows: list[dict[str, str]], times: list[float], t: float) -> dict[str, float] | None:
    upper = bisect.bisect_left(times, t)
    if upper == 0 or upper >= len(rows):
        return None
    first, second = rows[upper - 1], rows[upper]
    t0, t1 = times[upper - 1], times[upper]
    if t - t0 > 0.040 or t1 - t > 0.040 or t1 <= t0:
        return None
    alpha = (t - t0) / (t1 - t0)
    result: dict[str, float] = {}
    for target, source in (
        ("u_mps", "truth_speed_mps"),
        ("v_mps", "truth_lateral_speed_mps"),
        ("r_radps", "yaw_rate_radps"),
        ("e_y_m", "tracking_error_m"),
        ("e_psi_rad", "truth_heading_error_rad"),
    ):
        a, b = value(first, source), value(second, source)
        if a is None or b is None:
            return None
        delta = wrap_angle(b - a) if target == "e_psi_rad" else b - a
        result[target] = wrap_angle(a + alpha * delta) if target == "e_psi_rad" else a + alpha * delta
    return result


def interpolate_curvature(progress: float, path_s: np.ndarray,
                           curvature: np.ndarray, lap_length: float) -> float:
    return float(np.interp(progress % lap_length, path_s, curvature))


def residual_rate(model: dict[str, Any], state: ModelState,
                  control: ModelControl, curvature: float,
                  modeled_longitudinal_accel: float) -> float:
    delta = float(state.actual_steering_angle)
    q_delta = float(control.steering_rate)
    q_v = float(control.target_speed_rate)
    u, r, v = float(state.u), float(state.r), float(state.v)
    features = np.asarray((
        u,
        r,
        delta,
        q_delta,
        v,
        curvature,
        modeled_longitudinal_accel,
        q_v,
        u * delta * abs(delta),
        delta * abs(q_delta),
        v * abs(delta),
        abs(delta) * modeled_longitudinal_accel,
        delta * abs(q_v),
    ), dtype=np.float64)
    z = (features - np.asarray(model["feature_mean"])) / np.asarray(model["feature_scale"])
    coeff = np.asarray(model["coefficients_with_intercept"])
    correction = float(coeff[0] + np.dot(z, coeff[1:]))
    limit = float(model["correction_clip_radps2"])
    correction = max(-limit, min(limit, correction))

    gate = model.get("support_gate")
    if gate is None:
        return correction

    def smoothstep(value: float, start: float, end: float) -> float:
        if not end > start:
            raise ValueError("yaw-residual support gate bounds must be increasing")
        x = max(0.0, min(1.0, (value - start) / (end - start)))
        return x * x * (3.0 - 2.0 * x)

    if gate.get("kind") == "smoothstep_speed_abs_steering_v1":
        speed_weight = smoothstep(
            float(state.u), gate["speed_zero_mps"], gate["speed_full_mps"])
    elif gate.get("kind") == "smoothstep_target_tracking_steering_v1":
        target_weight = smoothstep(
            float(state.target_speed), gate["target_speed_zero_mps"],
            gate["target_speed_full_mps"])
        speed_deficit = float(state.target_speed) - float(state.u)
        tracking_weight = 1.0 - smoothstep(
            speed_deficit, gate["speed_deficit_full_mps"],
            gate["speed_deficit_zero_mps"])
        speed_weight = target_weight * tracking_weight
    else:
        raise ValueError(f"unsupported yaw-residual support gate: {gate.get('kind')}")
    steering_weight = smoothstep(
        abs(delta), gate["abs_steering_zero_rad"],
        gate["abs_steering_full_rad"])
    return correction * speed_weight * steering_weight


def apply_yaw_correction(state: ModelState, stage: StageResult, control: ModelControl,
                         dt: float, curvature: float, correction_rate: float) -> tuple[ModelState, float]:
    """Re-integrate the production Frenet midpoint equations with corrected r."""
    next_state = clone_state(stage.next)
    next_state.r = float(stage.next.r) + dt * correction_rate
    u0 = max(0.0, min(16.0, float(state.u)))
    u_mid = 0.5 * (u0 + float(next_state.u))
    r_mid = 0.5 * (float(state.r) + float(next_state.r))

    den0 = 1.0 - curvature * float(state.e_y)
    if abs(den0) < 0.05:
        raise ArithmeticError("candidate rollout reached the production Frenet singularity")
    s_dot0 = (u0 * math.cos(float(state.e_psi)) - float(state.v) * math.sin(float(state.e_psi))) / den0
    ey_dot0 = u0 * math.sin(float(state.e_psi)) + float(state.v) * math.cos(float(state.e_psi))
    epsi_dot0 = float(state.r) - curvature * s_dot0
    ey_mid = float(state.e_y) + 0.5 * dt * ey_dot0
    epsi_mid = float(state.e_psi) + 0.5 * dt * epsi_dot0
    den_mid = 1.0 - curvature * ey_mid
    if abs(den_mid) < 0.05:
        raise ArithmeticError("candidate rollout reached the production midpoint singularity")
    s_dot_mid = (u_mid * math.cos(epsi_mid) - float(state.v) * math.sin(epsi_mid)) / den_mid
    ey_dot_mid = u_mid * math.sin(epsi_mid) + float(state.v) * math.cos(epsi_mid)
    epsi_dot_mid = r_mid - curvature * s_dot_mid
    next_state.e_y = float(state.e_y) + dt * ey_dot_mid
    next_state.e_psi = wrap_angle(float(state.e_psi) + dt * epsi_dot_mid)
    return next_state, dt * s_dot_mid


def metric(values: list[float]) -> dict[str, float | int | None]:
    a = np.asarray(values, dtype=np.float64)
    if not len(a):
        return {"count": 0, "rmse": None, "mae": None, "p95_abs": None}
    return {
        "count": int(len(a)),
        "rmse": float(np.sqrt(np.mean(a * a))),
        "mae": float(np.mean(np.abs(a))),
        "p95_abs": float(np.quantile(np.abs(a), 0.95)),
    }


def run_replay(report_dir: Path, library_path: Path, config_path: Path,
               trajectory_path: Path, model: dict[str, Any],
               residual_gain: float,
               minimum_lap_count: int,
               score_end_time_s: float | None,
               output_rows: list[dict[str, Any]]) -> dict[str, Any]:
    tracking = read_csv(report_dir / "tracking_error.csv")
    controls = read_csv(report_dir / "controller_saturation.csv")
    if score_end_time_s is not None:
        # Receipt time is only a censoring boundary; each retained row is one
        # fixed 25 ms model sample despite transport jitter in the bag.
        tracking = [row for row in tracking if float(row["time_s"]) < score_end_time_s]
        controls = [row for row in controls if float(row["time_s"]) < score_end_time_s]
    truth_times = [float(row["time_s"]) for row in tracking]
    control_times = [float(row["time_s"]) for row in controls]
    trajectory = np.loadtxt(trajectory_path, delimiter=",", comments="#", dtype=np.float64)
    path_s, path_curvature = trajectory[:, 0], trajectory[:, 4]

    library = ctypes.CDLL(str(library_path.resolve()))
    library.mpc_vehicle_model_step.argtypes = [
        ctypes.POINTER(ModelState), ctypes.POINTER(ModelControl), ctypes.c_float, ctypes.c_float]
    library.mpc_vehicle_model_step.restype = StageResult

    baseline_errors: dict[tuple[float, str, str], list[float]] = defaultdict(list)
    candidate_errors: dict[tuple[float, str, str], list[float]] = defaultdict(list)
    accepted_windows = 0
    skipped_windows = 0
    parity_max = 0.0
    start_status_counts: dict[str, int] = defaultdict(int)

    # This object initializes the compiled C model from the exact P0 YAML and
    # trajectory. The solver is never called; only its authoritative model step.
    with ProductionMpc(library_path, config_path, trajectory_path) as plant:
        lap_length = plant.lap_length_m
        projection_segment = (2**64 - 1)
        for start_index, start_control in enumerate(controls):
            start_t = float(start_control["time_s"])
            status = start_control.get("status", "")
            start_status_counts[status] += 1
            nearest = bisect.bisect_left(truth_times, start_t)
            candidates = [idx for idx in (nearest - 1, nearest) if 0 <= idx < len(tracking)]
            if not candidates:
                continue
            truth_index = min(candidates, key=lambda idx: abs(truth_times[idx] - start_t))
            current_truth = tracking[truth_index]
            lap_value = current_truth.get("lap_count", "")
            if (not lap_value.lstrip("-").isdigit()
                    or not minimum_lap_count <= int(lap_value) <= 11):
                continue
            needed = max(HORIZONS.values())
            if (score_end_time_s is not None
                    and start_t + needed * DT_S > score_end_time_s):
                skipped_windows += 1
                continue
            if start_index + needed >= len(controls):
                skipped_windows += 1
                continue
            if abs(truth_times[truth_index] - start_t) > 0.012:
                skipped_windows += 1
                continue
            command_window = controls[start_index:start_index + needed]
            if any(
                value(row, "first_control_steering_rate_radps") is None
                or value(row, "first_control_target_speed_rate_mps2") is None
                for row in command_window
            ):
                skipped_windows += 1
                continue

            required_state_fields = (
                "state_target_speed_mps", "state_steering_command_rad",
                "state_delayed_steering_command_1_rad", "state_delayed_steering_command_2_rad",
            )
            if any(value(start_control, field) is None for field in required_state_fields):
                skipped_windows += 1
                continue
            initial_values = (
                value(current_truth, "tracking_error_m"),
                value(current_truth, "truth_heading_error_rad"),
                value(current_truth, "truth_speed_mps"),
                value(current_truth, "truth_lateral_speed_mps"),
                value(current_truth, "yaw_rate_radps"),
                value(start_control, "state_target_speed_mps"),
                value(start_control, "state_steering_command_rad"),
                value(start_control, "state_delayed_steering_command_1_rad"),
                value(start_control, "state_delayed_steering_command_2_rad"),
                value(current_truth, "steering_feedback_rad"),
                value(current_truth, "s_m"),
            )
            if any(item is None for item in initial_values):
                skipped_windows += 1
                continue
            (ey, epsi, u, v, r, target, steer_command, delayed1, delayed2,
             actual_steer, progress) = map(float, initial_values)
            baseline_state = ModelState(ey, epsi, u, v, r, target, steer_command,
                                        delayed1, delayed2, actual_steer)
            candidate_state = clone_state(baseline_state)
            baseline_progress = progress
            candidate_progress = progress
            valid = True
            initial_steer_band = "high_steer" if abs(actual_steer) >= 0.20 else "ordinary_steer"
            staged: list[tuple[float, dict[str, float], dict[str, float], dict[str, float]]] = []
            for step in range(needed):
                control_row = command_window[step]
                control = ModelControl(
                    float(control_row["first_control_steering_rate_radps"]),
                    float(control_row["first_control_target_speed_rate_mps2"]),
                )
                base_curvature = interpolate_curvature(
                    baseline_progress, path_s, path_curvature, lap_length)
                cand_curvature = interpolate_curvature(
                    candidate_progress, path_s, path_curvature, lap_length)
                base_stage = library.mpc_vehicle_model_step(
                    ctypes.byref(baseline_state), ctypes.byref(control),
                    ctypes.c_float(DT_S), ctypes.c_float(base_curvature))
                cand_stage = library.mpc_vehicle_model_step(
                    ctypes.byref(candidate_state), ctypes.byref(control),
                    ctypes.c_float(DT_S), ctypes.c_float(cand_curvature))
                if not base_stage.valid or not cand_stage.valid:
                    valid = False
                    break

                if step == 0:
                    zero_state, zero_progress = apply_yaw_correction(
                        baseline_state, base_stage, control, DT_S, base_curvature, 0.0)
                    parity_values = (
                        abs(float(zero_state.r) - float(base_stage.next.r)),
                        abs(float(zero_state.e_y) - float(base_stage.next.e_y)),
                        abs(wrap_angle(float(zero_state.e_psi) - float(base_stage.next.e_psi))),
                        abs(zero_progress - float(base_stage.delta_s_m)),
                    )
                    parity_max = max(parity_max, *parity_values)

                rate_correction = residual_gain * residual_rate(
                    model, candidate_state, control, cand_curvature,
                    float(cand_stage.body_accel_mps2))
                candidate_next, cand_ds = apply_yaw_correction(
                    candidate_state, cand_stage, control, DT_S, cand_curvature, rate_correction)
                baseline_state = clone_state(base_stage.next)
                candidate_state = candidate_next
                baseline_progress += float(base_stage.delta_s_m)
                candidate_progress += cand_ds

                horizon = next((seconds for seconds, count in HORIZONS.items() if step + 1 == count), None)
                if horizon is None:
                    continue
                target_index = truth_index + step + 1
                if target_index >= len(tracking):
                    valid = False
                    break
                target_row = tracking[target_index]
                target = {
                    "u_mps": value(target_row, "truth_speed_mps"),
                    "v_mps": value(target_row, "truth_lateral_speed_mps"),
                    "r_radps": value(target_row, "yaw_rate_radps"),
                    "e_y_m": value(target_row, "tracking_error_m"),
                    "e_psi_rad": value(target_row, "truth_heading_error_rad"),
                }
                if any(item is None for item in target.values()):
                    valid = False
                    break
                base_values = {
                    "u_mps": float(baseline_state.u),
                    "v_mps": float(baseline_state.v),
                    "r_radps": float(baseline_state.r),
                    "e_y_m": float(baseline_state.e_y),
                    "e_psi_rad": float(baseline_state.e_psi),
                }
                cand_values = {
                    "u_mps": float(candidate_state.u),
                    "v_mps": float(candidate_state.v),
                    "r_radps": float(candidate_state.r),
                    "e_y_m": float(candidate_state.e_y),
                    "e_psi_rad": float(candidate_state.e_psi),
                }
                staged.append((horizon, base_values, cand_values, target))

            if not valid or len(staged) != len(HORIZONS):
                skipped_windows += 1
                continue
            accepted_windows += 1
            for horizon, base_values, cand_values, target_state in staged:
                for channel in CHANNELS:
                    base_error = base_values[channel] - target_state[channel]
                    cand_error = cand_values[channel] - target_state[channel]
                    if channel == "e_psi_rad":
                        base_error, cand_error = wrap_angle(base_error), wrap_angle(cand_error)
                    key = (horizon, channel, initial_steer_band)
                    baseline_errors[key].append(base_error)
                    candidate_errors[key].append(cand_error)
                    output_rows.append({
                        "run_id": current_truth["run_id"],
                        "start_lap_count": int(lap_value),
                        "start_time_s": start_t,
                        "initial_speed_mps": u,
                        "initial_abs_steering_rad": abs(actual_steer),
                        "steer_band": initial_steer_band,
                        "horizon_s": horizon,
                        "channel": channel,
                        "production_error": base_error,
                        "candidate_error": cand_error,
                    })

    summary: dict[str, Any] = {
        "report_run_id": tracking[0].get("run_id", report_dir.name) if tracking else report_dir.name,
        "accepted_recursive_windows": accepted_windows,
        "skipped_recursive_windows": skipped_windows,
        "controller_status_counts": dict(start_status_counts),
        "zero_correction_equation_parity_max_abs": parity_max,
        "by_horizon_and_channel": {},
        "by_horizon_channel_and_steering_regime": {},
    }
    for horizon in HORIZONS:
        for channel in CHANNELS:
            key = (horizon, channel, "all")
            base = [v for band in ("high_steer", "ordinary_steer") for v in baseline_errors[(horizon, channel, band)]]
            candidate = [v for band in ("high_steer", "ordinary_steer") for v in candidate_errors[(horizon, channel, band)]]
            summary["by_horizon_and_channel"][f"{horizon:g}s:{channel}"] = {
                "production": metric(base),
                "candidate": metric(candidate),
            }
            for band in ("high_steer", "ordinary_steer"):
                b = baseline_errors[(horizon, channel, band)]
                c = candidate_errors[(horizon, channel, band)]
                if b:
                    summary["by_horizon_channel_and_steering_regime"][f"{horizon:g}s:{channel}:{band}"] = {
                        "production": metric(b), "candidate": metric(c),
                    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, action="append", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--residual-gain", type=float, default=1.0,
                        help="scale the fitted correction from 0 (production) to 1 (full candidate)")
    parser.add_argument("--minimum-lap-count", type=int, default=2,
                        help="lowest recorded lap counter to score (default: 2)")
    parser.add_argument("--score-end-time-s", type=float,
                        help="exclude rollout windows that would extend past this report-relative time")
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not 0.0 <= args.residual_gain <= 1.0:
        parser.error("--residual-gain must be between 0 and 1")
    if args.minimum_lap_count < -1:
        parser.error("--minimum-lap-count must be at least -1")
    if args.score_end_time_s is not None and args.score_end_time_s <= 0.0:
        parser.error("--score-end-time-s must be positive")
    report_dirs = [p if p.is_absolute() else ROOT / p for p in args.report_dir]
    model_path = args.model if args.model.is_absolute() else ROOT / args.model
    library_path = args.library if args.library.is_absolute() else ROOT / args.library
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    trajectory_path = args.trajectory if args.trajectory.is_absolute() else ROOT / args.trajectory
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error(f"output directory is not empty: {output_dir}")
    model = json.loads(model_path.read_text(encoding="utf-8"))
    if model.get("status") != "offline_one_step_candidate_only":
        parser.error("yaw residual file is not the expected offline candidate")
    all_rows: list[dict[str, Any]] = []
    reports = [run_replay(path, library_path, config_path, trajectory_path, model,
                          args.residual_gain, args.minimum_lap_count,
                          args.score_end_time_s, all_rows) for path in report_dirs]
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(all_rows[0]) if all_rows else ["run_id", "start_lap_count", "start_time_s", "horizon_s", "channel", "production_error", "candidate_error"]
    with (output_dir / "recursive_rollout_errors.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    summary = {
        "schema_version": 1,
        "model_id": model["model_id"],
        "model_status": model["status"],
        "residual_gain": args.residual_gain,
        "minimum_lap_count": args.minimum_lap_count,
        "score_end_time_s": args.score_end_time_s,
        "support_gate": model.get("support_gate"),
        "promotion_decision": "not_promoted_pending_independent_regime_and_parity_gates",
        "initial_state_source": "simulator truth for u,v,r,e_y,e_psi and measured steering feedback; recorded causal MPC command-history states",
        "future_inputs": "recorded controller command-rate rows in source order at the fixed 25 ms model step; curvature sampled at each model's own predicted track progress",
        "sample_time_policy": "one 25 ms step per recorded row; receive-time jitter is used only for collision censoring and causal initial-state matching",
        "future_truth_or_sensors_used_as_inputs": False,
        "horizons_s": list(HORIZONS),
        "per_run": reports,
        "outputs": ["recursive_rollout_errors.csv"],
        "limitations": [
            "P0 r01 trained the residual and r02 is the only independent practice holdout.",
            "This is teacher-initialized open-loop replay with recorded commands, not closed-loop MPC counterfactual evaluation.",
            "The two P0 captures do not independently validate open-plane high-steering transfer.",
        ],
    }
    (output_dir / "recursive_rollout_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "model_status": summary["model_status"],
        "residual_gain": summary["residual_gain"],
        "per_run": [{
            "run_id": row["report_run_id"],
            "accepted_recursive_windows": row["accepted_recursive_windows"],
            "skipped_recursive_windows": row["skipped_recursive_windows"],
            "zero_correction_equation_parity_max_abs": row["zero_correction_equation_parity_max_abs"],
        } for row in reports],
        "held_out_metrics": next((row["by_horizon_and_channel"] for row in reports if row["report_run_id"].endswith("r02")), {}),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
