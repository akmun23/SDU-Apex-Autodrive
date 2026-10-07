#!/usr/bin/env python3
"""Recursively replay P0 practice commands with a learned lateral-speed law.

Simulator truth is used only to initialize each replay and score predictions.
Logged MPC rates are treated as the exogenous command sequence; after the
initial state, no recorded future sensor or vehicle state is fed to either
plant. The 25 ms transition is the shared production MPC stage, with the
research candidate replacing only its held-v update.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from tools.racing.verify_optimizer_mpc_model_parity import python_step


ROOT = Path(__file__).resolve().parents[2]
DT_S = 0.025
SCORED_LAPS = set(range(2, 12))
DEFAULT_R01 = ROOT / "live_runs/practice_9g_parent_reproduced_20261006_r01"
DEFAULT_R02 = ROOT / "live_runs/practice_9g_parent_reproduced_20261006_r02"
DEFAULT_MODEL_REPORT = ROOT / (
    "live_runs/racing_model_diagnostics_20261006/"
    "lateral_velocity_increment_p0_r01_to_r02.json")
DEFAULT_TRAJECTORY = ROOT / (
    "live_runs/raceline_candidates/"
    "practice_9g_runtime_matched_wallmargin010_20261005/output/"
    "autodrive_mintime_raceline.csv")
MODEL_PARAMETERS = ROOT / "config/racing/racing_vehicle_model.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_row(row: dict[str, str], names: tuple[str, ...]
               ) -> dict[str, float] | None:
    try:
        values = {name: float(row[name]) for name in names}
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in values.values()):
        return None
    return values


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def read_trajectory(path: Path) -> dict[str, Any]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                rows.append([float(value) for value in stripped.split(",")])
    values = np.asarray(rows, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 9 or len(values) < 20:
        raise ValueError(f"unsupported raceline CSV: {path}")
    if not np.isfinite(values).all() or np.any(np.diff(values[:, 0]) <= 0.0):
        raise ValueError("raceline has invalid or non-monotonic samples")
    close = float(np.linalg.norm(values[0, 1:3] - values[-1, 1:3]))
    length = float(values[-1, 0] + close)
    if close <= 0.0:
        raise ValueError("raceline has no valid closing segment")
    return {"rows": values, "s": values[:, 0], "x": values[:, 1],
            "y": values[:, 2], "heading": values[:, 3],
            "curvature": values[:, 4], "length": length}


def periodic_linear(query: float, grid: np.ndarray, values: np.ndarray,
                    length: float) -> float:
    x = np.r_[grid, length]
    y = np.r_[values, values[0]]
    return float(np.interp(query % length, x, y))


def periodic_heading(query: float, path: dict[str, Any]) -> float:
    c = periodic_linear(query, path["s"], np.cos(path["heading"]),
                        path["length"])
    s = periodic_linear(query, path["s"], np.sin(path["heading"]),
                        path["length"])
    return math.atan2(s, c)


def wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def pose_xy(state: np.ndarray, progress: float,
            path: dict[str, Any]) -> tuple[float, float, float]:
    psi_ref = periodic_heading(progress, path)
    cx = periodic_linear(progress, path["s"], path["x"], path["length"])
    cy = periodic_linear(progress, path["s"], path["y"], path["length"])
    ey = float(state[0])
    return (cx - ey * math.sin(psi_ref),
            cy + ey * math.cos(psi_ref),
            wrap(psi_ref + float(state[1])))


def summarize(errors: dict[str, list[float]]) -> dict[str, Any]:
    result = {}
    for channel, values in errors.items():
        array = np.asarray(values, dtype=np.float64)
        if len(array) == 0:
            result[channel] = None
            continue
        absolute = np.abs(array)
        result[channel] = {
            "n": int(len(array)),
            "bias": float(array.mean()),
            "rmse": float(np.sqrt(np.mean(array * array))),
            "mae": float(absolute.mean()),
            "p95_abs": float(np.quantile(absolute, 0.95)),
            "max_abs": float(absolute.max()),
        }
    return result


def paired_lap_bootstrap(hold: dict[str, dict[str, Any]],
                         candidate: dict[str, dict[str, Any]],
                         channel: str) -> dict[str, Any] | None:
    laps = sorted(set(hold) & set(candidate))
    gains = []
    for lap in laps:
        h = hold[lap].get(channel)
        c = candidate[lap].get(channel)
        if h and c:
            gains.append(float(h["rmse"]) - float(c["rmse"]))
    if len(gains) < 2:
        return None
    values = np.asarray(gains, dtype=np.float64)
    rng = np.random.default_rng(20261006)
    boot = rng.choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
    return {
        "metric": "held_lateral_velocity_rmse_minus_candidate_rmse",
        "mean_gain": float(values.mean()),
        "within_run_lap_bootstrap_95pct_ci": [
            float(np.quantile(boot, 0.025)),
            float(np.quantile(boot, 0.975)),
        ],
        "laps": len(gains),
        "independent_validation_runs": 1,
    }


def load_inputs(run_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]],
                                       list[int]]:
    analysis = run_dir / "analysis/race_report_yaw_residual_input"
    truth_raw = read_csv(analysis / "tracking_error.csv")
    control_raw = read_csv(analysis / "controller_saturation.csv")
    truth: list[dict[str, Any]] = []
    for row in truth_raw:
        try:
            lap = int(row["lap_number"])
            values = finite_row(row, (
                "time_s", "s_m", "x_m", "y_m", "truth_speed_mps",
                "truth_lateral_speed_mps", "yaw_rate_radps", "truth_yaw_rad",
                "truth_heading_error_rad", "truth_lateral_offset_m",
                "steering_feedback_rad"))
        except (KeyError, TypeError, ValueError):
            continue
        if values is None:
            continue
        truth.append({**values, "lap_number": lap})
    truth.sort(key=lambda row: row["time_s"])

    controls: list[dict[str, Any]] = []
    for row in control_raw:
        try:
            lap = int(row["lap_number"])
            if lap not in SCORED_LAPS:
                continue
            values = finite_row(row, (
                "time_s", "first_control_steering_rate_radps",
                "first_control_target_speed_rate_mps2",
                "state_target_speed_mps", "state_steering_command_rad",
                "state_delayed_steering_command_1_rad",
                "state_delayed_steering_command_2_rad"))
        except (KeyError, TypeError, ValueError):
            continue
        if values is None:
            raise ValueError(f"missing controller state/action in scored lap {lap}")
        controls.append({**values, "lap_number": lap})
    controls.sort(key=lambda row: row["time_s"])
    if len(controls) < 1000:
        raise ValueError(f"too few scored controller samples in {run_dir}")

    truth_times = [row["time_s"] for row in truth]
    matched_indices: list[int] = []
    offsets = []
    for control in controls:
        index = bisect.bisect_left(truth_times, control["time_s"])
        candidates = [i for i in (index - 1, index) if 0 <= i < len(truth)]
        match = min(candidates, key=lambda i: abs(
            truth_times[i] - control["time_s"]))
        offset = truth_times[match] - control["time_s"]
        if abs(offset) > 0.010:
            raise ValueError(
                f"truth/control phase mismatch {offset:.6f}s at "
                f"{control['time_s']:.6f}s")
        matched_indices.append(match)
        offsets.append(offset)
    if any(b <= a for a, b in zip(matched_indices, matched_indices[1:])):
        raise ValueError("truth/control matching is not one-to-one and monotonic")
    return truth, controls, matched_indices


def initial_state(truth: dict[str, Any], control: dict[str, Any]
                  ) -> tuple[np.ndarray, float]:
    state = np.asarray((
        truth["truth_lateral_offset_m"],
        truth["truth_heading_error_rad"],
        truth["truth_speed_mps"],
        truth["truth_lateral_speed_mps"],
        truth["yaw_rate_radps"],
        control["state_target_speed_mps"],
        control["state_steering_command_rad"],
        control["state_delayed_steering_command_1_rad"],
        control["state_delayed_steering_command_2_rad"],
        truth["steering_feedback_rad"],
    ), dtype=np.float64)
    return state, float(truth["s_m"])


def make_errors() -> dict[str, list[float]]:
    return {name: [] for name in (
        "position_xy_m", "progress_s_m", "yaw_angle_rad", "lateral_offset_ey_m",
        "heading_error_epsi_rad", "forward_speed_u_mps", "lateral_speed_v_mps",
        "yaw_rate_r_radps", "physical_steering_rad")}


def compare_state(state: np.ndarray, progress: float, actual: dict[str, Any],
                  path: dict[str, Any], errors: dict[str, list[float]]) -> dict[str, float]:
    px, py, pyaw = pose_xy(state, progress, path)
    values = {
        "position_xy_m": math.hypot(px - actual["x_m"], py - actual["y_m"]),
    }
    length = float(path["length"])
    ds = (progress - actual["s_m"] + 0.5 * length) % length - 0.5 * length
    values.update({
        "progress_s_m": ds,
        "yaw_angle_rad": wrap(pyaw - actual["truth_yaw_rad"]),
        "lateral_offset_ey_m": (
            float(state[0]) - actual["truth_lateral_offset_m"]),
        "heading_error_epsi_rad": wrap(
            float(state[1]) - actual["truth_heading_error_rad"]),
        "forward_speed_u_mps": float(state[2]) - actual["truth_speed_mps"],
        "lateral_speed_v_mps": (
            float(state[3]) - actual["truth_lateral_speed_mps"]),
        "yaw_rate_r_radps": float(state[4]) - actual["yaw_rate_radps"],
        "physical_steering_rad": (
            float(state[9]) - actual["steering_feedback_rad"]),
    })
    for channel, error in values.items():
        errors[channel].append(error)
    return values


def feature_vector(model: dict[str, Any], state: np.ndarray,
                   steering_slew: float) -> np.ndarray:
    _, _, u, v, r, _, _, _, _, delta = map(float, state)
    values = {
        "v_mps": v, "u_mps": u, "r_radps": r,
        "steering_feedback_rad": delta,
        "steering_slew_radps": steering_slew,
        "u_times_r": u * r, "u_times_delta": u * delta,
        "r_times_delta": r * delta,
    }
    return np.asarray([values[name] for name in model["feature_names"]],
                      dtype=np.float64)


def replay(control_indices: list[int], truth: list[dict[str, Any]],
           controls: list[dict[str, Any]], matches: list[int],
           path: dict[str, Any], vehicle_model: dict[str, Any],
           learned_model: dict[str, Any]) -> dict[str, Any]:
    if len(control_indices) < 2:
        return {"status": "insufficient_steps", "steps": 0, "errors": make_errors()}
    first = control_indices[0]
    state, progress = initial_state(truth[matches[first]], controls[first])
    if matches[first] == 0:
        steering_slew = 0.0
    else:
        prior = truth[matches[first] - 1]
        steering_slew = (
            state[9] - prior["steering_feedback_rad"]) / DT_S

    errors = make_errors()
    by_lap: dict[str, dict[str, list[float]]] = {}
    checkpoint_steps = {1, 5, 10, 20, 40, 80, 120, 160, 175, 200, 220}
    checkpoints = []
    initial_px, initial_py, _ = pose_xy(state, progress, path)
    initial_position_error = math.hypot(
        initial_px - truth[matches[first]]["x_m"],
        initial_py - truth[matches[first]]["y_m"])
    valid_steps = 0
    failure = None
    for position in range(len(control_indices) - 1):
        control_index = control_indices[position]
        target_index = control_indices[position + 1]
        actual = truth[matches[target_index]]
        curvature = periodic_linear(
            progress, path["s"], path["curvature"], path["length"])
        q = np.asarray((
            controls[control_index]["first_control_steering_rate_radps"],
            controls[control_index]["first_control_target_speed_rate_mps2"]),
            dtype=np.float64)
        try:
            next_state, delta_s, _ = python_step(
                vehicle_model, state, q, DT_S, curvature)
        except (ArithmeticError, FloatingPointError) as exc:
            failure = {"step": position, "progress_m": progress,
                       "reason": str(exc)}
            break

        if learned_model is not None:
            features = feature_vector(learned_model, state, steering_slew)
            dv = float(predict(learned_model, features))
            if not math.isfinite(dv):
                failure = {"step": position, "progress_m": progress,
                           "reason": "non-finite learned lateral-speed increment"}
                break
            next_state[3] = state[3] + dv
        progress += delta_s
        if not np.isfinite(next_state).all() or not math.isfinite(progress):
            failure = {"step": position, "progress_m": progress,
                       "reason": "non-finite predicted state"}
            break
        steering_slew = (float(next_state[9]) - float(state[9])) / DT_S
        state = next_state
        step_errors = compare_state(state, progress, actual, path, errors)
        lap = str(controls[target_index]["lap_number"])
        by_lap.setdefault(lap, make_errors())
        compare_state(state, progress, actual, path, by_lap[lap])
        valid_steps += 1
        if valid_steps in checkpoint_steps:
            checkpoints.append({"step": valid_steps,
                                "elapsed_s": valid_steps * DT_S,
                                "lap": int(lap), "errors": step_errors})

    return {
        "status": "complete" if failure is None else "invalid_before_end",
        "steps": valid_steps,
        "requested_steps": len(control_indices) - 1,
        "failure": failure,
        "initial_position_error_m": initial_position_error,
        "checkpoints": checkpoints,
        "whole_sequence": summarize(errors),
        "by_lap": {lap: summarize(lap_errors)
                   for lap, lap_errors in sorted(by_lap.items(),
                                                 key=lambda item: int(item[0]))},
    }


def predict(model: dict[str, Any], features: np.ndarray) -> float:
    means = np.asarray(model["feature_means"], dtype=np.float64)
    scales = np.asarray(model["feature_scales"], dtype=np.float64)
    coefficients = np.asarray(
        model["standardized_coefficients"], dtype=np.float64)
    return float(((features - means) / scales) @ coefficients
                 + float(model["increment_intercept_mps"]))


def paired_lap_metrics(hold: dict[str, Any], candidate: dict[str, Any]
                       ) -> dict[str, Any]:
    channels = sorted(set(hold.get("whole_sequence", {}))
                      & set(candidate.get("whole_sequence", {})))
    output = {}
    for channel in channels:
        output[channel] = paired_lap_bootstrap(
            hold.get("by_lap", {}), candidate.get("by_lap", {}), channel)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-run", type=Path, default=DEFAULT_R01)
    parser.add_argument("--validation-run", type=Path, default=DEFAULT_R02)
    parser.add_argument("--model-report", type=Path, default=DEFAULT_MODEL_REPORT)
    parser.add_argument("--trajectory", type=Path, default=DEFAULT_TRAJECTORY)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    train_manifest = json.loads((args.train_run / "run_manifest.json").read_text())
    validation_manifest = json.loads(
        (args.validation_run / "run_manifest.json").read_text())
    trajectory_hash = sha256(args.trajectory)
    if train_manifest["trajectory_sha256"] != validation_manifest["trajectory_sha256"]:
        raise ValueError("P0 runs do not share the same reference trajectory")
    if trajectory_hash != validation_manifest["trajectory_sha256"]:
        raise ValueError("selected raceline is not the frozen P0 reference")
    if train_manifest.get("track") != "practice" or validation_manifest.get("track") != "practice":
        raise ValueError("recursive replay expects the practice P0 sessions")

    report_model = json.loads(args.model_report.read_text(encoding="utf-8"))
    if report_model.get("training", {}).get("run_id") != "practice_9g_parent_reproduced_20261006_r01":
        raise ValueError("learned model was not fit from the designated P0 r01")
    if report_model.get("validation", {}).get("run_id") != "practice_9g_parent_reproduced_20261006_r02":
        raise ValueError("one-step validation is not the designated P0 r02")
    learned_model = report_model["no_throttle_model"]

    path = read_trajectory(args.trajectory)
    vehicle_model = json.loads(MODEL_PARAMETERS.read_text(encoding="utf-8"))
    truth, controls, matches = load_inputs(args.validation_run)
    continuous_indices = list(range(len(controls)))
    hold_continuous = replay(continuous_indices, truth, controls, matches,
                             path, vehicle_model, None)
    learned_continuous = replay(continuous_indices, truth, controls, matches,
                                path, vehicle_model, learned_model)

    per_lap: dict[str, Any] = {}
    for lap in sorted(SCORED_LAPS):
        indices = [i for i, row in enumerate(controls)
                   if int(row["lap_number"]) == lap]
        hold = replay(indices, truth, controls, matches, path, vehicle_model, None)
        learned = replay(indices, truth, controls, matches, path, vehicle_model,
                         learned_model)
        per_lap[str(lap)] = {"hold_v": hold, "learned_v_increment": learned}

    summary = {
        "schema_version": 1,
        "candidate_id": "p0_no_throttle_lateral_velocity_increment_ridge_v1",
        "status": "research_only_not_integrated",
        "validation_run": validation_manifest["run_id"],
        "training_run": train_manifest["run_id"],
        "data_split": "P0 r01 fit; P0 r02 whole-run holdout; scored laps 2-11",
        "trajectory_sha256": trajectory_hash,
        "trajectory_length_m": float(path["length"]),
        "vehicle_model_sha256": sha256(MODEL_PARAMETERS),
        "model_report_sha256": sha256(args.model_report),
        "source_csv_sha256": {
            "validation_tracking": sha256(args.validation_run / "analysis/race_report_yaw_residual_input/tracking_error.csv"),
            "validation_controller": sha256(args.validation_run / "analysis/race_report_yaw_residual_input/controller_saturation.csv"),
        },
        "time_contract": {
            "step_s": DT_S,
            "timestamps_used_for": "stream alignment only; every model step is exactly 25 ms",
            "sequence_alignment_max_abs_receipt_offset_s": 0.010,
            "steps_are_not_rescaled_to_recorded_jitter": True,
        },
        "rollout_contract": {
            "transition": "shared production MPC 25 ms Python mirror; verified C/Python/CasADi parity",
            "candidate_change": "replace held lateral-speed state with fitted next-step increment",
            "command_inputs": "recorded MPC steering-rate and target-speed-rate sequence",
            "future_recorded_vehicle_or_sensor_states_used_as_inputs": False,
            "truth_use": "initial condition and offline scoring only",
            "control_feedback": "not recomputed; this is a forced-command plant replay, not a closed-loop MPC simulation",
        },
        "continuous_10_lap_replay": {
            "hold_v": hold_continuous,
            "learned_v_increment": learned_continuous,
            "paired_within_run_lap_bootstrap": paired_lap_metrics(
                hold_continuous, learned_continuous),
        },
        "truth_reinitialized_single_lap_replays": per_lap,
        "interpretation": (
            "This test determines whether the teacher-forced lateral-speed "
            "gain survives recursive state prediction under the same logged "
            "practice command sequences. The 10-lap replay starts from one "
            "truth initialization; each separate-lap result is additionally "
            "reinitialized from that lap's truth state. Both are forced-command "
            "plant tests, not closed-loop lap-time claims. P0 does not cover the "
            "joint high-speed/high-steering regime needed to justify a sub-5 s "
            "raceline."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "candidate_id": summary["candidate_id"],
        "continuous_hold_status": hold_continuous["status"],
        "continuous_candidate_status": learned_continuous["status"],
        "continuous_hold_errors": hold_continuous.get("whole_sequence"),
        "continuous_candidate_errors": learned_continuous.get("whole_sequence"),
        "single_lap_candidate_failures": {
            lap: item["learned_v_increment"]["failure"]
            for lap, item in per_lap.items()
            if item["learned_v_increment"]["status"] != "complete"
        },
        "output": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
