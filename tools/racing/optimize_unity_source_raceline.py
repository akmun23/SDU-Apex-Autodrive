#!/usr/bin/env python3
"""Optimize one practice lap by evaluating every candidate in source Unity.

The search starts from a recorded, repeatable practice command lap. It leaves
the warm-up commands untouched, then modifies the lap's steering/throttle
profile and command time scale. Each objective evaluation launches the
source-built Unity player through the established practice batch-mode path;
Unity's native 1 kHz physics and extracted practice course collider decide the
result. This is a simulator-in-the-loop command/raceline search, not a score
from the reduced Frenet surrogate or from MPC tracking.

Only the current best trial's full wheel-dynamics CSV is retained. Profiles,
compact metrics, and simulator logs for every trial remain in the campaign
directory for reproducibility.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import norm, qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel


REPO = Path(__file__).resolve().parents[2]
DEFAULT_PROFILE = REPO / (
    "live_runs/unity_source_practice_command_replay_20261013_r01/"
    "packet_command_profile.json")
DEFAULT_PARITY = REPO / (
    "live_runs/unity_source_practice_scene_command_replay_20261013_r07/"
    "parity_vs_practice_r07.json")
DEFAULT_CAPTURE = REPO / (
    "live_runs/unity_source_practice_scene_command_replay_20261013_r07/"
    "dynamics_capture/wheel_dynamics_unity_source_practice_race_scene_r07.csv")
DEFAULT_PLAYER = REPO / (
    "live_runs/unity_source_practice_scene_command_replay_20261013_r07/"
    "player_overlay")
START_SIM = REPO / "tools/start_simulator.sh"
PARAMETER_BOUNDS = np.asarray([
    [1.00, 2.80],  # positive throttle command multiplier
    [0.75, 1.25],  # steering command multiplier
    [1.00, 1.45],  # source command-profile playback rate
], dtype=np.float64)
PARAMETER_NAMES = ("throttle_scale", "steering_scale", "profile_rate")
SCORED_COMMAND_HORIZON = 360  # 9 s maximum, after the untouched warm-up lap
FIXED_DT_S = 0.001
TICKS_PER_COMMAND = 25
SLIP_DIAGNOSTIC_SPEED_THRESHOLDS_MPS = (6.0, 7.0)
KINEMATIC_REFERENCE_WHEELBASE_M = 0.324
INVALID_SCORE_S = 20.0


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _read_capture_xz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8") as stream:
        header = next(line for line in stream if not line.startswith("#"))
        columns = next(csv.reader([header]))
        wanted = ("sim_time_s", "pos_world_x_m", "pos_world_z_m")
        indices = [columns.index(name) for name in wanted]
        values = np.loadtxt(stream, delimiter=",", usecols=indices,
                            dtype=np.float64, ndmin=2)
    if values.shape[0] < 100 or not np.isfinite(values).all():
        raise ValueError(f"invalid/incomplete Unity dynamics capture: {path}")
    return values[:, 0], values[:, 1:3]


def _profile_parts(profile: dict[str, Any], parity: dict[str, Any]
                   ) -> tuple[list[dict[str, float]], list[dict[str, float]], int]:
    commands = profile.get("commands")
    boundaries = parity.get("lap_count_boundaries")
    if not isinstance(commands, list) or not isinstance(boundaries, list) or len(boundaries) < 2:
        raise ValueError("need a command profile and two lap-count boundaries")
    start = int(boundaries[0]["packet_index"])
    stop = int(boundaries[1]["packet_index"])
    if not (0 < start < stop < len(commands)):
        raise ValueError("lap boundary indices do not fit the source command profile")
    clean = []
    for row in commands:
        steering = float(row["steering_norm"])
        throttle = float(row["throttle_norm"])
        if not (math.isfinite(steering) and math.isfinite(throttle)
                and abs(steering) <= 1.0 and abs(throttle) <= 1.0):
            raise ValueError("source command profile contains invalid controls")
        clean.append({"steering_norm": steering, "throttle_norm": throttle})
    if abs(clean[0]["steering_norm"]) > 1.0e-6 or abs(clean[0]["throttle_norm"]) > 1.0e-6:
        raise ValueError("source profile must begin at neutral")
    return clean[:start], clean[start:stop], start


def _make_profile(warmup: list[dict[str, float]],
                  lap: list[dict[str, float]],
                  params: np.ndarray) -> dict[str, Any]:
    throttle_scale, steering_scale, profile_rate = map(float, params)
    scored = []
    for output_index in range(SCORED_COMMAND_HORIZON):
        source_index = min(int(output_index * profile_rate), len(lap) - 1)
        source = lap[source_index]
        scored.append({
            "steering_norm": float(np.clip(
                source["steering_norm"] * steering_scale, -1.0, 1.0)),
            "throttle_norm": float(np.clip(
                source["throttle_norm"] * throttle_scale, -1.0, 1.0)),
        })
    return {
        "schema_version": 1,
        "fixed_delta_time_s": FIXED_DT_S,
        "physics_ticks_per_command": TICKS_PER_COMMAND,
        "command_phase_fixed_steps": 0,
        "commands": warmup + scored,
    }


def _finish_gate(capture_path: Path, warmup_commands: int
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times, xz = _read_capture_xz(capture_path)
    crossing_time = warmup_commands * TICKS_PER_COMMAND * FIXED_DT_S
    center_index = int(np.argmin(np.abs(times - crossing_time)))
    if center_index < 30 or center_index + 30 >= len(times):
        raise ValueError("reference capture does not span the warm-up finish gate")
    origin = xz[center_index].copy()
    direction = xz[center_index + 25] - xz[center_index - 25]
    magnitude = float(np.linalg.norm(direction))
    if magnitude < 0.01:
        raise ValueError("could not determine forward direction at lap gate")
    direction /= magnitude
    return origin, direction, xz[center_index - 25:center_index + 26]


def _crossing_times(times: np.ndarray, xz: np.ndarray,
                    origin: np.ndarray, direction: np.ndarray) -> list[float]:
    signed = (xz - origin) @ direction
    indices = np.flatnonzero((signed[:-1] < 0.0) & (signed[1:] >= 0.0))
    result = []
    for index in indices:
        fraction = -signed[index] / max(signed[index + 1] - signed[index], 1.0e-12)
        result.append(float(times[index] + fraction * (times[index + 1] - times[index])))
    return result


def _path_progress(times: np.ndarray, xz: np.ndarray,
                   gate_time: float, lap_path: np.ndarray) -> float:
    after_start = xz[times >= gate_time + 0.10]
    if len(after_start) == 0:
        return 0.0
    # The path is one full measured lap; use ordered nearest-index progress to
    # distinguish a useful partial lap from an early collision.
    tree = cKDTree(lap_path)
    _, nearest = tree.query(after_start, k=1)
    nearest = np.maximum.accumulate(nearest)
    return float(np.clip(np.max(nearest) / max(len(lap_path) - 1, 1), 0.0, 1.0))


def _dynamics_summary(path: Path, start_s: float | None = None,
                      stop_s: float | None = None) -> dict[str, Any]:
    wheels = ("fl", "fr", "rl", "rr")
    names = ["sim_time_s", "throttle_command", "vel_body_x_mps",
             "vel_body_z_mps"]
    for wheel in wheels:
        names.extend((f"{wheel}_motor_torque_nm", f"{wheel}_brake_torque_nm",
                      f"{wheel}_grounded", f"{wheel}_forward_slip",
                      f"{wheel}_sideways_slip"))
    names.extend(("omega_body_y_rps", "fl_steer_deg", "fr_steer_deg"))
    curve_limits: dict[str, dict[str, tuple[float, float]]] = {}
    with path.open("r", encoding="utf-8") as stream:
        header = None
        wheel_names = {"front_left": "fl", "front_right": "fr",
                       "rear_left": "rl", "rear_right": "rr"}
        for line in stream:
            if line.startswith("# "):
                key, separator, raw_value = line[2:].strip().partition("=")
                if separator and key.endswith(("_forward_curve", "_sideways_curve")):
                    wheel_name, _, curve_name = key.rpartition("_")
                    wheel_name, _, axis = wheel_name.rpartition("_")
                    wheel = wheel_names.get(wheel_name)
                    curve = np.fromstring(raw_value, sep=",")
                    if (curve_name == "curve" and wheel and
                            curve.size == 5 and np.isfinite(curve).all()):
                        curve_limits.setdefault(wheel, {})[axis] = (
                            float(curve[0]), float(curve[2]))
                continue
            header = line
            break
        if header is None:
            raise ValueError(f"Unity dynamics capture has no CSV header: {path}")
        columns = next(csv.reader([header]))
        indices = [columns.index(name) for name in names]
        values = np.loadtxt(stream, delimiter=",", usecols=indices,
                            dtype=np.float64, ndmin=2)
    selection = np.ones(len(values), dtype=bool)
    if start_s is not None:
        selection &= values[:, 0] >= start_s
    if stop_s is not None:
        selection &= values[:, 0] <= stop_s
    if np.count_nonzero(selection) < 2:
        raise ValueError("Unity dynamics capture has too few samples in scored lap")
    selected = values[selection]
    lateral_velocity = np.abs(selected[:, 2])
    forward_velocity = selected[:, 3]
    speed = np.hypot(selected[:, 2], selected[:, 3])
    sideslip = np.arctan2(lateral_velocity,
                          np.maximum(np.abs(forward_velocity), 1.0e-6))
    # Source prefab dimensions and measured per-wheel front angles provide a
    # kinematic reference, not a tire model. A growing positive deficit is a
    # useful under-response diagnostic when the tires saturate.
    front_steer_rad = np.deg2rad(0.5 * (selected[:, 25] + selected[:, 26]))
    kinematic_yaw_rate = np.abs(
        forward_velocity * np.tan(front_steer_rad) /
        KINEMATIC_REFERENCE_WHEELBASE_M)
    observed_yaw_rate = np.abs(selected[:, 24])
    kinematic_yaw_deficit = np.maximum(
        0.0, kinematic_yaw_rate - observed_yaw_rate)
    motor_torques = []
    brake_torques = []
    forward_slips = []
    sideways_slips = []
    grounded_values = []
    column = 4
    for _ in wheels:
        motor_torques.append(selected[:, column])
        brake_torques.append(selected[:, column + 1])
        grounded_values.append(selected[:, column + 2] > 0.5)
        forward_slips.append(selected[:, column + 3])
        sideways_slips.append(np.abs(selected[:, column + 4]))
        column += 5
    motor_torques = np.asarray(motor_torques)
    brake_torques = np.asarray(brake_torques)
    forward_slips = np.asarray(forward_slips)
    sideways_slips = np.asarray(sideways_slips)
    grounded = np.asarray(grounded_values)
    total_grounded = int(np.count_nonzero(grounded))
    forward_samples_signed = forward_slips[grounded]
    forward_samples = np.abs(forward_samples_signed)
    sideways_samples = sideways_slips[grounded]
    driven_samples = grounded & (motor_torques > 1.0e-6) & (brake_torques <= 1.0e-6)
    reverse_torque_samples = (grounded & (motor_torques < -1.0e-6) &
                              (brake_torques <= 1.0e-6))
    braking_samples = grounded & (brake_torques > 1.0e-6)
    driven_forward_slips = forward_slips[driven_samples]
    reverse_forward_slips = forward_slips[reverse_torque_samples]
    braking_forward_slips = forward_slips[braking_samples]
    metrics: dict[str, Any] = {
        "scored_duration_s": float(selected[-1, 0] - selected[0, 0]),
        "peak_body_speed_mps": float(np.max(speed)),
        "p95_abs_lateral_velocity_mps": float(np.quantile(lateral_velocity, 0.95)),
        "max_abs_lateral_velocity_mps": float(np.max(lateral_velocity)),
        "p95_abs_sideslip_rad": float(np.quantile(sideslip, 0.95)),
        "max_abs_sideslip_rad": float(np.max(sideslip)),
        "p95_kinematic_yaw_underresponse_rps": float(
            np.quantile(kinematic_yaw_deficit, 0.95)),
        "max_kinematic_yaw_underresponse_rps": float(
            np.max(kinematic_yaw_deficit)),
        "p95_abs_forward_wheel_slip": (
            float(np.quantile(forward_samples, 0.95))
            if forward_samples.size else None),
        "max_abs_forward_wheel_slip": (
            float(np.max(forward_samples)) if forward_samples.size else None),
        "p95_signed_forward_wheel_slip": (
            float(np.quantile(forward_samples_signed, 0.95))
            if forward_samples_signed.size else None),
        "p05_signed_forward_wheel_slip": (
            float(np.quantile(forward_samples_signed, 0.05))
            if forward_samples_signed.size else None),
        "driven_wheel_sample_fraction": float(
            np.count_nonzero(driven_samples) / max(total_grounded, 1)),
        "braked_wheel_sample_fraction": float(
            np.count_nonzero(braking_samples) / max(total_grounded, 1)),
        "reverse_torque_wheel_sample_fraction": float(
            np.count_nonzero(reverse_torque_samples) / max(total_grounded, 1)),
        "p95_positive_slip_when_driven": (
            float(np.quantile(driven_forward_slips, 0.95))
            if driven_forward_slips.size else None),
        "p05_forward_slip_when_braking": (
            float(np.quantile(braking_forward_slips, 0.05))
            if braking_forward_slips.size else None),
        "p05_forward_slip_when_reverse_torque": (
            float(np.quantile(reverse_forward_slips, 0.05))
            if reverse_forward_slips.size else None),
        "p95_abs_sideways_wheel_slip": (
            float(np.quantile(sideways_samples, 0.95))
            if sideways_samples.size else None),
        "max_abs_sideways_wheel_slip": (
            float(np.max(sideways_samples)) if sideways_samples.size else None),
    }
    metrics["p95_abs_wheel_slip"] = max(
        value for value in (metrics["p95_abs_forward_wheel_slip"],
                            metrics["p95_abs_sideways_wheel_slip"])
        if value is not None)
    metrics["max_abs_wheel_slip"] = max(
        value for value in (metrics["max_abs_forward_wheel_slip"],
                            metrics["max_abs_sideways_wheel_slip"])
        if value is not None)
    metrics["wheel_forward_slip_over_extremum_fraction"] = None
    metrics["wheel_forward_slip_over_asymptote_fraction"] = None
    metrics["wheel_forward_positive_slip_over_asymptote_fraction"] = None
    metrics["wheel_forward_braking_slip_over_asymptote_fraction"] = None
    metrics["wheel_sideways_slip_over_extremum_fraction"] = None
    metrics["wheel_sideways_slip_over_asymptote_fraction"] = None
    if total_grounded and all("forward" in curve_limits.get(w, {}) for w in wheels):
        extrema = np.asarray([curve_limits[w]["forward"][0] for w in wheels])[:, None]
        asymptotes = np.asarray([curve_limits[w]["forward"][1] for w in wheels])[:, None]
        metrics["wheel_forward_slip_over_extremum_fraction"] = float(
            np.count_nonzero(grounded & (forward_slips > extrema)) / total_grounded)
        metrics["wheel_forward_slip_over_asymptote_fraction"] = float(
            np.count_nonzero(grounded & (forward_slips > asymptotes)) / total_grounded)
        metrics["wheel_forward_positive_slip_over_asymptote_fraction"] = float(
            np.count_nonzero(driven_samples & (forward_slips > asymptotes)) /
            max(np.count_nonzero(driven_samples), 1))
        metrics["wheel_forward_braking_slip_over_asymptote_fraction"] = float(
            np.count_nonzero(braking_samples &
                             (forward_slips < -asymptotes)) /
            max(np.count_nonzero(braking_samples), 1))
    if total_grounded and all("sideways" in curve_limits.get(w, {}) for w in wheels):
        extrema = np.asarray([curve_limits[w]["sideways"][0] for w in wheels])[:, None]
        asymptotes = np.asarray([curve_limits[w]["sideways"][1] for w in wheels])[:, None]
        metrics["wheel_sideways_slip_over_extremum_fraction"] = float(
            np.count_nonzero(grounded & (sideways_slips > extrema)) / total_grounded)
        metrics["wheel_sideways_slip_over_asymptote_fraction"] = float(
            np.count_nonzero(grounded & (sideways_slips > asymptotes)) / total_grounded)
    axle_wheels = {"front": (0, 1), "rear": (2, 3)}
    for axle, wheel_indices in axle_wheels.items():
        axle_grounded = grounded[list(wheel_indices)]
        axle_sideways = sideways_slips[list(wheel_indices)]
        axle_samples = axle_sideways[axle_grounded]
        metrics[f"p95_abs_{axle}_sideways_wheel_slip"] = (
            float(np.quantile(axle_samples, 0.95))
            if axle_samples.size else None)
        metrics[f"{axle}_sideways_slip_over_asymptote_fraction"] = None
        if (axle_samples.size and all(
                "sideways" in curve_limits.get(wheels[index], {})
                for index in wheel_indices)):
            axle_asymptotes = np.asarray([
                curve_limits[wheels[index]]["sideways"][1]
                for index in wheel_indices])[:, None]
            metrics[f"{axle}_sideways_slip_over_asymptote_fraction"] = float(
                np.count_nonzero(
                    axle_grounded & (axle_sideways > axle_asymptotes)) /
                np.count_nonzero(axle_grounded))
    for threshold_mps in SLIP_DIAGNOSTIC_SPEED_THRESHOLDS_MPS:
        suffix = f"speed_ge_{threshold_mps:g}_mps"
        fast = speed >= threshold_mps
        fast_wheels = grounded & fast[None, :]
        fast_grounded_count = int(np.count_nonzero(fast_wheels))
        if not np.any(fast):
            metrics[f"p95_abs_sideslip_rad_{suffix}"] = None
            metrics[f"p95_kinematic_yaw_underresponse_rps_{suffix}"] = None
            metrics[f"p95_abs_sideways_wheel_slip_{suffix}"] = None
            for axle in axle_wheels:
                metrics[f"p95_abs_{axle}_sideways_wheel_slip_{suffix}"] = None
                metrics[f"{axle}_sideways_slip_over_asymptote_fraction_{suffix}"] = None
            metrics[f"wheel_sideways_slip_over_extremum_fraction_{suffix}"] = None
            metrics[f"wheel_sideways_slip_over_asymptote_fraction_{suffix}"] = None
            metrics[f"p95_positive_slip_when_driven_{suffix}"] = None
            metrics[f"wheel_forward_positive_slip_over_asymptote_fraction_{suffix}"] = None
            continue

        metrics[f"p95_abs_sideslip_rad_{suffix}"] = float(
            np.quantile(sideslip[fast], 0.95))
        metrics[f"p95_kinematic_yaw_underresponse_rps_{suffix}"] = float(
            np.quantile(kinematic_yaw_deficit[fast], 0.95))
        fast_sideways = sideways_slips[fast_wheels]
        metrics[f"p95_abs_sideways_wheel_slip_{suffix}"] = (
            float(np.quantile(fast_sideways, 0.95))
            if fast_sideways.size else None)
        for axle, wheel_indices in axle_wheels.items():
            axle_fast_grounded = grounded[list(wheel_indices)] & fast[None, :]
            axle_fast_sideways = sideways_slips[list(wheel_indices)]
            axle_fast_samples = axle_fast_sideways[axle_fast_grounded]
            metrics[f"p95_abs_{axle}_sideways_wheel_slip_{suffix}"] = (
                float(np.quantile(axle_fast_samples, 0.95))
                if axle_fast_samples.size else None)
            if (axle_fast_samples.size and all(
                    "sideways" in curve_limits.get(wheels[index], {})
                    for index in wheel_indices)):
                axle_asymptotes = np.asarray([
                    curve_limits[wheels[index]]["sideways"][1]
                    for index in wheel_indices])[:, None]
                metrics[
                    f"{axle}_sideways_slip_over_asymptote_fraction_{suffix}"
                ] = float(np.count_nonzero(
                    axle_fast_grounded &
                    (axle_fast_sideways > axle_asymptotes)) /
                    np.count_nonzero(axle_fast_grounded))
            else:
                metrics[
                    f"{axle}_sideways_slip_over_asymptote_fraction_{suffix}"
                ] = None
        if (fast_grounded_count and
                all("sideways" in curve_limits.get(w, {}) for w in wheels)):
            extrema = np.asarray(
                [curve_limits[w]["sideways"][0] for w in wheels])[:, None]
            asymptotes = np.asarray(
                [curve_limits[w]["sideways"][1] for w in wheels])[:, None]
            metrics[f"wheel_sideways_slip_over_extremum_fraction_{suffix}"] = float(
                np.count_nonzero(fast_wheels & (sideways_slips > extrema)) /
                fast_grounded_count)
            metrics[f"wheel_sideways_slip_over_asymptote_fraction_{suffix}"] = float(
                np.count_nonzero(fast_wheels & (sideways_slips > asymptotes)) /
                fast_grounded_count)
        else:
            metrics[f"wheel_sideways_slip_over_extremum_fraction_{suffix}"] = None
            metrics[f"wheel_sideways_slip_over_asymptote_fraction_{suffix}"] = None

        fast_driven = driven_samples & fast[None, :]
        fast_driven_slips = forward_slips[fast_driven]
        metrics[f"p95_positive_slip_when_driven_{suffix}"] = (
            float(np.quantile(fast_driven_slips, 0.95))
            if fast_driven_slips.size else None)
        if (fast_driven_slips.size and
                all("forward" in curve_limits.get(w, {}) for w in wheels)):
            asymptotes = np.asarray(
                [curve_limits[w]["forward"][1] for w in wheels])[:, None]
            metrics[
                f"wheel_forward_positive_slip_over_asymptote_fraction_{suffix}"
            ] = float(np.count_nonzero(
                fast_driven & (forward_slips > asymptotes)) /
                np.count_nonzero(fast_driven))
        else:
            metrics[
                f"wheel_forward_positive_slip_over_asymptote_fraction_{suffix}"
            ] = None
    return metrics


def _trial_score(run_dir: Path, params: np.ndarray,
                 origin: np.ndarray, direction: np.ndarray,
                 warmup_commands: int, lap_path: np.ndarray,
                 player_dir: Path, timeout_s: float,
                 parameter_names: tuple[str, ...] = PARAMETER_NAMES
                 ) -> dict[str, Any]:
    trial_name = run_dir.name
    dynamics_dir = run_dir / "dynamics"
    dynamics_dir.mkdir(parents=True)
    env = os.environ.copy()
    env.update({
        "SDU_APEX_SIM_TRACK": "practice",
        "SDU_APEX_SIM_MODE": "batchmode",
        "SDU_APEX_SIM_CONTAINER": "sdu_apex_" + trial_name[:55],
        "SDU_APEX_SIM_PLAYER_DIR": str(player_dir),
        "SDU_APEX_SIM_DYNAMICS_DIR": str(dynamics_dir),
        "SDU_APEX_SIM_DYNAMICS_RUN_ID": trial_name,
        "SDU_APEX_SIM_OFFLINE_COMMAND_PROFILE": str(run_dir / "command_profile.json"),
        "SDU_APEX_SIM_OFFLINE_PRACTICE_TRACK": "1",
        "SDU_APEX_SIM_LOG_FILE": "-",
    })
    started = time.monotonic()
    timed_out = False
    with (run_dir / "simulator.log").open("wb") as log:
        try:
            completed = subprocess.run(
                [str(START_SIM)], cwd=REPO, env=env, stdout=log,
                stderr=subprocess.STDOUT, timeout=timeout_s, check=False)
            exit_code = completed.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            exit_code = None
    wall_s = time.monotonic() - started

    result_path = dynamics_dir / "offline_result.json"
    capture_path = next(dynamics_dir.glob("wheel_dynamics_*.csv"), None)
    result = _read_json(result_path) if result_path.is_file() else {}
    metrics: dict[str, Any] = {
        "trial": trial_name,
        "parameters": dict(zip(parameter_names, map(float, params))),
        "exit_code": exit_code,
        "timed_out": timed_out,
        "wall_time_s": wall_s,
        "unity_status": result.get("status", "missing_result"),
        "unity_reason": result.get("reason", ""),
        "fixed_steps_applied": result.get("fixed_steps_applied"),
        "collisions_observed": result.get("collisions_observed"),
        "spatial_selection_updates": result.get("spatial_selection_updates"),
        "spatial_path_index_advance_max": result.get(
            "spatial_path_index_advance_max"),
        "spatial_path_index_advance_mean": result.get(
            "spatial_path_index_advance_mean"),
        "spatial_progress_distance_scale": result.get(
            "spatial_progress_distance_scale"),
        "capture_path": str(capture_path) if capture_path else None,
    }
    lap_time = None
    lap_start_time = None
    progress = 0.0
    if capture_path is not None:
        times, xz = _read_capture_xz(capture_path)
        crossings = _crossing_times(times, xz, origin, direction)
        expected_start = warmup_commands * TICKS_PER_COMMAND * FIXED_DT_S
        possible_starts = [t for t in crossings if abs(t - expected_start) <= 0.25]
        if possible_starts:
            lap_start = min(possible_starts, key=lambda t: abs(t - expected_start))
            finishes = [t for t in crossings if t >= lap_start + 3.0]
            if finishes:
                lap_time = finishes[0] - lap_start
                lap_start_time = lap_start
            progress = _path_progress(times, xz, lap_start, lap_path)
        metrics["forward_gate_crossings_s"] = crossings
        metrics["reference_lap_progress_fraction"] = progress
        if lap_time is not None:
            metrics["unity_lap_time_s"] = float(lap_time)
        metrics.update(_dynamics_summary(
            capture_path, lap_start_time,
            lap_start_time + lap_time if lap_time is not None else None))

    collision = int(result.get("collisions_observed") or 0) > 0
    scored_lap_precedes_collision = (
        lap_time is not None and result.get("status") == "collision"
        and times[-1] >= (possible_starts[0] + lap_time))
    valid = (lap_time is not None and not timed_out
             and (not collision or scored_lap_precedes_collision))
    metrics["lap_valid"] = bool(valid)
    metrics["collision_after_scored_lap"] = bool(scored_lap_precedes_collision)
    if valid:
        metrics["objective_s"] = float(lap_time)
    else:
        metrics["objective_s"] = INVALID_SCORE_S + 5.0 * (1.0 - progress)
    metrics["best_capture_path"] = metrics["capture_path"] if valid else None
    return metrics


def _propose_ei(rows: list[dict[str, Any]], seed: int,
                trial_index: int) -> tuple[np.ndarray, dict[str, Any]]:
    lower, upper = PARAMETER_BOUNDS[:, 0], PARAMETER_BOUNDS[:, 1]
    points = np.asarray([[row["parameters"][name] for name in PARAMETER_NAMES]
                         for row in rows], dtype=np.float64)
    scores = np.asarray([float(row["objective_s"]) for row in rows])
    normalized = (points - lower) / (upper - lower)
    kernel = (ConstantKernel(1.0, (0.1, 10.0))
              * Matern(length_scale=np.full(3, 0.4),
                       length_scale_bounds=(0.08, 2.0), nu=2.5)
              + WhiteKernel(noise_level=1.0e-5,
                            noise_level_bounds=(1.0e-7, 0.05)))
    model = GaussianProcessRegressor(
        kernel=kernel, normalize_y=True, n_restarts_optimizer=2,
        random_state=seed + trial_index, alpha=1.0e-8)
    model.fit(normalized, scores)
    sampler = qmc.Sobol(d=3, scramble=True, seed=seed + trial_index)
    candidate_x = sampler.random_base2(m=12)
    mean, std = model.predict(candidate_x, return_std=True)
    best = float(np.min(scores))
    improvement = best - mean - 0.01
    z = improvement / np.maximum(std, 1.0e-9)
    ei = improvement * norm.cdf(z) + std * norm.pdf(z)
    for observed in normalized:
        close = np.linalg.norm(candidate_x - observed, axis=1) < 0.025
        ei[close] = -np.inf
    chosen = candidate_x[int(np.argmax(ei))]
    params = lower + chosen * (upper - lower)
    return params, {
        "method": "gaussian_process_expected_improvement",
        "expected_improvement_s": float(np.max(ei)),
        "observations": len(rows),
        "kernel": str(model.kernel_),
    }


def _initial_design() -> list[np.ndarray]:
    return [
        np.asarray([1.00, 1.00, 1.00]),
        np.asarray([1.35, 1.00, 1.08]),
        np.asarray([1.80, 1.00, 1.18]),
        np.asarray([2.35, 1.00, 1.30]),
        np.asarray([1.65, 0.90, 1.15]),
        np.asarray([1.65, 1.10, 1.15]),
        np.asarray([2.10, 0.92, 1.28]),
        np.asarray([2.10, 1.08, 1.28]),
    ]


def run_campaign(args: argparse.Namespace) -> int:
    profile_path = args.source_profile.resolve()
    parity_path = args.parity_report.resolve()
    capture_path = args.reference_capture.resolve()
    player_dir = args.player_dir.resolve()
    if not START_SIM.is_file() or not player_dir.is_dir():
        raise FileNotFoundError("established source-player launcher or player is missing")
    if not all(path.is_file() for path in (profile_path, parity_path, capture_path)):
        raise FileNotFoundError("source command/parity/capture input is missing")
    source_profile = _read_json(profile_path)
    parity = _read_json(parity_path)
    warmup, lap, warmup_commands = _profile_parts(source_profile, parity)
    gate_origin, gate_direction, _ = _finish_gate(capture_path, warmup_commands)
    ref_times, ref_xz = _read_capture_xz(capture_path)
    boundary_time = warmup_commands * TICKS_PER_COMMAND * FIXED_DT_S
    next_boundary_time = int(parity["lap_count_boundaries"][1]["packet_index"]) * TICKS_PER_COMMAND * FIXED_DT_S
    lap_path = ref_xz[(ref_times >= boundary_time) & (ref_times <= next_boundary_time)]
    if len(lap_path) < 100:
        raise ValueError("reference source replay has no complete gate-to-gate path")

    profile_hash = __import__("hashlib").sha256(profile_path.read_bytes()).hexdigest()
    if args.resume:
        if not args.output.is_dir():
            raise FileNotFoundError(f"cannot resume missing campaign: {args.output}")
        manifest = _read_json(args.output / "campaign.json")
        if (manifest.get("source_profile_sha256") != profile_hash
                or manifest.get("warmup_commands") != warmup_commands
                or manifest.get("base_lap_commands") != len(lap)):
            raise ValueError("resume inputs differ from the recorded campaign")
        trial_rows = []
        for trial_path in sorted((args.output / "trials").glob("trial_*/trial.json")):
            trial_rows.append(_read_json(trial_path))
        expected_names = [f"trial_{i:03d}" for i in range(len(trial_rows))]
        if [row.get("trial") for row in trial_rows] != expected_names:
            raise ValueError("campaign trial records are not contiguous")
        prior_best = [row for row in trial_rows if row.get("lap_valid")]
        best_capture = (Path(min(prior_best, key=lambda row: row["objective_s"])
                             ["best_capture_path"])
                        if prior_best else None)
        if best_capture is not None and not best_capture.is_file():
            raise FileNotFoundError(f"best Unity capture is missing: {best_capture}")
        manifest["status"] = "running"
    else:
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite campaign directory: {args.output}")
        args.output.mkdir(parents=True)
        (args.output / "trials").mkdir()
        trial_rows = []
        best_capture = None
        manifest = {
        "status": "running",
        "objective": "minimum source-Unity gate-to-gate lap time; invalidate collision before finish",
        "source_profile": str(profile_path),
        "source_profile_sha256": profile_hash,
        "reference_capture": str(capture_path),
        "warmup_commands": warmup_commands,
        "base_lap_commands": len(lap),
        "scored_command_horizon": SCORED_COMMAND_HORIZON,
        "physics_fixed_delta_time_s": FIXED_DT_S,
        "physics_ticks_per_command": TICKS_PER_COMMAND,
        "track": "practice",
        "player": str(player_dir),
        "player_digest": None,
        "reference_gate_origin_unity_xz_m": gate_origin.tolist(),
        "reference_gate_forward_unity_xz": gate_direction.tolist(),
        "parameter_bounds": {name: PARAMETER_BOUNDS[i].tolist()
                             for i, name in enumerate(PARAMETER_NAMES)},
        "trials": [],
        }
        (args.output / "campaign.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        (args.output / "baseline_reference_gate.json").write_text(
            json.dumps({"origin_xz_m": gate_origin.tolist(),
                        "forward_xz": gate_direction.tolist(),
                        "gate_time_s": boundary_time}, indent=2) + "\n",
            encoding="utf-8")
    max_evaluations = int(args.max_evaluations)
    if max_evaluations < 1:
        raise ValueError("max-evaluations must be positive; continue a campaign explicitly")
    initial = _initial_design()
    for trial_number in range(len(trial_rows), max_evaluations):
        if trial_number < len(initial):
            params = initial[trial_number]
            proposal = {"method": "empirical_initial_design"}
        else:
            params, proposal = _propose_ei(trial_rows, args.seed, trial_number)
        trial_name = f"trial_{trial_number:03d}"
        trial_dir = args.output / "trials" / trial_name
        trial_dir.mkdir()
        profile = _make_profile(warmup, lap, params)
        (trial_dir / "command_profile.json").write_text(
            json.dumps(profile, separators=(",", ":")) + "\n",
            encoding="utf-8")
        result = _trial_score(
            trial_dir, params, gate_origin, gate_direction,
            warmup_commands, lap_path, player_dir, args.timeout_s)
        result["proposal"] = proposal
        (trial_dir / "trial.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8")
        trial_rows.append(result)

        is_best = (result["lap_valid"] and
                   (not any(row["lap_valid"] for row in trial_rows[:-1]) or
                    result["objective_s"] < min(
                        row["objective_s"] for row in trial_rows[:-1]
                        if row["lap_valid"])))
        if result["capture_path"]:
            current_capture = Path(result["capture_path"])
            if is_best:
                if best_capture and best_capture.exists():
                    best_capture.unlink()
                best_capture = current_capture
            elif current_capture.exists():
                current_capture.unlink()
        result["capture_path"] = str(best_capture) if is_best else None
        manifest["trials"] = [{
            "trial": row["trial"],
            "parameters": row["parameters"],
            "proposal": row["proposal"],
            "lap_valid": row["lap_valid"],
            "unity_status": row["unity_status"],
            "collisions_observed": row["collisions_observed"],
            "unity_lap_time_s": row.get("unity_lap_time_s"),
            "objective_s": row["objective_s"],
            "peak_body_speed_mps": row.get("peak_body_speed_mps"),
            "p95_abs_wheel_slip": row.get("p95_abs_wheel_slip"),
            "best_capture_path": row.get("best_capture_path"),
        } for row in trial_rows]
        manifest["best_lap_time_s"] = min(
            (float(row["unity_lap_time_s"]) for row in trial_rows
             if row["lap_valid"]), default=None)
        manifest["status"] = (
            "target_reached" if manifest["best_lap_time_s"] is not None
            and manifest["best_lap_time_s"] < args.target_s else "running")
        (args.output / "campaign.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({
            "trial": trial_name,
            "parameters": result["parameters"],
            "lap_valid": result["lap_valid"],
            "unity_status": result["unity_status"],
            "collision_count": result["collisions_observed"],
            "lap_time_s": result.get("unity_lap_time_s"),
            "progress": result.get("reference_lap_progress_fraction"),
            "peak_speed_mps": result.get("peak_body_speed_mps"),
            "p95_wheel_slip": result.get("p95_abs_wheel_slip"),
            "best_so_far_s": manifest["best_lap_time_s"],
            "wall_time_s": result["wall_time_s"],
        }, separators=(",", ":")), flush=True)
        if manifest["status"] == "target_reached":
            break

    if manifest["best_lap_time_s"] is None:
        manifest["status"] = "search_budget_exhausted_without_valid_lap"
    elif manifest["best_lap_time_s"] >= args.target_s:
        manifest["status"] = "search_budget_exhausted_above_target"
    (args.output / "campaign.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return 0 if manifest["status"] == "target_reached" else 3


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--parity-report", type=Path, default=DEFAULT_PARITY)
    parser.add_argument("--reference-capture", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--player-dir", type=Path, default=DEFAULT_PLAYER)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-s", type=float, default=5.0)
    parser.add_argument("--max-evaluations", type=int, default=20)
    parser.add_argument("--resume", action="store_true",
                        help="Continue the existing campaign to max-evaluations total.")
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=20261014)
    args = parser.parse_args()
    if not (0.0 < args.target_s < 20.0 and args.timeout_s > 0.0):
        parser.error("target and simulator timeout must be positive")
    try:
        return run_campaign(args)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
