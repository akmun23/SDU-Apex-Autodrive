#!/usr/bin/env python3
"""Build the handoff's empirical vehicle-response atlas from frozen datasets."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

COM_X_M = 0.15532
WHEELBASE_M = 0.324
TRACK_WIDTH_M = 0.236
WHEEL_RADIUS_M = 0.059
DT_S = 0.025

DEFAULT_DATASET = (
    REPO_ROOT / "live_runs/derived_dynamics_learning_20260928/"
    "plant_teacher_race_domain_v1/cooldown_2s"
)
DEFAULT_THROTTLE = (
    REPO_ROOT / "live_runs/derived_dynamics_learning_20260928/"
    "throttle_surface_race_domain_v1"
)

TARGET_NAMES = (
    "ax_sim_body_mps2", "ay_sim_body_mps2", "yaw_acceleration_rps2",
    "next_u_com_mps", "next_v_com_mps", "next_yaw_rate_rps",
    "rear_left_wheel_acceleration_mps2",
    "rear_right_wheel_acceleration_mps2",
)


def _roll_mean(values: np.ndarray, width: int) -> np.ndarray:
    result = np.empty_like(values, dtype=np.float64)
    cumulative = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    positions = np.arange(1, len(values) + 1)
    starts = np.maximum(0, positions - width)
    counts = positions - starts
    result[:] = (cumulative[positions] - cumulative[starts]) / counts
    return result


def _range_stats(values: np.ndarray) -> dict[str, float | int]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {"count": 0}
    return {
        "count": int(len(finite)),
        "p01": float(np.quantile(finite, 0.01)),
        "p50": float(np.quantile(finite, 0.50)),
        "p90": float(np.quantile(finite, 0.90)),
        "p99": float(np.quantile(finite, 0.99)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def _finite_row_mean(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    count = np.sum(finite, axis=1)
    total = np.sum(np.where(finite, values, 0.0), axis=1)
    return np.divide(total, count, out=np.full(len(values), np.nan),
                     where=count > 0)


def _finite_mean(values: np.ndarray) -> float:
    finite = np.asarray(values)[np.isfinite(values)]
    return float(np.mean(finite)) if len(finite) else math.nan


def _mirror_symmetry_analysis(rows: list[dict[str, Any]],
                              high_steering_threshold_rad: float = 0.30
                              ) -> dict[str, Any]:
    """Compare sign-mirrored responses inside the same source run/condition."""
    paired: list[tuple[dict[str, Any], dict[str, Any]]] = []
    indexed: dict[tuple[Any, ...], dict[int, dict[str, Any]]] = {}
    for row in rows:
        steering = float(row["steering_command_rad"])
        if abs(steering) < 1e-8:
            continue
        key = (
            str(row["run_id"]), round(abs(steering), 6),
            round(float(row["throttle_start_norm"]), 6),
            round(float(row["throttle_target_norm"]), 6),
            int(row["window_index"]),
        )
        indexed.setdefault(key, {})[1 if steering > 0 else -1] = row
    for sides in indexed.values():
        if 1 in sides and -1 in sides:
            paired.append((sides[1], sides[-1]))

    response_parity = {
        "delta_u_rear_mps": "even",
        "mean_ax_sim_body_mps2": "even",
        "mean_wheel_surface_mps": "even",
        "mean_wheel_body_mismatch_mps": "even",
        "delta_v_com_mps": "odd",
        "delta_yaw_rate_rps": "odd",
        "mean_yaw_acceleration_rps2": "odd",
        "mean_ay_sim_body_mps2": "odd",
        "mean_roll_rad": "odd",
    }

    def summarize(selected: list[tuple[dict[str, Any], dict[str, Any]]]
                  ) -> dict[str, Any]:
        per_run: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
        for positive, negative in selected:
            per_run[str(positive["run_id"])].append((positive, negative))
        run_results: dict[str, Any] = {}
        for run_id, run_pairs in sorted(per_run.items()):
            metrics: dict[str, Any] = {}
            for name, parity in response_parity.items():
                positive = np.asarray([row[name] for row, _ in run_pairs],
                                      dtype=np.float64)
                negative = np.asarray([row[name] for _, row in run_pairs],
                                      dtype=np.float64)
                finite = np.isfinite(positive) & np.isfinite(negative)
                positive, negative = positive[finite], negative[finite]
                if not len(positive):
                    metrics[name] = {"pair_count": 0, "parity_rmse": None,
                                     "relative_parity_rmse": None}
                    continue
                residual = (positive - negative if parity == "even"
                            else positive + negative) / 2.0
                response_scale = (np.abs(positive) + np.abs(negative)) / 2.0
                scale_rms = float(np.sqrt(np.mean(response_scale ** 2)))
                metrics[name] = {
                    "pair_count": int(len(positive)),
                    "parity_rmse": float(np.sqrt(np.mean(residual ** 2))),
                    "relative_parity_rmse": (
                        float(np.sqrt(np.mean(residual ** 2)) / scale_rms)
                        if scale_rms > 1e-8 else None),
                }
            run_results[run_id] = {"pair_count": len(run_pairs),
                                   "metrics": metrics}
        metric_summary: dict[str, Any] = {}
        for name in response_parity:
            available = [result["metrics"][name]["relative_parity_rmse"]
                         for result in run_results.values()
                         if result["metrics"][name]["relative_parity_rmse"]
                         is not None]
            metric_summary[name] = {
                "parity": response_parity[name],
                "macro_mean_relative_parity_rmse": (
                    float(np.mean(available)) if available else None),
                "run_min_relative_parity_rmse": (
                    float(np.min(available)) if available else None),
                "run_max_relative_parity_rmse": (
                    float(np.max(available)) if available else None),
                "independent_run_count": len(available),
            }
        return {
            "matched_pair_count": len(selected),
            "independent_run_count": len(per_run),
            "metrics": metric_summary,
            "per_run": run_results,
        }

    return {
        "pairing": (
            "same source run, absolute steering, start/target throttle, and "
            "response window; positive and negative steering are compared"),
        "even_channels": [name for name, parity in response_parity.items()
                          if parity == "even"],
        "odd_channels": [name for name, parity in response_parity.items()
                         if parity == "odd"],
        "all_steering": summarize(paired),
        "high_steering_threshold_abs_rad": high_steering_threshold_rad,
        "high_steering": summarize([
            pair for pair in paired
            if abs(float(pair[0]["steering_command_rad"]))
            >= high_steering_threshold_rad]),
        "interpretation": (
            "Descriptive mirror-parity check, not a causal or population-level "
            "claim; run-specific results are retained because only two source "
            "runs contain replicated full-surface conditions."),
    }


def _error_stats(actual: np.ndarray, expected: np.ndarray,
                 mask: np.ndarray) -> dict[str, float | int]:
    selected = mask & np.isfinite(actual) & np.isfinite(expected)
    if not np.any(selected):
        return {"count": 0}
    error = actual[selected] - expected[selected]
    correlation = (float(np.corrcoef(actual[selected], expected[selected])[0, 1])
                   if np.std(actual[selected]) and np.std(expected[selected])
                   else None)
    return {
        "count": int(len(error)),
        "mean_error": float(np.mean(error)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "correlation": correlation,
    }


def _per_sequence_derive(data: dict[str, np.ndarray]
                         ) -> tuple[dict[str, np.ndarray], np.ndarray]:
    frames = data["frames"].astype(np.float64, copy=False)
    rigid = data["simulator_rigid_state"].astype(np.float64, copy=False)
    acceleration = data["simulator_linear_acceleration"].astype(
        np.float64, copy=False)
    attitude = data["imu_attitude_frames"].astype(np.float64, copy=False)
    bounds = data["sequence_bounds"]
    count = len(frames)
    derived: dict[str, np.ndarray] = {
        "u_rear_mps": frames[:, 0].copy(),
        "v_rear_mps": frames[:, 1].copy(),
        "yaw_rate_rps": rigid[:, 12].copy(),
        "u_com_mps": rigid[:, 7].copy(),
        "v_com_mps": rigid[:, 8].copy(),
        "steering_actual_rad": frames[:, 3].copy(),
        "throttle_actual_norm": frames[:, 4].copy(),
        "rear_left_surface_speed_mps": frames[:, 5].copy(),
        "rear_right_surface_speed_mps": frames[:, 6].copy(),
        "steering_command_rad": frames[:, 7].copy(),
        "throttle_command_norm": frames[:, 8].copy(),
        "simulator_ax_body_mps2": acceleration[:, 0].copy(),
        "simulator_ay_body_mps2": acceleration[:, 1].copy(),
        "roll_rad": attitude[:, 0].copy(),
        "pitch_rad": attitude[:, 1].copy(),
        "roll_rate_rps": attitude[:, 2].copy(),
        "pitch_rate_rps": attitude[:, 3].copy(),
    }
    derived["speed_com_mps"] = np.hypot(derived["u_com_mps"],
                                        derived["v_com_mps"])
    derived["lateral_acceleration_proxy_mps2"] = (
        derived["u_rear_mps"] * derived["yaw_rate_rps"])
    derived["curvature_inv_m"] = (
        derived["yaw_rate_rps"]
        / np.maximum(np.abs(derived["u_rear_mps"]), 0.20))

    # Rear wheel contact-point longitudinal speeds include yaw about the rear
    # axle. The wheel speeds are already surface speeds R*omega.
    half_track = TRACK_WIDTH_M / 2.0
    local_vx_left = derived["u_rear_mps"] - derived["yaw_rate_rps"] * half_track
    local_vx_right = derived["u_rear_mps"] + derived["yaw_rate_rps"] * half_track
    derived["rear_left_wheel_body_mismatch_mps"] = (
        derived["rear_left_surface_speed_mps"] - local_vx_left)
    derived["rear_right_wheel_body_mismatch_mps"] = (
        derived["rear_right_surface_speed_mps"] - local_vx_right)
    derived["rear_wheel_body_mismatch_mps"] = 0.5 * (
        derived["rear_left_wheel_body_mismatch_mps"]
        + derived["rear_right_wheel_body_mismatch_mps"])
    for side, wheel, local_vx in (
            ("left", derived["rear_left_surface_speed_mps"], local_vx_left),
            ("right", derived["rear_right_surface_speed_mps"], local_vx_right)):
        slip = np.full(count, np.nan, dtype=np.float64)
        valid = np.abs(local_vx) >= 0.5
        slip[valid] = (wheel[valid] - local_vx[valid]) / local_vx[valid]
        derived[f"rear_{side}_longitudinal_slip_proxy"] = slip

    # Official Ackermann geometry; front tire lateral velocity is projected
    # into each steered wheel frame. These are kinematic slip proxies, not
    # simulated tire forces.
    center = derived["steering_actual_rad"]
    tangent = np.tan(center)
    left_delta = np.arctan2(2.0 * WHEELBASE_M * tangent,
                            2.0 * WHEELBASE_M + TRACK_WIDTH_M * tangent)
    right_delta = np.arctan2(2.0 * WHEELBASE_M * tangent,
                             2.0 * WHEELBASE_M - TRACK_WIDTH_M * tangent)
    front_x = WHEELBASE_M - COM_X_M
    for side, y_wheel, delta in (("left", half_track, left_delta),
                                 ("right", -half_track, right_delta)):
        wheel_vx = (derived["u_com_mps"]
                    - derived["yaw_rate_rps"] * y_wheel)
        wheel_vy = (derived["v_com_mps"]
                    + derived["yaw_rate_rps"] * front_x)
        longitudinal = wheel_vx * np.cos(delta) + wheel_vy * np.sin(delta)
        lateral = -wheel_vx * np.sin(delta) + wheel_vy * np.cos(delta)
        slip = np.full(count, np.nan, dtype=np.float64)
        valid = np.abs(longitudinal) >= 0.5
        slip[valid] = lateral[valid] / np.abs(longitudinal[valid])
        derived[f"front_{side}_lateral_slip_proxy"] = slip

    for side, local_vx in (("left", local_vx_left),
                           ("right", local_vx_right)):
        slip = np.full(count, np.nan, dtype=np.float64)
        valid = np.abs(local_vx) >= 0.5
        slip[valid] = (derived["v_rear_mps"][valid]
                       / np.abs(local_vx[valid]))
        derived[f"rear_{side}_lateral_slip_proxy"] = slip
    derived["front_lateral_slip_proxy"] = 0.5 * (
        derived["front_left_lateral_slip_proxy"]
        + derived["front_right_lateral_slip_proxy"])
    derived["rear_lateral_slip_proxy"] = 0.5 * (
        derived["rear_left_lateral_slip_proxy"]
        + derived["rear_right_lateral_slip_proxy"])

    # Derivatives and causal trailing command summaries are calculated inside
    # each verified sequence only; no reset/gap boundary is crossed.
    derived["steering_command_rate_radps"] = np.full(count, np.nan)
    derived["throttle_command_rate_per_s"] = np.full(count, np.nan)
    derived["steering_reversal_indicator"] = np.zeros(count, dtype=np.float32)
    derived["throttle_cut_indicator"] = np.zeros(count, dtype=np.float32)
    derived["yaw_acceleration_rps2"] = np.full(count, np.nan)
    derived["rear_left_wheel_acceleration_mps2"] = np.full(count, np.nan)
    derived["rear_right_wheel_acceleration_mps2"] = np.full(count, np.nan)
    for seconds in (0.100, 0.250, 0.500, 1.000, 2.000):
        key = int(round(seconds * 1000))
        derived[f"recent_steering_mean_{key}ms"] = np.full(count, np.nan)
        derived[f"recent_steering_integral_{key}ms_rad_s"] = np.full(count, np.nan)
        derived[f"recent_throttle_mean_{key}ms"] = np.full(count, np.nan)
        derived[f"recent_throttle_integral_{key}ms_s"] = np.full(count, np.nan)
    next_state = np.full((count, 3), np.nan, dtype=np.float64)

    for start_value, end_value in bounds:
        start, end = int(start_value), int(end_value)
        if end - start < 2:
            continue
        sl = slice(start, end)
        steering_command = derived["steering_command_rad"][sl]
        throttle_command = derived["throttle_command_norm"][sl]
        steer_rate = np.diff(steering_command, prepend=steering_command[0]) / DT_S
        throttle_rate = np.diff(throttle_command,
                                prepend=throttle_command[0]) / DT_S
        derived["steering_command_rate_radps"][sl] = steer_rate
        derived["throttle_command_rate_per_s"][sl] = throttle_rate
        reversal = np.zeros(end - start, dtype=np.float32)
        if len(reversal) > 2:
            reversal[2:] = ((steer_rate[1:-1] * steer_rate[2:] < 0.0)
                            & (np.abs(steer_rate[1:-1]) > 0.02)
                            & (np.abs(steer_rate[2:]) > 0.02))
        derived["steering_reversal_indicator"][sl] = reversal
        derived["throttle_cut_indicator"][sl] = (
            throttle_rate < -0.05).astype(np.float32)
        derived["yaw_acceleration_rps2"][start:end - 1] = (
            np.diff(derived["yaw_rate_rps"][sl]) / DT_S)
        for key, wheel_name in (("left", "rear_left_surface_speed_mps"),
                                ("right", "rear_right_surface_speed_mps")):
            target_name = f"rear_{key}_wheel_acceleration_mps2"
            derived[target_name][start:end - 1] = (
                np.diff(derived[wheel_name][sl]) / DT_S)
        next_state[start:end - 1, 0] = derived["u_com_mps"][start + 1:end]
        next_state[start:end - 1, 1] = derived["v_com_mps"][start + 1:end]
        next_state[start:end - 1, 2] = derived["yaw_rate_rps"][start + 1:end]
        for seconds in (0.100, 0.250, 0.500, 1.000, 2.000):
            width = int(round(seconds / DT_S))
            steer_mean = _roll_mean(steering_command, width)
            throttle_mean = _roll_mean(throttle_command, width)
            key = int(round(seconds * 1000))
            derived[f"recent_steering_mean_{key}ms"][sl] = steer_mean
            derived[f"recent_steering_integral_{key}ms_rad_s"][sl] = (
                steer_mean * np.minimum(np.arange(1, end - start + 1), width) * DT_S)
            derived[f"recent_throttle_mean_{key}ms"][sl] = throttle_mean
            derived[f"recent_throttle_integral_{key}ms_s"][sl] = (
                throttle_mean * np.minimum(np.arange(1, end - start + 1), width) * DT_S)

    targets = {
        "ax_sim_body_mps2": derived["simulator_ax_body_mps2"],
        "ay_sim_body_mps2": derived["simulator_ay_body_mps2"],
        "yaw_acceleration_rps2": derived["yaw_acceleration_rps2"],
        "next_u_com_mps": next_state[:, 0],
        "next_v_com_mps": next_state[:, 1],
        "next_yaw_rate_rps": next_state[:, 2],
        "rear_left_wheel_acceleration_mps2":
            derived["rear_left_wheel_acceleration_mps2"],
        "rear_right_wheel_acceleration_mps2":
            derived["rear_right_wheel_acceleration_mps2"],
    }
    return {**derived, **{f"target_{key}": value
                          for key, value in targets.items()}}, next_state


def _bin_table(condition_name: str, values: np.ndarray, edges: list[float],
               targets: dict[str, np.ndarray], frame_run_index: np.ndarray,
               base_mask: np.ndarray) -> list[dict[str, Any]]:
    edge_array = np.asarray(edges, dtype=np.float64)
    bin_ids = np.searchsorted(edge_array, values, side="right") - 1
    rows: list[dict[str, Any]] = []
    for bin_index in range(len(edge_array) - 1):
        in_bin = base_mask & np.isfinite(values) & (bin_ids == bin_index)
        if not np.any(in_bin):
            continue
        for target_name, target in targets.items():
            selected = in_bin & np.isfinite(target)
            if not np.any(selected):
                continue
            y = target[selected]
            selected_run_index = frame_run_index[selected]
            run_sums = np.bincount(selected_run_index, weights=y)
            run_counts = np.bincount(selected_run_index)
            run_means_array = run_sums[run_counts > 0] / run_counts[run_counts > 0]
            runs = np.unique(selected_run_index)
            rows.append({
                "conditioning_variable": condition_name,
                "bin_index": bin_index,
                "bin_low": float(edge_array[bin_index]),
                "bin_high": float(edge_array[bin_index + 1]),
                "target": target_name,
                "sample_count": int(len(y)),
                "independent_run_count": int(len(runs)),
                "mean": float(np.mean(y)),
                "median": float(np.median(y)),
                "std": float(np.std(y)),
                "q05": float(np.quantile(y, 0.05)),
                "q25": float(np.quantile(y, 0.25)),
                "q75": float(np.quantile(y, 0.75)),
                "q95": float(np.quantile(y, 0.95)),
                "run_mean_q05": (float(np.quantile(run_means_array, 0.05))
                                  if len(run_means_array) else None),
                "run_mean_q95": (float(np.quantile(run_means_array, 0.95))
                                  if len(run_means_array) else None),
            })
    return rows


def _two_dimensional_table(name: str, x: np.ndarray, x_edges: list[float],
                           y: np.ndarray, y_edges: list[float],
                           targets: dict[str, np.ndarray],
                           frame_run_index: np.ndarray,
                           base_mask: np.ndarray) -> list[dict[str, Any]]:
    x_edges_array = np.asarray(x_edges, dtype=np.float64)
    y_edges_array = np.asarray(y_edges, dtype=np.float64)
    x_bins = np.searchsorted(x_edges_array, x, side="right") - 1
    y_bins = np.searchsorted(y_edges_array, y, side="right") - 1
    rows: list[dict[str, Any]] = []
    for xi in range(len(x_edges_array) - 1):
        for yi in range(len(y_edges_array) - 1):
            cell = (base_mask & np.isfinite(x) & np.isfinite(y)
                    & (x_bins == xi) & (y_bins == yi))
            if not np.any(cell):
                continue
            for target_name, target in targets.items():
                selected = cell & np.isfinite(target)
                if not np.any(selected):
                    continue
                v = target[selected]
                run_ids = np.unique(frame_run_index[selected])
                rows.append({
                    "map": name,
                    "x_bin": xi,
                    "x_low": float(x_edges_array[xi]),
                    "x_high": float(x_edges_array[xi + 1]),
                    "y_bin": yi,
                    "y_low": float(y_edges_array[yi]),
                    "y_high": float(y_edges_array[yi + 1]),
                    "target": target_name,
                    "sample_count": int(len(v)),
                    "independent_run_count": int(len(run_ids)),
                    "mean": float(np.mean(v)),
                    "median": float(np.median(v)),
                    "std": float(np.std(v)),
                    "q10": float(np.quantile(v, 0.10)),
                    "q90": float(np.quantile(v, 0.90)),
                })
    return rows


def _write_gzip_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with gzip.open(path, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _frame_semantics(data: dict[str, np.ndarray],
                     derived: dict[str, np.ndarray]) -> dict[str, Any]:
    frame = data["frames"]
    rigid = data["simulator_rigid_state"]
    accel = data["simulator_linear_acceleration"]
    sensors = data["sensor_frames"]
    mask = (np.isfinite(frame).all(axis=1)
            & np.isfinite(rigid).all(axis=1)
            & np.isfinite(accel).all(axis=1)
            & np.isfinite(sensors).all(axis=1))
    return {
        "verified_assumption": (
            "simulator rigid-state velocity and acceleration vectors are in "
            "the vehicle/body axes; linear velocity is at the COM, while the "
            "production odom lateral twist is rear-axle referenced"),
        "u_com_vs_u_rear_odom_mps": _error_stats(rigid[:, 7], frame[:, 0], mask),
        "v_com_vs_v_rear_plus_com_offset_times_yaw_mps": _error_stats(
            rigid[:, 8], frame[:, 1] + COM_X_M * frame[:, 2], mask),
        "sim_yaw_rate_vs_odom_yaw_rate_rps": _error_stats(
            rigid[:, 12], frame[:, 2], mask),
        "sim_ax_vs_imu_ax_mps2": _error_stats(accel[:, 0], sensors[:, 4], mask),
        "sim_ay_vs_imu_ay_mps2": _error_stats(accel[:, 1], sensors[:, 5], mask),
        "sim_ax_rotated_by_yaw_vs_imu_ax_mps2": None,
        "wheel_surface_definition": (
            "rear encoder surface speed R*omega; local body forward speed "
            "corrected for yaw and +/- track/2 before slip proxy"),
        "front_lateral_slip_definition": (
            "kinematic wheel-frame lateral/absolute-longitudinal velocity; "
            "Ackermann angles from the official wheelbase/track geometry"),
    }


def _throttle_surface_analysis(dataset_dir: Path, output_dir: Path
                               ) -> dict[str, Any]:
    npz_path = dataset_dir / "throttle_surface_sequences.npz"
    manifest_path = dataset_dir / "manifest.json"
    if not npz_path.is_file() or not manifest_path.is_file():
        return {"available": False, "reason": "throttle surface archive absent"}
    metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    z = np.load(npz_path, allow_pickle=False)
    if (metadata.get("dataset_role")
            != "throttle_surface_race_domain_response_analysis"):
        raise ValueError("throttle atlas requires the capped race-domain view")
    bounds = z["sequence_bounds"]
    body = z["body_state"]
    surface = z["encoder_surface_mps_100ms"]
    feedback = z["actuator_feedback"]
    attitude = z["imu_roll_pitch_rad"]
    time_from_stimulus = z["time_from_stimulus_s"][:, 0]
    domain_speed = z["frame_domain_speed_mps"].astype(np.float64, copy=False)
    if (float(z["domain_speed_cap_mps"][0]) != 12.0
            or not np.isfinite(domain_speed).all()
            or np.any(domain_speed > 12.0)):
        raise ValueError("throttle response archive contains out-of-domain rows")
    debug = z["bridge_debug_telemetry"]
    packet_sequence = z["packet_sequence"].reshape(-1)
    debug_names = metadata["bridge_debug_telemetry_names"]
    debug_index = {name: i for i, name in enumerate(debug_names)}
    ax = debug[:, debug_index["simulator_acceleration_x_mps2"]]
    ay = debug[:, debug_index["simulator_acceleration_y_mps2"]]
    run_ids = z["sequence_run_id"].astype(str)

    # The encoder is the only available wheel-rotation measurement. Validate
    # both its source-stamp identity and the 100 ms derivative against the
    # independent per-packet simulator telemetry before interpreting slip.
    ros_angle = z["encoder_position_rad"].astype(np.float64, copy=False)
    simulator_angle = debug[:, [
        debug_index["simulator_encoder_angle_left_rad"],
        debug_index["simulator_encoder_angle_right_rad"],
    ]]
    angle_error = ros_angle - simulator_angle
    consecutive = np.zeros(len(ros_angle), dtype=bool)
    for start_value, end_value in bounds:
        start, end = int(start_value), int(end_value)
        if end - start > 1:
            consecutive[start + 1:end] = (
                np.diff(packet_sequence[start:end]) == 1)
    interval_error = np.diff(ros_angle, axis=0) - np.diff(
        simulator_angle, axis=0)
    valid_interval = consecutive[1:]
    interval_error_valid = interval_error[valid_interval]

    ros_surface = np.full_like(ros_angle, np.nan, dtype=np.float64)
    simulator_surface = np.full_like(simulator_angle, np.nan,
                                      dtype=np.float64)
    for start_value, end_value in bounds:
        start, end = int(start_value), int(end_value)
        for index in range(start + 4, end):
            if packet_sequence[index] - packet_sequence[index - 4] != 4:
                continue
            ros_surface[index] = WHEEL_RADIUS_M * (
                ros_angle[index] - ros_angle[index - 4]) / (4 * DT_S)
            simulator_surface[index] = WHEEL_RADIUS_M * (
                simulator_angle[index] - simulator_angle[index - 4]) / (
                    4 * DT_S)
    surface_error = ros_surface - simulator_surface
    surface_valid = np.isfinite(surface_error)
    encoder_measurement_audit = {
        "source_archive": str(manifest_path),
        "source_stamp_aligned_archive": bool(
            "encoder_time_policy" in metadata
            and metadata.get("encoder_source_stamp_match_fraction")
            and all(all(float(fraction) == 1.0
                        for fraction in run_fractions.values())
                    for run_fractions in metadata[
                        "encoder_source_stamp_match_fraction"].values())),
        "ros_vs_same_packet_simulator_angle_rad": {
            side: _range_stats(np.abs(angle_error[:, index]))
            for index, side in enumerate(("left", "right"))
        },
        "consecutive_packets_checked": int(np.count_nonzero(valid_interval)),
        "max_abs_ros_vs_simulator_angle_increment_error_rad": (
            float(np.max(np.abs(interval_error_valid)))
            if interval_error_valid.size else None),
        "ros_vs_simulator_100ms_surface_speed_error_mps": {
            "count": int(np.count_nonzero(surface_valid)),
            "rmse": float(np.sqrt(np.mean(surface_error[surface_valid] ** 2)))
                if np.any(surface_valid) else None,
            "max_abs": float(np.max(np.abs(surface_error[surface_valid])))
                if np.any(surface_valid) else None,
        },
        "interpretation": (
            "Exact comparison uses source-stamp-aligned ROS wheel angles and "
            "independent bridge simulator encoder angles; derived speed uses "
            "four consecutive 25 ms packet intervals, without receipt-time dt."),
    }

    # Keep both independently sourced wheel-speed derivations available for
    # each response window. A mismatch here makes the associated window
    # unsuitable for a quantitative wheel-slip conclusion.
    debug_surface_mean = _finite_row_mean(simulator_surface)
    details: list[dict[str, Any]] = []
    window_specs = ((0.0, 0.100), (0.100, 0.250), (0.250, 0.500),
                    (0.500, 1.000), (1.000, 2.000), (2.000, 4.000),
                    (4.000, 8.000))
    sequences = metadata["sequences"]
    packet_contiguous = z["sequence_packet_contiguous"].astype(bool, copy=False)
    skipped_packet_gap = 0
    skipped_baseline = 0
    incomplete_window_count = 0
    for sequence_index, (start_value, end_value) in enumerate(bounds):
        start, end = int(start_value), int(end_value)
        if not packet_contiguous[sequence_index]:
            skipped_packet_gap += 1
            continue
        if end <= start:
            skipped_baseline += 1
            continue
        event = sequences[sequence_index]
        t = time_from_stimulus[start:end]
        state = body[start:end]
        local_domain = domain_speed[start:end] <= 12.0
        baseline = (t >= -3.75) & (t <= -0.25) & local_domain
        if np.count_nonzero(baseline) < 20:
            baseline = ((t < 0.0) & (t >= -4.0) & local_domain)
        if np.count_nonzero(baseline) < 20:
            skipped_baseline += 1
            continue
        base = np.mean(state[baseline], axis=0)
        base_feedback = np.mean(feedback[start:end][baseline], axis=0)
        start_throttle = float(event["throttle_start_norm"])
        target_throttle = float(event["throttle_end_norm"])
        steer = float(event["steering_command_rad"])
        wheel_body_mean = _finite_row_mean(surface[start:end])
        expected_rear_speed = state[:, 0]
        wheel_acceleration = np.full_like(surface[start:end], np.nan,
                                          dtype=np.float64)
        yaw_acceleration = np.full(end - start, np.nan, dtype=np.float64)
        if end - start > 1:
            wheel_acceleration[1:] = np.diff(surface[start:end], axis=0) / DT_S
            yaw_acceleration[1:] = np.diff(state[:, 2]) / DT_S
        for window_index, (lower, upper) in enumerate(window_specs):
            in_window = (t >= lower) & (t < upper) & local_domain
            required_samples = int(round((upper - lower) / DT_S))
            if np.count_nonzero(in_window) != required_samples:
                incomplete_window_count += 1
                continue
            local_state = state[in_window]
            local_feedback = feedback[start:end][in_window]
            local_roll_pitch = attitude[start:end][in_window]
            local_wheel = wheel_body_mean[in_window]
            local_encoder_surface_error = surface_error[start:end][in_window]
            local_wheel_accel = wheel_acceleration[in_window]
            delta_state = np.mean(local_state, axis=0) - base
            actual_throttle = local_feedback[:, 1]
            actual_steering = local_feedback[:, 0]
            body_wheel = local_wheel - expected_rear_speed[in_window]
            details.append({
                "sequence_index": sequence_index,
                "run_id": str(run_ids[sequence_index]),
                "reset_id": event.get("reset_id"),
                "phase_index": int(event["phase_index"]),
                "condition_label": event.get("label"),
                "replicate_index": int(event["replicate_index"]),
                "steering_command_rad": steer,
                "throttle_start_norm": start_throttle,
                "throttle_target_norm": target_throttle,
                "throttle_delta_norm": target_throttle - start_throttle,
                "window_index": window_index,
                "window_start_s": lower,
                "window_end_s": upper,
                "sample_count": int(np.count_nonzero(in_window)),
                "baseline_u_rear_mps": float(base[0]),
                "baseline_v_rear_mps": float(base[1]),
                "baseline_yaw_rate_rps": float(base[2]),
                "delta_u_rear_mps": float(delta_state[0]),
                "delta_v_rear_mps": float(delta_state[1]),
                "delta_v_com_mps": float(
                    delta_state[1] + COM_X_M * delta_state[2]),
                "delta_yaw_rate_rps": float(delta_state[2]),
                "mean_yaw_acceleration_rps2": _finite_mean(
                    yaw_acceleration[in_window]),
                "mean_ax_sim_body_mps2": _finite_mean(ax[start:end][in_window]),
                "mean_ay_sim_body_mps2": _finite_mean(ay[start:end][in_window]),
                "mean_rear_wheel_acceleration_mps2": (
                    _finite_mean(local_wheel_accel)),
                "mean_wheel_surface_mps": _finite_mean(local_wheel),
                "mean_debug_wheel_surface_mps": (
                    _finite_mean(debug_surface_mean[start:end][in_window])),
                "max_abs_encoder_surface_consistency_error_mps": (
                    float(np.nanmax(np.abs(local_encoder_surface_error)))
                    if np.isfinite(local_encoder_surface_error).any()
                    else math.nan),
                "mean_wheel_body_mismatch_mps": (
                    _finite_mean(body_wheel)),
                "mean_throttle_feedback_norm": _finite_mean(actual_throttle),
                "target_throttle_abs_error_norm": float(
                    abs(np.nanmean(actual_throttle) - target_throttle)),
                "mean_steering_feedback_rad": _finite_mean(actual_steering),
                "steering_abs_error_rad": float(abs(
                    _finite_mean(actual_steering) - steer)),
                "mean_tilt_magnitude_rad": float(np.nanmean(
                    np.hypot(local_roll_pitch[:, 0], local_roll_pitch[:, 1]))),
                "mean_roll_rad": _finite_mean(local_roll_pitch[:, 0]),
                "max_tilt_magnitude_rad": float(np.nanmax(
                    np.hypot(local_roll_pitch[:, 0], local_roll_pitch[:, 1]))),
                "baseline_throttle_feedback_norm": float(base_feedback[1]),
                "baseline_throttle_abs_error_norm": float(
                    abs(base_feedback[1] - start_throttle)),
            })
    z.close()

    csv_path = output_dir / "throttle_surface_time_windows.csv.gz"
    _write_gzip_csv(csv_path, details)
    grouped: dict[tuple[float, float, float, int], list[dict[str, Any]]] = defaultdict(list)
    for row in details:
        key = (row["steering_command_rad"], row["throttle_start_norm"],
               row["throttle_target_norm"], row["window_index"])
        grouped[key].append(row)
    aggregate = []
    for key, rows in sorted(grouped.items()):
        field_names = (
            "delta_u_rear_mps", "delta_v_rear_mps", "delta_v_com_mps",
            "delta_yaw_rate_rps", "mean_yaw_acceleration_rps2",
            "mean_ax_sim_body_mps2", "mean_ay_sim_body_mps2",
            "mean_wheel_surface_mps", "mean_wheel_body_mismatch_mps",
            "mean_debug_wheel_surface_mps",
            "max_abs_encoder_surface_consistency_error_mps",
            "mean_rear_wheel_acceleration_mps2",
            "mean_throttle_feedback_norm", "mean_tilt_magnitude_rad",
        )
        output = {
            "steering_command_rad": key[0],
            "throttle_start_norm": key[1],
            "throttle_target_norm": key[2],
            "window_index": key[3],
            "window_start_s": rows[0]["window_start_s"],
            "window_end_s": rows[0]["window_end_s"],
            "condition_replicates": len(rows),
            "independent_run_count": len({row["run_id"] for row in rows}),
            "feedback_tracking_fraction": float(np.mean([
                row["target_throttle_abs_error_norm"] <= 0.05
                and row["steering_abs_error_rad"] <= 0.02 for row in rows])),
        }
        for field in field_names:
            values = np.asarray([row[field] for row in rows], dtype=np.float64)
            values = values[np.isfinite(values)]
            output[field + "_mean"] = float(np.mean(values)) if len(values) else None
            output[field + "_median"] = float(np.median(values)) if len(values) else None
            output[field + "_min"] = float(np.min(values)) if len(values) else None
            output[field + "_max"] = float(np.max(values)) if len(values) else None
        aggregate.append(output)
    return {
        "available": True,
        "source_manifest": str(manifest_path),
        "domain_speed_cap_mps": 12.0,
        "zero_throttle_semantics": (
            "active brake torque in this simulator, not coast; the wire protocol "
            "has no separate coast command"),
        "condition_count": int(len(bounds)),
        "conditions_skipped_for_packet_gap": skipped_packet_gap,
        "conditions_skipped_for_missing_baseline": skipped_baseline,
        "incomplete_time_windows_excluded": incomplete_window_count,
        "conditions_with_complete_response_windows": int(len({
            row["sequence_index"] for row in details})),
        "sequence_count": len(bounds),
        "encoder_measurement_audit": encoder_measurement_audit,
        "window_count": len(window_specs),
        "window_specs_s": [list(row) for row in window_specs],
        "trajectory_window_rows": len(details),
        "time_window_csv_gz": str(csv_path),
        "condition_window_aggregates": aggregate,
        "mirror_symmetry": _mirror_symmetry_analysis(details),
        "feedback_tracking_by_window": {
            str(index): {
                "window_s": list(window_specs[index]),
                "trajectory_count": sum(row["window_index"] == index
                                         for row in details),
                "throttle_within_0p05_fraction": float(np.mean([
                    row["target_throttle_abs_error_norm"] <= 0.05
                    for row in details if row["window_index"] == index])),
                "steering_within_0p02_rad_fraction": float(np.mean([
                    row["steering_abs_error_rad"] <= 0.02
                    for row in details if row["window_index"] == index])),
            }
            for index in range(len(window_specs))
        },
        "baseline_feedback_tracking_fraction": (
            float(np.mean([row["baseline_throttle_abs_error_norm"] <= 0.05
                           for row in details])) if details else None),
        "target_feedback_tracking_fraction": (
            float(np.mean([row["target_throttle_abs_error_norm"] <= 0.05
                           for row in details])) if details else None),
        "interpretation": (
            "Reset-isolated prescribed throttle trajectories. Interpret response "
            "conditional on baseline state, steering and measured actuator "
            "feedback; replicate rows are not independent time samples."),
    }


def build(dataset_dir: Path, throttle_dir: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = dataset_dir / "openplane_dynamics.npz"
    manifest_path = dataset_dir / "manifest.json"
    data_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if data_manifest.get("dataset_role") != "race_domain_training_and_evaluation":
        raise ValueError("response atlas requires the clean race-domain dataset view")
    if float(data_manifest.get("speed_cap_mps", -1.0)) != 12.0:
        raise ValueError("response atlas requires a 12 m/s race-domain cap")
    z = np.load(npz_path, allow_pickle=False)
    required = ("frames", "sensor_frames", "imu_attitude_frames",
                "simulator_rigid_state", "simulator_linear_acceleration",
                "sequence_bounds", "sequence_run_index", "run_ids", "run_splits",
                "frame_domain_speed_mps")
    missing = [name for name in required if name not in z.files]
    if missing:
        raise ValueError("dataset lacks required schema-7 arrays: " + ", ".join(missing))
    data = {name: z[name] for name in required}
    data["frame_run_index"] = z["frame_run_index"]
    z.close()
    derived, _ = _per_sequence_derive(data)
    frames = data["frames"]
    race_speed = data["frame_domain_speed_mps"].astype(np.float64, copy=False)
    if (not np.isfinite(race_speed).all() or np.any(race_speed < 0.0)
            or np.any(race_speed > 12.0)):
        raise ValueError("race-domain atlas rows violate the declared speed cap")
    run_ids = data["run_ids"].astype(str)
    frame_run = data["frame_run_index"].astype(np.int32, copy=False)
    split_for_run = data["run_splits"].astype(str)
    represented_run_indices = np.unique(frame_run)
    represented_splits = split_for_run[represented_run_indices]
    run_split_per_frame = split_for_run[frame_run]
    truth = data["simulator_rigid_state"]
    accel = data["simulator_linear_acceleration"]
    base_valid = ((race_speed <= 12.0)
                  & np.isfinite(truth).all(axis=1)
                  & np.isfinite(accel).all(axis=1)
                  & np.isfinite(frames).all(axis=1)
                  & np.isin(run_split_per_frame, ("train", "validation")))
    targets = {name: derived[f"target_{name}"] for name in TARGET_NAMES}

    condition_specs = {
        "speed_com_mps": [-math.inf, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                           11, 12, math.inf],
        "steering_actual_rad": [-math.inf, -0.524, -0.45, -0.40, -0.35,
                                -0.30, -0.25, -0.20, -0.15, -0.10, -0.05,
                                0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30,
                                0.35, 0.40, 0.45, 0.524, math.inf],
        "throttle_actual_norm": [-math.inf, -1, -0.75, -0.5, -0.25, 0,
                                 0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
                                 0.60, 0.70, 0.80, 0.90, 1, math.inf],
        "rear_wheel_body_mismatch_mps": [-math.inf, -20, -10, -6, -4, -2,
                                          -1, -0.5, 0, 0.5, 1, 2, 4, 6,
                                          10, 20, math.inf],
        "lateral_acceleration_proxy_mps2": [-math.inf, -20, -15, -10, -8,
                                             -6, -4, -2, 0, 2, 4, 6, 8,
                                             10, 15, 20, math.inf],
        "recent_throttle_mean_500ms": [-math.inf, -1, 0, .1, .2, .3, .4,
                                       .5, .6, .7, .8, .9, 1, math.inf],
        "recent_steering_mean_500ms": [-math.inf, -.524, -.4, -.3, -.2, -.1,
                                       0, .1, .2, .3, .4, .524, math.inf],
    }
    table_rows: list[dict[str, Any]] = []
    for name, edges in condition_specs.items():
        table_rows.extend(_bin_table(
            name, derived[name], edges, targets, frame_run, base_valid))

    speed_edges = [-math.inf, 0, 2, 4, 6, 8, 9, 10, 11, 12, math.inf]
    abs_steering = np.abs(derived["steering_actual_rad"])
    abs_lateral = np.abs(derived["simulator_ay_body_mps2"])
    lat_demand_edges = [-math.inf, 0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15,
                        20, math.inf]
    steer_abs_edges = [-math.inf, 0, .05, .10, .15, .20, .25, .30, .35,
                       .40, .45, .50, .524, math.inf]
    throttle_edges = [-math.inf, -1, -.5, 0, .1, .2, .3, .4, .5, .6,
                       .7, .8, .9, 1, math.inf]
    map_rows = []
    map_rows.extend(_two_dimensional_table(
        "speed_x_abs_steering", derived["speed_com_mps"], speed_edges,
        abs_steering, steer_abs_edges,
        {"ax": targets["ax_sim_body_mps2"],
         "ay": targets["ay_sim_body_mps2"],
         "yaw_acceleration": targets["yaw_acceleration_rps2"],
         "mean_rear_longitudinal_slip": derived["rear_wheel_body_mismatch_mps"]},
        frame_run, base_valid))
    map_rows.extend(_two_dimensional_table(
        "speed_x_abs_lateral_acceleration", derived["speed_com_mps"],
        speed_edges, abs_lateral, lat_demand_edges,
        {"ax": targets["ax_sim_body_mps2"],
         "wheel_body_mismatch": derived["rear_wheel_body_mismatch_mps"],
         "rear_left_wheel_acceleration": targets[
             "rear_left_wheel_acceleration_mps2"]}, frame_run, base_valid))
    map_rows.extend(_two_dimensional_table(
        "throttle_x_abs_lateral_acceleration", derived["throttle_actual_norm"],
        throttle_edges, abs_lateral, lat_demand_edges,
        {"ax": targets["ax_sim_body_mps2"],
         "wheel_body_mismatch": derived["rear_wheel_body_mismatch_mps"],
         "wheel_acceleration": targets["rear_left_wheel_acceleration_mps2"]},
        frame_run, base_valid))

    table_path = output_dir / "response_atlas_conditioned_bins.csv.gz"
    map_path = output_dir / "response_atlas_2d_maps.csv.gz"
    _write_gzip_csv(table_path, table_rows)
    _write_gzip_csv(map_path, map_rows)

    regimes = {}
    for name, mask in {
        "speed_0_3mps": derived["speed_com_mps"] < 3,
        "speed_3_5mps": ((derived["speed_com_mps"] >= 3)
                          & (derived["speed_com_mps"] < 5)),
        "speed_5_7mps": ((derived["speed_com_mps"] >= 5)
                          & (derived["speed_com_mps"] < 7)),
        "speed_7_9mps": ((derived["speed_com_mps"] >= 7)
                          & (derived["speed_com_mps"] < 9)),
        "speed_9_10mps": ((derived["speed_com_mps"] >= 9)
                           & (derived["speed_com_mps"] < 10)),
        "speed_10_11mps": ((derived["speed_com_mps"] >= 10)
                            & (derived["speed_com_mps"] < 11)),
        "speed_11_12mps": ((derived["speed_com_mps"] >= 11)
                            & (derived["speed_com_mps"] <= 12)),
        "steering_abs_ge_0_30rad": abs_steering >= .30,
        "steering_abs_ge_0_40rad": abs_steering >= .40,
        "wheel_mismatch_abs_ge_1mps": (
            np.abs(derived["rear_wheel_body_mismatch_mps"]) >= 1.0),
        "rear_longitudinal_slip_abs_ge_0_15": (
            (np.abs(derived["rear_left_longitudinal_slip_proxy"]) >= .15)
            | (np.abs(derived["rear_right_longitudinal_slip_proxy"]) >= .15)),
        "steering_reversal": derived["steering_reversal_indicator"] > 0,
        "throttle_cut": derived["throttle_cut_indicator"] > 0,
    }.items():
        selected = base_valid & mask
        regimes[name] = {
            "sample_count": int(np.count_nonzero(selected)),
            "independent_run_count": int(len(np.unique(frame_run[selected]))),
        }

    frame_semantics = _frame_semantics(data, derived)
    report = {
        "schema_version": 1,
        "created_utc": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).isoformat(),
        "dataset_npz": str(npz_path),
        "dataset_manifest": str(manifest_path),
        "dataset_manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()).hexdigest(),
        "fixed_dt_s": DT_S,
        "sample_count": int(len(frames)),
        "analysis_splits_included": ["train", "validation"],
        "analysis_sample_count": int(np.count_nonzero(base_valid)),
        "sequence_count": int(len(data["sequence_bounds"])),
        "run_count": int(len(represented_run_indices)),
        "run_split_counts": {
            split: int(np.count_nonzero(represented_splits == split))
            for split in np.unique(represented_splits)},
        "label_frame_audit": frame_semantics,
        "dataset_role": data_manifest["dataset_role"],
        "speed_cap_mps": 12.0,
        "zero_throttle_semantics": (
            "active brake torque; no separate coast command is observable"),
        "support": {
            name: _range_stats(derived[name]) for name in (
                "u_rear_mps", "v_rear_mps", "u_com_mps", "v_com_mps",
                "yaw_rate_rps", "speed_com_mps", "steering_actual_rad",
                "throttle_actual_norm", "steering_command_rad",
                "throttle_command_norm", "rear_left_surface_speed_mps",
                "rear_right_surface_speed_mps",
                "rear_wheel_body_mismatch_mps",
                "rear_left_longitudinal_slip_proxy",
                "rear_right_longitudinal_slip_proxy",
                "front_lateral_slip_proxy", "rear_lateral_slip_proxy",
                "simulator_ax_body_mps2", "simulator_ay_body_mps2",
                "yaw_acceleration_rps2", "curvature_inv_m", "roll_rad",
                "pitch_rad", "roll_rate_rps", "pitch_rate_rps")},
        "regimes": regimes,
        "conditioned_bin_rows": len(table_rows),
        "conditioned_bin_csv_gz": str(table_path),
        "two_dimensional_map_rows": len(map_rows),
        "two_dimensional_map_csv_gz": str(map_path),
        "targets": list(TARGET_NAMES),
        "conditioning_dimensions": list(condition_specs),
        "two_dimensional_maps": [
            "speed_x_abs_steering", "speed_x_abs_lateral_acceleration",
            "throttle_x_abs_lateral_acceleration"],
        "caveats": [
            "Rows are correlated simulator samples; independent-run counts are reported per bin.",
            "Wheel-slip and tire-slip fields are kinematic proxies, not hidden simulator tire forces.",
            "Front wheel rotational speeds and per-wheel normal/tire forces are not logged.",
            "High quantiles of pooled rows are descriptive; run-level mean quantiles are included in bin tables.",
        ],
    }
    report["throttle_surface"] = _throttle_surface_analysis(
        throttle_dir, output_dir)
    report_path = output_dir / "response_atlas.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--throttle-dir", type=Path, default=DEFAULT_THROTTLE)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = build(args.dataset_dir, args.throttle_dir, args.output_dir)
    except (OSError, ValueError, KeyError) as exc:
        print(f"response atlas failed: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 2
    print(json.dumps({
        "samples": report["sample_count"],
        "sequences": report["sequence_count"],
        "conditioned_bin_rows": report["conditioned_bin_rows"],
        "throttle_surface_window_rows": report["throttle_surface"].get(
            "trajectory_window_rows"),
        "label_frame_audit": report["label_frame_audit"],
    }, indent=2))
    print(f"wrote {args.output_dir / 'response_atlas.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
