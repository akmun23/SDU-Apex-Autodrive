#!/usr/bin/env python3
"""Test whether IMU roll and throttle transients explain rear-wheel slip proxies.

Reads only existing Explore bags. It does not infer per-wheel tire force: the
wheel/ground speed difference is an encoder-based longitudinal-slip proxy,
with its conversion calibrated on steady, nearly straight training samples.
Whole runs reserved for the experiment fold and original test/final-test splits
are excluded.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import analyze_open_plane_dynamics as analysis  # noqa: E402
from tools import evaluate_open_plane_body_dynamics as body  # noqa: E402

ALIGNMENT_NS = 30_000_000
ENCODER_WINDOW_S = 0.100
MIN_SPEED_MPS = 2.0
HIGH_STEERING_RAD = 0.42


def _rpy(message: Any) -> tuple[float, float, float] | None:
    q = message.orientation
    values = (float(q.x), float(q.y), float(q.z), float(q.w))
    if not all(math.isfinite(value) for value in values):
        return None
    norm = math.sqrt(sum(value * value for value in values))
    if norm < 1.0e-8:
        return None
    x, y, z, w = (value / norm for value in values)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def _read_attitudes(path: Path) -> tuple[
        list[tuple[int, float, float, float, float, float, float]],
        list[tuple[float, float, float]],
        list[tuple[float, float, float]], int]:
    uri = path.resolve().as_uri() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        topics = analysis._topic_map(connection)
        if analysis.IMU not in topics or analysis.ODOM not in topics:
            raise ValueError("IMU or odometry topic is missing")
        imu_rows = []
        covariance = []
        for receipt_ns, message in analysis._messages(connection, topics, analysis.IMU):
            if len(message.orientation_covariance) and message.orientation_covariance[0] < 0:
                continue
            angles = _rpy(message)
            if angles is None:
                continue
            angular_velocity = (
                float(message.angular_velocity.x),
                float(message.angular_velocity.y),
                float(message.angular_velocity.z))
            if not all(math.isfinite(value) for value in angular_velocity):
                continue
            imu_rows.append((receipt_ns, *angles, *angular_velocity))
            covariance.append(float(message.orientation_covariance[0]))
        odom_rows = []
        for receipt_ns, message in analysis._messages(connection, topics, analysis.ODOM):
            angles = _rpy(message.pose.pose)
            if angles is not None:
                angular_velocity = (
                    float(message.twist.twist.angular.x),
                    float(message.twist.twist.angular.y),
                    float(message.twist.twist.angular.z))
                if all(math.isfinite(value) for value in angular_velocity):
                    odom_rows.append((receipt_ns, *angles, *angular_velocity))
    finally:
        connection.close()

    odom_times = [row[0] for row in odom_rows]
    paired = []
    angular_velocity_differences = []
    for imu in imu_rows:
        index = bisect.bisect_left(odom_times, imu[0])
        candidates = [i for i in (index - 1, index) if 0 <= i < len(odom_rows)]
        if not candidates:
            continue
        match = min(candidates, key=lambda i: abs(odom_rows[i][0] - imu[0]))
        if abs(odom_rows[match][0] - imu[0]) <= ALIGNMENT_NS:
            odom = odom_rows[match]
            paired.append((imu, odom))
            angular_velocity_differences.append(tuple(
                imu[4 + axis] - odom[4 + axis] for axis in range(3)))
    differences = [
        (math.atan2(math.sin(a[1] - b[1]), math.cos(a[1] - b[1])),
         math.atan2(math.sin(a[2] - b[2]), math.cos(a[2] - b[2])),
         math.atan2(math.sin(a[3] - b[3]), math.cos(a[3] - b[3])))
        for a, b in paired
    ]
    return imu_rows, differences, angular_velocity_differences, len(covariance)


def _aligned_roll(rows: list[tuple[int, float, float, float]],
                  times: list[int], target_ns: int) -> tuple[float, float] | None:
    index = bisect.bisect_right(times, target_ns) - 1
    if index < 0 or target_ns - times[index] > ALIGNMENT_NS:
        return None
    return rows[index][1], rows[index][2]


def _sample_records(sequence: tuple[body.MotionSample, ...],
                    imu_rows: list[tuple[int, float, float, float]],
                    imu_times: list[int], run_id: str) -> list[dict[str, float | str]]:
    if len(sequence) < 6:
        return []
    times = np.asarray([sample.time_s for sample in sequence], dtype=float)
    u = np.asarray([sample.state[0] for sample in sequence], dtype=float)
    r = np.asarray([sample.state[2] for sample in sequence], dtype=float)
    steering = np.asarray([sample.actuators[0] for sample in sequence], dtype=float)
    throttle = np.asarray([
        sample.actuator_history[3] if sample.actuator_history is not None else math.nan
        for sample in sequence
    ], dtype=float)
    wheels = np.asarray([
        sample.rear_wheel_surface_mps
        if sample.rear_wheel_surface_mps is not None else (math.nan, math.nan)
        for sample in sequence
    ], dtype=float)
    half_track = analysis.TRACK_WIDTH_M / 2.0
    ground_left = u - r * half_track
    ground_right = u + r * half_track
    records = []
    for index, sample in enumerate(sequence):
        if sample.time_s < 0.35 or not np.isfinite(wheels[index]).all():
            continue
        history_start = int(np.searchsorted(
            times, times[index] - ENCODER_WINDOW_S, side="left"))
        if history_start >= index:
            continue
        elapsed = times[index] - times[history_start]
        if not 0.075 <= elapsed <= 0.145:
            continue
        dt = np.diff(times[history_start:index + 1])
        if np.any((dt < 0.015) | (dt > 0.075)):
            continue
        attitude = _aligned_roll(imu_rows, imu_times, sample.receipt_ns)
        if attitude is None:
            continue
        command_deltas = np.abs(np.diff(throttle[history_start:index + 1]))
        if not np.isfinite(command_deltas).all():
            continue
        tv = float(np.sum(command_deltas) / elapsed)
        ground_l_window = ground_left[history_start:index + 1]
        ground_r_window = ground_right[history_start:index + 1]
        ground_l = float(np.sum(0.5 * (ground_l_window[:-1] + ground_l_window[1:]) * dt)
                         / elapsed)
        ground_r = float(np.sum(0.5 * (ground_r_window[:-1] + ground_r_window[1:]) * dt)
                         / elapsed)
        speed = math.hypot(float(sample.state[0]), float(sample.state[1]))
        records.append({
            "run_id": run_id,
            "speed_mps": speed,
            "lateral_velocity_mps": float(sample.state[1]),
            "steering_rad": float(steering[index]),
            "throttle_command": float(throttle[index]),
            "throttle_total_variation_per_s": tv,
            "longitudinal_accel_mps2": float((u[index] - u[history_start]) / elapsed),
            "roll_rad": attitude[0],
            "wheel_left_mps": float(wheels[index, 0]),
            "wheel_right_mps": float(wheels[index, 1]),
            "ground_left_mps": ground_l,
            "ground_right_mps": ground_r,
        })
    return records


def _median_ci_by_run(values: dict[str, list[float]], seed: int = 20260928
                      ) -> tuple[float | None, list[float] | None]:
    per_run = np.asarray([np.median(v) for v in values.values() if v], dtype=float)
    if not len(per_run):
        return None, None
    rng = np.random.default_rng(seed)
    draws = np.median(rng.choice(per_run, (2000, len(per_run)), replace=True), axis=1)
    return float(np.median(per_run)), np.quantile(draws, [0.025, 0.975]).tolist()


def _within_run_beta(rows: list[dict[str, float | str]], outcome_name: str,
                     feature_names: tuple[str, ...], minimum_steering: float,
                     seed: int) -> dict[str, Any]:
    selected = [row for row in rows
                if float(row["speed_mps"]) >= MIN_SPEED_MPS
                and abs(float(row["steering_rad"])) >= minimum_steering]
    groups: dict[str, list[dict[str, float | str]]] = defaultdict(list)
    for row in selected:
        groups[str(row["run_id"])].append(row)
    # Non-overlapping 100 ms encoder windows reduce, but do not eliminate,
    # serial dependence. Resampling is at the independent run level.
    groups = {run: values[::4] for run, values in groups.items() if len(values) >= 8}
    if len(groups) < 4:
        return {"samples": sum(map(len, groups.values())), "runs": len(groups),
                "status": "insufficient_independent_runs"}

    def arrays(run: str) -> tuple[np.ndarray, np.ndarray]:
        values = groups[run]
        x = np.asarray([[float(row[name]) for name in feature_names]
                        for row in values], dtype=float)
        y = np.asarray([float(row[outcome_name]) for row in values], dtype=float)
        return x, y

    pooled_x = np.concatenate([arrays(run)[0] for run in groups])
    scale = np.maximum(pooled_x.std(axis=0), 1.0e-8)
    centered = {}
    for run in groups:
        x, y = arrays(run)
        x = (x - np.median(x, axis=0)) / scale
        y = y - np.median(y)
        centered[run] = (x, y)

    def fit(run_ids: list[str]) -> np.ndarray:
        x = np.concatenate([centered[run][0] for run in run_ids])
        y = np.concatenate([centered[run][1] for run in run_ids])
        beta = np.linalg.lstsq(x, y, rcond=None)[0]
        # Huber IRLS limits the leverage of encoder bursts without deleting
        # the samples; these are the same bursts the odometry model must face.
        for _ in range(8):
            residual = y - x @ beta
            robust_scale = 1.4826 * np.median(np.abs(residual - np.median(residual)))
            if robust_scale < 1.0e-9:
                break
            cutoff = 1.345 * robust_scale
            weights = np.minimum(1.0, cutoff / np.maximum(np.abs(residual), 1.0e-12))
            xtwx = x.T @ (weights[:, None] * x)
            xtwy = x.T @ (weights * y)
            beta = np.linalg.lstsq(xtwx, xtwy, rcond=None)[0]
        return beta

    run_ids = list(groups)
    estimate = fit(run_ids)
    rng = np.random.default_rng(seed)
    bootstrap = np.asarray([
        fit(rng.choice(run_ids, size=len(run_ids), replace=True).tolist())
        for _ in range(500)
    ])
    ci = np.quantile(bootstrap, [0.025, 0.975], axis=0)
    return {
        "samples_after_100ms_stride": sum(map(len, groups.values())),
        "independent_runs": len(groups),
        "outcome": outcome_name,
        "features": {
            name: {"beta_mps_per_feature_sd": float(beta),
                   "run_bootstrap_95pct_ci": [float(ci[0, i]), float(ci[1, i])]}
            for i, (name, beta) in enumerate(zip(feature_names, estimate))
        },
        "interpretation": "within-run association, adjusted for the listed covariates; exploratory, not causal",
    }


def _summarize_fold(rows: list[dict[str, float | str]], fold: str) -> dict[str, Any]:
    result: dict[str, Any] = {"rows": len(rows),
                              "runs": len({str(row["run_id"]) for row in rows})}
    if not rows:
        return result
    speed = np.asarray([float(row["speed_mps"]) for row in rows])
    steer = np.asarray([abs(float(row["steering_rad"])) for row in rows])
    roll = np.asarray([float(row["roll_rad"]) for row in rows])
    left = np.asarray([float(row["slip_left_mps"]) for row in rows])
    right = np.asarray([float(row["slip_right_mps"]) for row in rows])
    result["aligned_roll_deg_p50_p95_abs"] = np.rad2deg(
        np.quantile(np.abs(roll), [0.5, 0.95])).tolist()
    high = (speed >= MIN_SPEED_MPS) & (steer >= HIGH_STEERING_RAD)
    result["high_steering"] = {
        "rows": int(high.sum()),
        "runs": len({str(rows[i]["run_id"]) for i in np.flatnonzero(high)}),
    }
    speed_bins = ((2.0, 4.0), (4.0, 6.0), (6.0, 100.0))
    result["high_steering_by_speed"] = []
    for low, high_speed in speed_bins:
        mask = high & (speed >= low) & (speed < high_speed)
        if not np.any(mask):
            continue
        slip = 0.5 * (np.abs(left[mask]) + np.abs(right[mask]))
        ground = np.maximum(1.0, 0.5 * (
            np.abs(np.asarray([float(row["ground_left_mps"]) for row in rows])[mask])
            + np.abs(np.asarray([float(row["ground_right_mps"]) for row in rows])[mask])))
        result["high_steering_by_speed"].append({
            "speed_mps": [low, high_speed if high_speed < 100 else None],
            "rows": int(mask.sum()),
            "runs": len({str(rows[i]["run_id"]) for i in np.flatnonzero(mask)}),
            "median_abs_rear_slip_proxy_mps": float(np.median(slip)),
            "p90_abs_rear_slip_proxy_mps": float(np.quantile(slip, 0.9)),
            "p99_abs_rear_slip_proxy_mps": float(np.quantile(slip, 0.99)),
            "median_abs_rear_slip_proxy_ratio": float(np.median(slip / ground)),
            "median_abs_roll_deg": float(np.rad2deg(np.median(
                np.abs(roll[mask])))),
        })
    common = ("speed_mps", "abs_steering_rad", "abs_throttle_command",
              "throttle_total_variation_per_s", "abs_roll_deg")
    interaction = common + ("steering_x_throttle_variation",)
    asym = ("speed_mps", "abs_steering_rad", "abs_throttle_command",
            "throttle_total_variation_per_s", "roll_in_turn_direction_deg")
    enriched = []
    for row in rows:
        copy = dict(row)
        copy["abs_steering_rad"] = abs(float(row["steering_rad"]))
        copy["steering_x_throttle_variation"] = (
            copy["abs_steering_rad"]
            * float(row["throttle_total_variation_per_s"]))
        copy["abs_throttle_command"] = abs(float(row["throttle_command"]))
        copy["abs_roll_deg"] = abs(math.degrees(float(row["roll_rad"])))
        copy["roll_in_turn_direction_deg"] = (
            math.degrees(float(row["roll_rad"]))
            * (1.0 if float(row["steering_rad"]) >= 0.0 else -1.0))
        copy["slip_common_mps"] = 0.5 * (
            abs(float(row["slip_left_mps"]))
            + abs(float(row["slip_right_mps"])))
        copy["slip_side_asymmetry_mps"] = (
            abs(float(row["slip_left_mps"]))
            - abs(float(row["slip_right_mps"])))
        enriched.append(copy)
    result["high_steering_within_run_models"] = {
        "common_slip": _within_run_beta(
            enriched, "slip_common_mps", common, HIGH_STEERING_RAD,
            20260928 + (fold == "validation")),
        "left_minus_right_abs_slip": _within_run_beta(
            enriched, "slip_side_asymmetry_mps", asym, HIGH_STEERING_RAD,
            20261937 + (fold == "validation")),
        "moderate_high_steering_throttle_interaction": _within_run_beta(
            enriched, "slip_common_mps", interaction, 0.30,
            20262946 + (fold == "validation")),
    }
    return result


def analyze(manifest_path: Path, selection_report_path: Path,
            output_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_report_path.read_text(encoding="utf-8"))
    config = selection["configuration"]
    validation_ids = set(config["additional_validation_runs"])
    experiment_ids = set(config["experiment_test_runs"])
    runs = {}
    for entry in manifest["runs"]:
        run_id = entry["run_id"]
        if (entry.get("effective_split") != "train"
                or not entry.get("clean_stream_and_collision_gate")
                or entry.get("duplicate_group_size") != 1
                or not entry.get("samples_exported")):
            continue
        if run_id in experiment_ids:
            continue
        fold = "validation" if run_id in validation_ids else "fit"
        bag = ROOT / entry["bag"]
        if bag.is_file():
            runs[run_id] = (fold, bag)

    records: dict[str, list[dict[str, float | str]]] = {"fit": [], "validation": []}
    calibration_candidates: dict[str, list[float]] = defaultdict(list)
    run_reports = []
    errors = []
    for run_number, (run_id, (fold, bag)) in enumerate(sorted(runs.items()), 1):
        extracted = []
        try:
            capture = body.load_capture(bag)
            if (capture.collision_count_start != 0 or capture.collision_count_end != 0
                    or capture.timing_faults != 0):
                errors.append({"run_id": run_id, "error": "capture quality gate changed since manifest"})
                continue
            (attitudes, attitude_diffs, angular_velocity_diffs,
             attitude_count) = _read_attitudes(bag)
            attitude_times = [row[0] for row in attitudes]
            for sequence in capture.sequences:
                for sample_row in _sample_records(sequence, attitudes,
                                                  attitude_times, run_id):
                    extracted.append(sample_row)
            # _sample_records embeds roll per sample; use those values for the
            # run attitude summary without a second timestamp lookup.
            if extracted:
                roll_values = np.asarray([float(row["roll_rad"]) for row in extracted])
                sequence_count = len(capture.sequences)
            else:
                roll_values = np.empty(0)
                sequence_count = len(capture.sequences)
            for row in extracted:
                speed = float(row["speed_mps"])
                steer = abs(float(row["steering_rad"]))
                yaw = abs(float(row["ground_right_mps"]) - float(row["ground_left_mps"])) / analysis.TRACK_WIDTH_M
                if speed >= 2.0 and steer < 0.12 and yaw < 0.25:
                    ground_l = float(row["ground_left_mps"])
                    ground_r = float(row["ground_right_mps"])
                    wheel_l = float(row["wheel_left_mps"])
                    wheel_r = float(row["wheel_right_mps"])
                    if min(ground_l, ground_r, wheel_l, wheel_r) > 1.5:
                        if (abs(float(row["lateral_velocity_mps"])) < 0.15
                                and abs(float(row["longitudinal_accel_mps2"])) < 1.0):
                            calibration_candidates[run_id].extend(
                                (ground_l / wheel_l, ground_r / wheel_r))
                records[fold].append(row)
            diff = np.asarray(attitude_diffs, dtype=float)
            angular_diff = np.asarray(angular_velocity_diffs, dtype=float)
            run_reports.append({
                "run_id": run_id,
                "fold": fold,
                "samples_with_roll_and_slip_window": len(extracted),
                "valid_motion_sequences": sequence_count,
                "imu_roll_deg_p50_p95_abs": (
                    np.rad2deg(np.quantile(np.abs(roll_values), [0.5, 0.95])).tolist()
                    if len(roll_values) else None),
                "imu_vs_odom_orientation_pairs": len(diff),
                "imu_vs_odom_roll_pitch_yaw_abs_difference_deg_p50_p95_max": (
                    np.rad2deg(np.asarray([
                        np.quantile(np.abs(diff), 0.5, axis=0),
                        np.quantile(np.abs(diff), 0.95, axis=0),
                        np.max(np.abs(diff), axis=0),
                    ])).tolist() if len(diff) else None),
                "imu_vs_odom_angular_velocity_pairs": len(angular_diff),
                "imu_vs_odom_angular_velocity_abs_difference_rad_s_p50_p95_max": (
                    np.asarray([
                        np.quantile(np.abs(angular_diff), 0.5, axis=0),
                        np.quantile(np.abs(angular_diff), 0.95, axis=0),
                        np.max(np.abs(angular_diff), axis=0),
                    ]).tolist() if len(angular_diff) else None),
            "imu_orientation_covariance_x_valid_samples": attitude_count,
        })
        except (OSError, ValueError, sqlite3.Error, KeyError) as exc:
            errors.append({"run_id": run_id, "error": f"{type(exc).__name__}: {exc}"})
        print(f"processed {run_number}/{len(runs)} {run_id}: {len(extracted)} aligned samples",
              flush=True)

    per_run_scale = {run: float(np.median(values)) for run, values
                     in calibration_candidates.items() if values}
    factor, factor_ci = _median_ci_by_run(calibration_candidates)
    if factor is None or not math.isfinite(factor) or factor <= 0.0:
        raise ValueError("no valid steady-straight wheel calibration samples")
    for fold_rows in records.values():
        for row in fold_rows:
            row["slip_left_mps"] = factor * float(row["wheel_left_mps"]) - float(row["ground_left_mps"])
            row["slip_right_mps"] = factor * float(row["wheel_right_mps"]) - float(row["ground_right_mps"])

    max_attitude_difference = np.zeros(3, dtype=float)
    paired_attitudes = 0
    max_angular_velocity_difference = np.zeros(3, dtype=float)
    paired_angular_velocities = 0
    for run in run_reports:
        paired_attitudes += int(run["imu_vs_odom_orientation_pairs"])
        per_run_diff = run["imu_vs_odom_roll_pitch_yaw_abs_difference_deg_p50_p95_max"]
        if per_run_diff is not None:
            max_attitude_difference = np.maximum(
                max_attitude_difference, np.asarray(per_run_diff[2], dtype=float))
        paired_angular_velocities += int(run["imu_vs_odom_angular_velocity_pairs"])
        per_run_rate_diff = run[
            "imu_vs_odom_angular_velocity_abs_difference_rad_s_p50_p95_max"]
        if per_run_rate_diff is not None:
            max_angular_velocity_difference = np.maximum(
                max_angular_velocity_difference,
                np.asarray(per_run_rate_diff[2], dtype=float))

    result = {
        "schema_version": 1,
        "manifest": str(manifest_path.resolve()),
        "run_selection_report": str(selection_report_path.resolve()),
        "method": {
            "wheel_slip_proxy": "encoder-derived rear wheel surface speed, calibrated by one training-only steady-straight scale, minus 100 ms mean rigid-body ground speed at each rear wheel (u +/- yaw_rate*track/2)",
            "wheel_radius_in_source_conversion_m": analysis.WHEEL_RADIUS_M,
            "track_width_m": analysis.TRACK_WIDTH_M,
            "minimum_speed_mps": MIN_SPEED_MPS,
            "high_steering_threshold_rad": HIGH_STEERING_RAD,
            "throttle_transient": "total variation of commanded throttle over the preceding encoder window, divided by window duration",
            "fold_policy": "fit and selected validation runs only; experiment-test, original test, and final-test are excluded",
            "inference_limit": "roll/slip associations are observational; the slip proxy is not direct tire slip, contact load, or force",
        },
        "encoder_scale": {
            "training_runs_with_steady_straight_support": len(per_run_scale),
            "per_run_median_factor_p10_p50_p90": (
                np.quantile(list(per_run_scale.values()), [0.1, 0.5, 0.9]).tolist()
                if per_run_scale else None),
            "training_only_global_factor": factor,
            "run_cluster_bootstrap_95pct_ci": factor_ci,
            "implied_effective_wheel_radius_m": float(analysis.WHEEL_RADIUS_M * factor),
            "calibration_region": "training samples: speed>=2m/s, |steering|<0.12rad, |yaw rate|<0.25rad/s, |lateral velocity|<0.15m/s, |longitudinal acceleration|<1m/s^2",
            "per_run_factors": per_run_scale,
        },
        "attitude_source_check": {
            "imu_odom_pairs": paired_attitudes,
            "maximum_abs_roll_pitch_yaw_difference_deg": max_attitude_difference.tolist(),
            "imu_orientation_matches_odom_pose": bool(np.max(max_attitude_difference) < 1.0e-5),
            "meaning": (
                "exact attitude and angular-rate matches mean the logged IMU roll and angular-x channels "
                "are duplicates of odometry pose and twist in this simulator bridge, not independent sensors"),
            "imu_odom_angular_velocity_pairs": paired_angular_velocities,
            "maximum_abs_angular_velocity_difference_rad_s": (
                max_angular_velocity_difference.tolist()),
            "imu_angular_velocity_matches_odom_twist": bool(
                np.max(max_angular_velocity_difference) < 1.0e-5),
        },
        "folds": {name: _summarize_fold(rows, name)
                  for name, rows in records.items()},
        "runs": run_reports,
        "errors": errors,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(f"wrote {output_path}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("run_selection_report", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    analyze(args.manifest, args.run_selection_report, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
