#!/usr/bin/env python3
"""Fit a low-order yaw-rate residual and hold out complete runs.

This is an offline screening tool only. It never changes the deployed vehicle
model. Body-state features and yaw-rate targets come from simulator truth, not
the derived odometry estimator. Practice truth samples are paired by source
order at the fixed 25 ms model step; packet receipt timestamps only associate
the current truth row with its controller command. Future truth is a training
label only and is never a runtime model input.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import ctypes
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from tools import evaluate_open_plane_body_dynamics as body
from tools.racing.offline.controller.production_mpc import ProductionMpc
from tools.racing.score_yaw_residual_rollouts import (
    ModelControl,
    ModelState,
    StageResult,
    interpolate_curvature,
)

from tools.racing.track_projection import TrackProjection


ROOT = Path(__file__).resolve().parents[2]
DT_S = 0.025
COM_X_M = 0.15532
OPENPLANE_TARGET_SPEED_MPS = 7.5
RIDGE_LAMBDA = 10.0
FEATURES = (
    "u_mps",
    "r_radps",
    "steering_feedback_rad",
    "steering_command_rate_radps",
    "lateral_speed_mps",
    "path_curvature_inv_m",
    "modeled_longitudinal_accel_mps2",
    "reference_speed_rate_mps2",
    "u_delta_abs_delta",
    "delta_abs_qdelta",
    "v_abs_delta",
    "delta_abs_ax",
    "delta_abs_qv",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def finite(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    if not value:
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def sector_for(track_s: float, sectors: list[dict[str, Any]], period: float) -> str:
    position = track_s % period
    for sector in sectors:
        start = float(sector["start_track_s_m"]) % period
        end = float(sector["end_track_s_m"]) % period
        if (start < end and start <= position < end) or (
            start > end and (position >= start or position < end)
        ):
            return str(sector["id"])
    raise ValueError(f"canonical sector map has a gap at {position:.6f} m")


def load_run(report_dir: Path, projection: TrackProjection,
             sectors: list[dict[str, Any]], period: float,
             library_path: Path, config_path: Path,
             trajectory_path: Path, steering_rate_limit_radps: float,
             target_speed_rate_increase_mps2: float,
             target_speed_rate_reduction_mps2: float) -> list[dict[str, Any]]:
    tracking = read_csv(report_dir / "tracking_error.csv")
    control = read_csv(report_dir / "controller_saturation.csv")
    times = [float(row["time_s"]) for row in tracking]
    trajectory = np.loadtxt(trajectory_path, delimiter=",", comments="#", dtype=np.float64)
    path_s, path_curvature = trajectory[:, 0], trajectory[:, 4]
    model_library = ctypes.CDLL(str(library_path.resolve()))
    model_library.mpc_vehicle_model_step.argtypes = [
        ctypes.POINTER(ModelState), ctypes.POINTER(ModelControl), ctypes.c_float, ctypes.c_float]
    model_library.mpc_vehicle_model_step.restype = StageResult
    selected: list[dict[str, Any]] = []
    with ProductionMpc(library_path, config_path, trajectory_path) as production_model:
        lap_length = production_model.lap_length_m
        for control_row in control:
            t = finite(control_row, "time_s")
            if t is None:
                continue
            index = bisect.bisect_left(times, t)
            candidates = [i for i in (index - 1, index) if 0 <= i < len(tracking)]
            if not candidates:
                continue
            current_index = min(candidates, key=lambda i: abs(times[i] - t))
            current = tracking[current_index]
            if abs(times[current_index] - t) > 0.012:
                continue
            lap_count = int(float(current.get("lap_count") or -1))
            if not 2 <= lap_count <= 11:
                continue

            u = finite(current, "truth_speed_mps")
            r = finite(current, "yaw_rate_radps")
            delta = finite(current, "steering_feedback_rad")
            qdelta = finite(control_row, "first_control_steering_rate_radps")
            qv = finite(control_row, "first_control_target_speed_rate_mps2")
            v = finite(current, "truth_lateral_speed_mps")
            curvature = finite(current, "reference_curvature_inv_m")
            # The simulator advances one model sample per 25 ms. Receipt and
            # header timestamps can be bursty, so do not time-interpolate the
            # label: the next source-ordered truth row is the next model step.
            if current_index + 1 >= len(tracking):
                continue
            following = tracking[current_index + 1]
            r_future = finite(following, "yaw_rate_radps")
            x, y = finite(current, "x_m"), finite(current, "y_m")
            initial = (
                finite(current, "tracking_error_m"),
                finite(current, "truth_heading_error_rad"),
                u, v, r,
                finite(control_row, "state_target_speed_mps"),
                finite(control_row, "state_steering_command_rad"),
                finite(control_row, "state_delayed_steering_command_1_rad"),
                finite(control_row, "state_delayed_steering_command_2_rad"),
                delta,
            )
            values = (u, r, delta, qdelta, qv, v, curvature, r_future, x, y, *initial)
            if any(value is None for value in values):
                continue
            qdelta = max(-steering_rate_limit_radps,
                         min(steering_rate_limit_radps, float(qdelta)))
            qv = max(-target_speed_rate_reduction_mps2,
                     min(target_speed_rate_increase_mps2, float(qv)))
            pose = projection.project(float(x), float(y), 0.0)
            progress = finite(current, "s_m")
            if progress is None:
                continue
            (e_y, e_psi, state_u, state_v, state_r, target_speed,
             steering_command, delayed_1, delayed_2, actual_steering) = map(float, initial)
            model_state = ModelState(
                e_y, e_psi, state_u, state_v, state_r, target_speed,
                steering_command, delayed_1, delayed_2, actual_steering,
            )
            model_control = ModelControl(float(qdelta), float(qv))
            model_curvature = interpolate_curvature(progress, path_s, path_curvature, lap_length)
            baseline_stage = model_library.mpc_vehicle_model_step(
                ctypes.byref(model_state), ctypes.byref(model_control),
                ctypes.c_float(DT_S), ctypes.c_float(model_curvature),
            )
            if not baseline_stage.valid:
                continue
            r_pred = float(baseline_stage.next.r)
            modeled_ax = float(baseline_stage.body_accel_mps2)
            feature_map = {
                "u_mps": float(u),
                "r_radps": float(r),
                "steering_feedback_rad": float(delta),
                "steering_command_rate_radps": float(qdelta),
                "lateral_speed_mps": float(v),
                "path_curvature_inv_m": float(curvature),
                "modeled_longitudinal_accel_mps2": modeled_ax,
                "reference_speed_rate_mps2": float(qv),
                "u_delta_abs_delta": float(u) * float(delta) * abs(float(delta)),
                "delta_abs_qdelta": float(delta) * abs(float(qdelta)),
                "v_abs_delta": float(v) * abs(float(delta)),
                "delta_abs_ax": abs(float(delta)) * modeled_ax,
                "delta_abs_qv": float(delta) * abs(float(qv)),
                "target_speed_mps": target_speed,
            }
            selected.append({
                "run_id": current["run_id"],
                "dataset": "practice_race",
                "lap_count": lap_count,
                "time_s": t,
                "truth_source_time_s": finite(current, "source_time_s"),
                "truth_next_source_time_s": finite(following, "source_time_s"),
                "truth_next_source_delta_s": (
                    finite(following, "source_time_s") -
                    finite(current, "source_time_s")
                    if (finite(following, "source_time_s") is not None and
                        finite(current, "source_time_s") is not None)
                    else None
                ),
                "track_s_m": pose.s_m,
                "sector": sector_for(pose.s_m, sectors, period),
                "steering_abs_rad": abs(float(delta)),
                "speed_mps": float(u),
                "dt_s": DT_S,
                "baseline_prediction_radps": r_pred,
                "truth_future_yaw_rate_radps": float(r_future),
                "residual_rate_target_radps2": (float(r_future) - r_pred) / DT_S,
                "features": feature_map,
                "model_state": tuple(map(float, initial)),
                "model_control": (float(qdelta), float(qv)),
                "model_curvature_inv_m": float(model_curvature),
                "modeled_longitudinal_accel_mps2": modeled_ax,
            })
    return selected


def load_open_plane_run(bag_path: Path, library_path: Path,
                        config_path: Path, trajectory_path: Path,
                        steering_rate_limit_radps: float,
                        target_speed_rate_increase_mps2: float,
                        target_speed_rate_reduction_mps2: float, *,
                        dataset_name: str = "openplane_highsteer_train"
                        ) -> list[dict[str, Any]]:
    """Create one-step labels from complete isolated open-plane captures."""
    capture = body.load_capture(bag_path)
    if (capture.aborted or capture.reason != "schedule complete"
            or capture.collision_count_end != capture.collision_count_start
            or capture.timing_faults != 0 or capture.invalid_phase_count != 0):
        raise ValueError(f"open-plane training capture failed quality gates: {bag_path}")
    phases = []
    for label, sequence in zip(capture.sequence_labels, capture.sequences):
        if label.startswith("multispeed_v"):
            target_speed_mps = float(label.split("_", 2)[1][1:])
        elif label.startswith("r1_steer_") or label == "r1_baseline_zero_steer":
            target_speed_mps = OPENPLANE_TARGET_SPEED_MPS
        elif label.startswith("lowdyn_v"):
            target_speed_mps = float(label.split("_", 2)[1][1:])
        elif label.startswith("yawdyn_v"):
            target_speed_mps = float(label.split("_", 2)[1][1:])
        else:
            continue
        phases.append((label, sequence, target_speed_mps))
    if any(label.startswith("yawdyn_v") for label, _, _ in phases):
        expected_phase_count = 18
    elif any(label.startswith("lowdyn_v") for label, _, _ in phases):
        expected_phase_count = 24
    else:
        expected_phase_count = 51 if any(
            label.startswith("multispeed_v") for label, _, _ in phases) else 17
    unique_phase_labels = {label for label, _, _ in phases}
    if len(unique_phase_labels) != expected_phase_count:
        raise ValueError(
            f"expected {expected_phase_count} unique valid high-steer phases, "
            f"found {len(unique_phase_labels)}")

    library = ctypes.CDLL(str(library_path.resolve()))
    library.mpc_vehicle_model_step.argtypes = [
        ctypes.POINTER(ModelState), ctypes.POINTER(ModelControl), ctypes.c_float,
        ctypes.c_float,
    ]
    library.mpc_vehicle_model_step.restype = StageResult
    run_id = bag_path.parent.parent.name
    selected: list[dict[str, Any]] = []
    with ProductionMpc(library_path, config_path, trajectory_path):
        for label, sequence, target_speed_mps in phases:
            for index in range(2, len(sequence) - 1):
                current = sequence[index]
                following = sequence[index + 1]
                previous = sequence[index - 1]
                previous2 = sequence[index - 2]
                # The capture loader attaches 500 ms of causal history to
                # each phase for transient diagnostics. It is not part of
                # the probe and must not be assigned the new phase's target.
                if current.time_s < 0.0:
                    continue
                if (current.packet_sequence < 0
                        or following.packet_sequence != current.packet_sequence + 1
                        or current.packet_sequence != previous.packet_sequence + 1
                        or previous.packet_sequence != previous2.packet_sequence + 1):
                    continue
                histories = (current.actuator_history, following.actuator_history,
                             previous.actuator_history, previous2.actuator_history)
                if any(history is None or len(history) != 4 for history in histories):
                    continue
                rigid = current.simulator_rigid_state
                future_rigid = following.simulator_rigid_state
                if (not isinstance(rigid, np.ndarray) or rigid.shape != (13,)
                        or not np.isfinite(rigid).all()
                        or not isinstance(future_rigid, np.ndarray)
                        or future_rigid.shape != (13,)
                        or not np.isfinite(future_rigid).all()):
                    continue

                u, v_com, r = float(rigid[7]), float(rigid[8]), float(rigid[12])
                v_rear = v_com - COM_X_M * r
                delta = float(current.actuators[0])
                command = float(current.actuator_history[2])
                command_next = float(following.actuator_history[2])
                q_delta = (command_next - command) / DT_S
                q_delta = max(-steering_rate_limit_radps,
                              min(steering_rate_limit_radps, q_delta))
                q_speed = 0.0
                if not all(math.isfinite(value) for value in
                           (u, v_rear, r, delta, command, command_next)):
                    continue

                initial = ModelState(
                    0.0, 0.0, u, v_rear, r, target_speed_mps,
                    command, float(previous.actuator_history[2]),
                    float(previous2.actuator_history[2]), delta,
                )
                control = ModelControl(q_delta, q_speed)
                stage = library.mpc_vehicle_model_step(
                    ctypes.byref(initial), ctypes.byref(control),
                    ctypes.c_float(DT_S), ctypes.c_float(0.0))
                if not stage.valid:
                    continue
                modeled_ax = float(stage.body_accel_mps2)
                feature_map = {
                    "u_mps": u,
                    "r_radps": r,
                    "steering_feedback_rad": delta,
                    "steering_command_rate_radps": q_delta,
                    "lateral_speed_mps": v_rear,
                    "path_curvature_inv_m": 0.0,
                    "modeled_longitudinal_accel_mps2": modeled_ax,
                    "reference_speed_rate_mps2": q_speed,
                    "u_delta_abs_delta": u * delta * abs(delta),
                    "delta_abs_qdelta": delta * abs(q_delta),
                    "v_abs_delta": v_rear * abs(delta),
                    "delta_abs_ax": abs(delta) * modeled_ax,
                    "delta_abs_qv": abs(delta) * abs(q_speed),
                    "target_speed_mps": target_speed_mps,
                }
                truth_future_r = float(future_rigid[12])
                selected.append({
                    "run_id": run_id,
                    "dataset": dataset_name,
                    "lap_count": 0,
                    "time_s": float(current.time_s),
                    "track_s_m": 0.0,
                    "sector": "open_plane",
                    "steering_abs_rad": abs(delta),
                    "speed_mps": u,
                    "dt_s": DT_S,
                    "baseline_prediction_radps": float(stage.next.r),
                    "truth_future_yaw_rate_radps": truth_future_r,
                    "residual_rate_target_radps2": (truth_future_r - float(stage.next.r)) / DT_S,
                    "features": feature_map,
                    "model_state": (
                        0.0, 0.0, u, v_rear, r, target_speed_mps, command,
                        float(previous.actuator_history[2]),
                        float(previous2.actuator_history[2]), delta,
                    ),
                    "model_control": (q_delta, q_speed),
                    "model_curvature_inv_m": 0.0,
                    "modeled_longitudinal_accel_mps2": modeled_ax,
                })
    if not selected:
        raise ValueError(f"no aligned open-plane training samples: {bag_path}")
    return selected


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def metrics(rows: list[dict[str, Any]], corrections: np.ndarray) -> dict[str, Any]:
    baseline = np.asarray([
        row["truth_future_yaw_rate_radps"] - row["baseline_prediction_radps"]
        for row in rows
    ], dtype=np.float64)
    candidate = baseline - corrections * np.asarray([row["dt_s"] for row in rows])

    def summary(error: np.ndarray) -> dict[str, float]:
        return {
            "count": int(len(error)),
            "bias_radps": float(np.mean(error)),
            "rmse_radps": float(np.sqrt(np.mean(error ** 2))),
            "p95_abs_radps": float(np.quantile(np.abs(error), 0.95)),
        }

    result: dict[str, Any] = {"production": summary(baseline), "residual_candidate": summary(candidate)}
    slices: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        steer, speed, sector = row["steering_abs_rad"], row["speed_mps"], row["sector"]
        steer_band = "steer_<0.10" if steer < 0.10 else "steer_0.10_0.20" if steer < 0.20 else "steer_0.20_0.30" if steer < 0.30 else "steer_>=0.30"
        speed_band = "speed_<3" if speed < 3.0 else "speed_3_5" if speed < 5.0 else "speed_5_8" if speed < 8.0 else "speed_>=8"
        slices[steer_band].append(i)
        slices[speed_band].append(i)
        if 2.0 <= speed < 4.0 and steer >= 0.25:
            speed_cell = min(2, max(0, int((speed - 2.25) / 0.5)))
            steer_cell = min(3, max(0, int((steer - 0.275) / 0.05)))
            slices[f"low_speed_high_steer:v{speed_cell}:d{steer_cell}"].append(i)
        slices[f"sector:{sector}"].append(i)
    result["by_regime"] = {}
    for name, indexes in sorted(slices.items()):
        if len(indexes) < 10:
            continue
        index = np.asarray(indexes, dtype=np.int64)
        result["by_regime"][name] = {
            "production": summary(baseline[index]),
            "residual_candidate": summary(candidate[index]),
        }
    return result


def smoothstep(value: float, start: float, end: float) -> float:
    if not end > start:
        raise ValueError("support-gate bounds must be increasing")
    x = max(0.0, min(1.0, (value - start) / (end - start)))
    return x * x * (3.0 - 2.0 * x)


def support_weights(rows: list[dict[str, Any]], gate: dict[str, Any] | None) -> np.ndarray:
    if gate is None:
        return np.ones(len(rows), dtype=np.float64)
    if gate["kind"] == "smoothstep_speed_abs_steering_v1":
        speed_weights = [
            smoothstep(row["features"]["u_mps"],
                       gate["speed_zero_mps"], gate["speed_full_mps"])
            for row in rows
        ]
    elif gate["kind"] == "smoothstep_target_tracking_steering_v1":
        speed_weights = []
        for row in rows:
            target_speed = row["features"]["target_speed_mps"]
            speed = row["features"]["u_mps"]
            target_weight = smoothstep(
                target_speed, gate["target_speed_zero_mps"],
                gate["target_speed_full_mps"])
            deficit = target_speed - speed
            tracking_weight = 1.0 - smoothstep(
                deficit, gate["speed_deficit_full_mps"],
                gate["speed_deficit_zero_mps"])
            speed_weights.append(target_weight * tracking_weight)
    else:
        raise ValueError(f"unsupported yaw-residual support gate: {gate['kind']}")
    weights = []
    for row, speed_weight in zip(rows, speed_weights):
        features = row["features"]
        abs_steering = abs(features["steering_feedback_rad"])
        steering_weight = smoothstep(
            abs_steering, gate["abs_steering_zero_rad"],
            gate["abs_steering_full_rad"])
        if "abs_steering_upper_zero_rad" in gate:
            steering_weight *= 1.0 - smoothstep(
                abs_steering, gate["abs_steering_upper_full_rad"],
                gate["abs_steering_upper_zero_rad"])
        actual_speed_weight = 1.0
        if "actual_speed_upper_zero_mps" in gate:
            actual_speed = features["u_mps"]
            actual_speed_weight = smoothstep(
                actual_speed, gate["actual_speed_zero_mps"],
                gate["actual_speed_full_mps"])
            actual_speed_weight *= 1.0 - smoothstep(
                actual_speed, gate["actual_speed_upper_full_mps"],
                gate["actual_speed_upper_zero_mps"])
        weights.append(speed_weight * steering_weight * actual_speed_weight)
    return np.asarray(weights, dtype=np.float64)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-report-dir", type=Path, required=True)
    parser.add_argument(
        "--train-openplane-bag", type=Path, action="append", default=[],
        help="optional complete speed-held high-steer training bag; may be repeated",
    )
    parser.add_argument(
        "--validation-openplane-bag", type=Path, action="append", default=[],
        help="optional complete held-out open-plane bag; never used for fitting",
    )
    parser.add_argument("--validation-report-dir", type=Path, required=True)
    parser.add_argument("--sectors", type=Path, required=True)
    parser.add_argument("--centerline", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--support-speed-zero-mps", type=float)
    parser.add_argument("--support-speed-full-mps", type=float)
    parser.add_argument("--support-steering-zero-rad", type=float)
    parser.add_argument("--support-steering-full-rad", type=float)
    parser.add_argument("--support-speed-deficit-full-mps", type=float)
    parser.add_argument("--support-speed-deficit-zero-mps", type=float)
    parser.add_argument("--support-actual-speed-zero-mps", type=float)
    parser.add_argument("--support-actual-speed-full-mps", type=float)
    parser.add_argument("--support-actual-speed-upper-full-mps", type=float)
    parser.add_argument("--support-actual-speed-upper-zero-mps", type=float)
    parser.add_argument("--support-steering-upper-full-rad", type=float)
    parser.add_argument("--support-steering-upper-zero-rad", type=float)
    parser.add_argument(
        "--fit-only-within-support", action="store_true",
        help="fit and select gain only from training samples with nonzero configured support weight",
    )
    args = parser.parse_args()
    support_values = (
        args.support_speed_zero_mps,
        args.support_speed_full_mps,
        args.support_steering_zero_rad,
        args.support_steering_full_rad,
    )
    if any(value is not None for value in support_values):
        if any(value is None for value in support_values):
            parser.error("all four support-gate bounds must be provided together")
        if not (args.support_speed_full_mps > args.support_speed_zero_mps
                and args.support_steering_full_rad > args.support_steering_zero_rad):
            parser.error("support-gate full bounds must exceed zero bounds")
    deficit_values = (
        args.support_speed_deficit_full_mps,
        args.support_speed_deficit_zero_mps,
    )
    if any(value is not None for value in deficit_values):
        if (any(value is None for value in support_values)
                or any(value is None for value in deficit_values)):
            parser.error("speed-deficit bounds require all four base gate bounds")
        if not args.support_speed_deficit_zero_mps > args.support_speed_deficit_full_mps:
            parser.error("speed-deficit zero bound must exceed its full bound")
    actual_speed_values = (
        args.support_actual_speed_zero_mps,
        args.support_actual_speed_full_mps,
        args.support_actual_speed_upper_full_mps,
        args.support_actual_speed_upper_zero_mps,
    )
    if any(value is not None for value in actual_speed_values):
        if any(value is None for value in support_values):
            parser.error("actual-speed bounds require the base speed/steering support gate")
        if any(value is None for value in actual_speed_values):
            parser.error("all four actual-speed support bounds must be provided together")
        if not (0.0 <= args.support_actual_speed_zero_mps
                < args.support_actual_speed_full_mps
                <= args.support_actual_speed_upper_full_mps
                < args.support_actual_speed_upper_zero_mps):
            parser.error("actual-speed support bounds must be strictly ordered")
    steering_upper_values = (
        args.support_steering_upper_full_rad,
        args.support_steering_upper_zero_rad,
    )
    if any(value is not None for value in steering_upper_values):
        if (any(value is None for value in support_values)
                or any(value is None for value in steering_upper_values)
                or not (args.support_steering_upper_full_rad
                        > args.support_steering_full_rad
                        and args.support_steering_upper_zero_rad
                        > args.support_steering_upper_full_rad)):
            parser.error("steering upper fade must start above the lower full bound")
    support_gate = None
    if all(value is not None for value in support_values):
        support_gate = {
            "kind": (
                "smoothstep_target_tracking_steering_v1"
                if all(value is not None for value in deficit_values)
                else "smoothstep_speed_abs_steering_v1"
            ),
            "abs_steering_zero_rad": args.support_steering_zero_rad,
            "abs_steering_full_rad": args.support_steering_full_rad,
            "rationale": (
                "smooth activation bounded to the explicitly configured "
                "training-supported target-speed, actual-speed, and steering "
                "domain; parameters are recorded with this candidate"
            ),
        }
        if all(value is not None for value in deficit_values):
            support_gate.update({
                "target_speed_zero_mps": args.support_speed_zero_mps,
                "target_speed_full_mps": args.support_speed_full_mps,
                "speed_deficit_full_mps": args.support_speed_deficit_full_mps,
                "speed_deficit_zero_mps": args.support_speed_deficit_zero_mps,
            })
        else:
            support_gate.update({
                "speed_zero_mps": args.support_speed_zero_mps,
                "speed_full_mps": args.support_speed_full_mps,
            })
        if all(value is not None for value in actual_speed_values):
            support_gate.update({
                "actual_speed_zero_mps": args.support_actual_speed_zero_mps,
                "actual_speed_full_mps": args.support_actual_speed_full_mps,
                "actual_speed_upper_full_mps": args.support_actual_speed_upper_full_mps,
                "actual_speed_upper_zero_mps": args.support_actual_speed_upper_zero_mps,
            })
        if all(value is not None for value in steering_upper_values):
            support_gate.update({
                "abs_steering_upper_full_rad": args.support_steering_upper_full_rad,
                "abs_steering_upper_zero_rad": args.support_steering_upper_zero_rad,
            })
    if args.fit_only_within_support and support_gate is None:
        parser.error("--fit-only-within-support requires all support-gate bounds")
    train_dir = args.train_report_dir if args.train_report_dir.is_absolute() else ROOT / args.train_report_dir
    validation_dir = args.validation_report_dir if args.validation_report_dir.is_absolute() else ROOT / args.validation_report_dir
    sectors_path = args.sectors if args.sectors.is_absolute() else ROOT / args.sectors
    centerline_path = args.centerline if args.centerline.is_absolute() else ROOT / args.centerline
    library_path = args.library if args.library.is_absolute() else ROOT / args.library
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    trajectory_path = args.trajectory if args.trajectory.is_absolute() else ROOT / args.trajectory
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    openplane_bags = [
        path if path.is_absolute() else ROOT / path
        for path in args.train_openplane_bag
    ]
    validation_openplane_bags = [
        path if path.is_absolute() else ROOT / path
        for path in args.validation_openplane_bag
    ]
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error(f"output directory is not empty: {output_dir}")
    import yaml
    sector_config = yaml.safe_load(sectors_path.read_text(encoding="utf-8"))
    mpc_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    mpc_params = mpc_config["/**"]["ros__parameters"]
    steering_rate_limit_radps = float(
        mpc_params.get("max_steering_rate_radps", 3.2))
    target_speed_rate_increase_mps2 = float(
        mpc_params.get("max_target_speed_rate_increase_mps2", 3.0))
    target_speed_rate_reduction_mps2 = float(
        mpc_params.get("max_target_speed_rate_reduction_mps2", 8.0))
    period = float(sector_config["period_m"])
    projection = TrackProjection.from_csv(centerline_path, closed=True)
    if not math.isclose(projection.total_length, period, rel_tol=0.0, abs_tol=1e-5):
        parser.error("centerline length does not match sector config")
    train = load_run(train_dir, projection, sector_config["sectors"], period,
                     library_path, config_path, trajectory_path,
                     steering_rate_limit_radps,
                     target_speed_rate_increase_mps2,
                     target_speed_rate_reduction_mps2)
    openplane_train: list[dict[str, Any]] = []
    openplane_hashes: dict[str, str] = {}
    for bag_path in openplane_bags:
        if not bag_path.is_file():
            parser.error(f"open-plane training bag does not exist: {bag_path}")
        rows = load_open_plane_run(
            bag_path, library_path, config_path, trajectory_path,
            steering_rate_limit_radps, target_speed_rate_increase_mps2,
            target_speed_rate_reduction_mps2)
        openplane_train.extend(rows)
        openplane_hashes[str(bag_path)] = sha256_file(bag_path)
    train.extend(openplane_train)
    validation = load_run(
        validation_dir, projection, sector_config["sectors"], period,
        library_path, config_path, trajectory_path,
        steering_rate_limit_radps, target_speed_rate_increase_mps2,
        target_speed_rate_reduction_mps2)
    validation_openplane_hashes: dict[str, str] = {}
    for bag_path in validation_openplane_bags:
        if not bag_path.is_file():
            parser.error(f"open-plane validation bag does not exist: {bag_path}")
        rows = load_open_plane_run(
            bag_path, library_path, config_path, trajectory_path,
            steering_rate_limit_radps, target_speed_rate_increase_mps2,
            target_speed_rate_reduction_mps2,
            dataset_name="openplane_highsteer_validation",
        )
        validation.extend(rows)
        validation_openplane_hashes[str(bag_path)] = sha256_file(bag_path)
    train_ids = {row["run_id"] for row in train}
    validation_ids = {row["run_id"] for row in validation}
    if not train or not validation or train_ids & validation_ids:
        parser.error("training and validation must be non-empty, distinct whole runs")

    x_train = np.asarray([[row["features"][name] for name in FEATURES] for row in train], dtype=np.float64)
    y_train = np.asarray([row["residual_rate_target_radps2"] for row in train], dtype=np.float64)
    train_gate_weights = support_weights(train, support_gate)
    fit_mask = (train_gate_weights > 0.0 if args.fit_only_within_support
                else np.ones(len(train), dtype=bool))
    if not np.any(fit_mask):
        parser.error("no training samples lie inside the configured model support")
    fit_run_ids = sorted({train[index]["run_id"] for index in np.flatnonzero(fit_mask)})
    if args.fit_only_within_support and len(fit_run_ids) < 2:
        parser.error("support-only fitting requires supported training samples from at least two whole runs")
    x_fit = x_train[fit_mask]
    y_fit = y_train[fit_mask]
    if args.fit_only_within_support:
        run_support_totals = {
            run_id: float(sum(train_gate_weights[index]
                              for index in np.flatnonzero(fit_mask)
                              if train[index]["run_id"] == run_id))
            for run_id in fit_run_ids
        }
        sample_weights = np.asarray([
            train_gate_weights[index] /
            (len(fit_run_ids) * run_support_totals[train[index]["run_id"]])
            for index in np.flatnonzero(fit_mask)
        ], dtype=np.float64)
        sample_weights *= len(x_fit)
        feature_mean = np.average(x_fit, axis=0, weights=sample_weights)
        feature_variance = np.average(
            (x_fit - feature_mean) ** 2, axis=0, weights=sample_weights)
        feature_scale = np.sqrt(feature_variance)
    else:
        feature_mean = x_train.mean(axis=0)
        feature_scale = x_train.std(axis=0)
        run_counts = Counter(row["run_id"] for row in train)
        sample_weights = np.asarray([
            len(train) / (len(run_counts) * run_counts[train[index]["run_id"]])
            for index in np.flatnonzero(fit_mask)
        ], dtype=np.float64)
    feature_scale[feature_scale < 1e-8] = 1.0
    z_train = (x_train - feature_mean) / feature_scale
    z_fit = z_train[fit_mask]
    design = np.column_stack((np.ones(len(z_fit)), z_fit))
    penalty = np.eye(design.shape[1], dtype=np.float64) * RIDGE_LAMBDA
    penalty[0, 0] = 0.0
    weighted_design = design * np.sqrt(sample_weights)[:, None]
    weighted_targets = y_fit * np.sqrt(sample_weights)
    coefficients = np.linalg.solve(
        weighted_design.T @ weighted_design + penalty,
        weighted_design.T @ weighted_targets)

    x_validation = np.asarray([[row["features"][name] for name in FEATURES] for row in validation], dtype=np.float64)
    z_validation = (x_validation - feature_mean) / feature_scale
    raw_validation_corrections = (
        np.column_stack((np.ones(len(z_validation)), z_validation)) @ coefficients)
    # Bound extrapolating correction magnitude to the largest training p99
    # magnitude; this is a candidate-only guard, not a production fallback.
    correction_limit = float(np.quantile(np.abs(y_fit), 0.99))
    raw_validation_corrections = np.clip(
        raw_validation_corrections, -correction_limit, correction_limit)
    raw_train_corrections = np.column_stack((np.ones(len(z_train)), z_train)) @ coefficients
    raw_train_corrections = np.clip(
        raw_train_corrections, -correction_limit, correction_limit)
    validation_gate_weights = support_weights(validation, support_gate)
    train_by_run: dict[str, Any] = {}
    gains = (0.0, 0.05, 0.10, 0.15, 0.25, 0.50, 1.0)
    score_run_ids = fit_run_ids if args.fit_only_within_support else sorted(
        {row["run_id"] for row in train})
    for run_id in score_run_ids:
        indexes = [index for index, row in enumerate(train)
                   if row["run_id"] == run_id and fit_mask[index]]
        run_rows = [train[index] for index in indexes]
        run_corrections = raw_train_corrections[indexes] * train_gate_weights[indexes]
        train_by_run[run_id] = {
            f"gain_{gain:g}": metrics(run_rows, gain * run_corrections)
            for gain in gains
        }
    macro_train_rmse_by_gain = {
        gain: statistics.fmean(
            train_by_run[run_id][f"gain_{gain:g}"]["residual_candidate"]["rmse_radps"]
            for run_id in score_run_ids
        )
        for gain in gains
    }
    selected_gain = min(
        gains, key=lambda gain: (macro_train_rmse_by_gain[gain], gain))
    train_corrections = (
        selected_gain * raw_train_corrections * train_gate_weights)
    train_metrics = metrics(train, train_corrections)
    training_supported_metrics = metrics(
        [row for row, active in zip(train, fit_mask) if active],
        train_corrections[fit_mask],
    )
    corrections = selected_gain * raw_validation_corrections * validation_gate_weights
    validation_metrics = metrics(validation, corrections)
    validation_metrics_by_run: dict[str, Any] = {}
    for run_id in sorted(validation_ids):
        indexes = [index for index, row in enumerate(validation)
                   if row["run_id"] == run_id]
        run_rows = [validation[index] for index in indexes]
        run_corrections = corrections[indexes]
        validation_metrics_by_run[run_id] = {
            "dataset": run_rows[0].get("dataset", "unknown"),
            "samples": len(run_rows),
            "metrics": metrics(run_rows, run_corrections),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    model = {
        "schema_version": 1,
        "model_id": (
            "GT_support_only_ridge_yaw_residual_candidate"
            if args.fit_only_within_support else
            "P0_plus_openplane_steering_transient_ridge_yaw_residual_candidate"
            if openplane_bags else "P0_ridge_yaw_residual_candidate"),
        "status": "offline_one_step_candidate_only",
        "parent_model": "configured_production_C_transition",
        "baseline_config_sha256": sha256_file(config_path),
        "effective_control_rate_limits": {
            "steering_rate_abs_radps": steering_rate_limit_radps,
            "target_speed_rate_increase_mps2": target_speed_rate_increase_mps2,
            "target_speed_rate_reduction_mps2": target_speed_rate_reduction_mps2,
        },
        "gain": selected_gain,
        "training_run_ids": sorted(train_ids),
        "fit_run_ids": fit_run_ids,
        "fit_only_within_support": args.fit_only_within_support,
        "fit_sample_count": int(np.count_nonzero(fit_mask)),
        "openplane_training_bags_sha256": openplane_hashes,
        "openplane_validation_bags_sha256": validation_openplane_hashes,
        "weighting": (
            "equal total weight per whole training run within nonzero model support"
            if args.fit_only_within_support else
            "equal total weight per complete training run"),
        "features": list(FEATURES),
        "feature_mean": feature_mean.tolist(),
        "feature_scale": feature_scale.tolist(),
        "coefficients_with_intercept": coefficients.tolist(),
        "ridge_lambda": RIDGE_LAMBDA,
        "correction_clip_radps2": correction_limit,
        "sample_period_s": DT_S,
        "ground_truth_provenance": {
            "practice_source": "/autodrive/roboracer_1/odom simulator truth stream",
            "open_plane_source": "simulator rigid-body state in the captured truth packet",
            "derived_odometry_estimator_used_as_truth": False,
            "runtime_ground_truth_input": False,
        },
        "source_input_policy": (
            "current simulator-ground-truth body state and physical steering are "
            "offline training features; next source-ordered simulator-truth sample "
            "is the yaw-rate label at the fixed 25 ms step; packet receipt time is "
            "used only to associate current truth with recorded MPC commands; "
            "derived odometry is not used as plant truth; no future sensor inputs"
        ),
    }
    if support_gate is not None:
        model["model_id"] += "_support_gated_v1"
        model["support_gate"] = support_gate
    (output_dir / "yaw_residual_candidate.json").write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")

    validation_rows: list[dict[str, Any]] = []
    for index, (row, correction) in enumerate(zip(validation, corrections)):
        state = row["model_state"]
        control = row["model_control"]
        validation_rows.append({
            key: row[key] for key in (
                "run_id", "lap_count", "time_s", "track_s_m", "sector", "steering_abs_rad",
                "speed_mps", "dt_s", "baseline_prediction_radps", "truth_future_yaw_rate_radps",
                "residual_rate_target_radps2",
            )
        } | {
            "target_speed_mps": float(row["features"]["target_speed_mps"]),
            "support_weight": float(validation_gate_weights[index]),
            "candidate_residual_rate_radps2": float(correction),
            "state_e_y_m": state[0],
            "state_e_psi_rad": state[1],
            "state_u_mps": state[2],
            "state_v_mps": state[3],
            "state_r_radps": state[4],
            "state_target_speed_mps": state[5],
            "state_steering_command_rad": state[6],
            "state_delayed_steering_command_1_rad": state[7],
            "state_delayed_steering_command_2_rad": state[8],
            "state_actual_steering_angle_rad": state[9],
            "control_steering_rate_radps": control[0],
            "control_target_speed_rate_mps2": control[1],
            "path_curvature_inv_m": row["model_curvature_inv_m"],
            "modeled_longitudinal_accel_mps2": row[
                "modeled_longitudinal_accel_mps2"],
        })
    fields = list(validation_rows[0])
    with (output_dir / "validation_samples.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(validation_rows)

    report = {
        "schema_version": 1,
        "train_run_ids": sorted(train_ids),
        "validation_run_ids": sorted(validation_ids),
        "training_samples": len(train),
        "training_sources": {
            dataset: sum(row.get("dataset") == dataset for row in train)
            for dataset in sorted({row.get("dataset", "unknown") for row in train})
        },
        "validation_samples": len(validation),
        "training_scored_laps": sorted({
            row["lap_count"] for row in train if row["lap_count"] > 0
        }),
        "validation_scored_laps": sorted({row["lap_count"] for row in validation}),
        "training_metrics": train_metrics,
        "training_supported_metrics": training_supported_metrics,
        "training_gain_sweep_by_run": train_by_run,
        "selected_gain": selected_gain,
        "macro_training_rmse_by_gain": {
            f"gain_{gain:g}": rmse
            for gain, rmse in macro_train_rmse_by_gain.items()
        },
        "held_out_whole_run_metrics": validation_metrics,
        "held_out_metrics_by_run": validation_metrics_by_run,
        "one_step_holdout_improved_rmse": (
            validation_metrics["residual_candidate"]["rmse_radps"]
            < validation_metrics["production"]["rmse_radps"]
        ),
        "promotion_decision": "not_promoted_one_step_only",
        "support_gate": support_gate,
        "required_next_gate": "recursive 25-750 ms replay on training and held-out whole runs, then practice MPC integration and a collision-monitored real run",
        "limitations": [
            "P0 r01 is the practice training run and P0 r02 is the held-out practice run; one held-out practice run does not establish run-level uncertainty.",
            "Open-plane high-steer data is development-domain training only when explicitly supplied; it is not practice-transfer evidence.",
            "Held-out open-plane captures are scored separately by run and never enter fitting; they do not establish whole-lap accuracy.",
            "Practice plant scores initialize from simulator truth; they do not score localization or the closed-loop estimator/controller stack.",
            "The residual model has not been inserted into recursive rollout or production MPC/odometry.",
            "No coefficients, residual gain, or runtime behavior are promoted by this fit.",
        ],
    }
    (output_dir / "yaw_residual_evaluation.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "train_run_ids": report["train_run_ids"],
        "validation_run_ids": report["validation_run_ids"],
        "training_samples": len(train),
        "validation_samples": len(validation),
        "production_validation": validation_metrics["production"],
        "candidate_validation": validation_metrics["residual_candidate"],
        "one_step_holdout_improved_rmse": report["one_step_holdout_improved_rmse"],
        "promotion_decision": report["promotion_decision"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
