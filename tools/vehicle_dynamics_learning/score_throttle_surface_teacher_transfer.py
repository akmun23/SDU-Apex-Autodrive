#!/usr/bin/env python3
"""Score an EDSSM on the independent, reset-isolated throttle capture.

The open-plane throttle export contains packet-aligned simulator telemetry.
This evaluator uses simulator COM velocity/yaw rate and pose as labels, causal
history through the 80-sample context, and only recorded plant commands during
free rollout. It reports errors separately and bootstraps by steering-angle
cluster rather than treating serialized samples as independent.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from tools.vehicle_dynamics_learning.effective_race_teacher import (
    DT_S,
    HISTORY_STEPS,
    integrate_pose,
)
from tools.vehicle_dynamics_learning.four_wheel_greybox import WHEEL_RADIUS_M
from tools.vehicle_dynamics_learning.score_effective_race_teacher import (
    _load_model,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = (REPO_ROOT / "live_runs/derived_dynamics_learning_20260928/"
                  "throttle_surface_40hz_sourcealigned_dataset_20260930/"
                  "throttle_surface_sequences.npz")
DEFAULT_MANIFEST = DEFAULT_SOURCE.with_name("manifest.json")
DEFAULT_CAPTURE = "openplane_throttle_5pct_5deg_20260930_r04"
HORIZONS = {
    "0.025s": 1,
    "0.1s": 4,
    "0.25s": 10,
    "0.75s": 30,
    "2s": 80,
    "5s": 200,
}


def _source_aligned_encoder_rates(
        position_rad: np.ndarray, source_match: np.ndarray,
        packet_sequence: np.ndarray, source_stamp_ns: np.ndarray,
        sequence_bounds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct causal 15–35 ms encoder rates used by the raw-state teacher."""
    position = np.asarray(position_rad, dtype=np.float64)
    matched = np.asarray(source_match, dtype=bool)
    packet = np.asarray(packet_sequence, dtype=np.int64).reshape(-1)
    stamps = np.asarray(source_stamp_ns, dtype=np.int64).reshape(-1)
    bounds = np.asarray(sequence_bounds, dtype=np.int64)
    if (position.ndim != 2 or position.shape[1] != 2
            or matched.shape != position.shape
            or packet.shape != (len(position),)
            or stamps.shape != (len(position),)
            or bounds.ndim != 2 or bounds.shape[1] != 2):
        raise ValueError("source-aligned encoder arrays do not align")
    rates = np.full(position.shape, np.nan, dtype=np.float64)
    valid = np.zeros(len(position), dtype=bool)
    for start_raw, end_raw in bounds:
        start, end = int(start_raw), int(end_raw)
        if start < 0 or end > len(position) or end <= start:
            raise ValueError("invalid throttle sequence bounds")
        if end - start < 2:
            continue
        dt = np.diff(stamps[start:end]).astype(np.float64) / 1e9
        adjacent_packets = np.diff(packet[start:end]) == 1
        source_valid = (matched[start:end - 1]
                        & matched[start + 1:end]).all(axis=1)
        angle_delta = np.diff(position[start:end], axis=0)
        good = (adjacent_packets & source_valid & np.isfinite(angle_delta).all(axis=1)
                & (dt >= 0.015) & (dt <= 0.035))
        rows = np.flatnonzero(good) + start + 1
        rates[rows] = (angle_delta[good] * WHEEL_RADIUS_M
                       / dt[good, None])
        valid[rows] = True
    return rates, valid


def _metric_row(predicted: np.ndarray, truth: np.ndarray,
                predicted_pose: np.ndarray, truth_pose: np.ndarray,
                horizon: int, wheel_valid: np.ndarray | None = None
                ) -> dict[str, float]:
    pred = predicted[:horizon].astype(np.float64)
    target = truth[:horizon].astype(np.float64)
    wheel_valid = (np.ones(horizon, dtype=bool) if wheel_valid is None else
                   np.asarray(wheel_valid[:horizon], dtype=bool))
    if wheel_valid.shape != (horizon,):
        raise ValueError("wheel-valid mask does not match the requested horizon")
    pose_pred = predicted_pose[:horizon].astype(np.float64)
    pose_target = truth_pose[1:horizon + 1].astype(np.float64)
    angle_error = np.arctan2(
        np.sin(pose_pred[:, 2] - pose_target[:, 2]),
        np.cos(pose_pred[:, 2] - pose_target[:, 2]))
    result = {
        "u_com_mps": float(np.mean((pred[:, 0] - target[:, 0]) ** 2)),
        "v_com_mps": float(np.mean((pred[:, 1] - target[:, 1]) ** 2)),
        "yaw_rate_rps": float(np.mean((pred[:, 2] - target[:, 2]) ** 2)),
        "steering_feedback_rad": float(np.mean((pred[:, 3] - target[:, 3]) ** 2)),
        "throttle_feedback_norm": float(np.mean((pred[:, 4] - target[:, 4]) ** 2)),
        "rear_left_wheel_mps": (float(np.mean(
            (pred[wheel_valid, 5] - target[wheel_valid, 5]) ** 2))
            if np.any(wheel_valid) else None),
        "rear_right_wheel_mps": (float(np.mean(
            (pred[wheel_valid, 6] - target[wheel_valid, 6]) ** 2))
            if np.any(wheel_valid) else None),
        "body_speed_mps": float(np.mean((np.hypot(pred[:, 0], pred[:, 1])
                                         - np.hypot(target[:, 0], target[:, 1])) ** 2)),
        "position_radial_m": float(np.mean(np.sum(
            (pose_pred[:, :2] - pose_target[:, :2]) ** 2, axis=1))),
        "heading_rad": float(np.mean(angle_error ** 2)),
    }
    return result


def _cluster_summary(rows: list[dict[str, Any]], metric_names: list[str],
                     seed: int) -> dict[str, Any]:
    by_angle: dict[float, list[dict[str, Any]]] = {}
    for row in rows:
        by_angle.setdefault(row["angle_cluster"], []).append(row["mse"])
    clusters = sorted(by_angle)
    result: dict[str, Any] = {
        "sequence_count": len(rows),
        "steering_angle_cluster_count": len(clusters),
        "rmse": {},
        "rmse_ci95": {},
    }
    rng = np.random.default_rng(seed)
    for metric in metric_names:
        available = {
            angle: [sequence[metric] for sequence in by_angle[angle]
                    if sequence.get(metric) is not None]
            for angle in clusters
        }
        available = {angle: values for angle, values in available.items()
                     if values}
        values = np.asarray([
            np.mean(available[angle]) for angle in sorted(available)
        ], dtype=np.float64)
        result.setdefault("metric_condition_count", {})[metric] = int(sum(
            len(values) for values in available.values()))
        if not len(values):
            result["rmse"][metric] = None
            result["rmse_ci95"][metric] = None
            continue
        result["rmse"][metric] = float(np.sqrt(np.mean(values)))
        if len(values) >= 2:
            draws = rng.integers(0, len(values), size=(5000, len(values)))
            bootstrap = np.sqrt(values[draws].mean(axis=1))
            result["rmse_ci95"][metric] = np.quantile(
                bootstrap, (0.025, 0.975)).tolist()
        else:
            result["rmse_ci95"][metric] = None
    return result


def _steering_angle_breakdown(
        rows: list[dict[str, Any]], metric_names: list[str]
        ) -> dict[str, Any]:
    """Keep signed steering regimes separate for first-divergence diagnosis.

    Each row is one reset-isolated throttle condition. Counts are conditions,
    not independent whole-run replicates; this table is descriptive and must
    not be interpreted as run-level uncertainty.
    """
    by_angle: dict[float, list[dict[str, Any]]] = {}
    for row in rows:
        by_angle.setdefault(row["angle_cluster"], []).append(row)
    result: dict[str, Any] = {}
    for angle in sorted(by_angle):
        angle_rows = by_angle[angle]
        def rmse(metric: str, selected: list[dict[str, Any]]):
            values = [row["mse"][metric] for row in selected
                      if row["mse"][metric] is not None]
            return float(np.sqrt(np.mean(values))) if values else None

        high_throttle_rows = [row for row in angle_rows
                              if row["high_throttle"]]
        result[f"{angle:.4f}"] = {
            "condition_count": len(angle_rows),
            "high_throttle_condition_count": len(high_throttle_rows),
            "start_speed_mps_mean": float(np.mean([
                row["initial_speed_mps"] for row in angle_rows])),
            "rmse": {
                metric: rmse(metric, angle_rows)
                for metric in metric_names
            },
            "high_throttle_rmse": {
                metric: rmse(metric, high_throttle_rows)
                for metric in metric_names
            },
        }
    return result


def score(checkpoints: list[Path], source_path: Path, manifest_path: Path,
          output_path: Path, capture: str, device: str,
          max_speed_mps: float = 12.0) -> dict[str, Any]:
    with np.load(source_path, allow_pickle=False) as archive:
        source = {key: archive[key] for key in (
            "sequence_run_id", "sequence_bounds", "body_state",
            "plant_commands", "actuator_feedback",
            "encoder_surface_mps_100ms", "bridge_debug_telemetry",
            "encoder_position_rad", "encoder_source_stamp_match",
            "packet_sequence", "receipt_time_ns", "source_stamp_offset_ms",
            "dt_s")}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    debug_names = manifest["bridge_debug_telemetry_names"]
    debug_index = {name: index for index, name in enumerate(debug_names)}
    run_ids = source["sequence_run_id"].astype(str)
    selected_sequences = np.flatnonzero(run_ids == capture)
    if not len(selected_sequences):
        raise ValueError(f"capture is absent from exported throttle data: {capture}")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite score: {output_path}")

    sequence_manifest = {
        int(row["sequence_index"]): row for row in manifest["sequences"]
        if row.get("run_id") == capture
    }
    torch, model_records = None, []
    for checkpoint_path in checkpoints:
        loaded_torch, model, metadata = _load_model(checkpoint_path, device)
        if capture in set(metadata.get("training_runs", [])):
            raise ValueError(f"held-out capture leaked into training: {capture}")
        if bool(metadata.get("include_raw_encoder_history", False)):
            raise ValueError(
                "this transfer scorer does not supply extra raw-wheel history features")
        torch = loaded_torch
        model_records.append((checkpoint_path, model, metadata))

    per_model: dict[str, Any] = {}
    for checkpoint_path, model, metadata in model_records:
        if (int(metadata.get("history_steps", -1)) != HISTORY_STEPS
                or bool(metadata.get("include_roll_state", False))):
            raise ValueError("checkpoint history length does not match evaluator")
        sequence_scores: list[dict[str, Any]] = []
        excluded_speed = 0
        prepared: list[dict[str, Any]] = []
        wheel_state_source = str(metadata.get(
            "wheel_state_source", "filtered_odometry"))
        if wheel_state_source not in ("filtered_odometry", "raw_encoder"):
            raise ValueError(f"unsupported wheel state source: {wheel_state_source}")
        if wheel_state_source == "raw_encoder":
            source_stamps = (source["receipt_time_ns"].reshape(-1).astype(np.int64)
                             + np.rint(source["source_stamp_offset_ms"].reshape(-1)
                                       * 1e6).astype(np.int64))
            raw_wheels, raw_wheel_valid = _source_aligned_encoder_rates(
                source["encoder_position_rad"],
                source["encoder_source_stamp_match"],
                source["packet_sequence"], source_stamps,
                source["sequence_bounds"])
            wheel_signal = np.where(
                raw_wheel_valid[:, None], raw_wheels,
                source["encoder_surface_mps_100ms"])
        else:
            raw_wheel_valid = np.isfinite(
                source["encoder_surface_mps_100ms"]).all(axis=1)
            wheel_signal = source["encoder_surface_mps_100ms"]
        for sequence_id in selected_sequences:
            start, end = map(int, source["sequence_bounds"][sequence_id])
            if end - start < HISTORY_STEPS + max(HORIZONS.values()) + 1:
                continue
            if not np.allclose(source["dt_s"][start:end], DT_S,
                               rtol=0.0, atol=1e-7):
                raise ValueError("throttle transfer capture is not fixed 40 Hz")
            feedback = source["actuator_feedback"][start:end]
            commands = source["plant_commands"][start:end]
            wheels = wheel_signal[start:end]
            wheel_valid = raw_wheel_valid[start:end]
            debug = source["bridge_debug_telemetry"][start:end]
            body_truth = debug[:, [
                debug_index["simulator_velocity_x_mps"],
                debug_index["simulator_velocity_y_mps"],
                debug_index["simulator_angular_z_rps"],
            ]]
            pose_truth = debug[:, [
                debug_index["simulator_position_x_m"],
                debug_index["simulator_position_y_m"],
                debug_index["simulator_euler_z_rad"],
            ]]
            if not np.isfinite(body_truth).all() or not np.isfinite(pose_truth).all():
                continue
            frames = np.column_stack((
                source["body_state"][start:end], feedback, wheels, commands))
            valid_rows = np.isfinite(frames).all(axis=1)
            good_rows = np.flatnonzero(valid_rows)
            if not len(good_rows):
                continue
            first_valid = int(good_rows[0])
            if not np.all(valid_rows[first_valid:]):
                continue
            body_truth = body_truth[first_valid:]
            pose_truth = pose_truth[first_valid:]
            frames = frames[first_valid:]
            commands = commands[first_valid:]
            feedback = feedback[first_valid:]
            wheels = wheels[first_valid:]
            state = np.column_stack((body_truth, feedback, wheels))
            history_end = HISTORY_STEPS
            future_end = history_end + max(HORIZONS.values())
            history = np.column_stack((
                state[:history_end], frames[:history_end, 7:9]))
            initial = state[history_end - 1]
            delayed = frames[history_end - 2, 7:9]
            future_commands = commands[history_end:future_end]
            target = state[history_end:future_end]
            truth_speed = np.hypot(target[:, 0], target[:, 1])
            valid_horizons = {
                name: steps for name, steps in HORIZONS.items()
                if float(np.max(truth_speed[:steps])) <= max_speed_mps
            }
            if not valid_horizons:
                excluded_speed += 1
                continue
            row_meta = sequence_manifest.get(int(sequence_id), {})
            angle = float(row_meta.get(
                "steering_command_rad", commands[history_end - 1, 0]))
            start_throttle = float(row_meta.get(
                "throttle_start_norm", commands[history_end - 1, 1]))
            end_throttle = float(row_meta.get(
                "throttle_end_norm", commands[history_end, 1]))
            prepared.append({
                "sequence_index": int(sequence_id),
                "phase_index": int(row_meta.get("phase_index", sequence_id)),
                "angle_cluster": round(angle, 4),
                "steering_abs_rad": abs(angle),
                "initial_speed_mps": float(np.hypot(
                    initial[0], initial[1])),
                "throttle_start_norm": start_throttle,
                "throttle_end_norm": end_throttle,
                "high_throttle": max(start_throttle, end_throttle) > 0.50,
                "high_steering": abs(angle) >= 0.35,
                "initial": initial,
                "delayed": delayed,
                "history": history,
                "commands": future_commands,
                "target": target,
                "wheel_valid": wheel_valid[history_end:future_end],
                "wheel_valid_count": int(np.count_nonzero(
                    wheel_valid[history_end:future_end])),
                "wheel_target_source": (
                    "causal source-aligned 15-35 ms encoder rate"
                    if wheel_state_source == "raw_encoder"
                    else "source-aligned 100 ms encoder-angle difference"),
                "initial_pose": pose_truth[history_end - 1],
                "truth_pose": pose_truth[history_end - 1:future_end],
                "valid_horizons": valid_horizons,
            })
        batch_size = 24
        for batch_start in range(0, len(prepared), batch_size):
            batch = prepared[batch_start:batch_start + batch_size]
            with torch.no_grad():
                predicted, _, _, _ = model.rollout(
                    torch.as_tensor(np.stack([row["initial"] for row in batch]),
                                    dtype=torch.float32, device=device),
                    torch.as_tensor(np.stack([row["delayed"] for row in batch]),
                                    dtype=torch.float32, device=device),
                    torch.as_tensor(np.stack([row["history"] for row in batch]),
                                    dtype=torch.float32, device=device),
                    torch.as_tensor(np.stack([row["commands"] for row in batch]),
                                    dtype=torch.float32, device=device))
                pose_pred = integrate_pose(
                    torch, predicted,
                    torch.as_tensor(np.stack([row["initial_pose"] for row in batch]),
                                    dtype=torch.float32, device=device),
                    torch.as_tensor(np.stack([row["initial"] for row in batch]),
                                    dtype=torch.float32, device=device))
            predicted_batch = predicted.cpu().numpy()
            pose_batch = pose_pred.cpu().numpy()
            for index, row in enumerate(batch):
                for horizon_name, steps in row["valid_horizons"].items():
                    sequence_scores.append({
                        **{key: value for key, value in row.items()
                           if key not in ("initial", "delayed", "history",
                                          "commands", "target", "initial_pose",
                                          "truth_pose", "valid_horizons",
                                          "wheel_valid")},
                        "horizon": horizon_name,
                        "mse": _metric_row(
                            predicted_batch[index], row["target"],
                            pose_batch[index], row["truth_pose"], steps,
                            row["wheel_valid"]),
                    })
        report_groups: dict[str, Any] = {}
        for horizon_name in HORIZONS:
            horizon_rows = [row for row in sequence_scores
                            if row["horizon"] == horizon_name]
            subsets = {
                "all": horizon_rows,
                "high_throttle": [row for row in horizon_rows
                                  if row["high_throttle"]],
                "high_steering": [row for row in horizon_rows
                                  if row["high_steering"]],
                "high_throttle_and_steering": [row for row in horizon_rows
                                               if row["high_throttle"]
                                               and row["high_steering"]],
            }
            report_groups[horizon_name] = {
                name: _cluster_summary(
                    [dict(row, mse=row["mse"]) for row in rows],
                    list(next(iter(rows), {"mse": {}})["mse"].keys()),
                    seed=20261003)
                for name, rows in subsets.items()
            }
            report_groups[horizon_name]["per_signed_steering_angle"] = (
                _steering_angle_breakdown(
                    horizon_rows,
                    list(next(iter(horizon_rows), {"mse": {}})["mse"].keys())))
        model_name = checkpoint_path.parent.name
        per_model[model_name] = {
            "checkpoint": str(checkpoint_path),
            "max_truth_speed_mps": max_speed_mps,
            "wheel_state_source": wheel_state_source,
            "wheel_target_definition": (
                "causal source-aligned 15-35 ms encoder rate"
                if wheel_state_source == "raw_encoder"
                else "source-aligned 100 ms encoder-angle difference"),
            "history_seconds": (HISTORY_STEPS - 1) * DT_S,
            "future_inputs": ["recorded steering and throttle commands only"],
            "future_sensor_or_truth_feedback_used": False,
            "sequence_count_scored_at_any_horizon": len({
                row["sequence_index"] for row in sequence_scores}),
            "sequences_excluded_above_speed_envelope": excluded_speed,
            "horizons": report_groups,
        }

    result = {
        "schema_version": 1,
        "capture": capture,
        "source_dataset": str(source_path),
        "capture_use_note": (
            "The evaluated EDSSM checkpoints exclude this capture from training; "
            "older, separate GRU comparators did use the r04 capture for fitting."),
        "evaluated_checkpoints_exclude_capture_from_training": True,
        "ground_truth": (
            "packet-aligned simulator debug COM velocity, yaw rate, pose; "
            "wheel labels are signal-matched to each checkpoint: filtered-odometry "
            "models use the source-aligned 100 ms encoder-angle difference; raw-wheel "
            "models use causal source-aligned 15-35 ms encoder rates"),
        "statistical_unit": "steering-command angle cluster",
        "max_truth_speed_mps": max_speed_mps,
        "models": per_model,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--capture", default=DEFAULT_CAPTURE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-speed-mps", type=float, default=12.0)
    args = parser.parse_args()
    report = score(args.checkpoint, args.source, args.manifest, args.output,
                   args.capture, args.device, args.max_speed_mps)
    print(json.dumps({
        "capture": report["capture"],
        "models": {name: {
            "scored_sequences": model["sequence_count_scored_at_any_horizon"],
            "excluded_speed": model["sequences_excluded_above_speed_envelope"],
        } for name, model in report["models"].items()},
        "output": str(args.output.resolve()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
