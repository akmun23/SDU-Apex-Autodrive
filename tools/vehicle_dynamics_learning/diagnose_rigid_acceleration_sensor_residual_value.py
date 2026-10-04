#!/usr/bin/env python3
"""Whole-run test of wheel mismatch and IMU-roll value for the current plant.

This diagnostic asks whether current-sample wheel and roll measurements explain
the frozen no-latent acceleration plant's teacher-forced one-step residuals.
It never feeds future measurements into a recursive plant rollout and does not
modify or train the plant checkpoint.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tools.vehicle_dynamics_learning.rigid_acceleration_history_plant import (
    RigidAccelerationHistoryTransition,
    midpoint_acceleration_from_transition,
)
from tools.vehicle_dynamics_learning.run_history_context_sufficiency import (
    CONTEXT_STEPS,
    _history_window,
    _normalization,
    pack_history,
)
from tools.vehicle_dynamics_learning.run_truncated_history_transition import ROOT
from tools.vehicle_dynamics_learning.train_augmented_state_space_plant import (
    DEFAULT_DYNAMIC,
    DEFAULT_PRACTICE,
    _load_data,
    _training_windows_and_stats,
    sha256_file,
)


CHECKPOINT = (
    ROOT / "live_runs/derived_dynamics_learning_20260928"
    / "full_modeling_reset_20261001"
    / "replacement_offline_sim_raceline_20261003"
    / "full_throttle_domain_v1/next_phase_after_2129427"
    / "history_context_sufficiency_v1"
    / "rigid_acceleration_history_direct_supervision_5s_v1/checkpoint.pt")
OUTPUT = (
    ROOT / "live_runs/derived_dynamics_learning_20260928"
    / "full_modeling_reset_20261001"
    / "replacement_offline_sim_raceline_20261003"
    / "full_throttle_domain_v1/next_phase_after_2129427"
    / "history_context_sufficiency_v1"
    / "rigid_acceleration_sensor_residual_value_20261004.json")
DT_S = 0.025
CHANNELS = ("acceleration_x_mps2", "acceleration_y_mps2",
            "yaw_acceleration_rps2")
MODEL_NAMES = ("state_command_context", "plus_rear_wheel_mismatch",
               "plus_imu_roll_and_roll_rate")
MAX_TRAIN_ROWS_PER_RUN = 15000
MAX_ITER = 40
RUN_BOOTSTRAP_REPLICATES = 5000


def _sha256(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _transition_rows(data, capture_index: int, split_name: str,
                     imu_path: Path, config, allowed_run_ids: set[str] | None = None
                     ) -> dict[str, dict[str, np.ndarray]]:
    capture = data.captures[capture_index]
    with np.load(imu_path, allow_pickle=False) as archive:
        attitude = np.asarray(archive["imu_attitude_frames"], dtype=np.float32)
        attitude_valid = np.asarray(archive["imu_attitude_valid"], dtype=bool)
        if (attitude.shape != (len(capture.frames), 4)
                or attitude_valid.shape != (len(capture.frames),)
                or not np.array_equal(np.asarray(archive["packet_sequence"]),
                                      capture.packet)):
            raise ValueError("IMU labels are not packet-aligned to the model capture")

    result: dict[str, dict[str, list[np.ndarray]]] = defaultdict(
        lambda: defaultdict(list))
    feature_width = {"base": 13, "wheel": 16, "roll": 18}
    for sequence_index, ((begin_raw, end_raw), run_raw) in enumerate(
            zip(capture.bounds, capture.sequence_run)):
        run_index = int(run_raw)
        run_id = str(capture.run_ids[run_index])
        if (str(capture.splits[run_index]) != split_name
                or (allowed_run_ids is not None
                    and run_id not in allowed_run_ids)):
            continue
        begin, end = int(begin_raw), int(end_raw)
        for row in range(CONTEXT_STEPS["2.0s"] - 1, end - begin - 1):
            absolute = begin + row
            previous = absolute - 1
            if (not attitude_valid[absolute]
                    or not attitude_valid[absolute + 1]
                    or not np.isfinite(attitude[absolute, [0, 2]]).all()
                    or not np.isfinite(capture.input_features[
                        absolute - CONTEXT_STEPS["2.0s"] + 1:absolute + 1]).all()
                    or not np.isfinite(capture.body[[previous, absolute,
                                                     absolute + 1]]).all()
                    or not np.isfinite(capture.frames[
                        previous, [3, 4, 7, 8]]).all()
                    or not np.isfinite(capture.frames[
                        absolute, [3, 4, 5, 6, 7, 8]]).all()):
                continue

            body = capture.body[absolute]
            previous_body = capture.body[previous]
            frame = capture.frames[absolute]
            previous_frame = capture.frames[previous]
            wheel_mean = float(np.mean(frame[5:7]))
            wheel_delta = float(frame[5] - frame[6])
            mismatch = wheel_mean - float(body[0])
            base = np.asarray((
                *body,
                *frame[3:5],
                *frame[7:9],
                *((frame[3:5] - previous_frame[3:5]) / DT_S),
                *((body - previous_body) / DT_S),
                np.hypot(body[0], body[1]),
            ), dtype=np.float32)
            wheel_features = np.asarray((mismatch, abs(mismatch), wheel_delta),
                                        dtype=np.float32)
            roll_features = np.asarray(attitude[absolute, [0, 2]],
                                       dtype=np.float32)
            roll_target = attitude[absolute + 1, [0, 2]].astype(np.float32)
            if base.shape != (feature_width["base"],):
                raise RuntimeError(f"base diagnostic feature width {base.shape}")

            raw_history = _history_window(
                capture, sequence_index, row, CONTEXT_STEPS["2.0s"])
            history, mask = pack_history(
                raw_history, CONTEXT_STEPS["2.0s"],
                np.asarray(config.history_mean[:7], dtype=np.float32),
                np.asarray(config.history_scale[:7], dtype=np.float32))
            current_state = np.concatenate((body, frame[3:5])).astype(np.float32)
            target = midpoint_acceleration_from_transition(
                capture.body[absolute], capture.body[absolute + 1],
                DT_S).astype(np.float32)
            command = frame[7:9].astype(np.float32)
            result[run_id]["history"].append(history)
            result[run_id]["mask"].append(mask)
            result[run_id]["state"].append(current_state)
            result[run_id]["command"].append(command)
            result[run_id]["target"].append(target)
            result[run_id]["base_features"].append(base)
            result[run_id]["wheel_features"].append(wheel_features)
            result[run_id]["roll_features"].append(roll_features)
            result[run_id]["roll_target"].append(roll_target)

    packed: dict[str, dict[str, np.ndarray]] = {}
    for run_id, columns in result.items():
        packed[run_id] = {key: np.asarray(values, dtype=np.float32)
                          for key, values in columns.items()}
    return packed


def _predict_residuals(captures: dict[str, dict[str, np.ndarray]], model,
                       norm_np: dict[str, np.ndarray], device: torch.device,
                       batch_size: int = 2048) -> dict[str, np.ndarray]:
    predictions = {}
    model.eval()
    with torch.no_grad():
        for run_id, rows in captures.items():
            batches = []
            for start in range(0, len(rows["state"]), batch_size):
                end = min(start + batch_size, len(rows["state"]))
                history, mask = rows["history"][start:end], rows["mask"][start:end]
                history_t = torch.as_tensor(history, dtype=torch.float32,
                                            device=device)
                mask_t = torch.as_tensor(mask, dtype=torch.float32, device=device)
                state_t = torch.as_tensor(
                    (rows["state"][start:end] - norm_np["state_mean"])
                    / norm_np["state_scale"], dtype=torch.float32, device=device)
                command_t = torch.as_tensor(
                    (rows["command"][start:end] - norm_np["command_mean"])
                    / norm_np["command_scale"], dtype=torch.float32, device=device)
                _, acceleration = model.forward_with_acceleration(
                    history_t, mask_t, state_t, command_t)
                batches.append(acceleration.cpu().numpy())
            predictions[run_id] = np.concatenate(batches, axis=0)
    return predictions


def _feature_sets(rows: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "state_command_context": rows["base_features"],
        "plus_rear_wheel_mismatch": np.column_stack((
            rows["base_features"], rows["wheel_features"])),
        "plus_imu_roll_and_roll_rate": np.column_stack((
            rows["base_features"], rows["wheel_features"],
            rows["roll_features"])),
    }


def _run_cluster_ci(values: np.ndarray, seed: int) -> list[float] | None:
    """Bootstrap independent captures, never individual autocorrelated rows."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return None
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, len(values), size=(RUN_BOOTSTRAP_REPLICATES,
                                                 len(values)))
    means = values[indexes].mean(axis=1)
    return [float(value) for value in np.quantile(means, (0.025, 0.975))]


def _fit_score_group(rows_by_run: dict[str, dict[str, np.ndarray]],
                      model_residuals: dict[str, np.ndarray],
                      seed: int) -> dict[str, Any]:
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required for residual attribution") from exc
    run_ids = sorted(set(rows_by_run) & set(model_residuals))
    per_model: dict[str, dict[str, dict[str, float]]] = {
        name: defaultdict(dict) for name in MODEL_NAMES}
    if len(run_ids) < 2:
        raise RuntimeError("whole-run residual attribution needs >=2 runs")

    def regressor(random_state: int):
        return HistGradientBoostingRegressor(
            max_iter=MAX_ITER, max_leaf_nodes=7, min_samples_leaf=100,
            learning_rate=0.08, l2_regularization=5.0,
            random_state=random_state)

    for fold, test_run in enumerate(run_ids):
        train_runs = [run for run in run_ids if run != test_run]
        for target_index, channel in enumerate(CHANNELS):
            target_test = model_residuals[test_run][:, target_index]
            for variant_index, name in enumerate(MODEL_NAMES):
                feature_test = _feature_sets(rows_by_run[test_run])[name]
                rng = np.random.default_rng(seed + fold * 31 + target_index)
                kept = []
                for run in train_runs:
                    count = len(rows_by_run[run]["target"])
                    n = min(count, MAX_TRAIN_ROWS_PER_RUN)
                    kept.extend((run, int(i)) for i in np.sort(
                        rng.choice(count, n, replace=False)))
                indexes_by_run = {run: [] for run in train_runs}
                for run, index in kept:
                    indexes_by_run[run].append(index)
                train_indexes = np.concatenate([
                    np.asarray(indexes_by_run[run], dtype=np.int64)
                    + sum(len(rows_by_run[r]["target"]) for r in train_runs[:j])
                    for j, run in enumerate(train_runs)])
                features_by_run = [
                    _feature_sets(rows_by_run[run])[name] for run in train_runs]
                residuals_by_run = [model_residuals[run][:, target_index]
                                    for run in train_runs]
                features_train_all = np.concatenate(features_by_run)
                residual_train_all = np.concatenate(residuals_by_run)
                estimator = regressor(seed + fold * 101 + target_index * 7
                                      + variant_index)
                estimator.fit(features_train_all[train_indexes],
                              residual_train_all[train_indexes])
                corrected = target_test - estimator.predict(feature_test)
                per_model[name][test_run][channel] = float(np.sqrt(
                    np.mean(corrected ** 2)))

    output = {}
    for name in MODEL_NAMES:
        values_by_channel = {}
        for channel in CHANNELS:
            per_run = {run: per_model[name][run][channel] for run in run_ids}
            values_by_channel[channel] = {
                "run_macro_rmse_mps2": float(np.mean(list(per_run.values()))),
                "per_run_rmse_mps2": per_run,
            }
        output[name] = values_by_channel
    paired = {}
    for candidate_index, (candidate, reference) in enumerate((
            ("plus_rear_wheel_mismatch", "state_command_context"),
            ("plus_imu_roll_and_roll_rate", "plus_rear_wheel_mismatch"))):
        paired[candidate] = {}
        for channel_index, channel in enumerate(CHANNELS):
            deltas = np.asarray([
                per_model[candidate][run][channel]
                - per_model[reference][run][channel]
                for run in run_ids], dtype=np.float64)
            paired[candidate][channel] = {
                "candidate_minus_reference_run_macro_delta_mps2": float(deltas.mean()),
                "run_cluster_bootstrap_95pct_ci_mps2": _run_cluster_ci(
                    deltas, seed + candidate_index * 17 + channel_index),
                "per_run_delta_mps2": dict(zip(run_ids, deltas.tolist())),
                "independent_run_count": len(deltas),
            }
    return {"run_macro_scores": output, "paired_incremental_delta": paired,
            "run_ids": run_ids}


def run(device_name: str = "cpu", output_path: Path = OUTPUT
        ) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(1)

    saved = torch.load(CHECKPOINT, map_location=device, weights_only=True)
    metadata = saved["metadata"]
    data, wp20 = _load_data()
    config, _, _ = _training_windows_and_stats(data, wp20)
    norm_np = _normalization(data, config)
    model = RigidAccelerationHistoryTransition(
        norm_np["state_mean"], norm_np["state_scale"],
        np.asarray(metadata["acceleration_mean_train_only"], dtype=np.float32),
        np.asarray(metadata["acceleration_scale_train_only"], dtype=np.float32),
        norm_np["delta_mean"][3:], norm_np["delta_scale"][3:],
        dt_s=float(metadata["dt_s"]),
        rear_axle_to_com_x_m=float(metadata["rear_axle_to_com_x_m"])).to(device)
    model.load_state_dict(saved["state_dict"], strict=True)

    dynamic_rows = _transition_rows(
        data, 0, "validation", DEFAULT_DYNAMIC, config)
    practice_rows = _transition_rows(
        data, 1, "unseen_practice", DEFAULT_PRACTICE, config)
    dynamic_prediction = _predict_residuals(dynamic_rows, model, norm_np, device)
    practice_prediction = _predict_residuals(practice_rows, model, norm_np, device)
    dynamic_residuals = {
        run_id: dynamic_prediction[run_id] - rows["target"]
        for run_id, rows in dynamic_rows.items()}
    practice_residuals = {
        run_id: practice_prediction[run_id] - rows["target"]
        for run_id, rows in practice_rows.items()}

    report = {
        "study": "incremental wheel and IMU-roll value for frozen rigid-acceleration residuals",
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "dynamic_data_sha256": sha256_file(DEFAULT_DYNAMIC),
        "practice_data_sha256": sha256_file(DEFAULT_PRACTICE),
        "cadence_hz": 40.0,
        "transition_horizon_ms": 25,
        "measurement_timing": (
            "same-sample state/command/wheel/IMU features explain next 25ms "
            "acceleration residuals; no future observation is an input"),
        "target_label": "midpoint rigid-body acceleration inferred from adjacent simulator truth body states",
        "cross_validation": (
            "leave-one-whole-run-out HistGradientBoostingRegressor; fold fits "
            "are explanatory diagnostics only and do not change the plant"),
        "feature_models": {
            "state_command_context": [
                "current body u/v/yaw-rate and actuator feedback",
                "current steering/throttle command and their first differences",
                "current body-state first differences and body speed"],
            "plus_rear_wheel_mismatch": [
                "state_command_context",
                "mean rear-wheel surface-speed minus forward body speed",
                "absolute mismatch and left-right rear-wheel difference"],
            "plus_imu_roll_and_roll_rate": [
                "plus_rear_wheel_mismatch", "same-sample IMU roll and roll rate"],
        },
        "dynamic_validation": _fit_score_group(
            dynamic_rows, dynamic_residuals, seed=20261004),
        "practice_diagnostic": _fit_score_group(
            practice_rows, practice_residuals, seed=20261005),
        "transition_count_by_run": {
            "dynamic_validation": {
                run: len(rows["target"]) for run, rows in dynamic_rows.items()},
            "practice_diagnostic": {
                run: len(rows["target"]) for run, rows in practice_rows.items()},
        },
        "future_roll_used_in_rollout": False,
        "simulator_launched": False,
        "plant_checkpoint_changed": False,
        "production_integration": False,
        "interpretation_limit": (
            "This measures whether sensor/context covariates can explain current "
            "teacher-forced one-step errors on held-out whole runs. It does not "
            "prove the feature improves recursive prediction; any eventual plant "
            "must predict roll and wheel states internally."),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n",
                           encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    report = run(args.device, output)
    print(json.dumps({
        "output": output.relative_to(ROOT).as_posix(),
        "dynamic": report["dynamic_validation"]["paired_incremental_delta"],
        "practice": report["practice_diagnostic"]["paired_incremental_delta"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
