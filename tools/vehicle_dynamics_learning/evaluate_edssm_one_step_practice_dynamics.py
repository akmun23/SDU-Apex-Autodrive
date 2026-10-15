#!/usr/bin/env python3
"""Teacher-forced one-step acceleration audit on unseen practice captures.

The checkpoints are frozen. Each row is initialized from its own preceding
80-sample history and current simulator-truth state, then predicts one 25-ms
transition. Future truth is used only as a label. Test/final-test captures are
not loaded or scored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from tools.vehicle_dynamics_learning.effective_race_teacher import (
    ACCELERATION_NAMES,
    DT_S,
    HISTORY_STEPS,
    physical_state_from_dataset,
    raw_encoder_history_features,
    wheel_innovation_history_features,
    generalized_acceleration_targets,
)
from tools.vehicle_dynamics_learning.score_effective_race_teacher import (
    _load_model,
)
from tools.vehicle_dynamics_learning.evaluate_recursive_longitudinal_candidate import (
    _bundle_lags,
    _features as hgb_features,
    _state as filtered_wheel_state_from_dataset,
)
from tools.vehicle_dynamics_learning.train_nssm import _load_dataset


ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "full_modeling_reset_20261001/replacement_offline_sim_raceline_20261003/"
    "full_throttle_domain_v1/encoder_fixed40hz_v1")
DEFAULT_DATASET = MODEL_ROOT / "practice_dynamics_raw_wheels_fixed40hz.npz"
DEFAULT_OUTPUT = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "edssm_hgb_one_step_practice_dynamics_v4_20261009.json")
DEFAULT_HGB_BUNDLE = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "longitudinal_with_fixed40_wheels_candidate_20261009/"
    "longitudinal_raw_wheel_candidate.joblib")
CHECKPOINTS = {
    "five_second_parent": MODEL_ROOT / "model_fixed40hz_seed101_v1/best.pt",
    "wheel_head_body_distilled": (
        MODEL_ROOT / "wheel_head_body_distilled_v1/best.pt"),
    "ten_second_plant": MODEL_ROOT / "model_fixed40hz_10s_seed101_v1/best.pt",
}
STRATA_SPEED = (0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0)
STRATA_ABS_STEERING = (0.0, 0.1, 0.2, 0.3, 0.4, 0.500001)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def eligible_rows(data: dict[str, Any], state: np.ndarray,
                  acceleration: np.ndarray,
                  raw_history_lags: tuple[int, ...]
                  ) -> list[dict[str, int]]:
    splits = np.asarray(data["splits"]).astype(str)
    raw_valid = np.asarray(data["encoder_raw_valid"], dtype=bool)
    dt = np.asarray(data["dt_s"], dtype=np.float64)
    rows = []
    for sequence, ((begin_raw, end_raw), run_raw) in enumerate(
            zip(data["bounds"], data["seq_run"])):
        begin, end, run = int(begin_raw), int(end_raw), int(run_raw)
        if splits[run] != "unseen_practice":
            continue
        for index in range(begin + HISTORY_STEPS - 1, end - 1):
            lag_rows = index - np.asarray(raw_history_lags, dtype=np.int64)
            if (not raw_valid[index] or not raw_valid[index + 1]
                    or not raw_valid[lag_rows].all()
                    or not np.isclose(dt[index + 1], DT_S, rtol=0.0,
                                      atol=1e-7)
                    or not np.isfinite(state[index:index + 2]).all()
                    or not np.isfinite(acceleration[index]).all()):
                continue
            rows.append({"sequence": sequence, "run": run, "index": index})
    return rows


def run_hgb_one_step(bundle_path: Path, data: dict[str, Any],
                     state: np.ndarray,
                     rows: list[dict[str, int]],
                     targets: np.ndarray,
                     batch_size: int) -> dict[str, Any]:
    bundle = joblib.load(bundle_path)
    if not bundle.get("research_only") or not bundle.get("raw_wheel_models"):
        raise ValueError("HGB bundle must be research-only and include raw-wheel heads")
    practice_run_ids = set(np.asarray(data["run_ids"]).astype(str)[
        np.asarray(data["splits"]).astype(str) == "unseen_practice"])
    overlap = practice_run_ids.intersection(bundle.get("training_run_ids", []))
    if overlap:
        raise ValueError(f"HGB candidate trained on practice runs: {sorted(overlap)}")
    lags = _bundle_lags(bundle)
    raw_valid = np.asarray(data["encoder_raw_valid"], dtype=bool)
    if any(not raw_valid[row["index"] - np.asarray(lags)].all()
           for row in rows):
        raise ValueError("paired one-step rows lack HGB raw-history support")
    filtered_state = filtered_wheel_state_from_dataset(data)
    frames = np.asarray(data["frames"], dtype=np.float32)
    raw_truth = np.asarray(data["encoder_raw_surface_mps"], dtype=np.float64)
    run_ids = np.asarray(data["run_ids"]).astype(str)
    per_run_errors: dict[str, dict[str, list[np.ndarray]]] = {}
    for offset in range(0, len(rows), batch_size):
        batch = rows[offset:offset + batch_size]
        current = np.asarray([row["index"] for row in batch], dtype=np.int64)
        history = np.empty((len(batch), max(lags) + 1, 9), dtype=np.float32)
        for row_i, index in enumerate(current):
            source = np.arange(index - max(lags), index + 1)
            history[row_i, :, :7] = filtered_state[source].astype(np.float32)
            history[row_i, :, 7:9] = frames[source, 7:9]
        x = hgb_features(history, lags)
        raw_current = raw_truth[current].astype(np.float32)
        raw_parts = [raw_current]
        for lag in lags:
            raw_parts.append((raw_truth[current] - raw_truth[current - lag])
                             .astype(np.float32))
        x = np.column_stack((x, *raw_parts)).astype(np.float32)
        main_delta = np.column_stack([
            model.predict(x) for model in bundle["models"]]).astype(np.float64)
        aux_delta = np.column_stack([
            model.predict(x) for model in bundle["recursive_aux_models"]
        ]).astype(np.float64)
        raw_delta = np.column_stack([
            model.predict(x) for model in bundle["raw_wheel_models"]
        ]).astype(np.float64)
        hgb_current = filtered_state[current].astype(np.float64)
        hgb_next = hgb_current.copy()
        hgb_next[:, 0] += main_delta[:, 0]
        hgb_next[:, 1] += aux_delta[:, 0]
        hgb_next[:, 2] += aux_delta[:, 1]
        hgb_next[:, 3] += aux_delta[:, 2]
        hgb_next[:, 4] += aux_delta[:, 3]
        hgb_next[:, 5:7] += main_delta[:, 2:4]
        raw_next = raw_current.astype(np.float64) + raw_delta
        truth_next = state[current + 1]
        u_dot_truth = ((truth_next[:, 0] - state[current, 0]) / DT_S)
        ax_current = u_dot_truth - state[current, 2] * state[current, 1]
        r_mid = 0.5 * (state[current, 2] + truth_next[:, 2])
        v_mid = 0.5 * (state[current, 1] + truth_next[:, 1])
        ax_midpoint = u_dot_truth - r_mid * v_mid
        ax_implied_by_u = ((hgb_next[:, 0] - state[current, 0]) / DT_S
                           - state[current, 2] * state[current, 1])
        ax_direct = main_delta[:, 1]
        truth_raw_next = raw_truth[current + 1]
        run_for_rows = run_ids[np.asarray([row["run"] for row in batch])]
        for run_id in sorted(set(run_for_rows)):
            mask = run_for_rows == run_id
            bucket = per_run_errors.setdefault(run_id, {
                "u": [], "v": [], "yaw_rate": [], "ax_direct_mid": [],
                "ax_implied": [], "raw_wheels": [], "raw_persistence": [],
                "u_persistence": [], "filtered_wheels": [],
                "ax_current_target": [], "ax_mid_target": [],
            })
            bucket["u"].append(hgb_next[mask, 0] - truth_next[mask, 0])
            bucket["v"].append(hgb_next[mask, 1] - truth_next[mask, 1])
            bucket["yaw_rate"].append(hgb_next[mask, 2] - truth_next[mask, 2])
            bucket["ax_direct_mid"].append(ax_direct[mask] - ax_midpoint[mask])
            bucket["ax_implied"].append(ax_implied_by_u[mask] - ax_current[mask])
            bucket["raw_wheels"].append(raw_next[mask] - truth_raw_next[mask])
            bucket["raw_persistence"].append(
                raw_truth[current[mask]] - truth_raw_next[mask])
            bucket["u_persistence"].append(
                state[current[mask], 0] - truth_next[mask, 0])
            bucket["filtered_wheels"].append(
                hgb_next[mask, 5:7] - frames[current[mask] + 1, 5:7])
            bucket["ax_current_target"].append(ax_current[mask])
            bucket["ax_mid_target"].append(ax_midpoint[mask])

    per_run = {}
    for run_id, bucket in per_run_errors.items():
        values = {key: np.concatenate(parts, axis=0)
                  for key, parts in bucket.items()}
        def rmse(name: str) -> float:
            return float(np.sqrt(np.mean(values[name] ** 2)))
        per_run[run_id] = {
            "transitions": len(values["u"]),
            "u_rmse_mps": rmse("u"),
            "u_persistence_rmse_mps": rmse("u_persistence"),
            "v_rmse_mps": rmse("v"),
            "yaw_rate_rmse_rps": rmse("yaw_rate"),
            "effective_ax_direct_midpoint_rmse_mps2": rmse("ax_direct_mid"),
            "effective_ax_implied_by_u_current_transport_rmse_mps2": rmse("ax_implied"),
            "effective_ax_zero_rmse_current_transport_mps2": float(
                np.sqrt(np.mean(values["ax_current_target"] ** 2))),
            "effective_ax_zero_rmse_midpoint_mps2": float(
                np.sqrt(np.mean(values["ax_mid_target"] ** 2))),
            "raw_wheel_pair_rmse_mps": float(
                np.sqrt(np.mean(values["raw_wheels"] ** 2))),
            "raw_wheel_pair_persistence_rmse_mps": float(
                np.sqrt(np.mean(values["raw_persistence"] ** 2))),
            "filtered_wheel_pair_rmse_mps": float(
                np.sqrt(np.mean(values["filtered_wheels"] ** 2))),
        }
    metric_names = [name for name in next(iter(per_run.values()))
                    if name != "transitions"]
    return {
        "bundle": str(bundle_path.relative_to(ROOT)),
        "bundle_sha256": sha256(bundle_path),
        "paired_transitions": int(sum(row["transitions"] for row in per_run.values())),
        "per_run": per_run,
        "run_macro_mean": {
            name: float(np.mean([row[name] for row in per_run.values()]))
            for name in metric_names},
    }


def batched_inputs(data: dict[str, Any], state: np.ndarray,
                   rows: list[dict[str, int]], history_extra: np.ndarray | None,
                   indices: np.ndarray) -> tuple[np.ndarray, ...]:
    frames = np.asarray(data["frames"], dtype=np.float32)
    histories = []
    selected_rows = [rows[int(position)] for position in indices]
    for row in selected_rows:
        index = int(row["index"])
        hist_indices = np.arange(index - HISTORY_STEPS + 1, index + 1)
        history = np.column_stack((state[hist_indices], frames[hist_indices, 7:9]))
        if history_extra is not None:
            history = np.column_stack((history, history_extra[hist_indices]))
        histories.append(history)
    current = np.asarray([row["index"] for row in selected_rows], dtype=np.int64)
    history = np.asarray(histories, dtype=np.float32)
    initial = state[current].astype(np.float32)
    delayed = frames[current - 1, 7:9]
    command = frames[current + 1, 7:9][:, None, :]
    return history, initial, delayed, command, current


def run_model(name: str, checkpoint_path: Path,
              data: dict[str, Any], state: np.ndarray,
              targets: np.ndarray, rows: list[dict[str, int]],
              batch_size: int) -> dict[str, Any]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    torch, model, metadata = _load_model(checkpoint_path, "cpu")
    practice_ids = set(np.asarray(data["run_ids"]).astype(str)[
        np.asarray(data["splits"]).astype(str) == "unseen_practice"])
    if metadata.get("wheel_state_source") != "raw_encoder":
        raise ValueError(f"{name} is not a raw-encoder checkpoint")
    overlap = practice_ids.intersection(metadata.get("training_runs", []))
    if overlap:
        raise ValueError(f"{name} trained on practice data: {sorted(overlap)}")
    if not np.array_equal(np.asarray(metadata["state_scale"]).shape, (7,)):
        raise ValueError(f"{name} does not have a seven-state physical model")

    history_extra = (
        raw_encoder_history_features(data)
        if model.include_raw_encoder_history else
        wheel_innovation_history_features(data)
        if model.include_wheel_innovation_history else None)
    run_ids = np.asarray(data["run_ids"]).astype(str)
    prediction_by_run: dict[str, dict[str, list[np.ndarray]]] = {}
    torch.set_num_threads(4)
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            row_batch = rows[start:start + batch_size]
            local_indices = np.arange(len(row_batch), dtype=np.int64)
            history, initial, delayed, command, current = batched_inputs(
                data, state, row_batch, history_extra, local_indices)
            device = torch.device("cpu")
            predicted_state, predicted_acceleration, _, _ = model.rollout(
                torch.as_tensor(initial, dtype=torch.float32, device=device),
                torch.as_tensor(delayed, dtype=torch.float32, device=device),
                torch.as_tensor(history, dtype=torch.float32, device=device),
                torch.as_tensor(command, dtype=torch.float32, device=device))
            next_state = predicted_state[:, 0].cpu().numpy().astype(np.float64)
            next_acceleration = predicted_acceleration[:, 0].cpu().numpy().astype(np.float64)
            expected_u = initial[:, 0] + DT_S * (
                next_acceleration[:, 0] + initial[:, 2] * initial[:, 1])
            expected_v = initial[:, 1] + DT_S * (
                next_acceleration[:, 1] - initial[:, 2] * initial[:, 0])
            expected_yaw_rate = initial[:, 2] + DT_S * next_acceleration[:, 2]
            if not (np.allclose(next_state[:, 0], expected_u, atol=2e-6, rtol=0.0)
                    and np.allclose(next_state[:, 1], expected_v,
                                    atol=2e-6, rtol=0.0)
                    and np.allclose(next_state[:, 2], expected_yaw_rate,
                                    atol=2e-6, rtol=0.0)):
                raise AssertionError("EDSSM one-step output violates its declared body equations")
            truth_next = state[current + 1].astype(np.float64)
            truth_acc = targets[current].astype(np.float64)
            row_run_ids = run_ids[np.asarray([r["run"] for r in row_batch])]
            for run_id in sorted(set(row_run_ids)):
                mask = row_run_ids == run_id
                bucket = prediction_by_run.setdefault(run_id, {
                    "state_error": [], "acceleration_error": [],
                    "state_truth": [], "acceleration_truth": [],
                    "state_persistence_error": [],
                })
                bucket["state_error"].append(next_state[mask] - truth_next[mask])
                bucket["acceleration_error"].append(
                    next_acceleration[mask] - truth_acc[mask])
                bucket["state_truth"].append(truth_next[mask])
                bucket["acceleration_truth"].append(truth_acc[mask])
                bucket["state_persistence_error"].append(
                    state[current[mask]] - truth_next[mask])

    # Per-run RMSE is the statistical unit. Keep all row-level error arrays
    # only until reductions are complete.
    per_run = {}
    for run_id, bucket in prediction_by_run.items():
        state_error = np.concatenate(bucket["state_error"])
        acceleration_error = np.concatenate(bucket["acceleration_error"])
        state_truth = np.concatenate(bucket["state_truth"])
        acceleration_truth = np.concatenate(bucket["acceleration_truth"])
        state_persistence_error = np.concatenate(
            bucket["state_persistence_error"])
        per_run[run_id] = {
            "transitions": int(len(state_error)),
            "state_rmse": np.sqrt(np.mean(state_error ** 2, axis=0)).tolist(),
            "acceleration_rmse": np.sqrt(
                np.mean(acceleration_error ** 2, axis=0)).tolist(),
            "acceleration_zero_persistence_rmse": np.sqrt(
                np.mean(acceleration_truth ** 2, axis=0)).tolist(),
            "acceleration_bias": np.mean(acceleration_error, axis=0).tolist(),
            "state_zero_order_persistence_rmse": np.sqrt(np.mean(
                state_persistence_error ** 2, axis=0)).tolist(),
        }
    del model
    return {
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "checkpoint_sha256": sha256(checkpoint_path),
        "training_objective_mode": metadata.get("training_objective_mode"),
        "run_macro_mean_state_rmse": np.mean(
            np.asarray([row["state_rmse"] for row in per_run.values()]), axis=0).tolist(),
        "run_macro_mean_acceleration_rmse": np.mean(
            np.asarray([row["acceleration_rmse"] for row in per_run.values()]), axis=0).tolist(),
        "run_macro_mean_acceleration_zero_persistence_rmse": np.mean(
            np.asarray([row["acceleration_zero_persistence_rmse"]
                        for row in per_run.values()]), axis=0).tolist(),
        "run_macro_mean_acceleration_bias": np.mean(
            np.asarray([row["acceleration_bias"] for row in per_run.values()]), axis=0).tolist(),
        "run_macro_mean_state_zero_order_persistence_rmse": np.mean(
            np.asarray([row["state_zero_order_persistence_rmse"]
                        for row in per_run.values()]), axis=0).tolist(),
        "per_run": per_run,
    }


def score(dataset_path: Path, hgb_bundle_path: Path, output_path: Path,
          checkpoint_paths: dict[str, Path], batch_size: int) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    data = _load_dataset(dataset_path)
    state = physical_state_from_dataset(data, "raw_encoder").astype(np.float64)
    target_acceleration = generalized_acceleration_targets(
        state, data["bounds"], data["dt_s"], "surface_acceleration")
    hgb_bundle = joblib.load(hgb_bundle_path)
    raw_history_lags = _bundle_lags(hgb_bundle)
    rows = eligible_rows(data, state, target_acceleration, raw_history_lags)
    if not rows:
        raise ValueError("no valid unseen-practice one-step transitions")

    # Exact speed/steering support summary. Counts are descriptive, not
    # independent samples and not claims of uniform envelope coverage.
    frames = np.asarray(data["frames"], dtype=np.float64)
    speed = np.hypot(state[:, 0], state[:, 1])
    abs_steer = np.abs(state[:, 3])
    current = np.asarray([row["index"] for row in rows], dtype=np.int64)
    bins = {}
    for low_s, high_s in zip(STRATA_SPEED[:-1], STRATA_SPEED[1:]):
        for low_d, high_d in zip(STRATA_ABS_STEERING[:-1],
                                 STRATA_ABS_STEERING[1:]):
            key = f"speed_{low_s:g}_{high_s:g}_abssteer_{low_d:g}_{high_d:g}"
            bins[key] = int(np.count_nonzero(
                (speed[current] >= low_s) & (speed[current] < high_s)
                & (abs_steer[current] >= low_d)
                & (abs_steer[current] < high_d)))
    run_ids = np.asarray(data["run_ids"]).astype(str)
    transitions_by_run = {
        run_id: int(sum(run_ids[row["run"]] == run_id for row in rows))
        for run_id in sorted({run_ids[row["run"]] for row in rows})}
    commands = frames[current + 1, 7:9]
    manifest = {
        "run_ids": sorted(transitions_by_run),
        "transitions_by_run": transitions_by_run,
        "speed_mps_p05_p50_p95": np.quantile(speed[current],
                                               (0.05, 0.50, 0.95)).tolist(),
        "abs_physical_steering_rad_p05_p50_p95": np.quantile(
            abs_steer[current], (0.05, 0.50, 0.95)).tolist(),
        "throttle_command_norm_p05_p50_p95": np.quantile(
            commands[:, 1], (0.05, 0.50, 0.95)).tolist(),
        "speed_by_abs_steering_transition_counts": bins,
    }
    results = {}
    for name, checkpoint_path in checkpoint_paths.items():
        results[name] = run_model(name, checkpoint_path, data, state,
                                  target_acceleration, rows, batch_size)
    hgb_result = run_hgb_one_step(
        hgb_bundle_path, data, state, rows, target_acceleration, batch_size)
    report = {
        "schema_version": 1,
        "purpose": "teacher-forced one-step dynamics audit on unseen practice captures",
        "dataset": str(dataset_path.relative_to(ROOT)),
        "dataset_sha256": sha256(dataset_path),
        "hgb_bundle": str(hgb_bundle_path.relative_to(ROOT)),
        "hgb_bundle_sha256": sha256(hgb_bundle_path),
        "split_scored": "unseen_practice",
        "test_and_final_test_used": False,
        "future_truth_or_feedback_used_as_input": False,
        "state_names": [
            "u_com_mps", "v_com_mps", "yaw_rate_rps",
            "steering_actual_rad", "throttle_feedback_norm",
            "rear_left_surface_speed_mps", "rear_right_surface_speed_mps"],
        "acceleration_names": list(ACCELERATION_NAMES),
        "transition_count": len(rows),
        "independent_run_count": len(transitions_by_run),
        "transition_unit": "one exact 25 ms transition; initialized with the preceding 80 simulator-truth-labeled state samples",
        "practice_support": manifest,
        "models": results,
        "hgb_fixed40_one_step_candidate": hgb_result,
        "limitations": [
            "Teacher-forced one-step accuracy is not recursive-rollout accuracy.",
            "Only two practice runs are represented; bins are descriptive and many speed/steering combinations may be empty.",
            "The raw encoder wheel speed is the measured target, not direct tire angular velocity or tire force.",
            "The generalized acceleration target is derived from simulator-truth body velocities and yaw rate at fixed 25 ms intervals.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--hgb-bundle", type=Path, default=DEFAULT_HGB_BUNDLE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--parent", type=Path,
                        default=CHECKPOINTS["five_second_parent"])
    parser.add_argument("--wheel-distilled", type=Path,
                        default=CHECKPOINTS["wheel_head_body_distilled"])
    parser.add_argument("--ten-second", type=Path,
                        default=CHECKPOINTS["ten_second_plant"])
    args = parser.parse_args()
    paths = {
        "five_second_parent": args.parent.resolve(),
        "wheel_head_body_distilled": args.wheel_distilled.resolve(),
        "ten_second_plant": args.ten_second.resolve(),
    }
    report = score(args.dataset.resolve(), args.hgb_bundle.resolve(),
                   args.output.resolve(), paths, args.batch_size)
    print(json.dumps({
        "transitions": report["transition_count"],
        "runs": report["independent_run_count"],
        "support": report["practice_support"],
        "models": {
            name: {
                "state_rmse": model["run_macro_mean_state_rmse"],
                "acceleration_rmse": model["run_macro_mean_acceleration_rmse"],
                "acceleration_zero_persistence_rmse": model[
                    "run_macro_mean_acceleration_zero_persistence_rmse"],
                "acceleration_bias": model["run_macro_mean_acceleration_bias"],
                "state_persistence_rmse": model[
                    "run_macro_mean_state_zero_order_persistence_rmse"],
            }
            for name, model in report["models"].items()},
        "hgb_one_step": report["hgb_fixed40_one_step_candidate"]["run_macro_mean"],
        "output": str(args.output.resolve()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
