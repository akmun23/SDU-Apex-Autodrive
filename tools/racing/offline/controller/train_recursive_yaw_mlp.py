#!/usr/bin/env python3
"""Train a smooth yaw-response map with a recursive sequence objective.

The recurrence is explicit and contractive: r[k+1] = rho*r[k] + f(x[k]),
with either fixed rho=0.8 or a bounded, input-conditioned response pole and a
small differentiable MLP. Training uses training-run sequences only. Whole-run
race-domain development validation selects the checkpoint; paired-slew
validation and frozen practice are reported separately. During rollout,
future body u/v and steering-command history are oracle inputs, making this a
yaw-only model diagnostic, not a full plant.
"""

from __future__ import annotations

import copy
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from tools.racing.offline.controller import (
    evaluate_mpc_practice_yaw_residual as yaw,
    fit_causal_history_yaw_lateral_plant as teacher,
)
from tools.racing.offline.controller import fit_stable_ar_yaw_response as stable


OUTPUT = yaw.ANALYSIS_DIR / "recursive_yaw_mlp_20261012.json"
ARTIFACT = yaw.ANALYSIS_DIR / "recursive_yaw_mlp_20261012.pt"
RHO = 0.8
WINDOW = 40  # 1 second at the fixed 40 Hz sample grid
WINDOW_STRIDE = 8
EPOCHS = 80
EVAL_EVERY = 5
WINDOWS_PER_EPOCH = 8192
BATCH_SIZE = 128
SEED = 20261012
MODEL_CONFIG = {"hidden_sizes": [64, 64], "activation": "tanh"}
MODEL_KWARGS = {
    "loss": "squared_error", "learning_rate": 0.05,
    "max_iter": 120, "max_leaf_nodes": 15,
    "min_samples_leaf": 120, "l2_regularization": 1.0,
    "early_stopping": False,
}


class YawResidualMLP(nn.Module):
    def __init__(self, input_size: int, adaptive_pole: bool = False) -> None:
        super().__init__()
        self.adaptive_pole = adaptive_pole
        output_size = 2 if adaptive_pole else 1
        self.network = nn.Sequential(
            nn.Linear(input_size, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, output_size),
        )
        if adaptive_pole:
            # Initialize the pole near the established fixed-rho comparator.
            output_layer = self.network[-1]
            with torch.no_grad():
                output_layer.bias[0] = math.log((RHO - 0.2) / (0.98 - RHO))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def _feature_columns() -> tuple[list[str], list[int]]:
    names = teacher.FEATURE_NAMES
    selected_names, _ = stable._features()
    return selected_names, [names.index(name) for name in selected_names]


def _split_roster(path: Path) -> dict[str, set[str]]:
    with np.load(path, allow_pickle=False) as archive:
        ids = archive["run_ids"].astype(str)
        splits = archive["run_splits"].astype(str)
    return {split: set(ids[splits == split]) for split in np.unique(splits)}


def _feature_sequences(data: dict[str, Any], run_ids: set[str],
                       columns: list[int], mean: np.ndarray,
                       scale: np.ndarray, window_steps: int
                       ) -> tuple[list[dict[str, Any]], int]:
    sequences = []
    skipped = 0
    state, pose = data["state"], data["pose"]
    for sequence_id, (start_raw, end_raw) in enumerate(data["sequence_bounds"]):
        run_index = int(data["sequence_run_index"][sequence_id])
        run_id = str(data["run_ids"][run_index])
        if run_id not in run_ids:
            continue
        start, end = int(start_raw), int(end_raw)
        first = start + teacher.HISTORY
        if end - 1 - first < window_steps:
            skipped += 1
            continue
        if (not np.isfinite(state[start:end]).all()
                or not np.isfinite(pose[start:end]).all()
                or not np.all(data["sensor_valid"][start:end])):
            skipped += 1
            continue
        raw_features = np.asarray([
            teacher._features(state[index - teacher.HISTORY:index + 1])[columns]
            for index in range(first, end - 1)
        ], dtype=np.float32)
        current_r = state[first:end - 1, 2].astype(np.float32)
        next_r = state[first + 1:end, 2].astype(np.float32)
        length = len(raw_features)
        if length < window_steps:
            skipped += 1
            continue
        sequences.append({
            "run_id": run_id,
            "sequence_id": sequence_id,
            "x": (raw_features - mean.astype(np.float32))
                 / scale.astype(np.float32),
            "r_current": current_r,
            "r_next": next_r,
            "u": state[first:end, 0].astype(np.float32),
            "v": state[first:end, 1].astype(np.float32),
            "pose": pose[first:end, :3].astype(np.float64),
        })
    return sequences, skipped


def _model_outputs(model: YawResidualMLP, x: np.ndarray,
                   device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        output = model(torch.as_tensor(x, dtype=torch.float32, device=device))
    return output.detach().cpu().numpy().astype(np.float64)


def _advance(model: YawResidualMLP, outputs: np.ndarray,
             current_rate: float) -> tuple[float, float]:
    if model.adaptive_pole:
        pole = 0.2 + 0.78 / (1.0 + math.exp(-float(np.clip(outputs[0], -30, 30))))
        equilibrium = float(outputs[1])
        return pole * current_rate + (1.0 - pole) * equilibrium, pole
    return RHO * current_rate + float(outputs[0]), RHO


def _seq_metrics(model: YawResidualMLP, sequence: dict[str, Any],
                 device: torch.device) -> dict[str, Any]:
    outputs = _model_outputs(model, sequence["x"], device)
    true_current = sequence["r_current"].astype(np.float64)
    truth = sequence["r_next"].astype(np.float64)
    predicted = np.empty_like(truth)
    rate = float(true_current[0])
    poles = []
    for index, output in enumerate(outputs):
        rate, pole = _advance(model, output, rate)
        poles.append(pole)
        predicted[index] = rate
    error = predicted - truth
    full = {
        "yaw_rate_rmse_radps": float(np.sqrt(np.mean(error ** 2))),
        "yaw_rate_p95_abs_radps": float(np.quantile(np.abs(error), 0.95)),
    }
    result: dict[str, Any] = {"steps": int(len(error)), "full_sequence": full}
    for horizon in (4, 10, 20, 40):
        if len(error) < horizon:
            continue
        result[f"{horizon * teacher.DT:g}s"] = {
            "yaw_rate_rmse_radps": float(
                np.sqrt(np.mean(error[:horizon] ** 2))),
        }

    # Integrate yaw with oracle u/v so heading and path errors isolate the yaw
    # response map. This is not full plant validation.
    pose = sequence["pose"]
    heading = float(pose[0, 2])
    xy = np.asarray(pose[0, :2], dtype=np.float64).copy()
    heading_errors, position_errors = [], []
    for index, next_rate in enumerate(predicted):
        current_rate = (float(true_current[0]) if index == 0
                        else float(predicted[index - 1]))
        rate_mid = 0.5 * (current_rate + float(next_rate))
        heading_mid = heading + 0.5 * teacher.DT * rate_mid
        u_mid = 0.5 * (float(sequence["u"][index])
                       + float(sequence["u"][index + 1]))
        v_mid = 0.5 * (float(sequence["v"][index])
                       + float(sequence["v"][index + 1]))
        xy += teacher.DT * np.asarray((
            u_mid * math.cos(heading_mid) - v_mid * math.sin(heading_mid),
            u_mid * math.sin(heading_mid) + v_mid * math.cos(heading_mid)))
        heading = math.atan2(math.sin(heading + teacher.DT * 0.5 * (
            current_rate + float(next_rate))), math.cos(heading + teacher.DT * 0.5 * (
                current_rate + float(next_rate))))
        truth_pose = pose[index + 1]
        heading_errors.append(math.atan2(
            math.sin(heading - truth_pose[2]), math.cos(heading - truth_pose[2])))
        position_errors.append(float(np.linalg.norm(xy - truth_pose[:2])))
    result["full_sequence"].update({
        "heading_rmse_rad": float(np.sqrt(np.mean(np.asarray(heading_errors) ** 2))),
        "position_rmse_m": float(np.sqrt(np.mean(np.asarray(position_errors) ** 2))),
    })
    if model.adaptive_pole:
        result["response_pole"] = {
            "mean": float(np.mean(poles)),
            "p05": float(np.quantile(poles, 0.05)),
            "p95": float(np.quantile(poles, 0.95)),
        }
    return result


def _score_sequences(model: YawResidualMLP, sequences: list[dict[str, Any]],
                     device: torch.device) -> dict[str, Any]:
    by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sequence in sequences:
        metrics = _seq_metrics(model, sequence, device)
        metrics["eligible"] = True
        by_run[sequence["run_id"]].append(metrics)
    return {
        run_id: {"sequences": len(metrics), "eligible_sequences": len(metrics),
                 "sequence_metrics": metrics}
        for run_id, metrics in sorted(by_run.items())
    }


def _macro(scores: dict[str, Any]) -> dict[str, Any]:
    names = [f"{count * teacher.DT:g}s" for count in (4, 10, 20, 40)]
    by_metric: dict[str, list[float]] = defaultdict(list)
    for run in scores.values():
        for metric in names + ["full_sequence"]:
            values = [
                (item[metric]["yaw_rate_rmse_radps"] if metric != "full_sequence"
                 else item[metric]["yaw_rate_rmse_radps"])
                for item in run["sequence_metrics"] if metric in item]
            if values:
                by_metric[metric].append(float(np.mean(values)))
    output = {name: float(np.mean(values)) for name, values in by_metric.items()}
    output["independent_runs"] = min(
        (len(values) for values in by_metric.values()), default=0)
    required = names + ["full_sequence"]
    output["selection_composite_radps"] = (
        float(np.mean([output[name] for name in required]))
        if all(name in output for name in required) else None)
    return output


def _one_step_metrics(model: YawResidualMLP,
                      sequences: list[dict[str, Any]],
                      device: torch.device) -> dict[str, Any]:
    by_run: dict[str, list[np.ndarray]] = defaultdict(list)
    for sequence in sequences:
        outputs = _model_outputs(model, sequence["x"], device)
        current = sequence["r_current"].astype(np.float64)
        if model.adaptive_pole:
            prediction = np.asarray([
                _advance(model, output, rate)[0]
                for output, rate in zip(outputs, current)])
        else:
            prediction = RHO * current + outputs[:, 0]
        error = prediction - sequence["r_next"].astype(np.float64)
        by_run[sequence["run_id"]].append(error)

    def metrics(error: np.ndarray) -> dict[str, Any]:
        absolute = np.abs(error)
        return {
            "samples": int(len(error)),
            "rmse_radps": float(np.sqrt(np.mean(error ** 2))),
            "p95_abs_radps": float(np.quantile(absolute, 0.95)),
            "fraction_abs_le_0p1": float(np.mean(absolute <= 0.1)),
        }
    per_run = {run_id: metrics(np.concatenate(parts))
               for run_id, parts in sorted(by_run.items())}
    pooled = metrics(np.concatenate([
        np.concatenate(parts) for parts in by_run.values()]))
    return {**pooled, "per_run": per_run}


def run(window_steps: int = WINDOW, epochs: int = EPOCHS,
        windows_per_epoch: int = WINDOWS_PER_EPOCH,
        batch_size: int = BATCH_SIZE,
        output_path: Path = OUTPUT,
        artifact_path: Path = ARTIFACT,
        adaptive_pole: bool = False) -> dict[str, Any]:
    output_path, artifact_path = output_path.resolve(), artifact_path.resolve()
    if output_path.exists() or artifact_path.exists():
        raise FileExistsError("refusing to overwrite recursive yaw MLP run")
    if window_steps < max(40, 4 * teacher.HISTORY) or epochs < 1:
        raise ValueError("recursive training window/epoch count is too small")
    if windows_per_epoch < batch_size or windows_per_epoch % batch_size:
        raise ValueError("windows_per_epoch must be a positive batch-size multiple")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(2)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    benchmark = json.loads(yaw.BENCHMARK.read_text(encoding="utf-8"))
    base_path = yaw._workspace_path(benchmark["dataset"])
    if teacher._sha256(base_path) != benchmark["dataset_sha256"]:
        raise ValueError("frozen practice benchmark hash changed")
    base_splits = _split_roster(base_path)
    paired_splits = _split_roster(teacher.PAIRED_DATASET)
    sealed = (base_splits.get("test", set())
              | base_splits.get("final_test", set()))
    base_train = teacher._collect_rows(base_path, "train")
    paired_train = teacher._collect_rows(teacher.PAIRED_DATASET, "train")
    flat_train = teacher._combine_training((base_train, paired_train))
    if set(flat_train["runs"]) & (sealed | teacher.VALIDATION_IDS):
        raise ValueError("training runs overlap sealed or frozen practice captures")
    feature_names, columns = _feature_columns()
    train_x = flat_train["x"][:, columns]
    weights = flat_train["weight"]
    mean = np.average(train_x, axis=0, weights=weights)
    variance = np.average((train_x - mean) ** 2, axis=0, weights=weights)
    scale = np.sqrt(np.maximum(variance, 1e-8))

    train_sequences = []
    skipped_train = 0
    for source in (base_train, paired_train):
        sequences, skipped = _feature_sequences(
            source, set(flat_train["runs"]), columns, mean, scale, window_steps)
        train_sequences.extend(sequences)
        skipped_train += skipped
    base_dev_ids = base_splits.get("validation", set()) - teacher.VALIDATION_IDS
    paired_dev_ids = paired_splits.get("validation", set())
    base_dev_data = teacher._collect_rows(base_path, "validation", base_dev_ids)
    paired_dev_data = teacher._collect_rows(
        teacher.PAIRED_DATASET, "validation", paired_dev_ids)
    practice_data = teacher._collect_rows(
        base_path, "validation", teacher.VALIDATION_IDS)
    base_dev_sequences, skipped_base_dev = _feature_sequences(
        base_dev_data, base_dev_ids, columns, mean, scale, window_steps)
    paired_dev_sequences, skipped_paired_dev = _feature_sequences(
        paired_dev_data, paired_dev_ids, columns, mean, scale, window_steps)
    practice_sequences, skipped_practice = _feature_sequences(
        practice_data, teacher.VALIDATION_IDS, columns, mean, scale,
        window_steps)
    if not train_sequences or not base_dev_sequences or not practice_sequences:
        raise ValueError("insufficient complete sequences for recursive training/evaluation")

    model = YawResidualMLP(len(columns), adaptive_pole=adaptive_pole).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    runs = sorted(set(sequence["run_id"] for sequence in train_sequences))
    run_sequences: dict[str, list[int]] = defaultdict(list)
    windows_by_run: dict[str, list[tuple[int, int]]] = defaultdict(list)
    sequence_tensors = []
    for sequence_index, sequence in enumerate(train_sequences):
        run_sequences[sequence["run_id"]].append(sequence_index)
        sequence_tensors.append({
            "x": torch.as_tensor(sequence["x"], dtype=torch.float32, device=device),
            "r_current": torch.as_tensor(sequence["r_current"], dtype=torch.float32,
                                          device=device),
            "r_next": torch.as_tensor(sequence["r_next"], dtype=torch.float32,
                                       device=device),
        })
        last_start = len(sequence["x"]) - window_steps
        starts = list(range(0, last_start + 1, WINDOW_STRIDE))
        if starts[-1] != last_start:
            starts.append(last_start)
        windows_by_run[sequence["run_id"]].extend(
            (sequence_index, start) for start in starts)
    runs = [run for run in runs if windows_by_run[run]]
    if len(runs) < 3:
        raise ValueError("too few independent training runs have a full 1 s window")

    rng = np.random.default_rng(SEED)
    best_score = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    dev_history = []
    train_history = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.integers(0, len(runs), size=windows_per_epoch)
        sampled = []
        for run_index in order:
            run = runs[int(run_index)]
            choices = windows_by_run[run]
            sampled.append(choices[int(rng.integers(0, len(choices)))])
        epoch_losses = []
        for batch_start in range(0, windows_per_epoch, batch_size):
            batch = sampled[batch_start:batch_start + batch_size]
            x_batch = torch.stack([
                sequence_tensors[sequence_index]["x"][start:start + window_steps]
                for sequence_index, start in batch])
            current_batch = torch.stack([
                sequence_tensors[sequence_index]["r_current"][start:start + window_steps]
                for sequence_index, start in batch])
            truth_batch = torch.stack([
                sequence_tensors[sequence_index]["r_next"][start:start + window_steps]
                for sequence_index, start in batch])
            learned = model(x_batch.reshape(-1, len(columns))).reshape(
                len(batch), window_steps, 2 if adaptive_pole else 1)
            rate = current_batch[:, 0]
            rollout = []
            for step in range(window_steps):
                if adaptive_pole:
                    pole = 0.2 + 0.78 * torch.sigmoid(learned[:, step, 0])
                    equilibrium = learned[:, step, 1]
                    rate = pole * rate + (1.0 - pole) * equilibrium
                else:
                    rate = RHO * rate + learned[:, step, 0]
                rollout.append(rate)
            rollout = torch.stack(rollout, dim=1)
            recursive_loss = torch.mean((rollout - truth_batch) ** 2)
            if adaptive_pole:
                poles = 0.2 + 0.78 * torch.sigmoid(learned[:, :, 0])
                equilibrium = learned[:, :, 1]
                teacher_prediction = (poles * current_batch
                                      + (1.0 - poles) * equilibrium)
                teacher_loss = torch.mean((teacher_prediction - truth_batch) ** 2)
                pole_regularizer = torch.mean((poles - RHO) ** 2)
                loss = recursive_loss + 0.25 * teacher_loss + 0.01 * pole_regularizer
            else:
                teacher_loss = torch.mean((RHO * current_batch + learned[:, :, 0]
                                            - truth_batch) ** 2)
                pole_regularizer = torch.zeros((), device=device)
                loss = recursive_loss + 0.25 * teacher_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_losses.append((float(recursive_loss.detach().cpu()),
                                 float(teacher_loss.detach().cpu()),
                                 float(pole_regularizer.detach().cpu())))
        train_history.append({
            "epoch": epoch,
            "recursive_loss_radps2": float(np.mean([x[0] for x in epoch_losses])),
            "teacher_forced_loss_radps2": float(np.mean([x[1] for x in epoch_losses])),
            "pole_regularizer": float(np.mean([x[2] for x in epoch_losses])),
        })
        if epoch % EVAL_EVERY == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                race_scores = _score_sequences(model, base_dev_sequences, device)
                dev_macro = _macro(race_scores)
                score = dev_macro["selection_composite_radps"]
            if score is None or not math.isfinite(float(score)):
                raise ValueError("development selection score is not finite")
            dev_history.append({"epoch": epoch, "race_domain_macro": dev_macro})
            if float(score) < best_score:
                best_score = float(score)
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        race_scores = _score_sequences(model, base_dev_sequences, device)
        paired_scores = _score_sequences(model, paired_dev_sequences, device)
        practice_scores = _score_sequences(model, practice_sequences, device)
    one_step_practice = _one_step_metrics(model, practice_sequences, device)

    torch.save({
        "state_dict": model.cpu().state_dict(),
        "input_feature_names": feature_names,
        "input_mean": mean.astype(np.float32),
        "input_scale": scale.astype(np.float32),
        "rho": RHO,
        "adaptive_pole": adaptive_pole,
        "pole_bounds": [0.2, 0.98] if adaptive_pole else [RHO, RHO],
        "model_config": MODEL_CONFIG,
        "training_run_ids": flat_train["runs"],
        "training_dataset_hashes": {
            "race_domain": base_train["dataset_sha256"],
            "paired_throttle_slew": paired_train["dataset_sha256"],
        },
    }, artifact_path)

    report = {
        "study": ("differentiable yaw-response MLP trained with recursive loss; "
                  "bounded state-conditioned pole" if adaptive_pole else
                  "differentiable yaw-response MLP trained with recursive loss; fixed pole"),
        "offline_only": True,
        "runtime_model_changed": False,
        "simulator_was_run": False,
        "device": str(device),
        "rho": RHO,
        "adaptive_pole": adaptive_pole,
        "pole_bounds": [0.2, 0.98] if adaptive_pole else [RHO, RHO],
        "window_steps": window_steps,
        "window_duration_s": window_steps * teacher.DT,
        "model_config": MODEL_CONFIG,
        "feature_names": feature_names,
        "training": {
            "run_ids": flat_train["runs"],
            "sequence_count": len(train_sequences),
            "samples": int(len(flat_train["y"])),
            "skipped_sequences": skipped_train,
            "sampling": "uniform over training runs, then uniform over 1 s windows",
            "epochs": epochs,
            "windows_per_epoch": windows_per_epoch,
            "batch_size": batch_size,
            "seed": SEED,
            "loss": "recursive yaw-rate MSE over 1 s + 0.25 × teacher-forced one-step MSE",
            "history": train_history,
        },
        "development_checkpoint_selection": {
            "run_ids": sorted(base_dev_ids),
            "metric": "equal run-macro mean of recursive yaw-rate RMSE at 0.1/0.25/0.5/1.0 s and full sequence",
            "selected_epoch": best_epoch,
            "selected_composite_radps": best_score,
            "epoch_history": dev_history,
            "paired_slew_validation_used_for_selection": False,
            "practice_used_for_selection": False,
        },
        "development_recursive": {
            "race_domain": {"macro": _macro(race_scores), "per_run": race_scores,
                            "skipped_sequences": skipped_base_dev},
            "paired_throttle_slew": {
                "macro": _macro(paired_scores), "per_run": paired_scores,
                "skipped_sequences": skipped_paired_dev},
        },
        "frozen_practice_transfer": {
            "run_ids": sorted(teacher.VALIDATION_IDS),
            "recursive": practice_scores,
            "recursive_macro": _macro(practice_scores),
            "one_step_teacher_forced": one_step_practice,
            "skipped_sequences": skipped_practice,
            "selection_use": False,
        },
        "sealed_test_and_final_test_run_ids_in_source_archive": sorted(sealed),
        "sealed_numeric_arrays_access_policy": (
            "archives containing test/final_test are refused before numeric "
            "decompression"),
        "oracle_rollout_inputs": [
            "future body u/v from simulator truth",
            "future steering command history"],
        "limitations": [
            "This predicts yaw rate only; it is not an offline full-vehicle simulator.",
            "Future body u/v are oracle inputs, not predicted states.",
            "The model is differentiable but has not been exported or integrated into MPC/odometry.",
            "The practice evidence is two captures; independent repeated practice runs are not available here.",
            "The adaptive pole is a first-order response hypothesis; its fitted value is not a simulator physical parameter.",
        ],
        "artifact": str(artifact_path.relative_to(teacher.ROOT)),
    }
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-steps", type=int, default=WINDOW)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--windows-per-epoch", type=int,
                        default=WINDOWS_PER_EPOCH)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--artifact", type=Path, default=ARTIFACT)
    parser.add_argument("--adaptive-pole", action="store_true",
                        help="learn a smooth, bounded pole from current exogenous state")
    args = parser.parse_args()
    result = run(args.window_steps, args.epochs, args.windows_per_epoch,
                 args.batch_size, args.output, args.artifact, args.adaptive_pole)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "device": result["device"],
        "selected_epoch": result["development_checkpoint_selection"]["selected_epoch"],
        "dev_selection": result["development_checkpoint_selection"][
            "selected_composite_radps"],
        "practice_one_step": result["frozen_practice_transfer"][
            "one_step_teacher_forced"],
        "practice_recursive": result["frozen_practice_transfer"][
            "recursive_macro"],
        "artifact": result["artifact"],
    }, indent=2))
