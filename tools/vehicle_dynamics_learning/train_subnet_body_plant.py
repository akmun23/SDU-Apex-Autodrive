#!/usr/bin/env python3
"""Train the repository-native SUBNET body plant on the frozen body dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from subnet_body_plant import SubnetBodyPlant


ROOT = Path(__file__).resolve().parents[2]
RESET_ROOT = ROOT / "live_runs/derived_dynamics_learning_20260928/subnet_reset_20261004"
DATASET = RESET_ROOT / "body_sysid_v1.npz"
DEFAULT_OUTPUT = RESET_ROOT / "native_subnet_fit_v1"
HISTORY = 12
STATE_ORDER = 9
ROLLOUT_STEPS = 40
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
SEED = 101
DEFAULT_GRADIENT_CLIP_NORM = 10.0
DEFAULT_PATIENCE_EPOCHS = 80


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_training_data(path: Path, rollout_steps: int = ROLLOUT_STEPS):
    if rollout_steps < 1:
        raise ValueError("rollout_steps must be positive")
    with np.load(path, allow_pickle=False) as data:
        inputs = np.asarray(data["inputs"], dtype=np.float32)
        outputs = np.asarray(data["outputs"], dtype=np.float32)
        bounds = np.asarray(data["sequence_bounds"], dtype=np.int64)
        sequence_split = data["sequence_split"].astype(str)
        sequence_run_id = data["sequence_run_id"].astype(str)
        row_split = data["split"].astype(str)
    if set(np.unique(row_split)) - {"train", "validation"}:
        raise ValueError("test/final-test rows are forbidden in this training dataset")
    train_rows = row_split == "train"
    u_mean, u_std = inputs[train_rows].mean(0), inputs[train_rows].std(0)
    y_mean, y_std = outputs[train_rows].mean(0), outputs[train_rows].std(0)
    if not np.isfinite(np.r_[u_mean, u_std, y_mean, y_std]).all():
        raise ValueError("non-finite training normalization statistics")
    if np.any(u_std <= 1e-9) or np.any(y_std <= 1e-9):
        raise ValueError("degenerate training normalization scale")

    def window_starts_by_run(split: str) -> dict[str, np.ndarray]:
        starts_by_run: dict[str, list[np.ndarray]] = {}
        for (begin, end), role, run_id in zip(
                bounds, sequence_split, sequence_run_id):
            begin, end = int(begin), int(end)
            if role == split and end - begin >= HISTORY + rollout_steps:
                starts_by_run.setdefault(run_id, []).append(np.arange(
                    begin, end - HISTORY - rollout_steps + 1, dtype=np.int64))
        return {
            run_id: np.concatenate(parts)
            for run_id, parts in starts_by_run.items()
        }

    train_starts_by_run = window_starts_by_run("train")
    train_starts = np.concatenate(list(train_starts_by_run.values()))
    if len(train_starts) < BATCH_SIZE:
        raise ValueError(
            f"expanded dataset has too few training windows: {len(train_starts)}")
    if not np.any(sequence_split == "validation"):
        raise ValueError("whole-run validation sequences are required")
    return (inputs, outputs, bounds, sequence_split, sequence_run_id,
            train_starts, train_starts_by_run, u_mean, u_std, y_mean, y_std)


def _validate(model: SubnetBodyPlant, inputs: np.ndarray, outputs: np.ndarray,
              bounds: np.ndarray, sequence_split: np.ndarray,
              sequence_run_id: np.ndarray, y_std: np.ndarray,
              device: torch.device, aggregation: str) -> float:
    if aggregation not in {"sample_pooled", "run_macro"}:
        raise ValueError(f"unsupported validation aggregation: {aggregation}")
    squared_error = 0.0
    sample_count = 0
    squared_by_run: dict[str, float] = {}
    count_by_run: dict[str, int] = {}
    model.eval()
    with torch.no_grad():
        for (begin, end), split, run_id in zip(
                bounds, sequence_split, sequence_run_id):
            begin, end = int(begin), int(end)
            if split != "validation" or end - begin <= HISTORY:
                continue
            z = model.encode_history(
                torch.as_tensor(inputs[begin:begin + HISTORY], device=device),
                torch.as_tensor(outputs[begin:begin + HISTORY], device=device))
            future_u = torch.as_tensor(inputs[begin + HISTORY:end], device=device)
            prediction = model.rollout(z, future_u).cpu().numpy()
            truth = outputs[begin + HISTORY:end]
            normalized_error = (prediction - truth) / y_std[None, :]
            if not np.isfinite(normalized_error).all():
                return float("inf")
            run_squared = float(np.square(normalized_error).sum())
            squared_error += run_squared
            sample_count += normalized_error.size
            key = str(run_id)
            squared_by_run[key] = squared_by_run.get(key, 0.0) + run_squared
            count_by_run[key] = count_by_run.get(key, 0) + normalized_error.size
    if sample_count == 0:
        raise ValueError("validation split has no recursive samples")
    if aggregation == "run_macro":
        return float(np.mean([
            np.sqrt(squared_by_run[run] / count_by_run[run])
            for run in sorted(squared_by_run)
        ]))
    return float(np.sqrt(squared_error / sample_count))


def train(dataset: Path = DATASET, output_dir: Path = DEFAULT_OUTPUT,
          epochs: int = 1000, device_name: str = "cuda",
          gradient_clip_norm: float = DEFAULT_GRADIENT_CLIP_NORM,
          patience_epochs: int = DEFAULT_PATIENCE_EPOCHS,
          sampling_policy: str = "window_uniform",
          validation_aggregation: str = "sample_pooled",
          rollout_steps: int = ROLLOUT_STEPS,
          updates_per_epoch_limit: int | None = None,
          learning_rate: float = LEARNING_RATE) -> dict:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "subnet_body_plant_best_v1.npz"
    trace_path = output_dir / "validation_trace.jsonl"
    report_path = output_dir / "training_report.json"
    if any(path.exists() for path in (checkpoint, trace_path, report_path)):
        raise FileExistsError(f"refusing to overwrite an existing run in {output_dir}")
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if not np.isfinite(gradient_clip_norm) or gradient_clip_norm <= 0:
        raise ValueError("gradient clip norm must be finite and positive")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning rate must be finite and positive")
    if patience_epochs < 1:
        raise ValueError("patience epochs must be positive")
    if rollout_steps < 1:
        raise ValueError("rollout steps must be positive")
    if updates_per_epoch_limit is not None and updates_per_epoch_limit < 1:
        raise ValueError("updates per epoch must be positive when specified")
    if sampling_policy not in {"window_uniform", "run_balanced"}:
        raise ValueError(f"unsupported sampling policy: {sampling_policy}")
    if validation_aggregation not in {"sample_pooled", "run_macro"}:
        raise ValueError(
            f"unsupported validation aggregation: {validation_aggregation}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(1)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    (inputs, outputs, bounds, sequence_split, sequence_run_id, starts,
     starts_by_run, u_mean, u_std, y_mean, y_std) = _load_training_data(
        dataset, rollout_steps)
    model = SubnetBodyPlant(u_mean, u_std, y_mean, y_std,
                            history_steps=HISTORY, state_order=STATE_ORDER).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    offsets = np.arange(HISTORY + rollout_steps, dtype=np.int64)
    rng = np.random.default_rng(SEED)
    full_updates_per_epoch = len(starts) // BATCH_SIZE
    updates_per_epoch = (full_updates_per_epoch if updates_per_epoch_limit is None
                         else min(full_updates_per_epoch, updates_per_epoch_limit))
    if updates_per_epoch < 1:
        raise ValueError("training data has too few windows for one optimizer update")
    run_names = sorted(starts_by_run)
    run_start_arrays = [starts_by_run[name] for name in run_names]
    trace_file = trace_path.open("x", encoding="utf-8")
    fit_started = time.perf_counter()
    best_validation = float("inf")
    best_epoch = 0
    completed_epochs = 0
    stale_epochs = 0
    total_clipped_updates = 0
    stop_reason = "epoch_budget"
    try:
        for epoch in range(1, epochs + 1):
            model.train()
            epoch_loss = 0.0
            gradient_norm_sum = 0.0
            gradient_norm_max = 0.0
            clipped_updates = 0
            for _ in range(updates_per_epoch):
                if sampling_policy == "window_uniform":
                    selected_starts = starts[rng.integers(
                        0, len(starts), size=BATCH_SIZE)]
                else:
                    selected_runs = rng.integers(
                        0, len(run_start_arrays), size=BATCH_SIZE)
                    selected_starts = np.empty(BATCH_SIZE, dtype=np.int64)
                    for run_index, run_starts in enumerate(run_start_arrays):
                        mask = selected_runs == run_index
                        count = int(mask.sum())
                        if count:
                            selected_starts[mask] = run_starts[
                                rng.integers(0, len(run_starts), size=count)]
                rows = selected_starts[:, None] + offsets[None, :]
                u_batch = torch.as_tensor(inputs[rows], device=device)
                y_batch = torch.as_tensor(outputs[rows], device=device)
                u_hist = (u_batch[:, :HISTORY] - model.input_mean) / model.input_std
                y_hist = (y_batch[:, :HISTORY] - model.output_mean) / model.output_std
                encoded = model.encoder(torch.cat(
                    (u_hist.flatten(1), y_hist.flatten(1)), dim=1))
                future_u = ((u_batch[:, HISTORY:] - model.input_mean)
                            / model.input_std)
                target = ((y_batch[:, HISTORY:] - model.output_mean)
                          / model.output_std)
                predictions = []
                state = encoded
                for step in range(rollout_steps):
                    predictions.append(model.hn(state))
                    state = model.fn(torch.cat((state, future_u[:, step]), dim=1))
                loss = F.mse_loss(torch.stack(predictions, dim=1), target)
                if not torch.isfinite(loss):
                    stop_reason = f"nonfinite_training_loss_epoch_{epoch}"
                    break
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=gradient_clip_norm)
                gradient_norm_value = float(gradient_norm.detach())
                if not np.isfinite(gradient_norm_value):
                    stop_reason = f"nonfinite_gradient_norm_epoch_{epoch}"
                    break
                gradient_norm_sum += gradient_norm_value
                gradient_norm_max = max(gradient_norm_max, gradient_norm_value)
                clipped_updates += gradient_norm_value > gradient_clip_norm
                optimizer.step()
                epoch_loss += float(loss.detach())
            if stop_reason.startswith(("nonfinite_training_loss",
                                       "nonfinite_gradient_norm")):
                break
            validation_score = _validate(
                model, inputs, outputs, bounds, sequence_split,
                sequence_run_id, y_std, device, validation_aggregation)
            completed_epochs = epoch
            if np.isfinite(validation_score) and validation_score < best_validation:
                best_validation = validation_score
                best_epoch = epoch
                stale_epochs = 0
                model.cpu().save_npz(checkpoint)
                model.to(device)
            else:
                stale_epochs += 1
            total_clipped_updates += clipped_updates
            row = {
                "epoch": epoch,
                "optimizer_updates": epoch * updates_per_epoch,
                "mean_training_loss": epoch_loss / updates_per_epoch,
                "validation_sim_nrms": validation_score,
                "mean_preclip_gradient_norm": gradient_norm_sum / updates_per_epoch,
                "max_preclip_gradient_norm": gradient_norm_max,
                "clipped_optimizer_updates": clipped_updates,
                "sampling_policy": sampling_policy,
                "validation_aggregation": validation_aggregation,
                "rollout_steps": rollout_steps,
                "learning_rate": learning_rate,
                "updates_per_epoch": updates_per_epoch,
                "training_run_count": len(run_names),
                "elapsed_wall_s": time.perf_counter() - fit_started,
                "utc": datetime.now(timezone.utc).isoformat(),
            }
            trace_file.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")
            trace_file.flush()
            os.fsync(trace_file.fileno())
            if not np.isfinite(validation_score):
                stop_reason = f"nonfinite_validation_score_epoch_{epoch}"
                break
            if stale_epochs >= patience_epochs:
                stop_reason = f"validation_plateau_{patience_epochs}_epochs"
                break
    finally:
        trace_file.close()

    if best_epoch == 0:
        raise RuntimeError("no finite validation checkpoint was produced")
    report = {
        "status": "completed" if stop_reason == "epoch_budget" else "early_stopped",
        "stop_reason": stop_reason,
        "dataset": str(dataset.resolve()),
        "dataset_sha256": sha256(dataset),
        "epochs_requested": epochs,
        "gradient_clip_norm": gradient_clip_norm,
        "learning_rate": learning_rate,
        "patience_epochs": patience_epochs,
        "sampling_policy": sampling_policy,
        "validation_aggregation": validation_aggregation,
        "rollout_steps": rollout_steps,
        "epochs_completed": completed_epochs,
        "optimizer_updates": completed_epochs * updates_per_epoch,
        "windows_per_epoch": len(starts),
        "updates_per_epoch": updates_per_epoch,
        "updates_per_epoch_limit": updates_per_epoch_limit,
        "training_runs": run_names,
        "training_run_count": len(run_names),
        "validation_runs": sorted(set(
            sequence_run_id[sequence_split == "validation"].tolist())),
        "best_validation_epoch": best_epoch,
        "best_validation_sim_nrms": best_validation,
        "clipped_optimizer_updates": total_clipped_updates,
        "wall_time_s": time.perf_counter() - fit_started,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "training_runs_only": True,
        "truth_or_future_sensors_used_during_rollout": False,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return report


def finalize_existing_run(dataset: Path, output_dir: Path,
                          stop_reason: str,
                          gradient_clip_norm: float = DEFAULT_GRADIENT_CLIP_NORM,
                          patience_epochs: int = DEFAULT_PATIENCE_EPOCHS,
                          updates_per_epoch_limit: int | None = None,
                          learning_rate: float = LEARNING_RATE) -> dict:
    """Record the best saved checkpoint after an intentional early stop."""
    output_dir = output_dir.resolve()
    checkpoint = output_dir / "subnet_body_plant_best_v1.npz"
    trace_path = output_dir / "validation_trace.jsonl"
    report_path = output_dir / "training_report.json"
    if report_path.exists():
        raise FileExistsError(f"refusing to replace {report_path}")
    for path in (dataset, checkpoint, trace_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    trace = [json.loads(line) for line in trace_path.read_text(
        encoding="utf-8").splitlines() if line.strip()]
    if not trace:
        raise ValueError("cannot finalize an empty validation trace")
    scores = np.asarray([row.get("validation_sim_nrms", np.nan)
                         for row in trace], dtype=np.float64)
    finite = np.isfinite(scores)
    if not finite.any():
        raise ValueError("interrupted fit has no finite validation checkpoint")
    best_index = int(np.argmin(np.where(finite, scores, np.inf)))
    final = trace[-1]
    rollout_steps = int(final.get("rollout_steps", ROLLOUT_STEPS))
    loaded = _load_training_data(dataset, rollout_steps)
    windows_per_epoch = len(loaded[5])
    training_runs = sorted(loaded[6])
    validation_runs = sorted(set(
        loaded[4][loaded[3] == "validation"].tolist()))
    report = {
        "status": "early_stopped_and_finalized",
        "stop_reason": stop_reason,
        "dataset": str(dataset.resolve()),
        "dataset_sha256": sha256(dataset),
        "epochs_requested": 1000,
        "epochs_completed": int(final["epoch"]),
        "optimizer_updates": int(final["optimizer_updates"]),
        "windows_per_epoch": windows_per_epoch,
        "updates_per_epoch": int(final.get(
            "updates_per_epoch", windows_per_epoch // BATCH_SIZE)),
        "updates_per_epoch_limit": updates_per_epoch_limit,
        "training_runs": training_runs,
        "training_run_count": len(training_runs),
        "validation_runs": validation_runs,
        "validation_run_count": len(validation_runs),
        "best_validation_epoch": int(trace[best_index]["epoch"]),
        "best_validation_sim_nrms": float(scores[best_index]),
        "gradient_clip_norm": gradient_clip_norm,
        "learning_rate": learning_rate,
        "patience_epochs": patience_epochs,
        "rollout_steps": rollout_steps,
        "wall_time_s": float(final.get("elapsed_wall_s", 0.0)),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "validation_trace": str(trace_path),
        "validation_trace_rows": len(trace),
        "clipped_optimizer_updates": int(sum(
            int(row.get("clipped_optimizer_updates", 0)) for row in trace)),
        "sampling_policy": final.get("sampling_policy", "window_uniform"),
        "validation_aggregation": final.get(
            "validation_aggregation", "sample_pooled"),
        "training_runs_only": True,
        "truth_or_future_sensors_used_during_rollout": False,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gradient-clip-norm", type=float,
                        default=DEFAULT_GRADIENT_CLIP_NORM)
    parser.add_argument("--patience-epochs", type=int,
                        default=DEFAULT_PATIENCE_EPOCHS)
    parser.add_argument("--sampling-policy",
                        choices=("window_uniform", "run_balanced"),
                        default="window_uniform")
    parser.add_argument("--validation-aggregation",
                        choices=("sample_pooled", "run_macro"),
                        default="sample_pooled")
    parser.add_argument("--rollout-steps", type=int, default=ROLLOUT_STEPS,
                        help="recursive future body samples in each training window")
    parser.add_argument("--updates-per-epoch", type=int,
                        help="optional cap to bound long-rollout epoch cost")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--finalize-existing-run", action="store_true",
                        help="write a report for an interrupted fit's best checkpoint")
    parser.add_argument("--stop-reason", default="early_stopped_by_operator")
    args = parser.parse_args()
    if args.finalize_existing_run:
        report = finalize_existing_run(args.dataset, args.output_dir,
                                       args.stop_reason,
                                       args.gradient_clip_norm,
                                       args.patience_epochs,
                                       args.updates_per_epoch,
                                       args.learning_rate)
    else:
        report = train(args.dataset, args.output_dir, args.epochs, args.device,
                       args.gradient_clip_norm, args.patience_epochs,
                       args.sampling_policy, args.validation_aggregation,
                       args.rollout_steps, args.updates_per_epoch,
                       args.learning_rate)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
