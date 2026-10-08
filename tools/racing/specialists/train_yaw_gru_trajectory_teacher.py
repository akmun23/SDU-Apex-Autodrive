#!/usr/bin/env python3
"""Train a causal GRU encoder/decoder for full one-second yaw trajectories.

This is an offline research teacher. Its encoder sees only past permitted
sensor/actuator observations; its decoder sees only the candidate future
steering/throttle commands. Simulator rigid-state yaw rate is used only as the
training/validation target. No test/final-test capture is opened.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    from train_yaw_multihorizon_teacher import (
        FUTURE_COMMAND_STEPS,
        HISTORY_LAGS,
        OBSERVATION_NAMES,
        ROOT,
        _discover_series_with_safe_mixed_archives,
        _observations,
        _read_run,
        _metrics,
    )
except ModuleNotFoundError:
    from tools.racing.specialists.train_yaw_multihorizon_teacher import (
        FUTURE_COMMAND_STEPS,
        HISTORY_LAGS,
        OBSERVATION_NAMES,
        ROOT,
        _discover_series_with_safe_mixed_archives,
        _observations,
        _read_run,
        _metrics,
    )


DEFAULT_OUTPUT = (ROOT / "live_runs/racing_model_diagnostics_20261007/"
                  "yaw_gru_trajectory_teacher_v1")
HISTORY_STEPS = max(HISTORY_LAGS) + 1
FUTURE_STEPS = FUTURE_COMMAND_STEPS
HORIZONS = (1, 4, 10, 20, 30, 40)
SEED = 20261007
HIDDEN_SIZE = 96
NUM_LAYERS = 2
DROPOUT = 0.10
BATCH_SIZE = 128
SAMPLES_PER_EPOCH = 12_288
MAX_EPOCHS = 100
EVAL_EVERY = 2
PATIENCE_EVALS = 8
VALIDATION_STRIDE = 4
FINAL_VALIDATION_STRIDE = 1
LEARNING_RATE = 3.0e-4
WEIGHT_DECAY = 1.0e-5
GRADIENT_CLIP_NORM = 1.0


@dataclass
class RunWindows:
    run_id: str
    split: str
    observations: np.ndarray
    commands: np.ndarray
    yaw_rate: np.ndarray
    starts: np.ndarray


def _exclude_validation_runs(series: list[Any], excluded_run_ids: set[str]
                             ) -> list[Any]:
    """Keep named whole-run validation captures out of checkpoint selection."""
    if not excluded_run_ids:
        return series
    by_id = {row.run_id: row for row in series}
    missing = excluded_run_ids - by_id.keys()
    if missing:
        raise ValueError(f"excluded validation runs were not discovered: {sorted(missing)}")
    wrong_split = sorted(
        run_id for run_id in excluded_run_ids
        if by_id[run_id].split != "validation")
    if wrong_split:
        raise ValueError(
            f"only validation runs may be excluded from selection: {wrong_split}")
    return [row for row in series if row.run_id not in excluded_run_ids]


def window_starts(bounds: np.ndarray, observation_valid: np.ndarray,
                  command_valid: np.ndarray, yaw_valid: np.ndarray,
                  history_steps: int = HISTORY_STEPS,
                  future_steps: int = FUTURE_STEPS) -> np.ndarray:
    """Return current-row indices whose history/controls/targets stay in-run."""
    obs_bad = np.concatenate(([0], np.cumsum(~observation_valid, dtype=np.int64)))
    command_bad = np.concatenate(([0], np.cumsum(~command_valid, dtype=np.int64)))
    yaw_bad = np.concatenate(([0], np.cumsum(~yaw_valid, dtype=np.int64)))
    admitted: list[np.ndarray] = []
    for begin_value, end_value in bounds:
        begin, end = int(begin_value), int(end_value)
        if end - begin < history_steps + future_steps:
            continue
        candidates = np.arange(begin + history_steps - 1,
                               end - future_steps, dtype=np.int64)
        history_begin = candidates - history_steps + 1
        history_ok = (obs_bad[candidates + 1] - obs_bad[history_begin]) == 0
        controls_ok = (command_bad[candidates + future_steps]
                       - command_bad[candidates]) == 0
        target_begin = candidates + 1
        target_end = candidates + future_steps + 1
        targets_ok = (yaw_bad[target_end] - yaw_bad[target_begin]) == 0
        selected = candidates[history_ok & controls_ok & targets_ok]
        if len(selected):
            admitted.append(selected)
    return np.concatenate(admitted) if admitted else np.empty(0, dtype=np.int64)


def gather_windows(run: RunWindows, starts: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gather [past observations, future commands, future yaw targets]."""
    starts = np.asarray(starts, dtype=np.int64)
    history_offsets = np.arange(-(HISTORY_STEPS - 1), 1, dtype=np.int64)
    future_offsets = np.arange(FUTURE_STEPS, dtype=np.int64)
    target_offsets = np.arange(1, FUTURE_STEPS + 1, dtype=np.int64)
    past = run.observations[starts[:, None] + history_offsets[None, :]]
    future = run.commands[starts[:, None] + future_offsets[None, :]]
    targets = run.yaw_rate[starts[:, None] + target_offsets[None, :]]
    return past.astype(np.float32, copy=False), future.astype(np.float32, copy=False), targets.astype(np.float32, copy=False)


def make_model():
    """Build a history encoder and command-conditioned future decoder."""
    import torch
    import torch.nn as nn

    class YawTrajectoryGRU(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.history_encoder = nn.GRU(
                input_size=len(OBSERVATION_NAMES), hidden_size=HIDDEN_SIZE,
                num_layers=NUM_LAYERS, batch_first=True, dropout=DROPOUT)
            self.initial_decoder_state = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
            self.future_decoder = nn.GRU(
                input_size=2, hidden_size=HIDDEN_SIZE, batch_first=True)
            self.yaw_head = nn.Sequential(
                nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE // 2), nn.Tanh(),
                nn.Linear(HIDDEN_SIZE // 2, 1))

        def forward(self, past_observations, future_commands):
            _, encoder_state = self.history_encoder(past_observations)
            decoder_state = torch.tanh(
                self.initial_decoder_state(encoder_state[-1])).unsqueeze(0)
            decoded, _ = self.future_decoder(future_commands, decoder_state)
            return self.yaw_head(decoded).squeeze(-1)

    return YawTrajectoryGRU()


def _make_run(series: Any) -> RunWindows:
    capture = _read_run(series)
    observations = _observations(capture)
    commands = capture.sensors[:, 7:9].astype(np.float32, copy=False)
    yaw_rate = capture.rigid[:, 12].astype(np.float32, copy=False)
    observation_valid = (
        capture.sensor_valid & capture.attitude_valid
        & np.isfinite(observations).all(axis=1))
    command_valid = capture.sensor_valid & np.isfinite(commands).all(axis=1)
    yaw_valid = np.isfinite(yaw_rate)
    starts = window_starts(
        capture.bounds, observation_valid, command_valid, yaw_valid)
    return RunWindows(series.run_id, series.split, observations, commands,
                      yaw_rate, starts)


def _normalizers(training: list[RunWindows]) -> dict[str, np.ndarray]:
    rows = np.concatenate([
        run.observations[run.starts] for run in training if len(run.starts)])
    center = np.median(rows, axis=0)
    q25, q75 = np.quantile(rows, (0.25, 0.75), axis=0)
    scale = np.maximum((q75 - q25) / 1.349, 1.0e-3)
    yaw = np.concatenate([
        run.yaw_rate[run.starts + 1] for run in training if len(run.starts)])
    yaw_center = np.asarray(np.median(yaw), dtype=np.float32)
    yaw_q25, yaw_q75 = np.quantile(yaw, (0.25, 0.75))
    yaw_scale = np.asarray(max((yaw_q75 - yaw_q25) / 1.349, 0.1),
                            dtype=np.float32)
    return {
        "observation_center": center.astype(np.float32),
        "observation_scale": scale.astype(np.float32),
        "yaw_center": yaw_center,
        "yaw_scale": yaw_scale,
    }


def _normalize(past: np.ndarray, future: np.ndarray, targets: np.ndarray,
               normalizers: dict[str, np.ndarray]
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = normalizers["observation_center"]
    scale = normalizers["observation_scale"]
    past = np.clip((past - center[None, None, :])
                   / scale[None, None, :], -12.0, 12.0)
    command_center = center[7:9]
    command_scale = scale[7:9]
    future = np.clip((future - command_center[None, None, :])
                     / command_scale[None, None, :], -12.0, 12.0)
    targets = ((targets - normalizers["yaw_center"])
               / normalizers["yaw_scale"])
    return (past.astype(np.float32, copy=False),
            future.astype(np.float32, copy=False),
            targets.astype(np.float32, copy=False))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_safe(report), indent=2,
                               sort_keys=True) + "\n", encoding="utf-8")


def _evaluate(model, runs: list[RunWindows], normalizers: dict[str, np.ndarray],
              device, stride: int = 1) -> dict[str, Any]:
    import torch

    errors: dict[int, list[np.ndarray]] = {horizon: [] for horizon in HORIZONS}
    per_run: dict[str, dict[str, Any]] = {}
    model.eval()
    with torch.no_grad():
        for run in runs:
            run_errors: dict[int, list[np.ndarray]] = {
                horizon: [] for horizon in HORIZONS}
            starts = run.starts[::stride]
            for offset in range(0, len(starts), BATCH_SIZE):
                selected = starts[offset:offset + BATCH_SIZE]
                past, future, targets = gather_windows(run, selected)
                past, future, targets = _normalize(
                    past, future, targets, normalizers)
                prediction = model(
                    torch.from_numpy(past).to(device),
                    torch.from_numpy(future).to(device))
                predicted_yaw = (prediction.cpu().numpy()
                                 * normalizers["yaw_scale"]
                                 + normalizers["yaw_center"])
                target_yaw = (targets * normalizers["yaw_scale"]
                              + normalizers["yaw_center"])
                batch_error = predicted_yaw - target_yaw
                for horizon in HORIZONS:
                    sample_error = batch_error[:, horizon - 1]
                    run_errors[horizon].append(sample_error)
                    errors[horizon].append(sample_error)
            per_run[run.run_id] = {
                str(horizon): _metrics(np.concatenate(run_errors[horizon]))
                for horizon in HORIZONS
            }
    pooled = {str(horizon): _metrics(np.concatenate(errors[horizon]))
              for horizon in HORIZONS}
    run_macro = {
        str(horizon): float(np.mean([
            per_run[run.run_id][str(horizon)]["rmse_radps"]
            for run in runs if str(horizon) in per_run[run.run_id]]))
        for horizon in HORIZONS
    }
    score = float(np.mean(list(run_macro.values())))
    return {
        "validation_stride": stride,
        "score_run_macro_rmse_radps": score,
        "run_macro_rmse_radps": run_macro,
        "pooled": pooled,
        "per_run": per_run,
    }


def _sample_training_batch(training: list[RunWindows], rng: np.random.Generator,
                           count: int
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    run_choices = rng.integers(0, len(training), size=count)
    starts = np.empty(count, dtype=np.int64)
    for run_index in np.unique(run_choices):
        rows = np.flatnonzero(run_choices == run_index)
        available = training[int(run_index)].starts
        starts[rows] = available[rng.integers(0, len(available), size=len(rows))]
    past_parts, future_parts, target_parts = [], [], []
    for run_index in np.unique(run_choices):
        rows = np.flatnonzero(run_choices == run_index)
        past, future, target = gather_windows(training[int(run_index)], starts[rows])
        past_parts.append((rows, past))
        future_parts.append((rows, future))
        target_parts.append((rows, target))
    past = np.empty((count, HISTORY_STEPS, len(OBSERVATION_NAMES)), np.float32)
    future = np.empty((count, FUTURE_STEPS, 2), np.float32)
    targets = np.empty((count, FUTURE_STEPS), np.float32)
    for parts, destination in ((past_parts, past), (future_parts, future),
                               (target_parts, targets)):
        for rows, values in parts:
            destination[rows] = values
    return past, future, targets


def train(output_dir: Path, *, max_epochs: int = MAX_EPOCHS,
          samples_per_epoch: int = SAMPLES_PER_EPOCH,
          workers: int = 16,
          excluded_validation_runs: tuple[str, ...] = ()) -> dict[str, Any]:
    import torch

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output: {output_dir}")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.set_num_threads(max(1, workers))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    series, source_audit = _discover_series_with_safe_mixed_archives()
    series = _exclude_validation_runs(series, set(excluded_validation_runs))
    minimum_sequence_length = HISTORY_STEPS + FUTURE_STEPS
    short_runs = sorted(
        row.run_id for row in series
        if not np.any((row.bounds[:, 1] - row.bounds[:, 0])
                      >= minimum_sequence_length))
    eligible_series = [
        row for row in series
        if np.any((row.bounds[:, 1] - row.bounds[:, 0])
                  >= minimum_sequence_length)]
    runs = [_make_run(row) for row in eligible_series]
    training = [run for run in runs if run.split == "train" and len(run.starts)]
    validation = [run for run in runs if run.split == "validation" and len(run.starts)]
    if len(training) < 2 or not validation:
        raise RuntimeError("whole-run train and validation captures are required")
    normalizers = _normalizers(training)
    model = make_model().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    rng = np.random.default_rng(SEED)
    steps_per_epoch = max(1, int(np.ceil(samples_per_epoch / BATCH_SIZE)))
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "yaw_gru_trajectory_teacher_report.json"
    checkpoint_path = output_dir / "yaw_gru_trajectory_teacher.pt"
    report: dict[str, Any] = {
        "title": "Causal GRU yaw-trajectory teacher",
        "status": "running",
        "torch_version": torch.__version__,
        "device": str(device),
        "sample_period_s": 0.025,
        "history_steps": HISTORY_STEPS,
        "history_duration_s": (HISTORY_STEPS - 1) * 0.025,
        "future_steps": FUTURE_STEPS,
        "future_duration_s": FUTURE_STEPS * 0.025,
        "horizons_steps": list(HORIZONS),
        "horizons_ms": [step * 25 for step in HORIZONS],
        "input_contract": {
            "history_features": list(OBSERVATION_NAMES),
            "future_commands": ["steering_command_rad",
                                 "throttle_command_norm"],
            "future_feedback_or_truth_used": False,
            "target": "simulator_rigid_state yaw_rate at k+1 through k+40",
            "oracle_odometry_features_used": False,
            "runtime_status": "offline research only; no odom/MPC integration",
        },
        "architecture": {
            "history_encoder": "2-layer GRU",
            "future_command_decoder": "1-layer GRU initialized from history state",
            "hidden_size": HIDDEN_SIZE,
            "dropout": DROPOUT,
            "output": "direct yaw-rate prediction at every future 25ms step",
            "loss": "trajectory MSE plus 0.1 times first-difference MSE",
        },
        "training": {
            "seed": SEED,
            "batch_size": BATCH_SIZE,
            "samples_per_epoch": samples_per_epoch,
            "steps_per_epoch": steps_per_epoch,
            "max_epochs": max_epochs,
            "validation_every_epochs": EVAL_EVERY,
            "patience_validation_evaluations": PATIENCE_EVALS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip_norm": GRADIENT_CLIP_NORM,
        },
        "source_audit": source_audit,
        "short_runs_excluded_for_history_and_horizon": short_runs,
        "training_runs": sorted(run.run_id for run in training),
        "validation_runs": sorted(run.run_id for run in validation),
        "excluded_validation_runs_from_checkpoint_selection": sorted(
            excluded_validation_runs),
        "training_window_count": int(sum(len(run.starts) for run in training)),
        "validation_window_count": int(sum(len(run.starts) for run in validation)),
        "validation_history": [],
        "best_epoch": None,
        "best_validation_score_run_macro_rmse_radps": None,
        "test_and_final_test_arrays_read": False,
    }
    _write_report(report_path, report)

    best_score = float("inf")
    stale_evaluations = 0
    best_state: dict[str, Any] | None = None
    for epoch in range(1, max_epochs + 1):
        model.train()
        loss_sum = 0.0
        for _ in range(steps_per_epoch):
            past, future, targets = _sample_training_batch(
                training, rng, BATCH_SIZE)
            past, future, targets = _normalize(
                past, future, targets, normalizers)
            past_tensor = torch.from_numpy(past).to(device)
            future_tensor = torch.from_numpy(future).to(device)
            target_tensor = torch.from_numpy(targets).to(device)
            prediction = model(past_tensor, future_tensor)
            trajectory_loss = torch.mean((prediction - target_tensor) ** 2)
            slope_loss = torch.mean((prediction[:, 1:] - prediction[:, :-1]
                                     - target_tensor[:, 1:] + target_tensor[:, :-1]) ** 2)
            loss = trajectory_loss + 0.1 * slope_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
            optimizer.step()
            loss_sum += float(loss.detach().cpu())

        epoch_report: dict[str, Any] = {
            "epoch": epoch,
            "training_loss": loss_sum / steps_per_epoch,
        }
        if epoch % EVAL_EVERY == 0 or epoch == max_epochs:
            validation_report = _evaluate(
                model, validation, normalizers, device,
                stride=VALIDATION_STRIDE)
            epoch_report["validation"] = validation_report
            report["validation_history"].append(epoch_report)
            score = validation_report["score_run_macro_rmse_radps"]
            if score < best_score:
                best_score = score
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                report["best_epoch"] = epoch
                report["best_validation_score_run_macro_rmse_radps"] = score
                stale_evaluations = 0
                torch.save({
                    "model_state_dict": best_state,
                    "normalizers": normalizers,
                    "architecture": report["architecture"],
                    "input_contract": report["input_contract"],
                    "torch_version": torch.__version__,
                    "seed": SEED,
                }, checkpoint_path)
            else:
                stale_evaluations += 1
            print(f"epoch={epoch:03d} loss={epoch_report['training_loss']:.5f} "
                  f"validation_macro_rmse={score:.5f} "
                  f"best={best_score:.5f} patience={stale_evaluations}/"
                  f"{PATIENCE_EVALS}", flush=True)
            _write_report(report_path, report)
            if stale_evaluations >= PATIENCE_EVALS:
                break
        elif epoch == 1:
            print(f"epoch=001 loss={epoch_report['training_loss']:.5f}",
                  flush=True)

    if best_state is None:
        raise RuntimeError("training did not produce a validation checkpoint")
    model.load_state_dict(best_state)
    report["best_validation_full_resolution"] = _evaluate(
        model, validation, normalizers, device, stride=FINAL_VALIDATION_STRIDE)
    report["status"] = "complete"
    report["limitations"] = [
        "This model is a one-shot direct yaw forecast, not a recursively closed full vehicle plant.",
        "Validation is whole-run but not the sealed test/final-test set.",
        "The model predicts yaw rate only; speed, actuator and wheel state remain unmodeled.",
        "The network is not yet integrated into odometry or MPC.",
    ]
    _write_report(report_path, report)
    print(f"report: {report_path}", flush=True)
    print(f"checkpoint: {checkpoint_path}", flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--samples-per-epoch", type=int,
                        default=SAMPLES_PER_EPOCH)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--exclude-validation-run", action="append", default=[],
        help="keep this whole validation run out of checkpoint selection; may be repeated")
    args = parser.parse_args()
    train(args.output_dir, max_epochs=args.epochs,
          samples_per_epoch=args.samples_per_epoch, workers=args.workers,
          excluded_validation_runs=tuple(args.exclude_validation_run))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
