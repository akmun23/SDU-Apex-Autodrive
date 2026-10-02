#!/usr/bin/env python3
"""Train the handoff's nominal + history-state + residual offline plant."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

from tools.vehicle_dynamics_learning.experiment_artifacts import (
    write_standard_artifacts,
)
from tools.vehicle_dynamics_learning.family_condition_sampler import (
    build_sequence_sampler,
)
from tools.vehicle_dynamics_learning.four_wheel_greybox import (
    DT_S,
    WHEEL_RADIUS_M,
    _groups,
    _initial_state,
    _physical_model,
    _targets,
)
from tools.vehicle_dynamics_learning.hybrid_race_teacher import (
    HISTORY_FEATURE_SIZE,
    HISTORY_STEPS,
    LATENT_SIZE,
    RESIDUAL_LIMITS,
    RESIDUAL_NAMES,
    hybrid_model_type,
    integrate_pose,
    local_pose_targets,
)
from tools.vehicle_dynamics_learning.race_domain_objectives import (
    race_speed_mismatch_weights,
)
from tools.vehicle_dynamics_learning.train_nssm import _load_dataset, _torch


ROLLOUT_STEPS = 80
STATE_SCALE = np.asarray((0.5, 0.25, 0.25, 0.5, 0.5, 0.1, 0.2),
                         dtype=np.float32)
ACCELERATION_NAMES = (
    "ax_body_mps2", "ay_body_mps2", "yaw_accel_rps2",
    "rear_left_surface_accel_mps2", "rear_right_surface_accel_mps2",
)
POSE_HORIZONS_STEPS = (10, 30, 80)
VALIDATION_HORIZONS_STEPS = (10, 30, 80)
VALIDATION_POSITION_SCALES_M = (0.15, 0.5, 2.0)
VALIDATION_SPEED_SCALES_MPS = (0.2, 0.5, 1.0)
VALIDATION_HEADING_SCALES_RAD = (0.05, 0.15, 0.3)


def _history_features(data: dict[str, Any]) -> np.ndarray:
    if data.get("sensor_frames") is None or data.get("sensor_valid") is None:
        raise ValueError("hybrid teacher requires aligned legal sensor history")
    if data["frames"].shape[1] != 9 or data["sensor_frames"].shape[1] != 10:
        raise ValueError("unexpected offline-plant history feature layout")
    return np.concatenate((data["frames"], data["sensor_frames"]), axis=1)


def _wheel_accelerations(data: dict[str, Any]) -> np.ndarray:
    result = np.full((len(data["frames"]), 2), np.nan, dtype=np.float32)
    for start_raw, end_raw in data["bounds"]:
        start, end = int(start_raw), int(end_raw)
        result[start:end - 1] = np.diff(
            data["frames"][start:end, 5:7], axis=0) / DT_S
    return result


def _training_statistics(data: dict[str, Any], state: np.ndarray,
                         acceleration: np.ndarray, wheel_acceleration: np.ndarray,
                         history: np.ndarray,
                         groups: dict[int, list[tuple[int, int]]]
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                    tuple[float, float]]:
    indexes = np.concatenate([
        np.arange(start + HISTORY_STEPS - 1, end - ROLLOUT_STEPS)
        for sequences in groups.values()
        for start, end in sequences
    ]).astype(np.int64)
    if not len(indexes):
        raise ValueError("no training rows support 2 s history and rollout")
    features = history[indexes].astype(np.float64)
    history_mean = features.mean(axis=0).astype(np.float32)
    history_scale = np.maximum(features.std(axis=0), 1.0e-3).astype(np.float32)
    transition = np.column_stack((acceleration[indexes],
                                  wheel_acceleration[indexes])).astype(np.float64)
    transition = transition[np.isfinite(transition).all(axis=1)]
    acceleration_scale = np.maximum(
        np.std(transition, axis=0), (0.5, 0.5, 0.5, 2.0, 2.0)
    ).astype(np.float32)
    mismatch = np.abs(np.mean(data["frames"][indexes, 5:7], axis=1)
                      - state[indexes, 0])
    thresholds = tuple(np.quantile(mismatch, (1.0 / 3.0, 2.0 / 3.0)))
    return history_mean, history_scale, acceleration_scale, thresholds


def _sample_batch(data, state, acceleration, wheel_acceleration, history,
                  sampler, batch_size, mismatch_thresholds, rng):
    initials, histories, commands, targets, accel_targets, pose_targets = (
        [], [], [], [], [], [])
    speeds, mismatches = [], []
    for _ in range(batch_size):
        run, (sequence_start, end), _, _ = sampler.sample(rng)
        low = sequence_start + HISTORY_STEPS - 1
        high = end - ROLLOUT_STEPS - 1
        if high < low:
            raise ValueError("family sampler selected a too-short sequence")
        current = int(rng.integers(low, high + 1))
        context_start = current - HISTORY_STEPS + 1
        history_window = history[context_start:current + 1]
        if not np.isfinite(history_window).all() or not data["sensor_valid"][
                context_start:current + 1].all():
            raise ValueError("sampled history contains invalid legal sensors")
        initials.append(_initial_state(
            data, current, state, sequence_start))
        histories.append(history_window)
        commands.append(data["frames"][current:current + ROLLOUT_STEPS, 7:9])
        targets.append(np.column_stack((
            state[current + 1:current + ROLLOUT_STEPS + 1],
            data["frames"][current + 1:current + ROLLOUT_STEPS + 1, 3:5],
        )))
        accel_targets.append(np.column_stack((
            acceleration[current:current + ROLLOUT_STEPS],
            wheel_acceleration[current:current + ROLLOUT_STEPS],
        )))
        pose_targets.append(local_pose_targets(
            data["simulator_pose_xyyaw"][current:current + ROLLOUT_STEPS + 1]))
        speeds.append(float(np.linalg.norm(state[current, :2])))
        mismatches.append(float(abs(np.mean(data["frames"][current, 5:7])
                                    - state[current, 0])))

    weights = race_speed_mismatch_weights(
        np.asarray(speeds), np.asarray(mismatches), mismatch_thresholds)
    arrays = [np.stack(rows).astype(np.float32) for rows in (
        initials, histories, commands, targets, accel_targets, pose_targets)]
    return (*arrays, weights)


def _loss_functions(torch, nn):
    def weighted(prediction, target, scale, weights):
        normalized = (prediction - target) / scale
        per_time = nn.functional.smooth_l1_loss(
            normalized, torch.zeros_like(normalized), beta=0.5,
            reduction="none").mean(dim=-1)
        per_sample = per_time.mean(dim=-1)
        return torch.sum(per_sample * weights) / torch.clamp(
            torch.sum(weights), min=1.0e-12)

    def pose_loss(prediction, target_pose, weights):
        terms = []
        for step in POSE_HORIZONS_STEPS:
            position_error = ((prediction[:, step - 1, :2]
                               - target_pose[:, step - 1, :2]) / 2.0)
            heading_error = torch.atan2(
                torch.sin(prediction[:, step - 1, 2]
                          - target_pose[:, step - 1, 2]),
                torch.cos(prediction[:, step - 1, 2]
                          - target_pose[:, step - 1, 2]),
            ) / 0.5
            per_sample = 0.5 * (
                nn.functional.smooth_l1_loss(
                    position_error, torch.zeros_like(position_error),
                    beta=1.0, reduction="none").mean(dim=-1)
                + nn.functional.smooth_l1_loss(
                    heading_error, torch.zeros_like(heading_error), beta=1.0,
                    reduction="none")
            )
            terms.append(torch.sum(per_sample * weights) / torch.clamp(
                torch.sum(weights), min=1.0e-12))
        return torch.stack(terms).mean()

    return weighted, pose_loss


def _evaluate(torch, model, data, state, groups, history,
              device, seed,
              max_windows_per_run: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    per_run = {}
    model.eval()
    with torch.no_grad():
        for run, sequences in sorted(groups.items()):
            rows = []
            candidates = []
            for sequence_start, end in sequences:
                low = sequence_start + HISTORY_STEPS - 1
                high = end - ROLLOUT_STEPS - 1
                if high >= low:
                    candidates.append((sequence_start, end, low, high))
            if not candidates:
                continue
            for _ in range(max_windows_per_run):
                sequence_start, end, low, high = candidates[
                    int(rng.integers(0, len(candidates)))]
                current = int(rng.integers(low, high + 1))
                context_start = current - HISTORY_STEPS + 1
                history_np = history[context_start:current + 1]
                initial_np = _initial_state(
                    data, current, state, sequence_start)
                commands_np = data["frames"][
                    current:current + ROLLOUT_STEPS, 7:9]
                initial = torch.as_tensor(initial_np[None], dtype=torch.float32,
                                           device=device)
                history_t = torch.as_tensor(history_np[None], dtype=torch.float32,
                                            device=device)
                commands = torch.as_tensor(commands_np[None], dtype=torch.float32,
                                           device=device)
                predicted, _, _, _ = model.rollout(initial, history_t, commands)
                predicted_output = torch.cat((
                    predicted[:, :, :3],
                    predicted[:, :, 5:7] * WHEEL_RADIUS_M,
                    predicted[:, :, 17:19],
                ), dim=-1)[0].cpu().numpy()
                truth_output = np.column_stack((
                    state[current + 1:current + ROLLOUT_STEPS + 1],
                    data["frames"][current + 1:current + ROLLOUT_STEPS + 1, 3:5],
                ))
                pose_local = integrate_pose(torch, predicted, initial)[0].cpu().numpy()
                pose0 = data["simulator_pose_xyyaw"][current]
                cosine, sine = math.cos(float(pose0[2])), math.sin(float(pose0[2]))
                pose_pred = np.empty_like(pose_local)
                pose_pred[:, 0] = pose0[0] + cosine * pose_local[:, 0] - sine * pose_local[:, 1]
                pose_pred[:, 1] = pose0[1] + sine * pose_local[:, 0] + cosine * pose_local[:, 1]
                pose_pred[:, 2] = pose0[2] + pose_local[:, 2]
                pose_truth = data["simulator_pose_xyyaw"][
                    current + 1:current + ROLLOUT_STEPS + 1]
                position_error = pose_pred[:, :2] - pose_truth[:, :2]
                heading_error = np.arctan2(
                    np.sin(pose_pred[:, 2] - pose_truth[:, 2]),
                    np.cos(pose_pred[:, 2] - pose_truth[:, 2]))
                speed_error = (np.linalg.norm(predicted_output[:, :2], axis=1)
                               - np.linalg.norm(truth_output[:, :2], axis=1))
                horizon_scores = {}
                for step in VALIDATION_HORIZONS_STEPS:
                    horizon_scores[f"{step * DT_S:g}s"] = {
                        "position_error_m": float(np.linalg.norm(
                            position_error[step - 1])),
                        "speed_abs_error_mps": float(abs(speed_error[step - 1])),
                        "heading_abs_error_rad": float(abs(heading_error[step - 1])),
                    }
                rows.append({
                    "sequence_start_index": sequence_start,
                    "initial_speed_mps": float(np.linalg.norm(state[current, :2])),
                    "initial_wheel_body_mismatch_proxy_mps": float(abs(
                        np.mean(data["frames"][current, 5:7]) - state[current, 0])),
                    "position_radial_rmse_m": float(np.sqrt(np.mean(
                        np.sum(position_error ** 2, axis=1)))),
                    "position_endpoint_error_m": float(np.linalg.norm(
                        position_error[-1])),
                    "heading_rmse_rad": float(np.sqrt(np.mean(heading_error ** 2))),
                    "speed_rmse_mps": float(np.sqrt(np.mean(speed_error ** 2))),
                    "horizons": horizon_scores,
                })
            per_run[str(data["run_ids"][run])] = {
                "window_count": len(rows),
                "macro_position_radial_rmse_m": float(np.mean(
                    [row["position_radial_rmse_m"] for row in rows])),
                "macro_speed_rmse_mps": float(np.mean(
                    [row["speed_rmse_mps"] for row in rows])),
                "macro_heading_rmse_rad": float(np.mean(
                    [row["heading_rmse_rad"] for row in rows])),
                "windows": rows,
            }
    run_score = []
    for row in per_run.values():
        horizon_terms = []
        for step, position_scale, speed_scale, heading_scale in zip(
                VALIDATION_HORIZONS_STEPS,
                VALIDATION_POSITION_SCALES_M,
                VALIDATION_SPEED_SCALES_MPS,
                VALIDATION_HEADING_SCALES_RAD):
            key = f"{step * DT_S:g}s"
            horizon_terms.append(np.mean([
                item["horizons"][key]["position_error_m"] / position_scale
                + item["horizons"][key]["speed_abs_error_mps"] / speed_scale
                + item["horizons"][key]["heading_abs_error_rad"] / heading_scale
                for item in row["windows"]
            ]))
        run_score.append(float(np.mean(horizon_terms)))
    return {
        "independent_run_count": len(per_run),
        "macro_run_selection_score": float(np.mean(run_score)),
        "per_run": per_run,
    }


def train(dataset_path: Path, nominal_checkpoint_path: Path,
          output_dir: Path, *, max_steps: int = 1800,
          eval_every: int = 100, patience: int = 8, batch_size: int = 8,
          max_eval_windows_per_run: int = 8, latent_size: int = LATENT_SIZE,
          seed: int = 20261002, score_test: bool = False) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if max_steps < 1 or eval_every < 1 or patience < 1 or batch_size < 1:
        raise ValueError("training step, evaluation, patience, and batch values must be positive")
    torch, nn = _torch()
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = torch.device("cpu")

    data = _load_dataset(dataset_path)
    if data["schema_version"] < 8 or data.get("sensor_frames") is None:
        raise ValueError("hybrid training requires audited schema-8 sensor data")
    state, acceleration = _targets(data)
    wheel_acceleration = _wheel_accelerations(data)
    history = _history_features(data)
    horizon_steps = ROLLOUT_STEPS
    groups = {
        split: _groups(data, state, acceleration, split,
                       HISTORY_STEPS + horizon_steps)
        for split in ("train", "validation", "test", "final_test")
    }
    if not groups["train"] or not groups["validation"]:
        raise ValueError("need whole-run train and validation captures")
    if not all(np.isfinite(wheel_acceleration[start:end - 1]).all()
               for rows in groups["train"].values() for start, end in rows):
        raise ValueError("training sequences contain invalid rear wheel acceleration")

    history_mean, history_scale, acceleration_scale, mismatch_thresholds = _training_statistics(
        data, state, acceleration, wheel_acceleration, history, groups["train"])
    sampler = build_sequence_sampler(data, groups["train"])
    nominal_payload = torch.load(nominal_checkpoint_path, map_location="cpu",
                                 weights_only=False)
    nominal_metadata = nominal_payload["metadata"]
    if nominal_metadata["feature_names"] != data["feature_names"]:
        raise ValueError("nominal checkpoint and training capture features differ")
    nominal_type = _physical_model(
        torch, nn, bool(nominal_metadata["tire_relaxation_state_enabled"]))
    nominal = nominal_type()
    nominal.load_state_dict(nominal_payload["state_dict"], strict=True)
    nominal.eval()
    for parameter in nominal.parameters():
        parameter.requires_grad_(False)
    model_type = hybrid_model_type(
        torch, nn, nominal, history_mean, history_scale, latent_size)
    model = model_type().to(device)
    trainable_parameters = [parameter for parameter in model.parameters()
                            if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_parameters, lr=1.0e-3, weight_decay=1.0e-5)
    state_scale_t = torch.as_tensor(STATE_SCALE, dtype=torch.float32,
                                   device=device)
    acceleration_scale_t = torch.as_tensor(
        acceleration_scale, dtype=torch.float32, device=device)
    weighted_state_loss, pose_loss = _loss_functions(torch, nn)
    rng = np.random.default_rng(seed)
    best_score, best_step, stale = math.inf, 0, 0
    history_rows = []
    start_time = time.perf_counter()

    for step in range(1, max_steps + 1):
        batch = _sample_batch(
            data, state, acceleration, wheel_acceleration, history,
            sampler, batch_size, mismatch_thresholds, rng)
        initial, context, commands, target, accel_target, pose_target, weights = batch
        initial_t = torch.as_tensor(initial, dtype=torch.float32, device=device)
        context_t = torch.as_tensor(context, dtype=torch.float32, device=device)
        commands_t = torch.as_tensor(commands, dtype=torch.float32, device=device)
        target_t = torch.as_tensor(target, dtype=torch.float32, device=device)
        accel_target_t = torch.as_tensor(
            accel_target, dtype=torch.float32, device=device)
        pose_target_t = torch.as_tensor(pose_target, dtype=torch.float32,
                                        device=device)
        weights_t = torch.as_tensor(weights, dtype=torch.float32, device=device)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        predicted, predicted_acceleration, predicted_wheel_accel, residual = (
            model.rollout(initial_t, context_t, commands_t,
                          detach_every_steps=20))
        output = torch.cat((
            predicted[:, :, :3],
            predicted[:, :, 5:7] * WHEEL_RADIUS_M,
            predicted[:, :, 17:19],
        ), dim=-1)
        state_losses = []
        for length in POSE_HORIZONS_STEPS:
            state_losses.append(weighted_state_loss(
                output[:, :length], target_t[:, :length],
                state_scale_t, weights_t))
        loss_state = 0.25 * state_losses[0] + 0.30 * state_losses[1] + 0.45 * state_losses[2]
        acceleration_prediction = torch.cat((
            predicted_acceleration, predicted_wheel_accel), dim=-1)
        loss_acceleration = weighted_state_loss(
            acceleration_prediction, accel_target_t,
            acceleration_scale_t, weights_t)
        predicted_pose = integrate_pose(
            torch, predicted, initial_t, detach_every_steps=20)
        loss_pose = pose_loss(predicted_pose, pose_target_t, weights_t)
        loss_residual = torch.mean((residual / model.residual_limits) ** 2)
        loss = loss_state + 0.10 * loss_acceleration + 0.15 * loss_pose + 0.001 * loss_residual
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite hybrid loss at step {step}")
        loss.backward()
        if any(parameter.grad is not None
               and not torch.isfinite(parameter.grad).all()
               for parameter in model.parameters()):
            raise FloatingPointError(f"non-finite hybrid gradient at step {step}")
        torch.nn.utils.clip_grad_norm_(
            trainable_parameters, 2.0)
        optimizer.step()

        if step % eval_every == 0 or step == max_steps:
            validation = _evaluate(
                torch, model, data, state, groups["validation"], history,
                device, seed + 2001, max_eval_windows_per_run)
            score = validation["macro_run_selection_score"]
            history_rows.append({
                "step": step,
                "training_loss": float(loss.detach()),
                "state_loss": float(loss_state.detach()),
                "acceleration_loss": float(loss_acceleration.detach()),
                "pose_loss": float(loss_pose.detach()),
                "validation_score": score,
                "validation": validation,
            })
            print(json.dumps({"step": step, "loss": float(loss.detach()),
                              "validation_score": score}, sort_keys=True),
                  flush=True)
            if score < best_score:
                best_score, best_step, stale = score, step, 0
                torch.save({
                    "state_dict": model.state_dict(),
                    "metadata": {
                        "architecture": "four_wheel_nominal_history_residual",
                        "feature_names": data["feature_names"],
                        "training_runs": [str(data["run_ids"][run])
                                          for run in sorted(groups["train"])],
                        "history_steps": HISTORY_STEPS,
                        "history_feature_size": HISTORY_FEATURE_SIZE,
                        "latent_size": latent_size,
                        "nominal_checkpoint": str(nominal_checkpoint_path.resolve()),
                        "nominal_checkpoint_sha256": _sha256(nominal_checkpoint_path),
                        "residual_names": list(RESIDUAL_NAMES),
                        "future_truth_or_sensor_feedback": False,
                    },
                }, output_dir / "best_hybrid_race_teacher.pt")
            else:
                stale += 1
            (output_dir / "validation_history.json").write_text(
                json.dumps(history_rows, indent=2, sort_keys=True) + "\n",
                encoding="utf-8")
            if stale >= patience:
                break

    best = torch.load(output_dir / "best_hybrid_race_teacher.pt",
                      map_location=device, weights_only=False)
    model.load_state_dict(best["state_dict"], strict=True)
    validation = _evaluate(
        torch, model, data, state, groups["validation"], history,
        device,
        seed + 2001, max_eval_windows_per_run)
    test = None
    if score_test and groups["test"]:
        test = _evaluate(
            torch, model, data, state, groups["test"], history,
            device,
            seed + 2002, max_eval_windows_per_run)
    report = {
        "schema_version": 1,
        "architecture": "four_wheel_nominal_history_residual",
        "dataset": str(dataset_path.resolve()),
        "nominal_checkpoint": str(nominal_checkpoint_path.resolve()),
        "sample_period_s": DT_S,
        "history_seconds": (HISTORY_STEPS - 1) * DT_S,
        "rollout_seconds": ROLLOUT_STEPS * DT_S,
        "latent_size": latent_size,
        "residual_names": list(RESIDUAL_NAMES),
        "residual_limits": list(RESIDUAL_LIMITS),
        "optimizer_steps": step,
        "optimizer_steps_per_second": step / max(
            time.perf_counter() - start_time, 1.0e-9),
        "training_sampler": sampler.metadata,
        "mismatch_proxy_thresholds_mps": list(mismatch_thresholds),
        "best_step": best_step,
        "best_validation_score": best_score,
        "eligible_runs": {
            split: [str(data["run_ids"][run]) for run in sorted(rows)]
            for split, rows in groups.items()
        },
        "validation": validation,
        "test_scored_once": bool(score_test),
        "test": test,
        "checkpoint": str(output_dir / "best_hybrid_race_teacher.pt"),
        "limitations": [
            "simulator truth initializes training/scoring body state and supplies labels only",
            "nominal tire and front-wheel states remain effective/unobserved",
            "validation windows within a capture are correlated; independent runs are uncertainty units",
        ],
        "history": history_rows,
    }
    write_standard_artifacts(
        output_dir, dataset_path, data,
        {"architecture": report["architecture"],
         "nominal_checkpoint": str(nominal_checkpoint_path.resolve()),
         "history_steps": HISTORY_STEPS,
         "rollout_steps": ROLLOUT_STEPS,
         "latent_size": latent_size,
         "max_steps": max_steps,
         "eval_every": eval_every,
         "patience": patience,
         "batch_size": batch_size,
         "max_eval_windows_per_run": max_eval_windows_per_run,
         "mismatch_proxy_thresholds_mps": mismatch_thresholds,
         "score_test": score_test}, seed, report,
        ("tools/vehicle_dynamics_learning/hybrid_race_teacher.py",
         "tools/vehicle_dynamics_learning/train_hybrid_race_teacher.py",
         "tools/vehicle_dynamics_learning/four_wheel_greybox.py",
         "tools/vehicle_dynamics_learning/train_nssm.py"))
    return report


def _sha256(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("nominal_checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=1800)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-eval-windows-per-run", type=int, default=8)
    parser.add_argument("--latent-size", type=int, choices=(16, 32, 64),
                        default=LATENT_SIZE)
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument("--score-test", action="store_true")
    args = parser.parse_args()
    report = train(
        args.dataset, args.nominal_checkpoint, args.output_dir,
        max_steps=args.max_steps, eval_every=args.eval_every,
        patience=args.patience, batch_size=args.batch_size,
        max_eval_windows_per_run=args.max_eval_windows_per_run,
        latent_size=args.latent_size, seed=args.seed,
        score_test=args.score_test)
    print(json.dumps({"output_dir": str(args.output_dir),
                      "best_step": report["best_step"],
                      "best_validation_score": report["best_validation_score"],
                      "validation": report["validation"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
