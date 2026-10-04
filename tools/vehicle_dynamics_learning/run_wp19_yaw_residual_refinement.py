#!/usr/bin/env python3
"""Test a learned yaw-increment residual on the frozen WP19 plant parent.

The frozen WP19 GRU supplies the base 25 ms body and actuator transition. A
small residual head may correct only yaw-rate increment. This tests the
observed yaw-drift failure without replacing the full plant model. Research
only: no simulator, production odometry, localization, MPC, or actuator path
is modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from tools.vehicle_dynamics_learning.augmented_state_space_plant import (
    DT_S,
    HISTORY_STEPS,
    PoseIntegrator,
)
from tools.vehicle_dynamics_learning.run_wp19_target_ablation import (
    HIDDEN_SIZE,
    make_model as make_wp19_model,
    training_statistics,
)
from tools.vehicle_dynamics_learning.train_augmented_state_space_plant import (
    DEFAULT_DYNAMIC,
    DEFAULT_DYNAMIC_FIXED,
    DEFAULT_PRACTICE,
    DEFAULT_PRACTICE_FIXED,
    WP19_CHECKPOINT,
    _load_data,
    _predict_wp19_baseline,
    _sample_windows,
    _select_eval_windows,
    _training_windows_and_stats,
    sha256_file,
)
from tools.vehicle_dynamics_learning.run_reencoded_history_transition import (
    _bootstrap,
    _write_json,
)


ROOT = Path(__file__).resolve().parents[2]
TASK_ROOT = (ROOT / "live_runs/derived_dynamics_learning_20260928"
             / "full_modeling_reset_20261001"
             / "replacement_offline_sim_raceline_20261003/full_throttle_domain_v1")
OUTPUT_ROOT = TASK_ROOT / "next_phase_after_2129427/wp19_yaw_residual_v1"
FROZEN_EVAL_STARTS = (TASK_ROOT / "next_phase_after_2129427/frozen_eval_starts.json")
SOURCE_COMMIT = "2129427838101eee277e17e91727810bbe2df687"
EXPECTED_WP19_SHA256 = "045e98e1d0aafd9ec369bc6fdb1bdbce4e94491dec101a764f33195fe09e7541"
SEED = 20261003
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-5
BOOTSTRAP_REPLICATES = 5000
STAGES = ((20, 40, 8), (80, 60, 8), (200, 60, 4))
EVAL_HORIZONS = (1, 10, 30, 80, 200)


class YawResidual(nn.Module):
    def __init__(self, yaw_increment_scale: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(HIDDEN_SIZE + 7 + 3, 96), nn.SiLU(),
            nn.Linear(96, 64), nn.SiLU(),
            nn.Linear(64, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.register_buffer("yaw_increment_scale", torch.tensor(
            float(yaw_increment_scale), dtype=torch.float32))

    def forward(self, normalized_input: torch.Tensor, hidden: torch.Tensor,
                base_body_delta_normalized: torch.Tensor) -> torch.Tensor:
        features = torch.cat((normalized_input, hidden,
                              base_body_delta_normalized), dim=-1)
        return self.yaw_increment_scale * self.net(features).squeeze(-1)


def _git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                          check=True, capture_output=True, text=True).stdout.strip()


def _arrays(data, refs, horizon: int, device: torch.device):
    histories, bodies, actuators, poses, commands, targets, target_poses = (
        [] for _ in range(7))
    for ci, si, source_row in refs:
        capture = data.captures[ci]
        begin = int(capture.bounds[si, 0]) + int(source_row)
        if begin - HISTORY_STEPS + 1 < int(capture.bounds[si, 0]):
            raise ValueError("history crosses a sequence/reset boundary")
        if begin + horizon >= int(capture.bounds[si, 1]):
            raise ValueError("window lacks full target horizon")
        histories.append(capture.input_features[
            begin - HISTORY_STEPS + 1:begin + 1])
        bodies.append(capture.body[begin])
        actuators.append(capture.frames[begin, 3:5])
        poses.append(data.poses[ci][begin])
        commands.append(capture.frames[begin:begin + horizon, 7:9])
        targets.append(capture.body[begin + 1:begin + horizon + 1])
        target_poses.append(data.poses[ci][begin + 1:begin + horizon + 1])
    to_tensor = lambda value: torch.as_tensor(
        np.asarray(value), dtype=torch.float32, device=device)
    return tuple(map(to_tensor, (histories, bodies, actuators, poses, commands,
                                 targets, target_poses)))


def _initial_hidden(parent, history, input_mean, input_scale):
    hidden = torch.zeros(history.shape[0], HIDDEN_SIZE, device=history.device)
    for index in range(HISTORY_STEPS - 1):
        _, hidden = parent.step((history[:, index] - input_mean) / input_scale,
                                hidden)
    return hidden


def _rollout(parent, residual, arrays, stats, horizon: int,
             collect_loss: bool = False):
    history, body, actuator, pose, commands, truth_body, truth_pose = arrays
    input_mean, input_scale = stats["input"]
    body_delta_mean, body_delta_scale = stats["body"]["body_state_increment"]
    actuator_mean, actuator_scale = stats["actuator"]
    input_mean = torch.as_tensor(input_mean, dtype=torch.float32, device=body.device)
    input_scale = torch.as_tensor(input_scale, dtype=torch.float32, device=body.device)
    body_delta_mean = torch.as_tensor(body_delta_mean, dtype=torch.float32,
                                      device=body.device)
    body_delta_scale = torch.as_tensor(body_delta_scale, dtype=torch.float32,
                                       device=body.device)
    actuator_mean = torch.as_tensor(actuator_mean, dtype=torch.float32,
                                    device=body.device)
    actuator_scale = torch.as_tensor(actuator_scale, dtype=torch.float32,
                                     device=body.device)
    hidden = _initial_hidden(parent, history, input_mean, input_scale)
    body_state, current_pose = body, pose
    predicted_body, predicted_pose, losses = [], [], []
    integrator = PoseIntegrator(DT_S).to(body.device)
    for step in range(horizon):
        command = commands[:, step]
        physical = torch.cat((body_state, actuator, command), dim=-1)
        normalized_input = (physical - input_mean) / input_scale
        base_output, hidden = parent.step(normalized_input, hidden)
        base_delta = base_output[:, :3] * body_delta_scale + body_delta_mean
        yaw_correction = residual(normalized_input, hidden, base_output[:, :3])
        next_body = body_state + base_delta
        next_body = torch.cat((next_body[:, :2],
                               next_body[:, 2:3] + yaw_correction[:, None]), dim=-1)
        next_actuator = base_output[:, 3:5] * actuator_scale + actuator_mean
        # Match WP19's left-held pose integration so this isolates the learned
        # yaw transition correction rather than changing the pose integrator.
        next_pose = integrator(current_pose, body_state)
        if not torch.isfinite(next_body).all() or not torch.isfinite(next_pose).all():
            raise FloatingPointError(f"non-finite output at step {step + 1}")
        predicted_body.append(next_body)
        predicted_pose.append(next_pose)
        if collect_loss:
            # Normalize errors using WP22's training-only body statistics.
            scale = torch.as_tensor(stats["plant_state_scale"][:3],
                                    dtype=torch.float32, device=body.device)
            body_error = (next_body - truth_body[:, step]) / scale
            body_loss = F.smooth_l1_loss(body_error, torch.zeros_like(body_error))
            position_error = torch.linalg.vector_norm(
                next_pose[:, :2] - truth_pose[:, step, :2], dim=-1) / 0.5
            heading_error = next_pose[:, 2] - truth_pose[:, step, 2]
            heading_error = torch.atan2(torch.sin(heading_error),
                                        torch.cos(heading_error)) / 0.1
            position_loss = F.smooth_l1_loss(position_error,
                                              torch.zeros_like(position_error))
            heading_loss = F.smooth_l1_loss(heading_error,
                                             torch.zeros_like(heading_error))
            yaw_loss = F.smooth_l1_loss(
                (next_body[:, 2] - truth_body[:, step, 2]) / scale[2],
                torch.zeros_like(next_body[:, 2]))
            losses.append(body_loss + 0.25 * position_loss
                          + 0.5 * heading_loss + 0.5 * yaw_loss)
        body_state, actuator, current_pose = next_body, next_actuator, next_pose
    return {
        "body": torch.stack(predicted_body, dim=1),
        "pose": torch.stack(predicted_pose, dim=1),
        "loss": torch.stack(losses).mean() if losses else body_state.new_zeros(()),
    }


def _fit(data, parent, residual, stats, device):
    optimizer = torch.optim.AdamW(residual.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    counters = {name: Counter() for name in ("family", "run", "condition")}
    rng = np.random.default_rng(SEED)
    stages, sampler_draws, snapshots = [], [], []
    for horizon, updates, batch_size in STAGES:
        losses, gradients, draws = [], [], []
        residual.train()
        for _ in range(updates):
            refs = _sample_windows(data, horizon, batch_size, rng, counters)
            arrays = _arrays(data, refs, horizon, device)
            optimizer.zero_grad(set_to_none=True)
            result = _rollout(parent, residual, arrays, stats, horizon, True)
            loss = result["loss"]
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite yaw residual loss")
            loss.backward()
            gradient = float(torch.nn.utils.clip_grad_norm_(
                residual.parameters(), 5.0))
            if not np.isfinite(gradient):
                raise FloatingPointError("non-finite yaw residual gradient")
            optimizer.step()
            losses.append(float(loss.detach()))
            gradients.append(gradient)
            draws.append([[int(x) for x in ref] for ref in refs])
        stages.append({
            "horizon_steps": horizon, "updates": updates,
            "batch_size": batch_size,
            "mean_loss_last_10": float(np.mean(losses[-10:])),
            "mean_gradient_norm": float(np.mean(gradients)),
            "gradient_norm_p95": float(np.quantile(gradients, 0.95)),
            "sampler_draws": draws,
        })
        sampler_draws.extend(draws)
        snapshots.append({key: value.detach().cpu().clone()
                          for key, value in residual.state_dict().items()})
        print(f"completed {updates} yaw-residual updates at {horizon} steps; "
              f"last10_loss={stages[-1]['mean_loss_last_10']:.4f}", flush=True)
    return stages, sampler_draws, counters, snapshots


def _metrics(predicted_body, predicted_pose, truth_body, truth_pose, horizon):
    body = predicted_body[:, :horizon] - truth_body[:, :horizon]
    pose = predicted_pose[:, :horizon] - truth_pose[:, :horizon]
    heading = np.arctan2(np.sin(pose[..., 2]), np.cos(pose[..., 2]))
    radial = np.linalg.norm(pose[..., :2], axis=-1)
    return {
        "position_radial_trajectory_rmse_m": float(np.mean(np.sqrt(
            np.mean(radial ** 2, axis=1)))),
        "position_endpoint_error_m": float(np.mean(radial[:, -1])),
        "heading_trajectory_rmse_rad": float(np.mean(np.sqrt(
            np.mean(heading ** 2, axis=1)))),
        "u_rmse_mps": float(np.mean(np.sqrt(
            np.mean(body[..., 0] ** 2, axis=1)))),
        "v_rmse_mps": float(np.mean(np.sqrt(
            np.mean(body[..., 1] ** 2, axis=1)))),
        "yaw_rate_rmse_rps": float(np.mean(np.sqrt(
            np.mean(body[..., 2] ** 2, axis=1)))),
    }


def _predict(data, refs, parent, residual, stats, device, horizon):
    arrays = _arrays(data, refs, horizon, device)
    parent.eval()
    residual.eval()
    with torch.no_grad():
        result = _rollout(parent, residual, arrays, stats, horizon)
    truth_body, truth_pose = arrays[5].cpu().numpy(), arrays[6].cpu().numpy()
    return result["body"].cpu().numpy(), result["pose"].cpu().numpy(), \
        truth_body, truth_pose


def _score(data, parent, residual, stats, device, roles):
    result: dict[str, Any] = {}
    frozen = json.loads(FROZEN_EVAL_STARTS.read_text(encoding="utf-8"))
    for role, windows, horizon in roles:
        frozen_role = ("development_validation" if role == "validation"
                       else "practice_diagnostic" if role == "practice_diagnostic"
                       else None)
        selected = _select_eval_windows(windows, 64)
        if frozen_role is not None:
            expected_runs = frozen["split_roles"][frozen_role]
            if set(selected) != set(expected_runs):
                raise RuntimeError(f"{role}: runs differ from frozen WP24 evaluation")
            for run_id, refs in selected.items():
                actual = {(int(sequence), int(row)) for _, sequence, row in refs}
                expected = {(int(item["sequence_index"]), int(item["source_row"]))
                            for item in expected_runs[run_id]}
                if actual != expected:
                    raise RuntimeError(f"{run_id}: starts differ from frozen WP24 evaluation")
        result[role] = {}
        for run_id, refs in sorted(selected.items()):
            steps = min(horizon, min(
                int(data.captures[ci].bounds[si, 1]
                    - (int(data.captures[ci].bounds[si, 0]) + row)) - 1
                for ci, si, row in refs))
            candidate = _predict(data, refs, parent, residual, stats, device, steps)
            base = _predict_wp19_baseline(
                data.captures[refs[0][0]], data.poses[refs[0][0]], refs,
                steps, stats, parent, device)
            candidate_metrics, parent_metrics = {}, {}
            for h in (1, 10, 30, 80, 200):
                if h > steps:
                    continue
                candidate_metrics[str(h)] = _metrics(
                    candidate[0], candidate[1], candidate[2], candidate[3], h)
                base_state = base["body"]
                parent_metrics[str(h)] = _metrics(
                    base_state, base["pose"], base["truth_body"],
                    base["truth_pose"], h)
            result[role][run_id] = {
                "start_count": len(refs), "horizon_steps": steps,
                "yaw_residual": candidate_metrics,
                "wp19_parent": parent_metrics,
            }
    return result


def _stage_validation_score(data, parent, residual, stats, device):
    report = _score(data, parent, residual, stats, device,
                    [("validation", data.validation_windows, 200)])
    per_run = report["validation"]
    position = np.mean([row["yaw_residual"]["200"][
        "position_radial_trajectory_rmse_m"] for row in per_run.values()])
    yaw = np.mean([row["yaw_residual"]["200"][
        "yaw_rate_rmse_rps"] for row in per_run.values()])
    parent_position = np.mean([row["wp19_parent"]["200"][
        "position_radial_trajectory_rmse_m"] for row in per_run.values()])
    parent_yaw = np.mean([row["wp19_parent"]["200"][
        "yaw_rate_rmse_rps"] for row in per_run.values()])
    return float(position / parent_position + yaw / parent_yaw), report


def run(device_name="cpu", output_root: Path = OUTPUT_ROOT):
    if _git_head() != SOURCE_COMMIT:
        raise RuntimeError(f"expected source {SOURCE_COMMIT}; found {_git_head()}")
    if sha256_file(WP19_CHECKPOINT) != EXPECTED_WP19_SHA256:
        raise RuntimeError("frozen WP19 parent hash mismatch")
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    data, wp20 = _load_data()
    config, _, _ = _training_windows_and_stats(data, wp20)
    stats = training_statistics(data.captures, data.training_windows_80)
    stats["plant_state_scale"] = np.asarray(config.state_scale, dtype=np.float32)
    parent_checkpoint = torch.load(WP19_CHECKPOINT, map_location=device,
                                   weights_only=True)
    parent = make_wp19_model(torch.nn, hidden_size=HIDDEN_SIZE).to(device)
    parent.load_state_dict(parent_checkpoint["state_dict"], strict=True)
    parent.eval()
    for parameter in parent.parameters():
        parameter.requires_grad_(False)
    yaw_scale = float(stats["body"]["body_state_increment"][1][2])
    residual = YawResidual(yaw_scale).to(device)
    stages, sampler_draws, counters, snapshots = _fit(
        data, parent, residual, stats, device)

    # Select only on frozen held-out OpenPlane validation starts; practice runs
    # are opened only after selection, for a transfer diagnostic.
    stage_scores = []
    candidates = []
    for stage_index, (stage, snapshot) in enumerate(zip(stages, snapshots)):
        residual.load_state_dict(snapshot, strict=True)
        score, report = _stage_validation_score(
            data, parent, residual, stats, device)
        candidates.append(snapshot)
        stage_scores.append({
            "after_stage": stage_index + 1,
            "optimizer_updates": sum(item["updates"] for item in stages[:stage_index + 1]),
            "validation_5s_composite_ratio": score,
            "validation_report": report,
        })
    best_index = int(np.argmin([row["validation_5s_composite_ratio"]
                                for row in stage_scores]))
    residual.load_state_dict(candidates[best_index], strict=True)
    evaluation = _score(data, parent, residual, stats, device, [
        ("development_validation", data.validation_windows, 200),
        ("practice_diagnostic", data.practice_windows, 200),
    ])
    output_root.mkdir(parents=True, exist_ok=False)
    sampler_hash = hashlib.sha256(json.dumps(
        sampler_draws, separators=(",", ":")).encode()).hexdigest()
    state_dict = {key: value.detach().cpu() for key, value in residual.state_dict().items()}
    checkpoint = output_root / "yaw_residual.pt"
    torch.save({"state_dict": state_dict, "metadata": {
        "source_commit": SOURCE_COMMIT, "seed": SEED,
        "wp19_parent_sha256": EXPECTED_WP19_SHA256,
        "only_corrected_output": "yaw_rate_increment",
        "history_semantics": "frozen WP19 GRU; 79 causal measured rows initialize, then own predicted state and current command",
        "training_horizons_steps": [stage[0] for stage in STAGES],
        "selected_after_stage": best_index + 1,
    }}, checkpoint)
    checkpoint_hash = sha256_file(checkpoint)
    (output_root / "checkpoint.sha256").write_text(checkpoint_hash + "\n",
                                                   encoding="utf-8")
    from tools.vehicle_dynamics_learning.train_augmented_state_space_plant import sha256_file as hash_data
    manifest = {
        "source_commit": SOURCE_COMMIT,
        "parent_checkpoint_sha256": EXPECTED_WP19_SHA256,
        "training_runs": list(data.training_runs),
        "validation_runs": sorted(data.validation_windows),
        "practice_diagnostic_runs": sorted(data.practice_windows),
        "sampler_draws_sha256": sampler_hash,
        "datasets": {str(path.relative_to(ROOT)): hash_data(path)
                     for path in (DEFAULT_DYNAMIC, DEFAULT_DYNAMIC_FIXED,
                                  DEFAULT_PRACTICE, DEFAULT_PRACTICE_FIXED)},
        "split_policy": "train only for fitting; OpenPlane validation only for checkpoint selection; practice diagnostic after selection",
    }
    per_run_deltas = {}
    for role, runs in evaluation.items():
        per_run_deltas[role] = {}
        for horizon in ("10", "30", "80", "200"):
            per_run_deltas[role][horizon] = {}
            for metric in ("position_radial_trajectory_rmse_m", "position_endpoint_error_m",
                           "heading_trajectory_rmse_rad", "u_rmse_mps",
                           "v_rmse_mps", "yaw_rate_rmse_rps"):
                deltas = {
                    run_id: row["wp19_parent"][horizon][metric]
                    - row["yaw_residual"][horizon][metric]
                    for run_id, row in runs.items()
                    if horizon in row["yaw_residual"] and horizon in row["wp19_parent"]}
                per_run_deltas[role][horizon][metric] = {
                    "direction": "WP19 parent minus yaw-residual candidate; positive favors candidate",
                    "run_macro_delta": float(np.mean(list(deltas.values()))),
                    "run_cluster_bootstrap_95pct_ci": _bootstrap(
                        deltas, SEED + int(horizon) + len(metric)),
                    "independent_run_count": len(deltas),
                    "per_run_delta": deltas,
                }
    report = {
        "study": "WP19 frozen transition plus learned yaw-rate increment residual",
        "source_commit": SOURCE_COMMIT,
        "simulator_launched": False,
        "production_or_mpc_integration": False,
        "training": {
            "stages": [{key: value for key, value in stage.items()
                        if key != "sampler_draws"} for stage in stages],
            "validation_checkpoint_selection": stage_scores,
            "selected_stage_index_1based": best_index + 1,
            "sampler_counts": {key: dict(counter)
                               for key, counter in counters.items()},
        },
        "evaluation": {"per_run": evaluation,
                       "paired_wp19_parent_minus_candidate": per_run_deltas,
                       "practice_used_for_selection": False,
                       "frozen_start_manifest": FROZEN_EVAL_STARTS.relative_to(ROOT).as_posix(),
                       "metrics": "mean per-start RMSE within run, then macro mean across independent runs"},
        "checkpoint_sha256": checkpoint_hash,
        "training_sampler_draws_sha256": sampler_hash,
        "dataset_manifest": manifest,
        "limits": [
            "This is a one-channel yaw residual experiment, not a complete vehicle plant.",
            "Candidate selection uses only six whole OpenPlane validation runs; practice consists of two independent diagnostic runs.",
            "No test/final-test runs, future truth, or future sensor feedback enter recursive rollouts.",
            "A selected residual is not accepted for offline simulation unless whole-run errors materially improve.",
        ],
    }
    _write_json(output_root / "yaw_residual_report.json", report)
    _write_json(output_root / "dataset_manifest.json", manifest)
    _write_json(output_root / "training_sampler_draws.json", {
        "sha256": sampler_hash, "draw_batches": sampler_draws})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    output = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    result = run(args.device, output)
    print(json.dumps({
        "output": str(output),
        "selected_stage": result["training"]["selected_stage_index_1based"],
        "validation_5s_position_gain": result["evaluation"][
            "paired_wp19_parent_minus_candidate"]["development_validation"][
                "200"]["position_radial_trajectory_rmse_m"],
        "validation_5s_yaw_gain": result["evaluation"][
            "paired_wp19_parent_minus_candidate"]["development_validation"][
                "200"]["yaw_rate_rmse_rps"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
