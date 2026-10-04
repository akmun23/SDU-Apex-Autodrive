#!/usr/bin/env python3
"""Attribute frozen WP22 loss gradients on training windows; never optimizes."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from tools.vehicle_dynamics_learning.augmented_state_space_plant import (
    DT_S,
    load_checkpoint,
)
from tools.vehicle_dynamics_learning.trace_wp22_internal_rollout import (
    NEXT_ROOT,
    SOURCE_COMMIT,
    WP22_OUTPUT,
    WP25_ROOT,
    _write_json,
)
from tools.vehicle_dynamics_learning.train_augmented_state_space_plant import (
    ACCELERATION_CONSISTENCY_WEIGHT,
    BODY_LOSS_WEIGHT,
    HEADING_LOSS_WEIGHT,
    LATENT_LOSS_WEIGHT,
    MEASUREMENT_LOSS_WEIGHT,
    POSITION_LOSS_WEIGHT,
    POSITION_SCALE_M,
    HEADING_SCALE_RAD,
    ROOT,
    SUPPORT_LOSS_WEIGHT,
    _batch_arrays,
    _implied_acceleration,
    _load_data,
    _training_windows_and_stats,
    sha256_file,
)


CHECKPOINT = WP22_OUTPUT / "seed101_candidate.pt"
OUTPUT = WP25_ROOT / "wp25_loss_gradient_attribution.json"
HORIZONS = (10, 20, 40, 80, 200)
WEIGHTS = {
    "body": BODY_LOSS_WEIGHT,
    "acceleration_consistency": ACCELERATION_CONSISTENCY_WEIGHT,
    "heading": HEADING_LOSS_WEIGHT,
    "position": POSITION_LOSS_WEIGHT,
    "measurement": MEASUREMENT_LOSS_WEIGHT,
    "latent": LATENT_LOSS_WEIGHT,
    "support": SUPPORT_LOSS_WEIGHT,
}


def _wrap(value: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(value), torch.cos(value))


def _fixed_training_starts(data) -> list[tuple[int, int, int]]:
    refs = []
    for run_id, candidates in sorted(data.train_windows_by_horizon[200].items()):
        if run_id not in data.training_runs or not candidates:
            continue
        refs.append(candidates[len(candidates) // 2])
    if len(refs) < 5:
        raise RuntimeError("WP25 gradient probe requires five independent training runs")
    return refs


def _losses(model, data, refs, horizon: int, scales: dict[str, Any],
            device: torch.device) -> dict[str, torch.Tensor]:
    arrays = _batch_arrays(data, refs, horizon)
    tensors = [torch.as_tensor(value,
        dtype=torch.bool if index == 7 else torch.float32, device=device)
        for index, value in enumerate(arrays)]
    history, initial, pose, commands, truth_body, truth_pose, truth_encoder, enc_mask, truth_accel = tensors
    model.train(False)
    model.reset(history, initial, pose)
    predicted = model.rollout(commands)
    body = predicted["states"][..., :3]
    body_scale = torch.as_tensor(scales["body_state_scale"], dtype=torch.float32,
                                 device=device)
    body_error = (body - truth_body) / body_scale
    body_loss = F.smooth_l1_loss(body_error, torch.zeros_like(body_error))

    pose_error = predicted["poses"] - truth_pose
    position_error = torch.linalg.vector_norm(pose_error[..., :2], dim=-1) / POSITION_SCALE_M
    heading_error = _wrap(pose_error[..., 2]) / HEADING_SCALE_RAD
    position_loss = F.smooth_l1_loss(position_error, torch.zeros_like(position_error))
    heading_loss = F.smooth_l1_loss(heading_error, torch.zeros_like(heading_error))

    measurement_scale = torch.as_tensor(scales["measurement_loss_scale"],
                                        dtype=torch.float32, device=device)
    measurement_error = (predicted["measurements"] - truth_encoder) / measurement_scale
    if enc_mask.any():
        measurement_loss = F.smooth_l1_loss(
            measurement_error[enc_mask], torch.zeros_like(measurement_error[enc_mask]))
    else:
        measurement_loss = body_loss.new_zeros(())

    acceleration_mean = torch.as_tensor(scales["acceleration_mean"],
                                        dtype=torch.float32, device=device)
    acceleration_scale = torch.as_tensor(scales["acceleration_scale"],
                                         dtype=torch.float32, device=device)
    implied = _implied_acceleration(body, initial[:, :3])
    accel_error = ((implied - acceleration_mean) / acceleration_scale
                   - (truth_accel - acceleration_mean) / acceleration_scale)
    acceleration_loss = F.smooth_l1_loss(accel_error, torch.zeros_like(accel_error))

    latent = predicted["latents"]
    latent_loss = latent.square().mean() if latent.shape[-1] else body_loss.new_zeros(())
    if latent.shape[-1] and latent.shape[1] > 1:
        latent_loss = latent_loss + 0.1 * (latent[:, 1:] - latent[:, :-1]).square().mean()
    body_limit = torch.as_tensor(model.config.body_increment_limit,
                                 dtype=torch.float32, device=device)
    normalized_increment = predicted["body_increments"] / body_limit
    support_loss = ((1.0 - predicted["support_confidence"].detach())[..., None]
                    * normalized_increment.square()).mean()
    return {
        "body": body_loss,
        "acceleration_consistency": acceleration_loss,
        "heading": heading_loss,
        "position": position_loss,
        "measurement": measurement_loss,
        "latent": latent_loss,
        "support": support_loss,
    }


def _flatten_gradients(gradients, parameters) -> torch.Tensor:
    flattened = []
    for grad, parameter in zip(gradients, parameters):
        flattened.append((torch.zeros_like(parameter) if grad is None else grad)
                         .detach().reshape(-1))
    return torch.cat(flattened)


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float | None:
    denominator = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    if not torch.isfinite(denominator) or denominator <= 1e-20:
        return None
    return float(torch.dot(a, b) / denominator)


def _parameter_slices(model) -> tuple[list[torch.nn.Parameter], dict[str, slice]]:
    parameters, ranges = [], {}
    cursor = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        parameters.append(parameter)
        ranges[name] = slice(cursor, cursor + parameter.numel())
        cursor += parameter.numel()
    return parameters, ranges


def _block_vector(vector: torch.Tensor, ranges: dict[str, slice], prefix: str) -> torch.Tensor:
    pieces = [vector[span] for name, span in ranges.items() if name.startswith(prefix)]
    return torch.cat(pieces) if pieces else vector.new_zeros(1)


def _yaw_row(vector: torch.Tensor, ranges: dict[str, slice], model) -> torch.Tensor:
    weight = ranges["body_residual.head.weight"]
    bias = ranges["body_residual.head.bias"]
    weight_rows = model.body_residual.head.weight.shape[1]
    weight_start = weight.start + 2 * weight_rows
    weight_stop = weight_start + weight_rows
    bias_index = bias.start + 2
    return torch.cat((vector[weight_start:weight_stop], vector[bias_index:bias_index + 1]))


def _component_gradients(model, losses: dict[str, torch.Tensor],
                         parameters: list[torch.nn.Parameter],
                         ranges: dict[str, slice]) -> dict[str, Any]:
    raw, weighted = {}, {}
    for name, loss in losses.items():
        gradient = torch.autograd.grad(loss, parameters, retain_graph=True,
                                       allow_unused=True)
        vector = _flatten_gradients(gradient, parameters)
        raw[name] = vector
        weighted[name] = vector * WEIGHTS[name]
    objective = sum(weighted.values())
    trajectory = weighted["body"] + weighted["heading"] + weighted["position"]
    yaw = {name: _yaw_row(vector, ranges, model) for name, vector in weighted.items()}
    trajectory_yaw = yaw["body"] + yaw["heading"] + yaw["position"]
    support_yaw = yaw["support"]
    metrics = {}
    for name in losses:
        g = weighted[name]
        metrics[name] = {
            "loss_value": float(losses[name].detach()),
            "objective_weight": WEIGHTS[name],
            "raw_gradient_l2": float(torch.linalg.vector_norm(raw[name])),
            "weighted_gradient_l2": float(torch.linalg.vector_norm(g)),
            "body_residual_final_layer_weighted_gradient_l2": float(
                torch.linalg.vector_norm(_block_vector(g, ranges, "body_residual.head."))),
            "yaw_output_row_weighted_gradient_l2": float(torch.linalg.vector_norm(yaw[name])),
            "history_encoder_weighted_gradient_l2": float(
                torch.linalg.vector_norm(_block_vector(g, ranges, "history_encoder."))),
            "latent_transition_weighted_gradient_l2": float(
                torch.linalg.vector_norm(_block_vector(g, ranges, "latent_transition."))),
            "weighted_gradient_relative_to_body": float(
                torch.linalg.vector_norm(g)
                / max(float(torch.linalg.vector_norm(weighted["body"])), 1e-20)),
        }
    metrics["combined"] = {
        "weighted_total_gradient_l2": float(torch.linalg.vector_norm(objective)),
        "body_heading_position_gradient_l2": float(torch.linalg.vector_norm(trajectory)),
        "trajectory_vs_support_cosine": _cosine(trajectory, weighted["support"]),
        "yaw_row_trajectory_vs_support_cosine": _cosine(trajectory_yaw, support_yaw),
        "yaw_row_trajectory_gradient_l2": float(torch.linalg.vector_norm(trajectory_yaw)),
        "yaw_row_support_gradient_l2": float(torch.linalg.vector_norm(support_yaw)),
        "support_to_trajectory_gradient_norm_ratio": float(
            torch.linalg.vector_norm(weighted["support"])
            / max(float(torch.linalg.vector_norm(trajectory)), 1e-20)),
    }
    return metrics


def run(device_name: str = "cpu") -> dict[str, Any]:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                          text=True, capture_output=True).stdout.strip()
    if head != SOURCE_COMMIT:
        raise RuntimeError(f"WP25 requires frozen source commit {SOURCE_COMMIT}; found {head}")
    comparator_file = NEXT_ROOT / "frozen_comparators.json"
    if not comparator_file.is_file():
        raise FileNotFoundError("WP24 frozen comparator manifest is required")
    comparators = json.loads(comparator_file.read_text(encoding="utf-8"))
    expected = comparators["comparators"]["wp22_A2"]["sha256"]
    if sha256_file(CHECKPOINT) != expected:
        raise RuntimeError("WP22/A2 checkpoint changed after WP24 freeze")
    checkpoint_hash = sha256_file(CHECKPOINT)
    torch.set_num_threads(1)
    device = torch.device(device_name)
    data, wp20 = _load_data()
    model, _ = load_checkpoint(CHECKPOINT, device_name)
    config, scales, _ = _training_windows_and_stats(data, wp20)
    comparable_config = replace(
        config, body_state_normalized_limit=model.config.body_state_normalized_limit)
    if model.config != comparable_config:
        raise RuntimeError("reconstructed WP22 training scales differ from checkpoint config")
    if not np.allclose(scales["body_state_scale"], model.config.state_scale[:3],
                       rtol=0.0, atol=1e-7):
        raise RuntimeError("reconstructed WP22 body loss scale differs from checkpoint")
    refs = _fixed_training_starts(data)
    start_records = []
    for ci, si, row in refs:
        capture = data.captures[ci]
        begin = int(capture.bounds[si, 0]) + row
        run = int(capture.sequence_run[si])
        start_records.append({
            "run_id": str(capture.run_ids[run]),
            "sequence_index": int(si), "source_row": int(row),
            "absolute_row": begin,
            "split_role": "training",
        })
    parameters, ranges = _parameter_slices(model)
    results: dict[str, Any] = {}
    for horizon in HORIZONS:
        print(f"WP25 gradient attribution horizon={horizon * DT_S:.2f}s, "
              f"independent training runs={len(refs)}", flush=True)
        losses = _losses(model, data, refs, horizon, scales, device)
        values = _component_gradients(model, losses, parameters, ranges)
        results[f"{horizon * DT_S:g}s"] = values
        model.zero_grad(set_to_none=True)
    if sha256_file(CHECKPOINT) != checkpoint_hash:
        raise RuntimeError("WP25 diagnostic unexpectedly modified the frozen checkpoint")
    report = {
        "schema_version": 1,
        "work_package": "WP25.4",
        "source_commit": SOURCE_COMMIT,
        "candidate_checkpoint": CHECKPOINT.relative_to(ROOT).as_posix(),
        "candidate_checkpoint_sha256": checkpoint_hash,
        "split_role": "training only",
        "training_run_count": len(refs),
        "training_start_records": start_records,
        "horizons_seconds": [h * DT_S for h in HORIZONS],
        "gradient_norms_are_weighted_for_optimizer_effect": True,
        "raw_component_gradients_also_reported": True,
        "optimizer_steps": 0,
        "model_training_performed": False,
        "gradient_attribution": results,
        "interpretation": {
            "support_conflicts_with_trajectory_when_cosine_is_negative": True,
            "support_magnitude_can_be_assessed_by_norm_ratio": True,
            "no gradient result is treated as causal proof by itself": True,
        },
    }
    _write_json(OUTPUT, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    report = run(args.device)
    print(json.dumps({"report": str(OUTPUT),
                      "horizons": list(report["gradient_attribution"]),
                      "optimizer_steps": report["optimizer_steps"]},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
