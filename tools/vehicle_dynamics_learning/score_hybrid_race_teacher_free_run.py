#!/usr/bin/env python3
"""Compare a hybrid race teacher with its frozen nominal on unseen captures.

Each eligible sequence receives a causal 2 s context, then runs recursively
from commands alone for the rest of that uninterrupted sequence. Ground truth
is used only for initialization, context, and scoring.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from tools.vehicle_dynamics_learning.four_wheel_greybox import (
    DT_S,
    WHEEL_RADIUS_M,
    _initial_state,
    _physical_model,
    _targets,
)
from tools.vehicle_dynamics_learning.hybrid_race_teacher import (
    HISTORY_FEATURE_SIZE,
    HISTORY_STEPS,
    RESIDUAL_NAMES,
    hybrid_model_type,
    integrate_pose,
)
from tools.vehicle_dynamics_learning.score_four_wheel_greybox_free_run import (
    _rmse_bias,
    _sequence_rows,
    _wrap_angle,
)
from tools.vehicle_dynamics_learning.train_hybrid_race_teacher import (
    _history_features,
)
from tools.vehicle_dynamics_learning.train_nssm import _load_dataset, _torch


HORIZONS_S = (0.25, 0.5, 0.75, 1.0, 2.0, 5.0)
STATE_NAMES = (
    "u_com_mps", "v_com_mps", "yaw_rate_rps",
    "rear_left_surface_mps", "rear_right_surface_mps",
    "steering_feedback_rad", "throttle_feedback_norm",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _model_pair(torch, nn, nominal_checkpoint: Path,
                hybrid_checkpoint: Path):
    nominal_payload = torch.load(
        nominal_checkpoint, map_location="cpu", weights_only=False)
    hybrid_payload = torch.load(
        hybrid_checkpoint, map_location="cpu", weights_only=False)
    metadata = hybrid_payload["metadata"]
    if metadata.get("nominal_checkpoint_sha256") != _sha256(nominal_checkpoint):
        raise ValueError("hybrid checkpoint does not reference this nominal checkpoint")
    if metadata.get("architecture") != "four_wheel_nominal_history_residual":
        raise ValueError("unsupported hybrid checkpoint architecture")

    nominal_type = _physical_model(
        torch, nn, bool(nominal_payload["metadata"][
            "tire_relaxation_state_enabled"]))
    nominal = nominal_type()
    nominal.load_state_dict(nominal_payload["state_dict"], strict=True)
    nominal.eval()
    for parameter in nominal.parameters():
        parameter.requires_grad_(False)

    state = hybrid_payload["state_dict"]
    history_mean = state["history_mean"].cpu().numpy()
    history_scale = state["history_scale"].cpu().numpy()
    model_type = hybrid_model_type(
        torch, nn, nominal, history_mean, history_scale,
        int(metadata["latent_size"]))
    candidate = model_type()
    candidate.load_state_dict(state, strict=True)
    candidate.eval()

    baseline = copy.deepcopy(candidate)
    final_layer = baseline.residual_head[-1]
    with torch.no_grad():
        final_layer.weight.zero_()
        final_layer.bias.zero_()
    baseline.eval()
    return candidate, baseline, metadata


def _score_sequence(torch, model, data: dict[str, Any], state: np.ndarray,
                    history: np.ndarray, start: int, end: int,
                    condition: str, run_id: str, device) -> dict[str, Any]:
    context_end = start + HISTORY_STEPS - 1
    context_start = context_end - HISTORY_STEPS + 1
    if context_start < start:
        raise ValueError(f"{run_id}/{condition}: history crosses a sequence boundary")
    history_np = history[context_start:context_end + 1]
    if (history_np.shape != (HISTORY_STEPS, HISTORY_FEATURE_SIZE)
            or not np.isfinite(history_np).all()
            or not data["sensor_valid"][context_start:context_end + 1].all()):
        raise ValueError(f"{run_id}/{condition}: invalid or incomplete causal context")

    initial_np = _initial_state(data, context_end, state, start)
    commands_np = data["frames"][context_end:end - 1, 7:9]
    if len(commands_np) < round(2.0 / DT_S):
        raise ValueError(f"{run_id}/{condition}: less than 2 s after context")
    initial = torch.as_tensor(initial_np[None], dtype=torch.float32, device=device)
    history_t = torch.as_tensor(history_np[None], dtype=torch.float32, device=device)
    commands = torch.as_tensor(commands_np[None], dtype=torch.float32, device=device)

    with torch.no_grad():
        predicted, _, _, _ = model.rollout(initial, history_t, commands)
        local_pose = integrate_pose(torch, predicted, initial)[0].cpu().numpy()
    internal = predicted[0].cpu().numpy()
    predicted_state = np.column_stack((
        internal[:, :3], internal[:, 5:7] * WHEEL_RADIUS_M,
        internal[:, 17:19],
    )).astype(np.float64)
    truth_state = np.column_stack((
        state[context_end + 1:end],
        data["frames"][context_end + 1:end, 3:5],
    )).astype(np.float64)
    pose0 = data["simulator_pose_xyyaw"][context_end].astype(np.float64)
    pose_truth = data["simulator_pose_xyyaw"][context_end + 1:end].astype(
        np.float64)
    cosine, sine = math.cos(float(pose0[2])), math.sin(float(pose0[2]))
    pose_predicted = np.empty_like(local_pose, dtype=np.float64)
    pose_predicted[:, 0] = pose0[0] + cosine * local_pose[:, 0] - sine * local_pose[:, 1]
    pose_predicted[:, 1] = pose0[1] + sine * local_pose[:, 0] + cosine * local_pose[:, 1]
    pose_predicted[:, 2] = pose0[2] + local_pose[:, 2]

    position_error = pose_predicted[:, :2] - pose_truth[:, :2]
    heading_error = _wrap_angle(pose_predicted[:, 2] - pose_truth[:, 2])
    speed_error = (np.linalg.norm(predicted_state[:, :2], axis=1)
                   - np.linalg.norm(truth_state[:, :2], axis=1))
    state_error = predicted_state - truth_state
    radial_error = np.linalg.norm(position_error, axis=1)
    horizon_rows = {}
    for horizon_s in HORIZONS_S:
        index = round(horizon_s / DT_S) - 1
        if index < len(radial_error):
            horizon_rows[f"{horizon_s:g}s"] = {
                "position_error_m": float(radial_error[index]),
                "speed_abs_error_mps": float(abs(speed_error[index])),
                "heading_abs_error_rad": float(abs(heading_error[index])),
            }
    return {
        "condition": condition,
        "context_seconds": (HISTORY_STEPS - 1) * DT_S,
        "free_run_seconds": len(predicted_state) * DT_S,
        "free_run_steps": len(predicted_state),
        "initial_speed_mps": float(np.linalg.norm(state[context_end, :2])),
        "truth_speed_p05_p50_p95_mps": np.quantile(
            np.linalg.norm(truth_state[:, :2], axis=1), (0.05, 0.50, 0.95)).tolist(),
        "whole_free_run": {
            "state_names": list(STATE_NAMES),
            "state_error": _rmse_bias(state_error),
            "speed_rmse_mps": float(np.sqrt(np.mean(speed_error ** 2))),
            "position_xy_rmse_m": np.sqrt(
                np.mean(position_error ** 2, axis=0)).tolist(),
            "position_radial_rmse_m": float(np.sqrt(np.mean(radial_error ** 2))),
            "position_radial_p95_m": float(np.quantile(radial_error, 0.95)),
            "position_endpoint_m": float(radial_error[-1]),
            "heading_rmse_rad": float(np.sqrt(np.mean(heading_error ** 2))),
            "heading_endpoint_abs_rad": float(abs(heading_error[-1])),
        },
        "horizons": horizon_rows,
    }


def _one_step_diagnostic(torch, model, data: dict[str, Any], state: np.ndarray,
                         history: np.ndarray, start: int, end: int,
                         device, stride: int = 4) -> dict[str, Any]:
    """Teacher-forced local check, with causal histories and truth state init."""
    first = start + HISTORY_STEPS - 1
    indexes = np.arange(first, end - 1, stride, dtype=np.int64)
    rows = []
    residual_rows = []
    for offset in range(0, len(indexes), 512):
        current = indexes[offset:offset + 512]
        histories = np.stack([
            history[index - HISTORY_STEPS + 1:index + 1]
            for index in current
        ]).astype(np.float32)
        if not data["sensor_valid"][
                current[:, None] - np.arange(HISTORY_STEPS - 1, -1, -1)[None, :]
        ].all():
            raise ValueError("one-step diagnostic context contains invalid sensors")
        initial = np.stack([
            _initial_state(data, int(index), state, start) for index in current
        ]).astype(np.float32)
        commands_np = data["frames"][current, 7:9]
        with torch.no_grad():
            predicted, _, _, residual = model.rollout(
                torch.as_tensor(initial, dtype=torch.float32, device=device),
                torch.as_tensor(histories, dtype=torch.float32, device=device),
                torch.as_tensor(commands_np[:, None], dtype=torch.float32,
                                device=device),
            )
        internal = predicted[:, 0].cpu().numpy()
        predicted_output = np.column_stack((
            internal[:, :3], internal[:, 5:7] * WHEEL_RADIUS_M,
            internal[:, 17:19],
        ))
        target_output = np.column_stack((
            state[current + 1], data["frames"][current + 1, 3:5],
        ))
        speed = np.linalg.norm(state[current, :2], axis=1)
        steering = np.abs(commands_np[:, 0])
        rows.append((predicted_output - target_output, speed, steering))
        residual_rows.append(residual[:, 0].cpu().numpy())

    if not rows:
        raise ValueError("one-step diagnostic has no eligible samples")
    error = np.concatenate([row[0] for row in rows])
    speed = np.concatenate([row[1] for row in rows])
    steering = np.concatenate([row[2] for row in rows])
    residual = np.concatenate(residual_rows)
    limits = model.residual_limits.detach().cpu().numpy()

    def summarize(mask: np.ndarray) -> dict[str, Any]:
        if not mask.any():
            return {"count": 0}
        local = error[mask]
        local_speed_error = (np.linalg.norm(
            local[:, :2] + all_target_outputs[mask, :2], axis=1)
            - np.linalg.norm(all_target_outputs[mask, :2], axis=1))
        return {
            "count": int(mask.sum()),
            "state_rmse": np.sqrt(np.mean(local ** 2, axis=0)).tolist(),
            "state_bias": np.mean(local, axis=0).tolist(),
            "speed_rmse_mps": float(np.sqrt(np.mean(local_speed_error ** 2))),
            "speed_bias_mps": float(np.mean(local_speed_error)),
        }

    all_target_outputs = np.concatenate([
        np.column_stack((state[indexes[offset:offset + 512] + 1],
                         data["frames"][indexes[offset:offset + 512] + 1, 3:5]))
        for offset in range(0, len(indexes), 512)
    ])
    speed_bins = ((0.0, 3.0), (3.0, 6.0), (6.0, 9.0), (9.0, 12.1))
    steering_bins = ((0.0, 0.15), (0.15, 0.30), (0.30, 0.45), (0.45, 0.524))
    cells = {}
    for speed_low, speed_high in speed_bins:
        for steer_low, steer_high in steering_bins:
            mask = ((speed >= speed_low) & (speed < speed_high)
                    & (steering >= steer_low) & (steering < steer_high))
            cells[f"speed_{speed_low:g}_{speed_high:g}_steer_{steer_low:g}_{steer_high:g}"] = summarize(mask)
    return {
        "sample_period_stride": stride,
        "sample_count": int(len(error)),
        "state_names": list(STATE_NAMES),
        "state_rmse": np.sqrt(np.mean(error ** 2, axis=0)).tolist(),
        "state_bias": np.mean(error, axis=0).tolist(),
        "state_absolute_p95": np.quantile(np.abs(error), 0.95, axis=0).tolist(),
        "speed_rmse_mps": summarize(np.ones(len(speed), dtype=bool))["speed_rmse_mps"],
        "speed_steering_cells": cells,
        "residual_names": list(RESIDUAL_NAMES),
        "residual_abs_p50": np.quantile(np.abs(residual), 0.50, axis=0).tolist(),
        "residual_abs_p95": np.quantile(np.abs(residual), 0.95, axis=0).tolist(),
        "residual_mean": np.mean(residual, axis=0).tolist(),
        "residual_saturation_fraction": np.mean(
            np.abs(residual) >= 0.95 * limits, axis=0).tolist(),
    }


def score(hybrid_checkpoint: Path, nominal_checkpoint: Path,
          dataset_paths: list[Path], output_path: Path,
          selected_run_ids: list[str] | None = None,
          one_step_diagnostic: bool = False,
          ablation: str = "full") -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    torch, nn = _torch()
    torch.set_num_threads(1)
    device = torch.device("cpu")
    candidate, baseline, checkpoint_metadata = _model_pair(
        torch, nn, nominal_checkpoint, hybrid_checkpoint)
    if ablation not in ("full", "body-only", "wheel-only"):
        raise ValueError("ablation must be full, body-only, or wheel-only")
    ablation_handle = None
    if ablation != "full":
        mask_values = ((1.0, 1.0, 1.0, 0.0, 0.0)
                       if ablation == "body-only"
                       else (0.0, 0.0, 0.0, 1.0, 1.0))
        mask = torch.as_tensor(mask_values, dtype=torch.float32)
        ablation_handle = candidate.residual_head[-1].register_forward_hook(
            lambda _module, _inputs, output: output * mask)
    training_runs = set(map(str, checkpoint_metadata["training_runs"]))
    results = []

    for dataset_path in dataset_paths:
        data = _load_dataset(dataset_path)
        if (data["schema_version"] < 7 or data.get("sensor_frames") is None
                or data.get("sensor_valid") is None
                or data["simulator_pose_xyyaw"] is None
                or data["simulator_rigid_state"] is None
                or data.get("packet_sequence") is None
                or not np.allclose(data["dt_s"], DT_S, rtol=0.0, atol=1e-7)):
            raise ValueError(f"{dataset_path}: needs aligned sensor, pose, and 25 ms plant labels")
        if data["feature_names"] != checkpoint_metadata["feature_names"]:
            raise ValueError(f"{dataset_path}: checkpoint feature ordering differs")
        state, _ = _targets(data)
        history = _history_features(data)
        requested = set(selected_run_ids or map(str, data["run_ids"]))
        run_lookup = {str(run_id): index
                      for index, run_id in enumerate(data["run_ids"])}
        missing = sorted(requested - set(run_lookup))
        if missing:
            raise ValueError(f"{dataset_path}: requested runs missing: {missing}")
        overlap = sorted(requested & training_runs)
        if overlap:
            raise ValueError(f"refusing to score training captures: {overlap}")

        for run_id in sorted(requested):
            run_index = run_lookup[run_id]
            sequences = _sequence_rows(
                data, run_index, min_future_steps=round(2.0 / DT_S))
            if not sequences:
                raise ValueError(f"{run_id}: no eligible uninterrupted sequence")
            pair_rows = []
            for start, end, condition in sequences:
                hybrid_row = _score_sequence(
                    torch, candidate, data, state, history,
                    start, end, condition, run_id, device)
                nominal_row = _score_sequence(
                    torch, baseline, data, state, history,
                    start, end, condition, run_id, device)
                one_step_rows = None
                if one_step_diagnostic:
                    one_step_rows = {
                        "hybrid": _one_step_diagnostic(
                            torch, candidate, data, state, history,
                            start, end, device),
                        "nominal": _one_step_diagnostic(
                            torch, baseline, data, state, history,
                            start, end, device),
                    }
                pair_rows.append({
                    "sequence_start_index": start,
                    "sequence_end_index": end,
                    "hybrid": hybrid_row,
                    "nominal": nominal_row,
                    "one_step_diagnostic": one_step_rows,
                })
            run_result = {"run_id": run_id,
                          "dataset": str(dataset_path.resolve()),
                          "sequence_count": len(pair_rows),
                          "sequences": pair_rows}
            for model_name in ("hybrid", "nominal"):
                summaries = [row[model_name]["whole_free_run"]
                             for row in pair_rows]
                run_result[f"macro_{model_name}"] = {
                    metric: float(np.mean([row[metric] for row in summaries]))
                    for metric in (
                        "position_radial_rmse_m", "speed_rmse_mps",
                        "heading_rmse_rad", "position_endpoint_m",
                    )
                }
            run_result["paired_hybrid_minus_nominal"] = {
                metric: run_result[f"macro_hybrid"][metric]
                        - run_result[f"macro_nominal"][metric]
                for metric in run_result["macro_hybrid"]
            }
            results.append(run_result)

    report = {
        "schema_version": 1,
        "comparison": "hybrid checkpoint versus its identical nominal with residual head zeroed",
        "hybrid_residual_ablation": ablation,
        "hybrid_checkpoint": str(hybrid_checkpoint.resolve()),
        "hybrid_checkpoint_sha256": _sha256(hybrid_checkpoint),
        "nominal_checkpoint": str(nominal_checkpoint.resolve()),
        "nominal_checkpoint_sha256": _sha256(nominal_checkpoint),
        "training_runs_excluded": sorted(training_runs),
        "independent_capture_count": len(results),
        "capture_level_uncertainty_note": (
            "sequences within a capture are correlated; capture/run is the uncertainty unit"),
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    if ablation_handle is not None:
        ablation_handle.remove()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hybrid_checkpoint", type=Path)
    parser.add_argument("nominal_checkpoint", type=Path)
    parser.add_argument("datasets", type=Path, nargs="+")
    parser.add_argument("--run-id", action="append", dest="run_ids")
    parser.add_argument("--one-step-diagnostic", action="store_true")
    parser.add_argument("--ablation", choices=("full", "body-only", "wheel-only"),
                        default="full")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = score(args.hybrid_checkpoint, args.nominal_checkpoint,
                   args.datasets, args.output, args.run_ids,
                   args.one_step_diagnostic, args.ablation)
    concise = [{
        "run_id": row["run_id"],
        "sequence_count": row["sequence_count"],
        "hybrid": row["macro_hybrid"],
        "nominal": row["macro_nominal"],
        "hybrid_minus_nominal": row["paired_hybrid_minus_nominal"],
        "one_step": row["sequences"][0]["one_step_diagnostic"]
        if row["sequences"] else None,
        "ablation": report["hybrid_residual_ablation"],
    } for row in report["results"]]
    print(json.dumps({"independent_capture_count": len(concise),
                      "results": concise}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
