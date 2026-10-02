#!/usr/bin/env python3
"""Test whether causal history can recover latent front-wheel state.

For each held-out reset-separated sequence, fit only the two unmeasured front
wheel surface speeds against the preceding 0.5 s of body, rear-wheel, and
actuator history. Then predict the later command sequence without future
sensor/truth feedback. This is a diagnostic upper bound, not a deployable
observer or a model promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from tools.vehicle_dynamics_learning.four_wheel_greybox import (
    DT_S,
    WHEEL_RADIUS_M,
    _initial_state,
    _integrate_pose,
    _physical_model,
    _targets,
)
from tools.vehicle_dynamics_learning.train_nssm import _load_dataset, _torch


HISTORY_STEPS = 80
FIT_HORIZON_STEPS = 20
FRONT_SPEED_GRID_MPS = tuple(float(value) for value in range(0, 21, 2))
HORIZONS_S = (0.25, 0.75, 2.0, 5.0)
OUTPUT_SCALE = np.asarray((0.5, 0.25, 0.25, 0.5, 0.5, 0.1, 0.2),
                          dtype=np.float32)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _wrap_angle(values: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(values), np.cos(values))


def _sequences(data: dict[str, Any], run_index: int
               ) -> list[tuple[int, int, int, str]]:
    sequence_ids = np.flatnonzero(data["seq_run"] == run_index)
    rows = []
    for sequence_id in sequence_ids:
        start, end = map(int, data["bounds"][sequence_id])
        if end - start <= HISTORY_STEPS + round(0.25 / DT_S):
            continue
        condition_id = int(data["sequence_condition_id"][sequence_id])
        label = str(data["condition_labels"][condition_id])
        reset_id = int(data["sequence_reset_index"][sequence_id])
        rows.append((start, end, reset_id, label))
    rows.sort(key=lambda row: row[0])
    if not rows:
        raise ValueError("held-out run has no reset-separated usable sequences")
    for start, end, _, label in rows:
        if (not np.isfinite(data["frames"][start:end]).all()
                or not np.isfinite(data["simulator_pose_xyyaw"][start:end]).all()
                or not np.isfinite(data["simulator_rigid_state"][start:end]).all()):
            raise ValueError(f"{label}: sequence has incomplete labels")
        if end - start <= HISTORY_STEPS + round(2.0 / DT_S):
            raise ValueError(f"{label}: sequence is too short for the 2 s forecast")
    return rows


def _state_output(prediction: np.ndarray) -> np.ndarray:
    return np.column_stack((
        prediction[:, :3],
        prediction[:, 5:7] * WHEEL_RADIUS_M,
        prediction[:, 17:19],
    ))


def _sequence_metrics(prediction: np.ndarray, truth: np.ndarray,
                      pose_prediction: np.ndarray,
                      pose_truth: np.ndarray) -> dict[str, Any]:
    state_error = prediction - truth
    pose_error = pose_prediction - pose_truth
    pose_error[:, 2] = _wrap_angle(pose_error[:, 2])
    speed_error = (np.linalg.norm(prediction[:, :2], axis=1)
                   - np.linalg.norm(truth[:, :2], axis=1))
    output = {}
    for horizon_s in HORIZONS_S:
        index = round(horizon_s / DT_S) - 1
        if index >= len(state_error):
            continue
        output[f"{horizon_s:g}s"] = {
            "position_error_m": float(np.linalg.norm(pose_error[index, :2])),
            "heading_abs_error_rad": float(abs(pose_error[index, 2])),
            "speed_abs_error_mps": float(abs(speed_error[index])),
        }
    return {
        "free_run_seconds": len(prediction) * DT_S,
        "position_xy_rmse_m": np.sqrt(
            np.mean(pose_error[:, :2] ** 2, axis=0)).tolist(),
        "position_radial_rmse_m": float(np.sqrt(np.mean(
            np.sum(pose_error[:, :2] ** 2, axis=1)))),
        "position_endpoint_error_m": float(np.linalg.norm(pose_error[-1, :2])),
        "heading_rmse_rad": float(np.sqrt(np.mean(pose_error[:, 2] ** 2))),
        "speed_rmse_mps": float(np.sqrt(np.mean(speed_error ** 2))),
        "body_state_rmse": np.sqrt(np.mean(state_error[:, :3] ** 2,
                                             axis=0)).tolist(),
        "horizons": output,
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    horizon_keys = sorted(set.intersection(*[
        set(row["horizons"]) for row in rows
    ]), key=lambda item: float(item[:-1]))
    result = {
        "sequence_count": len(rows),
        "median_sequence_position_radial_rmse_m": float(np.median(
            [row["position_radial_rmse_m"] for row in rows])),
        "p90_sequence_position_radial_rmse_m": float(np.quantile(
            [row["position_radial_rmse_m"] for row in rows], 0.90)),
        "median_sequence_speed_rmse_mps": float(np.median(
            [row["speed_rmse_mps"] for row in rows])),
        "median_sequence_heading_rmse_rad": float(np.median(
            [row["heading_rmse_rad"] for row in rows])),
        "horizons": {},
    }
    for key in horizon_keys:
        result["horizons"][key] = {
            metric: float(np.median([row["horizons"][key][metric]
                                     for row in rows]))
            for metric in ("position_error_m", "speed_abs_error_mps",
                           "heading_abs_error_rad")
        }
    return result


def evaluate(checkpoint_path: Path, dataset_path: Path, run_id: str,
             output_path: Path) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    torch, nn = _torch()
    torch.set_num_threads(1)
    checkpoint = torch.load(checkpoint_path, map_location="cpu",
                            weights_only=False)
    metadata = checkpoint["metadata"]
    data = _load_dataset(dataset_path)
    if data["schema_version"] < 7 or data["simulator_pose_xyyaw"] is None:
        raise ValueError("diagnostic requires schema-7+ pose-labelled data")
    if (data["simulator_rigid_state"] is None
            or not np.allclose(data["dt_s"], DT_S, rtol=0.0, atol=1.0e-7)):
        raise ValueError("diagnostic requires complete 25 ms rigid-state labels")
    if metadata["feature_names"] != data["feature_names"]:
        raise ValueError("checkpoint and capture feature layouts differ")
    run_lookup = {str(value): index
                  for index, value in enumerate(data["run_ids"])}
    if run_id not in run_lookup:
        raise ValueError(f"held-out run is absent from dataset: {run_id}")
    run_index = run_lookup[run_id]
    if data["splits"][run_index] not in ("validation", "final_test"):
        raise ValueError(f"run is not held out: {data['splits'][run_index]}")
    if run_id in set(map(str, metadata["training_runs"])):
        raise ValueError("refusing to fit latent state on a training run")

    sequences = _sequences(data, run_index)
    physical_state, _ = _targets(data)
    model_type = _physical_model(
        torch, nn, bool(metadata["tire_relaxation_state_enabled"]))
    model = model_type()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()

    starts = []
    context_ends = []
    ends = []
    labels = []
    initials = []
    fit_commands = []
    fit_targets = []
    future_commands = []
    for start, end, _, label in sequences:
        context_end = start + HISTORY_STEPS - 1
        history_start = context_end - FIT_HORIZON_STEPS
        starts.append(history_start)
        context_ends.append(context_end)
        ends.append(end)
        labels.append(label)
        baseline_initial = _initial_state(
            data, history_start, physical_state, start)
        initials.append(baseline_initial)
        fit_commands.append(
            data["frames"][history_start:context_end, 7:9])
        fit_targets.append(np.column_stack((
            physical_state[history_start + 1:context_end + 1],
            data["frames"][history_start + 1:context_end + 1, 3:5],
        )))
        future_commands.append(data["frames"][context_end:end - 1, 7:9])

    initial_np = np.stack(initials).astype(np.float32)
    fit_commands_np = np.stack(fit_commands).astype(np.float32)
    fit_targets_np = np.stack(fit_targets).astype(np.float32)
    # Search a coarse, physically bounded grid plus each sequence's pure-roll
    # initialization. This avoids accepting a failed local optimizer as
    # evidence about hidden-state identifiability.
    candidate_surfaces = []
    candidate_initials = []
    for row, (history_start, sequence) in enumerate(zip(starts, sequences)):
        start = sequence[0]
        sequence_candidates = [
            _initial_state(data, history_start, physical_state, start,
                           front_surface_speed_mps=(left, right))
            for left in FRONT_SPEED_GRID_MPS
            for right in FRONT_SPEED_GRID_MPS
        ]
        baseline_surface = tuple(
            (initial_np[row, 3:5] * WHEEL_RADIUS_M).tolist())
        sequence_candidates.append(_initial_state(
            data, history_start, physical_state, start,
            front_surface_speed_mps=baseline_surface))
        candidate_initials.append(sequence_candidates)
        candidate_surfaces.append([
            (left, right) for left in FRONT_SPEED_GRID_MPS
            for right in FRONT_SPEED_GRID_MPS
        ] + [baseline_surface])

    candidate_count = len(candidate_surfaces[0])
    candidate_initial_np = np.asarray(candidate_initials, dtype=np.float32)
    candidate_initial_t = torch.as_tensor(
        candidate_initial_np.reshape(-1, candidate_initial_np.shape[-1]))
    fit_commands_t = torch.as_tensor(np.repeat(
        fit_commands_np[:, None, :, :], candidate_count, axis=1
    ).reshape(-1, FIT_HORIZON_STEPS, 2))
    fit_targets_t = torch.as_tensor(np.repeat(
        fit_targets_np[:, None, :, :], candidate_count, axis=1
    ).reshape(-1, FIT_HORIZON_STEPS, 7))
    output_scale_t = torch.as_tensor(OUTPUT_SCALE)
    with torch.no_grad():
        history_prediction, _ = model.rollout(
            candidate_initial_t, fit_commands_t)
        history_observed = torch.cat((
            history_prediction[:, :, :3],
            history_prediction[:, :, 5:7] * WHEEL_RADIUS_M,
            history_prediction[:, :, 17:19],
        ), dim=-1)
        history_error = (history_observed - fit_targets_t) / output_scale_t
        history_loss = nn.functional.smooth_l1_loss(
            history_error, torch.zeros_like(history_error),
            beta=0.5, reduction="none").mean(dim=(1, 2))
        history_loss = history_loss.reshape(len(sequences), candidate_count)
        best_candidate = history_loss.argmin(dim=1)
        base_candidate = candidate_count - 1
        row_indexes = torch.arange(len(sequences))
        fitted_surface_np = np.asarray([
            candidate_surfaces[row][int(best_candidate[row])]
            for row in range(len(sequences))
        ], dtype=np.float32)
        adapted_initial = history_prediction.reshape(
            len(sequences), candidate_count, FIT_HORIZON_STEPS, -1
        )[row_indexes, best_candidate, -1].clone()
        baseline_history_loss = history_loss[:, base_candidate].cpu().numpy()
        best_history_loss = history_loss[row_indexes, best_candidate].cpu().numpy()

        base_rows = []
        adapted_rows = []
        for row, (start, context_end, end, label) in enumerate(
                zip(starts, context_ends, ends, labels)):
            truth_at_context = physical_state[context_end]
            measured_frame = data["frames"][context_end]
            base_initial = torch.as_tensor(_initial_state(
                data, context_end, physical_state, sequences[row][0]
            )[None], dtype=torch.float32)
            hybrid_initial = adapted_initial[row:row + 1].clone()
            # Assimilate only the state channels available at forecast time;
            # preserve inferred front-wheel and tire-history latent states.
            hybrid_initial[:, :3] = torch.as_tensor(
                truth_at_context[:3][None], dtype=torch.float32)
            hybrid_initial[:, 5:7] = torch.as_tensor(
                measured_frame[5:7][None] / WHEEL_RADIUS_M,
                dtype=torch.float32)
            hybrid_initial[:, 17:19] = torch.as_tensor(
                measured_frame[3:5][None], dtype=torch.float32)

            future = torch.as_tensor(
                future_commands[row][None], dtype=torch.float32)
            base_prediction, _ = model.rollout(base_initial, future)
            adapted_prediction, _ = model.rollout(hybrid_initial, future)
            base_state = _state_output(base_prediction[0].numpy())
            adapted_state = _state_output(adapted_prediction[0].numpy())
            truth = np.column_stack((
                physical_state[context_end + 1:end],
                data["frames"][context_end + 1:end, 3:5],
            )).astype(np.float64)
            initial_truth = physical_state[context_end]
            initial_pose = data["simulator_pose_xyyaw"][context_end]
            pose_truth = data["simulator_pose_xyyaw"][context_end + 1:end]
            base_pose = _integrate_pose(base_state, initial_truth, initial_pose)
            adapted_pose = _integrate_pose(
                adapted_state, initial_truth, initial_pose)
            base_rows.append({
                "condition": label,
                **_sequence_metrics(base_state, truth, base_pose, pose_truth),
            })
            adapted_rows.append({
                "condition": label,
                "inferred_initial_front_surface_mps":
                    fitted_surface_np[row].tolist(),
                "history_loss_baseline": float(baseline_history_loss[row]),
                "history_loss_grid_best": float(best_history_loss[row]),
                **_sequence_metrics(adapted_state, truth,
                                   adapted_pose, pose_truth),
            })

    report = {
        "schema_version": 1,
        "purpose": "diagnostic upper bound for hidden front-wheel state; not a deployable observer or promoted plant",
        "future_truth_or_sensor_feedback": False,
        "history_fit": {
            "duration_s": FIT_HORIZON_STEPS * DT_S,
            "grid_candidates_per_sequence": candidate_count,
            "grid_speed_values_mps": list(FRONT_SPEED_GRID_MPS),
            "optimized_latents": ["front_left_surface_speed", "front_right_surface_speed"],
            "fit_signals": ["simulator-labelled body motion", "rear encoders", "actuator feedback"],
            "median_baseline_normalized_history_loss": float(np.median(
                baseline_history_loss)),
            "median_best_grid_normalized_history_loss": float(np.median(
                best_history_loss)),
        },
        "forecast": "2 s history context followed by commands only; inferred latent state is carried forward, measured body/rear-wheel/actuator state is assimilated once at the forecast start",
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "dataset": str(dataset_path.resolve()),
        "dataset_sha256": _sha256(dataset_path),
        "run_id": run_id,
        "whole_run_replicates": 1,
        "reset_separated_condition_sequences": len(sequences),
        "baseline": _summary(base_rows),
        "front_wheel_history_latent": _summary(adapted_rows),
        "per_sequence": [
            {"baseline": base, "front_wheel_history_latent": adapted}
            for base, adapted in zip(base_rows, adapted_rows)
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(args.checkpoint, args.dataset, args.run_id,
                      args.output)
    print(json.dumps({
        "output": str(args.output),
        "sequences": report["reset_separated_condition_sequences"],
        "baseline": report["baseline"],
        "front_wheel_history_latent": report["front_wheel_history_latent"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
