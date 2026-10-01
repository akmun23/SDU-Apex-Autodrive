#!/usr/bin/env python3
"""Run-level bootstrap sensitivity/identifiability analysis for grey-box fits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from tools.vehicle_dynamics_learning.four_wheel_greybox import (
    DT_S,
    WHEEL_RADIUS_M,
    _groups,
    _initial_state,
    _physical_model,
    _targets,
)
from tools.vehicle_dynamics_learning.train_nssm import _load_dataset, _torch


def _windows(data: dict[str, Any], physical_state: np.ndarray,
             acceleration: np.ndarray, horizon_steps: int,
             seed: int, max_runs: int) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    groups = _groups(data, physical_state, acceleration, "train", horizon_steps)
    rng = np.random.default_rng(seed)
    run_ids = sorted(groups)
    if len(run_ids) > max_runs:
        run_ids = sorted(int(x) for x in rng.choice(
            run_ids, size=max_runs, replace=False))
    windows = []
    for run in run_ids:
        sequences = groups[run]
        sequence_start, sequence_end = max(
            sequences, key=lambda item: item[1] - item[0])
        low = sequence_start
        high = sequence_end - horizon_steps - 1
        if high < low:
            continue
        index = int(rng.integers(low, high + 1))
        windows.append({
            "run_index": int(run),
            "run_id": str(data["run_ids"][run]),
            "sequence_start": sequence_start,
            "start_index": index,
            "initial": _initial_state(
                data, index, physical_state, sequence_start),
            "commands": data["frames"][index:index + horizon_steps, 7:9].copy(),
        })
    train_indices = np.concatenate([
        np.arange(start, end - 1)
        for sequences in groups.values()
        for start, end in sequences
    ])
    feedback = data["frames"][train_indices, 3:5]
    state_scale = np.maximum(
        np.std(np.column_stack((physical_state[train_indices], feedback)), axis=0),
        [0.5, 0.25, 0.25, 0.5, 0.5, 0.1, 0.2]).astype(np.float32)
    accel_scale = np.maximum(np.nanstd(acceleration[train_indices], axis=0),
                              [0.5, 0.5, 0.5]).astype(np.float32)
    if len(windows) < 8:
        raise ValueError(f"identifiability analysis has only {len(windows)} run windows")
    return windows, state_scale, accel_scale


def _normalized_outputs(torch, model, initials, commands,
                       state_scale: np.ndarray,
                       acceleration_scale: np.ndarray):
    with torch.no_grad():
        prediction, acceleration = model.rollout(initials, commands)
    body_and_wheels = torch.cat((
        prediction[:, :, :3], prediction[:, :, 5:7] * WHEEL_RADIUS_M,
        prediction[:, :, 17:19]), dim=-1)
    scales = torch.as_tensor(
        np.concatenate((state_scale, acceleration_scale)),
        dtype=body_and_wheels.dtype, device=body_and_wheels.device)
    output = torch.cat((body_and_wheels, acceleration), dim=-1) / scales
    return output.detach().cpu().numpy().astype(np.float64)


def analyze(checkpoints: list[Path], dataset_path: Path, output_path: Path,
            *, horizon_seconds: float = 2.0, max_runs: int = 32,
            bootstrap_replicates: int = 2_000, seed: int = 20261031,
            finite_difference_step: float = 0.01) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    if not 0.25 <= horizon_seconds <= 2.0:
        raise ValueError("sensitivity horizon must be in [0.25, 2.0] seconds")
    torch, nn = _torch()
    torch.set_num_threads(1)
    data = _load_dataset(dataset_path)
    physical_state, acceleration = _targets(data)
    horizon_steps = round(horizon_seconds / DT_S)
    windows, state_scale, acceleration_scale = _windows(
        data, physical_state, acceleration, horizon_steps, seed, max_runs)
    initials = torch.as_tensor(
        np.stack([row["initial"] for row in windows]), dtype=torch.float32)
    commands = torch.as_tensor(
        np.stack([row["commands"] for row in windows]), dtype=torch.float32)
    run_names = [row["run_id"] for row in windows]
    run_unique = sorted(set(run_names))
    run_row_indices = {name: np.flatnonzero(np.asarray(run_names) == name)
                       for name in run_unique}
    comparison_rows = []

    for checkpoint in checkpoints:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        metadata = payload["metadata"]
        if metadata["feature_names"] != data["feature_names"]:
            raise ValueError(f"feature mismatch in {checkpoint}")
        model_type = _physical_model(
            torch, nn, bool(metadata["tire_relaxation_state_enabled"]))
        model = model_type()
        model.load_state_dict(payload["state_dict"])
        model.eval()
        raw = model.raw_parameters.detach().clone()
        base_outputs = _normalized_outputs(
            torch, model, initials, commands, state_scale, acceleration_scale)
        jacobian_columns = []
        for parameter_index in range(len(model.PARAMETER_BOUNDS)):
            with torch.no_grad():
                model.raw_parameters.copy_(raw)
                model.raw_parameters[parameter_index] += finite_difference_step
            plus = _normalized_outputs(
                torch, model, initials, commands, state_scale, acceleration_scale)
            with torch.no_grad():
                model.raw_parameters.copy_(raw)
                model.raw_parameters[parameter_index] -= finite_difference_step
            minus = _normalized_outputs(
                torch, model, initials, commands, state_scale, acceleration_scale)
            jacobian_columns.append((plus - minus) / (2.0 * finite_difference_step))
        with torch.no_grad():
            model.raw_parameters.copy_(raw)
        jacobian = np.stack(jacobian_columns, axis=-1)
        if not np.isfinite(base_outputs).all() or not np.isfinite(jacobian).all():
            raise FloatingPointError(
                f"non-finite model outputs or sensitivities for {checkpoint}")

        # Each independent run contributes equally. Within-run time samples
        # remain correlated and are never treated as bootstrap units.
        per_run_jacobian = {
            run: jacobian[indices].reshape(-1, jacobian.shape[-1])
            for run, indices in run_row_indices.items()
        }
        parameter_names = list(model.PARAMETER_BOUNDS)
        per_run_norms = np.stack([
            np.sqrt(np.mean(per_run_jacobian[run] ** 2, axis=0))
            for run in run_unique
        ])
        pooled = np.concatenate([per_run_jacobian[run] for run in run_unique])
        column_norm = np.linalg.norm(pooled, axis=0)
        nonzero = column_norm > 1.0e-12
        normalized_j = np.zeros_like(pooled)
        normalized_j[:, nonzero] = pooled[:, nonzero] / column_norm[nonzero]
        gram = normalized_j.T @ normalized_j
        correlations = np.zeros_like(gram)
        gram_diag = np.sqrt(np.maximum(np.diag(gram), 1.0e-24))
        correlations = gram / np.outer(gram_diag, gram_diag)
        singular = np.linalg.svd(normalized_j, compute_uv=False)
        rank_cutoff = float(singular[0] * 1.0e-3) if len(singular) else 0.0
        effective_rank = int(np.count_nonzero(singular >= rank_cutoff))
        high_corr = []
        for left in range(len(parameter_names)):
            for right in range(left + 1, len(parameter_names)):
                value = float(correlations[left, right])
                if abs(value) >= 0.95:
                    high_corr.append({
                        "parameter_a": parameter_names[left],
                        "parameter_b": parameter_names[right],
                        "sensitivity_correlation": value,
                    })

        rng = np.random.default_rng(seed + len(comparison_rows) * 17)
        bootstrap = np.empty((bootstrap_replicates, len(parameter_names)),
                             dtype=np.float64)
        for replicate in range(bootstrap_replicates):
            sampled = rng.integers(0, len(run_unique), len(run_unique))
            bootstrap[replicate] = np.mean(per_run_norms[sampled], axis=0)
        parameters = model.export_parameters()
        bound_fraction = {
            name: float((parameters[name] - low) / (high - low))
            for name, (low, high) in model.PARAMETER_BOUNDS.items()
        }
        comparison_rows.append({
            "checkpoint": str(checkpoint.resolve()),
            "architecture": metadata["architecture"],
            "tire_relaxation_state_enabled": bool(
                metadata["tire_relaxation_state_enabled"]),
            "parameter_values": parameters,
            "parameter_fraction_of_bound_interval": bound_fraction,
            "finite_difference_step_in_bounded_raw_parameter": finite_difference_step,
            "run_balanced_sensitivity_rms": {
                name: {
                    "estimate": float(np.mean(per_run_norms[:, index])),
                    "run_bootstrap_95pct_interval": np.quantile(
                        bootstrap[:, index], (0.025, 0.975)).tolist(),
                    "across_run_min": float(np.min(per_run_norms[:, index])),
                    "across_run_max": float(np.max(per_run_norms[:, index])),
                } for index, name in enumerate(parameter_names)
            },
            "near_collinear_parameter_sensitivities_abs_corr_ge_0.95": high_corr,
            "normalized_sensitivity_singular_values": singular.tolist(),
            "effective_rank_at_relative_1e-3": effective_rank,
            "parameter_count": len(parameter_names),
            "minimum_to_maximum_singular_value_ratio": float(
                singular[-1] / singular[0]) if len(singular) else None,
            "independent_run_count": len(run_unique),
            "per_run_window_start_indices": {
                row["run_id"]: row["start_index"] for row in windows
            },
        })

    report = {
        "schema_version": 1,
        "dataset": str(dataset_path.resolve()),
        "split": "train only for sensitivity; whole runs are the uncertainty units",
        "horizon_seconds": horizon_steps * DT_S,
        "shooting_window_seconds": horizon_steps * DT_S,
        "sampled_run_ids": run_unique,
        "independent_run_count": len(run_unique),
        "normalization": {
            "body_and_actuator_state_scale": state_scale.tolist(),
            "effective_acceleration_scale": acceleration_scale.tolist(),
        },
        "method": "central finite-difference Jacobian of normalized command-driven grey-box outputs wrt bounded parameter logits; per-run blocked bootstrap",
        "warning": "local predictive identifiability only; it is not a parameter confidence interval, and a deficient rank means the fitted physical parameters must not be interpreted separately",
        "models": comparison_rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("checkpoints", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon-seconds", type=float, default=2.0)
    parser.add_argument("--max-runs", type=int, default=32)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261031)
    args = parser.parse_args()
    report = analyze(
        args.checkpoints, args.dataset, args.output,
        horizon_seconds=args.horizon_seconds, max_runs=args.max_runs,
        bootstrap_replicates=args.bootstrap_replicates, seed=args.seed)
    print(json.dumps({
        "models": [{
            "checkpoint": item["checkpoint"],
            "effective_rank": item["effective_rank_at_relative_1e-3"],
            "parameter_count": item["parameter_count"],
            "correlated_pairs": len(item[
                "near_collinear_parameter_sensitivities_abs_corr_ge_0.95"]),
        } for item in report["models"]],
        "runs": report["sampled_run_ids"],
        "output": str(args.output.resolve()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
