#!/usr/bin/env python3
"""Audit independent excitation and identifiability in the frozen body data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[2]
RESET_ROOT = ROOT / "live_runs/derived_dynamics_learning_20260928/subnet_reset_20261004"
DATASET = RESET_ROOT / "body_sysid_v1.npz"
OUTPUT = RESET_ROOT / "body_sysid_excitation_audit_v1.json"
DT_S = 0.025
SPEED_EDGES = (0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0)
STEER_CLASSES = (("near_straight", 0.0, 0.10),
                 ("moderate", 0.10, 0.30),
                 ("high", 0.30, 0.525))
FUTURE_STEPS = 20  # 0.5 s: distinguish state-matched outcomes, not long free-run drift.
MAX_NEIGHBOR_QUERIES = 256
NEIGHBOR_COUNT = 5
SEED = 31004


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _range(values: np.ndarray) -> list[float] | None:
    finite = values[np.isfinite(values)]
    return [float(finite.min()), float(finite.max())] if len(finite) else None


def _nearest_future_spread(indices: np.ndarray, run_ids: np.ndarray,
                           features: np.ndarray, future_outputs: np.ndarray,
                           rng: np.random.Generator) -> dict[str, Any]:
    eligible = indices[np.isfinite(future_outputs[indices]).all(axis=1)]
    if len(eligible) < 2:
        return {"query_count": 0, "cross_run_neighbor_count": 0,
                "rmse_spread_by_channel": None}
    queries = rng.choice(eligible, size=min(MAX_NEIGHBOR_QUERIES, len(eligible)),
                         replace=False)
    tree = cKDTree(features[eligible])
    k = min(32, len(eligible))
    neighbor_std = []
    cross_run_count = 0
    for query in queries:
        _, positions = tree.query(features[query], k=k)
        positions = np.atleast_1d(positions)
        candidates = [int(eligible[pos]) for pos in positions
                      if int(eligible[pos]) != int(query)
                      and run_ids[eligible[pos]] != run_ids[query]]
        if len(candidates) < 2:
            continue
        selected = candidates[:NEIGHBOR_COUNT]
        if len(selected) < 2:
            continue
        neighbor_std.append(np.std(future_outputs[selected], axis=0, ddof=0))
        cross_run_count += len(selected)
    if not neighbor_std:
        return {"query_count": len(queries), "cross_run_neighbor_count": 0,
                "rmse_spread_by_channel": None}
    spread = np.mean(np.stack(neighbor_std), axis=0)
    return {"query_count": int(len(queries)),
            "cross_run_neighbor_count": int(cross_run_count),
            "rmse_spread_by_channel": spread.astype(float).tolist()}


def _describe_group(indices: np.ndarray, *, split: str, speed_bin: str,
                    steering_class: str, run_ids: np.ndarray,
                    outputs: np.ndarray, inputs: np.ndarray, commands: np.ndarray,
                    steering_rate: np.ndarray, throttle_slew: np.ndarray,
                    output_derivative: np.ndarray, future_outputs: np.ndarray,
                    regression_scale: np.ndarray, nn_features: np.ndarray,
                    rng: np.random.Generator, include_neighbors: bool) -> dict[str, Any]:
    ids = np.unique(run_ids[indices])
    x = inputs[indices].astype(np.float64)
    covariance = (np.cov(x, rowvar=False, ddof=0).tolist()
                  if len(indices) > 1 else None)
    eigenvalues = None
    input_condition = None
    cross_correlation = None
    if covariance is not None:
        cov = np.asarray(covariance, dtype=np.float64)
        eigenvalues = np.linalg.eigvalsh(cov).tolist()
        if eigenvalues[0] > 1e-12:
            input_condition = float(eigenvalues[-1] / eigenvalues[0])
        if np.std(x[:, 0]) > 1e-12 and np.std(x[:, 1]) > 1e-12:
            cross_correlation = float(np.corrcoef(x.T)[0, 1])

    state_features = np.column_stack((outputs[indices], inputs[indices]))
    if len(state_features) > 10_000:
        selected = rng.choice(len(state_features), 10_000, replace=False)
        state_features = state_features[selected]
    design = np.column_stack((np.ones(len(state_features)),
                              state_features / regression_scale))
    rank = int(np.linalg.matrix_rank(design)) if len(design) else 0
    regression_condition = (float(np.linalg.cond(design))
                           if rank == design.shape[1] else None)
    derivative_var = (np.var(output_derivative[indices], axis=0, ddof=0).tolist()
                      if len(indices) else None)

    result = {
        "split": split,
        "speed_bin_mps": speed_bin,
        "steering_class": steering_class,
        "sample_count": int(len(indices)),
        "valid_duration_s": float(len(indices) * DT_S),
        "independent_run_count": int(len(ids)),
        "run_ids": ids.tolist(),
        "speed_mps_min_max": _range(np.hypot(
            outputs[indices, 0], outputs[indices, 1])),
        "steering_feedback_rad_min_max": _range(inputs[indices, 0]),
        "steering_command_rad_min_max": _range(commands[indices, 0]),
        "throttle_feedback_min_max": _range(inputs[indices, 1]),
        "throttle_command_min_max": _range(commands[indices, 1]),
        "steering_rate_radps_min_max": _range(steering_rate[indices]),
        "throttle_command_slew_per_s_min_max": _range(throttle_slew[indices]),
        "yaw_rate_rps_min_max": _range(outputs[indices, 2]),
        "input_covariance_feedback_steer_throttle": covariance,
        "input_covariance_eigenvalues": eigenvalues,
        "input_covariance_condition_number": input_condition,
        "steering_throttle_feedback_correlation": cross_correlation,
        "local_regression_design": "intercept + standardized [u, v, yaw_rate, steering_feedback, throttle_feedback]",
        "local_regression_rank": rank,
        "local_regression_condition_number": regression_condition,
        "one_step_output_derivative_variance": derivative_var,
    }
    if include_neighbors:
        result["cross_run_0p5s_future_output_neighbor_spread"] = \
            _nearest_future_spread(indices, run_ids, nn_features, future_outputs, rng)
    return result


def analyze(dataset_path: Path = DATASET, output_path: Path = OUTPUT) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    with np.load(dataset_path, allow_pickle=False) as data:
        inputs = np.asarray(data["inputs"], dtype=np.float64)
        outputs = np.asarray(data["outputs"], dtype=np.float64)
        commands = np.asarray(data["stored_commands"], dtype=np.float64)
        bounds = np.asarray(data["sequence_bounds"], dtype=np.int64)
        run_ids = data["run_id"].astype(str)
        split = data["split"].astype(str)
        sequence_splits = data["sequence_split"].astype(str)
    if inputs.shape != (len(outputs), 2) or outputs.shape[1] != 3:
        raise ValueError("unexpected body system-identification input/output layout")
    if commands.shape != inputs.shape:
        raise ValueError("stored command channels are not aligned with feedback")
    if set(np.unique(split)) - {"train", "validation"}:
        raise ValueError("test/final-test rows are excluded from this development audit")

    steering_rate = np.full(len(inputs), np.nan)
    steering_command_rate = np.full(len(inputs), np.nan)
    throttle_slew = np.full(len(inputs), np.nan)
    output_derivative = np.full_like(outputs, np.nan)
    future_outputs = np.full_like(outputs, np.nan)
    for start_raw, end_raw in bounds:
        start, end = int(start_raw), int(end_raw)
        if end - start > FUTURE_STEPS:
            steering_rate[start + 1:end] = np.diff(inputs[start:end, 0]) / DT_S
            steering_command_rate[start + 1:end] = (
                np.diff(commands[start:end, 0]) / DT_S)
            throttle_slew[start + 1:end] = np.diff(commands[start:end, 1]) / DT_S
            output_derivative[start + 1:end] = np.diff(outputs[start:end], axis=0) / DT_S
            future_outputs[start:end - FUTURE_STEPS] = outputs[
                start + FUTURE_STEPS:end]

    speed = np.hypot(outputs[:, 0], outputs[:, 1])
    abs_steering = np.abs(inputs[:, 0])
    finite = (np.isfinite(inputs).all(axis=1) & np.isfinite(outputs).all(axis=1)
              & (speed >= 0.0) & (speed <= SPEED_EDGES[-1]))
    train = split == "train"
    state_scale = np.std(np.column_stack((outputs[train], inputs[train])), axis=0)
    state_scale[state_scale < 1e-9] = 1.0
    nn_features = np.column_stack((outputs, inputs)) / state_scale
    rng = np.random.default_rng(SEED)

    event_masks: dict[str, np.ndarray] = {
        "throttle_pickup": (commands[:, 1] > 0.05) & (throttle_slew >= 0.25),
        "throttle_release": (commands[:, 1] > -0.05) & (throttle_slew <= -0.25),
        "negative_command_braking": commands[:, 1] < -0.05,
        "braking_release": (commands[:, 1] < -0.05) & (throttle_slew >= 0.25),
        "turn_in": ((np.abs(commands[:, 0]) >= 0.10)
                    & (np.abs(steering_command_rate) >= 0.5)
                    & (np.abs(commands[:, 0]) > np.abs(np.r_[commands[0, 0], commands[:-1, 0]]))),
        "unwind": ((np.abs(commands[:, 0]) >= 0.10)
                   & (np.abs(steering_command_rate) >= 0.5)
                   & (np.abs(commands[:, 0]) < np.abs(np.r_[commands[0, 0], commands[:-1, 0]]))),
        "simultaneous_steering_throttle_transition":
            (np.abs(steering_command_rate) >= 0.5)
            & (np.abs(throttle_slew) >= 0.25),
    }
    # Commands/rates at reset boundaries are not transition events.
    contiguous = np.zeros(len(inputs), dtype=bool)
    for start_raw, end_raw in bounds:
        start, end = int(start_raw), int(end_raw)
        contiguous[start + 1:end] = True

    regions = []
    for split_name in ("train", "validation"):
        split_mask = split == split_name
        for speed_index, (low, high) in enumerate(zip(
                SPEED_EDGES[:-1], SPEED_EDGES[1:])):
            speed_mask = (speed >= low) & (speed < high if high < 12 else speed <= high)
            speed_label = f"{low:g}-{high:g}"
            for steer_name, steer_low, steer_high in STEER_CLASSES:
                mask = (finite & split_mask & speed_mask
                        & (abs_steering >= steer_low)
                        & (abs_steering < steer_high if steer_high < 0.525
                           else abs_steering <= steer_high))
                indices = np.flatnonzero(mask)
                regions.append(_describe_group(
                    indices, split=split_name, speed_bin=speed_label,
                    steering_class=steer_name, run_ids=run_ids, outputs=outputs,
                    inputs=inputs, commands=commands, steering_rate=steering_rate,
                    throttle_slew=throttle_slew, output_derivative=output_derivative,
                    future_outputs=future_outputs, regression_scale=state_scale,
                    nn_features=nn_features, rng=rng, include_neighbors=True))

    events = []
    for event_name, event_mask in event_masks.items():
        event_mask &= contiguous & finite
        for split_name in ("train", "validation"):
            selected_event = event_mask & (split == split_name)
            for speed_index, (low, high) in enumerate(zip(
                    SPEED_EDGES[:-1], SPEED_EDGES[1:])):
                speed_mask = (speed >= low) & (speed < high if high < 12 else speed <= high)
                for steer_name, steer_low, steer_high in STEER_CLASSES:
                    mask = (selected_event & speed_mask & (abs_steering >= steer_low)
                            & (abs_steering < steer_high if steer_high < 0.525
                               else abs_steering <= steer_high))
                    indices = np.flatnonzero(mask)
                    events.append({
                        "event": event_name,
                        **_describe_group(
                            indices, split=split_name,
                            speed_bin=f"{low:g}-{high:g}",
                            steering_class=steer_name, run_ids=run_ids,
                            outputs=outputs, inputs=inputs, commands=commands,
                            steering_rate=steering_rate, throttle_slew=throttle_slew,
                            output_derivative=output_derivative,
                            future_outputs=future_outputs,
                            regression_scale=state_scale,
                            nn_features=nn_features, rng=rng,
                            include_neighbors=False),
                    })

    report = {
        "schema_version": 1,
        "study": "body-model excitation and local identifiability audit",
        "dataset": str(dataset_path.resolve()),
        "dataset_sha256": _sha256(dataset_path),
        "sample_period_s": DT_S,
        "independent_unit": "whole simulator run; windows/rows are not independent",
        "body_model_inputs": ["steering_feedback_rad", "throttle_feedback_norm"],
        "stored_commands_used_only_for_event_labels": True,
        "speed_bin_edges_mps": list(SPEED_EDGES),
        "steering_classes_abs_rad": [
            {"name": name, "lower": low, "upper": high}
            for name, low, high in STEER_CLASSES],
        "nearest_neighbor_definition": {
            "state": "standardized [u, v, yaw_rate, steering_feedback, throttle_feedback]",
            "future_horizon_s": FUTURE_STEPS * DT_S,
            "neighbors": f"up to {NEIGHBOR_COUNT} nearest from other whole runs, same speed/steering/split region",
            "queries_per_region_max": MAX_NEIGHBOR_QUERIES,
            "interpretation": "local dispersion in measured future body state, not an error bound",
        },
        "regions": regions,
        "events": events,
        "limitations": [
            "The fixed development validation-start cohort contains sparse 7-9 m/s and high-steer starts and none above 9 m/s; this audit reports dataset coverage, not new independent confirmation.",
            "Input covariance and local regression conditioning diagnose collinearity; they do not prove a unique nonlinear model is identifiable.",
            "Cross-run nearest-neighbor future spread can include unmeasured history/state effects and should be interpreted alongside run counts and event coverage.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    report = analyze(args.dataset, args.output)
    sparse = [row for row in report["regions"]
              if row["steering_class"] == "high" and row["speed_bin_mps"] in
              {"6-8", "8-10", "10-12"}]
    print(json.dumps({
        "output": str(args.output.resolve()),
        "regions": len(report["regions"]),
        "event_regions": len(report["events"]),
        "high_steer_6_12mps": [{
            "split": row["split"], "speed_bin_mps": row["speed_bin_mps"],
            "samples": row["sample_count"],
            "independent_run_count": row["independent_run_count"],
            "steering_rad": row["steering_feedback_rad_min_max"],
            "input_covariance_eigenvalues": row["input_covariance_eigenvalues"],
            "future_spread": row["cross_run_0p5s_future_output_neighbor_spread"],
        } for row in sparse],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
