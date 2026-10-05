#!/usr/bin/env python3
"""Fit and validate the independent command-to-feedback actuator block."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from tools.vehicle_dynamics_learning.actuator_model import (
    fit_actuator_model,
    rollout_channel,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = (
    ROOT / "live_runs/derived_dynamics_learning_20260928/subnet_reset_20261004/"
    "body_sysid_v1.npz")
DEFAULT_OUTPUT = (
    ROOT / "live_runs/derived_dynamics_learning_20260928/subnet_reset_20261004/"
    "actuator_model_report.json")
DEFAULT_STARTS = (
    ROOT / "live_runs/derived_dynamics_learning_20260928/subnet_reset_20261004/"
    "evaluation_starts.json")
CHANNELS = (
    ("steering", 0, "steering_feedback_rad", "steering_command_rad"),
    ("throttle", 1, "throttle_feedback_norm", "throttle_command_norm"),
)
HORIZONS = (1, 4, 10, 20, 40, 80)
DT_S = 0.025


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _cluster_ci(run_values: dict[str, float], seed: int = 20261004
                ) -> dict[str, Any]:
    values = np.asarray(list(run_values.values()), dtype=np.float64)
    if not len(values):
        return {"run_count": 0, "mean": None, "ci95": [None, None]}
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, (5000, len(values)), replace=True).mean(axis=1)
    return {
        "run_count": int(len(values)),
        "mean": float(values.mean()),
        "ci95": [float(value) for value in np.quantile(samples, (0.025, 0.975))],
        "per_run": {key: float(value) for key, value in run_values.items()},
    }


def _seq_rows(data: dict[str, Any], sequence: int) -> tuple[int, int]:
    start, end = data["sequence_bounds"][sequence]
    return int(start), int(end)


def _slew_thresholds(data: dict[str, Any], channel: int) -> list[float]:
    values: list[np.ndarray] = []
    for sequence, split in enumerate(data["sequence_split"].astype(str)):
        if split != "train":
            continue
        start, end = _seq_rows(data, sequence)
        delta = np.abs(np.diff(data["stored_commands"][start:end, channel])) / DT_S
        values.append(delta[delta > 1e-8])
    positive = np.concatenate([item for item in values if len(item)])
    q50, q90 = np.quantile(positive, (0.50, 0.90))
    return [float(q50), float(q90)]


def _slew_bucket(value: float, thresholds: list[float]) -> str:
    q50, q90 = thresholds
    if value <= 1e-8:
        return "zero"
    if value < q50:
        return "low_positive"
    if value < q90:
        return "mid_positive"
    return "high_positive"


def _one_step_slew_report(data: dict[str, Any], fit: Any) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for name, channel, _, _ in CHANNELS:
        model = getattr(fit, name)
        thresholds = _slew_thresholds(data, channel)
        bins: dict[str, dict[str, list[float]]] = {
            label: {} for label in
            ("zero", "low_positive", "mid_positive", "high_positive")}
        for sequence, split in enumerate(data["sequence_split"].astype(str)):
            if split != "validation":
                continue
            start, end = _seq_rows(data, sequence)
            run_id = str(data["sequence_run_id"][sequence])
            for row in range(start + 1, end):
                command_index = row - model.delay_steps
                if command_index < start:
                    continue
                feedback = float(data["inputs"][row - 1, channel])
                target = float(data["inputs"][row, channel])
                command = float(data["stored_commands"][command_index, channel])
                predicted = feedback + model.alpha * (command - feedback)
                slew = abs(float(data["stored_commands"][row, channel]
                                 - data["stored_commands"][row - 1, channel])) / DT_S
                bucket = _slew_bucket(slew, thresholds)
                bins[bucket].setdefault(run_id, []).append(predicted - target)
        channel_report = {"train_derived_abs_command_slew_edges_per_s": thresholds,
                          "bins": {}}
        for bucket, run_errors in bins.items():
            by_run = {run: float(np.sqrt(np.mean(np.square(errors))))
                      for run, errors in run_errors.items() if errors}
            all_errors = [error for values in run_errors.values() for error in values]
            channel_report["bins"][bucket] = {
                "transition_count": len(all_errors),
                "run_count": len(by_run),
                "one_step_rmse": (float(np.sqrt(np.mean(np.square(all_errors))))
                                  if all_errors else None),
                "macro_run_rmse": _cluster_ci(by_run),
            }
        results[name] = channel_report
    return results


def _response_event_report(data: dict[str, Any], fit: Any,
                           thresholds: dict[str, list[float]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, channel, _, _ in CHANNELS:
        model = getattr(fit, name)
        q90 = thresholds[name][1]
        events: list[dict[str, Any]] = []
        for sequence, split in enumerate(data["sequence_split"].astype(str)):
            if split != "validation":
                continue
            start, end = _seq_rows(data, sequence)
            command = data["stored_commands"][start:end, channel].astype(float)
            feedback = data["inputs"][start:end, channel].astype(float)
            for k in range(1, len(command) - 12):
                change = float(command[k] - command[k - 1])
                if abs(change) < q90 or abs(change) < 1e-8:
                    continue
                final = float(command[k])
                if np.max(np.abs(command[k:k + 12] - final)) > 0.02:
                    continue
                horizon = 10
                initial = float(feedback[k])
                amplitude = final - initial
                if abs(amplitude) < 0.02:
                    continue
                predicted = rollout_channel(
                    initial, command, k, horizon, model.delay_steps, model.alpha)
                observed = feedback[k:k + horizon + 1]

                def response_metrics(response: np.ndarray) -> dict[str, float | None]:
                    progress = (response - initial) / amplitude
                    def crossing(fraction: float) -> int | None:
                        indices = np.flatnonzero(progress >= fraction)
                        return int(indices[0]) if len(indices) else None
                    t10, t90 = crossing(0.10), crossing(0.90)
                    overshoot = max(0.0, float(np.max(progress) - 1.0))
                    return {
                        "effective_delay_s": (t10 * DT_S if t10 is not None else None),
                        "rise_or_fall_10_90_s": (
                            (t90 - t10) * DT_S
                            if t10 is not None and t90 is not None else None),
                        "overshoot_fraction": overshoot,
                    }

                events.append({
                    "run_id": str(data["sequence_run_id"][sequence]),
                    "direction": "rise" if amplitude > 0 else "fall",
                    "command_step_magnitude": abs(change),
                    "initial_feedback": initial,
                    "final_command": final,
                    "measured": response_metrics(observed),
                    "predicted": response_metrics(predicted),
                })
        summarized: dict[str, Any] = {}
        for kind in ("rise", "fall"):
            selected = [event for event in events if event["direction"] == kind]
            summarized[kind] = {
                "event_count": len(selected),
                "run_count": len({event["run_id"] for event in selected}),
                "measured": {},
                "predicted": {},
            }
            for source in ("measured", "predicted"):
                for metric in ("effective_delay_s", "rise_or_fall_10_90_s",
                               "overshoot_fraction"):
                    values = [event[source][metric] for event in selected
                              if event[source][metric] is not None]
                    summarized[kind][source][metric] = (
                        float(np.mean(values)) if values else None)
        output[name] = {
            "step_event_definition": (
                "validation command change at or above the train-set 90th "
                "percentile absolute slew, followed by 12 samples within 0.02 "
                "of the new command; response uses a 250 ms window"),
            "train_derived_slew_threshold_per_s": q90,
            "summary": summarized,
            "events": events,
        }
    return output


def evaluate(dataset_path: Path, output_path: Path,
             starts_path: Path = DEFAULT_STARTS) -> dict[str, Any]:
    dataset_path, output_path, starts_path = tuple(path.resolve() for path in (
        dataset_path, output_path, starts_path))
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    with np.load(dataset_path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    frozen_starts = json.loads(starts_path.read_text(encoding="utf-8"))
    fit, _ = fit_actuator_model(data)
    channel_models = {name: getattr(fit, name) for name, *_ in CHANNELS}
    run_metrics: dict[str, dict[str, dict[str, list[float]]]] = {}
    windows_per_run: dict[str, int] = {}
    max_horizon = max(HORIZONS)
    source_rows = np.asarray(data["source_frame_index"], dtype=np.int64)
    row_runs = np.asarray(data["run_id"]).astype(str)
    row_splits = np.asarray(data["split"]).astype(str)
    for run_id, run_starts in frozen_starts["split_roles"][
            "development_validation"].items():
        for window in run_starts:
            source_row = int(window["absolute_row"])
            matches = np.flatnonzero((source_rows == source_row)
                                     & (row_runs == str(run_id)))
            if len(matches) != 1:
                raise ValueError(
                    f"frozen validation start does not uniquely map to body dataset: "
                    f"{run_id} row {source_row}")
            global_start = int(matches[0])
            if row_splits[global_start] != "validation":
                raise ValueError("frozen start is not in the validation split")
            sequence_matches = np.flatnonzero(
                (data["sequence_bounds"][:, 0] <= global_start)
                & (global_start < data["sequence_bounds"][:, 1]))
            if len(sequence_matches) != 1:
                raise ValueError("frozen start does not map to one reset sequence")
            sequence = int(sequence_matches[0])
            start, end = _seq_rows(data, sequence)
            local_start = global_start - start
            if local_start < 1 or local_start + max_horizon >= end - start:
                raise ValueError(f"frozen validation start lacks a 2 s horizon: {run_id}")
            if str(data["sequence_run_id"][sequence]) != str(run_id):
                raise ValueError("frozen start sequence/run identity mismatch")
            windows_per_run[run_id] = windows_per_run.get(run_id, 0) + 1
            for name, channel, _, _ in CHANNELS:
                model = channel_models[name]
                command = data["stored_commands"][start:end, channel]
                initial = float(data["inputs"][global_start, channel])
                prediction = rollout_channel(
                    initial, command, local_start, max_horizon,
                    model.delay_steps, model.alpha)
                truth = data["inputs"][global_start:
                                       global_start + max_horizon + 1,
                                       channel].astype(np.float64)
                channel_metrics = run_metrics.setdefault(run_id, {}).setdefault(
                    name, {f"{horizon}_trajectory_rmse": []
                           for horizon in HORIZONS} | {
                               f"{horizon}_endpoint_abs_error": []
                               for horizon in HORIZONS})
                for horizon in HORIZONS:
                    errors = prediction[1:horizon + 1] - truth[1:horizon + 1]
                    channel_metrics[f"{horizon}_trajectory_rmse"].append(
                        float(np.sqrt(np.mean(errors ** 2))))
                    channel_metrics[f"{horizon}_endpoint_abs_error"].append(
                        abs(float(errors[-1])))

    horizon_report: dict[str, Any] = {}
    for name, *_ in CHANNELS:
        horizon_report[name] = {}
        for horizon in HORIZONS:
            horizon_report[name][f"{horizon * DT_S:.3f}s"] = {}
            for metric_suffix in ("trajectory_rmse", "endpoint_abs_error"):
                key = f"{horizon}_{metric_suffix}"
                by_run = {
                    run: float(np.mean(values[name][key]))
                    for run, values in run_metrics.items()
                    if name in values and values[name][key]
                }
                horizon_report[name][f"{horizon * DT_S:.3f}s"][
                    metric_suffix] = _cluster_ci(by_run)

    slew_thresholds = {
        name: _slew_thresholds(data, channel)
        for name, channel, _, _ in CHANNELS
    }
    report = {
        "schema_version": 1,
        "purpose": "independent actuator dynamics validation before body SUBNET fitting",
        "dataset": str(dataset_path),
        "dataset_sha256": sha256(dataset_path),
        "frozen_evaluation_starts": str(starts_path),
        "frozen_evaluation_starts_sha256": sha256(starts_path),
        "sample_period_s": DT_S,
        "fit_method": (
            "training-only hierarchical least-squares fit of y[k+1] = "
            "y[k] + alpha * (command[k-delay] - y[k]); delay in {0,1}"),
        "fit": fit.diagnostics,
        "free_running_validation": {
            "initial_state": "measured actuator feedback at each frozen sequence sample",
            "future_inputs": "known recorded commands only; no future feedback",
            "start_policy": "exact starts from frozen evaluation_starts.json",
            "horizons": horizon_report,
            "windows_per_independent_run": windows_per_run,
            "independent_validation_runs": len(windows_per_run),
        },
        "one_step_error_by_command_slew": _one_step_slew_report(data, fit),
        "command_step_response": _response_event_report(
            data, fit, slew_thresholds),
        "selection_uses_validation": False,
        "test_or_final_test_loaded": False,
        "split_policy": "actuator coefficients fit on train; reported primary accuracy on validation",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--starts", type=Path, default=DEFAULT_STARTS)
    args = parser.parse_args()
    report = evaluate(args.dataset, args.output, args.starts)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "fit": {name: {
            "delay_steps": report["fit"][name]["selected_delay_steps"],
            "alpha": report["fit"][name]["alpha"],
        } for name, *_ in CHANNELS},
        "validation_independent_runs": report["free_running_validation"][
            "independent_validation_runs"],
        "windows_per_run": report["free_running_validation"][
            "windows_per_independent_run"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
