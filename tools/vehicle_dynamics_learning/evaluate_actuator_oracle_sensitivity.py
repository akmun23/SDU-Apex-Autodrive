#!/usr/bin/env python3
"""Measure actuator-model importance by a matched future-feedback oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from tools.vehicle_dynamics_learning import intervene_effective_race_teacher as wp17
from tools.vehicle_dynamics_learning.effective_race_teacher import (
    append_roll_state,
    physical_state_from_dataset,
)
from tools.vehicle_dynamics_learning.score_effective_race_teacher import (
    _load_model,
    _window_batch,
)
from tools.vehicle_dynamics_learning.train_nssm import _load_dataset


ROOT = Path(__file__).resolve().parents[2]
RESET_ROOT = ROOT / "live_runs/derived_dynamics_learning_20260928/subnet_reset_20261004"
DEFAULT_STARTS = RESET_ROOT / "evaluation_starts.json"
DEFAULT_OUTPUT = RESET_ROOT / "actuator_oracle_sensitivity.json"
DEFAULT_CHECKPOINT = wp17.DEFAULT_CHECKPOINT
DEFAULT_DYNAMIC = wp17.DEFAULT_DYNAMIC_DATASET
DEFAULT_FIXED = wp17.DEFAULT_DYNAMIC_FIXED
HORIZONS = {"0.25s": 10, "0.75s": 30, "2s": 80, "5s": 200, "10s": 400}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _frozen_windows(data: dict[str, Any], starts: dict[str, Any],
                    history_steps: int, horizon: int) -> list[dict[str, int]]:
    run_ids = np.asarray(data["run_ids"]).astype(str)
    splits = np.asarray(data["splits"]).astype(str)
    run_lookup = {run: index for index, run in enumerate(run_ids)}
    result = []
    for run_id, rows in starts["split_roles"]["development_validation"].items():
        if run_id not in run_lookup:
            raise ValueError(f"frozen run is absent from parent dataset: {run_id}")
        run_index = run_lookup[run_id]
        if splits[run_index] != "validation":
            raise ValueError(f"frozen start is not a validation run: {run_id}")
        for item in rows:
            start = int(item["absolute_row"])
            sequence_start, sequence_end = wp17._window_sequence_bounds(
                data, run_index, start)
            if (item.get("run_id") != run_id
                    or int(item["sequence_index"]) != int(np.flatnonzero(
                        (data["bounds"][:, 0] == sequence_start)
                        & (data["bounds"][:, 1] == sequence_end))[0])
                    or start - history_steps + 1 < sequence_start
                    or start + horizon >= sequence_end):
                raise ValueError(f"frozen start fails sequence/horizon contract: {run_id}/{start}")
            result.append({"run": run_index, "start": start})
    if not result:
        raise ValueError("frozen evaluation start manifest is empty")
    return result


def evaluate(checkpoint: Path, dynamic_path: Path, fixed_path: Path,
             starts_path: Path, output_path: Path, device: str) -> dict[str, Any]:
    checkpoint, dynamic_path, fixed_path, starts_path, output_path = tuple(
        path.resolve() for path in
        (checkpoint, dynamic_path, fixed_path, starts_path, output_path))
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    expected_checkpoint = wp17.EXPECTED_CHECKPOINT_SHA256
    if sha256(checkpoint) != expected_checkpoint:
        raise ValueError("actuator oracle must use the frozen WP17 parent checkpoint")

    torch, model, metadata = _load_model(checkpoint, device)
    if set(metadata.get("training_runs", [])) & set(
            json.loads(starts_path.read_text(encoding="utf-8"))[
                "split_roles"]["development_validation"]):
        raise ValueError("frozen validation runs overlap the parent model's training runs")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    dynamic = _load_dataset(dynamic_path)
    fixed = _load_dataset(fixed_path)
    wp17._check_same_base_rows(dynamic, fixed, dynamic_path, fixed_path)
    starts = json.loads(starts_path.read_text(encoding="utf-8"))
    if set(starts["split_roles"]["development_validation"]) & set(
            metadata.get("training_runs", [])):
        raise ValueError("frozen validation runs overlap the parent model training set")

    state = physical_state_from_dataset(
        dynamic, str(metadata.get("wheel_state_source", "filtered_odometry")))
    if model.include_roll_state:
        state = append_roll_state(dynamic, state)
    windows = _frozen_windows(
        dynamic, starts, model.history_state_size, max(HORIZONS.values()))

    # Confirm the normal wrapper is numerically identical to the model's own
    # command-only rollout before applying the noncausal actuator oracle.
    probe = windows[:2]
    raw_history = wp17._history_features(model, dynamic)
    history, initial, delayed, commands, _, _ = _window_batch(
        dynamic, state, probe, model.history_state_size, raw_history,
        horizon_steps=10)
    with torch.no_grad():
        direct = model.rollout(
            torch.as_tensor(initial, dtype=torch.float32, device=device),
            torch.as_tensor(delayed, dtype=torch.float32, device=device),
            torch.as_tensor(history, dtype=torch.float32, device=device),
            torch.as_tensor(commands, dtype=torch.float32, device=device))[0]
        wrapped = wp17._simulate(
            model, torch, dynamic, state, probe, 10, "I0", None, device)
    if not np.allclose(wrapped["states"], direct.cpu().numpy(),
                       rtol=1e-6, atol=1e-6):
        raise ValueError("normal wrapper differs from frozen parent command-only rollout")

    horizons_report: dict[str, Any] = {}
    for label, horizon in HORIZONS.items():
        cohort = _frozen_windows(dynamic, starts, model.history_state_size,
                                 horizon)
        normal = wp17._simulate(
            model, torch, dynamic, state, cohort, horizon, "I0", None, device)
        oracle = wp17._simulate(
            model, torch, dynamic, state, cohort, horizon, "I5", fixed, device)
        metrics_normal = wp17._metric_arrays(normal, fixed)
        metrics_oracle = wp17._metric_arrays(oracle, fixed)
        horizons_report[label] = {
            "steps": horizon,
            "windows": len(cohort),
            "independent_runs": len({row["run"] for row in cohort}),
            "I0_command_only": wp17._run_horizon_metrics(
                normal, metrics_normal, horizon, dynamic),
            "I5_future_measured_actuator_feedback": wp17._run_horizon_metrics(
                oracle, metrics_oracle, horizon, dynamic),
            "paired_I5_minus_I0": wp17._paired_deltas(
                normal, oracle, fixed, horizon, dynamic,
                oracle_clamped_channels=()),
        }

    report = {
        "schema_version": 1,
        "purpose": "diagnostic causal-importance test of the command-to-actuator block",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "dynamic_dataset": str(dynamic_path),
        "dynamic_dataset_sha256": sha256(dynamic_path),
        "fixed_rate_dataset": str(fixed_path),
        "fixed_rate_dataset_sha256": sha256(fixed_path),
        "frozen_evaluation_starts": str(starts_path),
        "frozen_evaluation_starts_sha256": sha256(starts_path),
        "future_truth_or_feedback_used": True,
        "oracle_is_diagnostic_only": True,
        "I0": "command-only recursive rollout with internally predicted actuator feedback",
        "I5": (
            "same initial states, history, and commands; measured future steering/throttle "
            "feedback replaces predicted actuator state at every transition"),
        "primary_error_channels": ["u_mps", "v_mps", "yaw_rate_rps",
                                   "heading_rad", "radial_position_m"],
        "horizons": horizons_report,
        "training_performed": False,
        "test_or_final_test_scored": False,
        "decision_threshold": (
            "If paired body/pose changes remain consistently below 10% across "
            "independent runs, actuator prediction is not the primary bottleneck."),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dynamic-dataset", type=Path, default=DEFAULT_DYNAMIC)
    parser.add_argument("--fixed-rate-dataset", type=Path, default=DEFAULT_FIXED)
    parser.add_argument("--starts", type=Path, default=DEFAULT_STARTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    report = evaluate(args.checkpoint, args.dynamic_dataset,
                      args.fixed_rate_dataset, args.starts, args.output,
                      args.device)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "checkpoint_sha256": report["checkpoint_sha256"],
        "horizons": {
            name: {
                metric: value["paired_I5_minus_I0"]["channels"][metric][
                    "paired_run_relative_rmse_improvement_fraction"]
                for metric in ("radial_position_m", "heading_rad", "u_mps",
                               "v_mps", "yaw_rate_rps")
            }
            for name, value in report["horizons"].items()
        },
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
