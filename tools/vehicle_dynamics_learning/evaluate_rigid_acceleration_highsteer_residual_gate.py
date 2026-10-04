#!/usr/bin/env python3
"""Evaluate a data-localized high-steering residual against the frozen plant.

The specialist is the WP27 high-steering warm-start arm. Its correction is
blended into the frozen acceleration plant only near the measured 7.5 m/s,
large-steering training regime. This is an offline diagnostic; it does not
change the production controller or odometry.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch import nn

from tools.vehicle_dynamics_learning.evaluate_history_context_highsteer import (
    _capture_from_archive,
    _sample_all_valid_context_refs,
)
from tools.vehicle_dynamics_learning.run_history_context_sufficiency import (
    CONTEXT_STEPS,
    OUTPUT_ROOT as WP28_ROOT,
    _bootstrap_delta,
    _eval_metrics,
    _normalization,
    _training_windows_and_stats,
)
from tools.vehicle_dynamics_learning.run_rigid_acceleration_highsteer_transfer import (
    BASE_ACCELERATION_CHECKPOINT,
    _load_acceleration_model,
)
from tools.vehicle_dynamics_learning.run_truncated_history_transition import (
    FROZEN_EVAL_STARTS,
    ROOT,
)
from tools.vehicle_dynamics_learning.train_augmented_state_space_plant import (
    DEFAULT_DYNAMIC,
    DEFAULT_DYNAMIC_FIXED,
    DEFAULT_PRACTICE,
    DEFAULT_PRACTICE_FIXED,
    _collect_horizon_windows,
    _load_data,
    _select_eval_windows,
    sha256_file,
)


SPECIALIST_CHECKPOINT = (
    WP28_ROOT / "wp27_warmstarted_highsteer_v1/checkpoint.pt")
SPECIALIST_REPORT = (
    WP28_ROOT
    / "wp27_warmstarted_highsteer_v1/wp27_highsteer_transfer_report.json")
HIGHSTEER_VALIDATION_SOURCE = (
    ROOT / "live_runs/derived_dynamics_learning_20260928"
    / "full_modeling_reset_20261001"
    / "highsteer_75_heldout_validation_plus_r03_20261004"
    / "openplane_dynamics.npz")
HIGHSTEER_VALIDATION_RUNS = {
    "openplane_highsteer_75_validation_r01",
    "openplane_highsteer_75_validation_r02",
    "openplane_highsteer_75_validation_r03_20261004",
}
OUTPUT_PATH = (WP28_ROOT
               / "rigid_acceleration_highsteer_residual_gate_10s_20261004.json")
EVAL_HORIZONS = (20, 80, 200, 400)
SCORED_METRICS = (
    "position_radial_trajectory_rmse_m",
    "position_endpoint_error_m",
    "heading_trajectory_rmse_rad",
    "heading_endpoint_error_rad",
    "u_rmse_mps",
    "v_rmse_mps",
    "yaw_rate_rmse_rps",
)


def _smoothstep(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp(0.0, 1.0)
    return value.square() * (3.0 - 2.0 * value)


class HighSteerResidualGate(nn.Module):
    """Blend frozen and high-steering models only in measured joint support."""

    def __init__(self, baseline: nn.Module, specialist: nn.Module) -> None:
        super().__init__()
        self.baseline = baseline
        self.specialist = specialist

    @staticmethod
    def activation(state: torch.Tensor,
                   state_mean: torch.Tensor,
                   state_scale: torch.Tensor) -> torch.Tensor:
        physical = state * state_scale + state_mean
        speed = physical[:, 0]
        steering = physical[:, 3].abs()
        speed_in = _smoothstep((speed - 6.8) / 0.4)
        speed_out = 1.0 - _smoothstep((speed - 7.7) / 0.5)
        steering_gate = _smoothstep((steering - 0.30) / 0.15)
        return (speed_in * speed_out * steering_gate).clamp(0.0, 1.0)

    def forward(self, history: torch.Tensor, mask: torch.Tensor,
                state: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
        baseline_delta = self.baseline(history, mask, state, command)
        specialist_delta = self.specialist(history, mask, state, command)
        activation = self.activation(
            state, self.baseline.state_mean, self.baseline.state_scale)
        return baseline_delta + activation.unsqueeze(-1) * (
            specialist_delta - baseline_delta)


def _sample_refs(refs: dict[str, list[tuple[int, int, int]]],
                 per_run: int = 128) -> dict[str, list[tuple[int, int, int]]]:
    sampled = {}
    for run_id, rows in sorted(refs.items()):
        if len(rows) <= per_run:
            sampled[run_id] = rows
        else:
            indexes = np.linspace(0, len(rows) - 1, per_run, dtype=np.int64)
            sampled[run_id] = [rows[int(index)] for index in indexes]
    if not sampled or any(not rows for rows in sampled.values()):
        raise RuntimeError("no valid held-out high-steering evaluation starts")
    return sampled


def _coalesce_contiguous_capture_sequences(data, capture_index: int
                                           ) -> dict[str, Any]:
    """Join lap/phase fragments only when captured physics proves continuity."""
    capture = data.captures[capture_index]
    source = data.raw_sources[capture_index]
    conditions = np.asarray(source["sequence_condition_id"], dtype=np.int64)
    if len(conditions) != len(capture.bounds):
        raise RuntimeError("sequence-condition metadata is misaligned")

    groups: list[dict[str, int]] = []
    decisions = []
    for index, ((begin_raw, end_raw), run_raw, reset_raw, condition_raw) in enumerate(
            zip(capture.bounds, capture.sequence_run, capture.sequence_reset,
                conditions)):
        begin, end = int(begin_raw), int(end_raw)
        run_index, reset_id = int(run_raw), int(reset_raw)
        condition_id = int(condition_raw)
        if not groups:
            groups.append({"begin": begin, "end": end, "run": run_index,
                           "reset": reset_id, "condition": condition_id,
                           "count": 1})
            continue

        previous = groups[-1]
        left = previous["end"] - 1
        right = begin
        same_run = previous["run"] == run_index
        same_reset = previous["reset"] == reset_id
        same_condition = previous["condition"] == condition_id
        adjacent_rows = previous["end"] == begin
        consecutive_packets = (
            adjacent_rows and int(capture.packet[right])
            == int(capture.packet[left]) + 1)
        fixed_step = bool(
            adjacent_rows
            and np.isclose(capture.dt_s[left], 0.025, rtol=0.0, atol=1e-7)
            and np.isclose(capture.dt_s[right], 0.025, rtol=0.0, atol=1e-7))
        continuous_labels = bool(
            adjacent_rows
            and np.isfinite(capture.body[[left, right]]).all()
            and np.isfinite(capture.input_features[[left, right]]).all()
            and np.isfinite(data.poses[capture_index][[left, right]]).all())
        merge = bool(same_run and same_reset and same_condition and adjacent_rows
                     and consecutive_packets and fixed_step and continuous_labels)
        decision = {
            "left_sequence": index - 1,
            "right_sequence": index,
            "same_run": same_run,
            "same_reset_epoch": same_reset,
            "same_condition_id": same_condition,
            "adjacent_rows": adjacent_rows,
            "consecutive_packets": consecutive_packets,
            "fixed_25ms_step": fixed_step,
            "finite_body_input_pose_labels": continuous_labels,
            "merged": merge,
        }
        decisions.append(decision)
        if merge:
            previous["end"] = end
            previous["count"] += 1
        else:
            groups.append({"begin": begin, "end": end, "run": run_index,
                           "reset": reset_id, "condition": condition_id,
                           "count": 1})

    capture.bounds = np.asarray(
        [[group["begin"], group["end"]] for group in groups], dtype=np.int64)
    capture.sequence_run = np.asarray(
        [group["run"] for group in groups], dtype=np.int32)
    capture.sequence_reset = np.asarray(
        [group["reset"] for group in groups], dtype=np.int32)
    source["sequence_condition_id"] = np.asarray(
        [group["condition"] for group in groups], dtype=np.int64)
    return {
        "original_sequence_count": len(decisions) + 1 if decisions else len(groups),
        "coalesced_sequence_count": len(groups),
        "merged_boundary_count": sum(bool(row["merged"]) for row in decisions),
        "boundary_decisions": decisions,
        "coalesced_sample_count_by_run": {
            str(capture.run_ids[group["run"]]): int(group["end"] - group["begin"])
            for group in groups
        },
        "rule": (
            "merge only adjacent fragments of the same run, reset epoch and "
            "condition with consecutive packet IDs, fixed 25ms cadence, and "
            "finite body/input/pose labels across the join"),
    }


def _refs_for_horizon(data, split: str, capture_index: int,
                      expected_runs: set[str], horizon: int,
                      per_run: int = 64):
    by_run = _collect_horizon_windows(
        data.captures, data.raw_sources, data.poses, {split}, horizon,
        capture_indices={capture_index})
    refs = _select_eval_windows(by_run, per_run)
    if set(refs) != expected_runs or any(not rows for rows in refs.values()):
        raise RuntimeError(
            f"{horizon}-step {split} run roster/coverage mismatch: "
            f"expected={sorted(expected_runs)} actual={sorted(refs)}")
    return refs


def _score_group(data, refs, models, norm_np, config, device):
    per_model: dict[str, Any] = {}
    for model_name, model in models.items():
        model.eval()
        per_model[model_name] = {
            str(horizon): _eval_metrics(
                data, refs, model, norm_np, config,
                CONTEXT_STEPS["2.0s"], device, horizon)
            for horizon in EVAL_HORIZONS
        }
    paired = {}
    for horizon in map(str, EVAL_HORIZONS):
        paired[horizon] = {}
        for metric_index, metric in enumerate(SCORED_METRICS):
            paired[horizon][metric] = {
                "gate_minus_frozen": _bootstrap_delta(
                    per_model["highsteer_residual_gate"][horizon],
                    per_model["frozen_baseline"][horizon], metric,
                    seed=20261004 + int(horizon) + metric_index),
                "specialist_minus_frozen": _bootstrap_delta(
                    per_model["full_highsteer_specialist"][horizon],
                    per_model["frozen_baseline"][horizon], metric,
                    seed=20261104 + int(horizon) + metric_index),
            }
    macro = {
        model_name: {
            horizon: {
                metric: float(np.mean([
                    values[metric] for values in per_run.values()]))
                for metric in SCORED_METRICS
            }
            for horizon, per_run in model_results.items()
        }
        for model_name, model_results in per_model.items()
    }
    return {"run_macro_metrics": per_model, "macro_across_runs": macro,
            "paired_run_bootstrap": paired,
            "independent_run_count": len(refs),
            "starts_per_run": {run_id: len(rows)
                               for run_id, rows in refs.items()}}


def run(device_name: str = "cpu", output_path: Path = OUTPUT_PATH
        ) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(1)

    data, wp20 = _load_data()
    config, _, _ = _training_windows_and_stats(data, wp20)
    norm_np = _normalization(data, config)
    baseline, baseline_hash = _load_acceleration_model(
        BASE_ACCELERATION_CHECKPOINT, norm_np, device)
    specialist, specialist_hash = _load_acceleration_model(
        SPECIALIST_CHECKPOINT, norm_np, device)
    gate = HighSteerResidualGate(baseline, specialist).to(device)

    training_report = json.loads(SPECIALIST_REPORT.read_text(encoding="utf-8"))
    expected_hash = training_report.get("training_policy", {}).get(
        "initial_checkpoint_sha256")
    if expected_hash != baseline_hash:
        raise RuntimeError("specialist was not initialized from frozen baseline")

    frozen = json.loads(FROZEN_EVAL_STARTS.read_text(encoding="utf-8"))
    expected_validation = set(frozen["split_roles"]["development_validation"])
    expected_practice = set(frozen["split_roles"]["practice_diagnostic"])
    validation_refs = _refs_for_horizon(
        data, "validation", 0, expected_validation, horizon=400)
    practice_coalescing = _coalesce_contiguous_capture_sequences(data, 1)
    practice_refs = _refs_for_horizon(
        data, "unseen_practice", 1, expected_practice, horizon=400)
    validation_group = _score_group(
        data, validation_refs,
        {"frozen_baseline": baseline,
         "full_highsteer_specialist": specialist,
         "highsteer_residual_gate": gate}, norm_np, config, device)
    practice_group = _score_group(
        data, practice_refs,
        {"frozen_baseline": baseline,
         "full_highsteer_specialist": specialist,
         "highsteer_residual_gate": gate}, norm_np, config, device)

    high_capture, high_pose = _capture_from_archive(
        HIGHSTEER_VALIDATION_SOURCE, HIGHSTEER_VALIDATION_RUNS)
    high_refs = _sample_refs(_sample_all_valid_context_refs(
        high_capture, 400, history_steps=CONTEXT_STEPS["2.0s"],
        expected_runs=HIGHSTEER_VALIDATION_RUNS))
    high_data = SimpleNamespace(captures=[high_capture], poses=[high_pose])
    high_group = _score_group(
        high_data, high_refs,
        {"frozen_baseline": baseline,
         "full_highsteer_specialist": specialist,
         "highsteer_residual_gate": gate}, norm_np, config, device)

    report = {
        "study": "joint speed-steering gated residual from matched high-steer arm",
        "frozen_baseline_checkpoint_sha256": baseline_hash,
        "highsteer_specialist_checkpoint_sha256": specialist_hash,
        "dynamic_data_sha256": sha256_file(DEFAULT_DYNAMIC),
        "dynamic_fixed40hz_data_sha256": sha256_file(DEFAULT_DYNAMIC_FIXED),
        "practice_data_sha256": sha256_file(DEFAULT_PRACTICE),
        "practice_fixed40hz_data_sha256": sha256_file(DEFAULT_PRACTICE_FIXED),
        "highsteer_data_sha256": sha256_file(HIGHSTEER_VALIDATION_SOURCE),
        "control_and_data_rate_hz": 40.0,
        "gate": {
            "formula": (
                "smoothstep(speed 6.8->7.2) * "
                "(1-smoothstep(speed 7.7->8.2)) * "
                "smoothstep(abs(steering_feedback) 0.30->0.45 rad)"),
            "activation_inputs": "current predicted speed and steering feedback only",
            "purpose": "localize the learned residual to added 7.5m/s high-steering support",
            "learned": False,
        },
        "horizons_steps": list(EVAL_HORIZONS),
        "development_validation": validation_group,
        "practice_diagnostic": practice_group,
        "practice_contiguous_laps": practice_coalescing,
        "highsteer_development": high_group,
        "validation_status": {
            "all_three_highsteer_runs_have_prior_candidate_results": True,
            "results_are_development_not_untouched_confirmation": True,
            "untouched_final_confirmation_available": False,
            "test_or_final_test_opened": False,
        },
        "simulator_launched": False,
        "production_integration": False,
        "interpretation_limit": (
            "This evaluates 10 s free-recursive prediction, including practice "
            "windows formed by joining lap fragments only at consecutive, "
            "same-reset, fixed-cadence finite-state boundaries. The practice and "
            "high-steering captures were previously used as development data; "
            "this is not untouched confirmation."),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n",
                           encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    report = run(args.device, output)
    key_metrics = (
        "position_radial_trajectory_rmse_m", "u_rmse_mps",
        "v_rmse_mps", "yaw_rate_rmse_rps")
    print(json.dumps({
        "output": output.relative_to(ROOT).as_posix(),
        "gate_sha256": report["highsteer_specialist_checkpoint_sha256"],
        "development_2s": {
            name: {metric: values["80"][metric]
                   for metric in key_metrics}
            for name, values in report["development_validation"][
                "macro_across_runs"].items()},
        "highsteer_10s": {
            name: {metric: values["400"][metric]
                   for metric in key_metrics}
            for name, values in report[
                "highsteer_development"][
                    "macro_across_runs"].items()},
        "development_10s": {
            name: {metric: values["400"][metric]
                   for metric in key_metrics}
            for name, values in report["development_validation"][
                "macro_across_runs"].items()},
        "practice_10s": {
            name: {metric: values["400"][metric]
                   for metric in key_metrics}
            for name, values in report["practice_diagnostic"][
                "macro_across_runs"].items()},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
