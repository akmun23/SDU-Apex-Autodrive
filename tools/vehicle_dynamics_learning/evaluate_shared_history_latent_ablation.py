#!/usr/bin/env python3
"""Inference-only test of replacing a learned latent step with its own GRU.

The parent EDSSM weights remain frozen. The candidate starts from the same
encoded 80-frame history, but after each transition updates the history GRU
with its own predicted physical state and the next known command. This isolates
the latent-transition/re-encoding mismatch found in WP25 without retraining.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from tools.vehicle_dynamics_learning.score_effective_race_teacher import (
    DEFAULT_DATASET,
    _fixed_validation_windows,
    _load_model,
    evaluate_model,
)
from tools.vehicle_dynamics_learning.train_consistent_history_teacher import (
    PARENT_CHECKPOINT,
    _json_safe,
    _state_and_targets,
)
from tools.vehicle_dynamics_learning.train_nssm import _load_dataset


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = (ROOT / "live_runs/derived_dynamics_learning_20260928"
          / "full_modeling_reset_20261001/replacement_offline_sim_raceline_20261003"
          / "full_throttle_domain_v1/next_phase_after_2129427"
          / "shared_history_latent_ablation_v1_20261004.json")
HORIZONS = {"0.75s": 30, "2s": 80, "5s": 200}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class SharedHistoryRollout:
    """Keep a frozen EDSSM transition but share its history GRU at rollout."""

    def __init__(self, base: Any) -> None:
        self.base = base
        self.history_state_size = base.history_state_size
        self.state_size = base.state_size
        self.include_raw_encoder_history = base.include_raw_encoder_history
        self.include_wheel_innovation_history = base.include_wheel_innovation_history

    def eval(self):
        self.base.eval()
        return self

    def rollout(self, initial_state, delayed_command, history, commands):
        import torch

        if (history.ndim != 3 or history.shape[1] != 80
                or history.shape[2] != len(self.base.history_mean)
                or commands.ndim != 3 or commands.shape[2] != 2):
            raise ValueError("shared-history ablation received invalid rollout shapes")
        normalized_history = ((history - self.base.history_mean)
                              / self.base.history_scale)
        _, hidden = self.base.history_encoder(normalized_history)
        latent = self.base.history_projection(hidden[-1])
        state = initial_state
        previous_command = delayed_command
        states, accelerations, latents, gates = [], [], [], []
        for index in range(commands.shape[1]):
            command = commands[:, index]
            state_next, _, latent_from_old_transition, acceleration, gate = (
                self.base.transition(state, previous_command, latent, command))
            states.append(state_next)
            accelerations.append(acceleration)
            latents.append(latent)
            gates.append(gate)
            if index + 1 < commands.shape[1]:
                history_row = torch.cat((
                    state_next[:, :self.history_state_size],
                    commands[:, index + 1]), dim=-1)
                normalized_row = ((history_row - self.base.history_mean)
                                  / self.base.history_scale)[:, None, :]
                _, hidden = self.base.history_encoder(normalized_row, hidden)
                latent = self.base.history_projection(hidden[-1])
            else:
                latent = latent_from_old_transition
            previous_command = command
            state = state_next
        return (torch.stack(states, dim=1),
                torch.stack(accelerations, dim=1),
                torch.stack(latents, dim=1),
                torch.stack(gates, dim=1))


def _cluster_summary(values: dict[str, float], seed: int = 20261004
                     ) -> dict[str, Any]:
    run_names = sorted(values)
    array = np.asarray([values[name] for name in run_names], dtype=np.float64)
    if not len(array):
        return {"independent_runs": 0, "macro_run_mean": None,
                "run_cluster_bootstrap_95pct_ci": None, "per_run": {}}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(array), size=(5000, len(array)))
    return {
        "independent_runs": int(len(array)),
        "macro_run_mean": float(np.mean(array)),
        "run_cluster_bootstrap_95pct_ci": np.quantile(
            array[draws].mean(axis=1), (0.025, 0.975)).tolist(),
        "per_run": {name: float(values[name]) for name in run_names},
    }


def _paired_changes(parent: dict[str, Any], candidate: dict[str, Any]
                    ) -> dict[str, Any]:
    result = {}
    for horizon in HORIZONS:
        result[horizon] = {}
        fields = ("position_radial_m", "heading_rad", "body", "wheel",
                  "speed_mps", "raw/u_com_mps", "raw/v_com_mps",
                  "raw/yaw_rate_rps", "raw/rear_left_surface_speed_mps",
                  "raw/rear_right_surface_speed_mps")
        for field in fields:
            left = candidate["horizons"][horizon][field]["per_run"]
            right = parent["horizons"][horizon][field]["per_run"]
            shared = sorted(set(left).intersection(right))
            if not shared:
                continue
            deltas = {run: float(left[run] - right[run]) for run in shared}
            result[horizon][field] = {
                "direction": "candidate minus frozen parent; negative favors candidate",
                **_cluster_summary(deltas),
            }
    return result


def evaluate(output_path: Path = OUTPUT, device_name: str = "cpu") -> dict[str, Any]:
    import torch

    output_path = output_path.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing ablation report: {output_path}")
    checkpoint_hash = _sha256(PARENT_CHECKPOINT)
    torch.set_num_threads(2)
    data = _load_dataset(DEFAULT_DATASET)
    state, _ = _state_and_targets(data)
    windows = _fixed_validation_windows(
        data, state, max_windows_per_run=8, horizon_steps=200,
        max_throttle_command=0.50, split="validation")
    if len({int(row["run"]) for row in windows}) != 6:
        raise ValueError("expected six independent dynamic validation captures")
    _, parent, metadata = _load_model(PARENT_CHECKPOINT, device_name)
    parent.eval()
    candidate = SharedHistoryRollout(parent)
    parent_metrics = evaluate_model(
        parent, data, state, metadata["state_scale"], device_name,
        windows=windows, batch_size=8, horizon_steps=HORIZONS,
        command_offset_frames=-1)
    candidate_metrics = evaluate_model(
        candidate, data, state, metadata["state_scale"], device_name,
        windows=windows, batch_size=8, horizon_steps=HORIZONS,
        command_offset_frames=-1)
    report = {
        "study": "inference-only shared GRU history/rollout update",
        "parent_checkpoint": str(PARENT_CHECKPOINT),
        "parent_checkpoint_sha256": checkpoint_hash,
        "dataset": str(DEFAULT_DATASET),
        "dataset_sha256": _sha256(DEFAULT_DATASET),
        "validation_role": "six whole-run dynamic development captures",
        "training_or_checkpoint_selection_performed": False,
        "practice_test_or_final_test_used": False,
        "future_truth_or_sensor_feedback_used": False,
        "command_offset_frames": -1,
        "prediction_policy": (
            "initial measured state/pose/history only; after initialization, "
            "recorded commands and recursively predicted state only"),
        "parent_metrics": parent_metrics,
        "shared_history_metrics": candidate_metrics,
        "paired_candidate_minus_parent": _paired_changes(
            parent_metrics, candidate_metrics),
        "decision": (
            "mechanism ablation only; do not integrate unless whole-run "
            "validation improves without body/wheel regressions"),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_json_safe(report), indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    report = evaluate(args.output, args.device)
    print(json.dumps({
        "report": str(args.output.resolve()),
        "parent_5s_position_m": report["parent_metrics"]["horizons"][
            "5s"]["position_radial_m"]["macro_run_mean"],
        "shared_5s_position_m": report["shared_history_metrics"]["horizons"][
            "5s"]["position_radial_m"]["macro_run_mean"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
