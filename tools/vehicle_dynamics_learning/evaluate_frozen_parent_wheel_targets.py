#!/usr/bin/env python3
"""Score the frozen parent EDSSM against matched held-out wheel targets.

This is a one-step, teacher-forced-context evaluation only: each prediction
uses the prior 2 s state/history and the known next command, then is scored
against three wheel-target definitions on the exact same validation rows.
No checkpoint is trained or modified; future sensor/truth rows are labels only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from tools.vehicle_dynamics_learning.effective_race_teacher import (
    DT_S,
    HISTORY_STEPS,
    append_roll_state,
    physical_state_from_dataset,
    raw_encoder_history_features,
    wheel_innovation_history_features,
)
from tools.vehicle_dynamics_learning.score_effective_race_teacher import (
    _cluster_summary,
    _load_model,
    _rmse,
    _window_batch,
)
from tools.vehicle_dynamics_learning.signal_semantics import WHEEL_RADIUS_M
from tools.vehicle_dynamics_learning.train_nssm import _load_dataset


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = (ROOT / "live_runs/derived_dynamics_learning_20260928"
    / "full_modeling_reset_20261001/replacement_offline_sim_raceline_20261002"
    / "encoder_raw_state_teacher_v1/edssm_gru_z32_e2_rollresidual_10s_yawonly_wheelonly_lowthrottle4_joint_lr3e5_seed101"
    / "best.pt")
EXPECTED_CHECKPOINT_SHA256 = (
    "8f84fa54306492bd9750e4e1e3fb0be014195491eccd8f4fd67f2943402bfe8e")
DEFAULT_DATASET = (ROOT / "live_runs/derived_dynamics_learning_20260928"
    / "full_modeling_reset_20261001/replacement_offline_sim_raceline_20261003"
    / "full_throttle_domain_v1/encoder_fixed40hz_v1"
    / "openplane_dynamics_raw_wheels_fixed40hz.npz")
DEFAULT_PRACTICE_DATASET = (ROOT / "live_runs/derived_dynamics_learning_20260928"
    / "full_modeling_reset_20261001/replacement_offline_sim_raceline_20261003"
    / "full_throttle_domain_v1/encoder_fixed40hz_v1"
    / "practice_dynamics_raw_wheels_fixed40hz.npz")
DEFAULT_OUTPUT = (ROOT / "live_runs/derived_dynamics_learning_20260928"
    / "full_modeling_reset_20261001/replacement_offline_sim_raceline_20261003"
    / "full_throttle_domain_v1/wp16_frozen_parent_one_step_v3.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _transition_windows(data: dict[str, Any], split: str,
                        state: np.ndarray,
                        max_throttle_command: float,
                        history_steps: int = HISTORY_STEPS
                        ) -> list[dict[str, Any]]:
    run_ids = np.asarray(data["run_ids"]).astype(str)
    splits = np.asarray(data["splits"]).astype(str)
    packet = np.asarray(data["packet_sequence"], dtype=np.int64)
    dt = np.asarray(data["dt_s"], dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for sequence_id, (bounds, run_index) in enumerate(
            zip(data["bounds"], data["seq_run"])):
        start, end = map(int, bounds)
        run = int(run_index)
        if splits[run] != split or end - start <= history_steps:
            continue
        for state_index in range(start + history_steps - 1, end - 1):
            history_start = state_index - history_steps + 1
            target_index = state_index + 1
            interval_rows = np.arange(history_start, state_index + 2)
            if (not np.all(np.isclose(dt[interval_rows[1:]], DT_S,
                                      rtol=0.0, atol=1e-7))
                    or np.any(np.diff(packet[interval_rows]) != 1)
                    or np.hypot(*state[target_index, :2]) > 12.0
                    or data["frames"][target_index, 8] > max_throttle_command):
                continue
            rows.append({"sequence_id": sequence_id, "start": state_index,
                         "run": run, "regimes": []})
    return rows


def _batch_predict(model: Any, torch: Any, data: dict[str, Any],
                   state: np.ndarray, windows: list[dict[str, Any]],
                   raw_history: np.ndarray | None,
                   batch_size: int, device: str) -> tuple[np.ndarray, np.ndarray]:
    predictions, target_rows = [], []
    for offset in range(0, len(windows), batch_size):
        batch = windows[offset:offset + batch_size]
        history, initial, delayed, commands, _, _ = _window_batch(
            data, state, batch, model.history_state_size, raw_history,
            horizon_steps=1, command_offset_frames=0)
        with torch.no_grad():
            predicted, _, _, _ = model.rollout(
                torch.as_tensor(initial, dtype=torch.float32, device=device),
                torch.as_tensor(delayed, dtype=torch.float32, device=device),
                torch.as_tensor(history, dtype=torch.float32, device=device),
                torch.as_tensor(commands, dtype=torch.float32, device=device))
        predictions.append(predicted[:, 0, 5:7].detach().cpu().numpy().astype(
            np.float64))
        target_rows.extend(int(row["start"]) + 1 for row in batch)
    return np.concatenate(predictions), np.asarray(target_rows, dtype=np.int64)


def _target_metrics(prediction: np.ndarray, target: np.ndarray,
                    mask: np.ndarray,
                    angle_target: np.ndarray | None = None
                    ) -> dict[str, Any]:
    pred, target = prediction[mask], target[mask]
    error = pred - target
    absolute_error = np.abs(error)
    result = {
        "paired_transition_count": int(mask.sum()),
        "wheel_pair_rmse_mps": float(_rmse(error)),
        "left_rmse_mps": float(_rmse(error[:, 0])),
        "right_rmse_mps": float(_rmse(error[:, 1])),
        "left_bias_mps": float(np.mean(error[:, 0])),
        "right_bias_mps": float(np.mean(error[:, 1])),
        "wheel_pair_bias_mps": float(np.mean(error)),
        "wheel_pair_mae_mps": float(np.mean(absolute_error)),
        "wheel_pair_absolute_error_p50_mps": float(
            np.quantile(absolute_error, 0.50)),
        "wheel_pair_absolute_error_p95_mps": float(
            np.quantile(absolute_error, 0.95)),
    }
    if angle_target is not None:
        angle_error = (prediction[mask] * DT_S / WHEEL_RADIUS_M
                       - angle_target[mask])
        result["angle_increment_pair_rmse_rad"] = float(_rmse(angle_error))
        result["angle_increment_rmse_mps_equivalent"] = float(
            _rmse(angle_error) * WHEEL_RADIUS_M / DT_S)
    return result


def _score_one_dataset(checkpoint: Path, dataset_path: Path,
                       split: str, device: str, batch_size: int
                       ) -> dict[str, Any]:
    torch, model, metadata = _load_model(checkpoint, device)
    data = _load_dataset(dataset_path)
    if not np.allclose(data["dt_s"], DT_S, rtol=0.0, atol=1e-7):
        raise ValueError(f"{dataset_path}: expected fixed 25 ms samples")
    parent_dataset_sha = str(data.get(
        "encoder_raw_source_dataset_sha256", "unknown"))
    if (parent_dataset_sha != str(metadata["dataset_sha256"])
            and split != "unseen_practice"):
        raise ValueError("fixed40 sidecar does not derive from the frozen checkpoint dataset")
    if data["schema_version"] not in (8, 9):
        raise ValueError("external validation data must retain supported schema 8/9 semantics")
    run_ids = np.asarray(data["run_ids"]).astype(str)
    splits = np.asarray(data["splits"]).astype(str)
    selected_ids = set(run_ids[splits == split])
    overlap = selected_ids.intersection(
        str(run) for run in metadata.get("training_runs", []))
    if overlap:
        raise ValueError(f"frozen checkpoint training/evaluation run overlap: {sorted(overlap)}")

    state = physical_state_from_dataset(
        data, wheel_state_source=str(metadata.get(
            "wheel_state_source", "filtered_odometry"))).astype(np.float64)
    if bool(metadata.get("include_roll_state", False)):
        state = append_roll_state(data, state).astype(np.float64)
    max_throttle = float(metadata.get("max_throttle_command_norm", 0.50))
    windows = _transition_windows(data, split, state, max_throttle)
    if not windows:
        return {"split": split, "independent_runs": 0, "run_reports": {}}
    if model.include_raw_encoder_history:
        raw_history = raw_encoder_history_features(data)
    elif model.include_wheel_innovation_history:
        raw_history = wheel_innovation_history_features(data)
    else:
        raw_history = None
    predictions, target_rows = _batch_predict(
        model, torch, data, state, windows, raw_history, batch_size, device)
    encoder = data.get("encoder_raw_surface_mps")
    encoder_valid = data.get("encoder_raw_valid")
    if encoder is None or encoder_valid is None:
        raise ValueError("fixed-25ms encoder target sidecar is required")
    raw = np.asarray(encoder, dtype=np.float64)[target_rows]
    valid = np.asarray(encoder_valid, dtype=bool)[target_rows]
    stored = np.asarray(data["frames"], dtype=np.float64)[target_rows, 5:7]
    angle_increment = raw * DT_S / WHEEL_RADIUS_M
    same_mask = (valid & np.isfinite(raw).all(axis=1)
                 & np.isfinite(stored).all(axis=1)
                 & np.isfinite(angle_increment).all(axis=1))
    report_by_run: dict[str, Any] = {}
    selected_window_runs = np.asarray([
        run_ids[int(row["run"])] for row in windows], dtype=str)
    for run_id in sorted(set(selected_window_runs.tolist())):
        local = selected_window_runs == run_id
        common = local & same_mask
        report_by_run[run_id] = {
            "split": split,
            "candidate_transitions": int(local.sum()),
            "same_mask_transitions": int(common.sum()),
            "all_targets_same_physical_transition_mask": True,
            "fixed25_encoder_rate": _target_metrics(
                predictions, raw, common, angle_increment),
            "stored_100ms_rate": _target_metrics(
                predictions, stored, common),
            "encoder_angle_increment": _target_metrics(
                predictions, raw, common, angle_increment),
        }

    metrics: dict[str, dict[str, Any]] = {}
    for target_name in ("fixed25_encoder_rate", "stored_100ms_rate",
                        "encoder_angle_increment"):
        per_run = {run_id: values[target_name]["wheel_pair_rmse_mps"]
                   for run_id, values in report_by_run.items()
                   if values["same_mask_transitions"] > 0}
        metrics[target_name] = _cluster_summary(per_run)
        metrics[target_name]["per_run"] = per_run
    return {
        "split": split,
        "dataset": str(dataset_path.relative_to(ROOT)),
        "dataset_sha256": _sha256(dataset_path),
        "dataset_parent_sha256": parent_dataset_sha,
        "external_dataset_evaluation": parent_dataset_sha != str(
            metadata["dataset_sha256"]),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_training_dataset_sha256": metadata["dataset_sha256"],
        "checkpoint_training_run_overlap": sorted(overlap),
        "future_truth_or_feedback_used_as_model_input": False,
        "input_policy": "prior 2 s state/history plus known next steering/throttle command; next-step truth/encoder channels are scoring labels only",
        "transition_window_policy": "all transitions with 80-step contiguous history, consecutive packet sequence, fixed 25 ms dt; run/reset boundaries remain sequence-separated",
        "model_support_filter": {"max_speed_mps": 12.0,
                                 "max_throttle_command_norm": max_throttle},
        "shared_target_mask": "fixed25 encoder valid + finite fixed25 rate + finite stored 100ms rate + finite angle increment; identical rows for all scores",
        "shared_transition_count": int(same_mask.sum()),
        "independent_run_count": len(report_by_run),
        "macro_run_metrics": metrics,
        "run_reports": report_by_run,
    }


def evaluate(checkpoint: Path, openplane: Path, practice: Path,
             output: Path, device: str = "cuda", batch_size: int = 512
             ) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite model evaluation: {output}")
    for path in (checkpoint, openplane, practice):
        if not path.is_file():
            raise FileNotFoundError(path)
    checkpoint_sha256 = _sha256(checkpoint)
    if checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(
            "WP16/WP17 frozen-parent comparison requires the checkpoint used "
            "by the registered WP14 first-divergence report; got "
            f"{checkpoint_sha256}")
    openplane_result = _score_one_dataset(
        checkpoint, openplane, "validation", device, batch_size)
    practice_result = _score_one_dataset(
        checkpoint, practice, "unseen_practice", device, batch_size)
    report = {
        "schema_version": 1,
        "purpose": "WP16.4 same-transition one-step error of the frozen parent under distinct rear-wheel targets",
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_unchanged": True,
        "training_performed": False,
        "device": device,
        "batch_size": batch_size,
        "openplane_validation": openplane_result,
        "unseen_practice": practice_result,
        "interpretation": {
            "fixed25_vs_angle_increment": "The angle target is the same measured fixed-cadence increment expressed in radians; its m/s-equivalent score must equal the fixed25 rate score up to floating-point tolerance.",
            "stored_100ms": "A distinct smoothed wheel measurement, scored on exactly the same one-step transitions and validity mask.",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--openplane-dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--practice-dataset", type=Path,
                        default=DEFAULT_PRACTICE_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    report = evaluate(args.checkpoint.resolve(), args.openplane_dataset.resolve(),
        args.practice_dataset.resolve(), args.output.resolve(),
        args.device, args.batch_size)
    print(json.dumps({
        "checkpoint_sha256": report["checkpoint_sha256"],
        "openplane_shared_transitions": report["openplane_validation"]["shared_transition_count"],
        "practice_shared_transitions": report["unseen_practice"]["shared_transition_count"],
        "openplane_metrics": report["openplane_validation"]["macro_run_metrics"],
        "practice_metrics": report["unseen_practice"]["macro_run_metrics"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
