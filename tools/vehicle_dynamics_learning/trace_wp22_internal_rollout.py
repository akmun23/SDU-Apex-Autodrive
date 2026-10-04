#!/usr/bin/env python3
"""Freeze WP24 comparators and run WP25 scoring-only A2 diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tools.vehicle_dynamics_learning.augmented_state_space_plant import (
    DT_S,
    WHEEL_RADIUS_M,
    load_checkpoint,
)
from tools.vehicle_dynamics_learning.diagnose_wp22_candidate_failure import (
    _start_tags,
)
from tools.vehicle_dynamics_learning.operating_regions import region_masks
from tools.vehicle_dynamics_learning.run_wp19_target_ablation import (
    make_model as make_wp19_model,
    training_statistics,
)
from tools.vehicle_dynamics_learning.train_augmented_state_space_plant import (
    BOOTSTRAP_REPLICATES,
    DEFAULT_OUTPUT as WP22_OUTPUT,
    EVAL_HORIZONS,
    EXPECTED_WP19_SHA256,
    ROOT,
    SEED,
    TASK_ROOT,
    _evaluate,
    _load_data,
    _per_window_metrics,
    _training_windows_and_stats,
    sha256_file,
)


SOURCE_COMMIT = "2129427838101eee277e17e91727810bbe2df687"
NEXT_ROOT = TASK_ROOT / "next_phase_after_2129427"
WP25_ROOT = NEXT_ROOT / "wp25_failure_localization"
WP22_REPORT = WP22_OUTPUT / "wp22_training_report.json"
WP19_CHECKPOINT = TASK_ROOT / "wp19_target_ablation_v2_common_encoder_mask" / (
    "07_body_state_increment__encoder_angle_increment/model.pt")
WP23_ROOT = TASK_ROOT / "wp23_structural_ablations_seed101_v1"
WP20_REPORT = TASK_ROOT / "wp20_support_calibration_v2_model_train_runs" / (
    "wp20_support_calibration_report.json")
REGISTRY = ROOT / "live_runs/derived_dynamics_learning_20260928/experiment_registry_20261003.json"

VARIANTS = ("M0_native", "M1_neutral_measurement", "M2_hold_last_measurement",
            "M3_zero_differential", "M4_future_encoder_oracle")
PRIMARY_HORIZONS = {"250ms": 10, "750ms": 30, "2s": 80, "5s": 200}
METRICS = ("position_radial_trajectory_rmse_m", "heading_trajectory_rmse_rad",
           "u_rmse_mps", "v_rmse_mps", "yaw_rate_rmse_rps",
           "encoder_angle_increment_rmse_rad")
REGION_KEYS = {
    "speed": tuple(f"S{i}" for i in range(6)),
    "steering": tuple(f"D{i}" for i in range(5)),
    "steering_rate": tuple(f"R{i}" for i in range(4)),
    "throttle_command": tuple(f"throttle_{i}" for i in range(7)),
    "throttle_slew": tuple(f"T{i}" for i in range(4)),
    "wheel_body_mismatch": tuple(f"M{i}" for i in range(7)),
    "combined": ("high_speed_near_straight", "high_speed_moderate_steering",
                 "7_to_9mps_high_steering", "low_speed_high_steering",
                 "negative_command_braking", "braking_release", "throttle_pickup",
                 "steering_turn_in", "steering_unwind",
                 "simultaneous_steering_throttle_transition",
                 "large_wheel_body_mismatch"),
}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(value), indent=2, sort_keys=True,
                               allow_nan=False) + "\n",
                    encoding="utf-8")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(child) for child in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                          text=True, capture_output=True).stdout.strip()


def _registry_check() -> dict[str, Any]:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    if registry.get("repository_head") != SOURCE_COMMIT:
        raise RuntimeError("experiment registry is not indexed at frozen source commit")
    if not registry.get("provenance_verification", {}).get(
            "active_black_box_artifacts_all_present_and_hashed"):
        raise RuntimeError("experiment registry lacks explicit WP19-WP23 artifact indexing")
    required = {"WP19 selected checkpoint", "WP20 report", "WP22 A2 checkpoint",
                "WP22 failure diagnosis", "WP23 A0 checkpoint", "WP23 A1 checkpoint"}
    indexed = {item["label"] for item in registry.get(
        "active_black_box_work_package_artifacts", [])}
    if not required.issubset(indexed):
        raise RuntimeError(f"registry omitted required artifacts: {sorted(required-indexed)}")
    return registry


def _checkpoint_inventory() -> dict[str, dict[str, str]]:
    paths = {
        "wp19": WP19_CHECKPOINT,
        "wp22_A2": WP22_OUTPUT / "seed101_candidate.pt",
        "wp23_A0": WP23_ROOT / "A0/seed101_A0_candidate.pt",
        "wp23_A1": WP23_ROOT / "A1/seed101_A1_candidate.pt",
    }
    inventory = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        inventory[name] = {
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(path),
        }
    if inventory["wp19"]["sha256"] != EXPECTED_WP19_SHA256:
        raise RuntimeError("WP19 comparator hash differs from the frozen trainer")
    return inventory


def _frozen_refs(data) -> dict[str, dict[str, list[tuple[int, int, int]]]]:
    return {"development_validation": data.validation_windows,
            "practice_diagnostic": data.practice_windows}


def _start_record(data, ref: tuple[int, int, int], role: str) -> dict[str, Any]:
    capture_index, sequence_index, source_row = map(int, ref)
    capture = data.captures[capture_index]
    begin_seq, end_seq = map(int, capture.bounds[sequence_index])
    begin = begin_seq + source_row
    run_index = int(capture.sequence_run[sequence_index])
    tags, _ = _start_tags(capture, sequence_index, source_row)
    maximum_steps = end_seq - begin - 1
    return {
        "split_role": role,
        "run_id": str(capture.run_ids[run_index]),
        "sequence_index": sequence_index,
        "source_row": source_row,
        "absolute_row": begin,
        "condition_id": int(data.raw_sources[capture_index][
            "sequence_condition_id"][sequence_index]),
        "canonical_regime_labels": tags,
        "maximum_available_horizon_steps": maximum_steps,
        "maximum_available_horizon_seconds": maximum_steps * DT_S,
    }


def _freeze_starts(data) -> dict[str, Any]:
    output = {}
    for role, runs in _frozen_refs(data).items():
        output[role] = {
            run_id: [_start_record(data, ref, role) for ref in refs]
            for run_id, refs in sorted(runs.items())
        }
    return {
        "source_commit": SOURCE_COMMIT,
        "start_selection": "Exact deterministic WP22 _select_eval_windows output; no reselection",
        "horizon_semantics": "Maximum contiguous available steps from each frozen source row; 25 ms/sample",
        "split_roles": output,
    }


def _compare_reproduction(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    differences = []
    for split in ("openplane_validation", "practice_transfer"):
        previous = old["evaluation"]["splits"][split]["per_run"]
        current = new["splits"][split]["per_run"]
        if set(previous) != set(current):
            raise RuntimeError(f"WP24 run inventory mismatch in {split}")
        for run_id, entry in current.items():
            old_horizons = previous[run_id]["horizons"]
            if set(entry["horizons"]) != set(old_horizons):
                raise RuntimeError(f"WP24 horizon inventory mismatch: {split}/{run_id}")
            for horizon, now in entry["horizons"].items():
                before = old_horizons[horizon]
                for method in ("candidate_macro_window_metrics",
                               "wp19_parent_macro_window_metrics"):
                    for metric in METRICS:
                        a, b = now[method].get(metric), before[method].get(metric)
                        if a is not None and b is not None:
                            differences.append({
                                "absolute_difference": abs(float(a) - float(b)),
                                "split": split, "run_id": run_id,
                                "horizon": horizon, "method": method,
                                "metric": metric,
                            })
    maximum = max(differences, key=lambda item: item["absolute_difference"])
    gate_matches = new["gate"] == old["evaluation"]["gate"]
    if maximum["absolute_difference"] > 1e-6 or not gate_matches:
        raise RuntimeError("WP24 A2 reproduction mismatch; stop before WP25")
    return {
        "status": "passed",
        "compared_metric_count": len(differences),
        "maximum_absolute_metric_difference": maximum,
        "gate_matches": gate_matches,
        "reproduced_gate": new["gate"],
        "reproduced_evaluation": new,
        "tolerance": 1e-6,
    }


def _freeze_and_reproduce(data, device: torch.device) -> dict[str, Any]:
    registry = _registry_check()
    checkpoint_inventory = _checkpoint_inventory()
    indexed_hashes = registry["provenance_verification"]["checkpoint_hashes"]
    for item in checkpoint_inventory.values():
        if indexed_hashes.get(item["path"]) != item["sha256"]:
            raise RuntimeError(f"registry did not index checkpoint/hash: {item['path']}")
    comparator = {
        "source_commit": SOURCE_COMMIT,
        "repository_head_at_run": _git_head(),
        "registry_path": REGISTRY.relative_to(ROOT).as_posix(),
        "registry_sha256": sha256_file(REGISTRY),
        "registry_worktree_dirty_at_generation": registry["worktree_dirty_at_generation"],
        "registry_index_counts": registry["counts"],
        "comparators": checkpoint_inventory,
        "dataset_inputs": [],
        "wp22_report": {
            "path": WP22_REPORT.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(WP22_REPORT),
        },
    }
    report = json.loads(WP22_REPORT.read_text(encoding="utf-8"))
    for item in report["inputs"]["training_datasets"]:
        path = ROOT / item["path"]
        actual = sha256_file(path)
        if actual != item["sha256"]:
            raise RuntimeError(f"training dataset hash changed: {item['path']}")
        comparator["dataset_inputs"].append({
            "path": item["path"], "sha256": actual,
        })
    for path in (WP19_CHECKPOINT, WP20_REPORT):
        if not path.is_file():
            raise FileNotFoundError(path)
    comparator["support_calibration"] = {
        "path": WP20_REPORT.relative_to(ROOT).as_posix(),
        "sha256": sha256_file(WP20_REPORT),
    }
    _write_json(NEXT_ROOT / "frozen_comparators.json", comparator)
    starts = _freeze_starts(data)
    _write_json(NEXT_ROOT / "frozen_eval_starts.json", starts)

    old = report
    model, _ = load_checkpoint(ROOT / comparator["comparators"]["wp22_A2"]["path"],
                               str(device))
    parent_raw = torch.load(WP19_CHECKPOINT, map_location=device, weights_only=True)
    parent = make_wp19_model(torch.nn).to(device)
    parent.load_state_dict(parent_raw["state_dict"], strict=True)
    stats = training_statistics(data.captures, data.training_windows_80)
    reproduced = _evaluate(data, model, parent, stats, device, "WP24 reproduction")
    result = _compare_reproduction(old, reproduced)
    result.update({
        "source_commit": SOURCE_COMMIT,
        "candidate_sha256": comparator["comparators"]["wp22_A2"]["sha256"],
        "wp19_sha256": comparator["comparators"]["wp19"]["sha256"],
        "future_truth_or_feedback_used_as_rollout_input": False,
        "split_roles": ["development_validation", "practice_diagnostic"],
    })
    _write_json(WP25_ROOT / "wp24_reproduction_report.json", result)
    return result


def _neutral_measurement_mean(data) -> np.ndarray:
    run_means = []
    for run_id in data.training_runs:
        values = []
        for capture in data.captures:
            for seq, ((start, end), run_raw) in enumerate(
                    zip(capture.bounds, capture.sequence_run)):
                run = int(run_raw)
                if str(capture.run_ids[run]) != run_id or str(capture.splits[run]) != "train":
                    continue
                start, end = int(start), int(end)
                valid = capture.encoder_valid[start + 1:end]
                increments = capture.encoder_rate[start + 1:end] * (DT_S / WHEEL_RADIUS_M)
                if valid.any():
                    values.append(increments[valid])
        if values:
            run_means.append(np.concatenate(values, axis=0).mean(axis=0))
    if len(run_means) != len(data.training_runs):
        raise RuntimeError("cannot calculate WP25 train-run-balanced neutral measurement")
    return np.mean(run_means, axis=0).astype(np.float32)


def _support_features(model) -> torch.Tensor:
    state = model._state
    measurement_history = model._encoder_history
    valid_history = model._encoder_valid_history
    previous_actuator = model._previous_command
    if measurement_history.shape[1] < 4:
        pad = measurement_history[:, :1].expand(-1, 4 - measurement_history.shape[1], -1)
        measurement_history = torch.cat((pad, measurement_history), dim=1)
    wheel_rates = measurement_history * (model.config.wheel_radius_m / model.config.dt_s)
    filtered = wheel_rates.mean(dim=1)
    return torch.stack((
        torch.linalg.vector_norm(state[:, :2], dim=1),
        state[:, 3], (state[:, 3] - previous_actuator[:, 0]) / model.config.dt_s,
        state[:, 4], (state[:, 4] - previous_actuator[:, 1]) / model.config.dt_s,
        state[:, 2], torch.abs(filtered.mean(dim=1) - state[:, 0]),
    ), dim=1)


def _make_arrays(data, refs, horizon: int, device: torch.device):
    from tools.vehicle_dynamics_learning.train_augmented_state_space_plant import _batch_arrays

    raw = _batch_arrays(data, refs, horizon)
    tensors = [torch.as_tensor(value,
        dtype=torch.bool if i == 7 else torch.float32, device=device)
        for i, value in enumerate(raw)]
    history, initial, pose, commands, truth_body, truth_pose, truth_encoder, valid, _ = tensors
    truth_actuator = []
    for capture_index, sequence_index, source_row in refs:
        capture = data.captures[capture_index]
        begin = int(capture.bounds[sequence_index, 0]) + source_row
        truth_actuator.append(capture.frames[begin + 1:begin + horizon + 1, 3:5])
    return (history, initial, pose, commands, truth_body, truth_pose,
            truth_encoder, valid, torch.as_tensor(np.asarray(truth_actuator),
            dtype=torch.float32, device=device))


def _rollout(data, model, refs, horizon: int, variant: str,
             neutral: np.ndarray, device: torch.device,
             keep_trace: bool = False):
    history, initial, pose, commands, truth_body, truth_pose, truth_enc, valid, truth_act = \
        _make_arrays(data, refs, horizon, device)
    model.eval()
    model.reset(history, initial, pose)
    neutral_t = torch.as_tensor(neutral, dtype=torch.float32, device=device)[None]
    outputs: dict[str, list[np.ndarray]] = defaultdict(list)
    hooks = []
    captured: dict[str, torch.Tensor] = {}
    current_step = 0

    def body_hook(_module, _inputs, output):
        captured["proposal"] = output.detach()

    def measurement_hook(_module, _inputs, output):
        captured["measurement"] = output.detach()

    def support_pre_hook(_module, inputs):
        captured["support_features"] = inputs[0].detach()

    def support_hook(_module, _inputs, output):
        captured["support_score"] = output["score"].detach()
        captured["support_confidence"] = output["confidence"].detach()

    def latent_pre_hook(_module, inputs):
        features, latent = inputs
        source = features[..., -2:]
        if variant == "M1_neutral_measurement":
            replacement = neutral_t.expand_as(source)
        elif variant == "M2_hold_last_measurement":
            previous = model._measurement
            replacement = previous.detach()
        elif variant == "M3_zero_differential":
            mean = source.mean(dim=-1, keepdim=True)
            replacement = mean.expand_as(source)
        elif variant == "M4_future_encoder_oracle":
            mask = valid[:, current_step, None]
            truth = truth_enc[:, current_step]
            replacement = torch.where(mask, truth, neutral_t.expand_as(truth))
        else:
            return None
        return (torch.cat((features[..., :-2], replacement), dim=-1), latent)

    def latent_hook(_module, _inputs, output):
        captured["latent_next"] = output.detach()

    if keep_trace:
        hooks.extend((model.body_residual.register_forward_hook(body_hook),
                      model.measurement_head.register_forward_hook(measurement_hook),
                      model.support_estimator.register_forward_pre_hook(support_pre_hook),
                      model.support_estimator.register_forward_hook(support_hook),
                      model.latent_transition.register_forward_pre_hook(latent_pre_hook),
                      model.latent_transition.register_forward_hook(latent_hook)))
    elif variant != "M0_native":
        hooks.append(model.latent_transition.register_forward_pre_hook(latent_pre_hook))

    state_rows, pose_rows, measurement_rows, proposal_rows = [], [], [], []
    latent_rows, latent_delta_rows, body_delta_rows = [], [], []
    support_feature_rows, support_score_rows, support_confidence_rows = [], [], []
    initial_state = model._state.detach().clone()
    initial_pose = model._pose.detach().clone()
    initial_latent = model._latent.detach().clone()
    initial_measurement = model._measurement.detach().clone()
    initial_support = model._support
    initial_support_features = _support_features(model).detach()
    try:
        with torch.no_grad():
            for current_step in range(horizon):
                before_state = model._state.detach().clone()
                before_latent = model._latent.detach().clone()
                _state, measurement, support = model.step(commands[:, current_step])
                if keep_trace:
                    state_rows.append(model._state.detach().cpu().numpy())
                    pose_rows.append(model._pose.detach().cpu().numpy())
                    measurement_rows.append(measurement.detach().cpu().numpy())
                    proposal_rows.append(captured["proposal"].cpu().numpy())
                    latent_rows.append(model._latent.detach().cpu().numpy())
                    latent_delta_rows.append((model._latent - before_latent).cpu().numpy())
                    body_delta_rows.append((model._state[:, :3] - before_state[:, :3]).cpu().numpy())
                    support_feature_rows.append(captured["support_features"].cpu().numpy())
                    support_score_rows.append(captured["support_score"].cpu().numpy())
                    support_confidence_rows.append(captured["support_confidence"].cpu().numpy())
                else:
                    state_rows.append(model._state.detach().cpu().numpy())
                    pose_rows.append(model._pose.detach().cpu().numpy())
                    measurement_rows.append(measurement.detach().cpu().numpy())
    finally:
        for hook in hooks:
            hook.remove()
    result = {
        "body": np.stack([row[..., :3] for row in state_rows], axis=1),
        "actuator": np.stack([row[..., 3:5] for row in state_rows], axis=1),
        "pose": np.stack(pose_rows, axis=1),
        "encoder": np.stack(measurement_rows, axis=1),
        "truth_body": truth_body.detach().cpu().numpy(),
        "truth_pose": truth_pose.detach().cpu().numpy(),
        "truth_encoder": truth_enc.detach().cpu().numpy(),
        "encoder_valid": valid.detach().cpu().numpy(),
        "truth_actuator": truth_act.detach().cpu().numpy(),
    }
    if keep_trace:
        result.update({
            "state_full": np.stack(state_rows, axis=1),
            "proposal": np.stack(proposal_rows, axis=1),
            "normalized_transition_components": np.stack(proposal_rows, axis=1)
                / np.asarray(model.config.body_increment_limit, dtype=np.float32),
            "normalized_transition_magnitude": np.linalg.norm(
                np.stack(proposal_rows, axis=1)
                / np.asarray(model.config.body_increment_limit, dtype=np.float32), axis=-1),
            "latent": np.stack(latent_rows, axis=1),
            "latent_delta": np.stack(latent_delta_rows, axis=1),
            "body_delta": np.stack(body_delta_rows, axis=1),
            "support_features": np.stack(support_feature_rows, axis=1),
            "support_score": np.stack(support_score_rows, axis=1),
            "support_confidence": np.stack(support_confidence_rows, axis=1),
            "initial_state": initial_state.cpu().numpy(),
            "initial_pose": initial_pose.cpu().numpy(),
            "initial_latent": initial_latent.cpu().numpy(),
            "initial_measurement": initial_measurement.cpu().numpy(),
            "initial_support_score": initial_support["score"].detach().cpu().numpy(),
            "initial_support_confidence": initial_support["confidence"].detach().cpu().numpy(),
            "initial_support_features": initial_support_features.cpu().numpy(),
        })
        score_path = np.column_stack((
            initial_support["score"].detach().cpu().numpy(),
            np.stack(support_score_rows, axis=1)))
        result["support_score_with_initial"] = score_path
        result["support_category_with_initial"] = np.where(
            score_path <= model.config.support_supported_upper, "supported",
            np.where(score_path <= model.config.support_weak_upper,
                     "weak", "unsupported"))
    return result


def _first_time(mask: np.ndarray, dt_s: float = DT_S) -> float | None:
    indexes = np.flatnonzero(mask)
    return float((indexes[0] + 1) * dt_s) if len(indexes) else None


def _divergence_markers(prediction: dict[str, np.ndarray], initial_support: np.ndarray,
                        weak_upper: float, unsupported_upper: float,
                        encoder_thresholds: dict[str, float]) -> list[dict[str, Any]]:
    records = []
    for i in range(len(prediction["body"])):
        p_error = np.linalg.norm(prediction["pose"][i, :, :2]
                                 - prediction["truth_pose"][i, :, :2], axis=1)
        heading_error = np.abs(np.arctan2(np.sin(
            prediction["pose"][i, :, 2] - prediction["truth_pose"][i, :, 2]),
            np.cos(prediction["pose"][i, :, 2] - prediction["truth_pose"][i, :, 2])))
        body_error = np.abs(prediction["body"][i] - prediction["truth_body"][i])
        encoder_error = prediction["encoder"][i] - prediction["truth_encoder"][i]
        mean_error = np.abs(encoder_error.mean(axis=-1))
        differential_error = np.abs(encoder_error[:, 0] - encoder_error[:, 1])
        encoder_valid = prediction["encoder_valid"][i]
        support = np.r_[initial_support[i], prediction["support_score"][i]]
        support_t = lambda threshold: (0.0 if support[0] > threshold else
            _first_time(support[1:] > threshold))
        records.append({
            "position_gt_0p10m_s": _first_time(p_error > 0.10),
            "position_gt_0p25m_s": _first_time(p_error > 0.25),
            "position_gt_0p50m_s": _first_time(p_error > 0.50),
            "heading_gt_0p05rad_s": _first_time(heading_error > 0.05),
            "heading_gt_0p10rad_s": _first_time(heading_error > 0.10),
            "u_gt_0p25mps_s": _first_time(body_error[:, 0] > 0.25),
            "yaw_rate_gt_0p25rps_s": _first_time(body_error[:, 2] > 0.25),
            "encoder_pair_mean_error_gt_train_q99_s": _first_time(
                (mean_error > encoder_thresholds["pair_mean_abs_error_q99_rad"]) & encoder_valid),
            "encoder_differential_error_gt_train_q99_s": _first_time(
                (differential_error > encoder_thresholds["differential_abs_error_q99_rad"]) & encoder_valid),
            "support_weak_boundary_s": support_t(weak_upper),
            "support_unsupported_boundary_s": support_t(unsupported_upper),
        })
    return records


def _region_summary(data, refs, prediction: dict[str, np.ndarray]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for i, (capture_index, sequence_index, source_row) in enumerate(refs):
        capture = data.captures[capture_index]
        begin = int(capture.bounds[sequence_index, 0]) + source_row
        steps = prediction["body"].shape[1]
        indexes = np.arange(begin + 1, begin + steps + 1)
        frames = capture.frames[indexes]
        valid = np.isfinite(frames).all(axis=1)
        reset_rows = np.full(steps, capture.sequence_reset[sequence_index], dtype=np.int32)
        masks = region_masks(frames, reset_rows,
                             capture.packet[indexes], valid, DT_S)
        pose_error = np.linalg.norm(prediction["pose"][i, :, :2]
                                    - prediction["truth_pose"][i, :, :2], axis=1)
        body_error = np.abs(prediction["body"][i] - prediction["truth_body"][i])
        scores = prediction["support_score"][i]
        for axis, names in REGION_KEYS.items():
            for name in names:
                mask = masks.get(name)
                if mask is None or not np.any(mask):
                    continue
                key = f"{axis}/{name}"
                bucket = summary.setdefault(key, {
                    "samples": 0, "position_error_sum_m": 0.0,
                    "u_abs_error_sum_mps": 0.0, "v_abs_error_sum_mps": 0.0,
                    "yaw_abs_error_sum_rps": 0.0, "support_weak_count": 0,
                    "support_unsupported_count": 0,
                })
                bucket["samples"] += int(mask.sum())
                bucket["position_error_sum_m"] += float(pose_error[mask].sum())
                bucket["u_abs_error_sum_mps"] += float(body_error[mask, 0].sum())
                bucket["v_abs_error_sum_mps"] += float(body_error[mask, 1].sum())
                bucket["yaw_abs_error_sum_rps"] += float(body_error[mask, 2].sum())
                bucket["support_weak_count"] += int(np.sum(scores[mask] > 0.0417017919022596))
                bucket["support_unsupported_count"] += int(np.sum(scores[mask] > 0.16416021114474064))
    for bucket in summary.values():
        n = max(bucket["samples"], 1)
        for key in ("position_error_sum_m", "u_abs_error_sum_mps",
                    "v_abs_error_sum_mps", "yaw_abs_error_sum_rps"):
            bucket[key.removesuffix("_sum_m").removesuffix("_sum_mps").removesuffix("_sum_rps")
                   + "_mean"] = bucket[key] / n
        bucket["support_weak_fraction"] = bucket["support_weak_count"] / n
        bucket["support_unsupported_fraction"] = bucket["support_unsupported_count"] / n
    return summary


def _window_metric_summary(predictions: dict[str, dict[str, np.ndarray]],
                           refs_by_role: dict[str, dict[str, list[tuple[int, int, int]]]]) -> dict[str, Any]:
    run_results: dict[str, Any] = {}
    for role, runs in refs_by_role.items():
        for run_id, refs in sorted(runs.items()):
            pred = predictions[role][run_id]
            max_horizon = pred["body"].shape[1]
            horizon_results = {}
            for horizon_name, horizon in PRIMARY_HORIZONS.items():
                if max_horizon < horizon:
                    continue
                metrics = _per_window_metrics(pred, horizon)
                horizon_results[horizon_name] = {
                    metric: float(np.mean([row[metric] for row in metrics]))
                    for metric in METRICS
                }
            run_results.setdefault(role, {})[run_id] = horizon_results
    return run_results


def _bootstrap(values: dict[str, float], seed: int) -> list[float] | None:
    finite = np.asarray([v for v in values.values() if np.isfinite(v)], dtype=np.float64)
    if not len(finite):
        return None
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(finite), size=(BOOTSTRAP_REPLICATES, len(finite)))
    return np.quantile(finite[draws].mean(axis=1), [0.025, 0.975]).tolist()


def _intervention_report(per_variant: dict[str, Any], run_metrics: dict[str, Any]) -> dict[str, Any]:
    base = run_metrics["M0_native"]
    report: dict[str, Any] = {"per_run_metrics": run_metrics, "paired_deltas": {}}
    for variant, roles in run_metrics.items():
        if variant == "M0_native":
            continue
        for role, runs in roles.items():
            for horizon in PRIMARY_HORIZONS:
                common = sorted(set(runs) & set(base.get(role, {})))
                common = [run for run in common if horizon in runs[run]
                          and horizon in base[role][run]]
                if not common:
                    continue
                for metric in METRICS:
                    delta = {run: runs[run][horizon][metric]
                             - base[role][run][horizon][metric] for run in common}
                    report["paired_deltas"][f"{variant}/{role}/{horizon}/{metric}"] = {
                        "direction": "intervention_minus_M0; negative favors intervention",
                        "per_run": delta,
                        "run_macro_mean": float(np.mean(list(delta.values()))),
                        "run_cluster_bootstrap_95pct_ci": _bootstrap(
                            delta, SEED + len(delta) + sum(map(ord, metric + variant))),
                        "independent_runs": len(delta),
                    }
    return report


def _chronology_report(markers: dict[str, Any], starts: dict[str, Any]) -> dict[str, Any]:
    outputs: dict[str, Any] = {}
    for role, runs in markers.items():
        by_run: dict[str, Any] = {}
        for run_id, records in runs.items():
            pairs = {
                "unsupported_minus_encoder_mean_error": (
                    "support_unsupported_boundary_s",
                    "encoder_pair_mean_error_gt_train_q99_s"),
                "unsupported_minus_encoder_differential_error": (
                    "support_unsupported_boundary_s",
                    "encoder_differential_error_gt_train_q99_s"),
                "unsupported_minus_yaw_error": ("support_unsupported_boundary_s",
                                                "yaw_rate_gt_0p25rps_s"),
                "unsupported_minus_position_0p25m": ("support_unsupported_boundary_s",
                                                      "position_gt_0p25m_s"),
                "unsupported_minus_u_error": ("support_unsupported_boundary_s",
                                              "u_gt_0p25mps_s"),
                "weak_minus_yaw_error": ("support_weak_boundary_s",
                                          "yaw_rate_gt_0p25rps_s"),
            }
            result = {}
            for name, (support_key, error_key) in pairs.items():
                valid = [(r[support_key], r[error_key]) for r in records
                         if r[support_key] is not None and r[error_key] is not None]
                diffs = [error - support for support, error in valid]
                result[name] = {
                    "paired_windows": len(diffs),
                    "error_after_support_fraction": (float(np.mean(np.asarray(diffs) > 0))
                                                       if diffs else None),
                    "median_error_minus_support_s": (float(np.median(diffs))
                                                     if diffs else None),
                    "per_window_error_minus_support_s": diffs,
                }
            labels = starts[role][run_id]
            event_keys = ("support_unsupported_boundary_s",
                          "encoder_pair_mean_error_gt_train_q99_s",
                          "encoder_differential_error_gt_train_q99_s",
                          "yaw_rate_gt_0p25rps_s", "position_gt_0p25m_s")
            first_events = _first_event_labels(records, event_keys)
            by_run[run_id] = {
                "all_starts": result,
                "first_event_counts": {event: first_events.count(event)
                                       for event in sorted(set(first_events))},
                "first_event_count_total": len(first_events),
                "by_start_region": {},
            }
            groups: dict[str, list[int]] = defaultdict(list)
            for i, record in enumerate(labels):
                for axis, value in record["canonical_regime_labels"].items():
                    groups[f"{axis}/{value}"].append(i)
            for region, indices in groups.items():
                regional = {}
                for name, (support_key, error_key) in pairs.items():
                    diffs = [records[i][error_key] - records[i][support_key]
                             for i in indices
                             if records[i][support_key] is not None
                             and records[i][error_key] is not None]
                    regional[name] = {
                        "paired_windows": len(diffs),
                        "error_after_support_fraction": (float(np.mean(np.asarray(diffs) > 0))
                                                           if diffs else None),
                        "median_error_minus_support_s": float(np.median(diffs)) if diffs else None,
                    }
                region_events = [first_events[i] for i in indices]
                regional["first_event_counts"] = {
                    event: region_events.count(event)
                    for event in sorted(set(region_events))}
                regional["first_event_count_total"] = len(region_events)
                by_run[run_id]["by_start_region"][region] = regional
        outputs[role] = by_run
    # Equal-run summaries, never treating overlapping windows as independent.
    run_summaries = {}
    for role, runs in outputs.items():
        for measure in ("unsupported_minus_encoder_mean_error",
                        "unsupported_minus_encoder_differential_error",
                        "unsupported_minus_yaw_error", "unsupported_minus_position_0p25m",
                        "unsupported_minus_u_error", "weak_minus_yaw_error"):
            values = {run: item["all_starts"][measure]["error_after_support_fraction"]
                      for run, item in runs.items()
                      if item["all_starts"][measure]["error_after_support_fraction"] is not None}
            run_summaries[f"{role}/{measure}"] = {
                "run_macro_fraction": float(np.mean(list(values.values()))) if values else None,
                "run_cluster_bootstrap_95pct_ci": _bootstrap(values, SEED + len(values)),
                "per_run": values,
            }
    return {"per_run_and_start_region": outputs, "run_macro_summary": run_summaries}


def _first_event_labels(records: list[dict[str, Any]],
                        event_keys: tuple[str, ...]) -> list[str]:
    labels = []
    for record in records:
        available = [(record[key], key) for key in event_keys
                     if record[key] is not None]
        if not available:
            labels.append("none_within_rollout")
            continue
        earliest = min(value for value, _ in available)
        labels.append("+".join(key for value, key in available
                               if value == earliest))
    return labels


def _training_encoder_error_thresholds(data, model, device: torch.device) -> dict[str, float]:
    from tools.vehicle_dynamics_learning.train_augmented_state_space_plant import _batch_arrays

    errors_mean, errors_diff = [], []
    model.eval()
    with torch.no_grad():
        for run_id, refs in sorted(data.training_windows_80.items()):
            if run_id not in data.training_runs or not refs:
                continue
            if len(refs) > 512:
                refs = [refs[index] for index in np.linspace(
                    0, len(refs) - 1, 512, dtype=np.int64)]
            arrays = _batch_arrays(data, refs, 1)
            tensors = [torch.as_tensor(value,
                dtype=torch.bool if index == 7 else torch.float32, device=device)
                for index, value in enumerate(arrays)]
            history, initial, pose, commands, _, _, truth_encoder, valid, _ = tensors
            model.reset(history, initial, pose)
            _, predicted, _ = model.step(commands[:, 0])
            residual = torch.abs(predicted - truth_encoder[:, 0])
            mask = valid[:, 0]
            if mask.any():
                pair_mean = torch.abs((predicted - truth_encoder[:, 0]).mean(dim=-1))
                differential = torch.abs((predicted - truth_encoder[:, 0])[:, 0]
                                         - (predicted - truth_encoder[:, 0])[:, 1])
                errors_mean.extend(pair_mean[mask].cpu().tolist())
                errors_diff.extend(differential[mask].cpu().tolist())
    if not errors_mean or not errors_diff:
        raise RuntimeError("no valid training encoder residuals for chronology threshold")
    return {
        "threshold_role": "training_only, frozen A2 one-step encoder residuals",
        "quantile": 0.99,
        "pair_mean_abs_error_q99_rad": float(np.quantile(errors_mean, 0.99)),
        "differential_abs_error_q99_rad": float(np.quantile(errors_diff, 0.99)),
        "valid_label_count": len(errors_mean),
    }


def _latent_consistency(data, model, device: torch.device) -> dict[str, Any]:
    horizons = (1, 4, 10, 20, 40, 80)
    ref_groups = {
        "training": data.train_windows_by_horizon[200],
        "development_validation": data.validation_windows,
    }
    selected: dict[str, dict[str, list[tuple[int, int, int]]]] = {}
    from tools.vehicle_dynamics_learning.train_augmented_state_space_plant import _select_eval_windows
    for role, groups in ref_groups.items():
        selected[role] = _select_eval_windows(groups, 64)
    train_latents = []
    all_records: dict[str, Any] = {}
    model.eval()
    with torch.no_grad():
        for role, runs in selected.items():
            all_records[role] = {}
            for run_id, refs in sorted(runs.items()):
                horizon_results = {}
                for k in horizons:
                    chosen = [ref for ref in refs if (
                        int(data.captures[ref[0]].bounds[ref[1], 1])
                        - (int(data.captures[ref[0]].bounds[ref[1], 0]) + ref[2]) - 1 >= k)]
                    if not chosen:
                        continue
                    history, initial, pose, commands, *_ = _make_arrays(data, chosen, k, device)
                    model.reset(history, initial, pose)
                    z_start = model._latent.detach().clone()
                    z_roll = model.rollout(commands)["latents"][:, -1]
                    future_histories = []
                    for capture_index, sequence_index, source_row in chosen:
                        capture = data.captures[capture_index]
                        future_histories.append(
                            __import__("tools.vehicle_dynamics_learning.train_augmented_state_space_plant",
                                fromlist=["_history_at"])._history_at(
                                    capture, sequence_index, source_row + k))
                    future = torch.as_tensor(np.asarray(future_histories),
                                             dtype=torch.float32, device=device)
                    z_encoded = model.history_encoder(
                        (future - model.history_mean) / model.history_scale)
                    if role == "training" and k == 1:
                        train_latents.append(z_encoded.detach().cpu().numpy())
                    difference = (z_roll - z_encoded).cpu().numpy()
                    start = z_start.cpu().numpy()
                    end = z_encoded.cpu().numpy()
                    cosine = np.sum(z_roll.cpu().numpy() * end, axis=1) / np.maximum(
                        np.linalg.norm(z_roll.cpu().numpy(), axis=1)
                        * np.linalg.norm(end, axis=1), 1e-12)
                    region_groups: dict[str, dict[str, list[int]]] = defaultdict(
                        lambda: defaultdict(list))
                    for sample_index, (capture_index, sequence_index, source_row) in enumerate(chosen):
                        capture = data.captures[capture_index]
                        tags, _ = _start_tags(capture, sequence_index, source_row)
                        tag_axes = {
                            "speed": "speed_bin",
                            "steering": "steering_bin",
                            "steering_rate": "steering_rate_bin",
                            "throttle_command": "throttle_command_bin",
                            "throttle_slew": "throttle_slew_bin",
                            "wheel_body_mismatch": "mismatch_bin",
                            "combined": "combined",
                        }
                        for axis, tag_name in tag_axes.items():
                            if tag_name in tags:
                                region_groups[axis][tags[tag_name]].append(sample_index)

                    by_start_region: dict[str, dict[str, Any]] = {}
                    for axis, labels in region_groups.items():
                        by_start_region[axis] = {}
                        for label, indices in sorted(labels.items()):
                            regional_difference = difference[indices]
                            regional_norm = np.linalg.norm(regional_difference, axis=1)
                            regional_per_dimension = np.sqrt(
                                np.mean(regional_difference**2, axis=0))
                            by_start_region[axis][label] = {
                                "sample_count": len(indices),
                                "latent_l2_rmse": float(np.sqrt(
                                    np.mean(np.sum(regional_difference**2, axis=1)))),
                                "latent_l2_median": float(np.median(regional_norm)),
                                "cosine_similarity_mean": float(np.mean(cosine[indices])),
                                "per_dimension_rmse": regional_per_dimension.tolist(),
                            }
                    horizon_results[str(k)] = {
                        "seconds": k * DT_S,
                        "sample_count": len(chosen),
                        "latent_l2_rmse": float(np.sqrt(np.mean(np.sum(difference**2, axis=1)))),
                        "latent_l2_median": float(np.median(np.linalg.norm(difference, axis=1))),
                        "cosine_similarity_mean": float(np.mean(cosine)),
                        "per_dimension_rmse": np.sqrt(np.mean(difference**2, axis=0)).tolist(),
                        "by_start_region": by_start_region,
                        "encoder_vs_rollout_latent_mean": end.mean(axis=0).tolist(),
                        "rollout_latent_mean": z_roll.cpu().numpy().mean(axis=0).tolist(),
                        "within_run_growth_slope_l2_per_s": None,
                    }
                rows = [(float(item["seconds"]), float(item["latent_l2_rmse"]))
                        for item in horizon_results.values()]
                slope = (float(np.polyfit([x for x, _ in rows], [y for _, y in rows], 1)[0])
                         if len(rows) >= 2 else None)
                for item in horizon_results.values():
                    item["within_run_growth_slope_l2_per_s"] = slope
                all_records[role][run_id] = horizon_results
    if not train_latents:
        raise RuntimeError("latent consistency probe produced no training re-encodings")
    latent_pool = np.concatenate(train_latents, axis=0)
    train_std = np.maximum(latent_pool.std(axis=0), 1e-6)
    normalized = {}
    for role, runs in all_records.items():
        normalized[role] = {}
        for run_id, horizons_by_k in runs.items():
            normalized[role][run_id] = {}
            for k, item in horizons_by_k.items():
                per_dim = np.asarray(item["per_dimension_rmse"], dtype=np.float64)
                updated = dict(item)
                updated["per_dimension_normalized_rmse"] = (per_dim / train_std).tolist()
                updated["mean_per_dimension_normalized_rmse"] = float(np.mean(per_dim / train_std))
                updated_regions = {}
                for axis, labels in item["by_start_region"].items():
                    updated_regions[axis] = {}
                    for label, region_item in labels.items():
                        region_updated = dict(region_item)
                        regional_per_dim = np.asarray(
                            region_item["per_dimension_rmse"], dtype=np.float64)
                        normalized_region = regional_per_dim / train_std
                        region_updated["per_dimension_normalized_rmse"] = (
                            normalized_region.tolist())
                        region_updated["mean_per_dimension_normalized_rmse"] = float(
                            np.mean(normalized_region))
                        updated_regions[axis][label] = region_updated
                updated["by_start_region"] = updated_regions
                normalized[role][run_id][k] = updated
    # Adjacent causal history encodings define an empirical state-motion scale.
    adjacent_distances = []
    with torch.no_grad():
        for run_id, refs in selected["training"].items():
            for ref in refs:
                ci, si, row = ref
                cap = data.captures[ci]
                if cap.bounds[si, 1] - (cap.bounds[si, 0] + row) < 2:
                    continue
                from tools.vehicle_dynamics_learning.train_augmented_state_space_plant import _history_at
                hist0 = _history_at(cap, si, row)
                hist1 = _history_at(cap, si, row + 1)
                ht = torch.as_tensor(np.stack((hist0, hist1)), dtype=torch.float32, device=device)
                z = model.history_encoder((ht - model.history_mean) / model.history_scale)
                adjacent_distances.append(float(torch.linalg.vector_norm(z[1] - z[0]).cpu()))
    return {
        "split_roles": ["training", "development_validation"],
        "per_run": normalized,
        "training_latent_dimension_std": train_std.tolist(),
        "adjacent_real_window_encoder_state_distance": {
            "sample_count": len(adjacent_distances),
            "median_l2": float(np.median(adjacent_distances)),
            "mean_l2": float(np.mean(adjacent_distances)),
        },
        "start_selection": "Deterministic run-balanced WP22 horizon windows; up to 64 starts/run",
        "future_truth_policy": "Only causal history ending at t+K is used as the re-encoding target; rollout receives recorded commands only",
    }


def _lead_lag(trace_by_role: dict[str, dict[str, dict[str, np.ndarray]]],
              max_lag_steps: int = 60) -> dict[str, Any]:
    output: dict[str, Any] = {}
    targets = ("yaw_rate_abs_error", "u_abs_error", "position_error_growth")
    for role, runs in trace_by_role.items():
        output[role] = {}
        for run_id, trace in runs.items():
            measurements = trace["encoder"]
            true_measurements = trace["truth_encoder"]
            valid = trace["encoder_valid"]
            pred_body, truth_body = trace["body"], trace["truth_body"]
            pose, truth_pose = trace["pose"], trace["truth_pose"]
            pos = np.linalg.norm(pose[..., :2] - truth_pose[..., :2], axis=-1)
            meas_err = measurements - true_measurements
            pair_mean = np.abs(meas_err.mean(axis=-1))
            differential = np.abs(meas_err[..., 0] - meas_err[..., 1])
            yaw_error = np.abs(pred_body[..., 2] - truth_body[..., 2])
            u_error = np.abs(pred_body[..., 0] - truth_body[..., 0])
            values = {"pair_mean_abs_error": pair_mean,
                      "left_right_differential_abs_error": differential}
            targets_array = {"yaw_rate_abs_error": yaw_error,
                             "u_abs_error": u_error}
            correlations: dict[str, list[float | None]] = {}
            lags = range(max_lag_steps + 1)
            for measurement_name, x in values.items():
                for target_name in targets:
                    correlations[f"{measurement_name}->{target_name}"] = []
            for lag in lags:
                for measurement_name, x in values.items():
                    for target_name in targets:
                        scores = []
                        for b in range(len(x)):
                            end = x.shape[1] - lag
                            if end < 3:
                                continue
                            xx = x[b, :end]
                            if target_name == "position_error_growth":
                                yy = pos[b, lag:lag + end] - pos[b, :end]
                            else:
                                yy = targets_array[target_name][b, lag:lag + end]
                            mask = valid[b, :end] if target_name != "position_error_growth" else valid[b, :end]
                            xx, yy = xx[mask], yy[mask]
                            if len(xx) < 3 or np.std(xx) < 1e-12 or np.std(yy) < 1e-12:
                                continue
                            scores.append(float(np.corrcoef(xx, yy)[0, 1]))
                        correlations[f"{measurement_name}->{target_name}"].append(
                            float(np.mean(scores)) if scores else None)
            output[role][run_id] = {
                "lags_seconds": [i * DT_S for i in lags],
                "per_rollout_mean_pearson_correlation": correlations,
                "window_count": int(len(measurements)),
            }
    return output


def _trace_archive(path: Path, prediction: dict[str, np.ndarray], refs,
                   data, markers: list[dict[str, Any]]) -> None:
    if path.exists():
        with np.load(path, allow_pickle=False) as existing:
            if "first_divergence_markers_json" not in existing.files:
                raise ValueError(f"incomplete WP25 trace archive: {path}")
        return
    run_rows = []
    region_rows = {axis: [] for axis in REGION_KEYS}
    for ci, si, row in refs:
        cap = data.captures[ci]
        begin = int(cap.bounds[si, 0]) + row
        horizon = prediction["body"].shape[1]
        indexes = np.arange(begin + 1, begin + horizon + 1)
        reset_rows = np.full(horizon, cap.sequence_reset[si], dtype=np.int32)
        masks = region_masks(cap.frames[indexes], reset_rows,
                             cap.packet[indexes], np.isfinite(cap.frames[indexes]).all(axis=1), DT_S)
        run_rows.append(_start_record(data, (ci, si, row), "frozen"))
        for axis, names in REGION_KEYS.items():
            labels = np.full(horizon, "", dtype="U48")
            for name in names:
                labels[masks[name]] = name
            region_rows[axis].append(labels)
    heading_error = np.arctan2(np.sin(prediction["pose"][..., 2]
                                     - prediction["truth_pose"][..., 2]),
                               np.cos(prediction["pose"][..., 2]
                                      - prediction["truth_pose"][..., 2]))
    arrays = {key: value for key, value in prediction.items()
              if isinstance(value, np.ndarray)}
    arrays.update({
        "pose_radial_error_m": np.linalg.norm(
            prediction["pose"][..., :2] - prediction["truth_pose"][..., :2], axis=-1),
        "heading_error_rad": heading_error,
        "body_error": prediction["body"] - prediction["truth_body"],
        "encoder_error": prediction["encoder"] - prediction["truth_encoder"],
        "actuator_error": prediction["actuator"] - prediction["truth_actuator"],
        "first_divergence_markers_json": np.asarray(json.dumps(markers)),
        "frozen_start_metadata_json": np.asarray(json.dumps(run_rows)),
    })
    for axis, rows in region_rows.items():
        arrays[f"region_{axis}"] = np.asarray(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def run_wp25(device_name: str = "cpu") -> dict[str, Any]:
    if _git_head() != SOURCE_COMMIT:
        raise RuntimeError(f"expected frozen source commit {SOURCE_COMMIT}; got {_git_head()}")
    torch.set_num_threads(1)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    data, _ = _load_data()
    if not (NEXT_ROOT / "frozen_comparators.json").exists():
        reproduction = _freeze_and_reproduce(data, device)
    else:
        reproduction = json.loads((WP25_ROOT / "wp24_reproduction_report.json").read_text())
        if reproduction.get("status") != "passed":
            raise RuntimeError("WP24 reproduction has not passed")
    starts = json.loads((NEXT_ROOT / "frozen_eval_starts.json").read_text())
    refs_by_role = _frozen_refs(data)
    expected_start_counts = {
        role: {run: len(rows) for run, rows in by_run.items()}
        for role, by_run in starts["split_roles"].items()}
    actual_start_counts = {
        "development_validation": {run: len(rows) for run, rows in data.validation_windows.items()},
        "practice_diagnostic": {run: len(rows) for run, rows in data.practice_windows.items()},
    }
    if expected_start_counts != actual_start_counts:
        raise RuntimeError("frozen exact-start manifest no longer matches WP22 loader")

    comparator = json.loads((NEXT_ROOT / "frozen_comparators.json").read_text())
    model, _ = load_checkpoint(ROOT / comparator["comparators"]["wp22_A2"]["path"], device_name)
    neutral = _neutral_measurement_mean(data)
    encoder_thresholds = _training_encoder_error_thresholds(data, model, device)
    variants: dict[str, dict[str, dict[str, np.ndarray]]] = {name: {} for name in VARIANTS}
    markers: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(dict)
    region_results: dict[str, dict[str, Any]] = {}
    for role, runs in refs_by_role.items():
        for run_index, (run_id, refs) in enumerate(sorted(runs.items()), 1):
            print(f"WP25 trace {role} {run_index}/{len(runs)}: {run_id}", flush=True)
            horizon = min(400 if role == "development_validation" else 200,
                min(int(data.captures[ci].bounds[si, 1]
                    - (int(data.captures[ci].bounds[si, 0]) + row)) - 1
                    for ci, si, row in refs))
            prediction = _rollout(data, model, refs, horizon, "M0_native",
                                  neutral, device, keep_trace=True)
            variants["M0_native"].setdefault(role, {})[run_id] = prediction
            initial_score = prediction["initial_support_score"]
            markers[role][run_id] = _divergence_markers(
                prediction, initial_score, model.config.support_supported_upper,
                model.config.support_weak_upper, encoder_thresholds)
            region_results.setdefault(role, {})[run_id] = _region_summary(data, refs, prediction)
            _trace_archive(WP25_ROOT / "traces" / f"M0_{role}_{run_id}.npz",
                           prediction, refs, data, markers[role][run_id])
    run_metrics = {"M0_native": _window_metric_summary(
        variants["M0_native"], refs_by_role)}

    for variant in VARIANTS[1:]:
        for role, runs in refs_by_role.items():
            if variant == "M4_future_encoder_oracle" and role != "development_validation":
                continue
            for run_index, (run_id, refs) in enumerate(sorted(runs.items()), 1):
                print(f"WP25 {variant} {role} {run_index}/{len(runs)}: {run_id}", flush=True)
                horizon = min(200, min(int(data.captures[ci].bounds[si, 1]
                    - (int(data.captures[ci].bounds[si, 0]) + row)) - 1
                    for ci, si, row in refs))
                prediction = _rollout(data, model, refs, horizon, variant,
                                      neutral, device, keep_trace=False)
                variants[variant].setdefault(role, {})[run_id] = prediction
        run_metrics[variant] = _window_metric_summary(
            variants[variant], {role: refs_by_role[role]
                for role in variants[variant]})

    intervention = _intervention_report(variants, run_metrics)
    oracle_coverage = {}
    for role, runs in variants["M4_future_encoder_oracle"].items():
        valid_rows = sum(int(np.asarray(prediction["encoder_valid"]).sum())
                         for prediction in runs.values())
        total_rows = sum(int(np.asarray(prediction["encoder_valid"]).size)
                         for prediction in runs.values())
        oracle_coverage[role] = {
            "valid_future_encoder_rows": valid_rows,
            "total_rows": total_rows,
            "valid_fraction": valid_rows / max(total_rows, 1),
            "invalid_rows_fallback_to_training_mean": total_rows - valid_rows,
        }
    chronology = _chronology_report(markers, starts["split_roles"])
    latent = _latent_consistency(data, model, device)
    leadlag = _lead_lag(variants["M0_native"])

    report = {
        "schema_version": 1,
        "work_package": "WP25",
        "source_commit": SOURCE_COMMIT,
        "split_roles": ["training", "development_validation", "practice_diagnostic"],
        "simulator_launched": False,
        "model_training_or_optimizer_steps": 0,
        "checkpoint_selection_performed": False,
        "frozen_candidate": comparator["comparators"]["wp22_A2"],
        "wp24_reproduction": {
            "status": reproduction["status"],
            "compared_metric_count": reproduction["compared_metric_count"],
            "maximum_absolute_metric_difference": reproduction[
                "maximum_absolute_metric_difference"],
            "gate_matches": reproduction["gate_matches"],
        },
        "frozen_start_counts": actual_start_counts,
        "measurement_neutral_mean_train_run_balanced": neutral.tolist(),
        "encoder_divergence_thresholds": encoder_thresholds,
        "measurement_feedback_interventions": intervention,
        "M4_future_encoder_oracle_label_coverage": oracle_coverage,
        "support_chronology": chronology,
        "first_divergence_markers_by_run": markers,
        "canonical_operating_region_summary_by_run": region_results,
        "latent_reconstruction_consistency": latent,
        "measurement_body_error_lead_lag": leadlag,
        "primary_horizons_seconds": {name: steps * DT_S
                                      for name, steps in PRIMARY_HORIZONS.items()},
        "intervention_semantics": {
            "M0_native": "Frozen checkpoint and native generated measurement feedback",
            "M1_neutral_measurement": "Latent transition receives training-run-balanced encoder mean",
            "M2_hold_last_measurement": "Latent transition receives prior generated encoder increment",
            "M3_zero_differential": "Latent transition receives [pair_mean, pair_mean]",
            "M4_future_encoder_oracle": "Validation-only next measured encoder labels; invalid rows use neutral mean and are counted by validity",
        },
        "limits": [
            "Validation starts were already used for development gates; results are not untouched confirmation.",
            "Practice comprises two runs and is diagnostic only.",
            "Overlapping windows are summarized within runs; bootstrap resamples independent runs.",
            "M4 uses future encoder labels only as a deliberately invalid causal oracle intervention.",
            "Support boundaries remain calibrated to WP19, not WP22, and are diagnostic rather than error bounds.",
        ],
    }
    _write_json(WP25_ROOT / "wp25_score_only_diagnostics.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--latent-only", action="store_true",
                        help="recompute WP25 latent consistency and update only that report section")
    args = parser.parse_args()
    if args.latent_only:
        if _git_head() != SOURCE_COMMIT:
            raise RuntimeError(f"expected frozen source commit {SOURCE_COMMIT}; got {_git_head()}")
        report_path = WP25_ROOT / "wp25_score_only_diagnostics.json"
        if not report_path.is_file():
            raise FileNotFoundError("existing WP25 report is required for latent-only refresh")
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.set_num_threads(1)
        data, _ = _load_data()
        comparator = json.loads((NEXT_ROOT / "frozen_comparators.json").read_text())
        model, _ = load_checkpoint(ROOT / comparator["comparators"]["wp22_A2"]["path"],
                                   args.device)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["latent_reconstruction_consistency"] = _latent_consistency(
            data, model, device)
        _write_json(report_path, report)
        print(json.dumps({
            "report": str(report_path),
            "latent_horizons_steps": [1, 4, 10, 20, 40, 80],
            "region_breakdowns": "per run, horizon, and canonical start regime",
        }, indent=2, sort_keys=True))
        return 0
    report = run_wp25(args.device)
    print(json.dumps({
        "report": str(WP25_ROOT / "wp25_score_only_diagnostics.json"),
        "wp24_reproduction": report["wp24_reproduction"],
        "model_training_or_optimizer_steps": report["model_training_or_optimizer_steps"],
        "intervention_count": len(report["measurement_feedback_interventions"]["paired_deltas"]),
        "trace_files": len(list((WP25_ROOT / "traces").glob("M0_*.npz"))),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
