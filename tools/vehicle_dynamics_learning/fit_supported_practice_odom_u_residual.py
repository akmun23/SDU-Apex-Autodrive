#!/usr/bin/env python3
"""Fit and score a causal, support-gated practice Odom-u residual.

The observer replay bags contain only states produced from encoder and IMU
inputs. Simulator truth from the source bag is used offline as the residual
label and pose-score reference. Fit the training run only; the validation run
is opened after fitting and is never used to select coefficients.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from tools import evaluate_open_plane_body_dynamics as body
from tools.vehicle_dynamics_learning.score_practice_odom_replay import (
    _align_diagnostics,
    _read_diagnostics,
)
from tools.vehicle_dynamics_learning.score_replayed_production_odom import (
    _read_pose,
)


FEATURE_NAMES = (
    "u_mps", "v_mps", "wheel_mapped_mps", "wheel_packet_mps",
    "pre_wheel_speed_mps", "imu_ax_mps2", "imu_ay_mps2",
    "yaw_rate_radps", "abs_yaw_rate_radps", "yaw_rate_squared",
    "u_abs_yaw_rate", "u_yaw_rate_squared", "ax_abs_yaw_rate",
    "ay_abs_yaw_rate", "wheel_mapped_minus_u_mps",
    "wheel_packet_minus_mapped_mps", "pre_wheel_minus_u_mps", "dt_s",
    "turn_mode", "wheel_update_used", "wheel_burst_rejected",
    "turn_speed_bias_mps",
)
DIAGNOSTIC_SIZE = 42
ALPHA = 0.01
GATE_MIN_U_MPS = 2.0
GATE_MAX_U_MPS = 8.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_run(raw_bag: Path, replay_bag: Path
              ) -> tuple[list[Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    capture = body.load_capture(raw_bag)
    if (capture.aborted or capture.collision_count_start != 0
            or capture.collision_count_end != 0 or capture.timing_faults
            or capture.final_lap_count != 12):
        raise ValueError(f"capture failed clean 12-lap quality gate: {raw_bag}")
    quality_stats = capture.phase_stream_stats or capture.stream_stats
    bad_streams = [
        name for name, (rate_hz, p95_dt_ms, max_gap_ms) in quality_stats.items()
        if rate_hz < 38.0 or p95_dt_ms > 35.0
        or max_gap_ms > (120.0 if name in body.COMMAND_STREAM_TOPICS else 60.0)
    ]
    if bad_streams:
        raise ValueError(f"capture failed 40 Hz stream quality gate: {bad_streams}")

    samples_by_stamp: dict[int, Any] = {}
    group_by_stamp: dict[int, str] = {}
    for sequence_index, sequence in enumerate(capture.sequences):
        label = capture.sequence_labels[sequence_index]
        for sample in sequence:
            stamp = int(sample.source_stamp_ns)
            samples_by_stamp.setdefault(stamp, sample)
            group_by_stamp.setdefault(stamp, label)
    stamps = sorted(samples_by_stamp)
    samples = [samples_by_stamp[stamp] for stamp in stamps]
    diagnostics = _align_diagnostics(
        _read_diagnostics(replay_bag, "/replayed_odom_diagnostics"),
        stamps, "/replayed_odom_diagnostics")
    if diagnostics.shape != (len(stamps), DIAGNOSTIC_SIZE):
        raise ValueError(f"expected {DIAGNOSTIC_SIZE}-value diagnostics")
    poses = _read_pose(replay_bag, "/replayed_odom")
    missing = [stamp for stamp in stamps if stamp not in poses]
    if missing:
        raise ValueError(f"replay misses {len(missing)}/{len(stamps)} pose samples")
    replay_pose = np.asarray([poses[stamp] for stamp in stamps], dtype=np.float64)
    groups = np.asarray([group_by_stamp[stamp] for stamp in stamps])
    if not np.isfinite(diagnostics).all() or not np.isfinite(replay_pose).all():
        raise ValueError("replay contains non-finite diagnostics or pose")
    return samples, diagnostics, groups, replay_pose, np.asarray(stamps, dtype=np.int64)


def _features(diagnostics: np.ndarray) -> np.ndarray:
    u, v = diagnostics[:, 7], diagnostics[:, 8]
    mapped, packet, pre = diagnostics[:, 4], diagnostics[:, 25], diagnostics[:, 40]
    ax, ay, yaw_rate, dt = (
        diagnostics[:, 9], diagnostics[:, 10], diagnostics[:, 11], diagnostics[:, 2])
    return np.column_stack((
        u, v, mapped, packet, pre, ax, ay, yaw_rate, np.abs(yaw_rate),
        yaw_rate ** 2, u * np.abs(yaw_rate), u * yaw_rate ** 2,
        ax * np.abs(yaw_rate), ay * np.abs(yaw_rate), mapped - u,
        packet - mapped, pre - u, dt, diagnostics[:, 13], diagnostics[:, 12],
        diagnostics[:, 26], diagnostics[:, 27],
    ))


def _weights(groups: np.ndarray) -> np.ndarray:
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    result = 1.0 / counts[inverse]
    return result / np.mean(result)


def _fit(features: np.ndarray, residual: np.ndarray,
         groups: np.ndarray, alpha: float = ALPHA
         ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    weights = _weights(groups)
    mean = np.average(features, axis=0, weights=weights)
    scale = np.sqrt(np.maximum(
        np.average((features - mean) ** 2, axis=0, weights=weights), 1.0e-8))
    target_mean = float(np.average(residual, weights=weights))
    normalized = (features - mean) / scale
    gram = normalized.T @ (weights[:, None] * normalized)
    rhs = normalized.T @ (weights * (residual - target_mean))
    coefficient = np.linalg.solve(gram + alpha * np.eye(features.shape[1]), rhs)
    return mean, scale, coefficient, target_mean


def _metrics(error: np.ndarray) -> dict[str, float | int]:
    return {
        "samples": int(len(error)),
        "rmse": float(np.sqrt(np.mean(error ** 2))) if len(error) else 0.0,
        "bias": float(np.mean(error)) if len(error) else 0.0,
        "p95_abs": float(np.quantile(np.abs(error), 0.95)) if len(error) else 0.0,
    }


def _integrate(u: np.ndarray, v: np.ndarray, dt: np.ndarray,
               yaw: np.ndarray, origin_xy: np.ndarray) -> np.ndarray:
    if len(u) < 2:
        raise ValueError("pose integration requires at least two samples")
    dyaw = np.arctan2(np.sin(np.diff(yaw)), np.cos(np.diff(yaw)))
    yaw_mid = yaw[:-1] + 0.5 * dyaw
    u_mid = 0.5 * (u[:-1] + u[1:])
    v_mid = 0.5 * (v[:-1] + v[1:])
    dx = dt[1:] * (u_mid * np.cos(yaw_mid) - v_mid * np.sin(yaw_mid))
    dy = dt[1:] * (u_mid * np.sin(yaw_mid) + v_mid * np.cos(yaw_mid))
    return origin_xy + np.vstack((np.zeros((1, 2)), np.cumsum(np.column_stack((dx, dy)), axis=0)))


def _pose_metrics(samples: list[Any], groups: np.ndarray,
                  diagnostics: np.ndarray, replay_pose: np.ndarray,
                  correction: np.ndarray
                  ) -> dict[str, Any]:
    truth_pose = np.asarray([sample.pose_xyyaw for sample in samples], dtype=np.float64)
    if not np.isfinite(truth_pose).all():
        raise ValueError("source capture lacks finite simulator-truth pose")
    yaw_offset = math.atan2(
        math.sin(truth_pose[0, 2] - replay_pose[0, 2]),
        math.cos(truth_pose[0, 2] - replay_pose[0, 2]))
    yaw = replay_pose[:, 2] + yaw_offset
    dt = diagnostics[:, 2]
    base_u, v = diagnostics[:, 7], diagnostics[:, 8]
    baseline_path = _integrate(base_u, v, dt, yaw, truth_pose[0, :2])
    corrected_path = _integrate(base_u + correction, v, dt, yaw, truth_pose[0, :2])
    # This is an offline pointwise residual integration, not a recursive
    # observer replay; the production-node replay is the recursive test.
    baseline_error = np.linalg.norm(baseline_path - truth_pose[:, :2], axis=1)
    corrected_error = np.linalg.norm(corrected_path - truth_pose[:, :2], axis=1)
    replay_delta = replay_pose[:, :2] - replay_pose[0, :2]
    aligned_replay = truth_pose[0, :2] + np.column_stack((
        np.cos(yaw_offset) * replay_delta[:, 0] - np.sin(yaw_offset) * replay_delta[:, 1],
        np.sin(yaw_offset) * replay_delta[:, 0] + np.cos(yaw_offset) * replay_delta[:, 1],
    ))
    integration_mismatch = np.linalg.norm(baseline_path - aligned_replay, axis=1)
    if float(np.max(integration_mismatch)) > 1.0e-6:
        raise ValueError("independent pose integration does not reproduce replayed Odom")

    per_sequence = []
    labels = list(dict.fromkeys(groups.tolist()))
    for label in labels:
        mask = groups == label
        if not np.any(mask):
            continue
        per_sequence.append({
            "sequence": str(label),
            "samples": int(np.sum(mask)),
            "baseline_position_rmse_m": float(np.sqrt(np.mean(baseline_error[mask] ** 2))),
            "residual_position_rmse_m": float(np.sqrt(np.mean(corrected_error[mask] ** 2))),
            "baseline_endpoint_error_m": float(baseline_error[np.flatnonzero(mask)[-1]]),
            "residual_endpoint_error_m": float(corrected_error[np.flatnonzero(mask)[-1]]),
        })
    return {
        "evaluation": "pointwise residual integrated with production trapezoidal pose equation; not recursive",
        "baseline": _metrics(baseline_error),
        "residual_candidate": _metrics(corrected_error),
        "baseline_endpoint_error_m": float(baseline_error[-1]),
        "residual_endpoint_error_m": float(corrected_error[-1]),
        "baseline_integration_max_disagreement_with_replay_m": float(np.max(integration_mismatch)),
        "per_lap": per_sequence,
    }


def _evaluate(samples: list[Any], diagnostics: np.ndarray, groups: np.ndarray,
              replay_pose: np.ndarray, mean: np.ndarray, scale: np.ndarray,
              coefficient: np.ndarray, intercept: float) -> dict[str, Any]:
    features = _features(diagnostics)
    baseline_u = diagnostics[:, 7]
    truth_u = np.asarray([sample.state[0] for sample in samples])
    prediction = ((features - mean) / scale) @ coefficient + intercept
    gate = (baseline_u >= GATE_MIN_U_MPS) & (baseline_u < GATE_MAX_U_MPS)
    correction = prediction * gate
    baseline_error = baseline_u - truth_u
    candidate_error = baseline_error + correction

    speed = np.hypot(
        truth_u, np.asarray([sample.state[1] for sample in samples]))
    speed_regions = {}
    for name, mask in {
        "0_to_2_mps": speed < 2.0,
        "2_to_4_mps": (speed >= 2.0) & (speed < 4.0),
        "4_to_6_mps": (speed >= 4.0) & (speed < 6.0),
        "6_to_8_mps": (speed >= 6.0) & (speed < 8.0),
        "8_to_12_mps": (speed >= 8.0) & (speed <= 12.0),
    }.items():
        speed_regions[name] = {
            "baseline": _metrics(baseline_error[mask]),
            "candidate": _metrics(candidate_error[mask]),
        }

    per_group = []
    for group in dict.fromkeys(groups.tolist()):
        mask = groups == group
        before = _metrics(baseline_error[mask])
        after = _metrics(candidate_error[mask])
        per_group.append({
            "group": str(group),
            "baseline_rmse_mps": before["rmse"],
            "candidate_rmse_mps": after["rmse"],
            "improved": bool(after["rmse"] < before["rmse"]),
        })

    pose = _pose_metrics(samples, groups, diagnostics, replay_pose, correction)
    return {
        "samples": len(samples),
        "gate_active_samples": int(np.sum(gate)),
        "gate_active_fraction": float(np.mean(gate)),
        "correction_active_quantiles_mps": np.quantile(
            correction[gate], [0.0, 0.01, 0.5, 0.99, 1.0]).tolist(),
        "forward_speed": {
            "baseline": _metrics(baseline_error),
            "candidate": _metrics(candidate_error),
            "by_truth_speed": speed_regions,
        },
        "sequence_macro": {
            "group_count": len(per_group),
            "groups_improved": sum(item["improved"] for item in per_group),
            "per_group": per_group,
        },
        "pose": pose,
    }


def fit_and_score(train_raw: Path, train_replay: Path,
                  validation_raw: Path, validation_replay: Path,
                  output_dir: Path, overwrite: bool = False) -> dict[str, Any]:
    return fit_and_score_multiple(
        [(train_raw, train_replay)], [(validation_raw, validation_replay)],
        output_dir, overwrite)


def fit_and_score_multiple(
        train_pairs: list[tuple[Path, Path]],
        validation_pairs: list[tuple[Path, Path]],
        output_dir: Path, overwrite: bool = False) -> dict[str, Any]:
    if not train_pairs or not validation_pairs:
        raise ValueError("at least one complete training and validation pair is required")
    expected_outputs = {"fit_and_validation_report.json", "odom_u_residual_model.json"}
    if output_dir.exists() and any(output_dir.iterdir()):
        existing = {item.name for item in output_dir.iterdir()}
        if not overwrite or existing != expected_outputs:
            raise FileExistsError(
                f"refusing to overwrite output directory with files {sorted(existing)}")

    train_data = []
    for raw_bag, replay_bag in train_pairs:
        samples, diagnostics, groups, _, stamps = _load_run(raw_bag, replay_bag)
        run_id = raw_bag.parent.parent.name
        groups = np.asarray([f"{run_id}:{group}" for group in groups])
        train_data.append((samples, diagnostics, groups, stamps))
    train_samples = [sample for samples, _, _, _ in train_data for sample in samples]
    train_diagnostics = np.concatenate([diagnostics for _, diagnostics, _, _ in train_data])
    train_groups = np.concatenate([groups for _, _, groups, _ in train_data])
    train_stamps = np.concatenate([stamps for _, _, _, stamps in train_data])
    train_features = _features(train_diagnostics)
    train_baseline_u = train_diagnostics[:, 7]
    train_truth_u = np.asarray([sample.state[0] for sample in train_samples])
    train_residual = train_truth_u - train_baseline_u
    mean, scale, coefficient, target_mean = _fit(
        train_features, train_residual, train_groups)
    raw_coefficient = coefficient / scale
    raw_intercept = target_mean - float(np.dot(mean, raw_coefficient))

    # Leave one whole lap sequence out at a time; never split adjacent samples.
    heldout_prediction = np.zeros(len(train_residual), dtype=np.float64)
    cross_validation = []
    for group in dict.fromkeys(train_groups.tolist()):
        training = train_groups != group
        heldout = ~training
        fold_mean, fold_scale, fold_coefficient, fold_intercept = _fit(
            train_features[training], train_residual[training],
            train_groups[training])
        heldout_prediction[heldout] = (
            ((train_features[heldout] - fold_mean) / fold_scale)
            @ fold_coefficient + fold_intercept)
        cross_validation.append({
            "group": str(group),
            "baseline_rmse_mps": _metrics(train_residual[heldout])["rmse"],
            "candidate_rmse_mps": _metrics(
                train_residual[heldout] - heldout_prediction[heldout])["rmse"],
        })

    train_gate = ((train_baseline_u >= GATE_MIN_U_MPS)
                  & (train_baseline_u < GATE_MAX_U_MPS))
    train_prediction = ((train_features - mean) / scale) @ coefficient + target_mean
    train_correction = train_prediction * train_gate
    cv_error = train_residual - heldout_prediction * train_gate

    output_dir.mkdir(parents=True, exist_ok=True)
    train_ids = [raw.parent.parent.name for raw, _ in train_pairs]
    validation_ids = [raw.parent.parent.name for raw, _ in validation_pairs]
    model = {
        "schema_version": 1,
        "model_id": "practice_u_residual_" + "_and_".join(train_ids) +
                    "_validated_" + "_and_".join(validation_ids),
        "target": "simulator_truth_rear_axle_u_minus_current_observer_u_mps",
        "feature_names": list(FEATURE_NAMES),
        "feature_evaluation": "linear dot product on unnormalized causal feature values",
        "intercept_mps": raw_intercept,
        "coefficients": raw_coefficient.tolist(),
        "activation_gate": {
            "baseline_observer_u_min_mps": GATE_MIN_U_MPS,
            "baseline_observer_u_max_exclusive_mps": GATE_MAX_U_MPS,
            "reason": "training capture has negligible support below 2 m/s and above 8 m/s",
        },
        "ridge_alpha_on_standardized_features": ALPHA,
        "training": {
            "raw_bags": [{"path": str(raw.resolve()), "sha256": _sha256(raw)}
                         for raw, _ in train_pairs],
            "replay_bags": [{"path": str(replay.resolve()),
                             "sha256": _sha256(replay)}
                            for _, replay in train_pairs],
            "sample_count": len(train_samples),
            "whole_lap_group_count": int(len(np.unique(train_groups))),
            "first_last_source_stamp_ns": [int(np.min(train_stamps)),
                                           int(np.max(train_stamps))],
            "ground_truth_policy": "offline target only; never a runtime feature",
        },
        "runtime_inputs": "current observer state and its existing encoder/IMU-derived diagnostics only",
    }
    (output_dir / "odom_u_residual_model.json").write_text(
        json.dumps(model, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")

    validations = []
    for validation_raw, validation_replay in validation_pairs:
        validation_samples, validation_diagnostics, validation_groups, validation_pose, _ = _load_run(
            validation_raw, validation_replay)
        validations.append({
            "run_id": validation_raw.parent.parent.name,
            "raw_bag": str(validation_raw.resolve()),
            "replay_bag": str(validation_replay.resolve()),
            "metrics": _evaluate(
                validation_samples, validation_diagnostics, validation_groups,
                validation_pose, mean, scale, coefficient, target_mean),
        })
    # Training pose is intentionally omitted from the fit report; only heldout
    # recursive observer replay can establish deployable pose accuracy.
    report = {
        "validation_policy": "whole independent run; coefficients fixed from training only",
        "training_run": str(train_pairs[0][0].resolve()) if len(train_pairs) == 1 else None,
        "training_runs": [str(raw.resolve()) for raw, _ in train_pairs],
        "validation_runs": [item["raw_bag"] for item in validations],
        "feature_names": list(FEATURE_NAMES),
        "training": {
            "samples": len(train_samples),
            "groups": len(np.unique(train_groups)),
            "baseline_one_step_rmse_mps": _metrics(train_residual)["rmse"],
            "leave_one_sequence_out": {
                "baseline_macro_rmse_mps": float(np.mean([
                    item["baseline_rmse_mps"] for item in cross_validation])),
                "candidate_macro_rmse_mps": float(np.mean([
                    item["candidate_rmse_mps"] for item in cross_validation])),
                "groups_improved": int(sum(
                    item["candidate_rmse_mps"] < item["baseline_rmse_mps"]
                    for item in cross_validation)),
                "per_group": cross_validation,
            },
            "full_fit_active_correction_quantiles_mps": np.quantile(
                train_correction[train_gate], [0.0, 0.01, 0.5, 0.99, 1.0]).tolist(),
            "full_fit_one_step_rmse_mps": _metrics(
                train_residual - train_correction)["rmse"],
            "heldout_cv_one_step_rmse_mps": _metrics(cv_error)["rmse"],
            "speed_gate_active_samples": int(np.sum(train_gate)),
        },
        "validation": validations[0]["metrics"] if len(validations) == 1 else None,
        "validation_runs": validations,
        "source_provenance": {
            "training": [{
                "replay_metadata": str((replay.parent.parent / "replay_metadata.json").resolve()),
                "config_sha256": json.loads(
                    (replay.parent.parent / "replay_metadata.json").read_text()
                )["localization_source_sha256"]["f1tenth_localization/config/sensor_odometry.yaml"],
            } for _, replay in train_pairs],
            "validation": [{
                "replay_metadata": str((replay.parent.parent / "replay_metadata.json").resolve()),
                "config_sha256": json.loads(
                    (replay.parent.parent / "replay_metadata.json").read_text()
                )["localization_source_sha256"]["f1tenth_localization/config/sensor_odometry.yaml"],
            } for _, replay in validation_pairs],
        },
        "caveats": [
            "The validation pose score is a pointwise residual integrated through the production pose equation, not recursive observer output.",
            "Speed support gate is based only on the current observer estimate; 0-2 and 8+ m/s are deliberately uncorrected.",
            "This model is a candidate and must pass recursive replay and a fresh real-drive Odom evaluation before production activation.",
        ],
    }
    (output_dir / "fit_and_validation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    return report


def _integrated_u_correction_design(
        features: np.ndarray, diagnostics: np.ndarray,
        replay_pose: np.ndarray, samples: list[Any],
        mean: np.ndarray, scale: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map the causal per-sample residual coefficients to full-run XY error."""
    baseline_u = diagnostics[:, 7]
    gate = ((baseline_u >= GATE_MIN_U_MPS) &
            (baseline_u < GATE_MAX_U_MPS)).astype(np.float64)
    normalized = np.column_stack((
        np.ones(len(features)), (features - mean) / scale))
    gated_design = normalized * gate[:, None]

    truth_pose = np.asarray([sample.pose_xyyaw for sample in samples],
                            dtype=np.float64)
    if not np.isfinite(truth_pose).all():
        raise ValueError("source capture lacks finite simulator-truth pose")
    yaw_offset = math.atan2(
        math.sin(truth_pose[0, 2] - replay_pose[0, 2]),
        math.cos(truth_pose[0, 2] - replay_pose[0, 2]))
    yaw = replay_pose[:, 2] + yaw_offset
    baseline_path = _integrate(
        baseline_u, diagnostics[:, 8], diagnostics[:, 2], yaw,
        truth_pose[0, :2])
    position_target = truth_pose[:, :2] - baseline_path

    position_design = np.zeros((2 * len(samples), gated_design.shape[1]))
    for index in range(1, len(samples)):
        dyaw = math.atan2(math.sin(yaw[index] - yaw[index - 1]),
                          math.cos(yaw[index] - yaw[index - 1]))
        yaw_mid = yaw[index - 1] + 0.5 * dyaw
        scale_dt = 0.5 * diagnostics[index, 2]
        contribution = scale_dt * (gated_design[index - 1] + gated_design[index])
        position_design[2 * index] = position_design[2 * (index - 1)] + (
            math.cos(yaw_mid) * contribution)
        position_design[2 * index + 1] = position_design[2 * (index - 1) + 1] + (
            math.sin(yaw_mid) * contribution)
    return position_design, position_target.reshape(-1)


def fit_path_aware_and_score(
        train_pairs: list[tuple[Path, Path]],
        validation_pairs: list[tuple[Path, Path]],
        output_dir: Path, position_loss_weight: float = 1.0,
        overwrite: bool = False) -> dict[str, Any]:
    """Fit a sensor-only residual against speed and complete-run pose loss."""
    if not train_pairs or not validation_pairs:
        raise ValueError("path-aware fitting requires training and validation runs")
    if not math.isfinite(position_loss_weight) or position_loss_weight <= 0.0:
        raise ValueError("position-loss weight must be finite and positive")
    expected_outputs = {"fit_and_validation_report.json", "odom_u_residual_model.json"}
    if output_dir.exists() and any(output_dir.iterdir()):
        existing = {item.name for item in output_dir.iterdir()}
        if not overwrite or existing != expected_outputs:
            raise FileExistsError(
                f"refusing to overwrite output directory with files {sorted(existing)}")

    training_data = []
    for raw_bag, replay_bag in train_pairs:
        samples, diagnostics, groups, replay_pose, _ = _load_run(raw_bag, replay_bag)
        run_id = raw_bag.parent.parent.name
        groups = np.asarray([f"{run_id}:{group}" for group in groups])
        training_data.append((samples, diagnostics, groups, replay_pose))
    features_by_run = [_features(item[1]) for item in training_data]
    all_features = np.concatenate(features_by_run)
    all_groups = np.concatenate([item[2] for item in training_data])
    fit_weights = _weights(all_groups)
    mean = np.average(all_features, axis=0, weights=fit_weights)
    scale = np.sqrt(np.maximum(
        np.average((all_features - mean) ** 2, axis=0, weights=fit_weights), 1.0e-8))

    speed_designs = []
    speed_targets = []
    position_designs = []
    position_targets = []
    baseline_speed_rmse = []
    baseline_position_rmse = []
    for (samples, diagnostics, _, replay_pose), features in zip(
            training_data, features_by_run):
        baseline_u = diagnostics[:, 7]
        gate = ((baseline_u >= GATE_MIN_U_MPS) &
                (baseline_u < GATE_MAX_U_MPS)).astype(np.float64)
        normalized = np.column_stack((
            np.ones(len(features)), (features - mean) / scale))
        speed_design = normalized * gate[:, None]
        speed_target = np.asarray([sample.state[0] for sample in samples]) - baseline_u
        position_design, position_target = _integrated_u_correction_design(
            features, diagnostics, replay_pose, samples, mean, scale)
        position_error_scale = math.sqrt(
            np.mean(position_target.reshape(-1, 2).__pow__(2).sum(axis=1)))
        speed_error_scale = math.sqrt(np.mean(speed_target ** 2))
        if speed_error_scale <= 1.0e-9 or position_error_scale <= 1.0e-9:
            raise ValueError("training run has degenerate speed or pose residual")
        run_scale = math.sqrt(len(training_data))
        speed_designs.append(
            speed_design / (speed_error_scale * math.sqrt(len(samples)) * run_scale))
        speed_targets.append(
            speed_target / (speed_error_scale * math.sqrt(len(samples)) * run_scale))
        position_designs.append(
            position_design / (position_error_scale *
                               math.sqrt(2 * len(samples)) * run_scale))
        position_targets.append(
            position_target / (position_error_scale *
                               math.sqrt(2 * len(samples)) * run_scale))
        baseline_speed_rmse.append(speed_error_scale)
        baseline_position_rmse.append(position_error_scale)

    design = np.vstack((np.vstack(speed_designs),
                        math.sqrt(position_loss_weight) *
                        np.vstack(position_designs)))
    target = np.concatenate((np.concatenate(speed_targets),
                             math.sqrt(position_loss_weight) *
                             np.concatenate(position_targets)))
    penalty = ALPHA * np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    parameters = np.linalg.solve(design.T @ design + penalty, design.T @ target)
    raw_coefficients = parameters[1:] / scale
    raw_intercept = float(parameters[0] - np.dot(mean, raw_coefficients))

    validation_runs = []
    for raw_bag, replay_bag in validation_pairs:
        samples, diagnostics, groups, replay_pose, _ = _load_run(raw_bag, replay_bag)
        metrics = _evaluate(
            samples, diagnostics, groups, replay_pose,
            mean, scale, parameters[1:], float(parameters[0]))
        validation_runs.append({
            "run_id": raw_bag.parent.parent.name,
            "raw_bag": str(raw_bag.resolve()),
            "replay_bag": str(replay_bag.resolve()),
            "metrics": metrics,
        })

    train_ids = [raw.parent.parent.name for raw, _ in train_pairs]
    validation_ids = [raw.parent.parent.name for raw, _ in validation_pairs]
    model = {
        "schema_version": 1,
        "model_id": "practice_u_residual_path_" + "_and_".join(train_ids) +
                    "_validated_" + "_and_".join(validation_ids),
        "target": "simulator_truth_rear_axle_u_minus_current_observer_u_mps",
        "feature_names": list(FEATURE_NAMES),
        "feature_evaluation": "linear dot product on unnormalized causal feature values",
        "intercept_mps": raw_intercept,
        "coefficients": raw_coefficients.tolist(),
        "activation_gate": {
            "baseline_observer_u_min_mps": GATE_MIN_U_MPS,
            "baseline_observer_u_max_exclusive_mps": GATE_MAX_U_MPS,
            "reason": "training captures have support from 2 to 8 m/s; outside remains uncorrected",
        },
        "training_objective": {
            "type": "equal relative forward-speed RMSE and one-alignment full-run XY RMSE",
            "position_loss_weight": position_loss_weight,
            "ridge_alpha_on_standardized_features": ALPHA,
        },
        "training": {
            "raw_bags": [{"path": str(raw.resolve()), "sha256": _sha256(raw)}
                         for raw, _ in train_pairs],
            "replay_bags": [{"path": str(replay.resolve()), "sha256": _sha256(replay)}
                            for _, replay in train_pairs],
            "sample_count": int(len(all_features)),
            "whole_lap_group_count": int(len(np.unique(all_groups))),
            "baseline_speed_rmse_per_run_mps": baseline_speed_rmse,
            "baseline_position_rmse_per_run_m": baseline_position_rmse,
            "ground_truth_policy": "offline targets only; runtime uses encoder/IMU-derived features",
        },
        "runtime_inputs": "current observer state and its existing encoder/IMU-derived diagnostics only",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "odom_u_residual_model.json").write_text(
        json.dumps(model, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    report = {
        "model_id": model["model_id"],
        "training_objective": model["training_objective"],
        "training_runs": model["training"]["raw_bags"],
        "validation_policy": "independent complete runs; recursive replay required before promotion",
        "validation_runs": validation_runs,
        "caveats": [
            "Training uses simulator truth only to form offline speed and full-run path residual targets.",
            "The path-aware fit is still a candidate; validation metrics are pointwise observer replay, not this candidate's recursive replay.",
            "Speed residual is disabled outside the measured 2-8 m/s support band.",
        ],
    }
    (output_dir / "fit_and_validation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-raw", type=Path)
    parser.add_argument("--train-replay", type=Path)
    parser.add_argument("--validation-raw", type=Path)
    parser.add_argument("--validation-replay", type=Path)
    parser.add_argument(
        "--train-pair", type=Path, nargs=2, action="append", metavar=("RAW_BAG", "REPLAY_BAG"),
        help="additional independent training run; repeat to pool run-level data",
    )
    parser.add_argument(
        "--validation-pair", type=Path, nargs=2, action="append", metavar=("RAW_BAG", "REPLAY_BAG"),
        help="independent whole-run validation; repeat to evaluate transfer",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--overwrite-existing-artifacts", action="store_true",
        help="replace only this tool's two known output files in an otherwise isolated output directory",
    )
    parser.add_argument(
        "--objective", choices=("samplewise", "path-aware"), default="samplewise",
        help="fit per-sample speed residuals or jointly fit speed and integrated full-run XY error",
    )
    parser.add_argument(
        "--position-loss-weight", type=float, default=1.0,
        help="relative weight for full-run XY loss in path-aware mode",
    )
    args = parser.parse_args()
    legacy_paths = (args.train_raw, args.train_replay,
                    args.validation_raw, args.validation_replay)
    if args.train_pair or args.validation_pair:
        if any(path is not None for path in legacy_paths):
            parser.error("use either --train-pair/--validation-pair or the four single-run path options")
        train_pairs = [(raw.resolve(), replay.resolve())
                       for raw, replay in (args.train_pair or [])]
        validation_pairs = [(raw.resolve(), replay.resolve())
                            for raw, replay in (args.validation_pair or [])]
    else:
        if any(path is None for path in legacy_paths):
            parser.error("single-run mode requires --train-raw, --train-replay, --validation-raw, and --validation-replay")
        train_pairs = [(args.train_raw.resolve(), args.train_replay.resolve())]
        validation_pairs = [(args.validation_raw.resolve(), args.validation_replay.resolve())]
    if args.objective == "path-aware":
        report = fit_path_aware_and_score(
            train_pairs, validation_pairs, args.output_dir.resolve(),
            args.position_loss_weight, args.overwrite_existing_artifacts)
        summary = {
            "model_id": report["model_id"],
            "training_objective": report["training_objective"],
            "validation_runs": [{
                "run_id": item["run_id"],
                "forward_speed": item["metrics"]["forward_speed"],
                "pose": item["metrics"]["pose"],
            } for item in report["validation_runs"]],
        }
    else:
        report = fit_and_score_multiple(
            train_pairs, validation_pairs, args.output_dir.resolve(),
            args.overwrite_existing_artifacts)
        summary = {
            "training_cv": report["training"]["leave_one_sequence_out"],
            "validation_runs": [{
                "run_id": item["run_id"],
                "forward_speed": item["metrics"]["forward_speed"],
                "pose": item["metrics"]["pose"],
            } for item in report["validation_runs"]],
        }
    summary["model"] = str((args.output_dir / "odom_u_residual_model.json").resolve())
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
