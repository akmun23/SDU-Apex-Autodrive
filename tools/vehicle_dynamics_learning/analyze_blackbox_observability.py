#!/usr/bin/env python3
"""WP18 run-balanced matched-history analysis for black-box observability.

No model is trained. Current/past rows form the matching vectors; future
simulator and encoder values are scoring responses only. Test/final-test runs
are excluded from both queries and neighbor banks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from tools.vehicle_dynamics_learning.effective_race_teacher import (
    DT_S,
)
from tools.vehicle_dynamics_learning.operating_regions import region_masks
from tools.vehicle_dynamics_learning.signal_semantics import WHEEL_RADIUS_M
from tools.vehicle_dynamics_learning.train_nssm import _load_dataset


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = (ROOT / "live_runs/derived_dynamics_learning_20260928"
    / "full_modeling_reset_20261001")
PARENT_ROOT = DATA_ROOT / "replacement_offline_sim_raceline_20261002" \
    / "encoder_raw_state_teacher_v1"
FIXED_ROOT = DATA_ROOT / "replacement_offline_sim_raceline_20261003" \
    / "full_throttle_domain_v1/encoder_fixed40hz_v1"
DEFAULT_DYNAMIC = PARENT_ROOT / "openplane_dynamics_raw_wheels.npz"
DEFAULT_PRACTICE = PARENT_ROOT / "practice_dynamics_raw_wheels.npz"
DEFAULT_DYNAMIC_FIXED = FIXED_ROOT / "openplane_dynamics_raw_wheels_fixed40hz.npz"
DEFAULT_PRACTICE_FIXED = FIXED_ROOT / "practice_dynamics_raw_wheels_fixed40hz.npz"
DEFAULT_WP17 = FIXED_ROOT.parent / "wp17_frozen_parent_interventions_v4.json"
DEFAULT_OUTPUT = FIXED_ROOT.parent / "blackbox_observability_report_v1.json"

HISTORY_LENGTHS = {
    "0ms": 0, "100ms": 4, "250ms": 10, "500ms": 20,
    "1s": 40, "2s": 80, "4s": 160,
}
FUTURE_STEPS = {"25ms": 1, "100ms": 4, "250ms": 10, "500ms": 20,
                "750ms": 30, "2s": 80}
QUERY_SPLITS = {"validation", "unseen_practice"}
CANDIDATE_SPLITS = {"train", "validation", "unseen_practice"}
QUERY_CAP_PER_RUN_REGION = 200
CANDIDATE_CAP_PER_RUN = 1800
CANDIDATE_EXTRA_PER_RUN_REGION = 160
NEIGHBOR_ROWS_PER_RUN = 4
NEIGHBOR_INDEPENDENT_RUNS = 5
FEATURE_VARIANTS = ("body_actuator", "body_actuator_plus_filtered_wheel")


@dataclass
class SequenceData:
    run_id: str
    split: str
    frames: np.ndarray
    packet: np.ndarray
    reset: np.ndarray
    state: np.ndarray
    pose: np.ndarray
    raw_wheel: np.ndarray
    raw_valid: np.ndarray
    core_features: np.ndarray
    filtered_features: np.ndarray
    responses: np.ndarray
    response_names: tuple[str, ...]
    eligible: np.ndarray
    region_masks: dict[str, np.ndarray]
    source_path: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _wrap_angle(angle: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(angle), np.cos(angle))


def _assert_fixed_view_rows(source: dict[str, Any], fixed: dict[str, Any],
                            label: str) -> None:
    for key in ("run_ids", "splits", "bounds", "seq_run", "packet_sequence",
                "dt_s", "frames", "simulator_rigid_state",
                "simulator_pose_xyyaw", "sample_time_ns"):
        a, b = np.asarray(source[key]), np.asarray(fixed[key])
        if a.shape != b.shape or not np.array_equal(a, b):
            raise ValueError(f"{label}: fixed-40Hz sidecar row mismatch at {key}")
    if "encoder_raw_surface_mps" not in fixed or "encoder_raw_valid" not in fixed:
        raise ValueError(f"{label}: fixed-cadence encoder fields are missing")


def _future_responses(state: np.ndarray, frames: np.ndarray,
                      pose: np.ndarray, raw_wheel: np.ndarray,
                      raw_valid: np.ndarray
                      ) -> tuple[np.ndarray, tuple[str, ...]]:
    """Construct label-only responses at each source row; no future inputs."""
    count = len(state)
    names: list[str] = []
    columns: list[np.ndarray] = []

    # Effective COM accelerations follow the explicit rigid-body equations:
    # du/dt = ax + r*v and dv/dt = ay - r*u.
    ax = np.full(count, np.nan)
    ay = np.full(count, np.nan)
    yaw_accel = np.full(count, np.nan)
    if count > 1:
        du = np.diff(state[:, 0]) / DT_S
        dv = np.diff(state[:, 1]) / DT_S
        dr = np.diff(state[:, 2]) / DT_S
        ax[:-1] = du - state[:-1, 2] * state[:-1, 1]
        ay[:-1] = dv + state[:-1, 2] * state[:-1, 0]
        yaw_accel[:-1] = dr
    accel = np.column_stack((ax, ay, yaw_accel))
    accel_names = ("effective_ax_mps2", "effective_ay_mps2",
                   "yaw_accel_rps2")
    prefix = np.vstack((np.zeros((1, 3)), np.nancumsum(
        np.nan_to_num(accel, nan=0.0), axis=0)))
    finite_accel = np.isfinite(accel).all(axis=1).astype(np.int32)
    finite_prefix = np.r_[0, np.cumsum(finite_accel)]
    for duration, steps in FUTURE_STEPS.items():
        if steps not in (1, 4, 10, 20):
            continue
        valid = np.zeros(count, dtype=bool)
        available = max(count - steps, 0)
        mean = np.full((count, 3), np.nan, dtype=np.float64)
        if available:
            valid[:available] = (
                (finite_prefix[steps:count] - finite_prefix[:available])
                == steps)
            mean[:available] = (
                prefix[steps:count] - prefix[:available]) / steps
        for index, name in enumerate(accel_names):
            values = np.full(count, np.nan)
            values[:available] = mean[:available, index]
            values[~valid] = np.nan
            columns.append(values)
            names.append(f"{name}_mean_{duration}")

    for duration in ("250ms", "750ms", "2s"):
        steps = FUTURE_STEPS[duration]
        target = np.full((count, 3), np.nan)
        if count > steps:
            target[:count - steps] = state[steps:, :3] - state[:-steps, :3]
        for index, name in enumerate(("delta_u_mps", "delta_v_mps",
                                      "delta_yaw_rate_rps")):
            columns.append(target[:, index])
            names.append(f"{name}_{duration}")
        heading = np.full(count, np.nan)
        if count > steps:
            heading[:count - steps] = _wrap_angle(
                pose[steps:, 2] - pose[:-steps, 2])
        columns.append(heading)
        names.append(f"heading_change_rad_{duration}")
        wheel = np.full((count, 2), np.nan)
        if count > steps:
            wheel[:count - steps] = frames[steps:, 5:7]
        columns.extend((wheel[:, 0], wheel[:, 1]))
        names.extend((f"filtered_wheel_left_mps_{duration}",
                      f"filtered_wheel_right_mps_{duration}"))

    angle_increment = np.full((count, 2), np.nan)
    if count > 1:
        angle_increment[:-1] = raw_wheel[1:] * DT_S / WHEEL_RADIUS_M
        angle_increment[:-1][~raw_valid[1:]] = np.nan
    columns.extend((angle_increment[:, 0], angle_increment[:, 1]))
    names.extend(("fixed25_encoder_angle_increment_left_rad",
                  "fixed25_encoder_angle_increment_right_rad"))
    return np.column_stack(columns).astype(np.float64), tuple(names)


def _edge_validity(packet: np.ndarray, dt: np.ndarray,
                   reset: np.ndarray, finite: np.ndarray) -> np.ndarray:
    if len(packet) < 2:
        return np.zeros(0, dtype=bool)
    return ((np.diff(packet) == 1) & (reset[1:] == reset[:-1])
            & np.isclose(dt[1:], DT_S, rtol=0.0, atol=1e-7)
            & finite[1:] & finite[:-1])


def _valid_candidate_rows(sequence: SequenceData, dt: np.ndarray,
                          history_steps: int = 160,
                          future_steps: int = 80) -> np.ndarray:
    finite = (np.isfinite(sequence.core_features).all(axis=1)
              & np.isfinite(sequence.filtered_features).all(axis=1)
              & np.isfinite(sequence.state[:, :3]).all(axis=1)
              & np.isfinite(sequence.pose).all(axis=1))
    edges = _edge_validity(sequence.packet, dt, sequence.reset, finite)
    n = len(sequence.frames)
    # An invalid edge splits one capture into independent causal segments.
    edge_bad_prefix = np.r_[0, np.cumsum(~edges)]
    rows = np.arange(n, dtype=np.int64)
    start = rows - history_steps
    end_edge = rows + future_steps
    in_bounds = (start >= 0) & (end_edge < n)
    valid = np.zeros(n, dtype=bool)
    selected = rows[in_bounds]
    # Edges [t-H, t+future_steps) must all be contiguous and valid.
    valid[in_bounds] = ((edge_bad_prefix[end_edge[in_bounds]]
                         - edge_bad_prefix[start[in_bounds]]) == 0)
    valid &= finite
    return rows[valid]


def _sequence_views(source_data: dict[str, Any], fixed_data: dict[str, Any],
                    dataset_path: Path, allowed_splits: set[str]
                    ) -> list[SequenceData]:
    frames_all = np.asarray(source_data["frames"], dtype=np.float64)
    rigid = np.asarray(source_data["simulator_rigid_state"], dtype=np.float64)
    truth_body_all = np.column_stack((rigid[:, 7], rigid[:, 8], rigid[:, 12]))
    pose_all = np.asarray(source_data["simulator_pose_xyyaw"], dtype=np.float64)
    raw_all = np.asarray(fixed_data["encoder_raw_surface_mps"], dtype=np.float64)
    raw_valid_all = np.asarray(fixed_data["encoder_raw_valid"], dtype=bool)
    dt_all = np.asarray(source_data["dt_s"], dtype=np.float64)
    packet_all = np.asarray(source_data["packet_sequence"], dtype=np.int64)
    reset_all = np.asarray(source_data["sequence_reset_index"], dtype=np.int64)
    run_ids = np.asarray(source_data["run_ids"]).astype(str)
    splits = np.asarray(source_data["splits"]).astype(str)
    sequences: list[SequenceData] = []
    for sequence_index, ((begin_raw, end_raw), run_raw) in enumerate(zip(
            source_data["bounds"], source_data["seq_run"])):
        begin, end, run_index = int(begin_raw), int(end_raw), int(run_raw)
        split = str(splits[run_index])
        if split not in allowed_splits:
            continue
        frames = frames_all[begin:end]
        state = truth_body_all[begin:end].copy()
        pose = pose_all[begin:end]
        raw = raw_all[begin:end]
        raw_valid = raw_valid_all[begin:end]
        packet = packet_all[begin:end]
        # sequence_reset_index is stored once per extracted sequence (not per
        # frame); bounds already prevent crossing the associated reset epoch.
        reset = np.full(end - begin, int(reset_all[sequence_index]),
                        dtype=np.int64)
        dt = dt_all[begin:end]
        # Only current odometry, actuator feedback, and commands match queries;
        # simulator rigid-state truth remains a future-response label.
        core = np.column_stack((frames[:, :3], frames[:, 3:5], frames[:, 7:9]))
        filtered = np.column_stack((core, frames[:, 5:7]))
        responses, response_names = _future_responses(
            state, frames, pose, raw, raw_valid)
        provisional = SequenceData(
            run_id=str(run_ids[run_index]), split=split, frames=frames,
            packet=packet, reset=reset, state=state, pose=pose,
            raw_wheel=raw, raw_valid=raw_valid, core_features=core,
            filtered_features=filtered, responses=responses,
            response_names=response_names, eligible=np.empty(0, dtype=np.int64),
            region_masks={}, source_path=str(dataset_path.relative_to(ROOT)))
        finite = (np.isfinite(core).all(axis=1)
                  & np.isfinite(filtered).all(axis=1)
                  & np.isfinite(state[:, :3]).all(axis=1)
                  & np.isfinite(pose).all(axis=1))
        provisional.eligible = _valid_candidate_rows(provisional, dt)
        provisional.region_masks = region_masks(
            frames, reset, packet, finite & np.isfinite(responses[:, :3]).all(axis=1),
            dt_s=DT_S)
        sequences.append(provisional)
    return sequences


def _history_context(sequence: SequenceData, rows: np.ndarray,
                     history_steps: int, include_wheels: bool) -> np.ndarray:
    base = (sequence.filtered_features if include_wheels
            else sequence.core_features)
    rows = np.asarray(rows, dtype=np.int64)
    current = base[rows]
    if history_steps == 0:
        return current.astype(np.float32, copy=False)
    offsets = np.arange(history_steps, -1, -1, dtype=np.int64)
    samples = base[rows[:, None] - offsets[None, :]]
    mean = np.mean(samples, axis=1)
    centered_time = (offsets.mean() - offsets.astype(np.float64)) * DT_S
    denominator = float(centered_time @ centered_time)
    slope = np.einsum("t,ntd->nd", centered_time, samples) / denominator
    return np.column_stack((current, mean, slope)).astype(np.float32)


def _run_balanced_scale(train_contexts: dict[str, np.ndarray]
                        ) -> tuple[np.ndarray, np.ndarray]:
    if len(train_contexts) < 3:
        raise ValueError("WP18 normalization requires at least three train runs")
    medians, iqrs = [], []
    for values in train_contexts.values():
        medians.append(np.median(values, axis=0))
        q25, q75 = np.quantile(values, (0.25, 0.75), axis=0)
        iqrs.append(q75 - q25)
    center = np.mean(np.stack(medians), axis=0)
    scale = np.maximum(np.mean(np.stack(iqrs), axis=0), 1e-2)
    return center.astype(np.float32), scale.astype(np.float32)


def _run_balanced_target_variance(train_targets: dict[str, np.ndarray]
                                  ) -> np.ndarray:
    if not train_targets:
        raise ValueError("WP18 requires training-run response targets")
    dimensions = {values.shape[1] for values in train_targets.values()}
    if len(dimensions) != 1:
        raise ValueError("WP18 training response dimensions do not match")
    result = np.full(next(iter(dimensions)), np.nan, dtype=np.float64)
    for index in range(len(result)):
        run_variances = []
        for values in train_targets.values():
            finite = values[:, index][np.isfinite(values[:, index])]
            if len(finite):
                run_variances.append(float(np.var(finite)))
        if run_variances:
            result[index] = float(np.mean(run_variances))
    if not np.isfinite(result).all():
        raise ValueError("at least one WP18 response has no training support")
    return np.maximum(result, 1e-8)


def _bootstrap_mean(values: list[float], seed: int) -> list[float] | None:
    data = np.asarray(values, dtype=np.float64)
    data = data[np.isfinite(data)]
    if not len(data):
        return None
    rng = np.random.default_rng(seed + len(data))
    draw = rng.integers(0, len(data), size=(5000, len(data)))
    return np.quantile(data[draw].mean(axis=1), (0.025, 0.975)).tolist()


def _aggregate_query_metrics(
        query_rows_by_run: dict[str, list[tuple[int, int]]],
        contexts: dict[tuple[str, int], np.ndarray],
        responses: dict[tuple[str, int], np.ndarray],
        candidate_contexts: dict[str, np.ndarray],
        candidate_responses: dict[str, np.ndarray],
        center: np.ndarray, scale: np.ndarray,
        target_variance: np.ndarray, target_names: tuple[str, ...],
        seed: int, keep_query_residuals: bool = False
        ) -> dict[str, Any]:
    candidate_run_ids = sorted(candidate_contexts)
    trees: dict[str, cKDTree] = {}
    normalized_bank: dict[str, np.ndarray] = {}
    for run_id in candidate_run_ids:
        bank = (candidate_contexts[run_id] - center) / scale
        normalized_bank[run_id] = bank.astype(np.float32, copy=False)
        trees[run_id] = cKDTree(normalized_bank[run_id])

    per_query_run: dict[str, dict[str, dict[str, list[float]]]] = {}
    residual_for_pca: list[np.ndarray] = []
    for query_run in sorted(query_rows_by_run):
        keys = query_rows_by_run[query_run]
        if not keys:
            continue
        query_x = np.concatenate([contexts[key] for key in keys], axis=0)
        query_y = np.concatenate([responses[key] for key in keys], axis=0)
        query_norm = (query_x - center) / scale
        run_distances: list[np.ndarray] = []
        run_targets: list[np.ndarray] = []
        run_names: list[str] = []
        for candidate_run in candidate_run_ids:
            if candidate_run == query_run:
                continue
            distances, indices = trees[candidate_run].query(
                query_norm, k=min(NEIGHBOR_ROWS_PER_RUN,
                                  len(candidate_responses[candidate_run])))
            if distances.ndim == 1:
                distances = distances[:, None]
                indices = indices[:, None]
            selected_y = candidate_responses[candidate_run][indices]
            finite = np.isfinite(selected_y)
            count = finite.sum(axis=1)
            run_y = np.divide(
                np.where(finite, selected_y, 0.0).sum(axis=1), count,
                out=np.full(count.shape, np.nan, dtype=np.float64),
                where=count > 0)
            run_distances.append(np.mean(distances, axis=1))
            run_targets.append(run_y)
            run_names.append(candidate_run)
        distance_matrix = np.stack(run_distances, axis=1)
        target_cube = np.stack(run_targets, axis=1)
        order = np.argsort(distance_matrix, axis=1)
        selected_count = min(NEIGHBOR_INDEPENDENT_RUNS, len(run_names))
        selected_indices = order[:, :selected_count]
        selected_distances = np.take_along_axis(distance_matrix,
                                                selected_indices, axis=1)
        selected_targets = np.take_along_axis(
            target_cube, selected_indices[:, :, None], axis=1)
        channel_metrics = per_query_run.setdefault(query_run, {
            name: {key: [] for key in (
                "normalized_conditional_variance", "neighbor_target_spread",
                "query_neighbor_distance", "effective_independent_run_count")}
            for name in target_names})
        channel_mean_targets = np.full(
            (len(query_y), len(target_names)), np.nan, dtype=np.float64)
        for target_index in range(len(target_names)):
            target_values = selected_targets[:, :, target_index]
            finite = np.isfinite(target_values)
            neighbor_count = finite.sum(axis=1)
            valid = neighbor_count > 0
            if not np.any(valid):
                continue
            safe_values = np.where(finite, target_values, 0.0)
            mean = np.divide(safe_values.sum(axis=1), neighbor_count,
                             out=np.full(len(query_y), np.nan), where=valid)
            channel_mean_targets[:, target_index] = mean
            centered = np.where(finite, target_values - mean[:, None], 0.0)
            variance = np.divide((centered ** 2).sum(axis=1), neighbor_count,
                                 out=np.full(len(query_y), np.nan), where=valid)
            spread = np.full(len(query_y), np.nan)
            for row_index in np.flatnonzero(neighbor_count >= 2):
                row_values = target_values[row_index, finite[row_index]]
                q10, q90 = np.quantile(row_values, (0.10, 0.90))
                spread[row_index] = q90 - q10
            channel = channel_metrics[target_names[target_index]]
            channel["normalized_conditional_variance"].extend(
                (variance[valid] / target_variance[target_index]).tolist())
            channel["neighbor_target_spread"].extend(
                spread[np.isfinite(spread)].tolist())
            channel["query_neighbor_distance"].extend(
                np.mean(selected_distances, axis=1)[valid].tolist())
            channel["effective_independent_run_count"].extend(
                neighbor_count[valid].astype(float).tolist())
        if keep_query_residuals:
            valid_targets = (np.isfinite(query_y).all(axis=1)
                             & np.isfinite(channel_mean_targets).all(axis=1))
            if np.any(valid_targets):
                residual_for_pca.append(
                    (query_y[valid_targets]
                     - channel_mean_targets[valid_targets])
                    / np.sqrt(target_variance)[None, :])

    metrics: dict[str, Any] = {}
    metric_names = ("normalized_conditional_variance",
                    "neighbor_target_spread_p50",
                    "neighbor_target_spread_p90", "query_neighbor_distance",
                    "effective_independent_run_count")
    for key_index, key in enumerate(metric_names):
        per_target: dict[str, Any] = {}
        for target_index, target_name in enumerate(target_names):
            per_run = {}
            for run, channels in per_query_run.items():
                source_key = ("neighbor_target_spread" if key.startswith(
                    "neighbor_target_spread_") else key)
                observations = channels[target_name][source_key]
                if observations:
                    if key == "neighbor_target_spread_p50":
                        per_run[run] = float(np.quantile(observations, 0.50))
                    elif key == "neighbor_target_spread_p90":
                        per_run[run] = float(np.quantile(observations, 0.90))
                    else:
                        per_run[run] = float(np.mean(observations))
            values = list(per_run.values())
            per_target[target_name] = {
                "macro_run_mean": float(np.mean(values)) if values else None,
                "run_cluster_bootstrap_95pct_ci": _bootstrap_mean(
                    values, seed + key_index * 101 + target_index),
                "independent_query_run_count": len(values),
                "weak_evidence_under_three_runs": len(values) < 3,
                "per_run": per_run,
            }
        target_means = [item["macro_run_mean"] for item in per_target.values()
                        if item["macro_run_mean"] is not None]
        metrics[key] = {
            "macro_run_mean": (float(np.mean(target_means))
                               if target_means else None),
            "per_target": per_target,
        }
    return {
        "metrics": metrics,
        "per_query_run": per_query_run,
        "query_residuals_for_pca": (
            np.concatenate(residual_for_pca, axis=0)
            if residual_for_pca else np.empty((0, len(target_names)))),
    }


def _pca_summary(residuals: np.ndarray) -> dict[str, Any]:
    residuals = np.asarray(residuals, dtype=np.float64)
    if residuals.ndim != 2 or len(residuals) < 3:
        return {"status": "insufficient residual vectors",
                "sample_count": int(len(residuals))}
    residuals = residuals - np.mean(residuals, axis=0)
    singular = np.linalg.svd(residuals, full_matrices=False,
                            compute_uv=False)
    variance = singular ** 2
    ratios = variance / max(float(np.sum(variance)), 1e-12)
    cumulative = np.cumsum(ratios)
    return {
        "status": "diagnostic only; latent directions are not physical tire quantities",
        "sample_count": int(len(residuals)),
        "response_dimension": int(residuals.shape[1]),
        "explained_variance_ratio_first_10": ratios[:10].tolist(),
        "modes_for_80pct": int(np.searchsorted(cumulative, 0.80) + 1),
        "modes_for_90pct": int(np.searchsorted(cumulative, 0.90) + 1),
        "modes_for_95pct": int(np.searchsorted(cumulative, 0.95) + 1),
    }


def _analyze_setting(sequences: list[SequenceData],
                     run_splits: dict[str, str],
                     candidate_samples: dict[str, dict[int, np.ndarray]],
                     query_samples_by_region: dict[str, dict[str, list[tuple[int, int]]]],
                     history_name: str, history_steps: int,
                     feature_variant: str, setting_index: int,
                     keep_pca: bool = False) -> dict[str, Any]:
    sequence_by_run: dict[str, list[tuple[int, SequenceData]]] = {}
    for index, sequence in enumerate(sequences):
        sequence_by_run.setdefault(sequence.run_id, []).append((index, sequence))
    include_wheels = feature_variant.endswith("filtered_wheel")
    context_by_key: dict[tuple[str, int], np.ndarray] = {}
    response_by_key: dict[tuple[str, int], np.ndarray] = {}
    candidate_contexts: dict[str, np.ndarray] = {}
    candidate_responses: dict[str, np.ndarray] = {}
    train_contexts: dict[str, np.ndarray] = {}
    train_targets: dict[str, np.ndarray] = {}

    for run_id, sequence_rows in sequence_by_run.items():
        all_rows = sorted({int(row) for seq_index, _ in sequence_rows
                           for row in candidate_samples.get(run_id, {}).get(
                               seq_index, np.empty(0, dtype=np.int64))})
        if not all_rows:
            continue
        by_sequence_context, by_sequence_response = [], []
        for seq_index, sequence in sequence_rows:
            rows = candidate_samples.get(run_id, {}).get(
                seq_index, np.empty(0, dtype=np.int64))
            if not len(rows):
                continue
            context = _history_context(sequence, rows, history_steps,
                                       include_wheels)
            target = sequence.responses[rows]
            context_by_key[(run_id, seq_index)] = context
            response_by_key[(run_id, seq_index)] = target
            by_sequence_context.append(context)
            by_sequence_response.append(target)
        if by_sequence_context:
            candidate_contexts[run_id] = np.concatenate(by_sequence_context)
            candidate_responses[run_id] = np.concatenate(by_sequence_response)
            if run_splits[run_id] == "train":
                train_contexts[run_id] = candidate_contexts[run_id]
                train_targets[run_id] = candidate_responses[run_id]

    center, scale = _run_balanced_scale(train_contexts)
    target_variance = _run_balanced_target_variance(train_targets)
    response_names = sequences[0].response_names
    # Keep the output list focused and comparable, while preserving every
    # requested future response channel from the detailed arrays.
    primary_response_indices = list(range(len(response_names)))
    candidate_responses = {
        run: values[:, primary_response_indices]
        for run, values in candidate_responses.items()}
    target_variance = target_variance[primary_response_indices]
    target_names = tuple(response_names[index]
                         for index in primary_response_indices)

    metrics_by_region = {}
    pca_summary = None
    region_key_list = sorted(query_samples_by_region)
    for region in region_key_list:
        rows_by_run = query_samples_by_region[region]
        query_contexts: dict[tuple[str, int], np.ndarray] = {}
        query_responses: dict[tuple[str, int], np.ndarray] = {}
        rows_by_query_run: dict[str, list[tuple[int, int]]] = {}
        for run_id, items in rows_by_run.items():
            for sequence_index, rows in items.items():
                if not len(rows):
                    continue
                sequence = sequences[sequence_index]
                key = (run_id, sequence_index)
                query_contexts[key] = _history_context(
                    sequence, rows, history_steps, include_wheels)
                query_responses[key] = sequence.responses[rows]
                rows_by_query_run.setdefault(run_id, []).append(key)
        result = _aggregate_query_metrics(
            rows_by_query_run, query_contexts, query_responses,
            candidate_contexts, candidate_responses, center, scale,
            target_variance, target_names,
            seed=20261003 + setting_index * 1009 + len(region),
            keep_query_residuals=(keep_pca and region == "all_valid"))
        metrics_by_region[region] = result["metrics"]
        if keep_pca and region == "all_valid":
            pca_summary = _pca_summary(result["query_residuals_for_pca"])

    return {
        "history_length": history_name,
        "history_steps": history_steps,
        "feature_variant": feature_variant,
        "matching_feature_dimensions": int(len(center)),
        "independent_candidate_run_count": int(len(candidate_contexts)),
        "candidate_rows_by_run": {
            run: int(len(values)) for run, values in candidate_contexts.items()},
        "run_balanced_feature_scale": {
            "center": center.tolist(), "iqr_scale": scale.tolist()},
        "global_training_response_variance": {
            name: float(value) for name, value in zip(target_names, target_variance)},
        "regions": metrics_by_region,
        "residual_response_pca": pca_summary,
        "response_names": list(target_names),
    }


def _select_samples(sequences: list[SequenceData], rng: np.random.Generator,
                    max_query_rows: int, candidate_cap: int,
                    candidate_extra_cap: int = CANDIDATE_EXTRA_PER_RUN_REGION
                    ) -> tuple[dict[str, dict[int, np.ndarray]],
                               dict[str, dict[str, dict[int, np.ndarray]]],
                               dict[str, str]]:
    # Handoff-required named regimes, plus the common Section 9 bins.
    required_regions = (
        "all_valid", "high_speed_near_straight",
        "high_speed_moderate_steering", "7_to_9mps_high_steering",
        "low_speed_high_steering", "negative_command_braking",
        "braking_release", "throttle_pickup", "steering_turn_in",
        "steering_unwind", "simultaneous_steering_throttle_transition",
        "large_wheel_body_mismatch", "ordinary_practice_track_region",
        *(f"S{i}" for i in range(6)), *(f"D{i}" for i in range(5)),
        *(f"R{i}" for i in range(4)), *(f"T{i}" for i in range(4)),
        *(f"M{i}" for i in range(7)),
    )
    run_splits: dict[str, str] = {}
    candidate_base: dict[str, list[tuple[int, np.ndarray]]] = {}
    candidate_regions: dict[str, dict[str, list[tuple[int, np.ndarray]]]] = {}
    query_regions: dict[str, dict[str, list[tuple[int, np.ndarray]]]] = {}
    for sequence_index, sequence in enumerate(sequences):
        previous = run_splits.setdefault(sequence.run_id, sequence.split)
        if previous != sequence.split:
            raise ValueError(f"run {sequence.run_id} appears in multiple splits")
        eligible = sequence.eligible
        if sequence.split in CANDIDATE_SPLITS:
            candidate_base.setdefault(sequence.run_id, []).append(
                (sequence_index, eligible))
            for region in required_regions:
                if region == "all_valid":
                    continue
                rows = eligible[sequence.region_masks[region][eligible]]
                if len(rows):
                    candidate_regions.setdefault(sequence.run_id, {}).setdefault(
                        region, []).append((sequence_index, rows))
        if sequence.split in QUERY_SPLITS:
            for region in required_regions:
                region_mask = (np.ones(len(sequence.frames), dtype=bool)
                               if region == "all_valid"
                               else sequence.region_masks[region])
                rows = eligible[region_mask[eligible]]
                if len(rows):
                    query_regions.setdefault(sequence.run_id, {}).setdefault(
                        region, []).append((sequence_index, rows))

    def encode(parts: list[tuple[int, np.ndarray]]) -> np.ndarray:
        pieces = [((np.int64(sequence_index) << np.int64(32))
                   | np.asarray(rows, dtype=np.int64))
                  for sequence_index, rows in parts if len(rows)]
        return np.concatenate(pieces) if pieces else np.empty(0, dtype=np.int64)

    def decode(keys: np.ndarray) -> dict[int, np.ndarray]:
        grouped: dict[int, list[np.ndarray]] = {}
        for sequence_index in np.unique(keys >> np.int64(32)):
            local = (keys >> np.int64(32)) == sequence_index
            grouped[int(sequence_index)] = [
                np.asarray(keys[local] & np.int64(0xFFFFFFFF), dtype=np.int64)]
        return {index: np.sort(np.concatenate(rows))
                for index, rows in grouped.items()}

    candidate_by_run: dict[str, dict[int, np.ndarray]] = {}
    for run_id, parts in candidate_base.items():
        base_pool = encode(parts)
        if len(base_pool) > candidate_cap:
            selected = rng.choice(base_pool, size=candidate_cap, replace=False)
        else:
            selected = base_pool.copy()
        selected = np.unique(selected)
        for region in required_regions:
            if region == "all_valid":
                continue
            region_pool = encode(candidate_regions.get(run_id, {}).get(region, []))
            if not len(region_pool):
                continue
            available = region_pool[~np.isin(region_pool, selected,
                                             assume_unique=False)]
            if len(available) > candidate_extra_cap:
                extra = rng.choice(available, size=candidate_extra_cap,
                                   replace=False)
            else:
                extra = available
            selected = np.unique(np.concatenate((selected, extra)))
        if len(selected):
            candidate_by_run[run_id] = decode(selected)

    query_by_region: dict[str, dict[str, dict[int, np.ndarray]]] = {
        region: {} for region in required_regions}
    for run_id, regions in query_regions.items():
        for region, parts in regions.items():
            keys = encode(parts)
            if len(keys) > max_query_rows:
                keys = rng.choice(keys, size=max_query_rows, replace=False)
            if len(keys):
                query_by_region[region][run_id] = decode(keys)
    # Ensure at least five distinct candidate runs remain after excluding a query.
    for run_id in query_by_region["all_valid"]:
        available = [candidate for candidate in candidate_by_run
                     if candidate != run_id]
        if len(available) < 5:
            raise ValueError(f"only {len(available)} independent neighbors for {run_id}")
    return candidate_by_run, query_by_region, run_splits


def _history_length_decision(settings: list[dict[str, Any]],
                             response_names: list[str]) -> dict[str, Any]:
    body_indices = [index for index, name in enumerate(response_names)
        if name.startswith(("effective_ax", "effective_ay", "yaw_accel",
                            "delta_u", "delta_v", "delta_yaw_rate",
                            "heading_change"))]
    names_by_history = {setting["history_length"]: setting
                        for setting in settings
                        if setting["feature_variant"] == "body_actuator"}
    lengths = list(HISTORY_LENGTHS)
    per_channel: dict[str, dict[str, float | None]] = {}
    complete_names: list[str] = []
    for index in body_indices:
        name = response_names[index]
        values: dict[str, float | None] = {}
        for length in lengths:
            target = (names_by_history[length]["regions"]["all_valid"]
                      ["normalized_conditional_variance"]["per_target"].get(name))
            values[length] = (target["macro_run_mean"] if target else None)
        per_channel[name] = values
        if all(value is not None and np.isfinite(value) for value in values.values()):
            complete_names.append(name)

    baseline_key, endpoint_key = lengths[0], lengths[-1]
    total_reduction: dict[str, float | None] = {}
    for name in complete_names:
        base = float(per_channel[name][baseline_key])
        endpoint = float(per_channel[name][endpoint_key])
        total_reduction[name] = base - endpoint

    reduction_fraction_by_history = {}
    for length in lengths:
        fractions = {}
        for name, reduction in total_reduction.items():
            fractions[name] = (
                (float(per_channel[name][baseline_key])
                 - float(per_channel[name][length])) / reduction
                if reduction > 1e-12 else None)
        reduction_fraction_by_history[length] = fractions

    evaluated: list[dict[str, Any]] = []
    selected: str | None = None
    for index, length in enumerate(lengths[:-1]):
        next_length = lengths[index + 1]
        achieved = []
        for name, reduction in total_reduction.items():
            current = float(per_channel[name][length])
            if reduction > 1e-12:
                achieved.append((float(per_channel[name][baseline_key]) - current)
                                / reduction >= 0.90)
            else:
                achieved.append(False)
        current_values = [float(per_channel[name][length])
                          for name in complete_names]
        next_values = [float(per_channel[name][next_length])
                       for name in complete_names]
        current_dispersion = (float(np.mean(current_values))
                              if current_values else None)
        next_dispersion = (float(np.mean(next_values))
                           if next_values else None)
        relative_next_improvement = (
            (current_dispersion - next_dispersion) / current_dispersion
            if current_dispersion is not None and current_dispersion > 0.0
            and next_dispersion is not None else None)
        majority = bool(achieved and sum(achieved) > len(achieved) / 2)
        plateau = (relative_next_improvement is not None
                   and relative_next_improvement < 0.05)
        evaluated.append({
            "history_length": length,
            "next_history_length": next_length,
            "channels_with_complete_run_estimates": len(complete_names),
            "channels_achieving_at_least_90pct_of_0_to_4s_reduction":
                int(sum(achieved)),
            "majority_channel_threshold_met": majority,
            "aggregate_normalized_dispersion": current_dispersion,
            "next_aggregate_normalized_dispersion": next_dispersion,
            "relative_improvement_from_next_history": relative_next_improvement,
            "less_than_5pct_next_improvement": plateau,
        })
        if selected is None and majority and plateau:
            selected = length

    unsolved = [name for name, reduction in total_reduction.items()
                if reduction <= 1e-12]
    return {
        "status": "history length selected by handoff rule" if selected else
                  "no supported history plateau; retain no fixed-history claim",
        "chosen_history_length": selected,
        "primary_body_response_channels": [response_names[i]
                                            for i in body_indices],
        "complete_primary_channels": complete_names,
        "channels_without_positive_0_to_4s_variance_reduction": unsolved,
        "per_channel_normalized_conditional_variance": per_channel,
        "per_channel_fraction_of_0_to_4s_reduction_by_history":
            reduction_fraction_by_history,
        "candidate_history_evaluations": evaluated,
        "rule": (
            "shortest tested history for which a strict majority of primary "
            "body response channels achieve >=90% of their 0-to-4s variance "
            "reduction and the next longer history improves mean normalized "
            "conditional dispersion by <5%"),
        "independence_note": (
            "run-level bootstrap intervals and per-channel independent query "
            "run counts are retained in each setting; history selection is "
            "descriptive, not a population guarantee"),
    }


def analyze(dynamic_path: Path, practice_path: Path,
            dynamic_fixed_path: Path, practice_fixed_path: Path,
            wp17_path: Path, output: Path,
            max_query_rows: int = QUERY_CAP_PER_RUN_REGION,
            candidate_cap: int = CANDIDATE_CAP_PER_RUN,
            ) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite WP18 report: {output}")
    for path in (dynamic_path, practice_path, dynamic_fixed_path,
                 practice_fixed_path, wp17_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    wp17 = json.loads(wp17_path.read_text(encoding="utf-8"))
    if (wp17.get("checkpoint_unchanged") is not True
            or wp17.get("training_performed") is not False
            or wp17.get("wp17_decision", {}).get("rear_processed_wheel_rate")
                != "B: measurement/output only"):
        raise ValueError("WP18 requires the completed WP17 measurement/output decision")
    if max_query_rows <= 0 or candidate_cap < 100:
        raise ValueError("query/candidate sample caps must be positive and useful")

    dynamic = _load_dataset(dynamic_path)
    practice = _load_dataset(practice_path)
    dynamic_fixed = _load_dataset(dynamic_fixed_path)
    practice_fixed = _load_dataset(practice_fixed_path)
    _assert_fixed_view_rows(dynamic, dynamic_fixed, "open-plane")
    _assert_fixed_view_rows(practice, practice_fixed, "practice")
    if not np.allclose(dynamic["dt_s"], DT_S, rtol=0.0, atol=1e-7):
        raise ValueError("WP18 requires exact fixed 25 ms base samples")

    sequences = _sequence_views(dynamic, dynamic_fixed, dynamic_path,
        {"train", "validation"})
    practice_sequences = _sequence_views(practice, practice_fixed,
        practice_path, {"unseen_practice"})
    sequences.extend(practice_sequences)
    rng = np.random.default_rng(20261018)
    candidate_samples, query_samples, run_splits = _select_samples(
        sequences, rng, max_query_rows, candidate_cap)
    all_query_runs = sorted(set().union(*(
        set(query_samples[name]) for name in query_samples)))
    if not all_query_runs:
        raise ValueError("WP18 has no validation or unseen-practice queries")
    if any(run_splits[run] not in QUERY_SPLITS for run in all_query_runs):
        raise ValueError("WP18 query selection leaked non-validation runs")
    if any(run_splits[run] not in CANDIDATE_SPLITS for run in candidate_samples):
        raise ValueError("WP18 candidate bank contains test/final-test run")
    candidate_counts = {
        run: int(sum(len(rows) for rows in seqs.values()))
        for run, seqs in candidate_samples.items()}
    query_counts_by_region = {
        region: {run: int(sum(len(rows) for rows in seqs.values()))
                 for run, seqs in runs.items()}
        for region, runs in query_samples.items()}
    settings = []
    setting_index = 0
    for feature_variant in FEATURE_VARIANTS:
        for history_name, history_steps in HISTORY_LENGTHS.items():
            settings.append(_analyze_setting(
                sequences, run_splits, candidate_samples, query_samples,
                history_name, history_steps, feature_variant, setting_index,
                keep_pca=(feature_variant == "body_actuator"
                          and history_name == "0ms")))
            setting_index += 1

    def setting_key(setting: dict[str, Any]) -> tuple[str, str]:
        return setting["feature_variant"], setting["history_length"]
    setting_map = {setting_key(setting): setting for setting in settings}
    primary_variant = "body_actuator"
    overall = {}
    for history_name in HISTORY_LENGTHS:
        setting = setting_map[(primary_variant, history_name)]
        overall[history_name] = setting["regions"]
    response_names = settings[0]["response_names"]
    history_selection_evidence = _history_length_decision(settings,
                                                            response_names)
    selected_setting = (setting_map[("body_actuator",
                                    history_selection_evidence["chosen_history_length"])]
                       if history_selection_evidence["chosen_history_length"]
                       else None)
    unresolved_regions = {}
    for region, region_metrics in overall.get("0ms", {}).items():
        target_metrics = region_metrics.get("normalized_conditional_variance", {}) \
            .get("per_target", {})
        weak = {name: metric.get("independent_query_run_count", 0)
                for name, metric in target_metrics.items()
                if metric.get("independent_query_run_count", 0) < 3}
        if weak:
            unresolved_regions[region] = weak
    all_valid_metrics = overall.get("0ms", {}).get("all_valid", {})
    all_valid_variance = all_valid_metrics.get(
        "normalized_conditional_variance", {}).get("per_target", {})
    unresolved_dispersion = {
        name: metric.get("macro_run_mean")
        for name, metric in all_valid_variance.items()
        if name.startswith(("effective_ax", "effective_ay", "yaw_accel",
                            "delta_u", "delta_v", "delta_yaw_rate",
                            "heading_change"))}
    pca = (setting_map[("body_actuator", "0ms")]
           .get("residual_response_pca"))
    modes90 = pca.get("modes_for_90pct") if pca else None
    latent_dimension = (8 if modes90 is not None and modes90 <= 8
                        else (16 if modes90 is not None else None))
    gate_decision = {
        "chosen_history_length": history_selection_evidence["chosen_history_length"],
        "heldout_normalized_conditional_dispersion_proxy_all_valid":
            unresolved_dispersion,
        "recommended_initial_latent_dimension": latent_dimension,
        "latent_dimension_basis": (
            "8 when <=8 PCA residual modes explain 90%; otherwise 16; "
            "diagnostic only, not physical state identification"),
        "regimes_with_unresolved_or_weak_evidence": unresolved_regions,
        "history_selection_status": history_selection_evidence["status"],
        "plant_training_authorized": False,
    }

    report = {
        "schema_version": 1,
        "purpose": "WP18 run-balanced future-response observability and causal-history sufficiency analysis",
        "generated_date": "2026-10-03",
        "wp17_decision": wp17["wp17_decision"],
        "training_performed": False,
        "query_policy": "whole held-out validation and unseen-practice runs only; maximum-length history and all future response horizons are complete within one contiguous 25 ms/reset sequence",
        "candidate_policy": "train/validation/unseen-practice runs only; candidate rows are run-balanced; query run excluded from its neighbor bank; test/final-test excluded",
        "matching_policy": {
            "current_body_observation_source": (
                "frames[:,0:3], the recorded bridge-odometry rear-axle u/v/r; "
                "simulator rigid-state truth is never in the matching vector"),
            "base_current_features": ["u_com", "v_com", "yaw_rate",
                "steering_feedback", "throttle_feedback", "steering_command",
                "throttle_command"],
            "wheel_feature_variants": list(FEATURE_VARIANTS),
            "history_summary": "for each requested causal trailing interval, append run-endpoint current-channel mean and least-squares time slope; the 0 ms baseline uses current values only",
            "history_lengths": HISTORY_LENGTHS,
            "future_values_used_in_matching": False,
            "future_response_body_frame": (
                "simulator COM body u/v/r and COM effective acceleration labels; "
                "matching state remains measured rear-axle odometry"),
            "query_cap_per_independent_run_and_region": max_query_rows,
            "candidate_base_cap_per_independent_run": candidate_cap,
            "candidate_extra_cap_per_independent_run_and_region":
                CANDIDATE_EXTRA_PER_RUN_REGION,
            "candidate_all_valid_base_rows_are_not_double_counted": True,
            "rows_per_candidate_run_per_query": NEIGHBOR_ROWS_PER_RUN,
            "nearest_independent_candidate_runs": NEIGHBOR_INDEPENDENT_RUNS,
            "feature_normalization": "run-balanced median/IQR across train runs only",
            "candidate_response_weighting": "average nearest rows within each candidate run, then weight nearest independent runs equally",
        },
        "data_provenance": {
            "dynamic_dataset": str(dynamic_path.relative_to(ROOT)),
            "dynamic_dataset_sha256": _sha256(dynamic_path),
            "dynamic_fixed_sidecar": str(dynamic_fixed_path.relative_to(ROOT)),
            "dynamic_fixed_sidecar_sha256": _sha256(dynamic_fixed_path),
            "practice_dataset": str(practice_path.relative_to(ROOT)),
            "practice_dataset_sha256": _sha256(practice_path),
            "practice_fixed_sidecar": str(practice_fixed_path.relative_to(ROOT)),
            "practice_fixed_sidecar_sha256": _sha256(practice_fixed_path),
            "wp17_report": str(wp17_path.relative_to(ROOT)),
            "wp17_report_sha256": _sha256(wp17_path),
            "source_fixed_view_rows_exactly_match": True,
        },
        "independent_run_roles": {
            "query_runs": {run: run_splits[run] for run in all_query_runs},
            "candidate_runs": {run: run_splits[run] for run in sorted(candidate_samples)},
            "candidate_row_count_by_run": candidate_counts,
            "query_row_count_by_region_and_run": query_counts_by_region,
        },
        "response_names": list(response_names),
        "future_truth_is_label_only": True,
        "regions_with_fewer_than_three_query_runs_are_weak_evidence": True,
        "history_selection": history_selection_evidence,
        "wp18_gate_decision": gate_decision,
        "settings": settings,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dynamic-dataset", type=Path, default=DEFAULT_DYNAMIC)
    parser.add_argument("--practice-dataset", type=Path, default=DEFAULT_PRACTICE)
    parser.add_argument("--dynamic-fixed", type=Path, default=DEFAULT_DYNAMIC_FIXED)
    parser.add_argument("--practice-fixed", type=Path, default=DEFAULT_PRACTICE_FIXED)
    parser.add_argument("--wp17-report", type=Path, default=DEFAULT_WP17)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-query-rows", type=int,
                        default=QUERY_CAP_PER_RUN_REGION)
    parser.add_argument("--candidate-cap", type=int,
                        default=CANDIDATE_CAP_PER_RUN)
    args = parser.parse_args()
    report = analyze(args.dynamic_dataset.resolve(),
        args.practice_dataset.resolve(), args.dynamic_fixed.resolve(),
        args.practice_fixed.resolve(), args.wp17_report.resolve(),
        args.output.resolve(), args.max_query_rows, args.candidate_cap)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "query_runs": len(report["independent_run_roles"]["query_runs"]),
        "candidate_runs": len(report["independent_run_roles"]["candidate_runs"]),
        "history_selection": report["history_selection"],
        "settings": len(report["settings"]),
        "regions": len(report["settings"][0]["regions"]),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
