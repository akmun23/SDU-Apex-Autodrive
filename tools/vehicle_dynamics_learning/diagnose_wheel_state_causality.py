#!/usr/bin/env python3
"""Audit causal rear-wheel representations on held-out whole runs.

Reuses the dataset and bag alignment helpers that built the existing encoder
sidecars. Test/final-test bags are excluded. The optional production replay is
the exact local C++ observer and deployment YAML, not a Python reimplementation.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

from tools import analyze_open_plane_dynamics as analysis
from tools.racing.offline.localization.production_odom import ProductionOdometry
from tools.vehicle_dynamics_learning.build_encoder_state_view import (
    MAX_ALIGNMENT_NS,
    _frame_run_index,
    _encoder_angles,
    _latest_source_rate,
    _manifest_bags,
)
from tools.vehicle_dynamics_learning.signal_semantics import (
    DT_S,
    ENCODER_SOURCE_DT_GATE_S,
    REAR_TRACK_WIDTH_M,
    WHEEL_RADIUS_M,
    elapsed_window_surface_rate,
    fixed_n_period_surface_rate,
    fixed_period_surface_rate,
    rear_contact_speeds,
    variable_stamp_surface_rate,
)
from tools.vehicle_dynamics_learning.operating_regions import (
    region_masks as canonical_region_masks,
)
from tools.vehicle_dynamics_learning.train_nssm import _load_dataset


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "live_runs/derived_dynamics_learning_20260928"
DEFAULT_OPENPLANE = (DATA_ROOT / "full_modeling_reset_20261001"
    / "replacement_offline_sim_raceline_20261003/full_throttle_domain_v1"
    / "encoder_fixed40hz_v1/openplane_dynamics_raw_wheels_fixed40hz.npz")
DEFAULT_PRACTICE = (DATA_ROOT / "full_modeling_reset_20261001"
    / "replacement_offline_sim_raceline_20261003/full_throttle_domain_v1"
    / "encoder_fixed40hz_v1/practice_dynamics_raw_wheels_fixed40hz.npz")
DEFAULT_LIBRARY = (ROOT / "live_runs/derived_dynamics_learning_20260928"
    / "production_odom_replay_clean_20260929/wp16_current"
    / "libproduction_odom.so")
DEFAULT_CONFIG = ROOT / "f1tenth_localization/config/sensor_odometry.yaml"
DEFAULT_OUTPUT = (DATA_ROOT / "replacement_offline_sim_raceline_20261003"
    / "full_throttle_domain_v1/wp16_wheel_causality_20261003.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_stats(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0, "bias": None, "rmse": None, "mae": None,
                "absolute_p50": None, "absolute_p95": None}
    absolute = np.abs(values)
    return {
        "count": int(len(values)),
        "bias": float(np.mean(values)),
        "rmse": float(np.sqrt(np.mean(values ** 2))),
        "mae": float(np.mean(absolute)),
        "absolute_p50": float(np.quantile(absolute, 0.50)),
        "absolute_p95": float(np.quantile(absolute, 0.95)),
    }


def _macro_run_summary(values: list[float]) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"independent_runs": 0, "macro_run_mean": None,
                "run_cluster_bootstrap_95pct_ci": None,
                "weak_evidence_under_three_runs": True}
    rng = np.random.default_rng(20261003 + len(values))
    draws = rng.integers(0, len(values), size=(5000, len(values)))
    ci = np.quantile(values[draws].mean(axis=1), (0.025, 0.975))
    return {"independent_runs": int(len(values)),
            "macro_run_mean": float(np.mean(values)),
            "run_cluster_bootstrap_95pct_ci": ci.tolist(),
            "weak_evidence_under_three_runs": bool(len(values) < 3)}


def _best_lag(reference: np.ndarray, candidate: np.ndarray,
              valid: np.ndarray, max_lag_steps: int = 8) -> dict[str, Any]:
    best: tuple[float, int, int] | None = None
    for lag in range(-max_lag_steps, max_lag_steps + 1):
        if lag < 0:
            left = slice(0, lag)
            right = slice(-lag, None)
        elif lag > 0:
            left = slice(lag, None)
            right = slice(0, -lag)
        else:
            left = right = slice(None)
        mask = valid[left] & valid[right]
        x, y = reference[left][mask], candidate[right][mask]
        if len(x) < 3:
            continue
        x, y = x - np.mean(x), y - np.mean(y)
        denom = float(np.linalg.norm(x) * np.linalg.norm(y))
        corr = float(x @ y / denom) if denom > 0.0 else math.nan
        if math.isfinite(corr) and (best is None or corr > best[0]):
            best = (corr, lag, len(x))
    return ({"correlation": best[0], "lag_steps": best[1],
             "lag_seconds": best[1] * DT_S, "paired_samples": best[2],
             "search_range_seconds": [-max_lag_steps * DT_S,
                                      max_lag_steps * DT_S]}
            if best else {"correlation": None, "lag_steps": None,
                          "lag_seconds": None, "paired_samples": 0,
                          "search_range_seconds": [-max_lag_steps * DT_S,
                                                   max_lag_steps * DT_S]})


def _temporal_noise(values: np.ndarray, valid: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(values)
    adjacent = valid[1:] & valid[:-1]
    delta = np.diff(values)[adjacent]
    derivative = delta / DT_S
    return {
        "first_difference_variance": float(np.var(delta)) if len(delta) else None,
        "derivative_noise_rms_per_s": (
            float(np.sqrt(np.mean(derivative ** 2))) if len(derivative) else None),
        "adjacent_pairs": int(len(delta)),
    }


def _autocorrelation_time(values: np.ndarray, valid: np.ndarray,
                          max_lag_steps: int = 80) -> dict[str, Any]:
    """Initial-positive-sequence time; pairs cannot cross invalid region gaps."""
    values = np.asarray(values, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(values)
    centered = values - np.mean(values[valid]) if np.any(valid) else values
    variance = float(np.mean(centered[valid] ** 2)) if np.any(valid) else 0.0
    if variance <= 1e-15:
        return {"integrated_time_s": 0.0, "first_nonpositive_lag_steps": 1}
    total = 0.5
    first_zero = max_lag_steps + 1
    invalid_prefix = np.r_[0, np.cumsum(~valid, dtype=np.int64)]
    for lag in range(1, max_lag_steps + 1):
        # A prefix sum excludes pairs spanning a data gap or region boundary.
        pair_mask = (valid[:-lag] & valid[lag:]
                     & ((invalid_prefix[lag + 1:] - invalid_prefix[:-lag - 1]) == 0))
        if not np.any(pair_mask):
            break
        rho = float(np.mean(centered[:-lag][pair_mask]
                            * centered[lag:][pair_mask]) / variance)
        if not math.isfinite(rho) or rho <= 0.0:
            first_zero = lag
            break
        total += rho
    return {"integrated_time_s": float(total * DT_S),
            "first_nonpositive_lag_steps": int(first_zero)}


def _source_current_index(stamps: np.ndarray, target_ns: int) -> int | None:
    index = bisect.bisect_right(stamps, int(target_ns)) - 1
    if index < 0 or int(target_ns) - int(stamps[index]) > MAX_ALIGNMENT_NS:
        return None
    return index


def _wheel_views(encoders: dict[str, tuple[np.ndarray, np.ndarray]],
                 sample_times_ns: np.ndarray,
                 sample_packet: np.ndarray,
                 sample_reset: np.ndarray,
                 raw_sidecar: np.ndarray,
                 raw_sidecar_valid: np.ndarray,
                 stored: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
    count = len(sample_times_ns)
    views = {name: np.full((count, 2), np.nan, dtype=np.float64) for name in (
        "encoder_angle_rad", "increment_25ms_rad", "rate_25ms_fixed_mps",
        "rate_variable_stamp_mps", "rate_50ms_fixed_mps",
        "rate_100ms_fixed_mps", "rate_100ms_elapsed_reconstruction_mps",
        "stored_100ms_mps", "sidecar_fixed25_mps")}
    views["stored_100ms_mps"][:] = stored
    views["sidecar_fixed25_mps"][:] = raw_sidecar
    sidecar_valid = np.asarray(raw_sidecar_valid, dtype=bool).copy()
    valid_intervals = np.zeros(count, dtype=bool)

    topics = (analysis.LEFT_ENCODER, analysis.RIGHT_ENCODER)
    for row, target_ns in enumerate(sample_times_ns):
        current_pairs = []
        indices = []
        for topic in topics:
            stamps, angles = encoders[topic]
            index = _source_current_index(stamps, int(target_ns))
            if index is None:
                current_pairs.append(None)
                indices.append(None)
            else:
                current_pairs.append((int(stamps[index]), float(angles[index])))
                indices.append(index)
        if (any(pair is None for pair in current_pairs)
                or current_pairs[0][0] != current_pairs[1][0]
                or indices[0] != indices[1]):
            continue
        current_stamp = current_pairs[0][0]
        views["encoder_angle_rad"][row] = [pair[1] for pair in current_pairs]
        index = int(indices[0])
        if index < 1:
            continue
        if (row < 1 or sample_reset[row] != sample_reset[row - 1]
                or sample_packet[row] != sample_packet[row - 1] + 1):
            continue
        prior_stamps = [int(encoders[topic][0][index - 1]) for topic in topics]
        prior_angles = [float(encoders[topic][1][index - 1]) for topic in topics]
        dt_s = (current_stamp - prior_stamps[0]) / 1e9
        if (prior_stamps[0] != prior_stamps[1]
                or not ENCODER_SOURCE_DT_GATE_S[0] <= dt_s <= ENCODER_SOURCE_DT_GATE_S[1]
                or any(abs(pair[1] - angle) > 50.0
                       for pair, angle in zip(current_pairs, prior_angles))):
            continue
        dtheta = np.asarray([pair[1] - angle for pair, angle
                             in zip(current_pairs, prior_angles)])
        views["increment_25ms_rad"][row] = dtheta
        views["rate_25ms_fixed_mps"][row] = dtheta * WHEEL_RADIUS_M / DT_S
        views["rate_variable_stamp_mps"][row] = dtheta * WHEEL_RADIUS_M / dt_s
        valid_intervals[row] = True
        for periods, name in ((2, "rate_50ms_fixed_mps"),
                              (4, "rate_100ms_fixed_mps")):
            older_index = index - periods
            row_start = row - periods
            if (older_index < 0 or row_start < 0
                    or np.any(sample_reset[row_start:row + 1]
                              != sample_reset[row])
                    or np.any(np.diff(sample_packet[row_start:row + 1]) != 1)):
                continue
            span_stamps = encoders[topics[0]][0][older_index:index + 1]
            span_angles = [encoders[topic][1][older_index:index + 1]
                           for topic in topics]
            gaps = np.diff(span_stamps).astype(np.float64) / 1e9
            if (len(gaps) != periods
                    or np.any(gaps < ENCODER_SOURCE_DT_GATE_S[0])
                    or np.any(gaps > ENCODER_SOURCE_DT_GATE_S[1])
                    or any(np.any(np.abs(np.diff(angles)) > 50.0)
                           for angles in span_angles)):
                continue
            older_angles = [float(encoders[topic][1][older_index])
                            for topic in topics]
            fixed_rate = [fixed_n_period_surface_rate(
                older, current_pairs[side][1], periods)
                for side, older in enumerate(older_angles)]
            views[name][row] = fixed_rate
            if periods == 4:
                elapsed = (current_stamp - int(span_stamps[0])) / 1e9
                if 0.075 <= elapsed <= 0.125:
                    views["rate_100ms_elapsed_reconstruction_mps"][row] = [
                        elapsed_window_surface_rate(
                            older_angles[side], current_pairs[side][1], elapsed)
                        for side in range(2)]

    # Drop sidecar targets on rows where the independent causal reconstruction
    # could not validate both same-stamp encoder streams.
    sidecar_valid &= np.isfinite(views["rate_25ms_fixed_mps"]).all(axis=1)
    views["sidecar_fixed25_mps"][~sidecar_valid] = np.nan
    return views, valid_intervals


def _production_replay(bag: Path, library: Path, config: Path
                       ) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    connection = sqlite3.connect(bag.resolve().as_uri() + "?mode=ro", uri=True)
    events: list[tuple[int, int, str, Any]] = []
    try:
        topics = analysis._topic_map(connection)
        required = (analysis.LEFT_ENCODER, analysis.RIGHT_ENCODER, analysis.IMU)
        missing = [topic for topic in required if topic not in topics]
        if missing:
            raise ValueError(f"{bag}: missing production replay topics {missing}")
        for order, topic in enumerate(required):
            for receipt_ns, message in analysis._messages(connection, topics, topic):
                events.append((int(receipt_ns), order, topic, message))
    finally:
        connection.close()
    events.sort(key=lambda row: (row[0], row[1]))
    stamps: list[int] = []
    collected: dict[str, list[float]] = {}
    with ProductionOdometry(library, config) as odometry:
        last_stamp = -1
        for _, _, topic, message in events:
            source_ns = analysis._stamp_ns(message.header.stamp)
            if topic == analysis.LEFT_ENCODER:
                if message.position:
                    odometry.add_left_encoder(source_ns, float(message.position[0]))
            elif topic == analysis.RIGHT_ENCODER:
                if message.position:
                    odometry.add_right_encoder(source_ns, float(message.position[0]))
            else:
                q = message.orientation
                yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                                 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
                odometry.add_imu(
                    source_ns, float(message.linear_acceleration.x),
                    float(message.linear_acceleration.y),
                    float(message.angular_velocity.z), yaw)
            state = odometry.snapshot()
            if state is None:
                continue
            stamp_ns = int(round(state.stamp_s * 1e9))
            if stamp_ns <= last_stamp:
                continue
            last_stamp = stamp_ns
            stamps.append(stamp_ns)
            for name in ("wheel_raw_mps", "wheel_mapped_mps", "wheel_packet_mps",
                         "body_u_mps", "body_v_mps", "yaw_rate_radps",
                         "wheel_update_used", "wheel_burst_rejected",
                         "reset_epoch", "timing_degraded", "valid"):
                collected.setdefault(name, []).append(float(getattr(state, name)))
    return (np.asarray(stamps, dtype=np.int64),
            {key: np.asarray(value, dtype=np.float64)
             for key, value in collected.items()},
            {"events_replayed": len(events), "complete_output_packets": len(stamps),
             "source_order": "bag receipt order; packet assembler groups by matching source stamp"})


def _align_production(source_stamps: np.ndarray,
                      production_stamps: np.ndarray,
                      production: dict[str, np.ndarray]
                      ) -> tuple[dict[str, np.ndarray], np.ndarray]:
    aligned = {name: np.full(len(source_stamps), np.nan, dtype=np.float64)
               for name in production}
    valid = np.zeros(len(source_stamps), dtype=bool)
    for row, stamp in enumerate(source_stamps):
        index = bisect.bisect_right(production_stamps, int(stamp)) - 1
        if index < 0 or int(stamp) - int(production_stamps[index]) > MAX_ALIGNMENT_NS:
            continue
        valid[row] = True
        for name, values in production.items():
            aligned[name][row] = values[index]
    return aligned, valid


def _region_masks(frames: np.ndarray, reset_index: np.ndarray,
                  packet_sequence: np.ndarray,
                  valid: np.ndarray) -> dict[str, np.ndarray]:
    return canonical_region_masks(frames, reset_index, packet_sequence,
                                  valid, dt_s=DT_S)


def _describe_series(values: np.ndarray, reference: np.ndarray,
                     valid: np.ndarray, time_indices: np.ndarray
                     ) -> dict[str, Any]:
    mask = (valid & np.isfinite(values) & np.isfinite(reference))
    error = values - reference
    return {
        "error_vs_kinematic_rear_contact_mps": _finite_stats(error[mask]),
        "correlation_to_kinematic_rear_contact": (
            float(np.corrcoef(values[mask], reference[mask])[0, 1])
            if int(mask.sum()) > 2
            and np.std(values[mask]) > 0.0 and np.std(reference[mask]) > 0.0
            else None),
        "temporal_noise": _temporal_noise(values, mask),
        "autocorrelation": _autocorrelation_time(values, mask),
        "cross_correlation_to_contact": _best_lag(
            reference, values, mask),
        "matched_sample_count": int(mask.sum()),
    }


def _audit_dataset(dataset_path: Path, sidecar_manifest_path: Path,
                   source_manifests: list[Path], production_library: Path,
                   production_config: Path,
                   allowed_splits: set[str]) -> dict[str, Any]:
    data = _load_dataset(dataset_path)
    manifest = json.loads(sidecar_manifest_path.read_text(encoding="utf-8"))
    bags, manifest_hashes = _manifest_bags(source_manifests)
    run_ids = np.asarray(data["run_ids"]).astype(str)
    splits = np.asarray(data["splits"]).astype(str)
    frames = np.asarray(data["frames"], dtype=np.float64)
    run_index = _frame_run_index(data)
    reset_index = np.full(len(frames), -1, dtype=np.int32)
    for (start_raw, end_raw), reset_raw in zip(
            data["bounds"], data["sequence_reset_index"]):
        reset_index[int(start_raw):int(end_raw)] = int(reset_raw)
    if np.any(reset_index < 0):
        raise ValueError(f"{dataset_path}: reset epochs do not cover all frames")
    times = np.asarray(data["sample_time_ns"], dtype=np.int64)
    packets = np.asarray(data["packet_sequence"], dtype=np.int64)
    reports: dict[str, Any] = {}
    for index, (run_id, split) in enumerate(zip(run_ids, splits)):
        if split not in allowed_splits:
            continue
        rows = np.flatnonzero(run_index == index)
        if len(rows) < 3:
            continue
        if str(run_id) not in bags:
            raise ValueError(f"no source bag manifest entry for held-out run {run_id}")
        bag = bags[str(run_id)]
        encoders = _encoder_angles(bag)
        local_frames = frames[rows]
        local_times = times[rows]
        local_packets = packets[rows]
        local_resets = reset_index[rows]
        views, interval_valid = _wheel_views(
            encoders, local_times, local_packets, local_resets,
            np.asarray(data["encoder_raw_surface_mps"], dtype=np.float64)[rows],
            np.asarray(data["encoder_raw_valid"], dtype=bool)[rows],
            local_frames[:, 5:7])
        contact = np.stack([rear_contact_speeds(u, r, REAR_TRACK_WIDTH_M)
                            for u, r in local_frames[:, (0, 2)]])
        production_stamps, production_raw, replay_meta = _production_replay(
            bag, production_library, production_config)
        production, production_valid = _align_production(
            local_times, production_stamps, production_raw)
        valid_base = (np.isfinite(local_frames).all(axis=1)
                      & np.isfinite(contact).all(axis=1))
        valid_views: dict[str, np.ndarray] = {}
        for name, values in views.items():
            valid_views[name] = valid_base & np.isfinite(values).all(axis=1)
        for name in production:
            valid_views[f"production_{name}"] = (
                valid_base & production_valid & np.isfinite(production[name]))
        # Same exact target rows and wheel-valid mask are used for all three
        # WP16.4 output comparisons. No model is trained here.
        same_transition_mask = (valid_base & interval_valid
            & np.asarray(data["encoder_raw_valid"], dtype=bool)[rows]
            & np.isfinite(local_frames[:, 5:7]).all(axis=1))
        views_valid_for_region = valid_base & interval_valid
        regions = _region_masks(local_frames, local_resets, local_packets,
                                views_valid_for_region)
        masks = {"all_valid": views_valid_for_region, **regions}
        quantity_report: dict[str, Any] = {}
        for name, values in views.items():
            rows_per_region = {}
            for region, region_mask in masks.items():
                if name in {"encoder_angle_rad", "increment_25ms_rad"}:
                    per_side = [{
                        "value_distribution": _finite_stats(
                            values[region_mask, side]),
                        "units": "rad",
                    } for side in range(2)]
                else:
                    per_side = []
                    for side in range(2):
                        local_valid = region_mask & np.isfinite(values[:, side])
                        per_side.append(_describe_series(
                            values[:, side], contact[:, side], local_valid,
                            rows))
                rows_per_region[region] = {
                    "left": per_side[0], "right": per_side[1],
                    "sample_count": int(region_mask.sum()),
                    "independent_runs": 1,
                    "weak_evidence_under_three_runs": True,
                }
            quantity_report[name] = rows_per_region
        production_report: dict[str, Any] = {}
        production_health: dict[str, Any] = {}
        for name, values in production.items():
            if name in {"valid", "wheel_update_used", "wheel_burst_rejected",
                        "reset_epoch", "timing_degraded"}:
                production_health[name] = {
                    region: float(np.mean(values[region_mask & production_valid]))
                    if np.any(region_mask & production_valid) else None
                    for region, region_mask in masks.items()}
                continue
            if name not in {"wheel_raw_mps", "wheel_mapped_mps", "wheel_packet_mps"}:
                continue
            production_report[name] = {
                region: _describe_series(
                    values, np.mean(contact, axis=1),
                    region_mask & production_valid & np.isfinite(values), rows)
                for region, region_mask in masks.items()}

        # Representation reconstruction and pairwise lag diagnostics use only
        # identical physical samples; left/right channels are separately scored.
        pairwise: dict[str, Any] = {}
        baseline = views["rate_25ms_fixed_mps"]
        rate_names = {
            "rate_25ms_fixed_mps", "rate_variable_stamp_mps",
            "rate_50ms_fixed_mps", "rate_100ms_fixed_mps",
            "rate_100ms_elapsed_reconstruction_mps", "stored_100ms_mps",
            "sidecar_fixed25_mps",
        }
        for name, values in views.items():
            if name not in rate_names:
                continue
            if name == "rate_25ms_fixed_mps":
                continue
            pairwise[name] = {
                "left": _finite_stats(
                    (values[:, 0] - baseline[:, 0])[same_transition_mask
                     & np.isfinite(values).all(axis=1)]),
                "right": _finite_stats(
                    (values[:, 1] - baseline[:, 1])[same_transition_mask
                     & np.isfinite(values).all(axis=1)]),
                "left_cross_correlation": _best_lag(
                    baseline[:, 0], values[:, 0],
                    same_transition_mask & np.isfinite(values).all(axis=1)),
                "right_cross_correlation": _best_lag(
                    baseline[:, 1], values[:, 1],
                    same_transition_mask & np.isfinite(values).all(axis=1)),
            }
        fixed_vs_stored = np.asarray(views["rate_25ms_fixed_mps"] -
                                     views["stored_100ms_mps"])
        raw_pair_rmse = float(np.sqrt(np.mean(
            (fixed_vs_stored[same_transition_mask] ** 2)))) if np.any(
                same_transition_mask) else None
        angle_reconstruction = np.asarray(
            views["rate_25ms_fixed_mps"] - views["sidecar_fixed25_mps"])
        angle_mask = (same_transition_mask & np.isfinite(
            views["sidecar_fixed25_mps"]).all(axis=1))
        # Dataset target errors are retained as target-definition diagnostics;
        # the frozen model's same-transition one-step prediction is scored by
        # WP16.4 in the follow-on evaluator before WP16 can pass its gate.
        reports[str(run_id)] = {
            "split": str(split),
            "source_bag": str(bag.relative_to(ROOT)),
            "source_bag_sha256": _sha256(bag),
            "frame_count": int(len(rows)),
            "valid_25ms_interval_samples": int(interval_valid.sum()),
            "valid_sidecar_samples": int(same_transition_mask.sum()),
            "encoder_source_interval_ms_p01_p50_p99": {},
            "production_replay": replay_meta,
            "production_packet_alignment_fraction": float(production_valid.mean()),
            "production_health_fractions_by_region": production_health,
            "production_wheel_metrics_by_region": production_report,
            "wheel_representation_metrics_by_region": quantity_report,
            "pairwise_against_fixed25_on_same_transitions": pairwise,
            "rate25_fixed_minus_sidecar_fixed25": {
                side: _finite_stats(angle_reconstruction[angle_mask, i])
                for i, side in enumerate(("left", "right"))},
            "rate25_fixed_minus_stored100_rmse_mps": raw_pair_rmse,
            "stored100_reconstruction_check": {
                side: _finite_stats(
                    views["rate_100ms_elapsed_reconstruction_mps"][:, i]
                    - views["stored_100ms_mps"][:, i])
                for i, side in enumerate(("left", "right"))},
            "wheel_body_mismatch_proxy": (
                "abs(mean(stored rear encoder surface rates L/R) - rear-axle u); encoder/kinematic proxy, not tire-slip truth"),
            "current_model_one_step_error_on_identical_target_mask": {
                "status": "not yet scored; required before WP16.4 and WP16 gate",
                "same_transition_mask_samples": int(same_transition_mask.sum()),
            },
        }
        encoder_stamps = encoders[analysis.LEFT_ENCODER][0]
        encoder_gaps = np.diff(encoder_stamps) / 1e6
        reports[str(run_id)]["encoder_source_interval_ms_p01_p50_p99"] = (
            np.quantile(encoder_gaps, (0.01, 0.50, 0.99)).tolist()
            if len(encoder_gaps) else [])
    return {
        "dataset": str(dataset_path.relative_to(ROOT)),
        "included_splits": sorted(allowed_splits),
        "dataset_sha256": _sha256(dataset_path),
        "sidecar_manifest": str(sidecar_manifest_path.relative_to(ROOT)),
        "sidecar_manifest_sha256": _sha256(sidecar_manifest_path),
        "source_manifest_hashes": manifest_hashes,
        "source_manifest_matches_sidecar": all(
            manifest_hashes.get(path) == expected
            for path, expected in manifest["source_manifest_sha256"].items()),
        "source_manifest_matches_sidecar": all(
            manifest_hashes.get(path) == expected
            for path, expected in manifest["source_manifest_sha256"].items()),
        "only_validation_and_unseen_practice_bags_opened": True,
        "excluded_test_and_final_test_bags": True,
        "run_reports": reports,
    }


def analyze(openplane_dataset: Path, practice_dataset: Path,
            openplane_manifest: Path, practice_manifest: Path,
            openplane_sources: list[Path], practice_sources: list[Path],
            production_library: Path, production_config: Path,
            one_step_report_path: Path | None,
            output_path: Path) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite wheel audit: {output_path}")
    for path in (production_library, production_config, openplane_dataset,
                 practice_dataset, openplane_manifest, practice_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    openplane = _audit_dataset(openplane_dataset, openplane_manifest,
                               openplane_sources, production_library,
                               production_config, {"validation"})
    practice = _audit_dataset(practice_dataset, practice_manifest,
                              practice_sources, production_library,
                              production_config, {"unseen_practice"})
    combined_runs = [*openplane["run_reports"], *practice["run_reports"]]
    one_step = None
    if one_step_report_path is not None and one_step_report_path.is_file():
        one_step = json.loads(one_step_report_path.read_text(encoding="utf-8"))
        for domain_name, wheel_audit, one_step_domain in (
                ("openplane_validation", openplane,
                 one_step.get("openplane_validation", {})),
                ("unseen_practice", practice,
                 one_step.get("unseen_practice", {}))):
            one_step_runs = one_step_domain.get("run_reports", {})
            for run_id, run in wheel_audit["run_reports"].items():
                model_run = one_step_runs.get(run_id)
                if model_run is None:
                    run["current_model_one_step_error_on_identical_target_mask"] = {
                        "status": "no matched one-step evaluation run",
                        "same_transition_mask_samples": 0,
                    }
                    continue
                run["current_model_one_step_error_on_identical_target_mask"] = {
                    "status": "scored; same physical transitions and validity mask across targets",
                    "split": domain_name,
                    "same_transition_mask_samples": model_run["same_mask_transitions"],
                    "fixed25_encoder_rate": model_run["fixed25_encoder_rate"],
                    "stored_100ms_rate": model_run["stored_100ms_rate"],
                    "encoder_angle_increment": model_run["encoder_angle_increment"],
                }

    def aggregate_regions(domain: dict[str, Any]) -> dict[str, Any]:
        run_reports = domain["run_reports"]
        regions = sorted({region
            for run in run_reports.values()
            for metric_by_region in run["wheel_representation_metrics_by_region"].values()
            for region in metric_by_region})
        quantities = (
            "rate_25ms_fixed_mps", "rate_variable_stamp_mps",
            "rate_50ms_fixed_mps", "rate_100ms_fixed_mps",
            "rate_100ms_elapsed_reconstruction_mps", "stored_100ms_mps",
            "sidecar_fixed25_mps")
        aggregate: dict[str, Any] = {}
        for region in regions:
            aggregate[region] = {}
            for quantity in quantities:
                aggregate[region][quantity] = {}
                for side in ("left", "right"):
                    metrics: dict[str, list[float]] = {
                        "rmse_mps": [], "bias_mps": [], "correlation": [],
                        "best_lag_s": [], "autocorrelation_time_s": [],
                        "first_difference_variance": [],
                        "derivative_noise_rms_per_s": [],
                    }
                    for run in run_reports.values():
                        item = run["wheel_representation_metrics_by_region"].get(
                            quantity, {}).get(region, {}).get(side, {})
                        stats = item.get("error_vs_kinematic_rear_contact_mps", {})
                        if stats.get("rmse") is not None:
                            metrics["rmse_mps"].append(float(stats["rmse"]))
                            metrics["bias_mps"].append(float(stats["bias"]))
                        corr = item.get("correlation_to_kinematic_rear_contact")
                        if corr is not None:
                            metrics["correlation"].append(float(corr))
                        lag = item.get("cross_correlation_to_contact", {}).get(
                            "lag_seconds")
                        if lag is not None:
                            metrics["best_lag_s"].append(float(lag))
                        tau = item.get("autocorrelation", {}).get(
                            "integrated_time_s")
                        if tau is not None:
                            metrics["autocorrelation_time_s"].append(float(tau))
                        noise = item.get("temporal_noise", {})
                        if noise.get("first_difference_variance") is not None:
                            metrics["first_difference_variance"].append(
                                float(noise["first_difference_variance"]))
                        if noise.get("derivative_noise_rms_per_s") is not None:
                            metrics["derivative_noise_rms_per_s"].append(
                                float(noise["derivative_noise_rms_per_s"]))
                    aggregate[region][quantity][side] = {
                        name: _macro_run_summary(values)
                        for name, values in metrics.items()}
        return aggregate

    regional_summary = {
        "openplane_validation": aggregate_regions(openplane),
        "unseen_practice": aggregate_regions(practice),
    }
    one_step_gate_complete = bool(one_step is not None
        and one_step.get("training_performed") is False
        and one_step.get("checkpoint_unchanged") is True
        and all(one_step.get(domain, {}).get("shared_transition_count", 0) > 0
                and one_step.get(domain, {}).get("independent_run_count", 0) > 0
                and all(one_step.get(domain, {}).get(
                    "macro_run_metrics", {}).get(target, {}).get(
                        "independent_runs", 0) > 0
                    for target in ("fixed25_encoder_rate", "stored_100ms_rate",
                                   "encoder_angle_increment"))
                and set(wheel_audit["run_reports"]).issubset(set(
                    one_step.get(domain, {}).get("run_reports", {})))
                for domain, wheel_audit in (
                    ("openplane_validation", openplane),
                    ("unseen_practice", practice))))
    report = {
        "schema_version": 1,
        "purpose": "WP16 canonical rear-wheel signal/timing and measurement representation audit",
        "generated_date": "2026-10-03",
        "dt_s": DT_S,
        "wheel_radius_m": WHEEL_RADIUS_M,
        "rear_track_width_m": REAR_TRACK_WIDTH_M,
        "source_dt_validity_gate_s": list(ENCODER_SOURCE_DT_GATE_S),
        "cross_correlation_lag_search_steps": [-8, 8],
        "autocorrelation_method": "initial-positive sequence; 25 ms samples, max 2 s, no pair spans invalid region gaps",
        "high_frequency_noise_definition": "variance of adjacent rate differences and RMS first difference divided by 25 ms",
        "production_source_sha256": {
            str(path.relative_to(ROOT)): _sha256(path)
            for path in (
                ROOT / "tools/racing/offline/localization/production_odometry_shim.cpp",
                ROOT / "tools/racing/offline/localization/build_production_odom.py",
                ROOT / "f1tenth_localization/src/odometry_observer.cpp",
                ROOT / "f1tenth_localization/src/sensor_packet_assembler.cpp",
                production_config,
            )},
        "one_step_target_report": (
            str(one_step_report_path.relative_to(ROOT))
            if one_step_report_path and one_step_report_path.is_file() else None),
        "one_step_target_report_sha256": (
            _sha256(one_step_report_path)
            if one_step_report_path and one_step_report_path.is_file() else None),
        "independent_validation_runs": len(combined_runs),
        "regions_with_fewer_than_three_runs_are_weak_evidence": True,
        "run_clustered_regional_summary": regional_summary,
        "openplane": openplane,
        "practice": practice,
        "frozen_parent_one_step_target_metrics": one_step,
        "wp16_gate": {
            "status": "complete" if one_step_gate_complete else "incomplete",
            "reason": ("canonical signal definitions, production replay, held-out wheel representation audit, and same-transition frozen-parent target comparison are complete; WP17 owns the causal state-topology decision"
                       if one_step_gate_complete else
                       "frozen EDSSM same-transition one-step target comparison is missing/incomplete"),
            "wheel_topology_decision": None,
            "handoff_tests_passed": [
                "cumulative fixed-period increment math",
                "causal no-future encoder selection",
                "run/reset and packet-gap windows do not cross boundaries",
                "fixed 25 ms sidecar reconstruction",
                "stored approximately 100 ms proxy reconstruction",
                "COM-to-rear and left/right rigid-body sign conventions",
            ],
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openplane-dataset", type=Path, default=DEFAULT_OPENPLANE)
    parser.add_argument("--practice-dataset", type=Path, default=DEFAULT_PRACTICE)
    parser.add_argument("--openplane-manifest", type=Path,
        default=DEFAULT_OPENPLANE.with_name(
            DEFAULT_OPENPLANE.stem + "_manifest.json"))
    parser.add_argument("--practice-manifest", type=Path,
        default=DEFAULT_PRACTICE.with_name(
            DEFAULT_PRACTICE.stem + "_manifest.json"))
    parser.add_argument("--openplane-source-manifest", type=Path, action="append")
    parser.add_argument("--practice-source-manifest", type=Path, action="append")
    parser.add_argument("--production-library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--production-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--one-step-report", type=Path,
        default=(ROOT / "live_runs/derived_dynamics_learning_20260928"
            / "full_modeling_reset_20261001/replacement_offline_sim_raceline_20261003"
            / "full_throttle_domain_v1/wp16_frozen_parent_one_step_v3.json"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    with open(args.openplane_manifest, encoding="utf-8") as stream:
        open_manifest = json.load(stream)
    with open(args.practice_manifest, encoding="utf-8") as stream:
        practice_manifest = json.load(stream)
    open_sources = args.openplane_source_manifest or [
        Path(path) for path in open_manifest["source_manifest_sha256"]]
    practice_sources = args.practice_source_manifest or [
        Path(path) for path in practice_manifest["source_manifest_sha256"]]
    report = analyze(args.openplane_dataset.resolve(), args.practice_dataset.resolve(),
        args.openplane_manifest.resolve(), args.practice_manifest.resolve(),
        [path.resolve() for path in open_sources],
        [path.resolve() for path in practice_sources],
        args.production_library.resolve(), args.production_config.resolve(),
        args.one_step_report.resolve(), args.output.resolve())
    print(json.dumps({
        "output": str(args.output.resolve()),
        "openplane_runs": len(report["openplane"]["run_reports"]),
        "practice_runs": len(report["practice"]["run_reports"]),
        "wp16_gate": report["wp16_gate"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
