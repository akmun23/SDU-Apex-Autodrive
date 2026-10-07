#!/usr/bin/env python3
"""Fit and whole-run evaluate local yaw-transition models.

Uses only clean, unique train/validation captures. Test/final-test archives are
excluded from their manifests before any numeric arrays are read. Models are
diagnostic GT-labelled one-step transition specialists, not runtime MPC code.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
LIVE_RUNS = ROOT / "live_runs"
DT_S = 0.025
COM_X_M = 0.15532
SPEED_BIN_MPS = 0.5
STEERING_BIN_RAD = 0.025
SPEED_CENTERS = np.arange(0.25, 12.0, SPEED_BIN_MPS)
STEERING_CENTERS = np.arange(-0.525, 0.525 + 1.0e-9, STEERING_BIN_RAD)
MIN_CELL_SAMPLES = 40
MIN_CELL_RUNS = 2
MIN_SAMPLES_PER_RUN = 8
RIDGE = 1.0
HUBER_K = 1.345
FEATURE_NAMES = (
    "current_yaw_rate_radps",
    "previous_yaw_rate_increment_radps",
    "speed_offset_from_cell_center_mps",
    "steering_offset_from_cell_center_rad",
    "measured_steering_rate_radps",
    "throttle_feedback_norm",
    "throttle_feedback_rate_per_s",
    "rear_wheel_mean_minus_body_speed_mps",
    "rear_axle_lateral_speed_mps",
    "body_speed_rate_mps2",
)
# Fixed physical scales avoid amplifying near-constant command/feedback
# differences by dividing by a tiny per-cell standard deviation.
FEATURE_SCALES = np.asarray((1.0, 0.1, 0.25, 0.0125, 1.6,
                             0.25, 1.0, 0.5, 0.5, 2.0), dtype=np.float64)
PHASES = (-1, 0, 1)  # unwind, steady/transition-neutral, turn-in


@dataclass
class RunSeries:
    run_id: str
    split: str
    source: str
    frames: np.ndarray
    rigid: np.ndarray
    bounds: np.ndarray


def _clean_run(row: dict[str, Any], split: str) -> bool:
    return (
        row.get("effective_split") == split
        and not row.get("aborted")
        and row.get("reason") == "schedule complete"
        and bool(row.get("clean_stream_and_collision_gate"))
        and not row.get("quality_failures")
        and not row.get("whole_bag_quality_failures")
        and int(row.get("timing_faults", -1)) == 0
        and all(int(value) == 0 for value in row.get("collisions", []))
    )


def _discover_run_series() -> tuple[list[RunSeries], dict[str, Any]]:
    candidates = []
    run_split_owner: dict[str, str] = {}
    skipped = Counter()
    for npz_path in LIVE_RUNS.rglob("openplane_dynamics.npz"):
        manifest_path = npz_path.with_name("manifest.json")
        if not manifest_path.is_file():
            skipped["missing_manifest"] += 1
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            skipped["bad_manifest"] += 1
            continue
        rows = manifest.get("runs", [])
        if not rows:
            skipped["manifest_without_run_provenance"] += 1
            continue
        splits = {row.get("effective_split") for row in rows}
        if len(splits) != 1:
            skipped["mixed_or_missing_split"] += 1
            continue
        split = next(iter(splits))
        if split not in ("train", "validation"):
            skipped[f"excluded_{split}"] += 1
            continue
        if not all(_clean_run(row, split) for row in rows):
            skipped[f"failed_clean_{split}_gate"] += 1
            continue
        run_ids = tuple(str(row.get("run_id", "")) for row in rows)
        if not all(run_ids) or len(set(run_ids)) != len(run_ids):
            skipped["invalid_run_ids"] += 1
            continue
        for run_id in run_ids:
            owner = run_split_owner.get(run_id)
            if owner is not None and owner != split:
                raise ValueError(f"run {run_id} appears in both {owner} and {split}")
            run_split_owner[run_id] = split
        sample_total = sum(int(row.get("samples_exported", 0)) for row in rows)
        candidates.append((len(run_ids), -sample_total, split, run_ids,
                           npz_path, rows))

    candidates.sort(key=lambda row: (row[0], row[1], row[2], row[3], str(row[4])))
    seen_run_ids: set[str] = set()
    seen_fingerprints: set[str] = set()
    output: list[RunSeries] = []
    for _, _, split, expected_ids, npz_path, manifest_rows in candidates:
        try:
            with np.load(npz_path, allow_pickle=False) as archive:
                required_meta = ("run_ids", "run_splits")
                if any(key not in archive.files for key in required_meta):
                    skipped["missing_split_metadata"] += 1
                    continue
                archive_ids = tuple(str(value) for value in archive["run_ids"])
                archive_splits = tuple(str(value) for value in archive["run_splits"])
                if (archive_ids != expected_ids or not archive_splits
                        or set(archive_splits) != {split}):
                    skipped["archive_manifest_mismatch"] += 1
                    continue
                needed = ("frames", "simulator_rigid_state", "sequence_bounds")
                if any(key not in archive.files for key in needed):
                    skipped["missing_motion_arrays"] += 1
                    continue
                frames = np.asarray(archive["frames"], dtype=np.float32)
                rigid = np.asarray(archive["simulator_rigid_state"], dtype=np.float32)
                bounds = np.asarray(archive["sequence_bounds"], dtype=np.int64)
                dt = (np.asarray(archive["dt_s"], dtype=np.float32)
                      if "dt_s" in archive.files else np.empty(0))
                sequence_run_index = (
                    np.asarray(archive["sequence_run_index"], dtype=np.int64)
                    if "sequence_run_index" in archive.files else None)
        except (OSError, ValueError, KeyError, EOFError):
            skipped["unreadable_archive"] += 1
            continue
        if (frames.ndim != 2 or frames.shape[1] != 9
                or rigid.shape != (len(frames), 13)
                or bounds.ndim != 2 or bounds.shape[1] != 2
                or len(dt) != len(frames)
                or not np.allclose(dt, DT_S, rtol=0.0, atol=1.0e-7)
                or not np.isfinite(frames).all()
                or not np.isfinite(rigid).all()):
            skipped["invalid_array_schema"] += 1
            continue
        if sequence_run_index is not None and len(sequence_run_index) != len(bounds):
            skipped["sequence_run_index_mismatch"] += 1
            continue
        if len(expected_ids) > 1 and sequence_run_index is None:
            skipped["multi_run_archive_without_sequence_index"] += 1
            continue
        for run_index, run_id in enumerate(expected_ids):
            if run_id in seen_run_ids:
                skipped["duplicate_run_view"] += 1
                continue
            row = manifest_rows[run_index]
            fingerprint = str(row.get("fingerprint", ""))
            if fingerprint and fingerprint in seen_fingerprints:
                skipped["duplicate_fingerprint"] += 1
                seen_run_ids.add(run_id)
                continue
            sequence_ids = (np.arange(len(bounds), dtype=np.int64)
                            if sequence_run_index is None else
                            np.flatnonzero(sequence_run_index == run_index))
            local_frames, local_rigid, local_bounds = [], [], []
            cursor = 0
            for sequence_id in sequence_ids:
                begin, end = map(int, bounds[int(sequence_id)])
                if begin < 0 or end > len(frames) or end - begin < 3:
                    continue
                local_frames.append(frames[begin:end])
                local_rigid.append(rigid[begin:end])
                local_bounds.append((cursor, cursor + end - begin))
                cursor += end - begin
            if not local_frames:
                skipped["run_without_sequences"] += 1
                continue
            output.append(RunSeries(
                run_id, split, str(npz_path.relative_to(ROOT)),
                np.concatenate(local_frames), np.concatenate(local_rigid),
                np.asarray(local_bounds, dtype=np.int64)))
            seen_run_ids.add(run_id)
            if fingerprint:
                seen_fingerprints.add(fingerprint)
    output.sort(key=lambda series: (series.split, series.run_id))
    return output, {
        "selected_runs": {split: sum(row.split == split for row in output)
                          for split in ("train", "validation")},
        "selected_archives": sorted({row.source for row in output}),
        "selected_samples": {split: sum(len(row.frames) for row in output
                                         if row.split == split)
                             for split in ("train", "validation")},
        "skipped_archive_reasons": dict(skipped),
        "test_and_final_test_arrays_read": False,
    }


def _rear_lateral_velocity(rigid: np.ndarray) -> np.ndarray:
    # Rigid-state velocity is body-frame COM velocity, not world-frame velocity.
    return rigid[:, 8] - COM_X_M * rigid[:, 12]


def _make_rows(series: RunSeries, phase_threshold: float = 0.05):
    if phase_threshold < 0.0 or not math.isfinite(phase_threshold):
        raise ValueError("phase threshold must be finite and non-negative")
    frames = series.frames.astype(np.float64, copy=False)
    rigid = series.rigid.astype(np.float64, copy=False)
    speed = np.hypot(rigid[:, 7], rigid[:, 8])
    yaw = rigid[:, 12]
    lateral = _rear_lateral_velocity(rigid)
    features, targets, cells, phases, ids, sequence_ids, frame_indices = (
        [], [], [], [], [], [], [])
    for sequence_index, (begin_raw, end_raw) in enumerate(series.bounds):
        begin, end = int(begin_raw), int(end_raw)
        for k in range(begin + 2, end - 1):
            v = float(speed[k])
            delta = float(frames[k, 3])
            if not (0.0 <= v < 12.0 and -0.525 <= delta <= 0.525):
                continue
            speed_cell = min(23, int(v // SPEED_BIN_MPS))
            steer_cell = int(np.clip(round((delta + 0.525) /
                                           STEERING_BIN_RAD), 0, 42))
            speed_offset = v - float(SPEED_CENTERS[speed_cell])
            steering_offset = delta - float(STEERING_CENTERS[steer_cell])
            delta_rate = (delta - float(frames[k - 1, 3])) / DT_S
            throttle_rate = (float(frames[k, 4]) - float(frames[k - 1, 4])) / DT_S
            speed_rate = (v - float(speed[k - 1])) / DT_S
            wheel_mismatch = 0.5 * float(frames[k, 5] + frames[k, 6]) - v
            feature = (
                float(yaw[k]),
                float(yaw[k] - yaw[k - 1]),
                speed_offset,
                steering_offset,
                delta_rate,
                float(frames[k, 4]),
                throttle_rate,
                wheel_mismatch,
                float(lateral[k]),
                speed_rate,
            )
            features.append(feature)
            targets.append(float(yaw[k + 1] - yaw[k]))
            cells.append((speed_cell, steer_cell))
            response_phase = delta_rate * delta
            phases.append(1 if response_phase > phase_threshold else
                          (-1 if response_phase < -phase_threshold else 0))
            ids.append(series.run_id)
            sequence_ids.append(sequence_index)
            frame_indices.append(k)
    return (np.asarray(features, dtype=np.float64),
            np.asarray(targets, dtype=np.float64),
            np.asarray(cells, dtype=np.int16),
            np.asarray(phases, dtype=np.int8),
            np.asarray(ids, dtype=object),
            np.asarray(sequence_ids, dtype=np.int32),
            np.asarray(frame_indices, dtype=np.int64))


def _metrics(errors: np.ndarray) -> dict[str, Any]:
    if not len(errors):
        return {"samples": 0}
    absolute = np.abs(errors)
    return {
        "samples": int(len(errors)),
        "rmse_radps": float(np.sqrt(np.mean(errors * errors))),
        "mae_radps": float(np.mean(absolute)),
        "p95_abs_radps": float(np.quantile(absolute, 0.95)),
        "p99_abs_radps": float(np.quantile(absolute, 0.99)),
        "max_abs_radps": float(np.max(absolute)),
        "fraction_abs_error_below_0p1": float(np.mean(absolute < 0.1)),
    }


def _fit_model(x: np.ndarray, target_delta: np.ndarray, run_ids: np.ndarray,
               min_samples: int = MIN_CELL_SAMPLES) -> dict[str, Any] | None:
    run_counts = Counter(run_ids.tolist())
    supported_runs = sorted(run_id for run_id, count in run_counts.items()
                            if count >= MIN_SAMPLES_PER_RUN)
    use = np.isin(run_ids, supported_runs)
    x, target_delta, run_ids = x[use], target_delta[use], run_ids[use]
    if len(x) < min_samples or len(supported_runs) < MIN_CELL_RUNS:
        return None
    mean = np.mean(x, axis=0)
    z = (x - mean) / FEATURE_SCALES
    design = np.column_stack((np.ones(len(z)), z))
    run_weight = np.zeros(len(run_ids), dtype=np.float64)
    for run_id in supported_runs:
        mask = run_ids == run_id
        run_weight[mask] = 1.0 / np.count_nonzero(mask)
    run_weight *= len(supported_runs) / run_weight.sum()
    weight = run_weight.copy()
    coefficients = np.zeros(design.shape[1], dtype=np.float64)
    for _ in range(8):
        gram = design.T @ (weight[:, None] * design)
        penalty = np.eye(gram.shape[0]) * RIDGE
        penalty[0, 0] = 0.0
        coefficients = np.linalg.solve(
            gram + penalty, design.T @ (weight * target_delta))
        residual = target_delta - design @ coefficients
        robust_scale = 1.4826 * float(np.median(np.abs(
            residual - np.median(residual))))
        if robust_scale < 1.0e-8:
            break
        cutoff = HUBER_K * robust_scale
        robust_weight = np.minimum(1.0, cutoff / np.maximum(np.abs(residual), 1.0e-12))
        weight = run_weight * robust_weight
        weight *= len(supported_runs) / weight.sum()
    return {
        "mean": mean,
        "coefficients": coefficients,
        "training_samples": int(len(x)),
        "training_runs": len(supported_runs),
        "training_run_ids": supported_runs,
    }


def _predict(model: dict[str, Any], x: np.ndarray) -> float:
    z = (x - model["mean"]) / FEATURE_SCALES
    delta = float(model["coefficients"][0] +
                  np.dot(model["coefficients"][1:], z))
    return float(x[0] + delta)


def _bilinear(models: dict[tuple[int, ...], dict[str, Any]], x: np.ndarray,
              speed: float, steering: float, phase: int,
              phase_specific: bool) -> float | None:
    sc = (speed - SPEED_CENTERS[0]) / SPEED_BIN_MPS
    dc = (steering - STEERING_CENTERS[0]) / STEERING_BIN_RAD
    if not (0.0 <= sc < len(SPEED_CENTERS) - 1
            and 0.0 <= dc < len(STEERING_CENTERS) - 1):
        return None
    s0, d0 = int(math.floor(sc)), int(math.floor(dc))
    fs, fd = sc - s0, dc - d0
    base = ((s0, d0), (s0 + 1, d0), (s0, d0 + 1), (s0 + 1, d0 + 1))
    keys = tuple((*key, phase) if phase_specific else key for key in base)
    if any(key not in models for key in keys):
        return None
    predictions = [_predict(models[key], x) for key in keys]
    low = (1.0 - fs) * predictions[0] + fs * predictions[1]
    high = (1.0 - fs) * predictions[2] + fs * predictions[3]
    return float((1.0 - fd) * low + fd * high)


def run(output: Path, phase_threshold: float = 0.05) -> dict[str, Any]:
    if phase_threshold < 0.0 or not math.isfinite(phase_threshold):
        raise ValueError("phase threshold must be finite and non-negative")
    run_series, source_audit = _discover_run_series()
    train_series = [row for row in run_series if row.split == "train"]
    validation_series = [row for row in run_series if row.split == "validation"]
    if not train_series or not validation_series:
        raise RuntimeError("clean whole-run train and validation captures are required")
    train_parts = [_make_rows(row, phase_threshold) for row in train_series]
    valid_parts = [_make_rows(row, phase_threshold) for row in validation_series]

    def combine(parts):
        return tuple(np.concatenate([part[i] for part in parts], axis=0)
                     for i in range(6))

    tx, ty, tc, tp, tr, _ = combine(train_parts)
    vx, vy_delta, vc, vp, vr, _ = combine(valid_parts)
    keys_unphased: dict[tuple[int, int], dict[str, Any]] = {}
    keys_phased: dict[tuple[int, int, int], dict[str, Any]] = {}
    train_count = np.zeros((24, 43), dtype=np.int64)
    train_runs = np.zeros((24, 43), dtype=np.int64)
    valid_count = np.zeros((24, 43), dtype=np.int64)
    for s, d in set(map(tuple, tc.tolist())):
        mask = (tc[:, 0] == s) & (tc[:, 1] == d)
        train_count[s, d] = int(mask.sum())
        train_runs[s, d] = len(set(tr[mask].tolist()))
        model = _fit_model(tx[mask], ty[mask], tr[mask])
        if model is not None:
            keys_unphased[(s, d)] = model
        for phase in PHASES:
            pmask = mask & (tp == phase)
            model = _fit_model(tx[pmask], ty[pmask], tr[pmask])
            if model is not None:
                keys_phased[(s, d, phase)] = model
    for s, d in set(map(tuple, vc.tolist())):
        valid_count[s, d] = int(np.count_nonzero(
            (vc[:, 0] == s) & (vc[:, 1] == d)))

    methods = ("persistence", "direct_cell", "direct_phase",
               "bilinear_cell", "bilinear_phase")
    error_by_run: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list))
    error_by_band: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list))
    error_by_cell: dict[tuple[int, int], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list))
    error_by_phase: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list))
    for x, y_delta, cell, phase, run_id_raw in zip(vx, vy_delta, vc, vp, vr):
        s, d = map(int, cell)
        run_id = str(run_id_raw)
        # Recover absolute values from the feature's local cell offsets.
        speed = float(SPEED_CENTERS[s] + x[2])
        steering = float(STEERING_CENTERS[d] + x[3])
        target = float(x[0] + y_delta)
        predictions = {
            "persistence": float(x[0]),
            "direct_cell": (_predict(keys_unphased[(s, d)], x)
                            if (s, d) in keys_unphased else None),
            "direct_phase": (_predict(keys_phased[(s, d, int(phase))], x)
                             if (s, d, int(phase)) in keys_phased else None),
            "bilinear_cell": _bilinear(keys_unphased, x, speed, steering,
                                       int(phase), False),
            "bilinear_phase": _bilinear(keys_phased, x, speed, steering,
                                        int(phase), True),
        }
        for method in methods:
            pred = predictions[method]
            if pred is None:
                continue
            error = pred - target
            error_by_run[run_id][method].append(error)
            error_by_band[s][method].append(error)
            error_by_cell[(s, d)][method].append(error)
            error_by_phase[int(phase)][method].append(error)
            if method != "persistence":
                baseline_name = f"persistence_same_{method}_support"
                baseline_error = float(x[0] - target)
                error_by_run[run_id][baseline_name].append(baseline_error)
                error_by_band[s][baseline_name].append(baseline_error)
                error_by_cell[(s, d)][baseline_name].append(baseline_error)
                error_by_phase[int(phase)][baseline_name].append(baseline_error)

    def flatten(method: str) -> np.ndarray:
        return np.asarray([e for values in error_by_run.values()
                           for e in values.get(method, [])], dtype=np.float64)

    summary = {}
    for method in methods:
        errors = flatten(method)
        summary[method] = {
            "coverage_samples": int(len(errors)),
            "coverage_fraction": float(len(errors) / len(vy_delta)),
            "metrics": _metrics(errors),
        }
        if method != "persistence":
            baseline_name = f"persistence_same_{method}_support"
            summary[method]["persistence_baseline_same_coverage"] = _metrics(
                flatten(baseline_name))
    per_run = {
        run_id: {method: _metrics(np.asarray(errors, dtype=np.float64))
                 for method, errors in sorted(methods_for_run.items())}
        for run_id, methods_for_run in sorted(error_by_run.items())
    }
    run_uncertainty = {}
    rng = np.random.default_rng(20261007)
    for method in methods:
        if method == "persistence":
            continue
        baseline_name = f"persistence_same_{method}_support"
        paired = [
            per_run[run_id][method]["rmse_radps"]
            - per_run[run_id][baseline_name]["rmse_radps"]
            for run_id in per_run
            if method in per_run[run_id]
            and baseline_name in per_run[run_id]
        ]
        values = np.asarray(paired, dtype=np.float64)
        if len(values):
            draws = rng.choice(values, size=(10000, len(values)), replace=True)
            # Report the paired run-macro RMSE difference itself; bootstrap
            # its mean to preserve whole-capture clustering.
            boot_mean = np.mean(draws, axis=1)
            run_uncertainty[method] = {
                "independent_runs": int(len(values)),
                "mean_run_rmse_difference_vs_persistence_radps": float(np.mean(values)),
                "median_run_rmse_difference_vs_persistence_radps": float(np.median(values)),
                "run_wins": int(np.count_nonzero(values < 0.0)),
                "run_losses": int(np.count_nonzero(values > 0.0)),
                "bootstrap_95pct_ci_of_mean_difference_radps": [
                    float(np.quantile(boot_mean, 0.025)),
                    float(np.quantile(boot_mean, 0.975)),
                ],
            }
    bands = []
    for s in range(24):
        band = {
            "speed_band_mps": [s * 0.5, (s + 1) * 0.5],
            "train_samples": int(train_count[s].sum()),
            "validation_samples": int(valid_count[s].sum()),
            **{method: _metrics(np.asarray(
                error_by_band[s].get(method, []), dtype=np.float64))
               for method in methods},
        }
        for method in methods:
            if method != "persistence":
                baseline_name = f"persistence_same_{method}_support"
                band[baseline_name] = _metrics(np.asarray(
                    error_by_band[s].get(baseline_name, []), dtype=np.float64))
        bands.append(band)
    per_cell = []
    for s in range(24):
        for d in range(43):
            if not valid_count[s, d]:
                continue
            values = error_by_cell.get((s, d), {})
            row = {
                "speed_cell": s,
                "speed_center_mps": float(SPEED_CENTERS[s]),
                "steering_cell": d,
                "steering_center_rad": float(STEERING_CENTERS[d]),
                "training_samples": int(train_count[s, d]),
                "training_runs": int(train_runs[s, d]),
                "validation_samples": int(valid_count[s, d]),
            }
            for method in methods:
                row[method] = _metrics(np.asarray(
                    values.get(method, []), dtype=np.float64))
                if method != "persistence":
                    row[f"persistence_same_{method}_support"] = _metrics(
                        np.asarray(values.get(
                            f"persistence_same_{method}_support", []),
                            dtype=np.float64))
            per_cell.append(row)
    phase_grid = {}
    for phase in PHASES:
        phase_grid[str(phase)] = {
            "train_supported_cells": int(sum(
                key[2] == phase for key in keys_phased)),
            "validation_cells_with_at_least_30_samples": int(sum(
                np.count_nonzero((vc[:, 0] == s) & (vc[:, 1] == d)
                                 & (vp == phase)) >= 30
                for s in range(24) for d in range(43))),
            "validation_samples": int(np.count_nonzero(vp == phase)),
            "direct_phase_metrics": _metrics(np.asarray(
                error_by_phase[phase].get("direct_phase", []), dtype=np.float64)),
            "bilinear_phase_metrics": _metrics(np.asarray(
                error_by_phase[phase].get("bilinear_phase", []), dtype=np.float64)),
        }
    model_rows = []
    for family, models in (("cell", keys_unphased), ("phase", keys_phased)):
        for key, model in sorted(models.items()):
            s, d = key[:2]
            model_rows.append({
                "family": family,
                "speed_cell": s,
                "speed_center_mps": float(SPEED_CENTERS[s]),
                "steering_cell": d,
                "steering_center_rad": float(STEERING_CENTERS[d]),
                "phase": key[2] if family == "phase" else None,
                "training_samples": model["training_samples"],
                "training_runs": model["training_runs"],
                "training_run_ids": model["training_run_ids"],
                "feature_mean": model["mean"].tolist(),
                "feature_scales": FEATURE_SCALES.tolist(),
                "coefficients_on_delta_yaw_rate": model["coefficients"].tolist(),
            })

    report = {
        "title": "Full-band phase-conditioned local yaw-transition atlas",
        "sample_period_s": DT_S,
        "source_audit": source_audit,
        "domain": {
            "requested_speed_mps": [0.0, 12.0],
            "observed_max_speed_mps": float(max(
                np.max(np.hypot(run.rigid[:, 7], run.rigid[:, 8]))
                for run in run_series)),
            "observed_max_abs_steering_rad": float(max(
                np.max(np.abs(run.frames[:, 3])) for run in run_series)),
            "full_cartesian_domain_supported": False,
            "unsupported_cells_filled": False,
        },
        "model": {
            "target": "simulator-truth yaw_rate[k+1] from current time k",
            "training_target_representation": "yaw_rate[k+1] - yaw_rate[k]",
            "local_axes": ["0.5 m/s speed bins", "0.025 rad signed steering bins"],
            "phase": {"-1": "steering unwind", "0": "near-steady or low steering-rate transition",
                      "1": "steering turn-in"},
            "phase_threshold_rad2_per_s": phase_threshold,
            "feature_names": list(FEATURE_NAMES),
            "feature_scales": FEATURE_SCALES.tolist(),
            "features_at_k_only": True,
            "future_truth_used_as_input": False,
            "runtime_integration": "none",
            "feature_scaling": "fixed physical scales, not per-cell standard deviations",
            "fit": "run-balanced Huber ridge on yaw-rate increment; fit phases separately",
            "ridge": RIDGE,
            "minimum_training_samples_per_cell_phase": MIN_CELL_SAMPLES,
            "minimum_independent_training_runs": MIN_CELL_RUNS,
            "minimum_samples_per_contributing_run": MIN_SAMPLES_PER_RUN,
        },
        "coverage": {
            "grid_shape_speed_by_signed_steering": [24, 43],
            "unphased_trained_cells": len(keys_unphased),
            "turn_phase_trained_cells": int(sum(k[2] == 1 for k in keys_phased)),
            "unwind_phase_trained_cells": int(sum(k[2] == -1 for k in keys_phased)),
            "steady_phase_trained_cells": int(sum(k[2] == 0 for k in keys_phased)),
            "phase_detail": phase_grid,
            "train_sample_count_grid": train_count.tolist(),
            "train_run_support_grid": train_runs.tolist(),
            "validation_sample_count_grid": valid_count.tolist(),
        },
        "validation": {
            "prediction_methods": summary,
            "run_cluster_uncertainty": run_uncertainty,
            "per_run": per_run,
            "per_speed_band": bands,
            "per_speed_steering_cell": per_cell,
        },
        "coefficient_models": model_rows,
        "interpretation_limits": [
            "Only clean, unique whole-run train/validation archives were read; test/final-test and mixed-split sample arrays were not opened.",
            "This predicts one 25 ms yaw-rate transition from current GT state and measured actuator/encoder features; it is not a sensor-only observer or recursively validated plant.",
            "Phase-local models abstain when their cell lacks adequate training support; no neighboring fallback is used for direct-cell predictions.",
            "Bilinear scores require all four independently trained neighboring cells in the same phase; coverage is reported, not imputed.",
            "The observed data stop below 12 m/s and do not cover every physically possible speed/steering combination.",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    path = output / "fullband_yaw_regime_atlas_report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    print(f"artifact: {path}")
    print("unique runs:", source_audit["selected_runs"])
    print("unphased / turn-in / unwind / steady cells:",
          len(keys_unphased), sum(k[2] == 1 for k in keys_phased),
          sum(k[2] == -1 for k in keys_phased), sum(k[2] == 0 for k in keys_phased))
    for method, result in summary.items():
        print(method, result)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "live_runs/racing_model_diagnostics_20261007/"
                "fullband_yaw_regime_atlas_v3",
    )
    parser.add_argument("--phase-threshold", type=float, default=0.05)
    args = parser.parse_args()
    run(args.output, args.phase_threshold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
