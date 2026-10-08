#!/usr/bin/env python3
"""Fit a causal, direct multi-horizon yaw-rate teacher from admitted captures.

This is an offline research model, not runtime MPC/odometry code. It predicts
simulator-truth yaw rate at several future horizons using only current and past
IMU, encoder, actuator-feedback and command observations plus the candidate
steering/throttle sequence up to that horizon. Simulator-oracle odometry is
never a model input. Future measured feedback and future truth are never
features. Only clean train/validation archive rows are used; test/final-test
arrays are excluded before opening numeric arrays.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

try:
    import fit_fullband_yaw_regime_atlas as atlas
except ModuleNotFoundError:  # Importable as either a script or a repo module.
    from tools.racing.specialists import fit_fullband_yaw_regime_atlas as atlas


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = (ROOT / "live_runs/racing_model_diagnostics_20261007/"
                  "yaw_multihorizon_teacher_v1")
HORIZONS = (1, 4, 10, 20, 30, 40)  # 25, 100, 250, 500, 750, 1000 ms.
HISTORY_LAGS = (0, 1, 2, 4, 8, 16, 32, 64)  # Up to 1.6 s at 40 Hz.
FUTURE_COMMAND_STEPS = 40
OBSERVATION_NAMES = (
    "steering_feedback_rad", "throttle_feedback_norm",
    "rear_left_surface_mps", "rear_right_surface_mps",
    "imu_ax_mps2", "imu_ay_mps2", "imu_yaw_rate_rps",
    "steering_command_rad", "throttle_command_norm",
    "imu_roll_rad", "imu_roll_rate_rps",
)
HISTORY_VARIANTS = {
    "current": (0,),
    "history_1p6s": HISTORY_LAGS,
}
ESTIMATOR = {
    "n_estimators": 80,
    "max_depth": 10,
    "min_samples_leaf": 6,
    "max_features": 0.8,
    "random_state": 20261007,
    "n_jobs": -1,
}


@dataclass
class CaptureArrays:
    run_id: str
    split: str
    sensors: np.ndarray
    sensor_valid: np.ndarray
    attitude: np.ndarray
    attitude_valid: np.ndarray
    rigid: np.ndarray
    bounds: np.ndarray
    path: str


def _read_run(series: atlas.RunSeries) -> CaptureArrays:
    """Read sensor channels only from an archive already admitted by atlas."""
    path = ROOT / series.source
    with np.load(path, allow_pickle=False) as archive:
        run_ids = archive["run_ids"].astype(str).tolist()
        if series.run_id not in run_ids:
            raise ValueError(f"run provenance mismatch: {series.run_id}")
        run_index = run_ids.index(series.run_id)
        required = (
            "sensor_frames", "sensor_valid", "imu_attitude_frames",
            "imu_attitude_valid", "simulator_rigid_state", "sequence_bounds",
            "sequence_run_index", "sensor_feature_names", "attitude_feature_names",
        )
        missing = [name for name in required if name not in archive.files]
        if missing:
            raise ValueError(f"{series.run_id}: missing arrays {missing}")
        sensor_names = archive["sensor_feature_names"].astype(str).tolist()
        attitude_names = archive["attitude_feature_names"].astype(str).tolist()
        expected_sensors = [
            "steering_feedback_rad", "throttle_feedback_norm",
            "rear_left_surface_mps", "rear_right_surface_mps",
            "imu_ax_mps2", "imu_ay_mps2", "imu_yaw_rate_rps",
            "steering_command_rad", "throttle_command_norm",
        ]
        if sensor_names[:9] != expected_sensors:
            raise ValueError(f"{series.run_id}: unexpected sensor schema")
        if attitude_names[:3] != ["imu_roll_rad", "imu_pitch_rad",
                                  "imu_roll_rate_rps"]:
            raise ValueError(f"{series.run_id}: unexpected IMU attitude schema")

        indices = np.flatnonzero(archive["sequence_run_index"] == run_index)
        source_bounds = archive["sequence_bounds"]
        arrays: dict[str, list[np.ndarray]] = defaultdict(list)
        local_bounds = []
        cursor = 0
        for sequence_index in indices:
            begin, end = map(int, source_bounds[int(sequence_index)])
            if end - begin < max(HISTORY_LAGS) + max(HORIZONS) + 1:
                continue
            for key in ("sensor_frames", "sensor_valid",
                        "imu_attitude_frames", "imu_attitude_valid",
                        "simulator_rigid_state"):
                arrays[key].append(np.asarray(archive[key][begin:end]))
            local_bounds.append((cursor, cursor + end - begin))
            cursor += end - begin

    if not arrays["sensor_frames"]:
        raise ValueError(f"{series.run_id}: no sequence long enough for horizons")
    result = CaptureArrays(
        series.run_id, series.split,
        np.concatenate(arrays["sensor_frames"]).astype(np.float32, copy=False),
        np.concatenate(arrays["sensor_valid"]).astype(bool, copy=False),
        np.concatenate(arrays["imu_attitude_frames"]).astype(np.float32, copy=False),
        np.concatenate(arrays["imu_attitude_valid"]).astype(bool, copy=False),
        np.concatenate(arrays["simulator_rigid_state"]).astype(np.float32, copy=False),
        np.asarray(local_bounds, dtype=np.int64), series.source,
    )
    n = len(result.sensors)
    if any(len(value) != n for value in (result.sensor_valid,
                                         result.attitude, result.attitude_valid,
                                         result.rigid)):
        raise ValueError(f"{series.run_id}: joined channel lengths differ")
    if (result.sensors.shape[1] < 9 or result.attitude.shape[1] < 3
            or result.rigid.shape[1] != 13):
        raise ValueError(f"{series.run_id}: unsupported numeric schema")
    return result


def _split_safe_mixed_train_validation(splits: set[str]) -> bool:
    """Only permit a mixed archive when it contains train/validation alone."""
    return len(splits) > 1 and splits <= {"train", "validation"}


def _discover_series_with_safe_mixed_archives(
        ) -> tuple[list[atlas.RunSeries], dict[str, Any]]:
    """Add clean run-level train/validation rows from otherwise mixed archives.

    The shared atlas loader intentionally requires one split per archive.  This
    yaw-only view also admits archives that mix *only* train and validation,
    after verifying every run-level manifest row and its stored split metadata.
    Any archive manifest mentioning test/final-test is rejected before numeric
    arrays are opened.
    """
    selected, audit = atlas._discover_run_series()
    initial_selected_count = len(selected)
    known_ids = {series.run_id for series in selected}
    known_fingerprints: set[str] = set()
    for archive_rel in audit["selected_archives"]:
        manifest_path = ROOT / archive_rel
        manifest_path = manifest_path.with_name("manifest.json")
        try:
            rows = json.loads(manifest_path.read_text(encoding="utf-8")).get("runs", [])
        except (OSError, json.JSONDecodeError):
            continue
        known_fingerprints.update(str(row.get("fingerprint", ""))
                                  for row in rows if row.get("fingerprint"))

    candidates = []
    for npz_path in atlas.LIVE_RUNS.rglob("openplane_dynamics.npz"):
        manifest_path = npz_path.with_name("manifest.json")
        if not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        rows = manifest.get("runs", [])
        splits = {str(row.get("effective_split", "")) for row in rows}
        if not _split_safe_mixed_train_validation(splits):
            continue
        if not all(atlas._clean_run(row, str(row.get("effective_split", "")))
                   for row in rows):
            continue
        ids = tuple(str(row.get("run_id", "")) for row in rows)
        if not ids or not all(ids) or len(ids) != len(set(ids)):
            continue
        samples = sum(int(row.get("samples_exported", 0)) for row in rows)
        candidates.append((-samples, str(npz_path), npz_path, ids, rows))

    # Prefer the most complete clean archive; identical views are then skipped
    # by run ID/fingerprint, avoiding double weighting of repeated captures.
    candidates.sort(key=lambda item: (item[0], item[1]))
    added_archives = []
    skipped_duplicate_runs = []
    row_count_differences = []
    for _, _, npz_path, expected_ids, rows in candidates:
        new_rows = []
        for run_id, row in zip(expected_ids, rows):
            fingerprint = str(row.get("fingerprint", ""))
            if run_id in known_ids or (fingerprint and fingerprint in known_fingerprints):
                skipped_duplicate_runs.append(run_id)
                continue
            new_rows.append((run_id, row))
        if not new_rows:
            continue
        with np.load(npz_path, allow_pickle=False) as archive:
            needed = ("run_ids", "run_splits", "frames", "simulator_rigid_state",
                      "sequence_bounds", "sequence_run_index", "sensor_frames",
                      "sensor_valid", "imu_attitude_frames", "imu_attitude_valid",
                      "dt_s")
            if any(key not in archive.files for key in needed):
                continue
            archive_ids = tuple(str(value) for value in archive["run_ids"])
            archive_splits = tuple(str(value) for value in archive["run_splits"])
            if (archive_ids != expected_ids or len(archive_splits) != len(rows)
                    or archive_splits != tuple(str(row["effective_split"])
                                                for row in rows)):
                continue
            frames = np.asarray(archive["frames"], dtype=np.float32)
            rigid = np.asarray(archive["simulator_rigid_state"], dtype=np.float32)
            bounds = np.asarray(archive["sequence_bounds"], dtype=np.int64)
            sequence_run_index = np.asarray(archive["sequence_run_index"],
                                            dtype=np.int64)
            dt = np.asarray(archive["dt_s"], dtype=np.float32)
            if (frames.ndim != 2 or frames.shape[1] != 9
                    or rigid.shape != (len(frames), 13)
                    or len(dt) != len(frames)
                    or not np.allclose(dt, atlas.DT_S, rtol=0.0, atol=1.0e-7)
                    or not np.isfinite(frames).all()
                    or not np.isfinite(rigid).all()
                    or len(sequence_run_index) != len(bounds)):
                continue
            for run_id, row in new_rows:
                run_index = expected_ids.index(run_id)
                seq_ids = np.flatnonzero(sequence_run_index == run_index)
                local_frames, local_rigid, local_bounds = [], [], []
                local_attitude, local_attitude_valid = [], []
                cursor = 0
                for seq_id in seq_ids:
                    begin, end = map(int, bounds[int(seq_id)])
                    if begin < 0 or end > len(frames) or end - begin < 3:
                        continue
                    local_frames.append(frames[begin:end])
                    local_rigid.append(rigid[begin:end])
                    local_attitude.append(archive["imu_attitude_frames"][begin:end])
                    local_attitude_valid.append(archive["imu_attitude_valid"][begin:end])
                    local_bounds.append((cursor, cursor + end - begin))
                    cursor += end - begin
                if not local_frames:
                    continue
                series = atlas.RunSeries(
                    run_id, str(row["effective_split"]),
                    str(npz_path.relative_to(ROOT)),
                    np.concatenate(local_frames), np.concatenate(local_rigid),
                    np.asarray(local_bounds, dtype=np.int64),
                    np.concatenate(local_attitude),
                    np.concatenate(local_attitude_valid),
                )
                if sum(end - begin for begin, end in series.bounds) != int(
                        row.get("samples_exported", -1)):
                    archive_samples = sum(end - begin for begin, end in series.bounds)
                    manifest_samples = int(row.get("samples_exported", -1))
                    if (manifest_samples <= 0
                            or abs(archive_samples - manifest_samples)
                            / manifest_samples > 0.01):
                        continue
                    row_count_differences.append({
                        "run_id": run_id,
                        "manifest_samples": manifest_samples,
                        "archive_samples": archive_samples,
                    })
                selected.append(series)
                known_ids.add(run_id)
                fingerprint = str(row.get("fingerprint", ""))
                if fingerprint:
                    known_fingerprints.add(fingerprint)
                added_archives.append(str(npz_path.relative_to(ROOT)))

    audit["selected_runs"] = {
        split: sum(series.split == split for series in selected)
        for split in ("train", "validation")
    }
    audit["selected_archives"] = sorted({series.source for series in selected})
    audit["added_clean_train_validation_archive_runs"] = (
        len(selected) - initial_selected_count)
    audit["mixed_train_validation_archives_added"] = sorted(set(added_archives))
    audit["duplicate_mixed_archive_run_rows_skipped"] = sorted(set(skipped_duplicate_runs))
    audit["mixed_archive_row_count_differences_under_1pct"] = row_count_differences
    audit["test_and_final_test_arrays_read"] = False
    return sorted(selected, key=lambda series: (series.split, series.run_id)), audit


def _observations(capture: CaptureArrays) -> np.ndarray:
    """Build only legal current/past sensor and actuator observations.

    Dataset ``frames[:, 0:3]`` is the simulator-oracle
    ``/autodrive/roboracer_1/odom`` twist, not production odometry. It must
    never enter this model's inputs.
    """
    return np.column_stack((
        capture.sensors[:, :9],
        capture.attitude[:, (0, 2)],
    )).astype(np.float32, copy=False)


def _build_examples(capture: CaptureArrays, horizon: int,
                    history_lags: tuple[int, ...]
                    ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Build direct targets without crossing reset/sequence boundaries.

    Future commands are sampled from [k, k+horizon-1] only.  The command mask
    makes the zero-padded remainder distinguishable from a real zero command.
    No future feedback, IMU, odometry, or simulator-truth values enter X.
    """
    if horizon not in HORIZONS:
        raise ValueError(f"unsupported horizon: {horizon}")
    if not history_lags or history_lags[0] != 0:
        raise ValueError("history must include current sample first")
    observations = _observations(capture)
    commands = capture.sensors[:, 7:9].astype(np.float32, copy=False)
    features, targets, run_ids, frame_indices = [], [], [], []
    speeds, steerings, current_imu_yaw = [], [], []
    for begin_raw, end_raw in capture.bounds:
        begin, end = int(begin_raw), int(end_raw)
        first = begin + max(history_lags)
        last = end - horizon
        if last <= first:
            continue
        rows = np.arange(first, last, dtype=np.int64)
        history = np.concatenate(
            [observations[rows - lag] for lag in history_lags], axis=1)

        future = np.zeros((len(rows), FUTURE_COMMAND_STEPS, 2), dtype=np.float32)
        future[:, :horizon, :] = np.stack(
            [commands[rows + offset] for offset in range(horizon)], axis=1)
        command_mask = np.zeros((len(rows), FUTURE_COMMAND_STEPS), dtype=np.float32)
        command_mask[:, :horizon] = 1.0
        x = np.concatenate((history, future.reshape(len(rows), -1), command_mask), axis=1)
        y = capture.rigid[rows + horizon, 12].astype(np.float32, copy=False)
        valid_history = np.ones(len(rows), dtype=bool)
        for lag in history_lags:
            valid_history &= capture.sensor_valid[rows - lag]
            valid_history &= capture.attitude_valid[rows - lag]
        valid_history &= np.isfinite(x).all(axis=1) & np.isfinite(y)
        rows, x, y = rows[valid_history], x[valid_history], y[valid_history]
        if not len(rows):
            continue
        features.append(x)
        targets.append(y)
        run_ids.extend([capture.run_id] * len(rows))
        frame_indices.append(rows.copy())
        speeds.append(np.hypot(capture.rigid[rows, 7], capture.rigid[rows, 8]))
        steerings.append(capture.sensors[rows, 0].copy())
        current_imu_yaw.append(capture.sensors[rows, 6].copy())
    if not features:
        return (np.empty((0, len(history_lags) * len(OBSERVATION_NAMES)
                            + 3 * FUTURE_COMMAND_STEPS), dtype=np.float32),
                np.empty(0, dtype=np.float32),
                {"run_id": np.empty(0, dtype=object),
                 "frame_index": np.empty(0, dtype=np.int64),
                 "speed_mps": np.empty(0), "steering_rad": np.empty(0),
                 "imu_yaw_rate_rps": np.empty(0)})
    return (np.concatenate(features), np.concatenate(targets), {
        "run_id": np.asarray(run_ids, dtype=object),
        "frame_index": np.concatenate(frame_indices),
        "speed_mps": np.concatenate(speeds).astype(np.float32, copy=False),
        "steering_rad": np.concatenate(steerings).astype(np.float32, copy=False),
        "imu_yaw_rate_rps": np.concatenate(current_imu_yaw).astype(np.float32, copy=False),
    })


def _run_balanced_weights(run_ids: np.ndarray) -> np.ndarray:
    ids, counts = np.unique(np.asarray(run_ids, dtype=str), return_counts=True)
    count_by_id = dict(zip(ids.tolist(), counts.tolist()))
    weights = np.asarray([1.0 / count_by_id[str(run_id)] for run_id in run_ids],
                         dtype=np.float64)
    weights *= len(ids) / weights.sum()
    return weights


def _metrics(errors: np.ndarray) -> dict[str, float | int]:
    errors = np.asarray(errors, dtype=np.float64)
    if not len(errors):
        return {"samples": 0}
    absolute = np.abs(errors)
    return {
        "samples": int(len(errors)),
        "rmse_radps": float(np.sqrt(np.mean(errors * errors))),
        "mae_radps": float(absolute.mean()),
        "p95_abs_radps": float(np.quantile(absolute, 0.95)),
        "p99_abs_radps": float(np.quantile(absolute, 0.99)),
        "max_abs_radps": float(absolute.max()),
        "fraction_abs_error_below_0p1": float(np.mean(absolute < 0.1)),
        "fraction_abs_error_below_0p05": float(np.mean(absolute < 0.05)),
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _score(y: np.ndarray, prediction: np.ndarray, metadata: dict[str, np.ndarray]
           ) -> dict[str, Any]:
    error = np.asarray(prediction, dtype=np.float64) - y
    baseline_error = metadata["imu_yaw_rate_rps"].astype(np.float64) - y
    per_run = {}
    baseline_per_run = {}
    for run_id in sorted(set(metadata["run_id"].tolist())):
        mask = metadata["run_id"] == run_id
        per_run[run_id] = _metrics(error[mask])
        baseline_per_run[run_id] = _metrics(baseline_error[mask])
    per_run_rmse = [row["rmse_radps"] for row in per_run.values()]
    paired_run_delta = np.asarray([
        per_run[run_id]["rmse_radps"] - baseline_per_run[run_id]["rmse_radps"]
        for run_id in sorted(per_run)
    ], dtype=np.float64)
    if len(paired_run_delta):
        rng = np.random.default_rng(20261007)
        bootstrap = rng.choice(
            paired_run_delta, size=(10_000, len(paired_run_delta)), replace=True
        ).mean(axis=1)
        paired_ci = np.quantile(bootstrap, (0.025, 0.975))
    else:
        paired_ci = np.asarray((np.nan, np.nan))
    result: dict[str, Any] = {
        "pooled": _metrics(error),
        "run_macro_rmse_radps": float(np.mean(per_run_rmse)),
        "worst_run_rmse_radps": float(np.max(per_run_rmse)),
        "per_run": per_run,
        "baseline_imu_persistence": {
            "pooled": _metrics(baseline_error),
            "run_macro_rmse_radps": float(np.mean([
                row["rmse_radps"] for row in baseline_per_run.values()])),
            "per_run": baseline_per_run,
        },
        "paired_run_rmse_delta_model_minus_imu_radps": {
            "mean": float(paired_run_delta.mean()),
            "bootstrap_95pct_ci": [float(paired_ci[0]), float(paired_ci[1])],
            "runs_model_better": int(np.count_nonzero(paired_run_delta < 0.0)),
            "runs_compared": int(len(paired_run_delta)),
            "bootstrap_resamples": 10_000,
        },
    }
    bands = ((0.0, 2.0), (2.0, 4.0), (4.0, 6.0), (6.0, 8.0),
             (8.0, 10.0), (10.0, 12.0))
    result["by_speed_band"] = {}
    for low, high in bands:
        mask = ((metadata["speed_mps"] >= low)
                & (metadata["speed_mps"] < high))
        result["by_speed_band"][f"{low:.0f}-{high:.0f}mps"] = _metrics(error[mask])
    steer_bands = ((0.0, 0.10), (0.10, 0.20), (0.20, 0.35), (0.35, 0.525))
    result["by_abs_steering_band"] = {}
    for low, high in steer_bands:
        mask = ((np.abs(metadata["steering_rad"]) >= low)
                & (np.abs(metadata["steering_rad"]) < high))
        result["by_abs_steering_band"][f"{low:.3f}-{high:.3f}rad"] = _metrics(error[mask])
    return result


def fit(output_dir: Path, workers: int = -1) -> dict[str, Any]:
    run_series, source_audit = _discover_series_with_safe_mixed_archives()
    if not any(row.split == "train" for row in run_series):
        raise RuntimeError("clean whole-run training captures are required")
    minimum_sequence_length = max(HISTORY_LAGS) + max(HORIZONS) + 1
    eligible_series = [
        row for row in run_series
        if np.any((row.bounds[:, 1] - row.bounds[:, 0]) >= minimum_sequence_length)
    ]
    eligible_ids = {row.run_id for row in eligible_series}
    short_runs = sorted(row.run_id for row in run_series
                        if row.run_id not in eligible_ids)
    captures = [_read_run(row) for row in eligible_series]
    train = [row for row in captures if row.split == "train"]
    validation = [row for row in captures if row.split == "validation"]
    report: dict[str, Any] = {
        "title": "Causal direct multi-horizon yaw-rate teacher",
        "sample_period_s": atlas.DT_S,
        "horizons_steps": list(HORIZONS),
        "horizons_ms": [int(step * atlas.DT_S * 1000) for step in HORIZONS],
        "history_lags_steps": list(HISTORY_LAGS),
        "history_lags_ms": [int(step * atlas.DT_S * 1000) for step in HISTORY_LAGS],
        "input_contract": {
            "history": list(OBSERVATION_NAMES),
            "future_known_commands": ["steering_command_rad", "throttle_command_norm"],
            "future_feedback_or_truth_used": False,
            "target": "simulator_rigid_state yaw_rate at k+h",
            "runtime_status": "offline research only; no odom/MPC integration",
        },
        "source_audit": source_audit,
        "short_runs_excluded_from_horizon_fit": short_runs,
        "training_runs": sorted(row.run_id for row in train),
        "validation_runs": sorted(row.run_id for row in validation),
        "estimator": {**ESTIMATOR, "n_jobs": workers,
                      "run_balanced_sample_weight": True},
        "status": "running",
        "variants": {},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "yaw_multihorizon_teacher_report.json"
    for variant, lags in HISTORY_VARIANTS.items():
        variant_results = {}
        report["variants"][variant] = variant_results
        for horizon in HORIZONS:
            tx_parts, ty_parts, tr_parts = [], [], []
            for capture in train:
                x, y, meta = _build_examples(capture, horizon, lags)
                if len(y):
                    tx_parts.append(x)
                    ty_parts.append(y)
                    tr_parts.append(meta["run_id"])
            vx_parts, vy_parts, vm_parts = [], [], []
            for capture in validation:
                x, y, meta = _build_examples(capture, horizon, lags)
                if len(y):
                    vx_parts.append(x)
                    vy_parts.append(y)
                    vm_parts.append(meta)
            x_train, y_train = np.concatenate(tx_parts), np.concatenate(ty_parts)
            train_run_ids = np.concatenate(tr_parts)
            x_validation, y_validation = np.concatenate(vx_parts), np.concatenate(vy_parts)
            val_meta = {key: np.concatenate([part[key] for part in vm_parts])
                        for key in vm_parts[0]}
            estimator = ExtraTreesRegressor(**{**ESTIMATOR, "n_jobs": workers})
            estimator.fit(x_train, y_train,
                          sample_weight=_run_balanced_weights(train_run_ids))
            prediction = estimator.predict(x_validation)
            model_path = output_dir / f"yaw_teacher_{variant}_h{horizon:02d}.joblib"
            joblib.dump(estimator, model_path, compress=3)
            variant_results[str(horizon)] = {
                "training_samples": int(len(y_train)),
                "training_run_count": int(len(set(train_run_ids.tolist()))),
                "validation_samples": int(len(y_validation)),
                "input_feature_count": int(x_train.shape[1]),
                "model_file": model_path.name,
                "validation": _score(y_validation, prediction, val_meta),
            }
            report_path.write_text(
                json.dumps(report, indent=2, sort_keys=True,
                           default=_json_default) + "\n",
                encoding="utf-8")
            print(f"{variant} {horizon*25:4d} ms: "
                  f"RMSE={variant_results[str(horizon)]['validation']['pooled']['rmse_radps']:.4f} "
                  f"max={variant_results[str(horizon)]['validation']['pooled']['max_abs_radps']:.4f}",
                  flush=True)
            del x_train, y_train, train_run_ids, x_validation, y_validation
            del val_meta, prediction
    report["status"] = "complete"
    report["limitations"] = [
        "Whole-run validation is exploratory model comparison, not a final sealed test.",
        "Recorded future commands are used as known candidate MPC inputs; future measured actuator feedback and truth are not inputs.",
        "The target is yaw rate only; the model does not predict actuator state, wheel state, or pose.",
        "ExtraTrees is a benchmark teacher, not differentiable and not ready for MPC integration.",
        "A low average error does not satisfy the requested maximum-error bound; per-run maxima remain gating metrics.",
    ]
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True,
                                      default=_json_default) + "\n",
                           encoding="utf-8")
    print(f"report: {report_path}", flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=-1,
                        help="joblib workers for ExtraTrees; -1 uses all CPUs")
    args = parser.parse_args()
    if args.workers == 0 or args.workers < -1:
        parser.error("--workers must be -1 or a positive integer")
    fit(args.output_dir, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
