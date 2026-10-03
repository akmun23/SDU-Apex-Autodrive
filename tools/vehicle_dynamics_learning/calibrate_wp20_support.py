#!/usr/bin/env python3
"""Calibrate the two WP20 support estimators against held-out WP19 rollouts.

Training data fit only feature normalization/support banks. Validation truth is
only the recursive-error calibration target. Practice runs are transfer
diagnostics and do not set thresholds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from tools.vehicle_dynamics_learning.run_wp19_target_ablation import (
    DEFAULT_DYNAMIC,
    DEFAULT_DYNAMIC_FIXED,
    DEFAULT_DYNAMIC_PARENT,
    DEFAULT_PRACTICE,
    DEFAULT_PRACTICE_FIXED,
    DEFAULT_PRACTICE_PARENT,
    DT_S,
    HIDDEN_SIZE,
    ROLLOUT_STEPS,
    SEED,
    SUPPORT_FEATURE_NAMES,
    TASK_ROOT,
    WHEEL_RADIUS_M,
    _make_window_groups,
    _predict_batch,
    _run_balanced_normalizer,
    _support_features,
    load_capture,
    make_model,
    sha256_file,
    training_statistics,
)


ROOT = Path(__file__).resolve().parents[2]
WP19_ROOT = TASK_ROOT / "wp19_target_ablation_v2_common_encoder_mask"
WP19_REPORT = WP19_ROOT / "wp19_target_ablation_report.json"
SELECTED_NAME = "07_body_state_increment__encoder_angle_increment"
SELECTED_CHECKPOINT = WP19_ROOT / SELECTED_NAME / "model.pt"
DEFAULT_OUTPUT = TASK_ROOT / "wp20_support_calibration_v2_model_train_runs"
MAX_ROWS_PER_TRAIN_RUN = 1800
KNN_INDEPENDENT_RUNS = 3
MAHALANOBIS_NEIGHBORS_PER_RUN = 8
MAHALANOBIS_RIDGE_RELATIVE = 1e-4
ERROR_BIN_COUNT = 5
SUPPORTED_PERCENTILE = 50.0
UNSUPPORTED_PERCENTILE = 90.0
BOOTSTRAP_REPLICATES = 5000
ERROR_NAMES = (
    "normalized_body_trajectory_rmse_2s",
    "u_rear_mps",
    "v_rear_mps",
    "yaw_rate_rps",
    "wheel_rate_trajectory_rmse_2s_mps",
    "encoder_angle_increment_rmse_2s_rad",
)


def _torch_modules():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("WP20 requires the WP19 PyTorch environment") from exc
    return torch


def _rank_average(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    left = 0
    while left < len(values):
        right = left + 1
        while right < len(values) and sorted_values[right] == sorted_values[left]:
            right += 1
        ranks[order[left:right]] = 0.5 * (left + 1 + right)
        left = right
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(valid) < 3:
        return None
    rx, ry = _rank_average(x[valid]), _rank_average(y[valid])
    if np.std(rx) == 0.0 or np.std(ry) == 0.0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def _bootstrap_mean(values_by_run: dict[str, float], seed: int) -> list[float] | None:
    values = np.asarray(list(values_by_run.values()), dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, len(values),
                           size=(BOOTSTRAP_REPLICATES, len(values)))
    return np.quantile(values[indexes].mean(axis=1), (0.025, 0.975)).tolist()


def _bootstrap_spearman(rows_by_run: dict[str, dict[str, np.ndarray]],
                        score_name: str, error_name: str, seed: int
                        ) -> list[float] | None:
    run_ids = sorted(rows_by_run)
    if len(run_ids) < 3:
        return None
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(BOOTSTRAP_REPLICATES):
        draw = rng.integers(0, len(run_ids), size=len(run_ids))
        x = np.concatenate([rows_by_run[run_ids[index]][score_name]
                            for index in draw])
        y = np.concatenate([rows_by_run[run_ids[index]][error_name]
                            for index in draw])
        rho = _spearman(x, y)
        if rho is not None:
            values.append(rho)
    return (np.quantile(values, (0.025, 0.975)).astype(float).tolist()
            if values else None)


def _feature_rows(frames: np.ndarray) -> np.ndarray:
    """Build observable WP20 features at t using t and t-1 only."""
    frames = np.asarray(frames, dtype=np.float64)
    if frames.ndim != 2 or frames.shape[1] != 9 or len(frames) < 2:
        return np.empty((0, len(SUPPORT_FEATURE_NAMES)), dtype=np.float32)
    current, previous = frames[1:], frames[:-1]
    features = np.column_stack((
        np.hypot(current[:, 0], current[:, 1]),
        current[:, 3],
        (current[:, 3] - previous[:, 3]) / DT_S,
        current[:, 4],
        (current[:, 4] - previous[:, 4]) / DT_S,
        current[:, 2],
        np.abs(np.mean(current[:, 5:7], axis=1) - current[:, 0]),
    ))
    valid = np.isfinite(features).all(axis=1)
    return features[valid].astype(np.float32, copy=False)


def _training_feature_bank(captures, training_windows,
                           allowed_run_ids: set[str] | None = None
                           ) -> tuple[dict[str, np.ndarray],
                                      dict[str, dict[str, int]]]:
    by_run: dict[str, np.ndarray] = {}
    counts: dict[str, dict[str, int]] = {}
    for run_id, refs in sorted(training_windows.items()):
        if allowed_run_ids is not None and run_id not in allowed_run_ids:
            continue
        rows = []
        sequence_ids = set()
        for capture_index, sequence_index, source_row in refs:
            capture = captures[capture_index]
            rows.append(_support_features(
                capture, sequence_index, source_row))
            sequence_ids.add(int(sequence_index))
        if rows:
            by_run[run_id] = np.asarray(rows, dtype=np.float32)
            counts[run_id] = {"sequences": len(sequence_ids),
                              "available_rows": len(rows)}
    if not by_run:
        raise ValueError("WP20 found no eligible optimizer-window support rows")
    return by_run, counts


def _balanced_sample_bank(by_run: dict[str, np.ndarray]
                          ) -> dict[str, np.ndarray]:
    result = {}
    for run_id, rows in sorted(by_run.items()):
        if len(rows) > MAX_ROWS_PER_TRAIN_RUN:
            indices = np.linspace(0, len(rows) - 1, MAX_ROWS_PER_TRAIN_RUN,
                                  dtype=np.int64)
            rows = rows[indices]
        result[run_id] = rows
    return result


def _normalize_runs(bank: dict[str, np.ndarray], mean: np.ndarray,
                    scale: np.ndarray) -> dict[str, np.ndarray]:
    return {run_id: (values - mean) / scale
            for run_id, values in bank.items()}


def _support_scores(query_features: np.ndarray,
                    normalized_bank: dict[str, np.ndarray]
                    ) -> dict[str, np.ndarray]:
    query = np.asarray(query_features, dtype=np.float64)
    run_ids = sorted(normalized_bank)
    per_run_nearest = np.empty((len(query), len(run_ids)), dtype=np.float64)
    per_run_neighbors: list[list[np.ndarray]] = [[] for _ in range(len(query))]
    query_norm = np.sum(query * query, axis=1, keepdims=True)
    for run_index, run_id in enumerate(run_ids):
        points = np.asarray(normalized_bank[run_id], dtype=np.float64)
        point_norm = np.sum(points * points, axis=1)[None, :]
        squared = np.maximum(query_norm + point_norm - 2.0 * query @ points.T,
                             0.0)
        per_run_nearest[:, run_index] = np.sqrt(np.min(squared, axis=1))
        neighbor_count = min(MAHALANOBIS_NEIGHBORS_PER_RUN, len(points))
        for row_index in range(len(query)):
            nearest = np.argpartition(
                squared[row_index], neighbor_count - 1)[:neighbor_count]
            per_run_neighbors[row_index].append(points[nearest])

    run_order = np.argsort(per_run_nearest, axis=1)
    nearest_k = run_order[:, :min(KNN_INDEPENDENT_RUNS, len(run_ids))]
    knn_score = np.take_along_axis(
        per_run_nearest, nearest_k, axis=1).mean(axis=1)
    nearest_run_index = run_order[:, 0]
    nearest_run_distance = per_run_nearest[
        np.arange(len(query)), nearest_run_index]
    nearest_run_name = np.asarray(run_ids, dtype=object)[nearest_run_index]

    mahalanobis = np.empty(len(query), dtype=np.float64)
    for row_index in range(len(query)):
        local_rows = np.concatenate(per_run_neighbors[row_index], axis=0)
        center = local_rows.mean(axis=0)
        covariance = np.cov(local_rows, rowvar=False, ddof=1)
        variance_scale = float(np.trace(covariance) / covariance.shape[0])
        ridge = max(variance_scale * MAHALANOBIS_RIDGE_RELATIVE, 1e-8)
        inverse = np.linalg.pinv(covariance + ridge * np.eye(len(center)),
                                 rcond=1e-10)
        delta = query[row_index] - center
        mahalanobis[row_index] = np.sqrt(max(float(delta @ inverse @ delta), 0.0))
    return {
        "run_balanced_knn_distance": knn_score,
        "local_mahalanobis_distance": mahalanobis,
        "nearest_independent_run_distance": nearest_run_distance,
        "nearest_independent_run_id": nearest_run_name,
    }


def _evaluate_windows(model, torch, captures, windows_by_run, stats,
                      device: str) -> list[dict[str, Any]]:
    output = []
    body_scale = np.asarray(stats["physical_body"][1], dtype=np.float64)
    for run_id, refs in sorted(windows_by_run.items()):
        prediction = _predict_batch(
            model, torch, captures, refs, stats,
            "body_state_increment", "encoder_angle_increment", device)
        body_error = prediction["body"] - prediction["truth_body"]
        rate_error = prediction["wheel_rate"] - prediction["truth_wheel_rate"]
        angle_error = (prediction["wheel_angle_increment"]
                       - prediction["truth_wheel_rate"] * DT_S
                       / WHEEL_RADIUS_M)
        for index, (capture_index, sequence_index, source_row) in enumerate(refs):
            capture = captures[capture_index]
            local_error = body_error[index]
            normalized_error = local_error / body_scale
            wheel_valid = prediction["wheel_valid"][index]
            if np.any(wheel_valid):
                wheel_mask = wheel_valid[:, None]
                wheel_rmse = float(np.sqrt(np.nanmean(np.where(
                    wheel_mask, rate_error[index] ** 2, np.nan))))
                angle_rmse = float(np.sqrt(np.nanmean(np.where(
                    wheel_mask, angle_error[index] ** 2, np.nan))))
            else:
                wheel_rmse = None
                angle_rmse = None
            output.append({
                "run_id": run_id,
                "capture": capture.name,
                "sequence_index": int(sequence_index),
                "source_row_within_sequence": int(source_row),
                "support_features": _support_features(
                    capture, sequence_index, source_row).astype(float).tolist(),
                "normalized_body_trajectory_rmse_2s": float(np.sqrt(
                    np.mean(normalized_error ** 2))),
                "body_trajectory_rmse_2s_by_channel": {
                    "u_rear_mps": float(np.sqrt(np.mean(local_error[:, 0] ** 2))),
                    "v_rear_mps": float(np.sqrt(np.mean(local_error[:, 1] ** 2))),
                    "yaw_rate_rps": float(np.sqrt(np.mean(local_error[:, 2] ** 2))),
                },
                "wheel_rate_trajectory_rmse_2s_mps": wheel_rmse,
                "encoder_angle_increment_rmse_2s_rad": angle_rmse,
                "valid_encoder_steps": int(np.count_nonzero(wheel_valid)),
            })
    return output


def _error_value(row: dict[str, Any], name: str) -> float:
    if name in ("u_rear_mps", "v_rear_mps", "yaw_rate_rps"):
        return float(row["body_trajectory_rmse_2s_by_channel"][name])
    value = row[name]
    return float(value) if value is not None else float("nan")


def _per_run_error(rows: list[dict[str, Any]], mask: np.ndarray,
                   error_name: str) -> dict[str, float]:
    values: dict[str, list[float]] = {}
    for row, selected in zip(rows, mask):
        if not selected:
            continue
        value = _error_value(row, error_name)
        if np.isfinite(value):
            values.setdefault(row["run_id"], []).append(value)
    return {run_id: float(np.mean(samples))
            for run_id, samples in values.items()}


def _support_bins(rows: list[dict[str, Any]], scores: np.ndarray,
                  error_name: str, edges: np.ndarray, seed: int
                  ) -> dict[str, Any]:
    indexes = np.minimum(np.searchsorted(edges[1:-1], scores, side="right"),
                         len(edges) - 2)
    bins = []
    for index in range(len(edges) - 1):
        mask = indexes == index
        by_run = _per_run_error(rows, mask, error_name)
        bins.append({
            "support_percentile_range": [
                float(index * 100.0 / (len(edges) - 1)),
                float((index + 1) * 100.0 / (len(edges) - 1))],
            "support_score_range": [float(edges[index]), float(edges[index + 1])],
            "rows": int(np.count_nonzero(mask)),
            "independent_runs": len(by_run),
            "macro_run_mean_error": (float(np.mean(list(by_run.values())))
                                     if by_run else None),
            "run_cluster_bootstrap_95pct_ci": _bootstrap_mean(
                by_run, seed + index),
            "per_run_error": by_run,
        })
    return {"edges": edges.astype(float).tolist(), "bins": bins}


def _category_errors(rows: list[dict[str, Any]], scores: np.ndarray,
                     error_names: tuple[str, ...],
                     supported_edge: float, unsupported_edge: float
                     ) -> dict[str, Any]:
    categories = {
        "supported": scores <= supported_edge,
        "weak_support": (scores > supported_edge)
        & (scores <= unsupported_edge),
        "unsupported": scores > unsupported_edge,
    }
    output = {}
    for category, mask in categories.items():
        output[category] = {
            "rows": int(np.count_nonzero(mask)),
            "independent_runs": len({row["run_id"] for row, selected
                                     in zip(rows, mask) if selected}),
            "errors": {},
        }
        for error_name in error_names:
            by_run = _per_run_error(rows, mask, error_name)
            output[category]["errors"][error_name] = {
                "macro_run_mean": (float(np.mean(list(by_run.values())))
                                   if by_run else None),
                "run_cluster_bootstrap_95pct_ci": _bootstrap_mean(
                    by_run, len(error_name) + len(category)),
                "per_run": by_run,
            }
    return output


def _calibrate_estimator(rows: list[dict[str, Any]], score_key: str,
                         score_label: str, validation_run_ids: list[str],
                         seed: int) -> dict[str, Any]:
    scores = np.asarray([row["support_scores"][score_key] for row in rows],
                        dtype=np.float64)
    edges = np.quantile(scores, np.linspace(0.0, 1.0, ERROR_BIN_COUNT + 1))
    percentiles = {
        str(percentile): float(np.percentile(scores, percentile))
        for percentile in (SUPPORTED_PERCENTILE, 80.0, UNSUPPORTED_PERCENTILE)}
    score_by_run = {
        run_id: np.asarray([row["support_scores"][score_key] for row in rows
                            if row["run_id"] == run_id], dtype=np.float64)
        for run_id in validation_run_ids}
    result: dict[str, Any] = {
        "support_score_name": score_label,
        "low_score_means": "stronger support",
        "validation_score_percentiles": percentiles,
        "metrics": {},
        "validation_error_by_support_percentile": {},
        "candidate_thresholds_by_validation_percentile": {},
        "supported_weak_unsupported_candidate_thresholds": None,
    }
    for metric_index, error_name in enumerate(ERROR_NAMES):
        errors = np.asarray([_error_value(row, error_name) for row in rows],
                            dtype=np.float64)
        pooled_rho = _spearman(scores, errors)
        per_run_rho: dict[str, float] = {}
        for run_id in validation_run_ids:
            selected = [row for row in rows if row["run_id"] == run_id]
            run_scores = np.asarray([
                row["support_scores"][score_key] for row in selected],
                dtype=np.float64)
            run_errors = np.asarray([_error_value(row, error_name)
                                     for row in selected], dtype=np.float64)
            rho = _spearman(run_scores, run_errors)
            if rho is not None:
                per_run_rho[run_id] = rho

        rows_by_run = {
            run_id: {
                "score": np.asarray([
                    row["support_scores"][score_key] for row in rows
                    if row["run_id"] == run_id], dtype=np.float64),
                "error": np.asarray([
                    _error_value(row, error_name) for row in rows
                    if row["run_id"] == run_id], dtype=np.float64),
            } for run_id in validation_run_ids
        }
        valid = np.isfinite(scores) & np.isfinite(errors)
        rho_ci = _bootstrap_spearman(
            rows_by_run, "score", "error", seed + metric_index)
        bins = _support_bins(rows, scores, error_name, edges,
                             seed + metric_index * 11)
        high_support = bins["bins"][0]["per_run_error"]
        low_support = bins["bins"][-1]["per_run_error"]
        shared_runs = sorted(set(high_support) & set(low_support))
        low_minus_high = {
            run_id: low_support[run_id] - high_support[run_id]
            for run_id in shared_runs}
        low_minus_high_ci = _bootstrap_mean(
            low_minus_high, seed + metric_index * 19 + 3)
        macro_run = _per_run_error(rows, valid, error_name)
        result["metrics"][error_name] = {
            "pooled_window_spearman_rho": pooled_rho,
            "run_cluster_bootstrap_95pct_ci": rho_ci,
            "per_run_spearman_rho": per_run_rho,
            "runs_with_positive_within_run_correlation": sum(
                value > 0.0 for value in per_run_rho.values()),
            "independent_validation_runs_with_estimable_rho": len(per_run_rho),
            "macro_run_error_mean": (float(np.mean(list(macro_run.values())))
                                     if macro_run else None),
            "macro_run_error_by_run": macro_run,
            "highest_distance_minus_lowest_distance_quintile_error": {
                "macro_run_mean_difference": (
                    float(np.mean(list(low_minus_high.values())))
                    if low_minus_high else None),
                "run_cluster_bootstrap_95pct_ci": low_minus_high_ci,
                "paired_independent_runs": len(low_minus_high),
            },
        }
        result["validation_error_by_support_percentile"][error_name] = bins

    primary = result["metrics"]["normalized_body_trajectory_rmse_2s"]
    rho_ci = primary["run_cluster_bootstrap_95pct_ci"]
    delta_ci = primary[
        "highest_distance_minus_lowest_distance_quintile_error"][
            "run_cluster_bootstrap_95pct_ci"]
    positive_runs = primary["runs_with_positive_within_run_correlation"]
    passed = bool(rho_ci and rho_ci[0] > 0.0
                  and delta_ci and delta_ci[0] > 0.0
                  and positive_runs >= 4)
    result["calibration_gate"] = {
        "predeclared_rule": (
            "Pass only if normalized-body-error Spearman's run-bootstrap 95% CI "
            "is entirely above zero, the highest-distance minus lowest-distance "
            "quintile error-difference 95% CI is entirely above zero, and "
            "within-run correlation is positive on at least 4 of 6 validation runs."),
        "primary_rho_ci_entirely_positive": bool(rho_ci and rho_ci[0] > 0.0),
        "primary_low_support_error_difference_ci_entirely_positive": bool(
            delta_ci and delta_ci[0] > 0.0),
        "primary_positive_run_count_at_least_4_of_6": positive_runs >= 4,
        "positive_runs": positive_runs,
        "required_positive_runs": 4,
        "status": "calibrated" if passed else "not_calibrated",
    }
    result["candidate_thresholds_by_validation_percentile"] = {
        "supported_upper_score": percentiles[str(SUPPORTED_PERCENTILE)],
        "weak_support_upper_score": percentiles[str(UNSUPPORTED_PERCENTILE)],
        "source": "validation support-score p50 and p90",
        "categories": _category_errors(
            rows, scores, ERROR_NAMES,
            percentiles[str(SUPPORTED_PERCENTILE)],
            percentiles[str(UNSUPPORTED_PERCENTILE)]),
        "confidence_authorized": passed,
    }
    if passed:
        result["supported_weak_unsupported_candidate_thresholds"] = {
            "supported_if_score_at_or_below": percentiles[
                str(SUPPORTED_PERCENTILE)],
            "weak_support_if_score_above_p50_and_at_or_below_p90": percentiles[
                str(UNSUPPORTED_PERCENTILE)],
            "unsupported_if_score_above": percentiles[
                str(UNSUPPORTED_PERCENTILE)],
            "threshold_source": "validation score p50/p90 after error calibration gate",
            "research_only_until_WP24_WP25": True,
        }
    return result


def _training_metrics(bank: dict[str, np.ndarray],
                      counts: dict[str, dict[str, int]]) -> dict[str, Any]:
    return {
        "observable_feature_rows_available": int(sum(
            item["available_rows"] for item in counts.values())),
        "support_bank_rows": int(sum(map(len, bank.values()))),
        "sequence_count": int(sum(item["sequences"] for item in counts.values())),
        "independent_run_count": len(bank),
        "per_run": {
            run_id: {**counts[run_id], "support_bank_rows": len(bank[run_id])}
            for run_id in sorted(bank)},
    }


def _score_metadata(capture_rows: list[dict[str, Any]],
                    split_windows: dict[str, list[tuple[int, int, int]]]
                    ) -> dict[str, Any]:
    return {
        "rows": len(capture_rows),
        "sequence_count": len({(row["run_id"], row["sequence_index"])
                               for row in capture_rows}),
        "independent_run_count": len(split_windows),
        "windows_per_run": {
            run_id: sum(row["run_id"] == run_id for row in capture_rows)
            for run_id in sorted(split_windows)},
    }


def _practice_transfer(rows: list[dict[str, Any]], windows_by_run,
                       score_names: tuple[str, ...]) -> dict[str, Any]:
    output = {}
    for score_name in score_names:
        by_run = {}
        for run_id in sorted(windows_by_run):
            run_rows = [row for row in rows if row["run_id"] == run_id]
            by_run[run_id] = {
                "rows": len(run_rows),
                "support_score_median": float(np.median([
                    row["support_scores"][score_name] for row in run_rows])),
                "normalized_body_error_mean": float(np.mean([
                    row["normalized_body_trajectory_rmse_2s"]
                    for row in run_rows])),
                "u_error_mean_mps": float(np.mean([
                    row["body_trajectory_rmse_2s_by_channel"]["u_rear_mps"]
                    for row in run_rows])),
                "v_error_mean_mps": float(np.mean([
                    row["body_trajectory_rmse_2s_by_channel"]["v_rear_mps"]
                    for row in run_rows])),
                "yaw_rate_error_mean_rps": float(np.mean([
                    row["body_trajectory_rmse_2s_by_channel"]["yaw_rate_rps"]
                    for row in run_rows])),
            }
        scores = [row["support_scores"][score_name] for row in rows]
        output[score_name] = {
            "independent_runs": len(by_run),
            "rows": len(rows),
            "support_score_percentiles": {
                str(percentile): float(np.percentile(scores, percentile))
                for percentile in (10, 50, 90)},
            "per_run": by_run,
            "thresholds_fitted_on_practice": False,
        }
    return output


def run_wp20(dynamic: Path, dynamic_fixed: Path, dynamic_parent: Path,
              practice: Path, practice_fixed: Path, practice_parent: Path,
              checkpoint_path: Path, source_report_path: Path,
              output: Path, device: str = "auto") -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite WP20 result directory: {output}")
    source_report = json.loads(source_report_path.read_text(encoding="utf-8"))
    if source_report.get("experimental_status") == "superseded_not_for_selection":
        raise ValueError("WP20 refuses the superseded first WP19 report")
    source_candidate = next((item for item in source_report["candidates"]
                             if item["candidate"] == SELECTED_NAME), None)
    if source_candidate is None:
        raise ValueError("WP19 report lacks the selected B-B/W-D candidate")
    checkpoint_hash = sha256_file(checkpoint_path)
    if checkpoint_hash != source_candidate["checkpoint_sha256"]:
        raise ValueError("selected WP19 checkpoint hash differs from its report")
    for path, field in (
            (dynamic, "dynamic_sha256"),
            (dynamic_fixed, "dynamic_fixed_sha256"),
            (practice, "practice_sha256"),
            (practice_fixed, "practice_fixed_sha256")):
        if sha256_file(path) != source_report["data"][field]:
            raise ValueError(f"WP20 input changed since WP19: {path}")

    captures = [
        load_capture("openplane", dynamic, dynamic_fixed, dynamic_parent,
                     {"train", "validation"}),
        load_capture("practice", practice, practice_fixed, practice_parent,
                     {"unseen_practice"}),
    ]
    train_windows, validation_windows, practice_windows = _make_window_groups(captures)
    torch = _torch_modules()
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device == "cpu":
        torch.set_num_threads(1)
    checkpoint = torch.load(checkpoint_path, map_location=device,
                             weights_only=True)
    metadata = checkpoint.get("metadata", {})
    if (metadata.get("body_target") != "body_state_increment"
            or metadata.get("wheel_target") != "encoder_angle_increment"
            or metadata.get("seed") != SEED
            or metadata.get("history_steps") != 80
            or metadata.get("rollout_steps") != ROLLOUT_STEPS
            or metadata.get("hidden_size") != HIDDEN_SIZE):
        raise ValueError("selected WP19 checkpoint metadata is not B-B/W-D")
    model_train_runs = set(metadata.get("training_runs", []))
    if model_train_runs != set(train_windows):
        raise ValueError("WP19 checkpoint and eligible optimizer windows use different train runs")

    raw_training_features, training_counts = _training_feature_bank(
        captures, train_windows, allowed_run_ids=model_train_runs)
    if set(raw_training_features) != model_train_runs:
        missing_runs = sorted(model_train_runs - set(raw_training_features))
        extra_runs = sorted(set(raw_training_features) - model_train_runs)
        raise ValueError(f"WP20 support bank/model training-run mismatch; missing={missing_runs}, extra={extra_runs}")
    feature_mean, feature_scale = _run_balanced_normalizer(
        raw_training_features, minimum=1e-3)
    raw_bank = _balanced_sample_bank(raw_training_features)
    normalized_bank = _normalize_runs(raw_bank, feature_mean, feature_scale)
    model_stats = training_statistics(captures, train_windows)

    model = make_model(torch.nn).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()

    validation_rows = []
    for rows in [_evaluate_windows(
            model, torch, captures, validation_windows, model_stats, device),
            _evaluate_windows(
                model, torch, captures, practice_windows, model_stats, device)]:
        features = np.asarray([row["support_features"] for row in rows],
                              dtype=np.float64)
        normalized = (features - feature_mean) / feature_scale
        scores = _support_scores(normalized, normalized_bank)
        for index, row in enumerate(rows):
            row["support_scores"] = {
                "run_balanced_knn_distance": float(
                    scores["run_balanced_knn_distance"][index]),
                "local_mahalanobis_distance": float(
                    scores["local_mahalanobis_distance"][index]),
                "nearest_independent_run_distance": float(
                    scores["nearest_independent_run_distance"][index]),
                "nearest_independent_run_id": str(
                    scores["nearest_independent_run_id"][index]),
            }
        if rows and rows[0]["capture"] == "openplane":
            validation_rows = rows
        else:
            practice_rows = rows

    validation_run_ids = sorted(validation_windows)
    estimators = {
        "run_balanced_knn_distance": _calibrate_estimator(
            validation_rows, "run_balanced_knn_distance", "run-balanced kNN",
            validation_run_ids, SEED + 20),
        "local_mahalanobis_distance": _calibrate_estimator(
            validation_rows, "local_mahalanobis_distance",
            "local Mahalanobis", validation_run_ids, SEED + 21),
    }
    passing = [name for name, result in estimators.items()
               if result["calibration_gate"]["status"] == "calibrated"]
    if len(passing) == 2:
        selected_support = max(
            passing,
            key=lambda name: estimators[name]["metrics"][
                "normalized_body_trajectory_rmse_2s"][
                    "run_cluster_bootstrap_95pct_ci"][0])
        gate_status = "both_calibrated_one_frozen"
    elif len(passing) == 1:
        selected_support = passing[0]
        gate_status = "one_calibrated"
    else:
        selected_support = None
        gate_status = "support_not_calibrated"

    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 1,
        "work_package": "WP20",
        "purpose": "calibrate the two prescribed observable-feature support estimators against selected WP19 recursive errors",
        "support_definition_frozen_before_validation_scoring": True,
        "plant_promoted": False,
        "support_confidence_authorized": selected_support is not None,
        "future_truth_used_in_support_features": False,
        "test_or_final_test_opened": False,
        "selected_wp19_candidate": SELECTED_NAME,
        "selected_wp19_checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "selected_wp19_checkpoint_sha256": checkpoint_hash,
        "wp19_report_sha256": sha256_file(source_report_path),
        "fixed_configuration": {
            "device": device,
            "support_features": list(SUPPORT_FEATURE_NAMES),
            "feature_definition": (
                "speed=hypot(odom_u,odom_v); signed steering feedback; "
                "one-step backward steering feedback rate; throttle feedback; "
                "one-step backward throttle-feedback slew; yaw rate; "
                "abs(mean(filtered rear-wheel rates)-bridge rear-axle u)"),
            "normalization": "training-only equal-run mean and standard deviation",
            "maximum_support_rows_per_training_run": MAX_ROWS_PER_TRAIN_RUN,
            "knn_definition": (
                f"mean nearest distance to each of the {KNN_INDEPENDENT_RUNS} "
                "closest independent training runs"),
            "local_mahalanobis_definition": (
                f"local covariance from up to {MAHALANOBIS_NEIGHBORS_PER_RUN} "
                "nearest samples per independent training run; diagonal ridge "
                f"{MAHALANOBIS_RIDGE_RELATIVE:g} times mean variance"),
            "error_bins": ERROR_BIN_COUNT,
            "validation_starts_per_run": 64,
            "bootstrap_unit": "independent whole validation run",
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "candidate_threshold_percentiles": {
                "supported_upper": SUPPORTED_PERCENTILE,
                "weak_support_upper": UNSUPPORTED_PERCENTILE},
        },
        "data": {
            "training": _training_metrics(raw_bank, training_counts),
            "validation": _score_metadata(validation_rows, validation_windows),
            "practice_transfer": _score_metadata(practice_rows, practice_windows),
        },
        "normalization_train_only": {
            "mean": feature_mean.astype(float).tolist(),
            "scale": feature_scale.astype(float).tolist(),
            "training_run_ids": sorted(raw_bank),
        },
        "estimators": estimators,
        "gate_decision": {
            "status": gate_status,
            "selected_support_definition": selected_support,
            "confidence_authorized_only_for_selected_estimator": (
                selected_support is not None),
            "thresholds_frozen_for_wp21": selected_support is not None,
            "selection_rule_if_both_pass": (
                "freeze the estimator with the larger lower 95% run-bootstrap "
                "bound on primary normalized-body-error Spearman correlation"),
            "decision_note": (
                "If neither estimator passes, do not expose support as confidence "
                "and do not train a support-regularized replacement plant."),
        },
        "practice_transfer_diagnostics": _practice_transfer(
            practice_rows, practice_windows,
            ("run_balanced_knn_distance", "local_mahalanobis_distance")),
        "validation_window_records": validation_rows,
        "practice_window_records": practice_rows,
    }
    report_path = output / "wp20_support_calibration_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dynamic", type=Path, default=DEFAULT_DYNAMIC)
    parser.add_argument("--dynamic-fixed", type=Path, default=DEFAULT_DYNAMIC_FIXED)
    parser.add_argument("--dynamic-parent", type=Path, default=DEFAULT_DYNAMIC_PARENT)
    parser.add_argument("--practice", type=Path, default=DEFAULT_PRACTICE)
    parser.add_argument("--practice-fixed", type=Path, default=DEFAULT_PRACTICE_FIXED)
    parser.add_argument("--practice-parent", type=Path, default=DEFAULT_PRACTICE_PARENT)
    parser.add_argument("--checkpoint", type=Path, default=SELECTED_CHECKPOINT)
    parser.add_argument("--wp19-report", type=Path, default=WP19_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    report = run_wp20(
        args.dynamic.resolve(), args.dynamic_fixed.resolve(),
        args.dynamic_parent.resolve(), args.practice.resolve(),
        args.practice_fixed.resolve(), args.practice_parent.resolve(),
        args.checkpoint.resolve(), args.wp19_report.resolve(),
        args.output.resolve(), args.device)
    print(json.dumps({
        "report": str((args.output / "wp20_support_calibration_report.json").resolve()),
        "training_runs": report["data"]["training"]["independent_run_count"],
        "validation_runs": report["data"]["validation"]["independent_run_count"],
        "validation_windows": report["data"]["validation"]["rows"],
        "gate_status": report["gate_decision"]["status"],
        "estimators": {key: value["calibration_gate"]["status"]
                       for key, value in report["estimators"].items()},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
