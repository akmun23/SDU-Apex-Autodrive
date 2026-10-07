#!/usr/bin/env python3
"""Fit and independently validate a support-limited throttle response lookup.

The target is the measured step-minus-ramp response, not absolute tire force or
an absolute plant model. Only explicitly designated training captures fit the
lookup; validation captures are evaluated separately and never enter the fit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


METRICS = {
    "wheel_residual_mps": (
        "step_minus_ramp_mean_abs_rear_wheel_residual_mps_change_0.10_1.00s"),
    "wheel_slip_ratio_proxy": (
        "step_minus_ramp_mean_abs_rear_wheel_longitudinal_slip_ratio_proxy_change_0.10_1.00s"),
    "longitudinal_accel_mps2": (
        "step_minus_ramp_rigid_body_longitudinal_acceleration_mps2_change_0.10_1.00s"),
    "abs_lateral_accel_mps2": (
        "step_minus_ramp_abs_rigid_body_lateral_acceleration_mps2_change_0.10_1.00s"),
    "abs_roll_rad": (
        "step_minus_ramp_abs_imu_roll_rad_change_0.10_1.00s"),
    "abs_roll_rate_rps": (
        "step_minus_ramp_abs_imu_roll_rate_rps_change_0.10_1.00s"),
}
WINDOWED_METRICS = {
    "wheel_residual_early_mps": (
        "step_minus_ramp_mean_abs_rear_wheel_residual_mps_change_0.10_0.35s"),
    "wheel_residual_middle_mps": (
        "step_minus_ramp_mean_abs_rear_wheel_residual_mps_change_0.35_0.65s"),
    "wheel_residual_late_mps": (
        "step_minus_ramp_mean_abs_rear_wheel_residual_mps_change_0.65_1.00s"),
    "wheel_slip_ratio_early": (
        "step_minus_ramp_mean_abs_rear_wheel_longitudinal_slip_ratio_proxy_change_0.10_0.35s"),
    "wheel_slip_ratio_middle": (
        "step_minus_ramp_mean_abs_rear_wheel_longitudinal_slip_ratio_proxy_change_0.35_0.65s"),
    "wheel_slip_ratio_late": (
        "step_minus_ramp_mean_abs_rear_wheel_longitudinal_slip_ratio_proxy_change_0.65_1.00s"),
    "longitudinal_accel_early_mps2": (
        "step_minus_ramp_rigid_body_longitudinal_acceleration_mps2_change_0.10_0.35s"),
    "longitudinal_accel_middle_mps2": (
        "step_minus_ramp_rigid_body_longitudinal_acceleration_mps2_change_0.35_0.65s"),
    "longitudinal_accel_late_mps2": (
        "step_minus_ramp_rigid_body_longitudinal_acceleration_mps2_change_0.65_1.00s"),
}
ALL_METRICS = METRICS | WINDOWED_METRICS
DECISION_METRICS = (
    "wheel_residual_mps", "wheel_slip_ratio_proxy",
    "longitudinal_accel_mps2",
)
FEATURE_FIELDS = (
    "speed_target_mps", "throttle_delta_direction", "throttle_delta_norm",
    "throttle_rise_rate_norm_per_sec", "abs_steering_command_rad",
    "turn_direction",
)
REPO_ROOT = Path(__file__).resolve().parents[2]
BOOTSTRAP_SEED = 20261007


def _normalise_feature(value: Any, field: str) -> str:
    if field in ("turn_direction", "throttle_delta_direction"):
        return str(value)
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite feature {field}: {value!r}")
    return f"{number:.3f}"


def _cell_key(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(_normalise_feature(row[field], field) for field in FEATURE_FIELDS)


def _typed_features(key: tuple[str, ...]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for field, value in zip(FEATURE_FIELDS, key):
        values[field] = (value if field in ("turn_direction", "throttle_delta_direction")
                         else float(value))
    return values


def _read_split_rows(paths: list[Path], required_split: str
                     ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    seen_run_ids: set[str] = set()
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        source_rows = payload.get("paired_results", [])
        source_splits = {str(row.get("split", "unknown")) for row in source_rows}
        sealed = source_splits & {"test", "final_test"}
        if sealed:
            raise ValueError(f"refusing sealed split rows in {path}: {sorted(sealed)}")
        accepted_splits = source_splits & {"train", "validation"}
        unexpected = source_splits - {"train", "validation"}
        if unexpected:
            raise ValueError(f"unsupported split labels in {path}: {sorted(unexpected)}")
        selected = [row for row in source_rows
                    if row.get("split") == required_split]
        run_ids = {str(row["run_id"]) for row in selected if row.get("run_id")}
        run_reports = {str(report["run_id"]): report
                       for report in payload.get("runs", [])
                       if report.get("run_id")}
        missing_reports = run_ids - set(run_reports)
        if missing_reports:
            raise ValueError(f"missing capture quality reports in {path}: "
                             f"{sorted(missing_reports)}")
        for run_id in sorted(run_ids):
            report = run_reports[run_id]
            collision_counts = report.get("collision_count_start_end", [])
            encoder_matches = report.get("encoder_source_stamp_match_fraction", {})
            hard_gate_failures = []
            if report.get("aborted") is not False:
                hard_gate_failures.append("experiment_aborted")
            if not collision_counts or any(int(count) != 0
                                           for count in collision_counts):
                hard_gate_failures.append("collision_count_nonzero_or_missing")
            if int(report.get("bridge_timing_faults", -1)) != 0:
                hard_gate_failures.append("bridge_timing_faults_nonzero_or_missing")
            if report.get("reset_recovery_pass") is not True:
                hard_gate_failures.append("reset_recovery_gate_failed")
            if report.get("streams_meet_40hz_receive_gate") is not True:
                hard_gate_failures.append("40hz_stream_gate_failed")
            if (not encoder_matches
                    or any(float(value) < 0.999 for value in
                           encoder_matches.values())):
                hard_gate_failures.append("encoder_packet_join_incomplete")
            if hard_gate_failures:
                raise ValueError(f"capture {run_id} failed required data gates: "
                                 f"{hard_gate_failures}")
        duplicate_ids = seen_run_ids & run_ids
        if duplicate_ids:
            raise ValueError(f"duplicate capture IDs in {required_split} inputs: "
                             f"{sorted(duplicate_ids)}")
        seen_run_ids.update(run_ids)
        sources.append({
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "available_splits": sorted(accepted_splits),
            "selected_split": required_split,
            "selected_run_ids": sorted(run_ids),
        })
        for raw in selected:
            if not raw.get("valid"):
                continue
            if (int(raw.get("ramp_speed_governor_ticks", 0)) != 0
                    or int(raw.get("step_speed_governor_ticks", 0)) != 0):
                continue
            if any(field not in raw or raw[field] is None
                   for field in (*FEATURE_FIELDS, *METRICS.values(), "run_id")):
                continue
            try:
                numeric = [float(raw[field]) for field in
                           (*[field for field in FEATURE_FIELDS
                              if field not in ("turn_direction",
                                               "throttle_delta_direction")],
                            *METRICS.values())]
            except (TypeError, ValueError):
                continue
            if not all(math.isfinite(value) for value in numeric):
                continue
            rows.append(raw)
    if not rows:
        raise ValueError(f"no valid, governor-clean {required_split} response rows")
    return rows, sources


def _run_cell_means(rows: list[dict[str, Any]]) -> dict[
        str, dict[tuple[str, ...], dict[str, float]]]:
    grouped: dict[tuple[str, tuple[str, ...], str], list[float]] = defaultdict(list)
    for row in rows:
        run_id = str(row["run_id"])
        key = _cell_key(row)
        for metric, field in ALL_METRICS.items():
            value = row.get(field)
            if value is None:
                continue
            value = float(value)
            if math.isfinite(value):
                grouped[(run_id, key, metric)].append(value)
    by_run: dict[str, dict[tuple[str, ...], dict[str, float]]] = defaultdict(dict)
    for (run_id, key, metric), values in grouped.items():
        by_run[run_id].setdefault(key, {})[metric] = float(np.mean(values))
    return dict(by_run)


def _mean_by_key(run_cells: dict[str, dict[tuple[str, ...], dict[str, float]]],
                 run_ids: set[str], key_fn) -> dict[
                     tuple[str, ...], dict[str, float]]:
    grouped: dict[tuple[str, ...], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list))
    for run_id in sorted(run_ids):
        for cell, metrics in run_cells[run_id].items():
            for metric, value in metrics.items():
                grouped[key_fn(cell)][metric].append(value)
    return {key: {metric: float(np.mean(values))
                  for metric, values in metrics.items()}
            for key, metrics in grouped.items()}


def _rmse(errors: list[float]) -> float | None:
    return math.sqrt(float(np.mean(np.square(errors)))) if errors else None


def _mae(errors: list[float]) -> float | None:
    return float(np.mean(np.abs(errors))) if errors else None


def _run_metrics(actual: dict[tuple[str, ...], dict[str, float]],
                 exact: dict[tuple[str, ...], dict[str, float]],
                 coarse: dict[tuple[str, ...], dict[str, float]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for metric in ALL_METRICS:
        available = [(cell, values[metric]) for cell, values in actual.items()
                     if metric in values]
        exact_pairs = [(value, exact[cell][metric]) for cell, value in available
                       if cell in exact and metric in exact[cell]]
        coarse_pairs = [(value, coarse.get(cell[:3], {}).get(metric))
                        for cell, value in available]
        coarse_pairs = [(actual_value, prediction)
                        for actual_value, prediction in coarse_pairs
                        if prediction is not None]
        exact_errors = [prediction - value for value, prediction in exact_pairs]
        coarse_errors = [prediction - value for value, prediction in coarse_pairs]
        output[metric] = {
            "available_cells": len(available),
            "exact_supported_cells": len(exact_pairs),
            "exact_support_fraction": (
                len(exact_pairs) / len(available) if available else 0.0),
            "exact_cell_rmse": _rmse(exact_errors),
            "exact_cell_mae": _mae(exact_errors),
            "coarse_supported_cells": len(coarse_pairs),
            "coarse_rmse_all_supported": _rmse(coarse_errors),
            "coarse_mae_all_supported": _mae(coarse_errors),
        }
        common = [(cell, actual_value) for cell, actual_value in available
                  if cell in exact and metric in exact[cell]
                  and metric in coarse.get(cell[:3], {})]
        output[metric]["common_support_cells"] = len(common)
        output[metric]["exact_cell_rmse_common_support"] = _rmse([
            exact[cell][metric] - value for cell, value in common])
        output[metric]["coarse_rmse_common_support"] = _rmse([
            coarse[cell[:3]][metric] - value for cell, value in common])
    return output


def _bootstrap_run_gain(gains: list[float]) -> list[float] | None:
    if len(gains) < 2:
        return None
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.choice(np.asarray(gains), size=(20_000, len(gains)), replace=True)
    return [float(value) for value in np.quantile(draws.mean(axis=1), (0.025, 0.975))]


def _evaluate_runs(run_cells: dict[str, dict[tuple[str, ...], dict[str, float]]],
                   training_ids: set[str], evaluation_ids: set[str]
                   ) -> dict[str, Any]:
    exact = _mean_by_key(run_cells, training_ids, lambda cell: cell)
    coarse = _mean_by_key(run_cells, training_ids, lambda cell: cell[:3])
    per_run: dict[str, Any] = {}
    metric_gains: dict[str, list[float]] = {metric: [] for metric in ALL_METRICS}
    for run_id in sorted(evaluation_ids):
        metrics = _run_metrics(run_cells[run_id], exact, coarse)
        per_run[run_id] = metrics
        for metric, result in metrics.items():
            exact_error = result["exact_cell_rmse_common_support"]
            coarse_error = result["coarse_rmse_common_support"]
            if exact_error is not None and coarse_error is not None:
                metric_gains[metric].append(coarse_error - exact_error)
    summary: dict[str, Any] = {}
    for metric in ALL_METRICS:
        results = [per_run[run][metric] for run in sorted(evaluation_ids)]
        exact_values = [row["exact_cell_rmse_common_support"] for row in results
                        if row["exact_cell_rmse_common_support"] is not None]
        coarse_values = [row["coarse_rmse_common_support"] for row in results
                         if row["coarse_rmse_common_support"] is not None]
        gains = metric_gains[metric]
        summary[metric] = {
            "evaluated_run_count": len(gains),
            "macro_run_exact_rmse_common_support": (
                float(np.mean(exact_values)) if exact_values else None),
            "macro_run_coarse_rmse_common_support": (
                float(np.mean(coarse_values)) if coarse_values else None),
            "per_run_rmse_gain_coarse_minus_exact": gains,
            "run_cluster_bootstrap_95pct_ci_gain": _bootstrap_run_gain(gains),
        }
    return {"per_run": per_run, "summary": summary}


def _training_leave_one_run_out(run_cells: dict[
        str, dict[tuple[str, ...], dict[str, float]]], train_ids: set[str]
        ) -> dict[str, Any]:
    folds = {}
    for held_out in sorted(train_ids):
        fit_ids = train_ids - {held_out}
        if not fit_ids:
            continue
        exact = _mean_by_key(run_cells, fit_ids, lambda cell: cell)
        coarse = _mean_by_key(run_cells, fit_ids, lambda cell: cell[:3])
        folds[held_out] = _run_metrics(run_cells[held_out], exact, coarse)
    return {"method": "leave-one-training-capture-out",
            "fold_count": len(folds), "folds": folds}


def _fit_cells(run_cells: dict[str, dict[tuple[str, ...], dict[str, float]]],
               train_ids: set[str]) -> list[dict[str, Any]]:
    means = _mean_by_key(run_cells, train_ids, lambda cell: cell)
    output = []
    for key, effects in sorted(means.items()):
        per_run = {
            metric: [run_cells[run][key][metric] for run in sorted(train_ids)
                     if key in run_cells[run] and metric in run_cells[run][key]]
            for metric in ALL_METRICS
        }
        paired_run_count = min(len(per_run[metric])
                               for metric in DECISION_METRICS)
        if paired_run_count < 2:
            throttle_action_evidence = "insufficient_independent_runs"
        elif (all(value > 0.0 for value in per_run["wheel_residual_mps"])
              and all(value > 0.0 for value in
                      per_run["wheel_slip_ratio_proxy"])
              and all(value < 0.0 for value in
                      per_run["longitudinal_accel_mps2"])):
            throttle_action_evidence = "ramp_dominates_step"
        elif (all(value < 0.0 for value in per_run["wheel_residual_mps"])
              and all(value < 0.0 for value in
                      per_run["wheel_slip_ratio_proxy"])
              and all(value > 0.0 for value in
                      per_run["longitudinal_accel_mps2"])):
            throttle_action_evidence = "step_dominates_ramp"
        else:
            throttle_action_evidence = "tradeoff_or_run_inconsistent"
        output.append({
            **_typed_features(key),
            "independent_training_run_count": paired_run_count,
            "throttle_action_evidence": throttle_action_evidence,
            "effects": {
                metric: {
                    "step_minus_ramp_mean": effects[metric],
                    "training_run_min": min(values),
                    "training_run_max": max(values),
                    "training_run_sd": (
                        float(np.std(values, ddof=1)) if len(values) > 1 else None),
                    "training_run_count": len(values),
                }
                for metric, values in per_run.items() if values
            },
        })
    return output


def _validate_action_evidence(
        validation_cells: dict[str, dict[tuple[str, ...], dict[str, float]]],
        fitted_cells: list[dict[str, Any]]) -> dict[str, Any]:
    decisions = {
        tuple(_normalise_feature(cell[field], field) for field in FEATURE_FIELDS):
        cell["throttle_action_evidence"]
        for cell in fitted_cells
    }
    per_run: dict[str, dict[str, int]] = {}
    per_cell: dict[tuple[str, ...], dict[str, Any]] = {}
    for run_id, cells in validation_cells.items():
        counts = {"supported_dominance_cells": 0, "direction_matches": 0,
                  "direction_mismatches": 0, "missing_response_metrics": 0}
        for key, effects in cells.items():
            action = decisions.get(key)
            if action not in ("ramp_dominates_step", "step_dominates_ramp"):
                continue
            if not all(metric in effects for metric in DECISION_METRICS):
                counts["missing_response_metrics"] += 1
                continue
            counts["supported_dominance_cells"] += 1
            wheel = effects["wheel_residual_mps"]
            slip_ratio = effects["wheel_slip_ratio_proxy"]
            accel = effects["longitudinal_accel_mps2"]
            matched = (
                wheel > 0.0 and slip_ratio > 0.0 and accel < 0.0
                if action == "ramp_dominates_step"
                else wheel < 0.0 and slip_ratio < 0.0 and accel > 0.0
            )
            counts["direction_matches" if matched else "direction_mismatches"] += 1
            cell_result = per_cell.setdefault(key, {
                **_typed_features(key),
                "training_action_evidence": action,
                "validation_run_count": 0,
                "validation_direction_matches": 0,
                "validation_direction_mismatches": 0,
            })
            cell_result["validation_run_count"] += 1
            cell_result["validation_direction_matches" if matched
                        else "validation_direction_mismatches"] += 1
        per_run[run_id] = counts
    total_supported = sum(row["supported_dominance_cells"]
                          for row in per_run.values())
    total_matches = sum(row["direction_matches"] for row in per_run.values())
    total_mismatches = sum(row["direction_mismatches"]
                           for row in per_run.values())
    per_cell_rows = []
    for key, row in sorted(per_cell.items()):
        row["recommendation_validated"] = (
            row["validation_run_count"] >= 2
            and row["validation_direction_mismatches"] == 0)
        per_cell_rows.append(row)
    return {
        "method": "paired sign check for training-only Pareto-dominance suggestions",
        "per_validation_run": per_run,
        "per_cell": per_cell_rows,
        "supported_validation_cells": total_supported,
        "direction_matches": total_matches,
        "direction_mismatches": total_mismatches,
        "match_fraction": (total_matches / total_supported
                           if total_supported else None),
    }


def build_model(training_paths: list[Path], validation_paths: list[Path], *,
                assume_left_right_symmetry: bool = False) -> dict[str, Any]:
    train_rows, train_sources = _read_split_rows(training_paths, "train")
    validation_rows, validation_sources = _read_split_rows(validation_paths, "validation")
    if assume_left_right_symmetry:
        # The simulator vehicle/track axes are mirror-symmetric. Pool the
        # randomized left/right probes within each capture so this remains a
        # tested modeling choice, not a one-sided data shortcut.
        for row in (*train_rows, *validation_rows):
            row["turn_direction"] = "both"
    train_cells = _run_cell_means(train_rows)
    validation_cells = _run_cell_means(validation_rows)
    train_ids = set(train_cells)
    validation_ids = set(validation_cells)
    overlap = train_ids & validation_ids
    if overlap:
        raise ValueError(f"training/validation capture leakage: {sorted(overlap)}")
    fitted_cells = _fit_cells(train_cells, train_ids)
    validation = _evaluate_runs(train_cells | validation_cells,
                                train_ids, validation_ids)
    return {
        "schema_version": 2,
        "model_kind": (
            "training-only left-right pooled multi-window exact-cell empirical throttle response lookup"
            if assume_left_right_symmetry else
            "training-only multi-window exact-cell empirical throttle response lookup"),
        "assume_left_right_symmetry": assume_left_right_symmetry,
        "interpretation": (
            "Predicts measured step-minus-ramp integrated response over "
            "0.10-1.00 s and, where recorded, separate 0.10-0.35, 0.35-0.65, "
            "and 0.65-1.00 s responses. It is not an absolute plant model or "
            "tire-force estimate, and it must not be interpolated or "
            "extrapolated beyond listed training cells."),
        "sources": {"training": train_sources, "validation": validation_sources},
        "training_pair_count": len(train_rows),
        "validation_pair_count": len(validation_rows),
        "training_capture_count": len(train_ids),
        "validation_capture_count": len(validation_ids),
        "feature_fields": list(FEATURE_FIELDS),
        "response_metrics": METRICS,
        "windowed_response_metrics": WINDOWED_METRICS,
        "training_cross_validation": _training_leave_one_run_out(train_cells,
                                                                   train_ids),
        "held_out_validation": {
            "method": "independent whole-capture validation; validation not fitted",
            **validation,
            "dominance_suggestion_check": _validate_action_evidence(
                validation_cells, fitted_cells),
        },
        "support_limited_lookup": {
            "matching": "exact rounded cell only; missing cell returns unsupported",
            "cells": fitted_cells,
        },
    }


def lookup_supported_effect(model: dict[str, Any], *, speed_target_mps: float,
                            throttle_delta_direction: str,
                            throttle_delta_norm: float,
                            throttle_rise_rate_norm_per_sec: float,
                            abs_steering_command_rad: float,
                            turn_direction: str,
                            min_independent_training_runs: int = 2
                            ) -> dict[str, float] | None:
    """Return a fitted paired effect only for an explicitly supported cell."""
    query = {
        "speed_target_mps": speed_target_mps,
        "throttle_delta_direction": throttle_delta_direction,
        "throttle_delta_norm": throttle_delta_norm,
        "throttle_rise_rate_norm_per_sec": throttle_rise_rate_norm_per_sec,
        "abs_steering_command_rad": abs_steering_command_rad,
        "turn_direction": turn_direction,
    }
    if model.get("assume_left_right_symmetry"):
        query["turn_direction"] = "both"
    key = tuple(_normalise_feature(query[field], field) for field in FEATURE_FIELDS)
    for cell in model.get("support_limited_lookup", {}).get("cells", []):
        if tuple(_normalise_feature(cell[field], field)
                 for field in FEATURE_FIELDS) == key and int(
                     cell.get("independent_training_run_count", 0)
                 ) >= min_independent_training_runs:
            return {metric: float(cell["effects"][metric]["step_minus_ramp_mean"])
                    for metric in ALL_METRICS if metric in cell["effects"]}
    return None


def recommend_throttle_profile(model: dict[str, Any], **features: Any) -> str | None:
    """Return a replicated, held-out-confirmed Pareto choice, else None."""
    query = {field: features[field] for field in FEATURE_FIELDS}
    if model.get("assume_left_right_symmetry"):
        query["turn_direction"] = "both"
    key = tuple(_normalise_feature(query[field], field) for field in FEATURE_FIELDS)
    for cell in model.get("support_limited_lookup", {}).get("cells", []):
        if (tuple(_normalise_feature(cell[field], field)
                  for field in FEATURE_FIELDS) == key
                and int(cell.get("independent_training_run_count", 0)) >= 2):
            evidence = cell.get("throttle_action_evidence")
            validation_cells = model.get("held_out_validation", {}).get(
                "dominance_suggestion_check", {}).get("per_cell", [])
            validation = next((entry for entry in validation_cells
                               if tuple(_normalise_feature(entry[field], field)
                                        for field in FEATURE_FIELDS) == key), None)
            if not validation or not validation.get("recommendation_validated"):
                return None
            if evidence == "ramp_dominates_step":
                return "ramp"
            if evidence == "step_dominates_ramp":
                return "step"
            return None
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-analysis", type=Path, action="append",
                        required=True, help="analysis JSON containing train rows")
    parser.add_argument("--validation-analysis", type=Path, action="append",
                        required=True, help="analysis JSON containing validation rows")
    parser.add_argument("--assume-left-right-symmetry", action="store_true",
                        help="pool left/right responses inside each run; validate this choice on held-out captures")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    training_paths = [path if path.is_absolute() else REPO_ROOT / path
                      for path in args.training_analysis]
    validation_paths = [path if path.is_absolute() else REPO_ROOT / path
                        for path in args.validation_analysis]
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    if output.exists():
        parser.error(f"refusing to overwrite existing output: {output}")
    model = build_model(
        training_paths, validation_paths,
        assume_left_right_symmetry=args.assume_left_right_symmetry)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(model, indent=2, sort_keys=True, allow_nan=False) + "\n",
                      encoding="utf-8")
    print(json.dumps({
        "training_pairs": model["training_pair_count"],
        "validation_pairs": model["validation_pair_count"],
        "training_captures": model["training_capture_count"],
        "validation_captures": model["validation_capture_count"],
        "training_cells": len(model["support_limited_lookup"]["cells"]),
        "validation_summary": model["held_out_validation"]["summary"],
        "output": str(output),
    }, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
