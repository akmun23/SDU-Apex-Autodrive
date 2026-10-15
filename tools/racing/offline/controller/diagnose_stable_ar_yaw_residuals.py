#!/usr/bin/env python3
"""Localize stable-AR yaw errors by operating regime on held-out captures."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from tools.racing.offline.controller import (
    evaluate_mpc_practice_yaw_residual as yaw,
    fit_causal_history_yaw_lateral_plant as teacher,
)


OUTPUT = yaw.ANALYSIS_DIR / "stable_ar_yaw_residual_signal_diagnosis_20261012.json"
MODEL_REPORT = yaw.ANALYSIS_DIR / "stable_ar_yaw_response_20261012.json"
BOUNDS = {
    "speed_mps": (0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, np.inf),
    "abs_steering_command_rad": (0.0, 0.10, 0.20, 0.30, 0.40, 0.50, np.inf),
    "abs_steering_rate_radps": (0.0, 0.2, 0.5, 1.0, 2.0, np.inf),
    "abs_throttle_slew_normps": (0.0, 0.1, 0.5, 1.0, 2.0, np.inf),
    "throttle_command_norm": (0.0, 0.10, 0.25, 0.40, 0.50, np.inf),
    "abs_steering_feedback_command_error_rad": (0.0, 0.01, 0.03, 0.06, 0.10, np.inf),
    "abs_steering_feedback_rate_radps": (0.0, 0.2, 0.5, 1.0, 2.0, np.inf),
    "abs_throttle_feedback_command_error_norm": (0.0, 0.01, 0.03, 0.06, 0.10, np.inf),
    "abs_throttle_feedback_slew_normps": (0.0, 0.1, 0.5, 1.0, 2.0, np.inf),
    "abs_wheel_body_speed_mismatch_mps": (0.0, 0.1, 0.25, 0.5, 1.0, np.inf),
    "abs_wheel_asymmetry_mps": (0.0, 0.1, 0.25, 0.5, 1.0, np.inf),
    "abs_imu_lateral_acceleration_mps2": (0.0, 1.0, 2.0, 4.0, 8.0, np.inf),
}


def _metric(error: np.ndarray) -> dict[str, Any]:
    if not len(error):
        return {"samples": 0, "rmse_radps": None, "p95_abs_radps": None,
                "max_abs_radps": None, "fraction_abs_le_0p1": None}
    absolute = np.abs(error)
    return {
        "samples": int(len(error)),
        "rmse_radps": float(np.sqrt(np.mean(error ** 2))),
        "p95_abs_radps": float(np.quantile(absolute, 0.95)),
        "max_abs_radps": float(np.max(absolute)),
        "fraction_abs_le_0p1": float(np.mean(absolute <= 0.1)),
    }


def _sequence_rows(data: dict[str, Any], run_ids: set[str], model: Any,
                   columns: list[int], rho: float) -> dict[str, np.ndarray]:
    collected: dict[str, list[np.ndarray]] = defaultdict(list)
    sequence_serial = 0
    state = data["state"]
    for sequence_id, (start_raw, end_raw) in enumerate(data["sequence_bounds"]):
        run_index = int(data["sequence_run_index"][sequence_id])
        run_id = str(data["run_ids"][run_index])
        if run_id not in run_ids:
            continue
        start, end = int(start_raw), int(end_raw)
        first = start + teacher.HISTORY
        if end - first < 2:
            continue
        histories = [row.copy() for row in state[start:first + 1]]
        exogenous = []
        for index in range(first, end - 1):
            exogenous.append(teacher._features(np.asarray(
                histories[-(teacher.HISTORY + 1):]))[columns])
            # The selected feature set excludes yaw and its history, so the
            # recorded exogenous history gives identical inputs during rollout.
            histories.append(state[index + 1].copy())
        delta = np.asarray(model.predict(np.asarray(exogenous)), dtype=np.float64)
        truth_current = state[first:end - 1, 2]
        truth_next = state[first + 1:end, 2]
        teacher_forced = rho * truth_current + delta
        recursive = np.empty_like(delta)
        current_rate = float(truth_current[0])
        for offset, learned_delta in enumerate(delta):
            current_rate = rho * current_rate + float(learned_delta)
            recursive[offset] = current_rate

        current = state[first:end - 1]
        prior1 = state[first - 1:end - 2]
        previous4 = state[first - teacher.HISTORY:end - 1 - teacher.HISTORY]
        records = np.column_stack((
            teacher_forced - truth_next,
            recursive - truth_next,
            np.hypot(current[:, 0], current[:, 1]),
            np.abs(current[:, 7]),
            np.abs((current[:, 7] - state[first - 1:end - 2, 7]) / teacher.DT),
            np.abs((current[:, 8] - previous4[:, 8])
                   / (teacher.HISTORY * teacher.DT)),
            current[:, 8],
            np.abs(current[:, 3] - current[:, 7]),
            np.abs((current[:, 3] - prior1[:, 3]) / teacher.DT),
            np.abs(current[:, 4] - current[:, 8]),
            np.abs((current[:, 4] - previous4[:, 4])
                   / (teacher.HISTORY * teacher.DT)),
            np.abs(0.5 * (current[:, 5] + current[:, 6]) - current[:, 0]),
            np.abs(current[:, 5] - current[:, 6]),
            np.abs(current[:, 10]),
            np.full(len(current), sequence_serial, dtype=np.float64),
        ))
        collected[run_id].append(records)
        sequence_serial += 1
    return {run_id: np.concatenate(parts) for run_id, parts in collected.items()}


def _summarize(rows: np.ndarray) -> dict[str, Any]:
    if rows.size == 0:
        return {"samples": 0, "teacher_forced": _metric(np.empty(0)),
                "recursive": _metric(np.empty(0)), "by_regime": {}}
    result: dict[str, Any] = {
        "samples": int(len(rows)),
        "sequence_count": int(len(np.unique(rows[:, 14]))),
        "teacher_forced": _metric(rows[:, 0]),
        "recursive": _metric(rows[:, 1]),
        "by_regime": {},
    }
    for name, edges in BOUNDS.items():
        column = {
            "speed_mps": 2,
            "abs_steering_command_rad": 3,
            "abs_steering_rate_radps": 4,
            "abs_throttle_slew_normps": 5,
            "throttle_command_norm": 6,
            "abs_steering_feedback_command_error_rad": 7,
            "abs_steering_feedback_rate_radps": 8,
            "abs_throttle_feedback_command_error_norm": 9,
            "abs_throttle_feedback_slew_normps": 10,
            "abs_wheel_body_speed_mismatch_mps": 11,
            "abs_wheel_asymmetry_mps": 12,
            "abs_imu_lateral_acceleration_mps2": 13,
        }[name]
        groups = {}
        for lower, upper in zip(edges[:-1], edges[1:]):
            mask = (rows[:, column] >= lower) & (rows[:, column] < upper)
            groups[f"{lower:g}_{upper:g}"] = {
                "teacher_forced": _metric(rows[mask, 0]),
                "recursive": _metric(rows[mask, 1]),
            }
        result["by_regime"][name] = groups

    # Find consecutive fixed-grid stretches where recursive error is over the
    # user's 0.1 rad/s target; this reveals whether failures cluster in time.
    segments = []
    for sequence_serial in np.unique(rows[:, 14]):
        sequence_rows = rows[rows[:, 14] == sequence_serial]
        bad = np.abs(sequence_rows[:, 1]) > 0.1
        start = None
        for index, is_bad in enumerate(np.r_[bad, False]):
            if is_bad and start is None:
                start = index
            elif not is_bad and start is not None:
                end = index
                segment = sequence_rows[start:end]
                segments.append({
                    "sequence_serial": int(sequence_serial),
                    "start_row": start,
                    "duration_s": (end - start) * teacher.DT,
                    "samples": int(end - start),
                    "recursive_error_max_abs_radps": float(
                        np.max(np.abs(segment[:, 1]))),
                    "mean_speed_mps": float(np.mean(segment[:, 2])),
                    "mean_abs_steering_command_rad": float(np.mean(segment[:, 3])),
                    "mean_abs_steering_rate_radps": float(np.mean(segment[:, 4])),
                    "mean_abs_throttle_slew_normps": float(np.mean(segment[:, 5])),
                })
                start = None
    result["over_0p1_stretches"] = {
        "count": len(segments),
        "longest": sorted(segments, key=lambda item: item["duration_s"],
                           reverse=True)[:10],
    }
    return result


def run() -> dict[str, Any]:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    model_report = json.loads(MODEL_REPORT.read_text(encoding="utf-8"))
    artifact = joblib.load(teacher.ROOT / model_report["artifact"])
    model = artifact["model"]
    rho = float(artifact["rho"])
    columns = [teacher.FEATURE_NAMES.index(name)
               for name in artifact["feature_names"]]
    benchmark = json.loads(yaw.BENCHMARK.read_text(encoding="utf-8"))
    base_path = yaw._workspace_path(benchmark["dataset"])
    if teacher._sha256(base_path) != benchmark["dataset_sha256"]:
        raise ValueError("frozen benchmark dataset hash changed")
    base_ids = {}
    with np.load(base_path, allow_pickle=False) as archive:
        run_ids = archive["run_ids"].astype(str)
        splits = archive["run_splits"].astype(str)
        for split in np.unique(splits):
            base_ids[str(split)] = sorted(set(run_ids[splits == split]))
    datasets = {
        "race_domain_development": (
            teacher._collect_rows(base_path, "validation",
                                  set(base_ids["validation"])
                                  - teacher.VALIDATION_IDS),
            set(base_ids["validation"]) - teacher.VALIDATION_IDS),
        "frozen_practice_transfer": (
            teacher._collect_rows(base_path, "validation",
                                  teacher.VALIDATION_IDS),
            set(teacher.VALIDATION_IDS)),
    }
    with np.load(teacher.PAIRED_DATASET, allow_pickle=False) as archive:
        paired_ids = set(archive["run_ids"].astype(str)[
            archive["run_splits"].astype(str) == "validation"])
    datasets["paired_throttle_slew_development"] = (
        teacher._collect_rows(teacher.PAIRED_DATASET, "validation", paired_ids),
        paired_ids,
    )

    report = {
        "study": "held-out stable-AR yaw residual localization by speed, steering, and throttle history",
        "offline_only": True,
        "runtime_changed": False,
        "simulator_was_run": False,
        "rho": rho,
        "oracle_inputs": ["future u/v", "steering-command history"],
        "diagnostic_only_context": [
            "actuator feedback mismatch", "encoder-derived wheel/body-speed mismatch",
            "wheel asymmetry", "IMU lateral acceleration"],
        "excluded_inputs": ["future measured sensors", "future ground truth yaw rate"],
        "data": {},
        "limitations": [
            "The row-level recursive sequence is scored with oracle future u/v and steering-command histories.",
            "Run-level bins are descriptive; adjacent samples are not independent observations.",
            "Only frozen validation and practice-transfer rows are selected; packed archives containing test/final-test are now refused before numeric loading.",
        ],
    }
    for label, (data, run_ids_for_source) in datasets.items():
        per_run = _sequence_rows(data, run_ids_for_source, model, columns, rho)
        report["data"][label] = {
            "run_ids": sorted(per_run),
            "per_run": {run_id: _summarize(rows)
                        for run_id, rows in sorted(per_run.items())},
            "pooled": _summarize(np.concatenate(list(per_run.values()))
                                 if per_run else np.empty((0, 15))),
        }
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return report


if __name__ == "__main__":
    result = run()
    print(json.dumps({
        "output": str(OUTPUT),
        "pooled": {name: value["pooled"] for name, value in result["data"].items()},
    }, indent=2))
