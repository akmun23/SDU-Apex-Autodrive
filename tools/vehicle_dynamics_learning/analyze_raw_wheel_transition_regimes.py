#!/usr/bin/env python3
"""Locate raw encoder one-step errors and support gaps by measured regime."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from tools.vehicle_dynamics_learning import evaluate_hybrid_longitudinal_edssm_candidate as ev
from tools.vehicle_dynamics_learning.evaluate_yaw_atlas_recursive_gain import _run_windows
from tools.vehicle_dynamics_learning.effective_race_teacher import (
    REAR_TRACK_WIDTH_M, physical_state_from_dataset)
from tools.vehicle_dynamics_learning.fit_full_regime_yaw_atlas_dagger_corrector import (
    HIGHSTEER_TRAIN_DATASET)
from tools.vehicle_dynamics_learning.fit_raw_encoder_transition_residual import (
    MODEL_PATH, OUT)
from tools.vehicle_dynamics_learning.fit_yaw_conditioned_u_wheel_dagger import sha256
from tools.vehicle_dynamics_learning.score_effective_race_teacher import _load_model
from tools.vehicle_dynamics_learning.train_nssm import _load_dataset


REPORT = OUT / "regime_support_analysis.json"
CHUNK_SIZE = 8192
SPECS = (
    ("broad_train", ev.HGB_TRAIN_DATASET, "train"),
    ("highsteer_train", HIGHSTEER_TRAIN_DATASET, "train"),
    ("broad_validation", ev.HGB_TRAIN_DATASET, "validation"),
    ("highsteer_validation", ev.HIGHSTEER_DATASET, "validation"),
    ("practice_unseen", ev.PRACTICE_DATASET, "unseen_practice"),
)


def _quantile_edges(values: np.ndarray) -> list[float]:
    edges = np.unique(np.quantile(values, (0.0, 0.25, 0.5, 0.75, 1.0)))
    return [float(value) for value in edges[1:-1]]


def _evaluate_source(name: str, path: Path, split: str, hgb: dict[str, Any],
                     metadata: dict[str, Any], models: list[Any],
                     gate: dict[str, tuple[float, float]]) -> dict[str, Any]:
    data = _load_dataset(path)
    windows = _run_windows(data, split, ev._bundle_lags(hgb),
                           2_000_000, horizon_steps=1)
    frames = np.asarray(data["frames"], dtype=np.float64)
    raw = np.asarray(data["encoder_raw_surface_mps"], dtype=np.float64)
    raw_valid = np.asarray(data["encoder_raw_valid"], dtype=bool)
    windows = [row for row in windows
               if raw_valid[row["start"]] and raw_valid[row["start"] + 1]
               and np.isfinite(raw[row["start"]]).all()
               and np.isfinite(raw[row["start"] + 1]).all()]
    if not windows:
        raise ValueError(f"no valid raw encoder transitions in {name}")
    state = physical_state_from_dataset(
        data, str(metadata.get("wheel_state_source", "filtered_odometry")))
    lags = ev._bundle_lags(hgb)
    max_lag = max(lags)
    starts = np.asarray([row["start"] for row in windows], dtype=np.int64)
    run_idx = np.asarray([row["run"] for row in windows], dtype=np.int32)
    run_ids = np.asarray(data["run_ids"]).astype(str)
    splits = np.asarray(data["splits"]).astype(str)
    if any(splits[row["run"]] != split for row in windows):
        raise ValueError(f"split mismatch in {name}")

    factors: dict[str, np.ndarray] = {}
    output = {key: [] for key in ("run_id", "baseline_error", "candidate_error")}
    raw_speeds, steers, throttles, throttle_slews, mismatches = [], [], [], [], []
    for offset in range(0, len(starts), CHUNK_SIZE):
        row = slice(offset, offset + CHUNK_SIZE)
        s = starts[row]
        hist_idx = s[:, None] - max_lag + np.arange(max_lag + 1)[None, :]
        history = np.empty((len(s), max_lag + 1, 9), dtype=np.float32)
        history[:, :, :7] = state[hist_idx, :7].astype(np.float32)
        history[:, :, 7:9] = frames[hist_idx, 7:9].astype(np.float32)
        x = ev._features(history, lags)
        raw_history = raw[hist_idx].astype(np.float32)
        raw_current = raw_history[:, -1]
        raw_parts = [raw_current]
        for lag in lags:
            raw_parts.append(raw_current - raw_history[:, -1 - lag])
        x = np.column_stack((x, *raw_parts)).astype(np.float32)
        if x.shape[1] != hgb["raw_wheel_models"][0].n_features_in_:
            raise ValueError("reconstructed raw-wheel feature contract differs")
        base_delta = np.column_stack([
            model.predict(x) for model in hgb["raw_wheel_models"]])
        residual = np.column_stack([model.predict(x) for model in models])
        body_speed = np.hypot(state[s, 0], state[s, 1])
        absolute_steering = np.abs(state[s, 3])
        in_support = (
            (body_speed >= gate["speed_mps"][0])
            & (body_speed <= gate["speed_mps"][1])
            & (absolute_steering >= gate["absolute_steering_rad"][0])
            & (absolute_steering <= gate["absolute_steering_rad"][1]))
        corrected_delta = base_delta.copy()
        corrected_delta[in_support] += residual[in_support]
        truth_next = raw[s + 1]
        output["run_id"].append(run_ids[run_idx[row]])
        output["baseline_error"].append(raw_current + base_delta - truth_next)
        output["candidate_error"].append(raw_current + corrected_delta - truth_next)

        current_speed = body_speed
        current_steer = absolute_steering
        throttle_command = frames[s, 8]
        throttle_slew = frames[s, 8] - frames[s - 1, 8]
        half_track = REAR_TRACK_WIDTH_M / 2.0
        contact_speed = np.column_stack((
            state[s, 0] - half_track * state[s, 2],
            state[s, 0] + half_track * state[s, 2]))
        wheel_mismatch = np.mean(np.abs(raw[s] - contact_speed), axis=1)
        raw_speeds.append(current_speed)
        steers.append(current_steer)
        throttles.append(throttle_command)
        throttle_slews.append(throttle_slew)
        mismatches.append(wheel_mismatch)

    speed = np.concatenate(raw_speeds)
    steer = np.concatenate(steers)
    throttle = np.concatenate(throttles)
    throttle_slew = np.concatenate(throttle_slews)
    mismatch = np.concatenate(mismatches)
    factors["speed_bin"] = np.floor(speed).astype(np.int16)
    factors["steering_bin"] = np.floor(steer / 0.05).astype(np.int16)
    factors["joint_bin"] = np.column_stack((
        factors["speed_bin"], factors["steering_bin"])).astype(np.int16)
    factors["throttle_bin"] = np.digitize(throttle, _THROTTLE_EDGES)
    factors["throttle_slew_sign"] = np.sign(throttle_slew).astype(np.int8)
    factors["wheel_mismatch_bin"] = np.digitize(
        mismatch, _MISMATCH_EDGES).astype(np.int8)
    return {
        "name": name,
        "split": split,
        "run_ids": sorted(set(np.concatenate(output["run_id"]).tolist())),
        "rows": len(windows),
        "factors": factors,
        "run_ids_per_row": np.concatenate(output["run_id"]),
        "baseline_error": np.concatenate(output["baseline_error"]),
        "candidate_error": np.concatenate(output["candidate_error"]),
        "factor_edges": {
            "speed_bin_width_mps": 1.0,
            "steering_bin_width_rad": 0.05,
            "throttle_command_quartiles_train_only": _THROTTLE_EDGES,
            "abs_encoder_contact_speed_mismatch_quartiles_train_only": _MISMATCH_EDGES,
        },
    }


def _summarize(source: dict[str, Any], factor_values: np.ndarray
               ) -> dict[str, Any]:
    by_bucket: dict[tuple[int, ...], dict[str, Any]] = {}
    keys = np.asarray(factor_values)
    if keys.ndim == 1:
        keys = keys[:, None]
    run_ids = source["run_ids_per_row"]
    for key in sorted({tuple(int(v) for v in row) for row in keys.tolist()}):
        row_mask = np.all(keys == np.asarray(key), axis=1)
        runs = {}
        for run_id in sorted(set(run_ids[row_mask].tolist())):
            mask = row_mask & (run_ids == run_id)
            baseline = source["baseline_error"][mask]
            candidate = source["candidate_error"][mask]
            runs[run_id] = {
                "samples": int(np.count_nonzero(mask)),
                "baseline_rmse_mps": float(np.sqrt(np.mean(baseline ** 2))),
                "candidate_rmse_mps": float(np.sqrt(np.mean(candidate ** 2))),
            }
        by_bucket[",".join(str(v) for v in key)] = {
            "samples": int(np.count_nonzero(row_mask)),
            "run_count": len(runs),
            "run_macro_baseline_rmse_mps": float(np.mean([
                row["baseline_rmse_mps"] for row in runs.values()])),
            "run_macro_candidate_rmse_mps": float(np.mean([
                row["candidate_rmse_mps"] for row in runs.values()])),
            "run_macro_candidate_minus_baseline_mps": float(np.mean([
                row["candidate_rmse_mps"] - row["baseline_rmse_mps"]
                for row in runs.values()])),
            "per_run": runs,
        }
    return by_bucket


def _metric_summary(source: dict[str, Any]) -> dict[str, Any]:
    result = {"rows": source["rows"], "run_ids": source["run_ids"],
              "all_run_macro_baseline_rmse_mps": float(np.mean([
                  np.sqrt(np.mean(source["baseline_error"][
                      source["run_ids_per_row"] == run] ** 2))
                  for run in source["run_ids"]])),
              "all_run_macro_candidate_rmse_mps": float(np.mean([
                  np.sqrt(np.mean(source["candidate_error"][
                      source["run_ids_per_row"] == run] ** 2))
                  for run in source["run_ids"]]))}
    for factor, values in source["factors"].items():
        result[factor] = _summarize(source, values)
    return result


def run() -> dict[str, Any]:
    if REPORT.exists():
        raise FileExistsError(f"refusing to overwrite {REPORT}")
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(MODEL_PATH)
    hgb = joblib.load(ev.HGB_BUNDLE)
    bundle = joblib.load(MODEL_PATH)
    if bundle.get("hgb_bundle_sha256") != sha256(ev.HGB_BUNDLE):
        raise ValueError("raw-wheel candidate HGB hash mismatch")
    models = bundle["models"]
    gate = {key: tuple(value) for key, value in bundle["training_gate"].items()}
    _, _, metadata = _load_model(ev.EDSSM_CHECKPOINT, "cpu")
    # Quantile boundaries are derived only from training transitions.
    train_raw_speeds, train_steers, train_throttle, train_mismatch = [], [], [], []
    for name, path, split in SPECS[:2]:
        data = _load_dataset(path)
        frames = np.asarray(data["frames"], dtype=np.float64)
        raw = np.asarray(data["encoder_raw_surface_mps"], dtype=np.float64)
        valid = np.asarray(data["encoder_raw_valid"], dtype=bool)
        state = physical_state_from_dataset(
            data, str(metadata.get("wheel_state_source", "filtered_odometry")))
        windows = _run_windows(data, split, ev._bundle_lags(hgb),
                               2_000_000, horizon_steps=1)
        starts = np.asarray([row["start"] for row in windows], dtype=np.int64)
        good = valid[starts] & valid[starts + 1]
        starts = starts[good]
        speed = np.hypot(state[starts, 0], state[starts, 1])
        half_track = REAR_TRACK_WIDTH_M / 2.0
        contact = np.column_stack((
            state[starts, 0] - half_track * state[starts, 2],
            state[starts, 0] + half_track * state[starts, 2]))
        mismatch = np.mean(np.abs(raw[starts] - contact), axis=1)
        train_raw_speeds.append(speed)
        train_steers.append(np.abs(state[starts, 3]))
        train_throttle.append(frames[starts, 8])
        train_mismatch.append(mismatch)
    global _THROTTLE_EDGES, _MISMATCH_EDGES
    _THROTTLE_EDGES = _quantile_edges(np.concatenate(train_throttle))
    _MISMATCH_EDGES = _quantile_edges(np.concatenate(train_mismatch))
    sources = []
    for name, path, split in SPECS:
        sources.append(_evaluate_source(
            name, path, split, hgb, metadata, models, gate))
    train_cells: dict[str, int] = {}
    for source in sources[:2]:
        for key, value in _summarize(
                source, source["factors"]["joint_bin"]).items():
            train_cells[key] = train_cells.get(key, 0) + value["samples"]
    cohorts = {}
    for source in sources:
        summary = _metric_summary(source)
        joint = summary["joint_bin"]
        for key, value in joint.items():
            value["train_support_rows"] = train_cells.get(key, 0)
        summary.pop("joint_bin", None)
        summary["joint_speed_steering_bins"] = joint
        summary["validation_cells_without_training_support"] = [
            {"speed_bin": int(key.split(",")[0]),
             "steering_bin": int(key.split(",")[1]),
             "validation_rows": value["samples"]}
            for key, value in joint.items()
            if source["split"] != "train" and value["train_support_rows"] == 0]
        cohorts[source["name"]] = summary
    report = {
        "schema_version": 1,
        "purpose": "determine whether remaining raw-wheel one-step errors align with speed/steering, throttle, slew, or encoder/body mismatch and identify unsupported validation cells",
        "simulator_was_run": False,
        "test_or_final_test_used": False,
        "training_only_quantile_boundaries": {
            "throttle_command_quartiles": _THROTTLE_EDGES,
            "encoder_contact_speed_mismatch_quartiles_mps": _MISMATCH_EDGES,
        },
        "speed_steering_cells": {
            "speed_bin_width_mps": 1.0,
            "absolute_steering_bin_width_rad": 0.05,
        },
        "wheel_mismatch_definition": (
            "mean absolute raw encoder surface-speed minus rigid-body rear-contact "
            "longitudinal speed; proxy only, not simulator tire-slip truth"),
        "cohorts": cohorts,
        "limitations": [
            "raw encoders are measured wheel signals, not simulator wheel angular-speed truth",
            "groups are descriptive diagnostics; they did not select or refit the model",
            "practice contains only two independent captures",
        ],
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


if __name__ == "__main__":
    print(json.dumps({"report": str(REPORT),
                      "unsupported_practice_bins": run()["cohorts"][
                          "practice_unseen"][
                              "validation_cells_without_training_support"]},
                     indent=2))
