#!/usr/bin/env python3
"""Fit one causal lateral-velocity increment model on the trusted P0 runs.

P0 r01 is the fit set and P0 r02 is evaluated once as a whole-run holdout.
Only scored laps 2--11 are used. All intervals are treated as exactly 25 ms,
per the simulator bridge contract; message timestamps are not used as dt.
This is a sensor-conditioned one-step candidate, not a full plant or runtime
odometry model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DT_S = 0.025
TRAIN_DEFAULT = ROOT / (
    "live_runs/practice_9g_parent_reproduced_20261006_r01/analysis/"
    "race_report_yaw_residual_input/tracking_error.csv"
)
VALIDATION_DEFAULT = ROOT / (
    "live_runs/practice_9g_parent_reproduced_20261006_r02/analysis/"
    "race_report_yaw_residual_input/tracking_error.csv"
)
FEATURES = (
    "v_mps", "u_mps", "r_radps", "steering_feedback_rad", "throttle_feedback",
    "steering_slew_radps", "throttle_slew_per_s", "u_times_r", "u_times_delta",
    "r_times_delta",
)
NO_THROTTLE_FEATURE_INDICES = (0, 1, 2, 3, 5, 7, 8, 9)
NO_THROTTLE_FEATURES = tuple(
    FEATURES[index] for index in NO_THROTTLE_FEATURE_INDICES)
LAPS = set(range(2, 12))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_run(path: Path, run_id: str) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    by_lap: dict[int, list[dict[str, float]]] = {}
    required = (
        "truth_speed_mps", "truth_lateral_speed_mps", "yaw_rate_radps",
        "steering_feedback_rad", "throttle_feedback",
    )
    with path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            try:
                lap = int(row["lap_number"])
                values = {name: float(row[name]) for name in required}
            except (KeyError, TypeError, ValueError):
                continue
            if lap not in LAPS or not all(math.isfinite(value) for value in values.values()):
                continue
            by_lap.setdefault(lap, []).append(values)

    features: list[list[float]] = []
    increments: list[float] = []
    metadata: list[dict[str, Any]] = []
    for lap, rows in sorted(by_lap.items()):
        # Do not create a synthetic zero slew or join history across lap
        # boundaries; the first within-lap pair is omitted.
        for index in range(1, len(rows) - 1):
            current = rows[index]
            previous = rows[index - 1]
            following = rows[index + 1]
            u = current["truth_speed_mps"]
            v = current["truth_lateral_speed_mps"]
            r = current["yaw_rate_radps"]
            delta = current["steering_feedback_rad"]
            throttle = current["throttle_feedback"]
            steering_slew = (
                delta - previous["steering_feedback_rad"]) / DT_S
            throttle_slew = (
                throttle - previous["throttle_feedback"]) / DT_S
            features.append([
                v, u, r, delta, throttle, steering_slew, throttle_slew,
                u * r, u * delta, r * delta,
            ])
            increments.append(
                following["truth_lateral_speed_mps"] - v)
            metadata.append({
                "run_id": run_id,
                "lap_number": lap,
                "u_mps": u,
                "v_mps": v,
                "r_radps": r,
                "steering_feedback_rad": delta,
                "throttle_feedback": throttle,
            })
    if not features:
        raise ValueError(f"No scored 25 ms pairs found in {path}")
    return (np.asarray(features, dtype=np.float64),
            np.asarray(increments, dtype=np.float64), metadata)


def metrics(error: np.ndarray) -> dict[str, float | int]:
    absolute = np.abs(error)
    return {
        "n": int(len(error)),
        "bias_mps": float(np.mean(error)),
        "rmse_mps": float(np.sqrt(np.mean(error * error))),
        "mae_mps": float(np.mean(absolute)),
        "p95_abs_mps": float(np.quantile(absolute, 0.95)),
    }


def fit_ridge(features: np.ndarray, increments: np.ndarray, alpha: float,
              feature_names: tuple[str, ...] = FEATURES) -> dict[str, Any]:
    means = features.mean(axis=0)
    scales = features.std(axis=0)
    scales[scales < 1.0e-10] = 1.0
    normalized = (features - means) / scales
    target_mean = float(increments.mean())
    centered_target = increments - target_mean
    coefficients = np.linalg.solve(
        normalized.T @ normalized + alpha * np.eye(normalized.shape[1]),
        normalized.T @ centered_target,
    )
    return {
        "feature_names": list(feature_names),
        "feature_means": means.tolist(),
        "feature_scales": scales.tolist(),
        "standardized_coefficients": coefficients.tolist(),
        "increment_intercept_mps": target_mean,
        "ridge_alpha": alpha,
    }


def predict_increment(model: dict[str, Any], features: np.ndarray) -> np.ndarray:
    means = np.asarray(model["feature_means"], dtype=np.float64)
    scales = np.asarray(model["feature_scales"], dtype=np.float64)
    coefficients = np.asarray(
        model["standardized_coefficients"], dtype=np.float64)
    return ((features - means) / scales) @ coefficients + float(
        model["increment_intercept_mps"])


def validation_summary(increments: np.ndarray, metadata: list[dict[str, Any]],
                       predictions: dict[str, np.ndarray]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "whole_run": {},
        "per_lap_rmse_mps": {},
        "support": {},
    }
    for name, predicted in predictions.items():
        result["whole_run"][name] = metrics(predicted - increments)
        per_lap = {}
        for lap in sorted({int(row["lap_number"]) for row in metadata}):
            mask = np.asarray([int(row["lap_number"]) == lap for row in metadata])
            per_lap[str(lap)] = metrics(predicted[mask] - increments[mask])[
                "rmse_mps"]
        result["per_lap_rmse_mps"][name] = per_lap

    held_rmse = np.asarray(list(result["per_lap_rmse_mps"]["hold_v"].values()))
    candidate_rmse = np.asarray(
        list(result["per_lap_rmse_mps"]["ridge_v_increment"].values()))
    paired_gain = held_rmse - candidate_rmse
    rng = np.random.default_rng(20261006)
    bootstrap = rng.choice(
        paired_gain, size=(10000, len(paired_gain)), replace=True).mean(axis=1)
    result["paired_lap_rmse_gain_hold_minus_candidate_mps"] = {
        "mean": float(paired_gain.mean()),
        "bootstrap_95pct_ci": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "laps": int(len(paired_gain)),
    }

    regimes = {
        "absolute_steering_ge_0p2_rad": lambda row: abs(
            float(row["steering_feedback_rad"])) >= 0.2,
        "speed_ge_6_mps": lambda row: float(row["u_mps"]) >= 6.0,
        "joint_speed_ge_6_and_abs_steering_ge_0p2": lambda row: (
            float(row["u_mps"]) >= 6.0
            and abs(float(row["steering_feedback_rad"])) >= 0.2
        ),
    }
    for regime, predicate in regimes.items():
        mask = np.asarray([predicate(row) for row in metadata])
        result["support"][regime] = {
            "n": int(mask.sum()),
            "hold_v": metrics(predictions["hold_v"][mask] - increments[mask])
                if mask.any() else None,
            "ridge_v_increment": metrics(
                predictions["ridge_v_increment"][mask] - increments[mask])
                if mask.any() else None,
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=TRAIN_DEFAULT)
    parser.add_argument("--validation", type=Path, default=VALIDATION_DEFAULT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    args = parser.parse_args()
    if args.ridge_alpha <= 0.0:
        raise ValueError("ridge alpha must be positive")

    train_x, train_y, train_meta = load_run(args.train, "P0-r01")
    validation_x, validation_y, validation_meta = load_run(
        args.validation, "P0-r02")
    model = fit_ridge(train_x, train_y, args.ridge_alpha)
    train_prediction = predict_increment(model, train_x)
    validation_prediction = predict_increment(model, validation_x)
    no_throttle_train_x = train_x[:, NO_THROTTLE_FEATURE_INDICES]
    no_throttle_validation_x = validation_x[:, NO_THROTTLE_FEATURE_INDICES]
    no_throttle_model = fit_ridge(
        no_throttle_train_x, train_y, args.ridge_alpha, NO_THROTTLE_FEATURES)
    no_throttle_train_prediction = predict_increment(
        no_throttle_model, no_throttle_train_x)
    no_throttle_validation_prediction = predict_increment(
        no_throttle_model, no_throttle_validation_x)

    def static_increment(meta: dict[str, Any]) -> float:
        speed = max(float(meta["u_mps"]), 0.0)
        yaw_rate = float(meta["r_radps"])
        at_reference = float(np.clip(
            yaw_rate * (0.167 - 0.0063 * speed), -0.35, 0.35))
        estimated_v = at_reference - yaw_rate * 0.15532
        return estimated_v - float(meta["v_mps"])

    hold_train = np.zeros_like(train_y)
    hold_validation = np.zeros_like(validation_y)
    static_validation = np.asarray(
        [static_increment(row) for row in validation_meta], dtype=np.float64)
    no_throttle_summary = validation_summary(
        validation_y, validation_meta, {
            "hold_v": hold_validation,
            "ridge_v_increment": no_throttle_validation_prediction,
        })
    no_throttle_summary["whole_run"]["no_throttle_v_increment"] = (
        no_throttle_summary["whole_run"].pop("ridge_v_increment"))
    no_throttle_summary["per_lap_rmse_mps"]["no_throttle_v_increment"] = (
        no_throttle_summary["per_lap_rmse_mps"].pop("ridge_v_increment"))
    for regime in no_throttle_summary["support"].values():
        regime["no_throttle_v_increment"] = regime.pop("ridge_v_increment")
    report = {
        "schema_version": 1,
        "candidate_id": "p0_lateral_velocity_increment_ridge_v1",
        "status": "research_only_not_full_plant_validated",
        "dt_s_assumed": DT_S,
        "timestamp_policy": "fixed 25 ms; recorded timestamp jitter is ignored",
        "training": {
            "run_id": "practice_9g_parent_reproduced_20261006_r01",
            "scored_laps": sorted(LAPS),
            "source_csv": str(args.train.resolve()),
            "source_sha256": sha256(args.train),
            "samples": int(len(train_y)),
            "hold_v": metrics(hold_train - train_y),
            "ridge_v_increment": metrics(train_prediction - train_y),
        },
        "validation": {
            "run_id": "practice_9g_parent_reproduced_20261006_r02",
            "scored_laps": sorted(LAPS),
            "source_csv": str(args.validation.resolve()),
            "source_sha256": sha256(args.validation),
            "samples": int(len(validation_y)),
            **validation_summary(validation_y, validation_meta, {
                "hold_v": hold_validation,
                "existing_static_v_map": static_validation,
                "ridge_v_increment": validation_prediction,
            }),
        },
        "model": model,
        "no_throttle_training": {
            "hold_v": metrics(hold_train - train_y),
            "no_throttle_v_increment": metrics(
                no_throttle_train_prediction - train_y),
        },
        "no_throttle_validation": no_throttle_summary,
        "no_throttle_model": no_throttle_model,
        "interpretation": (
            "Inputs are current measured body speed, lateral speed, yaw rate, "
            "steering feedback, throttle feedback, and their one-step actuator "
            "slews; the target is the next 25 ms lateral-speed increment. The "
            "fit is teacher-forced one-step evaluation only. It does not yet "
            "establish recursive vehicle/pose accuracy, practice-raceline "
            "transfer beyond this P0 track, or support the unobserved joint "
            "high-speed/high-steering regime. Do not integrate into MPC or "
            "odometry from this result alone. A second candidate removes all "
            "throttle terms and uses body state, current steering, measured "
            "one-step steering slew, and their interactions. It is retained "
            "for recursive testing where throttle history is not represented "
            "as an internal predicted state."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "candidate_id": report["candidate_id"],
        "train_hold_rmse_mps": report["training"]["hold_v"]["rmse_mps"],
        "train_candidate_rmse_mps": report["training"]["ridge_v_increment"]["rmse_mps"],
        "validation": report["validation"]["whole_run"],
        "no_throttle_validation": report["no_throttle_validation"]["whole_run"],
        "paired_lap_gain": report["validation"][
            "paired_lap_rmse_gain_hold_minus_candidate_mps"],
        "support": report["validation"]["support"],
        "output": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
