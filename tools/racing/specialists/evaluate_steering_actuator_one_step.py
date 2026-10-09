#!/usr/bin/env python3
"""Evaluate causal one-packet steering-actuator models on a clean probe run.

The target is measured steering feedback at k+1. Features use only sensor and
command history through k; simulator truth is not an input. Training is
restricted to clean train-split captures in the requested speed/steering
domain. Validation phases are read from the completed open-plane experiment
bag and remain excluded from fitting.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

try:
    import fit_fullband_yaw_regime_atlas as atlas
    from score_yaw_atlas_transition_events import (
        PROFILE_PREFIX, _bucket, _phase_lookup, _probe_phases)
except ModuleNotFoundError:
    from tools.racing.specialists import fit_fullband_yaw_regime_atlas as atlas
    from tools.racing.specialists.score_yaw_atlas_transition_events import (
        PROFILE_PREFIX, _bucket, _phase_lookup, _probe_phases)


DT_S = 0.025
FEATURE_NAMES = (
    "steering_feedback_k_rad",
    "steering_command_k_rad",
    "steering_command_k_minus_1_rad",
    "steering_command_k_minus_2_rad",
    "steering_feedback_k_minus_1_rad",
    "steering_feedback_k_minus_2_rad",
    "throttle_feedback_k_norm",
    "throttle_command_k_norm",
)


def _rows(series: Any, speed_range: tuple[float, float],
          steering_min: float) -> dict[str, np.ndarray]:
    features: list[list[float]] = []
    target: list[float] = []
    simple_predictions: dict[str, list[float]] = defaultdict(list)
    speed_values: list[float] = []
    steering_values: list[float] = []
    frame_indices: list[int] = []
    for begin_raw, end_raw in series.bounds:
        begin, end = int(begin_raw), int(end_raw)
        for k in range(begin + 2, end - 1):
            speed = float(np.hypot(series.rigid[k, 7], series.rigid[k, 8]))
            steering = float(series.frames[k, 3])
            if not (speed_range[0] <= speed < speed_range[1]
                    and steering_min <= abs(steering) <= 0.525):
                continue
            command_delay1 = float(series.frames[k - 1, 7])
            command_delay2 = float(series.frames[k - 2, 7])
            current_command = float(series.frames[k, 7])
            feedback_k1 = float(series.frames[k - 1, 3])
            feedback_k2 = float(series.frames[k - 2, 3])
            values = [
                steering, current_command, command_delay1, command_delay2,
                feedback_k1, feedback_k2, float(series.frames[k, 4]),
                float(series.frames[k, 8]),
            ]
            if not np.isfinite(values).all():
                continue
            for delay, command in enumerate((current_command,
                                             command_delay1,
                                             command_delay2)):
                magnitude_decrease = abs(command) < abs(steering)
                same_sign_release = (
                    steering * command >= -1.0e-6
                    and abs(command) + 0.001 < abs(steering))
                step = 3.2 * DT_S
                rate_limited = steering + float(np.clip(
                    command - steering, -step, step))
                simple_predictions[f"direct_command_k_minus_{delay}"].append(command)
                simple_predictions[f"rate_limit_3p2_k_minus_{delay}"].append(
                    rate_limited)
                simple_predictions[
                    f"hybrid_magnitude_decrease_or_limit_k_minus_{delay}"].append(
                        command if magnitude_decrease else rate_limited)
                simple_predictions[
                    f"hybrid_same_sign_release_or_limit_k_minus_{delay}"].append(
                        command if same_sign_release else rate_limited)
            features.append(values)
            target.append(float(series.frames[k + 1, 3]))
            speed_values.append(speed)
            steering_values.append(steering)
            frame_indices.append(k)
    return {
        "x": np.asarray(features, dtype=np.float32),
        "y": np.asarray(target, dtype=np.float32),
        **{name: np.asarray(values, dtype=np.float32)
           for name, values in simple_predictions.items()},
        "speed": np.asarray(speed_values, dtype=np.float32),
        "steering": np.asarray(steering_values, dtype=np.float32),
        "frame": np.asarray(frame_indices, dtype=np.int64),
    }


def _metrics(error: np.ndarray) -> dict[str, float | int]:
    error = np.asarray(error, dtype=np.float64)
    absolute = np.abs(error)
    if not len(error):
        return {"samples": 0}
    return {
        "samples": int(len(error)),
        "rmse_rad": float(np.sqrt(np.mean(error * error))),
        "mae_rad": float(np.mean(absolute)),
        "p95_abs_rad": float(np.quantile(absolute, 0.95)),
        "max_abs_rad": float(np.max(absolute)),
        "fraction_abs_error_below_0p005": float(np.mean(absolute < 0.005)),
    }


def evaluate(run_id: str, bag_path: Path, speed_range: tuple[float, float],
             steering_min: float, output: Path) -> dict[str, Any]:
    runs, source_audit = atlas._discover_run_series()
    validation = [row for row in runs
                  if row.run_id == run_id and row.split == "validation"]
    training = [row for row in runs if row.split == "train"]
    if len(validation) != 1 or len(training) < 2:
        raise ValueError("requires one clean validation capture and >=2 train runs")

    train_rows = [_rows(row, speed_range, steering_min) for row in training]
    train_rows = [row for row in train_rows if len(row["y"])]
    train_x = np.concatenate([row["x"] for row in train_rows])
    train_y = np.concatenate([row["y"] for row in train_rows])
    val_series = validation[0]
    val = _rows(val_series, speed_range, steering_min)
    if not len(val["y"]):
        raise ValueError("validation capture has no samples in the requested domain")

    # Learn only the correction to the empirically supported hybrid baseline.
    residual_model = ExtraTreesRegressor(
        n_estimators=240, max_depth=12, min_samples_leaf=4,
        max_features=1.0, n_jobs=-1, random_state=20261008,
    )
    hybrid_key = "hybrid_magnitude_decrease_or_limit_k_minus_1"
    train_hybrid = np.concatenate([row[hybrid_key] for row in train_rows])
    residual_model.fit(train_x, train_y - train_hybrid)
    residual_prediction = val[hybrid_key] + residual_model.predict(val["x"])
    current_feedback = val["x"][:, 0]
    command_k_minus_1 = val["x"][:, 2]
    command_k_minus_2 = val["x"][:, 3]
    # A newly observed command sign change may still be in the actuator's
    # second queue slot. If feedback remains on the old-command side, use the
    # older queued command for this transition only; otherwise use k-1.
    pending_reversal = (
        (command_k_minus_1 * command_k_minus_2 < 0.0)
        & (current_feedback * command_k_minus_2 > 0.0))
    pending_target = np.where(
        pending_reversal, command_k_minus_2, command_k_minus_1)
    pending_rate_limited = current_feedback + np.clip(
        pending_target - current_feedback, -3.2 * DT_S, 3.2 * DT_S)
    pending_prediction = np.where(
        np.abs(pending_target) < np.abs(current_feedback),
        pending_target, pending_rate_limited)
    predictions = {
        "hold_feedback": np.asarray([
            val_series.frames[k, 3] for k in val["frame"]], dtype=np.float32),
        **{name: val[name] for name in sorted(val)
           if name.startswith(("direct_command_", "rate_limit_", "hybrid_"))},
        "hybrid_plus_train_only_extra_trees_residual": residual_prediction,
        "hybrid_one_extra_queued_sample_on_pending_sign_reversal":
            pending_prediction,
    }
    errors = {name: pred - val["y"] for name, pred in predictions.items()}

    # The experiment's randomized probe labels provide an event-level, paired
    # score. Exclude rows crossing probe boundaries.
    probes, experiment_end = _probe_phases(bag_path)
    archive_path = atlas.ROOT / val_series.source
    with np.load(archive_path, allow_pickle=False) as archive:
        sample_times = np.asarray(archive["sample_time_ns"], dtype=np.int64)
    if sample_times.shape != (len(val_series.frames),):
        raise ValueError("validation timestamps do not align with admitted samples")
    times_k = sample_times[val["frame"]]
    times_next = sample_times[val["frame"] + 1]
    phase_k, labels = _phase_lookup(times_k, probes)
    phase_next, _ = _phase_lookup(times_next, probes)
    in_same_probe = (phase_k >= 0) & (phase_k == phase_next)
    groups: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list))
    condition_groups: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list))
    by_event_and_turn: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list)))
    by_event_cell: dict[str, dict[str, dict[str, dict[str, list[float]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    worst_probe_rows: list[dict[str, Any]] = []
    for idx in np.flatnonzero(in_same_probe):
        event_label = labels[int(phase_k[idx])]
        event, condition = _bucket(event_label)
        turn = "right" if "_turn-" in event_label else "left"
        condition_parts = condition.split("/")
        speed_key = condition_parts[1]
        angle_key = condition_parts[2]
        for name, error in errors.items():
            groups[event][name].append(float(error[idx]))
            condition_groups[condition][name].append(float(error[idx]))
            by_event_and_turn[event][turn][name].append(float(error[idx]))
            by_event_cell[event][speed_key][angle_key][name].append(
                float(error[idx]))
        worst_probe_rows.append({
            "absolute_learned_error_rad": float(abs(errors[
                "hybrid_plus_train_only_extra_trees_residual"][idx])),
            "absolute_hybrid_error_rad": float(abs(errors[
                hybrid_key][idx])),
            "absolute_pending_reversal_error_rad": float(abs(errors[
                "hybrid_one_extra_queued_sample_on_pending_sign_reversal"][idx])),
            "event_label": event_label,
            "event": event,
            "turn": turn,
            "speed_mps": float(val["speed"][idx]),
            "steering_feedback_k_rad": float(val["x"][idx, 0]),
            "steering_command_k_rad": float(val["x"][idx, 1]),
            "steering_command_k_minus_1_rad": float(val["x"][idx, 2]),
            "target_feedback_k_plus_1_rad": float(val["y"][idx]),
            "hybrid_prediction_rad": float(val[hybrid_key][idx]),
            "pending_reversal_prediction_rad": float(pending_prediction[idx]),
            "pending_reversal_extra_queue_selected": bool(
                pending_reversal[idx]),
            "hybrid_plus_extra_trees_prediction_rad": float(
                residual_prediction[idx]),
            "hybrid_signed_error_rad": float(errors[
                hybrid_key][idx]),
            "learned_signed_error_rad": float(errors[
                "hybrid_plus_train_only_extra_trees_residual"][idx]),
        })

    by_event = {
        event: {name: _metrics(np.asarray(values))
                for name, values in sorted(methods.items())}
        for event, methods in sorted(groups.items())
    }
    by_event_turn = {
        event: {
            turn: {name: _metrics(np.asarray(values))
                   for name, values in sorted(methods.items())}
            for turn, methods in sorted(turns.items())
        }
        for event, turns in sorted(by_event_and_turn.items())
    }
    by_event_speed_steering = {
        event: {
            speed: {
                angle: {name: _metrics(np.asarray(values))
                        for name, values in sorted(methods.items())}
                for angle, methods in sorted(angles.items())
            }
            for speed, angles in sorted(speeds.items())
        }
        for event, speeds in sorted(by_event_cell.items())
    }
    per_condition_rmse = {}
    for name in predictions:
        per_condition_rmse[name] = {
            condition: _metrics(np.asarray(values[name]))["rmse_rad"]
            for condition, values in sorted(condition_groups.items())
            if len(values[name]) >= 4
        }
    uncertainty = {}
    if hybrid_key in per_condition_rmse:
        names = [hybrid_key,
                 "hybrid_plus_train_only_extra_trees_residual"]
        common = sorted(set(per_condition_rmse[names[0]])
                        & set(per_condition_rmse[names[1]]))
        differences = np.asarray([
            per_condition_rmse[names[1]][condition]
            - per_condition_rmse[names[0]][condition]
            for condition in common], dtype=np.float64)
        if len(differences):
            rng = np.random.default_rng(20261008)
            samples = rng.choice(differences, size=(10000, len(differences)),
                                 replace=True).mean(axis=1)
            uncertainty = {
                "paired_probe_conditions": int(len(differences)),
                "mean_condition_rmse_delta_extra_trees_minus_hybrid_rad": float(
                    differences.mean()),
                "conditions_improved": int(np.count_nonzero(differences < 0.0)),
                "conditions_worsened": int(np.count_nonzero(differences > 0.0)),
                "bootstrap_95pct_ci_mean_delta_rad": [
                    float(np.quantile(samples, 0.025)),
                    float(np.quantile(samples, 0.975)),
                ],
            }
    comparison_specs = (
        ("hybrid_one_extra_queued_sample_on_pending_sign_reversal",
         "hybrid_magnitude_decrease_or_limit_k_minus_1"),
        ("hybrid_magnitude_decrease_or_limit_k_minus_1",
         "rate_limit_3p2_k_minus_1"),
        ("hybrid_magnitude_decrease_or_limit_k_minus_1",
         "hybrid_same_sign_release_or_limit_k_minus_1"),
        ("hybrid_magnitude_decrease_or_limit_k_minus_1",
         "direct_command_k_minus_1"),
        ("hybrid_plus_train_only_extra_trees_residual",
         "hybrid_magnitude_decrease_or_limit_k_minus_1"),
    )
    pairwise_condition_comparisons = {}
    rng = np.random.default_rng(20261008)
    for left, right in comparison_specs:
        common = sorted(set(per_condition_rmse[left])
                        & set(per_condition_rmse[right]))
        differences = np.asarray([
            per_condition_rmse[left][condition]
            - per_condition_rmse[right][condition]
            for condition in common], dtype=np.float64)
        if not len(differences):
            continue
        draws = rng.choice(differences, size=(10000, len(differences)),
                           replace=True).mean(axis=1)
        pairwise_condition_comparisons[f"{left}_minus_{right}"] = {
            "paired_conditions": int(len(differences)),
            "mean_condition_rmse_delta_rad": float(differences.mean()),
            "conditions_left_better": int(np.count_nonzero(differences < 0.0)),
            "conditions_left_worse": int(np.count_nonzero(differences > 0.0)),
            "bootstrap_95pct_ci_mean_delta_rad": [
                float(np.quantile(draws, 0.025)),
                float(np.quantile(draws, 0.975)),
            ],
        }

    result: dict[str, Any] = {
        "title": "Causal steering-actuator next-packet prediction",
        "validation_run_id": run_id,
        "bag": str(bag_path),
        "target": "measured steering_feedback[k+1]",
        "sample_period_ms": 25,
        "future_truth_or_sensor_used_as_input": False,
        "feature_names": list(FEATURE_NAMES),
        "training_runs": [row.run_id for row in training],
        "training_runs_with_domain_rows": [row.run_id for row in training
                                           if len(_rows(row, speed_range,
                                                        steering_min)["y"])],
        "training_samples": int(len(train_y)),
        "validation_samples_in_domain": int(len(val["y"])),
        "validation_samples_in_controlled_probe_phases": int(
            np.count_nonzero(in_same_probe)),
        "pending_sign_reversal_extra_queue_samples": int(
            np.count_nonzero(pending_reversal)),
        "domain": {"speed_mps": list(speed_range),
                   "absolute_physical_steering_rad": [steering_min, 0.525]},
        "simple_model_definitions": {
            "direct_command_k_minus_d": "command[k-d]",
            "rate_limit_3p2_k_minus_d": "feedback[k] + clip(command[k-d]-feedback[k], +/- 3.2*0.025)",
            "hybrid_magnitude_decrease_or_limit_k_minus_d": "direct command when abs(command[k-d]) < abs(feedback[k]), including sign reversal; otherwise 3.2 rad/s limited; d=0,1,2",
            "hybrid_same_sign_release_or_limit_k_minus_d": "direct command only on same-sign magnitude release; otherwise 3.2 rad/s limited; d=0,1,2",
            "hybrid_one_extra_queued_sample_on_pending_sign_reversal": (
                "use command[k-2] only when command[k-1] changed sign from "
                "command[k-2] and current feedback still has the old command "
                "sign; otherwise use command[k-1], then apply the measured "
                "magnitude-decrease/rate-limit rule"),
        },
        "model": {
            "family": "ExtraTrees residual correction to one-packet-delayed magnitude-decrease/rate-limit hybrid",
            "trees": 240,
            "max_depth": 12,
            "min_samples_leaf": 4,
            "train_split_only": True,
            "validation_capture_used_for_fit_or_selection": False,
        },
        "overall_in_domain": {name: _metrics(errors[name])
                              for name in predictions},
        "by_controlled_event": by_event,
        "paired_condition_uncertainty_vs_hybrid": uncertainty,
        "pairwise_condition_comparisons": pairwise_condition_comparisons,
        "by_event_and_turn_sign": by_event_turn,
        "by_event_speed_and_steering": by_event_speed_steering,
        "worst_controlled_probe_predictions": sorted(
            worst_probe_rows,
            key=lambda row: row["absolute_learned_error_rad"], reverse=True)[:40],
        "worst_pending_reversal_predictions": sorted(
            worst_probe_rows,
            key=lambda row: row["absolute_pending_reversal_error_rad"],
            reverse=True)[:40],
        "experiment_end": experiment_end,
        "source_audit": source_audit,
        "per_condition_rmse": per_condition_rmse,
        "notes": [
            "This is a high-steering 3-4 m/s actuator specialist evaluation, not a full-range actuator model.",
            "Command alignments k, k-1, and k-2 are compared explicitly; this is an empirical discrete-delay check, not an assumption that one delay is universally correct.",
            "A single validation capture is not independent-run uncertainty; randomized reset-isolated probe conditions are the paired units reported here.",
            "No model artifact is emitted unless a subsequent review elects to retain this candidate.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(output)
    for name, metrics in result["overall_in_domain"].items():
        print(f"{name}: n={metrics['samples']} RMSE={metrics['rmse_rad']:.6f} "
              f"p95={metrics['p95_abs_rad']:.6f} max={metrics['max_abs_rad']:.6f}")
    for event, methods in by_event.items():
        print(event, " ".join(
            f"{name}={values['rmse_rad']:.6f}/{values['p95_abs_rad']:.6f}/"
            f"{values['max_abs_rad']:.6f} (n={values['samples']})"
            for name, values in sorted(methods.items())))
    print("paired condition uncertainty:", json.dumps(uncertainty, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--speed-min", type=float, default=2.5)
    parser.add_argument("--speed-max", type=float, default=4.5)
    parser.add_argument("--steering-min", type=float, default=0.30)
    args = parser.parse_args()
    evaluate(args.run_id, args.bag,
             (args.speed_min, args.speed_max), args.steering_min, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
