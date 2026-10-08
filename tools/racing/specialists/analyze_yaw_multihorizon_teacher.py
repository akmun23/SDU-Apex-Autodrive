#!/usr/bin/env python3
"""Diagnose held-out errors from the sensor-only yaw teacher fits."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np

try:
    import train_yaw_multihorizon_teacher as teacher
except ModuleNotFoundError:
    from tools.racing.specialists import train_yaw_multihorizon_teacher as teacher


DEFAULT_OUTPUT = teacher.DEFAULT_OUTPUT
# Refine the low-speed tail because the 0–2 m/s audit cell had both sparse
# support and the largest high-steer residuals; keep racing-speed bands at 2 m/s.
SPEED_EDGES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0,
               6.0, 8.0, 10.0, 12.0)
STEERING_EDGES = (0.0, 0.1, 0.2, 0.35, 0.525, 1.0)
STEERING_GAP_EDGES = (0.0, 0.025, 0.05, 0.1, 0.2, 1.0)
WHEEL_BODY_GAP_EDGES = (0.0, 0.1, 0.25, 0.5, 1.0, 5.0)
THROTTLE_GAP_EDGES = (0.0, 0.025, 0.05, 0.1, 0.25, 1.0)
SIGNED_STEERING_EDGES = (-0.525, -0.35, -0.2, -0.1, 0.0,
                         0.1, 0.2, 0.35, 0.525)
STEERING_COMMAND_STEP_EDGES = (0.0, 0.001, 0.005, 0.02, 0.1, 0.5, 1.0)
THROTTLE_COMMAND_STEP_EDGES = (0.0, 0.001, 0.005, 0.02, 0.05, 0.15, 0.5)
ABS_LATERAL_ACCEL_EDGES = (0.0, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 20.0, 100.0)
ABS_ROLL_EDGES = (0.0, 0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 2.0)
ABS_ROLL_RATE_EDGES = (0.0, 0.025, 0.05, 0.1, 0.2, 0.5, 1.0, 5.0)
ABS_SIDESLIP_EDGES = (0.0, 0.01, 0.025, 0.05, 0.1, 0.2, 0.5, np.pi)


def _capture_sets(fit_report: dict[str, Any],
                  include_new_validation_runs: bool = False):
    series, discovery_audit = teacher._discover_series_with_safe_mixed_archives()
    fitted_train_ids = set(map(str, fit_report.get("training_runs", [])))
    fitted_validation_ids = set(map(str, fit_report.get("validation_runs", [])))
    by_id = {row.run_id: row for row in series}
    missing_train = sorted(fitted_train_ids - by_id.keys())
    missing_validation = sorted(fitted_validation_ids - by_id.keys())
    if missing_train or missing_validation:
        raise RuntimeError(
            "fit source runs are unavailable: "
            f"train={missing_train[:8]}, validation={missing_validation[:8]}")
    if any(by_id[run_id].split != "train" for run_id in fitted_train_ids):
        raise RuntimeError("fit report training runs no longer have train split")
    if any(by_id[run_id].split != "validation"
           for run_id in fitted_validation_ids):
        raise RuntimeError("fit report validation runs no longer have validation split")
    extra_validation_ids = sorted(
        row.run_id for row in series
        if row.split == "validation" and row.run_id not in fitted_validation_ids)
    validation_ids = fitted_validation_ids | (
        set(extra_validation_ids) if include_new_validation_runs else set())
    selected_ids = fitted_train_ids | validation_ids
    selected_series = [by_id[run_id] for run_id in sorted(selected_ids)]
    minimum = max(teacher.HISTORY_LAGS) + max(teacher.HORIZONS) + 1
    eligible = [row for row in selected_series
                if np.any((row.bounds[:, 1] - row.bounds[:, 0]) >= minimum)]
    captures = [teacher._read_run(row) for row in eligible]
    capture_by_id = {row.run_id: row for row in captures}
    missing_eligible = sorted(selected_ids - capture_by_id.keys())
    if missing_eligible:
        raise RuntimeError(
            "fit runs lack a usable continuous sequence: "
            f"{missing_eligible[:8]}")
    audit = {
        "fit_training_runs": sorted(fitted_train_ids),
        "fit_validation_runs": sorted(fitted_validation_ids),
        "additional_validation_runs": sorted(validation_ids - fitted_validation_ids),
        "current_discovery_audit": discovery_audit,
    }
    return ([capture_by_id[run_id] for run_id in sorted(fitted_train_ids)],
            [capture_by_id[run_id] for run_id in sorted(validation_ids)], audit)


def _group_metrics(errors: np.ndarray, values: np.ndarray,
                   edges: tuple[float, ...]) -> dict[str, dict[str, Any]]:
    result = {}
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (values >= low) & (values < high)
        result[f"{low:g}..{high:g}"] = teacher._metrics(errors[mask])
    return result


def _paired_ci(deltas: np.ndarray, seed: int) -> list[float]:
    if not len(deltas):
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    samples = rng.choice(deltas, size=(10_000, len(deltas)), replace=True).mean(1)
    return [float(value) for value in np.quantile(samples, (0.025, 0.975))]


def _bin_labels(values: np.ndarray, edges: tuple[float, ...]) -> np.ndarray:
    indices = np.digitize(values, edges, right=False) - 1
    labels = np.empty(len(values), dtype=object)
    for index in np.unique(indices):
        low = edges[int(index)] if 0 <= index < len(edges) - 1 else edges[-1]
        high = (edges[int(index) + 1]
                if 0 <= index < len(edges) - 1 else float("inf"))
        labels[indices == index] = f"{low:g}..{high:g}"
    return labels


def _regime_values(capture: teacher.CaptureArrays,
                   frame_indices: np.ndarray,
                   horizon: int) -> dict[str, np.ndarray]:
    rows = np.asarray(frame_indices, dtype=np.int64)
    final_command_row = rows + horizon - 1
    speed = np.hypot(capture.rigid[rows, 7], capture.rigid[rows, 8])
    body_sideslip = np.abs(np.arctan2(
        capture.rigid[rows, 8], capture.rigid[rows, 7]))
    abs_lateral_accel = np.abs(capture.sensors[rows, 5])
    abs_roll = np.abs(capture.attitude[rows, 0])
    abs_roll_rate = np.abs(capture.attitude[rows, 2])
    signed_steering = capture.sensors[rows, 0]
    steering = np.abs(signed_steering)
    steering_gap = np.abs(capture.sensors[rows, 7] - capture.sensors[rows, 0])
    throttle_gap = np.abs(capture.sensors[rows, 8] - capture.sensors[rows, 1])
    rear_mean = 0.5 * (capture.sensors[rows, 2] + capture.sensors[rows, 3])
    wheel_body_gap = np.abs(rear_mean - speed)
    steering_command = capture.sensors[:, 7]
    throttle_command = capture.sensors[:, 8]
    if horizon > 1:
        max_steering_command_step = np.maximum.reduce([
            np.abs(steering_command[rows + offset + 1]
                   - steering_command[rows + offset])
            for offset in range(horizon - 1)
        ])
        max_throttle_command_step = np.maximum.reduce([
            np.abs(throttle_command[rows + offset + 1]
                   - throttle_command[rows + offset])
            for offset in range(horizon - 1)
        ])
    else:
        max_steering_command_step = np.zeros(len(rows), dtype=np.float32)
        max_throttle_command_step = np.zeros(len(rows), dtype=np.float32)
    # This realized response is an offline diagnosis label, never a model input.
    steering_feedback = capture.sensors[:, 0]
    max_steering_feedback_step = (
        np.maximum.reduce([
            np.abs(steering_feedback[rows + offset + 1]
                   - steering_feedback[rows + offset])
            for offset in range(horizon - 1)
        ]) if horizon > 1 else np.zeros(len(rows), dtype=np.float32))
    sequence_index = np.searchsorted(capture.bounds[:, 0], rows,
                                     side="right") - 1
    return {
        "speed": speed,
        "abs_lateral_accel": abs_lateral_accel,
        "abs_roll": abs_roll,
        "abs_roll_rate": abs_roll_rate,
        # Ground-truth body sideslip is diagnosis-only; it is not a model input.
        "abs_sideslip": body_sideslip,
        "steering": steering,
        "signed_steering": signed_steering,
        # Recorded future command is known over this direct prediction horizon.
        "future_command_steering": steering_command[final_command_row],
        "steering_gap": steering_gap,
        "throttle_gap": throttle_gap,
        "wheel_body_gap": wheel_body_gap,
        "steering_command_step": max_steering_command_step,
        "throttle_command_step": max_throttle_command_step,
        "steering_feedback_step": max_steering_feedback_step,
        "future_command_steering": steering_command[final_command_row],
        "sequence_index": sequence_index,
    }


def _joint_regime_table(captures: list[teacher.CaptureArrays],
                        horizon: int, history_lags: tuple[int, ...],
                        errors_by_run: dict[str, tuple[np.ndarray, np.ndarray]]
                        | None = None) -> dict[str, dict[str, Any]]:
    """Count independent support and held-out errors in joint regimes.

    The regime labels use simulator-truth speed only for offline diagnosis;
    none of these labels are model inputs. Support counts distinguish dense
    adjacent packets from independent runs/reset-isolated sequences.
    """
    specs = {
        "speed_x_steering":
            (("speed", "steering"), (SPEED_EDGES, STEERING_EDGES)),
        "speed_x_signed_steering":
            (("speed", "signed_steering"),
             (SPEED_EDGES, SIGNED_STEERING_EDGES)),
        "speed_x_steering_x_future_command_steering":
            (("speed", "steering", "future_command_steering"),
             (SPEED_EDGES, STEERING_EDGES, SIGNED_STEERING_EDGES)),
        "speed_x_steering_x_steering_gap":
            (("speed", "steering", "steering_gap"),
             (SPEED_EDGES, STEERING_EDGES, STEERING_GAP_EDGES)),
        "speed_x_steering_x_wheel_body_gap":
            (("speed", "steering", "wheel_body_gap"),
             (SPEED_EDGES, STEERING_EDGES, WHEEL_BODY_GAP_EDGES)),
        "speed_x_steering_x_throttle_gap":
            (("speed", "steering", "throttle_gap"),
             (SPEED_EDGES, STEERING_EDGES, THROTTLE_GAP_EDGES)),
        "speed_x_steering_x_command_steering_step":
            (("speed", "steering", "steering_command_step"),
             (SPEED_EDGES, STEERING_EDGES, STEERING_COMMAND_STEP_EDGES)),
        "speed_x_steering_x_command_throttle_step":
            (("speed", "steering", "throttle_command_step"),
             (SPEED_EDGES, STEERING_EDGES, THROTTLE_COMMAND_STEP_EDGES)),
        "speed_x_steering_x_steering_feedback_step":
            (("speed", "steering", "steering_feedback_step"),
             (SPEED_EDGES, STEERING_EDGES, STEERING_COMMAND_STEP_EDGES)),
        "speed_x_steering_x_abs_lateral_accel":
            (("speed", "steering", "abs_lateral_accel"),
             (SPEED_EDGES, STEERING_EDGES, ABS_LATERAL_ACCEL_EDGES)),
        "speed_x_steering_x_abs_roll":
            (("speed", "steering", "abs_roll"),
             (SPEED_EDGES, STEERING_EDGES, ABS_ROLL_EDGES)),
        "speed_x_steering_x_abs_roll_rate":
            (("speed", "steering", "abs_roll_rate"),
             (SPEED_EDGES, STEERING_EDGES, ABS_ROLL_RATE_EDGES)),
        "speed_x_steering_x_abs_sideslip":
            (("speed", "steering", "abs_sideslip"),
             (SPEED_EDGES, STEERING_EDGES, ABS_SIDESLIP_EDGES)),
    }
    accumulators: dict[str, dict[str, dict[str, Any]]] = {
        name: {} for name in specs
    }
    for capture in captures:
        _, targets, meta = teacher._build_examples(
            capture, horizon, history_lags)
        frames = meta["frame_index"]
        if not len(frames):
            continue
        regimes = _regime_values(capture, frames, horizon)
        errors = None
        if errors_by_run is not None:
            scored_frames, scored_errors = errors_by_run[capture.run_id]
            if not np.array_equal(frames, scored_frames):
                raise RuntimeError(f"diagnostic frame mismatch: {capture.run_id}")
            errors = scored_errors
        for table_name, (regime_names, edge_sets) in specs.items():
            labels = tuple(_bin_labels(regimes[name], edge)
                           for name, edge in zip(regime_names, edge_sets))
            keys = np.asarray(["|".join(map(str, values))
                               for values in zip(*labels)], dtype=object)
            for key in np.unique(keys):
                mask = keys == key
                row = accumulators[table_name].setdefault(str(key), {
                    "samples": 0,
                    "runs": set(),
                    "sequences": set(),
                    "errors": [],
                    "errors_by_run": defaultdict(list),
                    "targets": 0,
                })
                row["samples"] += int(mask.sum())
                row["runs"].add(capture.run_id)
                row["sequences"].update(
                    f"{capture.run_id}:{index}"
                    for index in np.unique(regimes["sequence_index"][mask]))
                if errors is not None:
                    row["errors"].append(errors[mask])
                    row["errors_by_run"][capture.run_id].append(errors[mask])
                    row["targets"] += int(np.count_nonzero(
                        np.abs(errors[mask]) > 0.1))

    result: dict[str, dict[str, Any]] = {}
    for table_name, rows in accumulators.items():
        result[table_name] = {}
        for key, row in sorted(rows.items()):
            summary: dict[str, Any] = {
                "samples": row["samples"],
                "independent_runs": len(row["runs"]),
                "independent_sequences": len(row["sequences"]),
            }
            if errors_by_run is not None:
                errors = (np.concatenate(row["errors"])
                          if row["errors"] else np.empty(0))
                summary.update(teacher._metrics(errors))
                summary["count_abs_error_over_0p1"] = row["targets"]
                summary["per_run"] = {
                    run_id: teacher._metrics(np.concatenate(parts))
                    for run_id, parts in sorted(row["errors_by_run"].items())
                }
            result[table_name][key] = summary
    return result


def diagnose(output_dir: Path,
             diagnosis_output: Path | None = None,
             include_new_validation_runs: bool = False
             ) -> dict[str, Any]:
    report_path = output_dir / "yaw_multihorizon_teacher_report.json"
    fit_report = json.loads(report_path.read_text(encoding="utf-8"))
    if fit_report.get("status") != "complete":
        raise RuntimeError("teacher fit must be complete before diagnosis")
    train, validation, audit = _capture_sets(
        fit_report, include_new_validation_runs)
    results: dict[str, Any] = {
        "title": "Held-out sensor-only yaw-teacher error diagnosis",
        "fit_report": report_path.name,
        "source_audit": audit,
        "diagnostic_regime_labels": {
            "future_command_steering": (
                "recorded candidate command over the scored horizon"),
            "steering_feedback_step": (
                "realized future actuator response; diagnosis only, never a feature"),
            "abs_lateral_accel": "current IMU lateral acceleration magnitude",
            "abs_roll": "current IMU roll magnitude",
            "abs_roll_rate": "current IMU roll-rate magnitude",
            "abs_sideslip": (
                "current simulator-truth body sideslip; diagnosis only, never a feature"),
        },
        "variants": {},
        "training_input_support": {},
    }

    train_speed, train_steer_gap, train_throttle_gap = [], [], []
    for capture in train:
        for begin, end in capture.bounds:
            rows = np.arange(int(begin), int(end))
            train_speed.append(np.hypot(capture.rigid[rows, 7],
                                        capture.rigid[rows, 8]))
            train_steer_gap.append(np.abs(capture.sensors[rows, 7]
                                          - capture.sensors[rows, 0]))
            train_throttle_gap.append(np.abs(capture.sensors[rows, 8]
                                             - capture.sensors[rows, 1]))
    for name, parts in (("speed_mps", train_speed),
                        ("abs_steering_command_feedback_gap_rad", train_steer_gap),
                        ("abs_throttle_command_feedback_gap", train_throttle_gap)):
        values = np.concatenate(parts)
        results["training_input_support"][name] = {
            "samples": int(len(values)),
            "min": float(values.min()), "p50": float(np.median(values)),
            "p95": float(np.quantile(values, 0.95)), "max": float(values.max()),
        }

    for variant, lags in teacher.HISTORY_VARIANTS.items():
        results["variants"][variant] = {}
        for horizon in teacher.HORIZONS:
            path = output_dir / f"yaw_teacher_{variant}_h{horizon:02d}.joblib"
            model = joblib.load(path)
            per_run: dict[str, dict[str, Any]] = {}
            sample_errors, sample_truth, sample_imu = [], [], []
            sample_speed, sample_steer, sample_steer_gap = [], [], []
            sample_throttle_gap, sample_frame_index, sample_run_id = [], [], []
            error_by_run: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            top_rows = []
            for capture in validation:
                x, y, meta = teacher._build_examples(capture, horizon, lags)
                if not len(y):
                    continue
                pred = model.predict(x).astype(np.float64)
                error = pred - y.astype(np.float64)
                ids = meta["run_id"]
                for run_id in (capture.run_id,):
                    mask = ids == run_id
                    per_run[run_id] = {
                        "model": teacher._metrics(error[mask]),
                        "imu_persistence": teacher._metrics(
                            meta["imu_yaw_rate_rps"][mask] - y[mask]),
                    }
                frame = meta["frame_index"]
                error_by_run[capture.run_id] = (frame.copy(), error.copy())
                sample_errors.append(error)
                sample_truth.append(y.astype(np.float64))
                sample_imu.append(meta["imu_yaw_rate_rps"].astype(np.float64))
                sample_speed.append(meta["speed_mps"].astype(np.float64))
                sample_steer.append(meta["steering_rad"].astype(np.float64))
                sample_steer_gap.append(np.abs(capture.sensors[frame, 7]
                                               - capture.sensors[frame, 0]))
                sample_throttle_gap.append(np.abs(capture.sensors[frame, 8]
                                                  - capture.sensors[frame, 1]))
                sample_frame_index.append(frame)
                sample_run_id.append(ids)
                count = min(10, len(error))
                worst = np.argpartition(np.abs(error), -count)[-count:]
                for i in worst:
                    row = int(frame[i])
                    truth = capture.rigid[row]
                    sensors = capture.sensors[row]
                    attitude = capture.attitude[row]
                    wheel_mean = 0.5 * float(sensors[2] + sensors[3])
                    body_speed = float(np.hypot(truth[7], truth[8]))
                    top_rows.append({
                        "run_id": capture.run_id,
                        "frame_index_in_exported_run": row,
                        "horizon_ms": int(round(horizon * teacher.atlas.DT_S * 1000)),
                        "yaw_truth_radps": float(y[i]),
                        "yaw_prediction_radps": float(pred[i]),
                        "yaw_error_radps": float(error[i]),
                        "imu_yaw_now_radps": float(sensors[6]),
                        "speed_truth_mps_for_diagnosis_only": body_speed,
                        "physical_steering_rad": float(sensors[0]),
                        "steering_command_rad": float(sensors[7]),
                        "steering_command_feedback_gap_rad": float(sensors[7] - sensors[0]),
                        "throttle_feedback": float(sensors[1]),
                        "throttle_command": float(sensors[8]),
                        "throttle_command_feedback_gap": float(sensors[8] - sensors[1]),
                        "rear_wheel_mean_mps": wheel_mean,
                        "wheel_body_speed_mismatch_truth_for_diagnosis_only_mps":
                            wheel_mean - body_speed,
                        "imu_ax_mps2": float(sensors[4]),
                        "imu_ay_mps2": float(sensors[5]),
                        "imu_roll_rad": float(attitude[0]),
                        "imu_roll_rate_rps": float(attitude[2]),
                    })
            errors = np.concatenate(sample_errors)
            truth = np.concatenate(sample_truth)
            imu = np.concatenate(sample_imu)
            speed = np.concatenate(sample_speed)
            steering = np.concatenate(sample_steer)
            steering_gap = np.concatenate(sample_steer_gap)
            throttle_gap = np.concatenate(sample_throttle_gap)
            run_id = np.concatenate(sample_run_id)
            run_names = sorted(per_run)
            model_run_rmse = np.asarray([
                per_run[name]["model"]["rmse_radps"] for name in run_names])
            baseline_run_rmse = np.asarray([
                per_run[name]["imu_persistence"]["rmse_radps"] for name in run_names])
            deltas = model_run_rmse - baseline_run_rmse
            top_rows.sort(key=lambda row: abs(row["yaw_error_radps"]), reverse=True)
            results["variants"][variant][str(horizon)] = {
                "samples": int(len(errors)),
                "model": teacher._metrics(errors),
                "imu_persistence": teacher._metrics(imu - truth),
                "paired_run_rmse_delta_model_minus_imu": {
                    "mean": float(deltas.mean()),
                    "bootstrap_95pct_ci": _paired_ci(deltas, 20261007 + horizon),
                    "runs_model_better": int(np.count_nonzero(deltas < 0.0)),
                    "runs_compared": int(len(deltas)),
                },
                "by_speed_band": _group_metrics(
                    errors, speed, (0, 2, 4, 6, 8, 10, 12, 100)),
                "by_abs_steering_band": _group_metrics(
                    errors, np.abs(steering), (0, 0.1, 0.2, 0.35, 0.525, 1)),
                "by_steering_command_feedback_gap": _group_metrics(
                    errors, steering_gap, (0, 0.025, 0.05, 0.10, 1)),
                "by_throttle_command_feedback_gap": _group_metrics(
                    errors, throttle_gap, (0, 0.025, 0.05, 0.10, 1)),
                "per_run": per_run,
                "worst_heldout_samples": top_rows[:50],
                "joint_validation_error_by_regime": _joint_regime_table(
                    validation, horizon, lags, error_by_run),
                "joint_training_support_by_regime": _joint_regime_table(
                    train, horizon, lags),
            }
            del model

    out_path = diagnosis_output or (
        output_dir / "yaw_multihorizon_teacher_error_diagnosis.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        results, indent=2, sort_keys=True, default=teacher._json_default
    ) + "\n", encoding="utf-8")
    print(out_path)
    for variant, horizons in results["variants"].items():
        for horizon, row in horizons.items():
            metric = row["model"]
            print(f"{variant:12s} {int(horizon)*25:4d} ms: "
                  f"RMSE={metric['rmse_radps']:.4f} "
                  f"p95={metric['p95_abs_radps']:.4f} "
                  f"max={metric['max_abs_radps']:.4f} "
                  f"<0.1={metric['fraction_abs_error_below_0p1']:.3%}")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--diagnosis-output", type=Path,
        help="write the expanded audit separately without replacing the fit's existing report")
    parser.add_argument(
        "--include-new-validation-runs", action="store_true",
        help="also score clean validation runs added after this model was fitted; "
             "training support remains frozen to the fit report")
    args = parser.parse_args()
    diagnose(args.output_dir, args.diagnosis_output,
             args.include_new_validation_runs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
