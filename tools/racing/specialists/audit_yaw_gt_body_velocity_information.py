#!/usr/bin/env python3
"""Test whether GT body-motion state explains targeted yaw residuals offline.

The audited regimes are high-steer 2-packet reversal and unwind packet classes
present in the held-out capture.
The normal candidate uses only causal sensor/command/timing features. A
separate diagnostic adds current simulator-truth body-frame COM (u, v) solely
to measure whether unobserved speed/sideslip could explain residuals. GT is
never used as an inference input or selector. Training captures are used for
fitting and the whole-run r07 capture is scored once; test splits stay sealed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

try:
    import evaluate_yaw_packet_response_conditioned_models as response
    import fit_sensor_only_yaw_regime_atlas as atlas
    import train_yaw_multihorizon_teacher as teacher
except ModuleNotFoundError:
    from tools.racing.specialists import evaluate_yaw_packet_response_conditioned_models as response
    from tools.racing.specialists import fit_sensor_only_yaw_regime_atlas as atlas
    from tools.racing.specialists import train_yaw_multihorizon_teacher as teacher


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = (ROOT / "live_runs/racing_model_diagnostics_20261008/"
          "yaw_large_error_audit_v1/gt_body_velocity_information_r09_two_packet.json")
VALIDATION_RUN_ID = "openplane_yaw_error_packet_phase_validation_r09_20261008"
# Three-packet and all other non-two-packet rows are invalid for this study.
TARGET_GROUPS = ((2, "reversal"), (2, "unwind"))
MODEL = {
    "n_estimators": 240,
    "max_depth": 14,
    "min_samples_leaf": 4,
    "max_features": 0.9,
    "random_state": 20261008,
    "n_jobs": 4,
}


def _gt_body_state_rows(series: Any) -> np.ndarray:
    """Create GT-only current body-motion diagnostics on the 25-ms grid."""
    run = atlas._read_sensor_run(
        series, history_lags=response.EXPANDED_HISTORY_LAGS,
        exact_packet_imu=True)
    values: list[tuple[float, ...]] = []
    for sensors, sensor_valid, attitude, attitude_valid, rigid in run["sequences"]:
        if (not np.isfinite(sensors).all() or not np.isfinite(attitude).all()
                or not np.isfinite(rigid).all()):
            continue
        for k in range(max(response.EXPANDED_HISTORY_LAGS), len(sensors) - 1):
            if (not all(sensor_valid[k - lag]
                        and attitude_valid[k - lag]
                        for lag in response.EXPANDED_HISTORY_LAGS)):
                continue
            wheel_mean = 0.5 * float(sensors[k, 2] + sensors[k, 3])
            speed_cell = int(wheel_mean // atlas.SPEED_BIN_MPS)
            steering = float(sensors[k, 0])
            if (speed_cell < 0 or speed_cell >= len(atlas.SPEED_CENTERS)
                    or not np.isfinite(steering)
                    or steering < atlas.STEERING_CENTERS[0] - 0.0125
                    or steering > atlas.STEERING_CENTERS[-1] + 0.0125):
                continue
            # GT-only upper-bound probes: current body velocity, finite-difference
            # body acceleration on the assumed 25-ms packet grid, sideslip angle,
            # and rear-wheel/body longitudinal-speed mismatch.
            u, v = float(rigid[k, 7]), float(rigid[k, 8])
            u_prev, v_prev = float(rigid[k - 1, 7]), float(rigid[k - 1, 8])
            values.append((
                u, v,
                (u - u_prev) / atlas.DT_S,
                (v - v_prev) / atlas.DT_S,
                float(np.arctan2(v, u)),
                wheel_mean - u,
            ))
    return np.asarray(values, dtype=np.float32)


def _phase_rows(rows: dict[str, Any], phases: dict[str, dict[str, Any]],
                step: int, event: str) -> tuple[np.ndarray, np.ndarray]:
    selected: list[int] = []
    phase_ids: list[int] = []
    for phase_index, phase in enumerate(phases.values()):
        if (int(phase["true_steps"]) != step
                or phase["event"] != event):
            continue
        indices = np.asarray(phase["indices"], dtype=np.int64)
        indices = indices[rows["event"][indices].astype(str) == event]
        selected.extend(indices.tolist())
        phase_ids.extend([phase_index] * len(indices))
    return np.asarray(selected, dtype=np.int64), np.asarray(phase_ids, dtype=np.int64)


def _fit_score(train_x: np.ndarray, train_y: np.ndarray,
               validation_x: np.ndarray, validation_y: np.ndarray
               ) -> dict[str, Any]:
    model = ExtraTreesRegressor(**MODEL).fit(train_x, train_y)
    prediction = model.predict(validation_x)
    return response._metric(prediction - validation_y)


def run(output: Path = OUTPUT) -> dict[str, Any]:
    admitted, source_audit = teacher._discover_series_with_safe_mixed_archives()
    collected, phase_audits = response._collect_exact_two_packet(admitted)
    rows_by_run = collected["rows"]
    phases_by_run = collected["phases"]
    gt_body_state: dict[str, np.ndarray] = {}
    selected_runs = [row for row in admitted
                     if row.run_id in response.TIMING_SUFFIX]
    for series in selected_runs:
        values = _gt_body_state_rows(series)
        if len(values) != len(rows_by_run[series.run_id]["residual"]):
            raise ValueError(f"{series.run_id}: GT body state/feature rows misalign")
        gt_body_state[series.run_id] = values

    train_ids = sorted(run_id for run_id in response.TIMING_SUFFIX
                       if "_train_" in run_id)
    validation_id = VALIDATION_RUN_ID
    if validation_id not in response.TIMING_SUFFIX:
        raise ValueError(f"unregistered validation run: {validation_id}")
    reports: dict[str, Any] = {}

    for step, event in TARGET_GROUPS:
        group_name = f"{event}/{step}_packet"
        group_report: dict[str, Any] = {
            "training_rows_by_run": {},
            "leave_one_training_capture_out": {},
            "held_out_validation": {},
        }

        def combined(run_ids: list[str]) -> tuple[np.ndarray, ...]:
            features, targets, truths, phase_labels = [], [], [], []
            for run_id in run_ids:
                rows = rows_by_run[run_id]
                phases = phases_by_run[run_id]
                indices, phase_ids = _phase_rows(rows, phases, step, event)
                group_report["training_rows_by_run"][run_id] = {
                    "rows": int(len(indices)),
                    "phases": int(len(np.unique(phase_ids))),
                }
                if not len(indices):
                    continue
                features.append(rows["x_short_timed"][indices])
                targets.append(rows["residual"][indices])
                truths.append(gt_body_state[run_id][indices])
                phase_labels.append(np.asarray(
                    [f"{run_id}:{phase_id}" for phase_id in phase_ids],
                    dtype="U160"))
            if not features:
                raise ValueError(f"no training rows for {group_name}")
            return tuple(np.concatenate(part, axis=0) for part in
                         (features, targets, truths, phase_labels))

        for held_out in train_ids:
            fit_ids = [run_id for run_id in train_ids if run_id != held_out]
            train_x, train_y, train_state, _ = combined(fit_ids)
            val_rows = rows_by_run[held_out]
            val_indices, _ = _phase_rows(
                val_rows, phases_by_run[held_out], step, event)
            if not len(val_indices):
                continue
            val_x = val_rows["x_short_timed"][val_indices]
            val_y = val_rows["residual"][val_indices]
            val_state = gt_body_state[held_out][val_indices]
            group_report["leave_one_training_capture_out"][held_out] = {
                "fit_capture": fit_ids,
                "fit_rows": int(len(train_y)),
                "validation_rows": int(len(val_y)),
                "legal_sensor_only": _fit_score(train_x, train_y, val_x, val_y),
                "plus_forbidden_gt_body_u_v_diagnostic": _fit_score(
                    np.column_stack((train_x, train_state[:, :2])), train_y,
                    np.column_stack((val_x, val_state[:, :2])), val_y),
                "plus_forbidden_gt_body_dynamics_diagnostic": _fit_score(
                    np.column_stack((train_x, train_state)), train_y,
                    np.column_stack((val_x, val_state)), val_y),
            }

        train_x, train_y, train_state, _ = combined(train_ids)
        val_rows = rows_by_run[validation_id]
        val_indices, _ = _phase_rows(
            val_rows, phases_by_run[validation_id], step, event)
        if not len(val_indices):
            group_report["held_out_validation"] = {
                "run_id": validation_id,
                "validation_rows": 0,
                "status": "no held-out phases realized this event/packet-response class",
            }
            reports[group_name] = group_report
            continue
        val_x = val_rows["x_short_timed"][val_indices]
        val_y = val_rows["residual"][val_indices]
        val_state = gt_body_state[validation_id][val_indices]
        group_report["held_out_validation"] = {
            "fit_captures": train_ids,
            "fit_rows": int(len(train_y)),
            "fit_phase_count": int(sum(
                group_report["training_rows_by_run"][run_id]["phases"]
                for run_id in train_ids)),
            "validation_rows": int(len(val_y)),
            "legal_sensor_only": _fit_score(train_x, train_y, val_x, val_y),
            "plus_forbidden_gt_body_u_v_diagnostic": _fit_score(
                np.column_stack((train_x, train_state[:, :2])), train_y,
                np.column_stack((val_x, val_state[:, :2])), val_y),
            "plus_forbidden_gt_body_dynamics_diagnostic": _fit_score(
                np.column_stack((train_x, train_state)), train_y,
                np.column_stack((val_x, val_state)), val_y),
            "truth_scope": "GT body velocity, finite-difference acceleration, sideslip, and wheel/body mismatch are forbidden predictor inputs; this is an information upper-bound diagnostic only.",
        }
        reports[group_name] = group_report

    result = {
        "title": "Do unobserved body speed and sideslip explain the two largest yaw-error groups?",
        "status": "offline diagnostic only; no runtime model changed",
        "target": "next simulator-truth body-frame yaw rate minus current exact-source IMU body yaw rate",
        "split_policy": {
            "training_runs": train_ids,
            "held_out_validation_run": validation_id,
            "test_and_final_test_opened": False,
            "whole_capture_validation": True,
        },
        "input_comparison": {
            "baseline": "100-ms causal sensor/command history, current derived sensor states, and legal command/feedback timestamp features",
            "diagnostic_only_additions": [
                "simulator_truth_body_u_com_mps",
                "simulator_truth_body_v_com_mps",
                "finite_difference_gt_body_acceleration_mps2",
                "gt_sideslip_angle_rad",
                "rear_wheel_mean_minus_gt_body_u_mps",
            ],
            "gt_additions_never_used_for_inference_or_routing": True,
            "model": MODEL,
        },
        "phase_audits": phase_audits,
        "source_audit": source_audit,
        "groups": reports,
        "interpretation": "A held-out gain from GT body-motion state would show that unobserved physical state contains missing explanatory information; neither result authorizes GT runtime input. All fit/score rows are exact-two-packet phases only.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True,
                                 default=atlas._json_value) + "\n",
                      encoding="utf-8")
    print(json.dumps({
        "output": str(output.relative_to(ROOT)),
        "groups": {key: value["held_out_validation"] for key, value in reports.items()},
        "leave_one_training_capture_out": {
            key: value["leave_one_training_capture_out"]
            for key, value in reports.items()},
    }, indent=2))
    return result


if __name__ == "__main__":
    run()
