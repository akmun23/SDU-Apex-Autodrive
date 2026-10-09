#!/usr/bin/env python3
"""Fit/evaluate a narrow two-packet high-steer unwind yaw-response candidate.

This is an offline research artifact, not a runtime node. Ground-truth yaw is
used only to fit and score the one-step response. Bridge packet timing is used
only to select/stratify the two-packet training/validation examples; it is not
part of the candidate's inference inputs. The candidate explicitly assumes
that the command-to-feedback transition takes exactly two packets. Every
other response count is invalid and excluded from fitting and evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

try:
    import fit_sensor_only_yaw_regime_atlas as atlas
    import train_yaw_multihorizon_teacher as teacher
    from score_yaw_atlas_transition_events import (
        _phase_lookup, _probe_phases)
except ModuleNotFoundError:
    from tools.racing.specialists import fit_sensor_only_yaw_regime_atlas as atlas
    from tools.racing.specialists import train_yaw_multihorizon_teacher as teacher
    from tools.racing.specialists.score_yaw_atlas_transition_events import (
        _phase_lookup, _probe_phases)


ROOT = Path(__file__).resolve().parents[3]
BASELINE_DIR = (ROOT / "live_runs/racing_model_diagnostics_20261008/"
                "sensor_only_yaw_regime_atlas_command_intent_v2")
DEFAULT_OUTPUT = (ROOT / "live_runs/racing_model_diagnostics_20261008/"
                  "yaw_release_decay_two_packet_v1")
TARGET_RUNS = {
    "openplane_yaw_error_highsteer_reversal_train_r01_20261008",
    "openplane_yaw_error_highsteer_reversal_train_r02_20261008",
    "openplane_yaw_error_highsteer_reversal_validation_r03_20261008",
}
WHEEL_SPEED_CELL = 8  # 4.0 <= mean rear-wheel surface speed < 4.5 m/s.
STEERING_CELLS = (4, 38)  # Mirrored centers -0.425 and +0.425 rad.
EVENT = "unwind"
HISTORY_LAGS = (0, 1, 2, 4)
MIN_COMMAND_AGE_S = 0.020
MAX_COMMAND_AGE_S = 0.030
MIN_FEEDBACK_AGE_S = 0.250


def _trigger(x: np.ndarray, names: list[str]) -> np.ndarray:
    index = {name: position for position, name in enumerate(names)}
    steering = x[:, index["steering_feedback_rad_lag0ms"]]
    command = x[:, index["steering_command_rad_lag0ms"]]
    yaw = x[:, index["imu_yaw_rate_rps_lag0ms"]]
    command_age = x[:, index["steering_command_age_s"]]
    feedback_age = x[:, index["steering_feedback_age_s"]]
    # At k the full-unwind command changed one packet ago, while measured
    # steering is still stale and carries substantial yaw momentum.
    return (
        (command_age >= MIN_COMMAND_AGE_S)
        & (command_age <= MAX_COMMAND_AGE_S)
        & (feedback_age >= MIN_FEEDBACK_AGE_S)
        & (np.abs(steering) >= 0.30)
        & (np.abs(command) <= 0.05)
        & (np.abs(command - steering) >= 0.30)
        & (np.abs(yaw) >= 0.50)
    )


def _trigger_events(series: Any, timing_dir: Path) -> list[dict[str, Any]]:
    """Find gated sensor states and join to offline phase timing labels."""
    timing_tag = series.run_id.rsplit("_", 2)[-2]
    timing_path = timing_dir / f"steering_timing_{timing_tag}.json"
    if not timing_path.is_file():
        raise FileNotFoundError(f"missing timing report for {series.run_id}: {timing_path}")
    timing = json.loads(timing_path.read_text(encoding="utf-8"))
    onset_by_phase = {
        row["phase"]: int(row["command_to_feedback_onset_steps"])
        for row in timing["packet_grid_measurements"]
    }
    bag_path = ROOT / "live_runs" / series.run_id / "run" / "run_0.db3"
    phases, _ = _probe_phases(bag_path)
    events: list[dict[str, Any]] = []

    # The run has already passed the safe train/validation archive admission
    # gate. Re-read its time vector so every event is joined by packet-grid
    # sample time to the matching controlled probe phase.
    with np.load(ROOT / series.source, allow_pickle=False) as archive:
        run_ids = archive["run_ids"].astype(str).tolist()
        run_splits = archive["run_splits"].astype(str).tolist()
        if series.run_id not in run_ids:
            raise ValueError(f"run provenance mismatch: {series.run_id}")
        run_index = run_ids.index(series.run_id)
        if run_splits[run_index] != series.split:
            raise ValueError(f"run split changed after admission: {series.run_id}")
        sensors = np.asarray(archive["sensor_frames"], dtype=np.float32)
        sensor_valid = np.asarray(archive["sensor_valid"], dtype=bool)
        attitude_valid = np.asarray(archive["imu_attitude_valid"], dtype=bool)
        rigid = np.asarray(archive["simulator_rigid_state"], dtype=np.float32)
        times = np.asarray(archive["sample_time_ns"], dtype=np.int64)
        bounds = np.asarray(archive["sequence_bounds"], dtype=np.int64)
        sequence_run = np.asarray(archive["sequence_run_index"], dtype=np.int64)

    for sequence_id in np.flatnonzero(sequence_run == run_index):
        begin, end = map(int, bounds[int(sequence_id)])
        seq = sensors[begin:end]
        signal_ages = (
            atlas._age_since_change(seq[:, 7], 0.005),
            atlas._age_since_change(seq[:, 0], 0.005),
            atlas._age_since_change(seq[:, 8], 0.005),
            atlas._age_since_change(seq[:, 1], 0.005),
        )
        for k in range(max(HISTORY_LAGS), end - begin - 1):
            absolute_k = begin + k
            if (not all(sensor_valid[absolute_k - lag]
                        and attitude_valid[absolute_k - lag]
                        for lag in HISTORY_LAGS)):
                continue
            wheel_speed = 0.5 * float(seq[k, 2] + seq[k, 3])
            steering_cell = int(atlas._steer_cell(
                np.asarray([seq[k, 0]], dtype=np.float32))[0])
            if (int(wheel_speed // atlas.SPEED_BIN_MPS) != WHEEL_SPEED_CELL
                    or steering_cell not in STEERING_CELLS
                    or atlas._event(seq, k, "command_intent") != EVENT):
                continue

            command_age = float(signal_ages[0][k])
            feedback_age = float(signal_ages[1][k])
            steering = float(seq[k, 0])
            command = float(seq[k, 7])
            yaw_now = float(seq[k, 6])
            if not (
                MIN_COMMAND_AGE_S <= command_age <= MAX_COMMAND_AGE_S
                and feedback_age >= MIN_FEEDBACK_AGE_S
                and abs(steering) >= 0.30
                and abs(command) <= 0.05
                and abs(command - steering) >= 0.30
                and abs(yaw_now) >= 0.50
            ):
                continue

            phase_indices, labels = _phase_lookup(
                np.asarray([times[absolute_k], times[absolute_k + 1]],
                           dtype=np.int64), phases)
            phase_index = int(phase_indices[0])
            if (phase_index < 0
                    or int(phase_indices[1]) != phase_index):
                continue
            label = labels[phase_index]
            steps = onset_by_phase.get(label)
            events.append({
                "run_id": series.run_id,
                "split": series.split,
                "phase": label,
                "onset_steps": steps,
                "steering_cell": steering_cell,
                "current_imu_yaw_rate_radps": yaw_now,
                "next_gt_yaw_rate_radps": float(rigid[absolute_k + 1, 12]),
                "command_age_s": command_age,
                "steering_feedback_age_s": feedback_age,
            })
    return events


def _metrics(error: np.ndarray) -> dict[str, Any]:
    error = np.asarray(error, dtype=np.float64)
    absolute = np.abs(error)
    if not len(error):
        return {"samples": 0}
    return {
        "samples": int(len(error)),
        "rmse_radps": float(np.sqrt(np.mean(error * error))),
        "mae_radps": float(np.mean(absolute)),
        "p95_abs_radps": float(np.quantile(absolute, 0.95)),
        "max_abs_radps": float(np.max(absolute)),
        "fraction_abs_error_below_0p1": float(np.mean(absolute < 0.1)),
    }


def fit(baseline_dir: Path, output: Path) -> dict[str, Any]:
    baseline_dir = baseline_dir.resolve()
    baseline_report = json.loads((baseline_dir /
        "sensor_only_yaw_regime_atlas_report.json").read_text(encoding="utf-8"))
    bundle = joblib.load(baseline_dir / "sensor_only_yaw_regime_atlas.joblib")
    if baseline_report["model"]["event_definition"] != "command_intent":
        raise ValueError("baseline must use the command-intent event definition")

    admitted, source_audit = teacher._discover_series_with_safe_mixed_archives()
    selected = [row for row in admitted if row.run_id in TARGET_RUNS]
    if {row.run_id for row in selected} != TARGET_RUNS:
        raise ValueError("the two train captures and independent r03 validation are required")
    timing_dir = ROOT / "live_runs/racing_model_diagnostics_20261008/"
    timing_dir = timing_dir / "sensor_only_yaw_regime_atlas_v2"
    timing_events = [event for series in selected
                     for event in _trigger_events(series, timing_dir)
                     if event["onset_steps"] == 2]
    train_two = [event for event in timing_events
                 if event["split"] == "train" and event["onset_steps"] == 2]
    if len(train_two) < 1:
        raise ValueError("no two-packet training events met the regime gate")
    current = np.asarray([
        event["current_imu_yaw_rate_radps"] for event in train_two],
        dtype=np.float64)
    next_yaw = np.asarray([
        event["next_gt_yaw_rate_radps"] for event in train_two],
        dtype=np.float64)
    retention = float(np.sum(current * next_yaw) / np.sum(current * current))
    decay = 1.0 - retention

    history_lags = tuple(ms // 25 for ms in
                         baseline_report["input_contract"]["sensor_history_lags_ms"])
    age_report_path = (ROOT / "live_runs/racing_model_diagnostics_20261008/"
                       "sensor_only_yaw_regime_atlas_command_age_v4/"
                       "sensor_only_yaw_regime_atlas_report.json")
    age_report = json.loads(age_report_path.read_text(encoding="utf-8"))
    age_feature_names = age_report["input_contract"]["features"]
    feature_index = {name: i for i, name in enumerate(age_feature_names)}
    baseline_feature_indices = [age_feature_names.index(name)
                                for name in baseline_report["input_contract"]["features"]]

    validation_parts = []
    for series in admitted:
        if series.split != "validation":
            continue
        run = atlas._read_sensor_run(series, history_lags)
        rows = atlas._rows(run, "command_intent", history_lags)
        mask = (
            (atlas._speed_cell(rows["wheel_speed"]) == WHEEL_SPEED_CELL)
            & np.isin(atlas._steer_cell(rows["steering"]), STEERING_CELLS)
            & (rows["event"] == EVENT)
        )
        if np.any(mask):
            validation_parts.append({
                key: value[mask] for key, value in rows.items()
            })
    if not validation_parts:
        raise ValueError("no held-out rows occupy the targeted speed/steering cells")
    validation = {
        key: np.concatenate([part[key] for part in validation_parts], axis=0)
        for key in validation_parts[0]
    }
    x_age = validation["x"]
    x_base = x_age[:, baseline_feature_indices]
    baseline_prediction = np.full(len(x_base), np.nan, dtype=np.float32)
    speed_cell = atlas._speed_cell(validation["wheel_speed"])
    steering_cell = atlas._steer_cell(validation["steering"])
    for event, model in bundle["global_event_models"].items():
        mask = validation["event"] == event
        if np.any(mask):
            baseline_prediction[mask] = model.predict(
                x_base[mask]).astype(np.float32)
    for (speed, steering, event), model in bundle["local_expert_models"].items():
        mask = ((speed_cell == speed) & (steering_cell == steering)
                & (validation["event"] == event))
        if np.any(mask):
            baseline_prediction[mask] = model.predict(x_base[mask]).astype(np.float32)
    if not np.isfinite(baseline_prediction).all():
        raise ValueError("baseline has unsupported validation rows in target cells")

    gate = (
        (x_age[:, feature_index["steering_command_age_s"]] >= MIN_COMMAND_AGE_S)
        & (x_age[:, feature_index["steering_command_age_s"]] <= MAX_COMMAND_AGE_S)
        & (x_age[:, feature_index["steering_feedback_age_s"]] >= MIN_FEEDBACK_AGE_S)
        & (np.abs(x_age[:, feature_index["steering_feedback_rad_lag0ms"]]) >= 0.30)
        & (np.abs(x_age[:, feature_index["steering_command_rad_lag0ms"]]) <= 0.05)
        & (np.abs(x_age[:, feature_index["steering_command_gap_rad"]]) >= 0.30)
        & (np.abs(x_age[:, feature_index["imu_yaw_rate_rps_lag0ms"]]) >= 0.50)
    )
    candidate_prediction = baseline_prediction.copy()
    yaw_now = x_age[:, feature_index["imu_yaw_rate_rps_lag0ms"]]
    candidate_prediction[gate] = -decay * yaw_now[gate]
    target_residual = validation["residual"]

    validation_events = [event for event in timing_events
                         if event["split"] == "validation"
                         and event["onset_steps"] == 2]
    event_scores: dict[str, Any] = {}
    event_error = np.asarray([
        (1.0 - decay) * event["current_imu_yaw_rate_radps"]
        - event["next_gt_yaw_rate_radps"]
        for event in validation_events
    ], dtype=np.float64)
    event_scores["2_packet"] = {
        **_metrics(event_error),
        "events": validation_events,
        "assumes_two_packet_response": True,
    }

    report = {
        "title": "Two-packet full-unwind yaw-decay specialist (research counterfactual)",
        "candidate_model": {
            "law": "yaw_rate_next = (1 - decay) * current_imu_yaw_rate",
            "decay_fraction": decay,
            "retention_fraction": retention,
            "training_two_packet_event_count": len(train_two),
            "training_independent_runs": sorted({e["run_id"] for e in train_two}),
            "speed_proxy_bin_mps": [4.0, 4.5],
            "physical_steering_centers_rad": [-0.425, 0.425],
            "event": EVENT,
            "trigger": {
                "steering_command_age_s": [MIN_COMMAND_AGE_S, MAX_COMMAND_AGE_S],
                "steering_feedback_age_at_least_s": MIN_FEEDBACK_AGE_S,
                "abs_steering_feedback_at_least_rad": 0.30,
                "abs_steering_command_at_most_rad": 0.05,
                "abs_command_feedback_gap_at_least_rad": 0.30,
                "abs_current_imu_yaw_at_least_radps": 0.50,
            },
            "runtime_inputs_are_sensor_command_history_only": True,
            "requires_two_packet_actuator_response": True,
            "bridge_timing_used_for_training_filter_and_offline_score_only": True,
            "simulator_truth_used_only_as_fit_target_and_evaluation": True,
        },
        "packet_response_policy": "Only exactly two-packet phases are valid; all other response counts are excluded.",
        "training_events": train_two,
        "validation_trigger_events_by_measured_timing": event_scores,
        "whole_validation_cell": {
            "rows": int(len(target_residual)),
            "runs": int(len(np.unique(validation["run_id"]))),
            "gate_matches": int(np.count_nonzero(gate)),
            "global_or_local_atlas_baseline": _metrics(
                baseline_prediction - target_residual),
            "two_packet_decay_override": _metrics(
                candidate_prediction - target_residual),
            "per_steering_sign": {},
        },
        "source_audit": source_audit,
        "limitations": [
            "Only five two-packet training transition events support this coefficient.",
            "Only exactly-two-packet validation phases are scored; all other response counts are excluded.",
            "This is a single narrow regime and not a full speed/steering yaw model.",
            "Do not integrate into competition odometry/MPC without resolving the command-to-request phase or validating a permitted causal way to predict it.",
        ],
    }
    for cell in STEERING_CELLS:
        mask = steering_cell == cell
        report["whole_validation_cell"]["per_steering_sign"][str(cell)] = {
            "rows": int(np.count_nonzero(mask)),
            "gate_matches": int(np.count_nonzero(gate & mask)),
            "baseline": _metrics(baseline_prediction[mask] - target_residual[mask]),
            "candidate": _metrics(candidate_prediction[mask] - target_residual[mask]),
        }

    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "yaw_two_packet_release_specialist_report.json"
    model_path = output / "yaw_two_packet_release_specialist.json"
    model = {
        "law": report["candidate_model"]["law"],
        "decay_fraction": decay,
        "retention_fraction": retention,
        "trigger": report["candidate_model"]["trigger"],
        "requires_two_packet_response": True,
        "research_only": True,
        "competition_runtime_integration": False,
    }
    report_path.write_text(json.dumps(
        report, indent=2, sort_keys=True, default=atlas._json_value) + "\n",
                           encoding="utf-8")
    model_path.write_text(json.dumps(
        model, indent=2, sort_keys=True, default=atlas._json_value) + "\n",
                          encoding="utf-8")
    print(json.dumps({
        "report": str(report_path.relative_to(ROOT)),
        "model": str(model_path.relative_to(ROOT)),
        "training_two_packet_events": len(train_two),
        "decay_fraction": decay,
        "validation_two_packet": event_scores["2_packet"],
        "whole_validation_cell": report["whole_validation_cell"],
    }, indent=2, default=atlas._json_value))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, default=BASELINE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    fit(args.baseline_dir, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
