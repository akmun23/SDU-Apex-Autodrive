#!/usr/bin/env python3
"""Historical broad packet-alignment audit; do not execute under current policy.

Its old whole-run and non-two-packet subgroup scores are superseded. Importable
bag/source-stamp utilities remain available; current yaw scoring uses only
exact-two-packet response phases.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

try:
    import analyze_yaw_transition_timing_residuals as timing_audit
    import audit_sensor_yaw_large_errors as audit
    import fit_sensor_only_yaw_regime_atlas as atlas
    import train_yaw_multihorizon_teacher as teacher
    import yaw_source_packet_alignment as packet_alignment
except ModuleNotFoundError:
    from tools.racing.specialists import analyze_yaw_transition_timing_residuals as timing_audit
    from tools.racing.specialists import audit_sensor_yaw_large_errors as audit
    from tools.racing.specialists import fit_sensor_only_yaw_regime_atlas as atlas
    from tools.racing.specialists import train_yaw_multihorizon_teacher as teacher
    from tools.racing.specialists import yaw_source_packet_alignment as packet_alignment


ROOT = Path(__file__).resolve().parents[3]
BASELINE_DIR = (ROOT / "live_runs/racing_model_diagnostics_20261008/"
                "sensor_only_yaw_regime_atlas_command_intent_v2")
CANDIDATE_DIR = (ROOT / "live_runs/racing_model_diagnostics_20261008/"
                 "yaw_large_error_audit_v1/unwind_reversal_neighborhood_v2_supported")
EXACT_REFIT_DIR = (ROOT / "live_runs/racing_model_diagnostics_20261008/"
                   "sensor_only_yaw_regime_atlas_exact_packet_v1")
OUTPUT = (ROOT / "live_runs/racing_model_diagnostics_20261008/"
          "yaw_large_error_audit_v1/exact_packet_sensor_alignment_counterfactual.json")


def _bag_for_run(series: Any) -> Path:
    archive_path = ROOT / series.source
    manifest = json.loads(archive_path.with_name("manifest.json").read_text(
        encoding="utf-8"))
    matches = [row for row in manifest.get("runs", [])
               if row.get("run_id") == series.run_id
               and row.get("effective_split") == series.split]
    if len(matches) != 1:
        raise ValueError(f"{series.run_id}: expected one split-matched bag row")
    bag = ROOT / matches[0]["bag"]
    if not bag.is_file():
        raise FileNotFoundError(f"{series.run_id}: raw bag missing: {bag}")
    return bag


def _read_patched_run(
        series: Any) -> tuple[dict[str, Any], dict[str, Any], dict[str, int]]:
    """Read one admitted run and replace receipt-causal IMU with packet IMU."""
    path = ROOT / series.source
    with np.load(path, allow_pickle=False) as archive:
        run_ids = archive["run_ids"].astype(str).tolist()
        splits = archive["run_splits"].astype(str).tolist()
        run_index = run_ids.index(series.run_id)
        if splits[run_index] != series.split:
            raise ValueError(f"{series.run_id}: split metadata changed")
        times = np.asarray(archive["sample_time_ns"], dtype=np.int64)
        bounds = np.asarray(archive["sequence_bounds"], dtype=np.int64)
        sequence_runs = np.asarray(archive["sequence_run_index"], dtype=np.int64)
        source_sensor = np.asarray(archive["sensor_frames"], dtype=np.float32)
        source_attitude = np.asarray(archive["imu_attitude_frames"], dtype=np.float32)

    exact_by_receipt, odom_packets = packet_alignment.exact_imu_by_odom_receipt(
        _bag_for_run(series))
    patched_sensor = source_sensor.copy()
    patched_attitude = source_attitude.copy()
    changed_yaw = 0
    changed_yaw_over_0p1 = 0
    exact_rows = 0
    for index, receipt_ns in enumerate(times):
        values = exact_by_receipt.get(int(receipt_ns))
        if values is None:
            continue
        exact_rows += 1
        old_yaw = float(source_sensor[index, 6])
        new_yaw = float(values[2])
        if abs(old_yaw - new_yaw) > 1.0e-6:
            changed_yaw += 1
        if abs(old_yaw - new_yaw) > 0.1:
            changed_yaw_over_0p1 += 1
        patched_sensor[index, 4:7] = values[:3]
        patched_attitude[index, :] = values[3:]

    run = atlas._read_sensor_run(series)
    original_sequences = list(run["sequences"])
    sequences = []
    sequence_times = []
    cursor = 0
    for sequence_id in np.flatnonzero(sequence_runs == run_index):
        begin, end = map(int, bounds[int(sequence_id)])
        sensors, sensor_valid, attitude, attitude_valid, rigid = run["sequences"][cursor]
        cursor += 1
        sensors = sensors.copy()
        attitude = attitude.copy()
        sensors[:, 4:7] = patched_sensor[begin:end, 4:7]
        attitude[:, :] = patched_attitude[begin:end, :]
        sequences.append((sensors, sensor_valid, attitude, attitude_valid, rigid))
        sequence_times.append(times[begin:end])
    if cursor != len(run["sequences"]):
        raise ValueError(f"{series.run_id}: sequence alignment mismatch")
    original_run = {**run, "sequences": original_sequences,
                    "sequence_times": sequence_times}
    patched_run = {**run, "sequences": sequences,
                   "sequence_times": sequence_times}
    return original_run, patched_run, {
        "archive_samples": int(len(times)),
        "raw_odometry_packets": int(odom_packets),
        "exact_source_stamp_imu_matches": int(exact_rows),
        "archive_yaw_values_changed": int(changed_yaw),
        "archive_yaw_values_changed_by_more_than_0p1_radps": int(
            changed_yaw_over_0p1),
    }


def _row_sample_times(run: dict[str, Any],
                      history_lags: tuple[int, ...] = atlas.HISTORY_LAGS
                      ) -> np.ndarray:
    """Mirror atlas row gates to retain each row's raw packet receipt time."""
    result: list[int] = []
    lags = history_lags
    for (sensors, sensor_valid, attitude, attitude_valid, rigid), times in zip(
            run["sequences"], run["sequence_times"]):
        if (not np.isfinite(sensors).all() or not np.isfinite(attitude).all()
                or not np.isfinite(rigid).all()):
            continue
        for k in range(max(lags), len(sensors) - 1):
            if (not all(sensor_valid[k - lag] for lag in lags)
                    or not all(attitude_valid[k - lag] for lag in lags)):
                continue
            wheel_mean = 0.5 * float(sensors[k, 2] + sensors[k, 3])
            speed_cell = int(wheel_mean // atlas.SPEED_BIN_MPS)
            steering = float(sensors[k, 0])
            if (speed_cell < 0 or speed_cell >= len(atlas.SPEED_CENTERS)
                    or not np.isfinite(steering)
                    or steering < atlas.STEERING_CENTERS[0] - 0.0125
                    or steering > atlas.STEERING_CENTERS[-1] + 0.0125):
                continue
            result.append(int(times[k]))
    return np.asarray(result, dtype=np.int64)


def run(output: Path = OUTPUT,
        run_id: str = "openplane_yaw_error_highsteer_reversal_validation_r03_20261008"
        ) -> dict[str, Any]:
    base_report = json.loads((BASELINE_DIR /
        "sensor_only_yaw_regime_atlas_report.json").read_text(encoding="utf-8"))
    base_bundle = joblib.load(BASELINE_DIR / "sensor_only_yaw_regime_atlas.joblib")
    candidate_bundle = joblib.load(CANDIDATE_DIR / "candidate.joblib")
    feature_count = len(base_report["input_contract"]["features"])
    admitted, source_audit = teacher._discover_series_with_safe_mixed_archives()
    matches = [series for series in admitted
               if series.run_id == run_id and series.split == "validation"]
    if len(matches) != 1:
        raise ValueError(f"expected one admitted validation run {run_id!r}")
    series = matches[0]
    original_run, exact_run, join = _read_patched_run(series)
    original_rows = atlas._rows(original_run, "command_intent")
    exact_rows = atlas._rows(exact_run, "command_intent")
    row_times = _row_sample_times(original_run)
    if len(row_times) != len(original_rows["residual"]):
        raise ValueError(f"{series.run_id}: packet-time row alignment mismatch")
    if not np.array_equal(original_rows["run_id"], exact_rows["run_id"]):
        raise ValueError(f"{series.run_id}: alignment changed row provenance")
    original_next_gt = original_rows["residual"] + original_rows["imu_yaw"]
    exact_next_gt = exact_rows["residual"] + exact_rows["imu_yaw"]
    if not np.allclose(original_next_gt, exact_next_gt, rtol=0.0, atol=1.0e-6):
        raise ValueError(f"{series.run_id}: alignment changed GT yaw targets")

    # Exact-source IMU changes only a small part of the row history. Score that
    # union first; unaffected rows retain their frozen predictions exactly.
    affected = np.any(np.abs(original_rows["x"] - exact_rows["x"]) > 1.0e-6,
                      axis=1)
    if not np.any(affected):
        raise ValueError(f"{series.run_id}: exact-source join changed no model rows")
    selected_original = {key: value[affected]
                         for key, value in original_rows.items()}
    selected_exact = {key: value[affected]
                      for key, value in exact_rows.items()}
    baseline_original = audit._predict(
        selected_original, base_bundle, feature_count)[2]
    candidate_original = timing_audit._predict_candidate(
        selected_original, base_bundle, candidate_bundle, feature_count)[1]
    baseline_exact = audit._predict(
        selected_exact, base_bundle, feature_count)[2]
    candidate_exact = timing_audit._predict_candidate(
        selected_exact, base_bundle, candidate_bundle, feature_count)[1]
    baseline_original_error = baseline_original - selected_original["residual"]
    candidate_original_error = candidate_original - selected_original["residual"]
    baseline_exact_error = baseline_exact - selected_exact["residual"]
    candidate_exact_error = candidate_exact - selected_exact["residual"]
    exact_report = json.loads((EXACT_REFIT_DIR /
        "sensor_only_yaw_regime_atlas_report.json").read_text(encoding="utf-8"))
    exact_bundle = joblib.load(EXACT_REFIT_DIR / "sensor_only_yaw_regime_atlas.joblib")
    exact_local, exact_global, exact_prediction = audit._predict(
        exact_rows, exact_bundle, len(exact_report["input_contract"]["features"]))
    exact_error = exact_prediction - exact_rows["residual"]
    phases, _ = timing_audit._probe_phases(_bag_for_run(series))
    phase_index, phase_labels = timing_audit._phase_lookup(row_times, phases)
    next_phase_index, _ = timing_audit._phase_lookup(
        row_times + 25_000_000, phases)
    timing_path = timing_audit.TIMING_DIR / "steering_timing_r03.json"
    timing_report = json.loads(timing_path.read_text(encoding="utf-8"))
    response_steps = {
        row["phase"]: int(row["command_to_feedback_onset_steps"])
        for row in timing_report["packet_grid_measurements"]
    }
    packet_group = np.full(len(row_times), "outside_probe", dtype="U24")
    for i, phase_id in enumerate(phase_index):
        if phase_id < 0 or next_phase_index[i] != phase_id:
            continue
        label = phase_labels[int(phase_id)]
        spec = timing_audit._phase_spec(label)
        if spec is not None:
            packet_group[i] = f"{spec['probe_event']}/{response_steps.get(label, 0)}_packet"
    changed_rows: dict[str, Any] = {}
    event_values = selected_original["event"].astype(str)
    for event in sorted(set(event_values)):
        mask = event_values == event
        changed_rows[event] = {
            "rows_affected_by_alignment": int(np.count_nonzero(mask)),
            "baseline_before": audit._metric(baseline_original_error[mask]),
            "baseline_after": audit._metric(baseline_exact_error[mask]),
            "candidate_before": audit._metric(candidate_original_error[mask]),
            "candidate_after": audit._metric(candidate_exact_error[mask]),
            "candidate_outliers_removed": int(
                np.count_nonzero(np.abs(candidate_original_error[mask]) > 0.1)
                - np.count_nonzero(np.abs(candidate_exact_error[mask]) > 0.1)),
        }
    remaining = []
    affected_indices = np.flatnonzero(affected)
    for selected_index in np.flatnonzero(np.abs(candidate_exact_error) > 0.1):
        row_index = int(affected_indices[selected_index])
        x = exact_rows["x"][row_index]
        steering_feature = atlas.OBSERVATION_NAMES.index("steering_feedback_rad")
        command_feature = atlas.OBSERVATION_NAMES.index("steering_command_rad")
        remaining.append({
            "sample_time_ns": int(row_times[row_index]),
            "transition_event": str(exact_rows["event"][row_index]),
            "measured_packet_response_group": str(packet_group[row_index]),
            "wheel_mean_mps": float(exact_rows["wheel_speed"][row_index]),
            "steering_feedback_rad": float(exact_rows["steering"][row_index]),
            "steering_command_rad": float(x[command_feature]),
            "steering_command_feedback_gap_rad": float(
                x[command_feature] - x[steering_feature]),
            "current_exact_packet_imu_yaw_radps": float(
                exact_rows["imu_yaw"][row_index]),
            "next_gt_yaw_rate_radps": float(exact_rows["residual"][row_index]
                                             + exact_rows["imu_yaw"][row_index]),
            "predicted_next_yaw_rate_radps": float(candidate_exact[selected_index]
                                                   + exact_rows["imu_yaw"][row_index]),
            "signed_error_radps": float(candidate_exact_error[selected_index]),
        })
    run_report = {
        "split": series.split,
        "source_join": join,
        "all_one_step_rows": int(len(original_rows["residual"])),
        "rows_affected_by_alignment": int(np.count_nonzero(affected)),
        "baseline_before": audit._metric(baseline_original_error),
        "baseline_after": audit._metric(baseline_exact_error),
        "candidate_before": audit._metric(candidate_original_error),
        "candidate_after": audit._metric(candidate_exact_error),
        "candidate_outliers_removed": int(
            np.count_nonzero(np.abs(candidate_original_error) > 0.1)
            - np.count_nonzero(np.abs(candidate_exact_error) > 0.1)),
        "by_transition_event": changed_rows,
        "by_measured_packet_response_group": {},
        "remaining_over_0p1_radps": remaining,
        "refit_exact_packet_model_full_run": {
            "metrics": audit._metric(exact_error),
            "local_expert_coverage_fraction": float(np.mean(
                np.isfinite(exact_local))),
            "global_fallback_coverage_fraction": float(np.mean(
                np.isfinite(exact_global))),
            "by_measured_packet_response_group": {},
            "by_packet_group_and_transition_event": {},
            "worst_over_0p1": [],
        },
    }
    probe_mask = packet_group != "outside_probe"
    for group in sorted(set(packet_group[probe_mask].tolist())):
        mask = packet_group == group
        run_report["refit_exact_packet_model_full_run"][
            "by_measured_packet_response_group"][group] = audit._metric(
                exact_error[mask])
        for event in sorted(set(exact_rows["event"][mask].astype(str))):
            group_event = mask & (exact_rows["event"] == event)
            run_report["refit_exact_packet_model_full_run"][
                "by_packet_group_and_transition_event"][f"{group}/{event}"] = (
                    audit._metric(exact_error[group_event]))
    worst_indices = np.flatnonzero(np.abs(exact_error) > 0.1)
    worst_indices = worst_indices[np.argsort(np.abs(exact_error[worst_indices]))[::-1]]
    for row_index in worst_indices[:50]:
        run_report["refit_exact_packet_model_full_run"]["worst_over_0p1"].append({
            "sample_time_ns": int(row_times[row_index]),
            "event": str(exact_rows["event"][row_index]),
            "packet_response_group": str(packet_group[row_index]),
            "wheel_mean_mps": float(exact_rows["wheel_speed"][row_index]),
            "steering_feedback_rad": float(exact_rows["steering"][row_index]),
            "current_imu_yaw_radps": float(exact_rows["imu_yaw"][row_index]),
            "next_gt_yaw_rate_radps": float(exact_rows["residual"][row_index]
                                             + exact_rows["imu_yaw"][row_index]),
            "predicted_next_yaw_rate_radps": float(exact_prediction[row_index]
                                                   + exact_rows["imu_yaw"][row_index]),
            "signed_error_radps": float(exact_error[row_index]),
        })
    for group in sorted(set(packet_group[affected].tolist())):
        mask = affected & (packet_group == group)
        if not np.any(mask):
            continue
        run_report["by_measured_packet_response_group"][group] = {
            "rows_affected_by_alignment": int(np.count_nonzero(mask)),
            "candidate_before": audit._metric(candidate_original_error[
                mask[affected]]),
            "candidate_after": audit._metric(candidate_exact_error[mask[affected]]),
        }
    result = {
        "title": "Exact-source IMU alignment counterfactual for held-out yaw models",
        "validation_run": series.run_id,
        "test_and_final_test_arrays_opened": False,
        "fit_or_runtime_change": False,
        "runtime_sensor_packet_contract": (
            "current odometry assembles left/right encoder and IMU by exact header source stamp"),
        "alignment_change": (
            "replace only IMU acceleration, yaw rate, roll/pitch, and gyro x/y with the raw IMU message whose header stamp exactly matches each odometry source stamp; retain receipt-causal actuator command/feedback and existing encoder-derived wheel speed"),
        "affected_row_frozen_model_counterfactual": run_report,
        "source_split_audit": source_audit,
        "interpretation": [
            "This is a frozen-model intervention: model parameters were trained on receipt-causal features, so exact-source scores are diagnostic and require a same-split refit before claiming improvement.",
            "Exact-source IMU is valid only if the runtime observer has the completed source-stamped packet when predicting; it never uses future simulator truth.",
            "A lower mean error does not satisfy the user's per-sample <=0.1 rad/s requirement; inspect maximum and >0.1 counts by event and run.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True,
                                 default=atlas._json_value) + "\n",
                      encoding="utf-8")
    print(json.dumps({
        "output": str(output.relative_to(ROOT)),
        "run_id": series.run_id,
        "source_join": join,
        "rows_affected": int(np.count_nonzero(affected)),
        "candidate_outliers_removed": run_report["candidate_outliers_removed"],
        "candidate_after": run_report["candidate_after"],
        "remaining_large_errors": remaining,
        "by_measured_packet_response_group": run_report[
            "by_measured_packet_response_group"],
        "refit_exact_packet_model_full_run": run_report[
            "refit_exact_packet_model_full_run"],
    }, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--run-id", default=(
        "openplane_yaw_error_highsteer_reversal_validation_r03_20261008"))
    args = parser.parse_args()
    run(args.output, args.run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(
        "Deprecated broad alignment audit: only exact-two-packet response "
        "phases are valid for current yaw analysis.")
