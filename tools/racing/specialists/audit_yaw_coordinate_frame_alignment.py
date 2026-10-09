#!/usr/bin/env python3
"""Verify the angular-velocity frame used by the yaw one-step target.

Repository semantics specify that simulator rigid-state angular velocity is
already vehicle-body-frame. This audit verifies exact-source IMU against that
same-frame truth and retains a rotated-to-world calculation only as a
counterfactual showing why a frame conversion would be wrong here. It is
diagnostic only; no runtime model is changed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    import audit_yaw_exact_packet_sensor_alignment as alignment
    import analyze_yaw_transition_timing_residuals as timing
    import fit_sensor_only_yaw_regime_atlas as atlas
    import score_yaw_atlas_transition_events as phase_tools
    import train_yaw_multihorizon_teacher as teacher
except ModuleNotFoundError:
    from tools.racing.specialists import audit_yaw_exact_packet_sensor_alignment as alignment
    from tools.racing.specialists import analyze_yaw_transition_timing_residuals as timing
    from tools.racing.specialists import fit_sensor_only_yaw_regime_atlas as atlas
    from tools.racing.specialists import score_yaw_atlas_transition_events as phase_tools
    from tools.racing.specialists import train_yaw_multihorizon_teacher as teacher


ROOT = Path(__file__).resolve().parents[3]
RUN_ID = "openplane_yaw_error_highsteer_reversal_validation_r03_20261008"
OUTPUT = (ROOT / "live_runs/racing_model_diagnostics_20261008/"
          "yaw_large_error_audit_v1/imu_world_frame_yaw_audit.json")


def _rotate_body_to_world(quaternion_xyzw: np.ndarray,
                          vector_body: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion_xyzw, dtype=np.float64)
    v = np.asarray(vector_body, dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if not np.isfinite(q).all() or not np.isfinite(v).all() or norm < 1.0e-8:
        return np.full(3, np.nan)
    qx, qy, qz, qw = q / norm
    qv = np.asarray((qx, qy, qz), dtype=np.float64)
    return v + 2.0 * np.cross(qv, np.cross(qv, v) + qw * v)


def _metric(error: np.ndarray) -> dict[str, Any]:
    error = np.asarray(error, dtype=np.float64)
    if not len(error):
        return {"samples": 0}
    absolute = np.abs(error)
    return {
        "samples": int(len(error)),
        "rmse_radps": float(np.sqrt(np.mean(error * error))),
        "p95_abs_radps": float(np.quantile(absolute, 0.95)),
        "max_abs_radps": float(np.max(absolute)),
        "samples_over_0p1": int(np.count_nonzero(absolute > 0.1)),
    }


def run(output: Path = OUTPUT) -> dict[str, Any]:
    admitted, source_audit = teacher._discover_series_with_safe_mixed_archives()
    matches = [row for row in admitted
               if row.run_id == RUN_ID and row.split == "validation"]
    if len(matches) != 1:
        raise ValueError(f"expected one whole-run held-out capture: {RUN_ID}")
    series = matches[0]
    bag = alignment._bag_for_run(series)
    _original, exact, join = alignment._read_patched_run(series)
    rows = atlas._rows(exact, "command_intent")
    times = alignment._row_sample_times(exact)

    rotated_world_gyro_z: list[float] = []
    rotated_world_gyro_xyz: list[list[float]] = []
    current_body_gt_xyz: list[list[float]] = []
    body_gyro_xyz: list[list[float]] = []
    # This reproduces atlas row eligibility/order and retains the current
    # quaternion/gyro for the same exact source-packet one-step targets.
    lags = atlas.HISTORY_LAGS
    for (sensors, sensor_valid, attitude, attitude_valid, rigid), _ in zip(
            exact["sequences"], exact["sequence_times"]):
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
            omega_body = np.asarray((attitude[k, 2], attitude[k, 3],
                                     sensors[k, 6]), dtype=np.float64)
            omega_world = _rotate_body_to_world(rigid[k, 3:7], omega_body)
            body_gyro_xyz.append(omega_body.tolist())
            rotated_world_gyro_xyz.append(omega_world.tolist())
            rotated_world_gyro_z.append(float(omega_world[2]))
            current_body_gt_xyz.append([float(value)
                                        for value in rigid[k, 10:13]])

    body_z = rows["imu_yaw"].astype(np.float64)
    next_gt = (rows["residual"] + rows["imu_yaw"]).astype(np.float64)
    current_gt = rows["gt_yaw_current"].astype(np.float64)
    rotated_world_z = np.asarray(rotated_world_gyro_z, dtype=np.float64)
    rotated_world_xyz = np.asarray(rotated_world_gyro_xyz, dtype=np.float64)
    body_xyz = np.asarray(body_gyro_xyz, dtype=np.float64)
    gt_body_xyz = np.asarray(current_body_gt_xyz, dtype=np.float64)
    if not (len(times) == len(rows["residual"]) == len(rotated_world_z)):
        raise ValueError("coordinate-frame rows do not align to atlas one-step rows")
    if not np.allclose(gt_body_xyz[:, 2], current_gt, rtol=1.0e-6,
                       atol=1.0e-6):
        raise ValueError("current body-yaw truth is not aligned to atlas rows")
    if not np.allclose(body_xyz, gt_body_xyz, rtol=1.0e-6, atol=1.0e-6):
        raise ValueError("exact-source IMU body gyro is not aligned to body-frame GT")
    if not np.isfinite(rotated_world_z).all():
        raise ValueError("non-finite transformed gyro samples")

    phases, _ = timing._probe_phases(bag)
    phase_index, phase_labels = phase_tools._phase_lookup(times, phases)
    timing_path = timing.TIMING_DIR / "steering_timing_r03.json"
    timing_report = json.loads(timing_path.read_text(encoding="utf-8"))
    response_steps = {
        row["phase"]: int(row["command_to_feedback_onset_steps"])
        for row in timing_report["packet_grid_measurements"]
    }
    packet_group = np.full(len(times), "outside_probe", dtype="U24")
    event_group = rows["event"].astype(str)
    for index, phase_id in enumerate(phase_index):
        if phase_id < 0:
            continue
        label = phase_labels[int(phase_id)]
        event = timing._phase_spec(label)
        steps = response_steps.get(label)
        if event is not None and steps == 2:
            packet_group[index] = f"{event['probe_event']}/2_packet"

    current_body_error = body_z - current_gt
    predictor_errors = {
        "body_z_imu_persistence_to_next_gt": body_z - next_gt,
        "incorrectly_rotated_world_z_counterfactual_to_next_gt": (
            rotated_world_z - next_gt),
    }
    result = {
        "title": "IMU and simulator yaw-rate frame audit",
        "run_id": RUN_ID,
        "split": "validation",
        "exact_source_packet_imu": True,
        "test_and_final_test_opened": False,
        "frame_semantics": "Repository metadata identifies simulator rigid-state linear/angular velocity as vehicle-body-frame; exact-source IMU angular velocity is compared in that same frame.",
        "counterfactual": "The quaternion-rotated gyro is world-frame and intentionally compared only as a wrong-frame counterfactual.",
        "inputs": "IMU body gyro and attitude; quaternion and angular velocity truth used only for offline diagnostics/targets",
        "counts": {"rows": int(len(body_z)), "probe_rows": int(np.count_nonzero(
            packet_group != "outside_probe"))},
        "current_sample_frame_agreement": {
            "body_gyro_z_vs_gt_body_z": _metric(current_body_error),
            "body_gyro_xyz_vs_gt_body_xyz_rmse_by_axis": [
                float(np.sqrt(np.mean((body_xyz[:, axis]
                                       - gt_body_xyz[:, axis]) ** 2)))
                for axis in range(3)
            ],
            "mean_abs_counterfactual_rotation_difference_radps": float(np.mean(
                np.abs(rotated_world_z - body_z))),
            "p95_abs_counterfactual_rotation_difference_radps": float(np.quantile(
                np.abs(rotated_world_z - body_z), 0.95)),
        },
        "next_sample_prediction_errors": {
            name: _metric(error) for name, error in predictor_errors.items()},
        "next_sample_by_response_group": {},
        "next_sample_by_transition_event": {},
        "largest_wrong_frame_counterfactual_samples": [],
        "join": join,
        "source_audit": source_audit,
        "conclusion": "",
    }
    for group in sorted(set(packet_group.tolist())):
        mask = packet_group == group
        if np.any(mask):
            result["next_sample_by_response_group"][group] = {
                name: _metric(error[mask])
                for name, error in predictor_errors.items()}
    for event in ("hold", "turn_in", "unwind", "reversal"):
        mask = event_group == event
        if np.any(mask):
            result["next_sample_by_transition_event"][event] = {
                name: _metric(error[mask])
                for name, error in predictor_errors.items()}
    correction_order = np.argsort(np.abs(rotated_world_z - body_z))[::-1]
    for index in correction_order[:30]:
        result["largest_wrong_frame_counterfactual_samples"].append({
            "sample_time_ns": int(times[index]),
            "packet_group": str(packet_group[index]),
            "event": str(event_group[index]),
            "body_gyro_xyz_radps": [float(value) for value in body_xyz[index]],
            "counterfactual_rotated_world_gyro_xyz_radps": [
                float(value) for value in rotated_world_xyz[index]],
            "current_gt_body_yaw_rate_radps": float(current_gt[index]),
            "next_gt_body_yaw_rate_radps": float(next_gt[index]),
            "body_z_next_error_radps": float(body_z[index] - next_gt[index]),
            "incorrectly_rotated_world_z_next_error_radps": float(
                rotated_world_z[index] - next_gt[index]),
        })
    improvement = (result["next_sample_prediction_errors"][
        "body_z_imu_persistence_to_next_gt"]["rmse_radps"]
        - result["next_sample_prediction_errors"][
            "incorrectly_rotated_world_z_counterfactual_to_next_gt"]["rmse_radps"])
    result["body_minus_wrong_frame_rmse_radps"] = float(improvement)
    result["conclusion"] = (
        "The wrong-frame rotation unexpectedly improves the numerical score; frame semantics require investigating label alignment."
        if improvement > 1.0e-4 else
        "Rigid-state and IMU yaw rates share the vehicle-body frame; the world-frame rotation does not improve next-step yaw persistence on this held-out run.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True,
                                  default=atlas._json_value) + "\n",
                      encoding="utf-8")
    print(json.dumps({
        "output": str(output.relative_to(ROOT)),
        "current_sample_frame_agreement": result["current_sample_frame_agreement"],
        "next_sample_prediction_errors": result["next_sample_prediction_errors"],
        "response_groups": result["next_sample_by_response_group"],
        "body_minus_wrong_frame_rmse_radps": improvement,
    }, indent=2))
    return result


if __name__ == "__main__":
    run()
