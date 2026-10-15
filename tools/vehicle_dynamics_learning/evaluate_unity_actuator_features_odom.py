#!/usr/bin/env python3
"""Test Unity-source actuator physics features for practice Odom residuals.

Fits on complete R01/R03 training captures and evaluates untouched R02/R04
captures. Runtime-allowed steering/throttle feedback are causal features;
simulator truth is only the offline target and scoring label. This is a
pointwise feature-transfer experiment, not an observer promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from tools.vehicle_dynamics_learning import fit_supported_practice_odom_u_residual as base


WHEELBASE_M = 0.324
TRACK_WIDTH_M = 0.236
WHEEL_RADIUS_M = 0.059
COM_TO_REAR_M = 0.15532
MOTOR_TORQUE_NM = 85.6
GATE_MIN_U_MPS = base.GATE_MIN_U_MPS
GATE_MAX_U_MPS = base.GATE_MAX_U_MPS
ALPHA = base.ALPHA

SOURCE_FEATURE_NAMES = (
    "unity_drive_torque_per_wheel_nm",
    "unity_zero_throttle_brake_torque_per_wheel_nm",
    "unity_drive_torque_slew_nmps",
    "measured_center_steering_rad",
    "measured_center_steering_rate_radps",
    "unity_kinematic_lateral_demand_mps2",
    "unity_steering_curvature_inv_m",
    "unity_drivetrain_power_proxy_w",
    "rear_wheel_surface_acceleration_mps2",
    "wheel_surface_accel_minus_unity_rear_body_ax_mps2",
    "drive_torque_times_wheel_body_speed_gap_nm_mps",
    "steer_demand_times_wheel_body_speed_gap_mps2",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _features(samples: list[Any], diagnostics: np.ndarray) -> np.ndarray:
    actuator = np.asarray([sample.actuators for sample in samples], dtype=np.float64)
    if actuator.shape != (len(samples), 4) or not np.isfinite(actuator).all():
        raise ValueError("capture lacks finite aligned steering/throttle feedback and commands")
    # MotionSample actuator order is [measured steering, measured throttle,
    # throttle command, steering command]. Unity VehicleController.cs maps
    # nonzero throttle to 85.6 N m total CAWD torque, equally divided among
    # four wheels. Exactly zero throttle instead applies 85.6 N m brake torque
    # at each wheel and no motor torque.
    steering, throttle = actuator[:, 0], actuator[:, 1]
    drive_per_wheel = np.where(throttle != 0.0,
                               MOTOR_TORQUE_NM * throttle / 4.0, 0.0)
    brake_per_wheel = np.where(throttle == 0.0,
                               MOTOR_TORQUE_NM, 0.0)
    steering_rate = np.zeros(len(samples), dtype=np.float64)
    throttle_rate = np.zeros(len(samples), dtype=np.float64)
    for index in range(1, len(samples)):
        if samples[index].packet_sequence == samples[index - 1].packet_sequence + 1:
            steering_rate[index] = (steering[index] - steering[index - 1]) / 0.025
            throttle_rate[index] = (throttle[index] - throttle[index - 1]) / 0.025
    drive_torque_rate = np.where(
        throttle != 0.0,
        MOTOR_TORQUE_NM * throttle_rate / 4.0, 0.0)

    u = diagnostics[:, 7]
    wheel_surface_speed = diagnostics[:, 4]
    wheel_gap = wheel_surface_speed - u
    curvature = np.tan(steering) / WHEELBASE_M
    lateral_demand = u * u * curvature
    yaw_rate = diagnostics[:, 11]
    unity_rear_ax = diagnostics[:, 9] + COM_TO_REAR_M * yaw_rate ** 2
    wheel_acceleration = np.zeros(len(samples), dtype=np.float64)
    for index in range(1, len(samples)):
        if samples[index].packet_sequence == samples[index - 1].packet_sequence + 1:
            wheel_acceleration[index] = (
                wheel_surface_speed[index] - wheel_surface_speed[index - 1]) / 0.025
    drivetrain_power = drive_per_wheel * (wheel_surface_speed / WHEEL_RADIUS_M)
    return np.column_stack((
        drive_per_wheel,
        brake_per_wheel,
        drive_torque_rate,
        steering,
        steering_rate,
        lateral_demand,
        curvature,
        drivetrain_power,
        wheel_acceleration,
        wheel_acceleration - unity_rear_ax,
        drive_per_wheel * wheel_gap,
        lateral_demand * wheel_gap,
    ))


def _weighted_normalizer(features: np.ndarray, groups: np.ndarray
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weights = base._weights(groups)
    mean = np.average(features, axis=0, weights=weights)
    scale = np.sqrt(np.maximum(
        np.average((features - mean) ** 2, axis=0, weights=weights), 1.0e-8))
    return mean, scale, weights


def _fit_path(features_by_run: list[np.ndarray],
              data_by_run: list[tuple[list[Any], np.ndarray, np.ndarray,
                                      np.ndarray, np.ndarray]],
              position_loss_weight: float = 1.0) -> dict[str, Any]:
    all_features = np.concatenate(features_by_run)
    all_groups = np.concatenate([item[2] for item in data_by_run])
    mean, scale, _ = _weighted_normalizer(all_features, all_groups)
    design_blocks = []
    target_blocks = []
    for features, (samples, diagnostics, groups, replay_pose, _) in zip(
            features_by_run, data_by_run):
        baseline_u = diagnostics[:, 7]
        gate = ((baseline_u >= GATE_MIN_U_MPS) &
                (baseline_u < GATE_MAX_U_MPS)).astype(np.float64)
        normalized = np.column_stack((
            np.ones(len(features)), (features - mean) / scale))
        speed_design = normalized * gate[:, None]
        speed_target = np.asarray([sample.state[0] for sample in samples]) - baseline_u
        position_design, position_target = base._integrated_u_correction_design(
            features, diagnostics, replay_pose, samples, mean, scale)
        speed_scale = math.sqrt(float(np.mean(speed_target ** 2)))
        position_scale = math.sqrt(float(np.mean(
            position_target.reshape(-1, 2).__pow__(2).sum(axis=1))))
        if speed_scale <= 1.0e-9 or position_scale <= 1.0e-9:
            raise ValueError("training run has degenerate speed/position residual")
        run_scale = math.sqrt(len(data_by_run))
        design_blocks.extend((
            speed_design / (speed_scale * math.sqrt(len(samples)) * run_scale),
            math.sqrt(position_loss_weight) * position_design /
            (position_scale * math.sqrt(2 * len(samples)) * run_scale),
        ))
        target_blocks.extend((
            speed_target / (speed_scale * math.sqrt(len(samples)) * run_scale),
            math.sqrt(position_loss_weight) * position_target /
            (position_scale * math.sqrt(2 * len(samples)) * run_scale),
        ))
    design = np.vstack(design_blocks)
    target = np.concatenate(target_blocks)
    penalty = ALPHA * np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    parameters = np.linalg.solve(design.T @ design + penalty, design.T @ target)
    return {"mean": mean, "scale": scale, "parameters": parameters,
            "position_loss_weight": position_loss_weight}


def _evaluate_run(label: str, data: tuple[list[Any], np.ndarray, np.ndarray,
                                           np.ndarray, np.ndarray],
                  features: np.ndarray, model: dict[str, Any]) -> dict[str, Any]:
    samples, diagnostics, groups, replay_pose, _ = data
    baseline_u = diagnostics[:, 7]
    truth_u = np.asarray([sample.state[0] for sample in samples])
    mean, scale, parameters = model["mean"], model["scale"], model["parameters"]
    correction = ((np.column_stack((np.ones(len(features)),
                                    (features - mean) / scale)) @ parameters))
    active = (baseline_u >= GATE_MIN_U_MPS) & (baseline_u < GATE_MAX_U_MPS)
    correction *= active
    baseline_error = baseline_u - truth_u
    candidate_error = baseline_error + correction
    pose = base._pose_metrics(
        samples, groups, diagnostics, replay_pose, correction)
    per_sequence = []
    for group in dict.fromkeys(groups.tolist()):
        mask = groups == group
        before = base._metrics(baseline_error[mask])
        after = base._metrics(candidate_error[mask])
        per_sequence.append({
            "group": str(group),
            "baseline_rmse_mps": before["rmse"],
            "candidate_rmse_mps": after["rmse"],
            "improved": after["rmse"] < before["rmse"],
        })
    return {
        "run": label,
        "samples": len(samples),
        "gate_active_samples": int(np.sum(active)),
        "forward_speed_baseline": base._metrics(baseline_error),
        "forward_speed_candidate": base._metrics(candidate_error),
        "pose": pose,
        "lap_macro": {
            "groups": len(per_sequence),
            "improved": sum(item["improved"] for item in per_sequence),
            "per_group": per_sequence,
        },
    }


def _source_mode(samples: list[Any]) -> np.ndarray:
    throttle = np.asarray([sample.actuators[1] for sample in samples],
                          dtype=np.float64)
    # Unity VehicleController.Drive() has an exact zero-DriveTorque branch
    # that applies CAWB braking, distinct from every nonzero motor-torque case.
    return throttle == 0.0


def _fit_source_mode_experts(
        data_by_run: list[tuple[list[Any], np.ndarray, np.ndarray,
                                np.ndarray, np.ndarray]],
        features_by_run: list[np.ndarray]) -> dict[str, Any]:
    experts = {}
    for mode_name, brake_mode in (("unity_zero_throttle_brake", True),
                                  ("unity_nonzero_motor_torque", False)):
        selected_features = []
        selected_targets = []
        selected_groups = []
        source_samples = 0
        for data, features in zip(data_by_run, features_by_run):
            samples, diagnostics, groups, _, _ = data
            mode = _source_mode(samples) == brake_mode
            support = ((diagnostics[:, 7] >= GATE_MIN_U_MPS) &
                       (diagnostics[:, 7] < GATE_MAX_U_MPS))
            active = mode & support
            source_samples += int(np.sum(mode))
            if np.any(active):
                selected_features.append(features[active])
                selected_targets.append(
                    np.asarray([sample.state[0] for sample in samples])[active] -
                    diagnostics[active, 7])
                selected_groups.append(groups[active])
        if sum(map(len, selected_targets)) < 30:
            raise ValueError(f"insufficient support for source expert {mode_name}")
        feature = np.concatenate(selected_features)
        target = np.concatenate(selected_targets)
        groups = np.concatenate(selected_groups)
        mean, scale, coefficient, intercept = base._fit(feature, target, groups)
        experts[mode_name] = {
            "brake_mode": brake_mode,
            "source_mode_samples": source_samples,
            "fit_samples_2_to_8mps": int(len(target)),
            "mean": mean,
            "scale": scale,
            "coefficient": coefficient,
            "intercept": intercept,
        }
    return experts


def _evaluate_source_mode_experts(
        label: str,
        data: tuple[list[Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        features: np.ndarray,
        experts: dict[str, Any]) -> dict[str, Any]:
    samples, diagnostics, groups, replay_pose, _ = data
    baseline_u = diagnostics[:, 7]
    truth_u = np.asarray([sample.state[0] for sample in samples])
    active_speed = ((baseline_u >= GATE_MIN_U_MPS) &
                    (baseline_u < GATE_MAX_U_MPS))
    brake = _source_mode(samples)
    correction = np.zeros(len(samples), dtype=np.float64)
    mode_support = {}
    for name, expert in experts.items():
        mask = active_speed & (brake == expert["brake_mode"])
        mode_support[name] = int(mask.sum())
        correction[mask] = (
            ((features[mask] - expert["mean"]) / expert["scale"])
            @ expert["coefficient"] + expert["intercept"])
    baseline_error = baseline_u - truth_u
    candidate_error = baseline_error + correction
    pose = base._pose_metrics(samples, groups, diagnostics, replay_pose, correction)
    per_group = []
    for group in dict.fromkeys(groups.tolist()):
        mask = groups == group
        before = base._metrics(baseline_error[mask])
        after = base._metrics(candidate_error[mask])
        per_group.append({"group": str(group),
                          "baseline_rmse_mps": before["rmse"],
                          "candidate_rmse_mps": after["rmse"],
                          "improved": after["rmse"] < before["rmse"]})
    return {
        "run": label,
        "source_mode_samples_2_to_8mps": mode_support,
        "forward_speed_baseline": base._metrics(baseline_error),
        "forward_speed_candidate": base._metrics(candidate_error),
        "pose": pose,
        "lap_macro": {"groups": len(per_group),
                      "improved": sum(item["improved"] for item in per_group),
                      "per_group": per_group},
    }


def _runtime_affine(expert: dict[str, Any]) -> dict[str, Any]:
    mean = np.asarray(expert["mean"], dtype=np.float64)
    scale = np.asarray(expert["scale"], dtype=np.float64)
    coefficient = np.asarray(expert["coefficient"], dtype=np.float64)
    weights = coefficient / scale
    intercept = float(expert["intercept"] - np.dot(weights, mean))
    return {"intercept_mps": intercept, "coefficients": weights.tolist()}


def _load_recorded_odom_run(raw: Path) -> tuple[list[Any], np.ndarray, np.ndarray,
                                                np.ndarray, np.ndarray]:
    """Load this run's actual Odom diagnostics instead of a replayed baseline."""
    capture = base.body.load_capture(raw)
    if (capture.aborted or capture.collision_count_start != 0
            or capture.collision_count_end != 0 or capture.timing_faults
            or capture.final_lap_count != 12):
        raise ValueError(f"capture failed clean 12-lap quality gate: {raw}")
    quality_stats = capture.phase_stream_stats or capture.stream_stats
    bad_streams = [
        name for name, (rate_hz, p95_dt_ms, max_gap_ms) in quality_stats.items()
        if rate_hz < 38.0 or p95_dt_ms > 35.0
        or max_gap_ms > (120.0 if name in base.body.COMMAND_STREAM_TOPICS else 60.0)
    ]
    if bad_streams:
        raise ValueError(f"capture failed 40 Hz stream quality gate: {bad_streams}")

    samples_by_stamp: dict[int, Any] = {}
    group_by_stamp: dict[int, str] = {}
    for sequence_index, sequence in enumerate(capture.sequences):
        label = capture.sequence_labels[sequence_index]
        for sample in sequence:
            stamp = int(sample.source_stamp_ns)
            samples_by_stamp.setdefault(stamp, sample)
            group_by_stamp.setdefault(stamp, label)
    stamps = sorted(samples_by_stamp)
    samples = [samples_by_stamp[stamp] for stamp in stamps]
    diagnostics = base._align_diagnostics(
        base._read_diagnostics(raw, "/odom/diagnostics"),
        stamps, "/odom/diagnostics")
    poses = base._read_pose(raw, "/odom")
    missing = [stamp for stamp in stamps if stamp not in poses]
    if missing:
        raise ValueError(f"recorded Odom misses {len(missing)}/{len(stamps)} poses")
    replay_pose = np.asarray([poses[stamp] for stamp in stamps], dtype=np.float64)
    groups = np.asarray([group_by_stamp[stamp] for stamp in stamps])
    if (diagnostics.shape != (len(stamps), base.DIAGNOSTIC_SIZE)
            or not np.isfinite(diagnostics).all()
            or not np.isfinite(replay_pose).all()):
        raise ValueError("recorded Odom contains missing/non-finite diagnostics or pose")
    return samples, diagnostics, groups, replay_pose, np.asarray(stamps, dtype=np.int64)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", nargs=3, action="append", required=True,
                        metavar=("ROLE", "RAW_BAG", "REPLAY_BAG"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--recorded-odom", action="store_true",
        help="use /odom and /odom/diagnostics already recorded in each raw bag",
    )
    args = parser.parse_args()
    train_pairs = []
    validation_pairs = []
    for role, raw, replay in args.pair:
        item = (role, Path(raw), Path(replay))
        (train_pairs if role == "train" else validation_pairs).append(item)
    if not train_pairs or not validation_pairs:
        raise ValueError("need both train and validation whole-run pairs")

    train_data = []
    train_features = []
    for label, raw, replay in train_pairs:
        loader = _load_recorded_odom_run if args.recorded_odom else base._load_run
        loaded = loader(raw) if args.recorded_odom else loader(raw, replay)
        samples, diagnostics, groups, pose, stamps = loaded
        run_id = raw.parent.parent.name
        groups = np.asarray([f"{run_id}:{group}" for group in groups])
        data = (samples, diagnostics, groups, pose, stamps)
        train_data.append(data)
        train_features.append(np.column_stack((
            base._features(diagnostics), _features(samples, diagnostics))))
    model = _fit_path(train_features, train_data)

    baseline_features = [base._features(data[1]) for data in train_data]
    baseline_model = _fit_path(baseline_features, train_data)

    # Select the position/speed tradeoff using only leave-one-complete-training-
    # run-out transfer between R01 and R03. R02/R04 remain untouched below.
    position_weights = (0.25, 1.0, 4.0, 16.0)
    training_cv: dict[str, Any] = {}
    selected_weights: dict[str, float] = {}
    for model_name, feature_runs in (("plain", baseline_features),
                                     ("unity_actuator", train_features)):
        scores = []
        for position_weight in position_weights:
            fold_scores = []
            for heldout_index in range(len(train_data)):
                fit_indices = [i for i in range(len(train_data))
                               if i != heldout_index]
                fold_model = _fit_path(
                    [feature_runs[i] for i in fit_indices],
                    [train_data[i] for i in fit_indices], position_weight)
                heldout = _evaluate_run(
                    "training_run_holdout", train_data[heldout_index],
                    feature_runs[heldout_index], fold_model)
                speed_ratio = (
                    heldout["forward_speed_candidate"]["rmse"] /
                    max(heldout["forward_speed_baseline"]["rmse"], 1.0e-9))
                pose_ratio = (
                    heldout["pose"]["residual_candidate"]["rmse"] /
                    max(heldout["pose"]["baseline"]["rmse"], 1.0e-9))
                fold_scores.append({
                    "heldout_run": train_pairs[heldout_index][1].parent.parent.name,
                    "relative_speed_rmse": speed_ratio,
                    "relative_position_rmse": pose_ratio,
                    "equal_relative_macro_score": 0.5 * (speed_ratio + pose_ratio),
                })
            mean_score = float(np.mean([
                item["equal_relative_macro_score"] for item in fold_scores]))
            scores.append({"position_loss_weight": position_weight,
                           "mean_equal_relative_macro_score": mean_score,
                           "folds": fold_scores})
        selected = min(scores, key=lambda item:
                       item["mean_equal_relative_macro_score"])
        selected_weights[model_name] = float(selected["position_loss_weight"])
        training_cv[model_name] = {
            "selection": "leave-one-complete-training-run-out; R01/R03 only",
            "selected_position_loss_weight": selected_weights[model_name],
            "scores": scores,
        }
    baseline_model = _fit_path(
        baseline_features, train_data, selected_weights["plain"])
    model = _fit_path(
        train_features, train_data, selected_weights["unity_actuator"])
    source_experts = _fit_source_mode_experts(train_data, train_features)

    validation_results = []
    for label, raw, replay in validation_pairs:
        loader = _load_recorded_odom_run if args.recorded_odom else base._load_run
        loaded = loader(raw) if args.recorded_odom else loader(raw, replay)
        samples, diagnostics, groups, pose, stamps = loaded
        data = (samples, diagnostics, groups, pose, stamps)
        augmented = np.column_stack((
            base._features(diagnostics), _features(samples, diagnostics)))
        plain = base._features(diagnostics)
        validation_results.append({
            "run": raw.parent.parent.name,
            "raw_bag": str(raw.resolve()),
            "raw_bag_sha256": _sha256(raw),
            "plain_sensor_model": _evaluate_run(
                raw.parent.parent.name, data, plain, baseline_model),
            "unity_actuator_augmented_model": _evaluate_run(
                raw.parent.parent.name, data, augmented, model),
            "unity_controller_source_mode_experts": _evaluate_source_mode_experts(
                raw.parent.parent.name, data, augmented, source_experts),
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "method": "source-grounded causal actuator feature ablation; pointwise residual fit",
        "training_policy": "complete R01/R03 training runs only; whole R02/R04 captures held out",
        "truth_policy": "simulator truth is only the offline u residual target and pose scoring label",
        "runtime_compliance": "feedback features are permitted /steering and /throttle topics; no restricted inputs",
        "unity_source_equations": {
            "nonzero_throttle_total_motor_torque_nm": "85.6 * throttle_feedback",
            "CAWD_motor_torque_per_wheel_nm": "85.6 * throttle_feedback / 4",
            "zero_throttle_branch": "0 motor torque and 85.6 Nm brake torque at each wheel",
            "measured_steering": "VehicleController.CurrentSteeringAngle; rate-limited center angle in radians",
            "kinematic_lateral_demand": "u^2 * tan(measured_center_steering) / 0.324 m",
        },
        "additional_features": list(SOURCE_FEATURE_NAMES),
        "training_only_position_weight_selection": training_cv,
        "selected_position_loss_weights": selected_weights,
        "source_mode_expert_support": {
            name: {
                "unity_zero_throttle_brake_branch": expert["brake_mode"],
                "training_samples_2_to_8mps": expert["fit_samples_2_to_8mps"],
                "source_mode_samples": expert["source_mode_samples"],
            }
            for name, expert in source_experts.items()},
        "fitted_source_mode_experts": {
            name: {
                "brake_mode": bool(expert["brake_mode"]),
                "mean": np.asarray(expert["mean"], dtype=float).tolist(),
                "scale": np.asarray(expert["scale"], dtype=float).tolist(),
                "coefficient": np.asarray(expert["coefficient"], dtype=float).tolist(),
                "intercept": float(expert["intercept"]),
            }
            for name, expert in source_experts.items()},
        "runtime_model": {
            "schema_version": 1,
            "feature_names": list(base.FEATURE_NAMES) + list(SOURCE_FEATURE_NAMES),
            "speed_gate_mps": [GATE_MIN_U_MPS, GATE_MAX_U_MPS],
            "branch_rule": "Unity CAWD: throttle feedback exactly zero selects zero-throttle brake expert; any nonzero feedback selects motor-torque expert",
            "unity_zero_throttle_brake": _runtime_affine(
                source_experts["unity_zero_throttle_brake"]),
            "unity_nonzero_motor_torque": _runtime_affine(
                source_experts["unity_nonzero_motor_torque"]),
            "training_runs": [raw.parent.parent.name
                              for _, raw, _ in train_pairs],
        },
        "training_runs": [{
            "run": raw.parent.parent.name,
            "raw_bag": str(raw.resolve()),
            "raw_bag_sha256": _sha256(raw),
            "samples": len(data[0]),
        } for (label, raw, replay), data in zip(train_pairs, train_data)],
        "validation_runs": validation_results,
        "decision": "not a runtime model; recursive observer replay is required before integration",
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "validation": [{
            "run": result["run"],
            "plain": {
                "u_rmse": result["plain_sensor_model"]["forward_speed_candidate"]["rmse"],
                "position_rmse": result["plain_sensor_model"]["pose"]["residual_candidate"]["rmse"],
            },
            "actuator": {
                "u_rmse": result["unity_actuator_augmented_model"]["forward_speed_candidate"]["rmse"],
                "position_rmse": result["unity_actuator_augmented_model"]["pose"]["residual_candidate"]["rmse"],
                "endpoint": result["unity_actuator_augmented_model"]["pose"]["residual_endpoint_error_m"],
                "lap_groups_improved": result["unity_actuator_augmented_model"]["lap_macro"]["improved"],
            },
            "unity_source_mode_experts": {
                "u_rmse": result["unity_controller_source_mode_experts"]["forward_speed_candidate"]["rmse"],
                "position_rmse": result["unity_controller_source_mode_experts"]["pose"]["residual_candidate"]["rmse"],
                "endpoint": result["unity_controller_source_mode_experts"]["pose"]["residual_endpoint_error_m"],
                "lap_groups_improved": result["unity_controller_source_mode_experts"]["lap_macro"]["improved"],
                "support": result["unity_controller_source_mode_experts"]["source_mode_samples_2_to_8mps"],
            },
        } for result in validation_results],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
