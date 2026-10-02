#!/usr/bin/env python3
"""Run short, held-out-state branches through the production racing stack.

Only branch initialization uses recorded history/pose. After reset, the MPC
and actuator receive synthetic sensor outputs from the frozen plant; future
simulator truth is not read. This is an offline diagnostic, not a validated
counterfactual simulator or a deployment path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from tools.racing.offline.controller.production_actuator import ProductionActuator
from tools.racing.offline.controller.production_mpc import ProductionMpc
from tools.racing.offline.localization.amcl_surrogate import load_surrogate
from tools.racing.offline.localization.production_odom import ProductionOdometry
from tools.racing.offline.sensors.imu import (
    ImuNoiseProfile,
    SyntheticImu,
)
from tools.racing.offline.sensors.rear_encoders import SyntheticRearEncoders
from tools.vehicle_dynamics_learning.offline_plant import (
    load_historical_gru_plant,
    load_rssm_teacher_plant,
)
from tools.vehicle_dynamics_learning.train_nssm import _load_dataset


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ARCHIVE = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "plant_teacher_mixed_dataset_full3d_fixed25_20261001/openplane_dynamics.npz")
DEFAULT_GRU = ROOT / "live_runs/derived_dynamics_learning_20260928/plant_teacher_mixed_gru_5s_20260930"
DEFAULT_AMCL_BANK = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/"
    "amcl_surrogate_train/amcl_training_residual_bank.npz")
DEFAULT_ODOM_LIB = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/"
    "production_odom_build/libproduction_odom.so")
DEFAULT_MPC_LIB = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/"
    "production_mpc_build/libproduction_mpc.so")
DEFAULT_MPC_YAML = ROOT / "f1tenth_mpc/config/mpc_competition.yaml"
DEFAULT_ACTUATOR_YAML = ROOT / "sdu_apex_autodrive/config/actuator_interface.yaml"
DEFAULT_TRAJECTORY = ROOT / (
    "f1tenth_planning/trajectories/autodrive_practice_20260924_b/"
    "autodrive_practice_20260924_b_mintime_raceline.csv")
DT_S = 0.025
STAMP_BASE_NS = 1_000_000_000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _rmse(values: np.ndarray) -> float | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return (float(np.sqrt(np.mean(finite * finite)))
            if len(finite) else None)


def _nanmax_abs(values: np.ndarray) -> float | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.max(np.abs(finite))) if len(finite) else None


def _unwrapped_progress_distance(progress: np.ndarray,
                                lap_length_m: float) -> float:
    values = np.asarray(progress, dtype=np.float64)
    if len(values) < 2 or not np.isfinite(values).all():
        return 0.0
    increments = np.diff(values)
    half_lap = 0.5 * float(lap_length_m)
    increments[increments < -half_lap] += float(lap_length_m)
    increments[increments > half_lap] -= float(lap_length_m)
    return float(np.sum(increments))


def _trajectory(path: Path) -> np.ndarray:
    rows = np.loadtxt(path, delimiter=",", comments="#", dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] < 6 or len(rows) < 3:
        raise ValueError(f"invalid production raceline: {path}")
    if not np.isfinite(rows[:, :6]).all() or np.any(np.diff(rows[:, 0]) <= 0):
        raise ValueError("raceline columns must be finite with increasing arc length")
    return rows


def _speed_at(rows: np.ndarray, progress_m: float,
              lap_length_m: float) -> float:
    s = float(progress_m) % float(lap_length_m)
    return float(np.interp(s, rows[:, 0], rows[:, 5]))


def _candidate_starts(data: dict[str, Any], split: str, history_steps: int,
                      rollout_steps: int, controller: ProductionMpc
                      ) -> list[dict[str, Any]]:
    groups: dict[int, list[tuple[int, int]]] = {}
    for seq_id, (start_value, end_value) in enumerate(data["bounds"]):
        run = int(data["seq_run"][seq_id])
        if (data["splits"][run] != split
                or not str(data["run_ids"][run]).startswith("practice_")):
            continue
        start, end = int(start_value), int(end_value)
        if end - start > history_steps + rollout_steps + 1:
            groups.setdefault(run, []).append((start, end))

    selected: list[dict[str, Any]] = []
    for run, sequences in sorted(groups.items()):
        candidates: dict[str, list[tuple[float, int, int, tuple[float, ...]]]] = {
            "straight": [], "turn": []}
        for sequence_start, sequence_end in sequences:
            first = sequence_start + history_steps - 1
            last = sequence_end - rollout_steps - 1
            if last < first:
                continue
            for index in range(first, last + 1, 4):
                sensor_start = index - history_steps + 1
                if not data["sensor_valid"][sensor_start:index + 1].all():
                    continue
                if (not np.isfinite(data["frames"][sensor_start:index + 1]).all()
                        or not np.isfinite(data["sensor_frames"][sensor_start:index + 1]).all()
                        or not np.isfinite(data["simulator_pose_xyyaw"][index]).all()):
                    continue
                speed = float(np.hypot(*data["frames"][index, :2]))
                steering = float(data["frames"][index, 3])
                abs_steering = abs(steering)
                if not 2.0 <= speed <= 12.0:
                    continue
                mode = ("straight" if abs_steering <= 0.12 else
                        "turn" if 0.12 < abs_steering <= 0.40 else None)
                if mode is None:
                    continue
                try:
                    projection = controller.project(
                        data["simulator_pose_xyyaw"][index])
                except ValueError:
                    continue
                if projection[3] > 0.30:
                    continue
                if mode == "straight":
                    score = abs(speed - 4.5) + 0.15 * abs_steering
                else:
                    score = -abs_steering + 0.03 * abs(speed - 4.5)
                candidates[mode].append((
                    score, index, sequence_start, projection))
        chosen: list[tuple[str, int, int, tuple[float, ...]]] = []
        for mode in ("straight", "turn"):
            if candidates[mode]:
                _, index, sequence_start, projection = min(
                    candidates[mode], key=lambda item: item[0])
                chosen.append((mode, index, sequence_start, projection))
        for mode, index, sequence_start, projection in chosen:
            selected.append({
                "run_index": run,
                "run_id": str(data["run_ids"][run]),
                "split": split,
                "mode": mode,
                "index": index,
                "sequence_start": sequence_start,
                "initial_projection": projection,
            })
    return selected


def _replay_initial_sensors(data: dict[str, Any], start: int, end: int,
                            odometry: ProductionOdometry,
                            encoders: SyntheticRearEncoders,
                            actuator: ProductionActuator,
                            initial_yaw: float) -> tuple[int, np.ndarray]:
    yaw = float(initial_yaw)
    previous_yaw_rate: float | None = None
    last_imu = None
    for ordinal, row_index in enumerate(range(start, end)):
        sensor = data["sensor_frames"][row_index]
        stamp_ns = STAMP_BASE_NS + ordinal * 25_000_000
        if ordinal == 0:
            encoders.reset(stamp_s=0.0)
            left_angle = right_angle = 0.0
        else:
            sample = encoders.step(float(sensor[2]), float(sensor[3]), DT_S)
            left_angle, right_angle = sample.left_angle_rad, sample.right_angle_rad
            assert previous_yaw_rate is not None
            yaw += 0.5 * (previous_yaw_rate + float(sensor[6])) * DT_S
        odometry.add_left_encoder(stamp_ns, left_angle)
        odometry.add_right_encoder(stamp_ns, right_angle)
        odometry.add_imu(stamp_ns, float(sensor[4]), float(sensor[5]),
                         float(sensor[6]), yaw)
        state = odometry.snapshot()
        if state is None or state.valid != 1.0:
            raise RuntimeError(f"production odometry did not produce a valid packet at row {row_index}")
        actuator.observe(stamp_ns, state.body_u_mps, float(sensor[4]))
        previous_yaw_rate = float(sensor[6])
        last_imu = sensor
    if last_imu is None:
        raise ValueError("empty sensor initialization history")
    return STAMP_BASE_NS + (end - start - 1) * 25_000_000, last_imu


def _run_branch(data: dict[str, Any], branch: dict[str, Any], *,
                plant, odometry: ProductionOdometry, mpc: ProductionMpc,
                actuator: ProductionActuator, localization,
                encoders: SyntheticRearEncoders, imu: SyntheticImu,
                trajectory: np.ndarray, mpc_max_speed: float,
                steering_limit_rad: float, horizon_steps: int
                ) -> dict[str, Any]:
    index = int(branch["index"])
    history_steps = plant.history_steps
    history_start = index - history_steps + 1
    if history_start < int(branch["sequence_start"]):
        raise ValueError("branch context crosses a sequence/reset boundary")
    history = data["frames"][history_start:index + 1]
    initial_pose = data["simulator_pose_xyyaw"][index].astype(np.float64)
    estimate = plant.reset(history, initial_pose)
    if not estimate.within_speed_domain:
        raise ValueError("branch start is outside the plant's supported speed domain")
    odometry.reset()
    actuator.reset()
    encoders.reset()
    stamp_ns, last_sensor = _replay_initial_sensors(
        data, history_start, index + 1, odometry, encoders, actuator,
        float(data["simulator_pose_xyyaw"][history_start, 2]))
    initial_odom = odometry.snapshot()
    if initial_odom is None or initial_odom.valid != 1.0:
        raise RuntimeError("production odometry has no valid branch-start estimate")
    # The production speed controller is reset at each branch; its causal
    # speed estimator is warmed with the recorded sensor-only context above.
    actuator.speed_controller.reset()
    actuator.observe(stamp_ns, initial_odom.body_u_mps, float(last_sensor[4]))

    mpc.reset()
    localization.reset(
        time_s=0.0,
        initial_plant_pose_xyyaw=initial_pose,
        rollout_seconds=horizon_steps * DT_S,
    )
    map_pose = localization.get_state()
    if map_pose is None:
        raise RuntimeError("AMCL surrogate did not initialize its observed pose")
    projection = mpc.project(map_pose)
    progress_m, lateral_error, heading_error, localization_distance, segment = projection
    initial_target = _speed_at(trajectory, progress_m, mpc.lap_length_m)

    frames = []
    domain_exit = None
    actual_command_history = data["frames"][index - 2:index + 1, 7]
    steering_command = float(actual_command_history[-1])
    delayed_1 = float(actual_command_history[-2])
    delayed_2 = float(actual_command_history[-3])
    previous_steering_rate = (steering_command - delayed_1) / DT_S
    previous_target_speed_rate = 0.0
    previous_target_speed = initial_target
    truth_segment = 2**64 - 1
    imu.reset(0.0, estimate.state[3:6], estimate.state[:3])

    for step_index in range(horizon_steps):
        odom = odometry.snapshot()
        if odom is None or odom.valid != 1.0:
            raise RuntimeError(f"production odometry invalid at branch step {step_index}")
        vehicle_state = plant.get_state()
        mpc_state = np.asarray((
            lateral_error, heading_error, odom.body_u_mps, odom.body_v_mps,
            odom.yaw_rate_radps, previous_target_speed, steering_command,
            delayed_1, delayed_2, vehicle_state[6], previous_steering_rate,
            previous_target_speed_rate,
        ), dtype=np.float64)
        cycle = mpc.step(mpc_state, progress_m, mpc_max_speed)
        actuator_command = actuator.tick(
            cycle.steering_command_rad, cycle.target_speed_mps, DT_S, True)
        steering_command_rad = actuator_command.steering_normalized * steering_limit_rad
        throttle_command = actuator_command.throttle_normalized
        plant_estimate = plant.step(steering_command_rad, throttle_command, DT_S)
        next_vehicle_state = plant_estimate.state
        if not plant_estimate.within_speed_domain:
            domain_exit = {
                "time_s": (step_index + 1) * DT_S,
                "speed_mps": float(np.hypot(
                    next_vehicle_state[3], next_vehicle_state[4])),
                "state": next_vehicle_state.tolist(),
                "steering_command_rad": steering_command_rad,
                "throttle_command_norm": throttle_command,
                "reason": "predicted_state_exceeded_race_speed_cap",
            }
            break
        encoder_sample = encoders.step(
            float(next_vehicle_state[8]), float(next_vehicle_state[9]), DT_S)
        imu_sample = imu.step(next_vehicle_state[3:6],
                              next_vehicle_state[:3], DT_S)
        stamp_ns += 25_000_000
        odometry.add_left_encoder(stamp_ns, encoder_sample.left_angle_rad)
        odometry.add_right_encoder(stamp_ns, encoder_sample.right_angle_rad)
        odometry.add_imu(
            stamp_ns, imu_sample.linear_acceleration_x_mps2,
            imu_sample.linear_acceleration_y_mps2,
            imu_sample.angular_velocity_z_radps,
            imu_sample.orientation_yaw_rad,
        )
        next_odom = odometry.snapshot()
        if next_odom is None or next_odom.valid != 1.0:
            raise RuntimeError(f"production odometry failed to assemble at step {step_index}")
        actuator.observe(stamp_ns, next_odom.body_u_mps,
                         imu_sample.linear_acceleration_x_mps2)
        next_localization = localization.step(
            (step_index + 1) * DT_S, next_vehicle_state[:3])
        if next_localization is not None:
            map_pose = next_localization
        projection = mpc.project(map_pose, int(segment), local_search_radius=160)
        progress_m, lateral_error, heading_error, localization_distance, segment = projection
        try:
            truth_projection = mpc.project(
                next_vehicle_state[:3], int(truth_segment),
                local_search_radius=240)
            truth_segment = truth_projection[4]
            truth_lateral_error, truth_heading_error, truth_track_distance = (
                truth_projection[1], truth_projection[2], truth_projection[3])
        except ValueError:
            truth_lateral_error = truth_heading_error = truth_track_distance = math.nan
        frames.append({
            "time_s": (step_index + 1) * DT_S,
            "progress_m": progress_m,
            "true_progress_m": truth_projection[0] if np.isfinite(truth_track_distance) else math.nan,
            "localization_lateral_error_m": lateral_error,
            "truth_lateral_error_m": truth_lateral_error,
            "localization_heading_error_rad": heading_error,
            "truth_heading_error_rad": truth_heading_error,
            "localization_track_distance_m": localization_distance,
            "true_track_distance_m": truth_track_distance,
            "localization_error_m": float(np.linalg.norm(
                map_pose[:2] - next_vehicle_state[:2])),
            "production_odom_u_mps": next_odom.body_u_mps,
            "plant_u_mps": next_vehicle_state[3],
            "production_odom_v_mps": next_odom.body_v_mps,
            "plant_v_mps": next_vehicle_state[4],
            "production_odom_yaw_rate_rps": next_odom.yaw_rate_radps,
            "plant_yaw_rate_rps": next_vehicle_state[5],
            "raw_wheel_speed_mps": (next_odom.wheel_raw_mps),
            "steering_command_rad": steering_command_rad,
            "throttle_command_norm": throttle_command,
            "target_speed_mps": cycle.target_speed_mps,
            "mpc_status": cycle.status,
            "solver_iterations": cycle.solver_iterations,
            "nonlinear_failure_stage": cycle.nonlinear_failure_stage,
            "nonlinear_failure_reason": cycle.nonlinear_failure_reason,
            "best_effort_action": cycle.best_effort_action,
            "residual_candidate": cycle.residual_candidate,
            "rejection_speed_guard": cycle.rejection_speed_guard,
            "rti2_triggered": cycle.rti2_triggered,
            "plant_support_distance": (plant_estimate.support_distance
                                        if plant_estimate.support_distance is not None
                                        else math.nan),
        })
        delayed_2, delayed_1 = delayed_1, steering_command
        steering_command = cycle.steering_command_rad
        previous_steering_rate = cycle.steering_rate_radps
        previous_target_speed_rate = cycle.target_speed_rate_mps2
        previous_target_speed = cycle.target_speed_mps

    if not frames and domain_exit is None:
        raise RuntimeError("closed-loop branch produced no steps")
    columns = ({name: np.asarray([row[name] for row in frames])
                for name in frames[0]} if frames else {})
    distance = (_unwrapped_progress_distance(
        columns["progress_m"], mpc.lap_length_m) if frames else 0.0)
    status_counts = {
        "accepted_optimal": int(np.count_nonzero(columns["mpc_status"] == 0)) if frames else 0,
        "accepted_degraded": int(np.count_nonzero(columns["mpc_status"] == 1)) if frames else 0,
        "rejected_input": int(np.count_nonzero(columns["mpc_status"] == 2)) if frames else 0,
        "rejected_solver": int(np.count_nonzero(columns["mpc_status"] == 3)) if frames else 0,
        "rejected_residual": int(np.count_nonzero(columns["mpc_status"] == 4)) if frames else 0,
        "rejected_regularization": int(np.count_nonzero(columns["mpc_status"] == 5)) if frames else 0,
        "rejected_nonlinear_rollout": int(np.count_nonzero(columns["mpc_status"] == 6)) if frames else 0,
    }
    reason_names = {
        1: "invalid_input", 2: "invalid_model", 3: "command_limit",
        4: "state_limit", 5: "corridor",
    }
    failure_reason_counts = ({
        reason_names[int(reason)]: int(np.count_nonzero(
            columns["nonlinear_failure_reason"] == reason))
        for reason in np.unique(columns["nonlinear_failure_reason"])
        if int(reason) in reason_names
    } if frames else {})
    known_status_total = sum(status_counts.values())
    if known_status_total != len(frames):
        status_counts["other_or_unmapped"] = len(frames) - known_status_total
    support_values = (columns.get("plant_support_distance", np.asarray([]))
                      if columns else np.asarray([]))
    finite_support = support_values[np.isfinite(support_values)]
    return {
        "run_id": branch["run_id"],
        "run_family_split": f"{branch['split']} practice branch",
        "mode": branch["mode"],
        "start_index": index,
        "start_speed_mps": float(np.hypot(*history[-1, :2])),
        "start_steering_rad": float(history[-1, 3]),
        "requested_branch_seconds": horizon_steps * DT_S,
        "branch_seconds": len(frames) * DT_S,
        "step_count": len(frames),
        "terminated_out_of_domain": domain_exit is not None,
        "out_of_domain_event": domain_exit,
        "progress_m": distance,
        "lateral_error_rmse_m": (_rmse(columns["truth_lateral_error_m"])
                                 if frames else None),
        "max_abs_truth_lateral_error_m": (_nanmax_abs(
            columns["truth_lateral_error_m"]) if frames else None),
        "mean_localization_error_m": (float(np.mean(
            columns["localization_error_m"])) if frames else None),
        "odom_body_u_rmse_mps": (_rmse(
            columns["production_odom_u_mps"] - columns["plant_u_mps"])
            if frames else None),
        "odom_body_v_rmse_mps": (_rmse(
            columns["production_odom_v_mps"] - columns["plant_v_mps"])
            if frames else None),
        "odom_yaw_rate_rmse_radps": (_rmse(
            columns["production_odom_yaw_rate_rps"]
            - columns["plant_yaw_rate_rps"]) if frames else None),
        "mpc_status_counts": status_counts,
        "mpc_action_count": len(frames) + int(domain_exit is not None),
        "mpc_non_optimal_status_steps": int(len(frames) - status_counts["accepted_optimal"]),
        "mpc_best_effort_steps": int(np.count_nonzero(columns["best_effort_action"])) if frames else 0,
        "mpc_residual_candidate_steps": int(np.count_nonzero(columns["residual_candidate"])) if frames else 0,
        "mpc_speed_guard_steps": int(np.count_nonzero(columns["rejection_speed_guard"])) if frames else 0,
        "mpc_nonlinear_failure_steps": int(np.count_nonzero(
            columns["nonlinear_failure_stage"] >= 0)) if frames else 0,
        "mpc_rollout_failure_reason_counts": failure_reason_counts,
        "minimum_support_distance": (float(np.min(finite_support))
                                     if len(finite_support) else None),
        "maximum_support_distance": (float(np.max(finite_support))
                                     if len(finite_support) else None),
        "initial_localization_distance_to_raceline_m": float(
            branch["initial_projection"][3]),
        "trace": columns,
    }


def run(dataset_path: Path, output_dir: Path, *, split: str = "validation",
        branch_seconds: float = 2.0,
        plant_checkpoint: Path | None = None) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {output_dir}")
    if split not in ("validation", "test", "unseen_practice"):
        raise ValueError(
            "closed-loop branch starts must come from validation, test, "
            "or the isolated unseen-practice split")
    if not np.isfinite(branch_seconds) or not 0.5 <= branch_seconds <= 2.0:
        raise ValueError("branch horizon must be in [0.5, 2.0] seconds")
    output_dir.mkdir(parents=True, exist_ok=True)
    data = _load_dataset(dataset_path)
    if split == "unseen_practice":
        unseen_ids = [str(data["run_ids"][run])
                      for run in np.flatnonzero(data["splits"] == split)]
        if not unseen_ids or any(not run_id.startswith("practice_unseen_")
                                 for run_id in unseen_ids):
            raise ValueError(
                "unseen_practice accepts only explicitly named unseen practice runs")
    horizon_steps = round(branch_seconds / DT_S)
    plant = (load_rssm_teacher_plant([plant_checkpoint], dataset_path,
                                     device="cpu",
                                     max_supported_speed_mps=12.0)
             if plant_checkpoint is not None
             else load_historical_gru_plant(
                 DEFAULT_GRU, dataset_path, device="cpu",
                 max_supported_speed_mps=12.0))
    with ProductionMpc(DEFAULT_MPC_LIB, DEFAULT_MPC_YAML,
                       DEFAULT_TRAJECTORY) as selector:
        starts = _candidate_starts(
            data, split, plant.history_steps, horizon_steps, selector)
    if not starts:
        raise RuntimeError(f"no valid held-out practice branch starts for {split}")

    mpc_params = yaml.safe_load(DEFAULT_MPC_YAML.read_text(encoding="utf-8"))["/**"]["ros__parameters"]
    actuator_params = yaml.safe_load(DEFAULT_ACTUATOR_YAML.read_text(encoding="utf-8"))["autodrive_actuator_interface"]["ros__parameters"]
    trajectory = _trajectory(DEFAULT_TRAJECTORY)
    noise = ImuNoiseProfile.from_training_data(data, samples_per_run=5000)
    branches = []
    # Each branch owns its mutable production observer/controller state.
    for branch_number, branch in enumerate(starts):
        odometry = ProductionOdometry(
            DEFAULT_ODOM_LIB, ROOT / "f1tenth_localization/config/sensor_odometry.yaml")
        mpc = ProductionMpc(DEFAULT_MPC_LIB, DEFAULT_MPC_YAML,
                            DEFAULT_TRAJECTORY)
        actuator = ProductionActuator(DEFAULT_ACTUATOR_YAML)
        localization = load_surrogate(
            DEFAULT_AMCL_BANK, seed=20261014 + branch_number)
        encoders = SyntheticRearEncoders()
        imu = SyntheticImu(noise, seed=20261015 + branch_number)
        try:
            result = _run_branch(
                data, branch, plant=plant, odometry=odometry, mpc=mpc,
                actuator=actuator, localization=localization,
                encoders=encoders, imu=imu, trajectory=trajectory,
                mpc_max_speed=min(
                    float(mpc_params.get("max_speed_mps", 16.0)), 12.0),
                steering_limit_rad=float(actuator_params["max_steering_angle_rad"]),
                horizon_steps=horizon_steps,
            )
        finally:
            odometry.close()
            mpc.close()
        trace = result.pop("trace")
        trace_path = output_dir / (
            f"branch_{branch_number:02d}_{branch['run_id']}_{branch['mode']}.npz")
        np.savez_compressed(trace_path, **trace)
        result["trace_file"] = trace_path.name
        branches.append(result)

    run_ids = sorted({row["run_id"] for row in branches})
    report = {
        "schema_version": 1,
        "purpose": (f"{horizon_steps * DT_S:g}-second held-out-state "
                    "production-stack surrogate screen"),
        "validity": "offline diagnostic only; no claim of counterfactual truth or collision fidelity",
        "split": split,
        "independent_run_count": len(run_ids),
        "branch_count": len(branches),
        "branch_seconds": horizon_steps * DT_S,
        "physics_timebase_s": DT_S,
        "race_domain_constraints": {
            "speed_cap_mps": 12.0,
            "mpc_target_speed_cap_mps": 12.0,
            "predicted_state_outside_cap": (
                "terminate before forwarding it to synthetic sensors"),
        },
        "plant": (f"frozen prior-only RSSM: {plant_checkpoint.resolve()}"
                  if plant_checkpoint is not None else
                  "frozen historical mixed GRU; command-only after context reset"),
        "production_components": ["exact repository sensor odometry C++ core",
                                  "exact repository MPC RTI C core and competition YAML",
                                  "exact repository longitudinal actuator controller"],
        "localization": "train-practice empirical residual/latency trace; truth only anchors branch initialization and scores offline",
        "sensor_generation": {
            "rear_encoders": "rear wheel surface speed integrated at 0.059 m radius then quantized to 16 PPR x 120 conversion",
            "imu": "causal COM acceleration and yaw rate from plant; residuals resampled from training runs",
            "time": "exact 25 ms simulator packet cadence; no receipt-time jitter inserted",
            "orientation_yaw": "ideal plant yaw in the synthetic IMU; this is an optimistic simplification",
        },
        "controller_inputs": "synthetic sensor-derived production odom and empirical localization pose only; no future simulator truth",
        "initialization_limitations": [
            "MPC memory and speed-controller integral are reset per branch; the speed-state estimator is warmed only from recorded causal sensor context.",
            "Plant is a development-only command-driven prior rollout and predicts the archived motion-state target.",
            "AMCL uncertainty bank has five independent training practice runs; it is not a calibrated population model.",
            "No wall/collision geometry is simulated, so path clearance is diagnostic rather than a safety validation.",
        ],
        "artifact_hashes": {
            "dataset": _sha256(dataset_path),
            "plant_checkpoint": _sha256(
                plant_checkpoint if plant_checkpoint is not None
                else DEFAULT_GRU / "member_00.pt"),
            "amcl_residual_bank": _sha256(DEFAULT_AMCL_BANK),
            "mpc_yaml": _sha256(DEFAULT_MPC_YAML),
            "actuator_yaml": _sha256(DEFAULT_ACTUATOR_YAML),
            "trajectory": _sha256(DEFAULT_TRAJECTORY),
        },
        "branches": branches,
    }
    (output_dir / "closed_loop_branch_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test", "unseen_practice"),
                        default="validation")
    parser.add_argument("--branch-seconds", type=float, default=2.0)
    parser.add_argument("--plant-checkpoint", type=Path)
    args = parser.parse_args()
    report = run(args.dataset, args.output, split=args.split,
                 branch_seconds=args.branch_seconds,
                 plant_checkpoint=args.plant_checkpoint)
    print(json.dumps({
        "independent_runs": report["independent_run_count"],
        "branches": report["branch_count"],
        "output": str(args.output.resolve()),
        "mean_progress_m": float(np.mean(
            [row["progress_m"] for row in report["branches"]])),
        "best_effort_steps": sum(row["mpc_best_effort_steps"]
                                  for row in report["branches"]),
        "non_optimal_status_steps": sum(row["mpc_non_optimal_status_steps"]
                                         for row in report["branches"]),
        "nonlinear_failure_steps": sum(row["mpc_nonlinear_failure_steps"]
                                        for row in report["branches"]),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
