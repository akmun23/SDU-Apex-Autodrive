"""Command-only effective dynamics state-space teacher for offline simulation.

This is a track-independent vehicle model. It learns aggregate body and rear-
wheel accelerations from simulator labels while keeping body-frame transport,
actuator response, wheel integration, and pose integration explicit. It does
not use the rejected four-wheel tire-force model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


DT_S = 0.025
HISTORY_STEPS = 80
LATENT_SIZE = 32
PHYSICAL_STATE_NAMES = (
    "u_com_mps", "v_com_mps", "yaw_rate_rps", "steering_actual_rad",
    "throttle_feedback_norm", "rear_left_surface_speed_mps",
    "rear_right_surface_speed_mps",
)
ROLL_STATE_NAMES = ("imu_roll_rad", "imu_roll_rate_rps")
COMMAND_NAMES = ("steering_command_rad", "throttle_command_norm")
RAW_ENCODER_HISTORY_NAMES = (
    "raw_encoder_left_surface_mps",
    "raw_encoder_right_surface_mps",
    "raw_encoder_valid",
)
WHEEL_INNOVATION_HISTORY_NAMES = (
    "left_encoder_minus_filtered_mps",
    "right_encoder_minus_filtered_mps",
    "raw_encoder_valid",
)
HISTORY_NAMES = PHYSICAL_STATE_NAMES + COMMAND_NAMES
ACCELERATION_NAMES = (
    "ax_effective_mps2", "ay_effective_mps2", "yaw_accel_rps2",
    "rear_left_surface_accel_mps2", "rear_right_surface_accel_mps2",
)
REAR_AXLE_TO_COM_M = 0.15532
REAR_TRACK_WIDTH_M = 0.236
WHEEL_DYNAMICS_MODES = ("surface_acceleration", "contact_slip")


def acceleration_target_names(wheel_dynamics_mode: str) -> tuple[str, ...]:
    if wheel_dynamics_mode == "surface_acceleration":
        return ACCELERATION_NAMES
    if wheel_dynamics_mode == "contact_slip":
        return ("ax_effective_mps2", "ay_effective_mps2",
                "yaw_accel_rps2", "rear_left_slip_accel_mps2",
                "rear_right_slip_accel_mps2")
    raise ValueError(f"unknown wheel dynamics mode: {wheel_dynamics_mode}")


@dataclass(frozen=True)
class ActuatorChannel:
    """Train-fitted first-order command-to-feedback response."""

    delay_steps: int
    alpha: float

    def __post_init__(self) -> None:
        if self.delay_steps not in (0, 1):
            raise ValueError("actuator delay must be zero or one 25 ms sample")
        if not np.isfinite(self.alpha) or not 0.0 <= self.alpha <= 1.0:
            raise ValueError("actuator alpha must be finite and in [0, 1]")


@dataclass(frozen=True)
class ActuatorFit:
    steering: ActuatorChannel
    throttle: ActuatorChannel
    diagnostics: dict[str, Any]


def physical_state_from_dataset(
        data: dict[str, Any],
        wheel_state_source: str = "filtered_odometry") -> np.ndarray:
    """Build EDSSM state from truth body motion and measured rear/actuator data.

    The bridge rigid-state velocity/angular-velocity channels are already in
    the vehicle body frame (confirmed against aligned /odom rear-axle twist).
    Linear velocity is at the COM. Steering/throttle feedback and wheel speeds
    remain independent state channels. Wheel speed defaults to the existing
    filtered odometry signal; an explicit sidecar option selects packet-aligned
    raw encoder rates as the physical wheel state.
    """
    rigid = np.asarray(data["simulator_rigid_state"], dtype=np.float64)
    frames = np.asarray(data["frames"], dtype=np.float64)
    if rigid.shape != (len(frames), 13) or frames.shape[1] != 9:
        raise ValueError("dataset rigid-state/frame shapes do not match EDSSM")
    valid = np.isfinite(rigid).all(axis=1)
    if not np.all(valid):
        raise ValueError("EDSSM requires complete simulator rigid-state labels")
    if wheel_state_source not in ("filtered_odometry", "raw_encoder"):
        raise ValueError("wheel state source must be filtered_odometry or raw_encoder")
    wheel_state = frames[:, 5:7].astype(np.float64, copy=True)
    raw_wheels = data.get("encoder_raw_surface_mps")
    raw_valid = data.get("encoder_raw_valid")
    if wheel_state_source == "raw_encoder":
        if raw_wheels is None or raw_valid is None:
            raise ValueError("raw encoder state was requested but no sidecar exists")
        raw_wheels = np.asarray(raw_wheels, dtype=np.float64)
        raw_valid = np.asarray(raw_valid, dtype=bool)
        if (raw_wheels.shape != (len(frames), 2)
                or raw_valid.shape != (len(frames),)
                or not np.isfinite(raw_wheels[raw_valid]).all()):
            raise ValueError("raw encoder state sidecar has invalid shape/values")
        # Invalid intervals retain the production filtered estimate. This is
        # a causal history-only initialization/fill value; model targets and
        # wheel-error scores must still use raw_valid to avoid treating it as
        # a raw encoder label.
        wheel_state[raw_valid] = raw_wheels[raw_valid]
    physical = np.column_stack((
        rigid[:, 7], rigid[:, 8], rigid[:, 12],
        frames[:, 3], frames[:, 4], wheel_state[:, 0], wheel_state[:, 1],
    )).astype(np.float32)
    if not np.isfinite(physical).all():
        raise ValueError("EDSSM physical state contains non-finite values")
    return physical


def raw_encoder_history_features(data: dict[str, Any]) -> np.ndarray:
    """Return causal raw wheel history plus validity, filtered fill on gaps.

    Raw encoder rates are observed history only. Invalid intervals use the
    already-available filtered wheel estimate and are explicitly marked; no
    value from this array is used as a future rollout input or target.
    """
    frames = np.asarray(data["frames"], dtype=np.float64)
    raw = data.get("encoder_raw_surface_mps")
    valid = data.get("encoder_raw_valid")
    if raw is None or valid is None:
        raise ValueError("raw encoder history requires its verified sidecar")
    raw = np.asarray(raw, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if (raw.shape != (len(frames), 2) or valid.shape != (len(frames),)
            or not np.isfinite(raw[valid]).all()):
        raise ValueError("raw encoder history sidecar has invalid shape/values")
    frame_split = np.full(len(frames), "excluded", dtype="U16")
    for (start_raw, end_raw), run_raw in zip(data["bounds"], data["seq_run"]):
        start, end, run = int(start_raw), int(end_raw), int(run_raw)
        frame_split[start:end] = str(data["splits"][run])
    permitted = np.isin(frame_split, ("train", "validation", "unseen_practice"))
    usable_raw = valid & permitted
    wheel_history = np.zeros((len(frames), 2), dtype=np.float64)
    wheel_history[permitted] = frames[permitted, 5:7]
    wheel_history[usable_raw] = raw[usable_raw]
    return np.column_stack((wheel_history, usable_raw.astype(np.float64))).astype(
        np.float32)


def wheel_innovation_history_features(data: dict[str, Any]) -> np.ndarray:
    """Return raw-minus-filtered wheel innovations and source validity."""
    frames = np.asarray(data["frames"], dtype=np.float64)
    raw = data.get("encoder_raw_surface_mps")
    valid = data.get("encoder_raw_valid")
    if raw is None or valid is None:
        raise ValueError("wheel innovation history requires its encoder sidecar")
    raw = np.asarray(raw, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if (raw.shape != (len(frames), 2) or valid.shape != (len(frames),)
            or not np.isfinite(raw[valid]).all()):
        raise ValueError("wheel innovation sidecar has invalid shape/values")
    frame_split = np.full(len(frames), "excluded", dtype="U16")
    for (start_raw, end_raw), run_raw in zip(data["bounds"], data["seq_run"]):
        start, end, run = int(start_raw), int(end_raw), int(run_raw)
        frame_split[start:end] = str(data["splits"][run])
    permitted = np.isin(frame_split, ("train", "validation", "unseen_practice"))
    usable = valid & permitted
    innovation = np.zeros((len(frames), 2), dtype=np.float64)
    innovation[usable] = raw[usable] - frames[usable, 5:7]
    return np.column_stack((innovation, usable.astype(np.float64))).astype(
        np.float32)


def append_roll_state(data: dict[str, Any],
                      physical_state: np.ndarray) -> np.ndarray:
    """Append measured initial roll/rate labels for a recursive plant state.

    The values are training/initial-condition labels only. The transition
    model predicts roll rate and integrates roll after rollout begins.
    """
    state = np.asarray(physical_state, dtype=np.float32)
    attitude = data.get("imu_attitude_frames")
    valid = data.get("imu_attitude_valid")
    if (state.ndim != 2 or state.shape[1] != len(PHYSICAL_STATE_NAMES)
            or attitude is None or valid is None):
        raise ValueError("roll state requires aligned physical and IMU attitude data")
    attitude = np.asarray(attitude, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    if attitude.shape != (len(state), 4) or valid.shape != (len(state),):
        raise ValueError("IMU roll/rate labels do not align with plant state")
    selected = attitude[:, (0, 2)].copy()
    finite = valid & np.isfinite(selected).all(axis=1)
    if not finite.any():
        raise ValueError("roll-state training has no valid roll/rate labels")
    for start_raw, end_raw in data["bounds"]:
        start, end = int(start_raw), int(end_raw)
        good = np.flatnonzero(finite[start:end])
        if not len(good):
            raise ValueError("a plant sequence has no valid roll/rate labels")
        first = start + int(good[0])
        selected[start:first] = selected[first]
        last = selected[first].copy()
        for index in range(first, end):
            if finite[index]:
                last = selected[index].copy()
            else:
                selected[index] = last
    return np.column_stack((state, selected)).astype(np.float32, copy=False)


def generalized_acceleration_targets(
        physical_state: np.ndarray, sequence_bounds: np.ndarray,
        dt_s: np.ndarray,
        wheel_dynamics_mode: str = "surface_acceleration") -> np.ndarray:
    """Finite-difference effective COM accelerations without crossing resets.

    Uses the handoff convention: du/dt = ax + r*v and
    dv/dt = ay - r*u. Acceleration rows align with the transition beginning
    at each source index; the final row of each sequence is NaN.
    """
    state = np.asarray(physical_state, dtype=np.float64)
    if wheel_dynamics_mode not in WHEEL_DYNAMICS_MODES:
        raise ValueError("unknown rear-wheel dynamics target mode")
    bounds = np.asarray(sequence_bounds, dtype=np.int64)
    intervals = np.asarray(dt_s, dtype=np.float64)
    if state.ndim != 2 or state.shape[1] != len(PHYSICAL_STATE_NAMES):
        raise ValueError("physical state must have shape (N, 7)")
    if bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError("sequence bounds must have shape (M, 2)")
    if intervals.shape != (len(state),):
        raise ValueError("dt array must align with physical state")
    result = np.full((len(state), len(ACCELERATION_NAMES)), np.nan,
                     dtype=np.float32)
    for start_raw, end_raw in bounds:
        start, end = int(start_raw), int(end_raw)
        if (start < 0 or end > len(state) or end - start < 2
                or not np.isfinite(state[start:end]).all()):
            raise ValueError("invalid state sequence for acceleration labels")
        dt = intervals[start + 1:end]
        if (not np.isfinite(dt).all()
                or not np.allclose(dt, DT_S, rtol=0.0, atol=1e-7)):
            raise ValueError("EDSSM labels require the fixed 25 ms timebase")
        current = state[start:end - 1]
        following = state[start + 1:end]
        derivative = (following[:, :3] - current[:, :3]) / dt[:, None]
        u, v, yaw_rate = current[:, 0], current[:, 1], current[:, 2]
        result[start:end - 1, 0] = derivative[:, 0] - yaw_rate * v
        result[start:end - 1, 1] = derivative[:, 1] + yaw_rate * u
        result[start:end - 1, 2] = derivative[:, 2]
        if wheel_dynamics_mode == "surface_acceleration":
            result[start:end - 1, 3:5] = (
                following[:, 5:7] - current[:, 5:7]) / dt[:, None]
        else:
            half_track = REAR_TRACK_WIDTH_M / 2.0
            current_contact = np.column_stack((
                current[:, 0] - half_track * current[:, 2],
                current[:, 0] + half_track * current[:, 2]))
            following_contact = np.column_stack((
                following[:, 0] - half_track * following[:, 2],
                following[:, 0] + half_track * following[:, 2]))
            current_slip = current[:, 5:7] - current_contact
            following_slip = following[:, 5:7] - following_contact
            result[start:end - 1, 3:5] = (
                following_slip - current_slip) / dt[:, None]
    return result


def acceleration_targets_from_dataset(data: dict[str, Any],
                                      physical_state: np.ndarray,
                                      body_source: str = "state_derivative",
                                      wheel_dynamics_mode: str = "surface_acceleration"
                                      ) -> np.ndarray:
    """Build body/yaw/wheel targets, optionally using validated packet ax/ay.

    ``packet_interval_mean`` is a training-label option only. The source
    acceleration is never consumed by a model rollout.
    """
    if body_source not in ("state_derivative", "packet_interval_mean"):
        raise ValueError("unknown EDSSM body acceleration target source")
    state = np.asarray(physical_state, dtype=np.float64)
    targets = generalized_acceleration_targets(
        state, data["bounds"], data["dt_s"], wheel_dynamics_mode)
    if body_source == "state_derivative":
        return targets
    raw = data.get("simulator_linear_acceleration")
    if raw is None:
        raise ValueError("packet interval-mean labels require simulator acceleration")
    acceleration = np.asarray(raw, dtype=np.float64)
    if acceleration.shape != (len(state), 3):
        raise ValueError("simulator acceleration labels do not align with state")
    for start_raw, end_raw in data["bounds"]:
        start, end = int(start_raw), int(end_raw)
        if (not np.allclose(data["dt_s"][start + 1:end], DT_S,
                            rtol=0.0, atol=1e-7)
                or not np.isfinite(acceleration[start:end]).all()):
            raise ValueError("packet acceleration labels need complete 25 ms sequences")
        targets[start:end - 1, :2] = 0.5 * (
            acceleration[start:end - 1, :2]
            + acceleration[start + 1:end, :2])
    return targets


def fit_roll_oscillator(data: dict[str, Any], physical_state: np.ndarray
                       ) -> tuple[np.ndarray, dict[str, Any]]:
    """Identify a training-run roll oscillator driven by body lateral accel.

    The fitted continuous-time relation is
      roll_rate_dot = c_ay * ay + c_roll * roll + c_rate * roll_rate.
    Only train-split transitions are used. At rollout, ``ay`` must be the
    model's prediction; future IMU attitude is never an input.
    """
    attitude = data.get("imu_attitude_frames")
    valid = data.get("imu_attitude_valid")
    if attitude is None or valid is None:
        raise ValueError("roll oscillator fitting requires simulator IMU labels")
    attitude = np.asarray(attitude, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    state = np.asarray(physical_state, dtype=np.float64)
    if (attitude.shape != (len(state), 4) or valid.shape != (len(state),)
            or state.shape[1] != len(PHYSICAL_STATE_NAMES)):
        raise ValueError("roll oscillator labels do not align with body state")
    targets = generalized_acceleration_targets(
        state, data["bounds"], data["dt_s"])
    splits = np.asarray(data["splits"]).astype(str)
    rows, values = [], []
    selected_runs: set[int] = set()
    for (start_raw, end_raw), run_raw in zip(data["bounds"], data["seq_run"]):
        start, end, run = int(start_raw), int(end_raw), int(run_raw)
        if splits[run] != "train":
            continue
        if not np.allclose(data["dt_s"][start + 1:end], DT_S,
                           rtol=0.0, atol=1e-7):
            raise ValueError("roll oscillator requires exact 25 ms training data")
        indices = np.arange(start, end - 1, dtype=np.int64)
        mask = (valid[indices] & valid[indices + 1]
                & np.isfinite(attitude[indices][:, (0, 2)]).all(axis=1)
                & np.isfinite(attitude[indices + 1, 2])
                & np.isfinite(targets[indices, 1]))
        indices = indices[mask]
        if len(indices):
            rows.append(np.column_stack((targets[indices, 1],
                                         attitude[indices, 0],
                                         attitude[indices, 2])))
            values.append((attitude[indices + 1, 2]
                           - attitude[indices, 2]) / DT_S)
            selected_runs.add(run)
    if len(selected_runs) < 2 or not rows:
        raise ValueError("insufficient whole training runs for roll oscillator")
    features = np.concatenate(rows)
    target = np.concatenate(values)
    coefficients, _, rank, singular_values = np.linalg.lstsq(
        features, target, rcond=None)
    if (rank != 3 or not np.isfinite(coefficients).all()
            or coefficients[1] >= 0.0 or coefficients[2] >= 0.0):
        raise ValueError("training data did not identify a stable roll oscillator")
    residual = features @ coefficients - target
    return coefficients.astype(np.float32), {
        "equation": "roll_rate_dot = c_ay*predicted_ay + c_roll*roll + c_rate*roll_rate",
        "coefficients": coefficients.tolist(),
        "training_transition_count": int(len(target)),
        "training_run_count": int(len(selected_runs)),
        "training_run_ids": sorted(np.asarray(data["run_ids"]).astype(str)[
                                    list(sorted(selected_runs))].tolist()),
        "training_roll_rate_acceleration_rmse_rps2": float(
            np.sqrt(np.mean(residual ** 2))),
        "design_matrix_rank": int(rank),
        "design_matrix_singular_values": singular_values.tolist(),
        "fit_split": "train only",
    }


def fit_actuator_dynamics(data: dict[str, Any]) -> ActuatorFit:
    """Fit one-step actuator delay/response on training runs only.

    For each actuator and candidate delay d∈{0,1}, fit
    y[k+1] = y[k] + alpha*(command[k-d]-y[k]) by least squares. The fit loss
    gives equal weight to sequences within condition, conditions within run,
    runs within family, then applies the dataset's registered family weights.
    This prevents a long legacy capture from dominating actuator parameters.
    """
    frames = np.asarray(data["frames"], dtype=np.float64)
    bounds = np.asarray(data["bounds"], dtype=np.int64)
    sequence_runs = np.asarray(data["seq_run"], dtype=np.int32)
    splits = np.asarray(data["splits"]).astype(str)
    run_ids = np.asarray(data["run_ids"]).astype(str)
    families = np.asarray(
        data.get("training_families", data.get("run_families"))).astype(str)
    sequence_conditions = data.get("sequence_condition_id")
    if sequence_conditions is None:
        sequence_conditions = np.arange(len(bounds), dtype=np.int32)
    sequence_conditions = np.asarray(sequence_conditions, dtype=np.int64)
    family_weights: dict[str, float] = {}
    family_names = data.get("training_family_names")
    family_probabilities = data.get("training_family_probabilities")
    if family_names is not None and family_probabilities is not None:
        family_weights = dict(zip(
            np.asarray(family_names).astype(str).tolist(),
            np.asarray(family_probabilities, dtype=np.float64).tolist()))
    train_families = sorted(set(families[splits == "train"].tolist()))
    family_weights = {name: family_weights.get(name, 0.0)
                      for name in train_families}
    if sum(family_weights.values()) <= 0.0:
        family_weights = {name: 1.0 for name in train_families}
    normalizer = sum(family_weights.values())
    family_weights = {name: value / normalizer
                      for name, value in family_weights.items()}
    if frames.ndim != 2 or frames.shape[1] != 9:
        raise ValueError("actuator fitting requires the 9-channel frame layout")
    all_fits: dict[str, Any] = {}
    selected: dict[str, ActuatorChannel] = {}
    for label, feedback_col, command_col in (
            ("steering", 3, 7), ("throttle", 4, 8)):
        candidates = []
        for delay in (0, 1):
            records = []
            for sequence_id, ((start_raw, end_raw), run_raw) in enumerate(
                    zip(bounds, sequence_runs)):
                start, end, run = int(start_raw), int(end_raw), int(run_raw)
                if end - start <= delay + 1:
                    continue
                current_index = np.arange(start, end - 1)
                target_index = current_index + 1
                command_index = current_index - delay
                eligible = command_index >= start
                current_index = current_index[eligible]
                target_index = target_index[eligible]
                command_index = command_index[eligible]
                current = frames[current_index, feedback_col]
                target = frames[target_index, feedback_col]
                command = frames[command_index, command_col]
                if splits[run] != "train":
                    continue
                difference = command - current
                delta = target - current
                records.append({
                    "sequence": sequence_id,
                    "run": run,
                    "family": str(families[run]),
                    "condition": int(sequence_conditions[sequence_id]),
                    "current": current,
                    "target": target,
                    "command": command,
                    "numerator": float(np.mean(difference * delta)),
                    "denominator": float(np.mean(difference ** 2)),
                })

            nested: dict[str, dict[int, dict[int, list[dict[str, Any]]]]] = {}
            for record in records:
                nested.setdefault(record["family"], {}).setdefault(
                    record["run"], {}).setdefault(record["condition"], []).append(record)

            def macro_sufficient(field: str) -> float:
                family_value = 0.0
                for family, runs in nested.items():
                    run_values = []
                    for conditions in runs.values():
                        condition_values = [
                            float(np.mean([row[field] for row in sequences]))
                            for sequences in conditions.values()]
                        run_values.append(float(np.mean(condition_values)))
                    family_value += family_weights[family] * float(np.mean(run_values))
                return family_value

            numerator = macro_sufficient("numerator")
            denominator = macro_sufficient("denominator")
            alpha = numerator / denominator if denominator > 0.0 else 0.0
            alpha = float(np.clip(alpha, 0.0, 1.0))
            sequence_squared_errors = []
            for record in records:
                prediction = record["current"] + alpha * (
                    record["command"] - record["current"])
                sequence_squared_errors.append({
                    **record,
                    "squared_error": float(np.mean(
                        (prediction - record["target"]) ** 2)),
                })
            error_nested: dict[str, dict[int, dict[int, list[float]]]] = {}
            for record in sequence_squared_errors:
                error_nested.setdefault(record["family"], {}).setdefault(
                    record["run"], {}).setdefault(record["condition"], []).append(
                        record["squared_error"])
            train_loss = 0.0
            for family, runs in error_nested.items():
                run_losses = []
                for conditions in runs.values():
                    run_losses.append(float(np.mean([
                        np.mean(sequence_errors)
                        for sequence_errors in conditions.values()])))
                train_loss += family_weights[family] * float(np.mean(run_losses))
            candidates.append((train_loss, delay, alpha))
        _, delay, alpha = min(candidates, key=lambda row: (row[0], row[1]))
        channel = ActuatorChannel(delay_steps=delay, alpha=alpha)
        selected[label] = channel
        per_split: dict[str, Any] = {}
        for split in ("train", "validation"):
            current_values: list[np.ndarray] = []
            target_values: list[np.ndarray] = []
            command_values: list[np.ndarray] = []
            split_runs: list[np.ndarray] = []
            for (start_raw, end_raw), run_raw in zip(bounds, sequence_runs):
                start, end, run = int(start_raw), int(end_raw), int(run_raw)
                if splits[run] != split or end - start <= delay + 1:
                    continue
                current_index = np.arange(start, end - 1)
                command_index = current_index - delay
                eligible = command_index >= start
                current_index = current_index[eligible]
                command_index = command_index[eligible]
                current_values.append(frames[current_index, feedback_col])
                target_values.append(frames[current_index + 1, feedback_col])
                command_values.append(frames[command_index, command_col])
                split_runs.append(np.full(len(current_index), run, dtype=np.int32))
            if not current_values:
                per_split[split] = {
                    "transitions": 0,
                    "mae": None,
                    "rmse": None,
                    "absolute_error_p95": None,
                    "run_metrics": {},
                }
                continue
            current = np.concatenate(current_values)
            target = np.concatenate(target_values)
            command = np.concatenate(command_values)
            transition_run = np.concatenate(split_runs)
            prediction = current + alpha * (command - current)
            errors = prediction - target
            by_run = {}
            for run in sorted(set(transition_run.tolist())):
                mask = transition_run == run
                local = errors[mask]
                by_run[str(run_ids[run])] = {
                    "transitions": int(mask.sum()),
                    "mae": float(np.mean(np.abs(local))),
                    "rmse": float(np.sqrt(np.mean(local ** 2))),
                    "absolute_error_p95": float(np.quantile(np.abs(local), 0.95)),
                }
            per_split[split] = {
                "transitions": int(len(errors)),
                "mae": float(np.mean(np.abs(errors))),
                "rmse": float(np.sqrt(np.mean(errors ** 2))),
                "absolute_error_p95": float(np.quantile(np.abs(errors), 0.95)),
                "run_metrics": by_run,
            }
        all_fits[label] = {
            "selected_delay_steps": delay,
            "selected_delay_seconds": delay * DT_S,
            "alpha": alpha,
            "selection_rule": "minimum hierarchical training one-step squared error (family/run/condition/sequence)",
            "family_probabilities": family_weights,
            "validation_used_for_selection": False,
            "metrics": per_split,
        }
    return ActuatorFit(selected["steering"], selected["throttle"], all_fits)


def integrate_pose(torch, physical_states, initial_pose,
                   initial_state=None):
    """Integrate COM motion into rear-axle world pose, outside plant inputs."""
    if physical_states.ndim != 3:
        raise ValueError("predicted physical states must have shape (B,T,7)")
    previous = initial_state if initial_state is not None else physical_states[:, 0]
    pose = initial_pose
    trajectory = []
    for index in range(physical_states.shape[1]):
        following = physical_states[:, index]
        u_mid = 0.5 * (previous[:, 0] + following[:, 0])
        v_com_mid = 0.5 * (previous[:, 1] + following[:, 1])
        r_mid = 0.5 * (previous[:, 2] + following[:, 2])
        v_rear_mid = v_com_mid - REAR_AXLE_TO_COM_M * r_mid
        yaw_mid = pose[:, 2] + 0.5 * r_mid * DT_S
        dx = (u_mid * torch.cos(yaw_mid) - v_rear_mid * torch.sin(yaw_mid)) * DT_S
        dy = (u_mid * torch.sin(yaw_mid) + v_rear_mid * torch.cos(yaw_mid)) * DT_S
        pose = torch.stack((pose[:, 0] + dx, pose[:, 1] + dy,
                            pose[:, 2] + r_mid * DT_S), dim=-1)
        trajectory.append(pose)
        previous = following
    if not trajectory:
        raise ValueError("pose integration requires at least one state")
    return torch.stack(trajectory, dim=1)


def effective_model_type(torch, nn, history_mean: np.ndarray,
                         history_scale: np.ndarray,
                         state_mean: np.ndarray, state_scale: np.ndarray,
                         command_mean: np.ndarray, command_scale: np.ndarray,
                         acceleration_bounds: np.ndarray,
                         actuator_fit: ActuatorFit,
                         encoder: str = "gru", latent_size: int = LATENT_SIZE,
                         expert_count: int = 1,
                         include_roll_state: bool = False,
                         couple_roll_acceleration: bool = False,
                         roll_residual_mode: bool = False,
                         roll_oscillator_coefficients: np.ndarray | None = None,
                         include_raw_encoder_history: bool = False,
                         include_wheel_innovation_history: bool = False,
                         wheel_dynamics_mode: str = "surface_acceleration"):
    """Construct the deterministic GRU- or TCN-encoder EDSSM transition."""
    if encoder not in ("gru", "tcn"):
        raise ValueError("EDSSM history encoder must be 'gru' or 'tcn'")
    if wheel_dynamics_mode not in WHEEL_DYNAMICS_MODES:
        raise ValueError("unknown rear-wheel dynamics mode")
    if latent_size not in (16, 32, 64):
        raise ValueError("EDSSM latent size must be 16, 32, or 64")
    if expert_count not in (1, 2, 3):
        raise ValueError("EDSSM requires one, two, or three transition experts")
    if couple_roll_acceleration and not include_roll_state:
        raise ValueError("roll/acceleration coupling requires predicted roll state")
    if roll_residual_mode and not include_roll_state:
        raise ValueError("roll residual mode requires predicted roll state")
    if roll_oscillator_coefficients is not None:
        coefficients = np.asarray(roll_oscillator_coefficients, dtype=np.float32)
        if (not include_roll_state or coefficients.shape != (3,)
                or not np.isfinite(coefficients).all()
                or coefficients[1] >= 0.0 or coefficients[2] >= 0.0):
            raise ValueError("invalid fitted roll oscillator coefficients")
    else:
        coefficients = None
    state_names = (PHYSICAL_STATE_NAMES + ROLL_STATE_NAMES
                   if include_roll_state else PHYSICAL_STATE_NAMES)
    history_names = (PHYSICAL_STATE_NAMES + COMMAND_NAMES
                     if roll_residual_mode else state_names + COMMAND_NAMES)
    if include_raw_encoder_history and include_wheel_innovation_history:
        raise ValueError("select only one extra encoder-history representation")
    if include_raw_encoder_history:
        history_names += RAW_ENCODER_HISTORY_NAMES
    if include_wheel_innovation_history:
        history_names += WHEEL_INNOVATION_HISTORY_NAMES
    arrays = {
        "history_mean": (history_mean, (len(history_names),)),
        "history_scale": (history_scale, (len(history_names),)),
        "state_mean": (state_mean, (len(state_names),)),
        "state_scale": (state_scale, (len(state_names),)),
        "command_mean": (command_mean, (len(COMMAND_NAMES),)),
        "command_scale": (command_scale, (len(COMMAND_NAMES),)),
        "acceleration_bounds": (acceleration_bounds, (len(ACCELERATION_NAMES),)),
    }
    validated = {}
    for name, (raw, shape) in arrays.items():
        value = np.asarray(raw, dtype=np.float32)
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"invalid EDSSM {name}")
        if name.endswith("scale") or name == "acceleration_bounds":
            if np.any(value <= 0.0):
                raise ValueError(f"EDSSM {name} must be positive")
        validated[name] = value

    class EffectiveRaceTeacher(nn.Module):
        def __init__(self):
            super().__init__()
            for name, value in validated.items():
                self.register_buffer(name, torch.as_tensor(value.copy()))
            self.encoder_kind = encoder
            self.latent_size = latent_size
            self.expert_count = expert_count
            self.include_roll_state = include_roll_state
            self.include_raw_encoder_history = include_raw_encoder_history
            self.include_wheel_innovation_history = include_wheel_innovation_history
            self.couple_roll_acceleration = couple_roll_acceleration
            self.roll_residual_mode = roll_residual_mode
            self.use_roll_oscillator = coefficients is not None
            self.state_size = len(state_names)
            self.history_state_size = (
                len(PHYSICAL_STATE_NAMES) if roll_residual_mode
                else self.state_size)
            dynamics_state_size = self.history_state_size
            self.dynamics_state_size = dynamics_state_size
            self.wheel_dynamics_mode = wheel_dynamics_mode
            self.steering_delay_steps = actuator_fit.steering.delay_steps
            self.throttle_delay_steps = actuator_fit.throttle.delay_steps
            self.steering_alpha = float(actuator_fit.steering.alpha)
            self.throttle_alpha = float(actuator_fit.throttle.alpha)

            if encoder == "gru":
                self.history_encoder = nn.GRU(
                    len(history_names), 128, batch_first=True)
                self.history_projection = nn.Sequential(
                    nn.Linear(128, 128), nn.SiLU(),
                    nn.Linear(128, latent_size), nn.Tanh())
            else:
                channels = 64
                self.history_input = nn.Conv1d(
                    len(history_names), channels, kernel_size=1)
                blocks = []
                for dilation in (1, 2, 4, 8, 16, 32):
                    blocks.append(_CausalResidualBlock(
                        nn, channels, dilation))
                self.history_tcn = nn.Sequential(*blocks)
                self.history_projection = nn.Sequential(
                    nn.Linear(channels, 128), nn.SiLU(),
                    nn.Linear(128, latent_size), nn.Tanh())

            slip_feature_size = 2 if wheel_dynamics_mode == "contact_slip" else 0
            input_size = (dynamics_state_size + len(COMMAND_NAMES)
                          + latent_size + slip_feature_size)
            def make_expert():
                trunk = nn.Sequential(
                    nn.Linear(input_size, 256), nn.SiLU(),
                    nn.Linear(256, 256), nn.SiLU(),
                    nn.Linear(256, 256), nn.SiLU())
                heads = nn.ModuleList(
                    nn.Linear(256, 1) for _ in ACCELERATION_NAMES)
                for head in heads:
                    nn.init.zeros_(head.weight)
                    nn.init.zeros_(head.bias)
                return trunk, heads

            if expert_count == 1:
                # Preserve the pre-WP13 z16/z32/z64 checkpoint parameter keys.
                self.transition_trunk, self.acceleration_heads = make_expert()
                self.expert_transition_trunks = None
                self.expert_acceleration_heads = None
                self.expert_router = None
            else:
                experts = [make_expert() for _ in range(expert_count)]
                self.expert_transition_trunks = nn.ModuleList(
                    pair[0] for pair in experts)
                self.expert_acceleration_heads = nn.ModuleList(
                    pair[1] for pair in experts)
                self.expert_router = nn.Sequential(
                    nn.Linear(input_size, 128), nn.SiLU(),
                    nn.Linear(128, expert_count))
                self.transition_trunk = None
                self.acceleration_heads = None
            latent_input_size = (self.state_size
                                 + len(COMMAND_NAMES)
                                 + len(ACCELERATION_NAMES))
            if roll_residual_mode:
                latent_input_size = (dynamics_state_size
                                     + len(COMMAND_NAMES)
                                     + len(ACCELERATION_NAMES))
            self.latent_transition = nn.GRUCell(latent_input_size, latent_size)
            if include_roll_state and not self.use_roll_oscillator:
                roll_input_size = input_size + int(couple_roll_acceleration)
                self.roll_rate_transition = nn.Sequential(
                    nn.Linear(roll_input_size, 128), nn.SiLU(),
                    nn.Linear(128, 1))
                nn.init.zeros_(self.roll_rate_transition[-1].weight)
                nn.init.zeros_(self.roll_rate_transition[-1].bias)
            if self.use_roll_oscillator:
                self.register_buffer(
                    "roll_oscillator_coefficients",
                    torch.as_tensor(coefficients.copy()))
            if roll_residual_mode:
                self.roll_force_residual = nn.Sequential(
                    nn.Linear(input_size + 3, 128), nn.SiLU(),
                    nn.Linear(128, 2))
                nn.init.zeros_(self.roll_force_residual[-1].weight)
                nn.init.zeros_(self.roll_force_residual[-1].bias)

        def encode_history(self, history):
            if (history.ndim != 3 or history.shape[1] != HISTORY_STEPS
                    or history.shape[2] != len(history_names)):
                raise ValueError(
                    f"EDSSM requires an 80x{len(history_names)} history")
            normalized = (history - self.history_mean) / self.history_scale
            if self.encoder_kind == "gru":
                _, hidden = self.history_encoder(normalized)
                encoded = hidden[-1]
            else:
                encoded = self.history_input(normalized.transpose(1, 2))
                encoded = self.history_tcn(encoded)[..., -1]
            return self.history_projection(encoded)

        def transition(self, physical_state, delayed_command, latent, command):
            if (physical_state.ndim != 2
                    or physical_state.shape[1] != self.state_size):
                raise ValueError(
                    f"EDSSM state must have shape (B,{self.state_size})")
            if command.shape != (len(physical_state), 2):
                raise ValueError("EDSSM command must have shape (B,2)")
            if delayed_command.shape != command.shape:
                raise ValueError("EDSSM delayed command must align with command")
            normalized_state_full = ((physical_state - self.state_mean)
                                     / self.state_scale)
            normalized_state = normalized_state_full[:, :self.dynamics_state_size]
            normalized_command = ((command - self.command_mean)
                                  / self.command_scale)
            feature_values = [normalized_state, normalized_command, latent]
            if self.wheel_dynamics_mode == "contact_slip":
                half_track = REAR_TRACK_WIDTH_M / 2.0
                contact_speed = torch.stack((
                    physical_state[:, 0] - half_track * physical_state[:, 2],
                    physical_state[:, 0] + half_track * physical_state[:, 2]),
                    dim=-1)
                slip_speed = physical_state[:, 5:7] - contact_speed
                # Slip is a first-class physical feature/latent driver, with
                # an explicit SI-unit scale instead of an extra learned gain.
                feature_values.append(slip_speed)
            features = torch.cat(feature_values, dim=-1)
            if self.expert_count == 1:
                hidden = self.transition_trunk(features)
                raw = torch.cat(
                    [head(hidden) for head in self.acceleration_heads], dim=-1)
                acceleration = self.acceleration_bounds * torch.tanh(raw)
                gate_weights = torch.ones(
                    (len(physical_state), 1), dtype=physical_state.dtype,
                    device=physical_state.device)
            else:
                gate_weights = torch.softmax(self.expert_router(features), dim=-1)
                expert_accelerations = []
                for trunk, heads in zip(self.expert_transition_trunks,
                                        self.expert_acceleration_heads):
                    hidden = trunk(features)
                    raw = torch.cat([head(hidden) for head in heads], dim=-1)
                    expert_accelerations.append(
                        self.acceleration_bounds * torch.tanh(raw))
                expert_accelerations = torch.stack(expert_accelerations, dim=1)
                acceleration = torch.sum(
                    gate_weights[:, :, None] * expert_accelerations, dim=1)

            if self.roll_residual_mode:
                roll_state = normalized_state_full[:, 7:9]
                predicted_ay = (acceleration[:, 1:2]
                                / self.acceleration_bounds[1])
                roll_features = torch.cat(
                    (features, roll_state, predicted_ay), dim=-1)
                roll_residual = (self.acceleration_bounds[1:3]
                                 * torch.tanh(
                                     self.roll_force_residual(
                                         roll_features)))
                acceleration = torch.cat((
                    acceleration[:, :1],
                    acceleration[:, 1:3] + roll_residual,
                    acceleration[:, 3:]), dim=-1)

            u, v, yaw_rate = physical_state[:, 0], physical_state[:, 1], physical_state[:, 2]
            ax, ay, yaw_accel = acceleration[:, 0], acceleration[:, 1], acceleration[:, 2]
            dt = DT_S
            u_next = u + dt * (ax + yaw_rate * v)
            v_next = v + dt * (ay - yaw_rate * u)
            r_next = yaw_rate + dt * yaw_accel

            steering_target = (command[:, 0] if self.steering_delay_steps == 0
                               else delayed_command[:, 0])
            throttle_target = (command[:, 1] if self.throttle_delay_steps == 0
                               else delayed_command[:, 1])
            steering_next = physical_state[:, 3] + self.steering_alpha * (
                steering_target - physical_state[:, 3])
            throttle_next = physical_state[:, 4] + self.throttle_alpha * (
                throttle_target - physical_state[:, 4])
            if self.wheel_dynamics_mode == "surface_acceleration":
                wheel_next = physical_state[:, 5:7] + dt * acceleration[:, 3:5]
            else:
                half_track = REAR_TRACK_WIDTH_M / 2.0
                current_contact = torch.stack((
                    u - half_track * yaw_rate,
                    u + half_track * yaw_rate), dim=-1)
                next_contact = torch.stack((
                    u_next - half_track * r_next,
                    u_next + half_track * r_next), dim=-1)
                slip_next = (physical_state[:, 5:7] - current_contact
                             + dt * acceleration[:, 3:5])
                wheel_next = next_contact + slip_next
            next_state = torch.cat((
                torch.stack((u_next, v_next, r_next, steering_next,
                             throttle_next), dim=-1), wheel_next), dim=-1)
            if self.include_roll_state:
                roll, roll_rate = physical_state[:, 7], physical_state[:, 8]
                if self.use_roll_oscillator:
                    c_ay, c_roll, c_rate = self.roll_oscillator_coefficients
                    roll_rate_dot = (c_ay * ay + c_roll * roll
                                     + c_rate * roll_rate)
                    roll_rate_next = roll_rate + dt * roll_rate_dot
                else:
                    rate_scale = self.state_scale[8]
                    roll_features = features
                    if self.couple_roll_acceleration:
                        predicted_ay = (acceleration[:, 1:2]
                                        / self.acceleration_bounds[1])
                        roll_features = torch.cat((features, predicted_ay), dim=-1)
                    roll_rate_next = roll_rate + rate_scale * (
                        self.roll_rate_transition(roll_features).squeeze(-1))
                roll_next = roll + 0.5 * dt * (roll_rate + roll_rate_next)
                next_state = torch.cat((
                    next_state,
                    torch.stack((roll_next, roll_rate_next), dim=-1)), dim=-1)

            latent_input = torch.cat((normalized_state, normalized_command,
                                      acceleration / self.acceleration_bounds),
                                     dim=-1)
            next_latent = self.latent_transition(latent_input, latent)
            return next_state, command, next_latent, acceleration, gate_weights

        def rollout(self, initial_state, delayed_command, history, commands):
            if commands.ndim != 3 or commands.shape[2] != 2:
                raise ValueError("future commands must have shape (B,T,2)")
            state = initial_state
            command_delay = delayed_command
            latent = self.encode_history(history)
            states, accelerations, latent_states, gate_history = [], [], [], []
            for index in range(commands.shape[1]):
                state, command_delay, latent, acceleration, gates = self.transition(
                    state, command_delay, latent, commands[:, index])
                states.append(state)
                accelerations.append(acceleration)
                latent_states.append(latent)
                gate_history.append(gates)
            if not states:
                raise ValueError("EDSSM rollout requires at least one command")
            return (torch.stack(states, dim=1),
                    torch.stack(accelerations, dim=1),
                    torch.stack(latent_states, dim=1),
                    torch.stack(gate_history, dim=1))

    return EffectiveRaceTeacher


def _CausalResidualBlock(nn, channels: int, dilation: int):
    """Single-layer causal dilated residual block used by the TCN branch."""
    class CausalBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.dilation = dilation
            self.conv = nn.Conv1d(channels, channels, kernel_size=3,
                                  dilation=dilation)
            self.project = nn.Conv1d(channels, channels, kernel_size=1)

        def forward(self, value):
            padded = nn.functional.pad(value, (2 * self.dilation, 0))
            update = self.project(nn.functional.silu(self.conv(padded)))
            return nn.functional.silu(value + update)

    return CausalBlock()
