#!/usr/bin/env python3
"""Compare packet-held and exact Unity FixedUpdate inputs over packet intervals.

Development-only PR-02 diagnostic. It matches a ROS bag's bridge request
sequence to the Unity telemetry trace, initializes the in-process source plant
from the logged production plant state, and runs two 25-step forecasts:

* the production packet-held slot-2 steering/throttle for 25 fixed steps;
* that same input held for the observed Unity fixed-step count; and
* the per-FixedUpdate commands visible in the Unity trace.

Both forecasts use the same initial hidden state and endpoint epoch. Exact
trace-input rollouts also emit a row for each 1 ms Unity FixedUpdate so the
first state divergence can be located. Reported transition errors subtract the
motion error already present at the start of each step, so they do not confuse
initial-state error with new rollout error. Unity truth and the exact applied-
command trace are offline scoring/oracle inputs only.

Example::

  python3 tools/analyze_unity_fixedstep_rollout.py \
    live_runs/<run>/run \
    --trace live_runs/<run>/unity_trace/unity_fixedstep.csv \
    --telemetry live_runs/<run>/unity_trace/unity_fixedstep.csv.telemetry.csv \
    --library build/f1tenth_mpc/libunity_source_plant_eval.so \
    --output-prefix live_runs/<run>/fixedstep_rollout
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable

try:
    from tools import evaluate_mpc_bag as evaluator
except ModuleNotFoundError:
    import evaluate_mpc_bag as evaluator


SCHEMA = "sdu_apex_unity_fixedstep_rollout_ablation/v2"
FIXED_STEPS_PER_PACKET = 25
WHEEL_RPM_FIELDS = ("fl_rpm", "fr_rpm", "rl_rpm", "rr_rpm")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(line for line in stream if not line.startswith("#")))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_bag(path: Path) -> str:
    files = sorted(list(path.rglob("*.db3")) + list(path.rglob("*.mcap")))
    if not files:
        raise FileNotFoundError(f"no rosbag database found under {path}")
    digest = hashlib.sha256()
    for file_path in files:
        digest.update(file_path.name.encode("utf-8"))
        digest.update(bytes.fromhex(_sha256(file_path)))
    return digest.hexdigest()


def _metric(values: Iterable[float]) -> dict[str, float | int | None]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"n": 0, "bias": None, "rmse": None, "p95_abs": None,
                "max_abs": None}
    ordered_abs = sorted(abs(value) for value in finite)
    return {
        "n": len(finite),
        "bias": statistics.fmean(finite),
        "rmse": math.sqrt(statistics.fmean(value * value for value in finite)),
        "p95_abs": ordered_abs[math.ceil(0.95 * len(ordered_abs)) - 1],
        "max_abs": ordered_abs[-1],
    }


def _distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"n": 0, "mean": None, "min": None, "max": None,
                "stddev": None}
    mean = statistics.fmean(finite)
    return {
        "n": len(finite),
        "mean": mean,
        "min": min(finite),
        "max": max(finite),
        "stddev": math.sqrt(statistics.fmean(
            (value - mean) ** 2 for value in finite)),
    }


def unity_world_to_body(
    quaternion_xyzw: tuple[float, float, float, float],
    velocity_xyz: tuple[float, float, float],
    angular_velocity_xyz: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Return Unity-car (u, v_COM, yaw rate) using the verified axis mapping.

    Unity's raw rigidbody velocity is world-frame. Inverse-rotate it by the
    raw Unity transform quaternion. The active F1TENTH body uses local +Z for
    forward; ROS lateral is the negative local-X component. Unity yaw rate is
    the negative world-Y angular-velocity component in the ROS convention.
    """
    qx, qy, qz, qw = quaternion_xyzw
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("Unity trace quaternion must be finite and nonzero")
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    vx, vy, vz = velocity_xyz
    local_x = ((1.0 - 2.0 * (qy * qy + qz * qz)) * vx +
               2.0 * (qx * qy + qz * qw) * vy +
               2.0 * (qx * qz - qy * qw) * vz)
    local_z = (2.0 * (qx * qz + qy * qw) * vx +
               2.0 * (qy * qz - qx * qw) * vy +
               (1.0 - 2.0 * (qx * qx + qy * qy)) * vz)
    return local_z, -local_x, -angular_velocity_xyz[1]


def trace_body_motion(row: dict[str, str]) -> tuple[float, float, float]:
    quaternion = tuple(float(row[f"rotation_{axis}"]) for axis in "xyzw")
    velocity = tuple(float(row[f"velocity_{axis}"]) for axis in "xyz")
    angular = tuple(float(row[f"angular_velocity_{axis}"]) for axis in "xyz")
    return unity_world_to_body(quaternion, velocity, angular)


def transition_residual(
    predicted_end: float, modeled_start: float,
    truth_end: float, truth_start: float,
) -> float:
    """Remove the initial state mismatch from a forecast endpoint residual."""
    return (predicted_end - modeled_start) - (truth_end - truth_start)


def _source_state(row: dict[str, Any]) -> list[float]:
    state = [0.0] * evaluator.UNITY_SOURCE_PLANT_STATE_SIZE
    state[0] = float(row["u"])
    state[1] = float(row["v_com"])
    state[2] = float(row["r"])
    state[3:7] = [float(value) for value in row["wheel_omega"]]
    state[7:13] = [float(row[name]) for name in (
        "body_height", "body_height_rate", "roll", "roll_rate", "pitch",
        "pitch_rate")]
    state[13] = float(row["physical_steering"])
    state[14:18] = [float(value) for value in row["suspension_jounce"]]
    state[18:22] = [float(value) for value in row["low_forward_speed_time_s"]]
    state[22:26] = [float(value) for value in row["low_side_speed_time_s"]]
    return state


def _plant_stepper(library_path: Path):
    evaluator.validate_source_plant_abi()
    library = ctypes.CDLL(str(library_path.resolve()))
    step = library.unity_source_plant_step_c
    step.argtypes = [ctypes.POINTER(evaluator.UnitySourcePlantInputC),
                     ctypes.c_double,
                     ctypes.POINTER(evaluator.UnitySourcePlantOutputC)]
    step.restype = ctypes.c_int

    def rollout_path(
        state: list[float], wheel_angle: list[float], history_initialized: bool,
        controls: list[tuple[float, float, float]],
    ) -> tuple[list[float], list[float], list[list[float]]]:
        current = state[:]
        angles = wheel_angle[:]
        history = int(history_initialized)
        states_after_step: list[list[float]] = []
        for steering, throttle, dt_s in controls:
            plant_input = evaluator.UnitySourcePlantInputC()
            plant_input.state[:] = current
            plant_input.wheel_rotation_angle_rad[:] = angles
            plant_input.steering_target_rad = (
                max(-1.0, min(1.0, steering)) * evaluator.MAX_STEERING_RAD)
            plant_input.throttle_normalized = max(-1.0, min(1.0, throttle))
            plant_input.suspension_history_initialized = history
            output = evaluator.UnitySourcePlantOutputC()
            if not step(ctypes.byref(plant_input), dt_s, ctypes.byref(output)):
                raise RuntimeError("source plant rejected fixed-step trace input")
            current = list(output.state)
            angles = list(output.wheel_rotation_angle_rad)
            history = int(output.suspension_history_initialized)
            states_after_step.append(current[:])
        return current, angles, states_after_step

    def rollout(
        state: list[float], wheel_angle: list[float], history_initialized: bool,
        controls: list[tuple[float, float, float]],
    ) -> tuple[list[float], list[float]]:
        end_state, end_angles, _ = rollout_path(
            state, wheel_angle, history_initialized, controls)
        return end_state, end_angles

    return rollout, rollout_path


def _group_metrics(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    result = {}
    groups = {
        "all": lambda row: True,
        "throttle_unchanged_over_25ms": lambda row: row["throttle_range"] <= 1.0e-6,
        "throttle_changed_over_25ms": lambda row: row["throttle_range"] > 1.0e-6,
        "throttle_net_rising": lambda row: row["throttle_net_delta"] > 1.0e-6,
        "throttle_net_falling": lambda row: row["throttle_net_delta"] < -1.0e-6,
        "steering_changed_over_25ms": lambda row: row["steering_range"] > 1.0e-6,
        "near_straight_start": lambda row: abs(row["start_physical_steering_rad"]) < 0.10,
        "turning_start": lambda row: abs(row["start_physical_steering_rad"]) >= 0.10,
    }
    for name, predicate in groups.items():
        selected = [row for row in rows if predicate(row)]
        result[name] = {
            "count": len(selected),
            "packet_held_absolute_error": _metric(
                row[f"held_{key}_absolute_error"] for row in selected),
            "trace_input_absolute_error": _metric(
                row[f"trace_{key}_absolute_error"] for row in selected),
            "packet_held_transition_error": _metric(
                row[f"held_{key}_transition_error"] for row in selected),
            "trace_input_transition_error": _metric(
                row[f"trace_{key}_transition_error"] for row in selected),
        }
    return result


def _packet_interval_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    arms = ("held_25", "held_actual_steps", "trace_actual_steps")
    result: dict[str, Any] = {"count": len(rows)}
    for state in ("u", "v_com", "yaw_rate"):
        result[state] = {}
        for arm in arms:
            result[state][arm] = {
                "absolute_error": _metric(
                    row[f"{arm}_{state}_absolute_error"] for row in rows),
                "transition_error": _metric(
                    row[f"{arm}_{state}_transition_error"] for row in rows),
            }
    step_counts: dict[str, Any] = {}
    for count in sorted({int(row["actual_fixed_steps"]) for row in rows}):
        selected = [row for row in rows if int(row["actual_fixed_steps"]) == count]
        step_counts[str(count)] = {
            "count": len(selected),
            "u_transition_rmse_mps": {
                arm: _metric(row[f"{arm}_u_transition_error"] for row in selected)["rmse"]
                for arm in arms
            },
            "v_com_transition_rmse_mps": {
                arm: _metric(row[f"{arm}_v_com_transition_error"] for row in selected)["rmse"]
                for arm in arms
            },
        }
    result["by_actual_fixed_step_count"] = step_counts
    return result


def _command_epoch_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(rows),
        "trace_command_sequence_at_snapshot_minus_event_sequence": {
            str(offset): sum(
                int(row["start_trace_command_sequence"]) -
                int(row["event_command_sequence"]) == offset for row in rows)
            for offset in sorted({
                int(row["start_trace_command_sequence"]) -
                int(row["event_command_sequence"]) for row in rows})
        },
        "first_fixedupdate_offset_for_event_command": {
            str(offset): sum(
                int(row["event_command_apply_offset_steps"]) == offset
                for row in rows)
            for offset in sorted({
                int(row["event_command_apply_offset_steps"]) for row in rows})
        },
        "trace_command_sequence_at_next_snapshot_minus_event_sequence": {
            str(offset): sum(
                int(row["end_trace_command_sequence"]) -
                int(row["event_command_sequence"]) == offset for row in rows)
            for offset in sorted({
                int(row["end_trace_command_sequence"]) -
                int(row["event_command_sequence"]) for row in rows})
        },
        "requested_vs_received_command_abs_error": {
            "throttle": _metric(
                row["requested_received_throttle_error"] for row in rows),
            "steering": _metric(
                row["requested_received_steering_error"] for row in rows),
        },
        "positive_drive_torque_per_normalized_throttle_nm": _distribution(
            row["drive_torque_per_normalized_throttle_nm"] for row in rows
            if row["drive_torque_per_normalized_throttle_nm"] is not None),
        "source_plant_configured_motor_torque_nm": 428.0,
        "semantics": (
            "event_command_sequence is the telemetry-associated ROS request. "
            "The trace command_sequence and received/requested values show "
            "when that command became visible to VehicleController.FixedUpdate; "
            "drive_torque is checked against normalized throttle. This is a "
            "development trace association, not a competition runtime signal."),
    }


def _fixedstep_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize per-FixedUpdate residuals without pooling wheel truth claims."""
    groups = {
        "all": lambda row: True,
        "throttle_rising": lambda row: row["packet_throttle_net_delta"] > 1.0e-6,
        "throttle_falling": lambda row: row["packet_throttle_net_delta"] < -1.0e-6,
        "throttle_stable": lambda row: abs(row["packet_throttle_net_delta"]) <= 1.0e-6,
        "near_straight": lambda row: abs(row["packet_start_steering_rad"]) < 0.10,
        "turning": lambda row: abs(row["packet_start_steering_rad"]) >= 0.10,
    }
    result: dict[str, Any] = {"count": len(rows), "groups": {}}
    for group_name, predicate in groups.items():
        selected = [row for row in rows if predicate(row)]
        group: dict[str, Any] = {"count": len(selected)}
        for state in ("u", "v_com", "yaw_rate"):
            group[state] = {
                "absolute_error": _metric(
                    row[f"model_{state}_absolute_error"] for row in selected),
                "transition_error": _metric(
                    row[f"model_{state}_transition_error"] for row in selected),
            }
        group["wheel_omega_phase_comparison_radps"] = {
            wheel: _metric(row[f"model_{wheel}_omega_phase_error_radps"]
                           for row in selected)
            for wheel in WHEEL_RPM_FIELDS
        }
        result["groups"][group_name] = group
    result["wheel_semantics"] = (
        "Unity RPM is sampled before the next WheelCollider solve; these are "
        "phase-sensitive trace comparisons, not independent wheel truth labels.")
    return result


def analyze(
    bag_path: Path, trace_path: Path, telemetry_path: Path,
    library_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]],
           list[dict[str, Any]]]:
    if not library_path.is_file():
        raise FileNotFoundError(f"source-plant evaluator library not found: {library_path}")
    trace_rows = _read_csv(trace_path)
    telemetry_rows = _read_csv(telemetry_path)
    trace = {int(row["capture_row"]): row for row in trace_rows}
    if len(trace) != len(trace_rows):
        raise ValueError("Unity fixed-step trace has duplicate capture_row values")

    data = evaluator.read_bag(bag_path)
    truth_by_stamp = {int(row["stamp_ns"]): row for row in data.get(evaluator.GT, [])}
    plant_by_stamp = {int(row["stamp_ns"]): row for row in data.get(evaluator.PLANT, [])}
    bridge_by_sequence: dict[int, dict[str, Any]] = {}
    for row in data.get(evaluator.BRIDGE_TIMING, []):
        payload = row.get("json", {})
        sequence = payload.get("request_sequence")
        if sequence is not None:
            bridge_by_sequence[int(sequence)] = payload

    telemetry_by_sequence: dict[int, dict[str, str]] = {}
    for row in telemetry_rows:
        sequence = row.get("event_command_sequence")
        if sequence not in (None, "", "None"):
            telemetry_by_sequence[int(sequence)] = row

    rollout = _plant_stepper(library_path)
    rollout_with_path = rollout[1]
    rollout = rollout[0]
    result_rows: list[dict[str, Any]] = []
    packet_interval_rows: list[dict[str, Any]] = []
    fixedstep_state_rows: list[dict[str, Any]] = []
    command_epoch_rows: list[dict[str, Any]] = []
    alignment_rows = []
    trace_body_errors: dict[str, list[float]] = {key: [] for key in ("u", "v_com", "r")}
    fixed_step_gaps: list[int] = []

    matched_sequences = sorted(
        set(bridge_by_sequence) & set(telemetry_by_sequence))
    for sequence in matched_sequences:
        bridge = bridge_by_sequence[sequence]
        telemetry = telemetry_by_sequence[sequence]
        stamp_ns = bridge.get("bridge_request_ros_stamp_ns")
        if stamp_ns is None:
            continue
        stamp_ns = int(stamp_ns)
        gt = truth_by_stamp.get(stamp_ns)
        plant = plant_by_stamp.get(stamp_ns)
        fixed_index = int(telemetry["fixed_step_index"])
        fixed_row = trace.get(fixed_index)
        if gt is None or plant is None or fixed_row is None:
            continue
        trace_motion = trace_body_motion(fixed_row)
        for key, actual, expected in zip(
                ("u", "v_com", "r"), trace_motion,
                (float(gt["u"]), float(gt["v"]), float(gt["r"]))):
            trace_body_errors[key].append(actual - expected)
        alignment_rows.append({
            "request_sequence": sequence,
            "stamp_ns": stamp_ns,
            "fixed_step_index": fixed_index,
            "trace_gt_u_error_mps": trace_motion[0] - float(gt["u"]),
            "trace_gt_v_com_error_mps": trace_motion[1] - float(gt["v"]),
            "trace_gt_yaw_rate_error_radps": trace_motion[2] - float(gt["r"]),
            "trace_gt_position_x_error_m": (
                float(fixed_row["position_x"]) - float(bridge["simulator_position_x"])),
            "trace_gt_position_z_error_m": (
                float(fixed_row["position_z"]) - float(bridge["simulator_position_z"])),
        })

    valid_events = []
    for sequence in matched_sequences:
        bridge = bridge_by_sequence[sequence]
        telemetry = telemetry_by_sequence[sequence]
        stamp_ns = bridge.get("bridge_request_ros_stamp_ns")
        if stamp_ns is None or int(stamp_ns) not in plant_by_stamp:
            continue
        index = int(telemetry["fixed_step_index"])
        if index in trace:
            valid_events.append((sequence, index, plant_by_stamp[int(stamp_ns)]))
    valid_events.sort(key=lambda item: item[0])

    # Match each consecutive sensor/telemetry pair at its actual Unity
    # FixedUpdate endpoint. This separates (a) production's fixed 25-step
    # propagation, (b) the same queued command for the observed number of
    # steps, and (c) the actual per-step command sequence for that same span.
    for before, after in zip(valid_events, valid_events[1:]):
        sequence, start_index, start_plant = before
        next_sequence, end_index, _ = after
        if next_sequence != sequence + 1 or end_index <= start_index:
            continue
        if any(index not in trace for index in range(start_index, end_index + 1)):
            continue
        start_gt = trace_body_motion(trace[start_index])
        end_gt = trace_body_motion(trace[end_index])
        actual_controls = [
            (float(trace[index]["requested_steering"]),
             float(trace[index]["requested_throttle"]),
             float(trace[index]["fixed_delta_time"]))
            for index in range(start_index, end_index)]
        step_count = end_index - start_index
        event_command_apply_offset = next(
            (index - start_index for index in range(start_index, end_index)
             if int(trace[index]["command_sequence"]) == sequence),
            step_count)
        held_steering = float(start_plant["delayed_steering_2"])
        held_throttle = float(start_plant["delayed_throttle_2"])
        state = _source_state(start_plant)
        wheel_angles = [float(value) for value in start_plant["wheel_angle"]]
        history = bool(start_plant["suspension_history_initialized"])
        q25_controls = [(held_steering, held_throttle, 0.001)] * FIXED_STEPS_PER_PACKET
        q_actual_steps = [(held_steering, held_throttle, 0.001)] * step_count
        trace_end, _, trace_states_after_step = rollout_with_path(
            state, wheel_angles, history, actual_controls)
        predictions = {
            "held_25": rollout(state, wheel_angles, history, q25_controls)[0],
            "held_actual_steps": rollout(state, wheel_angles, history, q_actual_steps)[0],
            "trace_actual_steps": trace_end,
        }
        row: dict[str, Any] = {
            "request_sequence": sequence,
            "next_request_sequence": next_sequence,
            "start_trace_command_sequence": int(
                trace[start_index]["command_sequence"]),
            "event_command_apply_offset_steps": event_command_apply_offset,
            "end_trace_command_sequence": int(
                trace[end_index]["command_sequence"]),
            "start_fixed_step": start_index,
            "end_fixed_step": end_index,
            "actual_fixed_steps": step_count,
            "actual_fixed_time_s": sum(control[2] for control in actual_controls),
            "nominal_fixed_time_s": FIXED_STEPS_PER_PACKET * 0.001,
            "held_throttle": held_throttle,
            "held_steering": held_steering,
            "trace_throttle_start": actual_controls[0][1],
            "trace_throttle_end": actual_controls[-1][1],
            "trace_throttle_range": max(c[1] for c in actual_controls) -
                min(c[1] for c in actual_controls),
            "trace_steering_start": actual_controls[0][0],
            "trace_steering_end": actual_controls[-1][0],
            "trace_steering_range": max(c[0] for c in actual_controls) -
                min(c[0] for c in actual_controls),
            "event_command_sequence": sequence,
            "requested_received_throttle_error": max(
                abs(float(trace[index]["requested_throttle"]) -
                    float(trace[index]["received_throttle"]))
                for index in range(start_index, end_index)),
            "requested_received_steering_error": max(
                abs(float(trace[index]["requested_steering"]) -
                    float(trace[index]["received_steering"]))
                for index in range(start_index, end_index)),
            "drive_torque_per_normalized_throttle_nm": max(
                (float(trace[index]["drive_torque"]) /
                 float(trace[index]["requested_throttle"])
                 for index in range(start_index, end_index)
                 if float(trace[index]["requested_throttle"]) > 1.0e-6),
                default=None),
        }
        for name, component, gt_start, gt_end in (
                ("u", 0, start_gt[0], end_gt[0]),
                ("v_com", 1, start_gt[1], end_gt[1]),
                ("yaw_rate", 2, start_gt[2], end_gt[2])):
            for arm, prediction in predictions.items():
                row[f"{arm}_{name}_absolute_error"] = prediction[component] - gt_end
                row[f"{arm}_{name}_transition_error"] = transition_residual(
                    prediction[component], state[component], gt_end, gt_start)
        packet_interval_rows.append(row)
        command_epoch_rows.append(row)

        # Compare the predicted state after each exact FixedUpdate input with
        # the next pre-solve Unity trace sample. Keep both cumulative absolute
        # error and per-step transition error so the first divergence is
        # distinguishable from an already-wrong packet-start state.
        previous_model_state = state
        previous_trace_motion = start_gt
        packet_throttle_delta = actual_controls[-1][1] - actual_controls[0][1]
        packet_start_steering = float(start_plant["physical_steering"])
        for offset, (step_control, model_state) in enumerate(
                zip(actual_controls, trace_states_after_step)):
            trace_index = start_index + offset + 1
            trace_row = trace[trace_index]
            trace_motion = trace_body_motion(trace_row)
            step_record: dict[str, Any] = {
                "request_sequence": sequence,
                "fixed_step_index": trace_index,
                "step_offset_from_packet_start": offset + 1,
                "fixed_delta_time_s": step_control[2],
                "requested_steering_norm": step_control[0],
                "requested_throttle_norm": step_control[1],
                "command_sequence": int(trace[trace_index - 1]["command_sequence"]),
                "packet_throttle_net_delta": packet_throttle_delta,
                "packet_start_steering_rad": packet_start_steering,
            }
            for name, state_index, actual_now, actual_before in (
                    ("u", 0, trace_motion[0], previous_trace_motion[0]),
                    ("v_com", 1, trace_motion[1], previous_trace_motion[1]),
                    ("yaw_rate", 2, trace_motion[2], previous_trace_motion[2])):
                step_record[f"model_{name}_absolute_error"] = (
                    model_state[state_index] - actual_now)
                step_record[f"model_{name}_transition_error"] = transition_residual(
                    model_state[state_index], previous_model_state[state_index],
                    actual_now, actual_before)
            for wheel_index, field in enumerate(WHEEL_RPM_FIELDS):
                trace_omega = float(trace_row[field]) * (2.0 * math.pi / 60.0)
                step_record[f"model_{field}_omega_phase_error_radps"] = (
                    model_state[3 + wheel_index] - trace_omega)
            fixedstep_state_rows.append(step_record)
            previous_model_state = model_state
            previous_trace_motion = trace_motion

    for sequence, start_index, start_plant in valid_events:
        end_index = start_index + FIXED_STEPS_PER_PACKET
        if end_index not in trace:
            continue
        start_gt = trace_body_motion(trace[start_index])
        end_gt = trace_body_motion(trace[end_index])
        actual_controls: list[tuple[float, float, float]] = []
        q_values: list[float] = []
        steering_values: list[float] = []
        for index in range(start_index, end_index):
            row = trace[index]
            q = float(row["requested_throttle"])
            steering = float(row["requested_steering"])
            dt_s = float(row["fixed_delta_time"])
            actual_controls.append((steering, q, dt_s))
            q_values.append(q)
            steering_values.append(steering)

        start_state = _source_state(start_plant)
        wheel_angles = [float(value) for value in start_plant["wheel_angle"]]
        history = bool(start_plant["suspension_history_initialized"])
        held_steering = float(start_plant["delayed_steering_2"])
        held_throttle = float(start_plant["delayed_throttle_2"])
        held_controls = [(held_steering, held_throttle,
                          float(trace[start_index + step_index]["fixed_delta_time"]))
                         for step_index in range(FIXED_STEPS_PER_PACKET)]
        held_state, held_angles = rollout(
            start_state, wheel_angles, history, held_controls)
        trace_state, trace_angles = rollout(
            start_state, wheel_angles, history, actual_controls)

        row_out: dict[str, Any] = {
            "request_sequence": sequence,
            "start_fixed_step": start_index,
            "end_fixed_step": end_index,
            "fixed_steps": FIXED_STEPS_PER_PACKET,
            "fixed_time_s": sum(control[2] for control in actual_controls),
            "held_throttle": held_throttle,
            "held_steering": held_steering,
            "trace_throttle_start": q_values[0],
            "trace_throttle_end": q_values[-1],
            "throttle_net_delta": q_values[-1] - q_values[0],
            "throttle_range": max(q_values) - min(q_values),
            "steering_net_delta": steering_values[-1] - steering_values[0],
            "steering_range": max(steering_values) - min(steering_values),
            "start_physical_steering_rad": float(start_plant["physical_steering"]),
            "start_u_mps": float(start_plant["u"]),
            "start_gt_u_mps": start_gt[0],
            "start_u_error_mps": float(start_plant["u"]) - start_gt[0],
            "unity_end_u_mps": end_gt[0],
            "unity_end_v_com_mps": end_gt[1],
            "unity_end_yaw_rate_radps": end_gt[2],
            "held_end_u_mps": held_state[0],
            "held_end_v_com_mps": held_state[1],
            "held_end_yaw_rate_radps": held_state[2],
            "trace_end_u_mps": trace_state[0],
            "trace_end_v_com_mps": trace_state[1],
            "trace_end_yaw_rate_radps": trace_state[2],
        }
        for name, state_index, gt_start, gt_end in (
                ("u", 0, start_gt[0], end_gt[0]),
                ("v_com", 1, start_gt[1], end_gt[1]),
                ("yaw_rate", 2, start_gt[2], end_gt[2])):
            row_out[f"held_{name}_absolute_error"] = held_state[state_index] - gt_end
            row_out[f"trace_{name}_absolute_error"] = trace_state[state_index] - gt_end
            row_out[f"held_{name}_transition_error"] = transition_residual(
                held_state[state_index], start_state[state_index], gt_end, gt_start)
            row_out[f"trace_{name}_transition_error"] = transition_residual(
                trace_state[state_index], start_state[state_index], gt_end, gt_start)

        for wheel_index, field in enumerate(WHEEL_RPM_FIELDS):
            # RPM in this Unity trace is sampled before the next WheelCollider
            # solve; report it as a phase-sensitive wheel-state comparison.
            gt_omega = float(trace[end_index][field]) * (2.0 * math.pi / 60.0)
            row_out[f"wheel_{wheel_index}_unity_end_omega_radps"] = gt_omega
            row_out[f"wheel_{wheel_index}_held_end_error_radps"] = (
                held_state[3 + wheel_index] - gt_omega)
            row_out[f"wheel_{wheel_index}_trace_end_error_radps"] = (
                trace_state[3 + wheel_index] - gt_omega)
        row_out["held_rear_wheel_rmse_radps"] = math.sqrt(statistics.fmean(
            row_out[f"wheel_{index}_held_end_error_radps"] ** 2
            for index in (2, 3)))
        row_out["trace_rear_wheel_rmse_radps"] = math.sqrt(statistics.fmean(
            row_out[f"wheel_{index}_trace_end_error_radps"] ** 2
            for index in (2, 3)))
        result_rows.append(row_out)

    for (_, before_index, _), (_, after_index, _) in zip(valid_events, valid_events[1:]):
        if after_index > before_index:
            fixed_step_gaps.append(after_index - before_index)

    summary = {
        "schema": SCHEMA,
        "bag": str(bag_path),
        "bag_db_sha256": _hash_bag(bag_path),
        "trace": str(trace_path),
        "trace_sha256": _sha256(trace_path),
        "telemetry": str(telemetry_path),
        "telemetry_sha256": _sha256(telemetry_path),
        "source_plant_library": str(library_path),
        "source_plant_library_sha256": _sha256(library_path),
        "alignment": {
            "bridge_request_sequences": len(bridge_by_sequence),
            "unity_telemetry_sequences": len(telemetry_by_sequence),
            "matched_request_telemetry_sequences": len(matched_sequences),
            "exact_truth_plant_trace_matches": len(alignment_rows),
            "one_step_rollout_windows": len(result_rows),
            "ground_truth_alignment": (
                "exact bag request header stamp -> Bridge request_sequence -> "
                "Unity telemetry event_command_sequence/fixed_step_index"),
            "command_semantics": (
                "Unity trace requested_* is the normalized autonomous command "
                "visible to FixedUpdate; drive_torque cross-checks applied "
                "longitudinal torque. Packet-held arms use production delayed_*_2."),
            "truth_axis_mapping": (
                "inverse-rotate raw Unity Rigidbody world velocity by raw "
                "Unity transform quaternion; u=local +Z, v=-local X, "
                "yaw_rate=-Unity world angular Y"),
            "wheel_truth_semantics": (
                "WheelCollider RPM sampled at the Unity FixedUpdate trace "
                "phase; wheel comparison is phase-sensitive and development-only"),
        },
        "trace_to_bag_state_error": {
            key: _metric(values) for key, values in trace_body_errors.items()},
        "fixed_step_gap_between_telemetry_events": {
            "count": len(fixed_step_gaps),
            "counts": {str(gap): fixed_step_gaps.count(gap)
                       for gap in sorted(set(fixed_step_gaps))},
            "sum_deviation_from_25_steps": sum(
                gap - FIXED_STEPS_PER_PACKET for gap in fixed_step_gaps),
            "interpretation": (
                "Observed Unity fixed-step gap between telemetry-associated "
                "snapshots. This is not inferred from ROS header spacing and "
                "does not by itself identify why the gap differs from 25."),
        },
        "command_epoch_alignment": _command_epoch_metrics(command_epoch_rows),
        "forward_speed": _group_metrics(result_rows, "u"),
        "lateral_com_velocity": _group_metrics(result_rows, "v_com"),
        "yaw_rate": _group_metrics(result_rows, "yaw_rate"),
        "rear_wheel_speed": {
            "count": len(result_rows),
            "packet_held_endpoint_rmse_radps": _metric(
                [row["held_rear_wheel_rmse_radps"] for row in result_rows]),
            "trace_input_endpoint_rmse_radps": _metric(
                [row["trace_rear_wheel_rmse_radps"] for row in result_rows]),
        },
        "packet_interval_rollout": _packet_interval_metrics(packet_interval_rows),
        "fixedstep_state_rollout": _fixedstep_metrics(fixedstep_state_rows),
        "limitations": [
            "This capture contains many throttle/steering transitions, but it "
            "is not independent replication across runs.",
            "Exact per-step trace inputs are an offline oracle, never production inputs.",
            "The one-step rollout begins from the production-published hidden state; "
            "transition error subtracts its initial visible-motion mismatch.",
            "This diagnostic does not replay the observer's packet-boundary "
            "encoder/IMU corrections between horizons.",
        ],
    }
    return summary, result_rows, packet_interval_rows, fixedstep_state_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--telemetry", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    summary, rows, packet_rows, fixedstep_rows = analyze(
        args.bag, args.trace, args.telemetry, args.library)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = args.output_prefix.with_suffix(".json")
    csv_path = args.output_prefix.with_suffix(".csv")
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    packet_csv_path = args.output_prefix.with_name(
        args.output_prefix.name + "_packet_intervals.csv")
    if packet_rows:
        with packet_csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(packet_rows[0]))
            writer.writeheader()
            writer.writerows(packet_rows)
    fixedstep_csv_path = args.output_prefix.with_name(
        args.output_prefix.name + "_fixedsteps.csv")
    if fixedstep_rows:
        with fixedstep_csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(fixedstep_rows[0]))
            writer.writeheader()
            writer.writerows(fixedstep_rows)
    print(f"wrote {json_path}, {csv_path}, {packet_csv_path}, and "
          f"{fixedstep_csv_path}")
    print("exact 25-step windows:", summary["alignment"]["one_step_rollout_windows"])
    for name in ("forward_speed", "lateral_com_velocity", "yaw_rate"):
        group = summary[name]["all"]
        for arm in ("packet_held", "trace_input"):
            metric = group[f"{arm}_transition_error"]
            print(f"{name} {arm}: RMSE={metric['rmse']:.6f} "
                  f"p95={metric['p95_abs']:.6f} max={metric['max_abs']:.6f}")
    packet_u = summary["packet_interval_rollout"]["u"]
    print("packet interval forward-speed transition RMSE:", {
        arm: values["transition_error"]["rmse"]
        for arm, values in packet_u.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
