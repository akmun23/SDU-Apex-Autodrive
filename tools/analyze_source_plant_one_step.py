#!/usr/bin/env python3
"""Score exact-epoch one-step source-plant prediction errors by input regime.

This is an offline diagnostic. At each exact Unity epoch it initializes the
production source plant from the recorded, sensor-corrected UnityPlantState,
uses the logged newest command queue slot for one 25 ms interval, and scores
only against the next Unity ground-truth sample. Unity truth is never used as
an input. The logged queue slot is a host-side command association, not proof
of the command applied at every Unity FixedUpdate.

Example (ROS 2 bag types and the matching evaluator library are required)::

  python3 tools/analyze_source_plant_one_step.py live_runs/<run>/run \
    --library build/f1tenth_mpc/libunity_source_plant_eval.so \
    --output-prefix live_runs/<run>/source_one_step
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
from typing import Any, Callable

try:
    from tools import evaluate_mpc_bag as evaluator
    from tools import analyze_mpc_odom_drift as drift_report
except ModuleNotFoundError:
    import evaluate_mpc_bag as evaluator
    import analyze_mpc_odom_drift as drift_report


SCHEMA = "sdu_apex_source_plant_one_step_regimes/v1"
SPEED_BINS = ((0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0),
              (4.0, 5.0), (5.0, 8.0), (8.0, math.inf))
THROTTLE_BINS = ((0.0, 0.1), (0.1, 0.3), (0.3, 0.6), (0.6, 1.000001))
ELAPSED_TIME_BINS_S = (
    ("0.00_0.25s", 0.0, 0.25),
    ("0.25_0.50s", 0.25, 0.50),
    ("0.50_0.75s", 0.50, 0.75),
    ("0.75_1.00s", 0.75, 1.00),
    ("1.00_1.50s", 1.00, 1.50),
    ("1.50_2.00s", 1.50, 2.00),
    ("2.00_3.00s", 2.00, 3.00),
    ("3.00_5.00s", 3.00, 5.00),
    ("5.00s_plus", 5.00, math.inf),
)


def _metric(values: list[float]) -> dict[str, float | int | None]:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return {"n": 0, "bias": None, "rmse": None,
                "p95_abs": None, "max_abs": None}
    absolute = sorted(abs(value) for value in finite)
    return {
        "n": len(finite),
        "bias": statistics.mean(finite),
        "rmse": math.sqrt(statistics.mean(value * value for value in finite)),
        "p95_abs": absolute[min(len(absolute) - 1,
                                int(0.95 * (len(absolute) - 1)))],
        "max_abs": absolute[-1],
    }


def _source_state(row: dict[str, Any]) -> list[float]:
    state = [0.0] * evaluator.UNITY_SOURCE_PLANT_STATE_SIZE
    state[0] = row["u"]
    state[1] = row["v_com"]
    state[2] = row["r"]
    state[3:7] = row["wheel_omega"]
    state[7:13] = [row["body_height"], row["body_height_rate"],
                   row["roll"], row["roll_rate"], row["pitch"],
                   row["pitch_rate"]]
    state[13] = row["physical_steering"]
    state[14:18] = row["suspension_jounce"]
    state[18:22] = row["low_forward_speed_time_s"]
    state[22:26] = row["low_side_speed_time_s"]
    return state


def _safe_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def _groups(rows: list[dict[str, Any]],
            partitions: list[tuple[str, Callable[[dict[str, Any]], bool]]]
            ) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name, predicate in partitions:
        selected = [row for row in rows if predicate(row)]
        result[name] = {
            "count": len(selected),
            "forward_velocity_error_mps": _metric(
                [row["u_error_mps"] for row in selected]),
            "production_odom_forward_velocity_error_mps": _metric(
                [row["production_odom_u_error_mps"] for row in selected]),
            "initial_source_state_forward_velocity_error_mps": _metric(
                [row.get("initial_source_u_error_mps", math.nan)
                 for row in selected]),
            "slot1_forward_velocity_error_mps": _metric(
                [row.get("slot1_u_error_mps", math.nan)
                 for row in selected]),
            "next_feedback_forward_velocity_error_mps": _metric(
                [row.get("next_feedback_u_error_mps", math.nan)
                 for row in selected]),
            "same_epoch_GT_motion_state_oracle_error_mps": _metric(
                [row.get("oracle_motion_u_error_mps", math.nan)
                 for row in selected]),
            "IMU_trapezoid_from_source_start_error_mps": _metric(
                [row.get("imu_trapezoid_from_source_u_error_mps", math.nan)
                 for row in selected]),
            "IMU_trapezoid_from_GT_start_error_mps": _metric(
                [row.get("imu_trapezoid_from_GT_u_error_mps", math.nan)
                 for row in selected]),
            "lateral_velocity_error_mps": _metric(
                [row["v_com_error_mps"] for row in selected]),
            "yaw_rate_error_radps": _metric(
                [row["yaw_rate_error_radps"] for row in selected]),
            "paired_previous_queue_slot": _paired_alternative(
                selected, "slot1_u_error_mps"),
            "paired_next_epoch_feedback_oracle": _paired_alternative(
                selected, "next_feedback_u_error_mps"),
            "paired_same_epoch_GT_motion_state_oracle": _paired_alternative(
                selected, "oracle_motion_u_error_mps"),
            "paired_IMU_trapezoid_from_GT_start_oracle": _paired_alternative(
                selected, "imu_trapezoid_from_GT_u_error_mps"),
            "initial_u_error_vs_wheel_body_mismatch_correlation": _correlation(
                selected, "initial_source_u_error_mps",
                "rear_wheel_body_speed_mismatch_proxy_mps"),
            "initial_u_error_vs_source_rear_slip_correlation": _correlation(
                selected, "initial_source_u_error_mps",
                "max_abs_source_rear_longitudinal_slip"),
            "one_step_u_error_vs_throttle_delta_correlation": _correlation(
                selected, "u_error_mps", "queue_throttle_delta"),
        }
    return result


def _paired_alternative(rows: list[dict[str, Any]],
                        alternative_key: str) -> dict[str, Any]:
    pairs = [(row["u_error_mps"], _safe_float(row.get(alternative_key)))
             for row in rows]
    pairs = [(baseline, alternative) for baseline, alternative in pairs
             if math.isfinite(baseline) and math.isfinite(alternative)]
    if not pairs:
        return {"n": 0, "baseline_rmse_mps": None,
                "alternative_rmse_mps": None, "rmse_change_mps": None,
                "alternative_lower_abs_error": 0,
                "alternative_higher_abs_error": 0,
                "equal_abs_error": 0}
    baseline_errors = [baseline for baseline, _ in pairs]
    alternative_errors = [alternative for _, alternative in pairs]
    baseline_rmse = _metric(baseline_errors)["rmse"]
    alternative_rmse = _metric(alternative_errors)["rmse"]
    assert baseline_rmse is not None and alternative_rmse is not None
    lower = sum(abs(alternative) < abs(baseline)
                for baseline, alternative in pairs)
    higher = sum(abs(alternative) > abs(baseline)
                 for baseline, alternative in pairs)
    return {
        "n": len(pairs),
        "baseline_rmse_mps": baseline_rmse,
        "alternative_rmse_mps": alternative_rmse,
        "rmse_change_mps": alternative_rmse - baseline_rmse,
        "alternative_lower_abs_error": lower,
        "alternative_higher_abs_error": higher,
        "equal_abs_error": len(pairs) - lower - higher,
    }


def _correlation(rows: list[dict[str, Any]], x_key: str,
                 y_key: str) -> dict[str, int | float | None]:
    pairs = [(_safe_float(row.get(x_key)), _safe_float(row.get(y_key)))
             for row in rows]
    pairs = [(x, y) for x, y in pairs
             if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 2:
        return {"n": len(pairs), "pearson_r": None}
    mean_x = statistics.mean(x for x, _ in pairs)
    mean_y = statistics.mean(y for _, y in pairs)
    covariance = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
    variance_x = sum((x - mean_x) ** 2 for x, _ in pairs)
    variance_y = sum((y - mean_y) ** 2 for _, y in pairs)
    denominator = math.sqrt(variance_x * variance_y)
    return {"n": len(pairs),
            "pearson_r": covariance / denominator if denominator else None}


def _elapsed_time_summary(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Separate inherited motion-state error from each interval's new error.

    ``one_step_transition_error_mps`` is the next-state prediction error minus
    the error already present in the logged initial state. It answers whether
    that interval grew the model/truth gap; it is not an independent sample of
    absolute vehicle-speed accuracy. Time bins are relative to the first exact
    truth/plant epoch retained after the collision/fault cutoff.
    """
    result: dict[str, dict[str, Any]] = {}
    for name, lower_s, upper_s in ELAPSED_TIME_BINS_S:
        selected = [row for row in rows
                    if lower_s <= _safe_float(
                        row.get("elapsed_from_first_epoch_s")) < upper_s]

        def metric(key: str) -> dict[str, Any]:
            return _metric([_safe_float(row.get(key)) for row in selected])

        feedback_minus_queue = [
            _safe_float(row.get("throttle_feedback_next_epoch")) -
            _safe_float(row.get("queue_throttle_used"))
            for row in selected
            if math.isfinite(_safe_float(
                row.get("throttle_feedback_next_epoch"))) and
            math.isfinite(_safe_float(row.get("queue_throttle_used")))
        ]
        sent_minus_queue = [
            _safe_float(row.get("bridge_sent_throttle_start_epoch")) -
            _safe_float(row.get("queue_throttle_used"))
            for row in selected
            if math.isfinite(_safe_float(
                row.get("bridge_sent_throttle_start_epoch"))) and
            math.isfinite(_safe_float(row.get("queue_throttle_used")))
        ]
        result[name] = {
            "elapsed_interval_s": [lower_s,
                                   upper_s if math.isfinite(upper_s) else None],
            "count": len(selected),
            "initial_source_state_u_error_mps": metric(
                "initial_source_u_error_mps"),
            "one_step_next_u_error_mps": metric("u_error_mps"),
            "one_step_transition_error_mps": metric(
                "one_step_transition_error_mps"),
            "rear_slip_abs": metric("max_abs_source_rear_longitudinal_slip"),
            "rear_wheel_body_speed_mismatch_proxy_mps": metric(
                "rear_wheel_body_speed_mismatch_proxy_mps"),
            "queued_throttle": metric("queue_throttle_used"),
            "queued_throttle_delta": metric("queue_throttle_delta"),
            "next_feedback_minus_queue_throttle": _metric(
                feedback_minus_queue),
            "bridge_sent_minus_queue_throttle": _metric(sent_minus_queue),
            "interpretation": (
                "The transition error measures error added over this one-step "
                "interval; command/feedback agreement does not identify the "
                "Unity FixedUpdate at which torque was applied."
            ),
        }
    return result


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def analyze(bag: Path, library_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not library_path.is_file():
        raise FileNotFoundError(f"source plant library not found: {library_path}")
    data = evaluator.read_bag(bag)
    truth_rows = data.get(evaluator.GT, [])
    plant_rows = data.get(evaluator.PLANT, [])
    if not truth_rows or not plant_rows:
        raise ValueError("bag must contain Unity truth and /odom/unity_plant_state")

    truth: dict[int, dict[str, Any]] = {}
    plant: dict[int, dict[str, Any]] = {}
    for target, source, label in ((truth, truth_rows, "Unity truth"),
                                  (plant, plant_rows, "plant state")):
        for row in source:
            stamp_ns = row.get("stamp_ns")
            if not isinstance(stamp_ns, int):
                raise ValueError(f"{label} row lacks exact header nanoseconds")
            if stamp_ns in target:
                raise ValueError(f"duplicate {label} stamp: {stamp_ns}")
            target[stamp_ns] = row
    epochs = sorted(truth.keys() & plant.keys())
    if len(epochs) < 2:
        raise ValueError("fewer than two exact truth/plant epochs")
    capture_start_stamp_ns = epochs[0]

    cutoff_receipt_s, cutoff_reason = drift_report._collision_cutoff(data)
    if cutoff_receipt_s is not None:
        # Discard the event interval and one preceding interval. The collision
        # topic is headerless, so keep the exclusion conservative.
        epochs = [stamp for stamp in epochs
                  if truth[stamp]["bag_time"] < cutoff_receipt_s - 0.025]
    if len(epochs) < 2:
        raise ValueError("fewer than two exact epochs remain before fault cutoff")

    model = ctypes.CDLL(str(library_path.resolve()))
    step = model.unity_source_plant_step_c
    step.argtypes = [ctypes.POINTER(evaluator.UnitySourcePlantInputC),
                     ctypes.c_double,
                     ctypes.POINTER(evaluator.UnitySourcePlantOutputC)]
    step.restype = ctypes.c_int

    bridge_by_epoch: dict[int, dict[str, Any]] = {}
    for event in data.get(evaluator.BRIDGE_TIMING, []):
        payload = event.get("json", {})
        stamp = payload.get("bridge_request_ros_stamp_ns")
        if stamp is None:
            continue
        try:
            stamp_ns = int(stamp)
        except (TypeError, ValueError, OverflowError):
            continue
        if stamp_ns not in bridge_by_epoch:
            bridge_by_epoch[stamp_ns] = payload

    imu_by_epoch: dict[int, dict[str, Any]] = {}
    for row in data.get(evaluator.IMU, []):
        stamp_ns = row.get("stamp_ns")
        if not isinstance(stamp_ns, int):
            continue
        if stamp_ns in imu_by_epoch:
            raise ValueError(f"duplicate IMU stamp: {stamp_ns}")
        imu_by_epoch[stamp_ns] = row

    rows: list[dict[str, Any]] = []
    skipped_dt = 0
    failed_steps = 0
    odom_by_stamp = {row["stamp_ns"]: row
                     for row in data.get(evaluator.ODOM, [])}
    for start_ns, next_ns in zip(epochs, epochs[1:]):
        dt_s = (next_ns - start_ns) * 1.0e-9
        if not 0.020 <= dt_s <= 0.030:
            skipped_dt += 1
            continue
        start = plant[start_ns]
        actual_start = truth[start_ns]
        actual_next = truth[next_ns]
        # This source state is the production-published state after its
        # observer correction, not raw independent Unity hidden state.
        value = evaluator.UnitySourcePlantInputC()
        value.state[:] = _source_state(start)
        value.wheel_rotation_angle_rad[:] = start["wheel_angle"]
        value.steering_target_rad = (
            start["delayed_steering_2"] * evaluator.MAX_STEERING_RAD)
        value.throttle_normalized = start["delayed_throttle_2"]
        value.suspension_history_initialized = int(
            start["suspension_history_initialized"])
        output = evaluator.UnitySourcePlantOutputC()
        if not step(ctypes.byref(value), evaluator.ENCODER_SAMPLE_PERIOD_S,
                    ctypes.byref(output)):
            failed_steps += 1
            continue

        def alternate_throttle_prediction(throttle: float) -> float:
            alternate = evaluator.UnitySourcePlantInputC()
            ctypes.memmove(ctypes.byref(alternate), ctypes.byref(value),
                           ctypes.sizeof(value))
            alternate.throttle_normalized = throttle
            alternate_output = evaluator.UnitySourcePlantOutputC()
            if not step(ctypes.byref(alternate),
                        evaluator.ENCODER_SAMPLE_PERIOD_S,
                        ctypes.byref(alternate_output)):
                return math.nan
            return float(alternate_output.state[0])

        predicted_slot1_u = alternate_throttle_prediction(
            start["delayed_throttle_1"])
        feedback_next = _safe_float(
            plant.get(next_ns, {}).get("throttle_feedback", math.nan))
        predicted_next_feedback_u = (
            alternate_throttle_prediction(feedback_next)
            if math.isfinite(feedback_next) else math.nan)

        oracle_motion_input = evaluator.UnitySourcePlantInputC()
        ctypes.memmove(ctypes.byref(oracle_motion_input), ctypes.byref(value),
                       ctypes.sizeof(value))
        oracle_motion_input.state[0] = actual_start["u"]
        oracle_motion_input.state[1] = actual_start["v"]
        oracle_motion_input.state[2] = actual_start["r"]
        oracle_motion_output = evaluator.UnitySourcePlantOutputC()
        oracle_motion_ok = step(
            ctypes.byref(oracle_motion_input),
            evaluator.ENCODER_SAMPLE_PERIOD_S,
            ctypes.byref(oracle_motion_output))
        predicted_oracle_motion_u = (
            float(oracle_motion_output.state[0]) if oracle_motion_ok
            else math.nan)

        imu_start = imu_by_epoch.get(start_ns)
        imu_next = imu_by_epoch.get(next_ns)
        if imu_start is not None and imu_next is not None:
            accel_start = (imu_start["ax"] + actual_start["r"] *
                           actual_start["v"])
            accel_next = (imu_next["ax"] + actual_next["r"] *
                          actual_next["v"])
            imu_delta_u = 0.5 * (accel_start + accel_next) * dt_s
            imu_source_start_error = (
                start["u"] + imu_delta_u - actual_next["u"])
            imu_gt_start_error = (
                actual_start["u"] + imu_delta_u - actual_next["u"])
        else:
            accel_start = accel_next = imu_delta_u = math.nan
            imu_source_start_error = imu_gt_start_error = math.nan

        bridge_start = bridge_by_epoch.get(start_ns, {})
        bridge_next = bridge_by_epoch.get(next_ns, {})
        start_odom = odom_by_stamp.get(start_ns)
        rear_wheel_speed = evaluator.WHEEL_RADIUS_M * 0.5 * (
            start["wheel_omega"][2] + start["wheel_omega"][3])
        rows.append({
            "start_stamp_ns": start_ns,
            "next_stamp_ns": next_ns,
            "elapsed_from_first_epoch_s":
                (start_ns - capture_start_stamp_ns) * 1.0e-9,
            "dt_ms": dt_s * 1000.0,
            "start_speed_mps": math.hypot(actual_start["u"], actual_start["v"]),
            "start_gt_u_mps": actual_start["u"],
            "start_source_u_mps": start["u"],
            "initial_source_u_error_mps": start["u"] - actual_start["u"],
            "start_odom_u_mps": (start_odom["u"] if start_odom is not None
                                  else math.nan),
            "initial_odom_u_error_mps": (
                start_odom["u"] - actual_start["u"]
                if start_odom is not None else math.nan),
            "start_gt_v_com_mps": actual_start["v"],
            "start_source_v_com_mps": start["v_com"],
            "initial_source_v_com_error_mps": (
                start["v_com"] - actual_start["v"]),
            "start_gt_yaw_rate_radps": actual_start["r"],
            "start_source_yaw_rate_radps": start["r"],
            "initial_source_yaw_rate_error_radps": (
                start["r"] - actual_start["r"]),
            "next_gt_u_mps": actual_next["u"],
            "next_odom_u_mps": math.nan,
            "predicted_u_mps": output.state[0],
            "u_error_mps": output.state[0] - actual_next["u"],
            "one_step_transition_error_mps": (
                output.state[0] - actual_next["u"] -
                (start["u"] - actual_start["u"])),
            "predicted_slot1_u_mps": predicted_slot1_u,
            "slot1_u_error_mps": predicted_slot1_u - actual_next["u"],
            "predicted_next_feedback_u_mps": predicted_next_feedback_u,
            "next_feedback_u_error_mps": (
                predicted_next_feedback_u - actual_next["u"]),
            "predicted_oracle_motion_u_mps": predicted_oracle_motion_u,
            "oracle_motion_u_error_mps": (
                predicted_oracle_motion_u - actual_next["u"]),
            "start_imu_ax_mps2": (imu_start["ax"] if imu_start else math.nan),
            "next_imu_ax_mps2": (imu_next["ax"] if imu_next else math.nan),
            "IMU_body_longitudinal_accel_start_mps2": accel_start,
            "IMU_body_longitudinal_accel_next_mps2": accel_next,
            "IMU_trapezoid_delta_u_mps": imu_delta_u,
            "imu_trapezoid_from_source_u_error_mps": imu_source_start_error,
            "imu_trapezoid_from_GT_u_error_mps": imu_gt_start_error,
            "production_odom_u_error_mps": math.nan,
            "predicted_v_com_mps": output.state[1],
            "v_com_error_mps": output.state[1] - actual_next["v"],
            "predicted_yaw_rate_radps": output.state[2],
            "yaw_rate_error_radps": output.state[2] - actual_next["r"],
            "physical_steering_rad": start["physical_steering"],
            "command_steering_rad": value.steering_target_rad,
            "queue_throttle_previous": start["delayed_throttle_1"],
            "queue_throttle_used": value.throttle_normalized,
            "queue_throttle_delta": (
                value.throttle_normalized - start["delayed_throttle_1"]),
            "throttle_feedback_next_epoch": feedback_next,
            "bridge_sent_throttle_start_epoch": _safe_float(
                bridge_start.get("sent_throttle_norm")),
            "bridge_feedback_throttle_next_epoch": _safe_float(
                bridge_next.get("simulator_feedback_throttle_norm")),
            "bridge_physics_step_start": bridge_start.get(
                "simulation_physics_step"),
            "bridge_physics_step_next": bridge_next.get(
                "simulation_physics_step"),
            "rear_wheel_linear_speed_proxy_mps": rear_wheel_speed,
            "rear_wheel_body_speed_mismatch_proxy_mps": (
                rear_wheel_speed - actual_start["u"]),
            "max_abs_source_rear_longitudinal_slip": max(
                abs(start["long_slip"][2]), abs(start["long_slip"][3])),
            "source_rear_wheel_omega_radps": 0.5 * (
                start["wheel_omega"][2] + start["wheel_omega"][3]),
        })

    for row in rows:
        next_odom = odom_by_stamp.get(row["next_stamp_ns"])
        if next_odom is not None:
            row["next_odom_u_mps"] = next_odom["u"]
            row["production_odom_u_error_mps"] = (
                next_odom["u"] - row["next_gt_u_mps"])

    partitions: list[tuple[str, Callable[[dict[str, Any]], bool]]] = []
    for lo, hi in SPEED_BINS:
        partitions.append((f"speed_{lo:g}_{hi:g}_mps",
                           lambda row, lo=lo, hi=hi:
                           lo <= row["start_speed_mps"] < hi))
    for lo, hi in THROTTLE_BINS:
        partitions.append((f"throttle_{lo:g}_{hi:g}",
                           lambda row, lo=lo, hi=hi:
                           lo <= row["queue_throttle_used"] < hi))
    partitions.extend((
        ("throttle_rising_gt_0p10", lambda row: row["queue_throttle_delta"] > 0.10),
        ("throttle_falling_lt_minus_0p10", lambda row: row["queue_throttle_delta"] < -0.10),
        ("throttle_change_abs_le_0p10", lambda row: abs(row["queue_throttle_delta"]) <= 0.10),
        ("near_straight_abs_steer_lt_0p10", lambda row: abs(row["physical_steering_rad"]) < 0.10),
        ("steer_abs_ge_0p10", lambda row: abs(row["physical_steering_rad"]) >= 0.10),
        ("rear_wheel_mismatch_proxy_gt_0p5", lambda row: abs(row["rear_wheel_body_speed_mismatch_proxy_mps"]) > 0.5),
        ("source_rear_slip_abs_gt_0p10", lambda row: row["max_abs_source_rear_longitudinal_slip"] > 0.10),
    ))
    transition_predicates = {
        "rising": lambda row: row["queue_throttle_delta"] > 0.10,
        "falling": lambda row: row["queue_throttle_delta"] < -0.10,
        "stable": lambda row: abs(row["queue_throttle_delta"]) <= 0.10,
    }
    for speed_lo, speed_hi in ((3.0, 4.0), (4.0, 5.0), (5.0, 8.0)):
        for steer_name, steer_predicate in (
                ("straight", lambda row: abs(row["physical_steering_rad"]) < 0.10),
                ("turn", lambda row: abs(row["physical_steering_rad"]) >= 0.10)):
            for transition_name, transition_predicate in transition_predicates.items():
                name = (f"joint_speed_{speed_lo:g}_{speed_hi:g}_"
                        f"{steer_name}_{transition_name}")
                partitions.append((name, lambda row, lo=speed_lo, hi=speed_hi,
                                   steer=steer_predicate,
                                   transition=transition_predicate:
                                   lo <= row["start_speed_mps"] < hi
                                   and steer(row) and transition(row)))
    collision_messages = data.get(evaluator.COLLISION, [])
    collision_count = max((int(row["value"]) for row in collision_messages),
                          default=0)
    feedback_pairs = [row for row in rows
                      if math.isfinite(row["throttle_feedback_next_epoch"])
                      and math.isfinite(row["bridge_sent_throttle_start_epoch"])]
    feedback_alignment = [
        row["throttle_feedback_next_epoch"] -
        row["bridge_sent_throttle_start_epoch"] for row in feedback_pairs]
    summary = {
        "schema": SCHEMA,
        "bag": str(bag),
        "bag_sha256": _hash_bag(bag),
        "library": str(library_path),
        "library_sha256": _file_hash(library_path),
        "plant_source_sha256": _file_hash(
            Path(__file__).resolve().parents[1] /
            "f1tenth_mpc/src/unity_source_plant.cpp"),
        "exact_truth_plant_epochs": len(epochs),
        "one_step_rows": len(rows),
        "skipped_non_20_to_30ms_intervals": skipped_dt,
        "failed_source_steps": failed_steps,
        "collision_count_max": collision_count,
        "conservative_fault_cutoff_receipt_s": cutoff_receipt_s,
        "conservative_fault_cutoff_reason": cutoff_reason,
        "input_semantics": (
            "production-published sensor-corrected plant state plus the logged "
            "newest (slot-2) queued steering/throttle for one nominal 25ms "
            "source-model step; truth is only the next-epoch score"),
        "command_epoch_limits": [
            "Queue slot is a host-side ROS association, not a Unity applied-command sequence.",
            "Bridge feedback at the next request can confirm feedback/request lag, not the exact FixedUpdate torque-application epoch.",
            "Wheel/body mismatch is an encoder/estimated-wheel proxy, not independent per-wheel Unity truth.",
            "Plant initial state is the production-published corrected state; this is not a raw uncorrected plant-state test.",
        ],
        "next_epoch_feedback_minus_start_epoch_sent_command": _metric(
            feedback_alignment),
        "paired_alternative_command_comparison": {
            "previous_queue_slot": _paired_alternative(
                rows, "slot1_u_error_mps"),
            "next_epoch_throttle_feedback_oracle": _paired_alternative(
                rows, "next_feedback_u_error_mps"),
            "same_epoch_GT_motion_state_oracle": _paired_alternative(
                rows, "oracle_motion_u_error_mps"),
            "IMU_trapezoid_from_GT_start_oracle": _paired_alternative(
                rows, "imu_trapezoid_from_GT_u_error_mps"),
        },
        "overall_correlations": {
            "initial_u_error_vs_wheel_body_mismatch": _correlation(
                rows, "initial_source_u_error_mps",
                "rear_wheel_body_speed_mismatch_proxy_mps"),
            "initial_u_error_vs_source_rear_slip": _correlation(
                rows, "initial_source_u_error_mps",
                "max_abs_source_rear_longitudinal_slip"),
            "one_step_u_error_vs_throttle_delta": _correlation(
                rows, "u_error_mps", "queue_throttle_delta"),
        },
        "oracle_semantics": (
            "same-epoch GT u/v/r replace only the initial visible motion state; "
            "hidden wheel/suspension states and recorded command remain as "
            "logged. Diagnostic oracle only, never a production input."),
        "IMU_acceleration_semantics": (
            "Diagnostic only: trapezoid over adjacent 40Hz IMU samples of "
            "ax + r*v_COM; IMU acceleration is a sampled Unity FixedUpdate "
            "value, so this is not asserted to be the true interval-average "
            "acceleration or a production estimator."),
        "by_regime": _groups(rows, partitions),
        "by_elapsed_since_first_exact_epoch": _elapsed_time_summary(rows),
    }
    return summary, rows


def _hash_bag(bag: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(list(bag.rglob("*.db3")) + list(bag.rglob("*.mcap")))
    for path in files:
        digest.update(path.name.encode())
        digest.update(bytes.fromhex(_file_hash(path)))
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    summary, rows = analyze(args.bag, args.library)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = args.output_prefix.with_suffix(".json")
    csv_path = args.output_prefix.with_suffix(".csv")
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"wrote {json_path} and {csv_path}")
    print(f"exact one-step rows={summary['one_step_rows']}; "
          f"skipped intervals={summary['skipped_non_20_to_30ms_intervals']}; "
          f"failed source steps={summary['failed_source_steps']}")
    for name, values in summary["by_regime"].items():
        metric = values["forward_velocity_error_mps"]
        print(f"{name}: n={metric['n']} u_bias={metric['bias']} "
              f"u_RMSE={metric['rmse']} p95={metric['p95_abs']} "
              f"max={metric['max_abs']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
