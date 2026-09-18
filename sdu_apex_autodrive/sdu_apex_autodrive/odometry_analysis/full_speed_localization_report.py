"""Score a source-stamped full-speed run without feeding truth to runtime.

The model-ID timing recorder stores sensor/runtime events while the bridge
diagnostic stream stores simulator truth in the same packet order.  This
offline report joins those streams by the source header stamp and reports the
state/timing gates from the MPC recovery handoff.  Simulator truth is never
read by a runtime node or controller.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable


def _finite(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _angle_diff(a: float, b: float) -> float:
    return math.atan2(math.sin(a - b), math.cos(a - b))


def _yaw_from_quaternion(payload: dict[str, object]) -> float | None:
    x = _finite(payload.get("simulator_orientation_quaternion_x"))
    y = _finite(payload.get("simulator_orientation_quaternion_y"))
    z = _finite(payload.get("simulator_orientation_quaternion_z"))
    w = _finite(payload.get("simulator_orientation_quaternion_w"))
    if None in (x, y, z, w):
        return None
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"samples": 0, "p50": None, "p90": None, "p95": None,
                "p99": None, "max": None, "mean": None, "bias": None}
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        position = fraction * (len(ordered) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] + weight * (ordered[upper] - ordered[lower])

    return {
        "samples": len(values),
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": max(values),
        "mean": sum(values) / len(values),
        "bias": sum(values) / len(values),
    }


def _read_events(path: Path) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            grouped.setdefault(row.get("topic", ""), []).append(row)
    return grouped


def _payload(row: dict[str, str]) -> dict[str, object]:
    try:
        value = json.loads(row.get("payload_json", "{}"))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _metric(values: list[float], *, absolute: bool = True) -> dict[str, object]:
    finite = [value for value in values if math.isfinite(value)]
    if absolute:
        return _summary([abs(value) for value in finite])
    return _summary(finite)


def _position_error_truth_frame(
        rows: list[dict[str, object]]) -> dict[str, object]:
    """Project offline position residuals onto simulator truth heading axes."""
    longitudinal: list[float] = []
    lateral: list[float] = []
    for row in rows:
        x_est = _finite(row.get("estimate_x_m"))
        y_est = _finite(row.get("estimate_y_m"))
        x_truth = _finite(row.get("truth_x_m"))
        y_truth = _finite(row.get("truth_y_m"))
        yaw_truth = _finite(row.get("truth_yaw_rad"))
        if None in (x_est, y_est, x_truth, y_truth, yaw_truth):
            continue
        dx = float(x_est) - float(x_truth)
        dy = float(y_est) - float(y_truth)
        yaw = float(yaw_truth)
        longitudinal.append(dx * math.cos(yaw) + dy * math.sin(yaw))
        lateral.append(-dx * math.sin(yaw) + dy * math.cos(yaw))
    return {
        "offline_only": True,
        "frame": "simulator_truth_yaw",
        "longitudinal_error_abs_m": _metric(longitudinal),
        "longitudinal_error_signed_m": _metric(longitudinal, absolute=False),
        "lateral_error_abs_m": _metric(lateral),
        "lateral_error_signed_m": _metric(lateral, absolute=False),
    }


def _motion_regime_scores(
        report_rows: list[dict[str, object]]) -> dict[str, object]:
    """Score offline-only sensor-odometry errors by plant command/regime.

    The bridge's applied throttle and simulator yaw rate are used only to label
    recorded samples. They are not controller or estimator inputs. In this
    competition vehicle, zero applied throttle selects the configured brake
    branch; it must not be called coast-down data.
    """
    grouped: dict[str, dict[str, list[dict[str, object]]]] = {}
    for row in report_rows:
        if row.get("topic") not in ("/odom", "/ekf_odom"):
            continue
        throttle = _finite(row.get("offline_applied_throttle_norm"))
        yaw_rate = _finite(row.get("offline_truth_yaw_rate_radps"))
        pose_u = _finite(row.get("truth_pose_u_mps"))
        pose_v = _finite(row.get("truth_pose_v_mps"))
        if None in (throttle, yaw_rate, pose_u, pose_v):
            continue
        if math.hypot(pose_u, pose_v) <= 0.5:
            continue
        if abs(throttle) <= 1.0e-4:
            power_mode = "full_brake"
        elif throttle > 1.0e-4:
            power_mode = "powered"
        else:
            continue
        steering_mode = "straight" if abs(yaw_rate) < 0.1 else "turn"
        regime = f"{power_mode}_{steering_mode}"
        topic = str(row["topic"])
        grouped.setdefault(regime, {}).setdefault(topic, []).append(row)

    scores: dict[str, object] = {}
    for name, topics in sorted(grouped.items()):
        topic_scores: dict[str, object] = {}
        for topic, rows in sorted(topics.items()):
            u_errors = [
                value for row in rows
                if isinstance((value := _finite(row.get("u_error_mps"))), float)]
            com_u_errors = [
                value for row in rows
                if isinstance((value := _finite(
                    row.get("u_error_mps_com"))), float)]
            v_errors = [
                value for row in rows
                if isinstance((value := _finite(row.get("v_error_mps"))), float)]
            longitudinal_errors = [
                value for row in rows
                if isinstance((value := _finite(
                    row.get("longitudinal_error_truth_frame_m"))), float)]
            lateral_errors = [
                value for row in rows
                if isinstance((value := _finite(
                    row.get("lateral_error_truth_frame_m"))), float)]
            observer_modes = [
                value for row in rows
                if isinstance((value := _finite(
                    row.get("odom_observer_wheel_update_used"))), float)]
            observer_prediction_errors = [
                value for row in rows
                if isinstance((value := _finite(
                    row.get("odom_observer_speed_prediction_error_mps"))), float)]
            topic_scores[topic] = {
                "samples": len(rows),
                "u_error_mps": _metric(u_errors),
                "u_signed_error_mps": _metric(u_errors, absolute=False),
                "u_error_mps_com": _metric(com_u_errors),
                "u_signed_error_mps_com": _metric(
                    com_u_errors, absolute=False),
                "v_error_mps": _metric(v_errors),
                "v_signed_error_mps": _metric(v_errors, absolute=False),
                "longitudinal_position_error_truth_frame_m": _metric(
                    longitudinal_errors),
                "longitudinal_position_error_truth_frame_signed_m": _metric(
                    longitudinal_errors, absolute=False),
                "lateral_position_error_truth_frame_m": _metric(lateral_errors),
                "lateral_position_error_truth_frame_signed_m": _metric(
                    lateral_errors, absolute=False),
                "observer_wheel_usage": {
                    "samples": len(observer_modes),
                    "wheel_update_fraction": (
                        sum(observer_modes) / len(observer_modes)
                        if observer_modes else None),
                    "imu_prediction_u_error_mps": _metric(
                        observer_prediction_errors),
                },
                # Euclidean distance is secondary to the signed axis errors.
                "position_error_euclidean_m": _metric([
                    value for row in rows
                    if isinstance((value := _finite(
                        row.get("position_error_m"))), float)]),
            }
        scores[name] = {
            "samples_by_topic": {
                topic: len(rows) for topic, rows in sorted(topics.items())},
            "topics": topic_scores,
        }
    return {
        "label_policy": {
            "offline_only": True,
            "moving_speed_min_mps": 0.5,
            "full_brake": "abs(applied_throttle_norm) <= 0.0001",
            "powered": "applied_throttle_norm > 0.0001",
            "straight": "abs(simulator_yaw_rate_radps) < 0.1",
            "turn": "abs(simulator_yaw_rate_radps) >= 0.1",
        },
        "groups": scores,
    }


def _delivery_timing_report(
        timing_rows: list[dict[str, str]],
        events: dict[str, list[dict[str, str]]]) -> dict[str, object]:
    """Separate Unity source cadence, request pacing, transport, and ROS arrival.

    Monotonic bridge timestamps identify when responses reach the bridge. The
    recorder's monotonic callback timestamp is a separate downstream stage;
    its epoch timestamp, when present, also permits source-header age scoring.
    """
    packets: list[dict[str, float | None]] = []
    for row in timing_rows:
        payload = _payload(row)
        packets.append({
            "source_s": _finite(payload.get("simulation_time_s")),
            "request_ns": _finite(payload.get("request_monotonic_ns")),
            "bridge_arrival_ns": _finite(
                payload.get("bridge_arrival_monotonic_ns")),
            "event_arrival_ns": _finite(row.get("arrival_monotonic_ns")),
            "request_sequence": _finite(payload.get("request_sequence")),
            "physics_step": _finite(payload.get("simulation_physics_step")),
            "render_frame": _finite(payload.get("simulation_render_frame")),
        })

    def differences(key: str, scale: float) -> list[float]:
        return [
            (float(current[key]) - float(previous[key])) / scale
            for previous, current in zip(packets, packets[1:])
            if previous[key] is not None and current[key] is not None
        ]

    source_dt = differences("source_s", 1.0)
    request_dt = differences("request_ns", 1.0e9)
    bridge_arrival_dt = differences("bridge_arrival_ns", 1.0e9)
    recorder_callback_dt = differences("event_arrival_ns", 1.0e9)
    physics_step_delta = differences("physics_step", 1.0)
    render_frame_delta = differences("render_frame", 1.0)
    response_wait = [
        (float(packet["bridge_arrival_ns"]) - float(packet["request_ns"])) /
        1.0e9
        for packet in packets
        if packet["bridge_arrival_ns"] is not None and
        packet["request_ns"] is not None
    ]
    bridge_to_recorder = [
        (float(packet["event_arrival_ns"]) -
         float(packet["bridge_arrival_ns"])) / 1.0e9
        for packet in packets
        if packet["event_arrival_ns"] is not None and
        packet["bridge_arrival_ns"] is not None
    ]
    nominal_period_s = 0.025
    compensation_tolerance_s = 0.002
    short_arrivals = sum(interval < 0.015 for interval in bridge_arrival_dt)
    long_arrivals = sum(interval > 0.035 for interval in bridge_arrival_dt)
    short_then_long = sum(
        earlier < 0.015 and later > 0.035
        for earlier, later in zip(bridge_arrival_dt, bridge_arrival_dt[1:]))
    late_then_early = sum(
        earlier > nominal_period_s and later < nominal_period_s
        for earlier, later in zip(bridge_arrival_dt, bridge_arrival_dt[1:]))
    early_then_late = sum(
        earlier < nominal_period_s and later > nominal_period_s
        for earlier, later in zip(bridge_arrival_dt, bridge_arrival_dt[1:]))
    near_nominal_late_then_early = sum(
        earlier > nominal_period_s and later < nominal_period_s and
        abs(earlier + later - 2.0 * nominal_period_s) <=
        compensation_tolerance_s
        for earlier, later in zip(bridge_arrival_dt, bridge_arrival_dt[1:]))
    near_nominal_early_then_late = sum(
        earlier < nominal_period_s and later > nominal_period_s and
        abs(earlier + later - 2.0 * nominal_period_s) <=
        compensation_tolerance_s
        for earlier, later in zip(bridge_arrival_dt, bridge_arrival_dt[1:]))
    user_30ms_then_20ms_like = sum(
        0.028 <= earlier <= 0.032 and 0.018 <= later <= 0.022 and
        abs(earlier + later - 2.0 * nominal_period_s) <=
        compensation_tolerance_s
        for earlier, later in zip(bridge_arrival_dt, bridge_arrival_dt[1:]))
    source_outside_nominal_window = sum(
        interval < 0.015 - 1.0e-6 or interval > 0.035 + 1.0e-6
        for interval in source_dt)
    source_outside_supported_range = sum(
        interval < 0.001 - 1.0e-6 or interval > 0.250 + 1.0e-6
        for interval in source_dt)
    source_degraded_intervals = sum(
        interval > 0.035 + 1.0e-6 for interval in source_dt)

    topic_timing: dict[str, object] = {}
    monitored_topics = (
        "/autodrive/roboracer_1/imu",
        "/autodrive/roboracer_1/left_encoder",
        "/autodrive/roboracer_1/right_encoder",
        "/odom",
        "/ekf_odom",
        "/amcl_pose",
        "/current_map_pose",
        "/cmd/speed",
        "/pure_pursuit/diagnostics",
    )
    for topic in monitored_topics:
        rows = events.get(topic, [])
        arrival_ns = [_finite(row.get("arrival_monotonic_ns")) for row in rows]
        callback_dt = [
            (float(current) - float(previous)) / 1.0e9
            for previous, current in zip(arrival_ns, arrival_ns[1:])
            if previous is not None and current is not None
        ]
        header_ns = [_finite(row.get("header_stamp_ns")) for row in rows]
        header_dt = [
            (float(current) - float(previous)) / 1.0e9
            for previous, current in zip(header_ns, header_ns[1:])
            if previous is not None and current is not None
        ]
        header_age = [
            (float(epoch_ns) - float(stamp_ns)) / 1.0e9
            for row in rows
            if (stamp_ns := _finite(row.get("header_stamp_ns"))) is not None
            and (epoch_ns := _finite(row.get("arrival_epoch_ns"))) is not None
        ]
        if rows:
            topic_timing[topic] = {
                "callback_interval_s": _metric(callback_dt, absolute=False),
                "header_interval_s": _metric(header_dt, absolute=False),
                "header_age_on_callback_s": _metric(
                    header_age, absolute=False),
            }

    fault_details = [
        str(_payload(row).get("value", ""))
        for row in events.get(
            "/autodrive/roboracer_1/bridge_timing_fault_detail", [])
    ]
    return {
        "source_interval_s": _metric(source_dt, absolute=False),
        "request_interval_s": _metric(request_dt, absolute=False),
        "bridge_response_arrival_interval_s": _metric(
            bridge_arrival_dt, absolute=False),
        "bridge_arrival_nominal_period_s": nominal_period_s,
        "request_to_bridge_arrival_s": _metric(response_wait, absolute=False),
        "bridge_to_recorder_callback_s": _metric(
            bridge_to_recorder, absolute=False),
        "recorder_callback_interval_s": _metric(
            recorder_callback_dt, absolute=False),
        "physics_step_delta": _metric(physics_step_delta, absolute=False),
        "render_frame_delta": _metric(render_frame_delta, absolute=False),
        # The 15--35 ms band is nominal cadence, not a bridge acceptance rule.
        # Keep it as a diagnostic alongside the observer's supported range.
        "source_intervals_outside_nominal_15_35ms": source_outside_nominal_window,
        "source_intervals_degraded_over_35ms": source_degraded_intervals,
        "source_intervals_outside_supported_1_250ms": source_outside_supported_range,
        "bridge_arrival_intervals_under_15ms": short_arrivals,
        "bridge_arrival_intervals_over_35ms": long_arrivals,
        "short_then_long_arrival_pairs": short_then_long,
        # The nominal 15--35 ms source-cadence counters above do not answer whether
        # a 30 ms arrival gap is followed by a 20 ms gap. These separate
        # host-arrival pairs around the nominal 25 ms period without changing
        # the source timestamps or their measured intervals.
        "bridge_arrival_late_intervals_vs_25ms": sum(
            interval > nominal_period_s for interval in bridge_arrival_dt),
        "bridge_arrival_early_intervals_vs_25ms": sum(
            interval < nominal_period_s for interval in bridge_arrival_dt),
        "bridge_arrival_late_then_early_pairs": late_then_early,
        "bridge_arrival_early_then_late_pairs": early_then_late,
        "near_50ms_late_then_early_pairs_pm2ms": near_nominal_late_then_early,
        "near_50ms_early_then_late_pairs_pm2ms": near_nominal_early_then_late,
        "user_30ms_then_20ms_like_pairs": user_30ms_then_20ms_like,
        "topic_timing": topic_timing,
        "fault_details": fault_details,
    }


def build_report(run_dir: str | Path, output_json: str | Path | None = None,
                 output_csv: str | Path | None = None) -> dict[str, object]:
    root = Path(run_dir)
    events = _read_events(root / "events.csv")
    timing_rows = events.get("/autodrive/roboracer_1/bridge_packet_timing", [])
    imu_rows = events.get("/autodrive/roboracer_1/imu", [])
    if len(timing_rows) < 2 or len(timing_rows) != len(imu_rows):
        raise ValueError("bridge timing and IMU packet streams are incomplete")

    truth_by_stamp: dict[str, dict[str, object]] = {}
    # Float64MultiArray diagnostics have no Header. Their first two fields are
    # version and source epoch seconds; millisecond keys safely match those to
    # nanosecond sensor headers (Unity source time advances in 1 ms fixed
    # steps) despite the precision loss in the Float64 representation.
    odom_diagnostics_by_stamp: dict[int, list[object]] = {}
    for row in events.get("/odom/diagnostics", []):
        diagnostic_data = _payload(row).get("data")
        if isinstance(diagnostic_data, list) and len(diagnostic_data) > 1:
            diagnostic_stamp = _finite(diagnostic_data[1])
            if diagnostic_stamp is not None:
                odom_diagnostics_by_stamp[round(diagnostic_stamp * 1000.0)] = (
                    diagnostic_data)
    timing_dts: list[float] = []
    command_lags: list[float] = []
    previous_time: float | None = None
    previous_truth_pose: tuple[float, float, float, float] | None = None
    sequence_gaps = 0
    previous_sequence: int | None = None
    for timing, imu in zip(timing_rows, imu_rows):
        bridge = _payload(timing)
        source_stamp = imu.get("header_stamp_ns", "")
        truth = {
            "x_m": _finite(bridge.get("simulator_position_x")),
            "y_m": _finite(bridge.get("simulator_position_y")),
            "yaw_rad": _yaw_from_quaternion(bridge),
            "u_mps": _finite(bridge.get("simulator_linear_velocity_x")),
            # The raw simulator velocity is measured at the rigidbody/COM,
            # while the reported simulator position is the GPS vehicle-frame
            # point. Keep the COM value for diagnostics, but do not compare it
            # directly with the odometry twist of the reported pose point.
            "com_v_mps": _finite(bridge.get("simulator_linear_velocity_y")),
            "r_radps": _finite(bridge.get("simulator_angular_velocity_z")),
            "applied_throttle_norm": _finite(
                bridge.get("simulator_feedback_throttle_norm")),
            "simulation_time_s": _finite(bridge.get("simulation_time_s")),
        }
        source_time = truth["simulation_time_s"]
        if all(isinstance(truth.get(key), float) for key in
               ("x_m", "y_m", "yaw_rad", "simulation_time_s")):
            current_pose = (
                float(truth["x_m"]), float(truth["y_m"]),
                float(truth["yaw_rad"]), float(source_time))
            if previous_truth_pose is not None:
                previous_x, previous_y, previous_yaw, previous_stamp = previous_truth_pose
                dt_pose = current_pose[3] - previous_stamp
                if dt_pose > 0.0:
                    yaw_mid = previous_yaw + 0.5 * _angle_diff(
                        current_pose[2], previous_yaw)
                    dx = current_pose[0] - previous_x
                    dy = current_pose[1] - previous_y
                    truth["pose_u_mps"] = (
                        (dx * math.cos(yaw_mid) + dy * math.sin(yaw_mid)) /
                        dt_pose)
                    truth["pose_v_mps"] = (
                        (-dx * math.sin(yaw_mid) + dy * math.cos(yaw_mid)) /
                        dt_pose)
            previous_truth_pose = current_pose
        else:
            previous_truth_pose = None
        truth.setdefault("pose_u_mps", None)
        truth.setdefault("pose_v_mps", None)
        if source_stamp:
            truth_by_stamp[source_stamp] = truth
        if isinstance(source_time, float) and previous_time is not None:
            timing_dts.append(source_time - previous_time)
        if isinstance(source_time, float):
            previous_time = source_time
        sequence = _finite(bridge.get("packet_sequence"))
        if sequence is not None:
            integer_sequence = int(sequence)
            if previous_sequence is not None and integer_sequence != previous_sequence + 1:
                sequence_gaps += integer_sequence - previous_sequence - 1
            previous_sequence = integer_sequence
        request = _finite(bridge.get("request_sequence"))
        applied = _finite(bridge.get("applied_command_sequence"))
        if request is not None and applied is not None:
            command_lags.append(request - applied)

    first_truth = next(iter(truth_by_stamp.values()))
    initial_yaw = first_truth["yaw_rad"]
    if not isinstance(initial_yaw, float):
        raise ValueError("ground-truth stream has no finite initial yaw")
    initial_x = float(first_truth["x_m"] or 0.0)
    initial_y = float(first_truth["y_m"] or 0.0)

    report_rows: list[dict[str, object]] = []
    for topic in ("/odom", "/ekf_odom", "/amcl_pose", "/current_map_pose"):
        for row in events.get(topic, []):
            stamp = row.get("header_stamp_ns", "")
            truth = truth_by_stamp.get(stamp)
            if truth is None:
                continue
            payload = _payload(row)
            x = _finite(payload.get("x_m"))
            y = _finite(payload.get("y_m"))
            if None in (x, y, truth["x_m"], truth["y_m"]):
                continue
            if topic in ("/odom", "/ekf_odom"):
                # The local odom frame is initialized body-aligned at the
                # first coherent packet.  Convert it only for offline scoring.
                dx = x * math.cos(initial_yaw) - y * math.sin(initial_yaw)
                dy = x * math.sin(initial_yaw) + y * math.cos(initial_yaw)
                estimate_x = initial_x + dx
                estimate_y = initial_y + dy
                local_yaw = _finite(payload.get("yaw_rad"))
                estimate_yaw = (
                    _angle_diff(initial_yaw + local_yaw, 0.0)
                    if local_yaw is not None else None)
            else:
                estimate_x = x
                estimate_y = y
                estimate_yaw = _finite(payload.get("yaw_rad"))
            xy_covariance = [
                value for value in (
                    _finite(payload.get("covariance_xx_m2")),
                    _finite(payload.get("covariance_yy_m2")))
                if value is not None
            ]
            truth_yaw = _finite(truth["yaw_rad"])
            world_dx = estimate_x - float(truth["x_m"])
            world_dy = estimate_y - float(truth["y_m"])
            longitudinal_error = (
                world_dx * math.cos(truth_yaw) + world_dy * math.sin(truth_yaw)
                if truth_yaw is not None else None)
            lateral_error = (
                -world_dx * math.sin(truth_yaw) + world_dy * math.cos(truth_yaw)
                if truth_yaw is not None else None)
            observer_data = (
                odom_diagnostics_by_stamp.get(round(int(stamp) / 1.0e6))
                if topic == "/odom" and stamp.isdigit() else None)
            if not isinstance(observer_data, list):
                observer_data = []

            def observer_value(index: int) -> float | None:
                return (_finite(observer_data[index])
                        if len(observer_data) > index else None)

            observer_prediction = observer_value(5)
            truth_pose_u = _finite(truth["pose_u_mps"])
            truth_com_u = _finite(truth["u_mps"])
            report_rows.append({
                "topic": topic,
                "source_stamp_ns": stamp,
                "truth_x_m": truth["x_m"],
                "truth_y_m": truth["y_m"],
                "estimate_x_m": estimate_x,
                "estimate_y_m": estimate_y,
                "position_error_m": math.hypot(
                    world_dx, world_dy),
                "longitudinal_error_truth_frame_m": longitudinal_error,
                "lateral_error_truth_frame_m": lateral_error,
                "covariance_xy_max_m2": (
                    max(xy_covariance) if xy_covariance else None),
                "covariance_yaw_rad2": _finite(
                    payload.get("covariance_yawyaw_rad2")),
                "estimate_yaw_rad": estimate_yaw,
                "truth_yaw_rad": truth["yaw_rad"],
                "truth_pose_u_mps": truth["pose_u_mps"],
                "truth_pose_v_mps": truth["pose_v_mps"],
                "truth_com_u_mps": truth["u_mps"],
                "truth_com_v_mps": truth["com_v_mps"],
                "offline_applied_throttle_norm": truth["applied_throttle_norm"],
                "offline_truth_yaw_rate_radps": truth["r_radps"],
                "odom_observer_wheel_raw_speed_mps": observer_value(3),
                "odom_observer_wheel_mapped_speed_mps": observer_value(4),
                "odom_observer_speed_prediction_mps": observer_prediction,
                "odom_observer_wheel_update_used": observer_value(12),
                "odom_observer_wheel_packet_speed_mps": observer_value(25),
                "odom_observer_wheel_burst_rejected": observer_value(26),
                "odom_observer_speed_prediction_error_mps": (
                    observer_prediction - truth_pose_u
                    if observer_prediction is not None and
                    truth_pose_u is not None else None),
                "u_error_mps": (
                    _finite(payload.get("speed_mps")) - float(truth["pose_u_mps"])
                    if topic in ("/odom", "/ekf_odom") and
                    _finite(payload.get("speed_mps")) is not None and
                    isinstance(truth["pose_u_mps"], float) else None),
                # Keep the simulator COM body-forward speed as a second,
                # offline-only reference. A longitudinal COM-to-rear-axle
                # offset contributes only lateral rigid-body point velocity.
                "u_error_mps_com": (
                    _finite(payload.get("speed_mps")) - truth_com_u
                    if topic in ("/odom", "/ekf_odom") and
                    _finite(payload.get("speed_mps")) is not None and
                    truth_com_u is not None else None),
                "v_error_mps": (
                    _finite(payload.get("lateral_speed_mps")) - float(truth["pose_v_mps"])
                    if topic in ("/odom", "/ekf_odom") and
                    _finite(payload.get("lateral_speed_mps")) is not None and
                    isinstance(truth["pose_v_mps"], float) else None),
                "v_error_mps_com": (
                    _finite(payload.get("lateral_speed_mps")) - float(truth["com_v_mps"])
                    if topic in ("/odom", "/ekf_odom") and
                    _finite(payload.get("lateral_speed_mps")) is not None and
                    isinstance(truth["com_v_mps"], float) else None),
                "r_error_radps": (
                    _finite(payload.get("yaw_rate_radps")) - float(truth["r_radps"])
                    if topic in ("/odom", "/ekf_odom") and
                    _finite(payload.get("yaw_rate_radps")) is not None and
                    isinstance(truth["r_radps"], float) else None),
            })

    metrics: dict[str, object] = {}
    for topic in ("/odom", "/ekf_odom", "/amcl_pose", "/current_map_pose"):
        rows = [row for row in report_rows if row["topic"] == topic]
        position_components = _position_error_truth_frame(rows)
        metrics[topic] = {
            # Euclidean distance remains for continuity, but is not the
            # primary trajectory-error score because it hides direction.
            "position_error_m": _metric([
                float(row["position_error_m"]) for row in rows]),
            "position_error_euclidean_m": _metric([
                float(row["position_error_m"]) for row in rows]),
            "position_error_components_truth_frame_m": {
                "frame": position_components["frame"],
                "offline_only": position_components["offline_only"],
                "longitudinal_abs_m": position_components[
                    "longitudinal_error_abs_m"],
                "longitudinal_signed_m": position_components[
                    "longitudinal_error_signed_m"],
                "lateral_abs_m": position_components["lateral_error_abs_m"],
                "lateral_signed_m": position_components[
                    "lateral_error_signed_m"],
            },
            # Backwards-compatible key for earlier analysis notebooks.
            "position_error_truth_frame": position_components,
            "covariance_xy_max_m2": _metric([
                float(row["covariance_xy_max_m2"]) for row in rows
                if isinstance(row["covariance_xy_max_m2"], float)]),
            "covariance_yaw_rad2": _metric([
                float(row["covariance_yaw_rad2"]) for row in rows
                if isinstance(row["covariance_yaw_rad2"], float)]),
            "yaw_error_rad": _metric([
                _angle_diff(float(row["estimate_yaw_rad"]),
                            float(row["truth_yaw_rad"]))
                for row in rows
                if isinstance(row["estimate_yaw_rad"], float) and
                isinstance(row["truth_yaw_rad"], float)]),
            "u_error_mps": _metric([
                float(row["u_error_mps"]) for row in rows
                if isinstance(row["u_error_mps"], float)]),
            "u_error_mps_com": _metric([
                float(row["u_error_mps_com"]) for row in rows
                if isinstance(row["u_error_mps_com"], float)]),
            "u_error_mps_com_signed": _metric([
                float(row["u_error_mps_com"]) for row in rows
                if isinstance(row["u_error_mps_com"], float)], absolute=False),
            "v_error_mps": _metric([
                float(row["v_error_mps"]) for row in rows
                if isinstance(row["v_error_mps"], float)]),
            "v_error_mps_com": _metric([
                float(row["v_error_mps_com"]) for row in rows
                if isinstance(row["v_error_mps_com"], float)]),
            "r_error_radps": _metric([
                float(row["r_error_radps"]) for row in rows
                if isinstance(row["r_error_radps"], float)]),
        }

    report: dict[str, object] = {
        "schema_version": 2,
        "run_dir": str(root),
        "offline_only": True,
        "future_ground_truth_used": False,
        "truth_velocity_contract": {
            "position_point": "simulator GPS vehicle-frame point",
            "pose_u_mps": "finite difference of the reported position at source-time yaw midpoint",
            "pose_v_mps": "finite difference of the reported position at source-time yaw midpoint",
            "com_u_mps": "raw simulator rigidbody/COM body-forward velocity, offline diagnostic only",
            "com_v_mps": "raw simulator rigidbody/COM lateral velocity, diagnostic only",
            "odom_u_metric": "pose-point and rigidbody-COM longitudinal velocity are scored separately",
            "odom_v_metric": "pose-point lateral velocity; COM comparison is reported separately",
        },
        "source_timing": {
            "packets": len(timing_rows),
            "dt_s": _metric(timing_dts, absolute=False),
            "sequence_gaps": sequence_gaps,
            "command_lag_packets": _metric(command_lags, absolute=False),
        },
        "delivery_timing": _delivery_timing_report(timing_rows, events),
        "metrics": metrics,
        "offline_motion_regime_metrics": _motion_regime_scores(report_rows),
        "acceptance_targets": {
            # Directional targets intentionally replace the former single
            # 5 cm Euclidean gate. They are localization diagnostics, not a
            # multiple-lap Pure Pursuit acceptance claim.
            "current_map_longitudinal_position_p95_m": 0.15,
            "current_map_lateral_position_p95_m": 0.05,
            "current_map_yaw_p95_rad": 0.02,
            "odom_u_p95_mps": 0.10,
            "odom_v_p95_mps": 0.10,
            "odom_r_p95_radps": 0.05,
        },
        "acceptance": {
            "transport_sequence_clean": sequence_gaps == 0,
            "localization_targets_available": bool(report_rows),
            "directional_pose_targets_met": (
                metrics.get("/current_map_pose", {}).get(
                    "position_error_components_truth_frame_m", {}).get(
                        "longitudinal_abs_m", {}).get("p95") is not None and
                metrics.get("/current_map_pose", {}).get(
                    "position_error_components_truth_frame_m", {}).get(
                        "lateral_abs_m", {}).get("p95") is not None and
                float(metrics["/current_map_pose"][
                    "position_error_components_truth_frame_m"][
                        "longitudinal_abs_m"]["p95"]) < 0.15 and
                float(metrics["/current_map_pose"][
                    "position_error_components_truth_frame_m"][
                        "lateral_abs_m"]["p95"]) < 0.05),
        },
    }

    if output_csv is not None:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fields = sorted({field for row in report_rows for field in row})
        with output_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(report_rows)
    if output_json is not None:
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    args = parser.parse_args(argv)
    report = build_report(args.run_dir, args.output_json, args.output_csv)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
