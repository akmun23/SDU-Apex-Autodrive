#!/usr/bin/env python3
"""Exact-epoch odometry drift decomposition for retained Unity practice bags.

This is an offline diagnostic. It never feeds Unity truth into production.
Both the Unity GPS and production odometry use the rear-axle reference point;
Unity's F1TENTH prefab attaches GPS to ``Rear Axle Center``. Unity rigid-body
linear velocity is at COM, so it is converted to the rear axle before pose
integration and velocity comparison.

Example (in the ROS 2 environment used by evaluate_mpc_bag.py)::

  python3 tools/analyze_mpc_odom_drift.py live_runs/<run>/run \
    --output-prefix live_runs/<run>/odom_drift

The report uses exact ROS header nanoseconds for GT/odom/plant joins. Headerless
lap counters are attached by nearest bag-receipt time and explicitly marked as
such. MPC/CSV progress is joined only by embedded exact source stamps.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

try:
    from tools import evaluate_mpc_bag as evaluator
except ModuleNotFoundError:
    import evaluate_mpc_bag as evaluator


SCHEMA = "sdu_apex_exact_epoch_odom_drift/v1"
COM_TO_REAR_AXLE_M = evaluator.COM_TO_REAR_AXLE_M


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def exact_epoch_pairs(
    truth: list[dict[str, Any]], estimate: list[dict[str, Any]],
) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], dict[str, int]]:
    """Join by exact integer ROS stamp; refuse duplicate stamps."""
    def index(rows: list[dict[str, Any]], label: str) -> dict[int, dict[str, Any]]:
        result: dict[int, dict[str, Any]] = {}
        for row in rows:
            stamp_ns = row.get("stamp_ns")
            if not isinstance(stamp_ns, int):
                raise ValueError(f"{label} row lacks exact integer stamp_ns")
            if stamp_ns in result:
                raise ValueError(f"duplicate exact timestamp in {label}: {stamp_ns}")
            result[stamp_ns] = row
        return result

    truth_by_stamp = index(truth, "Unity truth")
    estimate_by_stamp = index(estimate, "odometry")
    common = sorted(truth_by_stamp.keys() & estimate_by_stamp.keys())
    return ([(truth_by_stamp[t], estimate_by_stamp[t]) for t in common], {
        "truth_rows": len(truth),
        "odom_rows": len(estimate),
        "exact_matches": len(common),
        "truth_unmatched": len(truth_by_stamp) - len(common),
        "odom_unmatched": len(estimate_by_stamp) - len(common),
    })


def rotate_to_initial_heading(
    x: float, y: float, origin_x: float, origin_y: float, yaw0: float,
) -> tuple[float, float]:
    dx, dy = x - origin_x, y - origin_y
    c, s = math.cos(yaw0), math.sin(yaw0)
    return c * dx + s * dy, -s * dx + c * dy


def _rear_v(row: dict[str, Any], source: str) -> float:
    if source == "gt":
        return float(row["v"]) - float(row["r"]) * COM_TO_REAR_AXLE_M
    if source == "odom":
        return float(row["v"])
    if source == "plant":
        return float(row["v_rear"])
    raise ValueError(f"unknown twist source {source!r}")


def integrate_twist(
    rows: list[dict[str, Any]], source: str,
    yaw_rows: list[dict[str, Any]] | None = None,
    substep_s: float | None = None,
) -> list[tuple[float, float]]:
    """Integrate rear-axle body twist in each stream's initial-yaw frame.

    ``substep_s=None`` reproduces the production midpoint pose integration.
    A finite substep linearly interpolates the same measured endpoint series;
    it is a numerical-method comparison, not additional simulator data.
    """
    if not rows:
        return []
    headings = yaw_rows if yaw_rows is not None else rows
    if len(headings) != len(rows):
        raise ValueError("twist and heading series must have equal length")
    yaw0 = float(headings[0]["yaw"])
    path = [(0.0, 0.0)]
    x = y = 0.0
    for i, (before, after, yaw_before_row, yaw_after_row) in enumerate(
            zip(rows, rows[1:], headings, headings[1:]), start=1):
        dt = (int(after["stamp_ns"]) - int(before["stamp_ns"])) * 1.0e-9
        if not (0.0 < dt <= 0.100):
            path.append((math.nan, math.nan))
            continue
        yaw0_rel = wrap(float(yaw_before_row["yaw"]) - yaw0)
        yaw1_rel = yaw0_rel + wrap(
            float(yaw_after_row["yaw"]) - float(yaw_before_row["yaw"]))
        u0, u1 = float(before["u"]), float(after["u"])
        v0, v1 = _rear_v(before, source), _rear_v(after, source)
        if substep_s is None:
            um = 0.5 * (u0 + u1)
            vm = 0.5 * (v0 + v1)
            ym = yaw0_rel + 0.5 * (yaw1_rel - yaw0_rel)
            x += dt * (um * math.cos(ym) - vm * math.sin(ym))
            y += dt * (um * math.sin(ym) + vm * math.cos(ym))
        else:
            count = max(1, math.ceil(dt / substep_s))
            ds = dt / count
            for step in range(count):
                f = (step + 0.5) / count
                u = u0 + f * (u1 - u0)
                v = v0 + f * (v1 - v0)
                yaw = yaw0_rel + f * (yaw1_rel - yaw0_rel)
                x += ds * (u * math.cos(yaw) - v * math.sin(yaw))
                y += ds * (u * math.sin(yaw) + v * math.cos(yaw))
        path.append((x, y))
    return path


def interval_displacement(
    before: dict[str, Any], after: dict[str, Any], source: str,
    yaw_reference_rad: float,
) -> tuple[float, float] | None:
    """Production midpoint displacement for one observed interval."""
    dt = (int(after["stamp_ns"]) - int(before["stamp_ns"])) * 1.0e-9
    if not (0.0 < dt <= 0.100):
        return None
    yaw0 = float(before["yaw"])
    yaw_mid = wrap(yaw0 - yaw_reference_rad +
                   0.5 * wrap(float(after["yaw"]) - yaw0))
    u_mid = 0.5 * (float(before["u"]) + float(after["u"]))
    v_mid = 0.5 * (_rear_v(before, source) + _rear_v(after, source))
    return (dt * (u_mid * math.cos(yaw_mid) - v_mid * math.sin(yaw_mid)),
            dt * (u_mid * math.sin(yaw_mid) + v_mid * math.cos(yaw_mid)))


def integrate_gyro_heading(
    imu_rows: list[dict[str, Any]], initial_yaw: float,
) -> list[float]:
    """Trapezoid-integrate the recorded gyro z rate over exact IMU epochs."""
    if not imu_rows:
        return []
    yaw = float(initial_yaw)
    output = [yaw]
    for before, after in zip(imu_rows, imu_rows[1:]):
        dt = (int(after["stamp_ns"]) - int(before["stamp_ns"])) * 1.0e-9
        if dt <= 0.0 or dt > 0.100:
            output.append(math.nan)
            continue
        yaw += 0.5 * (float(before["yaw_rate"]) +
                      float(after["yaw_rate"])) * dt
        output.append(yaw)
    return output


def _pose_path(rows: list[dict[str, Any]]) -> list[tuple[float, float]]:
    if not rows:
        return []
    first = rows[0]
    return [rotate_to_initial_heading(
        float(row["x"]), float(row["y"]), float(first["x"]),
        float(first["y"]), float(first["yaw"])) for row in rows]


def _plant_pose_path(rows: list[dict[str, Any]], yaw0: float) -> list[tuple[float, float]]:
    if not rows:
        return []
    x0 = float(rows[0]["physx_model_rear_axle_x_m"])
    y0 = float(rows[0]["physx_model_rear_axle_y_m"])
    return [rotate_to_initial_heading(
        float(row["physx_model_rear_axle_x_m"],),
        float(row["physx_model_rear_axle_y_m"]), x0, y0, yaw0)
        for row in rows]


def _path_errors(
    predicted: list[tuple[float, float]], actual: list[tuple[float, float]],
) -> list[dict[str, float]]:
    errors: list[dict[str, float]] = []
    for (px, py), (ax, ay) in zip(predicted, actual):
        if not all(math.isfinite(v) for v in (px, py, ax, ay)):
            errors.append({"forward_m": math.nan, "lateral_m": math.nan,
                           "xy_m": math.nan})
            continue
        dx, dy = px - ax, py - ay
        errors.append({"forward_m": dx, "lateral_m": dy,
                       "xy_m": math.hypot(dx, dy)})
    return errors


def _metric(values: list[float]) -> dict[str, float | int | None]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"n": 0, "bias": None, "rmse": None, "p95_abs": None,
                "max_abs": None}
    absolute = sorted(abs(value) for value in finite)
    p95_index = min(len(absolute) - 1, math.ceil(0.95 * len(absolute)) - 1)
    return {
        "n": len(finite),
        "bias": statistics.mean(finite),
        "rmse": math.sqrt(statistics.mean(value * value for value in finite)),
        "p95_abs": absolute[p95_index],
        "max_abs": max(absolute),
    }


def _interp(
    rows: list[dict[str, Any]], t_ns: int, stamps: list[int] | None = None,
) -> dict[str, float] | None:
    if stamps is None:
        stamps = [int(row["stamp_ns"]) for row in rows]
    i = bisect.bisect_left(stamps, t_ns)
    if i < len(rows) and stamps[i] == t_ns:
        row = rows[i]
        return {key: float(row[key]) for key in ("u", "v", "r", "yaw")}
    if i == 0 or i >= len(rows):
        return None
    a, b = rows[i - 1], rows[i]
    span = int(b["stamp_ns"]) - int(a["stamp_ns"])
    if span <= 0 or span > 100_000_000:
        return None
    f = (t_ns - int(a["stamp_ns"])) / span
    dyaw = wrap(float(b["yaw"]) - float(a["yaw"]))
    return {
        "u": float(a["u"]) + f * (float(b["u"]) - float(a["u"])),
        "v": float(a["v"]) + f * (float(b["v"]) - float(a["v"])),
        "r": float(a["r"]) + f * (float(b["r"]) - float(a["r"])),
        "yaw": float(a["yaw"]) + f * dyaw,
    }


def phase_sweep(
    rows: list[dict[str, Any]], source: str, offsets_ms: list[float],
) -> list[dict[str, float | int]]:
    """Post-hoc interval displacement phase diagnostic; never applies a shift."""
    result = []
    yaw0 = float(rows[0]["yaw"])
    stamps = [int(row["stamp_ns"]) for row in rows]
    for offset_ms in offsets_ms:
        residuals: list[float] = []
        residual_stamps: list[int] = []
        for a, b in zip(rows, rows[1:]):
            t0, t1 = int(a["stamp_ns"]), int(b["stamp_ns"])
            dt = (t1 - t0) * 1.0e-9
            if not (0.015 <= dt <= 0.040):
                continue
            off = round(offset_ms * 1.0e6)
            sa = _interp(rows, t0 + off, stamps)
            sb = _interp(rows, t1 + off, stamps)
            if sa is None or sb is None:
                continue
            um, vm = 0.5 * (sa["u"] + sb["u"]), 0.5 * (
                (sa["v"] - sa["r"] * COM_TO_REAR_AXLE_M if source == "gt" else sa["v"]) +
                (sb["v"] - sb["r"] * COM_TO_REAR_AXLE_M if source == "gt" else sb["v"]))
            yaw_delta = wrap(sb["yaw"] - sa["yaw"])
            yaw_mid = wrap(sa["yaw"] - yaw0) + 0.5 * yaw_delta
            dx = dt * (um * math.cos(yaw_mid) - vm * math.sin(yaw_mid))
            dy = dt * (um * math.sin(yaw_mid) + vm * math.cos(yaw_mid))
            actual_dx, actual_dy = (
                float(b["x"]) - float(a["x"]),
                float(b["y"]) - float(a["y"]))
            c, s = math.cos(yaw0), math.sin(yaw0)
            actual_local_x = c * actual_dx + s * actual_dy
            actual_local_y = -s * actual_dx + c * actual_dy
            residuals.append(math.hypot(dx - actual_local_x, dy - actual_local_y))
            residual_stamps.append(t1)
        worst_index = max(range(len(residuals)), key=residuals.__getitem__) \
            if residuals else None
        result.append({
            "offset_ms": offset_ms,
            "n": len(residuals),
            "rmse_m": math.sqrt(statistics.mean(v * v for v in residuals))
            if residuals else math.nan,
            "p95_m": sorted(residuals)[min(len(residuals) - 1,
                         math.ceil(0.95 * len(residuals)) - 1)]
            if residuals else math.nan,
            "max_m": max(residuals) if residuals else math.nan,
            "worst_interval_end_stamp_ns": (
                residual_stamps[worst_index] if worst_index is not None else -1),
        })
    return result


def _nearest_receipt(
    rows: list[dict[str, Any]], times: list[float], target: float,
    tolerance_s: float,
) -> dict[str, Any] | None:
    if not rows:
        return None
    i = bisect.bisect_left(times, target)
    candidates = [j for j in (i - 1, i) if 0 <= j < len(rows)]
    j = min(candidates, key=lambda index: abs(times[index] - target))
    return rows[j] if abs(times[j] - target) <= tolerance_s else None


def _embedded_progress(
    data: dict[str, list[dict[str, Any]]],
) -> dict[int, tuple[float, str]]:
    progress: dict[int, tuple[float, str]] = {}
    for topic, fields in ((evaluator.DIAGNOSTICS, ("progress_m",)),
                          (evaluator.CANDIDATE_DIAGNOSTICS, ("progress_m",)),
                          (evaluator.CSV_REPLAY,
                           ("csv_progress_m",))):
        for row in data.get(topic, []):
            payload = row.get("json") or {}
            stamp_candidates = [
                payload.get("gt_stamp_ns"), payload.get("source_stamp_ns"),
                payload.get("odom_source_stamp_ns"),
                (payload.get("control_time_prediction") or {}).get(
                    "plant_source_stamp_ns"),
            ]
            stamp_ns = next((int(value) for value in stamp_candidates
                             if isinstance(value, int)), None)
            if stamp_ns is None:
                continue
            value = next((payload.get(field) for field in fields
                          if isinstance(payload.get(field), (int, float)) and
                          math.isfinite(float(payload[field]))), None)
            if value is not None:
                progress[stamp_ns] = (float(value), topic)
    return progress


def _delivery_diagnostics(
    data: dict[str, list[dict[str, Any]]],
    matched_truth: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    """Summarize host-observed data age without inventing a Unity step ID."""
    bridge_by_request: dict[int, dict[str, Any]] = {}
    bridge_delays_ms: list[float] = []
    for row in data.get(evaluator.BRIDGE_TIMING, []):
        payload = row.get("json") or {}
        request_ns = payload.get("bridge_request_ros_stamp_ns")
        receive_ns = payload.get("bridge_receive_ros_stamp_ns")
        if not isinstance(request_ns, int) or not isinstance(receive_ns, int):
            continue
        bridge_by_request[request_ns] = payload
        bridge_delays_ms.append((receive_ns - request_ns) * 1.0e-6)

    header_to_receipt_ms = [
        (float(row["bag_time"]) - int(row["stamp_ns"]) * 1.0e-9) * 1e3
        for row in matched_truth]
    replay_rows = sorted(data.get(evaluator.CSV_REPLAY, []),
                         key=lambda row: row["bag_time"])
    replay_events: list[dict[str, Any]] = []
    ages = []
    for row in replay_rows:
        payload = row.get("json") or {}
        age = float(payload.get("gt_age_ms", math.nan))
        stamp_ns = payload.get("gt_stamp_ns")
        if math.isfinite(age):
            ages.append(age)
        if isinstance(stamp_ns, int):
            replay_events.append({
                "bag_time_s": row["bag_time"], "gt_stamp_ns": stamp_ns,
                "gt_age_ms": age,
                "commands_sent": payload.get("commands_sent"),
                "x_m": payload.get("ground_truth_x_m"),
                "y_m": payload.get("ground_truth_y_m"),
            })
    repeat_counts: dict[int, int] = {}
    repeated_age: dict[int, float] = {}
    longest_repeat_run = 0
    previous_stamp = None
    current_repeat_run = 0
    for event in replay_events:
        stamp_ns = event["gt_stamp_ns"]
        repeat_counts[stamp_ns] = repeat_counts.get(stamp_ns, 0) + 1
        repeated_age[stamp_ns] = max(
            repeated_age.get(stamp_ns, -math.inf), event["gt_age_ms"])
        if stamp_ns == previous_stamp:
            current_repeat_run += 1
        else:
            current_repeat_run = 1
            previous_stamp = stamp_ns
        longest_repeat_run = max(longest_repeat_run, current_repeat_run)

    def distribution(values: list[float]) -> dict[str, Any]:
        if not values:
            return {"count": 0, "p50_ms": None, "p95_ms": None,
                    "max_ms": None, "over_40ms": 0, "over_75ms": 0}
        return {
            "count": len(values),
            "p50_ms": evaluator.percentile(values, 0.50),
            "p95_ms": evaluator.percentile(values, 0.95),
            "max_ms": max(values),
            "over_40ms": sum(value > 40.0 for value in values),
            "over_75ms": sum(value > 75.0 for value in values),
        }

    detail = {
        "gt_header_to_bag_receipt_ms": distribution(header_to_receipt_ms),
        "bridge_request_to_response_ms": distribution(bridge_delays_ms),
        "csv_follower_reported_gt_age_ms": distribution(ages),
        "csv_replay_messages": len(replay_rows),
        "csv_replay_unique_gt_stamps": len(repeat_counts),
        "csv_replay_repeated_gt_stamp_messages":
            sum(max(0, count - 1) for count in repeat_counts.values()),
        "csv_replay_longest_consecutive_same_gt_stamp_count": longest_repeat_run,
        "csv_replay_stale_events_over_40ms": [
            event for event in replay_events
            if math.isfinite(event["gt_age_ms"]) and event["gt_age_ms"] > 40.0
        ][:100],
        "csv_replay_repeated_stamp_chains": [
            {"gt_stamp_ns": stamp_ns, "message_count": count,
             "max_reported_age_ms": repeated_age[stamp_ns]}
            for stamp_ns, count in sorted(
                repeat_counts.items(), key=lambda item: item[1], reverse=True)
            if count > 1
        ][:100],
        "bridge_requests_with_development_unity_step_id": sum(
            payload.get("simulation_physics_step") is not None
            for payload in bridge_by_request.values()),
        "bridge_requests_with_telemetry_sequence": sum(
            payload.get("telemetry_sequence") is not None
            for payload in bridge_by_request.values()),
        "interpretation": (
            "Header-to-receipt and bridge request/response delays are host-side "
            "delivery measurements. They do not identify the Unity FixedUpdate "
            "or WheelCollider physics epoch that produced a telemetry payload."
        ),
    }
    return detail, bridge_by_request


def _collision_cutoff(
    data: dict[str, list[dict[str, Any]]],
) -> tuple[float | None, str | None]:
    events = [(row["bag_time"], "collision_count")
              for row in data.get(evaluator.COLLISION, [])
              if float(row.get("value", 0.0)) > 0.0]
    for row in data.get(evaluator.BRIDGE_FAULT, []):
        value = row.get("value", 0.0)
        try:
            active = float(value) != 0.0
        except (TypeError, ValueError):
            active = bool(str(value).strip()) and str(value).strip().lower() not in (
                "0", "false", "none", "null", "ok")
        if active:
            events.append((row["bag_time"], "bridge_fault"))
    return min(events) if events else (None, None)


def analyze(
    bag: Path, trajectory: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    data = evaluator.read_bag(bag)
    truth_all = data.get(evaluator.GT, [])
    odom_all = data.get(evaluator.ODOM, [])
    plant_all = data.get(evaluator.PLANT, [])
    if not truth_all or not odom_all:
        raise ValueError("bag must contain Unity truth and production /odom")

    pairs, join = exact_epoch_pairs(truth_all, odom_all)
    if len(pairs) < 3:
        raise ValueError(f"only {len(pairs)} exact GT/odom epoch pairs")

    # Exclude the first interval whose contact/supervisor fault may be
    # contaminated. Collision count is headerless, so the preceding packet
    # interval is also marked indeterminate rather than called clean.
    cutoff_receipt_s, cutoff_reason = _collision_cutoff(data)
    if cutoff_receipt_s is not None:
        selected = [(gt, od) for gt, od in pairs
                    if gt["bag_time"] < cutoff_receipt_s - 0.025]
    else:
        selected = pairs
    if len(selected) < 3:
        raise ValueError("fewer than 3 pairs remain before conservative cutoff")
    truth = [pair[0] for pair in selected]
    odom = [pair[1] for pair in selected]
    stamps = [int(row["stamp_ns"]) for row in truth]
    if any(b <= a for a, b in zip(stamps, stamps[1:])):
        raise ValueError("exact matched epochs are not strictly increasing")

    # Only retain source plant states at exactly the same matched epochs.
    plant_by_stamp: dict[int, dict[str, Any]] = {}
    plant_duplicate_count = 0
    for row in plant_all:
        t = row.get("stamp_ns")
        if isinstance(t, int):
            if t in plant_by_stamp:
                plant_duplicate_count += 1
            else:
                plant_by_stamp[t] = row
    plant_rows = [plant_by_stamp.get(t) for t in stamps]
    source_pose_present = all(
        row is not None and "physx_model_rear_axle_x_m" in row
        for row in plant_rows)

    gt_pose = _pose_path(truth)
    odom_pose = _pose_path(odom)
    gt_mid = integrate_twist(truth, "gt")
    gt_fine = integrate_twist(truth, "gt", substep_s=0.001)
    odom_mid = integrate_twist(odom, "odom")
    odom_fine = integrate_twist(odom, "odom", substep_s=0.001)
    odom_with_gt_yaw = integrate_twist(odom, "odom", yaw_rows=truth)
    gt_with_odom_yaw = integrate_twist(truth, "gt", yaw_rows=odom)
    odom_gt_u_rows = [dict(row, u=gt["u"])
                      for row, gt in zip(odom, truth)]
    odom_gt_v_rows = [dict(row, v=_rear_v(gt, "gt"))
                      for row, gt in zip(odom, truth)]
    odom_gt_uv_rows = [dict(row, u=gt["u"], v=_rear_v(gt, "gt"))
                       for row, gt in zip(odom, truth)]
    odom_gt_u_mid = integrate_twist(odom_gt_u_rows, "odom")
    odom_gt_v_mid = integrate_twist(odom_gt_v_rows, "odom")
    odom_gt_uv_mid = integrate_twist(odom_gt_uv_rows, "odom")
    gt_com_as_rear_rows = [dict(row, v=row["v"]) for row in truth]
    gt_rear_plus_offset_rows = [
        dict(row, v=float(row["v"]) + float(row["r"]) * COM_TO_REAR_AXLE_M)
        for row in truth]
    gt_com_as_rear_mid = integrate_twist(
        gt_com_as_rear_rows, "odom", yaw_rows=truth)
    gt_rear_plus_offset_mid = integrate_twist(
        gt_rear_plus_offset_rows, "odom", yaw_rows=truth)
    errors = {
        "recorded_odom_pose_vs_GT": _path_errors(odom_pose, gt_pose),
        "GT_twist_integral_midpoint_vs_GT_pose": _path_errors(gt_mid, gt_pose),
        "GT_twist_integral_1ms_vs_GT_pose": _path_errors(gt_fine, gt_pose),
        "odom_twist_integral_midpoint_vs_GT_pose": _path_errors(odom_mid, gt_pose),
        "odom_twist_integral_1ms_vs_GT_pose": _path_errors(odom_fine, gt_pose),
        "odom_twist_GT_heading_ablation_vs_GT_pose": _path_errors(
            odom_with_gt_yaw, gt_pose),
        "odom_twist_GT_u_ablation_vs_GT_pose": _path_errors(
            odom_gt_u_mid, gt_pose),
        "odom_twist_GT_vrear_ablation_vs_GT_pose": _path_errors(
            odom_gt_v_mid, gt_pose),
        "odom_twist_GT_u_and_vrear_ablation_vs_GT_pose": _path_errors(
            odom_gt_uv_mid, gt_pose),
        "GT_twist_COM_v_mistaken_as_rear_v_vs_GT_pose": _path_errors(
            gt_com_as_rear_mid, gt_pose),
        "GT_twist_opposite_rear_v_offset_sign_vs_GT_pose": _path_errors(
            gt_rear_plus_offset_mid, gt_pose),
        "GT_twist_odom_heading_ablation_vs_GT_pose": _path_errors(
            gt_with_odom_yaw, gt_pose),
    }
    if source_pose_present:
        source_pose = _plant_pose_path(
            [row for row in plant_rows if row is not None],
            float(truth[0]["yaw"]))
        errors["source_model_rear_axle_pose_vs_GT"] = _path_errors(
            source_pose, gt_pose)
        plant_aligned = [row for row in plant_rows if row is not None]
        plant_yaw_integral = integrate_twist(
            plant_aligned, "plant", yaw_rows=odom)
        errors["source_plant_twist_integral_using_odom_yaw_vs_GT_pose"] = (
            _path_errors(plant_yaw_integral, gt_pose))

    def interval_closure(
        twist_rows: list[dict[str, Any]], twist_source: str,
        pose_rows: list[dict[str, Any]],
    ) -> list[dict[str, float]]:
        """Compare one midpoint twist step to its recorded pose increment."""
        twist_yaw0 = float(twist_rows[0]["yaw"])
        pose_yaw0 = float(pose_rows[0]["yaw"])
        result = [{"forward_m": math.nan, "lateral_m": math.nan,
                   "xy_m": math.nan}]
        for i in range(1, len(twist_rows)):
            predicted = interval_displacement(
                twist_rows[i - 1], twist_rows[i], twist_source, twist_yaw0)
            if predicted is None:
                result.append({"forward_m": math.nan, "lateral_m": math.nan,
                              "xy_m": math.nan})
                continue
            pa, pb = pose_rows[i - 1], pose_rows[i]
            dx, dy = float(pb["x"]) - float(pa["x"]), \
                float(pb["y"]) - float(pa["y"])
            c, s = math.cos(pose_yaw0), math.sin(pose_yaw0)
            actual = (c * dx + s * dy, -s * dx + c * dy)
            ex, ey = predicted[0] - actual[0], predicted[1] - actual[1]
            result.append({"forward_m": ex, "lateral_m": ey,
                           "xy_m": math.hypot(ex, ey)})
        return result

    interval_closures = {
        "GT_twist_to_GT_pose_increment": interval_closure(truth, "gt", truth),
        "odom_twist_to_odom_pose_increment": interval_closure(odom, "odom", odom),
        "odom_twist_to_GT_pose_increment": interval_closure(odom, "odom", truth),
    }

    gt_yaw_rel: list[float] = [0.0]
    odom_yaw_rel: list[float] = [0.0]
    gt_yaw_unwrapped = 0.0
    odom_yaw_unwrapped = 0.0
    for a, b in zip(truth, truth[1:]):
        gt_yaw_unwrapped += wrap(float(b["yaw"]) - float(a["yaw"]))
        gt_yaw_rel.append(gt_yaw_unwrapped)
    for a, b in zip(odom, odom[1:]):
        odom_yaw_unwrapped += wrap(float(b["yaw"]) - float(a["yaw"]))
        odom_yaw_rel.append(odom_yaw_unwrapped)
    yaw_error = [o - g for o, g in zip(odom_yaw_rel, gt_yaw_rel)]
    imu_by_stamp: dict[int, dict[str, Any]] = {}
    for row in data.get(evaluator.IMU, []):
        t = row.get("stamp_ns")
        if isinstance(t, int) and t not in imu_by_stamp:
            imu_by_stamp[t] = row
    imu_rows = [imu_by_stamp.get(t) for t in stamps]
    imu_exact_coverage = sum(row is not None for row in imu_rows)
    imu_orientation_coverage = sum(
        row is not None and math.isfinite(float(row.get("orientation_yaw", math.nan)))
        for row in imu_rows)
    imu_complete = all(row is not None for row in imu_rows)
    imu_orientation_complete = imu_complete and all(
        math.isfinite(float(row.get("orientation_yaw", math.nan)))
        for row in imu_rows if row is not None)
    gyro_yaw_values: list[float] = []
    imu_quaternion_rows: list[dict[str, Any]] = []
    imu_gyro_rows: list[dict[str, Any]] = []
    imu_yaw_errors: dict[str, list[float]] = {}
    if imu_complete:
        imu_gyro_rows = [row for row in imu_rows if row is not None]
        gyro_yaw_values = integrate_gyro_heading(
            imu_gyro_rows, float(odom[0]["yaw"]))
        if imu_orientation_complete:
            imu_quaternion_rows = [
                {"stamp_ns": row["stamp_ns"], "yaw": row["orientation_yaw"]}
                for row in imu_rows if row is not None]
            errors["odom_twist_IMU_quaternion_heading_ablation_vs_GT_pose"] = (
                _path_errors(integrate_twist(
                    odom, "odom", yaw_rows=imu_quaternion_rows), gt_pose))
            gyro_heading_rows = [
                {"stamp_ns": stamps[i], "yaw": gyro_yaw_values[i]}
                for i in range(len(stamps))]
            errors["odom_twist_integrated_gyro_heading_ablation_vs_GT_pose"] = (
                _path_errors(integrate_twist(
                    odom, "odom", yaw_rows=gyro_heading_rows), gt_pose))
            imu_quaternion_unwrapped = [0.0]
            for a, b in zip(imu_rows, imu_rows[1:]):
                imu_quaternion_unwrapped.append(
                    imu_quaternion_unwrapped[-1] + wrap(
                        float(b["orientation_yaw"]) -
                        float(a["orientation_yaw"])))
            imu_yaw_errors = {
                "IMU_quaternion_relative_minus_GT_relative_rad": [
                    q - g for q, g in zip(imu_quaternion_unwrapped, gt_yaw_rel)],
                "integrated_gyro_relative_minus_GT_relative_rad": [
                    (g - gyro_yaw_values[0]) - truth_yaw
                    for g, truth_yaw in zip(gyro_yaw_values, gt_yaw_rel)],
                "IMU_quaternion_relative_minus_odom_relative_rad": [
                    q - o for q, o in zip(imu_quaternion_unwrapped, odom_yaw_rel)],
            }
    velocity_errors = {
        "u_mps": [float(o["u"]) - float(g["u"]) for g, o in selected],
        "v_rear_mps": [float(o["v"]) - _rear_v(g, "gt")
                       for g, o in selected],
        "yaw_rate_radps": [float(o["r"]) - float(g["r"])
                           for g, o in selected],
        "relative_yaw_rad": yaw_error,
    }

    # Headerless lap counters are associated using recorder receipt time only;
    # exact epoch joins remain the sole basis for pose/twist scores.
    lap_rows = data.get(evaluator.LAP, [])
    lap_times = [float(row["bag_time"]) for row in lap_rows]
    progress_map = _embedded_progress(data)
    delivery_summary, bridge_by_request = _delivery_diagnostics(data, truth)
    row_errors = errors["recorded_odom_pose_vs_GT"]
    sample_rows: list[dict[str, Any]] = []
    for i, (gt, od, err) in enumerate(zip(truth, odom, row_errors)):
        dt_s = ((stamps[i] - stamps[i - 1]) * 1e-9) if i else math.nan
        lap = _nearest_receipt(lap_rows, lap_times, gt["bag_time"], 0.020)
        progress = progress_map.get(stamps[i])
        dx, dy = err["forward_m"], err["lateral_m"]
        yaw_rel = gt_yaw_rel[i]
        c, s = math.cos(yaw_rel), math.sin(yaw_rel)
        progress_value = progress[0] if progress else math.nan
        sample: dict[str, Any] = {
            "stamp_ns": stamps[i],
            "elapsed_s": (stamps[i] - stamps[0]) * 1e-9,
            "dt_s": dt_s,
            "gt_bag_time_s": gt["bag_time"],
            "odom_bag_time_s": od["bag_time"],
            "gt_header_to_bag_receipt_ms":
                (float(gt["bag_time"]) - stamps[i] * 1e-9) * 1e3,
            "gt_frame": gt.get("frame", ""), "gt_child": gt.get("child", ""),
            "odom_frame": od.get("frame", ""), "odom_child": od.get("child", ""),
            "gt_x_m": gt["x"], "gt_y_m": gt["y"], "gt_yaw_rad": gt["yaw"],
            "odom_x_m": od["x"], "odom_y_m": od["y"],
            "odom_yaw_rad": od["yaw"],
            "gt_u_mps": gt["u"], "gt_v_com_mps": gt["v"],
            "gt_v_rear_mps": _rear_v(gt, "gt"), "gt_yaw_rate_radps": gt["r"],
            "odom_u_mps": od["u"], "odom_v_rear_mps": od["v"],
            "odom_yaw_rate_radps": od["r"],
            "u_error_mps": velocity_errors["u_mps"][i],
            "v_rear_error_mps": velocity_errors["v_rear_mps"][i],
            "yaw_rate_error_radps": velocity_errors["yaw_rate_radps"][i],
            "relative_yaw_error_rad": yaw_error[i],
            "imu_gyro_z_radps": (
                float(imu_rows[i]["yaw_rate"])
                if imu_rows[i] is not None else math.nan),
            "imu_quaternion_yaw_rad": (
                float(imu_rows[i]["orientation_yaw"])
                if imu_rows[i] is not None else math.nan),
            "integrated_gyro_yaw_rad": (
                gyro_yaw_values[i] if gyro_yaw_values else math.nan),
            "pose_forward_error_m": dx, "pose_lateral_error_m": dy,
            "pose_error_xy_m": err["xy_m"],
            "pose_error_track_forward_m": c * dx + s * dy,
            "pose_error_track_lateral_m": -s * dx + c * dy,
            "lap_count_receipt_join": lap.get("value", math.nan) if lap else math.nan,
            "lap_count_receipt_offset_ms":
                (lap["bag_time"] - gt["bag_time"]) * 1e3 if lap else math.nan,
            "progress_m": progress_value,
            "progress_source": progress[1] if progress else "",
        }
        bridge = bridge_by_request.get(stamps[i], {})
        bridge_request_ns = bridge.get("bridge_request_ros_stamp_ns")
        bridge_receive_ns = bridge.get("bridge_receive_ros_stamp_ns")
        sample["bridge_request_sequence"] = bridge.get("request_sequence", math.nan)
        sample["bridge_packet_sequence"] = bridge.get("packet_sequence", math.nan)
        sample["bridge_request_to_response_ms"] = (
            (bridge_receive_ns - bridge_request_ns) * 1e-6
            if isinstance(bridge_request_ns, int) and
            isinstance(bridge_receive_ns, int) else math.nan)
        sample["bridge_unity_physics_step"] = bridge.get(
            "simulation_physics_step", math.nan)
        for name, closure_rows in interval_closures.items():
            sample[name + "_forward_m"] = closure_rows[i]["forward_m"]
            sample[name + "_lateral_m"] = closure_rows[i]["lateral_m"]
            sample[name + "_xy_m"] = closure_rows[i]["xy_m"]
        for key, values in errors.items():
            sample[key + "_x_m"] = (
                values[i]["forward_m"] if math.isfinite(values[i]["forward_m"])
                else math.nan)
            sample[key + "_y_m"] = (
                values[i]["lateral_m"] if math.isfinite(values[i]["lateral_m"])
                else math.nan)
            sample[key + "_xy_m"] = values[i]["xy_m"]
        if source_pose_present:
            sample["source_plant_u_mps"] = plant_rows[i]["u"]
            sample["source_plant_v_rear_mps"] = plant_rows[i]["v_rear"]
            sample["source_plant_yaw_rate_radps"] = plant_rows[i]["r"]
        if source_pose_present:
            sample["source_model_x_m"] = plant_rows[i][
                "physx_model_rear_axle_x_m"]
            sample["source_model_y_m"] = plant_rows[i][
                "physx_model_rear_axle_y_m"]
        sample_rows.append(sample)

    error_summary = {
        name: {
            "forward_m": _metric([row["forward_m"] for row in values]),
            "lateral_m": _metric([row["lateral_m"] for row in values]),
            "xy_m": _metric([row["xy_m"] for row in values]),
            "endpoint_xy_m": values[-1]["xy_m"],
        }
        for name, values in errors.items()
    }
    velocity_summary = {name: _metric(values)
                        for name, values in velocity_errors.items()}
    baseline_endpoint = error_summary[
        "odom_twist_integral_midpoint_vs_GT_pose"]["endpoint_xy_m"]
    numerical_fine_endpoint = error_summary[
        "odom_twist_integral_1ms_vs_GT_pose"]["endpoint_xy_m"]
    causal_budget = {
        "not_additive_endpoint_attribution_m": {
            "baseline_odom_twist": baseline_endpoint,
            "replace_odom_yaw_with_GT_yaw": error_summary[
                "odom_twist_GT_heading_ablation_vs_GT_pose"]["endpoint_xy_m"],
            "replace_odom_u_with_GT_u": error_summary[
                "odom_twist_GT_u_ablation_vs_GT_pose"]["endpoint_xy_m"],
            "replace_odom_vrear_with_GT_vrear": error_summary[
                "odom_twist_GT_vrear_ablation_vs_GT_pose"]["endpoint_xy_m"],
            "replace_odom_u_and_vrear_with_GT": error_summary[
                "odom_twist_GT_u_and_vrear_ablation_vs_GT_pose"]["endpoint_xy_m"],
            "replace_both_velocity_components_and_use_odom_yaw": error_summary[
                "GT_twist_odom_heading_ablation_vs_GT_pose"]["endpoint_xy_m"],
            "use_interpolated_1ms_integration": numerical_fine_endpoint,
            "midpoint_to_1ms_endpoint_change_m":
                numerical_fine_endpoint - baseline_endpoint,
            "relative_yaw_error_at_end_rad": yaw_error[-1],
        },
        "interpretation": (
            "Each row changes one measured signal family in an offline replay. "
            "Endpoint error changes are conditional and non-additive."
        ),
    }

    # Per-run progress bins use the exact embedded control/CSV stamp when it
    # exists. Lap count itself is receipt-time associated and is identified as
    # such; this distinction prevents it from changing the exact-match score.
    lap_groups: dict[str, list[int]] = {}
    progress_groups: dict[str, list[int]] = {}
    for i, row in enumerate(sample_rows):
        lap_key = row["lap_count_receipt_join"]
        if isinstance(lap_key, (int, float)) and math.isfinite(float(lap_key)):
            lap_groups.setdefault(str(int(lap_key)), []).append(i)
        if math.isfinite(float(row["progress_m"])):
            progress_groups.setdefault(
                f"{math.floor(float(row['progress_m']) / 5.0) * 5:04d}-"
                f"{math.floor(float(row['progress_m']) / 5.0) * 5 + 5:04d}m", []
            ).append(i)

    def summarize_groups(groups: dict[str, list[int]]) -> dict[str, Any]:
        output = {}
        for name, indices in groups.items():
            pose = [row_errors[i] for i in indices]
            final = pose[-1]
            group_summary: dict[str, Any] = {
                "samples": len(indices),
                "pose_xy_rmse_m": _metric([value["xy_m"] for value in pose])["rmse"],
                "pose_xy_p95_m": _metric([value["xy_m"] for value in pose])["p95_abs"],
                "pose_xy_max_m": max(value["xy_m"] for value in pose),
                "end_forward_error_m": final["forward_m"],
                "end_lateral_error_m": final["lateral_m"],
                "end_xy_error_m": final["xy_m"],
                "yaw_error_end_rad": yaw_error[indices[-1]],
                "u_error_bias_mps": _metric([velocity_errors["u_mps"][i]
                                               for i in indices])["bias"],
                "v_rear_error_bias_mps": _metric([
                    velocity_errors["v_rear_mps"][i] for i in indices])["bias"],
            }
            for method, method_errors in errors.items():
                magnitudes = [method_errors[i]["xy_m"] for i in indices
                              if math.isfinite(method_errors[i]["xy_m"])]
                if magnitudes:
                    group_summary[method] = {
                        "path_xy_rmse_m": _metric(magnitudes)["rmse"],
                        "path_xy_p95_m": _metric(magnitudes)["p95_abs"],
                        "path_xy_max_m": max(magnitudes),
                        "end_xy_m": method_errors[indices[-1]]["xy_m"],
                    }
            output[name] = group_summary
        return output

    dt_values = [(b - a) * 1e-9 for a, b in zip(stamps, stamps[1:])]
    finite_dt = [dt for dt in dt_values if math.isfinite(dt)]
    phase_offsets = [index * 0.5 for index in range(-20, 21)]
    phase = {
        "ground_truth_twist_to_truth_pose": phase_sweep(truth, "gt", phase_offsets),
        "odom_twist_to_odom_pose": phase_sweep(odom, "odom", phase_offsets),
        "interpretation": (
            "post-hoc ±10 ms phase sweep in 0.5 ms steps; not a production shift"
        ),
    }
    source_metadata = {}
    if trajectory and trajectory.is_file():
        source_metadata["trajectory_path"] = str(trajectory)
        source_metadata["trajectory_sha256"] = _sha256_file(trajectory)
    for label, rows in (("bag", list(bag.rglob("*.db3")) +
                         list(bag.rglob("*.mcap"))),):
        if rows:
            digest = hashlib.sha256()
            for path in sorted(rows):
                digest.update(path.name.encode())
                digest.update(bytes.fromhex(_sha256_file(path)))
            source_metadata[label + "_sha256"] = digest.hexdigest()

    summary = {
        "schema": SCHEMA,
        "bag": str(bag),
        "source_hashes": source_metadata,
        "alignment": {
            **join,
            "method": "exact integer ROS header stamp nanoseconds; no nearest-neighbor join",
            "gt_frame": truth[0].get("frame"),
            "gt_child": truth[0].get("child"),
            "odom_frame": odom[0].get("frame"),
            "odom_child": odom[0].get("child"),
            "fixed_initial_pose_alignment": True,
            "pose_reference": "rear axle; Unity GPS Transform is Rear Axle Center in F1TENTH prefab",
            "gt_velocity_reference_conversion":
                f"v_rear = v_COM - yaw_rate * {COM_TO_REAR_AXLE_M:.5f} m",
            "exact_pairs_retained_before_cutoff": len(pairs),
            "pairs_scored": len(selected),
            "pairs_removed_by_collision_or_fault_cutoff": len(pairs) - len(selected),
            "cutoff_receipt_time_s": cutoff_receipt_s,
            "cutoff_reason": cutoff_reason,
            "source_model_pose_exact_coverage": sum(row is not None
                                                      for row in plant_rows),
            "plant_duplicate_stamps": plant_duplicate_count,
            "imu_exact_epoch_coverage": imu_exact_coverage,
            "imu_orientation_valid_exact_coverage": imu_orientation_coverage,
        },
        "sample_interval_s": {
            "count": len(finite_dt),
            "median": statistics.median(finite_dt) if finite_dt else None,
            "p95": evaluator.percentile(finite_dt, 0.95) if finite_dt else None,
            "max": max(finite_dt) if finite_dt else None,
            "outside_20_to_30ms": sum(not 0.020 <= dt <= 0.030 for dt in finite_dt),
        },
        "velocity_and_heading_errors": velocity_summary,
        "causal_ablation_budget": causal_budget,
        "imu_yaw_ablation_errors": {
            name: _metric(values) for name, values in imu_yaw_errors.items()
        },
        "pose_error_decomposition": error_summary,
        "per_interval_pose_twist_closure": {
            name: {
                "forward_m": _metric([row["forward_m"] for row in values]),
                "lateral_m": _metric([row["lateral_m"] for row in values]),
                "xy_m": _metric([row["xy_m"] for row in values]),
            }
            for name, values in interval_closures.items()
        },
        "per_lap_count_receipt_join": summarize_groups(lap_groups),
        "per_5m_unwrapped_progress_bin": summarize_groups(progress_groups),
        "progress_join_semantics": (
            "only embedded exact diagnostic GT/source stamp; missing progress remains missing"
        ),
        "delivery_and_epoch_diagnostics": delivery_summary,
        "integration_phase_sweep": phase,
        "limits": [
            "Fixed initial rigid-pose alignment only; no post-hoc trajectory fit.",
            "1 ms integration linearly interpolates observed endpoint states; it is a numerical-method diagnostic, not 1 kHz sensor truth.",
            "Lap count is headerless and associated by nearest bag receipt within 20 ms.",
            "Phase minimum is diagnostic only; no timestamp correction is applied.",
            "Source-model pose is included only when plant state exists at every exact matched epoch.",
            "Velocity-replacement ablations are offline causal attribution, not corrected odometry estimates.",
        ],
    }
    return summary, sample_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path)
    parser.add_argument("--trajectory", type=Path)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    summary, rows = analyze(args.bag, args.trajectory)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = args.output_prefix.with_suffix(".json")
    csv_path = args.output_prefix.with_suffix(".csv")
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {json_path} and {csv_path}")
    print("exact GT↔odom stamps: "
          f"{summary['alignment']['exact_matches']}/"
          f"{summary['alignment']['truth_rows']} GT, "
          f"{summary['alignment']['odom_rows']} odom; scored="
          f"{summary['alignment']['pairs_scored']}")
    for name, values in summary["pose_error_decomposition"].items():
        print(f"{name}: endpoint={values['endpoint_xy_m']:.4f}m "
              f"path_RMSE={values['xy_m']['rmse']:.4f}m "
              f"p95={values['xy_m']['p95_abs']:.4f}m "
              f"max={values['xy_m']['max_abs']:.4f}m")
    for name, values in summary["velocity_and_heading_errors"].items():
        print(f"{name}: bias={values['bias']:+.5f} RMSE={values['rmse']:.5f} "
              f"p95_abs={values['p95_abs']:.5f} max_abs={values['max_abs']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
