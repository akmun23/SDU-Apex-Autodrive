"""Build a coherent, source-time diagnostic CSV from a telemetry recording.

The live recorder deliberately keeps one row for each callback.  That file is
the lossless run record, but callback order is not a valid way to compare a
ground-truth pose with AMCL or odometry.  This offline tool creates one row per
timestamped simulator ground-truth odometry sample and attaches only nearby
source events.  It never interpolates a controller signal and never feeds a
runtime node.

The output contains both the source timestamp and the match offset for every
attached event.  It also reports local-observer and map-frame errors using the
first clean pose as the frame alignment.  Samples after a simulator collision
are marked invalid instead of being interpreted as localization drift.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
from pathlib import Path
from typing import Iterable


EVENT_FIELDS: dict[str, tuple[str, ...]] = {
    "gt_odom": (
        "gt_x_m", "gt_y_m", "gt_z_m", "gt_yaw_rad", "gt_vx_mps",
        "gt_vy_mps", "gt_vz_mps", "gt_speed_mps", "gt_ax_mps2",
        "gt_ay_mps2", "gt_accel_mps2", "gt_longitudinal_accel_mps2",
        "gt_dt_s", "gt_yaw_rate_radps", "gt_collision_count",
    ),
    "odom_diagnostics": (
        "odom_observer_version", "odom_observer_source_stamp_s",
        "odom_observer_dt_s", "odom_raw_wheel_speed_mps",
        "odom_corrected_wheel_speed_mps", "odom_observer_speed_pred_mps",
        "odom_diagnostics_speed_mps", "odom_observer_body_u_mps",
        "odom_observer_body_v_mps", "odom_observer_ax_mps2",
        "odom_observer_ay_mps2", "odom_observer_yaw_rate_radps",
        "odom_observer_wheel_update_used", "odom_observer_turn_mode",
        "odom_observer_reset_epoch", "odom_observer_timing_degraded",
        "odom_observer_packet_drop_count",
        "odom_observer_packet_coherence_fault_count", "odom_observer_x_m",
        "odom_observer_y_m", "odom_observer_yaw_rad",
        "odom_observer_sensor_outlier", "odom_observer_left_angle_rad",
        "odom_observer_right_angle_rad", "odom_observer_imu_yaw_rad",
    ),
    "ekf_odom": (
        "x_ekf_odom_m", "y_ekf_odom_m", "yaw_ekf_odom_rad",
        "ekf_odom_xy_variance", "ekf_odom_yaw_variance",
    ),
    "ekf": (
        "x_ekf_m", "y_ekf_m", "yaw_ekf_rad", "ekf_xy_variance",
        "ekf_yaw_variance",
    ),
    "amcl": (
        "x_amcl_m", "y_amcl_m", "yaw_amcl_rad", "amcl_xy_variance",
        "amcl_yaw_variance",
    ),
    "current_map_pose": (
        "x_current_map_m", "y_current_map_m", "yaw_current_map_rad",
        "current_map_xy_variance", "current_map_yaw_variance",
    ),
    "pp_diagnostics": (
        "pp_pose_x_m", "pp_pose_y_m", "pp_pose_yaw_rad", "pp_velocity_mps",
        "pp_valid", "pp_cross_track_error_m", "pp_closest_distance_m",
        "pp_heading_error_rad", "pp_lookahead_distance_m",
        "pp_closest_index", "pp_target_index", "pp_target_speed_mps",
        "pp_raw_steering_rad", "pp_command_steering_rad",
        "pp_command_speed_mps", "pp_target_x_m", "pp_target_y_m",
    ),
    "amcl_scan_alignment": (
        "amcl_scan_stamp_s", "amcl_scan_matched_odom_stamp_s",
        "amcl_scan_source_error_ms", "amcl_scan_accepted",
        "amcl_scan_queued", "amcl_scan_dropped",
        "amcl_scan_bracket_before_stamp_s",
        "amcl_scan_bracket_after_stamp_s", "amcl_scan_processing_dropped",
    ),
    "amcl_localization_health": (
        "amcl_health_correction_age_s", "amcl_health_correction_accepted",
        "amcl_health_rejected_scans", "amcl_health_degraded",
        "amcl_health_xy_variance", "amcl_health_yaw_variance",
        "amcl_health_scan_correction_distance_m",
        "amcl_health_scan_correction_yaw_rad",
        "amcl_health_applied_xy_correction_m",
        "amcl_health_applied_yaw_correction_rad",
    ),
    "collision": ("gt_collision_count",),
    "imu": (
        "ax_mps2", "ay_mps2", "az_mps2", "imu_accel_norm_mps2",
        "imu_yaw_rate_radps", "imu_yaw_rad",
    ),
    "left_encoder": ("left_encoder_rad", "left_encoder_speed_radps"),
    "right_encoder": ("right_encoder_rad", "right_encoder_speed_radps"),
    "lidar": ("lidar_rate_hz",),
}


def _finite(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _angle_diff(a: float, b: float) -> float:
    return math.atan2(math.sin(a - b), math.cos(a - b))


def _read_events(path: str | Path) -> dict[str, list[tuple[float, dict[str, str]]]]:
    events: dict[str, dict[float, dict[str, str]]] = {}
    with Path(path).open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            name = row.get("source_event_name", "")
            stamp = _finite(row.get("source_event_stamp_s"))
            if not name or stamp is None:
                continue
            # Duplicate callback snapshots for one source event are expected;
            # the last one has the freshest recorder state.
            events.setdefault(name, {})[stamp] = row
    return {
        name: sorted(values.items())
        for name, values in events.items()
    }


def _nearest(
    samples: list[tuple[float, dict[str, str]]],
    stamps: list[float],
    target: float,
    tolerance_s: float,
) -> tuple[float, dict[str, str]] | None:
    if not samples:
        return None
    index = bisect.bisect_left(stamps, target)
    candidates = []
    if index < len(samples):
        candidates.append(samples[index])
    if index:
        candidates.append(samples[index - 1])
    match = min(candidates, key=lambda item: abs(item[0] - target))
    return match if abs(match[0] - target) <= tolerance_s else None


def _set_pose_frame(
    gt: tuple[float, float, float],
    initial_gt: tuple[float, float, float],
    initial_pose: tuple[float, float, float] | None,
) -> tuple[float, float, float] | None:
    if initial_pose is None:
        return None
    gx, gy, gyaw = gt
    gx0, gy0, gyaw0 = initial_gt
    px0, py0, pyaw0 = initial_pose
    angle = pyaw0 - gyaw0
    dx = gx - gx0
    dy = gy - gy0
    c = math.cos(angle)
    s = math.sin(angle)
    return (
        px0 + c * dx - s * dy,
        py0 + s * dx + c * dy,
        pyaw0 + _angle_diff(gyaw, gyaw0),
    )


def _add_error(row: dict[str, object], prefix: str, estimate: str, truth: str) -> None:
    x = _finite(row.get(f"{estimate}_x_m"))
    y = _finite(row.get(f"{estimate}_y_m"))
    tx = _finite(row.get(f"{truth}_x_m"))
    ty = _finite(row.get(f"{truth}_y_m"))
    yaw = _finite(row.get(f"{estimate}_yaw_rad"))
    tyaw = _finite(row.get(f"{truth}_yaw_rad"))
    if None not in (x, y, tx, ty):
        row[f"{prefix}_xy_error_m"] = math.hypot(x - tx, y - ty)
    if None not in (yaw, tyaw):
        row[f"{prefix}_yaw_error_rad"] = _angle_diff(yaw, tyaw)


def build_report(
    input_csv: str | Path,
    output_csv: str | Path,
    *,
    tolerance_s: float = 0.006,
    output_json: str | Path | None = None,
) -> dict[str, object]:
    """Write a source-time report and return its summary."""
    events = _read_events(input_csv)
    truth = events.get("gt_odom", [])
    if len(truth) < 2:
        raise ValueError("recording has fewer than two gt_odom source events")

    source_stamps = {
        name: [stamp for stamp, _ in samples]
        for name, samples in events.items()
    }

    first_gt = None
    for stamp, source in truth:
        values = tuple(_finite(source.get(key)) for key in (
            "gt_x_m", "gt_y_m", "gt_yaw_rad"))
        if None not in values:
            first_gt = (stamp, values, source)
            break
    if first_gt is None:
        raise ValueError("gt_odom events do not contain a finite pose")

    # AMCL is normally published before the first GT odometry callback during
    # startup.  Use the first finite AMCL pose, but only for frame alignment;
    # the output still preserves its source-time match and age.
    initial_map_pose = None
    for _, source in events.get("amcl", []):
        values = tuple(_finite(source.get(key)) for key in (
            "x_amcl_m", "y_amcl_m", "yaw_amcl_rad"))
        if None not in values:
            initial_map_pose = values
            break

    rows: list[dict[str, object]] = []
    previous_gt_pose: tuple[float, float, float] | None = None
    reset_seen = False
    for gt_stamp, gt_source in truth:
        row: dict[str, object] = {"gt_source_stamp_s": gt_stamp}
        for key in EVENT_FIELDS["gt_odom"]:
            if key in gt_source and gt_source[key] != "":
                row[key] = gt_source[key]

        for event, fields in EVENT_FIELDS.items():
            if event == "gt_odom":
                continue
            match = _nearest(events.get(event, []), source_stamps.get(event, []),
                             gt_stamp, tolerance_s)
            row[f"{event}_match_dt_ms"] = (
                (match[0] - gt_stamp) * 1000.0 if match is not None else "")
            row[f"{event}_source_stamp_s"] = match[0] if match is not None else ""
            if match is not None:
                for key in fields:
                    if key in match[1] and match[1][key] != "":
                        row[f"{event}_{key}"] = match[1][key]

        gt_values = tuple(_finite(row.get(key)) for key in (
            "gt_x_m", "gt_y_m", "gt_yaw_rad"))
        gt_step_distance = None
        gt_step_yaw = None
        gt_reset_detected = False
        if None not in gt_values:
            gt_pose = gt_values
            if previous_gt_pose is not None:
                gt_step_distance = math.hypot(
                    gt_pose[0] - previous_gt_pose[0],
                    gt_pose[1] - previous_gt_pose[1],
                )
                gt_step_yaw = abs(_angle_diff(gt_pose[2], previous_gt_pose[2]))
                # A normal 20 Hz vehicle sample cannot teleport by this much.
                # The yaw condition prevents a fast straight-line sample from
                # being mistaken for the simulator's reset.  The larger
                # distance-only limit catches resets with no yaw update.
                gt_reset_detected = (
                    (gt_step_distance > 0.75 and gt_step_yaw > 0.30)
                    or gt_step_distance > 1.50
                )
                reset_seen = reset_seen or gt_reset_detected
            previous_gt_pose = gt_pose
        if gt_step_distance is not None:
            row["gt_step_distance_m"] = gt_step_distance
        if gt_step_yaw is not None:
            row["gt_step_yaw_change_rad"] = gt_step_yaw
        row["gt_reset_detected"] = int(gt_reset_detected)

        collision = _finite(row.get("gt_collision_count"))
        if collision is None:
            collision = _finite(row.get("collision_gt_collision_count"))
        row["collision_epoch"] = "" if collision is None else collision
        row["pre_collision_valid"] = int(
            not reset_seen and (collision is None or collision <= 0.0))

        if None not in gt_values:
            gt_pose = gt_values
            initial_values = first_gt[1]
            # The observer starts in a body-aligned local frame. Rotate the
            # simulator's world-frame displacement by the initial truth yaw,
            # matching covariance_calibration.py and ReferenceObserver.
            c0 = math.cos(initial_values[2])
            s0 = math.sin(initial_values[2])
            dx = gt_pose[0] - initial_values[0]
            dy = gt_pose[1] - initial_values[1]
            local = (
                c0 * dx + s0 * dy,
                -s0 * dx + c0 * dy,
                _angle_diff(gt_pose[2], initial_values[2]),
            )
            row.update({
                "gt_local_x_m": local[0], "gt_local_y_m": local[1],
                "gt_local_yaw_rad": local[2],
            })
            map_pose = _set_pose_frame(gt_pose, initial_values, initial_map_pose)
            if map_pose is not None:
                row.update({
                    "gt_map_x_m": map_pose[0], "gt_map_y_m": map_pose[1],
                    "gt_map_yaw_rad": map_pose[2],
                })

        # Rename the event-specific estimates into a common form, allowing a
        # single spreadsheet expression to compare all localization stages.
        for event, x_field, y_field, yaw_field, target in (
            ("odom_diagnostics", "odom_observer_x_m", "odom_observer_y_m",
             "odom_observer_yaw_rad", "observer"),
            ("ekf_odom", "x_ekf_odom_m", "y_ekf_odom_m",
             "yaw_ekf_odom_rad", "ekf_odom"),
            ("ekf", "x_ekf_m", "y_ekf_m", "yaw_ekf_rad", "ekf"),
            ("amcl", "x_amcl_m", "y_amcl_m", "yaw_amcl_rad", "amcl"),
            ("current_map_pose", "x_current_map_m", "y_current_map_m",
             "yaw_current_map_rad", "current_map"),
            ("pp_diagnostics", "pp_pose_x_m", "pp_pose_y_m",
             "pp_pose_yaw_rad", "pp"),
        ):
            source_x = f"{event}_{x_field}"
            source_y = f"{event}_{y_field}"
            source_yaw = f"{event}_{yaw_field}"
            if source_x in row:
                row[f"{target}_x_m"] = row[source_x]
            if source_y in row:
                row[f"{target}_y_m"] = row[source_y]
            if source_yaw in row:
                row[f"{target}_yaw_rad"] = row[source_yaw]

        if all(key in row for key in ("gt_local_x_m", "gt_local_y_m")):
            _add_error(row, "observer", "observer", "gt_local")
            _add_error(row, "ekf_odom", "ekf_odom", "gt_local")
        if all(key in row for key in ("gt_map_x_m", "gt_map_y_m")):
            _add_error(row, "amcl", "amcl", "gt_map")
            _add_error(row, "current_map", "current_map", "gt_map")
            _add_error(row, "ekf", "ekf", "gt_map")
            _add_error(row, "pp", "pp", "gt_map")
        rows.append(row)

    # Keep a stable, spreadsheet-friendly schema while retaining every field
    # produced by this report.  Missing matches are intentionally blank.
    fields = ["gt_source_stamp_s"]
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    def values(name: str, valid_only: bool = False) -> list[float]:
        result = []
        for row in rows:
            if valid_only and not row.get("pre_collision_valid"):
                continue
            value = _finite(row.get(name))
            if value is not None:
                result.append(value)
        return result

    valid = [row for row in rows if row.get("pre_collision_valid")]
    collisions = [row for row in rows if not row.get("pre_collision_valid")]
    summary: dict[str, object] = {
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "rows": len(rows),
        "pre_collision_rows": len(valid),
        "post_collision_rows": len(collisions),
        "source_events": {name: len(samples) for name, samples in events.items()},
        "match_tolerance_ms": tolerance_s * 1000.0,
        "max_pre_collision_observer_error_m": max(
            values("observer_xy_error_m", True), default=None),
        "max_pre_collision_ekf_odom_error_m": max(
            values("ekf_odom_xy_error_m", True), default=None),
        "max_pre_collision_amcl_error_m": max(
            values("amcl_xy_error_m", True), default=None),
        "max_pre_collision_current_map_error_m": max(
            values("current_map_xy_error_m", True), default=None),
        "first_post_collision_stamp_s": (
            collisions[0]["gt_source_stamp_s"] if collisions else None),
    }
    if output_json is not None:
        json_path = Path(output_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv")
    parser.add_argument("output_csv")
    parser.add_argument("--match-tolerance-ms", type=float, default=6.0)
    parser.add_argument("--output-json", default="")
    args = parser.parse_args(argv)
    summary = build_report(
        args.input_csv,
        args.output_csv,
        tolerance_s=max(0.0, args.match_tolerance_ms) / 1000.0,
        output_json=args.output_json or None,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
