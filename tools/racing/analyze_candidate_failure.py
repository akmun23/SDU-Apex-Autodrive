#!/usr/bin/env python3
"""Extract first-collision timelines and conservative failure evidence from bags.

All truth is used offline as a scoring label only. This tool does not connect to
ROS, alter a bag, or feed simulator truth into runtime nodes.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
except ImportError as exc:  # pragma: no cover - supplied by the project ROS image
    raise SystemExit(f"ROS 2 Python modules are required: {exc}")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.racing.track_projection import TrackProjection, wrap_angle  # noqa: E402

COLLISION = "/autodrive/roboracer_1/collision_count"
TRUTH = "/autodrive/roboracer_1/odom"
MAP_POSE = "/current_map_pose"
ODOM_DIAG = "/odom/diagnostics"
MPC_DIAG = "/mpc/diagnostics"
STEERING = "/autodrive/roboracer_1/steering"
STEERING_COMMAND = "/autodrive/roboracer_1/steering_command"
THROTTLE = "/autodrive/roboracer_1/throttle"
THROTTLE_COMMAND = "/autodrive/roboracer_1/throttle_command"
IMU = "/autodrive/roboracer_1/imu"
LEFT_ENCODER = "/autodrive/roboracer_1/left_encoder"
RIGHT_ENCODER = "/autodrive/roboracer_1/right_encoder"
DEFAULT_CENTERLINE = Path(
    "live_runs/raceline_candidates/practice_exact_84cap_clearance035_20261004/"
    "track_prep/SmoothCenterline.csv"
)
ODOM_FIELDS = {
    "version": 0, "published_stamp_s": 1, "packet_dt_s": 2,
    "wheel_raw_mps": 3, "wheel_mapped_mps": 4,
    "speed_pred_mps": 5, "speed_mps": 6, "body_u_mps": 7,
    "body_v_mps": 8, "ax_mps2": 9, "ay_mps2": 10,
    "yaw_rate_radps": 11, "wheel_update_used": 12,
    "reset_epoch": 14, "timing_degraded": 15,
    "packet_drop_count": 16, "packet_coherence_fault_count": 17,
    "sensor_outlier": 21, "left_angle_rad": 22,
    "right_angle_rad": 23, "imu_yaw_rad": 24,
    "wheel_packet_mps": 25, "wheel_burst_rejected": 26,
    "packet_source_stamp_s": 28, "previous_packet_source_stamp_s": 29,
    "was_retimestamped": 30, "retimestamped_packet_count": 31,
    "latest_seen_source_stamp_s": 32, "last_emitted_source_stamp_s": 33,
    "source_reversal_count": 34, "duplicate_source_count": 35,
    "late_completed_packet_count": 36, "incomplete_packet_drop_count": 37,
    "maximum_reorder_depth": 38, "maximum_reorder_time_s": 39,
}


@dataclass
class Row:
    receipt_ns: int
    stamp_ns: int
    message: Any


def stamp_ns(message: Any) -> int:
    header = getattr(message, "header", None)
    if header is None:
        return 0
    stamp = header.stamp
    value = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
    return value if value > 0 else 0


def quat_yaw(q: Any) -> float:
    return math.atan2(
        2.0 * (float(q.w) * float(q.z) + float(q.x) * float(q.y)),
        1.0 - 2.0 * (float(q.y) ** 2 + float(q.z) ** 2),
    )


def pose_of(message: Any) -> Any:
    return message.pose.pose if hasattr(message.pose, "pose") else message.pose


def read_topic(
    connection: sqlite3.Connection,
    topics: dict[str, tuple[int, str]],
    name: str,
) -> list[Row]:
    if name not in topics:
        return []
    topic_id, type_name = topics[name]
    message_type = get_message(type_name)
    output: list[Row] = []
    for receipt, raw in connection.execute(
        "SELECT timestamp,data FROM messages WHERE topic_id=? ORDER BY timestamp,id",
        (topic_id,),
    ):
        message = deserialize_message(bytes(raw), message_type)
        output.append(Row(int(receipt), stamp_ns(message), message))
    return output


def nearest_prior(rows: list[Row], stamps: list[int], target_ns: int) -> Row | None:
    index = bisect.bisect_right(stamps, target_ns) - 1
    if index < 0 or target_ns - stamps[index] > 60_000_000:
        return None
    return rows[index]


def float_value(rows: list[Row], stamps: list[int], target_ns: int) -> float | None:
    row = nearest_prior(rows, stamps, target_ns)
    return None if row is None else float(row.message.data)


def source_order(
    rows: list[Row], *, vector_stamp_index: int | None = None,
    current_vector_stamp_index: int | None = None,
) -> dict[str, Any]:
    previous: int | None = None
    reversals = duplicates = 0
    first_reversal_receipt: int | None = None
    reversal_receipts: list[int] = []
    for row in rows:  # preserve bag receipt order; sorting would hide late arrivals
        source = row.stamp_ns
        if vector_stamp_index is not None:
            values = list(row.message.data)
            index = vector_stamp_index
            if (
                current_vector_stamp_index is not None
                and len(values) > current_vector_stamp_index
                and values[0] >= 6.0
            ):
                index = current_vector_stamp_index
            if len(values) <= index or not math.isfinite(values[index]):
                continue
            source = int(values[index] * 1e9)
        if source <= 0:
            continue
        if previous is not None:
            if source < previous:
                reversals += 1
                reversal_receipts.append(row.receipt_ns)
                if first_reversal_receipt is None:
                    first_reversal_receipt = row.receipt_ns
            elif source == previous:
                duplicates += 1
        previous = source
    return {
        "reversals": reversals,
        "duplicates": duplicates,
        "first_reversal_receipt_ns": first_reversal_receipt,
        "reversal_receipts_ns": reversal_receipts,
    }


def parse_odom_diagnostic(row: Row | None) -> dict[str, float | None]:
    if row is None:
        return {}
    values = list(row.message.data)
    return {
        name: float(values[index]) if index < len(values) else None
        for name, index in ODOM_FIELDS.items()
    }


def parse_mpc(row: Row | None) -> dict[str, Any]:
    if row is None:
        return {}
    try:
        data = json.loads(row.message.data)
    except (TypeError, json.JSONDecodeError):
        return {}
    state = data.get("state") or []
    prediction = next(
        (item for item in (data.get("predictions") or []) if int(item.get("n", -1)) == 1),
        None,
    )
    predicted_state = (prediction or {}).get("state") or []
    failure = data.get("r1_nonlinear_failure") or data.get("r2_nonlinear_failure") or {}
    reference = data.get("reference") or {}
    bounds = (prediction or {}).get("corridor") or []
    failure_reference = failure.get("reference") or []
    observed = data.get("observed_command") or {}
    action = data.get("first_action") or []
    return {
        "mpc_status": data.get("status"),
        "mpc_progress_m": data.get("progress_m"),
        "mpc_e_y_m": state[0] if len(state) > 0 else None,
        "mpc_e_psi_rad": state[1] if len(state) > 1 else None,
        "mpc_u_mps": state[2] if len(state) > 2 else None,
        "mpc_v_mps": state[3] if len(state) > 3 else None,
        "mpc_r_radps": state[4] if len(state) > 4 else None,
        "reference_speed_mps": reference.get("speed_mps"),
        "target_speed_mps": reference.get("target_speed_mps"),
        "target_speed_rate_mps2": action[3] if len(action) > 3 else observed.get("target_speed_rate_mps2"),
        "q_delta_radps": action[2] if len(action) > 2 else observed.get("steering_rate_radps"),
        "observed_target_speed_rate_mps2": observed.get("target_speed_rate_mps2"),
        "observed_steering_rate_radps": observed.get("steering_rate_radps"),
        "predicted_r_25ms_radps": predicted_state[4] if len(predicted_state) > 4 else None,
        "corridor_left_bound_m": bounds[0] if len(bounds) > 0 else failure_reference[3] if len(failure_reference) > 3 else None,
        "corridor_right_bound_m": bounds[1] if len(bounds) > 1 else failure_reference[4] if len(failure_reference) > 4 else None,
        "corridor_lower_bound_m": failure.get("lower_bound_m"),
        "corridor_upper_bound_m": failure.get("upper_bound_m"),
        "nonlinear_reject_reason": data.get("nonlinear_failure_reason"),
        "nonlinear_reject_stage": data.get("nonlinear_failure_stage"),
        "r1_status": data.get("r1_status"),
        "r2_status": data.get("r2_status"),
        "r1_failure_reason": data.get("r1_nonlinear_failure_reason"),
        "r2_failure_reason": data.get("r2_nonlinear_failure_reason"),
        "r1_failure_stage": data.get("r1_nonlinear_failure_stage"),
        "r2_failure_stage": data.get("r2_nonlinear_failure_stage"),
        "first_action_steering_rad": action[0] if len(action) > 0 else None,
        "first_action_speed_mps": action[1] if len(action) > 1 else None,
        "steering_feedback_used_rad": data.get("steering_feedback_rad"),
        "minimum_predicted_corridor_slack_m": data.get("minimum_predicted_corridor_slack_m"),
    }


def load_baseline_thresholds(
    report_dirs: list[Path], baseline_bags: list[Path]
) -> dict[str, float]:
    """Use P0 p99 residuals as event thresholds, not guessed physical cutoffs."""
    columns = {
        "yaw_rate_error_radps": "abs_yaw_rate_error_radps",
        "speed_error_mps": "abs_speed_error_mps",
        "lateral_error_m": "abs_lateral_error_m",
    }
    values: dict[str, list[float]] = {key: [] for key in columns}
    for directory in report_dirs:
        for filename, target in (
            ("model_prediction_error.csv", "model"),
            ("localization_error.csv", "localization"),
        ):
            path = directory / filename
            if not path.is_file():
                continue
            with path.open(newline="", encoding="utf-8") as stream:
                for row in csv.DictReader(stream):
                    if target == "model":
                        for field in ("yaw_rate_error_radps", "speed_error_mps", "lateral_error_m"):
                            try:
                                value = abs(float(row[field]))
                            except (KeyError, ValueError, TypeError):
                                continue
                            if math.isfinite(value):
                                values[field].append(value)
                    elif row.get("stream") == MAP_POSE:
                        for field in ("normal_error_m", "tangential_error_m"):
                            try:
                                value = abs(float(row[field]))
                            except (KeyError, ValueError, TypeError):
                                continue
                            if math.isfinite(value):
                                values.setdefault(field, []).append(value)
    thresholds: dict[str, float] = {}
    for field, samples in values.items():
        ordered = sorted(samples)
        if ordered:
            index = min(len(ordered) - 1, math.ceil(0.99 * len(ordered)) - 1)
            thresholds[field] = ordered[index]

    wheel_body: list[float] = []
    wheel_filter: list[float] = []
    for bag in baseline_bags:
        if not bag.is_file():
            continue
        connection = sqlite3.connect(bag.resolve().as_uri() + "?mode=ro", uri=True)
        try:
            topics = {
                name: (int(topic_id), message_type)
                for topic_id, name, message_type in connection.execute(
                    "SELECT id,name,type FROM topics"
                )
            }
            truth_rows = read_topic(connection, topics, TRUTH)
            diag_rows = read_topic(connection, topics, ODOM_DIAG)
            lap_rows = read_topic(connection, topics, "/autodrive/roboracer_1/lap_count")
        finally:
            connection.close()
        truth_rows.sort(key=lambda row: row.receipt_ns)
        truth_receipts = [row.receipt_ns for row in truth_rows]
        lap_rows.sort(key=lambda row: row.receipt_ns)
        lap_stamps = [row.receipt_ns for row in lap_rows]
        for diag in diag_rows:
            data = list(diag.message.data)
            if len(data) <= ODOM_FIELDS["wheel_mapped_mps"]:
                continue
            raw = float(data[ODOM_FIELDS["wheel_raw_mps"]])
            mapped = float(data[ODOM_FIELDS["wheel_mapped_mps"]])
            if math.isfinite(raw) and math.isfinite(mapped):
                wheel_filter.append(abs(raw - mapped))
            truth_row = nearest_prior(truth_rows, truth_receipts, diag.receipt_ns)
            if truth_row is not None:
                truth_u = float(truth_row.message.twist.twist.linear.x)
                lap_row = nearest_prior(lap_rows, lap_stamps, diag.receipt_ns)
                lap_number = int(lap_row.message.data) if lap_row is not None else 0
                moving = abs(truth_u) >= 1.0
                scored = 2 <= lap_number <= 11
                wheel_used = len(data) > ODOM_FIELDS["wheel_update_used"] and data[ODOM_FIELDS["wheel_update_used"]] >= 0.5
                burst_rejected = len(data) > ODOM_FIELDS["wheel_burst_rejected"] and data[ODOM_FIELDS["wheel_burst_rejected"]] >= 0.5
                if moving and scored and wheel_used and not burst_rejected and math.isfinite(mapped) and math.isfinite(truth_u):
                    wheel_body.append(abs(mapped - truth_u))
    for key, samples in (
        ("wheel_body_mismatch_mps", wheel_body),
        ("wheel_raw_mapped_delta_mps", wheel_filter),
    ):
        ordered = sorted(samples)
        if ordered:
            index = min(len(ordered) - 1, math.ceil(0.99 * len(ordered)) - 1)
            thresholds[key] = ordered[index]
    return thresholds


def integrated_lap_estimate(path: Path | None) -> float | None:
    if path is None or not path.is_file():
        return None
    try:
        from tools.racing.analyze_race_run import load_reference_speed
        from tools.racing.track_projection import TrackProjection
        trajectory = TrackProjection.from_csv(path, closed=True)
        speeds = load_reference_speed(path)
        if len(speeds) == len(trajectory.x) + 1:
            speeds = speeds[:-1]
        if len(speeds) != len(trajectory.x) or min(speeds) <= 0:
            return None
        return sum(
            math.hypot(
                trajectory.x[(i + 1) % len(speeds)] - trajectory.x[i],
                trajectory.y[(i + 1) % len(speeds)] - trajectory.y[i],
            ) / speeds[i]
            for i in range(len(speeds))
        )
    except (ImportError, OSError, ValueError, ZeroDivisionError):
        return None


def recursive_numeric_key(value: Any, names: set[str]) -> float | None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in names:
                try:
                    return float(item)
                except (TypeError, ValueError):
                    pass
        for item in value.values():
            found = recursive_numeric_key(item, names)
            if found is not None:
                return found
    elif isinstance(value, list):
        for item in value:
            found = recursive_numeric_key(item, names)
            if found is not None:
                return found
    return None


def candidate_path_clearance(
    trajectory_path: Path | None, centerline: TrackProjection
) -> tuple[float | None, float | None, float | None]:
    if trajectory_path is None or not trajectory_path.is_file():
        return None, None, None
    try:
        trajectory = TrackProjection.from_csv(trajectory_path, closed=True)
    except (OSError, ValueError):
        return None, None, None
    previous: int | None = None
    minimum = math.inf
    minimum_s: float | None = None
    for index, (x, y) in enumerate(zip(trajectory.x, trajectory.y)):
        following = (index + 1) % len(trajectory.x)
        yaw = math.atan2(trajectory.y[following] - y, trajectory.x[following] - x)
        projection = centerline.project(
            x, y, yaw,
            previous_segment=previous,
            local_search_radius=16 if previous is not None else None,
        )
        if projection.distance_m > 0.75:
            projection = centerline.project(x, y, yaw)
        previous = projection.segment_index
        left = projection.left_width_m - projection.lateral_offset_m - 0.1365
        right = projection.right_width_m + projection.lateral_offset_m - 0.1365
        margin = min(left, right)
        if math.isfinite(margin) and margin < minimum:
            minimum = margin
            minimum_s = projection.s_m
    if not math.isfinite(minimum):
        return None, None, None

    required: float | None = None
    for parent in (trajectory_path.parent, *trajectory_path.parents):
        for filename in ("optimizer_config.yaml", "resolved_config.yaml"):
            config_path = parent / filename
            if not config_path.is_file():
                continue
            try:
                import yaml
                config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            except (ImportError, OSError, ValueError):
                continue
            required = recursive_numeric_key(config, {"extra_wall_clearance_m"})
            if required is None:
                required = recursive_numeric_key(config, {"required_wall_clearance_m"})
            if required is not None:
                break
        if required is not None:
            break
    return minimum, minimum_s, required


def run_inputs(run_dir: Path) -> tuple[Path | None, Path | None, float | None]:
    trajectory: Path | None = None
    estimated_lap: float | None = None
    report_paths = list(run_dir.glob("analysis/**/summary.json"))
    manifest_path = run_dir / "run_manifest.json"
    candidates: list[dict[str, Any]] = []
    for report in report_paths:
        try:
            candidates.append(json.loads(report.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    for data in candidates:
        value = data.get("trajectory")
        if value:
            trajectory = normalize_path(str(value))
            break
    if trajectory is None and manifest.get("trajectory"):
        trajectory = normalize_path(str(manifest["trajectory"]))
    if estimated_lap is None:
        estimated_lap = integrated_lap_estimate(trajectory)
    track = DEFAULT_CENTERLINE if (ROOT / DEFAULT_CENTERLINE).is_file() else None
    return trajectory, track, estimated_lap


def normalize_path(value: str) -> Path:
    if value.startswith("/workspace/src/"):
        return ROOT / value.removeprefix("/workspace/src/")
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def classify_first_event(
    timeline: list[dict[str, Any]],
    thresholds: dict[str, float],
    steering_limit_rad: float,
    steering_rate_limit_radps: float,
) -> tuple[str, str, float | None]:
    def persistent(start: int, getter: Any, threshold: float, positive_only: bool = False) -> bool:
        group = timeline[start:start + 3]
        if len(group) < 3:
            return False
        values = [getter(sample) for sample in group]
        if any(value is None or not math.isfinite(float(value)) for value in values):
            return False
        return all(
            float(value) > threshold if positive_only else abs(float(value)) > threshold
            for value in values
        )

    for index, row in enumerate(timeline):
        odom = row["odom"]
        mpc = row["mpc"]
        speed = row.get("truth_u_mps")
        yaw = row.get("truth_yaw_rate_25ms_radps")
        predicted_yaw = mpc.get("predicted_r_25ms_radps")
        normal = row.get("localization_normal_error_m")
        tangent = row.get("localization_tangential_error_m")
        wheel_mismatch = None
        if odom.get("wheel_mapped_mps") is not None and speed is not None:
            wheel_mismatch = float(odom["wheel_mapped_mps"]) - float(speed)
        raw_mapped = row.get("wheel_raw_mapped_delta_mps")
        speed_pred_error = None
        if odom.get("speed_pred_mps") is not None and speed is not None:
            speed_pred_error = float(odom["speed_pred_mps"]) - float(speed)
        checks: list[tuple[bool, str, str, float | None]] = []
        wheel_threshold = thresholds.get("wheel_body_mismatch_mps", math.inf)
        if row.get("source_reversal_at_sample") and (
            (wheel_mismatch is not None and abs(wheel_mismatch) > wheel_threshold)
            or (speed_pred_error is not None and abs(speed_pred_error) > thresholds.get("speed_error_mps", math.inf))
        ):
            checks.append((True, "ODOM_SOURCE_ORDER", "source timestamp reversed in received sensor/packet order", None))
        if persistent(index, lambda sample: sample.get("wheel_body_mismatch_mps"), wheel_threshold) or (
            bool(odom.get("wheel_burst_rejected")) and raw_mapped is not None
            and abs(raw_mapped) > thresholds.get("wheel_raw_mapped_delta_mps", math.inf)
        ):
            checks.append((True, "ODOM_WHEEL_BURST", "wheel/body mismatch or observer burst-reject diagnostic", wheel_mismatch))
        if persistent(index, lambda sample: sample.get("localization_normal_error_m"), thresholds.get("normal_error_m", math.inf)):
            checks.append((True, "LOCALIZATION_NORMAL_ERROR", "map-pose normal error exceeded P0 p99", normal))
        if persistent(index, lambda sample: sample.get("localization_tangential_error_m"), thresholds.get("tangential_error_m", math.inf)):
            checks.append((True, "LOCALIZATION_TANGENTIAL_ERROR", "map-pose tangential error exceeded P0 p99", tangent))
        if predicted_yaw is not None and yaw is not None:
            residual = float(predicted_yaw) - float(yaw)
            if persistent(
                index,
                lambda sample: (
                    None if sample["mpc"].get("predicted_r_25ms_radps") is None
                    or sample.get("truth_yaw_rate_25ms_radps") is None
                    else float(sample["mpc"]["predicted_r_25ms_radps"])
                    - float(sample["truth_yaw_rate_25ms_radps"])
                ),
                thresholds.get("yaw_rate_error_radps", math.inf),
            ):
                kind = "MODEL_OVER_YAW" if abs(float(predicted_yaw)) > abs(float(yaw)) else "MODEL_UNDER_YAW"
                checks.append((True, kind, "25-ms MPC yaw prediction error exceeded P0 p99", residual))
        actual_steering = row.get("steering_feedback_rad")
        if persistent(index, lambda sample: sample.get("steering_feedback_rad"), 0.98 * steering_limit_rad):
            checks.append((True, "STEERING_MAG_SAT", "physical steering reached 98% of configured limit", actual_steering))
        steering_rate = mpc.get("observed_steering_rate_radps")
        if persistent(index, lambda sample: sample["mpc"].get("observed_steering_rate_radps"), 0.98 * steering_rate_limit_radps):
            checks.append((True, "STEERING_RATE_SAT", "requested steering rate reached 98% of configured limit", steering_rate))
        ref_speed = mpc.get("reference_speed_mps")
        speed_threshold = thresholds.get("speed_error_mps", math.inf)
        target_speed_rate = mpc.get("observed_target_speed_rate_mps2")
        if ref_speed is not None and speed is not None:
            speed_error = float(speed) - float(ref_speed)
            if persistent(index, lambda sample: (
                None if sample.get("truth_u_mps") is None or sample["mpc"].get("reference_speed_mps") is None
                else float(sample["truth_u_mps"]) - float(sample["mpc"]["reference_speed_mps"])
            ), speed_threshold, positive_only=True) and target_speed_rate is not None and float(target_speed_rate) < 0.0:
                checks.append((True, "BRAKING_OVERSHOOT", "truth speed exceeded reference beyond P0 p99 while target speed was falling", speed_error))
            else:
                group = timeline[index:index + 3]
                under = len(group) == 3 and all(
                    sample.get("truth_u_mps") is not None
                    and sample["mpc"].get("reference_speed_mps") is not None
                    and float(sample["truth_u_mps"]) - float(sample["mpc"]["reference_speed_mps"]) < -speed_threshold
                    for sample in group
                )
                if under:
                    checks.append((True, "SPEED_UNDERSHOOT", "truth speed fell below reference beyond P0 p99 for 3 samples", speed_error))
        clearance = row.get("estimated_body_wall_clearance_m")
        if clearance is not None and math.isfinite(float(clearance)) and float(clearance) <= 0.0:
            checks.append((True, "REAL_WALL_MARGIN", "truth footprint overlapped centerline-derived track boundary", clearance))
        status = str(mpc.get("mpc_status") or "")
        reason = str(mpc.get("nonlinear_reject_reason") or "")
        if status == "rejected_nonlinear_rollout" and reason == "corridor":
            checks.append((True, "CORRIDOR_PREDICTION_REJECT", "MPC nonlinear rollout rejected at corridor", mpc.get("nonlinear_reject_stage")))
        if status == "rejected_residual":
            checks.append((True, "RTI_RESIDUAL_REJECT", "MPC rejected RTI result on residual", None))
        if not checks:
            continue
        # Source-order and sensor evidence precede downstream controller symptoms
        # only when they occur on the same sample; otherwise chronology dominates.
        checks.sort(key=lambda item: [
            "ODOM_SOURCE_ORDER", "ODOM_WHEEL_BURST",
            "LOCALIZATION_NORMAL_ERROR", "LOCALIZATION_TANGENTIAL_ERROR",
            "MODEL_OVER_YAW", "MODEL_UNDER_YAW",
            "STEERING_MAG_SAT", "STEERING_RATE_SAT", "BRAKING_OVERSHOOT",
            "SPEED_UNDERSHOOT", "CORRIDOR_PREDICTION_REJECT",
            "RTI_RESIDUAL_REJECT", "REAL_WALL_MARGIN",
        ].index(item[1]))
        _, label, evidence, value = checks[0]
        return label, evidence, value
    return "UNKNOWN", "no pre-collision threshold crossing observed; or telemetry window is censored", None


def analyze_run(
    run_dir: Path,
    centerline: TrackProjection,
    thresholds: dict[str, float],
    timeline_s: float,
    steering_limit_rad: float,
    steering_rate_limit_radps: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    bag = run_dir / "run" / "run_0.db3"
    connection = sqlite3.connect(bag.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        topics = {
            name: (int(topic_id), message_type)
            for topic_id, name, message_type in connection.execute("SELECT id,name,type FROM topics")
        }
        required = (COLLISION, TRUTH)
        missing = [name for name in required if name not in topics]
        if missing:
            raise ValueError(f"{run_dir.name}: missing required bag topics {missing}")
        names = (
            COLLISION, TRUTH, MAP_POSE, ODOM_DIAG, MPC_DIAG, STEERING,
            STEERING_COMMAND, THROTTLE, THROTTLE_COMMAND, IMU,
            LEFT_ENCODER, RIGHT_ENCODER,
        )
        streams = {name: read_topic(connection, topics, name) for name in names}
    finally:
        connection.close()

    collision_rows = streams[COLLISION]
    collision_initial = int(collision_rows[0].message.data) if collision_rows else 0
    collision_final = int(collision_rows[-1].message.data) if collision_rows else collision_initial
    collision_event = next(
        (row for row in collision_rows if int(row.message.data) > collision_initial), None
    )
    censored = collision_event is None and collision_final > collision_initial or collision_initial > 0
    event_ns = collision_event.receipt_ns if collision_event else None

    truth = [row for row in streams[TRUTH] if row.stamp_ns > 0]
    truth.sort(key=lambda row: row.receipt_ns)
    truth_receipts = [row.receipt_ns for row in truth]
    map_rows = [row for row in streams[MAP_POSE] if row.stamp_ns > 0]
    map_rows.sort(key=lambda row: row.receipt_ns)
    map_receipts = [row.receipt_ns for row in map_rows]
    odom_rows = streams[ODOM_DIAG]
    odom_receipts = [row.receipt_ns for row in odom_rows]
    mpc_rows = streams[MPC_DIAG]
    mpc_receipts = [row.receipt_ns for row in mpc_rows]
    scalar_rows = {
        name: sorted(streams[name], key=lambda row: row.receipt_ns)
        for name in (STEERING, STEERING_COMMAND, THROTTLE, THROTTLE_COMMAND)
    }
    scalar_receipts = {name: [row.receipt_ns for row in rows] for name, rows in scalar_rows.items()}

    source_scope = lambda rows: [
        row for row in rows if event_ns is None or row.receipt_ns <= event_ns
    ]
    order = {
        IMU: source_order(source_scope(streams[IMU])),
        LEFT_ENCODER: source_order(source_scope(streams[LEFT_ENCODER])),
        RIGHT_ENCODER: source_order(source_scope(streams[RIGHT_ENCODER])),
        ODOM_DIAG: source_order(
            source_scope(odom_rows), vector_stamp_index=1, current_vector_stamp_index=28),
    }
    all_reversal_receipts = sorted(
        receipt
        for value in order.values()
        for receipt in value["reversal_receipts_ns"]
    )

    if event_ns is None:
        start_ns = truth[0].receipt_ns if truth else 0
        end_ns = truth[-1].receipt_ns if truth else start_ns
    else:
        start_ns = max(truth[0].receipt_ns, event_ns - int(timeline_s * 1e9)) if truth else event_ns
        end_ns = event_ns
    timeline: list[dict[str, Any]] = []
    previous_segment: int | None = None
    for truth_row in truth:
        pose = pose_of(truth_row.message)
        projection = centerline.project(
            float(pose.position.x), float(pose.position.y), quat_yaw(pose.orientation),
            previous_segment=previous_segment, local_search_radius=16,
        ) if previous_segment is not None else centerline.project(
            float(pose.position.x), float(pose.position.y), quat_yaw(pose.orientation)
        )
        if projection.distance_m > 0.75:
            projection = centerline.project(float(pose.position.x), float(pose.position.y), quat_yaw(pose.orientation))
        previous_segment = projection.segment_index
        if not start_ns <= truth_row.receipt_ns <= end_ns:
            continue

        target = truth_row.receipt_ns
        map_row = nearest_prior(map_rows, map_receipts, target)
        map_normal = map_tangent = map_yaw_error = None
        map_lateral = None
        if map_row is not None:
            estimate = pose_of(map_row.message)
            map_projection = centerline.project(
                float(estimate.position.x), float(estimate.position.y), quat_yaw(estimate.orientation),
                previous_segment=projection.segment_index, local_search_radius=16,
            )
            if map_projection.distance_m > 0.75:
                map_projection = centerline.project(float(estimate.position.x), float(estimate.position.y), quat_yaw(estimate.orientation))
            map_normal = map_projection.lateral_offset_m - projection.lateral_offset_m
            map_tangent = (map_projection.s_m - projection.s_m + 0.5 * centerline.total_length) % centerline.total_length - 0.5 * centerline.total_length
            map_yaw_error = wrap_angle(quat_yaw(estimate.orientation) - quat_yaw(pose.orientation))
            map_lateral = map_projection.lateral_offset_m

        odom_row = nearest_prior(odom_rows, odom_receipts, target)
        odom = parse_odom_diagnostic(odom_row)
        mpc_row = nearest_prior(mpc_rows, mpc_receipts, target)
        mpc = parse_mpc(mpc_row)
        truth_u = float(truth_row.message.twist.twist.linear.x)
        truth_v = float(truth_row.message.twist.twist.linear.y)
        truth_r = float(truth_row.message.twist.twist.angular.z)
        source_reversal = any(abs(target - reversal_ns) <= 25_000_000 for reversal_ns in all_reversal_receipts)
        left_clearance = projection.left_width_m - projection.lateral_offset_m - 0.1365
        right_clearance = projection.right_width_m + projection.lateral_offset_m - 0.1365
        physical_wall_margin = min(left_clearance, right_clearance)
        row = {
            "time_to_collision_s": None if event_ns is None else (target - event_ns) / 1e9,
            "receipt_time_s": (target - (truth[0].receipt_ns if truth else target)) / 1e9,
            "track_s_m": projection.s_m,
            "truth_x_m": float(pose.position.x),
            "truth_y_m": float(pose.position.y),
            "truth_lateral_m": projection.lateral_offset_m,
            "truth_u_mps": truth_u,
            "truth_v_mps": truth_v,
            "truth_speed_mps": math.hypot(truth_u, truth_v),
            "truth_yaw_rate_radps": truth_r,
            "truth_yaw_rad": quat_yaw(pose.orientation),
            "current_map_pose_lateral_m": map_lateral,
            "localization_normal_error_m": map_normal,
            "localization_tangential_error_m": map_tangent,
            "localization_yaw_error_rad": map_yaw_error,
            "estimated_body_wall_clearance_m": physical_wall_margin,
            "odom": odom,
            "mpc": mpc,
            "steering_feedback_rad": float_value(scalar_rows[STEERING], scalar_receipts[STEERING], target),
            "steering_command_normalized": float_value(scalar_rows[STEERING_COMMAND], scalar_receipts[STEERING_COMMAND], target),
            "throttle_feedback": float_value(scalar_rows[THROTTLE], scalar_receipts[THROTTLE], target),
            "throttle_command": float_value(scalar_rows[THROTTLE_COMMAND], scalar_receipts[THROTTLE_COMMAND], target),
            "source_reversal_at_sample": source_reversal,
            "source_reversal_count_so_far": sum(
                sum(1 for reversal_ns in value["reversal_receipts_ns"] if reversal_ns <= target)
                for value in order.values()
            ),
        }
        future_index = bisect.bisect_left(truth_receipts, target + 25_000_000)
        if future_index < len(truth):
            future_row = truth[future_index]
            if future_row.receipt_ns - (target + 25_000_000) <= 60_000_000:
                row["truth_yaw_rate_25ms_radps"] = float(future_row.message.twist.twist.angular.z)
            else:
                row["truth_yaw_rate_25ms_radps"] = None
        else:
            row["truth_yaw_rate_25ms_radps"] = None
        timeline.append(row)

    for row in timeline:
        mapped = row["odom"].get("wheel_mapped_mps")
        raw = row["odom"].get("wheel_raw_mps")
        row["wheel_body_mismatch_mps"] = None if mapped is None else float(mapped) - float(row["truth_u_mps"])
        row["wheel_raw_mapped_delta_mps"] = None if raw is None or mapped is None else float(raw) - float(mapped)

    trajectory, _, estimate = run_inputs(run_dir)
    path_margin, path_margin_s, required_path_margin = candidate_path_clearance(trajectory, centerline)
    available = 0.0 if event_ns is None or not truth else max(0.0, (event_ns - truth[0].receipt_ns) / 1e9)
    if collision_final <= collision_initial:
        label, evidence, value = "NONE_OBSERVED", "collision-free capture; no failure onset", None
        timeline = []
    elif available < timeline_s:
        label = "UNKNOWN"
        evidence = (
            f"censored pre-collision telemetry: {available:.3f}s available; "
            f"{timeline_s:.3f}s required"
        )
        value = None
    elif censored:
        label, evidence, value = "UNKNOWN", "bag begins after a collision or has no observed counter transition", None
    elif (path_margin is not None and required_path_margin is not None
          and path_margin < required_path_margin):
        label = "MAP_RAYCAST_MARGIN"
        evidence = "candidate path clearance fell below its configured extra wall margin; centerline-width estimate"
        value = path_margin - required_path_margin
    else:
        label, evidence, value = classify_first_event(
            timeline, thresholds, steering_limit_rad, steering_rate_limit_radps
        )
    summary = {
        "candidate": run_dir.name,
        "bag": str(bag.relative_to(ROOT)),
        "trajectory": None if trajectory is None else str(trajectory.relative_to(ROOT) if trajectory.is_relative_to(ROOT) else trajectory),
        "estimated_lap_s": estimate,
        "candidate_path_min_clearance_m": path_margin,
        "candidate_path_min_clearance_s_m": path_margin_s,
        "configured_extra_wall_clearance_m": required_path_margin,
        "collision_initial": collision_initial,
        "collision_final": collision_final,
        "collision_delta": collision_final - collision_initial,
        "first_collision_receipt_ns": event_ns,
        "pre_collision_coverage_s": available,
        "required_pre_collision_s": timeline_s,
        "timeline_complete": event_ns is not None and available >= timeline_s,
        "classification": label,
        "classification_evidence": evidence,
        "classification_value": value,
        "source_order": order,
        "timeline_samples": len(timeline),
        "timeline_start_to_collision_s": available,
        "thresholds_from_p0_p99": thresholds,
    }
    return summary, timeline


def write_timeline(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flattened: list[dict[str, Any]] = []
    for row in rows:
        out = {key: value for key, value in row.items() if key not in ("odom", "mpc")}
        out.update({f"odom_{key}": value for key, value in row["odom"].items()})
        out.update(row["mpc"])
        flattened.append(out)
    columns = list(dict.fromkeys(key for row in flattened for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(flattened)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=ROOT / "live_runs")
    parser.add_argument("--run-glob", action="append", default=[], help="run-directory glob, repeatable")
    parser.add_argument("--safe-run", action="append", type=Path, default=[], help="additional safe P0/fast run to include")
    parser.add_argument("--baseline-report", action="append", type=Path, default=[])
    parser.add_argument("--baseline-bag", action="append", type=Path, default=[])
    parser.add_argument("--centerline", type=Path, default=ROOT / DEFAULT_CENTERLINE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeline-s", type=float, default=2.0)
    parser.add_argument("--steering-limit-rad", type=float, default=0.5235987756)
    parser.add_argument("--steering-rate-limit-radps", type=float, default=3.2)
    args = parser.parse_args()
    if args.timeline_s <= 0 or args.steering_limit_rad <= 0 or args.steering_rate_limit_radps <= 0:
        parser.error("timeline and actuator limits must be positive")
    if not args.centerline.is_file():
        parser.error(f"centerline not found: {args.centerline}")
    centerline = TrackProjection.from_csv(args.centerline, closed=True)
    thresholds = load_baseline_thresholds(args.baseline_report, args.baseline_bag)
    if args.baseline_report and not thresholds:
        parser.error("baseline reports supplied but no usable residual thresholds were found")

    run_dirs = set(args.safe_run)
    for pattern in args.run_glob:
        run_dirs.update(path.parent.parent for path in args.runs_root.glob(pattern + "/run/run_0.db3"))
    if not run_dirs:
        parser.error("provide at least one --run-glob or --safe-run")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    for run_dir in sorted(run_dirs):
        if not (run_dir / "run" / "run_0.db3").is_file():
            print(f"skip {run_dir}: no run/run_0.db3", file=sys.stderr)
            continue
        try:
            summary, timeline = analyze_run(
                run_dir, centerline, thresholds, args.timeline_s,
                args.steering_limit_rad, args.steering_rate_limit_radps,
            )
        except (OSError, sqlite3.Error, ValueError, RuntimeError) as exc:
            summary = {
                "candidate": run_dir.name, "classification": "UNKNOWN",
                "classification_evidence": str(exc), "timeline_complete": False,
            }
            timeline = []
        timeline_path = args.output_dir / "timelines" / f"{run_dir.name}.csv"
        write_timeline(timeline_path, timeline)
        summary["timeline_csv"] = str(timeline_path.relative_to(ROOT) if timeline_path.is_relative_to(ROOT) else timeline_path)
        summaries.append(summary)
        print(
            f"{run_dir.name}: {summary.get('classification')} "
            f"pre={summary.get('pre_collision_coverage_s')}s "
            f"complete={summary.get('timeline_complete')}", flush=True,
        )

    summaries.sort(key=lambda row: row.get("candidate", ""))
    (args.output_dir / "failure_atlas.json").write_text(
        json.dumps(summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "failure_atlas.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = [
            "candidate", "estimated_lap_s", "classification", "classification_evidence",
            "classification_value", "pre_collision_coverage_s", "timeline_complete",
            "collision_delta", "timeline_samples", "bag", "trajectory", "timeline_csv",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)
    print(f"atlas: {args.output_dir / 'failure_atlas.csv'}")
    return 0 if all(row.get("timeline_complete", False) or row.get("collision_delta", 0) == 0 for row in summaries) else 3


if __name__ == "__main__":
    raise SystemExit(main())
