#!/usr/bin/env python3
"""Compare the production first-order yaw law with a train-only gain taper.

This is read-only diagnostic analysis. Its conditional open-loop rollout is not
a closed-loop MPC prediction and does not validate candidates for screening.
Failed bags are truncated at their first collision; post-impact samples are
never fitted or scored.
"""

from __future__ import annotations

import argparse
import bisect
import math
import sqlite3
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
except ImportError as exc:  # pragma: no cover - depends on ROS installation
    raise SystemExit(
        "ROS 2 Python modules are required (rclpy and rosidl_runtime_py): "
        f"{exc}"
    ) from exc


ODOM = "/autodrive/roboracer_1/odom"
STEERING = "/autodrive/roboracer_1/steering"
LIDAR = "/autodrive/roboracer_1/lidar"
COLLISIONS = "/autodrive/roboracer_1/collision_count"
TAU_S = 0.015
COM_X_M = 0.15532
NOMINAL_GAIN = 2.95
BASELINE_TAPER_SLOPE = 35.6
BASELINE_TAPER_START = 0.41
STEERING_LIMIT_RAD = 0.5235987756
ALIGNMENT_LIMIT_NS = 30_000_000
MIN_GAIN = 0.05
SPEED_BINS = ((0.0, 2.0), (2.0, 4.0), (4.0, 6.0), (6.0, 8.0), (8.0, math.inf))
DELTA_BINS = ((0.0, 0.10), (0.10, 0.20), (0.20, 0.30), (0.30, 0.35),
              (0.35, 0.40), (0.40, 0.45), (0.45, math.inf))


@dataclass(frozen=True)
class Row:
    receipt_ns: int
    source_ns: int
    speed_mps: float
    yaw_rate_rps: float
    pose_yaw_rad: float


@dataclass(frozen=True)
class AlignedRow:
    odom: Row
    steering_rad: float
    steering_receipt_ns: int
    offset_ms: float


@dataclass(frozen=True)
class Step:
    dt_s: float
    yaw_rate_start: float
    yaw_rate_end: float
    speed_mid: float
    steering_end: float
    speed_bin_value: float


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _stamp_ns(stamp: Any) -> int | None:
    value = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
    return value if value > 0 else None


def _pose_yaw(quaternion: Any) -> float:
    x, y, z, w = (float(quaternion.x), float(quaternion.y),
                  float(quaternion.z), float(quaternion.w))
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _decode(connection: sqlite3.Connection, topic_id: int, msg_type: str):
    cls = get_message(msg_type)
    for timestamp, payload in connection.execute(
        "SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp, id",
        (topic_id,),
    ):
        yield int(timestamp), deserialize_message(bytes(payload), cls)


def _topic_map(connection: sqlite3.Connection) -> dict[str, tuple[int, str]]:
    return {
        name: (int(topic_id), msg_type)
        for topic_id, name, msg_type in connection.execute(
            "SELECT id, name, type FROM topics"
        )
    }


def _gap_report(receipts_ns: list[int], headers_ns: list[int] | None = None,
                header_receipts_ns: list[int] | None = None) -> str:
    def describe(values: list[int]) -> str:
        gaps_ms = [(b - a) / 1e6 for a, b in zip(values, values[1:])]
        positive = [gap for gap in gaps_ms if gap > 0]
        span = (values[-1] - values[0]) / 1e9 if len(values) > 1 else 0.0
        rate = (len(values) - 1) / span if span > 0 else None
        return (
            f"n={len(values)}, rate={_fmt(rate)}Hz, "
            f"gaps_ms[p50/p95/p99/max]={_fmt(_percentile(positive, 50))}/"
            f"{_fmt(_percentile(positive, 95))}/{_fmt(_percentile(positive, 99))}/"
            f"{_fmt(max(positive) if positive else None)}, "
            f">30={sum(g > 30 for g in positive)}, >35={sum(g > 35 for g in positive)}, "
            f"duplicate={sum(g == 0 for g in gaps_ms)}, nonmonotonic={sum(g < 0 for g in gaps_ms)}"
        )

    output = "receipt: " + describe(receipts_ns)
    if headers_ns is not None:
        output += " | source header: " + describe(headers_ns)
        offset_receipts = header_receipts_ns if header_receipts_ns is not None else receipts_ns
        offsets_ms = [(header - receipt) / 1e6
                      for receipt, header in zip(offset_receipts, headers_ns)]
        output += (
            " | header-minus-receipt_ms[p50/p95/min/max]="
            f"{_fmt(_percentile(offsets_ms, 50))}/{_fmt(_percentile(offsets_ms, 95))}/"
            f"{_fmt(min(offsets_ms) if offsets_ms else None)}/"
            f"{_fmt(max(offsets_ms) if offsets_ms else None)}"
        )
    return output


def load_bag(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"bag does not exist: {path}")
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        topics = _topic_map(connection)
        required = (ODOM, STEERING, COLLISIONS)
        missing = [topic for topic in required if topic not in topics]
        if missing:
            raise ValueError(f"{path}: missing required topic(s): {', '.join(missing)}")

        start_ns_value = connection.execute("SELECT MIN(timestamp) FROM messages").fetchone()[0]
        if start_ns_value is None:
            raise ValueError(f"{path}: bag has no messages")
        bag_start_ns = int(start_ns_value)

        collision_id, collision_type = topics[COLLISIONS]
        collision_events = list(_decode(connection, collision_id, collision_type))
        if not collision_events:
            raise ValueError(f"{path}: no collision-count messages")
        initial_collision = int(collision_events[0][1].data)
        impact_ns: int | None = bag_start_ns if initial_collision > 0 else None
        final_collision = initial_collision
        for timestamp, message in collision_events:
            final_collision = int(message.data)
            if impact_ns is None and final_collision > initial_collision:
                impact_ns = timestamp

        odom_id, odom_type = topics[ODOM]
        odom_raw: list[Row] = []
        odom_receipts: list[int] = []
        odom_headers: list[int] = []
        for receipt_ns, message in _decode(connection, odom_id, odom_type):
            if impact_ns is not None and receipt_ns >= impact_ns:
                continue
            source_ns = _stamp_ns(message.header.stamp)
            vx_com = float(message.twist.twist.linear.x)
            vy_com = float(message.twist.twist.linear.y)
            yaw_rate = float(message.twist.twist.angular.z)
            vx = vx_com
            vy = vy_com - yaw_rate * COM_X_M
            pose_yaw = _pose_yaw(message.pose.pose.orientation)
            if source_ns is None or not all(map(math.isfinite, (vx, vy, yaw_rate, pose_yaw))):
                continue
            row = Row(receipt_ns, source_ns, math.hypot(vx, vy), yaw_rate, pose_yaw)
            odom_raw.append(row)
            odom_receipts.append(receipt_ns)
            odom_headers.append(source_ns)

        steering_id, steering_type = topics[STEERING]
        steering_data: list[tuple[int, float]] = []
        steering_receipts: list[int] = []
        for receipt_ns, message in _decode(connection, steering_id, steering_type):
            if impact_ns is not None and receipt_ns >= impact_ns:
                continue
            value = float(message.data)
            if math.isfinite(value):
                steering_data.append((receipt_ns, value))
                steering_receipts.append(receipt_ns)

        lidar_receipts: list[int] = []
        lidar_headers: list[int] = []
        lidar_header_receipts: list[int] = []
        if LIDAR in topics:
            lidar_id, lidar_type = topics[LIDAR]
            for receipt_ns, message in _decode(connection, lidar_id, lidar_type):
                if impact_ns is not None and receipt_ns >= impact_ns:
                    continue
                header_ns = _stamp_ns(message.header.stamp)
                lidar_receipts.append(receipt_ns)
                if header_ns is not None:
                    lidar_headers.append(header_ns)
                    lidar_header_receipts.append(receipt_ns)

        steering_times = [timestamp for timestamp, _ in steering_data]
        aligned: list[AlignedRow | None] = []
        matched_offsets: list[float] = []
        for odom in odom_raw:
            index = bisect.bisect_left(steering_times, odom.source_ns)
            candidates = [i for i in (index - 1, index) if 0 <= i < len(steering_data)]
            if not candidates:
                aligned.append(None)
                continue
            nearest = min(candidates, key=lambda i: abs(steering_data[i][0] - odom.source_ns))
            steering_receipt, steering_value = steering_data[nearest]
            offset_ms = (steering_receipt - odom.source_ns) / 1e6
            if abs(offset_ms) > 30.0:
                aligned.append(None)
                continue
            aligned.append(AlignedRow(odom, steering_value, steering_receipt, offset_ms))
            matched_offsets.append(offset_ms)

        steps: list[Step | None] = []
        for index in range(len(aligned) - 1):
            current, following = aligned[index], aligned[index + 1]
            if current is None or following is None:
                steps.append(None)
                continue
            dt_s = (following.odom.source_ns - current.odom.source_ns) / 1e9
            if not 0.0 < dt_s <= 0.10:
                steps.append(None)
                continue
            steps.append(Step(
                dt_s=dt_s,
                yaw_rate_start=current.odom.yaw_rate_rps,
                yaw_rate_end=following.odom.yaw_rate_rps,
                speed_mid=0.5 * (current.odom.speed_mps + following.odom.speed_mps),
                steering_end=following.steering_rad,
                speed_bin_value=0.5 * (current.odom.speed_mps + following.odom.speed_mps),
            ))

        return {
            "path": path,
            "bag_start_ns": bag_start_ns,
            "impact_ns": impact_ns,
            "initial_collision": initial_collision,
            "final_collision": final_collision,
            "odom": odom_raw,
            "aligned": aligned,
            "steps": steps,
            "odom_receipts": odom_receipts,
            "odom_headers": odom_headers,
            "steering_receipts": steering_receipts,
            "lidar_receipts": lidar_receipts,
            "lidar_headers": lidar_headers,
            "lidar_header_receipts": lidar_header_receipts,
            "matched_offsets_ms": matched_offsets,
        }
    finally:
        connection.close()


def _tapered_gain(delta: float, onset: float, slope: float) -> float:
    return NOMINAL_GAIN - slope * max(0.0, abs(delta) - onset)


def _predict(step: Step, gain: float) -> float:
    retention = math.exp(-step.dt_s / TAU_S)
    steady_yaw = gain * step.speed_mid * math.tan(step.steering_end)
    return retention * step.yaw_rate_start + (1.0 - retention) * steady_yaw


def _model_error(steps: list[Step], onset: float | None = None,
                 slope: float | None = None) -> dict[str, float | int | None]:
    errors: list[float] = []
    for step in steps:
        if onset is None or slope is None:
            gain = NOMINAL_GAIN - BASELINE_TAPER_SLOPE * min(
                max(0.0, abs(step.steering_end) - BASELINE_TAPER_START),
                0.05,
            )
        else:
            gain = _tapered_gain(step.steering_end, onset, slope)
        errors.append(_predict(step, gain) - step.yaw_rate_end)
    absolute = [abs(error) for error in errors]
    return {
        "n": len(errors),
        "bias": statistics.fmean(errors) if errors else None,
        "rmse": math.sqrt(statistics.fmean(error * error for error in errors)) if errors else None,
        "p95_abs": _percentile(absolute, 95),
    }


def _fit_taper(steps: list[Step]) -> tuple[float, float, float]:
    if len(steps) < 10:
        raise ValueError(f"training run has too few aligned one-step samples ({len(steps)})")
    best: tuple[float, float, float] | None = None
    onset_min = 0.30  # Preserve the full safe-practice steering range in the existing law.
    onset_steps = 64
    for onset_index in range(onset_steps + 1):
        onset = onset_min + onset_index * 0.0025
        bases: list[float] = []
        slopes: list[float] = []
        targets: list[float] = []
        for step in steps:
            retention = math.exp(-step.dt_s / TAU_S)
            x = max(0.0, abs(step.steering_end) - onset)
            control = step.speed_mid * math.tan(step.steering_end)
            base = retention * step.yaw_rate_start + (1.0 - retention) * NOMINAL_GAIN * control
            slope_coefficient = -(1.0 - retention) * x * control
            bases.append(base)
            slopes.append(slope_coefficient)
            targets.append(step.yaw_rate_end)
        denominator = sum(value * value for value in slopes)
        if denominator <= 1e-12:
            slope = 0.0
        else:
            slope = -sum(c * (base - target)
                         for c, base, target in zip(slopes, bases, targets)) / denominator
        max_positive_slope = (NOMINAL_GAIN - MIN_GAIN) / max(
            1e-9, STEERING_LIMIT_RAD - onset)
        slope = max(0.0, min(max_positive_slope, slope))
        mse = statistics.fmean(
            (base + coefficient * slope - target) ** 2
            for base, coefficient, target in zip(bases, slopes, targets)
        )
        if best is None or mse < best[0]:
            best = (mse, onset, slope)
    assert best is not None
    fitted_onset, fitted_slope = best[1], best[2]
    min_gain = _tapered_gain(STEERING_LIMIT_RAD, fitted_onset, fitted_slope)
    if min_gain <= 0.0:
        raise ValueError("fit produced non-positive steering gain at physical steering limit")
    return fitted_onset, fitted_slope, min_gain


def _valid_steps(data: dict[str, Any]) -> list[Step]:
    return [step for step in data["steps"] if step is not None]


def _five_step_error(data: dict[str, Any], onset: float | None = None,
                     slope: float | None = None) -> dict[str, float | int | None]:
    rows: list[AlignedRow | None] = data["aligned"]
    errors: list[float] = []
    for start in range(len(rows) - 5):
        window = rows[start:start + 6]
        if any(row is None for row in window):
            continue
        predicted = window[0].odom.yaw_rate_rps
        valid = True
        for index in range(5):
            current = window[index]
            following = window[index + 1]
            dt = (following.odom.source_ns - current.odom.source_ns) / 1e9
            if not 0.0 < dt <= 0.10:
                valid = False
                break
            step = Step(dt, predicted, following.odom.yaw_rate_rps,
                        0.5 * (current.odom.speed_mps + following.odom.speed_mps),
                        following.steering_rad,
                        0.5 * (current.odom.speed_mps + following.odom.speed_mps))
            if onset is None or slope is None:
                gain = NOMINAL_GAIN - BASELINE_TAPER_SLOPE * min(
                    max(0.0, abs(step.steering_end) - BASELINE_TAPER_START), 0.05)
            else:
                gain = _tapered_gain(step.steering_end, onset, slope)
            predicted = _predict(step, gain)
        if valid:
            errors.append(predicted - window[-1].odom.yaw_rate_rps)
    absolute = [abs(error) for error in errors]
    return {
        "n": len(errors),
        "bias": statistics.fmean(errors) if errors else None,
        "rmse": math.sqrt(statistics.fmean(error * error for error in errors)) if errors else None,
        "p95_abs": _percentile(absolute, 95),
    }


def _pose_yaw_consistency(rows: list[Row]) -> dict[str, float | int | None]:
    errors: list[float] = []
    for current, following in zip(rows, rows[1:]):
        dt = (following.source_ns - current.source_ns) / 1e9
        if not 0.0 < dt <= 0.10:
            continue
        delta_yaw = math.atan2(
            math.sin(following.pose_yaw_rad - current.pose_yaw_rad),
            math.cos(following.pose_yaw_rad - current.pose_yaw_rad),
        )
        pose_yaw_rate = delta_yaw / dt
        body_yaw_rate = 0.5 * (current.yaw_rate_rps + following.yaw_rate_rps)
        errors.append(pose_yaw_rate - body_yaw_rate)
    absolute = [abs(error) for error in errors]
    return {
        "n": len(errors),
        "bias": statistics.fmean(errors) if errors else None,
        "rmse": math.sqrt(statistics.fmean(error * error for error in errors)) if errors else None,
        "p95_abs": _percentile(absolute, 95),
        "p99_abs": _percentile(absolute, 99),
        "over_half": sum(error > 0.5 for error in absolute),
    }


def _bin_label(bounds: tuple[float, float], suffix: str) -> str:
    high = "inf" if math.isinf(bounds[1]) else f"{bounds[1]:.2f}"
    return f"[{bounds[0]:.2f},{high}) {suffix}"


def _print_dataset(data: dict[str, Any], onset: float, slope: float) -> None:
    path = data["path"]
    odom = data["odom"]
    steps = _valid_steps(data)
    impact = "none"
    if data["impact_ns"] is not None:
        if data["initial_collision"] > 0:
            impact = f"initial collision count={data['initial_collision']} at bag start"
        else:
            impact = f"{(data['impact_ns'] - data['bag_start_ns']) / 1e9:.3f}s from bag start"
    print(f"\nRun: {path}")
    print(f"  collision count: {data['initial_collision']} -> {data['final_collision']}; cutoff: {impact}")
    print(f"  source odom rows: {len(odom)}; steering-aligned: "
          f"{sum(row is not None for row in data['aligned'])}/{len(data['aligned'])} "
          f"({100.0 * sum(row is not None for row in data['aligned']) / max(1, len(data['aligned'])):.1f}%) "
          f"within 30 ms; valid one-step pairs: {len(steps)}")
    offsets = data["matched_offsets_ms"]
    abs_offsets = [abs(value) for value in offsets]
    print("  steering alignment offset, receipt minus odom header ms "
          f"p50/p95/abs-p95/abs-max={_fmt(_percentile(offsets, 50))}/"
          f"{_fmt(_percentile(offsets, 95))}/{_fmt(_percentile(abs_offsets, 95))}/"
          f"{_fmt(max(abs_offsets) if abs_offsets else None)}; "
          f"unmatched or >30 ms={len(data['aligned']) - len(offsets)}")
    if odom:
        print(f"  source speed range: {min(row.speed_mps for row in odom):.3f}.."
              f"{max(row.speed_mps for row in odom):.3f} m/s; "
              f"max |steering|={max((abs(row.steering_rad) for row in data['aligned'] if row), default=0.0):.3f} rad")
    print(f"  source odom timing: {_gap_report(data['odom_receipts'], data['odom_headers'])}")
    consistency = _pose_yaw_consistency(odom)
    print("  pose-heading derivative minus body angular.z: "
          f"n={consistency['n']}, bias={_fmt(consistency['bias'], 4)}, "
          f"RMSE={_fmt(consistency['rmse'], 4)}, p95/p99 |error|="
          f"{_fmt(consistency['p95_abs'], 4)}/{_fmt(consistency['p99_abs'], 4)} rad/s, "
          f"|error|>0.5={consistency['over_half']}")
    print(f"  measured steering timing: {_gap_report(data['steering_receipts'])} (no message header)")
    if data["lidar_receipts"]:
        print("  LiDAR timing: " + _gap_report(
            data["lidar_receipts"], data["lidar_headers"], data["lidar_header_receipts"]))
    else:
        print("  LiDAR timing: topic absent or no pre-impact messages")

    print("  valid one-step sample coverage by midpoint speed:")
    for bounds in SPEED_BINS:
        n = sum(bounds[0] <= step.speed_bin_value < bounds[1] for step in steps)
        print(f"    {_bin_label(bounds, 'm/s')}: {n}")
    print("  valid one-step sample coverage by |measured steering|:")
    for bounds in DELTA_BINS:
        n = sum(bounds[0] <= abs(step.steering_end) < bounds[1] for step in steps)
        print(f"    {_bin_label(bounds, 'rad')}: {n}")

    for label, fit in (("current law", _model_error(steps)),
                       ("fitted taper", _model_error(steps, onset, slope))):
        print(f"  one-step {label}: n={fit['n']}, bias={_fmt(fit['bias'], 5)} rad/s, "
              f"RMSE={_fmt(fit['rmse'], 5)}, p95 |error|={_fmt(fit['p95_abs'], 5)}")
    print("  one-step errors by |steering| bin (n, current/fitted RMSE and p95 |error|; rad/s):")
    for bounds in DELTA_BINS:
        group = [step for step in steps if bounds[0] <= abs(step.steering_end) < bounds[1]]
        base = _model_error(group)
        fit = _model_error(group, onset, slope)
        print(f"    {_bin_label(bounds, 'rad')}: {base['n']}, "
              f"RMSE={_fmt(base['rmse'], 4)}/{_fmt(fit['rmse'], 4)}, "
              f"p95={_fmt(base['p95_abs'], 4)}/{_fmt(fit['p95_abs'], 4)}")
    print("  one-step errors by midpoint speed bin (n, current/fitted RMSE and p95 |error|; rad/s):")
    for bounds in SPEED_BINS:
        group = [step for step in steps if bounds[0] <= step.speed_bin_value < bounds[1]]
        base = _model_error(group)
        fit = _model_error(group, onset, slope)
        print(f"    {_bin_label(bounds, 'm/s')}: {base['n']}, "
              f"RMSE={_fmt(base['rmse'], 4)}/{_fmt(fit['rmse'], 4)}, "
              f"p95={_fmt(base['p95_abs'], 4)}/{_fmt(fit['p95_abs'], 4)}")
    print("  high-angle samples at 6-8 m/s:")
    for bounds in (DELTA_BINS[3], DELTA_BINS[5]):
        group = [step for step in steps
                 if bounds[0] <= abs(step.steering_end) < bounds[1]
                 and 6.0 <= step.speed_bin_value < 8.0]
        base = _model_error(group)
        fit = _model_error(group, onset, slope)
        print(f"    {_bin_label(bounds, 'rad')}: n={base['n']}, "
              f"RMSE={_fmt(base['rmse'], 4)}/{_fmt(fit['rmse'], 4)}, "
              f"p95={_fmt(base['p95_abs'], 4)}/{_fmt(fit['p95_abs'], 4)} rad/s")
    for label, fit in (("current law", _five_step_error(data)),
                       ("fitted taper", _five_step_error(data, onset, slope))):
        print(f"  5-step conditional open-loop {label}: n={fit['n']}, "
              f"bias={_fmt(fit['bias'], 5)} rad/s, RMSE={_fmt(fit['rmse'], 5)}, "
              f"p95 |error|={_fmt(fit['p95_abs'], 5)}; uses measured future speed/steering")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare yaw response laws on separate read-only simulator bags.")
    parser.add_argument("train", type=Path, help="training .db3 bag used only to fit taper")
    parser.add_argument("held_out", type=Path, help="fully separate held-out .db3 bag")
    parser.add_argument("--practice-holdout", action="append", type=Path, default=[],
                        help="additional practice bag; may be repeated")
    args = parser.parse_args()

    try:
        input_paths = [args.train, args.held_out, *args.practice_holdout]
        for index, path in enumerate(input_paths):
            for earlier in input_paths[:index]:
                same_path = path.resolve() == earlier.resolve()
                if not same_path:
                    try:
                        same_path = path.samefile(earlier)
                    except OSError:
                        same_path = False
                if same_path:
                    raise ValueError(
                        "each training, held-out, and practice hold-out input must be a "
                        f"separate bag; duplicate paths: {earlier} and {path}"
                    )
        training = load_bag(args.train)
        held_out = load_bag(args.held_out)
        practice = [load_bag(path) for path in args.practice_holdout]
        train_steps = _valid_steps(training)
        onset, slope, min_gain = _fit_taper(train_steps)
    except (OSError, sqlite3.Error, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print("Yaw-response comparison (diagnostic only; fit is inadequate for raceline candidate screening)")
    print(f"Fixed tau={TAU_S:.3f}s; current gain=2.95 - 35.6*clamp(|delta|-0.41, 0, 0.05)")
    print("Fit uses only training-run one-step samples; gain is nominal minus a linear angle taper.")
    print(f"Fitted taper: onset={onset:.4f} rad, slope={slope:.3f} per rad, "
          f"gain at physical limit {STEERING_LIMIT_RAD:.4f} rad={min_gain:.4f} 1/m (positive)")
    print("Rows at/after first collision are excluded; an initially positive collision count excludes the full bag.")
    _print_dataset(training, onset, slope)
    _print_dataset(held_out, onset, slope)
    for data in practice:
        _print_dataset(data, onset, slope)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
