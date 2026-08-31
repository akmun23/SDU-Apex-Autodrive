#!/usr/bin/env python3
"""Summarize allowed-sensor calibration data and derive a feedforward table.

The input is produced by the single ``calibration`` node.  This tool is
offline-only: it reads recorded CSV data and never subscribes to simulator
state or publishes a command.
"""

import argparse
import csv
import math
from pathlib import Path
import re
import statistics


DOCUMENTED_WHEEL_RADIUS_M = 0.0590


def finite(value: str) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def row_stamp(row: dict[str, str]) -> float | None:
    """Return a shared absolute timestamp, with legacy elapsed-time fallback."""
    stamp = finite(row.get("stamp_s"))
    if stamp is not None:
        return stamp
    return finite(row.get("time_s"))


def valid_ground_truth_row(row: dict[str, str]) -> bool:
    """Reject samples after the open-ground vehicle leaves the world plane."""
    z = finite(row.get("gt_z_m"))
    return z is None or 0.0 <= z <= 0.20


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def trim_after_encoder_reset(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], float | None, str | None]:
    """Handle a simulator/encoder reset without mixing run segments.

    A reset at the beginning of a fresh recording is an expected consequence
    of starting the simulator from a previous run.  In that case discard the
    stale prefix and keep the new run.  A reset after the vehicle has already
    moved is a terminal diagnostic reset, so keep the pre-failure run instead.
    A normal forward encoder sample is never removed; only a large negative
    jump on both counters marks a reset.
    """
    previous_left = previous_right = None
    for index, row in enumerate(rows):
        left = finite(row.get("left_encoder_rad"))
        right = finite(row.get("right_encoder_rad"))
        if left is None or right is None:
            continue
        if previous_left is not None and previous_right is not None:
            if left - previous_left < -10.0 and right - previous_right < -10.0:
                reset_time = finite(row.get("time_s"))
                # The recorder is normally started immediately after the
                # diagnostic reset command.  Preserve the post-reset segment
                # in that case; otherwise retain the useful pre-failure run.
                early_reset = reset_time is not None and reset_time <= 5.0
                if early_reset:
                    return rows[index:], reset_time, "initial"
                return rows[:index], reset_time, "terminal"
        previous_left = left
        previous_right = right
    return rows, None, None


def read_trajectory(path: Path) -> list[tuple[float, float, float]]:
    """Read the project's commented-header raceline CSV."""
    with path.open(encoding="utf-8") as stream:
        lines = [line.strip() for line in stream if line.strip()]
    if not lines:
        return []
    header = lines[0].lstrip("# ").split(",")
    points = []
    for row in csv.DictReader([",".join(header), *lines[1:]]):
        try:
            x = float(row["x_m"])
            y = float(row["y_m"])
            arc = float(row["s_m"])
        except (KeyError, TypeError, ValueError):
            continue
        if all(math.isfinite(value) for value in (x, y, arc)):
            points.append((x, y, arc))
    return points


def encoder_distance_metrics(
    rows: list[dict[str, str]],
    ground_truth_path: Path,
    max_encoder_step_rad: float = 40.0,
) -> tuple[float, float, float, int, int] | None:
    """Estimate encoder scale from a diagnostics-only ground-truth run.

    The ground-truth file is produced by ``ground_truth_amcl_monitor`` and is
    never consumed by runtime localization or control. New recordings are
    restricted to their overlapping absolute ROS timestamp window before
    distance is accumulated. This prevents a recorder prefix or post-collision
    stationary tail from contaminating the fit. Older files without
    ``stamp_s`` retain the legacy whole-file fallback. Stationary samples and
    negative/reset jumps are ignored.
    """
    if max_encoder_step_rad <= 0.0:
        return None
    gt_rows = read_rows(ground_truth_path)
    telemetry_stamps = [row_stamp(row) for row in rows]
    telemetry_stamps = [value for value in telemetry_stamps if value is not None]
    gt_stamps = [row_stamp(row) for row in gt_rows]
    gt_stamps = [value for value in gt_stamps if value is not None]
    shared_window = None
    if telemetry_stamps and gt_stamps:
        start = max(min(telemetry_stamps), min(gt_stamps))
        end = min(max(telemetry_stamps), max(gt_stamps))
        if end > start + 1.0:
            shared_window = (start, end)

    def in_window(row: dict[str, str]) -> bool:
        if shared_window is None:
            return True
        stamp = row_stamp(row)
        return stamp is not None and shared_window[0] <= stamp <= shared_window[1]

    gt_rows = [row for row in gt_rows if in_window(row)]
    gt_distance = 0.0
    for before, after in zip(gt_rows, gt_rows[1:]):
        bx = finite(before.get("gt_x_m"))
        by = finite(before.get("gt_y_m"))
        ax = finite(after.get("gt_x_m"))
        ay = finite(after.get("gt_y_m"))
        if None not in (bx, by, ax, ay):
            step = math.hypot(ax - bx, ay - by)
            # Ground-truth rows are normally about 10 Hz. A legal 3.5 m/s
            # vehicle can move roughly 0.35 m per row; allow modest delivery
            # jitter but reject the simulator's ~0.88 m collision/reset jump.
            if 0.0 < step < 0.60:
                gt_distance += step

    encoder_distance = 0.0
    encoder_angle = 0.0
    valid_steps = 0
    rejected_steps = 0
    previous_left = previous_right = None
    for row in rows:
        if not in_window(row):
            continue
        left = finite(row.get("left_encoder_rad"))
        right = finite(row.get("right_encoder_rad"))
        if left is None or right is None:
            continue
        if previous_left is not None and previous_right is not None:
            dleft = left - previous_left
            dright = right - previous_right
            # At 3.5 m/s and the native 10 Hz cadence, a legitimate sample
            # can contain several wheel revolutions when DDS delivers a
            # delayed pair. The old 10 rad limit silently discarded those
            # samples and biased the fitted wheel radius. Reset/rollover
            # handling remains separate and still rejects negative jumps.
            if (dleft >= 0.0 and dright >= 0.0 and
                    dleft <= max_encoder_step_rad and dright <= max_encoder_step_rad):
                encoder_angle += 0.5 * (dleft + dright)
                valid_steps += 1
            elif dleft >= 0.0 and dright >= 0.0:
                rejected_steps += 1
        previous_left = left
        previous_right = right

    if gt_distance <= 1.0 or encoder_angle <= 1.0 or valid_steps == 0:
        return None
    nominal_distance = DOCUMENTED_WHEEL_RADIUS_M * encoder_angle
    scale = gt_distance / nominal_distance
    return gt_distance, nominal_distance, scale, valid_steps, rejected_steps


def raceline_metrics(
    rows: list[dict[str, str]], trajectory: list[tuple[float, float, float]]
) -> tuple[list[float], int]:
    """Return EKF-to-raceline distances and forward seam crossings."""
    if not trajectory:
        return [], 0
    distances = []
    nearest_indices = []
    for row in rows:
        x = finite(row.get("x_ekf_m"))
        y = finite(row.get("y_ekf_m"))
        if x is None or y is None:
            continue
        index, squared = min(
            enumerate((x - px) ** 2 + (y - py) ** 2
                      for px, py, _ in trajectory),
            key=lambda item: item[1],
        )
        nearest_indices.append(index)
        distances.append(math.sqrt(squared))

    seam_crossings = 0
    seam_start = int(0.8 * len(trajectory))
    seam_end = int(0.2 * len(trajectory))
    for before, after in zip(nearest_indices, nearest_indices[1:]):
        if before >= seam_start and after <= seam_end:
            seam_crossings += 1
    return distances, seam_crossings


def stable_tail(values: list[float], fraction: float = 0.4) -> list[float]:
    if not values:
        return []
    start = max(0, int(len(values) * (1.0 - fraction)))
    return values[start:]


def preferred_speed_field(rows: list[dict[str, str]]) -> str:
    """Prefer simulator truth when it is present, otherwise use local odom."""
    truth_samples = sum(
        valid_ground_truth_row(row) and finite(row.get("gt_speed_mps")) is not None
        for row in rows)
    return "gt_speed_mps" if truth_samples >= 10 else "speed_mps"


def phase_is_response(phase: str) -> bool:
    return (
        phase.startswith("throttle_") or
        phase.startswith("grid_throttle_") or
        phase.startswith("speed_") or
        phase in {"full_throttle_accelerate", "zero_throttle_decel"}
    )


def phase_command(phase: str) -> float | None:
    match = re.search(r"(?:throttle|grid_throttle|speed)_(-?\d+(?:\.\d+)?)", phase)
    if match:
        return finite(match.group(1))
    if phase == "full_throttle_accelerate":
        return 1.0
    if phase == "zero_throttle_decel":
        return 0.0
    return None


def phase_response_metrics(
    rows: list[dict[str, str]], speed_field: str
) -> list[tuple[str, float, float, float, float, float, int]]:
    """Summarize every commanded step using simulator speed ground truth.

    The returned columns are phase, command, initial speed, peak speed, tail
    speed, peak positive acceleration, and sample count. Acceleration is
    calculated from the timestamped truth velocity/speed trace, not from the
    controller's estimate.
    """
    groups: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        if not valid_ground_truth_row(row):
            continue
        phase = row.get("phase", "")
        interesting = phase_is_response(phase)
        if not interesting:
            continue
        stamp = row_stamp(row)
        speed = finite(row.get(speed_field))
        if stamp is not None and speed is not None and speed >= 0.0:
            groups.setdefault(phase, []).append((stamp, speed))

    result = []
    for phase, samples in groups.items():
        samples.sort()
        values = [value for _, value in samples]
        initial = statistics.median(values[:max(1, len(values) // 5)])
        tail = stable_tail(values)
        tail_speed = statistics.median(tail) if tail else values[-1]
        peak = max(values)
        accelerations = []
        # CSV rows are sampled at 50 Hz while truth is refreshed at about
        # 10 Hz, so adjacent rows usually repeat the same truth value. Use a
        # short timestamped lag instead of differentiating those repeats.
        for after_index, (after_t, after_v) in enumerate(samples):
            before_index = after_index - 1
            while before_index >= 0 and after_t - samples[before_index][0] < 0.08:
                before_index -= 1
            if before_index < 0:
                continue
            before_t, before_v = samples[before_index]
            dt = after_t - before_t
            if dt <= 0.50:
                acceleration = (after_v - before_v) / dt
                if math.isfinite(acceleration) and -50.0 <= acceleration <= 50.0:
                    accelerations.append(acceleration)
        peak_accel = max(accelerations, default=math.nan)
        command = phase_command(phase)
        if command is None:
            command = 0.0
        result.append((phase, command, initial, peak, tail_speed, peak_accel, len(values)))
    return sorted(result, key=lambda item: (item[0], item[1]))


def response_trace(
    rows: list[dict[str, str]], speed_field: str
) -> list[dict[str, float | str]]:
    """Build a truth-based throttle/speed/acceleration response trace."""
    groups: dict[str, list[tuple[float, float, float]]] = {}
    for row in rows:
        if not valid_ground_truth_row(row):
            continue
        phase = row.get("phase", "")
        interesting = phase_is_response(phase)
        stamp = row_stamp(row)
        speed = finite(row.get(speed_field))
        if not interesting or stamp is None or speed is None or speed < 0.0:
            continue
        if phase.startswith("throttle_") or phase.startswith("grid_throttle_"):
            throttle = phase_command(phase)
        elif phase == "full_throttle_accelerate":
            throttle = 1.0
        elif phase == "zero_throttle_decel":
            throttle = 0.0
        else:
            # Closed-loop phases use the actual simulator feedback, which is
            # the command applied to the vehicle rather than the target speed.
            throttle = finite(row.get("throttle_feedback"))
        if throttle is None:
            continue
        groups.setdefault(phase, []).append((stamp, speed, throttle))

    trace: list[dict[str, float | str]] = []
    for phase, samples in groups.items():
        samples.sort()
        for index, (stamp, speed, throttle) in enumerate(samples):
            acceleration = math.nan
            before = index - 1
            while before >= 0 and stamp - samples[before][0] < 0.08:
                before -= 1
            if before >= 0:
                before_stamp, before_speed, _ = samples[before]
                dt = stamp - before_stamp
                if dt <= 0.50:
                    candidate = (speed - before_speed) / dt
                    if math.isfinite(candidate) and -50.0 <= candidate <= 50.0:
                        acceleration = candidate
            trace.append({
                "phase": phase,
                "time_s": stamp,
                "throttle": throttle,
                "speed_mps": speed,
                "acceleration_mps2": acceleration,
            })
    return sorted(trace, key=lambda row: float(row["time_s"]))


def steering_response_metrics(
    rows: list[dict[str, str]],
) -> list[tuple[float, float, float, float, int]]:
    """Summarize direct steering response against simulator truth."""
    groups: dict[str, list[tuple[float, float, float]]] = {}
    for row in rows:
        if not valid_ground_truth_row(row):
            continue
        phase = row.get("phase", "")
        if not phase.startswith("steering_"):
            continue
        match = re.search(r"steering_(-?\d+(?:\.\d+)?)", phase)
        stamp = row_stamp(row)
        speed = finite(row.get("gt_speed_mps"))
        yaw_rate = finite(row.get("gt_yaw_rate_radps"))
        command = finite(match.group(1)) if match else None
        if None not in (stamp, speed, yaw_rate, command):
            groups.setdefault(phase, []).append((stamp, speed, yaw_rate))

    result = []
    for phase, samples in groups.items():
        samples.sort()
        tail = samples[max(0, int(len(samples) * 0.6)):]
        valid = [(speed, yaw) for _, speed, yaw in tail if speed > 0.2]
        if not valid:
            continue
        command_match = re.search(r"steering_(-?\d+(?:\.\d+)?)", phase)
        command = finite(command_match.group(1)) if command_match else None
        if command is None:
            continue
        speed = statistics.median(value[0] for value in valid)
        yaw_rate = statistics.median(value[1] for value in valid)
        curvature = yaw_rate / speed
        result.append((command, speed, yaw_rate, curvature, len(valid)))
    return sorted(result)


def acceleration_map(
    trace: list[dict[str, float | str]], speed_bin_mps: float = 0.5
) -> list[tuple[float, float, float, float, int]]:
    """Aggregate direct-throttle acceleration by throttle and body-speed bin."""
    groups: dict[tuple[float, float], list[float]] = {}
    for sample in trace:
        phase = str(sample["phase"])
        if not (phase.startswith("throttle_") or
                phase.startswith("grid_throttle_")):
            continue
        throttle = float(sample["throttle"])
        speed = float(sample["speed_mps"])
        acceleration = float(sample["acceleration_mps2"])
        if (not math.isfinite(acceleration) or speed_bin_mps <= 0.0 or
                not math.isfinite(speed) or not math.isfinite(throttle)):
            continue
        speed_bin = math.floor(max(0.0, speed) / speed_bin_mps) * speed_bin_mps
        groups.setdefault((throttle, speed_bin), []).append(acceleration)
    result = []
    for (throttle, speed_bin), values in sorted(groups.items()):
        result.append((
            throttle, speed_bin, statistics.median(values),
            statistics.pstdev(values) if len(values) > 1 else 0.0, len(values)))
    return result


def _solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """Solve a small dense system without adding a numerical dependency."""
    size = len(vector)
    augmented = [row[:] + [vector[index]] for index, row in enumerate(matrix)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1.0e-12:
            raise ValueError("singular longitudinal fit")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor == 0.0:
                continue
            augmented[row] = [
                left - factor * right
                for left, right in zip(augmented[row], augmented[column])
            ]
    return [augmented[index][-1] for index in range(size)]


def longitudinal_fit_samples(
    rows: list[dict[str, str]],
) -> list[tuple[float, float, float]]:
    """Build unique truth samples for ``a=f(throttle,speed)`` fitting.

    The recorder samples at 50 Hz, while simulator state events arrive at
    approximately 10 Hz.  Deduplicating by the simulator odometry event count
    prevents repeated truth rows from receiving five times the weight.
    """
    groups: dict[str, dict[int, tuple[float, float]]] = {}
    for row in rows:
        if not valid_ground_truth_row(row):
            continue
        phase = row.get("phase", "")
        if not (phase.startswith("throttle_") or
                phase.startswith("grid_throttle_")):
            continue
        stamp = row_stamp(row)
        speed = finite(row.get("gt_speed_mps"))
        throttle = phase_command(phase)
        event = finite(row.get("gt_odom_event_count"))
        if None in (stamp, speed, throttle, event) or speed < 0.0:
            continue
        groups.setdefault(phase, {})[int(event)] = (stamp, speed)

    result: list[tuple[float, float, float]] = []
    for phase, event_rows in groups.items():
        samples = sorted(event_rows.values())
        throttle = phase_command(phase)
        if throttle is None or len(samples) < 3:
            continue
        for index in range(1, len(samples) - 1):
            before_t, before_speed = samples[index - 1]
            center_t, center_speed = samples[index]
            after_t, after_speed = samples[index + 1]
            dt = after_t - before_t
            if not 0.05 <= dt <= 0.50:
                continue
            acceleration = (after_speed - before_speed) / dt
            if not math.isfinite(acceleration) or not -15.0 <= acceleration <= 15.0:
                continue
            if not 0.0 <= center_speed <= 22.88:
                continue
            result.append((throttle, center_speed, acceleration))
    return result


def longitudinal_acceleration_fit(
    rows: list[dict[str, str]],
    operating_speeds: tuple[float, ...] = (
        0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0,
        14.0, 16.0, 18.0, 20.0, 23.0,
    ),
) -> tuple[list[float], list[tuple[float, float, float]], float, int] | None:
    """Fit ``a = c0 + c1*u + c2*v + c3*u*v`` with Huber reweighting.

    This is a diagnostics-only model.  It captures the measured full-range
    throttle envelope while down-weighting derivative spikes caused by the
    native 10 Hz truth cadence.  The returned operating table contains the
    fitted zero-acceleration throttle and incremental throttle per m/s^2.
    """
    samples = longitudinal_fit_samples(rows)
    if len(samples) < 12:
        return None

    def features(sample: tuple[float, float, float]) -> list[float]:
        throttle, speed, _ = sample
        return [1.0, throttle, speed, throttle * speed]

    weights = [1.0] * len(samples)
    coefficients = [0.0] * 4
    for _ in range(8):
        matrix = [[0.0] * 4 for _ in range(4)]
        vector = [0.0] * 4
        for sample, weight in zip(samples, weights):
            row = features(sample)
            acceleration = sample[2]
            for left in range(4):
                vector[left] += weight * row[left] * acceleration
                for right in range(4):
                    matrix[left][right] += weight * row[left] * row[right]
        try:
            coefficients = _solve_linear_system(matrix, vector)
        except ValueError:
            return None
        residuals = [
            sample[2] - sum(c * x for c, x in zip(coefficients, features(sample)))
            for sample in samples
        ]
        center = statistics.median(residuals)
        deviations = [abs(value - center) for value in residuals]
        scale = max(1.4826 * statistics.median(deviations), 0.25)
        cutoff = 1.5 * scale
        weights = [
            1.0 if abs(residual - center) <= cutoff else cutoff / abs(residual - center)
            for residual in residuals
        ]

    residuals = [
        sample[2] - sum(c * x for c, x in zip(coefficients, features(sample)))
        for sample in samples
    ]
    rmse = math.sqrt(sum(value * value for value in residuals) / len(residuals))
    operating_table = []
    for speed in operating_speeds:
        slope = coefficients[1] + coefficients[3] * speed
        if slope <= 1.0e-6:
            continue
        hold = -(coefficients[0] + coefficients[2] * speed) / slope
        operating_table.append((speed, max(0.0, min(1.0, hold)), 1.0 / slope))
    return coefficients, operating_table, rmse, len(samples)


def wheel_speed_body_speed_samples(
    rows: list[dict[str, str]],
) -> list[tuple[float, float]]:
    """Return stable encoder-speed/truth-speed points for slip calibration.

    Encoder angles are converted with the documented wheel radius.  This is
    not a wheel-radius fit: the result identifies the speed-dependent slip
    correction that is applied after the documented mechanical conversion.
    One sample per simulator odometry event is retained, and only the stable
    tail of each direct-throttle phase is used.
    """
    groups: dict[str, dict[int, tuple[float, float, float, float]]] = {}
    for row in rows:
        if not valid_ground_truth_row(row):
            continue
        phase = row.get("phase", "")
        if not (phase.startswith("throttle_") or
                phase.startswith("grid_throttle_")):
            continue
        stamp = finite(row_stamp(row))
        event = finite(row.get("gt_odom_event_count"))
        truth_speed = finite(row.get("gt_speed_mps"))
        left = finite(row.get("left_encoder_rad"))
        right = finite(row.get("right_encoder_rad"))
        if None in (stamp, event, truth_speed, left, right):
            continue
        groups.setdefault(phase, {})[int(event)] = (
            stamp, truth_speed, left, right)

    result: list[tuple[float, float]] = []
    for event_rows in groups.values():
        samples = sorted(event_rows.values())
        interval_samples: list[tuple[float, float]] = []
        for before, after in zip(samples, samples[1:]):
            dt = after[0] - before[0]
            if not 0.05 <= dt <= 0.50:
                continue
            wheel_speed = DOCUMENTED_WHEEL_RADIUS_M * (
                (after[2] - before[2]) + (after[3] - before[3])) / (2.0 * dt)
            if (not math.isfinite(wheel_speed) or not math.isfinite(after[1]) or
                    wheel_speed < 0.0 or wheel_speed > 40.0 or
                    after[1] < 0.0 or after[1] > 25.0):
                continue
            interval_samples.append((wheel_speed, after[1]))
        if len(interval_samples) < 8:
            continue
        tail = interval_samples[max(0, int(len(interval_samples) * 0.60)):]
        wheel_speed = statistics.median(value[0] for value in tail)
        body_speed = statistics.median(value[1] for value in tail)
        # A stable phase cannot have body speed materially above its wheel
        # speed. Such points are reset/phase-transition contamination.
        if wheel_speed < 0.5 and body_speed > 0.5:
            continue
        if body_speed > wheel_speed + 2.0:
            continue
        result.append((wheel_speed, body_speed))
    return result


def wheel_speed_body_speed_map(
    rows: list[dict[str, str]],
    wheel_speed_knots: tuple[float, ...] = tuple(float(v) for v in range(0, 31, 2)),
) -> tuple[list[tuple[float, float, int]], float, int] | None:
    """Fit a monotonic wheel-speed to body-speed slip map.

    The direct sweep has repeated operating points.  Median one-metre bins
    suppress cadence noise, then a weighted pool-adjacent-violators pass keeps
    the physically required monotonic body-speed relationship.  The returned
    map is suitable for runtime interpolation in sensor odometry.
    """
    samples = wheel_speed_body_speed_samples(rows)
    if len(samples) < 8:
        return None
    bins: dict[int, list[float]] = {}
    for wheel_speed, body_speed in samples:
        bins.setdefault(int(math.floor(wheel_speed)), []).append(body_speed)
    binned: list[tuple[float, float, int]] = []
    for index in sorted(bins):
        values = bins[index]
        if not values:
            continue
        # Use the median wheel speed as the interpolation abscissa as well;
        # bin indices are only used to make the robust groups repeatable.
        wheel_values = [wheel for wheel, _ in samples
                        if int(math.floor(wheel)) == index]
        binned.append((statistics.median(wheel_values), statistics.median(values), len(values)))
    if not binned:
        return None

    # Weighted isotonic regression for a non-decreasing body-speed map.
    blocks: list[list[float | int]] = []
    for index, (_, value, weight) in enumerate(binned):
        blocks.append([index, index, value, weight])
        while len(blocks) >= 2 and blocks[-2][2] > blocks[-1][2]:
            right = blocks.pop()
            left = blocks.pop()
            total_weight = int(left[3]) + int(right[3])
            value = (float(left[2]) * int(left[3]) +
                     float(right[2]) * int(right[3])) / total_weight
            blocks.append([left[0], right[1], value, total_weight])
    fitted = [0.0] * len(binned)
    for first, last, value, _ in blocks:
        for index in range(int(first), int(last) + 1):
            fitted[index] = float(value)

    knots = [(0.0, 0.0)]
    knots.extend((wheel, value) for (wheel, _, _), value in zip(binned, fitted)
                 if wheel > 0.0)
    knots = sorted(dict(knots).items())

    def interpolate(wheel_speed: float) -> float:
        speed = max(0.0, wheel_speed)
        if speed <= knots[0][0]:
            return knots[0][1]
        if speed >= knots[-1][0]:
            return knots[-1][1]
        for (lower_speed, lower_body), (upper_speed, upper_body) in zip(knots, knots[1:]):
            if speed <= upper_speed:
                ratio = (speed - lower_speed) / (upper_speed - lower_speed)
                return lower_body + ratio * (upper_body - lower_body)
        return knots[-1][1]

    output = [
        (speed, interpolate(speed), len(samples))
        for speed in wheel_speed_knots
    ]
    residuals = [body - interpolate(wheel) for wheel, body in samples]
    rmse = math.sqrt(sum(value * value for value in residuals) / len(residuals))
    return output, rmse, len(samples)


def truth_speed_tracking_metrics(
    rows: list[dict[str, str]],
) -> tuple[list[float], list[float]]:
    """Return target-minus-truth speed errors for closed-loop speed phases."""
    errors = []
    per_phase: list[float] = []
    grouped: dict[str, list[float]] = {}
    for row in rows:
        if not valid_ground_truth_row(row):
            continue
        phase = row.get("phase", "")
        if not phase.startswith("speed_"):
            continue
        target = finite(row.get("target_speed_mps"))
        truth = finite(row.get("gt_speed_mps"))
        if None not in (target, truth) and target > 0.1:
            grouped.setdefault(phase, []).append(target - truth)
    for values in grouped.values():
        errors.extend(values)
        per_phase.append(statistics.median(stable_tail(values)))
    return errors, per_phase


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


def report_field(rows: list[dict[str, str]], field: str, label: str | None = None) -> None:
    values = [finite(row.get(field)) for row in rows]
    values = [value for value in values if value is not None]
    if not values:
        return
    name = label or field
    p95 = percentile(values, 0.95)
    print(f"{name}_median={statistics.median(values):.3f} {name}_p95={p95:.3f}")


def pose_difference(rows: list[dict[str, str]], left: str, right: str) -> list[float]:
    result = []
    for row in rows:
        lx = finite(row.get(f"x_{left}_m"))
        ly = finite(row.get(f"y_{left}_m"))
        rx = finite(row.get(f"x_{right}_m"))
        ry = finite(row.get(f"y_{right}_m"))
        if None not in (lx, ly, rx, ry):
            result.append(math.hypot(lx - rx, ly - ry))
    return result


def throttle_table(
    rows: list[dict[str, str]], speed_field: str,
    max_throttle: float | None = None,
) -> list[tuple[float, float, float, int]]:
    grouped: dict[float, list[float]] = {}
    for row in rows:
        if not valid_ground_truth_row(row):
            continue
        phase = row.get("phase", "")
        if not phase.startswith("throttle_"):
            continue
        throttle = finite(phase.removeprefix("throttle_"))
        speed = finite(row.get(speed_field))
        if (throttle is not None and speed is not None and speed >= 0.0 and
                (max_throttle is None or throttle <= max_throttle)):
            grouped.setdefault(throttle, []).append(speed)

    result = []
    for throttle in sorted(grouped):
        samples = stable_tail(grouped[throttle])
        if not samples:
            continue
        result.append((statistics.median(samples), throttle, statistics.pstdev(samples), len(samples)))
    # Speed is the interpolation domain and must be strictly increasing.
    result.sort()
    deduplicated: list[tuple[float, float, float, int]] = []
    for sample in result:
        if deduplicated and sample[0] <= deduplicated[-1][0]:
            if sample[1] > deduplicated[-1][1]:
                deduplicated[-1] = sample
            continue
        deduplicated.append(sample)
    return deduplicated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument(
        "--feedforward-output", type=Path,
        help="write speed_mps,throttle CSV for the actuator configuration")
    parser.add_argument(
        "--trajectory", type=Path,
        help="optional commented-header raceline for EKF path/lap metrics")
    parser.add_argument(
        "--ground-truth", type=Path,
        help="optional diagnostics-only monitor CSV for encoder scale metrics")
    parser.add_argument(
        "--max-encoder-step-rad", type=float, default=40.0,
        help="largest non-negative per-sample wheel-angle step included in the fit")
    parser.add_argument(
        "--feedforward-max-throttle", type=float, default=1.0,
        help=("highest direct-throttle step used for production feed-forward; "
              "the full-range response report is always retained"))
    parser.add_argument(
        "--response-output", type=Path,
        help="write the truth-derived throttle/speed/acceleration trace")
    parser.add_argument(
        "--acceleration-map-output", type=Path,
        help="write the truth-derived throttle x speed acceleration map")
    parser.add_argument(
        "--longitudinal-fit-output", type=Path,
        help="write the robust truth-derived throttle/acceleration fit")
    parser.add_argument(
        "--wheel-speed-map-output", type=Path,
        help="write the truth-derived encoder wheel-speed to body-speed slip map")
    parser.add_argument(
        "--steering-output", type=Path,
        help="write truth-derived direct steering response metrics")
    args = parser.parse_args()

    rows = read_rows(args.input)
    if not rows:
        raise SystemExit("input contains no telemetry rows")

    # A full-suite recording deliberately resets between every isolated step.
    # Do not discard the later steps as if those diagnostic resets were a
    # failed single run. Legacy recordings without reset phases retain the
    # original reset-prefix handling.
    has_isolated_resets = any(row.get("phase") == "reset" for row in rows)
    if has_isolated_resets:
        reset_time = reset_kind = None
    else:
        rows, reset_time, reset_kind = trim_after_encoder_reset(rows)
    valid_rows = [row for row in rows if valid_ground_truth_row(row)]
    rejected_ground_truth_rows = len(rows) - len(valid_rows)
    if rejected_ground_truth_rows:
        print(f"ground_truth_boundary_rows_removed={rejected_ground_truth_rows}")
    rows = valid_rows
    rates = [finite(row.get("lidar_rate_hz")) for row in rows]
    rates = [value for value in rates if value is not None and value > 0.0]
    speed_field = preferred_speed_field(rows)
    speed = [finite(row.get(speed_field)) for row in rows]
    speed = [value for value in speed if value is not None]
    print(f"rows={len(rows)}")
    print(f"speed_reference={speed_field}")
    if reset_time is not None:
        print(f"encoder_reset_handled_s={reset_time:.3f} kind={reset_kind}")
    if rates:
        print(f"lidar_rate_hz_median={statistics.median(rates):.3f}")
    for field in (
        "imu_rate_hz", "left_encoder_rate_hz", "right_encoder_rate_hz",
        "odom_rate_hz", "amcl_rate_hz", "ekf_rate_hz",
        "gt_odom_rate_hz", "gt_ips_rate_hz", "collision_rate_hz",
        "controller_command_rate_hz", "throttle_command_rate_hz",
        "steering_command_rate_hz", "throttle_feedback_rate_hz",
        "steering_feedback_rate_hz",
    ):
        report_field(rows, field)
    if speed:
        print(f"{speed_field}_min={min(speed):.3f} {speed_field}_max={max(speed):.3f}")

    collision_values = [finite(row.get("gt_collision_count")) for row in rows]
    collision_values = [value for value in collision_values if value is not None]
    if collision_values:
        print(f"ground_truth_collision_count_max={max(collision_values):.0f}")

    phase_metrics = phase_response_metrics(rows, speed_field)
    if phase_metrics:
        print(
            "phase,command,initial_speed_mps,peak_speed_mps,"
            "tail_speed_mps,peak_accel_mps2,samples"
        )
        for phase, command, initial, peak, tail, peak_accel, samples in phase_metrics:
            accel = "nan" if not math.isfinite(peak_accel) else f"{peak_accel:.6f}"
            print(
                f"{phase},{command:.6f},{initial:.6f},{peak:.6f},"
                f"{tail:.6f},{accel},{samples}"
            )

    steering_metrics = steering_response_metrics(rows)
    if steering_metrics:
        print("steering,gt_speed_mps,gt_yaw_rate_radps,gt_curvature_rad_per_m,samples")
        for command, speed, yaw_rate, curvature, samples in steering_metrics:
            print(f"{command:.6f},{speed:.6f},{yaw_rate:.6f},{curvature:.6f},{samples}")
        if args.steering_output:
            args.steering_output.parent.mkdir(parents=True, exist_ok=True)
            with args.steering_output.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow((
                    "steering", "gt_speed_mps", "gt_yaw_rate_radps",
                    "gt_curvature_rad_per_m", "samples"))
                writer.writerows(steering_metrics)
            print(f"steering_response={args.steering_output}")

    trace = response_trace(rows, speed_field)
    if args.response_output:
        args.response_output.parent.mkdir(parents=True, exist_ok=True)
        with args.response_output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=("phase", "time_s", "throttle", "speed_mps", "acceleration_mps2"),
            )
            writer.writeheader()
            writer.writerows(trace)
        print(f"response_trace={args.response_output}")

    if args.acceleration_map_output:
        args.acceleration_map_output.parent.mkdir(parents=True, exist_ok=True)
        with args.acceleration_map_output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("throttle", "speed_bin_mps", "acceleration_mps2", "acceleration_std_mps2", "samples"))
            writer.writerows(acceleration_map(trace))
        print(f"acceleration_map={args.acceleration_map_output}")

    longitudinal_fit = longitudinal_acceleration_fit(rows)
    if longitudinal_fit is not None:
        coefficients, operating_table, rmse, fit_samples = longitudinal_fit
        print(
            "longitudinal_fit_model="
            "a=c0+c1*throttle+c2*speed+c3*throttle*speed"
        )
        print(
            "longitudinal_fit_coefficients="
            + ",".join(f"{value:.9f}" for value in coefficients)
            + f" rmse_mps2={rmse:.3f} samples={fit_samples}"
        )
        print("speed_mps,fit_hold_throttle,fit_throttle_per_accel")
        for speed_mps, hold_throttle, throttle_per_accel in operating_table:
            print(f"{speed_mps:.6f},{hold_throttle:.6f},{throttle_per_accel:.6f}")
        if args.longitudinal_fit_output:
            args.longitudinal_fit_output.parent.mkdir(parents=True, exist_ok=True)
            with args.longitudinal_fit_output.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(("record_type", "name", "speed_mps", "value", "value_2", "samples"))
                for index, coefficient in enumerate(coefficients):
                    writer.writerow(("coefficient", f"c{index}", "", f"{coefficient:.9f}", "", fit_samples))
                for speed_mps, hold_throttle, throttle_per_accel in operating_table:
                    writer.writerow((
                        "operating_point", "", f"{speed_mps:.6f}",
                        f"{hold_throttle:.6f}", f"{throttle_per_accel:.6f}", fit_samples))
            print(f"longitudinal_fit={args.longitudinal_fit_output}")

    wheel_speed_map = wheel_speed_body_speed_map(rows)
    if wheel_speed_map is not None:
        map_points, map_rmse, map_samples = wheel_speed_map
        print(
            "wheel_speed_body_speed_map="
            f"rmse_mps={map_rmse:.3f} samples={map_samples}")
        print("wheel_speed_mps,body_speed_mps,map_samples")
        for wheel_speed, body_speed, samples in map_points:
            print(f"{wheel_speed:.6f},{body_speed:.6f},{samples}")
        if args.wheel_speed_map_output:
            args.wheel_speed_map_output.parent.mkdir(parents=True, exist_ok=True)
            with args.wheel_speed_map_output.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(("wheel_speed_mps", "body_speed_mps", "source_samples"))
                writer.writerows(map_points)
            print(f"wheel_speed_body_speed_map_output={args.wheel_speed_map_output}")

    truth_odom_position_error = []
    truth_odom_speed_error = []
    for row in rows:
        gx = finite(row.get("gt_x_m"))
        gy = finite(row.get("gt_y_m"))
        ox = finite(row.get("x_odom_m"))
        oy = finite(row.get("y_odom_m"))
        gv = finite(row.get("gt_speed_mps"))
        ov = finite(row.get("speed_mps"))
        if None not in (gx, gy, ox, oy):
            truth_odom_position_error.append(math.hypot(gx - ox, gy - oy))
        if None not in (gv, ov):
            truth_odom_speed_error.append(gv - ov)
    if truth_odom_position_error:
        print(
            "ground_truth_vs_odom_position_error_m="
            f"median={statistics.median(truth_odom_position_error):.3f} "
            f"p95={percentile(truth_odom_position_error, 0.95):.3f} "
            f"max={max(truth_odom_position_error):.3f}"
        )
    if truth_odom_speed_error:
        absolute = [abs(value) for value in truth_odom_speed_error]
        print(
            "ground_truth_vs_odom_speed_error_mps="
            f"median={statistics.median(truth_odom_speed_error):.3f} "
            f"abs_p95={percentile(absolute, 0.95):.3f}"
        )

    report_field(rows, "amcl_timing_ms")
    report_field(rows, "amcl_gpu_transfer_ms")
    report_field(rows, "amcl_gpu_pf_ms")
    report_field(rows, "amcl_gpu_callback_ms")
    report_field(rows, "lidar_event_count", "lidar_events")

    tracking_error = []
    absolute_tracking_error = []
    overspeed_error = []
    for row in rows:
        target = finite(row.get("controller_speed_mps"))
        measured = finite(row.get("speed_mps"))
        if target is not None and measured is not None and target > 0.1:
            error = target - measured
            tracking_error.append(error)
            absolute_tracking_error.append(abs(error))
            overspeed_error.append(max(0.0, -error))
    if tracking_error:
        print(
            "speed_tracking_error_mps="
            f"median={statistics.median(tracking_error):.3f} "
            f"p95={percentile(tracking_error, 0.95):.3f}"
        )
        print(
            "speed_tracking_abs_error_mps="
            f"median={statistics.median(absolute_tracking_error):.3f} "
            f"p95={percentile(absolute_tracking_error, 0.95):.3f} "
            f"max={max(absolute_tracking_error):.3f}"
        )
        overspeed_samples = sum(value > 0.10 for value in overspeed_error)
        print(
            "speed_overspeed_fraction="
            f"gt_0.10={overspeed_samples / len(overspeed_error):.3f}"
        )

    truth_tracking_error, truth_phase_errors = truth_speed_tracking_metrics(rows)
    if truth_tracking_error:
        absolute = [abs(value) for value in truth_tracking_error]
        print(
            "truth_speed_tracking_error_mps="
            f"median={statistics.median(truth_tracking_error):.3f} "
            f"abs_p95={percentile(absolute, 0.95):.3f} "
            f"max_abs={max(absolute):.3f} "
            f"phases={len(truth_phase_errors)}"
        )
        print(
            "truth_speed_tracking_tail_error_mps="
            f"median={statistics.median(truth_phase_errors):.3f} "
            f"abs_p95={percentile([abs(value) for value in truth_phase_errors], 0.95):.3f}"
        )

    amcl_ekf = pose_difference(rows, "amcl", "ekf")
    if amcl_ekf:
        print(
            "amcl_ekf_xy_difference_m="
            f"median={statistics.median(amcl_ekf):.3f} "
            f"p95={percentile(amcl_ekf, 0.95):.3f} max={max(amcl_ekf):.3f}"
        )

    if args.trajectory:
        trajectory = read_trajectory(args.trajectory)
        path_error, seam_crossings = raceline_metrics(rows, trajectory)
        if path_error:
            print(
                "ekf_raceline_error_m="
                f"median={statistics.median(path_error):.3f} "
                f"p95={percentile(path_error, 0.95):.3f} max={max(path_error):.3f}"
            )
            print(f"raceline_forward_seam_crossings={seam_crossings}")

    if args.ground_truth:
        metrics = encoder_distance_metrics(
            rows, args.ground_truth, args.max_encoder_step_rad)
        if metrics is None:
            print("encoder_scale=unavailable")
        else:
            gt_distance, nominal_distance, scale, samples, rejected_steps = metrics
            print(
                "encoder_distance_calibration="
                f"documented_radius_m={DOCUMENTED_WHEEL_RADIUS_M:.4f} "
                f"ground_truth_m={gt_distance:.3f} "
                f"nominal_m={nominal_distance:.3f} "
                f"scale={scale:.5f} "
                f"steps={samples} rejected_steps={rejected_steps}"
            )

    full_table = throttle_table(rows, speed_field)
    table = throttle_table(rows, speed_field, args.feedforward_max_throttle)
    if full_table and table != full_table:
        print(
            "feedforward_fit_range="
            f"throttle_0.000_to_{args.feedforward_max_throttle:.3f} "
            "(full-range response remains in phase table above)"
        )
    if table:
        print("speed_mps,throttle,speed_std_mps,samples")
        for speed_mps, throttle, spread, count in table:
            print(f"{speed_mps:.6f},{throttle:.6f},{spread:.6f},{count}")

    if args.feedforward_output:
        args.feedforward_output.parent.mkdir(parents=True, exist_ok=True)
        with args.feedforward_output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("speed_mps", "throttle"))
            writer.writerows((f"{speed_mps:.6f}", f"{throttle:.6f}")
                             for speed_mps, throttle, _, _ in table)
        print(f"feedforward_table={args.feedforward_output}")


if __name__ == "__main__":
    main()
