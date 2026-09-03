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
DOCUMENTED_MAX_SPEED_MPS = 22.88
DEFAULT_GROUND_TRUTH_STEP_JITTER_FACTOR = 1.5
DEFAULT_GROUND_TRUTH_POSITION_MARGIN_M = 0.15
DEFAULT_GROUND_TRUTH_MAX_GAP_S = 0.5
DOCUMENTED_LONGITUDINAL_EXTREMUM_SLIP = 0.15
DOCUMENTED_LONGITUDINAL_ASYMPTOTE_SLIP = 0.25


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


def truth_stamp(row: dict[str, str]) -> float | None:
    """Return the simulator odometry source time when available.

    Recorder rows are written by a 50 Hz timer, while the simulator topics
    arrive at their own native cadence. Truth derivatives and phase fits must
    use the timestamp carried by the ground-truth message, not the timer row
    time, otherwise callback scheduling becomes part of the vehicle model.
    """
    return finite(row.get("gt_odom_stamp_s")) or row_stamp(row)


def source_event(row: dict[str, str], event_field: str) -> int | None:
    """Return an integral source event counter when a recorder provides one."""
    value = finite(row.get(event_field))
    if value is None or value < 0.0 or value != float(int(value)):
        return None
    return int(value)


def deduplicate_source_events(
    rows: list[dict[str, str]],
    event_field: str = "gt_odom_event_count",
) -> tuple[list[dict[str, str]], int]:
    """Keep one recorder row for each source event.

    Calibration records are commonly written at 50 Hz while the simulator
    state is refreshed at about 10 Hz.  The event counter is therefore the
    authoritative identity of a source sample.  The last recorder row for an
    event is retained because it contains the freshest values from the other
    callbacks.  Rows without a counter are retained for compatibility with
    older files; their counts are reported as recorder rows, not source rows.

    The recorder's event counter is monotonic for the lifetime of one CSV, so
    phase transitions must not create a second copy of an event that straddles
    a timer tick. Separate files remain the boundary between experiments.
    """
    result: list[dict[str, str]] = []
    positions: dict[int, int] = {}
    duplicates = 0
    for row in rows:
        event = source_event(row, event_field)
        if event is None:
            result.append(row)
            continue
        key = event
        position = positions.get(key)
        if position is None:
            positions[key] = len(result)
            result.append(row)
        else:
            result[position] = row
            duplicates += 1
    return result, duplicates


def timestamped_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Return rows in timestamp order while preserving legacy no-stamp files."""
    indexed = [
        (index, row_stamp(row), row)
        for index, row in enumerate(rows)
    ]
    stamped = [item for item in indexed if item[1] is not None]
    if not stamped:
        return rows
    ordered = [row for _, _, row in sorted(
        stamped, key=lambda item: (float(item[1]), item[0]))]
    # A mixed-format file cannot be fully synchronized. Keep its legacy rows
    # after the timestamped portion rather than silently dropping data.
    return ordered + [row for _, stamp, row in indexed if stamp is None]


def ground_truth_step_limit_m(
    dt_s: float,
    max_speed_mps: float = DOCUMENTED_MAX_SPEED_MPS,
    jitter_factor: float = DEFAULT_GROUND_TRUTH_STEP_JITTER_FACTOR,
    position_margin_m: float = DEFAULT_GROUND_TRUTH_POSITION_MARGIN_M,
    max_gap_s: float = DEFAULT_GROUND_TRUTH_MAX_GAP_S,
) -> float | None:
    """Return the timestamp-aware maximum plausible truth displacement."""
    if (not math.isfinite(dt_s) or dt_s <= 0.0 or dt_s > max_gap_s or
            not math.isfinite(max_speed_mps) or max_speed_mps <= 0.0 or
            not math.isfinite(jitter_factor) or jitter_factor < 1.0 or
            not math.isfinite(position_margin_m) or position_margin_m < 0.0):
        return None
    return max_speed_mps * dt_s * jitter_factor + position_margin_m


def valid_ground_truth_step(
    step_m: float,
    before: dict[str, str],
    after: dict[str, str],
    max_speed_mps: float = DOCUMENTED_MAX_SPEED_MPS,
    jitter_factor: float = DEFAULT_GROUND_TRUTH_STEP_JITTER_FACTOR,
    position_margin_m: float = DEFAULT_GROUND_TRUTH_POSITION_MARGIN_M,
    max_gap_s: float = DEFAULT_GROUND_TRUTH_MAX_GAP_S,
    legacy_max_step_m: float = 0.60,
) -> bool:
    """Validate one truth displacement using source timing when available."""
    if not math.isfinite(step_m) or step_m < 0.0:
        return False
    before_stamp = row_stamp(before)
    after_stamp = row_stamp(after)
    if before_stamp is not None and after_stamp is not None:
        limit = ground_truth_step_limit_m(
            after_stamp - before_stamp,
            max_speed_mps,
            jitter_factor,
            position_margin_m,
            max_gap_s,
        )
        return limit is not None and step_m <= limit
    # Old monitor exports have no source timestamp. Preserve a conservative
    # compatibility path, but make the fallback explicit in the report.
    return step_m < legacy_max_step_m


def valid_ground_truth_row(row: dict[str, str]) -> bool:
    """Reject samples after the open-ground vehicle leaves the world plane."""
    z = finite(row.get("gt_odom_z_m"))
    if z is None:
        z = finite(row.get("gt_z_m"))
    return z is None or 0.0 <= z <= 0.20


def ground_truth_position(row: dict[str, str]) -> tuple[float | None, float | None]:
    """Return the timestamped simulator-odom position from a calibration row.

    New recordings expose the source explicitly as ``gt_odom_*``. The
    generic fields remain a compatibility path for older files, where IPS
    could have overwritten them in the recorder.
    """
    x = finite(row.get("gt_odom_x_m"))
    y = finite(row.get("gt_odom_y_m"))
    if x is None or y is None:
        x = finite(row.get("gt_x_m"))
        y = finite(row.get("gt_y_m"))
    return x, y


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
    max_ground_truth_speed_mps: float = DOCUMENTED_MAX_SPEED_MPS,
    ground_truth_step_jitter_factor: float = DEFAULT_GROUND_TRUTH_STEP_JITTER_FACTOR,
    ground_truth_position_margin_m: float = DEFAULT_GROUND_TRUTH_POSITION_MARGIN_M,
    max_ground_truth_gap_s: float = DEFAULT_GROUND_TRUTH_MAX_GAP_S,
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
    gt_rows, _ = deduplicate_source_events(gt_rows)
    gt_rows = timestamped_rows(gt_rows)
    rows, _ = deduplicate_source_events(rows)
    rows = timestamped_rows(rows)
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
        stamp = truth_stamp(row)
        return stamp is not None and shared_window[0] <= stamp <= shared_window[1]

    gt_rows = [row for row in gt_rows if in_window(row)]
    gt_distance = 0.0
    for before, after in zip(gt_rows, gt_rows[1:]):
        bx, by = ground_truth_position(before)
        ax, ay = ground_truth_position(after)
        if None not in (bx, by, ax, ay):
            step = math.hypot(ax - bx, ay - by)
            # Use source timing so valid high-speed motion is not rejected by
            # the old low-speed-only 0.60 m threshold.  A missing timestamp
            # takes the explicit legacy fallback inside this helper.
            if (0.0 < step and valid_ground_truth_step(
                    step, before, after, max_ground_truth_speed_mps,
                    ground_truth_step_jitter_factor,
                    ground_truth_position_margin_m, max_ground_truth_gap_s)):
                gt_distance += step

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
        phase.startswith("acceleration_") or
        phase in {"full_throttle_accelerate", "zero_throttle_decel"}
    )


def phase_is_diagnostic_reset(phase: str) -> bool:
    """Return whether a phase is a simulator-reset transient."""
    return (
        phase in {"reset", "boundary_reset"} or
        phase.startswith("grid_reset_")
    )


def phase_command(phase: str) -> float | None:
    match = re.search(
        r"(?:throttle|grid_throttle|speed|acceleration)_(-?\d+(?:\.\d+)?)",
        phase)
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
        stamp = truth_stamp(row)
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
        stamp = truth_stamp(row)
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
        stamp = truth_stamp(row)
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


def _weighted_isotonic_decreasing(
    values: list[float], weights: list[int]
) -> list[float]:
    """Weighted non-increasing fit, preserving one output per input knot."""
    blocks: list[dict[str, float | int]] = []
    for value, weight in zip(values, weights):
        blocks.append({
            "sum": value * weight,
            "weight": weight,
            "mean": value,
            "count": 1,
        })
        while len(blocks) >= 2 and blocks[-2]["mean"] < blocks[-1]["mean"]:
            left = blocks.pop(-2)
            right = blocks.pop(-1)
            weight_sum = int(left["weight"] + right["weight"])
            blocks.append({
                "sum": float(left["sum"]) + float(right["sum"]),
                "weight": weight_sum,
                "mean": (float(left["sum"]) + float(right["sum"])) /
                weight_sum,
                "count": int(left["count"] + right["count"]),
            })
    result: list[float] = []
    for block in blocks:
        result.extend([float(block["mean"])] * int(block["count"]))
    return result


def acceleration_envelope(
    trace: list[dict[str, float | str]],
    max_throttle: float = 1.0,
    speed_knots: tuple[float, ...] = (
        0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0,
        16.0, 18.0, 20.0, 22.0, 23.0,
    ),
) -> list[tuple[float, float, float, int]]:
    """Fit the positive-drive acceleration ceiling as a function of speed.

    Only direct full-throttle grid phases are used.  Samples are grouped by
    speed, positive acceleration is retained, and a weighted monotonic fit
    removes isolated derivative spikes while preserving the measured envelope
    shape.  This is intentionally a conservative capability model: it is
    used to bound a requested acceleration, not to claim that every sample
    reaches the fitted median.

    The returned columns are speed knot, observed median, monotonic fit, and
    sample count.  Ground truth is used here only by the offline fitter.
    """
    if (not math.isfinite(max_throttle) or max_throttle <= 0.0 or
            len(speed_knots) < 2 or any(
                not math.isfinite(value) or value < 0.0
                for value in speed_knots) or any(
                left >= right for left, right in zip(speed_knots, speed_knots[1:]))):
        raise ValueError("invalid acceleration-envelope inputs")

    groups: list[list[float]] = [[] for _ in speed_knots]
    for sample in trace:
        phase = str(sample.get("phase", ""))
        if not (phase.startswith("throttle_") or
                phase.startswith("grid_throttle_")):
            continue
        throttle = finite(sample.get("throttle"))
        speed = finite(sample.get("speed_mps"))
        acceleration = finite(sample.get("acceleration_mps2"))
        if (throttle is None or speed is None or acceleration is None or
                abs(throttle - max_throttle) > 1.0e-6 or speed < 0.0 or
                acceleration <= 0.0):
            continue
        index = len(speed_knots) - 1
        for candidate in range(len(speed_knots) - 1):
            boundary = 0.5 * (speed_knots[candidate] +
                              speed_knots[candidate + 1])
            if speed < boundary:
                index = candidate
                break
        groups[index].append(acceleration)

    if not any(groups):
        return []

    observed: list[float] = []
    sample_counts: list[int] = []
    for index, values in enumerate(groups):
        if values:
            observed.append(statistics.median(values))
            sample_counts.append(len(values))
            continue
        # A missing terminal/low-speed knot is filled from the closest
        # observed knot; the caller still sees a zero sample count.
        nearest = min(
            (other for other, candidate in enumerate(groups) if candidate),
            key=lambda other: abs(speed_knots[other] - speed_knots[index]),
            default=None,
        )
        if nearest is None:
            raise ValueError("no full-throttle acceleration samples")
        observed.append(statistics.median(groups[nearest]))
        sample_counts.append(0)

    fitted = _weighted_isotonic_decreasing(
        observed, [max(1, count) for count in sample_counts])
    return [
        (speed, raw, fit, count)
        for speed, raw, fit, count in zip(
            speed_knots, observed, fitted, sample_counts)
    ]


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
        stamp = truth_stamp(row)
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
        stamp = finite(truth_stamp(row))
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


def _slip_phase(phase: str) -> bool:
    """Return whether a phase contains useful longitudinal-slip evidence."""
    return (
        phase.startswith("grid_throttle_") or
        phase.startswith("grid_base_") or
        phase.startswith("grid_brake_")
    )


def _slip_motion_regime(row: dict[str, str]) -> str:
    """Classify slip using the truth derivative only during offline fitting."""
    if row.get("phase", "").startswith("grid_brake_"):
        return "braking"
    acceleration = finite(row.get("gt_longitudinal_accel_mps2"))
    if acceleration is not None:
        if acceleration <= -0.50:
            return "braking"
        if acceleration >= 0.50:
            return "traction"
    return "steady"


def _slip_force_region(slip_ratio: float) -> str:
    """Map measured slip to the official longitudinal force-curve region."""
    absolute_slip = abs(slip_ratio)
    if absolute_slip < DOCUMENTED_LONGITUDINAL_EXTREMUM_SLIP:
        return "pre_extremum"
    if absolute_slip < DOCUMENTED_LONGITUDINAL_ASYMPTOTE_SLIP:
        return "post_extremum"
    return "asymptotic"


def slip_model_samples(
    rows: list[dict[str, str]], min_body_speed_mps: float = 0.75,
) -> list[dict[str, float | str]]:
    """Build unique, timestamped slip samples for offline model fitting.

    AutoDRIVE defines longitudinal slip against the longitudinal body velocity
    ``v_x``.  The encoder speed is calculated from the two source-timestamped
    encoder callbacks by the recorder.  Ground truth is used here only to fit
    and score the model; this function is never called by runtime odometry.
    Samples near standstill are excluded because the official ratio is
    ill-conditioned there.  Braking samples are retained, including a frozen
    encoder, because ``S_x`` near -1 is valid evidence of wheel/ground motion
    disagreement rather than a reset.
    """
    if min_body_speed_mps <= 0.0 or not math.isfinite(min_body_speed_mps):
        raise ValueError("min_body_speed_mps must be finite and positive")
    unique_rows, _ = deduplicate_source_events(rows)
    samples: list[dict[str, float | str]] = []
    for row in timestamped_rows(unique_rows):
        if not valid_ground_truth_row(row) or not _slip_phase(row.get("phase", "")):
            continue
        vx = finite(row.get("gt_vx_mps"))
        wheel_speed = finite(row.get("encoder_wheel_speed_mps"))
        if wheel_speed is None:
            left_speed = finite(row.get("left_encoder_speed_radps"))
            right_speed = finite(row.get("right_encoder_speed_radps"))
            if left_speed is not None and right_speed is not None:
                wheel_speed = DOCUMENTED_WHEEL_RADIUS_M * 0.5 * (
                    left_speed + right_speed)
        if vx is None or wheel_speed is None or abs(vx) < min_body_speed_mps:
            continue
        if (not math.isfinite(wheel_speed) or abs(wheel_speed) > 60.0 or
                abs(vx) > DOCUMENTED_MAX_SPEED_MPS * 1.25):
            continue
        # Keep the signed definition from the technical guide.  For the
        # forward open-ground sweep vx is positive; retaining the sign also
        # makes braking (frozen wheel, moving body) explicit.
        slip_speed = wheel_speed - vx
        slip_ratio = slip_speed / vx
        if not math.isfinite(slip_ratio) or abs(slip_ratio) > 50.0:
            continue
        wheel_bin = math.floor(abs(wheel_speed) / 2.0) * 2.0
        samples.append({
            "phase": row.get("phase", ""),
            "motion_regime": _slip_motion_regime(row),
            "force_curve_region": _slip_force_region(slip_ratio),
            "wheel_speed_bin_mps": wheel_bin,
            "wheel_speed_mps": wheel_speed,
            "body_vx_mps": vx,
            "slip_speed_mps": slip_speed,
            "slip_ratio": slip_ratio,
            "longitudinal_accel_mps2": (
                finite(row.get("gt_longitudinal_accel_mps2")) or math.nan),
        })
    return samples


def slip_model_metrics(
    rows: list[dict[str, str]], min_body_speed_mps: float = 0.75,
) -> list[dict[str, float | int | str]]:
    """Aggregate a regime- and speed-conditioned longitudinal slip model.

    The median is the proposed correction centre; MAD and quantiles expose
    uncertainty so a future runtime observer can increase wheel covariance
    instead of blindly applying a correction in a broad/high-slip bin.
    """
    samples = slip_model_samples(rows, min_body_speed_mps)
    groups: dict[tuple[str, str, float], list[dict[str, float | str]]] = {}
    for sample in samples:
        key = (
            str(sample["motion_regime"]),
            str(sample["force_curve_region"]),
            float(sample["wheel_speed_bin_mps"]),
        )
        groups.setdefault(key, []).append(sample)

    result: list[dict[str, float | int | str]] = []
    for (motion_regime, force_region, wheel_bin), values in sorted(groups.items()):
        wheel_values = [float(value["wheel_speed_mps"]) for value in values]
        body_values = [float(value["body_vx_mps"]) for value in values]
        slip_speed_values = [float(value["slip_speed_mps"]) for value in values]
        slip_values = [float(value["slip_ratio"]) for value in values]
        median_slip = statistics.median(slip_values)
        mad = statistics.median(abs(value - median_slip) for value in slip_values)
        ordered = sorted(slip_values)
        result.append({
            "motion_regime": motion_regime,
            "force_curve_region": force_region,
            "wheel_speed_bin_mps": wheel_bin,
            "median_wheel_speed_mps": statistics.median(wheel_values),
            "median_body_vx_mps": statistics.median(body_values),
            "median_slip_speed_mps": statistics.median(slip_speed_values),
            "median_slip_ratio": median_slip,
            "slip_ratio_mad": mad,
            "slip_ratio_p10": percentile(ordered, 0.10),
            "slip_ratio_p90": percentile(ordered, 0.90),
            "samples": len(values),
        })
    return result


def frozen_encoder_brake_model(
    rows: list[dict[str, str]], min_body_speed_mps: float = 0.75,
    max_frozen_wheel_speed_mps: float = 0.15, speed_bin_width_mps: float = 2.0,
) -> tuple[dict[str, float | int], list[dict[str, float | int]]] | None:
    """Fit a bounded deceleration prior for a moving frozen driven encoder.

    This is an offline identification helper.  Ground-truth longitudinal
    acceleration selects the braking samples and is never an input to runtime
    odometry.  Robust per-speed-bin medians are fitted instead of individual
    samples because simulator callback timing and collision transients can
    produce large derivative outliers.  The resulting prior is appropriate
    only for the ambiguous ``encoder ~= 0`` state after the IMU has become
    quiet; it is not a generic slip-ratio correction.
    """
    if (min_body_speed_mps <= 0.0 or not math.isfinite(min_body_speed_mps) or
            max_frozen_wheel_speed_mps < 0.0 or
            not math.isfinite(max_frozen_wheel_speed_mps) or
            speed_bin_width_mps <= 0.0 or not math.isfinite(speed_bin_width_mps)):
        raise ValueError("invalid frozen-encoder brake-model bounds")

    samples = []
    for sample in slip_model_samples(rows, min_body_speed_mps):
        wheel_speed = float(sample["wheel_speed_mps"])
        body_speed = float(sample["body_vx_mps"])
        acceleration = float(sample["longitudinal_accel_mps2"])
        if (abs(wheel_speed) > max_frozen_wheel_speed_mps or
                not math.isfinite(acceleration) or acceleration >= -0.50):
            continue
        samples.append((body_speed, -acceleration))
    if not samples:
        return None

    grouped: dict[int, list[tuple[float, float]]] = {}
    for body_speed, deceleration in samples:
        index = math.floor(body_speed / speed_bin_width_mps)
        grouped.setdefault(index, []).append((body_speed, deceleration))
    bins: list[dict[str, float | int]] = []
    for index, values in sorted(grouped.items()):
        bins.append({
            "speed_bin_mps": index * speed_bin_width_mps,
            "median_body_speed_mps": statistics.median(value[0] for value in values),
            "median_deceleration_mps2": statistics.median(value[1] for value in values),
            "samples": len(values),
        })
    if not bins:
        return None

    x_values = [float(value["median_body_speed_mps"]) for value in bins]
    y_values = [float(value["median_deceleration_mps2"]) for value in bins]
    x_mean = statistics.mean(x_values)
    y_mean = statistics.mean(y_values)
    denominator = sum((value - x_mean) ** 2 for value in x_values)
    gain = (sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values)) /
            denominator if denominator > 1.0e-9 else 0.0)
    # A deceleration prior must not become propulsion or change sign at the
    # top of the identified envelope. Clamp only the fitted coefficients;
    # residuals remain reported so a poor fit cannot look exact.
    gain = max(0.0, gain)
    intercept = max(0.0, y_mean - gain * x_mean)
    predictions = [intercept + gain * value for value in x_values]
    residuals = [actual - predicted for actual, predicted in zip(y_values, predictions)]
    decelerations = [value[1] for value in samples]
    model = {
        "intercept_mps2": intercept,
        "speed_gain_per_s": gain,
        "max_deceleration_p95_mps2": percentile(sorted(decelerations), 0.95) or 0.0,
        "fit_rmse_mps2": math.sqrt(sum(value * value for value in residuals) / len(residuals)),
        "samples": len(samples),
        "bins": len(bins),
    }
    return model, bins


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
    """Derive a monotonic throttle-to-speed table from valid step tails.

    The identification grid deliberately starts several throttle probes from
    an already-moving operating point. Pooling all rows by throttle therefore
    makes a coast at 15 m/s look like the steady speed for a small throttle.
    Keep phase-local tails instead, reject phases whose tail is materially
    below their initial speed, and select the fastest non-decelerating phase
    for each command. This preserves the useful full-range envelope while
    preventing cross-point contamination from entering the actuator table.
    """
    phase_groups: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        if not valid_ground_truth_row(row):
            continue
        phase = row.get("phase", "")
        if not (phase.startswith("throttle_") or
                phase.startswith("grid_throttle_")):
            continue
        throttle = phase_command(phase)
        stamp = truth_stamp(row)
        speed = finite(row.get(speed_field))
        if (throttle is not None and stamp is not None and speed is not None and
                speed >= 0.0 and
                (max_throttle is None or throttle <= max_throttle)):
            phase_groups.setdefault(phase, []).append((stamp, speed))

    candidates: dict[float, list[tuple[float, float, int]]] = {}
    for phase, samples in phase_groups.items():
        samples.sort()
        values = [value for _, value in samples]
        if len(values) < 4:
            continue
        throttle = phase_command(phase)
        if throttle is None:
            continue
        initial = statistics.median(values[:max(1, len(values) // 5)])
        tail = stable_tail(values)
        tail_speed = statistics.median(tail) if tail else values[-1]
        # A target below the current operating point is a braking/coast
        # response, not evidence for the target's steady-speed feed-forward.
        if tail_speed < initial - 0.25:
            continue
        # The zero-throttle grid has intentional coast phases at every high
        # nominal band. Only the from-rest zero-throttle phase is a valid
        # neutral feed-forward point.
        if (throttle <= 1.0e-9 and phase.startswith("grid_throttle_") and
                not phase.endswith("_at_0.00")):
            continue
        spread = statistics.pstdev(tail) if len(tail) > 1 else 0.0
        candidates.setdefault(throttle, []).append(
            (max(values), spread, len(tail)))

    result = []
    for throttle, values in sorted(candidates.items()):
        peak, spread, count = max(values, key=lambda item: item[0])
        result.append((peak, throttle, spread, count))

    # Speed is the interpolation domain and must be strictly increasing. A
    # coarse simulator response can produce equal peaks for adjacent commands;
    # retain the higher command at that speed so the inverse remains monotonic.
    result.sort()
    deduplicated: list[tuple[float, float, float, int]] = []
    for sample in result:
        if deduplicated and sample[0] <= deduplicated[-1][0]:
            if sample[1] > deduplicated[-1][1]:
                deduplicated[-1] = sample
            continue
        deduplicated.append(sample)
    return deduplicated


def _truth_speed(row: dict[str, str]) -> float | None:
    speed = finite(row.get("gt_speed_mps"))
    if speed is not None and speed >= 0.0:
        return speed
    vx = finite(row.get("gt_vx_mps"))
    vy = finite(row.get("gt_vy_mps"))
    if vx is not None and vy is not None:
        return math.hypot(vx, vy)
    return None


def classify_motion_regime(
    row: dict[str, str], previous: dict[str, str] | None = None,
) -> str:
    """Classify one unique truth event for regime-based error reporting."""
    phase = row.get("phase", "")
    if phase_is_diagnostic_reset(phase):
        return "simulator_reset"
    if not valid_ground_truth_row(row):
        return "invalid_ground_truth"

    speed = _truth_speed(row)
    if speed is None:
        return "unknown"

    left = finite(row.get("left_encoder_rad"))
    right = finite(row.get("right_encoder_rad"))
    has_encoder_fields = "left_encoder_rad" in row or "right_encoder_rad" in row
    previous_left = finite(previous.get("left_encoder_rad")) if previous else None
    previous_right = finite(previous.get("right_encoder_rad")) if previous else None
    if has_encoder_fields and speed > 0.2:
        if left is None or right is None or previous_left is None or previous_right is None:
            return "encoder_dropout"
        if left - previous_left < -0.5 or right - previous_right < -0.5:
            return "encoder_reset_or_discontinuity"

    dt = None
    previous_speed = _truth_speed(previous) if previous else None
    if previous is not None:
        before_stamp = truth_stamp(previous)
        after_stamp = truth_stamp(row)
        if before_stamp is not None and after_stamp is not None:
            candidate = after_stamp - before_stamp
            if 1.0e-4 <= candidate <= 1.0:
                dt = candidate
    acceleration = 0.0
    if dt is not None and previous_speed is not None:
        acceleration = (speed - previous_speed) / dt

    yaw_rate = finite(row.get("gt_yaw_rate_radps"))
    yaw_rate = abs(yaw_rate) if yaw_rate is not None else 0.0
    lateral_accel = yaw_rate * speed
    curvature = yaw_rate / max(speed, 0.5)

    # Driven-wheel slip is a regime label, not a runtime correction.  Use the
    # fixed documented radius and the body-longitudinal truth velocity only
    # for development segmentation.
    if (has_encoder_fields and previous is not None and dt is not None and
            left is not None and right is not None and
            previous_left is not None and previous_right is not None):
        wheel_speed = DOCUMENTED_WHEEL_RADIUS_M * (
            (left - previous_left) + (right - previous_right)) / (2.0 * dt)
        vx = finite(row.get("gt_vx_mps"))
        if vx is not None and abs(vx) > 0.5 and math.isfinite(wheel_speed):
            slip_ratio = (abs(wheel_speed) - abs(vx)) / abs(vx)
            if slip_ratio > 0.15:
                return "high_longitudinal_slip"

    if speed < 0.10:
        return "stationary"
    if lateral_accel >= 2.0:
        return "high_lateral_acceleration"
    if curvature >= 0.15:
        return "high_curvature_turn"
    if curvature >= 0.03:
        return "low_curvature_turn"
    if acceleration >= 0.50:
        return "straight_accelerating"
    if acceleration <= -0.50:
        return "straight_coasting"
    return "straight_steady"


def motion_regime_metrics(
    rows: list[dict[str, str]],
) -> list[dict[str, float | int | str]]:
    """Summarize truth-vs-odom speed errors separately by motion regime."""
    groups: dict[str, dict[str, list[float]]] = {}
    previous = None
    previous_regime = None
    for row in timestamped_rows(rows):
        regime = classify_motion_regime(row, previous)
        group = groups.setdefault(regime, {
            "events": [], "duration": [], "truth_speed": [], "speed_error": [],
        })
        stamp = truth_stamp(row)
        truth_speed = _truth_speed(row)
        odom_speed = finite(row.get("speed_mps"))
        group["events"].append(1.0)
        if (previous_regime == regime and previous is not None and
                stamp is not None and truth_stamp(previous) is not None):
            dt = stamp - truth_stamp(previous)
            if 0.0 < dt <= 1.0:
                group["duration"].append(dt)
        if stamp is not None:
            # Keep source timestamps only for compatibility with callers that
            # inspect the intermediate shape; duration uses contiguous spans.
            group.setdefault("stamps", []).append(stamp)
        if truth_speed is not None:
            group["truth_speed"].append(truth_speed)
        if truth_speed is not None and odom_speed is not None:
            group["speed_error"].append(abs(truth_speed - odom_speed))
        previous = row
        previous_regime = regime

    result: list[dict[str, float | int | str]] = []
    for regime, group in sorted(groups.items()):
        errors = group["speed_error"]
        result.append({
            "regime": regime,
            "events": len(group["events"]),
            "duration_s": sum(group["duration"]),
            "truth_speed_median_mps": (
                statistics.median(group["truth_speed"])
                if group["truth_speed"] else math.nan),
            "odom_speed_mae_mps": (
                statistics.mean(errors) if errors else math.nan),
            "odom_speed_p95_abs_error_mps": (
                percentile(errors, 0.95) if errors else math.nan),
            "error_samples": len(errors),
        })
    return result


def acceleration_command_metrics(
    rows: list[dict[str, str]],
) -> tuple[int, float, float] | None:
    """Score requested acceleration against timestamped truth derivatives."""
    errors: list[float] = []
    for row in rows:
        if not row.get("phase", "").startswith("acceleration_"):
            continue
        requested = finite(row.get("acceleration_command_accel_mps2"))
        measured = finite(row.get("gt_longitudinal_accel_mps2"))
        if requested is None or measured is None or abs(measured) > 50.0:
            continue
        errors.append(measured - requested)
    if not errors:
        return None
    absolute = [abs(value) for value in errors]
    return len(errors), statistics.mean(errors), percentile(absolute, 0.95)


RELATIVE_ERROR_FIELDS = (
    "metric", "bin", "samples", "relative_samples",
    "absolute_error_median", "absolute_error_p95", "absolute_error_max",
    "relative_error_median_pct", "relative_error_p95_pct",
    "relative_error_max_pct", "reference_median", "reference_max",
)

RELATIVE_SPEED_BINS = (
    (0.0, 1.0), (1.0, 3.0), (3.0, 5.0), (5.0, 10.0),
    (10.0, 15.0), (15.0, 20.0), (20.0, 23.0),
)

POSITION_DISTANCE_THRESHOLDS_M = (1.0, 5.0, 10.0, 50.0, 100.0)


def _relative_metric_row(
    metric: str,
    bin_name: str,
    absolute_errors: list[float],
    references: list[float],
) -> dict[str, float | int | str]:
    """Summarize absolute error and error relative to a physical reference.

    ``references`` is the target speed for controller metrics, actual truth
    speed for odom velocity metrics, or cumulative truth distance for pose
    metrics.  A zero reference deliberately has no percentage error: a
    percentage at standstill or at the origin has no useful meaning.
    """
    absolute = [value for value in absolute_errors if math.isfinite(value)]
    paired = [
        (abs(error), reference)
        for error, reference in zip(absolute_errors, references)
        if math.isfinite(error) and math.isfinite(reference) and reference > 1.0e-6
    ]
    relative = [100.0 * error / reference for error, reference in paired]
    reference_values = [value for value in references if math.isfinite(value)]

    def value_or_nan(values: list[float], fraction: float | None = None) -> float:
        if not values:
            return math.nan
        if fraction is None:
            return statistics.median(values)
        result = percentile(values, fraction)
        return result if result is not None else math.nan

    return {
        "metric": metric,
        "bin": bin_name,
        "samples": len(absolute),
        "relative_samples": len(relative),
        "absolute_error_median": value_or_nan(absolute),
        "absolute_error_p95": value_or_nan(absolute, 0.95),
        "absolute_error_max": max(absolute, default=math.nan),
        "relative_error_median_pct": value_or_nan(relative),
        "relative_error_p95_pct": value_or_nan(relative, 0.95),
        "relative_error_max_pct": max(relative, default=math.nan),
        "reference_median": value_or_nan(reference_values),
        "reference_max": max(reference_values, default=math.nan),
    }


def _position_error_samples(
    rows: list[dict[str, str]],
) -> list[dict[str, float | int]]:
    """Return active pose errors with cumulative truth distance per reset.

    The calibration suite intentionally teleports the simulator between
    isolated points.  Reset phases split distance epochs, so a pose error is
    never divided by distance travelled in a different experiment.
    """
    ordered = sorted(
        enumerate(rows),
        key=lambda item: (
            truth_stamp(item[1]) if truth_stamp(item[1]) is not None else math.inf,
            item[0],
        ),
    )
    result: list[dict[str, float | int]] = []
    epoch = 0
    in_reset = False
    distance_m = 0.0
    previous_position: tuple[float, float] | None = None
    previous_stamp: float | None = None

    for order, row in ordered:
        phase = row.get("phase", "")
        if phase_is_diagnostic_reset(phase):
            if not in_reset:
                epoch += 1
            in_reset = True
            distance_m = 0.0
            previous_position = None
            previous_stamp = None
            continue

        in_reset = False
        position = ground_truth_position(row)
        stamp = truth_stamp(row)
        if position[0] is not None and position[1] is not None:
            current_position = (position[0], position[1])
            if previous_position is not None and previous_stamp is not None and stamp is not None:
                dt = stamp - previous_stamp
                step = math.hypot(
                    current_position[0] - previous_position[0],
                    current_position[1] - previous_position[1],
                )
                limit = ground_truth_step_limit_m(dt)
                if limit is not None and step <= limit:
                    distance_m += step
            previous_position = current_position
            previous_stamp = stamp

            odom_x = finite(row.get("x_odom_m"))
            odom_y = finite(row.get("y_odom_m"))
            if odom_x is not None and odom_y is not None:
                result.append({
                    "epoch": epoch,
                    "order": order,
                    "distance_m": distance_m,
                    "error_m": math.hypot(
                        current_position[0] - odom_x,
                        current_position[1] - odom_y,
                    ),
                })
    return result


def _stable_controller_pairs(
    rows: list[dict[str, str]],
    phase_prefix: str,
    target_field: str,
    measured_field: str,
) -> list[tuple[str, float, list[tuple[float, float]]]]:
    """Return target/measured samples from the stable tail of each phase."""
    groups: dict[str, list[tuple[float, float, float]]] = {}
    for row in rows:
        phase = row.get("phase", "")
        if not phase.startswith(phase_prefix):
            continue
        stamp = truth_stamp(row)
        target = finite(row.get(target_field))
        measured = finite(row.get(measured_field))
        if None in (stamp, target, measured):
            continue
        groups.setdefault(phase, []).append((stamp, target, measured))

    result = []
    for phase, samples in groups.items():
        samples.sort(key=lambda item: item[0])
        tail = samples[max(0, int(len(samples) * 0.6)):]
        if not tail:
            continue
        target = statistics.median(item[1] for item in tail)
        result.append((phase, target, [(item[1], item[2]) for item in tail]))
    return sorted(result, key=lambda item: (item[1], item[0]))


def relative_error_metrics(
    rows: list[dict[str, str]],
) -> list[dict[str, float | int | str]]:
    """Report scale-aware odom, pose, speed-target, and accel-target errors.

    Odom velocity percentages use actual simulator speed as the denominator.
    Pose percentages use cumulative simulator distance since the most recent
    reset. Controller rows use their requested target and only the stable tail
    of each isolated command phase. Absolute errors remain beside every
    percentage because percentages near zero are intrinsically ill-conditioned.
    """
    metrics: list[dict[str, float | int | str]] = []

    speed_groups: dict[str, tuple[list[float], list[float]]] = {
        f"{lower:g}-{upper:g}_mps": ([], [])
        for lower, upper in RELATIVE_SPEED_BINS
    }
    for row in rows:
        if phase_is_diagnostic_reset(row.get("phase", "")):
            continue
        truth = _truth_speed(row)
        odom = finite(row.get("speed_mps"))
        if truth is None or odom is None or truth < 0.0:
            continue
        for lower, upper in RELATIVE_SPEED_BINS:
            if lower <= truth < upper or (
                    upper == 23.0 and lower <= truth <= upper):
                errors, references = speed_groups[f"{lower:g}-{upper:g}_mps"]
                errors.append(abs(truth - odom))
                # Below 1 m/s, the percentage is dominated by native source
                # quantisation and is not a useful acceptance measure. Keep
                # the absolute error, but leave the percentage undefined.
                references.append(truth if truth >= 1.0 else math.nan)
                break
    for lower, upper in RELATIVE_SPEED_BINS:
        label = f"{lower:g}-{upper:g}_mps"
        errors, references = speed_groups[label]
        if errors:
            metrics.append(_relative_metric_row(
                "odom_speed_vs_truth", label, errors, references))

    position_samples = _position_error_samples(rows)
    for threshold in POSITION_DISTANCE_THRESHOLDS_M:
        selected = [
            sample for sample in position_samples
            if float(sample["distance_m"]) >= threshold
        ]
        if selected:
            metrics.append(_relative_metric_row(
                "odom_position_vs_truth",
                f"distance_ge_{threshold:g}m",
                [float(sample["error_m"]) for sample in selected],
                [float(sample["distance_m"]) for sample in selected],
            ))

    epochs: dict[int, list[dict[str, float | int]]] = {}
    for sample in position_samples:
        epochs.setdefault(int(sample["epoch"]), []).append(sample)
    for epoch, samples in sorted(epochs.items()):
        endpoint = samples[-1]
        metrics.append(_relative_metric_row(
            "odom_position_endpoint", f"epoch_{epoch}",
            [float(endpoint["error_m"])],
            [float(endpoint["distance_m"])],
        ))

    for phase, target, pairs in _stable_controller_pairs(
            rows, "speed_", "target_speed_mps", "gt_speed_mps"):
        if target <= 0.1:
            continue
        metrics.append(_relative_metric_row(
            "speed_target_vs_truth", phase,
            [abs(target - measured) for _, measured in pairs],
            [target for _ in pairs],
        ))

    for phase, target, pairs in _stable_controller_pairs(
            rows, "acceleration_", "target_accel_mps2", "gt_longitudinal_accel_mps2"):
        metrics.append(_relative_metric_row(
            "acceleration_target_vs_truth", phase,
            [abs(target - measured) for _, measured in pairs],
            [abs(target) for _ in pairs],
        ))
    return metrics


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
        "--max-ground-truth-speed-mps", type=float,
        default=DOCUMENTED_MAX_SPEED_MPS,
        help="maximum physical speed used by timestamp-aware truth validation")
    parser.add_argument(
        "--ground-truth-step-jitter-factor", type=float,
        default=DEFAULT_GROUND_TRUTH_STEP_JITTER_FACTOR,
        help="multiplicative allowance for timestamp jitter in truth steps")
    parser.add_argument(
        "--ground-truth-position-margin-m", type=float,
        default=DEFAULT_GROUND_TRUTH_POSITION_MARGIN_M,
        help="additive position allowance for timestamped truth steps")
    parser.add_argument(
        "--max-ground-truth-gap-s", type=float,
        default=DEFAULT_GROUND_TRUTH_MAX_GAP_S,
        help="largest timestamp gap eligible for truth-distance accumulation")
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
        "--acceleration-envelope-output", type=Path,
        help="write the monotonic full-throttle acceleration ceiling by speed")
    parser.add_argument(
        "--longitudinal-fit-output", type=Path,
        help="write the robust truth-derived throttle/acceleration fit")
    parser.add_argument(
        "--wheel-speed-map-output", type=Path,
        help="write the truth-derived encoder wheel-speed to body-speed slip map")
    parser.add_argument(
        "--slip-model-output", type=Path,
        help="write the truth-derived regime/speed-conditioned slip model")
    parser.add_argument(
        "--frozen-encoder-model-output", type=Path,
        help="write the offline frozen-encoder braking/deceleration model")
    parser.add_argument(
        "--steering-output", type=Path,
        help="write truth-derived direct steering response metrics")
    parser.add_argument(
        "--regime-output", type=Path,
        help="write unique-event truth-vs-odom metrics grouped by motion regime")
    parser.add_argument(
        "--relative-error-output", type=Path,
        help=("write scale-aware odom, pose, speed-target, and "
              "acceleration-target error metrics"))
    args = parser.parse_args()

    rows = read_rows(args.input)
    if not rows:
        raise SystemExit("input contains no telemetry rows")

    # A full-suite recording deliberately resets between every isolated step.
    # Do not discard the later steps as if those diagnostic resets were a
    # failed single run. Legacy recordings without reset phases retain the
    # original reset-prefix handling.
    has_isolated_resets = any(
        row.get("phase") == "reset" or
        row.get("phase") == "boundary_reset" or
        row.get("phase", "").startswith("grid_reset_")
        for row in rows)
    if has_isolated_resets:
        reset_time = reset_kind = None
    else:
        rows, reset_time, reset_kind = trim_after_encoder_reset(rows)
    valid_rows = [row for row in rows if valid_ground_truth_row(row)]
    rejected_ground_truth_rows = len(rows) - len(valid_rows)
    if rejected_ground_truth_rows:
        print(f"ground_truth_boundary_rows_removed={rejected_ground_truth_rows}")
    rows = valid_rows
    analysis_rows, duplicate_gt_rows = deduplicate_source_events(rows)
    rates = [finite(row.get("lidar_rate_hz")) for row in rows]
    rates = [value for value in rates if value is not None and value > 0.0]
    speed_field = preferred_speed_field(rows)
    speed = [finite(row.get(speed_field)) for row in analysis_rows]
    speed = [value for value in speed if value is not None]
    print(f"rows={len(rows)}")
    print(f"unique_gt_odom_rows={len(analysis_rows)}")
    print(f"duplicate_gt_odom_rows_removed={duplicate_gt_rows}")
    print(f"speed_reference={speed_field}")
    explicit_truth_positions = sum(
        finite(row.get("gt_odom_x_m")) is not None and
        finite(row.get("gt_odom_y_m")) is not None
        for row in rows)
    if explicit_truth_positions:
        print("ground_truth_position_reference=timestamped_gt_odom")
    else:
        print("ground_truth_position_reference=legacy_gt_x_m")
        if any((finite(row.get("gt_ips_event_count")) or 0.0) > 0.0 for row in rows):
            print(
                "warning=legacy_gt_x_m_may_mix_timestamped_gt_odom_and_untimestamped_ips; "
                "rerun with the current calibration recorder before accepting position metrics"
            )
    if reset_time is not None:
        print(f"encoder_reset_handled_s={reset_time:.3f} kind={reset_kind}")
    if rates:
        print(f"lidar_rate_hz_median={statistics.median(rates):.3f}")
    for field in (
        "imu_rate_hz", "left_encoder_rate_hz", "right_encoder_rate_hz",
        "odom_rate_hz", "amcl_rate_hz", "ekf_rate_hz",
        "gt_odom_rate_hz", "gt_ips_rate_hz", "collision_rate_hz",
        "speed_command_rate_hz", "acceleration_command_rate_hz",
        "steering_command_rate_hz", "throttle_feedback_rate_hz",
        "steering_feedback_rate_hz",
    ):
        report_field(rows, field)
    if speed:
        print(f"{speed_field}_min={min(speed):.3f} {speed_field}_max={max(speed):.3f}")

    collision_values = [finite(row.get("gt_collision_count")) for row in analysis_rows]
    collision_values = [value for value in collision_values if value is not None]
    if collision_values:
        print(f"ground_truth_collision_count_max={max(collision_values):.0f}")

    phase_metrics = phase_response_metrics(analysis_rows, speed_field)
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

    steering_metrics = steering_response_metrics(analysis_rows)
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

    trace = response_trace(analysis_rows, speed_field)
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

    envelope = acceleration_envelope(
        trace, max_throttle=args.feedforward_max_throttle)
    if envelope:
        print("speed_mps,observed_full_throttle_acceleration_mps2,"
              "monotonic_acceleration_limit_mps2,samples")
        for speed_mps, observed, limit, samples in envelope:
            print(f"{speed_mps:.6f},{observed:.6f},{limit:.6f},{samples}")
    if envelope and args.acceleration_envelope_output:
        args.acceleration_envelope_output.parent.mkdir(parents=True, exist_ok=True)
        with args.acceleration_envelope_output.open(
                "w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow((
                "speed_mps", "observed_full_throttle_acceleration_mps2",
                "monotonic_acceleration_limit_mps2", "samples"))
            writer.writerows(
                (f"{speed_mps:.6f}", f"{observed:.6f}", f"{limit:.6f}", samples)
                for speed_mps, observed, limit, samples in envelope)
        print(f"acceleration_envelope={args.acceleration_envelope_output}")

    longitudinal_fit = longitudinal_acceleration_fit(analysis_rows)
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

    wheel_speed_map = wheel_speed_body_speed_map(analysis_rows)
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

    slip_model = slip_model_metrics(analysis_rows)
    if slip_model:
        slip_fields = (
            "motion_regime", "force_curve_region", "wheel_speed_bin_mps",
            "median_wheel_speed_mps", "median_body_vx_mps",
            "median_slip_speed_mps", "median_slip_ratio", "slip_ratio_mad",
            "slip_ratio_p10", "slip_ratio_p90", "samples")
        print(
            "slip_model_rows="
            f"{len(slip_model)} samples={sum(int(row['samples']) for row in slip_model)}")
        print(",".join(slip_fields))

        def formatted_slip(value: float | int | str) -> str:
            if isinstance(value, str):
                return value
            if isinstance(value, float):
                return f"{value:.6f}"
            return str(value)

        for result in slip_model:
            print(",".join(formatted_slip(result[field]) for field in slip_fields))
        if args.slip_model_output:
            args.slip_model_output.parent.mkdir(parents=True, exist_ok=True)
            with args.slip_model_output.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=slip_fields)
                writer.writeheader()
                writer.writerows(slip_model)
            print(f"slip_model={args.slip_model_output}")

    frozen_model = frozen_encoder_brake_model(analysis_rows)
    if frozen_model is not None:
        model, model_bins = frozen_model
        print(
            "frozen_encoder_brake_model="
            f"intercept_mps2={model['intercept_mps2']:.3f} "
            f"speed_gain_per_s={model['speed_gain_per_s']:.3f} "
            f"fit_rmse_mps2={model['fit_rmse_mps2']:.3f} "
            f"samples={model['samples']} bins={model['bins']}")
        print("speed_bin_mps,median_body_speed_mps,median_deceleration_mps2,samples")
        for result in model_bins:
            print(
                f"{float(result['speed_bin_mps']):.6f},"
                f"{float(result['median_body_speed_mps']):.6f},"
                f"{float(result['median_deceleration_mps2']):.6f},"
                f"{int(result['samples'])}")
        if args.frozen_encoder_model_output:
            args.frozen_encoder_model_output.parent.mkdir(parents=True, exist_ok=True)
            with args.frozen_encoder_model_output.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(("record_type", "parameter", "value", "speed_bin_mps",
                                 "median_body_speed_mps", "median_deceleration_mps2", "samples"))
                for parameter in (
                    "intercept_mps2", "speed_gain_per_s", "max_deceleration_p95_mps2",
                    "fit_rmse_mps2", "samples", "bins"):
                    writer.writerow(("fit", parameter, f"{model[parameter]:.9f}", "", "", "", ""))
                for result in model_bins:
                    writer.writerow((
                        "speed_bin", "", "",
                        f"{float(result['speed_bin_mps']):.9f}",
                        f"{float(result['median_body_speed_mps']):.9f}",
                        f"{float(result['median_deceleration_mps2']):.9f}",
                        int(result["samples"])))
            print(f"frozen_encoder_brake_model_output={args.frozen_encoder_model_output}")

    truth_odom_position_error = []
    active_truth_odom_position_error = []
    truth_odom_speed_error = []
    active_truth_odom_speed_error = []
    for row in analysis_rows:
        gx, gy = ground_truth_position(row)
        ox = finite(row.get("x_odom_m"))
        oy = finite(row.get("y_odom_m"))
        gv = finite(row.get("gt_speed_mps"))
        ov = finite(row.get("speed_mps"))
        if None not in (gx, gy, ox, oy):
            error = math.hypot(gx - ox, gy - oy)
            truth_odom_position_error.append(error)
            if not phase_is_diagnostic_reset(row.get("phase", "")):
                active_truth_odom_position_error.append(error)
        if None not in (gv, ov):
            error = gv - ov
            truth_odom_speed_error.append(error)
            if not phase_is_diagnostic_reset(row.get("phase", "")):
                active_truth_odom_speed_error.append(error)
    if truth_odom_position_error:
        print(
            "ground_truth_vs_odom_position_error_m="
            f"median={statistics.median(truth_odom_position_error):.3f} "
            f"p95={percentile(truth_odom_position_error, 0.95):.3f} "
            f"max={max(truth_odom_position_error):.3f}"
        )
    if active_truth_odom_position_error:
        print(
            "ground_truth_vs_odom_position_error_active_m="
            f"median={statistics.median(active_truth_odom_position_error):.3f} "
            f"p95={percentile(active_truth_odom_position_error, 0.95):.3f} "
            f"max={max(active_truth_odom_position_error):.3f}"
        )
    if truth_odom_speed_error:
        absolute = [abs(value) for value in truth_odom_speed_error]
        print(
            "ground_truth_vs_odom_speed_error_mps="
            f"median={statistics.median(truth_odom_speed_error):.3f} "
            f"abs_p95={percentile(absolute, 0.95):.3f}"
        )
    if active_truth_odom_speed_error:
        absolute = [abs(value) for value in active_truth_odom_speed_error]
        print(
            "ground_truth_vs_odom_speed_error_active_mps="
            f"median={statistics.median(active_truth_odom_speed_error):.3f} "
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
    for row in analysis_rows:
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

    truth_tracking_error, truth_phase_errors = truth_speed_tracking_metrics(analysis_rows)
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

    acceleration_metrics = acceleration_command_metrics(analysis_rows)
    if acceleration_metrics is not None:
        samples, bias, absolute_p95 = acceleration_metrics
        print(
            "acceleration_command_tracking_error_mps2="
            f"bias={bias:.3f} abs_p95={absolute_p95:.3f} samples={samples}"
        )

    relative_metrics = relative_error_metrics(analysis_rows)
    if relative_metrics:
        print(",".join(RELATIVE_ERROR_FIELDS))

        def format_relative(value: float | int | str) -> str:
            if isinstance(value, str):
                return value
            if isinstance(value, float):
                return f"{value:.6f}"
            return str(value)

        for result in relative_metrics:
            print(",".join(format_relative(result[field]) for field in RELATIVE_ERROR_FIELDS))
        if args.relative_error_output:
            args.relative_error_output.parent.mkdir(parents=True, exist_ok=True)
            with args.relative_error_output.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=RELATIVE_ERROR_FIELDS)
                writer.writeheader()
                writer.writerows(relative_metrics)
            print(f"relative_error_metrics={args.relative_error_output}")

    amcl_ekf = pose_difference(analysis_rows, "amcl", "ekf")
    if amcl_ekf:
        print(
            "amcl_ekf_xy_difference_m="
            f"median={statistics.median(amcl_ekf):.3f} "
            f"p95={percentile(amcl_ekf, 0.95):.3f} max={max(amcl_ekf):.3f}"
        )

    if args.trajectory:
        trajectory = read_trajectory(args.trajectory)
        path_error, seam_crossings = raceline_metrics(analysis_rows, trajectory)
        if path_error:
            print(
                "ekf_raceline_error_m="
                f"median={statistics.median(path_error):.3f} "
                f"p95={percentile(path_error, 0.95):.3f} max={max(path_error):.3f}"
            )
            print(f"raceline_forward_seam_crossings={seam_crossings}")

    if args.ground_truth:
        metrics = encoder_distance_metrics(
            analysis_rows, args.ground_truth, args.max_encoder_step_rad,
            args.max_ground_truth_speed_mps,
            args.ground_truth_step_jitter_factor,
            args.ground_truth_position_margin_m,
            args.max_ground_truth_gap_s)
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

    full_table = throttle_table(analysis_rows, speed_field)
    table = throttle_table(analysis_rows, speed_field, args.feedforward_max_throttle)
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

    regimes = motion_regime_metrics(analysis_rows)
    if regimes:
        print(
            "motion_regime,events,duration_s,truth_speed_median_mps,"
            "odom_speed_mae_mps,odom_speed_p95_abs_error_mps,error_samples"
        )
        fields = (
            "regime", "events", "duration_s", "truth_speed_median_mps",
            "odom_speed_mae_mps", "odom_speed_p95_abs_error_mps",
            "error_samples")

        def formatted(value: float | int | str) -> str:
            if isinstance(value, str):
                return value
            if isinstance(value, float):
                return f"{value:.6f}"
            return str(value)

        for result in regimes:
            print(",".join(formatted(result[field]) for field in fields))
        if args.regime_output:
            args.regime_output.parent.mkdir(parents=True, exist_ok=True)
            with args.regime_output.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(regimes)
            print(f"motion_regime_metrics={args.regime_output}")


if __name__ == "__main__":
    main()
