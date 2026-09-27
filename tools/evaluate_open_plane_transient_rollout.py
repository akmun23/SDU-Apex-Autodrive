#!/usr/bin/env python3
"""Train/holdout test for a delay-aware empirical 4 m/s response model.

The training set is transient sweep repetitions 1--3; repetition 4 is never
used for parameter fitting or equilibrium maps. Recorded actual steering and
speed are conditional inputs, and each held-out phase starts from measured
legal odometry state. This tests the plant model, not closed-loop MPC.

Acceptance fixed before the repetition-4 evaluation: compared with the current
configured yaw law plus held lateral velocity, the fitted empirical yaw and
lateral-velocity states must each reduce RMSE by >=20% at three or more of the
25/125/250/500/750 ms horizons, with no horizon regressing by >10%. Parameters
must be finite; delay is quantized at 25 ms because the measured stream is
40 Hz. No runtime code is changed by this evaluator.
"""

from __future__ import annotations

import argparse
import bisect
import math
import re
import sqlite3
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    import analyze_open_plane_dynamics as common
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
except ImportError as exc:  # pragma: no cover - requires ROS 2 runtime
    raise SystemExit(f"ROS 2 Python modules are required: {exc}") from exc


ODOM = common.ODOM
STEERING = common.STEERING
COLLISIONS = common.COLLISIONS
TIMING_FAULT = common.TIMING_FAULT
PHASE = common.PHASE
KNOTS_RAD = (0.30, 0.42, 0.46, 0.50)
HORIZONS_S = (0.025, 0.125, 0.250, 0.500, 0.750)
TRAIN_REPETITIONS = (1, 2, 3)
HOLDOUT_REPETITION = 4
FIT_WINDOW_S = 0.750
TAUS_S = tuple(i / 1000.0 for i in range(5, 301, 5))
SHAPES = tuple(i / 4.0 for i in range(2, 17))
DELAYS = (0, 1, 2, 3)
PHASE_RE = re.compile(
    r"^transient_r(?P<rep>[1-4])_(?P<sign>[+-]1)_(?P<direction>up|down)_"
    r"(?P<angle>0\.\d+)_4\.0mps$")


@dataclass(frozen=True)
class Sample:
    time_s: float
    speed_mps: float
    vy_mps: float
    yaw_rate_rps: float
    steering_rad: float
    pose_x_m: float
    pose_y_m: float
    pose_yaw_rad: float


@dataclass(frozen=True)
class Probe:
    repetition: int
    commanded_steering_rad: float
    initial_steering_rad: float
    samples: tuple[Sample, ...]


def _decode(connection: sqlite3.Connection, topic_id: int, msg_type: str):
    cls = get_message(msg_type)
    for timestamp, payload in connection.execute(
            "SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp, id",
            (topic_id,)):
        yield int(timestamp), deserialize_message(bytes(payload), cls)


def load(path: Path) -> tuple[list[Probe], dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"bag database does not exist: {path}")
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        topics = common._topic_map(connection)
        required = (ODOM, STEERING, COLLISIONS, TIMING_FAULT, PHASE)
        missing = [name for name in required if name not in topics]
        if missing:
            raise ValueError("missing topic(s): " + ", ".join(missing))
        phases, experiment_end = common._phase_events(connection, topics)

        odom_id, odom_type = topics[ODOM]
        odometry = []
        for receipt_ns, message in _decode(connection, odom_id, odom_type):
            twist = message.twist.twist
            pose = message.pose.pose
            quaternion = pose.orientation
            pose_yaw = math.atan2(
                2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
                1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z))
            vx_com, vy_com = (float(twist.linear.x), float(twist.linear.y))
            yaw_rate = float(twist.angular.z)
            vx, vy = common._rear_axle_velocity(vx_com, vy_com, yaw_rate)
            values = (vx, vy, yaw_rate, float(pose.position.x),
                      float(pose.position.y), pose_yaw)
            if all(math.isfinite(x) for x in values):
                odometry.append((receipt_ns, *values))

        steer_id, steer_type = topics[STEERING]
        steering = [(receipt_ns, float(message.data))
                    for receipt_ns, message in _decode(connection, steer_id, steer_type)
                    if math.isfinite(float(message.data))]
        steering_times = [row[0] for row in steering]

        collision_id, collision_type = topics[COLLISIONS]
        collisions = [int(message.data) for _, message in _decode(
            connection, collision_id, collision_type)]
        fault_id, fault_type = topics[TIMING_FAULT]
        faults = [bool(message.data) for _, message in _decode(
            connection, fault_id, fault_type)]
    finally:
        connection.close()

    probes: list[Probe] = []
    valid_count = 0
    matched_starts = 0
    for phase in phases:
        match = PHASE_RE.match(phase.label)
        if not match:
            continue
        if phase.valid is True:
            valid_count += 1
        if (phase.initial_window_speed_mps is not None
                and phase.initial_window_abs_vy_mps is not None
                and phase.initial_window_abs_yaw_rate_rps is not None
                and phase.initial_steering_rad is not None
                and abs(phase.initial_window_speed_mps - 4.0) <= 0.20
                and phase.initial_window_abs_vy_mps <= 0.08
                and phase.initial_window_abs_yaw_rate_rps <= 0.12
                and abs(phase.initial_steering_rad) <= 0.02):
            matched_starts += 1

        block = []
        for receipt_ns, speed, vy, yaw_rate, pose_x, pose_y, pose_yaw in odometry:
            if receipt_ns < phase.start_ns or receipt_ns > phase.end_ns:
                continue
            index = bisect.bisect_left(steering_times, receipt_ns)
            candidates = [i for i in (index - 1, index) if 0 <= i < len(steering)]
            if not candidates:
                continue
            selected = min(candidates, key=lambda i: abs(steering_times[i] - receipt_ns))
            if abs(steering_times[selected] - receipt_ns) > 30_000_000:
                continue
            block.append(Sample((receipt_ns - phase.start_ns) / 1e9,
                                speed, vy, yaw_rate, steering[selected][1],
                                pose_x, pose_y, pose_yaw))
        # A 1.20 s command phase at a nominal 40 Hz can contain fewer than
        # 40 odometry samples because the actuator command/phase transition
        # consumes part of the interval. Keep only the requirements needed
        # by the 750 ms holdout horizon: a near-start state and dense enough
        # samples across the horizon.
        spans_rollout_horizon = (
            len(block) >= 20 and block[0].time_s <= 0.075
            and block[-1].time_s >= 0.785)
        if phase.valid is True and spans_rollout_horizon:
            sign = 1 if match.group("sign") == "+1" else -1
            target = sign * float(match.group("angle"))
            if abs(target - phase.commanded_steering_rad) > 1e-8:
                raise ValueError(f"command/label mismatch: {phase.label}")
            probes.append(Probe(int(match.group("rep")), target,
                                phase.initial_steering_rad or 0.0, tuple(block)))

    quality = {
        "valid_steps": valid_count,
        "matched_starts": matched_starts,
        "collisions": (collisions[0], collisions[-1]) if collisions else None,
        "timing_faults": sum(faults),
        "aborted": bool(experiment_end.get("aborted", True)),
        "probe_counts": {rep: sum(p.repetition == rep for p in probes)
                         for rep in (1, 2, 3, 4)},
    }
    return probes, quality


def _median(values: list[float]) -> float:
    return statistics.median(values)


def _training_maps(probes: list[Probe]) -> tuple[dict[float, float], dict[float, float]]:
    yaw_samples: dict[float, list[float]] = {angle: [] for angle in KNOTS_RAD}
    vy_samples: dict[float, list[float]] = {angle: [] for angle in KNOTS_RAD}
    for probe in probes:
        if probe.repetition not in TRAIN_REPETITIONS:
            continue
        angle = round(abs(probe.commanded_steering_rad), 2)
        if angle not in yaw_samples:
            continue
        settled = [sample for sample in probe.samples if sample.time_s >= 0.50]
        sign = 1.0 if probe.commanded_steering_rad > 0.0 else -1.0
        yaw_samples[angle].extend(
            sample.yaw_rate_rps / (sample.speed_mps * math.tan(sample.steering_rad))
            for sample in settled
            if sample.speed_mps > 1.0 and abs(sample.steering_rad) > 0.1)
        vy_samples[angle].extend(sign * sample.vy_mps for sample in settled)
    if any(not yaw_samples[angle] or not vy_samples[angle] for angle in KNOTS_RAD):
        raise ValueError("training repetitions do not cover all steering knots")
    return ({angle: _median(values) for angle, values in yaw_samples.items()},
            {angle: _median(values) for angle, values in vy_samples.items()})


def _interp(x: float, knots: tuple[float, ...], values: tuple[float, ...]) -> float:
    if x <= knots[0]:
        return values[0]
    if x >= knots[-1]:
        return values[-1]
    for index, (left, right) in enumerate(zip(knots, knots[1:])):
        if left <= x <= right:
            fraction = (x - left) / (right - left)
            return values[index] + fraction * (values[index + 1] - values[index])
    raise AssertionError("invalid interpolation interval")


def _yaw_target(speed: float, steering: float, shape: float,
                knots: dict[float, float]) -> float:
    magnitude = abs(steering)
    if magnitude < KNOTS_RAD[0]:
        gain0 = 2.95
        gain1 = knots[KNOTS_RAD[0]]
        gain = gain0 + (gain1 - gain0) * (magnitude / KNOTS_RAD[0]) ** shape
    else:
        gain = _interp(magnitude, KNOTS_RAD,
                       tuple(knots[angle] for angle in KNOTS_RAD))
    return speed * math.tan(steering) * gain


def _vy_target(steering: float, shape: float,
               knots: dict[float, float]) -> float:
    magnitude = abs(steering)
    sign = 1.0 if steering >= 0.0 else -1.0
    if magnitude < KNOTS_RAD[0]:
        return sign * knots[KNOTS_RAD[0]] * (
            magnitude / KNOTS_RAD[0]) ** shape
    return sign * _interp(magnitude, KNOTS_RAD,
                          tuple(knots[angle] for angle in KNOTS_RAD))


def _configured_target(speed: float, steering: float) -> float:
    gain = 2.95 - 35.6 * min(max(abs(steering) - 0.41, 0.0), 0.05)
    return speed * math.tan(steering) * gain


def _simulate(probe: Probe, end_index: int, *,
              yaw_knots: dict[float, float], vy_knots: dict[float, float],
              yaw_tau: float, yaw_shape: float, yaw_delay: int,
              vy_tau: float | None, vy_shape: float, vy_delay: int,
              configured: bool = False) -> tuple[float, float]:
    yaw_rate = probe.samples[0].yaw_rate_rps
    vy = probe.samples[0].vy_mps
    for index in range(1, end_index + 1):
        previous = probe.samples[index - 1]
        sample = probe.samples[index]
        dt = sample.time_s - previous.time_s
        if dt <= 0.0 or dt > 0.075:
            raise ValueError(f"invalid stream interval {dt:.4f}s")
        yaw_input_index = index - yaw_delay
        yaw_steer = (probe.initial_steering_rad if yaw_input_index < 0
                     else probe.samples[yaw_input_index].steering_rad)
        target_r = (_configured_target(sample.speed_mps, yaw_steer) if configured
                    else _yaw_target(sample.speed_mps, yaw_steer,
                                     yaw_shape, yaw_knots))
        retention = math.exp(-dt / yaw_tau)
        yaw_rate = retention * yaw_rate + (1.0 - retention) * target_r

        if vy_tau is not None:
            vy_input_index = index - vy_delay
            vy_steer = (probe.initial_steering_rad if vy_input_index < 0
                        else probe.samples[vy_input_index].steering_rad)
            target_vy = _vy_target(vy_steer, vy_shape, vy_knots)
            retention_vy = math.exp(-dt / vy_tau)
            vy = retention_vy * vy + (1.0 - retention_vy) * target_vy
    return yaw_rate, vy


def _fit_one(probes: list[Probe], *, kind: str,
             knots: dict[float, float]) -> tuple[float, float, int]:
    training = [probe for probe in probes if probe.repetition in TRAIN_REPETITIONS]
    best = (math.inf, 0.05, 1.0, 0)
    for tau in TAUS_S:
        for shape in SHAPES:
            for delay in DELAYS:
                squared = 0.0
                count = 0
                for probe in training:
                    indexes = [i for i, sample in enumerate(probe.samples)
                               if sample.time_s <= FIT_WINDOW_S]
                    if not indexes:
                        continue
                    prediction = (probe.samples[indexes[0]].yaw_rate_rps
                                  if kind == "yaw" else probe.samples[indexes[0]].vy_mps)
                    for left_index, index in zip(indexes, indexes[1:]):
                        previous, sample = probe.samples[left_index], probe.samples[index]
                        dt = sample.time_s - previous.time_s
                        input_index = index - delay
                        steer = (probe.initial_steering_rad if input_index < 0
                                 else probe.samples[input_index].steering_rad)
                        if kind == "yaw":
                            target = _yaw_target(sample.speed_mps, steer, shape, knots)
                            observed = sample.yaw_rate_rps
                        else:
                            target = _vy_target(steer, shape, knots)
                            observed = sample.vy_mps
                        retention = math.exp(-dt / tau)
                        prediction = retention * prediction + (1.0 - retention) * target
                        squared += (prediction - observed) ** 2
                        count += 1
                score = squared / count if count else math.inf
                if score < best[0]:
                    best = score, tau, shape, delay
    return best[1], best[2], best[3]


def _score(probes: list[Probe], horizons: tuple[float, ...], *,
           yaw_knots: dict[float, float], vy_knots: dict[float, float],
           yaw_tau: float, yaw_shape: float, yaw_delay: int,
           vy_tau: float | None, vy_shape: float, vy_delay: int,
           configured: bool = False) -> dict[float, tuple[float, float, int]]:
    result = {}
    holdout = [probe for probe in probes if probe.repetition == HOLDOUT_REPETITION]
    for horizon in horizons:
        yaw_errors: list[float] = []
        vy_errors: list[float] = []
        for probe in holdout:
            index = min(range(len(probe.samples)),
                        key=lambda i: abs(probe.samples[i].time_s - horizon))
            if abs(probe.samples[index].time_s - horizon) > 0.035:
                continue
            predicted_yaw, predicted_vy = _simulate(
                probe, index, yaw_knots=yaw_knots, vy_knots=vy_knots,
                yaw_tau=yaw_tau, yaw_shape=yaw_shape, yaw_delay=yaw_delay,
                vy_tau=vy_tau, vy_shape=vy_shape, vy_delay=vy_delay,
                configured=configured)
            actual = probe.samples[index]
            yaw_errors.append(predicted_yaw - actual.yaw_rate_rps)
            vy_errors.append(predicted_vy - actual.vy_mps)
        result[horizon] = (
            math.sqrt(sum(x * x for x in yaw_errors) / len(yaw_errors)),
            math.sqrt(sum(x * x for x in vy_errors) / len(vy_errors)),
            len(yaw_errors))
    return result


def evaluate(path: Path) -> bool:
    probes, quality = load(path)
    expected = {1: 14, 2: 14, 3: 14, 4: 14}
    if (quality["valid_steps"] != 56 or quality["matched_starts"] != 8
            or quality["collisions"] != (0, 0) or quality["timing_faults"] != 0
            or quality["aborted"] or quality["probe_counts"] != expected):
        raise ValueError(f"run does not satisfy the predeclared data gate: {quality}")

    yaw_knots, vy_knots = _training_maps(probes)
    yaw_tau, yaw_shape, yaw_delay = _fit_one(probes, kind="yaw", knots=yaw_knots)
    vy_tau, vy_shape, vy_delay = _fit_one(probes, kind="vy", knots=vy_knots)
    baseline = _score(
        probes, HORIZONS_S, yaw_knots=yaw_knots, vy_knots=vy_knots,
        yaw_tau=0.015, yaw_shape=1.0, yaw_delay=0,
        vy_tau=None, vy_shape=1.0, vy_delay=0, configured=True)
    candidate = _score(
        probes, HORIZONS_S, yaw_knots=yaw_knots, vy_knots=vy_knots,
        yaw_tau=yaw_tau, yaw_shape=yaw_shape, yaw_delay=yaw_delay,
        vy_tau=vy_tau, vy_shape=vy_shape, vy_delay=vy_delay)

    yaw_gains: list[float] = []
    vy_gains: list[float] = []
    print(f"bag: {path}")
    print(f"training repetitions={TRAIN_REPETITIONS}; untouched holdout={HOLDOUT_REPETITION}; "
          f"holdout probes={quality['probe_counts'][HOLDOUT_REPETITION]}")
    print("training-only steady response map:")
    for angle in KNOTS_RAD:
        print(f"  |steer|={angle:.2f}: K_yaw={yaw_knots[angle]:.4f} 1/m, "
              f"signed v_y={vy_knots[angle]:.4f} m/s")
    print(f"fitted yaw: tau={yaw_tau:.3f}s, input_delay={yaw_delay*25}ms, "
          f"sub-0.30rad gain_shape={yaw_shape:.2f}")
    print(f"fitted v_y: tau={vy_tau:.3f}s, input_delay={vy_delay*25}ms, "
          f"sub-0.30rad shape={vy_shape:.2f}")
    print("rep-4 conditional open-loop endpoint RMSE (n=14 each horizon):")
    print("ms  yaw configured/fitted (rad/s) gain  vy held/fitted (m/s) gain")
    for horizon in HORIZONS_S:
        base_r, base_v, n = baseline[horizon]
        fit_r, fit_v, fit_n = candidate[horizon]
        if n != 14 or fit_n != 14:
            raise ValueError(f"incomplete holdout scoring at {horizon:.3f}s: {n}/{fit_n}")
        yaw_gain = 1.0 - fit_r / base_r if base_r > 0.0 else 0.0
        vy_gain = 1.0 - fit_v / base_v if base_v > 0.0 else 0.0
        yaw_gains.append(yaw_gain)
        vy_gains.append(vy_gain)
        print(f"{horizon*1000:3.0f}  {base_r:.5f}/{fit_r:.5f} {yaw_gain:+.1%}  "
              f"{base_v:.5f}/{fit_v:.5f} {vy_gain:+.1%}")

    yaw_pass = sum(x >= 0.20 for x in yaw_gains) >= 3 and min(yaw_gains) >= -0.10
    vy_pass = sum(x >= 0.20 for x in vy_gains) >= 3 and min(vy_gains) >= -0.10
    finite = all(math.isfinite(x) for x in (yaw_tau, yaw_shape, vy_tau, vy_shape))
    print(f"acceptance: yaw={'PASS' if yaw_pass else 'FAIL'}, "
          f"lateral_velocity={'PASS' if vy_pass else 'FAIL'}, finite={finite}")
    return yaw_pass and vy_pass and finite


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path, help="path to run_0.db3")
    args = parser.parse_args()
    return 0 if evaluate(args.bag) else 1


if __name__ == "__main__":
    raise SystemExit(main())
