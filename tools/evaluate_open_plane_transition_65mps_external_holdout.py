#!/usr/bin/env python3
"""Blindly validate a frozen 6.5 m/s direct-yaw response curve on a new run.

Train only on repetitions 1-2 of the original fine-grid capture. The
confirmatory run's 48 probes must be valid and matched; the frozen curve must
achieve <=0.01 rad/s steady yaw-rate RMSE, and every held-out interval with
|measured d|r|/d|steer|| >=0.10 s^-1 must have the correct derivative sign.
It also fits one first-order yaw time constant on the original transient
traces and recursively scores the independent run. That is a yaw-submodel
test only; it never authorizes MPC integration or claims a complete u,v,r
plant.
"""

from __future__ import annotations

import argparse
import math
import re
import sqlite3
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tools import analyze_open_plane_dynamics as common
from tools import evaluate_open_plane_highspeed_crossfactor as crossfactor
from tools import evaluate_open_plane_yaw_spline as yaw_spline


SPEED_MPS = 6.5
ANGLES = tuple(round(0.145 + 0.005 * index, 3) for index in range(8))
TRAIN_REPETITIONS = (1, 2)
TEST_REPETITIONS = (1, 2, 3)
MIN_SPEED_ERROR_MPS = 0.20
MIN_SLOPE_MAGNITUDE = 0.10
MIN_RESOLVED_INTERVALS = 24
MAX_RATE_RMSE = 0.01
ROLLOUT_HORIZONS_S = (0.025, 0.125, 0.250, 0.500, 0.750)
ROLLOUT_TRAIN_REPETITIONS = (1, 2)
ROLLOUT_TEST_REPETITIONS = (1, 2, 3)
PRODUCTION_TAU_S = 0.015
TAU_GRID_S = tuple(index / 1000.0 for index in range(5, 201))
MIN_ROLLOUT_IMPROVEMENT = 0.20
MAX_ROLLOUT_REGRESSION = 0.10
NEAR_ZERO_BASELINE_RMSE = 0.01
PHASE_RE = re.compile(
    r"^isolated_r(?P<repetition>[123])_6\.5mps_steer_"
    r"(?P<steering>[+-]\d+\.\d+)rad$")


@dataclass(frozen=True)
class MotionSample:
    time_s: float
    forward_speed_mps: float
    yaw_rate_rps: float
    steering_rad: float


@dataclass(frozen=True)
class ProbeTrace:
    repetition: int
    steering_command_rad: float
    samples: tuple[MotionSample, ...]


def _index_rows(rows: list[dict]) -> dict[tuple[int, int, float], dict]:
    return {
        (row["repetition"],
         1 if row["steering_command"] > 0.0 else -1,
         round(abs(row["steering_command"]), 3)): row
        for row in rows
        if row["mode"] == "speed_hold"
        and abs(row["speed_target"] - SPEED_MPS) < 1e-6
    }


def _load_probe_traces(path: Path) -> list[ProbeTrace]:
    """Load causal 40 Hz yaw/steering traces for the fine 6.5 m/s probes."""
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        topics = common._topic_map(connection)
        required = (common.ODOM, common.STEERING, common.PHASE)
        missing = [name for name in required if name not in topics]
        if missing:
            raise ValueError("missing trace topic(s): " + ", ".join(missing))

        odometry = []
        for receipt_ns, message in common._messages(connection, topics, common.ODOM):
            twist = message.twist.twist
            forward_speed = float(twist.linear.x)
            yaw_rate = float(twist.angular.z)
            if math.isfinite(forward_speed) and math.isfinite(yaw_rate):
                odometry.append((receipt_ns, forward_speed, yaw_rate))

        steering = [
            common.ScalarRow(receipt_ns, receipt_ns, float(message.data))
            for receipt_ns, message in common._messages(
                connection, topics, common.STEERING)
            if math.isfinite(float(message.data))
        ]
        steering_times = [row.receipt_ns for row in steering]
        phases, _ = common._phase_events(connection, topics)
    finally:
        connection.close()

    traces = []
    for phase in phases:
        match = PHASE_RE.match(phase.label)
        if (match is None or phase.valid is not True
                or abs(phase.target_speed_mps - SPEED_MPS) > 1e-6):
            continue
        block = []
        for receipt_ns, forward_speed, yaw_rate in odometry:
            if receipt_ns < phase.start_ns or receipt_ns > phase.end_ns:
                continue
            steer_row = common._nearest_scalar(
                steering, steering_times, receipt_ns)
            if steer_row is None:
                continue
            block.append(MotionSample(
                (receipt_ns - phase.start_ns) / 1e9,
                forward_speed, yaw_rate, steer_row.value))
        if (len(block) < 25 or block[0].time_s > 0.075
                or block[-1].time_s < 0.80):
            raise ValueError(f"incomplete transient trace: {phase.label}")
        traces.append(ProbeTrace(
            int(match.group("repetition")),
            float(match.group("steering")), tuple(block)))

    expected = {
        (repetition, sign, angle)
        for repetition in (1, 2, 3)
        for sign in (-1, 1)
        for angle in ANGLES
    }
    present = {
        (trace.repetition,
         1 if trace.steering_command_rad > 0 else -1,
         round(abs(trace.steering_command_rad), 3))
        for trace in traces
    }
    if present != expected or len(traces) != len(expected):
        raise ValueError(
            f"trace coverage mismatch: {len(present)}/{len(expected)} probes")
    return traces


def _direct_target(curves: dict[int, tuple[np.ndarray, np.ndarray]],
                   steering_rad: float) -> float:
    sign = 1 if steering_rad >= 0.0 else -1
    x, y = curves[sign]
    magnitude = abs(steering_rad)
    if magnitude > x[-1] + 0.002:
        raise ValueError(f"steering outside trained curve: {magnitude:.4f} rad")
    if magnitude < x[1]:
        # The zero-response origin is a symmetry boundary, not an extrapolated
        # tire parameter. The measured range begins at 0.145 rad.
        return sign * float(y[1] * magnitude / x[1])
    value, _ = yaw_spline._pchip_value_and_slope(x, y, min(magnitude, x[-1]))
    return sign * value


def _rollout_trace(trace: ProbeTrace, tau_s: float,
                   curves: dict[int, tuple[np.ndarray, np.ndarray]] | None
                   ) -> list[float]:
    samples = trace.samples
    prediction = [samples[0].yaw_rate_rps]
    for previous, current in zip(samples, samples[1:]):
        dt = current.time_s - previous.time_s
        if not 0.0 < dt <= 0.075:
            raise ValueError(f"invalid trace interval {dt:.4f}s")
        if curves is None:
            gain = yaw_spline._configured_gain(previous.steering_rad)
            target = (abs(previous.forward_speed_mps)
                      * math.tan(previous.steering_rad) * gain)
        else:
            target = _direct_target(curves, previous.steering_rad)
        retention = math.exp(-dt / tau_s)
        prediction.append(retention * prediction[-1] + (1.0 - retention) * target)
    return prediction


def _fit_tau(traces: list[ProbeTrace],
             curves: dict[int, tuple[np.ndarray, np.ndarray]]) -> float:
    training = [trace for trace in traces
                if trace.repetition in ROLLOUT_TRAIN_REPETITIONS]
    best_tau = math.nan
    best_sse = math.inf
    for tau_s in TAU_GRID_S:
        squared_error = 0.0
        samples_scored = 0
        for trace in training:
            predicted = _rollout_trace(trace, tau_s, curves)
            for sample, estimate in zip(trace.samples, predicted):
                if 0.05 <= sample.time_s <= 0.75:
                    squared_error += (estimate - sample.yaw_rate_rps) ** 2
                    samples_scored += 1
        if samples_scored == 0:
            raise ValueError("no training transient samples in fit window")
        if squared_error < best_sse:
            best_sse, best_tau = squared_error, tau_s
    if not math.isfinite(best_tau):
        raise ValueError("could not fit a finite yaw response time constant")
    return best_tau


def _yaw_rollout_gate(training_path: Path, test_path: Path,
                      training_rows: list[dict],
                      run_end: dict) -> bool:
    training_traces = _load_probe_traces(training_path)
    test_traces = _load_probe_traces(test_path)
    curves: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    indexed_training = _index_rows(training_rows)
    for sign in (-1, 1):
        values = [indexed_training[(rep, sign, angle)]["yaw_rate"]
                  for angle in ANGLES for rep in ROLLOUT_TRAIN_REPETITIONS]
        if len(values) != 2 * len(ANGLES) or not all(map(math.isfinite, values)):
            print("yaw transient rollout: REJECT; incomplete finite training surface")
            return False
        means = [statistics.mean(
            sign * indexed_training[(rep, sign, angle)]["yaw_rate"]
            for rep in ROLLOUT_TRAIN_REPETITIONS)
            for angle in ANGLES]
        curves[sign] = (np.asarray((0.0, *ANGLES)),
                        np.asarray((0.0, *means)))

    tau_s = _fit_tau(training_traces, curves)
    scores: dict[float, dict[str, list[float]]] = {
        horizon: {"candidate": [], "production": []}
        for horizon in ROLLOUT_HORIZONS_S
    }
    for trace in test_traces:
        candidate = _rollout_trace(trace, tau_s, curves)
        production = _rollout_trace(trace, PRODUCTION_TAU_S, None)
        for horizon in ROLLOUT_HORIZONS_S:
            index = min(range(len(trace.samples)),
                        key=lambda i: abs(trace.samples[i].time_s - horizon))
            sample = trace.samples[index]
            if abs(sample.time_s - horizon) > 0.035:
                raise ValueError(f"no sample near {horizon:.3f}s in held-out trace")
            scores[horizon]["candidate"].append(
                candidate[index] - sample.yaw_rate_rps)
            scores[horizon]["production"].append(
                production[index] - sample.yaw_rate_rps)

    improvements = []
    no_bad_regression = True
    print("\nindependent recursive yaw-state rollout (actual speed/steering are "
          "conditional inputs; start yaw is measured once):")
    print(f"  fitted tau on original reps 1–2: {tau_s:.3f}s; "
          f"production tau: {PRODUCTION_TAU_S:.3f}s")
    for horizon in ROLLOUT_HORIZONS_S:
        candidate_rmse = math.sqrt(statistics.mean(
            error * error for error in scores[horizon]["candidate"]))
        production_rmse = math.sqrt(statistics.mean(
            error * error for error in scores[horizon]["production"]))
        improvement = (1.0 - candidate_rmse / production_rmse
                       if production_rmse > 1e-12 else 0.0)
        improvements.append(improvement)
        if production_rmse < NEAR_ZERO_BASELINE_RMSE:
            horizon_non_regression = candidate_rmse <= NEAR_ZERO_BASELINE_RMSE
        else:
            horizon_non_regression = improvement >= -MAX_ROLLOUT_REGRESSION
        no_bad_regression &= horizon_non_regression
        print(f"  {horizon*1000:>3.0f} ms: candidate={candidate_rmse:.4f}, "
              f"production={production_rmse:.4f} rad/s, "
              f"improvement={improvement:+.1%}, "
              f"non-regression={'PASS' if horizon_non_regression else 'FAIL'}")
    passed = (not run_end.get("aborted", True)
              and tau_s > TAU_GRID_S[0] + 1e-9
              and tau_s < TAU_GRID_S[-1] - 1e-9
              and sum(value >= MIN_ROLLOUT_IMPROVEMENT
                      for value in improvements) >= 3
              and no_bad_regression)
    print("  decision: " + ("PASS yaw submodel for subsequent coupled-state testing"
                          if passed else
                          "REJECT yaw transient model for MPC promotion"))
    if tau_s <= TAU_GRID_S[0] + 1e-9:
        print("  tau identification: lower search bound reached; do not interpret "
              "the fitted value as a resolved physical relaxation constant")
    return passed


def evaluate(training_path: Path, test_path: Path,
             training_quality_ok: bool, test_quality_ok: bool) -> bool:
    training_rows, _ = crossfactor.load_blocks(training_path)
    test_rows, run_end = crossfactor.load_blocks(test_path)
    training = _index_rows(training_rows)
    test = _index_rows(test_rows)

    expected_train = {(rep, sign, angle)
                      for rep in TRAIN_REPETITIONS
                      for sign in (-1, 1) for angle in ANGLES}
    expected_test = {(rep, sign, angle)
                     for rep in TEST_REPETITIONS
                     for sign in (-1, 1) for angle in ANGLES}
    train_complete = set(training) >= expected_train
    test_complete = set(test) == expected_test
    train_surfaces: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for sign in (-1, 1):
        if not train_complete:
            break
        values = []
        for angle in ANGLES:
            blocks = [training[(rep, sign, angle)]
                      for rep in TRAIN_REPETITIONS]
            values.extend(sign * block["yaw_rate"] for block in blocks)
        if not all(math.isfinite(value) for value in values):
            train_complete = False
            break
        y = np.asarray([
            statistics.mean(sign * training[(rep, sign, angle)]["yaw_rate"]
                            for rep in TRAIN_REPETITIONS)
            for angle in ANGLES
        ])
        train_surfaces[sign] = (np.asarray(ANGLES), y)

    phase_quality = test_complete and all(
        row["valid"] is True
        and math.isfinite(row["start_speed"])
        and abs(row["start_speed"] - SPEED_MPS) <= MIN_SPEED_ERROR_MPS
        and abs(row["speed"] - SPEED_MPS) <= MIN_SPEED_ERROR_MPS
        for row in test.values()
    )
    errors: list[float] = []
    resolved = 0
    sign_matches = 0
    slope_rows: list[tuple[int, float, float, float]] = []
    if train_complete and test_complete:
        for sign in (-1, 1):
            x, y = train_surfaces[sign]
            for rep in TEST_REPETITIONS:
                response: dict[float, float] = {}
                for angle in ANGLES:
                    block = test[(rep, sign, angle)]
                    observed = sign * block["yaw_rate"]
                    predicted, _ = yaw_spline._pchip_value_and_slope(x, y, angle)
                    response[angle] = observed
                    errors.append((predicted - observed) ** 2)
                for left_angle, right_angle in zip(ANGLES, ANGLES[1:]):
                    measured = ((response[right_angle] - response[left_angle])
                                / (right_angle - left_angle))
                    middle = 0.5 * (left_angle + right_angle)
                    _value, predicted_slope = yaw_spline._pchip_value_and_slope(
                        x, y, middle)
                    slope_rows.append((sign, middle, measured, predicted_slope))
                    if abs(measured) >= MIN_SLOPE_MAGNITUDE:
                        resolved += 1
                        sign_matches += measured * predicted_slope > 0.0

    rmse = math.sqrt(statistics.mean(errors)) if errors else math.inf
    jacobian_ok = (resolved >= MIN_RESOLVED_INTERVALS
                   and sign_matches == resolved)
    print(f"training bag: {training_path}")
    print(f"independent test bag: {test_path}")
    print(f"run_end: aborted={run_end.get('aborted')}, reason={run_end.get('reason')!r}")
    train_points_present = len(set(training) & expected_train)
    print(f"probe coverage: training={train_points_present}/{len(expected_train)} "
          f"required fit points; independent test={len(test)}/{len(expected_test)}")
    print(f"test phase quality: {'PASS' if phase_quality else 'FAIL'}")
    print(f"frozen direct-yaw RMSE: {rmse:.5f} rad/s "
          f"(limit {MAX_RATE_RMSE:.3f})")
    print(f"resolved Jacobian signs: {sign_matches}/{resolved} "
          f"(need at least {MIN_RESOLVED_INTERVALS}, all matching)")
    print("local slopes (sign, midpoint rad, measured, frozen model):")
    for sign, middle, measured, predicted in slope_rows:
        print(f"  {sign:+d} {middle:.4f}: {measured:+.3f}, {predicted:+.3f} s^-1")

    accepted = (training_quality_ok and test_quality_ok and train_complete
                and not run_end.get("aborted", True)
                and phase_quality and rmse <= MAX_RATE_RMSE and jacobian_ok)
    print("decision: " + ("ACCEPT direct yaw surface for full-state transition-model validation only"
                          if accepted else
                          "REJECT direct yaw surface; do not promote to MPC"))
    rollout_ok = (_yaw_rollout_gate(training_path, test_path,
                                    training_rows, run_end)
                  if accepted else False)
    print("combined direct-yaw and recursive-yaw gate: " +
          ("PASS" if accepted and rollout_ok else "FAIL"))
    return accepted and rollout_ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training_bag", type=Path)
    parser.add_argument("independent_test_bag", type=Path)
    args = parser.parse_args()
    try:
        training_quality_ok = common.analyze(args.training_bag) == 0
        test_quality_ok = common.analyze(args.independent_test_bag) == 0
        return 0 if evaluate(args.training_bag, args.independent_test_bag,
                             training_quality_ok, test_quality_ok) else 1
    except (OSError, sqlite3.Error, ValueError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
