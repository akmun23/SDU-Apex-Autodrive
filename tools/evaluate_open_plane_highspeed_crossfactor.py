#!/usr/bin/env python3
"""Held-out test of high-speed steering onset and throttle-conditioned yaw.

The third repetition is never used to fit the per-speed throttle coefficient.
All GT is an offline label; candidate predictors use measured legal actuator
feedback and measured body speed only.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from tools import analyze_open_plane_dynamics as common
from tools import evaluate_open_plane_yaw_spline as yaw_spline


EXPECTED_SPEEDS = (6.5, 7.5)
STEERING_LEVELS = (0.10, 0.15, 0.20)
TRAIN_REPETITIONS = (1, 2)
HOLDOUT_REPETITION = 3
MIN_SPEED_OVERLAP_MPS = 0.15
MIN_PEDAL_SEPARATION = 0.004
MIN_HOLDOUT_IMPROVEMENT = 0.10
SETTLE_NS = 350_000_000


def _median(values: list[float]) -> float:
    return statistics.median(values)


def load_blocks(path: Path) -> tuple[list[dict], dict]:
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        topics = common._topic_map(connection)
        required = (common.ODOM, common.STEERING,
                    common.THROTTLE_FEEDBACK, common.PHASE)
        missing = [name for name in required if name not in topics]
        if missing:
            raise ValueError("missing topics: " + ", ".join(missing))

        odom = []
        for receipt_ns, message in common._messages(connection, topics, common.ODOM):
            stamp = common._stamp_ns(message.header.stamp)
            vx = float(message.twist.twist.linear.x)
            vy = float(message.twist.twist.linear.y)
            yaw = float(message.twist.twist.angular.z)
            if all(math.isfinite(value) for value in (vx, vy, yaw)):
                odom.append((receipt_ns, stamp, vx, vy, yaw))

        def scalar_rows(topic: str) -> list[tuple[int, float]]:
            return [(receipt, float(message.data))
                    for receipt, message in common._messages(connection, topics, topic)
                    if math.isfinite(float(message.data))]

        steering = scalar_rows(common.STEERING)
        throttle_feedback = scalar_rows(common.THROTTLE_FEEDBACK)
        steering_times = [row[0] for row in steering]
        throttle_feedback_times = [row[0] for row in throttle_feedback]
        phases: dict[int, dict] = {}
        phase_ends: dict[int, dict] = {}
        run_end: dict = {}
        for receipt, message in common._messages(connection, topics, common.PHASE):
            try:
                event = json.loads(message.data)
            except (TypeError, json.JSONDecodeError):
                continue
            if event.get("event") == "phase_start":
                phases[int(event["phase_index"])] = event | {"start_ns": receipt}
            elif event.get("event") == "phase_end":
                phase_ends[int(event["phase_index"])] = event | {"end_ns": receipt}
            elif event.get("event") == "experiment_end":
                run_end = event
    finally:
        connection.close()

    rows: list[dict] = []
    for index, start in sorted(phases.items()):
        end = phase_ends.get(index)
        if end is None or not start.get("label", "").startswith("isolated_r"):
            continue
        begin_ns = int(start["start_ns"]) + SETTLE_NS
        end_ns = int(end["end_ns"])
        selected = [row for row in odom if begin_ns <= row[0] < end_ns]
        samples: dict[str, list[float]] = defaultdict(list)
        for receipt, _stamp, vx, vy, yaw in selected:
            steer = common._nearest_scalar(
                steering, steering_times, receipt)
            throttle = common._nearest_scalar(
                throttle_feedback, throttle_feedback_times, receipt)
            if steer is None or throttle is None:
                continue
            speed = math.hypot(vx, vy)
            delta = steer[1]
            samples["speed"].append(speed)
            samples["forward_speed"].append(abs(vx))
            samples["throttle_feedback"].append(throttle[1])
            samples["yaw_rate"].append(yaw)
            if abs(delta) >= 0.04 and abs(vx * math.tan(delta)) > 0.15:
                samples["yaw_gain"].append(yaw / (vx * math.tan(delta)))
        if not samples["speed"]:
            continue
        expected_throttle = start.get("fixed_throttle_command_norm")
        rows.append({
            "label": str(start["label"]),
            "repetition": int(start["label"].split("_", 2)[1][1:]),
            "speed_target": float(start["target_speed_mps"]),
            "steering_command": float(start["steering_command_rad"]),
            "mode": str(start.get("throttle_mode", "unknown")),
            "fixed_throttle": (float(expected_throttle)
                               if isinstance(expected_throttle, (int, float)) else None),
            "valid": end.get("valid"),
            "start_speed": (float(start["initial_window_speed_mps"])
                            if isinstance(start.get("initial_window_speed_mps"),
                                          (int, float)) else math.nan),
            "speed": _median(samples["speed"]),
            "forward_speed": _median(samples["forward_speed"]),
            "throttle_feedback": _median(samples["throttle_feedback"]),
            "yaw_rate": _median(samples["yaw_rate"]),
            "yaw_gain": _median(samples["yaw_gain"]) if samples["yaw_gain"] else math.nan,
            "sample_count": len(samples["speed"]),
        })
    return rows, run_end


def evaluate_6p5mps_steering_holdout(rows: list[dict]) -> None:
    """Score steady steering response and its Jacobian on repetition 3."""
    angles = (0.05, 0.075, 0.10, 0.125, 0.15,
              0.175, 0.20, 0.225, 0.25)
    heldout = {
        (1 if row["steering_command"] > 0.0 else -1,
         round(abs(row["steering_command"]), 3)): row
        for row in rows
        if row["mode"] == "speed_hold"
        and abs(row["speed_target"] - 6.5) < 1e-6
        and row["repetition"] == HOLDOUT_REPETITION
    }
    expected = {(sign, angle) for sign in (-1, 1) for angle in angles}
    if set(heldout) != expected:
        print("6.5 m/s steering holdout: incomplete repetition 3; Jacobian score skipped")
        return

    predicted_gains: list[float] = []
    actual_gains: list[float] = []
    configured_gains: list[float] = []
    predicted_rates: list[float] = []
    actual_rates: list[float] = []
    configured_rates: list[float] = []
    measured_slopes: list[float] = []
    model_slopes: list[float] = []
    production_sign_matches = 0
    spline_sign_matches = 0
    derivative_intervals = 0

    for sign in (-1, 1):
        train_by_angle: dict[float, list[float]] = defaultdict(list)
        for row in rows:
            if (row["mode"] == "speed_hold"
                    and abs(row["speed_target"] - 6.5) < 1e-6
                    and row["repetition"] in TRAIN_REPETITIONS
                    and (1 if row["steering_command"] > 0.0 else -1) == sign):
                train_by_angle[round(abs(row["steering_command"]), 3)].append(
                    row["yaw_gain"])
        if set(train_by_angle) != set(angles) or any(
                len(train_by_angle[angle]) != 2 for angle in angles):
            print(f"6.5 m/s sign={sign:+d}: training grid incomplete; skipped")
            return
        x = [float(angle) for angle in angles]
        y = [statistics.mean(train_by_angle[angle]) for angle in angles]

        for angle in angles:
            block = heldout[(sign, angle)]
            predicted_gain, _ = yaw_spline._pchip_value_and_slope(
                np.asarray(x), np.asarray(y), angle)
            configured_gain = yaw_spline._configured_gain(sign * angle)
            predicted_gains.append(predicted_gain)
            actual_gains.append(block["yaw_gain"])
            configured_gains.append(configured_gain)
            predicted_rates.append(
                block["forward_speed"] * math.tan(sign * angle) * predicted_gain)
            actual_rates.append(block["yaw_rate"])
            configured_rates.append(
                block["forward_speed"] * math.tan(sign * angle) * configured_gain)

        for left_angle, right_angle in zip(angles, angles[1:]):
            left = heldout[(sign, left_angle)]
            right = heldout[(sign, right_angle)]
            measured = (sign * right["yaw_rate"] - sign * left["yaw_rate"])
            measured /= right_angle - left_angle
            middle = 0.5 * (left_angle + right_angle)
            gain, gain_slope = yaw_spline._pchip_value_and_slope(
                np.asarray(x), np.asarray(y), middle)
            speed = 0.5 * (left["forward_speed"] + right["forward_speed"])
            model = speed * (gain / math.cos(middle) ** 2
                             + math.tan(middle) * gain_slope)
            measured_slopes.append(measured)
            model_slopes.append(model)
            if abs(measured) >= 0.05:
                derivative_intervals += 1
                spline_sign_matches += (measured * model > 0.0)
                production_sign_matches += measured > 0.0

    gain_rmse = math.sqrt(statistics.mean(
        (actual - predicted) ** 2
        for actual, predicted in zip(actual_gains, predicted_gains)))
    configured_rmse = math.sqrt(statistics.mean(
        (actual - predicted) ** 2
        for actual, predicted in zip(actual_gains, configured_gains)))
    rate_rmse = math.sqrt(statistics.mean(
        (actual - predicted) ** 2
        for actual, predicted in zip(actual_rates, predicted_rates)))
    configured_rate_rmse = math.sqrt(statistics.mean(
        (actual - predicted) ** 2
        for actual, predicted in zip(actual_rates, configured_rates)))
    print("6.5 m/s repetition-3 steering holdout (train reps 1–2):")
    print(f"  yaw-gain RMSE: local PCHIP={gain_rmse:.4f}, "
          f"production gain={configured_rmse:.4f} 1/m")
    print(f"  yaw-rate RMSE: local PCHIP={rate_rmse:.4f}, "
          f"production gain={configured_rate_rmse:.4f} rad/s")
    print(f"  held-out d|r|/d|steer| (measured): "
          f"{[round(value, 3) for value in measured_slopes]}")
    print(f"  PCHIP derivative: {[round(value, 3) for value in model_slopes]}")
    print(f"  local-Jacobian sign agreement: PCHIP="
          f"{spline_sign_matches}/{derivative_intervals}; production monotone="
          f"{production_sign_matches}/{derivative_intervals}")


def evaluate(path: Path) -> bool:
    rows, run_end = load_blocks(path)
    by_fixed: dict[tuple[float, float, int], list[dict]] = defaultdict(list)
    for row in rows:
        if row["mode"] == "fixed" and row["fixed_throttle"] is not None:
            by_fixed[(row["speed_target"], row["steering_command"],
                      row["repetition"])].append(row)

    print(f"bag: {path}")
    print(f"run_end: aborted={run_end.get('aborted')}, reason={run_end.get('reason')!r}")
    evaluate_6p5mps_steering_holdout(rows)
    print("steady speed-hold steering response (yaw gain 1/m):")
    for speed in EXPECTED_SPEEDS:
        for angle in (0.05, 0.075, *STEERING_LEVELS, 0.125, 0.175, 0.225, 0.25):
            for sign in (-1, 1):
                values = [row["yaw_gain"] for row in rows
                          if row["mode"] == "speed_hold"
                          and abs(row["speed_target"] - speed) < 1e-6
                          and abs(abs(row["steering_command"]) - angle) < 1e-6
                          and (1 if row["steering_command"] > 0 else -1) == sign
                          and row["repetition"] in TRAIN_REPETITIONS
                          and math.isfinite(row["yaw_gain"])]
                if values:
                    print(f"  {speed:.1f} m/s steer={sign * angle:+.3f}: "
                          f"K={statistics.mean(values):.4f} (n={len(values)})")

    gate_pass = True
    all_pairs: dict[float, list[tuple[dict, dict]]] = defaultdict(list)
    print("fixed-pedal paired contrasts (high minus low):")
    for speed in EXPECTED_SPEEDS:
        speed_pairs = []
        pair_quality = True
        for angle in STEERING_LEVELS:
            for sign in (-1, 1):
                for repetition in (1, 2, 3):
                    key = (speed, sign * angle, repetition)
                    blocks = sorted(by_fixed.get(key, []), key=lambda row: row["fixed_throttle"])
                    if len(blocks) != 2:
                        gate_pass = False
                        continue
                    low, high = blocks
                    overlap = abs(high["speed"] - low["speed"])
                    pedal_delta = high["throttle_feedback"] - low["throttle_feedback"]
                    k_delta = high["yaw_gain"] - low["yaw_gain"]
                    speed_pairs.append((low, high))
                    pair_quality &= abs(pedal_delta) >= MIN_PEDAL_SEPARATION and all(
                        block["valid"] is True for block in (low, high))
                    pair_quality &= (
                        math.isfinite(low["start_speed"])
                        and math.isfinite(high["start_speed"])
                        and abs(high["start_speed"] - low["start_speed"]) <= 0.10
                    )
                    print(f"  {speed:.1f} steer={sign * angle:+.2f} r{repetition}: "
                          f"v={low['speed']:.3f}/{high['speed']:.3f} m/s "
                          f"(gap={overlap:.3f}), throttle={low['throttle_feedback']:.3f}/"
                          f"{high['throttle_feedback']:.3f}, "
                          f"dK={k_delta:+.4f} 1/m")
        all_pairs[speed] = speed_pairs
        overlap_fraction = (sum(abs(high["speed"] - low["speed"])
                                <= MIN_SPEED_OVERLAP_MPS
                                for low, high in speed_pairs) / len(speed_pairs)
                            if speed_pairs else 0.0)
        print(f"  {speed:.1f} m/s matched-speed fraction: "
              f"{overlap_fraction:.1%} ({len(speed_pairs)} pairs)")
        gate_pass &= (pair_quality and len(speed_pairs) == 18
                      and overlap_fraction >= 0.80)

    print("rep-3 holdout: condition-only versus throttle-conditioned yaw-gain predictor")
    for speed in EXPECTED_SPEEDS:
        train_pairs = [pair for pair in all_pairs[speed]
                       if pair[0]["repetition"] in TRAIN_REPETITIONS]
        holdout_pairs = [pair for pair in all_pairs[speed]
                         if pair[0]["repetition"] == HOLDOUT_REPETITION]
        slopes = []
        slopes_by_sign: dict[int, list[float]] = {-1: [], 1: []}
        speed_slopes = []
        for low, high in train_pairs:
            dt = high["throttle_feedback"] - low["throttle_feedback"]
            if abs(dt) < MIN_PEDAL_SEPARATION:
                continue
            slope = (high["yaw_gain"] - low["yaw_gain"]) / dt
            slopes.append(slope)
            slopes_by_sign[1 if high["steering_command"] > 0 else -1].append(slope)
            dv = high["speed"] - low["speed"]
            if abs(dv) > 0.04:
                speed_slopes.append((high["yaw_gain"] - low["yaw_gain"]) / dv)
        if not slopes or not speed_slopes or not holdout_pairs:
            print(f"  {speed:.1f} m/s: insufficient training/holdout pairs")
            gate_pass = False
            continue
        beta = statistics.median(slopes)
        sign_betas = {sign: statistics.median(values)
                      for sign, values in slopes_by_sign.items() if values}
        sign_consistent = (len(sign_betas) == 2
                           and sign_betas[-1] * sign_betas[1] > 0.0)

        intercepts: dict[tuple[float, int], list[float]] = defaultdict(list)
        for low, high in train_pairs:
            condition = (abs(low["steering_command"]),
                         1 if low["steering_command"] > 0 else -1)
            for block in (low, high):
                intercepts[condition].append(
                    block["yaw_gain"] - beta * block["throttle_feedback"])
        baseline_by_condition: dict[tuple[float, int], list[float]] = defaultdict(list)
        for low, high in train_pairs:
            condition = (abs(low["steering_command"]),
                         1 if low["steering_command"] > 0 else -1)
            baseline_by_condition[condition].extend(
                (low["yaw_gain"], high["yaw_gain"]))

        baseline_errors: list[float] = []
        conditioned_errors: list[float] = []
        speed_conditioned_errors: list[float] = []
        speed_intercepts: dict[tuple[float, int], list[float]] = defaultdict(list)
        for low, high in train_pairs:
            condition = (abs(low["steering_command"]),
                         1 if low["steering_command"] > 0 else -1)
            for block in (low, high):
                speed_intercepts[condition].append(
                    block["yaw_gain"] - statistics.median(speed_slopes) * block["speed"])
        for low, high in holdout_pairs:
            condition = (abs(low["steering_command"]),
                         1 if low["steering_command"] > 0 else -1)
            baseline = statistics.mean(baseline_by_condition[condition])
            intercept = statistics.mean(intercepts[condition])
            speed_intercept = statistics.mean(speed_intercepts[condition])
            for block in (low, high):
                baseline_errors.append((baseline - block["yaw_gain"]) ** 2)
                prediction = intercept + beta * block["throttle_feedback"]
                conditioned_errors.append((prediction - block["yaw_gain"]) ** 2)
                speed_prediction = (speed_intercept
                                    + statistics.median(speed_slopes) * block["speed"])
                speed_conditioned_errors.append(
                    (speed_prediction - block["yaw_gain"]) ** 2)
        baseline_rmse = math.sqrt(statistics.mean(baseline_errors))
        conditioned_rmse = math.sqrt(statistics.mean(conditioned_errors))
        speed_rmse = math.sqrt(statistics.mean(speed_conditioned_errors))
        improvement = 1.0 - conditioned_rmse / baseline_rmse if baseline_rmse else 0.0
        print(f"  {speed:.1f} m/s beta={beta:+.4f} per throttle; "
              f"left/right beta={sign_betas}; rep-3 RMSE "
              f"{baseline_rmse:.4f}->{conditioned_rmse:.4f} 1/m "
              f"({improvement:+.1%}), sign-consistent={sign_consistent}; "
              f"speed-only RMSE={speed_rmse:.4f} 1/m")
        gate_pass &= improvement >= MIN_HOLDOUT_IMPROVEMENT and sign_consistent

    print("decision: " + ("ACCEPT throttle-conditioned local model for further transient validation"
                          if gate_pass else
                          "REJECT throttle-conditioned MPC model; retain only descriptive data"))
    return gate_pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path)
    args = parser.parse_args()
    try:
        quality_ok = common.analyze(args.bag) == 0
        model_ok = evaluate(args.bag)
        if not quality_ok:
            print("capture failed whole-run quality gates; partial speed-group results "
                  "remain diagnostic only")
            return 1
        return 0 if model_ok else 1
    except (OSError, sqlite3.Error, ValueError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
