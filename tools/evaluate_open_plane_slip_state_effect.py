#!/usr/bin/env python3
"""Describe the same-sample slip/yaw association in the 4 m/s sweep.

The steering-only response failed a held-out steady-state check on the
0.18--0.28 rad transition sweep. This evaluator compares that baseline with
one-coefficient front- or rear-|Sy| corrections. Repetitions 1--2 train and
repetition 3 is untouched. Capture gate and model acceptance criteria are
predeclared in docs/development/ENGINEERING_STATE.md.

These are same-sample kinematic wheel-slip proxies derived from odometry and
Ackermann geometry, not tire-contact force measurements. Since the feature
uses the same measured yaw state as the response, this fit is explanatory only
and cannot validate a causal MPC predictor; use the separate causal rollout.
"""

from __future__ import annotations

import argparse
import math
import statistics
from pathlib import Path

try:
    import evaluate_open_plane_transient_sweep as sweep
    import evaluate_open_plane_transient_rollout as transient_rollout
except ImportError as exc:  # pragma: no cover - ROS 2 runtime dependency
    raise SystemExit(f"ROS 2 analysis modules are required: {exc}") from exc


LEVELS = tuple(round(0.18 + 0.01 * index, 2) for index in range(11))
TRAINING_REPETITIONS = (1, 2)
HOLDOUT_REPETITION = 3
TRANSITION_ANGLES = (0.21, 0.22)


def _median(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot summarize an empty condition")
    return statistics.median(values)


def _rmse(errors: list[float]) -> float:
    if not errors:
        raise ValueError("cannot score an empty holdout condition")
    return math.sqrt(sum(error * error for error in errors) / len(errors))


def _fit(blocks, feature_name: str, repetitions: tuple[int, ...]):
    selected = [block for block in blocks if block.repetition in repetitions]
    base = {
        angle: _median([block.yaw_gain_per_m for block in selected
                        if block.angle_rad == angle])
        for angle in LEVELS
    }
    feature_center = {
        angle: _median([getattr(block, feature_name) for block in selected
                        if block.angle_rad == angle])
        for angle in LEVELS
    }
    numerator = 0.0
    denominator = 0.0
    for block in selected:
        centered = getattr(block, feature_name) - feature_center[block.angle_rad]
        numerator += centered * (block.yaw_gain_per_m - base[block.angle_rad])
        denominator += centered * centered
    if denominator <= 1.0e-12:
        raise ValueError(f"{feature_name} has no within-steering training variation")
    return base, feature_center, numerator / denominator


def _training_repetition_beta(blocks, feature_name: str, repetition: int) -> float:
    selected = [block for block in blocks if block.repetition == repetition]
    numerator = 0.0
    denominator = 0.0
    for angle in LEVELS:
        cells = [block for block in selected if block.angle_rad == angle]
        base = _median([block.yaw_gain_per_m for block in cells])
        center = _median([getattr(block, feature_name) for block in cells])
        for block in cells:
            centered = getattr(block, feature_name) - center
            numerator += centered * (block.yaw_gain_per_m - base)
            denominator += centered * centered
    if denominator <= 1.0e-12:
        raise ValueError(f"{feature_name} has no variation in repetition {repetition}")
    return numerator / denominator


def _score(holdout, base, feature_center, feature_name: str, beta: float):
    errors = {"all": [], "transition": [], "outside": []}
    by_cell: dict[tuple[float, str], list[tuple[float, float, float]]] = {}
    for block in holdout:
        correction = beta * (
            getattr(block, feature_name) - feature_center[block.angle_rad])
        predicted_gain = base[block.angle_rad] + correction
        predicted_yaw_rate = (
            predicted_gain * block.speed_mps * math.tan(block.angle_rad))
        error = predicted_yaw_rate - block.signed_yaw_rate_rps
        errors["all"].append(error)
        group = ("transition" if block.angle_rad in TRANSITION_ANGLES else "outside")
        errors[group].append(error)
        by_cell.setdefault((block.angle_rad, block.direction), []).append(
            (block.yaw_gain_per_m, predicted_gain, block.front_slip_abs
             if feature_name == "front_slip_abs" else block.rear_slip_abs))
    return {name: _rmse(values) for name, values in errors.items()}, by_cell


def evaluate(path: Path) -> bool:
    blocks, quality, _ = sweep.load(path)
    repetitions = sorted({block.repetition for block in blocks})
    angles = tuple(sorted({block.angle_rad for block in blocks}))
    expected_per_repetition = 2 * (2 * len(LEVELS) - 1)
    counts = {repetition: sum(block.repetition == repetition for block in blocks)
              for repetition in repetitions}
    rates_ok = all(
        metric["hz"] is not None and metric["hz"] >= 38.0
        and metric["p95_gap_ms"] is not None and metric["p95_gap_ms"] <= 35.0
        and metric["max_gap_ms"] is not None and metric["max_gap_ms"] <= 60.0
        and metric["nonpositive_count"] == 0
        for metric in quality["rates"].values())
    capture_ok = (
        angles == LEVELS
        and repetitions == [1, 2, 3]
        and counts == {1: expected_per_repetition,
                       2: expected_per_repetition,
                       3: expected_per_repetition}
        and quality["valid_steering_phases"] == 3 * expected_per_repetition
        and quality["matched_sweep_starts"] == 6
        and quality["collision_initial"] == 0
        and quality["collision_final"] == 0
        and quality["bridge_timing_faults"] == 0
        and not quality["aborted"]
        and rates_ok)
    if not capture_ok:
        raise ValueError(f"capture failed its preregistered quality gate: {quality}; "
                         f"angle_counts={counts}")

    training = [block for block in blocks
                if block.repetition in TRAINING_REPETITIONS]
    holdout = [block for block in blocks
               if block.repetition == HOLDOUT_REPETITION]
    base, _, _ = _fit(blocks, "front_slip_abs", TRAINING_REPETITIONS)
    base_only_errors = [
        base[block.angle_rad] * block.speed_mps * math.tan(block.angle_rad)
        - block.signed_yaw_rate_rps
        for block in holdout]
    baseline = {
        "all": _rmse(base_only_errors),
        "transition": _rmse([
            error for error, block in zip(base_only_errors, holdout)
            if block.angle_rad in TRANSITION_ANGLES]),
        "outside": _rmse([
            error for error, block in zip(base_only_errors, holdout)
            if block.angle_rad not in TRANSITION_ANGLES]),
    }

    print(f"bag: {path}")
    print("capture PASS: 126/126 valid steps; 6/6 matched starts; "
          "zero collisions/timing faults; train repetitions 1-2, holdout 3")
    for topic, metric in quality["rates"].items():
        print(f"  {topic}: {metric['hz']:.3f} Hz; p95/max gap="
              f"{metric['p95_gap_ms']:.2f}/{metric['max_gap_ms']:.2f} ms")
    print("training-only yaw-rate RMSE baseline (rad/s): "
          f"all={baseline['all']:.5f}, transition={baseline['transition']:.5f}, "
          f"outside={baseline['outside']:.5f}")

    for feature_name, display_name in (
            ("front_slip_abs", "front |Sy|"), ("rear_slip_abs", "rear |Sy|")):
        beta_by_rep = {
            repetition: _training_repetition_beta(
                training, feature_name, repetition)
            for repetition in TRAINING_REPETITIONS}
        fitted_base, centers, beta = _fit(
            blocks, feature_name, TRAINING_REPETITIONS)
        scores, by_cell = _score(
            holdout, fitted_base, centers, feature_name, beta)
        sign_stable = beta_by_rep[1] * beta_by_rep[2] > 0.0
        all_pass = scores["all"] <= 0.75 * baseline["all"]
        transition_pass = scores["transition"] <= 0.50 * baseline["transition"]
        outside_pass = scores["outside"] <= 1.10 * baseline["outside"]
        passed = sign_stable and all_pass and transition_pass and outside_pass
        print(f"{display_name}: beta_rep1={beta_by_rep[1]:+.4f}, "
              f"beta_rep2={beta_by_rep[2]:+.4f}, pooled_beta={beta:+.4f} "
              "(1/m per unit slip)")
        print(f"  holdout yaw RMSE all/transition/outside="
              f"{scores['all']:.5f}/{scores['transition']:.5f}/"
              f"{scores['outside']:.5f} rad/s; "
              f"improvements={1-scores['all']/baseline['all']:+.1%}/"
              f"{1-scores['transition']/baseline['transition']:+.1%}/"
              f"{1-scores['outside']/baseline['outside']:+.1%}; "
              f"gate={'PASS' if passed else 'FAIL'}")
        for angle in TRANSITION_ANGLES:
            for direction in ("up", "down"):
                cells = by_cell.get((angle, direction), [])
                observed = _median([cell[0] for cell in cells])
                predicted = _median([cell[1] for cell in cells])
                slip = _median([cell[2] for cell in cells])
                print(f"  holdout |steer|={angle:.2f} {direction}: "
                      f"K observed/predicted={observed:.4f}/{predicted:.4f} 1/m; "
                      f"{display_name}={slip:.4f}")

    print("decision: same-sample association only; it is not a causal prediction "
          "test and must not be promoted to the MPC")

    # Score the deployed competition lateral-velocity law using simulator
    # speed/yaw as optimistic inputs. The real observer obtains speed from
    # rear encoders and yaw from the legal IMU, so this is an upper-bound check
    # on its model structure, not a runtime-topic proposal.
    probes, _ = transient_rollout.load(path)
    truth_vy_errors: list[float] = []
    truth_slip_errors: list[float] = []
    transition_vy_errors: list[float] = []
    transition_slip_errors: list[float] = []
    for probe in probes:
        if probe.repetition != HOLDOUT_REPETITION:
            continue
        for sample in probe.samples:
            speed = max(sample.speed_mps, 0.0)
            velocity_at_reference = max(-0.35, min(
                0.35, sample.yaw_rate_rps * (0.167 - 0.0063 * speed)))
            estimated_vy = velocity_at_reference - sample.yaw_rate_rps * 0.15532

            def mean_rear_slip(vy: float) -> float:
                half_track = sweep.common.TRACK_WIDTH_M * 0.5
                left = sweep.common._wheel_slip(
                    speed, vy, sample.yaw_rate_rps, 0.0, half_track, 0.0)
                right = sweep.common._wheel_slip(
                    speed, vy, sample.yaw_rate_rps, 0.0, -half_track, 0.0)
                if left is None or right is None:
                    raise ValueError("rear-slip estimate undefined at measured speed")
                return 0.5 * (abs(left) + abs(right))

            # `sample.vy_mps` is shifted to the rear-axle frame by the loader.
            vy_error = estimated_vy - sample.vy_mps
            slip_error = mean_rear_slip(estimated_vy) - mean_rear_slip(sample.vy_mps)
            truth_vy_errors.append(vy_error)
            truth_slip_errors.append(slip_error)
            if round(abs(probe.commanded_steering_rad), 2) in TRANSITION_ANGLES:
                transition_vy_errors.append(vy_error)
                transition_slip_errors.append(slip_error)
    vy_rmse = _rmse(truth_vy_errors)
    slip_rmse = _rmse(truth_slip_errors)
    transition_vy_rmse = _rmse(transition_vy_errors)
    transition_slip_rmse = _rmse(transition_slip_errors)
    odom_feature_pass = vy_rmse <= 0.05 and slip_rmse <= 0.01
    print("current competition odometry lateral-velocity law, with true speed/yaw inputs:")
    print(f"  holdout v_y RMSE={vy_rmse:.5f} m/s "
          f"(transition={transition_vy_rmse:.5f}; gate <=0.05)")
    print(f"  reconstructed rear |Sy| RMSE={slip_rmse:.5f} "
          f"(transition={transition_slip_rmse:.5f}; gate <=0.01); "
          f"feature_state={'PASS' if odom_feature_pass else 'FAIL'}")
    return capture_ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path)
    args = parser.parse_args()
    return 0 if evaluate(args.bag) else 1


if __name__ == "__main__":
    raise SystemExit(main())
