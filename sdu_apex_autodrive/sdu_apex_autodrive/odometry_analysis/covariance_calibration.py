"""Empirical covariance calibration for the local odometry trust filter.

The runtime EKF is deliberately local and does not consume simulator truth.
This module uses truth only offline to select process-noise densities and to
report coverage/NEES.  It consumes the exact v3 observer diagnostics packet,
not callback snapshots that may mix sensor timestamps.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CovarianceSample:
    stamp_s: float
    distance_m: float
    yaw_distance_rad: float
    error_x_m: float
    error_y_m: float
    error_yaw_rad: float
    step_distance_m: float
    step_yaw_rad: float
    dt_s: float


def _finite(row: dict[str, str], key: str) -> float | None:
    try:
        value = float(row.get(key, ""))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _angle_diff(a: float, b: float) -> float:
    return math.atan2(math.sin(a - b), math.cos(a - b))


def _interpolate_truth(
    samples: list[tuple[float, float, float, float]], stamp_s: float,
) -> tuple[float, float, float] | None:
    if not samples or stamp_s < samples[0][0] or stamp_s > samples[-1][0]:
        return None
    for before, after in zip(samples, samples[1:]):
        if stamp_s > after[0]:
            continue
        dt = after[0] - before[0]
        if dt <= 0.0:
            return after[1:]
        ratio = (stamp_s - before[0]) / dt
        return (
            before[1] + ratio * (after[1] - before[1]),
            before[2] + ratio * (after[2] - before[2]),
            before[3] + ratio * _angle_diff(after[3], before[3]),
        )
    return samples[-1][1:]


def load_covariance_samples(
    path: str | Path,
    *,
    initial_xy_variance_m2: float = 0.01,
    initial_yaw_variance_rad2: float = 0.01,
) -> list[CovarianceSample]:
    """Extract valid local-odom errors from a recorder CSV.

    The ground-truth pose is rotated into the observer's initial local frame.
    Collision epochs, observer outliers, resets, and degraded timing packets
    are excluded so a teleport cannot inflate the fitted process noise.
    """
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    truth: list[tuple[float, float, float, float]] = []
    for row in rows:
        if row.get("source_event_name") != "gt_odom":
            continue
        stamp = _finite(row, "source_event_stamp_s")
        x = _finite(row, "gt_odom_x_m")
        y = _finite(row, "gt_odom_y_m")
        yaw = _finite(row, "gt_odom_yaw_rad")
        if None not in (stamp, x, y, yaw):
            truth.append((stamp, x, y, yaw))
    truth = sorted({sample[0]: sample for sample in truth}.values())
    if len(truth) < 2:
        raise ValueError("CSV has fewer than two timestamped ground-truth odom samples")

    diagnostics: list[tuple[float, dict[str, str]]] = []
    for row in rows:
        if row.get("source_event_name") != "odom_diagnostics":
            continue
        stamp = _finite(row, "odom_observer_source_stamp_s")
        if stamp is None:
            continue
        collision = _finite(row, "gt_collision_count")
        if collision is not None and collision > 0.0:
            continue
        if (_finite(row, "odom_observer_sensor_outlier") or 0.0) > 0.5:
            continue
        if (_finite(row, "odom_observer_reset_epoch") or 0.0) > 0.5:
            continue
        if (_finite(row, "odom_observer_timing_degraded") or 0.0) > 0.5:
            continue
        required = (
            "odom_observer_x_m", "odom_observer_y_m", "odom_observer_yaw_rad",
            "odom_observer_left_angle_rad", "odom_observer_right_angle_rad",
            "odom_observer_imu_yaw_rad",
        )
        if any(_finite(row, key) is None for key in required):
            # Old v2 logs remain useful for scoring, but cannot be used for a
            # source-exact covariance fit.
            continue
        diagnostics.append((stamp, row))
    diagnostics = sorted({stamp: row for stamp, row in diagnostics}.items())
    # The recorder can receive one observer-diagnostics packet before the
    # first simulator ground-truth odometry callback during startup.  It is a
    # valid runtime packet, but it cannot be scored source-time exactly. Drop
    # only diagnostics outside the truth bracket instead of failing the whole
    # fit or pairing it with a later truth sample.
    truth_start = truth[0][0]
    truth_end = truth[-1][0]
    diagnostics = [
        (stamp, row) for stamp, row in diagnostics
        if truth_start <= stamp <= truth_end
    ]
    if len(diagnostics) < 3:
        raise ValueError("CSV has fewer than three valid v3 odometry diagnostics samples")

    first_truth = _interpolate_truth(truth, diagnostics[0][0])
    if first_truth is None:
        raise ValueError("ground truth does not bracket the first odometry diagnostic")
    gx0, gy0, gyaw0 = first_truth
    c0 = math.cos(gyaw0)
    s0 = math.sin(gyaw0)

    result: list[CovarianceSample] = []
    previous: tuple[float, float, float, float, float, float] | None = None
    for stamp, row in diagnostics:
        gt = _interpolate_truth(truth, stamp)
        if gt is None:
            continue
        gx, gy, gyaw = gt
        dx = gx - gx0
        dy = gy - gy0
        # Truth expressed in the observer's initial body-aligned odom frame.
        tx = c0 * dx + s0 * dy
        ty = -s0 * dx + c0 * dy
        tyaw = _angle_diff(gyaw, gyaw0)
        ox = _finite(row, "odom_observer_x_m")
        oy = _finite(row, "odom_observer_y_m")
        oyaw = _finite(row, "odom_observer_yaw_rad")
        if None in (ox, oy, oyaw):
            continue
        if previous is None:
            step_distance = 0.0
            step_yaw = 0.0
            dt = 0.0
            distance = 0.0
            yaw_distance = 0.0
        else:
            dt = max(0.0, stamp - previous[0])
            step_distance = math.hypot(tx - previous[1], ty - previous[2])
            step_yaw = abs(_angle_diff(tyaw, previous[3]))
            distance = previous[4] + step_distance
            yaw_distance = previous[5] + step_yaw
        result.append(CovarianceSample(
            stamp_s=stamp,
            distance_m=distance,
            yaw_distance_rad=yaw_distance,
            error_x_m=ox - tx,
            error_y_m=oy - ty,
            error_yaw_rad=_angle_diff(oyaw, tyaw),
            step_distance_m=step_distance,
            step_yaw_rad=step_yaw,
            dt_s=dt,
        ))
        previous = (stamp, tx, ty, tyaw, distance, yaw_distance)

    if len(result) < 3 or result[-1].distance_m <= 0.25:
        raise ValueError("valid diagnostics contain insufficient clean motion for covariance fitting")
    return result


def _coverage(samples: list[CovarianceSample], q_xy_m: float, q_xy_s: float,
              q_yaw_m: float, q_yaw_rad: float,
              initial_xy: float, initial_yaw: float) -> dict[str, float]:
    if not samples:
        return {"component_coverage": 0.0, "ellipse_coverage": 0.0,
                "yaw_coverage": 0.0, "nees2_p95": math.inf}
    component = []
    ellipse = []
    yaw = []
    nees = []
    for sample in samples:
        xy_var = max(1.0e-12, initial_xy + q_xy_m * sample.distance_m +
                     q_xy_s * max(0.0, sample.stamp_s - samples[0].stamp_s))
        yaw_var = max(1.0e-12, initial_yaw + q_yaw_m * sample.distance_m +
                      q_yaw_rad * sample.yaw_distance_rad)
        sigma = math.sqrt(xy_var)
        component.append(abs(sample.error_x_m) <= 1.96 * sigma and
                         abs(sample.error_y_m) <= 1.96 * sigma)
        ellipse_value = (sample.error_x_m ** 2 + sample.error_y_m ** 2) / xy_var
        ellipse.append(ellipse_value <= 5.991)
        nees.append(ellipse_value)
        yaw.append(abs(sample.error_yaw_rad) <= 1.96 * math.sqrt(yaw_var))
    return {
        "component_coverage": sum(component) / len(component),
        "ellipse_coverage": sum(ellipse) / len(ellipse),
        "yaw_coverage": sum(yaw) / len(yaw),
        "nees2_p95": sorted(nees)[max(0, int(math.ceil(0.95 * len(nees))) - 1)],
    }


def calibrate_process_noise(
    samples: list[CovarianceSample],
    *,
    initial_xy_variance_m2: float = 0.01,
    initial_yaw_variance_rad2: float = 0.01,
) -> dict[str, object]:
    """Choose the smallest grid-search densities meeting 95% coverage."""
    if len(samples) < 3 or samples[-1].distance_m <= 0.25:
        raise ValueError("insufficient motion samples")
    step_samples = samples[1:]
    total_distance = max(sum(s.step_distance_m for s in step_samples), 1.0e-9)
    total_time = max(sum(s.dt_s for s in step_samples), 1.0e-9)
    total_yaw = max(sum(s.step_yaw_rad for s in step_samples), 1.0e-9)
    xy_residual = sum((s.error_x_m - samples[i].error_x_m) ** 2 +
                      (s.error_y_m - samples[i].error_y_m) ** 2
                      for i, s in enumerate(step_samples)) / (2.0 * len(step_samples))
    yaw_residual = sum((s.error_yaw_rad - samples[i].error_yaw_rad) ** 2
                       for i, s in enumerate(step_samples)) / len(step_samples)
    xy_base_m = max(1.0e-8, xy_residual / total_distance)
    xy_base_s = max(1.0e-9, xy_residual / total_time)
    yaw_base_m = max(1.0e-9, yaw_residual / total_distance)
    yaw_base_rad = max(1.0e-9, yaw_residual / total_yaw)

    def grid(base: float) -> list[float]:
        return [base * 10.0 ** exponent for exponent in (-3, -2, -1, 0, 1, 2, 3)]

    best: tuple[float, dict[str, float], dict[str, float]] | None = None
    for xy_m in grid(xy_base_m):
        for xy_s in grid(xy_base_s):
            for yaw_m in grid(yaw_base_m):
                for yaw_rad in grid(yaw_base_rad):
                    metrics = _coverage(
                        samples, xy_m, xy_s, yaw_m, yaw_rad,
                        initial_xy_variance_m2, initial_yaw_variance_rad2)
                    if (metrics["component_coverage"] < 0.95 or
                            metrics["ellipse_coverage"] < 0.95 or
                            metrics["yaw_coverage"] < 0.95):
                        continue
                    cost = (
                        xy_m * total_distance + xy_s * total_time +
                        yaw_m * total_distance + yaw_rad * total_yaw)
                    candidate = {
                        "process_noise_xy_m2_per_m": xy_m,
                        "process_noise_xy_m2_per_s": xy_s,
                        "process_noise_yaw2_per_m": yaw_m,
                        "process_noise_yaw2_per_rad": yaw_rad,
                    }
                    if best is None or cost < best[0]:
                        best = (cost, candidate, metrics)
    if best is None:
        raise ValueError("no process-noise candidate achieved 95% empirical coverage")
    _, candidate, metrics = best
    return {
        "samples": len(samples),
        "distance_m": samples[-1].distance_m,
        "candidate": candidate,
        "coverage": metrics,
        "baseline_rates": {
            "xy_m2_per_m": xy_base_m,
            "xy_m2_per_s": xy_base_s,
            "yaw2_per_m": yaw_base_m,
            "yaw2_per_rad": yaw_base_rad,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    samples = load_covariance_samples(args.input)
    result = calibrate_process_noise(samples)
    encoded = json.dumps(result, indent=2, sort_keys=True)
    print(encoded)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
