#!/usr/bin/env python3
"""Audit effective Unity physics cadence using permitted recorded IMU/LiDAR.

Input is an .npz capture with `imu` rows [stamp, receipt, yaw, gyro_z, ax, ay],
`cmd` rows [stamp, receipt, steering, throttle], and optional `scan_ranges`.
The inferred physics interval is d(yaw) / mean(gyro_z). It is observable only
in sufficiently steady turning: it is NOT a simulator clock or ground truth.

Example:
  python3 tools/analysis/audit_sim_physics_cadence.py batch.npz graphical.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def summarize(path: Path) -> None:
    with np.load(path) as capture:
        imu = np.asarray(capture["imu"], dtype=np.float64)
        commands = np.asarray(capture["cmd"], dtype=np.float64)
        if imu.ndim != 2 or imu.shape[1] < 4 or imu.shape[0] < 30:
            raise ValueError(f"{path}: missing or insufficient stamped IMU samples")
        if commands.ndim != 2 or commands.shape[0] < 2:
            raise ValueError(f"{path}: missing MPC command samples")

        stamp_dt = np.diff(imu[:, 0])
        yaw_delta = np.diff(np.unwrap(imu[:, 2]))
        yaw_rate_before, yaw_rate_after = imu[:-1, 3], imu[1:, 3]
        mean_yaw_rate = 0.5 * (yaw_rate_before + yaw_rate_after)
        rate_change = np.abs(yaw_rate_after - yaw_rate_before)

        # Nearly constant yaw rate limits the trapezoidal integration error.
        # Exclude startup packets and periods with zero-turn observability.
        stable = (
            (imu[:-1, 0] >= commands[0, 0])
            & (stamp_dt >= 0.021)
            & (stamp_dt <= 0.029)
            & (np.abs(mean_yaw_rate) >= 0.75)
            & (rate_change < 0.15)
        )
        effective_dt = np.divide(
            yaw_delta, mean_yaw_rate,
            out=np.full_like(stamp_dt, np.nan),
            where=np.abs(mean_yaw_rate) >= 0.75,
        )
        valid = stable & np.isfinite(effective_dt) & (
            effective_dt >= 0.005) & (effective_dt <= 0.050)
        if np.count_nonzero(valid) < 20:
            raise ValueError(
                f"{path}: fewer than 20 sufficiently steady turning samples")

        observed_ms = effective_dt[valid] * 1000.0
        stamped_ms = stamp_dt[valid] * 1000.0
        near_25_ms = np.mean(np.abs(observed_ms - 25.0) <= 2.5)
        out_of_band = np.mean((observed_ms < 21.0) | (observed_ms > 29.0))
        zero_rotation_high_yaw = np.mean(
            np.abs(yaw_delta[stable]) < 1e-6)
        lidar_duplicate = None
        if "scan_ranges" in capture:
            ranges = np.asarray(capture["scan_ranges"])
            if ranges.ndim == 2 and len(ranges) >= 2:
                lidar_duplicate = float(np.mean(np.all(
                    ranges[1:] == ranges[:-1], axis=1)))

        def q(values: np.ndarray) -> str:
            return "/".join(f"{v:.1f}" for v in np.percentile(
                values, [5, 50, 95]))

        print(f"{path.name} ({path})")
        print(f"  Valid turning intervals: {len(observed_ms)}")
        print(f"  ROS stamped interval P5/P50/P95: {q(stamped_ms)} ms")
        print(f"  IMU-inferred interval P5/P50/P95: {q(observed_ms)} ms")
        print(f"  Inferred within 25 ± 2.5 ms: {near_25_ms:.1%}")
        print(f"  Inferred outside 21–29 ms: {out_of_band:.1%}")
        print(f"  Zero orientation change with high yaw rate: "
              f"{zero_rotation_high_yaw:.1%}")
        if lidar_duplicate is not None:
            print(f"  Identical consecutive LiDAR arrays: {lidar_duplicate:.1%}")
        if near_25_ms < 0.50:
            print("  TIMING WARNING: request epochs disagree with effective "
                  "turning-motion intervals")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", nargs="+", type=Path)
    args = parser.parse_args()
    for filename in args.captures:
        summarize(filename)


if __name__ == "__main__":
    main()
