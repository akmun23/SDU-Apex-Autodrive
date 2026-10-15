#!/usr/bin/env python3
"""Compare causal endpoint estimates derived from Unity's cumulative encoder.

Unity's WheelEncoder integrates WheelCollider RPM, quantizes cumulative angle,
and publishes that angle. This diagnostic compares simple causal derivatives
of the same angle stream on exact source-packet joins. Simulator body speed is
an offline label only; the script does not call it wheel-speed truth.
"""

from __future__ import annotations

import argparse
import bisect
import json
from pathlib import Path
from typing import Any

import numpy as np

from tools.evaluate_open_plane_body_dynamics import MotionSample, load_capture
from tools.vehicle_dynamics_learning.score_practice_odom_replay import (
    _read_diagnostics,
)


DT_S = 0.025
WHEEL_RADIUS_M = 0.059
DIAGNOSTIC_SIZE = 42
SOURCE_STAMP_INDEX = 28


def _metric(error: np.ndarray) -> dict[str, Any]:
    if not len(error):
        return {"samples": 0, "bias_mps": None, "rmse_mps": None,
                "p95_abs_mps": None}
    return {
        "samples": int(len(error)),
        "bias_mps": float(np.mean(error)),
        "rmse_mps": float(np.sqrt(np.mean(error ** 2))),
        "p95_abs_mps": float(np.quantile(np.abs(error), 0.95)),
    }


def _join(capture_path: Path, replay_path: Path
          ) -> tuple[list[MotionSample], np.ndarray]:
    capture = load_capture(
        capture_path, include_nonvalid_phases=True, continuous_phased_run=True)
    by_sequence: dict[int, MotionSample] = {}
    for sequence in capture.sequences:
        for sample in sequence:
            if sample.packet_sequence >= 0 and np.isfinite(sample.state).all():
                by_sequence[sample.packet_sequence] = sample
    diagnostics = _read_diagnostics(replay_path, "/replayed_odom_diagnostics")
    stamps = sorted(diagnostics)
    joined: list[MotionSample] = []
    rows: list[np.ndarray] = []
    used: set[int] = set()
    for sample in (by_sequence[key] for key in sorted(by_sequence)):
        insertion = bisect.bisect_left(stamps, sample.source_stamp_ns)
        nearby = [stamps[i] for i in (insertion - 1, insertion)
                  if 0 <= i < len(stamps)]
        if not nearby:
            continue
        stamp = min(nearby, key=lambda value: abs(value - sample.source_stamp_ns))
        if abs(stamp - sample.source_stamp_ns) > 2_000 or stamp in used:
            continue
        row = diagnostics[stamp]
        if row.shape != (DIAGNOSTIC_SIZE,):
            raise ValueError(f"unexpected observer diagnostics shape {row.shape}")
        used.add(stamp)
        joined.append(sample)
        rows.append(row)
    if len(joined) < 5:
        raise ValueError("fewer than five exact source-packet joins")
    return joined, np.stack(rows)


def _endpoint_quadratic(angles: np.ndarray, index: int, count: int) -> float:
    """Causal derivative at the newest point of a trailing quadratic fit."""
    start = index - count + 1
    times = np.arange(-count + 1, 1, dtype=np.float64) * DT_S
    design = np.column_stack((np.ones(count), times, times * times))
    derivative_weights = np.linalg.pinv(design)[1]
    return float(derivative_weights @ angles[start:index + 1])


def _scale(raw_speed: float, speeds: np.ndarray, scales: np.ndarray) -> float:
    return abs(raw_speed) * float(np.interp(abs(raw_speed), speeds, scales))


def analyze(label: str, capture_path: Path, replay_path: Path) -> dict[str, Any]:
    samples, diag = _join(capture_path, replay_path)
    angles = diag[:, 22:24]
    scale_speeds = np.asarray(
        (0, .5, 1, 1.5, 2, 4, 6, 8, 10, 12, 14), dtype=np.float64)
    scale_values = np.asarray(
        (1, .998, .994, .992, .9947, .9788, .9561, .9548,
         .9479, .952, .9445), dtype=np.float64)
    names = ("production_100ms", "raw_25ms", "centered_50ms_mean",
             "endpoint_quadratic_3packets", "endpoint_quadratic_5packets")
    errors: dict[str, list[float]] = {name: [] for name in names}
    accepted_errors: dict[str, list[float]] = {name: [] for name in names}
    by_regime: dict[str, dict[str, list[float]]] = {
        name: {} for name in names}
    skipped_gaps = 0
    for index in range(1, len(samples)):
        current, previous = samples[index], samples[index - 1]
        if current.packet_sequence != previous.packet_sequence + 1:
            skipped_gaps += 1
            continue
        # The repository's fixed-clock contract is one physics packet per
        # 25 ms. Refuse a nonconsecutive sequence rather than reinterpret it.
        truth_u = float(current.state[0])
        raw25 = WHEEL_RADIUS_M * 0.5 * float(
            (angles[index, 0] - angles[index - 1, 0]) +
            (angles[index, 1] - angles[index - 1, 1])) / DT_S
        raw50 = WHEEL_RADIUS_M * 0.5 * float(
            (angles[index, 0] - angles[index - 2, 0]) +
            (angles[index, 1] - angles[index - 2, 1])) / (2 * DT_S) \
            if index >= 2 and samples[index].packet_sequence - \
            samples[index - 2].packet_sequence == 2 else raw25
        raw3 = 0.5 * WHEEL_RADIUS_M * sum(
            _endpoint_quadratic(angles[:, side], index, 3)
            for side in (0, 1)) if index >= 2 and \
            samples[index].packet_sequence - samples[index - 2].packet_sequence == 2 \
            else raw25
        raw5 = 0.5 * WHEEL_RADIUS_M * sum(
            _endpoint_quadratic(angles[:, side], index, 5)
            for side in (0, 1)) if index >= 4 and \
            samples[index].packet_sequence - samples[index - 4].packet_sequence == 4 \
            else raw25
        candidates = {
            "production_100ms": float(diag[index, 4]),
            "raw_25ms": _scale(raw25, scale_speeds, scale_values),
            "centered_50ms_mean": _scale(raw50, scale_speeds, scale_values),
            "endpoint_quadratic_3packets": _scale(raw3, scale_speeds, scale_values),
            "endpoint_quadratic_5packets": _scale(raw5, scale_speeds, scale_values),
        }
        speed = float(np.hypot(current.state[0], current.state[1]))
        steering = abs(float(current.actuators[0]))
        accel = float(np.linalg.norm(current.imu_acceleration_mps2)) \
            if current.imu_acceleration_mps2 is not None else 0.0
        speed_band = f"speed_{int(speed // 2) * 2}_{int(speed // 2) * 2 + 2}"
        steer_band = "steer_ge_0.3" if steering >= .3 else "steer_lt_0.3"
        accel_band = "imu_accel_ge_4" if accel >= 4 else "imu_accel_lt_4"
        for name, estimate in candidates.items():
            error = estimate - truth_u
            errors[name].append(error)
            if diag[index, 12] > 0.5:
                accepted_errors[name].append(error)
            for key in (speed_band, steer_band, accel_band):
                by_regime[name].setdefault(key, []).append(error)
    return {
        "run": label,
        "capture": str(capture_path),
        "replay": str(replay_path),
        "samples": len(samples),
        "skipped_sequence_gaps": skipped_gaps,
        "truth_semantics": "simulator body-u label; not wheel angular-speed truth",
        "estimators": {
            name: {
                **_metric(np.asarray(values, dtype=np.float64)),
                "on_production_wheel_update_samples": _metric(
                    np.asarray(accepted_errors[name], dtype=np.float64)),
                "by_regime": {
                    key: _metric(np.asarray(group, dtype=np.float64))
                    for key, group in by_regime[name].items()
                },
            }
            for name, values in errors.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", nargs=3, action="append", required=True,
                        metavar=("LABEL", "CAPTURE_DB3", "REPLAY_DB3"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = {
        "method": "exact source-packet join; fixed 25 ms; causal angle derivatives",
        "equation": "WheelEncoder angle is cumulative integral of WheelCollider RPM, quantized to ticks",
        "wheel_radius_m": WHEEL_RADIUS_M,
        "scale_table": {
            "speed_mps": [0, .5, 1, 1.5, 2, 4, 6, 8, 10, 12, 14],
            "scale": [1, .998, .994, .992, .9947, .9788, .9561,
                      .9548, .9479, .952, .9445],
        },
        "runs": [analyze(label, Path(capture), Path(replay))
                 for label, capture, replay in args.pair],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "runs": [{
            "run": run["run"],
            "estimators": {
                name: {key: value for key, value in metrics.items()
                       if key.endswith("_mps") or key == "samples"}
                for name, metrics in run["estimators"].items()
            },
        } for run in report["runs"]],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
