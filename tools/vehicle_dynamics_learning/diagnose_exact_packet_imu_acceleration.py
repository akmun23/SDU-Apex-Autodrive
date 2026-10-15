#!/usr/bin/env python3
"""Compare exact Odom-consumed IMU samples with packet-matched Unity truth.

Ground-truth body velocity is used only to infer the mean rear-axle
acceleration over each consecutive 25 ms packet interval. Production replay
diagnostics supply the exact IMU acceleration/yaw-rate values consumed by
Odom, keyed by their original packet source stamp. No model is fitted.
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
COM_TO_REAR_AXLE_M = 0.15532
DIAGNOSTIC_SIZE = 42
SOURCE_STAMP_INDEX = 28
AX_INDEX = 9
AY_INDEX = 10
YAW_RATE_INDEX = 11


def _metrics(errors: np.ndarray) -> dict[str, Any]:
    if errors.size == 0:
        return {"samples": 0, "bias": [None, None], "rmse": [None, None],
                "p95_abs": [None, None]}
    return {
        "samples": int(errors.shape[0]),
        "bias": np.mean(errors, axis=0).tolist(),
        "rmse": np.sqrt(np.mean(errors ** 2, axis=0)).tolist(),
        "p95_abs": np.quantile(np.abs(errors), 0.95, axis=0).tolist(),
    }


def _first_crossing(samples: list[MotionSample], error: np.ndarray,
                    threshold: float) -> dict[str, Any] | None:
    indices = np.flatnonzero(np.abs(error) >= threshold)
    if not len(indices):
        return None
    index = int(indices[0])
    return {
        "packet_sequence": int(samples[index].packet_sequence),
        "elapsed_from_first_joined_packet_s": float(
            (samples[index].packet_sequence - samples[0].packet_sequence) * DT_S),
        "signed_error": float(error[index]),
    }


def _state_arrays(samples: list[MotionSample], diagnostics: dict[int, np.ndarray]
                  ) -> tuple[list[MotionSample], np.ndarray, list[int]]:
    diagnostic_stamps = sorted(diagnostics)
    joined = []
    rows = []
    stamp_errors = []
    used_stamps = set()
    for sample in samples:
        insertion = bisect.bisect_left(diagnostic_stamps, sample.source_stamp_ns)
        candidates = [diagnostic_stamps[index]
                      for index in (insertion - 1, insertion)
                      if 0 <= index < len(diagnostic_stamps)]
        if not candidates:
            continue
        diagnostic_stamp = min(
            candidates, key=lambda stamp: abs(stamp - sample.source_stamp_ns))
        stamp_error = abs(diagnostic_stamp - sample.source_stamp_ns)
        # The Float64MultiArray diagnostics lose sub-microsecond precision
        # when carrying Unix-epoch seconds. This tolerance recovers that
        # serialization rounding; it is far below the 25 ms packet interval.
        if stamp_error > 2_000 or diagnostic_stamp in used_stamps:
            continue
        row = diagnostics[diagnostic_stamp]
        if row.shape != (DIAGNOSTIC_SIZE,):
            raise ValueError(f"unexpected Odom diagnostic shape: {row.shape}")
        used_stamps.add(diagnostic_stamp)
        joined.append(sample)
        rows.append(row)
        stamp_errors.append(stamp_error)
    if len(joined) < 2:
        raise ValueError("fewer than two exact Odom-diagnostic/truth packet joins")
    return joined, np.stack(rows), stamp_errors


def _regime_metrics(rows: list[tuple[float, float, np.ndarray]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    speed_edges = (0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.1)
    steer_edges = (0.0, 0.1, 0.2, 0.3, 0.4, 0.51)
    for si, (slow, shigh) in enumerate(zip(speed_edges, speed_edges[1:])):
        for di, (dlow, dhigh) in enumerate(zip(steer_edges, steer_edges[1:])):
            values = [error for speed, steering, error in rows
                      if slow <= speed < shigh and dlow <= steering < dhigh]
            if len(values) >= 20:
                result[f"speed_{slow:g}_{shigh:g}_steer_{dlow:g}_{dhigh:g}"] = \
                    _metrics(np.stack(values))
    return result


def analyze_pair(label: str, capture_path: Path, replay_path: Path) -> dict[str, Any]:
    capture = load_capture(
        capture_path, include_nonvalid_phases=True, continuous_phased_run=True)
    samples_by_seq: dict[int, MotionSample] = {}
    for sequence in capture.sequences:
        for sample in sequence:
            if sample.packet_sequence >= 0 and np.isfinite(sample.state).all():
                samples_by_seq[sample.packet_sequence] = sample
    diagnostics = _read_diagnostics(replay_path, "/replayed_odom_diagnostics")
    samples, diag, stamp_errors = _state_arrays(
        [samples_by_seq[key] for key in sorted(samples_by_seq)], diagnostics)

    # Each row is exact packet data consumed by Odom. Reconstruct rear-axle
    # acceleration from the Unity IMU acceleration at the COM and its gyro.
    ax_com = diag[:, AX_INDEX]
    ay_com = diag[:, AY_INDEX]
    gyro_r = diag[:, YAW_RATE_INDEX]
    if not np.isfinite(diag[:, [AX_INDEX, AY_INDEX, YAW_RATE_INDEX]]).all():
        raise ValueError(f"{label}: nonfinite consumed IMU diagnostics")

    errors: dict[str, list[np.ndarray]] = {
        "previous_packet": [],
        "current_packet": [],
        "interval_mean": [],
    }
    regimes: dict[str, list[tuple[float, float, np.ndarray]]] = {
        name: [] for name in errors
    }
    source_stamp_deltas: list[float] = []
    sequence_gaps = 0
    for index in range(1, len(samples)):
        previous, current = samples[index - 1], samples[index]
        if current.packet_sequence != previous.packet_sequence + 1:
            sequence_gaps += 1
            continue
        u0, v0, _ = previous.state
        u1, v1, _ = current.state
        r0, r1 = gyro_r[index - 1], gyro_r[index]
        yaw_accel = (r1 - r0) / DT_S

        # Body-frame translational acceleration at the rear axle follows from
        # Unity's COM acceleration and rigid-body offset. Ground truth gives
        # the interval-mean acceleration independently of its IMU endpoint.
        r_mid = 0.5 * (r0 + r1)
        u_mid = 0.5 * (u0 + u1)
        v_mid = 0.5 * (v0 + v1)
        truth_ax_rear = (u1 - u0) / DT_S - r_mid * v_mid
        truth_ay_rear = (v1 - v0) / DT_S + r_mid * u_mid
        candidates = {
            "previous_packet": np.asarray((
                ax_com[index - 1] + COM_TO_REAR_AXLE_M * r0 ** 2,
                ay_com[index - 1] - COM_TO_REAR_AXLE_M * yaw_accel)),
            "current_packet": np.asarray((
                ax_com[index] + COM_TO_REAR_AXLE_M * r1 ** 2,
                ay_com[index] - COM_TO_REAR_AXLE_M * yaw_accel)),
            "interval_mean": np.asarray((
                0.5 * (ax_com[index - 1] + ax_com[index]) +
                COM_TO_REAR_AXLE_M * 0.5 * (r0 ** 2 + r1 ** 2),
                0.5 * (ay_com[index - 1] + ay_com[index]) -
                COM_TO_REAR_AXLE_M * yaw_accel)),
        }
        truth_acceleration = np.asarray((truth_ax_rear, truth_ay_rear))
        speed = float(np.hypot(u_mid, v_mid))
        steering = float(abs(current.actuators[0]))
        source_stamp_deltas.append(
            (current.source_stamp_ns - previous.source_stamp_ns) / 1e9)
        for name, prediction in candidates.items():
            error = prediction - truth_acceleration
            errors[name].append(error)
            regimes[name].append((speed, steering, error))

    if not errors["current_packet"]:
        raise ValueError(f"{label}: no consecutive source packets")
    truth_metrics = {
        name: _metrics(np.stack(values)) for name, values in errors.items()
    }
    truth_state = np.stack([sample.state for sample in samples])
    observer_rows = {
        "pre_wheel_u": diag[:, 40],
        "published_u": diag[:, 7],
        "published_v": diag[:, 8],
        "imu_yaw_rate": diag[:, YAW_RATE_INDEX],
        "rolling_wheel_surface_speed": diag[:, 4],
        "current_packet_wheel_surface_speed": diag[:, 25],
    }
    observer_truth_indices = {
        "pre_wheel_u": 0,
        "published_u": 0,
        "published_v": 1,
        "imu_yaw_rate": 2,
        "rolling_wheel_surface_speed": 0,
        "current_packet_wheel_surface_speed": 0,
    }
    observer_errors = {
        name: values - truth_state[:, observer_truth_indices[name]]
        for name, values in observer_rows.items()
    }
    valid_pre = (diag[:, 41] > 0.5)
    used_wheel = (diag[:, 12] > 0.5) & valid_pre
    update_effect = {
        "samples_with_valid_pre_wheel_state_and_wheel_update": int(used_wheel.sum()),
        "pre_wheel_u_error": _metrics(
            observer_errors["pre_wheel_u"][used_wheel, None]),
        "published_u_error": _metrics(
            observer_errors["published_u"][used_wheel, None]),
        "mean_absolute_u_error_change_after_update_mps": (
            float(np.mean(np.abs(observer_errors["published_u"][used_wheel]) -
                          np.abs(observer_errors["pre_wheel_u"][used_wheel])))
            if np.any(used_wheel) else None),
        "interpretation": (
            "Compares the prior causal prediction with the published estimate "
            "only where diagnostics mark both the pre-wheel state and an actual "
            "wheel update as valid."),
    }
    first_divergence = {
        name: {
            str(threshold): _first_crossing(samples, error, threshold)
            for threshold in thresholds
        }
        for name, error, thresholds in (
            ("published_u_error_mps", observer_errors["published_u"],
             (0.05, 0.1, 0.2, 0.5)),
            ("published_v_error_mps", observer_errors["published_v"],
             (0.02, 0.05, 0.1, 0.2)),
            ("yaw_rate_error_radps", observer_errors["imu_yaw_rate"],
             (0.02, 0.05, 0.1, 0.2)),
        )
    }
    observer_state_metrics = {}
    for name, error in observer_errors.items():
        mask = valid_pre if name == "pre_wheel_u" else np.ones(len(error), dtype=bool)
        observer_state_metrics[name] = _metrics(error[mask, None])
    return {
        "label": label,
        "capture": str(capture_path),
        "replay_diagnostics": str(replay_path),
        "exactly_joined_packets": len(samples),
        "diagnostic_source_stamp_float_rounding_abs_error_ns": {
            "max": max(stamp_errors),
            "p95": float(np.quantile(stamp_errors, 0.95)),
        },
        "consecutive_packet_intervals": len(errors["current_packet"]),
        "packet_sequence_gaps": sequence_gaps,
        "source_stamp_interval_s": {
            "median": float(np.median(source_stamp_deltas)),
            "p05": float(np.quantile(source_stamp_deltas, 0.05)),
            "p95": float(np.quantile(source_stamp_deltas, 0.95)),
            "note": "diagnostic only; all physical differences use fixed 25 ms",
        },
        "rear_axle_acceleration_estimate_minus_truth_interval_mean_mps2":
            truth_metrics,
        "replayed_observer_minus_unity_body_state": {
            name: metrics for name, metrics in observer_state_metrics.items()
        },
        "first_error_threshold_crossing": first_divergence,
        "wheel_update_effect_on_forward_speed": update_effect,
        "wheel_speed_warning": (
            "The rolling/current encoder-derived surface speed is compared to "
            "Unity body-u only as an observer residual. It is not independent "
            "wheel-contact velocity truth, and this difference is not a direct "
            "measurement of tire slip."),
        "by_speed_and_absolute_steering": {
            name: _regime_metrics(rows) for name, rows in regimes.items()
        },
        "interpretation": (
            "The response with the smallest error identifies whether the exact "
            "Odom-consumed IMU packet behaves more like the previous endpoint, "
            "current endpoint, or a two-endpoint interval mean at 40 Hz. This "
            "does not identify tire force or prove long-horizon accuracy."),
    }


def _parse_pair(value: str) -> tuple[str, Path, Path]:
    try:
        label, paths = value.split("=", 1)
        capture, replay = paths.split(",", 1)
    except (ValueError, IndexError) as exc:
        raise argparse.ArgumentTypeError(
            "pair must be LABEL=CAPTURE.db3,REPLAY.db3") from exc
    return label, Path(capture), Path(replay)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", action="append", required=True, type=_parse_pair,
                        metavar="LABEL=CAPTURE.db3,REPLAY.db3")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = {
        "method": "exact packet join; fixed 25 ms; no fitted coefficients",
        "state_truth": "Unity packet-level /odom rear-axle [u,v,r]",
        "imu_input": "production Odom replay diagnostics at the same source stamp",
        "com_to_rear_axle_m": COM_TO_REAR_AXLE_M,
        "runs": [analyze_pair(label, capture, replay)
                 for label, capture, replay in args.pair],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "runs": [{
            "label": run["label"],
            "intervals": run["consecutive_packet_intervals"],
            "errors_mps2": {
                name: metrics["rmse"] for name, metrics in
                run["rear_axle_acceleration_estimate_minus_truth_interval_mean_mps2"].items()
            },
        } for run in report["runs"]],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
