#!/usr/bin/env python3
"""Cross-validate a smooth steering-to-yaw-gain map on open-plane blocks.

For each held-out repetition, fit a shape-preserving cubic Hermite yaw-gain
curve from the other repetitions while withholding one intermediate steering
level. This tests whether the nearly deterministic high-speed response has a
useful local steering derivative, rather than a flat lateral-acceleration
cap. It is offline analysis only.
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from tools.evaluate_open_plane_rear_slip_effect import Block, _load_blocks


GAIN_BASE = 2.95
GAIN_TAPER_PER_RAD = 35.6
GAIN_TAPER_START_RAD = 0.41
GAIN_TAPER_END_RAD = 0.46
WITHHELD_ANGLES_RAD = (0.42, 0.46)
VALIDATED_SPEED_MPS = 5.0
MAX_VALIDATED_RMSE_PER_M = 0.03


def _endpoint_slope(h0: float, h1: float, d0: float, d1: float) -> float:
    slope = ((2.0 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
    if slope * d0 <= 0.0:
        return 0.0
    if d0 * d1 < 0.0 and abs(slope) > 3.0 * abs(d0):
        return 3.0 * d0
    return slope


def _pchip_slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if len(x) != len(y) or len(x) < 2 or np.any(np.diff(x) <= 0.0):
        raise ValueError("PCHIP needs equal-length samples at increasing x values")
    h = np.diff(x)
    secants = np.diff(y) / h
    if len(x) == 2:
        return np.array([secants[0], secants[0]])

    slopes = np.zeros_like(x)
    slopes[0] = _endpoint_slope(h[0], h[1], secants[0], secants[1])
    slopes[-1] = _endpoint_slope(h[-1], h[-2], secants[-1], secants[-2])
    for index in range(1, len(x) - 1):
        left = secants[index - 1]
        right = secants[index]
        if left * right <= 0.0:
            slopes[index] = 0.0
            continue
        w_left = 2.0 * h[index] + h[index - 1]
        w_right = h[index] + 2.0 * h[index - 1]
        slopes[index] = (w_left + w_right) / (w_left / left + w_right / right)
    return slopes


def _pchip_value_and_slope(x: np.ndarray, y: np.ndarray,
                           query: float) -> tuple[float, float]:
    if query < x[0] or query > x[-1]:
        raise ValueError("PCHIP does not extrapolate beyond measured steering")
    slopes = _pchip_slopes(x, y)
    interval = min(int(np.searchsorted(x, query, side="right") - 1), len(x) - 2)
    width = x[interval + 1] - x[interval]
    t = (query - x[interval]) / width
    y0, y1 = y[interval], y[interval + 1]
    m0, m1 = slopes[interval], slopes[interval + 1]
    h00 = 2.0 * t**3 - 3.0 * t**2 + 1.0
    h10 = t**3 - 2.0 * t**2 + t
    h01 = -2.0 * t**3 + 3.0 * t**2
    h11 = t**3 - t**2
    value = h00 * y0 + h10 * width * m0 + h01 * y1 + h11 * width * m1
    derivative = (
        (6.0 * t**2 - 6.0 * t) * y0 / width
        + (3.0 * t**2 - 4.0 * t + 1.0) * m0
        + (-6.0 * t**2 + 6.0 * t) * y1 / width
        + (3.0 * t**2 - 2.0 * t) * m1
    )
    return float(value), float(derivative)


def _configured_gain(steering_rad: float) -> float:
    taper = min(max(abs(steering_rad) - GAIN_TAPER_START_RAD, 0.0),
                GAIN_TAPER_END_RAD - GAIN_TAPER_START_RAD)
    return GAIN_BASE - GAIN_TAPER_PER_RAD * taper


def _self_test() -> None:
    x = np.array([0.30, 0.42, 0.50])
    y = np.array([0.68, 0.51, 0.45])
    for x_value, y_value in zip(x, y):
        value, slope = _pchip_value_and_slope(x, y, float(x_value))
        assert math.isclose(value, float(y_value), rel_tol=1e-12)
        assert math.isfinite(slope)
    value, slope = _pchip_value_and_slope(x, y, 0.46)
    assert 0.45 < value < 0.51 and slope < 0.0
    try:
        _pchip_value_and_slope(x, y, 0.25)
    except ValueError:
        pass
    else:
        raise AssertionError("PCHIP must reject unsupported extrapolation")
    print("yaw-gain PCHIP math: PASS (knots, bounded interpolation, finite derivative)")


def _condition_means(blocks: list[Block]) -> dict[tuple[int, int], float]:
    values: dict[tuple[int, int], list[float]] = defaultdict(list)
    for block in blocks:
        key = (round(abs(block.steering_rad) * 100),
               1 if block.steering_rad > 0.0 else -1)
        values[key].append(block.yaw_gain_per_m)
    return {key: float(np.mean(rows)) for key, rows in values.items()}


def evaluate(path: Path) -> tuple[float, float, bool]:
    blocks = _load_blocks(path)
    repetitions = sorted({block.repetition for block in blocks})
    if repetitions != [1, 2, 3]:
        raise ValueError(f"expected repetitions 1,2,3; found {repetitions} in {path}")
    speeds = {round(block.target_speed_mps, 1) for block in blocks}
    if len(speeds) != 1:
        raise ValueError(f"provide one fixed-speed capture per evaluation: {speeds}")
    speed = speeds.pop()
    all_actual: list[float] = []
    all_spline: list[float] = []
    all_configured: list[float] = []
    yaw_rate_sensitivities: list[float] = []
    yaw_rate_errors: list[float] = []

    for heldout_rep in repetitions:
        training = [block for block in blocks if block.repetition != heldout_rep]
        heldout = [block for block in blocks
                   if block.repetition == heldout_rep
                   and any(abs(abs(block.steering_rad) - angle) < 1e-6
                           for angle in WITHHELD_ANGLES_RAD)]
        by_sign = {
            sign: _condition_means([
                block for block in training
                if (1 if block.steering_rad > 0.0 else -1) == sign
            ])
            for sign in (-1, 1)
        }

        for block in heldout:
            sign = 1 if block.steering_rad > 0.0 else -1
            heldout_angle = abs(block.steering_rad)
            means = by_sign[sign]
            train_angles = sorted({angle_centi / 100.0 for angle_centi, _ in means
                                   if angle_centi != round(heldout_angle * 100)})
            train_angles = [angle for angle in train_angles
                            if (round(angle * 100), sign) in means]
            if len(train_angles) != 3:
                raise ValueError(f"expected three interpolation knots for {block.phase}")
            x = np.array(train_angles)
            y = np.array([means[(round(angle * 100), sign)] for angle in train_angles])
            predicted_gain, gain_slope = _pchip_value_and_slope(
                x, y, heldout_angle)
            predicted_gain_rate = _configured_gain(block.steering_rad)
            delta = block.steering_rad
            d_abs_delta_d_delta = sign
            yaw_rate_sensitivity = speed * (
                predicted_gain / math.cos(delta) ** 2
                + math.tan(delta) * gain_slope * d_abs_delta_d_delta
            )
            all_actual.append(block.yaw_gain_per_m)
            all_spline.append(predicted_gain)
            all_configured.append(predicted_gain_rate)
            yaw_rate_sensitivities.append(yaw_rate_sensitivity)
            yaw_rate_errors.append(speed * math.tan(delta) *
                                   (predicted_gain - block.yaw_gain_per_m))

    actual = np.asarray(all_actual)
    spline = np.asarray(all_spline)
    configured = np.asarray(all_configured)
    spline_rmse = float(np.sqrt(np.mean((actual - spline) ** 2)))
    configured_rmse = float(np.sqrt(np.mean((actual - configured) ** 2)))
    max_gain_error = float(np.max(np.abs(actual - spline)))
    yaw_rate_rmse = float(np.sqrt(np.mean(np.square(yaw_rate_errors))))
    positive_sensitivity = all(value > 0.0 and math.isfinite(value)
                               for value in yaw_rate_sensitivities)
    passed = (speed == VALIDATED_SPEED_MPS
              and spline_rmse <= MAX_VALIDATED_RMSE_PER_M
              and positive_sensitivity)
    print(f"{path}: leave-one-repetition-out intermediate-angle predictions="
          f"{len(actual)}; speed={speed:.1f} m/s")
    print(f"  PCHIP yaw-gain RMSE={spline_rmse:.5f} 1/m; configured gain RMSE="
          f"{configured_rmse:.5f} 1/m")
    print(f"  PCHIP max |yaw-gain error|={max_gain_error:.5f} 1/m; "
          f"yaw-rate RMSE={yaw_rate_rmse:.5f} rad/s")
    print(f"  yaw-rate sensitivity d(r_ss)/d(steer): min/max="
          f"{min(yaw_rate_sensitivities):.4f}/{max(yaw_rate_sensitivities):.4f} 1/s")
    if speed == VALIDATED_SPEED_MPS:
        print("  high-speed gate: PCHIP RMSE≤0.03 1/m and finite positive steering "
              f"sensitivity: {'PASS' if passed else 'FAIL'}")
    else:
        print("  high-speed acceptance gate: not applicable; diagnostic only")
    return spline_rmse, configured_rmse, passed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bags", nargs="*", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        return 0
    if not args.bags:
        parser.error("provide fixed-speed bags or use --self-test")
    results = [evaluate(path) for path in args.bags]
    return 0 if any(result[2] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
