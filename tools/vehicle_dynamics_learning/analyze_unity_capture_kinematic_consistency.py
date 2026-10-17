#!/usr/bin/env python3
"""Compare captured pose derivatives with twist, while reporting frame provenance.

Legacy captures stored Rigidbody pose but converted twist through Transform,
which can be an interpolated presentation frame. Their residuals are only
diagnostics under a same-frame assumption. Schema-v2 captures log both frames
and Rigidbody-frame twist, making the direct closure check interpretable.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


FIELDS = (
    "sim_time_s",
    "pos_world_x_m", "pos_world_y_m", "pos_world_z_m",
    "rot_world_x", "rot_world_y", "rot_world_z", "rot_world_w",
    "vel_body_x_mps", "vel_body_y_mps", "vel_body_z_mps",
    "omega_body_x_rps", "omega_body_y_rps", "omega_body_z_rps",
)


def rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vectors by normalized xyzw quaternions, row-wise."""
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    u = q[:, :3]
    s = q[:, 3:4]
    return 2.0 * np.sum(u * v, axis=1, keepdims=True) * u \
        + (s * s - np.sum(u * u, axis=1, keepdims=True)) * v \
        + 2.0 * s * np.cross(u, v)


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    out = q.copy()
    out[:, :3] *= -1.0
    return out


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    av, aw = a[:, :3], a[:, 3:4]
    bv, bw = b[:, :3], b[:, 3:4]
    return np.concatenate((aw * bv + bw * av + np.cross(av, bv),
                           aw * bw - np.sum(av * bv, axis=1, keepdims=True)), axis=1)


def rotation_vector(q: np.ndarray) -> np.ndarray:
    """Shortest-arc rotation vector for xyzw quaternions."""
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    flip = q[:, 3] < 0
    q[flip] *= -1
    v = q[:, :3]
    vn = np.linalg.norm(v, axis=1)
    angle = 2.0 * np.arctan2(vn, np.clip(q[:, 3], -1.0, 1.0))
    scale = np.divide(angle, vn, out=np.full_like(angle, 2.0), where=vn > 1e-12)
    return v * scale[:, None]


def describe(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    return {
        "count": int(finite.size),
        "median": float(np.median(finite)),
        "p95": float(np.percentile(finite, 95)),
        "p99": float(np.percentile(finite, 99)),
        "p99_9": float(np.percentile(finite, 99.9)),
        "max": float(np.max(finite)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--inspect-rows", type=int, nargs="*", default=[],
                        help="inspect zero-based interval start rows")
    args = parser.parse_args()

    with args.capture.open(newline="") as f:
        metadata: dict[str, str] = {}
        while True:
            line = f.readline()
            if not line:
                raise ValueError("CSV has no column header")
            if line.startswith("#"):
                key, separator, value = line[1:].strip().partition("=")
                if separator:
                    metadata[key.strip()] = value.strip()
            else:
                header = next(csv.reader([line]))
                break
        indices = [header.index(name) for name in FIELDS]
        rows = []
        reader = csv.reader(f)
        for row in reader:
            if row:
                rows.append([float(row[i]) for i in indices])

    data = np.asarray(rows, dtype=np.float64)
    if len(data) < 2:
        raise ValueError("need at least two capture rows")
    t = data[:, 0]
    p = data[:, 1:4]
    q = data[:, 4:8]
    vb = data[:, 8:11]
    wb = data[:, 11:14]
    dt = np.diff(t)
    valid = np.isfinite(dt) & (dt > 0)
    if not np.all(valid):
        bad = np.flatnonzero(~valid)[:10].tolist()
        raise ValueError(f"non-increasing/non-finite time interval(s), first rows {bad}")

    # Position derivative is compared with the trapezoidal world velocity over
    # the interval. This avoids assuming Unity's semi-implicit integration order.
    vw = rotate(q, vb)
    p_rate = np.diff(p, axis=0) / dt[:, None]
    v_mid = 0.5 * (vw[:-1] + vw[1:])
    p_residual = p_rate - v_mid
    p_error = np.linalg.norm(p_residual, axis=1)

    # Compare shortest-arc world attitude change with world angular velocity.
    dq_world = quat_multiply(q[1:], quat_conjugate(q[:-1]))
    rot_rate_world = rotation_vector(dq_world) / dt[:, None]
    omega_world = rotate(q, wb)
    omega_mid = 0.5 * (omega_world[:-1] + omega_world[1:])
    w_residual = rot_rate_world - omega_mid
    w_error = np.linalg.norm(w_residual, axis=1)

    # Also report endpoint variants: capture/update ordering can make one side
    # a better timestamp match than the trapezoidal estimate.
    p_error_left = np.linalg.norm(p_rate - vw[:-1], axis=1)
    p_error_right = np.linalg.norm(p_rate - vw[1:], axis=1)
    w_error_left = np.linalg.norm(rot_rate_world - omega_world[:-1], axis=1)
    w_error_right = np.linalg.norm(rot_rate_world - omega_world[1:], axis=1)

    # Respawn/reset teleports are not vehicle motion. At this 1 kHz capture
    # rate, >10 cm translation or >0.1 rad rotation in one interval cannot be
    # ordinary car motion; report them separately rather than letting their
    # finite differences dominate the continuous-dynamics statistics.
    translation = np.linalg.norm(np.diff(p, axis=0), axis=1)
    angle = np.linalg.norm(rotation_vector(dq_world.copy()), axis=1)
    discontinuity = (translation > 0.1) | (angle > 0.1)
    continuous = ~discontinuity

    def top_rows(errors: np.ndarray, mask: np.ndarray | None = None) -> list[dict[str, object]]:
        candidates = np.flatnonzero(mask) if mask is not None else np.arange(errors.size)
        if not candidates.size:
            return []
        indices_top = candidates[np.argsort(errors[candidates])[-args.top:][::-1]]
        return [{
            "interval_start_row": int(i),
            "sim_time_s": float(t[i]),
            "dt_s": float(dt[i]),
            "error": float(errors[i]),
            "position_rate_world_mps": p_rate[i].tolist(),
            "position_twist_mid_world_mps": v_mid[i].tolist(),
            "rotation_rate_world_radps": rot_rate_world[i].tolist(),
            "angular_twist_mid_world_radps": omega_mid[i].tolist(),
        } for i in indices_top]

    inspected = []
    for i in args.inspect_rows:
        if i < 0 or i >= len(dt):
            raise ValueError(f"inspection row {i} is outside [0, {len(dt) - 1}]")
        inspected.append({
            "interval_start_row": i,
            "sim_time_s": float(t[i]),
            "dt_s": float(dt[i]),
            "translation_m": float(translation[i]),
            "rotation_angle_rad": float(angle[i]),
            "position_residual_mps": float(p_error[i]),
            "position_right_endpoint_residual_mps": float(p_error_right[i]),
            "angular_residual_radps": float(w_error[i]),
            "angular_right_endpoint_residual_radps": float(w_error_right[i]),
            "position_rate_world_mps": p_rate[i].tolist(),
            "world_velocity_left_mps": vw[i].tolist(),
            "world_velocity_right_mps": vw[i + 1].tolist(),
            "rotation_rate_world_radps": rot_rate_world[i].tolist(),
            "world_angular_velocity_left_radps": omega_world[i].tolist(),
            "world_angular_velocity_right_radps": omega_world[i + 1].tolist(),
        })

    result = {
        "capture": str(args.capture),
        "capture_schema_version": int(metadata.get("capture_schema_version", "1")),
        "twist_frame_interpretation": (
            "Rigidbody rotation frame; direct pose/twist closure is comparable"
            if int(metadata.get("capture_schema_version", "1")) >= 2 else
            "legacy Transform-local twist paired with Rigidbody pose; residuals "
            "are conditional diagnostics, not a clean physics-closure error"),
        "sample_count": int(len(data)),
        "interval_count": int(len(dt)),
        "fixed_dt_median_s": float(np.median(dt)),
        "fixed_dt_min_s": float(np.min(dt)),
        "fixed_dt_max_s": float(np.max(dt)),
        "pose_discontinuity_intervals_gt_0p1m_or_0p1rad": int(np.count_nonzero(discontinuity)),
        "position_derivative_residual_mps": describe(p_error),
        "position_derivative_residual_left_mps": describe(p_error_left),
        "position_derivative_residual_right_mps": describe(p_error_right),
        "angular_derivative_residual_radps": describe(w_error),
        "angular_derivative_residual_left_radps": describe(w_error_left),
        "angular_derivative_residual_right_radps": describe(w_error_right),
        "continuous_position_derivative_residual_mps": describe(p_error[continuous]),
        "continuous_position_derivative_residual_right_mps": describe(p_error_right[continuous]),
        "continuous_angular_derivative_residual_radps": describe(w_error[continuous]),
        "continuous_angular_derivative_residual_right_radps": describe(w_error_right[continuous]),
        "position_intervals_over_0p1mps": int(np.count_nonzero(p_error > 0.1)),
        "position_intervals_over_1mps": int(np.count_nonzero(p_error > 1.0)),
        "angular_intervals_over_0p1radps": int(np.count_nonzero(w_error > 0.1)),
        "angular_intervals_over_1radps": int(np.count_nonzero(w_error > 1.0)),
        "largest_position_residuals": top_rows(p_error, continuous),
        "largest_angular_residuals": top_rows(w_error, continuous),
        "largest_right_endpoint_angular_residuals": top_rows(w_error_right, continuous),
        "inspected_intervals": inspected,
    }
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
