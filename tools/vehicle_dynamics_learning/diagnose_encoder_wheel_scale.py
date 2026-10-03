#!/usr/bin/env python3
"""Cross-check encoder surface-speed scale against simulator body kinematics.

This is an offline diagnostic only. It uses training, validation, and explicitly
unseen-practice rows from the packet-aligned encoder sidecar, and excludes test
and final-test splits. The inferred value is an *effective rolling radius* in
selected low-slip conditions, not a direct geometric wheel measurement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = (
    ROOT / "live_runs/derived_dynamics_learning_20260928/"
    "full_modeling_reset_20261001/replacement_offline_sim_raceline_20261002/"
    "encoder_raw_state_teacher_v1/practice_dynamics_raw_wheels.npz")
DEFAULT_REPORT = (
    ROOT / "live_runs/derived_dynamics_learning_20260928/"
    "full_modeling_reset_20261001/replacement_offline_sim_raceline_20261002/"
    "encoder_raw_state_teacher_v1/wheel_scale_diagnostic_v1.json")
ENCODER_SPEED_RADIUS_M = 0.059
REAR_TRACK_WIDTH_M = 0.236
MINIMUM_RUN_ROWS = 30


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _side_result(expected: np.ndarray, observed: np.ndarray) -> dict[str, Any]:
    positive = (expected > 0.75) & (observed > 0.75)
    expected = expected[positive]
    observed = observed[positive]
    if len(expected) < MINIMUM_RUN_ROWS:
        return {"samples": int(len(expected)), "sufficient_support": False}
    slope = float(expected @ observed / (expected @ expected))
    ratios = observed / expected
    return {
        "samples": int(len(expected)),
        "raw_minus_contact_bias_mps": float(np.mean(observed - expected)),
        "raw_minus_contact_rmse_mps": float(
            np.sqrt(np.mean((observed - expected) ** 2))),
        "raw_to_contact_ratio_median": float(np.median(ratios)),
        "raw_to_contact_scale_zero_intercept_ls": slope,
        "implied_effective_radius_m": (
            ENCODER_SPEED_RADIUS_M / slope if slope > 0.0 else None),
        "sufficient_support": True,
    }


def analyze(dataset_path: Path, report_path: Path,
            comparison_radius_m: float = 0.0325) -> dict[str, Any]:
    dataset_path, report_path = dataset_path.resolve(), report_path.resolve()
    if not np.isfinite(comparison_radius_m) or comparison_radius_m <= 0.0:
        raise ValueError("comparison radius must be finite and positive")
    with np.load(dataset_path, allow_pickle=False) as archive:
        required = (
            "run_ids", "run_splits", "frames", "simulator_rigid_state",
            "dt_s", "frame_run_index", "frame_reset_index",
            "encoder_raw_surface_mps", "encoder_raw_valid",
            "encoder_raw_wheel_radius_m")
        missing = [name for name in required if name not in archive.files]
        if missing:
            raise ValueError(f"encoder sidecar is missing arrays: {missing}")
        data = {name: archive[name].copy() for name in required}

    frames = np.asarray(data["frames"], dtype=np.float64)
    rigid = np.asarray(data["simulator_rigid_state"], dtype=np.float64)
    raw = np.asarray(data["encoder_raw_surface_mps"], dtype=np.float64)
    valid = np.asarray(data["encoder_raw_valid"], dtype=bool)
    frame_run = np.asarray(data["frame_run_index"], dtype=np.int64)
    reset_epoch = np.asarray(data["frame_reset_index"], dtype=np.int64)
    dt_s = np.asarray(data["dt_s"], dtype=np.float64)
    run_ids = np.asarray(data["run_ids"]).astype(str)
    run_splits = np.asarray(data["run_splits"]).astype(str)
    row_count = len(frames)
    if (frames.shape != (row_count, 9)
            or rigid.shape != (row_count, 13)
            or raw.shape != (row_count, 2)
            or valid.shape != (row_count,)
            or frame_run.shape != (row_count,)
            or reset_epoch.shape != (row_count,)
            or dt_s.shape != (row_count,)
            or run_ids.shape != run_splits.shape
            or np.any((frame_run < 0) | (frame_run >= len(run_ids)))):
        raise ValueError("encoder sidecar arrays have inconsistent dimensions")
    declared_radius = float(np.asarray(
        data["encoder_raw_wheel_radius_m"]).reshape(-1)[0])
    if not np.isclose(declared_radius, ENCODER_SPEED_RADIUS_M,
                      rtol=0.0, atol=1e-12):
        raise ValueError("encoder-rate conversion radius differs from this diagnostic")

    u = rigid[:, 7]
    yaw_rate = rigid[:, 12]
    body_speed = np.hypot(rigid[:, 7], rigid[:, 8])
    same_epoch_as_previous = np.r_[False, (
        (frame_run[1:] == frame_run[:-1])
        & (reset_epoch[1:] == reset_epoch[:-1]))]
    interval_ok = (same_epoch_as_previous & np.isfinite(dt_s)
                   & (dt_s >= 0.015) & (dt_s <= 0.035))
    safe_dt = np.where(interval_ok, dt_s, 1.0)
    ax_effective = np.zeros(row_count, dtype=np.float64)
    du_dt = np.r_[0.0, np.diff(u) / safe_dt[1:]]
    v_previous = np.r_[0.0, rigid[:-1, 8]]
    yaw_previous = np.r_[0.0, yaw_rate[:-1]]
    ax_effective[1:] = du_dt[1:] - yaw_previous[1:] * v_previous[1:]

    # Rate labels are interval averages. Approximate rear contact-point speed
    # by the midpoint of adjacent 25 ms simulator-truth body states. For fixed
    # rear wheels, v_contact,x = u_com - r*y_contact.
    u_mid = np.r_[u[0], 0.5 * (u[:-1] + u[1:])]
    yaw_mid = np.r_[yaw_rate[0], 0.5 * (yaw_rate[:-1] + yaw_rate[1:])]
    half_track = REAR_TRACK_WIDTH_M / 2.0
    contact_speed = np.column_stack((
        u_mid - yaw_mid * half_track,
        u_mid + yaw_mid * half_track))
    low_slip_mask = (
        valid & interval_ok
        & np.isfinite(raw).all(axis=1)
        & (body_speed >= 2.0) & (body_speed <= 8.0)
        & (np.abs(frames[:, 3]) < 0.08)
        & (np.abs(yaw_rate) < 0.25)
        & (np.abs(ax_effective) < 1.5)
        & (frames[:, 4] < 0.15))

    per_run: list[dict[str, Any]] = []
    for run_index, (run_id, split) in enumerate(zip(run_ids, run_splits)):
        if split not in ("train", "validation", "unseen_practice"):
            continue
        rows = np.flatnonzero((frame_run == run_index) & low_slip_mask)
        if not len(rows):
            continue
        left = _side_result(contact_speed[rows, 0], raw[rows, 0])
        right = _side_result(contact_speed[rows, 1], raw[rows, 1])
        per_run.append({
            "run_id": run_id,
            "split": split,
            "left": left,
            "right": right,
        })

    split_summaries: dict[str, Any] = {}
    for split in ("train", "validation", "unseen_practice"):
        split_runs = [row for row in per_run if row["split"] == split]
        side_summary: dict[str, Any] = {}
        for side in ("left", "right"):
            radii = [row[side].get("implied_effective_radius_m")
                     for row in split_runs
                     if row[side].get("sufficient_support")]
            radii = [float(radius) for radius in radii if radius is not None]
            side_summary[side] = {
                "independent_run_count": len(radii),
                "median_run_implied_effective_radius_m": (
                    float(np.median(radii)) if radii else None),
                "run_range_implied_effective_radius_m": (
                    [float(min(radii)), float(max(radii))] if radii else None),
            }
        split_summaries[split] = {
            "run_count_with_selected_rows": len(split_runs),
            "low_slip_sample_count": int(sum(
                row["left"]["samples"] for row in split_runs)),
            "per_side": side_summary,
        }

    report = {
        "schema_version": 1,
        "purpose": "offline empirical check of encoder surface-speed scale against simulator-truth rear contact kinematics",
        "dataset": str(dataset_path),
        "dataset_sha256": _sha256(dataset_path),
        "allowed_splits": ["train", "validation", "unseen_practice"],
        "test_and_final_test_used": False,
        "rate_conversion_radius_m": declared_radius,
        "contact_speed_equation": "rear-left = u_com - yaw_rate*track_width/2; rear-right = u_com + yaw_rate*track_width/2",
        "interval_alignment": "adjacent 25 ms truth-state midpoint approximates the source-stamped encoder-rate interval mean",
        "selection": {
            "body_speed_mps": [2.0, 8.0],
            "abs_steering_rad_lt": 0.08,
            "abs_yaw_rate_rps_lt": 0.25,
            "abs_effective_longitudinal_acceleration_mps2_lt": 1.5,
            "throttle_feedback_lt": 0.15,
            "positive_raw_and_contact_speed_mps_gt": 0.75,
            "rear_track_width_m": REAR_TRACK_WIDTH_M,
            "minimum_rows_per_run_and_side": MINIMUM_RUN_ROWS,
        },
        "comparison": {
            "candidate_radius_m": float(comparison_radius_m),
            "raw_to_contact_scale_if_candidate_radius_were_true": (
                ENCODER_SPEED_RADIUS_M / comparison_radius_m),
            "production_odom_wheel_radius_m": 0.059,
            "production_odom_fallback_speed_scale": 0.968,
            "production_odom_fallback_effective_radius_m": 0.059 * 0.968,
            "note": "production odometry may use its speed-dependent scale table; the fallback value is shown only as a reference",
        },
        "split_summaries": split_summaries,
        "per_run": per_run,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--comparison-radius-m", type=float, default=0.0325,
                        help="candidate physical radius to compare; never applied to data")
    args = parser.parse_args()
    try:
        report = analyze(args.dataset, args.output, args.comparison_radius_m)
    except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps({
        "output": str(args.output.resolve()),
        "split_summaries": report["split_summaries"],
        "test_and_final_test_used": report["test_and_final_test_used"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
