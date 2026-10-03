#!/usr/bin/env python3
"""Compare fixed-40-Hz and source-stamp encoder rates on held-out runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from tools.vehicle_dynamics_learning.build_encoder_state_view import (
    MAX_ALIGNMENT_NS,
    _encoder_angles,
    _latest_source_rate,
    _manifest_bags,
)
from tools.vehicle_dynamics_learning.four_wheel_greybox import (
    DT_S,
    WHEEL_RADIUS_M,
)
from tools.vehicle_dynamics_learning.train_nssm import _load_dataset


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = (
    ROOT / "live_runs/derived_dynamics_learning_20260928/"
    "full_modeling_reset_20261001/replacement_offline_sim_raceline_20261002/"
    "encoder_raw_state_teacher_v1/openplane_dynamics_raw_wheels.npz")
REAR_TRACK_WIDTH_M = 0.236


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def encoder_rate_pair(previous: tuple[int, float],
                      current: tuple[int, float],
                      radius_m: float = WHEEL_RADIUS_M
                      ) -> tuple[float, float, float]:
    """Return variable-stamp rate, fixed-25-ms rate, and source dt."""
    dt_s = (int(current[0]) - int(previous[0])) / 1e9
    if (not np.isfinite(dt_s) or not 0.015 <= dt_s <= 0.035
            or not np.isfinite(radius_m) or radius_m <= 0.0):
        raise ValueError("encoder angle pair is outside the supported interval")
    delta_angle = float(current[1]) - float(previous[1])
    return (delta_angle * radius_m / dt_s,
            delta_angle * radius_m / DT_S,
            dt_s)


def _rate_metrics(contact: np.ndarray, rate: np.ndarray) -> dict[str, Any]:
    valid = (np.isfinite(contact) & np.isfinite(rate)
             & (contact > 0.75) & (rate > 0.75))
    contact, rate = contact[valid], rate[valid]
    if not len(contact):
        return {"samples": 0, "bias_mps": None, "rmse_mps": None,
                "median_ratio": None, "scale_zero_intercept_ls": None}
    return {
        "samples": int(len(contact)),
        "bias_mps": float(np.mean(rate - contact)),
        "rmse_mps": float(np.sqrt(np.mean((rate - contact) ** 2))),
        "median_ratio": float(np.median(rate / contact)),
        "scale_zero_intercept_ls": float(contact @ rate / (contact @ contact)),
    }


def _condition_masks(frames: np.ndarray, rigid: np.ndarray,
                     dt_s: np.ndarray, same_epoch: np.ndarray
                     ) -> dict[str, np.ndarray]:
    u, v, yaw = rigid[:, 7], rigid[:, 8], rigid[:, 12]
    speed = np.hypot(u, v)
    interval = same_epoch & np.isfinite(dt_s) & np.isclose(
        dt_s, DT_S, rtol=0.0, atol=1e-7)
    safe_dt = np.where(interval, dt_s, DT_S)
    ax = np.zeros(len(u), dtype=np.float64)
    ax[1:] = np.diff(u) / safe_dt[1:] - yaw[:-1] * v[:-1]
    return {
        "all_in_0_12mps": (speed <= 12.0),
        "low_slip_straight": (
            (speed >= 2.0) & (speed <= 8.0)
            & (np.abs(frames[:, 3]) < 0.08)
            & (np.abs(yaw) < 0.25) & (np.abs(ax) < 1.5)
            & (frames[:, 4] < 0.15)),
        "turning_speed_3_8mps": (
            (speed >= 3.0) & (speed <= 8.0)
            & ((np.abs(frames[:, 3]) >= 0.10) | (np.abs(yaw) >= 0.25))),
        "high_steering": np.abs(frames[:, 3]) >= 0.30,
    }


def analyze(dataset_path: Path, output_path: Path,
            sidecar_manifest_path: Path | None = None) -> dict[str, Any]:
    dataset_path, output_path = dataset_path.resolve(), output_path.resolve()
    manifest_path = (sidecar_manifest_path.resolve()
                     if sidecar_manifest_path is not None else
                     dataset_path.with_name(
                         dataset_path.stem + "_manifest.json"))
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite cadence report: {output_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for source, expected_hash in manifest["source_manifest_sha256"].items():
        path = Path(source)
        if not path.is_file() or _sha256(path) != expected_hash:
            raise ValueError(f"source manifest missing or changed: {path}")
    bags, _ = _manifest_bags(
        [Path(path) for path in manifest["source_manifest_sha256"]])
    data = _load_dataset(dataset_path)
    frames = np.asarray(data["frames"], dtype=np.float64)
    rigid = np.asarray(data["simulator_rigid_state"], dtype=np.float64)
    times = np.asarray(data["sample_time_ns"], dtype=np.int64)
    run_index = np.empty(len(frames), dtype=np.int32)
    reset_index = np.empty(len(frames), dtype=np.int64)
    for (start_raw, end_raw), run_raw, reset_raw in zip(
            data["bounds"], data["seq_run"], data["sequence_reset_index"]):
        start, end = int(start_raw), int(end_raw)
        run_index[start:end] = int(run_raw)
        reset_index[start:end] = int(reset_raw)
    splits = np.asarray(data["splits"]).astype(str)
    run_ids = np.asarray(data["run_ids"]).astype(str)
    row_reports: dict[str, Any] = {}
    for index, (run_id, split) in enumerate(zip(run_ids, splits)):
        if split not in ("validation", "unseen_practice"):
            continue
        rows = np.flatnonzero(run_index == index)
        if len(rows) < 3:
            continue
        bag = bags.get(str(run_id))
        if bag is None:
            raise ValueError(f"no source bag recorded for held-out run {run_id}")
        # The source-manifest resolver is authoritative; encoder topic constants
        # are read from the same analysis module that built the sidecar.
        from tools import analyze_open_plane_dynamics as source_analysis
        angles = _encoder_angles(bag)
        variable = np.full((len(rows), 2), np.nan, dtype=np.float64)
        fixed = np.full_like(variable, np.nan)
        intervals = np.full_like(variable, np.nan)
        raw_sidecar = np.asarray(data["encoder_raw_surface_mps"],
                                 dtype=np.float64)[rows]
        raw_valid = np.asarray(data["encoder_raw_valid"], dtype=bool)[rows]
        for local, row in enumerate(rows):
            pairs = [_latest_source_rate(angles[topic], int(times[row]))
                     for topic in (source_analysis.LEFT_ENCODER,
                                   source_analysis.RIGHT_ENCODER)]
            if (any(pair is None for pair in pairs)
                    or pairs[0][0][0] != pairs[1][0][0]
                    or pairs[0][1][0] != pairs[1][1][0]):
                continue
            for side, pair in enumerate(pairs):
                dt = (pair[1][0] - pair[0][0]) / 1e9
                if not 0.015 <= dt <= 0.035:
                    continue
                current, fixed_rate, _ = encoder_rate_pair(pair[0], pair[1])
                variable[local, side] = current
                fixed[local, side] = fixed_rate
                intervals[local, side] = dt
        previous_same_epoch = np.r_[False, (
            (run_index[rows[1:]] == run_index[rows[:-1]])
            & (reset_index[rows[1:]] == reset_index[rows[:-1]]))]
        masks = _condition_masks(
            frames[rows], rigid[rows], np.full(len(rows), DT_S),
            previous_same_epoch)
        u = rigid[rows, 7]
        yaw = rigid[rows, 12]
        contact = np.column_stack((
            u - yaw * REAR_TRACK_WIDTH_M / 2.0,
            u + yaw * REAR_TRACK_WIDTH_M / 2.0))
        mask_reports = {}
        for name, condition in masks.items():
            usable = condition & np.isfinite(variable).all(axis=1)
            mask_reports[name] = {
                "valid_encoder_rate_samples": int(usable.sum()),
                "stamp_interval_rate": {
                    side: _rate_metrics(contact[usable, side],
                                        variable[usable, side])
                    for side in range(2)},
                "fixed_25ms_rate": {
                    side: _rate_metrics(contact[usable, side],
                                        fixed[usable, side])
                    for side in range(2)},
                "existing_sidecar_matches_stamp_rate_fraction": float(
                    np.mean(np.isclose(
                        raw_sidecar[usable], variable[usable],
                        rtol=1e-4, atol=1e-5))) if np.any(usable) else None,
            }
        dt_valid = intervals[np.isfinite(intervals)]
        row_reports[str(run_id)] = {
            "split": str(split),
            "frame_count": int(len(rows)),
            "existing_sidecar_valid_rate_frames": int(raw_valid.sum()),
            "reconstructed_common_rate_frames": int(
                np.isfinite(variable).all(axis=1).sum()),
            "source_interval_dt_quantiles_ms": (
                (1000.0 * np.quantile(
                    dt_valid, (0.01, 0.10, 0.50, 0.90, 0.99))).tolist()
                if len(dt_valid) else []),
            "condition_metrics": mask_reports,
        }
    report = {
        "schema_version": 1,
        "purpose": "test whether encoder timestamp jitter should be treated as physical cadence or fixed 40 Hz",
        "dataset": str(dataset_path),
        "dataset_sha256": _sha256(dataset_path),
        "sidecar_manifest": str(manifest_path),
        "sidecar_manifest_sha256": _sha256(manifest_path),
        "held_out_splits_only": ["validation", "unseen_practice"],
        "test_and_final_test_used": False,
        "fixed_cadence_s": DT_S,
        "encoder_radius_m": WHEEL_RADIUS_M,
        "rear_track_width_m": REAR_TRACK_WIDTH_M,
        "source_interval_gate_s": [0.015, 0.035],
        "same_source_pairs_as_existing_sidecar": True,
        "run_reports": row_reports,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--sidecar-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = analyze(args.dataset, args.output, args.sidecar_manifest)
    except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps({
        "output": str(args.output.resolve()),
        "held_out_runs": len(report["run_reports"]),
        "run_ids": sorted(report["run_reports"]),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
