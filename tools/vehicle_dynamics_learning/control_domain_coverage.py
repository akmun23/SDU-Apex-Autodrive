#!/usr/bin/env python3
"""Audit independent-run coverage in the raceline-relevant 0–12 m/s domain."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = REPO_ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "plant_teacher_mixed_dataset_full3d_reset_safe_20261001/"
    "openplane_dynamics.npz")
DEFAULT_OUTPUT = REPO_ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "full_modeling_reset_20261001/control_domain_coverage_0_12_20261001.json")

SPEED_EDGES = (0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0)
STEERING_BINS = (
    ("left_high", -0.524, -0.40),
    ("left_mid", -0.40, -0.25),
    ("left_low", -0.25, -0.10),
    ("near_zero", -0.10, 0.10),
    ("right_low", 0.10, 0.25),
    ("right_mid", 0.25, 0.40),
    ("right_high", 0.40, 0.524),
)
THROTTLE_BINS = (
    ("brake_negative", -1.01, -0.01),
    ("zero_coast", -0.01, 0.01),
    ("low_positive", 0.01, 0.25),
    ("mid_positive", 0.25, 0.60),
    ("high_positive", 0.60, 1.01),
)


def _bin(value: float, bounds: tuple[tuple[str, float, float], ...]) -> str:
    for name, lower, upper in bounds:
        if lower <= value < upper or (name == "right_high" and value <= upper):
            return name
    return "outside_mechanical_range"


def analyze(dataset_path: Path, output_path: Path) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    archive = np.load(dataset_path, allow_pickle=False)
    if int(archive["schema_version"][0]) != 7:
        raise ValueError("coverage audit requires the handoff's schema-7 view")
    required = {
        "frames", "dt_s", "sequence_bounds", "sequence_run_index",
        "run_ids", "run_splits", "run_families", "frame_run_index",
        "simulator_rigid_state",
    }
    missing = required - set(archive.files)
    if missing:
        raise ValueError(f"schema-7 dataset lacks metadata: {sorted(missing)}")

    frames = archive["frames"].astype(np.float64, copy=False)
    rigid = archive["simulator_rigid_state"].astype(np.float64, copy=False)
    bounds = archive["sequence_bounds"].astype(np.int64, copy=False)
    sequence_run = archive["sequence_run_index"].astype(np.int32, copy=False)
    run_ids = archive["run_ids"].astype(str)
    splits = archive["run_splits"].astype(str)
    families = archive["run_families"].astype(str)
    frame_run = archive["frame_run_index"].astype(np.int32, copy=False)
    dt = archive["dt_s"].astype(np.float64, copy=False)
    if frames.shape[1] != 9 or rigid.shape != (len(frames), 13):
        raise ValueError("unexpected schema-7 state or rigid-label layout")
    if (frame_run.shape != (len(frames),) or len(splits) != len(run_ids)
            or len(families) != len(run_ids)):
        raise ValueError("schema-7 run metadata does not align")

    # Per-sequence command differences avoid artificial steps at resets.
    throttle_slew = np.full(len(frames), np.nan, dtype=np.float64)
    for start_raw, end_raw in bounds:
        start, end = int(start_raw), int(end_raw)
        local_dt = dt[start + 1:end]
        if len(local_dt):
            throttle_slew[start + 1:end] = np.diff(
                frames[start:end, 8]) / local_dt

    speed = np.hypot(rigid[:, 7], rigid[:, 8])
    steering_actual = frames[:, 3]
    throttle_command = frames[:, 8]
    eligible = (np.isfinite(speed) & np.isfinite(steering_actual)
                & np.isfinite(throttle_command)
                & (speed >= SPEED_EDGES[0]) & (speed <= SPEED_EDGES[-1]))

    groups: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    run_speed_sets: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    steering_speed_runs: dict[tuple[str, str], set[str]] = defaultdict(set)
    throttle_range = [float(np.min(throttle_command[eligible])),
                      float(np.max(throttle_command[eligible]))]
    negative_throttle = eligible & (throttle_command < -0.01)
    negative_slew = eligible & np.isfinite(throttle_slew) & (throttle_slew < -0.05)

    for index in np.flatnonzero(eligible):
        run_index = int(frame_run[index])
        run_id = str(run_ids[run_index])
        family = str(families[run_index])
        split = str(splits[run_index])
        speed_index = min(int(np.searchsorted(SPEED_EDGES, speed[index], side="right") - 1),
                          len(SPEED_EDGES) - 2)
        speed_bin = f"{SPEED_EDGES[speed_index]:g}-{SPEED_EDGES[speed_index + 1]:g}mps"
        steering_bin = _bin(float(steering_actual[index]), STEERING_BINS)
        throttle_bin = _bin(float(throttle_command[index]), THROTTLE_BINS)
        if np.isfinite(throttle_slew[index]):
            slew = float(throttle_slew[index])
            slew_bin = ("cut" if slew < -0.05 else
                        "increase" if slew > 0.05 else "steady")
        else:
            slew_bin = "sequence_start"
        key = (family, split, speed_bin, steering_bin,
               f"{throttle_bin}:{slew_bin}")
        row = groups.setdefault(key, {"sample_count": 0, "runs": set()})
        row["sample_count"] += 1
        row["runs"].add(run_id)
        run_speed_sets[(family, split, speed_bin)].add(run_id)
        steering_speed_runs[(speed_bin, steering_bin)].add(run_id)

    coverage_rows = [
        {
            "run_family": family,
            "split": split,
            "speed_bin": speed_bin,
            "actual_steering_bin": steering_bin,
            "throttle_command_and_slew_bin": command_bin,
            "samples": int(value["sample_count"]),
            "independent_run_count": len(value["runs"]),
            "run_ids": sorted(value["runs"]),
        }
        for (family, split, speed_bin, steering_bin, command_bin), value
        in sorted(groups.items())
    ]
    speed_summary = []
    for index in range(len(SPEED_EDGES) - 1):
        speed_bin = f"{SPEED_EDGES[index]:g}-{SPEED_EDGES[index + 1]:g}mps"
        speed_summary.append({
            "speed_bin": speed_bin,
            "sample_count": int(np.count_nonzero(
                eligible & (speed >= SPEED_EDGES[index])
                & (speed < SPEED_EDGES[index + 1] if index < len(SPEED_EDGES) - 2
                   else speed <= SPEED_EDGES[index + 1]))),
            "run_count_by_family_split": {
                f"{family}/{split}": len(run_ids_for_bin)
                for (family, split, local_bin), run_ids_for_bin
                in sorted(run_speed_sets.items()) if local_bin == speed_bin
            },
            "steering_coverage_run_count": {
                steering_bin: len(run_ids_for_bin)
                for (local_bin, steering_bin), run_ids_for_bin
                in sorted(steering_speed_runs.items()) if local_bin == speed_bin
            },
        })

    digest = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    report = {
        "schema_version": 1,
        "dataset": str(dataset_path.resolve()),
        "dataset_sha256": digest,
        "scope": "raceline-relevant speeds from 0 through 12 m/s",
        "speed_bin_edges_mps": list(SPEED_EDGES),
        "actual_steering_bins_rad": [
            {"name": name, "lower_inclusive": lower,
             "upper_exclusive": upper}
            for name, lower, upper in STEERING_BINS
        ],
        "throttle_bins": [
            {"name": name, "lower_inclusive": lower,
             "upper_exclusive": upper}
            for name, lower, upper in THROTTLE_BINS
        ],
        "throttle_command_min_max_in_domain": throttle_range,
        "domain_sample_count": int(np.count_nonzero(eligible)),
        "negative_throttle_sample_count": int(np.count_nonzero(negative_throttle)),
        "negative_throttle_run_count": len(set(
            str(run_ids[int(frame_run[i])]) for i in np.flatnonzero(negative_throttle))),
        "downward_throttle_slew_sample_count": int(np.count_nonzero(negative_slew)),
        "downward_throttle_slew_run_count": len(set(
            str(run_ids[int(frame_run[i])]) for i in np.flatnonzero(negative_slew))),
        "speed_summary": speed_summary,
        "conditioned_coverage": coverage_rows,
        "warning": (
            "Samples are temporally correlated; independent run counts, not row counts, "
            "are the generalization units. Steering bins use measured steering, "
            "and this audit does not by itself establish causal excitation."),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = analyze(args.dataset, args.output)
    print(json.dumps({
        "samples_0_12mps": report["domain_sample_count"],
        "negative_throttle_samples": report["negative_throttle_sample_count"],
        "downward_throttle_slew_samples": report["downward_throttle_slew_sample_count"],
        "speed_bins": report["speed_summary"],
        "output": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
