#!/usr/bin/env python3
"""Count independent race-domain support by speed, steering, slip, and command."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = REPO_ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "plant_teacher_race_domain_v1/cooldown_2s/openplane_dynamics.npz")
DEFAULT_OUTPUT = REPO_ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "race_domain_coverage_20261001.json")
SPEED_EDGES = (0.0, 3.0, 5.0, 7.0, 9.0, 10.0, 11.0, 12.0)
STEERING_EDGES = (0.0, 0.10, 0.20, 0.30, 0.40, 0.5240001)
THROTTLE_EDGES = (-1.001, -0.05, 0.05, 0.20, 0.40, 0.60, 0.80, 1.001)


def _interval_labels(values: np.ndarray, edges: tuple[float, ...],
                     suffix: str) -> np.ndarray:
    labels = np.full(len(values), "outside", dtype="U32")
    finite = np.isfinite(values)
    indices = np.searchsorted(np.asarray(edges), values[finite], side="right") - 1
    valid = (indices >= 0) & (indices < len(edges) - 1)
    selected = np.flatnonzero(finite)
    for row, bin_index in zip(selected[valid], indices[valid]):
        labels[row] = (
            f"{edges[bin_index]:g}-{edges[bin_index + 1]:g}{suffix}")
    # Include exactly the declared upper edge in the last bin.
    on_upper = finite & (values == edges[-1])
    labels[on_upper] = f"{edges[-2]:g}-{edges[-1]:g}{suffix}"
    return labels


def _command_direction(command: np.ndarray, acceleration: np.ndarray,
                       sequence_bounds: np.ndarray) -> np.ndarray:
    del acceleration  # Braking is a command label, not inferred from deceleration.
    slew = np.full(len(command), np.nan, dtype=np.float64)
    for start_raw, end_raw in sequence_bounds:
        start, end = int(start_raw), int(end_raw)
        if end - start > 1:
            slew[start + 1:end] = np.diff(command[start:end]) / 0.025
    result = np.full(len(command), "steady_positive_drive", dtype="U32")
    finite = np.isfinite(command)
    result[~finite] = "unknown"
    result[finite & (command < -1e-6)] = "negative_unsupported"
    result[finite & (np.abs(command) <= 1e-6)] = "active_brake_zero_throttle"
    active = finite & (command > 1e-6)
    result[active & (slew > 0.05)] = "accelerating"
    result[active & (slew < -0.05)] = "throttle_reduction"
    result[active & (np.abs(slew) <= 0.05)] = "steady_positive_drive"
    return result


def _counts(values: np.ndarray, axis_name: str, split: np.ndarray,
            run_id: np.ndarray, family: np.ndarray, sequence_id: np.ndarray,
            condition_id: np.ndarray) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, set[Any]]] = {}
    for index, label in enumerate(values.astype(str)):
        if label in {"outside", "unknown"}:
            continue
        key = (str(split[index]), label)
        row = grouped.setdefault(key, {
            "run_ids": set(), "families": set(), "sequences": set(),
            "conditions": set(),
        })
        row["run_ids"].add(str(run_id[index]))
        row["families"].add(str(family[index]))
        row["sequences"].add(int(sequence_id[index]))
        row["conditions"].add(int(condition_id[index]))
    sample_counts: dict[tuple[str, str], int] = defaultdict(int)
    for index, label in enumerate(values.astype(str)):
        if label not in {"outside", "unknown"}:
            sample_counts[(str(split[index]), label)] += 1
    return [{
        "axis": axis_name,
        "split": local_split,
        "bin": label,
        "rows": int(sample_counts[(local_split, label)]),
        "sequences": len(value["sequences"]),
        "independent_runs": len(value["run_ids"]),
        "conditions": len(value["conditions"]),
        "run_families": sorted(value["families"]),
        "run_ids": sorted(value["run_ids"]),
    } for (local_split, label), value in sorted(grouped.items())]


def analyze(dataset_path: Path, output_path: Path) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    z = np.load(dataset_path, allow_pickle=False)
    required = {
        "schema_version", "dataset_role", "domain_speed_cap_mps",
        "frame_domain_speed_mps", "frames", "simulator_rigid_state",
        "simulator_linear_acceleration", "sequence_bounds",
        "sequence_run_index", "sequence_condition_id", "run_ids",
        "run_splits", "training_families", "frame_run_index",
        "frame_source_sequence_index",
    }
    missing = required - set(z.files)
    if missing:
        raise ValueError(f"race view is missing coverage metadata: {sorted(missing)}")
    if int(z["schema_version"][0]) != 8:
        raise ValueError("coverage audit requires race-domain schema 8")
    if str(z["dataset_role"].astype(str)[0]) != "race_domain":
        raise ValueError("coverage audit rejects non-race-domain archives")
    if float(z["domain_speed_cap_mps"][0]) != 12.0:
        raise ValueError("coverage audit requires the <=12 m/s race domain")
    frames = z["frames"].astype(np.float64, copy=False)
    rigid = z["simulator_rigid_state"].astype(np.float64, copy=False)
    accel = z["simulator_linear_acceleration"].astype(np.float64, copy=False)
    bounds = z["sequence_bounds"].astype(np.int64, copy=False)
    sequence_runs = z["sequence_run_index"].astype(np.int32, copy=False)
    sequence_conditions = z["sequence_condition_id"].astype(np.int32, copy=False)
    run_ids = z["run_ids"].astype(str)
    splits = z["run_splits"].astype(str)
    families = z["training_families"].astype(str)
    frame_run_index = z["frame_run_index"].astype(np.int32, copy=False)
    frame_source_sequence = z["frame_source_sequence_index"].astype(np.int32,
                                                                      copy=False)
    speed = z["frame_domain_speed_mps"].astype(np.float64, copy=False)
    if (frames.shape != (len(speed), 9) or rigid.shape[0] != len(speed)
            or accel.shape[0] != len(speed)
            or frame_run_index.shape != speed.shape
            or frame_source_sequence.shape != speed.shape):
        raise ValueError("race-domain row arrays are not aligned")
    if not np.isfinite(speed).all() or np.any(speed < 0) or np.any(speed > 12.0):
        raise ValueError("race view contains non-finite or >12 m/s samples")

    dense_sequence = np.full(len(speed), -1, dtype=np.int32)
    sequence_condition_for_frame = np.full(len(speed), -1, dtype=np.int32)
    for local_sequence, (start_raw, end_raw) in enumerate(bounds):
        start, end = int(start_raw), int(end_raw)
        dense_sequence[start:end] = local_sequence
        sequence_condition_for_frame[start:end] = sequence_conditions[local_sequence]
        expected_run = int(sequence_runs[local_sequence])
        if np.any(frame_run_index[start:end] != expected_run):
            raise ValueError("frame/run metadata disagrees at a sequence boundary")
        if np.any(frame_source_sequence[start:end] < 0):
            raise ValueError("source sequence IDs must be present for every row")
    if np.any(dense_sequence < 0) or np.any(sequence_condition_for_frame < 0):
        raise ValueError("coverage includes rows not owned by a race sequence")
    local_run_ids = run_ids[frame_run_index]
    local_splits = splits[frame_run_index]
    local_families = families[frame_run_index]

    # Rear wheel surface/body speed mismatch is an observed kinematic proxy,
    # not a hidden tire-force or tire-slip measurement.
    mismatch = np.abs(0.5 * (frames[:, 5] + frames[:, 6]) - frames[:, 0])
    train_mask = local_splits == "train"
    finite_train = mismatch[train_mask & np.isfinite(mismatch)]
    if not len(finite_train):
        raise ValueError("no training-run wheel/body mismatch labels")
    mismatch_median, mismatch_p90 = np.quantile(finite_train, (0.50, 0.90))
    mismatch_band = np.full(len(mismatch), "unknown", dtype="U16")
    mismatch_band[np.isfinite(mismatch) & (mismatch <= mismatch_median)] = "low"
    mismatch_band[np.isfinite(mismatch) & (mismatch > mismatch_median)
                  & (mismatch <= mismatch_p90)] = "moderate"
    mismatch_band[np.isfinite(mismatch) & (mismatch > mismatch_p90)] = "high"

    speed_bins = _interval_labels(speed, SPEED_EDGES, "mps")
    steering_bins = _interval_labels(np.abs(frames[:, 3]), STEERING_EDGES, "rad")
    throttle_command_bins = _interval_labels(frames[:, 8], THROTTLE_EDGES, "norm")
    throttle_feedback_bins = _interval_labels(frames[:, 4], THROTTLE_EDGES, "norm")
    direction = _command_direction(frames[:, 8], accel[:, 0], bounds)
    axis_rows = []
    for axis, labels in (
        ("speed", speed_bins),
        ("absolute_actual_steering", steering_bins),
        ("throttle_command", throttle_command_bins),
        ("throttle_feedback", throttle_feedback_bins),
        ("absolute_rear_wheel_body_mismatch", mismatch_band),
        ("throttle_command_direction", direction),
    ):
        axis_rows.extend(_counts(labels, axis, local_splits, local_run_ids,
                                 local_families, dense_sequence,
                                 sequence_condition_for_frame))

    joint: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for index in range(len(speed)):
        labels = (str(local_splits[index]), str(speed_bins[index]),
                  str(steering_bins[index]), str(mismatch_band[index]),
                  str(direction[index]))
        row = joint.setdefault(labels, {
            "rows": 0, "sequences": set(), "run_ids": set(),
            "conditions": set(), "families": set(),
        })
        row["rows"] += 1
        row["sequences"].add(int(dense_sequence[index]))
        row["run_ids"].add(str(local_run_ids[index]))
        row["conditions"].add(int(sequence_condition_for_frame[index]))
        row["families"].add(str(local_families[index]))
    joint_rows = [{
        "split": key[0], "speed_bin": key[1], "steering_bin": key[2],
        "wheel_body_mismatch_bin": key[3], "command_direction": key[4],
        "rows": value["rows"], "sequences": len(value["sequences"]),
        "independent_runs": len(value["run_ids"]),
        "conditions": len(value["conditions"]),
        "run_families": sorted(value["families"]),
        "run_ids": sorted(value["run_ids"]),
    } for key, value in sorted(joint.items())]

    source_digest = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    report = {
        "schema_version": 1,
        "dataset": str(dataset_path.resolve()),
        "dataset_sha256": source_digest,
        "domain": "clean race domain; every row <=12 m/s",
        "sample_count": len(speed),
        "sequence_count": len(bounds),
        "condition_count": int(len(np.unique(sequence_conditions))),
        "independent_run_count": int(len(np.unique(local_run_ids))),
        "speed_bins_mps": list(SPEED_EDGES),
        "absolute_actual_steering_bins_rad": list(STEERING_EDGES),
        "throttle_bins_norm": list(THROTTLE_EDGES),
        "wheel_body_mismatch_definition": (
            "abs(mean(rear encoder-derived wheel surface speeds) - rear-axle u); "
            "kinematic proxy, not tire force"),
        "wheel_body_mismatch_thresholds_mps": {
            "low_upper_training_p50": float(mismatch_median),
            "moderate_upper_training_p90": float(mismatch_p90),
            "high_above_training_p90": True,
        },
        "command_direction_definition": {
            "accelerating": "positive throttle command with slope >0.05 norm/s",
            "steady_positive_drive": "positive throttle command with abs(slope)<=0.05 norm/s",
            "throttle_reduction": "positive throttle command with slope<-0.05 norm/s",
            "active_brake_zero_throttle": "zero-throttle wire command; production actuator documentation says this applies active brake torque",
            "coast": "not separately representable in the simulator wire interface; zero throttle is active braking",
            "negative_unsupported": "negative values are outside the production normalized-throttle contract",
            "limitation": "No separate brake/coast mode is recorded by the current input interface; coast response is not identifiable from these data.",
        },
        "axis_coverage": axis_rows,
        "joint_speed_steering_mismatch_command_coverage": joint_rows,
        "warning": (
            "Rows are temporally correlated. Independent source runs, sequences, "
            "and condition counts are reported separately; one run is not many "
            "independent replications."),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    table_path = output_path.with_name(output_path.stem + "_coverage.csv.gz")
    with gzip.open(table_path, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "split", "speed_bin", "steering_bin", "wheel_body_mismatch_bin",
            "command_direction", "rows", "sequences", "independent_runs",
            "conditions", "run_families", "run_ids"))
        writer.writeheader()
        for row in joint_rows:
            writer.writerow({**row,
                             "run_families": ";".join(row["run_families"]),
                             "run_ids": ";".join(row["run_ids"])})
    report["joint_csv_gz"] = str(table_path.resolve())
    # Add the table link to the JSON after writing it.
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    z.close()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = analyze(args.dataset, args.output)
    print(json.dumps({
        "samples": report["sample_count"],
        "sequences": report["sequence_count"],
        "conditions": report["condition_count"],
        "runs": report["independent_run_count"],
        "mismatch_thresholds_mps": report["wheel_body_mismatch_thresholds_mps"],
        "outputs": [str(args.output), report["joint_csv_gz"]],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
