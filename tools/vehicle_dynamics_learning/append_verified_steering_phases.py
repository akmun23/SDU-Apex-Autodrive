#!/usr/bin/env python3
"""Append phase-verified, train-only steering segments to a race-domain view.

The aborted parent captures remain rejected as whole runs. Only selected
individually audited ``verified`` phases are appended, each with run/condition
and reset boundaries intact. Existing frame and split prefixes are immutable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np


FRAME_SPECIAL = {
    "frame_run_index", "frame_reset_index", "frame_source_index",
    "frame_source_sequence_index", "frame_domain_speed_mps",
}
SEQUENCE_KEYS = {
    "sequence_bounds", "sequence_run_index", "sequence_labels",
    "sequence_condition_id", "sequence_reset_index",
    "sequence_replicate_index", "sequence_source_condition_id",
    "sequence_source_index",
}
RUN_KEYS = {"run_ids", "run_families", "run_splits", "training_families"}
CONDITION_KEYS = {"condition_labels", "condition_run_index"}
STATIC_KEYS = {
    "schema_version", "source_schema_version", "dataset_role",
    "domain_speed_cap_mps", "domain_cooldown_steps", "feature_names",
    "sensor_feature_names", "attitude_feature_names",
    "predicted_state_names", "training_family_names",
    "training_family_probabilities",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _replicate_index(label: str) -> int:
    match = re.search(r"(?:^|_)r(\d+)(?:_|$)", label)
    return int(match.group(1)) if match else -1


def _selected_sequences(salvage: dict[str, np.ndarray],
                        audit: dict[str, Any],
                        selected_runs: set[str]) -> list[tuple[int, int, int, str]]:
    run_ids = salvage["run_ids"].astype(str)
    run_splits = salvage["run_splits"].astype(str)
    if len(set(selected_runs)) != len(selected_runs):
        raise ValueError("selected run IDs must be unique")
    if not selected_runs or not selected_runs <= set(run_ids):
        raise ValueError("all selected runs must exist in the salvage archive")

    reports = {str(row["run_id"]): row for row in audit.get("runs", [])}
    if not selected_runs <= set(reports):
        raise ValueError("selected runs are missing from the salvage audit")
    valid_labels: dict[str, set[str]] = {}
    for run_id in selected_runs:
        report = reports[run_id]
        if report.get("source_split") != "train":
            raise ValueError(f"salvaged run is not train-only: {run_id}")
        if str(run_splits[int(np.flatnonzero(run_ids == run_id)[0])]) != "train":
            raise ValueError(f"archive split is not train-only: {run_id}")
        labels = set()
        for phase in report.get("phases", []):
            if (phase.get("status") != "salvaged"
                    or phase.get("quality_tier") != "verified"):
                continue
            if phase.get("reasons"):
                raise ValueError(
                    f"selected run contains a non-verified phase: {run_id}"
                )
            labels.add(str(phase["label"]))
        valid_labels[run_id] = labels

    seq_runs = salvage["sequence_run_index"].astype(np.int64)
    labels = salvage["sequence_labels"].astype(str)
    splits = salvage["sequence_splits"].astype(str)
    bounds = salvage["sequence_bounds"].astype(np.int64)
    chosen = []
    for sequence_id, (start, end) in enumerate(bounds):
        run_id = str(run_ids[int(seq_runs[sequence_id])])
        if run_id not in selected_runs:
            continue
        label = str(labels[sequence_id])
        if label not in valid_labels[run_id]:
            raise ValueError(f"sequence is not an audited verified phase: {label}")
        if splits[sequence_id] != "train":
            raise ValueError(f"sequence split is not train-only: {label}")
        if end <= start or end > len(salvage["frames"]):
            raise ValueError(f"invalid salvage sequence bounds: {label}")
        chosen.append((sequence_id, int(start), int(end), run_id))

    for run_id in selected_runs:
        expected = valid_labels[run_id]
        found = {label for sequence_id, _, _, candidate_run in chosen
                 if candidate_run == run_id
                 for label in (str(labels[sequence_id]),)}
        if expected != found:
            raise ValueError(
                f"audited phase/sequence labels disagree for {run_id}: "
                f"missing={sorted(expected - found)}, extra={sorted(found - expected)}"
            )
    if not chosen:
        raise ValueError("no verified sequences selected")
    return chosen


def append(base_path: Path, salvage_path: Path, audit_path: Path,
           output_dir: Path, selected_run_ids: list[str]) -> dict[str, Any]:
    for path in (base_path, salvage_path, audit_path, output_dir):
        if output_dir.exists() and path == output_dir:
            raise FileExistsError(f"refusing to overwrite {output_dir}")
        if not path.is_file() and path != output_dir:
            raise FileNotFoundError(path)
    base = _read_npz(base_path)
    salvage = _read_npz(salvage_path)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if int(base["schema_version"][0]) != 8:
        raise ValueError("base must be a schema-8 race-domain view")
    if int(salvage["schema_version"][0]) != 7:
        raise ValueError("salvage archive must be the audited schema-7 tier")
    expected_file = audit.get("export_by_quality_tier", {}).get(
        "verified", {}).get("file")
    if expected_file != salvage_path.name:
        raise ValueError("input archive is not the audit's verified-tier export")
    if not np.allclose(base["dt_s"], 0.025, rtol=0.0, atol=1e-7):
        raise ValueError("base timebase must be exactly 25 ms")
    if not np.allclose(salvage["dt_s"], 0.025, rtol=0.0, atol=1e-7):
        raise ValueError("salvage timebase must be exactly 25 ms")
    if "packet_sequence" not in salvage:
        raise ValueError("salvage archive lacks packet sequence IDs")
    for key in ("feature_names", "sensor_feature_names",
                "attitude_feature_names", "predicted_state_names"):
        if not np.array_equal(base[key], salvage[key]):
            raise ValueError(f"feature schema mismatch: {key}")

    selected_runs = set(map(str, selected_run_ids))
    chosen = _selected_sequences(salvage, audit, selected_runs)
    base_run_ids = base["run_ids"].astype(str)
    if selected_runs & set(base_run_ids):
        raise ValueError("a selected salvage run already exists in the base view")
    ordered_runs = sorted(selected_runs)
    added_run_index = {run_id: len(base_run_ids) + index
                       for index, run_id in enumerate(ordered_runs)}
    old_frames = len(base["frames"])
    old_sequences = len(base["sequence_bounds"])
    old_runs = len(base_run_ids)
    old_conditions = len(base["condition_labels"])

    frame_keys = {
        key for key, values in base.items()
        if values.ndim > 0 and values.shape[0] == old_frames
        and key not in (SEQUENCE_KEYS | RUN_KEYS | CONDITION_KEYS | STATIC_KEYS)
    }
    for key in frame_keys - FRAME_SPECIAL:
        if key not in salvage:
            raise ValueError(f"salvage archive lacks frame field: {key}")
        if (base[key].ndim != salvage[key].ndim
                or base[key].shape[1:] != salvage[key].shape[1:]
                or base[key].dtype != salvage[key].dtype):
            raise ValueError(f"frame schema mismatch: {key}")

    additions: dict[str, list[np.ndarray]] = {key: [] for key in frame_keys}
    new_bounds: list[tuple[int, int]] = []
    new_seq_runs: list[int] = []
    new_labels: list[str] = []
    new_condition_ids: list[int] = []
    new_reset_ids: list[int] = []
    new_replicates: list[int] = []
    new_source_conditions: list[int] = []
    new_source_sequences: list[int] = []
    new_condition_labels: list[str] = []
    new_condition_runs: list[int] = []
    condition_cursor = old_conditions
    frame_cursor = old_frames
    sequence_reset_by_run = {run_id: 0 for run_id in ordered_runs}

    for source_sequence, start, end, run_id in chosen:
        packet_steps = np.diff(salvage["packet_sequence"][start:end])
        if not np.all(packet_steps == 1):
            raise ValueError(f"packet gap inside verified phase: {run_id}")
        rigid = salvage["simulator_rigid_state"][start:end]
        speed = np.hypot(rigid[:, 7], rigid[:, 8]).astype(np.float32)
        if (not np.isfinite(speed).all() or np.max(speed) > 12.0
                or not np.isfinite(salvage["frames"][start:end]).all()):
            raise ValueError(f"selected phase is outside the finite <=12 m/s domain: {run_id}")
        label = str(salvage["sequence_labels"][source_sequence])
        local_reset = sequence_reset_by_run[run_id]
        sequence_reset_by_run[run_id] += 1
        left, right = frame_cursor, frame_cursor + end - start
        new_bounds.append((left, right))
        new_seq_runs.append(added_run_index[run_id])
        new_labels.append(label)
        new_condition_ids.append(condition_cursor)
        new_reset_ids.append(local_reset)
        new_replicates.append(_replicate_index(label))
        new_source_conditions.append(condition_cursor)
        new_source_sequences.append(source_sequence)
        new_condition_labels.append(label)
        new_condition_runs.append(added_run_index[run_id])
        condition_cursor += 1

        for key in frame_keys:
            if key == "frame_run_index":
                part = np.full(end - start, added_run_index[run_id], dtype=base[key].dtype)
            elif key == "frame_reset_index":
                part = np.full(end - start, local_reset, dtype=base[key].dtype)
            elif key == "frame_source_index":
                part = np.arange(start, end, dtype=base[key].dtype)
            elif key == "frame_source_sequence_index":
                part = np.full(end - start, source_sequence, dtype=base[key].dtype)
            elif key == "frame_domain_speed_mps":
                part = speed.astype(base[key].dtype, copy=False)
            else:
                part = salvage[key][start:end]
            additions[key].append(part)
        frame_cursor = right

    if len(chosen) != len(new_bounds):
        raise AssertionError("sequence metadata construction lost a selected phase")
    merged = {key: value.copy() for key, value in base.items()}
    for key in frame_keys:
        merged[key] = np.concatenate((base[key], *additions[key]), axis=0)
    merged["sequence_bounds"] = np.concatenate((
        base["sequence_bounds"], np.asarray(new_bounds, dtype=np.int64)), axis=0)
    merged["sequence_run_index"] = np.concatenate((
        base["sequence_run_index"], np.asarray(new_seq_runs, dtype=np.int32)))
    merged["sequence_labels"] = np.concatenate((
        base["sequence_labels"], np.asarray(new_labels, dtype="U256")))
    merged["sequence_condition_id"] = np.concatenate((
        base["sequence_condition_id"], np.asarray(new_condition_ids, dtype=np.int32)))
    merged["sequence_reset_index"] = np.concatenate((
        base["sequence_reset_index"], np.asarray(new_reset_ids, dtype=np.int32)))
    merged["sequence_replicate_index"] = np.concatenate((
        base["sequence_replicate_index"], np.asarray(new_replicates, dtype=np.int32)))
    merged["sequence_source_condition_id"] = np.concatenate((
        base["sequence_source_condition_id"],
        np.asarray(new_source_conditions, dtype=np.int32)))
    merged["sequence_source_index"] = np.concatenate((
        base["sequence_source_index"], np.asarray(new_source_sequences, dtype=np.int32)))

    merged["run_ids"] = np.concatenate((
        base["run_ids"], np.asarray(ordered_runs, dtype=base["run_ids"].dtype)))
    merged["run_families"] = np.concatenate((
        base["run_families"], np.full(len(ordered_runs), "highspeed_steering",
                                       dtype=base["run_families"].dtype)))
    merged["run_splits"] = np.concatenate((
        base["run_splits"], np.full(len(ordered_runs), "train",
                                     dtype=base["run_splits"].dtype)))
    merged["training_families"] = np.concatenate((
        base["training_families"], np.full(
            len(ordered_runs), "steering_transition_slew",
            dtype=base["training_families"].dtype)))
    merged["condition_labels"] = np.concatenate((
        base["condition_labels"], np.asarray(new_condition_labels,
                                             dtype=base["condition_labels"].dtype)))
    merged["condition_run_index"] = np.concatenate((
        base["condition_run_index"], np.asarray(new_condition_runs, dtype=np.int32)))

    for key in base:
        if key in frame_keys:
            if not np.array_equal(merged[key][:old_frames], base[key]):
                raise AssertionError(f"base frame prefix changed: {key}")
        elif key in SEQUENCE_KEYS:
            if not np.array_equal(merged[key][:old_sequences], base[key]):
                raise AssertionError(f"base sequence prefix changed: {key}")
        elif key in RUN_KEYS:
            if not np.array_equal(merged[key][:old_runs], base[key]):
                raise AssertionError(f"base run prefix changed: {key}")
        elif key in CONDITION_KEYS:
            if not np.array_equal(merged[key][:old_conditions], base[key]):
                raise AssertionError(f"base condition prefix changed: {key}")
        elif not np.array_equal(merged[key], base[key]):
            raise AssertionError(f"static base field changed: {key}")

    output_dir.mkdir(parents=True)
    dataset_path = output_dir / "openplane_dynamics.npz"
    np.savez_compressed(dataset_path, **merged)
    provenance = {
        "schema_version": 1,
        "dataset_role": "race_domain_training_and_evaluation",
        "speed_cap_mps": 12.0,
        "purpose": "train-only addition of phase-verified high-speed steering data",
        "base_dataset": str(base_path),
        "base_sha256": _sha256(base_path),
        "salvage_archive": str(salvage_path),
        "salvage_sha256": _sha256(salvage_path),
        "salvage_audit": str(audit_path),
        "salvage_audit_sha256": _sha256(audit_path),
        "selected_run_ids": ordered_runs,
        "selected_sequence_count": len(chosen),
        "selected_phase_labels": new_labels,
        "phase_evidence": "only verified-tier sequences whose per-phase salvage audit passed; aborted parent runs remain whole-run rejected",
        "whole_run_status": "the parent captures remain rejected as whole runs; only audited phase sequences were appended",
        "split": "all appended runs and sequences are train-only",
        "heldout_data_touched": False,
        "base_prefix_preserved_exactly": True,
        "maximum_selected_speed_mps": max(
            float(np.max(np.hypot(
                salvage["simulator_rigid_state"][start:end, 7],
                salvage["simulator_rigid_state"][start:end, 8])))
            for _, start, end, _ in chosen),
        "packet_continuity": "all packet sequence differences inside selected phases equal one",
        "timebase": "fixed 25 ms per sample; source capture dt_s was audited as exact 25 ms",
        "output_sha256": _sha256(dataset_path),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return {
        "dataset": str(dataset_path),
        "sha256": provenance["output_sha256"],
        "added_runs": len(ordered_runs),
        "added_sequences": len(chosen),
        "added_samples": frame_cursor - old_frames,
        "maximum_speed_mps": provenance["maximum_selected_speed_mps"],
        "base_rows_preserved": old_frames,
        "base_sequences_preserved": old_sequences,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("salvage_archive", type=Path)
    parser.add_argument("salvage_audit", type=Path)
    parser.add_argument("--run-id", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = append(args.base, args.salvage_archive, args.salvage_audit,
                    args.output_dir, args.run_id)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
