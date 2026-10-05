#!/usr/bin/env python3
"""Append quality-gated whole-run body datasets without rewriting their sources."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
FRAME_KEYS = (
    "inputs", "outputs", "stored_commands", "run_id", "sequence_id",
    "reset_epoch", "sample_index", "condition_id", "split",
    "sample_time_s", "source_frame_index", "dt_s",
)
SEQUENCE_KEYS = (
    "sequence_bounds", "sequence_run_id", "sequence_id_table",
    "sequence_split", "sequence_label", "sequence_reset_epoch",
    "sequence_condition_id",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def append(base_path: Path, base_manifest_path: Path,
           additions: list[tuple[Path, Path, str]],
           output_path: Path) -> dict[str, Any]:
    if not additions:
        raise ValueError("at least one whole-run body dataset is required")
    paths = [base_path, base_manifest_path, output_path]
    paths.extend(path for archive, manifest, _ in additions
                 for path in (archive, manifest))
    resolved = [path.resolve() for path in paths]
    for path in resolved:
        try:
            path.relative_to(ROOT)
        except ValueError as error:
            raise ValueError(f"all inputs and outputs must remain in workspace: {path}") from error
    base_path, base_manifest_path, output_path = resolved[:3]
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")

    base = _load(base_path)
    base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    if int(base["schema_version"][0]) != 1:
        raise ValueError("body schema version 1 is required")
    expected_keys = set(base)
    if not set(FRAME_KEYS + SEQUENCE_KEYS) <= expected_keys:
        raise ValueError("base body dataset is missing required arrays")
    base_rows = len(base["inputs"])
    base_sequences = len(base["sequence_bounds"])
    frame_keys = set(FRAME_KEYS)
    sequence_keys = set(SEQUENCE_KEYS)
    static_keys = expected_keys - frame_keys - sequence_keys
    merged = {key: value.copy() for key, value in base.items()}
    known_runs = set(base["sequence_run_id"].astype(str))
    addition_rows: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []

    for archive_path, manifest_path, expected_split in additions:
        if expected_split not in ("train", "validation"):
            raise ValueError("only train or development-validation runs may be appended")
        archive_path, manifest_path = archive_path.resolve(), manifest_path.resolve()
        addition = _load(archive_path)
        addition_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if set(addition) != expected_keys:
            raise ValueError(f"body schema differs in {archive_path}")
        if int(addition["schema_version"][0]) != 1:
            raise ValueError("addition body schema version 1 is required")
        runs = set(addition["sequence_run_id"].astype(str))
        if len(runs) != 1:
            raise ValueError("each addition must contain exactly one complete run")
        run_id = next(iter(runs))
        if run_id in known_runs:
            raise ValueError(f"duplicate run in merged body dataset: {run_id}")
        if set(addition["sequence_split"].astype(str)) != {expected_split}:
            raise ValueError(f"{run_id} is not assigned to {expected_split}")
        if set(addition["split"].astype(str)) != {expected_split}:
            raise ValueError(f"{run_id} has mixed or unexpected row splits")
        rows = len(addition["inputs"])
        if (addition["inputs"].shape != (rows, 2)
                or addition["outputs"].shape != (rows, 3)
                or not np.isfinite(addition["inputs"]).all()
                or not np.isfinite(addition["outputs"]).all()
                or not np.allclose(addition["dt_s"], 0.025, rtol=0.0, atol=1e-7)):
            raise ValueError(f"invalid finite 40 Hz body rows for {run_id}")
        bounds = addition["sequence_bounds"]
        if (bounds.ndim != 2 or bounds.shape[1] != 2
                or np.any(bounds < 0) or np.any(bounds[:, 1] > rows)
                or np.any(bounds[:, 1] <= bounds[:, 0])
                or int(bounds[0, 0]) != 0
                or int(bounds[-1, 1]) != rows
                or np.any(bounds[1:, 0] != bounds[:-1, 1])):
            raise ValueError(f"invalid reset-isolated sequence bounds for {run_id}")
        for key in static_keys:
            if (addition[key].shape != base[key].shape
                    or addition[key].dtype != base[key].dtype
                    or not np.array_equal(addition[key], base[key])):
                raise ValueError(f"static body schema differs for {run_id}: {key}")
        for key in frame_keys:
            if (addition[key].ndim != base[key].ndim
                    or addition[key].shape[1:] != base[key].shape[1:]
                    or addition[key].dtype != base[key].dtype):
                raise ValueError(f"frame body schema differs for {run_id}: {key}")
        for key in sequence_keys:
            if (addition[key].ndim != base[key].ndim
                    or addition[key].shape[1:] != base[key].shape[1:]
                    or addition[key].dtype != base[key].dtype):
                raise ValueError(f"sequence body schema differs for {run_id}: {key}")

        current_rows = len(merged["inputs"])
        for key in frame_keys:
            merged[key] = np.concatenate((merged[key], addition[key]), axis=0)
        for key in sequence_keys:
            values = addition[key].copy()
            if key == "sequence_bounds":
                values += current_rows
            merged[key] = np.concatenate((merged[key], values), axis=0)
        matching = [row for row in addition_manifest.get("runs", [])
                    if isinstance(row, dict)
                    and str(row.get("run_id")) == run_id]
        if matching:
            if len(matching) != 1:
                raise ValueError(f"manifest does not uniquely describe {run_id}")
            row = matching[0]
            if (row.get("aborted") or row.get("whole_bag_quality_failures")
                    or row.get("quality_failures")
                    or not row.get("clean_stream_and_collision_gate")
                    or row.get("reason") != "schedule complete"
                    or int(row.get("timing_faults", -1)) != 0
                    or any(int(value) != 0 for value in row.get("collisions", []))):
                raise ValueError(f"whole-run bag quality gate failed for {run_id}")
            alignment = row.get("packet_sequence_alignment", {})
            if float(alignment.get("match_fraction", 0.0)) < 0.999:
                raise ValueError(f"packet/odometry alignment failed for {run_id}")
            for topic, stats in row.get("streams", {}).items():
                if (float(stats.get("hz", 0.0)) < 38.0
                        or float(stats.get("gap_p95_ms", float("inf"))) > 35.0
                        or float(stats.get("gap_max_ms", float("inf"))) > 120.0):
                    raise ValueError(f"40 Hz stream gate failed for {run_id}: {topic}")
        else:
            quality = addition_manifest.get("capture_quality", {})
            if (quality.get("status") != "passed"
                    or str(quality.get("run_id")) != run_id
                    or not quality.get("packet_sequence_contiguous")
                    or float(quality.get("packet_alignment_fraction", 0.0)) < 0.999
                    or quality.get("collision_count_min_max") != [0, 0]
                    or not quality.get("all_stream_cadence_gates_passed")
                    or int(quality.get("admitted_samples", -1)) != rows):
                raise ValueError(f"captured-run provenance gate failed for {run_id}")
            role_runs = addition_manifest.get("runs", {})
            if run_id not in role_runs.get(expected_split, []):
                raise ValueError(f"capture quality split differs for {run_id}")
            row = {"run_id": run_id,
                   "capture_quality": quality,
                   "reason": "passed linked practice capture and branch-view gates",
                   "aborted": False,
                   "timing_faults": 0,
                   "collisions": [0, 0],
                   "quality_failures": []}
        addition_rows.append(row)
        provenance.append({
            "run_id": run_id,
            "split": expected_split,
            "dataset": str(archive_path.relative_to(ROOT)),
            "dataset_sha256": _sha256(archive_path),
            "manifest": str(manifest_path.relative_to(ROOT)),
            "manifest_sha256": _sha256(manifest_path),
        })
        known_runs.add(run_id)

    for key in frame_keys:
        if not np.array_equal(merged[key][:base_rows], base[key]):
            raise AssertionError(f"frozen base frame prefix changed: {key}")
    for key in sequence_keys:
        if not np.array_equal(merged[key][:base_sequences], base[key]):
            raise AssertionError(f"frozen base sequence prefix changed: {key}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **merged)
    manifest = dict(base_manifest)
    manifest["dataset"] = str(output_path)
    manifest["dataset_sha256"] = _sha256(output_path)
    manifest["source_dataset"] = "whole-run body sysid datasets"
    manifest["source_dataset_sha256"] = None
    manifest["run_split_policy"] = (
        "frozen base plus quality-gated complete runs; train and validation roles "
        "remain disjoint")
    manifest["runs"] = {
        "train": sorted(set(merged["sequence_run_id"][
            merged["sequence_split"] == "train"].astype(str))),
        "validation": sorted(set(merged["sequence_run_id"][
            merged["sequence_split"] == "validation"].astype(str))),
    }
    manifest["run_counts"] = {key: len(value)
                              for key, value in manifest["runs"].items()}
    manifest["sequence_count"] = len(merged["sequence_bounds"])
    manifest["row_count"] = len(merged["inputs"])
    manifest["append_provenance"] = {
        "operation": "append_quality_gated_whole_run_body_datasets",
        "frozen_base_dataset": str(base_path.relative_to(ROOT)),
        "frozen_base_sha256": _sha256(base_path),
        "additions": provenance,
        "historical_rows_and_sequences_preserved_exactly": True,
    }
    manifest_path = output_path.with_name(f"{output_path.stem}_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
    return {
        "dataset": str(output_path),
        "dataset_sha256": manifest["dataset_sha256"],
        "base_rows_preserved": base_rows,
        "rows": len(merged["inputs"]),
        "base_sequences_preserved": base_sequences,
        "sequences": len(merged["sequence_bounds"]),
        "added_runs": [item["run_id"] for item in provenance],
        "runs": len(manifest["runs"]["train"])
        + len(manifest["runs"]["validation"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("base_manifest", type=Path)
    parser.add_argument("--addition", action="append", nargs=3, required=True,
                        metavar=("ARCHIVE", "MANIFEST", "SPLIT"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    additions = [(Path(archive), Path(manifest), split)
                 for archive, manifest, split in args.addition]
    print(json.dumps(append(args.base, args.base_manifest, additions,
                            args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
