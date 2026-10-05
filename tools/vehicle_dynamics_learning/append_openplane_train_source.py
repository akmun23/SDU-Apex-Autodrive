#!/usr/bin/env python3
"""Append one quality-gated schema-7 train/validation run without re-extraction.

This preserves the frozen source arrays exactly.  It is intentionally limited
to one whole run at a time so a changed extractor cannot silently rewrite
historical training or held-out rows. Final/test roles cannot be appended here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_KEYS = ("run_ids", "run_families", "run_splits")
SEQUENCE_KEYS = (
    "sequence_bounds", "sequence_run_index", "sequence_labels",
    "sequence_condition_id", "sequence_reset_index",
    "sequence_replicate_index",
)
FRAME_INDEX_KEYS = ("frame_run_index",)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_archive(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _single_accepted_role_run(
        data: dict[str, np.ndarray], manifest: dict[str, Any],
        expected_split: str) -> str:
    if expected_split not in ("train", "validation"):
        raise ValueError("only train and development-validation runs may be appended")
    run_ids = data["run_ids"].astype(str)
    if len(run_ids) != 1:
        raise ValueError("addition must contain exactly one run")
    run_id = str(run_ids[0])
    rows = [row for row in manifest.get("runs", [])
            if str(row.get("run_id")) == run_id]
    if len(rows) != 1:
        raise ValueError("addition manifest must describe its single run")
    row = rows[0]
    if (str(data["run_splits"][0]) != expected_split
            or row.get("effective_split") != expected_split):
        raise ValueError(f"addition must be assigned to {expected_split}")
    if (row.get("aborted") or not row.get("clean_stream_and_collision_gate")
            or int(row.get("timing_faults", -1)) != 0
            or any(int(value) != 0 for value in row.get("collisions", []))
            or row.get("whole_bag_quality_failures")
            or row.get("quality_failures")):
        raise ValueError("addition run did not pass the whole-run quality gate")
    if int(data["schema_version"][0]) != 7:
        raise ValueError("schema-7 source archive required")
    if not np.allclose(data["dt_s"], 0.025, rtol=0.0, atol=1e-7):
        raise ValueError("addition timebase must be fixed 25 ms")
    return run_id


def append(base_path: Path, base_manifest_path: Path,
           addition_path: Path, addition_manifest_path: Path,
           output_dir: Path, expected_split: str = "train") -> dict[str, Any]:
    paths = (base_path, base_manifest_path, addition_path,
             addition_manifest_path, output_dir)
    resolved = [path.resolve() for path in paths]
    for path in resolved:
        try:
            path.relative_to(REPO_ROOT)
        except ValueError as error:
            raise ValueError(f"all inputs/outputs must remain in workspace: {path}") from error
    base_path, base_manifest_path, addition_path, addition_manifest_path, output_dir = resolved
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")

    base = _read_archive(base_path)
    addition = _read_archive(addition_path)
    base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    addition_manifest = json.loads(
        addition_manifest_path.read_text(encoding="utf-8"))
    if set(base) != set(addition):
        raise ValueError("source archive fields do not match")
    if int(base["schema_version"][0]) != 7:
        raise ValueError("frozen baseline must be schema 7")
    run_id = _single_accepted_role_run(
        addition, addition_manifest, expected_split)
    if set(base["run_ids"].astype(str)) & {run_id}:
        raise ValueError(f"run already exists in frozen baseline: {run_id}")

    base_frames = len(base["frames"])
    base_sequences = len(base["sequence_bounds"])
    base_runs = len(base["run_ids"])
    base_conditions = len(base["condition_labels"])
    added_frames = len(addition["frames"])
    added_sequences = len(addition["sequence_bounds"])
    added_runs = len(addition["run_ids"])
    added_conditions = len(addition["condition_labels"])

    frame_keys = {
        key for key, values in base.items()
        if values.ndim > 0 and values.shape[0] == base_frames
    }
    sequence_keys = set(SEQUENCE_KEYS)
    run_keys = set(RUN_KEYS)
    condition_keys = {"condition_labels"}
    frame_keys -= sequence_keys | run_keys | condition_keys
    if "frame_run_index" not in frame_keys:
        raise ValueError("source archive lacks frame_run_index")
    known = frame_keys | sequence_keys | run_keys | condition_keys
    static_keys = set(base) - known
    if not sequence_keys <= set(base) or not run_keys <= set(base):
        raise ValueError("source archive lacks expected sequence/run metadata")

    for key in static_keys:
        if (base[key].shape != addition[key].shape
                or base[key].dtype != addition[key].dtype
                or not np.array_equal(base[key], addition[key])):
            raise ValueError(f"static schema field differs: {key}")
    for key in frame_keys:
        if (base[key].ndim != addition[key].ndim
                or base[key].shape[1:] != addition[key].shape[1:]
                or base[key].dtype != addition[key].dtype):
            raise ValueError(f"frame schema differs: {key}")

    added_run_index = addition["frame_run_index"].astype(np.int64)
    added_sequence_run_index = addition["sequence_run_index"].astype(np.int64)
    added_condition_ids = addition["sequence_condition_id"].astype(np.int64)
    if (np.any(added_run_index < 0) or np.any(added_run_index >= added_runs)
            or np.any(added_sequence_run_index < 0)
            or np.any(added_sequence_run_index >= added_runs)):
        raise ValueError("addition has out-of-range run indices")
    if (np.any(added_condition_ids < 0)
            or np.any(added_condition_ids >= added_conditions)):
        raise ValueError("addition has out-of-range condition IDs")
    bounds = addition["sequence_bounds"]
    if (bounds.ndim != 2 or bounds.shape[1] != 2 or np.any(bounds < 0)
            or np.any(bounds[:, 1] > added_frames)
            or np.any(bounds[:, 1] <= bounds[:, 0])):
        raise ValueError("addition sequence bounds are invalid")

    merged = {key: value.copy() for key, value in base.items()}
    for key in frame_keys:
        values = addition[key].copy()
        if key == "frame_run_index":
            values = values + base_runs
        merged[key] = np.concatenate((base[key], values), axis=0)
    for key in sequence_keys:
        values = addition[key].copy()
        if key == "sequence_bounds":
            values += base_frames
        elif key == "sequence_run_index":
            values += base_runs
        elif key == "sequence_condition_id":
            values += base_conditions
        merged[key] = np.concatenate((base[key], values), axis=0)
    for key in RUN_KEYS:
        merged[key] = np.concatenate((base[key], addition[key]), axis=0)
    merged["condition_labels"] = np.concatenate(
        (base["condition_labels"], addition["condition_labels"]), axis=0)

    # Prove old frame, sequence, run, and condition rows are unchanged before
    # serializing the appended archive.
    for key in frame_keys:
        if not np.array_equal(merged[key][:base_frames], base[key]):
            raise AssertionError(f"frozen frame prefix changed: {key}")
    for key in sequence_keys:
        if not np.array_equal(merged[key][:base_sequences], base[key]):
            raise AssertionError(f"frozen sequence prefix changed: {key}")
    for key in RUN_KEYS:
        if not np.array_equal(merged[key][:base_runs], base[key]):
            raise AssertionError(f"frozen run prefix changed: {key}")
    if not np.array_equal(
            merged["condition_labels"][:base_conditions], base["condition_labels"]):
        raise AssertionError("frozen condition prefix changed")

    output_dir.mkdir(parents=True)
    dataset_path = output_dir / "openplane_dynamics.npz"
    np.savez_compressed(dataset_path, **merged)
    run_rows = base_manifest.get("runs", []) + addition_manifest.get("runs", [])
    if {str(row.get("run_id")) for row in run_rows} != set(
            merged["run_ids"].astype(str)):
        raise ValueError("merged manifest and archive run IDs disagree")
    manifest = dict(base_manifest)
    manifest.pop("training_coverage", None)
    manifest["plant_continuity_mode"] = "mixed_by_source"
    manifest["source_plant_continuity_modes"] = {
        "frozen_base": base_manifest.get("plant_continuity_mode"),
        "added_run": addition_manifest.get("plant_continuity_mode"),
        "note": "All exported sequence boundaries are preserved in the merged archive.",
    }
    manifest["selection"] = (
        f"Frozen baseline schema-7 rows plus one quality-gated {expected_split} run; "
        "historical bags were not re-extracted.")
    manifest["runs"] = run_rows
    manifest["errors"] = []
    manifest["export"] = {
        "file": dataset_path.name,
        "samples": len(merged["frames"]),
        "sequences": len(merged["sequence_bounds"]),
        "runs_in_archive": len(merged["run_ids"]),
        "runs_with_exported_sequences": merged["run_ids"].astype(str).tolist(),
        "compressed_bytes": dataset_path.stat().st_size,
    }
    manifest["append_provenance"] = {
        "operation": "append_one_quality_gated_role_run_without_reextracting_base",
        "frozen_base_dataset": str(base_path.relative_to(REPO_ROOT)),
        "frozen_base_sha256": _sha256(base_path),
        "frozen_base_manifest": str(base_manifest_path.relative_to(REPO_ROOT)),
        "frozen_base_manifest_sha256": _sha256(base_manifest_path),
        "added_dataset": str(addition_path.relative_to(REPO_ROOT)),
        "added_dataset_sha256": _sha256(addition_path),
        "added_manifest": str(addition_manifest_path.relative_to(REPO_ROOT)),
        "added_manifest_sha256": _sha256(addition_manifest_path),
        "added_run_id": run_id,
        "added_split": expected_split,
        "historical_rows_and_sequences_preserved_exactly": True,
    }
    manifest["export"]["sha256"] = _sha256(dataset_path)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return {
        "dataset": str(dataset_path),
        "sha256": manifest["export"]["sha256"],
        "rows": len(merged["frames"]),
        "sequences": len(merged["sequence_bounds"]),
        "runs": len(merged["run_ids"]),
        "added_run": run_id,
        "added_split": expected_split,
        "base_rows_preserved": base_frames,
        "base_sequences_preserved": base_sequences,
        "base_runs_preserved": base_runs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("base_manifest", type=Path)
    parser.add_argument("addition", type=Path)
    parser.add_argument("addition_manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation"),
                        default="train",
                        help="whole-run role assigned before scoring")
    args = parser.parse_args()
    print(json.dumps(append(args.base, args.base_manifest, args.addition,
                            args.addition_manifest, args.output_dir,
                            args.split), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
