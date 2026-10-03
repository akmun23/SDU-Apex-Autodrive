#!/usr/bin/env python3
"""Carry frozen practice windows onto an equivalent encoder-rate sidecar."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
UNCHANGED_ARRAYS = (
    "frames", "run_ids", "run_splits", "sequence_bounds",
    "sequence_run_index", "simulator_pose_xyyaw", "lap_count", "sensor_valid",
    "sample_time_ns", "packet_sequence", "sequence_reset_index",
    "encoder_raw_valid",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def derive(base_benchmark_path: Path, previous_dataset_path: Path,
           new_dataset_path: Path, output_path: Path) -> dict[str, Any]:
    paths = tuple(path.resolve() for path in (
        base_benchmark_path, previous_dataset_path, new_dataset_path))
    base_benchmark_path, previous_dataset_path, new_dataset_path = paths
    output_path = output_path.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite benchmark: {output_path}")

    benchmark = json.loads(base_benchmark_path.read_text(encoding="utf-8"))
    if (benchmark.get("benchmark_id") != "practice_transfer_benchmark_v1"
            or benchmark.get("frozen") is not True
            or benchmark.get("training_or_checkpoint_selection_use") is not False
            or benchmark.get("dataset_sha256") != _sha256(previous_dataset_path)):
        raise ValueError("base benchmark is not frozen against the previous dataset")

    previous_manifest_path = previous_dataset_path.with_name(
        previous_dataset_path.stem + "_manifest.json")
    new_manifest_path = new_dataset_path.with_name(
        new_dataset_path.stem + "_manifest.json")
    previous_manifest = json.loads(previous_manifest_path.read_text(
        encoding="utf-8"))
    new_manifest = json.loads(new_manifest_path.read_text(encoding="utf-8"))
    if (previous_manifest.get("source_dataset_sha256")
            != new_manifest.get("source_dataset_sha256")
            or previous_manifest.get("source_manifest_sha256")
            != new_manifest.get("source_manifest_sha256")):
        raise ValueError("encoder sidecars do not share identical source provenance")
    if (previous_manifest.get("split_policy") != new_manifest.get("split_policy")
            or new_manifest.get("split_policy", {}).get(
                "excluded_test_and_final_test_bags") is not True):
        raise ValueError("encoder sidecar split policies differ or include held-out test data")

    with np.load(previous_dataset_path, allow_pickle=False) as previous, \
            np.load(new_dataset_path, allow_pickle=False) as current:
        for name in UNCHANGED_ARRAYS:
            if name not in previous or name not in current:
                raise ValueError(f"encoder sidecar is missing required array {name}")
            left, right = previous[name], current[name]
            equal = (np.array_equal(left, right, equal_nan=True)
                     if np.issubdtype(left.dtype, np.number)
                     else np.array_equal(left, right))
            if not equal:
                raise ValueError(f"frozen practice inputs/windows changed in {name}")
        if not np.array_equal(previous["encoder_raw_valid"],
                              current["encoder_raw_valid"]):
            raise ValueError("wheel-target validity changed; frozen support is no longer exact")

    try:
        workspace_path = "/workspace/" + str(new_dataset_path.relative_to(ROOT))
    except ValueError as exc:
        raise ValueError("new practice dataset must remain inside this workspace") from exc
    result = dict(benchmark)
    result.update({
        "dataset": workspace_path,
        "dataset_sha256": _sha256(new_dataset_path),
        "derived_dataset_view": (
            "same causal packet-aligned encoder intervals as the previous view, "
            "with the known fixed 25 ms (40 Hz) physical period used instead "
            "of source-stamp dt; frozen windows and validity mask verified identical"),
        "derived_from_benchmark": benchmark["benchmark_id"],
        "derived_from_benchmark_sha256": _sha256(base_benchmark_path),
        "previous_encoder_dataset_sha256": _sha256(previous_dataset_path),
        "encoder_sidecar_manifest": str(new_manifest_path.relative_to(ROOT)),
        "encoder_sidecar_manifest_sha256": _sha256(new_manifest_path),
        "wheel_target_policy": (
            "score rear-wheel error only where causal 40 Hz source encoder "
            "intervals are valid; validity mask and all frozen starts are "
            "identical to the prior derived benchmark"),
    })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-benchmark", type=Path, required=True)
    parser.add_argument("--previous-dataset", type=Path, required=True)
    parser.add_argument("--new-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = derive(args.base_benchmark, args.previous_dataset,
                    args.new_dataset, args.output)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "dataset_sha256": result["dataset_sha256"],
        "window_count": result["window_count_after_category_deduplication"],
        "independent_runs": result["independent_run_count"],
        "validity_mask_and_frozen_windows_unchanged": True,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
