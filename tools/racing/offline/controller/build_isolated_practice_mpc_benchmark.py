#!/usr/bin/env python3
"""Rebind frozen practice start metadata to a whitelist-only bag dataset.

The source benchmark contributes start indices and categories only. Numeric
state arrays are loaded exclusively from the supplied dataset, which must
contain exactly the two admitted practice-validation captures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_RUN_IDS = {
    "practice_model_validation_r02",
    "practice_model_validation_r03",
}
EXPECTED_SOURCE_IDS = {
    "practice_unseen_model_validation_20261001_r02",
    "practice_unseen_model_validation_20261001_r03",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build(dataset_path: Path, source_path: Path,
          report_paths: list[Path], output_path: Path) -> dict[str, Any]:
    dataset_path = dataset_path.resolve()
    source_path = source_path.resolve()
    output_path = output_path.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite benchmark: {output_path}")

    reports: dict[str, dict[str, Any]] = {}
    for path in report_paths:
        path = path.resolve()
        report = json.loads(path.read_text(encoding="utf-8"))
        if (report.get("admitted") is not True
                or report.get("role") != "validation"
                or report.get("training_allowed") is not False
                or report.get("expected_laps") != 6):
            raise ValueError(f"not an admitted six-lap validation report: {path}")
        run_id = str(report.get("run_id", ""))
        if run_id in reports:
            raise ValueError(f"duplicate validation run: {run_id}")
        if run_id not in EXPECTED_RUN_IDS:
            raise ValueError(f"unexpected validation run: {run_id}")
        reports[run_id] = {
            "report": str(path),
            "report_sha256": _sha256(path),
            "bag_sha256": report["bag_sha256"],
        }
    if set(reports) != EXPECTED_RUN_IDS:
        raise ValueError("both admitted practice validation captures are required")

    source = json.loads(source_path.read_text(encoding="utf-8"))
    if (source.get("benchmark_id") != "practice_transfer_benchmark_v1"
            or source.get("frozen") is not True):
        raise ValueError("source start metadata is not the frozen practice benchmark")
    source_run_ids = {str(row["run_id"]) for row in source["windows"]}
    if source_run_ids != EXPECTED_SOURCE_IDS:
        raise ValueError("source start metadata includes unexpected runs")

    dataset_hash = _sha256(dataset_path)
    with np.load(dataset_path, allow_pickle=False) as archive:
        required = ("run_ids", "run_splits", "sequence_bounds",
                    "sequence_run_index", "packet_sequence", "sample_time_ns")
        missing = [name for name in required if name not in archive.files]
        if missing:
            raise ValueError("isolated dataset lacks fields: " + ", ".join(missing))
        run_ids = archive["run_ids"].astype(str)
        splits = archive["run_splits"].astype(str)
        if set(run_ids) != EXPECTED_RUN_IDS or set(splits) != {"validation"}:
            raise ValueError("dataset must contain only the two validation runs")
        packet_sequence = archive["packet_sequence"].astype(np.int64, copy=False)
        sample_time_ns = archive["sample_time_ns"].astype(np.int64, copy=False)
        bounds = archive["sequence_bounds"].astype(np.int64, copy=False)
        sequence_run = archive["sequence_run_index"].astype(np.int64, copy=False)

        row_indices: dict[str, np.ndarray] = {}
        for run_index, run_id in enumerate(run_ids):
            spans = [tuple(map(int, span)) for span, owner
                     in zip(bounds, sequence_run) if owner == run_index]
            spans.sort()
            if not spans or any(left[1] != right[0]
                                for left, right in zip(spans, spans[1:])):
                raise ValueError(f"{run_id} does not form one contiguous capture")
            indices = np.concatenate([
                np.arange(start, end, dtype=np.int64) for start, end in spans])
            if np.any(np.diff(packet_sequence[indices]) != 1):
                raise ValueError(f"{run_id} has a packet gap in the fixed-timebase view")
            row_indices[str(run_id)] = indices

        run_id_map = {
            "practice_model_validation_r02": "practice_model_validation_r02",
            "practice_model_validation_r03": "practice_model_validation_r03",
        }
        windows = []
        for source_window in source["windows"]:
            source_id = str(source_window["run_id"])
            suffix = source_id.rsplit("_", 1)[-1]
            run_id = f"practice_model_validation_{suffix}"
            if run_id not in row_indices:
                raise ValueError(f"start refers to non-whitelisted run {source_id}")
            local_index = int(source_window["run_local_start_index"])
            indices = row_indices[run_id]
            if not 2 <= local_index < len(indices) - 80:
                raise ValueError(f"start falls outside supported data for {run_id}")
            global_index = int(indices[local_index])
            if (int(packet_sequence[global_index])
                    != int(source_window["packet_sequence"])
                    or int(sample_time_ns[global_index])
                    != int(source_window["sample_time_ns"])):
                raise ValueError("practice start packet/time does not match source metadata")
            row = dict(source_window)
            row["run_id"] = run_id
            row["global_start_index"] = global_index
            windows.append(row)

    if len(windows) != len(source["windows"]):
        raise ValueError("not all frozen source windows were rebound")
    report = dict(source)
    report.update({
        "dataset": str(dataset_path),
        "dataset_sha256": dataset_hash,
        "validation_reports": reports,
        "windows": windows,
        "dataset_run_id_by_validation_run": run_id_map,
        "view_id": "practice_validation_bag_whitelist_20261014",
        "source_start_metadata": {
            "path": str(source_path),
            "sha256": _sha256(source_path),
            "numeric_arrays_read": False,
        },
        "sealed_archive_policy": (
            "numeric arrays are loaded only from the supplied two-run "
            "validation dataset; the shared parent archive was not opened"),
    })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("source_benchmark", type=Path)
    parser.add_argument("--validation-report", type=Path, action="append",
                        required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.dataset, args.source_benchmark,
                   args.validation_report, args.output)
    print(json.dumps({
        "view_id": result["view_id"],
        "whitelisted_runs": sorted(result["dataset_run_id_by_validation_run"]),
        "windows": len(result["windows"]),
        "dataset_sha256": result["dataset_sha256"],
        "output": str(args.output.resolve()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
