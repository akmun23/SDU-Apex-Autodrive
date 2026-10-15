"""Metadata-only guard for split-mixed compressed NPZ datasets."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def assert_no_sealed_runs_in_packed_archive(path: Path) -> None:
    """Refuse packed datasets with test/final-test runs before reading frames.

    NPZ numeric members are compressed as whole arrays. Filtering rows after
    loading ``frames`` still reads every run in that member, so only the small
    run/split roster may be inspected before deciding whether to load it.
    """
    path = Path(path)
    with np.load(path, allow_pickle=False) as metadata:
        required = {"run_ids", "run_splits"}
        missing = required - set(metadata.files)
        if missing:
            raise ValueError(f"dataset missing split metadata: {sorted(missing)}")
        run_ids = metadata["run_ids"].astype(str)
        run_splits = metadata["run_splits"].astype(str)
        if run_ids.ndim != 1 or run_splits.shape != run_ids.shape:
            raise ValueError("dataset run IDs and split roster disagree")
        sealed = (run_splits == "test") | (run_splits == "final_test")
        sealed_ids = run_ids[sealed].tolist()
    if sealed_ids:
        raise ValueError(
            f"refusing to decompress numeric arrays from {path}: packed archive "
            f"contains sealed test/final_test runs {sealed_ids}; use a clean "
            "per-run training/validation dataset")
