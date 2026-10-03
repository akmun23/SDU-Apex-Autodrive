"""Strict provenance checks for fixed-cadence encoder sidecar variants."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _repo_path(value: str) -> Path:
    path = Path(value)
    if str(path).startswith("/workspace/"):
        path = REPO_ROOT / str(path).removeprefix("/workspace/")
    elif not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def is_fixed_cadence_variant(metadata: dict[str, Any],
                             evaluation_dataset: Path) -> bool:
    """Allow transfer only if every source array except valid wheel rates matches."""
    trained_value = metadata.get("dataset_path")
    if (not trained_value
            or metadata.get("wheel_state_source") != "raw_encoder"):
        return False
    trained_dataset = _repo_path(str(trained_value))
    if (not trained_dataset.is_file()
            or _sha256(trained_dataset) != metadata.get("dataset_sha256")):
        return False
    trained_manifest_path = trained_dataset.with_name(
        trained_dataset.stem + "_manifest.json")
    evaluation_dataset = evaluation_dataset.resolve()
    evaluation_manifest_path = evaluation_dataset.with_name(
        evaluation_dataset.stem + "_manifest.json")
    if not trained_manifest_path.is_file() or not evaluation_manifest_path.is_file():
        return False
    trained_manifest = json.loads(trained_manifest_path.read_text(encoding="utf-8"))
    evaluation_manifest = json.loads(
        evaluation_manifest_path.read_text(encoding="utf-8"))
    if (trained_manifest.get("source_dataset_sha256")
            != evaluation_manifest.get("source_dataset_sha256")
            or trained_manifest.get("source_manifest_sha256")
            != evaluation_manifest.get("source_manifest_sha256")
            or trained_manifest.get("split_policy")
            != evaluation_manifest.get("split_policy")
            or "source-stamp dt" not in str(
                trained_manifest.get("raw_speed_definition", ""))
            or "divided by fixed 25 ms" not in str(
                evaluation_manifest.get("raw_speed_definition", ""))):
        return False
    try:
        with np.load(trained_dataset, allow_pickle=False) as trained, \
                np.load(evaluation_dataset, allow_pickle=False) as evaluated:
            if set(trained.files) != set(evaluated.files):
                return False
            changed_rates = False
            for name in trained.files:
                left, right = trained[name], evaluated[name]
                if name == "encoder_raw_surface_mps":
                    if (left.shape != right.shape
                            or not np.array_equal(trained["encoder_raw_valid"],
                                                  evaluated["encoder_raw_valid"])):
                        return False
                    mask = np.asarray(evaluated["encoder_raw_valid"], dtype=bool)
                    changed_rates = bool(np.any(
                        np.abs(left[mask] - right[mask]) > 1e-7))
                    continue
                equal = (np.array_equal(left, right, equal_nan=True)
                         if np.issubdtype(left.dtype, np.number)
                         else np.array_equal(left, right))
                if not equal:
                    return False
            return changed_rates
    except (OSError, KeyError, ValueError):
        return False
