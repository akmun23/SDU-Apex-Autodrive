from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.vehicle_dynamics_learning.build_experiment_registry import (
    validate_registry,
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_registry_rejects_checkpoint_or_dataset_hash_mismatch(tmp_path: Path) -> None:
    checkpoint = tmp_path / "candidate.pt"
    dataset = tmp_path / "dataset.npz"
    checkpoint.write_bytes(b"checkpoint")
    dataset.write_bytes(b"dataset")
    entry = {
        "experiment_id": "candidate",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": "0" * 64,
        "dataset_path": str(dataset),
        "dataset_sha256": _digest(dataset),
        "train_runs": ["train-a"],
        "final_runs": ["final-a"],
        "decision": "diagnostic",
    }
    registry = {"default_candidate": None, "experiment_entries": [entry]}
    with pytest.raises(ValueError, match="checkpoint hash mismatch"):
        validate_registry(registry, tmp_path)


def test_registry_rejects_train_final_overlap(tmp_path: Path) -> None:
    entry = {
        "experiment_id": "candidate",
        "train_runs": ["run-a"],
        "final_runs": ["run-a"],
        "decision": "diagnostic",
    }
    with pytest.raises(ValueError, match="train/final run overlap"):
        validate_registry({"default_candidate": None, "experiment_entries": [entry]}, tmp_path)


def test_rejected_experiment_cannot_be_default(tmp_path: Path) -> None:
    entry = {"experiment_id": "bad", "decision": "rejected",
             "train_runs": [], "final_runs": []}
    with pytest.raises(ValueError, match="accepted status"):
        validate_registry({"default_candidate": "bad", "experiment_entries": [entry]}, tmp_path)


def test_accepted_default_requires_existing_registry_entry(tmp_path: Path) -> None:
    entry = {"experiment_id": "good", "decision": "accepted",
             "train_runs": [], "final_runs": []}
    validate_registry({"default_candidate": "good", "experiment_entries": [entry]}, tmp_path)


def test_current_generated_registry_provenance_and_split_gates() -> None:
    repo = Path(__file__).resolve().parents[2]
    path = repo / "live_runs/derived_dynamics_learning_20260928/experiment_registry_20261003.json"
    if not path.is_file():
        pytest.skip("registry is generated from local ignored experiment artifacts")
    registry = json.loads(path.read_text(encoding="utf-8"))
    validate_registry(registry, repo)
    assert registry["default_candidate"] is None
    assert registry["provenance_verification"]["every_claimed_checkpoint_checked"]
    assert registry["mandatory_branch_status"]["wp17_current_parent_causal_intervention"].startswith("not found")
