"""Reproducible run manifests for offline vehicle-model experiments."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def write_standard_artifacts(output_dir: Path, dataset_path: Path,
                             data: dict[str, Any], config: dict[str, Any],
                             seed: int, report: dict[str, Any],
                             source_files: tuple[str, ...]) -> dict[str, Any]:
    """Write the handoff-required config, provenance, training/model reports."""
    repo_root = Path(__file__).resolve().parents[2]
    dataset_path = dataset_path.resolve()
    split_rows = [
        {"run_id": str(run_id), "split": str(split)}
        for run_id, split in zip(data["run_ids"], data["splits"])
    ]
    try:
        git_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True,
            text=True, capture_output=True).stdout.strip()
        git_dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=repo_root, check=True,
            text=True, capture_output=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        git_sha, git_dirty = None, None

    manifest_path = dataset_path.parent / "manifest.json"
    source_hashes = {}
    for relative in source_files:
        path = repo_root / relative
        if path.is_file():
            source_hashes[relative] = _sha256(path)
    provenance = {
        "dataset_path": str(dataset_path),
        "dataset_sha256": _sha256(dataset_path),
        "dataset_manifest_path": str(manifest_path) if manifest_path.is_file() else None,
        "dataset_manifest_sha256": (
            _sha256(manifest_path) if manifest_path.is_file() else None),
        "split_manifest_sha256": _canonical_hash(split_rows),
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "seed": int(seed),
        "source_sha256": source_hashes,
    }
    config_payload = {"seed": int(seed), **config}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(config_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    (output_dir / "job_manifest.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")

    report["config"] = config_payload
    report["provenance"] = provenance
    (output_dir / "training_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    model_report = {
        "architecture": report.get("architecture"),
        "best_step": report.get("best_step"),
        "best_validation_score": report.get("best_validation_score"),
        "validation": report.get("validation"),
        "test_scored_once": report.get("test_scored_once"),
        "test": report.get("test"),
        "limitations": report.get("limitations", []),
        "checkpoint": report.get("checkpoint"),
        "provenance": provenance,
    }
    (output_dir / "model_report.json").write_text(
        json.dumps(model_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return provenance
