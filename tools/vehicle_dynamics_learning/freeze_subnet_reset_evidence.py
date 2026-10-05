#!/usr/bin/env python3
"""Freeze source data, comparator hashes, splits, and starts for the SUBNET reset."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "live_runs/derived_dynamics_learning_20260928"
RESET_ROOT = DATA_ROOT / "subnet_reset_20261004"
REGISTRY = DATA_ROOT / "experiment_registry_20261003.json"
STARTS_SOURCE = (
    DATA_ROOT / "full_modeling_reset_20261001/replacement_offline_sim_raceline_20261003/"
    "full_throttle_domain_v1/next_phase_after_2129427/frozen_eval_starts.json")
ACTIVE_ROOT = (
    DATA_ROOT / "full_modeling_reset_20261001/replacement_offline_sim_raceline_20261003/"
    "full_throttle_domain_v1")
V2_DATASET = (
    DATA_ROOT / "full_modeling_reset_20261001/replacement_offline_sim_raceline_20261002/"
    "replacement_teacher_dataset_v2_20261004/openplane_dynamics.npz")

DATASETS = {
    "subnet_body_training_source": V2_DATASET,
    "plain_gru_source": DATA_ROOT / "oracle_dataset/openplane_dynamics.npz",
    "wp19_openplane_source": (
        DATA_ROOT / "full_modeling_reset_20261001/replacement_offline_sim_raceline_20261002/"
        "encoder_raw_state_teacher_v1/openplane_dynamics_raw_wheels.npz"),
    "wp19_openplane_fixed40hz_sidecar": (
        ACTIVE_ROOT / "encoder_fixed40hz_v1/openplane_dynamics_raw_wheels_fixed40hz.npz"),
    "wp19_practice_source": (
        DATA_ROOT / "full_modeling_reset_20261001/replacement_offline_sim_raceline_20261002/"
        "encoder_raw_state_teacher_v1/practice_dynamics_raw_wheels.npz"),
    "wp19_practice_fixed40hz_sidecar": (
        ACTIVE_ROOT / "encoder_fixed40hz_v1/practice_dynamics_raw_wheels_fixed40hz.npz"),
    "practice_diagnostic_r02": (
        DATA_ROOT / "practice_transfer_validation_20261001_r02/openplane_dynamics.npz"),
    "practice_diagnostic_r03": (
        DATA_ROOT / "practice_transfer_validation_20261001_r03/openplane_dynamics.npz"),
}

COMPARATORS = {
    "wp19_parent": ACTIVE_ROOT / (
        "wp19_target_ablation_v2_common_encoder_mask/"
        "07_body_state_increment__encoder_angle_increment/model.pt"),
    "wp22_a2": ACTIVE_ROOT / (
        "wp22_augmented_state_space_seed101_smoke_v1/seed101_candidate.pt"),
    "wp23_a0": ACTIVE_ROOT / (
        "wp23_structural_ablations_seed101_v1/A0/seed101_A0_candidate.pt"),
    "wp23_a1": ACTIVE_ROOT / (
        "wp23_structural_ablations_seed101_v1/A1/seed101_A1_candidate.pt"),
}

GRU_ROOTS = {
    "plain_gru": DATA_ROOT / "nonlinear_model_tournament_gru_20260928",
    "high_steering_gru": (
        DATA_ROOT / "nonlinear_model_tournament_gru_throttle_rate_highsteer_20260928"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def split_inventory(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        ids_key = "run_ids" if "run_ids" in data else None
        split_key = "run_splits" if "run_splits" in data else None
        if ids_key is None or split_key is None:
            return {"available": False, "reason": "run split arrays absent"}
        run_ids = data[ids_key].astype(str).tolist()
        splits = data[split_key].astype(str).tolist()
        if len(run_ids) != len(splits) or len(set(run_ids)) != len(run_ids):
            raise ValueError(f"invalid whole-run split metadata: {path}")
        return {
            "available": True,
            "run_count": len(run_ids),
            "by_split": {
                split: [run for run, role in zip(run_ids, splits) if role == split]
                for split in sorted(set(splits))
            },
        }


def file_record(path: Path, role: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing {role}: {path}")
    return {
        "path": relative(path),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "role": role,
    }


def write_json(name: str, value: Any) -> None:
    path = RESET_ROOT / name
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def freeze() -> dict[str, Any]:
    if RESET_ROOT.exists() and any(RESET_ROOT.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty reset directory: {RESET_ROOT}")
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    starts = json.loads(STARTS_SOURCE.read_text(encoding="utf-8"))
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    expected_artifacts = {
        "WP19 selected checkpoint", "WP19 report", "WP20 report",
        "WP22 A2 checkpoint", "WP23 A0 checkpoint", "WP23 A1 checkpoint",
        "WP23 ablation report",
    }
    indexed = {item["label"]: item for item in
               registry["active_black_box_work_package_artifacts"]}
    missing = expected_artifacts - indexed.keys()
    if missing:
        raise ValueError(f"registry omitted required artifacts: {sorted(missing)}")
    for label in expected_artifacts:
        item = indexed[label]
        path = ROOT / item["path"]
        if not path.is_file() or sha256(path) != item["sha256"]:
            raise ValueError(f"indexed artifact hash failed: {label}")

    dataset_records = []
    for role, path in DATASETS.items():
        record = file_record(path, role)
        record["split_inventory"] = split_inventory(path)
        dataset_records.append(record)

    comparator_records = []
    for role, path in COMPARATORS.items():
        comparator_records.append(file_record(path, role))
    for role, model_root in GRU_ROOTS.items():
        report_path = model_root / "training_report.json"
        comparator_records.append(file_record(report_path, f"{role}_training_report"))
        for checkpoint in sorted(model_root.glob("member_*.pt")):
            comparator_records.append(file_record(checkpoint, f"{role}_member"))

    reset_source = {
        "repository_head": head,
        "handoff_claimed_head": "2129427838101eee277e17e91727810bbe2df687",
        "head_matches_handoff": (
            head == "2129427838101eee277e17e91727810bbe2df687"),
        "experiment_registry": file_record(REGISTRY, "current_head_experiment_registry"),
        "data_manifest_sha256": sha256(V2_DATASET.with_name("manifest.json")),
        "evaluation_starts_source": file_record(
            STARTS_SOURCE, "previously_frozen_development_and_practice_starts"),
    }
    split_sources = {record["role"]: record["split_inventory"]
                     for record in dataset_records}
    final_runs = sorted({
        run_id
        for inventory in split_sources.values()
        for split in ("final_test", "test", "experiment_test")
        for run_id in inventory.get("by_split", {}).get(split, [])
    })
    split_manifest = {
        "schema_version": 1,
        "source_commit": head,
        "development_validation": {
            "source": reset_source["evaluation_starts_source"],
            "run_ids": sorted(starts["split_roles"]["development_validation"]),
            "use": "model fitting/selection only; whole-run clustered statistics",
        },
        "practice_diagnostic": {
            "source": reset_source["evaluation_starts_source"],
            "run_ids": sorted(starts["split_roles"]["practice_diagnostic"]),
            "use": "transfer diagnostics only; never early stopping or architecture selection",
        },
        "consumed_or_previously_reserved_runs_development_only": final_runs,
        "test_or_final_test_eligible_for_new_model_selection": False,
        "new_independent_final_confirmation": {
            "status": "reserved_not_yet_collected_or_opened",
            "run_ids": [],
            "requirement": "new whole runs, pre-registered after model family and all settings freeze",
        },
        "source_run_split_inventories": split_sources,
    }
    starts_output = {
        "schema_version": 1,
        "source_commit": head,
        "source_artifact": reset_source["evaluation_starts_source"],
        "reuse_policy": (
            "Exact previously frozen whole-run starts retained. These start rows are "
            "development validation and practice diagnostics only; no test/final-test rows."),
        "horizon_semantics": starts["horizon_semantics"],
        "split_roles": starts["split_roles"],
    }

    RESET_ROOT.mkdir(parents=True, exist_ok=True)
    write_json("dataset_manifest.json", {
        "schema_version": 1,
        "source_commit": head,
        "purpose": "Frozen source provenance for the 40 Hz SUBNET modeling reset.",
        "body_sysid_source": relative(V2_DATASET),
        "records": dataset_records,
    })
    write_json("comparator_manifest.json", {
        "schema_version": 1,
        "source_commit": head,
        "promotion_status": "all comparators remain offline research baselines",
        "required_wp19_wp20_wp22_wp23_artifacts_indexed": True,
        "comparators": comparator_records,
    })
    write_json("split_manifest.json", split_manifest)
    write_json("evaluation_starts.json", starts_output)
    (RESET_ROOT / "source_commit.txt").write_text(head + "\n", encoding="utf-8")
    return {
        "reset_root": relative(RESET_ROOT),
        "source_commit": head,
        "handoff_head_mismatch": not reset_source["head_matches_handoff"],
        "dataset_sources": len(dataset_records),
        "comparator_artifacts": len(comparator_records),
        "indexed_required_artifacts": sorted(expected_artifacts),
        "development_validation_runs": len(split_manifest["development_validation"]["run_ids"]),
        "practice_diagnostic_runs": len(split_manifest["practice_diagnostic"]["run_ids"]),
        "consumed_final_or_test_run_ids": len(final_runs),
    }


if __name__ == "__main__":
    print(json.dumps(freeze(), indent=2))
