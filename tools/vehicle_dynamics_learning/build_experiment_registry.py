#!/usr/bin/env python3
"""Index prior offline-plant experiments and verify their local provenance."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any
from zipfile import BadZipFile, ZipFile
import pickletools
import itertools


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "live_runs/derived_dynamics_learning_20260928/experiment_registry_20261003.json"
DEFAULT_DOC_OUTPUT = ROOT / "docs/development/OFFLINE_PLANT_EXPERIMENT_REGISTRY_20261003.md"
INDEXED_NAME = re.compile(
    r"(gru|rssm|nssm|edssm|effective|raw.?encoder|raw.?wheel|contact.?slip|"
    r"reflection|symmetr|roll|four.?wheel|grey.?box|direct|practice|replay|"
    r"blind|final|cadence|wheel.?scale|encoder.?state|observer|black.?box|"
    r"intervention|causality|observability|frozen.?parent)", re.I)
DOCS_TO_READ = (
    "docs/development/REPLACEMENT_OFFLINE_SIM_RACELINE_PROGRESS_20261002.md",
    "docs/development/MASTER_VEHICLE_MODEL_ODOM_MPC_PROGRESS_20261001.md",
    "docs/development/FULL_MODELING_RESET_PROGRESS_20261001.md",
    "docs/development/EXTERNAL_REVIEW_OFFLINE_PLANT_HANDOFF_20261001.md",
    "docs/development/RIGID_BODY_TEACHER_PROGRESS_20261001.md",
    "docs/development/NONLINEAR_MODEL_DISCOVERY_UPDATE_20260928.md",
    "docs/development/OPEN_PLANE_PER_WHEEL_DYNAMICS.md",
    "docs/development/VEHICLE_DYNAMICS_DATA_CATALOG_20260930.md",
    "docs/development/OPENPLANE_THROTTLE_SURFACE_RESULTS_20260930.md",
    "docs/development/OFFLINE_PLANT_NEXT_STEPS_20260930.md",
    "docs/development/OFFLINE_PLANT_TRAINING_UPDATE_20260930.md",
    "docs/development/SENSOR_OBSERVER_PROGRESS_20260929.md",
    "docs/development/ENGINEERING_STATE.md",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def resolve_recorded_path(value: Any) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    for prefix in ("/workspace/", str(ROOT) + "/"):
        if text.startswith(prefix):
            return ROOT / text[len(prefix):]
    path = Path(text)
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def values_for_key(value: Any, key: str) -> list[Any]:
    found: list[Any] = []
    if isinstance(value, dict):
        for name, child in value.items():
            if name == key:
                found.append(child)
            found.extend(values_for_key(child, key))
    elif isinstance(value, list):
        for child in value:
            found.extend(values_for_key(child, key))
    return found


CHECKPOINT_META_KEYS = {
    "model", "encoder", "latent_size", "expert_count", "history_steps",
    "history_seconds", "future_inputs", "physical_state_names",
    "wheel_state_source", "include_raw_encoder_history",
    "include_wheel_innovation_history", "include_roll_state",
    "training_objective_mode", "history_feature_names",
    "acceleration_target_names", "acceleration_target_source",
    "dataset_path", "dataset_sha256", "training_runs", "validation_runs",
    "seed", "sampling_seed", "max_recursive_rollout_steps",
}


def checkpoint_metadata(path: Path | None) -> dict[str, Any]:
    """Read primitive model metadata from torch ZIP/pickle opcodes, never unpickle."""
    if path is None or not path.is_file():
        return {}
    try:
        with ZipFile(path) as archive:
            pickle_name = next(name for name in archive.namelist()
                               if name.endswith("/data.pkl"))
            operations = list(pickletools.genops(archive.read(pickle_name)))
    except (OSError, BadZipFile, StopIteration, ValueError):
        return {"parse_status": "unsupported_checkpoint_container"}

    memo: dict[int, Any] = {}
    result: dict[str, Any] = {}
    op_index = 0
    while op_index < len(operations):
        op, arg, _ = operations[op_index]
        if op.name in {"BINPUT", "LONG_BINPUT"} and op_index:
            prev_op, prev_arg, _ = operations[op_index - 1]
            if prev_op.name in {"BINUNICODE", "SHORT_BINUNICODE"}:
                memo[int(arg)] = prev_arg
        if op.name == "BINUNICODE" and arg in CHECKPOINT_META_KEYS:
            value_index = op_index + 1
            while (value_index < len(operations)
                   and operations[value_index][0].name in {
                       "BINPUT", "LONG_BINPUT", "MEMOIZE"}):
                value_index += 1
            if value_index >= len(operations):
                break
            value_op, value_arg, _ = operations[value_index]
            if value_op.name in {"BINUNICODE", "SHORT_BINUNICODE"}:
                result[arg] = value_arg
            elif value_op.name in {"BININT", "BININT1", "BININT2", "LONG1", "LONG4", "BINFLOAT"}:
                result[arg] = value_arg
            elif value_op.name == "NEWTRUE":
                result[arg] = True
            elif value_op.name == "NEWFALSE":
                result[arg] = False
            elif value_op.name == "NONE":
                result[arg] = None
            elif value_op.name == "EMPTY_LIST":
                values: list[Any] = []
                cursor = value_index + 1
                while cursor < len(operations):
                    list_op, list_arg, _ = operations[cursor]
                    if list_op.name in {"APPENDS", "APPEND"}:
                        break
                    if list_op.name in {"BINUNICODE", "SHORT_BINUNICODE"}:
                        values.append(list_arg)
                    elif list_op.name in {"BINGET", "LONG_BINGET"} and int(list_arg) in memo:
                        memo_value = memo[int(list_arg)]
                        if isinstance(memo_value, str):
                            values.append(memo_value)
                    cursor += 1
                result[arg] = values
        op_index += 1
    result["parse_status"] = "primitive_checkpoint_metadata_read_without_unpickling"
    return result


def path_hash_references(value: Any, kind: str) -> list[dict[str, Any]]:
    suffix = ".pt" if kind == "checkpoint" else ".npz"
    output: list[dict[str, Any]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if (kind in key.lower() and isinstance(child, str)
                    and child.lower().endswith(suffix)):
                hash_value = value.get(f"{key}_sha256")
                if hash_value is None and key == "checkpoint":
                    hash_value = value.get("checkpoint_sha256")
                if hash_value is None and key == "dataset":
                    hash_value = value.get("dataset_sha256")
                output.append({"path": child,
                               "sha256": str(hash_value) if hash_value else None})
            output.extend(path_hash_references(child, kind))
    elif isinstance(value, list):
        for child in value:
            output.extend(path_hash_references(child, kind))
    return output


def _brace_expand(pattern: str) -> list[str]:
    match = re.search(r"\{([^{}]+)\}", pattern)
    if not match:
        return [pattern]
    alternatives = match.group(1).split(",")
    return [expanded for option in alternatives
            for expanded in _brace_expand(
                pattern[:match.start()] + option + pattern[match.end():])]


def dataset_index(live_runs: Path, wanted_hashes: set[str]) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = {}
    if not wanted_hashes:
        return result
    for path in live_runs.rglob("*.npz"):
        digest = sha256(path)
        if digest in wanted_hashes:
            result.setdefault(digest, []).append(path)
    return result


def dataset_splits(path: Path) -> dict[str, list[str]]:
    try:
        import numpy as np
        with np.load(path, allow_pickle=False) as data:
            ids_key = "run_ids" if "run_ids" in data else None
            splits_key = "run_splits" if "run_splits" in data else None
            if ids_key is None or splits_key is None:
                return {}
            ids = data[ids_key].astype(str).tolist()
            splits = data[splits_key].astype(str).tolist()
    except (OSError, ValueError, KeyError):
        return {}
    grouped: dict[str, list[str]] = {}
    for run_id, split in zip(ids, splits):
        grouped.setdefault(split, []).append(run_id)
    return grouped


def classify(path: Path, summary: dict[str, Any] | None,
             report: dict[str, Any] | None) -> str:
    haystack = (str(path) + " " + json.dumps(summary or {}, sort_keys=True)).lower()
    if "contact_slip" in haystack or "four_wheel" in haystack or "greybox" in haystack:
        return "rejected"
    if any(word in haystack for word in ("incomplete", "interrupted", "partial")):
        return "incomplete"
    if report and report.get("accepted") is True:
        return "accepted"
    if any(word in haystack for word in ("parent", "baseline", "lead_rssm")):
        return "diagnostic"
    return "diagnostic" if summary or report else "unknown"


def model_family(path: Path) -> str:
    name = str(path).lower()
    for pattern, label in (
        ("contact_slip", "contact-slip recurrent plant"),
        ("rawwheelhistory", "raw encoder-history EDSSM"),
        ("wheelinnovation", "wheel-innovation EDSSM"),
        ("rawwheel", "raw wheel-state EDSSM"),
        ("edssm", "effective deep state-space model"),
        ("rssm", "recurrent state-space model"),
        ("nssm", "neural state-space model"),
        ("gru", "GRU sequence model"),
        ("four_wheel", "four-wheel grey-box model"),
        ("observer", "sensor observer"),
        ("cadence", "encoder cadence diagnostic"),
    ):
        if pattern in name:
            return label
    return "offline-plant experiment"


def extract_doc_artifact_refs(docs_dir: Path) -> list[dict[str, Any]]:
    refs: dict[str, set[str]] = {}
    # Keep braces: several old progress documents use brace-expanded globs.
    # A comma inside {...} is part of the path expression, not punctuation.
    pattern = re.compile(r"(?:\.\./)?live_runs/[^\s`\]>]+")
    for doc in sorted(docs_dir.glob("*.md")):
        # This file is a generated rendering of the registry itself. Parsing
        # its "missing reference" section on the next refresh would create
        # self-referential unresolved paths and make the registry non-idempotent.
        if doc.name == DEFAULT_DOC_OUTPUT.name:
            continue
        text = doc.read_text(encoding="utf-8", errors="replace")
        for match in pattern.finditer(text):
            value = match.group(0).rstrip(".,;:")
            while value.endswith(")") and "(" not in value:
                value = value[:-1]
            path = ROOT / value.removeprefix("../")
            refs.setdefault(relative(path), set()).add(relative(doc))
    rows = []
    for path, source in sorted(refs.items()):
        is_glob = any(char in path for char in "*?{}[]")
        expanded = [candidate for one_pattern in _brace_expand(path)
                    for candidate in glob.glob(str(ROOT / one_pattern), recursive=True)] if is_glob else []
        matches = sorted({relative(Path(item)) for item in expanded})
        target = ROOT / path
        rows.append({
            "path": path,
            "exists": target.exists(),
            "is_glob_pattern": is_glob,
            "glob_match_count": len(matches) if is_glob else None,
            "glob_matches": matches,
            "is_directory": target.is_dir(),
            "referenced_by": sorted(source),
        })
    return rows


DOCUMENT_REVIEW = {
    "docs/development/REPLACEMENT_OFFLINE_SIM_RACELINE_PROGRESS_20261002.md":
        "Current baseline and whole-run recursive metrics; authoritative current plant status.",
    "docs/development/MASTER_VEHICLE_MODEL_ODOM_MPC_PROGRESS_20261001.md":
        "Historical production model/odom record; not evidence that an offline plant passed recursive validation.",
    "docs/development/FULL_MODELING_RESET_PROGRESS_20261001.md":
        "Earlier modeling reset and experiment history; superseded where later whole-run results conflict.",
    "docs/development/EXTERNAL_REVIEW_OFFLINE_PLANT_HANDOFF_20261001.md":
        "External review evidence and caveats; retained as historical review, not a current model promotion.",
    "docs/development/RIGID_BODY_TEACHER_PROGRESS_20261001.md":
        "Earlier rigid-body teacher experiments; no accepted replacement plant established.",
    "docs/development/NONLINEAR_MODEL_DISCOVERY_UPDATE_20260928.md":
        "Earlier throttle-slew/roll/GRU tests; weak or negative evidence, not a general plant.",
    "docs/development/OPEN_PLANE_PER_WHEEL_DYNAMICS.md":
        "Historical narrow per-wheel/yaw analyses; proxies do not identify tire-force truth or a full plant.",
    "docs/development/VEHICLE_DYNAMICS_DATA_CATALOG_20260930.md":
        "Historical data inventory; its schemas/counts predate the replacement dataset and are not current totals.",
    "docs/development/OPENPLANE_THROTTLE_SURFACE_RESULTS_20260930.md":
        "Throttle response-surface coverage; useful input-response data, not recursive whole-vehicle validation.",
    "docs/development/OFFLINE_PLANT_NEXT_STEPS_20260930.md":
        "Earlier proposed sequence; superseded by the 2026-10-03 black-box handoff order.",
    "docs/development/OFFLINE_PLANT_TRAINING_UPDATE_20260930.md":
        "Historical training report; later recursive validation controls current conclusions.",
    "docs/development/SENSOR_OBSERVER_PROGRESS_20260929.md":
        "Sensor-only observer is a separate estimator, not the offline plant; prior GRU is not a full pose/drift observer.",
    "docs/development/ENGINEERING_STATE.md":
        "Broad historical workspace state; use the dated plant report and handoff for current scope/status.",
    "docs/development/BLACK_BOX_OFFLINE_PLANT_PROGRESS_20261003.md":
        "Current WP15–WP18 evidence and gates for the black-box offline plant handoff.",
}


def _run_metadata(dataset: Path | None, validation_ids: list[str]) -> dict[str, list[str]]:
    split_map = dataset_splits(dataset) if dataset and dataset.is_file() else {}
    return {
        "train_runs": split_map.get("train", []),
        "validation_runs": validation_ids or split_map.get("validation", []),
        "test_runs": [],
        "final_runs": [],
        "dataset_split_inventory": split_map,
    }


def build_registry(repo_root: Path = ROOT, output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    live_runs = repo_root / "live_runs"
    docs_dir = repo_root / "docs/development"
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True,
        text=True, capture_output=True).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo_root, check=True,
        text=True, capture_output=True).stdout

    training_records: list[tuple[Path, dict[str, Any], Path | None]] = []
    comparison_paths: list[Path] = []
    gate_data_root = (live_runs / "derived_dynamics_learning_20260928"
        / "full_modeling_reset_20261001"
        / "replacement_offline_sim_raceline_20261003"
        / "full_throttle_domain_v1")
    gate_paths = {
        "wp16": gate_data_root / "wp16_wheel_causality_20261003_v4.json",
        "wp17": gate_data_root / "wp17_frozen_parent_interventions_v4.json",
        "wp18": gate_data_root / "blackbox_observability_report_v1.json",
    }
    gate_reports = {name: read_json(path) for name, path in gate_paths.items()}
    requested_dataset_hashes: set[str] = set()
    for path in live_runs.rglob("*.json"):
        basename = path.name
        if basename in {"training_summary.json", "training_report.json"}:
            summary = read_json(path) or {}
            raw_hash = summary.get("dataset_sha256")
            if isinstance(raw_hash, str):
                requested_dataset_hashes.add(raw_hash)
            cp = resolve_recorded_path(summary.get("best_checkpoint"))
            if cp is None or not cp.exists():
                candidate = path.parent / "best.pt"
                cp = candidate if candidate.exists() else None
            training_records.append((path, summary, cp))
        elif (basename == "paired_comparison.json"
              or basename in {"comparison.json", "wheel_scale_diagnostic_v1.json"}
              or "cadence_alignment" in basename
              or "blind_final" in basename
              or "practice_replay" in basename
              or basename.startswith(("wp16_", "wp17_"))
              or "blackbox_observability_report" in basename):
            comparison_paths.append(path)

    datasets = dataset_index(live_runs, requested_dataset_hashes)
    final_evaluations_by_checkpoint: dict[str, set[str]] = {}
    for path in comparison_paths:
        data = next((gate_reports[name] for name, gate_path in gate_paths.items()
                     if gate_path.resolve() == path.resolve()), None)
        if data is None:
            data = read_json(path) or {}
        if "blind_final" not in path.name.lower() and "final" not in str(data.get("purpose", "")).lower():
            continue
        run_ids = {str(value) for value in values_for_key(data, "run_id")}
        checkpoint_ids = {str(value) for value in values_for_key(data, "checkpoint_sha256")}
        for checkpoint_id in checkpoint_ids:
            final_evaluations_by_checkpoint.setdefault(checkpoint_id, set()).update(run_ids)
    entries: list[dict[str, Any]] = []
    checkpoint_hashes: dict[str, str] = {}
    for summary_path, summary, checkpoint in sorted(
            training_records, key=lambda row: relative(row[0])):
        if not INDEXED_NAME.search(str(summary_path.parent)):
            continue
        digest = summary.get("dataset_sha256")
        dataset_choices = datasets.get(digest, []) if isinstance(digest, str) else []
        dataset = dataset_choices[0] if dataset_choices else None
        validation_ids = summary.get("validation_run_ids", [])
        validation_ids = [str(value) for value in validation_ids]
        split_rows = _run_metadata(dataset, validation_ids)
        best_validation = read_json(summary_path.parent / "best_validation.json") or {}
        report = read_json(summary_path.parent / "training_report.json")
        claimed_checkpoint = resolve_recorded_path(summary.get("best_checkpoint"))
        checkpoint_rel = relative(checkpoint) if checkpoint else None
        checkpoint_hash = sha256(checkpoint) if checkpoint and checkpoint.is_file() else None
        if checkpoint_rel and checkpoint_hash:
            checkpoint_hashes[checkpoint_rel] = checkpoint_hash
        dataset_rel = relative(dataset) if dataset else None
        dataset_hash_actual = sha256(dataset) if dataset and dataset.is_file() else None
        meta = summary.get("metadata", {}) if isinstance(summary.get("metadata"), dict) else {}
        checkpoint_meta = checkpoint_metadata(checkpoint)
        checkpoint_dataset_hash = checkpoint_meta.get("dataset_sha256")
        if (digest and checkpoint_dataset_hash
                and digest != checkpoint_dataset_hash):
            raise ValueError(f"checkpoint/dataset provenance mismatch: {summary_path.parent}")
        train_run_ids = checkpoint_meta.get("training_runs") or split_rows["train_runs"]
        validation_run_ids = (checkpoint_meta.get("validation_runs")
                              or validation_ids or split_rows["validation_runs"])
        state_names = checkpoint_meta.get("physical_state_names") or meta.get(
            "state_definition", [])
        if not state_names:
            state_names = ["u_com_mps", "v_com_mps", "yaw_rate_rps",
                           "steering_actual_rad", "throttle_feedback_norm"]
            if "wheel" in json.dumps(summary).lower():
                state_names.extend(["rear_left_surface_speed_mps",
                                    "rear_right_surface_speed_mps"])
        mode = summary.get("wheel_dynamics_mode", "")
        wheel_source = checkpoint_meta.get("wheel_state_source")
        measurement = ("raw cumulative encoder angle-derived rate" if "raw" in str(summary_path).lower()
                       else f"rear wheel input from {wheel_source}" if wheel_source
                       else "stored filtered rear-wheel state; exact timing defined by dataset view")
        if mode == "contact_slip":
            measurement = "contact-speed difference / contact-slip target"
        horizon_match = re.search(
            r"(?:^|[_-])(\d+(?:\.\d+)?)s(?:[_-]|$)", summary_path.parent.name)
        history_seconds = checkpoint_meta.get("history_seconds", meta.get("history_seconds"))
        checkpoint_meta = {
            **checkpoint_meta,
            "seed": summary.get("seed"),
            "latent_size": checkpoint_meta.get("latent_size", summary.get("latent_size")),
            "encoder": checkpoint_meta.get("encoder", summary.get("encoder")),
            "expert_count": checkpoint_meta.get("expert_count", summary.get("expert_count")),
            "global_steps": summary.get("global_steps"),
            "objective": summary.get("training_objective_mode"),
            "wheel_dynamics_mode": mode or None,
        }
        entries.append({
            "experiment_id": relative(summary_path.parent),
            "artifact_type": "trained_model",
            "date": next((part for part in summary_path.parts if re.fullmatch(r"20\d{6}", part)), None),
            "code_commit": summary.get("git_sha") or meta.get("git_sha"),
            "model_family": model_family(summary_path.parent),
            "architecture": checkpoint_meta,
            "dataset_path": dataset_rel,
            "dataset_sha256": dataset_hash_actual or digest,
            "dataset_sha256_matches_record": (
                dataset_hash_actual == digest if dataset_hash_actual and digest else None),
            "train_runs": train_run_ids,
            "validation_runs": validation_run_ids,
            "test_runs": split_rows["test_runs"],
            "final_runs": sorted(final_evaluations_by_checkpoint.get(
                checkpoint_hash or "", set())),
            "dataset_split_inventory": split_rows["dataset_split_inventory"],
            "test_and_final_test_used_for_training": summary.get(
                "test_and_final_test_used"),
            "history_seconds": history_seconds,
            "training_horizon_seconds": meta.get(
                "training_horizon_seconds",
                float(horizon_match.group(1)) if horizon_match else None),
            "state_definition": state_names,
            "measurement_definition": [measurement],
            "losses": {key: value for key, value in summary.items()
                       if "loss_weight" in key or key.endswith("objective_mode")},
            "checkpoint": checkpoint_rel,
            "checkpoint_sha256": checkpoint_hash,
            "checkpoint_claimed_path": relative(claimed_checkpoint) if claimed_checkpoint else None,
            "key_metrics": {
                "best_validation_selection_score": summary.get("best_validation_selection_score"),
                "best_validation_report": relative(summary_path.parent / "best_validation.json")
                    if (summary_path.parent / "best_validation.json").is_file() else None,
                "training_steps": summary.get("global_steps"),
            },
            "decision": classify(summary_path.parent, summary, report),
            "decision_reason": "Indexed from local training metadata; see paired comparisons and progress record. No candidate is promoted by this registry.",
            "may_repeat": False,
            "repeat_only_if": "A named upstream work-package gate identifies a specific unresolved hypothesis; do not repeat the same architecture/split/budget.",
            "training_metadata_path": relative(summary_path),
            "checkpoint_metadata": checkpoint_meta,
            "unresolved_metadata": [key for key, value in {
                "code_commit": summary.get("git_sha") or meta.get("git_sha"),
                "history_seconds": history_seconds,
                "training_horizon_seconds": meta.get(
                    "training_horizon_seconds",
                    float(horizon_match.group(1)) if horizon_match else None),
                "train_run_ids": split_rows["train_runs"],
                "dataset_path": dataset_rel,
            }.items() if value in (None, [])],
        })

    diagnostic_entries: list[dict[str, Any]] = []
    for path in sorted(comparison_paths):
        if not INDEXED_NAME.search(str(path)):
            continue
        data = read_json(path) or {}
        if not data:
            continue
        cp_refs = []
        for reference in path_hash_references(data, "checkpoint"):
            target = resolve_recorded_path(reference["path"])
            actual = sha256(target) if target and target.is_file() else None
            cp_refs.append({
                "path": relative(target) if target else reference["path"],
                "exists": bool(target and target.is_file()),
                "sha256_recorded": reference["sha256"],
                "sha256_actual": actual,
                "sha256_matches": (actual == reference["sha256"]
                                   if actual and reference["sha256"] else None),
            })
        data_refs = []
        for reference in path_hash_references(data, "dataset"):
            target = resolve_recorded_path(reference["path"])
            actual = sha256(target) if target and target.is_file() else None
            data_refs.append({
                "path": relative(target) if target else reference["path"],
                "exists": bool(target and target.is_file()),
                "sha256_recorded": reference["sha256"],
                "sha256_actual": actual,
                "sha256_matches": (actual == reference["sha256"]
                                   if actual and reference["sha256"] else None),
            })
        diagnostic_entries.append({
            "experiment_id": relative(path),
            "artifact_type": "diagnostic_or_comparison",
            "date": next((part for part in path.parts if re.fullmatch(r"20\d{6}", part)), None),
            "path": relative(path),
            "sha256": sha256(path),
            "purpose": data.get("purpose") or data.get("evaluation"),
            "dataset_sha256": data.get("dataset_sha256") or data.get("dynamic_dataset_sha256"),
            "checkpoint_sha256": data.get("checkpoint_sha256"),
            "checkpoint_sha256_values": sorted({str(value) for value in
                values_for_key(data, "checkpoint_sha256")}),
            "checkpoint_references": cp_refs,
            "dataset_references": data_refs,
            "run_ids": sorted({str(value) for value in values_for_key(data, "run_id")}),
            "final_dataset": next((str(value) for value in values_for_key(data, "final_dataset")), None),
            "final_dataset_sha256": next((str(value) for value in values_for_key(data, "final_dataset_sha256")), None),
            "test_and_final_test_used": data.get("test_and_final_test_used"),
            "future_truth_or_feedback_used": data.get("future_truth_or_feedback_used"),
            "decision": "diagnostic",
            "decision_reason": "Evidence artifact; not a promotable model selection result by itself.",
        })

    # WP19-WP23 did not all emit the legacy training_summary.json schema.
    # Index their frozen reports and checkpoints explicitly so a registry
    # refresh cannot silently omit the active black-box plant branch.
    active_root = gate_data_root
    next_phase_root = active_root / "next_phase_after_2129427"
    wp25_root = next_phase_root / "wp25_failure_localization"
    wp26_root = next_phase_root / "wp26_mechanism_ablations"
    active_specs = (
        ("WP19 selected checkpoint", active_root / "wp19_target_ablation_v2_common_encoder_mask"
         / "07_body_state_increment__encoder_angle_increment/model.pt", "checkpoint",
         active_root / "wp20_support_calibration_v2_model_train_runs"
         / "wp20_support_calibration_report.json", "selected_wp19_checkpoint_sha256"),
        ("WP19 report", active_root / "wp19_target_ablation_v2_common_encoder_mask"
         / "wp19_target_ablation_report.json", "report", None, None),
        ("WP20 report", active_root / "wp20_support_calibration_v2_model_train_runs"
         / "wp20_support_calibration_report.json", "report", None, None),
        ("WP22 A2 checkpoint", active_root / "wp22_augmented_state_space_seed101_smoke_v1"
         / "seed101_candidate.pt", "checkpoint", active_root / "wp22_augmented_state_space_seed101_smoke_v1"
         / "wp22_training_report.json", "checkpoint_sha256"),
        ("WP22 training report", active_root / "wp22_augmented_state_space_seed101_smoke_v1"
         / "wp22_training_report.json", "report", None, None),
        ("WP22 failure diagnosis", active_root / "wp22_augmented_state_space_seed101_smoke_v1"
         / "wp22_failure_diagnosis.json", "diagnostic", None, None),
        ("WP23 A0 checkpoint", active_root / "wp23_structural_ablations_seed101_v1/A0"
         / "seed101_A0_candidate.pt", "checkpoint", active_root / "wp23_structural_ablations_seed101_v1/A0"
         / "wp23_A0_training_report.json", "checkpoint_sha256"),
        ("WP23 A0 training report", active_root / "wp23_structural_ablations_seed101_v1/A0"
         / "wp23_A0_training_report.json", "report", None, None),
        ("WP23 A1 checkpoint", active_root / "wp23_structural_ablations_seed101_v1/A1"
         / "seed101_A1_candidate.pt", "checkpoint", active_root / "wp23_structural_ablations_seed101_v1/A1"
         / "wp23_A1_training_report.json", "checkpoint_sha256"),
        ("WP23 A1 training report", active_root / "wp23_structural_ablations_seed101_v1/A1"
         / "wp23_A1_training_report.json", "report", None, None),
        ("WP23 ablation report", active_root / "wp23_structural_ablations_seed101_v1"
         / "wp23_structural_ablation_report.json", "diagnostic", None, None),
        ("WP24 frozen comparators", next_phase_root / "frozen_comparators.json",
         "manifest", None, None),
        ("WP24 frozen evaluation starts", next_phase_root / "frozen_eval_starts.json",
         "manifest", None, None),
        ("WP24 A2 reproduction", wp25_root / "wp24_reproduction_report.json",
         "diagnostic", None, None),
        ("WP25 scoring report", wp25_root / "wp25_score_only_diagnostics.json",
         "diagnostic", None, None),
        ("WP25 loss-gradient attribution", wp25_root / "wp25_loss_gradient_attribution.json",
         "diagnostic", None, None),
        ("WP26 mechanism report", wp26_root / "wp26_mechanism_ablation_report.json",
         "diagnostic", None, None),
        ("WP26 D1 checkpoint", wp26_root / "D1/checkpoint.pt", "checkpoint",
         wp26_root / "D1/training_report.json", "checkpoint_sha256"),
        ("WP26 D1 training report", wp26_root / "D1/training_report.json",
         "report", None, None),
        ("WP26 D2 checkpoint", wp26_root / "D2/checkpoint.pt", "checkpoint",
         wp26_root / "D2/training_report.json", "checkpoint_sha256"),
        ("WP26 D2 training report", wp26_root / "D2/training_report.json",
         "report", None, None),
        ("WP26 D3 checkpoint", wp26_root / "D3/checkpoint.pt", "checkpoint",
         wp26_root / "D3/training_report.json", "checkpoint_sha256"),
        ("WP26 D3 training report", wp26_root / "D3/training_report.json",
         "report", None, None),
    )
    active_artifacts = []
    for label, path, kind, report_path, report_hash_key in active_specs:
        if not path.is_file():
            raise FileNotFoundError(f"required active-branch artifact is missing: {path}")
        actual = sha256(path)
        expected = None
        if kind == "checkpoint" and report_path is not None:
            report = read_json(report_path)
            if not report:
                raise ValueError(f"cannot read checkpoint report: {report_path}")
            expected = report.get(report_hash_key)
            if not expected or expected != actual:
                raise ValueError(f"active checkpoint hash disagrees with its report: {path}")
        if kind == "checkpoint":
            checkpoint_hashes[relative(path)] = actual
        active_artifacts.append({
            "label": label,
            "kind": kind,
            "path": relative(path),
            "sha256": actual,
            "report_recorded_sha256": expected,
            "hash_matches_report": (actual == expected if expected else None),
        })

    doc_refs = extract_doc_artifact_refs(docs_dir)
    referenced_missing = [
        item for item in doc_refs
        if not item["exists"] and not item["is_glob_pattern"]
    ]
    unresolved_globs = [item for item in doc_refs
                        if item["is_glob_pattern"] and not item["glob_match_count"]]
    unresolved_checkpoints = [
        entry["checkpoint_claimed_path"] for entry in entries
        if entry.get("checkpoint_claimed_path") and not entry.get("checkpoint")
    ]
    unresolved_datasets = [
        {"sha256": entry["dataset_sha256"], "experiment_id": entry["experiment_id"]}
        for entry in entries if entry.get("dataset_sha256") and not entry.get("dataset_path")
    ]
    unresolved_diagnostic_paths = [
        {"experiment_id": entry["experiment_id"], "reference": reference}
        for entry in diagnostic_entries
        for reference in (entry["checkpoint_references"] + entry["dataset_references"])
        if reference["exists"] and reference["sha256_recorded"]
        and reference["sha256_matches"] is not True
    ]
    registry = {
        "schema_version": 1,
        "generated_at_local_date": "2026-10-03",
        "repository_head": head,
        "worktree_dirty_at_generation": bool(status.strip()),
        "default_candidate": None,
        "candidate_selection_policy": "A default can be selected only when a model has explicit accepted status at all required gates; no current model satisfies that condition.",
        "counts": {
            "training_metadata_files_found": len(training_records),
            "indexed_trained_models": len(entries),
            "indexed_diagnostic_artifacts": len(diagnostic_entries),
            "document_live_run_references": len(doc_refs),
            "unresolved_document_references": len(referenced_missing),
            "unresolved_document_globs": len(unresolved_globs),
        },
        "experiment_entries": entries,
        "diagnostic_entries": diagnostic_entries,
        "active_black_box_work_package_artifacts": active_artifacts,
        "document_artifact_references": doc_refs,
        "reviewed_documents": [
            {"path": path, "exists": (repo_root / path).is_file(),
             "disposition": disposition}
            for path, disposition in DOCUMENT_REVIEW.items()],
        "mandatory_branch_status": {
            "plain_gru": "preserved as comparator; no evidence of a validated recursive full-lap plant",
            "high_steering_gru": "preserved as comparator; regime gain did not establish whole-run plant accuracy",
            "raw_encoder_state": "diagnostic; mixed domain tradeoff; not promoted",
            "raw_encoder_history": "diagnostic; not a consistent global win; not promoted",
            "wheel_innovation_history": "diagnostic; not promoted",
            "contact_slip": "rejected on paired recursive dynamic and practice comparisons",
            "fixed_25ms_encoder": "separate causal measurement view; WP16 target audit complete and WP17 classifies measured processed wheel rate as measurement/output only",
            "current_parent_edssm": "frozen comparator only; recursive drift remains unacceptable",
            "moe_candidate": "rejected for plant use: pose gains accompanied by wheel/speed and practice-transfer regressions",
            "four_wheel_greybox": "rejected as race-domain nominal plant; prior grey-box relaxation did not remove recursive mismatch",
            "direct_transformer_and_direct_predictors": "diagnostic or incomplete; no accepted autonomous multi-step plant",
            "reflection_symmetry": "historical diagnostic branch only; no promoted whole-run result",
            "roll_conditioning": "not promoted; prior held-out practice evidence did not show material explanatory gain",
            "sensor_observer_gru": "separate sensor-only observer, not a plant; no full-pose drift acceptance",
            "wp14_blind_final": "already consumed for development interpretation; prohibited for further tuning",
            "wp16_signal_and_wheel_audit": (
                "complete" if gate_reports["wp16"] and gate_reports["wp16"].get(
                    "wp16_gate", {}).get("status") == "complete"
                else "incomplete or report not found"),
            "wp17_current_parent_causal_intervention": (
                "complete; rear processed wheel rate is "
                + str(gate_reports["wp17"].get("wp17_decision", {}).get(
                    "rear_processed_wheel_rate"))
                if gate_reports["wp17"] else "not found; required before WP18/WP19"),
            "wp18_history_sufficiency": (
                str(gate_reports["wp18"].get("history_selection", {}).get(
                    "status"))
                if gate_reports["wp18"] else "not found; no history claim"),
            "serious_replacement_training": (
                "WP19/WP20 closed; WP22/A0/A1 and WP26 D1/D2/D3 remain unpromoted; the A2 family stop gate was reached after WP26"),
        },
        "work_package_evidence": {
            name: {
                "path": relative(path),
                "exists": path.is_file(),
                "sha256": sha256(path) if path.is_file() else None,
                "wp16_status": (report.get("wp16_gate", {}).get("status")
                                if name == "wp16" and report else None),
                "wp17_decision": (report.get("wp17_decision", {}).get(
                    "rear_processed_wheel_rate")
                                  if name == "wp17" and report else None),
                "wp18_history_status": (report.get("history_selection", {}).get(
                    "status") if name == "wp18" and report else None),
                "wp18_chosen_history": (report.get("history_selection", {}).get(
                    "chosen_history_length") if name == "wp18" and report else None),
            }
            for name, (path, report) in {
                key: (gate_paths[key], gate_reports[key])
                for key in gate_paths}.items()},
        "provenance_verification": {
            "checkpoint_hashes": checkpoint_hashes,
            "active_black_box_artifacts_all_present_and_hashed": all(
                item["sha256"] for item in active_artifacts),
            "every_claimed_checkpoint_checked": not unresolved_checkpoints,
            "every_resolved_dataset_hash_checked": all(
                entry.get("dataset_sha256_matches_record") is True
                for entry in entries if entry.get("dataset_path")),
            "unresolved_checkpoints": unresolved_checkpoints,
            "unresolved_datasets": unresolved_datasets,
            "unresolved_diagnostic_paths": unresolved_diagnostic_paths,
            "unresolved_document_references": referenced_missing,
            "unresolved_document_globs": unresolved_globs,
        },
    }
    validate_registry(registry, repo_root)
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return registry


def validate_registry(registry: dict[str, Any], repo_root: Path = ROOT) -> None:
    if registry.get("default_candidate") is not None:
        selected = next((entry for entry in registry.get("experiment_entries", [])
                         if entry.get("experiment_id") == registry["default_candidate"]), None)
        if selected is None or selected.get("decision") != "accepted":
            raise ValueError("default candidate must exist and have accepted status")
    for entry in registry.get("experiment_entries", []):
        for path_key, hash_key in (("checkpoint", "checkpoint_sha256"),
                                   ("dataset_path", "dataset_sha256")):
            path_value, expected = entry.get(path_key), entry.get(hash_key)
            if path_value is None:
                continue
            path = resolve_recorded_path(path_value)
            if path is None or not path.is_file():
                if expected:
                    raise ValueError(f"claimed {path_key} is missing: {path_value}")
                continue
            if expected and sha256(path) != expected:
                raise ValueError(f"{path_key} hash mismatch: {path_value}")
        train_ids, final_ids = set(entry.get("train_runs", [])), set(entry.get("final_runs", []))
        if train_ids & final_ids:
            raise ValueError(f"train/final run overlap in {entry['experiment_id']}: {sorted(train_ids & final_ids)}")
        if entry.get("decision") == "rejected" and registry.get("default_candidate") == entry.get("experiment_id"):
            raise ValueError("rejected experiment cannot be selected as default")
    for entry in registry.get("diagnostic_entries", []):
        for reference in (entry.get("checkpoint_references", [])
                          + entry.get("dataset_references", [])):
            if not reference.get("exists") or not reference.get("sha256_recorded"):
                continue
            path = resolve_recorded_path(reference["path"])
            if path is None or not path.is_file():
                raise ValueError(f"diagnostic artifact reference is missing: {reference['path']}")
            if sha256(path) != reference["sha256_recorded"]:
                raise ValueError(f"diagnostic artifact hash mismatch: {reference['path']}")
    for item in registry.get("active_black_box_work_package_artifacts", []):
        path = resolve_recorded_path(item["path"])
        if path is None or not path.is_file() or sha256(path) != item["sha256"]:
            raise ValueError(f"active black-box artifact hash mismatch: {item['path']}")
        if item.get("report_recorded_sha256") and not item.get("hash_matches_report"):
            raise ValueError(f"active checkpoint disagrees with its report: {item['path']}")


def render_markdown(registry: dict[str, Any], output_path: Path = DEFAULT_DOC_OUTPUT) -> None:
    lines = [
        "# Offline Plant Experiment Registry — 2026-10-03", "",
        f"Repository HEAD: `{registry['repository_head']}`. Worktree was dirty at generation: `{registry['worktree_dirty_at_generation']}`.",
        "", "This is an evidence index, not a model promotion. The registry has no default candidate because no plant has passed the required recursive and task-level gates.",
        "", "## Coverage", "",
        f"- Training metadata files found: {registry['counts']['training_metadata_files_found']}; indexed model runs: {registry['counts']['indexed_trained_models']}.",
        f"- Indexed diagnostic/comparison artifacts: {registry['counts']['indexed_diagnostic_artifacts']}. ",
        f"- `live_runs/` references in development documents: {registry['counts']['document_live_run_references']}; missing literal paths: {registry['counts']['unresolved_document_references']}; unmatched globs: {registry['counts']['unresolved_document_globs']}.",
        f"- Checkpoint hashes checked: {len(registry['provenance_verification']['checkpoint_hashes'])}; unresolved dataset/checkpoint references: {len(registry['provenance_verification']['unresolved_datasets'])}/{len(registry['provenance_verification']['unresolved_checkpoints'])}; recorded diagnostic hash mismatches: {len(registry['provenance_verification']['unresolved_diagnostic_paths'])}.",
        "", "## Current branch decisions", "",
    ]
    for branch, status in registry["mandatory_branch_status"].items():
        lines.append(f"- **{branch}:** {status}.")
    lines.extend(["", "## Active black-box work-package artifacts", ""])
    for item in registry.get("active_black_box_work_package_artifacts", []):
        lines.append(f"- {item['label']}: `{item['path']}` — SHA-256 `{item['sha256']}`.")
    lines.extend(["", "## Current interpretation", "",
        "The raw-wheel, raw-history, wheel-innovation, and contact-slip branches do not provide a universally better recursively coherent model. Contact-slip is rejected. Raw encoder state/history are mixed tradeoffs; neither is accepted as a production state. The fixed-40-Hz view is a separate, causally aligned measurement representation and must not be conflated with the source-stamp-derived sidecar. The two-expert candidate improved some dynamic pose scores while degrading other state channels and practice transfer. The frozen EDSSM parent remains a comparator, not a usable offline plant.",
        "", "WP17 is complete: the fixed-25-ms and stored-filtered wheel-rate oracles predominantly regress the frozen parent on held-out dynamic recursive endpoints, so processed rear-wheel rate is measurement/output only. This does not reject an internally inferred latent traction state.",
        "", "WP18 is complete with no supported history-length plateau under the tested run-balanced neighbor estimator. This is not evidence that physical history is useless: the tested trailing-mean/slope summaries and changing match dimensions limit that claim. WP19 and WP20 are closed; WP22/A0/A1 and WP26 D1/D2/D3 are indexed below but none passed their material gates. The A2 family stop gate is reached; conditional WP27 is not run.",
        "", "## Reading the machine registry", "",
        "`live_runs/derived_dynamics_learning_20260928/experiment_registry_20261003.json` contains per-run metadata, split IDs, checkpoint/dataset SHA-256 values when resolved, all discovered comparison-artifact references, and missing metadata explicitly. Experiments default to `diagnostic`, `incomplete`, `rejected`, or `unknown`; none are promoted by filename or a single score.",
        "", "Older evidence is retained as historical context. Where it conflicts with the 2026-10-02 replacement-plant report or the 2026-10-03 handoff, the later whole-run recursive validation and current handoff's gates take precedence; no earlier local/conditional yaw result is treated as a validated black-box plant.",
        "", "## Reviewed source documents", ""])
    for item in registry["reviewed_documents"]:
        status = "present" if item["exists"] else "missing"
        lines.append(f"- `{item['path']}` ({status}): {item['disposition']}")
    lines.extend(["", "## Referenced artifacts not present", ""])
    missing = registry["provenance_verification"]["unresolved_document_references"]
    globs = registry["provenance_verification"]["unresolved_document_globs"]
    if not missing and not globs:
        lines.append("None.")
    for item in missing:
        lines.append(f"- Missing literal reference `{item['path']}` from {', '.join(item['referenced_by'])}; it is not a present indexed checkpoint/dataset/comparison artifact.")
    for item in globs:
        lines.append(f"- Unmatched historical glob `{item['path']}` from {', '.join(item['referenced_by'])}.")
    lines.extend(["", "The historical `model_eval_venv` and `vehicle_model_training_env` references are absent local directories, not missing training outputs. Wildcard/brace references are expanded and counted separately; they are not silently reported as resolved files.", ""])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_DOC_OUTPUT)
    args = parser.parse_args()
    registry = build_registry(output=args.output)
    render_markdown(registry, args.markdown)
    print(json.dumps({"registry": str(args.output), "markdown": str(args.markdown),
                      "counts": registry["counts"],
                      "mandatory_branch_status": registry["mandatory_branch_status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
