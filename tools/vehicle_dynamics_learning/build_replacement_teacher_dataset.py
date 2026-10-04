#!/usr/bin/env python3
"""Append admitted dynamic-coupled captures to the frozen race-domain view.

The 19-run schema-8 race view is read as the immutable prefix. Whole-run,
quality-gated dynamic captures are appended; legacy bags are never
re-extracted. The output is the schema-9 replacement teacher dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "live_runs/derived_dynamics_learning_20260928"
TASK_ROOT = (DATA_ROOT / "full_modeling_reset_20261001/"
             "replacement_offline_sim_raceline_20261002")
BASELINE_REGISTRY = (DATA_ROOT / "full_modeling_reset_20261001/"
                     "replacement_sim_baseline_registry_20261002.json")
BASE_DATASET = (DATA_ROOT / "plant_teacher_race_domain_moderate_braking_20261001/"
                "cooldown_2s/openplane_dynamics.npz")
BASE_MANIFEST = BASE_DATASET.with_name("manifest.json")
OUTPUT_DIR = TASK_ROOT / "replacement_teacher_dataset_v1"

EXPECTED_CAPTURES = {
    "openplane_dyn_coupled_train_r01_20261002":
        ("train", 20261002, "race_domain_dynamic_coupled_train"),
    "openplane_dyn_coupled_train_r02_20261002":
        ("train", 20261003, "race_domain_dynamic_coupled_train"),
    "openplane_dyn_coupled_train_r03_20261002":
        ("train", 20261004, "race_domain_dynamic_coupled_train"),
    "openplane_dyn_coupled_validation_r01_20261002":
        ("validation", 20261005, "race_domain_dynamic_coupled_validation"),
    "openplane_dyn_coupled_validation_r02_20261002":
        ("validation", 20261006, "race_domain_dynamic_coupled_validation"),
    "openplane_dyn_coupled_train_r04_20261004":
        ("train", 20261007, "race_domain_dynamic_coupled_train"),
}
DEFAULT_CAPTURE_DIRS = (
    TASK_ROOT / "capture_qc_r01",
    TASK_ROOT / "capture_qc_r02",
    TASK_ROOT / "capture_qc_r03",
    TASK_ROOT / "capture_qc_validation_r01",
    TASK_ROOT / "capture_qc_validation_r02",
    TASK_ROOT / "capture_qc_r04_20261004",
)

SIMULATOR_DT_S = 0.025
MAX_BODY_SPEED_MPS = 12.0
MAX_TILT_RAD = np.deg2rad(8.0)
COMMAND_TOPICS = {
    "/autodrive/roboracer_1/steering_command",
    "/autodrive/roboracer_1/throttle_command",
}
REQUIRED_STREAMS = {
    "/autodrive/roboracer_1/bridge_packet_timing",
    "/autodrive/roboracer_1/imu",
    "/autodrive/roboracer_1/left_encoder",
    "/autodrive/roboracer_1/odom",
    "/autodrive/roboracer_1/right_encoder",
    "/autodrive/roboracer_1/steering",
    "/autodrive/roboracer_1/steering_command",
    "/autodrive/roboracer_1/throttle",
    "/autodrive/roboracer_1/throttle_command",
}
FRAME_KEYS = (
    "frames", "sensor_frames", "sensor_valid", "imu_attitude_frames",
    "imu_attitude_valid", "dt_s", "packet_sequence", "sample_time_ns",
    "odom_pose_xyyaw", "simulator_pose_xyyaw", "lap_count",
    "simulator_rigid_state", "simulator_linear_acceleration",
    "frame_run_index", "frame_reset_index", "frame_source_index",
    "frame_source_sequence_index", "frame_domain_speed_mps",
)
SEQUENCE_KEYS = (
    "sequence_bounds", "sequence_run_index", "sequence_labels",
    "sequence_condition_id", "sequence_reset_index",
    "sequence_replicate_index", "sequence_source_condition_id",
    "sequence_source_index",
)
RUN_KEYS = (
    "run_ids", "run_families", "run_splits", "training_families",
)
NEW_RUN_METADATA_KEYS = ("run_capture_profiles", "run_capture_seeds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _frame_source_sequence_index(
        bounds: np.ndarray, frame_count: int) -> np.ndarray:
    """Map every source frame to its own no-cross-reset sequence index."""
    result = np.full(frame_count, -1, dtype=np.int32)
    cursor = 0
    for sequence_index, (start_value, end_value) in enumerate(bounds):
        start, end = int(start_value), int(end_value)
        if start != cursor or end <= start or end > frame_count:
            raise ValueError("source sequence bounds are not contiguous")
        result[start:end] = sequence_index
        cursor = end
    if cursor != frame_count or np.any(result < 0):
        raise ValueError("source sequences do not cover every frame")
    return result


def _condition_labels(seed: int, reset_indices: np.ndarray) -> list[str]:
    """Name each reset-isolated sequence by its seeded plan condition."""
    from tools.race_domain_dynamic_coupled_plan import build_dynamic_coupled_plan

    plan = build_dynamic_coupled_plan(seed)
    if len(plan) != 9 or not np.array_equal(reset_indices, np.arange(1, 10)):
        raise ValueError("dynamic-coupled sequence/reset layout changed")
    return [f"{row.condition_id}/reset_epoch_{reset_index:02d}"
            for row, reset_index in zip(plan, reset_indices)]


def _validate_streams(row: dict[str, Any]) -> None:
    streams = row.get("streams") or {}
    if set(streams) != REQUIRED_STREAMS:
        raise ValueError("capture stream set differs from the required set")
    for topic, values in streams.items():
        rate = float(values["hz"])
        p95_gap = float(values["gap_p95_ms"])
        max_gap = float(values["gap_max_ms"])
        max_allowed = 120.0 if topic in COMMAND_TOPICS else 60.0
        if rate < 38.0 or p95_gap > 35.0 or max_gap > max_allowed:
            raise ValueError(f"capture stream gate failed: {topic}")


def _validate_capture(capture_dir: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    manifest_path = capture_dir / "manifest.json"
    archive_path = capture_dir / "openplane_dynamics.npz"
    if not manifest_path.is_file() or not archive_path.is_file():
        raise ValueError(f"incomplete capture QC output: {capture_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest.get("runs") or []
    if len(rows) != 1:
        raise ValueError(f"capture must contain one whole run: {capture_dir}")
    row = rows[0]
    run_id = str(row.get("run_id", ""))
    if run_id not in EXPECTED_CAPTURES:
        raise ValueError(f"unexpected dynamic-coupled run: {run_id}")
    split, expected_seed, expected_profile = EXPECTED_CAPTURES[run_id]
    if run_id in manifest.get("errors", {}):
        raise ValueError(f"dataset preparation failed for {run_id}")
    if (row.get("effective_split") != split
            or row.get("explicit_split_override") != split):
        raise ValueError(f"whole-run split mismatch for {run_id}")
    if (row.get("aborted") is not False
            or row.get("reason") != "schedule complete"
            or row.get("quality_failures") not in ([], None)
            or row.get("whole_bag_quality_failures") not in ([], None)
            or not row.get("clean_stream_and_collision_gate")
            or row.get("collisions") != [0, 0]
            or int(row.get("timing_faults", -1)) != 0
            or int(row.get("valid_phases", -1)) != 66
            or int(row.get("invalid_phases", -1)) != 0
            or int(row.get("unscored_phases", -1)) != 18):
        raise ValueError(f"capture completion or quality gate failed: {run_id}")
    if (not row.get("continuous_whole_run_export")
            or row.get("quality_gate_scope") != "whole_bag"):
        raise ValueError(f"capture is not a whole-run export: {run_id}")
    _validate_streams(row)

    alignment = row.get("packet_sequence_alignment") or {}
    if (int(alignment.get("total_samples", 0)) <= 0
            or float(alignment.get("match_fraction", 0.0)) < 0.999):
        raise ValueError(f"packet alignment gate failed: {run_id}")
    reset = row.get("reset_metadata") or {}
    if (not reset.get("topic_present") or int(reset.get("epoch_count", 0)) != 9):
        raise ValueError(f"nine recorded reset epochs are required: {run_id}")

    bag_path = Path(str(row.get("bag", "")))
    if not bag_path.is_absolute():
        bag_path = REPO_ROOT / bag_path
    run_dir = bag_path.parent.parent
    log_path = run_dir / "experiment.log"
    if not bag_path.is_file() or not log_path.is_file():
        raise ValueError(f"capture bag/log is missing: {run_id}")
    log = log_path.read_text(encoding="utf-8", errors="replace")
    start = re.search(r"profile=([^, ]+), seed=(\d+)", log)
    if (start is None or start.group(1) != expected_profile
            or int(start.group(2)) != expected_seed):
        raise ValueError(f"capture profile/seed provenance mismatch: {run_id}")
    finished = ("finished: reason=schedule complete, aborted=False, "
                "phases=84/84, quality_failures=0")
    if finished not in log:
        raise ValueError(f"exact 84-phase schedule did not complete: {run_id}")

    approach_ids = re.findall(
        r"phase complete: approach_coupled_(\S+), measured_speed=", log)
    from tools.race_domain_dynamic_coupled_plan import build_dynamic_coupled_plan
    expected_approaches = [row.condition_id
                           for row in build_dynamic_coupled_plan(expected_seed)]
    if approach_ids != expected_approaches:
        raise ValueError(f"randomized condition-order provenance mismatch: {run_id}")

    arrays = _load_npz(archive_path)
    if int(arrays["schema_version"][0]) != 7:
        raise ValueError(f"expected schema-7 per-run export: {run_id}")
    if (len(arrays["run_ids"]) != 1
            or str(arrays["run_ids"][0]) != run_id
            or str(arrays["run_splits"][0]) != split
            or str(arrays["run_families"][0]) != "open_plane"):
        raise ValueError(f"capture archive metadata mismatch: {run_id}")
    if not np.array_equal(arrays["dt_s"],
                          np.full(len(arrays["dt_s"]), SIMULATOR_DT_S,
                                  dtype=arrays["dt_s"].dtype)):
        raise ValueError(f"capture is not on the fixed 25 ms timebase: {run_id}")
    n = len(arrays["frames"])
    bounds = arrays["sequence_bounds"]
    if (n != int(row.get("samples_exported", -1))
            or bounds.shape != (9, 2)
            or int(bounds[0, 0]) != 0
            or int(bounds[-1, 1]) != n
            or np.any(bounds[1:, 0] != bounds[:-1, 1])):
        raise ValueError(f"capture sequence layout mismatch: {run_id}")
    if not np.array_equal(arrays["sequence_reset_index"], np.arange(1, 10)):
        raise ValueError(f"capture sequences do not match reset epochs: {run_id}")
    if float(row.get("sensor_valid_fraction", 0.0)) < 0.999:
        raise ValueError(f"capture sensor/actuator feedback is incomplete: {run_id}")
    if (not arrays["sensor_valid"].any()
            or not np.isfinite(arrays["frames"]).all()
            or not np.isfinite(arrays["simulator_rigid_state"]).all()
            or not np.isfinite(arrays["simulator_linear_acceleration"]).all()):
        raise ValueError(f"capture has unusable plant labels: {run_id}")

    rigid = arrays["simulator_rigid_state"].astype(np.float64, copy=False)
    body_speed = np.hypot(rigid[:, 7], rigid[:, 8])
    speed_3d = np.linalg.norm(rigid[:, 7:10], axis=1)
    if float(max(body_speed.max(), speed_3d.max())) > MAX_BODY_SPEED_MPS:
        raise ValueError(f"capture exceeds the 12 m/s training domain: {run_id}")
    quaternion = rigid[:, 3:7]
    tilt = np.arccos(np.clip(
        1.0 - 2.0 * (quaternion[:, 0] ** 2 + quaternion[:, 1] ** 2),
        -1.0, 1.0))
    if float(tilt.max()) >= MAX_TILT_RAD:
        raise ValueError(f"capture exceeds the configured tilt cutoff: {run_id}")
    feedback = arrays["frames"]
    if (float(feedback[:, 3].min()) > -0.50
            or float(feedback[:, 3].max()) < 0.50
            or float(feedback[:, 4].min()) > 0.01
            or float(feedback[:, 4].max()) < 0.45):
        raise ValueError(f"steering/throttle feedback failed to span commands: {run_id}")

    provenance = {
        "run_id": run_id,
        "split": split,
        "seed": expected_seed,
        "profile": expected_profile,
        "bag": str(bag_path.relative_to(REPO_ROOT)),
        "bag_sha256": _sha256(bag_path),
        "experiment_log": str(log_path.relative_to(REPO_ROOT)),
        "experiment_log_sha256": _sha256(log_path),
        "qc_manifest": str(manifest_path.relative_to(REPO_ROOT)),
        "qc_manifest_sha256": _sha256(manifest_path),
        "qc_archive": str(archive_path.relative_to(REPO_ROOT)),
        "qc_archive_sha256": _sha256(archive_path),
        "samples": n,
        "sequences": len(bounds),
        "reset_epoch_count": 9,
        "quality_failures": [],
        "condition_plan": [
            row.__dict__ for row in build_dynamic_coupled_plan(expected_seed)
        ],
        "max_body_speed_mps": float(body_speed.max()),
        "max_3d_speed_mps": float(speed_3d.max()),
        "max_tilt_deg": float(np.rad2deg(tilt.max())),
    }
    return provenance, arrays


def _check_shared_schema(base: dict[str, np.ndarray],
                         source: dict[str, np.ndarray], run_id: str) -> None:
    for key in ("feature_names", "sensor_feature_names", "attitude_feature_names",
                "predicted_state_names"):
        if not np.array_equal(base[key], source[key]):
            raise ValueError(f"feature schema mismatch for {run_id}: {key}")
    for key in FRAME_KEYS:
        if key.startswith("frame_"):
            continue
        if key not in source:
            raise ValueError(f"capture missing frame array {key}: {run_id}")
        if source[key].shape[1:] != base[key].shape[1:]:
            raise ValueError(f"frame shape mismatch for {run_id}: {key}")


def _append_capture(
        output: dict[str, np.ndarray], source: dict[str, np.ndarray],
        provenance: dict[str, Any], frame_offset: int, run_index: int,
        condition_offset: int) -> tuple[int, int]:
    """Append aligned frame and reset-isolated sequence arrays in order."""
    run_id = provenance["run_id"]
    frame_count = len(source["frames"])
    source_seq_for_frame = _frame_source_sequence_index(
        source["sequence_bounds"], frame_count)
    body_speed = np.hypot(source["simulator_rigid_state"][:, 7],
                          source["simulator_rigid_state"][:, 8]).astype(np.float32)
    for key in FRAME_KEYS:
        if key in ("frame_run_index", "frame_source_index",
                   "frame_source_sequence_index", "frame_domain_speed_mps"):
            continue
        output[key] = np.concatenate((output[key], source[key]), axis=0)
    output["frame_run_index"] = np.concatenate((
        output["frame_run_index"],
        np.full(frame_count, run_index, dtype=np.int32)))
    output["frame_source_index"] = np.concatenate((
        output["frame_source_index"], np.arange(frame_count, dtype=np.int64)))
    output["frame_source_sequence_index"] = np.concatenate((
        output["frame_source_sequence_index"], source_seq_for_frame))
    output["frame_domain_speed_mps"] = np.concatenate((
        output["frame_domain_speed_mps"], body_speed))

    resets = source["sequence_reset_index"].astype(np.int32, copy=False)
    labels = _condition_labels(provenance["seed"], resets)
    sequence_count = len(labels)
    local_bounds = source["sequence_bounds"] + np.int64(frame_offset)
    output["sequence_bounds"] = np.concatenate(
        (output["sequence_bounds"], local_bounds), axis=0)
    output["sequence_run_index"] = np.concatenate((
        output["sequence_run_index"],
        np.full(sequence_count, run_index, dtype=np.int32)))
    output["sequence_labels"] = np.concatenate((
        output["sequence_labels"], np.asarray(labels, dtype="U256")))
    output["sequence_condition_id"] = np.concatenate((
        output["sequence_condition_id"],
        np.arange(condition_offset, condition_offset + sequence_count,
                  dtype=np.int32)))
    output["sequence_reset_index"] = np.concatenate((
        output["sequence_reset_index"], resets))
    output["sequence_replicate_index"] = np.concatenate((
        output["sequence_replicate_index"],
        source["sequence_replicate_index"].astype(np.int32, copy=False)))
    output["sequence_source_condition_id"] = np.concatenate((
        output["sequence_source_condition_id"],
        source["sequence_condition_id"].astype(np.int32, copy=False)))
    output["sequence_source_index"] = np.concatenate((
        output["sequence_source_index"],
        np.arange(sequence_count, dtype=np.int32)))
    output["condition_labels"] = np.concatenate((
        output["condition_labels"], np.asarray(labels, dtype="U256")))
    output["condition_run_index"] = np.concatenate((
        output["condition_run_index"],
        np.full(sequence_count, run_index, dtype=np.int32)))
    return frame_count, sequence_count


def _split_coverage(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    rows_by_split: dict[str, int] = {}
    sequences_by_split: dict[str, int] = {}
    runs_by_split: dict[str, list[str]] = {}
    run_ids = arrays["run_ids"].astype(str)
    run_splits = arrays["run_splits"].astype(str)
    for run_index, (run_id, split) in enumerate(zip(run_ids, run_splits)):
        runs_by_split.setdefault(split, []).append(run_id)
        mask = arrays["frame_run_index"] == run_index
        rows_by_split[split] = rows_by_split.get(split, 0) + int(mask.sum())
        sequence_count = int(np.count_nonzero(
            arrays["sequence_run_index"] == run_index))
        sequences_by_split[split] = (
            sequences_by_split.get(split, 0) + sequence_count)
    return {
        split: {
            "rows": rows_by_split[split],
            "sequences": sequences_by_split[split],
            "runs": len(runs),
            "run_ids": runs,
        }
        for split, runs in sorted(runs_by_split.items())
    }


def build(capture_dirs: tuple[Path, ...] | list[Path],
          output_dir: Path = OUTPUT_DIR,
          base_dataset: Path = BASE_DATASET,
          base_manifest: Path = BASE_MANIFEST,
          baseline_registry: Path = BASELINE_REGISTRY) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output: {output_dir}")
    registry = json.loads(baseline_registry.read_text(encoding="utf-8"))
    frozen_race = registry["historical_baseline"][
        "split_policy_and_pre-handoff_artifacts"]["race_dataset"]
    if (base_dataset.resolve() != (REPO_ROOT / frozen_race["path"]).resolve()
            or _sha256(base_dataset) != frozen_race["sha256"]
            or base_manifest.resolve()
            != (REPO_ROOT / frozen_race["manifest_path"]).resolve()
            or _sha256(base_manifest) != frozen_race["manifest_sha256"]):
        raise ValueError("schema-8 parent differs from the frozen WP0 race dataset")

    base_manifest_obj = json.loads(base_manifest.read_text(encoding="utf-8"))
    base = _load_npz(base_dataset)
    if (int(base["schema_version"][0]) != 8
            or str(base["dataset_role"][0]) != "race_domain"
            or len(base["run_ids"]) != 19
            or len(base["frames"]) != int(frozen_race["rows"])):
        raise ValueError("frozen parent archive has unexpected schema/content")
    if (int(base["sequence_bounds"].shape[0])
            != int(frozen_race["sequence_pieces"])):
        raise ValueError("frozen parent sequence count differs from WP0")
    if len(capture_dirs) != len(EXPECTED_CAPTURES):
        raise ValueError(
            f"exactly {len(EXPECTED_CAPTURES)} registered whole-run captures are required")

    captures: dict[str, tuple[dict[str, Any], dict[str, np.ndarray]]] = {}
    for capture_dir in capture_dirs:
        provenance, source = _validate_capture(capture_dir)
        run_id = provenance["run_id"]
        if run_id in captures:
            raise ValueError(f"duplicate new run: {run_id}")
        if run_id in set(base["run_ids"].astype(str)):
            raise ValueError(f"new run already exists in parent: {run_id}")
        _check_shared_schema(base, source, run_id)
        captures[run_id] = (provenance, source)
    if set(captures) != set(EXPECTED_CAPTURES):
        raise ValueError("capture set differs from the registered whole-run captures")

    output = {key: value.copy() for key, value in base.items()}
    old_frame_count = len(base["frames"])
    old_sequence_count = len(base["sequence_bounds"])
    old_condition_count = len(base["condition_labels"])
    old_run_count = len(base["run_ids"])
    prefix_keys = (*FRAME_KEYS, *SEQUENCE_KEYS, *RUN_KEYS,
                   "condition_labels", "condition_run_index",
                   "training_family_names", "training_family_probabilities")
    prefix_hashes = {key: _array_sha256(base[key]) for key in prefix_keys}

    frame_cursor = old_frame_count
    condition_cursor = old_condition_count
    appended_run_records = []
    capture_profile_by_run = ["legacy_schema8"] * old_run_count
    capture_seed_by_run = [-1] * old_run_count
    for run_id in EXPECTED_CAPTURES:
        provenance, source = captures[run_id]
        run_index = len(output["run_ids"])
        frame_count, _ = _append_capture(
            output, source, provenance, frame_cursor, run_index,
            condition_cursor)
        split, seed, profile = EXPECTED_CAPTURES[run_id]
        output["run_ids"] = np.concatenate((
            output["run_ids"], np.asarray([run_id], dtype=output["run_ids"].dtype)))
        output["run_families"] = np.concatenate((
            output["run_families"], np.asarray(["open_plane"],
                                              dtype=output["run_families"].dtype)))
        output["run_splits"] = np.concatenate((
            output["run_splits"], np.asarray([split], dtype=output["run_splits"].dtype)))
        from tools.vehicle_dynamics_learning.build_race_domain_dataset import (
            _family_for_run,
        )
        training_family = _family_for_run(run_id, "open_plane")
        output["training_families"] = np.concatenate((
            output["training_families"],
            np.asarray([training_family], dtype=output["training_families"].dtype)))
        capture_profile_by_run.append(profile)
        capture_seed_by_run.append(seed)
        appended_run_records.append({
            **provenance,
            "run_family": "open_plane",
            "training_family": training_family,
            "effective_split": split,
        })
        frame_cursor += frame_count
        condition_cursor += len(source["sequence_bounds"])

    output["schema_version"] = np.asarray([9], dtype=np.int32)
    output["source_schema_version"] = np.asarray([7], dtype=np.int32)
    output["parent_view_schema_version"] = np.asarray([8], dtype=np.int32)
    output["dataset_role"] = np.asarray(
        ["replacement_teacher_dataset_v1"], dtype="U40")
    output["run_capture_profiles"] = np.asarray(
        capture_profile_by_run, dtype="U64")
    output["run_capture_seeds"] = np.asarray(capture_seed_by_run,
                                              dtype=np.int64)

    for key in prefix_keys:
        prefix = output[key][:len(base[key])]
        if (prefix.dtype != base[key].dtype
                or prefix.shape != base[key].shape
                or prefix.tobytes() != base[key].tobytes()):
            raise AssertionError(f"frozen schema-8 prefix changed: {key}")
    expected_rows = old_frame_count + sum(
        len(captures[run_id][1]["frames"]) for run_id in EXPECTED_CAPTURES)
    if (len(output["frames"]) != expected_rows
            or len(output["run_ids"]) != old_run_count + len(EXPECTED_CAPTURES)
            or len(output["sequence_bounds"])
                != old_sequence_count + 9 * len(EXPECTED_CAPTURES)
            or not np.allclose(output["dt_s"], SIMULATOR_DT_S,
                               rtol=0.0, atol=1e-7)):
        raise AssertionError("schema-9 row/run/sequence/timebase invariant failed")
    if len(set(output["run_ids"].astype(str))) != len(output["run_ids"]):
        raise AssertionError("schema-9 contains duplicate run IDs")

    split_coverage = _split_coverage(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "openplane_dynamics.npz"
    np.savez_compressed(dataset_path, **output)
    run_records = list(base_manifest_obj.get("runs", [])) + appended_run_records
    manifest = {
        "schema_version": 9,
        "dataset_role": "replacement_teacher_dataset_v1",
        "dataset_path": str(dataset_path.relative_to(REPO_ROOT)),
        "dataset_sha256": _sha256(dataset_path),
        "parent_dataset": {
            "path": str(base_dataset.relative_to(REPO_ROOT)),
            "sha256": _sha256(base_dataset),
            "manifest": str(base_manifest.relative_to(REPO_ROOT)),
            "manifest_sha256": _sha256(base_manifest),
            "schema_version": 8,
            "frozen_registry": str(baseline_registry.relative_to(REPO_ROOT)),
            "frozen_registry_sha256": _sha256(baseline_registry),
        },
        "source_schema_version": 7,
        "append_only": True,
        "legacy_sources_reextracted": False,
        "source_rows_never_reordered": True,
        "old_schema8_prefix": {
            "rows": old_frame_count,
            "sequences": old_sequence_count,
            "runs": old_run_count,
            "arrays_bitwise_prefix_verified": list(prefix_keys),
            "array_sha256": prefix_hashes,
        },
        "new_capture_rows": sum(
            record["samples"] for record in appended_run_records),
        "run_split_preserved_whole_run": True,
        "run_provenance": run_records,
        "quality_gate": {
            "admitted_new_run_count": len(appended_run_records),
            "rejected_new_runs": [],
            "required_profile_phases": 84,
            "scored_dynamic_phases_per_run": 66,
            "unscored_approach_settle_phases_per_run": 18,
            "required_reset_epochs_per_run": 9,
            "max_training_body_speed_mps": MAX_BODY_SPEED_MPS,
            "max_tilt_rad": float(MAX_TILT_RAD),
            "packet_alignment_minimum": 0.999,
            "all_streams_minimum_hz": 38.0,
            "all_streams_p95_gap_max_ms": 35.0,
        },
        "export": {
            "file": dataset_path.name,
            "rows": int(len(output["frames"])),
            "sequences": int(len(output["sequence_bounds"])),
            "runs": int(len(output["run_ids"])),
            "compressed_bytes": dataset_path.stat().st_size,
        },
        "coverage_by_split": split_coverage,
        "training_families": {
            "names": output["training_family_names"].astype(str).tolist(),
            "probabilities": output["training_family_probabilities"].astype(
                float).tolist(),
            "new_run_family": "full_input_openplane",
            "note": "Existing family weights and old family labels are preserved.",
        },
        "blind_evaluation_policy": {
            "final_profile_used": False,
            "historical_test_and_final_test_rows_reordered_or_rescored": False,
            "new_validation_runs_are_whole_run_holdouts": True,
        },
        "frame_source_index_policy": (
            "Old prefix retains schema-8 source indices. Appended frame_source_index "
            "is the row index in that run's per-run QC archive; frame source-sequence "
            "indices are local to that archive and resolve through run_provenance."),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dataset", type=Path, default=BASE_DATASET)
    parser.add_argument("--base-manifest", type=Path, default=BASE_MANIFEST)
    parser.add_argument("--baseline-registry", type=Path,
                        default=BASELINE_REGISTRY)
    parser.add_argument("--capture-qc-dir", action="append", type=Path,
                        default=None,
                        help=("per-run QC output directory; repeat once for each "
                              "registered capture"))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    capture_dirs = args.capture_qc_dir or list(DEFAULT_CAPTURE_DIRS)
    try:
        manifest = build(capture_dirs, args.output_dir, args.base_dataset,
                         args.base_manifest, args.baseline_registry)
    except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
        print(f"replacement teacher dataset build failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(manifest["export"], indent=2))
    print(f"wrote {args.output_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
