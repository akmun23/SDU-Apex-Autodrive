#!/usr/bin/env python3
"""Create immutable <=12 m/s training views and a labeled OOD archive."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "live_runs/derived_dynamics_learning_20260928"
DEFAULT_SOURCE = DATA_ROOT / (
    "plant_teacher_mixed_dataset_full3d_reset_safe_20261001/"
    "openplane_dynamics.npz")
DEFAULT_MANIFEST = DEFAULT_SOURCE.with_name("manifest.json")
DEFAULT_RACE_ROOT = DATA_ROOT / "plant_teacher_race_domain_v1"
DEFAULT_OOD_ROOT = DATA_ROOT / "vehicle_dynamics_ood_extreme_v1"
MAX_SPEED_MPS = 12.0
SIMULATOR_DT_S = 0.025

FAMILY_WEIGHTS = {
    "practice_race": 0.35,
    "full_input_openplane": 0.20,
    "throttle_surface": 0.20,
    "steering_transition_slew": 0.15,
    "braking_down_transition": 0.10,
}

OOD_SPEED_GT_CAP = 1
OOD_POST_CAP_COOLDOWN = 2
OOD_EXTREME_WHEEL_MISMATCH = 4
OOD_HIGH_TILT_TAIL = 8
OOD_UNSUPPORTED_TRAIN_CELL = 16


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _family_for_run(run_id: str, broad_family: str) -> str:
    name = run_id.lower()
    if broad_family == "practice_track":
        return "practice_race"
    if "brak" in name or "decel" in name:
        return "braking_down_transition"
    if "throttle_slew" in name or "throttle_transitions" in name:
        return "steering_transition_slew"
    if ("surface" in name
            or "throttle_5pct_5deg" in name
            or "full_surface" in name):
        return "throttle_surface"
    if any(token in name for token in (
            "transition", "boundary", "fullsteer", "high_angle",
            "steeringonly", "combined_slip")):
        return "steering_transition_slew"
    # Remaining accepted open-plane experiments are continuous/general input
    # excitation rather than a new, unsupported sampling family.
    return "full_input_openplane"


def _practice_active_interval_passed(row: dict[str, Any]) -> bool:
    """Honor only the repository's explicit lap-0-to-12 practice gate."""
    if row.get("quality_gate_scope") != "complete_lap_0_to_12_active_interval":
        return False
    report = row.get("practice_active_interval_validation")
    if not isinstance(report, dict) or report.get("error"):
        return False
    if report.get("lap_count_transitions") != list(range(13)):
        return False
    if report.get("collision_min_max") != [0, 0]:
        return False
    if int(report.get("timing_faults_during_active_interval", -1)) != 0:
        return False
    cadence = report.get("stream_cadence")
    return bool(cadence) and all(
        isinstance(value, dict) and value.get("pass") is True
        for value in cadence.values())


def _quality_run_ids(manifest: dict[str, Any]) -> tuple[set[str], dict[str, str]]:
    accepted_splits = {"train", "validation", "test", "final_test"}
    accepted: set[str] = set()
    rejected: dict[str, str] = {}
    for row in manifest.get("runs", []):
        run_id = str(row["run_id"])
        split = str(row.get("effective_split", ""))
        collisions = row.get("collisions") or []
        active_interval_passed = _practice_active_interval_passed(row)
        race_profile_admitted = bool(
            (row.get("unscored_race_domain_capture_admission") or {}).get(
                "admitted"))
        reasons = []
        if split not in accepted_splits:
            reasons.append(f"non_evaluation_split:{split}")
        if row.get("aborted") and not active_interval_passed:
            reasons.append("aborted")
        if not row.get("clean_stream_and_collision_gate", False):
            reasons.append("stream_or_collision_gate")
        if int(row.get("timing_faults", 0)) != 0 and not active_interval_passed:
            reasons.append("bridge_timing_fault")
        if any(int(value) != 0 for value in collisions):
            reasons.append("collision")
        if row.get("whole_bag_quality_failures") and not active_interval_passed:
            reasons.extend(map(str, row["whole_bag_quality_failures"]))
        if row.get("quality_failures") and not active_interval_passed:
            reasons.extend(map(str, row["quality_failures"]))
        if (int(row.get("unscored_phases", 0)) > 0
                and int(row.get("valid_phases", 0)) == 0
                and not race_profile_admitted):
            reasons.append("unscored_phase_not_admitted_by_capture_protocol")
        if reasons:
            rejected[run_id] = ",".join(sorted(set(reasons)))
        else:
            accepted.add(run_id)
    return accepted, rejected


def _safe_domain_mask(speed: np.ndarray, label_valid: np.ndarray,
                      cooldown_steps: int) -> np.ndarray:
    """Mask in-domain rows and remove a cooldown after each >12 m/s episode."""
    inside = np.isfinite(speed) & (speed <= MAX_SPEED_MPS) & label_valid
    eligible = np.zeros(len(speed), dtype=bool)
    cooldown_remaining = 0
    above_cap = False
    for index, value in enumerate(speed):
        if np.isfinite(value) and value > MAX_SPEED_MPS:
            cooldown_remaining = cooldown_steps
            above_cap = True
            continue
        if not inside[index]:
            continue
        if above_cap:
            above_cap = False
        if cooldown_remaining:
            cooldown_remaining -= 1
            continue
        eligible[index] = True
    return eligible


def _intervals(mask: np.ndarray, *, minimum_length: int = 1
               ) -> list[tuple[int, int]]:
    changes = np.diff(np.r_[False, mask, False].astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return [(int(start), int(end)) for start, end in zip(starts, ends)
            if end - start >= minimum_length]


def _view_arrays(source: dict[str, np.ndarray], pieces: list[tuple[int, int, int]],
                 source_frame_count: int, *, role: str,
                 extra_frame_arrays: dict[str, np.ndarray] | None = None
                 ) -> dict[str, np.ndarray]:
    """Copy complete aligned frame rows and recreate sequence bounds/metadata."""
    sequence_keys = {
        "sequence_bounds", "sequence_run_index", "sequence_labels",
        "sequence_condition_id", "sequence_reset_index",
        "sequence_replicate_index",
        "condition_run_index",
    }
    frame_keys = [
        key for key, values in source.items()
        if key not in sequence_keys and values.ndim >= 1
        and values.shape[0] == source_frame_count
    ]
    if not pieces:
        raise ValueError(f"{role} view contains no qualifying sequence pieces")
    result = {key: value.copy() for key, value in source.items()
              if key not in sequence_keys and key not in frame_keys}
    cursor = 0
    new_bounds = []
    frame_parts: dict[str, list[np.ndarray]] = {key: [] for key in frame_keys}
    frame_source_indices = []
    frame_source_sequence_ids = []
    sequence_source_ids = []
    for sequence_id, start, end in pieces:
        length = end - start
        new_bounds.append((cursor, cursor + length))
        cursor += length
        sequence_source_ids.append(sequence_id)
        frame_source_indices.append(np.arange(start, end, dtype=np.int64))
        frame_source_sequence_ids.append(
            np.full(length, sequence_id, dtype=np.int32))
        for key in frame_keys:
            frame_parts[key].append(source[key][start:end])
    result.update({key: np.concatenate(parts, axis=0)
                   for key, parts in frame_parts.items()})
    for key in ("sequence_run_index", "sequence_labels", "sequence_condition_id",
                "sequence_reset_index", "sequence_replicate_index"):
        if key in source:
            result[key] = source[key][sequence_source_ids].copy()
    if "sequence_condition_id" in result:
        # The schema-7 source can intern identical condition labels across
        # multiple runs. Training metadata requires each condition ID to name
        # one run-local condition, so rebuild the catalog from (run, label)
        # rather than assuming the source has a condition_run_index array.
        source_condition_ids = result["sequence_condition_id"].astype(
            np.int64, copy=False)
        source_condition_labels = source["condition_labels"].astype(str)
        sequence_run_indices = result["sequence_run_index"].astype(
            np.int64, copy=False)
        if (np.any(source_condition_ids < 0)
                or np.any(source_condition_ids >= len(source_condition_labels))):
            raise ValueError("sequence condition IDs exceed condition labels")
        condition_ids: dict[tuple[int, str], int] = {}
        condition_labels: list[str] = []
        condition_run_indices: list[int] = []
        remapped_sequence_conditions = []
        for run_index, source_condition_id in zip(
                sequence_run_indices, source_condition_ids):
            label = str(source_condition_labels[int(source_condition_id)])
            key = (int(run_index), label)
            if key not in condition_ids:
                condition_ids[key] = len(condition_labels)
                condition_labels.append(label)
                condition_run_indices.append(int(run_index))
            remapped_sequence_conditions.append(condition_ids[key])
        result["sequence_source_condition_id"] = source_condition_ids.astype(
            np.int32, copy=True)
        result["sequence_condition_id"] = np.asarray(
            remapped_sequence_conditions, dtype=np.int32)
        result["condition_labels"] = np.asarray(condition_labels, dtype="U256")
        result["condition_run_index"] = np.asarray(
            condition_run_indices, dtype=np.int32)
    result["sequence_bounds"] = np.asarray(new_bounds, dtype=np.int64)
    result["sequence_source_index"] = np.asarray(sequence_source_ids,
                                                 dtype=np.int32)
    result["frame_source_index"] = np.concatenate(frame_source_indices)
    result["frame_source_sequence_index"] = np.concatenate(
        frame_source_sequence_ids)
    if extra_frame_arrays:
        for key, values in extra_frame_arrays.items():
            result[key] = np.concatenate([values[start:end]
                                          for _, start, end in pieces])
    result["schema_version"] = np.asarray([8], dtype=np.int32)
    result["source_schema_version"] = np.asarray([7], dtype=np.int32)
    result["dataset_role"] = np.asarray([role], dtype="U32")
    return result


def _build_race_pieces(source: dict[str, np.ndarray], accepted_runs: set[str],
                       cooldown_steps: int) -> tuple[list[tuple[int, int, int]],
                                                     dict[str, Any]]:
    frames = source["frames"]
    rigid = source["simulator_rigid_state"]
    acceleration = source["simulator_linear_acceleration"]
    bounds = source["sequence_bounds"]
    seq_runs = source["sequence_run_index"]
    run_ids = source["run_ids"].astype(str)
    speed = np.hypot(rigid[:, 7], rigid[:, 8])
    label_valid = (np.isfinite(frames).all(axis=1)
                   & np.isfinite(rigid).all(axis=1)
                   & np.isfinite(acceleration).all(axis=1))
    pieces: list[tuple[int, int, int]] = []
    eligible_rows = 0
    eligible_runs: set[str] = set()
    eligible_conditions: set[int] = set()
    per_split: dict[str, dict[str, int]] = {}
    for sequence_id, (start_raw, end_raw) in enumerate(bounds):
        run_index = int(seq_runs[sequence_id])
        run_id = str(run_ids[run_index])
        if run_id not in accepted_runs:
            continue
        start, end = int(start_raw), int(end_raw)
        local_mask = _safe_domain_mask(
            speed[start:end], label_valid[start:end], cooldown_steps)
        intervals = _intervals(local_mask, minimum_length=2)
        if not intervals:
            continue
        for local_start, local_end in intervals:
            left, right = start + local_start, start + local_end
            pieces.append((sequence_id, left, right))
            eligible_rows += right - left
            eligible_runs.add(run_id)
            eligible_conditions.add(int(source["sequence_condition_id"][sequence_id]))
            split = str(source["run_splits"][run_index])
            row = per_split.setdefault(split, {"rows": 0, "sequences": 0,
                                                "runs": 0})
            row["rows"] += right - left
            row["sequences"] += 1
    for split, summary in per_split.items():
        summary["runs"] = len({
            str(run_ids[int(seq_runs[seq_id])])
            for seq_id, _, _ in pieces
            if str(source["run_splits"][int(seq_runs[seq_id])]) == split
        })
    return pieces, {
        "rows": eligible_rows,
        "sequence_pieces": len(pieces),
        "independent_runs": len(eligible_runs),
        "conditions": len(eligible_conditions),
        "counts_by_split": per_split,
    }


def _support_cell(speed: np.ndarray, steering: np.ndarray,
                  throttle: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    speed_edges = np.asarray([0, 3, 5, 7, 9, 10, 11, 12.000001])
    steer_edges = np.asarray([0, 0.10, 0.20, 0.30, 0.40, 0.524001])
    throttle_edges = np.asarray([0, 0.01, 0.25, 0.60, 1.000001])
    valid = (np.isfinite(speed) & (speed >= 0) & (speed <= MAX_SPEED_MPS)
             & np.isfinite(steering) & np.isfinite(throttle))
    cells = np.full((len(speed), 3), -1, dtype=np.int16)
    cells[valid, 0] = np.searchsorted(speed_edges, speed[valid], side="right") - 1
    cells[valid, 1] = np.searchsorted(
        steer_edges, np.abs(steering[valid]), side="right") - 1
    cells[valid, 2] = np.searchsorted(
        throttle_edges, throttle[valid], side="right") - 1
    return cells, valid


def build(source_path: Path, manifest_path: Path, race_root: Path,
          ood_root: Path, cooldown_seconds: tuple[float, ...] = (1.0, 2.0)
          ) -> dict[str, Any]:
    if race_root.exists() or ood_root.exists():
        raise FileExistsError("refusing to overwrite a derived dataset view")
    if not cooldown_seconds or any(value < 0 for value in cooldown_seconds):
        raise ValueError("cooldown durations must be nonnegative")
    source_path = source_path.resolve()
    manifest_path = manifest_path.resolve()
    archive = np.load(source_path, allow_pickle=False)
    source = {key: archive[key] for key in archive.files}
    if int(source["schema_version"][0]) != 7:
        raise ValueError("race-domain view requires the audited schema-7 source")
    if not np.allclose(source["dt_s"], SIMULATOR_DT_S,
                       rtol=0.0, atol=1e-7):
        raise ValueError("source timebase is not exact 25 ms")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    accepted_runs, rejected_runs = _quality_run_ids(manifest)
    source_run_ids = set(source["run_ids"].astype(str))
    manifest_run_ids = {str(row["run_id"]) for row in manifest.get("runs", [])}
    if source_run_ids != manifest_run_ids:
        raise ValueError("source archive and manifest run IDs disagree")
    manifest_splits = {str(row["run_id"]): str(row.get("effective_split", ""))
                       for row in manifest.get("runs", [])}
    for run_id, split in zip(source["run_ids"].astype(str),
                             source["run_splits"].astype(str)):
        if manifest_splits[str(run_id)] != str(split):
            raise ValueError(f"split mismatch for source run {run_id}")

    race_root.mkdir(parents=True)
    views: dict[str, Any] = {}
    speed = np.hypot(source["simulator_rigid_state"][:, 7],
                     source["simulator_rigid_state"][:, 8])
    for seconds in cooldown_seconds:
        steps = int(round(seconds / SIMULATOR_DT_S))
        pieces, summary = _build_race_pieces(source, accepted_runs, steps)
        view_arrays = _view_arrays(
            source, pieces, len(speed), role="race_domain")
        view_arrays["domain_speed_cap_mps"] = np.asarray(
            [MAX_SPEED_MPS], dtype=np.float32)
        view_arrays["domain_cooldown_steps"] = np.asarray([steps], dtype=np.int32)
        training_families = np.asarray([
            _family_for_run(str(run_id), str(family))
            for run_id, family in zip(source["run_ids"], source["run_families"])
        ], dtype="U40")
        view_arrays["training_families"] = training_families
        view_arrays["training_family_names"] = np.asarray(
            list(FAMILY_WEIGHTS), dtype="U40")
        view_arrays["training_family_probabilities"] = np.asarray(
            list(FAMILY_WEIGHTS.values()), dtype=np.float32)
        view_arrays["frame_domain_speed_mps"] = np.concatenate(
            [speed[start:end].astype(np.float32)
             for _, start, end in pieces])
        label = f"cooldown_{seconds:g}s"
        output_dir = race_root / label
        output_dir.mkdir()
        dataset_out = output_dir / "openplane_dynamics.npz"
        np.savez_compressed(dataset_out, **view_arrays)
        view_manifest = {
            "schema_version": 1,
            "dataset_role": "race_domain_training_and_evaluation",
            "dataset_path": str(dataset_out.resolve()),
            "dataset_sha256": _sha256(dataset_out),
            "source_dataset": str(source_path),
            "source_dataset_sha256": _sha256(source_path),
            "source_manifest_sha256": _sha256(manifest_path),
            "source_schema_version": 7,
            "view_schema_version": 8,
            "speed_cap_mps": MAX_SPEED_MPS,
            "cooldown_seconds_after_gt12_excursion": seconds,
            "cooldown_steps": steps,
            "window_rule": "every context and every target sample must belong to one emitted race-domain sequence",
            "source_sequence_packet_and_reset_boundaries_preserved": True,
            "source_rows_never_reordered": True,
            "quality_gate": {
                "accepted_splits": ["train", "validation", "test", "final_test"],
                "requires_clean_stream_collision_gate": True,
                "requires_zero_collisions": True,
                "requires_not_aborted": True,
                "requires_zero_bridge_timing_faults": True,
                "requires_empty_whole_bag_quality_failures": True,
                "accepted_run_count": len(accepted_runs),
                "rejected_run_count": len(rejected_runs),
                "practice_active_interval_run_ids": sorted(
                    str(row["run_id"]) for row in manifest.get("runs", [])
                    if _practice_active_interval_passed(row)),
                "rejected_runs_and_reasons": rejected_runs,
            },
            "coverage": summary,
            "run_level_split_preserved": True,
            "sampling_family_weights_initial": FAMILY_WEIGHTS,
            "note": ("Rows above 12 m/s and all post-excursion cooldown rows are absent. "
                     "Race-relevant slip at <=12 m/s is retained. Practice runs may "
                     "pass only through the explicit complete lap-0-to-12 active-interval gate."),
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(view_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        views[label] = {"path": str(dataset_out), **summary}

    # An empirical tail threshold labels extreme wheel/body mismatch and tilt.
    # These labels annotate diagnostics; they do not remove <=12 m/s slip from
    # the race-domain teacher dataset.
    rigid = source["simulator_rigid_state"]
    frames = source["frames"]
    attitude = source["imu_attitude_frames"]
    label_valid = (np.isfinite(rigid).all(axis=1)
                   & np.isfinite(frames).all(axis=1)
                   & np.isfinite(source["simulator_linear_acceleration"]).all(axis=1))
    frame_runs = source["frame_run_index"].astype(np.int32)
    run_ids = source["run_ids"].astype(str)
    clean_frame = np.isin(run_ids[frame_runs], np.asarray(sorted(accepted_runs)))
    in_domain = np.isfinite(speed) & (speed <= MAX_SPEED_MPS) & label_valid & clean_frame
    mismatch = np.abs(0.5 * (frames[:, 5] + frames[:, 6]) - frames[:, 0])
    mismatch_threshold = float(np.quantile(mismatch[in_domain], 0.995))
    tilt = np.max(np.abs(attitude[:, :2]), axis=1)
    tilt_usable = in_domain & np.isfinite(tilt)
    tilt_threshold = float(np.quantile(tilt[tilt_usable], 0.995))

    # Support is defined by clean training observations in speed × measured
    # steering-magnitude × throttle bins; it is a coarse diagnostic mask.
    cells, valid_cells = _support_cell(speed, frames[:, 3], frames[:, 8])
    train_rows = in_domain & valid_cells & (source["run_splits"][frame_runs] == "train")
    observed_cells = {tuple(map(int, row)) for row in cells[train_rows]}
    supported = np.zeros(len(speed), dtype=bool)
    for index in np.flatnonzero(valid_cells):
        supported[index] = tuple(map(int, cells[index])) in observed_cells

    ood_reason = np.zeros(len(speed), dtype=np.uint8)
    ood_reason[clean_frame & np.isfinite(speed) & (speed > MAX_SPEED_MPS)] |= OOD_SPEED_GT_CAP
    for sequence_start, sequence_end in source["sequence_bounds"]:
        start, end = int(sequence_start), int(sequence_end)
        local_speed = speed[start:end]
        cooldown_mask = np.zeros(end - start, dtype=bool)
        above = False
        remaining = 0
        cooldown_steps = int(round(2.0 / SIMULATOR_DT_S))
        for local_index, value in enumerate(local_speed):
            if np.isfinite(value) and value > MAX_SPEED_MPS:
                above, remaining = True, cooldown_steps
                continue
            if above and np.isfinite(value) and value <= MAX_SPEED_MPS:
                above = False
            if remaining and np.isfinite(value) and value <= MAX_SPEED_MPS:
                cooldown_mask[local_index] = True
                remaining -= 1
        local_reasons = ood_reason[start:end]
        local_reasons[clean_frame[start:end] & cooldown_mask] |= OOD_POST_CAP_COOLDOWN
    ood_reason[clean_frame & in_domain & (mismatch >= mismatch_threshold)] |= OOD_EXTREME_WHEEL_MISMATCH
    ood_reason[clean_frame & tilt_usable & (tilt >= tilt_threshold)] |= OOD_HIGH_TILT_TAIL
    ood_reason[clean_frame & in_domain & valid_cells & ~supported] |= OOD_UNSUPPORTED_TRAIN_CELL

    ood_pieces: list[tuple[int, int, int]] = []
    for sequence_id, (start_raw, end_raw) in enumerate(source["sequence_bounds"]):
        start, end = int(start_raw), int(end_raw)
        run_id = str(run_ids[int(source["sequence_run_index"][sequence_id])])
        if run_id not in accepted_runs:
            continue
        for local_start, local_end in _intervals(ood_reason[start:end] != 0):
            ood_pieces.append((sequence_id, start + local_start, start + local_end))
    ood_arrays = _view_arrays(
        source, ood_pieces, len(speed), role="ood_diagnostics",
        extra_frame_arrays={
            "frame_ood_reason": ood_reason,
            "frame_domain_speed_mps": np.nan_to_num(speed, nan=-1.0).astype(np.float32),
            "frame_wheel_body_mismatch_mps": np.nan_to_num(
                mismatch, nan=-1.0).astype(np.float32),
            "frame_training_cell_supported": supported,
        })
    ood_root.mkdir(parents=True)
    ood_dataset = ood_root / "openplane_ood_diagnostics.npz"
    np.savez_compressed(ood_dataset, **ood_arrays)
    reason_counts = {
        "speed_gt_12mps": int(np.count_nonzero(ood_reason & OOD_SPEED_GT_CAP)),
        "post_12mps_excursion_2s_cooldown": int(np.count_nonzero(
            ood_reason & OOD_POST_CAP_COOLDOWN)),
        "empirical_p99_5_wheel_body_mismatch": int(np.count_nonzero(
            ood_reason & OOD_EXTREME_WHEEL_MISMATCH)),
        "empirical_p99_5_tilt_tail_not_verified_rollover": int(np.count_nonzero(
            ood_reason & OOD_HIGH_TILT_TAIL)),
        "no_clean_train_support_in_speed_steer_throttle_cell": int(np.count_nonzero(
            ood_reason & OOD_UNSUPPORTED_TRAIN_CELL)),
    }
    ood_manifest = {
        "schema_version": 1,
        "dataset_role": "ood_and_support_diagnostics_only",
        "dataset_path": str(ood_dataset.resolve()),
        "dataset_sha256": _sha256(ood_dataset),
        "source_dataset": str(source_path),
        "source_dataset_sha256": _sha256(source_path),
        "speed_cap_mps": MAX_SPEED_MPS,
        "wheel_body_mismatch_proxy": "abs(mean(rear wheel surface speeds) - rear-axle u)",
        "wheel_body_mismatch_threshold_mps": mismatch_threshold,
        "tilt_proxy": "max(abs(IMU roll), abs(IMU pitch))",
        "tilt_threshold_rad": tilt_threshold,
        "tilt_threshold_method": "99.5th percentile of clean labeled <=12 m/s samples; diagnostic tail, not a physical rollover threshold",
        "support_cell_edges": {
            "speed_mps": [0, 3, 5, 7, 9, 10, 11, 12],
            "absolute_steering_rad": [0, 0.10, 0.20, 0.30, 0.40, 0.524],
            "throttle_command": [0, 0.01, 0.25, 0.60, 1.0],
        },
        "ood_reason_bit_values": {
            "speed_gt_12mps": OOD_SPEED_GT_CAP,
            "post_12mps_excursion_2s_cooldown": OOD_POST_CAP_COOLDOWN,
            "empirical_p99_5_wheel_body_mismatch": OOD_EXTREME_WHEEL_MISMATCH,
            "empirical_p99_5_tilt_tail": OOD_HIGH_TILT_TAIL,
            "no_clean_train_support_in_cell": OOD_UNSUPPORTED_TRAIN_CELL,
        },
        "reason_sample_counts_overlap": reason_counts,
        "sequence_pieces": len(ood_pieces),
        "ood_frame_count": int(sum(end - start for _, start, end in ood_pieces)),
        "quality_gate_rejected_runs": rejected_runs,
        "primary_training_loss_use": False,
        "note": "High-slip <=12 m/s rows remain in the race-domain view; this archive marks empirical tail examples for separate support diagnostics.",
    }
    (ood_root / "manifest.json").write_text(
        json.dumps(ood_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return {"race_views": views, "ood": ood_manifest,
            "rejected_run_count": len(rejected_runs)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--race-root", type=Path, default=DEFAULT_RACE_ROOT)
    parser.add_argument("--ood-root", type=Path, default=DEFAULT_OOD_ROOT)
    parser.add_argument("--cooldown-seconds", type=float, nargs="+",
                        default=[1.0, 2.0])
    args = parser.parse_args()
    result = build(args.source, args.source_manifest, args.race_root,
                   args.ood_root, tuple(args.cooldown_seconds))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
