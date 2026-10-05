#!/usr/bin/env python3
"""Fit and audit a run-clustered empirical racing envelope from captured data.

Only whole-run ``train`` rows fit the envelope. Whole-run ``validation`` rows
and optional fresh practice-run reports are scored separately. ``test`` and
``final_test`` partitions in the source archive are never summarized or used.
The output is an offline static artifact; this script does not change MPC,
odometry, simulator physics, or the raceline planner.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET = Path(
    "live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/"
    "replacement_offline_sim_raceline_20261002/"
    "replacement_teacher_dataset_v2_20261004/openplane_dynamics.npz"
)
MANIFEST = DATASET.with_name("manifest.json")
OUTPUT = Path("live_runs/racing_envelope_20261005")
DT_S = 0.025  # The capture contract is fixed 40 Hz; receipt jitter is not dt.
MAX_SPEED_MPS = 12.0
MAX_TILT_RAD = math.radians(8.0)
MIN_GROUP_SAMPLES = 20
MIN_TRANSIENT_SAMPLES = 5  # Five 40 Hz samples span the 100 ms derivative window.
MIN_TRAIN_RUNS = 3
MIN_TRAIN_CONDITIONS = 3
MIN_VALIDATION_RUNS = 2
PLANNING_CAPABILITY_FRACTION = 0.90
SPEED_EDGES = (0.0, 2.0, 4.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0)
STEER_EDGES = (0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.5236)
LONG_DEMAND_THRESHOLD_MPS2 = 0.50
LATERAL_STRAIGHT_THRESHOLD_MPS2 = 1.0
LATERAL_DEMAND_THRESHOLD_MPS2 = 0.50
SUSTAINED_SAMPLES = 10  # 250 ms of contiguous 40 Hz support.
# The previous fit hit p=4 in every moving band; expand the training-only
# search to determine whether that was a real box-like frontier or a clip.
POWER_EXPONENTS = np.arange(1.0, 10.01, 0.10)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def q(values: Iterable[float], probability: float) -> float | None:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.quantile(array, probability)) if array.size else None


def stat_row(values: np.ndarray) -> dict[str, float | int | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return {
        "samples": int(finite.size),
        "p50": q(finite, 0.50),
        "p90": q(finite, 0.90),
        "p95": q(finite, 0.95),
        "p97_5": q(finite, 0.975),
        "max": float(finite.max()) if finite.size else None,
    }


def run_bootstrap_median(values: dict[str, float], seed: int = 20261005) -> list[float] | None:
    """Percentile CI resampling independent runs, never individual frames."""
    if len(values) < MIN_TRAIN_RUNS:
        return None
    observed = np.asarray(list(values.values()), dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(observed), size=(4000, len(observed)))
    medians = np.median(observed[indices], axis=1)
    return [float(np.quantile(medians, 0.05)), float(np.quantile(medians, 0.95))]


def speed_bin(speed: np.ndarray) -> np.ndarray:
    edges = np.asarray(SPEED_EDGES, dtype=np.float64)
    result = np.searchsorted(edges, speed, side="right") - 1
    result[speed >= edges[-1]] = len(edges) - 2
    return result.astype(np.int16)


def bin_label(index: int) -> str:
    return f"{SPEED_EDGES[index]:g}-{SPEED_EDGES[index + 1]:g}"


def unwrap_quaternion_tilt(quaternion: np.ndarray) -> np.ndarray:
    # Quaternion columns are x,y,z,w in the simulator rigid-state archive.
    x, y, z, w = (quaternion[:, i] for i in range(4))
    sin_roll = 2.0 * (w * x + y * z)
    cos_roll = 1.0 - 2.0 * (x * x + y * y)
    sin_pitch = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    roll = np.arctan2(sin_roll, cos_roll)
    pitch = np.arcsin(sin_pitch)
    return np.maximum(np.abs(roll), np.abs(pitch))


def consecutive_mask(sequence: np.ndarray, packet_sequence: np.ndarray) -> np.ndarray:
    """Require an intact local ±2-sample stencil inside one reset epoch."""
    n = len(sequence)
    valid = np.zeros(n, dtype=bool)
    if n < 5:
        return valid
    local = np.arange(2, n - 2)
    intact = np.ones(local.size, dtype=bool)
    for offset in (-2, -1, 0, 1):
        intact &= packet_sequence[local + offset + 1] - packet_sequence[local + offset] == 1
    valid[local] = intact
    return valid


def load_dataset(dataset: Path, manifest_path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = str(manifest.get("dataset_sha256", ""))
    actual = sha256(dataset)
    if not expected or actual != expected:
        raise ValueError("dataset hash does not match its manifest")
    quality = manifest.get("quality_gate") or {}
    if quality.get("rejected_new_runs"):
        raise ValueError("source manifest records rejected appended runs")
    if not manifest.get("blind_evaluation_policy", {}).get("new_validation_runs_are_whole_run_holdouts"):
        raise ValueError("source manifest does not guarantee whole-run validation")
    required = (
        "frames", "simulator_rigid_state", "simulator_linear_acceleration",
        "sensor_valid", "packet_sequence", "sequence_bounds",
        "sequence_run_index", "sequence_labels", "run_ids", "run_splits",
        "frame_run_index", "frame_reset_index", "sequence_reset_index",
    )
    with np.load(dataset, allow_pickle=False) as archive:
        missing = [key for key in required if key not in archive.files]
        if missing:
            raise ValueError("dataset missing required arrays: " + ", ".join(missing))
        arrays = {key: archive[key] for key in required}
        if "dt_s" in archive.files:
            arrays["dt_s"] = archive["dt_s"]
    n = len(arrays["frames"])
    if arrays["frames"].shape != (n, 9) or arrays["simulator_rigid_state"].shape != (n, 13):
        raise ValueError("source motion-state array shapes are not schema-compatible")
    if (arrays["frame_run_index"].shape != (n,)
            or arrays["frame_reset_index"].shape != (n,)
            or arrays["sensor_valid"].shape != (n,)):
        raise ValueError("source per-frame metadata shape mismatch")
    if len(arrays["sequence_reset_index"]) != len(arrays["sequence_bounds"]):
        raise ValueError("sequence and reset-epoch metadata differ in length")
    if len(arrays["run_splits"]) != len(arrays["run_ids"]):
        raise ValueError("run IDs and split labels differ in length")
    if arrays.get("dt_s") is not None and not np.allclose(arrays["dt_s"], DT_S, rtol=0.0, atol=1e-7):
        raise ValueError("source timebase is not the required fixed 25 ms interval")
    # The single combined archive also contains sealed test partitions. They
    # are intentionally not accessed beyond this exact split index mapping.
    allowed_run_indices = {
        split: np.flatnonzero(arrays["run_splits"].astype(str) == split)
        for split in ("train", "validation")
    }
    if not allowed_run_indices["train"].size or not allowed_run_indices["validation"].size:
        raise ValueError("both train and whole-run validation captures are required")
    arrays["allowed_run_indices"] = allowed_run_indices
    manifest["computed_dataset_sha256"] = actual
    return arrays, manifest


def build_motion_samples(
    arrays: dict[str, np.ndarray],
) -> tuple[dict[str, dict[int, dict[str, np.ndarray]]], dict[str, Any]]:
    frames = arrays["frames"].astype(np.float64, copy=False)
    rigid = arrays["simulator_rigid_state"].astype(np.float64, copy=False)
    sim_acceleration = arrays["simulator_linear_acceleration"].astype(np.float64, copy=False)
    sensor_valid = arrays["sensor_valid"].astype(bool, copy=False)
    packet = arrays["packet_sequence"].astype(np.int64, copy=False)
    run_ids = arrays["run_ids"].astype(str)
    run_splits = arrays["run_splits"].astype(str)
    bounds = arrays["sequence_bounds"].astype(np.int64, copy=False)
    sequence_runs = arrays["sequence_run_index"].astype(np.int64, copy=False)
    sequence_labels = arrays["sequence_labels"].astype(str)
    sequence_resets = arrays["sequence_reset_index"].astype(np.int64, copy=False)
    run_allowed = set(map(int, np.concatenate(list(arrays["allowed_run_indices"].values()))))

    fields = ("speed", "steer", "throttle", "u", "v", "r", "ax", "ay", "ay_exact", "curvature", "run", "seq", "index")
    pieces: dict[str, dict[int, dict[str, list[np.ndarray]]]] = {
        split: {i: {name: [] for name in fields} for i in range(len(SPEED_EDGES) - 1)}
        for split in ("train", "validation")
    }
    qc = {
        split: {"source_sequences": 0, "source_samples": 0, "eligible_samples": 0,
                "rejected_missing_or_nonfinite": 0, "rejected_sensor_invalid": 0,
                "rejected_speed_over_12": 0, "rejected_tilt_over_8deg": 0,
                "rejected_packet_gap": 0, "vertical_speed_p95_mps": None,
                "run_level_vertical_speed_chunks": defaultdict(list),
                "run_ids": sorted(run_ids[idx] for idx in arrays["allowed_run_indices"][split])}
        for split in ("train", "validation")
    }

    for sequence_id, (start, end) in enumerate(bounds):
        start, end = int(start), int(end)
        run_index = int(sequence_runs[sequence_id])
        if run_index not in run_allowed:
            continue
        split = str(run_splits[run_index])
        if split not in pieces:
            continue
        if start < 0 or end > len(frames) or end - start < 5:
            raise ValueError("sequence bounds contain a short/out-of-range sequence")
        if np.any(arrays["frame_run_index"][start:end] != run_index):
            raise ValueError("sequence crosses a run boundary")
        reset_ids = np.unique(arrays["frame_reset_index"][start:end])
        if len(reset_ids) != 1 or int(reset_ids[0]) != int(sequence_resets[sequence_id]):
            raise ValueError("sequence crosses or mislabels a reset epoch")
        qc[split]["source_sequences"] += 1
        qc[split]["source_samples"] += end - start
        local_frames = frames[start:end]
        local_rigid = rigid[start:end]
        local_acceleration = sim_acceleration[start:end]
        local_packet = packet[start:end]
        n = len(local_frames)
        u, v, yaw_rate = local_frames[:, 0], local_frames[:, 1], local_frames[:, 2]
        steering, throttle = local_frames[:, 3], local_frames[:, 4]
        # 100 ms central finite difference on the explicitly fixed 25 ms grid.
        du = (u[4:] - u[:-4]) / (4.0 * DT_S)
        dv = (v[4:] - v[:-4]) / (4.0 * DT_S)
        ax = du - v[2:-2] * yaw_rate[2:-2]
        ay_exact = dv + u[2:-2] * yaw_rate[2:-2]
        sample_u, sample_v, sample_r = u[2:-2], v[2:-2], yaw_rate[2:-2]
        sample_steer, sample_throttle = steering[2:-2], throttle[2:-2]
        rigid_center = local_rigid[2:-2]
        speed = np.abs(sample_u)
        ay = sample_u * sample_r
        curvature = sample_r / np.maximum(speed, 0.25)
        quaternion = rigid_center[:, 3:7]
        tilt = unwrap_quaternion_tilt(quaternion)
        vertical_speed = np.abs(rigid_center[:, 9])
        sensor_ok = sensor_valid[start + 2:start + n - 2]
        packet_ok = consecutive_mask(np.zeros(n, dtype=np.int8), local_packet)[2:-2]
        finite = (
            np.isfinite(local_frames[2:-2]).all(axis=1)
            & np.isfinite(rigid_center).all(axis=1)
            & np.isfinite(local_acceleration[2:-2]).all(axis=1)
            & np.isfinite(ax) & np.isfinite(ay) & np.isfinite(ay_exact)
            & np.isfinite(curvature)
        )
        speed_ok = speed <= MAX_SPEED_MPS
        tilt_ok = tilt < MAX_TILT_RAD
        accepted = finite & sensor_ok & speed_ok & tilt_ok & packet_ok
        if accepted.any():
            qc[split]["run_level_vertical_speed_chunks"][run_ids[run_index]].append(
                vertical_speed[accepted])
        qc[split]["rejected_missing_or_nonfinite"] += int((~finite).sum())
        qc[split]["rejected_sensor_invalid"] += int((finite & ~sensor_ok).sum())
        qc[split]["rejected_speed_over_12"] += int((finite & sensor_ok & ~speed_ok).sum())
        qc[split]["rejected_tilt_over_8deg"] += int((finite & sensor_ok & speed_ok & ~tilt_ok).sum())
        qc[split]["rejected_packet_gap"] += int((finite & sensor_ok & speed_ok & tilt_ok & ~packet_ok).sum())
        if not accepted.any():
            continue
        binned = speed_bin(speed)
        seq_array = np.full(speed.shape, sequence_id, dtype=np.int32)
        run_array = np.full(speed.shape, run_index, dtype=np.int32)
        qc[split]["eligible_samples"] += int(accepted.sum())
        for bin_index in range(len(SPEED_EDGES) - 1):
            selected = accepted & (binned == bin_index)
            if not selected.any():
                continue
            bucket = pieces[split][bin_index]
            values = {
                "speed": speed, "steer": sample_steer, "throttle": sample_throttle,
                "u": sample_u, "v": sample_v, "r": sample_r, "ax": ax,
                "ay": ay, "ay_exact": ay_exact, "curvature": curvature,
                "run": run_array, "seq": seq_array,
                "index": np.arange(n, dtype=np.int32)[2:-2],
            }
            for name in fields:
                bucket[name].append(values[name][selected])

    combined: dict[str, dict[int, dict[str, np.ndarray]]] = {
        split: {} for split in pieces
    }
    for split, bins in pieces.items():
        for bin_index, fields_by_name in bins.items():
            combined[split][bin_index] = {
                name: np.concatenate(values) if values else np.empty(0, dtype=np.float64)
                for name, values in fields_by_name.items()
            }
        run_vertical_stats = {}
        all_vertical = []
        for run_id, chunks in qc[split]["run_level_vertical_speed_chunks"].items():
            values = np.concatenate(chunks)
            all_vertical.append(values)
            run_vertical_stats[run_id] = {
                "samples": int(values.size),
                "p95_mps": float(np.quantile(values, 0.95)),
                "p99_mps": float(np.quantile(values, 0.99)),
                "max_mps": float(values.max()),
            }
        vertical = np.concatenate(all_vertical) if all_vertical else np.empty(0)
        qc[split]["vertical_speed_stats"] = {
            "samples": int(vertical.size),
            "p95_mps": float(np.quantile(vertical, 0.95)) if vertical.size else None,
            "p99_mps": float(np.quantile(vertical, 0.99)) if vertical.size else None,
            "max_mps": float(vertical.max()) if vertical.size else None,
            "by_run": run_vertical_stats,
        }
        del qc[split]["run_level_vertical_speed_chunks"]
        del qc[split]["vertical_speed_p95_mps"]
    return combined, qc


def support_counts(data: dict[str, np.ndarray], mask: np.ndarray) -> tuple[int, int]:
    return (len(np.unique(data["run"][mask])), len(np.unique(data["seq"][mask])))


def sequence_quantiles(
    data: dict[str, np.ndarray], mask: np.ndarray, values: np.ndarray,
    *, minimum_samples: int = MIN_GROUP_SAMPLES, probability: float = 0.90,
) -> tuple[dict[str, float], dict[str, int]]:
    by_run: dict[str, list[float]] = defaultdict(list)
    by_sequence: dict[int, float] = {}
    selected = np.flatnonzero(mask & np.isfinite(values))
    selected_seq = data["seq"][selected]
    boundaries = np.flatnonzero(np.diff(selected_seq) != 0) + 1
    for group in np.split(selected, boundaries):
        if not group.size:
            continue
        local_values = values[group]
        local_values = local_values[np.isfinite(local_values)]
        if local_values.size < minimum_samples:
            continue
        sequence_id = int(data["seq"][group[0]])
        run_idx = int(data["run"][group[0]])
        # run names are materialized by caller on arrays after the numeric pass.
        run_key = str(run_idx)
        value = float(np.quantile(local_values, probability))
        by_sequence[int(sequence_id)] = value
        by_run[run_key].append(value)
    # A capability envelope asks what each independent run repeatedly reached
    # in at least one condition. Each condition is summarized by p90 above;
    # take its largest supported condition p90, then aggregate runs below.
    run_cap = {run: max(values) for run, values in by_run.items() if values}
    return run_cap, {"conditions": len(by_sequence), "run_count": len(run_cap)}


def inner_capability(
    data: dict[str, np.ndarray], mask: np.ndarray, values: np.ndarray,
    *, minimum_samples: int = MIN_GROUP_SAMPLES,
) -> dict[str, Any]:
    run_cap, support = sequence_quantiles(data, mask, values, minimum_samples=minimum_samples)
    run_values = list(run_cap.values())
    enough = support["run_count"] >= MIN_TRAIN_RUNS and support["conditions"] >= MIN_TRAIN_CONDITIONS
    interval = run_bootstrap_median(run_cap)
    return {
        "supported": enough,
        "inner_capability": (
            PLANNING_CAPABILITY_FRACTION * float(np.median(run_values))
            if enough else None
        ),
        "run_capabilities": run_cap,
        "support": support,
        "run_bootstrap_median_ci90": interval,
    }


def lateral_and_longitudinal_tables(
    motion: dict[str, dict[int, dict[str, np.ndarray]]],
    *, minimum_samples: int = MIN_GROUP_SAMPLES,
    minimum_transient_samples: int = MIN_TRANSIENT_SAMPLES,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[tuple[str, int, str, str], dict[str, Any]]]:
    lateral_rows: list[dict[str, Any]] = []
    longitudinal_rows: list[dict[str, Any]] = []
    caps: dict[tuple[str, int, str, str], dict[str, Any]] = {}
    for split, bins in motion.items():
        for bin_index, data in bins.items():
            speed = data["speed"]
            ax = data["ax"]
            ay = data["ay"]
            base = np.isfinite(speed) & np.isfinite(ax) & np.isfinite(ay)
            demand_masks = {
                "low_longitudinal": np.abs(ax) <= LONG_DEMAND_THRESHOLD_MPS2,
                "accelerating": ax > LONG_DEMAND_THRESHOLD_MPS2,
                "braking": ax < -LONG_DEMAND_THRESHOLD_MPS2,
            }
            for demand, demand_mask in demand_masks.items():
                required_samples = (
                    minimum_samples if demand == "low_longitudinal"
                    else minimum_transient_samples
                )
                for side, side_mask in (("left", ay > 0.0), ("right", ay < 0.0)):
                    mask = base & demand_mask & side_mask
                    n_runs, n_conditions = support_counts(data, mask)
                    values = np.abs(ay[mask])
                    cap = inner_capability(data, mask, np.abs(ay), minimum_samples=required_samples) if split == "train" else {
                        "supported": False, "inner_capability": None,
                        "run_capabilities": {}, "support": {"conditions": 0, "run_count": 0},
                        "run_bootstrap_median_ci90": None,
                    }
                    lateral_rows.append({
                        "split": split, "speed_bin": bin_label(bin_index),
                        "longitudinal_demand": demand, "turn_direction": side,
                        **stat_row(values), "independent_runs": n_runs,
                        "conditions": n_conditions,
                        "capability_supported_runs": cap["support"]["run_count"],
                        "capability_supported_conditions": cap["support"]["conditions"],
                        "conservative_inner_capability_mps2": cap["inner_capability"],
                        "run_capabilities_mps2": cap["run_capabilities"],
                        "planner_support_gate": cap["supported"],
                        "run_cluster_median_ci90_mps2": cap["run_bootstrap_median_ci90"],
                    })
                    caps[(split, bin_index, demand, side)] = cap

            # Longitudinal limits measured only near straight-line operation so
            # lateral tire demand does not masquerade as motor/brake capacity.
            straight = base & (np.abs(ay) <= LATERAL_STRAIGHT_THRESHOLD_MPS2)
            for demand, demand_mask, values_all in (
                ("acceleration", ax > LONG_DEMAND_THRESHOLD_MPS2, np.maximum(ax, 0.0)),
                ("braking", ax < -LONG_DEMAND_THRESHOLD_MPS2, np.maximum(-ax, 0.0)),
            ):
                mask = straight & demand_mask
                values = values_all[mask]
                cap = inner_capability(data, mask, values_all, minimum_samples=minimum_transient_samples) if split == "train" else {
                    "supported": False, "inner_capability": None,
                    "run_capabilities": {}, "support": {"conditions": 0, "run_count": 0},
                    "run_bootstrap_median_ci90": None,
                }
                # Avoid an invalid upper quantile of all zeros for irrelevant
                # runs by the sign-specific mask above.
                longitudinal_rows.append({
                    "split": split, "speed_bin": bin_label(bin_index),
                    "demand": demand, **stat_row(values),
                    "independent_runs": support_counts(data, mask)[0],
                    "conditions": support_counts(data, mask)[1],
                    "capability_supported_runs": cap["support"]["run_count"],
                    "capability_supported_conditions": cap["support"]["conditions"],
                    "conservative_inner_capability_mps2": cap["inner_capability"],
                    "run_capabilities_mps2": cap["run_capabilities"],
                    "planner_support_gate": cap["supported"],
                    "run_cluster_median_ci90_mps2": cap["run_bootstrap_median_ci90"],
                })
    return lateral_rows, longitudinal_rows, caps


def curvature_table(
    motion: dict[str, dict[int, dict[str, np.ndarray]]],
    *, minimum_samples: int = MIN_GROUP_SAMPLES,
) -> list[dict[str, Any]]:
    result = []
    for split, bins in motion.items():
        for bin_index, data in bins.items():
            curvature = data["curvature"]
            side_masks = {"left": curvature > 0.0, "right": curvature < 0.0}
            for side, side_mask in side_masks.items():
                sustained = np.zeros(len(curvature), dtype=bool)
                # A supported curvature point must belong to a contiguous run
                # of at least 10 samples within the same source sequence/bin.
                indexes = np.flatnonzero(side_mask & (data["speed"] > 0.5))
                if indexes.size:
                    discontinuity = (
                        (np.diff(indexes) != 1)
                        | (np.diff(data["seq"][indexes]) != 0)
                        | (np.diff(data["index"][indexes]) != 1)
                    )
                    for segment in np.split(indexes, np.flatnonzero(discontinuity) + 1):
                        if len(segment) >= SUSTAINED_SAMPLES:
                            sustained[segment] = True
                values = np.abs(curvature[sustained])
                n_runs, n_conditions = support_counts(data, sustained)
                cap = inner_capability(data, sustained, np.abs(curvature), minimum_samples=minimum_samples) if split == "train" else {
                    "supported": False, "inner_capability": None,
                    "support": {"run_count": 0, "conditions": 0},
                    "run_cluster_median_ci90_m_inv": None,
                }
                result.append({
                    "split": split, "speed_bin": bin_label(bin_index),
                    "turn_direction": side, **stat_row(values),
                    "independent_runs": n_runs, "conditions": n_conditions,
                    "capability_supported_runs": cap.get("support", {}).get("run_count", 0),
                    "capability_supported_conditions": cap.get("support", {}).get("conditions", 0),
                    "conservative_curvature_m_inv": cap.get("inner_capability"),
                    "run_capabilities_m_inv": cap.get("run_capabilities", {}),
                    "planner_support_gate": cap.get("supported", False),
                    "run_cluster_median_ci90_m_inv": cap.get("run_bootstrap_median_ci90"),
                })
    return result


def steering_authority_table(
    motion: dict[str, dict[int, dict[str, np.ndarray]]],
    *, minimum_samples: int = MIN_GROUP_SAMPLES,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    def sequence_medians(
        data: dict[str, np.ndarray], mask: np.ndarray, values: np.ndarray,
    ) -> tuple[dict[int, float], dict[int, int]]:
        selected = np.flatnonzero(mask & np.isfinite(values))
        sequences = data["seq"][selected]
        boundaries = np.flatnonzero(np.diff(sequences) != 0) + 1
        responses: dict[int, float] = {}
        runs: dict[int, int] = {}
        for group in np.split(selected, boundaries):
            if group.size < minimum_samples:
                continue
            sequence = int(data["seq"][group[0]])
            responses[sequence] = float(np.median(values[group]))
            runs[sequence] = int(data["run"][group[0]])
        return responses, runs

    rows = []
    summaries = []
    steer_edges = np.asarray(STEER_EDGES)
    for split, bins in motion.items():
        for bin_index, data in bins.items():
            steer, rate = data["steer"], data["r"]
            abs_steer = np.abs(steer)
            steer_bin_index = np.searchsorted(steer_edges, abs_steer, side="right") - 1
            steer_bin_index[abs_steer >= steer_edges[-1]] = len(steer_edges) - 2
            for direction, sign in (("left", 1.0), ("right", -1.0)):
                responses_by_steer: list[dict[int, float]] = []
                runs_by_sequence: dict[int, int] = {}
                for steer_i in range(len(steer_edges) - 1):
                    mask_base = (
                        np.isfinite(steer) & np.isfinite(rate)
                        & (np.sign(steer) == sign)
                        & (steer_bin_index == steer_i)
                    )
                    sequence_responses, sequence_runs = sequence_medians(
                        data, mask_base, rate * sign)
                    responses_by_steer.append(sequence_responses)
                    runs_by_sequence.update(sequence_runs)
                    run_ids = [sequence_runs[seq] for seq in sequence_responses]
                    values = list(sequence_responses.values())
                    n_runs, n_conditions = len(set(run_ids)), len(sequence_responses)
                    rows.append({
                        "split": split, "speed_bin": bin_label(bin_index),
                        "turn_direction": direction,
                        "steering_lower_rad": float(steer_edges[steer_i]),
                        "steering_upper_rad": float(steer_edges[steer_i + 1]),
                        "conditions": n_conditions, "independent_runs": n_runs,
                        "support_run_indices": sorted(set(run_ids)),
                        "median_signed_yaw_rate_radps": float(np.median(values)) if values else None,
                        "sequence_response_count": len(values),
                    })

                # Paired adjacent steering bins within each continuous test
                # sequence isolate steering demand from speed/run differences.
                for steer_i in range(1, len(steer_edges) - 1):
                    lower = responses_by_steer[steer_i - 1]
                    upper = responses_by_steer[steer_i]
                    paired_sequences = sorted(set(lower) & set(upper))
                    differences = [upper[seq] - lower[seq] for seq in paired_sequences]
                    diff_runs = {runs_by_sequence[seq] for seq in paired_sequences}
                    positive = bool(
                        len(diff_runs) >= MIN_TRAIN_RUNS
                        and len(paired_sequences) >= MIN_TRAIN_CONDITIONS
                        and differences
                        and float(np.quantile(differences, 0.10)) > 0.0
                    )
                    summaries.append({
                        "split": split, "speed_bin": bin_label(bin_index),
                        "turn_direction": direction,
                        "steering_step_lower_rad": float(steer_edges[steer_i - 1]),
                        "steering_step_upper_rad": float(steer_edges[steer_i]),
                        "paired_conditions": len(paired_sequences),
                        "independent_runs": len(diff_runs),
                        "support_run_indices": sorted(diff_runs),
                        "incremental_yaw_authority_p10_radps": float(np.quantile(differences, 0.10)) if differences else None,
                        "incremental_yaw_authority_median_radps": float(np.median(differences)) if differences else None,
                        "positive_authority_supported": positive,
                    })
    return rows, summaries


def fit_superellipse(
    train_data: dict[str, np.ndarray],
    axis_caps: dict[str, float],
    *, minimum_samples: int = MIN_GROUP_SAMPLES,
) -> tuple[float | None, dict[str, Any]]:
    """Fit one p from run/condition-balanced training frontier slices."""
    plus = axis_caps["ax_accel"]
    minus = axis_caps["ax_brake"]
    y_accel_left = axis_caps["ay_accel_left"]
    y_accel_right = axis_caps["ay_accel_right"]
    y_brake_left = axis_caps["ay_brake_left"]
    y_brake_right = axis_caps["ay_brake_right"]
    if min(plus, minus, y_accel_left, y_accel_right,
           y_brake_left, y_brake_right) <= 0.0:
        return None, {"status": "missing_positive_axis_support"}
    ax, ay = train_data["ax"], train_data["ay"]
    ax_cap = np.where(ax >= 0.0, plus, minus)
    ay_accel = np.where(ay >= 0.0, y_accel_left, y_accel_right)
    ay_brake = np.where(ay >= 0.0, y_brake_left, y_brake_right)
    ay_cap = np.where(ax >= 0.0, ay_accel, ay_brake)
    x_norm = np.abs(ax) / ax_cap
    y_norm = np.abs(ay) / ay_cap
    active = np.isfinite(x_norm) & np.isfinite(y_norm) & (np.maximum(x_norm, y_norm) >= 0.45)
    if active.sum() < 100:
        return None, {"status": "insufficient_training_frontier", "active_samples": int(active.sum())}

    # Select each condition's upper frontier, retain the strongest supported
    # condition in each run/utilization slice, then give every independent run
    # one vote per slice. Long, densely sampled captures cannot dominate p.
    run_slice_frontier: dict[tuple[int, int], tuple[float, float]] = {}
    x_edges = np.linspace(0.0, 1.2, 7)
    selected = np.flatnonzero(active)
    selected_seq = train_data["seq"][selected]
    boundaries = np.flatnonzero(np.diff(selected_seq) != 0) + 1
    for sequence in np.split(selected, boundaries):
        if not sequence.size:
            continue
        run_index = int(train_data["run"][sequence[0]])
        for lower, upper in zip(x_edges[:-1], x_edges[1:]):
            local = (x_norm[sequence] >= lower) & (x_norm[sequence] < upper)
            if local.sum() >= minimum_samples:
                point = (float(np.median(x_norm[sequence][local])),
                         float(np.quantile(y_norm[sequence][local], 0.90)))
                key = (run_index, int(np.searchsorted(x_edges, (lower + upper) / 2.0) - 1))
                if key not in run_slice_frontier or point[1] > run_slice_frontier[key][1]:
                    run_slice_frontier[key] = point
    frontier_records = [
        (run_index, point[0], point[1])
        for (run_index, _), point in run_slice_frontier.items()
    ]
    frontier_x = [point[1] for point in frontier_records]
    frontier_y = [point[2] for point in frontier_records]
    fx, fy = np.asarray(frontier_x), np.asarray(frontier_y)
    if len(fx) < 12:
        return None, {"status": "insufficient_condition_frontier", "frontier_points": len(fx)}
    usable = (fx < 1.0) & (fy > 0.0) & (fy < 1.25)
    frontier_runs = {frontier_records[i][0] for i in np.flatnonzero(usable)}
    fx, fy = fx[usable], fy[usable]
    if len(fx) < 8:
        return None, {"status": "insufficient_interior_frontier", "frontier_points": len(fx)}
    if len(frontier_runs) < MIN_TRAIN_RUNS:
        return None, {
            "status": "insufficient_independent_frontier_runs",
            "frontier_points": int(len(fx)), "independent_runs": len(frontier_runs),
        }
    errors = []
    for exponent in POWER_EXPONENTS:
        predicted_y = np.maximum(1.0 - np.minimum(fx, 1.0) ** exponent, 0.0) ** (1.0 / exponent)
        errors.append(float(np.median(np.abs(predicted_y - fy))))
    best = int(np.argmin(errors))
    return float(POWER_EXPONENTS[best]), {
        "status": "fit", "frontier_points": int(len(fx)),
        "independent_runs": int(len(frontier_runs)),
        "frontier_median_abs_error": errors[best],
        "candidate_exponents": POWER_EXPONENTS.tolist(),
        "candidate_errors": errors,
    }


def ggv_tables(
    motion: dict[str, dict[int, dict[str, np.ndarray]]],
    lateral_rows: list[dict[str, Any]],
    longitudinal_rows: list[dict[str, Any]],
    *, minimum_samples: int = MIN_GROUP_SAMPLES,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    lateral_lookup = {
        (row["split"], row["speed_bin"], row["longitudinal_demand"], row["turn_direction"]): row
        for row in lateral_rows
    }
    long_lookup = {
        (row["split"], row["speed_bin"], row["demand"]): row
        for row in longitudinal_rows
    }
    envelope_rows = []
    validation_rows = []
    fits: dict[str, Any] = {}
    for bin_index in range(len(SPEED_EDGES) - 1):
        label = bin_label(bin_index)
        def cap(demand: str, side: str) -> dict[str, Any]:
            row = lateral_lookup.get(("train", label, demand, side), {})
            return {
                "value": row.get("conservative_inner_capability_mps2"),
                "supported": bool(row.get("planner_support_gate", False)),
                "runs": row.get("capability_supported_runs", 0),
                "conditions": row.get("capability_supported_conditions", 0),
            }
        def long_cap(demand: str) -> dict[str, Any]:
            row = long_lookup.get(("train", label, demand), {})
            return {
                "value": row.get("conservative_inner_capability_mps2"),
                "supported": bool(row.get("planner_support_gate", False)),
                "runs": row.get("capability_supported_runs", 0),
                "conditions": row.get("capability_supported_conditions", 0),
            }
        axis = {
            "ax_accel": long_cap("acceleration"),
            "ax_brake": long_cap("braking"),
            "ay_left": cap("low_longitudinal", "left"),
            "ay_right": cap("low_longitudinal", "right"),
            "ay_accel_left": cap("accelerating", "left"),
            "ay_accel_right": cap("accelerating", "right"),
            "ay_brake_left": cap("braking", "left"),
            "ay_brake_right": cap("braking", "right"),
        }
        values = {name: float(item["value"]) for name, item in axis.items() if item["value"] is not None}
        train_data = motion["train"][bin_index]
        p, fit = fit_superellipse(train_data, values, minimum_samples=minimum_samples) if len(values) == 8 else (None, {"status": "unsupported_axes"})
        supported = all(item["supported"] for item in axis.values()) and p is not None
        result = {
            "speed_bin": label, "speed_min_mps": SPEED_EDGES[bin_index],
            "speed_max_mps": SPEED_EDGES[bin_index + 1], "power_exponent_p": p,
            "axes_mps2": axis, "fit": fit, "train_support_gate": supported,
        }
        envelope_rows.append(result)
        fits[label] = result
        validation_data = motion["validation"][bin_index]
        practice_rows = motion.get("practice_validation", {}).get(bin_index)
        check_sets = [("openplane_validation", validation_data)]
        if practice_rows is not None:
            check_sets.append(("practice_validation", practice_rows))
        for validation_name, data in check_sets:
            ax, ay = data["ax"], data["ay"]
            sample_count = int(len(ax))
            if not supported or sample_count == 0:
                qvalue = np.full(sample_count, math.nan)
            else:
                xcap = np.where(ax >= 0.0, values["ax_accel"], values["ax_brake"])
                side_positive = ay >= 0.0
                y_low = np.where(side_positive, values["ay_left"], values["ay_right"])
                y_accel = np.where(side_positive, values["ay_accel_left"], values["ay_accel_right"])
                y_brake = np.where(side_positive, values["ay_brake_left"], values["ay_brake_right"])
                ycap = np.where(
                    ax > LONG_DEMAND_THRESHOLD_MPS2, y_accel,
                    np.where(ax < -LONG_DEMAND_THRESHOLD_MPS2, y_brake, y_low),
                )
                qvalue = (np.abs(ax / xcap) ** p) + (np.abs(ay / ycap) ** p)
            active = np.isfinite(qvalue) & (np.maximum(
                np.abs(ax) / max(values.get("ax_accel", 1e-9), values.get("ax_brake", 1e-9)),
                np.abs(ay) / max(values.get("ay_left", 1e-9), values.get("ay_right", 1e-9)),
            ) >= 0.35)
            runs, conditions = support_counts(data, np.isfinite(qvalue)) if sample_count else (0, 0)
            active_runs, active_conditions = support_counts(data, active) if active.any() else (0, 0)
            violation = float(np.mean(qvalue > 1.0)) if sample_count and np.isfinite(qvalue).all() else None
            active_violation = float(np.mean(qvalue[active] > 1.0)) if active.any() else None
            per_run = {}
            for run in np.unique(data["run"][active]):
                mask = active & (data["run"] == run)
                if mask.any():
                    per_run[str(int(run))] = float(np.mean(qvalue[mask] > 1.0))
            status = "unexercised" if active_runs < MIN_VALIDATION_RUNS or active_conditions < MIN_TRAIN_CONDITIONS else (
                "pass" if active_violation is not None and active_violation <= 0.05 and max(per_run.values(), default=0.0) <= 0.10 else "fail"
            )
            validation_rows.append({
                "speed_bin": label, "validation_source": validation_name,
                "samples": sample_count, "independent_runs": runs,
                "conditions": conditions, "active_samples": int(active.sum()),
                "active_runs": active_runs, "active_conditions": active_conditions,
                "sample_violation_fraction": violation,
                "active_violation_fraction": active_violation,
                "per_run_active_violation_fraction": per_run,
                "status": status,
            })
    return envelope_rows, validation_rows, fits


def make_practice_validation(paths: list[Path]) -> tuple[dict[int, dict[str, np.ndarray]], list[dict[str, Any]]]:
    by_bin: dict[int, dict[str, list[np.ndarray]]] = {
        i: {name: [] for name in ("speed", "steer", "throttle", "u", "v", "r", "ax", "ay", "ay_exact", "curvature", "run", "seq", "index")}
        for i in range(len(SPEED_EDGES) - 1)
    }
    provenance = []
    next_run_index = 0
    for path in paths:
        if not path.is_file():
            raise ValueError(f"practice validation tracking report missing: {path}")
        summary_path = path.with_name("summary.json")
        if not summary_path.is_file():
            raise ValueError(f"practice validation summary missing: {summary_path}")
        run_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (int(run_summary.get("scored_laps", 0)) != 10
                or int(run_summary.get("lap_count_final", 0)) < 12
                or run_summary.get("collision_delta") != 0
                or not run_summary.get("collision_topics_recorded")):
            raise ValueError(
                f"practice validation run is incomplete or not collision-free: {summary_path}")
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            raise ValueError(f"practice validation report is empty: {path}")
        selected = [
            (row_index, row) for row_index, row in enumerate(rows)
            if 2 <= int(float(row["lap_count"])) <= 11
        ]
        if not selected:
            raise ValueError(f"practice report contains no scored-lap samples: {path}")
        sample_by_bin: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        run_id = path.parents[2].name
        for row_index, row in selected:
            try:
                speed = abs(float(row["truth_speed_mps"]))
                steering = float(row["steering_feedback_rad"])
                throttle = float(row["throttle_feedback"])
                u, v = float(row["truth_speed_mps"]), float(row["truth_lateral_speed_mps"])
                yaw = float(row["yaw_rate_radps"])
                ax = float(row["ax_mps2"])
                ay = float(row["ay_proxy_mps2"])
                ay_exact = float(row["ay_body_mps2"])
                curvature = float(row["actual_curvature_inv_m"])
                lap = int(float(row["lap_count"]))
            except (KeyError, ValueError, TypeError):
                continue
            if not all(math.isfinite(value) for value in (speed, steering, throttle, u, v, yaw, ax, ay, ay_exact, curvature)):
                continue
            bin_index = int(speed_bin(np.asarray([speed]))[0])
            for name, value in (
                ("speed", speed), ("steer", steering), ("throttle", throttle),
                ("u", u), ("v", v), ("r", yaw), ("ax", ax),
                ("ay", ay), ("ay_exact", ay_exact), ("curvature", curvature),
            ):
                sample_by_bin[bin_index][name].append(value)
            sample_by_bin[bin_index]["run"].append(float(next_run_index))
            sample_by_bin[bin_index]["seq"].append(float(lap))
            sample_by_bin[bin_index]["index"].append(float(row_index))
        for bin_index, values in sample_by_bin.items():
            for name, items in values.items():
                by_bin[bin_index][name].append(np.asarray(items))
        provenance.append({"run_id": run_id, "source_report": str(path), "scored_samples": len(selected)})
        next_run_index += 1
    combined = {
        index: {
            name: np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float64)
            for name, chunks in fields.items()
        }
        for index, fields in by_bin.items()
    }
    return combined, provenance


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({
            key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
            for key, value in row.items()
        } for row in rows)


def svg_ggv_grid(output: Path, fits: dict[str, Any], motion: dict[str, dict[int, dict[str, np.ndarray]]]) -> None:
    cols, rows = 3, 4
    panel_w, panel_h = 360, 285
    width, height = cols * panel_w, rows * panel_h
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="white"/>']
    train_color, validation_color = "#1769aa", "#d94b3d"
    for bin_index in range(len(SPEED_EDGES) - 1):
        col, row_i = bin_index % cols, bin_index // cols
        x0, y0 = col * panel_w, row_i * panel_h
        left, top, right, bottom = x0 + 52, y0 + 42, x0 + panel_w - 20, y0 + panel_h - 36
        entry = fits[bin_label(bin_index)]
        axis = entry["axes_mps2"]
        supported = entry["train_support_gate"]
        max_x = max([float(v["value"]) for v in (axis.get("ax_accel"), axis.get("ax_brake")) if v.get("value") is not None] + [1.0]) * 1.1
        max_y = max([float(v["value"]) for v in (axis.get("ay_left"), axis.get("ay_right")) if v.get("value") is not None] + [1.0]) * 1.1
        sx = lambda x: left + (x + max_x) / (2 * max_x) * (right - left)
        sy = lambda y: bottom - (y + max_y) / (2 * max_y) * (bottom - top)
        out.append(f'<rect x="{x0+5}" y="{y0+5}" width="{panel_w-10}" height="{panel_h-10}" fill="none" stroke="#bbb"/>')
        out.append(f'<text x="{x0+14}" y="{y0+24}" font-family="sans-serif" font-size="16" font-weight="bold">{bin_label(bin_index)} m/s{" (supported)" if supported else " (insufficient support)"}</text>')
        for tick in range(5):
            xval = -max_x + 2 * max_x * tick / 4
            yval = -max_y + 2 * max_y * tick / 4
            out.append(f'<path d="M{sx(xval):.2f},{top} V{bottom} M{left},{sy(yval):.2f} H{right}" stroke="#e6e6e6"/>')
        out.append(f'<path d="M{sx(0):.2f},{top} V{bottom} M{left},{sy(0):.2f} H{right}" stroke="#666"/>')
        data = motion["train"][bin_index]
        valdata = motion["validation"][bin_index]
        for dataset, color, seed in ((data, train_color, bin_index + 4), (valdata, validation_color, bin_index + 40)):
            n = len(dataset["ax"])
            if not n:
                continue
            indexes = np.arange(n)
            if n > 800:
                indexes = np.random.default_rng(seed).choice(indexes, size=800, replace=False)
            for i in indexes:
                ax, ay = float(dataset["ax"][i]), float(dataset["ay"][i])
                if abs(ax) <= max_x and abs(ay) <= max_y:
                    out.append(f'<circle cx="{sx(ax):.2f}" cy="{sy(ay):.2f}" r="1.2" fill="{color}" opacity="0.23"/>')
        p = entry["power_exponent_p"]
        if supported and p is not None:
            theta = np.linspace(0.0, 2.0 * math.pi, 180)
            curve_x, curve_y = [], []
            for angle in theta:
                c, s = math.cos(angle), math.sin(angle)
                xcap = float(axis["ax_accel"]["value"]) if c >= 0 else float(axis["ax_brake"]["value"])
                ycap = float(axis["ay_left"]["value"]) if s >= 0 else float(axis["ay_right"]["value"])
                curve_x.append(sx(math.copysign(xcap * abs(c) ** (2.0 / p), c)))
                curve_y.append(sy(math.copysign(ycap * abs(s) ** (2.0 / p), s)))
            pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in zip(curve_x, curve_y))
            out.append(f'<polyline points="{pts}" fill="none" stroke="#111" stroke-width="1.7"/>')
        out.append(f'<text x="{(left+right)/2:.1f}" y="{y0+panel_h-8}" text-anchor="middle" font-family="sans-serif" font-size="11">ax (m/s²)</text>')
        out.append(f'<text x="{x0+12}" y="{(top+bottom)/2:.1f}" transform="rotate(-90 {x0+12} {(top+bottom)/2:.1f})" text-anchor="middle" font-family="sans-serif" font-size="11">ay = u·r (m/s²)</text>')
    out.append(f'<text x="{width-225}" y="{height-6}" font-family="sans-serif" font-size="12" fill="{train_color}">training</text>')
    out.append(f'<text x="{width-155}" y="{height-6}" font-family="sans-serif" font-size="12" fill="{validation_color}">held-out validation</text>')
    out.append("</svg>")
    output.write_text("\n".join(out), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--practice-report", type=Path, action="append", default=[])
    parser.add_argument("--minimum-samples", type=int, default=MIN_GROUP_SAMPLES,
                        help="minimum samples per sustained condition (default: 20)")
    parser.add_argument("--minimum-transient-samples", type=int,
                        default=MIN_TRANSIENT_SAMPLES,
                        help="minimum samples per longitudinal transient condition (default: 5)")
    args = parser.parse_args()
    if args.minimum_samples < 5 or args.minimum_transient_samples < 5:
        parser.error("both sample minimums must be at least five")
    if args.manifest is None:
        args.manifest = args.dataset.with_name("manifest.json")
    if not args.dataset.is_file() or not args.manifest.is_file():
        parser.error("dataset and manifest must exist")
    args.output.mkdir(parents=True, exist_ok=True)
    print("verifying source provenance and fixed 40 Hz timebase", flush=True)
    arrays, manifest = load_dataset(args.dataset, args.manifest)
    print("extracting only whole-run training and validation intervals", flush=True)
    motion, qc = build_motion_samples(arrays)
    lateral_rows, longitudinal_rows, _ = lateral_and_longitudinal_tables(
        motion, minimum_samples=args.minimum_samples,
        minimum_transient_samples=args.minimum_transient_samples)
    print("estimating run/condition-supported lateral, longitudinal, and curvature limits", flush=True)
    curvature_rows = curvature_table(motion, minimum_samples=args.minimum_samples)
    steer_rows, steer_steps = steering_authority_table(
        motion, minimum_samples=args.minimum_samples)
    practice_motion = None
    practice_provenance = []
    if args.practice_report:
        print("adding whole-run fresh practice data as separate validation only", flush=True)
        practice_motion, practice_provenance = make_practice_validation(args.practice_report)
        motion["practice_validation"] = practice_motion
    envelope_rows, validation_rows, fits = ggv_tables(
        motion, lateral_rows, longitudinal_rows,
        minimum_samples=args.minimum_transient_samples)

    # Persist human-auditable run identifiers, not internal array indexes.
    run_id_by_index = arrays["run_ids"].astype(str).tolist()
    for table in (lateral_rows, longitudinal_rows, curvature_rows,
                  steer_rows, steer_steps):
        for row in table:
            for key in ("run_capabilities_mps2", "run_capabilities_m_inv"):
                if key in row:
                    row[key] = {
                        run_id_by_index[int(index)]: value
                        for index, value in row[key].items()
                    }
            if "support_run_indices" in row:
                row["support_run_ids"] = [
                    run_id_by_index[int(index)]
                    for index in row.pop("support_run_indices")
                ]
    practice_run_ids = [str(row["run_id"]) for row in practice_provenance]
    for row in validation_rows:
        run_ids = run_id_by_index if row["validation_source"] == "openplane_validation" else practice_run_ids
        row["per_run_active_violation_fraction"] = {
            run_ids[int(index)]: value
            for index, value in row["per_run_active_violation_fraction"].items()
        }
    # Qualification of a planner-ready envelope is conjunctive: every speed
    # bin used by the candidate must have independent-run support and must be
    # exercised by separate validation runs. We report each bin, not a single
    # flattering pooled metric.
    statuses = defaultdict(list)
    for row in validation_rows:
        statuses[row["speed_bin"]].append(row["status"])
    for row in envelope_rows:
        validation_status = statuses.get(row["speed_bin"], [])
        row["validation_status"] = (
            "pass" if validation_status and all(item == "pass" for item in validation_status)
            else "unexercised" if not validation_status or all(item == "unexercised" for item in validation_status)
            else "fail"
        )
        row["planner_ready"] = row["train_support_gate"] and row["validation_status"] == "pass"

    def supported_bins(rows: list[dict[str, Any]], predicate) -> list[str]:
        result = []
        for index in range(len(SPEED_EDGES) - 1):
            label = bin_label(index)
            selected = [row for row in rows if row["split"] == "train"
                        and row["speed_bin"] == label]
            if predicate(selected):
                result.append(label)
        return result

    lateral_supported_bins = supported_bins(
        lateral_rows,
        lambda rows: all(
            any(row["longitudinal_demand"] == "low_longitudinal"
                and row["turn_direction"] == side
                and row["planner_support_gate"] for row in rows)
            for side in ("left", "right")),
    )
    longitudinal_supported_bins = supported_bins(
        longitudinal_rows,
        lambda rows: all(
            any(row["demand"] == demand and row["planner_support_gate"]
                for row in rows)
            for demand in ("acceleration", "braking")),
    )
    curvature_supported_bins = supported_bins(
        curvature_rows,
        lambda rows: all(
            any(row["turn_direction"] == side
                and row["planner_support_gate"] for row in rows)
            for side in ("left", "right")),
    )

    output_envelope = {
        "schema_version": 1,
        "source_dataset": str(args.dataset),
        "source_manifest": str(args.manifest),
        "source_dataset_sha256": manifest["computed_dataset_sha256"],
        "source_split_policy": "whole-run train only for fitting; whole-run validation for scoring; test and final_test excluded",
        "timebase_s": DT_S,
        "state_definitions": {
            "speed_mps": "absolute rear-axle longitudinal body velocity from simulator-aligned frame labels",
            "lateral_acceleration_mps2": "u_rear * yaw_rate; primary racing lateral demand",
            "longitudinal_acceleration_mps2": "100 ms central difference of u_rear on fixed 25 ms grid minus v_rear*yaw_rate",
            "body_lateral_acceleration_diagnostic_mps2": "100 ms central difference of v_rear plus u_rear*yaw_rate",
            "curvature_m_inv": "yaw_rate/max(abs(u_rear),0.25 m/s)",
        },
        "quality_policy": {
            "source_whole_run_gates": manifest.get("quality_gate"),
            "sample_rejections": qc,
            "airborne_observability": (
                "The archive has simulator COM vertical velocity but no explicit wheel-contact flag. "
                "Vertical velocity is reported as a diagnostic and is not threshold-rejected; source "
                "runs are whole-run collision-free and samples above the measured tilt limit are excluded."
            ),
            "longitudinal_demand_threshold_mps2": LONG_DEMAND_THRESHOLD_MPS2,
            "straight_line_lateral_threshold_mps2": LATERAL_STRAIGHT_THRESHOLD_MPS2,
            "planning_capability_fraction": PLANNING_CAPABILITY_FRACTION,
            "minimum_samples_sustained_condition": args.minimum_samples,
            "minimum_samples_transient_condition": args.minimum_transient_samples,
            "sustained_curvature_min_duration_s": SUSTAINED_SAMPLES * DT_S,
            "speed_bins_mps": [[SPEED_EDGES[i], SPEED_EDGES[i + 1]] for i in range(len(SPEED_EDGES)-1)],
            "practice_validation_runs": practice_provenance,
        },
        "lateral_envelope": lateral_rows,
        "longitudinal_envelope": longitudinal_rows,
        "curvature_envelope": curvature_rows,
        "steering_response": steer_rows,
        "incremental_steering_authority": steer_steps,
        "ggv_superellipse": envelope_rows,
        "ggv_validation": validation_rows,
        "lateral_supported_speed_bins": lateral_supported_bins,
        "longitudinal_supported_speed_bins": longitudinal_supported_bins,
        "curvature_supported_speed_bins": curvature_supported_bins,
        "planner_ready_speed_bins": [row["speed_bin"] for row in envelope_rows if row["planner_ready"]],
        "non_ready_speed_bins": [row["speed_bin"] for row in envelope_rows if not row["planner_ready"]],
    }
    (args.output / "empirical_racing_envelope.json").write_text(
        json.dumps(output_envelope, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_csv(args.output / "lateral_envelope_by_bin.csv", lateral_rows)
    write_csv(args.output / "longitudinal_envelope_by_bin.csv", longitudinal_rows)
    write_csv(args.output / "curvature_envelope_by_bin.csv", curvature_rows)
    write_csv(args.output / "steering_response_by_bin.csv", steer_rows)
    write_csv(args.output / "steering_authority_increments.csv", steer_steps)
    write_csv(args.output / "ggv_validation.csv", validation_rows)
    svg_ggv_grid(args.output / "ggv_train_validation.svg", fits, motion)

    lines = [
        "# Empirical vehicle envelope — 2026-10-05", "",
        f"Source dataset: `{args.dataset}`", f"SHA-256: `{manifest['computed_dataset_sha256']}`", "",
        "## Data split and quality", "",
        "The fit uses only whole-run `train` captures. Whole-run `validation` and fresh practice captures are scored separately. Sealed `test` and `final_test` partitions are intentionally absent from every summary and fit.", "",
        f"Training captures: {len(qc['train']['run_ids'])}; validation captures: {len(qc['validation']['run_ids'])}; eligible sample intervals: train {qc['train']['eligible_samples']}, validation {qc['validation']['eligible_samples']}.",
        f"Sample exclusions: non-finite {qc['train']['rejected_missing_or_nonfinite'] + qc['validation']['rejected_missing_or_nonfinite']}; invalid sensor {qc['train']['rejected_sensor_invalid'] + qc['validation']['rejected_sensor_invalid']}; over-12 m/s {qc['train']['rejected_speed_over_12'] + qc['validation']['rejected_speed_over_12']}; tilt ≥8° {qc['train']['rejected_tilt_over_8deg'] + qc['validation']['rejected_tilt_over_8deg']}; packet discontinuity {qc['train']['rejected_packet_gap'] + qc['validation']['rejected_packet_gap']}. Vertical-speed p95/p99/max (diagnostic, m/s): train {qc['train']['vertical_speed_stats']['p95_mps']}/{qc['train']['vertical_speed_stats']['p99_mps']}/{qc['train']['vertical_speed_stats']['max_mps']}; validation {qc['validation']['vertical_speed_stats']['p95_mps']}/{qc['validation']['vertical_speed_stats']['p99_mps']}/{qc['validation']['vertical_speed_stats']['max_mps']}.", "",
        "Lateral/longitudinal envelopes report raw sample quantiles and a conservative inner capability. Each condition contributes its p90; each independent run contributes its strongest supported condition p90; the planning boundary is 90% of the median independent-run capability. The 90% bootstrap interval for the run median is reported as run-level uncertainty, not used to select the boundary. Sustained conditions require 20 samples; transient longitudinal and combined-demand conditions require five 40 Hz samples, spanning the 100 ms acceleration-difference interval. A bin is support-qualified only with at least three runs and three conditions. Confidence intervals resample whole runs, not correlated 40 Hz frames.", "",
        "The source has no explicit wheel-contact flag. Vertical speed is retained with run-level tail statistics rather than threshold-rejected; whole-run collision gates and the measured 8° tilt gate are applied. Airborne detection remains limited by the recorded simulator state.", "",
        "## Supported speed bins", "", "| Speed (m/s) | Lateral | Longitudinal accel+brake | Curvature | GGV validation | Planner-ready | p |", "|---|---|---|---|---|---|---:|",
    ]
    for row in envelope_rows:
        lines.append(
            f"| {row['speed_bin']} | {'yes' if row['speed_bin'] in lateral_supported_bins else 'no'} | "
            f"{'yes' if row['speed_bin'] in longitudinal_supported_bins else 'no'} | "
            f"{'yes' if row['speed_bin'] in curvature_supported_bins else 'no'} | "
            f"{row['validation_status']} | {'yes' if row['planner_ready'] else 'no'} | "
            f"{row['power_exponent_p'] if row['power_exponent_p'] is not None else 'n/a'} |"
        )
    lines.extend([
        "", "## Interpretation and gates", "",
        f"Planner-ready bins: {', '.join(output_envelope['planner_ready_speed_bins']) or 'none'}.",
        f"Unsupported or failing bins: {', '.join(output_envelope['non_ready_speed_bins']) or 'none'}.",
        f"Lateral (both directions, low longitudinal demand) support: {', '.join(lateral_supported_bins) or 'none'}.",
        f"Longitudinal acceleration and braking support: {', '.join(longitudinal_supported_bins) or 'none'}.",
        f"Sustained curvature support (both directions): {', '.join(curvature_supported_bins) or 'none'}.",
        "The GGV fit is an offline candidate, not yet integrated into the optimizer. Per-bin validation checks the active portion of each whole held-out run; bins without active held-out excitation remain `unexercised`, not passed.",
        "The practice validation is separate from open-plane fitting and is used only as a transfer check. A low-demand practice lap cannot establish high-demand capability by itself.", "",
        "## Artifacts", "",
        "- `empirical_racing_envelope.json` — fitted, provenance-bearing offline artifact.",
        "- `lateral_envelope_by_bin.csv` — left/right and longitudinal-demand quantiles/support.",
        "- `longitudinal_envelope_by_bin.csv` — acceleration/braking robust caps.",
        "- `curvature_envelope_by_bin.csv` — sustained curvature support.",
        "- `steering_response_by_bin.csv` and `steering_authority_increments.csv` — measured yaw authority by steering.",
        "- `ggv_train_validation.svg` and `ggv_validation.csv` — train fit with independent validation points and violation rates.",
    ])
    (args.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"eligible intervals: train={qc['train']['eligible_samples']}, validation={qc['validation']['eligible_samples']}")
    print("planner-ready bins:", ", ".join(output_envelope["planner_ready_speed_bins"]) or "none")
    print("summary:", args.output / "summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
