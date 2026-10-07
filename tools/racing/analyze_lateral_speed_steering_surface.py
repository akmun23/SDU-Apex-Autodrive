#!/usr/bin/env python3
"""Audit train/validation coverage in a coarse lateral-response grid.

This fixed-grid report is only a coverage/consistency audit. Its boundaries
are not discovered physical regimes, and a cell passing its statistical gate
does not make that cell a vehicle limit or authorize optimizer use. It uses
only whole-run train and validation splits, with simulator state offline.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from tools.racing.build_empirical_racing_envelope import (
    DATASET,
    MANIFEST,
    MIN_TRAIN_CONDITIONS,
    MIN_TRAIN_RUNS,
    MIN_VALIDATION_RUNS,
    REPO_ROOT,
    SPEED_EDGES,
    STEER_EDGES,
    build_motion_samples,
    load_dataset,
    run_bootstrap_median,
)


DEMANDS = ("low_longitudinal", "accelerating", "braking")
DEMAND_THRESHOLD_MPS2 = 0.50
LOW_DEMAND_MIN_SAMPLES = 20
TRANSIENT_MIN_SAMPLES = 5
CAPABILITY_FRACTION = 0.90
VALIDATION_SAMPLE_MAX = 0.05
VALIDATION_RUN_MAX = 0.10
OUTPUT = Path("live_runs/racing_model_diagnostics_20261007/lateral_speed_steering")


def steering_bin(steering_rad: np.ndarray) -> np.ndarray:
    edges = np.asarray(STEER_EDGES, dtype=np.float64)
    absolute = np.abs(steering_rad)
    result = np.searchsorted(edges, absolute, side="right") - 1
    result[absolute >= edges[-1]] = len(edges) - 2
    return result.astype(np.int16)


def demand_mask(ax: np.ndarray, demand: str) -> np.ndarray:
    if demand == "low_longitudinal":
        return np.abs(ax) <= DEMAND_THRESHOLD_MPS2
    if demand == "accelerating":
        return ax > DEMAND_THRESHOLD_MPS2
    if demand == "braking":
        return ax < -DEMAND_THRESHOLD_MPS2
    raise ValueError(f"unknown longitudinal-demand class: {demand}")


def condition_quantiles(
    data: dict[str, np.ndarray], mask: np.ndarray, values: np.ndarray,
    minimum_samples: int,
) -> tuple[dict[int, tuple[int, float]], np.ndarray]:
    """Return one p90 per eligible source sequence and its eligible samples."""
    selected = np.flatnonzero(mask & np.isfinite(values))
    if not selected.size:
        return {}, selected
    sequence_ids = data["seq"][selected]
    boundaries = np.flatnonzero(np.diff(sequence_ids) != 0) + 1
    by_sequence: dict[int, tuple[int, float]] = {}
    eligible: list[np.ndarray] = []
    for group in np.split(selected, boundaries):
        if group.size < minimum_samples:
            continue
        seq = int(data["seq"][group[0]])
        run = int(data["run"][group[0]])
        by_sequence[seq] = (run, float(np.quantile(values[group], 0.90)))
        eligible.append(group)
    samples = np.concatenate(eligible) if eligible else np.empty(0, dtype=np.int64)
    return by_sequence, samples


def training_capability(
    data: dict[str, np.ndarray], mask: np.ndarray, values: np.ndarray,
    run_names: np.ndarray, minimum_samples: int,
) -> dict[str, Any]:
    by_sequence, eligible = condition_quantiles(
        data, mask, np.abs(values), minimum_samples)
    by_run: dict[str, list[float]] = defaultdict(list)
    for run_index, p90 in by_sequence.values():
        by_run[str(run_names[run_index])].append(p90)
    run_peak_p90 = {run: max(condition_values)
                    for run, condition_values in by_run.items()}
    supported = (
        len(run_peak_p90) >= MIN_TRAIN_RUNS
        and len(by_sequence) >= MIN_TRAIN_CONDITIONS
    )
    ci = run_bootstrap_median(run_peak_p90) if supported else None
    candidate_cap = (
        CAPABILITY_FRACTION * float(np.median(list(run_peak_p90.values())))
        if supported else None
    )
    return {
        "samples": int(eligible.size),
        "runs": len(set(data["run"][eligible].tolist())) if eligible.size else 0,
        "conditions": len(by_sequence),
        "supported_runs": len(run_peak_p90),
        "candidate_cap_mps2": candidate_cap,
        "run_peak_condition_p90_mps2": run_peak_p90,
        "run_median_ci90_mps2": ci,
        "support_pass": supported,
    }


def validation_score(
    data: dict[str, np.ndarray], mask: np.ndarray, values: np.ndarray,
    candidate_cap: float | None, minimum_samples: int,
    run_names: np.ndarray,
) -> dict[str, Any]:
    by_sequence, eligible = condition_quantiles(
        data, mask, values, minimum_samples)
    run_indices = sorted({run for run, _ in by_sequence.values()})
    per_run: dict[str, float] = {}
    if candidate_cap is not None and eligible.size:
        exceed = np.abs(values[eligible]) > candidate_cap
        for run in run_indices:
            run_samples = eligible[data["run"][eligible] == run]
            if run_samples.size:
                per_run[str(run_names[run])] = float(np.mean(
                    np.abs(values[run_samples]) > candidate_cap))
        pooled = float(np.mean(exceed))
        max_run = max(per_run.values(), default=None)
    else:
        pooled = None
        max_run = None
    exercised = len(run_indices) >= MIN_VALIDATION_RUNS and len(by_sequence) >= 3
    passed = bool(
        candidate_cap is not None and exercised and pooled is not None
        and max_run is not None and pooled <= VALIDATION_SAMPLE_MAX
        and max_run <= VALIDATION_RUN_MAX
    )
    return {
        "samples": int(eligible.size),
        "runs": len(run_indices),
        "conditions": len(by_sequence),
        "sample_violation_fraction": pooled,
        "max_run_violation_fraction": max_run,
        "per_run_violation_fraction": per_run,
        "status": "pass" if passed else "fail" if exercised and candidate_cap is not None else "unexercised",
    }


def cell_rows(
    motion: dict[str, dict[int, dict[str, np.ndarray]]],
    run_names: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    steer_edges = np.asarray(STEER_EDGES, dtype=np.float64)
    for speed_index in range(len(SPEED_EDGES) - 1):
        train = motion["train"][speed_index]
        validation = motion["validation"][speed_index]
        for steer_index in range(len(steer_edges) - 1):
            steer_label = f"{steer_edges[steer_index]:.4f}-{steer_edges[steer_index + 1]:.4f}"
            for turn, turn_sign in (("left", 1.0), ("right", -1.0)):
                for demand in DEMANDS:
                    minimum_samples = (
                        LOW_DEMAND_MIN_SAMPLES if demand == "low_longitudinal"
                        else TRANSIENT_MIN_SAMPLES
                    )

                    def local_mask(data: dict[str, np.ndarray]) -> np.ndarray:
                        angle = data["steer"]
                        signed_turn = np.sign(angle) == turn_sign
                        return (
                            signed_turn
                            & (steering_bin(angle) == steer_index)
                            & demand_mask(data["ax"], demand)
                            & np.isfinite(data["ay_exact"])
                        )

                    train_mask = local_mask(train)
                    validation_mask = local_mask(validation)
                    train_fit = training_capability(
                        train, train_mask, train["ay_exact"], run_names,
                        minimum_samples)
                    val_score = validation_score(
                        validation, validation_mask, validation["ay_exact"],
                        train_fit["candidate_cap_mps2"], minimum_samples,
                        run_names)
                    cell_pass = train_fit["support_pass"] and val_score["status"] == "pass"
                    rows.append({
                        "speed_bin_mps": f"{SPEED_EDGES[speed_index]:g}-{SPEED_EDGES[speed_index + 1]:g}",
                        "abs_steering_bin_rad": steer_label,
                        "turn_direction": turn,
                        "longitudinal_demand": demand,
                        "train_samples": train_fit["samples"],
                        "train_supported_runs": train_fit["supported_runs"],
                        "train_conditions": train_fit["conditions"],
                        "candidate_lateral_accel_p90_cap_mps2": train_fit["candidate_cap_mps2"],
                        "train_run_peak_condition_p90_mps2": train_fit["run_peak_condition_p90_mps2"],
                        "train_run_median_ci90_mps2": train_fit["run_median_ci90_mps2"],
                        "validation_samples": val_score["samples"],
                        "validation_runs": val_score["runs"],
                        "validation_conditions": val_score["conditions"],
                        "validation_sample_violation_fraction": val_score["sample_violation_fraction"],
                        "validation_max_run_violation_fraction": val_score["max_run_violation_fraction"],
                        "validation_per_run_violation_fraction": val_score["per_run_violation_fraction"],
                        "validation_status": val_score["status"],
                        "heldout_cell_pass": bool(cell_pass),
                        "acceleration_source": "simulator-truth body velocity, 100ms central difference on fixed 25ms packet grid: ay=dv/dt+u*r",
                    })
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("no speed/steering cells to write")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({
            key: json.dumps(value, sort_keys=True)
            if isinstance(value, (dict, list)) else value
            for key, value in row.items()
        } for row in rows)


def summarize(rows: list[dict[str, Any]], qc: dict[str, Any], dataset_hash: str) -> dict[str, Any]:
    train_supported = [row for row in rows if row["train_supported_runs"] >= MIN_TRAIN_RUNS]
    exercised = [row for row in rows if row["validation_status"] != "unexercised"]
    passed = [row for row in rows if row["heldout_cell_pass"]]
    return {
        "schema_version": 1,
        "dataset": str(DATASET),
        "dataset_sha256": dataset_hash,
        "included_splits": ["train", "validation"],
        "sealed_splits_loaded": False,
        "speed_bins_mps": [f"{a:g}-{b:g}" for a, b in zip(SPEED_EDGES[:-1], SPEED_EDGES[1:])],
        "absolute_steering_bins_rad": [f"{a:.4f}-{b:.4f}" for a, b in zip(STEER_EDGES[:-1], STEER_EDGES[1:])],
        "longitudinal_demand_threshold_mps2": DEMAND_THRESHOLD_MPS2,
        "training_cap_definition": "0.90 times median independent-run maximum of supported condition p90(|ay|); conditions and runs are weighted equally, not frames",
        "validation_gate": {
            "sample_violation_fraction_max": VALIDATION_SAMPLE_MAX,
            "per_run_violation_fraction_max": VALIDATION_RUN_MAX,
            "minimum_validation_runs": MIN_VALIDATION_RUNS,
            "minimum_validation_conditions": 3,
        },
        "cell_count": len(rows),
        "train_supported_cell_count": len(train_supported),
        "validation_exercised_cell_count": len(exercised),
        "heldout_cell_pass_count": len(passed),
        "quality": qc,
        "interpretation": (
            "This fixed-grid table is only a train/validation coverage audit. "
            "Its boundaries are not learned regimes; cell passes are not tire-force limits or optimizer approval."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=REPO_ROOT / DATASET)
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / MANIFEST)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / OUTPUT)
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    manifest_path = args.manifest.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output}")
    arrays, manifest = load_dataset(dataset, manifest_path)
    motion, qc = build_motion_samples(arrays)
    run_names = arrays["run_ids"].astype(str)
    rows = cell_rows(motion, run_names)
    output.mkdir(parents=True, exist_ok=True)
    table_path = output / "lateral_acceleration_by_speed_steering_demand.csv"
    summary_path = output / "summary.json"
    write_csv(table_path, rows)
    result = summarize(rows, qc, manifest["computed_dataset_sha256"])
    result["outputs"] = [table_path.name]
    summary_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "cells": result["cell_count"],
        "train_supported": result["train_supported_cell_count"],
        "validation_exercised": result["validation_exercised_cell_count"],
        "heldout_cell_pass": result["heldout_cell_pass_count"],
        "source_train_runs": len(qc["train"]["run_ids"]),
        "source_validation_runs": len(qc["validation"]["run_ids"]),
        "source_train_samples": qc["train"]["eligible_samples"],
        "source_validation_samples": qc["validation"]["eligible_samples"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
