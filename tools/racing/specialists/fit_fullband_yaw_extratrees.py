#!/usr/bin/env python3
"""Fit a research-only, run-balanced ExtraTrees yaw-transition atlas.

The model predicts the next 25 ms simulator-truth yaw-rate increment from the
current row features in one exact speed/steering/phase cell. It never fills an
unsupported cell and is not a recursive plant or a runtime controller model.
Only clean train archives are used for fitting; test/final-test arrays are
excluded by the shared archive discovery gate before they are read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from joblib import Parallel, delayed
from sklearn.ensemble import ExtraTreesRegressor

try:
    import fit_fullband_yaw_regime_atlas as atlas
except ModuleNotFoundError:  # Importable both as a script and as a repo module.
    from tools.racing.specialists import fit_fullband_yaw_regime_atlas as atlas


ROOT = Path(__file__).resolve().parents[3]
MODEL_FAMILY = "run_balanced_extra_trees_yaw_increment"
N_ESTIMATORS = 120
MAX_DEPTH = 5
MIN_SAMPLES_LEAF = 4
MAX_FEATURES = 0.8
RANDOM_STATE = 20261007
PHASE_THRESHOLD = 0.0025
INCLUDE_COMMAND_ERRORS = True
INCLUDE_COMMAND_RATES = True
INCLUDE_REAR_WHEEL_SPLIT = False


def _feature_schema() -> tuple[tuple[str, ...], np.ndarray]:
    names = atlas.FEATURE_NAMES + atlas.COMMAND_ERROR_FEATURE_NAMES
    scales = np.concatenate((atlas.FEATURE_SCALES,
                             atlas.COMMAND_ERROR_FEATURE_SCALES))
    names += atlas.COMMAND_RATE_FEATURE_NAMES
    scales = np.concatenate((scales, atlas.COMMAND_RATE_FEATURE_SCALES))
    return names, scales.astype(np.float64, copy=False)


def _run_balanced_weights(run_ids: np.ndarray) -> np.ndarray:
    ids = np.asarray(run_ids, dtype=object)
    counts = Counter(ids.tolist())
    if not counts:
        raise ValueError("cannot weight an empty training cell")
    weights = np.asarray([1.0 / counts[run_id] for run_id in ids],
                         dtype=np.float64)
    weights *= len(counts) / weights.sum()
    return weights


def _fit_cell(job: tuple[tuple[int, int, int], np.ndarray,
                            np.ndarray, np.ndarray,
                            np.ndarray]) -> tuple[tuple[int, int, int], dict[str, Any]]:
    key, x, target_delta, run_ids, feature_scales = job
    mean = np.mean(x, axis=0)
    z = (x - mean) / feature_scales
    model = ExtraTreesRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        max_features=MAX_FEATURES,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    model.fit(z, target_delta, sample_weight=_run_balanced_weights(run_ids))
    run_counts = Counter(np.asarray(run_ids, dtype=object).tolist())
    return key, {
        "estimator": model,
        "feature_mean": mean,
        "feature_scales": feature_scales.copy(),
        "training_samples": int(len(x)),
        "training_runs": sorted(run_counts),
        "training_run_count": len(run_counts),
    }


def fit_candidate(output_dir: Path, workers: int = 12) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("workers must be positive")
    run_series, source_audit = atlas._discover_run_series()
    train_series = [series for series in run_series if series.split == "train"]
    if not train_series:
        raise RuntimeError("clean whole-run training captures are required")
    feature_names, feature_scales = _feature_schema()
    parts = [atlas._make_rows(
        series,
        PHASE_THRESHOLD,
        INCLUDE_COMMAND_ERRORS,
        INCLUDE_COMMAND_RATES,
        INCLUDE_REAR_WHEEL_SPLIT,
    ) for series in train_series]
    x = np.concatenate([part[0] for part in parts], axis=0)
    target_delta = np.concatenate([part[1] for part in parts], axis=0)
    cells = np.concatenate([part[2] for part in parts], axis=0)
    phases = np.concatenate([part[3] for part in parts], axis=0)
    run_ids = np.concatenate([part[4] for part in parts], axis=0)

    grouped_indices: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for index, (cell, phase) in enumerate(zip(cells, phases)):
        grouped_indices[(int(cell[0]), int(cell[1]), int(phase))].append(index)

    jobs = []
    for key, rows in sorted(grouped_indices.items()):
        indices = np.asarray(rows, dtype=np.int64)
        counts = Counter(run_ids[indices].tolist())
        contributing_runs = {
            run_id for run_id, count in counts.items()
            if count >= atlas.MIN_SAMPLES_PER_RUN
        }
        keep = np.asarray([run_id in contributing_runs
                           for run_id in run_ids[indices]], dtype=bool)
        indices = indices[keep]
        if (len(indices) < atlas.MIN_CELL_SAMPLES
                or len(contributing_runs) < atlas.MIN_CELL_RUNS):
            continue
        jobs.append((key, x[indices], target_delta[indices],
                     run_ids[indices], feature_scales))

    fitted = {}
    for done, (key, model) in enumerate(Parallel(
            n_jobs=workers,
            prefer="threads",
            return_as="generator_unordered",
            batch_size=1,
            pre_dispatch=2 * workers,
    )(delayed(_fit_cell)(job) for job in jobs), 1):
        fitted[key] = model
        if done % 40 == 0 or done == len(jobs):
            print(f"fit {done}/{len(jobs)} exact phase-cells", flush=True)

    training_run_ids = sorted(series.run_id for series in train_series)
    package = {
        "format_version": 1,
        "model_family": MODEL_FAMILY,
        "sample_period_s": atlas.DT_S,
        "phase_threshold_rad2_per_s": PHASE_THRESHOLD,
        "feature_names": list(feature_names),
        "feature_scales": feature_scales,
        "speed_bin_mps": atlas.SPEED_BIN_MPS,
        "steering_bin_rad": atlas.STEERING_BIN_RAD,
        "speed_centers_mps": atlas.SPEED_CENTERS,
        "steering_centers_rad": atlas.STEERING_CENTERS,
        "phase_ids": {"-1": "unwind", "0": "steady_or_low_rate", "1": "turn_in"},
        "minimum_cell_samples": atlas.MIN_CELL_SAMPLES,
        "minimum_cell_runs": atlas.MIN_CELL_RUNS,
        "minimum_samples_per_run": atlas.MIN_SAMPLES_PER_RUN,
        "estimator_config": {
            "n_estimators": N_ESTIMATORS,
            "max_depth": MAX_DEPTH,
            "min_samples_leaf": MIN_SAMPLES_LEAF,
            "max_features": MAX_FEATURES,
            "random_state": RANDOM_STATE,
            "run_balanced_sample_weight": True,
        },
        "training_run_ids": training_run_ids,
        "training_archives": sorted({
            series.source for series in train_series
        }),
        "training_rows": int(sum(len(job[1]) for job in jobs)),
        "models": fitted,
        "test_or_final_test_arrays_read": False,
        "runtime_integration": "none",
        "recursive_plant_validated": False,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "yaw_extratrees_fullband_v1.joblib"
    joblib.dump(package, model_path, compress=3)
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    metadata = {
        "title": "Research-only exact-cell ExtraTrees yaw-transition atlas",
        "model_file": model_path.name,
        "model_sha256": digest,
        "model_family": MODEL_FAMILY,
        "format_version": 1,
        "python_sklearn_version": sklearn.__version__,
        "model_cells": len(fitted),
        "training_run_count": len(training_run_ids),
        "training_run_ids": training_run_ids,
        "training_archives": package["training_archives"],
        "training_rows": package["training_rows"],
        "feature_names": list(feature_names),
        "estimator_config": package["estimator_config"],
        "source_audit": {
            "selected_runs": source_audit["selected_runs"],
            "selected_archives": source_audit["selected_archives"],
            "skipped_archive_reasons": source_audit["skipped_archive_reasons"],
            "test_and_final_test_arrays_read": False,
        },
        "full_cartesian_domain_supported": False,
        "observed_max_speed_mps": float(max(
            np.max(np.hypot(series.rigid[:, 7], series.rigid[:, 8]))
            for series in run_series)),
        "unsupported_cells_filled": False,
        "recursive_plant_validated": False,
        "runtime_integration": "none",
    }
    metadata_path = output_dir / "yaw_extratrees_fullband_v1_manifest.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
    print(f"model: {model_path}")
    print(f"manifest: {metadata_path}")
    print(f"trained cells: {len(fitted)}; train runs: {len(training_run_ids)}")
    return {"package": package, "metadata": metadata,
            "model_path": model_path, "metadata_path": metadata_path}


def predict_transition(package: dict[str, Any], key: tuple[int, int, int],
                       features: np.ndarray) -> float | None:
    """Predict yaw_rate[k+1] or abstain when the exact phase-cell is absent."""
    model_data = package["models"].get(tuple(map(int, key)))
    if model_data is None:
        return None
    x = np.asarray(features, dtype=np.float64)
    if x.shape != (len(package["feature_names"]),) or not np.isfinite(x).all():
        raise ValueError("feature vector does not match the frozen model schema")
    z = (x - model_data["feature_mean"]) / model_data["feature_scales"]
    delta = float(model_data["estimator"].predict(z.reshape(1, -1))[0])
    return float(x[0] + delta)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "live_runs/racing_model_diagnostics_20261007/"
                "fullband_yaw_regime_atlas_v11_lowangle_unwind",
    )
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    fit_candidate(args.output_dir, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
