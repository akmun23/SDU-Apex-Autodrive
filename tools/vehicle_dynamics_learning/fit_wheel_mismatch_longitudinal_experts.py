#!/usr/bin/env python3
"""Test high-wheel/body-mismatch experts against the frozen HGB transition.

The current broad teacher is held fixed as the parent. Candidate specialists
are fitted on whole training runs only, using the parent feature contract and
the observed mismatch breakpoints from the held-out regime audit. Broad
validation selects whether any threshold is worth retaining; practice and
high-steer captures are transfer checks. Nothing is wired into runtime.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.base import clone

from tools.vehicle_dynamics_learning.fit_longitudinal_transition_candidate import (
    REAR_TRACK_M,
    _rmse,
)
from tools.vehicle_dynamics_learning.fit_longitudinal_with_fixed40_wheels import (
    BASE_VARIANT_NAMES,
    DEFAULT_DATASET as TRAIN_DATASET,
    DEFAULT_PRACTICE as PRACTICE_DATASET,
    _augment,
)
from tools.vehicle_dynamics_learning.fit_fixed40_wheel_response_candidate import (
    _prepare as _prepare_raw_wheels,
)
from tools.vehicle_dynamics_learning.train_nssm import _load_dataset


ROOT = Path(__file__).resolve().parents[2]
PARENT_BUNDLE = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "longitudinal_with_fixed40_wheels_candidate_20261009/"
    "longitudinal_raw_wheel_candidate.joblib")
HIGHSTEER_DATASET = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "full_modeling_reset_20261001/"
    "highsteer_75_heldout_validation_plus_r03_20261004/"
    "openplane_dynamics_fixed40hz.npz")
OUTPUT_DIR = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "wheel_mismatch_longitudinal_experts_20261009")
THRESHOLDS_M = (0.5, 1.0, 2.0, 4.0)
SPEED_LIMIT_MPS = 12.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _mismatch(data: dict[str, Any], source: np.ndarray) -> np.ndarray:
    frames = np.asarray(data["frames"], dtype=np.float64)
    rigid = np.asarray(data["simulator_rigid_state"], dtype=np.float64)
    half_track = REAR_TRACK_M / 2.0
    contact_left = rigid[source, 7] - half_track * rigid[source, 12]
    contact_right = rigid[source, 7] + half_track * rigid[source, 12]
    return np.abs(0.5 * (
        frames[source, 5] - contact_left
        + frames[source, 6] - contact_right))


def _collect(data: dict[str, Any], split: str,
             parent: dict[str, Any]) -> dict[str, Any]:
    (source, run_index, target, baseline, delta_target, _, _, features,
     _) = _augment(data, split)
    variant = BASE_VARIANT_NAMES.index(parent["selected_parent_feature_variant"])
    x = features[variant]
    if x.shape[1] != parent["models"][0].n_features_in_:
        raise ValueError("expert features do not match frozen parent model")
    run_ids = np.asarray(data["run_ids"]).astype(str)
    return {
        "source": source,
        "run_index": run_index,
        "run_ids": run_ids,
        "target": target,
        "baseline": baseline,
        "delta_target": delta_target,
        "x": x,
        "mismatch": _mismatch(data, source),
    }


def _predict_parent(data: dict[str, Any], parent: dict[str, Any]
                    ) -> dict[str, Any]:
    features = _collect(data, "validation", parent)
    delta = np.column_stack([
        model.predict(features["x"]) for model in parent["models"]])
    features["parent_prediction"] = features["baseline"] + delta
    return features


def _metrics(error: np.ndarray, run_index: np.ndarray,
             run_ids: np.ndarray) -> dict[str, Any]:
    per_run: dict[str, dict[str, float]] = {}
    for run in np.unique(run_index):
        selected = run_index == run
        per_run[str(run_ids[run])] = {
            "rows": int(selected.sum()),
            "rmse": _rmse(error[selected]),
            "bias": float(np.mean(error[selected])),
        }
    return {
        "runs": len(per_run),
        "rows": int(len(error)),
        "macro_run_rmse": float(np.mean(
            [item["rmse"] for item in per_run.values()])),
        "pooled_rmse": _rmse(error),
        "per_run": per_run,
    }


def _score(features: dict[str, Any], parent_prediction: np.ndarray,
           specialist_delta: np.ndarray | None,
           threshold: float | None) -> dict[str, Any]:
    prediction = parent_prediction.copy()
    active = np.zeros(len(prediction), dtype=bool)
    if specialist_delta is not None and threshold is not None:
        active = features["mismatch"] >= threshold
        prediction[active] = (
            features["baseline"][active] + specialist_delta[active])
    error = prediction - features["target"]
    parent_error = parent_prediction - features["target"]
    gates: dict[str, Any] = {}
    for low, high in ((0.0, 0.5), (0.5, 1.0), (1.0, 2.0),
                      (2.0, 4.0), (4.0, np.inf)):
        selected = ((features["mismatch"] >= low)
                    & (features["mismatch"] < high))
        if not np.any(selected):
            continue
        gates[f"{low:g}-{high:g}" if np.isfinite(high)
              else f">={low:g}"] = {
            "rows": int(selected.sum()),
            "u_parent_rmse": _rmse(parent_error[selected, 0]),
            "u_candidate_rmse": _rmse(error[selected, 0]),
            "ax_parent_rmse": _rmse(parent_error[selected, 1]),
            "ax_candidate_rmse": _rmse(error[selected, 1]),
            "filtered_wheel_pair_parent_rmse": _rmse(
                parent_error[selected, 2:4]),
            "filtered_wheel_pair_candidate_rmse": _rmse(
                error[selected, 2:4]),
        }
    return {
        "active_fraction": float(np.mean(active)),
        "all_rows": {
            "u_parent": _metrics(parent_error[:, 0], features["run_index"],
                                 features["run_ids"]),
            "u_candidate": _metrics(error[:, 0], features["run_index"],
                                    features["run_ids"]),
            "effective_ax_parent": _metrics(
                parent_error[:, 1], features["run_index"], features["run_ids"]),
            "effective_ax_candidate": _metrics(
                error[:, 1], features["run_index"], features["run_ids"]),
            "filtered_wheel_pair_parent": _metrics(
                parent_error[:, 2:4], features["run_index"], features["run_ids"]),
            "filtered_wheel_pair_candidate": _metrics(
                error[:, 2:4], features["run_index"], features["run_ids"]),
        },
        "mismatch_bins": gates,
    }


def _fit_expert(train: dict[str, Any], parent: dict[str, Any],
                threshold: float) -> list[Any]:
    selected = train["mismatch"] >= threshold
    if not np.any(selected):
        raise ValueError(f"no training samples at mismatch >= {threshold:g} m/s")
    run_counts = np.bincount(train["run_index"][selected],
                             minlength=len(train["run_ids"]))
    active_runs = np.unique(train["run_index"][selected])
    if len(active_runs) < 5:
        raise ValueError(
            f"mismatch expert at {threshold:g} m/s is supported by only "
            f"{len(active_runs)} training runs")
    weights = 1.0 / run_counts[train["run_index"][selected]]
    weights *= len(weights) / weights.sum()
    return [
        clone(model).fit(train["x"][selected],
                         train["delta_target"][selected, index],
                         sample_weight=weights)
        for index, model in enumerate(parent["models"])
    ]


def _evaluate_set(data: dict[str, Any], split: str,
                  parent: dict[str, Any], experts: dict[float, list[Any]]
                  ) -> dict[str, Any]:
    features = _collect(data, split, parent)
    parent_delta = np.column_stack([
        model.predict(features["x"]) for model in parent["models"]])
    parent_prediction = features["baseline"] + parent_delta
    result: dict[str, Any] = {
        "split": split,
        "run_ids": sorted(set(features["run_ids"][features["run_index"]].tolist())),
        "rows": int(len(features["source"])),
        "parent": _score(features, parent_prediction, None, None),
        "threshold_candidates": {},
    }
    for threshold, models in experts.items():
        delta = np.column_stack([
            model.predict(features["x"]) for model in models])
        result["threshold_candidates"][str(threshold)] = _score(
            features, parent_prediction, delta, threshold)
    return result


def run() -> dict[str, Any]:
    report_path = OUTPUT_DIR / "evaluation.json"
    model_path = OUTPUT_DIR / "selected_mismatch_expert.joblib"
    if report_path.exists() or model_path.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT_DIR}")
    for path in (PARENT_BUNDLE, TRAIN_DATASET, PRACTICE_DATASET,
                 HIGHSTEER_DATASET):
        if not path.is_file():
            raise FileNotFoundError(path)

    parent = joblib.load(PARENT_BUNDLE)
    train_data = _load_dataset(TRAIN_DATASET)
    train = _collect(train_data, "train", parent)
    training_splits = np.asarray(train_data["splits"]).astype(str)
    training_ids = np.asarray(train_data["run_ids"]).astype(str)
    selected_runs = sorted(set(
        training_ids[train["run_index"]].tolist()))
    if any(training_splits[index] != "train"
           for index in np.unique(train["run_index"])):
        raise ValueError("expert training rows crossed the train split")

    experts: dict[float, list[Any]] = {}
    fit_support: dict[str, Any] = {}
    for threshold in THRESHOLDS_M:
        selected = train["mismatch"] >= threshold
        active = np.unique(train["run_index"][selected])
        experts[threshold] = _fit_expert(train, parent, threshold)
        fit_support[str(threshold)] = {
            "threshold_mps": threshold,
            "rows": int(selected.sum()),
            "run_count": int(len(active)),
            "run_ids": sorted(set(training_ids[active].tolist())),
        }

    cohorts = (
        ("broad_openplane_validation", TRAIN_DATASET, "validation"),
        ("highsteer_validation", HIGHSTEER_DATASET, "validation"),
        ("unseen_practice_diagnostic", PRACTICE_DATASET, "unseen_practice"),
    )
    results: dict[str, Any] = {}
    for label, path, split in cohorts:
        data = _load_dataset(path)
        if split not in set(np.asarray(data["splits"]).astype(str)):
            raise ValueError(f"{path} lacks expected split {split!r}")
        results[label] = _evaluate_set(data, split, parent, experts)

    # Threshold selection is confined to broad whole-run validation. Practice
    # and high-steer results are reported only as transfer evidence.
    broad = results["broad_openplane_validation"]
    candidates: dict[str, Any] = {}
    for threshold in THRESHOLDS_M:
        score = broad["threshold_candidates"][str(threshold)]
        all_rows = score["all_rows"]
        high_bins = [row for key, row in score["mismatch_bins"].items()
                     if key in ("1-2", "2-4", ">=4")]
        parent_high = [row for key, row in broad["parent"]["mismatch_bins"].items()
                       if key in ("1-2", "2-4", ">=4")]
        candidates[str(threshold)] = {
            "validation_u_macro_run_rmse": all_rows["u_candidate"]["macro_run_rmse"],
            "validation_u_parent_macro_run_rmse": all_rows["u_parent"]["macro_run_rmse"],
            "validation_ax_macro_run_rmse": all_rows[
                "effective_ax_candidate"]["macro_run_rmse"],
            "validation_ax_parent_macro_run_rmse": all_rows[
                "effective_ax_parent"]["macro_run_rmse"],
            "validation_wheel_macro_run_rmse": all_rows[
                "filtered_wheel_pair_candidate"]["macro_run_rmse"],
            "validation_wheel_parent_macro_run_rmse": all_rows[
                "filtered_wheel_pair_parent"]["macro_run_rmse"],
            "high_mismatch_u_candidate_macro_bin_rmse": float(np.mean(
                [row["u_candidate_rmse"] for row in high_bins])) if high_bins else None,
            "high_mismatch_u_parent_macro_bin_rmse": float(np.mean(
                [row["u_parent_rmse"] for row in parent_high])) if parent_high else None,
        }

    report = {
        "schema_version": 1,
        "purpose": "training-only high wheel/body-mismatch experts for one-step longitudinal and filtered-wheel transitions",
        "yaw_atlas": "frozen; not modified; this is an isolated one-step u/ax/wheel fit, not a full recursive plant evaluation",
        "test_or_final_test_used": False,
        "simulator_was_run": False,
        "runtime_or_physics_changed": False,
        "fitting_data_split": "train only, grouped by complete run",
        "threshold_selection_split": "broad OpenPlane whole-run validation only",
        "practice_and_highsteer_are_transfer_only": True,
        "parent_bundle_sha256": _sha256(PARENT_BUNDLE),
        "dataset_sha256": {
            str(path.relative_to(ROOT)): _sha256(path)
            for path in (TRAIN_DATASET, PRACTICE_DATASET, HIGHSTEER_DATASET)},
        "training_run_ids": selected_runs,
        "expert_fit_support": fit_support,
        "threshold_candidates": candidates,
        "results": results,
        "candidate_decision": {
            "selected_threshold_mps": None,
            "retain_as_research_model": False,
            "reason": "selection requires held-out validation improvement in the high-mismatch bins without unacceptable full-domain regression; decide from measured results",
        },
        "limitations": [
            "mismatch uses simulator-truth contact speed for diagnostic gating; a deployed plant must compute it from its own predicted state",
            "fixed threshold candidates are the boundaries already exposed by held-out error stratification, not inferred tire-physics limits",
            "one-step transition improvement does not establish stable recursive or lap-length accuracy",
        ],
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return report


def main() -> None:
    report = run()
    print(json.dumps({
        "output": str(OUTPUT_DIR / "evaluation.json"),
        "training_support": report["expert_fit_support"],
        "broad_validation_candidates": report["threshold_candidates"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
