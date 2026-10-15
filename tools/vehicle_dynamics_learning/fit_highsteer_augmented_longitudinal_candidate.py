#!/usr/bin/env python3
"""Refit the fixed-40-Hz longitudinal/wheel heads with existing high-steer data.

The base candidate architecture and feature variant are frozen. Two existing
high-steer training captures are added at equal per-run weight after deriving
their causal encoder-rate sidecar from the original bags. Only training
captures fit the candidate; whole-run, high-steer, and practice datasets score
it. The frozen yaw atlas and EDSSM are not refit or modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from tools.vehicle_dynamics_learning.evaluate_composed_latent_reconciliation import (
    WINDOWS_PER_RUN,
)
from tools.vehicle_dynamics_learning.evaluate_hybrid_longitudinal_edssm_candidate import (
    DT_S,
    EDSSM_CHECKPOINT,
    HGB_BUNDLE,
    HGB_TRAIN_DATASET,
    HIGHSTEER_DATASET,
    PRACTICE_DATASET,
    YAW_ATLAS_BUNDLE,
    _bundle_lags,
    _rollout,
    _windows,
)
from tools.vehicle_dynamics_learning.evaluate_yaw_atlas_recursive_gain import (
    HORIZON,
    _run_windows,
)
from tools.vehicle_dynamics_learning.fit_fixed40_wheel_response_candidate import (
    VARIANT_NAMES as RAW_VARIANTS,
    _prepare as prepare_raw_wheel,
    _score as score_raw_wheel,
)
from tools.vehicle_dynamics_learning.fit_longitudinal_transition_candidate import (
    _balanced_training_rows,
    _fit_variant,
    _predict,
    _run_bootstrap_ci,
    _score_bundle,
)
from tools.vehicle_dynamics_learning.fit_longitudinal_with_fixed40_wheels import (
    BASE_VARIANT_NAMES,
    _augment,
)
from tools.vehicle_dynamics_learning.run_truncated_history_transition import ROOT
from tools.vehicle_dynamics_learning.score_effective_race_teacher import _load_model
from tools.vehicle_dynamics_learning.train_nssm import _load_dataset
from tools.racing.specialists import fit_sensor_only_yaw_regime_atlas as yaw_atlas


HIGHSTEER_TRAIN_DATASET = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "full_modeling_reset_20261001/highsteer_75_train_expanded_20261004/"
    "encoder_fixed40hz_v1/openplane_dynamics_raw_wheels_fixed40hz.npz")
OUTPUT_DIR = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "longitudinal_with_fixed40_wheels_candidate_20261009/"
    "highsteer_augmented_20261009")
CAP_PER_RUN = 8000
MAX_ITER = 220
THREADS = 2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _fit_rows(data: dict[str, Any], run_offset: int,
              feature_index: int, raw_feature_index: int
              ) -> dict[str, Any]:
    source, run_index, target, baseline, model_target, _, _, features, names = (
        _augment(data, "train"))
    state7 = np.column_stack((
        np.asarray(data["simulator_rigid_state"], dtype=np.float32)[:, (7, 8, 12)],
        np.asarray(data["frames"], dtype=np.float32)[:, 3:7]))
    aux_target = (state7[source + 1][:, 1:5] - state7[source][:, 1:5])
    if not np.isfinite(aux_target).all():
        raise ValueError("non-finite actuator/lateral training labels")

    raw_source, raw_run, raw_current, raw_target, raw_features, _, raw_names = (
        prepare_raw_wheel(data, "train"))
    if not len(raw_source):
        raise ValueError("no valid raw-wheel training transitions")
    if len(raw_names[raw_feature_index]) != raw_features[raw_feature_index].shape[1]:
        raise ValueError("raw-wheel feature schema is inconsistent")
    return {
        "main_source": source,
        "main_run": run_index.astype(np.int64) + run_offset,
        "main_target": model_target,
        "main_baseline": baseline,
        "main_features": features[feature_index],
        "aux_target": aux_target,
        "raw_run": raw_run.astype(np.int64) + run_offset,
        "raw_current": raw_current,
        "raw_target": raw_target,
        "raw_features": raw_features[raw_feature_index],
        "main_feature_names": names[feature_index],
        "raw_feature_names": raw_names[raw_feature_index],
    }


def _paired_comparison(candidate: dict[str, Any], reference: dict[str, Any],
                       metrics: tuple[str, ...]) -> dict[str, Any]:
    common = sorted(set(candidate["per_run"]) & set(reference["per_run"]))
    result = {}
    for metric in metrics:
        result[metric] = {
            "candidate_macro_run_rmse": candidate["macro_run_metrics"][metric],
            "reference_macro_run_rmse": reference["macro_run_metrics"][metric],
            "paired_run_bootstrap_95pct": _run_bootstrap_ci(
                candidate["per_run"], reference["per_run"], metric),
        }
    result["common_runs"] = common
    return result


def _score_one_step(data: dict[str, Any], split: str,
                    reference: dict[str, Any], candidate: dict[str, Any],
                    main_index: int, raw_index: int) -> dict[str, Any]:
    source, run_index, target, baseline, _, _, _, features, names = _augment(
        data, split)
    if tuple(names[main_index]) != tuple(candidate["feature_names"]):
        raise ValueError("evaluation main-feature contract mismatch")
    run_ids = np.asarray(data["run_ids"]).astype(str)
    reference_main, _ = _score_bundle(
        reference["models"], features[main_index], target, baseline,
        run_index, run_ids)
    candidate_main, _ = _score_bundle(
        candidate["models"], features[main_index], target, baseline,
        run_index, run_ids)

    raw_source, raw_run, raw_current, raw_target, raw_features, _, raw_names = (
        prepare_raw_wheel(data, split))
    if tuple(raw_names[raw_index]) != tuple(candidate["raw_wheel_feature_names"]):
        raise ValueError("evaluation raw-wheel feature contract mismatch")
    raw_baseline = raw_current.astype(np.float64)
    raw_reference_prediction = raw_baseline + _predict(
        reference["raw_wheel_models"], raw_features[raw_index])
    raw_candidate_prediction = raw_baseline + _predict(
        candidate["raw_wheel_models"], raw_features[raw_index])
    reference_raw = score_raw_wheel(
        raw_reference_prediction, raw_target, raw_baseline, raw_run, run_ids)
    candidate_raw = score_raw_wheel(
        raw_candidate_prediction, raw_target, raw_baseline, raw_run, run_ids)
    return {
        "main_rows": int(len(source)),
        "raw_wheel_rows": int(len(raw_source)),
        "reference_main": reference_main,
        "candidate_main": candidate_main,
        "main_paired_comparison": _paired_comparison(
            candidate_main, reference_main,
            ("u_rmse_mps", "ax_rmse_mps2", "rear_wheel_pair_rmse_mps")),
        "reference_raw_wheels": reference_raw,
        "candidate_raw_wheels": candidate_raw,
        "raw_wheel_paired_comparison": {
            "pair_rmse_mps": {
                "candidate_macro_run_rmse": candidate_raw["macro_run_metrics"][
                    "pair_rmse_mps"],
                "reference_macro_run_rmse": reference_raw["macro_run_metrics"][
                    "pair_rmse_mps"],
                "paired_run_bootstrap_95pct": _run_bootstrap_ci(
                    candidate_raw["per_run"], reference_raw["per_run"],
                    "pair_rmse_mps"),
            },
            "common_runs": sorted(set(candidate_raw["per_run"])
                                   & set(reference_raw["per_run"])),
        },
    }


def run(output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    required = (HGB_BUNDLE, EDSSM_CHECKPOINT, YAW_ATLAS_BUNDLE,
                HGB_TRAIN_DATASET, HIGHSTEER_TRAIN_DATASET,
                HIGHSTEER_DATASET, PRACTICE_DATASET)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    started = time.time()
    reference = joblib.load(HGB_BUNDLE)
    main_index = BASE_VARIANT_NAMES.index(reference["selected_variant"])
    raw_index = RAW_VARIANTS.index(reference["raw_wheel_feature_variant"])
    base_data = _load_dataset(HGB_TRAIN_DATASET)
    extra_data = _load_dataset(HIGHSTEER_TRAIN_DATASET)
    extra_splits = set(np.asarray(extra_data["splits"]).astype(str))
    if extra_splits != {"train"}:
        raise ValueError("the added high-steer source must contain train runs only")
    base_run_ids = np.asarray(base_data["run_ids"]).astype(str)
    extra_run_ids = np.asarray(extra_data["run_ids"]).astype(str)
    if set(base_run_ids) & set(extra_run_ids):
        raise ValueError("added high-steer training runs overlap the base dataset")
    combined_run_ids = np.concatenate((base_run_ids, extra_run_ids))

    base_rows = _fit_rows(base_data, 0, main_index, raw_index)
    extra_rows = _fit_rows(data=extra_data, run_offset=len(base_run_ids),
                           feature_index=main_index,
                           raw_feature_index=raw_index)
    if (base_rows["main_feature_names"] != extra_rows["main_feature_names"]
            or base_rows["raw_feature_names"] != extra_rows["raw_feature_names"]):
        raise ValueError("added data do not match the frozen model feature contract")
    main_run = np.concatenate((base_rows["main_run"], extra_rows["main_run"]))
    main_features = np.concatenate((base_rows["main_features"],
                                    extra_rows["main_features"]))
    main_target = np.concatenate((base_rows["main_target"],
                                  extra_rows["main_target"]))
    aux_target = np.concatenate((base_rows["aux_target"],
                                 extra_rows["aux_target"]))
    rows, weights, train_counts = _balanced_training_rows(
        main_run, combined_run_ids, CAP_PER_RUN, seed=20261009)
    raw_run = np.concatenate((base_rows["raw_run"], extra_rows["raw_run"]))
    raw_features = np.concatenate((base_rows["raw_features"],
                                   extra_rows["raw_features"]))
    raw_target_delta = np.concatenate((
        base_rows["raw_target"] - base_rows["raw_current"],
        extra_rows["raw_target"] - extra_rows["raw_current"]))
    raw_rows, raw_weights, raw_counts = _balanced_training_rows(
        raw_run, combined_run_ids, CAP_PER_RUN, seed=20261010)

    with threadpool_limits(limits=THREADS):
        main_models = _fit_variant(main_features, main_target, rows, weights,
                                   MAX_ITER)
        auxiliary_models = _fit_variant(main_features, aux_target, rows, weights,
                                        MAX_ITER)
        raw_models = _fit_variant(raw_features, raw_target_delta,
                                  raw_rows, raw_weights, MAX_ITER)

    candidate = dict(reference)
    candidate.update({
        "models": main_models,
        "recursive_aux_models": auxiliary_models,
        "raw_wheel_models": raw_models,
        "training_run_ids": sorted(set(base_run_ids[
            np.asarray(base_data["splits"]).astype(str) == "train"]) |
            set(extra_run_ids)),
        "training_dataset_sha256": _sha256(HGB_TRAIN_DATASET),
        "additional_training_datasets": [{
            "path": str(HIGHSTEER_TRAIN_DATASET.relative_to(ROOT)),
            "sha256": _sha256(HIGHSTEER_TRAIN_DATASET),
            "run_ids": sorted(extra_run_ids),
        }],
        "research_only": True,
        "production_integration_allowed": False,
    })
    candidate["raw_wheel_feature_names"] = extra_rows["raw_feature_names"]

    one_step_specs = (
        ("whole_run_validation", base_data, "validation"),
        ("highsteer_validation", _load_dataset(HIGHSTEER_DATASET), "validation"),
        ("unseen_practice_diagnostic", _load_dataset(PRACTICE_DATASET),
         "unseen_practice"),
    )
    one_step: dict[str, Any] = {}
    for name, data, split in one_step_specs:
        one_step[name] = _score_one_step(
            data, split, reference, candidate, main_index, raw_index)

    hgb_train_ids = set(reference["training_run_ids"])
    candidate_train_ids = set(candidate["training_run_ids"])
    edssm_train_ids = set(_load_model(EDSSM_CHECKPOINT, "cpu")[2]["training_runs"])
    atlas_series, _ = yaw_atlas.teacher._discover_series_with_safe_mixed_archives()
    atlas_train_ids = {series.run_id for series in atlas_series
                       if series.split == "train"}
    eval_data_by_name = {name: data for name, data, _ in one_step_specs}
    eval_windows = {
        "whole_run_validation": _run_windows(
            base_data, "validation", _bundle_lags(candidate), WINDOWS_PER_RUN),
        "highsteer_validation": _windows(
            eval_data_by_name["highsteer_validation"], HIGHSTEER_DATASET,
            HORIZON, _bundle_lags(candidate)),
        "unseen_practice_diagnostic": _windows(
            eval_data_by_name["unseen_practice_diagnostic"], PRACTICE_DATASET,
            HORIZON, _bundle_lags(candidate)),
    }
    recursive: dict[str, Any] = {}
    atlas = joblib.load(YAW_ATLAS_BUNDLE)
    for models in (atlas["global_event_models"].values(),
                   atlas["local_expert_models"].values()):
        for model in models:
            if hasattr(model, "n_jobs"):
                model.n_jobs = 1
    _, edssm, edssm_metadata = _load_model(EDSSM_CHECKPOINT, "cpu")
    for name, _, split in one_step_specs:
        data = eval_data_by_name[name]
        selected = eval_windows[name]
        ids = np.asarray(data["run_ids"]).astype(str)
        selected_ids = {str(ids[row["run"]]) for row in selected}
        if selected_ids & (candidate_train_ids | edssm_train_ids | atlas_train_ids):
            raise ValueError(f"recursive held-out split overlaps a fitted model: {name}")
        if any(str(data["splits"][row["run"]]) != split for row in selected):
            raise ValueError(f"unexpected split selected for {name}")
        baseline_result = _rollout(
            data, reference, edssm, edssm_metadata, atlas, selected, HORIZON,
            include_oracle_diagnostics=False,
            reconcile_composed_latent=True,
            latent_acceleration_scheme="current",
            atlas_acceleration_source="model_heads")
        candidate_result = _rollout(
            data, candidate, edssm, edssm_metadata, atlas, selected, HORIZON,
            include_oracle_diagnostics=False,
            reconcile_composed_latent=True,
            latent_acceleration_scheme="current",
            atlas_acceleration_source="model_heads")
        recursive[name] = {
            "windows": len(selected),
            "windows_by_run": candidate_result["windows_by_run"],
            "heldout_run_ids": sorted(selected_ids),
            "reference": baseline_result["macro_run_rmse"],
            "candidate": candidate_result["macro_run_rmse"],
            "candidate_per_run": candidate_result["per_run"],
        }

    output_dir.mkdir(parents=True, exist_ok=False)
    model_path = output_dir / "longitudinal_raw_wheel_candidate.joblib"
    joblib.dump(candidate, model_path)
    report = {
        "schema_version": 1,
        "purpose": "measure whether reusing existing high-steer training captures improves the fixed-40-Hz longitudinal and wheel heads",
        "research_only": True,
        "runtime_or_production_integration": False,
        "simulator_was_run": False,
        "test_or_final_test_used": False,
        "fit_policy": {
            "base_architecture": "frozen existing HGB feature contract and hyperparameters",
            "fit_split": "train only",
            "base_training_runs": sorted(set(base_run_ids[
                np.asarray(base_data["splits"]).astype(str) == "train"])),
            "added_training_runs": sorted(extra_run_ids),
            "balanced_weighting": "equal total sample weight per source run",
            "max_rows_per_run": CAP_PER_RUN,
            "feature_variant_reselected": False,
            "feature_variant": reference["selected_variant"],
            "fixed_dt_s": DT_S,
            "raw_encoder_validity_handled_as_label_and_input_gate": True,
            "validation_test_final_test_used_for_fit": False,
        },
        "training_rows_by_run": train_counts,
        "raw_wheel_training_rows_by_run": raw_counts,
        "input_sha256": {str(path.relative_to(ROOT)): _sha256(path)
                          for path in required},
        "candidate_bundle": str(model_path.relative_to(ROOT)),
        "candidate_bundle_sha256": _sha256(model_path),
        "one_step_validation": one_step,
        "recursive_five_second_validation": recursive,
        "yaw_atlas": {
            "bundle": str(YAW_ATLAS_BUNDLE.relative_to(ROOT)),
            "sha256": _sha256(YAW_ATLAS_BUNDLE),
            "refit": False,
        },
        "edssm": {"checkpoint": str(EDSSM_CHECKPOINT.relative_to(ROOT)),
                  "sha256": _sha256(EDSSM_CHECKPOINT), "refit": False},
        "training_run_overlap": {
            "base_candidate_training_runs": sorted(hgb_train_ids),
            "new_candidate_additions": sorted(candidate_train_ids - hgb_train_ids),
        },
        "elapsed_seconds": time.time() - started,
        "limitations": [
            "two added high-steer runs provide limited independent training variation",
            "recursive scoring remains a five-second seeded rollout, not full-lap validation",
            "the frozen EDSSM lateral/roll component may still dominate long-horizon divergence",
            "no MPC, odometry, AMCL, runtime physics, or controller configuration was changed",
        ],
    }
    (output_dir / "evaluation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    report = run(args.output_dir.resolve())
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "training_runs_added": report["fit_policy"]["added_training_runs"],
        "one_step": {
            name: {
                "u": row["main_paired_comparison"]["u_rmse_mps"],
                "ax": row["main_paired_comparison"]["ax_rmse_mps2"],
                "filtered_wheels": row["main_paired_comparison"][
                    "rear_wheel_pair_rmse_mps"],
                "raw_wheels": row["raw_wheel_paired_comparison"][
                    "pair_rmse_mps"],
            } for name, row in report["one_step_validation"].items()},
        "recursive_5s": {
            name: {"reference": values["reference"]["5.000s"],
                   "candidate": values["candidate"]["5.000s"]}
            for name, values in report["recursive_five_second_validation"].items()},
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
