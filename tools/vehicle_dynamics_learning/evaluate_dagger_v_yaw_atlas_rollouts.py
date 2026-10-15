#!/usr/bin/env python3
"""Evaluate the held-out DAgger v head inside the frozen yaw-atlas plant."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from tools.vehicle_dynamics_learning import evaluate_hybrid_longitudinal_edssm_candidate as evaluator
from tools.vehicle_dynamics_learning.evaluate_edssm_yaw_atlas_composition import (
    HORIZON_BY_COHORT,
    OUTPUT_DIR as COMPOSITION_DIR,
)
from tools.vehicle_dynamics_learning.evaluate_extended_yaw_atlas_plant_rollouts import (
    MAX_WINDOWS_PER_RUN,
    _sha256,
    _trace_curves,
)
from tools.vehicle_dynamics_learning.evaluate_yaw_atlas_recursive_gain import (
    _run_windows,
)
from tools.vehicle_dynamics_learning.fit_recursive_yaw_atlas_dagger_corrector import (
    CORRECTION_START_STEP,
    HighSteerYawResidual,
)
from tools.vehicle_dynamics_learning.fit_highsteer_lateral_acceleration_atlas_candidate import (
    LOCAL_GATE as HIGHSTEER_GATE,
)
from tools.vehicle_dynamics_learning.run_truncated_history_transition import ROOT
from tools.vehicle_dynamics_learning.score_effective_race_teacher import _load_model
from tools.vehicle_dynamics_learning.train_nssm import _load_dataset


DAgger_V_BUNDLE = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "lateral_dagger_head_isolation_20261009/"
    "parent_with_isolated_dagger_v_head.joblib")
YAW_CORRECTOR_PATH = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "hybrid_fixed40_longitudinal_yaw_atlas_candidate_20261009/"
    "recursive_yaw_atlas_dagger_highsteer_gated_v3_20261009.joblib")
OUTPUT_DIR = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "dagger_v_yaw_atlas_composition_20261009")


def _score(data: dict[str, Any], hgb: dict[str, Any], edssm: Any,
           metadata: dict[str, Any], atlas: dict[str, Any], yaw_model: Any,
           windows: list[dict[str, int]], horizon: int,
           gate: dict[str, tuple[float, float]] | None = None,
           lateral_v_update: str = "hgb_v_increment",
           fallback_model: Any | None = None
           ) -> dict[str, Any]:
    evaluator.HORIZONS = tuple(step for step in (30, 80, 200, 400, 480)
                               if step <= horizon)
    result = evaluator._rollout(
        data, hgb, edssm, metadata, atlas, windows, horizon,
        yaw_delta_gain=1.0,
        include_oracle_diagnostics=False,
        reconcile_composed_latent=True,
        atlas_acceleration_source="composed_transition",
        yaw_correction_model=yaw_model,
        yaw_correction_start_step=CORRECTION_START_STEP,
        lateral_v_update=lateral_v_update,
        hgb_main_gate=gate,
        hgb_v_fallback_model=fallback_model,
        capture_trajectory_trace=True)
    result["state_error_curve_macro_run_rmse"] = _trace_curves(
        result.pop("_trajectory_trace"),
        np.asarray(data["run_ids"]).astype(str))
    return result


def run() -> dict[str, Any]:
    output = OUTPUT_DIR / "evaluation.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    reference_path = COMPOSITION_DIR / "evaluation.json"
    required = (reference_path, evaluator.HGB_BUNDLE, evaluator.EDSSM_CHECKPOINT,
                evaluator.YAW_ATLAS_BUNDLE, DAgger_V_BUNDLE,
                YAW_CORRECTOR_PATH, evaluator.HGB_TRAIN_DATASET,
                evaluator.HIGHSTEER_DATASET, evaluator.PRACTICE_DATASET)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    hgb = joblib.load(evaluator.HGB_BUNDLE)
    dagger_v = joblib.load(DAgger_V_BUNDLE)
    if dagger_v.get("training_dataset_sha256") != hgb.get(
            "training_dataset_sha256"):
        raise ValueError("DAgger v head and HGB plant use different training data")
    if dagger_v.get("selected_parent_feature_variant") != hgb.get(
            "selected_parent_feature_variant"):
        raise ValueError("DAgger v head feature contract differs from plant")
    hgb["recursive_aux_models"] = list(hgb["recursive_aux_models"])
    hgb["recursive_aux_models"][0] = dagger_v["recursive_aux_models"][0]

    atlas = joblib.load(evaluator.YAW_ATLAS_BUNDLE)
    for model in (list(atlas["global_event_models"].values())
                  + list(atlas["local_expert_models"].values())):
        if hasattr(model, "n_jobs"):
            model.n_jobs = 1
    yaw_bundle = joblib.load(YAW_CORRECTOR_PATH)
    yaw_model = HighSteerYawResidual(
        yaw_bundle["estimator"],
        tuple(yaw_bundle["gate"]["rear_wheel_mean_mps"]))
    _, edssm, metadata = _load_model(evaluator.EDSSM_CHECKPOINT, "cpu")

    cohorts = (
        ("broad_openplane_validation", evaluator.HGB_TRAIN_DATASET,
         "validation", False),
        ("highsteer_validation", evaluator.HIGHSTEER_DATASET,
         "validation", True),
        ("unseen_practice_diagnostic", evaluator.PRACTICE_DATASET,
         "unseen_practice", False),
    )
    results: dict[str, Any] = {}
    for label, path, split, highsteer in cohorts:
        data = _load_dataset(path)
        horizon = HORIZON_BY_COHORT[label]
        if highsteer:
            windows = evaluator._windows(
                data, path, horizon, evaluator._bundle_lags(hgb))
        else:
            windows = _run_windows(
                data, split, evaluator._bundle_lags(hgb),
                MAX_WINDOWS_PER_RUN, horizon_steps=horizon)
        bounds = np.asarray(data["bounds"], dtype=np.int64)
        windows = [row for row in windows
                   if row["start"] + horizon < bounds[row["sequence"], 1]]
        prior = reference["results"][label]
        if (len(windows) != prior["windows"]
                or sorted({str(data["run_ids"][row["run"]]) for row in windows})
                != prior["run_ids"]):
            raise ValueError(f"held-out windows differ from reference for {label}")
        print(f"{label}: DAgger v + frozen yaw atlas, {len(windows)} windows",
              flush=True)
        candidate = _score(data, hgb, edssm, metadata, atlas, yaw_model,
                           windows, horizon)
        results[label] = {
            "split": split,
            "run_ids": prior["run_ids"],
            "windows": len(windows),
            "horizon_s": horizon * evaluator.DT_S,
            "baseline_hgb_edssm_frozen_yaw_atlas": prior[
                "hgb_edssm_yaw_atlas_reference"],
            "dagger_v_hgb_edssm_frozen_yaw_atlas": candidate,
        }

    report = {
        "schema_version": 1,
        "purpose": "test whether the held-out DAgger-trained next-v head improves the same causal whole-run composition while keeping the frozen yaw atlas",
        "test_or_final_test_used": False,
        "simulator_was_run": False,
        "runtime_or_physics_changed": False,
        "yaw_atlas_frozen": True,
        "yaw_corrector_frozen": True,
        "dagger_v_head_frozen": True,
        "changed_component_only": "recursive_aux_models[0] (next-v increment)",
        "future_sensor_or_truth_input": False,
        "comparison_reference_sha256": _sha256(reference_path),
        "input_sha256": {
            str(path.relative_to(ROOT)): _sha256(path) for path in required},
        "results": results,
        "limitations": [
            "initial measured histories are simulator truth; later rollout steps are causal",
            "practice contributes two five-second windows; it is a transfer diagnostic, not a broad statistical sample",
            "this is an offline plant-component comparison, not odometry/MPC integration or full-lap validation",
        ],
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return report


def _paired_run_deltas(candidate: dict[str, Any], baseline: dict[str, Any],
                       horizon_s: float) -> dict[str, Any]:
    horizon = f"{horizon_s:.3f}s"
    candidate_runs, baseline_runs = candidate["per_run"], baseline["per_run"]
    run_ids = sorted(set(candidate_runs) & set(baseline_runs))
    metrics = ("position_rmse_m", "u_rmse_mps", "v_rmse_mps",
               "yaw_rate_rmse_rps", "filtered_rear_wheel_pair_rmse_mps")
    rng = np.random.default_rng(20261011)
    output: dict[str, Any] = {"horizon": horizon, "run_ids": run_ids}
    for metric in metrics:
        deltas = np.asarray([
            candidate_runs[run][horizon]["hybrid_atlas"][metric]
            - baseline_runs[run][horizon]["hybrid_atlas"][metric]
            for run in run_ids], dtype=np.float64)
        draws = rng.choice(deltas, size=(20000, len(deltas)), replace=True).mean(1)
        output[metric] = {
            "candidate_minus_parent_mean": float(deltas.mean()),
            "paired_run_bootstrap_95pct": [
                float(value) for value in np.quantile(draws, (0.025, 0.975))],
            "per_run_delta": {
                run: float(delta) for run, delta in zip(run_ids, deltas)},
        }
    return output


def run_parent_fallback_gate() -> dict[str, Any]:
    """Use the DAgger head only in measured high-steer support; otherwise parent HGB v."""
    output = OUTPUT_DIR / "matched_parent_fallback_gate.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    matched_path = OUTPUT_DIR / "matched_evaluation.json"
    reference_path = COMPOSITION_DIR / "evaluation.json"
    required = (matched_path, reference_path, evaluator.HGB_BUNDLE,
                evaluator.EDSSM_CHECKPOINT, evaluator.YAW_ATLAS_BUNDLE,
                DAgger_V_BUNDLE, YAW_CORRECTOR_PATH,
                evaluator.HIGHSTEER_DATASET, evaluator.PRACTICE_DATASET)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    matched = json.loads(matched_path.read_text(encoding="utf-8"))
    parent_hgb = joblib.load(evaluator.HGB_BUNDLE)
    candidate_hgb = dict(parent_hgb)
    dagger_v = joblib.load(DAgger_V_BUNDLE)
    if dagger_v.get("training_dataset_sha256") != parent_hgb.get(
            "training_dataset_sha256"):
        raise ValueError("DAgger v head and parent use different training data")
    if dagger_v.get("selected_parent_feature_variant") != parent_hgb.get(
            "selected_parent_feature_variant"):
        raise ValueError("DAgger v head feature contract differs from parent")
    candidate_hgb["recursive_aux_models"] = list(
        parent_hgb["recursive_aux_models"])
    candidate_hgb["recursive_aux_models"][0] = dagger_v[
        "recursive_aux_models"][0]
    atlas = joblib.load(evaluator.YAW_ATLAS_BUNDLE)
    for model in (list(atlas["global_event_models"].values())
                  + list(atlas["local_expert_models"].values())):
        if hasattr(model, "n_jobs"):
            model.n_jobs = 1
    yaw_bundle = joblib.load(YAW_CORRECTOR_PATH)
    yaw_model = HighSteerYawResidual(
        yaw_bundle["estimator"],
        tuple(yaw_bundle["gate"]["rear_wheel_mean_mps"]))
    _, edssm, metadata = _load_model(evaluator.EDSSM_CHECKPOINT, "cpu")

    cohorts = (
        ("highsteer_validation", evaluator.HIGHSTEER_DATASET, "validation", True),
        ("unseen_practice_diagnostic", evaluator.PRACTICE_DATASET,
         "unseen_practice", False),
    )
    results: dict[str, Any] = {}
    for label, path, split, highsteer in cohorts:
        data = _load_dataset(path)
        horizon = HORIZON_BY_COHORT[label]
        windows = (evaluator._windows(
            data, path, horizon, evaluator._bundle_lags(parent_hgb))
            if highsteer else _run_windows(
                data, split, evaluator._bundle_lags(parent_hgb),
                MAX_WINDOWS_PER_RUN, horizon_steps=horizon))
        bounds = np.asarray(data["bounds"], dtype=np.int64)
        windows = [row for row in windows
                   if row["start"] + horizon < bounds[row["sequence"], 1]]
        prior = matched["results"][label]
        run_ids = sorted({str(data["run_ids"][row["run"]]) for row in windows})
        if len(windows) != prior["windows"] or run_ids != prior["run_ids"]:
            raise ValueError(f"matched evaluation window mismatch for {label}")
        print(f"{label}: DAgger-v with parent-HGB fallback, "
              f"{len(windows)} windows", flush=True)
        candidate = _score(
            data, candidate_hgb, edssm, metadata, atlas, yaw_model,
            windows, horizon, gate=HIGHSTEER_GATE,
            lateral_v_update="gated_hgb_v_increment_parent_fallback",
            fallback_model=parent_hgb["recursive_aux_models"][0])
        parent = prior["same_parent_hgb_v"]
        dagger_global = prior["dagger_v_global"]
        results[label] = {
            "split": split,
            "run_ids": run_ids,
            "windows": len(windows),
            "horizon_s": horizon * evaluator.DT_S,
            "gate": HIGHSTEER_GATE,
            "gate_active_fraction": candidate[
                "hgb_main_gate_active_fraction"],
            "parent_hgb_v": parent,
            "dagger_v_global_reference": dagger_global,
            "dagger_v_with_parent_hgb_fallback": candidate,
            "paired_run_delta_vs_parent_hgb_v": _paired_run_deltas(
                candidate, parent, horizon * evaluator.DT_S),
        }

    report = {
        "schema_version": 1,
        "purpose": "test the localized DAgger-v specialist while preserving the parent HGB-v model outside measured support",
        "changed_component_only": "next-v increment head selection; DAgger head inside 7-9 m/s and |steering| 0.30-0.525 rad, parent HGB v head elsewhere",
        "yaw_atlas_frozen": True,
        "yaw_corrector_frozen": True,
        "test_or_final_test_used": False,
        "simulator_was_run": False,
        "runtime_or_physics_changed": False,
        "future_sensor_or_truth_input": False,
        "comparison_sha256": _sha256(matched_path),
        "input_sha256": {
            str(path.relative_to(ROOT)): _sha256(path) for path in required},
        "results": results,
        "limitations": [
            "the support rectangle is based on existing measured high-steer data and evaluated using predicted state during rollout",
            "high-steer validation has three runs; practice has two five-second windows",
            "the frozen yaw atlas remains recursive and may dominate position error even if v improves",
        ],
    }
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return report


def run_dagger_uv_composition() -> dict[str, Any]:
    """Evaluate the already-isolated DAgger u/v heads with atlas yaw."""
    output = OUTPUT_DIR / "matched_uv_evaluation.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    matched_path = OUTPUT_DIR / "matched_evaluation.json"
    uv_bundle_path = ROOT / (
        "live_runs/derived_dynamics_learning_20260928/"
        "longitudinal_dagger_uv_head_isolation_20261009/"
        "parent_with_isolated_dagger_u_v_heads.joblib")
    required = (matched_path, evaluator.EDSSM_CHECKPOINT,
                evaluator.YAW_ATLAS_BUNDLE, YAW_CORRECTOR_PATH,
                evaluator.HGB_BUNDLE, DAgger_V_BUNDLE, uv_bundle_path,
                evaluator.HGB_TRAIN_DATASET, evaluator.HIGHSTEER_DATASET,
                evaluator.PRACTICE_DATASET)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    matched = json.loads(matched_path.read_text(encoding="utf-8"))
    parent_hgb = joblib.load(evaluator.HGB_BUNDLE)
    candidate_hgb = dict(parent_hgb)
    uv_bundle = joblib.load(uv_bundle_path)
    for bundle_name, target in (("training_dataset_sha256", "training dataset"),
                                ("selected_parent_feature_variant", "feature variant")):
        if uv_bundle.get(bundle_name) != parent_hgb.get(bundle_name):
            raise ValueError(f"isolated DAgger u/v {target} differs from parent")
    if (uv_bundle["models"][0].n_features_in_
            != parent_hgb["models"][0].n_features_in_
            or uv_bundle["recursive_aux_models"][0].n_features_in_
            != parent_hgb["recursive_aux_models"][0].n_features_in_):
        raise ValueError("isolated DAgger u/v feature widths differ from parent")
    candidate_hgb["models"] = list(parent_hgb["models"])
    candidate_hgb["models"][0] = uv_bundle["models"][0]
    candidate_hgb["recursive_aux_models"] = list(
        parent_hgb["recursive_aux_models"])
    candidate_hgb["recursive_aux_models"][0] = uv_bundle[
        "recursive_aux_models"][0]

    atlas = joblib.load(evaluator.YAW_ATLAS_BUNDLE)
    for model in (list(atlas["global_event_models"].values())
                  + list(atlas["local_expert_models"].values())):
        if hasattr(model, "n_jobs"):
            model.n_jobs = 1
    yaw_bundle = joblib.load(YAW_CORRECTOR_PATH)
    yaw_model = HighSteerYawResidual(
        yaw_bundle["estimator"],
        tuple(yaw_bundle["gate"]["rear_wheel_mean_mps"]))
    _, edssm, metadata = _load_model(evaluator.EDSSM_CHECKPOINT, "cpu")
    cohorts = (
        ("broad_openplane_validation", evaluator.HGB_TRAIN_DATASET,
         "validation", False),
        ("highsteer_validation", evaluator.HIGHSTEER_DATASET,
         "validation", True),
        ("unseen_practice_diagnostic", evaluator.PRACTICE_DATASET,
         "unseen_practice", False),
    )
    results: dict[str, Any] = {}
    for label, path, split, highsteer in cohorts:
        data = _load_dataset(path)
        horizon = HORIZON_BY_COHORT[label]
        windows = (evaluator._windows(
            data, path, horizon, evaluator._bundle_lags(parent_hgb))
            if highsteer else _run_windows(
                data, split, evaluator._bundle_lags(parent_hgb),
                MAX_WINDOWS_PER_RUN, horizon_steps=horizon))
        bounds = np.asarray(data["bounds"], dtype=np.int64)
        windows = [row for row in windows
                   if row["start"] + horizon < bounds[row["sequence"], 1]]
        prior = matched["results"][label]
        run_ids = sorted({str(data["run_ids"][row["run"]]) for row in windows})
        if len(windows) != prior["windows"] or run_ids != prior["run_ids"]:
            raise ValueError(f"matched evaluation window mismatch for {label}")
        print(f"{label}: isolated DAgger u/v + frozen yaw atlas, "
              f"{len(windows)} windows", flush=True)
        candidate = _score(
            data, candidate_hgb, edssm, metadata, atlas, yaw_model,
            windows, horizon, lateral_v_update="hgb_v_increment")
        parent = prior["same_parent_hgb_v"]
        dagger_v = prior["dagger_v_global"]
        results[label] = {
            "split": split,
            "run_ids": run_ids,
            "windows": len(windows),
            "horizon_s": horizon * evaluator.DT_S,
            "same_parent_hgb_uv": parent,
            "dagger_v_only_reference": dagger_v,
            "dagger_uv_candidate": candidate,
            "paired_run_delta_vs_parent_uv": _paired_run_deltas(
                candidate, parent, horizon * evaluator.DT_S),
            "paired_run_delta_vs_dagger_v_only": _paired_run_deltas(
                candidate, dagger_v, horizon * evaluator.DT_S),
        }
    report = {
        "schema_version": 1,
        "purpose": "test whether the existing coupled DAgger u/v heads improve the frozen-yaw-atlas plant when used together",
        "changed_components_only": ["HGB u increment", "HGB v increment"],
        "yaw_atlas_frozen": True,
        "yaw_corrector_frozen": True,
        "test_or_final_test_used": False,
        "simulator_was_run": False,
        "runtime_or_physics_changed": False,
        "future_sensor_or_truth_input": False,
        "matched_comparison_sha256": _sha256(matched_path),
        "input_sha256": {
            str(path.relative_to(ROOT)): _sha256(path) for path in required},
        "results": results,
        "limitations": [
            "the model was trained only on training captures; validation and practice are held out",
            "practice contributes two five-second windows; high-steer contributes three runs",
            "the yaw atlas remains sensor-history recursive, so a good u/v result does not establish stable yaw or full-lap position",
        ],
    }
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return report


def run_matched_comparison() -> dict[str, Any]:
    """Recompute all three v-head variants against one exact parent plant."""
    output = OUTPUT_DIR / "matched_evaluation.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    reference_path = COMPOSITION_DIR / "evaluation.json"
    required = (reference_path, evaluator.HGB_BUNDLE,
                evaluator.EDSSM_CHECKPOINT, evaluator.YAW_ATLAS_BUNDLE,
                DAgger_V_BUNDLE, YAW_CORRECTOR_PATH,
                evaluator.HGB_TRAIN_DATASET, evaluator.HIGHSTEER_DATASET,
                evaluator.PRACTICE_DATASET)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    parent_hgb = joblib.load(evaluator.HGB_BUNDLE)
    candidate_hgb = dict(parent_hgb)
    dagger_v = joblib.load(DAgger_V_BUNDLE)
    if dagger_v.get("training_dataset_sha256") != parent_hgb.get(
            "training_dataset_sha256"):
        raise ValueError("DAgger v head and parent use different training data")
    if dagger_v.get("selected_parent_feature_variant") != parent_hgb.get(
            "selected_parent_feature_variant"):
        raise ValueError("DAgger v head feature contract differs from parent")
    candidate_hgb["recursive_aux_models"] = list(
        parent_hgb["recursive_aux_models"])
    candidate_hgb["recursive_aux_models"][0] = dagger_v[
        "recursive_aux_models"][0]
    atlas = joblib.load(evaluator.YAW_ATLAS_BUNDLE)
    for model in (list(atlas["global_event_models"].values())
                  + list(atlas["local_expert_models"].values())):
        if hasattr(model, "n_jobs"):
            model.n_jobs = 1
    yaw_bundle = joblib.load(YAW_CORRECTOR_PATH)
    yaw_model = HighSteerYawResidual(
        yaw_bundle["estimator"],
        tuple(yaw_bundle["gate"]["rear_wheel_mean_mps"]))
    _, edssm, metadata = _load_model(evaluator.EDSSM_CHECKPOINT, "cpu")

    cohorts = (
        ("broad_openplane_validation", evaluator.HGB_TRAIN_DATASET,
         "validation", False),
        ("highsteer_validation", evaluator.HIGHSTEER_DATASET,
         "validation", True),
        ("unseen_practice_diagnostic", evaluator.PRACTICE_DATASET,
         "unseen_practice", False),
    )
    results: dict[str, Any] = {}
    for label, path, split, highsteer in cohorts:
        data = _load_dataset(path)
        horizon = HORIZON_BY_COHORT[label]
        if highsteer:
            windows = evaluator._windows(
                data, path, horizon, evaluator._bundle_lags(parent_hgb))
        else:
            windows = _run_windows(
                data, split, evaluator._bundle_lags(parent_hgb),
                MAX_WINDOWS_PER_RUN, horizon_steps=horizon)
        bounds = np.asarray(data["bounds"], dtype=np.int64)
        windows = [row for row in windows
                   if row["start"] + horizon < bounds[row["sequence"], 1]]
        prior = reference["results"][label]
        selected_runs = sorted({
            str(data["run_ids"][row["run"]]) for row in windows})
        if (len(windows) != prior["windows"]
                or selected_runs != prior["run_ids"]):
            raise ValueError(f"held-out windows differ from reference for {label}")
        print(f"{label}: matched EDSSM / parent-v / DAgger-v, "
              f"{len(windows)} windows", flush=True)
        baseline = _score(
            data, parent_hgb, edssm, metadata, atlas, yaw_model,
            windows, horizon, lateral_v_update="edssm")
        direct_parent_v = _score(
            data, parent_hgb, edssm, metadata, atlas, yaw_model,
            windows, horizon, lateral_v_update="hgb_v_increment")
        dagger_global = _score(
            data, candidate_hgb, edssm, metadata, atlas, yaw_model,
            windows, horizon, lateral_v_update="hgb_v_increment")
        dagger_gated = _score(
            data, candidate_hgb, edssm, metadata, atlas, yaw_model,
            windows, horizon, gate=HIGHSTEER_GATE,
            lateral_v_update="gated_hgb_v_increment")
        results[label] = {
            "split": split,
            "run_ids": selected_runs,
            "windows": len(windows),
            "horizon_s": horizon * evaluator.DT_S,
            "same_parent_edssm_v": baseline,
            "same_parent_hgb_v": direct_parent_v,
            "dagger_v_global": dagger_global,
            "dagger_v_highsteer_gated": dagger_gated,
        }

    report = {
        "schema_version": 1,
        "purpose": "matched causal composition test of the isolated DAgger v head with the frozen yaw atlas",
        "comparison_design": "all variants use identical initial windows, parent u/wheel/actuator heads, EDSSM latent, frozen yaw atlas and corrector; only v increment is varied",
        "test_or_final_test_used": False,
        "simulator_was_run": False,
        "runtime_or_physics_changed": False,
        "yaw_atlas_frozen": True,
        "yaw_corrector_frozen": True,
        "future_sensor_or_truth_input": False,
        "gate": HIGHSTEER_GATE,
        "invalidated_comparator_note": "The earlier evaluation.json and gated_evaluation.json compared candidates using HGB_BUNDLE against a baseline copied from the extended report whose parent used the separate highsteer-augmented HGB main heads and residual correction. Those records are preserved for audit but are not used here.",
        "comparison_reference_sha256": _sha256(reference_path),
        "input_sha256": {
            str(path.relative_to(ROOT)): _sha256(path) for path in required},
        "results": results,
        "limitations": [
            "initial measured histories are simulator truth; subsequent rollouts are causal",
            "practice is only two five-second windows and is a transfer diagnostic",
            "this is offline plant-component evaluation, not production odometry/MPC integration or full-lap validation",
        ],
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return report


def run_gated() -> dict[str, Any]:
    output = OUTPUT_DIR / "gated_evaluation.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    ungated_path = OUTPUT_DIR / "evaluation.json"
    reference_path = COMPOSITION_DIR / "evaluation.json"
    if not ungated_path.is_file() or not reference_path.is_file():
        raise FileNotFoundError("complete ungated and baseline reports first")
    ungated = json.loads(ungated_path.read_text(encoding="utf-8"))
    reference = json.loads(reference_path.read_text(encoding="utf-8"))

    hgb = joblib.load(evaluator.HGB_BUNDLE)
    dagger_v = joblib.load(DAgger_V_BUNDLE)
    if (dagger_v.get("training_dataset_sha256")
            != hgb.get("training_dataset_sha256")
            or dagger_v.get("selected_parent_feature_variant")
            != hgb.get("selected_parent_feature_variant")):
        raise ValueError("DAgger v head does not match plant feature contract")
    hgb["recursive_aux_models"] = list(hgb["recursive_aux_models"])
    hgb["recursive_aux_models"][0] = dagger_v["recursive_aux_models"][0]
    atlas = joblib.load(evaluator.YAW_ATLAS_BUNDLE)
    for model in (list(atlas["global_event_models"].values())
                  + list(atlas["local_expert_models"].values())):
        if hasattr(model, "n_jobs"):
            model.n_jobs = 1
    yaw_bundle = joblib.load(YAW_CORRECTOR_PATH)
    yaw_model = HighSteerYawResidual(
        yaw_bundle["estimator"],
        tuple(yaw_bundle["gate"]["rear_wheel_mean_mps"]))
    _, edssm, metadata = _load_model(evaluator.EDSSM_CHECKPOINT, "cpu")

    cohorts = (
        ("broad_openplane_validation", evaluator.HGB_TRAIN_DATASET,
         "validation", False),
        ("highsteer_validation", evaluator.HIGHSTEER_DATASET,
         "validation", True),
        ("unseen_practice_diagnostic", evaluator.PRACTICE_DATASET,
         "unseen_practice", False),
    )
    results: dict[str, Any] = {}
    for label, path, split, highsteer in cohorts:
        data = _load_dataset(path)
        horizon = HORIZON_BY_COHORT[label]
        if highsteer:
            windows = evaluator._windows(
                data, path, horizon, evaluator._bundle_lags(hgb))
        else:
            windows = _run_windows(
                data, split, evaluator._bundle_lags(hgb),
                MAX_WINDOWS_PER_RUN, horizon_steps=horizon)
        bounds = np.asarray(data["bounds"], dtype=np.int64)
        windows = [row for row in windows
                   if row["start"] + horizon < bounds[row["sequence"], 1]]
        prior = reference["results"][label]
        if (len(windows) != prior["windows"]
                or sorted({str(data["run_ids"][row["run"]]) for row in windows})
                != prior["run_ids"]):
            raise ValueError(f"held-out windows differ from reference for {label}")
        candidate = _score(data, hgb, edssm, metadata, atlas, yaw_model,
                           windows, horizon, gate=HIGHSTEER_GATE)
        results[label] = {
            "split": split,
            "run_ids": prior["run_ids"],
            "windows": len(windows),
            "gate": HIGHSTEER_GATE,
            "baseline": prior["hgb_edssm_yaw_atlas_reference"],
            "gated_dagger_v": candidate,
            "ungated_dagger_v": ungated["results"][label][
                "dagger_v_hgb_edssm_frozen_yaw_atlas"],
        }

    report = {
        "schema_version": 1,
        "purpose": "test whether the measured 7-9 m/s, 0.30-0.525 rad local support gate preserves the DAgger v-head gain without practice-transfer regression",
        "changed_component_only": "recursive next-v head, gated to the existing measured high-steer support",
        "yaw_atlas_frozen": True,
        "yaw_corrector_frozen": True,
        "test_or_final_test_used": False,
        "simulator_was_run": False,
        "runtime_or_physics_changed": False,
        "gate": HIGHSTEER_GATE,
        "reference_sha256": _sha256(reference_path),
        "ungated_candidate_sha256": _sha256(ungated_path),
        "results": results,
    }
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-fallback-gate", action="store_true",
                        help="evaluate the local DAgger-v head with parent HGB-v fallback")
    parser.add_argument("--dagger-uv", action="store_true",
                        help="evaluate the isolated DAgger u/v heads with frozen yaw-atlas yaw")
    args = parser.parse_args()
    if args.parent_fallback_gate and args.dagger_uv:
        parser.error("select only one focused evaluation")
    if args.parent_fallback_gate:
        output = OUTPUT_DIR / "matched_parent_fallback_gate.json"
        report = (json.loads(output.read_text(encoding="utf-8"))
                  if output.is_file() else run_parent_fallback_gate())
        summary = {}
        for cohort, values in report["results"].items():
            summary[cohort] = {}
            for name in ("parent_hgb_v", "dagger_v_global_reference",
                         "dagger_v_with_parent_hgb_fallback"):
                curve = values[name]["state_error_curve_macro_run_rmse"]
                summary[cohort][name] = {
                    time: curve[time] for time in
                    ("0.750s", "2.000s", "5.000s", "12.000s")
                    if time in curve}
        print(json.dumps({"output": str(output), "summary": summary},
                         indent=2, sort_keys=True))
        return
    if args.dagger_uv:
        output = OUTPUT_DIR / "matched_uv_evaluation.json"
        report = (json.loads(output.read_text(encoding="utf-8"))
                  if output.is_file() else run_dagger_uv_composition())
        summary = {}
        for cohort, values in report["results"].items():
            summary[cohort] = {}
            for name in ("same_parent_hgb_uv", "dagger_v_only_reference",
                         "dagger_uv_candidate"):
                curve = values[name]["state_error_curve_macro_run_rmse"]
                summary[cohort][name] = {
                    time: curve[time] for time in
                    ("0.750s", "2.000s", "5.000s", "12.000s")
                    if time in curve}
        print(json.dumps({"output": str(output), "summary": summary},
                         indent=2, sort_keys=True))
        return
    output = OUTPUT_DIR / "matched_evaluation.json"
    report = (json.loads(output.read_text(encoding="utf-8"))
              if output.is_file() else run_matched_comparison())
    summary = {}
    for cohort, values in report["results"].items():
        summary[cohort] = {}
        for name in ("same_parent_edssm_v", "same_parent_hgb_v",
                     "dagger_v_global", "dagger_v_highsteer_gated"):
            row = values[name]
            curve = row["state_error_curve_macro_run_rmse"]
            summary[cohort][name] = {
                horizon: curve[horizon]
                for horizon in ("0.250s", "0.750s", "2.000s", "5.000s",
                                "12.000s") if horizon in curve}
    print(json.dumps({"output": str(output),
                      "summary": summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
