#!/usr/bin/env python3
"""Isolate u versus rear-wheel DAgger residual effects on practice transfer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from tools.vehicle_dynamics_learning import evaluate_hybrid_longitudinal_edssm_candidate as ev
from tools.vehicle_dynamics_learning.fit_full_regime_yaw_atlas_dagger_corrector import (
    _training_source, _validation_sources, OUTPUT_DIR as YAW_DIR)
from tools.vehicle_dynamics_learning.fit_recursive_yaw_atlas_dagger_corrector import (
    HIGHSTEER_TRAIN_DATASET, CORRECTION_START_STEP)
from tools.vehicle_dynamics_learning.fit_yaw_conditioned_u_wheel_dagger import (
    OUT as UV_DIR, gate_from_training)
from tools.vehicle_dynamics_learning.evaluate_yaw_atlas_recursive_gain import (
    _run_windows)
from tools.vehicle_dynamics_learning.run_truncated_history_transition import ROOT
from tools.vehicle_dynamics_learning.score_effective_race_teacher import _load_model
from tools.vehicle_dynamics_learning.train_nssm import _load_dataset

OUT = ROOT / "live_runs/derived_dynamics_learning_20260928/yaw_conditioned_u_wheel_head_ablation_20261011"
REPORT = OUT / "evaluation.json"
INCREMENT_OUT = ROOT / "live_runs/derived_dynamics_learning_20260928/yaw_conditioned_u_wheel_increment_head_ablation_20261011"
STEPS = (30, 80, 200)
TRANSIENT_VALIDATION = tuple(
    (f"subnet_transient_validation_r{index:02d}",
     ROOT / "live_runs" /
     f"openplane_subnet_highsteer_transients_validation_r{index:02d}_20261004" /
     "source_continuous_v1/openplane_dynamics.npz")
    for index in (1, 2, 3))


class ZeroHead:
    def predict(self, x):
        return np.zeros(len(x), dtype=np.float32)


def score(trace, run_ids, steps=STEPS):
    p = np.asarray(trace["predicted_state"], dtype=float)
    t = np.asarray(trace["truth_state"], dtype=float)
    pp = np.asarray(trace["predicted_pose"], dtype=float)
    tp = np.asarray(trace["truth_pose"], dtype=float)
    ri = np.asarray(trace["run_indices"], dtype=int)
    pe = pp - tp
    pe[..., 2] = np.arctan2(np.sin(pe[..., 2]), np.cos(pe[..., 2]))
    output = {}
    for endpoint in steps:
        per_run = {}
        k = endpoint - 1
        for run in sorted(np.unique(ri)):
            sel = ri == run
            e = p[sel, k] - t[sel, k]
            raw_valid = np.asarray(
                trace["raw_encoder_target_valid"], dtype=bool)[sel, k]
            raw_error = (
                np.asarray(trace["predicted_raw_encoder_wheels"], dtype=float)[
                    sel, k]
                - np.asarray(trace["truth_raw_encoder_wheels"], dtype=float)[
                    sel, k])
            per_run[str(run_ids[run])] = {
                "u_rmse_mps": float(np.sqrt(np.mean(e[:, 0] ** 2))),
                "v_rmse_mps": float(np.sqrt(np.mean(e[:, 1] ** 2))),
                "yaw_rate_rmse_rps": float(np.sqrt(np.mean(e[:, 2] ** 2))),
                "rear_wheel_pair_rmse_mps": float(np.sqrt(np.mean(e[:, 5:7] ** 2))),
                "raw_encoder_pair_rmse_mps": (
                    float(np.sqrt(np.mean(raw_error[raw_valid] ** 2)))
                    if np.any(raw_valid) else None),
                "steering_feedback_rmse_rad": float(
                    np.sqrt(np.mean(e[:, 3] ** 2))),
                "throttle_feedback_rmse": float(
                    np.sqrt(np.mean(e[:, 4] ** 2))),
                "position_rmse_m": float(np.sqrt(np.mean(pe[sel, k, :2] ** 2))),
                "pose_yaw_rmse_rad": float(np.sqrt(np.mean(pe[sel, k, 2] ** 2))),
            }
        metrics = tuple(next(iter(per_run.values())))
        output[f"{endpoint * ev.DT_S:.3f}s"] = {
            "per_run": per_run,
            "macro_run_rmse": {
                name: (float(np.mean([
                    row[name] for row in per_run.values()
                    if row[name] is not None]))
                       if any(row[name] is not None
                              for row in per_run.values()) else None)
                for name in metrics},
        }
    return output


def run(candidate_model: Path, out: Path, all_heldouts: bool = False):
    candidate_model = (candidate_model if candidate_model.is_absolute()
                       else ROOT / candidate_model)
    out = out if out.is_absolute() else ROOT / out
    if not candidate_model.is_file():
        raise FileNotFoundError(candidate_model)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite {out}")
    hgb = joblib.load(ev.HGB_BUNDLE)
    atlas = joblib.load(ev.YAW_ATLAS_BUNDLE)
    for model in list(atlas["global_event_models"].values()) + list(
            atlas["local_expert_models"].values()):
        if hasattr(model, "n_jobs"):
            model.n_jobs = 1
    _, edssm, metadata = _load_model(ev.EDSSM_CHECKPOINT, "cpu")
    yaw = joblib.load(YAW_DIR / "full_regime_yaw_residual.joblib")["estimator"]
    bundle = joblib.load(candidate_model)
    models = bundle["models"]
    gate = bundle.get("training_gate")
    if gate is None:
        train_sources = [
            _training_source(ev.HGB_TRAIN_DATASET, "broad", hgb),
            _training_source(HIGHSTEER_TRAIN_DATASET, "highsteer", hgb)]
        gate = gate_from_training(train_sources)
    gate = {key: tuple(value) for key, value in gate.items()}
    variants = {
        "yaw_residual_only": None,
        "u_head_only": [models[0], ZeroHead(), ZeroHead()],
        "rear_wheel_heads_only": [ZeroHead(), models[1], models[2]],
        "u_and_rear_wheel_heads": models,
    }
    if all_heldouts:
        variants = {name: variants[name] for name in (
            "yaw_residual_only", "rear_wheel_heads_only",
            "u_and_rear_wheel_heads")}
    sources = _validation_sources(hgb)
    if not all_heldouts:
        sources = [source for source in sources
                   if source["name"] == "unseen_practice_diagnostic"]
    else:
        for name, path in TRANSIENT_VALIDATION:
            data = _load_dataset(path)
            windows = _run_windows(
                data, "validation", ev._bundle_lags(hgb), 100_000,
                horizon_steps=80)
            run_ids = sorted({
                str(data["run_ids"][row["run"]]) for row in windows})
            if not windows or any(
                    np.asarray(data["splits"]).astype(str)[row["run"]]
                    != "validation" for row in windows):
                raise ValueError(f"invalid held-out transient source: {name}")
            sources.append({
                "name": name, "data": data, "windows": windows,
                "horizon": 80, "split": "validation", "run_ids": run_ids})
    cohorts = {}
    for source in sources:
        horizon = min(200, int(source["horizon"]))
        ev.HORIZONS = tuple(step for step in STEPS if step <= horizon)
        split_values = np.asarray(source["data"]["splits"]).astype(str)
        if any(split_values[row["run"]] in ("test", "final_test")
               for row in source["windows"]):
            raise ValueError(f"{source['name']} selected a sealed partition")
        results = {}
        steps = tuple(step for step in STEPS if step <= horizon)
        for name, residual in variants.items():
            r = ev._rollout(
                source["data"], hgb, edssm, metadata, atlas,
                source["windows"], horizon,
                yaw_delta_gain=1.0, include_oracle_diagnostics=False,
                reconcile_composed_latent=True,
                atlas_acceleration_source="composed_transition",
                yaw_correction_model=yaw,
                yaw_correction_start_step=CORRECTION_START_STEP,
                hgb_main_gate=gate,
                hgb_main_residual_models=residual,
                hgb_main_residual_start_step=1,
                hgb_main_residual_gain=1.0,
                capture_trajectory_trace=True,
                lateral_v_update="edssm")
            trace = r.pop("_trajectory_trace")
            results[name] = score(
                trace, np.asarray(source["data"]["run_ids"]).astype(str),
                steps)
        cohorts[source["name"]] = {
            "split": source["split"], "run_ids": source["run_ids"],
            "windows": len(source["windows"]), "horizon_s": horizon * ev.DT_S,
            "variants": results}
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "purpose": "causal head ablation to distinguish u-head and rear-wheel-head effects after full-regime yaw correction",
        "simulator_was_run": False,
        "test_or_final_test_used": False,
        "all_heldouts": all_heldouts,
        "same_initial_states_commands_and_frozen_yaw_models": True,
        "candidate_model": str(candidate_model.relative_to(ROOT)),
        "candidate_target_mode": bundle.get("target_mode", "state_correction"),
        "training_gate": {k: list(v) for k, v in gate.items()},
        "cohorts": cohorts,
        "limitations": [
            "two practice captures only; diagnostic attribution, not a generalization estimate",
            "ablation changes head outputs but does not re-fit them",
            "candidate remains research-only",
        ],
    }
    (out / "evaluation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-model", type=Path,
        default=UV_DIR / "yaw_conditioned_u_wheel_residual.joblib")
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--all-heldouts", action="store_true",
                        help="also replay broad and high-steering validation; each source uses the same selected windows for all four ablation variants")
    args = parser.parse_args()
    report = run(args.candidate_model, args.output_dir, args.all_heldouts)
    print(json.dumps({"report": str(args.output_dir / "evaluation.json"), "cohorts": {
        cohort: {name: {h: row["macro_run_rmse"] for h, row in data.items()}
                 for name, data in values["variants"].items()}
        for cohort, values in report["cohorts"].items()}},
        indent=2, sort_keys=True))

