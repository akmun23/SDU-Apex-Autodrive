#!/usr/bin/env python3
"""Train u/rear-wheel DAgger residuals for the frozen corrected-yaw rollout.

Training uses training captures only. The yaw atlas and yaw residual remain
frozen. The candidate is compared on the same existing validation and practice
windows as the yaw-residual evaluation; it is not a runtime model.
"""
from __future__ import annotations

import hashlib
import argparse
import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from tools.vehicle_dynamics_learning import evaluate_hybrid_longitudinal_edssm_candidate as ev
from tools.vehicle_dynamics_learning.fit_full_regime_yaw_atlas_dagger_corrector import (
    HIGHSTEER_TRAIN_DATASET, OUTPUT_DIR as YAW_DIR,
    _training_source, _validation_sources)
from tools.vehicle_dynamics_learning.fit_recursive_yaw_atlas_dagger_corrector import (
    CORRECTION_START_STEP)
from tools.vehicle_dynamics_learning.evaluate_yaw_atlas_recursive_gain import (
    _run_windows)
from tools.vehicle_dynamics_learning.run_truncated_history_transition import ROOT
from tools.vehicle_dynamics_learning.score_effective_race_teacher import _load_model
from tools.vehicle_dynamics_learning.train_nssm import _load_dataset

OUT = ROOT / "live_runs/derived_dynamics_learning_20260928/yaw_conditioned_u_wheel_dagger_20261011"
MODEL_PATH = OUT / "yaw_conditioned_u_wheel_residual.joblib"
REPORT_PATH = OUT / "evaluation.json"
INCREMENT_OUT = ROOT / "live_runs/derived_dynamics_learning_20260928/yaw_conditioned_u_wheel_increment_dagger_20261011"
HORIZON = 200
GENERATIONS = 3
ADDITIONAL_TRAINING_WINDOWS_PER_RUN = 64
REPORT_STEPS = (30, 80, 200, 400, 480)
METRICS = ("u_rmse_mps", "v_rmse_mps", "yaw_rate_rmse_rps",
           "steering_feedback_rmse_rad", "throttle_feedback_rmse",
           "rear_wheel_pair_rmse_mps", "raw_encoder_pair_rmse_mps",
           "position_rmse_m",
           "pose_yaw_rmse_rad")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fit(x: np.ndarray, y: np.ndarray, runs: np.ndarray,
        run_ids: np.ndarray):
    if (x.ndim != 2 or y.shape != (len(x), 3) or runs.shape != (len(x),)
            or not len(x) or not np.isfinite(x).all() or not np.isfinite(y).all()):
        raise ValueError("invalid yaw-conditioned u/wheel training rows")
    counts = np.bincount(runs, minlength=len(run_ids)).astype(float)
    weights = 1.0 / counts[runs]
    weights *= len(weights) / weights.sum()
    models, report = [], {"rows": int(len(x)), "run_count": int(len(np.unique(runs))),
                          "per_head_training": {}}
    for i, name in enumerate(("u", "rear_left_wheel", "rear_right_wheel")):
        model = HistGradientBoostingRegressor(
            loss="squared_error", learning_rate=0.05, max_iter=180,
            max_leaf_nodes=15, min_samples_leaf=60, l2_regularization=2.0,
            early_stopping=False, random_state=20261011 + i)
        model.fit(x, y[:, i], sample_weight=weights)
        err = y[:, i] - model.predict(x)
        report["per_head_training"][name] = {
            "target_rmse": float(np.sqrt(np.mean(y[:, i] ** 2))),
            "residual_rmse": float(np.sqrt(np.mean(err ** 2)))}
        models.append(model)
    return models, report


def capture(source: dict[str, Any], hgb: dict[str, Any], edssm: Any,
            metadata: dict[str, Any], atlas: dict[str, Any], yaw_model: Any,
            gate: dict[str, tuple[float, float]], correction: Any,
            run_to_index: dict[str, int], label_mode: str,
            horizon_steps: int = HORIZON):
    r = ev._rollout(
        source["data"], hgb, edssm, metadata, atlas, source["windows"],
        horizon_steps,
        yaw_delta_gain=1.0, include_oracle_diagnostics=False,
        reconcile_composed_latent=True, atlas_acceleration_source="composed_transition",
        yaw_correction_model=yaw_model,
        yaw_correction_start_step=CORRECTION_START_STEP,
        hgb_main_gate=gate, hgb_main_residual_models=correction,
        hgb_main_residual_start_step=1, hgb_main_residual_gain=1.0,
        capture_hgb_main_residual_rows=True,
        hgb_main_residual_target_mode=label_mode, lateral_v_update="edssm")
    x = r.pop("_hgb_main_residual_training_features")
    y = r.pop("_hgb_main_residual_training_targets")
    local = r.pop("_hgb_main_residual_training_run_indices")
    ids = np.asarray(source["data"]["run_ids"]).astype(str)
    run_idx = np.asarray([run_to_index[str(ids[j])] for j in local], dtype=np.int32)
    return x, y, run_idx, {
        "source": source["name"], "windows": len(source["windows"]),
        "run_ids": sorted(source["run_ids"]), "rows": len(x),
        "gate_active_fraction": r["hgb_main_gate_active_fraction"]}


def gate_from_training(sources):
    speed, steer = [], []
    for source in sources:
        frames = np.asarray(source["data"]["frames"], dtype=float)
        horizon = int(source.get("training_horizon_steps", HORIZON))
        idx = np.concatenate([np.arange(w["start"], w["start"] + horizon)
                              for w in source["windows"]])
        speed.append(np.hypot(frames[idx, 0], frames[idx, 1]))
        steer.append(np.abs(frames[idx, 3]))
    return {"speed_mps": (0.0, float(np.max(np.concatenate(speed)))),
            "absolute_steering_rad": (
                0.0, float(np.max(np.concatenate(steer))))}


def additional_training_source(path: Path, name: str,
                               hgb: dict[str, Any],
                               horizon_steps: int) -> dict[str, Any]:
    if horizon_steps <= 0:
        raise ValueError("additional training horizon must be positive")
    path = path.resolve()
    data = _load_dataset(path)
    splits = np.asarray(data["splits"]).astype(str)
    windows = _run_windows(
        data, "train", ev._bundle_lags(hgb),
        ADDITIONAL_TRAINING_WINDOWS_PER_RUN,
        horizon_steps=horizon_steps)
    run_ids = {str(data["run_ids"][row["run"]]) for row in windows}
    if not windows or any(splits[row["run"]] != "train" for row in windows):
        raise ValueError(f"additional source {name} includes non-training data")
    return {"name": name, "path": path, "data": data,
            "windows": windows, "run_ids": run_ids,
            "training_horizon_steps": horizon_steps}


def metrics_from_trace(trace, ids):
    p = np.asarray(trace["predicted_state"], dtype=float)
    t = np.asarray(trace["truth_state"], dtype=float)
    pp = np.asarray(trace["predicted_pose"], dtype=float)
    tp = np.asarray(trace["truth_pose"], dtype=float)
    ri = np.asarray(trace["run_indices"], dtype=int)
    pe = pp - tp
    pe[..., 2] = np.arctan2(np.sin(pe[..., 2]), np.cos(pe[..., 2]))
    output = {}
    for endpoint in REPORT_STEPS:
        if endpoint > p.shape[1]:
            continue
        per_run = {}
        k = endpoint - 1
        for run in sorted(np.unique(ri)):
            sel = ri == run
            e = p[sel, k] - t[sel, k]
            raw_valid = np.asarray(
                trace["raw_encoder_target_valid"], dtype=bool)[sel, k]
            raw_error = (np.asarray(
                trace["predicted_raw_encoder_wheels"], dtype=float)[sel, k]
                - np.asarray(
                    trace["truth_raw_encoder_wheels"], dtype=float)[sel, k])
            raw_rmse = (float(np.sqrt(np.mean(raw_error[raw_valid] ** 2)))
                        if np.any(raw_valid) else None)
            per_run[str(ids[run])] = {
                "u_rmse_mps": float(np.sqrt(np.mean(e[:, 0] ** 2))),
                "v_rmse_mps": float(np.sqrt(np.mean(e[:, 1] ** 2))),
                "yaw_rate_rmse_rps": float(np.sqrt(np.mean(e[:, 2] ** 2))),
                "steering_feedback_rmse_rad": float(np.sqrt(np.mean(e[:, 3] ** 2))),
                "throttle_feedback_rmse": float(np.sqrt(np.mean(e[:, 4] ** 2))),
                "rear_wheel_pair_rmse_mps": float(np.sqrt(np.mean(e[:, 5:7] ** 2))),
                "raw_encoder_pair_rmse_mps": raw_rmse,
                "raw_encoder_pair_valid_fraction": float(np.mean(raw_valid)),
                "position_rmse_m": float(np.sqrt(np.mean(pe[sel, k, :2] ** 2))),
                "pose_yaw_rmse_rad": float(np.sqrt(np.mean(pe[sel, k, 2] ** 2)))}
        output[f"{endpoint * ev.DT_S:.3f}s"] = {
            "per_run": per_run,
            "macro_run_rmse": {
                name: (None if any(row[name] is None
                                   for row in per_run.values()) else
                       float(np.mean([row[name]
                                      for row in per_run.values()])))
                for name in METRICS}}
    return output


def score(source, hgb, edssm, metadata, atlas, yaw_model, gate, correction,
          raw_wheel_correction=None):
    horizon = source["horizon"]
    ev.HORIZONS = tuple(s for s in REPORT_STEPS if s <= horizon)
    r = ev._rollout(
        source["data"], hgb, edssm, metadata, atlas, source["windows"], horizon,
        yaw_delta_gain=1.0, include_oracle_diagnostics=False,
        reconcile_composed_latent=True, atlas_acceleration_source="composed_transition",
        yaw_correction_model=yaw_model,
        yaw_correction_start_step=CORRECTION_START_STEP,
        hgb_main_gate=gate, hgb_main_residual_models=correction,
        hgb_main_residual_start_step=1, hgb_main_residual_gain=1.0,
        raw_wheel_residual_models=raw_wheel_correction,
        raw_wheel_residual_start_step=0, raw_wheel_residual_gain=1.0,
        capture_trajectory_trace=True, lateral_v_update="edssm")
    trace = r.pop("_trajectory_trace")
    return metrics_from_trace(trace, np.asarray(source["data"]["run_ids"]).astype(str)), {
        "windows": r["windows"], "windows_by_run": r["windows_by_run"],
        "gate_active_fraction": r["hgb_main_gate_active_fraction"]}


def run(label_mode: str = "state_correction",
        additional_training_datasets: tuple[Path, ...] = (),
        additional_horizon_steps: int = 80,
        output_dir: Path | None = None):
    if label_mode not in ("state_correction", "truth_transition_increment"):
        raise ValueError(f"unsupported target mode: {label_mode}")
    out = (Path(output_dir).resolve() if output_dir is not None else
           OUT if label_mode == "state_correction" else INCREMENT_OUT)
    model_path = out / MODEL_PATH.name
    report_path = out / "evaluation.json"
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite {report_path}")
    resume = model_path.exists()
    if out.exists() and not resume:
        raise FileExistsError(f"refusing to reuse incomplete output {out}")
    yaw_report_path = YAW_DIR / "evaluation.json"
    yaw_model_path = YAW_DIR / "full_regime_yaw_residual.joblib"
    required = (yaw_report_path, yaw_model_path, ev.HGB_BUNDLE,
                ev.EDSSM_CHECKPOINT, ev.YAW_ATLAS_BUNDLE,
                ev.HGB_TRAIN_DATASET, HIGHSTEER_TRAIN_DATASET,
                ev.HIGHSTEER_DATASET, ev.PRACTICE_DATASET)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    yaw_report = json.loads(yaw_report_path.read_text())
    if yaw_report.get("test_or_final_test_used") is not False:
        raise ValueError("yaw candidate provenance check failed")
    hgb = joblib.load(ev.HGB_BUNDLE)
    atlas = joblib.load(ev.YAW_ATLAS_BUNDLE)
    for model in list(atlas["global_event_models"].values()) + list(
            atlas["local_expert_models"].values()):
        if hasattr(model, "n_jobs"):
            model.n_jobs = 1
    _, edssm, metadata = _load_model(ev.EDSSM_CHECKPOINT, "cpu")
    yaw_model = joblib.load(yaw_model_path)["estimator"]
    sources = [
        _training_source(ev.HGB_TRAIN_DATASET, "broad_train", hgb),
        _training_source(HIGHSTEER_TRAIN_DATASET, "highsteer_train", hgb)]
    for index, path in enumerate(additional_training_datasets, start=1):
        sources.append(additional_training_source(
            path, f"additional_train_{index:02d}", hgb,
            additional_horizon_steps))
    owners = {}
    for s in sources:
        for run_id in s["run_ids"]:
            if run_id in owners:
                raise ValueError(f"duplicate training run {run_id}")
            owners[run_id] = s["name"]
    run_ids = np.asarray(sorted(owners), dtype=str)
    run_to_index = {name: i for i, name in enumerate(run_ids)}
    gate = gate_from_training(sources)
    xs, ys, rs, generations, fits = [], [], [], [], []
    started = time.time()
    if resume:
        bundle = joblib.load(model_path)
        if (bundle.get("hgb_bundle_sha256") != sha256(ev.HGB_BUNDLE)
                or bundle.get("yaw_residual_sha256") != sha256(yaw_model_path)
                or bundle.get("target_mode", "state_correction") != label_mode
                or bundle.get("training_run_ids") != run_ids.tolist()
                or bundle.get("training_gate")
                != {k: list(v) for k, v in gate.items()}
                or bundle.get("additional_training_dataset_sha256", [])
                != [sha256(path) for path in additional_training_datasets]
                or bundle.get("additional_horizon_steps")
                != additional_horizon_steps
                or bundle.get("additional_training_windows_per_run")
                != ADDITIONAL_TRAINING_WINDOWS_PER_RUN):
            raise ValueError("saved residual does not match current training inputs")
        correction = bundle["models"]
        generations = bundle.get("dagger_generations", [])
        fits = bundle.get("fit_reports", [])
        print("resuming held-out replay from the saved fitted candidate",
              flush=True)
    else:
        correction = None
        with threadpool_limits(limits=2):
            for generation in range(GENERATIONS):
                source_reports = []
                for s in sources:
                    x, y, r, meta = capture(
                        s, hgb, edssm, metadata, atlas, yaw_model, gate,
                        correction, run_to_index, label_mode,
                        int(s.get("training_horizon_steps", HORIZON)))
                    xs.append(x)
                    ys.append(y)
                    rs.append(r)
                    source_reports.append(meta)
                correction, fit_report = fit(
                    np.concatenate(xs), np.concatenate(ys),
                    np.concatenate(rs), run_ids)
                fit_report["generation"] = generation
                fits.append(fit_report)
                generations.append({
                    "generation": generation, "sources": source_reports})
                print(f"fitted generation {generation + 1}/{GENERATIONS}, "
                      f"{fit_report['rows']} accumulated samples", flush=True)
        if correction is None:
            raise AssertionError("no residual model fitted")
        out.mkdir(parents=True, exist_ok=False)
        joblib.dump({
            "schema_version": 1, "models": correction, "research_only": True,
            "target_mode": label_mode,
            "production_or_runtime_integration": False,
            "training_run_ids": run_ids.tolist(),
            "training_gate": {k: list(v) for k, v in gate.items()},
            "additional_training_dataset_sha256": [
                sha256(path) for path in additional_training_datasets],
            "additional_horizon_steps": additional_horizon_steps,
            "additional_training_windows_per_run": (
                ADDITIONAL_TRAINING_WINDOWS_PER_RUN),
            "training_sources": [{
                "name": source["name"],
                "path": str(source["path"].resolve().relative_to(ROOT)),
                "sha256": sha256(source["path"]),
                "run_ids": sorted(source["run_ids"]),
                "windows": len(source["windows"]),
                "capture_horizon_steps": int(source.get(
                    "training_horizon_steps", HORIZON)),
            } for source in sources],
            "target": ("simulator-truth next u/wheel state correction" if
                       label_mode == "state_correction" else
                       "simulator-truth one-step u/rear-wheel transition increment minus frozen HGB predicted increment"),
            "yaw_residual_sha256": sha256(yaw_model_path),
            "hgb_bundle_sha256": sha256(ev.HGB_BUNDLE),
            "dagger_generations": generations,
            "fit_reports": fits}, model_path)
    heldout = {}
    for s in _validation_sources(hgb):
        if set(s["run_ids"]) & set(run_ids.tolist()):
            raise ValueError(f"training/held-out overlap for {s['name']}")
        print(f"replaying {s['name']} ({len(s['windows'])} paired windows)",
              flush=True)
        base, base_meta = score(s, hgb, edssm, metadata, atlas, yaw_model, gate, None)
        cand, cand_meta = score(s, hgb, edssm, metadata, atlas, yaw_model, gate,
                                correction)
        delta = {}
        for horizon, values in cand.items():
            b = base[horizon]
            delta[horizon] = {}
            for metric in METRICS:
                runs = sorted(set(values["per_run"]) & set(b["per_run"]))
                per_run_delta = {
                    run: (None if values["per_run"][run][metric] is None
                          or b["per_run"][run][metric] is None else
                          values["per_run"][run][metric]
                          - b["per_run"][run][metric])
                    for run in runs}
                numeric_delta = [value for value in per_run_delta.values()
                                 if value is not None]
                delta[horizon][metric] = {
                    "macro_run_candidate_minus_baseline": (
                        None if values["macro_run_rmse"][metric] is None
                        or b["macro_run_rmse"][metric] is None else
                        values["macro_run_rmse"][metric]
                        - b["macro_run_rmse"][metric]),
                    "per_run_candidate_minus_baseline": per_run_delta,
                    "runs_with_valid_target": len(numeric_delta)}
        heldout[s["name"]] = {
            "split": s["split"], "run_ids": s["run_ids"],
            "windows": len(s["windows"]), "baseline": base,
            "candidate": cand, "baseline_meta": base_meta,
            "candidate_meta": cand_meta, "paired_delta": delta}
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "purpose": "reduce recursive u/rear-wheel error under the frozen full-regime yaw residual",
        "simulator_was_run": False, "test_or_final_test_used": False,
        "yaw_atlas_frozen": True, "yaw_residual_frozen": True,
        "training_split_only": True, "training_run_ids": run_ids.tolist(),
        "target_mode": label_mode,
        "additional_training_windows_per_run": (
            ADDITIONAL_TRAINING_WINDOWS_PER_RUN),
        "training_windows": {s["name"]: len(s["windows"]) for s in sources},
        "training_sources": [{
            "name": source["name"],
            "path": str(source["path"].resolve().relative_to(ROOT)),
            "sha256": sha256(source["path"]),
            "run_ids": sorted(source["run_ids"]),
            "windows": len(source["windows"]),
            "capture_horizon_steps": int(source.get(
                "training_horizon_steps", HORIZON)),
        } for source in sources],
        "training_gate_from_measured_training_extent": {
            k: list(v) for k, v in gate.items()},
        "dagger_generations": generations, "fit_reports": fits,
        "heldout_results": heldout, "candidate_model": str(model_path.relative_to(ROOT)),
        "model_sha256": sha256(model_path), "runtime_seconds": time.time() - started,
        "limitations": [
            "DAgger truth labels are aligned by recorded capture time and are approximate if model state has diverged far from the recorded state",
            "measured marginal gate does not guarantee dense joint speed/steering support",
            "practice has only two diagnostic runs; candidate remains research-only",
            "no odometry/MPC integration or full-lap simulator validation"]}
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-mode", choices=(
        "state_correction", "truth_transition_increment"),
        default="state_correction")
    parser.add_argument("--additional-training-dataset", type=Path,
                        action="append", default=[])
    parser.add_argument("--additional-horizon-steps", type=int, default=80)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    report = run(args.label_mode, tuple(args.additional_training_dataset),
                 args.additional_horizon_steps, args.output_dir)
    summary = {cohort: {
        h: {m: v["macro_run_candidate_minus_baseline"]
            for m, v in row.items()}
        for h, row in values["paired_delta"].items()}
        for cohort, values in report["heldout_results"].items()}
    out = (Path(args.output_dir).resolve() if args.output_dir is not None else
           OUT if args.label_mode == "state_correction" else INCREMENT_OUT)
    report_path = out / "evaluation.json"
    print(json.dumps({"report": str(report_path), "paired_delta": summary},
                     indent=2, sort_keys=True))

