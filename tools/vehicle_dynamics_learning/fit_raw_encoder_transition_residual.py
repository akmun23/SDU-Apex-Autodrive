#!/usr/bin/env python3
"""Fit raw-wheel transition residuals from train-only fixed-40-Hz data.

The raw encoder signal is a measured wheel-speed target, not simulator wheel
rotation truth. It is kept distinct from the filtered odometry wheel state.
This research candidate corrects only predicted raw encoder increments; the
yaw atlas/residual, filtered-wheel model, simulator, and runtime are frozen.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from tools.vehicle_dynamics_learning import evaluate_hybrid_longitudinal_edssm_candidate as ev
from tools.vehicle_dynamics_learning.evaluate_yaw_atlas_recursive_gain import _run_windows
from tools.vehicle_dynamics_learning.fit_full_regime_yaw_atlas_dagger_corrector import (
    HIGHSTEER_TRAIN_DATASET, OUTPUT_DIR as YAW_DIR, _validation_sources)
from tools.vehicle_dynamics_learning.fit_yaw_conditioned_u_wheel_dagger import (
    score, sha256)
from tools.vehicle_dynamics_learning.run_truncated_history_transition import ROOT
from tools.vehicle_dynamics_learning.score_effective_race_teacher import _load_model
from tools.vehicle_dynamics_learning.train_nssm import _load_dataset


OUT = ROOT / "live_runs/derived_dynamics_learning_20260928/raw_encoder_transition_residual_20261011"
MODEL_PATH = OUT / "raw_encoder_transition_residual.joblib"
REPORT_PATH = OUT / "evaluation.json"
RECURSIVE_CHECKPOINT = OUT / "recursive_evaluation_checkpoint.json"
CHUNK_SIZE = 4096
STATE_METRICS = (
    "u_rmse_mps", "v_rmse_mps", "yaw_rate_rmse_rps",
    "steering_feedback_rmse_rad", "throttle_feedback_rmse",
    "rear_wheel_pair_rmse_mps", "raw_encoder_pair_rmse_mps",
    "position_rmse_m", "pose_yaw_rmse_rad")


def _training_gate(sources: list[dict[str, Any]]) -> dict[str, tuple[float, float]]:
    speeds, steering = [], []
    for source in sources:
        frames = np.asarray(source["data"]["frames"], dtype=np.float64)
        starts = np.asarray([row["start"] for row in source["windows"]],
                            dtype=np.int64)
        speeds.append(np.hypot(frames[starts, 0], frames[starts, 1]))
        steering.append(np.abs(frames[starts, 3]))
    speed = np.concatenate(speeds)
    steer = np.concatenate(steering)
    return {"speed_mps": (float(np.min(speed)), float(np.max(speed))),
            "absolute_steering_rad": (
                float(np.min(steer)), float(np.max(steer)))}


def _training_target_counts(source: dict[str, Any],
                            run_to_index: dict[str, int]
                            ) -> dict[str, int]:
    data = source["data"]
    starts = np.asarray([row["start"] for row in source["windows"]],
                        dtype=np.int64)
    window_runs = np.asarray([row["run"] for row in source["windows"]],
                             dtype=np.int32)
    raw = np.asarray(data["encoder_raw_surface_mps"], dtype=np.float64)
    valid = np.asarray(data["encoder_raw_valid"], dtype=bool)
    target_valid = (valid[starts] & valid[starts + 1]
                    & np.isfinite(raw[starts]).all(axis=1)
                    & np.isfinite(raw[starts + 1]).all(axis=1))
    run_ids = np.asarray(data["run_ids"]).astype(str)
    return {
        str(run_ids[run_id]): int(np.count_nonzero(
            target_valid & (window_runs == run_id)))
        for run_id in np.unique(window_runs)
    }


def _fit(features: np.ndarray, targets: np.ndarray, run_indices: np.ndarray,
         run_ids: list[str]):
    if (features.ndim != 2 or targets.shape != (len(features), 2)
            or run_indices.shape != (len(features),)
            or not np.isfinite(features).all()
            or not np.isfinite(targets).all()):
        raise ValueError("invalid raw encoder transition residual rows")
    counts = np.bincount(run_indices, minlength=len(run_ids)).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError("a declared training run has no valid raw-wheel targets")
    weights = 1.0 / counts[run_indices]
    weights *= len(weights) / weights.sum()
    models, diagnostics = [], {
        "rows": int(len(features)),
        "run_count": int(len(np.unique(run_indices))),
        "per_head_training": {},
    }
    for index, name in enumerate(("left_raw_encoder", "right_raw_encoder")):
        model = HistGradientBoostingRegressor(
            loss="squared_error", learning_rate=0.05, max_iter=180,
            max_leaf_nodes=15, min_samples_leaf=60, l2_regularization=2.0,
            early_stopping=False, random_state=20261012 + index)
        model.fit(features, targets[:, index], sample_weight=weights)
        error = targets[:, index] - model.predict(features)
        diagnostics["per_head_training"][name] = {
            "target_rmse_mps": float(np.sqrt(np.mean(targets[:, index] ** 2))),
            "residual_rmse_mps": float(np.sqrt(np.mean(error ** 2))),
        }
        models.append(model)
    return models, diagnostics


def _capture_training(source: dict[str, Any], hgb: dict[str, Any],
                      edssm: Any, metadata: dict[str, Any],
                      atlas: dict[str, Any], yaw_model: Any,
                      gate: dict[str, tuple[float, float]],
                      run_to_index: dict[str, int]):
    features, targets, run_rows = [], [], []
    windows = source["windows"]
    for offset in range(0, len(windows), CHUNK_SIZE):
        batch_windows = windows[offset:offset + CHUNK_SIZE]
        result = ev._rollout(
            source["data"], hgb, edssm, metadata, atlas,
            batch_windows, 1,
            yaw_delta_gain=1.0,
            include_oracle_diagnostics=False,
            reconcile_composed_latent=True,
            atlas_acceleration_source="composed_transition",
            yaw_correction_model=yaw_model,
            yaw_correction_start_step=0,
            hgb_main_gate=gate,
            hgb_main_residual_start_step=0,
            hgb_v_increment_residual_start_step=0,
            lateral_v_residual_start_step=0,
            capture_raw_wheel_residual_rows=True,
            raw_wheel_residual_start_step=0,
            lateral_v_update="edssm")
        x = result["_raw_wheel_residual_training_features"]
        y = result["_raw_wheel_residual_training_targets"]
        local = result["_raw_wheel_residual_training_run_indices"]
        ids = np.asarray(source["data"]["run_ids"]).astype(str)
        runs = np.asarray(
            [run_to_index[str(ids[index])] for index in local], dtype=np.int32)
        features.append(x)
        targets.append(y)
        run_rows.append(runs)
        done = min(offset + len(batch_windows), len(windows))
        if done == len(windows) or done % (CHUNK_SIZE * 8) == 0:
            print(f"captured {source['name']} raw-wheel transitions "
                  f"{done}/{len(windows)}", flush=True)
    return (np.concatenate(features), np.concatenate(targets),
            np.concatenate(run_rows))


def _one_step_sources(hgb: dict[str, Any]) -> list[dict[str, Any]]:
    specs = (
        ("broad_openplane_validation", ev.HGB_TRAIN_DATASET, "validation"),
        ("highsteer_validation", ev.HIGHSTEER_DATASET, "validation"),
        ("unseen_practice_diagnostic", ev.PRACTICE_DATASET,
         "unseen_practice"),
    )
    result = []
    for name, path, split in specs:
        data = _load_dataset(path)
        windows = _run_windows(data, split, ev._bundle_lags(hgb),
                               2_000_000, horizon_steps=1)
        raw_valid = np.asarray(data["encoder_raw_valid"], dtype=bool)
        windows = [row for row in windows
                   if raw_valid[row["start"]]
                   and raw_valid[row["start"] + 1]]
        splits = np.asarray(data["splits"]).astype(str)
        if not windows or any(splits[row["run"]] != split for row in windows):
            raise ValueError(f"invalid one-step raw validation rows for {name}")
        result.append({
            "name": name, "split": split, "data": data,
            "windows": windows,
            "run_ids": sorted({str(data["run_ids"][row["run"]])
                               for row in windows}),
        })
    return result


def _one_step_metrics(source: dict[str, Any], hgb: dict[str, Any],
                      edssm: Any, metadata: dict[str, Any], atlas: dict[str, Any],
                      yaw_model: Any, gate: dict[str, tuple[float, float]],
                      raw_correction: Any | None) -> dict[str, Any]:
    evaluator_horizons = ev.HORIZONS
    ev.HORIZONS = (1,)
    ids = np.asarray(source["data"]["run_ids"]).astype(str)
    sums: dict[str, dict[str, float]] = {}
    counts: dict[str, dict[str, int]] = {}
    try:
        for offset in range(0, len(source["windows"]), CHUNK_SIZE):
            windows = source["windows"][offset:offset + CHUNK_SIZE]
            result = ev._rollout(
                source["data"], hgb, edssm, metadata, atlas, windows, 1,
                yaw_delta_gain=1.0,
                include_oracle_diagnostics=False,
                reconcile_composed_latent=True,
                atlas_acceleration_source="composed_transition",
                yaw_correction_model=yaw_model,
                yaw_correction_start_step=0,
                hgb_main_gate=gate,
                hgb_main_residual_start_step=0,
                hgb_v_increment_residual_start_step=0,
                lateral_v_residual_start_step=0,
                raw_wheel_residual_models=raw_correction,
                raw_wheel_residual_start_step=0,
                capture_trajectory_trace=True,
                lateral_v_update="edssm")
            trace = result["_trajectory_trace"]
            p = np.asarray(trace["predicted_state"], dtype=np.float64)[:, 0]
            t = np.asarray(trace["truth_state"], dtype=np.float64)[:, 0]
            rp = np.asarray(trace["predicted_raw_encoder_wheels"], dtype=np.float64)[:, 0]
            rt = np.asarray(trace["truth_raw_encoder_wheels"], dtype=np.float64)[:, 0]
            rv = np.asarray(trace["raw_encoder_target_valid"], dtype=bool)[:, 0]
            run_indices = np.asarray(trace["run_indices"], dtype=int)
            for run in np.unique(run_indices):
                mask = run_indices == run
                run_id = str(ids[run])
                bucket = sums.setdefault(run_id, {})
                denom = counts.setdefault(run_id, {})
                terms = {
                    "u_rmse_mps": p[mask, 0] - t[mask, 0],
                    "v_rmse_mps": p[mask, 1] - t[mask, 1],
                    "yaw_rate_rmse_rps": p[mask, 2] - t[mask, 2],
                    "filtered_wheel_pair_rmse_mps": p[mask, 5:7] - t[mask, 5:7],
                    "raw_encoder_pair_rmse_mps": rp[mask] - rt[mask],
                }
                valid_mask = rv[mask]
                for metric, error in terms.items():
                    if metric == "raw_encoder_pair_rmse_mps":
                        error = error[valid_mask]
                    bucket[metric] = bucket.get(metric, 0.0) + float(
                        np.sum(np.square(error)))
                    denom[metric] = denom.get(metric, 0) + int(error.size)
    finally:
        ev.HORIZONS = evaluator_horizons
    per_run = {}
    for run_id, values in sums.items():
        per_run[run_id] = {
            metric: (None if counts[run_id][metric] == 0 else
                     float(np.sqrt(total / counts[run_id][metric])))
            for metric, total in values.items()}
    macro = {}
    for metric in next(iter(per_run.values())):
        numeric = [row[metric] for row in per_run.values()
                   if row[metric] is not None]
        macro[metric] = float(np.mean(numeric)) if numeric else None
    return {"windows": len(source["windows"]), "run_count": len(per_run),
            "per_run": per_run, "macro_run_rmse": macro}


def run(additional_train_datasets: tuple[Path, ...] = (),
        output_dir: Path = OUT) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    model_path = output_dir / MODEL_PATH.name
    report_path = output_dir / REPORT_PATH.name
    recursive_checkpoint = output_dir / RECURSIVE_CHECKPOINT.name
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite {report_path}")
    resume = model_path.exists()
    if output_dir.exists() and not resume:
        raise FileExistsError(f"refusing to reuse incomplete output {output_dir}")
    required = (
        YAW_DIR / "evaluation.json",
        YAW_DIR / "full_regime_yaw_residual.joblib",
        ev.HGB_BUNDLE, ev.EDSSM_CHECKPOINT, ev.YAW_ATLAS_BUNDLE,
        ev.HGB_TRAIN_DATASET, HIGHSTEER_TRAIN_DATASET,
        ev.HIGHSTEER_DATASET, ev.PRACTICE_DATASET)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    yaw_report = json.loads(required[0].read_text())
    if yaw_report.get("test_or_final_test_used") is not False:
        raise ValueError("yaw-residual provenance check failed")
    hgb = joblib.load(ev.HGB_BUNDLE)
    atlas = joblib.load(ev.YAW_ATLAS_BUNDLE)
    for model in (list(atlas["global_event_models"].values())
                  + list(atlas["local_expert_models"].values())):
        if hasattr(model, "n_jobs"):
            model.n_jobs = 1
    _, edssm, metadata = _load_model(ev.EDSSM_CHECKPOINT, "cpu")
    yaw_model = joblib.load(required[1])["estimator"]
    sources = [
        _training_source_all_transitions(path, name, hgb)
        for path, name in (
            (ev.HGB_TRAIN_DATASET, "broad_train"),
            (HIGHSTEER_TRAIN_DATASET, "highsteer_train"))]
    for path in additional_train_datasets:
        source = _training_source_all_transitions(
            path, f"additional_train_{path.parent.name}", hgb)
        sources.append(source)
    owners: dict[str, str] = {}
    for source in sources:
        for run_id in source["run_ids"]:
            if run_id in owners:
                raise ValueError(f"duplicate training run {run_id}")
            owners[run_id] = source["name"]
    training_ids = sorted(owners)
    run_to_index = {run_id: i for i, run_id in enumerate(training_ids)}
    gate = _training_gate(sources)
    if resume:
        bundle = joblib.load(model_path)
        if (bundle.get("hgb_bundle_sha256") != sha256(ev.HGB_BUNDLE)
                or bundle.get("yaw_residual_sha256") != sha256(required[1])
                or bundle.get("training_run_ids") != training_ids
                or bundle.get("training_gate")
                != {key: list(value) for key, value in gate.items()}):
            raise ValueError("saved raw-wheel model does not match training inputs")
        models = bundle["models"]
        source_reports = [{
            "name": source["name"],
            "transitions": sum(_training_target_counts(
                source, run_to_index).values()),
            "run_ids": sorted(source["run_ids"]),
            "per_run_valid_target_counts": _training_target_counts(
                source, run_to_index),
        } for source in sources]
        fit_report = bundle.get("fit_report", {
            "rows": sum(row["transitions"] for row in source_reports),
            "run_count": len(training_ids),
            "fit_diagnostics": (
                "not persisted by the first fit before its replay error; "
                "saved fitted model is being evaluated without refitting"),
        })
        print("resuming held-out evaluation from the saved fitted raw-wheel model",
              flush=True)
    else:
        xs, ys, run_rows, source_reports = [], [], [], []
        for source in sources:
            x, y, runs = _capture_training(
                source, hgb, edssm, metadata, atlas, yaw_model,
                gate, run_to_index)
            xs.append(x)
            ys.append(y)
            run_rows.append(runs)
            source_reports.append({
                "name": source["name"],
                "transitions": len(x),
                "run_ids": sorted(source["run_ids"]),
                "per_run_valid_target_counts": {
                    run_id: int(np.count_nonzero(runs == run_to_index[run_id]))
                    for run_id in sorted(source["run_ids"])},
            })
        with threadpool_limits(limits=2):
            models, fit_report = _fit(
                np.concatenate(xs), np.concatenate(ys),
                np.concatenate(run_rows), training_ids)
        output_dir.mkdir(parents=True, exist_ok=False)
        joblib.dump({
            "schema_version": 1,
            "models": models,
            "fit_report": fit_report,
            "training_sources": source_reports,
            "research_only": True,
            "production_or_runtime_integration": False,
            "target": "raw encoder measured next speed minus current raw speed and frozen base raw-wheel increment",
            "training_run_ids": training_ids,
            "training_gate": {key: list(value) for key, value in gate.items()},
            "hgb_bundle_sha256": sha256(ev.HGB_BUNDLE),
            "yaw_residual_sha256": sha256(required[1]),
        }, model_path)

    recursive = None
    if recursive_checkpoint.is_file():
        saved = json.loads(recursive_checkpoint.read_text())
        if (saved.get("candidate_sha256") != sha256(model_path)
                or saved.get("training_run_ids") != training_ids
                or saved.get("training_gate")
                != {key: list(value) for key, value in gate.items()}):
            raise ValueError("recursive checkpoint does not match this candidate")
        recursive = saved["recursive_results"]
        print("resuming after saved recursive held-out replays", flush=True)
    if recursive is None:
        recursive = {}
        for source in _validation_sources(hgb):
            if set(source["run_ids"]) & set(training_ids):
                raise ValueError(f"held-out overlap for {source['name']}")
            print(f"recursive paired replay: {source['name']} "
                  f"({len(source['windows'])} windows)", flush=True)
            base, base_meta = score(source, hgb, edssm, metadata, atlas,
                                    yaw_model, gate, None)
            candidate, candidate_meta = score(
                source, hgb, edssm, metadata, atlas, yaw_model,
                gate, None, raw_wheel_correction=models)
            deltas = {}
            for horizon, values in candidate.items():
                deltas[horizon] = {}
                for metric in STATE_METRICS:
                    base_value = base[horizon]["macro_run_rmse"][metric]
                    candidate_value = values["macro_run_rmse"][metric]
                    base_runs = base[horizon]["per_run"]
                    candidate_runs = values["per_run"]
                    common_runs = sorted(set(base_runs) & set(candidate_runs))
                    per_run = {
                        run: (None if base_runs[run][metric] is None
                              or candidate_runs[run][metric] is None else
                              candidate_runs[run][metric]
                              - base_runs[run][metric])
                        for run in common_runs}
                    deltas[horizon][metric] = {
                        "macro_run_candidate_minus_baseline": (
                            None if base_value is None or candidate_value is None
                            else candidate_value - base_value),
                        "per_run_candidate_minus_baseline": per_run,
                    }
            recursive[source["name"]] = {
                "split": source["split"], "run_ids": source["run_ids"],
                "windows": len(source["windows"]), "baseline": base,
                "candidate": candidate, "paired_delta": deltas,
                "baseline_meta": base_meta, "candidate_meta": candidate_meta,
            }
        recursive_checkpoint.write_text(json.dumps({
            "candidate_sha256": sha256(model_path),
            "training_run_ids": training_ids,
            "training_gate": {key: list(value) for key, value in gate.items()},
            "recursive_results": recursive,
        }, indent=2, sort_keys=True) + "\n")

    one_step = {}
    for source in _one_step_sources(hgb):
        if set(source["run_ids"]) & set(training_ids):
            raise ValueError(f"one-step held-out overlap for {source['name']}")
        baseline = _one_step_metrics(
            source, hgb, edssm, metadata, atlas, yaw_model, gate, None)
        candidate = _one_step_metrics(
            source, hgb, edssm, metadata, atlas, yaw_model, gate, models)
        per_run = {}
        for run_id in sorted(set(baseline["per_run"]) & set(candidate["per_run"])):
            per_run[run_id] = {
                metric: (None if baseline["per_run"][run_id][metric] is None
                         or candidate["per_run"][run_id][metric] is None else
                         candidate["per_run"][run_id][metric]
                         - baseline["per_run"][run_id][metric])
                for metric in baseline["macro_run_rmse"]}
        macro_delta = {
            metric: (None if baseline["macro_run_rmse"][metric] is None
                     or candidate["macro_run_rmse"][metric] is None else
                     candidate["macro_run_rmse"][metric]
                     - baseline["macro_run_rmse"][metric])
            for metric in baseline["macro_run_rmse"]}
        one_step[source["name"]] = {
            "split": source["split"],
            "raw_target_valid_transitions_only": True,
            "baseline": baseline, "candidate": candidate,
            "macro_run_candidate_minus_baseline": macro_delta,
            "per_run_candidate_minus_baseline": per_run,
        }
    report = {
        "schema_version": 1,
        "purpose": "improve raw wheel-speed sensor prediction while separating it from filtered odometry wheel state",
        "simulator_was_run": False,
        "test_or_final_test_used": False,
        "training_split_only": True,
        "yaw_atlas_frozen": True,
        "yaw_residual_frozen": True,
        "filtered_wheel_model_frozen": True,
        "training_run_ids": training_ids,
        "training_sources": source_reports,
        "training_gate": {key: list(value) for key, value in gate.items()},
        "fit_report": fit_report,
        "all_valid_one_step_transition_results": one_step,
        "recursive_same_window_results": recursive,
        "candidate_model": str(model_path.relative_to(ROOT)),
        "candidate_sha256": sha256(model_path),
        "limitations": [
            "raw encoder sidecar is a sensor measurement, not simulator wheel-rotation ground truth",
            "two practice captures are a small transfer diagnostic",
            "candidate is not integrated into odometry/MPC or the offline plant runtime",
        ],
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if recursive_checkpoint.exists():
        recursive_checkpoint.unlink()
    return report


def _training_source_all_transitions(path: Path, name: str,
                                     hgb: dict[str, Any]) -> dict[str, Any]:
    data = _load_dataset(path)
    windows = _run_windows(data, "train", ev._bundle_lags(hgb),
                           2_000_000, horizon_steps=1)
    splits = np.asarray(data["splits"]).astype(str)
    if not windows or any(splits[row["run"]] != "train" for row in windows):
        raise ValueError(f"all-transition source {name} leaked a split")
    return {"name": name, "data": data, "windows": windows,
            "run_ids": {str(data["run_ids"][row["run"]])
                        for row in windows}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--additional-train-dataset", type=Path, action="append",
                        default=[], help="additional whole-run train-only dataset with raw encoder sidecar")
    parser.add_argument("--output-dir", type=Path, default=OUT,
                        help="new candidate directory; existing contents are never overwritten")
    args = parser.parse_args()
    result = run(tuple(args.additional_train_dataset), args.output_dir)
    print(json.dumps({
        "report": str(args.output_dir.resolve() / REPORT_PATH.name),
        "training_rows": result["fit_report"]["rows"],
        "one_step_delta": {
            name: values["macro_run_candidate_minus_baseline"]
            for name, values in result[
                "all_valid_one_step_transition_results"].items()},
        "recursive_5s_delta": {
            name: {metric: delta["macro_run_candidate_minus_baseline"]
                   for metric, delta in values["paired_delta"]["5.000s"].items()}
            for name, values in result[
                "recursive_same_window_results"].items()},
    }, indent=2, sort_keys=True))
