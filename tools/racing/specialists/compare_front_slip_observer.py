#!/usr/bin/env python3
"""Compare a front-slip-specific causal observer with the shared slip model.

Training and model selection use only the two full-spectrum training captures,
with whole-run out-of-fold predictions. The already-inspected r02 capture is
reported as development evidence only; it is not used to select the candidate.
The active r04 capture is never opened by this script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

try:
    import analyze_yaw_slip_regime_factor as slip
    import evaluate_sensor_only_slip_observer as current
    import evaluate_yaw_full_spectrum_exact_two as evaluator
    import evaluate_yaw_predicted_next_steering as causal
    import fit_sensor_only_yaw_regime_atlas as atlas
    import train_yaw_multihorizon_teacher as teacher
except ModuleNotFoundError:
    from tools.racing.specialists import analyze_yaw_slip_regime_factor as slip
    from tools.racing.specialists import evaluate_sensor_only_slip_observer as current
    from tools.racing.specialists import evaluate_yaw_full_spectrum_exact_two as evaluator
    from tools.racing.specialists import evaluate_yaw_predicted_next_steering as causal
    from tools.racing.specialists import fit_sensor_only_yaw_regime_atlas as atlas
    from tools.racing.specialists import train_yaw_multihorizon_teacher as teacher


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = ROOT / (
    "live_runs/racing_model_diagnostics_20261009/"
    "front_slip_observer_candidate_r02/report.json")
FRONT_MODEL = {
    "n_estimators": 72,
    "max_depth": 12,
    "min_samples_leaf": 4,
    "max_features": 0.8,
    "n_jobs": 4,
    "random_state": 20261009,
}
SLIP_WIDTH = len(slip.SLIP_NAMES)


def _fit_front(data: dict[str, np.ndarray], target: np.ndarray
               ) -> ExtraTreesRegressor:
    """Fit only front lateral slip, excluding rows with invalid GT labels."""
    features = evaluator._model_features(data)
    sensor = features[:, :-len(evaluator.EVENTS)]
    event = features[:, -len(evaluator.EVENTS):]
    mirror_input = np.column_stack((
        sensor, np.zeros((len(sensor), 4), dtype=np.float32)))
    mirrored_sensor = causal._mirror_features(mirror_input)[:, :-4]
    mirrored_target = slip._mirror_slip_features(target)[:, 2:4]
    valid = target[:, SLIP_WIDTH + 2:SLIP_WIDTH + 4] > 0.5
    rows = np.flatnonzero(np.all(valid, axis=1))
    if not len(rows):
        raise ValueError("no rows have both valid front lateral slip labels")

    x = np.concatenate((features[rows], np.column_stack((
        mirrored_sensor[rows], event[rows]))), axis=0)
    y = np.concatenate((target[rows, 2:4], mirrored_target[rows]), axis=0)
    phase_ids = data["phase_id"].astype(str)[rows]
    weights = causal._phase_weights(np.concatenate((phase_ids, phase_ids)))
    return ExtraTreesRegressor(**FRONT_MODEL).fit(
        x, y, sample_weight=weights)


def _fit_shared(data: dict[str, np.ndarray], target: np.ndarray
                ) -> ExtraTreesRegressor:
    return current._fit_slip_observer(data, target)


def _replace_front(shared: np.ndarray, front: np.ndarray) -> np.ndarray:
    result = shared.copy()
    result[:, 2:4] = front
    return result


def _front_metric(data: dict[str, np.ndarray], truth: np.ndarray,
                  prediction: np.ndarray) -> dict[str, Any]:
    valid = np.all(truth[:, SLIP_WIDTH + 2:SLIP_WIDTH + 4] > 0.5, axis=1)
    error = prediction[:, 2:4][valid] - truth[:, 2:4][valid]
    absolute = np.abs(error)

    def metric(values: np.ndarray) -> dict[str, Any]:
        values = np.asarray(values, dtype=np.float64)
        if not len(values):
            return {"samples": 0, "rmse": None, "p95_abs": None}
        return {
            "samples": int(len(values)),
            "rmse": float(np.sqrt(np.mean(values ** 2))),
            "p95_abs": float(np.quantile(np.abs(values), 0.95)),
        }

    result: dict[str, Any] = {"pooled": metric(error)}
    speed = np.asarray(data["gt_speed"], dtype=np.float64)
    steer = np.abs(np.asarray(data["steering"], dtype=np.float64))
    speed_edges = (0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.01)
    steer_edges = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.501)
    for axis, values, edges in (("speed", speed, speed_edges),
                                ("abs_steering", steer, steer_edges)):
        bands: dict[str, Any] = {}
        for low, high in zip(edges[:-1], edges[1:]):
            mask = valid & (values >= low) & (values < high)
            bands[f"{low:g}-{high:g}"] = metric(
                (prediction[:, 2:4][mask] - truth[:, 2:4][mask]).reshape(-1))
        result[axis] = bands
    error_by_wheel = prediction[:, 2:4] - truth[:, 2:4]
    for key, labels in (("by_stimulus_event", data["phase_event"]),
                        ("by_causal_sensor_event", data["causal_event"])):
        result[key] = {}
        labels = np.asarray(labels).astype(str)
        for label in np.unique(labels):
            mask = valid & (labels == label)
            result[key][str(label)] = metric(error_by_wheel[mask].reshape(-1))
    return result


def run(output: Path) -> dict[str, Any]:
    admitted, source_audit = teacher._discover_series_with_safe_mixed_archives()
    by_id = {series.run_id: series for series in admitted}
    train_ids = current.TRAIN_IDS
    validation_id = current.VALIDATION_ID
    if any(by_id[run_id].split != "train" for run_id in train_ids):
        raise ValueError("candidate training source changed split")
    if by_id[validation_id].split != "validation":
        raise ValueError("development source is not an admitted validation run")

    train_runs: dict[str, dict[str, np.ndarray]] = {}
    slip_runs: dict[str, np.ndarray] = {}
    validation: dict[str, np.ndarray] | None = None
    cache_dir = output.parent / "exact_two_row_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for run_id in (*train_ids, validation_id):
        cache_path = cache_dir / f"{run_id}.npz"
        if cache_path.is_file():
            with np.load(cache_path, allow_pickle=False) as cache:
                data = {name: np.asarray(cache[name]) for name in cache.files}
            if ("cache_run_id" not in data
                    or str(data.pop("cache_run_id").item()) != run_id):
                raise ValueError(f"run-cache provenance mismatch: {cache_path}")
            target = data.pop("truth_slip")
            print(f"loaded verified row cache {run_id}", flush=True)
            data["truth_slip"] = target
        else:
            print(f"collecting admitted run {run_id}", flush=True)
            data, _ = evaluator._collect_run(by_id[run_id], atlas.HISTORY_LAGS)
            target, _ = slip._slip_features(data)
            data["truth_slip"] = target
            np.savez(cache_path, cache_run_id=np.asarray(run_id), **data)
        if run_id == validation_id:
            validation = data
        else:
            train_runs[run_id] = data
            slip_runs[run_id] = target
    assert validation is not None

    shared_oof: dict[str, np.ndarray] = {}
    front_oof: dict[str, np.ndarray] = {}
    fold_scores: dict[str, Any] = {}
    for held_id in sorted(train_runs):
        fit_id = next(run_id for run_id in train_runs if run_id != held_id)
        fit_data, held_data = train_runs[fit_id], train_runs[held_id]
        shared = _fit_shared(fit_data, slip_runs[fit_id])
        shared_prediction = shared.predict(
            evaluator._model_features(held_data)).astype(np.float32)
        front_model = _fit_front(fit_data, slip_runs[fit_id])
        front_prediction = front_model.predict(
            evaluator._model_features(held_data)).astype(np.float32)
        candidate_prediction = _replace_front(shared_prediction, front_prediction)
        shared_oof[held_id] = shared_prediction
        front_oof[held_id] = candidate_prediction
        fold_scores[held_id] = {
            "shared_observer": _front_metric(
                held_data, slip_runs[held_id], shared_prediction),
            "front_specific_valid_label_observer": _front_metric(
                held_data, slip_runs[held_id], candidate_prediction),
        }

    ordered_ids = sorted(train_runs)
    train = slip._combine(train_runs)
    train_target = np.concatenate([slip_runs[run_id] for run_id in ordered_ids])
    shared_oof_all = np.concatenate([shared_oof[run_id] for run_id in ordered_ids])
    front_oof_all = np.concatenate([front_oof[run_id] for run_id in ordered_ids])
    final_shared = _fit_shared(train, train_target)
    held_shared = final_shared.predict(
        evaluator._model_features(validation)).astype(np.float32)
    final_front = _fit_front(train, train_target)
    held_front = final_front.predict(
        evaluator._model_features(validation)).astype(np.float32)
    candidate_held = _replace_front(held_shared, held_front)
    validation_target = validation["truth_slip"]

    # Keep candidate yaw fitting honest: the training slip features are OOF.
    # The held-out rows are only scored, never used to fit either component.
    baseline_yaw = evaluator._fit(
        evaluator._model_features(train), train["y"],
        train["phase_id"].astype(str))
    baseline_error = (baseline_yaw.predict(
        evaluator._model_features(validation)) - validation["y"])
    yaw_reports: dict[str, Any] = {"baseline": current._yaw_metric(baseline_error)}
    for name, oof, held in (("shared", shared_oof_all, held_shared),
                            ("front_specific", front_oof_all, candidate_held)):
        yaw_model = slip._fit_with_oracle_features(
            train, oof, slip._mirror_slip_features(oof))
        yaw_error = yaw_model.predict(np.column_stack((
            evaluator._model_features(validation), held))) - validation["y"]
        gated = current._front_lateral_gate(
            validation, validation_target, held, baseline_error, yaw_error)
        yaw_reports[name] = {
            "ungated": current._yaw_metric(yaw_error),
            "front_peak_gated": gated["front_slip_gated_candidate"],
            "front_peak_gate_by_true_slip_regime": gated[
                "true_front_slip_strata"],
            "front_peak_gate_by_event": gated["by_event"],
            "front_peak_gate_condition_bootstrap": gated[
                "gated_minus_baseline_condition_bootstrap"],
            "gate_operating_point": {
                key: gated[key] for key in (
                    "selected_fraction", "truth_high_regime_precision",
                    "truth_high_regime_recall")},
        }

    report = {
        "purpose": "targeted test of whether a front-lateral-only observer trained on valid slip labels improves slip estimation and downstream yaw features",
        "status": "offline development candidate; no odometry or MPC integration",
        "train_run_ids": list(train_ids),
        "development_validation_run_id": validation_id,
        "r04_active_capture_read": False,
        "heldout_model_selection": "none; model structure is evaluated using whole-run OOF on training r01/r03; r02 is development scoring only",
        "front_observer": FRONT_MODEL,
        "fold_front_slip_scores": fold_scores,
        "development_r02_front_slip": {
            "shared_observer": _front_metric(
                validation, validation_target, held_shared),
            "front_specific_valid_label_observer": _front_metric(
                validation, validation_target, candidate_held),
        },
        "development_r02_yaw": yaw_reports,
        "yaw_train_oof_front_slip": {
            "shared": _front_metric(train, train_target, shared_oof_all),
            "front_specific": _front_metric(train, train_target, front_oof_all),
        },
        "source_admission_audit": source_audit,
        "limitations": [
            "Only two independent full-spectrum training captures are available for whole-run observer cross-validation.",
            "r02 has already been used as development evidence elsewhere; do not call its score blind validation.",
            "The GT-derived wheel-slip targets are kinematic proxies, not contact-force measurements.",
            "Do not promote unless the frozen candidate transfers to the still-sealed midpoint r04 capture.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(
        report, indent=2, sort_keys=True, default=slip._json_default) + "\n",
                      encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = run(args.output)
    shared_slip = report["development_r02_front_slip"]["shared_observer"]["pooled"]
    front_slip = report["development_r02_front_slip"][
        "front_specific_valid_label_observer"]["pooled"]
    yaw = report["development_r02_yaw"]
    summary = {
        "report": str(args.output.relative_to(ROOT)),
        "train_run_ids": report["train_run_ids"],
        "development_validation_run_id": report["development_validation_run_id"],
        "r04_active_capture_read": report["r04_active_capture_read"],
        "r02_front_slip_rmse_shared": shared_slip["rmse"],
        "r02_front_slip_rmse_front_specific": front_slip["rmse"],
        "r02_yaw_baseline": yaw["baseline"],
        "r02_yaw_shared_slip_gate": yaw["shared"]["front_peak_gated"],
        "r02_yaw_front_specific_slip_gate": yaw["front_specific"]["front_peak_gated"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True,
                     default=slip._json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
