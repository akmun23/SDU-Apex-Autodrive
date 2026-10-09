#!/usr/bin/env python3
"""Evaluate run-held-out local yaw experts on exact-two response data only.

Reuses the audited exact-two row collector and tests progressively finer
wheel-speed × signed-steering × causal-event regions. Unsupported regions stay
unpredicted; this tool deliberately has no global fallback. Train-only grouped
CV selects whether finer partitioning is justified before the separate
whole-run validation score is inspected.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold

from tools.racing.specialists import audit_yaw_full_domain_exact_two as audit
from tools.racing.specialists import train_yaw_multihorizon_teacher as teacher


ROOT = Path(__file__).resolve().parents[3]
SOURCE_DIR = ROOT / "live_runs/racing_model_diagnostics_20261008/yaw_full_domain_exact_two_v2"
DEFAULT_OUTPUT = SOURCE_DIR / "regime_frontier.json"
CV_ESTIMATORS = 64
FINAL_ESTIMATORS = 160
MODEL_KWARGS = {
    "max_depth": 14,
    "min_samples_leaf": 5,
    "max_features": 0.9,
    "random_state": 20261008,
    "n_jobs": 4,
}
CONFIGS = (
    {"name": "global_event_comparator", "global": True},
    {"name": "coarse_1m_0p10", "speed_width": 1.0,
     "steer_width": 0.10, "min_rows": 160, "min_runs": 2,
     "min_rows_per_run": 20},
    # The Explore car is left/right symmetric on the open plane. Pool the
    # selector cell by |steering|, but retain signed sensor/history features
    # so the expert still learns turn direction and transient state. This
    # specifically tests whether signed-cell sparsity, rather than a new
    # physical regime, is withholding supported predictions.
    {"name": "coarse_1m_0p10_abs_steering", "speed_width": 1.0,
     "steer_width": 0.10, "absolute_steering": True,
     "min_rows": 160, "min_runs": 2, "min_rows_per_run": 20},
    {"name": "medium_0p5m_0p05", "speed_width": 0.5,
     "steer_width": 0.05, "min_rows": 80, "min_runs": 2,
     "min_rows_per_run": 8},
    {"name": "medium_0p5m_0p05_abs_steering", "speed_width": 0.5,
     "steer_width": 0.05, "absolute_steering": True,
     "min_rows": 80, "min_runs": 2, "min_rows_per_run": 8},
    {"name": "fine_0p5m_0p025", "speed_width": 0.5,
     "steer_width": 0.025, "min_rows": 40, "min_runs": 2,
     "min_rows_per_run": 8},
)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _metrics(errors: np.ndarray) -> dict[str, Any]:
    values = np.asarray(errors, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"samples": 0}
    absolute = np.abs(values)
    return {
        "samples": int(len(values)),
        "rmse_radps": float(np.sqrt(np.mean(values * values))),
        "p95_abs_radps": float(np.quantile(absolute, 0.95)),
        "p99_abs_radps": float(np.quantile(absolute, 0.99)),
        "max_abs_radps": float(np.max(absolute)),
        "samples_over_0p1_radps": int(np.count_nonzero(absolute > 0.1)),
        "fraction_at_most_0p1_radps": float(np.mean(absolute <= 0.1)),
    }


def _load_parts(source_dir: Path
                ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    admitted, source_audit = teacher._discover_series_with_safe_mixed_archives()
    admitted_by_id = {series.run_id: series for series in admitted}
    cache = json.loads((source_dir / "phase_audit.json").read_text(encoding="utf-8"))
    cached_ids = set(cache.get("admitted_run_ids", []))
    if not cached_ids <= set(admitted_by_id):
        raise ValueError("cached phase audit contains runs outside the admitted corpus")
    run_audits = {
        run_id: row for run_id, row in cache["runs"].items()
        if run_id in admitted_by_id
    }
    for series in admitted:
        if series.run_id in run_audits:
            continue
        phases, end = audit._read_phases(audit.alignment._bag_for_run(series))
        eligible = [phase for phase in phases
                    if audit._stimuli_for_phase(phase)]
        if eligible:
            run_audits[series.run_id] = audit._measure_run(series, phases, end)
            print(f"measured new held-out run {series.run_id}: "
                  f"{len(run_audits[series.run_id]['measurements'])} "
                  "response events", flush=True)
    parts: dict[str, list[dict[str, np.ndarray]]] = {
        "train": [], "validation": []}
    missing = []
    for run_id, run_audit in sorted(run_audits.items()):
        series = admitted_by_id.get(run_id)
        if series is None or series.split not in parts:
            missing.append(run_id)
            continue
        rows = audit._collect_rows(series, run_audit["measurements"])
        parts[series.split].append(rows)
    if missing:
        raise ValueError(f"phase cache contains non-admitted runs: {missing[:10]}")
    combined: dict[str, dict[str, np.ndarray]] = {}
    for split, split_parts in parts.items():
        if not split_parts:
            raise ValueError(f"no {split} exact-two rows")
        keys = tuple(key for key, value in split_parts[0].items()
                     if isinstance(value, np.ndarray))
        combined[split] = {
            key: np.concatenate([part[key] for part in split_parts], axis=0)
            for key in keys
        }
    if set(np.unique(combined["train"]["run_id"].astype(str))) & set(
            np.unique(combined["validation"]["run_id"].astype(str))):
        raise ValueError("train/validation run leakage")
    return combined["train"], combined["validation"], source_audit


def _cells(rows: dict[str, np.ndarray], config: dict[str, Any]
           ) -> np.ndarray:
    if config.get("global"):
        return np.zeros(len(rows["residual"]), dtype=np.int32)
    speed = np.floor(rows["wheel_speed"] / config["speed_width"]).astype(int)
    steering = rows["steering"]
    if config.get("absolute_steering"):
        steer = np.floor(np.abs(steering) / config["steer_width"] + 0.5).astype(int)
    else:
        steer = np.floor((steering + 0.525)
                         / config["steer_width"]).astype(int)
    event = rows["event"].astype(str)
    # Cell IDs are formed from observable wheel speed/steering and causal event.
    tuples = zip(speed.tolist(), steer.tolist(), event.tolist())
    lookup: dict[tuple[int, int, str], int] = {}
    ids = np.empty(len(speed), dtype=np.int32)
    for index, key in enumerate(tuples):
        ids[index] = lookup.setdefault(key, len(lookup))
    return ids


def _fit_predict(train: dict[str, np.ndarray], test: dict[str, np.ndarray],
                 config: dict[str, Any], estimators: int
                 ) -> tuple[np.ndarray, dict[str, Any]]:
    train_ids = train["run_id"].astype(str)
    train_cells = _cells(train, config)
    test_cells = _cells(test, config)
    prediction = np.full(len(test["residual"]), np.nan, dtype=np.float64)
    support: dict[str, Any] = {}
    unique_train = np.unique(train_cells)
    for cell in unique_train:
        train_mask = train_cells == cell
        test_mask = test_cells == cell
        if config.get("global"):
            # Separate causal event models are required; the event is included
            # in the cell key for local experts below. This comparator groups
            # the event rows directly rather than pooling maneuvers.
            events = np.unique(train["event"][train_mask].astype(str))
            for event in events:
                tm = train_mask & (train["event"].astype(str) == event)
                vm = test_mask & (test["event"].astype(str) == event)
                if not np.any(vm) or not np.any(tm):
                    continue
                run_count = len(np.unique(train_ids[tm]))
                if run_count < 3 or int(tm.sum()) < 300:
                    continue
                model = ExtraTreesRegressor(
                    n_estimators=estimators, **MODEL_KWARGS)
                model.fit(train["x_timed"][tm], train["residual"][tm],
                          sample_weight=audit._run_balanced_weights(train_ids[tm]))
                prediction[vm] = model.predict(test["x_timed"][vm])
            support["global_event_models"] = int(np.isfinite(prediction).sum())
            continue

        # Encode each local selector cell back to its train members. A test
        # cell ID is comparable only when the exact (speed, steer, event) key
        # appears in both sets, so use explicit keys rather than integer IDs.
        # The mapping is reconstructed from the row channels to avoid relying
        # on incidental insertion order.
    if not config.get("global"):
        train_speed = np.floor(train["wheel_speed"] / config["speed_width"]).astype(int)
        if config.get("absolute_steering"):
            train_steer = np.floor(
                np.abs(train["steering"]) / config["steer_width"] + 0.5
            ).astype(int)
            test_steer = np.floor(
                np.abs(test["steering"]) / config["steer_width"] + 0.5
            ).astype(int)
        else:
            train_steer = np.floor((train["steering"] + 0.525)
                                   / config["steer_width"]).astype(int)
            test_steer = np.floor((test["steering"] + 0.525)
                                  / config["steer_width"]).astype(int)
        test_speed = np.floor(test["wheel_speed"] / config["speed_width"]).astype(int)
        train_event = train["event"].astype(str)
        test_event = test["event"].astype(str)
        train_keys = set(zip(train_speed.tolist(), train_steer.tolist(),
                             train_event.tolist()))
        supported_keys = 0
        for speed_cell, steer_cell, event in train_keys:
            tm = ((train_speed == speed_cell) & (train_steer == steer_cell)
                  & (train_event == event))
            vm = ((test_speed == speed_cell) & (test_steer == steer_cell)
                  & (test_event == event))
            if not np.any(vm):
                continue
            counts = Counter(train_ids[tm].tolist())
            keep = {run_id for run_id, count in counts.items()
                    if count >= config["min_rows_per_run"]}
            fit_mask = tm & np.isin(train_ids, list(keep))
            if (int(fit_mask.sum()) < config["min_rows"]
                    or len(keep) < config["min_runs"]):
                continue
            model = ExtraTreesRegressor(
                n_estimators=estimators, **MODEL_KWARGS)
            model.fit(train["x_timed"][fit_mask], train["residual"][fit_mask],
                      sample_weight=audit._run_balanced_weights(train_ids[fit_mask]))
            prediction[vm] = model.predict(test["x_timed"][vm])
            supported_keys += 1
        support["supported_experts"] = supported_keys
    support["covered_rows"] = int(np.isfinite(prediction).sum())
    support["coverage_fraction"] = float(np.isfinite(prediction).mean())
    return prediction, support


def _by_region(rows: dict[str, np.ndarray], errors: np.ndarray) -> dict[str, Any]:
    valid = np.isfinite(errors)
    speed_bin = np.floor(rows["gt_speed"] / 1.0).astype(int)
    steer_bin = np.floor((rows["steering"] + 0.525) / 0.1).astype(int)
    event = rows["response_event"].astype(str)
    keys = set(zip(speed_bin[valid].tolist(), steer_bin[valid].tolist(),
                   event[valid].tolist()))
    result = {}
    for speed, steer, response_event in sorted(keys):
        mask = (valid & (speed_bin == speed) & (steer_bin == steer)
                & (event == response_event))
        result[f"v{speed}-{speed + 1}_d{steer}_event={response_event}"] = {
            "independent_runs": int(len(np.unique(rows["run_id"][mask]))),
            **_metrics(errors[mask]),
        }
    return result


def _run_macro(errors: np.ndarray, run_ids: np.ndarray) -> dict[str, Any]:
    run_rmse = []
    run_max = []
    for run_id in sorted(np.unique(run_ids.astype(str))):
        mask = (run_ids.astype(str) == run_id) & np.isfinite(errors)
        if np.any(mask):
            run_rmse.append(float(np.sqrt(np.mean(errors[mask] ** 2))))
            run_max.append(float(np.max(np.abs(errors[mask]))))
    return {
        "scored_runs": len(run_rmse),
        "run_macro_rmse_radps": float(np.mean(run_rmse)) if run_rmse else None,
        "median_run_rmse_radps": float(np.median(run_rmse)) if run_rmse else None,
        "worst_run_max_abs_radps": float(np.max(run_max)) if run_max else None,
    }


def evaluate(train: dict[str, np.ndarray], validation: dict[str, np.ndarray],
             cv_splits: int,
             cv_reference: dict[str, Any] | None = None) -> dict[str, Any]:
    train_ids = train["run_id"].astype(str)
    training_run_ids = sorted(np.unique(train_ids).tolist())
    unique_runs = np.asarray(training_run_ids)
    if cv_reference is not None:
        if (int(cv_reference.get("training_rows", -1)) != len(train_ids)
                or cv_reference.get("training_run_ids") != training_run_ids):
            raise ValueError("CV reference does not match current training split")
        folds = []
    else:
        if cv_splits < 2 or len(unique_runs) < cv_splits:
            raise ValueError("grouped CV requires at least two folds and runs")
        folds = list(GroupKFold(n_splits=cv_splits).split(
            np.zeros(len(train_ids)), groups=train_ids))
    results: dict[str, Any] = {}
    validation_predictions: dict[str, np.ndarray] = {}
    for config in CONFIGS:
        print(f"evaluate {config['name']}: {len(folds)} run-held-out folds",
              flush=True)
        if cv_reference is not None:
            try:
                cv_result = cv_reference["models"][config["name"]][
                    "train_only_grouped_cv"]
            except KeyError as exc:
                raise ValueError("CV reference is missing a candidate") from exc
        else:
            cv_prediction = np.full(len(train_ids), np.nan, dtype=np.float64)
            fold_rows = []
            for fold_index, (fit_idx, test_idx) in enumerate(folds, 1):
                train_fold = {key: value[fit_idx] for key, value in train.items()}
                test_fold = {key: value[test_idx] for key, value in train.items()}
                pred, support = _fit_predict(
                    train_fold, test_fold, config, CV_ESTIMATORS)
                cv_prediction[test_idx] = pred
                fold_errors = pred - test_fold["residual"]
                fold_rows.append({
                    "fold": fold_index,
                    "heldout_runs": sorted(np.unique(test_fold["run_id"].astype(str))),
                    "support": support,
                    "metrics": _metrics(fold_errors),
                })
            cv_error = cv_prediction - train["residual"]
            cv_result = {
                "folds": fold_rows,
                "metrics": _metrics(cv_error),
                "run_macro": _run_macro(cv_error, train_ids),
                "by_speed_signed_steering_response_event": _by_region(
                    train, cv_error),
            }
        val_prediction, val_support = _fit_predict(
            train, validation, config, FINAL_ESTIMATORS)
        validation_predictions[config["name"]] = val_prediction
        val_error = val_prediction - validation["residual"]
        results[config["name"]] = {
            "configuration": config,
            "train_only_grouped_cv": cv_result,
            "whole_run_validation": {
                "support": val_support,
                "metrics": _metrics(val_error),
                "run_macro": _run_macro(
                    val_error, validation["run_id"].astype(str)),
                "by_speed_signed_steering_response_event": _by_region(
                    validation, val_error),
            },
        }
        print("  CV", results[config["name"]]["train_only_grouped_cv"]["metrics"],
              "VAL", results[config["name"]]["whole_run_validation"]["metrics"],
              flush=True)

    global_prediction = validation_predictions["global_event_comparator"]
    target = validation["residual"]
    for config in CONFIGS:
        name = config["name"]
        if config.get("global"):
            continue
        local_prediction = validation_predictions[name]
        common = np.isfinite(global_prediction) & np.isfinite(local_prediction)
        local_only = ~np.isfinite(global_prediction) & np.isfinite(local_prediction)
        global_only = np.isfinite(global_prediction) & ~np.isfinite(local_prediction)
        results[name]["paired_comparison_to_global_event"] = {
            "common_support_rows": int(common.sum()),
            "global_event_on_common_support": _metrics(
                global_prediction[common] - target[common]),
            "local_expert_on_common_support": _metrics(
                local_prediction[common] - target[common]),
            "local_only_rows": int(local_only.sum()),
            "local_only_metrics": _metrics(
                local_prediction[local_only] - target[local_only]),
            "global_only_rows": int(global_only.sum()),
        }

    for config in CONFIGS:
        if not config.get("absolute_steering"):
            continue
        name = config["name"]
        signed_name = name.replace("_abs_steering", "")
        if signed_name not in validation_predictions:
            continue
        absolute_prediction = validation_predictions[name]
        signed_prediction = validation_predictions[signed_name]
        target = validation["residual"]
        common = (np.isfinite(absolute_prediction)
                  & np.isfinite(signed_prediction))
        absolute_only = (np.isfinite(absolute_prediction)
                         & ~np.isfinite(signed_prediction))
        signed_only = (np.isfinite(signed_prediction)
                       & ~np.isfinite(absolute_prediction))
        common_rows = validation
        results[name]["paired_comparison_to_signed_selector"] = {
            "common_support_rows": int(common.sum()),
            "common_support_signed_selector": _metrics(
                signed_prediction[common] - target[common]),
            "common_support_absolute_selector": _metrics(
                absolute_prediction[common] - target[common]),
            "absolute_selector_only_rows": int(absolute_only.sum()),
            "absolute_selector_only_metrics": _metrics(
                absolute_prediction[absolute_only] - target[absolute_only]),
            "signed_selector_only_rows": int(signed_only.sum()),
            "signed_selector_only_metrics": _metrics(
                signed_prediction[signed_only] - target[signed_only]),
            "absolute_selector_common_support_by_signed_region": _by_region(
                common_rows, np.where(common,
                                      absolute_prediction - target, np.nan)),
        }
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR,
                        help="exact-two audit directory with phase_audit.json")
    parser.add_argument("--output", type=Path,
                        help="output JSON (defaults beside --source-dir)")
    parser.add_argument("--cv-splits", type=int, default=4)
    parser.add_argument("--cv-reference", type=Path,
                        help="reuse grouped CV only when training rows/runs are unchanged")
    args = parser.parse_args()
    source_dir = args.source_dir.resolve()
    output_path = (args.output.resolve() if args.output
                   else source_dir / "regime_frontier.json")
    train, validation, source_audit = _load_parts(source_dir)
    cv_reference = (json.loads(args.cv_reference.read_text(encoding="utf-8"))
                    if args.cv_reference else None)
    print("loaded exact-two rows", len(train["residual"]),
          len(validation["residual"]), "runs",
          len(np.unique(train["run_id"])), len(np.unique(validation["run_id"])),
          flush=True)
    results = evaluate(train, validation, args.cv_splits, cv_reference)
    output = {
        "title": "Exact-two yaw response local-expert support frontier",
        "target": "GT yaw_rate[k+1] minus current exact-source-time IMU yaw_rate[k]",
        "dt_s": 0.025,
        "input_policy": "causal measured sensor/actuator histories only",
        "selection_policy": "causal rear-wheel mean speed × measured steering × causal event",
        "unsupported_regions": "no prediction; no global fallback",
        "packet_policy": "exactly two contiguous command-to-feedback packets only",
        "test_and_final_test_arrays_read": False,
        "cv_policy": (
            "reused grouped-by-run CV only for the identical training run IDs and row count"
            if cv_reference else
            f"{args.cv_splits}-fold grouped-by-run CV within training split"),
        "cv_estimators": CV_ESTIMATORS,
        "validation_estimators": FINAL_ESTIMATORS,
        "training_rows": int(len(train["residual"])),
        "validation_rows": int(len(validation["residual"])),
        "training_runs": int(len(np.unique(train["run_id"]))),
        "training_run_ids": sorted(np.unique(train["run_id"].astype(str)).tolist()),
        "validation_runs": int(len(np.unique(validation["run_id"]))),
        "source_corpus_audit": source_audit,
        "models": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True,
                                      default=_json_default) + "\n",
                           encoding="utf-8")
    print("wrote", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
