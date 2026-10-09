#!/usr/bin/env python3
"""Compare exact-two-packet yaw experts split by measured speed/steering.

Offline diagnostic only. Uses current/past permitted channels and a causal
one-step steering rule; simulator truth is the target. No runtime component is
modified. Every training, CV, and validation row is gated to exactly two
packets by the packet-response collector.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

try:
    import evaluate_yaw_packet_response_conditioned_models as response
    import evaluate_yaw_predicted_next_steering as causal
    import fit_sensor_only_yaw_regime_atlas as atlas
    import train_yaw_multihorizon_teacher as teacher
except ModuleNotFoundError:
    from tools.racing.specialists import evaluate_yaw_packet_response_conditioned_models as response
    from tools.racing.specialists import evaluate_yaw_predicted_next_steering as causal
    from tools.racing.specialists import fit_sensor_only_yaw_regime_atlas as atlas
    from tools.racing.specialists import train_yaw_multihorizon_teacher as teacher


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = (ROOT / "live_runs/racing_model_diagnostics_20261008/"
                  "yaw_large_error_audit_v1/"
                  "exact2_speed_steering_local_model_audit.json")
TRAIN_IDS = tuple(sorted(
    run_id for run_id in response.TIMING_SUFFIX if "_train_" in run_id))
VALIDATION_IDS = (
    "openplane_yaw_error_highsteer_reversal_validation_r03_20261008",
    "openplane_yaw_error_packet_phase_validation_r09_20261008",
)
GROUPS = ((2, "reversal"), (2, "unwind"))
ACTUATOR_RULE = "hybrid_release_or_limit_k_minus_1"
MODEL = {
    "n_estimators": 96,
    "max_depth": 10,
    "min_samples_leaf": 5,
    "max_features": 0.9,
    "random_state": 20261008,
    "n_jobs": 4,
}
VARIANTS = ("global", "speed_3", "steering_3", "speed_steering_9")
MIN_CELL_ROWS = 32
MIN_CELL_PHASES = 8


def _cell_ids(x: np.ndarray, variant: str) -> np.ndarray:
    """Choose experts using only measured wheel speed and steering feedback."""
    width = len(atlas.OBSERVATION_NAMES)
    n_history = causal._history_block_count(x)
    speed = x[:, n_history * width]
    steering = np.abs(x[:, 0])
    speed_bin = np.digitize(speed, (3.25, 3.75))
    steering_bin = np.digitize(steering, (0.385, 0.46))
    if variant == "global":
        return np.zeros(len(x), dtype=np.int64)
    if variant == "speed_3":
        return speed_bin
    if variant == "steering_3":
        return steering_bin
    if variant == "speed_steering_9":
        return speed_bin * 3 + steering_bin
    raise ValueError(f"unknown model variant: {variant}")


def _phase_ids(run_id: str, data: dict[str, np.ndarray]) -> np.ndarray:
    return np.asarray([f"{run_id}:{phase}" for phase in data["phase_id"]],
                      dtype="U160")


def _fit_model(x: np.ndarray, y: np.ndarray,
               phase_ids: np.ndarray) -> ExtraTreesRegressor:
    return causal._fit(x, y, phase_ids, mirror=True, signed_tail=True)


def _fit_predict(train_runs: dict[str, dict[str, np.ndarray]],
                 validation: dict[str, np.ndarray], variant: str
                 ) -> tuple[np.ndarray, dict[str, Any]]:
    x_train = np.concatenate([data["x"] for data in train_runs.values()])
    y_train = np.concatenate([data["y"] for data in train_runs.values()])
    rule_train = np.concatenate([
        data["actuator_rules"][ACTUATOR_RULE]
        for data in train_runs.values()])
    phases = np.concatenate([_phase_ids(run_id, data)
                             for run_id, data in train_runs.items()])
    features = np.column_stack((x_train, rule_train))
    validation_features = np.column_stack((
        validation["x"], validation["actuator_rules"][ACTUATOR_RULE]))
    train_cells = _cell_ids(x_train, variant)
    validation_cells = _cell_ids(validation["x"], variant)
    global_model = _fit_model(features, y_train, phases)
    global_prediction = global_model.predict(validation_features)
    if variant == "global":
        return global_prediction, {
            "expert_cells": 1,
            "fallback_rows": 0,
            "cell_support": {"global": {"rows": int(len(y_train)),
                                          "phases": int(len(np.unique(phases)))}}}

    prediction = np.asarray(global_prediction, dtype=np.float64).copy()
    fallback_rows = 0
    cell_support: dict[str, Any] = {}
    for cell in np.unique(validation_cells):
        train_idx = np.flatnonzero(train_cells == cell)
        validation_idx = np.flatnonzero(validation_cells == cell)
        phase_count = int(len(np.unique(phases[train_idx])))
        supported = (len(train_idx) >= MIN_CELL_ROWS
                     and phase_count >= MIN_CELL_PHASES)
        cell_support[str(int(cell))] = {
            "rows": int(len(train_idx)),
            "phases": phase_count,
            "independent_runs": int(len({
                str(value).split(":", 1)[0] for value in phases[train_idx]})),
            "supported": bool(supported),
        }
        if not supported:
            fallback_rows += len(validation_idx)
            continue
        model = _fit_model(features[train_idx], y_train[train_idx],
                           phases[train_idx])
        prediction[validation_idx] = model.predict(
            validation_features[validation_idx])
    return prediction, {
        "expert_cells": int(sum(cell["supported"]
                                for cell in cell_support.values())),
        "fallback_rows": int(fallback_rows),
        "cell_support": cell_support,
    }


def _capture_data(run_data: dict[str, Any], phases: dict[str, Any],
                  future_steering: np.ndarray, step: int, event: str
                  ) -> dict[str, np.ndarray]:
    return causal._group_data(run_data, phases, future_steering, step, event,
                              atlas.HISTORY_LAGS)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def run(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    admitted, source_audit = teacher._discover_series_with_safe_mixed_archives()
    collected, phase_audits = response._collect_exact_two_packet(admitted)
    rows_by_run, phases_by_run = collected["rows"], collected["phases"]
    series_by_id = {series.run_id: series for series in admitted}
    future_by_run = {
        run_id: causal._next_steering_rows(series_by_id[run_id])
        for run_id in (*TRAIN_IDS, *VALIDATION_IDS)
        if run_id in series_by_id
    }
    missing = (set(TRAIN_IDS) | set(VALIDATION_IDS)) - set(future_by_run)
    if missing:
        raise ValueError(f"missing admitted runs: {sorted(missing)}")
    run_data: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for step, event in GROUPS:
        run_data[event] = {
            run_id: _capture_data(rows_by_run[run_id], phases_by_run[run_id],
                                 future_by_run[run_id], step, event)
            for run_id in (*TRAIN_IDS, *VALIDATION_IDS)
        }

    report: dict[str, Any] = {
        "title": "Exact-two-packet yaw response: measured speed/steering local experts",
        "status": "offline experiment; no runtime component modified",
        "target": "next 25-ms simulator-GT yaw-rate residual, rad/s",
        "packet_policy": "only exactly two packets included; all other counts excluded",
        "test_and_final_test_opened": False,
        "validation_runs": list(VALIDATION_IDS),
        "training_runs": list(TRAIN_IDS),
        "actuator_feature": ACTUATOR_RULE,
        "actuator_feature_is_causal": True,
        "cell_selection_inputs": ["measured rear-wheel mean speed", "measured steering feedback"],
        "speed_bin_edges_mps": [3.25, 3.75],
        "absolute_steering_bin_edges_rad": [0.385, 0.46],
        "minimum_local_support": {"rows": MIN_CELL_ROWS,
                                  "independent_phase_rows": MIN_CELL_PHASES},
        "source_audit": source_audit,
        "phase_audits": phase_audits,
        "groups": {},
    }
    for _, event in GROUPS:
        train_by_run = {run_id: run_data[event][run_id]
                        for run_id in TRAIN_IDS}
        selection = causal._training_only_loco(train_by_run, event)
        group_report: dict[str, Any] = {
            "training_only_actuator_rule_selection": selection[
                "selected_rule_by_training_only_loco"],
            "variants": {},
        }
        for variant in VARIANTS:
            fold_metrics: dict[str, Any] = {}
            for held_out in TRAIN_IDS:
                fit_runs = {run_id: train_by_run[run_id]
                            for run_id in TRAIN_IDS if run_id != held_out}
                pred, support = _fit_predict(
                    fit_runs, train_by_run[held_out], variant)
                fold_metrics[held_out] = {
                    "metrics": causal._metric(pred - train_by_run[held_out]["y"]),
                    **support,
                }
            macro_rmse = float(np.mean([
                row["metrics"]["rmse_radps"]
                for row in fold_metrics.values()]))
            total_over = int(sum(row["metrics"]["samples_over_0p1"]
                                 for row in fold_metrics.values()))
            worst = float(max(row["metrics"]["max_abs_radps"]
                              for row in fold_metrics.values()))
            group_report["variants"][variant] = {
                "training_only_loco": {
                    "run_macro_rmse_radps": macro_rmse,
                    "total_samples_over_0p1": total_over,
                    "worst_capture_max_abs_radps": worst,
                    "heldout_capture_metrics": fold_metrics,
                },
                "whole_run_validation": {},
            }
            for validation_id in VALIDATION_IDS:
                pred, support = _fit_predict(
                    train_by_run, run_data[event][validation_id], variant)
                group_report["variants"][variant][
                    "whole_run_validation"][validation_id] = {
                        "metrics": causal._metric(
                            pred - run_data[event][validation_id]["y"]),
                        **support,
                    }
        selected = min(VARIANTS, key=lambda variant: (
            group_report["variants"][variant]["training_only_loco"][
                "total_samples_over_0p1"],
            group_report["variants"][variant]["training_only_loco"][
                "run_macro_rmse_radps"],
            group_report["variants"][variant]["training_only_loco"][
                "worst_capture_max_abs_radps"],
        ))
        group_report["selected_by_training_only_loco"] = selected
        report["groups"][event] = group_report

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True,
                                 default=_json_default) + "\n",
                      encoding="utf-8")
    summary = {
        event: {
            "selected": group["selected_by_training_only_loco"],
            "variants": {
                variant: {
                    "loco_over_0p1": details["training_only_loco"][
                        "total_samples_over_0p1"],
                    "loco_macro_rmse": details["training_only_loco"][
                        "run_macro_rmse_radps"],
                    "r03": details["whole_run_validation"][
                        VALIDATION_IDS[0]]["metrics"],
                    "r09": details["whole_run_validation"][
                        VALIDATION_IDS[1]]["metrics"],
                }
                for variant, details in group["variants"].items()
            },
        }
        for event, group in report["groups"].items()
    }
    print(json.dumps({"output": str(output), "summary": summary}, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
