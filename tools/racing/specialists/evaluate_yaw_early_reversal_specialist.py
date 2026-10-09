#!/usr/bin/env python3
"""Evaluate an exact-two-packet, early-reversal yaw specialist.

This is an offline model-selection tool only. Every fit and score is restricted
to response phases with exactly two packets. Other response counts are excluded.
The specialist is routed only during the diagnosed 25--50 ms reversal window;
the general candidate remains responsible for all other samples.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    import evaluate_yaw_packet_response_conditioned_models as response
    import evaluate_yaw_targeted_group_candidates as targeted
    import train_yaw_multihorizon_teacher as teacher
except ModuleNotFoundError:
    from tools.racing.specialists import evaluate_yaw_packet_response_conditioned_models as response
    from tools.racing.specialists import evaluate_yaw_targeted_group_candidates as targeted
    from tools.racing.specialists import train_yaw_multihorizon_teacher as teacher


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = (ROOT / "live_runs/racing_model_diagnostics_20261008/"
          "yaw_large_error_audit_v1/early_reversal_specialist_exact2.json")
CSV_OUTPUT = OUTPUT.with_name("early_reversal_specialist_exact2_rows.csv")
WINDOW_MS = (25.0, 50.0)
HOLDOUTS = (
    "openplane_yaw_error_highsteer_reversal_validation_r03_20261008",
    "openplane_yaw_error_packet_phase_validation_r09_20261008",
)


def _metric(error: np.ndarray) -> dict[str, Any]:
    error = np.asarray(error, dtype=np.float64)
    if not len(error):
        return {"samples": 0}
    absolute = np.abs(error)
    return {
        "samples": int(len(error)),
        "rmse_radps": float(np.sqrt(np.mean(error * error))),
        "mae_radps": float(np.mean(absolute)),
        "p95_abs_radps": float(np.quantile(absolute, 0.95)),
        "max_abs_radps": float(np.max(absolute)),
        "samples_over_0p1": int(np.count_nonzero(absolute > 0.1)),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _event_rows(rows: dict[str, Any], phases: dict[str, Any], run_id: str,
                event: str) -> dict[str, np.ndarray]:
    indices, phase_ids, _ = targeted._phase_rows(
        rows, phases, 2, event, require_row_event=True)
    ages = np.asarray([
        (int(rows["sample_time_ns"][index])
         - int(phases[str(label)]["command_receipt_ns"])) / 1.0e6
        for index, label in zip(indices, phase_ids)
    ], dtype=np.float64)
    return {
        "indices": indices,
        "x": rows["x_timed"][indices],
        "y": rows["residual"][indices],
        "age_ms": ages,
        "phase_ids": np.asarray([f"{run_id}:{label}" for label in phase_ids]),
    }


def _fit_predict(data: dict[str, np.ndarray], fit_mask: np.ndarray,
                 test_x: np.ndarray, mirror: bool) -> np.ndarray:
    model = targeted._model()
    x = data["x"][fit_mask]
    y = data["y"][fit_mask]
    phase_ids = data["phase_ids"][fit_mask]
    if not len(y):
        raise ValueError("empty specialist fit set")
    if mirror:
        x = np.concatenate((x, targeted._mirror_features(
            x, response.EXPANDED_HISTORY_LAGS)))
        y = np.concatenate((y, -y))
        phase_ids = np.concatenate((phase_ids, phase_ids))
    model.fit(x, y, sample_weight=targeted._weights(phase_ids))
    return model.predict(test_x).astype(np.float32)


def _fold_predictions(train: dict[str, dict[str, np.ndarray]],
                      held_out: str, base_mirror: bool,
                      specialist_mirror: bool) -> tuple[np.ndarray, np.ndarray]:
    fit_parts = [data for run_id, data in train.items() if run_id != held_out]
    fit = {key: np.concatenate([part[key] for part in fit_parts], axis=0)
           for key in ("x", "y", "age_ms", "phase_ids")}
    val = train[held_out]
    base = _fit_predict(fit, np.ones(len(fit["y"]), dtype=bool),
                        val["x"], base_mirror)
    expert_fit_mask = ((fit["age_ms"] >= WINDOW_MS[0])
                       & (fit["age_ms"] < WINDOW_MS[1]))
    expert_val_mask = ((val["age_ms"] >= WINDOW_MS[0])
                       & (val["age_ms"] < WINDOW_MS[1]))
    expert = _fit_predict(fit, expert_fit_mask,
                          val["x"][expert_val_mask], specialist_mirror)
    hybrid = base.copy()
    hybrid[expert_val_mask] = expert
    return base, hybrid


def _loco(train: dict[str, dict[str, np.ndarray]]) -> dict[str, Any]:
    # Compare routing a separately trained expert against the run-macro-selected
    # general model. Selection is limited to training-capture folds.
    variants = {
        "general_mirrored__early_plain": (True, False),
        "general_mirrored__early_mirrored": (True, True),
        "general_plain__early_mirrored": (False, True),
    }
    folds: dict[str, Any] = {}
    for held_out in sorted(train):
        val = train[held_out]
        truth = val["y"]
        baseline = _fit_predict(
            {key: np.concatenate([part[key] for part in train.values()], axis=0)
             for key in ("x", "y", "age_ms", "phase_ids")},
            np.concatenate([
                np.full(len(part["y"]), run_id != held_out, dtype=bool)
                for run_id, part in train.items()
            ]), val["x"], True)
        record: dict[str, Any] = {
            "baseline_general_mirrored": _metric(baseline - truth),
            "variants": {},
        }
        for name, (base_mirror, expert_mirror) in variants.items():
            base, hybrid = _fold_predictions(
                train, held_out, base_mirror, expert_mirror)
            record["variants"][name] = {
                "general": _metric(base - truth),
                "hybrid": _metric(hybrid - truth),
                "early_window_general": _metric(
                    (base - truth)[(val["age_ms"] >= WINDOW_MS[0])
                                   & (val["age_ms"] < WINDOW_MS[1])]),
                "early_window_hybrid": _metric(
                    (hybrid - truth)[(val["age_ms"] >= WINDOW_MS[0])
                                     & (val["age_ms"] < WINDOW_MS[1])]),
            }
        folds[held_out] = record
    summary: dict[str, Any] = {}
    for name in variants:
        rows = [fold["variants"][name]["hybrid"] for fold in folds.values()]
        baseline_rows = [fold["baseline_general_mirrored"]
                         for fold in folds.values()]
        summary[name] = {
            "run_macro_rmse_radps": float(np.mean(
                [row["rmse_radps"] for row in rows])),
            "total_samples_over_0p1": int(sum(
                row["samples_over_0p1"] for row in rows)),
            "worst_capture_max_abs_radps": float(max(
                row["max_abs_radps"] for row in rows)),
            "baseline_mirrored_run_macro_rmse_radps": float(np.mean(
                [row["rmse_radps"] for row in baseline_rows])),
            "baseline_total_samples_over_0p1": int(sum(
                row["samples_over_0p1"] for row in baseline_rows)),
        }
    selected = min(summary, key=lambda name: (
        summary[name]["total_samples_over_0p1"],
        summary[name]["run_macro_rmse_radps"],
        summary[name]["worst_capture_max_abs_radps"],
    ))
    return {
        "selection_rule": "training-only LOCO: fewest samples over 0.1 rad/s, then lowest run-macro RMSE, then lowest worst-capture maximum",
        "selected_variant": selected,
        "summary": summary,
        "folds": folds,
    }


def _score_holdout(train: dict[str, dict[str, np.ndarray]],
                   held_rows: dict[str, Any], held_phases: dict[str, Any],
                   held_id: str, selected: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    fit_parts = list(train.values())
    fit = {key: np.concatenate([part[key] for part in fit_parts], axis=0)
           for key in ("x", "y", "age_ms", "phase_ids")}
    val = _event_rows(held_rows, held_phases, held_id, "reversal")
    base_mirror, expert_mirror = {
        "general_mirrored__early_plain": (True, False),
        "general_mirrored__early_mirrored": (True, True),
        "general_plain__early_mirrored": (False, True),
    }[selected]
    baseline = _fit_predict(fit, np.ones(len(fit["y"]), dtype=bool),
                            val["x"], base_mirror)
    early = (val["age_ms"] >= WINDOW_MS[0]) & (val["age_ms"] < WINDOW_MS[1])
    expert_fit = (fit["age_ms"] >= WINDOW_MS[0]) & (fit["age_ms"] < WINDOW_MS[1])
    expert = _fit_predict(fit, expert_fit, val["x"][early], expert_mirror)
    hybrid = baseline.copy()
    hybrid[early] = expert
    truth = val["y"]
    report: dict[str, Any] = {
        "samples_exact_two_packet_reversal": int(len(truth)),
        "early_window_samples": int(np.count_nonzero(early)),
        "baseline_general": _metric(baseline - truth),
        "hybrid": _metric(hybrid - truth),
        "by_age": {},
    }
    for label, low, high in (("0_to_25ms", 0.0, 25.0),
                             ("25_to_50ms", 25.0, 50.0),
                             ("50_to_150ms", 50.0, 150.0),
                             ("150ms_plus", 150.0, float("inf"))):
        mask = (val["age_ms"] >= low) & (val["age_ms"] < high)
        if np.any(mask):
            report["by_age"][label] = {
                "general": _metric((baseline - truth)[mask]),
                "hybrid": _metric((hybrid - truth)[mask]),
            }
    rows = []
    for index, source_index in enumerate(val["indices"]):
        phase_label = str(next(
            label for label, phase in held_phases.items()
            if int(source_index) in set(map(int, phase["indices"]))
            and phase["true_steps"] == 2 and phase["event"] == "reversal"
            and held_rows["event"][source_index] == "reversal"))
        phase = held_phases[phase_label]
        rows.append({
            "run_id": held_id,
            "event": "reversal",
            "response_packet_count": 2,
            "phase": phase_label,
            "age_from_command_receipt_ms": float(val["age_ms"][index]),
            "truth_residual_radps": float(truth[index]),
            "general_prediction_residual_radps": float(baseline[index]),
            "hybrid_prediction_residual_radps": float(hybrid[index]),
            "general_abs_error_radps": float(abs(baseline[index] - truth[index])),
            "hybrid_abs_error_radps": float(abs(hybrid[index] - truth[index])),
            "requested_speed_mps": float(phase["condition"]["speed_mps"]),
            "requested_abs_steering_rad": float(
                phase["condition"]["steering_abs_rad"]),
            "turn_sign": int(phase["condition"]["turn_sign"]),
            "transition_mode": str(phase["condition"]["transition_mode"]),
            "transition_duration_s": float(phase["condition"]["duration_s"]),
        })
    return report, rows


def run() -> dict[str, Any]:
    admitted, source_audit = teacher._discover_series_with_safe_mixed_archives()
    collected, phase_audits = response._collect_exact_two_packet(admitted)
    rows_by_run, phases_by_run = collected["rows"], collected["phases"]
    train_ids = sorted(run_id for run_id in response.TIMING_SUFFIX
                       if "_train_" in run_id)
    train_ids = [run_id for run_id in train_ids if run_id in rows_by_run]
    for holdout in HOLDOUTS:
        if holdout not in rows_by_run:
            raise ValueError(f"missing registered holdout: {holdout}")
    train = {run_id: _event_rows(
        rows_by_run[run_id], phases_by_run[run_id], run_id, "reversal")
        for run_id in train_ids}
    loco = _loco(train)
    reports: dict[str, Any] = {}
    row_output: list[dict[str, Any]] = []
    for holdout in HOLDOUTS:
        report, rows = _score_holdout(
            train, rows_by_run[holdout], phases_by_run[holdout], holdout,
            loco["selected_variant"])
        reports[holdout] = report
        row_output.extend(rows)
    result = {
        "title": "Exact-two-packet early high-steering reversal specialist",
        "status": "offline research candidate; not integrated into odometry or MPC",
        "target": "next 25-ms simulator-truth yaw-rate residual relative to current exact-packet IMU yaw rate (rad/s)",
        "packet_policy": "only exactly two packets are fitted or scored; every other packet count is excluded",
        "routing": {
            "event": "causal reversal event label from current/past sensor-command history",
            "age_window_ms_inclusive_exclusive": list(WINDOW_MS),
            "window_basis": "pre-registered from independent r03/r10 residual localization; no window search on these holdouts",
        },
        "features": "current/past permitted sensor and command history plus causal receipt/source timing; no future steering, simulator truth, or bridge debug input",
        "training_runs": train_ids,
        "selected_by_training_only_loco": loco,
        "holdouts": reports,
        "excluded_non_two_packet_phase_counts": {
            run_id: phase_audits[run_id].get("response_packet_count_histogram", {})
            for run_id in rows_by_run if run_id in phase_audits
        },
        "data_admission": source_audit,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(_json_safe(result), indent=2) + "\n",
                      encoding="utf-8")
    with CSV_OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        fields = list(row_output[0]) if row_output else []
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(row_output)
    return result


if __name__ == "__main__":
    report = run()
    print(json.dumps({
        "report": str(OUTPUT.relative_to(ROOT)),
        "rows": str(CSV_OUTPUT.relative_to(ROOT)),
        "selected": report["selected_by_training_only_loco"]["selected_variant"],
        "loco": report["selected_by_training_only_loco"]["summary"],
        "holdouts": report["holdouts"],
    }, indent=2))
