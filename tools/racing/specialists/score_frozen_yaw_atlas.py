#!/usr/bin/env python3
"""Score a saved local yaw-atlas report on one clean validation run.

This replays the report's frozen per-cell coefficients without fitting or
reading test/final-test archives. Simulator-truth yaw is the one-step label;
the atlas feature contract remains exactly the one recorded in the report.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    import fit_fullband_yaw_regime_atlas as atlas
except ModuleNotFoundError:
    from tools.racing.specialists import fit_fullband_yaw_regime_atlas as atlas


def _metric(errors: list[float] | np.ndarray) -> dict[str, Any]:
    values = np.asarray(errors, dtype=np.float64)
    if not len(values):
        return {"samples": 0}
    absolute = np.abs(values)
    return {
        "samples": int(len(values)),
        "rmse_radps": float(np.sqrt(np.mean(values * values))),
        "mae_radps": float(np.mean(absolute)),
        "p95_abs_radps": float(np.quantile(absolute, 0.95)),
        "max_abs_radps": float(np.max(absolute)),
        "fraction_abs_error_below_0p1": float(np.mean(absolute < 0.1)),
    }


def _models(report: dict[str, Any]) -> dict[tuple[int, ...], dict[str, np.ndarray]]:
    result = {}
    for row in report["coefficient_models"]:
        key = (int(row["speed_cell"]), int(row["steering_cell"]))
        if row["family"] == "phase":
            key += (int(row["phase"]),)
        result[key] = {
            "mean": np.asarray(row["feature_mean"], dtype=np.float64),
            "scales": np.asarray(row["feature_scales"], dtype=np.float64),
            "coefficients": np.asarray(
                row["coefficients_on_delta_yaw_rate"], dtype=np.float64),
        }
    return result


def _predict(model: dict[str, np.ndarray], features: np.ndarray) -> float:
    normalized = (features - model["mean"]) / model["scales"]
    delta = model["coefficients"][0] + np.dot(
        model["coefficients"][1:], normalized)
    return float(features[0] + delta)


def _row_feature_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    names = config.get("feature_names", [])
    actuator_description = config.get("steering_actuator_feature") or ""
    release_rule = config.get("steering_release_rule")
    if release_rule is None:
        release_rule = ("same_sign_release"
                        if "same-sign magnitude release" in actuator_description
                        else "magnitude_decrease")
    return {
        "phase_threshold": float(config["phase_threshold_rad2_per_s"]),
        "include_command_errors": bool(config["command_tracking_errors_included"]),
        "include_command_rates": bool(config["command_slew_rates_included"]),
        "include_rear_wheel_split": bool(config["rear_wheel_split_included"]),
        "include_command_history": bool(config.get("command_history_included", False)),
        "include_predicted_steering_change": bool(
            config.get("predicted_steering_feedback_change_included", False)),
        "include_steering_release_indicator": (
            "predicted_steering_magnitude_decrease" in names
            or "predicted_same_sign_steering_release" in names
            or bool(config.get("predicted_steering_release_indicator_included", False))),
        "steering_release_rule": release_rule,
    }


def score(report_path: Path, run_id: str,
          compare_report_path: Path | None = None) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    compare_report = (json.loads(compare_report_path.read_text(encoding="utf-8"))
                      if compare_report_path is not None else None)
    if compare_report is not None:
        if (report["model"].get("phase_threshold_rad2_per_s")
                != compare_report["model"].get("phase_threshold_rad2_per_s")):
            raise ValueError("cannot compare atlases with different response-phase thresholds")
    run_series, source_audit = atlas._discover_run_series()
    matches = [row for row in run_series
               if row.run_id == run_id and row.split == "validation"]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one clean validation run {run_id!r}; "
            f"found {len(matches)}")

    config = report["model"]
    features, target_delta, cells, phases, _, _, _ = atlas._make_rows(
        matches[0], **_row_feature_kwargs(config))
    target = features[:, 0] + target_delta
    compare_features = None
    if compare_report is not None:
        other_config = compare_report["model"]
        compare_features, compare_delta, compare_cells, compare_phases, _, _, _ = (
            atlas._make_rows(matches[0], **_row_feature_kwargs(other_config)))
        if (not np.array_equal(cells, compare_cells)
                or not np.array_equal(phases, compare_phases)
                or not np.allclose(target_delta, compare_delta, rtol=0.0, atol=0.0)):
            raise ValueError("atlas reports do not produce aligned validation transitions")
    fitted = _models(report)
    compare_fitted = _models(compare_report) if compare_report is not None else None
    error_by_method: dict[str, list[float]] = defaultdict(list)
    persistence_by_method: dict[str, list[float]] = defaultdict(list)
    error_by_phase: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list))
    error_by_band: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list))
    supported_by_method: dict[str, int] = defaultdict(int)
    paired_errors: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list))
    paired_by_phase: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list)))

    rows = zip(features, target, cells, phases)
    if compare_features is not None:
        rows = zip(features, target, cells, phases, compare_features)
    for values in rows:
        if compare_features is None:
            x, truth, (speed_cell, steering_cell), phase = values
            compare_x = None
        else:
            x, truth, (speed_cell, steering_cell), phase, compare_x = values
        keys = {
            "direct_cell": (int(speed_cell), int(steering_cell)),
            "direct_phase": (int(speed_cell), int(steering_cell), int(phase)),
        }
        speed = atlas.SPEED_CENTERS[int(speed_cell)] + x[2]
        steering = atlas.STEERING_CENTERS[int(steering_cell)] + x[3]
        for method, key in keys.items():
            model = fitted.get(key)
            if model is None:
                continue
            error = _predict(model, x) - float(truth)
            error_by_method[method].append(error)
            error_by_phase[method][str(int(phase))].append(error)
            speed_bin = max(0, int(speed // 0.5))
            steer_bin = int(np.clip(round(abs(steering) / 0.025), 0, 21))
            band = (f"{speed_bin * 0.5:.1f}..{(speed_bin + 1) * 0.5:.1f}mps / "
                    f"|delta|={steer_bin * 0.025:.3f}rad")
            error_by_band[method][band].append(error)
            persistence_by_method[method].append(float(x[0] - truth))
            supported_by_method[method] += 1
            if compare_fitted is not None:
                compare_model = compare_fitted.get(key)
                if compare_model is not None:
                    compare_error = _predict(compare_model, compare_x) - float(truth)
                    paired_errors[method]["primary"].append(error)
                    paired_errors[method]["comparison"].append(compare_error)
                    paired_by_phase[method][str(int(phase))]["primary"].append(error)
                    paired_by_phase[method][str(int(phase))]["comparison"].append(
                        compare_error)

    output = {
        "title": "Frozen local yaw-atlas validation replay",
        "model_report": str(report_path),
        "run_id": run_id,
        "run_split": "validation",
        "one_step_period_s": atlas.DT_S,
        "feature_contract": config["feature_names"],
        "future_truth_used_as_input": False,
        "source_audit": source_audit,
        "candidate_rows": int(len(target)),
        "methods": {},
    }
    for method in ("direct_cell", "direct_phase"):
        errors = error_by_method[method]
        output["methods"][method] = {
            "coverage_samples": supported_by_method[method],
            "coverage_fraction": float(supported_by_method[method] / len(target))
            if len(target) else 0.0,
            "metrics": _metric(errors),
            "persistence_on_same_support": _metric(
                persistence_by_method[method]),
            "by_phase": {
                phase: _metric(rows)
                for phase, rows in sorted(error_by_phase[method].items())
            },
            "by_speed_and_abs_steering_sample_band": {
                band: _metric(rows)
                for band, rows in sorted(error_by_band[method].items())
            },
        }
    if compare_report is not None:
        output["same_support_comparison"] = {
            "primary_report": str(report_path),
            "comparison_report": str(compare_report_path),
            "comparison_feature_contract": compare_report["model"]["feature_names"],
            "metrics": {},
        }
        for method in ("direct_cell", "direct_phase"):
            rows = paired_errors[method]
            primary = _metric(rows["primary"])
            comparison = _metric(rows["comparison"])
            output["same_support_comparison"]["metrics"][method] = {
                "samples_scored_by_both": int(len(rows["primary"])),
                "fraction_of_run_scored_by_both": (
                    float(len(rows["primary"]) / len(target)) if len(target) else 0.0),
                "primary": primary,
                "comparison": comparison,
                "primary_minus_comparison_rmse_radps": (
                    primary.get("rmse_radps", float("nan"))
                    - comparison.get("rmse_radps", float("nan"))),
                "by_response_phase": {
                    phase: {
                        "primary": _metric(values["primary"]),
                        "comparison": _metric(values["comparison"]),
                    }
                    for phase, values in sorted(paired_by_phase[method].items())
                },
            }
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True,
                        help="saved fullband_yaw_regime_atlas_report.json")
    parser.add_argument("--run-id", required=True,
                        help="one run already admitted as a clean validation split")
    parser.add_argument("--compare-report", type=Path,
                        help="optionally score a second frozen atlas on identical samples")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = score(args.report, args.run_id, args.compare_report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(
        result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)
    for name, row in result["methods"].items():
        metrics = row["metrics"]
        print(f"{name}: coverage={row['coverage_fraction']:.3%}; "
              f"RMSE={metrics.get('rmse_radps', float('nan')):.5f}; "
              f"p95={metrics.get('p95_abs_radps', float('nan')):.5f}; "
              f"<0.1={metrics.get('fraction_abs_error_below_0p1', float('nan')):.3%}")
    for method, row in result.get("same_support_comparison", {}).get(
            "metrics", {}).items():
        print(f"same-support {method}: n={row['samples_scored_by_both']}; "
              f"primary RMSE={row['primary']['rmse_radps']:.5f}; "
              f"comparison RMSE={row['comparison']['rmse_radps']:.5f}; "
              f"delta={row['primary_minus_comparison_rmse_radps']:+.5f} rad/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
