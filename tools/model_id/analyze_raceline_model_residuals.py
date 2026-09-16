#!/usr/bin/env python3
"""Diagnose reachable one-step plant residuals on the raceline envelope.

This is an offline mechanism diagnostic for the active 0.75 s model path.  It
scores only CORE/GUARD origins from the candidate's validation split and
conditions the causal one-step ``u/v/r`` prediction residual on speed, slip,
steering, throttle, and run.  The leave-one-run-out regressions are evidence
for or against a missing mechanism; they do not create a runtime correction.

Simulator truth is used only as the offline next-state target.  No simulator,
production MPC, estimator, or controller file is modified.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable, Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_speed_regime_vehicle_model import (  # noqa: E402
    MAX_SPEED_MPS,
    MAX_STEERING_RAD,
    _load_runs,
    _raceline_context,
    _stats,
    _steering_command,
)
from score_raceline_model import (  # noqa: E402
    _nearest_reference,
    operating_class,
)
from structured_vehicle_plant import (  # noqa: E402
    MIN_SLIP_SPEED_MPS,
    PlantParameters,
    step,
)


RESIDUALS = ("u_error_mps", "v_error_mps", "r_error_radps")
FEATURES = (
    "speed_mps", "abs_yaw_rate_radps", "abs_steering_rad",
    "abs_commanded_steering_rate_radps", "wheel_slip_mps",
    "front_slip_angle_rad", "rear_slip_angle_rad", "throttle_norm",
    "throttle_abs_steering",
)
TRACK_S_BIN_WIDTH_M = 1.0
FEATURE_BINS: dict[str, tuple[float, ...]] = {
    "speed_mps": (2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0),
    "abs_yaw_rate_radps": (0.0, 0.25, 0.5, 1.0, 2.0),
    "abs_steering_rad": (0.0, 0.02, 0.05, 0.15, 0.30, 0.60),
    "abs_commanded_steering_rate_radps": (0.0, 0.25, 0.75, 1.5, 3.2, 8.0),
    "wheel_slip_mps": (-2.0, -0.5, -0.1, 0.1, 0.5, 2.0),
    "front_slip_angle_rad": (-0.30, -0.10, -0.03, 0.03, 0.10, 0.30),
    "rear_slip_angle_rad": (-0.30, -0.10, -0.03, 0.03, 0.10, 0.30),
    "throttle_norm": (0.0, 0.01, 0.20, 0.50, 0.80, 1.0),
    "throttle_abs_steering": (0.0, 0.01, 0.03, 0.08, 0.20, 0.60),
}


def _condition(value: float, bins: tuple[float, ...]) -> str:
    for lower, upper in zip(bins, bins[1:]):
        if lower <= value < upper:
            return f"[{lower:g},{upper:g})"
    if value < bins[0]:
        return f"below_{bins[0]:g}"
    return f"[{bins[-1]:g},plus)"


def _parameters(candidate: dict[str, Any]) -> PlantParameters:
    base = PlantParameters.from_manifest()
    values = candidate["candidate_parameters"]
    return PlantParameters(**{
        **base.__dict__,
        "steering_dynamics_kind": str(values["steering_dynamics_kind"]),
        "steering_lag_time_constant_s": float(
            values.get("steering_lag_time_constant_s", 0.0)),
        "tire_model": "regime_speed_combined_tanh",
        "lateral_cf_regimes_n_per_rad": tuple(
            float(value) for value in values["lateral_cf_regimes_n_per_rad"]),
        "lateral_cr_regimes_n_per_rad": tuple(
            float(value) for value in values["lateral_cr_regimes_n_per_rad"]),
        "lateral_df_regimes_n": tuple(
            float(value) for value in values["lateral_df_regimes_n"]),
        "lateral_dr_regimes_n": tuple(
            float(value) for value in values["lateral_dr_regimes_n"]),
        "combined_slip_gain": float(values["combined_slip_gain"]),
        "steering_rate_force_gain_n_per_radps": float(
            values.get("steering_rate_force_gain_n_per_radps", 0.0)),
        "steering_rate_moment_gain_nm_per_radps": float(
            values.get("steering_rate_moment_gain_nm_per_radps", 0.0)),
        "force_max_regimes_n": tuple(
            float(value) for value in values["force_max_regimes_n"]),
        "slip_gain_regimes_per_mps": tuple(
            float(value) for value in values["slip_gain_regimes_per_mps"]),
        "coast_speed_drag_n_per_mps": 0.0,
    })


def _residual_rows(candidate: dict[str, Any], context: dict[str, Any],
                   runs: dict[str, list[dict[str, float]]]) -> dict[
                       str, list[dict[str, float | str]]]:
    parameters = _parameters(candidate)
    output: dict[str, list[dict[str, float | str]]] = {}
    for run_name, rows in runs.items():
        run_output: list[dict[str, float | str]] = []
        for row in rows:
            if (not math.isfinite(row["wheel_k_mps"]) or
                    not math.isfinite(row["delta_k_rad"]) or
                    not (0.0 <= row["u_k_mps"] <= MAX_SPEED_MPS)):
                continue
            class_name = operating_class(
                row["x_k_m"], row["y_k_m"], row["u_k_mps"],
                context["raceline"], context["core_bins"], context["guard_bins"],
                context["speed_bin_width_mps"],
                context["curvature_bin_width_radpm"], MAX_SPEED_MPS)
            if class_name not in {"core", "guard"}:
                continue
            target_delta = (max(-1.0, min(1.0, _steering_command(row))) *
                            MAX_STEERING_RAD)
            dt = row["dt_sim_s"]
            if not math.isfinite(dt) or dt <= 0.0:
                continue
            predicted = step(
                np.asarray([
                    row["x_k_m"], row["y_k_m"], row["yaw_k_rad"],
                    row["u_k_mps"], row["v_k_mps"], row["r_k_radps"],
                    row["delta_k_rad"], row["wheel_k_mps"],
                ], dtype=float),
                _steering_command(row),
                row["applied_throttle_norm_k1"], dt, parameters)
            safe_u = math.copysign(
                max(abs(row["u_k_mps"]), MIN_SLIP_SPEED_MPS),
                row["u_k_mps"] or 1.0)
            front_slip = row["delta_k_rad"] - math.atan2(
                row["v_k_mps"] + parameters.lf_m * row["r_k_radps"], safe_u)
            rear_slip = -math.atan2(
                row["v_k_mps"] - parameters.lr_m * row["r_k_radps"], safe_u)
            steering_rate = (target_delta - row["delta_k_rad"]) / dt
            reference_index = _nearest_reference(
                row["x_k_m"], row["y_k_m"], context["raceline"])
            run_output.append({
                "run": run_name,
                "operating_class": class_name,
                "reference_s_m": float(
                    context["raceline"]["s_m"][reference_index]),
                "speed_mps": row["u_k_mps"],
                "abs_yaw_rate_radps": abs(row["r_k_radps"]),
                "abs_steering_rad": abs(row["delta_k_rad"]),
                "abs_commanded_steering_rate_radps": abs(steering_rate),
                "wheel_slip_mps": row["wheel_k_mps"] - row["u_k_mps"],
                "front_slip_angle_rad": front_slip,
                "rear_slip_angle_rad": rear_slip,
                "throttle_norm": row["applied_throttle_norm_k1"],
                "throttle_abs_steering": (
                    row["applied_throttle_norm_k1"] * abs(row["delta_k_rad"])),
                "u_error_mps": predicted[3] - row["u_k1_mps"],
                "v_error_mps": predicted[4] - row["v_k1_mps"],
                "r_error_radps": predicted[5] - row["r_k1_radps"],
            })
        output[run_name] = run_output
    return output


def _conditioned(rows: Iterable[dict[str, float | str]], feature: str,
                 residual: str) -> dict[str, Any]:
    grouped: dict[str, list[float]] = {}
    bins = FEATURE_BINS[feature]
    for row in rows:
        value = float(row[feature])
        grouped.setdefault(_condition(value, bins), []).append(
            float(row[residual]))
    return {key: _stats(values) for key, values in sorted(grouped.items())}


def _track_bins(rows: Iterable[dict[str, float | str]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, float | str]]] = {}
    for row in rows:
        lower = math.floor(float(row["reference_s_m"]) /
                           TRACK_S_BIN_WIDTH_M) * TRACK_S_BIN_WIDTH_M
        key = f"s[{lower:g},{lower + TRACK_S_BIN_WIDTH_M:g})"
        grouped.setdefault(key, []).append(row)
    return {
        key: {
            "sample_count": len(values),
            "operating_classes": {
                class_name: sum(
                    row["operating_class"] == class_name for row in values)
                for class_name in ("core", "guard")
            },
            "residuals": {
                residual: _stats(float(row[residual]) for row in values)
                for residual in RESIDUALS
            },
            "mean": {
                feature: float(np.mean([float(row[feature]) for row in values]))
                for feature in FEATURES
            },
        }
        for key, values in sorted(grouped.items())
    }


def _leave_one_run_out(rows_by_run: dict[str, list[dict[str, float | str]]],
                       feature: str, residual: str,
                       value_function: Callable[[dict[str, float | str]], float]
                       ) -> dict[str, Any]:
    folds: list[dict[str, Any]] = []
    for held_out, test_rows in rows_by_run.items():
        train_rows = [row for name, rows in rows_by_run.items()
                      if name != held_out for row in rows]
        if len(train_rows) < 20 or len(test_rows) < 20:
            continue
        x_train = np.asarray([value_function(row) for row in train_rows])
        y_train = np.asarray([float(row[residual]) for row in train_rows])
        x_test = np.asarray([value_function(row) for row in test_rows])
        y_test = np.asarray([float(row[residual]) for row in test_rows])
        design_train = np.column_stack((np.ones(len(x_train)), x_train))
        design_test = np.column_stack((np.ones(len(x_test)), x_test))
        coefficients, _, _, _ = np.linalg.lstsq(
            design_train, y_train, rcond=None)
        prediction = design_test @ coefficients
        baseline_rmse = float(np.sqrt(np.mean(y_test * y_test)))
        augmented_rmse = float(np.sqrt(np.mean((y_test - prediction) ** 2)))
        folds.append({
            "held_out_run": held_out,
            "train_samples": len(train_rows),
            "test_samples": len(test_rows),
            "baseline_rmse": baseline_rmse,
            "augmented_rmse": augmented_rmse,
            "relative_rmse_improvement": (
                (baseline_rmse - augmented_rmse) /
                max(baseline_rmse, 1.0e-12)),
            "coefficients": [float(value) for value in coefficients],
        })
    improvements = [fold["relative_rmse_improvement"] for fold in folds]
    return {
        "folds": folds,
        "mean_relative_rmse_improvement": (
            float(np.mean(improvements)) if improvements else None),
        "all_folds_improve": bool(improvements) and all(
            value > 0.0 for value in improvements),
    }


def analyze(candidate_path: Path, output: Path) -> dict[str, Any]:
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    context = _raceline_context(
        Path(candidate["operating_envelope"]["raceline_file"]))
    runs = _load_runs(candidate["validation_specs"])
    residual_runs = _residual_rows(candidate, context, runs)
    rows = [row for run_rows in residual_runs.values() for row in run_rows]
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "offline_core_guard_residual_diagnostic",
        "candidate": str(candidate_path),
        "simulator_modified": False,
        "production_mpc_modified": False,
        "ground_truth_use": "offline_one_step_residual_target_only",
        "operating_classes": ["core", "guard"],
        "transition_count": len(rows),
        "per_run": {
            name: {
                "sample_count": len(run_rows),
                "residuals": {
                    residual: _stats(float(row[residual]) for row in run_rows)
                    for residual in RESIDUALS
                },
            }
            for name, run_rows in residual_runs.items()
        },
        "overall": {
            residual: _stats(float(row[residual]) for row in rows)
            for residual in RESIDUALS
        },
        "per_track_s_bin": _track_bins(rows),
        "conditioned": {
            class_name: {
                residual: {
                    feature: _conditioned(
                        [row for row in rows
                         if row["operating_class"] == class_name],
                        feature, residual)
                    for feature in FEATURES
                }
                for residual in RESIDUALS
            }
            for class_name in ("core", "guard")
        },
        "leave_one_run_out": {
            class_name: {
                residual: {
                    feature: _leave_one_run_out(
                        {
                            name: [row for row in run_rows
                                   if row["operating_class"] == class_name]
                            for name, run_rows in residual_runs.items()
                        },
                        feature, residual, lambda row, f=feature: float(row[f]))
                    for feature in FEATURES
                }
                for residual in RESIDUALS
            }
            for class_name in ("core", "guard")
        },
        "interpretation": {
            "purpose": "identify a repeatable missing mechanism",
            "do_not_fit_runtime_correction": True,
            "candidate_selection_requires": [
                "same-sign effect across held-out runs",
                "improvement on the 0.75 s CORE/GUARD recursive score",
                "no stability or guard regression",
            ],
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.candidate, args.output)
    print(json.dumps({
        "output": str(args.output),
        "transition_count": result["transition_count"],
        "overall": result["overall"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
