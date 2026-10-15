#!/usr/bin/env python3
"""Compare Unity-guide tire-slip equations in free 1 kHz source rollouts.

Uses only source commands after each window's initial truth state. Two held-out
captures are scored under the existing formulation and documented slip ratios,
with the original versus guide-equation force-balance coefficients separated.
This is an offline research comparison, not runtime odometry or a Unity clone.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tools.vehicle_dynamics_learning import replay_unity_source_3d_plant as plant


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_R11 = ROOT / (
    "live_runs/unity_exact_binary_audit_20261012/"
    "dynamics_source_exact_replay_20261012_r11/"
    "wheel_dynamics_unity_vehicle_model_parity_source_exact_replay_20261012_r11.csv")
DEFAULT_R12G = ROOT / (
    "live_runs/unity_mechanics_dynamic_validation_20261013_r12g/"
    "unity_source_capture/"
    "wheel_dynamics_unity_mechanics_dynamic_validation_r12g.csv")
DEFAULT_GUIDE_FIT = ROOT / (
    "live_runs/unity_exact_binary_audit_20261012/"
    "tire_curve_force_balance_guide_slips_20261013.json")
DEFAULT_OUTPUT = ROOT / (
    "live_runs/unity_exact_binary_audit_20261012/"
    "documented_tire_slip_recursive_comparison_20261013.json")


def _evaluate_capture(path: Path, train_path: Path,
                      window_stride_s: float,
                      horizon_s: float) -> dict[str, Any]:
    metadata = plant._metadata(train_path)
    variants = (
        ("baseline_existing_slips_and_coefficients",
         plant.DEFAULT_TIRE_FIT, False, False),
        ("documented_slips_existing_coefficients",
         plant.DEFAULT_TIRE_FIT, True, False),
        ("documented_slips_guide_fitted_coefficients",
         DEFAULT_GUIDE_FIT, True, True),
    )
    results: dict[str, Any] = {}
    for name, tire_fit, documented_slips, tire_lateral in variants:
        constants, model = plant._prepare_constants(
            metadata, tire_fit, plant.DEFAULT_SLIP_FIT,
            plant.DEFAULT_SUSPENSION_FIT, plant.DEFAULT_WHEEL_FIT,
            use_tire_fit_lateral=tire_lateral)
        evaluation = plant._evaluate_capture(
            path, constants, window_stride_s, horizon_s,
            documented_tire_slips=documented_slips)
        results[name] = {
            "parameter_sources": {
                "tire_fit": str(tire_fit.relative_to(ROOT)),
                "lateral_curve": model["lateral_curve_parameter_source"],
                "slip_regularizer": str(plant.DEFAULT_SLIP_FIT.relative_to(ROOT)),
            },
            **evaluation,
        }
    # All variants must be scored on identical clean window starts.
    starts = [
        [row["start_sample"] for row in results[name]["window_diagnostics"]]
        for name, *_ in variants
    ]
    if any(candidate != starts[0] for candidate in starts[1:]):
        raise RuntimeError(f"candidate window sets differ for {path}")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=plant.DEFAULT_TRAIN)
    parser.add_argument("--capture", type=Path, action="append",
                        help="held-out 1 kHz source capture; repeatable")
    parser.add_argument("--window-stride-seconds", type=float, default=5.0)
    parser.add_argument("--horizon-seconds", type=float, default=5.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    captures = args.capture or [DEFAULT_R11, DEFAULT_R12G]
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if not DEFAULT_GUIDE_FIT.is_file():
        raise FileNotFoundError(DEFAULT_GUIDE_FIT)
    report = {
        "status": "held-out recursive model comparison; no runtime model changed",
        "training_capture_for_metadata_and_fits": str(args.train.relative_to(ROOT)),
        "training_only_guide_tire_fit": str(DEFAULT_GUIDE_FIT.relative_to(ROOT)),
        "future_inputs": "recorded steering and throttle at 1 ms; truth used only for each window initial state and scoring",
        "window_stride_s": args.window_stride_seconds,
        "horizon_s": args.horizon_seconds,
        "variants": {
            "baseline_existing_slips_and_coefficients": "existing relative-speed longitudinal slip; atan2 lateral slip; original tire coefficients",
            "documented_slips_existing_coefficients": "guide Sx=(r*wheelOmega-vx)/max(|vx|,regularizer), Sy=vy/max(|vx|,regularizer); original tire coefficients",
            "documented_slips_guide_fitted_coefficients": "same guide slips; force-balance coefficients refit on training capture using only rows with |vx|>1 m/s; fitted lateral coefficients used",
        },
        "captures": {},
        "limitations": [
            "the guide slip equations are singular at standstill; the existing independently fitted 0.1005 m/s regularizer is used only as the denominator floor",
            "the documented equations do not establish Unity WheelCollider's private native slip regularization or complete force solver",
            "5 s windowed rollouts are not full-lap validation",
        ],
    }
    for capture in captures:
        key = str(capture.relative_to(ROOT))
        report["captures"][key] = _evaluate_capture(
            capture, args.train, args.window_stride_seconds,
            args.horizon_seconds)
        for variant, result in report["captures"][key].items():
            last = result["rollout_errors"][
                f"{args.horizon_seconds:.3f}s"]
            print(key, variant,
                  "windows", result["window_count"],
                  "5s_pos_rmse", round(
                      last["position_horizontal_rmse_m"], 4),
                  "heading_rmse", round(last["heading_rmse_rad"], 5),
                  "u_rmse", round(
                      last["body_velocity_xyz_rmse_mps"][2], 4),
                  "wheel_rmse", [round(value, 3) for value in
                                  last["wheel_omega_each_rmse_radps"]])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
