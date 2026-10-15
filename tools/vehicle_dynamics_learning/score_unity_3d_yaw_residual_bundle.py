#!/usr/bin/env python3
"""Score a saved Unity-source plant bundle on whole-run captures.

This separates expensive model fitting from recursive held-out evaluation, so
additional untouched runs can be checked without retraining or model changes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

import evaluate_unity_3d_yaw_residual as evaluator


class EqualWeightEnsemble:
    """A prediction average used only for offline candidate comparison."""

    def __init__(self, estimators: list[object]) -> None:
        self.estimators = estimators

    def predict(self, values: np.ndarray) -> np.ndarray:
        predictions = [np.asarray(model.predict(values), dtype=np.float64)
                       for model in self.estimators]
        return np.mean(np.stack(predictions, axis=0), axis=0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--capture", type=Path, action="append", required=True)
    parser.add_argument(
        "--model", action="append",
        help="bundle model to score; repeat to compare candidates (default: preferred model)")
    parser.add_argument(
        "--mean-model", action="append",
        help="also score an equal-weight prediction mean of these bundle models; repeat at least twice")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon-seconds", type=float, default=5.0)
    parser.add_argument("--window-stride-seconds", type=float, default=1.0)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if not 0.025 <= args.horizon_seconds <= 5.0:
        parser.error("horizon must be between 0.025 and 5 seconds")
    if args.window_stride_seconds <= 0:
        parser.error("window stride must be positive")

    bundle = joblib.load(args.bundle)
    model_names = args.model or [bundle["preferred_model"]]
    if args.mean_model and len(args.mean_model) < 2:
        parser.error("--mean-model must be supplied at least twice")
    selected_names = list(dict.fromkeys([*model_names, *(args.mean_model or [])]))
    unknown_models = set(selected_names) - set(bundle["models"])
    if unknown_models:
        parser.error(f"unknown bundle model(s): {', '.join(sorted(unknown_models))}")
    models = {name: bundle["models"][name] for name in model_names}
    profiled_names = set(bundle.get("profiled_models", []))
    full_profile_names = set(bundle.get("full_profile_models", []))
    if args.mean_model:
        ensemble_name = "equal_mean[" + ",".join(args.mean_model) + "]"
        models[ensemble_name] = EqualWeightEnsemble(
            [bundle["models"][name] for name in args.mean_model])
        model_names.append(ensemble_name)
        if all(name in profiled_names for name in args.mean_model):
            profiled_names.add(ensemble_name)
        if all(name in full_profile_names for name in args.mean_model):
            full_profile_names.add(ensemble_name)
    report = {
        "schema_version": 1,
        "status": "held-out recursive scoring only; no model fitting",
        "bundle": evaluator._path_label(args.bundle),
        "bundle_sha256": evaluator._sha256(args.bundle),
        "models": model_names,
        "training_capture": bundle.get("training_capture"),
        "selection_capture": bundle.get("selection_capture"),
        "horizon_seconds": args.horizon_seconds,
        "window_stride_seconds": args.window_stride_seconds,
        "future_simulator_truth_used_during_rollout": False,
        "captures": {},
    }
    for path in args.capture:
        label = evaluator._path_label(path)
        try:
            scores = evaluator._score(
                path,
                bundle["constants"],
                models,
                args.horizon_seconds,
                args.window_stride_seconds,
                profiled_names & set(model_names),
                full_profile_names & set(model_names),
            )
        except ValueError as exc:
            if "no fully grounded evaluation windows" not in str(exc):
                raise
            report["captures"][label] = {
                "status": "excluded",
                "capture_sha256": evaluator._sha256(path),
                "reason": str(exc),
            }
            continue
        report["captures"][label] = {
            "status": "scored",
            "capture_sha256": evaluator._sha256(path),
            **scores,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(args.output)
    for capture, result in report["captures"].items():
        if result["status"] != "scored":
            print(capture, "excluded:", result["reason"])
            continue
        print(capture, "windows", result["window_count"])
        errors = result["errors"]
        for horizon in ("0.025s", "0.100s", "0.500s", "1.000s", "5.000s"):
            if horizon not in errors["baseline"]:
                continue
            baseline = errors["baseline"][horizon]
            print(
                f"  {horizon}: position "
                f"baseline {baseline['position_horizontal_rmse_m']:.4f} m, "
                f"{baseline['body_omega_xyz_rmse_radps'][1]:.4f} rad/s")
            for model_name in model_names:
                candidate = errors[model_name][horizon]
                print(
                    f"    {model_name}: "
                    f"{candidate['position_horizontal_rmse_m']:.4f} m, "
                    f"{candidate['body_omega_xyz_rmse_radps'][1]:.4f} rad/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
