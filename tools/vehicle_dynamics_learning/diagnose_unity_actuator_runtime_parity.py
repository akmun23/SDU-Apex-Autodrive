#!/usr/bin/env python3
"""Check source-expert runtime output against causal bag-derived features."""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace
from pathlib import Path
from typing import Any

import numpy as np

from tools.vehicle_dynamics_learning import evaluate_unity_actuator_features_odom as model
from tools.vehicle_dynamics_learning import fit_supported_practice_odom_u_residual as base


def _metrics(values: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(np.mean(values ** 2))),
        "bias": float(np.mean(values)),
        "p95_abs": float(np.quantile(np.abs(values), 0.95)),
        "max_abs": float(np.max(np.abs(values))),
    }


def diagnose(raw: Path, baseline: Path, candidate: Path,
             model_report: Path) -> dict[str, Any]:
    baseline_data = base._load_run(raw, baseline)
    candidate_data = base._load_run(raw, candidate)
    samples, diagnostics, groups, _, stamps = baseline_data
    candidate_samples, candidate_diagnostics, candidate_groups, _, candidate_stamps = candidate_data
    if (not np.array_equal(stamps, candidate_stamps)
            or not np.array_equal(groups, candidate_groups)
            or len(candidate_samples) != len(samples)):
        raise ValueError("baseline and candidate replay do not share exact source samples")

    runtime_model = json.loads(model_report.read_text(encoding="utf-8"))["runtime_model"]
    throttle = np.asarray([sample.actuators[1] for sample in samples], dtype=float)
    brake = throttle == 0.0
    baseline_u = diagnostics[:, 7]
    active = ((baseline_u >= runtime_model["speed_gate_mps"][0]) &
              (baseline_u < runtime_model["speed_gate_mps"][1]))
    actual = candidate_diagnostics[:, 7] - diagnostics[:, 7]
    expected_by_offset = {}
    details_by_offset = {}
    for offset in (-1, 0, 1):
        indices = np.clip(np.arange(len(samples)) + offset, 0, len(samples) - 1)
        shifted_samples = [SimpleNamespace(
            actuators=samples[int(source)].actuators,
            packet_sequence=samples[index].packet_sequence)
            for index, source in enumerate(indices)]
        feature_matrix = np.column_stack((
            base._features(diagnostics), model._features(shifted_samples, diagnostics)))
        shifted_throttle = np.asarray(
            [sample.actuators[1] for sample in shifted_samples], dtype=float)
        shifted_brake = shifted_throttle == 0.0
        expected = np.zeros(len(samples), dtype=float)
        details = {}
        for name, mode in (("unity_zero_throttle_brake", shifted_brake),
                           ("unity_nonzero_motor_torque", ~shifted_brake)):
            expert = runtime_model[name]
            weights = np.asarray(expert["coefficients"], dtype=float)
            mask = active & mode
            expected[mask] = expert["intercept_mps"] + feature_matrix[mask] @ weights
            details[name] = {
                "samples": int(mask.sum()),
                "expected_correction": _metrics(expected[mask]),
            }
        expected_by_offset[str(offset)] = {
            "expected": _metrics(expected),
            "runtime_minus_expected": _metrics(actual - expected),
            "branches": details,
        }
        details_by_offset[str(offset)] = details
    # Recompute the zero-offset prediction for scalar report fields.
    current_features = np.column_stack((
        base._features(diagnostics), model._features(samples, diagnostics)))
    expected = np.zeros(len(samples), dtype=float)
    for name, mode in (("unity_zero_throttle_brake", brake),
                       ("unity_nonzero_motor_torque", ~brake)):
        expert = runtime_model[name]
        weights = np.asarray(expert["coefficients"], dtype=float)
        mask = active & mode
        expected[mask] = expert["intercept_mps"] + current_features[mask] @ weights
    residual = actual - expected
    return {
        "raw_bag": str(raw.resolve()),
        "baseline_replay": str(baseline.resolve()),
        "candidate_replay": str(candidate.resolve()),
        "samples": len(samples),
        "active_samples": int(active.sum()),
        "actual_correction": _metrics(actual),
        "expected_correction": _metrics(expected),
        "runtime_minus_offline_prediction": _metrics(residual),
        "by_unity_branch": details_by_offset["0"],
        "actuator_packet_offset_ablation": expected_by_offset,
        "non_u_diagnostic_max_abs_delta": {
            str(index): float(np.max(np.abs(
                candidate_diagnostics[:, index] - diagnostics[:, index])))
            for index in range(candidate_diagnostics.shape[1])
            if index != 7
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--model-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = diagnose(args.raw, args.baseline, args.candidate, args.model_report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps({
        key: value for key, value in report.items()
        if key not in ("non_u_diagnostic_max_abs_delta", "by_unity_branch")
    } | {"by_unity_branch": report["by_unity_branch"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
