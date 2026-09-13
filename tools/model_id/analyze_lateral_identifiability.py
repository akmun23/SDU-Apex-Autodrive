#!/usr/bin/env python3
"""Check whether lateral ``I_z`` and tire stiffness are identifiable.

This is an offline diagnostic.  It profiles yaw inertia while refitting the
two tire stiffnesses, then reports the local sensitivity condition number.
It prevents a numerically good but physically arbitrary ``I_z``/tire split
from being promoted into the native plant.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_lateral_model import (  # noqa: E402
    _candidate_parameters,
    _examples,
    _one_step_scores,
    _predict_lateral,
    _read_runs,
)


def _residuals(rows: list[dict[str, float]], values: np.ndarray) -> np.ndarray:
    output = np.empty(2 * len(rows), dtype=float)
    for index, row in enumerate(rows):
        predicted_v, predicted_r = _predict_lateral(
            row, values, "linear_saturated")
        output[2 * index] = (predicted_v - row["v_k1_mps"]) / 0.05
        output[2 * index + 1] = (predicted_r - row["r_k1_radps"]) / 0.10
    return output


def _profile(train_rows: list[dict[str, float]], validation_rows: list[dict[str, float]],
             inertia_values: list[float]) -> list[dict[str, Any]]:
    # A deterministic, broad subset is sufficient for the profile and keeps
    # this diagnostic practical. Final candidate scoring still uses all rows.
    stride = max(1, len(train_rows) // 6000)
    fit_rows = train_rows[::stride]
    profiles = []
    for inertia in inertia_values:
        def residual(stiffness: np.ndarray) -> np.ndarray:
            return _residuals(
                fit_rows,
                np.asarray([stiffness[0], stiffness[1], inertia]))

        result = least_squares(
            residual, np.asarray([800.0, 800.0]),
            bounds=(np.asarray([1.0, 1.0]), np.asarray([5000.0, 5000.0])),
            x_scale="jac", loss="soft_l1", max_nfev=60)
        values = np.asarray([result.x[0], result.x[1], inertia], dtype=float)
        validation = _one_step_scores(validation_rows, values,
                                      "linear_saturated")
        profiles.append({
            "iz_kgm2": inertia,
            "cf_n_per_rad": float(result.x[0]),
            "cr_n_per_rad": float(result.x[1]),
            "fit_cost": float(result.cost),
            "fit_rmse_normalized": float(np.sqrt(2.0 * result.cost / len(fit_rows))),
            "optimizer_success": bool(result.success),
            "validation_one_step": validation,
        })
    return profiles


def _sensitivity(train_rows: list[dict[str, float]], values: np.ndarray) -> dict[str, Any]:
    stride = max(1, len(train_rows) // 6000)
    rows = train_rows[::stride]
    baseline = _residuals(rows, values)
    columns = []
    names = ("cf_n_per_rad", "cr_n_per_rad", "iz_kgm2")
    steps = (max(1.0e-2, abs(values[0]) * 1.0e-4),
             max(1.0e-2, abs(values[1]) * 1.0e-4),
             max(1.0e-6, abs(values[2]) * 1.0e-4))
    for index, delta in enumerate(steps):
        perturbed = values.copy()
        perturbed[index] += delta
        columns.append((_residuals(rows, perturbed) - baseline) / delta)
    jacobian = np.column_stack(columns)
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    condition = (float(singular_values[0] / singular_values[-1])
                 if singular_values[-1] > 1.0e-12 else float("inf"))
    return {
        "sample_count": len(rows),
        "parameter_names": list(names),
        "singular_values": [float(value) for value in singular_values],
        "condition_number": condition,
        "rank": int(np.linalg.matrix_rank(jacobian, tol=1.0e-8)),
    }


def analyze(root: Path, train_names: list[str], validation_names: list[str],
            lateral_report: Path, output: Path) -> dict[str, Any]:
    report = json.loads(lateral_report.read_text(encoding="utf-8"))
    candidate = report["candidate_comparison"]["Y1_linear_saturated"]
    candidate_values = candidate["parameters"]["parameters"]
    values = np.asarray([
        candidate_values["cf_n_per_rad"], candidate_values["cr_n_per_rad"],
        candidate_values["iz_kgm2"],
    ], dtype=float)
    train_rows = _examples(_read_runs(root, train_names))
    validation_rows = _examples(_read_runs(root, validation_names))
    grid = [0.005, 0.010, 0.020, 0.030, 0.040, 0.050, 0.060, 0.070, 0.080]
    profiles = _profile(train_rows, validation_rows, grid)
    best = min(profiles, key=lambda item: item["fit_cost"])
    at_bound = best["iz_kgm2"] in (min(grid), max(grid))
    costs = [float(item["fit_cost"]) for item in profiles]
    flat = (max(costs) - min(costs)) / max(abs(min(costs)), 1.0e-12) < 0.05
    sensitivity = _sensitivity(train_rows, values)
    identified = (
        not at_bound and not flat and sensitivity["rank"] == 3 and
        sensitivity["condition_number"] < 1.0e4)
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "identified" if identified else "not_identified",
        "ground_truth_use": "offline_identification_and_scoring_only",
        "recursive_prediction_uses_future_gt": False,
        "candidate_source": str(lateral_report),
        "train_runs": train_names,
        "validation_runs": validation_names,
        "profile_parameter": "iz_kgm2",
        "profile_grid_kgm2": grid,
        "profile": profiles,
        "best_profile_point": best,
        "fitted_candidate_point": {
            "cf_n_per_rad": float(values[0]),
            "cr_n_per_rad": float(values[1]),
            "iz_kgm2": float(values[2]),
        },
        "local_sensitivity": sensitivity,
        "decision": {
            "inertia_profile_at_bound": at_bound,
            "profile_is_flat": flat,
            "parameter_split_is_accepted": identified,
            "reason": (
                "I_z and tire stiffness are not separately identifiable under "
                "the current candidate/data excitation; obtain independent "
                "inertia information or richer lateral excitation before plant "
                "promotion."
                if not identified else
                "I_z profile and local sensitivity support separate identification."
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accepted-root", type=Path, required=True)
    parser.add_argument("--train-runs", required=True)
    parser.add_argument("--validation-runs", required=True)
    parser.add_argument("--lateral-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(
        args.accepted_root,
        [value for value in args.train_runs.split(",") if value],
        [value for value in args.validation_runs.split(",") if value],
        args.lateral_report, args.output)
    print(json.dumps({
        "output": str(args.output),
        "status": result["status"],
        "best_profile_point": result["best_profile_point"],
        "condition_number": result["local_sensitivity"]["condition_number"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
