#!/usr/bin/env python3
"""Compare discrete and smooth throttle-response models by held-out capture.

Only training-split rows are loaded. Mirrored left/right probes are pooled
within each capture and condition, so repeated samples within one bag do not
inflate the number of independent observations. This is a training-only model
screen; a separate whole-capture validation cohort is required before using
the selected model as a controller or plant component.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from tools.racing.build_empirical_throttle_response_model import (
    WINDOWED_METRICS,
    _read_split_rows,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
BOOTSTRAP_SEED = 20261007
BOOTSTRAP_DRAWS = 20_000
TARGETS = (
    "wheel_residual_early_mps", "wheel_residual_middle_mps",
    "wheel_residual_late_mps", "wheel_slip_ratio_early",
    "wheel_slip_ratio_middle", "wheel_slip_ratio_late",
    "longitudinal_accel_early_mps2", "longitudinal_accel_middle_mps2",
    "longitudinal_accel_late_mps2",
)


def _cell_key(row: dict[str, Any]) -> tuple[float | str, ...]:
    return (
        round(float(row["speed_target_mps"]), 3),
        str(row["throttle_delta_direction"]),
        round(float(row["throttle_delta_norm"]), 3),
        round(float(row["throttle_rise_rate_norm_per_sec"]), 3),
        round(abs(float(row["abs_steering_command_rad"])), 3),
    )


def _make_examples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, tuple[float | str, ...]], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["run_id"]), _cell_key(row))].append(row)

    examples = []
    for (run_id, cell), members in sorted(grouped.items()):
        first = members[0]
        direction = 1.0 if first["throttle_delta_direction"] == "up" else -1.0
        features = [
            float(first["speed_target_mps"]),
            direction * float(first["throttle_delta_norm"]),
            direction * float(first["throttle_rise_rate_norm_per_sec"]),
            abs(float(first["abs_steering_command_rad"])),
        ]
        targets: dict[str, float] = {}
        for name, field in WINDOWED_METRICS.items():
            values = [float(row[field]) for row in members
                      if row.get(field) is not None
                      and math.isfinite(float(row[field]))]
            if values:
                targets[name] = float(np.mean(values))
        if targets:
            examples.append({
                "run_id": run_id,
                "cell": cell,
                "features": features,
                "targets": targets,
            })
    return examples


def _models() -> dict[str, Callable[[], Any]]:
    return {
        "ridge_quadratic": lambda: make_pipeline(
            StandardScaler(), PolynomialFeatures(2, include_bias=False),
            Ridge(alpha=3.0)),
        "ridge_cubic": lambda: make_pipeline(
            StandardScaler(), PolynomialFeatures(3, include_bias=False),
            Ridge(alpha=10.0)),
        "kernel_rbf": lambda: make_pipeline(
            StandardScaler(),
            KernelRidge(alpha=0.1, kernel="rbf", gamma=0.25)),
    }


def _rmse(values: list[float]) -> float | None:
    return math.sqrt(float(np.mean(np.square(values)))) if values else None


def _gain_ci(gains: list[float]) -> list[float] | None:
    if len(gains) < 2:
        return None
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.choice(np.asarray(gains),
                       size=(BOOTSTRAP_DRAWS, len(gains)), replace=True)
    return [float(value) for value in np.quantile(draws.mean(axis=1), (0.025, 0.975))]


def compare(training_paths: list[Path]) -> dict[str, Any]:
    rows, sources = _read_split_rows(training_paths, "train")
    examples = _make_examples(rows)
    run_ids = sorted({example["run_id"] for example in examples})
    if len(run_ids) < 3:
        raise ValueError("whole-capture model comparison requires at least 3 training captures")

    output: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "training-only model-family screening; not final validation",
        "split_used": "train only; test/final_test rows are rejected by the loader",
        "mirror_turns_pooled_within_capture": True,
        "features": [
            "speed_target_mps", "signed_throttle_delta_norm",
            "signed_throttle_rate_norm_per_sec", "abs_steering_command_rad",
        ],
        "training_pair_rows": len(rows),
        "mirror_pooled_run_cell_samples": len(examples),
        "independent_training_captures": len(run_ids),
        "sources": sources,
        "split_method": "leave-one-whole-training-capture-out",
        "baseline_definitions": {
            "exact_cell": "mean of identical speed/delta/rate/steering cells in other training captures",
            "coarse_speed_delta": "mean by speed/delta-direction/delta-size in other training captures",
            "smooth_models": "fixed quadratic/cubic ridge and RBF kernel ridge; no per-sample split tuning",
        },
        "targets": {},
    }
    model_factories = _models()
    splitter = LeaveOneGroupOut()
    for target in TARGETS:
        selected = [example for example in examples
                    if target in example["targets"]]
        if not selected:
            continue
        x = np.asarray([example["features"] for example in selected], dtype=float)
        y = np.asarray([example["targets"][target] for example in selected], dtype=float)
        groups = np.asarray([example["run_id"] for example in selected])
        per_run_errors: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list))
        per_run_counts: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int))

        for fit_idx, held_idx in splitter.split(x, y, groups):
            held_run = str(groups[held_idx[0]])
            exact: dict[tuple[float | str, ...], list[float]] = defaultdict(list)
            coarse: dict[tuple[float | str, ...], list[float]] = defaultdict(list)
            for index in fit_idx:
                cell = selected[index]["cell"]
                exact[cell].append(float(y[index]))
                coarse[cell[:3]].append(float(y[index]))

            held_cells = [selected[index]["cell"] for index in held_idx]
            exact_supported = []
            coarse_supported = []
            exact_errors = []
            coarse_errors = []
            for index, cell in zip(held_idx, held_cells):
                if cell in exact:
                    exact_errors.append(float(np.mean(exact[cell])) - float(y[index]))
                    exact_supported.append(index)
                if cell[:3] in coarse:
                    coarse_errors.append(float(np.mean(coarse[cell[:3]])) - float(y[index]))
                    coarse_supported.append(index)
            per_run_counts[held_run]["held_cells"] = len(held_idx)
            per_run_counts[held_run]["exact_supported_cells"] = len(exact_supported)
            per_run_counts[held_run]["coarse_supported_cells"] = len(coarse_supported)
            per_run_counts[held_run]["smooth_common_cells"] = len(exact_supported)
            if exact_errors:
                per_run_errors[held_run]["exact_cell"] = exact_errors
            if coarse_errors:
                per_run_errors[held_run]["coarse_speed_delta"] = coarse_errors

            # Score the smooth models on the same support as exact-cell lookup.
            if exact_supported:
                train_common = np.asarray([
                    i for i in fit_idx
                    if any(selected[i]["cell"] == selected[j]["cell"]
                           for j in exact_supported)
                ], dtype=int)
                if len(train_common):
                    held_common = np.asarray(exact_supported, dtype=int)
                    for name, factory in model_factories.items():
                        model = factory()
                        model.fit(x[train_common], y[train_common])
                        predictions = model.predict(x[held_common])
                        per_run_errors[held_run][name] = [
                            float(prediction - actual)
                            for prediction, actual in zip(predictions, y[held_common])
                        ]

        # Report cell RMSE per run, then macro-average the run-level errors.
        model_names = sorted({name for values in per_run_errors.values()
                              for name in values})
        per_model: dict[str, Any] = {}
        run_rmse: dict[str, dict[str, float]] = defaultdict(dict)
        for run_id in sorted(per_run_errors):
            for name, errors in per_run_errors[run_id].items():
                value = _rmse(errors)
                if value is not None:
                    run_rmse[name][run_id] = value
        for name in model_names:
            scores = run_rmse.get(name, {})
            per_model[name] = {
                "evaluated_capture_count": len(scores),
                "macro_capture_rmse": (
                    float(np.mean(list(scores.values()))) if scores else None),
                "per_capture_rmse": scores,
                "macro_capture_rmse_gain_ci_vs_exact": None,
            }
        exact_scores = run_rmse.get("exact_cell", {})
        for name in model_names:
            if name == "exact_cell":
                continue
            other = run_rmse.get(name, {})
            common_runs = sorted(set(exact_scores) & set(other))
            gains = [other[run] - exact_scores[run] for run in common_runs]
            per_model[name]["macro_capture_rmse_gain_ci_vs_exact"] = {
                "runs": common_runs,
                "gain_ci": _gain_ci(gains),
                "positive_favors_exact": True,
            }
        held_rows = sum(int(v["held_cells"]) for v in per_run_counts.values())
        exact_rows = sum(int(v["exact_supported_cells"])
                         for v in per_run_counts.values())
        output["targets"][target] = {
            "sample_count": len(selected),
            "capture_count": len({example["run_id"] for example in selected}),
            "exact_cell_support_fraction": exact_rows / held_rows if held_rows else 0.0,
            "per_capture_cell_counts": per_run_counts,
            "models": per_model,
        }
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-analysis", type=Path, action="append",
                        required=True, help="analysis JSON with training-split rows")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = [path if path.is_absolute() else REPO_ROOT / path
             for path in args.training_analysis]
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    if output.exists():
        parser.error(f"refusing to overwrite existing output: {output}")
    result = compare(paths)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True,
                                 allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "training_pair_rows": result["training_pair_rows"],
        "run_cell_samples": result["mirror_pooled_run_cell_samples"],
        "training_captures": result["independent_training_captures"],
        "targets": {
            name: {model: metrics["macro_capture_rmse"]
                   for model, metrics in values["models"].items()}
            for name, values in result["targets"].items()
        },
        "output": str(output),
    }, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
