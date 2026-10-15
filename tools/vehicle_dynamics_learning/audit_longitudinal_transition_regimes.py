#!/usr/bin/env python3
"""Stratify the frozen longitudinal/wheel teacher by measured driving regime.

No fitting occurs here. The broad fixed-40-Hz HGB transition and its already
trained high-steer correction are scored on whole-run validation and unseen
practice captures. Test/final-test splits are never selected or scored.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from tools.vehicle_dynamics_learning.fit_longitudinal_with_fixed40_wheels import (
    DEFAULT_DATASET as HGB_TRAIN_DATASET,
    DEFAULT_PRACTICE as PRACTICE_DATASET,
)
from tools.vehicle_dynamics_learning.fit_fixed40_wheel_response_candidate import (
    VARIANT_NAMES as RAW_WHEEL_VARIANTS,
    _prepare as _prepare_raw_wheels,
)
from tools.vehicle_dynamics_learning.fit_longitudinal_transition_candidate import (
    DT_S,
    REAR_TRACK_M,
    _rmse,
)
from tools.vehicle_dynamics_learning.fit_longitudinal_with_fixed40_wheels import (
    BASE_VARIANT_NAMES,
    _augment,
)
from tools.vehicle_dynamics_learning.train_nssm import _load_dataset


ROOT = Path(__file__).resolve().parents[2]
HGB_BUNDLE = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "longitudinal_with_fixed40_wheels_candidate_20261009/"
    "longitudinal_raw_wheel_candidate.joblib")
HIGHSTEER_DATASET = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "full_modeling_reset_20261001/"
    "highsteer_75_heldout_validation_plus_r03_20261004/"
    "openplane_dynamics_fixed40hz.npz")
HIGHSTEER_RESIDUAL_BUNDLE = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "highsteer_recursive_longitudinal_residual_gain0p5_training_20261009/"
    "u_filtered_wheel_residual_models.joblib")
OUTPUT = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "longitudinal_transition_regime_audit_20261009_v4.json")
SPEED_EDGES = (0.0, 3.0, 6.0, 9.0, 12.0)
STEERING_EDGES = (0.0, 0.10, 0.20, 0.30, 0.40, 0.525)
THROTTLE_EDGES = (0.0, 0.25, 0.50, 0.75, 1.001)
MISMATCH_EDGES = (0.0, 0.25, 0.50, 1.0, 2.0, 4.0, np.inf)
SIGNED_MISMATCH_EDGES = (
    -np.inf, -4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0, np.inf)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _macro_metrics(prediction: np.ndarray, target: np.ndarray,
                   run_index: np.ndarray, run_ids: np.ndarray,
                   baseline: np.ndarray | None = None
                   ) -> dict[str, Any]:
    per_run: dict[str, Any] = {}
    for run in sorted(np.unique(run_index).tolist()):
        mask = run_index == run
        error = prediction[mask] - target[mask]
        row = {
            "transitions": int(mask.sum()),
            "rmse": _rmse(error),
            "bias": float(np.mean(error)),
            "p95_abs_error": float(np.quantile(np.abs(error), 0.95)),
        }
        if baseline is not None:
            row["persistence_rmse"] = _rmse(baseline[mask] - target[mask])
        per_run[str(run_ids[run])] = row
    return {
        "runs": len(per_run),
        "transitions": int(len(target)),
        "macro_run_rmse": float(np.mean([row["rmse"]
                                          for row in per_run.values()])),
        "pooled_rmse": _rmse(prediction - target),
        "macro_run_bias": float(np.mean([row["bias"]
                                          for row in per_run.values()])),
        "per_run": per_run,
    }


def _conditional_table(values: np.ndarray, edges: tuple[float, ...],
                       labels: tuple[str, ...],
                       prediction: np.ndarray, target: np.ndarray,
                       run_index: np.ndarray, run_ids: np.ndarray,
                       persistence: np.ndarray,
                       ) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for index, (lower, upper, label) in enumerate(zip(
            edges[:-1], edges[1:], labels)):
        upper_mask = values <= upper if index == len(labels) - 1 else values < upper
        mask = (values >= lower) & upper_mask
        if not np.any(mask):
            continue
        rows: dict[str, Any] = {"transitions": int(mask.sum())}
        rows["parent"] = _macro_metrics(
            prediction[mask], target[mask], run_index[mask], run_ids,
            persistence[mask])
        result[label] = rows
    return result


def _evaluate(data: dict[str, Any], split: str, hgb: dict[str, Any],
              residual_bundle: dict[str, Any]) -> dict[str, Any]:
    (source, run_index, target, baseline, _, state, _, feature_sets,
     _) = _augment(data, split)
    variant = BASE_VARIANT_NAMES.index(hgb["selected_parent_feature_variant"])
    x = feature_sets[variant]
    delta = np.column_stack([model.predict(x) for model in hgb["models"]])
    parent_prediction = baseline + delta

    rigid = np.asarray(data["simulator_rigid_state"], dtype=np.float64)
    frames = np.asarray(data["frames"], dtype=np.float64)
    speed = np.hypot(rigid[source, 7], rigid[source, 8])
    absolute_steering = np.abs(frames[source, 3])
    speed_bounds = tuple(residual_bundle["gate"]["speed_mps"])
    steering_bounds = tuple(residual_bundle["gate"]["absolute_steering_rad"])
    gate = (
        (speed >= speed_bounds[0]) & (speed <= speed_bounds[1])
        & (absolute_steering >= steering_bounds[0])
        & (absolute_steering <= steering_bounds[1]))
    raw_bundle = hgb
    raw_source, raw_runs, raw_base, raw_target, raw_features, _, _ = (
        _prepare_raw_wheels(data, split))
    if not np.array_equal(run_index[np.searchsorted(source, raw_source)], raw_runs):
        raise AssertionError("filtered and raw wheel source rows lost alignment")
    raw_variant = RAW_WHEEL_VARIANTS.index(
        raw_bundle["raw_wheel_feature_variant"])
    raw_x = raw_features[raw_variant]
    raw_delta = np.column_stack([
        model.predict(raw_x) for model in raw_bundle["raw_wheel_models"]])
    raw_prediction = raw_base + raw_delta

    run_ids = np.asarray(data["run_ids"]).astype(str)
    prediction_metrics = {
        "u_next_mps": {
            "parent": _macro_metrics(parent_prediction[:, 0], target[:, 0],
                                     run_index, run_ids, baseline[:, 0]),
        },
        "effective_ax_mps2": {
            "parent": _macro_metrics(parent_prediction[:, 1], target[:, 1],
                                     run_index, run_ids, baseline[:, 1]),
        },
        "filtered_rear_wheel_pair_mps": {
            "parent": _macro_metrics(parent_prediction[:, 2:4], target[:, 2:4],
                                     run_index, run_ids, baseline[:, 2:4]),
        },
        "raw_fixed40_encoder_pair_mps": {
            "parent": _macro_metrics(raw_prediction, raw_target, raw_runs,
                                     run_ids, raw_base),
        },
    }

    throttle = np.clip(frames[source, 4], 0.0, 1.0)
    throttle_slew = np.abs(frames[source, 4] - frames[source - 1, 4]) / DT_S
    half_track = REAR_TRACK_M / 2.0
    contact_left = rigid[source, 7] - half_track * rigid[source, 12]
    contact_right = rigid[source, 7] + half_track * rigid[source, 12]
    wheel_mismatch = np.abs(0.5 * (
        frames[source, 5] - contact_left
        + frames[source, 6] - contact_right))
    signed_wheel_mismatch = 0.5 * (
        frames[source, 5] - contact_left
        + frames[source, 6] - contact_right)
    speed_edges = SPEED_EDGES
    speed_labels = tuple(f"{a:g}-{b:g} m/s" for a, b in zip(
        speed_edges[:-1], speed_edges[1:]))
    steer_edges = STEERING_EDGES
    steer_labels = tuple(f"{a:.2f}-{b:.3f} rad" for a, b in zip(
        steer_edges[:-1], steer_edges[1:]))
    throttle_edges = THROTTLE_EDGES
    throttle_labels = tuple(f"{a:.2f}-{min(b, 1.0):.2f}" for a, b in zip(
        throttle_edges[:-1], throttle_edges[1:]))
    mismatch_labels = tuple(
        f"{a:g}-{b:g} m/s" if np.isfinite(b) else f">={a:g} m/s"
        for a, b in zip(MISMATCH_EDGES[:-1], MISMATCH_EDGES[1:]))
    signed_mismatch_labels = tuple(
        f"{a:g} to {b:g} m/s" if np.isfinite(a) and np.isfinite(b)
        else f"<{b:g} m/s" if not np.isfinite(a)
        else f">={a:g} m/s"
        for a, b in zip(SIGNED_MISMATCH_EDGES[:-1],
                        SIGNED_MISMATCH_EDGES[1:]))
    conditions = {
        "ground_truth_body_speed": (speed, speed_edges, speed_labels),
        "absolute_steering_feedback": (
            absolute_steering, steer_edges, steer_labels),
        "throttle_feedback": (throttle, throttle_edges, throttle_labels),
        "filtered_wheel_to_contact_speed_common_mismatch_proxy": (
            wheel_mismatch, MISMATCH_EDGES, mismatch_labels),
        "signed_filtered_wheel_to_contact_speed_common_mismatch_proxy": (
            signed_wheel_mismatch, SIGNED_MISMATCH_EDGES,
            signed_mismatch_labels),
    }
    groups: dict[str, Any] = {}
    for condition_name, (values, edges, labels) in conditions.items():
        groups[condition_name] = {
            "u_next_mps": _conditional_table(
                values, edges, labels, parent_prediction[:, 0], target[:, 0],
                run_index, run_ids, baseline[:, 0]),
            "effective_ax_mps2": _conditional_table(
                values, edges, labels, parent_prediction[:, 1], target[:, 1],
                run_index, run_ids, baseline[:, 1]),
            "filtered_rear_wheel_pair_mps": _conditional_table(
                values, edges, labels, parent_prediction[:, 2:4],
                target[:, 2:4], run_index, run_ids, baseline[:, 2:4]),
        }
    directional_load: dict[str, Any] = {}
    for direction, mismatch_mask in (
            ("wheel_deficit_below_minus_1mps", signed_wheel_mismatch < -1.0),
            ("wheel_excess_above_plus_1mps", signed_wheel_mismatch > 1.0)):
        directional_load[direction] = {}
        for throttle_name, throttle_mask in (
                ("feedback_below_0p25", throttle < 0.25),
                ("feedback_at_least_0p25", throttle >= 0.25)):
            selected = mismatch_mask & throttle_mask
            if not np.any(selected):
                continue
            directional_load[direction][throttle_name] = {
                "transitions": int(selected.sum()),
                "runs": int(len(np.unique(run_index[selected]))),
                "u_next_mps": _macro_metrics(
                    parent_prediction[selected, 0], target[selected, 0],
                    run_index[selected], run_ids, baseline[selected, 0]),
                "effective_ax_mps2": _macro_metrics(
                    parent_prediction[selected, 1], target[selected, 1],
                    run_index[selected], run_ids, baseline[selected, 1]),
                "filtered_rear_wheel_pair_mps": _macro_metrics(
                    parent_prediction[selected, 2:4], target[selected, 2:4],
                    run_index[selected], run_ids, baseline[selected, 2:4]),
            }
    nonzero_slew = throttle_slew > 0.0
    groups["throttle_feedback_slew"] = {
        "zero_slew_fraction": float(np.mean(~nonzero_slew)),
        "zero_slew_transitions": int(np.count_nonzero(~nonzero_slew)),
        "nonzero_slew_transitions": int(np.count_nonzero(nonzero_slew)),
        "nonzero_slew_quantiles_per_s": {
            f"p{percentile}": float(np.quantile(
                throttle_slew[nonzero_slew], percentile / 100.0))
            for percentile in (50, 90, 99)},
    }
    return {
        "split": split,
        "transitions": int(len(source)),
        "run_ids": sorted(set(run_ids[run_index].tolist())),
        "selected_feature_variant": hgb["selected_parent_feature_variant"],
        "highsteer_recursive_correction_gate_fraction": float(np.mean(gate)),
        "metrics": prediction_metrics,
        "transition_error_by_condition": groups,
        "signed_mismatch_by_throttle_feedback": directional_load,
        "target_conventions": {
            "effective_ax": "truth (u[t+1]-u[t])/dt - r_mid*v_mid, equivalently du/dt - r*v",
            "filtered_wheels": "existing trailing-filter odometry wheel channels",
            "raw_wheels": "fixed-25-ms encoder-derived rates; not ground-truth tire RPM or slip",
            "mismatch": "filtered encoder speed minus kinematic rear contact-speed proxy",
            "signed_mismatch": "mean signed filtered encoder speed minus kinematic rear contact speed; positive is wheel-speed excess, negative is wheel-speed deficit",
        },
    }


def run(output_path: Path = OUTPUT) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    hgb = joblib.load(HGB_BUNDLE)
    residual = joblib.load(HIGHSTEER_RESIDUAL_BUNDLE)
    results: dict[str, Any] = {}
    datasets = (
        ("broad_openplane_validation", HGB_TRAIN_DATASET, "validation"),
        ("highsteer_whole_run_validation", HIGHSTEER_DATASET, "validation"),
        ("unseen_practice_diagnostic", PRACTICE_DATASET, "unseen_practice"),
    )
    hashes = {str(path.relative_to(ROOT)): _sha256(path)
              for path in (HGB_BUNDLE, HIGHSTEER_RESIDUAL_BUNDLE,
                           HGB_TRAIN_DATASET, HIGHSTEER_DATASET,
                           PRACTICE_DATASET)}
    for label, path, split in datasets:
        data = _load_dataset(path)
        if split not in set(np.asarray(data["splits"]).astype(str)):
            raise ValueError(f"dataset {path} lacks split {split!r}")
        results[label] = _evaluate(data, split, hgb, residual)
    report = {
        "schema_version": 1,
        "purpose": "locate held-out longitudinal and wheel transition errors over speed, steering, throttle, and encoder-derived wheel/body mismatch",
        "frozen_inputs_sha256": hashes,
        "results": results,
        "test_and_final_test_used": False,
        "fitting_or_model_selection_performed": False,
        "recursive_residual_not_scored_here": (
            "the residual was trained on recursive-rollout features, not the "
            "teacher-forced features used by this audit"),
        "recursive_residual_not_scored_here": (
            "the residual was trained on recursive-rollout features, not the "
            "teacher-forced features used by this audit"),
        "yaw_atlas_note": "This is a teacher-forced transition audit. The frozen yaw atlas is used in the separate causal recursive evaluator; here simulator truth defines the transition targets so current longitudinal errors are isolated.",
        "no_simulator_or_runtime_changes": True,
        "limitations": [
            "descriptive bins are summaries, not inferred physical regime boundaries",
            "the broad validation set selected the parent feature variant; high-steer whole-run validation is a separate cohort",
            "two practice runs are diagnostics only",
            "wheel/body mismatch is an encoder-derived proxy, not measured tire slip or tire force",
            "transition rows are temporally correlated; uncertainty is summarized by whole-run errors",
        ],
    }
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return report


def main() -> None:
    report = run()
    summary: dict[str, Any] = {}
    for label, cohort in report["results"].items():
        summary[label] = {
            "runs": len(cohort["run_ids"]),
            "transitions": cohort["transitions"],
            "recursive_residual_gate_fraction": cohort[
                "highsteer_recursive_correction_gate_fraction"],
            "u_rmse_parent": cohort["metrics"]["u_next_mps"]["parent"][
                "macro_run_rmse"],
            "ax_rmse": cohort["metrics"]["effective_ax_mps2"]["parent"][
                "macro_run_rmse"],
            "filtered_wheel_rmse_parent": cohort["metrics"][
                "filtered_rear_wheel_pair_mps"]["parent"]["macro_run_rmse"],
            "raw_encoder_rmse": cohort["metrics"][
                "raw_fixed40_encoder_pair_mps"]["parent"]["macro_run_rmse"],
        }
    print(json.dumps({"output": str(OUTPUT), "summary": summary},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
