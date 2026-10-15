#!/usr/bin/env python3
"""Fit a rollout-conditioned HGB transition candidate using train captures only.

The base HGB model is accurate for one-step prediction from measured state but
drifts when recursively fed its own outputs. This experiment generates
additional causal training inputs from free-running model trajectories, labels
the next state from the same training capture, and refits the transition heads.
Validation evaluates whole-run recursive transfer. Practice is not used for
fitting or model selection; test/final-test are not scored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from tools.vehicle_dynamics_learning.evaluate_recursive_longitudinal_candidate import (
    HORIZONS,
    _bundle_lags,
    _features,
    _run_rollout,
    _select_windows,
    _state,
)
from tools.vehicle_dynamics_learning.fit_longitudinal_transition_candidate import (
    DT_S,
    MAX_SPEED_MPS,
    _fit_variant,
)
from tools.vehicle_dynamics_learning.fit_longitudinal_with_fixed40_wheels import (
    BASE_VARIANT_NAMES,
    DEFAULT_DATASET,
    DEFAULT_OUTPUT as BASE_OUTPUT,
    _augment,
)
from tools.vehicle_dynamics_learning.train_nssm import _load_dataset


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE = BASE_OUTPUT / "longitudinal_raw_wheel_candidate.joblib"
DEFAULT_OUTPUT_DIR = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "longitudinal_recursive_dagger_candidate_20261009_v2")
DAGGER_WINDOWS_PER_RUN = 24
DAGGER_CAP_PER_SOURCE_RUN = 2000
DAGGER_SEED = 20261009


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _filter_rollout_windows(data: dict[str, Any],
                           windows: list[dict[str, int]], max_lag: int,
                           horizon: int) -> tuple[list[dict[str, int]], dict[str, int]]:
    raw_valid = np.asarray(data["encoder_raw_valid"], dtype=bool)
    dt = np.asarray(data["dt_s"], dtype=np.float64)
    eligible = []
    skipped_context = 0
    skipped_cadence = 0
    for window in windows:
        start = int(window["start"])
        begin = start - max_lag
        end = start + horizon
        if not raw_valid[begin:start + 1].all():
            skipped_context += 1
            continue
        if not np.isclose(dt[begin + 1:end + 1], DT_S,
                          rtol=0.0, atol=1e-7).all():
            skipped_cadence += 1
            continue
        eligible.append(window)
    return eligible, {
        "windows_skipped_invalid_raw_context": skipped_context,
        "windows_skipped_bad_cadence": skipped_cadence,
    }


def generate_rollout_examples(
        data: dict[str, Any], bundle: dict[str, Any], split: str,
        windows_per_run: int) -> tuple[np.ndarray, ...]:
    """Generate causal predicted-state features and GT next-state labels."""
    lags = _bundle_lags(bundle)
    max_lag = max(lags)
    frames = np.asarray(data["frames"], dtype=np.float64)
    state_truth = _state(data)
    raw_truth = np.asarray(data["encoder_raw_surface_mps"], dtype=np.float64)
    bounds = np.asarray(data["bounds"], dtype=np.int64)
    run_ids = np.asarray(data["run_ids"]).astype(str)
    windows = _select_windows(
        data, split=split, horizon=200,
        max_windows_per_run=windows_per_run,
        initial_context_steps=max_lag)
    # These examples model fixed-40-Hz state transitions. Require the entire
    # rollout cadence and initial measured raw-encoder context to be valid.
    windows, window_filter = _filter_rollout_windows(
        data, windows, max_lag, horizon=200)
    if not windows:
        raise ValueError(f"no cadence-valid training rollouts for split={split!r}")
    models = bundle["models"]
    aux_models = bundle["recursive_aux_models"]
    raw_models = bundle["raw_wheel_models"]
    starts = np.asarray([row["start"] for row in windows], dtype=np.int64)
    sequences = np.asarray([row["sequence"] for row in windows], dtype=np.int64)
    window_runs = np.asarray([row["run"] for row in windows], dtype=np.int32)
    window_bounds = bounds[sequences]
    if (np.any(starts - max_lag < window_bounds[:, 0])
            or np.any(starts + 200 >= window_bounds[:, 1])):
        raise ValueError("rollout window crosses its sequence boundary")
    if not np.array_equal(np.asarray(data["seq_run"])[sequences], window_runs):
        raise ValueError("rollout window run does not match its sequence")
    initial_indices = starts[:, None] + np.arange(-max_lag, 1)[None, :]
    state_history = state_truth[initial_indices].copy()
    command_history = frames[initial_indices, 7:9].copy()
    history = np.concatenate((state_history, command_history), axis=2)
    raw_history = raw_truth[initial_indices].astype(np.float32).copy()
    max_speed_by_window = np.max(
        np.hypot(state_history[:, :, 0], state_history[:, :, 1]), axis=1)
    x_rows: list[np.ndarray] = []
    main_targets: list[np.ndarray] = []
    aux_targets: list[np.ndarray] = []
    row_runs: list[np.ndarray] = []
    active = (np.hypot(state_history[:, -1, 0], state_history[:, -1, 1])
              <= MAX_SPEED_MPS)
    terminated_out_of_domain = np.zeros(len(windows), dtype=bool)

    # Batch all selected windows at each 25-ms step. This preserves the
    # recursive state/command flow while avoiding hundreds of thousands of
    # tiny estimator calls.
    for step in range(200):
        source = starts + step
        feature = _features(history, lags)
        raw_current = raw_history[:, -1, :]
        raw_parts = [raw_current]
        for lag in lags:
            raw_parts.append(raw_current - raw_history[:, -1 - lag, :])
        feature = np.column_stack((feature, *raw_parts)).astype(np.float32)
        current = state_history[:, -1, :].copy()
        gt_now = state_truth[source]
        gt_next = state_truth[source + 1]
        gt_next_speed = np.hypot(gt_next[:, 0], gt_next[:, 1])
        label_active = active & (gt_next_speed <= MAX_SPEED_MPS)

        # The step-zero teacher-forced rows are present in the original
        # training data; retain only inputs after the first recursive update.
        if step > 0:
            u_dot = (gt_next[:, 0] - gt_now[:, 0]) / DT_S
            r_mid = 0.5 * (gt_now[:, 2] + gt_next[:, 2])
            v_mid = 0.5 * (gt_now[:, 1] + gt_next[:, 1])
            main_targets.append(np.column_stack((
                gt_next[:, 0] - gt_now[:, 0],
                u_dot - r_mid * v_mid,
                frames[source + 1, 5] - frames[source, 5],
                frames[source + 1, 6] - frames[source, 6],
            ))[label_active].astype(np.float32))
            aux_targets.append((gt_next[:, 1:5] - gt_now[:, 1:5])[
                label_active].astype(np.float32))
            x_rows.append(feature[label_active])
            row_runs.append(window_runs[label_active])

        main_delta = np.column_stack([model.predict(feature) for model in models])
        aux_delta = np.column_stack([model.predict(feature) for model in aux_models])
        raw_delta = np.column_stack([model.predict(feature) for model in raw_models])
        following = current.copy()
        following[:, 1:5] += aux_delta
        following[:, 5:7] += main_delta[:, 2:4]
        following[:, 0] += main_delta[:, 0]
        raw_following = raw_history[:, -1, :].astype(np.float64) + raw_delta
        if not np.isfinite(following).all() or not np.isfinite(raw_following).all():
            raise FloatingPointError("non-finite state in training-only DAgger rollout")
        state_history = np.concatenate((state_history[:, 1:, :],
                                        following[:, None, :]), axis=1)
        next_commands = frames[source + 1, 7:9]
        command_history = np.concatenate((command_history[:, 1:, :],
                                          next_commands[:, None, :]), axis=1)
        history = np.concatenate((state_history, command_history), axis=2)
        raw_history = np.concatenate((raw_history[:, 1:, :],
                                      raw_following[:, None, :].astype(np.float32)), axis=1)
        max_speed_by_window = np.maximum(
            max_speed_by_window, np.hypot(following[:, 0], following[:, 1]))
        next_active = (active & (gt_next_speed <= MAX_SPEED_MPS)
                       & (np.hypot(following[:, 0], following[:, 1])
                          <= MAX_SPEED_MPS))
        terminated_out_of_domain |= active & ~next_active
        active = next_active

    if not x_rows or sum(len(rows) for rows in x_rows) == 0:
        raise ValueError("no in-domain DAgger examples were generated")
    rollout_speed_max = {}
    terminated_by_run = {}
    for run in sorted(np.unique(window_runs).tolist()):
        run_mask = window_runs == run
        rollout_speed_max[str(run_ids[run])] = float(max_speed_by_window[run_mask].max())
        terminated_by_run[str(run_ids[run])] = int(
            terminated_out_of_domain[run_mask].sum())
    return (
        np.vstack(x_rows).astype(np.float32),
        np.vstack(main_targets).astype(np.float32),
        np.vstack(aux_targets).astype(np.float32),
        np.concatenate(row_runs).astype(np.int32),
        {
            "windows_used": len(windows),
            "rollout_windows_terminated_outside_0_12mps": terminated_by_run,
            **window_filter,
        },
        rollout_speed_max,
    )


def _mixed_rows(original_runs: np.ndarray, dagger_runs: np.ndarray,
                run_ids: np.ndarray, cap_per_source_run: int,
                seed: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    report: dict[str, Any] = {}
    # The caller concatenates original rows before DAgger rows, so run groups
    # are selected separately and then mapped back to the combined row index.
    # Some whole training runs are shorter than the 5-s rollout horizon; they
    # remain represented by their original teacher-forced transitions.
    n_original = len(original_runs)
    for run in sorted(set(original_runs.tolist())):
        original_idx = np.flatnonzero(original_runs == run)
        dagger_idx_local = np.flatnonzero(dagger_runs == run)
        if not len(original_idx):
            raise ValueError(f"run {run_ids[run]} lacks original training rows")
        n_original_selected = min(cap_per_source_run, len(original_idx))
        selected_for_run: list[np.ndarray]
        if len(dagger_idx_local):
            n = min(n_original_selected, len(dagger_idx_local))
            original_pick = rng.choice(original_idx, size=n, replace=False)
            dagger_pick = n_original + rng.choice(
                dagger_idx_local, size=n, replace=False)
            selected_for_run = [original_pick, dagger_pick]
            selected.extend(selected_for_run)
            selected_count = 2 * n
        else:
            original_pick = rng.choice(
                original_idx, size=n_original_selected, replace=False)
            selected_for_run = [original_pick]
            selected.extend(selected_for_run)
            selected_count = n_original_selected
        # Equal total weight per source run prevents long captures or runs
        # with more generated windows from dominating the fitted heads.
        weights.extend(np.full(len(pick), 1.0 / selected_count)
                       for pick in selected_for_run)
        report[str(run_ids[run])] = {
            "original_rows_selected": int(n if len(dagger_idx_local)
                                           else n_original_selected),
            "rollout_conditioned_rows_selected": int(
                n if len(dagger_idx_local) else 0),
            "original_rows_available": int(len(original_idx)),
            "rollout_conditioned_rows_available": int(len(dagger_idx_local)),
            "has_rollout_conditioned_rows": bool(len(dagger_idx_local)),
        }
    rows = np.concatenate(selected).astype(np.int64)
    sample_weight = np.concatenate(weights).astype(np.float64)
    if len(sample_weight) != len(rows):
        raise AssertionError("balanced row weights are not aligned")
    return rows, sample_weight, report


def _validation_report(data: dict[str, Any], candidate: dict[str, Any],
                       baseline: dict[str, Any], windows_per_run: int
                       ) -> dict[str, Any]:
    lags = _bundle_lags(candidate)
    windows = _select_windows(
        data, split="validation", horizon=200,
        max_windows_per_run=windows_per_run,
        initial_context_steps=max(lags))
    windows, window_filter = _filter_rollout_windows(
        data, windows, max(lags), horizon=200)
    if not windows:
        raise ValueError("no cadence-valid whole-run validation rollouts")
    run_names = np.asarray(data["run_ids"]).astype(str)
    summary: dict[str, Any] = {
        "window_count": len(windows),
        "window_filter": window_filter,
        "independent_run_count": len({row["run"] for row in windows}),
        "windows_by_run": {
            str(run_names[run]): sum(row["run"] == run for row in windows)
            for run in sorted({row["run"] for row in windows})},
        "metrics": {},
    }
    for horizon in HORIZONS:
        label = f"{horizon * DT_S:.3f}s"
        base = _run_rollout(
            data, baseline, windows, horizon, lags,
            u_update="increment", raw_input_mode="predicted")
        dagger = _run_rollout(
            data, candidate, windows, horizon, lags,
            u_update="increment", raw_input_mode="predicted")
        base_macro = base["macro_run_metrics"][label]["candidate_macro_run_rmse"]
        dagger_macro = dagger["macro_run_metrics"][label]["candidate_macro_run_rmse"]
        per_run = {}
        for run_id, baseline_run in base["per_run"].items():
            base_values = baseline_run[label]["candidate"]
            candidate_values = dagger["per_run"][run_id][label]["candidate"]
            per_run[run_id] = {
                key: {
                    "baseline": base_values.get(key),
                    "dagger": candidate_values.get(key),
                    "dagger_minus_baseline": (
                        candidate_values[key] - base_values[key]
                        if key in candidate_values and key in base_values else None),
                }
                for key in ("position_rmse_m", "u_rmse_mps", "v_rmse_mps",
                            "yaw_rate_rmse_rps", "raw_encoder_pair_rmse_mps")}
        paired = {}
        for metric in ("position_rmse_m", "u_rmse_mps", "v_rmse_mps",
                       "yaw_rate_rmse_rps", "raw_encoder_pair_rmse_mps"):
            differences = np.asarray([
                per_run[run_id][metric]["dagger_minus_baseline"]
                for run_id in per_run
                if per_run[run_id][metric]["dagger_minus_baseline"] is not None
            ], dtype=np.float64)
            if not len(differences):
                continue
            rng = np.random.default_rng(20261009 + horizon)
            bootstrap = rng.choice(
                differences, size=(10000, len(differences)), replace=True).mean(axis=1)
            paired[metric] = {
                "runs": int(len(differences)),
                "mean_candidate_minus_baseline": float(differences.mean()),
                "paired_run_bootstrap_95pct": [
                    float(value) for value in np.quantile(bootstrap, (0.025, 0.975))],
            }
        summary["metrics"][label] = {
            "baseline_macro_run_rmse": base_macro,
            "dagger_macro_run_rmse": dagger_macro,
            "paired_run_delta": paired,
            "per_run": per_run,
        }
    return summary


def fit(dataset_path: Path, bundle_path: Path, output_dir: Path,
        windows_per_run: int, cap_per_source_run: int, max_iter: int,
        threads: int) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    if not dataset_path.is_file() or not bundle_path.is_file():
        raise FileNotFoundError("training dataset or frozen HGB bundle is missing")
    started = time.time()
    data = _load_dataset(dataset_path)
    baseline = joblib.load(bundle_path)
    if not baseline.get("research_only") or not baseline.get("raw_wheel_models"):
        raise ValueError("baseline must be the frozen raw-wheel HGB research candidate")
    dataset_digest = sha256(dataset_path)
    if baseline.get("training_dataset_sha256") != dataset_digest:
        raise ValueError("frozen baseline was not trained from this exact dataset")
    if not baseline.get("uses_predicted_raw_encoder_features"):
        raise ValueError("baseline does not use predicted raw-encoder inputs")

    train = _augment(data, "train")
    src_train, run_train, _, _, model_target, _, raw_train, feature_sets, _ = train
    selected_name = str(baseline["selected_parent_feature_variant"])
    if selected_name not in BASE_VARIANT_NAMES:
        raise ValueError("baseline uses an unknown parent feature variant")
    selected_index = BASE_VARIANT_NAMES.index(selected_name)
    original_x = feature_sets[selected_index]
    if len(original_x) != len(src_train) or len(raw_train) != len(src_train):
        raise ValueError("original train features lost source-row alignment")

    rigid = np.asarray(data["simulator_rigid_state"], dtype=np.float64)
    frames = np.asarray(data["frames"], dtype=np.float64)
    state_filtered = np.column_stack((rigid[:, (7, 8, 12)], frames[:, 3:7]))
    aux_original = (state_filtered[src_train + 1][:, 1:5]
                    - state_filtered[src_train][:, 1:5]).astype(np.float32)
    if not np.isfinite(aux_original).all():
        raise ValueError("non-finite original auxiliary transition label")

    (dagger_x, dagger_main, dagger_aux, dagger_runs,
     window_counts, rollout_speed_max) = generate_rollout_examples(
         data, baseline, "train", windows_per_run)
    main_x = np.vstack((original_x, dagger_x)).astype(np.float32)
    main_y = np.vstack((model_target, dagger_main)).astype(np.float32)
    aux_y = np.vstack((aux_original, dagger_aux)).astype(np.float32)
    combined_runs = np.concatenate((run_train, dagger_runs))
    if main_x.shape[1] != getattr(baseline["models"][0], "n_features_in_", -1):
        raise ValueError("generated DAgger feature width differs from the base model")
    rows, sample_weight, balanced_report = _mixed_rows(
        run_train, dagger_runs, np.asarray(data["run_ids"]).astype(str),
        cap_per_source_run, DAGGER_SEED)
    if not np.array_equal(np.sort(np.unique(combined_runs[rows])),
                          np.sort(np.unique(run_train))):
        raise ValueError("balanced DAgger sampling omitted a train run")

    with threadpool_limits(limits=threads):
        main_models = _fit_variant(
            main_x, main_y, rows, sample_weight, max_iter)
        aux_models = _fit_variant(
            main_x, aux_y, rows, sample_weight, max_iter)
    candidate = dict(baseline)
    candidate["models"] = main_models
    candidate["recursive_aux_models"] = aux_models
    candidate["dagger_training"] = {
        "method": "one-iteration rollout-conditioned physical transition fit",
        "base_bundle_sha256": sha256(bundle_path),
        "training_dataset_sha256": sha256(dataset_path),
        "training_split_only": True,
        "test_and_final_test_used": False,
        "training_run_ids": sorted(set(
            np.asarray(data["run_ids"]).astype(str)[run_train].tolist())),
        "rollout_windows_requested_per_run": windows_per_run,
        **window_counts,
        "generated_rows": int(len(dagger_x)),
        "rows_per_source_per_run_cap": cap_per_source_run,
        "balanced_sample_counts_by_run": balanced_report,
        "max_predicted_speed_mps_during_training_rollouts": rollout_speed_max,
        "max_iter": max_iter,
        "random_seed": DAGGER_SEED,
        "rollout_u_update": "increment",
        "generated_transition_labels": (
            "recorded next-minus-current truth increment; not correction-to-future-truth"),
        "rollout_training_domain_speed_mps": [0.0, MAX_SPEED_MPS],
        "rollout_is_terminated_outside_speed_domain_without_clamping": True,
        "future_truth_or_sensor_feedback_used_in_generated_inputs": False,
        "ground_truth_used_only_as_next_state_training_labels": True,
    }

    validation = _validation_report(data, candidate, baseline, windows_per_run)
    output_dir.mkdir(parents=True, exist_ok=False)
    bundle_output = output_dir / "dagger_longitudinal_candidate.joblib"
    joblib.dump(candidate, bundle_output)
    report = {
        "schema_version": 1,
        "purpose": "test whether rollout-conditioned train-only examples reduce HGB recursive drift",
        "training_dataset": str(dataset_path.relative_to(ROOT)),
        "training_dataset_sha256": sha256(dataset_path),
        "base_candidate": str(bundle_path.relative_to(ROOT)),
        "base_candidate_sha256": sha256(bundle_path),
        "candidate_bundle": str(bundle_output.relative_to(ROOT)),
        "candidate_bundle_sha256": sha256(bundle_output),
        "fitting_split": "train",
        "selection_split": "validation",
        "test_and_final_test_used": False,
        "practice_split_used_for_fitting_or_selection": False,
        "dagger_training": candidate["dagger_training"],
        "validation": validation,
        "no_simulator_or_runtime_changes": True,
        "elapsed_seconds": time.time() - started,
        "limitations": [
            "Rollout-conditioned labels train next-state correction on recorded training trajectories; they are not new physical measurements.",
            "Only one DAgger generation is fitted; validation must show improvement before practice transfer is inspected.",
            "The candidate remains a research-only transition model, not a production observer or full-lap simulator.",
        ],
    }
    (output_dir / "evaluation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--windows-per-run", type=int, default=DAGGER_WINDOWS_PER_RUN)
    parser.add_argument("--cap-per-source-run", type=int,
                        default=DAGGER_CAP_PER_SOURCE_RUN)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    report = fit(args.dataset.resolve(), args.bundle.resolve(),
                 args.output_dir.resolve(), args.windows_per_run,
                 args.cap_per_source_run, args.max_iter, args.threads)
    print(json.dumps({
        "candidate": report["candidate_bundle"],
        "generated_rows": report["dagger_training"]["generated_rows"],
        "validation_runs": report["validation"]["independent_run_count"],
        "validation_windows": report["validation"]["window_count"],
        "validation_5s": report["validation"]["metrics"]["5.000s"],
        "output": str(args.output_dir.resolve()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
