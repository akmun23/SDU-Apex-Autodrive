#!/usr/bin/env python3
"""Fit and evaluate a causal one-step longitudinal/wheel transition candidate.

Only whole-run ``train`` captures fit the estimators. Whole-run ``validation``
captures select between two prespecified feature sets; a separate
``unseen_practice`` dataset is scored only after selection. The output is a
research artifact, not an odometry/MPC integration. Targets are next-packet
simulator COM u, effective body-frame a_x derived from simulator-truth
transitions, and the stored filtered rear-wheel surface speeds. The distinct
fixed-25-ms encoder target is deliberately not
treated as the same wheel state.

Inputs are current/past state, actuator feedback, wheel state, and commands.
No target-frame state, acceleration, wheel measurement, or future feedback is
used as an input. Scores are one-step teacher-forced-context scores; recursive
plant accuracy is not claimed by this script.
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
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from tools.vehicle_dynamics_learning.train_nssm import _load_dataset


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "full_modeling_reset_20261001/"
    "replacement_offline_sim_raceline_20261002/"
    "encoder_raw_state_teacher_v1/openplane_dynamics_raw_wheels.npz")
DEFAULT_PRACTICE = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "full_modeling_reset_20261001/"
    "replacement_offline_sim_raceline_20261003/"
    "full_throttle_domain_v1/encoder_fixed40hz_v1/"
    "practice_dynamics_raw_wheels_fixed40hz.npz")
DEFAULT_OUTPUT = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "longitudinal_transition_candidate_20261009")
DT_S = 0.025
MAX_SPEED_MPS = 12.0
REAR_TRACK_M = 0.236
LAGS = (1, 2, 4, 8, 16, 32, 64)  # Causal history through 1.6 s.
VARIANT_NAMES = ("current_state", "history_to_1600ms",
                 "history_to_1600ms_plus_mismatch")
TARGET_NAMES = ("u_com_increment_mps", "effective_ax_fd_mps2",
                "rear_left_filtered_increment_mps",
                "rear_right_filtered_increment_mps")
AUX_STATE_NAMES = ("v_com_increment_mps", "yaw_rate_increment_rps",
                   "steering_feedback_increment_rad",
                   "throttle_feedback_increment_norm")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _state(data: dict[str, Any]) -> np.ndarray:
    rigid = np.asarray(data["simulator_rigid_state"], dtype=np.float64)
    if rigid.ndim != 2 or rigid.shape[1] != 13:
        raise ValueError("simulator rigid-state labels have unexpected shape")
    # These are simulator-truth COM body velocities and yaw rate, used as
    # current-state inputs for the one-step score and as offline labels.
    return rigid[:, (7, 8, 12)]


def _collect(data: dict[str, Any], selected_split: str
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                        np.ndarray, tuple[np.ndarray, ...], np.ndarray,
                        list[str]]:
    """Build source rows without crossing sequence/reset boundaries."""
    state = _state(data)
    frames = np.asarray(data["frames"], dtype=np.float64)
    ax = np.asarray(data["simulator_linear_acceleration"], dtype=np.float64)
    splits = np.asarray(data["splits"]).astype(str)
    run_ids = np.asarray(data["run_ids"]).astype(str)
    source_parts: list[np.ndarray] = []
    run_parts: list[np.ndarray] = []
    for (start_raw, end_raw), run_raw in zip(data["bounds"], data["seq_run"]):
        start, end, run = int(start_raw), int(end_raw), int(run_raw)
        if splits[run] != selected_split or end - start <= max(LAGS) + 1:
            continue
        # Source k predicts target k+1; eight prior samples are required by
        # the prespecified history feature set.
        source = np.arange(start + max(LAGS), end - 1, dtype=np.int64)
        if not len(source):
            continue
        valid = (np.isclose(data["dt_s"][source + 1], DT_S,
                            rtol=0.0, atol=1e-7)
                 & (np.hypot(state[source, 0], state[source, 1]) <= MAX_SPEED_MPS)
                 & (np.hypot(state[source + 1, 0], state[source + 1, 1])
                    <= MAX_SPEED_MPS)
                 & np.isfinite(state[source]).all(axis=1)
                 & np.isfinite(state[source + 1]).all(axis=1)
                 & np.isfinite(ax[source + 1, 0]))
        source = source[valid]
        source_parts.append(source)
        run_parts.append(np.full(len(source), run, dtype=np.int32))
    if not source_parts:
        raise ValueError(f"no usable contiguous rows for split={selected_split!r}")
    source = np.concatenate(source_parts)
    run_index = np.concatenate(run_parts)

    # State/actuator/wheel/command channels at the transition source.
    current = np.column_stack((
        state[source], frames[source, 3:9]))
    # Frames 3:9 are steering feedback, throttle feedback, two filtered wheel
    # speeds, and steering/throttle commands, respectively.
    base_names = ("u_com", "v_com", "yaw_rate", "steering_feedback",
                  "throttle_feedback", "wheel_left_filtered",
                  "wheel_right_filtered", "steering_command",
                  "throttle_command")
    base = current.astype(np.float32)

    # Recent changes are computed only from source and earlier packets.
    history_parts: list[np.ndarray] = []
    history_names: list[str] = []
    history_channels = (
        ("u_com", state[:, 0]), ("v_com", state[:, 1]),
        ("yaw_rate", state[:, 2]),
        ("steering_feedback", frames[:, 3]),
        ("throttle_feedback", frames[:, 4]),
        ("wheel_left_filtered", frames[:, 5]),
        ("wheel_right_filtered", frames[:, 6]),
        ("steering_command", frames[:, 7]),
        ("throttle_command", frames[:, 8]),
    )
    for lag in LAGS:
        for name, values in history_channels:
            history_parts.append((values[source] - values[source - lag])[:, None])
            history_names.append(f"delta_{name}_over_{lag}_steps")
    history = np.concatenate(history_parts, axis=1).astype(np.float32)

    # Explicit kinematic rear-contact mismatch and actuator errors. These are
    # deterministic transforms of current/past features, not extra labels.
    half_track = REAR_TRACK_M / 2.0
    contact_left = state[source, 0] - half_track * state[source, 2]
    contact_right = state[source, 0] + half_track * state[source, 2]
    wheels = frames[source, 5:7]
    mismatch = np.column_stack((
        0.5 * (wheels[:, 0] + wheels[:, 1]) - state[source, 0],
        wheels[:, 0] - wheels[:, 1],
        wheels[:, 0] - contact_left,
        wheels[:, 1] - contact_right,
        frames[source, 8] - frames[source, 4],
        frames[source, 7] - frames[source, 3],
    )).astype(np.float32)
    mismatch_names = ("rear_mean_minus_com_u", "rear_wheel_differential",
                      "left_wheel_contact_speed_residual",
                      "right_wheel_contact_speed_residual",
                      "throttle_command_minus_feedback",
                      "steering_command_minus_feedback")

    # Labels at k+1 only. The model predicts increments for the velocity and
    # wheel states, preserving the source state exactly when it predicts zero
    # change. Effective a_x is derived from simulator-truth COM motion with
    # midpoint body-frame transport; the separate packet acceleration signal
    # is retained only for target-semantics diagnostics below.
    dt = np.asarray(data["dt_s"], dtype=np.float64)[source + 1]
    u_dot = (state[source + 1, 0] - state[source, 0]) / dt
    r_mid = 0.5 * (state[source, 2] + state[source + 1, 2])
    v_mid = 0.5 * (state[source, 1] + state[source + 1, 1])
    ax_effective = u_dot - r_mid * v_mid
    target = np.column_stack((
        state[source + 1, 0], ax_effective,
        frames[source + 1, 5], frames[source + 1, 6])).astype(np.float32)
    # Persistence/constant-speed baselines use only source-state information.
    baseline = np.column_stack((
        state[source, 0], np.zeros(len(source)),
        frames[source, 5], frames[source, 6]
    )).astype(np.float32)
    model_target = (target - baseline).astype(np.float32)
    x_state = base
    x_history = np.column_stack((base, history)).astype(np.float32)
    x_physics = np.column_stack((base, history, mismatch)).astype(np.float32)
    feature_names = [
        list(base_names),
        list(base_names) + history_names,
        list(base_names) + history_names + list(mismatch_names),
    ]
    features = (x_state, x_history, x_physics)
    if not all(np.isfinite(x).all() for x in features):
        raise ValueError(f"non-finite causal feature in {selected_split}")
    return (source, run_index, target, baseline, model_target, features,
            state, feature_names)


def _balanced_training_rows(run_index: np.ndarray, run_ids: np.ndarray,
                            cap_per_run: int, seed: int
                            ) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    counts: dict[str, int] = {}
    for run in sorted(np.unique(run_index).tolist()):
        rows = np.flatnonzero(run_index == run)
        if len(rows) > cap_per_run:
            rows = np.sort(rng.choice(rows, size=cap_per_run, replace=False))
        selected.append(rows)
        # Each whole training capture has equal total weight, so the long
        # throttle sweep cannot dominate just by contributing more packets.
        weights.append(np.full(
            len(rows), len(run_index) / (len(np.unique(run_index)) * len(rows)),
            dtype=np.float64))
        counts[str(run_ids[run])] = len(rows)
    return np.concatenate(selected), np.concatenate(weights), counts


def _fit_variant(x: np.ndarray, y: np.ndarray, rows: np.ndarray,
                 sample_weight: np.ndarray, max_iter: int
                 ) -> list[HistGradientBoostingRegressor]:
    models = []
    for target_col in range(y.shape[1]):
        model = HistGradientBoostingRegressor(
            loss="squared_error", learning_rate=0.06, max_iter=max_iter,
            max_leaf_nodes=19, min_samples_leaf=40, l2_regularization=2.0,
            early_stopping=False, random_state=101)
        model.fit(x[rows], y[rows, target_col], sample_weight=sample_weight)
        models.append(model)
    return models


def _predict(models: list[HistGradientBoostingRegressor], x: np.ndarray
             ) -> np.ndarray:
    return np.column_stack([model.predict(x) for model in models]).astype(np.float64)


def _rmse(error: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(error))))


def _metrics_by_run(prediction: np.ndarray, target: np.ndarray,
                    baseline: np.ndarray, run_index: np.ndarray,
                    run_ids: np.ndarray) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    for run in sorted(np.unique(run_index).tolist()):
        mask = run_index == run
        err = prediction[mask] - target[mask]
        base_err = baseline[mask] - target[mask]
        reports[str(run_ids[run])] = {
            "transitions": int(mask.sum()),
            "u_rmse_mps": _rmse(err[:, 0]),
            "u_absolute_error_p95_mps": float(np.quantile(np.abs(err[:, 0]), 0.95)),
            "ax_rmse_mps2": _rmse(err[:, 1]),
            "ax_absolute_error_p95_mps2": float(np.quantile(np.abs(err[:, 1]), 0.95)),
            "rear_wheel_pair_rmse_mps": _rmse(err[:, 2:4]),
            "rear_wheel_pair_absolute_error_p95_mps": float(
                np.quantile(np.abs(err[:, 2:4]), 0.95)),
            "baseline_u_rmse_mps": _rmse(base_err[:, 0]),
            "baseline_ax_rmse_mps2": _rmse(base_err[:, 1]),
            "baseline_rear_wheel_pair_rmse_mps": _rmse(base_err[:, 2:4]),
            "u_bias_mps": float(np.mean(err[:, 0])),
            "ax_bias_mps2": float(np.mean(err[:, 1])),
            "rear_wheel_pair_bias_mps": float(np.mean(err[:, 2:4])),
        }
    names = ("u_rmse_mps", "ax_rmse_mps2", "rear_wheel_pair_rmse_mps",
             "baseline_u_rmse_mps", "baseline_ax_rmse_mps2",
             "baseline_rear_wheel_pair_rmse_mps")
    macro = {name: float(np.mean([row[name] for row in reports.values()]))
             for name in names}
    macro["runs_better_than_persistence"] = {
        "u": int(sum(row["u_rmse_mps"] < row["baseline_u_rmse_mps"]
                     for row in reports.values())),
        "ax": int(sum(row["ax_rmse_mps2"] < row["baseline_ax_rmse_mps2"]
                      for row in reports.values())),
        "rear_wheel_pair": int(sum(
            row["rear_wheel_pair_rmse_mps"]
            < row["baseline_rear_wheel_pair_rmse_mps"]
            for row in reports.values())),
    }
    macro["paired_macro_relative_score"] = float(np.mean((
        np.asarray([macro["u_rmse_mps"] / macro["baseline_u_rmse_mps"]]),
        np.asarray([macro["ax_rmse_mps2"] / macro["baseline_ax_rmse_mps2"]]),
        np.asarray([macro["rear_wheel_pair_rmse_mps"]
                    / macro["baseline_rear_wheel_pair_rmse_mps"]]),
    )))
    return {"macro_run_metrics": macro, "per_run": reports}


def _slice_metrics(candidate: np.ndarray, target: np.ndarray,
                   persistence: np.ndarray) -> dict[str, Any]:
    if len(target) == 0:
        return {"transitions": 0}
    error = candidate - target
    base_error = persistence - target
    return {
        "transitions": int(len(target)),
        "candidate": {
            "u_rmse_mps": _rmse(error[:, 0]),
            "u_abs_error_p95_mps": float(np.quantile(np.abs(error[:, 0]), 0.95)),
            "effective_ax_rmse_mps2": _rmse(error[:, 1]),
            "effective_ax_abs_error_p95_mps2": float(
                np.quantile(np.abs(error[:, 1]), 0.95)),
            "rear_wheel_pair_rmse_mps": _rmse(error[:, 2:4]),
            "rear_wheel_pair_abs_error_p95_mps": float(
                np.quantile(np.abs(error[:, 2:4]), 0.95)),
            "rear_wheel_mean_bias_mps": float(np.mean(error[:, 2:4])),
        },
        "persistence_or_zero_ax_baseline": {
            "u_rmse_mps": _rmse(base_error[:, 0]),
            "effective_ax_rmse_mps2": _rmse(base_error[:, 1]),
            "rear_wheel_pair_rmse_mps": _rmse(base_error[:, 2:4]),
        },
    }


def _stratified_metrics(data: dict[str, Any], source: np.ndarray,
                        target: np.ndarray, baseline: np.ndarray,
                        prediction: np.ndarray, state: np.ndarray
                        ) -> dict[str, Any]:
    frames = np.asarray(data["frames"], dtype=np.float64)
    speed = np.hypot(state[source, 0], state[source, 1])
    steering = frames[source, 3]
    throttle = frames[source, 8]
    throttle_slew = np.abs(frames[source, 8] - frames[source - 1, 8]) / DT_S
    wheel_mismatch = np.abs(0.5 * (frames[source, 5] + frames[source, 6])
                            - state[source, 0])

    def fixed_bins(values: np.ndarray, edges: tuple[float, ...],
                   labels: list[str]) -> dict[str, Any]:
        index = np.digitize(values, edges[1:-1], right=False)
        result: dict[str, Any] = {}
        for bucket, label in enumerate(labels):
            mask = index == bucket
            if np.any(mask):
                result[label] = _slice_metrics(
                    prediction[mask], target[mask], baseline[mask])
        return result

    def quantile_bins(values: np.ndarray, fractions: tuple[float, ...]
                      ) -> dict[str, Any]:
        edges = np.unique(np.quantile(values, fractions))
        if len(edges) < 2:
            return {"all_equal": _slice_metrics(prediction, target, baseline)}
        index = np.digitize(values, edges[1:-1], right=False)
        result: dict[str, Any] = {}
        for bucket in range(len(edges) - 1):
            mask = index == bucket
            if np.any(mask):
                result[f"quantile_bin_{bucket+1}"] = {
                    "value_range": [float(edges[bucket]), float(edges[bucket + 1])],
                    **_slice_metrics(prediction[mask], target[mask], baseline[mask]),
                }
        return result

    return {
        "interpretation": "descriptive slices only; boundaries are reporting partitions, not asserted physical regimes",
        "speed_magnitude_mps": fixed_bins(
            speed, (0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.000001),
            ["0-2", "2-4", "4-6", "6-8", "8-10", "10-12"]),
        "absolute_steering_rad": fixed_bins(
            np.abs(steering), (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, np.inf),
            ["0-.1", ".1-.2", ".2-.3", ".3-.4", ".4-.5", ">=.5"]),
        "signed_steering_side": {
            "left": _slice_metrics(prediction[steering < -0.025],
                                    target[steering < -0.025],
                                    baseline[steering < -0.025]),
            "center": _slice_metrics(prediction[np.abs(steering) <= 0.025],
                                      target[np.abs(steering) <= 0.025],
                                      baseline[np.abs(steering) <= 0.025]),
            "right": _slice_metrics(prediction[steering > 0.025],
                                     target[steering > 0.025],
                                     baseline[steering > 0.025]),
        },
        "throttle_command": fixed_bins(
            throttle, (0.0, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, np.inf),
            ["0-.05", ".05-.10", ".10-.20", ".20-.40",
             ".40-.60", ".60-.80", ">=.80"]),
        "absolute_throttle_command_slew_per_s": quantile_bins(
            throttle_slew, (0.0, 0.50, 0.90, 0.99, 1.0)),
        "absolute_rear_wheel_body_mismatch_mps": quantile_bins(
            wheel_mismatch, (0.0, 0.25, 0.50, 0.75, 0.90, 1.0)),
    }


def _u_absolute_vs_increment_ablation(
        absolute_prediction: np.ndarray, increment_prediction: np.ndarray,
        target: np.ndarray, baseline: np.ndarray, run_index: np.ndarray,
        run_ids: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for label, prediction in (("absolute_state_head", absolute_prediction),
                              ("increment_head", increment_prediction),
                              ("persistence", baseline[:, 0])):
        per_run = {}
        for run in sorted(np.unique(run_index).tolist()):
            mask = run_index == run
            pred = prediction[mask] if prediction.ndim == 1 else prediction[mask, 0]
            per_run[str(run_ids[run])] = _rmse(pred - target[mask, 0])
        result[label] = {
            "macro_run_rmse_mps": float(np.mean(list(per_run.values()))),
            "per_run_rmse_mps": per_run,
        }
    result["increment_vs_absolute_run_bootstrap"] = _run_bootstrap_ci(
        {key: {"u": value} for key, value in result["increment_head"][
            "per_run_rmse_mps"].items()},
        {key: {"u": value} for key, value in result["absolute_state_head"][
            "per_run_rmse_mps"].items()}, "u")
    return result


def _motion_consistency(prediction: np.ndarray, target: np.ndarray,
                        source: np.ndarray, state: np.ndarray,
                        dt_s: np.ndarray
                        ) -> dict[str, float]:
    midpoint_r = 0.5 * (state[source, 2] + state[source + 1, 2])
    midpoint_v = 0.5 * (state[source, 1] + state[source + 1, 1])
    ax_from_predicted_u = ((prediction[:, 0] - state[source, 0])
                           / dt_s[source + 1] - midpoint_r * midpoint_v)
    difference = prediction[:, 1] - ax_from_predicted_u
    return {
        "predicted_ax_vs_ax_implied_by_predicted_u_rmse_mps2": _rmse(difference),
        "predicted_ax_vs_ax_implied_by_predicted_u_bias_mps2": float(
            np.mean(difference)),
        "ax_implied_by_predicted_u_rmse_against_gt_effective_ax_mps2": _rmse(
            ax_from_predicted_u - target[:, 1]),
    }


def _target_semantics(data: dict[str, Any], source: np.ndarray,
                      run_index: np.ndarray, run_ids: np.ndarray
                      ) -> dict[str, Any]:
    state = _state(data)
    ax = np.asarray(data["simulator_linear_acceleration"], dtype=np.float64)[:, 0]
    following = source + 1
    dt = np.asarray(data["dt_s"], dtype=np.float64)[following]
    u_derivative = (state[following, 0] - state[source, 0]) / dt
    r_mid = 0.5 * (state[source, 2] + state[following, 2])
    v_mid = 0.5 * (state[source, 1] + state[following, 1])
    ax_midpoint_fd = u_derivative - r_mid * v_mid
    ax_source_fd = u_derivative - state[source, 2] * state[source, 1]
    ax_packet_mean = 0.5 * (ax[source] + ax[following])
    output: dict[str, Any] = {}
    for run in sorted(np.unique(run_index).tolist()):
        mask = run_index == run
        direct = ax_packet_mean[mask]
        midpoint = ax_midpoint_fd[mask]
        source_conv = ax_source_fd[mask]
        output[str(run_ids[run])] = {
            "transitions": int(mask.sum()),
            "packet_mean_vs_midpoint_fd_rmse_mps2": _rmse(direct-midpoint),
            "packet_mean_vs_source_fd_rmse_mps2": _rmse(direct-source_conv),
            "packet_mean_vs_midpoint_fd_bias_mps2": float(np.mean(direct-midpoint)),
            "packet_mean_midpoint_fd_correlation": float(
                np.corrcoef(direct, midpoint)[0, 1]) if np.std(direct) and np.std(midpoint) else None,
        }
    return output


def _target_semantics_summary(data: dict[str, Any], source: np.ndarray,
                             state: np.ndarray) -> dict[str, Any]:
    direct = np.asarray(data["simulator_linear_acceleration"],
                        dtype=np.float64)[:, 0]
    following = source + 1
    dt = np.asarray(data["dt_s"], dtype=np.float64)[following]
    du_dt = (state[following, 0] - state[source, 0]) / dt
    midpoint_r = 0.5 * (state[source, 2] + state[following, 2])
    midpoint_v = 0.5 * (state[source, 1] + state[following, 1])
    effective = du_dt - midpoint_r * midpoint_v
    packet_mean = 0.5 * (direct[source] + direct[following])
    mismatch = packet_mean - effective
    return {
        "packet_mean_direct_ax_quantiles_mps2": [float(v) for v in np.quantile(
            packet_mean, (0.0, 0.01, 0.50, 0.95, 0.99, 1.0))],
        "finite_difference_effective_ax_quantiles_mps2": [float(v) for v in np.quantile(
            effective, (0.0, 0.01, 0.50, 0.95, 0.99, 1.0))],
        "direct_minus_effective_ax_rmse_mps2": _rmse(mismatch),
        "direct_minus_effective_ax_bias_mps2": float(np.mean(mismatch)),
        "absolute_disagreement_over_5mps2_count": int(np.sum(np.abs(mismatch) > 5.0)),
        "absolute_disagreement_over_5mps2_fraction": float(np.mean(np.abs(mismatch) > 5.0)),
        "absolute_disagreement_over_10mps2_count": int(np.sum(np.abs(mismatch) > 10.0)),
        "absolute_disagreement_over_10mps2_fraction": float(np.mean(np.abs(mismatch) > 10.0)),
    }


def _run_bootstrap_ci(candidate: dict[str, Any], baseline: dict[str, Any],
                      metric: str, samples: int = 10000, seed: int = 108
                      ) -> dict[str, float]:
    common = sorted(set(candidate) & set(baseline))
    differences = np.asarray([candidate[key][metric] - baseline[key][metric]
                              for key in common], dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.choice(differences, size=(samples, len(differences)), replace=True)
    means = draws.mean(axis=1)
    return {"runs": len(common), "mean_candidate_minus_baseline": float(differences.mean()),
            "run_cluster_bootstrap_95pct": [float(v) for v in np.quantile(means, (0.025, 0.975))]}


def _score_bundle(models: list[HistGradientBoostingRegressor],
                  feature: np.ndarray, target: np.ndarray,
                  baseline: np.ndarray, run_index: np.ndarray,
                  run_ids: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
    # The learned outputs are causal increments; reconstruct state/acceleration
    # predictions from the source-only persistence baseline.
    prediction = baseline.astype(np.float64) + _predict(models, feature)
    scored = _metrics_by_run(prediction, target, baseline, run_index, run_ids)
    scored["run_cluster_bootstrap_candidate_minus_persistence"] = {}
    persistence = _metrics_by_run(baseline, target, baseline, run_index, run_ids)
    for metric in ("u_rmse_mps", "ax_rmse_mps2", "rear_wheel_pair_rmse_mps"):
        scored["run_cluster_bootstrap_candidate_minus_persistence"][metric] = (
            _run_bootstrap_ci(scored["per_run"], persistence["per_run"], metric))
    return scored, prediction


def _select_score(metrics: dict[str, Any]) -> float:
    row = metrics["macro_run_metrics"]
    return float(np.mean((
        row["u_rmse_mps"] / row["baseline_u_rmse_mps"],
        row["ax_rmse_mps2"] / row["baseline_ax_rmse_mps2"],
        row["rear_wheel_pair_rmse_mps"]
        / row["baseline_rear_wheel_pair_rmse_mps"],
    )))


def run(dataset_path: Path, practice_path: Path, output_dir: Path,
        cap_per_run: int, max_iter: int, threads: int) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite analysis output: {output_dir}")
    if not dataset_path.is_file() or not practice_path.is_file():
        raise FileNotFoundError("required open-plane or practice dataset is missing")
    start_time = time.time()
    data = _load_dataset(dataset_path)
    if data["schema_version"] < 6:
        raise ValueError("longitudinal candidate requires simulator rigid-state labels")
    source_train, runs_train, y_train, _, dy_train, x_train, _, feature_names = _collect(
        data, "train")
    source_val, runs_val, y_val, base_val, _, x_val, state_val, _ = _collect(
        data, "validation")
    run_ids = np.asarray(data["run_ids"]).astype(str)
    training_rows, sample_weight, train_counts = _balanced_training_rows(
        runs_train, run_ids, cap_per_run, seed=101)
    if set(np.unique(runs_train).tolist()) != set(
            np.unique(runs_train[training_rows]).tolist()):
        raise AssertionError("training row sampler omitted an observed train run")
    train_run_names = sorted(set(run_ids[runs_train].tolist()))
    if not train_run_names or any(name in {"test", "final_test"} for name in train_run_names):
        raise AssertionError("sealed test/final-test data entered training")

    models_by_variant: dict[str, list[HistGradientBoostingRegressor]] = {}
    validation_reports: dict[str, Any] = {}
    validation_predictions: dict[str, np.ndarray] = {}
    with threadpool_limits(limits=threads):
        for variant_index, name in enumerate(VARIANT_NAMES):
            models = _fit_variant(x_train[variant_index], dy_train, training_rows,
                                  sample_weight, max_iter)
            scored, prediction = _score_bundle(
                models, x_val[variant_index], y_val, base_val, runs_val, run_ids)
            models_by_variant[name] = models
            validation_reports[name] = scored
            validation_predictions[name] = prediction
    selected_variant = min(validation_reports, key=lambda name:
                           _select_score(validation_reports[name]))
    selected_index = VARIANT_NAMES.index(selected_variant)
    with threadpool_limits(limits=threads):
        absolute_u_model = HistGradientBoostingRegressor(
            loss="squared_error", learning_rate=0.06, max_iter=max_iter,
            max_leaf_nodes=19, min_samples_leaf=40, l2_regularization=2.0,
            early_stopping=False, random_state=101)
        absolute_u_model.fit(x_train[selected_index][training_rows],
                             y_train[training_rows, 0],
                             sample_weight=sample_weight)
    absolute_u_validation = absolute_u_model.predict(x_val[selected_index])
    selected_validation_prediction = validation_predictions[selected_variant]
    u_parameterization_ablation = _u_absolute_vs_increment_ablation(
        absolute_u_validation, selected_validation_prediction[:, 0],
        y_val, base_val, runs_val, run_ids)

    # These four one-step heads do not change the longitudinal candidate
    # selection. They provide predicted lateral/yaw and actuator state so a
    # later recursive evaluation can avoid feeding future feedback/truth.
    frames_train = np.asarray(data["frames"], dtype=np.float32)
    rigid_train = np.asarray(data["simulator_rigid_state"], dtype=np.float32)
    train_state7 = np.column_stack((rigid_train[:, (7, 8, 12)],
                                    frames_train[:, 3:7]))
    aux_target = (train_state7[source_train + 1][:, (1, 2, 3, 4)]
                  - train_state7[source_train][:, (1, 2, 3, 4)])
    with threadpool_limits(limits=threads):
        aux_models = _fit_variant(x_train[selected_index], aux_target,
                                  training_rows, sample_weight, max_iter)

    practice = _load_dataset(practice_path)
    if practice["schema_version"] < 6:
        raise ValueError("practice transfer requires simulator rigid-state labels")
    practice_split = "unseen_practice" if "unseen_practice" in set(
        np.asarray(practice["splits"]).astype(str)) else "validation"
    p_source, p_runs, p_target, p_baseline, _, p_features, p_state, _ = _collect(
        practice, practice_split)
    practice_report, practice_prediction = _score_bundle(
        models_by_variant[selected_variant], p_features[selected_index],
        p_target, p_baseline, p_runs, np.asarray(practice["run_ids"]).astype(str))
    practice_report["split_used"] = practice_split
    practice_report["stratified_diagnostics"] = _stratified_metrics(
        practice, p_source, p_target, p_baseline, practice_prediction, p_state)
    practice_report["motion_consistency"] = _motion_consistency(
        practice_prediction, p_target, p_source, p_state, practice["dt_s"])

    val_semantics = _target_semantics(data, source_val, runs_val, run_ids)
    train_semantics = _target_semantics(data, source_train[training_rows],
                                        runs_train[training_rows], run_ids)
    output_dir.mkdir(parents=True, exist_ok=False)
    model_path = output_dir / "selected_longitudinal_candidate.joblib"
    bundle = {
        "research_only": True,
        "selected_variant": selected_variant,
        "models": models_by_variant[selected_variant],
        "recursive_aux_models": aux_models,
        "recursive_aux_target_names": AUX_STATE_NAMES,
        "recursive_aux_purpose": "predict next v, yaw-rate, steering-feedback, and throttle-feedback increments so free rollouts need no future measurements",
        "feature_names": feature_names[selected_index],
        "target_names": TARGET_NAMES,
        "training_run_ids": train_run_names,
        "history_lags_steps": list(LAGS),
        "training_dataset_sha256": _sha256(dataset_path),
        "feature_policy": "current source state and commands; optional strictly prior deltas and derived current wheel/body/actuator mismatch",
        "target_policy": "u and filtered rear-wheel state increments plus effective a_x derived from simulator-truth COM velocity differences and midpoint body-frame transport",
        "fixed_dt_s": DT_S,
        "recursive_accuracy_established": False,
        "production_integration_allowed": False,
    }
    joblib.dump(bundle, model_path, compress=3)

    report = {
        "schema_version": 1,
        "purpose": "causal one-step longitudinal and rear-wheel response candidate",
        "created_utc_date": "2026-10-09",
        "openplane_dataset": str(dataset_path.relative_to(ROOT)),
        "openplane_dataset_sha256": _sha256(dataset_path),
        "practice_dataset": str(practice_path.relative_to(ROOT)),
        "practice_dataset_sha256": _sha256(practice_path),
        "test_and_final_test_policy": "neither split was selected, fitted, scored, nor used for candidate selection; only train and validation whole runs were processed",
        "fit_split": "train only, equal total weight per run, uniform deterministic cap per run",
        "training_run_ids": train_run_names,
        "training_transitions_by_run": {str(run_ids[run]): int(np.sum(runs_train[training_rows] == run))
                                        for run in sorted(np.unique(runs_train[training_rows]))},
        "training_sampling_cap_per_run": cap_per_run,
        "training_sample_count": int(len(training_rows)),
        "validation_split": "seven whole-run captures; no row-level split",
        "validation_run_ids": sorted(set(run_ids[runs_val].tolist())),
        "history_lags_steps": list(LAGS),
        "independent_validation_transition_count": int(len(source_val)),
        "practice_transfer_run_ids": sorted(set(
            np.asarray(practice["run_ids"]).astype(str)[p_runs].tolist())),
        "practice_transfer_transition_count": int(len(p_source)),
        "inputs": {
            "available_at_source_only": True,
            "truth_used_as_offline_current_state_and_training_label": True,
            "future_truth_or_feedback_used_as_model_input": False,
            "rear_wheel_target": "stored filtered ~100ms state; fixed-25ms encoder rate remains a distinct measurement target",
            "variants": {
                "current_state": feature_names[0],
                VARIANT_NAMES[1]: feature_names[1],
                VARIANT_NAMES[2]: feature_names[2],
            },
        },
        "model": {
            "class": "sklearn HistGradientBoostingRegressor, one estimator per target",
        "targets": list(TARGET_NAMES),
        "output_parameterization": "predict increments for u and rear-wheel states; predict effective a_x; reconstruct states from source values",
            "max_iter": max_iter,
            "learning_rate": 0.06,
            "max_leaf_nodes": 19,
            "min_samples_leaf": 40,
            "l2_regularization": 2.0,
            "random_state": 101,
            "thread_limit": threads,
        },
        "recursive_auxiliary_heads": {
            "target_names": list(AUX_STATE_NAMES),
            "fit_split": "train only",
            "does_not_affect_longitudinal_feature_selection": True,
            "reason": "predict the remaining minimum state needed to test causal recursive rollouts without future GT/feedback",
        },
        "validation_candidates": validation_reports,
        "selected_validation_stratified_diagnostics": _stratified_metrics(
            data, source_val, y_val, base_val,
            selected_validation_prediction, state_val),
        "selected_validation_motion_consistency": _motion_consistency(
            selected_validation_prediction, y_val, source_val, state_val,
            data["dt_s"]),
        "u_absolute_vs_increment_parameterization_ablation": u_parameterization_ablation,
        "candidate_selection": {
            "selected_variant": selected_variant,
            "rule": "lowest mean of three macro-run RMSE ratios to persistence: u, finite-difference effective ax, filtered rear-wheel pair",
            "selected_score": _select_score(validation_reports[selected_variant]),
            "practice_did_not_select_or_tune_candidate": True,
        },
        "practice_transfer_diagnostic": practice_report,
        "acceleration_target_semantics": {
            "description": "packet-mean direct simulator a_x compared with the finite-difference effective acceleration used as the model label; discrepancies are training-label diagnostics, not model inputs",
            "validation_per_run": val_semantics,
            "validation_pooled": _target_semantics_summary(
                data, source_val, state_val),
            "practice_pooled": _target_semantics_summary(
                practice, p_source, p_state),
            "training_sampled_per_run": train_semantics,
        },
        "artifact": str(model_path.relative_to(ROOT)),
        "artifact_sha256": _sha256(model_path),
        "elapsed_seconds": time.time() - start_time,
        "limitations": [
            "one-step teacher-forced-context only; no recursive/full-lap accuracy claim",
            "simulator truth supplies current body-state inputs and labels offline; this is not yet a sensor-only observer",
            "filtered wheel-state prediction does not predict instantaneous fixed-25ms encoder increments",
            "practice transfer contains only the archived independent practice captures",
            "no odometry/MPC/runtime integration is performed",
        ],
    }
    report_path = output_dir / "evaluation.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--practice-dataset", type=Path, default=DEFAULT_PRACTICE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cap-per-run", type=int, default=8000)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.cap_per_run < 100 or args.max_iter < 1 or args.threads < 1:
        parser.error("cap-per-run, max-iter, and threads must be positive")
    report = run(args.dataset.resolve(), args.practice_dataset.resolve(),
                 args.output_dir.resolve(), args.cap_per_run,
                 args.max_iter, args.threads)
    print(json.dumps({
        "output": str(args.output_dir),
        "selected_variant": report["candidate_selection"]["selected_variant"],
        "validation": {name: values["macro_run_metrics"]
                       for name, values in report["validation_candidates"].items()},
        "practice_transfer": report["practice_transfer_diagnostic"]["macro_run_metrics"],
        "elapsed_seconds": report["elapsed_seconds"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
