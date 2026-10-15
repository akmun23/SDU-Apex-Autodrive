#!/usr/bin/env python3
"""Free-run the fitted longitudinal candidate without future sensor truth.

Recorded commands are the prescribed replay input. The model predicts body
velocity, yaw rate, actuator feedback, and filtered rear-wheel state at every
step. Ground truth initializes the model's required causal lag context and
scores future outputs. This is a research-only recursive state/pose replay,
not a production observer or a validated full-lap simulator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from tools.vehicle_dynamics_learning.fit_longitudinal_transition_candidate import (
    DEFAULT_DATASET,
    DEFAULT_PRACTICE,
    DT_S,
    LAGS,
    MAX_SPEED_MPS,
    REAR_TRACK_M,
    _rmse,
)
from tools.vehicle_dynamics_learning.train_nssm import _load_dataset


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "longitudinal_recursive_candidate_20261009_longhistory_v5/"
    "selected_longitudinal_candidate.joblib")
DEFAULT_OUTPUT = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "longitudinal_recursive_evaluation_20261009_v1.json")
HORIZONS = (1, 4, 10, 20, 30, 40, 50, 60, 70, 80, 120, 200)
STATE_NAMES = ("u_com_mps", "v_com_mps", "yaw_rate_rps",
               "steering_feedback_rad", "throttle_feedback_norm",
               "rear_left_filtered_mps", "rear_right_filtered_mps")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _state(data: dict[str, Any]) -> np.ndarray:
    rigid = np.asarray(data["simulator_rigid_state"], dtype=np.float64)
    frames = np.asarray(data["frames"], dtype=np.float64)
    return np.column_stack((rigid[:, (7, 8, 12)], frames[:, 3:7]))


def _bundle_lags(bundle: dict[str, Any]) -> tuple[int, ...]:
    explicit = bundle.get("history_lags_steps")
    if explicit is not None:
        lags = tuple(int(value) for value in explicit)
    else:
        # The earlier short-history bundle predates explicit lag metadata.
        names = bundle.get("feature_names", ())
        lags = tuple(sorted({
            int(name.removeprefix("delta_u_com_over_").removesuffix("_steps"))
            for name in names
            if name.startswith("delta_u_com_over_") and name.endswith("_steps")
        }))
    expected_features = (9 + 9 * len(lags) + 6
                         + (2 + 2 * len(lags) if bundle.get(
                             "uses_predicted_raw_encoder_features") else 0))
    contact_slip_features = expected_features + 2 + 2 * len(lags)
    body_features = (contact_slip_features
                     if bundle.get("uses_contact_slip_state_features")
                     else expected_features)
    body_models = (list(bundle.get("models", ()))
                   + list(bundle.get("recursive_aux_models", ())))
    lateral_acceleration_model = bundle.get(
        "recursive_lateral_acceleration_model")
    if lateral_acceleration_model is not None:
        body_models.append(lateral_acceleration_model)
    raw_models = list(bundle.get("raw_wheel_models", ()))
    contact_slip_models = list(bundle.get("contact_slip_models", ()))
    if (not lags or tuple(sorted(set(lags))) != lags or lags[0] < 1
            or any(getattr(model, "n_features_in_", expected_features)
                   != body_features for model in body_models)
            or any(getattr(model, "n_features_in_", expected_features)
                   != expected_features for model in raw_models)
            or any(getattr(model, "n_features_in_", contact_slip_features)
                   != contact_slip_features for model in contact_slip_models)):
        raise ValueError("candidate bundle has invalid or inconsistent history features")
    return lags


def _select_windows(data: dict[str, Any], split: str, horizon: int,
                    max_windows_per_run: int,
                    initial_context_steps: int) -> list[dict[str, int]]:
    bounds = np.asarray(data["bounds"], dtype=np.int64)
    seq_run = np.asarray(data["seq_run"], dtype=np.int32)
    splits = np.asarray(data["splits"]).astype(str)
    state = _state(data)
    run_windows: dict[int, list[dict[str, int]]] = {}
    history = initial_context_steps
    for sequence, ((start_raw, end_raw), run_raw) in enumerate(zip(bounds, seq_run)):
        start, end, run = int(start_raw), int(end_raw), int(run_raw)
        if splits[run] != split or end - start < history + horizon + 1:
            continue
        first = start + history
        stop = end - horizon
        candidates = np.arange(first, stop, dtype=np.int64)
        candidates = candidates[
            np.hypot(state[candidates, 0], state[candidates, 1]) <= MAX_SPEED_MPS]
        if not len(candidates):
            continue
        run_windows.setdefault(run, []).extend(
            {"run": run, "sequence": sequence, "start": int(value)}
            for value in candidates)
    selected: list[dict[str, int]] = []
    for run, candidates in sorted(run_windows.items()):
        candidates.sort(key=lambda item: (item["sequence"], item["start"]))
        if len(candidates) > max_windows_per_run:
            indices = np.linspace(0, len(candidates) - 1,
                                  max_windows_per_run, dtype=np.int64)
            candidates = [candidates[int(i)] for i in np.unique(indices)]
        selected.extend(candidates)
    return selected


def _features(history: np.ndarray, lags: tuple[int, ...]) -> np.ndarray:
    current = history[:, -1, :]
    parts = [current]
    for lag in lags:
        parts.append(current - history[:, -1 - lag, :])
    half_track = REAR_TRACK_M / 2.0
    extra = np.column_stack((
        0.5 * (current[:, 5] + current[:, 6]) - current[:, 0],
        current[:, 5] - current[:, 6],
        current[:, 5] - (current[:, 0] - half_track * current[:, 2]),
        current[:, 6] - (current[:, 0] + half_track * current[:, 2]),
        current[:, 8] - current[:, 4],
        current[:, 7] - current[:, 3],
    ))
    parts.append(extra)
    return np.column_stack(parts).astype(np.float32)


def _append_contact_slip_features(feature: np.ndarray, history: np.ndarray,
                                  raw_history: np.ndarray,
                                  lags: tuple[int, ...],
                                  half_track_m: float) -> np.ndarray:
    contact_history = np.stack((
        history[:, :, 0] - half_track_m * history[:, :, 2],
        history[:, :, 0] + half_track_m * history[:, :, 2]), axis=2)
    slip_history = raw_history.astype(np.float64) - contact_history
    slip_current = slip_history[:, -1, :]
    parts = [slip_current]
    for lag in lags:
        parts.append(slip_current - slip_history[:, -1 - lag, :])
    return np.column_stack((feature, *parts)).astype(np.float32)


def _pose_step(pose: np.ndarray, current: np.ndarray,
               following: np.ndarray) -> np.ndarray:
    result = pose.copy()
    u_mid = 0.5 * (current[:, 0] + following[:, 0])
    v_mid = 0.5 * (current[:, 1] + following[:, 1])
    r_mid = 0.5 * (current[:, 2] + following[:, 2])
    heading_mid = pose[:, 2] + 0.5 * DT_S * r_mid
    result[:, 0] += DT_S * (u_mid * np.cos(heading_mid)
                            - v_mid * np.sin(heading_mid))
    result[:, 1] += DT_S * (u_mid * np.sin(heading_mid)
                            + v_mid * np.cos(heading_mid))
    heading = pose[:, 2] + DT_S * r_mid
    result[:, 2] = np.arctan2(np.sin(heading), np.cos(heading))
    return result


def _wrap_angle(value: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(value), np.cos(value))


def _run_rollout(data: dict[str, Any], bundle: dict[str, Any],
                 windows: list[dict[str, int]], horizon: int,
                 lags: tuple[int, ...], u_update: str,
                 raw_input_mode: str,
                 teacher_force_state: str | None = None,
                 ) -> dict[str, Any]:
    frames = np.asarray(data["frames"], dtype=np.float64)
    state = _state(data)
    pose_truth = np.asarray(data["simulator_pose_xyyaw"], dtype=np.float64)
    starts = np.asarray([window["start"] for window in windows], dtype=np.int64)
    run_index = np.asarray([window["run"] for window in windows], dtype=np.int32)
    run_ids = np.asarray(data["run_ids"]).astype(str)
    count = len(starts)
    history = np.empty((count, max(lags) + 1, 9), dtype=np.float32)
    for row, start in enumerate(starts):
        indices = np.arange(start - max(lags), start + 1)
        history[row, :, :7] = state[indices].astype(np.float32)
        history[row, :, 7:9] = frames[indices, 7:9].astype(np.float32)

    raw_models = bundle.get("raw_wheel_models")
    contact_slip_models = bundle.get("contact_slip_models")
    raw_history = None
    raw_prediction = None
    raw_baseline = None
    raw_truth = data.get("encoder_raw_surface_mps")
    if raw_models is not None:
        raw_lags = tuple(int(value) for value in
                         bundle.get("raw_wheel_history_lags_steps", ()))
        if (not raw_lags or raw_lags != lags
                or bundle.get("raw_wheel_feature_variant")
                != "history_mismatch_plus_raw"
                or bundle.get("selected_parent_feature_variant")
                != "history_to_1600ms_plus_mismatch"):
            raise ValueError("raw-wheel feature layout is incompatible with this rollout")
        if raw_truth is None or len(raw_models) != 2:
            raise ValueError("raw-wheel candidate requires aligned encoder data and two heads")
        raw_truth = np.asarray(raw_truth, dtype=np.float64)
        raw_history = np.empty((count, max(raw_lags) + 1, 2), dtype=np.float32)
        for row, start in enumerate(starts):
            raw_history[row] = raw_truth[
                start - max(raw_lags):start + 1].astype(np.float32)
        if raw_input_mode in ("predicted", "predicted_contact_slip"):
            if (raw_input_mode == "predicted"
                    and bundle.get("uses_contact_slip_state_features")):
                raise ValueError(
                    "slip-aware body heads require contact-relative wheel recursion")
            if raw_input_mode == "predicted_contact_slip" and (
                    contact_slip_models is None or len(contact_slip_models) != 2):
                raise ValueError("contact-relative mode requires two slip-response heads")
            raw_prediction = np.empty((count, horizon + 1, 2), dtype=np.float64)
            raw_prediction[:, 0] = raw_history[:, -1]
            raw_baseline = raw_history[:, -1].astype(np.float64)
        elif raw_input_mode != "oracle":
            raise ValueError(f"unknown raw input mode {raw_input_mode!r}")

    predicted_state = np.empty((count, horizon + 1, 7), dtype=np.float64)
    predicted_pose = np.empty((count, horizon + 1, 3), dtype=np.float64)
    baseline_state = np.empty_like(predicted_state)
    baseline_pose = np.empty_like(predicted_pose)
    predicted_state[:, 0] = history[:, -1, :7]
    baseline_state[:, 0] = predicted_state[:, 0]
    predicted_pose[:, 0] = pose_truth[starts]
    baseline_pose[:, 0] = pose_truth[starts]
    ax_prediction = np.empty((count, horizon), dtype=np.float64)
    ax_truth = np.empty((count, horizon), dtype=np.float64)
    lateral_acceleration_model = bundle.get(
        "recursive_lateral_acceleration_model")
    if lateral_acceleration_model is not None:
        ay_prediction = np.empty((count, horizon), dtype=np.float64)
        ay_truth = np.empty((count, horizon), dtype=np.float64)
    models = bundle["models"]
    aux_models = bundle.get("recursive_aux_models")
    if aux_models is None or len(aux_models) != 4:
        raise ValueError("bundle lacks the auxiliary recursive state heads")
    if len(models) != 4:
        raise ValueError("bundle has unexpected longitudinal model outputs")
    teacher_force_channels = {
        "u": (0,), "v": (1,), "yaw_rate": (2,),
        "steering_feedback": (3,), "throttle_feedback": (4,),
        "left_filtered_wheel": (5,), "right_filtered_wheel": (6,),
        "filtered_wheels": (5, 6),
    }
    if (teacher_force_state is not None
            and teacher_force_state not in teacher_force_channels):
        raise ValueError(f"unknown teacher-forced state {teacher_force_state!r}")

    for step in range(horizon):
        source = starts + step
        current = history[:, -1, :7].astype(np.float64)
        x = _features(history, lags)
        if raw_history is not None:
            if raw_input_mode == "oracle":
                raw_current = raw_truth[source].astype(np.float32)
                raw_parts = [raw_current]
                for lag in lags:
                    raw_parts.append(raw_current
                                     - raw_truth[source - lag].astype(np.float32))
            else:
                raw_current = raw_history[:, -1, :]
                raw_parts = [raw_current]
                for lag in lags:
                    raw_parts.append(raw_current - raw_history[:, -1 - lag, :])
            raw_extra = np.column_stack(raw_parts).astype(np.float32)
            x = np.column_stack((x, raw_extra)).astype(np.float32)
        if bundle.get("uses_contact_slip_state_features"):
            half_track = float(bundle.get(
                "contact_slip_rear_track_m", REAR_TRACK_M)) / 2.0
            x = _append_contact_slip_features(
                x, history, raw_history, lags, half_track)
        main_delta = np.column_stack([model.predict(x) for model in models])
        aux_delta = np.column_stack([model.predict(x) for model in aux_models])
        ay_step = (lateral_acceleration_model.predict(x)
                   if lateral_acceleration_model is not None else None)
        following = current.copy()
        if ay_step is None:
            following[:, 1] += aux_delta[:, 0]
        following[:, 2] += aux_delta[:, 1]
        following[:, 3] += aux_delta[:, 2]
        following[:, 4] += aux_delta[:, 3]
        following[:, 5] += main_delta[:, 2]
        following[:, 6] += main_delta[:, 3]
        if ay_step is not None and u_update == "effective_ax_midpoint":
            # Implicit midpoint integration of the rigid-body frame equations:
            # u_dot = a_x + r*v, v_dot = a_y - r*u. This changes only the
            # numerical integration ordering, not the vehicle dynamics.
            r_mid = 0.5 * (current[:, 2] + following[:, 2])
            coupling = 0.5 * DT_S * r_mid
            rhs_u = (current[:, 0] + DT_S * main_delta[:, 1]
                     + coupling * current[:, 1])
            rhs_v = (current[:, 1] + DT_S * ay_step
                     - coupling * current[:, 0])
            denominator = 1.0 + coupling * coupling
            following[:, 0] = (rhs_u + coupling * rhs_v) / denominator
            following[:, 1] = (rhs_v - coupling * rhs_u) / denominator
        elif u_update == "increment":
            following[:, 0] += main_delta[:, 0]
        elif u_update == "effective_ax_midpoint":
            # The learned effective a_x target is u_dot - r_mid*v_mid.
            # Reconstruct u_dot with the same body-frame transport term,
            # using this rollout's predicted midpoint lateral/yaw states.
            r_mid = 0.5 * (current[:, 2] + following[:, 2])
            v_mid = 0.5 * (current[:, 1] + following[:, 1])
            following[:, 0] += DT_S * (main_delta[:, 1] + r_mid * v_mid)
        else:
            raise ValueError(f"unknown u update mode {u_update!r}")
        if ay_step is not None and u_update == "increment":
            r_mid = 0.5 * (current[:, 2] + following[:, 2])
            u_mid = 0.5 * (current[:, 0] + following[:, 0])
            following[:, 1] = current[:, 1] + DT_S * (ay_step - u_mid * r_mid)
        if teacher_force_state is not None:
            # Deliberate diagnostic-only intervention: replace one predicted
            # state channel with its future simulator label at every step.
            # This identifies which recursive feedback path drives divergence;
            # it is never a deployable rollout mode.
            following[:, teacher_force_channels[teacher_force_state]] = (
                state[source + 1][:, teacher_force_channels[teacher_force_state]])
        if raw_history is not None and raw_input_mode == "predicted":
            raw_delta = np.column_stack([
                model.predict(x) for model in raw_models])
            raw_following = raw_history[:, -1, :].astype(np.float64) + raw_delta
            raw_prediction[:, step + 1] = raw_following
        elif (raw_history is not None
              and raw_input_mode == "predicted_contact_slip"):
            half_track = float(bundle.get(
                "contact_slip_rear_track_m", REAR_TRACK_M)) / 2.0
            slip_feature = (x if bundle.get("uses_contact_slip_state_features")
                            else _append_contact_slip_features(
                                x, history, raw_history, lags, half_track))
            contact_history = np.stack((
                history[:, :, 0] - half_track * history[:, :, 2],
                history[:, :, 0] + half_track * history[:, :, 2]), axis=2)
            slip_history = raw_history.astype(np.float64) - contact_history
            slip_current = slip_history[:, -1, :]
            slip_delta = np.column_stack([
                model.predict(slip_feature) for model in contact_slip_models])
            contact_following = np.column_stack((
                following[:, 0] - half_track * following[:, 2],
                following[:, 0] + half_track * following[:, 2]))
            raw_following = slip_current + slip_delta + contact_following
            raw_prediction[:, step + 1] = raw_following
        predicted_state[:, step + 1] = following
        ax_prediction[:, step] = main_delta[:, 1]
        actual_now, actual_next = state[source], state[source + 1]
        ax_truth[:, step] = ((actual_next[:, 0] - actual_now[:, 0]) / DT_S
                             - 0.5 * (actual_now[:, 2] + actual_next[:, 2])
                             * 0.5 * (actual_now[:, 1] + actual_next[:, 1]))
        if ay_step is not None:
            ay_prediction[:, step] = ay_step
            ay_truth[:, step] = (
                (actual_next[:, 1] - actual_now[:, 1]) / DT_S
                + 0.5 * (actual_now[:, 0] + actual_next[:, 0])
                * 0.5 * (actual_now[:, 2] + actual_next[:, 2]))
        predicted_pose[:, step + 1] = _pose_step(
            predicted_pose[:, step], current, following)
        baseline_state[:, step + 1] = baseline_state[:, 0]
        baseline_pose[:, step + 1] = _pose_step(
            baseline_pose[:, step], baseline_state[:, 0], baseline_state[:, 0])

        if step + 1 < horizon:
            next_commands = frames[source + 1, 7:9]
            next_row = np.column_stack((following, next_commands)).astype(np.float32)
            history = np.concatenate((history[:, 1:, :], next_row[:, None, :]), axis=1)
            if raw_history is not None and raw_input_mode in (
                    "predicted", "predicted_contact_slip"):
                raw_history = np.concatenate((
                    raw_history[:, 1:, :],
                    raw_following[:, None, :].astype(np.float32)), axis=1)

    report_by_run: dict[str, Any] = {}
    # Every outer horizon gets its own eligible, run-balanced window set; emit
    # only that endpoint here so the report does not duplicate shorter-prefix
    # scores from the same windows.
    horizons_to_score = (horizon,)
    for run in sorted(np.unique(run_index).tolist()):
        mask = run_index == run
        pred_runs: dict[str, Any] = {}
        for h in horizons_to_score:
            if h > horizon:
                continue
            p_state = predicted_state[mask, h]
            b_state = baseline_state[mask, h]
            target_state = state[starts[mask] + h]
            p_error = p_state - target_state
            b_error = b_state - target_state
            p_wheel_error = p_error[:, 5:7]
            b_wheel_error = b_error[:, 5:7]
            pose_rows = starts[mask] + h
            target_pose = pose_truth[pose_rows]
            valid_pose = np.isfinite(target_pose).all(axis=1)
            pose_err = predicted_pose[mask, h] - target_pose
            baseline_pose_err = baseline_pose[mask, h] - target_pose
            pose_err[:, 2] = _wrap_angle(pose_err[:, 2])
            baseline_pose_err[:, 2] = _wrap_angle(baseline_pose_err[:, 2])
            local = {
                "windows": int(mask.sum()),
                "candidate": {
                    "u_rmse_mps": _rmse(p_error[:, 0]),
                    "v_rmse_mps": _rmse(p_error[:, 1]),
                    "yaw_rate_rmse_rps": _rmse(p_error[:, 2]),
                    "steering_feedback_rmse_rad": _rmse(p_error[:, 3]),
                    "throttle_feedback_rmse_norm": _rmse(p_error[:, 4]),
                    "rear_wheel_pair_rmse_mps": _rmse(p_wheel_error),
                    "predicted_speed_over_12mps_fraction": float(np.mean(
                        np.hypot(p_state[:, 0], p_state[:, 1]) > MAX_SPEED_MPS)),
                },
                "constant_state_baseline": {
                    "u_rmse_mps": _rmse(b_error[:, 0]),
                    "v_rmse_mps": _rmse(b_error[:, 1]),
                    "yaw_rate_rmse_rps": _rmse(b_error[:, 2]),
                    "steering_feedback_rmse_rad": _rmse(b_error[:, 3]),
                    "throttle_feedback_rmse_norm": _rmse(b_error[:, 4]),
                    "rear_wheel_pair_rmse_mps": _rmse(b_wheel_error),
                },
                "pose_scored_windows": int(valid_pose.sum()),
            }
            if raw_prediction is not None:
                raw_error = (raw_prediction[mask, h]
                             - raw_truth[starts[mask] + h])
                raw_base_error = (raw_baseline[mask]
                                  - raw_truth[starts[mask] + h])
                local["candidate"]["raw_encoder_pair_rmse_mps"] = _rmse(raw_error)
                local["constant_state_baseline"][
                    "raw_encoder_pair_rmse_mps"] = _rmse(raw_base_error)
            if valid_pose.any():
                local["candidate"].update({
                    "position_rmse_m": _rmse(pose_err[valid_pose, :2]),
                    "position_error_p95_m": float(np.quantile(
                        np.linalg.norm(pose_err[valid_pose, :2], axis=1), 0.95)),
                    "yaw_pose_rmse_rad": _rmse(pose_err[valid_pose, 2]),
                })
                local["constant_state_baseline"].update({
                    "position_rmse_m": _rmse(baseline_pose_err[valid_pose, :2]),
                    "yaw_pose_rmse_rad": _rmse(baseline_pose_err[valid_pose, 2]),
                })
            pred_runs[f"{h * DT_S:.3f}s"] = local
        report_by_run[str(run_ids[run])] = pred_runs

    macro: dict[str, Any] = {}
    metric_names = ("u_rmse_mps", "v_rmse_mps", "yaw_rate_rmse_rps",
                    "steering_feedback_rmse_rad", "throttle_feedback_rmse_norm",
                    "rear_wheel_pair_rmse_mps", "position_rmse_m",
                    "yaw_pose_rmse_rad")
    if raw_prediction is not None:
        metric_names += ("raw_encoder_pair_rmse_mps",)
    for h in horizons_to_score:
        if h > horizon:
            continue
        key = f"{h * DT_S:.3f}s"
        candidate = {name: [] for name in metric_names}
        baseline = {name: [] for name in metric_names}
        for run_report in report_by_run.values():
            row = run_report[key]
            for name in metric_names:
                if name in row["candidate"] and name in row["constant_state_baseline"]:
                    candidate[name].append(row["candidate"][name])
                    baseline[name].append(row["constant_state_baseline"][name])
        macro[key] = {
            "eligible_whole_runs": len(report_by_run),
            "candidate_macro_run_rmse": {
                name: float(np.mean(values)) for name, values in candidate.items()
                if values},
            "constant_state_baseline_macro_run_rmse": {
                name: float(np.mean(values)) for name, values in baseline.items()
                if values},
            "candidate_runs_better_than_baseline": {
                name: int(sum(c < b for c, b in zip(candidate[name], baseline[name])))
                for name in candidate if candidate[name]},
        }
    recursive_ax_error = ax_prediction - ax_truth
    recursive_ay_report = None
    if lateral_acceleration_model is not None:
        recursive_ay_error = ay_prediction - ay_truth
        recursive_ay_by_run = {
            str(run_ids[run]): _rmse(recursive_ay_error[run_index == run])
            for run in sorted(np.unique(run_index).tolist())}
        recursive_ay_report = {
            "pooled_rmse_mps2": _rmse(recursive_ay_error),
            "pooled_bias_mps2": float(np.mean(recursive_ay_error)),
            "macro_run_rmse_mps2": float(np.mean(
                list(recursive_ay_by_run.values()))),
            "per_run_rmse_mps2": recursive_ay_by_run,
        }
    report = {
        "horizon_steps": horizon,
        "u_update": u_update,
        "teacher_force_state_diagnostic_only": teacher_force_state,
        "fixed_dt_s": DT_S,
        "windows": len(windows),
        "window_signature_sha256": hashlib.sha256(json.dumps(
            [(int(row["run"]), int(row["sequence"]), int(row["start"]))
             for row in windows], separators=(",", ":")).encode()).hexdigest(),
        "windows_by_run": {str(run_ids[run]): int(np.sum(run_index == run))
                           for run in sorted(np.unique(run_index).tolist())},
        "per_run": report_by_run,
        "macro_run_metrics": macro,
        "recursive_effective_ax_prediction": {
            "rmse_mps2": _rmse(recursive_ax_error),
            "bias_mps2": float(np.mean(recursive_ax_error)),
            "absolute_error_p95_mps2": float(np.quantile(
                np.abs(recursive_ax_error), 0.95)),
            "scored_recursive_transitions": int(recursive_ax_error.size),
        },
        "recursive_lateral_acceleration_prediction": recursive_ay_report,
    }
    return report


def evaluate(bundle_path: Path, dataset_path: Path, practice_path: Path,
             output_path: Path, max_windows_per_run: int,
             window_context_steps: int | None = None,
             u_update: str = "increment",
             require_raw_windows: bool = False,
             raw_input_mode: str = "predicted",
             teacher_force_state: str | None = None,
             allow_external_dataset: bool = False) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite recursive report: {output_path}")
    bundle = joblib.load(bundle_path)
    if not bundle.get("research_only") or not bundle.get("recursive_aux_models"):
        raise ValueError("bundle is not a recursive research candidate")
    lags = _bundle_lags(bundle)
    context_steps = max(lags) if window_context_steps is None else window_context_steps
    if context_steps < max(lags):
        raise ValueError("window context must cover the candidate's longest lag")
    same_training_dataset = (
        _sha256(dataset_path) == bundle["training_dataset_sha256"])
    if not same_training_dataset and not allow_external_dataset:
        raise ValueError("training dataset does not match the frozen candidate bundle")
    results: dict[str, Any] = {}
    for name, path, split in (("openplane_validation", dataset_path, "validation"),
                              ("unseen_practice", practice_path, "unseen_practice")):
        data = _load_dataset(path)
        if split not in set(np.asarray(data["splits"]).astype(str)):
            raise ValueError(f"{path} has no split {split!r}")
        if allow_external_dataset:
            score_runs = set(np.asarray(data["run_ids"])[
                np.asarray(data["splits"]).astype(str) == split].astype(str))
            overlap = score_runs.intersection(bundle["training_run_ids"])
            if overlap:
                raise ValueError(
                    f"external {split} runs overlap training: {sorted(overlap)}")
        results[name] = {}
        for horizon in HORIZONS:
            windows = _select_windows(data, split, horizon, max_windows_per_run,
                                      context_steps)
            require_raw = (bundle.get("raw_wheel_models") is not None
                           or require_raw_windows)
            if require_raw:
                raw_valid = data.get("encoder_raw_valid")
                if raw_valid is None:
                    raise ValueError("paired raw-window evaluation requires encoder validity")
                raw_valid = np.asarray(raw_valid, dtype=bool)
                dt = np.asarray(data["dt_s"], dtype=np.float64)
                window_lags = tuple(int(value) for value in bundle.get(
                    "raw_wheel_history_lags_steps", lags))
                minimum_lag = max(max(window_lags), context_steps)
                windows = [window for window in windows
                           if raw_valid[window["start"] - minimum_lag:
                                        window["start"] + horizon + 1].all()
                           and np.isclose(
                               dt[window["start"] - minimum_lag + 1:
                                  window["start"] + horizon + 1],
                               DT_S, rtol=0.0, atol=1e-7).all()]
            if not windows:
                results[name][f"{horizon * DT_S:.3f}s"] = {
                    "windows": 0, "reason": "no eligible contiguous whole-run windows"}
                continue
            results[name][f"{horizon * DT_S:.3f}s"] = _run_rollout(
                data, bundle, windows, horizon, lags, u_update,
                raw_input_mode, teacher_force_state)
    report = {
        "schema_version": 1,
        "purpose": "recursive free-run state and pose validation for the longitudinal/wheel transition candidate",
        "candidate_bundle": str(bundle_path.relative_to(ROOT)),
        "candidate_bundle_sha256": _sha256(bundle_path),
        "training_dataset_sha256": bundle["training_dataset_sha256"],
        "openplane_dataset_is_external": not same_training_dataset,
        "openplane_dataset_sha256": _sha256(dataset_path),
        "practice_dataset_sha256": _sha256(practice_path),
        "test_and_final_test_policy": "not used; only validation whole runs and unseen_practice runs are rolled out",
        "rollout_input_policy": {
            "ground_truth_used_only_for_initial_context_and_scoring": True,
            "future_actuator_feedback_or_vehicle_state_used_as_input": False,
            "recorded_command_sequence_used_as_exogenous_replay_input": True,
            "future_commands_are_known_replay_inputs_not_sensor_feedback": True,
            "no_state_clipping_or_safety_fallback": True,
            "predicted_raw_encoder_wheel_state_used_as_input": (
                bundle.get("raw_wheel_models") is not None
                and raw_input_mode == "predicted"),
            "recorded_future_raw_encoder_input_debug_only": (
                bundle.get("raw_wheel_models") is not None
                and raw_input_mode == "oracle"),
            "paired_raw_valid_fixed_dt_windows_required": bool(
                require_raw_windows or bundle.get("raw_wheel_models") is not None),
        },
        "feature_variant": bundle["selected_variant"],
        "u_update": u_update,
        "raw_input_mode": raw_input_mode,
        "teacher_force_state_diagnostic_only": teacher_force_state,
        "history_lags_steps": list(lags),
        "minimum_pre_rollout_context_steps": context_steps,
        "permitted_training_run_ids": bundle["training_run_ids"],
        "max_windows_per_run_per_horizon": max_windows_per_run,
        "horizons": results,
        "limitations": [
            "no odometry sensor-only input; initial body state is simulator truth",
            "no MPC or runtime integration",
            "pose is integrated from predicted body velocity and yaw rate; this candidate is not yet a production full-lap simulator",
            "window-level runs are correlated; run, not window, is the uncertainty cluster",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--practice-dataset", type=Path, default=DEFAULT_PRACTICE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-windows-per-run", type=int, default=24)
    parser.add_argument("--window-context-steps", type=int, default=None,
                        help="GT initialization context for fair model comparisons")
    parser.add_argument("--u-update", choices=("increment", "effective_ax_midpoint"),
                        default="increment")
    parser.add_argument("--require-raw-windows", action="store_true",
                        help="restrict a reference rollout to fixed-encoder-valid matched windows")
    parser.add_argument("--raw-input-mode", choices=("predicted", "oracle"),
                        default="predicted",
                        help="oracle mode is a non-deployable debug ablation using future encoder data")
    parser.add_argument("--teacher-force-state", choices=(
        "u", "v", "yaw_rate", "steering_feedback", "throttle_feedback",
        "left_filtered_wheel", "right_filtered_wheel", "filtered_wheels"),
        help="diagnostic-only: replace one predicted state channel with future GT at each step")
    parser.add_argument("--allow-external-dataset", action="store_true",
                        help="score a separate validation dataset after rejecting any training-run overlap")
    args = parser.parse_args()
    report = evaluate(args.bundle.resolve(), args.dataset.resolve(),
                      args.practice_dataset.resolve(), args.output.resolve(),
                      args.max_windows_per_run, args.window_context_steps,
                      args.u_update, args.require_raw_windows,
                      args.raw_input_mode, args.teacher_force_state,
                      args.allow_external_dataset)
    summary_horizons = ("0.025s", "0.100s", "0.250s", "0.500s",
                        "0.750s", "1.000s", "1.250s", "1.500s",
                        "1.750s", "2.000s", "3.000s", "5.000s")
    print(json.dumps({
        "output": str(args.output),
        "u_update": args.u_update,
        "raw_input_mode": args.raw_input_mode,
        "history_lags_steps": report["history_lags_steps"],
        "matched_window_hashes": {
            split: {key: report["horizons"][split][key].get(
                "window_signature_sha256") for key in summary_horizons
                if key in report["horizons"][split]}
            for split in ("openplane_validation", "unseen_practice")},
        "macro_run_rmse": {
            split: {
                key: report["horizons"][split][key]["macro_run_metrics"].get(
                    key, {}).get("candidate_macro_run_rmse", {})
                for key in summary_horizons
                if key in report["horizons"][split]
                and "macro_run_metrics" in report["horizons"][split][key]}
            for split in ("openplane_validation", "unseen_practice")},
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
