#!/usr/bin/env python3
"""Evaluate fixed-40-Hz longitudinal/wheel heads with EDSSM lateral state and yaw atlas.

The HGB transition has excellent one-step u/a_x and filtered-wheel accuracy,
but its auxiliary yaw head destabilizes recursive high-steer rollouts. This
research-only evaluator keeps HGB for u, actuator feedback, and rear wheels,
uses EDSSM for v and roll, and uses the established sensor-only yaw atlas for
yaw. After the initial measured history, atlas inputs are synthesized only
from predicted state, predicted actuator/wheel state, and replay commands.
Simulator truth is used only for initial plant state and scoring. No runtime
controller/odometry code is changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from tools.vehicle_dynamics_learning.evaluate_effective_teacher_highsteer_transfer import (
    _matched_windows,
)
from tools.vehicle_dynamics_learning.evaluate_recursive_longitudinal_candidate import (
    _bundle_lags,
    _features,
    _pose_step,
    _rmse,
    _run_rollout,
    _state as hgb_state_from_data,
)
from tools.vehicle_dynamics_learning.effective_race_teacher import (
    REAR_TRACK_WIDTH_M,
    append_roll_state,
    physical_state_from_dataset,
    raw_encoder_history_features,
    wheel_innovation_history_features,
)
from tools.vehicle_dynamics_learning.fit_longitudinal_with_fixed40_wheels import (
    DEFAULT_DATASET as HGB_TRAIN_DATASET,
    DEFAULT_PRACTICE,
)
from tools.vehicle_dynamics_learning.run_truncated_history_transition import ROOT
from tools.vehicle_dynamics_learning.score_effective_race_teacher import (
    _fixed_validation_windows,
    _load_model,
    _window_batch,
)
from tools.vehicle_dynamics_learning.train_nssm import _load_dataset
from tools.racing.specialists import fit_sensor_only_yaw_regime_atlas as yaw_atlas


HGB_BUNDLE = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "longitudinal_with_fixed40_wheels_candidate_20261009/"
    "longitudinal_raw_wheel_candidate.joblib")
YAW_ATLAS_BUNDLE = ROOT / (
    "live_runs/racing_model_diagnostics_20261008/"
    "sensor_only_yaw_regime_atlas_command_age_v4/"
    "sensor_only_yaw_regime_atlas.joblib")
EDSSM_CHECKPOINT = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "full_modeling_reset_20261001/replacement_offline_sim_raceline_20261002/"
    "encoder_raw_state_teacher_v1/"
    "edssm_gru_z32_e2_rollresidual_10s_yawonly_longitudinal_seed101/last.pt")
HIGHSTEER_DATASET = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "full_modeling_reset_20261001/highsteer_75_heldout_validation_plus_"
    "r03_20261004/openplane_dynamics_fixed40hz.npz")
PRACTICE_DATASET = DEFAULT_PRACTICE
OUTPUT = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "hybrid_fixed40_longitudinal_yaw_atlas_candidate_20261009/"
    "highsteer_practice_input_sensitivity_evaluation.json")
DT_S = 0.025
HORIZONS = (30, 80, 200)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _windows(data: dict[str, Any], dataset_path: Path,
             maximum_horizon: int,
             lags: tuple[int, ...]) -> list[dict[str, int]]:
    if dataset_path == HIGHSTEER_DATASET:
        _, raw_windows, *_ = _matched_windows(dataset_path)
        windows = [{"run": int(row["run"]),
                    "sequence": int(row["sequence_id"]),
                    "start": int(row["start"])} for row in raw_windows]
    else:
        state = physical_state_from_dataset(data, "filtered_odometry")
        selected = _fixed_validation_windows(
            data, state, max_windows_per_run=16,
            horizon_steps=maximum_horizon, split="unseen_practice")
        windows = [{"run": int(row["run"]),
                    "sequence": int(row.get("sequence_id", row.get("sequence"))),
                    "start": int(row["start"])} for row in selected]
    raw_valid_source = data.get("encoder_raw_valid")
    raw_valid = (None if raw_valid_source is None else
                 np.asarray(raw_valid_source, dtype=bool))
    sensor_valid = np.asarray(data["sensor_valid"], dtype=bool)
    attitude_valid = np.asarray(data["imu_attitude_valid"], dtype=bool)
    dt = np.asarray(data["dt_s"], dtype=np.float64)
    context = max(max(lags), 80)
    windows = [row for row in windows
               if int(data["bounds"][row["sequence"], 0])
               <= row["start"] - context
               and (raw_valid is None or raw_valid[
                   row["start"] - context:row["start"] + 1].all())
               and sensor_valid[row["start"] - max(yaw_atlas.HISTORY_LAGS):
                                row["start"] + 1].all()
               and attitude_valid[row["start"] - max(yaw_atlas.HISTORY_LAGS):
                                  row["start"] + 1].all()
               and np.isclose(
                   dt[row["start"] - context + 1:
                      row["start"] + maximum_horizon + 1],
                   DT_S, rtol=0.0, atol=1e-7).all()]
    if not windows:
        raise ValueError(f"no aligned fixed-40-Hz windows in {dataset_path}")
    return windows


def _atlas_initial_history(data: dict[str, Any], starts: np.ndarray,
                           windows: list[dict[str, int]]) -> tuple[np.ndarray, np.ndarray]:
    sensors = np.asarray(data["sensor_frames"], dtype=np.float32)
    attitude = np.asarray(data["imu_attitude_frames"], dtype=np.float32)
    history = np.empty((len(starts), max(yaw_atlas.HISTORY_LAGS) + 1, 11),
                       dtype=np.float32)
    ages = np.empty((len(starts), 4), dtype=np.float32)
    bounds = np.asarray(data["bounds"], dtype=np.int64)
    thresholds = yaw_atlas.AGE_CHANGE_THRESHOLDS
    signal_columns = (7, 0, 8, 1)  # command, feedback, command, feedback
    for row, (start, window) in enumerate(zip(starts, windows)):
        indices = np.arange(start - max(yaw_atlas.HISTORY_LAGS), start + 1)
        history[row] = np.column_stack((
            sensors[indices, :9], attitude[indices][:, (0, 2)]))
        sequence_start = int(bounds[window["sequence"], 0])
        age_start = max(sequence_start, int(start) - yaw_atlas.AGE_WINDOW_STEPS)
        age_indices = np.arange(age_start, int(start) + 1)
        for channel, (column, threshold) in enumerate(zip(
                signal_columns, thresholds)):
            ages[row, channel] = yaw_atlas._age_since_change(
                sensors[age_indices, column], threshold)[-1]
    return history, ages


def _atlas_event_labels(history: np.ndarray) -> np.ndarray:
    current = history[:, -1]
    previous = history[:, -2]
    two_back = history[:, -3]
    delta, command = current[:, 0], current[:, 7]
    delta_previous, command_two_back = previous[:, 0], two_back[:, 7]
    command_crossed_sign = (
        (np.abs(command) >= 0.008) & (np.abs(command_two_back) >= 0.008)
        & (command * command_two_back < 0.0))
    command_opposes_feedback = (
        (np.abs(command) >= 0.02) & (np.abs(delta) >= 0.025)
        & (command * delta < 0.0))
    reversal = command_crossed_sign | command_opposes_feedback | (
        delta * delta_previous < 0.0)
    command_unwind = np.abs(command) <= np.abs(delta) - 0.02
    command_turn_in = ((np.abs(command) >= np.abs(delta) + 0.02)
                       & (command * delta >= 0.0))
    feedback_turn_in = np.abs(delta) > np.abs(two_back[:, 0]) + 0.003
    feedback_unwind = np.abs(delta) < np.abs(two_back[:, 0]) - 0.003
    labels = np.full(len(history), "hold", dtype="U8")
    labels[command_unwind & ~reversal] = "unwind"
    labels[command_turn_in & ~command_unwind & ~reversal] = "turn_in"
    unresolved = ~(command_unwind | command_turn_in | reversal)
    labels[feedback_turn_in & unresolved] = "turn_in"
    labels[feedback_unwind & unresolved & ~feedback_turn_in] = "unwind"
    labels[reversal] = "reversal"
    return labels


def _atlas_features(history: np.ndarray, ages: np.ndarray) -> np.ndarray:
    """Build the frozen atlas's exact causal feature vector."""
    current = history[:, -1]
    previous = history[:, -2]
    wheel_mean = 0.5 * (current[:, 2] + current[:, 3])
    derived = np.column_stack((
        wheel_mean,
        current[:, 2] - current[:, 3],
        (wheel_mean - 0.5 * (previous[:, 2] + previous[:, 3])) / DT_S,
        (current[:, 0] - previous[:, 0]) / DT_S,
        (current[:, 1] - previous[:, 1]) / DT_S,
        current[:, 7] - current[:, 0],
        current[:, 8] - current[:, 1],
        current[:, 6] - previous[:, 6],
        ages[:, 0], ages[:, 1], ages[:, 2], ages[:, 3],
        ages[:, 1] - ages[:, 0], ages[:, 3] - ages[:, 2],
    )).astype(np.float32)
    lags = yaw_atlas.HISTORY_LAGS
    return np.concatenate(
        [history[:, -1 - lag, :] for lag in lags] + [derived], axis=1)


def _atlas_predict_core(history: np.ndarray, ages: np.ndarray,
                        bundle: dict[str, Any],
                        correction_model: Any | None = None
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Predict yaw and optionally correct it; expose causal fit features."""
    current = history[:, -1]
    wheel_mean = 0.5 * (current[:, 2] + current[:, 3])
    feature_history = _atlas_features(history, ages)
    speed_cell = np.floor(wheel_mean / yaw_atlas.SPEED_BIN_MPS).astype(np.int16)
    steering_cell = yaw_atlas._steer_cell(current[:, 0])
    events = _atlas_event_labels(history)
    grouped: dict[tuple[int, int, str], list[int]] = defaultdict(list)
    for i, key in enumerate(zip(speed_cell, steering_cell, events)):
        grouped[(int(key[0]), int(key[1]), str(key[2]))].append(i)
    residual = np.empty(len(history), dtype=np.float32)
    local_route = np.zeros(len(history), dtype=bool)
    for key, indices in grouped.items():
        selected = np.asarray(indices, dtype=np.int64)
        model = bundle["local_expert_models"].get(key)
        if model is None:
            model = bundle["global_event_models"].get(key[2])
        if model is None:
            raise ValueError(f"yaw atlas has no model for causal event {key[2]!r}")
        else:
            local_route[selected] = key in bundle["local_expert_models"]
        residual[selected] = model.predict(feature_history[selected]).astype(
            np.float32)
    atlas_prediction = current[:, 6] + residual
    correction_features = np.column_stack((feature_history, atlas_prediction))
    prediction = atlas_prediction
    if correction_model is not None:
        correction = np.asarray(
            correction_model.predict(correction_features), dtype=np.float32)
        if correction.shape != (len(history),) or not np.isfinite(correction).all():
            raise ValueError("yaw correction returned invalid values")
        prediction = atlas_prediction + correction
    return prediction, local_route, correction_features.astype(np.float32), atlas_prediction


def _atlas_predict(history: np.ndarray, ages: np.ndarray,
                   bundle: dict[str, Any],
                   correction_model: Any | None = None
                   ) -> tuple[np.ndarray, np.ndarray]:
    prediction, route, _, _ = _atlas_predict_core(
        history, ages, bundle, correction_model)
    return prediction, route


def _composed_acceleration_and_latent(edssm: Any, current: torch.Tensor,
                                      following: torch.Tensor,
                                      command: torch.Tensor,
                                      latent: torch.Tensor,
                                      acceleration_scheme: str = "current"
                                      ) -> tuple[torch.Tensor, torch.Tensor]:
    """Recompute EDSSM latent input from the transition the hybrid took.

    The prior composition advanced EDSSM's latent state using EDSSM's own
    predicted yaw/wheel accelerations, then replaced those states with the
    atlas/HGB outputs. This keeps the latent transition consistent with the
    composed physical state without changing the learned dynamics or physics.
    """
    dt = DT_S
    u, v, yaw = current[:, 0], current[:, 1], current[:, 2]
    u_next, v_next, yaw_next = following[:, 0], following[:, 1], following[:, 2]
    if acceleration_scheme == "current":
        u_transport, v_transport, yaw_transport = u, v, yaw
    elif acceleration_scheme == "midpoint":
        u_transport = 0.5 * (u + u_next)
        v_transport = 0.5 * (v + v_next)
        yaw_transport = 0.5 * (yaw + yaw_next)
    else:
        raise ValueError(
            f"unsupported acceleration scheme {acceleration_scheme!r}")
    ax = (u_next - u) / dt - yaw_transport * v_transport
    ay = (v_next - v) / dt + yaw_transport * u_transport
    yaw_acceleration = (yaw_next - yaw) / dt
    if edssm.wheel_dynamics_mode == "surface_acceleration":
        wheel_acceleration = (following[:, 5:7] - current[:, 5:7]) / dt
    elif edssm.wheel_dynamics_mode == "contact_slip":
        half_track = REAR_TRACK_WIDTH_M / 2.0
        contact_current = torch.stack((
            u - half_track * yaw, u + half_track * yaw), dim=-1)
        contact_next = torch.stack((
            u_next - half_track * yaw_next,
            u_next + half_track * yaw_next), dim=-1)
        slip_current = current[:, 5:7] - contact_current
        slip_next = following[:, 5:7] - contact_next
        wheel_acceleration = (slip_next - slip_current) / dt
    else:
        raise ValueError(f"unsupported EDSSM wheel mode {edssm.wheel_dynamics_mode!r}")
    acceleration = torch.cat((
        torch.stack((ax, ay, yaw_acceleration), dim=-1), wheel_acceleration),
        dim=-1)
    normalized_state = ((current - edssm.state_mean) / edssm.state_scale)[
        :, :edssm.dynamics_state_size]
    normalized_command = ((command - edssm.command_mean)
                          / edssm.command_scale)
    latent_input = torch.cat((
        normalized_state, normalized_command,
        acceleration / edssm.acceleration_bounds), dim=-1)
    next_latent = edssm.latent_transition(latent_input, latent)
    return acceleration, next_latent


def _rollout(data: dict[str, Any], hgb: dict[str, Any], edssm: Any,
             metadata: dict[str, Any], atlas_bundle: dict[str, Any],
             windows: list[dict[str, int]],
             horizon: int, yaw_delta_gain: float = 1.0,
             include_oracle_diagnostics: bool = True,
             reconcile_composed_latent: bool = True,
             latent_acceleration_scheme: str = "current",
             atlas_acceleration_source: str = "model_heads",
             yaw_correction_model: Any | None = None,
             yaw_correction_start_step: int = 0,
             lateral_v_update: str = "edssm",
             lateral_acceleration_model: Any | None = None,
             lateral_acceleration_gate: dict[str, tuple[float, float]] | None = None,
             lateral_acceleration_feature_variant: str = "base",
             hgb_main_override_models: list[Any] | None = None,
             hgb_main_gate: dict[str, tuple[float, float]] | None = None,
             hgb_main_residual_models: list[Any] | None = None,
             hgb_main_residual_start_step: int = 1,
             hgb_main_residual_gain: float = 1.0,
             raw_wheel_residual_models: list[Any] | None = None,
             raw_wheel_residual_start_step: int = 0,
             raw_wheel_residual_gain: float = 1.0,
             capture_hgb_main_residual_rows: bool = False,
             hgb_main_residual_target_mode: str = "state_correction",
             capture_raw_wheel_residual_rows: bool = False,
             hgb_v_increment_residual_model: Any | None = None,
             hgb_v_increment_residual_start_step: int = 1,
             hgb_v_increment_residual_gain: float = 1.0,
             hgb_v_increment_residual_use_edssm_latent: bool = False,
             hgb_v_fallback_model: Any | None = None,
             capture_hgb_v_increment_residual_rows: bool = False,
             lateral_v_residual_model: Any | None = None,
             lateral_v_residual_start_step: int = 1,
             lateral_v_residual_gain: float = 1.0,
             lateral_v_residual_feature_variant: str = "base",
             capture_lateral_v_residual_rows: bool = False,
             capture_trajectory_trace: bool = False,
             capture_yaw_correction_rows: bool = False,
             motion_source: str = "hgb_longitudinal") -> dict[str, Any]:
    if not np.isfinite(yaw_delta_gain):
        raise ValueError("yaw delta gain must be finite")
    if latent_acceleration_scheme not in ("current", "midpoint"):
        raise ValueError("latent acceleration scheme must be current or midpoint")
    if atlas_acceleration_source not in ("model_heads", "composed_transition"):
        raise ValueError(
            "atlas acceleration source must be model_heads or composed_transition")
    if motion_source not in ("hgb_longitudinal", "edssm_non_yaw"):
        raise ValueError("unknown non-yaw motion source")
    if (motion_source == "edssm_non_yaw"
            and (hgb_main_override_models is not None
                 or hgb_main_residual_models is not None)):
        raise ValueError("HGB longitudinal overrides require hgb_longitudinal motion")
    if lateral_v_update not in (
            "edssm", "kinematic_ay", "gated_lateral_acceleration",
            "gated_hgb_v_increment",
            "gated_hgb_v_increment_parent_fallback", "hgb_v_increment"):
        raise ValueError("unknown lateral v update")
    if lateral_v_update == "gated_lateral_acceleration":
        if lateral_acceleration_model is None or lateral_acceleration_gate is None:
            raise ValueError("gated lateral acceleration update requires model and gate")
    if (lateral_v_update in (
            "gated_hgb_v_increment",
            "gated_hgb_v_increment_parent_fallback")
            and hgb_main_gate is None):
        raise ValueError("gated HGB v increment requires the measured-support gate")
    if (lateral_v_update == "gated_hgb_v_increment_parent_fallback"
            and hgb_v_fallback_model is None):
        raise ValueError("parent-fallback v gate requires the frozen parent v head")
    if (hgb_v_fallback_model is not None
            and lateral_v_update != "gated_hgb_v_increment_parent_fallback"):
        raise ValueError("parent v fallback is only valid for the parent-fallback gate")
    if lateral_acceleration_feature_variant not in ("base", "roll"):
        raise ValueError("unknown lateral-acceleration feature variant")
    if lateral_acceleration_gate is not None:
        if set(lateral_acceleration_gate) != {"speed_mps", "absolute_steering_rad"}:
            raise ValueError("unexpected lateral-acceleration support gate")
        for bounds in lateral_acceleration_gate.values():
            if (len(bounds) != 2 or not np.isfinite(bounds).all()
                    or bounds[0] > bounds[1]):
                raise ValueError("invalid lateral-acceleration gate bounds")
    if (not isinstance(yaw_correction_start_step, int)
            or yaw_correction_start_step < 0
            or yaw_correction_start_step >= horizon):
        raise ValueError("yaw correction start step must be within the rollout")
    if hgb_main_override_models is not None and hgb_main_gate is None:
        raise ValueError("longitudinal head overrides require a support gate")
    if hgb_main_gate is not None:
        if set(hgb_main_gate) != {"speed_mps", "absolute_steering_rad"}:
            raise ValueError("unexpected longitudinal head support gate")
        for name, bounds in hgb_main_gate.items():
            if (len(bounds) != 2 or not np.isfinite(bounds).all()
                    or bounds[0] > bounds[1]):
                raise ValueError(f"invalid longitudinal gate bounds for {name}")
    if hgb_main_residual_models is not None and hgb_main_gate is None:
        raise ValueError("longitudinal residual models require a support gate")
    if (hgb_main_residual_models is not None
            and len(hgb_main_residual_models) != 3):
        raise ValueError("longitudinal residual needs u and two wheel heads")
    if (not isinstance(hgb_main_residual_start_step, int)
            or hgb_main_residual_start_step < 0
            or hgb_main_residual_start_step >= horizon):
        raise ValueError("longitudinal residual start step must be in the rollout")
    if not np.isfinite(hgb_main_residual_gain):
        raise ValueError("longitudinal residual gain must be finite")
    if raw_wheel_residual_models is not None and hgb_main_gate is None:
        raise ValueError("raw-wheel residual models require a training support gate")
    if (raw_wheel_residual_models is not None
            and len(raw_wheel_residual_models) != 2):
        raise ValueError("raw-wheel residual needs two wheel heads")
    if (not isinstance(raw_wheel_residual_start_step, int)
            or raw_wheel_residual_start_step < 0
            or raw_wheel_residual_start_step >= horizon):
        raise ValueError("raw-wheel residual start step must be within the rollout")
    if not np.isfinite(raw_wheel_residual_gain):
        raise ValueError("raw-wheel residual gain must be finite")
    if hgb_main_residual_target_mode not in (
            "state_correction", "truth_transition_increment"):
        raise ValueError("unknown longitudinal residual target mode")
    if (hgb_v_increment_residual_model is not None
            and lateral_v_update != "gated_hgb_v_increment"):
        raise ValueError("HGB v residual requires the gated HGB v update")
    if (not isinstance(hgb_v_increment_residual_start_step, int)
            or hgb_v_increment_residual_start_step < 0
            or hgb_v_increment_residual_start_step >= horizon):
        raise ValueError("HGB v residual start step must be within the rollout")
    if not np.isfinite(hgb_v_increment_residual_gain):
        raise ValueError("HGB v residual gain must be finite")
    if lateral_v_residual_model is not None and hgb_main_gate is None:
        raise ValueError("lateral residual model requires a support gate")
    if (not isinstance(lateral_v_residual_start_step, int)
            or lateral_v_residual_start_step < 0
            or lateral_v_residual_start_step >= horizon):
        raise ValueError("lateral residual start step must be within the rollout")
    if not np.isfinite(lateral_v_residual_gain):
        raise ValueError("lateral residual gain must be finite")
    if lateral_v_residual_feature_variant not in ("base", "roll_acceleration"):
        raise ValueError("unknown lateral residual feature variant")
    if capture_yaw_correction_rows and not np.all(
            np.asarray(data["splits"]).astype(str)[
                np.asarray([row["run"] for row in windows], dtype=np.int64)]
            == "train"):
        raise ValueError("yaw correction rows may only be captured from train runs")
    if capture_hgb_main_residual_rows and not np.all(
            np.asarray(data["splits"]).astype(str)[
                np.asarray([row["run"] for row in windows], dtype=np.int64)]
            == "train"):
        raise ValueError("longitudinal residual rows may only be captured from train runs")
    if capture_hgb_main_residual_rows and hgb_main_gate is None:
        raise ValueError("captured longitudinal residual rows require a support gate")
    if capture_raw_wheel_residual_rows and not np.all(
            np.asarray(data["splits"]).astype(str)[
                np.asarray([row["run"] for row in windows], dtype=np.int64)]
            == "train"):
        raise ValueError("raw-wheel residual rows may only be captured from train runs")
    if capture_raw_wheel_residual_rows and hgb_main_gate is None:
        raise ValueError("captured raw-wheel residual rows require a support gate")
    if capture_hgb_v_increment_residual_rows and not np.all(
            np.asarray(data["splits"]).astype(str)[
                np.asarray([row["run"] for row in windows], dtype=np.int64)]
            == "train"):
        raise ValueError("HGB v correction rows may only be captured from train runs")
    if capture_hgb_v_increment_residual_rows and (
            hgb_main_gate is None or lateral_v_update != "gated_hgb_v_increment"):
        raise ValueError("HGB v correction capture requires its gated update")
    if capture_lateral_v_residual_rows and not np.all(
            np.asarray(data["splits"]).astype(str)[
                np.asarray([row["run"] for row in windows], dtype=np.int64)]
            == "train"):
        raise ValueError("lateral residual rows may only be captured from train runs")
    if capture_lateral_v_residual_rows and hgb_main_gate is None:
        raise ValueError("captured lateral residual rows require a support gate")
    frames = np.asarray(data["frames"], dtype=np.float64)
    truth_state = hgb_state_from_data(data)
    truth_pose = np.asarray(data["simulator_pose_xyyaw"], dtype=np.float64)
    starts = np.asarray([row["start"] for row in windows], dtype=np.int64)
    run_indices = np.asarray([row["run"] for row in windows], dtype=np.int32)
    run_ids = np.asarray(data["run_ids"]).astype(str)
    lags = _bundle_lags(hgb)
    batch = len(starts)

    edssm_state = physical_state_from_dataset(
        data, str(metadata.get("wheel_state_source", "filtered_odometry")))
    if metadata.get("include_roll_state", False):
        edssm_state = append_roll_state(data, edssm_state)
    extra = (raw_encoder_history_features(data)
             if edssm.include_raw_encoder_history else
             wheel_innovation_history_features(data)
             if edssm.include_wheel_innovation_history else None)
    history_ed, initial_ed, delayed_ed, _, _, _ = _window_batch(
        data, edssm_state, windows, edssm.history_state_size, extra,
        horizon_steps=horizon, command_offset_frames=-1)
    pure_state = torch.as_tensor(initial_ed, dtype=torch.float32)
    hybrid_state = pure_state.clone()
    pure_delayed = torch.as_tensor(delayed_ed, dtype=torch.float32)
    hybrid_delayed = pure_delayed.clone()
    initial_latent = edssm.encode_history(
        torch.as_tensor(history_ed, dtype=torch.float32))
    pure_latent = initial_latent.clone()
    hybrid_latent = initial_latent.clone()

    # The HGB feature history is exactly the seven physical channels plus the
    # two commands. EDSSM's roll state remains in its own state vector.
    hgb_history = np.empty((batch, max(lags) + 1, 9), dtype=np.float32)
    for row, start in enumerate(starts):
        indices = np.arange(start - max(lags), start + 1)
        hgb_history[row, :, :7] = edssm_state[indices, :7]
        hgb_history[row, :, 7:9] = frames[indices, 7:9]
    raw_source = data.get("encoder_raw_surface_mps")
    raw_valid_source = data.get("encoder_raw_valid")
    raw_truth = (None if raw_source is None else
                 np.asarray(raw_source, dtype=np.float64))
    raw_valid = (None if raw_valid_source is None else
                 np.asarray(raw_valid_source, dtype=bool))
    if (raw_truth is None) != (raw_valid is None):
        raise ValueError("raw encoder values and validity must be supplied together")
    if capture_raw_wheel_residual_rows and (
            raw_truth is None or raw_valid is None):
        raise ValueError("raw-wheel residual capture requires raw encoder labels")
    raw_initial = (np.asarray(frames[:, 5:7], dtype=np.float64)
                   if raw_truth is None else raw_truth)
    atlas_history, atlas_ages = _atlas_initial_history(data, starts, windows)
    initial_correction = (yaw_correction_model
                          if yaw_correction_start_step == 0 else None)
    atlas_one_step_yaw, atlas_initial_route = _atlas_predict(
        atlas_history, atlas_ages, atlas_bundle, initial_correction)
    atlas_one_step_error = atlas_one_step_yaw - truth_state[starts + 1, 2]
    atlas_one_step_by_run: dict[str, Any] = {}
    for run in sorted(np.unique(run_indices).tolist()):
        mask = run_indices == run
        errors = atlas_one_step_error[mask]
        absolute = np.abs(errors)
        atlas_one_step_by_run[str(run_ids[run])] = {
            "windows": int(np.count_nonzero(mask)),
            "local_expert_coverage": float(np.mean(atlas_initial_route[mask])),
            "yaw_rate_rmse_rps": _rmse(errors),
            "yaw_rate_p95_abs_rps": float(np.quantile(absolute, 0.95)),
            "yaw_rate_max_abs_rps": float(np.max(absolute)),
            "fraction_abs_error_below_0p1": float(np.mean(absolute < 0.1)),
        }
    raw_history = np.empty((batch, max(lags) + 1, 2), dtype=np.float32)
    for row, start in enumerate(starts):
        indices = np.arange(start - max(lags), start + 1)
        raw_history[row] = raw_initial[indices].astype(np.float32)
        if raw_valid is not None:
            invalid = ~raw_valid[indices]
            raw_history[row, invalid] = frames[indices[invalid], 5:7]

    pose_hybrid = truth_pose[starts].copy()
    pose_edssm = truth_pose[starts].copy()
    raw_hybrid = np.empty((batch, horizon + 1, 2), dtype=np.float64)
    raw_hybrid[:, 0] = raw_history[:, -1]
    checkpoints: dict[int, dict[str, np.ndarray]] = {}
    oracle_groups = {
        "predicted_inputs": ((), ()),
        "true_imu_acceleration": ((4, 5), ()),
        "true_imu_yaw_rate": ((6,), ()),
        "true_rear_wheels": ((2, 3), ()),
        "true_actuator_feedback": ((0, 1), (1, 3)),
        "true_roll_state": ((9, 10), ()),
        "true_commands": ((7, 8), (0, 2)),
        "all_true_sensor_inputs": (tuple(range(11)), tuple(range(4))),
    }
    oracle_step_errors = {
        name: np.full((batch, horizon), np.nan, dtype=np.float64)
        for name in oracle_groups} if include_oracle_diagnostics else {}
    correction_feature_rows: list[np.ndarray] = []
    correction_target_rows: list[np.ndarray] = []
    correction_run_rows: list[np.ndarray] = []
    hgb_residual_feature_rows: list[np.ndarray] = []
    hgb_residual_target_rows: list[np.ndarray] = []
    hgb_residual_run_rows: list[np.ndarray] = []
    raw_wheel_residual_feature_rows: list[np.ndarray] = []
    raw_wheel_residual_target_rows: list[np.ndarray] = []
    raw_wheel_residual_run_rows: list[np.ndarray] = []
    hgb_v_residual_feature_rows: list[np.ndarray] = []
    hgb_v_residual_target_rows: list[np.ndarray] = []
    hgb_v_residual_run_rows: list[np.ndarray] = []
    lateral_residual_feature_rows: list[np.ndarray] = []
    lateral_residual_target_rows: list[np.ndarray] = []
    lateral_residual_run_rows: list[np.ndarray] = []
    hgb_gate_active_count = 0
    hgb_gate_total_count = 0
    lateral_accel_gate_active_count = 0
    lateral_accel_gate_total_count = 0
    trace_pred_state = (np.empty((batch, horizon, 7), dtype=np.float32)
                        if capture_trajectory_trace else None)
    trace_truth_state = (np.empty((batch, horizon, 7), dtype=np.float32)
                         if capture_trajectory_trace else None)
    trace_pred_pose = (np.empty((batch, horizon, 3), dtype=np.float64)
                       if capture_trajectory_trace else None)
    trace_truth_pose = (np.empty((batch, horizon, 3), dtype=np.float64)
                        if capture_trajectory_trace else None)
    trace_pred_raw_wheels = (np.empty((batch, horizon, 2), dtype=np.float64)
                             if capture_trajectory_trace else None)
    trace_truth_raw_wheels = (np.full((batch, horizon, 2), np.nan,
                                      dtype=np.float64)
                              if capture_trajectory_trace else None)
    trace_raw_wheels_valid = (np.zeros((batch, horizon), dtype=bool)
                              if capture_trajectory_trace else None)
    raw_models = hgb["raw_wheel_models"]
    if (len(hgb["models"]) != 4 or len(hgb["recursive_aux_models"]) != 4
            or len(raw_models) != 2):
        raise ValueError("unexpected HGB candidate head layout")

    edssm.eval()
    with torch.no_grad(), threadpool_limits(limits=2):
        for step in range(horizon):
            source = starts + step
            command_np = frames[source, 7:9].astype(np.float32)
            command = torch.as_tensor(command_np)
            pure_next, pure_delayed_next, pure_latent_next, _, _ = (
                edssm.transition(pure_state, pure_delayed, pure_latent,
                                 command))
            ed_next, hybrid_delayed_next, hybrid_latent_next, ed_acceleration, _ = (
                edssm.transition(hybrid_state, hybrid_delayed,
                                 hybrid_latent, command))

            x = _features(hgb_history, lags)
            raw_current = raw_history[:, -1, :]
            raw_parts = [raw_current]
            for lag in lags:
                raw_parts.append(raw_current - raw_history[:, -1 - lag, :])
            x = np.column_stack((x, *raw_parts)).astype(np.float32)
            if lateral_v_residual_feature_variant == "roll_acceleration":
                if hybrid_state.shape[1] < 9:
                    raise ValueError("roll-conditioned lateral residual needs predicted roll state")
                roll_features = hybrid_state[:, 7:9].cpu().numpy()
                predicted_ay = ed_acceleration[:, 1].cpu().numpy()
                lateral_x = np.column_stack((x, roll_features, predicted_ay))
            else:
                lateral_x = x
            if lateral_acceleration_feature_variant == "roll":
                if hybrid_state.shape[1] < 9:
                    raise ValueError("roll-conditioned lateral acceleration needs predicted roll")
                lateral_acceleration_x = np.column_stack((
                    x, hybrid_state[:, 7:9].cpu().numpy())).astype(np.float32)
            else:
                lateral_acceleration_x = x
            main_delta = np.column_stack([
                model.predict(x) for model in hgb["models"]])
            in_support: np.ndarray | None = None
            if hgb_main_gate is not None:
                current_body_speed = np.hypot(
                    hgb_history[:, -1, 0], hgb_history[:, -1, 1])
                current_abs_steering = np.abs(hgb_history[:, -1, 3])
                speed_bounds = hgb_main_gate["speed_mps"]
                steering_bounds = hgb_main_gate["absolute_steering_rad"]
                in_support = (
                    (current_body_speed >= speed_bounds[0])
                    & (current_body_speed <= speed_bounds[1])
                    & (current_abs_steering >= steering_bounds[0])
                    & (current_abs_steering <= steering_bounds[1]))
                hgb_gate_active_count += int(np.count_nonzero(in_support))
                hgb_gate_total_count += len(in_support)
            if hgb_main_override_models is not None:
                if len(hgb_main_override_models) != len(hgb["models"]):
                    raise ValueError("longitudinal override head count mismatch")
                override_delta = np.column_stack([
                    model.predict(x) for model in hgb_main_override_models])
                if in_support is None:
                    raise AssertionError("override gate mask was not constructed")
                main_delta = np.where(
                    in_support[:, None], override_delta, main_delta)
            if (capture_hgb_main_residual_rows
                    and step >= hgb_main_residual_start_step):
                if in_support is None:
                    raise AssertionError("residual capture gate mask is missing")
                predicted_current = hybrid_state[:, (0, 5, 6)].cpu().numpy()
                truth_next = truth_state[source + 1][:, (0, 5, 6)]
                base_delta = main_delta[:, (0, 2, 3)]
                if hgb_main_residual_target_mode == "truth_transition_increment":
                    truth_current = truth_state[source][:, (0, 5, 6)]
                    residual_target = (
                        truth_next - truth_current - base_delta)
                else:
                    residual_target = (
                        truth_next - predicted_current - base_delta)
                if np.any(in_support):
                    hgb_residual_feature_rows.append(x[in_support].copy())
                    hgb_residual_target_rows.append(
                        residual_target[in_support].astype(np.float32))
                    hgb_residual_run_rows.append(run_indices[in_support].copy())
            if (hgb_main_residual_models is not None
                    and step >= hgb_main_residual_start_step):
                if in_support is None:
                    raise AssertionError("residual correction gate mask is missing")
                residual_delta = np.column_stack([
                    model.predict(x) for model in hgb_main_residual_models])
                main_delta = main_delta.copy()
                for output, correction_index in zip((0, 2, 3), range(3)):
                    main_delta[in_support, output] += (
                        hgb_main_residual_gain * residual_delta[
                            in_support, correction_index])
            aux_delta = np.column_stack([
                model.predict(x) for model in hgb["recursive_aux_models"]])
            if (capture_hgb_v_increment_residual_rows
                    and step >= hgb_v_increment_residual_start_step):
                if in_support is None:
                    raise AssertionError("HGB v correction support gate is missing")
                hgb_v_base_next = (
                    hybrid_state[:, 1].cpu().numpy().astype(np.float64)
                    + aux_delta[:, 0])
                residual_target = (
                    truth_state[source + 1, 1] - hgb_v_base_next)
                if np.any(in_support):
                    residual_features = x
                    if hgb_v_increment_residual_use_edssm_latent:
                        residual_features = np.column_stack((
                            x, hybrid_latent.cpu().numpy()))
                    hgb_v_residual_feature_rows.append(
                        residual_features[in_support].copy())
                    hgb_v_residual_target_rows.append(
                        residual_target[in_support].astype(np.float32))
                    hgb_v_residual_run_rows.append(run_indices[in_support].copy())
            raw_delta = np.column_stack([
                model.predict(x) for model in raw_models])
            if (capture_raw_wheel_residual_rows
                    and step >= raw_wheel_residual_start_step):
                if in_support is None or raw_truth is None or raw_valid is None:
                    raise AssertionError("raw-wheel training capture is incomplete")
                target_indices = source + 1
                raw_target_valid = (
                    in_support & raw_valid[source] & raw_valid[target_indices]
                    & np.isfinite(raw_truth[source]).all(axis=1)
                    & np.isfinite(raw_truth[target_indices]).all(axis=1))
                if np.any(raw_target_valid):
                    target = (raw_truth[target_indices] - raw_truth[source]
                              - raw_delta)
                    raw_wheel_residual_feature_rows.append(
                        x[raw_target_valid].copy())
                    raw_wheel_residual_target_rows.append(
                        target[raw_target_valid].astype(np.float32))
                    raw_wheel_residual_run_rows.append(
                        run_indices[raw_target_valid].copy())
            if (raw_wheel_residual_models is not None
                    and step >= raw_wheel_residual_start_step):
                if in_support is None:
                    raise AssertionError("raw-wheel support gate is missing")
                raw_residual = np.column_stack([
                    model.predict(x) for model in raw_wheel_residual_models])
                raw_delta = raw_delta.copy()
                raw_delta[in_support] += (
                    raw_wheel_residual_gain * raw_residual[in_support])
            raw_following = raw_history[:, -1].astype(np.float64) + raw_delta

            # In the default splice EDSSM owns lateral/yaw/roll and HGB owns
            # forward speed, actuators, and rear wheels. The explicit
            # edssm_non_yaw ablation keeps EDSSM for every state except yaw,
            # which is still replaced by the frozen atlas below.
            hybrid_next = ed_next.clone()
            if (capture_lateral_v_residual_rows
                    and step >= lateral_v_residual_start_step):
                if in_support is None:
                    raise AssertionError("lateral residual capture gate is missing")
                residual_target = (
                    truth_state[source + 1, 1]
                    - ed_next[:, 1].cpu().numpy().astype(np.float64))
                if np.any(in_support):
                    lateral_residual_feature_rows.append(
                        lateral_x[in_support].copy())
                    lateral_residual_target_rows.append(
                        residual_target[in_support].astype(np.float32))
                    lateral_residual_run_rows.append(run_indices[in_support].copy())
            if (lateral_v_residual_model is not None
                    and step >= lateral_v_residual_start_step):
                if in_support is None:
                    raise AssertionError("lateral residual gate is missing")
                residual_delta = np.asarray(
                    lateral_v_residual_model.predict(lateral_x), dtype=np.float32)
                if (residual_delta.shape != (batch,)
                        or not np.isfinite(residual_delta).all()):
                    raise ValueError("lateral residual model returned invalid values")
                corrected_v = hybrid_next[:, 1].clone()
                corrected_v[in_support] += torch.as_tensor(
                    lateral_v_residual_gain * residual_delta[in_support],
                    dtype=torch.float32)
                hybrid_next[:, 1] = corrected_v
            if motion_source == "hgb_longitudinal":
                hybrid_next[:, 0] = (
                    hybrid_state[:, 0]
                    + torch.as_tensor(main_delta[:, 0], dtype=torch.float32))
                hybrid_next[:, 3:5] = (
                    hybrid_state[:, 3:5]
                    + torch.as_tensor(aux_delta[:, 2:4], dtype=torch.float32))
                hybrid_next[:, 5:7] = (
                    hybrid_state[:, 5:7]
                    + torch.as_tensor(main_delta[:, 2:4], dtype=torch.float32))

            # The yaw atlas predicts the next yaw rate from this step's
            # sensor-equivalent history. Replace EDSSM yaw before pose update.
            (atlas_yaw_next, atlas_local_route, correction_features,
             atlas_base_yaw_next) = _atlas_predict_core(
                atlas_history, atlas_ages, atlas_bundle,
                yaw_correction_model if step >= yaw_correction_start_step
                else None)
            if capture_yaw_correction_rows and step >= yaw_correction_start_step:
                correction_feature_rows.append(correction_features.copy())
                correction_target_rows.append(
                    truth_state[source + 1, 2].astype(np.float32)
                    - atlas_base_yaw_next.astype(np.float32))
                correction_run_rows.append(run_indices.copy())
            # The atlas was trained as a measured-input one-step observer.
            # This scalar is an explicit research diagnostic/correction over
            # its predicted yaw increment, not a change to the frozen atlas.
            atlas_current_yaw = atlas_history[:, -1, 6]
            atlas_yaw_next = atlas_current_yaw + yaw_delta_gain * (
                atlas_yaw_next - atlas_current_yaw)
            hybrid_next[:, 2] = torch.as_tensor(
                atlas_yaw_next, dtype=torch.float32)
            if lateral_v_update == "kinematic_ay":
                u_mid = 0.5 * (hybrid_state[:, 0] + hybrid_next[:, 0])
                r_mid = 0.5 * (hybrid_state[:, 2] + hybrid_next[:, 2])
                v_next = (hybrid_state[:, 1]
                          + DT_S * (ed_acceleration[:, 1] - u_mid * r_mid))
                hybrid_next[:, 1] = v_next
            elif lateral_v_update == "gated_lateral_acceleration":
                if lateral_acceleration_gate is None:
                    raise AssertionError("lateral acceleration gate is missing")
                current_body_speed = torch.linalg.vector_norm(
                    hybrid_state[:, :2], dim=1).cpu().numpy()
                current_abs_steering = torch.abs(
                    hybrid_state[:, 3]).cpu().numpy()
                speed_bounds = lateral_acceleration_gate["speed_mps"]
                steering_bounds = lateral_acceleration_gate[
                    "absolute_steering_rad"]
                lateral_supported = (
                    (current_body_speed >= speed_bounds[0])
                    & (current_body_speed <= speed_bounds[1])
                    & (current_abs_steering >= steering_bounds[0])
                    & (current_abs_steering <= steering_bounds[1]))
                lateral_accel_gate_active_count += int(
                    np.count_nonzero(lateral_supported))
                lateral_accel_gate_total_count += len(lateral_supported)
                predicted_ay = np.asarray(
                    lateral_acceleration_model.predict(lateral_acceleration_x),
                    dtype=np.float32)
                if (predicted_ay.shape != (batch,)
                        or not np.isfinite(predicted_ay).all()):
                    raise ValueError("lateral acceleration model returned invalid values")
                u_mid = 0.5 * (hybrid_state[:, 0] + hybrid_next[:, 0])
                r_mid = 0.5 * (hybrid_state[:, 2] + hybrid_next[:, 2])
                v_from_ay = (hybrid_state[:, 1]
                             + DT_S * (torch.as_tensor(
                                 predicted_ay, dtype=torch.float32) - u_mid * r_mid))
                hybrid_next[:, 1] = torch.where(
                    torch.as_tensor(lateral_supported), v_from_ay,
                    hybrid_next[:, 1])
            elif lateral_v_update in (
                    "gated_hgb_v_increment",
                    "gated_hgb_v_increment_parent_fallback",
                    "hgb_v_increment"):
                v_from_hgb = hybrid_state[:, 1] + torch.as_tensor(
                    aux_delta[:, 0], dtype=torch.float32)
                if (hgb_v_increment_residual_model is not None
                        and step >= hgb_v_increment_residual_start_step):
                    residual_features = x
                    if hgb_v_increment_residual_use_edssm_latent:
                        residual_features = np.column_stack((
                            x, hybrid_latent.cpu().numpy()))
                    residual_v = np.asarray(
                        hgb_v_increment_residual_model.predict(residual_features),
                        dtype=np.float32)
                    if (residual_v.shape != (batch,)
                            or not np.isfinite(residual_v).all()):
                        raise ValueError("HGB v residual model returned invalid values")
                    v_from_hgb = v_from_hgb + torch.as_tensor(
                        hgb_v_increment_residual_gain * residual_v,
                        dtype=torch.float32)
                if lateral_v_update == "hgb_v_increment":
                    hybrid_next[:, 1] = v_from_hgb
                elif lateral_v_update == "gated_hgb_v_increment":
                    hybrid_next[:, 1] = torch.where(
                        torch.as_tensor(in_support), v_from_hgb,
                        hybrid_next[:, 1])
                else:
                    if in_support is None or hgb_v_fallback_model is None:
                        raise AssertionError("parent-fallback v gate is incomplete")
                    parent_delta = np.asarray(
                        hgb_v_fallback_model.predict(x), dtype=np.float32)
                    if (parent_delta.shape != (batch,)
                            or not np.isfinite(parent_delta).all()):
                        raise ValueError("parent v fallback returned invalid values")
                    parent_v = hybrid_state[:, 1] + torch.as_tensor(
                        parent_delta, dtype=torch.float32)
                    hybrid_next[:, 1] = torch.where(
                        torch.as_tensor(in_support), v_from_hgb, parent_v)

            # Counterfactual sensor substitutions diagnose which predicted
            # atlas inputs cause recursive error. These future sensor values
            # are scoring-only and never update the plant rollout.
            if include_oracle_diagnostics:
                oracle_history, oracle_ages = _atlas_initial_history(
                    data, source, windows)
                sensor_valid = np.asarray(data["sensor_valid"], dtype=bool)
                attitude_valid = np.asarray(data["imu_attitude_valid"], dtype=bool)
                diagnostic_valid = np.ones(batch, dtype=bool)
                for lag in yaw_atlas.HISTORY_LAGS:
                    diagnostic_valid &= sensor_valid[source - lag]
                    diagnostic_valid &= attitude_valid[source - lag]
                target_yaw_next = truth_state[source + 1, 2]
                oracle_step_errors["predicted_inputs"][:, step] = (
                    atlas_yaw_next - target_yaw_next)
                for name, (columns, age_columns) in oracle_groups.items():
                    if name == "predicted_inputs":
                        continue
                    mixed_history = atlas_history.copy()
                    mixed_ages = atlas_ages.copy()
                    if columns:
                        mixed_history[:, :, columns] = oracle_history[:, :, columns]
                    if age_columns:
                        mixed_ages[:, age_columns] = oracle_ages[:, age_columns]
                    mixed_yaw, _ = _atlas_predict(
                        mixed_history, mixed_ages, atlas_bundle,
                        yaw_correction_model)
                    mixed_current = mixed_history[:, -1, 6]
                    mixed_yaw = mixed_current + yaw_delta_gain * (
                        mixed_yaw - mixed_current)
                    oracle_step_errors[name][:, step] = (
                        mixed_yaw - target_yaw_next)
                for values in oracle_step_errors.values():
                    values[~diagnostic_valid, step] = np.nan

            ed_current7 = pure_state[:, :7].cpu().numpy().astype(np.float64)
            hybrid_current7 = hybrid_state[:, :7].cpu().numpy().astype(np.float64)
            ed_next7 = pure_next[:, :7].cpu().numpy().astype(np.float64)
            hybrid_next7 = hybrid_next[:, :7].cpu().numpy().astype(np.float64)
            pose_edssm = np.stack([
                _pose_step(pose_edssm[i:i + 1], ed_current7[i:i + 1],
                           ed_next7[i:i + 1])[0]
                for i in range(batch)])
            pose_hybrid = np.stack([
                _pose_step(pose_hybrid[i:i + 1], hybrid_current7[i:i + 1],
                           hybrid_next7[i:i + 1])[0]
                for i in range(batch)])
            if capture_trajectory_trace:
                assert trace_pred_state is not None
                assert trace_truth_state is not None
                assert trace_pred_pose is not None
                assert trace_truth_pose is not None
                assert trace_pred_raw_wheels is not None
                assert trace_truth_raw_wheels is not None
                assert trace_raw_wheels_valid is not None
                trace_pred_state[:, step] = hybrid_next7
                trace_truth_state[:, step] = truth_state[source + 1]
                trace_pred_pose[:, step] = pose_hybrid
                trace_truth_pose[:, step] = truth_pose[source + 1]
                trace_pred_raw_wheels[:, step] = raw_following
                if raw_truth is not None and raw_valid is not None:
                    target_indices = source + 1
                    trace_truth_raw_wheels[:, step] = raw_truth[target_indices]
                    trace_raw_wheels_valid[:, step] = (
                        raw_valid[target_indices]
                        & np.isfinite(raw_truth[target_indices]).all(axis=1))
            raw_hybrid[:, step + 1] = raw_following

            composed_acceleration, hybrid_latent_corrected = (
                _composed_acceleration_and_latent(
                    edssm, hybrid_state, hybrid_next, command, hybrid_latent,
                    latent_acceleration_scheme))

            if step + 1 < horizon:
                # The atlas was trained on IMU acceleration channels. Deriving
                # both channels from the selected composed transition keeps
                # the sensor history consistent with the physical state.
                if atlas_acceleration_source == "composed_transition":
                    ax_next = composed_acceleration[:, 0].cpu().numpy()
                    ay_next = composed_acceleration[:, 1].cpu().numpy()
                else:
                    ax_next = main_delta[:, 1]
                    ay_next = ed_acceleration[:, 1].cpu().numpy()
                next_commands = frames[source + 1, 7:9].astype(np.float32)
                sensor_next = np.column_stack((
                    hybrid_next7[:, 3], hybrid_next7[:, 4],
                    hybrid_next7[:, 5], hybrid_next7[:, 6],
                    ax_next, ay_next, hybrid_next7[:, 2],
                    next_commands[:, 0], next_commands[:, 1],
                )).astype(np.float32)
                if hybrid_next.shape[1] < 9:
                    raise ValueError("yaw atlas rollout requires predicted roll and roll-rate states")
                attitude_next = hybrid_next[:, (7, 8)].cpu().numpy()
                atlas_observation_next = np.column_stack((
                    sensor_next, attitude_next)).astype(np.float32)
                atlas_observation_current = atlas_history[:, -1]
                age_channels = ((7, 0, 8, 1), yaw_atlas.AGE_CHANGE_THRESHOLDS)
                updated_ages = []
                for age_index, (column, threshold) in enumerate(zip(*age_channels)):
                    changed = (np.abs(atlas_observation_next[:, column]
                                      - atlas_observation_current[:, column])
                               > threshold)
                    updated_ages.append(np.where(
                        changed, 0.0,
                        np.minimum(atlas_ages[:, age_index] + DT_S,
                                   yaw_atlas.AGE_WINDOW_STEPS * DT_S)))
                atlas_ages = np.column_stack(updated_ages).astype(np.float32)
                atlas_history = np.concatenate((
                    atlas_history[:, 1:, :], atlas_observation_next[:, None, :]),
                    axis=1)

            if step + 1 in HORIZONS:
                checkpoints[step + 1] = {
                    "edssm_state": ed_next7.copy(),
                    "hybrid_atlas_state": hybrid_next7.copy(),
                    "edssm_pose": pose_edssm.copy(),
                    "hybrid_atlas_pose": pose_hybrid.copy(),
                    "raw_hybrid": raw_hybrid[:, step + 1].copy(),
                    "atlas_local_route": atlas_local_route.copy(),
                }

            if step + 1 < horizon:
                next_row = np.column_stack((
                    hybrid_next7,
                    frames[source + 1, 7:9])).astype(np.float32)
                hgb_history = np.concatenate((
                    hgb_history[:, 1:, :], next_row[:, None, :]), axis=1)
                raw_history = np.concatenate((
                    raw_history[:, 1:, :],
                    raw_following[:, None, :].astype(np.float32)), axis=1)
            pure_state = pure_next
            pure_delayed = pure_delayed_next
            pure_latent = pure_latent_next
            hybrid_state = hybrid_next
            hybrid_delayed = hybrid_delayed_next
            hybrid_latent = (hybrid_latent_corrected
                             if reconcile_composed_latent
                             else hybrid_latent_next)

    per_run: dict[str, Any] = {}
    for endpoint, saved in checkpoints.items():
        key = f"{endpoint * DT_S:.3f}s"
        for run in sorted(np.unique(run_indices).tolist()):
            mask = run_indices == run
            target_indices = starts[mask] + endpoint
            truth = truth_state[target_indices]
            target_pose = truth_pose[target_indices]
            run_report = per_run.setdefault(str(run_ids[run]), {})
            run_report[key] = {}
            for mode in ("edssm", "hybrid_atlas"):
                predicted = saved[f"{mode}_state"][mask]
                pose = saved[f"{mode}_pose"][mask]
                pose_error = pose - target_pose
                pose_error[:, 2] = np.arctan2(
                    np.sin(pose_error[:, 2]), np.cos(pose_error[:, 2]))
                metric = {
                    "u_rmse_mps": _rmse(predicted[:, 0] - truth[:, 0]),
                    "v_rmse_mps": _rmse(predicted[:, 1] - truth[:, 1]),
                    "yaw_rate_rmse_rps": _rmse(predicted[:, 2] - truth[:, 2]),
                    "position_rmse_m": _rmse(pose_error[:, :2]),
                    "yaw_pose_rmse_rad": _rmse(pose_error[:, 2]),
                    "filtered_rear_wheel_pair_rmse_mps": _rmse(
                        predicted[:, 5:7] - truth[:, 5:7]),
                }
                if mode == "hybrid_atlas":
                    metric["raw_encoder_pair_rmse_mps"] = None
                    metric["raw_encoder_pair_valid_fraction"] = 0.0
                    if raw_valid is not None:
                        valid_raw_targets = raw_valid[target_indices]
                        valid_raw_targets &= np.isfinite(
                            raw_truth[target_indices]).all(axis=1)
                        metric["raw_encoder_pair_valid_fraction"] = float(
                            np.mean(valid_raw_targets))
                        if np.any(valid_raw_targets):
                            metric["raw_encoder_pair_rmse_mps"] = _rmse(
                                saved["raw_hybrid"][mask][valid_raw_targets]
                                - raw_truth[target_indices][valid_raw_targets])
                    metric["yaw_atlas_local_expert_route_fraction"] = (
                        float(np.mean(saved["atlas_local_route"][mask])))
                run_report[key][mode] = metric
    macro: dict[str, Any] = {}
    for endpoint in checkpoints:
        key = f"{endpoint * DT_S:.3f}s"
        macro[key] = {}
        for mode in ("edssm", "hybrid_atlas"):
            metric_names = tuple(next(iter(per_run.values()))[key][mode])
            macro[key][mode] = {}
            for metric in metric_names:
                numeric = [report[key][mode][metric]
                           for report in per_run.values()
                           if report[key][mode][metric] is not None]
                macro[key][mode][metric] = (
                    float(np.mean(numeric)) if numeric else None)
    atlas_input_sensitivity: dict[str, Any] = {}
    for endpoint in checkpoints:
        key = f"{endpoint * DT_S:.3f}s"
        per_run_sensitivity: dict[str, Any] = {}
        for run in sorted(np.unique(run_indices).tolist()):
            mask = run_indices == run
            per_run_sensitivity[str(run_ids[run])] = {}
            for name, errors in oracle_step_errors.items():
                values = errors[mask, endpoint - 1]
                values = values[np.isfinite(values)]
                if not len(values):
                    continue
                per_run_sensitivity[str(run_ids[run])][name] = {
                    "windows": int(len(values)),
                    "yaw_rate_rmse_rps": _rmse(values),
                    "yaw_rate_p95_abs_rps": float(np.quantile(np.abs(values), 0.95)),
                    "yaw_rate_max_abs_rps": float(np.max(np.abs(values))),
                    "fraction_abs_error_below_0p1": float(np.mean(np.abs(values) < 0.1)),
                }
        atlas_input_sensitivity[key] = per_run_sensitivity
    result = {"windows": len(windows),
            "windows_by_run": {
                str(run_ids[run]): int(np.sum(run_indices == run))
                for run in sorted(np.unique(run_indices).tolist())},
            "per_run": per_run,
            "macro_run_rmse": macro,
            "yaw_atlas_one_step_selected_windows": {
                "per_run": atlas_one_step_by_run,
                "local_expert_coverage": float(np.mean(atlas_initial_route)),
                "rmse_rps": _rmse(atlas_one_step_error),
                "p95_abs_rps": float(np.quantile(
                    np.abs(atlas_one_step_error), 0.95)),
                "max_abs_rps": float(np.max(np.abs(atlas_one_step_error))),
                "fraction_abs_error_below_0p1": float(np.mean(
                    np.abs(atlas_one_step_error) < 0.1)),
            },
            "yaw_atlas_delta_gain": float(yaw_delta_gain),
            "latent_acceleration_scheme": latent_acceleration_scheme,
            "atlas_acceleration_source": atlas_acceleration_source,
            "motion_source": motion_source,
            "lateral_v_update": lateral_v_update,
            "lateral_acceleration_model_used": lateral_acceleration_model is not None,
            "lateral_acceleration_gate": lateral_acceleration_gate,
            "lateral_acceleration_gate_active_fraction": (
                None if lateral_acceleration_model is None else
                float(lateral_accel_gate_active_count
                      / max(1, lateral_accel_gate_total_count))),
            "lateral_acceleration_feature_variant": lateral_acceleration_feature_variant,
            "yaw_correction_model_used": yaw_correction_model is not None,
            "hgb_main_override_used": hgb_main_override_models is not None,
            "hgb_main_gate": hgb_main_gate,
            "hgb_main_gate_active_fraction": (
                None if hgb_main_gate is None else
                float(hgb_gate_active_count / hgb_gate_total_count)),
            "hgb_main_residual_used": hgb_main_residual_models is not None,
            "hgb_main_residual_gain": float(hgb_main_residual_gain),
            "raw_wheel_residual_used": raw_wheel_residual_models is not None,
            "raw_wheel_residual_gain": float(raw_wheel_residual_gain),
            "hgb_v_increment_residual_used": (
                hgb_v_increment_residual_model is not None),
            "hgb_v_increment_residual_gain": float(
                hgb_v_increment_residual_gain),
            "hgb_v_increment_residual_use_edssm_latent": bool(
                hgb_v_increment_residual_use_edssm_latent),
            "hgb_v_fallback_model_used": hgb_v_fallback_model is not None,
            "lateral_v_residual_used": lateral_v_residual_model is not None,
            "lateral_v_residual_gain": float(lateral_v_residual_gain),
            "lateral_v_residual_feature_variant": lateral_v_residual_feature_variant,
            "composed_transition_latent_reconciled": bool(
                reconcile_composed_latent),
            "yaw_atlas_input_sensitivity_diagnostic_only": atlas_input_sensitivity}
    if capture_yaw_correction_rows:
        result["_yaw_correction_training_features"] = np.concatenate(
            correction_feature_rows, axis=0)
        result["_yaw_correction_training_residuals"] = np.concatenate(
            correction_target_rows, axis=0)
        result["_yaw_correction_training_run_indices"] = np.concatenate(
            correction_run_rows, axis=0)
    if capture_hgb_main_residual_rows:
        if not hgb_residual_feature_rows:
            raise ValueError("no in-support longitudinal residual rows were captured")
        result["_hgb_main_residual_training_features"] = np.concatenate(
            hgb_residual_feature_rows, axis=0)
        result["_hgb_main_residual_training_targets"] = np.concatenate(
            hgb_residual_target_rows, axis=0)
        result["_hgb_main_residual_training_run_indices"] = np.concatenate(
            hgb_residual_run_rows, axis=0)
    if capture_raw_wheel_residual_rows:
        if not raw_wheel_residual_feature_rows:
            raise ValueError("no in-support valid raw-wheel rows were captured")
        result["_raw_wheel_residual_training_features"] = np.concatenate(
            raw_wheel_residual_feature_rows, axis=0)
        result["_raw_wheel_residual_training_targets"] = np.concatenate(
            raw_wheel_residual_target_rows, axis=0)
        result["_raw_wheel_residual_training_run_indices"] = np.concatenate(
            raw_wheel_residual_run_rows, axis=0)
    if capture_hgb_v_increment_residual_rows:
        if not hgb_v_residual_feature_rows:
            raise ValueError("no in-support HGB v-increment correction rows were captured")
        result["_hgb_v_increment_residual_training_features"] = np.concatenate(
            hgb_v_residual_feature_rows, axis=0)
        result["_hgb_v_increment_residual_training_targets"] = np.concatenate(
            hgb_v_residual_target_rows, axis=0)
        result["_hgb_v_increment_residual_training_run_indices"] = np.concatenate(
            hgb_v_residual_run_rows, axis=0)
    if capture_lateral_v_residual_rows:
        if not lateral_residual_feature_rows:
            raise ValueError("no in-support lateral residual rows were captured")
        result["_lateral_v_residual_training_features"] = np.concatenate(
            lateral_residual_feature_rows, axis=0)
        result["_lateral_v_residual_training_targets"] = np.concatenate(
            lateral_residual_target_rows, axis=0)
        result["_lateral_v_residual_training_run_indices"] = np.concatenate(
            lateral_residual_run_rows, axis=0)
    if capture_trajectory_trace:
        assert trace_pred_state is not None
        assert trace_truth_state is not None
        assert trace_pred_pose is not None
        assert trace_truth_pose is not None
        assert trace_pred_raw_wheels is not None
        assert trace_truth_raw_wheels is not None
        assert trace_raw_wheels_valid is not None
        result["_trajectory_trace"] = {
            "source_indices": starts[:, None] + np.arange(horizon)[None, :],
            "run_indices": run_indices.copy(),
            "predicted_state": trace_pred_state,
            "truth_state": trace_truth_state,
            "predicted_pose": trace_pred_pose,
            "truth_pose": trace_truth_pose,
            "predicted_raw_encoder_wheels": trace_pred_raw_wheels,
            "truth_raw_encoder_wheels": trace_truth_raw_wheels,
            "raw_encoder_target_valid": trace_raw_wheels_valid,
        }
    return result


def run(hgb_path: Path, checkpoint_path: Path,
        yaw_atlas_path: Path, highsteer_path: Path, practice_path: Path,
        output_path: Path) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    hgb = joblib.load(hgb_path)
    atlas_bundle = joblib.load(yaw_atlas_path)
    expected_atlas_features = (
        len(yaw_atlas.HISTORY_LAGS) * len(yaw_atlas.OBSERVATION_NAMES)
        + len(yaw_atlas.DERIVED_NAMES))
    if (len(atlas_bundle["contract"]["features"]) != expected_atlas_features
            or any(model.n_features_in_ != expected_atlas_features
                   for model in atlas_bundle["global_event_models"].values())):
        raise ValueError("yaw atlas feature contract does not match this evaluator")
    torch_module, edssm, metadata = _load_model(checkpoint_path, "cpu")
    if torch_module is not torch:
        raise AssertionError("unexpected PyTorch module instance")
    if not metadata.get("include_roll_state", False):
        raise ValueError("yaw atlas composition requires predicted roll and roll rate")
    hgb_training = set(hgb["training_run_ids"])
    edssm_training = set(metadata["training_runs"])
    yaw_series, _ = yaw_atlas.teacher._discover_series_with_safe_mixed_archives()
    yaw_training = {series.run_id for series in yaw_series
                    if series.split == "train"}
    reports: dict[str, Any] = {}
    sources = (("highsteer_validation", highsteer_path),
               ("unseen_practice", practice_path))
    for name, path in sources:
        data = _load_dataset(path)
        splits = np.asarray(data["splits"]).astype(str)
        if name == "highsteer_validation":
            selected_run_ids = set(np.asarray(data["run_ids"]).astype(str)[
                splits == "validation"])
        else:
            selected_run_ids = set(np.asarray(data["run_ids"]).astype(str)[
                splits == "unseen_practice"])
        if not selected_run_ids:
            raise ValueError(f"no intended held-out runs in {path}")
        overlap = ((selected_run_ids & hgb_training)
                   | (selected_run_ids & edssm_training)
                   | (selected_run_ids & yaw_training))
        if overlap:
            raise ValueError(f"held-out runs overlap model training: {sorted(overlap)}")
        windows = _windows(data, path, max(HORIZONS), _bundle_lags(hgb))
        reports[name] = _rollout(data, hgb, edssm, metadata, atlas_bundle, windows,
                                 max(HORIZONS))
        hgb_result = _run_rollout(
            data, hgb, windows, max(HORIZONS), _bundle_lags(hgb),
            "increment", "predicted")
        longest_key = f"{max(HORIZONS) * DT_S:.3f}s"
        hgb_horizons = {
            longest_key: hgb_result["macro_run_metrics"][longest_key][
                "candidate_macro_run_rmse"]}
        reports[name]["hgb_only_macro_run_rmse"] = hgb_horizons
        reports[name]["heldout_run_ids"] = sorted(selected_run_ids)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "purpose": "diagnostic composition of HGB longitudinal/wheel heads, EDSSM lateral/roll states, and the established sensor-only yaw atlas",
        "research_only": True,
        "runtime_or_production_integration": False,
        "simulator_was_run": False,
        "hgb_bundle": str(hgb_path.relative_to(ROOT)),
        "hgb_bundle_sha256": _sha256(hgb_path),
        "edssm_checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "edssm_checkpoint_sha256": _sha256(checkpoint_path),
        "yaw_atlas_bundle": str(yaw_atlas_path.relative_to(ROOT)),
        "yaw_atlas_bundle_sha256": _sha256(yaw_atlas_path),
        "datasets": {
            "highsteer_validation": {
                "path": str(highsteer_path.relative_to(ROOT)),
                "sha256": _sha256(highsteer_path)},
            "unseen_practice": {
                "path": str(practice_path.relative_to(ROOT)),
                "sha256": _sha256(practice_path)}},
        "rollout_policy": {
            "ground_truth_only_initial_context_and_scoring": True,
            "future_sensor_or_truth_input": False,
            "commands_are_recorded_exogenous_inputs": True,
            "fixed_dt_s": DT_S,
            "edssm_command_alignment_offset_frames": -1,
            "yaw_atlas_feature_contract": atlas_bundle["contract"],
            "predicted_future_yaw_inputs": [
                "predicted yaw rate", "predicted steering/throttle feedback",
                "predicted rear wheel speeds", "HGB effective ax and EDSSM ay",
                "predicted EDSSM roll/roll rate", "recorded exogenous command"],
            "sensor_input_sensitivity": (
                "future measured channels are used only in separate counterfactual scoring branches; they never update the recursive plant state"),
            "test_final_test_used": False,
            "window_uncertainty_unit": "whole run"},
        "head_ownership": {
            "HGB": ["u", "steering_feedback", "throttle_feedback",
                    "filtered_rear_wheels", "raw_encoder_rate"],
            "EDSSM": ["v", "roll", "roll_rate"],
            "sensor_only_yaw_atlas": ["yaw_rate"]},
        "training_run_overlap": {
            "hgb": sorted(hgb_training),
            "edssm": sorted(edssm_training),
            "yaw_atlas": sorted(yaw_training)},
        "results": reports,
        "limitations": [
            "three independent high-steer validation runs and two practice runs provide limited run-level uncertainty",
            "fixed-40-Hz raw encoder rates are an encoder-derived measurement, not direct tire slip or tire force",
            "initial state is simulator truth; this does not evaluate sensor-only odometry",
            "research-only offline comparison; no MPC, odometry, or runtime integration"],
    }
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hgb", type=Path, default=HGB_BUNDLE)
    parser.add_argument("--checkpoint", type=Path, default=EDSSM_CHECKPOINT)
    parser.add_argument("--yaw-atlas", type=Path, default=YAW_ATLAS_BUNDLE)
    parser.add_argument("--highsteer-dataset", type=Path,
                        default=HIGHSTEER_DATASET)
    parser.add_argument("--practice-dataset", type=Path,
                        default=PRACTICE_DATASET)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    report = run(args.hgb.resolve(), args.checkpoint.resolve(),
                 args.yaw_atlas.resolve(),
                 args.highsteer_dataset.resolve(),
                 args.practice_dataset.resolve(), args.output.resolve())
    print(json.dumps({
        "output": str(args.output),
        "macro_run_rmse": {
            name: value["macro_run_rmse"]
            for name, value in report["results"].items()},
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
