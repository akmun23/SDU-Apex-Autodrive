#!/usr/bin/env python3
"""Fit a training-only local yaw residual and evaluate it in exact MPC C."""

from __future__ import annotations

import ctypes
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import lsq_linear

from tools.racing.offline.controller import evaluate_mpc_lateral_vdot_rollout as base
from tools.racing.offline.controller.compare_mpc_odom_initialization import (
    _workspace_path,
)
from tools.racing.offline.controller.production_mpc import ProductionMpc
from tools.racing.offline.controller.score_production_mpc_model import (
    MpcModelControl,
    MpcModelState,
    MpcStageResult,
    _read_commands,
    _recorded_command_arrays,
)
from tools.vehicle_dynamics_learning.build_practice_transfer_benchmark import (
    _run_rows,
)
from tools.vehicle_dynamics_learning.train_direct_sequence_teacher import COM_X_M
from tools.vehicle_dynamics_learning.train_nssm import _load_dataset


ROOT = Path(__file__).resolve().parents[4]
ANALYSIS_DIR = base.ANALYSIS_DIR
BENCHMARK = base.BENCHMARK
MODEL_LIBRARY = base.MODEL_LIBRARY
CONFIG = base.CONFIG
TRAJECTORY = base.TRAJECTORY
TRAJECTORY_ROWS = np.loadtxt(TRAJECTORY, delimiter=",", comments="#",
                             dtype=np.float64)
TRAINING_RUN_ID = "practice_filter_none_12lap_20260925"
TRAINING_BAG = ROOT / "live_runs/practice_filter_none_12lap_20260925/run/run_0.db3"
OUTPUT = ANALYSIS_DIR / "practice_low_mid_steer_yaw_gain_sweep.json"
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
GATE_PRESETS = {
    "current_broad": (0.08, 0.12, 0.24, 0.29),
    "focused_0p12_0p20": (0.105, 0.13, 0.18, 0.205),
    "focused_0p14_0p20": (0.13, 0.15, 0.18, 0.20),
    "focused_0p12_0p18": (0.10, 0.13, 0.17, 0.19),
    "focused_0p16_0p24": (0.14, 0.17, 0.22, 0.25),
}
FEATURE_NAMES = (
    "u", "r", "delta", "q_delta", "v_rear", "curvature", "ax",
    "q_speed", "u_delta_absdelta", "delta_abs_qdelta", "v_absdelta",
    "absdelta_ax", "delta_abs_qspeed")
YAW_ERROR_NAMES = ("yaw_rate_rps", "e_psi_rad", "e_y_m")


class MpcYawRateResidualModel(ctypes.Structure):
    _fields_ = [
        ("enabled", ctypes.c_int), ("gain", ctypes.c_float),
        ("correction_clip_radps2", ctypes.c_float),
        ("feature_mean", ctypes.c_float * 13),
        ("feature_scale", ctypes.c_float * 13),
        ("coefficients", ctypes.c_float * 14),
        ("target_speed_zero_mps", ctypes.c_float),
        ("target_speed_full_mps", ctypes.c_float),
        ("speed_deficit_full_mps", ctypes.c_float),
        ("speed_deficit_zero_mps", ctypes.c_float),
        ("abs_steering_zero_rad", ctypes.c_float),
        ("abs_steering_full_rad", ctypes.c_float),
        ("actual_speed_zero_mps", ctypes.c_float),
        ("actual_speed_full_mps", ctypes.c_float),
        ("actual_speed_upper_full_mps", ctypes.c_float),
        ("actual_speed_upper_zero_mps", ctypes.c_float),
        ("abs_steering_upper_full_rad", ctypes.c_float),
        ("abs_steering_upper_zero_rad", ctypes.c_float),
    ]


def _wrap(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def _model_gates(state: MpcModelState,
                 steering_bounds: tuple[float, float, float, float]
                 ) -> tuple[float, float, float]:
    def smoothstep(value: float, start: float, end: float) -> float:
        t = min(1.0, max(0.0, (value - start) / (end - start)))
        return t * t * (3.0 - 2.0 * t)

    delta = abs(float(state.actual_steering_angle))
    speed_gate = (smoothstep(float(state.u), 2.4, 2.8)
                  * (1.0 - smoothstep(float(state.u), 6.8, 7.3)))
    target_gate = smoothstep(float(state.target_speed), 2.4, 2.8)
    tracking_gate = 1.0 - smoothstep(
        float(state.target_speed - state.u), 0.10, 0.20)
    steering_gate = (smoothstep(delta, steering_bounds[0], steering_bounds[1])
                     * (1.0 - smoothstep(delta, steering_bounds[2],
                                         steering_bounds[3])))
    return speed_gate * target_gate * tracking_gate * steering_gate, speed_gate, steering_gate


def _features(state: MpcModelState, control: MpcModelControl,
              curvature: float, acceleration: float) -> np.ndarray:
    u, r, delta, v = (float(state.u), float(state.r),
                      float(state.actual_steering_angle), float(state.v))
    q_delta = min(3.2, max(-3.2, float(control.steering_rate)))
    q_speed = min(3.8, max(-8.0, float(control.target_speed_rate)))
    abs_delta = abs(delta)
    return np.asarray((
        u, r, delta, q_delta, v, curvature, acceleration, q_speed,
        u * delta * abs_delta, delta * abs(q_delta), v * abs_delta,
        abs_delta * acceleration, delta * abs(q_speed)), dtype=np.float64)


def _configure_library(library: ctypes.CDLL) -> None:
    library.mpc_vehicle_model_step.argtypes = [
        ctypes.POINTER(MpcModelState), ctypes.POINTER(MpcModelControl),
        ctypes.c_float, ctypes.c_float]
    library.mpc_vehicle_model_step.restype = MpcStageResult
    library.vehicle_model_set_yaw_rate_residual_model.argtypes = [
        ctypes.POINTER(MpcYawRateResidualModel)]
    library.vehicle_model_set_yaw_rate_residual_model.restype = ctypes.c_int


def _install_residual(library: ctypes.CDLL,
                      model: dict[str, Any] | None) -> None:
    value = MpcYawRateResidualModel()
    if model is not None:
        value.enabled = 1
        value.gain = float(model.get("gain", 1.0))
        value.correction_clip_radps2 = float(model["correction_clip_radps2"])
        value.feature_mean[:] = model["feature_mean"]
        value.feature_scale[:] = model["feature_scale"]
        value.coefficients[:] = model["coefficients"]
        for name, item in model["gates"].items():
            setattr(value, name, float(item))
    if not library.vehicle_model_set_yaw_rate_residual_model(ctypes.byref(value)):
        raise RuntimeError("production C model rejected the yaw residual parameters")


def _fit(features: np.ndarray, target: np.ndarray, gate: np.ndarray,
         alpha: float) -> dict[str, Any]:
    keep = gate > 0.05
    x, y, w = features[keep], target[keep] / gate[keep], gate[keep]
    mean = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    scale = np.where(scale > 1e-9, scale, 1.0)
    normalized = (x - mean) / scale
    design = np.column_stack((np.ones(len(x)), normalized))
    weights = np.sqrt(w)
    weighted_design = design * weights[:, None]
    weighted_target = y * weights
    penalty = np.zeros((features.shape[1], design.shape[1]))
    penalty[:, 1:] = np.eye(features.shape[1]) * math.sqrt(alpha)
    weighted_design = np.vstack((weighted_design, penalty))
    weighted_target = np.concatenate((weighted_target,
                                      np.zeros(features.shape[1])))
    lower = np.full(design.shape[1], -np.inf)
    upper = np.full(design.shape[1], np.inf)
    # The C residual is dt times this correction. Bound its direct feedback
    # derivative so it cannot cancel the production yaw-rate contraction.
    lower[2] = -5.0 * scale[1]
    upper[2] = 5.0 * scale[1]
    fitted = lsq_linear(weighted_design, weighted_target,
                        bounds=(lower, upper), max_iter=500)
    if not fitted.success or not np.isfinite(fitted.x).all():
        raise RuntimeError(f"yaw residual regression failed: {fitted.message}")
    coefficients = fitted.x.tolist()
    prediction = design @ fitted.x
    clip = max(1.0, float(np.quantile(np.abs(prediction), 0.999)) * 1.25)
    return {
        "feature_mean": mean.tolist(), "feature_scale": scale.tolist(),
        "coefficients": coefficients,
        "correction_clip_radps2": clip,
        "fit_rows": int(len(x)), "ridge_alpha": float(alpha),
        "raw_d_yaw_correction_d_r_per_s": float(fitted.x[2] / scale[1]),
        "one_step_fit_rmse_radps2": float(np.sqrt(np.mean((prediction - y) ** 2))),
    }


def _load_run_data(dataset_path: Path, run_id: str, bag_path: Path,
                   controller: ProductionMpc,
                   data: dict[str, Any], archive: dict[str, np.ndarray]
                   ) -> dict[str, Any]:
    ids = np.asarray(data["run_ids"]).astype(str)
    matches = np.flatnonzero(ids == run_id)
    if len(matches) != 1:
        raise ValueError(f"run ID is not unique in dataset: {run_id}")
    run_index = int(matches[0])
    splits = np.asarray(data["splits"]).astype(str)
    if run_id == TRAINING_RUN_ID and splits[run_index] != "train":
        raise ValueError("yaw residual fit run is not in the training split")
    indices = np.flatnonzero(archive["frame_run_index"] == run_index)
    if len(indices) < 4 or not np.all(np.diff(indices) == 1):
        raise ValueError(f"capture is not a continuous fixed-grid run: {run_id}")
    commands = _read_commands(bag_path)
    data_with_stamps = {**data, "sample_time_ns": archive["sample_time_ns"]}
    steering, target_speed = _recorded_command_arrays(
        data_with_stamps, run_index, commands, indices)
    projection = base._project_run(
        controller, np.loadtxt(TRAJECTORY, delimiter=",", comments="#"),
        controller.lap_length_m, archive["simulator_pose_xyyaw"][indices])
    projected = dict(zip(("progress", "e_y", "e_psi", "curvature", "distance"),
                         projection))
    return {
        "run_id": run_id, "run_index": run_index, "split": splits[run_index],
        "indices": indices, "steering": steering, "target_speed": target_speed,
        "lap_count": archive["lap_count"][indices].astype(np.int32),
        "projection": projected,
    }


def _state_at(run: dict[str, Any], local: int,
              rigid: np.ndarray, frames: np.ndarray) -> MpcModelState:
    index = int(run["indices"][local])
    steering = run["steering"]
    return MpcModelState(
        float(run["projection"]["e_y"][local]),
        float(run["projection"]["e_psi"][local]),
        float(rigid[index, 7]),
        float(rigid[index, 8] - COM_X_M * rigid[index, 12]),
        float(rigid[index, 12]),
        float(run["target_speed"][local]),
        float(steering[local]), float(steering[local - 1]),
        float(steering[local - 2]), float(frames[index, 3]))


def _step_controls(run: dict[str, Any], local: int) -> MpcModelControl:
    steering, target = run["steering"], run["target_speed"]
    return MpcModelControl(
        float((steering[local + 1] - steering[local]) / base.DT_S),
        float((target[local + 1] - target[local]) / base.DT_S))


def _training_rows(library: ctypes.CDLL, run: dict[str, Any],
                   rigid: np.ndarray, frames: np.ndarray, lap_count: np.ndarray,
                   trajectory: np.ndarray,
                   steering_bounds: tuple[float, float, float, float]
                   ) -> tuple[np.ndarray, ...]:
    library.vehicle_model_set_yaw_rate_residual_model(
        ctypes.byref(MpcYawRateResidualModel()))
    features, residuals, gates, lap_ids = [], [], [], []
    for local in range(2, len(run["indices"]) - 1):
        if not 1 <= int(lap_count[run["indices"][local]]) <= 8:
            continue
        state = _state_at(run, local, rigid, frames)
        control = _step_controls(run, local)
        curvature = float(run["projection"]["curvature"][local])
        stage = library.mpc_vehicle_model_step(
            ctypes.byref(state), ctypes.byref(control),
            ctypes.c_float(base.DT_S), ctypes.c_float(curvature))
        if not stage.valid:
            continue
        gate, _, _ = _model_gates(state, steering_bounds)
        features.append(_features(state, control, curvature,
                                  float(stage.body_accel_mps2)))
        next_index = int(run["indices"][local + 1])
        residuals.append((float(rigid[next_index, 12])
                          - float(stage.next.r)) / base.DT_S)
        gates.append(gate)
        lap_ids.append(int(lap_count[run["indices"][local]]))
    return (np.asarray(features), np.asarray(residuals), np.asarray(gates),
            np.asarray(lap_ids, dtype=np.int32))


def _metric(error: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "bias": float(np.mean(error)),
        "p95_abs": float(np.quantile(np.abs(error), 0.95)),
        "max_abs": float(np.max(np.abs(error))),
    }


def _teacher_forced(library: ctypes.CDLL, run: dict[str, Any],
                    rigid: np.ndarray, frames: np.ndarray,
                    candidate: dict[str, Any] | None,
                    lap_bounds: tuple[int, int] | None = None
                    ) -> dict[str, Any]:
    _install_residual(library, candidate)
    errors = []
    for local in range(2, len(run["indices"]) - 1):
        if lap_bounds is not None:
            lap = int(run["lap_count"][local])
            if not lap_bounds[0] <= lap <= lap_bounds[1]:
                continue
        state = _state_at(run, local, rigid, frames)
        control = _step_controls(run, local)
        curvature = float(run["projection"]["curvature"][local])
        stage = library.mpc_vehicle_model_step(
            ctypes.byref(state), ctypes.byref(control),
            ctypes.c_float(base.DT_S), ctypes.c_float(curvature))
        if not stage.valid:
            continue
        next_index = int(run["indices"][local + 1])
        errors.append((float(stage.next.r) - float(rigid[next_index, 12]),
                       _wrap(float(stage.next.e_psi)
                             - float(run["projection"]["e_psi"][local + 1])),
                       float(stage.next.e_y)
                       - float(run["projection"]["e_y"][local + 1])))
    values = np.asarray(errors, dtype=np.float64)
    _install_residual(library, None)
    return {name: _metric(values[:, channel])
            for channel, name in enumerate(YAW_ERROR_NAMES)}


def _rollout(library: ctypes.CDLL, run: dict[str, Any], rigid: np.ndarray,
             frames: np.ndarray, candidate: dict[str, Any] | None,
             lap_count: np.ndarray, trajectory: np.ndarray,
             lap_length_m: float, local_start: int = 2,
             local_end: int | None = None) -> dict[str, Any]:
    _install_residual(library, candidate)
    steering, target = run["steering"], run["target_speed"]
    start = max(2, int(local_start))
    if local_end is None:
        local_end = len(run["indices"]) - 1
    local_end = min(int(local_end), len(run["indices"]) - 1)
    state = _state_at(run, start, rigid, frames)
    progress = float(run["projection"]["progress"][start])
    errors = []
    failure = None
    for local in range(start, local_end):
        control = _step_controls(run, local)
        curvature = float(np.interp(
            progress % lap_length_m, trajectory[:, 0], trajectory[:, 4]))
        stage = library.mpc_vehicle_model_step(
            ctypes.byref(state), ctypes.byref(control),
            ctypes.c_float(base.DT_S), ctypes.c_float(curvature))
        if not stage.valid:
            failure = {"local_step": local - start + 1,
                       "elapsed_s": (local - start + 1) * base.DT_S,
                       "progress_m": progress,
                       "e_y_m": float(state.e_y),
                       "e_psi_rad": float(state.e_psi),
                       "yaw_rate_rps": float(state.r),
                       "reason": "production_motion_model_invalid"}
            break
        state = stage.next
        progress += float(stage.delta_s_m)
        target_index = int(run["indices"][local + 1])
        truth_ey = float(run["projection"]["e_y"][local + 1])
        truth_epsi = float(run["projection"]["e_psi"][local + 1])
        errors.append((
            float(state.r - rigid[target_index, 12]),
            _wrap(float(state.e_psi - truth_epsi)),
            float(state.e_y - truth_ey),
            int(lap_count[target_index]), float(state.u), float(state.r),
            float(frames[target_index, 3]), float(control.steering_rate),
            float(state.target_speed)))
    values = np.asarray(errors, dtype=np.float64)
    _install_residual(library, None)
    by_lap = {}
    for lap in np.unique(values[:, 3].astype(int)) if len(values) else []:
        mask = values[:, 3].astype(int) == lap
        by_lap[str(int(lap))] = {
            "samples": int(mask.sum()),
            **{name: _metric(values[mask, channel])
               for channel, name in enumerate(YAW_ERROR_NAMES)},
        }
    first_crossings = {}
    if len(values):
        for channel, name in enumerate(YAW_ERROR_NAMES):
            crossings = np.flatnonzero(np.abs(values[:, channel]) >= 0.1)
            if len(crossings):
                step = int(crossings[0])
                first_crossings[name] = {
                    "elapsed_s": float((step + 1) * base.DT_S),
                    "lap": int(values[step, 3]),
                    "error": float(values[step, channel]),
                    "u_mps": float(values[step, 4]),
                    "predicted_yaw_rate_rps": float(values[step, 5]),
                    "actual_steering_rad": float(values[step, 6]),
                    "steering_rate_radps": float(values[step, 7]),
                    "target_speed_mps": float(values[step, 8]),
                }
    return {
        "predicted_until_s": float(len(values) * base.DT_S),
        "failure": failure,
        "first_abs_0p1_error_crossings": first_crossings,
        "per_state": {name: _metric(values[:, channel])
                      for channel, name in enumerate(YAW_ERROR_NAMES)}
        if len(values) else {},
        "per_lap": by_lap,
    }


def _candidate_from_fit(
        fit: dict[str, Any],
        steering_bounds: tuple[float, float, float, float],
        gain: float = 1.0
        ) -> dict[str, Any]:
    return {
        **fit,
        "gain": float(gain),
        "gates": {
            "target_speed_zero_mps": 2.4,
            "target_speed_full_mps": 2.8,
            "speed_deficit_full_mps": 0.10,
            "speed_deficit_zero_mps": 0.20,
            "abs_steering_zero_rad": steering_bounds[0],
            "abs_steering_full_rad": steering_bounds[1],
            "actual_speed_zero_mps": 2.4,
            "actual_speed_full_mps": 2.8,
            "actual_speed_upper_full_mps": 6.8,
            "actual_speed_upper_zero_mps": 7.3,
            "abs_steering_upper_full_rad": steering_bounds[2],
            "abs_steering_upper_zero_rad": steering_bounds[3],
        },
    }


def run() -> dict[str, Any]:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    benchmark = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    if not benchmark.get("frozen"):
        raise ValueError("practice benchmark is not frozen")
    dataset_path = _workspace_path(benchmark["dataset"])
    if base._sha256(dataset_path) != benchmark["dataset_sha256"]:
        raise ValueError("practice dataset hash changed")
    if not TRAINING_BAG.is_file():
        raise FileNotFoundError(f"practice training bag is missing: {TRAINING_BAG}")
    for run_id, entry in benchmark["validation_reports"].items():
        bag = base.BAGS[run_id]
        if base._sha256(bag) != entry["bag_sha256"]:
            raise ValueError(f"held-out practice bag hash mismatch: {run_id}")

    data = _load_dataset(dataset_path)
    with np.load(dataset_path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in (
            "sample_time_ns", "frame_run_index", "simulator_rigid_state",
            "simulator_pose_xyyaw", "lap_count", "frames")}
    rigid = arrays["simulator_rigid_state"].astype(np.float64)
    frames = arrays["frames"].astype(np.float64)
    lap_count = arrays["lap_count"].astype(np.int32)
    trajectory = TRAJECTORY_ROWS
    library = ctypes.CDLL(str(MODEL_LIBRARY.resolve()))
    _configure_library(library)
    with ProductionMpc(MODEL_LIBRARY, CONFIG, TRAJECTORY) as controller:
        training = _load_run_data(
            dataset_path, TRAINING_RUN_ID, TRAINING_BAG,
            controller, data, arrays)
        validation = {
            run_id: _load_run_data(
                dataset_path,
                f"practice_unseen_model_validation_20261001_{run_id.rsplit('_', 1)[-1]}",
                base.BAGS[run_id], controller, data, arrays)
            for run_id in benchmark["validation_reports"]}
        # Laps 9-12 belong to the training capture and select only the
        # prespecified ridge strength. External r02/r03 remain untouched here.
        selection_x, selection_y, selection_states, selection_laps = [], [], [], []
        _install_residual(library, None)
        for local in range(2, len(training["indices"]) - 1):
            lap = int(lap_count[training["indices"][local]])
            if not 9 <= lap <= 12:
                continue
            state = _state_at(training, local, rigid, frames)
            control = _step_controls(training, local)
            curvature = float(training["projection"]["curvature"][local])
            stage = library.mpc_vehicle_model_step(
                ctypes.byref(state), ctypes.byref(control),
                ctypes.c_float(base.DT_S), ctypes.c_float(curvature))
            if not stage.valid:
                continue
            selection_x.append(_features(
                state, control, curvature, float(stage.body_accel_mps2)))
            selection_states.append(state)
            selection_laps.append(lap)
            next_index = int(training["indices"][local + 1])
            selection_y.append((float(rigid[next_index, 12])
                                - float(stage.next.r)) / base.DT_S)
        selection_x = np.asarray(selection_x)
        selection_y = np.asarray(selection_y)
        selection_laps = np.asarray(selection_laps, dtype=np.int32)
        baseline = {}
        for run_id, run in validation.items():
            baseline[run_id] = {
                "one_step": _teacher_forced(library, run, rigid, frames, None),
                "full_run": _rollout(
                    library, run, rigid, frames, None, lap_count, trajectory,
                    controller.lap_length_m),
            }

        gate_results = {}
        for gate_name, steering_bounds in GATE_PRESETS.items():
            train_x, train_y, train_gate, train_laps = _training_rows(
                library, training, rigid, frames, lap_count, trajectory,
                steering_bounds)
            selection_gate = np.asarray([
                _model_gates(state, steering_bounds)[0]
                for state in selection_states])
            selected = selection_gate > 0.05
            if int(selected.sum()) < 100:
                gate_results[gate_name] = {
                    "steering_bounds": steering_bounds,
                    "status": "insufficient_selection_support",
                    "selection_rows": int(selected.sum()),
                }
                continue

            alpha_scores = {}
            alpha_fits = {}
            selected_laps = selection_laps[selected]
            for alpha in ALPHAS:
                fit = _fit(train_x, train_y, train_gate, alpha)
                predicted = np.column_stack((
                    np.ones(len(selection_x)),
                    (selection_x - np.asarray(fit["feature_mean"]))
                    / np.asarray(fit["feature_scale"]))) @ np.asarray(
                        fit["coefficients"])
                applied = predicted * selection_gate
                per_lap = []
                for lap in np.unique(selected_laps):
                    mask = selected & (selection_laps == lap)
                    per_lap.append(float(np.sqrt(np.mean(
                        (applied[mask] - selection_y[mask]) ** 2))))
                alpha_scores[str(alpha)] = {
                    "per_lap_rmse_radps2": per_lap,
                    "macro_lap_rmse_radps2": float(np.mean(per_lap)),
                    "selection_rows": int(selected.sum()),
                }
                alpha_fits[str(alpha)] = fit
            best_alpha = min(ALPHAS, key=lambda value: alpha_scores[str(value)][
                "macro_lap_rmse_radps2"])
            chosen = _candidate_from_fit(alpha_fits[str(best_alpha)],
                                         steering_bounds)
            training_holdout = {
                "one_step_baseline": _teacher_forced(
                    library, training, rigid, frames, None, (9, 12)),
                "one_step_candidate": _teacher_forced(
                    library, training, rigid, frames, chosen, (9, 12)),
            }
            baseline_laps = {}
            for lap in range(9, 13):
                local_ids = np.flatnonzero(training["lap_count"] == lap)
                if len(local_ids) < 3:
                    continue
                start = int(local_ids[0])
                end = int(local_ids[-1])
                baseline_laps[lap] = _rollout(
                    library, training, rigid, frames, None, lap_count,
                    trajectory, controller.lap_length_m, start, end)

            gain_selection = {}
            for gain in (0.0, 0.25, 0.5, 0.75, 1.0):
                model = _candidate_from_fit(alpha_fits[str(best_alpha)],
                                            steering_bounds, gain)
                per_lap = {}
                for lap in baseline_laps:
                    local_ids = np.flatnonzero(training["lap_count"] == lap)
                    per_lap[lap] = _rollout(
                        library, training, rigid, frames, model, lap_count,
                        trajectory, controller.lap_length_m,
                        int(local_ids[0]), int(local_ids[-1]))
                yaw_by_lap = [value["per_state"]["yaw_rate_rps"]["rmse"]
                              for value in per_lap.values()
                              if "yaw_rate_rps" in value.get("per_state", {})]
                gain_selection[str(gain)] = {
                    "macro_lap_yaw_rmse_radps": float(np.mean(yaw_by_lap)),
                    "per_lap": {str(lap): value["per_state"]
                                for lap, value in per_lap.items()},
                    "path_nonregression_within_5pct": all(
                        (per_lap[lap].get("failure") is None
                         or (baseline_laps[lap].get("failure") is not None
                             and per_lap[lap].get("predicted_until_s", 0.0)
                             >= baseline_laps[lap].get(
                                 "predicted_until_s", 0.0)))
                        and per_lap[lap].get("per_state", {}).get(state, {}).get(
                            "rmse", float("inf"))
                        <= baseline_laps[lap].get("per_state", {}).get(
                            state, {}).get("rmse", 0.0) * 1.05
                        for lap in per_lap
                        for state in ("e_y_m", "e_psi_rad")),
                }
            eligible_gains = [float(gain) for gain in (0.0, 0.25, 0.5, 0.75, 1.0)
                              if gain_selection[str(gain)][
                                  "path_nonregression_within_5pct"]]
            selected_gain = min(
                eligible_gains,
                key=lambda gain: gain_selection[str(gain)][
                    "macro_lap_yaw_rmse_radps"]) if eligible_gains else 0.0
            chosen = _candidate_from_fit(alpha_fits[str(best_alpha)],
                                         steering_bounds, selected_gain)
            training_holdout["one_step_candidate"] = _teacher_forced(
                library, training, rigid, frames, chosen, (9, 12))
            heldout = {}
            for run_id, run in validation.items():
                heldout[run_id] = {
                    "one_step_candidate": _teacher_forced(
                        library, run, rigid, frames, chosen),
                    "full_run_candidate": _rollout(
                        library, run, rigid, frames, chosen, lap_count,
                        trajectory, controller.lap_length_m),
                }
            gate_results[gate_name] = {
                "steering_bounds": steering_bounds,
                "fit_rows": int(len(train_y)),
                "fit_supported_rows": int(np.count_nonzero(train_gate > 0.05)),
                "alpha_selection": alpha_scores,
                "selected_alpha": float(best_alpha),
                "gain_selection_laps_9_12": gain_selection,
                "selected_gain": selected_gain,
                "candidate_parameters": chosen,
                "training_laps_9_12": training_holdout,
                "validation": heldout,
            }

    result = {
        "study": "training-only low/mid-steering MPC yaw residual; exact production C transition",
        "offline_only": True,
        "simulator_was_run": False,
        "runtime_model_changed": False,
        "test_or_final_test_used": False,
        "training": {
            "run_id": TRAINING_RUN_ID,
            "bag_sha256": base._sha256(TRAINING_BAG),
            "fit_laps": "1-8",
            "selection_laps": "9-12",
            "features": list(FEATURE_NAMES),
        },
        "reference_baseline": baseline,
        "gate_sweep": gate_results,
        "provenance": {
            "benchmark_sha256": base._sha256(BENCHMARK),
            "dataset_sha256": base._sha256(dataset_path),
            "model_library_sha256": base._sha256(MODEL_LIBRARY),
            "config_sha256": base._sha256(CONFIG),
            "trajectory_sha256": base._sha256(TRAJECTORY),
            "heldout_bag_sha256": {
                name: base._sha256(path) for name, path in base.BAGS.items()},
        },
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return result


if __name__ == "__main__":
    report = run()
    print(json.dumps({
        "output": str(OUTPUT),
        "reference_baseline": report["reference_baseline"],
        "gate_sweep": {
            name: {
                "selected_alpha": value.get("selected_alpha"),
                "validation": value.get("validation"),
            } for name, value in report["gate_sweep"].items()},
    }, indent=2, sort_keys=True))
