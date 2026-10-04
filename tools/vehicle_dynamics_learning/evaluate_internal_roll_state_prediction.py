#!/usr/bin/env python3
"""Test whether roll/roll-rate can be propagated as an internal plant state.

The roll dynamics are fitted only on the selected plant's training runs. In
validation roll forecasts, measured roll is used only to initialize each
forecast. After that, the model consumes its own roll and roll-rate predictions.
True body motion/commands are supplied as conditioning signals solely to
isolate roll-subsystem predictability; this is not a full-plant rollout and
cannot establish offline-simulator accuracy.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tools.vehicle_dynamics_learning.diagnose_rigid_acceleration_sensor_residual_value import (
    _transition_rows,
)
from tools.vehicle_dynamics_learning.run_truncated_history_transition import ROOT
from tools.vehicle_dynamics_learning.rigid_acceleration_history_plant import (
    RigidAccelerationHistoryTransition,
)
from tools.vehicle_dynamics_learning.run_history_context_sufficiency import (
    _batch_arrays,
    _normalization,
)
from tools.vehicle_dynamics_learning.train_augmented_state_space_plant import (
    DEFAULT_DYNAMIC,
    DEFAULT_PRACTICE,
    _collect_horizon_windows,
    _load_data,
    _select_eval_windows,
    _training_windows_and_stats,
    sha256_file,
)


DT_S = 0.025
CONTEXT_STEPS = 80
HORIZONS = {"2s": 80, "5s": 200}
OUTPUT = (
    ROOT / "live_runs/derived_dynamics_learning_20260928"
    / "full_modeling_reset_20261001"
    / "replacement_offline_sim_raceline_20261003"
    / "full_throttle_domain_v1/next_phase_after_2129427"
    / "history_context_sufficiency_v1"
    / "internal_roll_state_prediction_plant_driven_20261004.json")
CHECKPOINT = (
    ROOT / "live_runs/derived_dynamics_learning_20260928"
    / "full_modeling_reset_20261001"
    / "replacement_offline_sim_raceline_20261003"
    / "full_throttle_domain_v1/next_phase_after_2129427"
    / "history_context_sufficiency_v1"
    / "rigid_acceleration_history_direct_supervision_5s_v1/checkpoint.pt")


def _features(rows: dict[str, np.ndarray]) -> np.ndarray:
    base = rows["base_features"]
    return np.column_stack((
        rows["roll_features"], base[:, :5], base[:, 5:7], rows["target"][:, 1]
    )).astype(np.float64)


def _balanced_fit(rows: dict[str, dict[str, np.ndarray]]):
    from sklearn.linear_model import Ridge

    train_runs = sorted(rows)
    if len(train_runs) < 5:
        raise RuntimeError("roll model requires at least five training runs")
    features = np.concatenate([_features(rows[run]) for run in train_runs])
    roll_state = np.concatenate([rows[run]["roll_features"] for run in train_runs])
    roll_next = np.concatenate([rows[run]["roll_target"] for run in train_runs])
    alpha = (roll_next[:, 1] - roll_state[:, 1]) / DT_S
    weights = np.concatenate([
        np.full(len(rows[run]["target"]), 1.0 / len(rows[run]["target"]))
        for run in train_runs])
    weights *= len(train_runs)
    mean = np.average(features, axis=0, weights=weights)
    variance = np.average((features - mean) ** 2, axis=0, weights=weights)
    scale = np.maximum(np.sqrt(variance), 1e-5)
    alpha_mean = float(np.average(alpha, weights=weights))
    alpha_scale = max(float(np.sqrt(np.average(
        (alpha - alpha_mean) ** 2, weights=weights))), 1e-3)
    model = Ridge(alpha=1.0, fit_intercept=True)
    model.fit((features - mean) / scale, (alpha - alpha_mean) / alpha_scale,
              sample_weight=weights)
    physical_coefficients = alpha_scale * model.coef_ / scale
    physical_intercept = (alpha_mean + alpha_scale * model.intercept_
                          - float(physical_coefficients @ mean))
    roll_beta, rate_beta = physical_coefficients[:2]
    discrete_roll_matrix = np.asarray((
        (1.0 + 0.5 * DT_S ** 2 * roll_beta,
         DT_S + 0.5 * DT_S ** 2 * rate_beta),
        (DT_S * roll_beta, 1.0 + DT_S * rate_beta),
    ), dtype=np.float64)
    return {
        "model": model,
        "feature_mean": mean,
        "feature_scale": scale,
        "target_mean": alpha_mean,
        "target_scale": alpha_scale,
        "train_runs": train_runs,
        "train_row_count": int(len(alpha)),
        "feature_names": ["roll_rad", "roll_rate_radps", "u_mps",
                           "v_rear_mps", "yaw_rate_rps", "steer_feedback",
                           "throttle_feedback", "steering_command",
                           "throttle_command", "interval_lateral_accel_mps2"],
        "physical_coefficients_roll_accel": physical_coefficients.tolist(),
        "physical_intercept_roll_accel": float(physical_intercept),
        "zero_input_discrete_roll_rate_spectral_radius": float(np.max(
            np.abs(np.linalg.eigvals(discrete_roll_matrix)))),
    }


def _predict_alpha(fitted, feature: np.ndarray) -> float:
    normalized = (np.asarray(feature, dtype=np.float64)
                  - fitted["feature_mean"]) / fitted["feature_scale"]
    normalized_prediction = float(fitted["model"].predict(normalized[None, :])[0])
    return fitted["target_mean"] + fitted["target_scale"] * normalized_prediction


def _predict_alpha_batch(fitted, features: np.ndarray) -> np.ndarray:
    normalized = (np.asarray(features, dtype=np.float64)
                  - fitted["feature_mean"]) / fitted["feature_scale"]
    prediction = fitted["model"].predict(normalized)
    return fitted["target_mean"] + fitted["target_scale"] * prediction


def _feature(roll: float, rate: float, body: np.ndarray, frame: np.ndarray,
             body_acceleration: np.ndarray) -> np.ndarray:
    # Layout is exactly matched to _features(): roll/rate, body (u/v/r),
    # actuator feedback, command, and the model's current interval a_y.
    return np.asarray((roll, rate, *body, *frame[3:5], *frame[7:9],
                       body_acceleration[1]), dtype=np.float64)


def _input_body_accelerations(capture, begin: int, end: int) -> np.ndarray:
    from tools.vehicle_dynamics_learning.rigid_acceleration_history_plant import (
        midpoint_acceleration_from_transition,
    )
    return np.stack([
        midpoint_acceleration_from_transition(capture.body[index],
                                              capture.body[index + 1], DT_S)
        for index in range(begin, end - 1)
    ])


def _recursive_scores(capture, attitude: np.ndarray, valid: np.ndarray,
                      split_name: str, fitted, horizon: int,
                      max_windows_per_run: int = 32
                      ) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    starts_by_run: dict[str, list[tuple[int, int]]] = {}
    for sequence_index, ((begin_raw, end_raw), run_raw) in enumerate(
            zip(capture.bounds, capture.sequence_run)):
        run_index = int(run_raw)
        if str(capture.splits[run_index]) != split_name:
            continue
        begin, end = int(begin_raw), int(end_raw)
        first = begin + CONTEXT_STEPS - 1
        last = end - horizon - 1
        if first > last:
            continue
        starts = list(range(first, last + 1, horizon))
        if len(starts) > max_windows_per_run:
            starts = [starts[int(index)] for index in np.linspace(
                0, len(starts) - 1, max_windows_per_run, dtype=np.int64)]
        run_id = str(capture.run_ids[run_index])
        starts_by_run.setdefault(run_id, []).extend(
            (sequence_index, start) for start in starts)

    for run_id, starts in starts_by_run.items():
        forecast_roll_errors, forecast_rate_errors = [], []
        frozen_roll_errors, frozen_rate_errors = [], []
        for sequence_index, start in starts:
            begin, end = map(int, capture.bounds[sequence_index])
            if (not valid[start:start + horizon + 1].all()
                    or not np.isfinite(capture.body[
                        start:start + horizon + 1]).all()
                    or not np.isfinite(capture.frames[
                        start:start + horizon]).all()):
                continue
            predicted_roll = float(attitude[start, 0])
            predicted_rate = float(attitude[start, 2])
            initial_roll = predicted_roll
            predicted_rolls, predicted_rates = [], []
            for absolute in range(start, start + horizon):
                body_acceleration = _input_body_accelerations(
                    capture, absolute, absolute + 2)[0]
                features = _feature(predicted_roll, predicted_rate,
                                    capture.body[absolute],
                                    capture.frames[absolute], body_acceleration)
                roll_acceleration = _predict_alpha(fitted, features)
                next_rate = predicted_rate + DT_S * roll_acceleration
                next_roll = predicted_roll + 0.5 * DT_S * (
                    predicted_rate + next_rate)
                predicted_roll, predicted_rate = next_roll, next_rate
                predicted_rolls.append(predicted_roll)
                predicted_rates.append(predicted_rate)
            truth = attitude[start + 1:start + horizon + 1]
            pred_roll = np.asarray(predicted_rolls)
            pred_rate = np.asarray(predicted_rates)
            frozen_roll = np.full(horizon, initial_roll)
            frozen_rate = np.zeros(horizon)
            roll_error = np.arctan2(np.sin(pred_roll - truth[:, 0]),
                                    np.cos(pred_roll - truth[:, 0]))
            frozen_error = np.arctan2(np.sin(frozen_roll - truth[:, 0]),
                                      np.cos(frozen_roll - truth[:, 0]))
            forecast_roll_errors.append(roll_error ** 2)
            forecast_rate_errors.append((pred_rate - truth[:, 2]) ** 2)
            frozen_roll_errors.append(frozen_error ** 2)
            frozen_rate_errors.append((frozen_rate - truth[:, 2]) ** 2)
        if not forecast_roll_errors:
            continue
        output[run_id] = {
            "independent_forecast_windows": len(forecast_roll_errors),
            "roll_rmse_rad": float(np.sqrt(np.mean(forecast_roll_errors))),
            "roll_rate_rmse_radps": float(np.sqrt(np.mean(forecast_rate_errors))),
            "frozen_initial_roll_rmse_rad": float(np.sqrt(np.mean(frozen_roll_errors))),
            "zero_roll_rate_rmse_radps": float(np.sqrt(np.mean(frozen_rate_errors))),
        }
    return output


def _recursive_scores_with_plant(data, capture_index: int, split_name: str,
                                 attitude: np.ndarray, valid: np.ndarray,
                                 fitted, plant, norm_np, config,
                                 device, horizon: int,
                                 max_starts_per_run: int = 16
                                 ) -> dict[str, dict[str, Any]]:
    """Propagate roll while the frozen plant supplies its own body motion."""
    from tools.vehicle_dynamics_learning.run_history_context_sufficiency import (
        CONTEXT_STEPS as MODEL_CONTEXT_STEPS,
        advance_context,
    )
    from tools.vehicle_dynamics_learning.run_reencoded_history_transition import (
        _wrap,
    )

    by_run = _collect_horizon_windows(
        data.captures, data.raw_sources, data.poses, {split_name}, horizon,
        capture_indices={capture_index})
    by_run = _select_eval_windows(by_run, max_starts_per_run)
    capture = data.captures[capture_index]
    score: dict[str, dict[str, Any]] = {}
    plant.eval()
    history_mean = np.asarray(config.history_mean[:7], dtype=np.float32)
    history_scale = np.asarray(config.history_scale[:7], dtype=np.float32)
    with torch.no_grad():
        for run_id, refs in sorted(by_run.items()):
            retained = []
            for ref in refs:
                _, sequence_index, source_row = ref
                begin = int(capture.bounds[sequence_index, 0]) + int(source_row)
                if valid[begin:begin + horizon + 1].all():
                    retained.append(ref)
            if not retained:
                continue
            arrays = _batch_arrays(
                data, retained, horizon, MODEL_CONTEXT_STEPS["2.0s"],
                norm_np, config, device)
            history, mask, state, _, commands, _, _ = arrays
            starts = [int(capture.bounds[ref[1], 0]) + int(ref[2])
                      for ref in retained]
            roll = attitude[starts, 0].astype(np.float64)
            rate = attitude[starts, 2].astype(np.float64)
            predicted_rolls, predicted_rates = [], []
            for step in range(horizon):
                delta, acceleration = plant.forward_with_acceleration(
                    history, mask, state, commands[:, step])
                next_state = state + delta
                physical_state = (state.cpu().numpy() * norm_np["state_scale"]
                                  + norm_np["state_mean"])
                command_physical = (
                    commands[:, step].cpu().numpy() * norm_np["command_scale"]
                    + norm_np["command_mean"])
                acceleration_np = acceleration.cpu().numpy()
                feature_batch = np.column_stack((
                    roll, rate, physical_state[:, :5], command_physical,
                    acceleration_np[:, 1]))
                roll_acceleration = _predict_alpha_batch(fitted, feature_batch)
                next_rate = rate + DT_S * roll_acceleration
                next_roll = roll + 0.5 * DT_S * (rate + next_rate)
                predicted_rolls.append(next_roll)
                predicted_rates.append(next_rate)

                if step + 1 < horizon:
                    next_command = commands[:, step + 1]
                    next_physical_state = (
                        next_state.cpu().numpy() * norm_np["state_scale"]
                        + norm_np["state_mean"])
                    next_command_physical = (
                        next_command.cpu().numpy() * norm_np["command_scale"]
                        + norm_np["command_mean"])
                    raw_history_row = np.column_stack((
                        next_physical_state, next_command_physical))
                    normalized_row = ((raw_history_row - history_mean)
                                      / history_scale)
                    row_t = torch.as_tensor(
                        normalized_row, dtype=torch.float32, device=device)
                    history, mask = advance_context(
                        history, mask, row_t, MODEL_CONTEXT_STEPS["2.0s"])
                state = next_state
                roll, rate = next_roll, next_rate

            truth = np.stack([
                attitude[begin + 1:begin + horizon + 1][:, [0, 2]]
                for begin in starts])
            pred_roll = np.stack(predicted_rolls, axis=1)
            pred_rate = np.stack(predicted_rates, axis=1)
            roll_error = _wrap(pred_roll - truth[:, :, 0])
            rate_error = pred_rate - truth[:, :, 1]
            frozen_error = _wrap(attitude[starts, 0][:, None]
                                 - truth[:, :, 0])
            zero_rate_error = -truth[:, :, 1]
            score[run_id] = {
                "independent_forecast_starts": int(len(starts)),
                "roll_rmse_rad": float(np.mean(np.sqrt(
                    np.mean(roll_error ** 2, axis=1)))),
                "roll_rate_rmse_radps": float(np.mean(np.sqrt(
                    np.mean(rate_error ** 2, axis=1)))),
                "frozen_initial_roll_rmse_rad": float(np.mean(np.sqrt(
                    np.mean(frozen_error ** 2, axis=1)))),
                "zero_roll_rate_rmse_radps": float(np.mean(np.sqrt(
                    np.mean(zero_rate_error ** 2, axis=1)))),
            }
    return score


def _one_step_scores(fitted, rows_by_run):
    output = {}
    for run_id, rows in sorted(rows_by_run.items()):
        predicted = np.asarray([
            _predict_alpha(fitted, feature)
            for feature in _features(rows)], dtype=np.float64)
        roll_alpha = ((rows["roll_target"][:, 1]
                       - rows["roll_features"][:, 1]) / DT_S)
        output[run_id] = {
            "roll_acceleration_rmse_radps2": float(np.sqrt(np.mean(
                (predicted - roll_alpha) ** 2))),
            "roll_acceleration_truth_rmse_radps2": float(np.sqrt(np.mean(
                roll_alpha ** 2))),
            "transition_count": int(len(predicted)),
        }
    return output


def run(output: Path = OUTPUT) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    data, wp20 = _load_data()
    config, _, _ = _training_windows_and_stats(data, wp20)
    train_rows = _transition_rows(
        data, 0, "train", DEFAULT_DYNAMIC, config,
        allowed_run_ids=set(data.training_runs))
    dynamic_rows = _transition_rows(
        data, 0, "validation", DEFAULT_DYNAMIC, config)
    practice_rows = _transition_rows(
        data, 1, "unseen_practice", DEFAULT_PRACTICE, config)
    fitted = _balanced_fit(train_rows)

    def attitude_arrays(path: Path):
        with np.load(path, allow_pickle=False) as archive:
            attitude = np.asarray(archive["imu_attitude_frames"],
                                  dtype=np.float32)
            valid = np.asarray(archive["imu_attitude_valid"], dtype=bool)
        return attitude, valid

    dynamic_attitude, dynamic_valid = attitude_arrays(DEFAULT_DYNAMIC)
    practice_attitude, practice_valid = attitude_arrays(DEFAULT_PRACTICE)
    dynamic_capture, practice_capture = data.captures[0], data.captures[1]
    saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    metadata = saved["metadata"]
    norm_np = _normalization(data, config)
    plant = RigidAccelerationHistoryTransition(
        norm_np["state_mean"], norm_np["state_scale"],
        np.asarray(metadata["acceleration_mean_train_only"], dtype=np.float32),
        np.asarray(metadata["acceleration_scale_train_only"], dtype=np.float32),
        norm_np["delta_mean"][3:], norm_np["delta_scale"][3:],
        dt_s=float(metadata["dt_s"]),
        rear_axle_to_com_x_m=float(metadata["rear_axle_to_com_x_m"])).to("cpu")
    plant.load_state_dict(saved["state_dict"], strict=True)
    plant.eval()
    recursive = {}
    for horizon_name, horizon in HORIZONS.items():
        recursive[horizon_name] = {
            "dynamic_validation": _recursive_scores(
                dynamic_capture, dynamic_attitude, dynamic_valid,
                "validation", fitted, horizon),
            "practice_diagnostic": _recursive_scores(
                practice_capture, practice_attitude, practice_valid,
                "unseen_practice", fitted, horizon),
        }
    plant_driven_roll = {}
    for horizon_name, horizon in (("2s", 80), ("5s", 200), ("10s", 400)):
        plant_driven_roll[horizon_name] = {
            "dynamic_validation": _recursive_scores_with_plant(
                data, 0, "validation", dynamic_attitude, dynamic_valid,
                fitted, plant, norm_np, config, torch.device("cpu"), horizon),
            "practice_diagnostic": (
                _recursive_scores_with_plant(
                    data, 1, "unseen_practice", practice_attitude,
                    practice_valid, fitted, plant, norm_np, config,
                    torch.device("cpu"), horizon)
                if horizon <= 200 else {}),
        }
    report = {
        "study": "internally propagated roll state, conditioned on body/command truth",
        "dynamic_data_sha256": sha256_file(DEFAULT_DYNAMIC),
        "practice_data_sha256": sha256_file(DEFAULT_PRACTICE),
        "training_run_ids": fitted["train_runs"],
        "training_transition_count": fitted["train_row_count"],
        "roll_dynamics_fit": {
            "feature_names": fitted["feature_names"],
            "physical_coefficients_roll_accel":
                fitted["physical_coefficients_roll_accel"],
            "physical_intercept_roll_accel":
                fitted["physical_intercept_roll_accel"],
            "zero_input_discrete_roll_rate_spectral_radius":
                fitted["zero_input_discrete_roll_rate_spectral_radius"],
        },
        "cadence_hz": 40.0,
        "roll_state_update": "roll_next=roll+0.5*dt*(rate+rate_next); rate_next=rate+dt*predicted_roll_acceleration",
        "roll_acceleration_inputs": [
            "own predicted roll and roll rate",
            "body u/v/yaw-rate, actuator feedback and command at current step",
            "current interval lateral acceleration (oracle truth in this subsystem test; must be replaced by the plant prediction in a coupled test)"],
        "one_step_teacher_forced": {
            "dynamic_validation": _one_step_scores(fitted, dynamic_rows),
            "practice_diagnostic": _one_step_scores(fitted, practice_rows),
        },
        "recursive_roll_state_only": recursive,
        "recursive_roll_state_driven_by_frozen_plant": plant_driven_roll,
        "frozen_body_plant_checkpoint_sha256": sha256_file(CHECKPOINT),
        "future_measured_roll_used_after_initialization": False,
        "future_body_or_command_truth_used": True,
        "full_plant_rollout": False,
        "production_integration": False,
        "interpretation_limit": (
            "This intentionally isolates the roll subsystem by supplying true "
            "body state, commands, and interval lateral acceleration at each "
            "step in the roll-only section. The plant-driven section instead "
            "uses the frozen body's own recursive predictions, but roll does "
            "not feed back into that body's acceleration. Neither section proves "
            "that roll improves body motion; they only gate whether to include "
            "internal roll propagation in a coupled candidate."),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    report = run()
    print(json.dumps({
        "output": OUTPUT.relative_to(ROOT).as_posix(),
        "one_step_dynamic": report["one_step_teacher_forced"]["dynamic_validation"],
        "recursive_dynamic": report["recursive_roll_state_only"],
    }, indent=2))
