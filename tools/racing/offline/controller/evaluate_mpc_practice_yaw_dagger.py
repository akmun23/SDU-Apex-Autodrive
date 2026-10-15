#!/usr/bin/env python3
"""DAgger-fit low/mid-steer yaw residuals and test recursive practice transfer."""

from __future__ import annotations

import ctypes
import json
from pathlib import Path
from typing import Any

import numpy as np

from tools.racing.offline.controller import evaluate_mpc_practice_yaw_residual as yaw
from tools.racing.offline.controller.production_mpc import ProductionMpc
from tools.vehicle_dynamics_learning.train_nssm import _load_dataset


OUTPUT = yaw.ANALYSIS_DIR / "practice_yaw_dagger_recursive_trace_20261012.json"
GATE_SWEEP = yaw.ANALYSIS_DIR / "practice_low_mid_steer_yaw_gain_sweep.json"
PROFILE_NAMES = ("current_broad", "focused_0p12_0p20")
ITERATIONS = 2


def _profile_bounds(name: str) -> tuple[float, float, float, float]:
    sweep = json.loads(GATE_SWEEP.read_text(encoding="utf-8"))
    return tuple(map(float, sweep["gate_sweep"][name]["steering_bounds"]))


def _lap_bounds(run: dict[str, Any], lap: int) -> tuple[int, int] | None:
    local = np.flatnonzero(run["lap_count"] == lap)
    if len(local) < 3:
        return None
    return int(local[0]), int(local[-1])


def _collect_dagger_rows(library: ctypes.CDLL, run: dict[str, Any],
                         rigid: np.ndarray, frames: np.ndarray,
                         lap_count: np.ndarray, trajectory: np.ndarray,
                         lap_length_m: float,
                         candidate: dict[str, Any],
                         steering_bounds: tuple[float, float, float, float]
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    features: list[np.ndarray] = []
    targets: list[float] = []
    gates: list[float] = []
    episode_report: dict[str, Any] = {}
    for lap in range(1, 9):
        bounds = _lap_bounds(run, lap)
        if bounds is None:
            continue
        start, end = bounds
        rows_before = len(gates)
        state = yaw._state_at(run, start, rigid, frames)
        progress = float(run["projection"]["progress"][start])
        invalid = None
        yaw._install_residual(library, None)
        for local in range(start, end):
            control = yaw._step_controls(run, local)
            curvature = float(np.interp(
                progress % lap_length_m, trajectory[:, 0], trajectory[:, 4]))
            base_stage = library.mpc_vehicle_model_step(
                ctypes.byref(state), ctypes.byref(control),
                ctypes.c_float(yaw.base.DT_S), ctypes.c_float(curvature))
            if not base_stage.valid:
                invalid = {"local_step": local - start + 1,
                           "reason": "base_transition_invalid"}
                break
            gate = yaw._model_gates(state, steering_bounds)[0]
            if gate > 0.05:
                features.append(yaw._features(
                    state, control, curvature,
                    float(base_stage.body_accel_mps2)))
                truth_index = int(run["indices"][local + 1])
                targets.append((float(rigid[truth_index, 12])
                                - float(base_stage.next.r)) / yaw.base.DT_S)
                gates.append(gate)

            yaw._install_residual(library, candidate)
            candidate_stage = library.mpc_vehicle_model_step(
                ctypes.byref(state), ctypes.byref(control),
                ctypes.c_float(yaw.base.DT_S), ctypes.c_float(curvature))
            if not candidate_stage.valid:
                invalid = {"local_step": local - start + 1,
                           "reason": "candidate_transition_invalid"}
                break
            state = candidate_stage.next
            progress += float(candidate_stage.delta_s_m)
            yaw._install_residual(library, None)
        episode_report[str(lap)] = {
            "samples": end - start,
            "dagger_rows": int(len(gates) - rows_before),
            "failure": invalid,
        }
    yaw._install_residual(library, None)
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    w = np.asarray(gates, dtype=np.float64)
    if not len(y):
        raise RuntimeError("DAgger rollouts produced no supported samples")
    return x, y, w, {
        "supported_dagger_rows": int(len(y)),
        "episode_rollouts": episode_report,
    }


def _score_training_laps(library: ctypes.CDLL, run: dict[str, Any],
                         rigid: np.ndarray, frames: np.ndarray,
                         lap_count: np.ndarray, trajectory: np.ndarray,
                         lap_length_m: float,
                         candidate: dict[str, Any]) -> dict[str, Any]:
    per_lap = {}
    for lap in range(9, 13):
        bounds = _lap_bounds(run, lap)
        if bounds is None:
            continue
        result = yaw._rollout(
            library, run, rigid, frames, candidate, lap_count,
            trajectory, lap_length_m, bounds[0], bounds[1])
        per_lap[str(lap)] = result
    return per_lap


def run() -> dict[str, Any]:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    benchmark = json.loads(yaw.BENCHMARK.read_text(encoding="utf-8"))
    if not benchmark.get("frozen"):
        raise ValueError("practice benchmark is not frozen")
    dataset_path = yaw._workspace_path(benchmark["dataset"])
    if yaw.base._sha256(dataset_path) != benchmark["dataset_sha256"]:
        raise ValueError("practice dataset hash changed")
    sweep = json.loads(GATE_SWEEP.read_text(encoding="utf-8"))
    data = _load_dataset(dataset_path)
    with np.load(dataset_path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in (
            "sample_time_ns", "frame_run_index", "simulator_rigid_state",
            "simulator_pose_xyyaw", "lap_count", "frames")}
    rigid = arrays["simulator_rigid_state"].astype(np.float64)
    frames = arrays["frames"].astype(np.float64)
    lap_count = arrays["lap_count"].astype(np.int32)
    trajectory = yaw.TRAJECTORY_ROWS
    library = ctypes.CDLL(str(yaw.MODEL_LIBRARY.resolve()))
    yaw._configure_library(library)

    with ProductionMpc(yaw.MODEL_LIBRARY, yaw.CONFIG,
                       yaw.TRAJECTORY) as controller:
        training = yaw._load_run_data(
            dataset_path, yaw.TRAINING_RUN_ID, yaw.TRAINING_BAG,
            controller, data, arrays)
        validation = {
            run_id: yaw._load_run_data(
                dataset_path,
                f"practice_unseen_model_validation_20261001_{run_id.rsplit('_', 1)[-1]}",
                yaw.base.BAGS[run_id], controller, data, arrays)
            for run_id in benchmark["validation_reports"]}
        results = {}
        for profile in PROFILE_NAMES:
            bounds = _profile_bounds(profile)
            profile_data = sweep["gate_sweep"][profile]
            alpha = float(profile_data["selected_alpha"])
            candidate = profile_data["candidate_parameters"]
            candidate["gain"] = 1.0
            initial_x, initial_y, initial_gate, _ = yaw._training_rows(
                library, training, rigid, frames, lap_count, trajectory, bounds)
            accumulated_x = [initial_x]
            accumulated_y = [initial_y]
            accumulated_gate = [initial_gate]
            iteration_reports = []
            for iteration in range(ITERATIONS + 1):
                fit = yaw._fit(
                    np.concatenate(accumulated_x),
                    np.concatenate(accumulated_y),
                    np.concatenate(accumulated_gate), alpha)
                candidate = yaw._candidate_from_fit(fit, bounds, 1.0)
                training_laps = _score_training_laps(
                    library, training, rigid, frames, lap_count,
                    trajectory, controller.lap_length_m, candidate)
                validation_runs = {}
                for run_id, run in validation.items():
                    validation_runs[run_id] = {
                        "one_step": yaw._teacher_forced(
                            library, run, rigid, frames, candidate),
                        "full_run": yaw._rollout(
                            library, run, rigid, frames, candidate,
                            lap_count, trajectory, controller.lap_length_m),
                    }
                iteration_reports.append({
                    "iteration": iteration,
                    "training_rows_total": int(sum(map(len, accumulated_y))),
                    "fit_rows_supported": fit["fit_rows"],
                    "fit_rmse_radps2": fit["one_step_fit_rmse_radps2"],
                    "training_laps_9_12_recursive": training_laps,
                    "unseen_practice": validation_runs,
                    "candidate_parameters": candidate,
                })
                if iteration == ITERATIONS:
                    break
                dagger_x, dagger_y, dagger_gate, dagger_report = (
                    _collect_dagger_rows(
                        library, training, rigid, frames, lap_count,
                        trajectory, controller.lap_length_m, candidate, bounds))
                accumulated_x.append(dagger_x)
                accumulated_y.append(dagger_y)
                accumulated_gate.append(dagger_gate)
                iteration_reports[-1]["dagger_data_collection"] = dagger_report
            results[profile] = {
                "steering_bounds_rad": bounds,
                "ridge_alpha_from_train_laps_9_12": alpha,
                "iterations": iteration_reports,
            }

    result = {
        "study": "DAgger yaw residual for production MPC motion transition",
        "offline_only": True,
        "simulator_was_run": False,
        "runtime_model_changed": False,
        "test_or_final_test_used": False,
        "training_run_id": yaw.TRAINING_RUN_ID,
        "fit_laps": "1-8; each DAgger episode starts from truth at that lap boundary",
        "selection_laps": "9-12, recursive whole-lap replay",
        "validation_runs": list(validation),
        "iterations_per_profile": ITERATIONS,
        "profiles": results,
        "provenance": {
            "dataset_sha256": yaw.base._sha256(dataset_path),
            "benchmark_sha256": yaw.base._sha256(yaw.BENCHMARK),
            "gate_sweep_sha256": yaw.base._sha256(GATE_SWEEP),
            "model_library_sha256": yaw.base._sha256(yaw.MODEL_LIBRARY),
            "config_sha256": yaw.base._sha256(yaw.CONFIG),
            "trajectory_sha256": yaw.base._sha256(yaw.TRAJECTORY),
        },
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return result


if __name__ == "__main__":
    report = run()
    print(json.dumps({
        "output": str(OUTPUT),
        "profiles": {
            name: [{
                "iteration": iteration["iteration"],
                "training_rows_total": iteration["training_rows_total"],
                "train_lap_rmse": {
                    lap: value.get("per_state", {})
                    for lap, value in iteration[
                        "training_laps_9_12_recursive"].items()},
                "validation": {
                    run: value["full_run"].get("per_state", {})
                    for run, value in iteration["unseen_practice"].items()},
            } for iteration in profile["iterations"]]
            for name, profile in report["profiles"].items()},
    }, indent=2, sort_keys=True))
