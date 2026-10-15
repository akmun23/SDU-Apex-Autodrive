#!/usr/bin/env python3
"""Compare production MPC plant rollouts from truth and paired odom states.

The frozen practice benchmark, recorded controls, map pose, and future truth
are shared across all initializations. Only the initial (u, v, r) differs:
simulator truth, baseline odometry, or corrected odometry. This isolates the
effect of odometry's motion-state initialization from map-pose/localization
error and from changes to the production model.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from tools.racing.offline.controller.score_production_mpc_model import (
    DT_S,
    HORIZONS_S,
    STATE_NAMES,
    MpcModelControl,
    MpcModelState,
    _metric,
    _project_run,
    _read_commands,
    _recorded_command_arrays,
)
from tools.racing.offline.controller.production_mpc import ProductionMpc
from tools.vehicle_dynamics_learning.build_practice_transfer_benchmark import (
    _run_rows,
)
from tools.vehicle_dynamics_learning.train_direct_sequence_teacher import COM_X_M
from tools.vehicle_dynamics_learning.train_nssm import _load_dataset


REPO_ROOT = Path(__file__).resolve().parents[4]
ODOM_ALIGNMENT_LIMIT_NS = 3_000_000


class MpcStageResult(ctypes.Structure):
    _fields_ = [("next", MpcModelState), ("delta_s_m", ctypes.c_float),
                ("body_accel_mps2", ctypes.c_float),
                ("branch_flags", ctypes.c_uint), ("valid", ctypes.c_int)]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _workspace_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_file():
        return path.resolve()
    if path.is_absolute():
        try:
            relative = path.relative_to(Path("/workspace"))
        except ValueError:
            return path
        candidate = REPO_ROOT / relative
        if candidate.is_file():
            return candidate.resolve()
    return path


def _load_odom_archive(path: Path, sample_stamps: np.ndarray
                       ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    archive = np.load(path, allow_pickle=False)
    required = {"source_stamp_ns", "baseline", "candidate"}
    if not required.issubset(archive.files):
        raise ValueError(f"paired odom archive missing {sorted(required - set(archive.files))}: {path}")
    stamps = archive["source_stamp_ns"].astype(np.int64, copy=False)
    if (stamps.ndim != 1 or len(stamps) < 2 or np.any(np.diff(stamps) <= 0)
            or archive["baseline"].shape != (len(stamps), 3)
            or archive["candidate"].shape != (len(stamps), 3)):
        raise ValueError(f"invalid paired odom archive dimensions or stamps: {path}")

    insertion = np.searchsorted(stamps, sample_stamps)
    insertion = np.clip(insertion, 1, len(stamps) - 1)
    nearest = np.where(
        np.abs(stamps[insertion] - sample_stamps)
        < np.abs(stamps[insertion - 1] - sample_stamps),
        insertion, insertion - 1)
    delta = stamps[nearest] - sample_stamps
    if (np.any(np.abs(delta) > ODOM_ALIGNMENT_LIMIT_NS)
            or np.any(np.diff(nearest) <= 0)):
        raise ValueError(
            f"odometry state timestamps do not uniquely align to dataset within "
            f"{ODOM_ALIGNMENT_LIMIT_NS / 1e6:g} ms: {path}")
    aligned = {
        "baseline_odom": archive["baseline"][nearest].astype(np.float64),
        "corrected_odom": archive["candidate"][nearest].astype(np.float64),
    }
    if not all(np.isfinite(value).all() for value in aligned.values()):
        raise ValueError(f"non-finite odometry state in {path}")
    return aligned, {
        "archive": str(path.resolve()),
        "archive_sha256": _sha256(path),
        "sample_count": int(len(sample_stamps)),
        "unique_monotonic_matches": True,
        "nearest_signed_delta_ms_quantiles": np.quantile(
            delta / 1e6, [0, 0.01, 0.5, 0.99, 1]).tolist(),
        "nearest_abs_delta_ms_max": float(np.max(np.abs(delta)) / 1e6),
    }


def _run_rollout(model_library: ctypes.CDLL, initial: MpcModelState,
                 progress: float, local_index: int,
                 steering: np.ndarray, target_speed: np.ndarray,
                 trajectory: np.ndarray, lap_length: float,
                 horizon_steps: int) -> np.ndarray | None:
    state = initial
    predictions = []
    for step in range(horizon_steps):
        command_index = local_index + step
        steer_rate = (steering[command_index + 1] - steering[command_index]) / DT_S
        speed_rate = (target_speed[command_index + 1] - target_speed[command_index]) / DT_S
        control = MpcModelControl(float(steer_rate), float(speed_rate))
        curvature = float(np.interp(
            progress % lap_length, trajectory[:, 0], trajectory[:, 4]))
        stage = model_library.mpc_vehicle_model_step(
            ctypes.byref(state), ctypes.byref(control),
            ctypes.c_float(DT_S), ctypes.c_float(curvature))
        if not stage.valid or not math.isfinite(stage.delta_s_m):
            return None
        state = stage.next
        progress += float(stage.delta_s_m)
        predictions.append((
            float(state.u), float(state.v + COM_X_M * state.r),
            float(state.r), float(state.e_y), float(state.e_psi),
            float(state.actual_steering_angle), float(state.target_speed),
            float(state.steering_command),
        ))
    return np.asarray(predictions, dtype=np.float64)


def compare(benchmark_path: Path, library_path: Path, config_path: Path,
            trajectory_path: Path, bag_paths: dict[str, Path],
            odom_archives: dict[str, Path], output_path: Path
            ) -> dict[str, Any]:
    benchmark_path, library_path, config_path, trajectory_path, output_path = map(
        Path.resolve, (benchmark_path, library_path, config_path,
                      trajectory_path, output_path))
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite MPC/odom report: {output_path}")
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    dataset_path = _workspace_path(benchmark["dataset"])
    if (benchmark.get("benchmark_id") != "practice_transfer_benchmark_v1"
            or benchmark.get("frozen") is not True
            or _sha256(dataset_path) != benchmark["dataset_sha256"]):
        raise ValueError("practice benchmark is not frozen or dataset hash changed")
    if set(bag_paths) != set(benchmark["validation_reports"]):
        raise ValueError("bag inputs must exactly match frozen validation runs")
    if set(odom_archives) != {"r02", "r03"}:
        raise ValueError("paired odom archives must be provided for r02 and r03")
    trajectory = np.loadtxt(trajectory_path, delimiter=",", comments="#",
                            dtype=np.float64)
    for run_id, bag_path in bag_paths.items():
        bag_path = bag_path.resolve()
        if _sha256(bag_path) != benchmark["validation_reports"][run_id]["bag_sha256"]:
            raise ValueError(f"frozen validation bag hash mismatch: {run_id}")

    data = _load_dataset(dataset_path)
    with np.load(dataset_path, allow_pickle=False) as archive:
        data["sample_time_ns"] = archive["sample_time_ns"].astype(np.int64, copy=False)
    window_data = {**data, "sequence_bounds": data["bounds"],
                   "sequence_run_index": data["seq_run"]}
    targets = np.column_stack((
        data["simulator_rigid_state"][:, 7],
        data["simulator_rigid_state"][:, 8],
        data["simulator_rigid_state"][:, 12],
        data["frames"][:, 3:7],
    )).astype(np.float64)

    run_index_by_name = {str(name): index
                         for index, name in enumerate(data["run_ids"])}
    needed_run_ids = {
        f"practice_unseen_model_validation_20261001_{key.rsplit('_', 1)[-1]}"
        for key in benchmark["validation_reports"]}
    if not needed_run_ids.issubset(run_index_by_name):
        raise ValueError("frozen dataset lacks one or more validation runs")

    commands_by_run = {
        run_id: _read_commands(path.resolve()) for run_id, path in bag_paths.items()}
    odom_by_run: dict[str, dict[str, np.ndarray]] = {}
    alignment_report: dict[str, Any] = {}
    global_indices_by_run: dict[str, np.ndarray] = {}
    for suffix in ("r02", "r03"):
        dataset_run_id = f"practice_unseen_model_validation_20261001_{suffix}"
        run_index = run_index_by_name[dataset_run_id]
        indices = _run_rows(window_data, run_index)
        global_indices_by_run[dataset_run_id] = indices
        aligned, quality = _load_odom_archive(
            odom_archives[suffix].resolve(), data["sample_time_ns"][indices])
        odom_by_run[dataset_run_id] = aligned
        alignment_report[dataset_run_id] = quality

    model_library = ctypes.CDLL(str(library_path))
    model_library.mpc_vehicle_model_step.argtypes = [
        ctypes.POINTER(MpcModelState), ctypes.POINTER(MpcModelControl),
        ctypes.c_float, ctypes.c_float]
    model_library.mpc_vehicle_model_step.restype = MpcStageResult

    run_names = ("simulator_truth", "baseline_odom", "corrected_odom")
    errors: dict[str, dict[str, dict[str, list[np.ndarray]]]] = {
        mode: defaultdict(lambda: defaultdict(list)) for mode in run_names}
    stepwise_errors: dict[str, dict[str, list[np.ndarray]]] = {
        mode: defaultdict(list) for mode in run_names}
    first_step_category_errors: dict[
        str, dict[str, dict[str, list[np.ndarray]]]] = {
            mode: defaultdict(lambda: defaultdict(list)) for mode in run_names}
    init_errors: dict[str, dict[str, list[np.ndarray]]] = {
        mode: defaultdict(list) for mode in run_names if mode != "simulator_truth"}
    failures: dict[str, list[dict[str, Any]]] = {mode: [] for mode in run_names}
    window_count = 0
    scored_starts = []

    with ProductionMpc(library_path, config_path, trajectory_path) as controller:
        lap_length = controller.lap_length_m
        run_projection: dict[str, dict[str, np.ndarray]] = {}
        run_controls: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for report_run_id in benchmark["validation_reports"]:
            suffix = report_run_id.rsplit("_", 1)[-1]
            dataset_run_id = f"practice_unseen_model_validation_20261001_{suffix}"
            run_index = run_index_by_name[dataset_run_id]
            indices = global_indices_by_run[dataset_run_id]
            projection = _project_run(
                controller, trajectory, lap_length,
                data["simulator_pose_xyyaw"][indices])
            run_projection[dataset_run_id] = dict(zip(
                ("progress", "e_y", "e_psi", "curvature", "distance"),
                projection))
            run_controls[dataset_run_id] = _recorded_command_arrays(
                data, run_index, commands_by_run[report_run_id], indices)

        max_steps = round(max(HORIZONS_S) / DT_S)
        for window in benchmark["windows"]:
            dataset_run_id = str(window["run_id"])
            indices = global_indices_by_run[dataset_run_id]
            global_start = int(window["global_start_index"])
            local_index = int(np.searchsorted(indices, global_start))
            if local_index >= len(indices) or indices[local_index] != global_start:
                raise ValueError("frozen common start is not in its continuous run")
            if local_index < 2 or local_index + max_steps >= len(indices):
                raise ValueError("frozen common start lacks context/future samples")
            projection = run_projection[dataset_run_id]
            steering, target_speed = run_controls[dataset_run_id]
            values = targets[indices]
            frames = data["frames"][indices]
            truth = np.asarray((values[local_index, 0],
                                values[local_index, 1] - COM_X_M * values[local_index, 2],
                                values[local_index, 2]))
            odom = odom_by_run[dataset_run_id]
            initial_states = {
                "simulator_truth": truth,
                "baseline_odom": odom["baseline_odom"][local_index],
                "corrected_odom": odom["corrected_odom"][local_index],
            }
            for mode, initial_motion in initial_states.items():
                if mode != "simulator_truth":
                    init_errors[mode][dataset_run_id].append(initial_motion - truth)
                state = MpcModelState(
                    float(projection["e_y"][local_index]),
                    float(projection["e_psi"][local_index]),
                    float(initial_motion[0]), float(initial_motion[1]),
                    float(initial_motion[2]), float(target_speed[local_index]),
                    float(steering[local_index]),
                    float(steering[local_index - 1]),
                    float(steering[local_index - 2]),
                    float(frames[local_index, 3]),
                )
                predicted = _run_rollout(
                    model_library, state, float(projection["progress"][local_index]),
                    local_index, steering, target_speed,
                    trajectory, lap_length, max_steps)
                if predicted is None:
                    failures[mode].append({
                        "run_id": dataset_run_id,
                        "packet_sequence": int(window["packet_sequence"]),
                    })
                    continue
                full_truth = np.column_stack((
                    values[local_index + 1:local_index + 1 + max_steps, 0],
                    values[local_index + 1:local_index + 1 + max_steps, 1],
                    values[local_index + 1:local_index + 1 + max_steps, 2],
                    projection["e_y"][local_index + 1:local_index + 1 + max_steps],
                    projection["e_psi"][local_index + 1:local_index + 1 + max_steps],
                    frames[local_index + 1:local_index + 1 + max_steps, 3],
                    target_speed[local_index + 1:local_index + 1 + max_steps],
                    steering[local_index + 1:local_index + 1 + max_steps],
                ))
                full_error = predicted - full_truth
                full_error[:, 4] = np.arctan2(
                    np.sin(full_error[:, 4]), np.cos(full_error[:, 4]))
                stepwise_errors[mode][dataset_run_id].append(full_error)
                for category in window.get("categories", []):
                    first_step_category_errors[mode][category][dataset_run_id].append(
                        full_error[0])
                for horizon in HORIZONS_S:
                    steps = round(horizon / DT_S)
                    errors[mode][dataset_run_id][f"{horizon:g}s"].append(
                        full_error[:steps])
            window_count += 1
            scored_starts.append({"run_id": dataset_run_id,
                                  "packet_sequence": int(window["packet_sequence"])})

    report: dict[str, Any] = {}
    for mode in run_names:
        report[mode] = {"by_run": {}, "macro_run_mean_rmse": {}}
        for horizon in HORIZONS_S:
            key = f"{horizon:g}s"
            per_run: dict[str, dict[str, float]] = {}
            for run_id, horizons in errors[mode].items():
                matrices = horizons.get(key, [])
                if matrices:
                    # Preserve the established window-macro scoring contract.
                    per_run[run_id] = {
                        name: float(np.mean([_metric(matrix[:, channel])["rmse"]
                                             for matrix in matrices]))
                        for channel, name in enumerate(STATE_NAMES)}
            report[mode]["by_run"][key] = per_run
            names = [name for name in STATE_NAMES if per_run]
            report[mode]["macro_run_mean_rmse"][key] = {
                name: float(np.mean([per_run[run_id][name] for run_id in per_run]))
                for name in names}

    init_report = {}
    for mode, runs in init_errors.items():
        init_report[mode] = {}
        for run_id, rows in runs.items():
            matrix = np.asarray(rows)
            init_report[mode][run_id] = {
                "u_rmse_mps": float(np.sqrt(np.mean(matrix[:, 0] ** 2))),
                "v_rmse_mps": float(np.sqrt(np.mean(matrix[:, 1] ** 2))),
                "yaw_rate_rmse_radps": float(np.sqrt(np.mean(matrix[:, 2] ** 2))),
                "u_bias_mps": float(np.mean(matrix[:, 0])),
                "v_bias_mps": float(np.mean(matrix[:, 1])),
                "yaw_rate_bias_radps": float(np.mean(matrix[:, 2])),
                "start_count": int(len(matrix)),
            }

    stepwise_report: dict[str, Any] = {}
    for mode, runs in stepwise_errors.items():
        stepwise_report[mode] = {}
        for run_id, rollouts in runs.items():
            stacked = np.stack(rollouts)
            stepwise_report[mode][run_id] = [
                {
                    "time_s": float((step + 1) * DT_S),
                    "state": {
                        name: {
                            "rmse": float(np.sqrt(np.mean(
                                stacked[:, step, channel] ** 2))),
                            "bias": float(np.mean(stacked[:, step, channel])),
                            "p95_abs": float(np.quantile(
                                np.abs(stacked[:, step, channel]), 0.95)),
                        }
                        for channel, name in enumerate(STATE_NAMES)
                    },
                }
                for step in range(stacked.shape[1])]

    category_report: dict[str, Any] = {}
    for mode, categories in first_step_category_errors.items():
        category_report[mode] = {}
        for category, runs in categories.items():
            category_report[mode][category] = {}
            for run_id, rows in runs.items():
                matrix = np.stack(rows)
                category_report[mode][category][run_id] = {
                    "samples": int(len(matrix)),
                    "state_rmse": {
                        name: float(np.sqrt(np.mean(matrix[:, channel] ** 2)))
                        for channel, name in enumerate(STATE_NAMES)},
                }

    source_path = REPO_ROOT / "f1tenth_mpc/src/vehicle_model.c"
    result = {
        "schema_version": 1,
        "analysis_id": "production_mpc_paired_odom_initialization_20261012",
        "benchmark_id": benchmark["benchmark_id"],
        "benchmark_sha256": _sha256(benchmark_path),
        "dataset_sha256": benchmark["dataset_sha256"],
        "validation_bag_sha256": {
            key: _sha256(path.resolve()) for key, path in bag_paths.items()},
        "model_source": str(source_path),
        "model_source_sha256": _sha256(source_path),
        "model_library": str(library_path),
        "model_library_sha256": _sha256(library_path),
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "trajectory": str(trajectory_path),
        "trajectory_sha256": _sha256(trajectory_path),
        "odom_archive_alignment": alignment_report,
        "odom_initial_state_errors": init_report,
        "initialization_modes": {
            "simulator_truth": "truth u_com, v_rear, yaw rate; truth-derived map pose",
            "baseline_odom": "baseline observer u, v at base_link/rear axle, gyro yaw rate; truth-derived map pose",
            "corrected_odom": "3-5 m/s^2 gated observer u, v at base_link/rear axle, gyro yaw rate; truth-derived map pose",
        },
        "future_model_inputs": "same recorded /cmd/speed commands and map curvature for every mode",
        "future_truth_used_as_input": False,
        "map_pose_and_actuator_initialization": "held at truth/recorded values to isolate u/v/r odometry initialization",
        "score_horizons_s": list(HORIZONS_S),
        "stepwise_open_loop_errors": stepwise_report,
        "first_step_errors_by_frozen_benchmark_category": category_report,
        "independent_run_count": len(needed_run_ids),
        "window_count": window_count,
        "rollout_failures": failures,
        "scores": report,
        "limitations": [
            "This is open-loop recorded-command replay, not a closed-loop controller run.",
            "Map pose and actuator state are held to recorded truth/feedback; AMCL, EKF pose fusion, command timing and MPC solve decisions are not tested here.",
            "The two held-out practice runs cover ordinary practice motion, not the full 0-12 m/s and full-steering envelope.",
            "Only two independent runs are available, so run-level uncertainty is coarse.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("--bag", action="append", required=True,
                        help="validation report key=bag database")
    parser.add_argument("--odom-states", action="append", required=True,
                        help="run suffix r02/r03=paired_source_states.npz")
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bags, archives = {}, {}
    for item, dest, label in [(value, bags, "--bag") for value in args.bag] + [
        (value, archives, "--odom-states") for value in args.odom_states]:
        key, separator, value = item.partition("=")
        if not separator or key in dest:
            raise ValueError(f"{label} entries must be unique key=path pairs")
        dest[key] = Path(value)
    result = compare(args.benchmark, args.library, args.config,
                     args.trajectory, bags, archives, args.output)
    print(json.dumps({
        "windows": result["window_count"],
        "failures": {key: len(value)
                     for key, value in result["rollout_failures"].items()},
        "0.75s_rmse": {
            mode: result["scores"][mode]["macro_run_mean_rmse"]["0.75s"]
            for mode in result["initialization_modes"]},
        "output": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
