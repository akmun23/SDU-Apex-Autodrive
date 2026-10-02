#!/usr/bin/env python3
"""Replay the compiled production MPC motion model on frozen practice starts.

The model receives the current truth-initialized state, known map curvature,
and the recorded steering/target-speed command trace. Future simulator state
is used only as a score target. This isolates the production model from MPC
solver decisions and from odometry initialization error.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

from tools.racing.offline.controller.production_mpc import ProductionMpc
from tools.vehicle_dynamics_learning.build_practice_transfer_benchmark import (
    _run_rows,
)
from tools.vehicle_dynamics_learning.train_direct_sequence_teacher import (
    COM_X_M,
)
from tools.vehicle_dynamics_learning.train_nssm import _load_dataset


DT_S = 0.025
HORIZONS_S = (0.25, 0.5, 0.75, 1.0, 2.0)
STATE_NAMES = (
    "u_com_mps", "v_com_mps", "yaw_rate_rps", "e_y_m", "e_psi_rad",
    "actual_steering_rad", "target_speed_mps", "steering_command_rad",
)
REPO_ROOT = Path(__file__).resolve().parents[4]


class MpcModelState(ctypes.Structure):
    _fields_ = [(name, ctypes.c_float) for name in (
        "e_y", "e_psi", "u", "v", "r", "target_speed",
        "steering_command", "delayed_steering_command_1",
        "delayed_steering_command_2", "actual_steering_angle")]


class MpcModelControl(ctypes.Structure):
    _fields_ = [("steering_rate", ctypes.c_float),
                ("target_speed_rate", ctypes.c_float)]


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


def _read_commands(bag_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with sqlite3.connect(bag_path.resolve().as_uri() + "?mode=ro", uri=True) as db:
        topic = db.execute("SELECT id,type FROM topics WHERE name=?",
                           ("/cmd/speed",)).fetchone()
        if topic is None:
            raise ValueError(f"bag is missing production command /cmd/speed: {bag_path}")
        topic_id, type_name = int(topic[0]), str(topic[1])
        message_type = get_message(type_name)
        rows = []
        for stamp, serialized in db.execute(
                "SELECT timestamp,data FROM messages WHERE topic_id=? "
                "ORDER BY timestamp,id", (topic_id,)):
            msg = deserialize_message(bytes(serialized), message_type)
            steering = float(msg.drive.steering_angle)
            speed = float(msg.drive.speed)
            if math.isfinite(steering) and math.isfinite(speed):
                rows.append((int(stamp), steering, speed))
    if len(rows) < 2:
        raise ValueError("production command stream has fewer than two finite samples")
    values = np.asarray(rows, dtype=np.float64)
    if np.any(np.diff(values[:, 0]) < 0):
        raise ValueError("production command timestamps are not ordered")
    return values[:, 0].astype(np.int64), values[:, 1], values[:, 2]


def _load_path(registry_path: Path, library: Path, trajectory: Path,
               config_path: Path) -> tuple[dict[str, Any], np.ndarray]:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    production = registry["production_pipeline"]
    config_entry = production["mpc_config"]
    if _sha256(config_path) != config_entry["sha256"]:
        raise ValueError("production MPC config hash differs from WP0 registry")
    source = production["vehicle_model_c"]
    source_path = REPO_ROOT / source["path"]
    if _sha256(source_path) != source["sha256"]:
        raise ValueError("production vehicle model source differs from WP0 registry")
    if not library.is_file() or not trajectory.is_file():
        raise FileNotFoundError("production MPC library or frozen practice trajectory missing")
    rows = np.loadtxt(trajectory, delimiter=",", comments="#", dtype=np.float64)
    if (rows.ndim != 2 or rows.shape[1] < 6 or len(rows) < 3
            or not np.isfinite(rows[:, :6]).all()
            or np.any(np.diff(rows[:, 0]) <= 0)):
        raise ValueError("practice trajectory does not match the production MPC format")
    return registry, rows


def _project_run(controller: ProductionMpc, trajectory: np.ndarray,
                 lap_length: float, pose: np.ndarray
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                            np.ndarray]:
    count = len(pose)
    projection = np.empty((count, 4), dtype=np.float64)
    previous_segment = 2**64 - 1
    for index, value in enumerate(pose):
        result = controller.project(value, previous_segment=previous_segment,
                                    local_search_radius=160)
        progress, e_y, e_psi, distance, segment = result
        projection[index] = (progress, e_y, e_psi, distance)
        previous_segment = segment
    progress = np.mod(projection[:, 0], lap_length)
    curvature = np.interp(progress, trajectory[:, 0], trajectory[:, 4])
    return (progress, projection[:, 1], projection[:, 2], curvature,
            projection[:, 3])


def _metric(values: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(np.mean(values ** 2))),
        "bias": float(np.mean(values)),
        "p95_abs": float(np.quantile(np.abs(values), 0.95)),
    }


def _summarize_runs(values: dict[str, float], seed: int) -> dict[str, Any]:
    items = np.asarray(list(values.values()), dtype=np.float64)
    rng = np.random.default_rng(seed)
    sample = rng.integers(0, len(items), size=(10000, len(items)))
    mean_draws = items[sample].mean(axis=1)
    return {
        "independent_run_count": len(items),
        "macro_run_mean_rmse": float(np.mean(items)),
        "run_min": float(np.min(items)),
        "run_max": float(np.max(items)),
        "run_cluster_bootstrap_95pct_ci": np.quantile(
            mean_draws, (0.025, 0.975)).tolist(),
        "per_run_rmse": dict(sorted(values.items())),
    }


def _recorded_command_arrays(data: dict[str, Any], run_index: int,
                             commands: tuple[np.ndarray, np.ndarray, np.ndarray],
                             indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    command_stamps, command_steering, command_speed = commands
    sample_stamps = data["sample_time_ns"][indices]
    command_index = np.searchsorted(command_stamps, sample_stamps,
                                    side="right") - 1
    if np.any(command_index < 0):
        raise ValueError(f"run index {run_index} begins before command history")
    steering = command_steering[command_index]
    speed = command_speed[command_index]
    if not np.isfinite(steering).all() or not np.isfinite(speed).all():
        raise ValueError("aligned production command contains non-finite values")
    logged_steering = data["frames"][indices, 7]
    difference = np.abs(steering - logged_steering)
    if np.quantile(difference, 0.99) > 0.03:
        raise ValueError(
            "recorded /cmd/speed steering does not align with the packet-aligned "
            "bridge steering command")
    return steering, speed


def score(benchmark_path: Path, registry_path: Path, bag_paths: dict[str, Path],
          library_path: Path, trajectory_path: Path, config_path: Path,
          output_path: Path) -> dict[str, Any]:
    benchmark_path, registry_path, output_path = (
        benchmark_path.resolve(), registry_path.resolve(), output_path.resolve())
    library_path, trajectory_path, config_path = (
        library_path.resolve(), trajectory_path.resolve(), config_path.resolve())
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite production MPC report: {output_path}")
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    if (benchmark.get("benchmark_id") != "practice_transfer_benchmark_v1"
            or benchmark.get("frozen") is not True
            or _sha256(Path(benchmark["dataset"])) != benchmark["dataset_sha256"]):
        raise ValueError("practice benchmark is not frozen or its dataset changed")
    registry, trajectory = _load_path(registry_path, library_path,
                                      trajectory_path, config_path)
    dataset_path = Path(benchmark["dataset"])
    data = _load_dataset(dataset_path)
    with np.load(dataset_path, allow_pickle=False) as archive:
        if "sample_time_ns" not in archive.files:
            raise ValueError("practice dataset lacks command-alignment timestamps")
        data["sample_time_ns"] = archive["sample_time_ns"].astype(
            np.int64, copy=False)
    window_data = {
        **data,
        "sequence_bounds": data["bounds"],
        "sequence_run_index": data["seq_run"],
    }
    targets = np.column_stack((
        data["simulator_rigid_state"][:, 7],
        data["simulator_rigid_state"][:, 8],
        data["simulator_rigid_state"][:, 12],
        data["frames"][:, 3:7],
    )).astype(np.float64)
    model_library = ctypes.CDLL(str(library_path))
    model_library.mpc_vehicle_model_step.argtypes = [
        ctypes.POINTER(MpcModelState), ctypes.POINTER(MpcModelControl),
        ctypes.c_float, ctypes.c_float,
    ]
    model_library.mpc_vehicle_model_step.restype = MpcStageResult

    reports = benchmark["validation_reports"]
    bags_by_run = {run_id: path.resolve() for run_id, path in bag_paths.items()}
    if set(bags_by_run) != set(reports):
        raise ValueError("bag inputs must exactly match benchmark validation runs")
    commands_by_run = {}
    for run_id, bag_path in bags_by_run.items():
        validation = json.loads(Path(reports[run_id]["report"]).read_text(
            encoding="utf-8"))
        if (_sha256(bag_path) != reports[run_id]["bag_sha256"]
                or validation.get("bag_sha256") != reports[run_id]["bag_sha256"]):
            raise ValueError(f"validation bag hash mismatch for {run_id}")
        commands_by_run[run_id] = _read_commands(bag_path)

    run_index_by_name = {str(run_id): index
                         for index, run_id in enumerate(data["run_ids"])}
    projections: dict[str, dict[str, np.ndarray]] = {}
    local_index_by_global: dict[str, dict[int, int]] = {}
    command_values: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    indices_by_run: dict[str, np.ndarray] = {}
    with ProductionMpc(library_path, config_path, trajectory_path) as controller:
        lap_length = controller.lap_length_m
        for report_run_id in reports:
            suffix = report_run_id.rsplit("_", 1)[-1]
            dataset_run_id = f"practice_unseen_model_validation_20261001_{suffix}"
            if dataset_run_id not in run_index_by_name:
                raise ValueError(f"benchmark dataset lacks {dataset_run_id}")
            run_index = run_index_by_name[dataset_run_id]
            indices = _run_rows(window_data, run_index)
            indices_by_run[dataset_run_id] = indices
            pose = data["simulator_pose_xyyaw"][indices]
            progress, e_y, e_psi, curvature, distance = _project_run(
                controller, trajectory, lap_length, pose)
            projections[dataset_run_id] = {
                "progress": progress, "e_y": e_y, "e_psi": e_psi,
                "curvature": curvature, "distance": distance,
            }
            local_index_by_global[dataset_run_id] = {
                int(global_index): local_index
                for local_index, global_index in enumerate(indices)}
            command_values[dataset_run_id] = _recorded_command_arrays(
                data, run_index, commands_by_run[report_run_id], indices)

        by_run: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
            lambda: defaultdict(list))
        by_category: dict[str, dict[str, dict[str, list[dict[str, Any]]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list)))
        failures = []
        all_start_rows = []
        for window in benchmark["windows"]:
            dataset_run_id = str(window["run_id"])
            local_index = local_index_by_global[dataset_run_id].get(
                int(window["global_start_index"]))
            if local_index is None:
                raise ValueError("benchmark start is outside its continuous run rows")
            indices = indices_by_run[dataset_run_id]
            steering, target_speed = command_values[dataset_run_id]
            project = projections[dataset_run_id]
            values = targets[indices]
            frames = data["frames"][indices]
            predicted = []
            start_steer = float(steering[local_index])
            if local_index < 2:
                raise ValueError("frozen common start lacks two steering-delay samples")
            state = MpcModelState(
                float(project["e_y"][local_index]),
                float(project["e_psi"][local_index]),
                float(values[local_index, 0]),
                float(values[local_index, 1] - COM_X_M * values[local_index, 2]),
                float(values[local_index, 2]),
                float(target_speed[local_index]),
                start_steer,
                float(steering[local_index - 1]),
                float(steering[local_index - 2]),
                float(frames[local_index, 3]),
            )
            progress = float(project["progress"][local_index])
            valid = True
            limit = 80
            for step in range(limit):
                command_index = local_index + step
                steer_rate = (steering[command_index + 1]
                              - steering[command_index]) / DT_S
                speed_rate = (target_speed[command_index + 1]
                              - target_speed[command_index]) / DT_S
                control = MpcModelControl(float(steer_rate), float(speed_rate))
                curvature = float(np.interp(
                    progress % lap_length, trajectory[:, 0], trajectory[:, 4]))
                stage = model_library.mpc_vehicle_model_step(
                    ctypes.byref(state), ctypes.byref(control),
                    ctypes.c_float(DT_S), ctypes.c_float(curvature))
                if not stage.valid or not math.isfinite(stage.delta_s_m):
                    failures.append({
                        "run_id": dataset_run_id,
                        "packet_sequence": int(window["packet_sequence"]),
                        "failed_step": step + 1,
                        "branch_flags": int(stage.branch_flags),
                    })
                    valid = False
                    break
                state = stage.next
                progress += float(stage.delta_s_m)
                next_index = local_index + step + 1
                predicted.append((
                    float(state.u), float(state.v + COM_X_M * state.r),
                    float(state.r), float(state.e_y), float(state.e_psi),
                    float(state.actual_steering_angle), float(state.target_speed),
                    float(state.steering_command),
                ))
            if not valid:
                continue
            predicted_array = np.asarray(predicted, dtype=np.float64)
            horizon_rows = []
            for horizon in HORIZONS_S:
                steps = round(horizon / DT_S)
                future = slice(local_index + 1, local_index + 1 + steps)
                truth = np.column_stack((
                    values[future, 0],
                    values[future, 1],
                    values[future, 2],
                    project["e_y"][future],
                    project["e_psi"][future],
                    frames[future, 3],
                    target_speed[future],
                    steering[future],
                ))
                error = predicted_array[:steps] - truth
                error[:, 4] = np.arctan2(np.sin(error[:, 4]), np.cos(error[:, 4]))
                row = {name: _metric(error[:, channel])
                       for channel, name in enumerate(STATE_NAMES)}
                by_run[dataset_run_id][f"{horizon:g}s"].append(row)
                for category in window["categories"]:
                    by_category[category][dataset_run_id][f"{horizon:g}s"].append(row)
                horizon_rows.append(row)
            all_start_rows.append({
                "run_id": dataset_run_id,
                "packet_sequence": int(window["packet_sequence"]),
                "categories": window["categories"],
                "projection_distance_m": float(project["distance"][local_index]),
            })

    per_run_report: dict[str, Any] = {}
    macro_report: dict[str, Any] = {}
    for horizon in HORIZONS_S:
        key = f"{horizon:g}s"
        channel_runs: dict[str, dict[str, float]] = {
            name: {} for name in STATE_NAMES}
        for run_id, horizons in by_run.items():
            rows = horizons.get(key, [])
            if not rows:
                continue
            per_run_report.setdefault(run_id, {})[key] = {
                "window_count": len(rows),
                "state_rmse": {
                    name: float(np.mean([row[name]["rmse"] for row in rows]))
                    for name in STATE_NAMES},
            }
            for name in STATE_NAMES:
                channel_runs[name][run_id] = float(np.mean([
                    row[name]["rmse"] for row in rows]))
        macro_report[key] = {
            name: _summarize_runs(values, 9600 + round(horizon * 100) + i)
            for i, (name, values) in enumerate(channel_runs.items()) if values
        }

    category_report: dict[str, Any] = {}
    for category, runs in by_category.items():
        category_report[category] = {}
        for horizon in HORIZONS_S:
            key = f"{horizon:g}s"
            category_report[category][key] = {}
            for run_id, horizons in runs.items():
                rows = horizons.get(key, [])
                if rows:
                    category_report[category][key][run_id] = {
                        name: float(np.mean([row[name]["rmse"] for row in rows]))
                        for name in STATE_NAMES}

    result = {
        "schema_version": 1,
        "model_id": "MPC0_production_vehicle_model",
        "benchmark_id": benchmark["benchmark_id"],
        "benchmark_path": str(benchmark_path),
        "benchmark_sha256": _sha256(benchmark_path),
        "dataset_sha256": benchmark["dataset_sha256"],
        "production_model_source": registry["production_pipeline"][
            "vehicle_model_c"],
        "production_model_library": str(library_path),
        "production_model_library_sha256": _sha256(library_path),
        "production_config": str(config_path),
        "production_config_sha256": _sha256(config_path),
        "trajectory": str(trajectory_path),
        "trajectory_sha256": _sha256(trajectory_path),
        "initial_state_source": "simulator truth for body motion and map pose; recorded actuator state/commands",
        "future_model_inputs": "recorded /cmd/speed steering-angle and target-speed sequence; map curvature sampled at predicted progress",
        "future_truth_or_sensors_used_as_inputs": False,
        "command_alignment": "latest recorded command at/before each 40 Hz source sample; finite-difference rate over fixed 25 ms step",
        "steering_command_alignment": "packet-aligned bridge steering feature; p99 absolute difference <=0.03 rad",
        "score_horizons_s": list(HORIZONS_S),
        "independent_run_count": len(per_run_report),
        "window_count_scored": len(all_start_rows),
        "nonlinear_rollout_failures": failures,
        "per_run": per_run_report,
        "macro_run_summary": macro_report,
        "by_category": category_report,
        "start_projection_distance_m": {
            "p50": float(np.median([row["projection_distance_m"] for row in all_start_rows])),
            "p95": float(np.quantile([row["projection_distance_m"] for row in all_start_rows], 0.95)),
            "max": float(np.max([row["projection_distance_m"] for row in all_start_rows])),
        },
        "limitations": [
            "This is open-loop prediction under the recorded command trace, with truth-initialized state; it does not measure observer error or closed-loop counterfactual behavior.",
            "Practice starts reach only about 8 m/s and 0.283 rad steering; this is not validation of the full 0–12 m/s/high-steering envelope.",
            "The controller command stream was generated feedback-wise during the run; the model is scored on those recorded commands, not a new MPC policy.",
            "Two independent captures make run-level confidence intervals coarse.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("registry", type=Path)
    parser.add_argument("--bag", type=str, action="append", required=True,
                        help="source-run-id=rosbag SQLite database")
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bag_paths = {}
    for item in args.bag:
        run_id, separator, path = item.partition("=")
        if not separator or run_id in bag_paths:
            raise ValueError("--bag must be source-run-id=path and unique")
        bag_paths[run_id] = Path(path)
    report = score(args.benchmark, args.registry, bag_paths, args.library,
                   args.trajectory, args.config, args.output)
    print(json.dumps({
        "model": report["model_id"],
        "runs": report["independent_run_count"],
        "windows": report["window_count_scored"],
        "failures": len(report["nonlinear_rollout_failures"]),
        "0.75s_rmse": {
            channel: round(report["macro_run_summary"]["0.75s"][channel][
                "macro_run_mean_rmse"], 4)
            for channel in STATE_NAMES
        },
        "output": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
