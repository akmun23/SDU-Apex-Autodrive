#!/usr/bin/env python3
"""Score the command-driven 3D rigid-body teacher on held-out whole captures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from tools.vehicle_dynamics_learning.train_nssm import _load_dataset, _torch
from tools.vehicle_dynamics_learning.train_rigid_body_teacher import (
    SIMULATOR_DT_S,
    _full_sequence_scores,
    _state_arrays,
    _torch_model,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _summarize(rows: list[dict[str, Any]], run_ids: list[str]) -> dict[str, Any]:
    by_horizon: dict[str, dict[str, Any]] = {}
    horizon_keys = sorted(set.intersection(*[
        set(row["horizons"]) for row in rows
    ]), key=lambda value: float(value[:-1])) if rows else []
    for horizon in horizon_keys:
        by_horizon[horizon] = {}
        for run_id in run_ids:
            local = [row["horizons"][horizon] for row in rows
                     if row["run_id"] == run_id and horizon in row["horizons"]]
            if local:
                by_horizon[horizon][run_id] = {
                    "sequence_count": len(local),
                    "macro_sequence_position_error_m": float(np.mean(
                        [item["xy_error_m"] for item in local])),
                    "macro_sequence_orientation_error_rad": float(np.mean(
                        [item["orientation_error_rad"] for item in local])),
                    "macro_sequence_body_velocity_error_mps": float(np.mean(
                        [item["body_velocity_error_mps"] for item in local])),
                }
    whole_run = {}
    for run_id in run_ids:
        local = [row for row in rows if row["run_id"] == run_id]
        if local:
            whole_run[run_id] = {
                "sequence_count": len(local),
                "macro_sequence_position_rmse_m": float(np.mean(
                    [row["xy_position_rmse_m"] for row in local])),
                "macro_sequence_endpoint_position_error_m": float(np.mean(
                    [row["xy_position_endpoint_m"] for row in local])),
                "macro_sequence_orientation_rmse_rad": float(np.mean(
                    [row["orientation_rmse_rad"] for row in local])),
                "macro_sequence_body_state_rmse": {
                    name: float(np.mean([row["state_rmse"][name]
                                         for row in local]))
                    for name in local[0]["state_rmse"]
                },
            }
    return {"horizons": by_horizon, "whole_free_run": whole_run}


def _score_practice_windows(torch, model, data, state_all, quaternion_all,
                            position_all, windows, history_steps: int
                            ) -> list[dict[str, Any]]:
    reports = []
    model.eval()
    for window in windows:
        start = int(window["global_start_index"])
        steps = int(window["future_command_steps"])
        context_start = start - history_steps + 1
        future_end = start + steps + 1
        if (int(window["context_steps"]) != history_steps
                or context_start < 0 or future_end > len(data["frames"])):
            raise ValueError("practice benchmark window is incompatible with checkpoint")
        run_id = str(window["run_id"])
        run_index = int(np.flatnonzero(data["run_ids"].astype(str) == run_id)[0])
        if np.any(data["frame_run_index"][context_start:future_end] != run_index):
            raise ValueError(f"{run_id}: benchmark window crosses a run boundary")
        if np.any(np.diff(data["packet_sequence"][context_start:future_end]) != 1):
            raise ValueError(f"{run_id}: benchmark window contains a packet gap")
        hidden = torch.zeros(1, model.cell.hidden_size, dtype=torch.float32)
        with torch.no_grad():
            for index in range(context_start, start):
                hidden = model.update_hidden(
                    torch.as_tensor(state_all[index:index + 1], dtype=torch.float32),
                    torch.as_tensor(quaternion_all[index:index + 1],
                                    dtype=torch.float32),
                    torch.as_tensor(data["frames"][index, 7:9][None],
                                    dtype=torch.float32),
                    hidden)
            state = torch.as_tensor(state_all[start:start + 1], dtype=torch.float32)
            quaternion = torch.as_tensor(
                quaternion_all[start:start + 1], dtype=torch.float32)
            position = torch.as_tensor(position_all[start:start + 1],
                                       dtype=torch.float32)
            predicted_state = []
            predicted_quaternion = []
            predicted_position = []
            for step in range(steps):
                command_index = start + step
                command = torch.as_tensor(
                    data["frames"][command_index, 7:9][None],
                    dtype=torch.float32)
                dt = torch.as_tensor([SIMULATOR_DT_S], dtype=torch.float32)
                state, quaternion, position, hidden, _ = model.transition(
                    state, quaternion, position, command, hidden, dt)
                predicted_state.append(state[0].numpy().copy())
                predicted_quaternion.append(quaternion[0].numpy().copy())
                predicted_position.append(position[0].numpy().copy())
        predicted_state = np.asarray(predicted_state)
        predicted_quaternion = np.asarray(predicted_quaternion)
        predicted_position = np.asarray(predicted_position)
        target_indices = np.arange(start + 1, start + steps + 1)
        truth_state = state_all[target_indices]
        truth_position = position_all[target_indices]
        truth_quaternion = quaternion_all[target_indices]
        relative = (Rotation.from_quat(truth_quaternion).inv()
                    * Rotation.from_quat(predicted_quaternion))
        orientation_error = relative.magnitude()
        position_error = predicted_position[:, :2] - truth_position[:, :2]
        radial_error = np.linalg.norm(position_error, axis=1)
        velocity_error = np.linalg.norm(
            predicted_state[:, :3] - truth_state[:, :3], axis=1)
        horizon_scores = {}
        for seconds in (0.25, 0.5, 0.75, 1.0, 2.0, 5.0):
            index = round(seconds / SIMULATOR_DT_S) - 1
            if index < len(radial_error):
                horizon_scores[f"{seconds:g}s"] = {
                    "position_error_m": float(radial_error[index]),
                    "orientation_error_rad": float(orientation_error[index]),
                    "body_velocity_error_mps": float(velocity_error[index]),
                }
        reports.append({
            "run_id": run_id,
            "categories": list(window.get("categories", [])),
            "initial_speed_mps": float(window["initial_speed_mps"]),
            "initial_steering_feedback_rad": float(
                window["initial_steering_feedback_rad"]),
            "initial_wheel_body_mismatch_proxy_mps": float(
                window["initial_wheel_body_mismatch_proxy_mps"]),
            "free_run_seconds": steps * SIMULATOR_DT_S,
            "position_rmse_m": float(np.sqrt(np.mean(radial_error ** 2))),
            "position_endpoint_error_m": float(radial_error[-1]),
            "orientation_rmse_rad": float(np.sqrt(
                np.mean(orientation_error ** 2))),
            "body_velocity_rmse_mps": float(np.sqrt(
                np.mean(velocity_error ** 2))),
            "horizons": horizon_scores,
        })
    return reports


def _practice_window_summary(rows: list[dict[str, Any]], run_ids: list[str]
                             ) -> dict[str, Any]:
    horizons = sorted(set.intersection(*[
        set(row["horizons"]) for row in rows
    ]), key=lambda value: float(value[:-1])) if rows else []
    per_run = {}
    per_category = {}
    for horizon in horizons:
        per_run[horizon] = {}
        for run_id in run_ids:
            local = [row["horizons"][horizon] for row in rows
                     if row["run_id"] == run_id]
            if local:
                per_run[horizon][run_id] = {
                    "window_count": len(local),
                    "macro_position_error_m": float(np.mean(
                        [item["position_error_m"] for item in local])),
                    "macro_orientation_error_rad": float(np.mean(
                        [item["orientation_error_rad"] for item in local])),
                    "macro_body_velocity_error_mps": float(np.mean(
                        [item["body_velocity_error_mps"] for item in local])),
                }
        category_names = sorted({category for row in rows
                                 for category in row["categories"]})
        per_category[horizon] = {}
        for category in category_names:
            per_category[horizon][category] = {}
            for run_id in run_ids:
                local = [row["horizons"][horizon] for row in rows
                         if row["run_id"] == run_id
                         and category in row["categories"]]
                if local:
                    per_category[horizon][category][run_id] = {
                        "window_count": len(local),
                        "macro_position_error_m": float(np.mean(
                            [item["position_error_m"] for item in local])),
                        "macro_orientation_error_rad": float(np.mean(
                            [item["orientation_error_rad"] for item in local])),
                        "macro_body_velocity_error_mps": float(np.mean(
                            [item["body_velocity_error_mps"] for item in local])),
                    }
    return {"per_run": per_run, "by_category": per_category}


def score(checkpoint_path: Path, dataset_path: Path, run_ids: list[str],
          output_path: Path, quality_manifest_path: Path | None = None,
          benchmark_path: Path | None = None
          ) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    if not run_ids or len(set(run_ids)) != len(run_ids):
        raise ValueError("provide unique held-out run IDs")
    torch, nn = _torch()
    torch.set_num_threads(1)
    checkpoint = torch.load(checkpoint_path, map_location="cpu",
                            weights_only=False)
    metadata = checkpoint["metadata"]
    data = _load_dataset(dataset_path)
    if (data["schema_version"] < 7
            or data["simulator_rigid_state"] is None
            or data["simulator_pose_xyyaw"] is None
            or not np.allclose(data["dt_s"], SIMULATOR_DT_S,
                               rtol=0.0, atol=1.0e-7)):
        raise ValueError("score requires audited 25 ms schema-7+ rigid/pose data")
    frame_run_index = np.full(len(data["frames"]), -1, dtype=np.int32)
    for sequence_id, (start_raw, end_raw) in enumerate(data["bounds"]):
        start, end = int(start_raw), int(end_raw)
        frame_run_index[start:end] = int(data["seq_run"][sequence_id])
    data["frame_run_index"] = frame_run_index
    lookup = {str(value): index for index, value in enumerate(data["run_ids"])}
    missing = sorted(set(run_ids) - set(lookup))
    if missing:
        raise ValueError(f"held-out run IDs are missing: {missing}")
    train_ids = set(map(str, metadata["training_run_ids"]))
    overlap = sorted(set(run_ids).intersection(train_ids))
    if overlap:
        raise ValueError(f"refusing to score training runs: {overlap}")
    splits = {run_id: str(data["splits"][lookup[run_id]]) for run_id in run_ids}
    if any(value not in ("validation", "test", "final_test", "unseen_practice")
           for value in splits.values()):
        raise ValueError(f"selected captures are not held out: {splits}")

    if quality_manifest_path is not None:
        manifest = json.loads(quality_manifest_path.read_text(encoding="utf-8"))
        manifest_rows = {str(row["run_id"]): row
                         for row in manifest.get("runs", [])}
        for run_id in run_ids:
            row = manifest_rows.get(run_id)
            if (row is None or row.get("aborted")
                    or not row.get("clean_stream_and_collision_gate")
                    or row.get("quality_failures")
                    or any(int(value) != 0
                           for value in row.get("collisions", []))):
                raise ValueError(f"quality manifest rejects held-out run {run_id}")

    state_all, quaternion_all, position_all = _state_arrays(data)
    for run_id in run_ids:
        run_index = lookup[run_id]
        for seq_id, (start_raw, end_raw) in enumerate(data["bounds"]):
            if int(data["seq_run"][seq_id]) != run_index:
                continue
            start, end = int(start_raw), int(end_raw)
            speed = np.hypot(state_all[start:end, 0], state_all[start:end, 1])
            if not np.isfinite(speed).all() or np.any(speed > 12.0 + 1.0e-5):
                raise ValueError(f"{run_id}: selected capture exceeds 12 m/s domain")

    model_type = _torch_model(
        torch, nn, int(metadata["hidden_size"]),
        np.asarray(metadata["state_mean"], dtype=np.float32),
        np.asarray(metadata["state_scale"], dtype=np.float32),
        np.asarray(metadata["command_mean"], dtype=np.float32),
        np.asarray(metadata["command_scale"], dtype=np.float32),
        np.asarray(metadata["rate_mean"], dtype=np.float32),
        np.asarray(metadata["rate_scale"], dtype=np.float32),
    )
    model = model_type()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()

    benchmark = None
    if benchmark_path is not None:
        benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
        if (benchmark.get("benchmark_id") != "practice_transfer_benchmark_v1"
                or benchmark.get("frozen") is not True
                or benchmark.get("training_or_checkpoint_selection_use") is not False
                or _sha256(dataset_path) != benchmark.get("dataset_sha256")):
            raise ValueError("benchmark/dataset integrity or role check failed")
        expected_runs = sorted({str(window["run_id"])
                                for window in benchmark["windows"]})
        if expected_runs != sorted(run_ids):
            raise ValueError("selected run IDs must exactly match frozen benchmark runs")
        rows = _score_practice_windows(
            torch, model, data, state_all, quaternion_all, position_all,
            benchmark["windows"], int(metadata["history_steps"]))
    else:
        rows = []
        for split in sorted(set(splits.values())):
            rows.extend(_full_sequence_scores(
                torch, model, data, state_all, quaternion_all, position_all,
                split, "cpu", int(metadata["history_steps"])))
        rows = [row for row in rows if row["run_id"] in set(run_ids)]
    found = {row["run_id"] for row in rows}
    if found != set(run_ids):
        raise ValueError(f"no free-running sequences for {sorted(set(run_ids)-found)}")
    report = {
        "schema_version": 1,
        "purpose": "recursive command-only free-running validation of the 3D rigid-body residual teacher",
        "future_truth_or_sensor_inputs": False,
        "initialization": (
            "simulator state/quaternion/position over the declared context; "
            "after context, predicted state and recorded commands only"),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_training_runs": metadata["training_run_ids"],
        "dataset": str(dataset_path.resolve()),
        "dataset_sha256": _sha256(dataset_path),
        "quality_manifest": (str(quality_manifest_path.resolve())
                             if quality_manifest_path else None),
        "history_steps": int(metadata["history_steps"]),
        "history_seconds": (int(metadata["history_steps"]) - 1)
                            * SIMULATOR_DT_S,
        "rollout_training_steps": int(metadata["rollout_steps"]),
        "selected_run_splits": splits,
        "independent_unit": "whole capture run; conditions/probes within a run are repeated measures",
        "benchmark": (str(benchmark_path.resolve())
                      if benchmark_path else None),
        "summary": (_practice_window_summary(rows, run_ids) if benchmark
                    else _summarize(rows, run_ids)),
        "runs": [{
            "run_id": run_id,
            "split": splits[run_id],
            "sequence_count": sum(row["run_id"] == run_id for row in rows),
            "sequences": [row for row in rows if row["run_id"] == run_id],
        } for run_id in run_ids],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--run-id", action="append", dest="run_ids",
                        required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quality-manifest", type=Path)
    parser.add_argument("--benchmark", type=Path)
    args = parser.parse_args()
    report = score(args.checkpoint, args.dataset, args.run_ids,
                   args.output, args.quality_manifest, args.benchmark)
    print(json.dumps({
        "output": str(args.output),
        "summary": report["summary"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
