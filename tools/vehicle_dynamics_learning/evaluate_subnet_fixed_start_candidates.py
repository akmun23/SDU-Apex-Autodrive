#!/usr/bin/env python3
"""Compare native SUBNET checkpoints on the frozen same-start whole-run cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from subnet_body_plant import SubnetBodyPlant


ROOT = Path(__file__).resolve().parents[2]
RESET_ROOT = ROOT / "live_runs/derived_dynamics_learning_20260928/subnet_reset_20261004"
BODY_DATASET = RESET_ROOT / "body_sysid_v1.npz"
BODY_MANIFEST = RESET_ROOT / "body_sysid_v1_manifest.json"
STARTS = RESET_ROOT / "evaluation_starts.json"
DT_S = 0.025
HORIZONS = (1, 10, 30, 40, 80, 150, 200, 400)
CHANNELS = ("u_rear_mps", "v_rear_mps", "yaw_rate_rps")
BOOTSTRAP_REPLICATES = 5000
SEED = 20261004


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _fixed_windows(body: Any, source: Any, starts_path: Path):
    starts_doc = json.loads(starts_path.read_text(encoding="utf-8"))
    start_runs = starts_doc["split_roles"]["development_validation"]
    frame_ids = np.asarray(body["source_frame_index"], dtype=np.int64)
    inputs = np.asarray(body["inputs"], dtype=np.float32)
    outputs = np.asarray(body["outputs"], dtype=np.float32)
    sequence_bounds = np.asarray(body["sequence_bounds"], dtype=np.int64)
    sequence_run_ids = np.asarray(body["sequence_run_id"]).astype(str)
    sequence_splits = np.asarray(body["sequence_split"]).astype(str)
    source_pose = np.asarray(source["simulator_pose_xyyaw"], dtype=np.float64)
    references = []
    for run_id, run_starts in sorted(start_runs.items()):
        for item in run_starts:
            source_row = int(item["absolute_row"])
            local = int(np.searchsorted(frame_ids, source_row))
            if local >= len(frame_ids) or frame_ids[local] != source_row:
                raise ValueError(f"frozen start does not map exactly: {run_id}/{source_row}")
            sequence = int(np.searchsorted(
                sequence_bounds[:, 0], local, side="right") - 1)
            if (sequence < 0 or local >= sequence_bounds[sequence, 1]
                    or sequence_run_ids[sequence] != run_id
                    or sequence_splits[sequence] != "validation"):
                raise ValueError(f"frozen start maps outside validation run: {run_id}")
            begin, end = map(int, sequence_bounds[sequence])
            if (local - 11 < begin or local + max(HORIZONS) >= end
                    or int(item["maximum_available_horizon_steps"]) < max(HORIZONS)):
                raise ValueError(f"frozen start lacks an isolated 10 s continuation: {run_id}")
            references.append({"run_id": run_id, "local": local,
                               "sequence_start": begin, "source_row": source_row})
    if len(references) != 384:
        raise ValueError(f"expected 384 frozen starts, found {len(references)}")
    history_u = np.stack([
        inputs[row["local"] - 11:row["local"] + 1] for row in references])
    history_y = np.stack([
        outputs[row["local"] - 11:row["local"] + 1] for row in references])
    future_u = np.stack([
        inputs[row["local"] + 1:row["local"] + 401] for row in references])
    truth = np.stack([
        outputs[row["local"] + 1:row["local"] + 401] for row in references])
    initial_body = np.stack([outputs[row["local"]] for row in references])
    initial_pose = np.stack([source_pose[row["source_row"]] for row in references])
    truth_pose = np.stack([
        source_pose[row["source_row"] + 1:row["source_row"] + 401]
        for row in references])
    return (references, history_u, history_y, future_u, truth,
            initial_body, initial_pose, truth_pose)


def _predict(model: SubnetBodyPlant, history_u: np.ndarray,
             history_y: np.ndarray, future_u: np.ndarray,
             device: torch.device) -> np.ndarray:
    model = model.to(device).eval()
    with torch.no_grad():
        z = model.encode_history(
            torch.as_tensor(history_u, dtype=torch.float32, device=device),
            torch.as_tensor(history_y, dtype=torch.float32, device=device))
        prediction = model.rollout(
            z, torch.as_tensor(future_u, dtype=torch.float32, device=device))
    result = prediction.cpu().numpy().astype(np.float64)
    if not np.isfinite(result).all():
        raise FloatingPointError("SUBNET checkpoint produced non-finite rollout")
    return result


def _integrate_pose(body: np.ndarray, initial_body: np.ndarray,
                    initial_pose: np.ndarray) -> np.ndarray:
    pose = initial_pose.astype(np.float64, copy=True)
    previous = initial_body.astype(np.float64, copy=True)
    path = np.empty((len(body), body.shape[1], 3), dtype=np.float64)
    for step in range(body.shape[1]):
        u, v, yaw_rate = previous.T
        dtheta = yaw_rate * DT_S
        moving = np.abs(yaw_rate) > 1.0e-7
        dx_body = u * DT_S
        dy_body = v * DT_S
        dx_body[moving] = (
            u[moving] * np.sin(dtheta[moving])
            + v[moving] * (np.cos(dtheta[moving]) - 1.0)
        ) / yaw_rate[moving]
        dy_body[moving] = (
            u[moving] * (1.0 - np.cos(dtheta[moving]))
            + v[moving] * np.sin(dtheta[moving])
        ) / yaw_rate[moving]
        cosine, sine = np.cos(pose[:, 2]), np.sin(pose[:, 2])
        pose[:, 0] += cosine * dx_body - sine * dy_body
        pose[:, 1] += sine * dx_body + cosine * dy_body
        pose[:, 2] = np.arctan2(np.sin(pose[:, 2] + dtheta),
                                np.cos(pose[:, 2] + dtheta))
        path[:, step] = pose
        previous = body[:, step]
    return path


def _bootstrap(per_run: dict[str, float], seed: int) -> dict[str, Any]:
    names = sorted(per_run)
    values = np.asarray([per_run[name] for name in names], dtype=np.float64)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(values),
                           size=(BOOTSTRAP_REPLICATES, len(values)))
    means = values[sampled].mean(axis=1)
    return {
        "run_macro_mean": float(values.mean()),
        "run_cluster_bootstrap_95pct_ci": [
            float(x) for x in np.quantile(means, (0.025, 0.975))],
        "independent_run_count": len(names),
        "per_run": {name: float(value)
                    for name, value in zip(names, values)},
    }


def _score(prediction: np.ndarray, truth: np.ndarray,
           predicted_pose: np.ndarray, truth_pose: np.ndarray,
           references: list[dict[str, Any]], train_std: np.ndarray,
           model_index: int) -> dict[str, Any]:
    result = {}
    for horizon in HORIZONS:
        by_run: dict[str, dict[str, float]] = {}
        for run_id in sorted({row["run_id"] for row in references}):
            indexes = [i for i, row in enumerate(references)
                       if row["run_id"] == run_id]
            state_error = prediction[indexes, :horizon] - truth[indexes, :horizon]
            rmse = np.sqrt(np.mean(state_error ** 2, axis=(0, 1)))
            pose_error = predicted_pose[indexes, :horizon] - truth_pose[indexes, :horizon]
            pose_error[..., 2] = np.arctan2(np.sin(pose_error[..., 2]),
                                            np.cos(pose_error[..., 2]))
            position = np.linalg.norm(pose_error[..., :2], axis=-1)
            by_run[run_id] = {
                **{f"{name}_rmse": float(value)
                   for name, value in zip(CHANNELS, rmse)},
                **{f"{name}_nrms": float(value)
                   for name, value in zip(CHANNELS, rmse / train_std)},
                "body_macro_channel_nrms": float(np.mean(rmse / train_std)),
                "position_trajectory_rmse_m": float(np.sqrt(np.mean(position ** 2))),
                "position_endpoint_rmse_m": float(np.sqrt(np.mean(position[:, -1] ** 2))),
                "heading_trajectory_rmse_rad": float(
                    np.sqrt(np.mean(pose_error[..., 2] ** 2))),
                "heading_endpoint_rmse_rad": float(
                    np.sqrt(np.mean(pose_error[:, -1, 2] ** 2))),
            }
        metrics = sorted(by_run[next(iter(by_run))])
        result[str(horizon)] = {
            "duration_s": horizon * DT_S,
            "per_run": by_run,
            "run_cluster_summary": {
                metric: _bootstrap(
                    {run: values[metric] for run, values in by_run.items()},
                    SEED + horizon + model_index * 100 + index)
                for index, metric in enumerate(metrics)
            },
        }
    return result


def evaluate(checkpoints: dict[str, Path], output: Path,
             device_name: str = "cuda") -> dict[str, Any]:
    for path in (BODY_DATASET, BODY_MANIFEST, STARTS):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    manifest = json.loads(BODY_MANIFEST.read_text(encoding="utf-8"))
    source_path = Path(manifest["source_dataset"])
    with np.load(BODY_DATASET, allow_pickle=False) as body, np.load(
            source_path, allow_pickle=False) as source:
        if _sha256(BODY_DATASET) != manifest["dataset_sha256"]:
            raise ValueError("frozen body dataset hash mismatch")
        run_splits = np.asarray(body["split"]).astype(str)
        train_outputs = np.asarray(body["outputs"], dtype=np.float64)[
            run_splits == "train"]
        train_std = train_outputs.std(axis=0, ddof=0)
        if not np.isfinite(train_std).all() or np.any(train_std <= 0):
            raise ValueError("training-only output standard deviations are invalid")
        windows = _fixed_windows(body, source, STARTS)
    (references, history_u, history_y, future_u, truth,
     initial_body, initial_pose, truth_pose) = windows
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but is not available")

    scores = {}
    checkpoint_hashes = {}
    for model_index, (name, path) in enumerate(sorted(checkpoints.items())):
        if not path.is_file():
            raise FileNotFoundError(path)
        model = SubnetBodyPlant.load_npz(path)
        prediction = _predict(model, history_u, history_y, future_u, device)
        if prediction.shape != truth.shape:
            raise ValueError(f"{name} generated an unexpected prediction shape")
        predicted_pose = _integrate_pose(prediction, initial_body, initial_pose)
        scores[name] = _score(prediction, truth, predicted_pose,
                              truth_pose, references, train_std, model_index)
        checkpoint_hashes[name] = _sha256(path)

    report = {
        "schema_version": 1,
        "study": "native SUBNET fixed-start recursive rollout on frozen whole-run validation cohort",
        "start_count": len(references),
        "independent_validation_run_count": len({r["run_id"] for r in references}),
        "validation_runs": sorted({r["run_id"] for r in references}),
        "training_output_std": dict(zip(CHANNELS, train_std.astype(float).tolist())),
        "body_dataset_sha256": manifest["dataset_sha256"],
        "source_dataset_sha256": _sha256(source_path),
        "frozen_start_sha256": _sha256(STARTS),
        "checkpoint_sha256": checkpoint_hashes,
        "future_logged_actuator_feedback_is_used": True,
        "future_truth_or_sensor_feedback_is_used": False,
        "horizons": {
            name: {str(h): data[str(h)] for h in HORIZONS}
            for name, data in scores.items()
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    checkpoints = {"candidate": args.candidate}
    if args.baseline:
        checkpoints["baseline"] = args.baseline
    report = evaluate(checkpoints, args.output, args.device)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "validation_runs": report["validation_runs"],
        "models": {
            name: {h: {
                metric: values["run_cluster_summary"][metric]["run_macro_mean"]
                for metric in ("body_macro_channel_nrms",
                               "u_rear_mps_rmse", "v_rear_mps_rmse",
                               "yaw_rate_rps_rmse", "position_trajectory_rmse_m",
                               "heading_trajectory_rmse_rad")
            } for h, values in item.items()}
            for name, item in report["horizons"].items()
        },
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
