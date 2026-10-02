#!/usr/bin/env python3
"""Score frozen plant candidates on identical unseen practice starts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from tools.vehicle_dynamics_learning import evaluate_free_running_plant
from tools.vehicle_dynamics_learning.build_practice_transfer_benchmark import (
    _run_rows,
)
from tools.vehicle_dynamics_learning.train_direct_sequence_teacher import (
    COM_X_M,
    _pose_rollout,
    _targets,
    _torch_model,
)
from tools.vehicle_dynamics_learning.train_nssm import _load_dataset, _torch
from tools.vehicle_dynamics_learning.train_rssm_teacher import _rssm_model


SCORE_HORIZONS_S = (0.25, 0.5, 0.75, 1.0, 2.0, 5.0)
STATE_NAMES = (
    "u_com_mps", "v_com_mps", "yaw_rate_rps", "steering_feedback_rad",
    "throttle_feedback_norm", "rear_left_surface_mps",
    "rear_right_surface_mps",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _registry_checkpoints(registry_path: Path) -> dict[str, dict[str, Any]]:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    entries = {item["role"]: item for item in registry["model_checkpoints"]}
    expected_roles = {
        "historical_gru": "historical broad GRU comparator",
        "rssm_256": "lead race-domain RSSM 2s context, hidden 256, latent 32",
        "rssm_128": "race-domain RSSM 2s context, hidden 128, latent 16",
        "direct": "race-domain direct 2s-context/5s-target comparator",
    }
    result = {}
    for name, role in expected_roles.items():
        item = entries[role]
        path = Path(item["path"])
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[2] / path
        if not path.is_file():
            path = Path.cwd() / item["path"]
        if not path.is_file():
            raise FileNotFoundError(f"frozen checkpoint missing: {item['path']}")
        digest = _sha256(path)
        if digest != item["sha256"]:
            raise ValueError(f"checkpoint hash changed: {path}")
        result[name] = {"path": path.resolve(), "sha256": digest,
                        "entry": item}
    return result


def _per_state_metrics(errors: np.ndarray) -> dict[str, dict[str, float]]:
    result = {}
    for column, name in enumerate(STATE_NAMES):
        values = errors[:, column]
        result[name] = {
            "rmse": float(np.sqrt(np.mean(values ** 2))),
            "bias": float(np.mean(values)),
            "p95_abs": float(np.quantile(np.abs(values), 0.95)),
        }
    return result


def _pose_error(prediction: np.ndarray, initial_state: np.ndarray,
                initial_pose: np.ndarray, truth_pose: np.ndarray
                ) -> tuple[np.ndarray, np.ndarray]:
    predicted_pose = _pose_rollout(prediction[:, :3], initial_state,
                                   initial_pose)
    error = predicted_pose - truth_pose
    error[:, 2] = np.arctan2(np.sin(error[:, 2]), np.cos(error[:, 2]))
    return predicted_pose, error


def _load_candidate(name: str, checkpoint_path: Path,
                    torch, nn, data: dict[str, Any], device):
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metadata = payload["metadata"]
    if metadata["feature_names"] != data["feature_names"]:
        raise ValueError(f"{name}: checkpoint feature layout does not match dataset")
    training_runs = set(map(str, metadata["training_runs"]))
    practice_ids = {str(value) for value in data["run_ids"]
                    if str(value).startswith("practice_unseen_model_validation_")}
    if training_runs.intersection(practice_ids):
        raise ValueError(f"{name}: fresh practice validation run appears in training metadata")

    if name.startswith("rssm_"):
        model_type = _rssm_model(
            torch, nn, int(metadata["hidden_size"]), int(metadata["latent_size"]),
            payload["x_mean"], payload["x_scale"], payload["y_mean"],
            payload["y_scale"])
    elif name == "direct":
        model_type = _torch_model(
            torch, nn, int(metadata["context_steps"]),
            int(metadata["future_steps"]), int(metadata["width"]),
            int(metadata["layer_count"]), int(metadata["head_count"]),
            float(metadata.get("dropout", 0.05)))
    elif name == "historical_gru":
        _, models, first, loaded_metadata, report = (
            evaluate_free_running_plant._load_models(
                checkpoint_path.parent, data, "cpu"))
        return {"kind": "gru", "payload": payload,
                "metadata": loaded_metadata, "models": models,
                "first": first, "report": report}
    else:
        raise ValueError(f"unknown frozen candidate {name}")

    model = model_type().to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return {"kind": name, "payload": payload, "metadata": metadata,
            "model": model}


def _predict_window(candidate: dict[str, Any], name: str, torch,
                    data: dict[str, Any], targets: np.ndarray,
                    start: int) -> np.ndarray:
    metadata = candidate["metadata"]
    if name.startswith("rssm_"):
        payload = candidate["payload"]
        x_mean = _as_numpy(payload["x_mean"])
        x_scale = _as_numpy(payload["x_scale"])
        y_mean = _as_numpy(payload["y_mean"])
        y_scale = _as_numpy(payload["y_scale"])
        context_steps = int(metadata["context_steps"])
        rollout_steps = int(metadata["rollout_steps"])
        context_start = start - context_steps + 1
        context_end = start + 1
        future_start = start + 1
        context = ((data["frames"][context_start:context_end]
                    - x_mean) / x_scale)
        commands = ((data["frames"][future_start:future_start + rollout_steps, 7:9]
                     - x_mean[7:9]) / x_scale[7:9])
        initial = ((targets[start, :7] - y_mean[:7]) / y_scale[:7])
        with torch.no_grad():
            normalized_prediction = candidate["model"](
                torch.as_tensor(context[None], dtype=torch.float32, device="cpu"),
                torch.as_tensor(commands[None], dtype=torch.float32, device="cpu"),
                initial_state=torch.as_tensor(initial[None], dtype=torch.float32,
                                              device="cpu"),
                sample_prior=False)[0].cpu().numpy()
        return normalized_prediction * y_scale + y_mean

    if name == "direct":
        payload = candidate["payload"]
        x_mean = _as_numpy(payload["x_mean"])
        x_scale = _as_numpy(payload["x_scale"])
        y_mean = _as_numpy(payload["y_mean"])
        y_scale = _as_numpy(payload["y_scale"])
        context_steps = int(metadata["context_steps"])
        future_steps = int(metadata["future_steps"])
        context = (data["frames"][start - context_steps + 1:start + 1]
                   - x_mean) / x_scale
        commands = ((data["frames"][start + 1:start + 1 + future_steps, 7:9]
                     - x_mean[7:9]) / x_scale[7:9])
        with torch.no_grad():
            normalized_prediction = candidate["model"](
                torch.as_tensor(context[None], dtype=torch.float32, device="cpu"),
                torch.as_tensor(commands[None], dtype=torch.float32, device="cpu"),
            )[0].cpu().numpy()
        return normalized_prediction * y_scale + y_mean

    if name == "historical_gru":
        history_steps = int(metadata["history_steps"])
        rollout_steps = int(metadata["rollout_steps"])
        frames = data["frames"][start - history_steps + 1:start + 1 + rollout_steps]
        dt = data["dt_s"][start - history_steps + 1:start + 1 + rollout_steps]
        states = evaluate_free_running_plant._free_rollout(
            torch, candidate["models"][0], frames, dt, history_steps,
            candidate["first"]["feature_mean"],
            candidate["first"]["feature_scale"])
        rear_state = states[history_steps:]
        com_state = rear_state.copy()
        com_state[:, 1] += COM_X_M * com_state[:, 2]
        return com_state
    raise ValueError(f"prediction path not implemented for {name}")


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _macro_run_summary(values_by_run: dict[str, float], seed: int) -> dict[str, Any]:
    values = np.asarray(list(values_by_run.values()), dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(10000, len(values)))
    means = np.mean(values[draws], axis=1)
    return {
        "independent_run_count": int(len(values)),
        "macro_run_mean": float(np.mean(values)),
        "run_min": float(np.min(values)),
        "run_max": float(np.max(values)),
        "run_cluster_bootstrap_95pct_ci": np.quantile(means, [0.025, 0.975]).tolist(),
        "per_run": dict(sorted(values_by_run.items())),
    }


def score(benchmark_path: Path, registry_path: Path, output_path: Path,
          extra_checkpoints: list[tuple[str, Path]] | None = None
          ) -> dict[str, Any]:
    benchmark_path = benchmark_path.resolve()
    registry_path = registry_path.resolve()
    output_path = output_path.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite score report: {output_path}")
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    if (benchmark.get("benchmark_id") != "practice_transfer_benchmark_v1"
            or benchmark.get("frozen") is not True
            or benchmark.get("training_or_checkpoint_selection_use") is not False):
        raise ValueError("input is not a frozen validation-only practice benchmark")
    dataset_path = Path(benchmark["dataset"])
    if _sha256(dataset_path) != benchmark["dataset_sha256"]:
        raise ValueError("benchmark source dataset changed after freezing")

    data = _load_dataset(dataset_path)
    targets = _targets(data)
    torch, nn = _torch()
    torch.set_num_threads(1)
    device = torch.device("cpu")
    checkpoints = _registry_checkpoints(registry_path)
    for name, checkpoint_path in extra_checkpoints or []:
        checkpoint_path = checkpoint_path.resolve()
        if not name.startswith("rssm_") or name in checkpoints:
            raise ValueError(
                f"extra checkpoint name must be a unique rssm_* name: {name}")
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        checkpoints[name] = {
            "path": checkpoint_path,
            "sha256": _sha256(checkpoint_path),
        }
    candidates = {
        name: _load_candidate(name, item["path"], torch, nn, data, device)
        for name, item in checkpoints.items()
    }
    windows = benchmark["windows"]
    expected_runs = set(map(str, benchmark["validation_reports"].keys()))
    dataset_run_ids = set(map(str, data["run_ids"]))
    mapped_runs = {f"practice_unseen_model_validation_20261001_{run_id.rsplit('_', 1)[-1]}"
                   for run_id in expected_runs}
    if not mapped_runs.issubset(dataset_run_ids):
        raise ValueError("benchmark validation runs are missing from scoring dataset")

    model_accumulators: dict[str, dict[str, Any]] = {}
    for name, candidate in candidates.items():
        per_run_horizon: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
            lambda: defaultdict(list))
        per_category_horizon: dict[str, dict[str, dict[str, list[dict[str, Any]]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list)))
        failures = []
        for window in windows:
            run_suffix = window["run_id"].rsplit("_", 1)[-1]
            source_run_id = f"practice_model_validation_{run_suffix}"
            if source_run_id not in expected_runs:
                raise ValueError(f"benchmark window has unregistered source run: {window}")
            start = int(window["global_start_index"])
            if (start != int(window["global_start_index"])
                    or int(data["packet_sequence"][start])
                    != int(window["packet_sequence"])):
                raise ValueError("benchmark start no longer maps to its frozen packet")
            prediction = _predict_window(candidate, name, torch, data, targets, start)
            available_steps = len(prediction)
            if name == "direct":
                available_steps = min(available_steps, 200)
            if name == "historical_gru":
                available_steps = min(available_steps, 200)
            truth_start = start + 1
            truth = targets[truth_start:truth_start + available_steps]
            initial = targets[start, :7]
            initial_pose = data["simulator_pose_xyyaw"][start]
            truth_pose = data["simulator_pose_xyyaw"][truth_start:truth_start + available_steps]
            if prediction.shape[1] < 7:
                raise ValueError(f"{name}: candidate returned fewer than 7 state channels")
            finite_steps = np.isfinite(prediction[:, :7]).all(axis=1)
            if not finite_steps.all():
                first_bad = int(np.flatnonzero(~finite_steps)[0])
                failures.append({"window_id": window["packet_sequence"],
                                 "run_id": window["run_id"],
                                 "first_nonfinite_step": first_bad + 1})

            horizon_values = []
            for horizon_s in SCORE_HORIZONS_S:
                steps = round(horizon_s / 0.025)
                if steps > available_steps:
                    continue
                if not finite_steps[:steps].all():
                    continue
                predicted_state = prediction[:steps, :7]
                target_state = truth[:steps, :7]
                state_errors = predicted_state - target_state
                _, pose_errors = _pose_error(
                    predicted_state, initial, initial_pose, truth_pose[:steps])
                row = {
                    "state": _per_state_metrics(state_errors),
                    "position_xy_rmse_m": np.sqrt(np.mean(
                        pose_errors[:, :2] ** 2, axis=0)).tolist(),
                    "heading_rmse_rad": float(np.sqrt(np.mean(
                        pose_errors[:, 2] ** 2))),
                }
                per_run_horizon[window["run_id"]][f"{horizon_s:g}s"].append(row)
                for category in window["categories"]:
                    per_category_horizon[category][window["run_id"]][
                        f"{horizon_s:g}s"].append(row)
                horizon_values.append(horizon_s)
            if not horizon_values:
                failures.append({"window_id": window["packet_sequence"],
                                 "run_id": window["run_id"],
                                 "reason": "no finite scored horizon"})

        run_reports: dict[str, dict[str, Any]] = {}
        summary: dict[str, Any] = {}
        for horizon in SCORE_HORIZONS_S:
            key = f"{horizon:g}s"
            run_state_error = {name: {} for name in STATE_NAMES}
            run_position_error = {"x": {}, "y": {}}
            run_heading_error = {}
            for run_id, horizons in per_run_horizon.items():
                rows = horizons.get(key, [])
                if not rows:
                    continue
                run_state = {}
                for channel in STATE_NAMES:
                    run_state[channel] = float(np.mean([
                        row["state"][channel]["rmse"] for row in rows]))
                    run_state_error[channel][run_id] = run_state[channel]
                run_position = np.mean([row["position_xy_rmse_m"] for row in rows], axis=0)
                run_position_error["x"][run_id] = float(run_position[0])
                run_position_error["y"][run_id] = float(run_position[1])
                run_heading_error[run_id] = float(np.mean([
                    row["heading_rmse_rad"] for row in rows]))
                run_reports.setdefault(run_id, {})[key] = {
                    "window_count": len(rows),
                    "state_rmse": run_state,
                    "position_xy_rmse_m": run_position.tolist(),
                    "heading_rmse_rad": run_heading_error[run_id],
                }
            channels = {channel: _macro_run_summary(values, 9300 + int(horizon * 100)
                       + channel_index)
                       for channel_index, (channel, values)
                       in enumerate(run_state_error.items()) if values}
            positions = {axis: _macro_run_summary(values, 9400 + int(horizon * 100)
                        + axis_index)
                         for axis_index, (axis, values)
                         in enumerate(run_position_error.items()) if values}
            heading = (_macro_run_summary(run_heading_error, 9500 + int(horizon * 100))
                       if run_heading_error else None)
            if channels:
                summary[key] = {
                    "state_rmse_by_channel": channels,
                    "position_xy_rmse_m": positions,
                    "heading_rmse_rad": heading,
                }

        category_reports: dict[str, Any] = {}
        for category, runs in per_category_horizon.items():
            category_reports[category] = {}
            for horizon in SCORE_HORIZONS_S:
                key = f"{horizon:g}s"
                per_run = {}
                for run_id, horizons in runs.items():
                    rows = horizons.get(key, [])
                    if rows:
                        per_run[run_id] = float(np.mean([
                            np.sqrt(np.mean([
                                row["state"][name]["rmse"] ** 2
                                for name in ("u_com_mps", "v_com_mps",
                                             "yaw_rate_rps")]))
                            for row in rows]))
                if per_run:
                    category_reports[category][key] = {
                        "normalized_body_state_score_not_defined": True,
                        "body_state_rmse_components_macro_by_run": per_run,
                        "independent_run_count": len(per_run),
                    }

        model_accumulators[name] = {
            "checkpoint": str(checkpoints[name]["path"]),
            "checkpoint_sha256": checkpoints[name]["sha256"],
            "architecture": candidate["metadata"].get("architecture"),
            "training_run_ids": candidate["metadata"]["training_runs"],
            "future_inputs": "recorded steering/throttle commands only",
            "future_truth_or_sensors_used_as_inputs": False,
            "deterministic_rssm_prior_mean": name.startswith("rssm_"),
            "window_count": len(windows),
            "nonfinite_or_unscored_windows": failures,
            "per_run": run_reports,
            "macro_run_summary": summary,
            "by_category": category_reports,
        }

    report = {
        "schema_version": 1,
        "benchmark_id": benchmark["benchmark_id"],
        "benchmark_path": str(benchmark_path),
        "benchmark_sha256": _sha256(benchmark_path),
        "dataset_path": str(dataset_path.resolve()),
        "dataset_sha256": benchmark["dataset_sha256"],
        "registry_path": str(registry_path),
        "independent_run_count": len(expected_runs),
        "window_count": len(windows),
        "score_horizons_s": list(SCORE_HORIZONS_S),
        "models": model_accumulators,
        "limitations": [
            "Fresh practice captures cover two independent runs, speeds only through about 8 m/s, and no >=0.40 rad steering starts.",
            "The 5 s direct/GRU scores use whole command traces; each RSSM is scored only through its trained rollout horizon.",
            "Pose metrics integrate each short predicted segment from simulator-truth initial pose; they are not full-lap drift estimates.",
            "Bootstrap intervals resample independent runs; with two runs, uncertainty remains coarse.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("registry", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--extra-checkpoint", action="append", default=[], metavar="NAME=PATH",
        help="add an RSSM candidate without changing the frozen registry")
    args = parser.parse_args()
    extras = []
    for value in args.extra_checkpoint:
        name, separator, path = value.partition("=")
        if not separator or not name or not path:
            parser.error("--extra-checkpoint must be NAME=PATH")
        extras.append((name, Path(path)))
    report = score(args.benchmark, args.registry, args.output, extras)
    print(json.dumps({
        "independent_runs": report["independent_run_count"],
        "windows": report["window_count"],
        "models": {
            name: {horizon: {
                channel: round(summary["macro_run_summary"][horizon][
                    "state_rmse_by_channel"][channel]["macro_run_mean"], 4)
                for channel in ("u_com_mps", "v_com_mps", "yaw_rate_rps")
                if channel in summary["macro_run_summary"].get(horizon, {})[
                    "state_rmse_by_channel"]}
                    for horizon in summary["macro_run_summary"]}
            for name, summary in report["models"].items()
        },
        "output": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
