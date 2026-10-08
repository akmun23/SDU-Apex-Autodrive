#!/usr/bin/env python3
"""One-shot scoring of frozen yaw teachers on explicit final-test captures.

This tool deliberately requires a dataset whose manifest contains only the
requested final_test runs. It performs no fitting or checkpoint selection.
Simulator rigid yaw rate is used only as the offline target; speed from that
stream is diagnostic metadata and never a model input.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import joblib
import numpy as np

try:
    import train_yaw_multihorizon_teacher as tree_teacher
    import train_yaw_gru_trajectory_teacher as gru_teacher
except ModuleNotFoundError:
    from tools.racing.specialists import (
        train_yaw_gru_trajectory_teacher as gru_teacher,
        train_yaw_multihorizon_teacher as tree_teacher,
    )


ROOT = tree_teacher.ROOT


def _admit_final_runs(dataset_dir: Path, expected_ids: list[str]
                      ) -> list[SimpleNamespace]:
    manifest_path = dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest.get("runs", [])
    if not rows or {str(row.get("effective_split", "")) for row in rows} != {
            "final_test"}:
        raise ValueError("dataset must contain final_test runs only")
    run_ids = [str(row.get("run_id", "")) for row in rows]
    if sorted(run_ids) != sorted(expected_ids) or len(set(run_ids)) != len(run_ids):
        raise ValueError("manifest run IDs do not match the explicit requested set")
    for row in rows:
        if (row.get("aborted")
                or row.get("reason") != "schedule complete"
                or not row.get("clean_stream_and_collision_gate")
                or int(row.get("timing_faults", -1)) != 0
                or any(int(value) != 0 for value in row.get("collisions", []))
                or row.get("quality_failures")
                or row.get("whole_bag_quality_failures")):
            raise ValueError(f"final-test run failed quality gate: {row['run_id']}")
    archive_path = dataset_dir / "openplane_dynamics.npz"
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    source = str(archive_path.relative_to(ROOT))
    return [SimpleNamespace(run_id=run_id, split="final_test", source=source)
            for run_id in run_ids]


def _metadata(capture, rows: np.ndarray) -> dict[str, np.ndarray]:
    sensors = capture.sensors[rows]
    rigid = capture.rigid[rows]
    return {
        "speed_mps_for_diagnosis_only": np.hypot(rigid[:, 7], rigid[:, 8]),
        "physical_steering_rad": sensors[:, 0],
        "steering_command_rad": sensors[:, 7],
        "steering_command_feedback_gap_rad": sensors[:, 7] - sensors[:, 0],
        "current_imu_yaw_rate_radps": sensors[:, 6],
    }


def _score_rows(capture, horizon: int, rows: np.ndarray,
                truth: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(
        truth, dtype=np.float64)
    metadata = _metadata(capture, rows)
    gap = np.abs(metadata["steering_command_feedback_gap_rad"])
    bands = ((0.0, 0.025), (0.025, 0.05), (0.05, 0.10), (0.10, np.inf))
    by_gap = {
        f"{low:g}..{high:g}rad": tree_teacher._metrics(
            error[(gap >= low) & (gap < high)])
        for low, high in bands
    }
    max_index = int(np.argmax(np.abs(error)))
    i = max_index
    worst = {
        "exported_frame_index": int(rows[i]),
        "horizon_ms": int(horizon * 25),
        "speed_mps_for_diagnosis_only": float(
            metadata["speed_mps_for_diagnosis_only"][i]),
        "physical_steering_rad": float(metadata["physical_steering_rad"][i]),
        "steering_command_rad": float(metadata["steering_command_rad"][i]),
        "steering_command_feedback_gap_rad": float(
            metadata["steering_command_feedback_gap_rad"][i]),
        "current_imu_yaw_rate_radps": float(
            metadata["current_imu_yaw_rate_radps"][i]),
        "truth_yaw_rate_radps": float(truth[i]),
        "predicted_yaw_rate_radps": float(prediction[i]),
        "signed_error_radps": float(error[i]),
    }
    return {
        "samples": int(len(error)),
        "overall": tree_teacher._metrics(error),
        "by_abs_command_feedback_gap": by_gap,
        "worst_sample": worst,
    }


def _score_gru(checkpoint_path: Path, captures) -> dict[str, Any]:
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu",
                            weights_only=False)
    model = gru_teacher.make_model().cpu()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    normalizers = checkpoint["normalizers"]
    results: dict[str, Any] = {}
    with torch.inference_mode():
        for series in captures:
            capture = tree_teacher._read_run(series)
            run = gru_teacher._make_run(series)
            if not len(run.starts):
                raise ValueError(f"{run.run_id}: no full history/horizon windows")
            predictions: list[np.ndarray] = []
            targets: list[np.ndarray] = []
            rows: list[np.ndarray] = []
            for offset in range(0, len(run.starts), gru_teacher.BATCH_SIZE):
                selected = run.starts[offset:offset + gru_teacher.BATCH_SIZE]
                past, future, target = gru_teacher.gather_windows(run, selected)
                past, future, target = gru_teacher._normalize(
                    past, future, target, normalizers)
                raw = model(torch.from_numpy(past), torch.from_numpy(future))
                prediction = (raw.numpy() * normalizers["yaw_scale"]
                              + normalizers["yaw_center"])
                predictions.append(prediction)
                targets.append(target * normalizers["yaw_scale"]
                                + normalizers["yaw_center"])
                rows.append(selected)
            prediction = np.concatenate(predictions)
            target = np.concatenate(targets)
            current_rows = np.concatenate(rows)
            results[run.run_id] = {
                str(horizon): _score_rows(
                    capture, horizon, current_rows,
                    target[:, horizon - 1], prediction[:, horizon - 1])
                for horizon in gru_teacher.HORIZONS
            }
    return results


def _score_trees(model_dir: Path, captures, variant: str
                 ) -> dict[str, Any]:
    if variant not in tree_teacher.HISTORY_VARIANTS:
        raise ValueError(f"unknown ExtraTrees history variant: {variant}")
    results: dict[str, Any] = {}
    for series in captures:
        capture = tree_teacher._read_run(series)
        run_result = {}
        for horizon in tree_teacher.HORIZONS:
            features, truth, meta = tree_teacher._build_examples(
                capture, horizon, tree_teacher.HISTORY_VARIANTS[variant])
            if not len(truth):
                raise ValueError(f"{series.run_id}: no scored rows at {horizon}")
            model_path = model_dir / f"yaw_teacher_{variant}_h{horizon:02d}.joblib"
            prediction = joblib.load(model_path).predict(features)
            run_result[str(horizon)] = _score_rows(
                capture, horizon, meta["frame_index"], truth, prediction)
        results[series.run_id] = run_result
    return results


def score(dataset_dir: Path, expected_ids: list[str],
          gru_checkpoint: Path | None, tree_dir: Path | None,
          tree_variant: str, output: Path) -> dict[str, Any]:
    captures = _admit_final_runs(dataset_dir, expected_ids)
    results: dict[str, Any] = {
        "title": "Frozen yaw-teacher final-test score",
        "dataset": str(dataset_dir.relative_to(ROOT)),
        "final_test_runs": sorted(expected_ids),
        "fitting_or_checkpoint_selection_performed": False,
        "simulator_truth_used_only_as_offline_target_and_diagnostic_metadata": True,
        "models": {},
    }
    if gru_checkpoint is not None:
        results["models"]["causal_gru_trajectory"] = _score_gru(
            gru_checkpoint, captures)
    if tree_dir is not None:
        results["models"][f"extratrees_{tree_variant}"] = _score_trees(
            tree_dir, captures, tree_variant)
    if not results["models"]:
        raise ValueError("provide at least one frozen model")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--run-id", action="append", required=True)
    parser.add_argument("--gru-checkpoint", type=Path)
    parser.add_argument("--tree-dir", type=Path)
    parser.add_argument("--tree-variant", default="history_1p6s",
                        choices=tuple(tree_teacher.HISTORY_VARIANTS))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    score(args.dataset_dir, args.run_id, args.gru_checkpoint, args.tree_dir,
          args.tree_variant, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
