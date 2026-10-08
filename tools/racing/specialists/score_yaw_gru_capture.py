#!/usr/bin/env python3
"""Score a frozen yaw-trajectory GRU on one clean, held-out capture.

The selected capture must be admitted as a whole-run validation run. The
checkpoint's sibling report is checked to prevent scoring a training capture.
This evaluates a direct causal yaw forecast; it does not claim recursive plant
accuracy or runtime readiness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from train_yaw_gru_trajectory_teacher import (
        BATCH_SIZE,
        HORIZONS,
        _discover_series_with_safe_mixed_archives,
        _evaluate,
        _make_run,
        _normalize,
        _read_run,
        gather_windows,
        make_model,
    )
except ModuleNotFoundError:
    from tools.racing.specialists.train_yaw_gru_trajectory_teacher import (
        BATCH_SIZE,
        HORIZONS,
        _discover_series_with_safe_mixed_archives,
        _evaluate,
        _make_run,
        _normalize,
        _read_run,
        gather_windows,
        make_model,
    )


def _regime_metrics(model, run, capture, normalizers: dict[str, np.ndarray]
                    ) -> dict[str, Any]:
    """Break out errors by current truth speed and measured steering.

    Truth speed is used only to label diagnostic bins after inference; the
    predictor receives the same sensor/history/command inputs as in training.
    """
    speed_edges = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                   8.0, 10.0, 12.0, float("inf"))
    steering_edges = (0.0, 0.1, 0.2, 0.35, 0.525, float("inf"))
    starts = run.starts
    current_speed = np.hypot(capture.rigid[starts, 0], capture.rigid[starts, 1])
    current_steering = np.abs(capture.sensors[starts, 0])
    accum: dict[int, dict[str, list[np.ndarray]]] = {
        horizon: {} for horizon in HORIZONS}
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(starts), BATCH_SIZE):
            selected = starts[offset:offset + BATCH_SIZE]
            past, future, targets = gather_windows(run, selected)
            past, future, targets = _normalize(
                past, future, targets, normalizers)
            prediction = model(torch.from_numpy(past), torch.from_numpy(future))
            predicted_yaw = (prediction.numpy() * normalizers["yaw_scale"]
                             + normalizers["yaw_center"])
            target_yaw = (targets * normalizers["yaw_scale"]
                          + normalizers["yaw_center"])
            local_speed = np.hypot(capture.rigid[selected, 0],
                                   capture.rigid[selected, 1])
            local_steering = np.abs(capture.sensors[selected, 0])
            speed_bin = np.searchsorted(speed_edges, local_speed,
                                        side="right") - 1
            steering_bin = np.searchsorted(steering_edges, local_steering,
                                           side="right") - 1
            for horizon in HORIZONS:
                error = np.abs(predicted_yaw[:, horizon - 1]
                               - target_yaw[:, horizon - 1])
                for speed_index in np.unique(speed_bin):
                    for steering_index in np.unique(steering_bin):
                        mask = ((speed_bin == speed_index)
                                & (steering_bin == steering_index))
                        if not np.any(mask):
                            continue
                        speed_label = (
                            f"{speed_edges[speed_index]:g}.."
                            f"{speed_edges[speed_index + 1]:g}")
                        steering_label = (
                            f"{steering_edges[steering_index]:g}.."
                            f"{steering_edges[steering_index + 1]:g}")
                        key = f"{speed_label}|{steering_label}"
                        accum[horizon].setdefault(key, []).append(error[mask])

    result: dict[str, Any] = {}
    for horizon in HORIZONS:
        rows: dict[str, Any] = {}
        for key, chunks in sorted(accum[horizon].items()):
            error = np.concatenate(chunks)
            rows[key] = {
                "samples": int(len(error)),
                "fraction_abs_error_at_most_0p1_radps": float(
                    np.mean(error <= 0.1)),
                "count_abs_error_over_0p1_radps": int(np.count_nonzero(
                    error > 0.1)),
                "p95_abs_error_radps": float(np.quantile(error, 0.95)),
                "rmse_radps": float(np.sqrt(np.mean(error ** 2))),
                "max_abs_error_radps": float(np.max(error)),
            }
        result[str(horizon * 25)] = rows
    return result


def _validate_report_for_score(report: dict[str, Any], run_id: str,
                               allow_running_checkpoint: bool) -> bool:
    """Return whether this is provisional; never score a selection run."""
    status = report.get("status")
    model_selection_runs = (set(report.get("training_runs", []))
                            | set(report.get("validation_runs", [])))
    if run_id in model_selection_runs:
        raise ValueError(
            f"refusing to score a training or checkpoint-selection run {run_id}")
    if status == "complete":
        return False
    if (status == "running" and allow_running_checkpoint
            and isinstance(report.get("best_epoch"), int)
            and report["best_epoch"] > 0):
        return True
    raise ValueError("checkpoint report must be complete; a running report "
                     "requires --allow-running-checkpoint and a saved best epoch")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def score(checkpoint_path: Path, run_id: str, output_path: Path, *,
          allow_running_checkpoint: bool = False) -> dict[str, Any]:
    checkpoint_path = checkpoint_path.resolve()
    output_path = output_path.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")

    report_path = checkpoint_path.with_name(
        "yaw_gru_trajectory_teacher_report.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    provisional_checkpoint = _validate_report_for_score(
        report, run_id, allow_running_checkpoint)

    series, source_audit = _discover_series_with_safe_mixed_archives()
    matches = [item for item in series if item.run_id == run_id]
    if len(matches) != 1:
        raise ValueError(f"expected one clean admitted run {run_id}, found {len(matches)}")
    selected = matches[0]
    if selected.split != "validation":
        raise ValueError(f"run must be whole-run validation, got {selected.split!r}")

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    torch.set_num_threads(1)
    model = make_model().cpu()
    model.load_state_dict(payload["model_state_dict"])
    capture = _read_run(selected)
    run = _make_run(selected)
    if not len(run.starts):
        raise ValueError(f"run {run_id} has no eligible history/forecast windows")
    metrics = _evaluate(model, [run], payload["normalizers"],
                        torch.device("cpu"), stride=1)
    result = {
        "title": "Held-out whole-run yaw GRU score",
        "status": "complete",
        "run_id": run_id,
        "split": selected.split,
        "source_archive": selected.source,
        "eligible_windows": int(len(run.starts)),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "checkpoint_training_report_status": report.get("status"),
        "checkpoint_best_epoch": report.get("best_epoch"),
        "provisional_checkpoint": provisional_checkpoint,
        "future_sensor_or_truth_inputs_used": False,
        "target_is_simulator_yaw_rate": True,
        "metrics": metrics,
        "diagnostic_regime_metrics_by_horizon": _regime_metrics(
            model, run, capture, payload["normalizers"]),
        "regime_labels": {
            "speed": "simulator truth magnitude at current sample; diagnostics only",
            "steering": "absolute measured steering feedback at current sample",
            "truth_values_used_as_model_inputs": False,
        },
        "source_audit": source_audit,
        "limitations": [
            "Direct one-shot yaw-rate forecast only; not a recursively closed vehicle plant.",
            "Scores do not establish odometry or MPC readiness.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_json_safe(result), indent=2,
                                      sort_keys=True) + "\n",
                           encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-running-checkpoint", action="store_true",
        help="score the saved best epoch from an active trainer; result is explicitly provisional")
    args = parser.parse_args()
    result = score(args.checkpoint, args.run_id, args.output,
                   allow_running_checkpoint=args.allow_running_checkpoint)
    print(json.dumps({
        "run_id": result["run_id"],
        "eligible_windows": result["eligible_windows"],
        "score_run_macro_rmse_radps": result["metrics"]["score_run_macro_rmse_radps"],
        "horizons": result["metrics"]["run_macro_rmse_radps"],
        "output": str(args.output.resolve()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
