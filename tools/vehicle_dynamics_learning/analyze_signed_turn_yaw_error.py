#!/usr/bin/env python3
"""Check whether recursive yaw-acceleration error is signed-turn dependent.

Groups use held-out measured source-state steering/speed only for post-hoc
stratification. Predictions remain command-only and recursively closed-loop.
Independent capture runs, not 25 ms samples, are the uncertainty units.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from tools.vehicle_dynamics_learning.diagnose_effective_race_residuals import (
    _acceleration_targets_with_contiguous_joins,
    _candidate_rollouts,
)
from tools.vehicle_dynamics_learning.effective_race_teacher import (
    physical_state_from_dataset,
)
from tools.vehicle_dynamics_learning.score_effective_race_teacher import (
    _load_model,
)
from tools.vehicle_dynamics_learning.train_nssm import _load_dataset


ROOT = Path(__file__).resolve().parents[2]
BASE = (ROOT / "live_runs/derived_dynamics_learning_20260928"
        / "full_modeling_reset_20261001/replacement_offline_sim_raceline_20261003"
        / "full_throttle_domain_v1/next_phase_after_2129427"
        / "history_context_sufficiency_v1")
DEFAULT_SOURCE = BASE / "effective_teacher_residual_internal_roll_offset_minus1_20261004" / "residual_attribution.json"
DEFAULT_OUTPUT = BASE / "signed_turn_yaw_error_20261004.json"
HORIZONS = {"0.025s": 0, "0.75s": 29, "2s": 79, "5s": 199}
STEERING_BINS = {
    "left_le_-0.10rad": lambda x: x <= -0.10,
    "near_center_abs_lt_0.10rad": lambda x: np.abs(x) < 0.10,
    "right_ge_0.10rad": lambda x: x >= 0.10,
}
SPEED_BINS = {
    "0_to_5mps": lambda x: x < 5.0,
    "5_to_8mps": lambda x: (x >= 5.0) & (x < 8.0),
    "8_to_12mps": lambda x: (x >= 8.0) & (x <= 12.0),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_dataset(path_text: str) -> Path:
    return Path(path_text.replace("/workspace/", str(ROOT) + "/")).resolve()


def _windows(source: dict[str, Any], data: dict[str, Any], domain: str
             ) -> list[dict[str, Any]]:
    run_lookup = {str(name): index for index, name in
                  enumerate(np.asarray(data["run_ids"]).astype(str))}
    result = []
    for row in source["domains"][domain]["windows"]:
        name = str(row["run_id"])
        if name not in run_lookup:
            raise ValueError(f"frozen window references unknown run {name}")
        result.append({"run": run_lookup[name], "start": int(row["start"]),
                       "run_id": name})
    return result


def _bootstrap(values: np.ndarray, seed: int) -> list[float] | None:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return None
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(10000, len(values)))]
    return np.quantile(draws.mean(axis=1), (0.025, 0.975)).tolist()


def _summarize_group(yaw_accel_error: np.ndarray, yaw_rate_error: np.ndarray,
                     sample_mask: np.ndarray, run_ids: np.ndarray,
                     window_runs: np.ndarray) -> dict[str, Any]:
    by_run = {}
    for run in sorted(set(window_runs.tolist())):
        selected = sample_mask & (window_runs[:, None] == run)
        n = int(selected.sum())
        if n:
            ya = yaw_accel_error[selected]
            yr = yaw_rate_error[selected]
            by_run[str(run_ids[run])] = {
                "transitions": n,
                "yaw_acceleration_bias_rps2": float(np.mean(ya)),
                "yaw_acceleration_rmse_rps2": float(np.sqrt(np.mean(ya ** 2))),
                "yaw_rate_bias_rps": float(np.mean(yr)),
                "yaw_rate_rmse_rps": float(np.sqrt(np.mean(yr ** 2))),
            }
    if not by_run:
        return {"transitions": 0, "independent_runs": 0,
                "per_run": {}, "macro_run": None}
    names = tuple(next(iter(by_run.values())).keys())[1:]
    macro = {name: float(np.mean([row[name] for row in by_run.values()]))
             for name in names}
    result = {"transitions": int(sum(row["transitions"] for row in by_run.values())),
              "independent_runs": len(by_run), "per_run": by_run,
              "macro_run": macro}
    result["run_cluster_bootstrap_95pct_ci"] = {
        name: _bootstrap(np.asarray([row[name] for row in by_run.values()]),
                         20261004 + len(name) + len(by_run))
        for name in names}
    return result


def _domain(torch: Any, model: Any, data: dict[str, Any],
            windows: list[dict[str, Any]], device: str,
            command_offset: int) -> dict[str, Any]:
    state = physical_state_from_dataset(
        data, str(model.metadata.get("wheel_state_source", "filtered_odometry")))
    predicted, acceleration, truth, source_rows, _ = _candidate_rollouts(
        torch, model, data, state, windows, device,
        command_offset_frames=command_offset)
    targets = _acceleration_targets_with_contiguous_joins(data, state)
    yaw_accel_truth = targets[source_rows, 2]
    yaw_accel_error = acceleration[:, :, 2] - yaw_accel_truth
    yaw_rate_error = predicted[:, :, 2] - truth[:, :, 2]
    speed = np.hypot(state[source_rows, 0], state[source_rows, 1])
    steering = state[source_rows, 3]
    run_ids = np.asarray(data["run_ids"]).astype(str)
    window_runs = np.asarray([int(row["run"]) for row in windows], dtype=np.int32)
    results = {}
    for horizon_name, step_index in HORIZONS.items():
        results[horizon_name] = {}
        speed_at = speed[:, step_index]
        steering_at = steering[:, step_index]
        accel_at = yaw_accel_error[:, step_index]
        rate_at = yaw_rate_error[:, step_index]
        for steering_name, steering_mask in STEERING_BINS.items():
            steering_keep = steering_mask(steering_at)
            for speed_name, speed_mask in SPEED_BINS.items():
                keep = steering_keep & speed_mask(speed_at)
                results[horizon_name][f"{steering_name}__{speed_name}"] = (
                    _summarize_group(
                        accel_at[:, None], rate_at[:, None], keep[:, None],
                        run_ids, window_runs))
        # Direction-only aggregate helps expose a sign imbalance without
        # silently pooling away the speed-specific table above.
        for steering_name, steering_mask in STEERING_BINS.items():
            keep = steering_mask(steering_at)
            results[horizon_name][steering_name] = _summarize_group(
                accel_at[:, None], rate_at[:, None], keep[:, None],
                run_ids, window_runs)
    return {
        "window_count": len(windows),
        "independent_run_ids": sorted(set(str(run_ids[r]) for r in window_runs)),
        "command_offset_frames_from_target_state_row": command_offset,
        "signed_steering_threshold_rad": 0.10,
        "speed_domain_mps": [0.0, 12.0],
        "yaw_residuals_by_rollout_horizon": results,
        "interpretation_limit": (
            "Steering/speed are measured held-out labels used only to stratify "
            "the recursive rollout after prediction. Run-cluster intervals "
            "are weak with fewer than three independent captures per cell."),
    }


def run(source_path: Path, output_path: Path, device: str) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if (source.get("command_offset_frames_from_target_state_row") != -1
            or source.get("future_truth_or_sensor_feedback_used_in_rollout") is not False
            or source.get("test_and_final_test_used") is not False):
        raise ValueError("residual source fails frozen provenance")
    checkpoint = Path(source["candidate"])
    dynamic_path = Path(source["training_dataset"])
    benchmark_path = Path(source["practice_benchmark"])
    if _sha256(checkpoint) != source["candidate_sha256"]:
        raise ValueError("frozen checkpoint hash changed")
    if _sha256(dynamic_path) != source["training_dataset_sha256"]:
        raise ValueError("frozen dynamic dataset hash changed")
    dynamic = _load_dataset(dynamic_path)
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    practice_path = _resolve_dataset(benchmark["dataset"])
    if _sha256(practice_path) != benchmark["dataset_sha256"]:
        raise ValueError("frozen practice dataset hash changed")
    torch, model, metadata = _load_model(checkpoint, device)
    model.metadata = metadata
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    domains = {}
    for domain, data in (("dynamic_validation", dynamic),
                         ("practice_transfer", _load_dataset(practice_path))):
        domains[domain] = _domain(
            torch, model, data, _windows(source, data, domain), device, -1)
    result = {
        "schema_version": 1,
        "purpose": "held-out signed-turn yaw error attribution for the frozen candidate",
        "candidate": str(checkpoint.relative_to(ROOT)),
        "candidate_sha256": source["candidate_sha256"],
        "source_residual_report": str(source_path.relative_to(ROOT)),
        "source_residual_report_sha256": _sha256(source_path),
        "dynamic_dataset_sha256": source["training_dataset_sha256"],
        "practice_dataset_sha256": benchmark["dataset_sha256"],
        "test_or_final_test_used": False,
        "future_truth_or_sensor_feedback_used_in_rollout": False,
        "domains": domains,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available()
                        else "cpu")
    args = parser.parse_args()
    result = run(args.source_report.resolve(), args.output.resolve(), args.device)
    print(json.dumps({"output": str(args.output.resolve()), "domains": {
        name: {"windows": value["window_count"],
               "runs": value["independent_run_ids"]}
        for name, value in result["domains"].items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
