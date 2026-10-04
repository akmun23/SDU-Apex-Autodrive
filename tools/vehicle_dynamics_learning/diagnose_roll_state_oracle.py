#!/usr/bin/env python3
"""Measure whether roll-state error limits the frozen plant's motion rollout.

The oracle arm deliberately injects future measured IMU roll/rate at each
transition and is diagnostic only. Body, wheel, actuator, latent, and pose
states remain recursively predicted. It must never be used as a deployable
plant or observer result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from tools.vehicle_dynamics_learning.compare_effective_race_teacher import (
    _build_inputs,
)
from tools.vehicle_dynamics_learning.diagnose_effective_race_residuals import (
    _command_rows,
)
from tools.vehicle_dynamics_learning.effective_race_teacher import (
    DT_S,
    append_roll_state,
    integrate_pose,
    physical_state_from_dataset,
    raw_encoder_history_features,
    wheel_innovation_history_features,
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
DEFAULT_OUTPUT = BASE / "roll_state_oracle_causal_test_20261004" / "roll_state_oracle.json"
HORIZON = 200
COMMAND_OFFSET = -1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _windows_from_report(report: dict[str, Any], data: dict[str, Any],
                         domain: str) -> list[dict[str, Any]]:
    run_lookup = {str(value): index for index, value in
                  enumerate(np.asarray(data["run_ids"]).astype(str))}
    source = report["domains"][domain]["windows"]
    windows = []
    for item in source:
        run_id = str(item["run_id"])
        if run_id not in run_lookup:
            raise ValueError(f"frozen start references unknown run {run_id}")
        windows.append({"run": run_lookup[run_id],
                        "run_id": run_id,
                        "start": int(item["start"])})
    return windows


def _window_rows(data: dict[str, Any], windows: list[dict[str, Any]],
                 horizon: int) -> list[dict[str, Any]]:
    bounds = np.asarray(data["bounds"], dtype=np.int64)
    seq_run = np.asarray(data["seq_run"], dtype=np.int64)
    valid = np.asarray(data["imu_attitude_valid"], dtype=bool)
    reset = data.get("sequence_reset_index")
    reset = None if reset is None else np.asarray(reset, dtype=np.int64)
    packet = np.asarray(data["packet_sequence"], dtype=np.int64)
    dt = np.asarray(data["dt_s"], dtype=np.float64)
    sequence_at = np.full(len(valid), -1, dtype=np.int64)
    for sequence, (begin, end) in enumerate(bounds):
        sequence_at[int(begin):int(end)] = sequence
    selected = []
    for row in windows:
        start, run = int(row["start"]), int(row["run"])
        left, right = start - 79, start + horizon
        if left < 0 or right >= len(valid):
            raise ValueError("frozen start lacks its full history/rollout")
        for edge in range(left, right):
            a, b = int(sequence_at[edge]), int(sequence_at[edge + 1])
            if a < 0 or b < 0:
                raise ValueError("frozen window crosses an unindexed data gap")
            if a != b:
                same_reset = (reset is None or reset[a] == reset[b])
                if (int(seq_run[a]) != run or int(seq_run[b]) != run
                        or not same_reset
                        or packet[edge + 1] - packet[edge] != 1
                        or not np.isclose(dt[edge + 1], DT_S,
                                          rtol=0.0, atol=1e-7)):
                    raise ValueError("frozen window crosses an unverified packet join")
            elif int(seq_run[a]) != run:
                raise ValueError("frozen window crosses into another run")
        # The oracle intervention must be defined at each source state row.
        if not np.all(valid[start:start + horizon + 1]):
            continue
        selected.append(row)
    return selected


def _history_features(model: Any, data: dict[str, Any]) -> np.ndarray | None:
    if model.include_raw_encoder_history:
        return raw_encoder_history_features(data)
    if model.include_wheel_innovation_history:
        return wheel_innovation_history_features(data)
    return None


def _rollouts(torch: Any, model: Any, data: dict[str, Any], state: np.ndarray,
              windows: list[dict[str, Any]], device: str,
              batch_size: int = 24
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    measured = np.asarray(data["imu_attitude_frames"], dtype=np.float32)[:, (0, 2)]
    raw_history = _history_features(model, data)
    base_states, no_roll_states, oracle_states = [], [], []
    base_poses, no_roll_poses = [], []
    frames = np.asarray(data["frames"], dtype=np.float32)
    for offset in range(0, len(windows), batch_size):
        batch = windows[offset:offset + batch_size]
        history, initial, delayed, _, truth, truth_pose = _build_inputs(
            data, state, batch, model.history_state_size, raw_history,
            rollout_steps=HORIZON)
        starts = np.asarray([int(row["start"]) for row in batch], dtype=np.int64)
        source_rows = starts[:, None] + np.arange(HORIZON, dtype=np.int64)[None, :]
        command_rows = _command_rows(source_rows, COMMAND_OFFSET)
        if np.any(command_rows < 0) or np.any(command_rows >= len(frames)):
            raise ValueError("aligned command rows leave the dataset")
        commands = frames[command_rows, 7:9]
        roll_rows = torch.as_tensor(measured[source_rows], dtype=torch.float32,
                                    device=device)
        history_t = torch.as_tensor(history, dtype=torch.float32, device=device)
        initial_t = torch.as_tensor(initial, dtype=torch.float32, device=device)
        delayed_t = torch.as_tensor(delayed, dtype=torch.float32, device=device)
        commands_t = torch.as_tensor(commands, dtype=torch.float32, device=device)
        with torch.no_grad():
            baseline, _, _, _ = model.rollout(
                initial_t, delayed_t, history_t, commands_t)
            original_roll_residual_mode = model.roll_residual_mode
            try:
                model.roll_residual_mode = False
                no_roll, _, _, _ = model.rollout(
                    initial_t, delayed_t, history_t, commands_t)
            finally:
                model.roll_residual_mode = original_roll_residual_mode

            # Oracle is intentionally noncausal: replace only the explicit
            # roll/rate state at each source row; keep latent and other plant
            # states recursive and never teacher-force them.
            oracle_state = initial_t
            command_delay = delayed_t
            latent = model.encode_history(history_t)
            oracle_sequence = []
            for step in range(HORIZON):
                transition_state = oracle_state.clone()
                transition_state[:, 7:9] = roll_rows[:, step]
                oracle_state, command_delay, latent, _, _ = model.transition(
                    transition_state, command_delay, latent, commands_t[:, step])
                oracle_sequence.append(oracle_state)
            oracle = torch.stack(oracle_sequence, dim=1)
            baseline_pose = integrate_pose(
                torch, baseline,
                torch.as_tensor(truth_pose[:, 0], dtype=torch.float32,
                                device=device), initial_t)
            no_roll_pose = integrate_pose(
                torch, no_roll,
                torch.as_tensor(truth_pose[:, 0], dtype=torch.float32,
                                device=device), initial_t)
        base_states.append(baseline.detach().cpu().numpy().astype(np.float64))
        no_roll_states.append(no_roll.detach().cpu().numpy().astype(np.float64))
        oracle_states.append(oracle.detach().cpu().numpy().astype(np.float64))
        base_poses.append(baseline_pose.detach().cpu().numpy().astype(np.float64))
        no_roll_poses.append(no_roll_pose.detach().cpu().numpy().astype(np.float64))
    # Pose integrates all arms separately from identical initial conditions.
    base = np.concatenate(base_states)
    no_roll = np.concatenate(no_roll_states)
    oracle = np.concatenate(oracle_states)
    pose_oracle = []
    for offset in range(0, len(windows), batch_size):
        batch = windows[offset:offset + batch_size]
        _, initial, _, _, _, truth_pose = _build_inputs(
            data, state, batch, model.history_state_size, raw_history,
            rollout_steps=HORIZON)
        count = len(batch)
        begin = offset
        end = offset + count
        with torch.no_grad():
            p = integrate_pose(
                torch,
                torch.as_tensor(oracle[begin:end], dtype=torch.float32,
                                device=device),
                torch.as_tensor(truth_pose[:, 0], dtype=torch.float32,
                                device=device),
                torch.as_tensor(initial, dtype=torch.float32, device=device))
        pose_oracle.append(p.detach().cpu().numpy().astype(np.float64))
    return (base, no_roll, oracle,
            np.stack((np.concatenate(base_poses), np.concatenate(no_roll_poses),
                      np.concatenate(pose_oracle)), axis=0))


def _metric_errors(predicted: np.ndarray, poses: np.ndarray,
                   truth: np.ndarray, truth_pose: np.ndarray
                   ) -> dict[str, np.ndarray]:
    error = predicted - truth
    pose_delta = poses - truth_pose
    heading = np.arctan2(np.sin(pose_delta[:, :, 2]),
                         np.cos(pose_delta[:, :, 2]))
    return {
        "position_2d_m": np.hypot(pose_delta[:, :, 0], pose_delta[:, :, 1]),
        "heading_rad": heading,
        "forward_speed_u_mps": error[:, :, 0],
        "lateral_speed_v_mps": error[:, :, 1],
        "body_speed_mps": (np.hypot(predicted[:, :, 0], predicted[:, :, 1])
                           - np.hypot(truth[:, :, 0], truth[:, :, 1])),
        "yaw_rate_rps": error[:, :, 2],
        "steering_feedback_rad": error[:, :, 3],
        "throttle_feedback_norm": error[:, :, 4],
        "rear_left_wheel_mps": error[:, :, 5],
        "rear_right_wheel_mps": error[:, :, 6],
        "rear_wheel_pair_mps": np.sqrt(np.mean(error[:, :, 5:7] ** 2, axis=2)),
        "roll_prediction_rad": error[:, :, 7],
        "roll_rate_prediction_rps": error[:, :, 8],
    }


def _score_arm(predicted: np.ndarray, pose: np.ndarray, truth: np.ndarray,
               truth_pose: np.ndarray, windows: list[dict[str, Any]],
               run_ids: np.ndarray, exclude_roll_scores: bool = False
               ) -> dict[str, Any]:
    metrics = _metric_errors(predicted, pose, truth, truth_pose)
    if exclude_roll_scores:
        metrics = {name: values for name, values in metrics.items()
                   if not name.startswith("roll_")}
    run_indices = np.asarray([int(row["run"]) for row in windows])
    by_run: dict[str, dict[str, float]] = {}
    for run in sorted(set(run_indices.tolist())):
        mask = run_indices == run
        by_run[str(run_ids[run])] = {
            name: float(np.sqrt(np.mean(values[mask] ** 2)))
            for name, values in metrics.items()
        }
    macro = {name: float(np.mean([values[name] for values in by_run.values()]))
             for name in metrics}
    return {"macro_run_rmse": macro, "per_run_rmse": by_run,
            "independent_runs": len(by_run), "window_count": len(windows)}


def _paired_improvement(base: dict[str, Any], treatment: dict[str, Any],
                        treatment_label: str
                        ) -> dict[str, Any]:
    metrics = {name: value for name, value in base["macro_run_rmse"].items()
               if name in treatment["macro_run_rmse"]}
    result = {}
    for name, base_rmse in metrics.items():
        treatment_rmse = treatment["macro_run_rmse"][name]
        base_runs = base["per_run_rmse"]
        treatment_runs = treatment["per_run_rmse"]
        run_deltas = {
            run: (base_runs[run][name] - treatment_runs[run][name])
            / max(base_runs[run][name], 1e-12)
            for run in base_runs
        }
        values = np.asarray(list(run_deltas.values()), dtype=np.float64)
        rng = np.random.default_rng(20261004 + len(name))
        draws = values[rng.integers(0, len(values), size=(10000, len(values)))]
        result[name] = {
            "baseline_macro_run_rmse": base_rmse,
            "treatment": treatment_label,
            "treatment_macro_run_rmse": treatment_rmse,
            "relative_improvement_fraction": (base_rmse - treatment_rmse)
                                              / max(base_rmse, 1e-12),
            "per_run_relative_improvement_fraction": run_deltas,
            "independent_run_cluster_bootstrap_95pct_ci":
                np.quantile(draws.mean(axis=1), (0.025, 0.975)).tolist(),
        }
    return result


def _domain(torch: Any, model: Any, data: dict[str, Any], state: np.ndarray,
            windows: list[dict[str, Any]], device: str) -> dict[str, Any]:
    windows = _window_rows(data, windows, HORIZON)
    if not windows:
        raise ValueError("no frozen windows have complete aligned IMU roll/rate")
    base, no_roll, oracle, poses = _rollouts(
        torch, model, data, state, windows, device)
    truth, truth_pose = [], []
    for row in windows:
        start = int(row["start"])
        truth.append(state[start + 1:start + HORIZON + 1])
        truth_pose.append(data["simulator_pose_xyyaw"][start + 1:start + HORIZON + 1])
    truth = np.asarray(truth, dtype=np.float64)
    truth_pose = np.asarray(truth_pose, dtype=np.float64)
    base_pose, no_roll_pose, oracle_pose = poses
    baseline_report = _score_arm(base, base_pose, truth, truth_pose, windows,
                                 np.asarray(data["run_ids"]).astype(str))
    no_roll_report = _score_arm(
        no_roll, no_roll_pose, truth, truth_pose, windows,
        np.asarray(data["run_ids"]).astype(str))
    oracle_report = _score_arm(oracle, oracle_pose, truth, truth_pose, windows,
                               np.asarray(data["run_ids"]).astype(str),
                               exclude_roll_scores=True)
    oracle_report["excluded_nonpredictive_scores"] = [
        "roll_prediction_rad", "roll_rate_prediction_rps"]
    oracle_report["excluded_score_reason"] = (
        "roll and rate are overwritten by measured values before every "
        "transition; the oracle arm cannot be scored as predicting them")
    return {
        "windows_with_complete_roll": len(windows),
        "baseline_recursive": baseline_report,
        "roll_acceleration_residual_disabled_at_inference": no_roll_report,
        "paired_change_when_disabling_roll_residual": _paired_improvement(
            baseline_report, no_roll_report, "roll residual disabled"),
        "measured_roll_rate_oracle_diagnostic_only": oracle_report,
        "paired_relative_improvement_fraction": _paired_improvement(
            baseline_report, oracle_report, "measured roll/rate oracle"),
        "interpretation_limit": (
            "The disabled-residual arm removes only the learned roll-conditioned "
            "acceleration correction; all other frozen weights and the internal "
            "roll prediction remain unchanged. The oracle replaces explicit roll "
            "and roll-rate state at each transition, so it is noncausal and only "
            "an upper-bound diagnostic."),
    }


def run(source_path: Path, output_path: Path, device: str) -> dict[str, Any]:
    report = json.loads(source_path.read_text(encoding="utf-8"))
    if (report.get("command_offset_frames_from_target_state_row") != COMMAND_OFFSET
            or report.get("future_truth_or_sensor_feedback_used_in_rollout") is not False
            or report.get("test_and_final_test_used") is not False):
        raise ValueError("source report failed frozen/no-leakage provenance")
    checkpoint = Path(report["candidate"])
    dataset_path = Path(report["training_dataset"])
    benchmark_path = Path(report["practice_benchmark"])
    for path in (checkpoint, dataset_path, benchmark_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if _sha256(checkpoint) != report["candidate_sha256"]:
        raise ValueError("frozen candidate hash changed")
    if _sha256(dataset_path) != report["training_dataset_sha256"]:
        raise ValueError("frozen dynamic dataset hash changed")
    if _sha256(benchmark_path) != report["practice_benchmark_sha256"]:
        raise ValueError("frozen practice manifest hash changed")
    if output_path.exists():
        previous = json.loads(output_path.read_text(encoding="utf-8"))
        same_experiment = (
            previous.get("purpose") ==
            "noncausal roll-state intervention to test whether predicted roll error limits motion rollout"
            and previous.get("checkpoint_sha256") == report["candidate_sha256"]
            and previous.get("source_residual_report_sha256") == _sha256(source_path))
        if not same_experiment:
            raise FileExistsError(
                f"refusing to overwrite unrelated diagnostic output {output_path}")

    torch, model, metadata = _load_model(checkpoint, device)
    if not model.include_roll_state or model.history_state_size != 7:
        raise ValueError("oracle diagnosis requires the frozen seven-channel history and 9-state roll plant")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    dynamic = _load_dataset(dataset_path)
    practice_benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    practice_path = Path(str(practice_benchmark["dataset"]).replace(
        "/workspace/", str(ROOT) + "/"))
    if _sha256(practice_path) != practice_benchmark["dataset_sha256"]:
        raise ValueError("frozen practice dataset hash changed")
    practice = _load_dataset(practice_path)
    dynamic_state = append_roll_state(dynamic, physical_state_from_dataset(
        dynamic, str(metadata.get("wheel_state_source", "filtered_odometry"))))
    practice_state = append_roll_state(practice, physical_state_from_dataset(
        practice, str(metadata.get("wheel_state_source", "filtered_odometry"))))
    domains = {}
    for name, data, state in (("dynamic_validation", dynamic, dynamic_state),
                              ("practice_transfer", practice, practice_state)):
        windows = _windows_from_report(report, data, name)
        domains[name] = _domain(torch, model, data, state, windows, device)
    result = {
        "schema_version": 1,
        "purpose": "noncausal roll-state intervention to test whether predicted roll error limits motion rollout",
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": _sha256(checkpoint),
        "source_residual_report": str(source_path.relative_to(ROOT)),
        "source_residual_report_sha256": _sha256(source_path),
        "dynamic_dataset_sha256": _sha256(dataset_path),
        "practice_dataset": str(practice_path.relative_to(ROOT)),
        "practice_dataset_sha256": _sha256(practice_path),
        "command_offset_frames_from_target_state_row": COMMAND_OFFSET,
        "horizon_s": HORIZON * DT_S,
        "future_measured_imu_used_only_in_oracle_arm": True,
        "baseline_future_measurements_used": False,
        "training_or_checkpoint_selection_performed": False,
        "production_or_simulator_modified": False,
        "domains": domains,
        "decision_rule": (
            "A material, same-direction reduction in pose/body errors across "
            "independent dynamic runs and both practice captures would justify "
            "a causal internal-roll model experiment. Otherwise do not spend "
            "another training run on roll."),
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
    print(json.dumps({
        "output": str(args.output.resolve()),
        "domains": {name: {
            "windows": value["windows_with_complete_roll"],
            "position_rmse_baseline": value["baseline_recursive"]["macro_run_rmse"]["position_2d_m"],
            "position_rmse_no_roll_residual": value["roll_acceleration_residual_disabled_at_inference"]["macro_run_rmse"]["position_2d_m"],
            "position_rmse_oracle": value["measured_roll_rate_oracle_diagnostic_only"]["macro_run_rmse"]["position_2d_m"],
            "position_relative_change_no_roll_residual": value["paired_change_when_disabling_roll_residual"]["position_2d_m"]["relative_improvement_fraction"],
            "position_relative_change_oracle": value["paired_relative_improvement_fraction"]["position_2d_m"]["relative_improvement_fraction"],
        } for name, value in result["domains"].items()},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
