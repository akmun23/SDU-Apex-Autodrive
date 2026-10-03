#!/usr/bin/env python3
"""Run the frozen, common plant-regression suite for one RSSM candidate.

The suite is intentionally one command and one immutable output directory:
practice common starts, paired braking/release fixtures, held-out 7.5 m/s
high-steer runs, frontier r03, and ordinary race-domain validation. Every
recursive replay is initialized from one past history/pose and receives only
future steering/throttle commands. Future measured state is used only as the
score target.

The candidate adapter is RSSM for the current baseline. The report schema and
suite boundaries are model-independent; when the replacement plant is added,
its loader will implement the same reset/step surface without changing which
held-out captures define regression.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from tools.vehicle_dynamics_learning import score_practice_transfer_benchmark
from tools.vehicle_dynamics_learning import score_rssm_free_running_capture
from tools.vehicle_dynamics_learning.offline_plant import (
    BODY_STATE_NAMES,
    load_rssm_teacher_plant,
)
from tools.vehicle_dynamics_learning.train_nssm import _load_dataset, _torch


REPO_ROOT = Path(__file__).resolve().parents[2]
SUITE_ROOT = REPO_ROOT / "live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001"
TRAINING_DATASET = REPO_ROOT / "live_runs/derived_dynamics_learning_20260928/plant_teacher_race_domain_moderate_braking_20261001/cooldown_2s/openplane_dynamics.npz"
BASELINE_REGISTRY = SUITE_ROOT / "baseline_registry.json"
PRACTICE_BENCHMARK = SUITE_ROOT / "practice_transfer_benchmark_v1.json"
BRAKING_FIXTURES = SUITE_ROOT / "braking_wheel_regression_fixtures_v1.json"
HIGHSTEER_DATASET = SUITE_ROOT / "highsteer_75_heldout_validation_source_20261002/openplane_dynamics.npz"
HIGHSTEER_MANIFEST = SUITE_ROOT / "highsteer_75_heldout_validation_source_20261002/manifest.json"
FRONTIER_DATASET = SUITE_ROOT / "steering_frontier_validation_r03_20261002/openplane_dynamics.npz"
FRONTIER_MANIFEST = SUITE_ROOT / "steering_frontier_validation_r03_20261002/manifest.json"
RACE_MANIFEST = REPO_ROOT / "live_runs/derived_dynamics_learning_20260928/race_domain_moderate_braking_source_20261001/manifest.json"
RACE_VALIDATION_RUNS = (
    "openplane_race_domain_validation_20261001_r04",
    "openplane_race_domain_validation_20261001_r05",
)
DT_S = 0.025


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _repo_path(value: str | Path) -> Path:
    """Map paths recorded inside the pinned /workspace container to checkout."""
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


def _wrap_angle(value: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(value), np.cos(value))


def _validate_dataset(data: dict[str, Any], label: str,
                      required_runs: set[str]) -> dict[str, int]:
    if data["simulator_pose_xyyaw"] is None:
        raise ValueError(f"{label}: dataset has no simulator pose truth")
    if not np.allclose(data["dt_s"], DT_S, rtol=0.0, atol=1e-7):
        raise ValueError(f"{label}: dataset does not use the fixed 25 ms period")
    ids = {str(value): index for index, value in enumerate(data["run_ids"])}
    missing = sorted(required_runs - ids.keys())
    if missing:
        raise ValueError(f"{label}: missing held-out run IDs: {missing}")
    wrong_split = sorted(run_id for run_id in required_runs
                         if str(data["splits"][ids[run_id]]) != "validation")
    if wrong_split:
        raise ValueError(f"{label}: runs are not validation-only: {wrong_split}")
    return ids


def _score_braking_fixtures(checkpoint: Path, training_dataset: Path,
                            fixtures_path: Path) -> dict[str, Any]:
    fixtures = json.loads(fixtures_path.read_text(encoding="utf-8"))
    if (fixtures.get("fixture_set") != "braking_wheel_regression_fixtures_v1"
            or fixtures.get("future_truth_or_sensor_inputs") is not False):
        raise ValueError("braking fixture contract is not the frozen command-only set")
    dataset_path = _repo_path(fixtures["dataset"])
    if _sha256(dataset_path) != fixtures["dataset_sha256"]:
        raise ValueError("braking fixture source dataset hash changed")
    data = _load_dataset(dataset_path)
    if data["simulator_pose_xyyaw"] is None:
        raise ValueError("braking source lacks simulator pose labels")
    plant = load_rssm_teacher_plant([checkpoint], training_dataset,
                                    device="cpu", max_supported_speed_mps=12.0)
    run_index = {str(value): index for index, value in enumerate(data["run_ids"])}
    details = []
    for event in fixtures["events"]:
        run_id = str(event["run_id"])
        if run_id not in run_index:
            raise ValueError(f"braking fixture references absent run {run_id}")
        index = run_index[run_id]
        if str(data["splits"][index]) != event["run_split"]:
            raise ValueError(f"{run_id}: braking fixture split label changed")
        bounds = [tuple(map(int, pair)) for sequence_id, pair
                  in enumerate(data["bounds"])
                  if int(data["seq_run"][sequence_id]) == index]
        history_start = int(event["brake_history_start_index"])
        brake_initial = int(event["brake_initial_state_index"])
        release_history_start = int(event["release_history_start_index"])
        release_initial = int(event["release_initial_state_index"])
        samples = []
        for phase, start, initial, command_start in (
                ("active_braking", history_start, brake_initial,
                 int(event["brake_command_index"])),
                ("brake_release", release_history_start, release_initial,
                 int(event["brake_release_command_index"]))):
            matching = [(lo, hi) for lo, hi in bounds
                        if lo <= start < initial < hi]
            if len(matching) != 1 or initial + 1 != command_start:
                raise ValueError(f"{run_id}/{event['fixture_name']}/{phase}: "
                                 "fixture indices no longer map to one sequence")
            lo, hi = matching[0]
            if (initial - start + 1 != plant.history_steps
                    or not np.allclose(data["dt_s"][start:hi], DT_S,
                                       rtol=0.0, atol=1e-7)
                    or np.any(np.diff(data["packet_sequence"][start:hi]) != 1)):
                raise ValueError(f"{run_id}/{event['fixture_name']}/{phase}: "
                                 "history/future is not packet-contiguous 40 Hz data")
            horizon_steps = [int(round(float(value) / DT_S))
                             for value in event["score_horizons_s"]]
            steps = min(max(horizon_steps), hi - command_start)
            if steps < max(horizon_steps):
                raise ValueError(f"{run_id}/{event['fixture_name']}/{phase}: "
                                 "full 2 s future fixture is unavailable")
            history = data["frames"][start:initial + 1]
            initial_pose = data["simulator_pose_xyyaw"][initial]
            plant.reset(history, initial_pose)
            predicted_state = []
            predicted_pose = []
            truth_state = data["frames"][command_start:command_start + steps, :7]
            truth_pose = data["simulator_pose_xyyaw"][command_start:command_start + steps]
            for row in range(command_start, command_start + steps):
                estimate = plant.step(float(data["frames"][row, 7]),
                                      float(data["frames"][row, 8]),
                                      float(data["dt_s"][row]))
                if not np.isfinite(estimate.state).all():
                    raise FloatingPointError(
                        f"{run_id}/{event['fixture_name']}/{phase}: non-finite plant state")
                predicted_state.append(estimate.state[3:].astype(np.float64))
                predicted_pose.append(estimate.state[:3].astype(np.float64))
            predicted_state = np.asarray(predicted_state)
            predicted_pose = np.asarray(predicted_pose)
            state_error = predicted_state - truth_state
            pose_error = predicted_pose - truth_pose
            pose_error[:, 2] = _wrap_angle(pose_error[:, 2])
            horizon_report = {}
            for horizon_s, steps_at_horizon in zip(
                    event["score_horizons_s"], horizon_steps):
                pos = pose_error[steps_at_horizon - 1]
                horizon_report[f"{float(horizon_s):g}s"] = {
                    "body_state_abs_error": np.abs(
                        state_error[steps_at_horizon - 1]).tolist(),
                    "position_error_m": float(np.linalg.norm(pos[:2])),
                    "heading_abs_error_rad": float(abs(pos[2])),
                    "speed_abs_error_mps": float(abs(
                        np.linalg.norm(predicted_state[steps_at_horizon - 1, :2])
                        - np.linalg.norm(truth_state[steps_at_horizon - 1, :2]))),
                }
            samples.append({
                "phase": phase,
                "truth_initialized_from": "past history ending at initial state, plus one pose anchor",
                "future_inputs": ["steering_command_rad", "throttle_command_norm"],
                "future_truth_or_sensor_inputs": False,
                "horizon_scores": horizon_report,
                "two_second_free_run": {
                    "body_state_names": list(BODY_STATE_NAMES),
                    "body_state_rmse": np.sqrt(np.mean(state_error ** 2, axis=0)).tolist(),
                    "position_radial_rmse_m": float(np.sqrt(np.mean(
                        np.sum(pose_error[:, :2] ** 2, axis=1)))),
                    "position_endpoint_m": float(np.linalg.norm(pose_error[-1, :2])),
                    "heading_rmse_rad": float(np.sqrt(np.mean(pose_error[:, 2] ** 2))),
                    "heading_endpoint_abs_rad": float(abs(pose_error[-1, 2])),
                },
            })
        details.append({
            "fixture_name": event["fixture_name"],
            "run_id": run_id,
            "run_split": event["run_split"],
            "speed_before_brake_mps": event["speed_before_brake_mps"],
            "steering_command_before_brake_rad": event[
                "steering_command_before_brake_rad"],
            "brake_and_release_replays": samples,
        })
    validation = [row for row in details if row["run_split"] == "validation"]
    training = [row for row in details if row["run_split"] == "train"]
    return {
        "fixture_set": fixtures["fixture_set"],
        "fixture_sha256": _sha256(fixtures_path),
        "source_dataset": str(dataset_path),
        "source_dataset_sha256": fixtures["dataset_sha256"],
        "fixture_count": len(details),
        "training_fixture_count": len(training),
        "validation_fixture_count": len(validation),
        "statistical_unit": fixtures["statistical_unit"],
        "acceptance_split": "validation; training fixtures are diagnostic only",
        "future_truth_or_sensor_inputs": False,
        "events": details,
    }


def _score_capture(checkpoint: Path, training_dataset: Path,
                   validation_dataset: Path, run_ids: list[str],
                   quality_manifest: Path, output_path: Path
                   ) -> dict[str, Any]:
    return score_rssm_free_running_capture.score(
        checkpoint, training_dataset, validation_dataset, run_ids,
        output_path, quality_manifest, "cpu", continuous_run=True,
        phase_onsets=False)


def run(checkpoint: Path, training_dataset: Path, output_dir: Path,
        candidate_name: str, practice_benchmark: Path = PRACTICE_BENCHMARK,
        baseline_registry: Path = BASELINE_REGISTRY,
        braking_fixtures: Path = BRAKING_FIXTURES) -> dict[str, Any]:
    checkpoint = checkpoint.resolve()
    training_dataset = _repo_path(training_dataset)
    practice_benchmark = _repo_path(practice_benchmark)
    baseline_registry = _repo_path(baseline_registry)
    braking_fixtures = _repo_path(braking_fixtures)
    output_dir = output_dir.resolve()
    if not checkpoint.is_file() or not training_dataset.is_file():
        raise FileNotFoundError("candidate checkpoint or training dataset is missing")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite regression output: {output_dir}")
    payload = _torch()[0].load(checkpoint, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata", {})
    if metadata.get("architecture") != "rssm_posterior_prior_teacher":
        raise ValueError("current canonical runner accepts the frozen RSSM adapter only")
    with np.load(training_dataset, allow_pickle=False) as archive:
        training_features = archive["feature_names"].astype(str).tolist()
    if metadata.get("feature_names") != training_features:
        raise ValueError("checkpoint and training dataset feature layouts differ")

    frozen = json.loads(practice_benchmark.read_text(encoding="utf-8"))
    if (frozen.get("benchmark_id") != "practice_transfer_benchmark_v1"
            or frozen.get("frozen") is not True
            or frozen.get("training_or_checkpoint_selection_use") is not False):
        raise ValueError("practice input is not the frozen validation-only benchmark")
    practice_data_path = _repo_path(frozen["dataset"])
    if _sha256(practice_data_path) != frozen["dataset_sha256"]:
        raise ValueError("frozen practice benchmark dataset hash changed")
    practice_run_ids = {
        f"practice_unseen_model_validation_20261001_{str(row['run_id']).rsplit('_', 1)[-1]}"
        for row in frozen["windows"]
    }
    training_overlap = sorted(set(map(str, metadata.get("training_runs", [])))
                              & practice_run_ids)
    if training_overlap:
        raise ValueError(f"candidate training overlaps practice validation: {training_overlap}")

    high_ids = {"openplane_highsteer_75_validation_r01",
                 "openplane_highsteer_75_validation_r02"}
    high_data = _load_dataset(HIGHSTEER_DATASET)
    _validate_dataset(high_data, "high-steer", high_ids)
    del high_data
    frontier_id = {"openplane_steering_frontier_validation_r03"}
    frontier_data = _load_dataset(FRONTIER_DATASET)
    _validate_dataset(frontier_data, "steering frontier", frontier_id)
    del frontier_data
    race_data = _load_dataset(TRAINING_DATASET)
    _validate_dataset(race_data, "race-domain", set(RACE_VALIDATION_RUNS))
    del race_data

    # Prove held-out whole runs are disjoint from checkpoint training metadata.
    heldout_ids = high_ids | frontier_id | set(RACE_VALIDATION_RUNS)
    overlap = sorted(set(map(str, metadata.get("training_runs", []))) & heldout_ids)
    if overlap:
        raise ValueError(f"candidate training overlaps held-out suite runs: {overlap}")

    output_dir.mkdir(parents=True)
    hashes = {
        "checkpoint": _sha256(checkpoint),
        "training_dataset": _sha256(training_dataset),
        "practice_benchmark": _sha256(practice_benchmark),
        "practice_dataset": frozen["dataset_sha256"],
        "highsteer_dataset": _sha256(HIGHSTEER_DATASET),
        "highsteer_manifest": _sha256(HIGHSTEER_MANIFEST),
        "frontier_dataset": _sha256(FRONTIER_DATASET),
        "frontier_manifest": _sha256(FRONTIER_MANIFEST),
        "race_domain_dataset": _sha256(TRAINING_DATASET),
        "race_domain_manifest": _sha256(RACE_MANIFEST),
        "braking_fixtures": _sha256(braking_fixtures),
    }

    # Practice transfer keeps the established 135-start scorer and comparator
    # set; append a candidate only when it is not already the frozen h256/z32.
    checkpoint_index = score_practice_transfer_benchmark._registry_checkpoints(
        baseline_registry)
    lead_hash = checkpoint_index["rssm_256"]["sha256"]
    extras = []
    if hashes["checkpoint"] != lead_hash:
        if not candidate_name.startswith("rssm_") or candidate_name in checkpoint_index:
            raise ValueError("a non-baseline RSSM candidate needs a unique rssm_* name")
        extras.append((candidate_name, checkpoint))
    practice_path = output_dir / "practice_transfer.json"
    practice_report = score_practice_transfer_benchmark.score(
        practice_benchmark, baseline_registry, practice_path, extras)
    scored_candidate = "rssm_256" if not extras else candidate_name
    if scored_candidate not in practice_report["models"]:
        raise RuntimeError("candidate was not emitted by the practice transfer scorer")

    high_path = output_dir / "highsteer_75_free_run.json"
    high_report = _score_capture(
        checkpoint, training_dataset, HIGHSTEER_DATASET,
        sorted(high_ids), HIGHSTEER_MANIFEST, high_path)
    frontier_path = output_dir / "steering_frontier_r03_free_run.json"
    frontier_report = _score_capture(
        checkpoint, training_dataset, FRONTIER_DATASET,
        sorted(frontier_id), FRONTIER_MANIFEST, frontier_path)
    race_path = output_dir / "race_domain_validation_free_run.json"
    race_report = _score_capture(
        checkpoint, training_dataset, TRAINING_DATASET,
        list(RACE_VALIDATION_RUNS), RACE_MANIFEST, race_path)
    braking_report = _score_braking_fixtures(
        checkpoint, training_dataset, braking_fixtures)
    braking_path = output_dir / "braking_wheel_fixtures.json"
    braking_path.write_text(json.dumps(braking_report, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")

    report = {
        "schema_version": 1,
        "suite_id": "replacement_offline_plant_regression_v1",
        "candidate_backend": "rssm_prior_mean_command_only",
        "candidate_name": scored_candidate,
        "candidate_checkpoint": str(checkpoint),
        "candidate_checkpoint_sha256": hashes["checkpoint"],
        "candidate_training_runs": metadata.get("training_runs", []),
        "future_inputs": ["steering_command_rad", "throttle_command_norm"],
        "future_truth_or_sensor_inputs": False,
        "test_and_final_test_used": False,
        "input_sha256": hashes,
        "primary_statistical_unit": "independent whole run; never individual windows/events",
        "suites": {
            "practice_benchmark": {
                "path": str(practice_path),
                "report_sha256": _sha256(practice_path),
                "independent_run_count": practice_report["independent_run_count"],
                "common_start_count": practice_report["window_count"],
                "candidate_metrics": practice_report["models"][scored_candidate],
            },
            "braking_fixtures": {
                "path": str(braking_path),
                "report_sha256": _sha256(braking_path),
                "validation_whole_run_count": 1,
                "fixture_count": braking_report["validation_fixture_count"],
                "acceptance_split": "validation only; paired train events diagnostic",
            },
            "highsteer_75": {
                "path": str(high_path),
                "report_sha256": _sha256(high_path),
                "independent_run_count": high_report["independent_validation_run_count"],
                "run_ids": high_report["validation_run_ids"],
            },
            "steering_frontier_r03": {
                "path": str(frontier_path),
                "report_sha256": _sha256(frontier_path),
                "independent_run_count": frontier_report["independent_validation_run_count"],
                "run_ids": frontier_report["validation_run_ids"],
            },
            "ordinary_race_domain": {
                "path": str(race_path),
                "report_sha256": _sha256(race_path),
                "independent_run_count": race_report["independent_validation_run_count"],
                "run_ids": race_report["validation_run_ids"],
            },
        },
        "limitations": [
            "Current adapter scores frozen RSSM prior-mean plants; EDSSM must be added to this same suite before it can be considered a replacement.",
            "The validation cohorts are the already frozen captures; they do not prove sufficiency of upcoming dynamic-coupled bands.",
            "The practice set is a fixed common-start short-horizon benchmark; full-lap free-run is measured only on the listed whole-run cohorts.",
        ],
    }
    report_path = output_dir / "suite_report.json"
    report["report_path"] = str(report_path)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-dataset", type=Path, default=TRAINING_DATASET)
    parser.add_argument("--candidate-name", default="rssm_candidate",
                        help="unique rssm_* name when checkpoint differs from frozen h256/z32")
    parser.add_argument("--practice-benchmark", type=Path, default=PRACTICE_BENCHMARK)
    parser.add_argument("--baseline-registry", type=Path, default=BASELINE_REGISTRY)
    parser.add_argument("--braking-fixtures", type=Path, default=BRAKING_FIXTURES)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="new persistent directory; existing output is never overwritten")
    args = parser.parse_args()
    report = run(args.checkpoint, args.training_dataset, args.output_dir,
                 args.candidate_name, args.practice_benchmark,
                 args.baseline_registry, args.braking_fixtures)
    print(json.dumps({
        "suite": report["suite_id"],
        "candidate": report["candidate_name"],
        "runs": {name: row.get("independent_run_count",
                               row.get("validation_whole_run_count"))
                 for name, row in report["suites"].items()},
        "output": report["report_path"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
