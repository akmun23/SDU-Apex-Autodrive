#!/usr/bin/env python3
"""Compare MPC's next solve after either exact-validated 75 ms recovery prefix.

This development-only replay branches the recorded saved state through the
candidate or retained-plan prefix with the exact source plant, then runs the
same cold MPC replay from each predicted endpoint. It never publishes a live
command and is not evidence about Unity's unobserved future.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any

from tools import audit_mpc_trajectory_feasibility as audit
from tools import evaluate_mpc_bag as evaluator
from tools import search_mpc_source_plant_continuation as search


STATE_HEADER = (
    "stamp_ns", "progress_m", "speed_ceiling_mps", *search.STATE_COLUMNS,
    "suspension_history_initialized", "yaw_residual_radps",
    "yaw_residual_valid", *search.PLANT_STATE_COLUMNS,
)

REPLAY_ARGUMENTS = [
    "7", "0", "0", "1.6", "0.001", "0", "0", "0", "32", "10",
    "0.02", "independent", "cold", "8", "100", "0.25", "1", "1",
    "0.10", "0.10", "0.05", "0.05", "7", "2000", "10", "1", "3",
    "1", "4000", "1500", "7.5", "60", "25", "5", "7.8125", "3",
    "5", "0.1", "2000", "100",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def branch_diagnostic(diag: dict[str, Any], stage: dict[str, Any]) -> dict[str, Any]:
    if len(stage.get("mpc_state", [])) != 10:
        raise ValueError("source rollout trace is missing the 10-state MPC vector")
    if len(stage.get("plant_internal_state", [])) != 22:
        raise ValueError("source rollout trace is missing 22 hidden plant states")
    if not math.isfinite(float(stage["progress_m"])):
        raise ValueError("predicted branch progress is non-finite")
    return {
        "stamp_ns": int(diag.get("source_stamp_ns", 0)) +
                    int(stage["stage"]) * 25_000_000,
        "progress_m": float(stage["progress_m"]),
        "speed_ceiling_mps": float(
            diag.get("active_speed_ceiling_mps", 16.0)),
        "state": [float(value) for value in stage["mpc_state"]],
        "plant_internal_state": [
            float(value) for value in stage["plant_internal_state"]],
        "suspension_history_initialized": int(
            stage["suspension_history_initialized"]),
        # No new sensor correction is observed in this counterfactual branch.
        "yaw_residual_radps": 0.0,
        "yaw_residual_valid": 0,
        "plant_state_flags": [
            0, int(stage["suspension_history_initialized"]), 0],
        "status": "predicted_after_exact_75ms_prefix",
    }


def saved_state_row(diag: dict[str, Any]) -> list[float]:
    row = [
        float(diag["stamp_ns"]), float(diag["progress_m"]),
        float(diag["speed_ceiling_mps"]), *diag["state"],
        float(diag["suspension_history_initialized"]),
        float(diag["yaw_residual_radps"]), float(diag["yaw_residual_valid"]),
        *diag["plant_internal_state"],
    ]
    if len(row) != len(STATE_HEADER) or not all(math.isfinite(v) for v in row):
        raise ValueError("invalid saved MPC state row")
    return row


def get_prefix_controls(diag: dict[str, Any]) -> tuple[list[list[float]],
                                                        list[list[float]]]:
    quality = diag.get("quality_aware_prefix_selection", {})
    if quality.get("attempted"):
        candidate_logged = quality.get("candidate_prefix_controls")
        prior_logged = quality.get("prior_prefix_controls")
        if (isinstance(candidate_logged, list) and len(candidate_logged) >= 3 and
                isinstance(prior_logged, list) and len(prior_logged) >= 3):
            return (
                [[float(u[0]), float(u[1])] for u in candidate_logged[:3]],
                [[float(u[0]), float(u[1])] for u in prior_logged[:3]],
            )

    candidate = diag.get("candidate_controls")
    snapshot = diag.get("certified_plan_snapshot")
    if not isinstance(snapshot, dict):
        snapshot = diag.get("reconstructed_certified_plan_snapshot")
    if not isinstance(candidate, list) or len(candidate) != 30:
        raise ValueError("candidate full sequence is unavailable")
    if not isinstance(snapshot, dict) or not snapshot.get("valid"):
        raise ValueError("certified prior plan snapshot is unavailable")
    prior = snapshot.get("controls")
    cursor = int(snapshot.get("cursor", -1))
    # The diagnostic snapshot is emitted after the publication decision. A
    # candidate-prefix publication advances the cursor and replaces the old
    # plan action now at `cursor`; a same-state counterfactual compares against
    # that action. A rejected cycle publishes nothing, so its next retained
    # action is `cursor + 1`.
    prior_start = (cursor if diag.get("status") == "accepted_candidate_prefix"
                   else cursor + 1)
    if not isinstance(prior, list) or not 0 <= prior_start < len(prior):
        raise ValueError("certified prior-plan cursor is invalid")
    candidate_prefix = [[float(u[0]), float(u[1])] for u in candidate[:3]]
    prior_prefix = [[float(u[0]), float(u[1])]
                    for u in (prior[min(prior_start + i, len(prior) - 1)]
                              for i in range(3))]
    return candidate_prefix, prior_prefix


def has_candidate_prior_prefix_comparison(diag: dict[str, Any]) -> bool:
    if diag.get("status") == "accepted_candidate_prefix":
        return True
    quality = diag.get("quality_aware_prefix_selection", {})
    return bool(quality.get("attempted"))


def run_replay(binary: Path, trajectory: Path, states_path: Path,
               output_path: Path, stderr_path: Path) -> list[dict[str, str]]:
    command = [str(binary), str(trajectory), str(states_path), *REPLAY_ARGUMENTS]
    env = os.environ.copy()
    env.pop("MPC_REPLAY_NOMINAL_SEED_CSV", None)
    env.pop("MPC_REPLAY_DIAG_ROW", None)
    result = subprocess.run(command, text=True, capture_output=True,
                            env=env, check=False)
    output_path.write_text(result.stdout)
    stderr_path.write_text(result.stderr)
    if result.returncode:
        raise RuntimeError(f"MPC replay failed; see {stderr_path}")
    with output_path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--replay-binary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not args.output_dir.is_dir() or any(args.output_dir.iterdir()):
        parser.error("output directory must already exist and be empty")
    for source in (args.bag, args.trajectory, args.library, args.replay_binary):
        if not source.exists():
            parser.error(f"input does not exist: {source}")

    bag = evaluator.read_bag(args.bag)
    diagnostics = [dict(record.get("json", {}),
                        bag_time=record.get("bag_time"))
                   for record in bag.get(evaluator.DIAGNOSTICS, [])]
    diagnostics, reconstruction = audit.reconstruct_certified_plan_history(
        diagnostics)
    points, length = audit.load_trajectory(args.trajectory)
    model = ctypes.CDLL(str(args.library.resolve()))
    step = model.unity_source_plant_step_c
    step.argtypes = [ctypes.POINTER(evaluator.UnitySourcePlantInputC),
                     ctypes.c_double,
                     ctypes.POINTER(evaluator.UnitySourcePlantOutputC)]
    step.restype = ctypes.c_int
    corridor = audit.read_runtime_corridor_profile(
        Path("f1tenth_mpc/config/mpc_competition.yaml"))
    objective = audit.read_runtime_objective_profile(
        Path("f1tenth_mpc/config/mpc_competition.yaml"))

    branches: list[dict[str, Any]] = []
    valid_branch_refs: list[dict[str, Any]] = []
    state_rows: list[list[float]] = []
    for diag in diagnostics:
        if not has_candidate_prior_prefix_comparison(diag):
            continue
        try:
            candidate, prior = get_prefix_controls(diag)
        except (TypeError, ValueError):
            continue
        for source, controls in (("candidate", candidate), ("prior", prior)):
            trial = dict(diag, candidate_controls=controls)
            replay = audit._source_rollout(
                trial, points, length, step, csv.writer(_NullWriter()),
                corridor["corridor_margin_m"],
                corridor["first_prediction_corridor_margin_m"],
                corridor["corridor_preview_halfwidth_m"],
                corridor["nonlinear_corridor_tolerance_m"],
                scenario_label=f"{source}_75ms_prefix_branch",
                objective_profile=objective,
                stop_at_corridor_violation=False,
                collect_stage_trace=True)
            trace = replay["stage_trace"] or []
            exact_valid = (
                replay["source_plant_replay_steps"] == 3 and
                replay["first_corridor_violation_stage"] is None and
                replay.get("replay_failure_stage") is None and
                replay["min_corridor_slack_m"] is not None and
                replay["min_corridor_slack_m"] >= 0.0)
            if not exact_valid:
                branches.append({
                    "bag_time_s": diag.get("bag_time"),
                    "progress_m": diag.get("progress_m"),
                    "source": source,
                    "prefix_exact_valid": False,
                    "prefix_minimum_slack_m": replay["min_corridor_slack_m"],
                    "next_solve": None,
                })
                continue
            endpoint = branch_diagnostic(diag, trace[-1])
            branches.append({
                "bag_time_s": diag.get("bag_time"),
                "source_stamp_ns": diag.get("source_stamp_ns"),
                "progress_m": float(diag["progress_m"]),
                "comparison_status": diag.get("status"),
                "source": source,
                "issued_prefix": controls,
                "prefix_exact_valid": True,
                "prefix_objective": replay["scalar_nonlinear_objective"],
                "prefix_minimum_slack_m": replay["min_corridor_slack_m"],
                "next_state_progress_m": endpoint["progress_m"],
            })
            branches[-1]["branch_state"] = endpoint
            valid_branch_refs.append(branches[-1])
            state_rows.append(saved_state_row(endpoint))

    if not state_rows:
        raise RuntimeError("no exactly valid candidate/prior 75 ms prefixes found")
    states_path = args.output_dir / "predicted_next_epoch_states.csv"
    with states_path.open("w", newline="") as stream:
        stream.write("# " + ",".join(STATE_HEADER) + "\n")
        csv.writer(stream, lineterminator="\n").writerows(state_rows)
    replay_rows = run_replay(
        args.replay_binary, args.trajectory, states_path,
        args.output_dir / "next_epoch_mpc_replay.csv",
        args.output_dir / "next_epoch_mpc_replay.stderr.log")
    if len(replay_rows) != len(state_rows):
        raise RuntimeError("MPC replay row count does not match branched states")

    for branch, result in zip(valid_branch_refs, replay_rows):
        branch["next_solve"] = {
            key: result.get(key) for key in (
                "status", "iterations", "primal", "dual", "cycle_us",
                "line_search_trials", "line_search_accepted",
                "line_search_selected_alpha", "min_corridor_slack_m",
                "nonlinear_objective", "nonlinear_failure_reason",
                "nonlinear_failure_stage", "first_steering", "first_throttle")
        }

    by_state: dict[str, dict[str, Any]] = {}
    for branch in branches:
        key = str(branch.get("source_stamp_ns"))
        by_state.setdefault(key, {"progress_m": branch.get("progress_m")})[
            branch["source"]] = branch
    pairs = []
    for stamp, arms in by_state.items():
        if "candidate" not in arms or "prior" not in arms:
            continue
        cand, prior = arms["candidate"], arms["prior"]
        if not cand.get("prefix_exact_valid") or not prior.get("prefix_exact_valid"):
            continue
        c_cost = cand["prefix_objective"]["total"]
        p_cost = prior["prefix_objective"]["total"]
        pairs.append({
            "source_stamp_ns": int(stamp),
            "progress_m": cand["progress_m"],
            "candidate_prefix_cost": c_cost,
            "prior_prefix_cost": p_cost,
            "lower_local_cost_arm": "candidate" if c_cost <= p_cost else "prior",
            "candidate_next_solver_status": cand["next_solve"]["status"],
            "prior_next_solver_status": prior["next_solve"]["status"],
            "candidate_next_solver_slack_m":
                cand["next_solve"]["min_corridor_slack_m"],
            "prior_next_solver_slack_m":
                prior["next_solve"]["min_corridor_slack_m"],
            "candidate_next_objective": cand["next_solve"]["nonlinear_objective"],
            "prior_next_objective": prior["next_solve"]["nonlinear_objective"],
            "candidate_next_throttle": cand["next_solve"]["first_throttle"],
            "prior_next_throttle": prior["next_solve"]["first_throttle"],
        })
    report = {
        "schema": "sdu_apex_mpc_prefix_branch_viability/v1",
        "diagnostic_only": True,
        "interpretation": (
            "Each branch applies the logged 75ms prefix from the same saved "
            "state using the exact source plant, then cold-solves MPC at that "
            "predicted state. This is modeled next-epoch viability, not an "
            "observed Unity continuation or proof of closed-loop stability."),
        "bag": str(args.bag),
        "bag_diagnostic_count": len(diagnostics),
        "retained_plan_reconstruction": reconstruction,
        "source_hashes": {
            "trajectory": sha256(args.trajectory),
            "source_plant_library": sha256(args.library),
            "replay_binary": sha256(args.replay_binary),
            "source_plant_cpp": sha256(Path(
                "f1tenth_mpc/src/unity_source_plant.cpp")),
        },
        "objective_profile": objective,
        "corridor_profile": corridor,
        "summary": {
            "branch_count": len(branches),
            "exact_prefix_valid_branch_count": sum(
                bool(branch.get("prefix_exact_valid")) for branch in branches),
            "paired_valid_prefix_states": len(pairs),
            "candidate_lower_prefix_objective_count": sum(
                pair["lower_local_cost_arm"] == "candidate" for pair in pairs),
            "prior_lower_prefix_objective_count": sum(
                pair["lower_local_cost_arm"] == "prior" for pair in pairs),
            "candidate_next_solve_full_accepts": sum(
                pair["candidate_next_solver_status"] in {"0", "1", "8"}
                for pair in pairs),
            "prior_next_solve_full_accepts": sum(
                pair["prior_next_solver_status"] in {"0", "1", "8"}
                for pair in pairs),
        },
        "paired_states": pairs,
        "branches": branches,
    }
    (args.output_dir / "prefix_branch_viability.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0


class _NullWriter:
    def write(self, value: str) -> int:
        return len(value)


if __name__ == "__main__":
    raise SystemExit(main())
