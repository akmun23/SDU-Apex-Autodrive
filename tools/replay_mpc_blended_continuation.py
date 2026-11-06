#!/usr/bin/env python3
"""Replay source-plant-feasible candidate/prior blends as MPC nominal seeds.

This is an offline diagnostic. It does not publish commands or change the
production MPC. Input states must be the same saved states used by the source
plant blend audit. Only blends that passed the exact 30-step source rollout
with nonnegative measured corridor slack are selected.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any


ACCEPTED_FULL = {"0", "1", "8"}
SEED_ENV = "MPC_REPLAY_NOMINAL_SEED_CSV"
REPLAY_ARGUMENTS = [
    "7", "0", "0", "1.6", "0.001", "0", "0", "0", "32", "10",
    "0.02", "independent", "cold", "8", "100", "0.25", "1", "1",
    "0.10", "0.10", "0.05", "0.05", "7", "2000", "10", "1", "0",
    "1", "4000", "1500", "7.5", "60", "25", "5", "7.8125", "3",
    "5", "0.1", "2000", "100",
]
REPLAY_LINE_SEARCH_ARGUMENT_INDEX = 26


def replay_arguments(line_search_backtracks: int = 0) -> list[str]:
    """Return fixed replay settings with an explicit offline backtrack count."""
    if (isinstance(line_search_backtracks, bool) or
            not isinstance(line_search_backtracks, int) or
            not 0 <= line_search_backtracks <= 3):
        raise ValueError("line-search backtracks must be an integer in [0, 3]")
    arguments = list(REPLAY_ARGUMENTS)
    arguments[REPLAY_LINE_SEARCH_ARGUMENT_INDEX] = str(line_search_backtracks)
    return arguments


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_numeric_states(path: Path) -> list[list[float]]:
    rows = []
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        row = [float(value) for value in line.split(",")]
        if len(row) != 38 or not all(math.isfinite(value) for value in row):
            raise ValueError(f"invalid 38-column state row in {path}")
        rows.append(row)
    return rows


def load_feasible_blends(
        path: Path, candidate_weight: float | None = None
        ) -> dict[float, dict[str, Any]]:
    best: dict[float, dict[str, Any]] = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            row_weight = float(row["candidate_weight"])
            if candidate_weight is not None and not math.isclose(
                    row_weight, candidate_weight, abs_tol=1.0e-9):
                continue
            if row.get("full_horizon_source_valid") != "True":
                continue
            if row.get("command_bounds_valid") != "True":
                continue
            if int(row["source_steps"]) != 30:
                continue
            # The audit allows its configured 5 mm nonlinear tolerance. For
            # a seed we require a nonnegative measured body-envelope margin.
            if float(row["minimum_corridor_slack_m"]) < 0.0:
                continue
            progress = float(row["progress_m"])
            controls = json.loads(row["controls_json"])
            if len(controls) != 30 or any(
                    len(control) != 2 or
                    not (-0.5235987756 <= float(control[0]) <= 0.5235987756) or
                    not (0.001 - 1.0e-7 <= float(control[1]) <= 1.0 + 1.0e-7)
                    for control in controls):
                raise ValueError("source-valid blend has invalid controls")
            candidate = {
                "progress_m": progress,
                "bag_time_s": float(row["bag_time_s"]),
                "candidate_weight": row_weight,
                "minimum_corridor_slack_m":
                    float(row["minimum_corridor_slack_m"]),
                "speed_error_rmse_mps": float(row["speed_error_rmse_mps"]),
                "objective_total": float(row["objective_total"]),
                "controls": controls,
            }
            previous = best.get(progress)
            if previous is None or candidate["objective_total"] < \
                    previous["objective_total"]:
                best[progress] = candidate
    return best


def select_matched_states(
    states: list[list[float]], blends: dict[float, dict[str, Any]],
    tolerance_m: float = 1.0e-4,
) -> list[tuple[list[float], dict[str, Any]]]:
    return [(state, blend) for _, state, blend in
            select_matched_state_indices(states, blends, tolerance_m)]


def select_matched_state_indices(
    states: list[list[float]], blends: dict[float, dict[str, Any]],
    tolerance_m: float = 1.0e-4,
) -> list[tuple[int, list[float], dict[str, Any]]]:
    """Match each exact-valid source blend to one saved-state row."""
    selected = []
    remaining = set(blends)
    for index, state in enumerate(states):
        progress = state[1]
        matches = [value for value in remaining
                   if abs(value - progress) <= tolerance_m]
        if len(matches) > 1:
            raise ValueError(f"ambiguous source-blend match at s={progress}")
        if matches:
            key = matches[0]
            selected.append((index, state, blends[key]))
            remaining.remove(key)
    if not selected:
        raise ValueError("no positive-slack source blends match input states")
    return selected


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1,
                       max(0, math.ceil(p * len(ordered)) - 1))]


def summarize_replay(rows: list[dict[str, str]]) -> dict[str, Any]:
    statuses = sorted({row["status"] for row in rows})
    cycles = [float(row["cycle_us"]) / 1000.0 for row in rows]
    objectives = [float(row["nonlinear_objective"]) for row in rows
                  if row["status"] in ACCEPTED_FULL]
    return {
        "rows": len(rows),
        "status_counts": {status: sum(row["status"] == status for row in rows)
                          for status in statuses},
        "exact_full_horizon_accepts": sum(
            row["status"] in ACCEPTED_FULL for row in rows),
        "median_cycle_ms": percentile(cycles, 0.5),
        "p95_cycle_ms": percentile(cycles, 0.95),
        "max_cycle_ms": max(cycles) if cycles else None,
        "median_accepted_nonlinear_objective": percentile(objectives, 0.5),
    }


def _candidate_metrics(row: dict[str, str]) -> dict[str, Any]:
    try:
        objective = float(row["nonlinear_objective"])
        slack = float(row["min_corridor_slack_m"])
    except (KeyError, TypeError, ValueError):
        objective, slack = math.inf, -math.inf
    valid = (row.get("status") in ACCEPTED_FULL and
             math.isfinite(objective) and math.isfinite(slack) and slack >= 0.0)
    return {"valid": valid, "objective": objective, "slack": slack,
            "status": row.get("status")}


def build_quality_selector_rows(
    states: list[list[float]],
    selected: list[tuple[int, list[float], dict[str, Any]]],
    cold_rows: list[dict[str, str]],
    seeded_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """Choose the lowest-cost exact-valid action among replayed candidates.

    Saved states without a tested source-valid seed use only their ordinary
    replay result. Seeded states compare the ordinary cold result, source seed,
    and same-solve seeded MPC result. The selector never treats an invalid
    candidate as safe. This is an offline counterfactual: it assumes every
    candidate was computed before selection and does not claim live timing.
    """
    if len(cold_rows) != len(states):
        raise ValueError("cold replay row count must match saved-state count")
    if len(seeded_rows) != len(selected):
        raise ValueError("seeded replay row count must match seeded-state count")
    seed_by_index = {index: (blend, seeded_rows[seed_row])
                     for seed_row, (index, _, blend) in enumerate(selected)}
    rows = []
    for index, state in enumerate(states):
        cold = _candidate_metrics(cold_rows[index])
        source = seed_by_index.get(index)
        seed_available = source is not None
        if seed_available:
            blend, candidate_row = source
            candidate = _candidate_metrics(candidate_row)
            seed_objective = float(blend["objective_total"])
            seed_slack = float(blend["minimum_corridor_slack_m"])
            seed_valid = (int(blend.get("source_steps", 30)) == 30 and
                          math.isfinite(seed_objective) and seed_slack >= 0.0)
            if not seed_valid:
                raise ValueError("selected source seed is not exact-valid")
            if candidate["valid"] and candidate["objective"] <= seed_objective:
                same_cycle_action = "accept_seeded_mpc_candidate"
                same_cycle_objective = candidate["objective"]
                same_cycle_slack = candidate["slack"]
            else:
                same_cycle_action = "retain_exact_valid_source_seed"
                same_cycle_objective = seed_objective
                same_cycle_slack = seed_slack
            choices = [("retain_exact_valid_source_seed", seed_objective,
                        seed_slack)]
            if cold["valid"]:
                choices.append(("ordinary_exact_valid_mpc_candidate",
                                cold["objective"], cold["slack"]))
            if candidate["valid"]:
                choices.append(("mpc_candidate_from_source_seed",
                                candidate["objective"], candidate["slack"]))
            choice, chosen_objective, chosen_slack = min(
                choices, key=lambda item: item[1])
        else:
            candidate = cold
            seed_objective = None
            seed_slack = None
            same_cycle_action = None
            same_cycle_objective = None
            same_cycle_slack = None
            if candidate["valid"]:
                choice = "ordinary_exact_valid_mpc_candidate"
                chosen_objective = candidate["objective"]
                chosen_slack = candidate["slack"]
            else:
                choice = "no_exact_valid_full_horizon_action"
                chosen_objective = None
                chosen_slack = None
        rows.append({
            "state_row": index,
            "stamp_ns": int(state[0]),
            "progress_m": float(state[1]),
            "source_seed_available": seed_available,
            "ordinary_cold_status": cold["status"],
            "ordinary_cold_exact_valid": cold["valid"],
            "ordinary_cold_objective": cold["objective"] if cold["valid"] else None,
            "seeded_mpc_status": candidate["status"] if seed_available else None,
            "seeded_mpc_exact_valid": candidate["valid"] if seed_available else None,
            "seeded_mpc_objective": (
                candidate["objective"] if seed_available and candidate["valid"]
                else None),
            "seeded_mpc_slack_m": (
                candidate["slack"] if seed_available and candidate["valid"]
                else None),
            "source_seed_objective": seed_objective,
            "source_seed_slack_m": seed_slack,
            "same_cycle_guard_action": same_cycle_action,
            "same_cycle_guard_objective": same_cycle_objective,
            "same_cycle_guard_slack_m": same_cycle_slack,
            "selected_action": choice,
            "selected_objective": chosen_objective,
            "selected_slack_m": chosen_slack,
        })
    return rows


def summarize_quality_selector(rows: list[dict[str, Any]],
                               cold_rows: list[dict[str, str]],
                               seeded_rows: list[dict[str, str]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["selected_action"]] = counts.get(row["selected_action"], 0) + 1
    selected = [row for row in rows if row["selected_slack_m"] is not None]
    objectives = [float(row["selected_objective"]) for row in selected]
    slacks = [float(row["selected_slack_m"]) for row in selected]
    jointly_valid = [row for row in rows
                     if row["ordinary_cold_exact_valid"] and
                     row["selected_objective"] is not None]
    objective_deltas = [
        (float(row["selected_objective"]) -
         float(row["ordinary_cold_objective"]), row)
        for row in jointly_valid]
    objective_worse = [(delta, row) for delta, row in objective_deltas
                       if delta > max(1.0e-4,
                                      abs(float(row["ordinary_cold_objective"])) * 1.0e-6)]
    objective_better = [(delta, row) for delta, row in objective_deltas
                        if delta < -max(1.0e-4,
                                        abs(float(row["ordinary_cold_objective"])) * 1.0e-6)]
    seeded_regressions = [row for row in rows
                          if row["seeded_mpc_exact_valid"] is True and
                          row["seeded_mpc_objective"] is not None and
                          row["source_seed_objective"] is not None and
                          row["seeded_mpc_objective"] >
                          row["source_seed_objective"] + max(
                              1.0e-4,
                              abs(row["source_seed_objective"]) * 1.0e-6)]
    guarded_rows = [row for row in rows
                    if row["same_cycle_guard_objective"] is not None]
    guarded_slacks = [float(row["same_cycle_guard_slack_m"])
                      for row in guarded_rows]
    guarded_cold_comparisons = [row for row in guarded_rows
                                if row["ordinary_cold_exact_valid"]]
    guarded_cold_regressions = [row for row in guarded_cold_comparisons
                                if float(row["same_cycle_guard_objective"]) >
                                float(row["ordinary_cold_objective"]) + max(
                                    1.0e-4,
                                    abs(float(row["ordinary_cold_objective"])) * 1.0e-6)]
    worst_guarded = max(
        guarded_cold_regressions,
        key=lambda row: (float(row["same_cycle_guard_objective"]) -
                         float(row["ordinary_cold_objective"])),
        default=None)
    guarded_choices: dict[str, int] = {}
    for row in guarded_rows:
        action = row["same_cycle_guard_action"]
        guarded_choices[action] = guarded_choices.get(action, 0) + 1
    worst_regression = max(objective_worse, key=lambda item: item[0],
                           default=None)
    return {
        "saved_state_count": len(rows),
        "source_seed_coverage": sum(row["source_seed_available"] for row in rows),
        "ordinary_cold_full_horizon_accepts": sum(
            _candidate_metrics(row)["valid"] for row in cold_rows),
        "seeded_mpc_full_horizon_accepts": sum(
            _candidate_metrics(row)["valid"] for row in seeded_rows),
        "selector_full_horizon_valid_count": len(selected),
        "selection_counts": counts,
        "selected_objective_median": percentile(objectives, 0.5),
        "selected_min_corridor_slack_m": min(slacks) if slacks else None,
        "same_cycle_source_nominal_guard": {
            "seeded_state_count": len(guarded_rows),
            "choice_counts": guarded_choices,
            "minimum_selected_slack_m": min(guarded_slacks)
                if guarded_slacks else None,
            "comparison_to_cold_on_jointly_valid_states": {
                "states": len(guarded_cold_comparisons),
                "higher_objective_count": len(guarded_cold_regressions),
                "worst_higher_objective": ({
                    "progress_m": float(worst_guarded["progress_m"]),
                    "cold_objective": float(
                        worst_guarded["ordinary_cold_objective"]),
                    "guard_objective": float(
                        worst_guarded["same_cycle_guard_objective"]),
                    "relative_increase": float(
                        worst_guarded["same_cycle_guard_objective"] -
                        worst_guarded["ordinary_cold_objective"]) /
                        max(abs(float(worst_guarded[
                            "ordinary_cold_objective"])), 1.0e-12),
                } if guarded_cold_regressions else None),
            },
        },
        "selected_vs_ordinary_cold_on_jointly_valid_states": {
            "states": len(jointly_valid),
            "higher_objective_count": len(objective_worse),
            "lower_objective_count": len(objective_better),
            "equal_within_tolerance_count": (
                len(jointly_valid) - len(objective_worse) - len(objective_better)),
            "worst_higher_objective": ({
                "progress_m": float(worst_regression[1]["progress_m"]),
                "cold_objective": float(
                    worst_regression[1]["ordinary_cold_objective"]),
                "selected_objective": float(
                    worst_regression[1]["selected_objective"]),
                "absolute_increase": float(worst_regression[0]),
                "relative_increase": float(worst_regression[0]) /
                    max(abs(float(worst_regression[1]["ordinary_cold_objective"])),
                        1.0e-12),
            } if worst_regression else None),
        },
        "seeded_mpc_cost_above_exact_valid_source_seed_count":
            len(seeded_regressions),
        "ordinary_cold_timing_ms": summarize_replay(cold_rows),
        "source_seeded_mpc_timing_ms": summarize_replay(seeded_rows),
        "timing_limitation": (
            "Does not include the offline search or exact plant rollouts used "
            "to find and certify the source seed."),
    }


def run_replay(binary: Path, trajectory: Path, state_csv: Path,
               output_csv: Path, stderr_path: Path,
               seed_csv: Path | None,
               line_search_backtracks: int = 0) -> dict[str, Any]:
    command = [str(binary), str(trajectory), str(state_csv),
               *replay_arguments(line_search_backtracks)]
    env = os.environ.copy()
    env.pop(SEED_ENV, None)
    if seed_csv is not None:
        env[SEED_ENV] = str(seed_csv.resolve())
    result = subprocess.run(command, text=True, capture_output=True,
                            env=env, check=False)
    output_csv.write_text(result.stdout)
    stderr_path.write_text(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(
            f"MPC replay failed ({result.returncode}); see {stderr_path}")
    with output_csv.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    return summarize_replay(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--blends", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--replay-binary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--candidate-weight", type=float, choices=(0.25, 0.50, 0.75),
        help="restrict to one tested candidate/prior blend weight")
    parser.add_argument(
        "--line-search-backtracks", type=int, choices=range(4), default=0,
        help="offline exact nonlinear recovery/merit trials (0 keeps baseline)")
    args = parser.parse_args()

    states = load_numeric_states(args.states)
    blends = load_feasible_blends(args.blends, args.candidate_weight)
    selected = select_matched_state_indices(states, blends)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    state_csv = args.output_dir / "matched_states.csv"
    all_state_csv = args.output_dir / "all_saved_states.csv"
    seed_csv = args.output_dir / "source_plant_seed_controls.csv"
    with all_state_csv.open("w", newline="") as stream:
        stream.write("# all saved states; input row order preserved\n")
        csv.writer(stream, lineterminator="\n").writerows(states)
    with state_csv.open("w", newline="") as stream:
        stream.write("# exact saved-state rows matched by progress to the blend audit\n")
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerows(state for _, state, _ in selected)
    with seed_csv.open("w", newline="") as stream:
        stream.write("# row_index followed by 30 steering/throttle pairs\n")
        writer = csv.writer(stream, lineterminator="\n")
        for index, (_, _, blend) in enumerate(selected):
            writer.writerow([index] + [
                value for control in blend["controls"]
                for value in control])

    baseline = run_replay(
        args.replay_binary, args.trajectory, all_state_csv,
        args.output_dir / "ordinary_cold_all_saved_states.csv",
        args.output_dir / "ordinary_cold_all_saved_states.stderr.log", None,
        args.line_search_backtracks)
    seeded = run_replay(
        args.replay_binary, args.trajectory, state_csv,
        args.output_dir / "source_seeded.csv",
        args.output_dir / "source_seeded.stderr.log", seed_csv,
        args.line_search_backtracks)
    baseline_rows = list(csv.DictReader(
        (args.output_dir / "ordinary_cold_all_saved_states.csv").open(newline="")))
    seeded_rows = list(csv.DictReader(
        (args.output_dir / "source_seeded.csv").open(newline="")))
    selector_rows = build_quality_selector_rows(
        states, selected, baseline_rows, seeded_rows)
    selector_csv = args.output_dir / "quality_selector.csv"
    with selector_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(selector_rows[0]),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(selector_rows)
    selector_summary = summarize_quality_selector(
        selector_rows, baseline_rows, seeded_rows)
    seeded_baseline_rows = [baseline_rows[index]
                            for index, _, _ in selected]
    report = {
        "schema": "sdu_apex_source_plant_continuation_seed_replay/v1",
        "diagnostic_only": True,
        "source_inputs": {
            "states": str(args.states), "states_sha256": sha256(args.states),
            "blend_trials": str(args.blends), "blend_trials_sha256": sha256(args.blends),
            "trajectory": str(args.trajectory), "trajectory_sha256": sha256(args.trajectory),
            "replay_binary": str(args.replay_binary),
            "replay_binary_sha256": sha256(args.replay_binary),
        },
        "seed_selection": (
            "lowest exact nonlinear objective per matching state among 30-step "
            "source-valid blends with strictly nonnegative corridor slack; "
            "candidate/prior interpolation; no production command publication"),
        "candidate_weight_filter": args.candidate_weight,
        "line_search_backtracks": args.line_search_backtracks,
        "population": {
            "all_saved_states": len(states),
            "states_with_exact_source_valid_blend_seed": len(selected),
            "states_without_a_tested_source_valid_blend_seed":
                len(states) - len(selected),
        },
        "selected_states": [{
            "row": input_index,
            "progress_m": float(state[1]),
            "source_blend_candidate_weight": blend["candidate_weight"],
            "source_blend_corridor_slack_m": blend[
                "minimum_corridor_slack_m"],
            "source_blend_speed_error_rmse_mps": blend[
                "speed_error_rmse_mps"],
            "source_blend_objective": blend["objective_total"],
            "source_blend_first_action": blend["controls"][0],
        } for input_index, state, blend in selected],
        "ordinary_cold": baseline,
        "ordinary_cold_on_seeded_subset": summarize_replay(seeded_baseline_rows),
        "source_plant_seeded": seeded,
        "quality_selector": selector_summary,
        "rowwise_status_changes_on_seeded_subset": [
            input_index for seed_row, (input_index, _, _) in enumerate(selected)
            if baseline_rows[input_index]["status"] != seeded_rows[seed_row]["status"]],
        "output_files": {
            "all_saved_states": str(all_state_csv),
            "matched_states": str(state_csv),
            "seed_controls": str(seed_csv),
            "ordinary_cold_csv": str(
                args.output_dir / "ordinary_cold_all_saved_states.csv"),
            "source_seeded_csv": str(args.output_dir / "source_seeded.csv"),
            "quality_selector_csv": str(selector_csv),
        },
        "quality_selector_limitation": (
            "For states with a tested exact-valid candidate/prior blend, the "
            "offline selector compares the cold MPC result, source seed, and "
            "seeded MPC result, selecting the lowest exact objective among "
            "valid candidates. Other saved states use only the ordinary cold "
            "MPC result. All candidates are computed offline; source blend "
            "scan/certification cost and the extra cold/seeded solve costs are "
            "not a live-policy timing result."),
        "interpretation": (
            "The source rollout certifies only the seed sequence from each saved "
            "state. MPC may alter that sequence; only its own exact nonlinear "
            "full-horizon acceptance counts as solver success. Saved-state replay "
            "does not establish live closed-loop benefit or historical ADMM parity."),
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"report": str(report_path),
                      "selected_states": len(selected),
                      "all_saved_states": len(states),
                      "ordinary_cold": baseline,
                      "source_plant_seeded": seeded,
                      "quality_selector": selector_summary,
                      "rowwise_status_changes": report[
                          "rowwise_status_changes_on_seeded_subset"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
