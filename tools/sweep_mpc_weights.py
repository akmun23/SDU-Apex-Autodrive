#!/usr/bin/env python3
"""Screen small production-weight changes on saved practice MPC states.

This is a solver-feasibility/convergence screen, not a closed-loop lap-time
score. It ranks one-at-a-time, small increments on the training capture, then
checks only the top candidates on every independent validation capture. All
raw solver rows and the exact command/configuration are retained.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import subprocess
import sys
import time
from pathlib import Path


WEIGHT_NAMES = (
    "weight_e_y", "weight_e_psi", "weight_u", "weight_u_overspeed",
    "weight_longitudinal_slip", "weight_v", "weight_r",
    "weight_steering_command", "weight_steering_target",
    "weight_throttle_command", "weight_steering_slew",
    "weight_throttle_slew",
)
BASELINE = {
    "weight_e_y": 4000.0,
    "weight_e_psi": 1500.0,
    "weight_u": 7.5,
    "weight_u_overspeed": 60.0,
    "weight_longitudinal_slip": 25.0,
    "weight_v": 5.0,
    "weight_r": 7.8125,
    "weight_steering_command": 3.0,
    "weight_steering_target": 5.0,
    "weight_throttle_command": 0.1,
    "weight_steering_slew": 2000.0,
    "weight_throttle_slew": 10.0,
}
ACCEPTED = {0, 1, 2, 8}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def quantile(values: list[float], fraction: float) -> float | None:
    clean = sorted(value for value in values if math.isfinite(value))
    if not clean:
        return None
    return clean[min(len(clean) - 1, max(0, math.ceil(fraction * len(clean)) - 1))]


def read_solver_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def summarize(rows: list[dict[str, str]]) -> dict:
    if not rows:
        raise RuntimeError("solver replay returned no rows")
    statuses = [int(row["status"]) for row in rows]
    admm = [int(row["admm_status"]) for row in rows]
    accepted = [status in ACCEPTED for status in statuses]
    longest_rejected = current_rejected = 0
    for passed in accepted:
        current_rejected = 0 if passed else current_rejected + 1
        longest_rejected = max(longest_rejected, current_rejected)
    finite_slack = []
    cycle_ms = []
    first_steer = []
    first_throttle = []
    for row in rows:
        for source, target in (("min_corridor_slack_m", finite_slack),
                               ("cycle_us", cycle_ms),
                               ("first_steering", first_steer),
                               ("first_throttle", first_throttle)):
            try:
                value = float(row[source])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(value):
                target.append(value)
    steer_step = [abs(b - a) for a, b in zip(first_steer, first_steer[1:])]
    throttle_step = [abs(b - a) for a, b in zip(first_throttle, first_throttle[1:])]
    status_counts = {str(value): statuses.count(value) for value in sorted(set(statuses))}
    admm_counts = {str(value): admm.count(value) for value in sorted(set(admm))}
    return {
        "rows": len(rows),
        "status_counts": status_counts,
        "admm_status_counts": admm_counts,
        "exactly_accepted": sum(accepted),
        "exact_acceptance_fraction": sum(accepted) / len(rows),
        "optimal_solve_fraction": sum(value == 0 for value in admm) / len(admm),
        "iteration_limited_count": sum(value == 1 for value in admm),
        "longest_consecutive_rejected_cycles": longest_rejected,
        "finite_corridor_slack_m": {
            "p05": quantile(finite_slack, 0.05),
            "p50": quantile(finite_slack, 0.50),
            "minimum": min(finite_slack) if finite_slack else None,
        },
        "offline_cycle_ms": {
            "p50": quantile([value * 1.0e-3 for value in cycle_ms], 0.50),
            "p95": quantile([value * 1.0e-3 for value in cycle_ms], 0.95),
            "max": max(cycle_ms) * 1.0e-3 if cycle_ms else None,
        },
        "first_command_delta": {
            "steering_p95_rad": quantile(steer_step, 0.95),
            "throttle_p95": quantile(throttle_step, 0.95),
        },
    }


def run_replay(binary: Path, trajectory: Path, inputs: Path,
               raw_output: Path, weights: dict[str, float]) -> dict:
    # Match mpc_competition.yaml solver settings. The final twelve arguments
    # override all objective weights directly; no replay-only cost scaling is
    # used to approximate the production objective.
    command = [
        str(binary), str(trajectory), str(inputs),
        "7", "0", "0", "1.6", "1",       # rho, input rho, alpha, legacy scale
        "0", "0", "0", "32", "10", "0.02", # state rho, rho cap/period, tol
        "sequence", "warm", "8", "100", ".25", # runtime sequence, workers, cap, seed
        "1", "1", ".10", ".10", ".05", ".05", # legacy multipliers/scales/corridor
        "7", "2000", "10", "1", "0", "1", # input rho, slews, adaptive, line search
        *(format(weights[name], ".9g") for name in WEIGHT_NAMES),
    ]
    stderr_path = raw_output.with_suffix(raw_output.suffix + ".stderr.log")
    with raw_output.open("x") as output, stderr_path.open("x") as errors:
        completed = subprocess.run(command, stdout=output, stderr=errors,
                                   text=True, check=False)
    if completed.returncode:
        raise RuntimeError(
            f"solver replay failed ({completed.returncode}); see {stderr_path}")
    rows = read_solver_rows(raw_output)
    return {
        "command": command,
        "weights": dict(weights),
        "summary": summarize(rows),
    }


def candidate_set(small_grid: bool = False) -> list[tuple[str, dict[str, float]]]:
    if small_grid:
        # A compact factorial screen of coupled path/heading/steering-rate
        # tradeoffs. All settings stay within the neighborhood of the only
        # five-lap profile; the held-out run set determines whether a change is
        # robust enough to merit a closed-loop trial.
        trials: dict[str, dict[str, float]] = {"baseline": dict(BASELINE)}
        for e_y, e_psi, steering_slew in itertools.product(
                (3900.0, 4000.0, 4100.0),
                (1450.0, 1500.0, 1550.0, 1600.0),
                (1950.0, 2000.0, 2050.0)):
            weights = dict(BASELINE)
            weights["weight_e_y"] = e_y
            weights["weight_e_psi"] = e_psi
            weights["weight_steering_slew"] = steering_slew
            name = f"ey{e_y:g}_epsi{e_psi:g}_steerslew{steering_slew:g}"
            if weights != BASELINE:
                trials[name] = weights
        return list(trials.items())

    # Vary one term at a time around the five-lap baseline. Increments are
    # deliberately small because the much larger 6000 lateral / 2000 heading /
    # 1000 steering-slew experiments already showed clear regressions.
    trials: list[tuple[str, dict[str, float]]] = [("baseline", dict(BASELINE))]
    for name, values in (
        ("weight_e_y", (3750, 3875, 4125, 4250)),
        ("weight_e_psi", (1400, 1450, 1550, 1600)),
        ("weight_steering_slew", (1900, 1950, 2050, 2100)),
    ):
        for value in values:
            weights = dict(BASELINE)
            weights[name] = float(value)
            trials.append((f"{name}_{value:g}", weights))
    return trials


def rank_key(record: dict) -> tuple:
    summary = record["training"]["summary"]
    return (
        summary["exact_acceptance_fraction"],
        summary["optimal_solve_fraction"],
        -summary["longest_consecutive_rejected_cycles"],
        summary["finite_corridor_slack_m"]["p05"]
        if summary["finite_corridor_slack_m"]["p05"] is not None else -math.inf,
    )


def validation_label(index: int, path: Path) -> str:
    """Create a stable, filesystem-safe label for a held-out capture."""
    safe_parent = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in path.parent.name
    ).strip("_") or f"capture_{index}"
    return f"v{index}_{safe_parent}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--training-input", type=Path, required=True)
    parser.add_argument(
        "--validation-input", type=Path, action="append", nargs="+",
        required=True,
        help="held-out solver input CSV; repeat this option for independent runs",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validate-top", type=int, default=3)
    parser.add_argument(
        "--small-grid", action="store_true",
        help="screen a compact 36-point coupled grid in e_y, e_psi, and steering slew",
    )
    parser.add_argument(
        "--validate-all", action="store_true",
        help="run every candidate on every held-out capture instead of only the training-ranked top candidates",
    )
    args = parser.parse_args()
    validation_inputs = [
        path for group in args.validation_input for path in group
    ]
    if not validation_inputs:
        parser.error("at least one --validation-input is required")
    for path in (args.binary, args.trajectory, args.training_input,
                 *validation_inputs):
        if not path.is_file():
            parser.error(f"file does not exist: {path}")
    if not 1 <= args.validate_top <= 8:
        parser.error("--validate-top must be in [1, 8]")
    if args.output_dir.exists():
        parser.error(f"refusing to overwrite existing directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "inputs.json").write_text(json.dumps({
        "training_input": str(args.training_input.resolve()),
        "training_sha256": sha256(args.training_input),
        "validation_inputs": [
            {
                "label": validation_label(index, path),
                "path": str(path.resolve()),
                "sha256": sha256(path),
            }
            for index, path in enumerate(validation_inputs, start=1)
        ],
        "trajectory": str(args.trajectory.resolve()),
        "trajectory_sha256": sha256(args.trajectory),
        "binary": str(args.binary.resolve()),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scope": "saved-state solver feasibility/convergence screen only; not closed-loop tracking or lap-time validation",
    }, indent=2) + "\n")

    records = []
    trials = candidate_set(args.small_grid)
    for index, (name, weights) in enumerate(trials, start=1):
        print(f"training {index}/{len(trials)}: {name}", flush=True)
        result = run_replay(args.binary.resolve(), args.trajectory.resolve(),
                            args.training_input.resolve(),
                            args.output_dir / f"{name}_train.csv", weights)
        records.append({"name": name, "training": result})

    ranked = sorted(records, key=rank_key, reverse=True)
    selected_names = (
        {record["name"] for record in ranked}
        if args.validate_all else
        {record["name"] for record in ranked[:args.validate_top]}
    )
    selected_names.add("baseline")
    validation_labels = [
        validation_label(index, path)
        for index, path in enumerate(validation_inputs, start=1)
    ]
    for index, record in enumerate(records, start=1):
        if record["name"] not in selected_names:
            continue
        record["validation"] = {}
        for validation_index, validation_input in enumerate(
                validation_inputs, start=1):
            label = validation_labels[validation_index - 1]
            print(f"validation {label}: {record['name']}", flush=True)
            record["validation"][label] = run_replay(
                args.binary.resolve(), args.trajectory.resolve(),
                validation_input.resolve(),
                args.output_dir / f"{record['name']}_{label}.csv",
                record["training"]["weights"],
            )

    for record in records:
        validation_summaries = [
            result["summary"]
            for result in record.get("validation", {}).values()
        ]
        if validation_summaries:
            record["worst_validation"] = {
                "minimum_exact_acceptance_fraction": min(
                    item["exact_acceptance_fraction"]
                    for item in validation_summaries),
                "minimum_optimal_solve_fraction": min(
                    item["optimal_solve_fraction"]
                    for item in validation_summaries),
                "maximum_rejected_cycle_streak": max(
                    item["longest_consecutive_rejected_cycles"]
                    for item in validation_summaries),
                "minimum_corridor_slack_p05_m": min(
                    item["finite_corridor_slack_m"]["p05"]
                    for item in validation_summaries
                    if item["finite_corridor_slack_m"]["p05"] is not None
                ),
            }

    if args.validate_all:
        ranked = sorted(
            records,
            key=lambda record: (
                record["worst_validation"][
                    "minimum_exact_acceptance_fraction"],
                record["worst_validation"][
                    "minimum_optimal_solve_fraction"],
                -record["worst_validation"][
                    "maximum_rejected_cycle_streak"],
                record["worst_validation"][
                    "minimum_corridor_slack_p05_m"],
            ),
            reverse=True,
        )

    report = {
        "schema": "sdu-apex-mpc-small-weight-sweep-v3",
        "candidate_mode": "small_grid" if args.small_grid else "one_at_a_time",
        "rank_order": [record["name"] for record in ranked],
        "selection_rule": (
            "worst held-out exact-acceptance fraction, then worst held-out "
            "optimal-solve fraction, longest held-out rejection streak, then "
            "minimum held-out p05 exact corridor slack"
            if args.validate_all else
            "training exact-acceptance fraction, then optimal-solve fraction, "
            "longest rejection streak, then p05 exact corridor slack; top "
            "candidates checked on every separate validation capture"
        ),
        "warning": "Saved-state replay does not execute the candidate command sequence and cannot establish path tracking, repeatability, lap time, or Unity-model fidelity. A candidate must still pass an uncapped practice run.",
        "trials": records,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    with (args.output_dir / "summary.csv").open("x", newline="") as output:
        fields = ["name", "weights_changed", "training_accept_fraction",
                  "training_optimal_fraction", "training_max_reject_streak"]
        for label in validation_labels:
            fields.extend((f"{label}_accept_fraction",
                           f"{label}_optimal_fraction",
                           f"{label}_max_reject_streak"))
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for record in ranked:
            changed = [key for key in WEIGHT_NAMES
                       if record["training"]["weights"][key] != BASELINE[key]]
            train = record["training"]["summary"]
            row = {
                "name": record["name"],
                "weights_changed": ";".join(changed) or "none",
                "training_accept_fraction": train["exact_acceptance_fraction"],
                "training_optimal_fraction": train["optimal_solve_fraction"],
                "training_max_reject_streak": train["longest_consecutive_rejected_cycles"],
            }
            for label in validation_labels:
                validation = record.get("validation", {}).get(
                    label, {}).get("summary", {})
                row[f"{label}_accept_fraction"] = validation.get(
                    "exact_acceptance_fraction", "")
                row[f"{label}_optimal_fraction"] = validation.get(
                    "optimal_solve_fraction", "")
                row[f"{label}_max_reject_streak"] = validation.get(
                    "longest_consecutive_rejected_cycles", "")
            writer.writerow(row)
    print(f"saved {len(records)} training trials and "
          f"{sum(len(record.get('validation', {})) for record in records)} "
          f"candidate/capture validation replays in {args.output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, csv.Error) as error:
        print(f"sweep_mpc_weights: {error}", file=sys.stderr)
        raise SystemExit(2)
