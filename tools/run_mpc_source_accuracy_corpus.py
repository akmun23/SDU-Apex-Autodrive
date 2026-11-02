#!/usr/bin/env python3
"""Reproduce three difficult MPC source-shooting cases (offline only).

The source-adjoint prototype deliberately has no ROS publisher and no runtime
fallback behavior. It validates the full nonlinear source plant and the
original-practice hard corridor, while comparing against frozen-QP decisions.
Do not infer real-time performance from this offline diagnostic.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CORPUS = REPO / "live_runs/mpc_phase13_solver_consistency_audit_20261101"
TRAJECTORY = REPO / (
    "f1tenth_planning/trajectories/autodrive_practice_20260924_b/"
    "source_command_4p800.csv"
)
BINARY = REPO / "build/f1tenth_mpc/mpc_bag_solver_replay"
WEIGHTS = ("4500", "1450", "45", "20", "5", "5", "7.8125",
           "4.5", "5", "0.1", "1500", "50")
REPLAY_ARGS = (
    "7", "7", "7", "1.6", "0.001", "0", "0", "0", "32", "10",
    "0.02", "independent", "warm", "6", "150", "0.25", "1", "1",
    "0.1", "0.1", "0.02", "0.02", "7", "1500", "50", "1", "4", "1",
    *WEIGHTS, "1", "0.5", "0.6", "0", "1", "0.0", "0",
)
assert len(REPLAY_ARGS) + 3 == 50


def items(line: str) -> dict[str, str]:
    return dict(re.findall(r"([a-zA-Z_]+)=([^\s]+)", line))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=REPO / "live_runs/mpc_phase13_prediction_fidelity_20261101")
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--finite-difference", action="store_true",
                        help="also benchmark slower 60D exact-source finite difference gradient")
    args = parser.parse_args()
    if args.iterations < 1 or args.iterations > 8:
        parser.error("iterations must be between 1 and 8")
    if not BINARY.exists():
        parser.error(f"missing {BINARY}; build target mpc_bag_solver_replay")
    for case in (0, 4, 7):
        if not (CORPUS / f"practice_case_{case}.csv").exists():
            parser.error(f"missing practice_case_{case}.csv; restore saved-state corpus")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    modes = ["source_adjoint"]
    if args.finite_difference:
        modes.append("finite_difference")
    for mode in modes:
        for case in (0, 4, 7):
            env = os.environ.copy()
            env.update(
                MPC_REPLAY_COMMAND_INCREMENT_QP="1",
                MPC_REPLAY_FEEDBACK_GAIN="0.25",
                MPC_REPLAY_GRADIENT_AUDIT_ROW="0",
                MPC_REPLAY_EXACT_SOURCE_ORACLE_ITERATIONS=str(args.iterations),
                MPC_REPLAY_ORACLE_HARD_GEOMETRY="1",
            )
            if mode == "source_adjoint":
                env["MPC_REPLAY_ORACLE_USE_SOURCE_ADJOINT"] = "1"
            command = [
                str(BINARY), str(TRAJECTORY),
                str(CORPUS / f"practice_case_{case}.csv"),
                *REPLAY_ARGS,
            ]
            trial = subprocess.run(command, cwd=REPO, env=env,
                                   capture_output=True, text=True, timeout=120)
            (output / f"{mode}_case_{case}.log").write_text(trial.stderr)
            if trial.returncode:
                raise RuntimeError(f"{mode} case {case}: replay exited {trial.returncode}")
            initial = None
            final = None
            gradients = []
            for line in trial.stderr.splitlines():
                if line.startswith("SOURCE_ORACLE progress="):
                    value = items(line)
                    if value.get("iter") == "0":
                        initial = float(value["exact_cost"])
                elif line.startswith("SOURCE_ORACLE_BEST progress="):
                    final = items(line)
                elif line.startswith("SOURCE_ADJOINT_PARITY progress="):
                    value = items(line)
                    if "relative_gradient_error" in value:
                        gradients.append(float(value["relative_gradient_error"]))
            if initial is None or final is None:
                raise RuntimeError(f"{mode} case {case}: missing source diagnostics")
            row = {
                "mode": mode,
                "case": case,
                "progress_m": float(final["progress"]),
                "initial_exact_cost": initial,
                "final_exact_cost": float(final["exact_cost"]),
                "exact_cost_improvement_pct": (
                    100.0 * (initial - float(final["exact_cost"])) / initial
                ),
                "original_corridor_hard_valid": int(final["hard_valid"]),
                "minimum_original_clearance_m": float(final["hard_min_clearance"]),
                "first_iteration_gradient_relative_error": (
                    gradients[0] if gradients else ""
                ),
                "worst_iteration_gradient_relative_error": (
                    max(gradients) if gradients else ""
                ),
                "iterations_requested": args.iterations,
            }
            rows.append(row)
            print(mode, f"progress={row['progress_m']:.2f}m",
                  f"cost={initial:.2f}->{row['final_exact_cost']:.2f}",
                  f"hard_valid={row['original_corridor_hard_valid']}",
                  f"clearance={row['minimum_original_clearance_m']:.3f}m")
            if row["original_corridor_hard_valid"] != 1:
                raise RuntimeError(f"{mode} case {case} lost physical corridor validity")
    result = output / "source_accuracy_corpus_summary.csv"
    with result.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print("Report:", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
