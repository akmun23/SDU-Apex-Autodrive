#!/usr/bin/env python3
"""Evaluate frozen, exact live MPC QPs under alternative ADMM penalty policies.

Offline only: never changes the controller, frozen Hessian/objective, physical
constraints or warm states. Scores convergence against independently audited
linearized feasibility; infeasible QPs are not called solver failures.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BINARY = REPO / "build/f1tenth_mpc/mpc_qp_fixture_replay"
FIELDS = (
    "case", "progress_m", "pass", "linearly_feasible", "seed_mode", "policy",
    "state_rho", "input_rho", "rho_max", "over_relaxation",
    "iterations", "status", "primal_residual", "dual_residual",
    "primal_objective", "max_primal_dynamics_equality_residual",
    "max_primal_state_bound_violation", "max_primal_input_bound_violation",
    "recorded_result_exact_parity"
)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--all-cases", action="store_true")
    parser.add_argument("--max-cases", type=int, default=7)
    parser.add_argument("--max-iterations", type=int, default=150)
    args = parser.parse_args()
    if args.max_iterations < 1 or args.max_iterations > 300:
        parser.error("max-iterations outside solver compiled 1..300")
    if not BINARY.is_file():
        parser.error(f"compile {BINARY} first")
    audit = json.loads(args.audit.read_text())
    cases = audit["cases"]
    if not args.all_cases:
        # Include earliest and worst residuals of each class, preserving
        # independently verified feasibility/infeasibility distinction.
        grouped: dict[str, list[dict]] = {}
        for case in cases:
            grouped.setdefault(case["classification"], []).append(case)
        selected = {group[0]["directory"]: group[0] for group in grouped.values()}
        for group in grouped.values():
            worst = max(group, key=lambda c: max(
                (p["raw_primal_residual"] + p["raw_dual_residual"])
                for p in c["passes"].values()) if c["passes"] else 0.)
            selected[worst["directory"]] = worst
        cases = sorted(selected.values(), key=lambda c: c["progress_m"])
        cases = cases[:args.max_cases]
    # Numeric search is explicitly diagnostic; do not promote the winning rho
    # without cross-fixture and full-replay parity / realtime qualification.
    configs = [
        ("recorded", 0., 0., 0., 0.),
        ("balanced_4", 4., 4., 32., 1.6),
        ("balanced_12", 12., 12., 32., 1.6),
        ("balanced_24", 24., 24., 48., 1.6),
        ("state_24_input_7", 24., 7., 48., 1.6),
        ("state_7_input_24", 7., 24., 48., 1.6),
        ("state_24_input_24_alpha_1p2", 24., 24., 48., 1.2),
        ("state_7_input_7_alpha_1p9", 7., 7., 32., 1.9),
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=FIELDS)
        writer.writeheader()
        for case in cases:
            for pass_name, captured in case["passes"].items():
                fixture = Path(captured["fixture"])
                for label, rho_x, rho_u, rho_max, alpha in configs:
                    env = os.environ.copy()
                    env["MPC_QP_BENCH_MAX_ITER"] = str(args.max_iterations)
                    if label != "recorded":
                        env.update(
                            MPC_QP_BENCH_STATE_RHO=str(rho_x),
                            MPC_QP_BENCH_INPUT_RHO=str(rho_u),
                            MPC_QP_BENCH_RHO_MAX=str(rho_max),
                            MPC_QP_BENCH_OVER_RELAXATION=str(alpha),
                        )
                    p = subprocess.run(
                        [str(BINARY), str(fixture), "recorded"],
                        cwd=REPO, env=env, text=True, capture_output=True,
                        timeout=40,
                    )
                    if p.returncode:
                        raise RuntimeError(f"{fixture} {label}: {p.stderr[-500:]}")
                    result = json.loads(p.stdout)
                    row = {
                        "case": case["directory"],
                        "progress_m": case["progress_m"],
                        "pass": pass_name,
                        "linearly_feasible": captured["linear_feasible"],
                        "seed_mode": "recorded",
                        "policy": label,
                        "state_rho": rho_x or captured.get("initial_rho"),
                        "input_rho": rho_u or captured.get("initial_rho_u"),
                        "rho_max": rho_max or "recorded",
                        "over_relaxation": alpha or "recorded",
                        **{k: result[k] for k in FIELDS if k in result},
                    }
                    writer.writerow(row)
                    print(f"{case['progress_m']:.3f} {pass_name:2} {label:33}"
                          f" status={row['status']} r=({row['primal_residual']:.4g},"
                          f"{row['dual_residual']:.4g})", flush=True)

if __name__ == "__main__":
    main()
