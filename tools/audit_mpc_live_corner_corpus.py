#!/usr/bin/env python3
"""Audit exact live MPC QP snapshots against an independent double-precision LP.

The records are written asynchronously after a rejected controller cycle.
A feasible linearized QP does not imply a feasible nonlinear Unity trajectory.
An infeasible linearized QP does NOT prove the actual physical state is
unrecoverable; it only proves that particular affine local approximation
cannot simultaneously satisfy the encoded constraints.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from tools.audit_mpc_frozen_qp_feasibility import audit
from tools.mpc_qp_fixture import parse_payload

def classify(case: dict[str, Any]) -> str:
    if case["nonlinear_failure_stage"] == 0:
        return "current_epoch_corridor_violation"
    passes = case["passes"]
    if passes and all(p["linear_feasible"] is False for p in passes.values()):
        return "infeasible_local_linearization"
    if any(p["linear_feasible"] is False for p in passes.values()):
        return "some_passes_linearly_infeasible"
    if any(p["linear_feasible"] is True and p["raw_status"] != 0
           for p in passes.values()):
        return "solvable_linearization_but_admm_not_converged"
    if any(p["linear_feasible"] is True for p in passes.values()):
        return "linearized_qp_solved_but_nonlinear_rollout_rejected"
    return "unclassified"

def read_case(case_dir: Path) -> dict[str, Any]:
    context = json.loads((case_dir / "context.json").read_text())
    with (case_dir / "physical_state.csv").open() as file:
        rows = list(csv.reader(file))
    if len(rows) != 1 or len(rows[0]) != 38:
        raise ValueError(f"{case_dir}: expected 38 source state fields")
    with (case_dir / "nominal_source_seed.csv").open() as file:
        seed_rows = list(csv.reader(file))
    if len(seed_rows) != 1 or len(seed_rows[0]) != 61:
        raise ValueError(f"{case_dir}: expected zero + 30 nominal control pairs")
    if int(rows[0][0]) != context["command_epoch_ns"]:
        raise ValueError(f"{case_dir}: snapshot epoch mismatch")
    case = dict(context)
    case["directory"] = str(case_dir)
    case["passes"] = {}
    for stage in ("r1", "r2"):
        fixture = case_dir / f"{stage}.fqp"
        if not fixture.is_file():
            continue
        payload = fixture.read_bytes()
        source = parse_payload(payload)
        lp = audit(fixture)
        case["passes"][stage] = {
            "fixture": str(fixture),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "raw_status": int(source["raw_status"]),
            "raw_iterations": int(source["raw_iterations"]),
            "raw_primal_residual": float(source["raw_residuals"][0]),
            "raw_dual_residual": float(source["raw_residuals"][1]),
            "initial_solve_budget_us": source["initial_max_solve_time_us"],
            "linear_feasible": lp["feasible"],
            "minimum_linear_slack": lp.get("minimum_uniform_ltv_state_slack"),
            "worst_linear_constraints": lp.get("worst_stage_faces", [])[:3],
        }
    case["classification"] = classify(case)
    return case

def audit_directory(root: Path) -> dict[str, Any]:
    case_dirs = sorted(
        (p.parent for p in root.glob("epoch_*/context.json")),
        key=lambda d: json.loads((d / "context.json").read_text())["command_epoch_ns"],
    )
    cases = [read_case(directory) for directory in case_dirs]
    counts = Counter(case["classification"] for case in cases)
    return {
        "source": str(root.resolve()),
        "case_count": len(cases),
        "classification_counts": dict(sorted(counts.items())),
        "first_infeasible_local_linearization": next(
            (case["progress_m"] for case in cases
             if any(p["linear_feasible"] is False
                    for p in case["passes"].values())), None),
        "first_current_epoch_corridor_violation": next(
            (case["progress_m"] for case in cases
             if case["nonlinear_failure_stage"] == 0), None),
        "cases": cases,
        "limits": [
            "LP feasibility is for the captured LTV model, not the exact source plant.",
            "No local LTV infeasibility certificate establishes global nonlinear unreachability.",
            "A rejected exact-source 750ms rollout does not prove a Unity collision.",
            "No controller timing or completion qualification follows from QP captures.",
        ],
    }

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("capture_root", type=Path)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    result = audit_directory(args.capture_root)
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)
    print(json.dumps({k: v for k,v in result.items() if k != "cases"}, indent=2))

if __name__ == "__main__":
    main()
