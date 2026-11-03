#!/usr/bin/env python3
"""Compare EXACT frozen QPs against OSQP and full Unity-source rollouts.

No simulator writes, controller configuration changes, or publishable actions.
Run from the repo root with an OSQP-capable Python environment and the
`mpc_live_failure_replay` C++ development binary built. This deliberately
separates: (1) QP feasibility, (2) OSQP solution, (3) exact-source validity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tempfile

from tools.mpc_qp_fixture import HORIZON, NX, NU
from tools.mpc_qp_oracle import _load_osqp, build_osqp_matrices, decode_problem

REPLAY = Path("build/f1tenth_mpc/mpc_live_failure_replay")


def strictify_restoration(problem: dict, relaxation_m: float) -> None:
    """Undo ONLY explicit +relaxation on stored wall constraints.

    This leaves the captured restoration Hessian/objective unchanged. It is a
    stricter *linearized feasibility oracle*, not a recreated racing QP.
    """
    for k in range(1, HORIZON):
        problem["steps"][k][8][0] += relaxation_m
        problem["steps"][k][9][0] -= relaxation_m
    problem["terminal_lb"][0] += relaxation_m
    problem["terminal_ub"][0] -= relaxation_m
    for k in range(1, HORIZON + 1):
        problem["state_polygons"][k] = [
            (ey, epsi, limit - relaxation_m)
            for ey, epsi, limit in problem["state_polygons"][k]
        ]


def score_control_decisions(case: Path, pass_name: str,
                            decisions, workspace: Path) -> dict:
    filename = workspace / f"{case.name}_{pass_name.lower()}_osqp_controls.txt"
    filename.write_text("\n".join(
        f"{decision[0]:.15g} {decision[1]:.15g}" for decision in decisions
    ) + "\n")
    mode = "--external-qp-decisions-r2" if pass_name == "R2" else \
        "--external-qp-decisions"
    p = subprocess.run([str(REPLAY), str(case), f"{mode}={filename}"],
                       capture_output=True, text=True, timeout=40, check=True)
    records = []
    for line in p.stdout.splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("external_source_controls"):
            records.append(record)
    if len(records) != 1:
        raise ValueError(f"missing unique source outcome: {case}, {pass_name}")
    return records[0]


def run(cases_dir: Path, case_numbers: list[int], relaxation_m: float,
        output: Path, time_limit: float):
    osqp = _load_osqp()
    if not REPLAY.exists():
        raise FileNotFoundError(f"build the developer tool first: {REPLAY}")
    cases = sorted(p for p in cases_dir.iterdir() if p.is_dir())
    if case_numbers:
        cases = [p for p in cases if any(p.name.endswith(f"_case_{i}")
                                         for i in case_numbers)]
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mpc-osqp-source-") as tmpdir, \
         output.open("w") as saved:
        for case in cases:
            for pass_name in ("R1", "R2"):
                qpfile = case / (pass_name.lower() + ".fqp")
                if not qpfile.exists() or not (case / "source_context.mctx").exists():
                    continue
                for variant in ("as_captured", "undo_restoration_relaxation"):
                    problem = decode_problem(qpfile.read_bytes())
                    if variant != "as_captured":
                        strictify_restoration(problem, relaxation_m)
                    P, q, A, lo, hi = build_osqp_matrices(problem)
                    solver = osqp.OSQP()
                    solver.setup(P=P, q=q, A=A, l=lo, u=hi,
                                 eps_abs=1e-7, eps_rel=1e-7,
                                 max_iter=25000, time_limit=time_limit,
                                 polishing=True, verbose=False)
                    solved = solver.solve(raise_error=False)
                    entry = dict(case=case.name, pass_name=pass_name,
                                 variant=variant, qp_status=solved.info.status,
                                 qp_iterations=int(solved.info.iter),
                                 qp_primal_residual=float(solved.info.prim_res),
                                 qp_dual_residual=float(solved.info.dual_res),
                                 qp_solve_ms=float(solved.info.run_time) * 1000,
                                 source_world_map_callback_replayed=False)
                    if solved.x is not None and solved.info.status.startswith("solved"):
                        controls = solved.x[(HORIZON + 1)*NX:].reshape(HORIZON, NU)
                        entry.update(score_control_decisions(case, pass_name,
                                                             controls, Path(tmpdir)))
                    saved.write(json.dumps(entry, allow_nan=False) + "\n")
                    saved.flush()
                    print(f"{case.name} {pass_name} {variant}: "
                          f"{entry['qp_status']}, iterations={entry['qp_iterations']}, "
                          f"source_valid={entry.get('physical_valid', 'not tested')}, "
                          f"source_failure_stage={entry.get('failure_stage', '-')}, "
                          f"state_error={entry.get('qp_source_max_state_error', '-')}",
                          flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_directory", type=Path)
    parser.add_argument("--cases", nargs="*", type=int, default=[])
    parser.add_argument("--restoration-relaxation-m", type=float,
                        default=.18, help="Known capture-specific relaxation; verify it")
    parser.add_argument("--time-limit-s", type=float, default=1.5)
    parser.add_argument("--output", type=Path,
                        default=Path("/tmp/mpc_osqp_source_fidelity.jsonl"))
    args = parser.parse_args()
    if args.restoration_relaxation_m <= 0:
        parser.error("restoration relaxation must be positive")
    run(args.case_directory, args.cases,
        args.restoration_relaxation_m, args.output, args.time_limit_s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
