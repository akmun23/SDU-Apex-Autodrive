#!/usr/bin/env python3
"""Offline IPOPT reference for the original 60-control condensed source NLP.

IPOPT owns globalization, restoration, curvature updates and termination.
The native oracle owns the exact source objective, dynamics and derivatives.
This tool neither links into the controller nor publishes driving commands.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

# Small dense NLPs must not spawn a BLAS worker per host CPU. Set before imports.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import casadi as ca
import numpy as np

from tools import competition_failure_benchmark as benchmark
from tools.qualify_mpc_source_phase1 import CONTROL_SCALE, SourceOracle, physical_metrics


class SourceProblem:
    """One-point cache shared by IPOPT value/derivative callbacks."""
    def __init__(self, tools: Path, case: Path, controls: np.ndarray):
        self.oracle = SourceOracle(tools, case)
        self.key = None
        self.score = None
        self.last_z = None
        self.value(controls.ravel() / CONTROL_SCALE)
        self.row_types = np.asarray(self.score["row_types"])
        self.physical_rows = np.flatnonzero(self.row_types != "box")
        self.box_rows = np.flatnonzero(self.row_types == "box")
        if len(self.physical_rows) != 360 or len(self.box_rows) != 120:
            raise ValueError("Reference NLP must retain all original physical constraints")
        self.lower = np.tile([-self.score["max_steering_rad"], .001], 30) / CONTROL_SCALE
        self.upper = np.tile([self.score["max_steering_rad"], 1.], 30) / CONTROL_SCALE

    def value(self, coordinates):
        z = np.asarray(coordinates, dtype=float).ravel()
        if z.shape != (60,) or not np.all(np.isfinite(z)):
            raise ValueError("IPOPT controls must contain 60 finite coordinates")
        # The authoritative source uses float commands. Do not smooth or
        # replace that interface with a different continuous vehicle model.
        controls = (z * CONTROL_SCALE).astype(np.float32).reshape(30, 2)
        key = controls.tobytes()
        if key != self.key:
            score = self.oracle.evaluate(controls.tolist())
            physical_metrics(score)  # Validate dimensions/finiteness at the boundary.
            if self.score is not None and score["row_types"] != self.score["row_types"]:
                raise ValueError("Original physical row layout changed during reference solve")
            self.score, self.key = score, key
        self.last_z = controls.ravel().astype(float) / CONTROL_SCALE
        return self.score

    def multipliers(self, score, lam_g, lam_x):
        # CasADi uses L = f + lam_g*g + lam_x*z. Original physical rows
        # use L = J - lambda*g, lambda >= 0. Legal box rows are represented
        # once through IPOPT bounds and mapped back for independent KKT checks.
        result = np.zeros(480)
        result[self.physical_rows] = -np.asarray(lam_g).ravel()
        jacobian = np.asarray(score["hard_jacobian_scaled"])
        bound_dual = np.asarray(lam_x).ravel()
        for row in self.box_rows:
            indices = np.flatnonzero(jacobian[row])
            if len(indices) != 1:
                raise ValueError("Original command box is no longer a coordinate bound")
            column = indices[0]
            slope = jacobian[row, column]
            result[row] = max(0., -bound_dual[column] / slope)
        return result


class SourceJacobian(ca.Callback):
    def __init__(self, name, problem, opts):
        ca.Callback.__init__(self)
        self.problem = problem
        self.construct(name, opts)

    def get_n_in(self): return 2  # z and the original function's output
    def get_n_out(self): return 1
    def get_sparsity_in(self, index):
        return ca.Sparsity.dense(60 if index == 0 else 361, 1)
    def get_sparsity_out(self, _): return ca.Sparsity.dense(361, 60)

    def eval(self, args):
        score = self.problem.value(args[0])
        matrix = np.vstack([score["gradient_scaled"],
            np.asarray(score["hard_jacobian_scaled"])[self.problem.physical_rows]])
        return [matrix]


class SourceFunction(ca.Callback):
    def __init__(self, problem):
        ca.Callback.__init__(self)
        self.problem = problem
        self.jacobian_callbacks = []  # CasADi retains weak callback references.
        self.construct("original_source")

    def get_n_in(self): return 1
    def get_n_out(self): return 1
    def get_sparsity_in(self, _): return ca.Sparsity.dense(60, 1)
    def get_sparsity_out(self, _): return ca.Sparsity.dense(361, 1)
    def has_jacobian(self): return True

    def get_jacobian(self, name, inames, onames, opts):
        callback = SourceJacobian(name, self.problem, opts)
        self.jacobian_callbacks.append(callback)
        return callback

    def eval(self, args):
        score = self.problem.value(args[0])
        return [np.r_[score["original_cost"],
                      np.asarray(score["hard_margins"])[self.problem.physical_rows]]]


class IterationRecorder(ca.Callback):
    def __init__(self, problem, journal):
        ca.Callback.__init__(self)
        self.problem = problem
        self.history = []
        self.journal = journal
        self.construct("reference_iterations")

    def get_n_in(self): return ca.nlpsol_n_out()
    def get_n_out(self): return 1
    def get_name_in(self, index): return ca.nlpsol_out(index)
    def get_name_out(self, _): return "stop"
    def get_sparsity_in(self, index):
        name = ca.nlpsol_out(index)
        return ca.Sparsity.dense({"x": 60, "f": 1, "g": 360,
            "lam_x": 60, "lam_g": 360, "lam_p": 0}[name], 1)
    def get_sparsity_out(self, _): return ca.Sparsity.scalar()

    def eval(self, args):
        values = dict(zip(ca.nlpsol_out(), args))
        score = self.problem.value(values["x"])
        dual = self.problem.multipliers(score, values["lam_g"], values["lam_x"])
        record = {"iteration": len(self.history),
            "controls": (self.problem.last_z * CONTROL_SCALE).reshape(30, 2).tolist(),
            "oracle_evaluations": self.problem.oracle.calls,
            **physical_metrics(score, dual.tolist())}
        self.history.append(record)
        self.journal.write(json.dumps(record, allow_nan=False) + "\n")
        self.journal.flush()
        return [0]


def solve(case: Path, tools: Path, output: Path, maximum_iterations: int) -> dict:
    controls = np.loadtxt(case / "initial.controls").astype(np.float32)
    if controls.shape != (30, 2):
        raise ValueError("Reference seed must be the original 30-stage recorded controls")
    output.mkdir(parents=True, exist_ok=True)
    journal = (output / "iterations.jsonl").open("w")
    problem = SourceProblem(tools, case, controls)
    try:
        initial = physical_metrics(problem.score)
        callback = SourceFunction(problem)
        recorder = IterationRecorder(problem, journal)
        z = ca.MX.sym("z", 60)
        values = callback(z)
        options = {"print_time": False, "iteration_callback": recorder,
            "ipopt.print_level": 0, "ipopt.sb": "yes",
            "ipopt.max_iter": maximum_iterations,
            "ipopt.hessian_approximation": "limited-memory",
            "ipopt.tol": 1e-7, "ipopt.dual_inf_tol": 1e-4,
            "ipopt.constr_viol_tol": 1e-9, "ipopt.compl_inf_tol": 1e-6,
            "ipopt.acceptable_iter": 0, "ipopt.bound_relax_factor": 0.,
            "ipopt.bound_push": 1e-9, "ipopt.bound_frac": 1e-9,
            "ipopt.nlp_scaling_method": "gradient-based"}
        solver = ca.nlpsol("source_reference", "ipopt", {"x": z, "f": values[0],
            "g": values[1:]}, options)
        started = time.monotonic()
        solution = solver(x0=controls.ravel().astype(float) / CONTROL_SCALE,
            lbx=problem.lower, ubx=problem.upper, lbg=0., ubg=ca.inf)
        elapsed = time.monotonic() - started
        stats = solver.stats()
        final_controls = (np.asarray(solution["x"]).ravel() * CONTROL_SCALE).astype(np.float32).reshape(30, 2)
        # New native process independently reintegrates the final commands.
        check = SourceOracle(tools, case)
        try:
            score = check.evaluate(final_controls.tolist())
            dual = problem.multipliers(score, solution["lam_g"], solution["lam_x"])
            final = physical_metrics(score, dual.tolist())
        finally:
            check.close()
        output.mkdir(parents=True, exist_ok=True)
        np.savetxt(output / "final.controls", final_controls, fmt="%.17g")
        result = {"schema": "source-ipopt-reference-case/v1", "case": case.name,
            "solver": "IPOPT", "casadi_version": ca.__version__,
            "hessian": "IPOPT limited-memory", "exact_source_first_derivatives": True,
            "extra_reserve_penalty": False, "source_command_precision": "float32",
            "control_dimension": 60, "physical_inequalities": 480,
            "physical_rows_in_ipopt": 360, "legal_control_bounds": 120,
            "maximum_iterations": maximum_iterations, "wall_clock_deadline": None,
            "witness_initialization": False, "controller_publication": False,
            "seed_sha256": benchmark.sha(case / "initial.controls"),
            "context_sha256": benchmark.sha(case / "source_context.mctx"),
            "oracle_sha256": benchmark.sha(tools / "mpc_source_nlp_oracle"),
            "initial": initial, "final": final,
            "ipopt_return_status": stats["return_status"],
            "ipopt_iteration_residuals": stats.get("iterations", {}),
            "ipopt_success": bool(stats["success"]), "iterations": int(stats["iter_count"]),
            "elapsed_seconds": elapsed, "oracle_evaluations": problem.oracle.calls,
            "physical_multipliers": dual.tolist(),
            "qualified": stats["return_status"] == "Solve_Succeeded" and final["kkt_converged"],
            "physical_unrecoverability_proven": False,
            "history": recorder.history}
        benchmark.write_json(output / "result.json.gz", result)
        return result
    finally:
        problem.oracle.close()
        journal.close()


def prepare_cases(bundle: Path, tools: Path, output: Path, epochs: list[int]) -> list[Path]:
    """Reconstruct frozen problems without calling the customized SQP."""
    manifest = benchmark.read_json(bundle / "manifest.json")
    for name, expected in manifest["bundle_sha256"].items():
        if benchmark.sha(bundle / name) != expected:
            raise ValueError(f"Corrupt frozen competition corpus: {name}")
    sequence = benchmark.read_json(bundle / "sequence.json.gz")
    frozen = benchmark.read_json(bundle / "oracle_baseline.json.gz")["epochs"]
    (output / "raceline.csv").write_bytes(gzip.decompress((bundle / "raceline.csv.gz").read_bytes()))
    benchmark.write_json(output / "params.json", sequence["parameters"])
    cases = []
    for entry in sequence["epochs"]:
        record = entry["record_index"]
        if record not in epochs:
            continue
        diagnostic = dict(entry["diagnostic"])
        diagnostic["frozen_source_problem"] = frozen[str(record)]["frozen_problem"]["frozen_source_problem"]
        case = output / f"epoch_{record}"
        case.mkdir()
        benchmark.write_json(case / "recorded.json", diagnostic)
        subprocess.run([str(tools / "mpc_source_live_fixture"), str(case / "recorded.json"),
            str(output / "params.json"), str(output / "raceline.csv"), str(case)],
            capture_output=True, text=True, check=True)
        if not benchmark.parity(benchmark.read_json(case / "frozen_problem.json"), diagnostic)["passed"]:
            raise ValueError(f"Original-source fixture parity failed: {record}")
        if not np.array_equal(np.loadtxt(case / "initial.controls").astype(np.float32),
                              np.asarray(diagnostic["nominal_controls"], dtype=np.float32)):
            raise ValueError(f"Recorded warm start changed: {record}")
        cases.append(case)
    if len(cases) != len(epochs):
        raise ValueError("Requested epochs are not in the 22-state corpus")
    return cases


def solve_repeats(case: Path, native: Path, iterations: int, repeat: int) -> dict:
    runs = [solve(case, native, case / f"repeat_{index}", iterations)
            for index in range(repeat)]
    signature = lambda run: {key: run[key] for key in
        ("final", "ipopt_return_status", "iterations", "oracle_evaluations", "history")}
    deterministic = all(signature(run) == signature(runs[0]) for run in runs[1:])
    result = {key: value for key, value in runs[0].items()
              if key not in ("history", "physical_multipliers", "ipopt_iteration_residuals")}
    result.update(deterministic=deterministic, repeat=repeat)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=benchmark.DEFAULT_BUNDLE)
    parser.add_argument("--tools", type=Path, default=benchmark.DEFAULT_TOOLS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, nargs="+", default=list(range(754, 776)))
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--jobs", type=int, default=1,
                        help="independent case processes; numerical settings are unchanged")
    args = parser.parse_args()
    if args.iterations < 1 or args.repeat < 1 or args.jobs < 1 or len(set(args.epochs)) != len(args.epochs):
        parser.error("Positive budgets and unique recorded epochs required")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    native = output / "native"
    native.mkdir()
    for name in ("mpc_source_live_fixture", "mpc_source_nlp_oracle"):
        shutil.copy2(args.tools.resolve() / name, native / name)
    cases = prepare_cases(args.bundle.resolve(), native, output, args.epochs)
    records = []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        pending = [pool.submit(solve_repeats, case, native, args.iterations, args.repeat) for case in cases]
        for completed in as_completed(pending):
            result = completed.result()
            records.append(result)
            records.sort(key=lambda row: row["case"])
            summary = {"schema": "source-ipopt-reference/v1", "epochs": records,
                "casadi_version": ca.__version__, "maximum_iterations": args.iterations,
                "repeat": args.repeat, "jobs": args.jobs,
                "corpus_sha256": benchmark.sha(args.bundle / "manifest.json"),
                "complete_corpus": len(records) == 22,
                "physically_feasible_cases": sum(row["final"]["physically_feasible"] for row in records),
                "kkt_cases": sum(row["qualified"] for row in records),
                "passed": len(records) == 22 and args.repeat >= 2 and
                    all(row["qualified"] and row["deterministic"] for row in records),
                "runtime_replacement_justified": len(records) == 22 and args.repeat >= 2 and
                    all(row["qualified"] and row["deterministic"] for row in records)}
            benchmark.write_json(output / "reference_report.json", summary)
            print(json.dumps({"case": result["case"], "status": result["ipopt_return_status"],
                "final": result["final"], "iterations": result["iterations"]}), flush=True)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
