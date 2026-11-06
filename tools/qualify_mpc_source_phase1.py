#!/usr/bin/env python3
"""Qualify the production condensed source SQP from recorded warm starts.

No witness, alternate optimizer, command projection or wall-clock deadline is
used. A budget-exhausted or failed solve is evidence, never proof that the
recorded problem is physically unrecoverable. The native solver's iteration
limit remains a termination guard; reaching it does not count as convergence.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import subprocess
import shutil

import numpy as np

from tools import competition_failure_benchmark as benchmark

CONTROL_SCALE = np.tile([.15, .25], 30)
STATIONARITY_TOLERANCE = 1e-2
PRIMAL_TOLERANCE = 6e-7
COMPLEMENTARITY_TOLERANCE = 1e-4
DUAL_TOLERANCE = 1e-8


def physical_metrics(score: dict, multipliers: list[float] | None = None) -> dict:
    """Original objective/constraints only; no preferred-clearance penalty."""
    if not score.get("valid"):
        raise ValueError(f"invalid independent source evaluation: {score.get('reason')}")
    margins = np.asarray(score["hard_margins"], dtype=float)
    gradient = np.asarray(score["gradient_scaled"], dtype=float)
    jacobian = np.asarray(score["hard_jacobian_scaled"], dtype=float)
    if margins.shape != (480,) or gradient.shape != (60,) or jacobian.shape != (480, 60):
        raise ValueError("independent source oracle changed the original 30-stage problem")
    if not all(np.all(np.isfinite(v)) for v in (margins, gradient, jacobian)):
        raise ValueError("nonfinite independent source derivative")
    if not all(math.isfinite(score[key]) for key in ("original_cost", "minimum_clearance_m",
            "minimum_rate_margin", "minimum_box_margin")):
        raise ValueError("nonfinite independent physical score")
    negative = np.minimum(0., margins)
    result = {
        "original_racing_cost": score["original_cost"],
        "minimum_physical_clearance_m": score["minimum_clearance_m"],
        "maximum_hard_violation": float(max(0., -margins.min())),
        "physical_violation_half_squared": float(.5 * negative @ negative),
        # Wall penetration never receives a feasibility tolerance.
        "physically_feasible": score["minimum_clearance_m"] >= 0. and
            score["minimum_rate_margin"] >= -1e-7 and score["minimum_box_margin"] >= -1e-7,
        "dual_estimate_available": multipliers is not None and len(multipliers) == len(margins),
        "kkt_converged": False,
    }
    if result["dual_estimate_available"]:
        dual = np.asarray(multipliers, dtype=float)
        if not np.all(np.isfinite(dual)):
            raise ValueError("nonfinite source constraint multipliers")
        lagrangian_gradient = gradient - jacobian.T @ dual
        result.update(
            stationarity_inf_scaled=float(np.max(np.abs(lagrangian_gradient))),
            stationarity_inf_physical=float(np.max(np.abs(lagrangian_gradient / CONTROL_SCALE))),
            complementarity_inf=float(np.max(np.abs(dual * margins))),
            dual_feasibility_inf=float(max(0., -dual.min())),
        )
        result["kkt_converged"] = bool(result["physically_feasible"] and
            result["maximum_hard_violation"] <= PRIMAL_TOLERANCE and
            result["stationarity_inf_scaled"] <= STATIONARITY_TOLERANCE and
            result["complementarity_inf"] <= COMPLEMENTARITY_TOLERANCE and
            result["dual_feasibility_inf"] <= DUAL_TOLERANCE)
    return result


def affine_feasibility(score: dict, controls: list) -> dict:
    """Independent HiGHS LP; no Hessian, racing cost or SQP multipliers.

    A zero objective finds a physical affine witness in the legal control box.
    If HiGHS reports inconsistency, an independent minimum-common-relaxation LP
    must also have a strictly positive, dual-certified optimum. Solver failures
    are indeterminate, never evidence of physical or affine unrecoverability.
    """
    from scipy.optimize import linprog

    margins = np.asarray(score["hard_margins"], dtype=float)
    jacobian = np.asarray(score["hard_jacobian_scaled"], dtype=float)
    command = np.asarray(controls, dtype=float).ravel()
    maximum = float(score["max_steering_rad"])
    lower = (np.tile([-maximum, .001], 30) - command) / CONTROL_SCALE
    upper = (np.tile([maximum, 1.], 30) - command) / CONTROL_SCALE
    if jacobian.shape != (len(margins), 60) or command.shape != (60,) or not all(
            np.all(np.isfinite(value)) for value in (margins, jacobian, lower, upper)):
        raise ValueError("invalid original affine physical problem")
    scale = 1. / np.maximum.reduce([np.full(len(margins), 1e-6),
        np.max(np.abs(jacobian), axis=1), np.abs(margins)])
    matrix = jacobian * scale[:, None]
    rhs = margins * scale
    bounds = list(zip(lower, upper))
    options = {"primal_feasibility_tolerance": 1e-9, "dual_feasibility_tolerance": 1e-9}
    lp = linprog(np.zeros(60), A_ub=-matrix, b_ub=rhs, bounds=bounds,
                 method="highs", options=options)
    result = {"classification": "indeterminate", "lp_status": int(lp.status),
              "message": lp.message, "physical_unrecoverability_proven": False}
    if lp.success:
        residual = float(max(0., -np.min(margins + jacobian @ lp.x),
            np.max(lower - lp.x), np.max(lp.x - upper)))
        result["witness_maximum_violation"] = residual
        if residual <= 1e-8:
            result.update(classification="feasible", step_scaled=lp.x.tolist())
        return result
    if lp.status != 2:
        return result
    phase = linprog(np.r_[np.zeros(60), 1.],
        A_ub=np.column_stack([-matrix, -np.ones(len(margins))]), b_ub=rhs,
        bounds=bounds + [(0., None)], method="highs", options=options)
    result["relaxation_lp_status"] = int(phase.status)
    if phase.success:
        dual_bound = float(rhs @ phase.ineqlin.marginals + lower @ phase.lower.marginals[:60] +
                           upper @ phase.upper.marginals[:60])
        residual = float(max(0., -np.min(rhs + matrix @ phase.x[:60] + phase.x[60]),
            np.max(lower - phase.x[:60]), np.max(phase.x[:60] - upper)))
        dual_stationarity = np.r_[np.zeros(60), 1.] -             np.column_stack([-matrix, -np.ones(len(margins))]).T @ phase.ineqlin.marginals -             phase.lower.marginals - phase.upper.marginals
        dual_residual = float(np.max(np.abs(dual_stationarity)))
        result.update(minimum_scaled_common_relaxation=float(phase.fun),
            dual_lower_bound=dual_bound, primal_residual=residual, dual_residual=dual_residual,
            duality_gap=abs(float(phase.fun) - dual_bound))
        if residual <= 1e-8 and dual_residual <= 1e-8 and                 abs(float(phase.fun) - dual_bound) <= 1e-7 and dual_bound > 1e-8:
            result["classification"] = "infeasible"
    return result


class SourceOracle:
    """Persistent independent source integrator; never generates controls."""
    def __init__(self, tools: Path, case: Path):
        self.process = subprocess.Popen([str(tools / "mpc_source_nlp_oracle"), str(case)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        self.calls = 0

    def evaluate(self, controls, **options) -> dict:
        if len(controls) != 30 or any(len(u) != 2 for u in controls):
            raise ValueError("qualification controls must contain exactly 30 steering/throttle pairs")
        request = {"controls": controls, "derivatives": True, "preferred_clearance_weight": 0.}
        request.update(options)
        self.process.stdin.write(json.dumps(request, allow_nan=False) + "\n")
        self.process.stdin.flush()
        self.calls += 1
        line = self.process.stdout.readline()
        if not line:
            raise ValueError("independent source oracle terminated")
        return json.loads(line)

    def close(self):
        self.process.stdin.close()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.process.stdout.close()


def accepted_controls(trace: dict) -> list:
    if not trace["accepted"]:
        return trace["warm_start"]["controls"]
    if trace.get("normal_accepted",False):
        return trace["normal"]["controls"]
    if trace["safeguarded_accepted"]:
        return trace["safeguarded_trials"][-1]["primal"]["controls"]
    if trace["correction_accepted"]:
        return trace["correction"]["controls"]
    if not trace["trial_evaluated"]:
        raise ValueError("accepted SQP iteration has no original-source trial")
    return trace["trial"]["controls"]


def close(actual: float, expected: float, tolerance: float = 1e-7) -> bool:
    return math.isfinite(actual) and math.isfinite(expected) and \
        abs(actual - expected) <= tolerance * max(1., abs(expected))


def qualify_case(case: Path, tools: Path) -> dict:
    run = benchmark.read_json(case / "replay.json.gz")
    history, traces = run["history"], run.get("primal_trace", [])
    if not history or len(history) != len(traces):
        raise ValueError(f"missing complete per-SQP trace: {case}")
    initial_controls = [list(map(float, line.split())) for line in
        (case / "initial.controls").read_text().splitlines() if line.strip()]
    if traces[0]["warm_start"]["controls"] != initial_controls:
        # Text fixtures use enough digits to round-trip the source float.
        if not np.array_equal(np.asarray(traces[0]["warm_start"]["controls"], dtype=np.float32),
                              np.asarray(initial_controls, dtype=np.float32)):
            raise ValueError("primary qualification did not start from recorded controls")
    oracle = SourceOracle(tools, case)
    records, errors = [], []
    try:
        affine_cache = {}
        controls = traces[0]["warm_start"]["controls"]
        score = oracle.evaluate(controls)
        initial = physical_metrics(score)
        first_feasible_cost = initial["original_racing_cost"] if initial["physically_feasible"] else None
        for native, trace in zip(history, traces):
            if trace["warm_start"]["controls"] != controls or native["iteration"] != trace["iteration"]:
                raise ValueError("SQP trace does not preserve the accepted source primal")
            before = physical_metrics(score, trace.get("physical_multipliers") if native["original_duals_trustworthy"] else None)
            following = accepted_controls(trace)
            after_score = score if following == controls else oracle.evaluate(following)
            after = physical_metrics(after_score)
            if after["physically_feasible"] and first_feasible_cost is None:
                first_feasible_cost = after["original_racing_cost"]
            checks = {
                "complete_solver_iteration": not native["linearization_failed"] and
                    not native["qp_failed"] and native["qp_attempted"],
                "racing_cost_parity": close(after["original_racing_cost"], native["cost_after"]),
                "physical_violation_parity": close(after["physical_violation_half_squared"], native["physical_violation_after"]),
                "hard_violation_parity": close(after["maximum_hard_violation"], native["maximum_hard_violation_after"]),
                "clearance_parity": close(after["minimum_physical_clearance_m"], native["minimum_source_clearance_after"]),
                "feasible_iterate_preserved": not before["physically_feasible"] or after["physically_feasible"],
                "physical_violation_not_traded_for_cost": before["physically_feasible"] or
                    after["physical_violation_half_squared"] <= before["physical_violation_half_squared"],
                "feasible_racing_cost_not_increased": not before["physically_feasible"] or
                    after["original_racing_cost"] <= before["original_racing_cost"],
                "original_stationarity_parity": not native["original_duals_trustworthy"] or (
                    before["dual_estimate_available"] and
                    close(before["stationarity_inf_scaled"], native["original_nlp_stationarity_inf"])),
            }
            step = float(np.max(np.abs((np.asarray(following) - np.asarray(controls)).ravel() / CONTROL_SCALE)))
            checks["accepted_step_parity"] = close(step, native["accepted_step_inf_scaled"])
            low, high = native["qp_hessian_min_eigenvalue"], native["qp_hessian_max_eigenvalue"]
            checks["finite_positive_qp_curvature"] = math.isfinite(low) and math.isfinite(high) and 0. < low <= high
            checks["honest_qp_convergence"] = not native["qp_converged"] or (
                native["qp_primal"] <= 1e-8 and native["qp_dual"] <= native["qp_tolerance_used"] and
                native["qp_complementarity"] <= native["qp_tolerance_used"])
            if not all(checks.values()):
                errors.append({"iteration": native["iteration"], "failed_checks":
                    [key for key, value in checks.items() if not value]})
            audit = None
            if not trace["accepted"] or not native["qp_converged"] or not before["physically_feasible"]:
                key = np.asarray(controls, dtype=np.float32).tobytes()
                if key not in affine_cache:
                    affine_cache[key] = affine_feasibility(score, controls)
                audit = affine_cache[key]
            records.append({"affine_physical_feasibility": audit, "iteration": native["iteration"], "accepted": trace["accepted"],
                "before": before, "after": after, "checks": checks,
                "original_duals_trustworthy": native["original_duals_trustworthy"],
                "normal_step_computed": native["normal_step_computed"],
                "normal_only_step": native["normal_only_step"],
                "normal_primal_residual": native["normal_primal_residual"],
                "normal_dual_residual": native["normal_dual_residual"],
                "normal_linearized_violation": native["normal_linearized_violation"],
                "linearized_physical_violation_inf": native["linearized_physical_violation_inf"],
                "qp_primal_residual": native["qp_primal"], "qp_dual_residual": native["qp_dual"],
                "qp_complementarity_residual": native["qp_complementarity"],
                "qp_hessian_min_eigenvalue": low, "qp_hessian_max_eigenvalue": high,
                "qp_hessian_condition": high / low if low > 0. else None,
                "accepted_step_inf_scaled": step,
                "exact_nonlinear_evaluations": native["exact_source_evaluations"]})
            controls, score = following, after_score
        final_controls = [list(map(float, line.split())) for line in
            (case / "final.controls").read_text().splitlines() if line.strip()]
        if not np.array_equal(np.asarray(controls, dtype=np.float32), np.asarray(final_controls, dtype=np.float32)):
            raise ValueError("returned controls differ from the accepted SQP trace")
        final = physical_metrics(score, run.get("final_physical_multipliers"))
        passed = not errors and final["physically_feasible"] and \
            run["source_feasible_final"] == final["physically_feasible"]
        if run["original_nlp_kkt_converged"] and not final["kkt_converged"]:
            errors.append({"failed_checks": ["false_original_nlp_kkt_certificate"]})
            passed = False
        result = {"record_index": int(case.name.removeprefix("epoch_")),
            "recorded_warm_start_verified": True, "initial": initial, "final": final,
            "termination_reason": run["termination_reason"], "numerical_failure": run["numerical_failure"],
            "affine_classifications": {kind: sum(record["affine_physical_feasibility"] is not None and
                record["affine_physical_feasibility"]["classification"] == kind for record in records)
                for kind in ("feasible", "infeasible", "indeterminate")},
            "iterations": records, "oracle_evaluations": oracle.calls, "errors": errors,
            "first_feasible_racing_cost": first_feasible_cost,
            "racing_cost_improved_after_feasibility": first_feasible_cost is not None and
                final["original_racing_cost"] < first_feasible_cost,
            "feasibility_qualified": passed,
            "kkt_qualified": passed and final["kkt_converged"] and run["original_nlp_kkt_converged"]}
        benchmark.write_json(case / "phase1_qualification.json.gz", result)
        return result
    finally:
        oracle.close()


def qualify_perturbations(output: Path, tools: Path, iterations: int, repeat: int,
                          count: int, radius: float, jobs: int = 1) -> list[dict]:
    """Diagnostics always supplement the primary recorded warm start."""
    if count == 0:
        return []
    rng = np.random.default_rng(775)
    env = benchmark.solver_environment(benchmark.read_json(output / "params.json"), controls_only_trace=True)
    cases = []
    for record in (765, 769, 773, 774, 775):
        original = output / f"epoch_{record}"
        seed = np.loadtxt(original / "initial.controls")
        oracle = SourceOracle(tools, original)
        try:
            maximum_steering = oracle.evaluate(seed.tolist())["max_steering_rad"]
        finally:
            oracle.close()
        for index in range(count):
            case = output / "perturbations" / str(index) / f"epoch_{record}"
            case.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original / "source_context.mctx", case / "source_context.mctx")
            # Construction is explicit and logged; the SQP itself receives
            # these exact floats, without a hidden warm-start projection.
            displacement = rng.uniform(-radius, radius, 60).reshape(30, 2) * [.15, .25]
            controls = np.clip(seed + displacement, [-maximum_steering, .001],
                               [maximum_steering, 1.]).astype(np.float32)
            np.savetxt(case / "initial.controls", controls, fmt="%.17g")
            cases.append((record, index, case,
                float(np.max(np.abs((controls - seed).ravel() / CONTROL_SCALE)))))
    # Generate all seeds in fixed corpus/RNG order before concurrent work.
    # Each job owns its files and native oracle; repeat solves stay serial.
    def solve(entry):
        record, index, case, displacement = entry
        runs = []
        for _ in range(repeat):
            process = subprocess.run([str(tools / "mpc_source_exact_offline_optimize"), str(case),
                str(case / "initial.controls"), str(iterations), str(case / "final.controls")],
                env=env, capture_output=True, text=True, check=True, timeout=None)
            runs.append(json.loads(process.stdout))
        deterministic = all(benchmark.stable_result(run) == benchmark.stable_result(runs[0])
                            for run in runs[1:])
        benchmark.write_json(case / "replay.json.gz", runs[0])
        check = qualify_case(case, tools)
        return {"record_index": record, "perturbation": index,
            "maximum_perturbation_inf_scaled": displacement,
            "controls_sha256": benchmark.sha(case / "initial.controls"),
            "deterministic": deterministic, "final": check["final"],
            "feasibility_qualified": check["feasibility_qualified"] and deterministic,
            "kkt_qualified": check["kkt_qualified"] and deterministic}
    if jobs == 1:
        return list(map(solve, cases))
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        return list(pool.map(solve, cases))


def qualify(bundle: Path, output: Path, tools: Path, iterations: int = 2000,
            repeat: int = 2, baseline_report: Path | None = None,
            perturbations: int = 2, perturbation_radius: float = .025, jobs: int = 1) -> dict:
    if not 1 <= iterations <= 2000 or repeat < 2:
        raise ValueError("qualification requires 1..2000 SQP iterations and at least two identical runs")
    if perturbations < 0 or not math.isfinite(perturbation_radius) or not 0. < perturbation_radius <= 1.:
        raise ValueError("perturbation count must be nonnegative; normalized radius must be in (0,1]")
    if jobs < 1:
        raise ValueError("qualification jobs must be positive")
    if output.resolve().is_relative_to(bundle.resolve()):
        raise ValueError("qualification output must be outside the immutable corpus")
    if baseline_report and baseline_report.resolve().is_relative_to(output.resolve()):
        raise ValueError("comparison baseline must be outside the qualification output directory")
    if output.exists() and any(output.iterdir()):
        raise ValueError("qualification output must be empty; preserve previous evidence")
    names=("mpc_source_nlp_oracle", "mpc_source_live_fixture",
           "mpc_source_exact_offline_optimize", "mpc_source_feasibility_witness_probe")
    for name in names:
        if not (tools/name).is_file():
            raise ValueError(f"build {name} before qualification")
    native=output/"native"
    native.mkdir(parents=True)
    for name in names:
        shutil.copy2(tools/name, native/name)
    tools=native
    replay = benchmark.replay(bundle, output, tools, str(iterations), repeat,
                              solver_timeout=None, controls_only_trace=True, jobs=jobs)
    frozen = benchmark.read_json(bundle / "oracle_baseline.json.gz")["epochs"]
    reference = None
    if baseline_report:
        prior = benchmark.read_json(baseline_report)
        if prior.get("schema") != "source-sqp-phase1-qualification/v1":
            raise ValueError("comparison baseline must be a prior Phase 1 qualification report")
        reference = {row["record_index"]: row for row in prior["epochs"]}
        if set(reference) != {row["record_index"] for row in replay["epochs"]}:
            raise ValueError("comparison baseline must contain the same complete recorded corpus")
    epochs = []
    for row in replay["epochs"]:
        record = row["record_index"]
        check = qualify_case(output / f"epoch_{record}", tools)
        old = reference[record] if reference else frozen[str(record)]["reconstructed_sqp"]
        old_feasible = old["final"]["physically_feasible"] if "final" in old else old["source_feasible_final"]
        old_cost = old["final"]["original_racing_cost"] if "final" in old else old["final_cost"]
        epochs.append({"record_index": record, "initial": check["initial"], "final": check["final"],
            "termination_reason": check["termination_reason"], "numerical_failure": check["numerical_failure"],
            "affine_classifications": check["affine_classifications"],
            "deterministic": row["deterministic"], "feasibility_qualified": check["feasibility_qualified"],
            "kkt_qualified": check["kkt_qualified"], "iteration_check_failures": len(check["errors"]),
            "baseline_feasible": old_feasible,
            "baseline_racing_cost_delta": check["final"]["original_racing_cost"] - old_cost,
            "regressed_feasibility": old_feasible and not check["final"]["physically_feasible"]})
    perturbed = qualify_perturbations(output, tools, iterations, repeat, perturbations, perturbation_radius, jobs)
    perturbations_qualified = len(perturbed) >= 10 and all(row["kkt_qualified"] for row in perturbed)
    result = {"schema": "source-sqp-phase1-qualification/v1", "epochs": epochs,
        "perturbations": perturbed, "perturbation_radius_scaled": perturbation_radius,
        "perturbations_qualified": perturbations_qualified,
        "corpus_sha256": benchmark.sha(bundle / "manifest.json"),
        "solver_sha256": benchmark.sha(tools / "mpc_source_exact_offline_optimize"),
        "oracle_sha256": benchmark.sha(tools / "mpc_source_nlp_oracle"),
        "baseline_sha256": benchmark.sha(baseline_report or bundle / "oracle_baseline.json.gz"),
        "maximum_sqp_iterations": iterations, "repeat": repeat, "wall_clock_deadline": None,
        "witness_initialization": False, "physical_unrecoverability_proven": False,
        "feasible_cases": sum(row["final"]["physically_feasible"] for row in epochs),
        "kkt_cases": sum(row["kkt_qualified"] for row in epochs),
        "feasibility_qualified": replay["passed"] and all(row["feasibility_qualified"] for row in epochs),
        "passed": replay["passed"] and perturbations_qualified and all(row["kkt_qualified"] and not row["regressed_feasibility"] for row in epochs),
        "scope": "offline source qualification; no Unity or real-time claim"}
    benchmark.write_json(output / "phase1_report.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=benchmark.DEFAULT_BUNDLE)
    parser.add_argument("--tools", type=Path, default=benchmark.DEFAULT_TOOLS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--jobs", type=int, default=1,
                        help="independent recorded and perturbed cases to solve concurrently")
    parser.add_argument("--baseline-report", type=Path)
    parser.add_argument("--perturbations", type=int, default=2,
                        help="starts per difficult epoch; 0 runs primary qualification only")
    parser.add_argument("--perturbation-radius", type=float, default=.025,
                        help="maximum displacement in normalized control coordinates")
    args = parser.parse_args()
    try:
        result = qualify(args.bundle.resolve(), args.output.resolve(), args.tools.resolve(),
            args.iterations, args.repeat, args.baseline_report, args.perturbations,
            args.perturbation_radius, args.jobs)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0 if result["passed"] else 1
    except (ValueError, KeyError, OSError, subprocess.SubprocessError) as error:
        print(json.dumps({"passed": False, "error": str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
