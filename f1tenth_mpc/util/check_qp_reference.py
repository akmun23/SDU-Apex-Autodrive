#!/usr/bin/env python3
"""Check MPC QP solutions against OSQP, installed with `pip install osqp`."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import osqp
from scipy import sparse


def objective(p, x):
    t = np.maximum(0, -(np.array(p["elastic_jacobian"]).reshape(-1, len(x)) @ x + p["elastic_margin"]))
    return .5 * x @ p["hessian"] @ x + np.array(p["gradient"]) @ x + \
        .5 * np.array(p["elastic_weight"]) @ (t*t) + np.array(p["elastic_linear_cost"]) @ t


def reference(p):
    n, m = len(p["gradient"]), len(p["elastic_margin"])
    H, g = np.array(p["hessian"]), np.array(p["gradient"])
    C = np.array(p["elastic_jacobian"]).reshape(m, n)
    A = np.array(p["hard_jacobian"]).reshape(-1, n)
    scale = max(1., np.max(np.abs(H)), np.max(np.abs(g)),
                max(p["elastic_weight"], default=0), max(p["elastic_linear_cost"], default=0))
    P = sparse.block_diag([H, sparse.diags(p["elastic_weight"])], format="csc") / scale
    q = np.r_[g, p["elastic_linear_cost"]] / scale
    constraints = sparse.vstack([
        sparse.eye(n+m), sparse.hstack([A, sparse.csc_matrix((len(A), m))]),
        sparse.hstack([C, sparse.eye(m)])], format="csc")
    lower = np.r_[p["lower"], np.zeros(m), -np.array(p["hard_margin"]), -np.array(p["elastic_margin"])]
    upper = np.r_[p["upper"], np.full(m + len(A) + m, np.inf)]
    solver = osqp.OSQP()
    solver.setup(P=P, q=q, A=constraints, l=lower, u=upper, eps_abs=1.e-8,
                 eps_rel=1.e-9, max_iter=200000, polishing=True, verbose=False)
    result = solver.solve(raise_error=False)
    return result.info.status, result.x[:n] if result.x is not None else None


def generated(count):
    rng = np.random.default_rng(4070)
    for i in range(count):
        n = (1, 2, 6, 60)[i % 4]
        f = rng.normal(size=(n+2, n)) / np.sqrt(n)
        H = f.T @ f + .1*np.eye(n)
        g = rng.normal(size=n)*2
        a = rng.normal(size=(2*n, n)) / np.sqrt(n)
        c = rng.normal(size=(2*n, n)) / np.sqrt(n)
        p = dict(hessian=H.tolist(), gradient=g.tolist(), lower=(-np.ones(n)).tolist(),
                 upper=np.ones(n).tolist(), hard_jacobian=a.tolist(),
                 hard_margin=rng.uniform(.01, 1, 2*n).tolist(), elastic_jacobian=c.tolist(),
                 elastic_margin=rng.uniform(-.5, .5, 2*n).tolist(),
                 elastic_weight=rng.uniform(.1, 10, 2*n).tolist(),
                 elastic_linear_cost=(np.zeros(2*n) if i % 3 == 0 else rng.uniform(0, 100, 2*n)).tolist())
        if i % 5 == 0:
            p["lower"][0] = p["upper"][0] = 0.
        yield f"random_{i}", p, True
    p = dict(hessian=[[1.]], gradient=[0.], lower=[-1.], upper=[1.],
             hard_jacobian=[[1.], [-1.]], hard_margin=[-2., 1.],
             elastic_jacobian=[], elastic_margin=[], elastic_weight=[], elastic_linear_cost=[])
    yield "infeasible", p, False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", type=Path, required=True)
    parser.add_argument("--count", type=int, default=80)
    args = parser.parse_args()
    problems = list(generated(args.count))
    response = subprocess.run([str(args.solver.resolve())], input="".join(json.dumps(p)+"\n" for _, p, _ in problems),
                              capture_output=True, text=True, check=True)
    results = [json.loads(line) for line in response.stdout.splitlines()]
    if len(results) != len(problems):
        raise RuntimeError("missing solver results")
    failures = []
    iterations = []
    max_gap = 0.
    for (name, p, feasible), result in zip(problems, results):
        status, ref = reference(p)
        if not feasible:
            if result["converged"] or status != "primal infeasible":
                failures.append([name, result, status])
            continue
        if not result["converged"] or status != "solved":
            failures.append([name, result, status])
            continue
        x = np.array(result["step"])
        gap = abs(objective(p, x) - objective(p, ref))
        max_gap = max(max_gap, gap)
        a = np.array(p["hard_jacobian"]).reshape(-1, len(x))
        violation = max(0., np.max(np.array(p["lower"])-x), np.max(x-p["upper"]),
                        np.max(-(a @ x + p["hard_margin"]), initial=0.))
        if gap > 1.e-4*max(1., abs(objective(p, ref))) or violation > 1.e-7:
            failures.append([name, {"objective_gap": gap, "primal_violation": violation}, status])
        bound = result.get("objective_error_bound")
        if bound is None or gap > bound + 1.e-6*max(1., abs(objective(p, ref))):
            failures.append([name, {"objective_gap": gap, "certificate_bound": bound}, status])
        iterations.append(result["iterations"])
    print(json.dumps({"problems": len(problems), "failures": failures, "max_objective_gap": max_gap,
                      "iterations_p50": float(np.median(iterations)) if iterations else None,
                      "iterations_p99": float(np.percentile(iterations, 99)) if iterations else None}, indent=2))
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
