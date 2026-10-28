#!/usr/bin/env python3
"""Solve a frozen controller QP with OSQP as an independent offline oracle.

The stored QP is reconstructed exactly from the explicit little-endian fixture
schema. This tool never loads the ROS controller or uses simulator truth.
Install tools/requirements_mpc_qp_oracle.txt in the analysis environment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from tools.mpc_qp_fixture import (Cursor, FixtureError, HEADER_SIZE, HORIZON,
                                  NX, NU, parse_payload)


def _load_osqp():
    try:
        import osqp
        return osqp
    except ModuleNotFoundError:
        vendor = Path(__file__).resolve().parents[1] / "build_task/qp_oracle_vendor"
        if vendor.is_dir():
            sys.path.insert(0, str(vendor))
            import osqp
            return osqp
        raise


def decode_problem(payload: bytes) -> dict:
    parse_payload(payload)  # schema, finite values, bounds, and exact EOF.
    cursor = Cursor(payload)
    cursor.offset = HEADER_SIZE
    steps = []
    for _ in range(HORIZON):
        A = np.asarray(cursor.f32_array(NX * NX)).reshape(NX, NX)
        B = np.asarray(cursor.f32_array(NX * NU)).reshape(NX, NU)
        d = np.asarray(cursor.f32_array(NX))
        Q = np.asarray(cursor.f32_array(NX))
        q = np.asarray(cursor.f32_array(NX))
        N = np.asarray(cursor.f32_array(NX * NU)).reshape(NX, NU)
        R = np.asarray(cursor.f32_array(NU))
        r = np.asarray(cursor.f32_array(NU))
        x_lb = np.asarray(cursor.f32_array(NX))
        x_ub = np.asarray(cursor.f32_array(NX))
        u_lb = np.asarray(cursor.f32_array(NU))
        u_ub = np.asarray(cursor.f32_array(NU))
        steps.append((A, B, d, Q, q, N, R, r, x_lb, x_ub, u_lb, u_ub))
    terminal_Q = np.asarray(cursor.f32_array(NX))
    terminal_q = np.asarray(cursor.f32_array(NX))
    terminal_lb = np.asarray(cursor.f32_array(NX))
    terminal_ub = np.asarray(cursor.f32_array(NX))
    x0 = np.asarray(cursor.f32_array(NX))
    polygons = []
    for _ in range(HORIZON + 1):
        count = cursor.signed()
        planes = [cursor.f32_array(3) for _ in range(8)]
        polygons.append(planes[:count])
    return {"steps": steps, "terminal_Q": terminal_Q,
            "terminal_q": terminal_q, "terminal_lb": terminal_lb,
            "terminal_ub": terminal_ub, "x0": x0, "state_polygons": polygons}


def build_osqp_matrices(problem: dict):
    nx, nu, horizon = NX, NU, HORIZON
    x_size = (horizon + 1) * nx
    nvar = x_size + horizon * nu

    def xidx(k, i):
        return k * nx + i

    def uidx(k, i):
        return x_size + k * nu + i

    pr, pc, pv = [], [], []
    linear = np.zeros(nvar, dtype=np.float64)

    def add_hessian(i, j, value):
        if value == 0.0:
            return
        if i > j:
            i, j = j, i
        pr.append(i)
        pc.append(j)
        pv.append(float(value))

    for k, step in enumerate(problem["steps"]):
        _, _, _, Q, q, N, R, r, *_ = step
        for i in range(nx):
            add_hessian(xidx(k, i), xidx(k, i), Q[i])
            linear[xidx(k, i)] += q[i]
        for a in range(nu):
            add_hessian(uidx(k, a), uidx(k, a), R[a])
            linear[uidx(k, a)] += r[a]
        for i in range(nx):
            for a in range(nu):
                add_hessian(xidx(k, i), uidx(k, a), N[i, a])
    for i in range(nx):
        add_hessian(xidx(horizon, i), xidx(horizon, i),
                    problem["terminal_Q"][i])
        linear[xidx(horizon, i)] += problem["terminal_q"][i]
    P = sparse.coo_matrix((pv, (pr, pc)), shape=(nvar, nvar)).tocsc()

    rows, cols, vals = [], [], []
    lower, upper = [], []

    def add_constraint(entries, lo, hi):
        row = len(lower)
        for column, value in entries:
            if value != 0.0:
                rows.append(row)
                cols.append(column)
                vals.append(float(value))
        lower.append(float(lo))
        upper.append(float(hi))

    # Pin the supplied initial state and impose all linear dynamics exactly.
    for i in range(nx):
        add_constraint([(xidx(0, i), 1.0)], problem["x0"][i],
                       problem["x0"][i])
    for k, step in enumerate(problem["steps"]):
        A, B, d = step[:3]
        for i in range(nx):
            entries = [(xidx(k + 1, i), 1.0)]
            entries.extend((xidx(k, j), -A[i, j]) for j in range(nx))
            entries.extend((uidx(k, a), -B[i, a]) for a in range(nu))
            add_constraint(entries, d[i], d[i])

    # Riccati ADMM deliberately excludes x0 from the split box constraints.
    # Its constrained state slots are x1..x(H-1) using step_data[k] bounds,
    # then xH using terminal bounds; step 0's state bounds are metadata only.
    for k in range(1, horizon):
        step = problem["steps"][k]
        for i in range(nx):
            add_constraint([(xidx(k, i), 1.0)], step[8][i], step[9][i])
    for i in range(nx):
        add_constraint([(xidx(horizon, i), 1.0)],
                       problem["terminal_lb"][i],
                       problem["terminal_ub"][i])
    for k, step in enumerate(problem["steps"]):
        for a in range(nu):
            add_constraint([(uidx(k, a), 1.0)], step[10][a], step[11][a])
    for k, polygon in enumerate(problem.get("state_polygons", [])):
        if k == 0:
            continue  # measured initial state is immutable, just as in Riccati
        for normal0, normal1, bound in polygon:
            add_constraint([(xidx(k, 0), normal0), (xidx(k, 1), normal1)],
                           -np.inf, bound)
    Acon = sparse.coo_matrix((vals, (rows, cols)),
                             shape=(len(lower), nvar)).tocsc()
    return P, linear, Acon, np.asarray(lower), np.asarray(upper)


def minimum_uniform_box_relaxation(A, lower: np.ndarray, upper: np.ndarray,
                                  equality_count: int,
                                  selected_variables: np.ndarray) -> dict:
    """Find the smallest common box widening that restores QP feasibility.

    Dynamics and the initial state remain exact. A single nonnegative slack is
    added only to the selected state/input box constraints. This diagnoses
    which constraint family blocks the linearized QP; it does not modify or
    propose weakening production constraints.
    """
    nvar = A.shape[1]
    box = A[equality_count:].tocsr()
    # Widen selected box rows only. Coupled footprint rows remain exact.
    is_box = np.diff(box.indptr) == 1
    variables = np.zeros(box.shape[0], dtype=int)
    for row in np.flatnonzero(is_box):
        offset = box.indptr[row]
        is_box[row] = abs(box.data[offset] - 1.) <= 1.e-12
        variables[row] = box.indices[offset]
    selected = is_box & selected_variables(variables)

    upper_rows = np.flatnonzero(np.isfinite(upper[equality_count:]))
    lower_rows = np.flatnonzero(np.isfinite(lower[equality_count:]))
    A_upper = box[upper_rows]
    A_lower = -box[lower_rows]
    upper_slack = sparse.csr_matrix(
        (-selected[upper_rows].astype(np.float64),
         (np.arange(upper_rows.size), np.zeros(upper_rows.size, dtype=int))),
        shape=(upper_rows.size, 1))
    lower_slack = sparse.csr_matrix(
        (-selected[lower_rows].astype(np.float64),
         (np.arange(lower_rows.size), np.zeros(lower_rows.size, dtype=int))),
        shape=(lower_rows.size, 1))
    A_ub = sparse.vstack((
        sparse.hstack((A_upper, upper_slack)),
        sparse.hstack((A_lower, lower_slack))), format="csr")
    b_ub = np.concatenate((upper[equality_count:][upper_rows],
                           -lower[equality_count:][lower_rows]))
    A_eq = sparse.hstack((A[:equality_count],
                          sparse.csr_matrix((equality_count, 1))), format="csr")
    c = np.zeros(nvar + 1, dtype=np.float64)
    c[-1] = 1.0
    result = linprog(
        c, A_ub=A_ub, b_ub=b_ub,
        A_eq=A_eq, b_eq=lower[:equality_count],
        bounds=[(None, None)] * nvar + [(0.0, None)], method="highs")
    status = {0: "feasible", 2: "infeasible"}.get(
        result.status, f"status-{result.status}")
    return {
        "status": status,
        "minimum_uniform_relaxation": (
            float(result.x[-1]) if result.x is not None else None),
    }


def solve(path: Path, max_iter: int = 25_000,
          time_limit_s: float = 0.2) -> dict:
    osqp = _load_osqp()
    payload = path.read_bytes()
    problem = decode_problem(payload)
    P, q, A, lower, upper = build_osqp_matrices(problem)
    solver = osqp.OSQP()
    solver.setup(P=P, q=q, A=A, l=lower, u=upper,
                 eps_abs=1e-8, eps_rel=1e-8, max_iter=max_iter,
                 polishing=True, verbose=False, adaptive_rho=True,
                 check_termination=25, time_limit=time_limit_s)
    result = solver.solve(raise_error=False)
    status = result.info.status
    if result.x is None:
        raise FixtureError(f"OSQP failed on {path.name}: {status}")
    stage_hessian_min_eigenvalue = min(
        float(np.linalg.eigvalsh(np.block([
            [np.diag(step[3]), step[5]],
            [step[5].T, np.diag(step[6])],
        ])).min()) for step in problem["steps"])
    stage_hessian_min_eigenvalue = min(
        stage_hessian_min_eigenvalue,
        float(np.min(problem["terminal_Q"])))
    feasibility_lp_status = None
    infeasibility_relaxation_by_group = None
    if "primal infeasible" in status.lower():
        equality_count = NX * (HORIZON + 1)
        inequality = A[equality_count:]
        lo, hi = lower[equality_count:], upper[equality_count:]
        lower_rows, upper_rows = np.isfinite(lo), np.isfinite(hi)
        lp = linprog(
            np.zeros(P.shape[0]),
            A_ub=sparse.vstack((inequality[upper_rows], -inequality[lower_rows])).tocsr(),
            b_ub=np.r_[hi[upper_rows], -lo[lower_rows]],
            A_eq=A[:equality_count], b_eq=lower[:equality_count],
            bounds=[(-np.inf, np.inf)]*P.shape[0], method="highs")
        feasibility_lp_status = {
            0: "feasible", 2: "infeasible"}.get(lp.status, f"status-{lp.status}")
        if feasibility_lp_status == "infeasible":
            x_size = (HORIZON + 1) * NX
            groups = {
                "lateral_error_corridor_e_y_m":
                    lambda variables: (variables < x_size) &
                        ((variables % NX) == 0),
                "other_state_bounds":
                    lambda variables: (variables < x_size) &
                        ((variables % NX) != 0),
                "control_bounds": lambda variables: variables >= x_size,
                "all_state_bounds": lambda variables: variables < x_size,
                "all_box_bounds": lambda variables: np.ones(
                    variables.shape, dtype=bool),
            }
            infeasibility_relaxation_by_group = {
                name: minimum_uniform_box_relaxation(
                    A, lower, upper, equality_count, select)
                for name, select in groups.items()
            }
    return {
        "fixture": path.name,
        "oracle": "OSQP 1.0.4, eps_abs=eps_rel=1e-8, polishing",
        "status": status,
        "feasibility_lp_status": feasibility_lp_status,
        "infeasibility_relaxation_by_group":
            infeasibility_relaxation_by_group,
        "iterations": result.info.iter,
        "solve_time_s": float(result.info.run_time),
        "configured_iteration_limit": max_iter,
        "configured_time_limit_s": time_limit_s,
        "objective": float(result.info.obj_val),
        "primal_residual": float(result.info.prim_res),
        "dual_residual": float(result.info.dual_res),
        "variables": int(P.shape[0]),
        "constraints": int(A.shape[0]),
        "hessian_min_eigenvalue": stage_hessian_min_eigenvalue,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--max-iter", type=int, default=25_000)
    parser.add_argument("--time-limit-s", type=float, default=0.2)
    args = parser.parse_args()
    try:
        print(json.dumps(solve(args.fixture, args.max_iter,
                               args.time_limit_s), sort_keys=True))
        return 0
    except (OSError, FixtureError, ImportError, RuntimeError,
            ValueError) as error:
        print(f"mpc_qp_oracle: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
