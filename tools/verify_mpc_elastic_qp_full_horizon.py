#!/usr/bin/env python3
"""Verify a native elastic 30-stage Riccati QP against explicit-slack OSQP.

The native ADMM eliminates wall slack analytically in its 2D/3D proximal.
The independent oracle instead introduces one nonnegative scalar xi for
EVERY active wall at EVERY stage and enforces a*x-b <= xi.

This is a QP-only mathematical parity test. It NEVER certifies a source
trajectory against real physical walls or Unity.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.mpc_qp_oracle import build_osqp_matrices, decode_problem


def load_osqp(optional_vendor: Path | None):
    if optional_vendor:
        sys.path.insert(0, str(optional_vendor))
    import osqp
    return osqp


def compare(fixture: Path, native_csv: Path | None, vendor: Path | None,
            tolerance: float) -> dict:
    raw = fixture.read_bytes()
    problem = decode_problem(raw)
    P, q, A, l, u = build_osqp_matrices(problem)
    osqp = load_osqp(vendor)
    solver = osqp.OSQP()
    solver.setup(P=P, q=q, A=A, l=l, u=u, verbose=False,
                 eps_abs=1e-9, eps_rel=1e-9, polishing=True, max_iter=80000)
    solved = solver.solve(raise_error=False)
    if solved.info.status != "solved" or solved.x is None:
        raise RuntimeError(f"independent OSQP not solved: {solved.info.status}")
    x = solved.x
    ax = np.asarray(A @ x)
    positive_upper = np.isfinite(u)
    positive_lower = np.isfinite(l)
    max_excess = max(
        float(np.max(np.maximum(0., ax[positive_upper] - u[positive_upper]))),
        float(np.max(np.maximum(0., l[positive_lower] - ax[positive_lower]))),
    )
    horizon = 30
    x_size = 31 * 23
    u_size = 30 * 2
    elastic = [(k, f, plane, problem["polygon_elastic_weights"][k])
               for k, polygon in enumerate(problem["state_polygons"])
               if k > 0 and problem["polygon_elastic_weights"][k] > 0
               for f, plane in enumerate(polygon)]
    if not elastic:
        raise RuntimeError("fixture has no elastic wall planes")
    max_slack_difference = 0.
    max_physical_violation = 0.
    total_slack_objective = 0.
    for i, (k, f, (ey, eps, bound, progress), weight) in enumerate(elastic):
        excess = (ey * x[k * 23] + eps * x[k * 23 + 1] +
                  progress * x[k * 23 + 20] - bound)
        xi = float(x[x_size + u_size + i])
        max_physical_violation = max(max_physical_violation, float(excess))
        max_slack_difference = max(max_slack_difference,
                                   abs(max(0., excess) - xi))
        total_slack_objective += .5 * weight * xi * xi
    report = {
        "fixture": str(fixture),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "oracled_problem": "30-stage 23-state 2-input QP with explicit positive wall slacks",
        "oracle_status": solved.info.status,
        "oracle_iterations": solved.info.iter,
        "oracle_primal_residual": float(solved.info.prim_res),
        "oracle_dual_residual": float(solved.info.dual_res),
        "oracle_objective": float(solved.info.obj_val),
        "max_constraint_violation": max_excess,
        "elastic_faces": len(elastic),
        "max_uncertified_physical_wall_violation_m": max_physical_violation,
        "max_slack_optimality_gap": max_slack_difference,
        "elastic_penalty_objective": total_slack_objective,
    }
    if max_excess > 5e-6 or max_slack_difference > 5e-6:
        raise AssertionError("independent OSQP violates explicit slack constraints")
    if native_csv:
        nominal_x = np.zeros((31, 23), dtype=np.float64)
        nominal_u = np.zeros((30, 2), dtype=np.float64)
        present_x, present_u = set(), set()
        with native_csv.open(newline="") as handle:
            for kind, stage, *values in csv.reader(handle):
                k = int(stage)
                if kind == "x" and 0 <= k <= 30 and len(values) == 23:
                    nominal_x[k] = list(map(float, values))
                    present_x.add(k)
                elif kind == "u" and 0 <= k < 30 and len(values) == 2:
                    nominal_u[k] = list(map(float, values))
                    present_u.add(k)
                else:
                    raise ValueError(f"bad native trace {kind!r} at stage {stage}")
        if len(present_x) != 31 or len(present_u) != 30:
            raise AssertionError("missing native state/control stage")
        dx = np.abs(nominal_x - x[:x_size].reshape(31, 23))
        du = np.abs(nominal_u -
                    x[x_size:x_size + u_size].reshape(30, 2))
        report["native_state_max_abs_error"] = float(np.max(dx))
        report["native_control_max_abs_error"] = float(np.max(du))
        report["worst_native_state_channel"] = list(
            map(int, np.unravel_index(np.argmax(dx), dx.shape)))
        report["worst_native_control_channel"] = list(
            map(int, np.unravel_index(np.argmax(du), du.shape)))
        if np.max(dx) > tolerance or np.max(du) > tolerance:
            raise AssertionError(f"native vs OSQP parity failed: {report}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path,
                        default=Path("/tmp/mpc_elastic_horizon_parity_20261103.fqp"))
    parser.add_argument("--native-csv", type=Path,
                        default=Path("/tmp/mpc_elastic_horizon_native_20261103.csv"))
    parser.add_argument("--no-native", action="store_true",
                        help="validate frozen OSQP feasibility only")
    parser.add_argument("--osqp-vendor", type=Path, default=None)
    parser.add_argument("--max-native-discrepancy", type=float, default=1e-4)
    args = parser.parse_args()
    try:
        report = compare(args.fixture,
                         None if args.no_native else args.native_csv,
                         args.osqp_vendor,args.max_native_discrepancy)
        print(json.dumps(report, sort_keys=True, indent=2))
        return 0
    except (OSError, ValueError, AssertionError, RuntimeError, ImportError) as error:
        print(f"elastic QP parity FAILED: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
