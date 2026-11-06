#!/usr/bin/env python3
"""Measure the minimum *required* wall slack of a frozen MPC elastic QP.

This is a geometrically meaningful linear-programming diagnostic, NOT a new
controller or a nonlinear source feasibility certificate. Keep the frozen
23-state LTV dynamics, command constraints and unrelated state/terminal bounds
HARD. Relax ONLY the explicit elastic wall faces by ONE nonnegative scalar t:

  min t  subject to x[k+1]=A_k*x[k]+B_k*u[k]+d_k,
                    hard state and input bounds,
                    original wall[k,f] <= t.

If optimum t is positive, the frozen local affine QP has no hard-wall-feasible
trajectory, regardless of SQP objective weights or ADMM iterations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.mpc_qp_oracle import build_osqp_matrices,decode_problem


def audit(fixture:Path) -> dict:
    problem=decode_problem(fixture.read_bytes())
    P,_,A,lower,upper=build_osqp_matrices(problem)
    n=P.shape[0]
    x_count=31*23
    u_count=30*2
    weights=problem["polygon_elastic_weights"]
    elastic=[(stage,i) for stage,poly in enumerate(problem["state_polygons"])
             if stage>0 and weights[stage]>0.
             for i in range(len(poly))]
    if not elastic:
        raise ValueError("the frozen QP has no elastic wall faces")
    assert n==x_count+u_count+len(elastic)
    new_col=sparse.csc_matrix((A.shape[0],1))
    Aext=sparse.hstack([A,new_col],format="csc")
    equality=np.isfinite(lower)&np.isfinite(upper)&(lower==upper)
    upper_constrained=np.isfinite(upper)&(~equality)
    lower_constrained=np.isfinite(lower)&(~equality)
    Aub=sparse.vstack([Aext[upper_constrained],
                       -Aext[lower_constrained]],format="csr")
    bub=np.concatenate([upper[upper_constrained],-lower[lower_constrained]])
    # Every xi >=0 is already a hard constraint in the original explicit
    # slack QP. Add xi <= t, where t is the only SOFT wall allowance.
    rows=np.arange(len(elastic))
    bound_slack=sparse.coo_matrix((
         np.r_[np.ones(len(elastic)),-np.ones(len(elastic))],
         (np.r_[rows,rows],
          np.r_[x_count+u_count+rows,np.full(len(elastic),n)])),
        shape=(len(elastic),n+1)).tocsr()
    Aub=sparse.vstack([Aub,bound_slack],format="csr")
    bub=np.r_[bub,np.zeros(len(elastic))]
    Aeq=Aext[equality]
    beq=upper[equality]
    c=np.zeros(n+1)
    c[-1]=1.
    boundaries=[(None,None)]*n+[(0.,None)]
    result=linprog(c,A_ub=Aub,b_ub=bub,A_eq=Aeq,b_eq=beq,
                   bounds=boundaries,method="highs",
                   options={"primal_feasibility_tolerance":1e-8,
                            "dual_feasibility_tolerance":1e-8})
    if result.status:
        raise RuntimeError("LP failed: "+str(result.message))
    t=max(0.,float(result.x[-1]))
    # LP slack variables themselves need NOT be minimal: at optimum t
    # they can be arbitrarily set to t, even where the original wall is
    # perfectly satisfied. Report physically meaningful actual wall-plane
    # residuals from solved state x, NOT unidentifiable slack allocations.
    achieved={}
    x=result.x
    for stage,face in elastic:
        n0,n1,bound,progress=problem["state_polygons"][stage][face]
        actual_excess=(n0*x[stage*23]+n1*x[stage*23+1]+
                       progress*x[stage*23+20]-bound)
        achieved[stage]=max(achieved.get(stage,0.),float(max(0.,actual_excess)))
    active=[{"stage":k,"actual_wall_violation_m":v} for k,v in
            sorted(achieved.items()) if v>=max(t-1e-5,1e-6)]
    return {
        "fixture":str(fixture),
        "local_affine_hard_wall_feasible":t<=1e-6,
        "minimum_required_uniform_wall_slack_m":t,
        "active_constraint_stages":active[:15],
        "stage1_required_wall_slack_m":achieved.get(1,0.),
        "stage30_required_wall_slack_m":achieved.get(30,0.),
        "elastic_faces":len(elastic),
        "meaning":"LP lower bound for local affine QP, NOT true source/Unity feasibility",
        "lp_iterations":int(result.nit),
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture",type=Path)
    args=parser.parse_args()
    try:
        print(json.dumps(audit(args.fixture),indent=2,sort_keys=True))
        return 0
    except (OSError,ValueError,RuntimeError) as error:
        print(f"elastic wall feasibility audit: {error}",file=sys.stderr)
        return 2


if __name__=="__main__":
    raise SystemExit(main())
