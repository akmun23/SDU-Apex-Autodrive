#!/usr/bin/env python3
"""Condense the *exact* frozen MPC LTV equality constraints to 60 inputs.

No dynamics approximation is made: given x0, the 30-stage linear dynamics
implies x_stack = c + G*u_stack exactly (in float64). This eliminates all
23*31 state decision variables and their 23*30 equality constraints.

The reduced QP retains the captured objective/cross Hessians, physical
input boxes, all original hard wall polygons and other stage/terminal state
constraints. Compare OSQP on both the original 773-variable QP and 60-input
condensed QP. This is an OFFLINE feasibility/numerical-performance audit,
not a substitute for the controller's exact nonlinear rollout.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.mpc_qp_oracle import build_osqp_matrices,decode_problem,_load_osqp

NX, NU, H = 23, 2, 30


def condense(problem, hard_walls=True):
    p=copy.deepcopy(problem)
    if hard_walls:
        p['polygon_elastic_weights']=[0.]*(H+1)
    P, q, A, l, u=build_osqp_matrices(p)
    if P.shape[0]!=(H+1)*NX+H*NU:
        raise ValueError("condensing only supports no explicit slack: use --hard-walls")
    nvar=P.shape[0]; ninputs=H*NU
    T=np.zeros((nvar,ninputs),dtype=np.float64)
    c=np.zeros(nvar,dtype=np.float64)
    c[:NX]=p['x0']
    for k,step in enumerate(p['steps']):
        a,b,d=step[:3]
        old=slice(k*NX,(k+1)*NX)
        new=slice((k+1)*NX,(k+2)*NX)
        c[new]=a @ c[old]+d
        T[new,:]=a @ T[old,:]
        T[new,k*NU:(k+1)*NU]+=b
    T[(H+1)*NX:,:]=np.eye(ninputs)
    # The independent oracle stores P in OSQP UPPER-TRIANGLE storage.
    # For an affine objective substitution we must form the full symmetric
    # quadratic form; using triangular P@T is mathematically WRONG.
    Pfull=P+P.T-sparse.diags(P.diagonal(),format='csc')
    reduced_h=T.T @ (Pfull@T)
    reduced_h=.5*(reduced_h+reduced_h.T)
    reduced_q=T.T @ (Pfull@c+q)
    AT=A@T
    offset=A@c
    shifted_l=l-offset
    shifted_u=u-offset
    # Remove dynamics and x0 equality rows which are mathematically
    # eliminated by x=c+G*u. Tiny float64 cancellation may remain.
    norm=np.max(np.abs(AT),axis=1)
    redundant=norm<=5.e-8
    equality=np.isfinite(shifted_l)&np.isfinite(shifted_u)&(np.abs(shifted_l-shifted_u)<1.e-7)
    # An equality row with norm 0 is valid only if its shifted 0 interval
    # includes 0; this is a necessary parity check of the elimination.
    failing=np.flatnonzero(redundant & ((np.isfinite(shifted_l)&(shifted_l>5.e-5)) |
                          (np.isfinite(shifted_u)&(shifted_u< -5.e-5))))
    if len(failing):raise AssertionError(f"improper dynamics elimination {failing[:5].tolist()}")
    keep=~redundant
    return {
        'P':sparse.csc_matrix(reduced_h),
        'q':reduced_q,
        'A':sparse.csc_matrix(AT[keep,:]),
        'l':shifted_l[keep],
        'u':shifted_u[keep],
        'c':c,
        'T':T,
        'original_P':P, 'original_q':q, 'original_A':A,
        'original_l':l, 'original_u':u,
        'rows_eliminated':int(np.sum(redundant)),
        'equations_eliminated':int(np.sum(redundant & equality)),
    }


def solve_qp(P,q,A,l,u, *, eps, max_iter=100000, polish=True):
    osqp=_load_osqp()
    solver=osqp.OSQP()
    t0=time.perf_counter()
    solver.setup(P=P,q=q,A=A,l=l,u=u,eps_abs=eps,eps_rel=eps,
                 max_iter=max_iter,polishing=polish,verbose=False,
                 adaptive_rho=True,scaled_termination=False)
    t1=time.perf_counter()
    x=solver.solve(raise_error=False)
    t2=time.perf_counter()
    return x,1000*(t1-t0),1000*(t2-t1)


def compare(fixture:Path,eps:float,run_full:bool=True):
    p=decode_problem(fixture.read_bytes())
    hard=copy.deepcopy(p)
    hard['polygon_elastic_weights']=[0.]*(H+1)
    c=condense(hard)
    reduced,setup_ms,solve_ms=solve_qp(c['P'],c['q'],c['A'],c['l'],c['u'],eps=eps)
    data={
      'fixture':str(fixture),
      'reduced_variables':int(c['P'].shape[0]),
      'reduced_constraints':int(c['A'].shape[0]),
      'eliminated_rows':c['rows_eliminated'],
      'reduced_equality_rows_eliminated':c['equations_eliminated'],
      'condensed_status':reduced.info.status,
      'condensed_iterations':int(reduced.info.iter),
      'condensed_objective_offset':None,
      'condensed_setup_ms':round(setup_ms,3),
      'condensed_solve_ms':round(solve_ms,3),
      'condensed_max_absolute_linear_constraint_violation':None,
      'condensed_raw_dynamics_equality_residual':None,
    }
    if reduced.info.status=='solved':
        u_controls=reduced.x
        x=c['c']+c['T']@u_controls
        residual=c['original_A']@x
        lo=c['original_l'];hi=c['original_u']
        lower=np.isfinite(lo);upper=np.isfinite(hi)
        max_v=max(float(np.max(np.maximum(0.,lo[lower]-residual[lower]))),
                  float(np.max(np.maximum(0.,residual[upper]-hi[upper]))))
        original=c['original_P']
        Pfull=original+original.T-sparse.diags(original.diagonal(),format='csc')
        objective=.5*x@(Pfull@x)+c['original_q']@x
        data['condensed_objective']=float(objective)
        data['condensed_max_absolute_linear_constraint_violation']=max_v
        # Compare the original H=23-state/2-control scalar physical cost.
        data['first_steering_decision']=float(u_controls[0])
        data['first_throttle_decision']=float(u_controls[1])
    if run_full:
        P,q,A,l,u=build_osqp_matrices(hard)
        full,fs,ft=solve_qp(P,q,A,l,u,eps=eps)
        data.update(
           full_status=full.info.status,
           full_iterations=int(full.info.iter),
           full_setup_ms=round(fs,3),
           full_solve_ms=round(ft,3)
        )
        if full.info.status=='solved' and reduced.info.status=='solved':
            full_inputs=full.x[(H+1)*NX:]
            data['max_control_difference']=float(np.max(np.abs(full_inputs-reduced.x)))
            data['full_objective']=float(full.info.obj_val)
            data['objective_difference']=float(abs(data['condensed_objective']-full.info.obj_val))
    return data


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('fixtures',type=Path,nargs='+')
    a.add_argument('--eps',type=float,default=1e-6)
    a.add_argument('--skip-full',action='store_true')
    a.add_argument('--osqp-vendor',type=Path,default=None)
    args=a.parse_args()
    if args.osqp_vendor:
        # Import NumPy/SciPy first from the active analysis environment.
        # Only OSQP is resolved from the explicitly supplied vendor.
        sys.path.insert(0,str(args.osqp_vendor))
    for f in args.fixtures:
        try:
            print(json.dumps(compare(f,args.eps,not args.skip_full),sort_keys=True),flush=True)
        except Exception as e:
            print(json.dumps({'fixture':str(f),'error':str(e)}),flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
