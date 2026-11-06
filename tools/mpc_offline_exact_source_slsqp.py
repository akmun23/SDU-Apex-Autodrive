#!/usr/bin/env python3
"""Offline full-source nonlinear SQP oracle, not a runtime controller.

Uses a BFGS approximation of the nonlinear Lagrangian Hessian (SciPy SLSQP),
exact-source cost/gradient, and *unrelaxed* source-plant footprint clearances.
Its independent 60-control formulation answers a research question: can an SQP
with source-constrained QPs converge in fewer iterations than the current
first-order, hard-affine-wall R1/R2 relinearization?

Physical actuator steering rate is explicit. No wall offset or race test.
Source contact branch changes may make finite-difference clearance Jacobians
nonsmooth; results are experimental and must be cross-validated with C++.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.optimize import minimize, Bounds
from tools.mpc_offline_source_constraint_jacobian import (
    source_tangent_constraint_jacobian)
from tools.mpc_offline_source_sqp_iteration import build_qp

H=30
ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case",type=Path)
    parser.add_argument("initial_commands",type=Path)
    parser.add_argument("--iterations",type=int,default=12)
    parser.add_argument("--out",type=Path,
                        default=Path("/tmp/mpc_slsqp_source_probe"))
    parser.add_argument("--source",type=Path,
                        default=Path("/tmp/mpc_offline_source_merit"))
    parser.add_argument("--fd-eps",type=float,default=1e-4)
    parser.add_argument("--constraint-jac",choices=("source_qp","finite_diff"),
                        default="finite_diff")
    parser.add_argument("--qp-source",type=Path,
                        default=Path("/tmp/mpc_offline_exact_source_relinearize"))
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    initial=np.loadtxt(args.initial_commands).reshape(H,2)
    calls=0
    cache={}
    last={}
    history=[]
    finite_failures=0
    jacobian_calls=0
    jacobian_cache={}
    active_file=args.out/"active_controls.txt"

    def evaluate(x):
        nonlocal calls,finite_failures
        x=np.asarray(x).reshape(H,2)
        key=x.tobytes()
        if key in cache:
            return cache[key]
        np.savetxt(active_file,x,fmt="%.12f")
        r=subprocess.run([str(args.source),str(args.case),str(active_file)],
               capture_output=True,text=True,timeout=10)
        calls+=1
        if r.returncode:
            finite_failures+=1
            # Nonlinear integrator can fail for impossible commands. A huge
            # penalty prevents interpreting an invalid source state as safe.
            report=dict(objective=1e9,source_gradient_flat=[0.]*60,
                        original_valid=False,left_clearance_m=[-10.]*31,
                        right_clearance_m=[-10.]*31,
                        rack_steering_rad=[0.]*31,
                        max_steering_rad=0.5235987756,
                        max_steering_command_step_rad=.08,
                        prediction_dt=.025)
        else:
            report=json.loads(r.stdout)
        cache[key]=report
        if len(cache)>6000:cache.clear()
        return report

    def cfun(x):
        q=evaluate(x)
        commands=np.asarray(x).reshape(H,2)
        rack=np.asarray(q['rack_steering_rad'])[:-1]
        step=min(float(q['max_steering_command_step_rad']),
                 .995*3.2*float(q['prediction_dt']))
        clearance=np.c_[q['left_clearance_m'][1:],
                        q['right_clearance_m'][1:]].ravel()
        slew=commands[:,0]-rack
        return np.r_[clearance, step-slew,step+slew]

    def cjac(x):
        nonlocal jacobian_calls
        key=np.asarray(x).tobytes()
        if key in jacobian_cache:return jacobian_cache[key]
        jacobian_calls+=1
        proposal=np.asarray(x).reshape(H,2)
        material=args.out/f"jacobian_nominal_{jacobian_calls:03d}.txt"
        fixture=args.out/f"jacobian_nominal_{jacobian_calls:03d}.fqp"
        np.savetxt(material,proposal,fmt="%.12f")
        build_qp(args.qp_source,args.case,material,fixture)
        A,diagnostic=source_tangent_constraint_jacobian(
            fixture,proposal,evaluate(x))
        if jacobian_calls<=3:
            print(json.dumps(dict(stage="jacobian",call=jacobian_calls,
                                  diagnostic=diagnostic)),flush=True)
        jacobian_cache[key]=A
        if len(jacobian_cache)>500:jacobian_cache.clear()
        return A

    def fun(x):
        return evaluate(x)['objective']/10000.

    def jac(x):
        g=np.asarray(evaluate(x)['source_gradient_flat'],dtype=float)
        return g/10000.

    def monitor(x):
        r=evaluate(x)
        c=cfun(x)
        record=dict(iteration=len(history)+1,calls=calls,
           objective=r['objective'],min_original_source_clearance_m=float(
               min(np.min(np.asarray(r['left_clearance_m'])[1:]),
                   np.min(np.asarray(r['right_clearance_m'])[1:])),
           ),min_total_inequality=float(min(c)),
           original_valid=bool(r['original_valid']))
        history.append(record)
        print(json.dumps(record),flush=True)
        np.savetxt(args.out/f"iterate_{len(history):02d}.txt",
                   np.asarray(x).reshape(H,2),fmt="%.12f")

    starting=evaluate(initial.ravel())
    print(json.dumps(dict(stage="initial",cost=starting['objective'],
         min_clearance=min(starting['left_clearance_m'][1:]+
                           starting['right_clearance_m'][1:]),
         valid=starting['original_valid'])),flush=True)
    max_steer=float(starting['max_steering_rad'])
    low=np.tile([-max_steer,0.001],H)
    high=np.tile([max_steer,1.],H)
    solution=minimize(fun,initial.ravel(),method="SLSQP",jac=jac,
         bounds=Bounds(low,high),
         constraints=[{"type":"ineq","fun":cfun,
                       **({"jac":cjac} if args.constraint_jac=="source_qp"
                          else {})}],
         options={"maxiter":args.iterations,"ftol":1e-7,
                  "eps":args.fd_eps,"disp":False},
         callback=monitor)
    report=evaluate(solution.x)
    result=dict(success=bool(solution.success),message=solution.message,
         iterations=int(solution.nit),source_eval_count=calls,
         jacobian_calls=jacobian_calls,
         constraint_jacobian=args.constraint_jac,
         finite_failures=finite_failures,final_cost=report['objective'],
         source_valid=report['original_valid'],
         min_hard_inequality=float(min(cfun(solution.x))))
    np.savetxt(args.out/"final.txt",solution.x.reshape(H,2),fmt="%.12f")
    (args.out/"summary.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


if __name__=="__main__":
    main()
