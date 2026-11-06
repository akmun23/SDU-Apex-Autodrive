#!/usr/bin/env python3
"""Exact finite-direction second-order audit of the REAL native SQP QP.

This program never publishes controls and does not approximate the source
vehicle: every sampled cost/corridor is obtained from the same C++ source
scorer as native source-SQP. It holds the objective schedule and physical
initial state fixed for alpha in a real 60-command QP direction.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import subprocess

import numpy as np

ROOT = pathlib.Path("/home/akselmo/Documents/GitHub/SDU-Apex-Autodrive")
EXEC = ROOT / "build/f1tenth_mpc/mpc_native_source_sqp_replay"
SCALES = np.array([0.15, 0.25])

def source_oracle(case: pathlib.Path, physical_controls: np.ndarray, name: str):
    filename = case / f"source_direction_probe_{name}.controls"
    assert physical_controls.shape == (30, 2)
    np.savetxt(filename, physical_controls, fmt="%.12f")
    env=dict(os.environ,OMP_NUM_THREADS="6",OMP_PROC_BIND="spread",
             OMP_PLACES="threads")
    cmd=["taskset","-c","0,2,4,6,8,10",str(EXEC),str(case),str(filename),
         "1","--oracle-only"]
    p = subprocess.run(cmd,cwd=ROOT,env=env,capture_output=True,
                       text=True,timeout=60)
    if p.returncode:
        raise RuntimeError(f"{name}: source oracle status {p.returncode}: "
                           f"{p.stderr[-900:]}")
    output=json.loads(p.stdout)
    assert output["source_exact_nominal"] is True
    return dict(alpha=name,cost=float(output["cost"]),
                wall_violation_m2=float(output["wall_violation_m2"]),
                walls=np.asarray(output["walls"],dtype=float))

def analyze_one(case: pathlib.Path):
    failure=json.loads((case/"failure.json").read_text())
    u=np.asarray(failure["nominal_controls"],dtype=float)
    direction=np.loadtxt(case/"first_qp_qp_direction.txt")[:,1]
    assert u.shape==(30,2) and direction.shape==(60,)
    du=direction.reshape(30,2)*SCALES
    grad=np.loadtxt(case/"first_qp_gradient.txt")[:,1]
    jac=np.loadtxt(case/"first_qp_jacobian.txt",usecols=range(3,63))[:60]
    H=np.loadtxt(case/"first_qp_qp_hessian.txt")
    qpmode=(case/"first_qp_qp_direction.txt").read_text().splitlines()[0]
    phase1="phase1=1" in qpmode
    assert jac.shape==(60,60) and H.shape==(60,60)
    h=(.01,.02,.05,.1)
    alphas=sorted(set([0.,*[-x for x in h],*h,.125,.25,.5,.75,1.]))
    oracle={}
    for alpha in alphas:
        trial=u+alpha*du
        in_physical_bounds=bool(np.all(
            (trial[:,0]>=-0.5235987756-1.e-6)&
            (trial[:,0]<=.5235987756+1.e-6)&
            (trial[:,1]>=.001-1.e-6)&(trial[:,1]<=1.+1.e-6)))
        response=source_oracle(case,trial,f"{alpha:+.5f}".replace(".","_"))
        response["input_within_physical_bounds"]=in_physical_bounds
        oracle[alpha]=response
    base=oracle[0.]
    seed_replay=json.loads((case/"baseline.json").read_text())
    # The public scoring oracle and original internal SQP evaluator must
    # measure the SAME frozen-source racing objective and footprint.
    cost_error=abs(seed_replay["initial_cost"]-base["cost"])
    corridor_error=abs(seed_replay["initial_wall_V"]-base["wall_violation_m2"])
    assert cost_error<.002,("score mismatch",cost_error)
    assert corridor_error<1.e-7,("corridor mismatch",corridor_error)
    gdot=float(grad@direction)
    Jdir=jac@direction
    qp_curvature=float(direction@H@direction)
    candidates=[]
    for hh in h:
        vp=oracle[hh];vm=oracle[-hh]
        source_curvature=(vp["cost"]-2*base["cost"]+vm["cost"])/(hh*hh)
        wall_curvature=(vp["walls"]-2*base["walls"]+vm["walls"])/(hh*hh)
        for alpha in (.125,.25,.5,.75,1.):
            exact=oracle[alpha]
            affine=base["walls"]+alpha*Jdir
            quadratic=affine+.5*alpha*alpha*wall_curvature
            def wall_v(w):return float(np.sum(np.minimum(w,0.)**2))
            def defect(w):return float(np.max(abs(w-exact["walls"])))
            # The exact (finite-difference) local second derivative can
            # still be a poor model at a distant nonsmooth branch switch.
            candidates.append(dict(
                h=hh,alpha=alpha,
                source_true_cost_change=exact["cost"]-base["cost"],
                source_local_gradient=gdot,
                source_directional_second_derivative=source_curvature,
                qp_cost_hessian_direction=qp_curvature if not phase1 else None,
                actual_wall_violation_m2=exact["wall_violation_m2"],
                affine_wall_violation_m2=wall_v(affine),
                exact_local_quadratic_wall_violation_m2=wall_v(quadratic),
                max_affine_wall_defect_m=defect(affine),
                max_second_order_wall_defect_m=defect(quadratic),
                source_local_linear_cost_delta=alpha*gdot,
                source_local_quadratic_cost_delta=alpha*gdot+
                    .5*alpha*alpha*source_curvature,
                qp_predicted_cost_delta=(
                    alpha*gdot+.5*alpha*alpha*qp_curvature
                    if not phase1 else None
                ),
            ))
    summary=dict(case=case.name,full_source_state_reconstructed=True,
        source_objective_parity_error=cost_error,
        source_corridor_parity_error=corridor_error,
        phase1=phase1,qp_direction_norm_normalized=float(np.linalg.norm(direction)),
        local_gradient_dot_qp=gdot,qp_direction_cost_curvature=qp_curvature,
        base_cost=base["cost"],base_wall_V=base["wall_violation_m2"],
        stagewise_samples=[{
            "alpha":alpha, "source_cost":point["cost"],
            "source_wall_violation_m2":point["wall_violation_m2"],
            "within_vehicle_bounds":point["input_within_physical_bounds"],
        } for alpha,point in oracle.items()],
        comparisons=candidates)
    output=case/"exact_source_real_direction_second_order.json"
    output.write_text(json.dumps(summary,indent=2))
    print("CASE",case.name,"phase1",phase1,"norm",round(summary["qp_direction_norm_normalized"],4),
          "objective_delta(alpha=1)",round(oracle[1.]["cost"]-base["cost"],3))
    for hh in h:
        e=next(x for x in candidates if x["h"]==hh and x["alpha"]==1.)
        print("h",hh,"wall V true/affine/quadratic",
              *[round(e[k],6) for k in ("actual_wall_violation_m2",
                     "affine_wall_violation_m2",
                     "exact_local_quadratic_wall_violation_m2")],
              "max wall defect affine/quadratic",
              round(e["max_affine_wall_defect_m"],4),
              round(e["max_second_order_wall_defect_m"],4),
              "cost_delta true/source quadratic",
              round(e["source_true_cost_change"],3),
              round(e["source_local_quadratic_cost_delta"],3))
    return summary

if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cases",type=pathlib.Path,nargs="+")
    args=ap.parse_args()
    for folder in args.cases:
        analyze_one(folder)
