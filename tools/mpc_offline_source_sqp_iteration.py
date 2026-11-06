#!/usr/bin/env python3
"""Repeat exact-source relinearization on a *frozen* MPC corner case.

This is an offline experiment to determine whether repeated accurate QPs,
source-exact nonlinear constraint checks and objective agreement can produce
a truly zero-slack source-feasible horizon. It never publishes commands or
modifies the production controller. It does NOT accept a merely elastic
or affine-feasible QP as a valid vehicle trajectory.

The local QP uses exactly the C++ source and original frozen racing objective,
and the affine equalities and HARD original wall planes are independently
solved with a 60-input condensed OSQP. Each full step and accepted intermediate
is reintegrated through the full C++ source plant at the SAME command epoch.
"""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy import sparse

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.mpc_qp_oracle import decode_problem,build_osqp_matrices
from tools.audit_mpc_condensed_qp import condense,solve_qp

H=30
NX=23
NU=2
RACK_STATE=5
THROTTLE_HISTORY=22
ALPHAS=(1.,.5,.25,.125,.0625,.03125,.015625,.0078125,.00390625)


def load_controls(path:Path)->np.ndarray:
    arr=np.loadtxt(path,dtype=np.float64).reshape(H,2)
    if not np.isfinite(arr).all():raise ValueError("nonfinite command sequence")
    return arr


def save_controls(path:Path,values:np.ndarray):
    np.savetxt(path,values,fmt="%.12f")


def build_qp(offline_binary:Path,case:Path,controls:Path,output:Path,
             candidate:Path|None=None)->dict:
    cmd=[str(offline_binary),str(case),str(controls),str(output),"mode26"]
    if candidate:cmd.append(str(candidate))
    response=subprocess.run(cmd,capture_output=True,text=True,timeout=20)
    if response.returncode:
        raise RuntimeError(f"QP rebuild rc={response.returncode}: {response.stderr[-1000:]}")
    if not response.stdout.strip():
        raise RuntimeError("source QP builder emitted no diagnostics")
    return json.loads(response.stdout.strip().splitlines()[-1])


def controls_from_qp_decisions(p:dict,condensed:dict,values:np.ndarray):
    z=condensed["c"]+condensed["T"]@values
    states=z[:(H+1)*NX].reshape(H+1,NX)
    decisions=z[(H+1)*NX:].reshape(H,NU)
    commands=np.stack([
        decisions[:,0]+states[:H,RACK_STATE],
        decisions[:,1]+states[:H,THROTTLE_HISTORY]
    ],axis=1)
    return commands,states


def qp_objective_on_commands(p:dict,command:np.ndarray):
    state=np.asarray(p["x0"],dtype=np.float64).copy()
    states=[state.copy()]
    decisions=[]
    for k in range(H):
        d=np.array([command[k,0]-state[RACK_STATE],
                    command[k,1]-state[THROTTLE_HISTORY]])
        a,b,affine=p["steps"][k][:3]
        state=a@state+b@d+affine
        states.append(state.copy())
        decisions.extend(d)
    z=np.r_[np.asarray(states).ravel(),np.asarray(decisions)]
    P,q,_,_,_=build_osqp_matrices(p)
    if len(q)!=len(z):
        raise ValueError("QP has elastic slacks, cannot evaluate hard command model")
    Pfull=P+P.T-sparse.diags(P.diagonal(),format="csc")
    return float(.5*z@(Pfull@z)+q@z)


def iterate(case:Path,workspace:Path,binary:Path,niter:int,
            alpha_list:tuple[float,...],eps:float,verbose:bool=True)->list[dict]:
    workspace.mkdir(parents=True,exist_ok=True)
    records=[]
    seed=workspace/"initial.fqp"
    base=build_qp(binary,case,Path("-"),seed)
    initial_controls=Path(str(seed)+".seed.txt")
    original=load_controls(initial_controls)
    nominal=original.copy()
    nominal_file=initial_controls
    for iteration in range(niter):
        qp_file=workspace/f"iteration_{iteration:02d}.fqp"
        if iteration==0:
            qp_file=seed;nominal_info=base
        else:
            nominal_info=build_qp(binary,case,nominal_file,qp_file)
        problem=decode_problem(qp_file.read_bytes())
        condensed=condense(problem,hard_walls=True)
        solution,setup_ms,solve_ms=solve_qp(
            condensed["P"],condensed["q"],condensed["A"],
            condensed["l"],condensed["u"],eps=eps,max_iter=100000)
        row={"iteration":iteration,"qp_status":solution.info.status,
             "qp_iterations":int(solution.info.iter),
             "original_source_violation_m2":nominal_info["exact_source_violation_m2"],
             "original_source_cost":nominal_info["physical_objective"],
             "original_source_valid":nominal_info["strict_source_valid"],
             "original_source_failure_stage":nominal_info["strict_source_failure_stage"],
             "qp_solve_ms":solve_ms}
        if solution.info.status!="solved" or solution.x is None:
            row["failure"]="local_original_hard_QP_not_solved"
            records.append(row)
            if verbose:print(json.dumps(row),flush=True)
            break
        candidate,qp_states=controls_from_qp_decisions(problem,condensed,solution.x)
        candidate_file=workspace/f"iteration_{iteration:02d}_candidate.txt"
        save_controls(candidate_file,candidate)
        source_comparison=build_qp(binary,case,nominal_file,
                                  workspace/f"candidate_probe_{iteration:02d}.fqp",
                                  candidate_file)
        row["full_source_state_error"]=source_comparison.get(
            "maximum_source_state_prediction_error")
        row["full_source_footprint_error_m"]=source_comparison.get(
            "maximum_source_footprint_prediction_error_m")
        row["qp_hard_wall_violation_max"]=max(
            (float(nx*qp_states[k,0]+np_*qp_states[k,1]+
                   progress*qp_states[k,20]-limit)
             for k,planes in enumerate(problem["state_polygons"]) if k>0
             for nx,np_,limit,progress in planes),
            default=0.)
        row["qp_predicted_cost_delta_at_full_step"]=(
            qp_objective_on_commands(problem,candidate)-
            qp_objective_on_commands(problem,nominal))
        row["source_cost_delta_at_full_step"]=(
            source_comparison["proposal_actual_source_objective_delta"])
        row["source_violation_at_full_step_m2"]=(
            source_comparison["proposal_actual_source_violation_m2"])
        row["source_valid_at_full_step"]=(
            source_comparison["proposal_actual_source_violation_m2"]<=1e-12)

        accepted=None
        for alpha in alpha_list:
            trial=nominal+alpha*(candidate-nominal)
            trial_file=workspace/f"iteration_{iteration:02d}_trial_{alpha:g}.txt"
            save_controls(trial_file,trial)
            try:
                probe=build_qp(binary,case,trial_file,
                       workspace/f"trial_qp_{iteration:02d}_{alpha:g}.fqp")
            except RuntimeError:
                continue
            j=probe["physical_objective"]
            v=probe["exact_source_violation_m2"]
            old_j=nominal_info["physical_objective"]
            old_v=nominal_info["exact_source_violation_m2"]
            if v>1e-10:
                improved=old_v>1e-10 and (
                    v<old_v-max(1e-11,old_v*1e-4))
            else:
                # Once the horizon becomes physically feasible, optimize
                # exact cost without allowing a new wall incursion.
                improved=(old_v>1e-10 or (j is not None and
                    old_j is not None and j<old_j-1e-6))
            if improved and (accepted is None or
                 (v<accepted["violation"]-1e-10) or
                 (abs(v-accepted["violation"])<=1e-10 and
                  j is not None and accepted["cost"] is not None and
                  j<accepted["cost"])):
                accepted=dict(alpha=alpha,violation=v,cost=j,
                              controls=trial,control_file=trial_file,
                              source_valid=probe["strict_source_valid"])
        if accepted:
            nominal=accepted.pop("controls")
            nominal_file=accepted.pop("control_file")
            row.update(accepted_alpha=accepted["alpha"],
                       accepted_source_violation_m2=accepted["violation"],
                       accepted_source_cost=accepted["cost"],
                       accepted_source_valid=accepted["source_valid"])
            row["filter_improved"]=True
        else:
            row["filter_improved"]=False
            row["failure"]="no_source_violation_or_cost_descent_along_hard_QP_direction"
        records.append(row)
        if verbose:print(json.dumps(row),flush=True)
        if not accepted:break
        if nominal_info["exact_source_violation_m2"]<=1e-10 and            abs(row["source_cost_delta_at_full_step"] or 0.)<1e-5:
            break
    (workspace/"results.jsonl").write_text(
        "".join(json.dumps(row)+"\n" for row in records))
    return records


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case",type=Path)
    parser.add_argument("--workspace",type=Path,
                        default=Path("/tmp/mpc_offline_source_sqp_experiment"))
    parser.add_argument("--binary",type=Path,
                        default=Path("build/f1tenth_mpc/mpc_offline_exact_source_relinearize"))
    parser.add_argument("--iterations",type=int,default=8)
    parser.add_argument("--osqp-vendor",type=Path,
                        default=Path("/tmp/sdu-mpc-osqp-v104"))
    parser.add_argument("--eps",type=float,default=1e-6)
    args=parser.parse_args()
    sys.path.insert(0,str(args.osqp_vendor))
    if not (1<=args.iterations<=40):parser.error("iterations outside [1,40]")
    iterate(args.case,args.workspace,args.binary,args.iterations,ALPHAS,args.eps)


if __name__=="__main__":
    main()
