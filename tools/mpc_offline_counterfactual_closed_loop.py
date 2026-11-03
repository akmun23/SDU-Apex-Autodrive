#!/usr/bin/env python3
"""Counterfactual source-model receding-horizon SQP study.

Unlike independent frozen QP snapshots, accepted first commands causally
change the SOURCE vehicle's next state, steering queue, wheel/suspension
state, trajectory progress and next optimization problem.

This is neither a live Unity run nor a proof of Unity agreement. A source-
valid complete horizon must be distinguished from (i) an exact-source
infeasible but improving nominal and (ii) first-command-only validation.
Only one command is applied per source 25 ms epoch.

No state is taken from later historical captures: every later snapshot is
created from the previously chosen first command and the actual source plant.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np


def command(cmd:list[str],timeout=45)->tuple[int,str,str]:
    p=subprocess.run(cmd,capture_output=True,text=True,timeout=timeout)
    return p.returncode,p.stdout,p.stderr


def strict_source(case:Path,controls:Path,binary:Path):
    rc,out,err=command([str(binary),str(case),str(controls)],20)
    if rc or not out.strip():
        raise RuntimeError("source oracle failed: "+err[-600:])
    return json.loads(out.splitlines()[-1])


def simulate(args:argparse.Namespace):
    args.out.mkdir(parents=True,exist_ok=True)
    case=args.initial_case.resolve()
    if not (case/"source_context.mctx").is_file():
        raise ValueError("missing initial captured case")
    results=[]
    for k in range(args.cycles):
        epoch=args.out/f"epoch_{k:02d}"
        epoch.mkdir(parents=True,exist_ok=True)
        if k==0:
            rc,out,err=command([str(args.qp_binary),str(case),"-",
                    str(epoch/"initial.fqp"),"mode26"],20)
            if rc:raise RuntimeError(err[-600:])
            seed=epoch/"initial.fqp.seed.txt"
        else:
            seed=case/"counterfactual_shifted_seed.txt"
        baseline=strict_source(case,seed,args.source_binary)
        if args.strategy=="optimize":
            cmd=[sys.executable,str(args.optimizer),str(case),
                 str(seed),"--iterations",str(args.iterations),
                 "--constraint-jacobian","parallel_fd",
                 "--workers",str(args.workers),
                 "--source-binary",str(args.source_binary),
                 "--qp-binary",str(args.qp_binary),
                 "--workspace",str(epoch/"optimization")]
            rc,out,err=command(cmd,60)
            if rc:raise RuntimeError("SQP failed: "+err[-600:])
            report=json.loads((epoch/"optimization"/"performance.json").read_text())
            accepted=sorted((epoch/"optimization").glob("accepted_*.txt"))
            selected=accepted[-1] if accepted else seed
        else:
            selected=seed
            report={"source_model_evaluations":0,
                    "first_feasible_iteration":None,
                    "final_source_feasible":baseline["original_valid"]}
        selected_source=strict_source(case,selected,args.source_binary)
        next_case=args.out/f"counterfactual_case_{k+1:02d}"
        rc,out,err=command([str(args.advance_binary),str(case),str(selected),
                    str(next_case)],30)
        step=json.loads(out.splitlines()[-1]) if out.strip().startswith("{") else None
        record=dict(epoch=k,strategy=args.strategy,initial_case=str(case),
            selected_controls=str(selected),
            original_seed_source_valid=bool(baseline["original_valid"]),
            original_seed_min_clearance_m=baseline["physical_minimum_clearance"],
            original_seed_source_objective=baseline["objective"],
            selected_source_valid=bool(selected_source["original_valid"]),
            selected_source_min_clearance_m=selected_source["physical_minimum_clearance"],
            selected_source_objective=selected_source["objective"],
            selected_source_squared_corridor_violation_m2=float(
                np.square(np.minimum(
                    np.c_[selected_source["left_clearance_m"][1:],
                          selected_source["right_clearance_m"][1:]],0.)).sum()),
            first_step_source_valid=bool(step and step["advanced"]),
            optimizer_first_feasible_iteration=report.get("first_feasible_iteration"),
            sqp_iterations=args.iterations if args.strategy=="optimize" else 0,
            applied_steering_rad=(step["applied_steering_rad"]
                                  if step and step["advanced"] else None),
            applied_throttle=(step["applied_throttle"]
                              if step and step["advanced"] else None),
            next_progress_m=(step["progress_after_m"]
                             if step and step["advanced"] else None),
            next_source_speed_mps=(step["speed_after_mps"]
                                   if step and step["advanced"] else None),
            source_step_error=err[-400:] if rc else "")
        results.append(record)
        print(json.dumps(record),flush=True)
        (args.out/"results.json").write_text(json.dumps(results,indent=2))
        if rc:
            print("COUNTERFACTUAL_STOP: first action invalid at epoch",k,flush=True)
            break
        case=next_case
    print(json.dumps(dict(summary={
        "strategy":args.strategy,
        "epochs_advanced":sum(int(r["first_step_source_valid"]) for r in results),
        "full_horizon_source_valid_at_selected":sum(
             int(r["selected_source_valid"]) for r in results),
        "remaining_source_infeasible_at_selected":sum(
             int(not r["selected_source_valid"]) for r in results)})),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("initial_case",type=Path)
    ap.add_argument("--cycles",type=int,default=5)
    ap.add_argument("--iterations",type=int,default=2)
    ap.add_argument("--strategy",choices=("optimize","shift_seed"),
                    default="optimize")
    ap.add_argument("--workers",type=int,default=8)
    ap.add_argument("--out",type=Path,default=Path("/tmp/mpc_counterfactual_source"))
    ap.add_argument("--optimizer",type=Path,
        default=Path(__file__).with_name("mpc_offline_fast_source_sqp.py"))
    ap.add_argument("--qp-binary",type=Path,
        default=Path("/tmp/mpc_offline_source_qp_server"))
    ap.add_argument("--source-binary",type=Path,
        default=Path("/tmp/mpc_source_merit_persistent"))
    ap.add_argument("--advance-binary",type=Path,
        default=Path("/tmp/mpc_offline_advance_source_epoch"))
    args=ap.parse_args()
    if not 1<=args.cycles<=100:ap.error("cycles must be in [1,100]")
    if not 1<=args.iterations<=20:ap.error("SQP iterations must be in [1,20]")
    simulate(args)


if __name__=="__main__":
    main()
