#!/usr/bin/env python3
"""Causal closed loop: native controller -> exact source stage -> next epoch.

Every epoch solves actual mpc_rti_solve_cycle, independently validates the
entire selected source horizon, publishes just its first command to the
source-model stage function, carries all 22 hidden physical states and four
command-history slots, and reoptimizes from the new predicted state.

shift_seed is an intentionally non-controlling comparator that advances
the original shifted controls even if the *future horizon* is infeasible.
It is NOT a safe publishable controller. No Unity/race claim is made.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[1]

def call(cmd,timeout=60):
    cmd=[str(c) for c in cmd]
    r=subprocess.run(cmd,capture_output=True,text=True,cwd=ROOT,timeout=timeout)
    if r.returncode:
        raise RuntimeError(f"command failure rc={r.returncode} {cmd} {r.stderr[-900:]} {r.stdout[-300:]}")
    data=[s for s in r.stdout.splitlines() if s.strip().startswith("{")]
    if not data:raise RuntimeError("no JSON response")
    return json.loads(data[-1])

def compare(a,b):
    if a is None or b is None:raise AssertionError("missing physical state")
    diffs={}
    for key in ("e_y","e_psi","u","v","r","rack","progress"):
        diffs[key]=abs(a[key]-b[key])
    for key in ("queue","internal"):
        if len(a[key])!=len(b[key]):raise AssertionError("state dimension mismatch")
        diffs[key]=max((abs(u-v) for u,v in zip(a[key],b[key])),default=0.)
    assert a["source_initialized"]==b["source_initialized"]
    if max(diffs.values())>1e-8:raise AssertionError(f"source state continuity error {diffs}")
    return diffs

def run(args):
    args.out.mkdir(parents=True,exist_ok=True)
    case=args.initial_case.resolve()
    results=[]
    expected_state=None
    previous_memory=None
    for k in range(args.cycles):
        folder=args.out/f"epoch_{k:03d}"
        folder.mkdir(parents=True,exist_ok=True)
        controlfile=folder/"selected.txt"
        if args.strategy=="native":
            memory_output=folder/"certified_memory.bin"
            cmd=[args.controller,case,controlfile,args.iterations,
                 previous_memory if previous_memory else "-",memory_output]
            if args.terminal3:cmd.append("--terminal3")
            if args.opt20:cmd.append("--opt20")
            if args.suffix_fd:cmd.append("--suffix-fd")
            if args.adaptive:cmd.append("--adaptive")
            if args.forward_source:cmd.append("--full-source-forward")
            if args.deadline_us:cmd.append(f"--deadline-us={args.deadline_us:.0f}")
            solved=call(cmd)
            accepted=solved["accepted"]
            if solved["prior_memory_loaded"] != (previous_memory is not None):
                raise AssertionError("native controller did not use the previous MPC memory")
            if previous_memory is not None:
                assert solved["epoch_shift_stages"]==1,solved
                assert solved["prior_memory_next_state_error"]<1e-9,solved
                assert solved["prior_memory_next_progress_error"]<1e-9,solved
            if k>0 and expected_state is not None:
                compare(expected_state,solved["initial_state"])
            if accepted:
                assert solved["strict_revalidation"]
                assert solved["memory_match"] and solved["epoch_match"]
                assert solved["published_first_control_matches"]
                assert solved["candidate_count"]==30
                assert solved["final_wall_violation_m2"]<1e-10
                assert controlfile.is_file()
                assert memory_output.is_file()
            else:
                assert not controlfile.exists()
                assert not memory_output.exists()
        else:
            if k==0:
                fixture=folder/"seed.fqp"
                call([args.relinearizer,case,"-",fixture,"mode26"])
                controlfile=Path(str(fixture)+".seed.txt")
            else:
                controlfile=case/"counterfactual_shifted_seed.txt"
            accepted=True
            solved=None
        if not accepted:
            record=dict(epoch=k,strategy=args.strategy,applied=False,
                reason="controller emitted no source-validated complete horizon",
                status=solved["status"],final_wall_violation_m2=
                solved["final_wall_violation_m2"])
            results.append(record)
            print(json.dumps(record),flush=True)
            break
        source=call([args.source_merit,case,controlfile])
        if args.strategy=="native" and not source["original_valid"]:
            raise AssertionError("controller accepted physically invalid source horizon")
        nextcase=args.out/f"source_epoch_{k+1:03d}"
        advanced=call([args.advance,case,controlfile,nextcase])
        if not advanced["advanced"]:
            if args.strategy=="native":
                raise AssertionError("controller published invalid first source action")
            results.append(dict(epoch=k,strategy=args.strategy,applied=False,
                reason="comparison policy reached invalid first action"))
            break
        predicted=solved["predicted_next_state"] if solved else None
        actual=advanced["exact_state_after"]
        errors=compare(predicted,actual) if predicted else None
        first=solved["command"] if solved else [
            advanced["applied_steering_rad"],advanced["applied_throttle"]]
        assert abs(first[0]-advanced["applied_steering_rad"])<1e-7
        assert abs(first[1]-advanced["applied_throttle"])<1e-7
        record=dict(epoch=k,strategy=args.strategy,applied=True,
            epoch_ns=advanced["epoch_before_ns"],
            next_epoch_ns=advanced["epoch_after_ns"],
            horizon_source_valid=source["original_valid"],
            source_objective=source["objective"],
            source_wall_violation_m2=sum(min(0.,c)**2 for side in (
                "left_clearance_m","right_clearance_m")
                for c in source[side][1:]),
            command=first,
            actual_next_state=actual,
            predicted_next_state=predicted,
            model_continuity_error_max=max(errors.values()) if errors else None,
            model_continuity_error_by_channel=errors,
            iterations=solved["sqp_iterations"] if solved else 0,
            actual_warm_start_epoch_shift=solved["epoch_shift_stages"] if solved else None,
            persisted_memory_loaded=solved["prior_memory_loaded"] if solved else None,
            persisted_memory_state_error=solved["prior_memory_next_state_error"] if solved else None,
            solve_ms=solved["solver_time_ms"] if solved else None)
        results.append(record)
        (args.out/"results.json").write_text(json.dumps(results,indent=2))
        print(json.dumps({x:y for x,y in record.items()
                         if x not in ("actual_next_state","predicted_next_state",
                                      "model_continuity_error_by_channel")}),flush=True)
        expected_state=actual
        if args.strategy=="native":
            previous_memory=memory_output
        case=nextcase
    summary=dict(strategy=args.strategy,requested=args.cycles,
        applied=sum(bool(r.get("applied")) for r in results),
        source_valid_horizons=sum(bool(r.get("horizon_source_valid")) for r in results),
        max_model_continuity_error=max((
            r.get("model_continuity_error_max") or 0 for r in results),default=0),
        aborted=not results or not results[-1].get("applied",False))
    (args.out/"results.json").write_text(json.dumps(results,indent=2))
    (args.out/"summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps({"summary":summary}),flush=True)
    return summary

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("initial_case",type=Path)
    p.add_argument("--cycles",type=int,default=5)
    p.add_argument("--iterations",type=int,default=2)
    p.add_argument("--deadline-us",type=float,default=0.,
        help="absolute per-epoch solver deadline in microseconds; 0 disables")
    p.add_argument("--strategy",choices=("native","shift_seed"),default="native")
    p.add_argument("--terminal3",action="store_true",
        help="opt-in exact 3-stage held-command terminal continuation")
    p.add_argument("--opt20",action="store_true",
        help="optimize first 20 command stages; retain exact 30-stage horizon")
    p.add_argument("--suffix-fd",action="store_true",
        help="reuse exact baseline physical prefixes in source FD derivatives")
    p.add_argument("--adaptive",action="store_true",
        help="source-exact stop after first SQP step with >=2pct cost decrease and >=2mm clearance")
    p.add_argument("--forward-source",action="store_true",
        help="experimental full 26-physical-state source forward sensitivity")
    p.add_argument("--out",type=Path,default=Path("/tmp/mpc_native_source_closed_loop"))
    p.add_argument("--controller",type=Path,default=
                   ROOT/"build/f1tenth_mpc/mpc_native_source_sqp_controller_epoch")
    p.add_argument("--advance",type=Path,default=
                   ROOT/"build/f1tenth_mpc/mpc_offline_advance_source_epoch")
    p.add_argument("--relinearizer",type=Path,default=Path("/tmp/mpc_offline_source_qp_server"))
    p.add_argument("--source-merit",type=Path,default=Path("/tmp/mpc_source_merit_persistent"))
    a=p.parse_args()
    if not 1<=a.cycles<=100 or not 1<=a.iterations<=8:p.error("invalid cycles/iterations")
    if a.deadline_us and not 1000<=a.deadline_us<=1e6:
        p.error("deadline must be between 1000 and 1000000 microseconds")
    run(a)

if __name__=="__main__":
    main()
