#!/usr/bin/env python3
"""Frozen-corpus qualification for the NEW native full-source SQP backend.

No ROS, Unity or physical driving. Uses the actual saved source-context
r1_nominal controls, so no ephemeral /tmp initial seed is required.
All accepted full-horizon source plans have been independently validated by
the same production source-rollout guard inside the native optimizer.
This test is deliberately separate from the 22 portable synthetic CTests:
recorded live_runs fixture folders are not necessarily present on CI.
"""
from __future__ import annotations
import argparse
import json
import subprocess
from pathlib import Path

HERE=Path(__file__).resolve().parents[1]
CORPUS=(
    "epoch_1793641756087085412_case_6",
    "epoch_1793692853295529252_case_6",
    "epoch_1793692917333601071_case_6",
    "epoch_1793693111767116537_case_6",
    "epoch_1793693111641647593_case_2",
    "epoch_1793693111867025977_case_10",
)
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cases",type=Path,default=HERE/
        "live_runs/comp_gradient_nearwall_capture_20261102_01/qp_captures")
    p.add_argument("--binary",type=Path,default=HERE/
        "build/f1tenth_mpc/mpc_native_source_sqp_replay")
    p.add_argument("--iterations",type=int,default=8)
    p.add_argument("--output",type=Path,
                   default=Path("/tmp/mpc_native_source_sqp_qualified_corpus.json"))
    a=p.parse_args()
    if not a.binary.is_file():
        p.error(f"not built: {a.binary}")
    missing=[name for name in CORPUS
             if not (a.cases/name/"source_context.mctx").exists()]
    if missing:
        p.error(f"missing recorded source contexts: {missing}")
    rows=[]
    for name in CORPUS:
        r=subprocess.run([str(a.binary),str(a.cases/name),"-",
            str(a.iterations)],cwd=HERE,check=True,
            capture_output=True,text=True,timeout=40)
        state=json.loads(r.stdout)
        assert state["complete"] and state["native_source_sqp"]
        initial=state["initial_first_failure_stage"]
        final=state["final_first_failure_stage"]
        improved=final==-1 or (initial>=1 and final>initial)
        # A decreasing squared violation alone is not a substitute for
        # checking where the first inevitable source invalid state moves.
        valid=(state["feasible"] and state["certified_valid"] and
               state["final_wall_V"]<1e-10)
        assert state["final_wall_V"] <= state["initial_wall_V"]+1e-9
        row=dict(case=name,source_feasible=valid,improved_prefix=improved,
            original_first_failure_stage=initial,
            selected_first_failure_stage=final,
            initial_wall_violation_m2=state["initial_wall_V"],
            final_wall_violation_m2=state["final_wall_V"],
            final_cost=state["final_cost"],
            accepted_sqp_iterations=state["accepted"],
            total_elapsed_ms=state["total_elapsed_ms"],
            full_step_defects=[{
                "max_wall_m":v["max_wall_model_defect_m"],
                "max_actuator_rate":v["max_rate_model_defect"]}
                for v in state["accepted_history"]])
        rows.append(row)
        print(json.dumps(row),flush=True)
    count=sum(x["source_feasible"] for x in rows)
    improved=sum(x["improved_prefix"] for x in rows)
    # This is a source-only frozen regression, not a certification of
    # a full race or actual Unity/source agreement.
    assert count>=4,(count,rows)
    assert improved==6,(improved,rows)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(rows,indent=2))
    print(json.dumps(dict(cases=len(rows),full_horizon_feasible=count,
        prefixes_improved=improved,result=str(a.output))),flush=True)

if __name__=="__main__":
    main()
