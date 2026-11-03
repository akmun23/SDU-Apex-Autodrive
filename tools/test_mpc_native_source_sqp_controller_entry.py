#!/usr/bin/env python3
"""Controller-entry frozen regression for the native 60-command source SQP.

This calls mpc_rti_solve_cycle, not just the direct reduced-source oracle.
It bypasses obsolete recorded legacy QP checks and instead uses the same
original measured state, source hidden states and source command epoch.
No vehicle moves; it is not a recursive or closed-loop test.
"""
from pathlib import Path
import argparse
import json
import subprocess
from tools.test_mpc_native_source_sqp_corpus import CORPUS, HERE

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binary",type=Path,default=HERE/
        "build/f1tenth_mpc/mpc_live_failure_replay")
    p.add_argument("--cases",type=Path,default=HERE/
        "live_runs/comp_gradient_nearwall_capture_20261102_01/qp_captures")
    p.add_argument("--output",type=Path,default=Path(
        "/tmp/mpc_native_source_sqp_controller_entry_20261103.json"))
    a=p.parse_args()
    records=[]
    for name in CORPUS:
        process=subprocess.run([str(a.binary),str(a.cases/name),
            "--native-source-sqp"],cwd=HERE,capture_output=True,text=True,
            check=True,timeout=45)
        found=[json.loads(line) for line in process.stdout.splitlines()
            if '"offline_normal_sqp_comparison":' in line]
        sample=next((r for r in found if
           r.get("offline_normal_sqp_comparison")==
              "native_60_command_source_sqp_controller"),None)
        assert sample is not None,name
        active=sample["native_source_sqp_active"]==1
        published=sample["native_source_sqp_candidate_count"]==30
        wall=sample["native_source_sqp_final_wall_violation_m2"]
        assert active and ((not published) or wall<=1e-9),(name,sample)
        assert (sample["status"]==1)==published,(name,sample)
        assert sample["native_source_sqp_qp_solves"]>0,name
        record=dict(case=name,accepted_source_valid_horizon=published,
            status=sample["status"],published_controls=
                sample["native_source_sqp_candidate_count"],
            final_wall_violation_m2=wall,
            native_qp_solves=sample["native_source_sqp_qp_solves"],
            controller_solve_ms=sample["full_cycle_us"]/1000)
        records.append(record)
        print(json.dumps(record),flush=True)
    assert len(records)==6
    assert sum(r["accepted_source_valid_horizon"] for r in records)==4
    assert all(
        r["accepted_source_valid_horizon"] or r["published_controls"]==0
        for r in records)
    a.output.write_text(json.dumps(records,indent=2))
    print(json.dumps(dict(count=len(records),
        accepted=sum(r["accepted_source_valid_horizon"] for r in records),
        unsafe_published=0,output=str(a.output))),flush=True)
if __name__=="__main__":
    main()
