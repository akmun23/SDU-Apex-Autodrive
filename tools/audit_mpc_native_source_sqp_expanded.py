#!/usr/bin/env python3
"""Stratified native source SQP failure-corpus audit (not Unity or real-time).

Captures are often adjacent states from the same physical episode; this
reports per-collection counts, NOT an independent statistical success rate.
Cases use *only* recorded initial source states and save per-QP full source
violation, prefix and numerical status. No optimizer parameters are tuned.
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
import subprocess
from collections import Counter, defaultdict

ROOT=Path(__file__).resolve().parents[1]
def pick(paths,n):
    if len(paths)<=n:return paths
    indices={int(round(i*(len(paths)-1)/(n-1))) for i in range(n)}
    return [paths[i] for i in sorted(indices)]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs",type=Path,default=ROOT/"live_runs")
    p.add_argument("--binary",type=Path,default=
                   ROOT/"build/f1tenth_mpc/mpc_native_source_sqp_replay")
    p.add_argument("--per-collection",type=int,default=8)
    p.add_argument("--iterations",type=int,default=8)
    p.add_argument("--output",type=Path,
        default=Path("/tmp/mpc_native_source_sqp_expanded_corpus_20261103.json"))
    args=p.parse_args()
    if args.per_collection<2: p.error("per-collection must be at least 2")
    all_contexts=sorted(args.runs.rglob("source_context.mctx"))
    groups=defaultdict(list)
    for path in all_contexts:
        case=path.parent
        if not (case/"r1.fqp").is_file():
            continue
        groups[case.parent.parent.name].append(case)
    cases=[]
    for group,context in sorted(groups.items()):
        contexts=sorted(context,key=lambda c:int(re.search(
             r"epoch_([0-9]+)",c.name).group(1)))
        cases.extend((group,c) for c in pick(contexts,args.per_collection))
    print(f"SELECTED_CASES={len(cases)} COLLECTIONS={len(groups)}"
          f" AVAILABLE_CONTEXTS={len(all_contexts)}",flush=True)
    rows=[]
    for i,(group,case) in enumerate(cases):
        proc=subprocess.run([str(args.binary),str(case),"-",
                 str(args.iterations)],cwd=ROOT,text=True,
                 capture_output=True,timeout=38)
        try:r=json.loads(proc.stdout)
        except Exception:
            item={"collection":group,"case":case.name,"readable":False,
                  "exitcode":proc.returncode,
                  "diagnostic":(proc.stderr+proc.stdout)[-450:]}
            rows.append(item)
            print(json.dumps(item),flush=True)
            continue
        start=r.get("initial_first_failure_stage",-1)
        end=r.get("final_first_failure_stage",-1)
        initial_v=r["initial_wall_V"]
        final_v=r["final_wall_V"]
        history=r.get("accepted_history",[])
        item=dict(collection=group,case=case.name,readable=True,
                  elapsed_ms=r["total_elapsed_ms"],
                  iteration_count=r["iterations"],
                  accepted_count=r["accepted"],
                  qp_solves=r["qp_solves"],qp_status=r["qp_status"],
                  initial_wall_violation_m2=initial_v,
                  final_wall_violation_m2=final_v,
                  initial_first_failure_stage=start,
                  final_first_failure_stage=end,
                  strictly_source_feasible=bool(r["feasible"] and
                    r["certified_valid"] and final_v<1e-9),
                  source_violation_nonincreasing=final_v<=initial_v+1e-8,
                  violation_reduction_fraction=(
                    max(0.,1-final_v/initial_v) if initial_v>1e-12 else None),
                  prefix_advanced=(
                    end==-1 or (start>0 and end>start)
                    if start>0 else end==-1),
                  worst_full_step_wall_prediction_defect_m=max((
                    h["max_wall_model_defect_m"] for h in history),default=0.))
        rows.append(item)
        if (i+1)%5==0 or not item["source_violation_nonincreasing"]:
            print(json.dumps(item),flush=True)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(rows,indent=2))
    summaries={}
    for group in sorted(groups):
        sub=[r for r in rows if r["collection"]==group]
        readable=[r for r in sub if r["readable"]]
        summaries[group]=dict(examined=len(sub),
          readable=len(readable),
          full_source_feasible=sum(r["strictly_source_feasible"] for r in readable),
          prefix_advanced=sum(r["prefix_advanced"] for r in readable),
          nonincreasing_source_violation=sum(r["source_violation_nonincreasing"] for r in readable),
          initial_source_infeasible=sum(r["initial_wall_violation_m2"]>1e-10 for r in readable))
    complete=sum(r.get("readable",False) for r in rows)
    print(json.dumps({"summary":{"selected":len(rows),"readable":complete,
       "source_feasible":sum(r.get("strictly_source_feasible",False) for r in rows),
       "source_violation_nonincreasing":sum(r.get("source_violation_nonincreasing",False)
                                              for r in rows),
       "prefix_advanced":sum(r.get("prefix_advanced",False) for r in rows),
       "collections":summaries,"output":str(args.output)}}),flush=True)
    (args.output.with_suffix(".summary.json")).write_text(json.dumps(
        summaries,indent=2))

if __name__=="__main__":
    main()
