#!/usr/bin/env python3
"""Audit early-stage feasibility and recovery of frozen source-SQP trajectories.

This does not interpret a frozen bad starting state as a counterfactual
online MPC rollout. It compares *same-epoch candidate command plans*
against the original source-model plant. A source-feasible prefix is
necessary but not sufficient for receding-horizon recursive feasibility.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.mpc_offline_fast_source_sqp import PersistentSource

def metrics(source, controls):
    r=source.evaluate(np.asarray(controls).ravel(),gradient=False)
    walls=np.asarray(r['wall'],dtype=float).reshape(30,2)
    worst=np.min(walls,axis=1)
    bad=np.flatnonzero(worst<-1e-7)
    return dict(
       first_negative_stage=int(bad[0]+1) if len(bad) else None,
       feasible_prefix_stages=int(bad[0]) if len(bad) else 30,
       squared_corridor_violation_m2=float(r['wall_v']),
       minimum_corridor_clearance_m=float(np.min(walls)),
       source_valid=bool(r['source_valid']),
       exact_cost=float(r['J']),
       negative_wall_stages=int(len(bad)),
       stage_minimum_clearance_m=worst.tolist(),
    )

def audit(case, iterations_dir, source_binary):
    seeds=sorted(iterations_dir.glob('accepted_*.txt'))
    original=Path('/tmp/mpc_sqp_offline_corpus_v2')/case.name/'initial.fqp.seed.txt'
    if not original.exists():
        raise FileNotFoundError(f"no frozen original nominal seed: {original}")
    source=PersistentSource(case,source_binary,iterations_dir,persistent=True)
    try:
        initial=metrics(source,np.loadtxt(original))
        rows=[dict(iteration=0,label="frozen initial nominal",**initial)]
        for i,seed in enumerate(seeds,1):
            current=metrics(source,np.loadtxt(seed))
            rows.append(dict(iteration=i,label="accepted exact-source SQP iterate",**current))
    finally:
        source.close()
    # Report prefix progress relative to the original frozen failure stage.
    # This is NOT the hypothetical online result of optimizing earlier epochs.
    initial_first=initial['first_negative_stage']
    for row in rows:
        row['historical_failure_stage']=initial_first
        row['safe_before_historical_failure']=(
            initial_first is None or row['first_negative_stage'] is None or
            row['first_negative_stage']>=initial_first)
        row['advanced_first_failure_by_stages']=(
            (31 if row['first_negative_stage'] is None else row['first_negative_stage'])-
            (31 if initial_first is None else initial_first))
    return rows

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases',type=Path,default=ROOT/'live_runs/comp_gradient_nearwall_capture_20261102_01/qp_captures')
    parser.add_argument('--iterations',type=Path,default=Path('/tmp/mpc_fast_source_sqp_20261103/full_corpus'))
    parser.add_argument('--source',type=Path,default=Path('/tmp/mpc_source_merit_persistent'))
    parser.add_argument('--output',type=Path,default=Path('/tmp/mpc_fast_source_sqp_20261103/prefix_analysis.json'))
    args=parser.parse_args()
    data=[]
    for folder in sorted(args.iterations.iterdir()):
        if not folder.is_dir() or not (args.cases/folder.name).is_dir():
            continue
        history=audit(args.cases/folder.name,folder,args.source)
        a=history[0]
        b=history[-1]
        row=dict(case=folder.name,records=history,
           original_first_negative_stage=a['first_negative_stage'],
           latest_first_negative_stage=b['first_negative_stage'],
           original_feasible_prefix_stages=a['feasible_prefix_stages'],
           latest_feasible_prefix_stages=b['feasible_prefix_stages'],
           original_squared_violation_m2=a['squared_corridor_violation_m2'],
           latest_squared_violation_m2=b['squared_corridor_violation_m2'],
           latest_strict_source_valid=b['source_valid'])
        data.append(row)
        print(json.dumps({k:v for k,v in row.items() if k!='records'}),flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(data,indent=2))
    print(json.dumps(dict(
       count=len(data),
       prefixes_improved=sum(x['latest_feasible_prefix_stages']>x['original_feasible_prefix_stages'] for x in data),
       prefixes_unchanged=sum(x['latest_feasible_prefix_stages']==x['original_feasible_prefix_stages'] for x in data),
       prefixes_worsened=sum(x['latest_feasible_prefix_stages']<x['original_feasible_prefix_stages'] for x in data),
       source_feasible=sum(x['latest_strict_source_valid'] for x in data),
       output=str(args.output))),flush=True)

if __name__=="__main__":
    main()
