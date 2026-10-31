#!/usr/bin/env python3
"""Find physical-recoverability loss preceding a Unity MPC failure.

This is a read-only diagnostic. It reports an actual recorded interval, not a
predicted safety guarantee. Compare accepted short-prefix commands against
subsequent physical displacement, ADMM convergence, and control deadlines.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import statistics

FALLBACK={'accepted_prior_plan','accepted_candidate_prefix'}


def summarize(run_dir:Path,lookback_s:float=1.0):
    watcher=run_dir/'watcher.log'
    faults=[]
    for line in watcher.read_text().splitlines():
        if line.startswith('MPC_CYCLE_REJECTED: '):
            faults.append(json.loads(line.partition(': ')[2]))
    with (run_dir/'mpc_cycle_detail.csv').open(newline='') as f:
        samples=list(csv.DictReader(f))
    if not faults:raise ValueError('no rejection snapshot in watcher.log')
    def number(row,key,default=None):
        try:return float(row[key])
        except (ValueError,TypeError,KeyError):return default
    result=[]
    for fault in faults:
        progression=fault['progress_m']
        past=[r for r in samples if number(r,'progress_m') is not None
              and number(r,'progress_m') <= progression+0.01]
        if not past:raise ValueError('missing cycle records before fault')
        last_t=number(past[-1],'bag_time_from_start_s')
        interval=[r for r in past if number(r,'bag_time_from_start_s') is not None
                 and last_t-number(r,'bag_time_from_start_s')<=lookback_s]
        if not interval:raise ValueError('missing lookback period')
        fbs=[r for r in interval if r['status'] in FALLBACK]
        solved=[r for r in interval if r['status'].startswith('accepted')
                and r['status'] not in FALLBACK]
        iters=[number(r,'iterations') for r in interval if number(r,'iterations') is not None]
        residuals=[number(r,'primal_residual') for r in interval if number(r,'primal_residual') is not None]
        elapsed=[number(r,'callback_to_gate_ms') for r in interval if number(r,'callback_to_gate_ms') is not None]
        ey=[number(r,'e_y_m') for r in interval if number(r,'e_y_m') is not None]
        throttles=[number(r,'issued_command_throttle') for r in fbs
                   if number(r,'issued_command_throttle') is not None]
        result.append({
            'fault_progress_m':progression,
            'fault_speed_mps':fault['state'][2],
            'fault_lateral_error_m':fault['state'][0],
            'fault_heading_error_rad':fault['state'][1],
            'last_cycle_runtime_ms':fault['end_to_end_control_ms'],
            'under_20ms_gate':fault['end_to_end_control_ms']<20,
            'nonlinear_failure':fault['nonlinear_failure'],
            'lookback_s':lookback_s,
            'sampled_cycles':len(interval),
            'short_prefix_reuses':len(fbs),
            'new_accepted_full_solutions':len(solved),
            'short_prefix_fraction':len(fbs)/len(interval),
            'first_lateral_error_m':ey[0] if ey else None,
            'last_lateral_error_m':ey[-1] if ey else None,
            'fallback_throttle_first':throttles[0] if throttles else None,
            'fallback_throttle_last':throttles[-1] if throttles else None,
            'admm_iteration_median':statistics.median(iters) if iters else None,
            'admm_iteration_max':max(iters,default=None),
            'admm_primal_residual_median':statistics.median(residuals) if residuals else None,
            'runtime_p95_ms':sorted(elapsed)[round((len(elapsed)-1)*.95)] if elapsed else None,
            'runtime_max_ms':max(elapsed,default=None),
            'budget_misses_in_window':sum(str(r.get('publication_gate_budget_violation','')).lower()=='true' for r in interval)})
    return {'schema':'sdu_apex_mpc_failure_precursors/v1','run':run_dir.name,'faults':result}


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('run_dir',type=Path)
    ap.add_argument('--lookback-s',type=float,default=1.0)
    ap.add_argument('--output-json',type=Path)
    args=ap.parse_args()
    res=summarize(args.run_dir,args.lookback_s)
    print(json.dumps(res,indent=2))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True,exist_ok=True)
        args.output_json.write_text(json.dumps(res,indent=2)+'\n')
