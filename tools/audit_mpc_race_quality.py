#!/usr/bin/env python3
"""Score MPC lap time and control quality without conflating accuracy and speed.

Read-only analysis of production Unity bag exports. A low tracking RMS alone
is NOT an improvement: require scored lap times, zero collision and zero
command-deadline misses, plus the control/clearance distributions.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
import statistics


def summarize(run_dir: Path, target_total_s: float = 49.8,
              target_mean_s: float = 4.95) -> dict:
    summary = json.loads((run_dir/'mpc_run_report.json').read_text())['summary']
    lap_times = summary['recorded_lap_times_s']
    scored = lap_times[1:]
    metrics = {'run': run_dir.name, 'scored_laps': len(scored),
               'mean_lap_s': statistics.mean(scored) if scored else None,
               'total_s': sum(scored), 'fastest_lap_s': min(scored) if scored else None,
               'lap_standard_deviation_s': statistics.stdev(scored) if len(scored)>1 else None,
               'missed_publication_deadlines': summary['gate_ms']['at_or_over_budget'],
               'score_cutoff_reason': summary['score_cutoff_reason'],
               'collision': summary['first_collision_from_start_s'] is not None,
               'accepted_recovery_cycles': sum(v for k,v in summary['status_counts'].items()
                                              if k in ('accepted_prior_plan','accepted_candidate_prefix',
                                                       'accepted_viability_recovery')),
               'passed_total_target': len(scored)==10 and sum(scored)<target_total_s,
               'passed_mean_target': len(scored)==10 and statistics.mean(scored)<target_mean_s,
               'clean_ten_lap_completion': len(scored)==10 and summary['score_cutoff_reason'] is None
               and summary['first_collision_from_start_s'] is None
               and summary['gate_ms']['at_or_over_budget']==0}
    with (run_dir/'mpc_cycle_detail.csv').open(newline='') as f:
        all_cycles = list(csv.DictReader(f))
    rows = [r for r in all_cycles if r['status'].startswith('accepted')
            and r.get('u_mps') and r.get('target_u_mps')]
    def percentile(values, q):
        if not values: return None
        v=sorted(values);return v[round((len(v)-1)*q)]
    def finite(key):
        out=[]
        for row in rows:
            try:
                x=float(row[key]);
                if math.isfinite(x):out.append(x)
            except (TypeError,ValueError,KeyError):pass
        return out
    ey=finite('e_y_m');psi=finite('e_psi_rad');u=finite('u_mps');target=finite('target_u_mps')
    thr=finite('issued_command_throttle');steer=finite('issued_command_steering_rad')
    def rms(v):return math.sqrt(statistics.mean(x*x for x in v)) if v else None
    def timing_values(key):
        values=[]
        for row in all_cycles:
            try:
                x=float(row[key])
                if math.isfinite(x) and x >= 0: values.append(x)
            except (ValueError,TypeError,KeyError):
                continue
        return values
    timing={}
    for key in ('callback_to_gate_ms','node_solve_ms','qp_build_ms',
                'jacobian_wall_ms','nominal_build_ms','admm_ms',
                'exact_rollout_ms'):
        times=timing_values(key)
        timing[key]={'cycles':len(times),
            'median':statistics.median(times) if times else None,
            'p95':percentile(times,.95),'p99':percentile(times,.99),
            'max':max(times) if times else None}
    gates=timing_values('callback_to_gate_ms')
    timing['callback_headroom_ms']={
        'minimum':20.-max(gates) if gates else None,
        'p01':20.-percentile(gates,.99) if gates else None,
        'fraction_over_16ms':(sum(x>=16. for x in gates)/len(gates))
            if gates else None}
    sqp=[float(r['sqp_iterations_attempted']) for r in all_cycles
         if r.get('sqp_iterations_attempted') in ('1','2')]
    timing['two_sqp_fraction']=sum(x==2 for x in sqp)/len(sqp) if sqp else None
    metrics['timing']=timing
    metrics['quality']={'cycles':len(rows),'lateral_error_rms_m':rms(ey),
        'lateral_error_abs_p95_m':percentile([abs(x) for x in ey],.95),
        'heading_error_rms_rad':rms(psi),
        'heading_error_abs_p95_rad':percentile([abs(x) for x in psi],.95),
        'speed_undershoot_mean_mps':statistics.mean(max(0,ref-actual)
            for actual,ref in zip(u,target)) if u else None,
        'speed_overshoot_mean_mps':statistics.mean(max(0,actual-ref)
            for actual,ref in zip(u,target)) if u else None,
        'throttle_command_delta_abs_p95':percentile([abs(a-b) for a,b in zip(thr[1:],thr)],.95),
        'steering_command_delta_abs_p95_rad':percentile([abs(a-b) for a,b in zip(steer[1:],steer)],.95),
        'corridor_slack_min_m':min(finite('corridor_slack_m'),default=None)}
    return metrics

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('run_dirs',nargs='+',type=Path)
    ap.add_argument('--output-json',type=Path)
    ap.add_argument('--enforce-repeatability',action='store_true',
        help='exit nonzero unless at least three independent clean, fast runs pass')
    ap.add_argument('--minimum-clean-runs',type=int,default=3)
    a=ap.parse_args()
    if a.minimum_clean_runs < 2:
        ap.error('independent repeatability requires at least two clean runs')
    result={'schema':'sdu_apex_mpc_race_quality/v1',
            'runs':[summarize(p) for p in a.run_dirs]}
    qualified=[r for r in result['runs'] if r['clean_ten_lap_completion']
               and r['passed_total_target'] and r['passed_mean_target']]
    result['repeatability_gate']={
        'minimum_clean_runs':a.minimum_clean_runs,
        'qualified_runs':len(qualified),
        'distinct_qualified_runs':len({r['run'] for r in qualified}),
        'passed':len({r['run'] for r in qualified})>=a.minimum_clean_runs,
        'criteria':'ten scored laps; no collisions/faults/deadlines; '
                   'total<49.8s AND mean<4.95s on each independent run'}
    print(json.dumps(result,indent=2))
    if a.output_json:
        a.output_json.parent.mkdir(parents=True,exist_ok=True)
        a.output_json.write_text(json.dumps(result,indent=2)+'\n')
    if a.enforce_repeatability and not result['repeatability_gate']['passed']:
        raise SystemExit(1)
