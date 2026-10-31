#!/usr/bin/env python3
"""Trace a fatal MPC cycle backward to its solver/fallback-command history.

Requires the run's `mpc_cycle_detail.csv` (from export_mpc_run_report.py),
`watcher.log`, and `mpc_run_report.json`. Does not infer that a short, exact-
validated prefix makes later states recursively viable. No solver or actuator
control is modified.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics


def number(row, name):
    try:
        v=float(row.get(name) or '')
        return v if math.isfinite(v) else None
    except (TypeError,ValueError):
        return None


def audit(directory: Path, window_s: float=1.0):
    rows=list(csv.DictReader((directory/'mpc_cycle_detail.csv').open()))
    if not rows:raise ValueError('no MPC cycles in recording')
    faults=[]
    for line in (directory/'watcher.log').read_text().splitlines():
        if line.startswith('MPC_CYCLE_REJECTED: '):
            faults.append(json.loads(line.partition(': ')[2]))
    if not faults:raise ValueError('no fatal MPC_CYCLE_REJECTED event in this run')
    report=json.loads((directory/'mpc_run_report.json').read_text())['summary']
    out=[]
    for fault in faults:
        closest=None
        for i,row in enumerate(rows):
            progress=number(row,'progress_m')
            if progress is None:continue
            error=abs(progress-float(fault['progress_m']))
            if closest is None or error<closest[0]:closest=(error,i)
        if closest is None:continue
        idx=closest[1]
        event_epoch=number(rows[idx],'bag_time_from_start_s')
        if event_epoch is None:continue
        recent=[r for r in rows[:idx+1]
                if number(r,'bag_time_from_start_s') is not None and
                event_epoch-window_s <= number(r,'bag_time_from_start_s') <= event_epoch]
        full=[r for r in recent if r.get('status') in
              ('accepted_optimal','accepted_feasible','accepted_line_search')]
        fallback=[r for r in recent if r.get('status') in
                  ('accepted_prior_plan','accepted_candidate_prefix')]
        timed=[number(r,'callback_to_gate_ms') for r in rows
               if number(r,'callback_to_gate_ms') is not None]
        lateral=[number(r,'e_y_m') for r in recent if number(r,'e_y_m') is not None]
        source=[{
            'elapsed_s':number(r,'bag_time_from_start_s'),
            'unwrapped_progress_m':number(r,'progress_m'),
            'lateral_error_m':number(r,'e_y_m'),
            'heading_error_rad':number(r,'e_psi_rad'),
            'forward_speed_mps':number(r,'u_mps'),
            'throttle':number(r,'issued_command_throttle'),
            'steering_rad':number(r,'issued_command_steering_rad'),
            'candidate_slack_m':number(r,'quality_candidate_prefix_min_slack_m'),
            'retained_slack_m':number(r,'quality_prior_prefix_min_slack_m'),
            'action_source':r.get('action_source'),
            'status':r.get('status')
        } for r in fallback]
        out.append({
            'fatal_status':fault.get('status'),
            'nonlinear_failure_reason':fault.get('nonlinear_failure_reason'),
            'nonlinear_failure_stage':fault.get('nonlinear_failure_stage'),
            'at_progress_m':float(fault['progress_m']),
            'command_time_ey_m':fault.get('state',[None])[0],
            'prediction_ey_m':(fault.get('nonlinear_failure') or {}).get('state',[None])[0],
            'measured_callback_ms':fault.get('end_to_end_control_ms'),
            'window_s':window_s,
            'full_horizon_solve_acceptances':len(full),
            'published_short_prefix_actions':len(fallback),
            'lateral_error_start_m':lateral[0] if lateral else None,
            'lateral_error_end_m':lateral[-1] if lateral else None,
            'max_callback_to_gate_ms_over_run':max(timed) if timed else None,
            'p99_callback_to_gate_ms_over_run':sorted(timed)[int(.99*(len(timed)-1))] if timed else None,
            'deadline_misses_over_run':report.get('gate_ms',{}).get('at_or_over_budget'),
            'physical_collision_from_start_s':report.get('first_collision_from_start_s'),
            'fallback_sequence':source
        })
    return {'run':str(directory),'fatals':out}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir',type=Path)
    parser.add_argument('--window-s',type=float,default=1.0)
    parser.add_argument('--output-json',type=Path)
    args=parser.parse_args()
    if args.window_s<=0:parser.error('window_s must be positive')
    out=audit(args.run_dir,args.window_s)
    rendered=json.dumps(out,indent=2)
    if args.output_json:
        if args.output_json.exists():raise FileExistsError(args.output_json)
        args.output_json.write_text(rendered+'\n')
    for fault in out['fatals']:
        print('REJECTED:',fault['fatal_status'],'reason',fault['nonlinear_failure_reason'],
              'stage',fault['nonlinear_failure_stage'],'progress_m',fault['at_progress_m'])
        print('time_to_gate_ms',fault['measured_callback_ms'],
              'max_run_ms',fault['max_callback_to_gate_ms_over_run'],
              'deadlines',fault['deadline_misses_over_run'],
              'recorded_collision_s',fault['physical_collision_from_start_s'])
        print('last',fault['window_s'],'s: full accepted',
              fault['full_horizon_solve_acceptances'],'short prefix actions',
              fault['published_short_prefix_actions'],
              'lateral_error',fault['lateral_error_start_m'],'->',fault['lateral_error_end_m'])


if __name__=='__main__':main()
