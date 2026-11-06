#!/usr/bin/env python3
"""Fail closed: accept a practice MPC only after identical full-length Unity repeats.

Examples:
  python3 tools/check_mpc_practice_repeatability.py live_runs/practice_*_01 live_runs/practice_*_02

The default gate requires three genuinely different run directories with the
same recorded base config, optional overlay, source trajectory, and map.
This is a *promotion* gate, not a single-run statistical significance test.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def digest(path):
    p = Path(path)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None


def evaluate(run_dir, expected_laps=10, max_mean_s=4.95, max_total_s=49.8):
    run_dir=Path(run_dir)
    summary=json.loads((run_dir/'mpc_run_report.json').read_text())['summary']
    times=summary.get('recorded_lap_times_s') or []
    failures=[]
    if len(times)!=expected_laps+1:
        failures.append(f'expected {expected_laps} scored laps + 1 warmup, recorded {len(times)}')
    scored=times[1:]
    if any(not isinstance(x,(int,float)) or not math.isfinite(x) or x<=0 for x in scored):
        failures.append('nonfinite/invalid lap time')
        scored=[]
    total=sum(scored) if scored else math.inf
    mean=statistics.mean(scored) if scored else math.inf
    if total>=max_total_s:failures.append(f'10-lap total {total:.5f} s >= {max_total_s:.5f} s')
    if mean>=max_mean_s:failures.append(f'average {mean:.5f} s >= {max_mean_s:.5f} s')
    if summary.get('first_collision_from_start_s') is not None:failures.append('simulator collision')
    if summary.get('score_cutoff_reason'):failures.append('fatal controller or experiment cutoff')
    gates=summary.get('gate_ms') or {}
    if gates.get('at_or_over_budget',-1)!=0:failures.append(f"publication gate violations: {gates.get('at_or_over_budget')}")
    statuses=summary.get('status_counts') or {}
    for status,count in statuses.items():
        if status.startswith(('rejected_','fault_')) and count:
            failures.append(f'controller rejection: {status}={count}')
    manifest=json.loads((run_dir/'run_manifest.json').read_text())
    fingerprint=(digest(run_dir/'mpc_parameters.yaml'),
                 digest(run_dir/'mpc_overlay.yaml'),
                 digest(run_dir/'amcl_parameters.yaml'),
                 digest(run_dir/'amcl_overlay.yaml'),
                 manifest.get('trajectory_sha256'),
                 manifest.get('map_sha256'),
                 manifest.get('source',{}).get('commit_sha'),
                 manifest.get('source',{}).get('tracked_diff_sha256'))
    if fingerprint[0] is None:failures.append('no sealed MPC base parameters')
    if not manifest.get('trajectory_sha256'):failures.append('missing source trajectory digest')
    return dict(run=str(run_dir),mean_s=mean,total_s=total,fastest_s=min(scored) if scored else None,
                scored_laps=len(scored),deadline_misses=gates.get('at_or_over_budget'),
                failures=failures,fingerprint=fingerprint)


def check_runs(dirs, minimum_runs=3, **kwargs):
    if len(set(str(Path(d).resolve()) for d in dirs))!=len(dirs):
        raise ValueError('duplicate runs cannot count as independent trials')
    evaluations=[evaluate(d,**kwargs) for d in dirs]
    errors=[]
    if len(evaluations)<minimum_runs:
        errors.append(f'only {len(evaluations)} independent runs, require {minimum_runs}')
    if evaluations and any(x['fingerprint']!=evaluations[0]['fingerprint'] for x in evaluations[1:]):
        errors.append('runs do not share identical configuration/trajectory/source fingerprints')
    for x in evaluations:errors.extend(f"{x['run']}: {f}" for f in x['failures'])
    return evaluations,errors


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dirs',nargs='+',type=Path)
    parser.add_argument('--min-runs',type=int,default=3)
    parser.add_argument('--max-mean-s',type=float,default=4.95)
    parser.add_argument('--max-total-s',type=float,default=49.8)
    args=parser.parse_args()
    if args.min_runs<2:parser.error('minimum independent runs must be >=2')
    reports,errors=check_runs(args.run_dirs,args.min_runs,
                              max_mean_s=args.max_mean_s,max_total_s=args.max_total_s)
    for r in reports:
        print(f"{r['run']}: {r['scored_laps']} scored laps, mean={r['mean_s']:.5f}s "
              f"total={r['total_s']:.5f}s best={r['fastest_s']}s "
              f"deadline_misses={r['deadline_misses']}")
    for error in errors:print('FAIL:',error)
    print('PROMOTION GATE:', 'PASS' if not errors else 'FAIL')
    raise SystemExit(1 if errors else 0)


if __name__=='__main__': main()
