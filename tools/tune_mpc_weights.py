#!/usr/bin/env python3
"""Compare tracking/damping bundles on identical saved MPC states and nominals.

No Unity runs or production profile changes. Every profile sees the same
measured states and recorded nominal controls; ADMM memory starts cold per row.
Replay acceptance is an offline screen, not evidence of successful driving.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.export_mpc_solver_inputs_from_bag import to_solver_row, HEADER
from tools.benchmark_mpc_replay import summarize_rows, sha256_file

ROOT=Path(__file__).resolve().parents[1]
WEIGHTS=('weight_e_y','weight_e_psi','weight_u','weight_u_overspeed',
         'weight_longitudinal_slip','weight_v','weight_r','weight_steering_command',
         'weight_steering_target','weight_throttle_command','weight_steering_slew',
         'weight_throttle_slew')
# Coupled bundles change tracking authority and command damping together.
PROFILES={
    'baseline': {},
    'tracking': dict(weight_e_y=500.,weight_e_psi=120.,weight_steering_target=.05,
                     weight_steering_slew=5.),
    'balanced': dict(weight_e_y=300.,weight_e_psi=90.,weight_steering_target=.5,
                     weight_steering_slew=2.),
    'heading': dict(weight_e_y=250.,weight_e_psi=150.,weight_steering_target=5.,
                    weight_steering_slew=5.),
    'native_lower': dict(weight_e_psi=60.,weight_steering_target=5.,
                         weight_steering_slew=2.),
    'strong_rate': dict(weight_e_y=500.,weight_e_psi=150.,weight_steering_target=.1,
                        weight_steering_slew=15.),
    'wide_balanced': dict(weight_e_y=300.,weight_e_psi=90.,weight_steering_target=.5,
                         weight_steering_slew=2.,sqp_steering_trust_region_rad=.12),
    'free_balanced': dict(weight_e_y=300.,weight_e_psi=90.,weight_steering_target=.5,
                         weight_steering_slew=2.,sqp_steering_trust_region_rad=0.),
    'reference_balanced': dict(weight_e_y=300.,weight_e_psi=90.,weight_steering_target=.5,
                         weight_steering_slew=2.,replay_reference_seed=True),
    'wide_reference': dict(weight_e_y=300.,weight_e_psi=90.,weight_steering_target=.5,
                         weight_steering_slew=2.,sqp_steering_trust_region_rad=.12,
                         replay_reference_seed=True),
    'open_loop_baseline': dict(feedback_rollout=False),
    'open_loop_balanced': dict(weight_e_y=300.,weight_e_psi=90.,weight_steering_target=.5,
                         weight_steering_slew=2.,feedback_rollout=False),
    'open_loop_wide': dict(weight_e_y=300.,weight_e_psi=90.,weight_steering_target=.5,
                         weight_steering_slew=2.,sqp_steering_trust_region_rad=.12,
                         feedback_rollout=False),
    'precise_balanced': dict(weight_e_y=300.,weight_e_psi=90.,weight_steering_target=.5,
                         weight_steering_slew=2.,solver_tolerance=.005),
    'precise_wide': dict(weight_e_y=300.,weight_e_psi=90.,weight_steering_target=.5,
                         weight_steering_slew=2.,solver_tolerance=.005,
                         sqp_steering_trust_region_rad=.12),
}


def parameters(path):
    return yaml.safe_load(path.read_text())['/**']['ros__parameters']


def replay_arguments(p):
    return [p.get('admm_rho',7.),p.get('admm_rho_steering',0.),
        p.get('admm_rho_throttle',0.),p.get('admm_over_relaxation',1.6),1.,
        p.get('admm_rho_ey',0.),p.get('admm_rho_forward_speed',0.),
        p.get('admm_rho_actual_steering',0.),p.get('admm_rho_max',32.),
        p.get('adaptive_rho_update_period',10),p.get('solver_tolerance',.02),
        ('reference-seed' if p.get('replay_reference_seed') else 'independent'),
        'warm',p.get('jacobian_threads',8),p['max_solver_iterations'],
        p.get('startup_nominal_throttle_seed',.25),1.,1.,.1,.1,
        p.get('corridor_margin_m',.05),p.get('first_prediction_corridor_margin_m',.05),
        p.get('admm_rho_input',7.),p['weight_steering_slew'],p['weight_throttle_slew'],
        int(p.get('adaptive_rho',True)),p.get('line_search_max_backtracks',4),1.,
        *[p[k] for k in WEIGHTS],p.get('max_sqp_iterations',2),
        p.get('overspeed_curvature_gain_m',0.),p.get('underspeed_curvature_gain_m',0.),
        p.get('longitudinal_slip_curvature_gain_m',0.),
        p.get('sqp_steering_trust_region_rad',0.),p.get('sqp_throttle_trust_region',0.),
        int(p.get('feedback_rollout',False)),p.get('reference_clearance_reserve_m',0.),
        p.get('nonlinear_corridor_tolerance_m',.005)]


def export_corpus(run, destination, max_samples):
    rows=[]
    for line in (run/'mpc_cycle_detail_diagnostics.jsonl').open():
        record=json.loads(line)['payload']
        try:
            values=to_solver_row(record)
            controls=record['nominal_controls']
            if len(controls)!=30 or any(len(pair)!=2 for pair in controls):
                continue
            if not all(math.isfinite(float(v)) for pair in controls for v in pair):
                continue
        except (KeyError,TypeError,ValueError):
            continue
        rows.append((values,controls))
    if not rows:
        raise ValueError(f'No complete saved state/nominal pairs: {run}')
    stride=max(1,math.ceil(len(rows)/max_samples))
    indices=sorted(set(range(0,len(rows),stride)) | set(range(max(0,len(rows)-20),len(rows))))
    inputs,seeds=destination/'states.csv',destination/'nominals.csv'
    destination.mkdir(parents=True)
    with inputs.open('w') as out,seeds.open('w') as nominal:
        out.write('# '+HEADER+'\n')
        a,b=csv.writer(out),csv.writer(nominal)
        for i,index in enumerate(indices):
            state,controls=rows[index]
            a.writerow(state)
            b.writerow([i,*[v for pair in controls for v in pair]])
    return inputs,seeds,indices


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='append',type=Path,required=True)
    parser.add_argument('--parameters',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-samples',type=int,default=80)
    parser.add_argument('--profiles',nargs='+',choices=tuple(PROFILES),default=list(PROFILES))
    parser.add_argument('--binary',type=Path,default=ROOT/'build/f1tenth_mpc/mpc_bag_solver_replay')
    args=parser.parse_args()
    if args.max_samples<1:parser.error('--max-samples must be positive')
    args.output.mkdir(parents=True,exist_ok=False)
    base=parameters(ROOT/'f1tenth_mpc/config/mpc_competition.yaml')
    base.update(parameters(args.parameters))
    corpus=[]
    for run in args.run:
        inputs,seeds,indices=export_corpus(run,args.output/run.name,args.max_samples)
        corpus.append((run,inputs,seeds,indices))
    report=dict(schema='mpc-weight-bundle-replay/v1',binary_sha256=sha256_file(args.binary),
        source='recorded states and nominal controls; cold ADMM per state',
        runtime_validated=False,profiles={})
    for name in args.profiles:
        changes=PROFILES[name]
        p=dict(base,**changes)
        profile_path=args.output/f'{name}.yaml'
        profile_path.write_text(yaml.safe_dump({'/**':{'ros__parameters':{
            k:v for k,v in p.items() if k!='replay_reference_seed'}}},sort_keys=False))
        result=dict(parameters=str(profile_path),runs={})
        for run,inputs,seeds,indices in corpus:
            env=dict(os.environ,MPC_REPLAY_NOMINAL_SEED_CSV=str(seeds.resolve()))
            if p.get('replay_reference_seed'):
                env.pop('MPC_REPLAY_NOMINAL_SEED_CSV',None)
            raw=args.output/run.name/f'{name}.csv'
            command=[str(args.binary.resolve()),str((run/'raceline.csv').resolve()),
                     str(inputs.resolve()),*[str(v) for v in replay_arguments(p)]]
            with raw.open('w') as out,raw.with_suffix('.stderr.log').open('w') as err:
                subprocess.run(command,stdout=out,stderr=err,env=env,check=True)
            with raw.open() as stream:rows=list(csv.DictReader(stream))
            summary=summarize_rows(rows)
            accepted=[r for r in rows if r['status'] in ('0','1','8')]
            summary['predicted_tracking_mean']={}
            for key in ('predicted_cross_track_rms_m','predicted_heading_rms_rad',
                        'predicted_speed_rms_mps','max_plan_steering_delta'):
                values=[float(r[key]) for r in accepted if math.isfinite(float(r[key]))]
                summary['predicted_tracking_mean'][key]=sum(values)/len(values) if values else None
            summary['diagnostic_indices']=indices
            result['runs'][run.name]=summary
            print(name,run.name,'accepted',summary['exact_accepted'],'/',summary['rows'],
                  'converged',summary['admm_termination_counts'].get('converged',0),
                  'tracking',summary['predicted_tracking_mean'],
                  'cycle_ms',summary['timing_ms']['cycle_ms'],flush=True)
        report['profiles'][name]=result
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
