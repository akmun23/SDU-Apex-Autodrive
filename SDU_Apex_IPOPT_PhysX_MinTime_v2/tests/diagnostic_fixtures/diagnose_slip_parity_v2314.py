#!/usr/bin/env python3
"""Compare the ACTUAL v2.3.14 generated OCP slip model to saved native samples.

Does not launch native PhysX or IPOPT. Reads v2.3.13 per_wheel.csv from the
same 2.4 m/s safe flying lap; validates wheel ordering and trial completion.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from recover import load_modules
from source_vehicle import current_model, bind_current_wheel_moment_arms
from .diagnose_slip_v2313 import WHEELS, build_casadi_audit, build_audit, summarize


def reconstruct_records(rows: list[dict]) -> list[dict]:
    if not rows or len(rows)%4:
        raise ValueError('Expected four wheel records per sample')
    output=[]
    for k in range(0,len(rows),4):
        group=rows[k:k+4]
        if [r['wheel'] for r in group]!=list(WHEELS):
            raise ValueError('Native wheel CSV is unordered / missing a wheel')
        for field in ('time_s','progress_m','native_u_mps','native_v_mps',
                      'native_yaw_rate_radps','native_steering_rad'):
            x=np.array([float(r[field]) for r in group])
            if not np.isfinite(x).all() or np.max(abs(x-x[0]))>1e-9:
                raise ValueError('Inconsistent native measurement: '+field)
        def col(name):
            vals=np.array([float(r[name]) for r in group])
            if not np.isfinite(vals).all():raise ValueError('Nonfinite column '+name)
            return vals
        r=group[0]
        output.append({
            'time_s':float(r['time_s']),
            'progress_m':float(r['progress_m']),
            'u_mps':float(r['native_u_mps']),
            'v_mps':float(r['native_v_mps']),
            'yaw_rate_radps':float(r['native_yaw_rate_radps']),
            'steering_rad':float(r['native_steering_rad']),
            'wheel_omega_radps':col('native_omega_radps'),
            'normal_load_n':col('native_suspension_force_n'),
            'longitudinal_slip':col('native_long_slip'),
            'lateral_slip':col('native_lat_slip'),
            'tire_longitudinal_force_n':col('native_callback_fx_n'),
            'tire_lateral_force_n':col('native_callback_fy_n'),
        })
    if len(output)!=481:
        raise ValueError('Expected complete 481-sample flying lap, got '+str(len(output)))
    return output


def compare_baseline(old_rows: list[dict], new_rows: list[dict]) -> dict:
    if len(old_rows)!=len(new_rows):raise ValueError('Baseline and corrected lengths differ')
    def metrics(rows, modeled, actual):
        err=np.array([float(r[modeled])-float(r[actual]) for r in rows])
        return {'rms':float(np.sqrt(np.mean(err**2))),
                'max_abs':float(np.max(abs(err)))}
    fields={'lateral_slip':('model_lat_slip','native_lat_slip'),
            'longitudinal_slip':('model_long_slip','native_long_slip'),
            'lateral_tire_force_n':('model_reconstructed_fy_n','native_callback_fy_n'),
            'longitudinal_tire_force_n':('model_reconstructed_fx_n','native_callback_fx_n')}
    result={k:{'before':metrics(old_rows,*v),'after':metrics(new_rows,*v)}
            for k,v in fields.items()}
    if result['lateral_tire_force_n']['after']['rms']>=.5*result['lateral_tire_force_n']['before']['rms']:
        raise RuntimeError('Corrected tire-force RMS did not improve at least twofold')
    if result['lateral_slip']['after']['rms']>=.5*result['lateral_slip']['before']['rms']:
        raise RuntimeError('Corrected lateral-slip RMS did not improve at least twofold')
    if abs(result['longitudinal_slip']['after']['rms']-result['longitudinal_slip']['before']['rms'])>1e-9:
        raise RuntimeError('Longitudinal slip changed unexpectedly')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo-root',type=Path,default=Path('..'))
    p.add_argument('--source-dir',type=Path,default=Path('./native_slip_v2313'))
    p.add_argument('--output',type=Path,default=Path('./native_slip_v2314'))
    p.add_argument('--allow-history-fetch',action='store_true')
    a=p.parse_args()
    src=a.source_dir.resolve();output=a.output.resolve()
    trial=json.loads((src/'trial.json').read_text())
    if not trial.get('completed_flying_lap') or trial.get('failure'):
        raise ValueError('Refusing to use incomplete/unsafe native lap')
    with (src/'per_wheel.csv').open(newline='') as f:
        previous=list(csv.DictReader(f))
    records=reconstruct_records(previous)
    root=a.repo_root.resolve();pkg=Path(__file__).resolve().parent
    original,wheel,_,_,meta=load_modules(root,pkg,a.allow_history_fetch,'limited-memory')
    if not meta.get('native_lateral_slip_sign_corrected'):
        raise RuntimeError('Loaded wheel model does not have v2.3.14 native slip frame fix')
    model,_=current_model(root,wheel)
    bind_current_wheel_moment_arms(root,original,wheel)
    fun=build_casadi_audit(wheel,model)
    new_rows=build_audit(records,fun)
    comparison=compare_baseline(previous,new_rows)
    summary=summarize(new_rows,trial)
    summary.update({'diagnostic':'v2314_generated_OCP_native_slip_coordinate_parity',
                    'physical_model_changed':True,
                    'comparison_to_v2313':comparison,
                    'native_replay_ran':False,
                    'actual_ipopt_solve_ran':False,
                    'working_wheel_model_sha256':meta['working_wheel_model_sha256']})
    output.mkdir(parents=True,exist_ok=True)
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (output/'per_wheel.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(new_rows[0]));writer.writeheader();writer.writerows(new_rows)
    print('Reevaluated',len(new_rows),'wheel records from the previous native lap')
    print(json.dumps(comparison,indent=2))
    print('Wrote',output/'summary.json',output/'per_wheel.csv')
    print('No native replay or IPOPT solve ran')

if __name__=='__main__': main()
