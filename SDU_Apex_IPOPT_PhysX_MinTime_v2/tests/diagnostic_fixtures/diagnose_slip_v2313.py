#!/usr/bin/env python3
"""v2.3.13: audit PhysX tire slip geometry against the actual IPOPT wheel model.

Captures a *new* wall-safe 2.4 m/s native PhysX lap at 25 ms output cadence,
including per-wheel angular speeds in the same sample as actual tire slips,
forces and suspension loads. Does not start IPOPT or change OCP physics.

CAUTION: wheel angular speeds/body velocities are observed *after* scene
integration of each 1 ms tick; the wheel slip reported by PhysX originates
from the vehicle update preceding that scene step. Finite timing bias can
therefore remain. This is a slip-reconstruction diagnostic, not an exact
native contact-velocity proof.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import casadi as ca

from map_track import MapGrid, map_to_track
from run_ipopt import validate_geometry
from native_seed import one_trial
from recover import load_modules
from source_vehicle import bind_current_wheel_moment_arms, current_model

WHEELS=('fl','fr','rl','rr')


def build_casadi_audit(wheel, model):
    """Run the genuine OCP CasADi slip and native-callback tire equations."""
    u, v, yaw, steering=[ca.SX.sym(x) for x in ('u','v','yaw_rate','steering')]
    omegas=ca.SX.sym('omega',4)
    loads=ca.SX.sym('load',4)
    body_fx,body_fy,_,fx,fy,sx,sy=wheel.tire_forces_casadi(
        model,u,v,yaw,steering,[omegas[i] for i in range(4)],
        [loads[i] for i in range(4)])
    x=ca.SX.sym('measured_long_slip',4)
    y=ca.SX.sym('measured_lateral_slip',4)
    raw_fx,raw_fy=[],[]
    for i in range(4):
        _,f_long,f_lat,_=wheel._unity_tire_force_shader_casadi(
            x[i],y[i],loads[i],model.wheel_radius_m,
            model.forward_curve[i],model.lateral_curve[i])
        raw_fx.append(f_long);raw_fy.append(f_lat)
    return ca.Function('v2313_contact_reconstruction',
                       [u,v,yaw,steering,omegas,loads,x,y],
                       [ca.vertcat(*sx),ca.vertcat(*sy),ca.vertcat(*fx),
                        ca.vertcat(*fy),ca.vertcat(*raw_fx),
                        ca.vertcat(*raw_fy),body_fx,body_fy])


def build_audit(records, tire_func):
    """Compare reconstructed contact slip and forces on identical native inputs."""
    if len(records)<20:
        raise ValueError('Missing native flying-lap contact records')
    output=[]
    for t in records:
        omega=np.asarray(t['wheel_omega_radps'],float)
        loads=np.asarray(t['normal_load_n'],float)
        sx_native=np.asarray(t['longitudinal_slip'],float)
        sy_native=np.asarray(t['lateral_slip'],float)
        native_fx=np.asarray(t['tire_longitudinal_force_n'],float)
        native_fy=np.asarray(t['tire_lateral_force_n'],float)
        if any(a.shape!=(4,) or not np.isfinite(a).all() for a in
               (omega,loads,sx_native,sy_native,native_fx,native_fy)):
            raise ValueError('Invalid shape/nonfinite native wheel measurement')
        values=tire_func(float(t['u_mps']),float(t['v_mps']),
                         float(t['yaw_rate_radps']),float(t['steering_rad']),
                         omega,loads,sx_native,sy_native)
        predicted=[np.asarray(val.full(),float).reshape(-1) for val in values]
        sx_model,sy_model,fx_model,fy_model,fx_given_native,fy_given_native,_,_=predicted
        for i,w in enumerate(WHEELS):
            output.append({
                'time_s':float(t['time_s']),
                'progress_m':float(t['progress_m']),
                'wheel':w,
                'native_u_mps':float(t['u_mps']),
                'native_v_mps':float(t['v_mps']),
                'native_yaw_rate_radps':float(t['yaw_rate_radps']),
                'native_steering_rad':float(t['steering_rad']),
                'native_omega_radps':float(omega[i]),
                'native_suspension_force_n':float(loads[i]),
                'native_long_slip':float(sx_native[i]),
                'model_long_slip':float(sx_model[i]),
                'long_slip_difference':float(sx_model[i]-sx_native[i]),
                'native_lat_slip':float(sy_native[i]),
                'model_lat_slip':float(sy_model[i]),
                'lat_slip_difference':float(sy_model[i]-sy_native[i]),
                'native_callback_fx_n':float(native_fx[i]),
                'model_reconstructed_fx_n':float(fx_model[i]),
                'model_at_native_slip_fx_n':float(fx_given_native[i]),
                'native_callback_fy_n':float(native_fy[i]),
                'model_reconstructed_fy_n':float(fy_model[i]),
                'model_at_native_slip_fy_n':float(fy_given_native[i]),
            })
    return output


def error_metrics(rows, prefix):
    arr=np.asarray([r[prefix] for r in rows],float)
    if not np.isfinite(arr).all():raise ValueError('Nonfinite modeled slip/force error')
    return {'rms':float(np.sqrt(np.mean(arr*arr))),
            'max_abs':float(np.max(np.abs(arr))),
            'signed_mean':float(np.mean(arr))}


def summarize(rows, trial):
    if len(rows)%4 or len(rows)<80:
        raise ValueError('Missing one or more wheel observations')
    for r in rows:
        r['model_vs_native_fx_error_n']=r['model_reconstructed_fx_n']-r['native_callback_fx_n']
        r['model_vs_native_fy_error_n']=r['model_reconstructed_fy_n']-r['native_callback_fy_n']
        r['native_slip_shader_fx_error_n']=r['model_at_native_slip_fx_n']-r['native_callback_fx_n']
        r['native_slip_shader_fy_error_n']=r['model_at_native_slip_fy_n']-r['native_callback_fy_n']
    metric_fields={
        'longitudinal_slip':'long_slip_difference',
        'lateral_slip':'lat_slip_difference',
        'reconstructed_fx_n':'model_vs_native_fx_error_n',
        'reconstructed_fy_n':'model_vs_native_fy_error_n',
        'callback_at_native_slip_fx_n':'native_slip_shader_fx_error_n',
        'callback_at_native_slip_fy_n':'native_slip_shader_fy_error_n',
    }
    s={'diagnostic':'native_physx_contact_slip_reconstruction_vs_actual_generated_casadi_OCP',
       'native_flying_lap':trial, 'wheel_samples':len(rows),
       'native_data_temporal_resolution_s':0.025,
       'internal_physx_step_s':0.001,
       'actual_ipopt_solve_ran':False,
       'physical_model_changed':False,
       'sample_timing_caveat':(
           'Measured native tire slips originate from the preceding vehicle update; '
           'body velocities and wheel angular speeds are sampled after PhysX scene integration. '
           'This 1 ms state staggering may bias slip error, especially due to stiff wheel damping.'),
       'metrics':{k:error_metrics(rows,v) for k,v in metric_fields.items()},
       'per_wheel':{w:{k:error_metrics([r for r in rows if r['wheel']==w],v)
                      for k,v in metric_fields.items()} for w in WHEELS}}
    s['worst_lateral_slip_samples']=sorted(
        ({'time_s':r['time_s'],'progress_m':r['progress_m'],
          'wheel':r['wheel'],'native':r['native_lat_slip'],
          'modeled':r['model_lat_slip'],'difference':r['lat_slip_difference']}
         for r in rows),key=lambda v:abs(v['difference']),reverse=True)[:12]
    s['worst_force_samples']=sorted(
        ({'time_s':r['time_s'],'progress_m':r['progress_m'],
          'wheel':r['wheel'],'native_fy_n':r['native_callback_fy_n'],
          'modeled_fy_n':r['model_reconstructed_fy_n'],
          'difference_fy_n':r['model_vs_native_fy_error_n']}
         for r in rows),key=lambda v:abs(v['difference_fy_n']),reverse=True)[:12]
    return s


def run():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo-root',type=Path,default=Path('..'))
    p.add_argument('--physx-binary',type=Path,default=Path('./native_physx_vehicle_step'))
    p.add_argument('--output',type=Path,default=Path('./native_slip_v2313'))
    p.add_argument('--allow-history-fetch',action='store_true')
    a=p.parse_args()
    root=a.repo_root.resolve(); out=a.output.resolve();pkg=Path(__file__).resolve().parent
    old,wheel,trackmod,unused,meta=load_modules(root,pkg,a.allow_history_fetch,'limited-memory')
    if not meta.get('working_wheel_model_parity_applied'):
        raise RuntimeError('Working source-native tire callback patch missing')
    model,_=current_model(root,wheel)
    bind_current_wheel_moment_arms(root,old,wheel)
    fn=build_casadi_audit(wheel,model)
    grid=MapGrid.load(root/'f1tenth_planning/maps/autodrive_practice_20260924_b.pgm')
    center=root/'f1tenth_planning/trajectories/autodrive_practice_20260924_b/autodrive_practice_20260924_b_mintime_raceline.csv'
    ref=map_to_track(grid,center,spacing=.12)
    track=trackmod.load_centerline_csv(center,.20)
    validate_geometry(track,grid,.15)
    native_cpp=(root/'f1tenth_localization/src/physx_native_vehicle_step.cpp').read_text()
    if ('longitudinalSlip' not in native_cpp
        or 'g_tire_lateral_force_n[i]' not in native_cpp):
        raise RuntimeError('Native contact telemetry source unexpected')
    out.mkdir(parents=True,exist_ok=True)
    records=[]
    print('Native 2.4 m/s flight + genuine OCP slip reconstruction; no IPOPT.',flush=True)
    _,trial=one_trial(root,a.physx_binary.resolve(),'native',ref,grid,track,
                      out/'warmstart_capture',2.4,.15,contact_records=records)
    (out/'trial.json').write_text(json.dumps(trial,indent=2)+'\n')
    if not trial.get('completed_flying_lap'):
        raise RuntimeError('Native replay did not complete safely: '+str(trial))
    rows=build_audit(records,fn)
    summary=summarize(rows,trial)
    summary['working_wheel_model_sha256']=meta['working_wheel_model_sha256']
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (out/'per_wheel.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(summary,indent=2),flush=True)
    print('Saved',out/'summary.json','and',out/'per_wheel.csv',flush=True)
    print('NO IPOPT solve or racing candidate generated.')

if __name__=='__main__': run()
