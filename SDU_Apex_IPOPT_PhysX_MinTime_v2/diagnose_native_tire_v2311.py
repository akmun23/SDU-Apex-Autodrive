#!/usr/bin/env python3
"""Audit native PxVehicle tire callback forces during a safe flying lap.

Uses the existing 2.4 m/s native warm-start controller, 25 x 1 ms real
PhysX vehicle ticks per control packet. NO IPOPT run or racing trajectory.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np

from map_track import MapGrid, map_to_track
from run_ipopt import validate_geometry
from native_seed import one_trial
from recover import load_modules
from source_vehicle import bind_current_wheel_moment_arms, current_model

WHEELS = ("fl", "fr", "rl", "rr")
NATIVE_FRICTION = 1.0   # f1tenth_localization/src/physx_native_vehicle_step.cpp


def curve_response_without_stiffness(slip, curve):
    """Literal source translation of C++ curveResponse(float, curve)."""
    peak_s, peak_y, asym_s, asym_y, _ = curve
    x = abs(float(slip))
    if x < peak_s:
        t = x / peak_s
        val = peak_y * (t + t*t - t*t*t)
    elif x < asym_s:
        t = (x-peak_s)/(asym_s-peak_s)
        val = peak_y + (asym_y-peak_y) * (3*t*t-2*t*t*t)
    else:
        val = asym_y
    return math.copysign(val, float(slip))


def filtered_tire_load(suspension_force, sprung_mass, wheel_mass, gravity=9.81):
    """PhysX 4.1 normalized-load default filter, as used by CasADi."""
    rest = (float(sprung_mass)+float(wheel_mass))*gravity
    normalized = float(suspension_force)/rest
    filtered = (0.2308 if normalized <= 0.0
                else 3.0 if normalized >= 3.0
                else 0.2308 + normalized * (3.0-0.2308)/3.0)
    return filtered * rest


def predicted_native_callback(slip_x, slip_y, tire_load, forward, lateral,
                              stiffness_inside_combined_norm=False):
    """Compare native C++ order to historical CasADi stiffness placement."""
    raw_fx = curve_response_without_stiffness(slip_x, forward)
    raw_fy = curve_response_without_stiffness(slip_y, lateral)
    norm_fx = (raw_fx * (forward[4] if stiffness_inside_combined_norm else 1.0)
               / forward[1])
    norm_fy = (raw_fy * (lateral[4] if stiffness_inside_combined_norm else 1.0)
               / lateral[1])
    combined = 1.0 / math.sqrt(max(1.0, norm_fx*norm_fx+norm_fy*norm_fy))
    return (float(tire_load)*NATIVE_FRICTION*raw_fx*combined*forward[4],
            -float(tire_load)*NATIVE_FRICTION*raw_fy*combined*lateral[4])


def calculate_contact_audit(records, wheels):
    """Compare actual native Fx/Fy at supplied native slip, no reconstructed slip."""
    if not records:
        raise ValueError("No native flying-lap tire-force observations")
    rows = []
    for obs in records:
        for i, wheel in enumerate(wheels):
            sx=float(obs['longitudinal_slip'][i])
            sy=float(obs['lateral_slip'][i])
            fz=float(obs['normal_load_n'][i])
            load=filtered_tire_load(
                fz, wheel['sprung_mass_kg'], wheel['mass_kg'])
            fwd=tuple(float(v) for v in wheel['forward_curve'])
            lat=tuple(float(v) for v in wheel['sideways_curve'])
            new_x,new_y=predicted_native_callback(sx,sy,load,fwd,lat)
            old_x,old_y=predicted_native_callback(
                sx,sy,load,fwd,lat,stiffness_inside_combined_norm=True)
            true_x=float(obs['tire_longitudinal_force_n'][i])
            true_y=float(obs['tire_lateral_force_n'][i])
            # Infer the effective PhysX shader tireLoad from either force
            # component when not close to zero. This separates input-load
            # discrepancies from the tire-friction-curve implementation.
            unit_x,unit_y=predicted_native_callback(sx,sy,1.,fwd,lat)
            if abs(unit_y) >= 0.1:
                inferred_load=true_y/unit_y
                inferred_from='lateral'
            elif abs(unit_x) >= 0.1:
                inferred_load=true_x/unit_x
                inferred_from='longitudinal'
            else:
                inferred_load=float('nan')
                inferred_from='no_reliable_force_component'
            rows.append({
                'time_s':float(obs['time_s']),
                'progress_m':float(obs['progress_m']),
                'wheel':WHEELS[i],
                'u_mps':float(obs['u_mps']),
                'v_mps':float(obs['v_mps']),
                'yaw_rate_radps':float(obs['yaw_rate_radps']),
                'steering_rad':float(obs['steering_rad']),
                'throttle_norm':float(obs['throttle_norm']),
                'native_suspension_force_n':fz,
                'symbolic_filtered_tire_load_n':load,
                'native_long_slip':sx,
                'native_lat_slip':sy,
                'native_callback_fx_n':true_x,
                'native_callback_fy_n':true_y,
                'source_shader_fx_estimate_n':new_x,
                'source_shader_fy_estimate_n':new_y,
                'historical_shader_fx_estimate_n':old_x,
                'historical_shader_fy_estimate_n':old_y,
                'inferred_callback_tire_load_n':inferred_load,
                'load_inferred_from':inferred_from,
            })
    def stats(prefix):
        d=np.asarray([
            [r[f'{prefix}_fx_n' if prefix=='native_callback'
               else f'{prefix}_fx_estimate_n'],
             r[f'{prefix}_fy_n' if prefix=='native_callback'
               else f'{prefix}_fy_estimate_n']] for r in rows],float)
        return d
    native=stats('native_callback')
    source=stats('source_shader')
    historical=stats('historical_shader')
    err_source=source-native
    err_old=historical-native
    inferred=np.asarray([r['inferred_callback_tire_load_n'] for r in rows])
    filtered=np.asarray([r['symbolic_filtered_tire_load_n'] for r in rows])
    valid=np.isfinite(inferred)&(inferred>0.)
    summary={
        'diagnostic':'native_1khz_physx_tire_callback_vs_scalar_shader',
        'samples':len(records),
        'wheel_samples':len(rows),
        'native_callback_observed_nonzero_fraction':float(np.mean(np.max(np.abs(native),axis=1)>0.01)),
        'source_stiffness_outside_norm_rms_force_error_n':float(np.sqrt(np.mean(err_source**2))),
        'historical_stiffness_inside_norm_rms_force_error_n':float(np.sqrt(np.mean(err_old**2))),
        'source_stiffness_outside_norm_max_force_error_n':float(np.max(np.abs(err_source))),
        'historical_stiffness_inside_norm_max_force_error_n':float(np.max(np.abs(err_old))),
        'source_minus_historical_predicted_max_component_n':float(np.max(np.abs(source-historical))),
        'inferred_callback_load_coverage_fraction':float(np.mean(valid)),
        'inferred_callback_load_minus_symbolic_filtered_load_rms_n':(
            float(np.sqrt(np.mean((inferred[valid]-filtered[valid])**2)))
            if valid.any() else None),
        'inferred_callback_load_minus_symbolic_filtered_load_mean_n':(
            float(np.mean(inferred[valid]-filtered[valid])) if valid.any() else None),
        'native_friction_pair':'1.0 (verified exact PxVehicleDrivableSurfaceToTireFrictionPairs source)',
        'important_caveat':(
            'Native shader receives PhysX tireLoad that is not directly present '
            'in the existing compact native response; load comparisons use the '
            'documented filter and separately infer effective callback load. '
            'The native contact output is sampled every 25ms after the last '
            '1ms PxVehicleUpdates, not a 143-node IPOPT collocation trace.'),
        'never_accept_as_racing_lap':True,
    }
    return summary,rows


def run():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo-root',type=Path,default=Path('..'))
    ap.add_argument('--physx-binary',type=Path,default=Path('./native_physx_vehicle_step'))
    ap.add_argument('--output',type=Path,default=Path('native_contact_v2311'))
    ap.add_argument('--allow-history-fetch',action='store_true')
    args=ap.parse_args()
    root=args.repo_root.resolve()
    output=args.output.resolve()
    pkg=Path(__file__).resolve().parent
    native_cpp=root/'f1tenth_localization/src/physx_native_vehicle_step.cpp'
    native_text=native_cpp.read_text()
    if 'frictionPairs->setTypePairFriction(0, 0, 1.0f)' not in native_text:
        raise RuntimeError('Native friction constant changed; do not assume 1.0')
    mapping=root/'f1tenth_planning/maps/autodrive_practice_20260924_b.pgm'
    center=root/'f1tenth_planning/trajectories/autodrive_practice_20260924_b/autodrive_practice_20260924_b_mintime_raceline.csv'
    old,wheel,trackmod,_,_=load_modules(root,pkg,args.allow_history_fetch,'limited-memory')
    model,_=current_model(root,wheel)
    bind_current_wheel_moment_arms(root,old,wheel)
    grid=MapGrid.load(mapping)
    ref=map_to_track(grid,center,spacing=.12)
    track=trackmod.load_centerline_csv(center,.20)
    validate_geometry(track,grid,.15)
    wheels=json.loads((root/'f1tenth_localization/config/physx_odom_reset_initial.json').read_text())['initial']['wheel_inputs']
    output.mkdir(parents=True,exist_ok=True)
    records=[]
    print('Capturing native PhysX tire forces/slips during safe 2.4m/s flying lap; no IPOPT.',flush=True)
    _,trial=one_trial(root,args.physx_binary.resolve(),'native',ref,grid,track,
                      output/'warmstart_capture',2.4,.15,
                      contact_records=records)
    (output/'trial.json').write_text(json.dumps(trial,indent=2)+'\n')
    if not trial.get('completed_flying_lap'):
        raise RuntimeError('Native contact lap failed: '+str(trial.get('failure')))
    summary,rows=calculate_contact_audit(records,wheels)
    summary['native_trial']=trial
    summary['source_hash_note']='native source and current restored wheel model; no fitting'
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (output/'per_wheel.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary,indent=2),flush=True)
    print('Saved',output/'summary.json','and',output/'per_wheel.csv',flush=True)
    print('NO IPOPT solve or racing candidate was generated.',flush=True)
    return 0

if __name__=='__main__':
    raise SystemExit(run())
