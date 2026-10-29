#!/usr/bin/env python3
"""Independent native-force-to-chassis-frame parity audit (no IPOPT/replay).

Uses saved 481-station, 25-ms native contact telemetry from v2.3.14 and
compares native wheel forces rotated into vehicle left-positive coordinates
against observed dv/dt + u*yaw_rate. Validates that the *actual generated*
CasADi wheel force function now performs the same rotation. The 25-ms finite
velocity derivative and time-staggered native slips limit exact agreement.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np

from .diagnose_slip_parity_v2314 import reconstruct_records
from .diagnose_slip_v2313 import build_casadi_audit
from recover import load_modules
from source_vehicle import current_model, bind_current_wheel_moment_arms


def metric(pred, truth):
    residual = np.asarray(pred, float) - np.asarray(truth, float)
    return {'rms_mps2': float(np.sqrt(np.mean(residual**2))),
            'max_abs_mps2': float(np.max(np.abs(residual))),
            'mean_error_mps2': float(np.mean(residual)),
            'pearson_r': float(np.corrcoef(pred, truth)[0, 1])}


def rotation_prediction(records, mass, drag_per_s):
    """Rotate measured raw callback force from native wheel right to model left."""
    times = np.array([r['time_s'] for r in records], float)
    if np.any(np.diff(times) <= 0) or np.max(np.diff(times)) > 0.027:
        raise ValueError('Unexpected nonuniform / unordered native capture times')
    u = np.array([r['u_mps'] for r in records], float)
    v = np.array([r['v_mps'] for r in records], float)
    yaw = np.array([r['yaw_rate_radps'] for r in records], float)
    steer = np.array([r['steering_rad'] for r in records], float)
    fx = np.stack([r['tire_longitudinal_force_n'] for r in records])
    fy = np.stack([r['tire_lateral_force_n'] for r in records])
    if len(records) != 481 or fx.shape != (481,4) or fy.shape != (481,4):
        raise ValueError('Expected full 481-observation four-wheel native lap')
    tan = np.tan(steer)
    # Center steering is left-positive in this model; front native steering
    # angles are the corresponding Ackermann left/right geometries.
    wb, track = 0.324, 0.236
    angles = np.column_stack((
        np.arctan2(2*wb*tan, 2*wb-track*tan),
        np.arctan2(2*wb*tan, 2*wb+track*tan),
        np.zeros_like(tan), np.zeros_like(tan)))
    predicted_before = (fx*np.sin(angles) + fy*np.cos(angles)).sum(axis=1)/mass - drag_per_s*v
    predicted_after = (fx*np.sin(angles) - fy*np.cos(angles)).sum(axis=1)/mass - drag_per_s*v
    measured_ay = np.gradient(v,times,edge_order=2) + u*yaw
    return (times, measured_ay, predicted_before, predicted_after, angles)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root',type=Path,default=Path('..'))
    parser.add_argument('--source-dir',type=Path,default=Path('native_slip_v2314'))
    parser.add_argument('--output',type=Path,default=Path('native_force_frame_v2315'))
    parser.add_argument('--allow-history-fetch',action='store_true')
    a=parser.parse_args()
    src=a.source_dir.resolve();out=a.output.resolve()
    summary=json.loads((src/'summary.json').read_text())
    trial=summary.get('native_flying_lap',{})
    if not trial.get('completed_flying_lap') or trial.get('failure'):
        raise ValueError('Native source lap is incomplete or unsafe')
    with (src/'per_wheel.csv').open(newline='') as f:
        records=reconstruct_records(list(csv.DictReader(f)))
    pkg=Path(__file__).resolve().parent
    old,wheel,_,_,meta=load_modules(a.repo_root.resolve(),pkg,
                                  a.allow_history_fetch,'limited-memory')
    if not (meta.get('native_lateral_force_frame_corrected')
            and meta.get('native_lateral_slip_sign_corrected')):
        raise RuntimeError('Actual generated working CasADi model lacks frame corrections')
    model,_=current_model(a.repo_root.resolve(),wheel)
    bind_current_wheel_moment_arms(a.repo_root.resolve(),old,wheel)
    times, actual_ay, previous_ay, corrected_ay, angles=rotation_prediction(
        records,model.mass_kg,model.drag_per_s)
    fun=build_casadi_audit(wheel,model)
    calculated=[]; rotation_errors=[]
    for i,r in enumerate(records):
        values=fun(r['u_mps'],r['v_mps'],r['yaw_rate_radps'],r['steering_rad'],
                   r['wheel_omega_radps'],r['normal_load_n'],
                   r['longitudinal_slip'],r['lateral_slip'])
        x=[np.asarray(v.full(),float).reshape(-1) for v in values]
        # Function output order: modeled sx,sy, four callback fx,fy,
        # callback at measured slip fx/fy, total chassis Fx, total chassis Fy.
        expected=float(np.sum(x[2]*np.sin(angles[i])-x[3]*np.cos(angles[i])))
        total_y=float(x[7][0])
        rotation_errors.append(abs(expected-total_y))
        calculated.append(total_y/model.mass_kg - model.drag_per_s*r['v_mps'])
    if max(rotation_errors)>1e-9:
        raise RuntimeError('Generated CasADi chassis force is not right-to-left transformed')
    predicted_casadi=np.array(calculated)
    result={
        'diagnostic':'v2315_native_callback_force_body_axis_parity',
        'wheel_records':len(records)*4,'timestamps':len(records),
        'native_lap_completed':True,
        'historical_source_untouched':meta['historical_wheel_model_untouched'],
        'working_wheel_model_sha256':meta['working_wheel_model_sha256'],
        'force_rotation_exact_max_n':max(rotation_errors),
        'observed_ay_from_post_scene_25ms_dvdt_mps2':True,
        'before_wrong_native_fy_as_left_positive':metric(previous_ay,actual_ay),
        'after_measured_native_fy_converted_to_left_positive':metric(corrected_ay,actual_ay),
        'after_actual_CasADi_reconstructed_slip':metric(predicted_casadi,actual_ay),
        'caveat':'Native tire Fx/Fy and slips correspond to the vehicle update before the scene solver; post-scene body velocity and wheel omega are staggered by 1 ms. Measured ay is a 25 ms velocity finite difference and can include contact impulses. These checks do not certify full native physics fidelity or IPOPT feasibility.'
    }
    if result['after_measured_native_fy_converted_to_left_positive']['rms_mps2'] >= (
        0.25*result['before_wrong_native_fy_as_left_positive']['rms_mps2']):
        raise RuntimeError('Chassis force frame correction failed native acceleration test')
    out.mkdir(parents=True,exist_ok=True)
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    with (out/'per_sample.csv').open('w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(('time_s','observed_lateral_acceleration_mps2',
                         'legacy_wrong_frame_predicted_ay_mps2',
                         'native_corrected_frame_predicted_ay_mps2',
                         'generated_CasADi_predicted_ay_mps2'))
        writer.writerows(zip(times,actual_ay,previous_ay,corrected_ay,predicted_casadi))
    print(json.dumps(result,indent=2),flush=True)
    print('NO IPOPT or native replay executed.',flush=True)

if __name__=='__main__':main()
