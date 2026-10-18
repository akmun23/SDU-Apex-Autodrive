#!/usr/bin/env python3
"""Verify the ACTUAL v2.3.16 CasADi tire model against native tire telemetry.

No new PhysX run or IPOPT. Use the archived complete 481-station native 2.4m/s
flying lap (v2.3.14) and 143-point native warm-start from the same controller.
Interpolate recorded roll/pitch body rates by periodic track progress. For each
native timestamp compare the actual CasADi OCP tire forces with rates 0 versus
interpolated true rates. Ground truth tire slips and forces are direct native
wheel telemetry. A post-scene 25ms finite-difference ay check is supporting
information only; 1ms contact update/state timing is not exactly synchronized.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import casadi as ca
from diagnose_slip_parity_v2314 import reconstruct_records
from diagnose_force_frame_v2315 import rotation_prediction, metric
from recover import load_modules
from source_vehicle import current_model, bind_current_wheel_moment_arms
from diagnose_slip_v2313 import WHEELS


def interpolate_native_body_rates(seed: dict, records: list[dict], length_m: float):
    n=len(seed['u_mps']); station=np.arange(n,dtype=float)*(length_m/n)
    phase=np.mod(np.asarray([row['progress_m'] for row in records],float),length_m)
    if len(records)!=481 or n!=143:
        raise ValueError('Expected complete 481 native telemetry records and 143 seed stations')
    if not np.isfinite(phase).all() or length_m<25 or length_m>30:
        raise ValueError('Unrecognized track phase/length')
    def interp(key):
        array=np.asarray(seed[key],float)
        if array.shape!=(n,) or not np.isfinite(array).all():
            raise ValueError('Invalid saved native seed rate/state: '+key)
        return np.interp(phase,np.r_[station,length_m],np.r_[array,array[0]])
    alignment={}
    for key,observed in (('u_mps','u_mps'),('v_mps','v_mps'),
                         ('yaw_rate_radps','yaw_rate_radps'),('steering_rad','steering_rad')):
        val=interp(key)
        truth=np.asarray([row[observed] for row in records],float)
        alignment[key+'_rms']=float(np.sqrt(np.mean((val-truth)**2)))
    if alignment['u_mps_rms']>0.035 or alignment['v_mps_rms']>0.03 or alignment['yaw_rate_radps_rms']>0.1 or alignment['steering_rad_rms']>0.03:
        raise RuntimeError('Native seed does not phase-align with contact telemetry: '+repr(alignment))
    return interp('roll_rate_radps'),interp('pitch_rate_radps'),alignment


def build_actual_contact_audit(wheel, model):
    u,v,r,d,rr,pr=[ca.MX.sym(k) for k in ('u','v','yaw','steering','roll_rate','pitch_rate')]
    omega=ca.MX.sym('omega',4);load=ca.MX.sym('load',4)
    fx,fy,_,f_wx,f_wy,sx,sy=wheel.tire_forces_casadi(
        model,u,v,r,d,[omega[i] for i in range(4)],
        [load[i] for i in range(4)],rr,pr)
    return ca.Function('v2316_actual_contact',
        [u,v,r,d,omega,load,rr,pr],
        [ca.vertcat(*sx),ca.vertcat(*sy),
         ca.vertcat(*f_wx),ca.vertcat(*f_wy),fx,fy])


def rms(err):
    a=np.asarray(err,dtype=float)
    if not np.isfinite(a).all():raise ValueError('Nonfinite comparison')
    return {'rms':float(np.sqrt(np.mean(a*a))),
            'max_abs':float(np.max(abs(a))),
            'signed_mean':float(np.mean(a))}


def audit_contact(records, rate_roll, rate_pitch, contact_func, model):
    if len(records)!=len(rate_roll) or len(records)!=len(rate_pitch):
        raise ValueError('Unaligned contact record and rate lengths')
    old,new=[],[]
    for idx,row in enumerate(records):
        args=(row['u_mps'],row['v_mps'],row['yaw_rate_radps'],
              row['steering_rad'],row['wheel_omega_radps'],row['normal_load_n'])
        def get(rr,pr):
            return [np.asarray(z.full(),dtype=float).reshape(-1)
                    for z in contact_func(*args,rr,pr)]
        old.append(get(0.,0.))
        new.append(get(float(rate_roll[idx]),float(rate_pitch[idx])))
    truth={
        'longitudinal_slip':np.stack([q['longitudinal_slip'] for q in records]),
        'lateral_slip':np.stack([q['lateral_slip'] for q in records]),
        'tire_longitudinal_force_n':np.stack([q['tire_longitudinal_force_n'] for q in records]),
        'tire_lateral_force_n':np.stack([q['tire_lateral_force_n'] for q in records]),
    }
    labels=('longitudinal_slip','lateral_slip','tire_longitudinal_force_n',
            'tire_lateral_force_n')
    metrics={};wheel_records=[]
    for j,label in enumerate(labels):
        before=np.stack([row[j] for row in old]);after=np.stack([row[j] for row in new])
        target=truth[label]
        metrics[label]={'planar':rms(before-target),'full_contact':rms(after-target)}
        for k in range(len(records)):
            for i,w in enumerate(WHEELS):
                if j==0:
                    wheel_records.append({'time_s':records[k]['time_s'],
                                          'progress_m':records[k]['progress_m'],
                                          'wheel':w,
                                          'roll_rate_radps':float(rate_roll[k]),
                                          'pitch_rate_radps':float(rate_pitch[k])})
                wheel_records[k*4+i].update({
                    'native_'+label:float(target[k,i]),
                    'planar_'+label:float(before[k,i]),
                    'full_contact_'+label:float(after[k,i])})
    _,observed,_,native_based,_=rotation_prediction(records,model.mass_kg,model.drag_per_s)
    b=np.asarray([float(out[-1][0])/model.mass_kg - model.drag_per_s*records[i]['v_mps'] for i,out in enumerate(old)])
    a=np.asarray([float(out[-1][0])/model.mass_kg - model.drag_per_s*records[i]['v_mps'] for i,out in enumerate(new)])
    m_ay={'native_measured_forces':metric(native_based,observed),
          'planar_CasADi':metric(b,observed),
          'full_contact_CasADi':metric(a,observed)}
    per_sample=[{'time_s':records[i]['time_s'],
                 'observed_lateral_acceleration_mps2':float(observed[i]),
                 'native_measured_force_predicted_ay_mps2':float(native_based[i]),
                 'planar_CasADi_predicted_ay_mps2':float(b[i]),
                 'full_contact_CasADi_predicted_ay_mps2':float(a[i])}
                for i in range(len(records))]
    return metrics,m_ay,wheel_records,per_sample


def write_csv(path: Path, rows: list[dict]):
    if not rows:raise ValueError('Nothing to save')
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo-root',type=Path,default=Path('..'))
    p.add_argument('--native-dir',type=Path,default=Path('native_slip_v2314'))
    p.add_argument('--seed',type=Path,default=Path('optimized_practice_v2_3_0/warmstart_physx/native_warmstart.npz'))
    p.add_argument('--output',type=Path,default=Path('native_contact_v2316'))
    p.add_argument('--allow-history-fetch',action='store_true')
    a=p.parse_args();src=a.native_dir.resolve()
    trial=json.loads((src/'summary.json').read_text())['native_flying_lap']
    if not trial['completed_flying_lap'] or trial.get('failure'):
        raise RuntimeError('Refusing incomplete/unsafe native contact capture')
    with (src/'per_wheel.csv').open(newline='') as f:
        records=reconstruct_records(list(csv.DictReader(f)))
    root=a.repo_root.resolve();pkg=Path(__file__).resolve().parent
    solver,wheel,track_module,_,provenance=load_modules(root,pkg,a.allow_history_fetch,'limited-memory')
    if not provenance.get('native_full_contact_velocity_corrected'):
        raise RuntimeError('Generated working CasADi wheel model did not upgrade to v2.3.16')
    model,_=current_model(root,wheel)
    bind_current_wheel_moment_arms(root,solver,wheel)
    # Source-derived geometry: contact force-application points exactly one
    # wheel radius below wheel mount relative to Unity Rigidbody COM.
    arm=np.array([wheel.RIGIDBODY_CENTER_OF_MASS_BODY_M[1]+model.wheel_radius_m
                  -v[1] for v in model.suspension_base_xyz_body_m])
    expected=np.array([0.0634351388,0.0634351388,0.07334,0.07334])
    if arm.shape!=(4,) or np.max(abs(arm-expected))>0.00002:
        raise RuntimeError('Unexpected native tire contact application arm geometry: '+repr(arm))
    track=track_module.load_centerline_csv(
        root/'f1tenth_planning/trajectories/autodrive_practice_20260924_b/autodrive_practice_20260924_b_mintime_raceline.csv',.20)
    with np.load(a.seed.resolve()) as npz:
        seed={k:np.asarray(npz[k]) for k in npz.files}
    rr,pr,alignment=interpolate_native_body_rates(seed,records,track.length)
    fun=build_actual_contact_audit(wheel,model)
    metrics,accel,per_wheel,per_sample=audit_contact(records,rr,pr,fun,model)
    if metrics['lateral_slip']['full_contact']['rms']>=0.3*metrics['lateral_slip']['planar']['rms']:
        raise RuntimeError('Full contact model did not improve measured native lateral slip at least 3x')
    if metrics['tire_lateral_force_n']['full_contact']['rms']>=0.3*metrics['tire_lateral_force_n']['planar']['rms']:
        raise RuntimeError('Full contact model did not improve measured tire lateral force at least 3x')
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    summary={'diagnostic':'v2316_actual_CasADi_full_contact_angular_velocity_parity',
             'flying_lap_completed':True,'native_sample_count':len(records),
             'wheel_sample_count':len(per_wheel),
             'native_sampling_s':0.025,'saved_seed_nodes':len(seed['u_mps']),
             'source_contact_arm_down_m':arm.tolist(),
             'saved_seed_phase_alignment':alignment,
             'per_wheel_error_metrics':metrics,
             'native_vs_measured_ay_mps2':accel,
             'working_wheel_model_sha256':provenance['working_wheel_model_sha256'],
             'working_solver_sha256':provenance['working_solver_sha256'],
             'historical_source_untouched':provenance['historical_wheel_model_untouched'],
             'actual_generated_solver_rate_argument_checked':
                 'roll_rate[k], pitch_rate[k]' in __import__('inspect').getsource(solver),
             'caveat':'Pre-scene 1ms PhysX tire slip and post-scene body/omega timestamps are staggered; 143-point roll/pitch-rate interpolation by progress is approximate. 25ms body acceleration derivative includes scene contact solver. Strict feasible IPOPT and Unity parity unverified.'}
    if not summary['actual_generated_solver_rate_argument_checked']:
        raise RuntimeError('Actual generated OCP did not receive roll and pitch rates')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    write_csv(out/'per_wheel.csv',per_wheel)
    write_csv(out/'per_sample.csv',per_sample)
    print(json.dumps(summary,indent=2),flush=True)
    print('No IPOPT or native replay executed.',flush=True)

if __name__=='__main__':main()
