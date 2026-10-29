#!/usr/bin/env python3
"""Evaluate a suspected Unity-x/right versus OCP-y/left suspension-sign error.

This is an audit, not a model change. It does not fit forces, run IPOPT,
launch PhysX, or create a racing trajectory. Candidate transforms are evaluated
on the same 143-node saved native warm start.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from .diagnose_force_consistency_v238 import evaluate_scalar_tire_forces
from recover import load_modules
from seed_projection import load_native_seed, validate_seed_provenance
from source_vehicle import bind_current_wheel_moment_arms, current_model

WHEELS=('fl','fr','rl','rr')


def lateral_transfer(loads: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    """Left minus right in original native FL, FR, RL, RR order."""
    return loads[0]-loads[1],loads[2]-loads[3]


def evaluate_loads(seed: dict, model, wheel) -> np.ndarray:
    return np.stack([np.asarray(wheel.suspension_loads_numpy(
        model,seed['body_height_m'][k],seed['body_height_rate_mps'][k],
        seed['roll_rad'][k],seed['roll_rate_radps'][k],
        seed['pitch_rad'][k],seed['pitch_rate_radps'][k]),dtype=float)
        for k in range(len(seed['u_mps']))],axis=1)


def load_metrics(native: np.ndarray, calculated: np.ndarray) -> dict:
    native=np.asarray(native,float);calculated=np.asarray(calculated,float)
    if native.shape!=calculated.shape or native.shape[0]!=4:
        raise ValueError('Expected matching FL/FR/RL/RR wheel load arrays')
    if not np.isfinite(native).all() or not np.isfinite(calculated).all():
        raise ValueError('Non-finite load comparison')
    delta=calculated-native
    nfront,nrear=lateral_transfer(native)
    sfront,srear=lateral_transfer(calculated)
    def transfer_metrics(n,s):
        return {'correlation':float(np.corrcoef(n,s)[0,1]) if np.std(n)>1e-12 and np.std(s)>1e-12 else None,
                'sign_mismatch_fraction':float(np.mean(np.sign(n)!=np.sign(s))),
                'rms_error_n':float(np.sqrt(np.mean((s-n)**2))),
                'native_mean_left_minus_right_n':float(np.mean(n)),
                'symbolic_mean_left_minus_right_n':float(np.mean(s))}
    return {'per_wheel_rms_load_error_n':{w:float(np.sqrt(np.mean(delta[i]**2))) for i,w in enumerate(WHEELS)},
            'global_rms_load_error_n':float(np.sqrt(np.mean(delta**2))),
            'global_max_abs_load_error_n':float(np.max(np.abs(delta))),
            'native_total_load_mean_n':float(np.mean(np.sum(native,axis=0))),
            'symbolic_total_load_mean_n':float(np.mean(np.sum(calculated,axis=0))),
            'symbolic_minus_native_total_load_mean_n':float(np.mean(np.sum(delta,axis=0))),
            'front_left_minus_right':transfer_metrics(nfront,sfront),
            'rear_left_minus_right':transfer_metrics(nrear,srear)}


def apply_sign_options(seed: dict, *, flip_roll=False, flip_pitch=False) -> dict:
    result=dict(seed)
    for field in ('roll_rad','roll_rate_radps') if flip_roll else ():
        result[field]=-seed[field]
    for field in ('pitch_rad','pitch_rate_radps') if flip_pitch else ():
        result[field]=-seed[field]
    return result


def mirrored_lateral_geometry(model):
    """Test converting wheel positions from Unity x-right to model y-left.

    The historical suspension expression uses sin(roll)*wheel_x. The native
    model roll is expressed in forward/left/up coordinates, while the prefab
    wheel x is right-positive. Flip ONLY that lateral mount coordinate.
    All other model parameters, equations and wheel indexes are identical.
    """
    base=tuple((-float(p[0]),float(p[1]),float(p[2]))
               for p in model.suspension_base_xyz_body_m)
    return replace(model,suspension_base_xyz_body_m=base)


def compare_candidates(seed: dict, model, wheel):
    # v2.3.10 loads the source model with corrected LEFT-positive lateral
    # mounts. Retain the historical candidate labels to make v2.3.9 output
    # directly comparable: the old mixed-frame baseline is reflected back.
    first_x=float(model.suspension_base_xyz_body_m[0][0])
    second_x=float(model.suspension_base_xyz_body_m[1][0])
    if first_x>0 and second_x<0:  # v2.3.10 model, LEFT positive
        baseline_model=mirrored_lateral_geometry(model)
    elif first_x<0 and second_x>0:  # v2.3.9 model, Unity RIGHT positive
        baseline_model=model
    else:
        raise ValueError('Unexpected lateral mount order: cannot audit safely')
    mirror=mirrored_lateral_geometry(baseline_model)
    original=evaluate_loads(seed,baseline_model,wheel)
    candidates={'original_unmodified':original,
                'mirrored_lateral_mount_x':evaluate_loads(seed,mirror,wheel),
                'reversed_roll_and_roll_rate':evaluate_loads(
                    apply_sign_options(seed,flip_roll=True),baseline_model,wheel),
                'reversed_pitch_and_pitch_rate':evaluate_loads(
                    apply_sign_options(seed,flip_pitch=True),baseline_model,wheel),
                'reversed_roll_and_pitch':evaluate_loads(
                    apply_sign_options(seed,flip_roll=True,flip_pitch=True),baseline_model,wheel)}
    equiv=float(np.max(np.abs(candidates['mirrored_lateral_mount_x']-
                              candidates['reversed_roll_and_roll_rate'])))
    if equiv>1e-8:
        raise AssertionError('Geometry-reflection and native-roll reflection do not agree')
    native=seed['normal_load_n']; report={}
    for name,loads in candidates.items():
        if np.min(loads)<=0:
            report[name]={'error':'nonpositive_load', 'minimum_load_n':float(np.min(loads))}
            continue
        _,fy,*_=evaluate_scalar_tire_forces(model,wheel,seed,loads)
        ay=fy/model.mass_kg-model.drag_per_s*seed['v_mps']
        m=load_metrics(native,loads)
        m['symbolic_tire_ay_vs_native_motion_max_abs_mps2']=float(np.max(np.abs(ay-seed['ay_mps2'])))
        m['symbolic_tire_ay_vs_native_motion_rms_mps2']=float(np.sqrt(np.mean((ay-seed['ay_mps2'])**2)))
        report[name]=m
    result={
        'diagnostic':'source_wheel_suspension_coordinate_frame_candidate_audit',
        'nodes':len(seed['u_mps']),
        'frame_question':'Unity wheel mount x is right-positive; native source-state roll uses forward/left/up axes. Does the symbolic flat-plane travel expression mix those bases?',
        'geometry_reflection_equals_sign_reversed_native_roll_max_error_n':equiv,
        'candidate_results':report,
        'no_physics_equation_changed':True,
        'no_ipopt_or_native_replay':True,
        'limitation':'Changes to load algebraics may not account for native tire slip, constraint impulses, transient jounce, or reconstructed accelerations. These are diagnostic candidates, NOT fitted physics.',
    }
    rows=[]
    for k in range(result['nodes']):
        row={'k':k, 'native_roll_rad':float(seed['roll_rad'][k]),
             'native_pitch_rad':float(seed['pitch_rad'][k]),
             'native_yaw_rate_radps':float(seed['yaw_rate_radps'][k]),
             'native_ay_mps2':float(seed['ay_mps2'][k])}
        for i,w in enumerate(WHEELS):
            row[f'{w}_native_load_n']=float(native[i,k])
            row[f'{w}_original_load_n']=float(original[i,k])
            row[f'{w}_mirrored_load_n']=float(candidates['mirrored_lateral_mount_x'][i,k])
        rows.append(row)
    return result,rows


def run():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root',type=Path,default=Path('..'))
    parser.add_argument('--seed-file',type=Path,default=Path('optimized_practice_v2_3_0/warmstart_physx/native_warmstart.npz'))
    parser.add_argument('--output',type=Path,default=Path('suspension_frame_v239'))
    parser.add_argument('--allow-history-fetch',action='store_true')
    args=parser.parse_args()
    root=args.repo_root.resolve();here=Path(__file__).resolve().parent
    seed_path=args.seed_file.resolve()
    mapfile=root/'f1tenth_planning/maps/autodrive_practice_20260924_b.pgm'
    center=root/'f1tenth_planning/trajectories/autodrive_practice_20260924_b/autodrive_practice_20260924_b_mintime_raceline.csv'
    checksums=validate_seed_provenance(seed_path,mapfile,center)
    old,wheel,trackmod,_,provenance=load_modules(root,here,args.allow_history_fetch,'limited-memory')
    model,_=current_model(root,wheel)
    bind_current_wheel_moment_arms(root,old,wheel)
    track=trackmod.load_centerline_csv(center,.20)
    seed=load_native_seed(seed_path,track.count)
    result,rows=compare_candidates(seed,model,wheel)
    # Guard that the v2.3.10 correction is actually active in the loaded
    # source model, not merely a hypothetical mirrored diagnostic candidate.
    result['source_model_mounts_model_left_positive']=bool(
        model.suspension_base_xyz_body_m[0][0]>0>model.suspension_base_xyz_body_m[1][0]
        and model.suspension_base_xyz_body_m[2][0]>0>model.suspension_base_xyz_body_m[3][0])
    result['source_model_mount_lateral_arms_m']=[float(p[0]) for p in model.suspension_base_xyz_body_m]
    result['map_provenance']=checksums
    result['working_solver_sha256']=provenance['working_solver_sha256']
    destination=args.output.resolve();destination.mkdir(parents=True,exist_ok=True)
    (destination/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    with (destination/'per_node.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print('Source suspension frame audit (NO IPOPT, NO physics modification, NO PhysX replay)')
    print('Baseline vs mirrored lateral mount x:')
    for name in ('original_unmodified','mirrored_lateral_mount_x',
                 'reversed_pitch_and_pitch_rate','reversed_roll_and_pitch'):
        m=result['candidate_results'][name]
        if 'error' in m:
            print(name,m)
            continue
        print(f'  {name}: load_rms={m["global_rms_load_error_n"]:.6f} N, '
              f'load_max={m["global_max_abs_load_error_n"]:.6f} N, '
              f'front_transfer_corr={m["front_left_minus_right"]["correlation"]}, '
              f'lateral_accel_rms={m["symbolic_tire_ay_vs_native_motion_rms_mps2"]:.6f} m/s²')
    print('Saved',destination/'summary.json','and',destination/'per_node.csv')
    return 0

if __name__=='__main__':
    raise SystemExit(run())
