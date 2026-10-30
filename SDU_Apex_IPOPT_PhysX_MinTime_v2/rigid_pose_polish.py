"""Numerical boundary polishing of an almost source-feasible trajectory.

Small SE(2) transformations preserve 25ms source physics mathematically.
The output is NEVER accepted from the linear program alone: the original
production source simulator and independent 1ms full-footprint verifier must
recertify it. This cannot increase *physical* track clearance significantly
when all contact constraints are active.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

from command_ocp import save_command_checkpoint, save_json_checkpoint
from map_track import MapGrid, _FOOTPRINT_LOCAL, _FOOTPRINT_UNSAMPLED_CORRECTION


def transform_pose(states,translation,rotation,center):
    moved=np.asarray(states,dtype=float).copy()
    angle=float(rotation)
    c,s=math.cos(angle),math.sin(angle)
    delta=moved[:2]-np.asarray(center,dtype=float).reshape(2,1)
    moved[0]=c*delta[0]-s*delta[1]+center[0]+translation[0]
    moved[1]=s*delta[0]+c*delta[1]+center[1]+translation[1]
    moved[2]+=angle
    return moved


def footprint_points(states):
    yaw=states[2]
    c,s=np.cos(yaw)[:,None],np.sin(yaw)[:,None]
    return np.stack([states[0,:,None]+c*_FOOTPRINT_LOCAL[:,0]-s*_FOOTPRINT_LOCAL[:,1],
                     states[1,:,None]+s*_FOOTPRINT_LOCAL[:,0]+c*_FOOTPRINT_LOCAL[:,1]],axis=-1)


def rigid_correction(grid,source_trace,margin=.05,critical_window_m=.002,
                     bound_translation_m=.001,bound_rotation_rad=.01):
    """Return an exact-distance-checked rigid pose transform, not a certificate."""
    points=footprint_points(source_trace)
    center=np.mean(source_trace[:2],axis=1)
    raw_slack=grid.distance(points)-_FOOTPRINT_UNSAMPLED_CORRECTION-margin
    current=float(raw_slack.min())
    if not -1e-6 <= current < 0.:
        return None
    active=points[raw_slack<critical_window_m]
    slack=raw_slack[raw_slack<critical_window_m]
    if len(active)==0:
        return None
    eps=1e-4
    gx=(grid.distance(active+np.array([eps,0.]))-
        grid.distance(active-np.array([eps,0.])))/(2*eps)
    gy=(grid.distance(active+np.array([0.,eps]))-
        grid.distance(active-np.array([0.,eps])))/(2*eps)
    relative=active-center
    grot=-gx*relative[:,1]+gy*relative[:,0]
    result=linprog([0.,0.,0.,-1.],A_ub=np.column_stack([
        -gx,-gy,-grot,np.ones(len(slack))]),b_ub=slack,
        bounds=[(-bound_translation_m,bound_translation_m)]*2+
               [(-bound_rotation_rad,bound_rotation_rad),(None,None)],
        method='highs')
    if not result.success:
        return None
    def measured(shift):
        dx,dy,theta=shift
        c,s=np.cos(theta),np.sin(theta)
        delta=points-center
        changed=np.empty_like(points)
        changed[...,0]=c*delta[...,0]-s*delta[...,1]+center[0]+dx
        changed[...,1]=s*delta[...,0]+c*delta[...,1]+center[1]+dy
        return float(np.min(grid.distance(changed)-
                            _FOOTPRINT_UNSAMPLED_CORRECTION-margin))
    candidates=[(measured(result.x[:3]*alpha),result.x[:3]*alpha)
                for alpha in (.125,.25,.5,.75,1.)]
    positive=[item for item in candidates if item[0]>0.]
    if not positive:
        return None
    score,shift=max(positive,key=lambda item:item[0])
    return dict(shift_xy_m=shift[:2].tolist(),rotation_rad=float(shift[2]),
                initial_slack_m=current,shifted_slack_m=score,
                numerical_only=score<1e-6,
                active_probe_count=len(slack))


def recover_numerical_near_miss(repo,seed,map_path,output,*,margin=.05,
                                certify=None):
    """Attempt only nanometre/micrometre-scale near-misses, never a wall collision."""
    from command_ocp import certify_witness
    repo,seed,map_path,output=map(Path,(repo,seed,map_path,output))
    report_path=seed.with_name('report.json')
    trace_path=seed.with_name('source_trace.npz')
    if not (report_path.exists() and trace_path.exists()):
        return None
    report=json.loads(report_path.read_text())
    rollout=report.get('rollout') or {}
    slack=rollout.get('minimum_1ms_footprint_slack_m')
    if (report.get('status')!='no_validated_witness' or
            slack is None or not -1e-6<=float(slack)<0 or
            float(report.get('max_constraint_violation',math.inf))>1e-5 or
            float(rollout.get('full_lap_xy_max_m',math.inf))>1e-4):
        return None
    with np.load(seed,allow_pickle=False) as original:
        if str(original['schema'].item())!='production-command-ocp/v1':
            return None
        x=np.asarray(original['state'])
        controls=np.asarray(original['controls'])
        digest=str(original['source_sha256'].item())
    with np.load(trace_path,allow_pickle=False) as saved:
        traced=np.asarray(saved['state'])
        if (traced.shape!=(21,controls.shape[1]*25+1) or
                not np.array_equal(saved['commands'],controls)):
            return None
    grid=MapGrid.load(map_path)
    correction=rigid_correction(grid,traced,margin=margin)
    if correction is None:
        return None
    candidate=transform_pose(x,correction['shift_xy_m'],
                              correction['rotation_rad'],
                              np.mean(traced[:2],axis=1))
    output.mkdir(parents=True,exist_ok=False)
    proposal=output/'proposal.npz'
    save_command_checkpoint(proposal,candidate,controls,digest,
                            float(report['max_constraint_violation']),-1)
    certified_dir=output/'certified'
    verified=(certify or certify_witness)(repo,proposal,map_path,certified_dir,
                                         margin=margin)
    accepted=verified['status']=='source_witness_passed'
    record=dict(status='source_witness_passed' if accepted else 'not_certified',
                candidate=str(proposal),certificate=str(certified_dir/'report.json'),
                witness=str(certified_dir/'witness.npz') if accepted else None,
                original_report=str(report_path),source_only=True,
                result_lap_s=verified.get('trial_lap_s'),**correction)
    save_json_checkpoint(output/'recovery.json',record)
    return record if accepted else None
