"""Physics-propagated alternate steering-command basins for exact source IPOPT.

All 21 plant state variables are propagated through the production 25 ms map.
A separate distributed periodic projection supplies a numerically suitable
*initial guess*; it is not a plant rollout, feasible witness or certificate.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import numpy as np


def control_profile(count:int, amplitude_rad:float, harmonic:int,
                    phase_rad:float=0.) -> np.ndarray:
    if (count < 30 or not math.isfinite(amplitude_rad) or
            not 0. < abs(amplitude_rad) <= .08 or not 1 <= harmonic <= 8 or
            not math.isfinite(phase_rad)):
        raise ValueError('invalid steering perturbation')
    index=(np.arange(count,dtype=float)+.5)/count
    return amplitude_rad*np.sin(2*math.pi*harmonic*index+phase_rad)


def propagate_source(function, initial, held):
    """Exact 25ms discrete source map; no interpolated dynamics."""
    count=held.shape[1]
    states=np.empty((21,count+1))
    states[:,0]=initial
    for k in range(count):
        states[:,k+1]=np.asarray(
            function(np.r_[states[:,k],held[:,k]]),dtype=float).reshape(-1)
    if not np.isfinite(states).all():
        raise ValueError('nonfinite source rollout')
    return states


def project_periodic_guess(states, winding:int):
    """Distribute periodic endpoint mismatch over the horizon (UNCERTIFIED)."""
    count=states.shape[1]-1
    if count<30 or abs(winding)!=1:
        raise ValueError('invalid winding or horizon')
    difference=states[:,-1]-states[:,0]
    difference[2]-=2*math.pi*winding
    projected=states-difference[:,None]*np.linspace(0.,1.,count+1)[None,:]
    projected[2,-1]=projected[2,0]+2*math.pi*winding
    projected[:, -1] = projected[:,0] + np.r_[0.,0.,2*math.pi*winding,np.zeros(18)]
    return projected, difference


def source_alternative_seed(repo,seed,destination,amplitude_rad,harmonic,
                            phase_rad=0.,settling_laps=3, map_path=None,
                            max_initial_clearance_deficit_m=.01):
    """Legacy global harmonic perturbation with propagated source state."""
    from command_campaign import source_count
    return source_profile_seed(
        repo,seed,destination,
        control_profile(source_count(Path(seed)),amplitude_rad,harmonic,phase_rad),
        settling_laps=settling_laps,map_path=map_path,
        max_initial_clearance_deficit_m=max_initial_clearance_deficit_m)


def source_profile_seed(repo,seed,destination,steering_delta,*,
                        settling_laps=3,map_path=None,
                        max_initial_clearance_deficit_m=.01):
    """Source-propagate steering controls; output remains uncertified."""
    from command_ocp import production_command_map, save_command_checkpoint
    repo,seed,destination=map(Path,(repo,seed,destination))
    if not 1 <= settling_laps <= 12:
        raise ValueError('bad source-history settling count')
    function,digest=production_command_map(repo,repo/'build_task/command_ocp')
    with np.load(seed,allow_pickle=False) as stored:
        if (str(stored['schema'].item())!='production-command-ocp/v1' or
                str(stored['source_sha256'].item())!=digest):
            raise ValueError('source seed is not compatible with current plant')
        original=np.asarray(stored['state'],dtype=float)
        controls=np.asarray(stored['controls'],dtype=float).copy()
    count=controls.shape[1]
    winding=round((original[2,-1]-original[2,0])/(2*math.pi))
    profile=np.asarray(steering_delta,dtype=float).reshape(-1)
    if (profile.shape!=(count,) or not np.isfinite(profile).all() or
            np.max(np.abs(profile))>.0800001):
        raise ValueError('invalid source-profile perturbation')
    controls[0] = np.clip(controls[0]+profile,-.5235987755982988,.5235987755982988)
    initial=original[:,0].copy()
    for _ in range(settling_laps):
        cycle=propagate_source(function,initial,controls)
        initial[3:]=cycle[3:,-1] # settle hidden memory, speeds and steering rack
    raw=propagate_source(function,initial,controls)
    guess,difference=project_periodic_guess(raw,winding)
    # Constraint residual here is informative only. Never call this certified.
    from command_ocp import STATE_SCALES
    z=np.vstack([guess[:,:-1],controls])
    mapped=np.asarray(function.map(count,'serial')(z))
    defect=np.max(np.abs((mapped-guess[:,1:])/STATE_SCALES[:,None]))
    if not np.isfinite(defect):
        raise ValueError('invalid source-derived initializer')
    # Preflight avoids spending 1200 IPOPT iterations on a seed that already
    # leaves the physical track corridor by centimetres. This is an initializer
    # rejection only; successful candidates still need independent 1ms checks.
    preflight_slack=None
    if map_path is not None:
        from map_track import MapGrid
        grid=MapGrid.load(Path(map_path))
        preflight_slack=float(min(grid.footprint_clearance(
            *guess[:3,k],margin=.05) for k in range(count+1)))
        if preflight_slack < -max_initial_clearance_deficit_m:
            return dict(status='rejected_initializer',path=None,
                        initial_scaled_dynamic_defect=float(defect),
                        initial_minimum_node_slack_m=preflight_slack,
                        source_certified=False)
    destination.parent.mkdir(parents=True,exist_ok=True)
    save_command_checkpoint(destination,guess,controls,digest,float(defect),-1)
    return dict(path=str(destination),source_sha256=digest,
                initial_scaled_dynamic_defect=float(defect),
                raw_rollout_closure_xy_m=float(np.linalg.norm(difference[:2])),
                raw_rollout_closure_yaw_rad=float(difference[2]),
                raw_rollout_hidden_closure_scaled=float(
                    np.max(np.abs(difference[3:]/STATE_SCALES[3:]))),
                initial_state_hash=hashlib.sha256(initial.tobytes()).hexdigest(),
                initial_minimum_node_slack_m=preflight_slack,
                source_certified=False)
