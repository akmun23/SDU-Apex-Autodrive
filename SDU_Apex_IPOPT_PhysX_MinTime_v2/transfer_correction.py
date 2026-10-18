"""PhysX-in-the-loop map-geometry correction for restored IPOPT solver.

Only a source-identified *tracking residual* is transferred to the IPOPT
corridor constraints. This is NOT claimed to correct all tire/contact dynamics.
"""
from __future__ import annotations
import csv
import math
from pathlib import Path
import numpy as np


def native_lateral_bias(trace:Path,candidate,track,*,max_bias_m=.20,
                        support_radius_m=.55):
    s=np.asarray(track.s,float)
    l=float(track.length)
    opt_s=np.asarray(candidate['s_ref_m'],float)
    cx=np.asarray(candidate['x_m'],float)
    cy=np.asarray(candidate['y_m'],float)
    observations=[]
    with Path(trace).open(newline='') as f:
        for row in csv.DictReader(f):
            station=float(row['progress_m'])
            if not (0<=station<=l): continue
            px=np.interp(station,np.r_[opt_s,l],np.r_[cx,cx[0]])
            py=np.interp(station,np.r_[opt_s,l],np.r_[cy,cy[0]])
            psi=float(np.interp(station,np.r_[s,l],np.r_[track.psi,track.psi[0]]))
            dx=float(row['x_m'])-px;dy=float(row['y_m'])-py
            delta=-math.sin(psi)*dx+math.cos(psi)*dy
            if math.isfinite(delta) and abs(delta)<1.0:
                observations.append((station,delta))
    if len(observations)<20:
        return np.zeros_like(s),{'status':'too_few_native_samples','samples':len(observations)}
    obs=np.asarray(observations,float)
    # Resample into periodic local bins. Do not extrapolate a failed sector's
    # residual across the unseen remainder of the course.
    bias=np.zeros_like(s)
    supported=np.zeros_like(s,dtype=bool)
    for i,station in enumerate(s):
        d=np.abs(obs[:,0]-station)
        d=np.minimum(d,l-d)
        valid=d<support_radius_m
        if not np.any(valid):continue
        w=np.exp(-.5*(d[valid]/(support_radius_m*.4))**2)
        bias[i]=np.clip(float(np.average(obs[valid,1],weights=w)),-max_bias_m,max_bias_m)
        supported[i]=True
    return bias,{
        'status':'computed_from_native_open_loop_tracking_difference',
        'samples':len(observations),
        'supported_nodes':int(supported.sum()),
        'max_abs_bias_m':float(np.max(np.abs(bias))),
        'rms_bias_m':float(np.sqrt(np.mean(bias*bias))),
        'note':'bias shifts IPOPT footprint constraints; does not reproduce native PhysX dynamics or certify Unity.'}


def candidate_warm_state(candidate):
    renaming={
        'u_mps':'speed_mps','v_mps':'lateral_speed_mps',
        'ey_m':'ey_m','epsi_rad':'epsi_rad','yaw_rate_radps':'yaw_rate_radps',
        'wheel_omega_radps':'wheel_omega_radps',
        'steering_rad':'steering_rad',
        'steering_rate_radps':'steering_rate_radps',
        'throttle_norm':'throttle_norm','ax_mps2':'ax_mps2',
        'ay_mps2':'ay_mps2','normal_load_n':'normal_load_n',
        'body_height_m':'body_height_m',
        'body_height_rate_mps':'body_height_rate_mps',
        'roll_rad':'roll_rad','roll_rate_radps':'roll_rate_radps',
        'pitch_rad':'pitch_rad','pitch_rate_radps':'pitch_rate_radps',
    }
    return {new:np.asarray(candidate[old],float) for new,old in renaming.items()}
