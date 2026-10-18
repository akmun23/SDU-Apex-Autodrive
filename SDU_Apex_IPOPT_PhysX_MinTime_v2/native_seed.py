"""Use actual native-PhysX wheel/body transitions to seed the recovered IPOPT OCP.

A steering/speed tracking controller is ONLY used to collect a safe, complete
flying-lap *initial guess*. It is never substituted for IPOPT controls in the
optimizer's native-PhysX acceptance replay. The record is not claimed to be
periodic or to satisfy the *symbolic* model exactly.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from physics import NativePhysX, SourcePlant
from native_validate import wrap
from map_track import _FOOTPRINT_RADIUS

DT = .025
L = .324
DEFAULT_SPEED_TRIALS = (2.4, 1.8, 1.3)


def command(ref, station, ey, yaw, v, speed, target_speed):
    # Curvature feedforward + cross-track, heading and sideslip feedback.
    _, _, heading, kappa = ref.at(station)
    err = wrap(yaw - heading)
    steer = math.atan(L * kappa) - 1.6 * ey - .8 * err - .08 * v
    steer = float(np.clip(steer, -math.pi/6, math.pi/6))
    # Actual Unity has 428 Nm CAWD motor torque and 0.25 Nms wheel damping.
    # A strictly positive control avoids accidentally requesting CAWB braking.
    equilibrium = 4 * .25 * max(.1, speed) / (.059 * 428.)
    throttle = float(np.clip(equilibrium + .035 * (target_speed-speed), .015, .20))
    return steer, throttle


def build_seed(trace, track, ref_length):
    """Resample a measured flying lap to IPOPT station nodes, preserving states.

    Each input is (absolute_progress, time, x, y, yaw, throttle, 18-state,
    4 normal loads). `absolute_progress` is unwrapped via map matching.
    """
    if len(trace) < 20:
        raise ValueError('A physics seed needs at least 20 flying-lap steps')
    a = sorted(trace, key=lambda rec: rec[0])
    stations = np.asarray([r[0] for r in a], float)
    if np.any(~np.isfinite(stations)) or np.any(np.diff(stations) <= 1e-8):
        raise ValueError('Physics seed progress is not strictly increasing')
    if stations[-1]-stations[0] < .90 * ref_length:
        raise ValueError('Physics seed does not cover a complete lap')
    # Convert measured station units to the slightly different OCP spline length.
    s = (stations - ref_length) * (track.length/ref_length)
    original_times=np.asarray([r[1] for r in a],float)
    original_yaw=np.asarray([r[4] for r in a],float)
    if np.any(np.diff(original_times)<=0):
        raise ValueError('Measured physics time must increase monotonically')
    source_sdot=np.gradient(s,original_times)
    if np.any(source_sdot<=0):
        raise ValueError('Measured physics lap has zero or reverse progress')
    # Orientations have a 2*pi Winding jump across one completed lap.
    # Interpolating yaw as a periodic scalar created a fake 1+ rad heading
    # error at the gate. Interpolate sin/cos of *heading error* instead.
    road_cos=np.interp(np.mod(s,track.length),np.r_[track.s,track.length],
                       np.r_[np.cos(track.psi),np.cos(track.psi[0])])
    road_sin=np.interp(np.mod(s,track.length),np.r_[track.s,track.length],
                       np.r_[np.sin(track.psi),np.sin(track.psi[0])])
    measured_epsi=np.arctan2(np.sin(original_yaw-np.arctan2(road_sin,road_cos)),
                             np.cos(original_yaw-np.arctan2(road_sin,road_cos)))
    columns = np.array([[r[1],r[2],r[3],r[4],r[5],*r[6],*r[7],
                         source_sdot[i],np.sin(measured_epsi[i]),
                         np.cos(measured_epsi[i])] for i,r in enumerate(a)],float)
    if columns.shape[1] != 5+18+4+3 or not np.isfinite(columns).all():
        raise ValueError('Invalid or incomplete native wheel/body state')
    # Periodic interpolation tolerates the discretization's small gate gaps.
    # Wrap the actual second flying lap at the gate, not the rest-start lap.
    valid = (s >= -.1 * track.length) & (s <= 1.1*track.length)
    s, columns = s[valid], columns[valid]
    if len(s)<20:
        raise ValueError('Physics seed has insufficient near-lap samples')
    si = np.r_[s[-1]-track.length, s, s[0]+track.length]
    vals = np.vstack([columns[-1],columns,columns[0]])
    if np.any(np.diff(si)<=0):
        raise ValueError('Physics seed contains ambiguous or duplicated laps')
    t = np.asarray(track.s,float)
    sampled = np.column_stack([np.interp(t,si,vals[:,i]) for i in range(vals.shape[1])])
    _,xx,yy,yaw,throttle = (sampled[:,i] for i in range(5))
    plant=sampled[:,5:23]; loads=sampled[:,23:27].T
    nx=-np.sin(track.psi); ny=np.cos(track.psi)
    ey=(xx-np.asarray(track.x))*nx+(yy-np.asarray(track.y))*ny
    epsi=np.arctan2(sampled[:,-2],sampled[:,-1])
    u,v,r=plant[:,0],plant[:,1],plant[:,2]
    steer=plant[:,13]
    station_rate=np.maximum(sampled[:,27],.3)
    # Measured acceleration is used only as CasADi initialization. Reconstruct
    # its derivative from ds/dt instead of a discontinuous wrapped lap time.
    du_dt=np.gradient(u,t,edge_order=1)*station_rate
    dv_dt=np.gradient(v,t,edge_order=1)*station_rate
    dsteer_dt=np.gradient(steer,t,edge_order=1)*station_rate
    seed={
        'ey_m':ey,'epsi_rad':epsi,'u_mps':np.maximum(u,.36),
        'v_mps':v,'yaw_rate_radps':r,
        'wheel_omega_radps':np.maximum(plant[:,3:7].T,0.),
        'steering_rad':np.clip(steer,-math.pi/6,math.pi/6),
        'steering_rate_radps':np.clip(dsteer_dt,-3.2,3.2),
        'throttle_norm':np.clip(throttle,.015,1.),
        'ax_mps2':du_dt-r*v,'ay_mps2':dv_dt+r*u,
        'normal_load_n':np.maximum(loads,.05),
        'body_height_m':plant[:,7],
        'body_height_rate_mps':plant[:,8],
        'roll_rad':plant[:,9],
        'roll_rate_radps':plant[:,10],
        'pitch_rad':plant[:,11],
        'pitch_rate_radps':plant[:,12],
    }
    for key,val in seed.items():
        if not np.isfinite(val).all():raise ValueError(f'Nonfinite PhysX seed variable: {key}')
    return seed


def _get_state(phys, state):
    if isinstance(phys, NativePhysX):
        arr=getattr(phys,'last_source_state',None)
    else:
        arr=list(phys.input.state) if isinstance(phys,SourcePlant) else None
    if arr is None or len(arr)!=18:
        raise RuntimeError('Native PhysX protocol did not expose the 18-state vehicle vector')
    return np.asarray(arr,dtype=float)


def one_trial(repo, binary, backend, ref, grid, track, output, target_speed,
              wall_margin, max_seconds=65., contact_records=None):
    factory=(lambda:NativePhysX(repo,binary)) if backend=='native' else (
             lambda:SourcePlant(repo,output/'source_plant_build'))
    phys=factory(); records=[]; samples=[]
    x0,y0,heading,_=ref.at(0.)
    previous=None; progress=0.; failure='time_limit'; last_clearance=float('inf')
    try:
        previous=phys.reset(x0,y0,heading,target_speed)
        last_clearance=grid.footprint_clearance(x0,y0,heading,margin=wall_margin)
        if last_clearance<0:raise ValueError('Warm-start position violates the track footprint')
        last_abs=0.
        for tick in range(math.ceil(max_seconds/DT)):
            loc=ref.project(previous.x,previous.y,progress)
            if loc is None:
                failure='lost_reference_projection';break
            station,ey,_=loc
            if station<progress-.15:
                failure='reverse_progress';break
            progress=max(progress,station)
            current_target=min(target_speed, math.sqrt(5.0 / max(.05,abs(ref.at(progress)[3]))))
            steering,throttle=command(ref,progress,ey,previous.yaw,
                                       previous.v,previous.u,current_target)
            nxt=phys.step(steering,throttle)
            xy=ref.project(nxt.x,nxt.y,progress)
            if xy is None:
                failure='lost_reference_projection';break
            new_progress=xy[0]
            if new_progress<progress-.15:
                failure='reverse_progress';break
            progress=max(progress,new_progress)
            if (not np.isfinite([nxt.x,nxt.y,nxt.yaw,nxt.u,nxt.v,nxt.r]).all()
                    or nxt.u<.25):
                failure='invalid_vehicle_state';break
            clear=grid.footprint_clearance(nxt.x,nxt.y,nxt.yaw,margin=wall_margin)
            # Swept perimeter check: map walls must not be crossed between
            # native 25ms control packets, even when endpoints are clear.
            travel=math.hypot(nxt.x-previous.x,nxt.y-previous.y)
            turning=abs(wrap(nxt.yaw-previous.yaw))
            swept=travel+_FOOTPRINT_RADIUS*turning
            if min(clear,last_clearance)<=.5*swept:
                sub=max(2,math.ceil(swept/max(.02,.6*grid.resolution)))
                for j in range(1,sub):
                    fraction=j/sub
                    x=previous.x+fraction*(nxt.x-previous.x)
                    y=previous.y+fraction*(nxt.y-previous.y)
                    heading_step=previous.yaw+fraction*wrap(nxt.yaw-previous.yaw)
                    clear=min(clear,grid.footprint_clearance(x,y,heading_step,
                                                              margin=wall_margin))
                clear-=swept/(2*sub)
            if clear<0:
                failure='swept_footprint_clearance';break
            last_clearance=clear
            if progress>=ref.length and not any(row[1]>=ref.length for row in samples):
                gate_error=math.hypot(nxt.x-x0,nxt.y-y0)
                heading_error=abs(wrap(nxt.yaw-heading))
                if gate_error>.65 or heading_error>.70:
                    failure='first_gate_pose_or_heading';break
            if progress>=2*ref.length-.02:
                gate_error=math.hypot(nxt.x-x0,nxt.y-y0)
                heading_error=abs(wrap(nxt.yaw-heading))
                if gate_error>.65 or heading_error>.70:
                    failure='second_gate_pose_or_heading';break
            raw=_get_state(phys,nxt)
            t=(tick+1)*DT
            samples.append((t,progress,nxt.x,nxt.y,nxt.yaw,nxt.u,ey,throttle,clear))
            if ref.length<=progress<=2*ref.length:
                if progress>last_abs+1e-8:
                    records.append((progress,t,nxt.x,nxt.y,nxt.yaw,throttle,raw.copy(),
                                    np.asarray(nxt.normal_load,float)))
                    if contact_records is not None:
                        observed = getattr(phys, 'last_native_contact', None)
                        if observed is None:
                            raise RuntimeError(
                                'Native binary did not expose per-wheel tire force data')
                        contact_records.append({
                            'time_s': float(t), 'progress_m': float(progress),
                            'u_mps': float(nxt.u), 'v_mps': float(nxt.v),
                            'yaw_rate_radps': float(nxt.r),
                            'steering_rad': float(raw[13]),
                            'throttle_norm': float(throttle),
                            # Source-state wheel speeds sampled together with
                            # the PhysX per-wheel contact query at this control
                            # packet's 25 ms endpoint. These are not asserted
                            # to be the pre-contact 1 ms wheel angular speeds.
                            'wheel_omega_radps': raw[3:7].copy(),
                            **{name: np.asarray(values,dtype=float).copy()
                               for name,values in observed.items()}})
                    last_abs=progress
            previous=nxt
            if progress>=2*ref.length-.02:
                failure=None;break
        output.mkdir(parents=True,exist_ok=True)
        with (output/f'seed_trace_{target_speed:.1f}.csv').open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['time_s','progress_m','x_m','y_m','yaw_rad',
                'u_mps','ey_m','throttle','clearance_beyond_margin_m']);w.writerows(samples)
        result={'backend':backend,'trial_speed_mps':target_speed,
            'progress_m':progress,'elapsed_s':len(samples)*DT,
            'failure':failure,'flying_lap_samples':len(records),
            'min_clearance_beyond_margin_m':min((r[-1] for r in samples),default=None)}
        if failure is None and len(records)>=20:
            try:
                seed=build_seed(records,track,ref.length)
            except ValueError as error:
                result['failure']='seed_resampling: '+str(error)
            else:
                with (output/'native_warmstart.npz').open('wb') as f:
                    np.savez_compressed(f,**seed)
                result['completed_flying_lap']=True
                return seed,result
        result['completed_flying_lap']=False
        return None,result
    finally:
        phys.close()


def generate(repo,binary,backend,ref,grid,track,output,wall_margin,
             speeds=DEFAULT_SPEED_TRIALS):
    """Try moderate-speed PhysX tracking laps; reject unsafe/incomplete data."""
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    results=[]
    for speed in speeds:
        print(f'== Native physical-state warm start: {backend} at {speed:.1f}m/s ==',
              flush=True)
        try:
            seed,report=one_trial(repo,binary,backend,ref,grid,track,output,
                                  float(speed),wall_margin)
        except Exception as exc:
            report={'backend':backend,'trial_speed_mps':float(speed),
                    'failure':type(exc).__name__+': '+str(exc),
                    'completed_flying_lap':False}
            seed=None
        results.append(report)
        print('Warm-start trial:',json.dumps(report,sort_keys=True),flush=True)
        (output/'native_seed_report.json').write_text(json.dumps(results,indent=2)+'\n')
        if seed is not None:
            return seed,results
    return None,results
