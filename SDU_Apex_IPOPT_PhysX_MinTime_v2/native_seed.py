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

from physics import (NativePhysX, SourcePlant, SOURCE_PLANT_STATE_SIZE,
                     SOURCE_PLANT_CORE_STATE_SIZE)
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
    # Differentiate the actual native samples before spatial resampling.
    # Differentiating a coarse target mesh attenuates fast state changes and
    # creates a mesh-dependent acceleration witness.
    native_u=np.asarray([r[6][0] for r in a],dtype=np.float64)
    native_v=np.asarray([r[6][1] for r in a],dtype=np.float64)
    native_du_dt=np.gradient(native_u,original_times,edge_order=1)
    native_dv_dt=np.gradient(native_v,original_times,edge_order=1)
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
                         source_sdot[i],native_du_dt[i],native_dv_dt[i],
                         np.sin(measured_epsi[i]),
                         np.cos(measured_epsi[i])] for i,r in enumerate(a)],float)
    if columns.shape[1] != 5+18+4+5 or not np.isfinite(columns).all():
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
    # The active v251 OCP algebraics are the body-coordinate state derivatives
    # du/dt and dv/dt; its force balances include the rotating-frame terms.
    du_dt=sampled[:,28]
    dv_dt=sampled[:,29]
    dsteer_dt=np.gradient(steer,t,edge_order=1)*station_rate
    seed={
        'ey_m':ey,'epsi_rad':epsi,'u_mps':np.maximum(u,.36),
        'v_mps':v,'yaw_rate_radps':r,
        'wheel_omega_radps':np.maximum(plant[:,3:7].T,0.),
        'steering_rad':np.clip(steer,-math.pi/6,math.pi/6),
        'steering_rate_radps':np.clip(dsteer_dt,-3.2,3.2),
        'throttle_norm':np.clip(throttle,.015,1.),
        'ax_mps2':du_dt,'ay_mps2':dv_dt,
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
        if arr is None or len(arr)!=SOURCE_PLANT_CORE_STATE_SIZE:
            raise RuntimeError(
                'Native PhysX protocol did not expose the 18-state vehicle vector')
        return np.asarray(arr,dtype=float)
    arr=list(phys.input.state) if isinstance(phys,SourcePlant) else None
    if arr is None or len(arr)!=SOURCE_PLANT_STATE_SIZE:
        raise RuntimeError(
            'Plant adapter did not expose the complete 26-entry source state')
    # The OCP has 18 optimized physical state channels. The remaining eight
    # source-plant history values are maintained inside the adapter, not added
    # to the IPOPT seed vector.
    return np.asarray(arr[:SOURCE_PLANT_CORE_STATE_SIZE],dtype=float)


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'Unsupported native diagnostic value: {type(value).__name__}')


def _write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as stream:
        for row in records:
            stream.write(json.dumps(row, default=_json_default,
                                    allow_nan=False, separators=(',', ':')))
            stream.write('\n')


def one_trial(repo, binary, backend, ref, grid, track, output, target_speed,
              wall_margin, max_seconds=65., contact_records=None, state_records=None):
    capture_diagnostics = contact_records is not None or state_records is not None
    factory=(lambda:NativePhysX(repo,binary,capture_diagnostics=capture_diagnostics)) if backend=='native' else (
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
                            'packet_index': int(tick + 1),
                            'time_s': float(t), 'progress_m': float(progress),
                            'sample_phase': 'native_wheel_callback_snapshot',
                            'substep_timestamp_known': False,
                            'u_mps': float(nxt.u), 'v_mps': float(nxt.v),
                            'yaw_rate_radps': float(nxt.r),
                            'steering_rad': float(raw[13]),
                            'steering_command_rad': float(steering),
                            'throttle_norm': float(throttle),
                            # Wheel speed and native contact values are kept in
                            # one record for packet association, but the
                            # producer does not expose an exact 1 ms callback
                            # timestamp. Do not treat them as synchronized
                            # pre/post-contact substep state.
                            'wheel_omega_radps': raw[3:7].copy(),
                            **{name: np.asarray(values,dtype=float).copy()
                               for name,values in observed.items()}})
                    if state_records is not None:
                        # Synchronized 25 ms post-scene native body state;
                        # wheel callback forces were sampled before the scene
                        # step and are NOT falsely labeled instantaneous.
                        state_records.append({
                            'packet_index': int(tick + 1),
                            'time_s': float(t), 'progress_m': float(progress),
                            'sample_phase': 'post_scene_solve_body_state',
                            'native_math_scope': (
                                'returned for the 25ms adapter interval; retain '
                                'each field native shape/semantics'),
                            'steering_rack_rad_by_tick': getattr(
                                phys, 'last_steering_rack_rad_by_tick', None),
                            'wheel_commands_by_tick': getattr(
                                phys, 'last_wheel_commands_by_tick', None),
                            'body_state_18': raw.copy(),
                            'world_xy_m': [float(nxt.x), float(nxt.y)],
                            'world_yaw_rad': float(nxt.yaw),
                            'steering_command_rad': float(steering),
                            'throttle_norm': float(throttle),
                            'native_math':getattr(phys,'last_native_math',None),
                        })
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
            'min_clearance_beyond_margin_m':min((r[-1] for r in samples),default=None),
            'backend_code_hash':getattr(phys,'code_hash',None),
            'backend_binary':str(getattr(phys,'binary','')) or None}
        if failure is None and len(records)>=20:
            try:
                seed=build_seed(records,track,ref.length)
            except ValueError as error:
                result['failure']='seed_resampling: '+str(error)
            else:
                with (output/'native_warmstart.npz').open('wb') as f:
                    np.savez_compressed(
                        f, **seed,
                        acceleration_convention=np.asarray(
                            'body_velocity_derivative_v1'))
                result['completed_flying_lap']=True
                return seed,result
        result['completed_flying_lap']=False
        return None,result
    finally:
        phys.close()


def generate(repo,binary,backend,ref,grid,track,output,wall_margin,
             speeds=DEFAULT_SPEED_TRIALS,capture_native_diagnostics=False):
    """Try moderate-speed PhysX tracking laps; reject unsafe/incomplete data."""
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    if capture_native_diagnostics and backend != 'native':
        raise ValueError('Per-wheel native contact diagnostics require backend=native')
    results=[]
    capture_manifest = {
        'schema_version': 1,
        'backend': backend,
        'control_packet_period_s': DT,
        'recording': 'opt-in; one record per increasing-progress packet in the flying lap',
        'commands': [
            'steering_command_rad', 'throttle_norm',
            'steering_rack_rad_by_tick', 'wheel_commands_by_tick '
            '(steer angle, drive torque, brake torque for each wheel and 1ms tick)'],
        'contact_record_fields': [
            'tire_longitudinal_force_n', 'tire_lateral_force_n',
            'normal_load_n', 'longitudinal_slip', 'lateral_slip',
            'wheel_omega_radps', 'u_mps', 'v_mps', 'yaw_rate_radps',
            'steering_rad', 'steering_command_rad', 'throttle_norm'],
        'state_record_fields': [
            'body_state_18', 'world_xy_m', 'world_yaw_rad',
            'steering_command_rad', 'throttle_norm',
            'steering_rack_rad_by_tick', 'wheel_commands_by_tick',
            'native_math'],
        'time_alignment_limits': (
            'Records share adapter packet index/progress. The native contact '
            'producer does not expose its exact 1 ms substep timestamp; contact '
            'and post-scene body state must not be treated as simultaneous.'),
        'native_math_semantics': (
            'Preserve native field names, units, shapes, and solver-diagnostic '
            'structure; do not infer a substep timestamp that the protocol omits.'),
        'files': [],
    }
    for speed in speeds:
        print(f'== Native physical-state warm start: {backend} at {speed:.1f}m/s ==',
              flush=True)
        contacts=[] if capture_native_diagnostics else None
        states=[] if capture_native_diagnostics else None
        try:
            seed,report=one_trial(repo,binary,backend,ref,grid,track,output,
                                  float(speed),wall_margin,
                                  contact_records=contacts,
                                  state_records=states)
        except Exception as exc:
            report={'backend':backend,'trial_speed_mps':float(speed),
                    'failure':type(exc).__name__+': '+str(exc),
                    'completed_flying_lap':False}
            seed=None
        if capture_native_diagnostics:
            contact_path = output/f'native_contact_{speed:.1f}.jsonl'
            state_path = output/f'native_state_{speed:.1f}.jsonl'
            _write_jsonl(contact_path,contacts)
            _write_jsonl(state_path,states)
            capture_manifest['files'].append({
                'trial_speed_mps': float(speed),
                'contact_path': contact_path.name,
                'contact_records': len(contacts),
                'state_path': state_path.name,
                'state_records': len(states),
                'backend_code_hash': report.get('backend_code_hash'),
                'backend_binary': report.get('backend_binary'),
            })
            (output/'native_diagnostics_manifest.json').write_text(
                json.dumps(capture_manifest, indent=2, allow_nan=False)+'\n',
                encoding='utf-8')
        results.append(report)
        print('Warm-start trial:',json.dumps(report,sort_keys=True),flush=True)
        (output/'native_seed_report.json').write_text(json.dumps(results,indent=2)+'\n')
        if seed is not None:
            return seed,results
    return None,results
