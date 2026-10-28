"""Source-native plant replay of IPOPT feasibility witnesses + target export.

This replay does not test the MPC: it executes the optimizer's internal
steering/throttle witness. It also does not replace Unity. The checked-in
source-native plant is a separate model, not Unity's native WheelCollider
solver.
"""
from __future__ import annotations
import csv
import json
import math
from pathlib import Path
import numpy as np
from map_track import (DEFAULT_WALL_MARGIN_M, _FOOTPRINT_RADIUS, MapGrid,
                       Track as ReferenceTrack)
from physics import NativePhysX, STEER_MAX
from path_frame import path_progress_numpy

DT=0.025

def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def _periodic(s, xs, vals, length):
    return float(np.interp(float(s)%float(length),
                           np.r_[xs,length],np.r_[vals,vals[0]]))


def _time_schedule(candidate, track):
    s=np.asarray(candidate['s_ref_m'],float)
    v=np.asarray(candidate['speed_mps'],float)
    vv=np.asarray(candidate['lateral_speed_mps'],float)
    epsi=np.asarray(candidate['epsi_rad'],float)
    ey=np.asarray(candidate['ey_m'],float)
    sdot=path_progress_numpy(
        track,s,ey,epsi,v,vv,roll=candidate.get('roll_rad',0.0),
        pitch=candidate.get('pitch_rad',0.0),
        height_rate=candidate.get('body_height_rate_mps',0.0),
        yaw_rate=candidate.get('yaw_rate_radps',0.0),
        pitch_rate=candidate.get('pitch_rate_radps',0.0))
    if np.any(sdot<=.10) or np.any(~np.isfinite(sdot)):
        raise ValueError('IPOPT result has nonpositive/invalid spatial progress')
    seg=np.diff(np.r_[s,track.length])*(1/sdot+1/np.roll(sdot,-1))*.5
    t=np.r_[0.,np.cumsum(seg)]
    lap=float(t[-1])
    if lap<=0 or not math.isfinite(lap):
        raise ValueError('Invalid predicted time integral')
    return t,lap


def _menger_curvature(xy: np.ndarray) -> np.ndarray:
    """Discrete geometric curvature for the optimizer's closed path samples."""
    points = np.asarray(xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
        raise ValueError('Raceline geometry needs at least three XY samples')
    before = points - np.roll(points, 1, axis=0)
    after = np.roll(points, -1, axis=0) - points
    chord = np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)
    denominator = np.maximum(
        np.linalg.norm(before, axis=1) * np.linalg.norm(after, axis=1)
        * np.linalg.norm(chord, axis=1), 1.0e-10)
    return 2.0 * (before[:, 0] * after[:, 1]
                  - before[:, 1] * after[:, 0]) / denominator


def replay(repo:Path,binary:Path|None,candidate,model_track,grid:MapGrid,
           ref:ReferenceTrack,output:Path,
           wall_margin:float=DEFAULT_WALL_MARGIN_M,
           warmup_laps:int=1,max_seconds:float=18.,
           minimum_drive_throttle:float=0.01):
    """Replay the OCP's internal feasible-control witness, not the target.

    This checks whether the OCP's hidden steering/throttle witness can execute
    in the source-native plant. It does not measure whether an MPC can track
    the exported geometry/speed/curvature target.
    """
    output.mkdir(parents=True,exist_ok=True)
    if (not math.isfinite(minimum_drive_throttle)
            or not 0.0 < minimum_drive_throttle < 1.0):
        raise ValueError('minimum drive throttle must be finite and in (0, 1)')
    s=np.asarray(candidate['s_ref_m'],float)
    length=float(model_track.length)
    time_nodes, predicted_lap=_time_schedule(candidate,model_track)
    x0=float(candidate['x_m'][0]);y0=float(candidate['y_m'][0])
    heading=float(model_track.psi[0]+candidate['epsi_rad'][0])
    speed=float(candidate['speed_mps'][0])
    if not (0.1<speed<16.01): raise ValueError('invalid IPOPT start velocity')
    phys=NativePhysX(repo,binary)
    trace=[]; controls=[]; crosses=[]
    failure=None; last_gap=None; min_gap=float('inf'); max_position_error=0.;max_speed_error=0.
    prev_progress=None; previous=None
    try:
        previous=phys.reset(x0,y0,heading,speed)
        proj=ref.project(previous.x,previous.y)
        if proj is None: raise RuntimeError('initial body pose cannot be projected onto track')
        first_s=proj[0]; prev_progress=first_s
        last_gap=grid.footprint_clearance(previous.x,previous.y,previous.yaw,margin=wall_margin)
        if last_gap<0: raise RuntimeError('native start pose violates wall clearance')
        min_gap=last_gap
        for step in range(int(math.ceil(max_seconds/DT))):
            loc=ref.project(previous.x,previous.y,prev_progress)
            if loc is None:
                failure='lost_progress_projection';break
            station_abs,ey,course=loc
            if station_abs < prev_progress-max(1.,.06*length):
                failure='reverse_progress';break
            prev_progress=max(prev_progress,station_abs)
            progress=prev_progress-first_s
            q=progress%length
            # Optimize road-wheel angle in IPOPT; advance the target by one
            # controller period to compensate Unity's rate-limited actuator.
            desired=_periodic(q,s,np.asarray(candidate['steering_rad']),length)
            steering_rate=_periodic(q,s,np.asarray(candidate['steering_rate_radps']),length)
            target=float(np.clip(desired+steering_rate*DT,-STEER_MAX,STEER_MAX))
            idx=int(np.searchsorted(s,q,side='right')-1)%len(s)
            throttle=float(candidate['throttle_norm'][idx])
            throttle=float(np.clip(throttle,minimum_drive_throttle,1.0))
            next_state=phys.step(target,throttle)
            motion=math.hypot(next_state.x-previous.x,next_state.y-previous.y)
            turning=abs(wrap(next_state.yaw-previous.yaw))
            swept=motion+_FOOTPRINT_RADIUS*turning
            end_gap=grid.footprint_clearance(next_state.x,next_state.y,next_state.yaw,margin=wall_margin)
            gap=min(last_gap,end_gap)
            if gap<=.5*swept:
                subdivisions=max(2,int(math.ceil(swept/max(.02,.6*grid.resolution))))
                for j in range(1,subdivisions):
                    frac=j/subdivisions
                    xm=previous.x+frac*(next_state.x-previous.x)
                    ym=previous.y+frac*(next_state.y-previous.y)
                    hm=previous.yaw+frac*wrap(next_state.yaw-previous.yaw)
                    gap=min(gap,grid.footprint_clearance(xm,ym,hm,margin=wall_margin))
                gap-=swept/(2*subdivisions)
            min_gap=min(min_gap,gap)
            last_gap=end_gap
            loc_next=ref.project(next_state.x,next_state.y,prev_progress)
            if loc_next is not None and loc_next[0]>=prev_progress-max(1.,.06*length):
                prev_progress=max(prev_progress,loc_next[0])
            actual_progress=prev_progress-first_s
            pred_x=_periodic(actual_progress,s,np.asarray(candidate['x_m']),length)
            pred_y=_periodic(actual_progress,s,np.asarray(candidate['y_m']),length)
            pred_speed=_periodic(actual_progress,s,np.asarray(candidate['speed_mps']),length)
            max_position_error=max(max_position_error,math.hypot(next_state.x-pred_x,next_state.y-pred_y))
            max_speed_error=max(max_speed_error,abs(next_state.u-pred_speed))
            controls.append([step*DT,q,target,throttle,desired,steering_rate])
            trace.append([(step+1)*DT,next_state.x,next_state.y,next_state.yaw,
                          next_state.u,next_state.v,next_state.r,actual_progress,
                          gap,throttle,target,*next_state.longitudinal_slip,*next_state.lateral_slip])
            previous=next_state
            if not np.isfinite([next_state.x,next_state.y,next_state.yaw,
                                next_state.u,next_state.v,next_state.r]).all():
                failure='nonfinite_native_state';break
            if gap<0:
                failure='footprint_wall_clearance';break
            if actual_progress>=(len(crosses)+1)*length:
                position_ok=math.hypot(next_state.x-x0,next_state.y-y0)<max(.65,.04*length)
                direction_ok=abs(wrap(next_state.yaw-heading))<.70
                if position_ok and direction_ok and next_state.u>0:
                    crosses.append((step+1)*DT)
                else:
                    failure='invalid_lap_gate_pose_or_heading';break
                if len(crosses)>=warmup_laps+1: break
        completed=failure is None and len(crosses)>=warmup_laps+1
        scored_time=(crosses[-1]-(crosses[warmup_laps-1] if warmup_laps else 0.)) if completed else None
        report={
            'backend':'native-physx-4.1',
            'validation_kind':('open-loop replay of optimizer feasibility-witness controls; '
                               'not MPC tracking and not Unity validation'),
            'control_witness_only':True,
            'mpc_tracking_validated':False,
            'unverified_unity_player_parity':True,
            'completed':bool(completed),
            'native_flying_lap_time_s':scored_time,
            'predicted_ipopt_lap_time_s':predicted_lap,
            'goal_sub_5_s':bool(completed and scored_time<5.),
            'collision_free':bool(min_gap>=0.),
            'track_length_m':length,
            'progress_m':float(max(0.,prev_progress-first_s)),
            'total_elapsed_s':len(trace)*DT,
            'crossing_times_s':crosses,
            'min_swept_footprint_clearance_beyond_margin_m':float(min_gap),
            'requested_wall_margin_m':wall_margin,
            'max_position_difference_at_matched_progress_m':max_position_error,
            'max_speed_difference_at_matched_progress_mps':max_speed_error,
            'failure':failure if failure is not None else (None if completed else 'time_limit'),
            'controls':len(controls),
            'source_notes':'PhysX scene from recovered Unity shapes and source callback on flat ground; map walls checked with swept 2D footprint (not simulated as PhysX colliders).',
        }
        with (output/'native_trace.csv').open('w',newline='') as f:
            wr=csv.writer(f);wr.writerow(['time_s','x_m','y_m','yaw_rad','u_mps','v_mps','yaw_rate_radps','progress_m','swept_clearance_beyond_margin_m','throttle','steering_target_rad',
                                      'long_slip_fl','long_slip_fr','long_slip_rl','long_slip_rr','lat_slip_fl','lat_slip_fr','lat_slip_rl','lat_slip_rr']);wr.writerows(trace)
        with (output/'commands_40hz.csv').open('w',newline='') as f:
            wr=csv.writer(f);wr.writerow(['time_s','station_m','steering_target_rad','throttle_normalized','desired_physical_steer_rad','ipopt_steer_rate_radps']);wr.writerows(controls)
        (output/'native_report.json').write_text(json.dumps(report,indent=2)+'\n')
        return report
    finally:
        phys.close()


def export_mpc(candidate,grid:MapGrid,path:Path,
               wall_margin:float=DEFAULT_WALL_MARGIN_M):
    """Export the speed/curvature target with continuous swept-map clearance.

    Steering and throttle from the OCP are feasibility witnesses and are not
    part of this runtime trajectory. The target path is checked between knots
    using a conservative footprint-motion bound, not only at sample points.
    """
    xy=np.column_stack([candidate['x_m'],candidate['y_m']])
    segments=np.linalg.norm(np.roll(xy,-1,axis=0)-xy,axis=1)
    if np.any(segments<=1e-7): raise ValueError('Degenerate optimized geometry')
    s=np.r_[0.,np.cumsum(segments[:-1])]
    u=np.asarray(candidate['speed_mps'],float)
    diff=np.roll(xy,-1,axis=0)-np.roll(xy,1,axis=0)
    psi=np.unwrap(np.arctan2(diff[:,1],diff[:,0]))
    kappa=np.asarray(candidate['path_curvature_per_m'],dtype=np.float64)
    geometry_kappa=_menger_curvature(xy)
    if (kappa.shape != (len(xy),) or not np.isfinite(kappa).all()
            or not np.allclose(kappa,geometry_kappa,rtol=1.0e-8,atol=1.0e-9)):
        raise ValueError(
            'Optimizer curvature target does not match exported XY geometry; '
            'refusing to replace it with a separately approximated curvature')
    # Geometric acceleration is sourced from optimized source-model ax, not a
    # controller-generated speed request or a generic physical capability.
    acc=np.asarray(candidate['ax_mps2'],float)
    records=[]
    for i,p in enumerate(xy):
        nx=-math.sin(psi[i]);ny=math.cos(psi[i]);widths=[]
        for sign in (+1,-1):
            d=np.arange(0.,3.01,max(.0125,grid.resolution*.5))
            points=p+sign*d[:,None]*np.array([nx,ny])
            clear=np.asarray(grid.distance(points))
            hit=np.flatnonzero(clear<grid.resolution*.55)
            widths.append(float(d[hit[0]] if len(hit) else d[-1]))
        records.append((s[i],p[0],p[1],psi[i],kappa[i],u[i],acc[i],*widths))
    # The distance-to-occupied map is 1-Lipschitz. Between adjacent pose
    # samples every body point moves by at most translation + R*rotation;
    # subtract half that interval bound to certify the unsampled midpoint and
    # all points between it and either sample.
    max_sample_motion = 0.5 * grid.resolution
    for i in range(len(xy)):
        j = (i + 1) % len(xy)
        heading_delta = wrap(float(psi[j] - psi[i]))
        motion_bound = float(segments[i]) + _FOOTPRINT_RADIUS * abs(heading_delta)
        subdivisions = max(1, int(math.ceil(motion_bound / max_sample_motion)))
        sampled_clearance = math.inf
        # A coarse Lipschitz certificate can be inconclusive even when the
        # entire footprint clears the requested margin. Refine that interval
        # before rejecting it; preserve the same conservative proof and margin.
        for refinement in range(12):
            samples = (range(subdivisions + 1) if refinement == 0
                       else range(1, subdivisions, 2))
            for sample in samples:
                fraction = sample / subdivisions
                position = xy[i] + fraction * (xy[j] - xy[i])
                heading = psi[i] + fraction * heading_delta
                sampled_clearance = min(
                    sampled_clearance,
                    grid.footprint_clearance(float(position[0]),
                                             float(position[1]),
                                             float(heading), margin=wall_margin))
            conservative_clearance = sampled_clearance - 0.5 * motion_bound / subdivisions
            if conservative_clearance >= 0.0 or sampled_clearance < 0.0:
                break
            subdivisions *= 2
        if conservative_clearance < 0.0:
            raise ValueError(
                'IPOPT path violates swept original-map footprint clearance '
                f'between knots {i} and {j}; conservative clearance '
                f'{conservative_clearance:.6f} m; sampled clearance '
                f'{sampled_clearance:.6f} m')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        f.write('# s_m,x_m,y_m,psi_rad,kappa_radpm,body_forward_speed_mps,acceleration_mps2,d_left_m,d_right_m\n')
        wr=csv.writer(f);wr.writerows([[f'{v:.9f}' for v in row] for row in records])
    return path
