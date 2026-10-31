#!/usr/bin/env python3
"""Transfer old (opposite-direction) optimized XY geometry to Unity-verified traversal.

Never reverse old controls/steering/yaw-rate or promote an old source certificate.
Result is an UNCERTIFIED 9-column geometric initializer, usable by
track_optimizer.geometric_seed to initialize an entirely new dynamics OCP.

The verified Unity initial path owns direction, map-relative starting phase,
and safety baseline; the legacy witness contributes lateral offsets and a
conservatively scaled speed-shape suggestion only.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[1]
OPT = REPO/"SDU_Apex_IPOPT_PhysX_MinTime_v2"
sys.path.insert(0,str(OPT))
from map_track import MapGrid
from track_optimizer import verified_unity_direction

SCHEMA = "sdu-apex-opposite-witness-geometry-transfer/v1"
HEADER = "s_m,x_m,y_m,psi_rad,kappa_radpm,vx_mps,ax_mps2,d_left_m,d_right_m"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def area(xy: np.ndarray) -> float:
    return .5*float(np.sum(xy[:,0]*np.roll(xy[:,1],-1)-xy[:,1]*np.roll(xy[:,0],-1)))


def make_candidate(
    *, old_csv: Path, unity_csv: Path, base_csv: Path, map_pgm: Path,
    output: Path, prefer_blend: float=.8, speed_fraction: float=.65,
    min_extra_clearance: float=.005,
    speed_cap: float=7.5, speed_floor: float=1.25,
) -> dict:
    old_csv, unity_csv, base_csv, map_pgm, output = (
        Path(p).resolve() for p in (old_csv,unity_csv,base_csv,map_pgm,output))
    old=np.loadtxt(old_csv,delimiter=",",comments="#",dtype=float)
    reference=np.loadtxt(base_csv,delimiter=",",comments="#",dtype=float)
    if (old.ndim!=2 or old.shape[1]<9 or len(old)<50 or reference.ndim!=2 or
        reference.shape[1]<9 or len(reference)<50 or not np.isfinite(old).all()
        or not np.isfinite(reference).all() or not 0<=prefer_blend<=1
        or not .1<=speed_fraction<=1):
        raise ValueError("invalid witness/base geometry or transfer parameters")
    verified=verified_unity_direction(unity_csv)
    if math.copysign(1,area(old[:,1:3]))==math.copysign(1,verified['winding']):
        raise ValueError("legacy witness is not opposite the verified Unity travel direction")
    if math.copysign(1,area(reference[:,1:3]))!=math.copysign(1,verified['winding']):
        raise ValueError("baseline geometry does not match verified Unity travel direction")
    grid=MapGrid.load(map_pgm)
    base_xy=reference[:,1:3].copy()
    # NOTE: Old source state, wheel/motor commands, yaw and rack are discarded.
    # Reversal affects only the geometric XY ordering.
    old_xy=old[::-1,1:3].copy()
    old_speed=old[::-1,5].copy()
    tree=cKDTree(old_xy)
    proximity, index=tree.query(base_xy)
    if np.quantile(proximity,.95)>1.0:
        raise ValueError("legacy line too far from verified Unity baseline")
    base_heading=np.unwrap(reference[:,3])
    normal=np.column_stack([-np.sin(base_heading),np.cos(base_heading)])
    offset=np.sum((old_xy[index]-base_xy)*normal,axis=1)
    offset=gaussian_filter1d(np.clip(offset,-1.1,1.1),sigma=5.,mode="wrap")
    # Speeds are merely initialization guesses. Transfer by map position,
    # then smooth and reduce; not a verified speed profile for reverse travel.
    speed_hint=gaussian_filter1d(old_speed[index],sigma=5.,mode="wrap")
    speed=np.clip(speed_fraction*speed_hint,speed_floor,speed_cap)
    speed=gaussian_filter1d(speed,sigma=4.,mode="wrap")
    # Try the most aggressive geometric transfer that retains 5+ mm extra
    # clearance at each sampled perimeter and the half-step path samples.
    # End physical 1ms source-tick clearance is *still* independently certified.
    candidates=[round(v,4) for v in np.linspace(prefer_blend,0.,21)]
    result=None
    for blend in candidates:
        xy=base_xy + (blend*offset)[:,None]*normal
        closed=np.vstack([xy,xy[0]])
        edges=np.diff(closed,axis=0)
        if np.any(np.linalg.norm(edges,axis=1)<1e-4):
            continue
        yaw=np.unwrap(np.arctan2(edges[:,1],edges[:,0]))
        mid_xy=xy+.5*edges
        next_yaw=np.roll(yaw,-1)
        mid_yaw=yaw+.5*np.arctan2(np.sin(next_yaw-yaw),np.cos(next_yaw-yaw))
        clear=np.array([grid.footprint_clearance(float(x),float(y),float(h))
                        for point,h in zip(np.vstack([xy,mid_xy]),np.r_[yaw,mid_yaw])
                        for x,y in [point]])
        if float(np.min(clear))<min_extra_clearance:
            continue
        # Periodic cubic needed because track_optimizer uses a periodic cubic
        # of the output; curvature and clearance are checked again at 2x
        # station density to catch spline overshoot.
        from scipy.interpolate import CubicSpline
        seg=np.linalg.norm(edges,axis=1)
        s=np.r_[0.,np.cumsum(seg)]
        curve=CubicSpline(s,closed,bc_type="periodic")
        at=(s[:-1]+s[1:])/2
        pp=curve(np.r_[s[:-1],at])
        v=curve(np.r_[s[:-1],at],1)
        test_heading=np.arctan2(v[:,1],v[:,0])
        spline_clear=np.array([grid.footprint_clearance(float(x),float(y),float(h))
                               for (x,y),h in zip(pp,test_heading)])
        if float(np.min(spline_clear))<min_extra_clearance:
            continue
        winding=round((float(np.unwrap(np.arctan2(v[:len(xy),1],v[:len(xy),0]))[-1]-
                          np.unwrap(np.arctan2(v[:len(xy),1],v[:len(xy),0]))[0]))/
                        (2*math.pi))
        if winding!=verified['winding']:
            continue
        tangent=curve(s[:-1],1)
        acceleration=np.gradient(speed**2,s[:-1],edge_order=1)/2.
        side=grid.side_clearances(xy,np.arctan2(tangent[:,1],tangent[:,0]))
        second=curve(s[:-1],2)
        curvature=(tangent[:,0]*second[:,1]-tangent[:,1]*second[:,0])/(
            np.maximum(np.linalg.norm(tangent,axis=1)**3,1e-12))
        heading=np.unwrap(np.arctan2(tangent[:,1],tangent[:,0]))
        result=(blend,xy,heading,curvature,s[:-1],acceleration,side,
                float(min(clear.min(),spline_clear.min())),float(s[-1]))
        break
    if result is None:
        raise ValueError("cannot safely transfer legacy geometry even after reducing blend")
    blend,xy,heading,curvature,s,accel,side,min_slack,length=result
    table=np.column_stack([s,xy,heading,curvature,speed,accel,side])
    if not np.isfinite(table).all() or (table[:,7:9]<=0).any():
        raise ValueError("invalid geometry transferred")
    output.parent.mkdir(parents=True,exist_ok=True)
    np.savetxt(output,table,delimiter=",",fmt="%.12g",header=HEADER)
    report=dict(schema=SCHEMA,status="uncertified_geometry_initializer",
                old_witness_path=str(old_csv),old_witness_sha256=sha(old_csv),
                unity_reference_path=str(unity_csv),
                unity_reference_sha256=sha(unity_csv),
                base_geometry_path=str(base_csv),base_geometry_sha256=sha(base_csv),
                output_csv=str(output),output_sha256=sha(output),
                transferred_blend=blend, speed_fraction=speed_fraction,
                speed_min_mps=float(speed.min()),speed_max_mps=float(speed.max()),
                speed_median_mps=float(np.median(speed)),
                length_m=length,baseline_length_m=float(np.linalg.norm(
                  np.diff(np.vstack([base_xy,base_xy[:1]]),axis=0),axis=1).sum()),
                old_length_m=float(np.linalg.norm(
                  np.diff(np.vstack([old_xy,old_xy[:1]]),axis=0),axis=1).sum()),
                min_sampled_extra_footprint_clearance_m=min_slack,
                expected_winding=verified["winding"],
                # The source model MUST decide whether a shorter lap is feasible.
                ready_for_runtime=False,source_witness_validated=False)
    output.with_suffix(".json").write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--old-csv",type=Path,required=True)
    p.add_argument("--unity-csv",type=Path,required=True)
    p.add_argument("--baseline-csv",type=Path,required=True)
    p.add_argument("--map",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--blend",type=float,default=.8)
    p.add_argument("--speed-fraction",type=float,default=.65)
    a=p.parse_args()
    print(json.dumps(make_candidate(old_csv=a.old_csv,unity_csv=a.unity_csv,
        base_csv=a.baseline_csv,map_pgm=a.map,output=a.output,
        prefer_blend=a.blend,speed_fraction=a.speed_fraction),indent=2))


if __name__=="__main__":main()
