#!/usr/bin/env python3
"""Direct reversal of OLD XY geometry into the Unity-confirmed travel direction.

A *geometric* transfer only: resample/smooth old XY positions; discard every
old physical state, steering command, throttle, yaw-rate and source certificate.
The output is never marked as a valid MPC reference or PhysX witness.
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

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"SDU_Apex_IPOPT_PhysX_MinTime_v2"))
from map_track import Track,MapGrid
from track_optimizer import verified_unity_direction

HEADER="s_m,x_m,y_m,psi_rad,kappa_radpm,vx_mps,ax_mps2,d_left_m,d_right_m"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_candidate(*, old_csv, unity_csv, map_pgm, output, speed_fraction=.65,
                   smoothing=.003, spacing=.07, decimation=5):
    old_csv,unity_csv,map_pgm,output=(Path(p).resolve() for p in
                                     (old_csv,unity_csv,map_pgm,output))
    if not 0.1<speed_fraction<=1. or not .001<=smoothing<=.03:
        raise ValueError("invalid speed fraction or smoothing")
    old=np.loadtxt(old_csv,comments="#",delimiter=",",dtype=float)
    if old.ndim!=2 or old.shape[1]<9 or len(old)<100 or not np.isfinite(old).all():
        raise ValueError("invalid original witness geometry")
    verified=verified_unity_direction(unity_csv)
    old_area=.5*np.sum(old[:,1]*np.roll(old[:,2],-1)-old[:,2]*np.roll(old[:,1],-1))
    if old_area*verified["winding"]>=0:
        raise ValueError("source witness is not opposite Unity driving orientation")
    track=Track.from_xy(old[::-decimation,1:3],spacing=spacing,smoothing=smoothing)
    area=.5*np.sum(track.xy[:,0]*np.roll(track.xy[:,1],-1)-
                     track.xy[:,1]*np.roll(track.xy[:,0],-1))
    headings=np.unwrap(track.heading)
    # Track.heading wraps periodically, use tangent differentiation to
    # establish total turn rather than end-heading subtraction alone.
    delta=np.arctan2(np.sin(np.roll(track.heading,-1)-track.heading),
                     np.cos(np.roll(track.heading,-1)-track.heading))
    winding=round(float(np.sum(delta))/(2*math.pi))
    if winding!=verified['winding'] or area*winding<=0:
        raise ValueError("direct geometric reverse failed direction check")
    grid=MapGrid.load(map_pgm)
    slack=[]
    for i in range(len(track.xy)):
        x,y=track.xy[i]
        slack.append(grid.footprint_clearance(float(x),float(y),float(track.heading[i])))
        x,y,heading,_=track.at(track.s[i]+.5*spacing)
        slack.append(grid.footprint_clearance(x,y,heading))
    min_clear=float(np.min(slack))
    if min_clear<0:
        raise ValueError(f"direct reversed geometry intersects footprint margin: {min_clear:.6f} m")
    if abs(track.kappa).max()>math.tan(math.pi/6)/.324:
        raise ValueError("transferred geometry violates steering-implied curvature limit")
    unity=np.loadtxt(unity_csv,comments='#',delimiter=',')
    start=unity[0,1:3]
    start_idx=int(np.argmin(np.linalg.norm(track.xy-start,axis=1)))
    # The seam must occur near the simulator start rather than old witness's
    # arbitrary phase. This changes indexing only, not path geometry.
    xy=np.roll(track.xy,-start_idx,axis=0)
    heading=np.unwrap(np.roll(track.heading,-start_idx))
    kappa=np.roll(track.kappa,-start_idx)
    seg=np.linalg.norm(np.diff(np.vstack([xy,xy[:1]]),axis=0),axis=1)
    station=np.r_[0.,np.cumsum(seg[:-1])]
    old_xy=old[::-1,1:3]
    _,ind=cKDTree(old_xy).query(xy)
    speeds=np.clip(gaussian_filter1d(old[::-1,5][ind],sigma=4,mode='wrap')*
                   speed_fraction,1.15,8.0)
    accel=np.gradient(speeds**2,station,edge_order=1)/2.
    left_right=grid.side_clearances(xy,heading)
    out=np.column_stack([station,xy,heading,kappa,speeds,accel,left_right])
    if (not np.isfinite(out).all() or np.any(out[:,7:9]<=0)):
        raise ValueError("invalid transferred track table")
    output.parent.mkdir(parents=True,exist_ok=True)
    np.savetxt(output,out,delimiter=',',fmt='%.12g',header=HEADER)
    metadata=dict(schema="sdu-apex-direct-opposite-geometry-seed/v1",
        status="uncertified_geometry_initializer",source_only=True,
        source_witness_validated=False,ready_for_runtime=False,
        old_witness_path=str(old_csv),old_witness_sha256=sha256(old_csv),
        unity_reference_path=str(unity_csv),unity_reference_sha256=sha256(unity_csv),
        base_geometry_path=str(output),base_geometry_sha256=sha256(output),
        output_csv=str(output),output_sha256=sha256(output),
        geometry_transfer="reverse_source_xy_smooth_no_controls",
        speed_fraction=speed_fraction,smoothing=smoothing,decimation=decimation,
        length_m=float(seg.sum()),old_length_m=float(np.linalg.norm(
          np.diff(np.vstack([old[:,1:3],old[:1,1:3]]),axis=0),axis=1).sum()),
        speed_min_mps=float(speeds.min()),speed_max_mps=float(speeds.max()),
        minimum_sampled_footprint_slack_m=min_clear,
        max_abs_curvature_radpm=float(abs(kappa).max()),
        start_alignment_m=float(np.linalg.norm(xy[0]-start)),
        expected_winding=winding)
    output.with_suffix(".json").write_text(json.dumps(metadata,indent=2)+"\n")
    return metadata


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old-csv',type=Path,required=True)
    p.add_argument('--unity-csv',type=Path,required=True)
    p.add_argument('--map',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--speed-fraction',type=float,default=.65)
    p.add_argument('--smoothing',type=float,default=.003)
    a=p.parse_args()
    print(json.dumps(make_candidate(old_csv=a.old_csv,unity_csv=a.unity_csv,
        map_pgm=a.map,output=a.output,
        speed_fraction=a.speed_fraction,smoothing=a.smoothing),indent=2))


if __name__=="__main__":main()
