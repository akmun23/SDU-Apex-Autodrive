#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np


def load_traj(path: Path):
    rows=[]
    with path.open('r',encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if not line or line.startswith('#'): continue
            try: vals=[float(x) for x in line.split(',')]
            except ValueError: continue
            if len(vals)>=7: rows.append(vals)
    a=np.asarray(rows,float)
    if len(a)>1 and np.hypot(a[-1,1]-a[0,1],a[-1,2]-a[0,2])<1e-5:
        a=a[:-1]
    return a


def summarize(a):
    s=a[:,0]; v=a[:,5]; k=a[:,4]; ax=a[:,6]
    ds=np.diff(np.r_[s, s[-1]+np.hypot(a[0,1]-a[-1,1],a[0,2]-a[-1,2])])
    vm=0.5*(v+np.roll(v,-1))
    lap=float(np.sum(ds/np.maximum(vm,1e-6)))
    ay=np.abs(k)*v*v
    out={
      'points':int(len(a)),'lap_time_s':lap,'length_m':float(np.sum(ds)),
      'speed_min_mps':float(v.min()),'speed_mean_mps':float(np.mean(v)),'speed_max_mps':float(v.max()),
      'curvature_abs_p95_m_inv':float(np.quantile(np.abs(k),.95)),'curvature_abs_max_m_inv':float(np.max(np.abs(k))),
      'ax_min_mps2':float(ax.min()),'ax_max_mps2':float(ax.max()),
      'ay_abs_p95_mps2':float(np.quantile(ay,.95)),'ay_abs_max_mps2':float(ay.max()),
    }
    if a.shape[1]>=9:
        out['left_min_m']=float(a[:,7].min()); out['right_min_m']=float(a[:,8].min())
    return out


def main():
    p=argparse.ArgumentParser()
    p.add_argument('new',type=Path)
    p.add_argument('--baseline',type=Path,default=None)
    p.add_argument('--optimizer-report',type=Path,default=None)
    p.add_argument('--output',type=Path,default=None)
    args=p.parse_args()
    result={'new':summarize(load_traj(args.new))}
    if args.baseline:
        result['baseline']=summarize(load_traj(args.baseline))
        result['delta']={
          'lap_time_s':result['new']['lap_time_s']-result['baseline']['lap_time_s'],
          'lap_time_percent':100*(result['new']['lap_time_s']/result['baseline']['lap_time_s']-1),
          'length_m':result['new']['length_m']-result['baseline']['length_m'],
          'max_speed_mps':result['new']['speed_max_mps']-result['baseline']['speed_max_mps'],
        }
    if args.optimizer_report and args.optimizer_report.exists():
        result['optimizer_report']=json.loads(args.optimizer_report.read_text())
    text=json.dumps(result,indent=2)
    print(text)
    if args.output: args.output.write_text(text,encoding='utf-8')

if __name__=='__main__': main()
