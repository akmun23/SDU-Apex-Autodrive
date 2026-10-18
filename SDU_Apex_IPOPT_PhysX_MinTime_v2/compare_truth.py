#!/usr/bin/env python3
"""Held-out offline comparison between PhysX candidate trace and Unity truth.

Does not access simulator-only truth at runtime; truth CSV is an offline label.
No default tolerance is silently treated as evidence of 1:1 equivalence.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np

COLUMNS={
    'x_m': ('x_m',), 'y_m': ('y_m',),
    'yaw_rad': ('physical_yaw_rad','yaw_rad','psi_rad'),
    'u_mps': ('u_mps','forward_velocity_mps'),
    'v_mps': ('v_mps','lateral_velocity_mps'),
    'yaw_rate_radps': ('yaw_rate_radps','r_radps')}


def read(path):
    with Path(path).open() as f:
        reader=csv.DictReader(f)
        data=list(reader)
        if not data: raise ValueError(f'CSV has no rows: {path}')
        result={}
        for name in COLUMNS:
            col=next((v for v in COLUMNS[name] if v in reader.fieldnames),None)
            if col is not None: result[name]=np.array([float(row[col]) for row in data])
        if 'time_s' not in reader.fieldnames:
            raise ValueError(f'CSV must include time_s: {path}')
        result['time_s']=np.array([float(row['time_s']) for row in data])
        if np.any(np.diff(result['time_s'])<=0):
            raise ValueError(f'Timestamps must strictly increase: {path}')
        return result


def analyze(truth,source,offset):
    actual=read(truth);predicted=read(source)
    t=predicted['time_s']+offset
    mask=(t>=actual['time_s'][0])&(t<=actual['time_s'][-1])
    if int(mask.sum())<10:
        raise ValueError('Fewer than 10 time-aligned samples; check --time-offset')
    result={'paired_samples':int(mask.sum()),'time_offset_s':offset,
            'compared_channels':{},'warning':'Comparison cannot prove bit-for-bit Unity parity.'}
    for name in COLUMNS:
        if name not in actual or name not in predicted:continue
        actual_values=np.interp(t[mask],actual['time_s'],actual[name])
        if name=='yaw_rad':
            truth_unwrapped=np.unwrap(actual[name])
            actual_values=np.interp(t[mask],actual['time_s'],truth_unwrapped)
            e=np.arctan2(np.sin(predicted[name][mask]-actual_values),
                         np.cos(predicted[name][mask]-actual_values))
        else:
            e=predicted[name][mask]-actual_values
        result['compared_channels'][name]={
            'rmse':float(np.sqrt(np.mean(e*e))),
            'mean_error':float(np.mean(e)),
            'p95_absolute_error':float(np.percentile(np.abs(e),95)),
            'max_absolute_error':float(np.max(np.abs(e)))}
    return result


def main():
    ap=argparse.ArgumentParser(description='Compare native prediction to independent recorded Unity ground truth')
    ap.add_argument('--truth',required=True,type=Path)
    ap.add_argument('--trace',required=True,type=Path)
    ap.add_argument('--time-offset',type=float,default=0.)
    ap.add_argument('--output',type=Path,default=Path('physx_vs_unity_validation.json'))
    args=ap.parse_args()
    result=analyze(args.truth,args.trace,args.time_offset)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    print('This is an offline comparison only; no 1:1 parity claim is made.')

if __name__=='__main__':main()
