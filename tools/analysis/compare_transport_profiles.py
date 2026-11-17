#!/usr/bin/env python3
"""Conservative gate before treating batchmode as a graphical timing proxy.

Passing proves similarity of *observed transport statistics only*, not that
PhysX frames, steering limits, tire-force behavior, or control policy match.
"""
import argparse
import json
from pathlib import Path
import sys


def load_report(path: Path) -> dict:
    contents=path.read_text(encoding='utf-8')
    if not contents.strip().startswith('{'):
        # A live monitor prints one AWAITING_UNITY_CONNECT line before JSON.
        brace=contents.find('\n{')
        if brace == -1:raise ValueError(f'No completed JSON measurement in {path}')
        contents=contents[brace+1:]
    result=json.loads(contents)
    if result.get('schema_version')!=1 or 'channels' not in result:
        raise ValueError(f'Not a transport latency measurement: {path}')
    return result


def compare(dev:dict,gui:dict)->list[str]:
    failures=[]
    def check(condition, description):
        print(('OK ' if condition else 'FAIL ') + description)
        if not condition:failures.append(description)
    for channel in ('imu','odom','pose'):
        a=dev['channels'][channel];b=gui['channels'][channel]
        check(min(a['count'],b['count']) >= 120,
              f'{channel}: >=120 messages in each trial ({a["count"]}/{b["count"]})')
        for key,tolerance in [('age_p50_ms',5.0),('age_p95_ms',8.0)]:
            x=a.get(key);y=b.get(key)
            check(x is not None and y is not None and abs(x-y)<=tolerance,
                  f'{channel}: {key} dev={x} GUI={y}, tolerance={tolerance} ms')
        x=a.get('rate_hz');y=b.get('rate_hz')
        check(x is not None and y is not None and x>=36.0 and y>=36.0 and abs(x-y)<=3.0,
              f'{channel}: rates dev={x} GUI={y}, each >=36Hz and within 3Hz')
    for label in ('mpc_command','bridge_feedback'):
        a=dev['channels'][label];b=gui['channels'][label]
        check(min(a['count'],b['count']) >= 100,
              f'{label}: >=100 messages in both ({a["count"]}/{b["count"]})')
        a_max=a.get('interarrival_max_ms');b_max=b.get('interarrival_max_ms')
        check(a_max is not None and b_max is not None and a_max<=140 and b_max<=140,
              f'{label}: maximum interarrival dev={a_max} GUI={b_max}, each <=140 ms')
    a=dev.get('effective_command_to_unity_feedback_alignment')
    b=gui.get('effective_command_to_unity_feedback_alignment')
    for term in ('steering','throttle'):
        x=a.get(term,{}).get('best_alignment_delay_ms') if isinstance(a,dict) else None
        y=b.get(term,{}).get('best_alignment_delay_ms') if isinstance(b,dict) else None
        check(x is not None and y is not None and abs(x-y)<=10,
              f'{term}: effective response alignment dev={x} GUI={y}, tolerance=10 ms')
    return failures


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--development',type=Path,required=True)
    p.add_argument('--graphical',type=Path,required=True)
    args=p.parse_args()
    bad=compare(load_report(args.development),load_report(args.graphical))
    if bad:
        print(f'NOT CALIBRATED: {len(bad)} timing/response checks failed. Do not use dev as a graphical proxy.')
        raise SystemExit(1)
    print('TRANSPORT CALIBRATION GATE PASS (not proof of identical vehicle dynamics or collision behavior)')

if __name__=='__main__':main()
