#!/usr/bin/env python3
"""Evaluate solved hard 60-control condensed QP in the actual C++ source plant.

Use a captured matching command epoch (source_context.mctx) and an elastic
QP captured by --compare-normal-sqp (mode26, rack-relative increment).
Solve the ORIGINAL hard corridor QP by removing slack weights, then map
QP decisions to actual steering/throttle commands, and ask C++ replay to
integrate those commands with full hidden PhysX source state, actuator FIFO
and original reference/path geometry.

This is an offline diagnostic, not a controller policy or safety certificate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.mpc_qp_oracle import decode_problem
from tools.audit_mpc_condensed_qp import condense,solve_qp,_load_osqp


def evaluate(fixture:Path,case:Path,vendor:Path,output:Path):
    sys.path.insert(0,str(vendor))
    p=decode_problem(fixture.read_bytes())
    c=condense(p,hard_walls=True)
    solved,setup_ms,solve_ms=solve_qp(
        c['P'],c['q'],c['A'],c['l'],c['u'],eps=1.e-6,max_iter=70000)
    info=dict(fixture=str(fixture),case=str(case),
              qp_status=solved.info.status,
              condensed_solver_ms=solve_ms)
    if solved.info.status!='solved':
        return info
    decision=solved.x.reshape(30,2)
    states=(c['c']+c['T']@solved.x)[:31*23].reshape(31,23)
    # mode26 specifically enabled source_reachable_steering_qp: steering
    # increment is relative to the actual wheel/rack angle x[k,5], not
    # previous command x[k,21]. Throttle increment is relative to x[k,22].
    commands=np.column_stack([decision[:,0]+states[:30,5],
                              decision[:,1]+states[:30,22]])
    with output.open('w') as f:
        for steer,throttle in commands:
            f.write(f'{steer:.12f} {throttle:.12f}\n')
    command_max_steer=float(np.max(np.abs(commands[:,0])))
    command_min_throttle=float(np.min(commands[:,1]))
    command_max_throttle=float(np.max(commands[:,1]))
    info.update(first_command=commands[0].tolist(),
                maximum_absolute_steering_command_rad=command_max_steer,
                minimum_throttle_command=command_min_throttle,
                maximum_throttle_command=command_max_throttle,
                hard_qp_max_wall_excess=max(0.,max(
                    float(normal0*states[k,0]+normal1*states[k,1]+
                          progress*states[k,20]-bound)
                    for k,poly in enumerate(p['state_polygons']) if k>0
                    for normal0,normal1,bound,progress in poly)),
                command_file=str(output))
    cmd=[str(ROOT/'build/f1tenth_mpc/mpc_live_failure_replay'),
         str(case),'--external-controls='+str(output)]
    r=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                     text=True,timeout=20)
    info['replay_returncode']=r.returncode
    if r.returncode:
        info['replay_error']=r.stderr[-450:]
    for line in r.stdout.splitlines():
        if '"external_source_controls":true' in line:
            data=json.loads(line)
            for k in ('physical_valid','failure_stage','complete',
                      'source_improvement','qp_source_max_state_error',
                      'qp_source_max_footprint_error_m'):
                info['source_'+k]=data.get(k)
            # QP error here uses original captured R1 QP, not independently
            # generated mode26 QP. Do not treat that number as parity.
            del info['source_qp_source_max_state_error']
            del info['source_qp_source_max_footprint_error_m']
    return info


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('fixture',type=Path)
    parser.add_argument('case',type=Path)
    parser.add_argument('--vendor',type=Path,default=Path('/tmp/sdu-mpc-osqp-v104'))
    parser.add_argument('--output',type=Path,default=Path('/tmp/mpc_condensed_hard_source_controls.txt'))
    args=parser.parse_args()
    try:
        print(json.dumps(evaluate(args.fixture,args.case,args.vendor,args.output),sort_keys=True))
    except Exception as e:
        print(json.dumps({'error':str(e),'fixture':str(args.fixture)}))
        return 2
    return 0


if __name__=='__main__':
    raise SystemExit(main())
