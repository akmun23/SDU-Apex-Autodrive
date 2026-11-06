#!/usr/bin/env python3
"""Continue minimum-time search from a certified direct reversed-XY geometric seed.

Shares no mutable optimizer workspace with the archived opposite-direction
search or the slower map-only bootstrap. Pauses initial startup until the
seed's Unity/MPC ten-lap validation attempt has exited; neither result is
silently promoted into the competition controller.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'SDU_Apex_IPOPT_PhysX_MinTime_v2'))
from command_campaign import run_campaign, save_json_checkpoint
from track_optimizer import publish_best, require_travel_orientation, verified_unity_direction

stop=False
def on_stop(_signal,_frame):
    global stop
    stop=True


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--iterations',type=int,default=900)
    p.add_argument('--stagnation',type=int,default=180)
    p.add_argument('--phases',type=int,default=4)
    p.add_argument('--refinements',type=int,default=6)
    p.add_argument('--preflight-only',action='store_true')
    args=p.parse_args()
    workspace=args.workspace.resolve()
    path=workspace/'optimizer.json'
    state=json.loads(path.read_text())
    best=json.loads((workspace/'best.json').read_text())
    source=state['contract']['source_sha256']
    seed=Path(state.get('continuation_source_seed') or best['files']['witness'])
    verified=verified_unity_direction(ROOT/'live_runs/iros_mpc_initial_20261029_02/raceline.csv')
    if best['direction']!='unity_verified' or best['direction_reference_sha256']!=verified['sha256']:
        raise ValueError('only the Unity-direction source-certified witness can be continued')
    require_travel_orientation(seed,verified['winding'],verified=True)
    evidence=json.loads(Path(best['files']['validation']).read_text())
    if (evidence['status']!='source_witness_passed' or
            evidence['source_plant_sha256']!=source or
            sha(best['files']['witness'])!=best['witness_sha256']):
        raise ValueError('source witness certification/provenance failure')
    if args.preflight_only:
        print(json.dumps(dict(source_certified=True,lap_s=best['lap_s'],
                 geometry_direction=verified['geometry_orientation'],
                 current_seed=str(seed),validation=str(best['files']['validation'])),indent=2))
        return 0
    signal.signal(signal.SIGTERM,on_stop)
    signal.signal(signal.SIGINT,on_stop)
    # Allow the first Unity/MPC experiment to finish before allocating another
    # large IPOPT instance. This avoids biasing MPC timing with solver CPU load.
    trial_state=workspace/'mpc_validation/state.json'
    while True:
        trial=json.loads(trial_state.read_text()) if trial_state.exists() else {}
        if trial.get('attempts') and trial.get('worker',{}).get('status')=='idle':
            break
        if stop:
            print('Stopped while waiting for initial Unity trial; no optimizer changes.',flush=True)
            return 130
        time.sleep(5)
    campaign=workspace/'starts/unity_reverse_seed/campaign'
    state['continuation_source_seed']=str(seed)
    state['continuation_source_seed_sha256']=sha(seed)
    state['contract'].update(iterations=args.iterations,
                             stagnation_iterations=args.stagnation,
                             phases=args.phases,refinements=args.refinements)
    state.update(stage='adaptive_minimum_time_search',status='running',
                 active_start='unity_reverse_seed',
                 geometry_starts=[dict(name='unity_reverse_seed',
                                      csv=str(workspace.parent/'direct_old_curve_seed.csv'))],
                 starts=dict(unity_reverse_seed=dict(witness=str(seed))),
                 pid=os.getpid(),updated_epoch_s=time.time())
    save_json_checkpoint(path,state)
    print(json.dumps(dict(event='continuation_start',lap_s=best['lap_s'],
                 seed=str(seed),closed_loop_attempts=len(trial['attempts']))),flush=True)

    def publish(witness,lap):
        publish_best(workspace,witness,lap,state,source_digest=source,
                     verified_direction=verified)
        state.update(updated_epoch_s=time.time(),pid=os.getpid())
        save_json_checkpoint(path,state)
    try:
        result=run_campaign(repo=ROOT,seed=seed,
            map_path=ROOT/'f1tenth_planning/maps/autodrive_iros_2026_compete.pgm',
            output=campaign,target_seconds=None,
            phases=args.phases,workers=args.workers,iterations=args.iterations,
            refinements=args.refinements,stagnation_iterations=args.stagnation,
            coarse_fraction=.1,resume=(campaign/'campaign.json').exists(),
            cancel_requested=lambda:stop,on_improvement=publish)
        status=result['status']
        state.update(status=status,stage='finished',active=None,
                     updated_epoch_s=time.time(),pid=os.getpid())
        save_json_checkpoint(path,state)
        print('Minimum-time campaign finished with status:',status,flush=True)
        return 0
    except KeyboardInterrupt:
        state.update(status='interrupted',stage='interrupted',
                     updated_epoch_s=time.time(),pid=os.getpid())
        save_json_checkpoint(path,state)
        print('Source campaign interrupted; durable checkpoints retained.',flush=True)
        return 130


if __name__=='__main__':
    raise SystemExit(main())
