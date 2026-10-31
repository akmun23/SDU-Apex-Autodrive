#!/usr/bin/env python3
"""Run an independent source-dynamics reoptimization of transferred old XY geometry.

NEVER import reversed old controls or source-plant state. A transfer is
geometrically checked but uncertified until solve_witness and certify_witness.
This tool uses its own folder, leaving the active optimizer and Unity trial
alone. Resume explicitly after a crash; never overwrite a completed solve.
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
sys.path.insert(0,str(ROOT/"SDU_Apex_IPOPT_PhysX_MinTime_v2"))
from command_ocp import solve_witness, certify_witness, production_command_map
from track_optimizer import require_travel_orientation, publish_best, verified_unity_direction
from command_campaign import save_json_checkpoint, source_count

stop=False
def on_signal(_number,_frame):
    global stop
    stop=True


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--transfer",type=Path,required=True)
    p.add_argument("--seed",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--workers",type=int,default=4)
    p.add_argument("--iterations",type=int,default=900)
    p.add_argument("--stagnation",type=int,default=200)
    p.add_argument("--attempts",type=int,default=3)
    a=p.parse_args()
    transfer=json.loads(a.transfer.read_text())
    if transfer['status']!='uncertified_geometry_initializer':
        raise ValueError("cannot trust transfer status")
    for key,field in [('old_witness_path','old_witness_sha256'),
                      ('unity_reference_path','unity_reference_sha256'),
                      ('base_geometry_path','base_geometry_sha256'),
                      ('output_csv','output_sha256')]:
        if digest(transfer[key])!=transfer[field]:
            raise ValueError(f"geometric seed provenance mismatch for {key}")
    verified=verified_unity_direction(transfer['unity_reference_path'])
    if verified['winding']!=transfer['expected_winding']:
        raise ValueError("Unity driving direction mismatch")
    output=a.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    manifest=output/'optimizer.json'
    plant_digest=production_command_map(ROOT,ROOT/'build_task/command_ocp')[1]
    contract=dict(schema=1,mode='reverse_geometry_only',
                  old_reference_sha256=transfer['old_witness_sha256'],
                  unity_reference_sha256=verified['sha256'],
                  transferred_geometry_sha256=transfer['output_sha256'],
                  source_sha256=plant_digest,
                  seed_sha256=digest(a.seed),workers=a.workers,iterations=a.iterations,
                  stagnation=a.stagnation,attempts=a.attempts,
                  direction='unity_verified',expected_winding=verified['winding'])
    data=json.loads(manifest.read_text()) if manifest.is_file() else dict(
        status='initializing',stage='seed_reoptimization',best_lap_s=None,
        attempts=[],contract=contract,source_only=True)
    if data['contract']!=contract:
        raise ValueError("reoptimization contract changed; choose new folder")
    def save(**updates):
        data.update(updates,updated_epoch_s=time.time(),pid=os.getpid())
        save_json_checkpoint(manifest,data)
    save(status='running')
    map_pgm=ROOT/'f1tenth_planning/maps/autodrive_iros_2026_compete.pgm'
    signal.signal(signal.SIGTERM,on_signal)
    signal.signal(signal.SIGINT,on_signal)
    previous=a.seed.resolve()
    for index in range(a.attempts):
        if stop:
            save(status='interrupted')
            return 130
        folder=output/f'solve_{index:03d}'
        report_path=folder/'report.json'
        if not report_path.is_file():
            if folder.exists():
                print("Existing incomplete solve; preserving evidence:",folder,flush=True)
                if (folder/'best_iterate.npz').is_file():
                    previous=folder/'best_iterate.npz'
                else:
                    previous=folder/'witness.npz'
                continue
            save(stage='seed_reoptimization',active=dict(output=str(folder),iteration=index))
            print(json.dumps(dict(event='solve',output=str(folder),seed=str(previous),
                                  trial_number=index,source_only=True)),flush=True)
            report=solve_witness(repo=ROOT,checkpoint=None,mesh=None,audit=None,
                map_path=map_pgm,output=folder,count=source_count(previous),
                iterations=a.iterations,cpu_seconds=1.e20,
                command_seed=previous,workers=a.workers,
                stagnation_iterations=a.stagnation,cancel_requested=lambda: stop)
        else:
            report=json.loads(report_path.read_text())
        outcome=dict(attempt=index,report=str(report_path),
                     status=report['status'],
                     residual=report.get('max_constraint_violation'),
                     trial_lap_s=report.get('trial_lap_s'))
        if not any(x['attempt']==index for x in data['attempts']):
            data['attempts'].append(outcome)
        save()
        if report['status']=='source_witness_passed':
            checked=output/f'certified_{index:03d}'
            if not checked.exists():
                result=certify_witness(ROOT,folder/'witness.npz',map_pgm,checked)
            else:
                result=json.loads((checked/'report.json').read_text())
            if result['status']=='source_witness_passed':
                require_travel_orientation(checked/'witness.npz',verified['winding'],verified=True)
                publish_best(output,checked/'witness.npz',result['trial_lap_s'],
                    data,source_digest=plant_digest,verified_direction=verified)
                save(status='certified',stage='seed_verified',active=None,
                     best_lap_s=result['trial_lap_s'])
                print("TRANSFERRED SEED CERTIFIED:",result['trial_lap_s'],flush=True)
                return 0
        previous=folder/'best_iterate.npz'
        if not previous.exists(): previous=folder/'witness.npz'
        print(json.dumps(dict(event='uncertified_seed',attempt=index,
                              residual=outcome['residual'],
                              status=outcome['status'],
                              next_seed=str(previous))),flush=True)
    save(status='not_certified',stage='seed_reoptimization',active=None)
    print("NO SOURCE CERTIFICATE: kept all saved trial evidence; no MPC launch.",flush=True)
    return 2


if __name__=="__main__":
    raise SystemExit(main())
