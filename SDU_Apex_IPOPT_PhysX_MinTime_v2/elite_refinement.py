"""Durable equal-duration source-witness polishing and clearance-ranked archive.

The racing time stays fixed. Every accepted witness must pass the independent
1 ms source-model verifier. This is not a Unity/MPC runtime certificate.
"""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np

from pathlib import Path

from command_ocp import save_json_checkpoint, save_command_checkpoint
from command_campaign import source_count


def quality_of_certificate(report):
    if report.get('status') != 'source_witness_passed':
        return None
    margin = (report.get('rollout') or {}).get('minimum_1ms_footprint_slack_m')
    if margin is None or not math.isfinite(float(margin)) or float(margin) < 0.:
        return None
    return float(margin)


def geometry_seed(seed, destination, lateral_offset_m, harmonic):
    """Uncertified, smoothly displaced geometry; source hidden states unchanged.

    This is an alternate local basin, NOT an integration and NOT a certificate.
    IPOPT and the independent full-tick gate must restore realizability.
    """
    if not math.isfinite(lateral_offset_m) or not 0 < abs(lateral_offset_m) <= .5:
        raise ValueError('bad lateral perturbation')
    if not 1 <= harmonic <= 8:
        raise ValueError('bad geometry harmonic')
    with np.load(seed, allow_pickle=False) as data:
        states=np.asarray(data['state'],dtype=float).copy()
        controls=np.asarray(data['controls'],dtype=float).copy()
        digest=str(data['source_sha256'].item())
    count=controls.shape[1]
    theta=2*np.pi*np.arange(count+1)/count
    profile=lateral_offset_m*np.sin(harmonic*theta)
    heading=states[2].copy()
    states[0] -= profile*np.sin(heading)
    states[1] += profile*np.cos(heading)
    # Enforce precisely the periodic closure already present in the seed.
    states[0,-1]=states[0,0]
    states[1,-1]=states[1,0]
    destination.parent.mkdir(parents=True,exist_ok=True)
    save_command_checkpoint(destination,states,controls,digest,float('inf'),-1)
    return destination


def state_diversity_m(a,b):
    """RMS XY displacement; trajectories share the same fixed phase and count."""
    with np.load(a,allow_pickle=False) as aa, np.load(b,allow_pickle=False) as bb:
        xa=np.asarray(aa['state'][:2,:-1],dtype=float)
        xb=np.asarray(bb['state'][:2,:-1],dtype=float)
        if xa.shape != xb.shape:
            return float('inf')
        return float(np.sqrt(np.mean(np.sum((xa-xb)**2,axis=0))))


def polish_equal_time(*, repo, seed, map_path, output, iterations=300, workers=4,
                      margin_steps=(.0001, .0005, .002),
                      geometry_seeds=(), diversity_threshold_m=.015,
                      max_elites=6, solve=None,
                      certify=None, stagnation_iterations=90,
                      cancel_requested=None):
    """Search at unchanged duration for higher-clearance certified seeds."""
    if solve is None or certify is None:
        from command_ocp import solve_witness, certify_witness
        solve = solve or solve_witness
        certify = certify or certify_witness
    repo, seed, map_path, output = map(Path, (repo, seed, map_path, output))
    if not margin_steps and not geometry_seeds:
        raise ValueError('at least one elite-search strategy required')
    if not all(math.isfinite(d) and 0 < d <= .05 for d in margin_steps):
        raise ValueError('invalid strictly positive equal-time clearance targets')
    if not math.isfinite(diversity_threshold_m) or diversity_threshold_m <= 0:
        raise ValueError('invalid diversity threshold')
    if not isinstance(max_elites,int) or not 1 <= max_elites <= 16:
        raise ValueError('invalid elite count')
    for offset,harmonic in geometry_seeds:
        if not math.isfinite(offset) or not 0 < abs(offset) <= .5 or not 1 <= harmonic <= 8:
            raise ValueError('invalid alternate geometry seed')
    count = source_count(seed)
    seed_hash = hashlib.sha256(seed.read_bytes()).hexdigest()
    map_hash = hashlib.sha256(map_path.read_bytes()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    path = output/'elite.json'
    contract = dict(seed_sha256=seed_hash, map_sha256=map_hash, intervals=count,
                    extra_margin_m=list(margin_steps),
                    geometry_seeds=[list(entry) for entry in geometry_seeds],
                    diversity_threshold_m=diversity_threshold_m,max_elites=max_elites)
    if path.exists():
        state = json.loads(path.read_text())
        if state['contract'] != contract:
            raise ValueError('elite archive input contract mismatch')
    else:
        baseline = seed.with_name('report.json')
        if not baseline.exists():
            raise ValueError('elite archive requires certified input witness/report')
        checked = json.loads(baseline.read_text())
        score = quality_of_certificate(checked)
        if score is None or abs(float(checked['trial_lap_s'])-count*.025)>1.e-8:
            raise ValueError('elite baseline did not pass source and clearance gates')
        state = dict(schema='source-elite-refinement/v1', contract=contract,
                     baseline_clearance_slack_m=score, best_clearance_slack_m=score,
                     best_seed=str(seed), attempts=[], accepted=[])
        save_json_checkpoint(path,state)
    strategies = ([('margin',value) for value in margin_steps] +
                  [('geometry',(offset,harmonic)) for offset,harmonic in geometry_seeds])
    for index,(strategy,value) in enumerate(strategies):
        if cancel_requested is not None and cancel_requested():
            raise KeyboardInterrupt('equal-duration polishing interrupted')
        if any(a['index']==index for a in state['attempts']):
            continue
        if len(state['accepted'])>=max_elites:
            break
        target=.05+value if strategy=='margin' else .05
        directory=output/f'polish_{index:02d}'
        if directory.exists() and not (directory/'report.json').exists():
            # CasADi/IPOPT stops are durable, but a partially written output
            # cannot be overwritten. Resume the saved primal in a new folder.
            previous=directory
            resume_seed=previous/'best_iterate.npz'
            directory=output/f'polish_{index:02d}_retry_001'
            attempt_no=1
            while directory.exists() and not (directory/'report.json').exists():
                attempt_no+=1
                directory=output/f'polish_{index:02d}_retry_{attempt_no:03d}'
            state.setdefault('retry_history',[]).append(str(previous))
        else:
            resume_seed=None
        result_file=directory/'report.json'
        initializer=Path(state['best_seed'])
        if resume_seed is not None and resume_seed.exists():
            initializer=resume_seed
        if strategy=='geometry' and not (resume_seed is not None and resume_seed.exists()):
            initializer=output/f'geometry_seed_{index:02d}.npz'
            if not initializer.exists():
                geometry_seed(seed,initializer,*value)
        state['active_attempt']=dict(index=index,strategy=strategy,output=str(directory),
                                     trial_lap_s=count*.025,margin=target)
        save_json_checkpoint(path,state)
        if result_file.exists():
            report=json.loads(result_file.read_text())
        else:
            print(json.dumps(dict(event='solve', output=str(directory),
                                  key=f'elite_{index:02d}_{strategy}',refinement=0,
                                  target_lap_s=count*.025,source_only=True)),flush=True)
            report=solve(repo=repo,checkpoint=None,mesh=None,audit=None,
                map_path=map_path,output=directory,count=count,
                iterations=iterations,cpu_seconds=1.e20,margin=target,
                command_seed=initializer,workers=workers,
                stagnation_iterations=stagnation_iterations,
                cancel_requested=cancel_requested)
        attempt = dict(index=index, strategy=strategy, required_margin_m=target,
                       ipopt_status=report.get('ipopt_status'),
                       status=report['status'], report=str(result_file))
        if report['status']=='source_witness_passed':
            certificate_dir=output/f'certified_{index:02d}'
            independent=certificate_dir/'report.json'
            if independent.exists():
                checked=json.loads(independent.read_text())
            else:
                checked=certify(repo, directory/'witness.npz',map_path,certificate_dir)
            score=quality_of_certificate(checked)
            if abs(float(checked.get('trial_lap_s',float('inf')))-count*.025)>1.e-8:
                score=None
            attempt['independent_status']=checked['status']
            attempt['clearance_slack_m']=score
            if score is not None:
                alternative=certificate_dir/'witness.npz'
                current=[seed]+[Path(e['witness']) for e in state['accepted']]
                diversity=min(state_diversity_m(alternative,other) for other in current)
                better_clearance=score>state['best_clearance_slack_m']+1.e-6
                attempt['geometry_diversity_m']=diversity
                if diversity>=diversity_threshold_m or better_clearance:
                    record=dict(witness=str(alternative),
                        sha256=hashlib.sha256(alternative.read_bytes()).hexdigest(),
                        clearance_slack_m=score, lap_s=count*.025,
                        geometry_diversity_m=diversity,
                        required_margin_m=target,strategy=strategy)
                    state['accepted'].append(record)
                    if better_clearance:
                        state['best_seed']=str(alternative)
                        state['best_clearance_slack_m']=score
        state['attempts'].append(attempt)
        state.pop('active_attempt',None)
        save_json_checkpoint(path,state)
    state['complete']=True
    save_json_checkpoint(path,state)
    return state
