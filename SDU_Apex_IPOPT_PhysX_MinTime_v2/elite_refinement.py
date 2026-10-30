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

from command_ocp import (save_json_checkpoint, save_command_checkpoint,
                         witness_quality, quality_key)
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
                      geometry_seeds=(), control_seeds=(), diversity_threshold_m=.015,
                      max_elites=6, solve=None,
                      certify=None, stagnation_iterations=90,
                      cancel_requested=None, quality_objectives=(), on_candidate=None):
    """Search at unchanged duration for higher-clearance certified seeds."""
    if solve is None or certify is None:
        from command_ocp import solve_witness, certify_witness
        solve = solve or solve_witness
        certify = certify or certify_witness
    repo, seed, map_path, output = map(Path, (repo, seed, map_path, output))
    if not margin_steps and not geometry_seeds and not quality_objectives and not control_seeds:
        raise ValueError('at least one elite-search strategy required')
    if any(mode not in ('clearance', 'smooth') for mode in quality_objectives):
        raise ValueError('unknown equal-time quality objective')
    if not all(math.isfinite(d) and 0 < d <= .05 for d in margin_steps):
        raise ValueError('invalid strictly positive equal-time clearance targets')
    if not math.isfinite(diversity_threshold_m) or diversity_threshold_m <= 0:
        raise ValueError('invalid diversity threshold')
    if not isinstance(max_elites,int) or not 1 <= max_elites <= 16:
        raise ValueError('invalid elite count')
    for offset,harmonic in geometry_seeds:
        if not math.isfinite(offset) or not 0 < abs(offset) <= .5 or not 1 <= harmonic <= 8:
            raise ValueError('invalid alternate geometry seed')
    for amplitude,harmonic in control_seeds:
        from causal_seed import control_profile
        control_profile(40,amplitude,harmonic)
    count = source_count(seed)
    seed_hash = hashlib.sha256(seed.read_bytes()).hexdigest()
    map_hash = hashlib.sha256(map_path.read_bytes()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    path = output/'elite.json'
    contract = dict(seed_sha256=seed_hash, map_sha256=map_hash, intervals=count,
                    extra_margin_m=list(margin_steps),
                    quality_objectives=list(quality_objectives),
                    geometry_seeds=[list(entry) for entry in geometry_seeds],
                    control_seeds=[list(entry) for entry in control_seeds],
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
                     best_quality=witness_quality(seed, checked),
                     best_seed=str(seed), attempts=[], accepted=[])
        save_json_checkpoint(path,state)
    strategies = ([('objective',mode) for mode in quality_objectives] +
                  [('control',(amplitude,harmonic))
                   for amplitude,harmonic in control_seeds] +
                  [('margin',value) for value in margin_steps] +
                  [('geometry',(offset,harmonic)) for offset,harmonic in geometry_seeds])
    for index,(strategy,value) in enumerate(strategies):
        if cancel_requested is not None and cancel_requested():
            raise KeyboardInterrupt('equal-duration polishing interrupted')
        if any(a['index']==index for a in state['attempts']):
            continue
        if len(state['accepted'])>=max_elites:
            break
        target=.05+value if strategy=='margin' else .05
        objective_mode=value if strategy=='objective' else 'repair'
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
        if strategy=='control' and not (resume_seed is not None and resume_seed.exists()):
            initializer=output/f'causal_seed_{index:02d}.npz'
            if not initializer.exists():
                from causal_seed import source_alternative_seed
                diagnostics=source_alternative_seed(
                    repo,seed,initializer,*value,map_path=map_path)
                state.setdefault('control_seed_diagnostics',[]).append(
                    dict(index=index,**diagnostics))
                if diagnostics.get('status')=='rejected_initializer':
                    state['attempts'].append(dict(index=index,strategy=strategy,
                        status='rejected_initializer',
                        node_slack_m=diagnostics['initial_minimum_node_slack_m'],
                        reason='source-propagated initializer outside map preflight'))
                    save_json_checkpoint(path,state)
                    continue
                save_json_checkpoint(path,state)
        active_attempt=state.get('active_attempt') or {}
        restoring_this_index=(active_attempt.get('index')==index and
                              active_attempt.get('strategy')=='proposal_restoration')
        if not restoring_this_index:
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
                cancel_requested=cancel_requested, objective_mode=objective_mode)
        quality_directory=directory
        # Restore the same candidate against newly discovered 1 ms wall cuts.
        # Previously equal-time candidates were discarded here even when the
        # only missing constraints were sub-packet clearance samples.
        cuts=set()
        for repair in range(1, 4):
            if report['status']=='source_witness_passed':
                break
            rollout=report.get('rollout') or {}
            discovered={(int(c['interval']),int(c['tick'])) for c in
                        rollout.get('violating_source_ticks', [])}
            if (not discovered-cuts or report.get('max_constraint_violation',1.)>1.e-3
                    or rollout.get('full_lap_xy_max_m',1.)>.05):
                break
            cuts.update(discovered)
            initializer=directory/'witness.npz'
            directory=output/f'polish_{index:02d}_cut_{repair:02d}'
            result_file=directory/'report.json'
            if result_file.exists():
                report=json.loads(result_file.read_text())
                continue
            if directory.exists():
                if (directory/'best_iterate.npz').exists():
                    initializer=directory/'best_iterate.npz'
                from command_campaign import unused_directory
                directory=unused_directory(directory)
                result_file=directory/'report.json'
            state['active_attempt']=dict(index=index,strategy=strategy,output=str(directory),
                trial_lap_s=count*.025,margin=target,source_clearance_cuts=sorted(cuts))
            save_json_checkpoint(path,state)
            report=solve(repo=repo,checkpoint=None,mesh=None,audit=None,
                map_path=map_path,output=directory,count=count,
                iterations=iterations,cpu_seconds=1.e20,margin=target,
                command_seed=initializer,workers=workers,clearance_cuts=sorted(cuts),
                objective_mode=objective_mode,stagnation_iterations=stagnation_iterations,
                cancel_requested=cancel_requested)
        restoration_status=None
        proposal=quality_directory/'best_quality_proposal.npz'
        if strategy=='objective' and proposal.exists():
            # Objective runs retain a separate bounded near-feasible proposal.
            # Restore it with the source repair objective before deciding whether
            # it can enter the independently certified equal-time archive.
            restore_base=output/f'polish_{index:02d}_restore'
            restore_seed=proposal
            active=state.get('active_attempt') or {}
            resume_active=(active.get('index')==index and
                           active.get('strategy')=='proposal_restoration')
            restore_directory=Path(active['output']) if resume_active else restore_base
            restore_cuts={tuple(map(int,cut)) for cut in
                          active.get('source_clearance_cuts', [])} if resume_active else set()
            restore_report_path=restore_directory/'report.json'
            if restore_report_path.exists():
                restore_report=json.loads(restore_report_path.read_text())
            else:
                if restore_directory.exists():
                    saved=restore_directory/'best_iterate.npz'
                    if saved.exists():
                        restore_seed=saved
                    from command_campaign import unused_directory
                    restore_directory=unused_directory(restore_directory)
                    restore_report_path=restore_directory/'report.json'
                state['active_attempt']=dict(index=index,strategy='proposal_restoration',
                    output=str(restore_directory),proposal=str(proposal),
                    trial_lap_s=count*.025,margin=.05,
                    source_clearance_cuts=sorted(restore_cuts))
                save_json_checkpoint(path,state)
                restore_report=solve(repo=repo,checkpoint=None,mesh=None,audit=None,
                    map_path=map_path,output=restore_directory,count=count,
                    iterations=iterations,cpu_seconds=1.e20,margin=.05,
                    command_seed=restore_seed,workers=workers,
                    clearance_cuts=sorted(restore_cuts),
                    stagnation_iterations=stagnation_iterations,
                    cancel_requested=cancel_requested,objective_mode='repair')
            for repair in range(1,4):
                if restore_report['status']=='source_witness_passed':
                    break
                rollout=restore_report.get('rollout') or {}
                discovered={(int(c['interval']),int(c['tick'])) for c in
                            rollout.get('violating_source_ticks', [])}
                if (not discovered-restore_cuts or
                        restore_report.get('max_constraint_violation',1.)>1.e-3 or
                        rollout.get('full_lap_xy_max_m',1.)>.05):
                    break
                restore_cuts.update(discovered)
                restore_seed=restore_directory/'witness.npz'
                next_directory=output/f'polish_{index:02d}_restore_cut_{repair:02d}'
                next_report=next_directory/'report.json'
                if next_report.exists():
                    restore_report=json.loads(next_report.read_text())
                    restore_directory=next_directory
                    continue
                if next_directory.exists():
                    saved=next_directory/'best_iterate.npz'
                    if saved.exists():
                        restore_seed=saved
                    from command_campaign import unused_directory
                    next_directory=unused_directory(next_directory)
                restore_directory=next_directory
                state['active_attempt']=dict(index=index,strategy='proposal_restoration',
                    output=str(restore_directory),proposal=str(proposal),
                    trial_lap_s=count*.025,margin=.05,
                    source_clearance_cuts=sorted(restore_cuts))
                save_json_checkpoint(path,state)
                restore_report=solve(repo=repo,checkpoint=None,mesh=None,audit=None,
                    map_path=map_path,output=restore_directory,count=count,
                    iterations=iterations,cpu_seconds=1.e20,margin=.05,
                    command_seed=restore_seed,workers=workers,
                    clearance_cuts=sorted(restore_cuts),
                    stagnation_iterations=stagnation_iterations,
                    cancel_requested=cancel_requested,objective_mode='repair')
            restoration_status=restore_report.get('status')
            if restoration_status=='source_witness_passed':
                report=restore_report
                directory=restore_directory
        attempt = dict(index=index, strategy=strategy, required_margin_m=target,
                       objective_mode=objective_mode,
                       ipopt_status=report.get('ipopt_status'),
                       status=report['status'], report=str(directory/'report.json'),
                       proposal_restoration_status=restoration_status)
        if report['status']=='source_witness_passed':
            candidate_hash=hashlib.sha256((directory/'witness.npz').read_bytes()).hexdigest()
            certificate_dir=output/f'certified_{index:02d}_{candidate_hash[:12]}'
            independent=certificate_dir/'report.json'
            if independent.exists():
                checked=json.loads(independent.read_text())
            else:
                if certificate_dir.exists():
                    from command_campaign import unused_directory
                    certificate_dir=unused_directory(certificate_dir)
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
                quality=witness_quality(alternative,checked)
                prior_quality=state.get('best_quality') or witness_quality(seed,
                    json.loads(seed.with_name('report.json').read_text()))
                smoother=quality['steering_step_rms_rad'] < prior_quality['steering_step_rms_rad']-1.e-4
                attempt['geometry_diversity_m']=diversity
                if diversity>=diversity_threshold_m or better_clearance or smoother:
                    record=dict(witness=str(alternative),
                        sha256=hashlib.sha256(alternative.read_bytes()).hexdigest(),
                        clearance_slack_m=score, lap_s=count*.025,
                        geometry_diversity_m=diversity,
                        required_margin_m=target,strategy=strategy,quality=quality)
                    state['accepted'].append(record)
                    if quality_key(quality)>quality_key(prior_quality):
                        state['best_seed']=str(alternative)
                        state['best_clearance_slack_m']=score
                        state['best_quality']=quality
                    # Export immediately; a same-time alternative must not wait
                    # for every other IPOPT restart before Unity can test it.
                    if on_candidate is not None:
                        on_candidate(alternative,count*.025)
        state['attempts'].append(attempt)
        state.pop('active_attempt',None)
        save_json_checkpoint(path,state)
    state['complete']=True
    save_json_checkpoint(path,state)
    return state
