"""Resumable, source-checked raceline search; never auto-promote runtime inputs.

IPOPT solves the production-command OCP. Recursive source replay discovers
wall cuts, then IPOPT repairs the SAME saved primal. Shorter durations and
cyclic starting phases provide new initializations, not new vehicle fits.
Plateau is an honest result, not evidence of a global vehicle-time floor.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import time

import numpy as np

from command_ocp import (certify_witness, load_command_seed, production_command_map,
                         save_command_checkpoint, save_json_checkpoint, solve_witness,
                         solve_time_predictor, validate_margin)


def unused_directory(path):
    """Never overwrite an incomplete certification after interruption."""
    if not path.exists(): return path
    index = 1
    while path.with_name(path.name+f'_{index:03d}').exists(): index += 1
    return path.with_name(path.name+f'_{index:03d}')


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def phase_seed(path, phase, destination):
    """Rotate complete packets, preserving physical history and yaw winding.

    No dynamics are changed or certified here. The unchanged closing state is
    reconstructed from the selected phase, not interpolated across the yaw seam.
    """
    with np.load(path, allow_pickle=False) as data:
        count = data['controls'].shape[1]
    x, u, winding, _ = load_command_seed(path, count)
    if not 0 <= phase < count: raise ValueError('invalid packet phase')
    indices = np.arange(count+1)+phase
    state = x[:, indices % count].copy()
    state[2] += indices//count*(winding*2*math.pi)
    controls = np.roll(u, -phase, axis=1)
    save_command_checkpoint(destination, state, controls, 'uncertified-phase-initializer',
                            0., -1)
    return destination


def source_count(path):
    with np.load(path, allow_pickle=False) as data:
        return data['controls'].shape[1]


def trial_counts(count, fraction, minimum):
    """Coarse-to-fine duration trials; always include the one-packet step."""
    if not 0. <= fraction < 1. or minimum < 2:
        raise ValueError('invalid duration search domain')
    step = max(1, int(count*fraction))
    result = []
    while True:
        candidate = max(minimum, count-step)
        if candidate < count and candidate not in result:
            result.append(candidate)
        if step == 1:
            return result
        step = max(1, step//2)


def search_starts(counts, rotations, old_count, include_packet_starts):
    """Back off duration before rephasing the same unsuccessful large jump.

    Near the incumbent, removing one packet preserves every other exact source
    transition. Try that initializer before globally resampling hidden states.
    All configured phases/durations are still visited if none succeeds.
    """
    return [(count, rotation, strategy) for rotation in rotations for count in counts
            for strategy in (('drop-packet', 'resample')
                if include_packet_starts and count == old_count-1 else ('resample',))]


def budget_increase_only(previous, requested):
    """Changing patience/iteration ceilings cannot transfer a physics certificate."""
    keys = ('iterations', 'stagnation_iterations')
    a = {k: v for k, v in previous.items() if k not in keys}
    b = {k: v for k, v in requested.items() if k not in keys}
    return a == b and all(requested.get(k, 50) >= previous.get(k, 50) for k in keys)


def quality_search_update_only(previous, requested):
    """Migrate search objectives while requiring identical map/source/seed inputs."""
    keys = ('command_objective_scaling', 'equal_time_search')
    return (any(previous.get(k) != requested.get(k) for k in keys) and
            requested.get('equal_time_search') in ('clearance-smooth-v3','causal-control-basin-v4') and
            budget_increase_only({k:v for k,v in previous.items() if k not in keys},
                                 {k:v for k,v in requested.items() if k not in keys}))


def reopen_budget_limited_attempts(state):
    """Give stopped primals a real retry after increasing exploration budgets."""
    retained = []
    for attempt in state['attempts']:
        path = Path(attempt['report'])
        report = json.loads(path.read_text()) if path.exists() else {}
        limited = (report.get('termination_reason') == 'feasibility_stagnation' or
                   report.get('ipopt_status') == 'Maximum_Iterations_Exceeded')
        if not attempt['certified'] and limited:
            seed = path.with_name('best_iterate.npz')
            if not seed.exists():
                seed = path.with_name('witness.npz')
            if seed.exists():
                state.setdefault('budget_limited_history', []).append(attempt)
                state.setdefault('retry_starts', {})[attempt['key']] = dict(
                    seed=str(seed), cuts=report.get('source_clearance_cuts', []))
                continue
        retained.append(attempt)
    state['attempts'] = retained


def run_campaign(*, repo, seed, map_path, output, target_seconds=4.7,
                 phases=8, workers=8, iterations=1500, stagnation_iterations=250,
                 refinements=6, resume=False, include_packet_starts=True,
                 solve=solve_witness, certify=certify_witness, digest=None,
                 cancel_requested=None, coarse_fraction=0., on_improvement=None,
                 hybrid_time=False, predict=solve_time_predictor,
                 elite_refinements=0, elite_polish=None, elite_first=False,
                 margin=.05, time_sectors=1):
    repo, seed, map_path, output = [Path(p).resolve() for p in (repo, seed, map_path, output)]
    margin = validate_margin(margin)
    if not isinstance(time_sectors,int) or not 1 <= time_sectors <= 32:
        raise ValueError('invalid time predictor sector count')
    if ((target_seconds is not None and
            (not math.isfinite(target_seconds) or target_seconds < .75)) or phases < 1
            or not math.isfinite(coarse_fraction) or not 0. <= coarse_fraction < 1.
            or refinements < 0 or iterations < 1 or stagnation_iterations < 5
            or elite_refinements < 0 or elite_refinements > 3):
        raise ValueError('invalid campaign target or search settings')
    if digest is None:
        _, digest = production_command_map(repo, repo/'build_task/command_ocp')
    contract = dict(seed_sha256=file_hash(seed), map_sha256=file_hash(map_path),
        source_sha256=digest, target_intervals=(None if target_seconds is None else
            math.floor(target_seconds/.025+1.e-9)),
        phases=phases, workers=workers, iterations=iterations,
        stagnation_iterations=stagnation_iterations, refinements=refinements,
        include_packet_starts=include_packet_starts, coarse_fraction=coarse_fraction,
        required_margin_m=margin,
        map_metadata_sha256=(file_hash(map_path.with_suffix('.yaml'))
                             if map_path.with_suffix('.yaml').exists() else None))
    contract['elite_refinements'] = elite_refinements
    contract['elite_first'] = bool(elite_first)
    contract['start_order'] = 'duration-before-phase/drop-first-v1'
    contract['command_objective_scaling'] = 'periodic-regularizer-v2'
    contract['equal_time_search'] = 'causal-control-basin-v4'
    if hybrid_time:
        contract['time_predictor'] = ('source-24-25ms-sector-chord/v2'
                                      if time_sectors>1 else 'source-24-25ms-chord/v1')
        contract['time_sectors']=time_sectors
    state_path = output/'campaign.json'
    if output.exists() and not resume:
        raise FileExistsError('campaign exists; use --resume, never overwrite it')
    output.mkdir(parents=True, exist_ok=True)
    lock = (output/'.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = None

    def save():
        state['updated_epoch_s'] = time.time()
        save_json_checkpoint(state_path, state)

    def event(name, **values):
        print(json.dumps(dict(event=name, **values)), flush=True)

    # Keep the historical injected-call contract for the default margin. A
    # nondefault campaign margin must be explicit at every physics gate.
    margin_kwargs = {} if margin == .05 else {'margin': margin}

    def certify_with_margin(*args):
        return certify(*args, **margin_kwargs)

    def solve_with_margin(**kwargs):
        kwargs.update(margin_kwargs)
        return solve(**kwargs)

    def time_prediction(incumbent, old_count):
        """One time-objective proposal per incumbent, always followed by exact repair."""
        ident = file_hash(incumbent)
        stored = state.setdefault('time_predictions', {})
        if ident in stored:
            return stored[ident]
        active = state.get('active_predictor')
        parent = output/f'time_from{old_count:03d}'
        destination = unused_directory(parent/'solve')
        resume_checkpoint = None
        if active and active['incumbent_sha256'] == ident:
            previous = Path(active['output'])
            if (previous/'report.json').exists():
                stored[ident] = json.loads((previous/'report.json').read_text())
                state.pop('active_predictor', None)
                save()
                return stored[ident]
            for name in ('best_time_candidate.npz', 'best_predictor.npz'):
                if (previous/name).exists():
                    resume_checkpoint = previous/name
                    break
        state['active_predictor'] = dict(incumbent_sha256=ident,
            output=str(destination), seed=str(incumbent))
        save()
        event('minimum_time_prediction', source_lap_s=old_count*.025,
              output=str(destination), source_certified=False)
        report = predict(repo=repo, command_seed=incumbent, map_path=map_path,
            # The hybrid-time objective is a complementary search basin,
            # not an expendable 200/80 trial. Give it the same per-solve budget
            # as fixed 25 ms and equal-time IPOPT.
            output=destination, workers=workers, iterations=iterations,
            stagnation_iterations=stagnation_iterations,
            cancel_requested=cancel_requested, resume_checkpoint=resume_checkpoint,
            time_sectors=time_sectors, **margin_kwargs)
        stored[ident] = report
        state.pop('active_predictor', None)
        save()
        event('time_prediction_finished', status=report['status'],
              predicted_lap_s=report.get('predicted_lap_s'),
              trial_lap_s=report.get('trial_lap_s'), source_certified=False)
        return report

    def explore_elites(incumbent):
        """Run alternative same-time seed search from a certified incumbent."""
        improved=False
        if not elite_refinements:
            return False
        # Escape a single-basin plateau without weakening the primary
        # min-time objective. Equal-time candidates are only used as
        # independently certified alternative search seeds; they do
        # NOT silently replace the fastest published witness.
        from elite_refinement import polish_equal_time
        baseline_hash = file_hash(incumbent)
        exploration_key = baseline_hash+':'+contract['equal_time_search']
        explored = state.setdefault('elite_explored', [])
        if exploration_key not in explored:
            elite_dir = output/'elite_causal_v4'/baseline_hash[:16]
            state['active_elite_polish'] = dict(seed=str(incumbent),
                output=str(elite_dir), source_lap_s=source_count(incumbent)*.025)
            save()
            event('elite_polish_started', source_lap_s=source_count(incumbent)*.025,
                output=str(elite_dir))
            archive = (elite_polish or polish_equal_time)(
                repo=repo, seed=incumbent, map_path=map_path,
                output=elite_dir, workers=workers,
                # Give equal-time feasibility restoration the same budget as
                # primary 25 ms solves (e.g. 1200/300). Independent attempts
                # remain sequential and independently certified.
                iterations=iterations,
                margin_steps=(.0001, .0005, .002)[:elite_refinements],
                quality_objectives=('clearance','smooth'),
                # Better-conditioned, source-propagated changes before XY-only
                # variations. Every candidate still needs original source
                # feasibility and independent 1 ms footprint certification.
                control_seeds=((.002,1),(-.002,1),(.004,2),(-.004,2),(.008,1)),
                on_candidate=on_improvement,
                # Explore other local geometry basins at exactly the
                # incumbent duration, in addition to improving clearance.
                # Both child minimizers run with EXACT 25 ms source maps
                # and with their own hybrid-time proposals.
                geometry_seeds=((.04, 1), (-.04, 1),
                                (.09, 2), (-.09, 2), (.14, 3)),
                max_elites=5, diversity_threshold_m=.015,
                stagnation_iterations=stagnation_iterations,
                cancel_requested=cancel_requested,
                solve=solve_with_margin, certify=certify_with_margin)
            state.setdefault('elite_archives', {})[baseline_hash] = str(
                elite_dir/'elite.json')
            state.pop('active_elite_polish',None)
            save()
            for candidate in archive['accepted']:
                if cancel_requested is not None and cancel_requested():
                    raise KeyboardInterrupt
                child_dir = output/'elite_campaigns'/candidate['sha256'][:16]
                state['active_elite'] = dict(
                    seed=candidate['witness'], sha256=candidate['sha256'],
                    lap_s=candidate['lap_s'], clearance_slack_m=candidate['clearance_slack_m'],
                    output=str(child_dir))
                save()
                event('elite_branch_started', seed=candidate['witness'],
                      lap_s=candidate['lap_s'], output=str(child_dir),
                      clearance_slack_m=candidate['clearance_slack_m'])
                child = run_campaign(
                    repo=repo, seed=Path(candidate['witness']),
                    map_path=map_path, output=child_dir,
                    target_seconds=target_seconds,
                    phases=phases, workers=workers,
                    iterations=iterations,
                    stagnation_iterations=stagnation_iterations,
                    refinements=refinements,
                    resume=(child_dir/'campaign.json').exists(),
                    include_packet_starts=include_packet_starts,
                    solve=solve, certify=certify, digest=digest,
                    cancel_requested=cancel_requested,
                    coarse_fraction=coarse_fraction,
                    on_improvement=on_improvement,
                    hybrid_time=hybrid_time, predict=predict,
                    elite_refinements=0, elite_first=False, margin=margin,
                    time_sectors=time_sectors)
                if child['best_lap_s'] < state['best_lap_s']-1.e-9:
                    state['best_lap_s'] = child['best_lap_s']
                    state['best_witness'] = child['best_witness']
                    improved = True
                    save()
                    event('elite_faster_source_witness',
                          lap_s=state['best_lap_s'],
                          seed=candidate['witness'],
                          witness=state['best_witness'])
            state.pop('active_elite',None)
            explored.append(exploration_key)
            save()
        return improved

    try:
        if resume:
            state = json.loads(state_path.read_text())
            previous_contract = dict(state['contract'])
            # Campaigns created before explicit margin tracking used the same
            # 5 cm default. Treat that absent field as the historical default.
            previous_contract.setdefault('required_margin_m', .05)
            if previous_contract != contract:
                objective_update = quality_search_update_only(previous_contract, contract)
                if not (budget_increase_only(previous_contract, contract) or objective_update):
                    raise ValueError('campaign source/map/settings changed; create a new campaign')
                if objective_update:
                    state.setdefault('search_updates', []).append(dict(
                        previous=state['contract'], requested=contract, epoch_s=time.time()))
                    state['elite_first_done'] = False
                    if state['status']=='plateau':
                        state['status']='running'
                state.setdefault('budget_updates', []).append(dict(
                    previous=state['contract'], requested=contract, epoch_s=time.time()))
                state['contract'] = contract
                reopen_budget_limited_attempts(state)
                if state.get('retry_starts') and state['status'] == 'plateau':
                    state['status'] = 'running'
                save()
                event('budgets_increased', iterations=iterations,
                      stagnation_iterations=stagnation_iterations,
                      reopened=len(state.get('retry_starts', {})))
            elif state['contract'] != contract:
                # Persist the explicit default while preserving the validity
                # of campaigns written before the margin field existed.
                state['contract'] = contract
                save()
            if state['status'] in ('target_reached', 'plateau', 'search_domain_reached'):
                event('already_finished', status=state['status'], best_lap_s=state['best_lap_s'])
                return state
            state['status'] = 'running' if state['status'] != 'initializing' else 'initializing'
        else:
            # Baseline is rechecked by current code; a saved label is not a
            # certificate. No search starts from an unvalidated incumbent.
            state = dict(schema='sdu-apex-command-campaign/v1', contract=contract,
                status='initializing', best_witness=None, best_lap_s=None,
                attempts=[], active=None, source_only=True,
                unity_validated=False, runtime_promoted=False)
            save()
        if state['status'] == 'initializing':
            baseline_dir = unused_directory(output/'baseline')
            baseline = certify(repo, seed, map_path, baseline_dir, **margin_kwargs)
            if baseline['status'] != 'source_witness_passed':
                raise ValueError('initial witness fails current source gates')
            state.update(status='running', best_witness=str(baseline_dir/'witness.npz'),
                         best_lap_s=baseline['trial_lap_s'])
        save()
        if on_improvement is not None:
            on_improvement(Path(state['best_witness']), state['best_lap_s'])
        if elite_first and elite_refinements and not state.get('elite_first_done'):
            event('elite_first_search', best_lap_s=state['best_lap_s'])
            explore_elites(Path(state['best_witness']))
            state['elite_first_done']=True
            save()
        while True:
            if cancel_requested is not None and cancel_requested():
                raise KeyboardInterrupt
            incumbent = Path(state['best_witness'])
            old_count = source_count(incumbent)
            # 30 intervals is a numerical search-domain guard, not a physical
            # vehicle-time floor. No requested lap target stops an open search.
            minimum = contract['target_intervals'] or 30
            counts = trial_counts(old_count, coarse_fraction, minimum)
            if not counts:
                state.update(status=('target_reached' if target_seconds is not None
                                     else 'search_domain_reached'), active=None)
                save()
                break
            rotations = sorted(set(int(source_count(incumbent)*i/phases) for i in range(phases)))
            improved = False
            starts = search_starts(counts, rotations, old_count, include_packet_starts)
            prediction = state.get('time_predictions', {}).get(file_hash(incumbent))
            if hybrid_time and state.get('active_predictor'):
                prediction = time_prediction(incumbent, old_count)
            if prediction and prediction.get('status') == 'uncertified_time_proposal':
                suggested = max(minimum, prediction['proposed_intervals'])
                if suggested < old_count:
                    starts.insert(0, (suggested, 0, 'time-predictor'))
            active_key = (state.get('active') or {}).get('key')
            if active_key:
                # Increasing budgets can reopen an EARLIER phase. Finish the
                # interrupted phase first, rather than overwrite its context.
                starts.sort(key=lambda item:
                    f'from{old_count:03d}_{item[0]:03d}_phase{item[1]:03d}_{item[2]}' != active_key)
            for start_index, (count, rotation, strategy) in enumerate(starts):
                key = f'from{old_count:03d}_{count:03d}_phase{rotation:03d}_{strategy}'
                if any(a['key'] == key for a in state['attempts']): continue
                directory = output/key
                directory.mkdir(exist_ok=True)
                active = state.get('active')
                if active is not None and active['key'] == key:
                    cuts = {tuple(c) for c in active['cuts']}
                    previous = Path(active['output'])
                    if (previous/'report.json').exists():
                        report = json.loads((previous/'report.json').read_text())
                        start = previous/'witness.npz'
                    elif (previous/'best_iterate.npz').exists():
                        report = None
                        start = previous/'best_iterate.npz'
                    else:
                        report, start = None, Path(active['seed'])
                    refinement = active['refinement']
                elif key in state.get('retry_starts', {}):
                    retry = state['retry_starts'][key]
                    start = Path(retry['seed'])
                    cuts = {tuple(c) for c in retry['cuts']}
                    report, refinement = None, 0
                else:
                    start = (Path(prediction['proposed_seed'])
                             if strategy == 'time-predictor' else incumbent)
                    if rotation:
                        start = phase_seed(incumbent, rotation, directory/'phase_seed.npz')
                    cuts, report, refinement = set(), None, 0
                while True:
                    if report is None:
                        index = 0
                        while (directory/f'solve_{index:03d}').exists(): index += 1
                        destination = directory/f'solve_{index:03d}'
                        state['active'] = dict(key=key, output=str(destination), seed=str(start),
                            cuts=sorted(cuts), refinement=refinement)
                        save()
                        event('solve', key=key, refinement=refinement,
                              target_lap_s=count*.025, output=str(destination))
                        report = solve(repo=repo, checkpoint=None, mesh=None, audit=None,
                            map_path=map_path, output=destination, count=count,
                            iterations=iterations, cpu_seconds=1.e20, command_seed=start,
                            workers=workers, clearance_cuts=sorted(cuts),
                            shorten_packet=strategy == 'drop-packet' and source_count(start) == count+1,
                            recondition_seed=hybrid_time and source_count(start) != count,
                            stagnation_iterations=stagnation_iterations,
                            cancel_requested=cancel_requested, **margin_kwargs)
                        start = destination/'witness.npz'
                    if report['status'] != 'source_witness_passed':
                        # A source-correct, single-picometre wall shortfall is
                        # a numerical boundary issue, NOT evidence of a
                        # physical collision. Try a rigid SE(2) perturbation,
                        # then independently rerun the unchanged source
                        # dynamics and 1ms perimeter validator.
                        from rigid_pose_polish import recover_numerical_near_miss
                        if (report.get('rollout') and
                                report.get('max_constraint_violation',float('inf'))<=1e-5 and
                                -1e-6 <= report['rollout'].get(
                                    'minimum_1ms_footprint_slack_m',-float('inf')) < 0.):
                            rigid_dir=unused_directory(directory/'rigid_recovery')
                            rigid=recover_numerical_near_miss(
                                repo,start,map_path,rigid_dir,margin=margin,certify=certify)
                            if rigid is not None:
                                start=Path(rigid['witness'])
                                report=json.loads(Path(rigid['certificate']).read_text())
                                event('rigid_near_miss_certified',key=key,
                                      lap_s=report['trial_lap_s'],
                                      numerical_only=rigid['numerical_only'],
                                      shifted_slack_m=rigid['shifted_slack_m'])
                    if report['status'] == 'source_witness_passed':
                        destination = unused_directory(directory/'certified')
                        checked = certify(repo, start, map_path, destination,
                                          **margin_kwargs)
                        if checked['status'] == 'source_witness_passed':
                            state.update(best_witness=str(destination/'witness.npz'),
                                best_lap_s=checked['trial_lap_s'])
                            improved = True
                            save()
                            if on_improvement is not None:
                                on_improvement(Path(state['best_witness']), state['best_lap_s'])
                            event('faster_source_witness', lap_s=state['best_lap_s'],
                                  witness=state['best_witness'], unity_validated=False)
                        break
                    discovered = {(int(c['interval']), int(c['tick'])) for c in
                                  report['rollout']['violating_source_ticks']}
                    if (report['max_constraint_violation'] <= 1.e-3 and
                            report['rollout']['full_lap_xy_max_m'] <= .05 and
                            refinement < refinements and discovered-cuts):
                        cuts.update(discovered)
                        refinement += 1
                        report = None
                        continue
                    # A close dynamic residual can need repair even when
                    # no wall cut is missing. Do not confuse "no new wall
                    # cuts" with full feasibility or numerical stagnation.
                    if (report.get('ipopt_status') in ('Maximum_Iterations_Exceeded', 'User_Requested_Stop')
                            and report['max_constraint_violation'] < report.get('initial_residual', 0.)*.5
                            and report['max_constraint_violation'] > 1.e-5):
                        report = None
                        event('continue_improving_primal', key=key)
                        continue
                    break
                state['attempts'].append(dict(key=key, status=report['status'],
                    report=str(start.with_name('report.json')),
                    residual=report.get('max_constraint_violation'), certified=improved))
                state['active'] = None
                state.get('retry_starts', {}).pop(key, None)
                save()
                if improved: break
                if hybrid_time and prediction is None:
                    prediction = time_prediction(incumbent, old_count)
                    if prediction.get('status') == 'uncertified_time_proposal':
                        suggested = max(minimum, prediction['proposed_intervals'])
                        if suggested < old_count:
                            starts.insert(start_index+1, (suggested, 0, 'time-predictor'))
            if not improved:
                improved=explore_elites(incumbent)
            if not improved:
                state.update(status='plateau', active=None)
                save()
                event('plateau', best_lap_s=state['best_lap_s'],
                      note='All configured starts tried; not a physical/global lower bound.')
                break
        event('finished', status=state['status'], best_lap_s=state['best_lap_s'])
        return state
    except BaseException:
        if state is not None and state['status'] == 'running':
            state['status'] = 'interrupted'
            save()
        raise
    finally:
        lock.close()


def show_status(output):
    state = json.loads((output/'campaign.json').read_text())
    best = f"{state['best_lap_s']:.6f} s" if state['best_lap_s'] is not None else 'baseline validation pending'
    print(f"Campaign: {state['status']} | best source lap {best}")
    print(f"Completed starts: {len(state['attempts'])} | runtime changed: no")
    active = state.get('active')
    if active:
        print(f"Active: {active['key']} | clearance pass {active['refinement']}")
        checkpoint = Path(active['output'])/'best_iterate.npz'
        if checkpoint.exists():
            with np.load(checkpoint, allow_pickle=False) as data:
                print(f"Durable best: iteration {int(data['iteration'])} | residual "
                      f"{float(data['max_constraint_violation']):.6g}")
        print(f"Output: {active['output']}")
    print(f"Best witness: {state['best_witness']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--status', type=Path, help='Read-only campaign status; no solve')
    parser.add_argument('--seed', type=Path)
    parser.add_argument('--map', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--target-seconds', type=float, default=None,
                        help='Optional stop target; default searches until configured starts plateau')
    parser.add_argument('--coarse-fraction', type=float, default=.1,
                        help='Try coarse duration reductions before single-packet refinement')
    parser.add_argument('--phases', type=int, default=8)
    parser.add_argument('--workers', type=int, default=os.cpu_count() or 1)
    parser.add_argument('--iterations', type=int, default=1500)
    parser.add_argument('--stagnation-iterations', type=int, default=250)
    parser.add_argument('--refinements', type=int, default=6)
    parser.add_argument('--margin', type=float, default=.05,
                        help='Required footprint clearance beyond the vehicle envelope, in metres')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--resume-if-exists', action='store_true',
                        help='For unattended service restarts: resume a matching saved campaign')
    parser.add_argument('--no-packet-starts', action='store_true')
    parser.add_argument('--time-sectors', type=int, default=1,
                        help='cyclic timing degrees of freedom in the 24/25 ms predictor')
    parser.add_argument('--hybrid-time-search', action='store_true',
                        help='After a failed shortening, propose geometry/controls with a time objective')
    args = parser.parse_args()
    if args.status:
        show_status(args.status)
        return
    if not all((args.seed, args.map, args.output)):
        parser.error('--seed, --map and --output required')
    stop_requested = [False]
    def interrupted(signum, frame):
        stop_requested[0] = True
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        run_campaign(repo=Path(__file__).resolve().parents[1], seed=args.seed,
            map_path=args.map, output=args.output, target_seconds=args.target_seconds,
            phases=args.phases, workers=args.workers, iterations=args.iterations,
            stagnation_iterations=args.stagnation_iterations, refinements=args.refinements,
            resume=args.resume or (args.resume_if_exists and (args.output/'campaign.json').exists()),
            include_packet_starts=not args.no_packet_starts,
            coarse_fraction=args.coarse_fraction,
            hybrid_time=args.hybrid_time_search,time_sectors=args.time_sectors,
            margin=args.margin,
            cancel_requested=lambda: stop_requested[0])
    except KeyboardInterrupt:
        print('Interrupted; best witness and active primal retained. Use --resume.', flush=True)


if __name__ == '__main__':
    main()
