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
                         save_command_checkpoint, save_json_checkpoint, solve_witness)


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


def run_campaign(*, repo, seed, map_path, output, target_seconds=4.7,
                 phases=8, workers=8, iterations=300, stagnation_iterations=50,
                 refinements=6, resume=False, include_packet_starts=True,
                 solve=solve_witness, certify=certify_witness, digest=None,
                 cancel_requested=None):
    repo, seed, map_path, output = [Path(p).resolve() for p in (repo, seed, map_path, output)]
    if (not math.isfinite(target_seconds) or target_seconds < .75 or phases < 1
            or refinements < 0 or iterations < 1 or stagnation_iterations < 5):
        raise ValueError('invalid campaign target or search settings')
    if digest is None:
        _, digest = production_command_map(repo, repo/'build_task/command_ocp')
    contract = dict(seed_sha256=file_hash(seed), map_sha256=file_hash(map_path),
        source_sha256=digest, target_intervals=math.floor(target_seconds/.025+1.e-9),
        phases=phases, workers=workers, iterations=iterations,
        stagnation_iterations=stagnation_iterations, refinements=refinements,
        include_packet_starts=include_packet_starts)
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

    try:
        if resume:
            state = json.loads(state_path.read_text())
            if state['contract'] != contract:
                raise ValueError('campaign source/map/settings changed; create a new campaign')
            if state['status'] in ('target_reached', 'plateau'):
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
            baseline = certify(repo, seed, map_path, baseline_dir)
            if baseline['status'] != 'source_witness_passed':
                raise ValueError('initial witness fails current source gates')
            state.update(status='running', best_witness=str(baseline_dir/'witness.npz'),
                         best_lap_s=baseline['trial_lap_s'])
        save()
        while True:
            if cancel_requested is not None and cancel_requested():
                raise KeyboardInterrupt
            incumbent = Path(state['best_witness'])
            count = source_count(incumbent)-1
            if count < contract['target_intervals']:
                state.update(status='target_reached', active=None)
                save()
                break
            rotations = sorted(set(int(source_count(incumbent)*i/phases) for i in range(phases)))
            strategies = ('resample', 'drop-packet') if include_packet_starts else ('resample',)
            improved = False
            for rotation in rotations:
                for strategy in strategies:
                    key = f'{count:03d}_phase{rotation:03d}_{strategy}'
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
                    else:
                        start = incumbent
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
                                stagnation_iterations=stagnation_iterations,
                                cancel_requested=cancel_requested)
                            start = destination/'witness.npz'
                        if report['status'] == 'source_witness_passed':
                            destination = unused_directory(directory/'certified')
                            checked = certify(repo, start, map_path, destination)
                            if checked['status'] == 'source_witness_passed':
                                state.update(best_witness=str(destination/'witness.npz'),
                                    best_lap_s=checked['trial_lap_s'])
                                improved = True
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
                    save()
                    if improved: break
                if improved: break
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
    parser.add_argument('--target-seconds', type=float, default=4.7)
    parser.add_argument('--phases', type=int, default=8)
    parser.add_argument('--workers', type=int, default=os.cpu_count() or 1)
    parser.add_argument('--iterations', type=int, default=300)
    parser.add_argument('--stagnation-iterations', type=int, default=50)
    parser.add_argument('--refinements', type=int, default=6)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--resume-if-exists', action='store_true',
                        help='For unattended service restarts: resume a matching saved campaign')
    parser.add_argument('--no-packet-starts', action='store_true')
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
            cancel_requested=lambda: stop_requested[0])
    except KeyboardInterrupt:
        print('Interrupted; best witness and active primal retained. Use --resume.', flush=True)


if __name__ == '__main__':
    main()
