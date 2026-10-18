"""v2.4.0: auditable, fail-closed continuation of Phase-I IPOPT.

A checkpoint satisfies all hard constraints of the *relaxed* Phase-I problem,
NOT the exact vehicle dynamics. This module never certifies a racing lap.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from seed_projection import validate_native_seed


def validate_restart_checkpoint(checkpoint, n, expected_length, expected_phases):
    if not isinstance(checkpoint, dict) or not checkpoint.get('hard_constraints_satisfied'):
        raise ValueError('Missing hard-feasible Phase-I checkpoint')
    if checkpoint.get('strict_original_dynamics_feasible') is not False:
        raise ValueError('Phase-I checkpoint must be marked dynamics-infeasible')
    seed = validate_native_seed(checkpoint['seed'], n)
    stations = np.asarray(checkpoint['station_nodes_m'], dtype=float)
    lengths = np.asarray(checkpoint['phase_lengths_m'], dtype=float)
    defect = float(checkpoint['raw_max_dynamic_defect'])
    energy = float(checkpoint['elastic_merit'])
    if (stations.shape != (n,) or lengths.shape != (expected_phases,)
            or not np.isfinite(stations).all() or not np.isfinite(lengths).all()
            or abs(stations[0]) > 1e-8 or np.any(np.diff(stations) <= 0)
            or stations[-1] >= expected_length
            or np.any(lengths <= 0)
            or abs(np.sum(lengths) - expected_length) > 1e-3
            or not np.isfinite([defect, energy]).all()
            or defect <= 0 or energy < 0):
        raise ValueError('Phase-I restart fails mesh/defect/energy checks')
    return seed, stations, lengths, defect, energy


def continue_phase1(old, track, model, kwargs, *, output: Path, phase: int,
                    max_restarts: int, minimum_improvement: float):
    """Save every report and seed successive solves with strict checkpoints.

    Restarts occur only when both the maximal defect and merit *improved*,
    and the retained snapshot meets every hard constraint. No failed solve
    ever proceeds to the minimum-time or native-replay stages.
    """
    if max_restarts < 0 or max_restarts > 10:
        raise ValueError('Phase-I restarts must be between 0 and 10')
    if minimum_improvement <= 0:
        raise ValueError('Minimum defect progress must be positive')
    output = Path(output)
    progress = []
    active_kwargs = dict(kwargs)
    baseline = None
    candidate, report = None, None
    for attempt in range(max_restarts + 1):
        print(f'== Phase-I solve attempt {attempt + 1}/{max_restarts + 1} ==', flush=True)
        candidate, report = old.solve_bottom_up(track, model, **active_kwargs)
        stage = (report.get('solve_stages') or [{}])[0]
        row = dict(attempt=attempt, return_status=report.get('solver', {}).get('return_status'),
                   initial_max_defect=stage.get('initial_raw_max_defect'),
                   best_max_defect=stage.get('best_raw_max_defect'),
                   iterations=stage.get('ipopt_iterations'),
                   hard_feasible_checkpoint=bool(report.get('phase1_restart_checkpoint')),
                   strict_feasible=bool(report.get('solver', {}).get('feasible_incumbent')))
        progress.append(row)
        (output / f'ipopt_report_pass_{phase:02d}_restart_{attempt:02d}.json').write_text(
            json.dumps(report, indent=2, default=str) + '\n')
        print('Phase-I attempt diagnostic:', json.dumps(row, sort_keys=True), flush=True)
        if candidate and report.get('solver', {}).get('feasible_incumbent'):
            break
        checkpoint = report.get('phase1_restart_checkpoint')
        if not checkpoint or attempt == max_restarts:
            break
        n = track.count
        lengths_initial = active_kwargs.get('phase_lengths_initial')
        expected_phases = (len(lengths_initial) if lengths_initial is not None
                           else 2 * sum(1 for k in range(1, n)
                             if kwargs['brake_mode_seed'][k] and not kwargs['brake_mode_seed'][k-1]) + 1)
        seed, stations, lengths, defect, energy = validate_restart_checkpoint(
            checkpoint, n, track.length, expected_phases)
        reference = (float(stage['initial_raw_max_defect']) if baseline is None
                     else baseline)
        if not (np.isfinite(reference) and defect < reference - minimum_improvement):
            print('Phase-I continuation stopped: no meaningful strict-max-defect reduction', flush=True)
            break
        if (baseline is not None and defect >= baseline - minimum_improvement):
            print('Phase-I continuation stopped: no cumulative improvement', flush=True)
            break
        baseline = defect
        active_kwargs.update(initial_state_seed=seed,
                             initial_state_seed_s=stations,
                             initial_state_seed_is_model_consistent=True,
                             phase_lengths_initial=lengths,
                             brake_mode_seed_source='v2.4.0 hard-feasible Phase-I diagnostic checkpoint (not exact dynamics)')
    (output / f'phase1_progress_pass_{phase:02d}.json').write_text(
        json.dumps({'runs':progress,'final_strict_feasible':bool(report.get('solver',{}).get('feasible_incumbent')),
                    'checkpoint_is_not_physics_certification':True},indent=2)+'\n')
    return candidate, report, progress
