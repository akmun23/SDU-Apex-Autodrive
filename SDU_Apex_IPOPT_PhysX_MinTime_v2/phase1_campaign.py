"""v2.4.0: auditable, fail-closed continuation of Phase-I IPOPT.

A checkpoint satisfies all hard constraints of the *relaxed* Phase-I problem,
NOT the exact vehicle dynamics. This module never certifies a racing lap.
"""
from __future__ import annotations
import json
import hashlib
from pathlib import Path
import numpy as np
from seed_projection import validate_native_seed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


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


def load_phase1_resume_report(report_path, *, map_path, centerline_path,
                              mesh_path, track, model_metadata,
                              optimizer_source_sha256,
                              phase1_formulation):
    """Load a dynamics-infeasible Phase-I seed only from the identical OCP.

    A saved elastic checkpoint is useful as an IPOPT starting point, but it is
    not a feasible vehicle trajectory.  Resume only when the map, user
    centerline, generated spatial mesh, source model, restored optimizer, and
    Phase-I formulation all exactly match the new run.
    """
    report_path = Path(report_path).resolve()
    provenance_path = report_path.parent / 'provenance.json'
    if not report_path.is_file() or not provenance_path.is_file():
        raise ValueError('Phase-I resume requires its report and sibling provenance.json')
    try:
        report = json.loads(report_path.read_text())
        provenance = json.loads(provenance_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f'Cannot read Phase-I resume artifacts: {error}') from error

    expected_hashes = {
        'map_sha256': _sha256(map_path),
        'centerline_sha256': _sha256(centerline_path),
    }
    for name, actual in expected_hashes.items():
        if provenance.get(name) != actual:
            raise ValueError(f'Phase-I resume {name} does not match this run')

    mesh = provenance.get('optimizer_reference_track_mesh')
    if not isinstance(mesh, dict) or mesh.get('sha256') != _sha256(mesh_path):
        raise ValueError('Phase-I resume spatial mesh does not match this run')
    if (int(mesh.get('nodes', -1)) != int(track.count)
            or abs(float(mesh.get('length_m', -1.0))
                   - float(track.length)) > 1e-6):
        raise ValueError('Phase-I resume track dimensions do not match this run')

    source_report = provenance.get('phase1_solver_overlay', {})
    candidate_overlay = report.get('phase1_solver_overlay', {})
    if (source_report.get('phase1_formulation') != phase1_formulation
            or candidate_overlay.get('phase1_formulation') != phase1_formulation):
        raise ValueError('Phase-I resume formulation differs from this run')
    if (source_report.get('optimizer_source_sha256') != optimizer_source_sha256
            or candidate_overlay.get('optimizer_source_sha256') != optimizer_source_sha256):
        raise ValueError('Phase-I resume optimizer source differs from this run')
    if report.get('unity_vehicle_model') != json.loads(
            json.dumps(model_metadata)):
        raise ValueError('Phase-I resume vehicle model differs from this run')

    source_track = report.get('track', {})
    if (int(source_track.get('count', -1)) != int(track.count)
            or abs(float(source_track.get('length_m', -1.0))
                   - float(track.length)) > 1e-6):
        raise ValueError('Phase-I resume report track differs from this run')
    if (report.get('phase1_strict_feasible') is not False
            or report.get('solver', {}).get('feasible_incumbent') is not False
            or report.get('phase1_restart_checkpoint') is None):
        raise ValueError('Resume source is not a failed Phase-I report with a checkpoint')

    checkpoint = validate_restart_checkpoint(
        report['phase1_restart_checkpoint'], int(track.count),
        float(track.length), 1)
    if not np.allclose(checkpoint[1], np.asarray(track.s, dtype=float),
                       rtol=0.0, atol=1e-8):
        raise ValueError('Phase-I resume checkpoint station mesh differs from this run')
    return checkpoint


def continue_phase1(old, track, model, kwargs, *, output: Path, phase: int,
                    max_restarts: int, minimum_improvement: float):
    """Save every report and seed successive solves with strict checkpoints.

    Restarts occur only when the retained hard-feasible checkpoint reduces
    the strict gate's maximum original-dynamics defect. No failed solve ever
    proceeds to the minimum-time or native-replay stages.
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
        expected_phases = len(lengths_initial) if lengths_initial is not None else 1
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
                             phase_lengths_initial=lengths)
    (output / f'phase1_progress_pass_{phase:02d}.json').write_text(
        json.dumps({'runs':progress,'final_strict_feasible':bool(report.get('solver',{}).get('feasible_incumbent')),
                    'checkpoint_is_not_physics_certification':True},indent=2)+'\n')
    return candidate, report, progress
