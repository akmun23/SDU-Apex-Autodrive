"""Command-realizable periodic feasibility polishing, using production code.

This is deliberately distinct from the free spatial path/time search. At a
fixed trial lap duration (integer 25 ms periods), optimize an entire periodic
state/command witness with the exact runtime transition map. Do not score a
controller or pretend spatial steering-rate decisions are held commands.
"""
from __future__ import annotations

import hashlib
import argparse
import ctypes as C
import json
import math
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import casadi as ca
import numpy as np

from command_realizability import periodic, time_grid
from map_track import (MapGrid, _FOOTPRINT_LOCAL, _FOOTPRINT_UNSAMPLED_CORRECTION,
                       CLEARANCE_NUMERICAL_TOLERANCE_M, clearance_passes)
from path_frame import path_frame_numpy
from sector_time import periodic_sector_weights, resample_source_time_initialization, predictor_timing
from physics import SourcePlant


STATE_SCALES = np.array([1., 1., 1., 2., 1., 2., *([150.]*4),
                          .02, 1., .12, 2., .1, 2., .35, *([.04]*4)])


def validate_margin(margin):
    """Require a finite, nonnegative footprint clearance margin in metres."""
    try:
        value = float(margin)
    except (TypeError, ValueError):
        raise ValueError('margin must be finite and nonnegative') from None
    if not math.isfinite(value) or value < 0.:
        raise ValueError('margin must be finite and nonnegative')
    return value


def witness_quality(path, report):
    """Comparable quality at a fixed duration, including the periodic input seam.

    Clearance is extra space beyond the verifier's 5 cm footprint envelope.
    These metrics rank source witnesses; online MPC trials establish tracking.
    """
    with np.load(path, allow_pickle=False) as data:
        controls = np.asarray(data['controls'], dtype=float)
    changes = controls-np.roll(controls, 1, axis=1)
    margin = (report.get('rollout') or {}).get('minimum_1ms_footprint_slack_m')
    return dict(clearance_slack_m=(float(margin) if margin is not None else None),
        steering_step_rms_rad=float(np.sqrt(np.mean(changes[0]**2))),
        steering_step_max_rad=float(np.max(np.abs(changes[0]))),
        throttle_step_rms=float(np.sqrt(np.mean(changes[1]**2))))


def quality_key(quality):
    """Prefer more clearance, then smoother controls for equal-time witnesses."""
    slack = quality.get('clearance_slack_m')
    # Sub-0.1 mm numerical changes must not outrank a smoother alternative.
    return (round(float(slack)/1.e-4) if slack is not None else -math.inf,
            -quality.get('steering_step_rms_rad', math.inf),
            -quality.get('throttle_step_rms', math.inf))


def iterate_rank(residual, objective):
    """First restore source feasibility, then retain the best feasible objective."""
    if not math.isfinite(residual) or not math.isfinite(objective):
        return (2, math.inf, math.inf)
    return (0, objective, residual) if residual <= 1.e-5 else (1, residual, objective)


def quality_proposal_merit(residual, objective, baseline_objective):
    """Rank bounded exploratory points separately from feasible checkpoints."""
    if (not math.isfinite(residual) or residual > .01 or
            not math.isfinite(objective) or not math.isfinite(baseline_objective)):
        return None
    materiality = 1.e-5*max(1., abs(baseline_objective))
    if objective >= baseline_objective-materiality:
        return None
    return objective+10.*residual


def save_command_checkpoint(path, state, controls, digest, residual, iteration):
    """Power-loss-safe primal checkpoint; feasibility must still be checked."""
    save_npz_checkpoint(path, schema='production-command-ocp/v1',
        state=state, controls=controls, dt_s=.025,
        lap_time_s=controls.shape[1]*.025, source_sha256=digest,
        max_constraint_violation=residual, iteration=iteration)


def save_npz_checkpoint(path, **values):
    temporary = path.with_suffix('.tmp.npz')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **values)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(directory)
    finally: os.close(directory)


def save_json_checkpoint(path, value):
    temporary = path.with_suffix('.tmp.json')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(directory)
    finally: os.close(directory)


def production_command_map(repo: Path, build: Path):
    repo, build = Path(repo).resolve(), Path(build).resolve()
    files = [repo/'f1tenth_mpc/src/unity_source_plant.cpp',
             repo/'f1tenth_mpc/src/unity_source_plant_c.cpp',
             repo/'f1tenth_mpc/include/unity_source_plant.hpp',
             repo/'f1tenth_mpc/include/unity_source_plant_c.h',
             Path(__file__).with_name('command_ocp_kernel.cpp')]
    compiler = subprocess.check_output(['g++', '--version'])
    digest = hashlib.sha256(compiler+b'-O3 -ffp-contract=off')
    for path in files: digest.update(path.read_bytes())
    build.mkdir(parents=True, exist_ok=True)
    library = build/f'command_ocp_{digest.hexdigest()[:20]}.so'
    if not library.exists():
        temporary = library.with_suffix(f'.{os.getpid()}.tmp.so')
        subprocess.run(['g++', '-std=c++17', '-O3', '-DNDEBUG',
            '-ffp-contract=off', '-shared', '-fPIC',
            '-I'+str(repo/'f1tenth_mpc/include'),
            str(files[0]), str(files[1]), str(files[4]), '-o', str(temporary)], check=True)
        os.replace(temporary, library)
    return ca.external('sdu_command_step', str(library)), digest.hexdigest()


def load_seed(checkpoint, mesh, audit, count, repo):
    with np.load(checkpoint, allow_pickle=False) as source:
        old = np.asarray(source['state_x'])
        station = np.asarray(source['sample_s'])
        length = float(np.sum(source['phase_lengths_m']))
        lap = float(source['predicted_lap_time_s'])
        throttle = np.asarray(source['throttle_norm']).reshape(-1)
        decisions = np.asarray(source['decision_x'])
    table = np.loadtxt(mesh, delimiter=',')
    if table.shape != (len(station), 7) or not np.allclose(station, table[:, 0], atol=1.e-7):
        raise ValueError('source mesh differs from checkpoint')
    track = SimpleNamespace(count=len(station), s=station, length=length,
                            x=table[:, 1], y=table[:, 2], psi=table[:, 3])
    with np.load(audit, allow_pickle=False) as graph:
        if not np.array_equal(graph['decision_x'], decisions):
            raise ValueError('NLP audit belongs to a different primal')
        clock = time_grid(track, old, lap,
            density=graph['node_time_density_s_per_m'], steps=graph['spatial_steps_m'])
    query = np.arange(count)*lap/count
    fields = np.array([periodic(clock, row, query) for row in old])
    cx, cy, heading = path_frame_numpy(track, station)
    yaw = np.unwrap(heading+old[1])
    winding = round((yaw[-1]-yaw[0])/(2*math.pi))
    if abs(winding) != 1: raise ValueError('initializer does not describe a complete track circuit')
    poses = np.array([periodic(clock, cx-np.sin(heading)*old[0], query),
                      periodic(clock, cy+np.cos(heading)*old[0], query),
                      np.interp(query, clock, np.r_[yaw, yaw[0]+2*math.pi*winding])])
    physical = np.vstack([fields[2:5], fields[5:9], fields[10:16], fields[9], np.zeros((4, count))])
    # Reconcile jounce only for the INITIAL GUESS, not during an NLP rollout.
    # All four histories remain independent physical shooting states.
    plant = SourcePlant(repo, repo/'build_task/source_witness')
    reconcile = plant.lib.unity_source_plant_reconcile_suspension_jounce_c
    reconcile.argtypes = [C.POINTER(C.c_double), C.POINTER(C.c_int)]
    reconcile.restype = C.c_int
    for k in range(count):
        raw = (C.c_double*26)(*physical[:, k], *([0.]*8))
        initialized = C.c_int(0)
        if not reconcile(raw, C.byref(initialized)): raise ValueError('invalid seed jounce')
        physical[14:18, k] = list(raw)[14:18]
    x = np.column_stack([np.vstack([poses, physical]), np.r_[poses[:, 0], physical[:, 0]]])
    x[2, -1] += 2*math.pi*winding
    u = np.vstack([periodic(clock, old[9], query+.5*lap/count),
                   periodic(clock, throttle, query)])
    return x, u, winding, lap


def clearance_interpolant(grid):
    x = (np.arange(grid.image.shape[1])+.5)*grid.resolution
    y = (np.arange(grid.image.shape[0])+.5)*grid.resolution
    return ca.interpolant('command_wall_distance', 'linear', [x, y],
                          grid.clearance[::-1, :].T.ravel(order='F'))


def production_fixed_prefix(repo, build, digest, packed, tick, *, workers=1):
    """Only bind literal integer ticks. A variable-duration derivative is NOT supplied."""
    if not isinstance(tick, int) or not 1 <= tick <= 25:
        raise ValueError('source prefix must be a literal tick in [1,25]')
    library = Path(build).resolve()/f'command_ocp_{digest[:20]}.so'
    prefix = ca.external('sdu_command_prefix', str(library))
    columns = packed.size2()
    inputs = ca.vertcat(packed, ca.DM.ones(1, columns)*(tick*.001))
    return mapped_transition(prefix, columns, workers)(inputs)


def mapped_transition(function, columns, workers):
    if not isinstance(workers, int) or not 1 <= workers <= (os.cpu_count() or 1):
        raise ValueError('workers exceed available CPUs or are invalid')
    return (function.map(columns, 'serial') if workers == 1 else
            function.map(columns, 'thread', workers))


def load_command_seed(path, count):
    with np.load(path, allow_pickle=False) as archive:
        if str(archive['schema'].item()) != 'production-command-ocp/v1':
            raise ValueError('not a production command initializer')
        x, u = np.asarray(archive['state']), np.asarray(archive['controls'])
        if (x.shape != (21, u.shape[1]+1) or u.shape[0] != 2 or
                not np.isfinite(x).all() or not np.isfinite(u).all()
                or float(archive['dt_s']) != .025):
            raise ValueError('malformed command initializer')
    winding = round((x[2, -1]-x[2, 0])/(2*math.pi))
    if abs(winding) != 1: raise ValueError('initializer is not one track circuit')
    if count == u.shape[1]:
        # A clearance-cut restart MUST preserve the optimized shooting states
        # and held commands bit for bit. Replacing shooting states with the
        # recursive trace discards the saved primal. Floating query*count then
        # floor also selected the PREVIOUS packet at exact integer boundaries
        # (15 commands in the 195-packet witness). Never resample a warm start.
        return x.copy(), u.copy(), winding, count*.025
    old_grid = np.linspace(0., 1., x.shape[1])
    query = np.linspace(0., 1., count+1)
    dense_path = path.with_name('source_trace.npz')
    if dense_path.exists():
        with np.load(dense_path, allow_pickle=False) as trace:
            dense = np.asarray(trace['state'])
            if (not np.array_equal(trace['commands'], u) or
                    not np.allclose(dense[:, 0], x[:, 0], atol=1.e-12, rtol=0.) or
                    float(trace['dt_s']) != .001 or dense.shape != (21, u.shape[1]*25+1)):
                raise ValueError('dense initializer trace does not match its command witness')
        # Fast suspension motion between 25 ms nodes is not linear. Sample the
        # already computed 1 ms source history instead of inventing that state.
        old_grid = np.linspace(0., 1., dense.shape[1])
        states = np.array([np.interp(query, old_grid, row) for row in dense])
    else:
        states = np.array([np.interp(query, old_grid, row) for row in x])
    # Held commands stay held in the initializer. The new trial must solve all
    # its dynamics again; changing duration never transfers certification.
    # Integer division gives exact left-closed packet ownership, including
    # shared boundaries when the old and new sample counts have common factors.
    control_index = np.arange(count, dtype=np.int64)*u.shape[1]//count
    controls = u[:, control_index].copy()
    if count != u.shape[1] and dense_path.exists():
        # End rack is a physically meaningful initial target; continuous rack
        # interpolation was not a realizable command in the spatial problem.
        # This is only initialization, never a post-solve control correction.
        controls[0] = states[16, 1:]
    return states, controls, winding, u.shape[1]*.025


def shorten_seed_by_packet(path, count, function, workers):
    """Alternative one-packet continuation; no resampling of hidden history.

    Delete one interior shooting node and its preceding held input. All other
    25 ms transitions remain the saved primal; only the bridge across the gap
    needs repair. Select the gap by measured source-map defect, not by a
    guessed speed/steering regime. This is INITIALIZATION, not feasibility.
    """
    with np.load(path, allow_pickle=False) as archive:
        old_count = archive['controls'].shape[1]
    if count != old_count-1:
        raise ValueError('packet shortening requires exactly one fewer interval')
    state, control, winding, lap = load_command_seed(path, old_count)
    predicted = np.asarray(mapped_transition(function, count, workers)(
        np.vstack([state[:, :count], control[:, 1:]])))
    defects = np.max(np.abs((predicted-state[:, 2:])/STATE_SCALES[:, None]), axis=0)
    packet = int(np.argmin(defects))
    print(json.dumps(dict(seed_strategy='drop-one-packet', removed_packet=packet,
        gap_scaled_dynamic_defect=float(defects[packet]),
        source_lap_s=lap, trial_lap_s=count*.025)), flush=True)
    return (np.delete(state, packet+1, axis=1), np.delete(control, packet, axis=1),
            winding, lap)


def seed_dynamic_defect(function, state, controls, workers):
    predicted = np.asarray(mapped_transition(function, controls.shape[1], workers)(
        np.vstack([state[:, :-1], controls])))
    dynamic = float(np.max(np.abs((predicted-state[:, 1:])/STATE_SCALES[:, None])))
    periodic = state[:, -1]-state[:, 0]
    periodic[2] -= round(periodic[2]/(2*math.pi))*2*math.pi
    return max(dynamic, float(np.max(np.abs(periodic/STATE_SCALES))))


def recondition_command_seed(function, state, controls, workers, *, max_laps=12):
    """Initialize fast internal histories with source propagation, not time warping.

    Prescribe ONLY the guessed pose and u/v/r at each node. Propagate wheels,
    rack, heave, attitude and jounce through the new held commands to settle
    their periodic history. This is an initializer with prescribed body motion,
    NOT a recursive motion prediction or feasibility certificate. The NLP still
    optimizes all 21 coordinates and the final replay has no state resets.
    """
    before = seed_dynamic_defect(function, state, controls, workers)
    best, score, carry = state.copy(), before, state[6:, 0].copy()
    laps = 0
    for laps in range(1, max_laps+1):
        trial = state.copy()
        start = carry.copy()
        try:
            for k in range(controls.shape[1]):
                trial[6:, k] = carry
                propagated = np.asarray(function(np.r_[trial[:, k], controls[:, k]])).reshape(-1)
                carry = propagated[6:]
            trial[6:, -1] = carry
            measured = seed_dynamic_defect(function, trial, controls, workers)
        except RuntimeError:
            break  # retain the better finite initializer; no certificate exists
        if math.isfinite(measured) and measured < score:
            best, score = trial, measured
        if np.max(np.abs((carry-start)/STATE_SCALES[6:])) <= 1.e-7:
            break
    return best, dict(source_settling_laps=laps, scaled_defect_before=before,
        scaled_defect_after=score, prescribed_body_initializer=True, source_certified=False)


def check_full_witness(function, state, controls, grid, margin, *, repo=None, trace_path=None):
    predicted = state[:, 0].copy()
    errors, clearance, state_errors = [], [], []
    for k in range(controls.shape[1]):
        predicted = np.asarray(function(np.r_[predicted, controls[:, k]])).reshape(-1)
        errors.append(np.linalg.norm(predicted[:2]-state[:2, k+1]))
        state_errors.append(predicted-state[:, k+1])
        clearance.append(grid.footprint_clearance(*predicted[:3], margin=margin))
    result = dict(full_lap_xy_max_m=float(max(errors)),
        full_lap_xy_endpoint_m=float(errors[-1]),
        minimum_sampled_footprint_slack_m=float(min(clearance)),
        recursive_start_only=True, reset_count=0,
        collision_swept_continuous_certificate=False)
    errors_by_state = np.asarray(state_errors)
    result['recursive_motion_errors'] = {name: dict(
        rmse=float(np.sqrt(np.mean(errors_by_state[:, index]**2))),
        max_abs=float(np.max(np.abs(errors_by_state[:, index]))))
        for name, index in (('yaw_rad', 2), ('u_mps', 3), ('v_com_mps', 4),
            ('yaw_rate_radps', 5), ('wheel_fl_radps', 6), ('wheel_fr_radps', 7),
            ('wheel_rl_radps', 8), ('wheel_rr_radps', 9), ('rack_rad', 16))}
    if repo is not None:
        # Independent 1 ms replay of the SAME fixed controls; inspect the full
        # perimeter at every source tick, not just the 40 Hz shooting nodes.
        plant = SourcePlant(repo, repo/'build_task/source_witness')
        plant.reset(*state[:3, 0], state[3, 0])
        plant.input.state[:18] = state[3:, 0]
        plant.input.suspension_history_initialized = 1
        smallest = float('inf')
        violations, samples = [], []
        worst_sample = None
        trace = [state[:, 0].copy()]
        for k in range(controls.shape[1]):
            interval_minimum, worst_tick = float('inf'), None
            for tick in range(1, 26):
                current = plant.step(*controls[:, k], duration_s=.001)
                trace.append(np.r_[current.x, current.y, current.yaw, plant.input.state[:18]])
                slack = grid.footprint_clearance(
                    current.x, current.y, current.yaw, margin=margin)
                if worst_sample is None or slack < smallest:
                    worst_sample = dict(interval=k, tick=tick, slack_m=float(slack),
                        lap_time_s=(k*25+tick)*.001,
                        x_m=float(current.x), y_m=float(current.y), yaw_rad=float(current.yaw))
                smallest = min(smallest, slack)
                if not clearance_passes(slack):
                    samples.append(dict(interval=k, tick=tick, slack_m=float(slack)))
                if slack < interval_minimum: interval_minimum, worst_tick = slack, tick
            if not clearance_passes(interval_minimum):
                violations.append(dict(interval=k, tick=worst_tick, slack_m=interval_minimum))
        result['minimum_1ms_footprint_slack_m'] = smallest
        result['independent_1ms_vs_25ms_xy_endpoint_m'] = float(np.hypot(
            plant.state.x-predicted[0], plant.state.y-predicted[1]))
        result['source_tick_perimeter_validated'] = clearance_passes(smallest)
        result['clearance_numerical_tolerance_m'] = CLEARANCE_NUMERICAL_TOLERANCE_M
        result['violating_source_ticks'] = violations
        result['violating_source_samples'] = samples
        result['worst_source_sample'] = worst_sample
        result['clearance_measurement'] = dict(map_resolution_m=float(grid.resolution),
            perimeter_probe_correction_m=float(_FOOTPRINT_UNSAMPLED_CORRECTION),
            source_tick_s=.001, source_model_only=True,
            numerical_tolerance_m=CLEARANCE_NUMERICAL_TOLERANCE_M,
            continuous_intertick_certificate=False)
        if trace_path is not None:
            if trace_path.exists(): raise FileExistsError('source trace already exists')
            np.savez_compressed(trace_path, state=np.asarray(trace).T, dt_s=.001,
                                commands=controls, command_dt_s=.025)
            result['source_trace'] = str(trace_path)
    return result


def source_clearance_refinement(rollout, existing_cuts, count):
    """Worst newly violated literal source tick per interval; never relax margins.

    Already constrained prefixes do not make useful new cuts. If only those
    fail, more identical constraints cannot repair recursive state drift.
    """
    if not math.isfinite(rollout.get('full_lap_xy_max_m', math.inf)) or \
            rollout['full_lap_xy_max_m'] > .05:
        return []
    covered = {tuple(c) for c in existing_cuts}
    base_ticks = {6, 12, 18, 24, 25}
    worst = {}
    for sample in rollout.get('violating_source_samples', rollout.get('violating_source_ticks', [])):
        interval, tick = sample['interval'], sample['tick']
        slack = sample.get('slack_m', -math.inf)
        if (not isinstance(interval, int) or not isinstance(tick, int)
                or not 0 <= interval < count or not 1 <= tick <= 25):
            raise ValueError('invalid source replay clearance cut')
        if (tick in base_ticks or (interval, tick) in covered or clearance_passes(slack)):
            continue
        if interval not in worst or slack < worst[interval]['slack_m']:
            worst[interval] = dict(interval=interval, tick=tick, slack_m=slack)
    return [(i, worst[i]['tick']) for i in sorted(worst)]


def reported_clearance_cuts(report):
    """Prefer explicit missing-tick requests over already constrained worst ticks."""
    requested = report.get('requested_source_clearance_cuts')
    if requested:
        return {tuple(c) for c in requested}
    return {(int(c['interval']), int(c['tick'])) for c in
            (report.get('rollout') or {}).get('violating_source_ticks', [])}


def export_reference(trace_path, grid, output):
    """Sample the actual fixed-control source trajectory; no path smoothing.

    The 9-column runtime reference stays separate from the control witness.
    Body-forward speed integral and actual witness duration are distinct.
    """
    with np.load(trace_path, allow_pickle=False) as trace:
        x = np.asarray(trace['state'])
        dt = float(trace['dt_s'])
        commands = np.asarray(trace['commands']) if 'commands' in trace else None
    xy = x[:2, :-1].T
    ds = np.linalg.norm(np.roll(xy, -1, axis=0)-xy, axis=1)
    if np.any(ds <= 0.) or not np.isfinite(x).all():
        raise ValueError('source reference has nonforward/degenerate geometry')
    station = np.r_[0., np.cumsum(ds)]
    length = station[-1]
    # Keep source-tick positions exactly: no geometric shortening or smoothing
    # after the tight footprint gate has passed.
    query = station[:-1]
    point = xy
    backward = point-np.roll(point, 1, axis=0)
    forward = np.roll(point, -1, axis=0)-point
    chord = backward+forward
    heading = np.unwrap(np.arctan2(chord[:, 1], chord[:, 0]))
    curvature = 2.*(backward[:, 0]*forward[:, 1]-backward[:, 1]*forward[:, 0]) / (
        np.linalg.norm(backward, axis=1)*np.linalg.norm(forward, axis=1)*np.linalg.norm(chord, axis=1))
    speed = np.interp(query, station, np.r_[x[3, :-1], x[3, 0]])
    acceleration = np.interp(query, station, np.r_[np.gradient(x[3, :-1], dt),
                                                   (x[3, 1]-x[3, 0])/dt])
    widths = grid.side_clearances(point, heading)
    # Use actual output chord distances in the runtime's arc coordinate too.
    actual_ds = np.linalg.norm(np.roll(point, -1, axis=0)-point, axis=1)
    actual_s = np.r_[0., np.cumsum(actual_ds)[:-1]]
    path = output/'source_command_raceline.csv'
    np.savetxt(path, np.column_stack([actual_s, point, heading, curvature, speed,
                                    acceleration, widths]), delimiter=',', fmt='%.12g',
        header='s_m,x_m,y_m,psi_rad,kappa_radpm,body_forward_speed_mps,acceleration_mps2,d_left_m,d_right_m')
    result = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        points=len(point), spacing_max_m=float(max(actual_ds)),
        exact_command_lap_s=(x.shape[1]-1)*dt,
        body_forward_reference_integral_s=float(np.sum(2.*actual_ds/(speed+np.roll(speed, -1)))),
        source_trace=str(trace_path), post_export_geometry_smoothing=False,
        speed_mps_range=[float(min(speed)), float(max(speed))],
        unity_validated=False, mpc_validated=False)
    if commands is not None:
        if commands.shape != (2, len(point)//25) or len(point) % 25:
            raise ValueError('source command/reference epochs differ')
        applied = np.repeat(commands, 25, axis=1)
        heading_error = np.arctan2(np.sin(x[2, :-1]-heading), np.cos(x[2, :-1]-heading))
        source_values = np.column_stack([heading_error,
            x[4, :-1]-.15532*x[5, :-1], x[5, :-1], x[16, :-1], applied.T])
        witness_path = output/'source_command_raceline_with_witness.csv'
        np.savetxt(witness_path, np.column_stack([actual_s, point, heading, curvature,
            speed, acceleration, widths, source_values]), delimiter=',', fmt='%.12g',
            header='sdu_apex_source_witness/v1\ns_m,x_m,y_m,psi_rad,kappa_radpm,body_forward_speed_mps,acceleration_mps2,d_left_m,d_right_m,source_heading_error_rad,source_v_rear_mps,source_yaw_rate_radps,source_physical_steering_rad,source_steering_target_rad,source_throttle')
        result['source_witness_csv'] = str(witness_path)
        result['source_witness_sha256'] = hashlib.sha256(witness_path.read_bytes()).hexdigest()
    return result


def certify_witness(repo, path, map_path, output, margin=.05):
    """Recheck a stored witness using this build; no solve or certificate transfer."""
    margin = validate_margin(margin)
    if output.exists(): raise FileExistsError('refusing to overwrite certification')
    with np.load(path, allow_pickle=False) as archive:
        if str(archive['schema'].item()) != 'production-command-ocp/v1':
            raise ValueError('unknown command schema')
        states, controls = np.asarray(archive['state']), np.asarray(archive['controls'])
        if states.shape != (21, controls.shape[1]+1) or controls.shape[0] != 2:
            raise ValueError('invalid command witness dimensions')
        if float(archive['dt_s']) != .025: raise ValueError('unsupported command epoch')
    function, digest = production_command_map(repo, repo/'build_task/command_ocp')
    local = np.asarray(function.map(controls.shape[1], 'serial')(
        np.vstack([states[:, :-1], controls])))
    defect = float(np.max(np.abs((local-states[:, 1:])/STATE_SCALES[:, None])))
    periodic_error = states[:, -1]-states[:, 0]
    winding = round(periodic_error[2]/(2*math.pi))
    periodic_error[2] -= winding*2*math.pi
    periodic_max = float(np.max(np.abs(periodic_error/STATE_SCALES)))
    output.mkdir(parents=True)
    grid = MapGrid.load(map_path)
    rollout = check_full_witness(function, states, controls, grid, margin, repo=repo,
                                trace_path=output/'source_trace.npz')
    passed = (abs(winding) == 1 and max(defect, periodic_max) <= 1.e-5
        and rollout['full_lap_xy_max_m'] <= .05 and clearance_passes(rollout['minimum_1ms_footprint_slack_m']))
    report = dict(schema=1, status='source_witness_passed' if passed else 'no_validated_witness',
        source_plant_sha256=digest, original_witness=str(path),
        original_witness_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        trial_lap_s=controls.shape[1]*.025, max_local_dynamic_defect=defect,
        max_periodicity_defect=periodic_max, rollout=rollout,
        required_margin_m=margin,
        clearance_numerical_tolerance_m=CLEARANCE_NUMERICAL_TOLERANCE_M,
        unity_validated=False, mpc_validated=False, runtime_promoted=False)
    np.savez_compressed(output/'witness.npz', schema='production-command-ocp/v1',
        state=states, controls=controls, dt_s=.025, lap_time_s=controls.shape[1]*.025,
        max_constraint_violation=max(defect, periodic_max), source_sha256=digest)
    report['quality'] = witness_quality(output/'witness.npz', report)
    if passed: report['reference_export'] = export_reference(output/'source_trace.npz', grid, output)
    save_json_checkpoint(output/'report.json', report)
    print(json.dumps({key:report[key] for key in ('status', 'trial_lap_s',
        'max_local_dynamic_defect', 'max_periodicity_defect')}), flush=True)
    return report


def command_problem(repo, function, digest, initial_x, initial_u, winding, grid,
                    margin, workers, clearance_cuts=(), *, time_predictor=False,
                    objective_mode='repair', time_sectors=1):
    """Shared physical bounds, winding, geometry and wall constraints.

    The optional 24/25 ms chord is a time-search INITIALIZER, never a runtime
    dynamics replacement or a source certificate. Literal-tick map derivatives
    remain valid; no false duration derivative is taken through the C ABI.
    """
    if objective_mode not in ('repair', 'clearance', 'smooth'):
        raise ValueError('unknown command objective mode')
    if time_predictor and objective_mode != 'repair':
        raise ValueError('time predictor requires its own time objective')
    count = initial_u.shape[1]
    if not isinstance(time_sectors,int) or not 1 <= time_sectors <= count:
        raise ValueError('invalid sector timing dimension')
    if not time_predictor and time_sectors != 1:
        raise ValueError('sector timing only applies to time prediction')
    wall_distance = clearance_interpolant(grid)
    opti = ca.Opti()
    x, u = opti.variable(21, count+1), opti.variable(2, count)
    scales = ca.repmat(ca.DM(STATE_SCALES), 1, count+1)
    opti.set_linear_scale(x, scales)
    opti.set_linear_scale(u, ca.repmat(ca.DM([.35, .1]), 1, count))
    opti.set_initial(x, initial_x)
    opti.set_initial(u, initial_u)
    packed = ca.vertcat(x[:, :count], u)
    mapped = mapped_transition(function, count, workers)(packed)
    prefix24 = production_fixed_prefix(repo, repo/'build_task/command_ocp', digest,
                                       packed, 24, workers=workers)
    fraction = None
    if time_predictor:
        fraction = opti.variable(time_sectors)
        opti.subject_to(opti.bounded(0., fraction, 1.))
        opti.set_initial(fraction, 1.)  # incumbent is an exact feasible endpoint
        fractions = ca.mtimes(ca.DM(periodic_sector_weights(count,time_sectors)),fraction)
        mapped = prefix24+ca.repmat(fractions.T,21,1)*(mapped-prefix24)
    opti.subject_to(ca.vec((x[:, 1:]-mapped)/scales[:, 1:]) == 0.)
    periodic_delta = x[:, -1]-x[:, 0]
    periodic_delta[2] -= 2*math.pi*winding
    opti.subject_to(periodic_delta/ca.DM(STATE_SCALES) == 0.)
    limit = float('0.52359877559829887308')
    opti.subject_to(opti.bounded(-limit, u[0, :], limit))
    opti.subject_to(opti.bounded(.001, u[1, :], 1.))
    opti.subject_to(opti.bounded(.35, x[3, :], 16.))
    # Numerical/domain guards, NOT claims of physical limits or a global floor.
    opti.subject_to(opti.bounded(-math.pi/3, x[12, :], math.pi/3))
    opti.subject_to(opti.bounded(-math.pi/3, x[14, :], math.pi/3))
    opti.subject_to(opti.bounded(-.1, x[10, :], .1))
    opti.subject_to(opti.bounded(-limit, x[16, :], limit))
    # Keep the ordered one-circuit basin while allowing metre-scale geometry
    # moves. This is a witness polisher, not the unrestricted geometry search.
    opti.subject_to(ca.sum1((x[:2, :]-initial_x[:2, :])**2) <= 4.)
    tangent = np.array([math.cos(initial_x[2, 0]), math.sin(initial_x[2, 0])])
    opti.subject_to(ca.dot(x[:2, 0]-initial_x[:2, 0], tangent) == 0.)
    # Actual source prefixes, not linearly interpolated midpoint poses. This
    # closes the measured 3 mm wall-margin gap between 25 ms shooting nodes.
    pose_parts = [x[:3, :], *[
        production_fixed_prefix(repo, repo/'build_task/command_ocp', digest, packed,
                                tick, workers=workers)[:3, :]
        for tick in (6, 12, 18)], prefix24[:3, :]]
    for interval, tick in clearance_cuts:
        if not isinstance(interval, int) or not 0 <= interval < count:
            raise ValueError('invalid clearance-cut interval')
        pose_parts.append(production_fixed_prefix(repo, repo/'build_task/command_ocp',
            digest, ca.vertcat(x[:, interval], u[:, interval]), tick)[:3])
    poses = ca.horzcat(*pose_parts)
    pose_count = poses.size2()
    extra_clearance = None
    if objective_mode == 'clearance':
        extra_clearance = opti.variable()
        opti.subject_to(opti.bounded(0., extra_clearance, .10))
        opti.set_initial(extra_clearance, 0.)
    c, s = ca.cos(poses[2, :]), ca.sin(poses[2, :])
    map_c, map_s = math.cos(grid.yaw), math.sin(grid.yaw)
    for along, side in _FOOTPRINT_LOCAL:
        wx = poses[0, :]+along*c-side*s-grid.origin[0]
        wy = poses[1, :]+along*s+side*c-grid.origin[1]
        local = ca.vertcat(map_c*wx+map_s*wy, -map_s*wx+map_c*wy)
        distance = wall_distance.map(pose_count, 'serial')(local)
        opti.subject_to(distance >= margin+_FOOTPRINT_UNSAMPLED_CORRECTION+
                        (extra_clearance if extra_clearance is not None else 0.))
    if time_predictor:
        # This is a LOCAL predictor. Keep its derivatives useful across tire
        # and steering branches rather than taking unrestricted free-time
        # Newton steps from a good witness. Bounds are numerical trust boxes,
        # not vehicle limits; the exact-grid repair retains its original bounds.
        scaled_step = (x-initial_x)/scales
        control_step = (u-initial_u)/ca.repmat(ca.DM([.35, .1]), 1, count)
        opti.subject_to(opti.bounded(-.25, ca.vec(scaled_step), .25))
        opti.subject_to(opti.bounded(-.25, ca.vec(control_step), .25))
        # Distribute timing gains across sectors; use periodic timing smoothing.
        time_cost=count*.024+.001*ca.sum1(fractions)
        cyclic=ca.vertcat(fraction[1:],fraction[:1])-fraction
        opti.minimize(time_cost
                      +.01*(ca.sumsqr(scaled_step)+ca.sumsqr(control_step))/count
                      +.00005*ca.sumsqr(cyclic)/time_sectors)
    elif objective_mode == 'repair':
        # Multiply the previous objective by count/.001. This positive
        # constant preserves its minimizers and ALL relative penalties, while
        # avoiding 1e-6-scale objective gradients against unit-scale constraints.
        opti.minimize(ca.sumsqr((x-initial_x)/scales)
                     +ca.sumsqr(u-initial_u)
                     +.1*ca.sumsqr(ca.horzcat(u[:, 1:]-u[:, :-1], u[:, :1]-u[:, -1:])))
    else:
        # Fixed duration is the primary constraint. Do not make the old path
        # and old command sequence the minimizer of equal-time exploration.
        # Penalize periodic steering/throttle changes, including lap closure.
        # A weak state prior keeps the local numerical problem conditioned;
        # geometry remains free inside the same ordered one-circuit basin.
        changes = ca.horzcat(u[:, 1:]-u[:, :-1], u[:, :1]-u[:, -1:])
        smoothness = (ca.sumsqr(changes[0, :]/.10)+
                      .1*ca.sumsqr(changes[1, :]/.10))/count
        movement = ca.sumsqr((x-initial_x)/scales)/(21*count)
        objective = smoothness+.001*movement
        if extra_clearance is not None:
            objective = -extra_clearance/.05+.02*smoothness+.001*movement
        opti.minimize(objective)
    return opti, x, u, fraction


def solve_witness(*, repo, checkpoint, mesh, audit, map_path, output,
                  count=197, iterations=80, cpu_seconds=120., margin=.05,
                  command_seed=None, workers=8, clearance_cuts=(),
                  shorten_packet=False, stagnation_iterations=0, cancel_requested=None,
                  recondition_seed=False, objective_mode='repair',
                  early_clearance_target_m=None):
    margin = validate_margin(margin)
    if early_clearance_target_m is not None and (
            not math.isfinite(early_clearance_target_m) or
            early_clearance_target_m < 0. or early_clearance_target_m > .10 or
            objective_mode != 'repair'):
        raise ValueError('invalid early certification target for repair objective')
    if output.exists(): raise FileExistsError('refusing to overwrite command OCP')
    output.mkdir(parents=True)
    function, digest = production_command_map(repo, repo/'build_task/command_ocp')
    if shorten_packet:
        if command_seed is None:
            raise ValueError('packet shortening requires a command witness')
        initial_x, initial_u, winding, source_lap = shorten_seed_by_packet(
            command_seed, count, function, workers)
    else:
        initial_x, initial_u, winding, source_lap = (
            load_command_seed(command_seed, count) if command_seed is not None else
            load_seed(checkpoint, mesh, audit, count, repo))
    conditioning = None
    if recondition_seed:
        initial_x, conditioning = recondition_command_seed(function, initial_x, initial_u, workers)
        print(json.dumps(dict(stage='source history initialization', **conditioning)), flush=True)
    grid = MapGrid.load(map_path)
    # A power loss between callback checkpointing and parent-loop restart must
    # not discard the newly measured constraint. Reuse only same-map/source/
    # count/margin requests attached to this exact saved-primal directory.
    pending = Path(command_seed).with_name('clearance_refinement.json') if command_seed else None
    if pending is not None and pending.exists():
        saved = json.loads(pending.read_text())
        if (saved.get('source_plant_sha256') == digest and saved.get('intervals') == count
                and saved.get('map_sha256') == hashlib.sha256(map_path.read_bytes()).hexdigest()
                and saved.get('required_margin_m') == margin):
            clearance_cuts = sorted({tuple(c) for c in clearance_cuts} |
                                    {tuple(c) for c in saved['new_cuts']})
    opti, x, u, _ = command_problem(repo, function, digest, initial_x, initial_u,
                                   winding, grid, margin, workers, clearance_cuts,
                                   objective_mode=objective_mode)
    opti.solver('ipopt', {'expand': False, 'print_time': False}, {
        'max_iter': iterations, 'max_cpu_time': cpu_seconds,
        'hessian_approximation': 'limited-memory', 'mu_strategy': 'adaptive',
        'bound_relax_factor': 0.,
        'tol': 1.e-6, 'acceptable_tol': 1.e-5, 'print_level': 5})
    evaluate_g = ca.Function('command_feasibility', [opti.x], [opti.g])
    evaluate_f = ca.Function('command_objective', [opti.x], [opti.f])
    lower = np.asarray(opti.value(opti.lbg, opti.initial())).reshape(-1)
    upper = np.asarray(opti.value(opti.ubg, opti.initial())).reshape(-1)
    def violation(flat):
        values = np.asarray(evaluate_g(flat)).reshape(-1)
        return float(np.max(np.maximum(np.maximum(lower-values, values-upper), 0.)))
    initial = np.asarray(opti.value(opti.x, opti.initial())).reshape(-1)
    initial_objective = float(evaluate_f(initial))
    print(json.dumps(dict(stage='production-command witness', output=str(output.resolve()), intervals=count,
        trial_lap_s=count*.025, source_lap_s=source_lap, states=21,
        decisions=opti.nx, constraints=opti.ng, initial_residual=violation(initial),
        max_iterations=iterations, cpu_seconds=cpu_seconds, workers=workers,
        objective_scaling_factor=count/.001,
        source_clearance_cuts=len(clearance_cuts),
        objective_mode=objective_mode,
        source_sha256=digest)), flush=True)
    best = [None]
    best_quality_proposal = [None]
    significant = [float('inf'), 0]
    significant_objective = [float('inf')]
    stagnated = [False]
    early_verified = [None]
    early_refinement = [None]
    early_checked_iteration = [-40]
    def callback(iteration):
        if cancel_requested is not None and cancel_requested():
            raise RuntimeError('command campaign interruption requested')
        if iteration % 5: return
        flat = np.asarray(opti.debug.value(opti.x)).reshape(-1)
        residual = violation(flat)
        objective = float(evaluate_f(flat))
        print(json.dumps(dict(iteration=int(iteration), trial_lap_s=count*.025,
                               max_constraint_violation=residual,
                               objective=objective, objective_mode=objective_mode)), flush=True)
        if np.isfinite(residual) and np.isfinite(objective) and (
                best[0] is None or iterate_rank(residual, objective) <
                iterate_rank(best[0][2], best[0][3])):
            best[0] = (np.asarray(opti.debug.value(x)), np.asarray(opti.debug.value(u)),
                       residual, objective)
            save_command_checkpoint(output/'best_iterate.npz', *best[0][:2],
                                    digest, residual, int(iteration))
        # The restoration OCP can have a source-certified, materially better
        # checkpoint hundreds of iterations before IPOPT satisfies dual
        # optimality. A newly violated 1ms point must also interrupt this stale
        # NLP: its caller rebuilds with exact prefix constraints at those ticks.
        # Never keep optimizing an objective that cannot see the measured miss.
        if (iteration-early_checked_iteration[0]>=40 and
                0. <= residual <= 1.e-5):
            early_checked_iteration[0]=int(iteration)
            candidate_x=np.asarray(opti.debug.value(x))
            candidate_u=np.asarray(opti.debug.value(u))
            periodic=candidate_x[:,-1]-candidate_x[:,0]
            periodic[2]-=winding*2*math.pi
            periodic_scaled=float(np.max(np.abs(periodic/STATE_SCALES)))
            if periodic_scaled<=1.e-5:
                candidate_rollout=check_full_witness(
                    function,candidate_x,candidate_u,grid,margin,repo=repo)
                slack=float(candidate_rollout['minimum_1ms_footprint_slack_m'])
                target=early_clearance_target_m if early_clearance_target_m is not None else 0.
                passed=(candidate_rollout['full_lap_xy_max_m']<=.05 and clearance_passes(slack,target))
                print(json.dumps(dict(event='early_source_checkpoint_gate',
                    iteration=int(iteration), lap_s=count*.025,
                    clearance_slack_m=slack,
                    target_slack_m=target,
                    clearance_numerical_tolerance_m=CLEARANCE_NUMERICAL_TOLERANCE_M,
                    periodic_scaled_defect=periodic_scaled,
                    source_xy_error_m=candidate_rollout['full_lap_xy_max_m'],
                    worst_source_sample=candidate_rollout.get('worst_source_sample'),
                    clearance_measurement=candidate_rollout.get('clearance_measurement'),
                    accepted=passed)),flush=True)
                new_cuts=source_clearance_refinement(candidate_rollout,clearance_cuts,count)
                if new_cuts:
                    # Retain precisely the replayed primal. Restarting from a
                    # different, lower-objective saved point would lose causality.
                    best[0]=(candidate_x,candidate_u,residual,objective)
                    save_command_checkpoint(output/'best_iterate.npz',
                        candidate_x,candidate_u,digest,residual,int(iteration))
                    early_refinement[0]=dict(iteration=int(iteration),
                        source_plant_sha256=digest,intervals=count,
                        map_sha256=hashlib.sha256(map_path.read_bytes()).hexdigest(),
                        required_margin_m=margin,
                        new_cuts=[list(c) for c in new_cuts],
                        clearance_slack_m=slack,
                        worst_source_sample=candidate_rollout.get('worst_source_sample'))
                    save_json_checkpoint(output/'clearance_refinement.json',early_refinement[0])
                    print(json.dumps(dict(event='source_clearance_refinement_requested',
                        **early_refinement[0])),flush=True)
                    raise RuntimeError('source replay discovered missing clearance constraints; rebuilding NLP')
                if passed and early_clearance_target_m is not None:
                    # The normal solve_witness post-IPOPT report and export
                    # still reruns the full source verifier. No shortcut is a
                    # substitute for the two independent certification gates.
                    best[0]=(candidate_x,candidate_u,residual,objective)
                    save_command_checkpoint(output/'best_iterate.npz',
                                            candidate_x,candidate_u,
                                            digest,residual,int(iteration))
                    early_verified[0]=dict(iteration=int(iteration),
                        clearance_slack_m=slack,source_xy_error_m=
                        candidate_rollout['full_lap_xy_max_m'])
                    raise RuntimeError('source-certified restoration checkpoint accepted')
        if objective_mode in ('clearance', 'smooth'):
            merit = quality_proposal_merit(residual, objective, initial_objective)
            if merit is not None and (best_quality_proposal[0] is None or
                    merit < best_quality_proposal[0][0]):
                proposal = (merit, np.asarray(opti.debug.value(x)),
                            np.asarray(opti.debug.value(u)), residual, objective,
                            int(iteration))
                best_quality_proposal[0] = proposal
                save_npz_checkpoint(output/'best_quality_proposal.npz',
                    schema='production-command-ocp/v1', state=proposal[1],
                    controls=proposal[2], dt_s=.025,
                    lap_time_s=count*.025, source_sha256=digest,
                    max_constraint_violation=residual, iteration=proposal[5],
                    objective_mode=objective_mode, objective_value=objective,
                    baseline_objective=initial_objective, proposal_merit=merit)
        if residual < significant[0]*.98:
            significant[:] = [residual, int(iteration)]
        if residual <= 1.e-5 and objective < significant_objective[0]-1.e-5*max(
                1., abs(significant_objective[0]) if math.isfinite(significant_objective[0]) else 1.):
            significant_objective[0] = objective
            significant[1] = int(iteration)
        if stagnation_iterations and iteration-significant[1] >= stagnation_iterations:
            stagnated[0] = True
            raise RuntimeError('command search stagnated; durable best iterate saved')
    opti.callback(callback)
    started = time.monotonic()
    error = None
    try: opti.solve_limited()
    except RuntimeError as exception: error = str(exception).splitlines()[-1]
    # CasADi converts exceptions thrown inside a Python callback (including
    # KeyboardInterrupt) into solver errors. Check the signal flag OUTSIDE its
    # C++ stack so a service stop cannot be swallowed as another failed solve.
    if cancel_requested is not None and cancel_requested():
        raise KeyboardInterrupt('campaign interrupted; durable primal retained')
    stats = opti.stats()
    flat = np.asarray(opti.debug.value(opti.x)).reshape(-1)
    residual = violation(flat)
    objective = float(evaluate_f(flat))
    states, controls = np.asarray(opti.debug.value(x)), np.asarray(opti.debug.value(u))
    if best[0] is not None and (early_refinement[0] is not None or
            iterate_rank(best[0][2], best[0][3]) < iterate_rank(residual, objective)):
        states, controls, residual, objective = best[0]
    rollout = check_full_witness(function, states, controls, grid, margin, repo=repo,
                                trace_path=output/'source_trace.npz')
    passed = (residual <= 1.e-5 and rollout['full_lap_xy_max_m'] <= .05
              and clearance_passes(rollout['minimum_1ms_footprint_slack_m']))
    save_command_checkpoint(output/'witness.npz', states, controls, digest, residual,
                            int(stats.get('iter_count', -1)))
    report = dict(schema=1, status='source_witness_passed' if passed else 'no_validated_witness',
        source_plant_sha256=digest, trial_lap_s=count*.025, elapsed_s=time.monotonic()-started,
        required_margin_m=margin,
        clearance_numerical_tolerance_m=CLEARANCE_NUMERICAL_TOLERANCE_M,
        initial_residual=violation(initial), last_significant_iteration=significant[1],
        termination_reason=('source_clearance_refinement' if early_refinement[0]
            else 'early_source_checkpoint' if early_verified[0]
            else (('objective_stagnation' if residual <= 1.e-5 else
            'feasibility_stagnation') if stagnated[0] else stats.get('return_status'))),
        early_source_checkpoint=early_verified[0],
        source_clearance_refinement=early_refinement[0],
        requested_source_clearance_cuts=(early_refinement[0]['new_cuts']
                                        if early_refinement[0] else []),
        max_constraint_violation=residual, ipopt_status=stats.get('return_status'),
        ipopt_iterations=stats.get('iter_count'), error=error, rollout=rollout,
        jacobian_workers=workers, initializer=str(command_seed or checkpoint),
        objective_scaling_factor=count/.001,
        stagnation_iterations=stagnation_iterations,
        initializer_strategy='drop-one-packet' if shorten_packet else 'resample-or-exact-restart',
        seed_reconditioning=conditioning,
        source_clearance_cuts=[list(item) for item in clearance_cuts],
        objective_mode=objective_mode, objective_value=objective,
        initializer_sha256=hashlib.sha256(Path(command_seed or checkpoint).read_bytes()).hexdigest(),
        map_sha256=hashlib.sha256(map_path.read_bytes()).hexdigest(),
        unity_validated=False, mpc_validated=False, runtime_promoted=False,
        limitations=['Fixed trial duration, not a global minimum-time certificate.',
            'Two-metre ordered initial-geometry basin; free pose on the phase cross-section.',
            'Exact production transition is not Unity private native solver parity.',
            'Perimeter checked at every 1 ms source tick; continuous inter-tick contact and Unity remain unvalidated.'])
    report['quality'] = witness_quality(output/'witness.npz', report)
    if passed:
        report['reference_export'] = export_reference(output/'source_trace.npz', grid, output)
    save_json_checkpoint(output/'report.json', report)
    print(json.dumps(dict(status=report['status'], trial_lap_s=count*.025,
        iterations=report['ipopt_iterations'], elapsed_s=report['elapsed_s'],
        residual=residual, recursive_xy_max_m=rollout['full_lap_xy_max_m'],
        source_tick_footprint_slack_m=rollout['minimum_1ms_footprint_slack_m'],
        report=str(output/'report.json'))), flush=True)
    return report


def time_prediction_seed(states, controls, fraction, time_sectors=1):
    """Round relaxed timing UP to full packets; never certify interpolation."""
    if time_sectors>1:
        return resample_source_time_initialization(states,controls,fraction,time_sectors)
    fraction=float(np.asarray(fraction).reshape(-1)[0])
    count = controls.shape[1]
    if (states.shape != (21, count+1) or controls.shape[0] != 2 or
            not np.isfinite(states).all() or not np.isfinite(controls).all() or
            not math.isfinite(fraction) or not 0. <= fraction <= 1.):
        raise ValueError('invalid time predictor primal')
    predicted_lap = count*(.024+.001*fraction)
    target = int(math.ceil(predicted_lap/.025-1.e-9))
    if not 30 <= target < count:
        return None
    query = np.linspace(0., 1., target+1)
    grid = np.linspace(0., 1., count+1)
    state = np.array([np.interp(query, grid, row) for row in states])
    held = controls[:, np.arange(target, dtype=np.int64)*count//target].copy()
    return state, held, predicted_lap


def solve_time_predictor(*, repo, command_seed, map_path, output, workers=8,
                         iterations=200, stagnation_iterations=80, margin=.05,
                         cancel_requested=None, resume_checkpoint=None,
                         time_sectors=1):
    """Minimum-time predictor followed by an obligatory exact-grid repair.

    Interpolates the fixed 24/25 ms source transition MAPS, not fitted vehicle
    forces. This relaxation has no physical certificate. It makes time, pose,
    hidden states and controls joint decisions from the feasible incumbent.
    Only solve_witness + certify_witness may approve the resampled proposal.
    """
    repo, output, command_seed = map(Path, (repo, output, command_seed))
    margin = validate_margin(margin)
    if output.exists():
        raise FileExistsError('refusing to overwrite minimum-time predictor')
    output.mkdir(parents=True)
    function, digest = production_command_map(repo, repo/'build_task/command_ocp')
    with np.load(command_seed, allow_pickle=False) as data:
        count = data['controls'].shape[1]
    initial_x, initial_u, winding, lap = load_command_seed(command_seed, count)
    opti, x, u, fraction = command_problem(repo, function, digest, initial_x, initial_u,
        winding, MapGrid.load(map_path), margin, workers, time_predictor=True,
        time_sectors=time_sectors)
    seed_hash = hashlib.sha256(command_seed.read_bytes()).hexdigest()
    map_hash = hashlib.sha256(Path(map_path).read_bytes()).hexdigest()
    metadata = Path(map_path).with_suffix('.yaml')
    metadata_hash = hashlib.sha256(metadata.read_bytes()).hexdigest() if metadata.exists() else ''
    if resume_checkpoint is not None:
        with np.load(resume_checkpoint, allow_pickle=False) as saved:
            if (str(saved['schema'].item()) != 'source-time-predictor/v1' or
                    str(saved['source_sha256'].item()) != digest or
                    str(saved['seed_sha256'].item()) != seed_hash or
                    str(saved['map_sha256'].item()) != map_hash or
                    str(saved['map_metadata_sha256'].item()) != metadata_hash or
                    saved['state'].shape != initial_x.shape or
                    np.asarray(saved['fraction']).size != time_sectors or
                    (int(saved['time_sectors']) if 'time_sectors' in saved else 1) != time_sectors):
                raise ValueError('time predictor checkpoint contract differs')
            opti.set_initial(x, saved['state'])
            opti.set_initial(u, saved['controls'])
            opti.set_initial(fraction, np.asarray(saved['fraction']).reshape(-1))
    opti.solver('ipopt', {'expand': False, 'print_time': False}, {
        'max_iter': iterations, 'hessian_approximation': 'limited-memory',
        'mu_strategy': 'adaptive', 'bound_relax_factor': 0.,
        'tol': 1.e-6, 'acceptable_tol': 1.e-5, 'print_level': 5})
    evaluator = ca.Function('time_predictor_constraints', [opti.x], [opti.g])
    lower = np.asarray(opti.value(opti.lbg, opti.initial())).reshape(-1)
    upper = np.asarray(opti.value(opti.ubg, opti.initial())).reshape(-1)
    def violation(flat):
        values = np.asarray(evaluator(flat)).reshape(-1)
        return float(np.max(np.maximum(np.maximum(lower-values, values-upper), 0.)))
    best, proposals, progress, proposal_progress = [None], {}, [lap, 0], [float('inf')]
    invalid_timing = dict(count=0, last=None)
    def checkpoint(name, candidate, iteration):
        save_npz_checkpoint(output/name, schema='source-time-predictor/v1',
            state=candidate['state'], controls=candidate['controls'], fraction=candidate['fraction'],
            time_sectors=time_sectors,
            lap_time_s=candidate['lap_s'], source_sha256=digest, seed_sha256=seed_hash,
            map_sha256=map_hash, map_metadata_sha256=metadata_hash,
            max_constraint_violation=candidate['residual'], iteration=iteration, source_certified=False)
    def observe(iteration, initial=False, terminal=False):
        if cancel_requested is not None and cancel_requested():
            raise RuntimeError('time predictor interrupted')
        if not initial and not terminal and iteration % 5:
            return
        getter = (lambda value: opti.value(value, opti.initial())) if initial else opti.debug.value
        flat = np.asarray(getter(opti.x)).reshape(-1)
        residual = violation(flat)
        theta = np.asarray(getter(fraction),dtype=float).reshape(-1)
        try:
            _,time_edges=predictor_timing(count,theta,time_sectors)
        except ValueError:
            # IPOPT may visit infeasible sector bounds, including at its final
            # iterate. Such a trial is not a usable timing proposal. Do not
            # clip it, weaken the domain check, or crash the whole campaign.
            # Invalid initial/resumed contracts still fail explicitly.
            if initial or theta.size != time_sectors:
                raise
            invalid_timing['count'] += 1
            invalid_timing['last'] = dict(iteration=int(iteration),
                sector_fractions=[float(v) if np.isfinite(v) else None for v in theta])
            print(json.dumps(dict(event='time_predictor_iterate_skipped',
                **invalid_timing['last'], reason='invalid sector timing decision',
                terminal=terminal, source_certified=False)), flush=True)
            return
        duration = float(time_edges[-1])
        rank = (residual > 1.e-4, duration if residual <= 1.e-4 else residual)
        current = dict(state=np.asarray(getter(x)), controls=np.asarray(getter(u)),
                       fraction=theta.copy(), residual=residual, lap_s=duration, rank=rank)
        if np.isfinite(residual) and (best[0] is None or rank < best[0]['rank']):
            best[0] = current
            checkpoint('best_predictor.npz', current, iteration)
        target = int(math.ceil(duration/.025-1.e-9))
        # A useful initial guess need not pass the relaxed NLP's feasibility
        # gate. Preserve shorter iterates for exact-map comparison/repair;
        # never call them source-feasible merely because they improve time.
        if (np.all((0. <= theta) & (theta <= 1.)) and 30 <= target < count
                and np.isfinite(residual)
                and (target not in proposals or residual < proposals[target]['residual'])):
            proposals[target] = current
            checkpoint(f'time_candidate_{target:04d}.npz', current, iteration)
            if residual <= min(c['residual'] for c in proposals.values()):
                checkpoint('best_time_candidate.npz', current, iteration)
        if residual <= 1.e-4 and duration < progress[0]-1.e-5:
            progress[:] = [duration, iteration]
        print(json.dumps(dict(stage='minimum-time predictor', iteration=iteration,
            source_lap_s=lap, predicted_lap_s=duration,
            time_sectors=time_sectors, sector_fraction_min=float(theta.min()),
            sector_fraction_max=float(theta.max()),
            max_constraint_violation=residual, source_certified=False)), flush=True)
        if 30 <= target < count and residual < proposal_progress[0]*.98:
            # Once a whole-packet gain is proposed, allow its feasibility to
            # improve before demanding another fully feasible time iterate.
            proposal_progress[0] = residual
            progress[1] = iteration
        if not initial and not terminal and stagnation_iterations and iteration-progress[1] >= stagnation_iterations:
            raise RuntimeError('no feasible time improvement; predictor primal retained')
    print(json.dumps(dict(stage='minimum-time predictor setup', output=str(output.resolve()), intervals=count,
        source_lap_s=lap, initial_residual=violation(
            np.asarray(opti.value(opti.x, opti.initial())).reshape(-1)),
        max_iterations=iterations, workers=workers, time_sectors=time_sectors,
        source_certified=False)), flush=True)
    observe(0, initial=True)
    opti.callback(observe)
    started, error = time.monotonic(), None
    try:
        opti.solve_limited()
    except RuntimeError as exception:
        error = str(exception).splitlines()[-1]
    if cancel_requested is not None and cancel_requested():
        raise KeyboardInterrupt('time predictor interrupted; checkpoint retained')
    # Include the terminal iterate even when it is between callback samples.
    observe(int(opti.stats().get('iter_count', 0)), terminal=True)
    candidate = best[0]
    report = dict(status='no_time_proposal', mode=('source-24-25ms-sector-chord/v2'
                  if time_sectors>1 else 'source-24-25ms-chord/v1'),
                  time_sectors=time_sectors,
        source_certified=False, unity_validated=False, runtime_promoted=False,
        required_margin_m=margin,
        source_plant_sha256=digest, initializer=str(command_seed),
        predicted_lap_s=candidate['lap_s'] if candidate else lap,
        source_lap_s=lap, max_constraint_violation=candidate['residual'] if candidate else None,
        ipopt_iterations=opti.stats().get('iter_count'), ipopt_status=opti.stats().get('return_status'),
        elapsed_s=time.monotonic()-started, error=error,
        invalid_timing_iterates=invalid_timing['count'],
        last_invalid_timing=invalid_timing['last'],
        limitation='Interpolated transition predictor only; exact 25ms repair and certification required.')
    comparisons = []
    selected = None
    for target in sorted(proposals):
        point = proposals[target]
        state, held, predicted_lap = time_prediction_seed(
            point['state'],point['controls'],point['fraction'],time_sectors=time_sectors)
        state, conditioning = recondition_command_seed(function, state, held, workers)
        defect = conditioning['scaled_defect_after']
        baseline_x, baseline_u, _, _ = load_command_seed(command_seed, target)
        _, baseline_conditioning = recondition_command_seed(function, baseline_x, baseline_u, workers)
        baseline_defect = baseline_conditioning['scaled_defect_after']
        comparisons.append(dict(intervals=target, exact_initial_dynamic_defect=defect,
            resampled_initial_dynamic_defect=baseline_defect, predictor_residual=point['residual'],
            proposed_conditioning=conditioning, baseline_conditioning=baseline_conditioning))
        if defect < baseline_defect and (selected is None or defect < selected[3]):
            selected = (state, held, predicted_lap, defect, baseline_defect, point['residual'])
    report['initializer_comparisons'] = comparisons
    report['retained_shorter_proposals'] = len(proposals)
    report['proposal_selection_reason'] = (
        'A shorter proposal has lower exact-map initialization defect than plain resampling.'
        if selected is not None else
        'Shorter relaxed proposals did not improve exact-map initialization defect versus resampling.'
        if proposals else
        'No iterate proposed a whole 25 ms packet reduction; the incumbent remains the best feasible predictor.')
    if selected is not None:
        state, held, predicted_lap, defect, baseline_defect, predictor_residual = selected
        seed = output/'proposed_seed.npz'
        save_command_checkpoint(seed, state, held, digest, defect, int(report['ipopt_iterations']))
        report.update(status='uncertified_time_proposal', proposed_seed=str(seed),
            proposed_intervals=held.shape[1], trial_lap_s=held.shape[1]*.025,
            predicted_lap_s=predicted_lap, exact_initial_dynamic_defect=defect,
            resampled_initial_dynamic_defect=baseline_defect, predictor_residual=predictor_residual,
            best_feasible_predictor_residual=report['max_constraint_violation'],
            max_constraint_violation=predictor_residual)
    save_json_checkpoint(output/'report.json', report)
    print(json.dumps(report), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, default=Path(__file__).resolve().parents[1])
    for name in ('checkpoint', 'mesh', 'audit', 'map', 'output'):
        parser.add_argument('--'+name, type=Path, required=name in ('map', 'output'))
    parser.add_argument('--command-seed', type=Path,
                        help='Resample an earlier command witness as an uncertified initializer')
    parser.add_argument('--shorten-packet', action='store_true',
                        help='Remove one source-selected packet instead of resampling all hidden history')
    parser.add_argument('--validate-only', action='store_true',
                        help='Recheck --command-seed and export only if source gates pass; no IPOPT')
    parser.add_argument('--time-predictor', action='store_true',
                        help='Minimum-time source-map relaxation; creates an uncertified exact-grid seed')
    parser.add_argument('--resume-predictor', type=Path)
    parser.add_argument('--recondition-seed', action='store_true',
                        help='Initialize internal histories by source propagation before the exact NLP')
    parser.add_argument('--workers', type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument('--refine-clearance-passes', type=int, default=0,
                        help='Add exact source-tick wall cuts after a failed margin check; bounded extra solves')
    parser.add_argument('--intervals', type=int, default=197)
    parser.add_argument('--iterations', type=int, default=80)
    parser.add_argument('--cpu-seconds', type=float, default=120.)
    parser.add_argument('--margin', type=float, default=.05,
                        help='Required footprint clearance beyond the vehicle envelope, in metres')
    args = parser.parse_args()
    if args.intervals < 30 or args.iterations < 1 or args.cpu_seconds <= 0:
        parser.error('positive budgets and at least 30 intervals required')
    if args.command_seed is None and not all((args.checkpoint, args.mesh, args.audit)):
        parser.error('provide --command-seed or all of --checkpoint --mesh --audit')
    if args.command_seed is not None and any((args.checkpoint, args.mesh, args.audit)):
        parser.error('choose one initializer source, not both')
    if args.refine_clearance_passes < 0 or args.refine_clearance_passes > 10:
        parser.error('clearance refinement must be in [0,10]')
    if args.output.exists(): parser.error('output already exists')
    if args.time_predictor:
        if args.command_seed is None or args.validate_only or args.shorten_packet:
            parser.error('--time-predictor requires --command-seed and its own output')
        solve_time_predictor(repo=args.repo_root.resolve(), command_seed=args.command_seed,
            map_path=args.map, output=args.output, workers=args.workers,
            iterations=args.iterations, resume_checkpoint=args.resume_predictor,
            margin=args.margin)
        return
    if args.validate_only:
        if args.command_seed is None: parser.error('--validate-only requires --command-seed')
        certify_witness(args.repo_root.resolve(), args.command_seed, args.map, args.output,
                        margin=args.margin)
        return
    seed, cuts = args.command_seed, set()
    for index in range(args.refine_clearance_passes+1):
        destination = (args.output/f'pass_{index:02d}' if args.refine_clearance_passes
                       else args.output)
        print(json.dumps(dict(clearance_refinement_pass=index, source_cuts=len(cuts))), flush=True)
        report = solve_witness(repo=args.repo_root.resolve(),
            checkpoint=args.checkpoint if seed is None else None,
            mesh=args.mesh if seed is None else None,
            audit=args.audit if seed is None else None,
            map_path=args.map, output=destination,
            count=args.intervals, iterations=args.iterations, cpu_seconds=args.cpu_seconds,
            command_seed=seed, workers=args.workers, clearance_cuts=sorted(cuts),
            recondition_seed=args.recondition_seed and index == 0,
            shorten_packet=args.shorten_packet and index == 0, margin=args.margin)
        if args.refine_clearance_passes:
            (args.output/'summary.json').write_text(json.dumps(dict(
                latest_pass=index, latest_report=str(destination/'report.json'),
                status=report['status'], trial_lap_s=report['trial_lap_s'],
                runtime_promoted=False, unity_validated=False), indent=2)+'\n')
        if (report['status'] == 'source_witness_passed' or
                report['max_constraint_violation'] > 1.e-3 or
                report['rollout']['full_lap_xy_max_m'] > .05):
            break
        discovered = {(int(item['interval']), int(item['tick'])) for item in
                      report['rollout']['violating_source_ticks']}
        if not discovered-cuts: break
        cuts.update(discovered)
        seed = destination/'witness.npz'


if __name__ == '__main__':
    main()
