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
from map_track import MapGrid, _FOOTPRINT_LOCAL, _FOOTPRINT_UNSAMPLED_CORRECTION
from path_frame import path_frame_numpy
from physics import SourcePlant


STATE_SCALES = np.array([1., 1., 1., 2., 1., 2., *([150.]*4),
                          .02, 1., .12, 2., .1, 2., .35, *([.04]*4)])


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
    control_index = np.minimum(u.shape[1]-1, (query[:-1]*u.shape[1]).astype(int))
    winding = round((x[2, -1]-x[2, 0])/(2*math.pi))
    if abs(winding) != 1: raise ValueError('initializer is not one track circuit')
    controls = u[:, control_index].copy()
    if count != u.shape[1] and dense_path.exists():
        # End rack is a physically meaningful initial target; continuous rack
        # interpolation was not a realizable command in the spatial problem.
        # This is only initialization, never a post-solve control correction.
        controls[0] = states[16, 1:]
    return states, controls, winding, u.shape[1]*.025


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
        violations = []
        trace = [state[:, 0].copy()]
        for k in range(controls.shape[1]):
            interval_minimum, worst_tick = float('inf'), None
            for tick in range(1, 26):
                current = plant.step(*controls[:, k], duration_s=.001)
                trace.append(np.r_[current.x, current.y, current.yaw, plant.input.state[:18]])
                slack = grid.footprint_clearance(
                    current.x, current.y, current.yaw, margin=margin)
                smallest = min(smallest, slack)
                if slack < interval_minimum: interval_minimum, worst_tick = slack, tick
            if interval_minimum < 0.:
                violations.append(dict(interval=k, tick=worst_tick, slack_m=interval_minimum))
        result['minimum_1ms_footprint_slack_m'] = smallest
        result['independent_1ms_vs_25ms_xy_endpoint_m'] = float(np.hypot(
            plant.state.x-predicted[0], plant.state.y-predicted[1]))
        result['source_tick_perimeter_validated'] = smallest >= 0.
        result['violating_source_ticks'] = violations
        if trace_path is not None:
            if trace_path.exists(): raise FileExistsError('source trace already exists')
            np.savez_compressed(trace_path, state=np.asarray(trace).T, dt_s=.001,
                                commands=controls, command_dt_s=.025)
            result['source_trace'] = str(trace_path)
    return result


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


def certify_witness(repo, path, map_path, output):
    """Recheck a stored witness using this build; no solve or certificate transfer."""
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
    rollout = check_full_witness(function, states, controls, grid, .05, repo=repo,
                                trace_path=output/'source_trace.npz')
    passed = (abs(winding) == 1 and max(defect, periodic_max) <= 1.e-5
        and rollout['full_lap_xy_max_m'] <= .05 and rollout['minimum_1ms_footprint_slack_m'] >= 0.)
    report = dict(schema=1, status='source_witness_passed' if passed else 'no_validated_witness',
        source_plant_sha256=digest, original_witness=str(path),
        original_witness_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        trial_lap_s=controls.shape[1]*.025, max_local_dynamic_defect=defect,
        max_periodicity_defect=periodic_max, rollout=rollout,
        unity_validated=False, mpc_validated=False, runtime_promoted=False)
    np.savez_compressed(output/'witness.npz', schema='production-command-ocp/v1',
        state=states, controls=controls, dt_s=.025, lap_time_s=controls.shape[1]*.025,
        max_constraint_violation=max(defect, periodic_max), source_sha256=digest)
    if passed: report['reference_export'] = export_reference(output/'source_trace.npz', grid, output)
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key:report[key] for key in ('status', 'trial_lap_s',
        'max_local_dynamic_defect', 'max_periodicity_defect')}), flush=True)
    return report


def solve_witness(*, repo, checkpoint, mesh, audit, map_path, output,
                  count=197, iterations=80, cpu_seconds=120., margin=.05,
                  command_seed=None, workers=8, clearance_cuts=()):
    if output.exists(): raise FileExistsError('refusing to overwrite command OCP')
    output.mkdir(parents=True)
    function, digest = production_command_map(repo, repo/'build_task/command_ocp')
    initial_x, initial_u, winding, source_lap = (
        load_command_seed(command_seed, count) if command_seed is not None else
        load_seed(checkpoint, mesh, audit, count, repo))
    grid = MapGrid.load(map_path)
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
        for tick in (6, 12, 18, 24)]]
    for interval, tick in clearance_cuts:
        if not isinstance(interval, int) or not 0 <= interval < count:
            raise ValueError('invalid clearance-cut interval')
        pose_parts.append(production_fixed_prefix(repo, repo/'build_task/command_ocp',
            digest, ca.vertcat(x[:, interval], u[:, interval]), tick)[:3])
    poses = ca.horzcat(*pose_parts)
    pose_count = poses.size2()
    c, s = ca.cos(poses[2, :]), ca.sin(poses[2, :])
    map_c, map_s = math.cos(grid.yaw), math.sin(grid.yaw)
    for along, side in _FOOTPRINT_LOCAL:
        wx = poses[0, :]+along*c-side*s-grid.origin[0]
        wy = poses[1, :]+along*s+side*c-grid.origin[1]
        local = ca.vertcat(map_c*wx+map_s*wy, -map_s*wx+map_c*wy)
        distance = wall_distance.map(pose_count, 'serial')(local)
        opti.subject_to(distance >= margin+_FOOTPRINT_UNSAMPLED_CORRECTION)
    opti.minimize(.001*ca.sumsqr((x-initial_x)/scales)/count
                 +.001*ca.sumsqr(u-initial_u)/count
                 +.0001*ca.sumsqr(u[:, 1:]-u[:, :-1])/count)
    opti.solver('ipopt', {'expand': False, 'print_time': False}, {
        'max_iter': iterations, 'max_cpu_time': cpu_seconds,
        'hessian_approximation': 'limited-memory', 'mu_strategy': 'adaptive',
        'bound_relax_factor': 0.,
        'tol': 1.e-6, 'acceptable_tol': 1.e-5, 'print_level': 5})
    evaluate_g = ca.Function('command_feasibility', [opti.x], [opti.g])
    lower = np.asarray(opti.value(opti.lbg, opti.initial())).reshape(-1)
    upper = np.asarray(opti.value(opti.ubg, opti.initial())).reshape(-1)
    def violation(flat):
        values = np.asarray(evaluate_g(flat)).reshape(-1)
        return float(np.max(np.maximum(np.maximum(lower-values, values-upper), 0.)))
    initial = np.asarray(opti.value(opti.x, opti.initial())).reshape(-1)
    print(json.dumps(dict(stage='production-command witness', intervals=count,
        trial_lap_s=count*.025, source_lap_s=source_lap, states=21,
        decisions=opti.nx, constraints=opti.ng, initial_residual=violation(initial),
        max_iterations=iterations, cpu_seconds=cpu_seconds, workers=workers,
        source_clearance_cuts=len(clearance_cuts),
        source_sha256=digest)), flush=True)
    best = [None]
    def callback(iteration):
        if iteration % 5: return
        flat = np.asarray(opti.debug.value(opti.x)).reshape(-1)
        residual = violation(flat)
        print(json.dumps(dict(iteration=int(iteration), trial_lap_s=count*.025,
                               max_constraint_violation=residual)), flush=True)
        if residual <= 1.e-4:
            best[0] = (np.asarray(opti.debug.value(x)), np.asarray(opti.debug.value(u)), residual)
    opti.callback(callback)
    started = time.monotonic()
    error = None
    try: opti.solve_limited()
    except RuntimeError as exception: error = str(exception).splitlines()[-1]
    stats = opti.stats()
    flat = np.asarray(opti.debug.value(opti.x)).reshape(-1)
    residual = violation(flat)
    states, controls = np.asarray(opti.debug.value(x)), np.asarray(opti.debug.value(u))
    if residual > 1.e-4 and best[0] is not None:
        states, controls, residual = best[0]
    rollout = check_full_witness(function, states, controls, grid, margin, repo=repo,
                                trace_path=output/'source_trace.npz')
    passed = (residual <= 1.e-4 and rollout['full_lap_xy_max_m'] <= .05
              and rollout['minimum_1ms_footprint_slack_m'] >= 0.)
    np.savez_compressed(output/'witness.npz', schema=np.asarray('production-command-ocp/v1'),
        state=states, controls=controls, dt_s=.025, lap_time_s=count*.025,
        max_constraint_violation=residual, source_sha256=digest)
    report = dict(schema=1, status='source_witness_passed' if passed else 'no_validated_witness',
        source_plant_sha256=digest, trial_lap_s=count*.025, elapsed_s=time.monotonic()-started,
        max_constraint_violation=residual, ipopt_status=stats.get('return_status'),
        ipopt_iterations=stats.get('iter_count'), error=error, rollout=rollout,
        jacobian_workers=workers, initializer=str(command_seed or checkpoint),
        source_clearance_cuts=[list(item) for item in clearance_cuts],
        initializer_sha256=hashlib.sha256(Path(command_seed or checkpoint).read_bytes()).hexdigest(),
        map_sha256=hashlib.sha256(map_path.read_bytes()).hexdigest(),
        unity_validated=False, mpc_validated=False, runtime_promoted=False,
        limitations=['Fixed trial duration, not a global minimum-time certificate.',
            'Two-metre ordered initial-geometry basin; free pose on the phase cross-section.',
            'Exact production transition is not Unity private native solver parity.',
            'Perimeter checked at every 1 ms source tick; continuous inter-tick contact and Unity remain unvalidated.'])
    if passed:
        report['reference_export'] = export_reference(output/'source_trace.npz', grid, output)
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(status=report['status'], trial_lap_s=count*.025,
        iterations=report['ipopt_iterations'], elapsed_s=report['elapsed_s'],
        residual=residual, recursive_xy_max_m=rollout['full_lap_xy_max_m'],
        source_tick_footprint_slack_m=rollout['minimum_1ms_footprint_slack_m'],
        report=str(output/'report.json'))), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, default=Path(__file__).resolve().parents[1])
    for name in ('checkpoint', 'mesh', 'audit', 'map', 'output'):
        parser.add_argument('--'+name, type=Path, required=name in ('map', 'output'))
    parser.add_argument('--command-seed', type=Path,
                        help='Resample an earlier command witness as an uncertified initializer')
    parser.add_argument('--validate-only', action='store_true',
                        help='Recheck --command-seed and export only if source gates pass; no IPOPT')
    parser.add_argument('--workers', type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument('--refine-clearance-passes', type=int, default=0,
                        help='Add exact source-tick wall cuts after a failed margin check; bounded extra solves')
    parser.add_argument('--intervals', type=int, default=197)
    parser.add_argument('--iterations', type=int, default=80)
    parser.add_argument('--cpu-seconds', type=float, default=120.)
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
    if args.validate_only:
        if args.command_seed is None: parser.error('--validate-only requires --command-seed')
        certify_witness(args.repo_root.resolve(), args.command_seed, args.map, args.output)
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
            command_seed=seed, workers=args.workers, clearance_cuts=sorted(cuts))
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
