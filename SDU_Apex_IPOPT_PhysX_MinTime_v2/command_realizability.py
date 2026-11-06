"""Measure spatial OCP versus production-plant 25 ms command realizability.

This is not Unity validation or a closed-loop controller. The entire command
witness is fixed before propagation; there are no future-state resets in a
recursive window. Independent local windows expose the first divergence.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from path_frame import path_frame_numpy, path_progress_numpy
from physics import SourcePlant, source_hash
from source_vehicle import _cpp_number


def time_grid(track, states, expected_lap, *, density=None, steps=None):
    if states.shape != (16, track.count) or not np.isfinite(states).all():
        raise ValueError('expected complete 16-state spatial checkpoint')
    if density is None:
        sdot = path_progress_numpy(track, track.s, *states[:4],
            roll=states[12], pitch=states[14], height_rate=states[11],
            yaw_rate=states[4], pitch_rate=states[15])
        density = 1./sdot
        steps = np.diff(np.r_[track.s, track.length])
    if (np.asarray(density).shape != (track.count,) or
            np.asarray(steps).shape != (track.count,) or
            not np.isfinite(density).all() or np.any(density <= 0) or
            not np.isfinite(steps).all() or np.any(steps <= 0)):
        raise ValueError('non-forward OCP witness')
    dt = .5*steps*(density + np.roll(density, -1))
    times = np.r_[0., np.cumsum(dt)]
    if abs(times[-1]-expected_lap) > 1.e-5:
        raise ValueError(f'rebuilt spatial time {times[-1]} != saved {expected_lap}')
    return times


def periodic(times, values, query):
    values = np.asarray(values)
    return np.interp(np.mod(query, times[-1]), times,
                     np.r_[values, values[0]])


def seed_plant(plant, state, pose):
    # No physical tire or steering constants are fitted to make this pass.
    plant.reset(*pose, state[2])
    plant.input.state[0:3] = state[2:5]
    plant.input.state[3:7] = state[5:9]
    plant.input.state[7:13] = state[10:16]
    plant.input.state[13] = state[9]
    # Jounce reconciles geometrically on the first substep. Sticky/contact
    # history is absent from the spatial OCP, and is explicitly not invented.
    plant.input.suspension_history_initialized = 0
    plant.state.u, plant.state.v, plant.state.r = state[2:5]
    plant.state.steering = state[9]


def metrics(values):
    a = np.asarray(values)
    return dict(rmse=float(np.sqrt(np.mean(a*a))),
                p95_abs=float(np.percentile(np.abs(a), 95)),
                max_abs=float(np.max(np.abs(a))))


def source_rack_history(current, target, rate_degps):
    """VehicleController.Steer ordering, including its asymmetric branch.

    Mirrors unity_source_plant.cpp, not a generic symmetric rate limiter.
    rate is parsed from that source at startup, never fitted to these errors.
    """
    # Exact C header literal and operation grouping matter at the source's
    # asymmetric branch boundary. math.pi/6 differs by one double ULP.
    norm = target/float('0.52359877559829887308')
    result = []
    internal = -current*(180./math.pi)
    target_deg = -norm*30.
    increment = rate_degps*.001
    for _ in range(25):
        if norm > internal/30.:
            internal = max(target_deg, internal-increment)
        else:
            internal = min(target_deg, internal+increment)
        result.append(-internal*(math.pi/180.))
    return np.array(result)


def inverse_rack_mean(current, desired_mean, rate):
    lo, hi = -math.pi/6., math.pi/6.
    best, smallest = current, float('inf')
    # Source branch can have finite jumps. Retain the best actual achievable
    # mean instead of asserting every desired mean has an exact inverse.
    for _ in range(32):
        target = .5*(lo+hi)
        mean = float(np.mean(source_rack_history(current, target, rate)))
        if abs(mean-desired_mean) < smallest:
            best, smallest = target, abs(mean-desired_mean)
        if mean < desired_mean: lo = target
        else: hi = target
    return best


def evaluate(plant, track, states, throttle, times, *, phase=.5, stride=1, rack_rate=None):
    """Each start is independent; each forecast is fully recursive."""
    if phase not in (0., .5, 1., 'source-mean') or stride < 1:
        raise ValueError('invalid target phase/window stride')
    cx, cy, heading = path_frame_numpy(track, track.s)
    x = cx-np.sin(heading)*states[0]
    y = cy+np.cos(heading)*states[0]
    yaw = np.unwrap(heading+states[1])
    winding = round((yaw[-1]-yaw[0])/(2*math.pi))
    yaw_times = np.r_[yaw, yaw[0]+winding*2*math.pi]
    def reference(t):
        cycle = math.floor(t/times[-1])
        query = t % times[-1]
        return np.array([periodic(times, x, query), periodic(times, y, query),
            np.interp(query, times, yaw_times)+cycle*winding*2*math.pi,
            *[periodic(times, row, query) for row in states[2:]]])
    errors = {h: [] for h in (1, 4, 10, 30)}
    rows = []
    starts = range(0, int(times[-1]/.025), stride)
    for start_index in starts:
        t0 = start_index*.025
        ref = reference(t0)
        state = np.array([periodic(times, row, t0) for row in states])
        seed_plant(plant, state, ref[:3])
        for step in range(30):
            t = t0 + step*.025
            if phase == 'source-mean':
                mean = np.mean(periodic(times, states[9], t+np.arange(1, 26)*.001))
                steering = inverse_rack_mean(plant.input.state[13], mean, rack_rate)
            else:
                steering = float(periodic(times, states[9], t+phase*.025))
            q = float(periodic(times, throttle, t))
            if not .001 <= q <= 1.:
                raise ValueError('witness throttle violates positive-drive contract')
            predicted = plant.step(steering, q)
            end = reference(t+.025)
            delta = np.r_[predicted.x-end[0], predicted.y-end[1],
                math.atan2(math.sin(predicted.yaw-end[2]), math.cos(predicted.yaw-end[2])),
                np.asarray(plant.input.state[0:3])-end[3:6],
                np.asarray(plant.input.state[3:7])-end[6:10],
                plant.input.state[13]-end[10]]
            if step+1 in errors:
                errors[step+1].append(delta)
                rows.append([t0, step+1, steering, q, *delta])
    result = {}
    for horizon, samples in errors.items():
        a = np.asarray(samples)
        result[f'{horizon*25}ms'] = dict(windows=len(a),
            xy_m=metrics(np.linalg.norm(a[:, :2], axis=1)),
            **{name: metrics(a[:, i]) for i, name in enumerate(
                ('yaw_rad', 'u_mps', 'v_mps', 'yaw_rate_radps',
                 'wheel_fl_radps', 'wheel_fr_radps', 'wheel_rl_radps',
                 'wheel_rr_radps', 'rack_rad'), start=2)})
    return result, np.asarray(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--mesh', type=Path, required=True)
    ap.add_argument('--nlp-audit', type=Path, required=True,
                    help='Exact rebuilt graph audit .npz, including original node time densities')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--repo-root', type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument('--stride', type=int, default=1)
    args = ap.parse_args()
    if args.output.exists(): ap.error('refusing to overwrite diagnostic')
    table = np.loadtxt(args.mesh, delimiter=',')
    with np.load(args.checkpoint, allow_pickle=False) as data:
        states = np.asarray(data['state_x'])
        throttle = np.asarray(data['throttle_norm']).reshape(-1)
        lap = float(data['predicted_lap_time_s'])
        length = float(np.asarray(data['phase_lengths_m']).sum())
        station = np.asarray(data['sample_s'])
    if table.shape != (states.shape[1], 7) or not np.allclose(station, table[:, 0], atol=1.e-7):
        ap.error('checkpoint and source mesh differ')
    track = SimpleNamespace(count=len(station), s=station, length=length,
                            x=table[:, 1], y=table[:, 2], psi=table[:, 3])
    with np.load(args.nlp_audit, allow_pickle=False) as audit:
        if not np.array_equal(audit['decision_x'], data_primal(args.checkpoint)):
            ap.error('NLP audit belongs to a different checkpoint')
        times = time_grid(track, states, lap,
            density=np.asarray(audit['node_time_density_s_per_m']),
            steps=np.asarray(audit['spatial_steps_m']))
    plant = SourcePlant(args.repo_root.resolve(), args.repo_root/'build_task/source_witness')
    rack_rate = _cpp_number(
        (args.repo_root/'f1tenth_mpc/src/unity_source_plant.cpp').read_text(),
        'kSteeringRateDegps')
    results = {}
    traces = {}
    for phase in (0., .5, 1., 'source-mean'):
        results[str(phase)], traces[str(phase)] = evaluate(
            plant, track, states, throttle, times, phase=phase,
            stride=args.stride, rack_rate=rack_rate)
        print('Command witness target phase', phase,
              json.dumps(results[str(phase)]['750ms']['xy_m']), flush=True)
    report = dict(schema=1, checkpoint_sha256=source_hash([args.checkpoint]),
        mesh_sha256=source_hash([args.mesh]), plant_source_sha256=plant.code_hash,
        predicted_spatial_lap_s=lap, command_hold_s=.025,
        command_timeline='applied epochs; issue each periodic command 50 ms earlier',
        target_phase_within_hold=results, unity_validated=False,
        limitations=['Source-plant consistency diagnostic, not Unity or MPC validation.',
            'No future observation or reset inside a forecast.',
            'Spatial OCP does not contain sticky-tire/jounce history.',
            'Exact graph time densities are required: rounded CSV mesh derivatives cannot reproduce knot-side lookup reliably.',
            'Steering-rate witness is converted to an actual held rack target; no perfect actuator assumed.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    np.savez_compressed(args.output.with_suffix('.npz'), **traces)


def data_primal(path):
    with np.load(path, allow_pickle=False) as archive:
        return np.asarray(archive['decision_x'])


def validate_exported_candidate(candidate, track, repo, output):
    """Default export gate using graph time densities, not rounded CSV data."""
    states = np.vstack([
        candidate['ey_m'], candidate['epsi_rad'], candidate['speed_mps'],
        candidate['lateral_speed_mps'], candidate['yaw_rate_radps'],
        candidate['wheel_omega_radps'], candidate['steering_rad'],
        candidate['body_height_m'], candidate['body_height_rate_mps'],
        candidate['roll_rad'], candidate['roll_rate_radps'],
        candidate['pitch_rad'], candidate['pitch_rate_radps']])
    density = np.asarray(candidate['node_time_density_s_per_m']).reshape(-1)
    steps = np.asarray(candidate['spatial_steps_m']).reshape(-1)
    lap = float(np.sum(.5*steps*(density+np.roll(density, -1))))
    times = time_grid(track, states, lap, density=density, steps=steps)
    plant = SourcePlant(repo, repo/'build_task/source_witness')
    rate = _cpp_number((repo/'f1tenth_mpc/src/unity_source_plant.cpp').read_text(),
                       'kSteeringRateDegps')
    results, traces = {}, {}
    for phase in (.5, 'source-mean'):
        results[str(phase)], traces[str(phase)] = evaluate(plant, track, states,
            np.asarray(candidate['throttle_norm']), times, phase=phase, rack_rate=rate)
    passing = [name for name, values in results.items()
               if all(result['xy_m']['max_abs'] <= .05 for result in values.values())]
    report = dict(schema=1, source_command_xy_gate_passed=bool(passing),
        passing_target_strategies=passing, maximum_xy_error_bound_m=.05,
        source_plant_sha256=plant.code_hash, command_hold_s=.025,
        applied_command_timeline=True, issue_lead_s=.05,
        predicted_spatial_lap_s=lap, strategies=results,
        unity_validated=False, mpc_validated=False,
        feedforward_promoted=False,
        limitations=['XY gate alone is not a velocity/yaw/traction certificate.',
            'Fully recursive local forecasts, not a driven complete lap.',
            'OCP omits persistent sticky/contact history; initial jounce reconciles geometrically.',
            'All commands generated from nominal OCP and internally propagated rack, never future observations.'])
    if output.exists() or output.with_suffix('.npz').exists():
        raise ValueError('refusing to overwrite command witness')
    output.write_text(json.dumps(report, indent=2)+'\n')
    np.savez_compressed(output.with_suffix('.npz'), **traces)
    print('Production-plant 25 ms witness: '+json.dumps({
        'xy_gate_passed': bool(passing),
        '750ms_max_xy_m': {k: v['750ms']['xy_m']['max_abs'] for k,v in results.items()}}), flush=True)
    return report


if __name__ == '__main__':
    main()
