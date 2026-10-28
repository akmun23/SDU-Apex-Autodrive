#!/usr/bin/env python3
"""Progress-aligned racing time accounting; not a causal or physics fit.

Use Unity's lap-counter receipt epochs to select complete scored laps, and
the MPC's command-application state epochs for the motion accounting. Report
the clock/boundary/projection remainder explicitly instead of assigning it
to a vehicle force or claiming controller-state estimates are ground truth.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def time_densities(u, v, epsi, ey, curvature, reference_u):
    """Seconds/metre: speed, outside-path length, heading/lateral motion.

    ds/dt = (u cos(epsi) - v sin(epsi)) / (1 - kappa*ey).
    The three terms telescope to model dt/ds minus reference dt/ds.
    """
    values = (u, v, epsi, ey, curvature, reference_u)
    if not all(math.isfinite(float(x)) for x in values):
        raise ValueError('nonfinite motion state')
    forward = u*math.cos(epsi) - v*math.sin(epsi)
    metric = 1. - curvature*ey
    if u <= 0. or reference_u <= 0. or forward <= 0. or metric <= 0.:
        raise ValueError('time accounting requires positive forward progress')
    return np.array([1./u - 1./reference_u,
                     -curvature*ey/u,
                     metric*(1./forward - 1./u)], dtype=float)


def clipped_samples(samples, start, end):
    samples = np.asarray(samples, dtype=float)
    if (samples.ndim != 2 or samples.shape[1] != 6 or len(samples) < 2 or
            not np.isfinite(samples).all() or
            np.any(np.diff(samples[:, 0]) <= 0.) or
            not samples[0, 0] <= start < end <= samples[-1, 0]):
        raise ValueError('sample epochs must bracket the complete scoring window')
    times = np.r_[start, samples[(samples[:, 0] > start) &
                                 (samples[:, 0] < end), 0], end]
    return np.column_stack([times, *[
        np.interp(times, samples[:, 0], samples[:, i]) for i in range(1, 6)]])


def account(samples, reference_s, reference_u, reference_kappa, length, bin_width):
    """Integrate every observed interval; never silently drop backward jumps."""
    samples = np.asarray(samples, dtype=float)
    s = np.r_[reference_s, length]
    speed = np.r_[reference_u, reference_u[0]]
    kappa = np.r_[reference_kappa, reference_kappa[0]]
    if (length <= 0 or bin_width <= 0 or len(reference_s) < 3 or
            not math.isclose(float(s[0]), 0., abs_tol=1.e-8) or
            np.any(np.diff(s) <= 0) or np.any(speed <= 0)):
        raise ValueError('invalid periodic reference geometry/speed')
    bins = np.zeros((math.ceil(length/bin_width), 7))
    backward = 0
    largest_gap = 0.
    for first, last in zip(samples[:-1], samples[1:]):
        dt, ds = last[:2] - first[:2]
        largest_gap = max(largest_gap, dt)
        if dt <= 0: raise ValueError('non-increasing state epochs')
        cuts = [0., 1.]
        if ds > 0:
            # Split at periodic sector boundaries, including the lap seam.
            lap0, lap1 = math.floor(first[1]/length), math.floor(last[1]/length)
            for lap in range(lap0, lap1+1):
                for boundary in np.r_[np.arange(0., length, bin_width), length]:
                    fraction = (lap*length + boundary-first[1])/ds
                    if 0. < fraction < 1.: cuts.append(float(fraction))
        elif ds < 0:
            backward += 1
        cuts = sorted(set(cuts))
        for lo, hi in zip(cuts[:-1], cuts[1:]):
            midpoint = first + .5*(lo+hi)*(last-first)
            station = midpoint[1] % length
            uref, curve = np.interp(station, s, speed), np.interp(station, s, kappa)
            # Samples: epoch, progress, horizontal forward/lateral velocity,
            # body-heading error, lateral path displacement.
            terms = time_densities(*midpoint[2:6], curve, uref)
            progress = ds*(hi-lo)
            elapsed = dt*(hi-lo)
            reference_time = progress/uref
            effects = progress*terms
            remainder = elapsed-reference_time-float(np.sum(effects))
            index = min(len(bins)-1, int(station/bin_width))
            bins[index] += [elapsed, reference_time, *effects, remainder, progress]
    segments = np.diff(s)
    midpoint_lap = float(np.sum(2.*segments/(speed[:-1]+speed[1:])))
    return dict(bins=bins, reference_midpoint_integral_s=midpoint_lap,
                backward_progress_intervals=backward,
                largest_state_epoch_gap_s=largest_gap)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bag', type=Path, required=True)
    parser.add_argument('--cycles', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--trajectory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--laps', type=int, default=10)
    parser.add_argument('--bin-width', type=float, default=1.)
    args = parser.parse_args()
    if args.laps < 1 or args.bin_width <= 0. or not math.isfinite(args.bin_width):
        parser.error('positive lap count and bin width required')
    if args.output.exists(): parser.error('refusing to overwrite an analysis')
    try:
        from tools import evaluate_mpc_bag as evaluator
    except ModuleNotFoundError:
        import evaluate_mpc_bag as evaluator
    bag = evaluator.read_bag(args.bag)
    transitions = {}
    previous = None
    for row in bag[evaluator.LAP]:
        count = int(row['value'])
        if previous is not None and count < previous:
            raise ValueError('lap-counter reset inside capture')
        if count not in transitions: transitions[count] = float(row['bag_time'])
        previous = count
    start, end = transitions[1], transitions[args.laps+1]
    summary = json.loads(args.report.read_text())['summary']
    laps = summary['recorded_lap_times_s'][1:args.laps+1]
    if len(laps) != args.laps: raise ValueError('incomplete Unity lap-time evidence')
    fault = summary['scored_through_first_fault_epoch_s']
    if fault is not None and float(fault) < end:
        raise ValueError('requested scoring window crosses the first fault')
    samples = []
    with args.cycles.open() as stream:
        for row in csv.DictReader(stream):
            try:
                values = [float(row['scheduled_command_stamp_ns'])*1.e-9,
                    *[float(row[k]) for k in
                      ('progress_m', 'u_mps', 'v_mps', 'e_psi_rad', 'e_y_m')]]
                if values[0] > 0 and np.isfinite(values).all(): samples.append(values)
            except (KeyError, TypeError, ValueError):
                continue
    # A repeated application epoch is one state epoch, not extra driving time.
    samples = np.array(list({x[0]: x for x in samples}.values()))
    samples = samples[np.argsort(samples[:, 0])]
    selected = clipped_samples(samples, start, end)
    reference = np.loadtxt(args.trajectory, delimiter=',')
    length = float(reference[-1, 0] + np.linalg.norm(reference[-1, 1:3]-reference[0, 1:3]))
    result = account(selected, reference[:, 0], reference[:, 5],
                     reference[:, 4], length, args.bin_width)
    bins = result.pop('bins')
    total = bins.sum(axis=0)
    scored_time, clock_time = sum(laps), end-start
    ideal = result['reference_midpoint_integral_s']*args.laps
    effects = dict(zip(('forward_speed_s', 'outside_path_s',
                       'heading_lateral_s', 'timing_projection_remainder_s'),
                      (total[2:6]/args.laps).tolist()))
    # Explicitly reconcile accounting quadrature/boundary phase and different
    # receipt-clock versus Unity-lap clocks. Never bury these in a force fit.
    effects['boundary_and_reference_quadrature_s'] = (total[1]-ideal)/args.laps
    effects['unity_vs_receipt_clock_s'] = (scored_time-clock_time)/args.laps
    gap = (scored_time-ideal)/args.laps
    sectors = [dict(start_m=i*args.bin_width, end_m=min(length, (i+1)*args.bin_width),
        elapsed_s_per_lap=float(b[0]/args.laps),
        reference_s_per_lap=float(b[1]/args.laps),
        loss_s_per_lap=float((b[0]-b[1])/args.laps),
        **dict(zip(('forward_speed_s', 'outside_path_s', 'heading_lateral_s',
                    'timing_projection_remainder_s'), (b[2:6]/args.laps).tolist())))
        for i, b in enumerate(bins)]
    result.update(schema=1, bag=str(args.bag), laps=args.laps,
        cycle_csv_sha256=sha256(args.cycles), trajectory_sha256=sha256(args.trajectory),
        state_source='MPC command-application prediction, not Unity truth',
        scoring_window_receipt_epochs=[start, end], scored_unity_time_s=scored_time,
        achieved_mean_lap_s=scored_time/args.laps, reference_to_execution_gap_s=gap,
        effects_per_lap=effects, accounting_closure_error_s=sum(effects.values())-gap,
        observed_progress_laps=float((selected[-1, 1]-selected[0, 1])/length),
        sectors=sectors,
        limitations=['Kinematic accounting is not causal attribution.',
                    'Controller states contain observer/localization and timing error.',
                    'Midpoint quadrature, receipt epoch phase and Unity clock differences are explicit.',
                    'A source-feasible reference is not a Unity tracking certificate.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('achieved_mean_lap_s',
        'reference_midpoint_integral_s', 'reference_to_execution_gap_s',
        'effects_per_lap', 'accounting_closure_error_s')}, indent=2))
    print('Worst sectors:', [(s['start_m'], round(1000*s['loss_s_per_lap'], 2))
        for s in sorted(sectors, key=lambda s:s['loss_s_per_lap'], reverse=True)[:6]])


if __name__ == '__main__':
    main()
