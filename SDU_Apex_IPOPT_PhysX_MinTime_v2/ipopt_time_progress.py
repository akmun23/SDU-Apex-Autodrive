"""Sampled time-stage incumbents and durable checkpoints, independent of IPOPT.

Only a full current-constraint audit can mark a sampled primal feasible. This
module never changes solver variables, constraints, tolerances or commands.
Snapshots are initializers, not claims of IPOPT convergence or Unity validity.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import time

import numpy as np


def durable_snapshot(path: Path, payload: dict) -> None:
    """Atomic same-directory replacement, fsync data and directory entry."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp.npz')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class TimeStageProgress:
    def __init__(self, root, factor, tolerance, snapshot, *, stride=25,
                 interval_s=120., clock=time.monotonic):
        if (stride < 1 or not math.isfinite(interval_s) or interval_s <= 0.
                or not math.isfinite(tolerance) or tolerance <= 0.):
            raise ValueError('checkpoint stride/interval/tolerance must be positive')
        self.root = Path(root) if root else None
        self.factor, self.tolerance = float(factor), float(tolerance)
        self.snapshot, self.clock = snapshot, clock
        self.stride, self.interval_s = int(stride), float(interval_s)
        self.last_write = clock()
        self.best = None
        self.errors = 0
        self.callback_wall_s = 0.
        self.sample_count = 0
        self.last_iteration = 0

    def observe(self, iteration, *, force=False):
        self.last_iteration = int(iteration)
        if self.root is None or (not force and iteration % self.stride):
            return
        started = self.clock()
        try:
            payload = {key: np.asarray(value).copy()
                       for key, value in self.snapshot().items()}
            required = ('sample_s', 'state_x', 'steering_rate_radps',
                        'throttle_norm', 'phase_lengths_m', 'decision_x',
                        'predicted_lap_time_s', 'max_constraint_violation')
            if any(name not in payload or not np.isfinite(payload[name]).all()
                   for name in required):
                raise ValueError('time-stage checkpoint has missing/nonfinite data')
            payload.update(schema=np.asarray(1, dtype=np.int64),
                           trust_factor=np.asarray(self.factor),
                           iteration=np.asarray(int(iteration), dtype=np.int64))
            lap = float(payload['predicted_lap_time_s'])
            violation = float(payload['max_constraint_violation'])
            if lap <= 0. or violation < 0.:
                raise ValueError('time-stage checkpoint has invalid time/residual')
            feasible = violation <= self.tolerance
            payload['sampled_source_feasible'] = np.asarray(feasible)
            self.sample_count += 1
            if feasible and (self.best is None or lap < float(
                    self.best['predicted_lap_time_s'])):
                self.best = payload
            if force or self.clock()-self.last_write >= self.interval_s:
                prefix = self.root / f'factor_{self.factor:g}'
                durable_snapshot(Path(str(prefix)+'_latest_iterate.npz'), payload)
                if self.best is not None:
                    durable_snapshot(Path(str(prefix)+'_best_feasible_iterate.npz'),
                                     self.best)
                self.last_write = self.clock()
                print('IPOPT durable time checkpoint: '+json.dumps({
                    'trust_factor': self.factor, 'iteration': int(iteration),
                    'objective_lap_time_s': lap,
                    'max_constraint_violation': violation,
                    'best_sampled_feasible_lap_time_s': None if self.best is None
                    else float(self.best['predicted_lap_time_s']),
                    'callback_errors': self.errors}), flush=True)
        except (OSError, RuntimeError, ValueError, KeyError, TypeError, OverflowError) as error:
            self.errors += 1
            # A failed checkpoint must be visible but must not destroy a solve.
            if self.errors <= 3:
                print(f'IPOPT time checkpoint failed: {error}', flush=True)
        finally:
            self.callback_wall_s += self.clock()-started

    def finish(self):
        self.observe(self.last_iteration, force=True)


def choose_sampled_incumbent(best, terminal_time, terminal_violation, tolerance):
    """Never replace a faster feasible terminal iterate or accept invalid data."""
    if best is None:
        return False
    time_s = float(best['predicted_lap_time_s'])
    violation = float(best['max_constraint_violation'])
    if not (math.isfinite(time_s) and time_s > 0. and math.isfinite(violation)
            and 0. <= violation <= tolerance):
        return False
    terminal_feasible = (terminal_violation is not None
        and math.isfinite(float(terminal_violation))
        and 0. <= float(terminal_violation) <= tolerance
        and math.isfinite(float(terminal_time)) and terminal_time > 0.)
    return not terminal_feasible or time_s < terminal_time


def primal_assignments(opti, vector):
    """Opti.value requires symbol assignments, not one concatenated x expression."""
    import casadi as ca
    flat = np.asarray(vector, dtype=float).reshape(-1)
    if flat.size != opti.nx or not np.isfinite(flat).all():
        raise ValueError('sampled primal does not match the current NLP')
    assignments, offset = [], 0
    for variable in ca.symvar(opti.x):
        count = variable.numel()
        value = ca.reshape(ca.DM(flat[offset:offset+count]),
                           variable.size1(), variable.size2())
        assignments.append(variable == value)
        offset += count
    if offset != flat.size:
        raise ValueError('sampled primal contains incomplete decision symbols')
    return assignments


def prefer_terminal_restart(selected_incumbent, terminal_time, terminal_violation,
                            incumbent_time, restart_tolerance):
    """Keep a promising near-feasible search point separate from certified output."""
    return bool(selected_incumbent and terminal_violation is not None
        and math.isfinite(float(terminal_violation))
        and 0. <= float(terminal_violation) <= restart_tolerance
        and math.isfinite(float(terminal_time)) and 0. < terminal_time < incumbent_time)
