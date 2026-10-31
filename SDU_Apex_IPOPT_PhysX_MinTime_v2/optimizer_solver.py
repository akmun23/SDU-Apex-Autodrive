"""Native IPOPT callback stops with explicit reasons, not deliberate errors.

Opti's Python callback has a void return type. Use the solver callback instead
so a normal stop returns 1. Preserve Opti's variable scaling when reading the
native primal; do not change equations, derivatives or solver tolerances.
"""
from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import dataclass
import json
import sys

import casadi as ca
import numpy as np


STOP_MESSAGES = {
    'objective_stagnation': 'insufficient objective progress; best checkpoint retained; proceeding to the next attempt',
    'feasibility_stagnation': 'insufficient feasibility progress; best checkpoint retained; proceeding to the next attempt',
    'time_stagnation': 'insufficient time-search progress; predictor checkpoints retained for exact-grid repair',
    'source_clearance_refinement': 'new 1 ms wall checks found; checked checkpoint retained; rebuilding constraints',
    'early_source_checkpoint': 'requested source-replay target passed; proceeding to independent certification',
    'campaign_interrupted': 'campaign interruption signal received; checkpoints retained',
}


class SolverCallback(ca.Callback):
    def __init__(self, opti, observe):
        ca.Callback.__init__(self)
        self.opti, self.observe = opti, observe
        self.iteration, self.reason = -1, None
        self.initialization = None
        self.readers = {}
        self.scale = np.asarray(opti.x_linear_scale).reshape(-1)
        self.offset = np.asarray(opti.x_linear_scale_offset).reshape(-1)
        self.initial = np.asarray(opti.value(opti.x, opti.initial())).reshape(-1)
        self.construct('owned_optimizer_callback')

    def get_n_in(self):
        return ca.nlpsol_n_out()

    def get_n_out(self):
        return 1

    def get_name_in(self, i):
        return ca.nlpsol_out(i)

    def get_name_out(self, _i):
        return 'stop'

    def get_sparsity_in(self, i):
        key = ca.nlpsol_out(i)
        size = (self.opti.nx if key in ('x', 'lam_x') else
                self.opti.ng if key in ('g', 'lam_g') else
                self.opti.np if key == 'lam_p' else 1)
        return ca.Sparsity.dense(size, 1)

    def get_sparsity_out(self, _i):
        return ca.Sparsity.scalar()

    def eval(self, args):
        self.iteration += 1
        raw = np.asarray(args[0]).reshape(-1)
        physical = self.offset+self.scale*raw
        if self.initialization is None:
            scaled_shift = np.abs((physical-self.initial)/self.scale)
            self.initialization = dict(
                scaled_primal_shift_max=float(scaled_shift.max()),
                largest_shift_coordinate=int(scaled_shift.argmax()))
        # Evaluate graph expressions at this primal, not Opti's stale debug
        # cache. A concatenation of several Opti variables cannot be used as
        # the LHS of value(..., assignments), so compile small cached readers.
        def getter(expression):
            key=hash(expression)
            if key not in self.readers:
                self.readers[key]=(expression,ca.Function('primal_reader',[self.opti.x],[expression]))
            retained,reader=self.readers[key]
            if not ca.is_equal(retained,expression):
                raise RuntimeError('primal-reader identity collision')
            return reader(physical)
        reason = self.observe(self.iteration, getter)
        if reason is not None:
            if reason not in STOP_MESSAGES:
                raise ValueError('unknown controlled solver stop: '+str(reason))
            self.reason = reason
            print(json.dumps(dict(event='solver_automatic_stop', reason=reason,
                iteration=self.iteration, message=STOP_MESSAGES[reason])), flush=True)
        return [1 if self.reason else 0]


class SolverLog:
    """Translate only a known callback-stop footer; stream all real errors."""
    def __init__(self, target, callback):
        self.target, self.callback, self.pending = target, callback, ''

    def write(self, text):
        self.pending += text
        while '\n' in self.pending:
            line, self.pending = self.pending.split('\n', 1)
            if (line.strip() == 'EXIT: Stopping optimization at current point as requested by user.'
                    and self.callback.reason):
                line = 'EXIT: Automatic stop: '+STOP_MESSAGES[self.callback.reason]+'.'
            self.target.write(line+'\n')
        return len(text)

    def flush(self):
        self.target.flush()

    def finish(self):
        if self.pending:
            self.target.write(self.pending)
            self.pending = ''
        self.target.flush()

    def __getattr__(self, name):
        return getattr(self.target, name)


@dataclass
class SolverOutcome:
    reason: str | None
    error: str | None
    stop_iteration: int | None
    initialization: dict | None
    options: dict
    native_bounds: bool


def solve_with_controlled_stop(opti, observe, options):
    callback = SolverCallback(opti, observe)
    # Exact box-constraint extraction removes redundant inequality/slack rows
    # and keeps line-search trials inside the plant's throttle/timing domain.
    # Avoid moving a certified boundary seed materially into the interior.
    options={'bound_push':1.e-9,'bound_frac':1.e-9,'bound_relax_factor':0.,**options}
    opti.solver('ipopt', dict(expand=False, print_time=False, iteration_callback=callback,
                             detect_simple_bounds=True), options)
    stream = SolverLog(sys.stdout, callback)
    error = None
    try:
        with redirect_stdout(stream):
            try:
                opti.solve_limited()
            except RuntimeError as exception:
                # Opti regards User_Requested_Stop as a non-success. Only a
                # reason explicitly returned by our callback makes it normal.
                if callback.reason is None:
                    error = str(exception).splitlines()[-1]
    finally:
        stream.finish()
    return SolverOutcome(callback.reason, error,
                         callback.iteration if callback.reason else None,
                         callback.initialization, dict(options), True)
