"""Compact CasADi evaluation of unchanged source-derived force equations.

Build one scalar SX kernel, then call it from the spatial MX graph. This lets
CasADi reuse force derivatives instead of expanding the same large expression
at every knot and midpoint. No fitted parameters, physics or bounds change.
"""
from __future__ import annotations

import casadi as ca
import os
import tempfile
import numpy as np


def make_force_kernel(model, tire_forces, *, jit=True):
    state = ca.SX.sym('source_force_state', 18)
    raw = tire_forces(
        model, state[0], state[1], state[2], state[3],
        [state[i] for i in range(4, 8)],
        [state[i] for i in range(8, 12)],
        *[state[i] for i in range(12, 18)])
    expressions = [*raw[:6], *[ca.vertcat(*values) for values in raw[6:]]]
    options = {'cse': True}
    temporary = None
    if jit:
        temporary = tempfile.TemporaryDirectory(prefix='sdu-ipopt-kernel-')
        compiler_options = {'directory': temporary.name+'/',
                            'flags': ['-O3', '-ffp-contract=off'], 'cleanup': False}
        # TemporaryDirectory owns generated files. Derivative functions can
        # outlive the Python wrapper briefly; suppress competing file cleanup.
        derivative_options = {'jit': True, 'jit_cleanup': False, 'compiler': 'shell',
                              'jit_options': compiler_options}
        options.update(derivative_options, der_options=derivative_options)
    kernel = ca.Function('source_force_kernel', [state], expressions, options)
    mapped = {}

    def evaluate(current_model, u, v, yaw_rate, steering, wheel_omega,
                 suspension_forces, body_height=None, body_height_rate=0.0,
                 roll=0.0, roll_rate=0.0, pitch=0.0, pitch_rate=0.0):
        if current_model is not model:
            raise ValueError('source force kernel used with a different model')
        if body_height is None:
            raise ValueError('source force kernel requires explicit body height')
        values = (u, v, yaw_rate, steering, *wheel_omega,
                  *suspension_forces, body_height, body_height_rate,
                  roll, roll_rate, pitch, pitch_rate)
        values = tuple(value if hasattr(value, 'size2') else ca.DM(value)
                       for value in values)
        columns = max(int(value.size2()) for value in values)
        rows = []
        for value in values:
            if value.size1() != 1 or value.size2() not in (1, columns):
                raise ValueError('source force kernel requires matching row arrays')
            rows.append(ca.repmat(value, 1, columns) if value.size2() == 1
                        and columns > 1 else value)
        packed = ca.vertcat(*rows)
        if columns == 1:
            output = kernel(packed)
        else:
            if columns not in mapped:
                mapped[columns] = kernel.map(columns, 'serial')
            output = mapped[columns](packed)
        return (*output[:6], *[tuple(value[i, :] for i in range(4))
                              for value in output[6:]])

    evaluate.temporary_directory = temporary
    return evaluate


class NodeValueCache:
    """Evaluate each immutable export node list once, in a single matrix call."""
    def __init__(self, evaluator):
        self.evaluator = evaluator
        self.cache = {}

    def __call__(self, nodes):
        key = id(nodes)
        if key not in self.cache:
            matrix = ca.horzcat(*[
                ca.vertcat(*item) if isinstance(item, (tuple, list)) else item
                for item in nodes])
            # Retain the list too: a later temporary list must not reuse its id.
            self.cache[key] = (nodes, np.asarray(self.evaluator(matrix), dtype=float))
        return self.cache[key][1]


def scale_solver_coordinates(opti, states, state_scales, steering_rate):
    """Optional solver-coordinate scaling, not a change to physical variables."""
    mode = os.environ.get('SDU_IPOPT_VARIABLE_SCALING', '0')
    if mode not in ('0', '1'):
        raise ValueError('SDU_IPOPT_VARIABLE_SCALING must be 0 or 1')
    if mode == '1':
        opti.set_linear_scale(states, state_scales)
        opti.set_linear_scale(steering_rate, 3.2)
