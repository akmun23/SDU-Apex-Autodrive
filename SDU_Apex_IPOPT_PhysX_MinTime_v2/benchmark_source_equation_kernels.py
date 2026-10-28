#!/usr/bin/env python3
"""Compare force value/Jacobian/exact-Hessian cost without running IPOPT."""
import argparse
import json
from pathlib import Path
import time

import casadi as ca
import numpy as np

from recover import load_modules
from source_vehicle import current_model, bind_current_wheel_moment_arms
from source_equation_kernels import make_force_kernel


def benchmark(seed, *, nodes=32, repeats=15, jit=False):
    package = Path(__file__).resolve().parent
    old, wheel, _, _, _ = load_modules(
        package.parent, package, phase1_hessian='exact',
        phase1_formulation='source-equalities')
    model, _ = current_model(package.parent, wheel)
    bind_current_wheel_moment_arms(package.parent, old, wheel)
    with np.load(seed, allow_pickle=False) as archive:
        values = np.vstack((*[archive[name] for name in (
            'u_mps', 'v_mps', 'yaw_rate_radps', 'steering_rad')],
            archive['wheel_omega_radps'], archive['normal_load_n'],
            *[archive[name] for name in (
                'body_height_m', 'body_height_rate_mps', 'roll_rad',
                'roll_rate_radps', 'pitch_rad', 'pitch_rate_radps')]))
    indices = np.linspace(0, values.shape[1]-1, nodes, dtype=int)
    point = values[:, indices]
    state = ca.MX.sym('state', 18, nodes)
    multipliers = ca.DM(np.linspace(.01, 1., 22*nodes))
    measurements, outputs = {}, {}
    variants = [('inline_mx', wheel.tire_forces_casadi),
                ('scalar_sx_kernel', make_force_kernel(
                    model, wheel.tire_forces_casadi, jit=False))]
    if jit:
        variants.append(('compiled_sx_kernel', make_force_kernel(
            model, wheel.tire_forces_casadi, jit=True)))
    for name, force in variants:
        started = time.perf_counter()
        columns = []
        for k in range(nodes):
            z = state[:, k]
            result = force(model, z[0], z[1], z[2], z[3],
                           [z[i] for i in range(4, 8)],
                           [z[i] for i in range(8, 12)],
                           *[z[i] for i in range(12, 18)])
            columns.append(ca.vertcat(*result[:6],
                                     *[ca.vertcat(*v) for v in result[6:]]))
        forces = ca.vec(ca.horzcat(*columns))
        vector = ca.vec(state)
        expressions = {
            'force': forces, 'jacobian': ca.jacobian(forces, vector),
            'hessian': ca.hessian(ca.dot(multipliers, forces), vector)[0]}
        functions = {kind: ca.Function(f'bench_{name}_{kind}', [state], [expr])
                     for kind, expr in expressions.items()}
        measurements[name] = {'construction_s': time.perf_counter()-started}
        outputs[name] = {}
        for kind, function in functions.items():
            outputs[name][kind] = function(point)
            samples = []
            for _ in range(repeats):
                started = time.perf_counter()
                function(point)
                samples.append((time.perf_counter()-started)*1000.)
            measurements[name][kind] = {
                'median_ms': float(np.median(samples)),
                'max_ms': max(samples), 'graph_nodes': function.n_nodes()}
    parity = {}
    for name, _ in variants[1:]:
        parity[name] = {}
        for kind in expressions:
            a = np.asarray(outputs['inline_mx'][kind].nonzeros())
            b = np.asarray(outputs[name][kind].nonzeros())
            # Sparse structural parity precedes comparison of nonzero order.
            assert outputs['inline_mx'][kind].sparsity() == outputs[name][kind].sparsity()
            np.testing.assert_allclose(a, b, rtol=2.e-11, atol=2.e-8)
            parity[name][kind] = float(np.max(np.abs(a-b), initial=0.))
    return {'nodes': nodes, 'repeats': repeats,
            'timings': measurements, 'maximum_absolute_differences': parity,
            'physics_changed': False, 'full_nlp_speedup_measured': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', required=True, type=Path)
    parser.add_argument('--nodes', default=32, type=int)
    parser.add_argument('--repeats', default=15, type=int)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--jit', action='store_true')
    args = parser.parse_args()
    if args.nodes < 1 or args.repeats < 1:
        parser.error('nodes and repeats must be positive')
    report = benchmark(args.seed, nodes=args.nodes, repeats=args.repeats, jit=args.jit)
    text = json.dumps(report, indent=2)+'\n'
    if args.output:
        args.output.write_text(text)
    print(text, end='')


if __name__ == '__main__':
    main()
