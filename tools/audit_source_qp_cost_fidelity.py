#!/usr/bin/env python3
"""Audit the true SOURCE racing objective against its frozen QP cost model.

Uses a source-feasible, recorded 30-stage nominal. Hessian and gradient are
captured before the QP solve (even if that solve fails). Exact source rollouts
of fixed, reproducible physical control perturbations are compared with the
single local quadratic model. This is a diagnostic, not a controller and not
a proof of global quadratic accuracy.
"""
from __future__ import annotations
import argparse
import json
import pathlib
import subprocess
import tempfile
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
BINARY = ROOT / 'build/f1tenth_mpc/mpc_native_source_sqp_replay'
DEFAULT_CASE = ROOT / 'live_runs/mpc_source_second_order_20261104_01/case_15'
SCALES = np.array([0.15, 0.25])


def checked_call(command, env=None):
    import os
    result = subprocess.run(command, text=True, capture_output=True,
                            cwd=ROOT, env={**os.environ, **(env or {})},
                            timeout=90)
    if result.returncode:
        raise RuntimeError(f'command failed ({result.returncode}): {result.stderr[-1500:]}')
    return json.loads(result.stdout)


def audit(case, controls, output, amplitudes):
    u = np.loadtxt(controls)
    if u.shape != (30, 2):
        raise ValueError('expected exactly 30 pairs of physical controls')
    with tempfile.TemporaryDirectory(prefix='source_qp_cost_') as temporary:
        temp = pathlib.Path(temporary)
        prefix = temp / 'first'
        replay = checked_call([str(BINARY), str(case), str(controls), '1',
                               '--full-source-forward', '--stage-jacobians',
                               '--compress-positive-drive-timers', '--gn-hessian'],
                              {'MPC_SOURCE_SQP_DUMP_JACOBIAN': '1',
                               'MPC_SOURCE_SQP_DUMP_PREFIX': str(prefix)})
        hpath = temp / 'first_source_cost_hessian.txt'
        if not hpath.exists():
            raise RuntimeError('source Hessian not available: nominal may be infeasible')
        g = np.loadtxt(temp / 'first_gradient.txt')[:, 1]
        H = np.loadtxt(hpath)
        if g.shape != (60,) or H.shape != (60, 60):
            raise ValueError('wrong gradient/Hessian dimensions')
        base = checked_call([str(BINARY), str(case), str(controls), '1',
                             '--oracle-only'])
        if abs(base['cost'] - replay['initial_cost']) > 0.002:
            raise RuntimeError('source objective scoring mismatch')
        directions = {
            # Stage 20 has interior steering/throttle commands in the
            # recorded witness. Stage 10 is at the throttle lower bound;
            # its symmetric perturbations are physically inadmissible.
            'steering_stage_20': np.eye(60)[40],
            'throttle_stage_20': np.eye(60)[41],
            'coupled_smooth': np.array([
                np.sin(i * .21) if i % 2 == 0 else .2 * np.cos(i * .17)
                for i in range(60)]),
        }
        records = []
        local_derivative_checks = []
        trial_file = temp / 'trial.controls'
        for name, direction in directions.items():
            direction = direction / max(1., np.linalg.norm(direction))
            slope = float(g @ direction)
            curvature = float(direction @ H @ direction)
            # A symmetric derivative comparison is meaningful only when
            # both perturbations remain in the physical input domain.
            epsilon = .001
            signed_costs = []
            for sign in (1., -1.):
                signed = u + sign * epsilon * direction.reshape(30, 2) * SCALES
                admissible = (np.all(abs(signed[:, 0]) <= .5235987756)
                              and np.all(signed[:, 1] >= .001)
                              and np.all(signed[:, 1] <= 1.))
                if not admissible:
                    break
                np.savetxt(trial_file, signed, fmt='%.12f')
                signed_costs.append(checked_call([
                    str(BINARY), str(case), str(trial_file), '1',
                    '--oracle-only'])['cost'])
            if len(signed_costs) == 2:
                measured_slope = (signed_costs[0] - signed_costs[1]) / (2 * epsilon)
                measured_curvature = (signed_costs[0] + signed_costs[1]
                                      - 2 * base['cost']) / (epsilon ** 2)
                local_derivative_checks.append({
                    'direction': name, 'epsilon': epsilon,
                    'source_fd_gradient': measured_slope,
                    'source_ad_gradient': slope,
                    'source_fd_curvature': measured_curvature,
                    'source_gn_curvature': curvature,
                    'gradient_error': abs(slope - measured_slope),
                    'curvature_error': abs(curvature - measured_curvature),
                })
            for alpha in amplitudes:
                trial = u + alpha * direction.reshape(30, 2) * SCALES
                if (np.any(abs(trial[:, 0]) > 0.5235987756 + 1e-6)
                        or np.any(trial[:, 1] < 0.001 - 1e-6)
                        or np.any(trial[:, 1] > 1. + 1e-6)):
                    records.append({'direction': name, 'alpha': alpha,
                                    'within_physical_input_bounds': False})
                    continue
                np.savetxt(trial_file, trial, fmt='%.12f')
                exact = checked_call([str(BINARY), str(case), str(trial_file),
                                      '1', '--oracle-details'])
                slip_energy = sum(
                    sum(float(slip) ** 2 for slip in stage['wheel_slip_excess'])
                    for stage in exact['source_states'][1:])
                actual = float(exact['cost'] - base['cost'])
                predicted = alpha * slope + .5 * alpha * alpha * curvature
                records.append({
                    'direction': name, 'alpha': alpha,
                    'within_physical_input_bounds': True,
                    'actual_cost_change': actual,
                    'qp_predicted_cost_change': predicted,
                    'absolute_cost_error': abs(predicted - actual),
                    'normalized_cost_error': abs(predicted - actual) / max(1., abs(actual)),
                    'predicted_direction_wrong': actual * predicted < 0,
                    'source_wheel_slip_excess_squared_sum': slip_energy,
                    'actual_wall_violation_m2': exact['wall_violation_m2'],
                })
        feasible = [r for r in records if r.get('within_physical_input_bounds')]
        wrong_sign = sum(r['predicted_direction_wrong'] for r in feasible)
        max_error = max((r['normalized_cost_error'] for r in feasible),
                        default=float('inf'))
        result = {'schema': 'source-qp-cost-fidelity/v2',
                  'full_step_cost_model_qualified': wrong_sign == 0 and max_error <= 0.1,
                  'max_allowed_normalized_error': 0.1,
                  'wrong_sign_predictions': wrong_sign,
                  'max_normalized_cost_error': max_error,
                  'nominal': str(controls),
                  'case': str(case),
                  'nominal_source_feasible': base['wall_violation_m2'] == 0,
                  'source_qp_status': replay['qp_status'],
                  'hessian_symmetric_max_error': float(np.max(abs(H-H.T))),
                  'hessian_min_eigenvalue': float(np.linalg.eigvalsh(H)[0]),
                  'local_derivative_checks': local_derivative_checks,
                  'records': records}
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps({'report': str(output),
                          'count': len(records),
                          'wrong_sign': sum(r.get('predicted_direction_wrong', False)
                                            for r in records),
                          'max_normalized_error': max(
                              (r.get('normalized_cost_error', 0) for r in records),
                              default=0)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=pathlib.Path, default=DEFAULT_CASE)
    parser.add_argument('--controls', type=pathlib.Path,
                        default=DEFAULT_CASE / 'strict_feasible_offline_witness_20261104.controls')
    parser.add_argument('--output', type=pathlib.Path,
                        default=ROOT / 'build/source_qp_cost_fidelity.json')
    parser.add_argument('--amplitudes', type=float, nargs='+',
                        default=[.001, .01, .1, .5, 1.])
    args = parser.parse_args()
    audit(args.case, args.controls, args.output, args.amplitudes)


if __name__ == '__main__':
    main()
