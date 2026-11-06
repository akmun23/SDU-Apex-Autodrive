#!/usr/bin/env python3
"""Independent branch-aware original-source gradient and Lagrangian Hessian audit.

The matrix is FD of original source-AD Lagrangian gradients, with fixed,
certified local-QP multipliers. HVPs use independent directions and half steps;
objective differences additionally check the source AD itself. Branch crossings
are explicit failures of classical-Hessian qualification, not smoothed away.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
from tools import competition_failure_benchmark as benchmark
from tools.qualify_mpc_source_phase1 import SourceOracle, CONTROL_SCALE


def audit(case: Path, tools: Path, step: float = 1e-3) -> dict:
    controls = np.loadtxt(case / 'initial.controls').astype(np.float32)
    oracle = SourceOracle(tools, case)
    try:
        baseline = oracle.evaluate(controls.tolist(), qp_diagnostics=True)
        if not baseline.get('qp_original_duals_trustworthy'):
            raise ValueError('Original physical local QP has no trustworthy multiplier estimate')
        dual = np.asarray(baseline['physical_multipliers'])
        maximum = baseline['max_steering_rad']
        lower, upper = np.tile([-maximum, .001], 30), np.tile([maximum, 1.], 30)
        origin = controls.astype(float).ravel()
        def evaluate(point):
            value = oracle.evaluate(point.reshape(30, 2).tolist(), reference_controls=controls.tolist())
            if not value['valid']:
                raise ValueError('Invalid source curvature point')
            gradient = np.asarray(value['gradient_scaled']) - \
                np.einsum('ij,i->j', np.asarray(value['hard_jacobian_scaled']), dual)
            objective = value['original_cost'] - dual @ np.asarray(value['hard_margins'])
            changes = dict(value['physical_branch_changes'], peak=value['peak_branch_changes'],
                           reference_geometry=value['geometry_branch_changes'])
            return gradient, float(objective), changes
        base_gradient = np.asarray(baseline['gradient_scaled']) - \
            np.einsum('ij,i->j', np.asarray(baseline['hard_jacobian_scaled']), dual)
        matrix = np.zeros((60, 60))
        numerical_gradient = np.zeros(60)
        columns = []
        for column in range(60):
            for shrink in range(4):
                delta = step * (.25 ** shrink) * CONTROL_SCALE[column]
                plus, minus = origin.copy(), origin.copy()
                plus[column] = np.float32(min(upper[column], origin[column] + delta))
                minus[column] = np.float32(max(lower[column], origin[column] - delta))
                forward, f_plus, branch_plus = evaluate(plus)
                backward, f_minus, branch_minus = evaluate(minus)
                smooth = not any(branch_plus.values()) and not any(branch_minus.values())
                if smooth or shrink == 3:
                    break
            denominator = (plus[column] - minus[column]) / CONTROL_SCALE[column]
            if denominator <= 0.:
                raise ValueError('Curvature perturbation rounded away')
            matrix[:, column] = (forward - backward) / denominator
            numerical_gradient[column] = (f_plus - f_minus) / denominator
            columns.append({'column': column, 'normalized_step': denominator / 2.,
                'same_active_branches': smooth, 'forward_branch_changes': branch_plus,
                'backward_branch_changes': branch_minus})
        symmetric = .5 * (matrix + matrix.T)
        # A finite-difference column crossing a source switch cannot define
        # the Hessian at the origin. Retain that matrix as a diagnostic,
        # then verify the analytical reference against independent,
        # branch-compatible gradient differences in unrelated directions.
        reference = oracle.evaluate(controls.tolist(),
            lagrangian_hessian_multipliers=dual.tolist()).get('lagrangian_hessian', {})
        source_matrix = None
        if reference.get('valid'):
            source_matrix = np.asarray(reference['scaled_row_major']).reshape(60, 60)
            if not np.all(np.isfinite(source_matrix)):
                raise ValueError('Nonfinite analytical source Hessian')
        asymmetry = float(np.linalg.norm(matrix - matrix.T) / max(1., np.linalg.norm(matrix)))
        gradient_error = float(np.linalg.norm(numerical_gradient - base_gradient) /
            max(1., np.linalg.norm(numerical_gradient), np.linalg.norm(base_gradient)))
        rng = np.random.default_rng(769)
        hvps = []
        for name in ('steering', 'throttle', 'coupled'):
            direction = rng.normal(size=60)
            if name == 'steering': direction[1::2] = 0.
            if name == 'throttle': direction[::2] = 0.
            direction[(origin - lower < step * CONTROL_SCALE) | (upper - origin < step * CONTROL_SCALE)] = 0.
            direction /= max(1., np.linalg.norm(direction))
            attempts = []
            for shrink in range(7):
                probe_step = step*(.25**shrink)
                plus = (origin + .5*probe_step*CONTROL_SCALE*direction).astype(np.float32).astype(float)
                minus = (origin - .5*probe_step*CONTROL_SCALE*direction).astype(np.float32).astype(float)
                displacement = (plus-minus)/CONTROL_SCALE
                effective_step = float(direction @ displacement)
                if effective_step <= 1e-15:
                    break
                forward, _, branch_plus = evaluate(plus)
                backward, _, branch_minus = evaluate(minus)
                smooth = not any(branch_plus.values()) and not any(branch_minus.values())
                attempts.append({'normalized_probe_step': probe_step,
                    'same_active_branches': smooth,
                    'float_direction_discrepancy_relative': float(np.linalg.norm(
                        displacement/probe_step-direction)),
                    'float_midpoint_offset_inf_scaled': float(np.max(np.abs(
                        (.5*(plus+minus)-origin)/CONTROL_SCALE)))})
                if smooth:
                    break
            if not attempts:
                raise ValueError('Curvature direction rounded away')
            # Recompute at the last representable pair if the next shrink
            # rounded to zero. Never label a zero displacement an HVP.
            probe_step = attempts[-1]['normalized_probe_step']
            plus = (origin+.5*probe_step*CONTROL_SCALE*direction).astype(np.float32).astype(float)
            minus = (origin-.5*probe_step*CONTROL_SCALE*direction).astype(np.float32).astype(float)
            displacement = (plus-minus)/CONTROL_SCALE
            effective_step = float(direction @ displacement)
            realized_direction = displacement/effective_step
            actual, fd_predicted = forward - backward, symmetric @ displacement
            fd_relative = float(np.linalg.norm(actual - fd_predicted) /
                max(1e-9, np.linalg.norm(actual), np.linalg.norm(fd_predicted)))
            predicted = source_matrix @ displacement if source_matrix is not None else None
            relative = None if predicted is None else float(np.linalg.norm(actual - predicted) /
                max(1e-9, np.linalg.norm(actual), np.linalg.norm(predicted)))
            path_score = oracle.evaluate(controls.tolist(),
                constraint_direction=(CONTROL_SCALE*realized_direction).reshape(30,2).tolist())
            model = path_score.get('constraint_path_model', {})
            path_first = path_second = first_error = second_error = None
            if model.get('valid') and 'original_objective_first' in model:
                path_first = float(model['original_objective_first']-dual @ np.asarray(model['first']))
                path_second = float(model['original_objective_second']-dual @ np.asarray(model['second']))
                gradient_first = float(base_gradient @ realized_direction)
                fd_second = float(realized_direction @ actual/effective_step)
                first_error = abs(path_first-gradient_first)/max(1.,abs(path_first),abs(gradient_first))
                second_error = abs(path_second-fd_second)/max(1e-9,abs(path_second),abs(fd_second))
            hvps.append({'direction': name, 'relative_error': relative,
                'finite_difference_matrix_hvp_relative_error': fd_relative,
                'probe_attempts': attempts,
                'normalized_probe_step': probe_step,
                'source_path_curvature_valid': bool(model.get('valid',False)),
                'source_lagrangian_directional_first': path_first,
                'source_lagrangian_directional_second': path_second,
                'source_directional_gradient_relative_error': first_error,
                'source_directional_second_vs_gradient_fd_relative_error': second_error,
                'same_active_branches': not any(branch_plus.values()) and not any(branch_minus.values()),
                'forward_branch_changes': branch_plus, 'backward_branch_changes': branch_minus})
        eigenvalues = np.linalg.eigvalsh(symmetric)
        fd_reference_passed = gradient_error <= .05 and asymmetry <= .05 and \
            all(c['same_active_branches'] for c in columns) and \
            all(h['same_active_branches'] and h['finite_difference_matrix_hvp_relative_error'] <= .05
                for h in hvps)
        result = {'schema': 'original-source-lagrangian-curvature-audit/v2', 'case': str(case),
            'fixed_multipliers': dual.tolist(), 'oracle_evaluations': oracle.calls,
            'gradient_vs_objective_fd_relative_error': gradient_error, 'hessian_antisymmetry_relative': asymmetry,
            'minimum_raw_eigenvalue': float(eigenvalues.min()), 'maximum_raw_eigenvalue': float(eigenvalues.max()),
            'steering_throttle_cross_curvature_norm': float(np.linalg.norm(symmetric[::2, 1::2])),
            'analytical_hessian_valid': bool(reference.get('valid', False)),
            'analytical_hessian_directional_evaluations': reference.get('source_directional_evaluations'),
            'analytical_hessian_quadratic_consistency_relative': reference.get('quadratic_consistency_relative'),
            'finite_difference_reference_passed': fd_reference_passed,
            'qualification_scope': 'three independent branch-compatible analytical HVP and scalar curvature probes; not an all-entries or global smoothness certificate',
            'columns': columns, 'hessian_vector_checks': hvps,
            'passed': bool(reference.get('valid', False)) and
                all(h['same_active_branches'] and h['relative_error'] is not None and h['relative_error'] <= .05 and
                    h['source_directional_gradient_relative_error'] is not None and
                    h['source_directional_gradient_relative_error'] <= .005 and
                    h['source_directional_second_vs_gradient_fd_relative_error'] <= .05
                    for h in hvps)}
        return result
    finally:
        oracle.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--tools', type=Path, default=benchmark.DEFAULT_TOOLS)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--step', type=float, default=1e-3)
    args = parser.parse_args()
    if not np.isfinite(args.step) or args.step <= 0.:
        parser.error('step must be finite and positive')
    result = audit(args.case.resolve(), args.tools.resolve(), args.step)
    benchmark.write_json(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('columns', 'fixed_multipliers')}, indent=2))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
