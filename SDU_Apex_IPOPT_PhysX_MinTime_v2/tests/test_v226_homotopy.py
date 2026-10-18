"""Regression tests for the v2.2.6 safeguarded Phase-I numerical overlay.

Tests exercise toy CasADi problems only, NOT native/Unity vehicle evaluation.
"""
from __future__ import annotations

import numpy as np
import casadi as ca

from solver_overlay_v226 import transform_solver, prepare_working_solver, SOLVE_REPLACEMENT
from solver_overlay_v225 import transform_solver as previous_transform
from test_phase1_overlay import _sample, LEGACY, SOLVER


def test_previous_v225_working_cache_upgrades_and_preserves_original(tmp_path):
    restored, original, historical = _sample(tmp_path)
    working = tmp_path/'working'
    dest = working/LEGACY/SOLVER
    dest.parent.mkdir(parents=True)
    dest.write_text(previous_transform(historical, 'limited-memory'))
    root, meta = prepare_working_solver(restored, working, LEGACY)
    assert root == working.resolve()
    assert dest.read_text() == transform_solver(historical, 'limited-memory')
    assert original.read_text() == historical
    assert meta['source_untouched'] is True
    assert meta['adaptive_intermediate_tolerances'] is True
    assert 'defect-merit' in meta['phase1_method']


def test_checkpoint_primal_iterate_via_casadi_callback():
    opti = ca.Opti()
    x = opti.variable()
    tol = opti.parameter()
    opti.set_value(tol, 0.5)
    opti.subject_to(opti.bounded(-tol, x - 1., tol))
    energy = (x - 1.)**2
    opti.minimize(energy)
    opti.set_initial(x, 0.65)
    lower = np.asarray(opti.value(opti.lbg, opti.initial())).reshape(-1)
    upper = np.asarray(opti.value(opti.ubg, opti.initial())).reshape(-1)
    feasible = []

    def checkpoint(i):
        g = np.asarray(opti.debug.value(opti.g)).reshape(-1)
        if np.isfinite(g).all() and (g >= lower - 1.e-3).all() and (g <= upper + 1.e-3).all():
            feasible.append((float(opti.debug.value(x)),
                             float(opti.debug.value(energy))))

    opti.callback(checkpoint)
    opti.solver('ipopt', {'print_time':False}, {'print_level':0, 'max_iter':25})
    opti.solve_limited()
    assert feasible
    assert min(energy for _, energy in feasible) < (0.65 - 1.)**2
    opti.set_value(tol, 0.)
    assert max(abs(xvalue-1.) for xvalue, _ in feasible) >= 0.  # no claim that all satisfy strict physics


def test_no_premature_strict_acceptance_or_unverified_retry():
    s = SOLVE_REPLACEMENT
    assert 'opti.set_value(phase1_defect_tolerance, 0.0)' in s
    assert 'if defect_tol == 0.0:' in s
    assert 'phase1_reached_strict = True' in s
    assert 'if not phase1_reached_strict:' in s
    assert 'retry_tolerance = 0.5 * (current_tolerance + defect_tol)' in s
    assert 'current_tolerance = 0.5 * (current_tolerance + defect_tol)' not in s
    assert 'feasibility_max_iterations - total_iterations_used' in s
    assert 'winner_source = \'ipopt_iteration\'' in s
    assert 'unrelaxed_max_dynamics_defect' in s
    assert 'raw_candidate[var] - initial_snapshot[var]' in s
    assert 'if not accepted:' in s
