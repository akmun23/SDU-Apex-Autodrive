"""Regression tests for projected elastic IPOPT checkpoint backtracking.

Tests use a tiny numerical surrogate, not native PhysX or a Porto solve.
"""
import hashlib
import textwrap
import numpy as np
import pytest

from solver_overlay import (
    SOLVE_ELASTIC, SOLVE_ELASTIC_V234,
    transform_solver, transform_solver_v234, prepare_working_solver,
)
from tests.test_phase1_overlay import _sample, LEGACY, SOLVER, mock_historical_source


def _get_checkpoint_function():
    begin = SOLVE_ELASTIC.index('        checkpoint_alphas =')
    end = SOLVE_ELASTIC.index('        def save_feasible_iterate(', begin)
    code = textwrap.dedent(SOLVE_ELASTIC[begin:end])
    physical_vars = ('x', 'elastic')

    def project(values):
        x = float(values['x'][0])
        if x < 0 or x > 1 or not np.isfinite(x):
            return None, None
        # A sample original equality x=0.8, with elastic e=x-0.8.
        raw = x - .8
        snap = {'x': np.array([x]), 'elastic': np.array([raw])}
        return snap, {'elastic_merit': raw*raw,
                      'raw_max_dynamic_defect': abs(raw),
                      'max_constraint_violation': 0.}

    scope = {'physical_vars': physical_vars, 'projected_trial': project, 'np': np}
    exec(code, scope)
    return scope['checked_checkpoint']


def test_backtrack_rejects_full_step_and_finds_feasible_lower_defect():
    checked = _get_checkpoint_function()
    ref = {'x': np.array([.2]), 'elastic': np.array([-.6])}
    trial = {'x': np.array([2.6]), 'elastic': np.array([50.])}
    proposal, diagnostics, alpha = checked(trial, ref, .36)
    assert alpha == pytest.approx(.25)  # x=.8, exact dynamics
    assert proposal['x'][0] == pytest.approx(.8)
    assert proposal['elastic'][0] == pytest.approx(0.)
    assert diagnostics['max_constraint_violation'] == 0
    assert diagnostics['elastic_merit'] < 1e-25


def test_backtrack_does_not_accept_nonimproving_or_invalid_candidates():
    checked = _get_checkpoint_function()
    ref = {'x': np.array([.8]), 'elastic': np.array([0.])}
    worse = {'x': np.array([2.6]), 'elastic': np.array([100.])}
    assert checked(worse, ref, 0.) == (None, None, None)
    bad = {'x': np.array([np.nan]), 'elastic': np.array([100.])}
    assert checked(bad, ref, 0.) == (None, None, None)


def test_v234_cache_exact_upgrade_with_unknown_edit_rejection(tmp_path):
    src = mock_historical_source()
    old = transform_solver_v234(src, 'limited-memory')
    new = transform_solver(src, 'limited-memory')
    assert old != new
    assert 'checkpoint_alphas' not in old
    assert 'checkpoint_alphas' in new
    assert '# An IPOPT trial is accepted only' in old
    assert 'elastic_bound_' in new
    assert "opti.set_value(elastic_cap, 0.0)" in new
    assert "'accepted_hard_feasible_backtracks'" in new
    restored, original, _ = _sample(tmp_path)
    target = tmp_path / 'working'
    dest = target / LEGACY / SOLVER
    dest.parent.mkdir(parents=True)
    dest.write_text(old)
    _, meta = prepare_working_solver(restored, target, LEGACY)
    assert dest.read_text() == new
    assert meta['working_solver_sha256'] == hashlib.sha256(new.encode()).hexdigest()
    assert original.read_text() == src
    dest.write_text(old+'\n# user modification')
    with pytest.raises(RuntimeError, match='unknown local edits'):
        prepare_working_solver(restored, target, LEGACY)


def test_hard_safety_and_final_strict_zero_slack_are_unchanged():
    assert 'if hard_viol > NLP_FEASIBILITY_TOLERANCE:' in SOLVE_ELASTIC
    assert "values[elastic] = raw.copy()" in SOLVE_ELASTIC
    assert "opti.set_value(elastic_cap, 0.0)" in SOLVE_ELASTIC
    assert "strict_ok = (eligible(strict_diag)" in SOLVE_ELASTIC
    assert "if not strict_ok:" in SOLVE_ELASTIC
