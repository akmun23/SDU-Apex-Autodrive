"""Numerical checks of objective-based, hard-safe Phase-I checkpoints.

These are small surrogate constraints. Actual Porto/PhysX validation is external.
"""
import hashlib
import textwrap

import numpy as np
import pytest

from solver_overlay import (
    SOLVE_ELASTIC, SOLVE_ELASTIC_V236, transform_solver,
    transform_solver_v236, prepare_working_solver,
)
from tests.test_phase1_overlay import _sample, LEGACY, SOLVER


def _checked_step():
    code = SOLVE_ELASTIC[
        SOLVE_ELASTIC.index('        checkpoint_alphas ='):
        SOLVE_ELASTIC.index('        def save_feasible_iterate(')]
    code = textwrap.dedent(code)
    def project(values):
        residual = np.asarray(values['x'], dtype=float).reshape(-1)
        if not np.isfinite(residual).all() or np.any(residual < -1) or np.any(residual > 1):
            return None, None
        snapshot = {'x': residual.copy(), 'elastic': residual.copy()}
        return snapshot, {'elastic_merit': float(np.sum(residual ** 2)),
                          'raw_max_dynamic_defect': float(np.max(np.abs(residual))),
                          'max_constraint_violation': 0.0}
    scope = dict(np=np, physical_vars=('x', 'elastic'), projected_trial=project)
    exec(code, scope)
    return scope


def test_checkpoint_priority_is_worst_defect_then_sum_of_squares():
    scope = _checked_step()
    ref = {'x': np.array([.4, .4, .4, .4]), 'elastic': np.array([.4]*4)}
    ref_measure = {'elastic_merit': .64, 'raw_max_dynamic_defect': .4}
    trial = {'x': np.array([.6, 0, 0, 0]), 'elastic': np.array([0.]*4)}
    # L2 improves 0.64 -> 0.36, but the worst residual worsens .4 -> .6;
    # this cannot replace the current candidate used for the strict max gate.
    assert scope['checked_checkpoint'](trial, ref, ref_measure) == (None, None, None)
    good = {'x': np.array([.3]*4), 'elastic': np.array([0.]*4)}
    candidate, diag, alpha = scope['checked_checkpoint'](good, ref, ref_measure)
    assert alpha == 1.
    assert diag['elastic_merit'] == pytest.approx(.36)
    assert diag['raw_max_dynamic_defect'] == pytest.approx(.3)
    assert np.max(np.abs(candidate['x'])) <= .4
    # With the worst defect tied, lower total residual remains a valid tie-break.
    tied = {'x': np.array([.4, .1, .1, .1]), 'elastic': np.zeros(4)}
    candidate, diag, alpha = scope['checked_checkpoint'](tied, ref, ref_measure)
    assert alpha == 1.
    assert diag['raw_max_dynamic_defect'] == pytest.approx(.4)
    assert diag['elastic_merit'] < ref_measure['elastic_merit']


def test_dynamic_breakdown_tracks_original_constraint_names_and_magnitudes():
    snippet = SOLVE_ELASTIC[
        SOLVE_ELASTIC.index('        def defect_breakdown('):
        SOLVE_ELASTIC.index('        def phase1_measure(')]
    code = textwrap.dedent(snippet)
    labels = [
        'periodic_collocation_state_u_mps',
        'elastic_bound_periodic_collocation_state_u_mps',
        'lateral_force_balance_k7',
        'elastic_bound_lateral_force_balance_k7',
        'lateral_force_balance_k9',
        'elastic_bound_lateral_force_balance_k9',
    ]
    scope = {'constraint_labels': labels, 'np': np}
    exec(code, scope)
    rows = scope['defect_breakdown'](np.array([.2, 3.4, -2.1]))
    assert rows[0]['group'] == 'lateral_force_balance'
    assert rows[0]['count'] == 2
    assert rows[0]['max_abs'] == pytest.approx(3.4)
    assert rows[0]['worst_label'] == 'lateral_force_balance_k7'
    assert rows[0]['worst_sample_index'] == 0
    assert rows[1]['group'] == 'periodic_collocation_state_u_mps'
    assert rows[1]['max_abs'] == pytest.approx(.2)
    assert rows[1]['worst_sample_index'] == 0
    # A synthetic test stack without physical labels must not fabricate attributions.
    scope['constraint_labels'] = ['placeholder'] * 3
    fallback = scope['defect_breakdown'](np.array([.1, .2, -.3]))
    assert fallback[0]['group'] == 'unlabeled_residual'


@pytest.mark.parametrize('mode', ('limited-memory', 'exact'))
def test_safe_byte_exact_v236_cache_upgrade(tmp_path, mode):
    restored, original, historical = _sample(tmp_path)
    target = tmp_path / 'working'
    dest = target / LEGACY / SOLVER
    dest.parent.mkdir(parents=True)
    old = transform_solver_v236(historical, mode)
    new = transform_solver(historical, mode)
    assert old != new
    assert 'defect_breakdown' not in old
    assert 'reference_max_defect' not in new
    assert 'phase1_checkpoint_stride' in new
    assert 'feasibility_trust_region)' in new
    dest.write_text(old)
    _, meta = prepare_working_solver(restored, target, LEGACY, mode)
    assert dest.read_text() == new
    assert meta['working_solver_sha256'] == hashlib.sha256(new.encode()).hexdigest()
    assert original.read_text() == historical
    dest.write_text(old + '\n# unknown local modification')
    with pytest.raises(RuntimeError, match='unknown local edits'):
        prepare_working_solver(restored, target, LEGACY, mode)


def test_exact_original_equations_and_strict_final_gate_preserved():
    assert 'physical_dynamics_changed' in open(__import__('pathlib').Path(__file__).resolve().parents[1] / 'solver_overlay.py').read()
    assert 'opti.set_value(elastic_cap, 0.0)' in SOLVE_ELASTIC
    assert 'strict_ok = (eligible(strict_diag)' in SOLVE_ELASTIC
    assert 'hard_viol > NLP_FEASIBILITY_TOLERANCE' in SOLVE_ELASTIC
    assert 'if not strict_ok:' in SOLVE_ELASTIC
    assert 'reference_max_defect' not in SOLVE_ELASTIC
    assert 'if int(iteration) % checkpoint_stride:' in SOLVE_ELASTIC
    assert 'phase1_checkpoint_callback_wall_s' in SOLVE_ELASTIC
    assert "f'Phase-I iter={int(iteration):4d} '" in SOLVE_ELASTIC
    assert "f'status={checkpoint_status} '" in SOLVE_ELASTIC
    assert 'best_max_defect={progress_measure.get("raw_max_dynamic_defect")}' in SOLVE_ELASTIC
    assert 'flush=True' in SOLVE_ELASTIC
    assert SOLVE_ELASTIC_V236 != SOLVE_ELASTIC
