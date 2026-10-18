"""Fail-closed diagnostics: preserve zero-slack violations after IPOPT failure.

No native PhysX/Unity rollouts or full-lap solve are performed.
"""
import hashlib
import textwrap
import numpy as np
import pytest

from solver_overlay import (
    SOLVE_ELASTIC, SOLVE_ELASTIC_V233, prepare_working_solver,
    transform_solver, transform_solver_v233,
)
from tests.test_phase1_overlay import _sample, LEGACY, SOLVER, mock_historical_source


def test_previous_working_solver_reconstructed_and_upgraded(tmp_path):
    src = mock_historical_source()
    historical = transform_solver_v233(src, 'limited-memory')
    patched = transform_solver(src, 'limited-memory')
    assert historical != patched
    assert "# We have already audited" in patched
    assert "# We have already audited" not in historical
    assert hashlib.sha256(historical.encode()).hexdigest() != hashlib.sha256(patched.encode()).hexdigest()
    restored, original, _ = _sample(tmp_path)
    target = tmp_path / 'working'
    old_file = target / LEGACY / SOLVER
    old_file.parent.mkdir(parents=True)
    old_file.write_text(historical)
    _, meta = prepare_working_solver(restored, target, LEGACY)
    assert old_file.read_text() == patched
    assert original.read_text() == src
    assert meta['source_untouched']
    assert meta['working_solver_sha256'] == hashlib.sha256(patched.encode()).hexdigest()
    old_file.write_text(patched + '\n# LOCAL HAND EDIT')
    with pytest.raises(RuntimeError, match='unknown local edits'):
        prepare_working_solver(restored, target, LEGACY)


def test_unsolved_phase1_produces_labeled_strict_report_directly():
    assert "if not strict_ok:" in SOLVE_ELASTIC
    assert "phase1_exact_zero_slack_audited" in SOLVE_ELASTIC
    block = SOLVE_ELASTIC.split("        if not strict_ok:\n", 1)[1].split("\n    def stage_values", 1)[0]
    # Exercise the exact injected code's failure path using no OptiSol at all.
    snippet = "def invoke():\n    if not strict_ok:\n" + textwrap.indent(textwrap.dedent(block), '        ')
    class Track:
        def to_dict(self): return {'length_m': 28.41}
    class Model:
        def metadata(self): return {'mass_kg': 3.47}
    import time
    env = dict(
        strict_ok=False, stage1_stats={'success': False},
        stats={'return_status': 'Maximum_Iterations_Exceeded', 'iter_count': 12},
        strict_diag={'max_constraint_violation': 1.064, 'worst_constraints': [
            {'label': 'elastic_bound_lateral_force_balance_k16', 'violation': 1.064}]},
        initial_constraints={'max_constraint_violation': 0.0},
        phase1_elastic_reports=[{'name': 'elastic model-consistent initialization',
                                 'best_raw_max_defect': 1.064}],
        final_error=None, track=Track(), model=Model(),
        time=time, started=time.perf_counter(), brake_mode_seed_source='native-PhysX motion',
    )
    exec(snippet, env)
    candidate, report = env['invoke']()
    assert candidate == {}
    assert report['solver']['feasible_incumbent'] is False
    assert report['solver']['iterations'] == 12
    assert report['nlp_diagnostics']['max_constraint_violation'] == pytest.approx(1.064)
    assert report['nlp_diagnostics']['worst_constraints'][0]['label'] == 'elastic_bound_lateral_force_balance_k16'
    assert report['nlp_diagnostics']['initial_guess']['max_constraint_violation'] == 0
    assert report['phase1_strict_feasible'] is False
    assert report['phase1_exact_zero_slack_audited'] is True
    assert report['solve_stages'][0]['best_raw_max_defect'] == pytest.approx(1.064)
