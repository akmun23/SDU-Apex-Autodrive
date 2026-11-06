from pathlib import Path
import json

import casadi as ca
import numpy as np
import pytest

from ipopt_nlp_audit import NlpAuditComplete, maybe_audit_checkpoint, summarize_rows


def test_lower_upper_equality_and_dual_sign_conventions():
    result = summarize_rows([0., 2., 3., 4.], [0., -np.inf, 3., -np.inf],
        [np.inf, 2., 3., 5.], [-2., 3., -7., -1.], ['lower', 'upper', 'eq', 'wrong'])
    assert result['max_constraint_violation'] == 0.
    assert result['constraint_groups']['lower']['active'] == 1
    assert result['constraint_groups']['upper']['active'] == 1
    assert result['constraint_groups']['eq']['equalities'] == 1
    assert result['constraint_groups']['eq']['active'] == 0
    assert result['wrong_sign_duals'] == 1


def test_complementarity_is_not_confused_with_feasibility():
    result = summarize_rows([1., 2.1], [0., -np.inf], [np.inf, 2.],
                            [-3., 4.], ['lower_0', 'upper_3'])
    assert result['max_complementarity'] == 3.
    assert result['max_constraint_violation'] == pytest.approx(.1)
    assert result['constraint_groups']['upper']['active'] == 1


def make_problem(tmp_path, monkeypatch):
    opti = ca.Opti()
    x = opti.variable()
    opti.subject_to(x >= 1.)
    opti.minimize(x*x)
    checkpoint = tmp_path/'checkpoint.npz'
    np.savez(checkpoint, decision_x=[1.], constraint_duals=[-2.],
             predicted_lap_time_s=1., max_constraint_violation=0., trust_factor=4.)
    monkeypatch.setenv('SDU_IPOPT_AUDIT_CHECKPOINT', str(checkpoint))
    monkeypatch.setenv('SDU_IPOPT_AUDIT_OUTPUT', str(tmp_path/'audit.json'))
    return opti, x


def test_exact_stationarity_without_invoking_solver(tmp_path, monkeypatch):
    opti, x = make_problem(tmp_path, monkeypatch)
    with pytest.raises(NlpAuditComplete):
        maybe_audit_checkpoint(opti, ['lower'], lap_time=x, factor=4.)
    report = json.loads((tmp_path/'audit.json').read_text())
    assert report['stationarity_physical_coordinates_inf'] == 0.
    assert not report['ipopt_invoked']
    with np.load(tmp_path/'audit.npz', allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive['labels'], ['lower'])
    with pytest.raises(ValueError, match='overwrite'):
        maybe_audit_checkpoint(opti, ['lower'], lap_time=x, factor=4.)


def test_different_graph_fails_closed(tmp_path, monkeypatch):
    opti, x = make_problem(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='differ'):
        maybe_audit_checkpoint(opti, ['lower'], lap_time=x+1., factor=4.)
    assert not (tmp_path/'audit.json').exists()


def test_factor_mismatch_and_disabled_hook(tmp_path, monkeypatch):
    opti, x = make_problem(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='dimensions/trust'):
        maybe_audit_checkpoint(opti, ['lower'], lap_time=x, factor=16.)
    monkeypatch.delenv('SDU_IPOPT_AUDIT_CHECKPOINT')
    maybe_audit_checkpoint(None, None, lap_time=None, factor=None)
