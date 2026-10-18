"""v2.2.7: elastic feasibility and strict physics gate, no simulator invoked."""
from __future__ import annotations
import textwrap
import numpy as np
import casadi as ca
import pytest
from pathlib import Path

from solver_overlay import (transform_solver, prepare_working_solver,
                            CONSTRAINT_ELASTIC, SOLVE_ELASTIC)
from solver_overlay_v226 import transform_solver as v226_transform
from test_phase1_overlay import _sample, LEGACY, SOLVER


def test_restore_and_upgrade_verified_v226_cache(tmp_path):
    restored, original, historical = _sample(tmp_path)
    working = tmp_path/'working'
    dest = working / LEGACY / SOLVER
    dest.parent.mkdir(parents=True)
    dest.write_text(v226_transform(historical, 'limited-memory'))
    result, meta = prepare_working_solver(restored, working, LEGACY)
    patched = dest.read_text()
    assert patched == transform_solver(historical, 'limited-memory')
    assert meta['source_untouched']
    assert meta['physical_dynamics_changed'] is True
    assert 'u=x[2,k]' in meta['physical_dynamics_change']
    assert 'opti.set_value(elastic_cap, 0.0)' in patched
    assert original.read_text() == historical
    again, meta_again = prepare_working_solver(restored, working, LEGACY)
    assert again == result and meta == meta_again


def test_rejects_unknown_solver_modifications(tmp_path):
    restored, original, historical = _sample(tmp_path)
    working = tmp_path/'working'/LEGACY/SOLVER
    working.parent.mkdir(parents=True)
    working.write_text('something unverified')
    with pytest.raises(RuntimeError,match='unknown local edits'):
        prepare_working_solver(restored,tmp_path/'working',LEGACY)
    assert working.read_text() == 'something unverified'


def test_hard_wall_is_not_relaxed_and_exact_physics_required():
    op = ca.Opti()
    x = op.variable()
    constraint_labels=[]
    elastic_cap=op.parameter()
    op.set_value(elastic_cap, 1.e5)
    elastic_pairs=[]; elastic_vars=[]; elastic_scaled=[]
    scope=dict(ca=ca, opti=op, constraint_labels=constraint_labels, elastic_cap=elastic_cap,
               elastic_pairs=elastic_pairs, elastic_vars=elastic_vars, elastic_scaled=elastic_scaled)
    exec(textwrap.dedent(CONSTRAINT_ELASTIC),scope)
    add=scope['add_constraint']
    add(x==1.,'lateral_force_balance_k0')
    add(x<=0.8,'right_track_boundary_k0')
    assert constraint_labels==[
        'lateral_force_balance_k0',
        'elastic_bound_lateral_force_balance_k0',
        'right_track_boundary_k0']
    assert len(elastic_pairs)==1
    elastic, residual = elastic_pairs[0]
    op.set_initial(x,0.8)
    op.set_initial(elastic,op.value(residual,op.initial()))
    # Synthetic initial point satisfies the elastic dynamics AND wall bound.
    g=np.asarray(op.value(op.g,op.initial())).reshape(-1)
    lo=np.asarray(op.value(op.lbg,op.initial())).reshape(-1)
    hi=np.asarray(op.value(op.ubg,op.initial())).reshape(-1)
    assert np.all(g>=lo-1e-10) and np.all(g<=hi+1e-10)
    # Physics x=1 cannot simultaneously satisfy wall x<=0.8;
    # the strict zero-defect problem must be rejected.
    op.set_value(elastic_cap,0.)
    lo=np.asarray(op.value(op.lbg,op.initial())).reshape(-1)
    hi=np.asarray(op.value(op.ubg,op.initial())).reshape(-1)
    assert np.max(np.maximum(lo-g,g-hi)) >= .199


def test_elastic_slack_solver_can_recover_feasible_full_physics():
    # This toy has a feasible x=1 physical solution. Starting x=0.6 is
    # hard-feasible at all times and has exactly consistent elastic slack.
    op=ca.Opti(); x=op.variable(); cap=op.parameter()
    op.set_value(cap,1.e5)
    slack=op.variable(); residual=x-1.
    op.subject_to(residual-slack==0)
    op.subject_to(op.bounded(-cap,slack,cap))
    op.subject_to(op.bounded(0.,x,1.2))
    op.set_initial(x,.6)
    op.set_initial(slack,-.4)
    op.minimize(slack**2 + 1.e-4*(x-.6)**2)
    op.solver('ipopt',{'print_time':False},{'print_level':0,'max_iter':80})
    sol=op.solve()
    assert abs(float(sol.value(slack))) < .001
    op.set_initial(x,sol.value(x));op.set_initial(slack,sol.value(slack))
    op.set_value(cap,0.)
    g=np.asarray(op.value(op.g,op.initial())).reshape(-1)
    low=np.asarray(op.value(op.lbg,op.initial())).reshape(-1)
    high=np.asarray(op.value(op.ubg,op.initial())).reshape(-1)
    assert max(np.max(low-g),np.max(g-high),0.) <= .001


def test_must_audit_raw_dynamics_not_only_slack():
    assert "raw_max_dynamic_defect" in SOLVE_ELASTIC
    assert "max_elastic_slack" in SOLVE_ELASTIC
    assert "<= NLP_FEASIBILITY_TOLERANCE" in SOLVE_ELASTIC
    assert 'opti.set_value(elastic_cap, 0.0)' in SOLVE_ELASTIC
    assert "if (strict_ok" in transform_solver(
        __import__('test_phase1_overlay').mock_historical_source(), 'exact')
