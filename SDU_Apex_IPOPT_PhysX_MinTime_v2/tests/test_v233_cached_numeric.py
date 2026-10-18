"""Direct regression for the generated v2.3.3 vectorized CasADi evaluator.

Small algebraic NLP only. No PhysX rollout or full-lap simulation.
"""
import numpy as np
import casadi as ca
import textwrap

from solver_overlay import SOLVE_ELASTIC, transform_solver, transform_solver_v232
from tests.test_phase1_overlay import mock_historical_source


def test_previous_working_overlay_is_reconstructible():
    old = transform_solver_v232(mock_historical_source(), 'limited-memory')
    new = transform_solver(mock_historical_source(), 'limited-memory')
    assert old != new
    assert 'lambda expr: opti.value(expr, opti.initial())' in old
    assert 'phase1_vectorized_evaluation' in new
    assert "elastic_initial_values[elastic] = values.copy()" in new


def test_vectorized_numeric_evaluator_matches_opti_initial_and_strict_cap():
    n=4
    opti=ca.Opti()
    x=opti.variable(16,n)
    qdelta=opti.variable(n)
    throttle=opti.variable(n)
    phase_lengths=opti.variable(2)
    normal_load=opti.variable(4,n)
    accel_xy=opti.variable(2,n)
    slack=opti.variable(1,n)
    cap=opti.parameter()
    opti.set_value(cap, 100)
    # All variables are connected and ordered as in the recovered solver.
    res=x[2,:]-qdelta.T-2
    opti.subject_to(res-slack==0)
    opti.subject_to(opti.bounded(-cap,slack,cap))
    opti.subject_to(opti.bounded(0,x,10))
    opti.subject_to(opti.bounded(-2,qdelta,2))
    opti.subject_to(opti.bounded(0.01,throttle,1))
    opti.subject_to(ca.sum1(phase_lengths)==5)
    opti.subject_to(normal_load>=1)
    opti.subject_to(opti.bounded(-100,accel_xy,100))
    opti.minimize(ca.sumsqr(slack))
    seed_x_values=np.tile(np.arange(16, dtype=float).reshape(16,1)/2,(1,n))
    seed_qdelta=np.linspace(.05,.15,n)
    seed_throttle=np.full(n,.1)
    phase_initial_lengths=np.array([2.,3.])
    seed_load=np.full((4,n),8.)
    seed_ax=np.full(n,.5)
    seed_ay=np.full(n,.7)
    ell=(seed_x_values[2,:]-seed_qdelta-2).reshape(1,n)
    for var, val in [(x,seed_x_values),(qdelta,seed_qdelta),(throttle,seed_throttle),
                     (phase_lengths,phase_initial_lengths), (normal_load,seed_load),
                     (accel_xy,np.vstack((seed_ax,seed_ay))),(slack,ell)]:
        opti.set_initial(var,val)
    # These match the names embedded in the solver overlay, not implementation
    # approximations. Extract the generated code verbatim and execute it.
    marker='        # CasADi Opti.initial() reconstructs substitutions'
    assert marker in SOLVE_ELASTIC
    start=SOLVE_ELASTIC.index(marker)
    end=SOLVE_ELASTIC.index('        def eligible(diag):', start)
    snippet=textwrap.dedent(SOLVE_ELASTIC[start:end])
    elastic_vars=[slack]
    elastic_initial_values={slack:ell}
    elastic_pairs=[(slack,res)]
    elastic_merit=ca.sumsqr(slack)
    physical_vars=(x,qdelta,throttle,phase_lengths,normal_load,accel_xy,slack)
    constraint_labels=['sample']*int(opti.g.numel())
    NLP_FEASIBILITY_TOLERANCE=1e-3
    def _constraint_diagnostics(g,l,u,labels):
        v=max(float(np.max(l-g)),float(np.max(g-u)),0.)
        return {'max_constraint_violation':v,'worst_constraints':[]}
    env=dict(globals(), **locals())
    exec(snippet,env)
    fast_g,fast_l,fast_h,fast_raw,fast_merit=env['numeric_evaluation'](
        env['primal_vector'](env['snapshot_current']()))
    # Independently check selected rows and bounds, including CasADi's
    # column-major vectorization and the parameterized slack cap.
    np.testing.assert_allclose(fast_g[:n],0.,atol=1e-12)
    np.testing.assert_allclose(fast_g[n:2*n],ell.reshape(-1,order='F'))
    np.testing.assert_allclose(fast_g[2*n:2*n+16*n],
                               seed_x_values.reshape(-1,order='F'))
    np.testing.assert_allclose(fast_l[:n],0.)
    np.testing.assert_allclose(fast_h[:n],0.)
    np.testing.assert_allclose(fast_l[n:2*n],-100.)
    np.testing.assert_allclose(fast_h[n:2*n],100.)
    np.testing.assert_allclose(fast_raw,ell.flatten(order='F'),atol=1e-12)
    np.testing.assert_allclose(fast_merit,float(np.sum(ell**2)),atol=1e-12)
    assert env['phase1_measure']()['max_constraint_violation']<1e-9
    opti.set_value(cap,0)
    assert env['phase1_measure']()['max_constraint_violation']>1e-2
    # A modified feasible numeric snapshot must evaluate without calling
    # opti.initial() in the fast code path.
    new=env['snapshot_current']()
    new[qdelta][:]=.08
    env['set_snapshot'](new)
    assert np.isfinite(env['phase1_measure']()['raw_max_dynamic_defect'])
