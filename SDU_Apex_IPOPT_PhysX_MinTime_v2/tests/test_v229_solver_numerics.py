"""v2.2.9: guard only infeasible line-search trials; retain exact physics."""
from pathlib import Path
import numpy as np
import casadi as ca
import pytest
from solver_numerics import (
    SPATIAL_SOURCE, SPATIAL_GUARDED, IPOPT_PHASE1_SOURCE,
    IPOPT_PHASE1_BOUNDED, guard_solver,
)
from solver_overlay import transform_solver, transform_solver_228, prepare_working_solver
from test_phase1_overlay import _sample, LEGACY, SOLVER


def historical_fragment():
    return ('    spatial_rhs = []\n' + SPATIAL_SOURCE + '\n'
            + IPOPT_PHASE1_SOURCE + '\n')


def test_undisturbed_feasible_equations_and_off_domain_finiteness():
    # The real recovered source uses exactly these two guards and still
    # imposes the hard den>=0.25 and sdot>=0.30 inequalities.
    stage=guard_solver(historical_fragment())
    assert SPATIAL_SOURCE not in stage
    assert SPATIAL_GUARDED in stage
    assert IPOPT_PHASE1_BOUNDED in stage
    assert stage.count('"detect_simple_bounds": True') == 1
    # Numerically prove reciprocal is unchanged inside the feasible region.
    den=np.array([.25,.50,.8,2.0])
    progress=np.array([.75,.6,2.4,1.0])
    original=progress/den
    stabilized=progress/np.maximum(den,.20)
    np.testing.assert_array_equal(stabilized,original)
    assert (np.maximum(stabilized,.20)==stabilized).all()
    # Outside domain, no pole from longitudinal progress at zero/negative den.
    trial_den=np.array([0.,-10.,1.e-9])
    trial_progress=np.array([0.,-1.,2.])
    trial_sdot=trial_progress/np.maximum(trial_den,.20)
    assert np.isfinite(1./np.maximum(trial_sdot,.20)).all()


def test_solver_guard_fails_closed_when_source_differs():
    s=historical_fragment().replace('"frenet_denominator"','"alternate_symbol"')
    with pytest.raises(RuntimeError,match='Frenet positive-speed division'):
        guard_solver(s)


def test_unchanged_old_solver_cache_is_accepted(tmp_path):
    restored, original, source = _sample(tmp_path)
    target=tmp_path/'working'
    dest=target/LEGACY/SOLVER
    dest.parent.mkdir(parents=True)
    dest.write_text(transform_solver_228(source,'limited-memory'))
    prepare_working_solver(restored,target,LEGACY)
    assert dest.read_text()==transform_solver(source,'limited-memory')
    assert original.read_text()==source


def test_detect_simple_bounds_is_valid_with_parameterized_elastic_caps():
    opti=ca.Opti()
    x=opti.variable();elastic=opti.variable();cap=opti.parameter()
    opti.set_value(cap,2.)
    opti.subject_to(x-elastic==1.)
    opti.subject_to(opti.bounded(-cap,elastic,cap))
    opti.subject_to(opti.bounded(0.,x,3.))
    opti.set_initial(x,0.)
    opti.set_initial(elastic,-1.)
    opti.minimize(elastic**2)
    opti.solver('ipopt',{'detect_simple_bounds':True,'print_time':False},
                {'print_level':0,'max_iter':50})
    sol=opti.solve()
    assert abs(float(sol.value(elastic)))<1e-4
    assert abs(float(sol.value(x))-1.)<1e-4
    # A strict zero-slack audit must use the current bounds, not OptiSol
    opti.set_initial(x,sol.value(x)); opti.set_initial(elastic,sol.value(elastic))
    opti.set_value(cap,0.)
    g=np.asarray(opti.value(opti.g,opti.initial())).reshape(-1)
    lb=np.asarray(opti.value(opti.lbg,opti.initial())).reshape(-1)
    ub=np.asarray(opti.value(opti.ubg,opti.initial())).reshape(-1)
    assert max(float(np.max(lb-g)), float(np.max(g-ub)),0.)<1e-3
