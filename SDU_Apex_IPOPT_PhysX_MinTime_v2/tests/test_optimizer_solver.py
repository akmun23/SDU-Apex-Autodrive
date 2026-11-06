"""Real tiny IPOPT problems test control flow, not vehicle performance."""
import io

import casadi as ca
import numpy as np

from optimizer_solver import SolverLog, solve_with_controlled_stop


def problem():
    opti=ca.Opti()
    x=opti.variable(2)
    opti.set_linear_scale(x,ca.DM([3.,.2]),ca.DM([2.,-1.]))
    opti.set_initial(x,ca.DM([9.,.5]))
    opti.subject_to(opti.bounded(ca.DM([0.,-2.]),x,ca.DM([10.,2.])))
    opti.minimize(ca.sumsqr(x-ca.DM([8.,.1])))
    return opti,x


def test_normal_callback_stop_has_no_traceback_and_preserves_scaled_primal(capsys):
    opti,x=problem();points=[]
    def observe(iteration,getter):
        points.append(np.asarray(getter(x)).reshape(-1))
        if iteration==2:
            return 'feasibility_stagnation'
    outcome=solve_with_controlled_stop(opti,observe,dict(print_level=5,max_iter=30))
    out=capsys.readouterr()
    assert 'Traceback' not in out.err
    assert 'intermediate_callback: Error' not in out.err
    assert 'requested by user' not in out.out
    assert 'EXIT: Automatic stop: insufficient feasibility progress' in out.out
    assert outcome.reason=='feasibility_stagnation' and outcome.error is None
    assert outcome.stop_iteration==2
    assert opti.stats()['return_status']=='User_Requested_Stop'
    np.testing.assert_allclose(points[0],[9.,.5])
    np.testing.assert_allclose(points[-1],np.asarray(opti.debug.value(x)).reshape(-1))


def test_unforced_convergence_keeps_native_success_footer(capsys):
    opti,x=problem()
    outcome=solve_with_controlled_stop(opti,lambda i,get:None,dict(print_level=5,max_iter=50))
    assert outcome.reason is None and outcome.error is None
    assert 'EXIT: Optimal Solution Found.' in capsys.readouterr().out
    np.testing.assert_allclose(np.asarray(opti.debug.value(x)).reshape(-1),[8.,.1],atol=1.e-5)


def test_native_bounds_keep_timing_trials_in_domain_without_large_seed_push():
    opti=ca.Opti();fraction=opti.variable()
    opti.set_initial(fraction,1.)
    opti.subject_to(opti.bounded(0.,fraction,1.))
    opti.minimize((fraction-2.)**2)
    seen=[]
    def observe(i,getter):
        seen.append(float(getter(fraction)))
        if i==5:
            return 'objective_stagnation'
    outcome=solve_with_controlled_stop(opti,observe,dict(print_level=0,max_iter=20))
    assert outcome.native_bounds is True
    assert all(0.<=v<=1. for v in seen)
    assert abs(seen[0]-1.)<1.e-7


def test_real_callback_errors_are_not_hidden_or_labeled_normal(capsys):
    opti,_=problem()
    def broken(iteration,getter):
        raise RuntimeError('fixture unexpected calculation failure')
    outcome=solve_with_controlled_stop(opti,broken,dict(print_level=5,max_iter=20))
    out=capsys.readouterr()
    assert outcome.reason is None and outcome.error is not None
    assert 'fixture unexpected calculation failure' in out.err
    assert 'EXIT: Automatic stop:' not in out.out


def test_split_footer_translation_does_not_mask_unexplained_stop():
    class Callback:
        reason=None
    target=io.StringIO();cb=Callback();stream=SolverLog(target,cb)
    stream.write('EXIT: Stopping optimization at current point as requested by user.\n')
    assert 'requested by user' in target.getvalue()
    cb.reason='source_clearance_refinement'
    stream.write('EXIT: Stopping optimization ');stream.flush()
    stream.write('at current point as requested by user.\nother diagnostic')
    stream.finish()
    assert 'EXIT: Automatic stop: new 1 ms wall checks found' in target.getvalue()
    assert target.getvalue().endswith('other diagnostic')
