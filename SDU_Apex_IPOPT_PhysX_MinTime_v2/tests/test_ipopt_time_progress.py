from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ipopt_time_progress import (TimeStageProgress, choose_sampled_incumbent,
                                 primal_assignments, prefer_terminal_restart)


def payload(time_s=5., violation=.0001):
    return dict(sample_s=np.arange(3.), state_x=np.zeros((16, 3)),
                steering_rate_radps=np.zeros(3), throttle_norm=np.full(3, .1),
                phase_lengths_m=np.ones(1), decision_x=np.arange(5.),
                predicted_lap_time_s=time_s, max_constraint_violation=violation)


def test_atomic_latest_can_be_infeasible_but_best_stays_feasible(tmp_path):
    now = [0.]
    current = [payload()]
    observer = TimeStageProgress(tmp_path, 4, .001, lambda: current[0],
                                  stride=2, interval_s=10, clock=lambda: now[0])
    observer.observe(0)
    current[0] = payload(4.9, .0002)
    now[0] = 11.
    observer.observe(2)
    current[0] = payload(4.5, .02)
    observer.observe(3)  # Unsampled iteration must not touch disk or best.
    assert observer.sample_count == 2
    observer.finish()
    with np.load(tmp_path/'factor_4_latest_iterate.npz', allow_pickle=False) as d:
        assert float(d['predicted_lap_time_s']) == 4.5
        assert not bool(d['sampled_source_feasible'])
    with np.load(tmp_path/'factor_4_best_feasible_iterate.npz', allow_pickle=False) as d:
        assert float(d['predicted_lap_time_s']) == 4.9
        assert bool(d['sampled_source_feasible'])
        np.testing.assert_array_equal(d['decision_x'], np.arange(5.))
    assert not list(tmp_path.glob('*.tmp.npz'))
    assert choose_sampled_incumbent(observer.best, 4.5, .02, .001)
    assert not choose_sampled_incumbent(observer.best, 4.8, .0001, .001)


def test_checkpoint_failure_is_visible_and_does_not_corrupt_incumbent(tmp_path):
    current = [payload()]
    observer = TimeStageProgress(tmp_path, 16, .001, lambda: current[0], stride=1)
    observer.observe(0)
    current[0] = payload(4.9, np.nan)
    observer.finish()
    assert observer.errors == 1
    assert float(observer.best['predicted_lap_time_s']) == 5.
    assert not choose_sampled_incumbent(payload(4., .01), 5., .1, .001)
    assert not choose_sampled_incumbent(None, 5., .1, .001)


def test_disabled_monitor_never_requests_snapshot():
    def forbidden():
        raise AssertionError('disabled monitor must not evaluate the graph')
    observer = TimeStageProgress(None, 4, .001, forbidden)
    observer.observe(0)
    observer.finish()
    assert observer.sample_count == 0


def test_sampled_primal_restores_each_symbol_without_mutating_initial_guess():
    import casadi as ca
    opti = ca.Opti()
    x, u, parameter = opti.variable(2, 2), opti.variable(2), opti.parameter()
    opti.minimize(ca.sumsqr(x)+ca.sumsqr(u))
    opti.subject_to(ca.vec(x) >= parameter)
    opti.set_value(parameter, 2.)
    opti.set_initial(x, -1.)
    assignments = primal_assignments(opti, np.arange(1., 7.))
    np.testing.assert_array_equal(opti.value(x, assignments), [[1., 3.], [2., 4.]])
    np.testing.assert_array_equal(np.asarray(opti.value(u, assignments)).reshape(-1),
                                  [5., 6.])
    assert opti.value(parameter, assignments) == 2.
    np.testing.assert_array_equal(opti.value(x, opti.initial()), -np.ones((2, 2)))


def test_near_feasible_search_point_is_not_confused_with_certified_output():
    assert prefer_terminal_restart(True, 4.90, .003, 4.94, .005)
    assert not prefer_terminal_restart(True, 4.90, .02, 4.94, .005)
    assert not prefer_terminal_restart(True, 4.95, .003, 4.94, .005)
    assert not prefer_terminal_restart(False, 4.90, .003, 4.94, .005)
