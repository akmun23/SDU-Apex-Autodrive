"""Search orchestration fixtures, not synthetic claims about vehicle physics."""
import json
from pathlib import Path

import numpy as np
import pytest

from command_campaign import phase_seed, run_campaign, trial_counts
from command_ocp import load_command_seed, save_command_checkpoint


def fixture_witness(path, count):
    x = np.zeros((21, count+1))
    x[0] = np.linspace(0., 1., count+1)
    x[2] = np.linspace(0., 2*np.pi, count+1)
    x[3] = 5.
    x[0, -1] = 0.
    u = np.vstack([np.linspace(-.2, .2, count), np.linspace(.1, .2, count)])
    save_command_checkpoint(path, x, u, 'fixture', 0., 0)


@pytest.fixture
def setup(tmp_path):
    seed, grid = tmp_path/'seed.npz', tmp_path/'map.pgm'
    fixture_witness(seed, 40)
    grid.write_bytes(b'fixture map; never passed to a physics evaluator')
    return dict(repo=tmp_path, seed=seed, map_path=grid, output=tmp_path/'campaign',
        target_seconds=.975, phases=2, workers=1, iterations=100,
        stagnation_iterations=20, digest='fixture-source')


def checked(repo, seed, map_path, destination):
    destination.mkdir()
    with np.load(seed, allow_pickle=False) as data:
        save_command_checkpoint(destination/'witness.npz', data['state'], data['controls'],
                                'fixture', 0., 0)
        count = data['controls'].shape[1]
    report = dict(status='source_witness_passed', trial_lap_s=count*.025)
    (destination/'report.json').write_text(json.dumps(report))
    return report


def solver_report(kwargs, *, passed, residual=1.e-7, cuts=()):
    destination, count = kwargs['output'], kwargs['count']
    destination.mkdir()
    fixture_witness(destination/'witness.npz', count)
    report = dict(status='source_witness_passed' if passed else 'no_validated_witness',
        trial_lap_s=count*.025, max_constraint_violation=residual,
        ipopt_status='Solve_Succeeded', rollout=dict(full_lap_xy_max_m=.001,
            violating_source_ticks=[dict(interval=i, tick=t) for i, t in cuts]))
    (destination/'report.json').write_text(json.dumps(report))
    return report


def test_source_wall_cuts_feed_exact_primal_back_to_ipopt_then_keep_faster(setup):
    calls = []
    def solve(**kwargs):
        calls.append(kwargs)
        return solver_report(kwargs, passed=len(calls)>1, cuts=((1, 7),))
    state = run_campaign(**setup, solve=solve, certify=checked)
    assert state['status'] == 'target_reached'
    assert state['best_lap_s'] == pytest.approx(.975)
    assert len(calls) == 2
    assert calls[1]['clearance_cuts'] == [(1, 7)]
    assert calls[1]['command_seed'] == calls[0]['output']/'witness.npz'
    assert all(c['cpu_seconds'] == 1.e20 for c in calls)
    assert not state['runtime_promoted'] and not state['unity_validated']


def test_failed_multistarts_plateau_without_repeating_or_losing_baseline(setup):
    calls = []
    def solve(**kwargs):
        calls.append(kwargs)
        return solver_report(kwargs, passed=False, residual=.1)
    state = run_campaign(**setup, solve=solve, certify=checked)
    assert state['status'] == 'plateau' and state['best_lap_s'] == 1.
    assert len(calls) == 4
    resumed = run_campaign(**setup, solve=solve, certify=checked, resume=True)
    assert resumed['attempts'] == state['attempts'] and len(calls) == 4


def test_interruption_resumes_saved_primal_not_original_bad_seed(setup):
    first = []
    def interrupt(**kwargs):
        kwargs['output'].mkdir()
        fixture_witness(kwargs['output']/'best_iterate.npz', kwargs['count'])
        first.append(kwargs)
        raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        run_campaign(**setup, solve=interrupt, certify=checked)
    state = json.loads((setup['output']/'campaign.json').read_text())
    assert state['status'] == 'interrupted'
    resumed = []
    def solve(**kwargs):
        resumed.append(kwargs)
        return solver_report(kwargs, passed=True)
    state = run_campaign(**setup, solve=solve, certify=checked, resume=True)
    assert state['status'] == 'target_reached'
    assert resumed[0]['command_seed'] == first[0]['output']/'best_iterate.npz'
    assert resumed[0]['output'] != first[0]['output']


def test_second_independent_source_gate_cannot_be_bypassed(setup):
    calls = []
    def certify(repo, seed, map_path, out):
        report = checked(repo, seed, map_path, out)
        calls.append(out)
        if len(calls)>1: report['status'] = 'no_validated_witness'
        return report
    state = run_campaign(**setup,
        solve=lambda **kw: solver_report(kw, passed=True), certify=certify)
    assert state['status'] == 'plateau' and state['best_lap_s'] == 1.
    assert all(not a['certified'] for a in state['attempts'])


def test_cyclic_phase_preserves_packets_hidden_states_and_heading_winding(tmp_path):
    path = tmp_path/'seed.npz'
    fixture_witness(path, 40)
    x, u, _, _ = load_command_seed(path, 40)
    shifted = phase_seed(path, 13, tmp_path/'phase.npz')
    a, b, winding, lap = load_command_seed(shifted, 40)
    np.testing.assert_array_equal(b, np.roll(u, -13, axis=1))
    np.testing.assert_array_equal(a[3:, :27], x[3:, 13:40])
    np.testing.assert_array_equal(a[:, 0], x[:, 13])
    assert a[2, -1]-a[2, 0] == pytest.approx(2*np.pi)
    assert winding == 1 and lap == 1.


def test_existing_campaign_is_never_overwritten_and_source_changes_fail_closed(setup):
    solve = lambda **kw: solver_report(kw, passed=True)
    run_campaign(**setup, solve=solve, certify=checked)
    with pytest.raises(FileExistsError):
        run_campaign(**setup, solve=solve, certify=checked)
    changed = dict(setup, digest='different-source')
    with pytest.raises(ValueError, match='source/map/settings changed'):
        run_campaign(**changed, solve=solve, certify=checked, resume=True)


def test_still_improving_iteration_limit_can_continue_from_its_own_primal(setup):
    calls = []
    def solve(**kwargs):
        calls.append(kwargs)
        result = solver_report(kwargs, passed=len(calls)>1, residual=.005)
        if len(calls) == 1:
            result.update(ipopt_status='Maximum_Iterations_Exceeded', initial_residual=.1,
                          last_significant_iteration=95)
        return result
    state = run_campaign(**setup, solve=solve, certify=checked)
    assert state['status'] == 'target_reached'
    assert len(calls) == 2
    assert calls[1]['command_seed'] == calls[0]['output']/'witness.npz'


def test_close_improving_dynamic_defect_is_not_discarded_when_no_wall_cut_exists(setup):
    calls = []
    def solve(**kwargs):
        calls.append(kwargs)
        result = solver_report(kwargs, passed=len(calls)>1, residual=.0002)
        if len(calls) == 1:
            result.update(ipopt_status='User_Requested_Stop', initial_residual=.01)
        return result
    state = run_campaign(**setup, solve=solve, certify=checked)
    assert state['status'] == 'target_reached' and len(calls) == 2


def test_incomplete_bootstrap_certification_is_recoverable(setup):
    def interrupted(repo, seed, map_path, out):
        out.mkdir()
        raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        run_campaign(**setup, certify=interrupted)
    state = run_campaign(**setup, resume=True, certify=checked,
        solve=lambda **kw: solver_report(kw, passed=True))
    assert state['status'] == 'target_reached'
    assert (setup['output']/'baseline_001/report.json').exists()


def test_coarse_to_fine_duration_trials_always_reach_packet_resolution():
    assert trial_counts(100, .1, 30) == [90, 95, 98, 99]
    assert trial_counts(40, .1, 39) == [39]
    assert trial_counts(30, .1, 30) == []
    with pytest.raises(ValueError):
        trial_counts(40, float('inf'), 30)


def test_open_search_has_no_track_specific_target_and_publishes_each_gain(setup):
    settings = dict(setup, target_seconds=None, coarse_fraction=.1, phases=1)
    calls, published = [], []
    def solve(**kwargs):
        calls.append(kwargs)
        return solver_report(kwargs, passed=kwargs['count'] == 36, residual=.1)
    state = run_campaign(**settings, solve=solve, certify=checked,
        on_improvement=lambda path, lap: published.append(lap))
    assert calls[0]['count'] == 36
    assert state['status'] == 'plateau'
    assert state['contract']['target_intervals'] is None
    assert state['best_lap_s'] == pytest.approx(.9)
    assert published == pytest.approx([1., .9])
    assert calls[-1]['count'] == 35


def test_resuming_rejects_changed_metric_metadata_even_with_identical_pixels(setup):
    metadata = setup['map_path'].with_suffix('.yaml')
    metadata.write_text('resolution: .05\n')
    run_campaign(**setup, solve=lambda **kw: solver_report(kw, passed=True), certify=checked)
    metadata.write_text('resolution: .025\n')
    with pytest.raises(ValueError, match='map/settings changed'):
        run_campaign(**setup, resume=True)
