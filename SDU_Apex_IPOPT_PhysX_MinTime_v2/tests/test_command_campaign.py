"""Search orchestration fixtures, not synthetic claims about vehicle physics."""
import json
from pathlib import Path

import numpy as np
import pytest

from command_campaign import phase_seed, run_campaign, trial_counts, budget_increase_only, search_starts
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


def test_padded_margin_reaches_baseline_predictor_shortening_and_certification(setup):
    margin = .12
    solves, predictions, certifications = [], [], []

    def solve(**kwargs):
        solves.append(kwargs)
        return solver_report(kwargs, passed=len(solves) > 1, residual=.1)

    def predict(**kwargs):
        predictions.append(kwargs)
        return dict(status='no_time_proposal')

    def certify(repo, seed, map_path, destination, margin=.05):
        certifications.append(margin)
        return checked(repo, seed, map_path, destination)

    result = run_campaign(**setup, margin=margin, solve=solve, predict=predict,
        certify=certify, hybrid_time=True)

    assert result['status'] == 'target_reached'
    assert result['contract']['required_margin_m'] == margin
    assert len(solves) >= 2 and solves[0]['shorten_packet']
    assert all(call['margin'] == margin for call in solves)
    assert predictions and predictions[0]['margin'] == margin
    assert certifications == [margin, margin]  # baseline and independent candidate gate


def test_legacy_campaign_contract_resumes_at_default_margin(setup):
    result = run_campaign(**setup, solve=lambda **kw: solver_report(kw, passed=True),
                          certify=checked)
    state_path = setup['output']/'campaign.json'
    state = json.loads(state_path.read_text())
    del state['contract']['required_margin_m']
    state_path.write_text(json.dumps(state))

    resumed = run_campaign(**setup, resume=True,
        solve=lambda **kw: solver_report(kw, passed=True), certify=checked)

    assert resumed['status'] == result['status']
    assert resumed['contract']['required_margin_m'] == .05


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


def test_larger_budgets_reopen_stagnated_primals_after_plateau(setup):
    def limited(**kw):
        report = solver_report(kw, passed=False, residual=.001)
        report.update(termination_reason='feasibility_stagnation',
                      ipopt_status='User_Requested_Stop')
        (kw['output']/'report.json').write_text(json.dumps(report))
        fixture_witness(kw['output']/'best_iterate.npz', kw['count'])
        return report
    state = run_campaign(**setup, solve=limited, certify=checked)
    assert state['status'] == 'plateau'
    previous_count = len(state['attempts'])
    calls = []
    def improved(**kw):
        calls.append(kw)
        return solver_report(kw, passed=True)
    increased = dict(setup, iterations=200, stagnation_iterations=100)
    state = run_campaign(**increased, solve=improved, certify=checked, resume=True)
    assert state['status'] == 'target_reached'
    assert len(state['budget_limited_history']) == previous_count
    assert calls[0]['command_seed'].name == 'best_iterate.npz'
    assert calls[0]['iterations'] == 200 and calls[0]['stagnation_iterations'] == 100
    assert len(state['budget_updates']) == 1


def test_budget_increase_cannot_hide_physics_changes_or_lower_patience():
    previous = dict(source_sha256='same', iterations=300)
    assert budget_increase_only(previous, dict(previous, iterations=1500, stagnation_iterations=250))
    assert not budget_increase_only(previous, dict(previous, iterations=150))
    assert not budget_increase_only(previous, dict(previous, stagnation_iterations=20))
    assert not budget_increase_only(previous, dict(previous, source_sha256='changed', iterations=1500))


def test_active_primal_is_not_lost_when_larger_budgets_reopen_earlier_phases(setup):
    settings = dict(setup, include_packet_starts=False)
    calls = []
    def interrupted(**kw):
        calls.append(kw)
        if len(calls) == 1:
            report = solver_report(kw, passed=False, residual=.01)
            report.update(termination_reason='feasibility_stagnation')
            (kw['output']/'report.json').write_text(json.dumps(report))
            return report
        kw['output'].mkdir()
        fixture_witness(kw['output']/'best_iterate.npz', kw['count'])
        raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        run_campaign(**settings, solve=interrupted, certify=checked)
    resumed = []
    def finish(**kw):
        resumed.append(kw)
        return solver_report(kw, passed=True)
    run_campaign(**dict(settings, iterations=200, stagnation_iterations=100),
                 solve=finish, certify=checked, resume=True)
    assert resumed[0]['command_seed'] == calls[1]['output']/'best_iterate.npz'


def predictor_report(kwargs, *, proposal=True):
    out = kwargs['output']
    out.mkdir(parents=True)
    seed = out/'proposed_seed.npz'
    fixture_witness(seed, 39)
    report = dict(status='uncertified_time_proposal' if proposal else 'no_time_proposal',
        source_certified=False, predicted_lap_s=.970, trial_lap_s=.975,
        proposed_intervals=39, proposed_seed=str(seed))
    (out/'report.json').write_text(json.dumps(report))
    return report


def test_hybrid_proposal_must_be_repaired_and_independently_certified(setup):
    calls, predictions = [], []
    def solve(**kw):
        calls.append(kw)
        return solver_report(kw, passed=kw['command_seed'].name == 'proposed_seed.npz', residual=.1)
    def predict(**kw):
        predictions.append(kw)
        return predictor_report(kw)
    state = run_campaign(**setup, solve=solve, certify=checked, hybrid_time=True, predict=predict)
    assert state['status'] == 'target_reached'
    assert len(predictions) == 1 and len(calls) == 2
    assert calls[1]['command_seed'] == predictions[0]['output']/'proposed_seed.npz'
    assert 'time-predictor' in state['attempts'][-1]['key']
    assert Path(state['best_witness']).parent.name == 'certified'


def test_search_tries_smaller_duration_reductions_before_repeating_hard_phases():
    counts = trial_counts(326, .1, 30)
    starts = search_starts(counts, [0, 81, 163, 244], 326, True)
    assert [c for c, phase, _ in starts if phase == 0] == [294, 310, 318, 322, 324, 325, 325]
    assert starts[5:7] == [(325, 0, 'drop-packet'), (325, 0, 'resample')]
    assert starts[7] == (294, 81, 'resample')
    assert len(set(starts)) == len(starts) == 28


def test_hybrid_no_proposal_continues_other_phases_without_repeating_predictor(setup):
    predictions = []
    def predict(**kw):
        predictions.append(kw)
        return predictor_report(kw, proposal=False)
    state = run_campaign(**setup, solve=lambda **kw: solver_report(kw, passed=False, residual=.1),
        certify=checked, hybrid_time=True, predict=predict)
    assert state['status'] == 'plateau' and state['best_lap_s'] == 1.
    assert len(predictions) == 1 and len(state['attempts']) == 4


def test_interrupted_time_predictor_resumes_checkpoint_before_exact_repair(setup):
    predictions = []
    def interrupt(**kw):
        predictions.append(kw)
        kw['output'].mkdir(parents=True)
        (kw['output']/'best_predictor.npz').write_bytes(b'checkpoint contract handled by predictor')
        raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        run_campaign(**setup, solve=lambda **kw: solver_report(kw, passed=False, residual=.1),
                     certify=checked, hybrid_time=True, predict=interrupt)
    def resume(**kw):
        assert kw['resume_checkpoint'] == predictions[0]['output']/'best_predictor.npz'
        assert kw['output'] != predictions[0]['output']
        return predictor_report(kw)
    state = run_campaign(**setup, solve=lambda **kw: solver_report(kw, passed=True),
                         certify=checked, hybrid_time=True, predict=resume, resume=True)
    assert state['status'] == 'target_reached'
    assert state['attempts'][-1]['key'].endswith('time-predictor')


def test_plateau_tries_both_packet_and_time_objectives_from_distinct_elites(
        setup, tmp_path):
    """A same-time elite archive is a branching search, not a new fastest lap."""
    seeds, solved, time_predictions, prediction_calls, callbacks = [], [], [], [], []
    certifications = []

    def certify_with_margin(repo, seed, map_path, destination, margin=.05):
        certifications.append(margin)
        return checked(repo, seed, map_path, destination)
    for index in (1, 2):
        folder=tmp_path/f'certified_elite_{index}'
        folder.mkdir()
        fixture_witness(folder/'witness.npz',40)
        with np.load(folder/'witness.npz') as data:
            states=np.asarray(data['state']).copy()
            controls=np.asarray(data['controls']).copy()
        states[0,:]=index*.02
        save_command_checkpoint(folder/'witness.npz',states,controls,'fixture',0.,0)
        (folder/'report.json').write_text(json.dumps(dict(
            status='source_witness_passed',trial_lap_s=1.,
            rollout=dict(minimum_1ms_footprint_slack_m=.0001*index))))
        seeds.append(folder/'witness.npz')

    def elite(**kwargs):
        assert kwargs['margin_steps'] and kwargs['geometry_seeds']
        assert kwargs['max_elites']>=2
        return dict(accepted=[
            dict(witness=str(path),sha256=f'elite{i}',lap_s=1.,
                 clearance_slack_m=.0001*i)
            for i,path in enumerate(seeds,1)])

    def solve(**kw):
        solved.append(kw)
        # The initial campaign must plateau. Each elite must first fail its
        # first shorter fixed-grid proposal, triggering its time predictor.
        with np.load(kw['command_seed'], allow_pickle=False) as data:
            marker=float(data['state'][0,0])
        refined = abs(marker)>.005
        branch_dir=str(kw['output'])
        in_child='elite_campaigns' in branch_dir
        preceding=[x for x in solved[:-1] if str(x['output']).split('/elite_campaigns/')[0] ==
                   branch_dir.split('/elite_campaigns/')[0]
                   and in_child and '/elite_campaigns/' in str(x['output'])
                   and str(x['output']).split('/elite_campaigns/')[1].split('/')[0] ==
                   branch_dir.split('/elite_campaigns/')[1].split('/')[0]]
        allowed=in_child and refined and len(preceding)>0
        return solver_report(kw,passed=allowed,residual=.08)

    def predict(**kw):
        time_predictions.append(str(kw['command_seed']))
        prediction_calls.append(kw)
        kw['output'].mkdir(parents=True,exist_ok=True)
        result=dict(status='no_time_proposal',trial_lap_s=None,
                    predicted_lap_s=1.,proposed_intervals=None)
        (kw['output']/'report.json').write_text(json.dumps(result))
        return result

    settings=dict(setup,target_seconds=.975,phases=1,hybrid_time=True,margin=.12,
                  elite_refinements=2,elite_polish=elite,predict=predict,
                  on_improvement=lambda path,lap:callbacks.append((path,lap)))
    state=run_campaign(**settings,solve=solve,certify=certify_with_margin)
    assert state['status']=='target_reached'
    assert state['best_lap_s']==pytest.approx(.975)
    assert len(state['elite_archives'])==1
    # Both elites get their own source-grid and time-objective attempts.
    assert any('/elite_campaigns/elite1/' in x for x in time_predictions)
    assert any('/elite_campaigns/elite2/' in x for x in time_predictions)
    assert all(call['margin'] == .12 for call in solved)
    assert prediction_calls and all(call['margin'] == .12 for call in prediction_calls)
    assert certifications and set(certifications) == {.12}
    child_folders=[x for x in (settings['output']/'elite_campaigns').iterdir() if x.is_dir()]
    assert len(child_folders)==2
    assert all((folder/'campaign.json').is_file() for folder in child_folders)


def test_elite_first_runs_before_any_shorter_duration_trial(setup):
    order=[]
    def elite(**kwargs):
        order.append('equal_duration_archive')
        assert kwargs['geometry_seeds']
        return {'accepted':[]}
    def solve(**kw):
        order.append('shorter_duration')
        return solver_report(kw,passed=True)
    result=run_campaign(**setup,solve=solve,certify=checked,
                        elite_first=True,elite_refinements=2,elite_polish=elite)
    assert result['best_lap_s']==pytest.approx(.975)
    assert result['elite_first_done'] is True
    assert order[:2]==['equal_duration_archive','shorter_duration']



def test_elite_uses_full_iteration_and_stagnation_budget(setup):
    elite_calls=[]
    def elite(**kw):
        elite_calls.append(kw)
        return dict(accepted=[])
    def failed(**kw):
        return solver_report(kw,passed=False,residual=.2)
    settings=dict(setup,iterations=1200,stagnation_iterations=300,
                  elite_first=True,elite_refinements=2,
                  elite_polish=elite)
    result=run_campaign(**settings,solve=failed,certify=checked)
    assert result['status']=='plateau'
    assert elite_calls
    assert elite_calls[0]['iterations']==1200
    assert elite_calls[0]['stagnation_iterations']==300



def test_hybrid_time_optimizer_uses_full_1200_300_budget(setup):
    predicted=[]
    def predict(**kw):
        predicted.append(kw)
        return predictor_report(kw,proposal=False)
    settings=dict(setup,iterations=1200,stagnation_iterations=300)
    state=run_campaign(**settings,
        solve=lambda **kw: solver_report(kw,passed=False,residual=.1),
        certify=checked,hybrid_time=True,predict=predict)
    assert state['status']=='plateau'
    assert len(predicted)==1
    assert predicted[0]['iterations']==1200
    assert predicted[0]['stagnation_iterations']==300


def test_sector_time_predictor_is_separate_versioned_search(setup):
    received=[]
    def predict(**kw):
        received.append(kw)
        return predictor_report(kw,proposal=False)
    state=run_campaign(**setup,hybrid_time=True,time_sectors=12,
        predict=predict,solve=lambda **kw:solver_report(kw,passed=False,residual=.1),
        certify=checked)
    assert state['status']=='plateau'
    assert state['contract']['time_sectors']==12
    assert state['contract']['time_predictor']=='source-24-25ms-sector-chord/v2'
    assert received and received[0]['time_sectors']==12


def test_picometre_near_miss_is_revalidated_not_silently_accepted(setup,monkeypatch):
    import rigid_pose_polish
    repairs=[]
    def fake_recovery(repo,seed,map_path,output,*,margin,certify):
        repairs.append(seed)
        output.mkdir(parents=True)
        archive=output/'certified'
        verified=checked(repo,seed,map_path,archive)
        verified['rollout']=dict(minimum_1ms_footprint_slack_m=1e-11)
        (archive/'report.json').write_text(json.dumps(verified))
        return dict(status='source_witness_passed',
            witness=str(archive/'witness.npz'),
            certificate=str(archive/'report.json'),
            numerical_only=True,shifted_slack_m=1e-11)
    monkeypatch.setattr(rigid_pose_polish,'recover_numerical_near_miss',fake_recovery)
    def near_miss(**kwargs):
        report=solver_report(kwargs,passed=False,residual=1e-9)
        report['rollout']['minimum_1ms_footprint_slack_m']=-4e-12
        (kwargs['output']/'report.json').write_text(json.dumps(report))
        return report
    calls=[]
    result=run_campaign(**setup,solve=near_miss,certify=checked,
        on_improvement=lambda path,lap:calls.append((path,lap)))
    assert result['status']=='target_reached'
    assert result['best_lap_s']==pytest.approx(.975)
    assert repairs and calls
    assert len(calls)==2  # baseline and newly independently certified witness
    assert calls[-1][1]==pytest.approx(.975)
    assert Path(result['best_witness']).exists()
    assert len(result['attempts'])==1
