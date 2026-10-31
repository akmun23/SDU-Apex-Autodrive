"""Observer fixtures describe file formats, not measured vehicle performance."""
import hashlib
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image

from tools.optimizer_dashboard import (Dashboard, LogObserver, Workspace,
                                       handler_for, json_bytes, mpc_failure_diagnostic,
                                       process_info, candidate_acceptance, describe_solve)


class DashboardTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.folder = self.root/'starts/clockwise/campaign/from100_90_phase000_resample/solve_000'
        self.folder.mkdir(parents=True)
        self.campaign_path = self.root/'starts/clockwise/campaign/campaign.json'
        self.context = dict(key='from100_90_phase000_resample', output=str(self.folder), refinement=0)
        self.write('optimizer.json', dict(status='running', pid=0,
            stage='adaptive_minimum_time_search', active_start='clockwise',
            contract=dict(iterations=1500, workers=8, stagnation_iterations=250, direction='clockwise'),
            geometry_starts=[]))
        self.write(self.campaign_path, dict(active=self.context, attempts=[]))
        self.log = self.root/'optimizer.log'
        self.log.write_text('')
        self.dashboard = Dashboard(self.root)

    def write(self, path, data):
        path = Path(path)
        if not path.is_absolute():
            path = self.root/path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))

    def start_log(self):
        self.log.write_text(json.dumps(dict(event='solve', **self.context))+'\n'+
            json.dumps(dict(stage='production-command witness', trial_lap_s=2.25,
                            initial_residual=.5, max_iterations=1500))+'\n')

    def test_snapshot_decodes_real_npz_metadata_without_claiming_running(self):
        self.start_log()
        with self.log.open('a') as stream:
            stream.write(json.dumps(dict(iteration=5, trial_lap_s=2.25,
                                         max_constraint_violation=.12))+'\n')
            stream.write('  6  1.234567e-04 1.0e-05 2.0e-06 -10 0.5 - 1 1h 1\n')
        np.savez_compressed(self.folder/'best_iterate.npz', iteration=5,
                            max_constraint_violation=.12, state=np.zeros((21, 91)))
        result = self.dashboard.snapshot()
        self.assertEqual(result['iteration'], 6)
        self.assertEqual(result['checkpoint']['iteration'], 5)
        self.assertEqual(result['checkpoint']['residual'], .12)
        self.assertEqual(result['active']['trial_lap_s'], 2.25)
        self.assertEqual(result['status'], 'not_running')
        self.assertEqual(result['warnings'], [])
        self.assertEqual(result['residuals'][-1]['residual'], .12)
        self.assertEqual(result['ipopt']['primal_infeasibility'], 1e-5)

    def test_never_mix_ipopt_scaled_primal_with_actual_constraint_residual(self):
        reader = LogObserver(self.log)
        reader.parse('{"iteration":5,"max_constraint_violation":0.4}')
        reader.parse('  5  9.5e-05 1.0e-08 1.0e-12 -10 0.1 - 1 1h 1')
        self.assertEqual(reader.points[-1]['residual'], .4)
        self.assertEqual(reader.ipopt['primal_infeasibility'], 1e-8)

    def test_feasible_iteration_zero_is_explained_as_quality_seed_not_finished_search(self):
        self.start_log()
        np.savez_compressed(self.folder/'best_iterate.npz', iteration=0,
                            max_constraint_violation=2.19e-10)
        with self.log.open('a') as stream:
            for item in (
                dict(stage='production-command witness', output=str(self.folder), objective_mode='clearance',
                     trial_lap_s=7.05, max_iterations=1200),
                dict(iteration=0, max_constraint_violation=2.19e-10, objective=.02),
                dict(event='early_source_checkpoint_gate', iteration=0,
                     periodic_scaled_defect=8.88e-16, source_xy_error_m=1.e-9,
                     clearance_slack_m=0.),
                dict(iteration=155, max_constraint_violation=.1, objective=-.2),
            ):
                stream.write(json.dumps(item)+'\n')
            stream.write(' 158 -9.9e+09 1e-1 1e6 -1 1 - 1 1h 1\n')
        snap=self.dashboard.snapshot()
        task=snap['search_progress']['task']
        self.assertEqual(task['label'], 'Increase wall clearance')
        self.assertTrue(task['fixed_duration'])
        self.assertIn('Fixed lap time: 7.050 s', task['goal'])
        self.assertIn('Passing feasibility does not finish', task['why_continuing'])
        self.assertTrue(task['feasible_seed_checked'])
        self.assertIn('No improved feasible checkpoint', task['progress'])
        self.assertIn('Live iteration: 158', task['evidence'])
        self.assertIn('Saved checkpoint: 0', task['evidence'])
        self.assertIn('Latest source replay: 0', task['evidence'])
        self.assertEqual(task['objective']['initial'], .02)
        self.assertEqual(task['objective']['saved'], .02)
        self.assertEqual(task['objective']['latest'], -.2)
        self.assertEqual(task['objective']['latest_iteration'], 155)
        self.assertEqual(task['objective']['latest_residual'], .1)

    def test_objective_context_resets_and_missing_headers_do_not_borrow_old_goal(self):
        self.start_log()
        with self.log.open('a') as stream:
            stream.write(json.dumps(dict(stage='production-command witness',
                                         objective_mode='clearance'))+'\n')
            stream.write(json.dumps(dict(event='solve', output=str(self.folder/'next')))+'\n')
        self.dashboard.snapshot()
        task=self.dashboard.snapshot()['search_progress']['task']
        self.assertIsNone(task['mode'])
        self.assertEqual(task['label'], 'Preparing the next solve')
        self.assertIsNone(task['objective']['latest'])

    def test_solver_header_alone_identifies_refinement_and_clears_prior_evidence(self):
        self.start_log()
        child=self.folder/'cut_01'
        self.write(self.campaign_path,dict(active=dict(output=str(child),trial_lap_s=2.25)))
        with self.log.open('a') as stream:
            for item in (
                dict(event='early_source_checkpoint_gate',iteration=90,clearance_slack_m=-.01),
                dict(stage='production-command witness',output=str(child),
                     objective_mode='repair',trial_lap_s=2.25),
                dict(iteration=10,max_constraint_violation=.0001,objective=.3),
            ):
                stream.write(json.dumps(item)+'\n')
        snap=self.dashboard.snapshot()
        self.assertEqual(snap['iteration'],10)
        self.assertEqual(snap['solver']['objective_mode'],'repair')
        self.assertEqual(snap['active']['identity_source'],'explicit output')
        self.assertIsNone(snap['search_progress']['acceptance']['replay_iteration'])
        self.assertEqual(snap['search_progress']['task']['objective']['latest'],.3)

    def test_header_path_is_authoritative_after_bounded_log_tail_or_new_observer(self):
        self.log.write_text(json.dumps(dict(stage='production-command witness',
            output=str(self.folder),objective_mode='smooth',trial_lap_s=2.25))+'\n'+
            json.dumps(dict(iteration=45,max_constraint_violation=1.e-6))+'\n')
        snap=self.dashboard.snapshot()
        self.assertEqual(snap['iteration'],45)
        self.assertEqual(snap['search_progress']['task']['mode'],'smooth')
        self.assertEqual(snap['active']['identity_source'],'explicit output')

    def refinement_handoff(self):
        """Reproduce a worker already running before self-identifying headers."""
        base=self.root/'starts/clockwise/campaign/elite/polish_02'
        child=base.with_name('polish_02_cut_01')
        child.mkdir(parents=True)
        active=dict(output=str(child),index=2,strategy='adaptive',trial_lap_s=7.05,
                    margin=.05,source_clearance_cuts=[[74,13],[75,3]])
        report=dict(source_plant_sha256='fixture',trial_lap_s=7.05,
                    required_margin_m=.05,requested_source_clearance_cuts=[[74,13],[75,3]],
                    source_clearance_cuts=[])
        self.write(base/'report.json',report)
        self.write(self.campaign_path,dict(active=active))
        with self.log.open('a') as stream:
            for item in (
                dict(event='solve',output=str(base)),
                dict(stage='production-command witness',objective_mode='repair',trial_lap_s=7.05),
                dict(event='early_source_checkpoint_gate',iteration=140,clearance_slack_m=-.001),
                dict(stage='production-command witness',objective_mode='repair',trial_lap_s=7.05,
                     source_sha256='fixture',source_clearance_cuts=2),
                dict(iteration=610,max_constraint_violation=2.e-12,objective=.168),
                dict(event='early_source_checkpoint_gate',iteration=610,clearance_slack_m=1.e-10,
                     periodic_scaled_defect=0.,source_xy_error_m=1.e-10),
            ):
                stream.write(json.dumps(item)+'\n')
        return base,child,active,report

    def test_inflight_cut_repair_recovers_from_exact_manifest_and_report_lineage(self):
        self.refinement_handoff()
        snap=self.dashboard.snapshot()
        self.assertEqual(snap['active']['identity_source'],'verified refinement handoff')
        self.assertEqual(snap['iteration'],610)
        self.assertEqual(snap['solver']['source_clearance_cuts'],2)
        self.assertEqual(snap['search_progress']['acceptance']['replay_iteration'],610)
        self.assertEqual(snap['search_progress']['task']['label'],
                         'Repair an alternative corner trajectory')
        self.assertTrue(snap['search_progress']['task']['fixed_duration'])
        self.assertEqual(snap['warnings'],[])

    def test_refinement_recovery_rejects_similar_time_without_exact_lineage(self):
        base,child,active,report=self.refinement_handoff()
        for changed in (
            dict(source_plant_sha256='wrong'),dict(trial_lap_s=7.075),
            dict(required_margin_m=.06),dict(requested_source_clearance_cuts=[[74,12],[75,3]]),
        ):
            self.write(base/'report.json',dict(report,**changed))
            self.dashboard.last_poll=0
            snap=self.dashboard.snapshot()
            self.assertIsNone(snap['iteration'])
            self.assertIsNone(snap['search_progress']['acceptance']['replay_iteration'])
            self.assertTrue(any('identity does not match' in w for w in snap['warnings']))
        self.write(base/'report.json',report)
        active['output']=str(child.with_name('polish_03_cut_01'))
        self.write(self.campaign_path,dict(active=active))
        self.dashboard.last_poll=0
        self.assertIsNone(self.dashboard.snapshot()['iteration'])

    def test_proposal_restore_handoff_is_identified_only_by_its_named_proposal(self):
        base,_,active,report=self.refinement_handoff()
        report.update(objective_mode='clearance')
        self.write(base/'report.json',report)
        active.update(output=str(base.with_name('polish_02_restore')),
                      strategy='proposal_restoration',proposal=str(base/'best_quality_proposal.npz'),
                      source_clearance_cuts=[])
        self.write(self.campaign_path,dict(active=active))
        with self.log.open('a') as stream:
            stream.write(json.dumps(dict(stage='production-command witness',objective_mode='repair',
                source_sha256='fixture',source_clearance_cuts=0,trial_lap_s=7.05))+'\n')
            stream.write(json.dumps(dict(iteration=5,max_constraint_violation=.01))+'\n')
        snap=self.dashboard.snapshot()
        self.assertEqual(snap['active']['identity_source'],'verified proposal handoff')
        self.assertEqual(snap['iteration'],5)
        self.assertIsNone(snap['search_progress']['acceptance']['replay_iteration'])
        active['proposal']=str(base/'other.npz')
        self.write(self.campaign_path,dict(active=active))
        self.dashboard.last_poll=0
        self.assertIsNone(self.dashboard.snapshot()['iteration'])

    def test_solve_objectives_do_not_all_claim_clearance_or_fixed_duration(self):
        acceptance=candidate_acceptance(None, None, None)
        for mode, label, fixed in (
            ('smooth', 'Smooth steering and throttle', True),
            ('repair', 'Find a model-feasible trajectory', False),
            (None, 'Preparing the next solve', False),
        ):
            task=describe_solve(dict(objective_mode=mode),dict(trial_lap_s=7.05),
                                acceptance,[],None,None,None)
            self.assertEqual(task['label'],label)
            self.assertEqual(task['fixed_duration'],fixed)
            self.assertFalse(task['feasible_seed_checked'])
        predictor=describe_solve({},dict(time_predictor=True,trial_lap_s=7.),
                                 acceptance,[],None,None,0)
        self.assertEqual(predictor['mode'],'time')
        self.assertFalse(predictor['fixed_duration'])
        self.assertIn('exact-command repair',predictor['goal'])

    def test_later_or_mismatched_checkpoint_is_not_called_verified_seed(self):
        for saved, replay in ((40,40),(0,40),(40,0)):
            checkpoint=dict(iteration=saved,residual=1.e-8)
            acceptance=candidate_acceptance(checkpoint,
                dict(iteration=replay,periodic_scaled_defect=0.,
                     source_xy_error_m=0.,clearance_slack_m=.001),None)
            task=describe_solve(dict(objective_mode='clearance'),{},acceptance,
                                [],checkpoint,None,80)
            self.assertFalse(task['feasible_seed_checked'])
            self.assertNotIn('No improved feasible checkpoint',task['progress'])
        task=describe_solve(dict(objective_mode='clearance'),{},acceptance,
                            [],checkpoint,dict(status='completed'),80)
        self.assertIn('Solve finished',task['why_continuing'])
        self.assertIn('completed selected witness',task['evidence'])

    def test_dashboard_labels_verification_separately_from_optimization(self):
        html=Path(__file__).with_name('optimizer_dashboard.html').read_text()
        for text in ('Current task:', 'Verification evidence — not optimization progress',
                     'Fixed lap time · quality search', 'id="taskWhy"',
                     'id="evidenceScope"', 'not a wall gap or a certified improvement'):
            self.assertIn(text,html)
        self.assertNotIn('Now trying ${num(trial,3)} s.',html)

    def test_time_prediction_uses_its_own_residuals_and_is_never_a_certified_trial(self):
        self.start_log()
        self.write(self.campaign_path, dict(active=None, attempts=[], active_predictor=dict(
            output=str(self.folder), incumbent_sha256='fixture')))
        with self.log.open('a') as stream:
            for event in (dict(event='minimum_time_prediction', output=str(self.folder)),
                          dict(stage='minimum-time predictor setup', source_lap_s=2.25,
                               max_iterations=200),
                          dict(stage='minimum-time predictor', iteration=5,
                               predicted_lap_s=2.22, max_constraint_violation=.002)):
                stream.write(json.dumps(event)+'\n')
        result = self.dashboard.snapshot()
        self.assertTrue(result['active']['time_predictor'])
        self.assertEqual(result['active']['trial_lap_s'], 2.22)
        self.assertEqual(result['iteration'], 5)
        self.assertEqual(result['residuals'][-1]['residual'], .002)
        self.assertEqual(result['passed_attempts'], 0)

    def test_idle_between_trials_accepts_null_active(self):
        self.write(self.campaign_path, dict(active=None, attempts=[]))
        result = self.dashboard.snapshot()
        self.assertIsNone(result['active']['trial_lap_s'])

    def test_multi_incumbent_branch_reads_child_and_archive_not_stale_parent(self):
        self.start_log()
        folder=self.root/'starts/clockwise/campaign'
        alternate=folder/'elite_campaigns/abcedf1234567890'
        alternate.mkdir(parents=True)
        self.write(folder/'elite_refinement/eeff/elite.json',dict(
            accepted=[dict(clearance_slack_m=.0008,lap_s=2.5,witness='example.npz')]))
        self.write(alternate/'campaign.json',dict(
            active=dict(key='from100_90_phase000_resample',
                        output=str(alternate/'from100_90_phase000_resample/solve_000'),
                        refinement=0),
            attempts=[dict(key='alternate',certified=False)]))
        self.write(self.campaign_path,dict(active=None,attempts=[dict(key='old',certified=True)],
            elite_archives=dict(original=str(folder/'elite_refinement/eeff/elite.json')),
            elite_explored=['oldseed'],
            active_elite=dict(sha256='abcedf1234567890',output=str(alternate),lap_s=2.5)))
        result=self.dashboard.snapshot()
        self.assertEqual(result['completed_attempts'],1)
        self.assertEqual(result['attempts'][0]['key'],'alternate')
        self.assertEqual(result['elite_progress']['archived_alternatives'],1)
        self.assertEqual(result['elite_progress']['explored_basins'],1)
        self.assertEqual(result['elite_progress']['active_branch']['sha256'],'abcedf1234567890')
        self.assertIn('Multi-incumbent optimization progress',
            Path(__file__).with_name('optimizer_dashboard.html').read_text())

    def test_polishing_budget_and_attempt_context(self):
        self.start_log()
        p=self.root/'starts/clockwise/campaign'
        p.mkdir(parents=True,exist_ok=True)
        folder=p/'elite_refinement/seed1'
        self.write(folder/'elite.json',dict(
            contract=dict(extra_margin_m=[.0001,.0005,.002],
                          geometry_seeds=[[.04,1],[-.04,1]],
                          quality_objectives=['clearance','smooth'],
                          control_seeds=[[.002,1],[-.002,1]]),
            active_attempt=dict(index=3,strategy='geometry',
                                output=str(folder/'polish_03')),
            attempts=[dict(index=i,status='no_validated_witness')
                      for i in range(3)],accepted=[]))
        self.write(self.campaign_path,dict(
            active_elite_polish=dict(seed='seed',output=str(folder),
                                     source_lap_s=7.125)))
        snap=self.dashboard.snapshot()
        ep=snap['elite_progress']
        self.assertEqual(ep['polishing_total'],9)
        self.assertEqual(ep['polishing_attempts'],3)
        self.assertEqual(ep['polishing_active']['index'],3)
        html=Path(__file__).with_name('optimizer_dashboard.html').read_text()
        self.assertIn('resets for each new candidate',html)

    def test_resumed_parent_polishing_does_not_show_suspended_child_predictor(self):
        folder=self.root/'starts/clockwise/campaign'
        polish=folder/'elite/new_seed'
        output=polish/'polish_01'
        child=folder/'elite_campaigns/old_seed'
        self.write(child/'campaign.json',dict(active_predictor=dict(
            output=str(child/'time_from280/solve')),attempts=[]))
        self.write(polish/'elite.json',dict(active_attempt=dict(
            index=1,strategy='objective',output=str(output),trial_lap_s=6.975),attempts=[]))
        self.write(self.campaign_path,dict(active_elite=dict(output=str(child)),
            active_elite_polish=dict(output=str(polish)),attempts=[]))
        self.log.write_text(json.dumps(dict(stage='production-command witness',
            output=str(output),objective_mode='clearance',trial_lap_s=6.975))+'\n'+
            json.dumps(dict(iteration=10,max_constraint_violation=.01,objective=.003))+'\n')
        snap=self.dashboard.snapshot()
        self.assertEqual(snap['iteration'],10)
        self.assertFalse(snap['active']['time_predictor'])
        self.assertEqual(snap['active']['output'],str(output.relative_to(self.root)))
        self.assertEqual(snap['search_progress']['task']['label'],'Increase wall clearance')
        self.assertEqual(snap['search_progress']['nlp_threshold'],1.e-5)
        self.assertEqual(snap['warnings'],[])

    def test_feasibility_and_ipopt_raw_use_separate_graphs(self):
        html = Path(__file__).with_name('optimizer_dashboard.html').read_text()
        actual = html.split('function residualChart(){', 1)[1].split('function ipoptChart(){', 1)[0]
        raw = html.split('function ipoptChart(){', 1)[1].split('function lapChart(){', 1)[0]
        self.assertIn('id="residualChart"', html)
        self.assertIn('id="ipoptChart"', html)
        self.assertIn('search_progress?.nlp_threshold||1e-5', actual)
        self.assertNotIn('ipopt_history', actual)
        self.assertIn('ipopt_history', raw)
        self.assertNotIn('nlp_threshold', raw)
        self.assertIn('Separate logarithmic scale', html)

    def test_quality_summary_retains_slower_roomier_tradeoff_and_queue(self):
        self.write('best.json', dict(lap_s=7.05, files=dict(validation='checked.json')))
        self.write('checked.json', dict(required_margin_m=.05,
            rollout=dict(minimum_1ms_footprint_slack_m=0.)))
        self.write('portfolio.json', dict(certified_count=6, pareto=[{}, {}],
            roomiest=dict(lap_s=7.075, required_margin_m=.05,
                          clearance_slack_m=.0298)))
        self.write(self.campaign_path, dict(active=self.context,
            elite_schedule=dict(policy='clearance-first-round-robin/v1',
                                queue=['a','b'],visits={'a':1})))
        snap=self.dashboard.snapshot()
        p=snap['search_progress']
        self.assertAlmostEqual(p['roomiest_clearance_cm'],7.98)
        self.assertEqual(p['fastest_clearance_cm'],5.)
        self.assertEqual(p['roomiest_lap_s'],7.075)
        self.assertEqual(p['pareto_choices'],2)
        self.assertEqual(p['nlp_threshold'],1.e-5)
        self.assertFalse(p['trial_certified'])
        self.assertEqual(snap['elite_progress']['schedule']['queue'],['a','b'])

    def test_small_saved_residual_does_not_hide_failed_different_iterate_replay(self):
        saved=dict(iteration=90,residual=7.405449399366937e-6)
        replay=dict(iteration=120,periodic_scaled_defect=0.,
            source_xy_error_m=.00022210294695753242,
            clearance_slack_m=-.0010878232921666536,target_slack_m=0.)
        result=candidate_acceptance(saved,replay,None)
        gates={g['id']:g for g in result['gates']}
        self.assertEqual(gates['constraints']['status'],'passed')
        self.assertEqual(gates['closure']['status'],'passed')
        self.assertEqual(gates['drift']['status'],'passed')
        self.assertEqual(gates['clearance']['status'],'failed')
        self.assertAlmostEqual(gates['clearance']['value'],.04891217670783335)
        self.assertEqual(gates['archive']['status'],'pending')
        self.assertIn('short by 1.09 mm',result['summary'])
        self.assertEqual(result['replay_iteration'],120)
        self.assertEqual(result['checkpoint_iteration'],90)
        self.assertFalse(result['replay_matches_checkpoint'])

    def test_passing_early_replay_is_not_a_final_archive_certificate(self):
        result=candidate_acceptance(dict(iteration=80,residual=1.e-7),
            dict(iteration=80,periodic_scaled_defect=0.,
                 source_xy_error_m=.001,clearance_slack_m=.01),None)
        self.assertTrue(result['replay_matches_checkpoint'])
        self.assertEqual(result['status'],'pending')
        self.assertEqual([g['status'] for g in result['gates']],['passed']*4+['pending'])
        self.assertIn('archiving are still pending',result['summary'])

    def test_completed_report_supersedes_early_replay_without_inventing_closure(self):
        result=candidate_acceptance(dict(iteration=90,residual=1.e-7),
            dict(iteration=80,source_xy_error_m=.1,clearance_slack_m=-.01),
            dict(max_constraint_violation=1.e-8,
                 rollout=dict(full_lap_xy_max_m=.001,minimum_1ms_footprint_slack_m=.002)))
        gates={g['id']:g for g in result['gates']}
        self.assertEqual(gates['drift']['status'],'passed')
        self.assertEqual(gates['clearance']['status'],'passed')
        self.assertEqual(gates['closure']['status'],'passed')
        self.assertIsNone(gates['closure']['value'])
        self.assertIsNone(result['replay_iteration'])

    def test_predictor_passing_constraints_cannot_pass_source_acceptance(self):
        result=candidate_acceptance(dict(iteration=20,residual=1.e-6),None,None,predicting=True)
        self.assertEqual(result['gates'][0]['status'],'passed')
        self.assertEqual(result['status'],'initializer_only')
        self.assertEqual(len(result['gates']),1)

    def test_snapshot_replay_scope_and_incumbent_certificate_are_not_mixed(self):
        self.start_log()
        self.write('best.json',dict(files=dict(validation='incumbent.json')))
        self.write('incumbent.json',dict(status='source_witness_passed',
            rollout=dict(full_lap_xy_max_m=0.,minimum_1ms_footprint_slack_m=.03)))
        np.savez_compressed(self.folder/'best_iterate.npz',iteration=90,
                            max_constraint_violation=7.41e-6)
        result=self.dashboard.snapshot()['search_progress']['acceptance']
        self.assertEqual(result['gates'][0]['status'],'passed')
        self.assertTrue(all(g['status']=='pending' for g in result['gates'][1:]))
        with self.log.open('a') as stream:
            stream.write(json.dumps(dict(event='early_source_checkpoint_gate',iteration=120,
                periodic_scaled_defect=0.,source_xy_error_m=.0002,
                clearance_slack_m=-.001,target_slack_m=0.,accepted=False))+'\n')
        self.dashboard.last_poll=0
        result=self.dashboard.snapshot()['search_progress']['acceptance']
        self.assertEqual(result['status'],'blocked')
        with self.log.open('a') as stream:
            stream.write(json.dumps(dict(event='solve',output=str(self.folder/'next')))+'\n')
        self.dashboard.last_poll=0
        result=self.dashboard.snapshot()['search_progress']['acceptance']
        self.assertEqual(result['gates'][3]['status'],'pending')
        self.assertIsNone(self.dashboard.log.source_gate)

    def test_nondefault_planning_margin_is_not_hardcoded_in_acceptance(self):
        result=candidate_acceptance(dict(iteration=1,residual=1.e-7),
            dict(iteration=1,clearance_slack_m=-.001),None,margin=.12)
        gap=next(g for g in result['gates'] if g['id']=='clearance')
        self.assertAlmostEqual(gap['value'],.119)
        self.assertEqual(gap['limit'],.12)

    def test_numerical_scale_miss_is_not_reported_as_zero_millimetres(self):
        result=candidate_acceptance(dict(iteration=10,residual=1.e-8),
            dict(iteration=10,clearance_slack_m=-5.e-7),None)
        self.assertIn('500.00 nm',result['summary'])
        self.assertNotIn('0.00 mm',result['summary'])

    def test_clearance_gate_matches_declared_verifier_tolerance(self):
        replay=dict(iteration=10,clearance_slack_m=-5.e-7,
                    clearance_numerical_tolerance_m=1.e-6)
        result=candidate_acceptance(dict(iteration=10,residual=1.e-8),replay,None)
        gate=next(g for g in result['gates'] if g['id']=='clearance')
        self.assertEqual(gate['status'],'passed')
        self.assertEqual(gate['tolerance_m'],1.e-6)
        self.assertAlmostEqual(gate['value'],.0499995)
        replay['clearance_slack_m']=-2.e-6
        result=candidate_acceptance(None,replay,None)
        self.assertEqual(next(g for g in result['gates'] if g['id']=='clearance')['status'],'failed')
        replay['clearance_numerical_tolerance_m']=.01
        result=candidate_acceptance(None,replay,None)
        self.assertEqual(next(g for g in result['gates'] if g['id']=='clearance')['status'],'failed')

    def test_replay_schedule_and_rounding_explanation_match_current_source_policy(self):
        html=Path(__file__).with_name('optimizer_dashboard.html').read_text()
        source=(Path(__file__).resolve().parents[1]/
            'SDU_Apex_IPOPT_PhysX_MinTime_v2/command_ocp.py').read_text()
        self.assertIn('iteration-early_checked_iteration[0]>=40',source)
        self.assertIn('source_replay_every_iterations??40',html)
        self.assertIn('id="replaySchedule"',html)
        self.assertIn('a saved checkpoint passing does not mean the current iterate passes',html)
        self.assertIn('Math.ceil(-Math.log10(miss*1000))+1',html)

    def test_intentional_stagnation_is_reported_without_claiming_optimizer_crash(self):
        self.start_log()
        with self.log.open('a') as stream:
            stream.write('RuntimeError: command feasibility stagnated; durable best iterate saved\n')
        result = self.dashboard.snapshot()
        event = next(event for event in result['events']
                     if event.get('event')=='feasibility_stagnation')
        self.assertEqual(event['event'], 'feasibility_stagnation')
        self.assertEqual(result['status'], 'not_running')
        self.assertTrue(any('RuntimeError: command feasibility stagnated' in line
                            for line in result['log_lines']))

    def test_raw_ipopt_history_updates_without_actual_residual_callback(self):
        self.start_log()
        with self.log.open('a') as stream:
            stream.write('  1  1.234567e-04 1.0e-02 2.0e-04 -3 0.5 - 1 1h 1\n')
            stream.write('  2  1.234567e-04 3.0e-03 2.0e-04 -3 0.5 - 1 1h 1\n')
        result = self.dashboard.snapshot()
        self.assertEqual(result['iteration'], 2)
        self.assertEqual(len(result['ipopt_history']), 2)
        self.assertEqual(result['ipopt_history'][-1]['primal_infeasibility'], .003)
        self.assertEqual(result['residuals'], [])
        with self.log.open('a') as stream:
            stream.write(json.dumps(dict(event='solve',output='different-solve'))+'\n')
        self.dashboard.log.poll()
        self.assertEqual(list(self.dashboard.log.ipopt_points), [])

    def test_partial_lines_and_new_solve_do_not_retain_old_iteration(self):
        reader = LogObserver(self.log)
        self.log.write_text('{"iteration":5,"max_constraint_violation":0.4}')
        reader.poll()
        self.assertIsNone(reader.live_iteration)
        with self.log.open('a') as stream:
            stream.write('\n')
        reader.poll()
        self.assertEqual(reader.live_iteration, 5)
        with self.log.open('a') as stream:
            stream.write('{"event":"solve","output":"new"}\n')
        reader.poll()
        self.assertIsNone(reader.live_iteration)
        self.assertEqual(list(reader.points), [])
        self.log.write_text('{"iteration":0,"max_constraint_violation":0.8}\n')
        reader.poll()
        self.assertEqual(reader.live_iteration, 0)
        self.assertEqual(len(reader.points), 1)

    def test_campaign_change_before_log_header_hides_previous_solve(self):
        self.start_log()
        with self.log.open('a') as stream:
            stream.write('{"iteration":70,"max_constraint_violation":0.03}\n')
        next_context = dict(key='from90_81_phase020_resample', output=str(self.root/'new_solve'))
        self.write(self.campaign_path, dict(active=next_context))
        result = self.dashboard.snapshot()
        self.assertIsNone(result['iteration'])
        self.assertEqual(result['residuals'], [])
        self.assertAlmostEqual(result['active']['trial_lap_s'], 2.025)
        self.assertEqual(result['active']['phase'], 20)

    def test_read_only_snapshot_does_not_touch_any_workspace_file(self):
        self.start_log()
        before = {path: (path.stat().st_mtime_ns, hashlib.sha256(path.read_bytes()).hexdigest())
                  for path in self.root.rglob('*') if path.is_file()}
        self.dashboard.snapshot()
        after = {path: (path.stat().st_mtime_ns, hashlib.sha256(path.read_bytes()).hexdigest())
                 for path in self.root.rglob('*') if path.is_file()}
        self.assertEqual(before, after)

    def test_model_only_exports_preserve_validity_flags(self):
        manifest = dict(lap_s=2.5, direction='clockwise', source_only=True,
                        unity_validated=False, mpc_validated=False, runtime_promoted=False,
                        files={})
        self.write('best.json', manifest)
        self.write('exports/abc/manifest.json', manifest)
        result = self.dashboard.snapshot()
        self.assertEqual(result['history'][0]['lap_s'], 2.5)
        self.assertFalse(result['best']['unity_validated'])
        self.assertFalse(result['best']['mpc_validated'])
        json_bytes(result)  # Strict JSON: never NaN / Infinity in the browser API.

    def test_mpc_failure_diagnostic_extracts_state_solver_and_exact_prefix_checks(self):
        run = self.root/'trial'
        run.mkdir()
        payload = dict(status='rejected_nonlinear_rollout', progress_m=12.5,
            path_curvature_per_m=.4, state=[-.12, -.08, 4.2, -.1, 1.8],
            reference=dict(speed_reference_mps=4.0, trajectory_acceleration_mps2=1.2),
            solver=dict(iterations=250, max_iterations_configured=250,
                        admm_stop_reason='iteration_cap', primal_residual=.3,
                        dual_residual=.08),
            candidate_prefix_revalidation=dict(exact_valid=False, failure_stage=1),
            prior_plan_revalidation=dict(exact_valid=False, failure_stage=2),
            candidate_corridor_slack_by_stage_m=[
                dict(stage=1, lower=.04, upper=.5),
                dict(stage=2, lower=-.007, upper=.4)],
            end_to_end_control_ms=14.5, end_to_end_cycle_budget_ms=20.0)
        recovery = dict(status='accepted_prior_plan', progress_m=12.4,
            action_source='revalidated_prior_plan',
            quality_aware_prefix_selection=dict(
                attempted=True, choice='retained_plan',
                near_boundary_override=False,
                candidate_minimum_corridor_slack_m=.006,
                prior_minimum_corridor_slack_m=.0002,
                candidate_objective=19.0, prior_objective=13.0))
        (run/'mpc_cycle_detail_diagnostics.jsonl').write_text(
            json.dumps(dict(payload=recovery))+'\n'+
            json.dumps(dict(payload=payload))+'\n')
        result=mpc_failure_diagnostic(run)
        self.assertEqual(result['progress_m'],12.5)
        self.assertEqual(result['cross_track_m'],-.12)
        self.assertEqual(result['reference_speed_mps'],4.0)
        self.assertEqual(result['candidate_invalid_stage'],2)
        self.assertEqual(result['candidate_invalid_slack_m'],-.007)
        self.assertEqual(result['iterations'],250)
        self.assertEqual(result['yaw_rate_radps'],1.8)
        self.assertEqual(result['reference_acceleration_mps2'],1.2)
        self.assertFalse(result['candidate_exact_valid'])
        self.assertFalse(result['prior_exact_valid'])
        self.assertEqual(result['last_recovery_decision']['choice'], 'retained_plan')
        self.assertAlmostEqual(result['last_recovery_decision']['objective_ratio'], 19/13)
        self.assertFalse(result['last_recovery_decision']['near_boundary_override'])

    def test_mpc_card_exposes_terminal_rollout_diagnosis(self):
        html=Path(__file__).with_name('optimizer_dashboard.html').read_text()
        self.assertIn('Last nonlinear rejection · decoded from bag',html)
        self.assertIn('Candidate corridor slack',html)
        self.assertIn('Path curvature',html)
        self.assertIn('Requested acceleration',html)
        self.assertIn('Last validated recovery choice before rejection',html)
        self.assertIn('Near-wall rule',html)

    def test_bad_checkpoint_is_reported_without_killing_dashboard(self):
        (self.folder/'best_iterate.npz').write_bytes(b'interrupted file')
        result = self.dashboard.snapshot()
        self.assertIsNone(result['checkpoint'])
        self.assertTrue(any('Checkpoint unavailable' in value for value in result['warnings']))

    def test_map_and_reference_are_read_in_their_own_frame(self):
        (self.root/'input').mkdir()
        Image.new('L', (20, 30), 255).save(self.root/'input/map.pgm')
        self.write('input/map.yaml', dict(origin=[3, -4, .7], resolution=.05))
        csv = self.root/'exports/abc/line.csv'
        csv.parent.mkdir(parents=True)
        np.savetxt(csv, [[0, 3, -4], [1, 3.5, -3.5]], delimiter=',')
        self.write('best.json', dict(files=dict(raceline=str(csv)), witness_sha256='abc'))
        data = self.dashboard.track()
        self.assertEqual(data['origin'], [3, -4, .7])
        self.assertEqual(data['resolution'], .05)
        self.assertEqual((data['width'], data['height']), (20, 30))
        self.assertEqual(data['layers'][0]['points'], [[3., -4.], [3.5, -3.5]])
        self.assertTrue(self.dashboard.map_payload()[0].startswith(b'\x89PNG'))

    def test_command_seed_campaign_extracts_and_caches_missing_centerline(self):
        # Command-seed workflows intentionally have no geometry_starts; this
        # must not silently remove the centerline layer from the viewer.
        from time import monotonic, sleep
        (self.root/'input').mkdir()
        Image.new('L', (20, 30), 255).save(self.root/'input/map.pgm')
        self.write('input/map.yaml', dict(origin=[0, 0, 0], resolution=.05))
        points = [[0., 0.], [1., 0.], [1., 1.], [0., 1.]]
        with patch('tools.optimizer_dashboard.map_centerline', return_value=points) as extract:
            self.dashboard.track()
            end = monotonic()+2.
            while self.dashboard.centerline_status == 'extracting' and monotonic() < end:
                sleep(.005)
            data = self.dashboard.track()
            self.assertEqual(data['centerline_status'], 'ready')
            self.assertEqual(data['layers'][0]['kind'], 'centerline')
            self.assertEqual(data['layers'][0]['points'], points)
            self.dashboard.track()
            extract.assert_called_once()

    def test_file_reads_reject_traversal_and_symlinks_outside_workspace(self):
        workspace = Workspace(self.root)
        with self.assertRaises(ValueError):
            workspace.path('../secret')
        (self.root/'outside').symlink_to(self.root.parent, target_is_directory=True)
        with self.assertRaises(ValueError):
            workspace.path('outside/secret')
        self.write('best.json', dict(files=dict(raceline='/etc/passwd')))
        with self.assertRaises(ValueError):
            self.dashboard.download('raceline.csv')
        with self.assertRaises(FileNotFoundError):
            self.dashboard.download('../../etc/passwd')

    def test_pid_reuse_does_not_claim_optimizer_running(self):
        with patch.object(Path, 'read_bytes', return_value=b'/usr/bin/other-process\x00'), \
             patch.object(Path, 'read_text', return_value='State:\tS\nVmRSS:\t120 kB\nThreads:\t1\n'):
            self.assertFalse(process_info(123)['alive'])

    def test_unity_queue_exposes_trial_metrics_and_honest_worker_status(self):
        self.write('mpc_validation/state.json', dict(
            pending=['def567'], worker=dict(status='running'),
            attempts=[dict(id='abc123', model_lap_s=7.35, status='completed',
                           run_id='apex_mpc_abc1234567890abc_a01', attempt=1,
                           metrics=dict(average_scored_s=7.6,
                                        scored_lap_times_s=[7.6]*10))]))
        result=self.dashboard.snapshot()['mpc_trials']
        self.assertEqual(result['pending'],1)
        self.assertEqual(result['attempts'][0]['metrics']['average_scored_s'],7.6)
        self.assertEqual(result['attempts'][0]['status'],'completed')
        self.assertFalse(result['supervisor_alive'])
        self.assertEqual(result['total_trials'], 1)

    def test_only_latest_unity_attempt_is_sent_to_gui(self):
        self.write('mpc_validation/state.json', dict(
            pending=[], worker=dict(status='idle'),
            attempts=[
                dict(id='original', model_lap_s=11.275, status='completed',
                     attempt=1, started_epoch_s=100.,
                     metrics=dict(completed_scored_laps=10)),
                dict(id='latest', model_lap_s=8.25, status='mpc_rejected',
                     attempt=1, started_epoch_s=300.,
                     reason='rejected_nonlinear_rollout; nonlinear=corridor@2',
                     metrics=dict(completed_scored_laps=1)),
                dict(id='middle', model_lap_s=9.15, status='mpc_rejected',
                     attempt=1, started_epoch_s=200.,
                     metrics=dict(completed_scored_laps=1))]))
        result=self.dashboard.snapshot()['mpc_trials']
        self.assertEqual(len(result['attempts']),1)
        self.assertEqual(result['attempts'][0]['id'],'latest')
        self.assertEqual(result['attempts'][0]['metrics']['completed_scored_laps'],1)

    def test_unity_lap_count_has_independent_height_and_no_clipping(self):
        html = Path(__file__).with_name('optimizer_dashboard.html').read_text()
        self.assertEqual(html.count('function renderMpc(data){'),1)
        self.assertNotIn('class="mpc-progress-label"',html)
        self.assertIn("mpcAppend(summary,top,quick,bar)",html)
        self.assertIn('mpc-summary-meta',html)
        self.assertIn("'Warm-up: '+mpcNum(m.warmup_lap_s)",html)
        self.assertIn('Warm-up not completed',html)
        self.assertIn('The rejection alone does not identify the cause',html)
        self.assertNotIn('The line may be too aggressive',html)
        self.assertIn("slice(0,1)",html)

    def test_http_endpoints_are_local_read_only_and_no_generic_file_server(self):
        server = ThreadingHTTPServer(('127.0.0.1', 0), handler_for(self.dashboard))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        url = f'http://127.0.0.1:{server.server_port}'
        with urlopen(url+'/api/snapshot') as response:
            self.assertEqual(json.load(response)['name'], self.root.name)
            self.assertEqual(response.headers['Cache-Control'], 'no-store')
        with urlopen(url+'/') as response:
            self.assertIn(b'Raceline optimizer', response.read())
        cases = [('/etc/passwd', None, 404), ('/stop', b'', 403),
                 ('/api/snapshot', None, 403)]
        for path, data, code in cases:
            headers = {'Origin': 'https://attacker.invalid'} if code == 403 else {}
            with self.assertRaises(HTTPError) as caught:
                urlopen(Request(url+path, data=data, headers=headers))
            self.assertEqual(caught.exception.code, code)

    def test_retry_endpoint_requires_same_origin_and_json(self):
        server = ThreadingHTTPServer(('127.0.0.1', 0), handler_for(self.dashboard))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        url = f'http://127.0.0.1:{server.server_port}'
        for origin in (None, 'https://attacker.invalid'):
            headers = {'Content-Type': 'application/json'}
            if origin:
                headers['Origin'] = origin
            with self.assertRaises(HTTPError) as caught:
                urlopen(Request(url+'/api/retry', data=b'{}', headers=headers))
            self.assertEqual(caught.exception.code, 403)
        with patch.object(self.dashboard, 'retry', return_value=dict(status='queued')) as retry:
            request = Request(url+'/api/retry', data=b'{"id":"aaaaaaaaaaaaaaaa"}',
                              headers={'Origin': url, 'Content-Type': 'application/json'})
            with urlopen(request) as response:
                self.assertEqual(response.status, 202)
                self.assertEqual(json.load(response), dict(status='queued'))
            retry.assert_called_once_with('aaaaaaaaaaaaaaaa')

    def test_retry_refuses_stopped_supervisor_and_invalid_candidates(self):
        with self.assertRaisesRegex(RuntimeError, 'supervisor'):
            self.dashboard.retry('a'*16)
        with patch('tools.optimizer_dashboard.validation_supervisor', return_value=True):
            with self.assertRaises(ValueError):
                self.dashboard.retry('../unsafe')
            with self.assertRaises(ValueError):
                self.dashboard.retry('a'*16)


if __name__ == '__main__':
    unittest.main()

def test_adaptive_corner_phase_is_visible():
    from pathlib import Path
    page=Path(__file__).with_name('optimizer_dashboard.html').read_text()
    assert 'Adaptive corner ' in page
    assert 'apex shift ' in page
    assert 'ADAPTIVE CORNER SEARCH' in page
