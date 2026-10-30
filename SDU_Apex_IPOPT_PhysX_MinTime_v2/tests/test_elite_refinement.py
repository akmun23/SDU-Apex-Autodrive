"""Mocked archive tests: no claims about physical car dynamics."""
import json
from pathlib import Path

import numpy as np
import pytest

from command_ocp import save_command_checkpoint, save_npz_checkpoint
from elite_refinement import polish_equal_time, quality_of_certificate


def witness(path, count=40, marker=0.):
    x=np.zeros((21,count+1))
    x[2]=np.linspace(0,2*np.pi,count+1)
    x[0]=marker
    x[3]=2.
    u=np.vstack([np.zeros(count),np.full(count,.1)])
    save_command_checkpoint(path,x,u,'fixture',0.,0)


def certificate(path, count=40, slack=0., marker=0., valid=True):
    path.mkdir(parents=True,exist_ok=True)
    witness(path/'witness.npz',count,marker)
    result=dict(status='source_witness_passed' if valid else 'no_validated_witness',
                trial_lap_s=count*.025,
                rollout=dict(minimum_1ms_footprint_slack_m=slack))
    (path/'report.json').write_text(json.dumps(result))
    return result


def test_quality_requires_independently_passed_positive_clearance():
    assert quality_of_certificate(dict(status='no_validated_witness',
        rollout=dict(minimum_1ms_footprint_slack_m=1.))) is None
    assert quality_of_certificate(dict(status='source_witness_passed',
        rollout=dict(minimum_1ms_footprint_slack_m=-1.e-9))) is None
    assert quality_of_certificate(dict(status='source_witness_passed',
        rollout=dict(minimum_1ms_footprint_slack_m=.003)))==pytest.approx(.003)


def test_equal_time_alternatives_archived_and_restored(tmp_path):
    baseline=tmp_path/'baseline'
    certificate(baseline,slack=.00001)
    grid=tmp_path/'map.pgm'
    grid.write_bytes(b'mock grid')
    calls=[]
    def solve(**kw):
        calls.append(kw)
        kw['output'].mkdir()
        witness(kw['output']/'witness.npz',40,len(calls))
        result=dict(status='source_witness_passed',trial_lap_s=1.)
        (kw['output']/'report.json').write_text(json.dumps(result))
        return result
    def certify(repo,seed,map_path,out):
        return certificate(out,slack=.0003 if len(calls)==1 else .0012,
                           marker=len(calls))
    kwargs=dict(repo=tmp_path,seed=baseline/'witness.npz',map_path=grid,
                output=tmp_path/'elite',margin_steps=(.0001,.0005),
                iterations=30,workers=1,solve=solve,certify=certify)
    state=polish_equal_time(**kwargs)
    assert state['complete'] and len(state['accepted'])==2
    assert state['best_clearance_slack_m']==pytest.approx(.0012)
    assert calls[0]['count']==calls[1]['count']==40
    assert calls[0]['margin']==pytest.approx(.0501)
    assert calls[1]['margin']==pytest.approx(.0505)
    assert calls[1]['command_seed']==Path(state['accepted'][0]['witness'])
    assert polish_equal_time(**kwargs)['attempts']==state['attempts']
    assert len(calls)==2
    assert all(Path(x['witness']).exists() for x in state['accepted'])
    grid.write_bytes(b'changed map')
    with pytest.raises(ValueError,match='contract'):
        polish_equal_time(**kwargs)


def test_failed_or_rejected_polish_is_never_promoted(tmp_path):
    baseline=tmp_path/'baseline'
    certificate(baseline,slack=.0004)
    grid=tmp_path/'map.pgm'
    grid.write_bytes(b'mock grid')
    calls=[]
    def solve(**kw):
        calls.append(kw)
        kw['output'].mkdir()
        witness(kw['output']/'witness.npz')
        result=dict(status='source_witness_passed',trial_lap_s=1.)
        (kw['output']/'report.json').write_text(json.dumps(result))
        return result
    def certify(repo,seed,map_path,out):
        return certificate(out,slack=-.001 if len(calls)==1 else .0002,
                           valid=len(calls)!=1)
    state=polish_equal_time(repo=tmp_path,seed=baseline/'witness.npz',
        map_path=grid,output=tmp_path/'elite',margin_steps=(.0001,.0002),
        solve=solve,certify=certify)
    assert state['accepted']==[]
    assert state['best_clearance_slack_m']==pytest.approx(.0004)
    assert all(kw['command_seed']==baseline/'witness.npz' for kw in calls)


def test_geometry_perturbation_is_a_distinct_uncertified_periodic_seed(tmp_path):
    from elite_refinement import geometry_seed, state_diversity_m
    seed=tmp_path/'witness.npz'
    witness(seed,count=40)
    shifted=geometry_seed(seed,tmp_path/'alternate.npz',.08,2)
    assert state_diversity_m(seed,shifted)>.03
    with np.load(seed,allow_pickle=False) as original, np.load(shifted,allow_pickle=False) as changed:
        np.testing.assert_array_equal(original['controls'],changed['controls'])
        np.testing.assert_array_equal(original['state'][3:],changed['state'][3:])
        assert changed['state'][0,-1]==pytest.approx(changed['state'][0,0])
        assert changed['state'][1,-1]==pytest.approx(changed['state'][1,0])
    with pytest.raises(ValueError):
        geometry_seed(seed,tmp_path/'invalid.npz',1.0,1)


def test_quality_search_repairs_tick_clearance_and_exports_equal_time_immediately(tmp_path):
    baseline=tmp_path/'baseline'
    certificate(baseline,slack=0.)
    grid=tmp_path/'map.pgm'
    grid.write_bytes(b'mock grid')
    calls,published=[],[]
    def solve(**kw):
        calls.append(kw)
        kw['output'].mkdir()
        witness(kw['output']/'witness.npz')
        result=dict(status='source_witness_passed' if len(calls)>1 else 'no_validated_witness',
            trial_lap_s=1.,max_constraint_violation=1.e-6,
            rollout=dict(full_lap_xy_max_m=.001,
                         violating_source_ticks=[dict(interval=3,tick=9)]))
        (kw['output']/'report.json').write_text(json.dumps(result))
        return result
    def certify(repo,seed,map_path,out):
        return certificate(out,slack=.008)
    state=polish_equal_time(repo=tmp_path,seed=baseline/'witness.npz',map_path=grid,
        output=tmp_path/'elite',margin_steps=(),quality_objectives=('clearance',),
        solve=solve,certify=certify,on_candidate=lambda path,lap:published.append((path,lap)))
    assert len(calls)==2
    assert calls[0]['objective_mode']==calls[1]['objective_mode']=='clearance'
    assert calls[1]['clearance_cuts']==[(3,9)]
    assert calls[1]['command_seed']==calls[0]['output']/'witness.npz'
    assert len(published)==1 and published[0][1]==pytest.approx(1.)
    assert published[0][0]==Path(state['accepted'][0]['witness'])


def test_infeasible_quality_proposal_is_restored_then_archived_at_same_duration(tmp_path):
    baseline=tmp_path/'baseline'
    certificate(baseline,slack=.0002)
    grid=tmp_path/'map.pgm'
    grid.write_bytes(b'mock grid')
    calls,published=[],[]
    def solve(**kw):
        calls.append(kw)
        kw['output'].mkdir()
        if kw['objective_mode'] in ('clearance','smooth'):
            # The quality solve falls back to the feasible baseline witness,
            # while preserving its improved geometry separately.
            witness(kw['output']/'witness.npz',40,0.)
            proposal_state=np.zeros((21,41))
            proposal_state[0]=.03
            proposal_state[2]=np.linspace(0.,2*np.pi,41)
            proposal_state[3]=2.
            save_npz_checkpoint(kw['output']/'best_quality_proposal.npz',
                schema='production-command-ocp/v1',state=proposal_state,
                controls=np.vstack([np.full(40,.02),np.full(40,.1)]),
                dt_s=.025,lap_time_s=1.,source_sha256='fixture',
                max_constraint_violation=.0036,iteration=25,
                objective_mode='clearance',objective_value=-.31,
                baseline_objective=.1,proposal_merit=-.274)
            result=dict(status='no_validated_witness',trial_lap_s=1.,
                max_constraint_violation=.0036,rollout=dict(full_lap_xy_max_m=.001,
                    violating_source_ticks=[]))
        else:
            assert kw['objective_mode']=='repair'
            with np.load(kw['command_seed'],allow_pickle=False) as proposal:
                assert float(proposal['max_constraint_violation'])==pytest.approx(.0036)
                assert float(proposal['objective_value']) < 0.
            witness(kw['output']/'witness.npz',40,.03)
            result=dict(status='source_witness_passed',trial_lap_s=1.,
                max_constraint_violation=1.e-6,rollout=dict(full_lap_xy_max_m=.001,
                    violating_source_ticks=[]))
        (kw['output']/'report.json').write_text(json.dumps(result))
        return result
    def certify(repo,seed,map_path,out):
        return certificate(out,slack=.001,marker=.03)
    state=polish_equal_time(repo=tmp_path,seed=baseline/'witness.npz',map_path=grid,
        output=tmp_path/'elite',margin_steps=(),quality_objectives=('clearance',),
        solve=solve,certify=certify,
        on_candidate=lambda path,lap:published.append((path,lap)))
    assert [kw['objective_mode'] for kw in calls]==['clearance','repair']
    assert calls[1]['command_seed']==calls[0]['output']/'best_quality_proposal.npz'
    assert calls[1]['output'].name=='polish_00_restore'
    assert calls[1]['count']==40
    assert calls[1]['early_clearance_target_m']==pytest.approx(.0022)
    assert state['attempts'][0]['proposal_restoration_status']=='source_witness_passed'
    assert len(state['accepted'])==1
    assert state['accepted'][0]['lap_s']==pytest.approx(1.)
    assert published==[(Path(state['accepted'][0]['witness']),pytest.approx(1.))]


def test_two_interrupted_restorations_resume_latest_folder_and_cuts(tmp_path):
    baseline=tmp_path/'baseline'
    certificate(baseline,slack=.0002)
    grid=tmp_path/'map.pgm'
    grid.write_bytes(b'mock grid')
    calls=[]
    def solve(**kw):
        calls.append(kw)
        kw['output'].mkdir()
        if kw['objective_mode'] in ('clearance','smooth'):
            witness(kw['output']/'witness.npz')
            save_npz_checkpoint(kw['output']/'best_quality_proposal.npz',
                schema='production-command-ocp/v1',state=np.zeros((21,41)),
                controls=np.zeros((2,40)),dt_s=.025,lap_time_s=1.,
                source_sha256='fixture',max_constraint_violation=.003,
                iteration=10,objective_mode='smooth',objective_value=-.2,
                baseline_objective=.1,proposal_merit=-.17)
            result=dict(status='no_validated_witness',trial_lap_s=1.,
                max_constraint_violation=.003,rollout=dict(full_lap_xy_max_m=.001,
                    violating_source_ticks=[]))
        else:
            assert kw['objective_mode']=='repair'
            repair_call=len([call for call in calls if call['objective_mode']=='repair'])
            if repair_call==1:
                witness(kw['output']/'best_iterate.npz')
                raise KeyboardInterrupt('fixture interruption')
            if repair_call==2:
                assert kw['output'].name=='polish_00_restore_001'
                assert kw['command_seed']==tmp_path/'elite'/'polish_00_restore'/'best_iterate.npz'
                witness(kw['output']/'witness.npz',40,.02)
                result=dict(status='no_validated_witness',trial_lap_s=1.,
                    max_constraint_violation=1.e-6,rollout=dict(full_lap_xy_max_m=.001,
                        violating_source_ticks=[dict(interval=3,tick=9)]))
            elif repair_call==3:
                assert kw['output'].name=='polish_00_restore_cut_01'
                assert kw['clearance_cuts']==[(3,9)]
                witness(kw['output']/'best_iterate.npz',40,.02)
                raise KeyboardInterrupt('fixture interruption after cut discovery')
            else:
                assert kw['output'].name=='polish_00_restore_cut_01_001'
                assert kw['command_seed']==tmp_path/'elite'/'polish_00_restore_cut_01'/'best_iterate.npz'
                assert kw['clearance_cuts']==[(3,9)]
                witness(kw['output']/'witness.npz',40,.03)
                result=dict(status='source_witness_passed',trial_lap_s=1.,
                    max_constraint_violation=1.e-6,rollout=dict(full_lap_xy_max_m=.001,
                        violating_source_ticks=[]))
        (kw['output']/'report.json').write_text(json.dumps(result))
        return result
    def certify(repo,seed,map_path,out):
        return certificate(out,slack=.001,marker=.03)
    kwargs=dict(repo=tmp_path,seed=baseline/'witness.npz',map_path=grid,
        output=tmp_path/'elite',margin_steps=(),quality_objectives=('smooth',),
        solve=solve,certify=certify)
    with pytest.raises(KeyboardInterrupt):
        polish_equal_time(**kwargs)
    active=json.loads((tmp_path/'elite'/'elite.json').read_text())['active_attempt']
    assert active['strategy']=='proposal_restoration'
    assert active['output'].endswith('polish_00_restore')
    assert (tmp_path/'elite'/'polish_00_restore'/'best_iterate.npz').exists()
    with pytest.raises(KeyboardInterrupt):
        polish_equal_time(**kwargs)
    active=json.loads((tmp_path/'elite'/'elite.json').read_text())['active_attempt']
    assert active['output'].endswith('polish_00_restore_cut_01')
    assert active['source_clearance_cuts']==[[3,9]]
    assert (tmp_path/'elite'/'polish_00_restore_cut_01'/'best_iterate.npz').exists()
    state=polish_equal_time(**kwargs)
    assert state['attempts'][0]['proposal_restoration_status']=='source_witness_passed'
    assert calls[-1]['output'].name=='polish_00_restore_cut_01_001'
    assert (tmp_path/'elite'/'polish_00_restore'/'best_iterate.npz').exists()
    assert calls[-1]['clearance_cuts']==[(3,9)]
    assert all(Path(record['witness']).parent.name.startswith('certified_00_')
               for record in state['accepted'])


def test_control_basin_generation_precedes_geometry_and_is_not_auto_certified(
        tmp_path,monkeypatch):
    import causal_seed
    origin=tmp_path/'source'
    certificate(origin,slack=.00005)
    grid=tmp_path/'map.pgm'
    grid.write_bytes(b'fixture map')
    control_calls=[]
    def generate(repo,seed,destination,amp,harmonic,**kwargs):
        control_calls.append((amp,harmonic))
        destination.parent.mkdir(parents=True,exist_ok=True)
        with np.load(seed,allow_pickle=False) as archive:
            x=archive['state'].copy()
            u=archive['controls'].copy()
        x[0]+=amp*10
        u[0]+=amp
        save_command_checkpoint(destination,x,u,'fixture',.003,-1)
        return dict(path=str(destination),source_certified=False,
                    initial_scaled_dynamic_defect=.003)
    monkeypatch.setattr(causal_seed,'source_alternative_seed',generate)
    attempts=[]
    def solve(**kw):
        attempts.append(kw)
        kw['output'].mkdir()
        with np.load(kw['command_seed'],allow_pickle=False) as candidate:
            x=candidate['state'].copy();u=candidate['controls'].copy()
        save_command_checkpoint(kw['output']/'witness.npz',x,u,'fixture',0.,3)
        result=dict(status='source_witness_passed',trial_lap_s=1.)
        (kw['output']/'report.json').write_text(json.dumps(result))
        return result
    def certify(repo,seed,map_path,out):
        out.mkdir()
        with np.load(seed,allow_pickle=False) as inp:
            save_command_checkpoint(out/'witness.npz',inp['state'],inp['controls'],'fixture',0.,3)
        result=dict(status='source_witness_passed',trial_lap_s=1.,
                    rollout=dict(minimum_1ms_footprint_slack_m=.00005))
        (out/'report.json').write_text(json.dumps(result))
        return result
    result=polish_equal_time(repo=tmp_path,seed=origin/'witness.npz',map_path=grid,
        output=tmp_path/'elites',margin_steps=(),control_seeds=((.002,1),(-.002,1)),
        solve=solve,certify=certify,diversity_threshold_m=.015)
    assert control_calls==[(.002,1),(-.002,1)]
    assert len(result['attempts'])==2
    assert all(a['strategy']=='control' for a in result['attempts'])
    assert len(result['accepted'])==2
    assert len(result['control_seed_diagnostics'])==2
