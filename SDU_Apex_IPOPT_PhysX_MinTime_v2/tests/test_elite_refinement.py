"""Mocked archive tests: no claims about physical car dynamics."""
import json
from pathlib import Path

import numpy as np
import pytest

from command_ocp import save_command_checkpoint
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
