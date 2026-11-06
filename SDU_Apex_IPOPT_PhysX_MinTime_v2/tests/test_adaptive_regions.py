import numpy as np
import pytest
from adaptive_regions import (localized_steering_profile,ranked_sectors,
                              choose_adaptive_region)


def test_localized_periodic_perturbations_have_small_net_steer_and_are_bounded():
    for sector in range(8):
        p=localized_steering_profile(284,sector=sector,sectors=8,
                                     amplitude_rad=.001)
        assert abs(p.mean())<1e-14
        assert np.max(np.abs(p))==pytest.approx(.001)
        assert np.count_nonzero(np.abs(p)>.0008)<80
    with pytest.raises(ValueError):
        localized_steering_profile(284,sector=8,sectors=8,amplitude_rad=.001)


def test_hotspot_scores_track_curvature_and_wall_proximity():
    N=160
    states=np.zeros((21,N+1))
    states[0]=np.linspace(0,20,N+1)
    states[2,65:85]=np.linspace(0,1.5,20)
    states[2,85:]=1.5
    states[3]=4.
    class Grid:
        def footprint_clearance(self,x,y,psi,margin):
            return .003 if 8.<x<12. else .08
    r=ranked_sectors(states,Grid(),sectors=8)
    assert len(r)==8
    assert r[0]['sector'] in (3,4)
    assert r[0]['peak_lateral_accel_mps2']>0


def test_adaptive_scheduler_shrinks_rejects_exploits_certified_and_explores():
    ranked=[dict(sector=i,score=10-i) for i in range(8)]
    first=choose_adaptive_region(ranked,[])
    assert first['sector']==0 and first['source']=='explore_ranked_region'
    failed=[dict(strategy='adaptive',status='rejected_initializer',
                 adaptive_spec=first)]
    shrunk=choose_adaptive_region(ranked,failed)
    assert shrunk['sector']==0 and shrunk['amplitude_rad']<first['amplitude_rad']
    failed.append(dict(strategy='adaptive',status='no_validated_witness',
                       adaptive_spec=shrunk))
    second=choose_adaptive_region(ranked,failed)
    assert second['sector']!=0
    winner=dict(strategy='adaptive',independent_status='source_witness_passed',
                accepted=True,certified_witness='certified.npz',
                adaptive_spec=second)
    exploit=choose_adaptive_region(ranked,failed+[winner])
    assert exploit['source']=='exploit_certified'
    assert exploit['seed']=='certified.npz'
    assert exploit['amplitude_rad']>second['amplitude_rad']
    assert exploit['center_shift_sectors']==pytest.approx(.25)
    first_profile=localized_steering_profile(284,sector=second['sector'],
        sectors=8,amplitude_rad=.0005)
    shifted_profile=localized_steering_profile(284,sector=second['sector'],
        sectors=8,amplitude_rad=.0005,
        center_shift_sectors=exploit['center_shift_sectors'])
    assert np.linalg.norm(first_profile-shifted_profile)>0.001
