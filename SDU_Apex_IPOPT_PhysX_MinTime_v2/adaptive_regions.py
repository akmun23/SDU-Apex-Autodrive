"""Checkpointable, deterministic adaptive corner exploration for source IPOPT.

This selects UNVALIDATED initializations only. A source-model NLP, exact 1ms
perimeter verifier and independent certification still control promotion.
No global optimality, robustness or Unity drivability is inferred.
"""
from __future__ import annotations

import math
import numpy as np


def cyclic_offset(values, center):
    return (np.asarray(values, dtype=float)-center+.5) % 1. - .5


def localized_steering_profile(count, *, sector, sectors, amplitude_rad,
                               width_fraction=.045, polarity=1, center_shift_sectors=0.):
    """Zero-net-mean periodic steer-and-countersteer to localize pose impact."""
    if (not isinstance(count,int) or count<30 or not isinstance(sectors,int)
            or not 2<=sectors<=32 or not 0<=sector<sectors
            or not np.isfinite(amplitude_rad) or not 0.<amplitude_rad<=.008
            or not np.isfinite(width_fraction) or
            not .01<=width_fraction<=.12 or polarity not in (-1,1)
            or not np.isfinite(center_shift_sectors)
            or abs(center_shift_sectors)>.5):
        raise ValueError('invalid adaptive sector profile')
    t=(np.arange(count,dtype=float)+.5)/count
    center=(sector+.5+center_shift_sectors)/sectors
    earlier=np.exp(-.5*(cyclic_offset(t,center)/width_fraction)**2)
    later=np.exp(-.5*(cyclic_offset(t,center+1.8*width_fraction)/width_fraction)**2)
    profile=earlier-later
    profile-=profile.mean()
    profile*=polarity*amplitude_rad/max(1.e-12,np.max(np.abs(profile)))
    return profile


def ranked_sectors(states, grid, *, sectors=8):
    """Prioritize high lateral acceleration and small source-node clearance."""
    state=np.asarray(states,dtype=float)
    if state.ndim!=2 or state.shape[0]<6 or state.shape[1]<32:
        raise ValueError('invalid source state')
    if not isinstance(sectors,int) or not 2<=sectors<=32:
        raise ValueError('invalid sectors')
    count=state.shape[1]-1
    delta=np.diff(state[:2,:],axis=1)
    ds=np.maximum(np.linalg.norm(delta,axis=0),.01)
    curvature=np.abs(np.diff(np.unwrap(state[2])))/ds
    speed=np.maximum(.25,np.hypot(state[3,:-1],state[4,:-1]))
    pressure=curvature*speed*speed
    slack=np.array([grid.footprint_clearance(
        float(state[0,k]),float(state[1,k]),float(state[2,k]),margin=.05)
        for k in range(count)],dtype=float)
    if not (np.isfinite(pressure).all() and np.isfinite(slack).all()):
        raise ValueError('invalid path or map clearance')
    # Per-sector pressure is a dimensionless ranking, not a dynamics constraint.
    curvature_scale=max(.001,float(np.quantile(pressure,.85)))
    clearance_scale=max(.002,float(np.quantile(np.maximum(0.,slack),.65)))
    values=[]
    for sector in range(sectors):
        left=(sector*count)//sectors
        right=((sector+1)*count)//sectors
        p=float(np.quantile(pressure[left:right],.8))/curvature_scale
        c=float(np.quantile(np.maximum(0.,slack[left:right]),.2))
        score=.65*p+.35/(1.+c/clearance_scale)
        values.append(dict(sector=sector,score=score,minimum_node_slack_m=float(
            slack[left:right].min()),peak_lateral_accel_mps2=float(
                pressure[left:right].max())))
    return sorted(values,key=lambda v:(-v['score'],v['sector']))


def choose_adaptive_region(hotspots, attempts, *, default_amplitude=.0005):
    """Choose next sector using durable feedback from prior adaptive attempts.

    A successful, independently certified branch is exploited once; an
    out-of-track initializer shrinks before a second attempt. Otherwise
    unexplored sectors receive preference over repeating failed sectors.
    """
    if not hotspots or not 0<default_amplitude<=.004:
        raise ValueError('invalid adaptive region search')
    sectors={int(x['sector']):x for x in hotspots}
    history=[a for a in attempts if a.get('strategy')=='adaptive'
             and a.get('adaptive_spec')]
    if history:
        previous=history[-1]
        prior=previous['adaptive_spec']
        region=int(prior['sector'])
        same=[a for a in history if int(a['adaptive_spec']['sector'])==region]
        if (previous.get('independent_status')=='source_witness_passed'
                and previous.get('accepted') and
                len(same)<3 and previous.get('certified_witness')):
            shifts=(.25,-.25)
            return dict(sector=region,sectors=len(sectors),
                amplitude_rad=min(.004,float(prior['amplitude_rad'])*1.35),
                polarity=int(prior['polarity']),
                center_shift_sectors=shifts[(len(same)-1)%2],
                seed=previous['certified_witness'],
                source='exploit_certified',
                hotspot_score=sectors[region]['score'])
        if (previous.get('status')=='rejected_initializer'
                and len(same)<2):
            return dict(sector=region,sectors=len(sectors),
                amplitude_rad=max(.00008,float(prior['amplitude_rad'])*.35),
                polarity=int(prior['polarity']),
                center_shift_sectors=float(prior.get('center_shift_sectors',0.)),
                source='shrink_infeasible',
                hotspot_score=sectors[region]['score'])
    counts={region:0 for region in sectors}
    for a in history:
        i=int(a['adaptive_spec']['sector'])
        counts[i]=counts.get(i,0)+1
    candidate=max(hotspots,key=lambda x:
        (float(x['score'])/(1.+2.*counts[int(x['sector'])]),
         -counts[int(x['sector'])],-int(x['sector'])))
    region=int(candidate['sector'])
    return dict(sector=region,sectors=len(sectors),
                amplitude_rad=default_amplitude*(.8**counts[region]),
                polarity=1 if counts[region]%2==0 else -1,
                center_shift_sectors=0.,
                source='explore_ranked_region',
                hotspot_score=candidate['score'])
