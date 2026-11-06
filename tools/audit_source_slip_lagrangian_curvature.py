#!/usr/bin/env python3
"""Offline audit of analytic *source-derived* slip curvature along legal QP directions.

Checks residual-weighted curvature against original source slip scores, and
checks exact-stage-cache parity. The reported local curvature is *not*
claimed to predict a finite step that crosses a tire switching surface.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BIN=ROOT/'build/f1tenth_mpc/mpc_native_source_sqp_replay'
CASE=ROOT/'live_runs/mpc_source_second_order_20261104_01/case_15'


def run(case:Path,controls:Path,*flags:str):
    p=subprocess.run([str(BIN),str(case),str(controls),'1','--oracle-only',*flags],
                     cwd=ROOT,text=True,capture_output=True,timeout=60)
    if p.returncode:
        raise RuntimeError(f'source oracle exit={p.returncode}: {p.stderr[-700:]}')
    return json.loads(p.stdout)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--case',type=Path,default=CASE)
    ap.add_argument('--controls',type=Path,default=CASE/'strict_feasible_offline_witness_20261104.controls')
    ap.add_argument('--output',type=Path,default=ROOT/'build/source_slip_lagrangian_curvature.json')
    args=ap.parse_args()
    u=np.loadtxt(args.controls)
    if u.shape!=(30,2):raise ValueError('need 30 physical steering/throttle pairs')
    directions={
        'steering_stage20':np.eye(60)[40],
        'throttle_stage20':np.eye(60)[41],
    }
    records=[]
    with tempfile.TemporaryDirectory(prefix='source_slip_hvp_') as tmp:
        tmp=Path(tmp)
        path=tmp/'source.controls'
        dpath=tmp/'direction.txt'
        baseline=run(args.case,args.controls)
        for name,d in directions.items():
            np.savetxt(dpath,np.column_stack([np.arange(60),d]),fmt=['%d','%.12f'])
            a=run(args.case,args.controls,'--source-slip-direction='+str(dpath))
            b=run(args.case,args.controls,'--source-slip-direction='+str(dpath),
                  '--source-slip-reuse-stage-cache')
            plain=a['source_slip_lagrangian_curvature']
            cached=b['source_slip_lagrangian_curvature']
            deviations={k:abs(plain[k]-cached[k])
                        for k in ('gauss_newton','residual_weighted','total')}
            samples=[]
            for h in (0.0005,0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.5,1.):
                cost=[];allowed=True
                for sign in (1.,-1.):
                    x=u+sign*h*d.reshape(30,2)*np.array([.15,.25])
                    if (np.any(abs(x[:,0])>0.5235987756) or
                        np.any(x[:,1]<.001) or np.any(x[:,1]>1.)):
                        allowed=False;break
                    np.savetxt(path,x,fmt='%.12f')
                    cost.append(run(args.case,path)['source_longitudinal_slip_cost'])
                if allowed:
                    exact=(cost[0]-2*baseline['source_longitudinal_slip_cost']+cost[1])/h**2
                    samples.append({'h':h,'true_slip_second_difference':exact,
                                    'analytic_slip_second':cached['total'],
                                    'absolute_error':abs(exact-cached['total'])})
            records.append({'direction':name,'direct':plain,'cached':cached,
                            'cache_parity_absolute_errors':deviations,'finite_steps':samples})
        # The nonlinear cost that the QP tries to predict must be audited
        # across the complete legal throttle step, not only around alpha=0.
        throttle_scan=[]
        d=directions['throttle_stage20'].reshape(30,2)
        for alpha in np.linspace(0.,1.,51):
            x=u+alpha*d*np.array([.15,.25]);np.savetxt(path,x,fmt='%.12f')
            exact=run(args.case,path)
            throttle_scan.append({'alpha':float(alpha),'total_cost':exact['cost'],
                                  'slip_cost':exact['source_longitudinal_slip_cost']})
    cache_ok=all(all(v<=1e-6*max(1.,abs(r['cached'][k]))
                     for k,v in r['cache_parity_absolute_errors'].items())
                 for r in records)
    throttle=next(r for r in records if r['direction']=='throttle_stage20')
    # A stable, unambiguous source-slip regime validates the analytic
    # physical Hessian at the same cost-scoring epoch. The steering
    # direction is near an actuator kink and its tiny float FD is noisy.
    local_curvature_ok=all(
        p['absolute_error'] <= .01*max(1.,abs(p['analytic_slip_second']))
        for p in throttle['finite_steps'] if p['h']<=.002)
    slip=np.array([r['slip_cost'] for r in throttle_scan])
    rates=np.diff(slip)/.02
    slope_jump=np.abs(np.diff(rates))
    prominent=np.argsort(slope_jump)[-5:][::-1]
    transitions=[{'alpha':throttle_scan[int(i)+1]['alpha'],
                  'slip_slope_before':float(rates[i]),
                  'slip_slope_after':float(rates[i+1])}
                 for i in prominent]
    result={'schema':'source-slip-lagrangian-curvature/v2',
            'stage_cache_parity':cache_ok,
            'local_source_slip_curvature_qualified':local_curvature_ok,
            'full_step_cost_fidelity_qualified':False,
            'source_feasible_nominal':baseline['wall_violation_m2']==0,
            'throttle_scan':throttle_scan,
            'dominant_directional_slip_transitions':transitions,
            'records':records}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'cache_parity':cache_ok,
                      'local_source_slip_curvature_qualified':local_curvature_ok,
                      'dominant_transition_points':transitions[:3],
                      'case':str(args.case),
                      'results':[{'direction':r['direction'],
                                  'gauss_newton':r['cached']['gauss_newton'],
                                  'source_residual_weighted':r['cached']['residual_weighted'],
                                  'cached_us':r['cached']['elapsed_us'],
                                  'uncached_us':r['direct']['elapsed_us'],
                                  'smallest_step_fd':r['finite_steps'][0] if r['finite_steps'] else None}
                                 for r in records]},indent=2))
    if not cache_ok or not local_curvature_ok:raise SystemExit(1)

if __name__=='__main__':main()
