#!/usr/bin/env python3
"""Independent interior-point comparison of the frozen original source NLP.

CasADi supplies the IPOPT interface only. All physical dynamics, costs,
gradients and inequalities come from the native source oracle. Offline only.
"""
import argparse
from collections import OrderedDict
import json
from pathlib import Path
import subprocess
import time

import casadi as ca
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SCALE=np.tile([.15,.25],30)
OBJECTIVE_SCALE=10000.


def compare(case,seed,output,iterations,oracle=None):
    output.mkdir(parents=True,exist_ok=True)
    process=subprocess.Popen([str(oracle or ROOT/'build/f1tenth_mpc/mpc_source_nlp_oracle'),str(case)],
        stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
    cache=OrderedDict();calls=0
    def evaluate(x):
        nonlocal calls
        x=np.asarray(x).reshape(60)
        key=x.tobytes()
        if key in cache:return cache[key]
        process.stdin.write(json.dumps({'controls':(x*SCALE).reshape(30,2).tolist(),
            'preferred_clearance_m':float(np.float32(.03))})+'\n');process.stdin.flush()
        data=json.loads(process.stdout.readline());calls+=1
        if not data['valid']:raise RuntimeError(data)
        cache[key]=data
        if len(cache)>128:cache.popitem(last=False)
        return data
    try:
        x0=np.loadtxt(seed).reshape(60)/SCALE
        initial=evaluate(x0)
        chosen=np.array([t=='body' or (t=='rate' and not initial['rate_constraints_redundant'])
                         for t in initial['row_types']])
        m=int(chosen.sum())
        class SourceJacobian(ca.Callback):
            def __init__(self,name):
                ca.Callback.__init__(self);self.construct(name)
            def get_n_in(self):return 2
            def get_n_out(self):return 1
            def get_sparsity_in(self,i):return ca.Sparsity.dense(60 if i==0 else m+1,1)
            def get_sparsity_out(self,i):return ca.Sparsity.dense(m+1,60)
            def eval(self,args):
                d=evaluate(args[0])
                return [ca.DM(np.vstack([np.asarray(d['gradient_scaled'])/OBJECTIVE_SCALE,
                    np.asarray(d['hard_jacobian_scaled'])[chosen]]))]
        class Source(ca.Callback):
            def __init__(self):
                ca.Callback.__init__(self);self.derivatives=[];self.construct('native_source')
            def get_n_in(self):return 1
            def get_n_out(self):return 1
            def get_sparsity_in(self,i):return ca.Sparsity.dense(60,1)
            def get_sparsity_out(self,i):return ca.Sparsity.dense(m+1,1)
            def eval(self,args):
                d=evaluate(args[0])
                return [ca.DM(np.r_[d['objective']/OBJECTIVE_SCALE,np.asarray(d['hard_margins'])[chosen]])]
            def has_jacobian(self):return True
            def get_jacobian(self,name,inames,onames,opts):
                derivative=SourceJacobian(name);self.derivatives.append(derivative);return derivative
        source=Source();x=ca.MX.sym('u',60);value=source(x)
        solver=ca.nlpsol('source_ipopt','ipopt',{'x':x,'f':value[0],'g':value[1:]},
            {'print_time':False,'error_on_fail':False,'ipopt.print_level':0,
             'ipopt.hessian_approximation':'limited-memory','ipopt.max_iter':iterations,
             'ipopt.nlp_scaling_method':'none','ipopt.mu_strategy':'adaptive',
             'ipopt.bound_relax_factor':0.,'ipopt.tol':1.e-7,
             'ipopt.constr_viol_tol':1.e-8,'ipopt.acceptable_iter':0})
        lower=np.tile([-initial['max_steering_rad'],float(np.float32(.001))],30)/SCALE
        upper=np.tile([initial['max_steering_rad'],1.],30)/SCALE
        start=time.perf_counter()
        result=solver(x0=x0,lbx=lower,ubx=upper,lbg=np.zeros(m),ubg=np.full(m,np.inf))
        wall_ms=(time.perf_counter()-start)*1000.
        final_x=np.asarray(result['x']).reshape(60)
        final=evaluate(final_x)
        lam_g=np.asarray(result['lam_g']).reshape(m)*OBJECTIVE_SCALE
        lam_x=np.asarray(result['lam_x']).reshape(60)*OBJECTIVE_SCALE
        A=np.asarray(final['hard_jacobian_scaled'])[chosen]
        margins=np.asarray(final['hard_margins'])
        stationarity=(np.asarray(final['gradient_scaled'])+A.T@lam_g+lam_x)/SCALE
        bound_margin=np.where(lam_x<0,final_x-lower,upper-final_x)
        stats=solver.stats()
        report={'schema':'source-ipopt-independent-comparison/v1','offline_only':True,
            'same_original_source_nlp':True,'casadi_version':ca.__version__,
            'status':stats['return_status'],'success':bool(stats['success']),
            'iterations':int(stats['iter_count']),'oracle_calls':calls,'wall_ms':wall_ms,
            'initial_cost':initial['objective'],'final_cost':final['objective'],
            'final_source_racing_cost':final['original_cost'],
            'minimum_clearance_m':final['minimum_clearance_m'],
            'source_feasible':bool(final['minimum_clearance_m']>=0. and margins.min()>=-1.e-7),
            'original_nlp_stationarity_inf':float(np.max(abs(stationarity))),
            'original_nlp_primal_inf':float(max(0.,-margins.min())),
            'original_nlp_complementarity_inf':float(max(
                np.max(abs(margins[chosen]*lam_g)),np.max(abs(bound_margin*lam_x)))),
            'original_nlp_dual_feasibility_inf':float(max(0.,lam_g.max())),
            'solver_history':stats.get('iterations',{})}
        np.savetxt(output/'ipopt.controls',(final_x*SCALE).reshape(30,2),fmt='%.10g')
        (output/'ipopt.json').write_text(json.dumps(report,indent=2)+'\n')
        return report
    finally:
        process.stdin.close();process.wait(timeout=10)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('case',type=Path);p.add_argument('--seed',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--iterations',type=int,default=300)
    p.add_argument('--oracle',type=Path)
    a=p.parse_args();r=compare(a.case,a.seed or a.case/'initial.controls',a.output,a.iterations,a.oracle)
    print(json.dumps({k:v for k,v in r.items() if k!='solver_history'},indent=2))
