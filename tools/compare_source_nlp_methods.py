#!/usr/bin/env python3
"""Offline comparison on one frozen source NLP; never a controller fallback.

Native SQP and independent SLSQP/BFGS consume the same source objective,
preferred clearance cost, exact source gradients and original inequalities.
No finite-difference tire model, alternate reference or restrictive rack box.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from scipy.optimize import Bounds, minimize, nnls

ROOT=Path(__file__).resolve().parents[1]
SCALE=np.tile([.15,.25],30)


def compare(case: Path, seed: Path, output: Path, iterations: int):
    output.mkdir(parents=True,exist_ok=True)
    process=subprocess.Popen([str(ROOT/'build/f1tenth_mpc/mpc_source_nlp_oracle'),str(case)],
        stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
    cache={};calls=0;history=[]
    reserve=float(np.float32(.03))
    def evaluate(x):
        nonlocal calls
        key=np.asarray(x).tobytes()
        if key in cache:return cache[key]
        request={'controls':(np.asarray(x)*SCALE).reshape(30,2).tolist(),
                 'preferred_clearance_m':reserve,'preferred_clearance_weight':1.e6}
        process.stdin.write(json.dumps(request)+'\n');process.stdin.flush()
        line=process.stdout.readline()
        if not line:raise RuntimeError('native source oracle ended')
        data=json.loads(line);calls+=1
        if not data['valid']:raise RuntimeError(f'original source oracle rejected input: {data}')
        cache[key]=data
        return data
    try:
        x0=np.loadtxt(seed).reshape(60)/SCALE
        initial=evaluate(x0)
        max_steer=initial['max_steering_rad']
        lower=np.tile([-max_steer,float(np.float32(.001))],30)/SCALE
        upper=np.tile([max_steer,1.],30)/SCALE
        # Absolute command boxes are already represented by Bounds. The
        # canonical full-domain rate inequalities are provably redundant;
        # retain rates when the recorded problem has restrictive limits.
        chosen=np.array([t=='body' or (t=='rate' and not initial['rate_constraints_redundant'])
                         for t in initial['row_types']])
        def fun(x):return evaluate(x)['objective']/10000.
        def jac(x):return np.asarray(evaluate(x)['gradient_scaled'])/10000.
        def constraint(x):return np.asarray(evaluate(x)['hard_margins'])[chosen]
        def constraint_jac(x):return np.asarray(evaluate(x)['hard_jacobian_scaled'])[chosen]
        def monitor(x):
            data=evaluate(x)
            history.append({'iteration':len(history)+1,'objective':data['objective'],
                'minimum_clearance_m':data['minimum_clearance_m'],'oracle_calls':calls})
        start=time.perf_counter()
        result=minimize(fun,x0,jac=jac,method='SLSQP',bounds=Bounds(lower,upper),
            constraints=[{'type':'ineq','fun':constraint,'jac':constraint_jac}],
            callback=monitor,options={'maxiter':iterations,'ftol':1.e-10})
        elapsed=(time.perf_counter()-start)*1000.
        final=evaluate(result.x)
        hard=np.asarray(final['hard_margins']);A=np.asarray(final['hard_jacobian_scaled'])
        g=np.asarray(final['gradient_scaled'])
        active=hard<=1.e-5
        multipliers=np.zeros(len(hard))
        if active.any():
            multipliers[active],_=nnls(A[active].T,g,maxiter=10000)
        stationarity=(g-A.T@multipliers)/SCALE
        report={'schema':'source-nlp-independent-comparison/v1',
            'method':'SLSQP with BFGS Lagrangian Hessian; offline only',
            'scipy_success':bool(result.success),'message':str(result.message),
            'iterations':int(result.nit),'oracle_calls':calls,'wall_ms':elapsed,
            'initial_cost':initial['objective'],'final_cost':final['objective'],
            'final_source_racing_cost':final['original_cost'],
            'minimum_clearance_m':final['minimum_clearance_m'],
            'source_feasible':bool(hard.min()>=-1e-7 and final['minimum_clearance_m']>=0.),
            'original_nlp_stationarity_inf':float(np.max(abs(stationarity))),
            'original_nlp_primal_inf':float(max(0.,-hard.min())),
            'original_nlp_complementarity_inf':float(np.max(abs(hard*multipliers))),
            'dual_estimate':'nonnegative least squares over source-active rows; not SLSQP internal duals',
            'history':history}
        np.savetxt(output/'slsqp.controls',(result.x*SCALE).reshape(30,2),fmt='%.10g')
        (output/'slsqp.json').write_text(json.dumps(report,indent=2)+'\n')
        env={k:v for k,v in os.environ.items() if not k.startswith('MPC_SOURCE_NLP_')}
        env.update(MPC_SOURCE_NLP_QP_REGULARIZATION='1000',MPC_SOURCE_NLP_MIN_CLEARANCE_M=str(reserve))
        native=subprocess.run([str(ROOT/'build/f1tenth_mpc/mpc_source_exact_offline_optimize'),
            str(case),str(seed),str(min(100,iterations)),str(output/'native.controls')],
            check=True,capture_output=True,text=True,env=env,timeout=60)
        (output/'native.json').write_text(native.stdout)
        return report
    finally:
        process.stdin.close();process.wait(timeout=10)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case',type=Path)
    parser.add_argument('--seed',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--iterations',type=int,default=100)
    args=parser.parse_args()
    result=compare(args.case,args.seed or args.case/'initial.controls',args.output,args.iterations)
    print(json.dumps({k:v for k,v in result.items() if k!='history'},indent=2))


if __name__=='__main__':main()
