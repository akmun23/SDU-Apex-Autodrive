#!/usr/bin/env python3
"""Compare normal SQP, source-informed curvature and terminal continuation.

Offline saved physical states only; no live ADMM memory or publication proof.
Costs with terminal viability include an additional frozen 75ms suffix and
must not be compared directly with the original 750ms objective.
"""
from __future__ import annotations
import argparse
import csv
import os
import statistics
import subprocess
from pathlib import Path
from tools.run_mpc_source_accuracy_corpus import REPLAY_ARGS, TRAJECTORY, CORPUS, REPO


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=REPO/'live_runs/mpc_normal_sqp_unification_20261101')
    parser.add_argument('--binary',type=Path,default=REPO/'build/f1tenth_mpc/mpc_bag_solver_replay')
    parser.add_argument('--baseline',type=Path)
    parser.add_argument('--repeats',type=int,default=1)
    parser.add_argument('--cycle-budget-us',type=float,default=0.)
    a=parser.parse_args()
    if a.repeats<1 or a.cycle_budget_us<0:parser.error('invalid repeat count or budget')
    a.output.mkdir(parents=True,exist_ok=True)
    cases={name:(CORPUS/f'practice_case_{i}.csv',None) for name,i in
           [('corner23',0),('corner39',4),('corner50',7)]}
    for name in ['corner66','corner293','corner303fail']:
        folder=REPO/'live_runs/mpc_phase16_qp_convergence_cases_20261101'/name
        cases[name]=(folder/'physical_state.csv',folder/'nominal_source_seed.csv')
    modes=['normal_r2','source_r2','terminal_r2','both_r2']
    if a.baseline:modes.insert(0,'baseline_r2')
    rows=[]
    for name,(state,seed) in cases.items():
        for mode in modes:
            for repeat in range(a.repeats):
                env=os.environ.copy()
                for flag in ['MPC_REPLAY_SOURCE_INFORMED_SQP','MPC_REPLAY_TERMINAL_VIABILITY',
                             'MPC_REPLAY_FEASIBILITY_FIRST','MPC_REPLAY_NOMINAL_SEED_CSV',
                             'MPC_REPLAY_CYCLE_BUDGET_US']:
                    env.pop(flag,None)
                env.update(MPC_REPLAY_COMMAND_INCREMENT_QP='1',MPC_REPLAY_FEEDBACK_GAIN='.25')
                if seed:env['MPC_REPLAY_NOMINAL_SEED_CSV']=str(seed)
                if mode in ['source_r2','both_r2']:env['MPC_REPLAY_SOURCE_INFORMED_SQP']='1'
                if mode in ['terminal_r2','both_r2']:env['MPC_REPLAY_TERMINAL_VIABILITY']='1'
                if a.cycle_budget_us:env['MPC_REPLAY_CYCLE_BUDGET_US']=str(a.cycle_budget_us)
                args=list(REPLAY_ARGS);args[43-3]='2'
                binary=a.baseline if mode=='baseline_r2' else a.binary
                result=subprocess.run([str(binary.resolve()),str(TRAJECTORY),str(state),*args],
                    cwd=REPO,env=env,text=True,capture_output=True,timeout=120)
                stem=f'{name}_{mode}_{repeat}'
                (a.output/f'{stem}.csv').write_text(result.stdout)
                (a.output/f'{stem}.log').write_text(result.stderr)
                data=list(csv.DictReader(result.stdout.splitlines()))
                if result.returncode or len(data)!=1:
                    raise RuntimeError(f'{stem}: replay failed ({result.returncode}): {result.stderr[-500:]}')
                value=data[0]
                keys=['status','admm_status','iterations','primal','dual',
                      'sqp_iterations_completed','sqp_iterations_accepted',
                      'nonlinear_objective','line_search_nominal_objective',
                      'first_pass_objective','second_pass_objective','exact_objective_improvement',
                      'source_curvature_steering','source_curvature_throttle',
                      'cycle_us','min_corridor_slack_m','line_search_selected_alpha']
                row={'case':name,'mode':mode,'repeat':repeat,
                     **{k:value[k] for k in keys if k in value}}
                rows.append(row)
            sample=[r for r in rows if r['case']==name and r['mode']==mode]
            costs=[float(r['nonlinear_objective']) for r in sample]
            print(name,mode,'cost',statistics.median(costs),'passes',sample[-1]['sqp_iterations_completed'],flush=True)
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with (a.output/'corner_comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
