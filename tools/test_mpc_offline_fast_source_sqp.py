#!/usr/bin/env python3
"""Regression gate for optimized exact-source SQP, no Unity/races.

Run on the frozen high-curvature failure corpus. Verify that:
- persistent source evaluator G/Q/D modes retain EXACT cost and wall/rate
  values, while the derivative-only mode makes no strict-feasibility claim;
- persistent source QP matches the one-shot binary fixture byte-for-byte;
- original source-footprint geometry and complete physical steering/throttle
  rate Jacobians are numerically consistent;
- the cheap parallel-FD Phase-I QP reproduces the original full-source
  two-iteration success, without ever accepting a QP-only feasible plan.

The source exact Hessian was never used on those infeasible nominals;
avoiding it is a mathematical equivalence for Phase-I, not a heuristic.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.mpc_offline_fast_source_sqp import PersistentSource
from tools.mpc_offline_source_constraint_jacobian import source_tangent_constraint_jacobian

CASE=ROOT/"live_runs/comp_gradient_nearwall_capture_20261102_01/qp_captures/epoch_1793692917333601071_case_6"
SOURCE=Path("/tmp/mpc_source_merit_persistent")
QP=Path("/tmp/mpc_offline_source_qp_server")
TARGET_COST=4478.462727498


def run_tests(case:Path,source_binary:Path,qp_binary:Path,worker_count:int):
    if not (case/"source_context.mctx").exists():
        raise RuntimeError("missing real frozen failure context "+str(case))
    for binary in (source_binary,qp_binary):
        if not binary.is_file():
            raise RuntimeError(f"missing compiled C++ source oracle {binary}")

    with tempfile.TemporaryDirectory(prefix="mpc_fast_source_regression_") as tmp:
        folder=Path(tmp)
        seed_qp=folder/"seed.fqp"
        run=subprocess.run([str(qp_binary),str(case),"-",str(seed_qp),"mode26"],
                           capture_output=True,text=True,timeout=20)
        assert run.returncode==0,run.stderr
        seed=Path(str(seed_qp)+".seed.txt")
        commands=np.loadtxt(seed).reshape(30,2)

        source=PersistentSource(case,source_binary,folder,True)
        try:
            full=source.evaluate(commands.ravel(),gradient=True)
            strict=source.evaluate(commands.ravel(),gradient=False)
            derivative=source.evaluate(commands.ravel(),gradient=False,
                                       derivative_only=True)
            assert strict['strict_checked'] and not derivative['strict_checked']
            for a,b in ((strict,derivative),(full,strict)):
                assert a['J']==b['J'],(a['J'],b['J'])
                np.testing.assert_array_equal(a['wall'],b['wall'])
                np.testing.assert_array_equal(a['rate'],b['rate'])
            assert full['gradient'].shape==(60,)
            assert full['wall_v']>0.
            print("PERSISTENT_EXACT_SOURCE_DERIVATIVE_PROTOCOL_PARITY_PASS")

            # Nominal full-source model and frozen hard QP must agree exactly
            # on physical state/geometry at the current nominal.
            source_report=dict(left_clearance_m=np.r_[0.,full['wall'][0::2]],
                right_clearance_m=np.r_[0.,full['wall'][1::2]],
                rack_steering_rad=full['rack_steering_rad'])
            jac,diag,_=source_tangent_constraint_jacobian(
                seed_qp,commands,source_report,full_rate=True,with_qp_hessian=True)
            assert jac.shape==(180,60)
            assert diag['max_at_nominal_hard_geometry_difference_m']<3e-5,diag
            assert diag['max_nominal_qp_rack_difference_rad']<2e-5,diag
            for k,a in ((0,0),(2,1),(12,0)):
                trial=commands.copy();trial[k,a]+=5e-4
                new=source.evaluate(trial.ravel(),gradient=False,
                                    derivative_only=True)
                measured=(new['rate']-full['rate'])/5e-4
                assert np.max(abs(measured-jac[60:,2*k+a]))<2e-4
            print("SOURCE_FULL_RATE_AND_FOOTPRINT_TANGENT_PARITY_PASS")
        finally:
            source.close()

        # Captured QP is byte-for-byte identical between one-shot and
        # persistent evaluation from the same frozen command epoch.
        second=folder/"persistent.fqp"
        server=subprocess.Popen([str(qp_binary),str(case),"--serve","mode26"],
              stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
        try:
            server.stdin.write(str(second)+" "+" ".join(
                f"{v:.12f}" for v in commands.ravel())+"\n")
            server.stdin.flush()
            response=json.loads(server.stdout.readline())
            assert response['qp_exported']
        finally:
            server.stdin.close()
            assert server.wait(timeout=5)==0
        h1=hashlib.sha256(seed_qp.read_bytes()).hexdigest()
        h2=hashlib.sha256(second.read_bytes()).hexdigest()
        assert h1==h2,(h1,h2)
        print("FULL_SOURCE_RELINEARIZATION_QP_BINARY_PARITY_PASS")

        output=folder/"two_iterations"
        proc=subprocess.run([sys.executable,
                  str(ROOT/"tools/mpc_offline_fast_source_sqp.py"),
                  str(case),str(seed),"--iterations","2","--source-soc",
                  "--constraint-jacobian","parallel_fd","--workers",str(worker_count),
                  "--workspace",str(output),
                  "--source-binary",str(source_binary),
                  "--qp-binary",str(qp_binary)],
                  text=True,capture_output=True,timeout=45)
        assert proc.returncode==0,proc.stderr[-700:]
        rows=[json.loads(x) for x in proc.stdout.splitlines()
              if x.startswith('{')]
        phases=[r for r in rows if 'iteration' in r]
        assert len(phases)==2,phases
        assert all(r['status']=='solved' for r in phases)
        assert all(r['hessian_method']=='minimum_norm_phase1' for r in phases)
        assert all(r['full_source_perturbations']==120 for r in phases)
        assert not phases[0]['source_exact_feasible']
        assert phases[1]['source_exact_feasible']
        assert phases[1]['next_wall_V']==0.
        assert abs(phases[1]['next_cost']-TARGET_COST)<.01,phases[1]
        speed=rows[-1]['performance']
        assert speed['first_feasible_iteration']==2
        assert speed['final_source_feasible']
        assert speed['source_perturbations']==240
        assert speed['final_original_wall_violation_m2']==0.
        assert abs(speed['final_source_cost']-TARGET_COST)<.01
        print("TWO_SQP_ITERATIONS_SOURCE_EXACT_FEASIBILITY_PARITY_PASS")
        print(json.dumps({'elapsed_ms':speed['total_elapsed_ms'],
                          'source_control_perturbations':speed['source_perturbations'],
                          'final_cost':phases[-1]['next_cost'],
                          'strict_source_feasible':True}))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case",type=Path,default=CASE)
    parser.add_argument("--source-binary",type=Path,default=SOURCE)
    parser.add_argument("--qp-binary",type=Path,default=QP)
    parser.add_argument("--workers",type=int,default=8)
    args=parser.parse_args()
    run_tests(args.case,args.source_binary,args.qp_binary,args.workers)
