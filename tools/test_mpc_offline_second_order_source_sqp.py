#!/usr/bin/env python3
"""Mathematical regression tests for source-exact condensed second-order SQP.

These tests are purposely independent of Unity race/ROS callbacks.
They certify QP source-derivative reconstruction, actual Hessian, hard-wall
feasibility priority and strict original-wall tightening.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.mpc_offline_second_order_source_sqp import (
    D,N,SCALE,WALL_BUFFER_M,derivatives,solve_subproblem)

class QuadraticOracle:
    def __init__(self):
        rng=np.random.default_rng(20261103)
        self.H=np.diag(np.linspace(1,3,D))+np.outer(
            np.sin(np.arange(D)),np.sin(np.arange(D)))*.007
        self.g=rng.normal(size=D)
        self.A=rng.normal(size=(60,D))*.01
        self.R=rng.normal(size=(120,D))*.03
        self.maxsteer=.52359879016876
        self.calls=0
    def evaluate(self,z):
        self.calls+=1
        y=z/SCALE
        x=y.copy()
        return {'J':float(.5*x@self.H@x+self.g@x),
            'wall':.4+self.A@x,
            'rate':1.+self.R@x,
            'gradient':(self.H@x+self.g)/SCALE,
            'wall_v':float(np.square(np.minimum(0.,.4+self.A@x)).sum()),
            'rate_v':float(np.square(np.minimum(0.,1.+self.R@x)).sum()),
            'source_valid':True,'failure_stage':-1}

def run():
    # SciPy is imported by the subject module before the old OSQP wheel.
    sys.path.insert(0,'/tmp/sdu-mpc-osqp-v104')
    import osqp
    oracle=QuadraticOracle()
    z=np.zeros(D)
    base=oracle.evaluate(z)
    g,Jw,Jr,H,info=derivatives(oracle,z,base,h=.002)
    assert oracle.calls==121,oracle.calls
    assert max(abs(g-oracle.g))<1e-8
    assert max(abs(Jw-oracle.A).ravel())<1e-8
    assert max(abs(Jr-oracle.R).ravel())<1e-8
    assert max(abs(H-oracle.H).ravel())<1e-7
    assert info['gradient_adjoint_vs_fd_relative_error']<1e-8
    assert info['number_negative_source_hessian_eigen']==0

    zero=np.zeros(D)
    walls=np.ones(60)
    rates=np.ones(120)
    jwall=np.zeros((60,D))
    jrates=np.zeros((120,D))
    jwall[0,0]=1.
    invalid=dict(wall=walls.copy(),wall_v=.25,
                 rate=rates,rate_v=0.,source_valid=False)
    invalid['wall'][0]=-.5
    dx,report=solve_subproblem(z,invalid,zero,jwall,jrates,
                             np.eye(D),.52359879016876,osqp)
    assert report['status']=='solved',report
    assert not report['slack_variables']
    assert abs(dx[0]-(.5+WALL_BUFFER_M))<1e-5,dx[0]
    assert abs(dx[2::2]).max()<1e-5
    # Physical throttle lower bound is .001; normalized control increment
    # therefore must be at least .001/.25 even in minimum-norm Phase-I.
    assert max(abs(dx[1::2]-.001/.25))<1e-5
    assert report['minimum_affine_wall_slack_sum_m']<=1e-7
    print('SOURCE_QP_PHASE1_HARD_WALL_PRIORITY_PASS')

    # A source-feasible nominal has negative cost gradient pushing into
    # the physical wall. Even a huge favorable cost reduction CANNOT buy
    # an original wall violation.
    feasible=dict(wall=walls.copy(),wall_v=0.,rate=rates,
                  rate_v=0.,source_valid=True)
    feasible['wall'][0]=.01
    jwall[0,0]=-1.
    gradient=np.zeros(D);gradient[0]=-1.e4
    dx,report=solve_subproblem(z,feasible,gradient,jwall,jrates,
                             np.eye(D),.52359879016876,osqp)
    assert report['status']=='solved',report
    assert not report['slack_variables']
    assert dx[0]<=.01-WALL_BUFFER_M+1e-5,dx[0]
    print('SOURCE_QP_STRICT_COST_VS_WALL_PRIORITY_PASS')

    # Physical wall violated by <1e-7m still invalid despite vanishing
    # squared violation; use a 0.25mm inward safety buffer in QP only.
    tiny=dict(wall=walls.copy(),wall_v=1e-16,rate=rates,
              rate_v=0.,source_valid=False)
    tiny['wall'][0]=-1e-8
    jwall[0,0]=1.
    dx,report=solve_subproblem(z,tiny,zero,jwall,jrates,
                             np.eye(D),.52359879016876,osqp)
    assert report['status']=='solved',report
    assert dx[0]>=WALL_BUFFER_M+1e-8-1e-5,dx[0]
    print('SOURCE_QP_FLOAT32_CLEARANCE_ROUNDING_GUARD_PASS')

if __name__=='__main__':
    run()
