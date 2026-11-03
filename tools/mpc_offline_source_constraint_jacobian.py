#!/usr/bin/env python3
"""Source-tangent reduced control-space footprint Jacobian for offline SQP.

Derives d(g_source)/d(absolute command) from the *same actual source
nominal* and 30-stage source sensitivity A/B already in MPC. No numerical
finite difference of the physical tire dynamics.

The stage footprint is piecewise smooth. Pick the active physical support
plane on the left/right at each exact-source nominal; linearize the same
progress-dependent corridor as the controller's hard polygon.

This is NOT itself an exact nonlinear finite-step constraint. Each SQP step
must still be checked through the full physical source model.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.mpc_qp_oracle import decode_problem
from tools.audit_mpc_condensed_qp import condense

NX=23
H=30
EY=0
EPS=1
RACK=5
PROGRESS=20
PREVIOUS_THROTTLE=22


def source_tangent_constraint_jacobian(
    fixture:Path,absolute_controls:np.ndarray,source_report:dict,
    full_rate:bool=False,with_qp_hessian:bool=False,
):
    p=decode_problem(fixture.read_bytes())
    data=condense(p,hard_walls=True)
    T=data['T']
    c=data['c']
    command=np.asarray(absolute_controls).reshape(60)
    G=np.eye(60)
    offset=np.zeros(60)
    for k in range(H):
        base=k*NX
        G[2*k]+=T[base+RACK,:]
        G[2*k+1]+=T[base+PREVIOUS_THROTTLE,:]
        offset[2*k]=c[base+RACK]
        offset[2*k+1]=c[base+PREVIOUS_THROTTLE]
    nominal_u=np.linalg.solve(G,command-offset)
    nominal_x=c+T@nominal_u
    physical=np.asarray(nominal_x[:(H+1)*NX]).reshape(H+1,NX)
    D=np.zeros((60+(120 if full_rate else 60),60))
    nominal_plane_slack=np.zeros(60)
    for k in range(1,H+1):
        planes=p['state_polygons'][k]
        for side in range(2):
            # The source-geometry API returns left/slack from the negative
            # e_y face, and right/slack from the positive e_y face.
            # Do NOT infer physical side from the word left alone.
            target_sign=-1 if side==0 else 1
            active=[]
            for plane in planes:
                a,b,bound,progress=plane
                if a*target_sign<0.5:continue
                slack=bound-a*physical[k,EY]-b*physical[k,EPS]-progress*physical[k,PROGRESS]
                active.append((slack,plane))
            if not active:
                raise ValueError(f"no hard footprint plane at k={k},side={side}")
            slack,(a,b,bound,progress)=min(active,key=lambda t:t[0])
            r=(k-1)*2+side
            nominal_plane_slack[r]=slack
            D[r]=-(a*T[k*NX+EY]+b*T[k*NX+EPS]+
                   progress*T[k*NX+PROGRESS])
    for k in range(H):
        rack_d=T[k*NX+RACK,:]
        command_r=np.zeros(60)
        command_r[2*k]=1.
        delta=command_r-rack_d
        if full_rate:
            # In QP increment coordinates, d(command - rack) / d(u_QP)
            # is simply the selector for this stage's steering increment;
            # do NOT subtract the rack sensitivity twice.
            # Exact same sequence as source_merit.cpp:
            # [steer_step - (command-rack), steer_step + (command-rack),
            #  throttle_step - (command-prev_throttle),
            #  throttle_step + (command-prev_throttle)]
            D[60+4*k,2*k]=-1.
            D[60+4*k+1,2*k]=1.
            D[60+4*k+2,2*k+1]=-1.
            D[60+4*k+3,2*k+1]=1.
        else:
            D[60+2*k]=-delta
            D[60+2*k+1]=delta
    # d(u_QP)/d(commands) = G^-1 ; do not explicitly invert G.
    jac=np.linalg.solve(G.T,D.T).T
    physical_slack=np.c_[source_report['left_clearance_m'][1:],
                         source_report['right_clearance_m'][1:]].ravel()
    mismatch=nominal_plane_slack-physical_slack
    if with_qp_hessian:
        # The source-linearized QP provides an inexpensive positive-
        # semidefinite Gauss-Newton approximation in command coordinates.
        # QP input = G^{-1} (absolute_command - offset).
        inv_G=np.linalg.solve(G,np.eye(60))
        hessian=inv_G.T@data['P'].toarray()@inv_G
        hessian=.5*(hessian+hessian.T)
    diagnostic={
        'max_at_nominal_hard_geometry_difference_m':float(np.max(np.abs(mismatch))),
        'worst_stage_side':list(map(int,np.unravel_index(
                np.argmax(np.abs(mismatch)),(H,2)))),
        'max_nominal_qp_rack_difference_rad':float(np.max(np.abs(
            physical[:-1,RACK]-np.asarray(source_report['rack_steering_rad'])[:-1]))),
        'max_command_reconstruction_delta':float(np.max(np.abs(
            G@nominal_u+offset-command))),
    }
    if with_qp_hessian:
        return jac,diagnostic,hessian
    return jac,diagnostic
