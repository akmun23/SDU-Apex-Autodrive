#!/usr/bin/env python3
"""Exact-source 60-control SECOND-ORDER condensed SQP (offline-only).

No racing, no fallback, no model surrogate. Original source-model
dynamics are eliminated by exact full-state, same-epoch single shooting.
Every local QP uses exact-source original left/right footprint clearances
and their central FD derivatives, original actuator/rack limits and a
true 60x60 source-cost Hessian from central FD of source adjoints.
For infeasible nominals, the strictly hierarchical phase-I LP finds
minimum wall slack; *it never lets racing cost buy wall penetration*.
All accepted SQP iterates are independently repropagated and strict
source-feasibility is not confused with local affine QP feasibility.
"""
from __future__ import annotations
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.optimize import linprog

N=30
D=60
SCALE=np.tile(np.array([.15,.25]),N)
WALL_BUFFER_M=.00025  # original wall tightening, not relaxation
ALPHAS=(1.,.5,.25,.125,.0625,.03125,.015625,.0078125,
        .00390625,.001953125,.0009765625)

class Source:
    def __init__(self,case,binary,folder):
        self.case=case
        self.binary=binary
        self.path=folder/'source_controls.txt'
        self.calls=0
        self.maximum_steering=.5235987756

    def evaluate(self,z,gradient=True,derivative_only=False):
        # Standalone oracle supports no score-only protocol; retain its
        # validated original behavior for parity experiments.
        cmd=z.reshape(N,2)
        np.savetxt(self.path,cmd,fmt='%.12f')
        process=subprocess.run([str(self.binary),str(self.case),str(self.path)],
             capture_output=True,text=True,timeout=25)
        if process.returncode:
            raise RuntimeError('source step failed: '+process.stderr[-350:])
        r=json.loads(process.stdout)
        self.calls+=1
        self.maximum_steering=r['max_steering_rad']
        left=np.asarray(r['left_clearance_m'],dtype=float)
        right=np.asarray(r['right_clearance_m'],dtype=float)
        assert len(left)==len(right)==31
        if min(left[0],right[0])<-1e-10:
            raise RuntimeError('measured stage0 already outside original physical wall')
        wall=np.stack([left[1:],right[1:]],axis=1).ravel()
        rack=np.asarray(r['rack_steering_rad'])[:N]
        previous_t=np.asarray(r['previous_throttle_command'])[:N]
        steering_rate=min(float(r['max_steering_command_step_rad']),
                          .995*3.2*float(r['prediction_dt']))
        throttle_rate=float(r['max_throttle_command_step'])
        ds=cmd[:,0]-rack
        dt=cmd[:,1]-previous_t
        rate=np.column_stack([
             steering_rate-ds,steering_rate+ds,
             throttle_rate-dt,throttle_rate+dt]).ravel()
        gradient=np.asarray(r['source_gradient_flat'],dtype=float)
        if not r.get('source_gradient_valid') or len(gradient)!=D:
            raise RuntimeError('source objective adjoint not valid')
        return dict(J=float(r['objective']), wall=wall,rate=rate,
                    gradient=gradient,source_valid=bool(r['original_valid']),
                    failure_stage=r['failure_stage'],
                    wall_v=float(np.minimum(wall,0)@np.minimum(wall,0)),
                    rate_v=float(np.minimum(rate,0)@np.minimum(rate,0)))

def derivatives(source,z,base,h):
    y=z/SCALE
    wall_J=np.empty((60,D));rate_J=np.empty((120,D))
    numeric_grad=np.empty(D);raw_hessian=np.empty((D,D))
    for j in range(D):
        dy=np.zeros(D);dy[j]=h
        plus=source.evaluate((y+dy)*SCALE)
        minus=source.evaluate((y-dy)*SCALE)
        wall_J[:,j]=(plus['wall']-minus['wall'])/(2*h)
        rate_J[:,j]=(plus['rate']-minus['rate'])/(2*h)
        numeric_grad[j]=(plus['J']-minus['J'])/(2*h)
        raw_hessian[:,j]=((plus['gradient']-minus['gradient'])*SCALE)/(2*h)
    raw_hessian=.5*(raw_hessian+raw_hessian.T)
    eigvals,eigvec=np.linalg.eigh(raw_hessian)
    # Convexified full source reduced Hessian, NOT a steering/throttle
    # heuristic. Record all negative curvature removed for audit.
    hessian=(eigvec*np.maximum(eigvals,1e-4))@eigvec.T
    gradient_relative=float(np.linalg.norm(numeric_grad-
            base['gradient']*SCALE)/max(1.,np.linalg.norm(numeric_grad)))
    info=dict(gradient_adjoint_vs_fd_relative_error=gradient_relative,
              minimum_actual_source_hessian_eigen=float(eigvals.min()),
              number_negative_source_hessian_eigen=int((eigvals<0).sum()),
              removed_negative_curvature_norm=float(
                  np.linalg.norm(np.minimum(0,eigvals))))
    return numeric_grad,wall_J,rate_J,hessian,info

def solve_subproblem(y,base,g,Jw,Jr,H,maxsteer,osqp):
    low=np.tile(np.array([-maxsteer,.001]),N)/SCALE-y
    high=np.tile(np.array([maxsteer,1.]),N)/SCALE-y
    m=60
    I=sparse.eye(D,format='csc')
    jw=sparse.csc_matrix(-Jw)
    jr=sparse.csc_matrix(-Jr)
    target_margins=base['wall']-WALL_BUFFER_M
    phase1=bool(np.any(target_margins < -1.e-12) or
                base['rate_v']>1e-10 or not base['source_valid'])
    min_slack=0.
    if phase1:
        # Physically impossible affine wall constraints are represented as
        # explicit nonnegative slacks in PHASE I only. Objective is pure
        # feasibility. All rack/actuator constraints remain hard.
        Z=sparse.csc_matrix((len(base['rate']),m))
        A=sparse.vstack([sparse.hstack([jw,-sparse.eye(m)]),
                         sparse.hstack([jr,Z])],format='csr')
        bounds=[(float(a),float(b)) for a,b in zip(low,high)]+[(0,None)]*m
        LP=linprog(np.r_[np.zeros(D),np.ones(m)],
                   A_ub=A,b_ub=np.r_[target_margins,base['rate']],
                   bounds=bounds,method='highs',
                   options={'primal_feasibility_tolerance':1e-8,
                            'dual_feasibility_tolerance':1e-8})
        if not LP.success:
            return None,dict(status='linearized_phase1_infeasible',
                             message=LP.message)
        min_slack=max(float(LP.fun),0.)
    elastic=phase1 and min_slack>2e-7
    if elastic:
        Z=sparse.csc_matrix((len(base['rate']),m))
        A=sparse.vstack([
            sparse.hstack([jw,-sparse.eye(m)]),
            sparse.hstack([jr,Z]),
            sparse.hstack([I,sparse.csc_matrix((D,m))]),
            sparse.hstack([sparse.csc_matrix((m,D)),sparse.eye(m)]),
            sparse.csc_matrix(np.r_[np.zeros(D),np.ones(m)][None,:])
        ],format='csc')
        lower=np.r_[np.full(m,-np.inf),np.full(len(base['rate']),-np.inf),
                    low,np.zeros(m),-np.inf]
        upper=np.r_[target_margins,base['rate'],high,
                    np.full(m,np.inf),min_slack+max(1e-7,1e-5*min_slack)]
        P=sparse.block_diag((I,1e-8*sparse.eye(m)),format='csc')
        q=np.zeros(D+m)
    else:
        A=sparse.vstack([jw,jr,I],format='csc')
        lower=np.r_[np.full(m,-np.inf),np.full(len(base['rate']),-np.inf),low]
        upper=np.r_[target_margins,base['rate'],high]
        # Exact-source feasibility BEFORE racing cost. If current source
        # trajectory is infeasible but affine problem has zero wall slack,
        # minimize control correction rather than the racing objective.
        P=I if phase1 else sparse.csc_matrix(H)
        q=np.zeros(D) if phase1 else g
    solver=osqp.OSQP()
    solver.setup(P=sparse.triu(P,format='csc'),q=q,A=A,l=lower,u=upper,
                 eps_abs=1e-6,eps_rel=1e-6,max_iter=90000,
                 polish=True,adaptive_rho=True,verbose=False,
                 scaled_termination=False)
    solution=solver.solve(raise_error=False)
    report={'status':solution.info.status,
            'iterations':int(solution.info.iter),
            'minimum_affine_wall_slack_sum_m':min_slack,
            'phase1':phase1,'slack_variables':elastic,
            'primal_residual':float(solution.info.prim_res),
            'dual_residual':float(solution.info.dual_res)}
    if solution.info.status!='solved' or solution.x is None:
        return None,report
    residual=A@solution.x
    mask=np.isfinite(upper)
    maxupper=float(np.max(np.maximum(0.,residual[mask]-upper[mask])))
    masklow=np.isfinite(lower)
    maxlower=float(np.max(np.maximum(0.,lower[masklow]-residual[masklow])))
    report['maximum_affine_constraint_violation']=max(maxupper,maxlower)
    if report['maximum_affine_constraint_violation']>1e-5:
        report['status']='uncertified'
        return None,report
    return solution.x[:D],report

def source_exact_second_order_constraint_correction(
        source,z,dz,Jwall,Jrate,osqp):
    """Standard SQP SOC in 60D reduced control space.

    The first-order QP predicts wall margins c0+J*d at a proposed full step,
    whereas full nonlinear source gives c_exact(y+d). Its defect
    r=c_exact-(c0+J*d) is the omitted finite-step nonlinear contribution,
    including steering, wheel slip and path-progress motion. Solve a
    minimal-norm correction delta with
        c_exact(y+d) + J_nom * delta >= WALL_BUFFER_M
    and likewise for the exact actuator-rate constraints. Strictly enforce
    original physical command bounds; reject if the corrected exact source
    trajectory does not improve the nonlinear merit. This cannot certify
    physical feasibility without independent exact-source validation.
    """
    trial=(z/SCALE+dz)*SCALE
    source_full=source.evaluate(trial)
    if (source_full['source_valid'] and
        np.all(source_full['wall']>=WALL_BUFFER_M)):
        return None,{'soc_required':False}
    y=trial/SCALE
    low=np.tile(np.array([-source.maximum_steering,.001]),N)/SCALE-y
    high=np.tile(np.array([source.maximum_steering,1.]),N)/SCALE-y
    A=sparse.vstack([sparse.csc_matrix(-Jwall),
                     sparse.csc_matrix(-Jrate),
                     sparse.eye(D,format='csc')],format='csc')
    lb=np.r_[np.full(60,-np.inf),np.full(120,-np.inf),low]
    ub=np.r_[source_full['wall']-WALL_BUFFER_M,source_full['rate'],high]
    solver=osqp.OSQP()
    solver.setup(P=sparse.eye(D,format='csc'),q=np.zeros(D),
                 A=A,l=lb,u=ub,max_iter=90000,
                 eps_abs=1e-6,eps_rel=1e-6,
                 polish=True,verbose=False)
    corrected=solver.solve(raise_error=False)
    result={'soc_required':True,
            'soc_solver_status':corrected.info.status,
            'soc_qp_iterations':int(corrected.info.iter),
            'soc_nominal_wall_violation_m2':source_full['wall_v']}
    if corrected.info.status!='solved' or corrected.x is None:
        return None,result
    dx=np.asarray(corrected.x,dtype=float)
    cert=np.asarray(A@dx)
    if np.max(cert[np.isfinite(ub)]-ub[np.isfinite(ub)])>1e-5:
        result['soc_certified']=False
        return None,result
    result['soc_certified']=True
    result['soc_correction_norm']=float(np.linalg.norm(dx))
    return trial+dx*SCALE,result


def run(source,seed,osqp,iterations,h,workspace,use_soc=False,
        curvature_gate=False,derivative_provider=None,
        early_full_step_feasible=False,first_accepted_globalization=False):
    z=seed.ravel().astype(float)
    report=[]
    for iteration in range(iterations):
        try:
            base=source.evaluate(z,gradient=True)
            grad,Jwall,Jrate,H,deriv=(
                (derivative_provider or derivatives)(source,z,base,h))
            dz,qp=solve_subproblem(z/SCALE,base,grad,Jwall,Jrate,
                                   H,source.maximum_steering,osqp)
            row={'iteration':iteration,'cost':base['J'],
                 'wall_V':base['wall_v'],'source_valid':base['source_valid'],
                 'source_first_failure':base['failure_stage'],
                 'source_evaluations':source.calls,**deriv,**qp}
            if dz is None:
                row['reason']='no_numerically_certified_second_order_QP'
                report.append(row);print(json.dumps(row),flush=True);break
            # Validate the *actual proposed direction* against exact source
            # cost at two small finite steps. A local source Hessian can be
            # strongly indefinite/non-smooth; merely projecting it onto the
            # PSD cone may greatly distort its actual physical curvature.
            # A convex directional secant fit selects how much of that
            # second-order model belongs in the current QP. Only the source
            # model (not a generic trust radius) selects this coefficient.
            if curvature_gate and base['source_valid']:
                b=(.025,.05)
                gdot=float(grad@dz)
                hdot=float(dz@H@dz)
                observed=[];term=[]
                for beta in b:
                    sample=source.evaluate((z/SCALE+beta*dz)*SCALE)
                    observed.append(sample['J']-base['J']-beta*gdot)
                    term.append(.5*beta*beta*hdot)
                if sum(x*x for x in term)>1e-12:
                    theta=float(np.clip(np.dot(term,observed)/
                                        np.dot(term,term),0.,1.))
                else:
                    theta=0.
                row['second_order_source_directional_calibration']=theta
                row['source_cost_small_step_linear_error_sum']=float(
                    sum(abs(x) for x in observed))
                row['source_cost_small_step_quadratic_error_sum']=float(
                    sum(abs(x-y) for x,y in zip(observed,term)))
                if theta<.9999:
                    Hnew=theta*H+(1.-theta)*1e-4*np.eye(D)
                    update,check=solve_subproblem(z/SCALE,base,
                        grad,Jwall,Jrate,Hnew,
                        source.maximum_steering,osqp)
                    if update is not None:
                        dz=update
                        H=Hnew
                        row['second_order_recalibration_resolved']=True
                        row['second_order_recalibration_qp_iterations']=check['iterations']
                    else:
                        row['second_order_recalibration_resolved']=False
            row['full_step_predicted_cost_delta']=float(
                grad@dz+.5*dz@H@dz)
            candidates=[]
            for alpha in ALPHAS:
                trial=(z/SCALE+alpha*dz)*SCALE
                measured=source.evaluate(trial,gradient=False)
                if alpha==1.:
                    local_wall=base['wall']+Jwall@dz
                    local_rate=base['rate']+Jrate@dz
                    row['full_step_max_source_wall_defect_m']=float(
                        np.max(np.abs(local_wall-measured['wall'])))
                    row['full_step_max_source_rate_defect']=float(
                        np.max(np.abs(local_rate-measured['rate'])))
                    row['full_step_actual_original_wall_V']=measured['wall_v']
                    row['full_step_local_wall_V']=float(
                        np.square(np.minimum(0.,local_wall)).sum())
                if base['source_valid'] and base['wall_v']<=1e-12:
                    accept=(measured['source_valid'] and
                            measured['wall_v']<=1e-12 and
                            measured['rate_v']<=1e-10 and
                            measured['J']<base['J']-1e-6)
                else:
                    # A signed margin of -1e-8 metres is not source-valid
                    # despite a visually zero squared-violation score.
                    # Tightening QP walls avoids float32 boundary artifacts.
                    olddef=np.maximum(0.,WALL_BUFFER_M-base['wall'])
                    newdef=np.maximum(0.,WALL_BUFFER_M-measured['wall'])
                    old_v=float(olddef@olddef)
                    new_v=float(newdef@newdef)
                    accept=(measured['source_valid'] and
                            measured['rate_v']<=1.e-10) or (
                            new_v<old_v-max(1.e-11,old_v*1.e-5)
                            and measured['rate_v']<=base['rate_v']+1e-8)
                candidates.append((accept,trial,alpha,measured))
                if first_accepted_globalization and accept:
                    # Standard monotone exact-source filter: take the first
                    # sufficiently improving nonlinear step. A quadratic
                    # accepted full step is never required to be physically
                    # feasible if it is only an intermediate SQP iterate;
                    # publication remains forbidden until source-valid.
                    # Full-step SOC is still compared to this trial below.
                    row['monotone_first_accepted_alpha']=alpha
                    break
                if (early_full_step_feasible and alpha==1. and
                    not base['source_valid'] and measured['source_valid']
                    and measured['rate_v']<=1e-10):
                    # Phase-I is lexicographic feasibility, not racing cost.
                    # This full step is already strictly source feasible, so
                    # smaller alpha trials cannot improve the primary goal.
                    row['line_search_early_feasible']=True
                    break
            if use_soc:
                try:
                    soc_z,soc_info=source_exact_second_order_constraint_correction(
                        source,z,dz,Jwall,Jrate,osqp)
                    row.update(soc_info)
                    if soc_z is not None:
                        actual=source.evaluate(soc_z,gradient=False)
                        olddef=np.maximum(0.,WALL_BUFFER_M-base['wall'])
                        newdef=np.maximum(0.,WALL_BUFFER_M-actual['wall'])
                        old_v=float(olddef@olddef)
                        new_v=float(newdef@newdef)
                        okay=(actual['source_valid'] and
                              actual['rate_v']<=1e-10) or (
                              new_v<old_v-max(1e-11,old_v*1e-5)
                              and actual['rate_v']<=base['rate_v']+1e-8)
                        if base['source_valid'] and base['wall_v']<=1e-12:
                            okay=(actual['source_valid'] and
                                  actual['wall_v']<=1e-12 and
                                  actual['rate_v']<=1e-10 and
                                  actual['J']<base['J']-1e-6)
                        row['soc_exact_source_violation_m2']=actual['wall_v']
                        row['soc_exact_source_valid']=actual['source_valid']
                        candidates.append((okay,soc_z,1.,actual))
                except RuntimeError as error:
                    row['soc_error']=str(error)
            legal=[c for c in candidates if c[0]]
            if not legal:
                row['reason']='no_exact_source_feasibility_or_cost_progress'
                row['best_trial_V']=min(c[3]['wall_v'] for c in candidates)
                report.append(row);print(json.dumps(row),flush=True);break
            chosen=min(legal,key=lambda c:
                 c[3]['J'] if base['source_valid'] else
                 (float(np.square(np.maximum(0.,WALL_BUFFER_M-
                                             c[3]['wall'])).sum()),
                  c[3]['rate_v']))
            z=chosen[1]
            np.savetxt(workspace/f'accepted_{iteration:02d}.txt',
                       z.reshape(N,2),fmt='%.12f')
            accepted_linear_cost=float(chosen[2]*grad@dz)
            accepted_second_order_cost=float(
                accepted_linear_cost+.5*chosen[2]**2*dz@H@dz)
            row.update(accepted_alpha=chosen[2],
                       next_cost=chosen[3]['J'],
                       accepted_linear_cost_prediction=accepted_linear_cost,
                       accepted_second_order_cost_prediction=accepted_second_order_cost,
                       accepted_linear_cost_error=abs(
                           chosen[3]['J']-base['J']-accepted_linear_cost),
                       accepted_second_order_cost_error=abs(
                           chosen[3]['J']-base['J']-accepted_second_order_cost),
                       actual_cost_delta=chosen[3]['J']-base['J'],
                       next_wall_V=chosen[3]['wall_v'],
                       source_exact_feasible=chosen[3]['source_valid'],
                       cost_prediction_error_at_step=float(
                         chosen[3]['J']-base['J']-
                         (chosen[2]*grad@dz+
                          .5*chosen[2]**2*dz@H@dz)))
            report.append(row);print(json.dumps(row),flush=True)
        except (ValueError,RuntimeError,AssertionError) as e:
            row={'iteration':iteration,'error':str(e)}
            report.append(row);print(json.dumps(row),flush=True);break
    (workspace/'results.json').write_text(json.dumps(report,indent=2))
    return report

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('case',type=Path)
    ap.add_argument('seed',type=Path)
    ap.add_argument('--iterations',type=int,default=8)
    ap.add_argument('--h',type=float,default=.002)
    ap.add_argument('--source-soc',action='store_true',
                    help='enable exact-source second-order constraint correction')
    ap.add_argument('--source-curvature-gate',action='store_true',
                    help='calibrate PSD Hessian on two source-exact directional cost probes')
    ap.add_argument('--workspace',type=Path,default=Path('/tmp/mpc_second_order_source_sqp'))
    ap.add_argument('--source-binary',type=Path,default=Path('build/f1tenth_mpc/mpc_offline_source_merit'))
    ap.add_argument('--osqp-vendor',type=Path,default=Path('/tmp/sdu-mpc-osqp-v104'))
    args=ap.parse_args()
    if not 1<=args.iterations<=30:ap.error('invalid SQP iteration count')
    args.workspace.mkdir(parents=True,exist_ok=True)
    # Import SciPy above the OSQP vendor path; its old bundled SciPy
    # conflicts with system NumPy 1.26 otherwise.
    sys.path.insert(0,str(args.osqp_vendor))
    import osqp
    return int(not bool(run(Source(args.case,args.source_binary,args.workspace),
                      np.loadtxt(args.seed),osqp,args.iterations,args.h,
                      args.workspace,args.source_soc,args.source_curvature_gate)))

if __name__=='__main__':
    raise SystemExit(main())
