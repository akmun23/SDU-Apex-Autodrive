#!/usr/bin/env python3
"""Fast candidate for exact-source 60-input SQP, OFFLINE and NOT deployable.

Replaces 120 source perturbations/iteration with the existing source-model
A/B Jacobians and same-epoch QP geometry. In source-infeasible Phase-I the
QP objective is minimum feasible correction: the entire source-cost Hessian
is mathematically unnecessary. Once source-feasible, use a source-linearized
Gauss-Newton cost Hessian updated by damped BFGS on exact source adjoint
gradients from accepted steps. Every proposed command and correction is
checked through the original nonlinear source model, never an approximate
or wall-relaxed surrogate.

The C++ evaluator optionally remains open, avoiding repeated source-context
parsing and binary process launches. This tool records exact source and QP
agreement, convergence, call counts and elapsed time on frozen corner cases.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.mpc_offline_source_constraint_jacobian import (
    source_tangent_constraint_jacobian)
from tools.mpc_offline_second_order_source_sqp import (
    D,N,SCALE,Source,run)


class PersistentSource(Source):
    def __init__(self,case,binary,folder,persistent=True):
        super().__init__(case,binary,folder)
        self.persistent=persistent
        self.server=None
        self.cache={}
        self.cache_hits=0
        self.server_roundtrip_ms=[]
        if persistent:
            self.server=subprocess.Popen(
                [str(binary),str(case),"--serve"],
                stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,bufsize=1,text=True)

    def evaluate(self,z,gradient=True,derivative_only=False):
        key=(np.asarray(z,dtype=np.float64).reshape(-1).tobytes(),
             bool(derivative_only))
        if key in self.cache and (not gradient or
            self.cache[key].get('gradient') is not None):
            self.cache_hits+=1
            return self.cache[key]
        if self.server is None:
            result=super().evaluate(z,gradient=gradient,
                                    derivative_only=derivative_only)
        else:
            if self.server.poll() is not None:
                raise RuntimeError("persistent source process terminated")
            start=time.perf_counter()
            self.server.stdin.write(
                ("D" if derivative_only else "G" if gradient else "Q")+" "+" ".join(
                f"{v:.12f}" for v in np.asarray(z).reshape(-1))+"\n")
            self.server.stdin.flush()
            line=self.server.stdout.readline()
            self.server_roundtrip_ms.append(1000*(time.perf_counter()-start))
            if not line:
                raise RuntimeError("persistent source no reply")
            r=json.loads(line)
            self.calls+=1
            self.maximum_steering=r['max_steering_rad']
            left=np.asarray(r['left_clearance_m'],dtype=float)
            right=np.asarray(r['right_clearance_m'],dtype=float)
            if left.shape!=(31,) or right.shape!=(31,):
                raise ValueError("source corridor must have 31 states")
            if min(left[0],right[0])<-1e-10:
                raise RuntimeError("measured stage0 already outside original wall")
            cmd=np.asarray(z).reshape(N,2)
            rack=np.asarray(r['rack_steering_rad'])[:N]
            previous_t=np.asarray(r['previous_throttle_command'])[:N]
            steer_step=min(float(r['max_steering_command_step_rad']),
                           .995*3.2*float(r['prediction_dt']))
            dt=cmd[:,1]-previous_t
            ds=cmd[:,0]-rack
            rate=np.column_stack([
                steer_step-ds,steer_step+ds,
                float(r['max_throttle_command_step'])-dt,
                float(r['max_throttle_command_step'])+dt]).ravel()
            source_gradient=(np.asarray(r['source_gradient_flat'],dtype=float)
                if gradient else None)
            if gradient and (not r.get('source_gradient_valid') or
                source_gradient.shape!=(D,)):
                raise RuntimeError("exact source adjoint unavailable")
            wall=np.stack([left[1:],right[1:]],axis=1).ravel()
            result=dict(J=float(r['objective']),wall=wall,rate=rate,
                gradient=source_gradient,source_valid=bool(r['original_valid']),
                rack_steering_rad=np.asarray(r['rack_steering_rad'],dtype=float),
                previous_throttle_command=np.asarray(
                    r['previous_throttle_command'],dtype=float),
                strict_checked=bool(r.get('strict_checked',True)),
                failure_stage=int(r['failure_stage']),
                wall_v=float(np.minimum(wall,0)@np.minimum(wall,0)),
                rate_v=float(np.minimum(rate,0)@np.minimum(rate,0)))
        if len(self.cache)>=1024:
            self.cache.clear()
        self.cache[key]=result
        return result

    def close(self):
        if self.server is None:return
        try:
            self.server.stdin.close()
            self.server.wait(timeout=3)
        except (OSError,subprocess.TimeoutExpired):
            self.server.terminate()
            self.server.wait(timeout=3)
        self.server=None


class TangentBfgsDerivatives:
    def __init__(self,case,relinearize_binary,folder,persistent=True):
        self.case=case
        self.relinearize_binary=relinearize_binary
        self.folder=folder
        self.server=(subprocess.Popen([
            str(relinearize_binary),str(case),"--serve","mode26"],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,bufsize=1,text=True)
            if persistent else None)
        self.calls=0
        self.gn_hessian=None
        self.previous_y=None
        self.previous_gradient=None
        self.bfgs_hessian=None
        self.bfgs_updates=0
        self.time_ms=[]

    def __call__(self,source,z,base,h_unused):
        started=time.perf_counter()
        self.calls+=1
        filename=self.folder/f"tangent_{self.calls:02d}.txt"
        fixture=self.folder/f"tangent_{self.calls:02d}.fqp"
        if self.server is None:
            np.savetxt(filename,z.reshape(N,2),fmt="%.12f")
            p=subprocess.run([str(self.relinearize_binary),str(self.case),
                      str(filename),str(fixture),"mode26"],
                      capture_output=True,text=True,timeout=20)
            if p.returncode:
                raise RuntimeError("source sensitivity QP rebuild failed: "+
                                   p.stderr[-450:])
        else:
            if self.server.poll() is not None:
                raise RuntimeError("persistent QP source process terminated")
            self.server.stdin.write(str(fixture)+" "+" ".join(
                f"{v:.12f}" for v in z.reshape(-1))+"\n")
            self.server.stdin.flush()
            msg=self.server.stdout.readline()
            if not msg:
                raise RuntimeError("persistent QP source emitted no reply")
            if not json.loads(msg).get('qp_exported'):
                raise RuntimeError("persistent QP source build not complete")
        try:
            jac,geom,gn=source_tangent_constraint_jacobian(
                fixture,z.reshape(N,2),
                {
                    "left_clearance_m":np.r_[0.,base['wall'][0::2]],
                    "right_clearance_m":np.r_[0.,base['wall'][1::2]],
                    "rack_steering_rad":base['rack_steering_rad'],
                },
                full_rate=True,with_qp_hessian=True)
        except Exception as e:
            raise RuntimeError("source tangent reduction failed: "+str(e))
        if jac.shape!=(180,D):
            raise AssertionError("all 60 source-wall and 120 actuator-rate rows required")
        if geom['max_at_nominal_hard_geometry_difference_m']>3e-5:
            raise AssertionError("source wall/reference geometry mismatch")
        if geom['max_nominal_qp_rack_difference_rad']>2e-5:
            raise AssertionError("source QP rack command state mismatch")
        grad=base['gradient']*SCALE
        y=z/SCALE
        feasible=(base['source_valid'] and base['wall_v']<=1e-12
                  and base['rate_v']<=1e-10)
        # True feasible-start reduced SQP requires positive-definite model.
        # Warm-start PSD cost curvature from the source-linearized racing
        # objective, not arbitrary steering/throttle penalties.
        if feasible:
            if self.bfgs_hessian is None:
                hessian=(gn*SCALE[:,None])*SCALE[None,:]
                hessian=.5*(hessian+hessian.T)
                eig,evec=np.linalg.eigh(hessian)
                hessian=(evec*np.maximum(eig,1e-4))@evec.T
            else:
                hessian=self.bfgs_hessian.copy()
                step=y-self.previous_y
                delta=grad-self.previous_gradient
                if np.linalg.norm(step)>1e-9:
                    Bs=hessian@step
                    sBs=float(step@Bs)
                    sy=float(step@delta)
                    if sBs>1e-10 and np.isfinite(sy):
                        theta=1. if sy>=.2*sBs else .8*sBs/(sBs-sy)
                        delta=theta*delta+(1.-theta)*Bs
                        sdelta=float(step@delta)
                        if sdelta>1e-10:
                            hessian-=np.outer(Bs,Bs)/sBs
                            hessian+=np.outer(delta,delta)/sdelta
                            hessian=.5*(hessian+hessian.T)
                            self.bfgs_updates+=1
            self.bfgs_hessian=hessian.copy()
            self.previous_y=y.copy()
            self.previous_gradient=grad.copy()
        else:
            # Infeasible-start SQP minimizes a *feasibility* QP. Its
            # quadratic term is I regardless of the actual racing Hessian.
            # Computing any 60x60 source Hessian here is wasted work.
            hessian=np.eye(D)
            self.bfgs_hessian=None
            self.previous_y=None
            self.previous_gradient=None
        elapsed=(time.perf_counter()-started)*1000
        self.time_ms.append(elapsed)
        info=dict(derivative_method="exact_source_tangent",
                  hessian_method="minimum_norm_phase1" if not feasible
                  else ("source_GN" if self.bfgs_updates==0 else "damped_BFGS"),
                  hessian_source_perturbations=0,
                  source_geometry_nominal_abs_error_m=
                    geom['max_at_nominal_hard_geometry_difference_m'],
                  bfgs_updates=self.bfgs_updates,
                  tangent_build_ms=elapsed)
        return grad,jac[:60],jac[60:],hessian,info

    def close(self):
        if self.server is None:return
        try:
            self.server.stdin.close()
            self.server.wait(timeout=3)
        except (OSError,subprocess.TimeoutExpired):
            self.server.terminate()
            self.server.wait(timeout=3)
        self.server=None


class ParallelFiniteDifferenceConstraints:
    """Full source wall/rate Jacobian, no unnecessary Hessian calls in Phase I.

    Each of the N workers owns a distinct process containing a frozen
    loaded model. This avoids thread-unsafe shared source-physics state,
    makes 120 EXACT source evaluations parallel, and computes no source
    adjoint on the perturbed line-search-like probes.

    This is the fidelity reference to qualify the cheaper tangent
    Jacobian. It is still much more expensive than future native C++
    source sensitivities.
    """
    def __init__(self,case,binary,folder,workers,fast_gn_provider):
        self.workers=[PersistentSource(case,binary,folder,True)
                      for _ in range(workers)]
        self.pool=ThreadPoolExecutor(max_workers=workers)
        self.fast_gn_provider=fast_gn_provider
        self.calls=0
        self.full_source_perturbations=0
        self.time_ms=[]

    def __call__(self,source,z,base,h):
        started=time.perf_counter()
        y=z/SCALE
        def on_worker(wid):
            results=[]
            worker=self.workers[wid]
            for j in range(wid,D,len(self.workers)):
                delta=np.zeros(D)
                delta[j]=h
                plus=worker.evaluate((y+delta)*SCALE,gradient=False,
                                     derivative_only=True)
                minus=worker.evaluate((y-delta)*SCALE,gradient=False,
                                      derivative_only=True)
                results.append((j,plus,minus))
            return results
        results=[]
        for part in self.pool.map(on_worker,range(len(self.workers))):
            results.extend(part)
        jac_wall=np.empty((60,D),dtype=float)
        jac_rate=np.empty((120,D),dtype=float)
        for j,plus,minus in results:
            jac_wall[:,j]=(plus['wall']-minus['wall'])/(2*h)
            jac_rate[:,j]=(plus['rate']-minus['rate'])/(2*h)
        self.full_source_perturbations+=2*D
        self.calls+=1
        base_gradient=base['gradient']*SCALE
        if (base['source_valid'] and base['wall_v']<=1e-12
                and base['rate_v']<=1e-10):
            _,_,_,H,gn_report=self.fast_gn_provider(source,z,base,h)
        else:
            H=np.eye(D)
            gn_report={}
        elapsed=(time.perf_counter()-started)*1000
        self.time_ms.append(elapsed)
        return base_gradient,jac_wall,jac_rate,H,{
            'derivative_method':'source_exact_parallel_fd_constraint',
            'hessian_method':'minimum_norm_phase1' if
                  not base['source_valid'] else 'source_GN_BFGS',
            'full_source_perturbations':2*D,
            'perturbation_workers':len(self.workers),
            'constraint_jacobian_ms':elapsed,
            **gn_report}

    def close(self):
        self.pool.shutdown(wait=True)
        for w in self.workers:w.close()



def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case",type=Path)
    parser.add_argument("seed",type=Path)
    parser.add_argument("--iterations",type=int,default=6)
    parser.add_argument("--workspace",type=Path,
                        default=Path("/tmp/mpc_fast_source_sqp"))
    parser.add_argument("--source-binary",type=Path,
                        default=Path("/tmp/mpc_source_merit_persistent"))
    parser.add_argument("--qp-binary",type=Path,
                        default=Path("/tmp/mpc_offline_source_qp_server"))
    parser.add_argument("--no-persistent-qp",action="store_true")
    parser.add_argument("--osqp-vendor",type=Path,
                        default=Path("/tmp/sdu-mpc-osqp-v104"))
    parser.add_argument("--no-persistent",action="store_true")
    parser.add_argument("--no-early-feasible",action="store_true")
    soc=parser.add_mutually_exclusive_group()
    soc.add_argument("--source-soc",dest="source_soc",action="store_true",
                     help="exact source second-order constraint correction (default)")
    soc.add_argument("--no-source-soc",dest="source_soc",action="store_false",
                     help="disable exact source second-order constraint correction for ablation")
    parser.set_defaults(source_soc=True)
    parser.add_argument("--exhaustive-line-search",action="store_true",
                        help="reference mode: sample every alpha instead of first accepted exact-source step")
    parser.add_argument("--calibrate-source-curvature",action="store_true",
                        help="for already source-feasible nominals, check source cost curvature with two exact directional probes")
    parser.add_argument("--constraint-jacobian",
                        choices=("tangent","parallel_fd"),
                        default="parallel_fd",
                        help="parallel_fd is the accuracy-qualified reference; tangent is only a diagnostic")
    parser.add_argument("--workers",type=int,default=8)
    args=parser.parse_args()
    if args.iterations<1 or args.iterations>30:
        parser.error("invalid iteration count")
    args.workspace.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(args.osqp_vendor))
    import osqp
    source=PersistentSource(args.case,args.source_binary,args.workspace,
                            persistent=not args.no_persistent)
    tangent=TangentBfgsDerivatives(args.case,args.qp_binary,args.workspace,
              persistent=not args.no_persistent_qp)
    model=(ParallelFiniteDifferenceConstraints(
        args.case,args.source_binary,args.workspace,args.workers,tangent)
        if args.constraint_jacobian=="parallel_fd" else tangent)
    started=time.perf_counter()
    try:
        result=run(source,np.loadtxt(args.seed),osqp,args.iterations,.002,
                   args.workspace,args.source_soc,args.calibrate_source_curvature,
                   derivative_provider=model,
                   early_full_step_feasible=not args.no_early_feasible,
                   first_accepted_globalization=not args.exhaustive_line_search)
    finally:
        source.close()
        if model is not tangent:model.close()
        tangent.close()
    # Failure to find an *improving* SQP step after an already strict-feasible
    # incumbent does not invalidate that incumbent. Report the real final
    # accepted physical state, never merely the last attempted optimization.
    last=result[-1] if result else {}
    final_valid=last.get('source_exact_feasible',
                         last.get('source_valid',False))
    final_cost=last.get('next_cost',last.get('cost'))
    final_wall_v=last.get('next_wall_V',last.get('wall_V'))
    report=dict(total_elapsed_ms=1000*(time.perf_counter()-started),
                source_model_evaluations=source.calls,
                source_evaluator_cache_hits=source.cache_hits,
                constraint_derivative_mode=args.constraint_jacobian,
                perturbation_workers=args.workers if model is not tangent else 0,
                source_perturbations=getattr(model,"full_source_perturbations",0),
                tangent_qp_relinearizations=tangent.calls,
                tangent_qp_ms=sum(tangent.time_ms),
                constraint_jacobian_ms=sum(model.time_ms),
                source_evaluator_ms=sum(source.server_roundtrip_ms),
                first_feasible_iteration=next(
                   (r['iteration']+1 for r in result
                    if r.get("source_exact_feasible")),None),
                final_source_feasible=bool(final_valid),
                final_source_cost=final_cost,
                final_original_wall_violation_m2=final_wall_v)
    (args.workspace/"performance.json").write_text(json.dumps(report,indent=2))
    print(json.dumps({"performance":report}),flush=True)


if __name__=="__main__":
    main()
