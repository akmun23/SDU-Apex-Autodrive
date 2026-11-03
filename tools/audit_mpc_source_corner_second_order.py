#!/usr/bin/env python3
"""Compare exact 25ms Unity-source plant transitions against first- and
second-order directional Taylor models across recorded high-slip states.

Offline numerical experiment only. No fitted tire curve, no controller
configuration changes. One local directional Hessian estimate requires
f(x+h d), f(x-h d); it is not the full 20x20 source Hessian.
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import math
from pathlib import Path
from statistics import median

from tools import evaluate_mpc_bag as ev
from tools import analyze_source_plant_one_step as one
from tools.audit_mpc_source_corner_jacobians import SourceSeeds, SourceJacobian

DIRECTIONS = (
    ("lateral_velocity",1,.004,.15),
    ("yaw_rate",2,.002,.20),
    ("physical_steering",13,.0003,.03),
    ("rear_left_jounce",16,.00015,.004),
    ("pitch",11,.0002,.008),
    ("steering_command",-1,.001,.07),
    ("throttle_command",-2,.003,.12),
)
CHANNELS = (
    ("lateral_speed_mps",lambda o:o.state[1]),
    ("yaw_rate_rps",lambda o:o.state[2]),
    ("pitch_rad",lambda o:o.state[11]),
    ("rear_left_jounce_m",lambda o:o.state[16]),
    ("integrated_lateral_m",lambda o:o.integrated_body_lateral_displacement_m),
    ("integrated_yaw_rad",lambda o:o.integrated_yaw_delta_rad),
)
SCALES = {
    "lateral_speed_mps": 1.0,
    "yaw_rate_rps": 1.0,
    "pitch_rad": .1,
    "rear_left_jounce_m": .01,
    "integrated_lateral_m": .025,
    "integrated_yaw_rad": .025,
}


def run(bag:Path,lib:Path,max_rows:int,h_scale:float)->list[dict]:
    data=ev.read_bag(bag)
    rows=sorted(data[ev.PLANT],key=lambda x:x["stamp_ns"])
    rows=[r for r in rows if abs(r["physical_steering"])>.3
          and 2.0<r["u"]<4.5]
    if len(rows)>max_rows:
        rows=[rows[round(i*(len(rows)-1)/(max_rows-1))]
              for i in range(max_rows)]
    native=ctypes.CDLL(str(lib.resolve()))
    step=native.unity_source_plant_step_c
    step.argtypes=[ctypes.POINTER(ev.UnitySourcePlantInputC),
                   ctypes.c_double,ctypes.POINTER(ev.UnitySourcePlantOutputC)]
    step.restype=ctypes.c_int
    sensitivity=native.unity_source_plant_step_sensitivity_c
    sensitivity.argtypes=[ctypes.POINTER(ev.UnitySourcePlantInputC),
                          ctypes.c_double,ctypes.POINTER(SourceSeeds),
                          ctypes.POINTER(SourceJacobian)]
    sensitivity.restype=ctypes.c_int
    results=[]
    for state in rows:
        inp=ev.UnitySourcePlantInputC()
        inp.state[:]=one._source_state(state)
        inp.wheel_rotation_angle_rad[:]=state["wheel_angle"]
        inp.steering_target_rad=state["delayed_steering_2"]*ev.MAX_STEERING_RAD
        inp.throttle_normalized=state["delayed_throttle_2"]
        inp.suspension_history_initialized=int(state["suspension_history_initialized"])
        base=ev.UnitySourcePlantOutputC()
        if not step(ctypes.byref(inp),.025,ctypes.byref(base)):
            continue
        seeds=SourceSeeds()
        for lane,(_,index,_,_) in enumerate(DIRECTIONS):
            if index>=0:seeds.state[index][lane]=1.
            elif index==-1:seeds.steering_target[lane]=1.
            else:seeds.throttle[lane]=1.
        jac=SourceJacobian()
        if not sensitivity(ctypes.byref(inp),.025,ctypes.byref(seeds),ctypes.byref(jac)):
            continue
        for lane,(name,index,nominal_h,delta) in enumerate(DIRECTIONS):
            h=nominal_h*h_scale
            def evaluate(amount):
                trial=ev.UnitySourcePlantInputC()
                ctypes.memmove(ctypes.byref(trial),ctypes.byref(inp),ctypes.sizeof(inp))
                if index>=0:trial.state[index]+=amount
                elif index==-1:trial.steering_target_rad+=amount
                else:trial.throttle_normalized+=amount
                out=ev.UnitySourcePlantOutputC()
                return out if step(ctypes.byref(trial),.025,ctypes.byref(out)) else None
            positive=evaluate(h)
            negative=evaluate(-h)
            actual=evaluate(delta)
            if positive is None or negative is None or actual is None:
                continue
            local=CHANNELS
            for label,getter in local:
                f0=getter(base);fp=getter(positive)
                fm=getter(negative);ftrue=getter(actual)
                if label=="integrated_lateral_m":
                    grad=jac.body_lateral_displacement[lane]
                elif label=="integrated_yaw_rad":
                    grad=jac.yaw_delta[lane]
                else:
                    ix={"lateral_speed_mps":1,"yaw_rate_rps":2,
                        "pitch_rad":11,"rear_left_jounce_m":16}[label]
                    grad=jac.state[ix][lane]
                curvature=(fp-2*f0+fm)/(h*h)
                linear=f0+grad*delta
                quadratic=linear+.5*curvature*delta*delta
                e1=abs(linear-ftrue);e2=abs(quadratic-ftrue)
                results.append({
                    "epoch_ns":state["stamp_ns"],"speed_mps":state["u"],
                    "physical_steering_rad":state["physical_steering"],
                    "direction":name,"step":delta,"h":h,
                    "output":label,"exact":ftrue,
                    "linear_prediction":linear,"quadratic_prediction":quadratic,
                    "first_order_error":e1,"second_order_error":e2,
                    "first_order_error_scaled":e1/SCALES[label],
                    "second_order_error_scaled":e2/SCALES[label],
                    "second_order_better":e2<e1,
                    "curvature":curvature,
                })
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("bag",type=Path)
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--library",type=Path,
      default=Path("build/f1tenth_mpc/libunity_source_plant_eval.so"))
    p.add_argument("--max-rows",type=int,default=18)
    p.add_argument("--h-scale",type=float,default=1.)
    a=p.parse_args()
    if not .1<=a.h_scale<=20.:p.error("--h-scale outside [.1,20]")
    results=run(a.bag,a.library,a.max_rows,a.h_scale)
    if not results:raise SystemExit("no valid source perturbations")
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open("w",newline="") as stream:
        w=csv.DictWriter(stream,fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
    print("samples",len(results),"states",len({x["epoch_ns"] for x in results}))
    for key in DIRECTIONS:
        subset=[x for x in results if x["direction"]==key[0]]
        if not subset:continue
        m1=median(x["first_order_error_scaled"] for x in subset)
        m2=median(x["second_order_error_scaled"] for x in subset)
        advantage=sum(x["second_order_better"] for x in subset)/len(subset)
        print(f'{key[0]:22} median_first_scaled={m1:.7g} '
              f'median_second_scaled={m2:.7g} '
              f'quadratic_better={advantage:.0%}')
    print("saved",a.output)


if __name__=="__main__":
    main()
