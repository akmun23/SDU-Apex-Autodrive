#!/usr/bin/env python3
"""Exact-epoch source Jacobian audit for QP dynamics (offline, no ROS control).

Finite-difference a 25 ms full-physics source transition at ACTUAL corrected
UnityPlantState + request-slot-2 input. Compare against the very same source
forward sensitivities used to construct the MPC QP. The simulator truth is
never injected into source dynamics, and finite-step accuracy versus Unity
is deliberately kept separate from local derivative accuracy.
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import math
from pathlib import Path
from tools import evaluate_mpc_bag as ev
from tools import analyze_source_plant_one_step as one


WIDTH = 20
NX = ev.UNITY_SOURCE_PLANT_STATE_SIZE

class SourceSeeds(ctypes.Structure):
    _fields_ = [
        ("state", (ctypes.c_double * WIDTH) * NX),
        ("steering_target", ctypes.c_double * WIDTH),
        ("throttle", ctypes.c_double * WIDTH),
    ]

class SourceJacobian(ctypes.Structure):
    _fields_ = [
        ("state", (ctypes.c_double * WIDTH) * NX),
        ("body_forward_displacement", ctypes.c_double * WIDTH),
        ("body_lateral_displacement", ctypes.c_double * WIDTH),
        ("yaw_delta", ctypes.c_double * WIDTH),
        ("maximum_longitudinal_slip_excess_mps", ctypes.c_double * 4),
        ("maximum_longitudinal_slip_excess_derivative",
         (ctypes.c_double * WIDTH) * 4),
    ]

# State columns include body pose and source-wheel dynamics; the last two
# columns are the physical first-order steering and throttle inputs.
DIRECTIONS = (
    ("u", 0, 0.0001), ("v", 1, 0.0001),
    ("r", 2, 0.0001), ("omega_RL", 5, 0.001),
    ("omega_RR", 6, 0.001), ("roll", 9, 0.00001),
    ("pitch", 11, 0.00001), ("jounce_RL", 16, 0.00001),
    ("steering_state", 13, 0.00001),
    ("steering_command", -1, 0.00003),
    ("throttle_command", -2, 0.0001),
)

def audit_bag(bag: Path, library: Path, max_rows: int, epsilon_scale: float) -> list[dict[str, object]]:
    data = ev.read_bag(bag)
    plant = sorted(data[ev.PLANT], key=lambda r: r["stamp_ns"])
    native = ctypes.CDLL(str(library.resolve()))
    step = native.unity_source_plant_step_c
    step.argtypes = [ctypes.POINTER(ev.UnitySourcePlantInputC),
                     ctypes.c_double, ctypes.POINTER(ev.UnitySourcePlantOutputC)]
    step.restype = ctypes.c_int
    sensitivity = native.unity_source_plant_step_sensitivity_c
    sensitivity.argtypes = [ctypes.POINTER(ev.UnitySourcePlantInputC),
                            ctypes.c_double, ctypes.POINTER(SourceSeeds),
                            ctypes.POINTER(SourceJacobian)]
    sensitivity.restype = ctypes.c_int
    choices = [r for r in plant if abs(r["physical_steering"]) > 0.3 and
               2.0 < r["u"] < 4.5]
    # Deterministic selection of high-absolute-rack regimes across all laps.
    # Include extremes and evenly spaced samples, not only failures.
    if len(choices) > max_rows:
        positions = sorted(set(
            round(i*(len(choices)-1)/(max_rows-1))
            for i in range(max_rows)))
        choices = [choices[i] for i in positions]
    records = []
    for state_row in choices:
        inp = ev.UnitySourcePlantInputC()
        inp.state[:] = one._source_state(state_row)
        inp.wheel_rotation_angle_rad[:] = state_row["wheel_angle"]
        inp.steering_target_rad = state_row["delayed_steering_2"] * ev.MAX_STEERING_RAD
        inp.throttle_normalized = state_row["delayed_throttle_2"]
        inp.suspension_history_initialized = int(state_row["suspension_history_initialized"])
        seeds = SourceSeeds()
        for lane, (_, col, _) in enumerate(DIRECTIONS):
            if col >= 0:
                seeds.state[col][lane] = 1.0
            elif col == -1:
                seeds.steering_target[lane] = 1.0
            else:
                seeds.throttle[lane] = 1.0
        jac = SourceJacobian()
        if not sensitivity(ctypes.byref(inp), .025, ctypes.byref(seeds), ctypes.byref(jac)):
            raise RuntimeError(f"sensitivity failed at {state_row['stamp_ns']}")
        for lane, (label, col, eps_nominal) in enumerate(DIRECTIONS):
            eps = eps_nominal * epsilon_scale
            plus = ev.UnitySourcePlantInputC()
            minus = ev.UnitySourcePlantInputC()
            ctypes.memmove(ctypes.byref(plus), ctypes.byref(inp), ctypes.sizeof(inp))
            ctypes.memmove(ctypes.byref(minus), ctypes.byref(inp), ctypes.sizeof(inp))
            if col >= 0:
                plus.state[col] += eps
                minus.state[col] -= eps
            elif col == -1:
                plus.steering_target_rad += eps
                minus.steering_target_rad -= eps
            else:
                plus.throttle_normalized += eps
                minus.throttle_normalized -= eps
            ahead = ev.UnitySourcePlantOutputC()
            behind = ev.UnitySourcePlantOutputC()
            valid_plus=bool(step(ctypes.byref(plus), .025, ctypes.byref(ahead)))
            valid_minus=bool(step(ctypes.byref(minus), .025, ctypes.byref(behind)))
            if not (valid_plus or valid_minus):
                raise RuntimeError(f"both perturbations invalid {label} {state_row['stamp_ns']}")
            scheme="central"
            baseline=ev.UnitySourcePlantOutputC()
            if not (valid_plus and valid_minus):
                # At saturated actuator bounds, an outward perturbation is
                # physically invalid. Compare the actual feasible one-sided
                # tangent; never invent an invalid full-physics derivative.
                if not step(ctypes.byref(inp), .025, ctypes.byref(baseline)):
                    raise RuntimeError(f"baseline invalid {label} {state_row['stamp_ns']}")
                scheme="forward" if valid_plus else "backward"
            def difference(attr,index=None):
                xplus=getattr(ahead,attr) if valid_plus else None
                xminus=getattr(behind,attr) if valid_minus else None
                xbase=getattr(baseline,attr) if scheme!="central" else None
                if index is not None:
                    if xplus is not None:xplus=xplus[index]
                    if xminus is not None:xminus=xminus[index]
                    if xbase is not None:xbase=xbase[index]
                if scheme=="central":return (xplus-xminus)/(2*eps)
                if scheme=="forward":return (xplus-xbase)/eps
                return (xbase-xminus)/eps
            components = [(f"state_{i}",jac.state[i][lane],
                           difference("state",i))
                          for i in (0,1,2,3,4,5,6,7,9,11,13)]
            components += [
                ("forward_integrated",jac.body_forward_displacement[lane],
                 difference("integrated_body_forward_displacement_m")),
                ("lateral_integrated",jac.body_lateral_displacement[lane],
                 difference("integrated_body_lateral_displacement_m")),
                ("yaw_integrated",jac.yaw_delta[lane],
                 difference("integrated_yaw_delta_rad")),
            ]
            for component, actual, fd in components:
                abs_error=abs(actual-fd)
                relative_error=abs_error/max(0.001,abs(fd))
                records.append({
                    "bag":str(bag),"stamp_ns":state_row["stamp_ns"],
                    "speed_mps":state_row["u"],
                    "steering_rad":state_row["physical_steering"],
                    "input":label,"output":component,
                    "analytic":actual,"finite_difference":fd,
                    "absolute_error":abs_error,
                    "scaled_relative_error":relative_error,
                    "epsilon":eps,
                    "difference_scheme":scheme,
                })
    return records

def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("bags",nargs="+",type=Path)
    p.add_argument("--library",type=Path,default=Path(
        "build/f1tenth_mpc/libunity_source_plant_eval.so"))
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--max-rows-per-bag",type=int,default=12)
    p.add_argument("--epsilon-scale",type=float,default=1.0)
    args=p.parse_args()
    if not 1e-4 <= args.epsilon_scale <= 100.0:
        p.error("--epsilon-scale outside [1e-4, 100]")
    if args.max_rows_per_bag < 2:
        p.error("--max-rows-per-bag must be >=2")
    rows=[]
    for bag in args.bags:
        new=audit_bag(bag,args.library,args.max_rows_per_bag,args.epsilon_scale)
        print(f"bag {bag}: {len(new)//(len(DIRECTIONS)*14)} corner states")
        rows.extend(new)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=rows[0].keys())
        writer.writeheader();writer.writerows(rows)
    for metric in ("absolute_error","scaled_relative_error"):
        selected=sorted(rows,key=lambda r:r[metric],reverse=True)
        print(f"{metric}: max {selected[0][metric]:.7g}")
        for r in selected[:8]:
            print("  ",r["input"],r["output"],
                  f"steer={r['steering_rad']:.3f}, speed={r['speed_mps']:.3f}",
                  f"jac={r['analytic']:.7g} fd={r['finite_difference']:.7g}",
                  f"err={r[metric]:.5g}")
    print(f"wrote {len(rows)} derivative checks -> {args.output}")

if __name__=="__main__":
    main()
