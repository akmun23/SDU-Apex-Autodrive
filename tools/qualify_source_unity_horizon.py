#!/usr/bin/env python3
"""Evaluate unmodified source plant over real multi-step Unity motion windows.

The published sensor-corrected 26-state source plant is initialized ONCE at
the start of each window. Thereafter every physical state and integrated
motion comes exclusively from the original C++ source evaluator. Recorded
Unity truth is used only as an initial WORLD pose and as independent scores.
Two command-association hypotheses are compared: newest queued and actual
bridge-sent requested physical steering and throttle. The latter is NOT proof
of the torque/steer applied by individual Unity FixedUpdate ticks.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
from pathlib import Path

try:
    from tools import evaluate_mpc_bag as eb
    from tools.analyze_source_plant_one_step import _source_state, _metric
except ModuleNotFoundError:
    import evaluate_mpc_bag as eb
    from analyze_source_plant_one_step import _source_state, _metric


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def impact_like_transitions(truth: dict[int, dict]) -> list[int]:
    """Flag suspect impulsive Unity events; these are NOT collision labels.

    A free-flight source plant cannot predict obstacle contact impulses.
    Preserve these windows in the full report, but separate their accuracy
    statistics from regular physical motion. The collision_count topic may
    report impact later than the physical transition.
    """
    observed = sorted(truth)
    events = []
    for a, b in zip(observed, observed[1:]):
        dt = (b - a) * 1e-9
        if not 0.020 <= dt <= 0.030:
            continue
        start, end = truth[a], truth[b]
        du = abs(end["u"] - start["u"])
        dyaw = abs(math.atan2(math.sin(end["yaw"] - start["yaw"]),
                              math.cos(end["yaw"] - start["yaw"])))
        if du >= 1.0 or dyaw >= 0.25:
            events.append(b)
    return events


def analyze(bag: Path, library: Path, horizon: int, stride: int,
            oracle_motion_start: bool = False) -> dict:
    data = eb.read_bag(bag)
    truth = {int(r["stamp_ns"]): r for r in data.get(eb.GT, [])}
    plant = {int(r["stamp_ns"]): r for r in data.get(eb.PLANT, [])}
    common = sorted(truth.keys() & plant.keys())
    impulse_stamps = impact_like_transitions(truth)
    bridges = {}
    for row in data.get(eb.BRIDGE_TIMING, []):
        p = row.get("json", {})
        stamp = p.get("bridge_request_ros_stamp_ns")
        if stamp is not None:
            bridges.setdefault(int(stamp), p)
    if len(common) <= horizon:
        raise RuntimeError("Not enough exact epochs for a full horizon")
    so = ctypes.CDLL(str(library.resolve()))
    step = so.unity_source_plant_step_c
    step.argtypes = [
        ctypes.POINTER(eb.UnitySourcePlantInputC), ctypes.c_double,
        ctypes.POINTER(eb.UnitySourcePlantOutputC)]
    step.restype = ctypes.c_int
    scored = {"queued": [], "bridge_sent": []}
    skipped = {}
    for start in range(0, len(common) - horizon, stride):
        stamps = common[start:start+horizon+1]
        if any(not .020 <= (b-a)*1.e-9 <= .030
               for a,b in zip(stamps,stamps[1:])):
            skipped["irregular_epoch_spacing"] = (
                skipped.get("irregular_epoch_spacing", 0)+1)
            continue
        initial = plant[stamps[0]]
        true = truth[stamps[0]]
        impact_in_window = next(
            (s for s in impulse_stamps if stamps[0] < s <= stamps[-1]),
            None)
        for hypothesis in scored:
            inp = eb.UnitySourcePlantInputC()
            inp.state[:] = _source_state(initial)
            if oracle_motion_start:
                # DIAGNOSTIC ONLY: substitute Unity truth at t0 for
                # three visible motion states, NOT for future states
                # or source hidden wheel/suspension/contact variables.
                # This isolates observer initialization from source
                # transition dynamics without modifying the plant.
                inp.state[0] = true["u"]
                inp.state[1] = true["v"]
                inp.state[2] = true["r"]
            inp.wheel_rotation_angle_rad[:] = initial["wheel_angle"]
            inp.suspension_history_initialized = int(
                initial["suspension_history_initialized"])
            x=float(true["x"])
            y=float(true["y"])
            yaw=float(true["yaw"])
            maximum_xy=0.
            maximum_heading=0.
            actual_commands=0
            good=True
            for index in range(horizon):
                stamp=stamps[index]
                previous=plant[stamp]
                if hypothesis=="bridge_sent":
                    sent=bridges.get(stamp, {})
                    try:
                        steering=float(sent["sent_steering_norm"])
                        throttle=float(sent["sent_throttle_norm"])
                    except (KeyError,ValueError,TypeError,OverflowError):
                        good=False
                        break
                    if not math.isfinite(steering) or not math.isfinite(throttle):
                        good=False
                        break
                    inp.steering_target_rad=steering*eb.MAX_STEERING_RAD
                    inp.throttle_normalized=throttle
                    actual_commands+=1
                else:
                    inp.steering_target_rad=(
                        previous["delayed_steering_2"]*eb.MAX_STEERING_RAD)
                    inp.throttle_normalized=previous["delayed_throttle_2"]
                # The original simulator/source dynamics advance on
                # exactly 25 internal 1ms ticks per ROS control epoch.
                # Header-stamp jitter is NOT an extra physical substep.
                duration=eb.ENCODER_SAMPLE_PERIOD_S
                out=eb.UnitySourcePlantOutputC()
                if not step(ctypes.byref(inp),duration,ctypes.byref(out)):
                    good=False
                    break
                local_x=out.integrated_body_forward_displacement_m
                local_y=out.integrated_body_lateral_displacement_m
                x+=math.cos(yaw)*local_x-math.sin(yaw)*local_y
                y+=math.sin(yaw)*local_x+math.cos(yaw)*local_y
                yaw+=out.integrated_yaw_delta_rad
                inp.state[:]=out.state[:]
                inp.wheel_rotation_angle_rad[:]=out.wheel_rotation_angle_rad[:]
                inp.suspension_history_initialized=(
                    out.suspension_history_initialized)
                target=truth[stamps[index+1]]
                xy=math.hypot(x-target["x"],y-target["y"])
                heading=math.atan2(math.sin(yaw-target["yaw"]),
                                   math.cos(yaw-target["yaw"]))
                maximum_xy=max(maximum_xy,xy)
                maximum_heading=max(maximum_heading,abs(heading))
            if not good:
                skipped[f"invalid_{hypothesis}"]=(
                    skipped.get(f"invalid_{hypothesis}",0)+1)
                continue
            initial_source_u=float(initial["u"])-float(true["u"])
            source_speed=math.hypot(true["u"],true["v"])
            xy_end=math.hypot(x-truth[stamps[-1]]["x"],
                              y-truth[stamps[-1]]["y"])
            scored[hypothesis].append({
                "start_stamp_ns":stamps[0],
                "end_stamp_ns":stamps[-1],
                "impact_like_truth_discontinuity_in_window":
                    impact_in_window is not None,
                "first_impact_like_truth_discontinuity_stamp_ns":
                    impact_in_window,
                "source_first_state_forward_error_mps":initial_source_u,
                "source_initial_speed_mps":source_speed,
                "source_initial_steering_rad":initial["physical_steering"],
                "horizon_s":(stamps[-1]-stamps[0])*1.e-9,
                "maximum_xy_error_m":maximum_xy,
                "endpoint_xy_error_m":xy_end,
                "maximum_yaw_error_rad":maximum_heading,
                "terminal_yaw_error_rad":math.atan2(
                    math.sin(yaw-truth[stamps[-1]]["yaw"]),
                    math.cos(yaw-truth[stamps[-1]]["yaw"])),
                "terminal_forward_speed_error_mps":
                    inp.state[0]-truth[stamps[-1]]["u"],
                "command_epochs_from_bridge":actual_commands,
            })
    def describe(items: list[dict]) -> dict:
        return {
            "windows":len(items),
            "maximum_path_error_m":_metric(
                [r["maximum_xy_error_m"] for r in items]),
            "endpoint_path_error_m":_metric(
                [r["endpoint_xy_error_m"] for r in items]),
            "maximum_yaw_error_rad":_metric(
                [r["maximum_yaw_error_rad"] for r in items]),
            "terminal_forward_speed_error_mps":_metric(
                [r["terminal_forward_speed_error_mps"] for r in items]),
            "full_path_over_5cm":sum(
                r["maximum_xy_error_m"]>.05 for r in items),
            "full_path_over_3cm":sum(
                r["maximum_xy_error_m"]>.03 for r in items),
        }

    # Never delete impacted windows or advertise the split as a source
    # equivalence certificate. Both full and partitioned errors are retained.
    summary = {name:describe(items) for name,items in scored.items()}
    no_impulse = {name:describe([
        r for r in items
        if not r["impact_like_truth_discontinuity_in_window"]])
        for name,items in scored.items()}
    with_impulse = {name:describe([
        r for r in items
        if r["impact_like_truth_discontinuity_in_window"]])
        for name,items in scored.items()}
    by_stamp={r["start_stamp_ns"]:r
              for r in scored["bridge_sent"]}
    paired=[(r,by_stamp[r["start_stamp_ns"]])
            for r in scored["queued"]
            if r["start_stamp_ns"] in by_stamp]
    return {
        "schema":"original-source-vs-unity-full-horizon/v1",
        "source_plant_unmodified":True,
        "source_from_published_sensor_corrected_state_only_at_t0":
            not oracle_motion_start,
        "diagnostic_unity_motion_states_at_t0_only":
            oracle_motion_start,
        "original_unity_pose_at_t0_only":True,
        "independent_unity_pose_used_only_for_score":True,
        "bridge_request_not_equivalent_to_FixedUpdate_applied_command":True,
        "source_physics_not_qualified_as_unity_equivalent":True,
        "source_library_sha256":_sha(library),
        "source_cpp_sha256":_sha(
            Path(__file__).resolve().parents[1]/
            "f1tenth_mpc/src/unity_source_plant.cpp"),
        "horizon_original_physical_stages":horizon,
        "candidate_window_stride":stride,
        "skipped":skipped,
        "summary":summary,
        "summary_without_impact_like_truth_discontinuities":no_impulse,
        "summary_containing_impact_like_truth_discontinuities":with_impulse,
        "impact_like_truth_transition_stamps_ns":impulse_stamps,
        "impact_like_truth_event_definition":
            "abs(delta forward speed)>=1.0m/s or abs(delta yaw)>=0.25rad"
            " over an approximately 25ms Unity epoch;"
            " diagnostic only, not a substitute for collision_count",
        "pair_count":len(paired),
        "bridge_sent_lower_maximum_error":sum(
            b["maximum_xy_error_m"]<a["maximum_xy_error_m"]
            for a,b in paired),
        "bridge_sent_higher_maximum_error":sum(
            b["maximum_xy_error_m"]>a["maximum_xy_error_m"]
            for a,b in paired),
        "source_windows_exceeding_5cm":{
            name:[r for r in rows if r["maximum_xy_error_m"]>.05]
            for name,rows in scored.items()},
        "worst_source_windows":{
            name:sorted(rows,key=lambda r:r["maximum_xy_error_m"],
                        reverse=True)[:8]
            for name,rows in scored.items()},
    }


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag",type=Path)
    parser.add_argument("--library",required=True,type=Path)
    parser.add_argument("--horizon",type=int,default=30)
    parser.add_argument("--stride",type=int,default=10)
    parser.add_argument("--oracle-motion-start",action="store_true",
                        help="Diagnostic-only GT u/v/r at t0; source hidden states and future dynamics unchanged")
    parser.add_argument("--output",required=True,type=Path)
    a=parser.parse_args()
    if a.horizon<1 or a.horizon>60 or a.stride<1:
        parser.error("Require horizon in [1,60], stride >=1")
    x=analyze(a.bag,a.library,a.horizon,a.stride,
              oracle_motion_start=a.oracle_motion_start)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(x,indent=2)+"\n")
    for mode,result in x["summary"].items():
        print(mode,"windows",result["windows"],
              "full path >5cm",result["full_path_over_5cm"],
              "max path RMSE m",
              result["maximum_path_error_m"]["rmse"],
              "p95",result["maximum_path_error_m"]["p95_abs"])
    return 0


if __name__=="__main__":
    raise SystemExit(main())
