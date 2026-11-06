#!/usr/bin/env python3
"""Audit every ROS-source SQP epoch for QP/true-source agreement.

Consumes exported *_diagnostics.jsonl from export_mpc_run_report.py or
direct payload JSONL. The ROS controller reports each SQP full-step QP
prediction and already-computed exact source rollout; this script adds
NO work to the hard 22 ms callback. It never treats a QP-predicted safe
trajectory as a source certificate.

Usage:
  python3 tools/analyze_mpc_qp_agreement.py \
      /tmp/mpc_live_diagnostics.jsonl --out /tmp/model_agreement.json

The output distinguishes uninstrumented logs from true agreement.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import pathlib
import statistics
from typing import Any

def number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        n=float(value)
    except (ValueError,TypeError):
        return None
    return n if math.isfinite(n) else None

def summary(values: list[float]) -> dict[str,float] | None:
    if not values:
        return None
    ordered=sorted(values)
    return {
        "count":len(ordered),
        "median":round(statistics.median(ordered),6),
        "p95":round(ordered[math.ceil(.95*len(ordered))-1],6),
        "p99":round(ordered[math.ceil(.99*len(ordered))-1],6),
        "max":round(ordered[-1],6),
    }

def report(path: pathlib.Path, *, wall_threshold_m2: float=1.e-5,
           defect_threshold_m:float=.05) -> dict[str,Any]:
    if wall_threshold_m2<=0. or defect_threshold_m<=0.:
        raise ValueError("positive thresholds are required")
    statuses=collections.Counter()
    stages={i:[] for i in range(2)}
    cost=[]
    accepted_cost=[]
    init=[]
    bad_streak=0
    worst_bad_streak=0
    cycles=0
    telemetry_cycles=0
    iteration_timing=collections.defaultdict(list)
    traces=[]
    with path.open(encoding="utf-8") as stream:
        for lineno,line in enumerate(stream,1):
            if not line.strip():
                continue
            try:
                outer=json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at line {lineno}: {exc}") from exc
            if not isinstance(outer,dict):
                raise ValueError(f"expected object at line {lineno}")
            row=outer.get("payload",outer)
            if not isinstance(row,dict):
                raise ValueError(f"expected payload object at line {lineno}")
            solver=row.get("solver") or {}
            if not isinstance(solver,dict):
                solver={}
            # Exclude other controller modes, but do not hide reported
            # control-cycle deferrals in the same source MPC run.
            if solver.get("native_source_sqp_enabled") is False:
                continue
            status=str(row.get("status",outer.get("status","unknown")))
            statuses[status]+=1
            cycles+=1
            published=bool(row.get("command_published",outer.get("command_published",False)))
            if published:
                bad_streak=0
            else:
                bad_streak+=1
                worst_bad_streak=max(worst_bad_streak,bad_streak)
            stamp=outer.get("candidate_command_stamp_ns",
                            row.get("candidate_command_stamp_ns"))
            progress=number(row.get("progress_m"))
            agreement=solver.get("native_source_sqp_full_step_model_agreement")
            full_cost_recorded=False
            iterations=solver.get("source_iterations")
            if isinstance(iterations,list) and iterations:
                telemetry_cycles+=1
                for idx,step in enumerate(iterations):
                    if not isinstance(step,dict):
                        continue
                    for key in ("linearization_ms","qp_ms","source_rollout_ms","total_ms"):
                        value=number(step.get(key))
                        if value is not None and value>=0.:
                            iteration_timing[key].append(value)
                    if step.get("full_step_evaluated") is not True:
                        continue
                    before=number(step.get("original_cost_before"))
                    after=number(step.get("full_step_original_cost"))
                    predicted=number(step.get("qp_predicted_cost_change"))
                    if before is None or after is None or predicted is None:
                        continue
                    actual=after-before
                    event=dict(line=lineno,stamp_ns=stamp,progress_m=progress,
                               pass_index=idx,predicted_delta=predicted,
                               actual_delta=actual,error=actual-predicted)
                    cost.append(event)
                    full_cost_recorded=True
                    # A corrected/backtracked acceptance is a different
                    # control trajectory. Never compare its cost with the
                    # full direction's quadratic prediction.
                    if step.get("full_step_accepted") is True:
                        accepted_cost.append(event)
            if isinstance(agreement,list):
                telemetry_cycles+=1
                for idx,step in enumerate(agreement[:2]):
                    if not isinstance(step,dict) or not step.get("valid"):
                        continue
                    pred=number(step.get("predicted_wall_violation_m2"))
                    exact=number(step.get("exact_source_wall_violation_m2"))
                    defect=number(step.get("maximum_wall_linearization_defect_m"))
                    if pred is None or exact is None or defect is None:
                        continue
                    event=dict(
                        line=lineno,stamp_ns=stamp,progress_m=progress,
                        status=status,published=published,pass_index=idx,
                        predicted_wall_violation_m2=pred,
                        exact_source_wall_violation_m2=exact,
                        signed_wall_prediction_error_m2=exact-pred,
                        max_wall_model_defect_m=defect,
                        accepted_alpha=number(step.get("accepted_alpha")),
                        racing_cost_model_valid=step.get("racing_cost_model_valid") is True,
                        racing_cost_prediction_error=(
                            number(step.get("racing_cost_prediction_error"))
                            if step.get("racing_cost_model_valid") is True else None),
                        accepted_racing_cost_model_valid=(
                            step.get("accepted_racing_cost_model_valid") is True),
                        accepted_racing_cost_prediction_error=(
                            number(step.get("accepted_racing_cost_prediction_error"))
                            if step.get("accepted_racing_cost_model_valid") is True else None),
                        false_safe_prediction=pred<=wall_threshold_m2 and
                            exact>wall_threshold_m2,
                        large_clearance_defect=defect>defect_threshold_m,
                    )
                    stages[idx].append(event)
                    if event["racing_cost_model_valid"]:
                        predicted=number(step.get("qp_predicted_racing_cost_change"))
                        actual=number(step.get("exact_source_racing_cost_change"))
                        if predicted is not None and actual is not None:
                            full_cost_recorded=True
                            cost.append(dict(
                                line=lineno,stamp_ns=stamp,progress_m=progress,
                                pass_index=idx,predicted_delta=predicted,
                                actual_delta=actual,error=actual-predicted))
                    if event["accepted_racing_cost_model_valid"]:
                        predicted=number(step.get(
                            "accepted_qp_predicted_racing_cost_change"))
                        actual=number(step.get(
                            "accepted_exact_source_racing_cost_change"))
                        if predicted is not None and actual is not None:
                            accepted_cost.append(dict(
                                line=lineno,stamp_ns=stamp,progress_m=progress,
                                pass_index=idx,predicted_delta=predicted,
                                actual_delta=actual,error=actual-predicted))
                    if event["false_safe_prediction"] or event["large_clearance_defect"]:
                        traces.append(event)
            # Older logs provide only accepted-alpha first-pass cost
            # comparisons. Avoid counting those twice when the new full-
            # step physical-source cost model is available in this epoch.
            if (not full_cost_recorded and
                solver.get("native_source_sqp_cost_prediction_valid") is True):
                predicted=number(solver.get("native_source_sqp_predicted_cost_change"))
                actual=number(solver.get("native_source_sqp_actual_cost_change"))
                if predicted is not None and actual is not None:
                    cost.append(dict(
                        line=lineno,stamp_ns=stamp,progress_m=progress,
                        predicted_delta=predicted,actual_delta=actual,
                        error=actual-predicted))
            if solver.get("native_source_sqp_feasibility_initialization_trials"):
                init.append(dict(
                    stamp_ns=stamp,
                    used=bool(solver.get(
                        "native_source_sqp_feasibility_initialization_accepted")),
                    original_wall_violation_m2=number(solver.get(
                        "native_source_sqp_original_wall_violation_m2")),
                    selected_initial_wall_violation_m2=number(solver.get(
                        "native_source_sqp_selected_initial_wall_violation_m2")),
                    extra_us=number(solver.get(
                        "native_source_sqp_feasibility_initialization_us")),
                ))
    def aggregate(events: list[dict]) -> dict[str,Any]:
        return dict(
            samples=len(events),
            false_safe_count=sum(e["false_safe_prediction"] for e in events),
            large_clearance_defect_count=sum(e["large_clearance_defect"] for e in events),
            wall_prediction_abs_error_m2=summary([
                abs(e["signed_wall_prediction_error_m2"]) for e in events]),
            max_clearance_defect_m=summary([
                e["max_wall_model_defect_m"] for e in events]),
            selected_alpha=summary([
                e["accepted_alpha"] for e in events if e["accepted_alpha"] is not None]),
            racing_cost_prediction_samples=sum(
                e["racing_cost_prediction_error"] is not None for e in events),
            racing_cost_prediction_abs_error=summary([
                abs(e["racing_cost_prediction_error"]) for e in events
                if e["racing_cost_prediction_error"] is not None]),
            accepted_racing_cost_prediction_abs_error=summary([
                abs(e["accepted_racing_cost_prediction_error"]) for e in events
                if e["accepted_racing_cost_prediction_error"] is not None]),
        )
    traces.sort(
        key=lambda e:(e["false_safe_prediction"],abs(e["signed_wall_prediction_error_m2"]),
                      e["max_wall_model_defect_m"]),reverse=True)
    return dict(schema="sdu_apex_source_sqp_agreement/v1",
        source=str(path),cycles=cycles,statuses=dict(statuses),
        longest_nonpublication_streak=worst_bad_streak,
        telemetry_cycles=telemetry_cycles,
        model_agreement_instrumented=telemetry_cycles>0,
        iteration_timing={key:summary(values) for key,values in iteration_timing.items()},
        per_pass={f"SQP_{i+1}":aggregate(stages[i]) for i in stages},
        cost_model=dict(samples=len(cost),
            absolute_cost_error=summary([abs(e["error"]) for e in cost]),
            sign_mismatch_count=sum(e["predicted_delta"]*e["actual_delta"]<0 for e in cost)),
        accepted_cost_model=dict(samples=len(accepted_cost),
            absolute_cost_error=summary([abs(e["error"]) for e in accepted_cost]),
            sign_mismatch_count=sum(
                e["predicted_delta"]*e["actual_delta"]<0 for e in accepted_cost)),
        initialization=dict(trials=len(init),
            selected=sum(e["used"] for e in init),
            additional_us=summary([e["extra_us"] for e in init
                                  if e["extra_us"] is not None])),
        worst_disagreement_epochs=traces[:20],
        interpretation="QP cost versus exact nonlinear SOURCE cost; not measured Unity cost. "
                       "QP-predicted safe is NOT independent source/Unity safety certification",
    )

def accepted_cost_precision_gate(
    result: dict[str,Any], *, max_p99: float=100.,
    minimum_samples: int=100,
) -> tuple[bool,str]:
    if not math.isfinite(max_p99) or max_p99<=0. or minimum_samples<1:
        raise ValueError("invalid accepted source cost precision gate")
    measured=result["accepted_cost_model"]["absolute_cost_error"]
    if measured is None or measured["count"]<minimum_samples:
        count=measured["count"] if measured is not None else 0
        return False,(
            f"insufficient source-exact accepted cost samples: "
            f"{count}/{minimum_samples}; not a passing model"
        )
    p99=measured["p99"]
    if p99>max_p99:
        return False,(
            f"accepted cost p99 {p99:.6g} exceeds limit {max_p99:.6g}"
        )
    return True,(
        f"accepted cost p99 {p99:.6g} <= {max_p99:.6g} "
        f"over {measured['count']} source-exact comparisons"
    )

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("diagnostics",type=pathlib.Path)
    p.add_argument("--out",type=pathlib.Path)
    p.add_argument("--wall-threshold-m2",type=float,default=1.e-5)
    p.add_argument("--defect-threshold-m",type=float,default=.05)
    p.add_argument("--max-accepted-cost-error-p99",type=float,
                   help="fail if empirical p99 exceeds this weighted-cost limit")
    p.add_argument("--min-accepted-cost-samples",type=int,default=100,
                   help="fail if fewer accepted source comparisons; default 100")
    args=p.parse_args()
    result=report(args.diagnostics,wall_threshold_m2=args.wall_threshold_m2,
                  defect_threshold_m=args.defect_threshold_m)
    rendered=json.dumps(result,indent=2,allow_nan=False)
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(rendered+"\n",encoding="utf-8")
    print(rendered)
    if not result["model_agreement_instrumented"]:
        print("WARNING: diagnostic source contains no full-step model-agreement instrumentation",flush=True)
    if args.max_accepted_cost_error_p99 is not None:
        import sys
        passed,message=accepted_cost_precision_gate(
            result,max_p99=args.max_accepted_cost_error_p99,
            minimum_samples=args.min_accepted_cost_samples)
        print(("PASS" if passed else "FAIL")+": "+message,file=sys.stderr)
        if not passed:
            raise SystemExit(1)

if __name__=="__main__":
    main()
