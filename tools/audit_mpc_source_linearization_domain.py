#!/usr/bin/env python3
"""Check whether an SQP step stays in the locally differentiated source regime.

Pure diagnostic. Reintegrates the original 30-stage source dynamics without
changing commands, vehicle physics, derivatives, constraints, or the optimizer.
The recorded SQP trace is used only to select the already-generated trial.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tools import competition_failure_benchmark as benchmark
from tools.qualify_mpc_source_phase1 import CONTROL_SCALE, SourceOracle


def path_error_decomposition(parameters, values, first, second, groups) -> dict:
    """Piecewise-quadratic path identity, with estimated midpoint jumps.

    Each half interval uses its endpoint's value/first/second jet. Their
    midpoint mismatch estimates value and derivative jumps. On a smooth
    interval it instead contains truncation/roundoff. Only refinement can
    distinguish these; branch event counts alone do not establish causality.
    All returned contributions have the units of the supplied values.
    """
    t = np.asarray(parameters, dtype=float)
    g, p, q = (np.asarray(a, dtype=float) for a in (values, first, second))
    if (t.ndim != 1 or len(t) < 2 or t[0] != 0. or t[-1] != 1. or
            np.any(np.diff(t) <= 0.) or g.ndim != 2 or
            g.shape != p.shape or g.shape != q.shape or len(g) != len(t) or
            len(groups) != len(t)-1 or
            not all(np.all(np.isfinite(a)) for a in (t, g, p, q))):
        raise ValueError("invalid constraint path jets")
    smooth = np.zeros(g.shape[1])
    jumps = {}
    counts = {}
    for i, label in enumerate(groups):
        a, b = t[i:i+2]
        h, midpoint = b-a, .5*(a+b)
        smooth += ((midpoint-a)*(1.-.5*(a+midpoint))*q[i] +
                   (b-midpoint)*(1.-.5*(midpoint+b))*q[i+1])
        value_jump = g[i+1]-g[i]-.5*h*(p[i+1]+p[i]) + .125*h*h*(q[i+1]-q[i])
        derivative_jump = p[i+1]-p[i]-.5*h*(q[i+1]+q[i])
        label = label or "unresolved_without_observed_branch_change"
        jumps.setdefault(label, np.zeros(g.shape[1]))
        jumps[label] += value_jump+(1.-midpoint)*derivative_jump
        counts[label] = counts.get(label, 0)+1
    residual = g[-1]-g[0]-p[0]
    reconstructed = smooth + sum(jumps.values(), np.zeros_like(smooth))
    return {"smooth_curvature_integral": smooth,
            "estimated_switch_and_unresolved_contributions": jumps,
            "interval_counts": counts,
            "nonlinear_remainder": residual,
            # This checks the identity only, NOT the accuracy of attribution.
            "identity_closure_inf": float(np.max(np.abs(residual-reconstructed)))}


def branch_group(score: dict) -> str:
    branch = score["physical_branch_changes"]
    names = []
    if branch["mismatched_substep_schedule_events"]:
        names.append("substep_schedule")
    if any(branch[k] for k in (
            "combined_slip_saturation_changes", "longitudinal_spline_region_changes",
            "lateral_spline_region_changes")):
        names.append("tire_slip")
    if score["geometry_branch_changes"]:
        names.append("reference")
    # Retain all other source events without assuming they are smooth.
    handled = {"mismatched_substep_schedule_events", "combined_slip_saturation_changes",
               "longitudinal_spline_region_changes", "lateral_spline_region_changes", "valid"}
    if any(v for k, v in branch.items() if k not in handled):
        names.append("other_physical")
    return names[0] if len(names) == 1 else ("mixed:"+"+".join(names) if names else "")


def audit(case: Path, tools: Path, iteration: int, fractions: list[float],
          path_partitions: list[int] | None = None) -> dict:
    replay = benchmark.read_json(case / "replay.json.gz")
    traces = replay.get("primal_trace", [])
    if not (0 <= iteration < len(traces)):
        raise ValueError("requested iteration is absent from the original primal trace")
    trace = traces[iteration]
    if not trace.get("trial_evaluated", False):
        raise ValueError("the selected iteration has no exact-source full-step trial")
    original = np.asarray(trace["warm_start"]["controls"], dtype=np.float32)
    trial = np.asarray(trace["trial"]["controls"], dtype=np.float32)
    if original.shape != (30, 2) or trial.shape != (30, 2):
        raise ValueError("original source control sequence must be 30 by 2")
    step = (trial.astype(float) - original.astype(float)).ravel() / CONTROL_SCALE
    if not np.all(np.isfinite(step)):
        raise ValueError("non-finite step")
    oracle = SourceOracle(tools, case)
    try:
        physical_direction = trial.astype(float) - original.astype(float)
        baseline = oracle.evaluate(
            original.tolist(), constraint_direction=physical_direction.tolist())
        if not baseline.get("valid"):
            raise ValueError("base source state could not be integrated")
        margins = np.asarray(baseline["hard_margins"], dtype=float)
        jacobian = np.asarray(baseline["hard_jacobian_scaled"], dtype=float)
        gradient = np.asarray(baseline["gradient_scaled"], dtype=float)
        if margins.shape != (480,) or jacobian.shape != (480, 60):
            raise ValueError("unexpected original physical-constraint layout")
        footprint = np.asarray([kind == "body" for kind in baseline["row_types"]])
        if np.count_nonzero(footprint) != 240:
            raise ValueError("original source footprint must contain 240 corner inequalities")
        # This is the original source second-order path derivative,
        # not a fitted curvature from an accepted/rejected SQP score.
        # It describes ONE branch at alpha=0, not an arbitrary finite
        # trajectory crossing source substep, tire or reference switches.
        path = baseline.get("constraint_path_model", {})
        path_valid = bool(path.get("valid", False))
        first_error = None
        objective_first = objective_second = None
        path_first = path_second = None
        if path_valid:
            path_first = np.asarray(path["first"], dtype=float)
            path_second = np.asarray(path["second"], dtype=float)
            if path_first.shape != margins.shape or path_second.shape != margins.shape:
                raise ValueError("source path-curvature row layout differs from original physics")
            first_error = float(np.max(np.abs(path_first-jacobian @ step)))
            if not all(np.all(np.isfinite(a)) for a in (path_first, path_second)):
                raise ValueError("nonfinite source path model")
            objective_first = path.get("original_objective_first")
            objective_second = path.get("original_objective_second")
        results = []
        for alpha in sorted(set([0., 1., *fractions])):
            if not (0. <= alpha <= 1.):
                raise ValueError("fractions must lie in the closed unit interval")
            candidate = (original.astype(float) +
                         alpha * (trial.astype(float) - original.astype(float)))
            score = oracle.evaluate(
                candidate.astype(np.float32).tolist(),
                reference_controls=original.tolist())
            if not score.get("valid"):
                raise ValueError(f"source rollout invalid at fraction {alpha}")
            actual = np.asarray(score["hard_margins"], dtype=float)
            prediction = margins + jacobian @ ((
                candidate.astype(np.float32).astype(float) -
                original.astype(float)).ravel() / CONTROL_SCALE)
            discrepancy = actual - prediction
            worst = int(np.flatnonzero(footprint)[np.argmax(np.abs(discrepancy[footprint]))])
            branch = score["physical_branch_changes"]
            exact_same_source_branches = (
                not any(int(value) != 0 for value in branch.values()) and
                int(score["geometry_branch_changes"]) == 0 and
                int(score["peak_branch_changes"]) == 0)
            quadratic = None
            if path_valid:
                quadratic = margins + alpha*path_first + .5*alpha*alpha*path_second
            results.append({
                "original_source_second_order_model_valid_at_nominal":
                    path_valid,
                # A locally valid second-order model is not a globally
                # certified model if the original source switches mode.
                "source_branch_identity_preserved_to_candidate":
                    exact_same_source_branches,
                "second_order_prediction_qualified_at_candidate":
                    path_valid and exact_same_source_branches,
                "quadratic_source_minimum_margin_m":
                    float(np.min(quadratic[footprint])) if quadratic is not None else None,
                "maximum_absolute_quadratic_constraint_model_error_m":
                    float(np.max(np.abs((quadratic-actual)[footprint])))
                    if quadratic is not None else None,
                "fraction": alpha,
                "original_racing_cost": score["original_cost"],
                "affine_racing_cost_error": float(score["original_cost"]-
                    baseline["original_cost"]-alpha*(gradient @ step)),
                "quadratic_racing_cost_error": float(score["original_cost"]-
                    baseline["original_cost"]-alpha*objective_first-.5*alpha*alpha*objective_second)
                    if objective_first is not None and objective_second is not None else None,
                "original_source_minimum_margin_m": float(actual[footprint].min()),
                "affine_source_minimum_margin_m": float(prediction[footprint].min()),
                "maximum_absolute_constraint_model_error_m":
                    float(np.max(np.abs(discrepancy[footprint]))),
                "worst_row": worst,
                "worst_row_type": baseline["row_types"][worst],
                "source_substep_schedule_changed_events":
                    branch["mismatched_substep_schedule_events"],
                "source_combined_slip_changed_events":
                    branch["combined_slip_saturation_changes"],
                "source_tire_curve_region_changed_events":
                    branch["longitudinal_spline_region_changes"] +
                    branch["lateral_spline_region_changes"],
                "reference_interpolation_changed_events":
                    score["geometry_branch_changes"],
                "source_peak_active_substep_changed_events":
                    score["peak_branch_changes"],
            })
        decompositions = []
        for partitions in sorted(set(path_partitions or [])):
            if not (2 <= partitions <= 4096):
                raise ValueError("path partitions must lie between 2 and 4096")
            values, first, second, groups = [], [], [], []
            previous = None
            invalid = None
            for t in np.linspace(0., 1., partitions+1):
                controls = (original.astype(float)+t*physical_direction).astype(np.float32).tolist()
                options = {"constraint_direction": physical_direction.tolist()}
                if previous is not None:
                    options["reference_controls"] = previous
                score = oracle.evaluate(controls, **options)
                model = score.get("constraint_path_model", {})
                if not score.get("valid") or not model.get("valid"):
                    invalid = {"parameter": float(t), "path_model": model}
                    break
                values.append(np.asarray(score["hard_margins"])[footprint])
                first.append(np.asarray(model["first"])[footprint])
                second.append(np.asarray(model["second"])[footprint])
                if previous is not None:
                    groups.append(branch_group(score))
                previous = controls
            if invalid is not None:
                decompositions.append({"partitions": partitions, "valid": False, "invalid": invalid})
                continue
            parts = path_error_decomposition(np.linspace(0., 1., partitions+1),
                                              values, first, second, groups)
            affine_remainder = np.asarray(values[-1])-values[0]-(jacobian @ step)[footprint]
            row = int(np.argmax(np.abs(affine_remainder)))
            def summarize(vector):
                return {"at_worst_affine_footprint_row_m": float(vector[row]),
                        "inf_m": float(np.max(np.abs(vector)))}
            decompositions.append({
                "partitions": partitions, "valid": True,
                "worst_affine_footprint_row": int(np.flatnonzero(footprint)[row]),
                "affine_model_error": summarize(affine_remainder),
                "one_sided_first_derivative_correction": summarize(
                    np.asarray(first[0])-(jacobian @ step)[footprint]),
                "smooth_curvature_integral": summarize(parts["smooth_curvature_integral"]),
                "estimated_switch_and_unresolved_contributions": {
                    label: summarize(vector) for label, vector in
                    parts["estimated_switch_and_unresolved_contributions"].items()},
                "interval_counts": parts["interval_counts"],
                "identity_closure_inf_m": parts["identity_closure_inf"],
            })
        return {
            "schema": "original-source-local-linearization-domain-audit/v2",
            "clearance_and_model_error_rows": "240 footprint inequalities only; metres",
            "case": str(case),
            "recorded_iteration": iteration,
            "full_step_max_abs_normalized_command_change":
                float(np.max(np.abs(step))),
            "full_step_euclidean_normalized_command_change":
                float(np.linalg.norm(step)),
            "original_hard_inequality_count": len(margins),
            "original_racing_gradient_norm_scaled":
                float(np.linalg.norm(gradient)),
            "normal_tangent_derivative_is_a_local_quantity": True,
            "path_curvature_first_derivative_agreement_inf": first_error,
            "original_objective_directional_first": objective_first,
            "original_objective_directional_second": objective_second,
            "objective_directional_gradient_agreement_abs":
                abs(float(gradient @ step)-objective_first) if objective_first is not None else None,
            "path_curvature_selected_source_branch_valid": path_valid,
            "path_curvature_switch_ties":
                path.get("physical_switch_ties", 0),
            "path_curvature_contact_switch_ties":
                path.get("contact_switches", 0),
            "path_curvature_discrete_substep_switch_ties":
                path.get("substep_switches", 0),
            "fractions": results,
            "path_decompositions": decompositions,
            "attribution_limits": (
                "Midpoint jet extrapolation estimates jumps; refine partitions before interpreting. "
                "Mixed intervals cannot uniquely attribute individual switches. Equal endpoint "
                "branches do not exclude interior switches. Float32 commands add roundoff. "
                "Identity closure validates bookkeeping, not causal attribution."),
            "physical_dynamics_modified": False,
            "alternative_controls_generated": False,
            "oracle_evaluations": oracle.calls,
        }
    finally:
        oracle.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--tools", type=Path, default=benchmark.DEFAULT_TOOLS)
    parser.add_argument("--iteration", type=int, default=0)
    parser.add_argument("--fractions", nargs="*", type=float,
                        default=[0.001, .003, .01, .03, .1, .2, .4, .6, .8])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--path-partitions", nargs="*", type=int, default=[],
                        help="refined path-jet decompositions, e.g. 32 64 128")
    args = parser.parse_args()
    report = audit(args.case.resolve(), args.tools.resolve(),
                   args.iteration, args.fractions, args.path_partitions)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
