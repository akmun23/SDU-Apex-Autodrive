#!/usr/bin/env python3
"""Source-only, branch-aware MPC derivative fidelity audit.

Checks both (+) and (-) original-source control perturbations, NOT a
symmetric surrogate gradient. The tool compares the full actual racing
objective + original physical-constraint merit against the selected
active-branch source AD directional differential. It must not be used
as a claim that Unity's private contact dynamics match our source model.

A failure does not invalidate exact nonlinear cost evaluation. It means a
local source-branch derivative may be misleading and MUST NOT be used as
a sufficient acceptance or rejection criterion for a finite MPC step.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "live_runs/mpc_source_second_order_20261104_01"
PROBE = ROOT / "build/f1tenth_mpc/mpc_source_exact_merit_probe"
CASES = ("case_15", "case_16", "case_live_fast_17")


def evaluate(name: str, step: float, rho: float) -> dict:
    result = subprocess.run(
        [
            str(PROBE), str(FIXTURES / name), "-", str(rho),
            f"{step:.12g}",
        ],
        cwd=ROOT, capture_output=True, text=True, timeout=120,
    )
    if result.returncode:
        raise RuntimeError(
            f"{name} h={step}: {result.stderr[-1200:]}"
        )
    return json.loads(result.stdout)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=float, nargs="+",
                        default=(3e-4, 3e-5))
    parser.add_argument("--rho", type=float, default=2e6)
    parser.add_argument("--maximum-scaled-error", type=float, default=.05)
    parser.add_argument("--report", type=Path,
                        default=ROOT / "build/source_directional_accuracy.json")
    args = parser.parse_args()
    if not PROBE.is_file():
        raise RuntimeError("Build target mpc_source_exact_merit_probe first.")
    report: dict = {
        "schema": "source-branch-directional-accuracy-qualification/v1",
        "scope": "independent source-model one-sided physical control perturbations",
        "original_source_nonlinear_score_is_authoritative": True,
        "does_not_establish_Unity_equivalence": True,
        "no_QP_cost_model_used": True,
        "maximum_allowed_scaled_error": args.maximum_scaled_error,
        "samples": {},
    }
    for name in CASES:
        case = {}
        for h in args.steps:
            result = evaluate(name, h, args.rho)
            rows = result["one_sided_directional_checks"]
            # Require stable, material finite source changes: otherwise
            # float output quantization makes any relative derivative
            # comparison ill-conditioned.
            material = [
                x for x in rows
                if abs(x["source_nonlinear_actual_merit_change"]) >= 1.
            ]
            negative_predictions_positive_actual = [
                x for x in material
                if x["source_active_branch_directional_change"] < 0.
                and x["source_nonlinear_actual_merit_change"] > 0.
            ]
            positive_predictions_negative_actual = [
                x for x in material
                if x["source_active_branch_directional_change"] > 0.
                and x["source_nonlinear_actual_merit_change"] < 0.
            ]
            # Exact original inner-tire and contact branch decisions are
            # independently compared for BOTH physical trajectories.
            # A finite secant crossing a real branch is not evidence
            # that a local branch derivative's algebra is wrong.
            source_branch_stable = [
                x for x in material
                if x.get("source_first_physical_branch_change_stage", -1) < 0
            ]
            source_branch_crossing = [
                x for x in material
                if x.get("source_first_physical_branch_change_stage", -1) >= 0
            ]
            source_float_sub_ulp = sum(
                x.get("original_source_float_sub_ulp_coordinates", 0)
                for x in material
            )
            full_source_valid = [
                x for x in material
                if x.get("physical_source_one_sided_directional_available",
                         False)
            ]
            full_source_worst = max(
                (x["physical_source_one_sided_scaled_error"]
                 for x in full_source_valid),
                default=None,
            )
            physically_stable_wrong_sign = sum(
                x["source_active_branch_directional_change"] *
                x["source_nonlinear_actual_merit_change"] < 0
                for x in source_branch_stable
            )
            branch_crossing_wrong_sign = sum(
                x["source_active_branch_directional_change"] *
                x["source_nonlinear_actual_merit_change"] < 0
                for x in source_branch_crossing
            )
            worst = sorted(
                material, key=lambda x: x["scaled_change_error"],
                reverse=True,
            )[:5]
            worst_error = max(
                (r["scaled_change_error"] for r in material),
                default=0.,
            )
            case[f"{h:.12g}"] = {
                "all_one_sided_source_trials": len(rows),
                "material_source_trials": len(material),
                "original_source_merit_primal_parity_error":
                    result["source_merit_adjoint_primal_parity"],
                "worst_material_scaled_derivative_error": worst_error,
                "material_wrong_sign_local_predictions":
                    len(negative_predictions_positive_actual) +
                    len(positive_predictions_negative_actual),
                "original_physical_branch_stable_trials":
                    len(source_branch_stable),
                "original_physical_branch_crossing_trials":
                    len(source_branch_crossing),
                "branch_stable_wrong_sign_predictions":
                    physically_stable_wrong_sign,
                "branch_crossing_wrong_sign_predictions":
                    branch_crossing_wrong_sign,
                "original_float_sub_half_ulp_state_changes":
                    source_float_sub_ulp,
                "complete_source_one_sided_available_trials":
                    len(full_source_valid),
                "complete_source_one_sided_worst_scaled_error":
                    full_source_worst,
                "source_branch_stable_worst_scaled_error": max(
                    (x["scaled_change_error"]
                     for x in source_branch_stable), default=None,
                ),
                "top_disagreements": worst,
                "qualified": (
                    bool(material)
                    # Full accuracy certificate requires ALL finite
                    # material trials to stay in the same ORIGINAL
                    # physical source branch AND within tolerance.
                    and not source_branch_crossing
                    and source_float_sub_ulp == 0
                    and worst_error <= args.maximum_scaled_error
                    and not negative_predictions_positive_actual
                    and not positive_predictions_negative_actual
                    and abs(result["source_merit_adjoint_primal_parity"]) < 1e-7
                ),
            }
        report["samples"][name] = case
    report["derivative_fidelity_qualified"] = all(
        cell["qualified"]
        for per_case in report["samples"].values()
        for cell in per_case.values()
    )
    report["production_controller_promotion_qualified"] = False
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    for name, trials in report["samples"].items():
        for h, result in trials.items():
            print(
                f"{name} h={h}: material={result['material_source_trials']} "
                f"worst_error={result['worst_material_scaled_derivative_error']:.6f} "
                f"wrong_sign={result['material_wrong_sign_local_predictions']} "
                f"source_switching={result['original_physical_branch_crossing_trials']} "
                f"float_sub_ulp={result['original_float_sub_half_ulp_state_changes']} "
                f"{'PASS' if result['qualified'] else 'FAIL'}"
            )
    print("Directional source-derivative qualification:",
          "PASS" if report["derivative_fidelity_qualified"] else "FAIL")
    # The gate intentionally fails until precision really meets the
    # threshold, rather than quietly treating source cost parity as
    # proof of control-gradient fidelity.
    return 0 if report["derivative_fidelity_qualified"] else 1


if __name__ == "__main__":
    sys.exit(main())
