#!/usr/bin/env python3
"""Fit one offline plant per regime and cross-score every regime.

This tool implements the regime test required by the full-speed recovery
handoff.  It deliberately keeps the fitted candidates offline-only: no
parameters are copied to the runtime MPC and no simulator truth enters a ROS
node.  Every recursive score starts from a measured origin and then uses only
the candidate state, applied commands, and source ``dt``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import fit_simulator_native_model as native


REGIME_PREFIXES: dict[str, tuple[str, ...]] = {
    # Static repeatability is retained as a low-demand reference, but it has
    # no dynamic origins by itself. Include straight replay so this regime
    # remains fit-able while keeping the static samples in its score domain.
    "low_demand": ("10_static_repeatability_run", "30_straight_replay_run"),
    "straight": (
        "01_longitudinal_throttle_from_rest_",
        "30_straight_replay_run",
        "30_zero_throttle_coast_map_",
    ),
    "corner_no_throttle": (
        "03_steering_response_multispeed_",
        "04_steering_response_single_regime_",
        "40_steering_replay_run",
        "50_steering_multisine_",
        "70_d1_d2_steady_corner_",
        "75_d3_lateral_envelope_",
    ),
    "corner_throttle": (
        "20_combined_throttle_steering_run",
        "60_e1_corner_throttle_",
        "61_e2_steering_acceleration_",
        "62_e3_steering_reversal_",
    ),
}


def _discover_groups(root: Path) -> dict[str, list[str]]:
    names = sorted(path.name for path in root.iterdir() if path.is_dir())
    groups: dict[str, list[str]] = {}
    for regime, prefixes in REGIME_PREFIXES.items():
        selected = [
            name for name in names
            if any(name.startswith(prefix) for prefix in prefixes)
        ]
        if not selected:
            raise ValueError(f"no accepted runs found for regime {regime}")
        groups[regime] = selected
    # Regime definitions are intentionally allowed to overlap.  For example,
    # a straight replay is both a low-demand reference and a straight-run
    # score domain; forcing a disjoint partition would remove useful coverage.
    return groups


def _score_summary(scores: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for horizon in ("0.50s", "0.75s"):
        horizon_scores = scores.get(horizon, {})
        position = horizon_scores.get("position_m", {})
        result[horizon] = {
            "origins": int(position.get("count", 0)),
            "p95_m": position.get("p95"),
            "p99_m": position.get("p99"),
            "max_m": position.get("max"),
        }
    return result


def _fit_regime(
        runs: dict[str, list[dict[str, float]]],
        contact_model: str,
        parameter_profile: str,
        fit_method: str,
        mixed_fit_origins: int,
        mixed_fit_max_nfev: int,
) -> tuple[Any, dict[str, Any]]:
    base = native._profile_factory(parameter_profile)(contact_model)
    if fit_method == "mixed_recursive":
        return native._fit_mixed_parameters(
            runs, contact_model, base, mixed_fit_origins,
            mixed_fit_max_nfev)
    return native._fit_parameters(runs, contact_model, base)


def analyze(
        accepted_root: Path,
        output: Path,
        contact_model: str = "moment_basis_split",
        parameter_profile: str = "f1tenth_prefab",
        fit_method: str = "derivative",
        max_origins_per_run: int = 20,
        mixed_fit_origins_per_regime: int = 35,
        mixed_fit_max_nfev: int = 18,
) -> dict[str, Any]:
    groups = _discover_groups(accepted_root)
    loaded = {name: native._read_runs(accepted_root, names)
              for name, names in groups.items()}
    fitted: dict[str, tuple[Any, dict[str, Any]]] = {}
    for regime, runs in loaded.items():
        fitted[regime] = _fit_regime(
            runs, contact_model, parameter_profile, fit_method,
            mixed_fit_origins_per_regime, mixed_fit_max_nfev)

    matrix: dict[str, dict[str, Any]] = {}
    fit_reports: dict[str, Any] = {}
    for fit_regime, (parameters, fit_report) in fitted.items():
        fit_reports[fit_regime] = fit_report
        matrix[fit_regime] = {}
        for score_regime, runs in loaded.items():
            recursive = native._recursive_scores(
                runs, parameters, max_origins_per_run)
            one_step = native._one_step_scores(runs, parameters)
            matrix[fit_regime][score_regime] = {
                "one_step": {
                    "position_p95_m": one_step["position_m"]["p95"],
                    "u_p95_mps": one_step["u_mps"]["p95"],
                    "v_p95_mps": one_step["v_mps"]["p95"],
                    "r_p95_radps": one_step["r_radps"]["p95"],
                },
                "recursive": _score_summary(recursive),
                "origins_by_run": recursive["evaluation"]["origins_used_per_run"],
            }

    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "regime_specific_cross_score_offline_only",
        "acceptance": "not_accepted",
        "ground_truth_use": "offline_identification_and_scoring_only",
        "recursive_prediction_uses_future_gt": False,
        "contact_model": contact_model,
        "parameter_profile": parameter_profile,
        "fit_method": fit_method,
        "accepted_root": str(accepted_root),
        "regimes": groups,
        "fit_reports": fit_reports,
        "fit_regime_to_score_regime": matrix,
        "evaluation": {
            "max_origins_per_run": max_origins_per_run,
            "horizons_s": [0.50, 0.75],
            "source_time_used_for_replay": True,
            "future_measured_states_used_for_rollout": False,
        },
        "selection": {
            "selected_candidate": None,
            "reason": (
                "Regime cross-score is diagnostic only. A production plant "
                "still requires blind validation, native parity, observer "
                "gates, and full-speed runtime acceptance."),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--fit-method", choices=("derivative", "mixed_recursive"),
        default="derivative",
        help="causal derivative fit by default; recursive fit is an explicit long run")
    parser.add_argument("--max-origins-per-run", type=int, default=20)
    parser.add_argument("--mixed-fit-origins-per-regime", type=int, default=35)
    parser.add_argument("--mixed-fit-max-nfev", type=int, default=18)
    args = parser.parse_args()
    if args.max_origins_per_run < 1:
        raise ValueError("--max-origins-per-run must be positive")
    report = analyze(
        args.accepted_root, args.output,
        fit_method=args.fit_method,
        max_origins_per_run=args.max_origins_per_run,
        mixed_fit_origins_per_regime=args.mixed_fit_origins_per_regime,
        mixed_fit_max_nfev=args.mixed_fit_max_nfev,
    )
    print(json.dumps({
        "output": str(args.output),
        "status": report["status"],
        "acceptance": report["acceptance"],
        "regimes": {name: len(runs) for name, runs in report["regimes"].items()},
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
