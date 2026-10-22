#!/usr/bin/env python3
"""Spend a longer independent-reference budget on hard frozen QPs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

from tools.mpc_qp_oracle import solve


def unresolved_rows(results: list[dict]) -> list[dict]:
    """Return timeout/nonconverged QPs not already proven infeasible."""
    return [row for row in results
            if not row["oracle_converged"] and
            row["oracle_feasibility_lp_status"] != "infeasible"]


def select_refinement_rows(results: list[dict], all_unresolved: bool) -> list[dict]:
    unresolved = unresolved_rows(results)
    if all_unresolved:
        return sorted(unresolved, key=lambda row: (
            row["case_id"], row["fixture"]))

    selected = []
    for case_id in sorted({row["case_id"] for row in results}):
        group = [row for row in unresolved if row["case_id"] == case_id]
        group.sort(key=lambda row: (
            row["oracle_primal_residual"] + row["oracle_dual_residual"]),
            reverse=True)
        # Preserve the original representative-screen behavior by default.
        selected.extend(group[:1])
        if len(group) > 2:
            selected.append(group[len(group) // 2])
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--time-limit-s", type=float, default=5.0)
    parser.add_argument("--max-iter", type=int, default=250_000)
    parser.add_argument(
        "--all-unresolved", action="store_true",
        help="refine every nonconverged fixture not proven infeasible")
    parser.add_argument(
        "--case-id", action="append", default=[],
        help="restrict refinement to this case ID (repeatable)")
    parser.add_argument(
        "--fixture", action="append", default=[],
        help="restrict refinement to this fixture basename (repeatable)")
    args = parser.parse_args()
    corpus = args.corpus.resolve()
    validation_path = (args.validation.resolve() if args.validation else
                       corpus / "validation_summary.json")
    validation = json.loads(validation_path.read_text())
    results = validation["native_recorded_results"]
    selected = select_refinement_rows(results, args.all_unresolved)
    if args.case_id:
        selected_case_ids = set(args.case_id)
        selected = [row for row in selected
                    if row["case_id"] in selected_case_ids]
    if args.fixture:
        selected_fixtures = set(args.fixture)
        selected = [row for row in selected
                    if Path(row["fixture"]).name in selected_fixtures]

    output = args.output.resolve() if args.output else \
        corpus / "oracle_refinement.json"
    records = []
    for row in selected:
        fixture = corpus / row["fixture"]
        oracle = solve(fixture, max_iter=args.max_iter,
                       time_limit_s=args.time_limit_s)
        records.append({
            "case_id": row["case_id"],
            "fixture": row["fixture"],
            "short_budget_status": row["oracle_status"],
            "short_budget_residuals": {
                "primal": row["oracle_primal_residual"],
                "dual": row["oracle_dual_residual"],
            },
            "long_budget_reference": oracle,
        })
        print(f"{row['case_id']} {Path(row['fixture']).name}: "
              f"{oracle['status']} ({oracle['iterations']} iters, "
              f"{oracle['solve_time_s']:.3f}s)", flush=True)

    summary = {
        "schema": "sdu-apex-frozen-qp-reference-refinement-v1",
        "experiment_regime": "identical_qp",
        "source_regime": "saved_state_reconstruction",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "oracle": "OSQP 1.0.4",
        "time_limit_per_fixture_s": args.time_limit_s,
        "iteration_limit_per_fixture": args.max_iter,
        "selection": ("all_unresolved_not_proven_infeasible"
                      if args.all_unresolved else
                      "hardest_and_mid_tail_per_case"),
        "case_id_filter": args.case_id,
        "fixture_filter": args.fixture,
        "selected_count": len(records),
        "records": records,
    }
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"oracle refinement failed: {error}", file=sys.stderr)
        raise SystemExit(2)
