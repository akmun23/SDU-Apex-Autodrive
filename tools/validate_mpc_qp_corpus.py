#!/usr/bin/env python3
"""Roundtrip, replay, and independently score the frozen MPC QP corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from tools.mpc_qp_fixture import FixtureError, validate_sidecar
from tools.mpc_qp_oracle import solve as solve_oracle


def run_json(command: list[str]) -> dict:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(command)}\n"
            f"{result.stderr[-4000:]}")
    try:
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as error:
        raise RuntimeError(f"invalid JSON from {' '.join(command)}") from error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--replay-binary", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    corpus = args.corpus.resolve()
    binary = args.replay_binary.resolve()
    manifest_path = corpus / "corpus_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("source_regime") != "saved_state_reconstruction" or \
            manifest.get("live_solver_memory_restored") is not False:
        raise FixtureError("corpus provenance incorrectly claims live state")

    records = []
    with tempfile.TemporaryDirectory(prefix="mpc_qp_roundtrip_") as temp:
        roundtrip_dir = Path(temp)
        for case in manifest["cases"]:
            fixture_dir = corpus / case["id"] / "fixtures"
            fixtures = sorted(fixture_dir.glob("qp_*.fqp"))
            if len(fixtures) != case["captured_fixture_count"]:
                raise FixtureError(f"fixture count changed for {case['id']}")
            for fixture in fixtures:
                sidecar = fixture.with_suffix(fixture.suffix + ".json")
                validate_sidecar(fixture, sidecar)
                roundtrip = roundtrip_dir / fixture.name
                subprocess.run([str(binary), "--roundtrip", str(fixture),
                                str(roundtrip)], check=True,
                               stdout=subprocess.DEVNULL)
                if fixture.read_bytes() != roundtrip.read_bytes():
                    raise FixtureError(f"non-identical roundtrip: {fixture}")

                native = run_json([str(binary), str(fixture), "recorded"])
                if not native["recorded_result_exact_parity"]:
                    raise FixtureError(f"saved solver output did not replay: {fixture}")
                oracle = solve_oracle(fixture)
                native["oracle_objective"] = oracle["objective"]
                native["oracle_converged"] = oracle["status"].lower() == "solved"
                native["native_primal_feasible_for_objective_compare"] = max(
                    native["max_primal_dynamics_equality_residual"],
                    native["max_primal_state_bound_violation"],
                    native["max_primal_input_bound_violation"]) <= 1.0e-5
                native["objective_delta_vs_oracle"] = (
                    native["primal_objective"] - oracle["objective"]
                    if native["oracle_converged"] and
                    native["native_primal_feasible_for_objective_compare"]
                    else None)
                native["oracle_primal_residual"] = oracle["primal_residual"]
                native["oracle_dual_residual"] = oracle["dual_residual"]
                native["oracle_status"] = oracle["status"]
                native["oracle_feasibility_lp_status"] = \
                    oracle["feasibility_lp_status"]
                native["oracle_iterations"] = oracle["iterations"]
                native["oracle_solve_time_s"] = oracle["solve_time_s"]
                native["oracle_hessian_min_eigenvalue"] = \
                    oracle["hessian_min_eigenvalue"]
                native["case_id"] = case["id"]
                native["source_outcome"] = case["outcome"]
                native["fixture"] = str(fixture.relative_to(corpus))
                records.append(native)
            print(f"validated {case['id']}: {len(fixtures)} fixtures")

    if len(records) < 100:
        raise FixtureError(f"only {len(records)} fixtures validated; expected >=100")
    if sum(c["outcome"] == "success" for c in manifest["cases"]) < 2 or \
            sum(c["outcome"] == "failure" for c in manifest["cases"]) < 3:
        raise FixtureError("required success/failure case diversity missing")

    # Repeat seed/rho comparisons on deterministically selected fixtures spread
    # across each source run. They remain identical-QP solver experiments.
    variant_records = []
    for case in manifest["cases"]:
        case_records = [r for r in records if r["case_id"] == case["id"]]
        selected = (case_records[0], case_records[len(case_records) // 2],
                    case_records[-1])
        for record in selected:
            fixture = corpus / record["fixture"]
            for seed in ("cold", "primal-only", "dual-reset", "penalty-reset"):
                variant_records.append({
                    "case_id": case["id"],
                    "fixture": record["fixture"],
                    "seed_mode": seed,
                    "rho_policy": "recorded",
                    **run_json([str(binary), str(fixture), seed]),
                })
            for rho_policy in ("fixed", "adaptive"):
                variant_records.append({
                    "case_id": case["id"],
                    "fixture": record["fixture"],
                    "seed_mode": "recorded",
                    "rho_policy": rho_policy,
                    **run_json([str(binary), str(fixture), "recorded", rho_policy]),
                })

    summary = {
        "schema": "sdu-apex-frozen-qp-validation-v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "experiment_regime": "identical_qp",
        "source_regime": "saved_state_reconstruction",
        "live_solver_memory_restored": False,
        "fixture_count": len(records),
        "source_case_count": len(manifest["cases"]),
        "roundtrip_byte_identical_count": len(records),
        "recorded_solver_exact_parity_count": sum(
            item["recorded_result_exact_parity"] for item in records),
        "oracle_status_counts": {
            status: sum(item["oracle_status"] == status for item in records)
            for status in sorted({item["oracle_status"] for item in records})
        },
        "oracle_converged_count": sum(item["oracle_converged"]
                                       for item in records),
        "native_primal_feasible_and_oracle_solved_count": sum(
            item["objective_delta_vs_oracle"] is not None for item in records),
        "oracle_budget_per_fixture_s": 0.2,
        "native_objective_comparison_feasibility_tolerance": 1.0e-5,
        "oracle_primal_residual_max": max(
            item["oracle_primal_residual"] for item in records),
        "oracle_dual_residual_max": max(
            item["oracle_dual_residual"] for item in records),
        "native_primal_objective_delta_abs_max": max(
            (abs(item["objective_delta_vs_oracle"]) for item in records
             if item["objective_delta_vs_oracle"] is not None), default=None),
        "oracle_infeasibility_confirmed_by_linprog_count": sum(
            item["oracle_feasibility_lp_status"] == "infeasible"
            for item in records),
        "native_recorded_results": records,
        "seed_and_rho_comparisons": variant_records,
        "interpretation": (
            "The QP and solver seed in each fixture are exact snapshots of the "
            "saved-state replay. Seed/rho variants hold that QP fixed. They do "
            "not establish that live bag ADMM memory was recorded or restored."),
    }
    output = args.output.resolve() if args.output else corpus / "validation_summary.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: summary[key] for key in (
        "fixture_count", "roundtrip_byte_identical_count",
        "recorded_solver_exact_parity_count", "oracle_primal_residual_max",
        "oracle_dual_residual_max", "native_primal_objective_delta_abs_max")},
        sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, FixtureError, RuntimeError, subprocess.SubprocessError,
            json.JSONDecodeError) as error:
        print(f"frozen-QP validation failed: {error}", file=sys.stderr)
        raise SystemExit(2)
