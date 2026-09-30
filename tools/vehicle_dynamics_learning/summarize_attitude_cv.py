#!/usr/bin/env python3
"""Pair no-attitude and roll-conditioned grouped-CV reports by whole run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_structured_body_cv import _aggregate_cv, _paired_comparisons


def summarize(base_path: Path, attitude_path: Path, output_path: Path,
              bootstrap_count: int = 2000, seed: int = 20260937) -> dict:
    base = json.loads(base_path.read_text(encoding="utf-8"))
    attitude = json.loads(attitude_path.read_text(encoding="utf-8"))
    if base.get("run_to_fold") != attitude.get("run_to_fold"):
        raise ValueError("CV reports do not use identical whole-run folds")
    if base.get("run_to_family") != attitude.get("run_to_family"):
        raise ValueError("CV reports do not use identical run-family groups")
    if "gru" not in base.get("per_model_per_run", {}):
        raise ValueError("base report has no plain-GRU comparator")
    if "gru" not in attitude.get("per_model_per_run", {}):
        raise ValueError("attitude report has no matched GRU comparator")

    models = {"gru": base["per_model_per_run"]["gru"],
              "gru_with_attitude": attitude["per_model_per_run"]["gru"]}
    for architecture, runs in attitude["per_model_per_run"].items():
        if architecture != "gru":
            models[f"{architecture}_with_attitude"] = runs
    paired = _paired_comparisons(models, bootstrap_count, seed)
    result = {
        "method": "paired outer-fold held-run scores; run is the uncertainty unit",
        "base_report": str(base_path),
        "attitude_report": str(attitude_path),
        "eligible_whole_runs": len(base["run_to_fold"]),
        "fold_count": base["fold_count"],
        "bootstrap_count": bootstrap_count,
        "attitude_input_policy": (
            "roll, pitch, roll-rate and pitch-rate are available from the 16-step "
            "observed history, then held at their last observed values throughout "
            "the prediction horizon; no future attitude samples are used"),
        "base_without_attitude": _aggregate_cv(
            {"gru": models["gru"]}, bootstrap_count, seed + 1),
        "attitude_conditioned_models": _aggregate_cv(
            {name: rows for name, rows in models.items() if name != "gru"},
            bootstrap_count, seed + 2),
        "paired_change_vs_plain_gru": paired,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True,
                                      allow_nan=False) + "\n",
                           encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_report", type=Path)
    parser.add_argument("attitude_report", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = summarize(args.base_report, args.attitude_report, args.output)
    print(json.dumps({"output": str(args.output),
                      "paired_models": list(result["paired_change_vs_plain_gru"]),
                      "whole_runs": result["eligible_whole_runs"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
