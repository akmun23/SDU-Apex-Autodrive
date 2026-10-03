#!/usr/bin/env python3
"""Run only the WP23 A0/A1 structural ablations against the frozen A2 result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tools.vehicle_dynamics_learning.train_augmented_state_space_plant import (
    DEFAULT_OUTPUT as WP22_OUTPUT,
    SEED,
    STAGE_STEPS,
    TASK_ROOT,
    run_wp22,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = TASK_ROOT / "wp23_structural_ablations_seed101_v1"
WP22_REPORT = WP22_OUTPUT / "wp22_training_report.json"
WP22_CANDIDATE_SHA256 = "312a3146e0f8a9ec9e860bdd80dd24de5a80883efb23fcd9ac8c17c44abcbde2"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _variant_summary(report: dict[str, Any], report_path: Path,
                     variant: str) -> dict[str, Any]:
    evaluation = report["evaluation"]
    return {
        "variant": variant,
        "work_package": report["work_package"],
        "checkpoint": report["checkpoint"],
        "checkpoint_sha256": report["checkpoint_sha256"],
        "report": str(report_path.relative_to(ROOT)),
        "gate": evaluation["gate"],
        "openplane_validation": evaluation["splits"]["openplane_validation"],
        "practice_transfer_diagnostic": evaluation["splits"]["practice_transfer"],
    }


def run_wp23(output: Path = DEFAULT_OUTPUT, device: str = "cpu") -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite WP23 directory: {output}")
    parent_report = _read_json(WP22_REPORT)
    candidate_path = ROOT / parent_report["checkpoint"]
    if sha256_file(candidate_path) != WP22_CANDIDATE_SHA256:
        raise ValueError("frozen WP22/A2 checkpoint hash changed")
    if parent_report["work_package"] != "WP22" or \
            parent_report["evaluation"]["gate"]["passed"]:
        raise ValueError("WP23 expects the recorded failed seed-101 A2 candidate")
    if parent_report["seed"] != SEED:
        raise ValueError("WP23 seed differs from the frozen WP22 seed")
    output.mkdir(parents=True, exist_ok=False)

    variants = {
        "A2": _variant_summary(parent_report, WP22_REPORT, "A2"),
    }
    for variant in ("A0", "A1"):
        report = run_wp22(output / variant, STAGE_STEPS, device,
                          variant=variant, work_package="WP23")
        report_path = output / variant / f"wp23_{variant}_training_report.json"
        variants[variant] = _variant_summary(report, report_path, variant)

    eligible = [variant for variant, summary in variants.items()
                if summary["gate"]["passed"]]
    selected = min(
        eligible,
        key=lambda variant: variants[variant]["openplane_validation"]
        ["macro_run_metrics"]["2s"]["metrics"]
        ["position_radial_trajectory_rmse_m"]["candidate_macro_run_mean"],
    ) if eligible else None
    report = {
        "schema_version": 1,
        "work_package": "WP23",
        "purpose": "matched, minimal structural ablations after WP22/A2 gate failure",
        "seed": SEED,
        "stage_steps": list(STAGE_STEPS),
        "same_frozen_training_validation_and_practice_splits": True,
        "test_or_final_test_opened": False,
        "future_truth_or_feedback_used_as_rollout_input": False,
        "simulator_launched": False,
        "wp22_a2_candidate_sha256": WP22_CANDIDATE_SHA256,
        "variants": variants,
        "a3_omission_reason": (
            "WP17 selected processed rear-wheel rate as measurement/output only; "
            "its interventions did not establish causal necessity for recursive body state."),
        "gate": {
            "eligible_variants": eligible,
            "selected_variant": selected,
            "status": "selected_for_wp24" if selected else "no_variant_passed_material_gate",
            "next_step": "WP24 fixed validation suite" if selected else
                         "return to state/target/support diagnosis; stop candidate training",
        },
    }
    report_path = output / "wp23_structural_ablation_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    report = run_wp23(args.output, args.device)
    print(json.dumps({"output": str(args.output), "gate": report["gate"]},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
