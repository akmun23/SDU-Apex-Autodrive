#!/usr/bin/env python3
"""Run only handoff-prescribed D1/D2/D3 fixed-budget WP26 ablations."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from tools.vehicle_dynamics_learning.trace_wp22_internal_rollout import (
    NEXT_ROOT,
    SOURCE_COMMIT,
    WP25_ROOT,
    _write_json,
)
from tools.vehicle_dynamics_learning.train_augmented_state_space_plant import (
    DEFAULT_OUTPUT as WP22_OUTPUT,
    ROOT,
    SEED,
    STAGE_HORIZONS,
    STAGE_STEPS,
    TASK_ROOT,
    run_wp22,
    sha256_file,
)


WP22_REPORT = WP22_OUTPUT / "wp22_training_report.json"
OUTPUT_ROOT = NEXT_ROOT / "wp26_mechanism_ablations"
FROZEN_STARTS = NEXT_ROOT / "frozen_eval_starts.json"
FROZEN_COMPARATORS = NEXT_ROOT / "frozen_comparators.json"
VARIANTS = {
    "D1": {"support_loss_weight": 0.0, "latent_measurement_feedback": True,
           "description": "A2 architecture; support regularization removed"},
    "D2": {"support_loss_weight": 0.02, "latent_measurement_feedback": False,
           "description": "A2 support objective; generated encoder input removed from latent transition"},
    "D3": {"support_loss_weight": 0.0, "latent_measurement_feedback": False,
           "description": "Both WP22 support objective and generated encoder feedback removed"},
}
REQUIRED_GATES = (
    "openplane_2s_position_improves_at_least_10pct_vs_A2",
    "openplane_2s_yaw_within_5pct_vs_A2",
    "openplane_2s_u_no_more_than_5pct_regression_vs_A2",
    "first_0p5m_divergence_later_on_at_least_4_of_6_validation_runs",
    "no_practice_position_and_yaw_regression_over_20pct_at_same_horizon",
    "all_recursive_outputs_finite",
    "improvement_seen_on_at_least_3_independent_runs",
)


def _git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                          text=True, capture_output=True).stdout.strip()


def _assert_frozen_starts() -> dict[str, Any]:
    if not FROZEN_STARTS.is_file() or not FROZEN_COMPARATORS.is_file():
        raise FileNotFoundError("WP24 frozen comparator/start manifests are required")
    frozen = json.loads(FROZEN_STARTS.read_text(encoding="utf-8"))
    comparators = json.loads(FROZEN_COMPARATORS.read_text(encoding="utf-8"))
    if frozen.get("source_commit") != SOURCE_COMMIT or comparators.get("source_commit") != SOURCE_COMMIT:
        raise RuntimeError("WP26 source commit differs from WP24 freeze")
    for name in ("wp22_A2", "wp19", "wp23_A0", "wp23_A1"):
        item = comparators["comparators"][name]
        if sha256_file(ROOT / item["path"]) != item["sha256"]:
            raise RuntimeError(f"frozen comparator changed: {name}")
    return frozen


def _assert_window_manifest(data, frozen: dict[str, Any]) -> None:
    expected_roles = {
        "development_validation": data.validation_windows,
        "practice_diagnostic": data.practice_windows,
    }
    for role, runs in expected_roles.items():
        manifest_runs = frozen["split_roles"][role]
        if set(manifest_runs) != set(runs):
            raise RuntimeError(f"WP26 run inventory differs from WP24: {role}")
        for run_id, refs in runs.items():
            rows = manifest_runs[run_id]
            actual = {(int(item["sequence_index"]), int(item["source_row"]))
                      for item in rows}
            expected = {(int(sequence), int(row)) for _capture, sequence, row in refs}
            if actual != expected:
                raise RuntimeError(f"WP26 changed the frozen start set: {role}/{run_id}")


def _a2_metric(report: dict[str, Any], role: str, run: str,
               horizon: str, metric: str) -> float:
    return float(report["evaluation"]["splits"][role]["per_run"][run]
                 ["horizons"][horizon]["candidate_macro_window_metrics"][metric])


def _variant_gate(evaluation: dict[str, Any], a2_report: dict[str, Any]) -> dict[str, Any]:
    val = evaluation["splits"]["openplane_validation"]["per_run"]
    a2_val = a2_report["evaluation"]["splits"]["openplane_validation"]["per_run"]
    runs = sorted(val)
    if runs != sorted(a2_val) or len(runs) != 6:
        raise RuntimeError("WP26 requires the same six frozen OpenPlane validation runs")
    candidate_position = {
        run: float(val[run]["horizons"]["2s"]["candidate_macro_window_metrics"]
                   ["position_radial_trajectory_rmse_m"]) for run in runs}
    a2_position = {run: _a2_metric(a2_report, "openplane_validation", run, "2s",
                                  "position_radial_trajectory_rmse_m") for run in runs}
    c_pos = sum(candidate_position.values()) / len(runs)
    a_pos = sum(a2_position.values()) / len(runs)
    candidate_yaw = {
        run: float(val[run]["horizons"]["2s"]["candidate_macro_window_metrics"]
                   ["yaw_rate_rmse_rps"]) for run in runs}
    a2_yaw = {run: _a2_metric(a2_report, "openplane_validation", run, "2s",
                             "yaw_rate_rmse_rps") for run in runs}
    candidate_u = {
        run: float(val[run]["horizons"]["2s"]["candidate_macro_window_metrics"]
                   ["u_rmse_mps"]) for run in runs}
    a2_u = {run: _a2_metric(a2_report, "openplane_validation", run, "2s", "u_rmse_mps")
            for run in runs}

    divergence_later = {}
    for run in runs:
        new = val[run]["horizons"]["5s"]["candidate_first_divergence_0p5m_s"]
        old = a2_val[run]["horizons"]["5s"]["candidate_first_divergence_0p5m_s"]
        # A censored crossing is later than any observed crossing in the 5 s window.
        new_censored = val[run]["horizons"]["5s"]["candidate_first_divergence_censored_windows"]
        old_censored = a2_val[run]["horizons"]["5s"]["candidate_first_divergence_censored_windows"]
        divergence_later[run] = (new_censored > old_censored if new == old else
                                 new is None or (old is not None and new > old))

    practice = evaluation["splits"]["practice_transfer"]["per_run"]
    a2_practice = a2_report["evaluation"]["splits"]["practice_transfer"]["per_run"]
    practice_regressions = {}
    for run in sorted(practice):
        for horizon in ("2s", "5s"):
            cp = practice[run]["horizons"][horizon]["candidate_macro_window_metrics"]
            bp = a2_practice[run]["horizons"][horizon]["candidate_macro_window_metrics"]
            position_change = cp["position_radial_trajectory_rmse_m"] / max(
                bp["position_radial_trajectory_rmse_m"], 1e-9) - 1.0
            yaw_change = cp["yaw_rate_rmse_rps"] / max(bp["yaw_rate_rmse_rps"], 1e-9) - 1.0
            practice_regressions[f"{run}/{horizon}"] = {
                "position_relative_change": float(position_change),
                "yaw_relative_change": float(yaw_change),
                "both_over_20pct": bool(position_change > 0.20 and yaw_change > 0.20),
            }
    finite = all(
        horizon.get("candidate_nonfinite_count", 0) == 0
        for split in evaluation["splits"].values()
        for run in split["per_run"].values()
        for horizon in run["horizons"].values())
    improved_runs = [run for run in runs if candidate_position[run] < a2_position[run]]
    relative_position_improvement = (a_pos - c_pos) / max(a_pos, 1e-9)
    checks = {
        REQUIRED_GATES[0]: relative_position_improvement >= 0.10,
        REQUIRED_GATES[1]: (sum(candidate_yaw.values()) / len(runs)
                            <= 1.05 * sum(a2_yaw.values()) / len(runs)),
        REQUIRED_GATES[2]: (sum(candidate_u.values()) / len(runs)
                            <= 1.05 * sum(a2_u.values()) / len(runs)),
        REQUIRED_GATES[3]: sum(divergence_later.values()) >= 4,
        REQUIRED_GATES[4]: not any(item["both_over_20pct"]
                                   for item in practice_regressions.values()),
        REQUIRED_GATES[5]: finite,
        REQUIRED_GATES[6]: len(improved_runs) >= 3,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "relative_openplane_2s_position_improvement": float(relative_position_improvement),
        "openplane_2s_position_candidate_macro_run_mean_m": c_pos,
        "openplane_2s_position_A2_macro_run_mean_m": a_pos,
        "openplane_2s_yaw_candidate_macro_run_mean_rps": float(sum(candidate_yaw.values()) / len(runs)),
        "openplane_2s_yaw_A2_macro_run_mean_rps": float(sum(a2_yaw.values()) / len(runs)),
        "openplane_2s_u_candidate_macro_run_mean_mps": float(sum(candidate_u.values()) / len(runs)),
        "openplane_2s_u_A2_macro_run_mean_mps": float(sum(a2_u.values()) / len(runs)),
        "first_0p5m_crossing_later_by_run": divergence_later,
        "openplane_position_improved_runs": improved_runs,
        "practice_relative_changes": practice_regressions,
        "independent_openplane_validation_runs": len(runs),
        "independent_practice_runs": len(practice),
        "gate_definition_source": "WP26 handoff, all seven criteria applied against frozen A2",
    }


def _write_variant_package(output: Path, variant: str,
                           report: dict[str, Any], gate: dict[str, Any],
                           dataset_manifest: dict[str, Any]) -> None:
    checkpoint_path = ROOT / report["checkpoint"]
    checkpoint_hash = sha256_file(checkpoint_path)
    if checkpoint_hash != report["checkpoint_sha256"]:
        raise RuntimeError(f"{variant} saved checkpoint hash mismatch")
    (output / "checkpoint.sha256").write_text(checkpoint_hash + "\n", encoding="utf-8")
    (output / "source_commit.txt").write_text(SOURCE_COMMIT + "\n", encoding="utf-8")
    _write_json(output / "dataset_manifest.json", dataset_manifest)
    _write_json(output / "metadata.json", {
        "work_package": "WP26",
        "variant": variant,
        "description": VARIANTS[variant]["description"],
        "source_commit": SOURCE_COMMIT,
        "seed": SEED,
        "training_updates": sum(STAGE_STEPS),
        "stage_updates": list(STAGE_STEPS),
        "stage_horizons_steps": list(STAGE_HORIZONS),
        "stage_horizons_seconds": [step * 0.025 for step in STAGE_HORIZONS],
        "support_loss_weight": VARIANTS[variant]["support_loss_weight"],
        "latent_measurement_feedback": VARIANTS[variant]["latent_measurement_feedback"],
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint": report["checkpoint"],
        "code_file_sha256": {
            path.relative_to(ROOT).as_posix(): sha256_file(path)
            for path in (
                ROOT / "tools/vehicle_dynamics_learning/augmented_state_space_plant.py",
                ROOT / "tools/vehicle_dynamics_learning/train_augmented_state_space_plant.py",
                ROOT / "tools/vehicle_dynamics_learning/run_wp26_mechanism_ablations.py",
            )
        },
        "worktree_dirty_at_training": True,
        "split_roles": ["training", "development_validation", "practice_diagnostic"],
        "production_or_mpc_integration": False,
    })
    _write_json(output / "evaluation_report.json", {
        "work_package": "WP26",
        "variant": variant,
        "source_commit": SOURCE_COMMIT,
        "candidate_checkpoint_sha256": checkpoint_hash,
        "evaluation_vs_wp19_parent": report["evaluation"],
        "mechanism_gate_vs_frozen_A2": gate,
        "test_or_final_test_opened": False,
        "untouched_confirmation": False,
    })


def run(device: str = "cpu") -> dict[str, Any]:
    if _git_head() != SOURCE_COMMIT:
        raise RuntimeError(f"WP26 requires frozen source commit {SOURCE_COMMIT}")
    if not (WP25_ROOT / "wp25_score_only_diagnostics.json").is_file():
        raise RuntimeError("WP25 scoring report must exist before any training")
    if not (WP25_ROOT / "wp25_loss_gradient_attribution.json").is_file():
        raise RuntimeError("WP25 gradient attribution report must exist before any training")
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"refusing to overwrite WP26 outputs: {OUTPUT_ROOT}")
    frozen = _assert_frozen_starts()
    from tools.vehicle_dynamics_learning.train_augmented_state_space_plant import _load_data

    data, _ = _load_data()
    _assert_window_manifest(data, frozen)
    a2 = json.loads(WP22_REPORT.read_text(encoding="utf-8"))
    comparator = json.loads(FROZEN_COMPARATORS.read_text(encoding="utf-8"))
    if sha256_file(WP22_OUTPUT / "seed101_candidate.pt") != comparator[
            "comparators"]["wp22_A2"]["sha256"]:
        raise RuntimeError("frozen A2 baseline hash no longer matches WP24")
    dataset_manifest = {
        "source_commit": SOURCE_COMMIT,
        "split_roles": ["training", "development_validation", "practice_diagnostic"],
        "datasets": comparator["dataset_inputs"],
        "frozen_starts_sha256": sha256_file(FROZEN_STARTS),
        "training_runs": list(a2["training_runs"]),
        "validation_runs": list(a2["validation_runs"]),
        "practice_diagnostic_runs": list(a2["practice_transfer_runs"]),
        "test_and_final_test_opened": False,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    result: dict[str, Any] = {
        "schema_version": 1,
        "work_package": "WP26",
        "source_commit": SOURCE_COMMIT,
        "seed": SEED,
        "stage_updates": list(STAGE_STEPS),
        "stage_horizon_steps": list(STAGE_HORIZONS),
        "same_frozen_starts": True,
        "test_or_final_test_opened": False,
        "simulator_launched": False,
        "production_or_mpc_integration": False,
        "variants": {},
    }
    draw_hashes = []
    for variant, config in VARIANTS.items():
        output = OUTPUT_ROOT / variant
        print(f"WP26 training {variant}: 120 updates; 0.5/2/5 s stages", flush=True)
        report = run_wp22(
            output,
            stage_steps=STAGE_STEPS,
            device=device,
            variant=variant,
            work_package="WP26",
            support_loss_weight=config["support_loss_weight"],
        )
        if report["configuration"]["latent_measurement_feedback"] != config[
                "latent_measurement_feedback"]:
            raise RuntimeError(f"{variant} latent-feedback configuration mismatch")
        draw_hashes.append(report["training_sampler"]["draws_sha256"])
        gate = _variant_gate(report["evaluation"], a2)
        _write_variant_package(output, variant, report, gate, dataset_manifest)
        result["variants"][variant] = {
            "description": config["description"],
            "checkpoint": report["checkpoint"],
            "checkpoint_sha256": report["checkpoint_sha256"],
            "training_report": (output / "training_report.json").relative_to(ROOT).as_posix(),
            "evaluation_report": (output / "evaluation_report.json").relative_to(ROOT).as_posix(),
            "training_sampler_draws_sha256": report["training_sampler"]["draws_sha256"],
            "training_sampler_draw_count": report["training_sampler"]["draw_count"],
            "gate_vs_A2": gate,
        }
    if len(set(draw_hashes)) != 1:
        raise RuntimeError(f"WP26 variants did not use identical sampler draws: {draw_hashes}")
    result["same_sampler_draws_across_variants"] = True
    result["shared_sampler_draws_sha256"] = draw_hashes[0]
    passed = [name for name, item in result["variants"].items()
              if item["gate_vs_A2"]["passed"]]
    result["gate"] = {
        "eligible_variants": passed,
        "selected_variant": min(passed, key=lambda name: result["variants"][name]
                                 ["gate_vs_A2"]["openplane_2s_position_candidate_macro_run_mean_m"])
        if passed else None,
        "status": "candidate_qualified_for_WP27" if passed else "no_variant_passed_stop_A2_family",
        "required_criteria": list(REQUIRED_GATES),
        "next_step": "redesign only the best qualifying mechanism with WP27 truncated simulation training"
        if passed else "stop A2 family; no WP27, optimizer, MPC, or raceline work on these candidates",
    }
    _write_json(OUTPUT_ROOT / "wp26_mechanism_ablation_report.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    report = run(args.device)
    print(json.dumps({"report": str(OUTPUT_ROOT / "wp26_mechanism_ablation_report.json"),
                      "gate": report["gate"],
                      "same_sampler_draws": report["same_sampler_draws_across_variants"]},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
