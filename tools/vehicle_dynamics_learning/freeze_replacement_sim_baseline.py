#!/usr/bin/env python3
"""Freeze the evidence and source artifacts named by the replacement-sim handoff."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = Path("live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001")
OUTPUT = ROOT / RUN_ROOT / "replacement_sim_baseline_registry_20261002.json"
LEGACY_REGISTRY = ROOT / RUN_ROOT / "baseline_registry.json"
HANDOFF = Path(
    "/home/akselmo/Downloads/SDU_APEX_REPLACEMENT_OFFLINE_SIM_AND_RACELINE_OPTIMIZER_HANDOFF_2026-10-02.md")

FILES: dict[str, str] = {
    "legacy_baseline_registry": "live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/baseline_registry.json",
    "practice_benchmark": "live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/practice_transfer_benchmark_v1.json",
    "practice_mpc0_existing_report": "live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/production_mpc_practice_model_v1.json",
    "braking_fixtures": "live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/braking_wheel_regression_fixtures_v1.json",
    "practice_r02_bag": "live_runs/practice_model_validation_r02/run/run_0.db3",
    "practice_r02_validation": "live_runs/practice_model_validation_r02/practice_capture_validation_final.json",
    "practice_r03_bag": "live_runs/practice_model_validation_r03/run/run_0.db3",
    "practice_r03_validation": "live_runs/practice_model_validation_r03/practice_capture_validation_final.json",
    "practice_r02_dataset": "live_runs/derived_dynamics_learning_20260928/practice_transfer_validation_20261001_r02/openplane_dynamics.npz",
    "practice_r03_dataset": "live_runs/derived_dynamics_learning_20260928/practice_transfer_validation_20261001_r03/openplane_dynamics.npz",
    "frontier_validation_bag": "live_runs/openplane_steering_frontier_validation_r03/run/run_0.db3",
    "frontier_validation_manifest": "live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/steering_frontier_validation_r03_20261002/manifest.json",
    "frontier_validation_dataset": "live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/steering_frontier_validation_r03_20261002/openplane_dynamics.npz",
    "frontier_rssm_report": "live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/steering_frontier_validation_r03_20261002/rssm_h256_baseline_free_run.json",
    "highsteer_train_bag": "live_runs/openplane_highsteer_75_train_r02/run/run_0.db3",
    "highsteer_validation_r01_bag": "live_runs/openplane_highsteer_75_validation_r01/run/run_0.db3",
    "highsteer_validation_r02_bag": "live_runs/openplane_highsteer_75_validation_r02/run/run_0.db3",
    "highsteer_train_dataset": "live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/highsteer_75_train_source_20261002/openplane_dynamics.npz",
    "highsteer_validation_dataset": "live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/highsteer_75_heldout_validation_source_20261002/openplane_dynamics.npz",
    "highsteer_rssm_candidate": "live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/highsteer_75_rssm_candidate_familyfix_20261002/c2_h256_z32_seed17035/best_rssm_teacher.pt",
    "dynamic_train_r01_bag": "live_runs/openplane_race_domain_dynamic_steering_train_r01_20261002/run/run_0.db3",
    "dynamic_train_r02_bag": "live_runs/openplane_race_domain_dynamic_steering_train_r02_20261002/run/run_0.db3",
    "dynamic_validation_r01_bag": "live_runs/openplane_race_domain_dynamic_steering_validation_r01_20261002/run/run_0.db3",
    "dynamic_train_dataset": "live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/dynamic_steering_r02_train_20261002/openplane_dynamics.npz",
    "dynamic_validation_dataset": "live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/dynamic_steering_validation_r01_20261002/openplane_dynamics.npz",
    "dynamic_rssm_candidate": "live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/dynamic_steering_r02_rssm_h2_f2_latent32_20261002/best_rssm_teacher.pt",
    "dynamic_pose_candidate": "live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/dynamic_steering_r02_poseaware_rssm_h2_f2_latent32_20261002/best_rssm_teacher.pt",
    "hybrid_candidate": "live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/hybrid_race_teacher_nominal_residual_20261002/best_hybrid_race_teacher.pt",
    "hybrid_3d_candidate": "live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/hybrid_teacher_3d_h2_f5_latent32_balanced_20261002/best_teacher.pt",
    "greybox_candidate": "live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/greybox_highspeed_steering_h0p5_20261002/best_four_wheel_greybox.pt",
    "mpc_model_source": "f1tenth_mpc/src/vehicle_model.c",
    "mpc_controller_source": "f1tenth_mpc/src/mpc_controller_node.cpp",
    "mpc_competition_config": "f1tenth_mpc/config/mpc_competition.yaml",
    "mpc_race_domain_overlay": "f1tenth_mpc/config/mpc_race_domain_12mps_overlay.yaml",
    "mpc_compiled_library": "live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/production_mpc_build_devcontainer/libproduction_mpc.so",
    "current_practice_map_yaml": "f1tenth_planning/maps/autodrive_practice_20260924_b.yaml",
    "current_practice_map_image": "f1tenth_planning/maps/autodrive_practice_20260924_b.pgm",
    "current_practice_trajectory": "f1tenth_planning/trajectories/autodrive_practice_20260924_b/autodrive_practice_20260924_b_mintime_raceline_speed_headroom_codex.csv",
    "exact_mintime_optimizer": "SDU_Apex_Autodrive_Exact_MinTime_Optimizer_v1_3/f1tenth_planning/autodrive_mintime",
    "tum_track_optimizer": "f1tenth_planning/global_racetrajectory_optimization",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True).strip()


def _entries() -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for role, value in FILES.items():
        path = ROOT / value
        if not path.exists():
            raise FileNotFoundError(f"{role}: {path}")
        members = sorted(path.rglob("*") if path.is_dir() else (path,))
        files = [member for member in members if member.is_file()]
        if not files:
            raise ValueError(f"{role}: no files under {path}")
        for member in files:
            rel = member.relative_to(ROOT).as_posix()
            entry = rows.setdefault(rel, {
                "path": rel, "bytes": member.stat().st_size,
                "sha256": sha256(member), "roles": [],
            })
            entry["roles"].append(role)
    for entry in rows.values():
        entry["roles"].sort()
    return [rows[key] for key in sorted(rows)]


def freeze() -> dict[str, Any]:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite frozen registry: {OUTPUT}")
    if not HANDOFF.is_file():
        raise FileNotFoundError(HANDOFF)
    legacy = json.loads(LEGACY_REGISTRY.read_text(encoding="utf-8"))
    status = _git("status", "--porcelain")
    result = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "WP0 frozen baseline for the replacement offline simulator and raceline optimizer handoff",
        "handoff": {
            "path": str(HANDOFF), "sha256": sha256(HANDOFF),
        },
        "checkout": {
            "branch": _git("branch", "--show-current"),
            "head": _git("rev-parse", "HEAD"),
            "worktree_clean_at_freeze": not bool(status),
            "status_porcelain": status.splitlines(),
        },
        "historical_baseline": {
            "path": LEGACY_REGISTRY.relative_to(ROOT).as_posix(),
            "sha256": sha256(LEGACY_REGISTRY),
            "split_policy_and_pre-handoff_artifacts": legacy,
        },
        "production_pipeline": {
            "vehicle_model_c": legacy["production_pipeline"]["vehicle_model_c"],
            "mpc_config": legacy["production_pipeline"]["mpc_config"],
        },
        "test_split_status": {
            "test_and_final_test_scored_for_this_handoff": False,
            "practice_validation_runs": [
                "practice_model_validation_r02", "practice_model_validation_r03"],
            "frontier_validation_run": "openplane_steering_frontier_validation_r03",
            "highsteer_validation_runs": [
                "openplane_highsteer_75_validation_r01",
                "openplane_highsteer_75_validation_r02"],
        },
        "artifacts": _entries(),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return result


if __name__ == "__main__":
    report = freeze()
    print(json.dumps({
        "registry": OUTPUT.relative_to(ROOT).as_posix(),
        "head": report["checkout"]["head"],
        "clean": report["checkout"]["worktree_clean_at_freeze"],
        "artifact_file_count": len(report["artifacts"]),
    }, indent=2))
