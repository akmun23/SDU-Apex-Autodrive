#!/usr/bin/env python3
"""Search lateral-envelope scale for the *current runtime pairing*.

This legacy outer search uses the current MPC's practice-run lap time and
trackability as its objective/gate. A collision or incomplete screen means
only that this candidate was not trackable with that runtime stack. It is not
evidence that the line is physically infeasible, and these observations must
not be used as the theoretical plant's feasibility labels or as the Stage-2
offline-plant objective. The theoretical search requires a separately
validated command-only plant evaluator.
"""

from __future__ import annotations

import argparse
import csv
import json
import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml


FIELDS = (
    "run_id", "scale", "robustness_weight", "mean_lap_s", "lap_count",
    "collision_delta", "cte_p95_m", "wall_clearance_min_m", "feasible",
    "objective_s", "controller_image_id", "simulator_image_digest",
    "mpc_config_sha256", "mpc_overlay_sha256", "optimizer_config_sha256",
    "centerline_sha256", "model_signature_sha256", "source_run_dir",
)
BOUNDS = (0.98, 1.15)
SCREEN_LAPS = 2
BASELINE_ROBUSTNESS_WEIGHT = 0.05
MAX_CTE_P95_M = 0.20
MIN_WALL_CLEARANCE_M = 0.15
DOE_SCALES = (1.075, 1.15, 1.035)


def _read_observations(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError(f"{path}: unexpected observation schema")
        return list(reader)


def _signature(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row[key] or "" for key in (
        "controller_image_id", "simulator_image_digest", "mpc_config_sha256",
        "mpc_overlay_sha256", "optimizer_config_sha256", "centerline_sha256",
        "model_signature_sha256"))


def _active_model_signature(run_dir: Path, manifest: dict[str, Any]) -> str:
    """Hash the effective optimizer model, excluding only the BO variable.

    ``resolved_config.yaml`` is written after CLI overlays and overrides. The
    source YAML alone is not a reliable record of what the optimizer solved.
    Normalize the envelope scale in both its model and active-settings copies;
    every other resolved parameter remains part of the campaign identity.
    """
    contract_path = run_dir / "vehicle_contract.json"
    if not contract_path.is_file():
        raise FileNotFoundError(f"required effective vehicle contract is missing: {contract_path}")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    resolved = contract.get("resolved_optimizer_config") or {}
    resolved_path_text = resolved.get("path")
    expected_hash = resolved.get("sha256")
    if not resolved_path_text or not expected_hash:
        raise ValueError(f"{contract_path}: no resolved optimizer config fingerprint")
    repo_root = Path(__file__).resolve().parents[2]
    configured_path = Path(resolved_path_text)
    if configured_path.is_absolute():
        resolved_path = configured_path.resolve()
    else:
        repo_candidate = (repo_root / configured_path).resolve()
        run_candidate = (run_dir / configured_path).resolve()
        resolved_path = repo_candidate if repo_candidate.is_file() else run_candidate
    try:
        resolved_path.relative_to(repo_root)
    except ValueError as exc:
        try:
            resolved_path.relative_to(run_dir.resolve())
        except ValueError:
            raise ValueError("resolved optimizer config is outside the run/repository") from exc
    if not resolved_path.is_file():
        raise FileNotFoundError(resolved_path)
    digest = hashlib.sha256(resolved_path.read_bytes()).hexdigest()
    if digest != expected_hash or digest != manifest.get("resolved_optimizer_config_sha256"):
        raise ValueError(f"resolved optimizer config hash mismatch: {resolved_path}")

    active = yaml.safe_load(resolved_path.read_text(encoding="utf-8")) or {}
    settings = active.get("config")
    model = active.get("vehicle_model")
    envelope = active.get("lateral_envelope")
    if not isinstance(settings, dict) or not isinstance(model, dict) or not isinstance(envelope, dict):
        raise ValueError(f"{resolved_path}: missing effective model/settings")
    settings = json.loads(json.dumps(settings))
    envelope = json.loads(json.dumps(envelope))
    for section in (settings.get("lateral_envelope"), envelope):
        if not isinstance(section, dict) or "scale" not in section:
            raise ValueError(f"{resolved_path}: BO scale is not recorded in both model copies")
        section["scale"] = "<bayesian-search-scale>"
    normalized = {
        "settings": settings,
        "vehicle_model": model,
        "lateral_envelope": envelope,
    }
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def record_observation(history: Path, run_dir: Path, scale: float,
                       robustness_weight: float) -> dict[str, Any]:
    """Record a runtime screen; ``feasible`` means runtime-screen pass only."""
    run_dir = run_dir.resolve()
    manifest_path = run_dir / "run_manifest.json"
    summary_path = run_dir / "analysis/race_run/summary.json"
    lap_times_path = run_dir / "analysis/race_run/lap_times.csv"
    for path in (manifest_path, summary_path, lap_times_path):
        if not path.is_file():
            raise FileNotFoundError(f"required completed-run artifact is missing: {path}")
    if not BOUNDS[0] <= scale <= BOUNDS[1]:
        raise ValueError(f"scale {scale:g} outside BO bounds {BOUNDS}")
    if not math.isclose(robustness_weight, BASELINE_ROBUSTNESS_WEIGHT,
                        rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError(
            "this search isolates envelope scale; steering robustness weight "
            f"must remain {BASELINE_ROBUSTNESS_WEIGHT:g}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    model_signature = _active_model_signature(run_dir, manifest)
    lap_rows: list[dict[str, str]]
    with lap_times_path.open("r", encoding="utf-8", newline="") as stream:
        lap_rows = list(csv.DictReader(stream))
    scored = [float(row["lap_time_s"]) for row in lap_rows
              if row.get("lap_role") == "scored"
              and row.get("complete", "").lower() == "true"
              and row.get("lap_time_s")]
    collision = int(summary.get("collision_delta", 0))
    tracking = summary.get("scored_lap_metrics", {}).get(
        "absolute_tracking_error_m", {})
    cte_p95 = tracking.get("p95")
    wall_min = summary.get("scored_lap_metrics", {}).get(
        "minimum_wall_clearance_m")
    enough_laps = len(scored) >= SCREEN_LAPS
    finite_quality = (cte_p95 is not None and math.isfinite(float(cte_p95))
                      and wall_min is not None and math.isfinite(float(wall_min)))
    # This flag is deliberately scoped to the unchanged runtime controller.
    # It must never be interpreted as a physical raceline-feasibility label.
    feasible = (collision == 0 and enough_laps and finite_quality
                and float(cte_p95) <= MAX_CTE_P95_M
                and float(wall_min) >= MIN_WALL_CLEARANCE_M)
    mean_lap = float(np.mean(scored)) if scored else None
    objective = mean_lap if feasible else 8.0 + max(0, SCREEN_LAPS - len(scored))
    row: dict[str, Any] = {
        "run_id": str(manifest.get("run_id", run_dir.name)),
        "scale": f"{scale:.8f}",
        "robustness_weight": f"{robustness_weight:.8f}",
        "mean_lap_s": "" if mean_lap is None else f"{mean_lap:.8f}",
        "lap_count": len(scored),
        "collision_delta": collision,
        "cte_p95_m": "" if cte_p95 is None else f"{float(cte_p95):.8f}",
        "wall_clearance_min_m": "" if wall_min is None else f"{float(wall_min):.8f}",
        "feasible": int(feasible),
        "objective_s": f"{objective:.8f}",
        "controller_image_id": str(manifest.get("controller_image_id") or ""),
        "simulator_image_digest": str(manifest.get("simulator_image_digest") or ""),
        "mpc_config_sha256": str(manifest.get("mpc_config_sha256") or ""),
        "mpc_overlay_sha256": str(manifest.get("mpc_overlay_sha256") or ""),
        "optimizer_config_sha256": str(manifest.get("optimizer_config_sha256") or ""),
        "centerline_sha256": str(manifest.get("centerline_sha256") or ""),
        "model_signature_sha256": model_signature,
        "source_run_dir": run_dir.as_posix(),
    }
    existing = _read_observations(history)
    if any(old["run_id"] == row["run_id"] for old in existing):
        raise ValueError(f"run {row['run_id']} is already registered")
    if existing and _signature(existing[0]) != _signature(row):
        raise ValueError("run image/model/map lineage differs from BO history")
    history.parent.mkdir(parents=True, exist_ok=True)
    write_header = not history.exists()
    with history.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return row


def _matern52(x: np.ndarray, y: np.ndarray, length_scale: float = 0.32
              ) -> np.ndarray:
    distance = np.abs(x[:, None] - y[None, :]) / length_scale
    scaled = math.sqrt(5.0) * distance
    return (1.0 + scaled + (5.0 / 3.0) * distance * distance) * np.exp(-scaled)


def propose_scale(rows: list[dict[str, str]], seed: int = 20261014) -> dict[str, Any]:
    unique = sorted({round(float(row["scale"]), 6) for row in rows})
    feasible_scales = sorted({round(float(row["scale"]), 6) for row in rows
                              if int(row["feasible"]) == 1})
    failed_scales = sorted({round(float(row["scale"]), 6) for row in rows
                            if int(row["feasible"]) == 0})
    brackets = [(safe, failed) for safe in feasible_scales
                for failed in failed_scales if safe < failed
                and failed - safe > 0.01]
    if brackets:
        safe, failed = min(brackets, key=lambda pair: pair[1] - pair[0])
        midpoint = round(0.5 * (safe + failed), 6)
        if all(abs(midpoint - observed) > 0.004 for observed in unique):
            return {"scale": midpoint, "method": "feasibility_boundary_refinement",
                    "safe_scale": safe, "failed_scale": failed,
                    "unique_trials": len(unique)}
    for scale in DOE_SCALES:
        if all(abs(scale - observed) > 0.005 for observed in unique):
            return {"scale": scale, "method": "initial_space_filling",
                    "unique_trials": len(unique)}

    if len(unique) < 3:
        remaining = [scale for scale in DOE_SCALES
                     if all(abs(scale - observed) > 0.005 for observed in unique)]
        if remaining:
            return {"scale": remaining[0], "method": "initial_space_filling",
                    "unique_trials": len(unique)}

    x = np.asarray([(float(row["scale"]) - BOUNDS[0]) /
                    (BOUNDS[1] - BOUNDS[0]) for row in rows])
    y = np.asarray([float(row["objective_s"]) for row in rows])
    kernel = _matern52(x, x)
    kernel.flat[::len(kernel) + 1] += 0.04 ** 2
    try:
        factor = np.linalg.cholesky(kernel)
    except np.linalg.LinAlgError:
        kernel.flat[::len(kernel) + 1] += 1.0e-8
        factor = np.linalg.cholesky(kernel)
    centered = y - float(np.mean(y))
    alpha = np.linalg.solve(factor.T, np.linalg.solve(factor, centered))
    candidates = np.linspace(BOUNDS[0], BOUNDS[1], 2001)
    normalized = (candidates - BOUNDS[0]) / (BOUNDS[1] - BOUNDS[0])
    cross = _matern52(normalized, x)
    mean = float(np.mean(y)) + cross @ alpha
    projection = np.linalg.solve(factor, cross.T)
    variance = np.maximum(1.0 - np.sum(projection * projection, axis=0), 1.0e-12)
    std = np.sqrt(variance)
    feasible_y = [float(row["objective_s"]) for row in rows
                  if int(row["feasible"]) == 1]
    best = min(feasible_y) if feasible_y else float(np.min(y))
    improvement = best - mean - 0.01
    z = improvement / std
    cdf = np.asarray([0.5 * (1.0 + math.erf(float(value) / math.sqrt(2.0)))
                      for value in z])
    pdf = np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    acquisition = improvement * cdf + std * pdf
    for scale in unique:
        acquisition[np.abs(candidates - scale) < 0.004] = -np.inf
    index = int(np.argmax(acquisition))
    return {"scale": float(candidates[index]), "method": "expected_improvement",
            "expected_improvement_s": float(acquisition[index]),
            "best_feasible_mean_lap_s": None if not feasible_y else best,
            "unique_trials": len(unique)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    propose_parser = subparsers.add_parser("propose")
    propose_parser.add_argument("--history", required=True, type=Path)
    propose_parser.add_argument("--seed", type=int, default=20261014)
    record_parser = subparsers.add_parser("record")
    record_parser.add_argument("--history", required=True, type=Path)
    record_parser.add_argument("--run-dir", required=True, type=Path)
    record_parser.add_argument("--scale", required=True, type=float)
    record_parser.add_argument("--robustness-weight", required=True, type=float)
    args = parser.parse_args()
    if args.command == "record":
        result = record_observation(args.history, args.run_dir, args.scale,
                                    args.robustness_weight)
    else:
        result = propose_scale(_read_observations(args.history), args.seed)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
