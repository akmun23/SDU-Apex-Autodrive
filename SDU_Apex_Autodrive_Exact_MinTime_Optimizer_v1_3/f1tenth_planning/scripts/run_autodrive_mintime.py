#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml


def find_repo_root(start: Path) -> Path:
    p = start.resolve()
    for _ in range(12):
        if (p / "f1tenth_planning").is_dir() and (p / "f1tenth_mpc").is_dir():
            return p
        p = p.parent
    raise RuntimeError("Could not locate repository root containing f1tenth_planning and f1tenth_mpc")


def main() -> int:
    parser = argparse.ArgumentParser(description="AutoDRIVE-specific minimum-lap-time optimizer")
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--centerline", type=Path, default=None,
                        help="Prepared centerline CSV. If absent, map->centerline is run using existing repo code.")
    parser.add_argument("--map", type=Path, default=None)
    parser.add_argument("--warm-raceline", type=Path, default=None,
                        help="Existing raceline CSV used only as an IPOPT initial guess.")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    # The optimizer may be either merged into the repository or extracted as
    # a standalone bundle inside it.  Keep these roots separate:
    #   tool_root -> contains this optimizer package/config
    #   repo_root -> contains the real f1tenth_mpc, maps and project data
    script_path = Path(__file__).resolve()
    tool_root = script_path.parents[2]
    root = (args.repo_root.resolve() if args.repo_root else find_repo_root(script_path))

    # Import the optimizer from the bundle that owns this runner.  This is
    # deliberately independent of whether the bundle was overlaid into the
    # repository or extracted into a nested directory.
    if str(tool_root) not in sys.path:
        sys.path.insert(0, str(tool_root))
    if str(root) not in sys.path:
        sys.path.insert(1, str(root))

    from f1tenth_planning.autodrive_mintime.model import (
        VehicleModel, LateralEnvelope, load_yaml, numerical_model_summary,
    )
    from f1tenth_planning.autodrive_mintime.track import (
        ensure_centerline_from_repo, make_circle_track, export_track_csv,
    )
    from f1tenth_planning.autodrive_mintime.optimizer import (
        build_guess, solve_track, run_continuation, export_solution,
    )

    if args.config is not None:
        config_path = args.config.resolve()
    else:
        # Prefer the config shipped with the optimizer bundle.  When installed
        # directly into the repository, tool_root == repo_root and this is the
        # normal project path.
        config_path = tool_root / "f1tenth_planning/config/autodrive_mintime_exact.yaml"
        if not config_path.exists():
            fallback = root / "f1tenth_planning/config/autodrive_mintime_exact.yaml"
            if fallback.exists():
                config_path = fallback
            else:
                raise FileNotFoundError(
                    "Could not find autodrive_mintime_exact.yaml in either the "
                    f"optimizer bundle ({tool_root}) or repository ({root})"
                )
    cfg = load_yaml(config_path)
    model = VehicleModel.from_repo(root, cfg)
    envelope = LateralEnvelope.from_repo(root, model, cfg)

    print("=== AutoDRIVE minimum-time optimizer ===")
    print(f"tool:   {tool_root}")
    print(f"repo:   {root}")
    print(f"config: {config_path}")
    print(json.dumps(numerical_model_summary(model), indent=2))
    print("lateral envelope:")
    print(json.dumps(envelope.to_dict(), indent=2))

    if args.self_test:
        test_cfg = dict(cfg)
        test_cfg.setdefault("solver", {})
        test_cfg["solver"] = dict(test_cfg["solver"])
        test_cfg["solver"]["print_level"] = 0
        test_track = make_circle_track(radius_m=5.0, half_width_m=1.0, spacing_m=0.6)
        guess = build_guess(test_track, model, envelope, test_cfg, "center")
        sol = solve_track(test_track, model, envelope, test_cfg, guess)
        print(f"SELF-TEST SOLVED: lap={sol.lap_time_s:.6f}s status={sol.solver_stats.get('return_status')}")
        return 0

    default_map = root / "f1tenth_planning/maps/autodrive_track_ftg_commit_20260909_025m.yaml"
    map_path = (args.map or default_map).resolve()
    output = (args.output or (root / "f1tenth_planning/trajectories/autodrive_mintime_exact")).resolve()
    output.mkdir(parents=True, exist_ok=True)

    centerline = args.centerline
    if centerline is None:
        centerline = ensure_centerline_from_repo(
            root, map_path, output / "track_prep",
            env_overrides={
                "MINTIME_OPTIMIZER_SMOOTHING_S": str(cfg.get("track", {}).get("prep_smoothing_s", 6.0)),
            },
        )
    centerline = centerline.resolve()

    warm = args.warm_raceline
    if warm is None:
        candidate = root / "f1tenth_planning/trajectories/autodrive_track_ftg_commit_20260909_025m_mintime_raceline.csv"
        if candidate.exists():
            warm = candidate

    print(f"centerline: {centerline}")
    print(f"warm line:  {warm}")
    print(f"output:     {output}")

    # Save provenance before the first nonlinear solve.  Even a failed coarse
    # optimization must leave enough information to reproduce the run.
    resolved = {
        "repo_root": str(root),
        "tool_root": str(tool_root),
        "config_path": str(config_path),
        "centerline": str(centerline),
        "map": str(map_path),
        "warm_raceline": str(warm) if warm else None,
        "output": str(output),
        "config": cfg,
        "vehicle_model": model.to_dict(),
        "lateral_envelope": envelope.to_dict(),
    }
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")

    solution, continuation = run_continuation(
        centerline, root, model, envelope, cfg, warm,
        checkpoint_dir=output / "checkpoints",
    )
    export_spacing = cfg.get("track", {}).get("export_spacing_m")
    report = export_solution(
        solution, model, envelope, cfg, output,
        output_spacing_m=(float(export_spacing)
                          if export_spacing is not None else None))

    # Replace approximate wall-distance columns with the same map ray-cast
    # used by the repository's existing trajectory pipeline, then enforce the
    # exact optimize_trajectory.py safety semantics as a hard postcondition.
    trajectory_path = Path(report["files"]["trajectory"])
    wallchecked_path = output / "autodrive_mintime_raceline_wallchecked.csv"
    wall_script = root / "f1tenth_planning/scripts/compute_wall_distances.py"
    extra_clearance = float(cfg.get("track", {}).get("extra_wall_clearance_m", 0.0))
    required_clearance = model.required_wall_clearance_m + extra_clearance
    cmd = [
        sys.executable, str(wall_script),
        "--map", str(map_path),
        "--trajectory", str(trajectory_path),
        "--output", str(wallchecked_path),
        "--car-width", str(model.planning_footprint_width_m),
        "--wall-clearance", str(required_clearance),
    ]
    print("\n=== Exact map wall-distance verification ===")
    subprocess.run(cmd, cwd=str(root), check=True)

    rows = []
    for raw in wallchecked_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        vals = [float(x) for x in line.split(",")]
        if len(vals) >= 9:
            rows.append(vals)
    if not rows:
        raise RuntimeError("Wall checker produced no trajectory rows")
    min_left = min(row[7] for row in rows)
    min_right = min(row[8] for row in rows)
    required_center_to_wall = 0.5 * model.planning_footprint_width_m + required_clearance
    if min_left + 1.0e-6 < required_center_to_wall or min_right + 1.0e-6 < required_center_to_wall:
        raise RuntimeError(
            "FINAL TRAJECTORY REJECTED BY MAP WALL CHECK: "
            f"required center-to-wall >= {required_center_to_wall:.3f} m, "
            f"observed left={min_left:.3f} m right={min_right:.3f} m. "
            "No unsafe raceline will be promoted."
        )
    wallchecked_path.replace(trajectory_path)
    report["map_wall_validation"] = {
        "status": "passed",
        "source": "f1tenth_planning/scripts/compute_wall_distances.py",
        "planning_footprint_width_m": model.planning_footprint_width_m,
        "required_side_of_car_clearance_m": required_clearance,
        "required_center_to_wall_m": required_center_to_wall,
        "minimum_left_center_to_wall_m": min_left,
        "minimum_right_center_to_wall_m": min_right,
    }
    report["continuation"] = continuation
    if continuation.get("status") != "complete":
        report["status"] = "partial_solution_best_converged_mesh"
        report["warning"] = (
            "A finer continuation mesh failed. This trajectory is the best "
            "converged coarser mesh and is exported for analysis, not silently "
            "mislabelled as the requested final discretization."
        )
    (output / "report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")

    print("\n=== RESULT ===")
    print(f"continuation:  {continuation.get('status')}")
    if continuation.get("failed_mesh_spacing_m") is not None:
        print(f"failed mesh:   {continuation['failed_mesh_spacing_m']:.3f} m")
    print(f"predicted lap: {report['predicted_lap_time_s']:.6f} s")
    print(f"export lap:    {report['export_recomputed_lap_time_s']:.6f} s")
    print(f"trajectory:    {report['files']['trajectory']}")
    print(f"report:        {output / 'report.json'}")
    print(f"checkpoints:   {output / 'checkpoints'}")
    if continuation.get("status") != "complete":
        print("WARNING: final fine mesh did not converge; exported the last converged mesh.")
    print("Send report.json, solution_nodes.csv, autodrive_mintime_raceline.csv, checkpoints/continuation_report.json and the full console log back for tuning.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
