#!/usr/bin/env python3
"""Bounded, deterministic weight sweep for the native AutoDRIVE MPC core.

The executable is deliberately run in separate short-lived processes because
the MPC configuration is initialized from the environment. The benchmark is
offline and uses no simulator or ground-truth topics. Keep the default worker
count low: this is a tuning aid, not a stress test for the development PC.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import subprocess


WEIGHT_ENV_NAMES = (
    "MPC_W_LAT_ERROR",
    "MPC_W_HEADING",
    "MPC_W_VELOCITY",
    "MPC_W_LAT_VEL",
    "MPC_W_YAW_RATE",
    "MPC_W_STEER_EFFORT",
    "MPC_W_ACCEL_EFFORT",
    "MPC_W_STEER_RATE",
    "MPC_W_ACCEL_RATE",
    "MPC_W_EFFECTIVE_STEERING",
)

BASE_WEIGHTS = {
    "MPC_W_LAT_ERROR": 2250.0,
    "MPC_W_HEADING": 75.0,
    "MPC_W_VELOCITY": 170.0,
    "MPC_W_LAT_VEL": 5.0,
    "MPC_W_YAW_RATE": 1.5,
    "MPC_W_STEER_EFFORT": 2.0,
    "MPC_W_ACCEL_EFFORT": 0.5,
    "MPC_W_STEER_RATE": 4.0,
    "MPC_W_ACCEL_RATE": 5.0,
    "MPC_W_EFFECTIVE_STEERING": 1.0,
}


@dataclass(frozen=True)
class Candidate:
    index: int
    weights: dict[str, float]


@dataclass
class Result:
    candidate: Candidate
    score: float
    mean_abs_ey_m: float
    max_abs_ey_m: float
    mean_abs_speed_error_mps: float
    max_abs_speed_error_mps: float
    max_solve_us: float
    solver_failures: int
    max_iteration_calls: int
    wall_violations: int
    return_code: int
    output: str


def make_candidates(count: int, seed: int) -> list[Candidate]:
    """Create reproducible log-uniform perturbations around the current profile."""
    rng = random.Random(seed)
    candidates: list[Candidate] = [Candidate(0, dict(BASE_WEIGHTS))]
    for index in range(1, count):
        weights = dict(BASE_WEIGHTS)
        for name in WEIGHT_ENV_NAMES:
            if name in {"MPC_W_LAT_ERROR", "MPC_W_HEADING", "MPC_W_VELOCITY"}:
                scale = 10 ** rng.uniform(-0.25, 0.25)
            elif name in {"MPC_W_LAT_VEL", "MPC_W_YAW_RATE"}:
                scale = 10 ** rng.uniform(-0.35, 0.35)
            else:
                scale = 10 ** rng.uniform(-0.45, 0.45)
            weights[name] *= scale
        candidates.append(Candidate(index, weights))
    return candidates


def parse_result(candidate: Candidate, completed: subprocess.CompletedProcess[str]) -> Result:
    row: dict[str, str] | None = None
    header = ("profile,score,mean_abs_ey_m,max_abs_ey_m,"
              "mean_abs_speed_error_mps,max_abs_speed_error_mps,max_solve_us,"
              "solver_failures,max_iteration_calls,wall_violations")
    for line in completed.stdout.splitlines():
        if line.startswith("candidate,"):
            row = next(csv.DictReader([header, line]))
            break
    if row is None:
        return Result(candidate, float("inf"), float("inf"), float("inf"),
                      float("inf"), float("inf"), float("inf"), 1, 0, 1,
                      completed.returncode, completed.stdout + completed.stderr)
    return Result(
        candidate=candidate,
        score=float(row["score"]),
        mean_abs_ey_m=float(row["mean_abs_ey_m"]),
        max_abs_ey_m=float(row["max_abs_ey_m"]),
        mean_abs_speed_error_mps=float(row["mean_abs_speed_error_mps"]),
        max_abs_speed_error_mps=float(row["max_abs_speed_error_mps"]),
        max_solve_us=float(row["max_solve_us"]),
        solver_failures=int(row["solver_failures"]),
        max_iteration_calls=int(row["max_iteration_calls"]),
        wall_violations=int(row["wall_violations"]),
        return_code=completed.returncode,
        output=completed.stdout + completed.stderr,
    )


def run_candidate(binary: Path, candidate: Candidate) -> Result:
    env = os.environ.copy()
    env.update({name: f"{candidate.weights[name]:.9g}" for name in WEIGHT_ENV_NAMES})
    env.update({
        "MPC_BENCHMARK_SINGLE": "1",
        "MPC_VERBOSE": "0",
        "MPC_ADAPTIVE_RHO": "1",
    })
    completed = subprocess.run(
        [str(binary)], env=env, capture_output=True, text=True, check=False
    )
    return parse_result(candidate, completed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path("build/test_closed_loop_benchmark"))
    parser.add_argument("--candidates", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()

    if args.candidates < 1 or args.candidates > 128:
        parser.error("--candidates must be between 1 and 128")
    if args.workers < 1 or args.workers > 4:
        parser.error("--workers must be between 1 and 4")
    binary = args.binary.resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        parser.error(f"benchmark binary is not executable: {binary}")

    candidates = make_candidates(args.candidates, args.seed)
    results: list[Result] = []
    print(f"Evaluating {len(candidates)} candidates with {args.workers} worker(s)")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_candidate, binary, candidate) for candidate in candidates]
        for completed_count, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            print(
                f"[{completed_count:02d}/{len(candidates)}] candidate={result.candidate.index:02d} "
                f"score={result.score:.5f} ey={result.mean_abs_ey_m:.4f}m "
                f"max_ey={result.max_abs_ey_m:.4f}m solve_max={result.max_solve_us:.1f}us "
                f"fail={result.solver_failures} walls={result.wall_violations}",
                flush=True,
            )

    passing = [
        result for result in results
        if result.return_code == 0 and result.solver_failures == 0 and result.wall_violations == 0
    ]
    if not passing:
        print("No collision-free, solver-clean candidate was found.", file=sys.stderr)
        return 1

    best = min(passing, key=lambda item: item.score)
    print("\nBest collision-free candidate:")
    print(f"candidate={best.candidate.index} score={best.score:.6f} "
          f"mean_abs_ey={best.mean_abs_ey_m:.6f} max_abs_ey={best.max_abs_ey_m:.6f} "
          f"max_solve_us={best.max_solve_us:.2f}")
    for name in WEIGHT_ENV_NAMES:
        print(f"export {name}={best.candidate.weights[name]:.9g}")

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as handle:
            fields = ["candidate", "score", "mean_abs_ey_m", "max_abs_ey_m",
                      "mean_abs_speed_error_mps", "max_abs_speed_error_mps",
                      "max_solve_us", "solver_failures", "max_iteration_calls",
                      "wall_violations", *WEIGHT_ENV_NAMES]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for result in results:
                writer.writerow({
                    "candidate": result.candidate.index,
                    "score": result.score,
                    "mean_abs_ey_m": result.mean_abs_ey_m,
                    "max_abs_ey_m": result.max_abs_ey_m,
                    "mean_abs_speed_error_mps": result.mean_abs_speed_error_mps,
                    "max_abs_speed_error_mps": result.max_abs_speed_error_mps,
                    "max_solve_us": result.max_solve_us,
                    "solver_failures": result.solver_failures,
                    "max_iteration_calls": result.max_iteration_calls,
                    "wall_violations": result.wall_violations,
                    **result.candidate.weights,
                })
        print(f"Wrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
