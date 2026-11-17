#!/usr/bin/env python3
"""Time frozen SQP cases and compare their numerical results with a saved run.

Usage:
    bench_cases.py BENCH CASE_DIR [--iterations 4] [--repeats 5] [--save OUT.json]
                   [--compare GOLDEN.json] [--quality GOLDEN.json]
"""
# Six workers use CPUs 0,2,4,6,8,10 unless --cpus overrides the layout.
# Report median per-case times over repeats, separating first and later iterations.
# --compare reports the maximum absolute fingerprint difference (zero for exact parity).
# --quality reports mean relative final-cost change (positive is worse), maximum
# control difference and mean accepted-step model error |1 - actual/predicted|.
# These measurements cover solver cost and equivalence, not closed-loop behavior.
import argparse
import glob
import json
import os
import statistics
import subprocess
import sys

PHASES = ("lin_ms", "model_ms", "qp_ms", "roll_ms", "ms")
WAIT = {"OMP_PROC_BIND": "spread", "OMP_WAIT_POLICY": "active", "GOMP_SPINCOUNT": "1000000"}


def run_case(bench, case, iterations, cpus):
    environment = dict(os.environ, **WAIT, OMP_NUM_THREADS=str(len(cpus)), MPC_THREADS=str(len(cpus)),
                       OMP_PLACES=",".join("{%d}" % c for c in cpus))
    out = subprocess.run(["taskset", "-c", ",".join(map(str, cpus)), bench, "solve", case, str(iterations)],
                         capture_output=True, text=True, env=environment, check=True).stdout
    return json.loads(out.splitlines()[0])


def fingerprint(result):
    history = result.get("history", [])
    return {
        "final_cost": result["final_cost"],
        "controls": [value for pair in result["controls"] for value in pair],
        "iterations": [[h["cost_before"], h["cost_trial"], h["radius"], h["predicted"], h["actual"]]
                       for h in history],
        "model_error": [abs(1. - h["actual"] / h["predicted"]) for h in history
                        if h["accepted"] and h["predicted"] > 0.],
    }


def max_difference(a, b):
    worst = abs(a["final_cost"] - b["final_cost"])
    worst = max([worst] + [abs(x - y) for x, y in zip(a["controls"], b["controls"])])
    if len(a["iterations"]) != len(b["iterations"]):
        return float("inf")
    for row_a, row_b in zip(a["iterations"], b["iterations"]):
        worst = max([worst] + [abs(x - y) for x, y in zip(row_a, row_b)])
    return worst


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bench")
    parser.add_argument("cases")
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--cpus", default="0,2,4,6,8,10", help="CPUs of the OpenMP workers (production: 0,2,4,6,8,10)")
    parser.add_argument("--save")
    parser.add_argument("--compare")
    parser.add_argument("--quality")
    args = parser.parse_args()
    cpus = [int(c) for c in args.cpus.split(",")]
    cases = sorted(glob.glob(os.path.join(args.cases, "*.case")))
    first = {p: [] for p in PHASES}
    later = {p: [] for p in PHASES}
    prints = {}
    for case in cases:
        per_first = {p: [] for p in PHASES}
        per_later = {p: [] for p in PHASES}
        for repeat in range(args.repeats):
            result = run_case(args.bench, case, args.iterations, cpus)
            if repeat == 0:
                prints[os.path.basename(case)] = fingerprint(result)
            history = result["history"]
            for phase in PHASES:
                per_first[phase].append(history[0][phase])
                if len(history) > 1:
                    per_later[phase].append(statistics.mean(h[phase] for h in history[1:]))
        for phase in PHASES:
            first[phase].append(statistics.median(per_first[phase]))
            if per_later[phase]:
                later[phase].append(statistics.median(per_later[phase]))
    print(f"{len(cases)} cases, {args.repeats} repeats, {args.iterations} iterations")
    print(f"{'phase (mean of case medians, ms)':36s} first iteration   later iterations")
    for phase in PHASES:
        print(f"{phase:36s} {statistics.mean(first[phase]):14.3f}   {statistics.mean(later[phase]):14.3f}")
    if args.save:
        json.dump(prints, open(args.save, "w"))
    if args.quality:
        golden = json.load(open(args.quality))
        names = [n for n in prints if n in golden]
        relative = [(prints[n]["final_cost"] - golden[n]["final_cost"]) / abs(golden[n]["final_cost"])
                    for n in names]
        control = max(max(abs(x - y) for x, y in zip(prints[n]["controls"], golden[n]["controls"]))
                      for n in names)
        error = statistics.mean(e for n in names for e in prints[n]["model_error"])
        base = statistics.mean(e for n in names for e in golden[n]["model_error"])
        print(f"final cost vs {args.quality}: mean {statistics.mean(relative):+.3e}, "
              f"worst {max(relative):+.3e}, best {min(relative):+.3e}; largest control difference "
              f"{control:.3e}; mean model error {error:.4f} (saved {base:.4f})")
    if args.compare:
        golden = json.load(open(args.compare))
        worst = max(max_difference(prints[name], golden[name]) for name in prints if name in golden)
        print(f"largest absolute difference to {args.compare}: {worst:.3e}")
        if worst != 0:
            sys.exit(1)


if __name__ == "__main__":
    main()
