#!/usr/bin/env python3
"""Screen MPC weights with Sobol samples and regress trial outcomes against the factors.

Usage:
    tools/analysis/weight_screening.py generate OUT.json [--runs 64] [--track competition]
    tools/analysis/sweep_mpc.py run OUT.json
    tools/analysis/weight_screening.py analyze OUT.json
"""
import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
from scipy.stats import qmc

ROOT = Path(__file__).resolve().parents[2]
LAP_LENGTH_M = 42.843
PASS_WINDOWS = {"hairpin": (13.5, 19.0), "chicane": (33.0, 39.0)}

# name: (low, high, kind); kind "log" draws low..high log-uniformly, "lin"
# uniformly. Bounds are absolute values.
FACTORS = {
    "weight_e_y": (1000.0, 16000.0, "log"),
    "weight_e_psi": (375.0, 6000.0, "log"),
    "weight_u": (8.75, 140.0, "log"),
    "weight_u_overspeed": (10.0, 160.0, "log"),
    "weight_v": (1.25, 20.0, "log"),
    "weight_r": (1.95, 31.25, "log"),
    "weight_steering_command": (0.75, 12.0, "log"),
    "weight_throttle_command": (0.025, 0.4, "log"),
    "weight_steering_slew": (500.0, 8000.0, "log"),
    "weight_throttle_slew": (12.5, 200.0, "log"),
    "weight_grip_loss": (50.0, 800.0, "log"),
    "grip_loss_tolerance": (0.05, 0.6, "lin"),
    "overspeed_curvature_gain_m": (0.0, 2.0, "lin"),
    "underspeed_curvature_gain_m": (0.0, 1.0, "lin"),
    "lateral_curvature_gain_m": (0.0, 3.0, "lin"),
    "heading_curvature_gain_m": (0.0, 3.0, "lin"),
    "corridor_margin_m": (0.02, 0.09, "lin"),
    "terminal_multiplier": (1.0, 9.0, "log"),
}
# Parameters that always take the value of another factor.
TIED = {"first_prediction_corridor_margin_m": "corridor_margin_m"}


def coordinate(name, value, narrow=1.0):
    """Unit coordinate of a value in the (narrowed) range; inverse of value_at."""
    low, high, kind = FACTORS[name]
    narrow_by = 1.0
    if kind == "log":
        low, high, value = math.log(low), math.log(high), math.log(value)
        narrow_by = narrow
    centre, half = .5 * (low + high), .5 * (high - low) / narrow_by
    return (value - (centre - half)) / (2 * half)


def value_at(name, unit, narrow=1.0):
    """Value at the unit coordinate; `narrow` > 1 shrinks the log-scale ranges (the weights,
    centred on their configured values) around their centre; the linear ranges keep their bounds."""
    low, high, kind = FACTORS[name]
    if kind == "log":
        low, high = math.log(low), math.log(high)
        narrow_by = narrow
    else:
        narrow_by = 1.0
    centre, half = .5 * (low + high), .5 * (high - low) / narrow_by
    value = centre - half + unit * 2 * half
    return float(math.exp(value) if kind == "log" else value)


def generate(args):
    sequence = qmc.Sobol(d=len(FACTORS), scramble=True, seed=args.seed)
    units = sequence.random(args.runs)
    candidates = []
    for index, row in enumerate(units):
        params = {name: round(value_at(name, unit, args.narrow), 6)
                  for name, unit in zip(FACTORS, row)}
        for tied, source in TIED.items():
            params[tied] = params[source]
        candidates.append({"name": f"s{index:03d}", "params": params})
    spec = {"name": Path(args.out).stem, "track": args.track, "laps": 11, "narrow": args.narrow,
            "runs_per_candidate": 1, "interleave": False, "base": {},
            "candidates": candidates}
    Path(args.out).write_text(json.dumps(spec, indent=1) + "\n")
    print(f"{args.runs} runs, {len(FACTORS)} factors -> {args.out}")


def run_outcomes(run_id):
    run_dir = ROOT / "live_runs" / run_id
    watcher = (run_dir / "watcher.log").read_text()
    laps = [float(t) for t in re.findall(r"LAP_TIME lap=\d+ time_s=([\d.]+)", watcher)]
    cycles = list(csv.DictReader(open(run_dir / "analysis" / "cycles.csv")))
    first_rejection = next((i for i, c in enumerate(cycles)
                            if "reject" in (c["status"] or "")), None)
    failed = len(laps) < 10
    stalled = False
    if failed:
        end = first_rejection if first_rejection is not None else len(cycles) - 1
        stalled = any(c["status"] == "state_stale_wait"
                      for c in cycles[max(0, end - 20):end + 1])
    lowest, overspeed = [], []
    passes = {}
    for c in cycles:
        if c["status"] != "accepted_feasible" or not c["progress_m"] or not c["e_y_m"]:
            continue
        progress = float(c["progress_m"])
        lap, position = int(progress // LAP_LENGTH_M), progress % LAP_LENGTH_M
        for name, (low, high) in PASS_WINDOWS.items():
            if low <= position <= high:
                passes.setdefault((name, lap), []).append(
                    (float(c["e_y_m"]), float(c["u_mps"]) - float(c["target_u_mps"] or 0)))
    for samples in passes.values():
        if len(samples) >= 10:
            lowest.append(min(e for e, _ in samples))
            overspeed.append(max(o for _, o in samples))
    return {
        "laps_driven": min(len(laps), 11),
        "failure": float(failed and not stalled),
        "stall": float(failed and stalled),
        "lowest_ey_mean": float(np.mean(lowest)) if lowest else math.nan,
        "lowest_ey_p10": float(np.percentile(lowest, 10)) if lowest else math.nan,
        "overspeed_mean": float(np.mean(overspeed)) if overspeed else math.nan,
        "lap_time": float(np.mean(laps[1:])) if len(laps) > 1 else math.nan,
    }


def effects(design, outcome, rng, resamples=400):
    """Least-squares slope per factor on coordinates centred to [-1, 1]."""
    keep = ~np.isnan(outcome)
    x, y = design[keep], outcome[keep]
    if len(y) < design.shape[1] + 2:
        return None
    x1 = np.hstack([np.ones((len(y), 1)), x])
    ridge = 1e-3 * np.eye(x1.shape[1])
    ridge[0, 0] = 0
    solve = lambda a, b: np.linalg.solve(a.T @ a + ridge, a.T @ b)[1:]
    point = solve(x1, y)
    boot = np.empty((resamples, len(point)))
    for k in range(resamples):
        pick = rng.integers(0, len(y), len(y))
        boot[k] = solve(x1[pick], y[pick])
    return point, np.percentile(boot, 2.5, axis=0), np.percentile(boot, 97.5, axis=0), len(y)


def analyze(args):
    spec = json.loads(Path(args.spec).read_text())
    names = list(FACTORS)
    rows, designs = [], []
    results = (ROOT / "live_runs" / "sweeps" / spec["name"] / "results.jsonl")
    done = {r["candidate"]: r for r in map(json.loads, results.read_text().splitlines())}
    for candidate in spec["candidates"]:
        if candidate["name"] not in done:
            continue
        rows.append(run_outcomes(done[candidate["name"]]["run_id"]))
        designs.append([2 * coordinate(n, candidate["params"][n], spec.get("narrow", 1.0)) - 1
                        for n in names])
    if not rows:
        sys.exit("no finished runs")
    design = np.array(designs)
    rng = np.random.default_rng(0)
    print(f"{len(rows)} runs; failures {int(sum(r['failure'] for r in rows))}, "
          f"stall-excluded {int(sum(r['stall'] for r in rows))}")
    print("effect = change of the outcome from the lowest to the highest value of the factor")
    for key in ("laps_driven", "failure", "lowest_ey_mean", "lowest_ey_p10",
                "overspeed_mean", "lap_time"):
        outcome = np.array([r[key] for r in rows])
        fit = effects(design, outcome, rng)
        if fit is None:
            continue
        point, low, high, count = fit
        print(f"\n{key} (n={count}, mean {np.nanmean(outcome):.4f})")
        for index in np.argsort(-np.abs(point)):
            flag = "*" if low[index] > 0 or high[index] < 0 else " "
            print(f"  {flag} {names[index]:30s} {2 * point[index]:+9.4f}"
                  f"  [{2 * low[index]:+9.4f}, {2 * high[index]:+9.4f}]")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate")
    gen.add_argument("out")
    gen.add_argument("--runs", type=int, default=64)
    gen.add_argument("--track", default="competition")
    gen.add_argument("--seed", type=int, default=1)
    gen.add_argument("--narrow", type=float, default=1.0,
                     help="shrink every factor range by this factor around its centre")
    gen.set_defaults(handler=generate)
    ana = sub.add_parser("analyze")
    ana.add_argument("spec")
    ana.set_defaults(handler=analyze)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
