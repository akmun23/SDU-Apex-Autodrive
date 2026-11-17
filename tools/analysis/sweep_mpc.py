#!/usr/bin/env python3
"""Resumable parameter sweep over closed-loop Unity trials.

    tools/analysis/sweep_mpc.py run SPEC.json      # run the candidates not yet done
    tools/analysis/sweep_mpc.py report SPEC.json   # ranked table of what has run

SPEC.json:
    {"name": "p1", "track": "practice", "laps": 6,          # warm-up + scored
     "runs_per_candidate": 1,
     "interleave": true,                                   # optional: one run of every
                                                           # candidate per round
     "base": {"weight_u": 65.0},                           # overlay for every candidate
     "candidates": [{"name": "u90", "params": {"weight_u": 90.0}},
                    {"name": "line2", "params": {},
                     "env": {"SDU_APEX_TRIAL_TRAJECTORY_FILE": "f1tenth_planning/..."}}]}
"""
import json
import shlex
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENVIRONMENT = "unset VIRTUAL_ENV; source /opt/ros/jazzy/setup.bash >/dev/null 2>&1"


def results_path(spec):
    return ROOT / "live_runs" / "sweeps" / spec["name"] / "results.jsonl"


def load_results(spec):
    path = results_path(spec)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_overlay(path, params):
    lines = ["/**:", "  ros__parameters:" if params else "  ros__parameters: {}"]
    lines += [f"    {key}: {json.dumps(value)}" for key, value in params.items()]
    path.write_text("\n".join(lines) + "\n")


# Candidate env overrides apply to this trial only. Completed result rows
# are skipped on resume; each supervised run uses CPU isolation and a lap limit.
def run_trial(spec, candidate, index):
    run_id = f"sweep_{spec['name']}_{candidate['name']}_{index}"
    overlay = ROOT / "live_runs" / "_overlays" / f"{run_id}.yaml"
    write_overlay(overlay, {**spec.get("base", {}), **candidate["params"]})
    laps = spec["laps"]
    extra = "".join(f"{key}={shlex.quote(str(value))} "
                    for key, value in candidate.get("env", {}).items())
    trial = (f"{extra}SDU_APEX_MPC_PARAMETER_OVERLAY={overlay.relative_to(ROOT)} "
             f"SDU_APEX_TRIAL_TOTAL_LAPS={laps} SDU_APEX_MPC_NATIVE_TUNING=true "
             f"SDU_APEX_MPC_LTO=true tools/sim/run_trial.sh {spec['track']} {run_id}")
    exit_code = subprocess.run(["bash", "-c", f"cd {ROOT} && {trial}"], stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL).returncode
    subprocess.run(["bash", "-c", f"cd {ROOT} && {ENVIRONMENT}; python3 tools/analysis/summarize_trial.py "
                    f"live_runs/{run_id} --keep-export live_runs/{run_id}/analysis"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # The supervisor's exit code is the outcome: 0 means the lap target was
    # reached. A rejected cycle that the car survived does not fail the run.
    row = {"candidate": candidate["name"], "index": index, "run_id": run_id,
           "params": candidate["params"], "laps_requested": laps,
           "completed": exit_code == 0, "exit": exit_code, "lap_times": []}
    try:
        report = json.loads((ROOT / "live_runs" / run_id / "analysis" / "report.json").read_text())
        summary = report["summary"]
        times = summary.get("recorded_lap_times_s") or []
        row["lap_times"] = times
        row["cutoff"] = summary.get("score_cutoff_reason")
    except (OSError, KeyError, json.JSONDecodeError):
        row["cutoff"] = "no_report"
    return row


def run(spec):
    path = results_path(spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = {(row["candidate"], row["index"]) for row in load_results(spec)}
    # interleave: one run of every candidate per round, so slow drift of the
    # machine affects all candidates equally.
    runs = spec.get("runs_per_candidate", 1)
    if spec.get("interleave"):
        order = [(c, i) for i in range(runs) for c in spec["candidates"]]
    else:
        order = [(c, i) for c in spec["candidates"] for i in range(runs)]
    for candidate, index in order:
        if (candidate["name"], index) in done:
            continue
        row = run_trial(spec, candidate, index)
        with path.open("a") as out:
            out.write(json.dumps(row) + "\n")
        scored = row["lap_times"][1:]
        print(f"{row['run_id']}: {'complete' if row['completed'] else row.get('cutoff')}"
              f" laps={len(scored)}" +
              (f" mean={statistics.mean(scored):.3f}" if scored else ""), flush=True)


def score(rows):
    """(completion rate, mean scored lap, SD, worst lap) over a candidate's runs."""
    complete = [row for row in rows if row["completed"]]
    scored = [t for row in rows for t in row["lap_times"][1:]]
    if not scored:
        return len(complete) / len(rows), None, None, None
    return (len(complete) / len(rows), statistics.mean(scored),
            statistics.pstdev(scored), max(scored))


# Rank completion first, then mean scored lap time plus twice its standard
# deviation. SDU_APEX_WATCH_CONTINUE_ON_MPC_REJECTION=1 permits rejected cycles.
def report(spec):
    rows = load_results(spec)
    table = []
    for candidate in spec["candidates"]:
        mine = [row for row in rows if row["candidate"] == candidate["name"]]
        if not mine:
            continue
        rate, mean, sd, worst = score(mine)
        key = (-rate, float("inf") if mean is None else mean + 2. * sd)
        table.append((key, candidate, len(mine), rate, mean, sd, worst))
    table.sort(key=lambda item: item[0])
    print(f"{'candidate':24s} {'runs':>4s} {'done':>5s} {'mean':>7s} {'sd':>6s} {'worst':>7s}  params")
    for _, candidate, runs, rate, mean, sd, worst in table:
        fmt = (lambda v: f"{v:7.3f}" if v is not None else "      -")
        print(f"{candidate['name']:24s} {runs:4d} {rate:5.0%} {fmt(mean)} "
              f"{fmt(sd).strip():>6s} {fmt(worst)}  {candidate['params']}")


def main():
    if len(sys.argv) != 3 or sys.argv[1] not in ("run", "report"):
        print(__doc__)
        return 2
    spec = json.loads(Path(sys.argv[2]).read_text())
    (run if sys.argv[1] == "run" else report)(spec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
