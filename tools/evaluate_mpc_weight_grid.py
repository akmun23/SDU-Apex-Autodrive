#!/usr/bin/env python3
"""Screen MPC objective weights against real MPC-authority event streams.

This is an offline *candidate screen*, not a replacement for a live authority
run.  It replays the runtime-legal /odom, /current_map_pose, and /cmd/speed
events through the same RTI core and records solver/nonlinear/corridor/action
metrics for each weight vector.  The measured vehicle state is not replaced by
simulator truth and the tool never uses PP shadow data.

The state trajectory is fixed by the recorded authority run, so this tool can
compare feasibility and local MPC behaviour quickly, but it cannot predict a
candidate's closed-loop lap time.  The highest-ranked candidates must still be
validated with the MPC as the live authority in batchmode.

Candidate CSV format:

    name,e_y,e_psi,u,u_overspeed,target_speed_state,v,r,steering_command,
    steering_rate,target_speed_rate,steering_rate_change,
    target_speed_rate_change,terminal_multiplier

Blank fields use the production defaults.  If --candidates is omitted, a small
set is evaluated that includes the current production profile, the
BachelorProject profile, and moderate-heading/velocity-priority profiles.
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class Weights:
    name: str = "production"
    e_y: float = 150.0
    e_psi: float = 10.0
    u: float = 50.0
    u_overspeed: float = 150.0
    target_speed_state: float = 20.0
    v: float = 0.0
    r: float = 1.5
    steering_command: float = 5.0
    steering_rate: float = 5.0
    target_speed_rate: float = 0.5
    steering_rate_change: float = 10.0
    target_speed_rate_change: float = 5.0
    terminal_multiplier: float = 3.0


WEIGHT_KEYS = [field.name for field in fields(Weights) if field.name != "name"]
OPTION_NAMES = {
    "e_y": "--weight-ey",
    "e_psi": "--weight-epsi",
    "u": "--weight-u",
    "u_overspeed": "--weight-u-overspeed",
    "target_speed_state": "--weight-target-speed-state",
    "v": "--weight-v",
    "r": "--weight-r",
    "steering_command": "--weight-steering-command",
    "steering_rate": "--weight-steering-rate",
    "target_speed_rate": "--weight-target-speed-rate",
    "steering_rate_change": "--weight-steering-rate-change",
    "target_speed_rate_change": "--weight-target-speed-rate-change",
    "terminal_multiplier": "--terminal-multiplier",
}


def default_candidates() -> list[Weights]:
    production = Weights()
    bachelor = Weights(
        name="bachelor_project",
        e_y=1500.0,
        e_psi=50.0,
        u=200.0,
        u_overspeed=200.0,
        v=5.0,
        r=1.5,
        steering_command=2.0,
        steering_rate=5.0,
        target_speed_rate=0.5,
        steering_rate_change=5.0,
        target_speed_rate_change=5.0,
    )
    return [
        production,
        bachelor,
        Weights(name="moderate_heading_velocity", e_y=100.0, e_psi=2.0,
                u=200.0, u_overspeed=200.0),
        Weights(name="low_heading_velocity", e_y=100.0, e_psi=1.0,
                u=250.0, u_overspeed=250.0),
        Weights(name="moderate_lateral_high_velocity", e_y=75.0,
                e_psi=2.0, u=300.0, u_overspeed=250.0),
        Weights(name="moderate_lateral_low_heading", e_y=150.0,
                e_psi=2.0, u=200.0, u_overspeed=200.0),
    ]


def parse_candidates(path: Path) -> list[Weights]:
    result: list[Weights] = []
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            values: dict[str, object] = {"name": row.get("name", "candidate")}
            for key in WEIGHT_KEYS:
                raw = row.get(key, "")
                if raw not in (None, ""):
                    values[key] = float(raw)
            result.append(Weights(**values))
    if not result:
        raise ValueError(f"candidate file is empty: {path}")
    return result


def parse_summary(stdout: str) -> dict[str, str]:
    metrics: dict[str, str] = {}
    for line in stdout.splitlines():
        if line.startswith("trace="):
            metrics["trace"] = line.removeprefix("trace=")
        match = re.search(
            r"solves=(\d+) accepted=(\d+) degraded=(\d+) rejected=(\d+)",
            line,
        )
        if match:
            metrics.update(dict(zip(
                ("solves", "accepted", "degraded", "rejected"),
                match.groups(),
            )))
            continue
        match = re.match(
            r"rejection_statuses\[input,solver,residual,regularization,nonlinear\]="
            r"\[(\d+),(\d+),(\d+),(\d+),(\d+)\]",
            line,
        )
        if match:
            metrics.update(dict(zip(
                ("reject_input", "reject_solver", "reject_residual",
                 "reject_regularization", "reject_nonlinear"),
                match.groups(),
            )))
            continue
        for prefix, key, pattern in (
            ("residual_p95[primal,dual]=", "residual_p95", r"([^,]+),([^,]+)"),
            ("predicted_min_corridor_clearance_m=", "min_predicted_clearance", r"(.+)"),
            ("solve_ms[p50,p95,p99,max]=", "solve_ms", r"(.+)"),
            ("iterations[p50,p95,p99]=", "iterations", r"(.+)"),
            ("first_action_delta[abs_steering_rate_p95,max,abs_target_rate_p95,max]=",
             "first_action_delta", r"(.+)"),
            ("prediction[horizon_steps,dt_s]=", "prediction", r"(.+)"),
        ):
            if line.startswith(prefix):
                match = re.fullmatch(pattern, line[len(prefix):])
                if match:
                    metrics[key] = ",".join(match.groups())
                break
    return metrics


def run_one(binary: str, event_path: str, candidate: Weights,
            max_iterations: int, tolerance: float,
            yaw_options: list[str]) -> dict[str, object]:
    command = [
        binary, event_path, str(max_iterations), str(tolerance),
        "--prefactorized", "--adaptive-rho", "--rti-mode", "adaptive",
        "--corridor-margin", "0.30",
        "--first-prediction-corridor-margin", "0.30",
    ]
    command.extend(yaw_options)
    for key in WEIGHT_KEYS:
        command.extend([OPTION_NAMES[key], str(getattr(candidate, key))])
    completed = subprocess.run(command, capture_output=True, text=True)
    metrics = parse_summary(completed.stdout)
    result: dict[str, object] = {
        "event_stream": event_path,
        "candidate": candidate.name,
        **{key: getattr(candidate, key) for key in WEIGHT_KEYS},
        "return_code": completed.returncode,
        "stderr": completed.stderr.strip(),
        **metrics,
    }
    solves = int(metrics.get("solves", "0"))
    accepted = int(metrics.get("accepted", "0"))
    nonlinear = int(metrics.get("reject_nonlinear", "0"))
    residual = int(metrics.get("reject_residual", "0"))
    result["accepted_fraction"] = accepted / solves if solves else 0.0
    result["nonlinear_rejection_fraction"] = nonlinear / solves if solves else 1.0
    result["residual_rejection_fraction"] = residual / solves if solves else 1.0
    # This ranks solver/model compatibility only.  It deliberately does not
    # claim to rank closed-loop lap time, since the state path is fixed.
    min_clearance = float(metrics.get("min_predicted_clearance", "-inf"))
    result["offline_screen_score"] = (
        result["accepted_fraction"]
        - 0.5 * result["nonlinear_rejection_fraction"]
        - 0.25 * result["residual_rejection_fraction"]
        + min(0.0, min_clearance) * 2.0
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True,
                        help="mpc_rti_offline_replay executable")
    parser.add_argument("--events", nargs="+", required=True,
                        help="MPC-authority events.csv files")
    parser.add_argument("--candidates", type=Path,
                        help="candidate CSV; defaults to built-in profiles")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--tolerance", type=float, default=0.01)
    parser.add_argument("--yaw-rate-tau", type=float,
                        help="optional identified yaw response time constant")
    parser.add_argument("--yaw-rate-gain", type=float,
                        help="optional identified yaw steering gain")
    parser.add_argument("--yaw-curvature-reduction", type=float,
                        help="optional high-curvature gain reduction per metre")
    parser.add_argument("--yaw-curvature-start", type=float,
                        help="curvature at which the gain correction starts")
    parser.add_argument("--yaw-curvature-end", type=float,
                        help="curvature at which the gain correction saturates")
    parser.add_argument("--yaw-low-speed-tau", type=float,
                        help="optional low-speed yaw response time constant")
    parser.add_argument("--yaw-low-speed-transition", type=float,
                        help="speed scale for the low-speed yaw response term")
    args = parser.parse_args()

    candidates = (parse_candidates(args.candidates)
                  if args.candidates else default_candidates())
    yaw_options: list[str] = []
    for value, option in (
        (args.yaw_rate_tau, "--yaw-rate-tau"),
        (args.yaw_rate_gain, "--yaw-rate-gain"),
        (args.yaw_curvature_reduction, "--yaw-curvature-reduction"),
        (args.yaw_curvature_start, "--yaw-curvature-start"),
        (args.yaw_curvature_end, "--yaw-curvature-end"),
        (args.yaw_low_speed_tau, "--yaw-low-speed-tau"),
        (args.yaw_low_speed_transition, "--yaw-low-speed-transition"),
    ):
        if value is not None:
            yaw_options.extend([option, str(value)])
    jobs = max(1, args.jobs)
    tasks = [(event, candidate) for event in args.events
             for candidate in candidates]
    rows: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {
            executor.submit(run_one, args.binary, event, candidate,
                            args.max_iterations, args.tolerance,
                            yaw_options): (event, candidate)
            for event, candidate in tasks
        }
        for future in as_completed(futures):
            rows.append(future.result())

    rows.sort(key=lambda row: (
        -float(row.get("offline_screen_score", float("-inf"))),
        str(row["candidate"]), str(row["event_stream"]),
    ))
    columns = ["event_stream", "candidate", *WEIGHT_KEYS,
               "return_code", "solves", "accepted", "degraded", "rejected",
               "reject_input", "reject_solver", "reject_residual",
               "reject_regularization", "reject_nonlinear",
               "accepted_fraction", "nonlinear_rejection_fraction",
               "residual_rejection_fraction", "min_predicted_clearance",
               "residual_p95", "solve_ms", "iterations", "first_action_delta",
               "prediction", "offline_screen_score", "stderr"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} replay results to {args.output}")
    print("offline_screen_score ranks fixed-trace solver/model compatibility; "
          "live MPC-authority validation is still required for lap time.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
