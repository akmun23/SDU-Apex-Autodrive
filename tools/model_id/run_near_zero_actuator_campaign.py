#!/usr/bin/env python3
"""Run the disposable Unity near-zero actuator identification campaign.

The player is a normal-graphics Unity build. ``-batchmode`` only removes the
interactive window; ``-nographics`` is deliberately not used because the
experiment must exercise the same rendered simulator build path.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from pathlib import Path


DEFAULT_SPEEDS = (3.0, 6.0, 9.0, 12.0, 15.0)
DEFAULT_COMMANDS = (0.0, 0.001, 0.002, 0.005, 0.010, 0.020, 0.050)


def condition_token(value: float) -> str:
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    return (text or "0").replace(".", "p")


def run_one(player: Path, output_root: Path, speed: float, command: float,
            repeat: int, warmup_seconds: float, warmup_throttle: float,
            measurement_seconds: float, timeout_seconds: float) -> dict:
    directory = output_root / (
        f"speed_{condition_token(speed)}mps_command_"
        f"{condition_token(command)}_repeat_{repeat:02d}")
    directory.mkdir(parents=True, exist_ok=True)
    trace = directory / "wheel_contact_trace.csv"
    dump = directory / "simulator_parameters.json"
    log = directory / "player.log"
    if trace.exists() and dump.exists():
        return {
            "status": "existing",
            "speed_mps": speed,
            "command_norm": command,
            "repeat": repeat,
            "directory": str(directory),
        }

    environment = os.environ.copy()
    environment.update({
        "AUTODRIVE_MODEL_ID_DIAGNOSTICS_DIR": str(directory),
        "AUTODRIVE_MODEL_ID_EXPERIMENT": "near_zero_actuator_v1",
        "AUTODRIVE_MODEL_ID_INITIAL_SPEED_MPS": f"{speed:.9g}",
        "AUTODRIVE_MODEL_ID_THROTTLE_NORM": f"{command:.9g}",
        "AUTODRIVE_MODEL_ID_WARMUP_SECONDS": f"{warmup_seconds:.9g}",
        "AUTODRIVE_MODEL_ID_WARMUP_THROTTLE_NORM": f"{warmup_throttle:.9g}",
        "AUTODRIVE_MODEL_ID_MEASUREMENT_SECONDS": f"{measurement_seconds:.9g}",
        "AUTODRIVE_SIMULATOR_BUILD_TAG": (
            f"e1-near-zero-s{condition_token(speed)}-"
            f"c{condition_token(command)}-r{repeat:02d}-20260914"
        ),
    })
    started = time.monotonic()
    status = "completed"
    return_code = None
    try:
        completed = subprocess.run(
            [str(player), "-batchmode", "-logFile", str(log)],
            env=environment,
            cwd=player.parent,
            timeout=timeout_seconds,
            check=False,
        )
        return_code = completed.returncode
        if return_code != 0:
            status = "nonzero_exit"
    except subprocess.TimeoutExpired:
        status = "timeout"
    elapsed = time.monotonic() - started
    if not trace.exists() or not dump.exists():
        status = "missing_output"

    row_count = 0
    if trace.exists():
        with trace.open(newline="") as stream:
            row_count = sum(1 for _ in csv.DictReader(stream))
    return {
        "status": status,
        "return_code": return_code,
        "elapsed_wall_s": elapsed,
        "speed_mps": speed,
        "command_norm": command,
        "repeat": repeat,
        "directory": str(directory),
        "trace_rows": row_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--player", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--speeds", type=float, nargs="+", default=DEFAULT_SPEEDS)
    parser.add_argument("--commands", type=float, nargs="+", default=DEFAULT_COMMANDS)
    parser.add_argument("--warmup-seconds", type=float, default=0.5)
    parser.add_argument("--warmup-throttle", type=float, default=0.65)
    parser.add_argument("--measurement-seconds", type=float, default=1.5)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    player = args.player.resolve()
    output_root = args.output_root.resolve()
    if not player.is_file():
        parser.error(f"Unity player does not exist: {player}")
    output_root.mkdir(parents=True, exist_ok=True)

    planned = len(args.speeds) * len(args.commands) * args.repeats
    manifest = {
        "schema": "autodrive.near_zero_actuator_campaign.v1",
        "experiment": "near_zero_actuator_v1",
        "player": str(player),
        "normal_graphics_batchmode": True,
        "nographics": False,
        "warmup_seconds": args.warmup_seconds,
        "warmup_throttle_norm": args.warmup_throttle,
        "measurement_seconds": args.measurement_seconds,
        "speeds_mps": args.speeds,
        "commands_norm": args.commands,
        "repeats": args.repeats,
        "planned_runs": planned,
        "runs": [],
    }
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return 0

    for speed in args.speeds:
        for command in args.commands:
            for repeat in range(1, args.repeats + 1):
                result = run_one(
                    player, output_root, speed, command, repeat,
                    args.warmup_seconds, args.warmup_throttle,
                    args.measurement_seconds, args.timeout_seconds)
                manifest["runs"].append(result)
                print(json.dumps(result, sort_keys=True), flush=True)

    manifest["completed_runs"] = sum(
        run["status"] in {"completed", "existing"}
        for run in manifest["runs"])
    manifest["failed_runs"] = planned - manifest["completed_runs"]
    (output_root / "campaign_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return 0 if manifest["failed_runs"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
