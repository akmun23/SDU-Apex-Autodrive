#!/usr/bin/env python3
"""Run repeated positive-drive-only Unity wheel-dynamics experiments."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from pathlib import Path


def run_one(player: Path, root: Path, repeat: int, duration: float,
            timeout: float) -> dict:
    directory = root / f"repeat_{repeat:02d}"
    directory.mkdir(parents=True, exist_ok=True)
    trace = directory / "wheel_contact_trace.csv"
    dump = directory / "simulator_parameters.json"
    log = directory / "player.log"
    if trace.exists() and dump.exists():
        return {"status": "existing", "repeat": repeat,
                "directory": str(directory)}

    environment = os.environ.copy()
    environment.update({
        "AUTODRIVE_MODEL_ID_DIAGNOSTICS_DIR": str(directory),
        "AUTODRIVE_MODEL_ID_EXPERIMENT": "powered_drive_repeat_v1",
        "AUTODRIVE_MODEL_ID_EXPERIMENT_DURATION_SECONDS": f"{duration:.9g}",
        "AUTODRIVE_MODEL_ID_INITIAL_SPEED_MPS": "0",
        "AUTODRIVE_SIMULATOR_BUILD_TAG":
            f"e2-powered-drive-repeat-r{repeat:02d}-20260914",
    })
    started = time.monotonic()
    status = "completed"
    return_code = None
    try:
        completed = subprocess.run(
            [str(player), "-batchmode", "-logFile", str(log)],
            env=environment,
            cwd=player.parent,
            timeout=timeout,
            check=False,
        )
        return_code = completed.returncode
        if return_code != 0:
            status = "nonzero_exit"
    except subprocess.TimeoutExpired:
        status = "timeout"
    row_count = 0
    if trace.exists():
        with trace.open(newline="", encoding="utf-8") as stream:
            row_count = sum(1 for _ in csv.DictReader(stream))
    if not trace.exists() or not dump.exists():
        status = "missing_output"
    return {
        "status": status,
        "return_code": return_code,
        "elapsed_wall_s": time.monotonic() - started,
        "repeat": repeat,
        "directory": str(directory),
        "trace_rows": row_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--player", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--duration-seconds", type=float, default=51.0)
    parser.add_argument("--timeout-seconds", type=float, default=150.0)
    args = parser.parse_args()

    player = args.player.resolve()
    root = args.output_root.resolve()
    if not player.is_file():
        parser.error(f"Unity player does not exist: {player}")
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "autodrive.powered_drive_campaign.v1",
        "experiment": "powered_drive_repeat_v1",
        "player": str(player),
        "normal_graphics_batchmode": True,
        "nographics": False,
        "duration_seconds": args.duration_seconds,
        "repeats": args.repeats,
        "runs": [],
    }
    for repeat in range(1, args.repeats + 1):
        result = run_one(player, root, repeat, args.duration_seconds,
                         args.timeout_seconds)
        manifest["runs"].append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
    manifest["completed_runs"] = sum(
        run["status"] in {"completed", "existing"}
        for run in manifest["runs"])
    manifest["failed_runs"] = args.repeats - manifest["completed_runs"]
    (root / "campaign_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return 0 if manifest["failed_runs"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
