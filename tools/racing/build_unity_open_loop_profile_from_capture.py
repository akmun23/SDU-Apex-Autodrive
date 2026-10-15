#!/usr/bin/env python3
"""Turn an exact-Unity feedback run into a 40 Hz open-loop command replay.

The resulting profile keeps the original warm-up command sequence and samples
the feedback run's actual steering/throttle at each 25 ms command boundary.
It is intended to test how much of a fast Unity lap is explained by its
feed-forward commands alone, before using that sequence as a black-box search
seed. It does not modify the source capture or the original profile.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


FIXED_DT_S = 0.001
TICKS_PER_COMMAND = 25
DEFAULT_RUN = Path(
    "live_runs/unity_source_raceline_feedback_mintime_20261014_r115")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def build_profile(source_profile: dict[str, Any], result: dict[str, Any],
                  capture_path: Path, *, spatial: bool = False,
                  progress_distance_scale: float = 3.0,
                  scored_horizon_commands: int = 360
                  ) -> dict[str, Any]:
    if source_profile.get("raceline_feedback_control") is not True:
        raise ValueError("source profile is not a Unity raceline-feedback run")
    warmup_count = int(source_profile.get("warmup_command_count", 0))
    scored_count = int(result.get("scored_control_updates", 0))
    ticks = int(source_profile.get("physics_ticks_per_command", 0))
    fixed_dt = float(source_profile.get("fixed_delta_time_s", 0.0))
    commands = source_profile.get("commands")
    if (warmup_count <= 0 or scored_count <= 0 or ticks != TICKS_PER_COMMAND
            or abs(fixed_dt - FIXED_DT_S) > 1e-9
            or not isinstance(commands, list) or len(commands) <= warmup_count):
        raise ValueError("source profile/result has an invalid 1 ms / 25 ms schedule")

    with capture_path.open("r", encoding="utf-8", newline="") as stream:
        header = next((line for line in stream if not line.startswith("#")), None)
        if header is None:
            raise ValueError(f"capture has no CSV header: {capture_path}")
        columns = next(csv.reader([header]))
        required = ["sim_time_s", "steering_command", "throttle_command"]
        if spatial:
            required.extend(("pos_world_x_m", "pos_world_z_m"))
        missing = [name for name in required if name not in columns]
        if missing:
            raise ValueError(f"capture is missing fields: {missing}")
        indices = [columns.index(name) for name in required]
        values = np.loadtxt(stream, delimiter=",", usecols=indices,
                            dtype=np.float64, ndmin=2)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("capture is empty or contains non-finite control samples")
    times = values[:, 0]
    dt = np.diff(times)
    if np.max(np.abs(dt - FIXED_DT_S)) > 2e-7:
        raise ValueError("capture is not a continuous 1 kHz fixed-step sequence")
    first_time = float(times[0])
    expected_indices = np.arange(len(times), dtype=np.float64)
    if np.max(np.abs((times - first_time) / FIXED_DT_S - expected_indices)) > 2e-3:
        raise ValueError("capture timestamps do not map cleanly to physics-step indices")

    first_scored_step = warmup_count * ticks
    sample_indices = first_scored_step + np.arange(scored_count) * ticks
    if sample_indices[-1] >= len(values):
        raise ValueError("capture ends before the full scored control sequence")
    if abs(times[first_scored_step] - first_time - first_scored_step * FIXED_DT_S) > 2e-6:
        raise ValueError("warm-up boundary does not align with capture fixed steps")

    output_commands = []
    for row in commands[:warmup_count]:
        warmup_command = {
            "steering_norm": float(row["steering_norm"]),
            "throttle_norm": float(row["throttle_norm"]),
        }
        if spatial:
            warmup_command.update({
                "has_reference_position": False,
                "reference_x_m": 0.0,
                "reference_z_m": 0.0,
            })
        output_commands.append(warmup_command)
    for index in sample_indices:
        steering, throttle = map(float, values[index, 1:3])
        if abs(steering) > 1.0 or abs(throttle) > 1.0:
            raise ValueError(f"captured command outside [-1, 1] at row {index}")
        command = {
            "steering_norm": steering,
            "throttle_norm": throttle,
        }
        if spatial:
            command.update({
                "has_reference_position": True,
                "reference_x_m": float(values[index, 3]),
                "reference_z_m": float(values[index, 4]),
            })
        output_commands.append(command)
    profile = {
        "schema_version": 1,
        "fixed_delta_time_s": FIXED_DT_S,
        "physics_ticks_per_command": TICKS_PER_COMMAND,
        "command_phase_fixed_steps": 0,
        "spatial_command_lookup": spatial,
        "commands": output_commands,
    }
    if spatial:
        if (not np.isfinite(progress_distance_scale) or
                progress_distance_scale < 0.0 or
                scored_horizon_commands <= 0):
            raise ValueError("invalid spatial progress scale or command horizon")
        profile.update({
            "spatial_progress_distance_scale": progress_distance_scale,
            "warmup_command_count": warmup_count,
            "scored_horizon_commands": scored_horizon_commands,
        })
    return profile


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spatial", action="store_true",
                        help="attach each command to its captured Unity position")
    parser.add_argument("--progress-distance-scale", type=float, default=3.0)
    parser.add_argument("--scored-horizon-commands", type=int, default=360)
    args = parser.parse_args()
    run = args.run.resolve()
    source = _read_json(run / "raceline_feedback_profile.json")
    result = _read_json(run / "dynamics/offline_result.json")
    capture = run / "dynamics" / "wheel_dynamics_feedback_mintime_20261014_r115.csv"
    if not capture.is_file():
        matches = sorted((run / "dynamics").glob("wheel_dynamics_*.csv"))
        if len(matches) != 1:
            raise FileNotFoundError(f"could not identify one dynamics capture in {run}")
        capture = matches[0]
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    profile = build_profile(
        source, result, capture, spatial=args.spatial,
        progress_distance_scale=args.progress_distance_scale,
        scored_horizon_commands=args.scored_horizon_commands)
    output.write_text(json.dumps(profile, separators=(",", ":")) + "\n",
                      encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "warmup_commands": int(source["warmup_command_count"]),
        "scored_commands": int(result["scored_control_updates"]),
        "total_commands": len(profile["commands"]),
        "sample_interval_ms": TICKS_PER_COMMAND,
        "feedback_controller_removed": True,
        "spatial_command_lookup": args.spatial,
    }, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
