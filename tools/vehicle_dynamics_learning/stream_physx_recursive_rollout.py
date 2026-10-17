#!/usr/bin/env python3
"""Run an existing recursive PhysX command rollout through its persistent server.

The server advances every recorded 1 ms command tick but retains only the
configured output stride. An optional batch CSV is compared at those exact
output ticks. This is an offline plant-engine check, not a ROS runtime node.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np


def _array(values: list[float]) -> str:
    return " ".join(f"{float(value):.9g}" for value in values)


def _read_batch(path: Path) -> list[list[str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    if any(len(row) != 6 for row in rows):
        raise ValueError("batch reference must use the six-column rollout CSV format")
    return rows


def run(input_path: Path, binary: Path, output_csv: Path,
        reference_csv: Path | None = None,
        control_block_steps: int = 25) -> dict[str, Any]:
    rollout = json.loads(input_path.read_text(encoding="utf-8"))
    if rollout.get("mode") != "recursive_command_only":
        raise ValueError("input must be a recursive command-only rollout")
    steps = rollout.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("rollout contains no command steps")
    if control_block_steps != 25:
        raise ValueError("the production command interval is fixed at 25 physics ticks")
    if len(steps) % control_block_steps:
        raise ValueError("command count must contain complete 25 ms control intervals")
    output_count = len(steps) // control_block_steps
    if reference_csv is not None:
        reference = _read_batch(reference_csv)
        if len(reference) != len(steps):
            raise ValueError("batch reference and rollout output counts differ")
    else:
        reference = None

    initial = {
        "mode": "interactive_recursive_server",
        "initial": rollout["initial"],
    }
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    max_position_error = 0.0
    max_orientation_error = 0.0
    max_velocity_error = 0.0
    written = 0
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="physx_stream_") as temp_dir:
        initial_path = Path(temp_dir) / "initial.json"
        initial_path.write_text(json.dumps(initial, separators=(",", ":")),
                                encoding="utf-8")
        process = subprocess.Popen(
            [str(binary.resolve()), "--server", str(initial_path),
             str(Path(rollout["shape_audit"]).resolve())],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None,
            text=True, bufsize=1)
        assert process.stdin is not None and process.stdout is not None
        try:
            with output_csv.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                for stage_index in range(output_count):
                    begin = stage_index * control_block_steps
                    stage_steps = steps[begin:begin + control_block_steps]
                    stage = {
                        "capture_row": int(stage_steps[0]["capture_row"]),
                        "time_s": float(stage_steps[0]["time_s"]),
                        "wheel_commands_by_tick": [
                            step["wheel_commands"] for step in stage_steps
                        ],
                    }
                    if any(
                        int(step["capture_row"]) != int(stage_steps[0]["capture_row"]) + offset
                        for offset, step in enumerate(stage_steps)
                    ):
                        raise ValueError(f"non-contiguous capture rows in stage {stage_index}")
                    process.stdin.write(json.dumps(
                        {"op": "rollout", "stages": [stage]},
                        separators=(",", ":")) + "\n")
                    process.stdin.flush()
                    line = process.stdout.readline()
                    if not line:
                        raise RuntimeError(
                            f"PhysX server exited before control stage {stage_index}")
                    state = json.loads(line)

                    position = state["position_world_m"]
                    rotation = state["rotation_world_xyzw"]
                    velocity = state["linear_velocity_world_mps"]
                    row = [
                        int(state["capture_row"]),
                        float(state["time_s"]),
                        float(position[0]), float(position[2]),
                        _array(rotation), _array(velocity),
                    ]
                    writer.writerow(row)
                    if reference is not None:
                        # The native streamed reference records each 1 ms step;
                        # compare the block endpoint, after all 25 commands.
                        ref = reference[begin + control_block_steps - 1]
                        max_position_error = max(
                            max_position_error,
                            abs(row[2] - float(ref[2])),
                            abs(row[3] - float(ref[3])))
                        q_ref = np.fromstring(ref[4], sep=" ")
                        v_ref = np.fromstring(ref[5], sep=" ")
                        max_orientation_error = max(
                            max_orientation_error,
                            float(np.max(np.abs(np.asarray(rotation) - q_ref))))
                        max_velocity_error = max(
                            max_velocity_error,
                            float(np.max(np.abs(np.asarray(velocity) - v_ref))))
                    written += 1
                process.stdin.write('{"op":"shutdown"}\n')
                process.stdin.flush()
                process.stdin.close()
            return_code = process.wait(timeout=30)
            if return_code != 0:
                raise RuntimeError(f"PhysX server exited with status {return_code}")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()

    elapsed = time.perf_counter() - started
    duration = len(steps) * float(rollout["initial"]["fixed_dt_s"])
    report = {
        "input": str(input_path.resolve()),
        "output_csv": str(output_csv.resolve()),
        "reference_csv": str(reference_csv.resolve()) if reference_csv else None,
        "simulation_duration_s": duration,
        "physics_steps": len(steps),
        "physics_steps_per_control_stage": control_block_steps,
        "output_samples": written,
        "wall_duration_s": elapsed,
        "real_time_factor": duration / elapsed,
        "streaming_batch_max_abs_delta": {
            "position_xz_m": max_position_error if reference is not None else None,
            "quaternion_component": (
                max_orientation_error if reference is not None else None),
            "world_velocity_mps": max_velocity_error if reference is not None else None,
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--reference-csv", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--control-block-steps", type=int, default=25)
    args = parser.parse_args()
    result = run(args.input, args.binary, args.output_csv, args.reference_csv,
                 args.control_block_steps)
    encoded = json.dumps(result, indent=2)
    print(encoded)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
