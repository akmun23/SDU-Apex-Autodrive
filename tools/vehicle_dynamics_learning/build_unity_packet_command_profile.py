#!/usr/bin/env python3
"""Export the commands actually sent for each packet in a closed Unity bag.

This is a development-only input for deterministic simulator parity replay.
It avoids replaying ROS command callbacks on a separate executor, where queued
callbacks can publish stale samples in a burst between simulator requests.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from pathlib import Path


PACKET_TIMING_TOPIC = "/autodrive/roboracer_1/bridge_packet_timing"


def _read_packet_rows(bag: Path) -> list[dict[str, object]]:
    database = bag / "run_0.db3" if bag.is_dir() else bag
    if not database.is_file():
        raise ValueError(f"closed bag database not found: {database}")
    connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        topic = connection.execute(
            "SELECT id FROM topics WHERE name = ?", (PACKET_TIMING_TOPIC,)
        ).fetchone()
        if topic is None:
            raise ValueError(f"bag has no {PACKET_TIMING_TOPIC} topic")
        rows = []
        for (serialized,) in connection.execute(
                "SELECT data FROM messages WHERE topic_id = ? ORDER BY timestamp",
                (int(topic[0]),)):
            raw = bytes(serialized)
            if len(raw) < 8:
                raise ValueError("packet timing message has invalid CDR data")
            length = int.from_bytes(raw[4:8], "little", signed=False)
            if length < 2 or 8 + length > len(raw):
                raise ValueError("packet timing message has invalid CDR string length")
            payload = json.loads(raw[8:8 + length].rstrip(b"\0"))
            rows.append(payload)
        return rows
    finally:
        connection.close()


def build_profile(bag: Path,
                  target_first_request_sequence: int | None = None,
                  command_phase_fixed_steps: int = 0,
                  ) -> dict[str, object]:
    if (target_first_request_sequence is not None and
            target_first_request_sequence < 1):
        raise ValueError("target first request sequence must be positive")
    if command_phase_fixed_steps not in (0, 1):
        raise ValueError("command phase must be 0 or 1 fixed step")
    rows = _read_packet_rows(bag)
    rows.sort(key=lambda row: int(row.get("request_sequence", -1)))
    commands = []
    source_requests = []
    for row in rows:
        request = row.get("request_sequence")
        steering = row.get("sent_steering_norm")
        throttle = row.get("sent_throttle_norm")
        if request is None or steering is None or throttle is None:
            continue
        request = int(request)
        steering = float(steering)
        throttle = float(throttle)
        if (not math.isfinite(steering) or not math.isfinite(throttle)
                or abs(steering) > 1.0 or abs(throttle) > 1.0):
            raise ValueError(f"invalid sent command at source request {request}")
        if source_requests and request <= source_requests[-1]:
            raise ValueError("source request sequence is duplicated or unordered")
        source_requests.append(request)
        commands.append({"steering_norm": steering, "throttle_norm": throttle})
    if len(commands) < 100:
        raise ValueError("source packet command profile is too short")
    if any(b != a + 1 for a, b in zip(source_requests, source_requests[1:])):
        raise ValueError("source request sequence contains a gap")
    if any(abs(float(value)) > 1.0e-6
           for value in (commands[0]["steering_norm"], commands[0]["throttle_norm"])):
        raise ValueError("profile must begin at neutral so startup packets remain safe")
    return {
        "schema_version": 1,
        "fixed_delta_time_s": 0.001,
        "physics_ticks_per_command": 25,
        "command_phase_fixed_steps": command_phase_fixed_steps,
        "source_bag": str(bag.resolve()),
        "source_first_request_sequence": source_requests[0],
        "source_last_request_sequence": source_requests[-1],
        "target_first_request_sequence": target_first_request_sequence,
        "commands": commands,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("closed_bag", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--target-first-request-sequence", type=int)
    parser.add_argument("--command-phase-fixed-steps", type=int, choices=(0, 1),
                        default=0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    profile = build_profile(args.closed_bag,
                            args.target_first_request_sequence,
                            args.command_phase_fixed_steps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "source_request_range": [profile["source_first_request_sequence"],
                                 profile["source_last_request_sequence"]],
        "target_first_request_sequence": profile["target_first_request_sequence"],
        "command_phase_fixed_steps": profile["command_phase_fixed_steps"],
        "command_pairs": len(profile["commands"]),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
