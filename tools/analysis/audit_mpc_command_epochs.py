#!/usr/bin/env python3
"""Audit issued MPC payloads against the next recorded host bridge request.

Uses RUN/analysis/diag.jsonl plus bridge timing JSON from the bag. A later
publication before the next request supersedes the earlier payload. This
checks host request epochs, not unrecorded Unity FixedUpdate application times.
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import sqlite3
import struct
from pathlib import Path


def analyze(run):
    publications = sorted((row["payload"] for row in map(json.loads,
        (run / "analysis/diag.jsonl").open())
        if row["payload"].get("command_published") and row["payload"].get("first_action")),
        key=lambda p: p["issued_command_stamp_ns"])
    stamps = [p["issued_command_stamp_ns"] for p in publications]
    requests = {}
    for database in sorted((run / "run").glob("*.db3")):
        with sqlite3.connect(database) as connection:
            ids = [row[0] for row in connection.execute(
                "select id from topics where name like '%/bridge_packet_timing'")]
            for topic_id in ids:
                for data, in connection.execute("select data from messages where topic_id=?", (topic_id,)):
                    size, = struct.unpack_from("<I", data, 4)
                    packet = json.loads(data[8:8+size-1])
                    requests[packet["bridge_request_ros_stamp_ns"]] = packet
    request_stamps = sorted(requests)
    superseded = missing_request = mismatched_payload = wrong_epoch = matched = 0
    failures = []
    continuation_matched = 0
    for i, publication in enumerate(publications):
        issued = publication["issued_command_stamp_ns"]
        j = bisect.bisect_right(request_stamps, issued)
        if j == len(request_stamps):
            missing_request += 1
            continue
        actual = request_stamps[j]
        if i+1 < len(stamps) and stamps[i+1] < actual:
            superseded += 1
            continue
        intended = publication["candidate_command_stamp_ns"]
        delta_ms = (actual-intended)*1.e-6
        packet = requests[actual]
        steering, throttle = publication["first_action"]
        payload_ok = (abs(packet["sent_steering_norm"]-round(steering/(math.pi/6), 3)) < 1.e-6
                      and abs(packet["sent_throttle_norm"]-round(throttle, 3)) < 1.e-6)
        epoch_ok = abs(delta_ms) < 12.5
        wrong_epoch += not epoch_ok
        mismatched_payload += not payload_ok
        matched += epoch_ok and payload_ok
        continuation_matched += epoch_ok and payload_ok and publication["status"] == "plan_continuation"
        if not payload_ok or not epoch_ok:
            failures.append(dict(issued_ns=issued, intended_ns=intended,
                actual_request_ns=actual, delta_ms=delta_ms, payload_ok=payload_ok,
                status=publication["status"]))
    return dict(run=run.name, publications=len(publications), matched=matched,
        continuation_matched=continuation_matched, superseded=superseded,
        no_later_recorded_request=missing_request, wrong_epoch=wrong_epoch,
        mismatched_payload=mismatched_payload, failures=failures)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = json.dumps([analyze(run) for run in args.runs], indent=2) + "\n"
    if args.output:
        args.output.write_text(result)
    else:
        print(result, end="")


if __name__ == "__main__":
    main()
