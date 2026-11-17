#!/usr/bin/env python3
"""Align state receipt, controller epochs and shutdown using RUN/analysis/diag.jsonl from tools/analysis/summarize_trial.py."""
from __future__ import annotations

import argparse
import bisect
import json
import sqlite3
import struct
from pathlib import Path

STATE_TOPICS = ("/odom", "/current_map_pose", "/amcl_pose",
                "/odom/unity_plant_state", "/autodrive/roboracer_1/odom",
                "/odom/bridge_request_command")


def distribution(values):
    values = sorted(values)
    if not values:
        return None
    return {key: values[round(q * (len(values) - 1))]
            for key, q in (("p50", .5), ("p99", .99), ("max", 1.))}


def analyze(run: Path):
    records = [json.loads(line) for line in (run / "analysis/diag.jsonl").open()]
    cycles = [record["payload"] for record in records
              if record["payload"].get("state")]
    failures = [record for record in records if record["status"].startswith("rejected")]
    topics = {}
    collision_ns = None
    for database in sorted((run / "run").glob("*.db3")):
        with sqlite3.connect(database) as connection:
            for topic_id, name, kind in connection.execute("select id,name,type from topics"):
                if name in STATE_TOPICS:
                    target = topics.setdefault(name, [])
                    for receipt, data in connection.execute(
                            "select timestamp,data from messages where topic_id=? order by timestamp",
                            (topic_id,)):
                        sec, nsec = struct.unpack_from("<iI", data, 4)
                        target.append((receipt, sec * 1_000_000_000 + nsec))
                if name.endswith("/collision_count") and kind in ("std_msgs/msg/Float32", "std_msgs/msg/Int32"):
                    for receipt, data in connection.execute(
                            "select timestamp,data from messages where topic_id=? order by timestamp",
                            (topic_id,)):
                        if struct.unpack_from("<f" if kind.endswith("Float32") else "<i", data, 4)[0] > 0:
                            collision_ns = min(collision_ns or receipt, receipt)
    for samples in topics.values():
        samples.sort()
    failure_ns = min([round(r["bag_receipt_epoch_s"] * 1.e9) for r in failures]
                     + ([collision_ns] if collision_ns else []) + [10**30])
    stale = []
    for record in records:
        if record["status"] != "state_stale_wait":
            continue
        payload = record["payload"]
        now = payload["control_ros_stamp_ns"]
        states = {}
        for topic, samples in topics.items():
            i = bisect.bisect_right(samples, (now, 10**30)) - 1
            if i < 0:
                continue
            receipt, stamp = samples[i]
            states[topic] = dict(stamp_ns=stamp, receipt_ns=receipt,
                                 receipt_age_ms=(now-receipt)*1.e-6,
                                 source_age_ms=(now-stamp)*1.e-6,
                                 stream_ended=i == len(samples)-1)
        stale.append(dict(**payload, after_abort=now > failure_ns, recorder=states))
    odom = topics.get("/odom", [])
    callback_lag = []
    behind = []
    for cycle in cycles:
        now = cycle["control_ros_stamp_ns"]
        if now >= failure_ns:
            continue
        i = bisect.bisect_right(odom, (now, 10**30)) - 1
        source = cycle["source_stamp_ns"]
        if i >= 0 and odom[i][1] > source:
            behind.append((now-odom[i][0])*1.e-6)
        j = next((j for j in range(max(0, i-5), i+1) if odom[j][1] == source), None)
        if j is not None:
            callback_lag.append((now-odom[j][0])*1.e-6)
    return dict(run=run.name, first_abort_receipt_ns=None if failure_ns == 10**30 else failure_ns,
                state_topics={topic: dict(last_receipt_ns=samples[-1][0],
                    last_stamp_ns=samples[-1][1], receipt_gap_ms=distribution([
                        (b[0]-a[0])*1.e-6 for a,b in zip(samples, samples[1:])]))
                    for topic,samples in topics.items()},
                recorder_to_control_ms=distribution(callback_lag),
                recorder_ahead_count=len(behind),
                recorder_ahead_receipt_age_ms=distribution(behind),
                callback_to_sync_ms=distribution([
                    (c["host_timing_ns"]["synchronize"]-c["host_timing_ns"]["callback"])*1.e-6
                    for c in cycles]),
                control_callback_ms=distribution([c["end_to_end_control_ms"] for c in cycles]),
                stale=stale)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = [analyze(run) for run in args.runs]
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded)
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
