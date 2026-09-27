#!/usr/bin/env python3
"""Fit the IMU acceleration lever arm from the transient open-plane bag.

Uses COM velocity/yaw derivatives and IMU lateral acceleration. Repetitions
1-2 fit lever arm and constant bias; repetition 3 is held out.
"""
from __future__ import annotations

import argparse
import bisect
import math
import sqlite3
import statistics
from pathlib import Path

import evaluate_open_plane_transient_sweep as sweep
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

COM_X_M = sweep.common.COM_X_M
IMU_FRAME_X_M = 0.08
HALF_WINDOW = 4


def _decode(db, topic):
    topic_id, msg_type = topic
    cls = get_message(msg_type)
    return [(int(stamp), deserialize_message(bytes(blob), cls))
            for stamp, blob in db.execute(
                "SELECT timestamp,data FROM messages WHERE topic_id=? "
                "ORDER BY timestamp,id", (topic_id,))]


def _samples(path, phases):
    db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        topics = sweep.common._topic_map(db)
        odom = _decode(db, topics[sweep.ODOM])
        imu = _decode(db, topics[sweep.common.IMU])
    finally:
        db.close()
    odom_times = [row[0] for row in odom]
    imu_times = [row[0] for row in imu]
    windows = []
    for phase in phases:
        match = sweep.PHASE_RE.match(phase.label)
        if match and phase.valid is True:
            windows.append((phase.start_ns, phase.end_ns,
                            int(match.group("rep"))))
    rows = []
    for i in range(HALF_WINDOW, len(odom) - HALF_WINDOW):
        stamp, msg = odom[i]
        rep = next((r for start, end, r in windows
                    if start + 250_000_000 <= stamp <= end - 250_000_000), None)
        if rep is None:
            continue
        j = bisect.bisect_left(imu_times, stamp)
        choices = [k for k in (j - 1, j) if 0 <= k < len(imu)]
        if not choices:
            continue
        j = min(choices, key=lambda k: abs(imu_times[k] - stamp))
        if abs(imu_times[j] - stamp) > 20_000_000:
            continue
        t0, before = odom[i - HALF_WINDOW]
        t1, after = odom[i + HALF_WINDOW]
        dt = (t1 - t0) / 1e9
        if not 0.12 <= dt <= 0.30:
            continue
        b, a, c = before.twist.twist, after.twist.twist, msg.twist.twist
        ay_imu = float(imu[j][1].linear_acceleration.y)
        values = (float(b.linear.y), float(a.linear.y), float(c.linear.x),
                  float(c.angular.z), float(b.angular.z), float(a.angular.z),
                  ay_imu)
        if not all(math.isfinite(value) for value in values):
            continue
        vy0, vy1, vx, yaw, yaw0, yaw1, ay_imu = values
        ay_com = (vy1 - vy0) / dt + yaw * vx
        alpha = (yaw1 - yaw0) / dt
        rows.append((rep, alpha, ay_com, ay_imu))
    return rows


def _fit(rows):
    alpha = [row[1] for row in rows]
    residual = [row[3] - row[2] for row in rows]
    ma, mr = statistics.mean(alpha), statistics.mean(residual)
    den = sum((v - ma) ** 2 for v in alpha)
    if len(rows) < 100 or den <= 1e-6:
        raise ValueError("insufficient yaw-acceleration excitation")
    slope = sum((a - ma) * (r - mr) for a, r in zip(alpha, residual)) / den
    bias = mr - slope * ma
    return COM_X_M + slope, bias


def _rmse(rows, x, bias):
    return math.sqrt(statistics.mean(
        (ay + alpha * (x - COM_X_M) + bias - observed) ** 2
        for _, alpha, ay, observed in rows))


def evaluate(path: Path):
    _, quality, phases = sweep.load(path)
    per_rep = 2 * (2 * len(sweep.TRANSITION_LEVELS) - 1)
    if (quality["valid_steering_phases"] != 3 * per_rep
            or quality["matched_sweep_starts"] != 6
            or quality["collision_initial"] != 0
            or quality["collision_final"] != 0
            or quality["bridge_timing_faults"] != 0 or quality["aborted"]):
        raise ValueError(f"capture-quality gate failed: {quality}")
    rows = _samples(path, phases)
    train = [row for row in rows if row[0] in (1, 2)]
    test = [row for row in rows if row[0] == 3]
    xfit, bias = _fit(train)
    candidates = (0.0, IMU_FRAME_X_M, COM_X_M)
    scores = {x: _rmse(test, x, bias) for x in candidates}
    score_fit = _rmse(test, xfit, bias)
    if (abs(xfit - COM_X_M) <= 0.03
            and scores[COM_X_M] <= 0.90 * scores[IMU_FRAME_X_M]
            and score_fit <= 1.10 * scores[COM_X_M]):
        decision = "PASS; acceleration payload is COM-referenced despite IMU TF x=0.08 m"
    elif (abs(xfit - IMU_FRAME_X_M) <= 0.03
          and scores[IMU_FRAME_X_M] <= 0.90 * scores[COM_X_M]
          and score_fit <= 1.10 * scores[IMU_FRAME_X_M]):
        decision = "PASS; acceleration payload is IMU-frame-referenced"
    else:
        decision = "INCONCLUSIVE; do not change observer lever arm"
    print(f"bag: {path}; train/holdout={len(train)}/{len(test)}")
    print(f"fitted x={xfit:.4f} m; lateral bias={bias:+.4f} m/s^2")
    print("holdout RMSE: " + ", ".join(
        f"x={x:.5f} m -> {scores[x]:.4f} m/s^2" for x in candidates)
        + f"; fitted x -> {score_fit:.4f}")
    print("decision: " + decision)
    return decision.startswith("PASS")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path)
    args = parser.parse_args()
    return 0 if evaluate(args.bag) else 1


if __name__ == "__main__":
    raise SystemExit(main())
