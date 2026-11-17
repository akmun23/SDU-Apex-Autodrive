#!/usr/bin/env python3
"""Export recorded sensor epochs and Unity truth for odometry replay.

Usage (full recorder profile or cached read_bag() JSON):
    source /opt/ros/jazzy/setup.bash
    python3 f1tenth_localization/util/export_odometry_replay.py live_runs/RUN OUT.csv
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import evaluate_mpc_bag as bag  # noqa: E402

COLUMNS = ("stamp_ns left right ax ay yaw_rate yaw roll pitch roll_rate pitch_rate "
           "steering_command throttle_command steering_feedback_rad throttle_feedback "
           "truth_u truth_v_com logged_u").split()


def microseconds(message):
    return round(message["stamp"] * 1e6)


def export(run, out):
    data = (json.loads(Path(run).read_text()) if str(run).endswith(".json")
            else bag.read_bag(Path(run) / "run"))
    by_stamp = {topic: {microseconds(m): m for m in rows if "stamp" in m}
                for topic, rows in data.items()}
    plant = data[bag.PLANT]
    with open(out, "w") as f:
        f.write(",".join(COLUMNS) + "\n")
        for p in plant:
            key = microseconds(p)
            try:
                left = by_stamp[bag.LEFT_ENCODER][key]
                right = by_stamp[bag.RIGHT_ENCODER][key]
                imu = by_stamp[bag.IMU][key]
                truth = by_stamp[bag.GT][key]
            except KeyError:
                continue
            row = [p["stamp_ns"], left["angle"], right["angle"], imu["ax"], imu["ay"], imu["yaw_rate"],
                   imu["orientation_yaw"], p["roll"], p["pitch"], p["roll_rate"], p["pitch_rate"],
                   p["steering_command"], p["throttle_command"], p["steering_feedback"],
                   p["throttle_feedback"], truth["u"], truth["v"], p["u"]]
            f.write(",".join(repr(float(x)) if i else str(int(x)) for i, x in enumerate(row)) + "\n")


if __name__ == "__main__":
    export(sys.argv[1], sys.argv[2])
