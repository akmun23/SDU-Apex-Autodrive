"""One-shot ROS input-readiness gate for manually connected Unity sessions.

The bridge and sensor odometry can start before the GUI's Connect button is
pressed. AMCL starts only after fresh IMU/odometry samples are flowing; MPC
starts only after AMCL publishes fresh map poses. No simulator truth or reset
interfaces are accessed, and no diagnostics or persistent logs are produced.
"""

import math
import sys
import time

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu

MAX_AGE_S = 0.12
MAX_GAP_S = 0.18
REQUIRED_CONSECUTIVE = 4
TOPICS = {
    "imu": (Imu, "/autodrive/roboracer_1/imu"),
    "odom": (Odometry, "/odom"),
    "pose": (PoseWithCovarianceStamped, "/current_map_pose"),
}


def _valid_pose(msg):
    p = msg.pose.pose
    q = p.orientation
    values = (p.position.x, p.position.y, q.x, q.y, q.z, q.w)
    return (all(math.isfinite(value) for value in values)
            and 0.9 < sum(value * value for value in values[2:]) < 1.1)


def wait_until_ready(phase):
    if phase not in ("sensors", "pose"):
        raise ValueError("expected sensors or pose readiness phase")
    required = ("imu", "odom") if phase == "sensors" else ("imu", "pose")
    rclpy.init(args=[])
    node = rclpy.create_node("competition_ready_" + phase, enable_rosout=False)
    state = {name: {"count": 0, "stamp_ns": None, "receive_at": 0.0}
             for name in required}
    callbacks = []
    subscriptions = []

    def callback(msg, name):
        stamp = msg.header.stamp
        ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
        source_age = time.time() - ns * 1e-9
        item = state[name]
        if (ns <= 0 or not -0.025 <= source_age <= MAX_AGE_S
                or (name == "pose" and not _valid_pose(msg))):
            item["count"] = 0
            return
        last_ns = item["stamp_ns"]
        period = (ns - last_ns) * 1e-9 if last_ns is not None else 0.0
        item["count"] = (item["count"] + 1
                         if 0.0 < period <= MAX_GAP_S else 1)
        item["stamp_ns"] = ns
        item["receive_at"] = time.monotonic()

    try:
        for name in required:
            msg_type, topic = TOPICS[name]
            def on_message(msg, name=name):
                callback(msg, name)
            callbacks.append(on_message)
            subscriptions.append(node.create_subscription(
                msg_type, topic, on_message, qos_profile_sensor_data))
        # No deadline: the operator may press Connect at any point.
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            if not all(state[name]["count"] >= REQUIRED_CONSECUTIVE
                       and time.monotonic() - state[name]["receive_at"] < MAX_AGE_S
                       for name in required):
                continue
            if abs(state[required[0]]["stamp_ns"]
                   - state[required[1]]["stamp_ns"]) * 1e-9 <= 0.06:
                return
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main():
    if len(sys.argv) != 2:
        raise SystemExit(2)
    wait_until_ready(sys.argv[1])


if __name__ == "__main__":
    main()
