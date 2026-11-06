"""Supervise a mapping session and save its sensor-derived occupancy grid."""

import json
import math
import signal
import subprocess
import time
from pathlib import Path

import numpy as np
import rclpy
import yaml
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import Float64MultiArray, Int32


class MappingSupervisor(Node):
    def __init__(self):
        super().__init__("mapping_supervisor")
        self.output = Path(self.declare_parameter("output_directory", "/mapping").value)
        self.map_prefix = Path(self.declare_parameter(
            "map_prefix", "/workspace/src/f1tenth_planning/maps/autodrive_iros_2026_compete").value)
        self.target_laps = int(self.declare_parameter("target_laps", 1).value)
        self.timeout = float(self.declare_parameter("timeout_s", 900.0).value)
        self.output.mkdir(parents=True, exist_ok=True)
        self.started = time.monotonic()
        self.last_progress = self.started
        self.distance = 0.0
        self.last_xy = None
        self.speed = 0.0
        self.collisions = None
        self.laps = None
        self.initial_laps = None
        self.grid = None
        self.diagnostic = None
        self.trace = []
        self.recorder = subprocess.Popen([
            "ros2", "bag", "record", "-o", str(self.output / "bag"),
            "/autodrive/roboracer_1/lidar", "/autodrive/roboracer_1/imu",
            "/autodrive/roboracer_1/left_encoder", "/autodrive/roboracer_1/right_encoder",
            "/autodrive/roboracer_1/steering", "/autodrive/roboracer_1/throttle",
            "/autodrive/roboracer_1/steering_command", "/autodrive/roboracer_1/throttle_command",
            "/autodrive/roboracer_1/collision_count", "/autodrive/roboracer_1/lap_count",
            "/odom", "/odom/unity_plant_state", "/mpc/drive_command",
            "/map", "/tf", "/tf_static", "/ftg/diagnostics"],
            stdout=(self.output / "recorder.log").open("w"), stderr=subprocess.STDOUT)
        self.create_subscription(OccupancyGrid, "/map", self.on_map,
                                 QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_subscription(Odometry, "/odom", self.on_odom, 10)
        self.create_subscription(Int32, "/autodrive/roboracer_1/collision_count",
                                 lambda msg: setattr(self, "collisions", msg.data), 10)
        self.create_subscription(Int32, "/autodrive/roboracer_1/lap_count", self.on_lap, 10)
        self.create_subscription(Float64MultiArray, "/ftg/diagnostics",
                                 lambda msg: setattr(self, "diagnostic", list(msg.data)), 1)
        self.create_timer(5.0, self.status)

    def on_map(self, msg):
        self.grid = msg

    def on_lap(self, msg):
        self.laps = msg.data
        if self.initial_laps is None:
            self.initial_laps = msg.data

    def on_odom(self, msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        xy = (p.x, p.y)
        if self.last_xy is not None:
            step = math.hypot(xy[0] - self.last_xy[0], xy[1] - self.last_xy[1])
            if 0.0 < step < 0.5:
                self.distance += step
                if step > 0.0001:
                    self.last_progress = time.monotonic()
        self.last_xy = xy
        self.speed = msg.twist.twist.linear.x
        yaw = math.atan2(2 * (q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))
        self.trace.append((msg.header.stamp.sec + msg.header.stamp.nanosec*1e-9,
                           p.x, p.y, yaw, self.speed))

    def status(self):
        state = self.report("running")
        (self.output / "status.json").write_text(json.dumps(state, indent=2) + "\n")
        print("MAPPING " + json.dumps(state), flush=True)

    def report(self, result):
        return dict(result=result, elapsed_s=round(time.monotonic()-self.started, 2),
                    distance_m=round(self.distance, 3), speed_mps=round(self.speed, 3),
                    collisions=self.collisions, lap_count=self.laps,
                    initial_lap_count=self.initial_laps,
                    map_size=None if self.grid is None else [self.grid.info.width, self.grid.info.height],
                    ftg=self.diagnostic)

    def result(self):
        elapsed = time.monotonic()-self.started
        if self.collisions is not None and self.collisions > 0:
            return "collision", 3
        if elapsed > self.timeout:
            return "timeout", 2
        if elapsed > 20 and time.monotonic()-self.last_progress > 15:
            return "stalled", 4
        if (self.grid is not None and self.initial_laps is not None and
                self.laps >= self.initial_laps+self.target_laps and
                self.distance >= 15*self.target_laps):
            return "completed_laps", 0
        return None

    def save_grid(self, prefix):
        grid = self.grid
        if grid is None:
            return
        prefix.parent.mkdir(parents=True, exist_ok=True)
        cells = np.array(grid.data, dtype=np.int16).reshape(grid.info.height, grid.info.width)
        pixels = np.full(cells.shape, 205, dtype=np.uint8)
        pixels[(cells >= 0) & (cells <= 25)] = 254
        pixels[cells >= 65] = 0
        prefix.with_suffix(".pgm").write_bytes(
            f"P5\n{grid.info.width} {grid.info.height}\n255\n".encode() + np.flipud(pixels).tobytes())
        q = grid.info.origin.orientation
        yaw = math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
        prefix.with_suffix(".yaml").write_text(yaml.safe_dump(dict(
            image=prefix.with_suffix(".pgm").name, mode="trinary", resolution=grid.info.resolution,
            origin=[grid.info.origin.position.x, grid.info.origin.position.y, yaw],
            negate=0, occupied_thresh=0.65, free_thresh=0.25), sort_keys=False))

    def finish(self, reason, code):
        # Host runner watches this marker and stops Unity before finalizing
        # the recording, so collision handling does not leave commands active.
        report = self.report(reason)
        if code == 0:
            report["map_yaml"] = str(self.map_prefix.with_suffix(".yaml"))
        (self.output / "report.json").write_text(json.dumps(report, indent=2)+"\n")
        if self.recorder.poll() is None:
            self.recorder.send_signal(signal.SIGINT)
            try:
                self.recorder.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.recorder.terminate()
                self.recorder.wait(timeout=3)
        self.save_grid(self.map_prefix if code == 0 else self.output / "partial_map")
        if self.trace:
            np.savetxt(self.output / "sensor_odom_trace.csv", self.trace,
                       delimiter=",", header="stamp,x,y,yaw,u")
        print("MAPPING_FINISHED " + json.dumps(report), flush=True)


def main():
    rclpy.init()
    node = MappingSupervisor()
    reason, code = "interrupted", 130
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.02)
            result = node.result()
            if result is not None:
                reason, code = result
                break
    except KeyboardInterrupt:
        pass
    finally:
        node.finish(reason, code)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return code
