"""Save a SLAM Toolbox map after repeated laps.

The mapping launch may use simulator ground-truth odometry for this offline
construction step. The topic is configurable and is never part of racing
localization.
"""

from pathlib import Path
import math
import time

from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from slam_toolbox.srv import SaveMap
from std_msgs.msg import Bool, Int32


class FiveLapMapSaver(Node):
    """Count sensor-odometry laps and request one map save without simulator pose."""

    def __init__(self) -> None:
        super().__init__("five_lap_map_saver")
        self._declare_parameters()

        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.lap_count_topic = str(self.get_parameter("lap_count_topic").value)
        self.require_simulator_lap_count = bool(
            self.get_parameter("require_simulator_lap_count").value)
        self.target_laps = int(self.get_parameter("target_laps").value)
        self.start_radius = float(self.get_parameter("start_radius_m").value)
        self.departure_radius = float(self.get_parameter("departure_radius_m").value)
        self.minimum_lap_distance = float(
            self.get_parameter("minimum_lap_distance_m").value)
        self.start_heading_tolerance = float(
            self.get_parameter("start_heading_tolerance_rad").value)
        self.maximum_odom_step = float(self.get_parameter("maximum_odom_step_m").value)
        self.settle_before_save_s = float(
            self.get_parameter("settle_before_save_s").value)
        self.stop_before_save = bool(self.get_parameter("stop_before_save").value)
        self.save_each_lap = bool(self.get_parameter("save_each_lap").value)
        self.lap_snapshot_settle_s = max(
            0.0, float(self.get_parameter("lap_snapshot_settle_s").value))
        self.reset_at_lap_boundary = bool(
            self.get_parameter("reset_at_lap_boundary").value)
        self.reset_pulse_sec = float(self.get_parameter("reset_pulse_sec").value)
        if self.reset_at_lap_boundary:
            raise ValueError(
                "Simulator resets are disabled for mapping; a collision is a crash")
        self.output_path = Path(str(self.get_parameter("output_directory").value))
        self.map_name = str(self.get_parameter("map_name").value)
        self.save_map_service = str(self.get_parameter("save_map_service").value)

        if self.target_laps <= 0 or min(
            self.start_radius,
            self.departure_radius,
            self.minimum_lap_distance,
            self.maximum_odom_step,
        ) <= 0.0:
            raise ValueError("five-lap mapping parameters must be positive")
        if not (0.0 < self.start_heading_tolerance <= math.pi):
            raise ValueError("start_heading_tolerance_rad must be in (0, pi]")
        if self.settle_before_save_s < 0.0:
            raise ValueError("settle_before_save_s must not be negative")
        if self.reset_pulse_sec <= 0.0:
            raise ValueError("reset_pulse_sec must be positive")
        if self.departure_radius <= self.start_radius:
            raise ValueError("departure_radius_m must exceed start_radius_m")
        if not self.map_name or Path(self.map_name).name != self.map_name:
            raise ValueError("map_name must be a filename without a directory")

        self._save_client = self.create_client(SaveMap, self.save_map_service)
        completion_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._completion_pub = self.create_publisher(
            Bool, str(self.get_parameter("completion_topic").value), completion_qos)
        self._completion_pub.publish(Bool(data=False))
        self._reset_pub = None
        self._odom_sub = self.create_subscription(
            Odometry, self.odom_topic, self._on_odom, rclpy.qos.qos_profile_sensor_data)
        self._lap_count_sub = None
        if self.require_simulator_lap_count:
            self._lap_count_sub = self.create_subscription(
                Int32,
                self.lap_count_topic,
                self._on_lap_count,
                10,
            )
        self._collision_sub = self.create_subscription(
            Int32,
            str(self.get_parameter("collision_topic").value),
            self._on_collision_count,
            10,
        )

        self._last_odom_xy = None
        # Re-anchor after every confirmed crossing. A fixed origin is not a
        # valid repeated-lap gate once wheel/IMU odometry has accumulated
        # drift; the map itself is allowed to correct that drift later.
        self._lap_anchor_xy = None
        self._lap_anchor_heading = None
        self._distance_m = 0.0
        self._lap_start_distance_m = 0.0
        self._departed_start = False
        self._laps = 0
        self._save_requested = False
        self._finished = False
        self._waiting_for_service_logged = False
        self._target_reached_monotonic = None
        self._stop_published = False
        self._reset_release_monotonic = None
        self._snapshot_pending_lap = None
        self._snapshot_ready_monotonic = None
        self._snapshot_in_flight = False
        self._collision_baseline_ready = False
        self._collision_baseline_candidate = None
        self._collision_baseline_candidate_since = None
        self._sim_lap_baseline = None
        self._sim_lap_count = None
        self._timer = self.create_timer(0.10, self._check_save)

        self.get_logger().info(
            f"Five-lap mapping armed: {self.target_laps} laps -> "
            f"{self.output_path}/{self.map_name}.[yaml|pgm]")
        if self.require_simulator_lap_count:
            self.get_logger().info(
                f"Requiring simulator lap-count increment on {self.lap_count_topic}; "
                "odometry return alone cannot complete mapping")

    def _declare_parameters(self) -> None:
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter(
            "lap_count_topic", "/autodrive/roboracer_1/lap_count")
        self.declare_parameter("require_simulator_lap_count", False)
        self.declare_parameter("save_map_service", "/slam_toolbox/save_map")
        self.declare_parameter("completion_topic", "/sdu/mapping_complete")
        self.declare_parameter("output_directory", "/workspace/src/f1tenth_planning/maps")
        self.declare_parameter("map_name", "autodrive_track_5laps")
        self.declare_parameter("target_laps", 5)
        self.declare_parameter("start_radius_m", 0.50)
        self.declare_parameter("departure_radius_m", 1.00)
        self.declare_parameter("minimum_lap_distance_m", 8.00)
        # Position-only returns can accept a U-turn that retraces the outgoing
        # path. A genuine directed lap must return with approximately the same
        # vehicle heading as at departure.
        self.declare_parameter("start_heading_tolerance_rad", 0.75)
        self.declare_parameter("maximum_odom_step_m", 1.00)
        # A return to the odometry start is only the trigger for stopping the
        # car. SLAM Toolbox still needs time to apply the loop constraint,
        # publish the corrected map, and serialize the optimized graph.
        self.declare_parameter("settle_before_save_s", 8.0)
        self.declare_parameter("stop_before_save", True)
        self.declare_parameter("save_each_lap", True)
        self.declare_parameter("lap_snapshot_settle_s", 1.0)
        # Ground-truth mapping may reset the simulator at each confirmed lap.
        # This repeats the known-safe first-lap departure instead of allowing
        # accumulated actuator state to turn the next lap into a collision.
        self.declare_parameter("reset_at_lap_boundary", False)
        self.declare_parameter("reset_topic", "/autodrive/reset_command")
        self.declare_parameter("reset_pulse_sec", 0.50)
        self.declare_parameter(
            "collision_topic", "/autodrive/roboracer_1/collision_count")
        self.declare_parameter("collision_baseline_stable_sec", 1.0)

    def _on_collision_count(self, msg: Int32) -> None:
        """Abort mapping on a new collision; never continue after a crash."""
        count = max(0, int(msg.data))
        now = time.monotonic()
        stable_s = max(
            0.1, float(self.get_parameter("collision_baseline_stable_sec").value))

        # The simulator bridge can publish a startup zero before its first
        # actual cumulative collision sample. Establish a stable baseline so
        # an old count is not mistaken for a collision in this run.
        if not self._collision_baseline_ready:
            if self._collision_baseline_candidate != count:
                self._collision_baseline_candidate = count
                self._collision_baseline_candidate_since = now
                return
            if (self._collision_baseline_candidate_since is not None and
                    now - self._collision_baseline_candidate_since >= stable_s):
                self._collision_baseline_ready = True
                self.get_logger().info(
                    f"Mapping collision baseline established at cumulative count {count}")
            return

        if count < int(self._collision_baseline_candidate):
            self._collision_baseline_ready = False
            self._collision_baseline_candidate = count
            self._collision_baseline_candidate_since = now
            return
        if count <= int(self._collision_baseline_candidate):
            return

        self._finished = True
        self._completion_pub.publish(Bool(data=True))
        self.get_logger().fatal(
            f"Collision count increased to {count}; aborting mapping run "
            "without reset or further lap capture")
        raise SystemExit(2)

    def _on_lap_count(self, msg: Int32) -> None:
        """Use the simulator's mapping-only lap counter as a completion guard."""
        count = int(msg.data)
        if self._sim_lap_baseline is None:
            self._sim_lap_baseline = count
            self._sim_lap_count = count
            self.get_logger().info(
                f"Simulator lap-count baseline established at {count}")
            return
        if count < self._sim_lap_baseline:
            self.get_logger().warn(
                f"Simulator lap count moved backwards from baseline "
                f"{self._sim_lap_baseline} to {count}; re-arming mapping guard")
            self._sim_lap_baseline = count
        self._sim_lap_count = count

    def _on_odom(self, msg: Odometry) -> None:
        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)
        orientation = msg.pose.pose.orientation
        heading = math.atan2(
            2.0 * (orientation.w * orientation.z + orientation.x * orientation.y),
            1.0 - 2.0 * (orientation.y * orientation.y + orientation.z * orientation.z),
        )
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(heading)):
            return
        current = (x, y)
        if self._last_odom_xy is not None:
            step = math.dist(current, self._last_odom_xy)
            if step <= self.maximum_odom_step:
                self._distance_m += step
            else:
                self.get_logger().warn(
                    f"Ignoring {step:.2f} m odometry/reset jump; re-arming lap gate")
                # A mapping-only simulator reset can occur after a collision
                # at any point on the track. Do not let the post-reset pose
                # inherit the pre-reset departure state or count a false lap.
                self._lap_anchor_xy = current
                self._lap_anchor_heading = heading
                self._lap_start_distance_m = self._distance_m
                self._departed_start = False
        self._last_odom_xy = current

        if self._finished or self._laps >= self.target_laps:
            return
        if self._lap_anchor_xy is None:
            self._lap_anchor_xy = current
            self._lap_anchor_heading = heading
            self._lap_start_distance_m = self._distance_m
            self.get_logger().info(
                f"Mapping start fixed at ({current[0]:.2f}, {current[1]:.2f}) "
                "in odom")
            return

        separation = math.dist(current, self._lap_anchor_xy)
        if not self._departed_start:
            if separation >= self.departure_radius:
                self._departed_start = True
            return

        lap_distance = self._distance_m - self._lap_start_distance_m
        if separation > self.start_radius or lap_distance < self.minimum_lap_distance:
            return

        heading_error = abs(math.atan2(
            math.sin(heading - float(self._lap_anchor_heading)),
            math.cos(heading - float(self._lap_anchor_heading)),
        ))
        if heading_error > self.start_heading_tolerance:
            self.get_logger().warn(
                f"Rejecting odometry return as reverse/retraced path: "
                f"distance={lap_distance:.1f} m, position error="
                f"{separation:.2f} m, heading error={heading_error:.2f} rad "
                f"(limit {self.start_heading_tolerance:.2f})")
            return

        if self.require_simulator_lap_count:
            simulator_laps = 0
            if (self._sim_lap_baseline is not None and
                    self._sim_lap_count is not None):
                simulator_laps = self._sim_lap_count - self._sim_lap_baseline
            if simulator_laps < self._laps + 1:
                self.get_logger().warn(
                    f"Rejecting odometry return as incomplete lap: "
                    f"distance={lap_distance:.1f} m, simulator lap delta="
                    f"{simulator_laps}, required={self._laps + 1}")
                return

        self._laps += 1
        self._lap_start_distance_m = self._distance_m
        # Use this confirmed crossing as the next lap's odometry anchor. The
        # anchor may move slowly in odom as drift accumulates, while the
        # minimum-distance gate still prevents short corner loops.
        self._lap_anchor_xy = current
        self._lap_anchor_heading = heading
        self._departed_start = False
        self.get_logger().info(
            f"Completed mapping lap {self._laps}/{self.target_laps} "
            f"({lap_distance:.1f} m)")
        if self.save_each_lap:
            self._queue_lap_snapshot(self._laps)
        if self._laps >= self.target_laps:
            self._target_reached_monotonic = time.monotonic()
            self.get_logger().info(
                f"{self.target_laps} mapping laps complete; stopping and "
                f"waiting {self.settle_before_save_s:.1f}s for loop closure")
        # No simulator reset is permitted between laps. Continue from the
        # actual vehicle pose; a collision is handled by the terminal abort
        # callback above.

    def _request_lap_reset(self) -> None:
        if self._reset_pub is None or self._reset_release_monotonic is not None:
            return
        self._reset_pub.publish(Bool(data=True))
        self._reset_release_monotonic = time.monotonic() + self.reset_pulse_sec
        # The simulator reset returns to the same physical track start. Re-arm
        # the departure gate so the reset itself cannot count as another lap.
        self._last_odom_xy = None
        self._departed_start = False
        self.get_logger().info(
            f"Lap {self._laps} complete; simulator reset pulse requested "
            f"for {self.reset_pulse_sec:.2f}s before the next mapping lap")

    def _service_lap_reset(self) -> None:
        if self._reset_release_monotonic is None:
            return
        if time.monotonic() < self._reset_release_monotonic:
            self._reset_pub.publish(Bool(data=True))
            return
        self._reset_pub.publish(Bool(data=False))
        self._reset_release_monotonic = None
        self.get_logger().info("Simulator reset pulse released; next lap armed")

    def _queue_lap_snapshot(self, lap_number: int) -> None:
        if self._snapshot_pending_lap is not None or self._snapshot_in_flight:
            self.get_logger().warn(
                f"Lap snapshot still active; not replacing pending lap {lap_number}")
            return
        self._snapshot_pending_lap = lap_number
        self._snapshot_ready_monotonic = (
            time.monotonic() + self.lap_snapshot_settle_s)

    def _service_lap_snapshot(self) -> None:
        if self._snapshot_pending_lap is None or self._snapshot_in_flight:
            return
        if (self._snapshot_ready_monotonic is not None and
                time.monotonic() < self._snapshot_ready_monotonic):
            return
        if not self._save_client.service_is_ready():
            return

        lap_number = self._snapshot_pending_lap
        self._snapshot_pending_lap = None
        self._snapshot_ready_monotonic = None
        request = SaveMap.Request()
        snapshot_name = f"{self.map_name}_lap{lap_number:02d}"
        request.name.data = str(self.output_path / snapshot_name)
        self._snapshot_in_flight = True
        future = self._save_client.call_async(request)
        future.add_done_callback(
            lambda result, lap=lap_number, name=snapshot_name:
            self._on_lap_snapshot_complete(result, lap, name))
        self.get_logger().info(
            f"Saving lap {lap_number} snapshot to "
            f"{request.name.data}.[yaml|pgm]")

    def _on_lap_snapshot_complete(self, future, lap_number: int, snapshot_name: str) -> None:
        try:
            response = future.result()
            success = response is not None and response.result == SaveMap.Response.RESULT_SUCCESS
        except Exception as exc:
            self.get_logger().error(
                f"Lap {lap_number} map snapshot failed: {exc}")
            success = False
        self._snapshot_in_flight = False
        if success:
            self.get_logger().info(
                f"Lap {lap_number} map snapshot saved: {snapshot_name}")
        else:
            self.get_logger().error(
                f"Lap {lap_number} map snapshot did not succeed")

    def _check_save(self) -> None:
        self._service_lap_snapshot()
        if self._finished or self._save_requested or self._laps < self.target_laps:
            return
        if self._snapshot_pending_lap is not None or self._snapshot_in_flight:
            return

        if self.stop_before_save and not self._stop_published:
            # This is published only after this process has observed the
            # target lap. The actuator subscription is volatile, so an old
            # completion event cannot stop a fresh run.
            self._completion_pub.publish(Bool(data=True))
            self._stop_published = True
            self.get_logger().info("Mapping target reached; actuator stop requested")

        if self._target_reached_monotonic is None:
            self._target_reached_monotonic = time.monotonic()
        settled_for = time.monotonic() - self._target_reached_monotonic
        if settled_for < self.settle_before_save_s:
            return
        self._request_save()

    def _request_save(self) -> None:
        self.output_path.mkdir(parents=True, exist_ok=True)
        if not self._save_client.service_is_ready():
            if not self._waiting_for_service_logged:
                self.get_logger().warn(
                    "Five laps complete; waiting for "
                    f"{self.save_map_service} before saving map")
                self._waiting_for_service_logged = True
            return

        request = SaveMap.Request()
        request.name.data = str(self.output_path / self.map_name)
        self._save_requested = True
        future = self._save_client.call_async(request)
        future.add_done_callback(self._on_save_complete)
        self.get_logger().info(f"Saving map to {request.name.data}.[yaml|pgm]")

    def _on_save_complete(self, future) -> None:
        try:
            response = future.result()
            success = response is not None and response.result == SaveMap.Response.RESULT_SUCCESS
        except Exception as exc:  # rclpy surfaces service failures through Future.result().
            self.get_logger().error(f"SLAM Toolbox map save failed: {exc}")
            success = False

        if success:
            self.get_logger().info(
                f"{self.target_laps}-lap map saved successfully after "
                "loop-closure settling")
        else:
            self.get_logger().error(
                "Map save did not succeed; stopping after the requested five laps")
        self._finished = True
        self._completion_pub.publish(Bool(data=True))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FiveLapMapSaver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
