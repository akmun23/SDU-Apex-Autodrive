"""Save a SLAM Toolbox map after five closed laps using team sensor odometry."""

from pathlib import Path
import math

from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from slam_toolbox.srv import SaveMap
from std_msgs.msg import Bool


class FiveLapMapSaver(Node):
    """Count sensor-odometry laps and request one map save without simulator pose."""

    def __init__(self) -> None:
        super().__init__("five_lap_map_saver")
        self._declare_parameters()

        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.target_laps = int(self.get_parameter("target_laps").value)
        self.start_radius = float(self.get_parameter("start_radius_m").value)
        self.departure_radius = float(self.get_parameter("departure_radius_m").value)
        self.minimum_lap_distance = float(
            self.get_parameter("minimum_lap_distance_m").value)
        self.maximum_odom_step = float(self.get_parameter("maximum_odom_step_m").value)
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
        self._odom_sub = self.create_subscription(
            Odometry, self.odom_topic, self._on_odom, rclpy.qos.qos_profile_sensor_data)

        self._last_odom_xy = None
        self._start_xy = None
        self._distance_m = 0.0
        self._lap_start_distance_m = 0.0
        self._departed_start = False
        self._laps = 0
        self._save_requested = False
        self._finished = False
        self._waiting_for_service_logged = False
        self._timer = self.create_timer(0.10, self._check_save)

        self.get_logger().info(
            f"Five-lap mapping armed: {self.target_laps} laps -> "
            f"{self.output_path}/{self.map_name}.[yaml|pgm]")

    def _declare_parameters(self) -> None:
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("save_map_service", "/slam_toolbox/save_map")
        self.declare_parameter("completion_topic", "/sdu/mapping_complete")
        self.declare_parameter("output_directory", "/workspace/src/f1tenth_planning/maps")
        self.declare_parameter("map_name", "autodrive_track_5laps")
        self.declare_parameter("target_laps", 5)
        self.declare_parameter("start_radius_m", 0.50)
        self.declare_parameter("departure_radius_m", 1.00)
        self.declare_parameter("minimum_lap_distance_m", 8.00)
        self.declare_parameter("maximum_odom_step_m", 1.00)

    def _on_odom(self, msg: Odometry) -> None:
        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)
        if not (math.isfinite(x) and math.isfinite(y)):
            return
        current = (x, y)
        if self._last_odom_xy is not None:
            step = math.dist(current, self._last_odom_xy)
            if step <= self.maximum_odom_step:
                self._distance_m += step
            else:
                self.get_logger().warn(
                    f"Ignoring {step:.2f} m odometry jump while counting laps")
        self._last_odom_xy = current

        if self._finished or self._laps >= self.target_laps:
            return
        if self._start_xy is None:
            self._start_xy = current
            self._lap_start_distance_m = self._distance_m
            self.get_logger().info(
                f"Mapping start fixed at ({current[0]:.2f}, {current[1]:.2f}) "
                "in odom")
            return

        separation = math.dist(current, self._start_xy)
        if not self._departed_start:
            if separation >= self.departure_radius:
                self._departed_start = True
            return

        lap_distance = self._distance_m - self._lap_start_distance_m
        if separation > self.start_radius or lap_distance < self.minimum_lap_distance:
            return

        self._laps += 1
        self._lap_start_distance_m = self._distance_m
        self._departed_start = False
        self.get_logger().info(
            f"Completed mapping lap {self._laps}/{self.target_laps} "
            f"({lap_distance:.1f} m)")
        if self._laps >= self.target_laps:
            self.get_logger().info("Five mapping laps complete; saving map")

    def _check_save(self) -> None:
        if self._finished or self._save_requested or self._laps < self.target_laps:
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
            self.get_logger().info("Five-lap map saved successfully")
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
