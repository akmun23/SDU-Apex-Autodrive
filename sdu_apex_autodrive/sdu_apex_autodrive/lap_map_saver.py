"""Save a SLAM Toolbox map after repeated laps.

The mapping launch may use simulator ground-truth odometry for this offline
construction step. The topic is configurable and is never part of racing
localization.
"""

from pathlib import Path
import hashlib
import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from slam_toolbox.srv import SaveMap
from std_msgs.msg import Bool, Int32
from tf2_ros import Buffer, TransformException, TransformListener


class FiveLapMapSaver(Node):
    """Save a development map after ground-truth lap coverage."""

    def __init__(self) -> None:
        super().__init__("five_lap_map_saver")
        self._declare_parameters()

        self.lap_count_topic = str(self.get_parameter("lap_count_topic").value)
        self.target_laps = int(self.get_parameter("target_laps").value)
        self.settle_before_save_s = float(
            self.get_parameter("settle_before_save_s").value)
        self.stop_before_save = bool(self.get_parameter("stop_before_save").value)
        self.save_each_lap = bool(self.get_parameter("save_each_lap").value)
        self.lap_snapshot_settle_s = max(
            0.0, float(self.get_parameter("lap_snapshot_settle_s").value))
        self.output_path = Path(str(self.get_parameter("output_directory").value))
        self.map_name = str(self.get_parameter("map_name").value)
        if not self.map_name or Path(self.map_name).name != self.map_name:
            raise ValueError("map_name must be a filename without a directory")
        self.output_path.mkdir(parents=True, exist_ok=True)
        if any(self.output_path.glob(f"{self.map_name}*")):
            raise FileExistsError(
                f"map output already exists for {self.map_name!r}; choose a new map_name")
        self.save_map_service = str(self.get_parameter("save_map_service").value)
        self.provenance_enabled = bool(self.get_parameter("provenance_enabled").value)
        self.provenance_file = str(self.get_parameter("provenance_file").value).strip()
        self.provenance_map_frame = str(
            self.get_parameter("provenance_map_frame").value)
        self.provenance_world_frame = str(
            self.get_parameter("provenance_world_frame").value)
        if self.provenance_enabled and not self.provenance_file:
            self.provenance_file = str(
                self.output_path / f"{self.map_name}.provenance.yaml")
        if self.provenance_enabled and (
                not self.provenance_map_frame or not self.provenance_world_frame):
            raise ValueError("map provenance frame names must not be empty")

        if self.target_laps <= 0:
            raise ValueError("target_laps must be positive")
        if self.settle_before_save_s < 0.0:
            raise ValueError("settle_before_save_s must not be negative")
        self._save_client = self.create_client(SaveMap, self.save_map_service)
        self._tf_buffer = Buffer() if self.provenance_enabled else None
        self._tf_listener = (
            TransformListener(self._tf_buffer, self) if self._tf_buffer is not None else None)
        completion_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._completion_pub = self.create_publisher(
            Bool, str(self.get_parameter("completion_topic").value), completion_qos)
        self._completion_pub.publish(Bool(data=False))
        self._lap_count_sub = self.create_subscription(
            Int32,
            self.lap_count_topic,
            self._on_lap_count,
            rclpy.qos.qos_profile_sensor_data,
        )
        self._initial_sim_lap_count = None
        self._last_sim_lap_count = None
        self._laps = 0
        self._save_requested = False
        self._finished = False
        self._waiting_for_service_logged = False
        self._target_reached_monotonic = None
        self._stop_published = False
        self._snapshot_pending_lap = None
        self._snapshot_ready_monotonic = None
        self._snapshot_in_flight = False
        self._timer = self.create_timer(0.10, self._check_save)

        self.get_logger().info(
            f"Five-lap mapping armed: {self.target_laps} laps -> "
            f"{self.output_path}/{self.map_name}.[yaml|pgm]")

    def _declare_parameters(self) -> None:
        self.declare_parameter("lap_count_topic", "/autodrive/roboracer_1/lap_count")
        self.declare_parameter("save_map_service", "/slam_toolbox/save_map")
        self.declare_parameter("completion_topic", "/sdu/mapping_complete")
        self.declare_parameter("output_directory", "/workspace/src/f1tenth_planning/maps")
        self.declare_parameter("map_name", "autodrive_track_5laps")
        self.declare_parameter("target_laps", 5)
        # Once the simulator reports the target number of complete laps, allow
        # SLAM to publish and serialize the final map.
        self.declare_parameter("settle_before_save_s", 8.0)
        self.declare_parameter("stop_before_save", True)
        self.declare_parameter("save_each_lap", True)
        self.declare_parameter("lap_snapshot_settle_s", 1.0)
        self.declare_parameter("provenance_enabled", False)
        self.declare_parameter("provenance_file", "")
        self.declare_parameter("provenance_map_frame", "map")
        self.declare_parameter("provenance_world_frame", "world")

    def _on_lap_count(self, msg: Int32) -> None:
        current = int(msg.data)
        if self._initial_sim_lap_count is None:
            self._initial_sim_lap_count = current
            self._last_sim_lap_count = current
            self.get_logger().info(
                f"Ground-truth lap counter baseline: {current}")
            return
        if current < self._last_sim_lap_count:
            self.get_logger().error(
                "Simulator lap counter reset during mapping; refusing a partial map")
            self._finished = True
            self._completion_pub.publish(Bool(data=True))
            return
        self._last_sim_lap_count = current
        completed = current - self._initial_sim_lap_count
        while self._laps < min(completed, self.target_laps):
            self._laps += 1
            self.get_logger().info(
                f"Completed simulator mapping lap {self._laps}/{self.target_laps}")
            if self.save_each_lap:
                self._queue_lap_snapshot(self._laps)
            if self._laps >= self.target_laps:
                self._target_reached_monotonic = time.monotonic()
                self.get_logger().info(
                    f"{self.target_laps} mapping laps complete; stopping and "
                    f"waiting {self.settle_before_save_s:.1f}s for map save")

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
            if self.provenance_enabled:
                self._write_provenance(
                    self.output_path / f"{snapshot_name}.provenance.yaml",
                    self.output_path / f"{snapshot_name}.yaml")
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
            if self.provenance_enabled:
                self._write_provenance(
                    Path(self.provenance_file),
                    self.output_path / f"{self.map_name}.yaml")
        else:
            self.get_logger().error(
                "Map save did not succeed; stopping after the requested five laps")
        self._finished = True
        self._completion_pub.publish(Bool(data=True))

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _write_provenance(self, output: Path, map_yaml: Path) -> bool:
        """Save the fixed map->simulator transform alongside a map artifact."""
        if self._tf_buffer is None:
            return False
        try:
            # TF lookup(target, source) returns the transform that maps source
            # coordinates into target coordinates. Thus this is map -> world.
            transform = self._tf_buffer.lookup_transform(
                self.provenance_world_frame,
                self.provenance_map_frame,
                rclpy.time.Time(),
            )
        except TransformException as exc:
            self.get_logger().error(
                f"Cannot write map provenance; missing "
                f"{self.provenance_map_frame}->{self.provenance_world_frame} TF: {exc}")
            return False

        rotation = transform.transform.rotation
        yaw = math.atan2(
            2.0 * (rotation.w * rotation.z + rotation.x * rotation.y),
            1.0 - 2.0 * (rotation.y * rotation.y + rotation.z * rotation.z),
        )
        image_path = map_yaml.with_suffix(".pgm")
        yaml_hash = self._sha256(map_yaml) if map_yaml.is_file() else ""
        image_hash = self._sha256(image_path) if image_path.is_file() else ""
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            "# Fixed transform captured from mapping-only ground-truth TF.\n"
            "# This file is for diagnostics/offline scoring; racing nodes never consume it.\n"
            f"map_yaml: {map_yaml}\n"
            f"map_frame: {self.provenance_map_frame}\n"
            f"world_frame: {self.provenance_world_frame}\n"
            "map_to_world:\n"
            f"  x_m: {transform.transform.translation.x:.12g}\n"
            f"  y_m: {transform.transform.translation.y:.12g}\n"
            f"  yaw_rad: {yaw:.12g}\n"
            "source: simulator_truth_tf_at_save\n"
            f"map_yaml_sha256: {yaml_hash}\n"
            f"map_image_sha256: {image_hash}\n",
            encoding="utf-8",
        )
        self.get_logger().info(
            f"Map provenance saved to {output} "
            f"(map->world yaw={yaw:.6f}, translation="
            f"({transform.transform.translation.x:.3f}, "
            f"{transform.transform.translation.y:.3f}))")
        return True


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
