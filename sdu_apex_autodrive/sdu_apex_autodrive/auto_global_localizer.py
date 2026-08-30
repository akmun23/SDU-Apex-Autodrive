"""Bootstrap Nav2 AMCL globally and report covariance-qualified readiness."""

import math

from geometry_msgs.msg import PoseWithCovarianceStamped
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from rclpy.time import Time
from std_msgs.msg import Bool
from std_srvs.srv import Empty
from tf2_ros import Buffer, TransformListener


class AutoGlobalLocalizer(Node):
    """Call AMCL global initialization, request scans, verify pose and TF."""

    def __init__(self) -> None:
        super().__init__('auto_global_localizer')
        self.declare_parameter(
            'global_localization_service', '/reinitialize_global_localization')
        self.declare_parameter('nomotion_update_service', '/request_nomotion_update')
        self.declare_parameter('pose_topic', '/amcl_pose')
        self.declare_parameter('ready_topic', '/localization/ready')
        self.declare_parameter('global_frame', 'map')
        self.declare_parameter('base_frame', 'roboracer_1')
        self.declare_parameter('max_position_variance_m2', 0.25)
        self.declare_parameter('max_yaw_variance_rad2', 0.12)
        self.declare_parameter('required_consecutive_updates', 5)
        self.declare_parameter('nomotion_request_period_sec', 0.5)
        self.declare_parameter('retry_timeout_sec', 15.0)

        self._global_frame = str(self.get_parameter('global_frame').value)
        self._base_frame = str(self.get_parameter('base_frame').value)
        self._max_xy = float(
            self.get_parameter('max_position_variance_m2').value)
        self._max_yaw = float(
            self.get_parameter('max_yaw_variance_rad2').value)
        self._required = max(
            1, int(self.get_parameter('required_consecutive_updates').value))
        self._nomotion_period = max(
            0.1, float(self.get_parameter('nomotion_request_period_sec').value))
        self._retry_timeout = max(
            1.0, float(self.get_parameter('retry_timeout_sec').value))

        self._global_client = self.create_client(
            Empty,
            str(self.get_parameter('global_localization_service').value),
        )
        self._nomotion_client = self.create_client(
            Empty,
            str(self.get_parameter('nomotion_update_service').value),
        )
        status_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self._ready_pub = self.create_publisher(
            Bool, str(self.get_parameter('ready_topic').value), status_qos)
        self._pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            str(self.get_parameter('pose_topic').value),
            self._on_pose,
            status_qos,
        )
        self._tf_buffer = Buffer(cache_time=Duration(seconds=5.0))
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._state = 'WAIT_FOR_SERVICE'
        self._qualified_updates = 0
        self._pose_qualified = False
        self._received_pose = False
        self._last_variance_xy = math.inf
        self._last_variance_yaw = math.inf
        self._started = self.get_clock().now()
        self._last_nomotion_request = self._started
        self._global_future = None
        self._publish_ready(False)
        self._timer = self.create_timer(0.2, self._tick)
        self.get_logger().info('Waiting for AMCL global-localization service')

    def _publish_ready(self, ready: bool) -> None:
        self._ready_pub.publish(Bool(data=ready))

    def _call_global_localization(self) -> None:
        self._qualified_updates = 0
        self._pose_qualified = False
        self._received_pose = False
        self._publish_ready(False)
        self._global_future = self._global_client.call_async(Empty.Request())
        self._started = self.get_clock().now()
        self._state = 'CALL_GLOBAL_LOCALIZATION'
        self.get_logger().info('Requested AMCL global localization')

    def _on_pose(self, message: PoseWithCovarianceStamped) -> None:
        if message.header.frame_id and message.header.frame_id != self._global_frame:
            self._qualified_updates = 0
            self._pose_qualified = False
            return
        variance_xy = max(message.pose.covariance[0], message.pose.covariance[7])
        variance_yaw = message.pose.covariance[35]
        self._received_pose = True
        self._last_variance_xy = variance_xy
        self._last_variance_yaw = variance_yaw
        qualified = (
            math.isfinite(variance_xy)
            and math.isfinite(variance_yaw)
            and 0.0 <= variance_xy <= self._max_xy
            and 0.0 <= variance_yaw <= self._max_yaw
        )
        self._qualified_updates = self._qualified_updates + 1 if qualified else 0
        self._pose_qualified = self._qualified_updates >= self._required

    def _tf_ready(self) -> bool:
        return self._tf_buffer.can_transform(
            self._global_frame,
            self._base_frame,
            Time(),
            timeout=Duration(seconds=0.05),
        )

    def _tick(self) -> None:
        now = self.get_clock().now()
        if self._state == 'WAIT_FOR_SERVICE':
            if self._global_client.service_is_ready():
                self._call_global_localization()
            return

        if self._state == 'CALL_GLOBAL_LOCALIZATION':
            if self._global_future is not None and self._global_future.done():
                try:
                    self._global_future.result()
                except Exception as error:  # rclpy service exceptions vary
                    self.get_logger().error(
                        'Global localization call failed: %s' % error)
                    self._state = 'WAIT_FOR_SERVICE'
                    return
                self._state = 'MONITOR_AMCL_POSE'
            return

        since_nomotion = (now - self._last_nomotion_request).nanoseconds / 1e9
        if since_nomotion >= self._nomotion_period:
            if self._nomotion_client.service_is_ready():
                self._nomotion_client.call_async(Empty.Request())
            self._last_nomotion_request = now

        # Keep forcing scan updates after readiness. Otherwise a stationary
        # vehicle can receive one qualified AMCL pose, then the controller's
        # pose watchdog expires before it has moved far enough to trigger a
        # normal motion-based AMCL update.
        if self._state == 'READY':
            if not self._pose_qualified or not self._tf_ready():
                self._state = 'MONITOR_AMCL_POSE'
                self._started = now
                self._publish_ready(False)
                self.get_logger().warn(
                    'Localization quality lost; controller gate closed')
                return
            self._publish_ready(True)
            return

        if self._pose_qualified and self._tf_ready():
            self._state = 'READY'
            self._publish_ready(True)
            self.get_logger().info(
                'Localization ready: covariance qualified and TF available')
            return

        elapsed = (now - self._started).nanoseconds / 1e9
        if elapsed >= self._retry_timeout:
            if self._received_pose:
                # Reinitializing here throws away a particle cloud that is
                # still converging under forced scan updates. Keep it and emit
                # an explicit diagnostic instead.
                self.get_logger().warn(
                    'Localization still converging: xy_var=%.3f yaw_var=%.3f' % (
                        self._last_variance_xy,
                        self._last_variance_yaw,
                    ))
                self._started = now
            elif self._global_client.service_is_ready():
                self.get_logger().warn(
                    'No AMCL pose after %.1fs; retrying globally' % elapsed)
                self._call_global_localization()
            else:
                self._state = 'WAIT_FOR_SERVICE'


def main(args=None) -> None:
    rclpy.init(args=args)
    node = AutoGlobalLocalizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
