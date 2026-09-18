"""Run the official AutoDRIVE bridge with competition-compatible pacing.

The competition image exposes no Unity source clock or response sequence.
Accordingly, this adapter uses the unrequested connect-time packet only as a
handshake, then keeps a small bounded window of Bridge requests and associates
responses by FIFO order. ROS headers use local receive time; host monotonic time
measures transport intervals and response deadlines. No sensor sample is
fabricated, interpolated, or repeated.

Packet callbacks are serialized so response association and message stamping
remain consistent while the official decoder publishes each packet.
"""

from __future__ import annotations

import base64
import gzip
import json
import math
import os
import time
from collections import deque
from copy import copy
from typing import Any

import gevent
from gevent.event import Event
from gevent.lock import Semaphore
from std_msgs.msg import Bool
from std_msgs.msg import String

from autodrive_roboracer import autodrive_bridge as official_bridge


EXPECTED_RATE_HZ = 40.0
DEFAULT_RATE_HZ = EXPECTED_RATE_HZ
PACKET_TIMING_TOPIC = "/autodrive/roboracer_1/bridge_packet_timing"
TIMING_FAULT_TOPIC = "/autodrive/roboracer_1/bridge_timing_fault"
TIMING_FAULT_DETAIL_TOPIC = "/autodrive/roboracer_1/bridge_timing_fault_detail"

# Requests are paced on a 25 ms host-clock grid. Actual response/arrival times
# are retained; a delayed packet is never relabeled as a nominal 25 ms sample.
MAX_BRIDGE_ARRIVAL_INTERVAL_S = 0.150
MAX_RESPONSE_WAIT_S = MAX_BRIDGE_ARRIVAL_INTERVAL_S
MAX_REQUEST_SCHEDULE_GAP_S = MAX_BRIDGE_ARRIVAL_INTERVAL_S
MIN_REQUEST_SPACING_S = 0.024
# The stock protocol has no response ID. Socket.IO preserves event order, so
# responses can be associated by FIFO while allowing enough requests in flight
# to cover the 150 ms response deadline at the fastest permitted send spacing.
MAX_OUTSTANDING_REQUESTS = math.ceil(
    MAX_RESPONSE_WAIT_S / MIN_REQUEST_SPACING_S)


def _next_request_deadline(next_deadline: float, emitted_at: float,
                           period: float) -> float:
    """Limit phase catch-up to a small, bounded request-rate correction."""
    minimum_spacing = min(period, MIN_REQUEST_SPACING_S)
    return max(next_deadline, emitted_at + minimum_spacing)


# Allow the competition image to finish startup and deliver its first response.
# This is not evidence of a 40 Hz stream; only measured packets count.
STARTUP_RESPONSE_GRACE_S = 15.0


def _env_enabled(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


_command_lock = Semaphore()
_latest_command: dict[str, str] = {
    "V1 Throttle": "0.0",
    "V1 Steering": "0.0",
    "V1 Reset": "False",
    # These fields are understood by the patched Unity simulator. Older
    # builds ignore unknown command fields, so the API remains backwards
    # compatible while a rebuilt simulator is being rolled out.
    "V1 Publish Camera": "False",
    "V1 Publish Rear Camera": "False",
    "V1 Publish LIDAR Intensity": "False",
}
_reset_level = False
_reset_deadline_monotonic: float | None = None
_RESET_HOLD_SEC = 0.75
_stop_sender = Event()
_emit_lock = Semaphore()
_request_slots = Semaphore(MAX_OUTSTANDING_REQUESTS)
_client_connected = Event()

_packet_stamp_lock = Semaphore()
_packet_stamp: Any = None
_packet_timestamp_patch_installed = False

_pending_request_lock = Semaphore()
_connection_generation = 0
_bridge_request_sequence = 0
_pending_requests: deque[tuple[Any, int, Semaphore, int, int, dict[str, str]]] = deque()
_pending_request_started_ns: int | None = None
_last_request_emit_ns: int | None = None
_first_request_sent = False
_first_valid_packet = False
_valid_packet_count = 0
_bootstrap_packet_seen = Event()
_bootstrap_wait_started_ns: int | None = None
_last_packet_arrival_ns: int | None = None
_active_request_stamp: Any = None
_active_packet_stamp: Any = None
_active_request_monotonic_ns: int | None = None
_active_packet_arrival_ns: int | None = None
_active_request_sequence: int | None = None
_active_command: dict[str, str] = {}
_active_applied_command_sequence: int | None = None
_active_commanded_throttle_norm: float | None = None
_active_commanded_steering_norm: float | None = None
_active_applied_throttle_norm: float | None = None
_active_applied_steering_norm: float | None = None
_active_simulation_time_s: float | None = None
_active_simulation_frame: int | None = None
_active_simulation_physics_step: int | None = None
_active_telemetry_sequence: int | None = None
_active_simulator_packet: dict[str, float | int | None] = {}
_active_request_slot_pool: Semaphore | None = None

_packet_timing_lock = Semaphore()
_packet_handler_lock = Semaphore(1)
_packet_timing_publisher: Any = None
_timing_fault_publisher: Any = None
_timing_fault_detail_publisher: Any = None
_packet_sequence = 0
_packet_contract_installed = False
_handler_timing_enabled = False
_handler_timing_count = 0
_handler_timing_total_ns = 0
_publication_diagnostic_enabled = False
_publication_diagnostic_count = 0
_source_lidar_diagnostic_count = 0
_timing_fault = False
_timing_fault_reason = ""
_shutdown_requested = False
_timing_debug_count = 0
_request_timing_debug_count = 0

# Unity can emit one startup packet before its LiDAR range buffer has been
# populated.  The official decoder treats that optional field as mandatory and
# raises KeyError, which otherwise stops the request/response loop forever.
# Drop only that incomplete packet and request the next one; no ROS sensor
# message is published and no value is invented.
_STARTUP_REQUIRED_KEYS = ("V1 LIDAR Range Array",)


def _copy_stamp(stamp: Any) -> Any:
    """Copy a ROS time value without sharing a mutable message object."""
    return copy(stamp)


def _set_pending_request(
        slot_pool: Semaphore,
        generation: int,
        request_sequence: int,
        command: dict[str, str]) -> bool:
    """Record one request boundary in the bounded response FIFO."""
    global _pending_request_started_ns, _first_request_sent
    node = getattr(official_bridge, "autodrive_bridge", None)
    if node is None:
        return False
    try:
        stamp = node.get_clock().now().to_msg()
    except Exception:
        return False
    request_ns = time.monotonic_ns()
    with _pending_request_lock:
        if (not _client_connected.is_set() or
                generation != _connection_generation):
            return False
        if len(_pending_requests) >= MAX_OUTSTANDING_REQUESTS:
            # Refuse only when the explicitly bounded FIFO is full.
            # Responses are consumed in Socket.IO order, preserving the
            # request-sequence association stored in each slot.
            return False
        _pending_requests.append((_copy_stamp(stamp), request_ns, slot_pool,
                                  generation, request_sequence, dict(command)))
        _pending_request_started_ns = _pending_requests[0][1]
        _first_request_sent = True
    return True


def _discard_pending_request() -> None:
    """Remove the newest request context when sending the request failed."""
    global _pending_request_started_ns
    with _pending_request_lock:
        if _pending_requests:
            _pending_requests.pop()
        _pending_request_started_ns = (
            _pending_requests[0][1] if _pending_requests else None)


def _next_request_sequence() -> int:
    """Allocate a unique sequence for one emitted Bridge request."""
    global _bridge_request_sequence
    with _pending_request_lock:
        _bridge_request_sequence += 1
        return _bridge_request_sequence


def _reset_request_pipeline() -> None:
    """Drop requests belonging to a closed Unity socket.

    Each connection gets its own semaphore. A late callback from the old
    socket can therefore release only its old pool and cannot over-credit the
    new connection's bounded request slots.
    """
    global _request_slots, _connection_generation
    global _pending_request_started_ns, _last_request_emit_ns
    global _first_request_sent, _first_valid_packet, _valid_packet_count
    global _bootstrap_wait_started_ns, _last_packet_arrival_ns
    global _active_request_stamp, _active_packet_stamp
    global _active_request_monotonic_ns, _active_packet_arrival_ns
    global _active_request_sequence, _active_command
    global _active_applied_command_sequence
    global _active_commanded_throttle_norm, _active_commanded_steering_norm
    global _active_applied_throttle_norm, _active_applied_steering_norm
    global _active_simulation_time_s, _active_simulation_frame
    global _active_simulation_physics_step, _active_telemetry_sequence
    global _active_simulator_packet
    global _active_request_slot_pool, _packet_stamp
    with _pending_request_lock:
        _connection_generation += 1
        _request_slots = Semaphore(MAX_OUTSTANDING_REQUESTS)
        _pending_requests.clear()
        _pending_request_started_ns = None
        _last_request_emit_ns = None
        _first_request_sent = False
        _first_valid_packet = False
        _valid_packet_count = 0
        _bootstrap_wait_started_ns = None
        _last_packet_arrival_ns = None
        _active_request_stamp = None
        _active_packet_stamp = None
        _active_request_monotonic_ns = None
        _active_packet_arrival_ns = None
        _active_request_sequence = None
        _active_command = {}
        _active_applied_command_sequence = None
        _active_commanded_throttle_norm = None
        _active_commanded_steering_norm = None
        _active_applied_throttle_norm = None
        _active_applied_steering_norm = None
        _active_simulation_time_s = None
        _active_simulation_frame = None
        _active_simulation_physics_step = None
        _active_telemetry_sequence = None
        _active_simulator_packet = {}
        _active_request_slot_pool = None
    _bootstrap_packet_seen.clear()
    with _packet_stamp_lock:
        _packet_stamp = None


def _finish_packet() -> None:
    """Clear packet state and return the response's bounded pipeline slot."""
    global _packet_stamp, _active_request_stamp, _active_packet_stamp
    global _active_request_monotonic_ns, _active_packet_arrival_ns
    global _active_request_sequence, _active_command
    global _active_applied_command_sequence
    global _active_commanded_throttle_norm, _active_commanded_steering_norm
    global _active_applied_throttle_norm, _active_applied_steering_norm
    global _active_simulation_time_s, _active_simulation_frame
    global _active_simulation_physics_step, _active_telemetry_sequence
    global _active_simulator_packet
    global _active_request_slot_pool, _pending_request_started_ns
    with _packet_stamp_lock:
        _packet_stamp = None
    with _pending_request_lock:
        slot_pool = _active_request_slot_pool
        _active_request_stamp = None
        _active_packet_stamp = None
        _active_request_monotonic_ns = None
        _active_packet_arrival_ns = None
        _active_request_sequence = None
        _active_command = {}
        _active_applied_command_sequence = None
        _active_commanded_throttle_norm = None
        _active_commanded_steering_norm = None
        _active_applied_throttle_norm = None
        _active_applied_steering_norm = None
        _active_simulation_time_s = None
        _active_simulation_frame = None
        _active_simulation_physics_step = None
        _active_telemetry_sequence = None
        _active_simulator_packet = {}
        _active_request_slot_pool = None
        _pending_request_started_ns = (
            _pending_requests[0][1] if _pending_requests else None)
    if slot_pool is not None:
        slot_pool.release()


def _packet_timing_pub() -> Any:
    """Create the bridge-side timing publisher after the official node exists."""
    global _packet_timing_publisher, _timing_fault_publisher
    global _timing_fault_detail_publisher
    node = getattr(official_bridge, "autodrive_bridge", None)
    if node is None:
        return None
    with _packet_timing_lock:
        if _packet_timing_publisher is None:
            try:
                _packet_timing_publisher = node.create_publisher(
                    String, PACKET_TIMING_TOPIC, 100)
            except Exception as exc:
                print(f"[autodrive_bridge_40hz] timing publisher unavailable: {exc}")
                return None
        if _timing_fault_publisher is None:
            _timing_fault_publisher = node.create_publisher(
                Bool, TIMING_FAULT_TOPIC, 10)
        if _timing_fault_detail_publisher is None:
            _timing_fault_detail_publisher = node.create_publisher(
                String, TIMING_FAULT_DETAIL_TOPIC, 10)
        return _packet_timing_publisher


def _trigger_timing_fault(reason: str) -> None:
    """Fail closed when the native 40 Hz source contract is broken."""
    global _timing_fault, _timing_fault_reason
    global _timing_fault_detail_publisher
    # An already-buffered Socket.IO response can arrive after orderly ROS
    # teardown has cleared the request FIFO. That is not a source-timing
    # fault, and must not publish a false fatal condition while stopping.
    if _shutdown_requested:
        return
    with _pending_request_lock:
        if _timing_fault:
            return
        _timing_fault = True
        _timing_fault_reason = reason
    with _command_lock:
        _latest_command["V1 Throttle"] = "0.0"
        _latest_command["V1 Steering"] = "0.0"
        _latest_command["V1 Reset"] = "False"
    _stop_sender.set()
    print(
        "[autodrive_bridge_40hz] FATAL timing fault: " + reason +
        "; stream stopped and neutral command latched",
        flush=True,
    )
    node = getattr(official_bridge, "autodrive_bridge", None)
    publisher = _timing_fault_publisher
    if publisher is None and node is not None:
        try:
            publisher = node.create_publisher(Bool, TIMING_FAULT_TOPIC, 10)
        except Exception as exc:
            print(
                f"[autodrive_bridge_40hz] timing fault publication failed: {exc}",
                flush=True,
            )
    if publisher is not None:
        try:
            publisher.publish(Bool(data=True))
        except Exception as exc:
            print(
                f"[autodrive_bridge_40hz] timing fault publication failed: {exc}",
                flush=True,
            )
    detail_publisher = _timing_fault_detail_publisher
    if detail_publisher is None and node is not None:
        try:
            detail_publisher = node.create_publisher(
                String, TIMING_FAULT_DETAIL_TOPIC, 10)
            _timing_fault_detail_publisher = detail_publisher
        except Exception as exc:
            print(
                f"[autodrive_bridge_40hz] timing fault detail publication failed: {exc}",
                flush=True,
            )
    if detail_publisher is not None:
        try:
            detail = String()
            detail.data = reason
            detail_publisher.publish(detail)
        except Exception as exc:
            print(
                f"[autodrive_bridge_40hz] timing fault detail publication failed: {exc}",
                flush=True,
            )


def _on_reset_command(message: Bool) -> None:
    """Apply a reset level with a bounded safety timeout.

    The calibration harness normally sends an explicit true/false pulse, but
    a one-shot ``ros2 topic pub ... true`` must not leave Unity in a permanent
    reset loop.  Repeated true messages extend the pulse; an explicit false
    still releases it immediately.
    """
    global _reset_level, _reset_deadline_monotonic
    global _first_valid_packet, _last_packet_arrival_ns, _valid_packet_count
    next_level = bool(message.data)
    with _command_lock:
        changed = next_level != _reset_level
        _reset_level = next_level
        _reset_deadline_monotonic = (
            time.monotonic() + _RESET_HOLD_SEC if _reset_level else None)
        _latest_command["V1 Reset"] = "True" if _reset_level else "False"
    if changed and next_level:
        # Do not publish packets while reset is asserted. The first packet
        # after release uses its actual bridge receive time; no simulator
        # source-time epoch is needed.
        with _pending_request_lock:
            _first_valid_packet = False
            _last_packet_arrival_ns = None
            _valid_packet_count = 0
    if changed:
        print(
            f"[autodrive_bridge_40hz] reset ROS callback: {next_level}",
            flush=True,
        )


def _capture_competition_packet(data: Any) -> bool:
    """Pair a stock-image telemetry packet with the oldest pending request.

    The first packet after Socket.IO connect is the simulator's unsolicited
    bootstrap sample. It only opens the request gate and is never published.
    Subsequent packets require one pending Bridge request and are associated
    strictly by FIFO order. No Unity clock, frame counter, or echoed request
    identifier is read.
    """
    global _active_request_stamp, _active_packet_stamp
    global _active_request_monotonic_ns, _active_packet_arrival_ns
    global _active_request_sequence, _active_command
    global _active_simulator_packet, _active_request_slot_pool
    global _pending_request_started_ns

    if not isinstance(data, dict):
        _trigger_timing_fault("non-dictionary simulator packet")
        return False

    with _pending_request_lock:
        has_pending_request = bool(_pending_requests)
        bootstrap_already_seen = _bootstrap_packet_seen.is_set()
        connected = _client_connected.is_set()
        if not has_pending_request and connected and not bootstrap_already_seen:
            bootstrap_packet = True
        else:
            bootstrap_packet = False

    if bootstrap_packet:
        _bootstrap_packet_seen.set()
        print(
            "[autodrive_bridge_40hz] competition bootstrap received; "
            "starting request stream",
            flush=True,
        )
        return False

    if not has_pending_request:
        _trigger_timing_fault(
            "simulator packet arrived without a pending Bridge request")
        return False

    node = getattr(official_bridge, "autodrive_bridge", None)
    if node is None:
        _trigger_timing_fault("cannot timestamp packet: official ROS node unavailable")
        return False
    try:
        receive_stamp = node.get_clock().now().to_msg()
    except Exception as exc:
        _trigger_timing_fault(f"cannot timestamp received packet: {exc}")
        return False
    arrival_ns = time.monotonic_ns()

    def parse_float(value: Any) -> float | None:
        try:
            return None if value in (None, "") else float(value)
        except (TypeError, ValueError):
            return None

    def parse_vector(value: Any, size: int) -> list[float | None]:
        try:
            values = [float(item) for item in str(value).split()]
        except (TypeError, ValueError):
            values = []
        return [values[index] if index < len(values) else None
                for index in range(size)]

    simulator_packet: dict[str, float | int | None] = {}
    vector_fields = (
        ("simulator_position", "V1 Position", ("x", "y", "z")),
        ("simulator_orientation_quaternion", "V1 Orientation Quaternion",
         ("x", "y", "z", "w")),
        ("simulator_orientation_euler", "V1 Orientation Euler Angles",
         ("x", "y", "z")),
        ("simulator_linear_velocity", "V1 Linear Velocity", ("x", "y", "z")),
        ("simulator_angular_velocity", "V1 Angular Velocity", ("x", "y", "z")),
        ("simulator_linear_acceleration", "V1 Linear Acceleration", ("x", "y", "z")),
        ("simulator_encoder_angles", "V1 Encoder Angles", ("left", "right")),
    )
    for prefix, key, suffixes in vector_fields:
        for suffix, value in zip(suffixes, parse_vector(data.get(key), len(suffixes))):
            simulator_packet[f"{prefix}_{suffix}"] = value
    simulator_packet.update({
        "simulator_feedback_throttle_norm": parse_float(data.get("V1 Throttle")),
        "simulator_feedback_steering_norm": parse_float(data.get("V1 Steering")),
        "simulator_collision_count": _parse_optional_int(data.get("V1 Collisions")),
    })

    association_lost = False
    with _pending_request_lock:
        if not _pending_requests:
            association_lost = True
        else:
            (request_stamp, request_ns, slot_pool, _, request_sequence,
             command) = _pending_requests.popleft()
            _pending_request_started_ns = (
                _pending_requests[0][1] if _pending_requests else None)
            _active_request_slot_pool = slot_pool
            _active_request_stamp = request_stamp
            _active_packet_stamp = receive_stamp
            _active_request_monotonic_ns = request_ns
            _active_packet_arrival_ns = arrival_ns
            _active_request_sequence = request_sequence
            _active_command = command
            _active_simulator_packet = simulator_packet
            # Enhanced Unity-only fields are not runtime inputs and may be
            # absent from the competition image. Keep diagnostic slots empty.
            _active_applied_command_sequence = None
            _active_commanded_throttle_norm = None
            _active_commanded_steering_norm = None
            _active_applied_throttle_norm = None
            _active_applied_steering_norm = None
            _active_simulation_time_s = None
            _active_simulation_frame = None
            _active_simulation_physics_step = None
            _active_telemetry_sequence = None
    if association_lost:
        _trigger_timing_fault(
            "pending Bridge request disappeared during packet association")
        return False
    return True


def _validate_received_packet() -> bool:
    """Validate local transport cadence without requiring simulator metadata."""
    global _first_valid_packet, _valid_packet_count, _last_packet_arrival_ns
    with _pending_request_lock:
        arrival_ns = _active_packet_arrival_ns
        previous_arrival_ns = _last_packet_arrival_ns
        request_ns = _active_request_monotonic_ns
        request_sequence = _active_request_sequence
    if arrival_ns is None:
        _trigger_timing_fault("received packet has no local arrival timestamp")
        return False

    with _command_lock:
        reset_active = _reset_level
    if reset_active:
        return False

    arrival_dt = (None if previous_arrival_ns is None else
                  (arrival_ns - previous_arrival_ns) / 1.0e9)
    if arrival_dt is not None and arrival_dt > MAX_BRIDGE_ARRIVAL_INTERVAL_S:
        request_age_s = (None if request_ns is None else
                         (arrival_ns - request_ns) / 1.0e9)
        _trigger_timing_fault(
            "bridge packet arrival interval exceeded "
            f"{MAX_BRIDGE_ARRIVAL_INTERVAL_S:.3f}s: "
            f"arrival_dt={arrival_dt:.6f}s "
            f"request_sequence={request_sequence} "
            f"request_response_age={request_age_s}")
        return False

    with _pending_request_lock:
        _last_packet_arrival_ns = arrival_ns
        _first_valid_packet = True
        _valid_packet_count += 1
    return True


def _parse_command_float(command: dict[str, str], key: str) -> float | None:
    try:
        return float(command[key])
    except (KeyError, TypeError, ValueError):
        return None


def _parse_optional_int(value: Any) -> int | None:
    try:
        return None if value in (None, "") else int(value)
    except (TypeError, ValueError):
        return None


def _record_packet_arrival() -> None:
    """Publish one diagnostic event at the start of each decoded packet."""
    global _packet_sequence
    arrival_ns = _active_packet_arrival_ns or time.monotonic_ns()
    with _pending_request_lock:
        request_ns = _active_request_monotonic_ns
        request_sequence = _active_request_sequence
        command = dict(_active_command)
        applied_command_sequence = _active_applied_command_sequence
        commanded_throttle_norm = _active_commanded_throttle_norm
        commanded_steering_norm = _active_commanded_steering_norm
        applied_throttle_norm = _active_applied_throttle_norm
        applied_steering_norm = _active_applied_steering_norm
        simulation_time_s = _active_simulation_time_s
        simulation_frame = _active_simulation_frame
        simulation_physics_step = _active_simulation_physics_step
        telemetry_sequence = _active_telemetry_sequence
        simulator_packet = dict(_active_simulator_packet)
        connection_generation = _connection_generation
        receive_stamp = _active_packet_stamp
    with _packet_timing_lock:
        _packet_sequence += 1
        sequence = _packet_sequence
    publisher = _packet_timing_pub()
    if publisher is None:
        return
    message = String()
    message.data = json.dumps({
        "schema_version": 3,
        "packet_sequence": sequence,
        "bridge_arrival_monotonic_ns": arrival_ns,
        "bridge_receive_ros_stamp_ns": (
            None if receive_stamp is None else
            int(receive_stamp.sec) * 1_000_000_000 + int(receive_stamp.nanosec)),
        "request_monotonic_ns": request_ns,
        "request_sequence": request_sequence,
        "connection_generation": connection_generation,
        "sent_throttle_norm": _parse_command_float(command, "V1 Throttle"),
        "sent_steering_norm": _parse_command_float(command, "V1 Steering"),
        "sent_reset": command.get("V1 Reset"),
        "applied_command_sequence": applied_command_sequence,
        "commanded_throttle_norm": commanded_throttle_norm,
        "commanded_steering_norm": commanded_steering_norm,
        "applied_throttle_norm": applied_throttle_norm,
        "applied_steering_norm": applied_steering_norm,
        "simulation_time_s": None,
        "simulation_frame": None,
        "simulation_render_frame": None,
        "simulation_physics_step": None,
        "telemetry_sequence": None,
        **simulator_packet,
    }, separators=(",", ":"))
    try:
        publisher.publish(message)
    except Exception as exc:
        print(f"[autodrive_bridge_40hz] timing event publish failed: {exc}")


def _install_packet_timestamp_patch() -> None:
    """Stamp every ROS message from a packet with its local receive time."""
    global _packet_timestamp_patch_installed
    if _packet_timestamp_patch_installed:
        return

    def stamped_creator(original: Any) -> Any:
        def create(*args: Any, **kwargs: Any) -> Any:
            global _packet_stamp
            message = original(*args, **kwargs)
            if not hasattr(message, "header"):
                return message
            with _packet_stamp_lock:
                if _packet_stamp is None:
                    with _pending_request_lock:
                        pending = _active_packet_stamp
                    _packet_stamp = _copy_stamp(
                        pending if pending is not None else message.header.stamp)
                message.header.stamp = _copy_stamp(_packet_stamp)
            return message
        return create

    for name in (
        "create_joint_state_msg", "create_imu_msg", "create_odom_msg",
        "create_laserscan_msg", "create_image_msg", "create_tf_msg",
    ):
        if hasattr(official_bridge, name):
            setattr(official_bridge, name, stamped_creator(
                getattr(official_bridge, name)))

    original_encoder_publish = official_bridge.publish_encoder_data

    def begin_packet(*args: Any, **kwargs: Any) -> Any:
        global _packet_stamp
        with _packet_stamp_lock:
            _packet_stamp = None
        _record_packet_arrival()
        return original_encoder_publish(*args, **kwargs)

    official_bridge.publish_encoder_data = begin_packet
    _packet_timestamp_patch_installed = True


def _rate_hz() -> float:
    value = os.environ.get("AUTODRIVE_BRIDGE_RATE_HZ", str(DEFAULT_RATE_HZ))
    try:
        rate = float(value)
    except ValueError as exc:
        raise RuntimeError("AUTODRIVE_BRIDGE_RATE_HZ must be exactly 40 Hz") from exc
    if not math.isclose(rate, EXPECTED_RATE_HZ, rel_tol=0.0, abs_tol=1e-9):
        raise RuntimeError(
            f"AUTODRIVE_BRIDGE_RATE_HZ must be exactly {EXPECTED_RATE_HZ:g} Hz")
    return rate


def _remember_command(data: Any) -> None:
    if not isinstance(data, dict) or "V1 Throttle" not in data:
        return
    command = {
        "V1 Throttle": str(data.get("V1 Throttle", "0.0")),
        "V1 Steering": str(data.get("V1 Steering", "0.0")),
        "V1 Reset": str(data.get("V1 Reset", "False")),
    }
    with _command_lock:
        _latest_command.update(command)


def _run_command_sender(original_emit: Any, rate_hz: float) -> None:
    """Send one request per source period with a response deadline."""
    global _request_slots, _connection_generation, _last_request_emit_ns
    global _request_timing_debug_count
    global _bootstrap_wait_started_ns
    period = 1.0 / rate_hz
    next_send = time.monotonic()
    while not _stop_sender.is_set():
        now = time.monotonic()
        if _timing_fault:
            return
        with _pending_request_lock:
            pending_started_ns = _pending_request_started_ns
        if pending_started_ns is not None:
            response_age_s = (
                time.monotonic_ns() - pending_started_ns) / 1e9
            response_deadline_s = (
                MAX_RESPONSE_WAIT_S if _first_valid_packet
                else STARTUP_RESPONSE_GRACE_S)
            if response_age_s > response_deadline_s:
                _trigger_timing_fault(
                    f"no simulator response within {response_deadline_s:.3f}s")
                return
        wait_time = next_send - now
        if wait_time > 0.0:
            gevent.sleep(wait_time)
            continue

        if (_valid_packet_count >= 2 and
                _last_request_emit_ns is not None and
                now - _last_request_emit_ns > MAX_REQUEST_SCHEDULE_GAP_S):
            _trigger_timing_fault(
                f"request transport schedule exceeded "
                f"{MAX_REQUEST_SCHEDULE_GAP_S:.3f}s")
            return

        next_send += period
        if next_send < now:
            missed = int((now - next_send) / period) + 1
            next_send += missed * period

        if not _client_connected.is_set():
            gevent.sleep(period)
            continue
        if not _bootstrap_packet_seen.is_set():
            if _bootstrap_wait_started_ns is None:
                _bootstrap_wait_started_ns = time.monotonic_ns()
            bootstrap_age_s = (
                time.monotonic_ns() - _bootstrap_wait_started_ns) / 1.0e9
            if bootstrap_age_s > STARTUP_RESPONSE_GRACE_S:
                _trigger_timing_fault(
                    "no competition bootstrap packet received within "
                    f"{STARTUP_RESPONSE_GRACE_S:.3f}s")
                return
            gevent.sleep(period)
            continue
        with _pending_request_lock:
            slot_pool = _request_slots
            generation = _connection_generation
        if not slot_pool.acquire(blocking=False):
            continue

        with _command_lock:
            global _reset_level, _reset_deadline_monotonic
            if (_reset_level and _reset_deadline_monotonic is not None and
                    now >= _reset_deadline_monotonic):
                _reset_level = False
                _reset_deadline_monotonic = None
                _latest_command["V1 Reset"] = "False"
            command = dict(_latest_command)
            command["V1 Reset"] = "True" if _reset_level else "False"

        request_sequence = _next_request_sequence()
        if not _set_pending_request(
                slot_pool, generation, request_sequence, command):
            slot_pool.release()
            continue
        try:
            # Keep all outgoing Socket.IO calls serialized. The official bridge
            # runs a gevent server and this pacing loop is a greenlet.
            with _emit_lock:
                emitted_at = time.monotonic()
                _last_request_emit_ns = emitted_at
                original_emit("Bridge", data=command)
                # Keep the nominal 25 ms phase schedule, but do not issue a
                # catch-up request less than 24 ms after an actual send. This
                # avoids the observed 16--20 ms bursts without accumulating
                # timer oversleep into a permanently slower cadence.
                next_send = _next_request_deadline(
                    next_send, emitted_at, period)
            if (_env_enabled("AUTODRIVE_TIMING_DEBUG", False) and
                    _request_timing_debug_count < 20):
                _request_timing_debug_count += 1
                print(
                    "[autodrive_bridge_40hz] request sent: "
                    f"sequence={request_sequence} monotonic={time.monotonic():.9f}",
                    flush=True,
                )
        except Exception as exc:
            _discard_pending_request()
            slot_pool.release()
            _trigger_timing_fault(f"Bridge request send failed: {exc}")
            return


def _install_packet_contract(publish_camera: bool) -> None:
    """Accept both legacy and source-camera-disabled simulator packets."""
    global _packet_contract_installed
    if _packet_contract_installed:
        return

    handlers = official_bridge.sio.handlers.get("/", {})
    original_handler = handlers.get("Bridge")
    if original_handler is None:
        raise RuntimeError("official AutoDRIVE bridge has no Bridge handler")

    def packet_handler(sid: Any, data: Any) -> Any:
        # Socket.IO may schedule event callbacks concurrently. Keep the
        # source packet contract and official decoder in arrival order so the
        # shared timing state cannot observe packet N+1 before packet N has
        # finished decoding/publication.
        with _packet_handler_lock:
            return _packet_handler_body(sid, data)

    def _packet_handler_body(sid: Any, data: Any) -> Any:
        global _handler_timing_count, _handler_timing_total_ns
        handler_start_ns = time.monotonic_ns()
        has_request = _capture_competition_packet(data)
        try:
            if _shutdown_requested:
                return None
            if _timing_fault:
                return None
            if not has_request:
                # Socket.OnConnect emits bootstrap telemetry without a Bridge
                # request. Unity may emit that callback more than once while
                # reconnecting, including after the first requested packet.
                # It is never a sensor sample and must never be associated
                # with a command or published to ROS. The outstanding
                # requested response remains deadline-protected by the sender.
                return None
            if not _validate_received_packet():
                return None
            if not publish_camera and isinstance(data, dict):
                # The official bridge accesses this key unconditionally.
                # Injecting an empty placeholder keeps its numeric decoder
                # compatible with the source-disabled packet and replaces a
                # legacy image field before base64/PIL work can happen.
                data = dict(data)
                data["V1 Front Camera Image"] = ""
            if _publication_diagnostic_enabled and isinstance(data, dict):
                _log_source_lidar_payload(data)
            if (_env_enabled("AUTODRIVE_ALLOW_MISSING_LIDAR", False) and
                    isinstance(data, dict)):
                _inject_empty_lidar_packet(data)
            if isinstance(data, dict) and any(
                    key not in data for key in _STARTUP_REQUIRED_KEYS):
                missing = ", ".join(
                    key for key in _STARTUP_REQUIRED_KEYS if key not in data)
                print(
                    "[autodrive_bridge_40hz] dropping incomplete startup packet "
                    f"(missing {missing})",
                    flush=True,
                )
                return None
            try:
                result = original_handler(sid, data)
            except Exception as exc:
                # Do not let a decoder/publication exception silently kill
                # the Socket.IO callback. A silent callback death leaves the
                # calibration node to report only a generic telemetry
                # timeout and can make the partial dataset look complete.
                _trigger_timing_fault(
                    "simulator packet decode/publication failed: "
                    f"{type(exc).__name__}: {exc}")
                return None
            if _handler_timing_enabled:
                duration_ns = time.monotonic_ns() - handler_start_ns
                _handler_timing_count += 1
                _handler_timing_total_ns += duration_ns
                if _handler_timing_count % 40 == 0:
                    average_ms = _handler_timing_total_ns / _handler_timing_count / 1e6
                    print(
                        "[autodrive_bridge_40hz] handler timing: "
                        f"last={duration_ns / 1e6:.2f} ms average={average_ms:.2f} ms",
                        flush=True,
                    )
            return result
        finally:
            _finish_packet()

    handlers["Bridge"] = packet_handler
    _packet_contract_installed = True


def _disable_camera_path() -> None:
    """Avoid camera decode/publication while retaining numeric telemetry."""
    official_bridge.publish_camera_images = lambda *_args, **_kwargs: None
    official_bridge.Image.open = lambda *_args, **_kwargs: (
        official_bridge.np.empty((1, 1, 3), dtype=official_bridge.np.uint8))


def _inject_empty_lidar_packet(data: dict[str, Any]) -> None:
    """Satisfy the official decoder for the no-LiDAR model-ID scene."""
    empty_ranges = base64.b64encode(gzip.compress(b"")).decode("ascii")
    data.setdefault("V1 LIDAR Scan Rate", "40.0")
    data.setdefault("V1 LIDAR Range Array", empty_ranges)
    data.setdefault("V1 Lap Count", "0")
    data.setdefault("V1 Lap Time", "0.0")
    data.setdefault("V1 Last Lap Time", "0.0")
    data.setdefault("V1 Best Lap Time", "0.0")
    data.setdefault("V1 Collisions", "0")


def _log_source_lidar_payload(data: dict[str, Any]) -> None:
    """Compare source-packet ranges with the decoded ROS scan, only on request."""
    global _source_lidar_diagnostic_count
    _source_lidar_diagnostic_count += 1
    if _source_lidar_diagnostic_count % 40 != 0:
        return

    encoded = data.get("V1 LIDAR Range Array")
    payload_chars = len(encoded) if isinstance(encoded, (str, bytes)) else -1
    try:
        decoded = gzip.decompress(base64.b64decode(encoded))
        values = official_bridge.np.fromstring(
            decoded.decode("utf-8"), dtype=float, sep="\n")
        finite_count = int(official_bridge.np.isfinite(values).sum())
        print(
            "[autodrive_bridge_40hz] source lidar payload: "
            f"packet={_source_lidar_diagnostic_count} "
            f"key_present={encoded is not None} payload_chars={payload_chars} "
            f"decoded_ranges={len(values)} finite_ranges={finite_count}",
            flush=True,
        )
    except Exception as exc:
        print(
            "[autodrive_bridge_40hz] source lidar payload: "
            f"packet={_source_lidar_diagnostic_count} "
            f"key_present={encoded is not None} payload_chars={payload_chars} "
            f"decode_error={type(exc).__name__}",
            flush=True,
        )


def _install_publication_diagnostic() -> None:
    """Optionally report whether decoded LaserScan messages reach DDS."""
    global _publication_diagnostic_count
    original_publish_lidar_scan = official_bridge.publish_lidar_scan

    def diagnostic_publish(*args: Any, **kwargs: Any) -> Any:
        global _publication_diagnostic_count
        _publication_diagnostic_count += 1
        result = original_publish_lidar_scan(*args, **kwargs)
        if _publication_diagnostic_count % 40 == 0:
            ranges = (args[1] if len(args) > 1 else
                      kwargs.get("lidar_range_array", ()))
            try:
                range_count = len(ranges)
                finite_count = sum(
                    math.isfinite(float(value)) for value in ranges)
            except (TypeError, ValueError):
                range_count = -1
                finite_count = -1
            node = getattr(official_bridge, "autodrive_bridge", None)
            publisher_count = "unknown"
            publisher_qos = "unknown"
            if node is not None:
                try:
                    lidar_publisher = official_bridge.publishers["pub_lidar"]
                    publisher_count = str(lidar_publisher.get_subscription_count())
                    publisher_qos = str(lidar_publisher.qos_profile)
                except Exception as exc:
                    publisher_count = f"error:{exc}"
            print(
                "[autodrive_bridge_40hz] lidar publication diagnostic: "
                f"calls={_publication_diagnostic_count} subscribers={publisher_count} "
                f"decoded_finite_ranges={finite_count}/{range_count} "
                f"qos={publisher_qos}",
                flush=True,
            )
        return result

    official_bridge.publish_lidar_scan = diagnostic_publish


def _force_ipv4_server() -> None:
    """Make the SDK bridge reachable by the prebuilt Unity player.

    The SDK calls ``WSGIServer(("", 4567), ...)``.  On this host gevent
    resolves that wildcard to an IPv6-only listener, while the Unity player
    is launched with its documented IPv4 loopback endpoint ``127.0.0.1``.
    Unity then reports a connection refusal or enters its null SocketIO state
    before any sensor packet exists.  Keep the SDK unchanged, but normalize
    only its wildcard listener to an explicit IPv4 address.
    """
    original_server = official_bridge.pywsgi.WSGIServer

    class IPv4WSGIServer(original_server):
        def __init__(self, listener, *args, **kwargs):
            if isinstance(listener, tuple) and len(listener) >= 2 and listener[0] == "":
                listener = ("0.0.0.0", listener[1], *listener[2:])
            super().__init__(listener, *args, **kwargs)

    official_bridge.pywsgi.WSGIServer = IPv4WSGIServer


def main() -> None:
    global _handler_timing_enabled, _publication_diagnostic_enabled
    global _shutdown_requested
    _shutdown_requested = False
    rate_hz = _rate_hz()
    _handler_timing_enabled = _env_enabled(
        "AUTODRIVE_BRIDGE_LOG_HANDLER_TIMING", False)
    _publication_diagnostic_enabled = _env_enabled(
        "AUTODRIVE_BRIDGE_LOG_PUBLICATION", False)
    _force_ipv4_server()
    _install_packet_timestamp_patch()
    publish_camera = _env_enabled("AUTODRIVE_BRIDGE_PUBLISH_CAMERA", False)
    publish_lidar_intensity = _env_enabled(
        "AUTODRIVE_BRIDGE_PUBLISH_LIDAR_INTENSITY", False)
    with _command_lock:
        _latest_command["V1 Publish Camera"] = "True" if publish_camera else "False"
        _latest_command["V1 Publish Rear Camera"] = "False"
        _latest_command["V1 Publish LIDAR Intensity"] = (
            "True" if publish_lidar_intensity else "False")
    _install_packet_contract(publish_camera)
    if not publish_camera:
        _disable_camera_path()
        print("[autodrive_bridge_40hz] camera decode/publication disabled")
    if _publication_diagnostic_enabled:
        _install_publication_diagnostic()
        print("[autodrive_bridge_40hz] publication diagnostic enabled")
    original_emit = official_bridge.sio.emit

    original_connect = official_bridge.sio.handlers.get("/", {}).get("connect")
    original_disconnect = official_bridge.sio.handlers.get("/", {}).get("disconnect")
    original_reset_callback = getattr(
        official_bridge, "callback_reset_command", None)

    # The official bridge registers its ROS subscriptions during
    # ``official_bridge.main()``. Wrap that callback before startup so reset
    # levels are received by the same executor/thread as the native bridge,
    # instead of creating a second subscription lazily from a Socket.IO
    # packet callback.
    if original_reset_callback is not None:
        def wrapped_reset_callback(message: Bool) -> Any:
            _on_reset_command(message)
            return original_reset_callback(message)

        official_bridge.callback_reset_command = wrapped_reset_callback

    def mark_connected(sid: Any, environ: Any) -> Any:
        global _bootstrap_wait_started_ns
        _reset_request_pipeline()
        _bootstrap_wait_started_ns = time.monotonic_ns()
        _client_connected.set()
        if original_connect is not None:
            return original_connect(sid, environ)
        return None

    def mark_disconnected(sid: Any) -> Any:
        had_active_session = _client_connected.is_set() or _first_request_sent
        _client_connected.clear()
        _reset_request_pipeline()
        if had_active_session and not _shutdown_requested:
            _trigger_timing_fault("simulator Socket.IO connection lost")
        if original_disconnect is not None:
            return original_disconnect(sid)
        return None

    official_bridge.sio.handlers.setdefault("/", {})["connect"] = mark_connected
    official_bridge.sio.handlers.setdefault("/", {})["disconnect"] = mark_disconnected

    def intercept_emit(event: str, data: Any = None, *args: Any, **kwargs: Any) -> Any:
        if event == "Bridge":
            _remember_command(data)
            return None
        return original_emit(event, data, *args, **kwargs)

    official_bridge.sio.emit = intercept_emit
    _stop_sender.clear()
    _client_connected.clear()
    _reset_request_pipeline()
    # The official server runs on gevent.  Keep the pacing loop in that same
    # event loop; calling Server.emit from a native thread can leave the
    # Socket.IO connection established while silently dropping the outbound
    # Bridge event that starts telemetry.
    sender = official_bridge.sio.start_background_task(
        _run_command_sender, original_emit, rate_hz)
    print(
        f"[autodrive_bridge_40hz] strict request pacing enabled at {rate_hz:g} Hz "
        f"({MAX_OUTSTANDING_REQUESTS} bounded FIFO requests; "
        "stock-image FIFO association; missing/deadline data is fatal)")
    try:
        official_bridge.main()
    finally:
        _shutdown_requested = True
        _stop_sender.set()
        _client_connected.clear()
        _reset_request_pipeline()
        sender.join(timeout=max(1.0, 2.0 / rate_hz))
        official_bridge.sio.emit = original_emit
        if original_reset_callback is not None:
            official_bridge.callback_reset_command = original_reset_callback


if __name__ == "__main__":
    main()
