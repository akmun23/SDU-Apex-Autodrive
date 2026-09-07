"""Run the official AutoDRIVE bridge with bounded 40 Hz request pacing.

The simulator returns one telemetry packet after each ``Bridge`` request. The
official bridge requests the next packet from inside the previous packet's
callback, so simulator image capture, image decoding, and ROS publication can
create a request backlog.
This wrapper keeps the official decoder and topic contract, but sends requests
from an independent 40 Hz clock and never allows more than one request to be
outstanding. A late packet therefore causes a missed slot, not a burst.

No ROS sensor sample is fabricated, interpolated, or repeated. The bridge also
publishes a diagnostic packet-arrival event containing monotonic bridge-side
timestamps. That diagnostic stream is not a sensor input and is used only by
the timing gate. Camera decoding and ROS publication are disabled by default;
the API accepts both the patched source packet without a camera field and the
legacy packet that still contains one.
"""

from __future__ import annotations

import json
import math
import os
import time
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
_response_ready = Event()
_emit_lock = Semaphore()

_packet_stamp_lock = Semaphore()
_packet_stamp: Any = None
_packet_timestamp_patch_installed = False

_pending_request_lock = Semaphore()
_pending_request_stamp: Any = None
_pending_request_monotonic_ns: int | None = None
_pending_simulation_time_s: float | None = None
_pending_simulation_frame: int | None = None

_packet_timing_lock = Semaphore()
_packet_timing_publisher: Any = None
_reset_subscription: Any = None
_packet_sequence = 0
_packet_contract_installed = False
_handler_timing_enabled = False
_handler_timing_count = 0
_handler_timing_total_ns = 0

# Unity can emit one startup packet before its LiDAR range buffer has been
# populated.  The official decoder treats that optional field as mandatory and
# raises KeyError, which otherwise stops the request/response loop forever.
# Drop only that incomplete packet and request the next one; no ROS sensor
# message is published and no value is invented.
_STARTUP_REQUIRED_KEYS = ("V1 LIDAR Range Array",)


def _copy_stamp(stamp: Any) -> Any:
    """Copy a ROS time value without sharing a mutable message object."""
    return copy(stamp)


def _set_pending_request() -> None:
    """Record the request boundary for the one packet it will produce."""
    global _pending_request_monotonic_ns, _pending_request_stamp
    node = getattr(official_bridge, "autodrive_bridge", None)
    if node is None:
        return
    try:
        stamp = node.get_clock().now().to_msg()
    except Exception:
        return
    with _pending_request_lock:
        _pending_request_stamp = _copy_stamp(stamp)
        _pending_request_monotonic_ns = time.monotonic_ns()


def _packet_timing_pub() -> Any:
    """Create the bridge-side timing publisher after the official node exists."""
    global _packet_timing_publisher, _reset_subscription
    node = getattr(official_bridge, "autodrive_bridge", None)
    if node is None:
        return None
    with _packet_timing_lock:
        if _reset_subscription is None:
            _reset_subscription = node.create_subscription(
                Bool, "/autodrive/reset_command", _on_reset_command, 10)
        if _packet_timing_publisher is None:
            try:
                _packet_timing_publisher = node.create_publisher(
                    String, PACKET_TIMING_TOPIC, 100)
            except Exception as exc:
                print(f"[autodrive_bridge_40hz] timing publisher unavailable: {exc}")
                return None
        return _packet_timing_publisher


def _on_reset_command(message: Bool) -> None:
    """Apply a reset level with a bounded safety timeout.

    The calibration harness normally sends an explicit true/false pulse, but
    a one-shot ``ros2 topic pub ... true`` must not leave Unity in a permanent
    reset loop.  Repeated true messages extend the pulse; an explicit false
    still releases it immediately.
    """
    global _reset_level, _reset_deadline_monotonic
    with _command_lock:
        _reset_level = bool(message.data)
        _reset_deadline_monotonic = (
            time.monotonic() + _RESET_HOLD_SEC if _reset_level else None)
        _latest_command["V1 Reset"] = "True" if _reset_level else "False"


def _capture_simulation_metadata(data: Any) -> None:
    """Capture optional Unity physics metadata before official decoding.

    The patched simulator sends these values in the same packet as the
    sensors. They are diagnostics only; ROS header stamps remain host-side
    packet-boundary stamps until a source-time ROS clock contract exists.
    """
    global _pending_simulation_time_s, _pending_simulation_frame
    if not isinstance(data, dict):
        return
    time_value = data.get("Simulation Time")
    frame_value = data.get("Simulation Frame")
    try:
        simulation_time_s = None if time_value in (None, "") else float(time_value)
    except (TypeError, ValueError):
        simulation_time_s = None
    try:
        simulation_frame = None if frame_value in (None, "") else int(frame_value)
    except (TypeError, ValueError):
        simulation_frame = None
    with _pending_request_lock:
        _pending_simulation_time_s = simulation_time_s
        _pending_simulation_frame = simulation_frame


def _record_packet_arrival() -> None:
    """Publish one diagnostic event at the start of each decoded packet."""
    global _packet_sequence
    arrival_ns = time.monotonic_ns()
    with _pending_request_lock:
        request_ns = _pending_request_monotonic_ns
        simulation_time_s = _pending_simulation_time_s
        simulation_frame = _pending_simulation_frame
    with _packet_timing_lock:
        _packet_sequence += 1
        sequence = _packet_sequence
    publisher = _packet_timing_pub()
    if publisher is None:
        return
    message = String()
    message.data = json.dumps({
        "packet_sequence": sequence,
        "bridge_arrival_monotonic_ns": arrival_ns,
        "request_monotonic_ns": request_ns,
        "simulation_time_s": simulation_time_s,
        "simulation_frame": simulation_frame,
    }, separators=(",", ":"))
    try:
        publisher.publish(message)
    except Exception as exc:
        print(f"[autodrive_bridge_40hz] timing event publish failed: {exc}")


def _install_packet_timestamp_patch() -> None:
    """Give every ROS message from one packet one request-boundary stamp.

    AutoDRIVE ROS messages use host-side clock stamps. With one outstanding
    request, the request boundary is an unambiguous packet boundary and is
    preferable to independently stamping every helper during callback
    processing. The patched simulator's physics time/frame are carried in the
    diagnostic packet and are not substituted into ROS headers.
    """
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
                        pending = _pending_request_stamp
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

    original_collision_publish = official_bridge.publish_collision_count_data

    def end_packet(*args: Any, **kwargs: Any) -> Any:
        global _packet_stamp, _pending_request_monotonic_ns, _pending_request_stamp
        global _pending_simulation_time_s, _pending_simulation_frame
        try:
            return original_collision_publish(*args, **kwargs)
        finally:
            with _packet_stamp_lock:
                _packet_stamp = None
            with _pending_request_lock:
                _pending_request_stamp = None
                _pending_request_monotonic_ns = None
                _pending_simulation_time_s = None
                _pending_simulation_frame = None

    official_bridge.publish_encoder_data = begin_packet
    official_bridge.publish_collision_count_data = end_packet
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
    """Send at scheduled slots, never with more than one request outstanding."""
    period = 1.0 / rate_hz
    next_send = time.monotonic()
    while not _stop_sender.is_set():
        now = time.monotonic()
        wait_time = next_send - now
        if wait_time > 0.0:
            gevent.sleep(wait_time)
            continue

        next_send += period
        if next_send < now:
            missed = int((now - next_send) / period) + 1
            next_send += missed * period

        if not _response_ready.is_set():
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

        _response_ready.clear()
        _set_pending_request()
        try:
            # Keep all outgoing Socket.IO calls serialized. The official bridge
            # runs a gevent server and this pacing loop is a greenlet.
            with _emit_lock:
                original_emit("Bridge", data=command)
        except Exception as exc:
            _response_ready.set()
            print(f"[autodrive_bridge_40hz] command send failed: {exc}")


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
        global _handler_timing_count, _handler_timing_total_ns
        handler_start_ns = time.monotonic_ns()
        _capture_simulation_metadata(data)
        if not publish_camera and isinstance(data, dict):
            # The official bridge accesses this key unconditionally. Injecting
            # an empty placeholder keeps its numeric decoder compatible with
            # the source-disabled packet and replaces a legacy image field
            # before base64/PIL work can happen.
            data = dict(data)
            data["V1 Front Camera Image"] = ""
        if isinstance(data, dict) and any(
                key not in data for key in _STARTUP_REQUIRED_KEYS):
            missing = ", ".join(
                key for key in _STARTUP_REQUIRED_KEYS if key not in data)
            print(
                "[autodrive_bridge_40hz] dropping incomplete startup packet "
                f"(missing {missing}); requesting retry",
                flush=True,
            )
            _retry_incomplete_packet()
            return None
        result = original_handler(sid, data)
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

    handlers["Bridge"] = packet_handler
    _packet_contract_installed = True


def _retry_incomplete_packet() -> None:
    """Release the paced sender after dropping incomplete telemetry."""
    # Treat the incomplete packet as the response to the outstanding request.
    # The normal sender will issue the next request at its next 40 Hz slot,
    # preserving the one-outstanding-request invariant.
    _response_ready.set()


def _disable_camera_path() -> None:
    """Avoid camera decode/publication while retaining numeric telemetry."""
    official_bridge.publish_camera_images = lambda *_args, **_kwargs: None
    official_bridge.Image.open = lambda *_args, **_kwargs: (
        official_bridge.np.empty((1, 1, 3), dtype=official_bridge.np.uint8))


def main() -> None:
    global _handler_timing_enabled
    rate_hz = _rate_hz()
    _handler_timing_enabled = _env_enabled(
        "AUTODRIVE_BRIDGE_LOG_HANDLER_TIMING", False)
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
    original_emit = official_bridge.sio.emit

    def intercept_emit(event: str, data: Any = None, *args: Any, **kwargs: Any) -> Any:
        if event == "Bridge":
            _remember_command(data)
            # The official callback emits Bridge only after publishing the
            # complete response packet. This releases exactly one next slot.
            _response_ready.set()
            return None
        return original_emit(event, data, *args, **kwargs)

    official_bridge.sio.emit = intercept_emit
    _stop_sender.clear()
    _response_ready.set()
    # The official server runs on gevent.  Keep the pacing loop in that same
    # event loop; calling Server.emit from a native thread can leave the
    # Socket.IO connection established while silently dropping the outbound
    # Bridge event that starts telemetry.
    sender = official_bridge.sio.start_background_task(
        _run_command_sender, original_emit, rate_hz)
    print(
        f"[autodrive_bridge_40hz] bounded request pacing enabled at {rate_hz:g} Hz "
        "(max one outstanding request)")
    try:
        official_bridge.main()
    finally:
        _stop_sender.set()
        _response_ready.set()
        sender.join(timeout=max(1.0, 2.0 / rate_hz))
        official_bridge.sio.emit = original_emit


if __name__ == "__main__":
    main()
