import csv
import math
import struct
from types import SimpleNamespace

from sdu_apex_autodrive.model_id_timing_recorder import (
    EVENT_FIELDS,
    ModelIdTimingRecorder,
    _flush_if_due,
    _pack_lidar_ranges,
    _pose_with_covariance_payload,
)


class _FlushCounter:
    def __init__(self):
        self.count = 0

    def flush(self):
        self.count += 1


def test_pose_event_records_the_covariance_used_by_the_controller_gate():
    pose = SimpleNamespace(
        position=SimpleNamespace(x=1.0, y=2.0),
        orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
    )
    message = SimpleNamespace(
        pose=SimpleNamespace(pose=pose, covariance=[0.0] * 36))
    message.pose.covariance[0] = 0.20
    message.pose.covariance[7] = 0.27
    message.pose.covariance[35] = 0.002

    payload = _pose_with_covariance_payload(message)

    assert payload["covariance_xx_m2"] == 0.20
    assert payload["covariance_yy_m2"] == 0.27
    assert payload["covariance_yawyaw_rad2"] == 0.002
    assert len(payload["covariance_6x6"]) == 36


def test_event_stream_flushes_only_when_interval_is_due():
    stream = _FlushCounter()

    last_flush = _flush_if_due(stream, 100, 0, 250)
    assert last_flush == 0
    assert stream.count == 0

    last_flush = _flush_if_due(stream, 249, last_flush, 250)
    assert last_flush == 0
    assert stream.count == 0

    last_flush = _flush_if_due(stream, 250, last_flush, 250)
    assert last_flush == 250
    assert stream.count == 1


def test_optional_lidar_range_sidecar_is_compact_and_preserves_invalid_ranges():
    values = (0.5, 2.0, float("nan"), float("inf"), float("-inf"))

    packed = _pack_lidar_ranges(values)
    decoded = struct.unpack("<5f", packed)

    assert len(packed) == 5 * 4
    assert decoded[:2] == values[:2]
    assert math.isnan(decoded[2])
    assert decoded[3] == float("inf")
    assert decoded[4] == float("-inf")


def test_event_partitions_are_streamed_from_canonical_event_csv(tmp_path):
    event_path = tmp_path / "events.csv"
    events = [
        {"event_index": "0", "arrival_monotonic_ns": "100",
         "arrival_epoch_ns": "1000", "topic":
         "/autodrive/roboracer_1/imu", "message_type": "Imu",
         "header_stamp_ns": "1000", "simulation_time_s": "",
         "payload_json": '{"yaw_rate_radps":0.25}'},
        {"event_index": "1", "arrival_monotonic_ns": "110",
         "arrival_epoch_ns": "1010", "topic":
         "/autodrive/roboracer_1/left_encoder", "message_type": "JointState",
         "header_stamp_ns": "1000", "simulation_time_s": "",
         "payload_json": '{"position":[1.0]}'},
        {"event_index": "2", "arrival_monotonic_ns": "120",
         "arrival_epoch_ns": "1020", "topic":
         "/current_map_pose", "message_type": "PoseWithCovarianceStamped",
         "header_stamp_ns": "1000", "simulation_time_s": "",
         "payload_json": '{"x_m":2.0}'},
        {"event_index": "3", "arrival_monotonic_ns": "130",
         "arrival_epoch_ns": "1030", "topic":
         "/cmd/speed", "message_type": "AckermannDriveStamped",
         "header_stamp_ns": "1000", "simulation_time_s": "",
         "payload_json": '{"speed_mps":4.0}'},
        {"event_index": "4", "arrival_monotonic_ns": "140",
         "arrival_epoch_ns": "1040", "topic":
         "/pure_pursuit/diagnostics", "message_type": "Float64MultiArray",
         "header_stamp_ns": "", "simulation_time_s": "",
         "payload_json": '{"data":[1.0,2.0]}'},
        {"event_index": "5", "arrival_monotonic_ns": "150",
         "arrival_epoch_ns": "1050", "topic":
         "/autodrive/roboracer_1/throttle_command", "message_type": "Float32",
         "header_stamp_ns": "", "simulation_time_s": "",
         "payload_json": '{"value":0.0}'},
        {"event_index": "6", "arrival_monotonic_ns": "160",
         "arrival_epoch_ns": "1060", "topic":
         "/autodrive/roboracer_1/steering_command", "message_type": "Float32",
         "header_stamp_ns": "", "simulation_time_s": "",
         "payload_json": '{"value":0.25}'},
    ]
    with event_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=EVENT_FIELDS)
        writer.writeheader()
        writer.writerows(events)

    ModelIdTimingRecorder._write_event_partitions(event_path, tmp_path)

    with (tmp_path / "imu.csv").open(newline="", encoding="utf-8") as stream:
        imu_rows = list(csv.DictReader(stream))
    with (tmp_path / "encoders.csv").open(newline="", encoding="utf-8") as stream:
        encoder_rows = list(csv.DictReader(stream))
    with (tmp_path / "runtime_state.csv").open(newline="", encoding="utf-8") as stream:
        state_rows = list(csv.DictReader(stream))
    with (tmp_path / "controller_trace.csv").open(
            newline="", encoding="utf-8") as stream:
        controller_rows = list(csv.DictReader(stream))
    with (tmp_path / "actuator_commands.csv").open(
            newline="", encoding="utf-8") as stream:
        actuator_command_rows = list(csv.DictReader(stream))

    assert [row["event_index"] for row in imu_rows] == ["0"]
    assert [row["event_index"] for row in encoder_rows] == ["1"]
    assert [row["event_index"] for row in state_rows] == ["2"]
    assert [row["event_index"] for row in controller_rows] == ["3", "4"]
    assert [row["event_index"] for row in actuator_command_rows] == ["5", "6"]
