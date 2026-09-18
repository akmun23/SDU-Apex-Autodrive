import importlib.util
import csv
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from scipy.ndimage import distance_transform_edt


_PATH = (Path(__file__).resolve().parents[2] /
         "tools/model_id/analyze_amcl_scan_observability.py")
_SPEC = importlib.util.spec_from_file_location("amcl_scan_observability", _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)


def _likelihood_on_vertical_wall():
    model = _MODULE.MapLikelihood.__new__(_MODULE.MapLikelihood)
    model.resolution = 0.05
    model.origin_x = -1.0
    model.origin_y = -2.5
    model.origin_yaw = 0.0
    model.height = 100
    model.width = 100
    model.laser_min = 0.06
    model.laser_max = 10.0
    model.laser_ox = 0.2733
    model.laser_oy = 0.0
    model.max_beams = 270
    model.sigma_hit = 0.05
    model.z_hit = 0.9
    model.z_rand = 0.1
    model.likelihood_scale = 2.0
    occupied = np.zeros((model.height, model.width), dtype=bool)
    # Cell 60's center is world x=2.025 m.
    occupied[:, 60] = True
    model.distance = (distance_transform_edt(~occupied) * model.resolution).astype(
        np.float32)
    return model


def test_configured_likelihood_prefers_pose_aligned_with_observed_wall():
    model = _likelihood_on_vertical_wall()
    angles = np.linspace(-0.3, 0.3, 61)
    ranges = (2.025 - model.laser_ox) / np.cos(angles)
    offsets = np.arange(-0.10, 0.1001, 0.005)
    scores = model.score_many(
        ranges, float(angles[0]), float(angles[1] - angles[0]),
        0.05 + offsets, np.zeros(len(offsets)), 0.0)

    best_offset = float(offsets[int(np.argmax(scores))])

    assert best_offset == pytest.approx(-0.05, abs=0.005)


def test_scan_with_no_valid_ranges_has_no_likelihood_evidence():
    model = _likelihood_on_vertical_wall()
    scores = model.score_many(
        np.full(8, np.inf), -0.2, 0.05,
        np.array([0.0, 0.1]), np.array([0.0, 0.0]), 0.0)

    assert np.all(np.isneginf(scores))


@pytest.mark.parametrize("all_scans_invalid", [False, True])
def test_end_to_end_report_requires_valid_lidar_returns(
        tmp_path, all_scans_invalid):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    height, width = 100, 200
    image = np.full((height, width), 255, dtype=np.uint8)
    image[:, 180] = 0
    Image.fromarray(image).save(tmp_path / "map.pgm")
    map_yaml = tmp_path / "map.yaml"
    map_yaml.write_text(
        "image: map.pgm\nmode: trinary\nresolution: 0.05\n"
        "origin: [-1.0, -2.5, 0.0]\nnegate: 0\n"
        "occupied_thresh: 0.65\nfree_thresh: 0.25\n",
        encoding="utf-8")

    event_fields = (
        "event_index", "arrival_monotonic_ns", "arrival_epoch_ns", "topic",
        "message_type", "header_stamp_ns", "simulation_time_s", "payload_json",
    )
    scan_index_fields = (
        "event_index", "header_stamp_ns", "file_offset_bytes", "byte_length",
        "range_count", "angle_min_rad", "angle_increment_rad", "time_increment_s",
        "scan_time_s", "range_min_m", "range_max_m",
    )
    events = []
    scan_index = []
    ranges_path = run_dir / "lidar_scan_ranges.f32le"
    sidecar_offset = 0
    angles = np.linspace(-0.3, 0.3, 61)
    wall_x = -1.0 + (180 + 0.5) * 0.05
    with ranges_path.open("wb") as range_stream:
        for sample in range(400):
            stamp_ns = str(1_700_000_000_000_000_000 + sample * 25_000_000)
            source_time = sample * 0.025
            truth_x = sample * 0.01875
            common = {
                "simulator_position_x": truth_x,
                "simulator_position_y": 0.0,
                "simulator_orientation_quaternion_x": 0.0,
                "simulator_orientation_quaternion_y": 0.0,
                "simulator_orientation_quaternion_z": 0.0,
                "simulator_orientation_quaternion_w": 1.0,
                "simulator_linear_velocity_x": 0.75,
                "simulator_linear_velocity_y": 0.0,
                "simulation_time_s": source_time,
            }
            events.append({
                "event_index": len(events),
                "arrival_monotonic_ns": sample * 25_000_000,
                "arrival_epoch_ns": sample * 25_000_000,
                "topic": "/autodrive/roboracer_1/bridge_packet_timing",
                "message_type": "String", "header_stamp_ns": "",
                "simulation_time_s": "", "payload_json": json.dumps(common),
            })
            events.append({
                "event_index": len(events),
                "arrival_monotonic_ns": sample * 25_000_000 + 1,
                "arrival_epoch_ns": sample * 25_000_000 + 1,
                "topic": "/autodrive/roboracer_1/imu", "message_type": "Imu",
                "header_stamp_ns": stamp_ns, "simulation_time_s": "",
                "payload_json": "{}",
            })
            scan_event_id = len(events)
            events.append({
                "event_index": scan_event_id,
                "arrival_monotonic_ns": sample * 25_000_000 + 2,
                "arrival_epoch_ns": sample * 25_000_000 + 2,
                "topic": "/autodrive/roboracer_1/lidar",
                "message_type": "LaserScan", "header_stamp_ns": stamp_ns,
                "simulation_time_s": "", "payload_json": "{}",
            })
            scan_ranges = (
                np.full(len(angles), np.inf) if all_scans_invalid else
                (wall_x - truth_x - 0.2733) / np.cos(angles))
            packed = np.asarray(scan_ranges, dtype="<f4").tobytes()
            range_stream.write(packed)
            scan_index.append({
                "event_index": scan_event_id, "header_stamp_ns": stamp_ns,
                "file_offset_bytes": sidecar_offset, "byte_length": len(packed),
                "range_count": len(scan_ranges), "angle_min_rad": angles[0],
                "angle_increment_rad": angles[1] - angles[0],
                "time_increment_s": 0.0, "scan_time_s": 0.025,
                "range_min_m": 0.06, "range_max_m": 10.0,
            })
            sidecar_offset += len(packed)
            events.append({
                "event_index": len(events),
                "arrival_monotonic_ns": sample * 25_000_000 + 3,
                "arrival_epoch_ns": sample * 25_000_000 + 3,
                "topic": "/current_map_pose",
                "message_type": "PoseWithCovarianceStamped",
                "header_stamp_ns": stamp_ns, "simulation_time_s": "",
                "payload_json": json.dumps({
                    "x_m": truth_x + 0.05, "y_m": 0.0, "yaw_rad": 0.0,
                }),
            })
    with (run_dir / "events.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=event_fields)
        writer.writeheader()
        writer.writerows(events)
    with (run_dir / "lidar_scan_index.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=scan_index_fields)
        writer.writeheader()
        writer.writerows(scan_index)

    report = _MODULE.analyze_run(
        run_dir, map_yaml, _MODULE.DEFAULT_AMCL_CONFIG)

    assert report["matched_moving_scans"] == 400
    assert report["source_scan_timing"]["period_p50_s"] == pytest.approx(0.025)
    if all_scans_invalid:
        assert report["scans_with_valid_sampled_returns"] == 0
        assert report["fit_gain_on_train"] is None
        assert report["decision"] == "invalid_no_valid_lidar_returns"
        assert report["holdout_candidate"] is None
    else:
        assert report["fit_gain_on_train"] == pytest.approx(1.0)
        assert report["decision"] == "candidate_warrants_matched_live_ab"
        assert report["holdout_candidate"]["along_abs_m"]["p95"] < 0.005
        assert report["holdout_baseline"]["along_abs_m"]["p95"] == pytest.approx(0.05)
