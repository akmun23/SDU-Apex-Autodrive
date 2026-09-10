"""Offline LiDAR scan-matching benchmark against recorded simulator truth.

This module deliberately separates the useful estimator experiment from the
runtime localization boundary.  The bag reader uses simulator ground truth
only to score a scan matcher and to put the local odometry seed in the same
map frame.  No ground-truth topic is consumed by AMCL or any controller.

The matcher is a small likelihood-field matcher:

* endpoints from a ``LaserScan`` are transformed through a candidate pose;
* each endpoint is scored by its distance to the nearest occupied map pixel;
* a coarse-to-fine search is run around an odometry-predicted pose.

It is intentionally independent of CUDA and the production AMCL node so it
can answer two questions before runtime tuning:

1. Is the saved map geometrically compatible with the LiDAR observations?
2. If the map is compatible, how much pose error can scan matching recover
   from the odometry error present in a recording?

Run in the Humble workspace container, for example::

  python3 -m sdu_apex_autodrive.odometry_analysis.scan_match_benchmark \
    --bag /workspace/src/f1tenth_planning/analysis/.../rosbag \
    --map /workspace/src/f1tenth_planning/maps/autodrive_track_ftg_commit_20260909_025m.yaml \
    --trajectory /workspace/src/f1tenth_planning/trajectories/..._raceline.csv \
    --output /workspace/src/f1tenth_planning/analysis/scan_match_benchmark

If a map-provenance file is supplied, its recorded ``map -> world`` transform
is used for the absolute comparison.  Otherwise an explicit map-start pose
may be supplied.  The trajectory's first pose remains a clearly labelled
fallback alignment only; it must not be mistaken for the simulator/map
transform.  This avoids the previous invalid assumption that local odometry
``(0, 0)`` is the same as the generated raceline origin.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from sdu_apex_autodrive.map_provenance import load_map_provenance


def angle_diff(a: float, b: float) -> float:
    return math.atan2(math.sin(a - b), math.cos(a - b))


def _read_pgm(path: Path) -> np.ndarray:
    """Read an 8-bit PGM without making Pillow a runtime requirement."""
    with path.open("rb") as stream:
        magic = stream.readline().strip()
        if magic not in (b"P5", b"P2"):
            raise ValueError(f"{path} is not a PGM image")

        tokens: list[bytes] = []
        while len(tokens) < 3:
            line = stream.readline()
            if not line:
                raise ValueError(f"truncated PGM header in {path}")
            line = line.split(b"#", 1)[0]
            tokens.extend(line.split())
        width, height, max_value = (int(token) for token in tokens[:3])
        if width <= 0 or height <= 0 or max_value <= 0 or max_value > 255:
            raise ValueError(f"invalid PGM header in {path}")
        if magic == b"P5":
            data = np.frombuffer(stream.read(width * height), dtype=np.uint8)
        else:
            remaining = stream.read().split()
            data = np.asarray([int(value) for value in remaining], dtype=np.uint8)
        if data.size != width * height:
            raise ValueError(
                f"PGM pixel count mismatch in {path}: {data.size} != {width * height}"
            )
        return data.reshape((height, width))


def _yaml_scalar(text: str, key: str) -> str:
    match = re.search(rf"^\s*{re.escape(key)}\s*:\s*(.+?)\s*$", text, re.MULTILINE)
    if match is None:
        raise ValueError(f"map YAML is missing {key}")
    return match.group(1).split("#", 1)[0].strip()


@dataclass(frozen=True)
class OccupancyMap:
    """Map image plus the ROS map-server world conversion."""

    pixels: np.ndarray
    resolution_m: float
    origin_x_m: float
    origin_y_m: float
    occupied_threshold: float = 0.65
    unknown_value: int = 205

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "OccupancyMap":
        yaml_path = Path(yaml_path).resolve()
        text = yaml_path.read_text(encoding="utf-8")
        image_name = _yaml_scalar(text, "image").strip("'\"")
        image_path = (yaml_path.parent / image_name).resolve()
        resolution = float(_yaml_scalar(text, "resolution"))
        origin_text = _yaml_scalar(text, "origin").strip("[]")
        origin = [float(value.strip()) for value in origin_text.split(",")]
        if len(origin) < 2:
            raise ValueError(f"map origin must contain x and y: {yaml_path}")
        occupied = _yaml_scalar(text, "occupied_thresh")
        return cls(
            pixels=_read_pgm(image_path),
            resolution_m=resolution,
            origin_x_m=origin[0],
            origin_y_m=origin[1],
            occupied_threshold=float(occupied),
        )

    @property
    def height(self) -> int:
        return int(self.pixels.shape[0])

    @property
    def width(self) -> int:
        return int(self.pixels.shape[1])

    @property
    def occupied(self) -> np.ndarray:
        # ROS trinary maps encode occupied cells as low pixel values.  Unknown
        # cells are not treated as walls: a scan endpoint must agree with an
        # actual occupied boundary, not with an unexplored/gray region.
        threshold = int(round(255.0 * (1.0 - self.occupied_threshold)))
        return (self.pixels <= threshold) & (self.pixels != self.unknown_value)

    def world_to_pixel(self, x_m: np.ndarray | float, y_m: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
        col = (np.asarray(x_m, dtype=float) - self.origin_x_m) / self.resolution_m
        row = (self.height - 1) - (
            (np.asarray(y_m, dtype=float) - self.origin_y_m) / self.resolution_m
        )
        return col, row

    def pixel_to_world(self, col: np.ndarray | float, row: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
        x = self.origin_x_m + np.asarray(col, dtype=float) * self.resolution_m
        y = self.origin_y_m + (self.height - 1 - np.asarray(row, dtype=float)) * self.resolution_m
        return x, y

    def distance_field(self) -> np.ndarray:
        """Return distance in metres from each pixel to an occupied pixel."""
        # scipy is available in the planning/ROS image, but keep a small
        # fallback so the pure matcher functions remain usable on a host.
        try:
            from scipy.ndimage import distance_transform_edt

            return distance_transform_edt(~self.occupied) * self.resolution_m
        except ImportError:  # pragma: no cover - exercised only on minimal hosts
            occupied = np.argwhere(self.occupied)
            if occupied.size == 0:
                return np.full(self.pixels.shape, np.inf, dtype=float)
            rows, cols = np.indices(self.pixels.shape)
            best = np.full(self.pixels.shape, np.inf, dtype=float)
            for row, col in occupied:
                best = np.minimum(best, np.hypot(rows - row, cols - col))
            return best * self.resolution_m


@dataclass(frozen=True)
class Scan:
    stamp_s: float
    ranges_m: np.ndarray
    angle_min_rad: float
    angle_increment_rad: float


@dataclass(frozen=True)
class Pose:
    x_m: float
    y_m: float
    yaw_rad: float


def scan_endpoints(
    scan: Scan,
    *,
    laser_offset_x_m: float = 0.2733,
    laser_offset_y_m: float = 0.0,
    min_range_m: float = 0.06,
    max_range_m: float = 10.0,
    max_beams: int = 180,
) -> np.ndarray:
    """Return valid LiDAR endpoints in the base frame."""
    ranges = np.asarray(scan.ranges_m, dtype=float)
    angles = scan.angle_min_rad + np.arange(ranges.size) * scan.angle_increment_rad
    valid = np.isfinite(ranges) & (ranges >= min_range_m) & (ranges <= max_range_m)
    indices = np.flatnonzero(valid)
    if indices.size == 0:
        return np.empty((0, 2), dtype=float)
    if max_beams > 0 and indices.size > max_beams:
        selected = np.linspace(0, indices.size - 1, max_beams).round().astype(int)
        indices = indices[selected]
    ranges = ranges[indices]
    angles = angles[indices]
    return np.column_stack((
        laser_offset_x_m + ranges * np.cos(angles),
        laser_offset_y_m + ranges * np.sin(angles),
    ))


def transform_points(points_base: np.ndarray, pose: Pose) -> np.ndarray:
    c = math.cos(pose.yaw_rad)
    s = math.sin(pose.yaw_rad)
    x = pose.x_m + c * points_base[:, 0] - s * points_base[:, 1]
    y = pose.y_m + s * points_base[:, 0] + c * points_base[:, 1]
    return np.column_stack((x, y))


def score_pose(
    occupancy_map: OccupancyMap,
    distance_field_m: np.ndarray,
    points_base: np.ndarray,
    pose: Pose,
    *,
    sigma_m: float = 0.075,
    out_of_bounds_penalty: float = 0.02,
) -> float:
    """Return a normalized likelihood-field score; larger is better."""
    if points_base.size == 0:
        return -math.inf
    points = transform_points(points_base, pose)
    cols, rows = occupancy_map.world_to_pixel(points[:, 0], points[:, 1])
    col_i = np.rint(cols).astype(int)
    row_i = np.rint(rows).astype(int)
    inside = (
        (col_i >= 0) & (col_i < occupancy_map.width) &
        (row_i >= 0) & (row_i < occupancy_map.height)
    )
    distances = np.full(points.shape[0], 2.0, dtype=float)
    distances[inside] = distance_field_m[row_i[inside], col_i[inside]]
    likelihood = np.exp(-0.5 * np.square(distances / max(sigma_m, 1.0e-6)))
    if not np.all(inside):
        likelihood[~inside] = out_of_bounds_penalty
    # Mean is independent of the number of beams and makes runs comparable.
    return float(np.mean(likelihood))


def _candidate_values(center: float, radius: float, step: float) -> np.ndarray:
    if radius <= 0.0:
        return np.asarray([center], dtype=float)
    count = int(math.floor(radius / step + 1.0e-9))
    return center + np.arange(-count, count + 1, dtype=float) * step


def search_pose(
    occupancy_map: OccupancyMap,
    distance_field_m: np.ndarray,
    points_base: np.ndarray,
    seed: Pose,
    *,
    xy_radius_m: float = 0.50,
    yaw_radius_rad: float = 0.50,
    coarse_xy_step_m: float = 0.10,
    coarse_yaw_step_rad: float = 0.10,
    refine_xy_step_m: float = 0.025,
    refine_yaw_step_rad: float = 0.025,
    sigma_m: float = 0.075,
) -> tuple[Pose, float, float]:
    """Coarse-to-fine local search; return pose, best score and runner-up."""
    if points_base.size == 0:
        return seed, -math.inf, -math.inf

    def evaluate(xs: Iterable[float], ys: Iterable[float], yaws: Iterable[float]) -> tuple[Pose, float, float]:
        ranked: list[tuple[float, Pose]] = []
        for x in xs:
            for y in ys:
                for yaw in yaws:
                    candidate = Pose(float(x), float(y), angle_diff(float(yaw), 0.0))
                    ranked.append((
                        score_pose(occupancy_map, distance_field_m, points_base, candidate, sigma_m=sigma_m),
                        candidate,
                    ))
        ranked.sort(key=lambda item: item[0], reverse=True)
        best_score, best_pose = ranked[0]
        second_score = ranked[1][0] if len(ranked) > 1 else -math.inf
        return best_pose, best_score, second_score

    best, best_score, second_score = evaluate(
        _candidate_values(seed.x_m, xy_radius_m, coarse_xy_step_m),
        _candidate_values(seed.y_m, xy_radius_m, coarse_xy_step_m),
        _candidate_values(seed.yaw_rad, yaw_radius_rad, coarse_yaw_step_rad),
    )
    best, best_score, second_score = evaluate(
        _candidate_values(best.x_m, coarse_xy_step_m, refine_xy_step_m),
        _candidate_values(best.y_m, coarse_xy_step_m, refine_xy_step_m),
        _candidate_values(best.yaw_rad, coarse_yaw_step_rad, refine_yaw_step_rad),
    )
    return best, best_score, second_score


def align_world_to_map(
    world_pose: Pose,
    world_start: Pose,
    map_start: Pose,
) -> Pose:
    """Apply the rigid offline world-to-map alignment from the first pose."""
    delta_yaw = angle_diff(map_start.yaw_rad, world_start.yaw_rad)
    dx = world_pose.x_m - world_start.x_m
    dy = world_pose.y_m - world_start.y_m
    c = math.cos(delta_yaw)
    s = math.sin(delta_yaw)
    return Pose(
        map_start.x_m + c * dx - s * dy,
        map_start.y_m + s * dx + c * dy,
        angle_diff(map_start.yaw_rad + angle_diff(world_pose.yaw_rad, world_start.yaw_rad), 0.0),
    )


def odom_seed_in_map(local_pose: Pose, local_start: Pose, map_start: Pose) -> Pose:
    """Put a local odometry pose in the aligned map frame."""
    return align_world_to_map(local_pose, local_start, map_start)


def _trajectory_start(path: Path) -> Pose:
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.reader(stream):
            if not row or row[0].startswith("#"):
                continue
            if len(row) < 4:
                raise ValueError(f"trajectory row has fewer than four columns: {path}")
            return Pose(float(row[1]), float(row[2]), float(row[3]))
    raise ValueError(f"trajectory has no data rows: {path}")


def _quaternion_yaw(q: object) -> float:
    # Avoid a tf_transformations dependency in the pure analysis module.
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


def _read_bag(path: Path) -> tuple[
    list[Scan], list[tuple[float, Pose]], list[tuple[float, Pose]], list[tuple[float, int]]
]:
    """Read scans, simulator truth and local odometry from a ROS 2 bag."""
    try:
        import rosbag2_py
        from nav_msgs.msg import Odometry
        from rclpy.serialization import deserialize_message
        from sensor_msgs.msg import LaserScan
        from std_msgs.msg import Int32
    except ImportError as exc:  # pragma: no cover - container-only entry point
        raise RuntimeError("ROS 2 Humble rosbag2_py is required to read a bag") from exc

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(path), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr", output_serialization_format="cdr"
        ),
    )
    types = {topic.name: topic.type for topic in reader.get_all_topics_and_types()}
    scans: list[Scan] = []
    truth: list[tuple[float, Pose]] = []
    local_odom: list[tuple[float, Pose]] = []
    collisions: list[tuple[float, int]] = []
    while reader.has_next():
        topic, raw, bag_ns = reader.read_next()
        stamp_s = bag_ns / 1.0e9
        msg_type = types[topic]
        if msg_type == "sensor_msgs/msg/LaserScan":
            msg = deserialize_message(raw, LaserScan)
            stamp_s = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) / 1.0e9
            scans.append(Scan(
                stamp_s, np.asarray(msg.ranges, dtype=float),
                float(msg.angle_min), float(msg.angle_increment)))
        elif msg_type == "nav_msgs/msg/Odometry":
            msg = deserialize_message(raw, Odometry)
            q = msg.pose.pose.orientation
            pose = Pose(
                float(msg.pose.pose.position.x),
                float(msg.pose.pose.position.y),
                _quaternion_yaw(q),
            )
            if topic.endswith("/roboracer_1/odom"):
                truth.append((stamp_s, pose))
            elif topic == "/odom":
                local_odom.append((stamp_s, pose))
        elif msg_type == "std_msgs/msg/Int32" and topic.endswith("/collision_count"):
            msg = deserialize_message(raw, Int32)
            collisions.append((stamp_s, int(msg.data)))
    scans.sort(key=lambda item: item.stamp_s)
    truth.sort(key=lambda item: item[0])
    local_odom.sort(key=lambda item: item[0])
    collisions.sort(key=lambda item: item[0])
    return scans, truth, local_odom, collisions


def _nearest_pose(poses: Sequence[tuple[float, Pose]], stamp_s: float) -> Pose | None:
    if not poses:
        return None
    stamps = [item[0] for item in poses]
    index = int(np.searchsorted(np.asarray(stamps), stamp_s))
    candidates = []
    if index < len(poses):
        candidates.append((abs(stamps[index] - stamp_s), poses[index][1]))
    if index:
        candidates.append((abs(stamps[index - 1] - stamp_s), poses[index - 1][1]))
    return min(candidates, key=lambda item: item[0])[1]


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_benchmark(
    bag_path: str | Path,
    map_path: str | Path,
    trajectory_path: str | Path | None,
    output_dir: str | Path,
    *,
    stride: int = 4,
    max_beams: int = 120,
    search_radius_m: float = 0.50,
    search_yaw_radius_rad: float = 0.50,
    provenance_path: str | Path | None = None,
    explicit_map_start: Pose | None = None,
) -> dict[str, object]:
    """Run the benchmark and write CSV/JSON artifacts."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    occupancy_map = OccupancyMap.from_yaml(map_path)
    distance_field = occupancy_map.distance_field()
    scans, truth, local_odom, collisions = _read_bag(Path(bag_path))
    if not scans or not truth or not local_odom:
        raise ValueError(
            f"bag needs scans, simulator truth and /odom; got {len(scans)}, {len(truth)}, {len(local_odom)}"
        )
    truth_start = truth[0][1]
    odom_start = local_odom[0][1]
    provenance = None
    alignment_mode = ""
    if provenance_path is not None:
        provenance = load_map_provenance(str(provenance_path))
        # Hash validation is intentionally opt-in through hashes in the
        # provenance document.  It still catches stale map artifacts when the
        # mapping run recorded hashes, without rejecting older provenance
        # files that predate hash recording.
        provenance.verify_files()
        map_start = Pose(*provenance.world_to_map(
            truth_start.x_m, truth_start.y_m, truth_start.yaw_rad))
        alignment_mode = "map_provenance"
    elif explicit_map_start is not None:
        map_start = explicit_map_start
        alignment_mode = "explicit_map_start"
    else:
        if trajectory_path is None:
            raise ValueError(
                "provide --provenance or --map-start-pose; trajectory fallback "
                "is unavailable without --trajectory")
        map_start = _trajectory_start(Path(trajectory_path))
        alignment_mode = "trajectory_start_fallback"
    truth_stamps = np.asarray([stamp for stamp, _ in truth], dtype=float)
    odom_stamps = np.asarray([stamp for stamp, _ in local_odom], dtype=float)
    rows: list[dict[str, object]] = []
    matcher_errors: list[float] = []
    seed_errors: list[float] = []
    first_collision_stamp = next(
        (stamp for stamp, count in collisions if count > 0), None)
    scored_scans = [
        scan for scan in scans
        if first_collision_stamp is None or scan.stamp_s < first_collision_stamp
    ]
    for scan_index, scan in enumerate(scored_scans[::max(1, stride)]):
        truth_index = int(np.searchsorted(truth_stamps, scan.stamp_s))
        truth_index = min(max(truth_index, 0), len(truth) - 1)
        if truth_index and abs(truth_stamps[truth_index - 1] - scan.stamp_s) < abs(truth_stamps[truth_index] - scan.stamp_s):
            truth_index -= 1
        odom_index = int(np.searchsorted(odom_stamps, scan.stamp_s))
        odom_index = min(max(odom_index, 0), len(local_odom) - 1)
        if odom_index and abs(odom_stamps[odom_index - 1] - scan.stamp_s) < abs(odom_stamps[odom_index] - scan.stamp_s):
            odom_index -= 1
        gt_world = truth[truth_index][1]
        local_pose = local_odom[odom_index][1]
        if provenance is not None:
            gt_map_values = provenance.world_to_map(
                gt_world.x_m, gt_world.y_m, gt_world.yaw_rad)
            gt_map = Pose(*gt_map_values)
        else:
            gt_map = align_world_to_map(gt_world, truth_start, map_start)
        seed = odom_seed_in_map(local_pose, odom_start, map_start)
        points = scan_endpoints(scan, max_beams=max_beams)
        matched, best_score, second_score = search_pose(
            occupancy_map, distance_field, points, seed,
            xy_radius_m=search_radius_m,
            yaw_radius_rad=search_yaw_radius_rad,
        )
        seed_error = math.hypot(seed.x_m - gt_map.x_m, seed.y_m - gt_map.y_m)
        match_error = math.hypot(matched.x_m - gt_map.x_m, matched.y_m - gt_map.y_m)
        seed_yaw_error = abs(angle_diff(seed.yaw_rad, gt_map.yaw_rad))
        match_yaw_error = abs(angle_diff(matched.yaw_rad, gt_map.yaw_rad))
        gt_score = score_pose(occupancy_map, distance_field, points, gt_map)
        seed_score = score_pose(occupancy_map, distance_field, points, seed)
        seed_errors.append(seed_error)
        matcher_errors.append(match_error)
        rows.append({
            "scan_index": scan_index * max(1, stride),
            "scan_stamp_s": scan.stamp_s,
            "gt_x_m": gt_map.x_m, "gt_y_m": gt_map.y_m, "gt_yaw_rad": gt_map.yaw_rad,
            "seed_x_m": seed.x_m, "seed_y_m": seed.y_m, "seed_yaw_rad": seed.yaw_rad,
            "match_x_m": matched.x_m, "match_y_m": matched.y_m, "match_yaw_rad": matched.yaw_rad,
            "seed_error_m": seed_error, "match_error_m": match_error,
            "seed_yaw_error_rad": seed_yaw_error, "match_yaw_error_rad": match_yaw_error,
            "match_score": best_score, "score_margin": best_score - second_score,
            "gt_score": gt_score, "seed_score": seed_score,
            "valid_beams": int(points.shape[0]),
        })

    def percentile(values: list[float], fraction: float) -> float:
        return float(np.percentile(values, fraction * 100.0)) if values else math.nan

    summary: dict[str, object] = {
        "bag": str(Path(bag_path).resolve()),
        "map": str(Path(map_path).resolve()),
        "trajectory": (
            str(Path(trajectory_path).resolve())
            if trajectory_path is not None else None
        ),
        "provenance": (
            str(Path(provenance_path).resolve())
            if provenance_path is not None else None
        ),
        "alignment_mode": alignment_mode,
        "map_start_pose": {
            "x_m": map_start.x_m,
            "y_m": map_start.y_m,
            "yaw_rad": map_start.yaw_rad,
        },
        "scans_total": len(scans),
        "first_collision_stamp_s": first_collision_stamp,
        "scans_scored": len(rows),
        "stride": max(1, stride),
        "max_beams": max_beams,
        "seed_error_p50_m": percentile(seed_errors, 0.50),
        "seed_error_p95_m": percentile(seed_errors, 0.95),
        "seed_error_max_m": max(seed_errors, default=math.nan),
        "match_error_p50_m": percentile(matcher_errors, 0.50),
        "match_error_p95_m": percentile(matcher_errors, 0.95),
        "match_error_max_m": max(matcher_errors, default=math.nan),
        "match_yaw_error_p95_rad": percentile(
            [float(row["match_yaw_error_rad"]) for row in rows], 0.95),
        "improved_samples": sum(
            float(row["match_error_m"]) < float(row["seed_error_m"]) for row in rows),
        "map_resolution_m": occupancy_map.resolution_m,
        "map_size_px": [occupancy_map.width, occupancy_map.height],
    }
    _write_csv(output / "scan_match_results.csv", rows)
    (output / "scan_match_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", required=True, type=Path)
    parser.add_argument("--map", required=True, type=Path)
    parser.add_argument(
        "--trajectory", type=Path,
        help="offline raceline fallback only; not a map/world transform",
    )
    parser.add_argument(
        "--provenance", type=Path,
        help="map provenance YAML containing the recorded map->world transform",
    )
    parser.add_argument(
        "--map-start-pose", type=float, nargs=3, metavar=("X", "Y", "YAW"),
        help="explicit map-frame pose of the first truth sample",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--max-beams", type=int, default=120)
    parser.add_argument("--search-radius-m", type=float, default=0.50)
    parser.add_argument("--search-yaw-radius-rad", type=float, default=0.50)
    args = parser.parse_args()
    if args.provenance is not None and args.map_start_pose is not None:
        parser.error("use only one of --provenance and --map-start-pose")
    explicit_map_start = (
        Pose(*args.map_start_pose) if args.map_start_pose is not None else None
    )
    summary = run_benchmark(
        args.bag, args.map, args.trajectory, args.output,
        stride=args.stride,
        max_beams=args.max_beams,
        search_radius_m=args.search_radius_m,
        search_yaw_radius_rad=args.search_yaw_radius_rad,
        provenance_path=args.provenance,
        explicit_map_start=explicit_map_start,
    )
    print(json.dumps(summary, indent=2, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
