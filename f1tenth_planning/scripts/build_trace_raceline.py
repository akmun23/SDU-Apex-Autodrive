#!/usr/bin/env python3
"""Build a map-frame raceline from a safe, sensor-odometry trace.

The trace is recorded from the competition-available encoder/IMU odometry
while FTG drives the circuit.  The simulator pose is deliberately not an
input to this tool.  A rigid 2-D fit places the trace in the saved SLAM map;
the result is then resampled and differentiated for Pure Pursuit/MPC.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize_scalar
from scipy.spatial import cKDTree


def load_xy(path: Path) -> np.ndarray:
    rows = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                x = float(row["odom_x"])
                y = float(row["odom_y"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(x) and math.isfinite(y):
                rows.append((x, y))
    points = np.asarray(rows, dtype=float)
    if len(points) < 10:
        raise ValueError(f"{path} does not contain enough odom samples")
    return points


def load_reference(path: Path) -> np.ndarray:
    points = []
    with path.open(newline="") as handle:
        for row in csv.reader(handle):
            if not row or row[0].startswith("#"):
                continue
            try:
                points.append((float(row[1]), float(row[2])))
            except (IndexError, TypeError, ValueError):
                continue
    result = np.asarray(points, dtype=float)
    if len(result) > 2 and np.linalg.norm(result[0] - result[-1]) < 1.0e-4:
        result = result[:-1]
    if len(result) < 10:
        raise ValueError(f"{path} does not contain enough reference points")
    return result


def rigid_rotation(angle: float) -> np.ndarray:
    return np.asarray(
        [[math.cos(angle), -math.sin(angle)],
         [math.sin(angle), math.cos(angle)]],
        dtype=float,
    )


def fit_trace_to_reference(trace: np.ndarray, reference: np.ndarray) -> tuple[float, np.ndarray, float]:
    """Fit a rigid transform to the reference point set with robust ICP."""
    tree = cKDTree(reference)

    def score(angle: float) -> tuple[float, np.ndarray]:
        rotated = trace @ rigid_rotation(angle).T
        translation = reference.mean(axis=0) - rotated.mean(axis=0)
        for _ in range(8):
            distances, indices = tree.query(rotated + translation, k=1)
            keep = distances <= np.quantile(distances, 0.85)
            translation = (reference[indices[keep]] - rotated[keep]).mean(axis=0)
        distances, _ = tree.query(rotated + translation, k=1)
        robust = np.minimum(distances, 1.0)
        return float(np.sqrt(np.mean(robust * robust))), translation

    candidates = []
    for angle in np.linspace(-math.pi, math.pi, 145, endpoint=False):
        value, translation = score(float(angle))
        candidates.append((value, float(angle), translation))
    _, best_angle, _ = min(candidates, key=lambda item: item[0])
    refined = minimize_scalar(
        lambda angle: score(float(angle))[0],
        bounds=(best_angle - 0.10, best_angle + 0.10),
        method="bounded",
        options={"xatol": 1.0e-5},
    )
    final_score, translation = score(float(refined.x))
    return float(refined.x), translation, final_score


def close_and_resample(points: np.ndarray, spacing: float) -> np.ndarray:
    """Resample an open sample sequence as a closed polyline."""
    if np.linalg.norm(points[0] - points[-1]) < 1.0e-5:
        points = points[:-1]
    next_points = np.roll(points, -1, axis=0)
    segment_lengths = np.linalg.norm(next_points - points, axis=1)
    if np.any(segment_lengths < 1.0e-5):
        keep = segment_lengths >= 1.0e-5
        points = points[keep]
        next_points = np.roll(points, -1, axis=0)
        segment_lengths = np.linalg.norm(next_points - points, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total = float(cumulative[-1])
    count = max(20, int(round(total / spacing)))
    samples = np.arange(count, dtype=float) * total / count
    output = np.empty((count, 2), dtype=float)
    for index, distance in enumerate(samples):
        segment = min(int(np.searchsorted(cumulative, distance, side="right") - 1), len(points) - 1)
        local = distance - cumulative[segment]
        ratio = local / segment_lengths[segment]
        output[index] = points[segment] + ratio * (next_points[segment] - points[segment])
    return output


def raycast_wall(map_img: np.ndarray, origin: tuple[float, float], resolution: float,
                 x: float, y: float, angle: float, max_range: float = 5.0) -> float:
    """Return distance to an occupied pixel, or max_range outside known walls."""
    height, width = map_img.shape[:2]
    distance = 0.0
    while distance < max_range:
        wx = x + distance * math.cos(angle)
        wy = y + distance * math.sin(angle)
        col = int((wx - origin[0]) / resolution)
        row = height - 1 - int((wy - origin[1]) / resolution)
        if col < 0 or col >= width or row < 0 or row >= height:
            return distance
        if map_img[row, col] < 65:
            return distance
        distance += resolution * 0.25
    return max_range


def write_raceline(points: np.ndarray, map_path: Path, output: Path,
                   spacing: float, speed: float,
                   start_position: np.ndarray | None = None) -> None:
    data = yaml.safe_load(map_path.with_suffix(".yaml").read_text())
    image_path = map_path
    if map_path.suffix.lower() == ".yaml":
        image_path = map_path.parent / data["image"]
    map_img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if map_img is None:
        raise ValueError(f"could not read map image {image_path}")
    origin = (float(data["origin"][0]), float(data["origin"][1]))
    resolution = float(data["resolution"])

    sampled = close_and_resample(points, spacing)
    if start_position is not None:
        # Make the known vehicle start the first waypoint.  This is important
        # on a repetitive track: a single startup scan can match several
        # straight sections, so global AMCL must not be the default race-start
        # prior.
        start_index = int(np.argmin(np.linalg.norm(sampled - start_position, axis=1)))
        sampled = np.roll(sampled, -start_index, axis=0)
    # The raw FTG trace already contains the physical route.  This very small
    # periodic filter removes encoder/scan scheduling jitter without moving the
    # route by more than a few centimetres.
    smoothed = np.column_stack(
        [gaussian_filter1d(sampled[:, axis], sigma=1.0, mode="wrap") for axis in range(2)]
    )
    dx = (np.roll(smoothed[:, 0], -1) - np.roll(smoothed[:, 0], 1)) / (2.0 * spacing)
    dy = (np.roll(smoothed[:, 1], -1) - np.roll(smoothed[:, 1], 1)) / (2.0 * spacing)
    ddx = (np.roll(smoothed[:, 0], -1) - 2.0 * smoothed[:, 0] + np.roll(smoothed[:, 0], 1)) / (spacing * spacing)
    ddy = (np.roll(smoothed[:, 1], -1) - 2.0 * smoothed[:, 1] + np.roll(smoothed[:, 1], 1)) / (spacing * spacing)
    tangent_speed = np.maximum(np.hypot(dx, dy), 1.0e-6)
    headings = np.arctan2(dy, dx)
    curvature = (dx * ddy - dy * ddx) / np.maximum(tangent_speed ** 3, 1.0e-6)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["# s_m,x_m,y_m,psi_rad,kappa_radpm,vx_mps,ax_mps2,d_left_m,d_right_m"])
        for index, (point, heading, kappa) in enumerate(zip(smoothed, headings, curvature)):
            left_angle = heading + math.pi / 2.0
            right_angle = heading - math.pi / 2.0
            d_left = raycast_wall(map_img, origin, resolution, point[0], point[1], left_angle)
            d_right = raycast_wall(map_img, origin, resolution, point[0], point[1], right_angle)
            writer.writerow([
                f"{index * spacing:.6f}", f"{point[0]:.6f}", f"{point[1]:.6f}",
                f"{heading:.6f}", f"{kappa:.6f}", f"{speed:.6f}", "0.000000",
                f"{d_left:.6f}", f"{d_right:.6f}",
            ])

    print(f"Wrote {len(smoothed)} waypoints to {output}")
    print(f"  length={len(smoothed) * spacing:.2f} m")
    print(f"  curvature=[{curvature.min():.3f}, {curvature.max():.3f}] rad/m")
    print(f"  map-frame bounds x=[{smoothed[:, 0].min():.2f}, {smoothed[:, 0].max():.2f}] "
          f"y=[{smoothed[:, 1].min():.2f}, {smoothed[:, 1].max():.2f}]")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense-trace", required=True, type=Path)
    parser.add_argument("--tail-trace", required=True, type=Path)
    parser.add_argument("--tail-start-index", required=True, type=int)
    parser.add_argument("--tail-end-index", required=True, type=int)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--map", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--spacing", type=float, default=0.05)
    parser.add_argument("--speed", type=float, default=0.25)
    args = parser.parse_args()

    dense = load_xy(args.dense_trace)
    tail = load_xy(args.tail_trace)
    if not 0 <= args.tail_start_index < args.tail_end_index < len(tail):
        raise ValueError("tail indices must select an ordered range in the tail trace")
    trace = np.vstack((dense, tail[args.tail_start_index:args.tail_end_index + 1]))
    reference = load_reference(args.reference)
    angle, translation, fit_error = fit_trace_to_reference(trace, reference)
    transformed = trace @ rigid_rotation(angle).T + translation
    print(f"Rigid map fit: yaw={angle:.4f} rad translation=({translation[0]:.3f}, {translation[1]:.3f}) "
          f"robust_error={fit_error:.3f} m")
    write_raceline(
        transformed,
        args.map,
        args.output,
        args.spacing,
        args.speed,
        start_position=translation,
    )


if __name__ == "__main__":
    main()
