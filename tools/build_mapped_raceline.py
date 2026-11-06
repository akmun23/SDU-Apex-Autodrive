#!/usr/bin/env python3
"""Build a safe constant-speed initial raceline from a ROS occupancy map.

Example:
  python3 tools/build_mapped_raceline.py \
    --map f1tenth_planning/maps/autodrive_practice_20260924_b.yaml \
    --output-dir /tmp/practice_seed --direction auto

An optional --driven-trace CSV should contain named map-frame x,y columns; a
commented header such as ``# stamp,x,y,yaw,u`` is supported. Headerless x,y
files are accepted, while other headerless multi-column files require
``--trace-cols X_INDEX Y_INDEX``. Odometry is used only to infer travel
direction and does not alter extracted geometry. The output is a constant
speed geometric baseline, not a lap-time optimum.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
MAP_TRACK_DIR = REPO_ROOT / "SDU_Apex_IPOPT_PhysX_MinTime_v2"
if str(MAP_TRACK_DIR) not in sys.path:
    sys.path.insert(0, str(MAP_TRACK_DIR))


STEERING_LIMIT_RAD = 0.523598
STEERING_WHEELBASE_M = 0.324
CSV_HEADER = ("s_m", "x_m", "y_m", "psi_rad", "kappa_radpm", "vx_mps",
              "ax_mps2", "d_left_m", "d_right_m")
ERODED_CENTERLINE_CLEARANCE_M = 0.20


def _trace_xy(path: Path, columns: tuple[int, int] | None = None) -> np.ndarray:
    """Read named CSV x/y columns, including headers commented with ``#``."""
    rows: list[list[str]] = []
    commented_headers: list[list[str]] = []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for line in stream:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                header_text = stripped.lstrip("# ").strip()
                if "," in header_text:
                    commented_headers.append(next(csv.reader([header_text])))
                continue
            rows.extend(csv.reader([line]))
    x_names = ("x", "x_m", "position.x", "pose.position.x", "odom_x")
    y_names = ("y", "y_m", "position.y", "pose.position.y", "odom_y")
    headers = commented_headers + (rows[:1] if rows else [])
    header = None
    indices = None
    for candidate in headers:
        normalized = [value.strip().lower() for value in candidate]
        xi = next((normalized.index(n) for n in x_names if n in normalized), None)
        yi = next((normalized.index(n) for n in y_names if n in normalized), None)
        if xi is not None and yi is not None:
            header, indices = candidate, (xi, yi)
            break
    start = 1 if header is not None and rows and rows[0] == header else 0
    if columns is not None:
        if min(columns) < 0 or columns[0] == columns[1]:
            raise ValueError("--trace-cols requires two distinct nonnegative column indexes")
        indices = columns
    elif indices is None and rows:
        # Headerless 2-column x,y is unambiguous. Timestamped/multi-field data
        # must declare its fields instead of silently reading stamp as x.
        try:
            [float(value.strip()) for value in rows[0]]
        except ValueError:
            start = 1
        else:
            if len(rows[0]) > 2:
                raise ValueError(
                    "Headerless multi-column driven trace is ambiguous; provide "
                    "a named x,y header or --trace-cols X_INDEX Y_INDEX")
            if len(rows[0]) < 2:
                raise ValueError("Driven trace rows need x,y columns")
        indices = (0, 1)
    if not rows:
        raise ValueError(f"Driven trace has no data rows: {path}")

    points_list: list[list[float]] = []
    for row in rows[start:]:
        try:
            assert indices is not None
            point = [float(row[indices[0]].strip()), float(row[indices[1]].strip())]
            if len(point) == 2 and np.all(np.isfinite(point)):
                points_list.append(point)
        except (ValueError, IndexError):
            continue
    points = np.asarray(points_list, dtype=float)
    if len(points) < 4:
        raise ValueError(f"Driven trace needs at least 4 finite x,y rows: {path}")
    # Drop stationary duplicates, which otherwise make progress direction noisy.
    keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1.e-4]
    points = points[keep]
    if len(points) < 4:
        raise ValueError(f"Driven trace has fewer than 4 distinct poses: {path}")
    return points


def _resample_trace(trace: np.ndarray, spacing: float = .1) -> np.ndarray:
    segment = np.linalg.norm(np.diff(trace, axis=0), axis=1)
    distance = np.r_[0., np.cumsum(segment)]
    keep = np.r_[True, np.diff(distance) > 1.e-8]
    distance, trace = distance[keep], trace[keep]
    if len(distance) < 2 or distance[-1] < .3:
        raise ValueError("Driven trace has insufficient traveled distance to infer direction")
    query = np.arange(0., distance[-1], spacing)
    if not len(query) or distance[-1] - query[-1] > .02 * spacing:
        query = np.r_[query, distance[-1]]
    return np.column_stack((np.interp(query, distance, trace[:, 0]),
                            np.interp(query, distance, trace[:, 1])))


def _infer_trace_direction(track: Track, trace: np.ndarray) -> tuple[str, dict]:
    trace = _resample_trace(trace)
    projections = []
    previous_s = None
    for x, y in trace:
        projection = track.project(float(x), float(y), previous_s=previous_s)
        if projection is not None:
            previous_s = float(projection[0])
            projections.append(previous_s)
    if len(projections) < 4:
        raise ValueError("Driven trace does not project reliably onto the extracted loop")
    progress = np.diff(projections)
    # Trace was spatially resampled, so this cutoff removes projection jitter
    # while retaining ordinary 40 Hz odometry motion after distance resampling.
    moving = progress[np.abs(progress) >= .035]
    if len(moving) < 3 or abs(float(np.sum(moving))) < .02 * track.length:
        raise ValueError("Driven trace is too short or directionally ambiguous")
    forward = float(np.sum(np.maximum(moving, 0.)))
    backward = float(np.sum(np.maximum(-moving, 0.)))
    current = _signed_direction(track.xy)
    confidence = float(max(forward, backward) / max(forward + backward, 1.e-12))
    if confidence < .55:
        raise ValueError("Driven trace progress is directionally ambiguous")
    direction = current if forward > backward else (
        "clockwise" if current == "counterclockwise" else "counterclockwise")
    return direction, {
        "trace_points_projected": len(projections),
        "trace_resampling_spacing_m": .1,
        "track_extractor_direction": current,
        "trace_forward_progress_m": forward,
        "trace_backward_progress_m": backward,
        "trace_direction_confidence": confidence,
        "trace_coordinate_note": "sensor odometry treated as approximate map-frame x,y",
    }


def _signed_direction(xy: np.ndarray) -> str:
    x, y = xy[:, 0], xy[:, 1]
    area2 = float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    return "counterclockwise" if area2 >= 0. else "clockwise"


def _dense_footprint_check(grid: MapGrid, track: Track) -> dict:
    # Check poses no farther apart than one third of a map cell. Subtract a
    # motion bound between poses (translation plus corner rotation), in addition
    # to map_track's conservative correction between its footprint probes.
    ds = min(.02, grid.resolution / 3.)
    count = max(1, int(math.ceil(track.length / ds)))
    query_s = np.arange(count, dtype=float) * (track.length / count)
    pose = np.asarray([track.at(float(s)) for s in query_s])
    xy = pose[:, :2]
    heading = pose[:, 2]
    pose_step = track.length / count
    next_heading = np.roll(heading, -1)
    yaw_step = np.abs((next_heading - heading + math.pi) % (2. * math.pi) - math.pi)
    motion_bound = .5 * (pose_step + .5 * np.max(yaw_step))
    clearance = np.asarray([
        grid.footprint_clearance(float(x), float(y), float(yaw),
                                 margin=DEFAULT_WALL_MARGIN_M)
        for (x, y), yaw in zip(xy, heading)
    ]) - motion_bound
    min_clearance = float(np.min(clearance))
    if not math.isfinite(min_clearance) or min_clearance < 0.:
        raise ValueError(
            "rear-axle footprint plus 0.050 m wall margin is unsafe on the "
            f"dense map-resolution sweep (minimum signed clearance {min_clearance:.4f} m)")
    return {
        "rear_axle_footprint_margin_safe": True,
        "footprint_margin_m": DEFAULT_WALL_MARGIN_M,
        "minimum_signed_footprint_clearance_m": min_clearance,
        "footprint_check_pose_spacing_m": pose_step,
        "footprint_check_poses": count,
        "footprint_source": "map_track production rear-axle footprint probes; 0.510 x 0.273 m body",
        "between_pose_motion_bound_m": motion_bound,
    }


def _extract_map_track(grid: MapGrid, image_path: Path, spacing: float):
    """Use map_track's normal extraction, then a wall-eroded skeleton fallback."""
    try:
        return map_to_track(grid, spacing=spacing, repo_root=REPO_ROOT,
                            map_path=image_path), {
            "method": "map_track.map_to_track",
            "eroded_clearance_threshold_m": None,
        }
    except ValueError as primary_error:
        try:
            # Removing wall-adjacent pixels before skeletonization prevents
            # one-cell gaps and wall tips from joining the medial skeleton.
            pixels = map_track_module._skeleton_cycle(
                grid.clearance >= ERODED_CENTERLINE_CLEARANCE_M)
            centerline_xy = grid.pixel_to_world(pixels[:, 0], pixels[:, 1])
            track = map_track_module._validate_initial_track(
                grid, centerline_xy, spacing,
                f"eroded-free-space skeleton ({ERODED_CENTERLINE_CLEARANCE_M:.2f} m)")
            return track, {
                "method": "clearance-eroded free-space skeleton",
                "eroded_clearance_threshold_m": ERODED_CENTERLINE_CLEARANCE_M,
                "map_track_rejection": str(primary_error),
            }
        except (ValueError, map_track_module.nx.NetworkXException) as fallback_error:
            raise ValueError(
                "map_track extraction rejected its raw-free-space cycle and the "
                f"{ERODED_CENTERLINE_CLEARANCE_M:.2f} m eroded skeleton failed: "
                f"{fallback_error}; original: {primary_error}") from fallback_error


def _minimum_curvature_track(grid: MapGrid, image_path: Path, track: Track,
                             spacing: float, speed: float) -> tuple[Track, dict]:
    """Refine an over-steering seed with the existing footprint-bounded helper."""
    side = grid.side_clearances(track.xy, track.heading)
    if np.any(~np.isfinite(side)) or np.any(side <= 0.):
        raise ValueError("Cannot optimize geometry with invalid raw map wall distances")
    with tempfile.TemporaryDirectory(prefix="mapped-raceline-") as temp_name:
        temp_dir = Path(temp_name)
        source = temp_dir / "map_seed.csv"
        output = temp_dir / "minimum_curvature.csv"
        source_rows = np.column_stack((
            track.s, track.xy, track.heading, track.kappa,
            np.full(len(track.s), speed), np.zeros(len(track.s)), side))
        with source.open("w", newline="", encoding="utf-8") as stream:
            stream.write("# " + ",".join(CSV_HEADER) + "\n")
            np.savetxt(stream, source_rows, delimiter=",")

        import min_curvature_seed

        optimization = min_curvature_seed.generate(
            image_path, source, output, spacing=spacing,
            margin=DEFAULT_WALL_MARGIN_M, max_offset=2.5,
            offset_step=0.025, harmonics=16, midline_smoothing_m=0.24,
            preserve_source_geometry=True)
        values = np.loadtxt(output, delimiter=",", comments="#", dtype=float)
        if values.ndim != 2 or values.shape[1] < 9 or len(values) < 12:
            raise ValueError("Minimum-curvature helper returned an invalid trajectory CSV")
        if not np.isfinite(values[:, :9]).all() or np.any(np.diff(values[:, 0]) <= 0.):
            raise ValueError("Minimum-curvature helper returned nonfinite or unordered geometry")
        length = float(optimization["mesh"]["length_m"])
        if not math.isfinite(length) or length <= values[-1, 0]:
            raise ValueError("Minimum-curvature helper returned an invalid closed-loop length")
        refined = Track(values[:, 1:3].copy(), values[:, 0].copy(), length,
                        values[:, 3].copy(), values[:, 4].copy(),
                        map_track_module.cKDTree(values[:, 1:3]))
        length_penalty = float(optimization["length_penalty_per_m2"])
        objective = "minimum curvature within explicit full-footprint-safe offsets"
        if length_penalty > 0.:
            objective += " with a path-length penalty"
        summary = {
            "method": "min_curvature_seed.generate",
            "objective": objective,
            "length_penalty_per_m2": length_penalty,
            "preserve_source_geometry": True,
            "optimizer_status": optimization["offset_solve"]["status"],
            "peak_abs_curvature_inv_m": float(
                optimization["curvature"]["seed_peak_abs_per_m"]),
            "input_peak_abs_curvature_inv_m": float(
                optimization["curvature"]["input_raceline_peak_abs_per_m"]),
            "helper_footprint_check": optimization["map_footprint_check"],
            "helper_mesh": optimization["mesh"],
        }
    return refined, summary


def _save_map_plot(grid: MapGrid, track: Track, path: Path, direction: str) -> None:
    import cv2

    image = cv2.cvtColor(grid.image, cv2.COLOR_GRAY2BGR)
    rc = grid.world_to_pixel(np.vstack([track.xy, track.xy[0]]))
    points = np.rint(rc[:, ::-1]).astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(image, [points], isClosed=True, color=(35, 45, 230),
                  thickness=max(1, int(round(.045 / grid.resolution))),
                  lineType=cv2.LINE_AA)
    start_rc = grid.world_to_pixel(track.xy[0])
    start = tuple(np.rint(start_rc[::-1]).astype(int))
    cv2.circle(image, start, max(2, int(round(.10 / grid.resolution))),
               (230, 100, 25), -1, lineType=cv2.LINE_AA)
    label = f"Initial raceline | {direction} | rear axle"
    font_scale = .65
    thickness = 2
    (text_width, text_height), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    if text_width > image.shape[1] - 20:
        font_scale *= (image.shape[1] - 20) / text_width
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    baseline = text_height + 8
    cv2.putText(image, label, (10, baseline), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (30, 30, 30), thickness + 2, cv2.LINE_AA)
    cv2.putText(image, label, (10, baseline), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not write map plot: {path}")


def build(args: argparse.Namespace) -> dict:
    # Also support importing this reusable stage from the track optimizer.
    global DEFAULT_WALL_MARGIN_M, MapGrid, Track, map_to_track, map_track_module
    import map_track as map_track_module
    from map_track import DEFAULT_WALL_MARGIN_M, MapGrid, Track, map_to_track
    yaml_path = args.map.expanduser().resolve()
    config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    image_path = Path(config["image"])
    if not image_path.is_absolute():
        image_path = yaml_path.parent / image_path
    grid = MapGrid.load(image_path, yaml_path)
    spacing = min(.12, grid.resolution * 2.4)
    track, extraction = _extract_map_track(grid, image_path, spacing)

    direction_evidence: dict = {}
    if args.direction == "auto":
        if args.driven_trace:
            chosen, direction_evidence = _infer_trace_direction(
                track, _trace_xy(
                    args.driven_trace.expanduser().resolve(),
                    tuple(args.trace_cols) if args.trace_cols else None))
            direction_reason = "inferred from projected sensor odometry trace"
        else:
            chosen = _signed_direction(track.xy)
            direction_reason = (
                "no driven trace supplied; retained map extractor order. This is "
                "a deterministic output orientation, not a preferred driving direction")
    else:
        chosen = args.direction
        direction_reason = "explicit --direction selection"
    if _signed_direction(track.xy) != chosen:
        track = Track.from_xy(track.xy[::-1].copy(), spacing=spacing)

    # A footprint-safe geometric seed is required before optimizing within its
    # explicit safe offset corridor. The optimized result is checked again below.
    _dense_footprint_check(grid, track)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    # Keep the extracted centerline separate from its curvature refinement.
    # Other geometry basins must not be seeded from an already wall-hugging
    # minimum-curvature result that has lost its two-sided offset corridor.
    centerline_path = output_dir / "centerline.csv"
    centerline_rows = np.column_stack((
        track.s, track.xy, track.heading, track.kappa,
        np.full(len(track.s), float(args.speed)), np.zeros(len(track.s)),
        grid.side_clearances(track.xy, track.heading)))
    np.savetxt(centerline_path, centerline_rows, delimiter=",",
               fmt="%.12g", header=",".join(CSV_HEADER))
    kappa_limit = math.tan(STEERING_LIMIT_RAD) / STEERING_WHEELBASE_M
    steering_required = np.arctan(STEERING_WHEELBASE_M * track.kappa)
    max_steering = float(np.max(np.abs(steering_required)))
    max_abs_kappa = float(np.max(np.abs(track.kappa)))
    steering_ok = max_steering <= STEERING_LIMIT_RAD + 1.e-9
    if not steering_ok:
        track, optimization = _minimum_curvature_track(
            grid, image_path, track, spacing, float(args.speed))
        if _signed_direction(track.xy) != chosen:
            raise ValueError("Minimum-curvature helper changed the selected loop direction")
        max_abs_kappa = float(np.max(np.abs(track.kappa)))
        max_steering = float(np.max(np.abs(
            np.arctan(STEERING_WHEELBASE_M * track.kappa))))
        steering_ok = max_steering <= STEERING_LIMIT_RAD + 1.e-9
        if not steering_ok:
            raise ValueError(
                f"minimum-curvature geometry still requires {max_steering:.6f} rad "
                f"steering; source limit is {STEERING_LIMIT_RAD:.6f} rad "
                f"(|kappa| <= {kappa_limit:.6f} 1/m)")
    else:
        optimization = None
    clearance_result = _dense_footprint_check(grid, track)

    side = grid.side_clearances(track.xy, track.heading)
    speed = float(args.speed)
    samples = np.column_stack((track.s, track.xy, track.heading, track.kappa,
                               np.full(len(track.s), speed), np.zeros(len(track.s)), side))
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "initial_raceline.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("# " + CSV_HEADER[0], *CSV_HEADER[1:]))
        writer.writerows((f"{v:.9f}" for v in row) for row in samples)
    plot_path = output_dir / "map.png"
    _save_map_plot(grid, track, plot_path, chosen)

    report = {
        "result": "completed",
        "map_yaml": str(yaml_path),
        "map_image": str(image_path.resolve()),
        "raceline_csv": str(csv_path),
        "extracted_centerline_csv": str(centerline_path),
        "map_plot_png": str(plot_path),
        "direction": chosen,
        "direction_reason": direction_reason,
        "direction_evidence": direction_evidence,
        "geometry_extraction": extraction,
        "geometry_optimization": optimization,
        "track_length_m": float(track.length),
        "sample_count": int(len(track.s)),
        "map_resolution_m": float(grid.resolution),
        "speed_mps": speed,
        "speed_profile": "constant requested baseline; not a lap-time optimum",
        "ax_mps2": 0.0,
        "wall_distances": "raw map distances along path normals; no car width or margin subtracted",
        "steering_check": {
            "source_steering_limit_rad": STEERING_LIMIT_RAD,
            "source_steering_wheelbase_m": STEERING_WHEELBASE_M,
            "equivalent_abs_curvature_limit_inv_m": kappa_limit,
            "maximum_abs_curvature_inv_m": max_abs_kappa,
            "maximum_required_bicycle_steering_rad": max_steering,
            "within_source_steering_limit": steering_ok,
        },
        "physical_limits": "No tire, acceleration, friction, or lap-time limits guessed or applied.",
        **clearance_result,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path, required=True,
                        help="ROS map YAML (its image path is resolved relative to it)")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="directory for initial_raceline.csv, map.png, report.json")
    parser.add_argument("--speed", type=float, default=1.2,
                        help="constant baseline speed in m/s (default: 1.2)")
    parser.add_argument("--direction", choices=("auto", "clockwise", "counterclockwise"),
                        default="auto", help="travel direction; auto uses trace if supplied")
    parser.add_argument("--driven-trace", type=Path,
                        help="optional sensor odometry CSV with approximate map-frame x,y")
    parser.add_argument("--trace-cols", nargs=2, type=int,
                        metavar=("X_INDEX", "Y_INDEX"),
                        help="0-based x,y columns for a headerless trace")
    args = parser.parse_args()
    if not math.isfinite(args.speed) or args.speed <= 0.:
        parser.error("--speed must be finite and positive")
    if args.trace_cols and not args.driven_trace:
        parser.error("--trace-cols requires --driven-trace")
    if args.driven_trace and args.direction != "auto":
        parser.error("--driven-trace is only used with --direction auto")
    try:
        # Keep --help usable in lean environments where map-processing
        # dependencies have not been installed.
        global DEFAULT_WALL_MARGIN_M, MapGrid, Track, map_to_track, map_track_module
        import map_track as map_track_module
        from map_track import DEFAULT_WALL_MARGIN_M, MapGrid, Track, map_to_track
        report = build(args)
    except (OSError, ImportError, KeyError, RuntimeError, TypeError,
            ValueError, yaml.YAMLError) as exc:
        print(f"build_mapped_raceline: rejected map: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
