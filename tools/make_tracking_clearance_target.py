#!/usr/bin/env python3
"""Make a separate tracking-reserve target; never certify vehicle dynamics.

Preserve the input speed samples unless explicitly scaled, minimally move geometry away from tight
walls, recompute geometric headings/curvature/corridors, and certify swept
map footprint clearance. The result needs closed-loop simulator validation;
the input OCP's dynamic feasibility certificate does NOT transfer to it.
Run with the existing MinTime .venv (map-processing dependencies).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import diags, lil_matrix
from scipy.sparse.linalg import spsolve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def smooth_offsets(left, right, required, spacing, smoothing_length, max_offset):
    """Convex periodic least-displacement/curvature objective, boxed by walls."""
    left, right = np.asarray(left), np.asarray(right)
    if (left.ndim != 1 or len(left) < 3 or right.shape != left.shape or
            not np.isfinite(left).all() or not np.isfinite(right).all() or
            min(required, spacing, smoothing_length, max_offset) <= 0):
        raise ValueError("invalid corridor/smoothing inputs")
    lower = np.maximum(required - right, -max_offset)
    upper = np.minimum(left - required, max_offset)
    if np.any(lower > upper):
        raise ValueError("tracking reserve cannot fit in the corridor/trust bound")
    initial = np.clip(np.zeros_like(left), lower, upper)
    if not np.any(initial):
        return initial
    gain = (smoothing_length / spacing) ** 4

    def cost_and_gradient(offset):
        second = np.roll(offset, 1) - 2 * offset + np.roll(offset, -1)
        fourth = np.roll(second, 1) - 2 * second + np.roll(second, -1)
        return (float(offset @ offset + gain * (second @ second)),
                2 * offset + 2 * gain * fourth)

    result = minimize(cost_and_gradient, initial, jac=True, method="L-BFGS-B",
        bounds=list(zip(lower, upper)),
        options={"maxiter": 5000, "maxls": 40, "ftol": 1.e-13, "gtol": 1.e-7})
    if not result.success:
        raise ValueError(f"clearance smoothing failed: {result.message}")
    return result.x


def path_geometry(xy):
    before = xy - np.roll(xy, 1, axis=0)
    after = np.roll(xy, -1, axis=0) - xy
    segments = np.linalg.norm(after, axis=1)
    if np.any(segments < 1.e-7):
        raise ValueError("degenerate target geometry")
    chord = before + after
    heading = np.unwrap(np.arctan2(chord[:, 1], chord[:, 0]))
    curvature = 2 * (before[:, 0] * after[:, 1] - before[:, 1] * after[:, 0]) / (
        np.linalg.norm(before, axis=1) * segments * np.linalg.norm(chord, axis=1))
    return segments, heading, curvature


def smooth_geometry(xy, smoothing_length, max_displacement):
    """Periodic, arc-weighted geometry regularization, not vehicle fitting.

    Minimize integral |p-p0|^2 + length^4 |d2p/ds2|^2 on the original
    nonuniform arc grid. The sparse SPD system is solved for both coordinates
    together. Never clip individual points: that would recreate curvature
    spikes. A downstream swept-footprint check remains mandatory.
    """
    xy = np.asarray(xy, dtype=float)
    if (xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 3 or
            not np.isfinite(xy).all() or not math.isfinite(smoothing_length) or
            smoothing_length < 0 or not math.isfinite(max_displacement) or
            max_displacement <= 0):
        raise ValueError("invalid geometry smoothing inputs")
    segments, _, _ = path_geometry(xy)
    if smoothing_length == 0:
        return xy.copy()
    previous = np.roll(segments, 1)
    weights = .5 * (previous + segments)
    second = lil_matrix((len(xy), len(xy)))
    for i in range(len(xy)):
        factor = 1. / weights[i]
        second[i, (i - 1) % len(xy)] = factor / previous[i]
        second[i, i] = -factor * (1. / previous[i] + 1. / segments[i])
        second[i, (i + 1) % len(xy)] = factor / segments[i]
    second = second.tocsc()
    weight = diags(weights).tocsc()
    system = weight + smoothing_length**4 * (second.T @ weight @ second)
    # Center before solving so large map origins do not amplify cancellation.
    origin = np.average(xy, axis=0, weights=weights)
    result = spsolve(system, weight @ (xy - origin)) + origin
    if not np.isfinite(result).all():
        raise ValueError("geometry smoothing produced nonfinite coordinates")
    displacement = np.linalg.norm(result - xy, axis=1)
    if float(np.max(displacement)) > max_displacement:
        raise ValueError("geometry smoothing exceeds displacement bound")
    return result


def corridor_widths(grid, xy, heading):
    distances = np.arange(0., 3.01, max(.0125, .5 * grid.resolution))
    normal = np.column_stack((-np.sin(heading), np.cos(heading)))
    widths = []
    for sign in (1, -1):
        points = xy[:, None, :] + sign * distances[None, :, None] * normal[:, None, :]
        occupied = grid.distance(points) < .55 * grid.resolution
        first = np.argmax(occupied, axis=1)
        widths.append(np.where(occupied.any(axis=1), distances[first], distances[-1]))
    return np.column_stack(widths)


def target_speed_profile(speed, segments, scale=1.):
    """Scale target speed and its geometric acceleration consistently.

    This is a requested profile, never a source-plant feasibility certificate.
    For fixed geometry acceleration scales quadratically with speed scale.
    """
    speed, segments = np.asarray(speed, dtype=float), np.asarray(segments, dtype=float)
    if (speed.ndim != 1 or speed.shape != segments.shape or len(speed) < 3 or
            not np.isfinite(speed).all() or not np.isfinite(segments).all() or
            np.any(speed <= 0) or np.any(segments <= 0) or
            not math.isfinite(scale) or scale <= 0):
        raise ValueError("finite positive speeds, segments and scale required")
    u = speed * scale
    acceleration = u * (np.roll(u, -1) - np.roll(u, 1)) / (segments + np.roll(segments, 1))
    return u, acceleration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--tracking-reserve", type=float, default=.05)
    parser.add_argument("--wall-margin", type=float, default=.05)
    parser.add_argument("--smoothing-length", type=float, default=.6)
    parser.add_argument("--max-offset", type=float, default=.10)
    parser.add_argument("--geometry-smoothing-length", type=float, default=0.,
        help="optional periodic XY regularization length in metres; zero preserves input")
    parser.add_argument("--max-geometry-displacement", type=float, default=.02)
    parser.add_argument("--speed-scale", type=float, default=1.,
        help="scale requested speed only; dynamic feasibility never transfers")
    args = parser.parse_args()
    if (not all(math.isfinite(v) for v in (args.tracking_reserve,
            args.wall_margin, args.smoothing_length, args.max_offset,
            args.geometry_smoothing_length, args.max_geometry_displacement,
            args.speed_scale)) or
            args.tracking_reserve < 0 or args.wall_margin < 0 or
            args.smoothing_length <= 0 or args.max_offset <= 0 or
            args.geometry_smoothing_length < 0 or args.max_geometry_displacement <= 0 or
            args.speed_scale <= 0):
        parser.error("finite nonnegative margins and positive smoothing/trust bounds required")
    if args.output.exists() or args.output.with_suffix('.json').exists():
        parser.error("refusing to overwrite an existing target/report")
    from SDU_Apex_IPOPT_PhysX_MinTime_v2.map_track import (
        MapGrid, _FOOTPRINT_RADIUS, _EDGE_ALONG, _EDGE_SIDE)
    from tools.vehicle_footprint import BODY_X, BODY_Y, WIDTH_M
    if not (np.array_equal(BODY_X, _EDGE_ALONG) and np.array_equal(BODY_Y, _EDGE_SIDE)):
        raise ValueError("map certifier and production footprint disagree")

    values = np.loadtxt(args.input, delimiter=',')
    if (values.ndim != 2 or values.shape[1] != 9 or len(values) < 3 or
            not np.isfinite(values).all() or np.any(values[:, 5] <= 0)):
        parser.error("expected finite nine-column positive-speed runtime target")
    grid = MapGrid.load(args.map)
    # This is an additional tracking allowance, not a relaxed physical wall
    # margin. Footprint dimensions come from the production controller header.
    required = .5 * WIDTH_M + args.wall_margin + args.tracking_reserve
    original = values[:, 1:3].copy()
    xy = smooth_geometry(original, args.geometry_smoothing_length,
                         args.max_geometry_displacement)
    geometry_displacement = float(np.max(np.linalg.norm(xy - original, axis=1)))
    if geometry_displacement > args.max_offset + 1.e-8:
        raise ValueError("geometry smoothing exceeds input-path trust bound")
    for passes in range(1, 9):
        segments, heading, curvature = path_geometry(xy)
        widths = corridor_widths(grid, xy, heading)
        if np.min(widths) >= required:
            break
        offset = smooth_offsets(widths[:, 0], widths[:, 1], required,
            float(np.mean(segments)), args.smoothing_length, args.max_offset)
        xy += offset[:, None] * np.column_stack((-np.sin(heading), np.cos(heading)))
        if np.max(np.linalg.norm(xy - original, axis=1)) > args.max_offset + 1.e-8:
            raise ValueError("cumulative displacement exceeds input-path trust bound")
    else:
        raise ValueError("map-resampled corridor reserve did not converge")

    minimum_clearance = math.inf
    for i in range(len(xy)):
        j = (i + 1) % len(xy)
        angle = math.atan2(math.sin(heading[j] - heading[i]), math.cos(heading[j] - heading[i]))
        motion_bound = segments[i] + _FOOTPRINT_RADIUS * abs(angle)
        divisions = max(1, math.ceil(motion_bound / (.25 * grid.resolution)))
        sampled = min(grid.footprint_clearance(
            *(xy[i] + f * (xy[j] - xy[i])), heading[i] + f * angle,
            margin=args.wall_margin) for f in np.linspace(0., 1., divisions + 1))
        minimum_clearance = min(minimum_clearance,
            sampled - .5 * motion_bound / divisions)
    if minimum_clearance < 0:
        raise ValueError(f"swept map footprint certificate fails: {minimum_clearance:.6f} m")

    output = values.copy()
    output[:, 0] = np.r_[0., np.cumsum(segments[:-1])]
    output[:, 1:3], output[:, 3], output[:, 4], output[:, 7:9] = xy, heading, curvature, widths
    # Kinematic target acceleration, not a newly certified source-force state.
    u, acceleration = target_speed_profile(values[:, 5], segments, args.speed_scale)
    output[:, 5], output[:, 6] = u, acceleration
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.output, output, delimiter=',', fmt='%.9f',
        header='s_m,x_m,y_m,psi_rad,kappa_radpm,body_forward_speed_mps,acceleration_mps2,d_left_m,d_right_m')
    report = dict(input=str(args.input), input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        output=str(args.output), map=str(args.map), points=len(xy), passes=passes,
        wall_margin_m=args.wall_margin, tracking_reserve_m=args.tracking_reserve,
        minimum_center_to_wall_m=float(np.min(widths)), required_center_to_wall_m=required,
        swept_footprint_clearance_beyond_margin_m=minimum_clearance,
        maximum_displacement_m=float(np.max(np.linalg.norm(xy-original, axis=1))),
        geometry_smoothing_length_m=args.geometry_smoothing_length,
        geometry_smoothing_displacement_m=geometry_displacement,
        speed_scale=args.speed_scale,
        speed_samples_unchanged=args.speed_scale == 1., source_OCP_feasibility_transfers=False,
        simulator_validated=False,
        speed_profile_integral_s=float(np.sum(2 * segments / (u + np.roll(u, -1)))))
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
