#!/usr/bin/env python3
"""Plot racelines and the competition car's swept footprint on a ROS map."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image


try:
    from tools.vehicle_footprint import BODY_X, BODY_Y
except ModuleNotFoundError:
    from vehicle_footprint import BODY_X, BODY_Y
BUFFER_M = 0.05


def read_path(path: Path) -> np.ndarray:
    rows = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.reader(line for line in stream if not line.startswith("#")):
            if row and len(row) >= 4:
                rows.append([float(value) for value in row[:4]])
    data = np.asarray(rows, dtype=float)
    if data.ndim != 2 or data.shape[0] < 4:
        raise ValueError(f"Expected s,x,y,heading rows in {path}")
    return data


def footprint(x: float, y: float, yaw: float, buffer: float = 0.0) -> np.ndarray:
    x0, x1 = BODY_X[0] - buffer, BODY_X[1] + buffer
    y0, y1 = BODY_Y[0] - buffer, BODY_Y[1] + buffer
    local = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]])
    c, s = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[c, -s], [s, c]])
    return local @ rotation.T + np.array([x, y])


def path_pose_near_y(path: np.ndarray, target_y: float) -> np.ndarray:
    candidates = np.flatnonzero((path[:, 1] >= -3.0) & (path[:, 1] <= -1.0)
                                & (path[:, 2] >= -4.0) & (path[:, 2] <= 4.0))
    if not len(candidates):
        candidates = np.arange(len(path))
    return path[candidates[np.argmin(np.abs(path[candidates, 2] - target_y))]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path, required=True, help="PGM map")
    parser.add_argument("--yaml", type=Path, required=True, help="ROS map YAML")
    parser.add_argument("--reference", type=Path, required=True, help="reference trajectory CSV")
    parser.add_argument("--candidate", type=Path, help="optional candidate trajectory CSV")
    parser.add_argument("--candidate-label", default="candidate",
                        help="legend label for the optional candidate")
    parser.add_argument("--output", type=Path, required=True, help="output PNG")
    parser.add_argument("--roi", nargs=4, type=float, default=(-3.7, -0.5, -5.0, 4.5),
                        metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
                        help="tight-section zoom bounds in map coordinates")
    args = parser.parse_args()

    config = yaml.safe_load(args.yaml.read_text(encoding="utf-8"))
    resolution = float(config["resolution"])
    origin_x, origin_y = map(float, config["origin"][:2])
    raster = np.asarray(Image.open(args.map).convert("L"))
    height, width = raster.shape
    extent = (origin_x, origin_x + width * resolution,
              origin_y, origin_y + height * resolution)
    reference = read_path(args.reference)
    candidate = read_path(args.candidate) if args.candidate else None

    fig, (full, zoom) = plt.subplots(1, 2, figsize=(15, 9), constrained_layout=True)
    for ax in (full, zoom):
        ax.imshow(np.flipud(raster), extent=extent, origin="lower", cmap="gray",
                  vmin=0, vmax=255, interpolation="nearest", zorder=0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("map x [m]")
        ax.set_ylabel("map y [m]")

    full.plot(reference[:, 1], reference[:, 2], color="#00a6d6", lw=1.7,
              label="v300 reference", zorder=3)
    zoom.plot(reference[:, 1], reference[:, 2], color="#00a6d6", lw=2.0,
              label="v300 reference", zorder=3)
    if candidate is not None:
        full.plot(candidate[:, 1], candidate[:, 2], color="#e34a8a", lw=1.8,
                  label=args.candidate_label, zorder=4)
        zoom.plot(candidate[:, 1], candidate[:, 2], color="#e34a8a", lw=2.0,
                  label=args.candidate_label, zorder=4)

    # Show representative car poses in the selected area. Solid outlines are
    # the body; dashed outlines include the requested extra 5 cm.
    # Include the middle connector curvature-transition pose explicitly.
    for target_y in (3.2, 1.0, -1.8, -3.4):
        for path, color in ((reference, "#075b91"), (candidate, "#c21870")):
            if path is None:
                continue
            row = path_pose_near_y(path, target_y)
            body = footprint(row[1], row[2], row[3])
            buffered = footprint(row[1], row[2], row[3], BUFFER_M)
            zoom.plot(*buffered.T, color=color, lw=0.9, ls="--", alpha=0.78,
                      zorder=5)
            zoom.plot(*body.T, color=color, lw=1.35, zorder=6)
            zoom.plot(row[1], row[2], "o", ms=2.4, color=color, zorder=7)

    full.set_title("Practice track: line and car scale")
    full.legend(loc="lower left", framealpha=0.92)
    zoom.set_xlim(args.roi[0], args.roi[1])
    zoom.set_ylim(args.roi[2], args.roi[3])
    zoom.set_title("Connector clearance: body outline + 5 cm envelope")
    zoom.legend(loc="lower right", framealpha=0.92)
    fig.suptitle("Vehicle footprint from MPC body anchors: 0.510 m × 0.273 m")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
