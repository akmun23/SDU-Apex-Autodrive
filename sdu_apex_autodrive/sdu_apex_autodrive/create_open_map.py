"""Create a featureless ROS occupancy map for map-server/software tests."""

import argparse
from pathlib import Path


def create_open_map(output_stem: Path, width_m: float, height_m: float, resolution: float):
    if width_m <= 0.0 or height_m <= 0.0 or resolution <= 0.0:
        raise ValueError("width, height, and resolution must be > 0")
    width = max(1, round(width_m / resolution))
    height = max(1, round(height_m / resolution))
    output_stem.parent.mkdir(parents=True, exist_ok=True)

    pgm = output_stem.with_suffix(".pgm")
    yaml = output_stem.with_suffix(".yaml")
    with pgm.open("wb") as stream:
        stream.write(f"P5\n{width} {height}\n255\n".encode("ascii"))
        stream.write(bytes([254]) * width * height)

    yaml.write_text(
        "image: %s\n"
        "mode: trinary\n"
        "resolution: %.9g\n"
        "origin: [%.9g, %.9g, 0.0]\n"
        "negate: 0\n"
        "occupied_thresh: 0.65\n"
        "free_thresh: 0.25\n"
        % (pgm.name, resolution, -0.5 * width * resolution, -0.5 * height * resolution),
        encoding="utf-8",
    )
    return pgm, yaml


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--width-m", type=float, default=50.0)
    parser.add_argument("--height-m", type=float, default=50.0)
    parser.add_argument("--resolution", type=float, default=0.05)
    parsed = parser.parse_args(args)
    pgm, yaml = create_open_map(
        parsed.output, parsed.width_m, parsed.height_m, parsed.resolution)
    print(f"Created {pgm} and {yaml}")
