"""Create a featureless ROS occupancy map for software-only map tests."""

import argparse
from pathlib import Path


def create_open_map(
    output_stem: Path,
    width_m: float,
    height_m: float,
    resolution: float,
) -> tuple[Path, Path]:
    if width_m <= 0.0 or height_m <= 0.0 or resolution <= 0.0:
        raise ValueError('width, height, and resolution must be > 0')
    width_pixels = max(1, round(width_m / resolution))
    height_pixels = max(1, round(height_m / resolution))
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    pgm_path = output_stem.with_suffix('.pgm')
    yaml_path = output_stem.with_suffix('.yaml')

    header = 'P5\n%d %d\n255\n' % (width_pixels, height_pixels)
    with pgm_path.open('wb') as stream:
        stream.write(header.encode('ascii'))
        stream.write(bytes([254]) * width_pixels * height_pixels)

    yaml_path.write_text(
        'image: %s\n'
        'mode: trinary\n'
        'resolution: %.9g\n'
        'origin: [%.9g, %.9g, 0.0]\n'
        'negate: 0\n'
        'occupied_thresh: 0.65\n'
        'free_thresh: 0.25\n' % (
            pgm_path.name,
            resolution,
            -0.5 * width_pixels * resolution,
            -0.5 * height_pixels * resolution,
        ),
        encoding='utf-8',
    )
    return pgm_path, yaml_path


def main(args=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--width-m', type=float, default=50.0)
    parser.add_argument('--height-m', type=float, default=50.0)
    parser.add_argument('--resolution', type=float, default=0.05)
    parsed = parser.parse_args(args)
    pgm_path, yaml_path = create_open_map(
        parsed.output, parsed.width_m, parsed.height_m, parsed.resolution)
    print('Created %s and %s' % (pgm_path, yaml_path))
