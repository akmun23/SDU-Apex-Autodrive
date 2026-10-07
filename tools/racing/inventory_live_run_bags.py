#!/usr/bin/env python3
"""Write a lightweight index of recorded ROS bag files under live_runs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def classify(path: Path) -> str:
    value = "/".join(part.lower() for part in path.parts)
    groups = (
        ("open-plane dynamics/throttle", ("openplane", "open_plane", "throttle", "swerve")),
        ("mapping/SLAM", ("map_", "/map", "mapping", "slam")),
        ("odometry/localization", ("odom", "observer", "localization", "ekf", "amcl")),
        ("MPC/controller", ("mpc", "controller")),
        ("practice/racing", ("practice", "raceline", "racing")),
        ("competition/bridge", ("competition", "bridge")),
    )
    for label, terms in groups:
        if any(term in value for term in terms):
            return label
    return "other"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("live_runs"))
    parser.add_argument("--output", type=Path, default=Path("live_runs/INDEX_20261007.csv"))
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    if not root.is_dir():
        parser.error(f"run-data directory does not exist: {root}")

    bags = sorted(root.rglob("*.db3"))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "category", "run_name", "size_bytes", "size_gib", "bag_path",
            "nearby_files",
        ))
        writer.writeheader()
        for bag in bags:
            run_dir = bag.parent.parent if bag.parent.name in {"run", "bag", "replayed"} else bag.parent
            companions = sorted(
                str(item.relative_to(run_dir))
                for item in run_dir.iterdir()
                if item.is_file() and item != bag
            )
            if bag.parent != run_dir:
                companions.extend(sorted(
                    str(item.relative_to(run_dir))
                    for item in bag.parent.iterdir()
                    if item.is_file() and item != bag
                ))
            size = bag.stat().st_size
            writer.writerow({
                "category": classify(bag),
                "run_name": run_dir.name,
                "size_bytes": size,
                "size_gib": f"{size / (1024 ** 3):.6f}",
                "bag_path": bag.relative_to(root.parent),
                "nearby_files": ";".join(companions),
            })
    print(f"indexed {len(bags)} bags ({sum(path.stat().st_size for path in bags) / (1024 ** 3):.3f} GiB) -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
