"""Build the offline shim around the exact production odometry C++ sources."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def build(repo: Path, output: Path, compiler: str = "g++") -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        compiler, "-shared", "-fPIC", "-O2", "-std=c++17",
        "-I", str(repo / "f1tenth_localization/include"),
        str(repo / "tools/racing/offline/localization/production_odometry_shim.cpp"),
        str(repo / "f1tenth_localization/src/odometry_observer.cpp"),
        str(repo / "f1tenth_localization/src/sensor_packet_assembler.cpp"),
        "-lyaml-cpp", "-o", str(output),
    ]
    subprocess.run(command, cwd=repo, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--repo", type=Path,
                        default=Path(__file__).resolve().parents[4])
    parser.add_argument("--compiler", default="g++")
    args = parser.parse_args()
    build(args.repo.resolve(), args.output.resolve(), args.compiler)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

