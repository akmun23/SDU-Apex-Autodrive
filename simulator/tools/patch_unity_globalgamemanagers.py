#!/usr/bin/env python3
"""Create an experimental Unity player-data copy with a different timestep.

This does not patch the simulator executable.  It edits only the serialized
TimeManager in a copied ``globalgamemanagers`` file, so the original player
remains untouched.  Changing Fixed Timestep changes simulator physics and is
not competition-safe without validation against the official player.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import UnityPy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fixed-timestep", type=float, required=True)
    parser.add_argument(
        "--disable-vsync",
        action="store_true",
        help="set vSyncCount=0 for every serialized quality level",
    )
    args = parser.parse_args()

    if args.fixed_timestep <= 0.0:
        parser.error("--fixed-timestep must be positive")
    if not args.input.is_file():
        parser.error(f"input does not exist: {args.input}")
    if args.output.resolve() == args.input.resolve():
        parser.error("refusing to overwrite the input asset")

    environment = UnityPy.load(str(args.input))
    matches = [obj for obj in environment.objects if obj.type.name == "TimeManager"]
    if len(matches) != 1:
        raise RuntimeError(f"expected one TimeManager, found {len(matches)}")

    obj = matches[0]
    data = obj.read()
    old_value = float(data.Fixed_Timestep)
    data.Fixed_Timestep = float(args.fixed_timestep)
    data.save()

    if args.disable_vsync:
        quality_matches = [
            candidate
            for candidate in environment.objects
            if candidate.type.name == "QualitySettings"
        ]
        if len(quality_matches) != 1:
            raise RuntimeError(
                f"expected one QualitySettings object, found {len(quality_matches)}")
        quality = quality_matches[0].read()
        for setting in quality.m_QualitySettings:
            setting.vSyncCount = 0
        quality_matches[0].save_typetree(quality)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(environment.file.save())
    print(f"Fixed_Timestep: {old_value:.9g} -> {args.fixed_timestep:.9g}")
    if args.disable_vsync:
        print("vSyncCount: all quality levels -> 0")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
