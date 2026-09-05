#!/usr/bin/env python3
"""Create a reversible Unity scene-file copy with Socket cameras unassigned.

This is an experiment for a prebuilt IL2CPP player when the Unity project is
not available.  It targets the known AutoDRIVE RoboRacer scene serialization
and removes the one front-camera PPtr from the Socket component.  It does not
change the executable or ROS API.  The modified player must not be used as a
competition-equivalent build without organizer approval because its serialized
scene differs from the supplied image.
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import UnityPy


SOCKET_SCRIPT_PATH_ID = 468
FRONT_CAMERA_ARRAY_OFFSET = 232
FRONT_CAMERA_ELEMENT_SIZE = 12


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"input does not exist: {args.input}")
    if args.output.resolve() == args.input.resolve():
        parser.error("refusing to overwrite the input asset")

    environment = UnityPy.load(str(args.input))
    matches = []
    for obj in environment.objects:
        if obj.type.name != "MonoBehaviour":
            continue
        raw = obj.get_raw_data()
        if len(raw) < 28:
            continue
        script_file_id = struct.unpack_from("<i", raw, 16)[0]
        script_path_id = struct.unpack_from("<q", raw, 20)[0]
        if script_file_id == 1 and script_path_id == SOCKET_SCRIPT_PATH_ID:
            matches.append((obj, raw))

    if len(matches) != 1:
        raise RuntimeError(f"expected one Socket component, found {len(matches)}")

    obj, raw = matches[0]
    offset = FRONT_CAMERA_ARRAY_OFFSET
    array_length = struct.unpack_from("<I", raw, offset)[0]
    if array_length != 1:
        raise RuntimeError(
            f"expected one front camera at offset {offset}, found {array_length}")

    new_raw = raw[:offset] + struct.pack("<I", 0) + raw[
        offset + 4 + FRONT_CAMERA_ELEMENT_SIZE:
    ]
    obj.set_raw_data(new_raw)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(environment.file.save())
    print(f"Socket component {obj.path_id}: FrontCameras 1 -> 0")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
