"""Map-to-simulator-frame provenance used by offline localization scoring.

The simulator pose is diagnostics-only.  This module only describes the
fixed transform recorded when a map is saved; it is never used by AMCL or a
controller at runtime.
"""

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _finite(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"map provenance field {name} must be finite")
    return result


@dataclass(frozen=True)
class MapProvenance:
    """Immutable SE(2) transform from map coordinates to simulator world."""

    map_yaml: str
    map_frame: str
    world_frame: str
    map_to_world_x_m: float
    map_to_world_y_m: float
    map_to_world_yaw_rad: float
    source: str = "unknown"
    map_yaml_sha256: str = ""
    map_image_sha256: str = ""

    def verify_files(self) -> None:
        """Reject a provenance file that no longer describes its map files."""
        if self.map_yaml_sha256:
            map_path = Path(self.map_yaml)
            if not map_path.is_file():
                raise FileNotFoundError(
                    f"map YAML referenced by provenance does not exist: {map_path}")
            if _sha256(map_path) != self.map_yaml_sha256:
                raise ValueError(f"map YAML hash does not match provenance: {map_path}")
        if self.map_image_sha256:
            image_path = Path(self.map_yaml).with_suffix(".pgm")
            if not image_path.is_file():
                raise FileNotFoundError(
                    f"map image referenced by provenance does not exist: {image_path}")
            if _sha256(image_path) != self.map_image_sha256:
                raise ValueError(f"map image hash does not match provenance: {image_path}")

    def world_to_map(
        self, x_m: float, y_m: float, yaw_rad: float
    ) -> Tuple[float, float, float]:
        """Transform a simulator-world pose into the map frame."""
        dx = float(x_m) - self.map_to_world_x_m
        dy = float(y_m) - self.map_to_world_y_m
        c = math.cos(self.map_to_world_yaw_rad)
        s = math.sin(self.map_to_world_yaw_rad)
        map_x = c * dx + s * dy
        map_y = -s * dx + c * dy
        map_yaw = wrap_angle(float(yaw_rad) - self.map_to_world_yaw_rad)
        return map_x, map_y, map_yaw

    def map_to_world(
        self, x_m: float, y_m: float, yaw_rad: float
    ) -> Tuple[float, float, float]:
        """Transform a map-frame pose into simulator world coordinates."""
        c = math.cos(self.map_to_world_yaw_rad)
        s = math.sin(self.map_to_world_yaw_rad)
        world_x = self.map_to_world_x_m + c * float(x_m) - s * float(y_m)
        world_y = self.map_to_world_y_m + s * float(x_m) + c * float(y_m)
        world_yaw = wrap_angle(float(yaw_rad) + self.map_to_world_yaw_rad)
        return world_x, world_y, world_yaw


def load_map_provenance(path: str) -> MapProvenance:
    """Load and validate the provenance file written by the map saver."""
    provenance_path = Path(path)
    if not provenance_path.is_file():
        raise FileNotFoundError(f"map provenance file does not exist: {path}")
    with provenance_path.open("r", encoding="utf-8") as stream:
        document: Dict[str, Any] = yaml.safe_load(stream) or {}

    transform = document.get("map_to_world") or {}
    if not isinstance(transform, dict):
        raise ValueError("map provenance map_to_world must be a mapping")
    map_yaml = str(document.get("map_yaml", ""))
    map_frame = str(document.get("map_frame", "map"))
    world_frame = str(document.get("world_frame", "gt_odom"))
    if not map_yaml:
        raise ValueError("map provenance must contain map_yaml")
    if not map_frame or not world_frame:
        raise ValueError("map provenance frame names must not be empty")

    return MapProvenance(
        map_yaml=map_yaml,
        map_frame=map_frame,
        world_frame=world_frame,
        map_to_world_x_m=_finite(transform.get("x_m"), "map_to_world.x_m"),
        map_to_world_y_m=_finite(transform.get("y_m"), "map_to_world.y_m"),
        map_to_world_yaw_rad=_finite(
            transform.get("yaw_rad"), "map_to_world.yaw_rad"),
        source=str(document.get("source", "unknown")),
        map_yaml_sha256=str(document.get("map_yaml_sha256", "")),
        map_image_sha256=str(document.get("map_image_sha256", "")),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
