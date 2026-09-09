import math
import hashlib

import pytest

from sdu_apex_autodrive.map_provenance import load_map_provenance


def test_map_provenance_round_trip(tmp_path):
    path = tmp_path / "track.provenance.yaml"
    path.write_text(
        """
map_yaml: /maps/track.yaml
map_frame: map
world_frame: gt_odom
map_to_world:
  x_m: 5.0
  y_m: -2.0
  yaw_rad: 0.5
source: test
""",
        encoding="utf-8",
    )

    provenance = load_map_provenance(str(path))
    world = provenance.map_to_world(1.0, 2.0, -0.25)
    recovered = provenance.world_to_map(*world)

    assert recovered[0] == pytest.approx(1.0)
    assert recovered[1] == pytest.approx(2.0)
    assert recovered[2] == pytest.approx(-0.25)
    assert provenance.source == "test"


def test_map_provenance_rejects_missing_transform_value(tmp_path):
    path = tmp_path / "invalid.yaml"
    path.write_text(
        "map_yaml: /maps/track.yaml\nmap_to_world: {x_m: 0.0, y_m: 0.0}\n",
        encoding="utf-8",
    )

    with pytest.raises((TypeError, ValueError)):
        load_map_provenance(str(path))


def test_map_provenance_wraps_heading(tmp_path):
    path = tmp_path / "heading.yaml"
    path.write_text(
        """
map_yaml: /maps/track.yaml
map_to_world: {x_m: 0.0, y_m: 0.0, yaw_rad: 3.0}
""",
        encoding="utf-8",
    )
    provenance = load_map_provenance(str(path))
    _, _, yaw = provenance.world_to_map(0.0, 0.0, -3.0)
    assert -math.pi <= yaw <= math.pi


def test_map_provenance_detects_changed_map(tmp_path):
    map_yaml = tmp_path / "track.yaml"
    map_image = tmp_path / "track.pgm"
    map_yaml.write_text("resolution: 0.025\n", encoding="utf-8")
    map_image.write_bytes(b"P5\n1 1\n255\n\x00")

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    path = tmp_path / "track.provenance.yaml"
    path.write_text(
        f"""
map_yaml: {map_yaml}
map_to_world: {{x_m: 0.0, y_m: 0.0, yaw_rad: 0.0}}
map_yaml_sha256: {digest(map_yaml)}
map_image_sha256: {digest(map_image)}
""",
        encoding="utf-8",
    )
    provenance = load_map_provenance(str(path))
    provenance.verify_files()

    map_yaml.write_text("resolution: 0.020\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        provenance.verify_files()
