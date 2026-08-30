from pathlib import Path

import pytest

from sdu_apex_autodrive.create_open_map import create_open_map


def test_create_open_map_writes_free_centered_grid(tmp_path: Path):
    pgm_path, yaml_path = create_open_map(
        tmp_path / 'open_map',
        width_m=2.0,
        height_m=1.0,
        resolution=0.5,
    )

    assert pgm_path.read_bytes() == b'P5\n4 2\n255\n' + bytes([254]) * 8
    yaml = yaml_path.read_text(encoding='utf-8')
    assert 'image: open_map.pgm' in yaml
    assert 'resolution: 0.5' in yaml
    assert 'origin: [-1, -0.5, 0.0]' in yaml


@pytest.mark.parametrize(
    'width_m,height_m,resolution',
    [(0.0, 1.0, 0.1), (1.0, -1.0, 0.1), (1.0, 1.0, 0.0)],
)
def test_create_open_map_rejects_nonpositive_dimensions(
    tmp_path: Path,
    width_m: float,
    height_m: float,
    resolution: float,
):
    with pytest.raises(ValueError):
        create_open_map(
            tmp_path / 'bad', width_m, height_m, resolution)
