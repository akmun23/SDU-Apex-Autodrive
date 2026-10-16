import numpy as np

from tools.racing.optimize_unity_spatial_raceline import (
    _unwrap_closed_route_indices,
)


def test_unwrap_closed_route_indices_assigns_sectors_across_finish_seam():
    indices = np.asarray([8, 9, 0, 1, 2])

    unwrapped = _unwrap_closed_route_indices(indices, route_count=10)
    sectors = ((unwrapped - unwrapped[0]) * 4 / 10).astype(int)

    assert np.array_equal(unwrapped, [8, 9, 10, 11, 12])
    assert np.array_equal(sectors, [0, 0, 0, 1, 1])
