import numpy as np

from tools.racing.export_unity_wheel_candidate_to_mpc import (
    MPC_RESAMPLE_SPACING_M,
    _closed_path_geometry,
)


def test_export_geometry_resamples_an_uneven_ocp_mesh_uniformly():
    angles = np.asarray((0.0, 0.005, 0.4, 1.1, 2.0, 3.1, 4.2, 5.2))
    xy = np.column_stack((np.cos(angles), np.sin(angles)))

    source_s, station, sampled_xy, heading, curvature, length = (
        _closed_path_geometry(xy)
    )

    assert np.min(np.diff(source_s)) < 0.01
    assert np.allclose(np.diff(station), MPC_RESAMPLE_SPACING_M)
    assert station[-1] < length
    assert sampled_xy.shape == (len(station), 2)
    assert heading.shape == curvature.shape == station.shape
    assert np.isfinite(sampled_xy).all()
    assert np.isfinite(curvature).all()
