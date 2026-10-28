"""A coarse clearance bound is inconclusive, not evidence of a collision."""
import sys
from pathlib import Path
import numpy as np
import pytest

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from native_validate import export_mpc, _menger_curvature  # noqa: E402


def test_export_refines_near_margin_and_still_rejects_violation(tmp_path):
    angle = np.linspace(0, 2*np.pi, 80, endpoint=False)
    xy = np.column_stack((np.cos(angle), np.sin(angle)))
    candidate = dict(x_m=xy[:,0], y_m=xy[:,1], speed_mps=np.ones(80),
                     path_curvature_per_m=_menger_curvature(xy),
                     ax_mps2=np.zeros(80))
    class Grid:
        resolution = .05
        residual = .004
        calls = 0
        def distance(self, points):
            return np.ones(len(points))
        def footprint_clearance(self, x, y, psi, *, margin):
            assert margin == .05
            self.calls += 1
            return self.residual
    grid = Grid()
    result = export_mpc(candidate, grid, tmp_path/'clear.csv', .05)
    assert result.exists()
    assert grid.calls > len(angle)*5
    grid.residual = -.0001
    with pytest.raises(ValueError, match='sampled clearance -0.000100'):
        export_mpc(candidate, grid, tmp_path/'bad.csv', .05)
    assert not (tmp_path/'bad.csv').exists()
