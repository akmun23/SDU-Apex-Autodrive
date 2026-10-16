import casadi as ca
import numpy as np
import pytest

from tools.racing.mpc_vehicle_geometry import (
    footprint_half_width_casadi,
    footprint_half_width_numpy,
)


@pytest.mark.parametrize("heading", (0.0, 0.2, 0.5, 0.631365, -0.5))
def test_optimizer_footprint_matches_production_mpc_geometry(heading):
    symbolic = ca.SX.sym("heading")
    footprint = ca.Function(
        "footprint", [symbolic],
        [footprint_half_width_casadi(symbolic, ca)],
    )

    actual = float(footprint(heading))
    expected = float(footprint_half_width_numpy(heading))
    assert actual == pytest.approx(expected, abs=1e-12)


def test_front_footprint_uses_production_bumper_extent():
    assert float(footprint_half_width_numpy(0.5)) == pytest.approx(
        0.5 * 0.273 * np.cos(0.5) + 0.430 * np.sin(0.5), abs=1e-12
    )
