"""Vehicle footprint used by production MPC wall-corridor checks.

Dimensions mirror ``f1tenth_mpc/include/mpc_types.h`` and
``f1tenth_planning/config/autodrive_sim_vehicle.yaml``. These are geometric
constraints only; they do not alter Unity physics or the vehicle dynamics.
"""

from __future__ import annotations

import numpy as np

MPC_CAR_WIDTH_M = 0.273
MPC_REAR_OVERHANG_M = 0.080
MPC_REAR_AXLE_TO_FRONT_BUMPER_M = 0.430
MPC_REQUIRED_WALL_CLEARANCE_M = 0.150


def footprint_half_width_numpy(heading_error_rad: np.ndarray | float):
    """Project the production MPC vehicle rectangle onto the track normal."""
    heading = np.asarray(heading_error_rad, dtype=np.float64)
    longitudinal_extent = max(
        MPC_REAR_AXLE_TO_FRONT_BUMPER_M, MPC_REAR_OVERHANG_M
    )
    return (0.5 * MPC_CAR_WIDTH_M * np.abs(np.cos(heading))
            + longitudinal_extent * np.abs(np.sin(heading)))


def footprint_half_width_casadi(heading_error_rad, casadi):
    """CasADi counterpart used to impose the same envelope in the OCP."""
    longitudinal_extent = max(
        MPC_REAR_AXLE_TO_FRONT_BUMPER_M, MPC_REAR_OVERHANG_M
    )
    return (0.5 * MPC_CAR_WIDTH_M * casadi.fabs(casadi.cos(heading_error_rad))
            + longitudinal_extent * casadi.fabs(casadi.sin(heading_error_rad)))
