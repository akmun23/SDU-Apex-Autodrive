#!/usr/bin/env python3
"""Canonical offline N=30, 40 Hz stage map.

The production contract is one command every 25 ms for 30 commands.  This
module gives all offline tools one entry point for the nonlinear stage map;
the optional internal substep is an integration-resolution choice, not a
change to the physical horizon or command cadence.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

import structured_vehicle_plant as plant


COMMAND_DT_S = 0.025
COMMAND_COUNT = 30
PHYSICAL_HORIZON_S = COMMAND_COUNT * COMMAND_DT_S
DEFAULT_INTERNAL_SUBSTEP_S = 0.0125


def f25(state: np.ndarray, steering_target_norm: float,
        throttle_norm: float, parameters: plant.PlantParameters,
        internal_substep_s: float = DEFAULT_INTERNAL_SUBSTEP_S) -> np.ndarray:
    """Advance exactly one 25 ms command stage causally."""
    if not math.isfinite(internal_substep_s) or not 0.0 < internal_substep_s <= COMMAND_DT_S:
        raise ValueError("internal stage substep must be in (0, 0.025]")
    return plant.step(state, steering_target_norm, throttle_norm, COMMAND_DT_S,
                      parameters, internal_substep_s=internal_substep_s)


def finite_difference_jacobian(state: np.ndarray, steering_target_norm: float,
                               throttle_norm: float,
                               parameters: plant.PlantParameters,
                               internal_substep_s: float = DEFAULT_INTERNAL_SUBSTEP_S,
                               state_step: float = 1.0e-5,
                               input_step: float = 1.0e-5) -> dict[str, Any]:
    """Return central-difference A/B and finite/bounded diagnostics."""
    if state.shape != (8,):
        raise ValueError("stage-map state must have eight elements")
    nominal = f25(state, steering_target_norm, throttle_norm, parameters,
                  internal_substep_s)
    a = np.empty((8, 8), dtype=float)
    for column in range(8):
        delta = np.zeros(8, dtype=float)
        delta[column] = state_step
        plus = f25(state + delta, steering_target_norm, throttle_norm,
                   parameters, internal_substep_s)
        minus = f25(state - delta, steering_target_norm, throttle_norm,
                    parameters, internal_substep_s)
        a[:, column] = (plus - minus) / (2.0 * state_step)
    b = np.empty((8, 2), dtype=float)
    for column, step in enumerate((input_step, input_step)):
        plus_input = [steering_target_norm, throttle_norm]
        minus_input = [steering_target_norm, throttle_norm]
        plus_input[column] += step
        minus_input[column] -= step
        plus = f25(state, plus_input[0], plus_input[1], parameters,
                   internal_substep_s)
        minus = f25(state, minus_input[0], minus_input[1], parameters,
                    internal_substep_s)
        b[:, column] = (plus - minus) / (2.0 * step)
    return {
        "A": a,
        "B": b,
        "nominal": nominal,
        "finite": bool(np.all(np.isfinite(a)) and np.all(np.isfinite(b)) and
                        np.all(np.isfinite(nominal))),
        "max_abs_jacobian": float(max(np.max(np.abs(a)), np.max(np.abs(b)))),
    }
