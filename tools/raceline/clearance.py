"""Read clearance policy and rear-axle footprint dimensions from production files."""
from __future__ import annotations

import json
import math
from pathlib import Path
import re

POLICY_PATH = Path(__file__).resolve().parents[2]/'raceline_optimizer/clearance_policy.json'
POLICY = json.loads(POLICY_PATH.read_text())
NUMERICAL_TOLERANCE_M = float(POLICY['numerical_tolerance_m'])
if (POLICY.get('schema') != 'sdu-apex-raceline-clearance/v1' or
        not math.isfinite(NUMERICAL_TOLERANCE_M) or not 0. <= NUMERICAL_TOLERANCE_M <= 1.e-4):
    raise ValueError('invalid raceline numerical-clearance policy')


def declared_tolerance(report):
    """Require an explicit numerical tolerance within the clearance policy."""
    value = report.get('clearance_numerical_tolerance_m')
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and 0. <= value <= NUMERICAL_TOLERANCE_M else None


_HEADER = (Path(__file__).resolve().parents[2] /
           'f1tenth_mpc/include/mpc_types.h').read_text()


def dimension(name: str) -> float:
    match = re.search(r'^#define\s+' + re.escape(name) + r'\s+([0-9.]+)f?\s*$',
                      _HEADER, re.MULTILINE)
    if match is None:
        raise ValueError(f'Missing production footprint dimension: {name}')
    return float(match[1])


WIDTH_M = dimension('MPC_CAR_WIDTH_M')
BODY_X = (-dimension('MPC_REAR_OVERHANG_M'),
          dimension('MPC_REAR_AXLE_TO_FRONT_BUMPER_M'))
BODY_Y = (-WIDTH_M / 2, WIDTH_M / 2)
