"""Read the production rear-axle footprint without loading plotting/ROS code."""
from pathlib import Path
import re

_HEADER = (Path(__file__).resolve().parents[1] /
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
