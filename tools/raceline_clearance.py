"""Lightweight shared clearance policy for certificate readers and the GUI."""
from __future__ import annotations

import json
import math
from pathlib import Path

POLICY_PATH = Path(__file__).resolve().parents[1]/'config/racing/raceline_clearance_policy.json'
POLICY = json.loads(POLICY_PATH.read_text())
NUMERICAL_TOLERANCE_M = float(POLICY['numerical_tolerance_m'])
if (POLICY.get('schema') != 'sdu-apex-raceline-clearance/v1' or
        not math.isfinite(NUMERICAL_TOLERANCE_M) or not 0. <= NUMERICAL_TOLERANCE_M <= 1.e-4):
    raise ValueError('invalid raceline numerical-clearance policy')


def declared_tolerance(report):
    """Older unannotated certificates remain strict; never invent permission."""
    value = report.get('clearance_numerical_tolerance_m',
        (report.get('rollout') or {}).get('clearance_numerical_tolerance_m', 0.))
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and 0. <= value <= NUMERICAL_TOLERANCE_M else None
