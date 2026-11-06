"""Periodic sector-local timing relaxation and exact-grid INITIALIZER.

The 24/25 ms source-map chord is only an IPOPT predictor, never a source
certificate. Final commands retain exact 25 ms timing and source certification.
"""
from __future__ import annotations

import math
import numpy as np


def periodic_sector_weights(count: int, sectors: int) -> np.ndarray:
    """Smoothly interpolate S cyclic timing knobs across N command intervals."""
    if not isinstance(count, int) or not isinstance(sectors, int) or count < 2 or not 1 <= sectors <= count:
        raise ValueError('bad number of commands or sectors')
    coordinate = np.arange(count, dtype=float)*sectors/count
    base = np.floor(coordinate).astype(int)
    residual = coordinate-base
    weights = np.zeros((count, sectors))
    rows = np.arange(count)
    weights[rows, base % sectors] += 1.-residual
    weights[rows, (base+1) % sectors] += residual
    return weights


def predictor_timing(count: int, fractions, sectors: int = 1):
    """Time edges in seconds. Uniform scalar fractions preserve legacy mode."""
    values = np.asarray(fractions, dtype=float).reshape(-1)
    if (values.size != sectors or not np.isfinite(values).all()
            or np.any(values < 0.) or np.any(values > 1.)):
        raise ValueError('invalid sector timing decision')
    weights = periodic_sector_weights(count, sectors)
    dt = .024 + .001*(weights @ values)
    edges = np.r_[0., np.cumsum(dt)]
    return dt, edges


def resample_source_time_initialization(states, controls, fractions, sectors=1):
    """Produce an *uncertified* command-grid seed from a variable-time guess.

    Resampling follows the actual cumulative commanded time, not a uniform
    phase approximation. Held controls are selected at the new interval start.
    No source feasibility or periodicity is assumed of the resulting seed.
    """
    states=np.asarray(states,dtype=float)
    controls=np.asarray(controls,dtype=float)
    if (states.ndim!=2 or controls.ndim!=2 or
            controls.shape[0]!=2 or states.shape!=(21, controls.shape[1]+1)
            or not np.isfinite(states).all() or not np.isfinite(controls).all()):
        raise ValueError('invalid source predictor primal')
    count=controls.shape[1]
    _, edges=predictor_timing(count,fractions,sectors)
    lap=float(edges[-1])
    target=int(math.ceil(lap/.025-1.e-9))
    if not 30 <= target < count:
        return None
    query=np.linspace(0.,lap,target+1)
    resampled=np.array([np.interp(query,edges,row) for row in states])
    indices=np.clip(np.searchsorted(edges,query[:-1],side='right')-1,0,count-1)
    held=controls[:,indices].copy()
    # This is a proposed primal, not a physics rollout or certificate.
    return resampled, held, lap
