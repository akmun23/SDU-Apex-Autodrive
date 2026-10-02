"""Shared speed-stratified objectives for the <=12 m/s race-domain teachers."""

from __future__ import annotations

import numpy as np


RACE_SPEED_CAP_MPS = 12.0
RACE_SPEED_BIN_EDGES_MPS = (0.0, 5.0, 7.0, 9.0, 12.000001)
RACE_SPEED_DOMAINS = ("core_0_to_9", "fast_9_to_12")


def race_speed_bin_weights(speed_mps: np.ndarray) -> np.ndarray:
    """Give each populated speed bin equal total weight; preserve mean weight 1."""
    speed = np.asarray(speed_mps, dtype=np.float64)
    if (not np.isfinite(speed).all() or np.any(speed < 0.0)
            or np.any(speed > RACE_SPEED_CAP_MPS + 1.0e-5)):
        raise ValueError("race-domain loss received invalid or >12 m/s labels")
    edges = np.asarray(RACE_SPEED_BIN_EDGES_MPS, dtype=np.float64)
    bins = np.searchsorted(edges, speed, side="right") - 1
    if np.any((bins < 0) | (bins >= len(edges) - 1)):
        raise ValueError("race-domain speed label is outside configured bins")
    counts = np.bincount(bins.reshape(-1), minlength=len(edges) - 1)
    populated = counts > 0
    weights_per_bin = np.zeros(len(counts), dtype=np.float64)
    weights_per_bin[populated] = (
        speed.size / (np.count_nonzero(populated) * counts[populated]))
    weights = weights_per_bin[bins]
    return weights.reshape(speed.shape).astype(np.float32)


def race_speed_mismatch_weights(speed_mps: np.ndarray,
                                wheel_body_mismatch_mps: np.ndarray,
                                mismatch_thresholds_mps: tuple[float, float]
                                ) -> np.ndarray:
    """Balance speed and observed wheel/body-mismatch proxy independently.

    The mismatch is not a tire-slip measurement. Thresholds must be computed
    from training runs only. The product of marginal weights emphasizes both
    fast and high-mismatch samples without pretending every joint cell has
    support.
    """
    speed = np.asarray(speed_mps, dtype=np.float64)
    mismatch = np.abs(np.asarray(wheel_body_mismatch_mps, dtype=np.float64))
    if speed.shape != mismatch.shape or speed.size == 0:
        raise ValueError("speed and wheel/body-mismatch arrays must align")
    low, high = map(float, mismatch_thresholds_mps)
    if (not np.isfinite((low, high)).all() or low < 0.0 or high <= low
            or not np.isfinite(mismatch).all()):
        raise ValueError("invalid training-derived mismatch thresholds/labels")
    speed_weights = race_speed_bin_weights(speed)
    mismatch_bin = np.where(mismatch <= low, 0,
                            np.where(mismatch <= high, 1, 2))
    counts = np.bincount(mismatch_bin.reshape(-1), minlength=3)
    populated = counts > 0
    per_bin = np.zeros(3, dtype=np.float64)
    per_bin[populated] = speed.size / (np.count_nonzero(populated)
                                       * counts[populated])
    weights = speed_weights * per_bin[mismatch_bin]
    weights /= np.mean(weights)
    return weights.astype(np.float32)


def race_speed_domain_labels(speed_mps: np.ndarray) -> np.ndarray:
    """Label validation starts as core (<9) or fast boundary ([9,12])."""
    speed = np.asarray(speed_mps, dtype=np.float64)
    if (not np.isfinite(speed).all() or np.any(speed < 0.0)
            or np.any(speed > RACE_SPEED_CAP_MPS + 1.0e-5)):
        raise ValueError("race-domain scoring received invalid or >12 m/s speed")
    labels = np.where(speed < 9.0, RACE_SPEED_DOMAINS[0],
                      RACE_SPEED_DOMAINS[1])
    return labels.astype("U16")


def weighted_smooth_l1(torch, prediction, target, sample_weights, beta: float):
    """Smooth-L1 over channels, then the supplied per-sample importance weights."""
    values = torch.nn.functional.smooth_l1_loss(
        prediction, target, beta=beta, reduction="none")
    per_sample = values.mean(dim=-1)
    weights = torch.as_tensor(sample_weights, dtype=per_sample.dtype,
                              device=per_sample.device)
    return torch.sum(per_sample * weights) / torch.clamp(
        torch.sum(weights), min=1.0e-12)
