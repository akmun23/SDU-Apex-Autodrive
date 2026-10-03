"""Longitudinal-axis reflection transforms shared by training and scoring."""

from __future__ import annotations

from typing import Any

import numpy as np


def _copy(value: Any) -> Any:
    clone = getattr(value, "clone", None)
    return clone() if callable(clone) else np.asarray(value).copy()


def reflect_state_channels(value: Any) -> Any:
    """Reflect EDSSM state [..., u, v, r, steer, throttle, wheel-L/R, roll?]."""
    if value.ndim < 1 or value.shape[-1] not in (7, 9):
        raise ValueError("state reflection requires a seven- or nine-channel state")
    result = _copy(value)
    result[..., 1:4] *= -1.0
    result[..., 5:7] = result[..., [6, 5]]
    if result.shape[-1] == 9:
        result[..., 7:9] *= -1.0
    return result


def reflect_commands(value: Any) -> Any:
    """Reflect steering while preserving throttle in [..., steer, throttle]."""
    if value.ndim < 1 or value.shape[-1] != 2:
        raise ValueError("command reflection requires steering/throttle pairs")
    result = _copy(value)
    result[..., 0] *= -1.0
    return result


def reflect_history_features(value: Any, history_state_size: int) -> Any:
    """Reflect state, commands, and optional raw/innovation wheel history."""
    if value.ndim < 2 or history_state_size not in (7, 9):
        raise ValueError("history reflection requires a valid state width")
    if value.shape[-1] < history_state_size + 2:
        raise ValueError("history lacks steering/throttle command channels")
    extra_width = value.shape[-1] - history_state_size - 2
    if extra_width not in (0, 3):
        raise ValueError("unsupported encoder-history feature width")
    result = _copy(value)
    result[..., :history_state_size] = reflect_state_channels(
        result[..., :history_state_size])
    command_start = history_state_size
    result[..., command_start:command_start + 2] = reflect_commands(
        result[..., command_start:command_start + 2])
    if extra_width == 3:
        start = command_start + 2
        result[..., start:start + 2] = result[..., [start + 1, start]]
    return result


def reflect_acceleration_targets(value: Any) -> Any:
    """Reflect [ax, ay, yaw acceleration, rear wheel-L/R acceleration]."""
    if value.ndim < 1 or value.shape[-1] != 5:
        raise ValueError("acceleration reflection requires five target channels")
    result = _copy(value)
    result[..., 1:3] *= -1.0
    result[..., 3:5] = result[..., [4, 3]]
    return result


def reflect_pose_sequence(poses: np.ndarray) -> np.ndarray:
    """Reflect world x/y/yaw samples about the initial body longitudinal axis."""
    value = np.asarray(poses)
    if value.ndim != 3 or value.shape[-1] != 3 or value.shape[1] < 1:
        raise ValueError("pose reflection requires shape (batch, time, 3)")
    result = value.copy()
    origin_xy = value[:, :1, :2]
    origin_yaw = value[:, :1, 2:3]
    dx = value[..., 0:1] - origin_xy[..., 0:1]
    dy = value[..., 1:2] - origin_xy[..., 1:2]
    cos_yaw = np.cos(origin_yaw)
    sin_yaw = np.sin(origin_yaw)
    local_x = cos_yaw * dx + sin_yaw * dy
    local_y = -sin_yaw * dx + cos_yaw * dy
    result[..., 0:1] = origin_xy[..., 0:1] + cos_yaw * local_x + sin_yaw * local_y
    result[..., 1:2] = origin_xy[..., 1:2] + sin_yaw * local_x - cos_yaw * local_y
    relative_yaw = np.arctan2(
        np.sin(value[..., 2:3] - origin_yaw),
        np.cos(value[..., 2:3] - origin_yaw))
    result[..., 2:3] = origin_yaw - relative_yaw
    return result

