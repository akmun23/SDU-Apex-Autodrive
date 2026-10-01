"""Empirical, latency-aware AMCL pose surrogate fit from train-run bags."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class LocalizationRun:
    run_id: str
    time_s: np.ndarray
    residual_xyyaw: np.ndarray
    latency_s: float


class EmpiricalAmclSurrogate:
    """Replay a sampled real localization-error trace over delayed plant pose.

    Calibration rows must come from training practice runs only. At runtime
    the surrogate consumes causal plant pose history and an independently
    sampled training-run residual trace; it never reads future plant truth.
    """

    def __init__(self, runs: tuple[LocalizationRun, ...], seed: int = 20261014,
                 history_seconds: float = 2.0,
                 rollout_seconds: float = 2.0) -> None:
        if len(runs) < 2:
            raise ValueError("AMCL surrogate needs at least two independent runs")
        for run in runs:
            times = np.asarray(run.time_s, dtype=np.float64)
            residual = np.asarray(run.residual_xyyaw, dtype=np.float64)
            if (times.ndim != 1 or residual.shape != (len(times), 3)
                    or len(times) < 2 or not np.isfinite(times).all()
                    or not np.isfinite(residual).all()
                    or np.any(np.diff(times) <= 0.0)
                    or not np.isfinite(run.latency_s) or run.latency_s < 0.0):
                raise ValueError(f"invalid AMCL residual trace: {run.run_id}")
        if (not np.isfinite((history_seconds, rollout_seconds)).all()
                or history_seconds <= 0.0 or rollout_seconds <= 0.0):
            raise ValueError("history and rollout durations must be positive")
        self.runs = runs
        self.rng = np.random.default_rng(seed)
        self.history_seconds = float(history_seconds)
        self.rollout_seconds = float(rollout_seconds)
        self.reset()

    def reset(self, time_s: float = 0.0,
              initial_plant_pose_xyyaw: np.ndarray | None = None,
              initial_localization_pose_xyyaw: np.ndarray | None = None,
              rollout_seconds: float | None = None) -> None:
        plant_pose = (np.zeros(3, dtype=np.float64)
                      if initial_plant_pose_xyyaw is None
                      else np.asarray(initial_plant_pose_xyyaw, dtype=np.float64))
        localization_pose = (
            None if initial_localization_pose_xyyaw is None else
            np.asarray(initial_localization_pose_xyyaw, dtype=np.float64))
        if (plant_pose.shape != (3,)
                or (localization_pose is not None
                    and localization_pose.shape != (3,))
                or not np.isfinite(plant_pose).all()
                or (localization_pose is not None
                    and not np.isfinite(localization_pose).all())):
            raise ValueError("initial plant/localization poses must be finite x/y/yaw")
        duration = (self.rollout_seconds if rollout_seconds is None
                    else float(rollout_seconds))
        if not np.isfinite(duration) or duration <= 0.0:
            raise ValueError("rollout_seconds must be positive and finite")
        self._run = self.runs[int(self.rng.integers(0, len(self.runs)))]
        self._latency_s = float(self._run.latency_s)
        margin = duration + abs(self._latency_s) + 0.025
        latest_origin = float(self._run.time_s[-1]) - margin
        earliest_origin = float(self._run.time_s[0])
        if latest_origin < earliest_origin:
            raise ValueError(
                f"AMCL trace {self._run.run_id} is too short for "
                f"{duration:.3f}s rollout and {self._latency_s:.3f}s latency")
        self._trace_origin_s = float(self.rng.uniform(
            earliest_origin, latest_origin))
        self._origin_error = self._sample_trace(self._trace_origin_s)
        if localization_pose is None:
            localization_pose = plant_pose + self._origin_error
            localization_pose[2] = math.atan2(
                math.sin(localization_pose[2]),
                math.cos(localization_pose[2]))
        self._simulation_start_s = float(time_s)
        self._plant_anchor = plant_pose.copy()
        self._localization_anchor = localization_pose.copy()
        self._yaw_anchor_delta = math.atan2(
            math.sin(localization_pose[2] - plant_pose[2]),
            math.cos(localization_pose[2] - plant_pose[2]))
        self._history_t = [float(time_s)]
        self._history_pose = [plant_pose.copy()]
        self._last_output = localization_pose.copy()

    @staticmethod
    def _interpolate_pose(times: np.ndarray, poses: np.ndarray,
                          query_s: float) -> np.ndarray | None:
        if query_s < times[0] or query_s > times[-1]:
            return None
        upper = int(np.searchsorted(times, query_s, side="right"))
        if upper == 0:
            return poses[0].copy()
        if upper >= len(times):
            return poses[-1].copy()
        lower = upper - 1
        ratio = (query_s - times[lower]) / (times[upper] - times[lower])
        yaw_delta = np.arctan2(
            np.sin(poses[upper, 2] - poses[lower, 2]),
            np.cos(poses[upper, 2] - poses[lower, 2]))
        return np.asarray((
            poses[lower, 0] + ratio * (poses[upper, 0] - poses[lower, 0]),
            poses[lower, 1] + ratio * (poses[upper, 1] - poses[lower, 1]),
            poses[lower, 2] + ratio * yaw_delta,
        ))

    def _sample_trace(self, trace_time_s: float) -> np.ndarray:
        value = self._interpolate_pose(
            self._run.time_s, self._run.residual_xyyaw, trace_time_s)
        if value is None:
            raise RuntimeError(
                f"AMCL residual query {trace_time_s:.6f}s is outside "
                f"training trace {self._run.run_id}")
        return value

    def step(self, time_s: float, plant_pose_xyyaw: np.ndarray
             ) -> np.ndarray | None:
        pose = np.asarray(plant_pose_xyyaw, dtype=np.float64)
        if pose.shape != (3,) or not np.isfinite(pose).all():
            raise ValueError("plant pose must be finite x/y/yaw")
        if not np.isfinite(time_s) or time_s <= self._history_t[-1]:
            raise ValueError("localization source time must strictly increase")
        self._history_t.append(float(time_s))
        self._history_pose.append(pose.copy())
        oldest = float(time_s) - self.history_seconds - self._latency_s - 0.1
        while len(self._history_t) > 2 and self._history_t[1] < oldest:
            self._history_t.pop(0)
            self._history_pose.pop(0)
        delayed = self._interpolate_pose(
            np.asarray(self._history_t), np.asarray(self._history_pose),
            float(time_s) - self._latency_s)
        if delayed is None:
            self._last_output = None
            return None
        residual_source_time = (
            self._trace_origin_s + float(time_s) - self._simulation_start_s
            - self._latency_s)
        residual = self._sample_trace(residual_source_time)
        residual = residual - self._origin_error
        yaw = self._yaw_anchor_delta
        c, s = math.cos(yaw), math.sin(yaw)
        delta = delayed[:2] - self._plant_anchor[:2]
        result = np.asarray((
            self._localization_anchor[0] + c * delta[0] - s * delta[1] + residual[0],
            self._localization_anchor[1] + s * delta[0] + c * delta[1] + residual[1],
            delayed[2] + yaw + residual[2],
        ), dtype=np.float64)
        result[2] = np.arctan2(np.sin(result[2]), np.cos(result[2]))
        self._last_output = result
        return result.copy()

    def get_state(self) -> np.ndarray | None:
        return None if self._last_output is None else self._last_output.copy()


def load_surrogate(path: Path, seed: int = 20261014
                   ) -> EmpiricalAmclSurrogate:
    archive = np.load(path, allow_pickle=False)
    run_ids = archive["run_ids"].astype(str)
    offsets = archive["offsets"].astype(np.int64)
    times = archive["time_s"].astype(np.float64)
    residuals = archive["residual_xyyaw"].astype(np.float64)
    latencies = archive["latency_s"].astype(np.float64)
    runs = tuple(LocalizationRun(
        str(run_id), times[offsets[index]:offsets[index + 1]],
        residuals[offsets[index]:offsets[index + 1]],
        float(latencies[index]))
        for index, run_id in enumerate(run_ids))
    return EmpiricalAmclSurrogate(runs, seed=seed)
