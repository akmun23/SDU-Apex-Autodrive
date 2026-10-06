"""Python binding to the exact production MPC cycle."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class NonlinearRolloutFailure:
    valid: bool
    stage: int
    progress_m: float
    state: tuple[float, ...]
    reference: tuple[float, ...]
    margin_m: float
    lower_bound_m: float
    upper_bound_m: float


@dataclass(frozen=True)
class MpcCycle:
    status: int
    steering_command_rad: float
    target_speed_mps: float
    steering_rate_radps: float
    target_speed_rate_mps2: float
    solver_iterations: int
    primal_residual: float
    dual_residual: float
    maximum_regularization: float
    nonlinear_failure_stage: int
    nonlinear_failure_reason: int
    best_effort_action: bool
    residual_candidate: bool
    rejection_speed_guard: bool
    rti_iterations: int
    rti2_triggered: bool
    r1_failure: NonlinearRolloutFailure
    r2_failure: NonlinearRolloutFailure


_FAILURE_DIAGNOSTIC_COUNT = 29
_FAILURE_STATE_COUNT = 12
_FAILURE_REFERENCE_COUNT = 11


def _failure_from_output(output: np.ndarray, offset: int) -> NonlinearRolloutFailure:
    values = output[offset:offset + _FAILURE_DIAGNOSTIC_COUNT]
    return NonlinearRolloutFailure(
        valid=bool(values[0]),
        stage=int(values[1]),
        progress_m=float(values[2]),
        state=tuple(map(float, values[3:3 + _FAILURE_STATE_COUNT])),
        reference=tuple(map(float, values[15:15 + _FAILURE_REFERENCE_COUNT])),
        margin_m=float(values[26]),
        lower_bound_m=float(values[27]),
        upper_bound_m=float(values[28]),
    )


class ProductionMpc:
    """Invoke the repository's MPC RTI core with its deployed YAML/trajectory."""

    def __init__(self, library: Path, config_yaml: Path,
                 trajectory_csv: Path) -> None:
        self.library = ctypes.CDLL(str(library.resolve()))
        self.library.offline_mpc_create.argtypes = [
            ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t]
        self.library.offline_mpc_create.restype = ctypes.c_void_p
        self.library.offline_mpc_destroy.argtypes = [ctypes.c_void_p]
        self.library.offline_mpc_reset.argtypes = [ctypes.c_void_p]
        self.library.offline_mpc_step.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_double,
            ctypes.c_double, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
        self.library.offline_mpc_step.restype = ctypes.c_int
        self.library.offline_mpc_project.argtypes = [
            ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.c_double,
            ctypes.c_size_t, ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
        self.library.offline_mpc_project.restype = ctypes.c_int
        self.library.offline_mpc_lap_length.argtypes = [ctypes.c_void_p]
        self.library.offline_mpc_lap_length.restype = ctypes.c_double
        error = ctypes.create_string_buffer(2048)
        self._handle = self.library.offline_mpc_create(
            str(config_yaml.resolve()).encode(),
            str(trajectory_csv.resolve()).encode(), error, len(error))
        if not self._handle:
            raise ValueError(error.value.decode(errors="replace"))

    @property
    def lap_length_m(self) -> float:
        return float(self.library.offline_mpc_lap_length(self._require_open()))

    def _require_open(self) -> int:
        handle = self._handle
        if not handle:
            raise RuntimeError("ProductionMpc is closed")
        return handle

    def reset(self) -> None:
        self.library.offline_mpc_reset(self._require_open())

    def project(self, pose_xyyaw: np.ndarray,
                previous_segment: int = (2**64 - 1),
                local_search_radius: int = 160) -> tuple[float, float, float,
                                                          float, int]:
        pose = np.asarray(pose_xyyaw, dtype=np.float64)
        if pose.shape != (3,) or not np.isfinite(pose).all():
            raise ValueError("map pose must be finite x/y/yaw")
        output = (ctypes.c_double * 5)()
        ok = self.library.offline_mpc_project(
            self._require_open(), float(pose[0]), float(pose[1]), float(pose[2]),
            int(previous_segment), int(local_search_radius), output, len(output))
        if not ok or not np.isfinite(np.asarray(output)[:4]).all():
            raise ValueError("production MPC could not project map pose")
        return (*map(float, output[:4]), int(output[4]))

    def step(self, state: np.ndarray, progress_m: float,
             speed_ceiling_mps: float) -> MpcCycle:
        values = np.asarray(state, dtype=np.float64)
        if values.shape != (12,) or not np.isfinite(values).all():
            raise ValueError("MPC state must be 12 finite production channels")
        state_buffer = (ctypes.c_double * 12)(*values)
        output = (ctypes.c_double * (16 + 2 * _FAILURE_DIAGNOSTIC_COUNT))()
        ok = self.library.offline_mpc_step(
            self._require_open(), state_buffer, float(progress_m),
            float(speed_ceiling_mps), output, len(output))
        if not ok:
            raise ValueError("production MPC rejected invalid offline inputs")
        if not np.isfinite(np.asarray(output)).all():
            raise FloatingPointError("production MPC emitted non-finite diagnostics")
        return MpcCycle(
            status=int(output[0]),
            steering_command_rad=float(output[1]),
            target_speed_mps=float(output[2]),
            steering_rate_radps=float(output[3]),
            target_speed_rate_mps2=float(output[4]),
            solver_iterations=int(output[5]),
            primal_residual=float(output[6]),
            dual_residual=float(output[7]),
            maximum_regularization=float(output[8]),
            nonlinear_failure_stage=int(output[9]),
            nonlinear_failure_reason=int(output[10]),
            best_effort_action=bool(output[11]),
            residual_candidate=bool(output[12]),
            rejection_speed_guard=bool(output[13]),
            rti_iterations=int(output[14]),
            rti2_triggered=bool(output[15]),
            r1_failure=_failure_from_output(output, 16),
            r2_failure=_failure_from_output(
                output, 16 + _FAILURE_DIAGNOSTIC_COUNT),
        )

    def close(self) -> None:
        if getattr(self, "_handle", None):
            self.library.offline_mpc_destroy(self._handle)
            self._handle = None

    def __enter__(self) -> "ProductionMpc":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
