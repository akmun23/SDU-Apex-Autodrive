"""Python binding to the exact production MPC cycle."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path

import numpy as np


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
        return float(self.library.offline_mpc_lap_length(self._handle))

    def reset(self) -> None:
        self.library.offline_mpc_reset(self._handle)

    def project(self, pose_xyyaw: np.ndarray,
                previous_segment: int = (2**64 - 1),
                local_search_radius: int = 160) -> tuple[float, float, float,
                                                          float, int]:
        pose = np.asarray(pose_xyyaw, dtype=np.float64)
        if pose.shape != (3,) or not np.isfinite(pose).all():
            raise ValueError("map pose must be finite x/y/yaw")
        output = (ctypes.c_double * 5)()
        ok = self.library.offline_mpc_project(
            self._handle, float(pose[0]), float(pose[1]), float(pose[2]),
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
        output = (ctypes.c_double * 16)()
        ok = self.library.offline_mpc_step(
            self._handle, state_buffer, float(progress_m),
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
        )

    def close(self) -> None:
        if getattr(self, "_handle", None):
            self.library.offline_mpc_destroy(self._handle)
            self._handle = None

    def __enter__(self) -> "ProductionMpc":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
