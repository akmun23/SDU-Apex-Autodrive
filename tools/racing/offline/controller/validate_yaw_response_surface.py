#!/usr/bin/env python3
"""Check C/Python yaw-surface parity and the production analytic Jacobian."""

from __future__ import annotations

import argparse
import ctypes
import math
from pathlib import Path
import sys

import numpy as np


class ModelState(ctypes.Structure):
    _fields_ = [(name, ctypes.c_float) for name in (
        "e_y", "e_psi", "u", "v", "r", "target_speed",
        "steering_command", "delayed_steering_command_1",
        "delayed_steering_command_2", "actual_steering_angle")]


class ModelControl(ctypes.Structure):
    _fields_ = [("steering_rate", ctypes.c_float),
                ("target_speed_rate", ctypes.c_float)]


class StageResult(ctypes.Structure):
    _fields_ = [("next", ModelState), ("delta_s_m", ctypes.c_float),
                ("body_accel_mps2", ctypes.c_float),
                ("branch_flags", ctypes.c_uint), ("valid", ctypes.c_int)]


class Linearization(ctypes.Structure):
    _fields_ = [
        ("A", (ctypes.c_float * 10) * 10),
        ("B", (ctypes.c_float * 2) * 10),
        ("d", ctypes.c_float * 10),
        ("nominal_branch_flags", ctypes.c_uint),
        ("nonsmooth_column_mask", ctypes.c_uint16),
        ("nonsmooth_column_count", ctypes.c_int),
        ("valid", ctypes.c_int),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library", type=Path)
    parser.add_argument("--repo", type=Path,
                        default=Path(__file__).resolve().parents[4])
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--trajectory", type=Path, default=None)
    parser.add_argument("--training-bag", type=Path, default=None)
    parser.add_argument("--independent-holdout-bag", type=Path, default=None)
    args = parser.parse_args()
    repo = args.repo.resolve()
    config = (args.config or repo / "f1tenth_mpc/config/mpc_competition.yaml").resolve()
    trajectory = (args.trajectory or repo /
        "f1tenth_planning/trajectories/autodrive_practice_20260924_b/"
        "autodrive_practice_20260924_b_mintime_raceline.csv").resolve()
    library_path = args.library.resolve()
    if not library_path.is_file():
        parser.error(f"library does not exist: {library_path}")

    tool_root = repo / "SDU_Apex_Autodrive_Exact_MinTime_Optimizer_v1_3"
    sys.path.insert(0, str(tool_root))
    sys.path.insert(1, str(repo))
    from f1tenth_planning.autodrive_mintime.model import VehicleModel, load_yaml
    from tools.racing.offline.controller.production_mpc import ProductionMpc
    from tools.evaluate_open_plane_speed_steering_surface import (
        _load_complete_speed_group_from_partial_run,
    )

    config_doc = load_yaml(repo /
        "SDU_Apex_Autodrive_Exact_MinTime_Optimizer_v1_3/f1tenth_planning/config/"
        "autodrive_mintime_exact.yaml")
    model = VehicleModel.from_repo(repo, config_doc)
    library = ctypes.CDLL(str(library_path))
    library.mpc_vehicle_model_step.argtypes = [
        ctypes.POINTER(ModelState), ctypes.POINTER(ModelControl),
        ctypes.c_float, ctypes.c_float]
    library.mpc_vehicle_model_step.restype = StageResult
    library.mpc_model_linearize.argtypes = [
        ctypes.POINTER(ModelState), ctypes.POINTER(ModelControl),
        ctypes.c_float, ctypes.c_float, ctypes.POINTER(Linearization)]
    library.mpc_model_linearize.restype = ctypes.c_int
    library.mpc_model_linearize_fd_oracle.argtypes = [
        ctypes.POINTER(ModelState), ctypes.POINTER(ModelControl),
        ctypes.c_float, ctypes.c_float, ctypes.POINTER(Linearization)]
    library.mpc_model_linearize_fd_oracle.restype = ctypes.c_int
    library.vehicle_model_steering_for_curvature_at_speed.argtypes = [
        ctypes.c_float, ctypes.c_float]
    library.vehicle_model_steering_for_curvature_at_speed.restype = ctypes.c_float

    with ProductionMpc(library_path, config, trajectory):
        yaw_errors = []
        dt = 0.025
        tau = 0.015
        retention = math.exp(-dt / tau)
        speeds = (2.96341375, 3.47, 3.97178625, 4.5,
                  4.98502375, 5.8)
        angles = (-0.50, -0.42, -0.30, -0.23, -0.21, -0.20, -0.15,
                  0.15, 0.20, 0.21, 0.23, 0.30, 0.42, 0.50)
        curvatures = (0.0, 0.65)
        for speed in speeds:
            target = float(model.steady_target_speed(speed))
            for angle in angles:
                for curvature in curvatures:
                    state = ModelState(0.0, 0.0, speed, 0.0, 0.0,
                        target, angle, angle, angle, angle)
                    control = ModelControl(0.0, 0.0)
                    stage = library.mpc_vehicle_model_step(
                        ctypes.byref(state), ctypes.byref(control), dt, curvature)
                    if not stage.valid:
                        raise RuntimeError("C vehicle model rejected a parity sample")
                    expected = (1.0 - retention) * model.steady_yaw_rate(
                        speed, angle, curvature)
                    yaw_errors.append(abs(float(stage.next.r) - expected))
        max_yaw_error = max(yaw_errors)
        if max_yaw_error > 2.0e-5:
            raise AssertionError(f"C/Python response mismatch: {max_yaw_error:.3g} rad/s")

        feedforward_errors = []
        for speed, curvature in ((3.0, 0.35), (4.2, 0.65), (5.0, -0.45),
                                 (6.5, 0.90)):
            c_steering = float(library.vehicle_model_steering_for_curvature_at_speed(
                curvature, speed))
            py_steering = model.steering_for_yaw_rate(
                speed, curvature * speed, curvature)
            feedforward_errors.append(abs(c_steering - py_steering))
        max_feedforward_error = max(feedforward_errors)
        if max_feedforward_error > 0.012:
            raise AssertionError(
                f"C/Python speed-aware feed-forward mismatch: {max_feedforward_error:.4f} rad")

        jacobian_errors = []
        for speed, steering, curvature in (
                (3.4, math.atan(0.70 / 3.4), 0.55),
                (4.3, math.atan(0.73 / 4.3), -0.60),
                (4.7, 0.30, 0.25),
                (5.2, 0.42, -0.35)):
            target = float(model.steady_target_speed(speed))
            state = ModelState(0.025, -0.035, speed, 0.08, -0.55,
                target, steering, steering, steering, steering)
            control = ModelControl(0.0, 0.0)
            analytic = Linearization()
            oracle = Linearization()
            if not library.mpc_model_linearize(
                    ctypes.byref(state), ctypes.byref(control), dt, curvature,
                    ctypes.byref(analytic)):
                raise RuntimeError("analytic MPC linearization failed")
            if not library.mpc_model_linearize_fd_oracle(
                    ctypes.byref(state), ctypes.byref(control), dt, curvature,
                    ctypes.byref(oracle)):
                raise RuntimeError("finite-difference MPC Jacobian oracle failed")
            a = np.ctypeslib.as_array(analytic.A).reshape(10, 10)
            b = np.ctypeslib.as_array(analytic.B).reshape(10, 2)
            a_fd = np.ctypeslib.as_array(oracle.A).reshape(10, 10)
            b_fd = np.ctypeslib.as_array(oracle.B).reshape(10, 2)
            a_diff = np.abs(a - a_fd)
            b_diff = np.abs(b - b_fd)
            error = max(float(np.max(a_diff)), float(np.max(b_diff)))
            if np.max(a_diff) >= np.max(b_diff):
                row, col = np.unravel_index(np.argmax(a_diff), a_diff.shape)
                location = f"A[{row},{col}]={a[row, col]:.5g}/{a_fd[row, col]:.5g}"
            else:
                row, col = np.unravel_index(np.argmax(b_diff), b_diff.shape)
                location = f"B[{row},{col}]={b[row, col]:.5g}/{b_fd[row, col]:.5g}"
            jacobian_errors.append(error)
            print(f"Jacobian probe u={speed:.3f} d={steering:.3f} "
                  f"k={curvature:.3f}: max={error:.4g} at {location}")
        max_jacobian_error = max(jacobian_errors)
        if max_jacobian_error > 0.035:
            raise AssertionError(
                f"analytic/FD Jacobian discrepancy: {max_jacobian_error:.4f}")

        training_bag = (args.training_bag or repo /
            "live_runs/openplane_rootless_3to5_surface_20260927_01/run/run_0.db3").resolve()
        holdout_bag = (args.independent_holdout_bag or repo /
            "live_runs/openplane_isolated_highspeed_surface_20260927/run/run_0.db3").resolve()
        if training_bag.is_file() and holdout_bag.is_file():
            heldout_groups = []
            for speed in (3.0, 4.0, 5.0):
                blocks = _load_complete_speed_group_from_partial_run(
                    training_bag, speed, repetitions=(3,))
                heldout_groups.extend(blocks)
            independent = _load_complete_speed_group_from_partial_run(
                holdout_bag, 4.5)
            heldout_groups.extend(independent)
            by_run: dict[str, list[tuple[float, float, float]]] = {}
            for block in heldout_groups:
                actual = block.lateral_acceleration_mps2 / block.measured_forward_speed_mps
                predicted = model.steady_yaw_rate(
                    block.measured_forward_speed_mps, block.steering_rad)
                baseline = model._legacy_yaw_rate_numeric(
                    block.measured_forward_speed_mps, block.steering_rad, 0.0)
                run = ("independent_4p5" if abs(block.target_speed_mps - 4.5) < 1.0e-6
                       else f"rep3_{block.target_speed_mps:.0f}mps")
                by_run.setdefault(run, []).append((actual, predicted, baseline))

            print("held-out steady yaw-rate response from whole open-plane repetitions:")
            for run, values in sorted(by_run.items()):
                data = np.asarray(values, dtype=float)
                high = np.abs(np.asarray([
                    block.steering_rad for block in heldout_groups
                    if (("independent_4p5" if abs(block.target_speed_mps - 4.5) < 1.0e-6
                         else f"rep3_{block.target_speed_mps:.0f}mps") == run)
                ])) >= 0.30
                model_rmse = float(np.sqrt(np.mean((data[:, 1] - data[:, 0]) ** 2)))
                base_rmse = float(np.sqrt(np.mean((data[:, 2] - data[:, 0]) ** 2)))
                high_model_rmse = float(np.sqrt(np.mean((data[high, 1] - data[high, 0]) ** 2)))
                high_base_rmse = float(np.sqrt(np.mean((data[high, 2] - data[high, 0]) ** 2)))
                print(f"  {run}: yaw RMSE {model_rmse:.4f} vs legacy {base_rmse:.4f} rad/s; "
                      f"|steer|>=0.30: {high_model_rmse:.4f} vs {high_base_rmse:.4f}")

    print(f"C/Python yaw response max |error|: {max_yaw_error:.3e} rad/s "
          f"({len(yaw_errors)} states)")
    print(f"C/Python speed-aware steering feed-forward max |error|: "
          f"{max_feedforward_error:.4f} rad")
    print(f"analytic vs finite-difference Jacobian max |error|: "
          f"{max_jacobian_error:.3e} ({len(jacobian_errors)} high-steer states)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
