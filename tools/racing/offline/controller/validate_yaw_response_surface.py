#!/usr/bin/env python3
"""Check C/Python yaw-surface parity and the production analytic Jacobian."""

from __future__ import annotations

import argparse
import ctypes
import math
from pathlib import Path
import sys
from contextlib import ExitStack
import tempfile

import yaml
import casadi as ca

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
    parser.add_argument("--overlay-config", type=Path, default=None,
                        help="ROS parameter overlay merged on top of --config")
    parser.add_argument("--optimizer-config", type=Path, default=None,
                        help="optimizer config used to load the matching Python plant model")
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

    optimizer_config = (args.optimizer_config or repo /
        "SDU_Apex_Autodrive_Exact_MinTime_Optimizer_v1_3/f1tenth_planning/config/"
        "autodrive_mintime_exact.yaml").resolve()
    config_doc = load_yaml(optimizer_config)
    model = VehicleModel.from_repo(repo, config_doc)
    casadi_speed = ca.MX.sym("yaw_surface_speed")
    casadi_steering = ca.MX.sym("yaw_surface_steering")
    casadi_yaw = ca.Function("yaw_surface_candidate_parity",
        [casadi_speed, casadi_steering],
        [model.steady_yaw_rate_casadi(
            casadi_speed, casadi_steering, ca.MX(0.0))])
    runtime_doc = load_yaml(config)
    runtime_params = dict(next(iter(runtime_doc.values())).get("ros__parameters", {}))
    if args.overlay_config is not None:
        overlay_doc = load_yaml(args.overlay_config.resolve())
        overlay_params = next(iter(overlay_doc.values())).get("ros__parameters", {})
        runtime_params.update(overlay_params)
        runtime_doc = {"/**": {"ros__parameters": runtime_params}}

    def smoothstep(value: float, low: float, high: float) -> float:
        t = min(max((value - low) / (high - low), 0.0), 1.0)
        return t * t * (3.0 - 2.0 * t)

    def expected_surface_blend(speed: float, steering: float) -> float:
        params = runtime_params
        demand_q = max(speed, 0.0) * abs(math.tan(steering))
        q_blend = smoothstep(
            demand_q,
            float(params.get("yaw_rate_response_surface_blend_q_start", 0.60)),
            float(params.get("yaw_rate_response_surface_blend_q_end", 0.85)))
        steer_start = float(params.get(
            "yaw_rate_response_surface_steering_blend_start_rad", 0.0))
        steer_full_start = float(params.get(
            "yaw_rate_response_surface_steering_blend_full_start_rad", 0.0))
        steer_full_end = float(params.get(
            "yaw_rate_response_surface_steering_blend_full_end_rad", 0.0))
        steer_end = float(params.get(
            "yaw_rate_response_surface_steering_blend_end_rad", 0.0))
        if any((steer_start, steer_full_start, steer_full_end, steer_end)):
            steer_blend = smoothstep(abs(steering), steer_start, steer_full_start)
            steer_blend *= 1.0 - smoothstep(abs(steering), steer_full_end, steer_end)
        else:
            steer_blend = 1.0
        low = float(model.yaw_surface_speed_mps[0])
        high = float(model.yaw_surface_speed_mps[-1])
        low_margin = float(params.get(
            "yaw_rate_response_surface_low_speed_blend_margin_mps", 0.0))
        high_margin = float(params.get(
            "yaw_rate_response_surface_speed_blend_margin_mps", 0.0))
        speed_blend = (smoothstep(speed, low - low_margin, low)
                       if low_margin > 0.0 else 1.0)
        if high_margin > 0.0:
            speed_blend *= 1.0 - smoothstep(speed, high, high + high_margin)
        support = 1.0
        low_fade = float(params.get(
            "yaw_rate_response_surface_low_speed_support_fadeout_mps", 0.0))
        high_fade = float(params.get(
            "yaw_rate_response_surface_high_speed_support_fadein_mps", 0.0))
        if low_fade > 0.0 or high_fade > 0.0:
            low_t = min(max((speed - low) / low_fade, 0.0), 1.0) \
                if low_fade > 0.0 else 0.0
            high_t = min(max((speed - (model.yaw_surface_speed_mps[2] - high_fade)) /
                             high_fade, 0.0), 1.0) if high_fade > 0.0 else 0.0
            support = (1.0 - low_t * low_t * (3.0 - 2.0 * low_t)
                       + high_t * high_t * (3.0 - 2.0 * high_t))
        return (float(params.get("yaw_rate_response_surface_response_scale", 1.0))
                * q_blend * steer_blend * speed_blend * support)

    def expected_response_tau(speed: float, steering: float,
                              steering_rate: float = 0.0) -> float:
        params = runtime_params
        tau = float(params.get("yaw_rate_response_time_constant_s", 0.015))
        low_tau = float(params.get("yaw_rate_low_speed_response_time_constant_s", 0.0))
        if low_tau > 0.0:
            transition = float(params.get(
                "yaw_rate_low_speed_transition_speed_mps", 1.0))
            tau += (low_tau - tau) * math.exp(-max(speed, 0.0) / transition)
        fitted_tau = float(params.get(
            "yaw_rate_response_surface_hold_time_constant_s", 0.0))
        if fitted_tau <= 0.0:
            return tau
        rate_full = float(params.get(
            "yaw_rate_response_surface_hold_rate_full_radps", 0.0))
        rate_zero = float(params.get(
            "yaw_rate_response_surface_hold_rate_zero_radps", 0.0))
        hold_weight = (1.0 - smoothstep(abs(steering_rate), rate_full, rate_zero)
                       if rate_zero > rate_full else 1.0)
        return tau + expected_surface_blend(speed, steering) * hold_weight * (
            fitted_tau - tau)
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

    with ExitStack() as stack:
        runtime_config_path = config
        if args.overlay_config is not None:
            temp_dir = stack.enter_context(tempfile.TemporaryDirectory(
                prefix="yaw-surface-parity-", dir=library_path.parent))
            runtime_config_path = Path(temp_dir) / "merged_mpc.yaml"
            runtime_config_path.write_text(
                yaml.safe_dump(runtime_doc, sort_keys=False), encoding="utf-8")
        stack.enter_context(ProductionMpc(
            library_path, runtime_config_path, trajectory))
        below_boundary_errors = []
        casadi_errors = []
        for speed in (2.5, 3.0, 3.5):
            for angle in (-0.30, -0.29, -0.25, -0.20, -0.15,
                          0.15, 0.20, 0.25, 0.29, 0.30):
                candidate = model.steady_yaw_rate(speed, angle)
                baseline = model._legacy_yaw_rate_numeric(speed, angle, 0.0)
                if abs(angle) <= 0.29:
                    below_boundary_errors.append(abs(candidate - baseline))
                casadi_value = float(casadi_yaw(speed, angle))
                casadi_errors.append(abs(candidate - casadi_value))
        if max(below_boundary_errors) > 1.0e-12:
            raise AssertionError(
                "candidate changes the legacy yaw response at/below 0.29 rad")
        max_casadi_error = max(casadi_errors)
        if max_casadi_error > 2.0e-5:
            raise AssertionError(
                f"Python/CasADi response mismatch: {max_casadi_error:.3g} rad/s")
        yaw_errors = []
        worst_yaw_probe = None
        dt = 0.025
        speeds = (2.40, 2.70, 2.96341375, 3.373, 3.47, 3.97178625, 4.5,
                  4.98502375, 5.8)
        angles = (-0.50, -0.42, -0.40, -0.38, -0.30, -0.23, -0.21, -0.20, -0.15,
                  0.15, 0.20, 0.21, 0.23, 0.30, 0.38, 0.40, 0.42, 0.50)
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
                    # The production stage evaluates yaw at midpoint body
                    # speed after applying the longitudinal model. Compare
                    # like-for-like rather than using the input speed.
                    speed_mid = 0.5 * (speed + float(stage.next.u))
                    effective_tau = expected_response_tau(speed_mid, angle)
                    expected = (1.0 - math.exp(-dt / effective_tau)) * model.steady_yaw_rate(
                        speed_mid, angle, curvature)
                    error = abs(float(stage.next.r) - expected)
                    yaw_errors.append(error)
                    if worst_yaw_probe is None or error > worst_yaw_probe[0]:
                        worst_yaw_probe = (error, speed, angle, curvature,
                                           float(stage.next.r), expected)
        phase_gate_errors = []
        for steering in (-0.35, 0.35):
            for steering_rate in (0.0, 0.10, 0.20):
                speed = 3.2
                target = float(model.steady_target_speed(speed))
                signed_rate = math.copysign(steering_rate, steering)
                state = ModelState(0.0, 0.0, speed, 0.0, 0.0,
                    target, steering, steering + signed_rate * dt,
                    steering, steering)
                control = ModelControl(0.0, 0.0)
                stage = library.mpc_vehicle_model_step(
                    ctypes.byref(state), ctypes.byref(control), dt, 0.0)
                if not stage.valid:
                    raise RuntimeError("C vehicle model rejected a steering-phase probe")
                speed_mid = 0.5 * (speed + float(stage.next.u))
                steering_next = steering + signed_rate * dt
                hold_weight = (1.0 - smoothstep(
                    steering_rate,
                    float(runtime_params.get(
                        "yaw_rate_response_surface_hold_rate_full_radps", 0.0)),
                    float(runtime_params.get(
                        "yaw_rate_response_surface_hold_rate_zero_radps", 0.0)))
                    if float(runtime_params.get(
                        "yaw_rate_response_surface_hold_rate_zero_radps", 0.0)) >
                        float(runtime_params.get(
                            "yaw_rate_response_surface_hold_rate_full_radps", 0.0))
                    else 1.0)
                mapped_steady = model.steady_yaw_rate(
                    speed_mid, steering_next, 0.0)
                legacy_steady = model._legacy_yaw_rate_numeric(
                    speed_mid, steering_next, 0.0)
                expected_steady = legacy_steady + hold_weight * (
                    mapped_steady - legacy_steady)
                effective_tau = expected_response_tau(
                    speed_mid, steering_next, steering_rate)
                expected = (1.0 - math.exp(-dt / effective_tau)) * expected_steady
                phase_gate_errors.append(abs(float(stage.next.r) - expected))
        max_phase_gate_error = max(phase_gate_errors)
        if max_phase_gate_error > 2.0e-5:
            raise AssertionError(
                "C steering-phase map/tau gate mismatch: "
                f"{max_phase_gate_error:.3g} rad/s")
        max_yaw_error = max(yaw_errors)
        if max_yaw_error > 2.0e-5:
            raise AssertionError(
                "C/Python response mismatch: "
                f"{max_yaw_error:.3g} rad/s at "
                f"u={worst_yaw_probe[1]:.3f}, delta={worst_yaw_probe[2]:.3f}, "
                f"k={worst_yaw_probe[3]:.3f}; C={worst_yaw_probe[4]:.6g}, "
                f"Python={worst_yaw_probe[5]:.6g}")

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
                (5.2, 0.42, -0.35),
                (3.2, 0.31, 0.0),
                (3.2, 0.40, 0.0),
                (3.2, 0.43, 0.0),
                (3.2, 0.44, 0.0),
                ):
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
    print(f"Python/CasADi yaw response max |error|: "
          f"{max_casadi_error:.3e} rad/s; baseline unchanged through 0.29 rad")
    print(f"C hold/turn-in map and tau gate max |error|: "
          f"{max_phase_gate_error:.3e} rad/s")
    print(f"C/Python speed-aware steering feed-forward max |error|: "
          f"{max_feedforward_error:.4f} rad")
    print(f"analytic vs finite-difference Jacobian max |error|: "
          f"{max_jacobian_error:.3e} ({len(jacobian_errors)} high-steer states)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
