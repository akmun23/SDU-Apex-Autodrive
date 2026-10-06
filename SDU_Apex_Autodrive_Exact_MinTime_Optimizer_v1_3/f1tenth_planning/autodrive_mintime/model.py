from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import csv
import math
import re
from typing import Any

import casadi as ca
import numpy as np
import yaml


_REQUIRED_MACROS = (
    "SOURCE_MAX_STEERING_RAD",
    "SOURCE_STEERING_RATE_RADPS",
    "MPC_YAW_RATE_RESPONSE_TIME_CONSTANT_SECONDS",
    "MPC_YAW_RATE_STEERING_GAIN_PER_M",
    "MPC_LONGITUDINAL_RESPONSE_BIAS_MPS2",
    "MPC_LONGITUDINAL_SPEED_COEFF_PER_S",
    "MPC_LONGITUDINAL_TARGET_ERROR_GAIN_PER_S",
    "MPC_LONGITUDINAL_TARGET_RATE_COEFF",
    "MPC_LONGITUDINAL_ACCEL_LIMIT_MPS2",
    "MPC_LONGITUDINAL_BRAKE_DECEL_INTERCEPT_MPS2",
    "MPC_LONGITUDINAL_BRAKE_DECEL_SLOPE_S_INV",
    "MPC_MAX_COMMAND_SPEED_MPS",
    "MPC_TARGET_SPEED_RATE_INCREASE_MAX_MPS2",
    "MPC_TARGET_SPEED_RATE_REDUCTION_MAX_MPS2",
)


def _parse_simple_c_number(expr: str) -> float:
    expr = expr.split("/*", 1)[0].split("//", 1)[0].strip()
    expr = expr.strip("() ")
    expr = re.sub(r"(?<=\d)[fF]\b", "", expr)
    if not re.fullmatch(r"[+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+\-]?\d+)?", expr):
        raise ValueError(f"Unsupported macro expression: {expr!r}")
    return float(expr)


def read_mpc_constants(header_path: str | Path) -> dict[str, float]:
    text = Path(header_path).read_text(encoding="utf-8")
    out: dict[str, float] = {}
    for name in _REQUIRED_MACROS:
        m = re.search(rf"^\s*#define\s+{re.escape(name)}\s+(.+?)\s*$", text, re.MULTILINE)
        if not m:
            raise KeyError(f"Missing required MPC macro {name} in {header_path}")
        out[name] = _parse_simple_c_number(m.group(1))
    return out


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _ros_params(doc: dict[str, Any]) -> dict[str, Any]:
    # Current path_tracking_autodrive.yaml uses pure_pursuit_node -> ros__parameters,
    # but accept the common /** layout too.
    for key in ("pure_pursuit_node", "/**"):
        block = doc.get(key)
        if isinstance(block, dict) and isinstance(block.get("ros__parameters"), dict):
            return block["ros__parameters"]
    if isinstance(doc.get("ros__parameters"), dict):
        return doc["ros__parameters"]
    raise ValueError("Could not find ros__parameters in path tracking config")


def _close(a: float, b: float, tol: float = 1.0e-6) -> bool:
    return abs(float(a) - float(b)) <= tol


def _pchip_slopes(x: tuple[float, ...], y: tuple[float, ...]) -> tuple[float, ...]:
    h = [x[i + 1] - x[i] for i in range(len(x) - 1)]
    d = [(y[i + 1] - y[i]) / h[i] for i in range(len(h))]

    def endpoint(h0: float, h1: float, d0: float, d1: float) -> float:
        slope = ((2.0 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
        if slope * d0 <= 0.0:
            return 0.0
        if d0 * d1 < 0.0 and abs(slope) > 3.0 * abs(d0):
            return 3.0 * d0
        return slope

    slopes = [0.0] * len(x)
    slopes[0] = endpoint(h[0], h[1], d[0], d[1])
    slopes[-1] = endpoint(h[-1], h[-2], d[-1], d[-2])
    for i in range(1, len(x) - 1):
        left, right = d[i - 1], d[i]
        if left * right <= 0.0:
            slopes[i] = 0.0
        else:
            w_left = 2.0 * h[i] + h[i - 1]
            w_right = h[i] + 2.0 * h[i - 1]
            slopes[i] = (w_left + w_right) / (w_left / left + w_right / right)
    return tuple(slopes)


def _pchip_value(x: tuple[float, ...], y: tuple[float, ...], query: float) -> float:
    if query < x[0]:
        return y[0]
    if query > x[-1]:
        return y[-1]
    slopes = _pchip_slopes(x, y)
    interval = next((i for i in range(len(x) - 1) if query <= x[i + 1]), len(x) - 2)
    width = x[interval + 1] - x[interval]
    t = (query - x[interval]) / width
    t2, t3 = t * t, t * t * t
    return (
        (2.0 * t3 - 3.0 * t2 + 1.0) * y[interval]
        + (t3 - 2.0 * t2 + t) * width * slopes[interval]
        + (-2.0 * t3 + 3.0 * t2) * y[interval + 1]
        + (t3 - t2) * width * slopes[interval + 1]
    )


def _pchip_casadi(x: tuple[float, ...], y: tuple[float, ...], query):
    slopes = _pchip_slopes(x, y)
    result = y[-1]
    for i in reversed(range(len(x) - 1)):
        width = x[i + 1] - x[i]
        t = (query - x[i]) / width
        t2, t3 = t * t, t * t * t
        segment = (
            (2.0 * t3 - 3.0 * t2 + 1.0) * y[i]
            + (t3 - 2.0 * t2 + t) * width * slopes[i]
            + (-2.0 * t3 + 3.0 * t2) * y[i + 1]
            + (t3 - t2) * width * slopes[i + 1]
        )
        result = ca.if_else(query <= x[i + 1], segment, result)
    return ca.if_else(query < x[0], y[0], ca.if_else(query > x[-1], y[-1], result))


@dataclass(frozen=True)
class VehicleModel:
    # Empirically identified command->vehicle model used by the MPC.
    max_steering_rad: float
    max_steering_rate_radps: float
    yaw_tau_s: float
    yaw_gain_per_m: float
    yaw_gain_reduction_per_rad: float
    yaw_gain_start_rad: float
    yaw_gain_end_rad: float
    yaw_curvature_gain_reduction_per_m: float
    yaw_curvature_gain_start_per_m: float
    yaw_curvature_gain_end_per_m: float
    yaw_surface_enabled: bool
    yaw_surface_response_scale: float
    yaw_surface_blend_q_start: float
    yaw_surface_blend_q_end: float
    yaw_surface_speed_blend_margin_mps: float
    yaw_surface_low_speed_blend_margin_mps: float
    yaw_surface_low_speed_support_fadeout_mps: float
    yaw_surface_high_speed_support_fadein_mps: float
    yaw_surface_speed_mps: tuple[float, ...]
    yaw_surface_q: tuple[tuple[tuple[float, ...], ...], ...]
    yaw_surface_rate_rps: tuple[tuple[tuple[float, ...], ...], ...]
    longitudinal_bias_mps2: float
    longitudinal_speed_coeff_per_s: float
    longitudinal_target_gain_per_s: float
    longitudinal_target_rate_coeff: float
    accel_limit_mps2: float
    brake_intercept_mps2: float
    brake_slope_s_inv: float
    max_command_speed_mps: float
    max_target_speed_rate_increase_mps2: float
    max_target_speed_rate_reduction_mps2: float

    # Actual AutoDRIVE geometry from f1tenth_planning/config/autodrive_sim_vehicle.yaml.
    car_width_m: float
    car_length_m: float
    rear_overhang_m: float
    rear_axle_to_front_bumper_m: float

    # Safety corridor semantics mirrored from optimize_trajectory.py / vehicle profile.
    planning_footprint_width_m: float
    required_wall_clearance_m: float

    # Current validated controller/runtime operating envelope.
    validated_lateral_accel_mps2: float

    # Sensor-only rear-axle lateral-velocity model used by production odometry.
    # The optimizer enables it only in explicit candidate configs; its
    # historical v=0 behavior remains the default.
    lateral_velocity_yaw_rate_gain_m: float
    lateral_velocity_speed_yaw_rate_gain_s: float
    lateral_velocity_max_mps: float
    lateral_velocity_reference_forward_offset_m: float

    # Numerical lower bound only; not a simulator physical constant.
    min_speed_mps: float
    max_body_speed_mps: float

    @classmethod
    def from_repo(cls, repo_root: str | Path, config: dict[str, Any]) -> "VehicleModel":
        repo_root = Path(repo_root).resolve()
        header = repo_root / "f1tenth_mpc/include/mpc_types.h"
        mpc_yaml_path = repo_root / "f1tenth_mpc/config/mpc_competition.yaml"
        surface_csv_path = repo_root / "f1tenth_mpc/config/yaw_response_surface.csv"
        sim_profile_path = repo_root / "f1tenth_planning/config/autodrive_sim_vehicle.yaml"
        pp_path = repo_root / "f1tenth_control/config/path_tracking_autodrive.yaml"
        odom_path = repo_root / "f1tenth_localization/config/sensor_odometry.yaml"

        constants = read_mpc_constants(header)
        sim_profile = _load_yaml(sim_profile_path)
        pp_doc = _load_yaml(pp_path)
        pp = _ros_params(pp_doc)
        mpc_params = _ros_params(_load_yaml(mpc_yaml_path))
        vehicle_model_overrides = config.get("vehicle_model_overrides", {})
        if not isinstance(vehicle_model_overrides, dict):
            raise ValueError("vehicle_model_overrides must be a mapping")
        supported_model_overrides = {
            "yaw_response_time_constant_s",
            "yaw_gain_reduction_per_rad",
            "yaw_gain_start_rad",
            "yaw_gain_end_rad",
            "yaw_surface_enabled",
            "yaw_surface_response_scale",
            "yaw_surface_csv",
            "yaw_surface_blend_q_start",
            "yaw_surface_blend_q_end",
            "yaw_surface_speed_blend_margin_mps",
            "yaw_surface_low_speed_blend_margin_mps",
            "yaw_surface_low_speed_support_fadeout_mps",
            "yaw_surface_high_speed_support_fadein_mps",
        }
        unknown_model_overrides = set(vehicle_model_overrides) - supported_model_overrides
        if unknown_model_overrides:
            raise ValueError(
                "unsupported vehicle_model_overrides: "
                + ", ".join(sorted(unknown_model_overrides)))
        odom_doc = _load_yaml(odom_path)
        odom_node = odom_doc.get("sensor_odometry", {})
        odom_params = (odom_node.get("ros__parameters", {})
                       if isinstance(odom_node, dict) else {})
        if not isinstance(odom_params, dict):
            raise ValueError(f"Could not find sensor_odometry.ros__parameters in {odom_path}")

        surface_override = vehicle_model_overrides.get("yaw_surface_enabled")
        if surface_override is not None and not isinstance(surface_override, bool):
            raise ValueError("vehicle_model_overrides.yaw_surface_enabled must be boolean")
        surface_enabled = (
            surface_override if surface_override is not None else
            bool(mpc_params.get("yaw_rate_response_surface_enabled", False))
        )
        surface_response_scale = float(vehicle_model_overrides.get(
            "yaw_surface_response_scale", 1.0))
        surface_blend_q_start = float(vehicle_model_overrides.get(
            "yaw_surface_blend_q_start",
            mpc_params.get("yaw_rate_response_surface_blend_q_start", 0.60)))
        surface_blend_q_end = float(vehicle_model_overrides.get(
            "yaw_surface_blend_q_end",
            mpc_params.get("yaw_rate_response_surface_blend_q_end", 0.85)))
        surface_speed_blend_margin = float(vehicle_model_overrides.get(
            "yaw_surface_speed_blend_margin_mps",
            mpc_params.get("yaw_rate_response_surface_speed_blend_margin_mps", 0.50)))
        surface_low_speed_blend_margin = float(vehicle_model_overrides.get(
            "yaw_surface_low_speed_blend_margin_mps",
            mpc_params.get("yaw_rate_response_surface_low_speed_blend_margin_mps",
                           surface_speed_blend_margin)))
        surface_low_speed_support_fadeout = float(vehicle_model_overrides.get(
            "yaw_surface_low_speed_support_fadeout_mps",
            mpc_params.get("yaw_rate_response_surface_low_speed_support_fadeout_mps", 0.0)))
        surface_high_speed_support_fadein = float(vehicle_model_overrides.get(
            "yaw_surface_high_speed_support_fadein_mps",
            mpc_params.get("yaw_rate_response_surface_high_speed_support_fadein_mps", 0.0)))
        surface_csv_override = vehicle_model_overrides.get("yaw_surface_csv")
        if surface_csv_override is not None:
            if not isinstance(surface_csv_override, str) or not surface_csv_override.strip():
                raise ValueError("vehicle_model_overrides.yaw_surface_csv must be a nonempty path")
            candidate_surface_path = Path(surface_csv_override)
            if not candidate_surface_path.is_absolute():
                candidate_surface_path = repo_root / candidate_surface_path
            surface_csv_path = candidate_surface_path.resolve()
            try:
                surface_csv_path.relative_to(repo_root)
            except ValueError as exc:
                raise ValueError("yaw_surface_csv must be inside the repository") from exc
        if (not math.isfinite(surface_blend_q_start)
                or not math.isfinite(surface_blend_q_end)
                or not math.isfinite(surface_speed_blend_margin)
                or not math.isfinite(surface_low_speed_blend_margin)
                or not math.isfinite(surface_low_speed_support_fadeout)
                or not math.isfinite(surface_high_speed_support_fadein)
                or not math.isfinite(surface_response_scale)
                or surface_blend_q_start < 0.0
                or surface_blend_q_end <= surface_blend_q_start
                or surface_speed_blend_margin < 0.0
                or surface_low_speed_blend_margin < 0.0
                or surface_low_speed_support_fadeout < 0.0
                or surface_high_speed_support_fadein < 0.0
                or not 0.0 <= surface_response_scale <= 1.0
                or ((surface_low_speed_support_fadeout > 0.0) !=
                    (surface_high_speed_support_fadein > 0.0))):
            raise ValueError("invalid yaw-response-surface blend override")
        surface_speeds: list[float] = []
        surface_q: list[list[list[float]]] = []
        surface_rates: list[list[list[float]]] = []
        if surface_enabled:
            steering_knots = (0.15, 0.20, 0.21, 0.22, 0.23,
                              0.25, 0.30, 0.35, 0.42, 0.50)
            with surface_csv_path.open("r", encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(
                    line for line in stream if not line.lstrip().startswith("#")))
            rows_per_speed = 2 * len(steering_knots)
            if (len(rows) % rows_per_speed != 0
                    or not 3 <= len(rows) // rows_per_speed <= 5):
                raise ValueError(
                    "yaw response surface must contain 3-5 complete speed knots: "
                    f"{surface_csv_path}")
            surface_speed_count = len(rows) // rows_per_speed
            for speed_index in range(surface_speed_count):
                speed_rows: list[list[float]] = []
                rate_rows: list[list[float]] = []
                for direction_index, expected_sign in enumerate((-1, 1)):
                    block = rows[(speed_index * 2 + direction_index) * len(steering_knots):
                                 (speed_index * 2 + direction_index + 1) * len(steering_knots)]
                    q_values: list[float] = []
                    rate_values: list[float] = []
                    speed_value = float(block[0]["speed_knot_mps"])
                    for knot, row in zip(steering_knots, block):
                        if int(row["turn_sign"]) != expected_sign or abs(
                                float(row["steering_rad"]) - knot) > 1.0e-5:
                            raise ValueError("yaw response surface has unexpected row order")
                        if abs(float(row["speed_knot_mps"]) - speed_value) > 1.0e-5:
                            raise ValueError("yaw response surface speed knot is inconsistent")
                        q_value = float(row["demand_q"])
                        rate_value = float(row["yaw_rate_abs_rps"])
                        if q_value <= 0.0 or rate_value <= 0.0 or (
                                q_values and q_value <= q_values[-1]):
                            raise ValueError("yaw response surface contains invalid/nonmonotone data")
                        q_values.append(q_value)
                        rate_values.append(rate_value)
                    if direction_index == 0:
                        surface_speeds.append(speed_value)
                    elif abs(surface_speeds[-1] - speed_value) > 1.0e-5:
                        raise ValueError("yaw response surface direction speeds do not match")
                    speed_rows.append(q_values)
                    rate_rows.append(rate_values)
                surface_q.append(speed_rows)
                surface_rates.append(rate_rows)
            if (surface_low_speed_support_fadeout > 0.0
                    and surface_low_speed_support_fadeout +
                    surface_high_speed_support_fadein >=
                    surface_speeds[2] - surface_speeds[0]):
                raise ValueError(
                    "yaw response support fades overlap the measured low/high speed bands")

        geom = sim_profile.get("geometry", {})
        mintime = sim_profile.get("mintime", {})
        limits = config.get("limits", {})

        required_geom = {
            "car_length_m": geom.get("car_length_m"),
            "car_width_m": geom.get("car_width_m"),
            "rear_overhang_m": geom.get("rear_overhang_m"),
        }
        missing = [k for k, v in required_geom.items() if v is None]
        if missing:
            raise ValueError(f"AutoDRIVE geometry is missing from {sim_profile_path}: {missing}")
        if mintime.get("optimizer_width_m") is None or mintime.get("wall_clearance_m") is None:
            raise ValueError(
                f"AutoDRIVE mintime safety width/clearance missing from {sim_profile_path}"
            )
        if pp.get("max_lateral_accel") is None:
            raise ValueError(f"max_lateral_accel missing from {pp_path}")

        # Fail hard only when shared vehicle limits drift apart. Target-speed
        # slew limits are controller policies, so Pure Pursuit's limits must
        # not be compared with the MPC's independently sourced limits below.
        checks = [
            ("max_speed", pp.get("max_speed"), constants["MPC_MAX_COMMAND_SPEED_MPS"]),
            ("max_steering", pp.get("max_steering"), constants["SOURCE_MAX_STEERING_RAD"]),
            ("max_steering_rate", pp.get("max_steering_rate"), constants["SOURCE_STEERING_RATE_RADPS"]),
        ]
        for name, observed, expected in checks:
            if observed is not None and not _close(float(observed), float(expected), tol=5.0e-4):
                raise ValueError(
                    f"Repository source mismatch for {name}: controller={observed}, MPC={expected}. "
                    "Resolve the repo inconsistency instead of guessing in the optimizer."
                )

        car_length = float(required_geom["car_length_m"])
        rear_overhang = float(required_geom["rear_overhang_m"])
        if rear_overhang < 0.0 or rear_overhang >= car_length:
            raise ValueError("Invalid AutoDRIVE rear_overhang/car_length geometry")

        yaw_gain_reduction = float(vehicle_model_overrides.get(
            "yaw_gain_reduction_per_rad",
            mpc_params.get("yaw_rate_steering_gain_reduction_per_rad", 0.0)))
        yaw_gain_start = float(vehicle_model_overrides.get(
            "yaw_gain_start_rad",
            mpc_params.get("yaw_rate_steering_gain_start_rad", 0.41)))
        yaw_gain_end = float(vehicle_model_overrides.get(
            "yaw_gain_end_rad",
            mpc_params.get("yaw_rate_steering_gain_end_rad", 0.46)))
        yaw_tau = float(vehicle_model_overrides.get(
            "yaw_response_time_constant_s",
            mpc_params.get("yaw_rate_response_time_constant_s",
                         constants["MPC_YAW_RATE_RESPONSE_TIME_CONSTANT_SECONDS"])))
        max_steering = float(constants["SOURCE_MAX_STEERING_RAD"])
        base_yaw_gain = float(constants["MPC_YAW_RATE_STEERING_GAIN_PER_M"])
        if (not all(math.isfinite(value) for value in
                    (yaw_tau, yaw_gain_reduction, yaw_gain_start, yaw_gain_end))
                or yaw_tau <= 0.0
                or yaw_gain_reduction < 0.0 or yaw_gain_start < 0.0
                or yaw_gain_end <= yaw_gain_start
                or yaw_gain_end > max_steering + 1.0e-9):
            raise ValueError("invalid steering-dependent yaw-gain override")
        gain_at_limit = base_yaw_gain - yaw_gain_reduction * min(
            max_steering - yaw_gain_start, yaw_gain_end - yaw_gain_start)
        if gain_at_limit <= 0.0:
            raise ValueError("steering-dependent yaw gain must remain positive")

        model = cls(
            max_steering_rad=constants["SOURCE_MAX_STEERING_RAD"],
            max_steering_rate_radps=constants["SOURCE_STEERING_RATE_RADPS"],
            yaw_tau_s=yaw_tau,
            yaw_gain_per_m=constants["MPC_YAW_RATE_STEERING_GAIN_PER_M"],
            yaw_gain_reduction_per_rad=yaw_gain_reduction,
            yaw_gain_start_rad=yaw_gain_start,
            yaw_gain_end_rad=yaw_gain_end,
            yaw_curvature_gain_reduction_per_m=float(mpc_params.get(
                "yaw_rate_curvature_gain_reduction_per_m", 0.0)),
            yaw_curvature_gain_start_per_m=float(mpc_params.get(
                "yaw_rate_curvature_gain_start_per_m", 0.20)),
            yaw_curvature_gain_end_per_m=float(mpc_params.get(
                "yaw_rate_curvature_gain_end_per_m", 0.40)),
            yaw_surface_enabled=surface_enabled,
            yaw_surface_response_scale=surface_response_scale,
            yaw_surface_blend_q_start=surface_blend_q_start,
            yaw_surface_blend_q_end=surface_blend_q_end,
            yaw_surface_speed_blend_margin_mps=surface_speed_blend_margin,
            yaw_surface_low_speed_blend_margin_mps=surface_low_speed_blend_margin,
            yaw_surface_low_speed_support_fadeout_mps=(
                surface_low_speed_support_fadeout),
            yaw_surface_high_speed_support_fadein_mps=(
                surface_high_speed_support_fadein),
            yaw_surface_speed_mps=tuple(surface_speeds),
            yaw_surface_q=tuple(tuple(tuple(row) for row in speed_rows)
                                 for speed_rows in surface_q),
            yaw_surface_rate_rps=tuple(tuple(tuple(row) for row in speed_rows)
                                       for speed_rows in surface_rates),
            longitudinal_bias_mps2=constants["MPC_LONGITUDINAL_RESPONSE_BIAS_MPS2"],
            longitudinal_speed_coeff_per_s=constants["MPC_LONGITUDINAL_SPEED_COEFF_PER_S"],
            longitudinal_target_gain_per_s=constants["MPC_LONGITUDINAL_TARGET_ERROR_GAIN_PER_S"],
            longitudinal_target_rate_coeff=constants["MPC_LONGITUDINAL_TARGET_RATE_COEFF"],
            accel_limit_mps2=constants["MPC_LONGITUDINAL_ACCEL_LIMIT_MPS2"],
            brake_intercept_mps2=constants["MPC_LONGITUDINAL_BRAKE_DECEL_INTERCEPT_MPS2"],
            brake_slope_s_inv=constants["MPC_LONGITUDINAL_BRAKE_DECEL_SLOPE_S_INV"],
            max_command_speed_mps=constants["MPC_MAX_COMMAND_SPEED_MPS"],
            max_target_speed_rate_increase_mps2=constants["MPC_TARGET_SPEED_RATE_INCREASE_MAX_MPS2"],
            max_target_speed_rate_reduction_mps2=constants["MPC_TARGET_SPEED_RATE_REDUCTION_MAX_MPS2"],
            car_width_m=float(required_geom["car_width_m"]),
            car_length_m=car_length,
            rear_overhang_m=rear_overhang,
            rear_axle_to_front_bumper_m=car_length - rear_overhang,
            planning_footprint_width_m=float(mintime["optimizer_width_m"]),
            required_wall_clearance_m=float(mintime["wall_clearance_m"]),
            validated_lateral_accel_mps2=float(pp["max_lateral_accel"]),
            lateral_velocity_yaw_rate_gain_m=float(
                odom_params["lateral_velocity_yaw_rate_gain_m"]),
            lateral_velocity_speed_yaw_rate_gain_s=float(
                odom_params["lateral_velocity_speed_yaw_rate_gain_s"]),
            lateral_velocity_max_mps=float(odom_params["lateral_velocity_max_mps"]),
            lateral_velocity_reference_forward_offset_m=float(
                odom_params["lateral_velocity_reference_forward_offset_m"]),
            min_speed_mps=float(limits.get("min_body_speed_mps", 0.8)),
            max_body_speed_mps=constants["MPC_MAX_COMMAND_SPEED_MPS"],
        )
        if model.planning_footprint_width_m < model.car_width_m:
            raise ValueError(
                "optimizer_width_m is narrower than the actual AutoDRIVE body; "
                "the planning footprint must not be less conservative than the car."
            )
        return model

    def to_dict(self) -> dict[str, float]:
        result = asdict(self)
        result.pop("yaw_surface_q")
        result.pop("yaw_surface_rate_rps")
        result["yaw_response_surface_samples"] = sum(
            len(row) for speed_rows in self.yaw_surface_q for row in speed_rows)
        return result

    def _legacy_yaw_rate_numeric(self, speed: float, steering: float,
                                 path_curvature: float) -> float:
        steer_delta = min(max(abs(steering) - self.yaw_gain_start_rad, 0.0),
                          self.yaw_gain_end_rad - self.yaw_gain_start_rad)
        steering_gain = self.yaw_gain_per_m - self.yaw_gain_reduction_per_rad * steer_delta
        curvature_delta = min(max(abs(path_curvature) - self.yaw_curvature_gain_start_per_m,
                                  0.0),
                              self.yaw_curvature_gain_end_per_m -
                              self.yaw_curvature_gain_start_per_m)
        path_gain = self.yaw_gain_per_m - self.yaw_curvature_gain_reduction_per_m * curvature_delta
        return speed * math.tan(steering) * (steering_gain - self.yaw_gain_per_m + path_gain)

    def steady_yaw_rate(self, speed: float, steering: float,
                        path_curvature: float = 0.0) -> float:
        legacy = self._legacy_yaw_rate_numeric(speed, steering, path_curvature)
        if not self.yaw_surface_enabled or self.yaw_surface_response_scale <= 0.0:
            return legacy
        speed_nonnegative = max(speed, 0.0)
        q = speed_nonnegative * abs(math.tan(steering))
        sign_index = 0 if steering < 0.0 else 1
        row_values = [
            _pchip_value(self.yaw_surface_q[i][sign_index],
                         self.yaw_surface_rate_rps[i][sign_index], q)
            for i in range(len(self.yaw_surface_speed_mps))
        ]
        knots = self.yaw_surface_speed_mps
        s = min(max(speed, knots[0]), knots[-1])
        interval = next((i for i in range(len(knots) - 1)
                         if s <= knots[i + 1]), len(knots) - 2)
        fraction = (s - knots[interval]) / (knots[interval + 1] - knots[interval])
        magnitude = row_values[interval] + fraction * (
            row_values[interval + 1] - row_values[interval])
        empirical = (-1.0 if steering < 0.0 else 1.0) * magnitude
        blend_t = min(max((q - self.yaw_surface_blend_q_start) /
                          (self.yaw_surface_blend_q_end - self.yaw_surface_blend_q_start),
                          0.0), 1.0)
        q_blend = blend_t * blend_t * (3.0 - 2.0 * blend_t)
        low, high = self.yaw_surface_speed_mps[0], self.yaw_surface_speed_mps[-1]
        low_margin = self.yaw_surface_low_speed_blend_margin_mps
        high_margin = self.yaw_surface_speed_blend_margin_mps
        if low_margin > 0.0:
            enter_t = min(max((speed - (low - low_margin)) / low_margin, 0.0), 1.0)
            enter = enter_t * enter_t * (3.0 - 2.0 * enter_t)
        else:
            enter = 1.0
        if high_margin > 0.0:
            leave_t = min(max(((high + high_margin) - speed) / high_margin, 0.0), 1.0)
            leave = leave_t * leave_t * (3.0 - 2.0 * leave_t)
        else:
            leave = 1.0
        speed_blend = enter * leave * self._yaw_surface_speed_support(speed)
        blend = self.yaw_surface_response_scale * q_blend * speed_blend
        return legacy + blend * (empirical - legacy)

    def _yaw_surface_speed_support(self, speed: float) -> float:
        low_margin = self.yaw_surface_low_speed_support_fadeout_mps
        high_margin = self.yaw_surface_high_speed_support_fadein_mps
        if low_margin <= 0.0 and high_margin <= 0.0:
            return 1.0
        low_knot = self.yaw_surface_speed_mps[0]
        high_knot = self.yaw_surface_speed_mps[2]
        low_t = min(max((speed - low_knot) / low_margin, 0.0), 1.0) \
            if low_margin > 0.0 else 0.0
        high_t = min(max((speed - (high_knot - high_margin)) / high_margin,
                         0.0), 1.0) if high_margin > 0.0 else 0.0
        low_weight = 1.0 - low_t * low_t * (3.0 - 2.0 * low_t)
        high_weight = high_t * high_t * (3.0 - 2.0 * high_t)
        return low_weight + high_weight

    def steady_yaw_rate_casadi(self, speed, steering, path_curvature):
        steer_delta = ca.fmin(ca.fmax(
            ca.fabs(steering) - self.yaw_gain_start_rad, 0.0),
            self.yaw_gain_end_rad - self.yaw_gain_start_rad)
        steering_gain = self.yaw_gain_per_m - self.yaw_gain_reduction_per_rad * steer_delta
        curvature_delta = ca.fmin(ca.fmax(
            ca.fabs(path_curvature) - self.yaw_curvature_gain_start_per_m, 0.0),
            self.yaw_curvature_gain_end_per_m - self.yaw_curvature_gain_start_per_m)
        path_gain = self.yaw_gain_per_m - self.yaw_curvature_gain_reduction_per_m * curvature_delta
        legacy = speed * ca.tan(steering) * (
            steering_gain - self.yaw_gain_per_m + path_gain)
        if not self.yaw_surface_enabled or self.yaw_surface_response_scale <= 0.0:
            return legacy

        query_speed = ca.fmax(speed, 0.0)
        q = query_speed * ca.fabs(ca.tan(steering))
        direction_values = []
        for direction in range(2):
            rows = [
                _pchip_casadi(self.yaw_surface_q[i][direction],
                              self.yaw_surface_rate_rps[i][direction], q)
                for i in range(len(self.yaw_surface_speed_mps))
            ]
            knots = self.yaw_surface_speed_mps
            s = ca.fmin(ca.fmax(speed, knots[0]), knots[-1])
            magnitude = rows[-1]
            for index in reversed(range(len(knots) - 1)):
                segment = rows[index] + (
                    (s - knots[index]) / (knots[index + 1] - knots[index])
                    * (rows[index + 1] - rows[index]))
                magnitude = ca.if_else(s <= knots[index + 1], segment, magnitude)
            direction_values.append(magnitude if direction == 1 else -magnitude)
        empirical = ca.if_else(steering < 0.0, direction_values[0], direction_values[1])
        blend_t = ca.fmin(ca.fmax(
            (q - self.yaw_surface_blend_q_start) /
            (self.yaw_surface_blend_q_end - self.yaw_surface_blend_q_start), 0.0), 1.0)
        q_blend = blend_t * blend_t * (3.0 - 2.0 * blend_t)
        low, high = self.yaw_surface_speed_mps[0], self.yaw_surface_speed_mps[-1]
        low_margin = self.yaw_surface_low_speed_blend_margin_mps
        high_margin = self.yaw_surface_speed_blend_margin_mps
        if low_margin > 0.0:
            enter_t = ca.fmin(ca.fmax((speed - (low - low_margin)) / low_margin, 0.0), 1.0)
            enter = enter_t * enter_t * (3.0 - 2.0 * enter_t)
        else:
            enter = 1.0
        if high_margin > 0.0:
            leave_t = ca.fmin(ca.fmax(((high + high_margin) - speed) / high_margin, 0.0), 1.0)
            leave = leave_t * leave_t * (3.0 - 2.0 * leave_t)
        else:
            leave = 1.0
        if (self.yaw_surface_low_speed_support_fadeout_mps > 0.0
                or self.yaw_surface_high_speed_support_fadein_mps > 0.0):
            low_knot = self.yaw_surface_speed_mps[0]
            high_knot = self.yaw_surface_speed_mps[2]
            low_margin = self.yaw_surface_low_speed_support_fadeout_mps
            high_margin = self.yaw_surface_high_speed_support_fadein_mps
            if low_margin > 0.0:
                low_t = ca.fmin(ca.fmax(
                    (speed - low_knot) / low_margin, 0.0), 1.0)
                low_support = 1.0 - low_t * low_t * (3.0 - 2.0 * low_t)
            else:
                low_support = 0.0
            if high_margin > 0.0:
                high_t = ca.fmin(ca.fmax(
                    (speed - (high_knot - high_margin)) / high_margin,
                    0.0), 1.0)
                high_support = high_t * high_t * (3.0 - 2.0 * high_t)
            else:
                high_support = 0.0
            support = low_support + high_support
        else:
            support = 1.0
        speed_blend = enter * leave * support
        blend = self.yaw_surface_response_scale * q_blend * speed_blend
        return legacy + blend * (empirical - legacy)

    def steering_for_yaw_rate(self, speed: float, desired_yaw_rate: float,
                               path_curvature: float = 0.0) -> float:
        # Match the production C inverse exactly: it evaluates 96 evenly
        # spaced magnitudes in the requested turn direction, with zero as the
        # initial candidate. This response surface is non-monotone, so a
        # Newton inverse or a different grid can select another steering root.
        direction = -1.0 if desired_yaw_rate < 0.0 else 1.0
        angles = direction * self.max_steering_rad * (
            np.arange(1, 97, dtype=np.float64) / 96.0)
        predicted = np.asarray([
            self.steady_yaw_rate(speed, float(angle), path_curvature)
            for angle in angles
        ])
        errors = np.concatenate((
            np.asarray([abs(desired_yaw_rate)]),
            np.abs(predicted - desired_yaw_rate)))
        best = int(np.argmin(errors))
        return 0.0 if best == 0 else float(angles[best - 1])

    def provenance(self) -> dict[str, Any]:
        return {
            "dynamics": "f1tenth_mpc/include/mpc_types.h (held-out AutoDRIVE identified model)",
            "geometry": "f1tenth_planning/config/autodrive_sim_vehicle.yaml:geometry",
            "planning_footprint_and_wall_clearance": "f1tenth_planning/config/autodrive_sim_vehicle.yaml:mintime",
            "lateral_accel": "f1tenth_control/config/path_tracking_autodrive.yaml:max_lateral_accel",
            "rear_axle_lateral_velocity": (
                "f1tenth_localization/config/sensor_odometry.yaml:"
                "lateral_velocity_* (causal wheel-speed/IMU-yaw model)"
            ),
            "explicitly_not_used": [
                "global_racetrajectory_optimization/params/racecar.ini dynamics",
                "Pacejka tire coefficients",
                "friction coefficient mu",
                "BachelorProject cornering stiffness",
                "TUM double-track dynamics",
            ],
        }

    def steady_target_speed(self, u: np.ndarray | float) -> np.ndarray | float:
        return u - (
            self.longitudinal_bias_mps2
            + self.longitudinal_speed_coeff_per_s * u
        ) / self.longitudinal_target_gain_per_s

    def raw_longitudinal_accel(self, u, target_speed, target_speed_rate):
        return (
            self.longitudinal_bias_mps2
            + self.longitudinal_speed_coeff_per_s * u
            + self.longitudinal_target_gain_per_s * (target_speed - u)
            + self.longitudinal_target_rate_coeff * target_speed_rate
        )

    def brake_limit(self, u):
        return self.brake_intercept_mps2 + self.brake_slope_s_inv * u

    def lateral_velocity_numeric(self, speed, yaw_rate):
        """Rear-axle lateral velocity from the runtime sensor-only model."""
        speed_nonnegative = np.maximum(speed, 0.0)
        at_reference = yaw_rate * (
            self.lateral_velocity_yaw_rate_gain_m
            + self.lateral_velocity_speed_yaw_rate_gain_s * speed_nonnegative
        )
        return (
            np.clip(at_reference, -self.lateral_velocity_max_mps,
                    self.lateral_velocity_max_mps)
            - yaw_rate * self.lateral_velocity_reference_forward_offset_m
        )

    def lateral_velocity_casadi(self, speed, yaw_rate):
        """CasADi form of the same causal lateral-velocity observer model."""
        speed_nonnegative = ca.fmax(speed, 0.0)
        at_reference = yaw_rate * (
            self.lateral_velocity_yaw_rate_gain_m
            + self.lateral_velocity_speed_yaw_rate_gain_s * speed_nonnegative
        )
        bounded = ca.fmin(
            ca.fmax(at_reference, -self.lateral_velocity_max_mps),
            self.lateral_velocity_max_mps,
        )
        return bounded - yaw_rate * self.lateral_velocity_reference_forward_offset_m

    @property
    def aligned_required_center_to_wall_m(self) -> float:
        return 0.5 * self.planning_footprint_width_m + self.required_wall_clearance_m


@dataclass(frozen=True)
class LateralEnvelope:
    speed_mps: np.ndarray
    steering_abs_rad: np.ndarray
    ay_max_mps2: np.ndarray
    source: str
    scale: float = 1.0

    @classmethod
    def from_repo(cls, repo_root: str | Path, model: VehicleModel,
                  config: dict[str, Any]) -> "LateralEnvelope":
        # Current repo source is deliberately the validated runtime envelope,
        # not the old TUM/BachelorProject 7.3 m/s^2 planning constant.
        section = config.get("lateral_envelope", {})
        scale = float(section.get("scale", 1.0))
        if scale <= 0.0:
            raise ValueError("lateral_envelope.scale must be positive")
        profile_keys = ("speed_mps", "steering_abs_rad", "ay_max_mps2")
        profile_present = [key in section for key in profile_keys]
        if any(profile_present) and not all(profile_present):
            raise ValueError(
                "lateral_envelope profiles require speed_mps, "
                "steering_abs_rad, and ay_max_mps2 together"
            )
        if all(profile_present):
            speed_mps = np.asarray(section["speed_mps"], dtype=float)
            steering_abs_rad = np.asarray(section["steering_abs_rad"], dtype=float)
            ay_max_mps2 = np.asarray(section["ay_max_mps2"], dtype=float)
            if (speed_mps.ndim != 1 or steering_abs_rad.ndim != 1 or
                    speed_mps.size < 2 or steering_abs_rad.size < 2 or
                    not np.all(np.isfinite(speed_mps)) or
                    not np.all(np.isfinite(steering_abs_rad)) or
                    not np.all(np.diff(speed_mps) > 0.0) or
                    not np.all(np.diff(steering_abs_rad) > 0.0) or
                    abs(float(speed_mps[0])) > 1.0e-9 or
                    speed_mps[-1] < model.max_body_speed_mps or
                    abs(float(steering_abs_rad[0])) > 1.0e-9 or
                    steering_abs_rad[-1] < model.max_steering_rad or
                    ay_max_mps2.shape != (speed_mps.size, steering_abs_rad.size) or
                    not np.all(np.isfinite(ay_max_mps2)) or
                    np.any(ay_max_mps2 <= 0.0)):
                raise ValueError(
                    "invalid speed/steering lateral-envelope profile or coverage"
                )
            source = str(section.get("source", "candidate speed/steering profile"))
        else:
            cap = model.validated_lateral_accel_mps2
            speed_mps = np.asarray([0.0, model.max_body_speed_mps], dtype=float)
            steering_abs_rad = np.asarray(
                [0.0, model.max_steering_rad], dtype=float)
            ay_max_mps2 = np.full((2, 2), cap, dtype=float)
            source = (
                "f1tenth_control/config/path_tracking_autodrive.yaml:max_lateral_accel "
                "(current measured runtime envelope)"
            )
        return cls(
            speed_mps=speed_mps,
            steering_abs_rad=steering_abs_rad,
            ay_max_mps2=ay_max_mps2,
            source=source,
            scale=scale,
        )

    def casadi_function(self) -> ca.Function:
        def interpolate(knots: np.ndarray, values: list[Any], query):
            # Cubic smoothstep preserves each measured knot value while making
            # the envelope C1 across cell boundaries for IPOPT's derivatives.
            result = values[-1]
            for i in reversed(range(len(knots) - 1)):
                t = (query - float(knots[i])) / float(knots[i + 1] - knots[i])
                t = ca.fmin(ca.fmax(t, 0.0), 1.0)
                weight = t * t * (3.0 - 2.0 * t)
                segment = values[i] + weight * (values[i + 1] - values[i])
                result = ca.if_else(query <= float(knots[i + 1]), segment, result)
            return ca.if_else(query < float(knots[0]), values[0],
                              ca.if_else(query > float(knots[-1]), values[-1], result))

        u = ca.MX.sym("u")
        steering = ca.MX.sym("steering")
        row_caps = [
            interpolate(self.steering_abs_rad,
                        [float(value) for value in row], ca.fabs(steering))
            for row in self.ay_max_mps2
        ]
        return ca.Function(
            "ay_cap", [u, steering],
            [self.scale * interpolate(self.speed_mps, row_caps, u)])

    @staticmethod
    def _smoothstep_coordinate(knots: np.ndarray, query: np.ndarray
                               ) -> tuple[np.ndarray, np.ndarray]:
        index = np.clip(np.searchsorted(knots, query, side="right") - 1,
                        0, knots.size - 2)
        t = np.clip((query - knots[index]) /
                    (knots[index + 1] - knots[index]), 0.0, 1.0)
        return index, t * t * (3.0 - 2.0 * t)

    def numpy(self, u: np.ndarray | float,
              steering: np.ndarray | float = 0.0) -> np.ndarray | float:
        speed, steer = np.broadcast_arrays(
            np.asarray(u, dtype=float), np.abs(np.asarray(steering, dtype=float)))
        speed_index, speed_weight = self._smoothstep_coordinate(self.speed_mps, speed)
        steer_index, steer_weight = self._smoothstep_coordinate(
            self.steering_abs_rad, steer)
        low = (self.ay_max_mps2[speed_index, steer_index] * (1.0 - steer_weight)
               + self.ay_max_mps2[speed_index, steer_index + 1] * steer_weight)
        high = (self.ay_max_mps2[speed_index + 1, steer_index] * (1.0 - steer_weight)
                + self.ay_max_mps2[speed_index + 1, steer_index + 1] * steer_weight)
        return self.scale * (low * (1.0 - speed_weight) + high * speed_weight)

    def to_dict(self) -> dict[str, Any]:
        return {
            "speed_mps": self.speed_mps.tolist(),
            "steering_abs_rad": self.steering_abs_rad.tolist(),
            "ay_max_mps2": self.ay_max_mps2.tolist(),
            "source": self.source,
            "scale": self.scale,
        }


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def smooth_abs(x, eps: float = 1.0e-6):
    return ca.sqrt(x * x + eps * eps)


def wrap_angle_np(x: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(x), np.cos(x))


def numerical_model_summary(model: VehicleModel) -> dict[str, Any]:
    return {
        **model.to_dict(),
        "steady_kappa_at_max_steer_m_inv": model.steady_yaw_rate(
            1.0, model.max_steering_rad),
        "steady_body_speed_at_max_target_mps": (
            model.longitudinal_target_gain_per_s * model.max_command_speed_mps
            - model.longitudinal_bias_mps2
        ) / (model.longitudinal_target_gain_per_s - model.longitudinal_speed_coeff_per_s),
        "aligned_required_center_to_wall_m": model.aligned_required_center_to_wall_m,
        "provenance": model.provenance(),
    }
