from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
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


@dataclass(frozen=True)
class VehicleModel:
    # Empirically identified command->vehicle model used by the MPC.
    max_steering_rad: float
    max_steering_rate_radps: float
    yaw_tau_s: float
    yaw_gain_per_m: float
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

    # Numerical lower bound only; not a simulator physical constant.
    min_speed_mps: float
    max_body_speed_mps: float

    @classmethod
    def from_repo(cls, repo_root: str | Path, config: dict[str, Any]) -> "VehicleModel":
        repo_root = Path(repo_root).resolve()
        header = repo_root / "f1tenth_mpc/include/mpc_types.h"
        sim_profile_path = repo_root / "f1tenth_planning/config/autodrive_sim_vehicle.yaml"
        pp_path = repo_root / "f1tenth_control/config/path_tracking_autodrive.yaml"

        constants = read_mpc_constants(header)
        sim_profile = _load_yaml(sim_profile_path)
        pp_doc = _load_yaml(pp_path)
        pp = _ros_params(pp_doc)

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

        # Fail hard if duplicated active controller values drift apart. This is
        # deliberate: the optimizer must never silently optimize a different car.
        checks = [
            ("max_speed", pp.get("max_speed"), constants["MPC_MAX_COMMAND_SPEED_MPS"]),
            ("max_steering", pp.get("max_steering"), constants["SOURCE_MAX_STEERING_RAD"]),
            ("max_steering_rate", pp.get("max_steering_rate"), constants["SOURCE_STEERING_RATE_RADPS"]),
            ("max_accel_cmd", pp.get("max_accel_cmd"), constants["MPC_TARGET_SPEED_RATE_INCREASE_MAX_MPS2"]),
            ("max_decel_cmd", pp.get("max_decel_cmd"), constants["MPC_TARGET_SPEED_RATE_REDUCTION_MAX_MPS2"]),
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

        model = cls(
            max_steering_rad=constants["SOURCE_MAX_STEERING_RAD"],
            max_steering_rate_radps=constants["SOURCE_STEERING_RATE_RADPS"],
            yaw_tau_s=constants["MPC_YAW_RATE_RESPONSE_TIME_CONSTANT_SECONDS"],
            yaw_gain_per_m=constants["MPC_YAW_RATE_STEERING_GAIN_PER_M"],
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
        return asdict(self)

    def provenance(self) -> dict[str, Any]:
        return {
            "dynamics": "f1tenth_mpc/include/mpc_types.h (held-out AutoDRIVE identified model)",
            "geometry": "f1tenth_planning/config/autodrive_sim_vehicle.yaml:geometry",
            "planning_footprint_and_wall_clearance": "f1tenth_planning/config/autodrive_sim_vehicle.yaml:mintime",
            "lateral_accel": "f1tenth_control/config/path_tracking_autodrive.yaml:max_lateral_accel",
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

    @property
    def aligned_required_center_to_wall_m(self) -> float:
        return 0.5 * self.planning_footprint_width_m + self.required_wall_clearance_m


@dataclass(frozen=True)
class LateralEnvelope:
    speed_mps: np.ndarray
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
        cap = model.validated_lateral_accel_mps2
        return cls(
            speed_mps=np.asarray([0.0, model.max_body_speed_mps], dtype=float),
            ay_max_mps2=np.asarray([cap, cap], dtype=float),
            source=(
                "f1tenth_control/config/path_tracking_autodrive.yaml:max_lateral_accel "
                "(current measured runtime envelope)"
            ),
            scale=scale,
        )

    def casadi_function(self) -> ca.Function:
        lut = ca.interpolant(
            "autodrive_ay_cap", "linear", [self.speed_mps.tolist()],
            self.ay_max_mps2.tolist())
        u = ca.MX.sym("u")
        return ca.Function("ay_cap", [u], [self.scale * lut(u)])

    def numpy(self, u: np.ndarray | float) -> np.ndarray | float:
        return self.scale * np.interp(u, self.speed_mps, self.ay_max_mps2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "speed_mps": self.speed_mps.tolist(),
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
        "steady_kappa_at_max_steer_m_inv": model.yaw_gain_per_m * math.tan(model.max_steering_rad),
        "steady_body_speed_at_max_target_mps": (
            model.longitudinal_target_gain_per_s * model.max_command_speed_mps
            - model.longitudinal_bias_mps2
        ) / (model.longitudinal_target_gain_per_s - model.longitudinal_speed_coeff_per_s),
        "aligned_required_center_to_wall_m": model.aligned_required_center_to_wall_m,
        "provenance": model.provenance(),
    }
