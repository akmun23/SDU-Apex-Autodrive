#!/usr/bin/env python3
"""Validate the canonical vehicle-model and odometry contracts.

This is deliberately a source/configuration gate, not a model-promotion gate.
The production MPC and the identified replay plant are currently different
architectures. The normal check records that fact; promotion checks can opt
into ``--require-candidate-compatibility`` when the migration is complete.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
from typing import Any


TOKEN = re.compile(r"\b[A-Z_][A-Z0-9_]*\b")
DEFINE = re.compile(r"^\s*#define\s+([A-Za-z_][A-Za-z0-9_]*)(?:\([^\n]*\))?\s+(.+?)\s*$")

CONSTANT_MACROS = {
    "model_signature": "MPC_MODEL_SIGNATURE",
    "control_rate_hz": "CONTROL_RATE_HZ",
    "prediction_dt_s": "PREDICTION_DT_SECONDS",
    "steering_rate_radps": "STEERING_RATE_LIMIT",
    "steering_time_constant_s": "STEERING_EFFECTIVE_TIME_CONSTANT_SECONDS",
    "max_steering_rad": "VP_MAX_STEERING_RAD",
    "max_velocity_mps": "VP_MAX_VELOCITY_MPS",
    "wheelbase_m": "VP_WHEELBASE_M",
    "front_axle_from_com_m": "VP_CG_TO_FRONT_AXLE_M",
    "rear_axle_from_com_m": "VP_CG_TO_REAR_AXLE_M",
    "mass_kg": "VP_MASS_KG",
    "yaw_inertia_kgm2": "VP_YAW_INERTIA_KGM2",
    "cg_height_m": "VP_CG_HEIGHT_M",
    "friction_coefficient": "VP_FRICTION_COEFF",
    "gravity_mps2": "GRAVITY_MPS2",
    "max_acceleration_mps2": "VP_MAX_ACCEL_MPS2",
    "min_acceleration_mps2": "VP_MIN_ACCEL_MPS2",
    "front_lateral_force_scale_n": "VP_D_FRONT",
    "rear_lateral_force_scale_n": "VP_D_REAR",
    "front_cornering_stiffness_per_rad": "VP_FRONT_CORNERING_STIFFNESS",
    "rear_cornering_stiffness_per_rad": "VP_REAR_CORNERING_STIFFNESS",
    "tire_shape_factor": "VP_C_SHAPE",
    "minimum_slip_velocity_mps": "MIN_SLIP_VELOCITY",
    "minimum_stiffness_scale": "MIN_STIFF_SCALE",
}

DEFAULT_INITIALIZER_MACROS = {
    "wheelbase_meters": "VP_WHEELBASE_M",
    "distance_cg_to_front_axle": "VP_CG_TO_FRONT_AXLE_M",
    "distance_cg_to_rear_axle": "VP_CG_TO_REAR_AXLE_M",
    "height_cg_to_ground": "VP_CG_HEIGHT_M",
    "vehicle_mass": "VP_MASS_KG",
    "yaw_moment_of_inertia": "VP_YAW_INERTIA_KGM2",
    "front_cornering_stiffness": "VP_FRONT_CORNERING_STIFFNESS",
    "rear_cornering_stiffness": "VP_REAR_CORNERING_STIFFNESS",
    "max_steering_angle": "VP_MAX_STEERING_RAD",
    "max_velocity": "VP_MAX_VELOCITY_MPS",
    "max_acceleration": "VP_MAX_ACCEL_MPS2",
    "min_acceleration": "VP_MIN_ACCEL_MPS2",
    "friction_coefficient": "VP_FRICTION_COEFF",
    "gravity_mps2": "GRAVITY_MPS2",
    "tire_shape_factor": "VP_C_SHAPE",
    "minimum_slip_velocity": "MIN_SLIP_VELOCITY",
    "steering_time_constant_seconds": "STEERING_EFFECTIVE_TIME_CONSTANT_SECONDS",
}

PLANT_DEFAULT_MACROS = {
    "mass_kg": "PLANT_DEFAULT_MASS_KG",
    "lf_m": "PLANT_DEFAULT_LF_M",
    "lr_m": "PLANT_DEFAULT_LR_M",
    "iz_kgm2": "PLANT_DEFAULT_IZ_KGM2",
    "position_offset_from_velocity_point_x_m": "PLANT_DEFAULT_POSITION_OFFSET_X_M",
    "max_steering_rad": "PLANT_DEFAULT_MAX_STEERING_RAD",
    "steering_rate_radps": "PLANT_DEFAULT_STEERING_RATE_RADPS",
    "max_speed_mps": "PLANT_DEFAULT_MAX_SPEED_MPS",
    "linear_damping_per_s": "PLANT_DEFAULT_LINEAR_DAMPING_PER_S",
    "angular_damping_per_s": "PLANT_DEFAULT_ANGULAR_DAMPING_PER_S",
    "force_max_n": "PLANT_DEFAULT_FORCE_MAX_N",
    "hard_brake_force_n": "PLANT_DEFAULT_HARD_BRAKE_FORCE_N",
    "slip_gain_per_mps": "PLANT_DEFAULT_SLIP_GAIN_PER_MPS",
    "coast_speed_drag_n_per_mps": "PLANT_DEFAULT_DRAG_N_PER_MPS",
    "cf_n_per_rad": "PLANT_DEFAULT_CF_N_PER_RAD",
    "cr_n_per_rad": "PLANT_DEFAULT_CR_N_PER_RAD",
    "df_n": "PLANT_DEFAULT_DF_N",
    "dr_n": "PLANT_DEFAULT_DR_N",
}


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _defines(header: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in header.splitlines():
        match = DEFINE.match(line.split("/*", 1)[0].split("//", 1)[0].rstrip())
        if match:
            result[match.group(1)] = match.group(2).strip()
    return result


def _macro_value(name: str, defines: dict[str, str], stack: tuple[str, ...] = ()) -> float:
    if name in stack:
        raise ValueError(f"recursive macro definition: {' -> '.join(stack + (name,))}")
    expression = defines[name]
    expression = re.sub(r"(?<=\d)[fF]\b", "", expression)

    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        if token == name:
            return token
        if token in defines:
            return f"({_macro_value(token, defines, stack + (name,))!r})"
        return token

    expression = TOKEN.sub(replace, expression)
    if re.search(r"[^0-9eE+*./()_ -]", expression):
        raise ValueError(f"unsupported expression for {name}: {expression}")
    try:
        tree = ast.parse(expression, mode="eval")
        if any(isinstance(node, (ast.Name, ast.Call, ast.Attribute))
               for node in ast.walk(tree)):
            raise ValueError(f"unresolved token in macro {name}: {expression}")
        return float(eval(compile(tree, f"<macro {name}>", "eval"),
                          {"__builtins__": {}}, {}))
    except (SyntaxError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"cannot evaluate macro {name}: {expression}") from exc


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"manifest field {field} is not numeric: {value!r}")
    return float(value)


def _compare(expected: Any, actual: Any, path: str, tolerance: float,
             errors: list[str]) -> None:
    if isinstance(expected, bool) or isinstance(actual, bool):
        if expected != actual:
            errors.append(f"{path}: expected {expected!r}, actual {actual!r}")
        return
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if abs(float(expected) - float(actual)) > tolerance:
            errors.append(f"{path}: expected {expected!r}, actual {actual!r}")
        return
    if isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
        if len(expected) != len(actual):
            errors.append(f"{path}: expected length {len(expected)}, actual {len(actual)}")
            return
        for index, (left, right) in enumerate(zip(expected, actual)):
            _compare(left, right, f"{path}[{index}]", tolerance, errors)
        return
    if expected != actual:
        errors.append(f"{path}: expected {expected!r}, actual {actual!r}")


def _cpp_deployment_defaults(source: str) -> dict[str, Any]:
    start = source.index("OdometryObserverConfig deployment_observer_config()")
    end = source.index("OdometryObserver::OdometryObserver", start)
    body = source[start:end]
    result: dict[str, Any] = {}
    for match in re.finditer(r"config\.(\w+)\s*=\s*(.*?);", body, re.DOTALL):
        field, expression = match.groups()
        expression = re.sub(r"//[^\n]*|/\*.*?\*/", "", expression, flags=re.DOTALL).strip()
        expression = re.sub(r"(?<=\d)[fF]\b", "", expression)
        expression = re.sub(r"\btrue\b", "True", expression)
        expression = re.sub(r"\bfalse\b", "False", expression)
        expression = expression.replace("{", "[").replace("}", "]")
        try:
            result[field] = ast.literal_eval(expression)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(
                f"cannot parse C++ deployment default {field}: {expression!r}") from exc
    return result


def _python_constructor_defaults(source: str) -> dict[str, Any]:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "__init__":
            defaults = node.args.defaults
            first_default = len(node.args.args) - len(defaults)
            result: dict[str, Any] = {}
            for argument, default in zip(node.args.args[first_default:], defaults):
                try:
                    result[argument.arg] = ast.literal_eval(default)
                except (SyntaxError, ValueError) as exc:
                    raise ValueError(
                        f"cannot parse Python observer default {argument.arg}") from exc
            return result
    raise ValueError("ReferenceObserver.__init__ was not found")


def _check_odometry(manifest: dict[str, Any], yaml_path: Path,
                    cpp_path: Path, python_path: Path, node_path: Path,
                    tolerance: float) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ValueError("odometry manifest check requires PyYAML") from exc

    profile = manifest.get("odometry_profile", {})
    document = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    parameters = document.get("sensor_odometry", {}).get("ros__parameters", {})
    if not isinstance(parameters, dict) or not parameters:
        raise ValueError("sensor odometry YAML has no ros__parameters mapping")
    actual_fingerprint = _fingerprint(parameters)
    expected_fingerprint = profile.get("fingerprint")
    if expected_fingerprint != actual_fingerprint:
        raise ValueError(
            "sensor_odometry.yaml fingerprint mismatch: "
            f"manifest={expected_fingerprint!r}, actual={actual_fingerprint!r}")

    cpp = _cpp_deployment_defaults(cpp_path.read_text(encoding="utf-8"))
    python = _python_constructor_defaults(python_path.read_text(encoding="utf-8"))
    mapping = profile.get("observer_parameter_mapping", [])
    if not mapping:
        raise ValueError("manifest has no exhaustive odometry observer mapping")
    errors: list[str] = []
    checked_yaml: set[str] = set()
    for item in mapping:
        yaml_name = item["yaml"]
        cpp_name = item["cpp"]
        python_name = item["python"]
        checked_yaml.add(yaml_name)
        if yaml_name not in parameters:
            errors.append(f"odometry YAML missing {yaml_name}")
            continue
        if cpp_name not in cpp:
            errors.append(f"C++ deployment defaults missing {cpp_name}")
            continue
        if python_name not in python:
            errors.append(f"Python observer defaults missing {python_name}")
            continue
        _compare(parameters[yaml_name], cpp[cpp_name], f"odom.{yaml_name}.cpp",
                 tolerance, errors)
        _compare(parameters[yaml_name], python[python_name], f"odom.{yaml_name}.python",
                 tolerance, errors)
    if errors:
        raise ValueError("odometry configuration drift:\n  " + "\n  ".join(errors))

    node_source = node_path.read_text(encoding="utf-8")
    undeclared = sorted(name for name in parameters
                        if f'"{name}"' not in node_source)
    if undeclared:
        raise ValueError(
            "sensor_odometry.yaml parameters are absent from sensor_odometry_node.cpp: "
            + ", ".join(undeclared))
    return {
        "status": "pass",
        "parameter_count": len(parameters),
        "observer_parameter_count": len(mapping),
        "fingerprint": actual_fingerprint,
        "yaml": str(yaml_path),
        "cpp_defaults": str(cpp_path),
        "python_reference": str(python_path),
        "node": str(node_path),
        "checked_yaml_fields": sorted(checked_yaml),
    }


def _check_candidate_reports(manifest: dict[str, Any], lateral_path: Path,
                             longitudinal_path: Path, tolerance: float) -> dict[str, Any]:
    candidate = manifest["profiles"][manifest["canonical_candidate_profile"]]
    lateral = json.loads(lateral_path.read_text(encoding="utf-8"))
    longitudinal = json.loads(longitudinal_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    selected_lateral = candidate["lateral_model"]
    lateral_params = lateral["candidate_comparison"][selected_lateral]["parameters"]["parameters"]
    resolved = candidate["parameters"]
    for name in ("cf_n_per_rad", "cr_n_per_rad", "df_n", "dr_n", "iz_kgm2"):
        _compare(resolved[name], lateral_params[name], f"candidate.lateral.{name}",
                 tolerance, errors)
    if lateral_params.get("tire_model") != resolved["tire_model"]:
        errors.append("candidate lateral tire model does not match its report")

    selected_longitudinal = candidate["longitudinal_model"]
    long_model = longitudinal["models"][selected_longitudinal]
    long_params = long_model["parameters"]
    for name in ("mass_kg", "slip_gain_per_mps", "force_max_n",
                 "hard_brake_force_n"):
        _compare(resolved[name], long_params[name], f"candidate.longitudinal.{name}",
                 tolerance, errors)
    damping = candidate["damping_profile"]
    for name in ("linear_damping_per_s", "angular_damping_per_s",
                 "coast_speed_drag_n_per_mps"):
        _compare(resolved[name], damping[name], f"candidate.damping.{name}",
                 tolerance, errors)
    # The longitudinal fit contains a speed-drag coefficient. The selected
    # runtime-equivalent offline profile removes that term because Unity's
    # explicit Rigidbody drag is already represented by linear_damping_per_s.
    _compare(
        damping["identified_source_report_coast_speed_drag_n_per_mps"],
        long_params["coast_speed_drag_n_per_mps"],
        "candidate.longitudinal.identified_coast_speed_drag_n_per_mps",
        tolerance, errors)
    _compare(resolved["wheel_coefficients"],
             long_model["parameters"]["wheel_dynamics"]["coefficients"],
             "candidate.longitudinal.wheel_coefficients", tolerance, errors)
    if errors:
        raise ValueError("candidate source-report drift:\n  " + "\n  ".join(errors))
    return {
        "status": "pass",
        "lateral_model": selected_lateral,
        "longitudinal_model": selected_longitudinal,
        "lateral_report": str(lateral_path),
        "longitudinal_report": str(longitudinal_path),
    }


def _load_python_plant(source_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("manifest_checked_plant", source_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load Python plant source: {source_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _check_offline_plant_defaults(manifest: dict[str, Any], source_path: Path,
                                  python_path: Path, tolerance: float
                                  ) -> dict[str, Any]:
    """Ensure native/Python offline defaults equal the manifest candidate."""
    candidate = manifest["profiles"][manifest["canonical_candidate_profile"]]
    values = candidate["parameters"]
    pose_reference = candidate["pose_reference"]
    expected = dict(values)
    expected["position_offset_from_velocity_point_x_m"] = pose_reference[
        "position_offset_from_velocity_point_x_m"]
    expected["wheel_dynamics_kind"] = "continuous"

    source = source_path.read_text(encoding="utf-8")
    defines = _defines(source)
    native: dict[str, Any] = {}
    for field, macro in PLANT_DEFAULT_MACROS.items():
        if macro not in defines:
            raise ValueError(f"offline plant source is missing {macro}")
        native[field] = _macro_value(macro, defines)
    match = re.search(r"\.wheel_coefficients\s*=\s*\{([^}]*)\}", source,
                      flags=re.DOTALL)
    if match is None:
        raise ValueError("offline plant source has no default wheel coefficients")
    native["wheel_coefficients"] = [
        float(re.sub(r"[fF]", "", item.strip()))
        for item in match.group(1).split(",") if item.strip()
    ]
    native["tire_model"] = "tanh" if "VEHICLE_PLANT_TIRE_TANH" in source else "unknown"

    python_module = _load_python_plant(python_path)
    python_parameters = python_module.PlantParameters()
    python = {
        field: getattr(python_parameters, field)
        for field in (*PLANT_DEFAULT_MACROS, "wheel_coefficients",
                      "tire_model", "wheel_dynamics_kind")
    }
    errors: list[str] = []
    for field, expected_value in expected.items():
        if field not in native and field not in python:
            continue
        if field in native:
            _compare(expected_value, native[field], f"plant.native.{field}",
                     tolerance, errors)
        if field in python:
            _compare(expected_value, python[field], f"plant.python.{field}",
                     tolerance, errors)
    if native["tire_model"] != expected["tire_model"]:
        errors.append("plant.native.tire_model is not the manifest tire model")
    if "VEHICLE_PLANT_WHEEL_DYNAMICS_CONTINUOUS" not in source:
        errors.append("plant.native default does not expose continuous wheel dynamics")
    if errors:
        raise ValueError("offline plant default drift:\n  " + "\n  ".join(errors))
    return {
        "status": "pass",
        "native_source": str(source_path),
        "python_reference": str(python_path),
        "checked_fields": sorted(expected),
    }


def check(manifest_path: Path, header_path: Path, vehicle_path: Path,
          mpc_path: Path | None = None, lateral_report: Path | None = None,
          longitudinal_report: Path | None = None, odom_yaml: Path | None = None,
          odom_cpp: Path | None = None, odom_python: Path | None = None,
          odom_node: Path | None = None, tolerance: float = 2.0e-5,
          require_candidate_compatibility: bool = False,
          plant_source: Path | None = None,
          plant_python: Path | None = None) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported vehicle-model manifest schema")
    profiles = manifest.get("profiles", {})
    runtime = profiles.get("production_mpc_baseline", {})
    constants = runtime.get("constants", {})
    if manifest.get("active_runtime_profile") != "production_mpc_baseline":
        raise ValueError("manifest active runtime profile is not production_mpc_baseline")
    if runtime.get("status") != "active_runtime":
        raise ValueError("production MPC profile is not marked active_runtime")
    candidate_name = manifest.get("canonical_candidate_profile")
    if candidate_name not in profiles or profiles[candidate_name].get("status") != "not_promoted":
        raise ValueError("canonical candidate profile must exist and remain not_promoted")

    header = header_path.read_text(encoding="utf-8")
    defines = _defines(header)
    errors: dict[str, float] = {}
    resolved: dict[str, float] = {}
    for field, macro in CONSTANT_MACROS.items():
        if macro not in defines:
            raise ValueError(f"MPC header is missing canonical macro {macro}")
        actual = _macro_value(macro, defines)
        expected = _number(constants[field], f"profiles.production_mpc_baseline.constants.{field}")
        resolved[field] = actual
        errors[field] = abs(actual - expected)
    max_error = max(errors.values(), default=0.0)

    # Keep the controller's compile-time horizon tied to the same 40 Hz
    # contract used by offline recursive scoring.  This is a timing/config
    # gate only; it does not alter vehicle physics or simulator behaviour.
    horizon_steps_macro = "PREDICTION_HORIZON"
    stage_dt_macro = "TIME_STEP_SECONDS"
    if horizon_steps_macro not in defines or stage_dt_macro not in defines:
        raise ValueError("MPC header is missing active horizon timing macros")
    expected_steps = _number(
        constants.get("prediction_horizon_commands"),
        "profiles.production_mpc_baseline.constants.prediction_horizon_commands")
    expected_stage_dt = _number(
        constants.get("physical_horizon_s"),
        "profiles.production_mpc_baseline.constants.physical_horizon_s") / expected_steps
    actual_steps = _macro_value(horizon_steps_macro, defines)
    actual_stage_dt = _macro_value(stage_dt_macro, defines)
    horizon_errors = {
        "prediction_horizon_commands": abs(actual_steps - expected_steps),
        "prediction_stage_dt_s": abs(actual_stage_dt - expected_stage_dt),
        "prediction_dt_matches_stage_dt_s": abs(
            resolved["prediction_dt_s"] - actual_stage_dt),
    }
    if max(horizon_errors.values()) > tolerance:
        raise ValueError(
            "stale MPC horizon timing detected: "
            + ", ".join(f"{field}={error:.9g}" for field, error in horizon_errors.items()))

    source = vehicle_path.read_text(encoding="utf-8")
    initializer_errors: dict[str, str] = {}
    for field, macro in DEFAULT_INITIALIZER_MACROS.items():
        if source.count(macro) < 2:
            initializer_errors[field] = macro
    if initializer_errors:
        raise ValueError(
            "vehicle_model.c default initializers no longer use canonical macros: "
            + ", ".join(f"{field}={macro}" for field, macro in initializer_errors.items()))
    if max_error > tolerance:
        raise ValueError(
            f"stale MPC constants detected: max error {max_error:.9g} > {tolerance:.9g}; "
            + ", ".join(f"{field}={error:.9g}" for field, error in errors.items()
                         if error > tolerance))

    required_header_markers = (
        "#define NX_GLOBAL 6", "#define NX_AUG 9", "#define NU 2",
        "Control vector width for steering-rate and longitudinal acceleration",
    )
    missing = [marker for marker in required_header_markers if marker not in header]
    if missing:
        raise ValueError("production MPC header contract markers missing: " + ", ".join(missing))
    production_contract = runtime.get("model_contract", {})
    if _fingerprint({"model_contract": production_contract, "resolved_parameters": constants}) != runtime.get("model_fingerprint"):
        raise ValueError("production MPC model fingerprint is stale")

    for marker in ("p->vehicle_mass * longitudinal_acceleration", "Forward Euler"):
        if marker not in source:
            raise ValueError(f"production vehicle-model contract marker missing: {marker}")
    if mpc_path is not None:
        mpc_source = mpc_path.read_text(encoding="utf-8")
        for marker in ("steering_dynamics_coefficients", "expf(-dt_seconds / tau)",
                       "steering_dynamics_next_effective"):
            if marker not in mpc_source:
                raise ValueError(f"production MPC contract marker missing: {marker}")

    candidate = profiles[candidate_name]
    candidate_fingerprint = _fingerprint({
        "model_contract": candidate["model_contract"],
        "resolved_parameters": candidate["parameters"],
    })
    if candidate_fingerprint != candidate.get("model_fingerprint"):
        raise ValueError("offline candidate model fingerprint is stale")
    compatibility = manifest.get("candidate_compatibility", {})
    if compatibility.get("production_profile_compatible") is not False:
        raise ValueError("candidate compatibility must remain explicitly false until migration")
    if require_candidate_compatibility and not compatibility["production_profile_compatible"]:
        raise ValueError("candidate plant is not structurally compatible with production MPC")

    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "pass",
        "manifest": str(manifest_path),
        "header": str(header_path),
        "source": str(vehicle_path),
        "profile": "production_mpc_baseline",
        "max_abs_error": max_error,
        "tolerance": tolerance,
        "resolved_constants": resolved,
        "candidate_compatibility": compatibility,
    }
    if lateral_report is not None and longitudinal_report is not None:
        report["candidate_reports"] = _check_candidate_reports(
            manifest, lateral_report, longitudinal_report, tolerance)
    if all(path is not None for path in (odom_yaml, odom_cpp, odom_python, odom_node)):
        report["odometry"] = _check_odometry(
            manifest, odom_yaml, odom_cpp, odom_python, odom_node, tolerance)
    if (plant_source is None) != (plant_python is None):
        raise ValueError("offline plant source arguments must be supplied together")
    if plant_source is not None and plant_python is not None:
        report["offline_candidate_defaults"] = _check_offline_plant_defaults(
            manifest, plant_source, plant_python, tolerance)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mpc-header", type=Path, required=True)
    parser.add_argument("--vehicle-source", type=Path, required=True)
    parser.add_argument("--mpc-source", type=Path)
    parser.add_argument("--candidate-lateral-report", type=Path)
    parser.add_argument("--candidate-longitudinal-report", type=Path)
    parser.add_argument("--odom-yaml", type=Path)
    parser.add_argument("--odom-cpp-source", type=Path)
    parser.add_argument("--odom-python-source", type=Path)
    parser.add_argument("--odom-node-source", type=Path)
    parser.add_argument("--plant-source", type=Path)
    parser.add_argument("--plant-python-source", type=Path)
    parser.add_argument("--require-candidate-compatibility", action="store_true")
    parser.add_argument("--tolerance", type=float, default=2.0e-5)
    args = parser.parse_args()
    if (args.candidate_lateral_report is None) != (args.candidate_longitudinal_report is None):
        parser.error("candidate report arguments must be supplied together")
    odom_args = (args.odom_yaml, args.odom_cpp_source,
                 args.odom_python_source, args.odom_node_source)
    if any(path is not None for path in odom_args) and not all(path is not None for path in odom_args):
        parser.error("all odometry source arguments must be supplied together")
    report = check(
        args.manifest, args.mpc_header, args.vehicle_source, args.mpc_source,
        args.candidate_lateral_report, args.candidate_longitudinal_report,
        *odom_args, args.tolerance, args.require_candidate_compatibility,
        args.plant_source, args.plant_python_source)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
