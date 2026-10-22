#!/usr/bin/env python3
"""Search a saved MPC state for a full-horizon source-plant continuation.

This is an offline diagnostic for PR-03/04. It optimizes a compact, smooth
30-stage steering/throttle sequence by replaying the production source plant
and enforcing every sampled orientation-aware corridor margin. It does not
publish commands, modify the raceline, or claim Unity-native feasibility.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize

try:
    from tools import audit_mpc_trajectory_feasibility as audit
except ModuleNotFoundError:
    import audit_mpc_trajectory_feasibility as audit


STAGES = 30
DECISION_KNOTS = 10
# Match the production C float command envelope exactly. Decimal CSV dumps of
# these legal values appear microscopically outside the corresponding Python
# double constants; the source plant clamps the steering representation at its
# boundary, just as production does.
THROTTLE_MIN = float(ctypes.c_float(0.01).value)
STEERING_COMMAND_LIMIT_RAD = float(ctypes.c_float(0.5235987756).value)
STATE_COLUMNS = (
    "e_y", "e_psi", "u", "v", "r", "delayed_steer_1_rad",
    "delayed_steer_2_rad", "actual_steering_rad", "delayed_throttle_1",
    "delayed_throttle_2",
)
PLANT_STATE_COLUMNS = (
    "wheel_omega_fl", "wheel_omega_fr", "wheel_omega_rl", "wheel_omega_rr",
    "body_height", "body_height_rate", "roll", "roll_rate", "pitch",
    "pitch_rate", "jounce_fl", "jounce_fr", "jounce_rl", "jounce_rr",
    "low_forward_fl", "low_forward_fr", "low_forward_rl", "low_forward_rr",
    "low_side_fl", "low_side_fr", "low_side_rl", "low_side_rr",
)


class _NullSink:
    def write(self, value: str) -> int:
        return len(value)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_saved_states(path: Path) -> tuple[list[str], list[list[float]]]:
    lines = path.read_text().splitlines()
    header = next((line[2:].split(",") for line in lines
                   if line.startswith("# stamp_ns,")), None)
    if header is None:
        raise ValueError(f"missing commented state header in {path}")
    rows = []
    for row in csv.reader(line for line in lines if line and not line.startswith("#")):
        if len(row) != len(header):
            raise ValueError(f"expected {len(header)} state fields, got {len(row)}")
        values = [float(value) for value in row]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("saved state contains a non-finite value")
        rows.append(values)
    if not rows:
        raise ValueError("saved-state file has no rows")
    return header, rows


def select_saved_state(header: list[str], rows: list[list[float]],
                       progress_m: float, tolerance_m: float,
                       row_index: int | None = None
                       ) -> tuple[list[float], int, float]:
    """Select one state by row or nearest progress, with an explicit tolerance."""
    if not math.isfinite(progress_m) or not math.isfinite(tolerance_m) or tolerance_m < 0.0:
        raise ValueError("progress must be finite and tolerance nonnegative")
    if not rows:
        raise ValueError("saved-state file has no rows")
    try:
        progress_index = header.index("progress_m")
    except ValueError as error:
        raise ValueError("saved-state header is missing progress_m") from error
    if row_index is None:
        index = min(range(len(rows)),
                    key=lambda i: abs(rows[i][progress_index] - progress_m))
    else:
        if not 0 <= row_index < len(rows):
            raise ValueError(f"state row {row_index} is outside [0,{len(rows) - 1}]")
        index = row_index
    distance = abs(rows[index][progress_index] - progress_m)
    if distance > tolerance_m:
        raise ValueError(
            f"selected state is {distance:.6f} m from requested progress")
    return rows[index], index, distance


def row_to_diagnostic(header: list[str], values: list[float]) -> dict[str, Any]:
    row = dict(zip(header, values))
    missing = (set(STATE_COLUMNS) | set(PLANT_STATE_COLUMNS) |
               {"progress_m", "stamp_ns", "suspension_history_initialized",
                "speed_ceiling_mps"}) - row.keys()
    if missing:
        raise ValueError(f"saved state is missing fields: {sorted(missing)}")
    return {
        "state": [row[name] for name in STATE_COLUMNS],
        "plant_internal_state": [row[name] for name in PLANT_STATE_COLUMNS],
        "plant_state_flags": [0, int(row["suspension_history_initialized"]), 0],
        "progress_m": row["progress_m"],
        "source_stamp_ns": int(row["stamp_ns"]),
        "active_speed_ceiling_mps": row["speed_ceiling_mps"],
        "status": "offline_source_continuation_search",
    }


def controls_from_knots(decision: np.ndarray, stages: int = STAGES,
                        knots: int = DECISION_KNOTS) -> list[list[float]]:
    """Linearly expand steering and throttle knot values to 25 ms stages."""
    values = np.asarray(decision, dtype=float)
    if values.shape != (2 * knots,) or not np.isfinite(values).all():
        raise ValueError(f"expected {2 * knots} finite decision values")
    knot_positions = np.linspace(0.0, stages - 1.0, knots)
    stage_positions = np.arange(stages, dtype=float)
    steering = np.interp(stage_positions, knot_positions, values[:knots])
    throttle = np.interp(stage_positions, knot_positions, values[knots:])
    return np.column_stack((steering, throttle)).tolist()


def load_stage_control_seed(path: Path) -> list[list[float]]:
    """Load an exact 30-stage seed exported by this tool's witness CSV."""
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"stage", "command_steering_rad", "command_throttle"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                "control seed must have stage, command_steering_rad, "
                "and command_throttle columns")
        rows = list(reader)
    if len(rows) != STAGES:
        raise ValueError(f"control seed must contain exactly {STAGES} stages")
    controls = []
    for index, row in enumerate(rows, start=1):
        if int(row["stage"]) != index:
            raise ValueError("control seed stages must be ordered 1..30")
        steer = float(row["command_steering_rad"])
        throttle = float(row["command_throttle"])
        if not (math.isfinite(steer) and math.isfinite(throttle)):
            raise ValueError("control seed contains non-finite controls")
        if not (-STEERING_COMMAND_LIMIT_RAD <= steer <= STEERING_COMMAND_LIMIT_RAD):
            raise ValueError("control seed steering is outside production bounds")
        if not (THROTTLE_MIN <= throttle <= 1.0):
            raise ValueError("control seed throttle is outside production bounds")
        controls.append([steer, throttle])
    return controls


def stage_controls_to_knots(
        controls: list[list[float]], knots: int = DECISION_KNOTS,
        throttle_offset: float = 0.0) -> np.ndarray:
    """Project a validated stage seed to knot variables without clipping."""
    values = np.asarray(controls, dtype=float)
    if values.shape != (STAGES, 2) or not np.isfinite(values).all():
        raise ValueError(f"control seed must have shape ({STAGES}, 2)")
    if knots < 2:
        raise ValueError("at least two decision knots are required")
    if not math.isfinite(throttle_offset):
        raise ValueError("throttle offset must be finite")
    throttle = values[:, 1] + throttle_offset
    if np.any(throttle < THROTTLE_MIN) or np.any(throttle > 1.0):
        raise ValueError("throttle offset moves seed outside legal bounds")
    knot_positions = np.linspace(0.0, STAGES - 1.0, knots)
    stage_positions = np.arange(STAGES, dtype=float)
    steer_knots = np.interp(knot_positions, stage_positions, values[:, 0])
    throttle_knots = np.interp(knot_positions, stage_positions, throttle)
    return np.concatenate((steer_knots, throttle_knots))


def seed_decision(diag: dict[str, Any], points: list[dict[str, float]],
                  length: float, steering_gain: float,
                  throttle: float,
                  knots: int = DECISION_KNOTS) -> np.ndarray:
    """Seed path feed-forward steering and constant legal throttle."""
    steering = []
    progress = float(diag["progress_m"])
    speed = max(0.0, float(diag["state"][2]))
    for knot in np.linspace(0.0, STAGES - 1.0, knots):
        reference = audit.sample_path(
            points, length, progress + speed * 0.025 * knot)
        requested = steering_gain * math.atan(
            audit.WHEELBASE_M * reference["curvature"])
        steering.append(max(-STEERING_COMMAND_LIMIT_RAD,
                            min(STEERING_COMMAND_LIMIT_RAD, requested)))
    return np.asarray(steering + [throttle] * knots, dtype=float)


def source_rollout(diag: dict[str, Any], controls: list[list[float]],
                   points: list[dict[str, float]], length: float,
                   step_function: Any, clearance_m: float,
                   preview_m: float, tolerance_m: float,
                   objective_profile: dict[str, float]) -> dict[str, Any]:
    trial = dict(diag)
    trial["candidate_controls"] = controls
    return audit._source_rollout(
        trial, points, length, step_function, csv.writer(_NullSink()),
        clearance_m, clearance_m, preview_m, tolerance_m,
        scenario_label="optimized_source_plant_sequence",
        objective_profile=objective_profile,
        stop_at_corridor_violation=False, collect_stage_trace=True)


def exact_production_objective(result: dict[str, Any]) -> float:
    objective = result.get("scalar_nonlinear_objective")
    if not isinstance(objective, dict):
        return 1.0e12
    total = float(objective["total"])
    return total if math.isfinite(total) else 1.0e12


def optimize_saved_state(diag: dict[str, Any], points: list[dict[str, float]],
                         length: float, step_function: Any,
                         clearance_m: float, preview_m: float,
                         tolerance_m: float,
                         objective_profile: dict[str, float],
                         max_iterations: int,
                         starts: list[tuple[float, float]],
                         knots: int = DECISION_KNOTS,
                         minimum_corridor_slack_m: float = 0.0,
                         extra_initial_decisions: list[tuple[str, np.ndarray]] | None = None
                         ) -> dict[str, Any]:
    if not math.isfinite(minimum_corridor_slack_m) or minimum_corridor_slack_m < 0.0:
        raise ValueError("minimum corridor slack must be finite and nonnegative")
    lower = np.asarray(
        [-STEERING_COMMAND_LIMIT_RAD] * knots + [THROTTLE_MIN] * knots)
    upper = np.asarray(
        [STEERING_COMMAND_LIMIT_RAD] * knots + [1.0] * knots)
    cache: dict[tuple[float, ...], dict[str, Any]] = {}

    def evaluate(vector: np.ndarray) -> dict[str, Any]:
        key = tuple(np.asarray(vector, dtype=float))
        if key not in cache:
            controls = controls_from_knots(vector, knots=knots)
            cache[key] = source_rollout(
                diag, controls, points, length, step_function,
                clearance_m, preview_m, tolerance_m, objective_profile)
        return cache[key]

    def objective(vector: np.ndarray) -> float:
        result = evaluate(vector)
        if result["source_plant_replay_steps"] != STAGES:
            return 1.0e12
        return exact_production_objective(result)

    def corridor_constraints(vector: np.ndarray) -> np.ndarray:
        trace = evaluate(vector)["stage_trace"] or []
        margins = [float(row["corridor_slack_m"]) - minimum_corridor_slack_m
                   for row in trace]
        if len(margins) < STAGES:
            margins.extend([-1.0] * (STAGES - len(margins)))
        return np.asarray(margins[:STAGES])

    arms = []
    initializations = [
        (f"geometric_gain_{steering_gain:g}_throttle_{initial_throttle:g}",
         seed_decision(diag, points, length, steering_gain,
                       initial_throttle, knots))
        for steering_gain, initial_throttle in starts
    ]
    if extra_initial_decisions:
        initializations.extend(extra_initial_decisions)
    if not initializations:
        raise ValueError("at least one optimizer initialization is required")
    for initialization_name, initial_decision in initializations:
        x0 = np.asarray(initial_decision, dtype=float)
        if x0.shape != (2 * knots,) or not np.isfinite(x0).all():
            raise ValueError("optimizer initialization has the wrong shape")
        if np.any(x0 < lower) or np.any(x0 > upper):
            raise ValueError("optimizer initialization is outside control bounds")
        result = minimize(
            objective, x0, method="SLSQP", bounds=list(zip(lower, upper)),
            constraints=[{"type": "ineq", "fun": corridor_constraints}],
            options={"maxiter": max_iterations, "ftol": 1.0e-7,
                     "disp": False})
        final = evaluate(result.x)
        trace = final["stage_trace"] or []
        controls = controls_from_knots(result.x, knots=knots)
        arms.append({
            "initialization": initialization_name,
            "initial_steering_gain": None,
            "initial_throttle": None,
            "solver_success": bool(result.success),
            "solver_message": str(result.message),
            "iterations": int(result.nit),
            "function_evaluations": int(result.nfev),
            "exact_source_steps": int(final["source_plant_replay_steps"]),
            "first_corridor_violation_stage": final["first_corridor_violation_stage"],
            "minimum_corridor_slack_m": final["min_corridor_slack_m"],
            "minimum_required_corridor_slack_m": minimum_corridor_slack_m,
            "progress_gain_m": (float(trace[-1]["progress_m"]) -
                                float(diag["progress_m"])) if trace else None,
            "end_speed_mps": float(trace[-1]["u_mps"]) if trace else None,
            "production_nonlinear_objective": final.get(
                "scalar_nonlinear_objective"),
            "max_abs_cross_track_m": max(
                (abs(float(row["e_y_m"])) for row in trace), default=None),
            "max_abs_heading_error_rad": max(
                (abs(float(row["e_psi_rad"])) for row in trace), default=None),
            "controls": controls,
            "stage_trace": trace,
        })
    arms.sort(key=lambda arm: (
        arm["exact_source_steps"] == STAGES and
        arm["minimum_corridor_slack_m"] is not None and
        arm["minimum_corridor_slack_m"] >= minimum_corridor_slack_m,
        -exact_production_objective(
            {"scalar_nonlinear_objective":
             arm["production_nonlinear_objective"]}),
    ), reverse=True)
    best = arms[0]
    return {
        "schema": "sdu_apex_source_plant_continuation_search/v2",
        "diagnostic_only": True,
        "objective": "minimize the production nonlinear MPC objective under exact source-plant corridor constraints",
        "hard_constraints": [
            "all 30 source-plant stages replayed from the saved 33-state plant and two-slot command queue",
            f"minimum source-plant footprint corridor slack >= {minimum_corridor_slack_m:.6f} m after configured wall clearance",
            "steering within production bounds",
            "MPC throttle within [0.01,1.0]",
        ],
        "decision_parameterization": {
            "stages": STAGES, "knot_count_per_control": knots,
            "stage_period_s": 0.025, "plant_substep_s": 0.001,
        },
        "optimization_runs": len(arms),
        "source_model_witness_found": bool(
            best["exact_source_steps"] == STAGES and
            best["minimum_corridor_slack_m"] is not None and
            best["minimum_corridor_slack_m"] >= minimum_corridor_slack_m),
        "best": best,
        "all_starts": arms,
        "limitation": (
            "A source-plant witness is not Unity-native validation, not a global "
            "infeasibility proof if none is found, and not a full MPC solve."),
    }


def _parse_starts(value: str) -> list[tuple[float, float]]:
    starts = []
    for item in value.split(","):
        gain_text, throttle_text = item.split(":", 1)
        gain, throttle = float(gain_text), float(throttle_text)
        if not math.isfinite(gain) or not 0.0 <= gain <= 2.0:
            raise ValueError("steering seed gains must be in [0,2]")
        if not math.isfinite(throttle) or not THROTTLE_MIN <= throttle <= 1.0:
            raise ValueError("seed throttle must be in [0.01,1]")
        starts.append((gain, throttle))
    if not starts:
        raise ValueError("at least one optimizer start is required")
    return starts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--progress-m", type=float, required=True)
    parser.add_argument("--progress-tolerance-m", type=float, default=0.01)
    parser.add_argument(
        "--state-row", type=int,
        help="select this zero-based saved-state row instead of nearest progress")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-iterations", type=int, default=60)
    parser.add_argument("--decision-knots", type=int, choices=(10, 15, 30),
                        default=DECISION_KNOTS)
    parser.add_argument("--starts", default="0.7:0.01,1.0:0.05")
    parser.add_argument("--seed-control-csv", type=Path,
                        help="30-stage exact source-witness CSV used as an additional initialization")
    parser.add_argument("--seed-throttle-offsets", default="0,0.05,0.10,0.20",
                        help="comma-separated constant throttle offsets for the supplied seed")
    parser.add_argument("--minimum-corridor-slack-m", type=float, default=0.0,
                        help="required extra source-model corridor slack (default 0)")
    parser.add_argument("--seed-only", action="store_true",
                        help="skip geometric initializations; requires --seed-control-csv")
    args = parser.parse_args()
    if not math.isfinite(args.progress_m) or args.progress_tolerance_m < 0.0:
        parser.error("progress must be finite and tolerance nonnegative")
    if args.max_iterations < 1:
        parser.error("max-iterations must be positive")
    if not math.isfinite(args.minimum_corridor_slack_m) or args.minimum_corridor_slack_m < 0.0:
        parser.error("minimum-corridor-slack-m must be finite and nonnegative")
    if args.seed_only and args.seed_control_csv is None:
        parser.error("--seed-only requires --seed-control-csv")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"output directory is not empty: {args.output_dir}")

    header, rows = read_saved_states(args.states)
    try:
        selected, selected_index, distance = select_saved_state(
            header, rows, args.progress_m, args.progress_tolerance_m,
            args.state_row)
    except ValueError as error:
        parser.error(str(error))
    diag = row_to_diagnostic(header, selected)
    points, length = audit.load_trajectory(args.trajectory)
    model = ctypes.CDLL(str(args.library.resolve()))
    step = model.unity_source_plant_step_c
    step.argtypes = [ctypes.POINTER(audit.evaluator.UnitySourcePlantInputC),
                     ctypes.c_double,
                     ctypes.POINTER(audit.evaluator.UnitySourcePlantOutputC)]
    step.restype = ctypes.c_int
    profile = audit.read_runtime_corridor_profile(
        Path("f1tenth_mpc/config/mpc_competition.yaml"))
    objective_profile = audit.read_runtime_objective_profile(
        Path("f1tenth_mpc/config/mpc_competition.yaml"))
    try:
        starts = [] if args.seed_only else _parse_starts(args.starts)
        extra_initial_decisions = []
        seed_controls = None
        if args.seed_control_csv is not None:
            seed_controls = load_stage_control_seed(args.seed_control_csv)
            offsets = [float(value) for value in args.seed_throttle_offsets.split(",")]
            if not offsets or not all(math.isfinite(value) for value in offsets):
                raise ValueError("seed throttle offsets must be finite and nonempty")
            for offset in offsets:
                decision = stage_controls_to_knots(
                    seed_controls, knots=args.decision_knots,
                    throttle_offset=offset)
                extra_initial_decisions.append(
                    (f"source_witness_throttle_offset_{offset:g}", decision))
        if not starts and not extra_initial_decisions:
            raise ValueError("no optimizer initialization was provided")
    except ValueError as error:
        parser.error(str(error))
    report = optimize_saved_state(
        diag, points, length, step,
        profile["corridor_margin_m"], profile["corridor_preview_halfwidth_m"],
        profile["nonlinear_corridor_tolerance_m"], objective_profile,
        args.max_iterations, starts, knots=args.decision_knots,
        minimum_corridor_slack_m=args.minimum_corridor_slack_m,
        extra_initial_decisions=extra_initial_decisions)
    report["inputs"] = {
        "saved_states": str(args.states), "saved_states_sha256": sha256(args.states),
        "trajectory": str(args.trajectory), "trajectory_sha256": sha256(args.trajectory),
        "source_plant_library": str(args.library),
        "source_plant_library_sha256": sha256(args.library),
        "source_plant_cpp_sha256": sha256(Path("f1tenth_mpc/src/unity_source_plant.cpp")),
        "seed_control_csv": (str(args.seed_control_csv)
                             if args.seed_control_csv else None),
        "seed_control_csv_sha256": (sha256(args.seed_control_csv)
                                    if args.seed_control_csv else None),
        "seed_throttle_offsets": args.seed_throttle_offsets,
        "minimum_corridor_slack_m": args.minimum_corridor_slack_m,
        "selected_progress_m": diag["progress_m"],
        "selected_source_stamp_ns": diag["source_stamp_ns"],
        "selected_state_row": selected_index,
        "selected_progress_distance_m": distance,
        "corridor_profile": profile,
        "objective_profile": objective_profile,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "continuation_search.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    best = report["best"]
    with (args.output_dir / "best_source_plant_witness.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "stage", "progress_m", "e_y_m", "e_psi_rad", "u_mps",
            "v_rear_mps", "r_radps", "actual_steering_rad",
            "command_steering_rad", "command_throttle", "applied_throttle",
            "corridor_slack_m",
        ])
        writer.writerows([
            [row[key] for key in (
                "stage", "progress_m", "e_y_m", "e_psi_rad", "u_mps",
                "v_rear_mps", "r_radps", "actual_steering_rad",
                "command_steering_rad", "command_throttle", "applied_throttle",
                "corridor_slack_m")]
            for row in best["stage_trace"]
        ])
    if report["source_model_witness_found"]:
        with (args.output_dir / "mpc_nominal_seed.csv").open("w", newline="") as stream:
            stream.write("# row_index followed by 30 steering/throttle pairs\n")
            csv.writer(stream, lineterminator="\n").writerow(
                [0] + [value for control in best["controls"] for value in control])
        with (args.output_dir / "matched_state.csv").open("w", newline="") as stream:
            stream.write("# exact saved state used by source-plant optimization\n")
            csv.writer(stream, lineterminator="\n").writerow(selected)
    print(json.dumps({
        "source_model_witness_found": report["source_model_witness_found"],
        "selected_progress_m": diag["progress_m"],
        "iterations": best["iterations"],
        "source_steps": best["exact_source_steps"],
        "minimum_corridor_slack_m": best["minimum_corridor_slack_m"],
        "progress_gain_m": best["progress_gain_m"],
        "end_speed_mps": best["end_speed_mps"],
        "output_dir": str(args.output_dir),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
