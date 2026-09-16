#!/usr/bin/env python3
"""Compare equal-physical-horizon MPC discretizations offline.

This is a numerical/model replay benchmark only.  It does not edit MPC
constants, start the simulator, or use future measured states during a
rollout.  Every candidate predicts exactly 0.75 s, while changing the stage
grid:

* N=25, dt=0.030 s (same 0.75 s physical horizon, coarser than 40 Hz);
* N=30, dt=0.025 s (one stage per 40 Hz control sample); and
* N=50, dt=0.015 s (resolution diagnostic).

The ``internal_substep_s`` value is varied independently so a finer internal
integrator is not confused with a different command grid. The active stage
grid stays N=30 in every comparison:

* one 25 ms internal step;
* two 12.5 ms internal steps;
* four 6.25 ms internal steps; and
* the 2 ms offline reference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import structured_vehicle_plant as plant  # noqa: E402
from n30_stage_map import f25  # noqa: E402
from fit_speed_regime_vehicle_model import (  # noqa: E402
    _raceline_context,
    _read_run,
    _stats,
    _steering_command,
)
from score_raceline_model import frenet_error, operating_class  # noqa: E402


PHYSICAL_HORIZON_S = 0.75
MAX_SPEED_MPS = 16.0


def _score_variant(runs: Sequence[list[dict[str, float]]],
                   parameters: plant.PlantParameters,
                   stage_count: int, stage_dt_s: float,
                   internal_substep_s: float,
                   origin_stride: int = 8,
                   context: dict[str, Any] | None = None) -> dict[str, Any]:
    if abs(stage_count * stage_dt_s - PHYSICAL_HORIZON_S) > 1.0e-9:
        raise ValueError("stage grid must represent exactly 0.75 s")
    fields = ("position_m", "e_cross_m", "e_along_m", "heading_rad",
              "e_heading_rad", "u_mps", "v_mps", "r_radps")
    context = context or _raceline_context()
    errors = {field: [] for field in fields}
    class_errors = {
        "core": {field: [] for field in fields},
        "guard": {field: [] for field in fields},
    }
    attempted = 0
    skipped = 0
    for rows in runs:
            # Source transition times are accumulated from dt_sim_s.  A
            # stage holds the command active at its start; no future state is
            # read or interpolated into the prediction.
            source_times = np.zeros(len(rows) + 1, dtype=float)
            for index, row in enumerate(rows):
                source_times[index + 1] = source_times[index] + row["dt_sim_s"]
            for origin in range(0, len(rows), origin_stride):
                first = rows[origin]
                first_class = operating_class(
                    first["x_k_m"], first["y_k_m"], first["u_k_mps"],
                    context["raceline"], context["core_bins"],
                    context["guard_bins"],
                    context["speed_bin_width_mps"],
                    context["curvature_bin_width_radpm"], MAX_SPEED_MPS)
                if (not np.isfinite(first["wheel_k_mps"]) or
                        not np.isfinite(first["delta_k_rad"]) or
                        first_class == "stress" or
                        not (0.0 <= first["u_k_mps"] <= MAX_SPEED_MPS)):
                    continue
                attempted += 1
                state = np.asarray([
                    first["x_k_m"], first["y_k_m"], first["yaw_k_rad"],
                    first["u_k_mps"], first["v_k_mps"], first["r_k_radps"],
                    first["delta_k_rad"], first["wheel_k_mps"],
                ], dtype=float)
                end_time = source_times[origin] + PHYSICAL_HORIZON_S
                source_index = origin
                invalid = False
                for stage in range(stage_count):
                    stage_time = source_times[origin] + stage * stage_dt_s
                    while (source_index + 1 < len(source_times) and
                           source_times[source_index + 1] <= stage_time + 1.0e-10):
                        source_index += 1
                    if source_index >= len(rows):
                        invalid = True
                        break
                    row = rows[source_index]
                    if (int(row["segment_id"]) != int(first["segment_id"]) or
                            operating_class(
                                row["x_k_m"], row["y_k_m"], row["u_k_mps"],
                                context["raceline"], context["core_bins"],
                                context["guard_bins"],
                                context["speed_bin_width_mps"],
                                context["curvature_bin_width_radpm"],
                                MAX_SPEED_MPS) == "stress" or
                            not (0.0 <= row["u_k_mps"] <= MAX_SPEED_MPS) or
                            not (0.0 <= row["u_k1_mps"] <= MAX_SPEED_MPS)):
                        invalid = True
                        break
                    state = f25(
                        state, _steering_command(row),
                        row["applied_throttle_norm_k1"], parameters,
                        internal_substep_s=internal_substep_s)
                if invalid:
                    skipped += 1
                    continue
                truth_index = int(np.searchsorted(
                    source_times, end_time, side="left")) - 1
                if truth_index < origin or truth_index >= len(rows):
                    skipped += 1
                    continue
                truth = rows[truth_index]
                frenet = frenet_error(
                    state[0], state[1], state[2], truth["x_k1_m"],
                    truth["y_k1_m"], truth["yaw_k1_rad"], context["raceline"])
                values = {
                    "position_m": float(frenet["position_m"]),
                    "e_cross_m": float(frenet["e_cross_m"]),
                    "e_along_m": float(frenet["e_along_m"]),
                    "heading_rad": float(frenet["e_heading_rad"]),
                    "e_heading_rad": float(frenet["e_heading_rad"]),
                    "u_mps": float(state[3] - truth["u_k1_mps"]),
                    "v_mps": float(state[4] - truth["v_k1_mps"]),
                    "r_radps": float(state[5] - truth["r_k1_radps"]),
                }
                for field, value in values.items():
                    errors[field].append(value)
                    class_errors[first_class][field].append(value)
    return {
        field: _stats(values) for field, values in errors.items()
    } | {
        "core": {
            field: _stats(values) for field, values in class_errors["core"].items()
        },
        "guard": {
            field: _stats(values) for field, values in class_errors["guard"].items()
        },
        "evaluation": {
            "attempted_origins": attempted,
            "valid_origins": len(errors["position_m"]),
            "skipped_origins": skipped,
            "physical_horizon_s": PHYSICAL_HORIZON_S,
            "stage_count": stage_count,
            "stage_dt_s": stage_dt_s,
            "internal_substep_s": internal_substep_s,
        }
    }


def run(run_specs: Sequence[str], candidate_json: Path,
        output: Path, origin_stride: int = 8) -> dict[str, Any]:
    report = json.loads(candidate_json.read_text(encoding="utf-8"))
    base = plant.PlantParameters.from_manifest()
    parameters = base
    envelope_path = report.get("operating_envelope", {}).get("raceline_file")
    context = _raceline_context(Path(envelope_path)) if envelope_path else _raceline_context()
    if "candidate_parameters" in report:
        values = report["candidate_parameters"]
        parameters = plant.PlantParameters(
            **{
                **base.__dict__,
                "steering_dynamics_kind": values[
                    "steering_dynamics_kind"],
                "steering_lag_time_constant_s": float(values.get(
                    "steering_lag_time_constant_s", 0.0)),
                "tire_model": "regime_speed_combined_tanh",
                "lateral_cf_regimes_n_per_rad": tuple(values[
                    "lateral_cf_regimes_n_per_rad"]),
                "lateral_cr_regimes_n_per_rad": tuple(values[
                    "lateral_cr_regimes_n_per_rad"]),
                "lateral_df_regimes_n": tuple(values[
                    "lateral_df_regimes_n"]),
                "lateral_dr_regimes_n": tuple(values[
                    "lateral_dr_regimes_n"]),
                "combined_slip_gain": float(values[
                    "combined_slip_gain"]),
                "steering_rate_force_gain_n_per_radps": float(values.get(
                    "steering_rate_force_gain_n_per_radps", 0.0)),
                "steering_rate_moment_gain_nm_per_radps": float(values.get(
                    "steering_rate_moment_gain_nm_per_radps", 0.0)),
                "force_max_regimes_n": tuple(values[
                    "force_max_regimes_n"]),
                "slip_gain_regimes_per_mps": tuple(values[
                    "slip_gain_regimes_per_mps"]),
                "coast_speed_drag_n_per_mps": 0.0,
            })
    runs = [_read_run(spec) for spec in run_specs]
    grids = {
        "N30_dt025_internal025": (30, 0.025, 0.025),
        "N30_dt025_internal0125": (30, 0.025, 0.0125),
        "N30_dt025_internal00625": (30, 0.025, 0.00625),
        "N30_dt025_internal002_reference": (30, 0.025, 0.002),
    }
    scores = {
        name: _score_variant(runs, parameters, *grid, origin_stride, context)
        for name, grid in grids.items()
    }
    result = {
        "schema_version": 1,
        "status": "offline_equal_horizon_discretization_benchmark",
        "simulator_modified": False,
        "production_mpc_modified": False,
        "candidate_source": str(candidate_json),
        "run_specs": list(run_specs),
        "physical_horizon_s": PHYSICAL_HORIZON_S,
        "scores": scores,
        "interpretation": {
            "same_horizon_comparison": True,
            "same_N30_stage_grid": True,
            "stage_grid_and_internal_substep_separated": True,
            "controls_are_zero_order_held_at_stage_start": True,
            "primary_mpc_horizon": "0.75s",
            "primary_mpc_commands": 30,
            "primary_mpc_command_dt_s": 0.025,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--origin-stride", type=int, default=8)
    args = parser.parse_args()
    result = run(args.run, args.candidate, args.output, args.origin_stride)
    print(json.dumps({
        "output": str(args.output),
        "status": result["status"],
        "scores": {
            name: value["position_m"]["p95"]
            for name, value in result["scores"].items()
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
