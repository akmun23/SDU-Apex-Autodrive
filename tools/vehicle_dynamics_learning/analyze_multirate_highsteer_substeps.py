#!/usr/bin/env python3
"""Compare internal substep rates on whole-run high-steering captures.

This evaluation-only companion uses the frozen continuous vector field from
the multirate study. It does not retrain the model or touch a simulator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from tools.vehicle_dynamics_learning.evaluate_history_context_highsteer import (
    _capture_from_archive,
)
from tools.vehicle_dynamics_learning.run_multirate_vector_field_study import (
    ContinuousStateVectorField,
    _rollout_run,
)
from tools.vehicle_dynamics_learning.run_truncated_history_transition import ROOT
from tools.vehicle_dynamics_learning.train_augmented_state_space_plant import (
    sha256_file,
)


STUDY_ROOT = (
    ROOT / "live_runs/derived_dynamics_learning_20260928"
    / "full_modeling_reset_20261001"
    / "replacement_offline_sim_raceline_20261003"
    / "full_throttle_domain_v1/next_phase_after_2129427"
    / "multirate_vector_field_v1")
CHECKPOINT = STUDY_ROOT / "ode_1khz/checkpoint.pt"
METADATA = STUDY_ROOT / "ode_1khz/metadata.json"
HIGHSTEER_SOURCE = (
    ROOT / "live_runs/derived_dynamics_learning_20260928"
    / "full_modeling_reset_20261001"
    / "highsteer_75_heldout_validation_plus_r03_20261004"
    / "openplane_dynamics.npz")
EXPECTED_RUNS = {
    "openplane_highsteer_75_validation_r01",
    "openplane_highsteer_75_validation_r02",
    "openplane_highsteer_75_validation_r03_20261004",
}
OUTPUT = STUDY_ROOT / "highsteer_substep_diagnostic_20261004.json"
HORIZON = 80
STARTS_PER_RUN = 128
SUBSTEP_RATES = {"40_hz": 1, "200_hz": 5, "1000_hz": 25}
BOOTSTRAP_REPLICATES = 10000


def _select_highsteer_starts(capture, horizon: int = HORIZON
                             ) -> dict[str, list[tuple[int, int, int]]]:
    selected = {run_id: [] for run_id in EXPECTED_RUNS}
    for sequence_index, ((begin_raw, end_raw), run_raw) in enumerate(
            zip(capture.bounds, capture.sequence_run)):
        run_id = str(capture.run_ids[int(run_raw)])
        if run_id not in selected or str(capture.splits[int(run_raw)]) != "validation":
            continue
        begin, end = int(begin_raw), int(end_raw)
        if (end - begin <= horizon + 1
                or not np.isfinite(capture.input_features[begin:end]).all()
                or not np.isfinite(capture.body[begin:end]).all()
                or not np.isfinite(capture.frames[begin:end]).all()
                or np.any(np.diff(capture.packet[begin:end]) != 1)):
            continue
        for row in range(1, end - begin - horizon):
            speed = abs(float(capture.body[begin + row, 0]))
            steering = abs(float(capture.frames[begin + row, 3]))
            if 7.0 <= speed < 9.0 and steering >= 0.30:
                selected[run_id].append((0, sequence_index, row))
    if any(not refs for refs in selected.values()):
        counts = {run: len(refs) for run, refs in selected.items()}
        raise RuntimeError(f"high-steering whole-run coverage missing: {counts}")
    for run_id, refs in selected.items():
        if len(refs) > STARTS_PER_RUN:
            indexes = np.linspace(0, len(refs) - 1, STARTS_PER_RUN,
                                  dtype=np.int64)
            selected[run_id] = [refs[int(index)] for index in indexes]
    return selected


def _macro_by_run(per_run: dict[str, dict[str, float]], horizon: str,
                  metric: str) -> float:
    return float(np.mean([row["horizons"][horizon][metric]
                          for row in per_run.values()]))


def _paired_delta(per_run: dict[str, dict[str, dict[str, float]]],
                   candidate: str, reference: str, horizon: str,
                   metric: str) -> dict[str, Any]:
    runs = sorted(set(per_run[candidate]) & set(per_run[reference]))
    deltas = np.asarray([
        per_run[candidate][run]["horizons"][horizon][metric]
        - per_run[reference][run]["horizons"][horizon][metric]
        for run in runs], dtype=np.float64)
    rng = np.random.default_rng(20261004)
    indexes = rng.integers(0, len(deltas),
                           size=(BOOTSTRAP_REPLICATES, len(deltas)))
    interval = np.quantile(deltas[indexes].mean(axis=1), [0.025, 0.975])
    return {
        "direction": "candidate minus reference; negative favors candidate",
        "run_macro_delta": float(deltas.mean()),
        "run_cluster_bootstrap_95pct_ci": interval.tolist(),
        "independent_run_count": len(runs),
        "per_run_delta": dict(zip(runs, deltas.tolist())),
    }


def run(device_name: str = "cpu", output_path: Path = OUTPUT
        ) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(1)

    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    norm = {key: np.asarray(value, dtype=np.float32)
            for key, value in metadata["normalization"].items()}
    model = ContinuousStateVectorField(norm["rate_mean"], norm["rate_scale"])
    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()

    capture, pose = _capture_from_archive(HIGHSTEER_SOURCE, EXPECTED_RUNS)
    refs = _select_highsteer_starts(capture)
    data = SimpleNamespace(captures=[capture], poses=[pose])
    per_run: dict[str, dict[str, dict[str, Any]]] = {}
    for name, substeps in SUBSTEP_RATES.items():
        per_run[name] = {}
        for run_id, rows in sorted(refs.items()):
            result = _rollout_run(
                data, run_id, rows, model, "ode", substeps, norm, device,
                HORIZON)
            per_run[name][run_id] = {
                "start_count": result["start_count"],
                "horizons": result["horizons"],
            }

    metrics = (
        "position_radial_trajectory_rmse_m",
        "heading_trajectory_rmse_rad",
        "u_rmse_mps", "v_rmse_mps", "yaw_rate_rmse_rps")
    macro = {
        name: {metric: _macro_by_run(per_run[name], str(HORIZON), metric)
               for metric in metrics}
        for name in SUBSTEP_RATES
    }
    paired = {
        candidate: {
            metric: _paired_delta(per_run, candidate, "40_hz",
                                  str(HORIZON), metric)
            for metric in metrics}
        for candidate in ("200_hz", "1000_hz")
    }
    report = {
        "study": "same frozen continuous plant; internal substep rate on high steering",
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "data_sha256": sha256_file(HIGHSTEER_SOURCE),
        "source_control_rate_hz": 40.0,
        "command_hold": "zero-order hold for each recorded 25 ms control interval",
        "available_intermediate_state_labels": False,
        "highsteer_start_region": {
            "forward_speed_mps": "[7, 9)",
            "absolute_steering_feedback_rad": ">=0.30",
            "horizon_seconds": HORIZON * 0.025,
            "starts_per_independent_run": {
                run_id: len(rows) for run_id, rows in refs.items()},
        },
        "internal_rates_hz": {name: round(40 * substeps)
                              for name, substeps in SUBSTEP_RATES.items()},
        "run_macro_metrics": macro,
        "paired_against_40_hz": paired,
        "per_run": per_run,
        "independent_run_count": len(EXPECTED_RUNS),
        "simulator_launched": False,
        "model_retrained": False,
        "production_integration": False,
        "interpretation_limit": (
            "Only three development high-steering captures are available and "
            "were previously used during model development. Substep states are "
            "predictions, not labels; this isolates numerical step size for one "
            "frozen vector field, not whether a different identified continuous "
            "plant could benefit from substeps."),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n",
                           encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    report = run(args.device, output)
    print(json.dumps({
        "output": output.relative_to(ROOT).as_posix(),
        "macro_2s": report["run_macro_metrics"],
        "paired_position_vs_40hz": {
            name: report["paired_against_40_hz"][name][
                "position_radial_trajectory_rmse_m"]
            for name in ("200_hz", "1000_hz")},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
