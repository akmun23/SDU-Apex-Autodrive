"""Fit and recursively score a first causal longitudinal model.

This is an M0 identification tool, not an MPC tuner.  It only consumes the
offline transition table produced by ``tools/model_id/assemble_transitions.py``
after the source-time and kinematic gates pass.  Simulator truth is therefore
used for offline targets and scoring only; no runtime node imports this module.

The model is intentionally still a candidate.  A 2% claim requires a frozen
model, untouched blind data, and recursive state/horizon scores.  This first
stage reports that gate honestly while the full lateral vehicle model is being
identified.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np


FEATURE_NAMES = (
    "bias", "u", "u_squared", "throttle", "u_times_throttle",
    "throttle_squared",
)
HORIZON_TARGETS_S = (0.05, 0.10, 0.25, 0.50, 1.00, 2.00)
TWO_PERCENT = 0.02


def _float(row: dict[str, str], key: str) -> float | None:
    try:
        value = float(row.get(key, ""))
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _read_transitions(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as stream:
        raw_rows = list(csv.DictReader(stream))
    required = {
        "simulation_time_k_s", "simulation_time_k1_s",
        "simulation_physics_step_k", "simulation_physics_step_k1",
        "dt_sim_s", "u_k_mps", "u_k1_mps", "throttle_k_norm",
    }
    missing = sorted(required.difference(raw_rows[0] if raw_rows else set()))
    if missing:
        raise ValueError(f"{path} is missing required transition fields: {missing}")

    rows: list[dict[str, float]] = []
    for raw in raw_rows:
        keys = (
            "simulation_time_k_s", "simulation_time_k1_s",
            "simulation_physics_step_k", "simulation_physics_step_k1",
            "dt_sim_s", "u_k_mps", "u_k1_mps", "throttle_k_norm",
        )
        values = {key: _float(raw, key) for key in keys}
        if any(value is None for value in values.values()):
            continue
        row = {key: float(value) for key, value in values.items()}
        row["steering_k_rad"] = _float(raw, "steering_k_rad") or 0.0
        rows.append(row)
    return rows


def _transition_view(row: dict[str, float]) -> dict[str, float]:
    """Map the versioned assembled schema to the fit's compact internal view."""
    return {
        "time": row["simulation_time_k1_s"],
        "previous_time": row["simulation_time_k_s"],
        "physics_step": row["simulation_physics_step_k1"],
        "previous_physics_step": row["simulation_physics_step_k"],
        "dt": row["dt_sim_s"],
        "u": row["u_k1_mps"],
        "previous_u": row["u_k_mps"],
        "throttle": row["throttle_k_norm"],
        "steering": row["steering_k_rad"],
    }


def _is_contiguous(previous: dict[str, float], current: dict[str, float]) -> bool:
    """Return whether two rows are adjacent source transitions."""
    return (
        abs(current["previous_time"] - previous["time"]) <= 1.0e-6 and
        abs(current["previous_physics_step"] - previous["physics_step"]) <= 0.5
    )


def _features(rows: list[dict[str, float]]) -> np.ndarray:
    return np.asarray([
        [1.0, row["previous_u"], row["previous_u"] ** 2,
         row["throttle"], row["previous_u"] * row["throttle"],
         row["throttle"] ** 2]
        for row in rows
    ], dtype=float)


def _fit(rows: list[dict[str, float]]) -> tuple[np.ndarray, np.ndarray]:
    if len(rows) < len(FEATURE_NAMES) + 2:
        raise ValueError(f"only {len(rows)} usable transitions; need more data")
    features = _features(rows)
    target = np.asarray([
        (row["u"] - row["previous_u"]) / row["dt"] for row in rows
    ], dtype=float)
    coefficients, _, rank, singular_values = np.linalg.lstsq(
        features, target, rcond=None)
    if rank < len(FEATURE_NAMES):
        raise ValueError(
            f"rank-deficient identification matrix: rank={rank}; "
            "collect richer throttle/speed excitation")
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("identification produced non-finite coefficients")
    return coefficients, singular_values


def _predict_u(
        u: float, throttle: float, dt: float, coefficients: np.ndarray) -> float:
    feature = np.asarray([
        1.0, u, u * u, throttle, u * throttle, throttle * throttle,
    ])
    result = u + dt * float(np.dot(feature, coefficients))
    return result if np.isfinite(result) else float("nan")


def _error_stats(errors: list[float], actual: list[float]) -> dict[str, float | int | None]:
    finite = [value for value in errors if np.isfinite(value)]
    relative = [
        error / abs(value)
        for error, value in zip(errors, actual)
        if np.isfinite(error) and np.isfinite(value) and abs(value) >= 1.0
    ]
    return {
        "count": len(errors),
        "finite_count": len(finite),
        "invalid_count": len(errors) - len(finite),
        "mae": float(np.mean(finite)) if finite else None,
        "p95": float(np.percentile(finite, 95)) if finite else None,
        "max": max(finite, default=None),
        "relative_eligible_count": len(relative),
        "relative_p95": float(np.percentile(relative, 95)) if relative else None,
        "two_percent_gate": bool(relative) and float(np.percentile(relative, 95)) <= TWO_PERCENT,
    }


def _scores(rows: list[dict[str, float]], coefficients: np.ndarray) -> dict[str, object]:
    one_step_errors: list[float] = []
    one_step_actual: list[float] = []
    recursive_errors: list[float] = []
    recursive_actual: list[float] = []
    invalid_rollouts = 0
    predicted: float | None = None
    previous_row: dict[str, float] | None = None

    for row in rows:
        one_step = _predict_u(
            row["previous_u"], row["throttle"], row["dt"], coefficients)
        one_step_errors.append(abs(one_step - row["u"]))
        one_step_actual.append(row["u"])

        if predicted is None or previous_row is None or not _is_contiguous(previous_row, row):
            # A source gap/reset starts a new independently scored rollout.
            predicted = row["previous_u"]
        predicted = _predict_u(predicted, row["throttle"], row["dt"], coefficients)
        if not np.isfinite(predicted):
            invalid_rollouts += 1
            predicted = None
            recursive_errors.append(float("nan"))
        else:
            recursive_errors.append(abs(predicted - row["u"]))
        recursive_actual.append(row["u"])
        previous_row = row

    return {
        "samples": len(rows),
        "invalid_rollout_count": invalid_rollouts,
        "one_step": _error_stats(one_step_errors, one_step_actual),
        "recursive": _error_stats(recursive_errors, recursive_actual),
    }


def _horizon_scores(
        rows: list[dict[str, float]], coefficients: np.ndarray) -> dict[str, object]:
    """Score independent open-loop rollouts at the requested physical horizons."""
    output: dict[str, object] = {}
    for target_s in HORIZON_TARGETS_S:
        errors: list[float] = []
        actual: list[float] = []
        invalid = 0
        for start, origin in enumerate(rows):
            predicted = origin["previous_u"]
            elapsed = 0.0
            previous = origin
            finished = False
            for row in rows[start:]:
                if row is not origin and not _is_contiguous(previous, row):
                    break
                predicted = _predict_u(predicted, row["throttle"], row["dt"], coefficients)
                if not np.isfinite(predicted):
                    invalid += 1
                    break
                elapsed += row["dt"]
                previous = row
                if elapsed + 1.0e-9 >= target_s:
                    errors.append(abs(predicted - row["u"]))
                    actual.append(row["u"])
                    finished = True
                    break
            if not finished and np.isfinite(predicted):
                invalid += 1
        stats = _error_stats(errors, actual)
        stats["invalid_rollout_count"] = invalid
        output[f"{target_s:.2f}s"] = stats
    return output


def _read_quality_report(run_dir: Path) -> dict[str, object]:
    path = run_dir / "assembled" / "timing_quality_report.json"
    if not path.exists():
        raise ValueError(
            f"missing {path}; run assemble_transitions.py before fitting")
    with path.open(encoding="utf-8") as stream:
        report = json.load(stream)
    if report.get("quality_gate_pass") is not True:
        raise ValueError(
            "refusing to fit: assembled timing/kinematic quality gate is false "
            f"(status={report.get('status')!r})")
    return report


def analyze(run_dir: Path) -> dict[str, object]:
    quality_report = _read_quality_report(run_dir)
    transition_path = run_dir / "assembled" / "model_transition_v2.csv"
    rows = [_transition_view(row) for row in _read_transitions(transition_path)]
    if len(rows) < 20:
        raise ValueError(f"only {len(rows)} usable transitions in {transition_path}")
    split = max(1, int(len(rows) * 0.7))
    train = rows[:split]
    validation = rows[split:]
    coefficients, singular_values = _fit(train)
    validation_score = _scores(validation, coefficients)
    report: dict[str, object] = {
        "schema_version": 2,
        "run_dir": str(run_dir),
        "model_boundary": "applied_throttle_to_simulator_body_u",
        "state_source": "simulator_packet_ground_truth_offline_only",
        "feature_names": list(FEATURE_NAMES),
        "coefficients": [float(value) for value in coefficients],
        "design_matrix_rank": len(FEATURE_NAMES),
        "design_matrix_singular_values": [float(value) for value in singular_values],
        "transition_count": len(rows),
        "training_count": len(train),
        "validation_count": len(validation),
        "training_score": _scores(train, coefficients),
        "validation_score": validation_score,
        "validation_recursive_horizon_scores": _horizon_scores(validation, coefficients),
        "source_quality_report": quality_report,
        "acceptance_criterion": {
            "target": "recursive open-loop state/horizon error <= 2%",
            "relative_error_denominator": "abs(actual body speed), eligible only when >= 1 m/s",
            "blind_data_required": True,
            "full_vehicle_state_required": True,
        },
        "acceptance_status": "candidate_only_full_state_blind_recursive_validation_required",
    }
    output_dir = run_dir / "assembled"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "longitudinal_model_v2.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    return report


def main(args=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    options = parser.parse_args(args)
    report = analyze(options.run_dir)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
