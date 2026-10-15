#!/usr/bin/env python3
"""Free-run raw rear encoder rates while oracle-conditioning other car states.

This isolates wheel-state recursion: future encoder truth is never an input,
but simulator-truth body state and measured actuator/filtered-wheel channels
are supplied at each step. It is not a free-running vehicle simulator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from tools.vehicle_dynamics_learning.fit_fixed40_wheel_response_candidate import (
    DEFAULT_DATASET,
    DEFAULT_OUTPUT,
    DEFAULT_PRACTICE,
    VARIANT_NAMES,
)
from tools.vehicle_dynamics_learning.train_nssm import _load_dataset


ROOT = Path(__file__).resolve().parents[2]
LAGS = (1, 2, 4, 8, 16, 32, 64)
HORIZONS = (1, 4, 10, 20, 30, 40, 60, 80, 120, 200)
MAX_SPEED_MPS = 12.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _windows(data: dict[str, Any], split: str, horizon: int,
             max_per_run: int) -> list[dict[str, int]]:
    bounds = np.asarray(data["bounds"], dtype=np.int64)
    seq_run = np.asarray(data["seq_run"], dtype=np.int32)
    splits = np.asarray(data["splits"]).astype(str)
    rigid = np.asarray(data["simulator_rigid_state"], dtype=np.float64)
    raw_valid = np.asarray(data["encoder_raw_valid"], dtype=bool)
    dt_s = np.asarray(data["dt_s"], dtype=np.float64)
    speed = np.hypot(rigid[:, 7], rigid[:, 8])
    context = max(LAGS)
    by_run: dict[int, list[dict[str, int]]] = {}
    for sequence, ((start_raw, end_raw), run_raw) in enumerate(zip(bounds, seq_run)):
        start, end, run = int(start_raw), int(end_raw), int(run_raw)
        if splits[run] != split or end - start < context + horizon + 1:
            continue
        candidates = np.arange(start + context, end - horizon, dtype=np.int64)
        candidates = candidates[(speed[candidates] <= MAX_SPEED_MPS)
                                & np.isfinite(rigid[candidates, 7:10]).all(axis=1)]
        for index in candidates:
            if (not raw_valid[index - context:index + horizon + 1].all()
                    or not np.isclose(
                        dt_s[index - context + 1:index + horizon + 1],
                        0.025, rtol=0.0, atol=1e-7).all()):
                continue
            by_run.setdefault(run, []).append({
                "run": run, "sequence": sequence, "start": int(index)})
    selected: list[dict[str, int]] = []
    for run, candidates in sorted(by_run.items()):
        candidates.sort(key=lambda row: (row["sequence"], row["start"]))
        if len(candidates) > max_per_run:
            ix = np.linspace(0, len(candidates) - 1, max_per_run, dtype=np.int64)
            candidates = [candidates[int(i)] for i in np.unique(ix)]
        selected.extend(candidates)
    return selected


def _parent_feature_arrays(data: dict[str, Any]
                           ) -> tuple[np.ndarray, np.ndarray]:
    rigid = np.asarray(data["simulator_rigid_state"], dtype=np.float32)
    frames = np.asarray(data["frames"], dtype=np.float32)
    body = np.column_stack((rigid[:, (7, 8, 12)], frames[:, 3:9]))
    half_track = 0.236 / 2.0
    mismatch = np.column_stack((
        0.5 * (frames[:, 5] + frames[:, 6]) - rigid[:, 7],
        frames[:, 5] - frames[:, 6],
        frames[:, 5] - (rigid[:, 7] - half_track * rigid[:, 12]),
        frames[:, 6] - (rigid[:, 7] + half_track * rigid[:, 12]),
        frames[:, 8] - frames[:, 4],
        frames[:, 7] - frames[:, 3],
    ))
    return body, mismatch.astype(np.float32)


def _parent_features(body: np.ndarray, mismatch: np.ndarray,
                     source: np.ndarray, variant: str) -> np.ndarray:
    current = body[source]
    parts = [current]
    if variant != "current_plus_raw":
        for lag in LAGS:
            parts.append(body[source] - body[source - lag])
    if variant == "history_mismatch_plus_raw":
        parts.append(mismatch[source])
    return np.column_stack(parts).astype(np.float32)


def _rollout(data: dict[str, Any], bundle: dict[str, Any],
             windows: list[dict[str, int]], horizon: int
             ) -> dict[str, Any]:
    raw_truth = np.asarray(data["encoder_raw_surface_mps"], dtype=np.float64)
    run_ids = np.asarray(data["run_ids"]).astype(str)
    starts = np.asarray([row["start"] for row in windows], dtype=np.int64)
    runs = np.asarray([row["run"] for row in windows], dtype=np.int32)
    count = len(starts)
    body, mismatch = _parent_feature_arrays(data)
    raw_history = np.stack([
        raw_truth[start - max(LAGS):start + 1] for start in starts
    ]).astype(np.float32)
    initial = raw_history[:, -1].astype(np.float64)
    prediction = np.empty((count, horizon + 1, 2), dtype=np.float64)
    prediction[:, 0] = initial
    models = bundle["models"]
    variant = str(bundle["feature_variant"])

    for step in range(horizon):
        source = starts + step
        parent = _parent_features(body, mismatch, source, variant)
        raw_parts = [raw_history[:, -1, :]]
        for lag in LAGS:
            raw_parts.append(raw_history[:, -1, :] - raw_history[:, -1 - lag, :])
        x = np.column_stack((parent, *raw_parts)).astype(np.float32)
        delta = np.column_stack([model.predict(x) for model in models])
        following = raw_history[:, -1, :].astype(np.float64) + delta
        prediction[:, step + 1] = following
        if step + 1 < horizon:
            raw_history = np.concatenate((
                raw_history[:, 1:, :], following[:, None, :].astype(np.float32)),
                axis=1)

    target = raw_truth[starts + horizon]
    baseline = np.broadcast_to(initial[:, None, :], (count, horizon + 1, 2))[:, horizon]
    error, baseline_error = prediction[:, horizon] - target, baseline - target
    per_run: dict[str, Any] = {}
    for run in sorted(np.unique(runs).tolist()):
        mask = runs == run
        per_run[str(run_ids[run])] = {
            "windows": int(mask.sum()),
            "candidate_pair_rmse_mps": float(np.sqrt(np.mean(error[mask] ** 2))),
            "persistence_pair_rmse_mps": float(np.sqrt(np.mean(baseline_error[mask] ** 2))),
            "candidate_left_rmse_mps": float(np.sqrt(np.mean(error[mask, 0] ** 2))),
            "candidate_right_rmse_mps": float(np.sqrt(np.mean(error[mask, 1] ** 2))),
        }
    differences = np.asarray([
        row["candidate_pair_rmse_mps"] - row["persistence_pair_rmse_mps"]
        for row in per_run.values()], dtype=np.float64)
    rng = np.random.default_rng(20261010 + horizon + len(differences))
    draws = differences[rng.integers(
        0, len(differences), (10000, len(differences)))].mean(axis=1)
    signature = hashlib.sha256(json.dumps(
        [(r["run"], r["sequence"], r["start"]) for r in windows],
        separators=(",", ":")).encode()).hexdigest()
    return {
        "horizon_steps": horizon,
        "horizon_seconds": horizon * 0.025,
        "windows": len(windows),
        "window_signature_sha256": signature,
        "independent_runs": len(per_run),
        "weak_evidence_under_three_runs": len(per_run) < 3,
        "candidate_macro_run_pair_rmse_mps": float(np.mean([
            row["candidate_pair_rmse_mps"] for row in per_run.values()])),
        "persistence_macro_run_pair_rmse_mps": float(np.mean([
            row["persistence_pair_rmse_mps"] for row in per_run.values()])),
        "candidate_minus_persistence_run_bootstrap_95pct_mps": [
            float(value) for value in np.quantile(draws, (0.025, 0.975))],
        "runs_better_than_persistence": int(np.sum(differences < 0.0)),
        "per_run": per_run,
    }


def evaluate(bundle_path: Path, dataset_path: Path, practice_path: Path,
             output_path: Path, max_per_run: int) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite wheel recursion report: {output_path}")
    bundle = joblib.load(bundle_path)
    if not bundle.get("research_only") or not bundle.get("models"):
        raise ValueError("not a research fixed-40 wheel candidate")
    if bundle.get("history_lags_steps") != list(LAGS):
        raise ValueError("wheel bundle lags do not match this evaluator")
    if _sha256(dataset_path) != bundle["training_dataset_sha256"]:
        raise ValueError("OpenPlane dataset differs from candidate training data")
    results: dict[str, Any] = {}
    for label, path, split in (("openplane_validation", dataset_path, "validation"),
                               ("unseen_practice", practice_path, "unseen_practice")):
        data = _load_dataset(path)
        if split not in set(np.asarray(data["splits"]).astype(str)):
            raise ValueError(f"{path} has no {split!r} split")
        results[label] = {}
        for horizon in HORIZONS:
            windows = _windows(data, split, horizon, max_per_run)
            results[label][f"{horizon * 0.025:.3f}s"] = (
                _rollout(data, bundle, windows, horizon) if windows else
                {"windows": 0, "reason": "no valid contiguous windows"})
    report = {
        "schema_version": 1,
        "candidate_bundle": str(bundle_path.relative_to(ROOT)),
        "candidate_bundle_sha256": _sha256(bundle_path),
        "openplane_dataset_sha256": _sha256(dataset_path),
        "practice_dataset_sha256": _sha256(practice_path),
        "conditioning_policy": {
            "future_raw_encoder_truth_used_as_input": False,
            "raw_wheel_history_initialized_from_truth_at_rollout_start": True,
            "future_simulator_truth_body_state_used_as_input": True,
            "future_actuator_feedback_and_filtered_wheels_used_as_input": True,
            "recorded_commands_are_replay_inputs": True,
            "interpretation": "wheel-only recursive diagnostic; not a free-running vehicle simulator",
        },
        "test_and_final_test_policy": "sealed; validation and unseen practice only",
        "horizons": results,
        "limitations": [
            "oracle conditioning isolates rear encoder-rate dynamics from body-model error",
            "two unseen practice runs are weak transfer evidence",
            "encoder rate is a measured wheel-rotation proxy, not true tire angular-speed truth",
            "no odometry/MPC/runtime integration",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_OUTPUT / "fixed40_wheel_response_candidate.joblib")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--practice-dataset", type=Path, default=DEFAULT_PRACTICE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "recursive_evaluation.json")
    parser.add_argument("--max-windows-per-run", type=int, default=24)
    args = parser.parse_args()
    report = evaluate(args.bundle.resolve(), args.dataset.resolve(),
                      args.practice_dataset.resolve(), args.output.resolve(),
                      args.max_windows_per_run)
    compact = {split: {h: {key: value.get(key) for key in (
        "candidate_macro_run_pair_rmse_mps",
        "persistence_macro_run_pair_rmse_mps",
        "candidate_minus_persistence_run_bootstrap_95pct_mps",
        "independent_runs", "runs_better_than_persistence")}
        for h, value in values.items()} for split, values in report["horizons"].items()}
    print(json.dumps({"output": str(args.output), "results": compact}, indent=2))


if __name__ == "__main__":
    main()
