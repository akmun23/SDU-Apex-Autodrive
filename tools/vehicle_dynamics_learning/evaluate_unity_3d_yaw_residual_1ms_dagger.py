#!/usr/bin/env python3
"""Refit the 1 ms yaw residual on source-plant rollout states.

The seed model has strong teacher-forced one-step transfer but unstable free
rollout. This experiment collects corrective labels on its own predicted r07
states, mixes them with r07 truth-state labels, and scores one command-only
free rollout on the untouched r11 capture. No r11 labels enter fitting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from tools.vehicle_dynamics_learning.evaluate_unity_3d_yaw_residual import (
    DEFAULT_SLIP_FIT,
    DEFAULT_SUSPENSION_FIT,
    DEFAULT_TIRE_FIT,
    DEFAULT_WHEEL_FIT,
    FEATURE_NAMES,
    ROOT,
    _evaluation_starts,
    _features,
)
from tools.vehicle_dynamics_learning.evaluate_unity_3d_yaw_residual_1ms import (
    _evaluate,
    _training_rows,
)
from tools.vehicle_dynamics_learning.replay_unity_source_3d_plant import (
    _advance,
    _initial_state,
    _metadata,
    _prepare_constants,
)
from tools.vehicle_dynamics_learning.replay_unity_source_mechanics import (
    DEFAULT_TRAIN,
    DEFAULT_VALIDATION,
    DT_S,
    WHEELS,
    _load_capture,
)


SEED_MODEL = ROOT / (
    "live_runs/unity_exact_binary_audit_20261012/"
    "source_mechanics_3d_yaw_residual_1ms_20261013.joblib")
DEFAULT_OUTPUT = ROOT / (
    "live_runs/unity_exact_binary_audit_20261012/"
    "source_mechanics_3d_yaw_residual_1ms_dagger_20261013.json")
ROLLOUT_HORIZON_S = 5.0
ROLLOUT_STRIDE_S = 5.0
DATA_STRIDE_STEPS = 5
MODEL_MIN_SPEED_MPS = 1.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _collect_rollout_rows(path: Path, constants: object,
                          seed_model: HistGradientBoostingRegressor
                          ) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    data, _ = _load_capture(path)
    horizon_steps = round(ROLLOUT_HORIZON_S / DT_S)
    stride_steps = round(ROLLOUT_STRIDE_S / DT_S)
    starts = _evaluation_starts(data, horizon_steps, stride_steps)
    if not starts:
        raise ValueError(f"no grounded rollout windows in {path}")

    rows_per_window = (horizon_steps + DATA_STRIDE_STEPS - 1) // DATA_STRIDE_STEPS
    features = np.empty((len(starts) * rows_per_window, len(FEATURE_NAMES)),
                        dtype=np.float32)
    targets = np.empty(len(features), dtype=np.float32)
    row = 0
    skipped_low_speed = 0
    maximum_state_error_m = 0.0
    with threadpool_limits(limits=1):
        for start in starts:
            state = _initial_state(data, start)
            for offset in range(horizon_steps):
                index = start + offset
                steering = float(data["steering_command"][index])
                throttle = float(data["throttle_command"][index])
                previous_steering = float(
                    data["steering_command"][max(0, index - 1)])
                previous_throttle = float(
                    data["throttle_command"][max(0, index - 1)])
                feature = _features(
                    state, steering, throttle, previous_steering,
                    previous_throttle, constants)
                base_next = _advance(state, steering, throttle, constants, DT_S)

                speed = float(np.linalg.norm(state[7:10]))
                if (offset % DATA_STRIDE_STEPS == 0
                        and speed >= MODEL_MIN_SPEED_MPS
                        and all(data[f"{wheel}_grounded"][index] > 0.5
                                for wheel in WHEELS)):
                    features[row] = feature
                    targets[row] = (_initial_state(data, index + 1)[11]
                                     - base_next[11])
                    row += 1
                elif offset % DATA_STRIDE_STEPS == 0:
                    skipped_low_speed += 1

                if speed >= MODEL_MIN_SPEED_MPS:
                    residual_delta = float(seed_model.predict(
                        feature.reshape(1, -1))[0])
                else:
                    residual_delta = 0.0
                state = _advance(
                    state, steering, throttle, constants, DT_S,
                    residual_delta / DT_S)

                if offset % DATA_STRIDE_STEPS == 0:
                    truth = _initial_state(data, index + 1)
                    maximum_state_error_m = max(
                        maximum_state_error_m,
                        float(np.linalg.norm(state[:3] - truth[:3])))

    return features[:row], targets[:row], {
        "capture": str(path.relative_to(ROOT)),
        "windows": len(starts),
        "horizon_s": ROLLOUT_HORIZON_S,
        "data_stride_ms": DATA_STRIDE_STEPS,
        "rows": row,
        "skipped_low_speed_or_airborne_rows": skipped_low_speed,
        "maximum_seed_position_error_m": maximum_state_error_m,
        "target_mean_radps_per_step": float(np.mean(targets[:row])),
        "target_rmse_radps_per_step": float(np.sqrt(
            np.mean(targets[:row] ** 2))),
        "target_abs_p95_radps_per_step": float(np.quantile(
            np.abs(targets[:row]), 0.95)),
        "label": "r07 next-sample truth yaw-rate minus source-plant yaw-rate from the seed-predicted current state",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--heldout", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--seed-model", type=Path, default=SEED_MODEL)
    parser.add_argument("--tire-fit", type=Path, default=DEFAULT_TIRE_FIT)
    parser.add_argument("--slip-fit", type=Path, default=DEFAULT_SLIP_FIT)
    parser.add_argument("--suspension-fit", type=Path,
                        default=DEFAULT_SUSPENSION_FIT)
    parser.add_argument("--wheel-fit", type=Path, default=DEFAULT_WHEEL_FIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    model_path = args.output.with_suffix(".joblib")
    if args.output.exists() or model_path.exists():
        raise FileExistsError(f"refusing to overwrite {args.output} or {model_path}")

    started = time.time()
    for path in (args.train, args.heldout, args.seed_model, args.tire_fit,
                 args.slip_fit, args.suspension_fit, args.wheel_fit):
        if not path.is_file():
            raise FileNotFoundError(path)
    seed_bundle = joblib.load(args.seed_model)
    seed_model = seed_bundle["model"]
    constants, plant_description = _prepare_constants(
        _metadata(args.train), args.tire_fit, args.slip_fit,
        args.suspension_fit, args.wheel_fit)

    x_truth, y_truth, truth_diagnostics = _training_rows(args.train, constants)
    x_dagger, y_dagger, dagger_diagnostics = _collect_rollout_rows(
        args.train, constants, seed_model)
    truth_keep = np.arange(0, len(y_truth), DATA_STRIDE_STEPS)
    x_fit = np.concatenate((x_truth[truth_keep], x_dagger), axis=0)
    y_fit = np.concatenate((y_truth[truth_keep], y_dagger), axis=0)
    model = HistGradientBoostingRegressor(
        loss="squared_error", learning_rate=0.05, max_iter=160,
        max_leaf_nodes=31, min_samples_leaf=200,
        l2_regularization=10.0, early_stopping=False,
        random_state=20261013,
    )
    with threadpool_limits(limits=4):
        model.fit(x_fit, y_fit)
    train_prediction = model.predict(x_fit)
    heldout = _evaluate(
        args.heldout, constants, model,
        horizon_s=ROLLOUT_HORIZON_S, stride_s=ROLLOUT_STRIDE_S,
        minimum_model_speed_mps=MODEL_MIN_SPEED_MPS)

    joblib.dump({
        "model": model,
        "feature_names": list(FEATURE_NAMES),
        "target": "per-1ms-step yaw-rate increment residual, rad/s",
        "sample_period_s": DT_S,
        "training_capture": str(args.train.relative_to(ROOT)),
        "training_capture_sha256": _sha256(args.train),
        "seed_model": str(args.seed_model.relative_to(ROOT)),
        "seed_model_sha256": _sha256(args.seed_model),
        "rollout_state_dagger": True,
    }, model_path, compress=3)
    report = {
        "schema_version": 1,
        "status": "research-only rollout-state 1 ms yaw residual; not runtime odometry",
        "training_capture": str(args.train.relative_to(ROOT)),
        "training_capture_sha256": _sha256(args.train),
        "heldout_capture": str(args.heldout.relative_to(ROOT)),
        "heldout_capture_sha256": _sha256(args.heldout),
        "seed_model": str(args.seed_model.relative_to(ROOT)),
        "seed_model_sha256": _sha256(args.seed_model),
        "model_artifact": str(model_path.relative_to(ROOT)),
        "model_artifact_sha256": _sha256(model_path),
        "sample_period_s": DT_S,
        "model_speed_gate_mps": MODEL_MIN_SPEED_MPS,
        "plant_description": plant_description,
        "truth_state_training_rows": truth_diagnostics,
        "rollout_state_training_rows": dagger_diagnostics,
        "combined_fit_rows": int(len(y_fit)),
        "combined_in_sample_rmse_radps_per_step": float(np.sqrt(np.mean(
            (train_prediction - y_fit) ** 2))),
        "heldout_recursive_command_only": heldout,
        "truth_or_future_sensors_used_during_rollout": False,
        "r11_used_for_fit_or_checkpoint_selection": False,
        "test_or_final_test_archives_used": False,
        "elapsed_seconds": time.time() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    summary = {
        "report": str(args.output),
        "model": str(model_path),
        "fit_rows": len(y_fit),
        "rollout_rows": len(y_dagger),
        "heldout_5s_errors": heldout["errors"],
        "elapsed_seconds": report["elapsed_seconds"],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
