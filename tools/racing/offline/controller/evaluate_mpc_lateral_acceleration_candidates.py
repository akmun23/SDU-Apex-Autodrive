#!/usr/bin/env python3
"""Train-only screening of causal lateral-acceleration models for MPC v."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tools.vehicle_dynamics_learning.train_direct_sequence_teacher import COM_X_M
from tools.vehicle_dynamics_learning.sealed_archive_guard import (
    assert_no_sealed_runs_in_packed_archive,
)


ROOT = Path(__file__).resolve().parents[4]
DATASET = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "practice_transfer_validation_20261001_r03/openplane_dynamics.npz")
OUTPUT_DIR = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "odom_wheel_signal_correction_20261011/"
    "mpc_prediction_analysis_20261012")
OUTPUT = OUTPUT_DIR / "lateral_transition_candidate_screen_v5.json"
DT_S = 0.025


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def _features(u: np.ndarray, v: np.ndarray, r: np.ndarray,
              delta: np.ndarray, delta_rate: np.ndarray) -> np.ndarray:
    abs_r = np.abs(r)
    abs_delta = np.abs(delta)
    return np.column_stack((
        u, v, r, delta, delta_rate, u * r, u * delta,
        u * delta * abs_delta, v * abs_r, r * abs_r,
        delta * abs(delta_rate),
    )).astype(np.float64)


def _main() -> dict[str, Any]:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    assert_no_sealed_runs_in_packed_archive(DATASET)
    with np.load(DATASET, allow_pickle=False) as archive:
        frames = archive["frames"].astype(np.float64)
        rigid = archive["simulator_rigid_state"].astype(np.float64)
        acceleration = archive["simulator_linear_acceleration"].astype(np.float64)
        dt = archive["dt_s"].astype(np.float64)
        bounds = archive["sequence_bounds"].astype(np.int64)
        sequence_run = archive["sequence_run_index"].astype(np.int32)
        splits = archive["run_splits"].astype(str)
        run_ids = archive["run_ids"].astype(str)
        frame_run = archive["frame_run_index"].astype(np.int32)
        lap_count = archive["lap_count"].astype(np.int32)

    # Simulator rigid-state lateral velocity is at COM. MPC's path-frame v is
    # at the rear axle, hence v_rear = v_COM - COM_X * yaw_rate.
    u = rigid[:, 7]
    r = rigid[:, 12]
    v_rear = rigid[:, 8] - COM_X_M * r
    delta = frames[:, 3]

    feature_rows: dict[str, list[np.ndarray]] = {}
    ay_rows: dict[str, list[np.ndarray]] = {}
    vdot_rows: dict[str, list[np.ndarray]] = {}
    transport_rows: dict[str, list[np.ndarray]] = {}
    row_counts: dict[str, int] = {}
    for (start_raw, end_raw), run_raw in zip(bounds, sequence_run):
        start, end, run = int(start_raw), int(end_raw), int(run_raw)
        if end - start < 2:
            continue
        if not np.allclose(dt[start + 1:end], DT_S, rtol=0.0, atol=1e-7):
            continue
        local = np.arange(start + 1, end, dtype=np.int64)
        current = local - 1
        if splits[run] == "train":
            if run_ids[run] == "practice_filter_none_12lap_20260925":
                lap = lap_count[current]
                selections = (
                    ("train", (lap >= 1) & (lap <= 8)),
                    ("practice_training_run_laps_9_12",
                     (lap >= 9) & (lap <= 12)),
                )
            else:
                selections = (("train", np.ones(len(current), dtype=bool)),)
        elif splits[run] == "unseen_practice":
            selections = ((str(run_ids[run]),
                           np.ones(len(current), dtype=bool)),)
        elif splits[run] == "validation":
            selections = (("openplane_validation",
                           np.ones(len(current), dtype=bool)),)
        else:
            # Sealed rows are never admitted to fit or score arrays. The
            # archive-level guard above additionally ensures their compressed
            # numeric members were not opened in the first place.
            continue

        for split_name, keep in selections:
            sample_current = current[keep]
            sample_next = local[keep]
            if not len(sample_current):
                continue
            delta_rate = (delta[sample_current] -
                          delta[np.maximum(sample_current - 1, start)]) / DT_S
            x = _features(
                u[sample_current], v_rear[sample_current], r[sample_current],
                delta[sample_current], delta_rate)
            ay_mid = 0.5 * (acceleration[sample_current, 1]
                            + acceleration[sample_next, 1])
            u_mid = 0.5 * (u[sample_current] + u[sample_next])
            r_mid = 0.5 * (r[sample_current] + r[sample_next])
            r_dot = (r[sample_next] - r[sample_current]) / DT_S
            actual_vdot = (v_rear[sample_next] - v_rear[sample_current]) / DT_S
            required_ay = actual_vdot + u_mid * r_mid + COM_X_M * r_dot

            feature_rows.setdefault(split_name, []).append(x)
            ay_rows.setdefault(split_name, []).append(ay_mid)
            vdot_rows.setdefault(split_name, []).append(actual_vdot)
            transport_rows.setdefault(split_name, []).append(np.column_stack((
                u_mid * r_mid, r_dot, required_ay, ay_mid)))
            row_counts[split_name] = row_counts.get(split_name, 0) + len(sample_current)
            if (split_name == "train"
                    and run_ids[run] == "practice_filter_none_12lap_20260925"):
                practice_name = "practice_training_run_laps_1_8"
                feature_rows.setdefault(practice_name, []).append(x)
                ay_rows.setdefault(practice_name, []).append(ay_mid)
                vdot_rows.setdefault(practice_name, []).append(actual_vdot)
                transport_rows.setdefault(practice_name, []).append(
                    np.column_stack((u_mid * r_mid, r_dot, required_ay, ay_mid)))
                row_counts[practice_name] = (
                    row_counts.get(practice_name, 0) + len(sample_current))

    if "train" not in feature_rows:
        raise RuntimeError("no eligible training transitions")
    x_train = np.concatenate(feature_rows["train"])
    ay_train = np.concatenate(ay_rows["train"])
    train_transport = np.concatenate(transport_rows["train"])
    train_speed = np.hypot(x_train[:, 0], x_train[:, 1])
    train_abs_steering = np.abs(x_train[:, 3])

    # The analytic candidate assumes a_y ~= k*u*r. Fit k using training only.
    ur_train = train_transport[:, 0]
    k = float(np.dot(ur_train, ay_train) /
              max(np.dot(ur_train, ur_train), 1e-12))
    learned = make_pipeline(StandardScaler(), Ridge(alpha=0.1))
    learned.fit(x_train, ay_train)
    vdot_train = np.concatenate(vdot_rows["train"])
    direct_vdot = make_pipeline(StandardScaler(), Ridge(alpha=0.1))
    direct_vdot.fit(x_train, vdot_train)
    x_practice_train = np.concatenate(
        feature_rows["practice_training_run_laps_1_8"])
    ay_practice_train = np.concatenate(
        ay_rows["practice_training_run_laps_1_8"])
    practice_transport = np.concatenate(
        transport_rows["practice_training_run_laps_1_8"])
    practice_ur = practice_transport[:, 0]
    k_practice = float(np.dot(practice_ur, ay_practice_train) /
                       max(np.dot(practice_ur, practice_ur), 1e-12))
    practice_learned = make_pipeline(StandardScaler(), Ridge(alpha=0.1))
    practice_learned.fit(x_practice_train, ay_practice_train)
    vdot_practice_train = np.concatenate(
        vdot_rows["practice_training_run_laps_1_8"])
    practice_direct_vdot = make_pipeline(
        StandardScaler(), Ridge(alpha=0.1))
    practice_direct_vdot.fit(x_practice_train, vdot_practice_train)
    no_rate_columns = np.asarray([0, 1, 2, 3, 5, 6, 7, 8, 9])
    practice_direct_vdot_no_rate = make_pipeline(
        StandardScaler(), Ridge(alpha=0.1))
    practice_direct_vdot_no_rate.fit(
        x_practice_train[:, no_rate_columns], vdot_practice_train)

    reports: dict[str, Any] = {}
    for split_name, x_parts in feature_rows.items():
        x = np.concatenate(x_parts)
        ay = np.concatenate(ay_rows[split_name])
        vdot = np.concatenate(vdot_rows[split_name])
        transport = np.concatenate(transport_rows[split_name])
        ur, r_dot = transport[:, 0], transport[:, 1]
        candidates = {
            "production_hold_v": None,
            "centripetal_a_y_equals_u_times_r": ur,
            "train_fitted_gain_a_y_equals_k_ur": k * ur,
            "train_ridge_causal_state_command": learned.predict(x),
            "practice_train_fitted_gain_a_y_equals_k_ur": k_practice * ur,
            "practice_train_ridge_causal_state_command":
                practice_learned.predict(x),
            "simulator_truth_a_y_upper_bound": ay,
        }
        candidate_errors: dict[str, Any] = {}
        # v update uses predicted/observed yaw increment plus candidate a_y.
        # In this one-step isolation, actual u/r increments are known only for
        # scoring; the separate free-rollout check must use model predictions.
        for name, ay_hat in candidates.items():
            vdot_hat = (np.zeros_like(vdot) if ay_hat is None else
                        ay_hat - (ur + COM_X_M * r_dot))
            candidate_errors[name] = {
                "lateral_acceleration_rmse_mps2": (
                    None if ay_hat is None else _rmse(ay_hat - ay)),
                "rear_v_increment_rmse_mps_per_step": _rmse(
                    (vdot_hat - vdot) * DT_S),
                "rear_v_increment_bias_mps_per_step": float(
                    np.mean((vdot_hat - vdot) * DT_S)),
            }
        direct_predictions = {
            "train_ridge_direct_vdot": direct_vdot.predict(x),
            "practice_train_ridge_direct_vdot":
                practice_direct_vdot.predict(x),
            "practice_train_ridge_direct_vdot_no_steering_rate":
                practice_direct_vdot_no_rate.predict(x[:, no_rate_columns]),
        }
        for name, vdot_hat in direct_predictions.items():
            increment_error = (vdot_hat - vdot) * DT_S
            candidate_errors[name] = {
                "lateral_acceleration_rmse_mps2": None,
                "rear_v_increment_rmse_mps_per_step": _rmse(increment_error),
                "rear_v_increment_bias_mps_per_step": float(
                    np.mean(increment_error)),
            }
        reports[split_name] = {
            "transitions": int(len(x)),
            "features_supported_by_runtime_state_or_command": [
                "u", "v_rear", "yaw_rate", "actual_steering",
                "actual_steering_rate", "u*yaw_rate", "u*steering",
                "u*steering*abs(steering)", "v*abs(yaw_rate)",
                "yaw_rate*abs(yaw_rate)",
                "steering*abs(steering_rate)"],
            "candidate_scores": candidate_errors,
        }
        speed = np.hypot(x[:, 0], x[:, 1])
        abs_steering = np.abs(x[:, 3])
        speed_edges = (0.0, 3.0, 5.0, 7.0, 9.0, 12.0, np.inf)
        steer_edges = (0.0, 0.10, 0.20, 0.30, 0.40, 0.53)
        regime_rows: dict[str, Any] = {}
        for speed_index, (speed_low, speed_high) in enumerate(
                zip(speed_edges[:-1], speed_edges[1:])):
            for steer_index, (steer_low, steer_high) in enumerate(
                    zip(steer_edges[:-1], steer_edges[1:])):
                label = (f"speed_{speed_low:g}_{speed_high:g}/"
                         f"abs_steer_{steer_low:g}_{steer_high:g}")
                mask = ((speed >= speed_low) & (speed < speed_high)
                        & (abs_steering >= steer_low)
                        & (abs_steering < steer_high))
                train_mask = ((train_speed >= speed_low)
                              & (train_speed < speed_high)
                              & (train_abs_steering >= steer_low)
                              & (train_abs_steering < steer_high))
                if np.count_nonzero(mask) < 30:
                    continue
                ur_eval = ur[mask]
                rdot_eval = r_dot[mask]
                actual_vdot_eval = vdot[mask]
                local_k = None
                if np.count_nonzero(train_mask) >= 1000:
                    ur_fit = train_transport[train_mask, 0]
                    ay_fit = ay_train[train_mask]
                    local_k = float(np.dot(ur_fit, ay_fit) /
                                     max(np.dot(ur_fit, ur_fit), 1e-12))
                local_predictions = {
                    "production_hold_v": np.zeros_like(actual_vdot_eval),
                    "centripetal_a_y_equals_u_times_r": ur_eval,
                    "global_train_k_ur": k * ur_eval,
                }
                if local_k is not None:
                    local_predictions["speed_steering_cell_train_k_ur"] = (
                        local_k * ur_eval)
                cell = {}
                for candidate_name, ay_hat in local_predictions.items():
                    vdot_hat = (ay_hat - ur_eval - COM_X_M * rdot_eval
                                if candidate_name != "production_hold_v"
                                else ay_hat)
                    increment_error = ((vdot_hat - actual_vdot_eval) * DT_S)
                    cell[candidate_name] = {
                        "rear_v_increment_rmse_mps_per_step": _rmse(
                            increment_error),
                        "rear_v_increment_bias_mps_per_step": float(
                            np.mean(increment_error)),
                    }
                for model_name, model in (
                        ("global_train_ridge_direct_vdot", direct_vdot),
                        ("practice_train_ridge_direct_vdot",
                         practice_direct_vdot),
                        ("practice_train_ridge_direct_vdot_no_steering_rate",
                         None)):
                    vdot_hat = (
                        practice_direct_vdot_no_rate.predict(
                            x[mask][:, no_rate_columns])
                        if model_name.endswith("no_steering_rate")
                        else model.predict(x[mask]))
                    increment_error = ((vdot_hat - actual_vdot_eval) * DT_S)
                    cell[model_name] = {
                        "rear_v_increment_rmse_mps_per_step": _rmse(
                            increment_error),
                        "rear_v_increment_bias_mps_per_step": float(
                            np.mean(increment_error)),
                    }
                regime_rows[label] = {
                    "transitions": int(np.count_nonzero(mask)),
                    "training_transitions": int(np.count_nonzero(train_mask)),
                    "local_k": local_k,
                    "candidates": cell,
                }
        reports[split_name]["speed_steering_cells"] = regime_rows

    reports["training_fit"] = {
        "practice_run_laps_used": "1-8 only",
        "practice_laps_9_to_12_used_for_fit": False,
        "fitted_centripetal_gain_k": k,
        "practice_only_fitted_centripetal_gain_k": k_practice,
        "ridge_alpha": 0.1,
        "ridge_feature_count": int(x_train.shape[1]),
        "direct_vdot_ridge_alpha": 0.1,
        "practice_direct_vdot_no_rate_features": [
            "u", "v_rear", "yaw_rate", "actual_steering",
            "u*yaw_rate", "u*steering", "u*steering*abs(steering)",
            "v*abs(yaw_rate)", "yaw_rate*abs(yaw_rate)"],
        "practice_direct_vdot_no_rate_standardized_model": {
            "feature_mean": practice_direct_vdot_no_rate[0].mean_.tolist(),
            "feature_scale": practice_direct_vdot_no_rate[0].scale_.tolist(),
            "coef": practice_direct_vdot_no_rate[1].coef_.tolist(),
            "intercept": float(practice_direct_vdot_no_rate[1].intercept_),
        },
        "practice_only_ridge_feature_count": int(x_practice_train.shape[1]),
        "fit_run_ids": sorted({
            str(run_ids[int(run)]) for run in np.unique(sequence_run)
            if splits[int(run)] == "train"}),
    }
    # Compare the simulator's lateral-acceleration label with the value
    # implied by adjacent body-state truth; this is a signal-semantic check.
    for split_name, parts in transport_rows.items():
        values = np.concatenate(parts)
        reports[split_name]["simulator_a_y_label_vs_body_state_identity"] = {
            "rmse_mps2": _rmse(values[:, 2] - values[:, 3]),
            "correlation": float(np.corrcoef(values[:, 2], values[:, 3])[0, 1]),
            "implied_a_y_rmse_mps2": _rmse(values[:, 2]),
            "logged_a_y_rmse_mps2": _rmse(values[:, 3]),
        }

    result = {
        "schema_version": 1,
        "offline_only": True,
        "simulator_was_run": False,
        "runtime_model_changed": False,
        "ground_truth_used_for_training_labels_and_scoring_only": True,
        "test_and_final_test_rows_used": False,
        "purpose": "screen causal lateral-acceleration models for a rear-axle lateral-velocity transition in the production MPC model",
        "equations": {
            "rear_v": "v_com - COM_X*r",
            "rear_v_dot": "a_y - u*r - COM_X*r_dot",
            "baseline": "v_next = v",
            "physics_candidate": "a_y_hat = k*u_mid*r_mid; fit k on training captures only",
        },
        "dataset_sha256": _sha256(DATASET),
        "dataset": str(DATASET.relative_to(ROOT)),
        "row_counts": row_counts,
        "reports": reports,
        "caveats": [
            "the one-step kinematic comparison uses truth u/r increments to isolate lateral-acceleration prediction; it is not a free-running MPC score",
            "the truth-a_y entry is an unattainable upper bound, not a candidate input",
            "the training-side practice fit uses laps 1-8; laps 9-12 are reported separately",
            "overlapping 25 ms samples are not independent statistical trials",
        ],
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return result


if __name__ == "__main__":
    report = _main()
    print(json.dumps({
        "output": str(OUTPUT),
        "training_fit": report["reports"]["training_fit"],
        "splits": {
            name: {
                "transitions": values["transitions"],
                "candidates": values["candidate_scores"],
                "label_identity": values["simulator_a_y_label_vs_body_state_identity"],
            }
            for name, values in report["reports"].items()
            if name not in ("training_fit", "practice_training_run_laps_1_8")
        },
    }, indent=2, sort_keys=True))
