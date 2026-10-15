#!/usr/bin/env python3
"""Fit and held-out-evaluate a support-gated high-steering Odom-u expert.

Only current-source sensor-only replay states are model inputs. Simulator truth
in the source captures is used offline as the correction target and score.
Simulator rigid-body truth supplies targets offline; causal sensor-only
diagnostics and permitted actuator feedback supply runtime features. An
interrupted throttle-sweep capture is admitted only through its phase-marked
valid sequences; its untouched repeat remains validation.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from tools import evaluate_open_plane_body_dynamics as body
from tools.vehicle_dynamics_learning import evaluate_unity_actuator_features_odom as unity
from tools.vehicle_dynamics_learning import fit_supported_practice_odom_u_residual as base
from tools.vehicle_dynamics_learning.score_practice_odom_replay import (
    _align_diagnostics,
    _read_diagnostics,
)


TRAIN_RUNS = (
    ("L01", "openplane_yaw_error_lowspeed_highsteer_train_r01_retry1_20261008",
     "odom_highsteer_current_source_replay_20261014/L01"),
    ("L02", "openplane_yaw_error_lowspeed_highsteer_train_r02_20261008",
     "odom_highsteer_current_source_replay_20261014/L02"),
    ("R01", "openplane_yaw_error_highsteer_reversal_train_r01_20261008",
     "odom_highsteer_current_source_replay_20261014/R01"),
    ("R02", "openplane_yaw_error_highsteer_reversal_train_r02_20261008",
     "odom_highsteer_current_source_replay_20261014/R02"),
    ("R04", "openplane_yaw_error_highsteer_reversal_train_r04_20261008",
     "odom_highsteer_current_source_replay_20261014/R04"),
    ("R06", "openplane_yaw_error_highsteer_reversal_train_r06_20261008",
     "odom_highsteer_current_source_replay_20261014/R06"),
    ("W03", "openplane_yaw_error_lowspeed_wheelspin_train_r03_20261008",
     "odom_highsteer_current_source_replay_20261014/W03"),
)
VALIDATION_RUNS = (
    ("L03", "openplane_yaw_error_lowspeed_highsteer_validation_r03_20261008",
     "odom_highsteer_current_source_replay_20261014/L03"),
    ("R03", "openplane_yaw_error_highsteer_reversal_validation_r03_20261008",
     "odom_highsteer_current_source_replay_20261014/R03_retry"),
    ("H01", "openplane_high_angle_boundary_capture_20260926_1636",
     "odom_highsteer_current_source_replay_20261014/H01"),
    ("T05", "openplane_throttle_5pct_5deg_20260930_r05_resume",
     "odom_highsteer_throttle_current_source_replay_20261014_r05"),
)
THROTTLE_SWEEP_TRAIN_RUN = (
    "T04", "openplane_throttle_5pct_5deg_20260930_r04",
    "odom_highsteer_throttle_current_source_replay_20261014_r04")
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
CORRECTION_GAINS = tuple(np.linspace(0.0, 1.0, 11))
LOW_THROTTLE_U_MAX_CANDIDATES_MPS = (3.4, 3.45, 3.5, 3.75, 4.0, 4.6)
HIGH_THROTTLE_SPLIT = 0.15
JOINT_BIN_MIN_SAMPLES_FOR_GATE_SELECTION = 40
JOINT_BIN_MAX_RELATIVE_REGRESSION = 1.05
RUNTIME_U_MIN_MPS = 3.2
RUNTIME_U_MAX_MPS = 4.6
TRUTH_SPEED_MIN_MPS = 3.0
TRUTH_SPEED_MAX_MPS = 4.1
STEERING_MIN_RAD = 0.31
STEERING_MAX_RAD = 0.50
THROTTLE_MIN = 0.11
THROTTLE_MAX = 0.18


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_run(label: str, directory: str, replay_directory: str,
              root: Path, replay_root: Path,
              allow_interrupted_valid_phases: bool = False
              ) -> dict[str, Any]:
    raw = root / directory / "run/run_0.db3"
    replay = replay_root / replay_directory / "replayed/replayed_0.db3"
    capture = body.load_capture(raw)
    if ((capture.aborted and not allow_interrupted_valid_phases) or
            (allow_interrupted_valid_phases and
             (not capture.has_phase_markers or not capture.valid_phase_count)) or
            capture.collision_count_start != 0 or
            capture.collision_count_end != 0 or capture.timing_faults or
            not capture.sequences):
        raise ValueError(f"capture failed phase/collision/timing quality: {raw}")

    samples_by_stamp: dict[int, Any] = {}
    groups_by_stamp: dict[int, str] = {}
    for sequence_index, sequence in enumerate(capture.sequences):
        group = capture.sequence_labels[sequence_index]
        for sample in sequence:
            stamp = int(sample.source_stamp_ns)
            samples_by_stamp.setdefault(stamp, sample)
            groups_by_stamp.setdefault(stamp, group)
    all_stamps = sorted(samples_by_stamp)
    diagnostic_rows = _read_diagnostics(
        replay, "/replayed_odom_diagnostics")
    diagnostic_stamps = sorted(diagnostic_rows)

    def has_diagnostic(stamp: int) -> bool:
        index = bisect.bisect_left(diagnostic_stamps, stamp)
        candidates = [i for i in (index - 1, index)
                      if 0 <= i < len(diagnostic_stamps)]
        return bool(candidates) and min(
            abs(diagnostic_stamps[i] - stamp) for i in candidates
        ) <= 2_000_000

    stamps = [stamp for stamp in all_stamps if has_diagnostic(stamp)]
    if len(stamps) < 0.98 * len(all_stamps):
        raise ValueError(
            f"current-source diagnostics cover only {len(stamps)}/"
            f"{len(all_stamps)} source samples in {label}")
    samples = [samples_by_stamp[stamp] for stamp in stamps]
    diagnostics = _align_diagnostics(
        diagnostic_rows, stamps, "/replayed_odom_diagnostics")
    if diagnostics.shape != (len(samples), base.DIAGNOSTIC_SIZE):
        raise ValueError(f"expected aligned v7 diagnostics for {label}")

    recorded_odom_state = np.stack([sample.state for sample in samples])
    if any(sample.simulator_rigid_state is None for sample in samples):
        raise ValueError(f"simulator rigid-body truth is missing in {label}")
    rigid_truth = np.stack([sample.simulator_rigid_state for sample in samples])
    # Unity packets report COM body velocity. Odom publishes rear-axle [u,v,r];
    # shift only lateral velocity from COM to rear axle using the rigid-body
    # point-velocity equation v_P = v_COM + omega x r_COM->P.
    truth_state = body.simulator_truth_rear_axle_twist(rigid_truth)
    truth_speed = np.hypot(truth_state[:, 0], truth_state[:, 1])
    steering = np.asarray([sample.actuators[0] for sample in samples])
    throttle = np.asarray([sample.actuators[1] for sample in samples])
    if not all(np.isfinite(value).all() for value in
               (diagnostics, recorded_odom_state, rigid_truth, truth_state,
                steering, throttle)):
        raise ValueError(f"non-finite training/validation values in {label}")
    features = np.column_stack((base._features(diagnostics),
                                unity._features(samples, diagnostics)))
    active = (
        (truth_speed >= TRUTH_SPEED_MIN_MPS) &
        (truth_speed < TRUTH_SPEED_MAX_MPS) &
        (np.abs(steering) > STEERING_MIN_RAD) &
        (np.abs(steering) <= STEERING_MAX_RAD) &
        (throttle >= THROTTLE_MIN) &
        (throttle <= THROTTLE_MAX))
    groups = np.asarray([f"{label}:{groups_by_stamp[stamp]}" for stamp in stamps])
    return {
        "label": label,
        "raw": raw,
        "replay": replay,
        "capture": capture,
        "samples": samples,
        "state": truth_state,
        "recorded_odom_state": recorded_odom_state,
        "steering": steering,
        "throttle": throttle,
        "diagnostics": diagnostics,
        "source_samples_before_diagnostic_match": len(all_stamps),
        "diagnostic_unmatched_source_samples": len(all_stamps) - len(stamps),
        "features": features,
        "groups": groups,
        "active": active,
        "truth_speed": truth_speed,
    }


def _fit(runs: list[dict[str, Any]], alpha: float, u_min_mps: float,
         u_max_mps: float = RUNTIME_U_MAX_MPS,
         low_throttle_u_max_mps: float | None = None
         ) -> dict[str, Any]:
    selected = [run["active"] & _runtime_gate(run, {
        "published_u_min_mps": u_min_mps,
        "published_u_max_mps": u_max_mps,
        "published_u_max_low_throttle_mps": low_throttle_u_max_mps,
        "high_throttle_split": HIGH_THROTTLE_SPLIT,
    }) for run in runs]
    features = np.concatenate([
        run["features"][mask] for run, mask in zip(runs, selected)])
    residual = np.concatenate([
        run["state"][mask, 0] - run["diagnostics"][mask, 7]
        for run, mask in zip(runs, selected)])
    groups = np.concatenate([
        run["groups"][mask] for run, mask in zip(runs, selected)])
    if len(residual) < 1000:
        raise ValueError(f"insufficient high-steer support: {len(residual)}")
    return {
        "regression": base._fit(features, residual, groups, alpha),
        "published_u_min_mps": u_min_mps,
        "published_u_max_mps": u_max_mps,
        "published_u_max_low_throttle_mps": low_throttle_u_max_mps,
        "high_throttle_split": HIGH_THROTTLE_SPLIT,
    }


def _predict(run: dict[str, Any], model: dict[str, Any]
             ) -> tuple[np.ndarray, np.ndarray]:
    mean, scale, coefficient, intercept = model["regression"]
    correction = ((run["features"] - mean) / scale) @ coefficient + intercept
    error = run["diagnostics"][:, 7] - run["state"][:, 0]
    candidate_error = error.copy()
    active = _runtime_gate(run, model)
    candidate_error[active] += correction[active]
    return error, candidate_error


def _runtime_gate(run: dict[str, Any], model: dict[str, Any]) -> np.ndarray:
    """Mirror the deployable gate exactly; simulator truth is never an input."""
    high_u_max = model["published_u_max_mps"]
    low_u_max = model.get("published_u_max_low_throttle_mps")
    if low_u_max is None:
        upper_speed = np.full(len(run["throttle"]), high_u_max)
    else:
        upper_speed = np.where(
            run["throttle"] < model["high_throttle_split"],
            low_u_max, high_u_max)
    return (
        (run["diagnostics"][:, 7] >= model["published_u_min_mps"]) &
        (run["diagnostics"][:, 7] <= upper_speed) &
        (np.abs(run["steering"]) > STEERING_MIN_RAD) &
        (np.abs(run["steering"]) <= STEERING_MAX_RAD) &
        (run["throttle"] >= THROTTLE_MIN) &
        (run["throttle"] <= THROTTLE_MAX))


def _metric(error: np.ndarray) -> dict[str, float | int]:
    return {
        "samples": int(len(error)),
        "rmse_mps": float(np.sqrt(np.mean(error ** 2))) if len(error) else None,
        "bias_mps": float(np.mean(error)) if len(error) else None,
        "p95_abs_mps": float(np.quantile(np.abs(error), 0.95)) if len(error) else None,
    }


def _joint_support_report(run: dict[str, Any], model: dict[str, Any]
                          ) -> dict[str, Any]:
    """Count independent phase support where speed/throttle interact."""
    result = {}
    for speed_name, speed_min, speed_max in (
            ("3p0_to_3p5", 3.0, 3.5), ("3p5_to_4p1", 3.5, 4.1)):
        for throttle_name, throttle_min, throttle_max in (
                ("0p11_to_0p15", 0.11, 0.15),
                ("0p15_to_0p18", 0.15, 0.18)):
            mask = (
                run["active"] &
                (run["truth_speed"] >= speed_min) &
                (run["truth_speed"] < speed_max) &
                (run["throttle"] >= throttle_min) &
                (run["throttle"] < throttle_max) &
                (np.abs(run["steering"]) > STEERING_MIN_RAD) &
                (np.abs(run["steering"]) <= STEERING_MAX_RAD))
            fitted_mask = mask & _runtime_gate(run, model)
            key = f"speed_{speed_name}_throttle_{throttle_name}"
            result[key] = {
                "truth_target_samples": int(mask.sum()),
                "truth_target_phases": int(len(np.unique(run["groups"][mask]))),
                "runtime_gate_samples": int(fitted_mask.sum()),
                "runtime_gate_phases": int(
                    len(np.unique(run["groups"][fitted_mask]))),
            }
    return result


def _run_report(run: dict[str, Any], model: dict[str, Any]
                ) -> dict[str, Any]:
    before, after = _predict(run, model)
    runtime_active = _runtime_gate(run, model)
    measured_domain = runtime_active & run["active"]
    runtime_outside_training_domain = runtime_active & ~run["active"]
    truth_speed = run["truth_speed"]
    angle = np.abs(run["steering"])
    bins = {}
    for name, mask in {
        "speed_1p2_to_2": measured_domain & (truth_speed < 2.0),
        "speed_2_to_2p5": measured_domain & (truth_speed >= 2.0) & (truth_speed < 2.5),
        "speed_2p5_to_3": measured_domain & (truth_speed >= 2.5) & (truth_speed < 3.0),
        "speed_3_to_3p5": measured_domain & (truth_speed >= 3.0) & (truth_speed < 3.5),
        "speed_3p5_to_4": measured_domain & (truth_speed >= 3.5),
        "steering_0p30_to_0p35": measured_domain & (angle < 0.35),
        "steering_0p35_to_0p40": measured_domain & (angle >= 0.35) & (angle < 0.40),
        "steering_0p40_to_0p45": measured_domain & (angle >= 0.40) & (angle < 0.45),
        "steering_0p45_to_0p50": measured_domain & (angle >= 0.45),
        "positive_steering": measured_domain & (run["steering"] > 0.0),
        "negative_steering": measured_domain & (run["steering"] < 0.0),
        "throttle_0p11_to_0p13": measured_domain & (run["throttle"] < 0.13),
        "throttle_0p13_to_0p15": measured_domain & (run["throttle"] >= 0.13) & (run["throttle"] < 0.15),
        "throttle_0p15_to_0p18": measured_domain & (run["throttle"] >= 0.15),
    }.items():
        bins[name] = {
            "baseline": _metric(before[mask]),
            "candidate": _metric(after[mask]),
        }
    return {
        "run": run["label"],
        "raw_bag": str(run["raw"].resolve()),
        "raw_bag_sha256": _sha256(run["raw"]),
        "valid_phases": run["capture"].valid_phase_count,
        "invalid_phases": run["capture"].invalid_phase_count,
        "unscored_phases": run["capture"].unscored_phase_count,
        "capture_interrupted_after_valid_phases": run["capture"].aborted,
        "samples": int(len(before)),
        "source_samples_before_diagnostic_match":
            run["source_samples_before_diagnostic_match"],
        "diagnostic_unmatched_source_samples":
            run["diagnostic_unmatched_source_samples"],
        "runtime_gate_active_samples": int(runtime_active.sum()),
        "runtime_gate_active_fraction": float(runtime_active.mean()),
        "runtime_gate_samples_inside_truth_labeled_training_domain": int(
            measured_domain.sum()),
        "runtime_gate_samples_outside_truth_labeled_training_domain": int(
            runtime_outside_training_domain.sum()),
        "candidate_published_u_support_mps": [
            model["published_u_min_mps"], model["published_u_max_mps"]],
        "throttle_feedback_bounds": [
            float(np.min(run["throttle"][runtime_active]))
            if runtime_active.any() else None,
            float(np.max(run["throttle"][runtime_active]))
            if runtime_active.any() else None],
        "all_samples": {
            "baseline": _metric(before), "candidate": _metric(after)},
        "truth_labelled_target_domain": {
            "baseline": _metric(before[run["active"]]),
            "candidate": _metric(after[run["active"]]),
        },
        "supported_region": {
            "baseline": _metric(before[measured_domain]),
            "candidate": _metric(after[measured_domain])},
        "runtime_active_all_truth_speeds": {
            "baseline": _metric(before[runtime_active]),
            "candidate": _metric(after[runtime_active])},
        "runtime_active_outside_truth_labeled_training_domain": {
            "baseline": _metric(before[runtime_outside_training_domain]),
            "candidate": _metric(after[runtime_outside_training_domain])},
        "supported_subregions": bins,
        "joint_speed_throttle_support": _joint_support_report(run, model),
        "receipt_cadence_diagnostic_only": {
            topic: {"rate_hz": values[0], "p95_gap_ms": values[1],
                    "max_gap_ms": values[2]}
            for topic, values in run["capture"].phase_stream_stats.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("live_runs"))
    parser.add_argument(
        "--replay-root", type=Path,
        default=Path("live_runs"))
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("live_runs/odom_highsteer_gt_corrected_analysis_20261014"))
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")

    selection_training = [
        _load_run(label, name, replay, args.root, args.replay_root)
        for label, name, replay in TRAIN_RUNS]
    sweep_training = _load_run(
        *THROTTLE_SWEEP_TRAIN_RUN, args.root, args.replay_root,
        allow_interrupted_valid_phases=True)
    training = [*selection_training, sweep_training]
    # Alpha and correction gain are selected on complete original training
    # captures only. Every held-out score covers the full truth-labelled target
    # region; samples outside the fixed runtime support stay uncorrected.
    cv = []
    for alpha_candidate in ALPHAS:
        folds = []
        for heldout_index, heldout in enumerate(selection_training):
            fit_runs = [run for index, run in enumerate(selection_training)
                        if index != heldout_index]
            fold_model = _fit(fit_runs, alpha_candidate, RUNTIME_U_MIN_MPS)
            before, after = _predict(heldout, fold_model)
            active = heldout["active"]
            baseline_rmse = _metric(before[active])["rmse_mps"]
            candidate_rmse = _metric(after[active])["rmse_mps"]
            folds.append({
                "held_out_run": heldout["label"],
                "target_samples": int(active.sum()),
                "runtime_gate_samples": int(
                    (active & _runtime_gate(heldout, fold_model)).sum()),
                "baseline_rmse_mps": baseline_rmse,
                "candidate_rmse_mps": candidate_rmse,
                "relative_rmse": candidate_rmse / baseline_rmse,
            })
        cv.append({
            "alpha": alpha_candidate,
            "mean_run_relative_rmse": float(np.mean(
                [fold["relative_rmse"] for fold in folds])),
            "all_runs_improved": all(
                fold["candidate_rmse_mps"] < fold["baseline_rmse_mps"]
                for fold in folds),
            "folds": folds,
        })
    selected_cv = min(cv, key=lambda item: item["mean_run_relative_rmse"])
    alpha = float(selected_cv["alpha"])
    gate_gain_cv = []
    for low_throttle_u_max_candidate in LOW_THROTTLE_U_MAX_CANDIDATES_MPS:
        fold_models = []
        for heldout_index in range(len(selection_training)):
            fit_runs = [run for index, run in enumerate(selection_training)
                        if index != heldout_index]
            fold_models.append(_fit(
                fit_runs, alpha, RUNTIME_U_MIN_MPS, RUNTIME_U_MAX_MPS,
                low_throttle_u_max_candidate))
        for gain_candidate in CORRECTION_GAINS:
            folds = []
            joint_regressions = []
            for heldout, fold_model in zip(selection_training, fold_models):
                before = heldout["diagnostics"][:, 7] - heldout["state"][:, 0]
                mean, scale, coefficient, intercept = fold_model["regression"]
                correction = ((heldout["features"] - mean) / scale @ coefficient
                              + intercept)
                runtime_active = _runtime_gate(heldout, fold_model)
                after = before.copy()
                after[runtime_active] += (
                    float(gain_candidate) * correction[runtime_active])
                active = heldout["active"]
                baseline_rmse = _metric(before[active])["rmse_mps"]
                candidate_rmse = _metric(after[active])["rmse_mps"]
                folds.append({
                    "held_out_run": heldout["label"],
                    "target_samples": int(active.sum()),
                    "runtime_gate_samples": int(
                        (active & runtime_active).sum()),
                    "baseline_rmse_mps": baseline_rmse,
                    "candidate_rmse_mps": candidate_rmse,
                    "relative_rmse": candidate_rmse / baseline_rmse,
                })
                joint_bins = {
                    "low_throttle_speed_3p0_to_3p5": (
                        heldout["active"] &
                        (heldout["truth_speed"] >= 3.0) &
                        (heldout["truth_speed"] < 3.5) &
                        (heldout["throttle"] < HIGH_THROTTLE_SPLIT)),
                    "low_throttle_speed_3p5_to_4p1": (
                        heldout["active"] &
                        (heldout["truth_speed"] >= 3.5) &
                        (heldout["truth_speed"] < TRUTH_SPEED_MAX_MPS) &
                        (heldout["throttle"] < HIGH_THROTTLE_SPLIT)),
                    "high_throttle_speed_3p0_to_3p5": (
                        heldout["active"] &
                        (heldout["truth_speed"] >= 3.0) &
                        (heldout["truth_speed"] < 3.5) &
                        (heldout["throttle"] >= HIGH_THROTTLE_SPLIT)),
                    "high_throttle_speed_3p5_to_4p1": (
                        heldout["active"] &
                        (heldout["truth_speed"] >= 3.5) &
                        (heldout["truth_speed"] < TRUTH_SPEED_MAX_MPS) &
                        (heldout["throttle"] >= HIGH_THROTTLE_SPLIT)),
                }
                for bin_name, bin_mask in joint_bins.items():
                    if int(bin_mask.sum()) < JOINT_BIN_MIN_SAMPLES_FOR_GATE_SELECTION:
                        continue
                    bin_before = _metric(before[bin_mask])["rmse_mps"]
                    bin_after = _metric(after[bin_mask])["rmse_mps"]
                    if bin_after > bin_before * JOINT_BIN_MAX_RELATIVE_REGRESSION:
                        joint_regressions.append({
                            "held_out_run": heldout["label"],
                            "bin": bin_name,
                            "samples": int(bin_mask.sum()),
                            "baseline_rmse_mps": bin_before,
                            "candidate_rmse_mps": bin_after,
                        })
            gate_gain_cv.append({
                "low_throttle_published_u_max_mps":
                    float(low_throttle_u_max_candidate),
                "high_throttle_published_u_max_mps": RUNTIME_U_MAX_MPS,
                "throttle_split": HIGH_THROTTLE_SPLIT,
                "gain": float(gain_candidate),
                "mean_run_relative_rmse": float(np.mean(
                    [fold["relative_rmse"] for fold in folds])),
                "all_runs_improved": all(
                    fold["candidate_rmse_mps"] < fold["baseline_rmse_mps"]
                    for fold in folds),
                "joint_regression_count": len(joint_regressions),
                "joint_regressions": joint_regressions,
                "folds": folds,
            })
    eligible_gate_gain_cv = [
        item for item in gate_gain_cv
        if item["joint_regression_count"] == 0]
    if not eligible_gate_gain_cv:
        raise ValueError(
            "no throttle-conditioned speed gate passed training joint-bin "
            "non-regression selection")
    selected_gate_gain_cv = min(
        eligible_gate_gain_cv,
        key=lambda item: item["mean_run_relative_rmse"])
    gain = float(selected_gate_gain_cv["gain"])
    selected_low_throttle_u_max = float(
        selected_gate_gain_cv["low_throttle_published_u_max_mps"])
    model = _fit(training, alpha, RUNTIME_U_MIN_MPS, RUNTIME_U_MAX_MPS,
                 selected_low_throttle_u_max)
    mean, scale, coefficient, intercept = model["regression"]
    model["regression"] = (
        mean, scale, coefficient * gain, intercept * gain)
    validation_runs = [
        _load_run(label, directory, replay_directory, args.root, args.replay_root)
        for label, directory, replay_directory in VALIDATION_RUNS]
    validation_reports = [_run_report(run, model) for run in validation_runs]

    mean, scale, coefficient, intercept = model["regression"]
    raw_coefficients = coefficient / scale
    raw_intercept = float(intercept - np.dot(mean, raw_coefficients))
    model_payload = {
        "model": "high_steer_unity_actuator_u_residual_v1",
        "target": "Unity bridge rigid-state rear-axle forward speed minus current production Odom u",
        "truth_source": "bridge_packet_timing simulator rigid-state fields; never recorded /odom twist",
        "training_runs": [
            {"label": label,
             "raw_bag": str((args.root / name / "run/run_0.db3").resolve()),
             "sha256": _sha256(args.root / name / "run/run_0.db3")}
            for label, name, _ in [*TRAIN_RUNS, THROTTLE_SWEEP_TRAIN_RUN]],
        "feature_names": [*base.FEATURE_NAMES, *unity.SOURCE_FEATURE_NAMES],
        "runtime_affine": {
            "intercept_mps": raw_intercept,
            "coefficients": raw_coefficients.tolist(),
        },
        "support": {
            "published_forward_speed_min_mps": model["published_u_min_mps"],
            "published_forward_speed_max_inclusive_mps": model["published_u_max_mps"],
            "published_forward_speed_max_low_throttle_mps":
                model["published_u_max_low_throttle_mps"],
            "throttle_feedback_split": model["high_throttle_split"],
            "training_truth_speed_min_mps": TRUTH_SPEED_MIN_MPS,
            "training_truth_speed_max_exclusive_mps": TRUTH_SPEED_MAX_MPS,
            "abs_steering_min_exclusive_rad": STEERING_MIN_RAD,
            "abs_steering_max_inclusive_rad": STEERING_MAX_RAD,
            "throttle_feedback_min_inclusive": THROTTLE_MIN,
            "throttle_feedback_max_inclusive": THROTTLE_MAX,
        },
        "training_samples": int(sum(
            (run["active"] & _runtime_gate(run, model)).sum()
            for run in training)),
        "selected_alpha": alpha,
        "selected_correction_gain": gain,
        "selected_runtime_u_min_mps": RUNTIME_U_MIN_MPS,
        "selected_runtime_u_max_high_throttle_mps": RUNTIME_U_MAX_MPS,
        "selected_runtime_u_max_low_throttle_mps": selected_low_throttle_u_max,
    }
    report = {
        "method": "causal high-steer forward-speed residual; source truth used only offline",
        "truth_label": (
            "Unity bridge rigid-state body-frame COM velocity and angular rate "
            "converted to rear-axle [u,v,r]; u is unchanged and "
            "v_rear = v_com - 0.15532*r"),
        "runtime_gate_definition": (
            f"production Odom u in [{RUNTIME_U_MIN_MPS}, {RUNTIME_U_MAX_MPS}] m/s "
            f"for throttle >= {HIGH_THROTTLE_SPLIT}, otherwise through "
            f"{selected_low_throttle_u_max} m/s; absolute steering in "
            f"({STEERING_MIN_RAD}, {STEERING_MAX_RAD}] rad, "
            "and throttle feedback in [0.11, 0.18]; no yaw-rate gate"),
        "runtime_inputs": "current production observer diagnostics, encoders, IMU, and permitted actuator feedback",
        "timebase": "only source-packet-consecutive samples; each accepted interval is 25 ms",
        "training_runs": [
            _run_report(run, model) for run in training],
        "hyperparameter_training_runs": [label for label, _, _ in TRAIN_RUNS],
        "targeted_data_added_after_hyperparameter_selection":
            THROTTLE_SWEEP_TRAIN_RUN[0],
        "leave_one_complete_run_out_regularization": cv,
        "selected_alpha": alpha,
        "leave_one_complete_run_out_gate_and_correction_gain": gate_gain_cv,
        "selected_correction_gain": gain,
        "selected_runtime_u_max_high_throttle_mps": RUNTIME_U_MAX_MPS,
        "selected_runtime_u_max_low_throttle_mps": selected_low_throttle_u_max,
        "selected_runtime_u_min_mps": RUNTIME_U_MIN_MPS,
        "selected_gate_and_correction_gain_improved_every_training_fold":
            selected_gate_gain_cv["all_runs_improved"],
        "held_out_validation": validation_reports,
        "runtime_model": model_payload,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "high_steer_odom_u_expert.json").write_text(
        json.dumps(model_payload, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "fit_and_validation_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "selected_alpha": alpha,
        "training_fold_gains": selected_cv["folds"],
        "selected_runtime_gate": {
            "published_u_min_mps": RUNTIME_U_MIN_MPS,
            "published_u_max_low_throttle_mps": selected_low_throttle_u_max,
            "published_u_max_high_throttle_mps": RUNTIME_U_MAX_MPS,
            "throttle_split": HIGH_THROTTLE_SPLIT,
            "training_only_gate_gain_cv_mean_relative_rmse":
                selected_gate_gain_cv["mean_run_relative_rmse"],
            "all_training_folds_improved":
                selected_gate_gain_cv["all_runs_improved"],
        },
        "held_out_validation": [
            {"run": item["run"],
             "supported_region": item["supported_region"],
             "all_samples": item["all_samples"],
             "subregions": item["supported_subregions"]}
            for item in validation_reports],
        "model_path": str(args.output_dir / "high_steer_odom_u_expert.json"),
        "report_path": str(args.output_dir / "fit_and_validation_report.json"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
