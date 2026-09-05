#!/usr/bin/env python3
"""Fit causal, regime-specific IMU/encoder speed models.

Ground truth is used only to label and score the offline fit.  The generated
C++ evaluator consumes IMU, encoder, and timing features only.  The normal
models deliberately exclude moving frozen-encoder samples; those samples are
fit by the frozen branch instead of being mixed with rolling-wheel motion.
"""

import argparse
from bisect import bisect_right
import csv
import importlib.util
import math
from pathlib import Path
import re
import statistics

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor


FEATURE_NAMES = (
    "imu_speed_mps",
    "mapped_wheel_speed_mps",
    "raw_wheel_speed_mps",
    "signed_wheel_imu_gap_mps",
    "abs_wheel_imu_gap_mps",
    "wheel_observation_confidence",
    "imu_acceleration_mps2",
    "imu_acceleration_filtered_mps2",
    "abs_imu_lateral_acceleration_mps2",
    "abs_imu_yaw_rate_radps",
    "encoder_dt_s",
    "encoder_pair_skew_s",
    "encoder_frozen",
    "median_raw_wheel_speed_mps",
    "median_mapped_wheel_speed_mps",
    "median_imu_speed_mps",
    "median_signed_wheel_imu_gap_mps",
    "raw_wheel_speed_delta_mps",
    "mapped_wheel_speed_delta_mps",
    "imu_speed_delta_mps",
    "window_raw_wheel_speed_mps",
    "window_mapped_wheel_speed_mps",
    "imu_acceleration_bias_mps2",
    "imu_observer_acceleration_mps2",
)

# Frozen encoder motion has two distinct causal causes in the simulator: the
# driven wheel can be frozen while the body is still launching, or it can be
# frozen while the body is braking/coasting.  Keep those branches separate so
# a braking tail cannot bias the launch estimate.  The first four names retain
# the diagnostic regime numbering; the fifth is an internal model branch.
REGIMES = (
    "accelerating", "steady", "decelerating", "frozen", "frozen_accelerating",
)
WHEEL_FROZEN_SPEED_MPS = 0.15
MOVING_SPEED_MPS = 0.75
REGIME_ACCELERATION_MPS2 = 0.50
STATIONARY_SPEED_MPS = 0.30
DEFAULT_SENSOR_ODOMETRY_CONFIG = (
    Path(__file__).resolve().parents[3] /
    "f1tenth_localization" / "config" / "sensor_odometry.yaml")
HISTORY_WINDOW = 9
WHEEL_RADIUS_M = 0.0590
WHEEL_MAP_WHEEL_MPS = np.asarray(
    (0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0,
     16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0),
    dtype=np.float64)
WHEEL_MAP_BODY_MPS = np.asarray(
    (0.0, 1.981831, 3.954033, 5.858258, 7.737783, 9.633347,
     11.447797, 13.230562, 15.094604, 16.841838, 18.522608,
     20.287895, 21.928664, 22.882700, 22.882700, 22.882700),
    dtype=np.float64)

MODEL_KWARGS = {
    # ExtraTrees keeps the model causal while averaging away the
    # burst/quantisation artefacts in the native encoder stream.  The median
    # tree aggregate below is robust to a small number of bad training
    # packets, so the deeper forest can represent the narrow low-speed
    # frozen-wheel relationship without letting one leaf dominate output.
    "n_estimators": 100,
    "max_depth": 16,
    "min_samples_leaf": 1,
    "max_features": 0.9,
    "random_state": 4,
    "n_jobs": -1,
}


def load_imu_filter_alpha(config_path):
    """Read the runtime longitudinal IMU filter alpha from its YAML file.

    The fitter deliberately reads the same source-of-truth parameter as the
    node instead of carrying a second training-only constant.  Keep this
    parser dependency-free because this script already runs outside the ROS
    environment during offline calibration.
    """
    config_path = Path(config_path)
    text = config_path.read_text(encoding="utf-8")
    matches = re.findall(
        r"^\s*imu_acceleration_filter_alpha:\s*"
        r"([-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+))\s*(?:#.*)?$",
        text,
        flags=re.MULTILINE,
    )
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one imu_acceleration_filter_alpha in "
            f"{config_path}, found {len(matches)}")
    alpha = float(matches[0])
    if not 0.0 < alpha <= 1.0:
        raise ValueError(
            f"imu_acceleration_filter_alpha must be in (0, 1], got {alpha}")
    return alpha


def load_analysis_module():
    path = Path(__file__).with_name("analyze_calibration.py")
    spec = importlib.util.spec_from_file_location("analyze_calibration", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load analysis module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def finite(analysis, row, field, default=None):
    value = analysis.finite(row.get(field))
    return default if value is None else value


def sensor_regime(imu_speed, raw_wheel, acceleration):
    if (imu_speed < STATIONARY_SPEED_MPS and
            abs(acceleration) < REGIME_ACCELERATION_MPS2):
        return "stationary"
    if raw_wheel <= WHEEL_FROZEN_SPEED_MPS and imu_speed > MOVING_SPEED_MPS:
        return "frozen"
    if acceleration > REGIME_ACCELERATION_MPS2:
        return "accelerating"
    if acceleration < -REGIME_ACCELERATION_MPS2:
        return "decelerating"
    return "steady"


def ground_truth_regime(speed, raw_wheel, acceleration):
    if speed < STATIONARY_SPEED_MPS and abs(acceleration) < REGIME_ACCELERATION_MPS2:
        return "stationary"
    if raw_wheel <= WHEEL_FROZEN_SPEED_MPS and speed > MOVING_SPEED_MPS:
        return "frozen"
    if acceleration > REGIME_ACCELERATION_MPS2:
        return "accelerating"
    if acceleration < -REGIME_ACCELERATION_MPS2:
        return "decelerating"
    return "steady"


def logged_runtime_regime(analysis, row, imu, raw_wheel, acceleration):
    """Return the exact runtime regime when the diagnostic field is present.

    The estimator uses hysteresis and dwell time before switching regimes.
    Reconstructing a regime from the raw acceleration in this offline fitter
    can therefore train a model for a different branch than the C++ node
    actually selects.  New recordings expose the runtime branch explicitly;
    retain the raw-feature fallback for older calibration files.
    """
    logged = finite(analysis, row, "odom_sensor_motion_regime")
    if logged is not None:
        index = int(round(logged))
        if 0 <= index < len(REGIMES):
            return REGIMES[index]
    return sensor_regime(imu, raw_wheel, acceleration)


def median(history, current):
    values = list(history[-HISTORY_WINDOW:])
    values.append(current)
    return statistics.median(values)


def body_speed_from_wheel_speed(wheel_speed):
    return float(np.interp(
        max(0.0, wheel_speed), WHEEL_MAP_WHEEL_MPS, WHEEL_MAP_BODY_MPS))


_SOURCE_SENSOR_FIELDS = {
    "imu": (
        "ax_mps2", "ay_mps2", "az_mps2", "imu_accel_norm_mps2",
        "yaw_rate_radps", "imu_yaw_rate_radps", "imu_yaw_rad",
        "imu_stamp_s",
    ),
    "left_encoder": (
        "left_encoder_rad", "left_encoder_speed_radps", "left_encoder_dt_s",
        "left_encoder_stamp_s",
    ),
    "right_encoder": (
        "right_encoder_rad", "right_encoder_speed_radps", "right_encoder_dt_s",
        "right_encoder_stamp_s",
    ),
}


def _source_event_series(raw_rows, event_name, analysis):
    """Return unique source callbacks ordered by their message timestamp."""
    unique = {}
    for row in raw_rows:
        if row.get("source_event_name") != event_name:
            continue
        stamp = finite(analysis, row, "source_event_stamp_s")
        if stamp is None:
            continue
        event = row.get("source_event_count", "")
        unique[event or f"stamp:{stamp:.17g}"] = row
    rows = sorted(
        unique.values(),
        key=lambda row: finite(analysis, row, "source_event_stamp_s"),
    )
    return rows, [finite(analysis, row, "source_event_stamp_s") for row in rows]


def merge_coherent_sensor_snapshot(raw_rows, diagnostic_rows, analysis):
    """Overlay the sensor callbacks that produced each diagnostic vector.

    ``odom_diagnostics`` has no ROS header.  The recorder therefore stores
    its vector in a callback snapshot, and that snapshot can be delivered
    before the recorder has processed the IMU/encoder callbacks carrying the
    same source timestamp.  The vector fields themselves are coherent, but
    raw ``ax`` and encoder metadata in the surrounding snapshot may be one
    packet old.  Rebuild those fields from the latest exact source callback at
    or before the vector timestamp.  This is still causal and uses no truth.
    """
    series = {
        name: _source_event_series(raw_rows, name, analysis)
        for name in _SOURCE_SENSOR_FIELDS
    }
    merged = []
    for row in diagnostic_rows:
        stamp = finite(analysis, row, "source_event_stamp_s")
        if stamp is None:
            merged.append(row)
            continue
        aligned = dict(row)
        for event_name, fields in _SOURCE_SENSOR_FIELDS.items():
            events, stamps = series[event_name]
            index = bisect_right(stamps, stamp) - 1
            if index < 0:
                continue
            source = events[index]
            for field in fields:
                value = source.get(field)
                if value not in (None, "", "nan", "NaN"):
                    aligned[field] = value
        merged.append(aligned)
    return merged


def build_examples(path, analysis, run_id, imu_filter_alpha):
    raw_rows = analysis.read_rows(path)
    odom_events, odom_stamps = _source_event_series(
        raw_rows, "odom", analysis)
    # A timer/GT callback snapshot can lag the odom source event by one
    # bridge packet. When exact source events are available, fit the causal
    # sensor features against truth interpolated at that odom timestamp. This
    # keeps the offline target aligned with the runtime pair being modelled.
    # Model features are emitted in the diagnostics vector before /odom is
    # published. The recorder's later /odom callback may run after another
    # IMU/encoder callback and therefore mix an odom speed with an older
    # diagnostic snapshot. Prefer the exact diagnostics event for fitting;
    # retain the odom-event path for legacy recordings without diagnostics.
    rows = analysis.align_odom_source_events(raw_rows, "odom_diagnostics")
    if rows is None:
        rows = analysis.align_odom_source_events(raw_rows)
    else:
        rows = merge_coherent_sensor_snapshot(raw_rows, rows, analysis)
    if rows is None:
        rows, _ = analysis.deduplicate_source_events(raw_rows)
        rows = [row for row in rows if analysis.valid_ground_truth_row(row)]
    rows = analysis.timestamped_rows(rows)

    examples = []
    raw_history = []
    mapped_history = []
    imu_history = []
    gap_history = []
    encoder_history = []
    filtered_acceleration = 0.0
    previous_imu_stamp = None
    previous_active_deceleration = False

    for row in rows:
        phase = row.get("phase", "")
        # The grid harness labels the settled post-reset interval as
        # ``grid_settle_*`` rather than ``grid_reset_*``.  It is nevertheless
        # a hard causal boundary: SensorOdometryNode clears all of its
        # histories when the reset command is accepted.  Do the same here so
        # a previous grid point cannot leak a wheel/IMU window into the next
        # model example.
        if (analysis.phase_is_diagnostic_reset(phase) or
                phase.startswith("grid_settle_")):
            raw_history.clear()
            mapped_history.clear()
            imu_history.clear()
            gap_history.clear()
            encoder_history.clear()
            filtered_acceleration = 0.0
            previous_imu_stamp = None
            previous_active_deceleration = False
            continue

        truth = finite(analysis, row, "gt_speed_mps")
        # The runtime model is evaluated at the encoder-pair timestamp.  The
        # plain observer speed is still at the latest IMU timestamp when an
        # encoder callback arrives; SensorOdometryNode extrapolates that
        # state to the pair timestamp and passes the extrapolated value to
        # sensor_fusion_features().  Prefer the explicitly logged pair value
        # so offline training has the same causal input as the generated C++
        # model. Keep the old field as a compatibility fallback for legacy
        # recordings without pair-time diagnostics.
        imu = finite(analysis, row, "odom_imu_pair_speed_mps")
        if imu is None:
            imu = finite(analysis, row, "odom_imu_speed_mps")
        raw = finite(analysis, row, "odom_raw_wheel_speed_mps")
        confidence = finite(analysis, row, "odom_wheel_observation_confidence")
        acceleration = finite(analysis, row, "ax_mps2", 0.0)
        lateral = finite(analysis, row, "ay_mps2", 0.0)
        yaw_rate = finite(analysis, row, "imu_yaw_rate_radps", 0.0)
        gt_acceleration = finite(analysis, row, "gt_longitudinal_accel_mps2", 0.0)
        imu_acceleration_bias = finite(
            analysis, row, "odom_imu_acceleration_bias_mps2", 0.0)
        imu_observer_acceleration = finite(
            analysis, row, "odom_imu_observer_acceleration_mps2", acceleration)
        left_angle = finite(analysis, row, "left_encoder_rad")
        right_angle = finite(analysis, row, "right_encoder_rad")
        left_stamp = finite(analysis, row, "left_encoder_stamp_s")
        right_stamp = finite(analysis, row, "right_encoder_stamp_s")
        if any(value is None for value in (
                truth, imu, raw, confidence, left_angle, right_angle,
                left_stamp, right_stamp)):
            continue

        imu = max(0.0, imu)
        raw = abs(raw)
        # Recompute the mapped wheel observation from the raw encoder speed.
        # Recordings made before the corrected 40 Hz map was installed contain
        # a different diagnostic mapped value; using that logged value would
        # make a cross-recording fit learn the map revision instead of the
        # sensor-fusion relationship.
        mapped = body_speed_from_wheel_speed(raw)
        confidence = max(0.0, confidence)
        imu_stamp = finite(analysis, row, "imu_stamp_s")
        if imu_stamp is None:
            imu_stamp = finite(analysis, row, "stamp_s", 0.0)
        if previous_imu_stamp is None or imu_stamp <= previous_imu_stamp:
            filtered_acceleration = acceleration
        else:
            filtered_acceleration = (
                imu_filter_alpha * acceleration +
                (1.0 - imu_filter_alpha) * filtered_acceleration)
        previous_imu_stamp = imu_stamp

        # Keep the fitter's short causal history consistent with the runtime
        # node. A braking boundary changes the observability regime: the
        # driven encoder can freeze, so pre-brake acceleration/steady samples
        # must not be mixed into the post-brake recovery feature vector.
        active_deceleration = (
            filtered_acceleration < -REGIME_ACCELERATION_MPS2)
        if active_deceleration != previous_active_deceleration:
            raw_history.clear()
            mapped_history.clear()
            imu_history.clear()
            gap_history.clear()
        previous_active_deceleration = active_deceleration

        signed_gap = mapped - imu
        gap = abs(signed_gap)
        encoder_dt = max(0.0, finite(analysis, row, "left_encoder_dt_s", 0.0))
        pair_skew = 0.0 if left_stamp is None or right_stamp is None else abs(
            left_stamp - right_stamp)
        encoder_stamp = max(
            value for value in (left_stamp, right_stamp) if value is not None)
        encoder_position = WHEEL_RADIUS_M * 0.5 * (left_angle + right_angle)
        if encoder_history and encoder_stamp > encoder_history[0][0]:
            window_raw = abs(encoder_position - encoder_history[0][1]) / (
                encoder_stamp - encoder_history[0][0])
        else:
            window_raw = raw
        logged_window_raw = finite(
            analysis, row, "odom_sensor_fusion_window_raw_speed_mps")
        logged_window_mapped = finite(
            analysis, row, "odom_sensor_fusion_window_mapped_speed_mps")
        if logged_window_raw is not None:
            window_raw = abs(logged_window_raw)
        window_mapped = (
            abs(logged_window_mapped) if logged_window_mapped is not None else
            body_speed_from_wheel_speed(window_raw))
        previous_raw = raw_history[-1] if raw_history else raw
        previous_mapped = mapped_history[-1] if mapped_history else mapped
        previous_imu = imu_history[-1] if imu_history else imu
        features = np.asarray([
            imu,
            mapped,
            raw,
            signed_gap,
            gap,
            confidence,
            acceleration,
            filtered_acceleration,
            abs(lateral),
            abs(yaw_rate),
            encoder_dt,
            max(0.0, pair_skew),
            1.0 if raw <= WHEEL_FROZEN_SPEED_MPS else 0.0,
            median(raw_history, raw),
            median(mapped_history, mapped),
            median(imu_history, imu),
            median(gap_history, signed_gap),
            raw - previous_raw,
            mapped - previous_mapped,
            imu - previous_imu,
            window_raw,
            window_mapped,
            imu_acceleration_bias,
            imu_observer_acceleration,
        ], dtype=np.float64)
        runtime_regime = logged_runtime_regime(
            analysis, row, imu, raw, acceleration)
        if runtime_regime == "frozen" and filtered_acceleration >= -0.50:
            runtime_regime = "frozen_accelerating"
        # Match the diagnostic callback to the actual odom source callback by
        # message timestamp. The node publishes diagnostics immediately
        # before /odom, but the diagnostic vector stores its internal speed
        # member while the odom message carries the function argument. The
        # source event is the authoritative deployed output when available.
        diagnostics_stamp = finite(
            analysis, row, "odom_diagnostics_stamp_s")
        deployed_prediction = None
        deployed_source = "odom_source_event"
        if diagnostics_stamp is not None and odom_stamps:
            odom_index = bisect_right(odom_stamps, diagnostics_stamp) - 1
            if (odom_index >= 0 and
                    abs(odom_stamps[odom_index] - diagnostics_stamp) <= 1.0e-6):
                deployed_prediction = finite(
                    analysis, odom_events[odom_index], "speed_mps")
        if deployed_prediction is None:
            deployed_prediction = finite(
                analysis, row, "odom_diagnostics_speed_mps")
            deployed_source = "odom_diagnostics"
        if deployed_prediction is None:
            deployed_prediction = finite(analysis, row, "speed_mps")
            deployed_source = "odom"
        model_active = finite(
            analysis, row, "odom_sensor_fusion_model_active")
        frozen_model_active = finite(
            analysis, row, "odom_frozen_encoder_model_active")
        if model_active is not None and model_active > 0.5:
            deployed_branch = "learned_model"
        elif frozen_model_active is not None and frozen_model_active > 0.5:
            deployed_branch = "frozen_encoder_model"
        elif runtime_regime == "decelerating":
            deployed_branch = "braking_observer"
        elif runtime_regime == "frozen":
            deployed_branch = "frozen_observer"
        else:
            deployed_branch = "observer_or_wheel"
        examples.append({
            "features": features,
            "target": max(0.0, truth),
            "run_id": run_id,
            "sensor_regime": runtime_regime,
            "truth_regime": ground_truth_regime(
                truth, raw, gt_acceleration),
            "deployed_prediction": deployed_prediction,
            "deployed_source": deployed_source,
            "deployed_branch": deployed_branch,
            "row": row,
        })

        # Match SensorOdometryNode exactly: the instantaneous wheel/map values
        # are exposed as current features, while the short history is updated
        # with the timestamp-window values after prediction.  Using the
        # instantaneous burst/zero derivative here trained a different model
        # from the one the runtime actually evaluates.
        raw_history.append(window_raw)
        mapped_history.append(window_mapped)
        imu_history.append(imu)
        gap_history.append(abs(window_mapped - imu))
        encoder_history.append((encoder_stamp, encoder_position))
        del raw_history[:-HISTORY_WINDOW]
        del mapped_history[:-HISTORY_WINDOW]
        del imu_history[:-HISTORY_WINDOW]
        del gap_history[:-HISTORY_WINDOW]
        del encoder_history[:-HISTORY_WINDOW]

    return examples


def sample_weights(targets):
    return np.where(
        (targets >= 1.0) & (targets < 3.0), 10.0,
        np.where(
            (targets >= 3.0) & (targets < 5.0), 8.0,
            np.where((targets >= 5.0) & (targets < 10.0), 4.0, 1.0)))


def train_models(examples):
    models = {}
    counts = {}
    for regime in REGIMES:
        selected = [example for example in examples
                    if example["sensor_regime"] == regime and
                    example["target"] >= MOVING_SPEED_MPS]
        if len(selected) < 50:
            raise RuntimeError(
                f"not enough runtime-regime examples for {regime}: {len(selected)}")
        features = np.asarray([example["features"] for example in selected])
        targets = np.asarray([example["target"] for example in selected])
        model = ExtraTreesRegressor(**MODEL_KWARGS)
        model.fit(features, targets, sample_weight=sample_weights(targets))
        models[regime] = model
        counts[regime] = len(selected)
    return models, counts


def predict_examples(models, examples):
    predictions = np.zeros(len(examples), dtype=np.float64)
    spreads = np.zeros(len(examples), dtype=np.float64)
    used = np.zeros(len(examples), dtype=bool)
    for regime, model in models.items():
        indices = [index for index, example in enumerate(examples)
                   if example["sensor_regime"] == regime]
        if not indices:
            continue
        features = np.asarray([examples[index]["features"] for index in indices])
        trees = np.asarray([tree.predict(features) for tree in model.estimators_])
        # A median aggregate is less sensitive than a mean to quantised
        # encoder bursts that survive the causal source-time guard.  The
        # generated C++ evaluator uses the same aggregate.
        predictions[indices] = np.clip(np.median(trees, axis=0), 0.0, 30.0)
        spreads[indices] = trees.std(axis=0)
        used[indices] = True
    return predictions, spreads, used


def metrics(rows, model_name, group_key="sensor_regime", groups=None):
    groups = tuple(REGIMES if groups is None else groups)
    result = []
    for regime in (*groups, "all"):
        selected = [row for row in rows if regime == "all" or
                    row.get(group_key) == regime]
        for lower, upper in ((1.0, 3.0), (3.0, 5.0), (5.0, 10.0),
                             (10.0, 15.0), (15.0, 20.0), (20.0, 23.0)):
            selected_bin = [row for row in selected
                            if lower <= row["target"] < upper or
                            (upper == 23.0 and lower <= row["target"] <= upper)]
            if not selected_bin:
                continue
            errors = np.asarray([abs(row["prediction"] - row["target"])
                                 for row in selected_bin])
            targets = np.asarray([row["target"] for row in selected_bin])
            relative = 100.0 * errors / targets
            result.append({
                "model": model_name,
                "regime": regime,
                "bin": f"{lower:g}-{upper:g}_mps",
                "samples": len(selected_bin),
                "absolute_error_median_mps": float(np.median(errors)),
                "absolute_error_p95_mps": float(np.percentile(errors, 95)),
                "relative_error_median_pct": float(np.median(relative)),
                "relative_error_p95_pct": float(np.percentile(relative, 95)),
            })
    return result


def export_array(name, type_name, values, formatter=str, width=12):
    lines = []
    for start in range(0, len(values), width):
        lines.append(", ".join(formatter(value) for value in values[start:start + width]))
    body = ",\n    ".join(lines)
    return f"  inline static constexpr std::array<{type_name}, {len(values)}> {name} = {{\n    {body}\n  }};\n"


def format_float(value):
    text = f"{float(value):.9g}"
    if "." not in text and "e" not in text and "E" not in text:
        text += ".0"
    return text + "f"


def tree_arrays(model):
    trees = [tree.tree_ for tree in model.estimators_]
    roots = []
    features = []
    thresholds = []
    left = []
    right = []
    values = []
    offset = 0
    for tree in trees:
        roots.append(offset)
        features.extend(255 if feature < 0 else int(feature)
                        for feature in tree.feature)
        thresholds.extend(float(value) for value in tree.threshold)
        left.extend(int(value) + offset if value >= 0 else -1
                    for value in tree.children_left)
        right.extend(int(value) + offset if value >= 0 else -1
                     for value in tree.children_right)
        values.extend(float(value[0][0]) for value in tree.value)
        offset += tree.node_count
    return roots, features, thresholds, left, right, values


def export_header(models, destination):
    arrays = {}
    for regime, model in models.items():
        arrays[regime] = tree_arrays(model)
    names = {
        "accelerating": "Accelerating",
        "steady": "Steady",
        "decelerating": "Decelerating",
        "frozen": "Frozen",
        "frozen_accelerating": "FrozenAccelerating",
    }
    blocks = []
    for regime in REGIMES:
        roots, features, thresholds, left, right, values = arrays[regime]
        blocks.append(
            export_array(f"k{names[regime]}Roots", "int32_t", roots) +
            export_array(f"k{names[regime]}Features", "uint8_t", features) +
            export_array(f"k{names[regime]}Thresholds", "float", thresholds, format_float) +
            export_array(f"k{names[regime]}Left", "int32_t", left) +
            export_array(f"k{names[regime]}Right", "int32_t", right) +
            export_array(f"k{names[regime]}Values", "float", values, format_float))

    content = f'''#pragma once

// Generated by fit_regime_sensor_fusion_model.py.
// Ground truth is an offline target only; it is not a runtime input.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace f1tenth_localization
{{

enum class SensorMotionRegime : uint8_t
{{
  ACCELERATING = 0,
  STEADY = 1,
  DECELERATING = 2,
  FROZEN = 3,
  FROZEN_ACCELERATING = 4,
}};

struct SensorFusionPrediction
{{
  double speed_mps;
  double spread_mps;
  bool valid;
}};

class SensorFusionModel final
{{
public:
  static constexpr std::size_t kFeatureCount = {len(FEATURE_NAMES)};

  SensorFusionPrediction predict(
    SensorMotionRegime regime,
    const std::array<double, kFeatureCount> & input) const noexcept
  {{
    switch (regime) {{
      case SensorMotionRegime::ACCELERATING:
        return evaluate(input, kAcceleratingRoots, kAcceleratingFeatures,
          kAcceleratingThresholds, kAcceleratingLeft, kAcceleratingRight,
          kAcceleratingValues);
      case SensorMotionRegime::STEADY:
        return evaluate(input, kSteadyRoots, kSteadyFeatures,
          kSteadyThresholds, kSteadyLeft, kSteadyRight, kSteadyValues);
      case SensorMotionRegime::DECELERATING:
        return evaluate(input, kDeceleratingRoots, kDeceleratingFeatures,
          kDeceleratingThresholds, kDeceleratingLeft, kDeceleratingRight,
          kDeceleratingValues);
      case SensorMotionRegime::FROZEN:
        return evaluate(input, kFrozenRoots, kFrozenFeatures,
          kFrozenThresholds, kFrozenLeft, kFrozenRight, kFrozenValues);
      case SensorMotionRegime::FROZEN_ACCELERATING:
        return evaluate(input, kFrozenAcceleratingRoots,
          kFrozenAcceleratingFeatures, kFrozenAcceleratingThresholds,
          kFrozenAcceleratingLeft, kFrozenAcceleratingRight,
          kFrozenAcceleratingValues);
    }}
    return {{0.0, 0.0, false}};
  }}

private:
  template<std::size_t TreeCount, std::size_t NodeCount>
  static SensorFusionPrediction evaluate(
    const std::array<double, kFeatureCount> & input,
    const std::array<int32_t, TreeCount> & roots,
    const std::array<uint8_t, NodeCount> & features,
    const std::array<float, NodeCount> & thresholds,
    const std::array<int32_t, NodeCount> & left,
    const std::array<int32_t, NodeCount> & right,
    const std::array<float, NodeCount> & values) noexcept
  {{
    std::array<double, TreeCount> predictions{{}};
    double sum = 0.0;
    double sum_squared = 0.0;
    std::size_t prediction_index = 0;
    for (const int32_t root : roots) {{
      int32_t node = root;
      while (features[static_cast<std::size_t>(node)] != 255U) {{
        const auto index = static_cast<std::size_t>(node);
        const auto feature = static_cast<std::size_t>(features[index]);
        node = input[feature] <= static_cast<double>(thresholds[index]) ?
          left[index] : right[index];
      }}
      const double value = values[static_cast<std::size_t>(node)];
      predictions[prediction_index++] = value;
      sum += value;
      sum_squared += value * value;
    }}
    std::sort(predictions.begin(), predictions.end());
    const double count = static_cast<double>(roots.size());
    const double mean = sum / count;
    const double median = TreeCount % 2U == 0U ?
      0.5 * (predictions[TreeCount / 2U - 1U] +
      predictions[TreeCount / 2U]) : predictions[TreeCount / 2U];
    const double variance = std::max(
      0.0, sum_squared / count - mean * mean);
    return {{std::clamp(median, 0.0, 30.0), std::sqrt(variance), true}};
  }}

{''.join(blocks)}}};  // class SensorFusionModel

}}  // namespace f1tenth_localization
'''
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding="utf-8")


def write_metrics(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ("model", "regime", "bin", "samples",
              "absolute_error_median_mps", "absolute_error_p95_mps",
              "relative_error_median_pct", "relative_error_p95_pct")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def score_examples(models, examples):
    predictions, spreads, used = predict_examples(models, examples)
    result = []
    for example, prediction, spread, model_used in zip(
            examples, predictions, spreads, used):
        if not model_used or example["truth_regime"] == "stationary":
            continue
        result.append({
            "target": example["target"],
            "prediction": float(prediction),
            "spread": float(spread),
            "sensor_regime": example["sensor_regime"],
        })
    return result


def score_deployed_examples(examples):
    """Score the speed actually published by the recorded runtime node.

    The exact ``/odom`` source event matched to the diagnostic timestamp is
    preferred. ``odom_diagnostics_speed_mps`` is the first fallback for older
    recordings, followed by the recorder callback's ``speed_mps`` snapshot.
    This is an acceptance metric for the deployed observer, not a replay of
    the offline forest.
    """
    result = []
    for example in examples:
        prediction = example["deployed_prediction"]
        if prediction is None or example["truth_regime"] == "stationary":
            continue
        result.append({
            "target": example["target"],
            "prediction": max(0.0, float(prediction)),
            "sensor_regime": example["sensor_regime"],
            "runtime_branch": example["deployed_branch"],
            "prediction_source": example["deployed_source"],
        })
    return result


def report_rows(models, examples, forest_model_name,
                deployed_model_name="deployed_odom_validation"):
    """Return separate offline-forest and deployed-runtime metric rows."""
    forest_rows = score_examples(models, examples)
    deployed_rows = score_deployed_examples(examples)
    rows = metrics(forest_rows, forest_model_name)
    rows.extend(metrics(deployed_rows, deployed_model_name))
    branches = sorted({row["runtime_branch"] for row in deployed_rows})
    rows.extend(metrics(
        deployed_rows,
        "deployed_odom_branch",
        group_key="runtime_branch",
        groups=branches,
    ))
    return rows, forest_rows, deployed_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs", type=Path, nargs="+",
        help="recordings used for training; never include the holdout here")
    parser.add_argument(
        "--validation", type=Path, nargs="+",
        help="independent recordings scored after fitting on inputs")
    parser.add_argument(
        "--sensor-odometry-config", type=Path,
        default=DEFAULT_SENSOR_ODOMETRY_CONFIG,
        help="runtime sensor_odometry.yaml used for the IMU filter alpha")
    parser.add_argument(
        "--imu-filter-alpha", type=float,
        help="explicit alpha override for controlled experiments")
    parser.add_argument("--header", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    args = parser.parse_args()
    if args.imu_filter_alpha is None:
        imu_filter_alpha = load_imu_filter_alpha(args.sensor_odometry_config)
        alpha_source = str(args.sensor_odometry_config)
    else:
        imu_filter_alpha = args.imu_filter_alpha
        if not 0.0 < imu_filter_alpha <= 1.0:
            raise ValueError(
                f"--imu-filter-alpha must be in (0, 1], got "
                f"{imu_filter_alpha}")
        alpha_source = "--imu-filter-alpha"
    analysis = load_analysis_module()
    runs = []
    for index, path in enumerate(args.inputs):
        examples = build_examples(path, analysis, index, imu_filter_alpha)
        if examples:
            runs.append((path, examples))
    if len(runs) < 2:
        if not args.validation:
            raise RuntimeError(
                "at least two usable recordings are required without --validation")

    if args.validation:
        validation = []
        for index, path in enumerate(args.validation):
            examples = build_examples(
                path, analysis, index, imu_filter_alpha)
            if examples:
                validation.extend(examples)
        if not validation:
            raise RuntimeError("no usable validation recordings")
        training = [example for _, examples in runs for example in examples]
        models, counts = train_models(training)
        metric_rows, scored, deployed_scored = report_rows(
            models, validation, "offline_forest_validation")
        export_header(models, args.header)
        write_metrics(args.metrics, metric_rows)
        print(f"imu_filter_alpha={imu_filter_alpha:.6g} "
              f"source={alpha_source}")
        print(f"training_recordings={len(runs)} "
              f"training_examples={len(training)}")
        print(f"validation_recordings={len(args.validation)} "
              f"validation_examples={len(validation)}")
        print("regime_training_counts=" + ",".join(
            f"{regime}:{counts[regime]}" for regime in REGIMES))
        print(f"offline_forest_examples={len(scored)} "
              f"deployed_odom_examples={len(deployed_scored)}")
        for row in metric_rows:
            if row["regime"] == "all":
                print(f"{row['bin']}: median={row['relative_error_median_pct']:.3f}% "
                      f"p95={row['relative_error_p95_pct']:.3f}% "
                      f"[{row['model']}]")
        print(f"header={args.header}")
        print(f"metrics={args.metrics}")
        return 0

    validation_rows = []
    deployed_validation_rows = []
    for holdout_path, holdout in runs:
        train = [example for path, examples in runs if path != holdout_path
                 for example in examples]
        models, _ = train_models(train)
        validation_rows.extend(score_examples(models, holdout))
        deployed_validation_rows.extend(score_deployed_examples(holdout))

    final_examples = [example for _, examples in runs for example in examples]
    models, counts = train_models(final_examples)
    export_header(models, args.header)
    metric_rows = metrics(
        validation_rows, "offline_forest_leave_one_recording_out")
    metric_rows.extend(metrics(
        deployed_validation_rows, "deployed_odom_leave_one_recording_out"))
    branches = sorted({row["runtime_branch"]
                       for row in deployed_validation_rows})
    metric_rows.extend(metrics(
        deployed_validation_rows,
        "deployed_odom_branch_leave_one_recording_out",
        group_key="runtime_branch",
        groups=branches,
    ))
    write_metrics(args.metrics, metric_rows)
    print(f"imu_filter_alpha={imu_filter_alpha:.6g} "
          f"source={alpha_source}")
    print(f"recordings={len(runs)} examples={len(final_examples)}")
    print("regime_training_counts=" + ",".join(
        f"{regime}:{counts[regime]}" for regime in REGIMES))
    for row in metric_rows:
        if row["regime"] == "all":
            print(f"{row['bin']}: median={row['relative_error_median_pct']:.3f}% "
                  f"p95={row['relative_error_p95_pct']:.3f}% "
                  f"[{row['model']}]")
    print(f"header={args.header}")
    print(f"metrics={args.metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
