#!/usr/bin/env python3
"""Fit causal, regime-specific IMU/encoder speed models.

Ground truth is used only to label and score the offline fit.  The generated
C++ evaluator consumes IMU, encoder, and timing features only.  The normal
models deliberately exclude moving frozen-encoder samples; those samples are
fit by the frozen branch instead of being mixed with rolling-wheel motion.
"""

import argparse
import csv
import importlib.util
import math
from pathlib import Path
import statistics

import numpy as np
from sklearn.ensemble import RandomForestRegressor


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
)

REGIMES = ("accelerating", "steady", "decelerating", "frozen")
WHEEL_FROZEN_SPEED_MPS = 0.15
MOVING_SPEED_MPS = 0.75
REGIME_ACCELERATION_MPS2 = 0.50
STATIONARY_SPEED_MPS = 0.30
IMU_FILTER_ALPHA = 0.70
HISTORY_WINDOW = 9
WHEEL_RADIUS_M = 0.0590
WHEEL_MAP_WHEEL_MPS = np.asarray(
    (0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0,
     16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0),
    dtype=np.float64)
WHEEL_MAP_BODY_MPS = np.asarray(
    (0.0, 1.988601, 3.967925, 5.888532, 8.549802, 9.662430,
     11.536114, 13.473328, 15.270020, 16.855770, 18.570086,
     19.279928, 19.987528, 20.287600, 20.287600, 20.287600),
    dtype=np.float64)

MODEL_KWARGS = {
    "n_estimators": 50,
    "max_depth": 10,
    "min_samples_leaf": 8,
    "max_features": 0.8,
    "random_state": 10,
    "n_jobs": -1,
}


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


def median(history, current):
    values = list(history[-HISTORY_WINDOW:])
    values.append(current)
    return statistics.median(values)


def body_speed_from_wheel_speed(wheel_speed):
    return float(np.interp(
        max(0.0, wheel_speed), WHEEL_MAP_WHEEL_MPS, WHEEL_MAP_BODY_MPS))


def build_examples(path, analysis, run_id):
    rows = analysis.read_rows(path)
    rows, _ = analysis.deduplicate_source_events(rows)
    rows = analysis.timestamped_rows(
        [row for row in rows if analysis.valid_ground_truth_row(row)])

    examples = []
    raw_history = []
    mapped_history = []
    imu_history = []
    gap_history = []
    encoder_history = []
    filtered_acceleration = 0.0
    previous_imu_stamp = None

    for row in rows:
        phase = row.get("phase", "")
        if analysis.phase_is_diagnostic_reset(phase):
            raw_history.clear()
            mapped_history.clear()
            imu_history.clear()
            gap_history.clear()
            encoder_history.clear()
            filtered_acceleration = 0.0
            previous_imu_stamp = None
            continue

        truth = finite(analysis, row, "gt_speed_mps")
        imu = finite(analysis, row, "odom_imu_speed_mps")
        mapped = finite(analysis, row, "odom_corrected_wheel_speed_mps")
        raw = finite(analysis, row, "odom_raw_wheel_speed_mps")
        confidence = finite(analysis, row, "odom_wheel_observation_confidence")
        acceleration = finite(analysis, row, "ax_mps2", 0.0)
        lateral = finite(analysis, row, "ay_mps2", 0.0)
        yaw_rate = finite(analysis, row, "imu_yaw_rate_radps", 0.0)
        gt_acceleration = finite(analysis, row, "gt_longitudinal_accel_mps2", 0.0)
        left_angle = finite(analysis, row, "left_encoder_rad")
        right_angle = finite(analysis, row, "right_encoder_rad")
        if any(value is None for value in (
                truth, imu, mapped, raw, confidence, left_angle, right_angle)):
            continue

        imu = max(0.0, imu)
        mapped = abs(mapped)
        raw = abs(raw)
        confidence = max(0.0, confidence)
        imu_stamp = finite(analysis, row, "imu_stamp_s")
        if imu_stamp is None:
            imu_stamp = finite(analysis, row, "stamp_s", 0.0)
        if previous_imu_stamp is None or imu_stamp <= previous_imu_stamp:
            filtered_acceleration = acceleration
        else:
            filtered_acceleration = (
                IMU_FILTER_ALPHA * acceleration +
                (1.0 - IMU_FILTER_ALPHA) * filtered_acceleration)
        previous_imu_stamp = imu_stamp

        signed_gap = mapped - imu
        gap = abs(signed_gap)
        encoder_dt = max(0.0, finite(analysis, row, "left_encoder_dt_s", 0.0))
        left_stamp = finite(analysis, row, "left_encoder_stamp_s")
        right_stamp = finite(analysis, row, "right_encoder_stamp_s")
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
        ], dtype=np.float64)
        examples.append({
            "features": features,
            "target": max(0.0, truth),
            "run_id": run_id,
            "sensor_regime": sensor_regime(imu, raw, acceleration),
            "truth_regime": ground_truth_regime(
                truth, raw, gt_acceleration),
            "row": row,
        })

        raw_history.append(raw)
        mapped_history.append(mapped)
        imu_history.append(imu)
        gap_history.append(signed_gap)
        encoder_history.append((encoder_stamp, encoder_position))
        del raw_history[:-HISTORY_WINDOW]
        del mapped_history[:-HISTORY_WINDOW]
        del imu_history[:-HISTORY_WINDOW]
        del gap_history[:-HISTORY_WINDOW]
        del encoder_history[:-HISTORY_WINDOW]

    return examples


def sample_weights(targets):
    return np.where(
        (targets >= 1.0) & (targets < 3.0), 5.0,
        np.where(
            (targets >= 3.0) & (targets < 5.0), 5.0,
            np.where((targets >= 5.0) & (targets < 10.0), 4.0, 1.0)))


def train_models(examples):
    models = {}
    counts = {}
    for regime in REGIMES:
        selected = [example for example in examples
                    if example["truth_regime"] == regime and
                    (regime == "frozen" or
                     example["features"][12] < 0.5)]
        if len(selected) < 50:
            raise RuntimeError(
                f"not enough examples for {regime}: {len(selected)}")
        features = np.asarray([example["features"] for example in selected])
        targets = np.asarray([example["target"] for example in selected])
        model = RandomForestRegressor(**MODEL_KWARGS)
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
                   if example["sensor_regime"] == regime and
                   (regime == "frozen" or example["features"][12] < 0.5)]
        if not indices:
            continue
        features = np.asarray([examples[index]["features"] for index in indices])
        trees = np.asarray([tree.predict(features) for tree in model.estimators_])
        predictions[indices] = np.clip(trees.mean(axis=0), 0.0, 30.0)
        spreads[indices] = trees.std(axis=0)
        used[indices] = True
    return predictions, spreads, used


def metrics(rows, model_name):
    result = []
    for regime in (*REGIMES, "all"):
        selected = [row for row in rows if regime == "all" or
                    row["sensor_regime"] == regime]
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
    double sum = 0.0;
    double sum_squared = 0.0;
    for (const int32_t root : roots) {{
      int32_t node = root;
      while (features[static_cast<std::size_t>(node)] != 255U) {{
        const auto index = static_cast<std::size_t>(node);
        const auto feature = static_cast<std::size_t>(features[index]);
        node = input[feature] <= static_cast<double>(thresholds[index]) ?
          left[index] : right[index];
      }}
      const double value = values[static_cast<std::size_t>(node)];
      sum += value;
      sum_squared += value * value;
    }}
    const double count = static_cast<double>(roots.size());
    const double mean = std::clamp(sum / count, 0.0, 30.0);
    const double variance = std::max(0.0, sum_squared / count - mean * mean);
    return {{mean, std::sqrt(variance), true}};
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--header", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    args = parser.parse_args()
    analysis = load_analysis_module()
    runs = []
    for index, path in enumerate(args.inputs):
        examples = build_examples(path, analysis, index)
        if examples:
            runs.append((path, examples))
    if len(runs) < 2:
        raise RuntimeError("at least two usable recordings are required")

    validation_rows = []
    for holdout_path, holdout in runs:
        train = [example for path, examples in runs if path != holdout_path
                 for example in examples]
        models, _ = train_models(train)
        predictions, spreads, used = predict_examples(models, holdout)
        for example, prediction, spread, model_used in zip(
                holdout, predictions, spreads, used):
            if not model_used or example["truth_regime"] == "stationary":
                continue
            validation_rows.append({
                "target": example["target"],
                "prediction": float(prediction),
                "spread": float(spread),
                "sensor_regime": example["sensor_regime"],
            })

    final_examples = [example for _, examples in runs for example in examples]
    models, counts = train_models(final_examples)
    export_header(models, args.header)
    write_metrics(args.metrics,
                   metrics(validation_rows, "leave_one_recording_out"))
    print(f"recordings={len(runs)} examples={len(final_examples)}")
    print("regime_training_counts=" + ",".join(
        f"{regime}:{counts[regime]}" for regime in REGIMES))
    for row in metrics(validation_rows, "leave_one_recording_out"):
        if row["regime"] == "all":
            print(f"{row['bin']}: median={row['relative_error_median_pct']:.3f}% "
                  f"p95={row['relative_error_p95_pct']:.3f}%")
    print(f"header={args.header}")
    print(f"metrics={args.metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
