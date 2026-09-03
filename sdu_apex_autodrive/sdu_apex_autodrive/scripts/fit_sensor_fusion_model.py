#!/usr/bin/env python3
"""Fit and export the sensor-only longitudinal speed model.

The fit is offline-only. Ground truth is used as the training target and for
cross-validation, but the generated C++ model consumes only the IMU, encoder,
and timing features available to the production odometry node.
"""

import argparse
import csv
import importlib.util
import math
from pathlib import Path
import statistics
from typing import Iterable

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold


FEATURE_NAMES = (
    "imu_speed_mps",
    "mapped_wheel_speed_mps",
    "raw_wheel_speed_mps",
    "wheel_imu_gap_mps",
    "wheel_observation_confidence",
    "abs_imu_longitudinal_acceleration_mps2",
    "abs_imu_lateral_acceleration_mps2",
    "abs_imu_yaw_rate_radps",
    "encoder_dt_s",
    "median_raw_wheel_speed_mps",
    "median_mapped_wheel_speed_mps",
    "median_imu_speed_mps",
    "median_wheel_imu_gap_mps",
    "mapped_wheel_speed_delta_mps",
    "imu_speed_delta_mps",
    "raw_wheel_speed_delta_mps",
)

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


def _median(values: Iterable[float], current: float) -> float:
    samples = list(values)[-5:]
    samples.append(current)
    return statistics.median(samples)


def _previous(values: list[float], current: float) -> float:
    return values[-1] if values else current


def build_examples(rows: list[dict[str, str]], analysis):
    """Build causal sensor features and labels in source-event order."""
    examples = []
    raw_history: list[float] = []
    mapped_history: list[float] = []
    imu_history: list[float] = []
    gap_history: list[float] = []
    episode = 0

    for row in analysis.timestamped_rows(rows):
        phase = row.get("phase", "")
        if analysis.phase_is_diagnostic_reset(phase):
            episode += 1
            raw_history.clear()
            mapped_history.clear()
            imu_history.clear()
            gap_history.clear()
            continue

        def value(name: str):
            return analysis.finite(row.get(name))

        truth = value("gt_speed_mps")
        imu = value("odom_imu_speed_mps")
        mapped = value("odom_corrected_wheel_speed_mps")
        raw = value("odom_raw_wheel_speed_mps")
        confidence = value("odom_wheel_observation_confidence")
        if any(item is None for item in (truth, imu, mapped, raw, confidence)):
            continue

        # The identification run is forward-only. Absolute values make the
        # runtime feature contract explicit and leave reverse motion for a
        # separately identified model rather than silently changing signs.
        imu = max(0.0, imu)
        mapped = abs(mapped)
        raw = abs(raw)
        gap = abs(mapped - imu)
        ax = abs(value("ax_mps2") or 0.0)
        ay = abs(value("ay_mps2") or 0.0)
        yaw_rate = abs(value("imu_yaw_rate_radps") or 0.0)
        encoder_dt = value("left_encoder_dt_s") or 0.0
        encoder_dt = max(0.0, encoder_dt)

        features = np.asarray([
            imu,
            mapped,
            raw,
            gap,
            max(0.0, confidence),
            ax,
            ay,
            yaw_rate,
            encoder_dt,
            _median(raw_history, raw),
            _median(mapped_history, mapped),
            _median(imu_history, imu),
            _median(gap_history, gap),
            mapped - _previous(mapped_history, mapped),
            imu - _previous(imu_history, imu),
            raw - _previous(raw_history, raw),
        ], dtype=np.float64)
        examples.append((features, max(0.0, truth), episode, row))

        raw_history.append(raw)
        mapped_history.append(mapped)
        imu_history.append(imu)
        gap_history.append(gap)
        for history in (raw_history, mapped_history, imu_history, gap_history):
            del history[:-5]

    return examples


def sample_weights(targets: np.ndarray) -> np.ndarray:
    """Prioritize the requested 0-10 m/s operating envelope."""
    return np.where(
        (targets >= 1.0) & (targets < 3.0), 5.0,
        np.where(
            (targets >= 3.0) & (targets < 5.0), 5.0,
            np.where((targets >= 5.0) & (targets < 10.0), 4.0, 1.0)))


def metric_rows(targets: np.ndarray, predictions: np.ndarray, model_name: str):
    rows = []
    for lower, upper in ((1.0, 3.0), (3.0, 5.0), (5.0, 10.0), (1.0, 10.0)):
        mask = (targets >= lower) & (targets < upper)
        errors = np.abs(predictions[mask] - targets[mask])
        relative = 100.0 * errors / targets[mask]
        rows.append({
            "model": model_name,
            "bin": f"{lower:g}-{upper:g}_mps",
            "samples": int(mask.sum()),
            "absolute_error_median_mps": float(np.median(errors)),
            "absolute_error_p95_mps": float(np.percentile(errors, 95)),
            "relative_error_median_pct": float(np.median(relative)),
            "relative_error_p95_pct": float(np.percentile(relative, 95)),
        })
    return rows


def position_metric_rows(examples, predictions: np.ndarray, analysis):
    """Replay model speed causally into planar pose for each reset episode."""
    state = {}
    samples = []
    for example, prediction in zip(examples, predictions):
        _, _, episode, row = example
        stamp = analysis.finite(row.get("odom_stamp_s"))
        if stamp is None:
            stamp = analysis.finite(row.get("stamp_s"))
        x, y = analysis.ground_truth_position(row)
        yaw = analysis.finite(row.get("yaw_odom_rad"))
        if None in (stamp, x, y, yaw):
            continue
        if episode not in state:
            state[episode] = [0.0, 0.0, stamp, x, y]
            continue
        pose_x, pose_y, previous_stamp, origin_x, origin_y = state[episode]
        dt = stamp - previous_stamp
        state[episode][2] = stamp
        if not 1.0e-4 < dt <= 0.5:
            continue
        pose_x += float(prediction) * dt * math.cos(yaw)
        pose_y += float(prediction) * dt * math.sin(yaw)
        state[episode][0] = pose_x
        state[episode][1] = pose_y
        truth_x = x - origin_x
        truth_y = y - origin_y
        distance = math.hypot(truth_x, truth_y)
        error = math.hypot(pose_x - truth_x, pose_y - truth_y)
        samples.append((distance, error))

    rows = []
    for label, selected in (
        ("all_active", [item for item in samples if item[0] >= 1.0]),
        ("distance_ge_10m", [item for item in samples if item[0] >= 10.0]),
        ("distance_ge_50m", [item for item in samples if item[0] >= 50.0]),
        ("distance_ge_100m", [item for item in samples if item[0] >= 100.0]),
    ):
        if not selected:
            continue
        errors = np.asarray([item[1] for item in selected], dtype=np.float64)
        references = np.asarray([item[0] for item in selected], dtype=np.float64)
        rows.append({
            "model": "random_forest_grouped_cv_replayed_pose",
            "bin": label,
            "samples": len(selected),
            "absolute_error_median_m": float(np.median(errors)),
            "absolute_error_p95_m": float(np.percentile(errors, 95)),
            "relative_error_median_pct": float(np.median(100.0 * errors / references)),
            "relative_error_p95_pct": float(np.percentile(100.0 * errors / references, 95)),
        })
    return rows


def format_float(value: float) -> str:
    if not math.isfinite(value):
        return "0.0f"
    rendered = f"{value:.9g}"
    if "." not in rendered and "e" not in rendered and "E" not in rendered:
        rendered += ".0"
    return f"{rendered}f"


def export_header(model: RandomForestRegressor, destination: Path) -> None:
    """Export the sklearn forest as a small dependency-free C++ evaluator."""
    features: list[int] = []
    thresholds: list[float] = []
    left: list[int] = []
    right: list[int] = []
    values: list[float] = []
    roots: list[int] = []
    offset = 0

    for estimator in model.estimators_:
        tree = estimator.tree_
        roots.append(offset)
        for node in range(tree.node_count):
            feature = int(tree.feature[node])
            features.append(255 if feature < 0 else feature)
            thresholds.append(float(tree.threshold[node]))
            left.append(-1 if tree.children_left[node] < 0 else offset + int(tree.children_left[node]))
            right.append(-1 if tree.children_right[node] < 0 else offset + int(tree.children_right[node]))
            values.append(float(tree.value[node][0][0]))
        offset += tree.node_count

    def array(name: str, type_name: str, values: list, formatter) -> str:
        lines = [f"  inline static constexpr std::array<{type_name}, {len(values)}> {name} = {{"]
        for index in range(0, len(values), 12):
            lines.append("    " + ", ".join(formatter(item) for item in values[index:index + 12]) + ",")
        lines.append("  };")
        return "\n".join(lines)

    content = f'''#pragma once

// Generated by fit_sensor_fusion_model.py from the open-world identification
// grid. Ground truth is a fit target only; it is not a runtime input.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace f1tenth_localization
{{

struct SensorFusionPrediction
{{
  double speed_mps;
  double spread_mps;
}};

class SensorFusionModel final
{{
public:
  static constexpr std::size_t kFeatureCount = {len(FEATURE_NAMES)};

  SensorFusionPrediction predict(
    const std::array<double, kFeatureCount> & input) const noexcept
  {{
    double sum = 0.0;
    double sum_squared = 0.0;
    for (const int32_t root : kRoots) {{
      int32_t node = root;
      while (kFeatures[static_cast<std::size_t>(node)] != 255U) {{
        const auto index = static_cast<std::size_t>(node);
        const auto feature = static_cast<std::size_t>(kFeatures[index]);
        node = input[feature] <= static_cast<double>(kThresholds[index]) ?
          kLeft[index] : kRight[index];
      }}
      const double value = kValues[static_cast<std::size_t>(node)];
      sum += value;
      sum_squared += value * value;
    }}
    const double count = static_cast<double>(kRoots.size());
    const double mean = sum / count;
    const double variance = std::max(0.0, sum_squared / count - mean * mean);
    return {{std::clamp(mean, 0.0, 30.0), std::sqrt(variance)}};
  }}

{array("kRoots", "int32_t", roots, str)}
{array("kFeatures", "uint8_t", features, str)}
{array("kThresholds", "float", thresholds, format_float)}
{array("kLeft", "int32_t", left, str)}
{array("kRight", "int32_t", right, str)}
{array("kValues", "float", values, format_float)}
}};  // class SensorFusionModel

}}  // namespace f1tenth_localization
'''
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--header", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--position-metrics", type=Path)
    args = parser.parse_args()

    analysis = load_analysis_module()
    rows = analysis.read_rows(args.input)
    rows, _ = analysis.deduplicate_source_events(rows)
    examples = build_examples(rows, analysis)
    if len(examples) < 100 or len({item[2] for item in examples}) < 5:
        raise RuntimeError("identification data does not contain enough independent episodes")

    features = np.asarray([item[0] for item in examples], dtype=np.float64)
    targets = np.asarray([item[1] for item in examples], dtype=np.float64)
    groups = np.asarray([item[2] for item in examples], dtype=np.int64)
    weights = sample_weights(targets)

    cv_predictions = np.zeros_like(targets)
    for train, test in GroupKFold(n_splits=5).split(features, targets, groups):
        candidate = RandomForestRegressor(**MODEL_KWARGS)
        candidate.fit(features[train], targets[train], sample_weight=weights[train])
        cv_predictions[test] = np.clip(candidate.predict(features[test]), 0.0, 30.0)

    final_model = RandomForestRegressor(**MODEL_KWARGS)
    final_model.fit(features, targets, sample_weight=weights)
    export_header(final_model, args.header)

    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "model", "bin", "samples", "absolute_error_median_mps",
        "absolute_error_p95_mps", "relative_error_median_pct",
        "relative_error_p95_pct",
    )
    with args.metrics.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metric_rows(targets, cv_predictions, "random_forest_grouped_cv"))

    if args.position_metrics is not None:
        args.position_metrics.parent.mkdir(parents=True, exist_ok=True)
        with args.position_metrics.open("w", newline="", encoding="utf-8") as stream:
            position_fields = (
                "model", "bin", "samples", "absolute_error_median_m",
                "absolute_error_p95_m", "relative_error_median_pct",
                "relative_error_p95_pct",
            )
            writer = csv.DictWriter(stream, fieldnames=position_fields)
            writer.writeheader()
            writer.writerows(position_metric_rows(examples, cv_predictions, analysis))

    print(f"examples={len(examples)} episodes={len(set(groups))}")
    for row in metric_rows(targets, cv_predictions, "random_forest_grouped_cv"):
        print(
            f"{row['bin']}: median={row['relative_error_median_pct']:.3f}% "
            f"p95={row['relative_error_p95_pct']:.3f}% "
            f"abs_median={row['absolute_error_median_mps']:.4f}m/s")
    print(f"header={args.header}")
    print(f"metrics={args.metrics}")
    if args.position_metrics is not None:
        print(f"position_metrics={args.position_metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
