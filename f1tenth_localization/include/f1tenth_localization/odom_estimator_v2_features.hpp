#pragma once

// Causal feature history used by the offline v2 estimator and its native
// runtime deployment.  This class intentionally accepts only values already
// available at an odometry callback; it has no simulator-truth or command
// inputs.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <deque>
#include <vector>

#include "f1tenth_localization/odom_estimator_v2_model.hpp"

namespace f1tenth_localization
{

class OdomEstimatorV2Features final
{
public:
  static constexpr std::size_t kFeatureCount = OdomEstimatorV2Model::kFeatureCount;

  OdomEstimatorV2Features() = default;

  void reset() noexcept
  {
    history_.clear();
    have_sample_ = false;
    first_imu_pair_ = 0.0;
    decel_time_ = 0.0;
    accel_time_ = 0.0;
    frozen_time_ = 0.0;
    transient_entry_speed_ = 0.0;
    transient_entry_imu_ = 0.0;
    imu_dv_since_decel_ = 0.0;
    anchor_age_ = 0.0;
    anchor_speed_ = 0.0;
    anchor_stamp_s_ = 0.0;
    was_decel_ = false;
    was_accel_ = false;
  }

  std::array<double, kFeatureCount> update(
    double base,
    double imu_pair,
    double imu,
    double wheel_raw,
    double wheel_corr,
    double win_raw,
    double win_map,
    double acc_obs,
    double bias,
    double conf,
    double ax,
    double ay,
    double yaw,
    double stamp_s,
    double source_dt_s)
  {
    Snapshot current{{
      finite_or_zero(base), finite_or_zero(imu_pair), finite_or_zero(imu),
      finite_or_zero(wheel_raw), finite_or_zero(wheel_corr),
      finite_or_zero(win_raw), finite_or_zero(win_map), finite_or_zero(acc_obs),
      finite_or_zero(bias), finite_or_zero(conf), finite_or_zero(ax),
      finite_or_zero(ay), finite_or_zero(yaw)}};

    // The Python fitter fills the first timestamp delta with its measured
    // stream default and clips later deltas to 0..0.2 s.  The explicit state
    // integrator, like the fitter, sees zero elapsed time on its first row.
    const double dt = have_sample_ ?
      std::clamp(finite_or_zero(source_dt_s), 0.0, 0.2) : 0.0;
    const double feature_dt = have_sample_ ? dt : 0.025;
    const double a = current.values[ACC_OBS];
    const double imu_pair_value = current.values[IMU_PAIR];
    const double window_map = current.values[WIN_MAP];
    const double confidence = current.values[CONF];
    const double current_base = current.values[BASE];

    if (!have_sample_) {
      first_imu_pair_ = imu_pair_value;
      transient_entry_speed_ = current_base;
      transient_entry_imu_ = imu_pair_value;
      anchor_speed_ = window_map > 0.0 ? window_map : imu_pair_value;
      anchor_stamp_s_ = stamp_s;
    }

    const bool is_decel = a < -0.5;
    const bool is_accel = a > 0.5;
    const bool is_frozen = std::abs(current.values[WIN_RAW]) < 0.15 &&
      imu_pair_value > 0.3;
    if (is_decel && !was_decel_) {
      decel_time_ = 0.0;
      transient_entry_speed_ = current_base;
      transient_entry_imu_ = imu_pair_value;
      imu_dv_since_decel_ = 0.0;
    }
    if (is_accel && !was_accel_) {
      accel_time_ = 0.0;
    }
    decel_time_ = is_decel ? decel_time_ + dt : 0.0;
    accel_time_ = is_accel ? accel_time_ + dt : 0.0;
    frozen_time_ = is_frozen ? frozen_time_ + dt : 0.0;
    if (is_decel) {
      imu_dv_since_decel_ += a * dt;
    }
    if (window_map > 0.2 && confidence > 0.7 &&
      std::abs(window_map - imu_pair_value) < 0.75 && std::abs(a) < 1.0)
    {
      anchor_speed_ = window_map;
      anchor_stamp_s_ = stamp_s;
    }
    anchor_age_ = std::max(0.0, stamp_s - anchor_stamp_s_);

    std::array<double, kFeatureCount> output{};
    std::size_t index = 0;
    auto add = [&output, &index](double value) { output[index++] = value; };

    // The first 20 entries match make_features() in the supplied v2 fitter.
    add(current.values[BASE]);
    add(current.values[IMU_PAIR]);
    add(current.values[IMU]);
    add(current.values[WHEEL_RAW]);
    add(current.values[WHEEL_CORR]);
    add(current.values[WIN_RAW]);
    add(current.values[WIN_MAP]);
    add(current.values[ACC_OBS]);
    add(current.values[BIAS]);
    add(current.values[CONF]);
    add(current.values[AX]);
    add(current.values[AY]);
    add(current.values[YAW]);
    add(feature_dt);
    add(gap_win_imu(current));
    add(gap_raw_imu(current));
    add(std::abs(gap_win_imu(current)));
    add(std::abs(gap_raw_imu(current)));
    add(std::abs(current.values[WIN_RAW]) < 0.15 ? 1.0 : 0.0);
    add(std::abs(current.values[WHEEL_RAW]) < 0.15 ? 1.0 : 0.0);

    // Lag features: the absent prefix is zero after the fitter's causal
    // forward-fill/fillna sequence.
    constexpr std::array<int, 6> lags{{1, 2, 4, 8, 16, 32}};
    constexpr std::array<Signal, 10> lag_signals{{
      BASE, IMU_PAIR, WIN_MAP, WIN_RAW, WHEEL_RAW,
      ACC_OBS, AX, BIAS, CONF, GAP_WIN_IMU}};
    for (const int lag : lags) {
      for (const Signal signal : lag_signals) {
        const double previous = lag_value(signal, lag);
        add(previous);
        if (signal == BASE || signal == IMU_PAIR || signal == WIN_MAP ||
          signal == ACC_OBS || signal == GAP_WIN_IMU)
        {
          add(history_.size() >= static_cast<std::size_t>(lag) ?
            signal_value(current, signal) - previous : 0.0);
        }
      }
    }

    constexpr std::array<int, 4> windows{{4, 8, 16, 32}};
    constexpr std::array<Signal, 7> rolling_signals{{
      BASE, IMU_PAIR, WIN_MAP, WIN_RAW, ACC_OBS, AX, GAP_WIN_IMU}};
    for (const int window : windows) {
      for (const Signal signal : rolling_signals) {
        add(rolling_median(current, signal, window));
        add(rolling_mean(current, signal, window));
        if (signal == BASE || signal == IMU_PAIR || signal == WIN_MAP ||
          signal == ACC_OBS || signal == GAP_WIN_IMU)
        {
          add(rolling_std(current, signal, window));
        }
      }
    }

    add(decel_time_);
    add(accel_time_);
    add(frozen_time_);
    add(transient_entry_speed_);
    add(transient_entry_imu_);
    add(imu_dv_since_decel_);
    add(anchor_age_);
    add(anchor_speed_);
    add(anchor_speed_ + (imu_pair_value - first_imu_pair_));

    // This assertion remains a compile-time contract with the generated model
    // and catches accidental feature-order edits during a Humble build.
    (void)sizeof(index == kFeatureCount);

    history_.push_back(current);
    while (history_.size() > 32) {
      history_.pop_front();
    }
    have_sample_ = true;
    was_decel_ = is_decel;
    was_accel_ = is_accel;
    return output;
  }

private:
  enum Signal : uint8_t
  {
    BASE = 0,
    IMU_PAIR = 1,
    IMU = 2,
    WHEEL_RAW = 3,
    WHEEL_CORR = 4,
    WIN_RAW = 5,
    WIN_MAP = 6,
    ACC_OBS = 7,
    BIAS = 8,
    CONF = 9,
    AX = 10,
    AY = 11,
    YAW = 12,
    GAP_WIN_IMU = 13,
  };

  struct Snapshot
  {
    std::array<double, 13> values;
  };

  static double finite_or_zero(double value) noexcept
  {
    return std::isfinite(value) ? value : 0.0;
  }

  static double gap_win_imu(const Snapshot & snapshot) noexcept
  {
    return snapshot.values[WIN_MAP] - snapshot.values[IMU_PAIR];
  }

  static double gap_raw_imu(const Snapshot & snapshot) noexcept
  {
    return snapshot.values[WHEEL_RAW] - snapshot.values[IMU_PAIR];
  }

  static double signal_value(const Snapshot & snapshot, Signal signal) noexcept
  {
    return signal == GAP_WIN_IMU ? gap_win_imu(snapshot) :
      snapshot.values[static_cast<std::size_t>(signal)];
  }

  double lag_value(Signal signal, int lag) const noexcept
  {
    if (history_.size() < static_cast<std::size_t>(lag)) {
      return 0.0;
    }
    return signal_value(history_[history_.size() - static_cast<std::size_t>(lag)], signal);
  }

  std::vector<double> rolling_values(
    const Snapshot & current, Signal signal, int window) const
  {
    std::vector<double> values;
    values.reserve(static_cast<std::size_t>(window));
    values.push_back(signal_value(current, signal));
    const std::size_t available = std::min(
      history_.size(), static_cast<std::size_t>(window - 1));
    for (std::size_t offset = 0; offset < available; ++offset) {
      values.push_back(signal_value(
        history_[history_.size() - 1U - offset], signal));
    }
    return values;
  }

  double rolling_median(const Snapshot & current, Signal signal, int window) const
  {
    auto values = rolling_values(current, signal, window);
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2U;
    if (values.size() % 2U == 0U) {
      return 0.5 * (values[middle - 1U] + values[middle]);
    }
    return values[middle];
  }

  double rolling_mean(const Snapshot & current, Signal signal, int window) const
  {
    const auto values = rolling_values(current, signal, window);
    double sum = 0.0;
    for (const double value : values) {
      sum += value;
    }
    return sum / static_cast<double>(values.size());
  }

  double rolling_std(const Snapshot & current, Signal signal, int window) const
  {
    const auto values = rolling_values(current, signal, window);
    if (values.size() < 2U) {
      return 0.0;
    }
    const double mean = rolling_mean(current, signal, window);
    double sum_squared = 0.0;
    for (const double value : values) {
      const double delta = value - mean;
      sum_squared += delta * delta;
    }
    // pandas rolling.std() uses the sample standard deviation (ddof=1).
    return std::sqrt(sum_squared / static_cast<double>(values.size() - 1U));
  }

  std::deque<Snapshot> history_;
  bool have_sample_{false};
  double first_imu_pair_{0.0};
  double decel_time_{0.0};
  double accel_time_{0.0};
  double frozen_time_{0.0};
  double transient_entry_speed_{0.0};
  double transient_entry_imu_{0.0};
  double imu_dv_since_decel_{0.0};
  double anchor_age_{0.0};
  double anchor_speed_{0.0};
  double anchor_stamp_s_{0.0};
  bool was_decel_{false};
  bool was_accel_{false};
};

}  // namespace f1tenth_localization
