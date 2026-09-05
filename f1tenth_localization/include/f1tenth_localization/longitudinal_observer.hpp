#pragma once

#include <algorithm>
#include <cmath>

namespace f1tenth_localization
{

/**
 * @brief Small causal two-state Kalman observer for longitudinal motion.
 *
 * The state is [body speed, IMU longitudinal acceleration bias].  IMU
 * acceleration is the prediction input.  A wheel/body-speed estimate is an
 * optional scalar measurement; callers decide whether the wheel is physically
 * informative before calling update().  This keeps wheel spin and a frozen
 * driven encoder out of the state update without using simulator truth.
 */
class LongitudinalObserver final
{
public:
  void configure(
    double acceleration_noise_mps2,
    double bias_random_walk_mps3,
    double initial_speed_variance,
    double initial_bias_variance,
    double maximum_speed_mps)
  {
    acceleration_noise_mps2_ = std::max(0.01, acceleration_noise_mps2);
    bias_random_walk_mps3_ = std::max(0.0001, bias_random_walk_mps3);
    initial_speed_variance_ = std::max(1.0e-6, initial_speed_variance);
    initial_bias_variance_ = std::max(1.0e-6, initial_bias_variance);
    maximum_speed_mps_ = std::max(1.0, maximum_speed_mps);
    reset();
  }

  void reset()
  {
    speed_mps_ = 0.0;
    bias_mps2_ = 0.0;
    covariance_speed_speed_ = initial_speed_variance_;
    covariance_speed_bias_ = 0.0;
    covariance_bias_bias_ = initial_bias_variance_;
  }

  void predict(double acceleration_mps2, double dt_s)
  {
    if (!std::isfinite(acceleration_mps2) ||
      !std::isfinite(dt_s) || dt_s <= 1.0e-4 || dt_s > 0.5)
    {
      return;
    }

    const double dt = std::clamp(dt_s, 1.0e-4, 0.5);
    const double old_speed = speed_mps_;
    const double old_bias = bias_mps2_;
    const double old_p_ss = covariance_speed_speed_;
    const double old_p_sb = covariance_speed_bias_;
    const double old_p_bb = covariance_bias_bias_;

    speed_mps_ = std::clamp(
      old_speed + (acceleration_mps2 - old_bias) * dt,
      0.0, maximum_speed_mps_);

    // F = [[1, -dt], [0, 1]].  Acceleration noise enters speed through dt;
    // the bias is a random walk whose variance grows with elapsed time.
    covariance_speed_speed_ = old_p_ss - 2.0 * dt * old_p_sb +
      dt * dt * old_p_bb +
      acceleration_noise_mps2_ * acceleration_noise_mps2_ * dt * dt;
    covariance_speed_bias_ = old_p_sb - dt * old_p_bb;
    covariance_bias_bias_ = old_p_bb +
      bias_random_walk_mps3_ * bias_random_walk_mps3_ * dt;
    clamp_covariance();
  }

  bool update(double measured_speed_mps, double measurement_variance, double gate_mps)
  {
    if (!std::isfinite(measured_speed_mps) ||
      !std::isfinite(measurement_variance) || measurement_variance <= 0.0)
    {
      return false;
    }

    const double measurement = std::clamp(
      measured_speed_mps, 0.0, maximum_speed_mps_);
    const double innovation = measurement - speed_mps_;
    if (std::isfinite(gate_mps) && gate_mps > 0.0 &&
      std::abs(innovation) > gate_mps)
    {
      return false;
    }

    const double innovation_variance = std::max(
      1.0e-9, covariance_speed_speed_ + measurement_variance);
    const double gain_speed = covariance_speed_speed_ / innovation_variance;
    const double gain_bias = covariance_speed_bias_ / innovation_variance;
    const double old_p_ss = covariance_speed_speed_;
    const double old_p_sb = covariance_speed_bias_;
    const double old_p_bb = covariance_bias_bias_;

    speed_mps_ = std::clamp(
      speed_mps_ + gain_speed * innovation, 0.0, maximum_speed_mps_);
    bias_mps2_ += gain_bias * innovation;

    // Joseph-equivalent scalar update for H=[1,0].
    covariance_speed_speed_ = (1.0 - gain_speed) * old_p_ss;
    covariance_speed_bias_ = (1.0 - gain_speed) * old_p_sb;
    covariance_bias_bias_ = old_p_bb - gain_bias * old_p_sb;
    clamp_covariance();
    return true;
  }

  void set_speed(double speed_mps)
  {
    speed_mps_ = std::clamp(
      std::isfinite(speed_mps) ? speed_mps : 0.0,
      0.0, maximum_speed_mps_);
  }

  void add_speed_delta(double delta_mps)
  {
    if (std::isfinite(delta_mps)) {
      speed_mps_ = std::clamp(
        speed_mps_ + delta_mps, 0.0, maximum_speed_mps_);
    }
  }

  double speed() const {return speed_mps_;}
  double bias() const {return bias_mps2_;}
  double speed_variance() const {return covariance_speed_speed_;}

private:
  void clamp_covariance()
  {
    covariance_speed_speed_ = std::max(1.0e-9, covariance_speed_speed_);
    covariance_bias_bias_ = std::max(1.0e-9, covariance_bias_bias_);
    const double limit = std::sqrt(
      covariance_speed_speed_ * covariance_bias_bias_);
    covariance_speed_bias_ = std::clamp(
      covariance_speed_bias_, -limit, limit);
  }

  double acceleration_noise_mps2_{1.0};
  double bias_random_walk_mps3_{0.1};
  double initial_speed_variance_{1.0};
  double initial_bias_variance_{0.25};
  double maximum_speed_mps_{30.0};

  double speed_mps_{0.0};
  double bias_mps2_{0.0};
  double covariance_speed_speed_{1.0};
  double covariance_speed_bias_{0.0};
  double covariance_bias_bias_{0.25};
};

}  // namespace f1tenth_localization
