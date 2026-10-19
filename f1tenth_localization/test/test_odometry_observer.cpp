#include "f1tenth_localization/odometry_observer.hpp"

#include <cmath>
#include <stdexcept>
#include <gtest/gtest.h>

namespace f1tenth_localization
{
namespace
{

OdometryObservation plant_observation(double stamp_s)
{
  OdometryObservation observation;
  observation.stamp_s = stamp_s;
  observation.physx_forward_velocity_mps = 2.0;
  observation.physx_state_valid = true;
  observation.physx_pose_valid = true;
  return observation;
}

TEST(OdometryObserver, AppliesSameEpochAccelerationInnovationForOnePhysicsTick)
{
  OdometryObserver observer;
  auto observation = plant_observation(0.0);
  (void)observer.update(observation);

  observation = plant_observation(0.025);
  observation.ax_mps2 = 1.0;
  auto estimate = observer.update(observation);
  EXPECT_NEAR(estimate.body_u_mps, 2.001, 1.0e-12);

  observation = plant_observation(0.050);
  observation.physx_forward_velocity_mps = estimate.body_u_mps;
  observation.ax_mps2 = 1.0;
  estimate = observer.update(observation);
  EXPECT_NEAR(estimate.body_u_mps, 2.002, 1.0e-12);
}

TEST(OdometryObserver, PreservesSuspensionJouncePlantChannels)
{
  OdometryObserver observer;
  auto observation = plant_observation(0.0);
  observation.physx_state[14] = 0.001;
  observation.physx_state[15] = 0.002;
  observation.physx_state[16] = 0.003;
  observation.physx_state[17] = 0.004;
  auto estimate = observer.update(observation);
  for (std::size_t state = 14; state < observation.physx_state.size(); ++state) {
    EXPECT_DOUBLE_EQ(estimate.unity_state[state], observation.physx_state[state])
        << "initial plant state index " << state;
  }

  observation = plant_observation(0.025);
  observation.physx_forward_velocity_mps = estimate.body_u_mps;
  observation.physx_state[14] = 0.005;
  observation.physx_state[15] = 0.006;
  observation.physx_state[16] = 0.007;
  observation.physx_state[17] = 0.008;
  estimate = observer.update(observation);
  for (std::size_t state = 14; state < observation.physx_state.size(); ++state) {
    EXPECT_DOUBLE_EQ(estimate.unity_state[state], observation.physx_state[state])
        << "corrected plant state index " << state;
  }
}

TEST(OdometryObserver, DoesNotStretchOneMillisecondAccelerationSpikeAcrossPacket)
{
  OdometryObserver observer;
  auto observation = plant_observation(0.0);
  observation.physx_forward_velocity_mps = 1.0;
  (void)observer.update(observation);

  observation = plant_observation(0.025);
  observation.physx_forward_velocity_mps = 1.0;
  observation.ax_mps2 = -80.0;
  auto estimate = observer.update(observation);
  EXPECT_NEAR(estimate.body_u_mps, 0.92, 1.0e-12);

  observation = plant_observation(0.050);
  observation.physx_forward_velocity_mps = estimate.body_u_mps;
  observation.ax_mps2 = 0.0;
  estimate = observer.update(observation);
  EXPECT_NEAR(estimate.body_u_mps, 0.92, 1.0e-12);
}

TEST(OdometryObserver, ConvertsYawRateInnovationIntoBodyLateralVelocity)
{
  OdometryObserver observer;
  auto observation = plant_observation(0.0);
  (void)observer.update(observation);

  observation = plant_observation(0.025);
  observation.yaw_rate_radps = 1.0;
  observation.yaw_rad = 0.0125;
  const auto estimate = observer.update(observation);

  // The gyro/model yaw-rate residual applies at the same 1 ms endpoint as
  // the acceleration samples, rather than across the 25 ms packet interval.
  EXPECT_NEAR(estimate.body_v_com_mps, -0.002, 1.0e-12);
  EXPECT_NEAR(estimate.body_v_mps,
      -0.002 - 0.15532, 1.0e-12);
  EXPECT_NEAR(estimate.yaw_rad, 0.0125, 1.0e-12);
}

TEST(OdometryObserver, UsesMeasuredImuAttitudeForPoseInsteadOfIntegratingGyro)
{
  OdometryObserver observer;
  auto observation = plant_observation(0.0);
  observation.yaw_rad = 0.20;
  (void)observer.update(observation);

  observation = plant_observation(0.025);
  observation.yaw_rad = 0.26;
  observation.yaw_rate_radps = 0.0;
  const auto estimate = observer.update(observation);

  EXPECT_DOUBLE_EQ(estimate.yaw_rad, 0.26);
  EXPECT_NEAR(estimate.x_m, 2.0 * 0.025 * std::cos(0.23), 1.0e-12);
  EXPECT_NEAR(estimate.y_m, 2.0 * 0.025 * std::sin(0.23), 1.0e-12);
  EXPECT_DOUBLE_EQ(estimate.yaw_rate_radps, 0.0);
}

TEST(OdometryObserver, LateralAccelerationGainScalesOnlyAccelerationResidual)
{
  OdometryObserverConfig config;
  config.lateral_acceleration_residual_gain = 0.0;
  OdometryObserver observer(config);
  auto observation = plant_observation(0.0);
  (void)observer.update(observation);

  observation = plant_observation(0.025);
  observation.ay_mps2 = 2.0;
  auto estimate = observer.update(observation);
  EXPECT_DOUBLE_EQ(estimate.body_v_com_mps, 0.0);

  observation = plant_observation(0.050);
  observation.physx_forward_velocity_mps = 2.0;
  observation.ay_mps2 = 0.0;
  observation.yaw_rate_radps = 1.0;
  estimate = observer.update(observation);
  EXPECT_NEAR(estimate.body_v_com_mps, -0.002, 1.0e-12);
}

TEST(OdometryObserver, RejectsLateralAccelerationGainOutsideUnitInterval)
{
  OdometryObserverConfig config;
  config.lateral_acceleration_residual_gain = 1.01;
  EXPECT_THROW(OdometryObserver observer(config), std::invalid_argument);
}

TEST(OdometryObserver, FadesLateralAccelerationInnovationAboveMeasuredSteeringRegion)
{
  OdometryObserver observer;
  auto observation = plant_observation(0.0);
  observation.actuator_feedback_valid = true;
  observation.steering_feedback_rad = 0.46;
  observation.throttle_feedback = 0.5;
  observation.physx_wheel_omega_radps[2] = 4.2 / 0.059;
  observation.physx_wheel_omega_radps[3] = 4.2 / 0.059;
  (void)observer.update(observation);

  observation = plant_observation(0.025);
  observation.actuator_feedback_valid = true;
  observation.steering_feedback_rad = 0.46;
  observation.throttle_feedback = 0.5;
  observation.physx_wheel_omega_radps[2] = 4.2 / 0.059;
  observation.physx_wheel_omega_radps[3] = 4.2 / 0.059;
  observation.left_angle_rad = (4.2 / 0.059) * 0.025;
  observation.right_angle_rad = (4.2 / 0.059) * 0.025;
  observation.ay_mps2 = 2.0;
  const auto estimate = observer.update(observation);
  const double steering_limit = 30.0 * 3.14159265358979323846 / 180.0;
  const double expected_gain = (steering_limit - 0.46) /
    (steering_limit - 0.4);
  EXPECT_NEAR(estimate.lateral_acceleration_residual_gain, expected_gain, 1.0e-12);
  EXPECT_NEAR(estimate.body_v_com_mps,
    expected_gain * 2.0 * 0.001, 1.0e-12);
}

TEST(OdometryObserver, KeepsFullInnovationWithoutHighThrottleWheelSpin)
{
  OdometryObserver observer;
  auto observation = plant_observation(0.0);
  observation.actuator_feedback_valid = true;
  observation.steering_feedback_rad = 0.46;
  observation.throttle_feedback = 0.2;
  observation.physx_wheel_omega_radps[2] = 4.2 / 0.059;
  observation.physx_wheel_omega_radps[3] = 4.2 / 0.059;
  (void)observer.update(observation);

  observation = plant_observation(0.025);
  observation.actuator_feedback_valid = true;
  observation.steering_feedback_rad = 0.46;
  observation.throttle_feedback = 0.2;
  observation.physx_wheel_omega_radps[2] = 4.2 / 0.059;
  observation.physx_wheel_omega_radps[3] = 4.2 / 0.059;
  observation.ay_mps2 = 2.0;
  const auto estimate = observer.update(observation);
  EXPECT_DOUBLE_EQ(estimate.lateral_acceleration_residual_gain, 1.0);
  EXPECT_NEAR(estimate.body_v_com_mps, 2.0 * 0.001, 1.0e-12);
}

TEST(OdometryObserver, IntegratesCorrectedVelocityAndExposesPhysxPoseResidual)
{
  OdometryObserver observer;
  auto observation = plant_observation(0.0);
  observation.physx_rear_axle_x_m = 1.25;
  observation.physx_rear_axle_y_m = -0.4;
  auto estimate = observer.update(observation);
  EXPECT_DOUBLE_EQ(estimate.x_m, 0.0);
  EXPECT_DOUBLE_EQ(estimate.y_m, 0.0);

  observation = plant_observation(0.025);
  observation.physx_rear_axle_x_m = 1.31;
  observation.physx_rear_axle_y_m = -0.37;
  estimate = observer.update(observation);
  EXPECT_NEAR(estimate.x_m, 0.05, 1.0e-12);
  EXPECT_NEAR(estimate.y_m, 0.0, 1.0e-12);
  EXPECT_NEAR(estimate.physx_model_x_m, 0.06, 1.0e-12);
  EXPECT_NEAR(estimate.physx_model_y_m, 0.03, 1.0e-12);
}

TEST(OdometryObserver, CorrectsRearWheelSpinFromIntegratedEncoderResidual)
{
  OdometryObserver observer;
  auto observation = plant_observation(0.0);
  observation.physx_wheel_omega_radps[2] = 10.0;
  observation.physx_wheel_omega_radps[3] = 10.0;
  (void)observer.update(observation);

  observation = plant_observation(0.025);
  observation.physx_wheel_omega_radps[2] = 12.0;
  observation.physx_wheel_omega_radps[3] = 12.0;
  observation.physx_wheel_rotation_angle_rad[2] = 0.275;
  observation.physx_wheel_rotation_angle_rad[3] = 0.275;
  observation.physx_wheel_encoder_angle_prediction_rad[2] = 0.275;
  observation.physx_wheel_encoder_angle_prediction_rad[3] = 0.275;
  observation.left_angle_rad = 0.30;
  observation.right_angle_rad = 0.30;

  const auto estimate = observer.update(observation);
  constexpr double encoder_quantum_rad = 2.0 * 3.14159265358979323846 / 1920.0;
  const double expected_correction = 2.0 * (0.025 - encoder_quantum_rad) / 0.025;
  EXPECT_NEAR(estimate.encoder_model_rotation_residual_rad[0], 0.025, 1.0e-12);
  EXPECT_NEAR(estimate.wheel_omega_correction_radps[0], expected_correction, 1.0e-10);
  EXPECT_NEAR(estimate.wheel_omega_radps[2], 12.0 + expected_correction, 1.0e-10);
  EXPECT_NEAR(estimate.unity_state[5], estimate.wheel_omega_radps[2], 1.0e-12);
  EXPECT_NEAR(estimate.encoder_rear_axle_speed_mps, 0.059 * 12.0, 1.0e-12);
}

TEST(OdometryObserver, DoesNotCorrectSpinInsideEncoderQuantization)
{
  OdometryObserver observer;
  auto observation = plant_observation(0.0);
  observation.physx_wheel_omega_radps[2] = 10.0;
  observation.physx_wheel_omega_radps[3] = 10.0;
  (void)observer.update(observation);

  observation = plant_observation(0.025);
  observation.physx_wheel_omega_radps[2] = 10.0;
  observation.physx_wheel_omega_radps[3] = 10.0;
  observation.physx_wheel_rotation_angle_rad[2] = 0.025;
  observation.physx_wheel_rotation_angle_rad[3] = 0.025;
  observation.physx_wheel_encoder_angle_prediction_rad[2] = 0.025;
  observation.physx_wheel_encoder_angle_prediction_rad[3] = 0.025;
  observation.left_angle_rad = 0.025 + 0.5 *
    (2.0 * 3.14159265358979323846 / 1920.0);
  observation.right_angle_rad = observation.left_angle_rad;
  const auto estimate = observer.update(observation);

  EXPECT_DOUBLE_EQ(estimate.wheel_omega_correction_radps[0], 0.0);
  EXPECT_DOUBLE_EQ(estimate.wheel_omega_correction_radps[1], 0.0);
}

TEST(OdometryObserver, UsesUnwrappedEncoderPredictionAcrossPhysxAngleBoundary)
{
  OdometryObserver observer;
  auto observation = plant_observation(0.0);
  observation.physx_wheel_omega_radps[2] = 8.0;
  observation.physx_wheel_omega_radps[3] = 8.0;
  observation.physx_wheel_rotation_angle_rad[2] =
    10.0 * 3.14159265358979323846 - 0.1;
  observation.physx_wheel_rotation_angle_rad[3] =
    observation.physx_wheel_rotation_angle_rad[2];
  observation.physx_wheel_encoder_angle_prediction_rad[2] =
    observation.physx_wheel_rotation_angle_rad[2];
  observation.physx_wheel_encoder_angle_prediction_rad[3] =
    observation.physx_wheel_encoder_angle_prediction_rad[2];
  (void)observer.update(observation);

  observation = plant_observation(0.025);
  observation.physx_wheel_omega_radps[2] = 8.0;
  observation.physx_wheel_omega_radps[3] = 8.0;
  observation.physx_wheel_rotation_angle_rad[2] = 0.1;
  observation.physx_wheel_rotation_angle_rad[3] = 0.1;
  observation.physx_wheel_encoder_angle_prediction_rad[2] = 10.0 *
    3.14159265358979323846 + 0.1;
  observation.physx_wheel_encoder_angle_prediction_rad[3] =
    observation.physx_wheel_encoder_angle_prediction_rad[2];
  observation.left_angle_rad = 0.2;
  observation.right_angle_rad = 0.2;
  const auto estimate = observer.update(observation);

  EXPECT_NEAR(estimate.encoder_model_rotation_residual_rad[0], 0.0, 1.0e-12);
  EXPECT_DOUBLE_EQ(estimate.wheel_omega_correction_radps[0], 0.0);
  EXPECT_DOUBLE_EQ(estimate.wheel_omega_radps[2], 8.0);
}

TEST(OdometryObserver, RemovesPreSolveEncoderSampleFromBrakeOnsetSpin)
{
  OdometryObserver observer;
  auto observation = plant_observation(0.0);
  observation.physx_wheel_omega_radps[2] = 84.0;
  observation.physx_wheel_omega_radps[3] = 84.0;
  (void)observer.update(observation);

  // The encoder samples 84 rad/s before the first braking physics solve,
  // then sees the wheel locked for the remaining 24 ms. PxVehicle's physical
  // wheel angle advances after torque integration and remains unchanged.
  observation = plant_observation(0.025);
  observation.physx_wheel_omega_radps[2] = 0.0;
  observation.physx_wheel_omega_radps[3] = 0.0;
  observation.physx_wheel_encoder_angle_prediction_rad[2] = 0.084;
  observation.physx_wheel_encoder_angle_prediction_rad[3] = 0.084;
  observation.left_angle_rad = 0.0818125;
  observation.right_angle_rad = 0.0818125;

  const auto estimate = observer.update(observation);
  EXPECT_NEAR(estimate.encoder_model_rotation_residual_rad[0], -0.0021875, 1.0e-12);
  EXPECT_DOUBLE_EQ(estimate.wheel_omega_correction_radps[0], 0.0);
  EXPECT_DOUBLE_EQ(estimate.wheel_omega_correction_radps[1], 0.0);
  EXPECT_DOUBLE_EQ(estimate.wheel_omega_radps[2], 0.0);
  EXPECT_DOUBLE_EQ(estimate.wheel_omega_radps[3], 0.0);
}

}  // namespace
}  // namespace f1tenth_localization
