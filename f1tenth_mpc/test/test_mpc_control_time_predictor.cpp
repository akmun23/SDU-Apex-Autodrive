#include "mpc_control_time_predictor.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>

namespace {

constexpr double kPi = 3.14159265358979323846;

std::array<MpcTrajectorySample_t, 4> square_trajectory(double *lap_length)
{
  std::array<MpcTrajectorySample_t, 4> points{{
      {0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.5, 0.5, 0.0},
      {1.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.5, 0.5, 0.0},
      {2.0, 1.0, 1.0, 0.5 * kPi, 0.0, 2.0, 0.5, 0.5, 0.0},
      {3.0, 0.0, 1.0, kPi, 0.0, 2.0, 0.5, 0.5, 0.0},
  }};
  std::size_t count = points.size();
  EXPECT_TRUE(mpc_trajectory_prepare(points.data(), &count, lap_length));
  EXPECT_EQ(count, points.size());
  return points;
}

TEST(MpcControlTimePredictor, HandlesFractionalMillisecondCommandAndSensorTimes)
{
  constexpr int64_t source_stamp_ns = 1000000000;
  constexpr int64_t target_stamp_ns = source_stamp_ns + 25200000;
  double lap_length = 0.0;
  const auto trajectory = square_trajectory(&lap_length);

  f1tenth_mpc::MpcSynchronizedState source{};
  source.source_stamp_ns = source_stamp_ns;
  source.map_x = 0.0;
  source.map_y = 0.0;
  source.map_yaw = 0.0;
  source.u = 2.0;

  f1tenth_mpc::MpcCommandHistory history;
  ASSERT_TRUE(history.push({source_stamp_ns + 8400000, 0.0, 2.0}));
  ASSERT_TRUE(history.push({source_stamp_ns + 18600000, 0.0, 1.9}));

  MpcPathProjection_t source_projection{};
  source_projection.s = 0.0;
  source_projection.segment = 0;
  f1tenth_mpc::MpcControlTimePredictorConfig config;
  f1tenth_mpc::MpcControlTimePrediction prediction{};
  const auto status = f1tenth_mpc::predict_to_control_time(
      f1tenth_mpc::MpcControlTimeMode::kAcceptedModelCommandHistory,
      source, target_stamp_ns, history, trajectory.data(), trajectory.size(),
      lap_length, 0, &source_projection, nullptr, 0, 0.0, config,
      &prediction);

  ASSERT_EQ(status, f1tenth_mpc::MpcControlTimeStatus::kOk);
  EXPECT_EQ(prediction.command_changes_used, 2U);
  EXPECT_NEAR(prediction.age_s, 0.0252, 1.0e-12);
  EXPECT_NEAR(prediction.target_speed_mps, 1.9, 1.0e-6);
  EXPECT_TRUE(std::isfinite(prediction.state.u));
  EXPECT_TRUE(std::isfinite(prediction.state.map_x));
  EXPECT_TRUE(std::isfinite(prediction.state.map_y));
}

TEST(MpcControlTimePredictor, PreservesCarriedSteeringStateAcrossUpdates)
{
  constexpr int64_t source_stamp_ns = 2000000000;
  constexpr int64_t target_stamp_ns = source_stamp_ns + 25000000;
  double lap_length = 0.0;
  const auto trajectory = square_trajectory(&lap_length);

  f1tenth_mpc::MpcSynchronizedState source{};
  source.source_stamp_ns = source_stamp_ns;
  source.map_x = 0.0;
  source.map_y = 0.0;
  source.map_yaw = -0.7;
  source.u = 5.0;
  source.yaw_rate = -0.7;

  f1tenth_mpc::MpcCommandHistory history;
  ASSERT_TRUE(history.push({source_stamp_ns, -0.35, 5.0}));
  MpcPathProjection_t source_projection{};
  source_projection.s = 0.0;
  source_projection.heading_error = -0.7;
  source_projection.segment = 0;

  MpcModelState_t carried{};
  carried.actual_steering_angle = -0.35f;
  carried.steering_command = -0.35f;
  carried.delayed_steering_command_1 = -0.35f;
  carried.delayed_steering_command_2 = -0.35f;
  carried.source_throttle_command = 0.30f;
  carried.source_last_target_speed = 5.0f;
  carried.source_target_initialized = 1;
  carried.source_actuator_speed_mps = 5.0f;
  carried.source_actuator_speed_initialized = 1;

  f1tenth_mpc::MpcControlTimePredictorConfig config;
  f1tenth_mpc::MpcControlTimePrediction cold_prediction{};
  f1tenth_mpc::MpcControlTimePrediction carried_prediction{};
  const auto cold_status = f1tenth_mpc::predict_to_control_time(
      f1tenth_mpc::MpcControlTimeMode::kAcceptedModelCommandHistory,
      source, target_stamp_ns, history, trajectory.data(), trajectory.size(),
      lap_length, 0, &source_projection, nullptr, 0, 0.0, config,
      &cold_prediction);
  const auto carried_status = f1tenth_mpc::predict_to_control_time(
      f1tenth_mpc::MpcControlTimeMode::kAcceptedModelCommandHistory,
      source, target_stamp_ns, history, trajectory.data(), trajectory.size(),
      lap_length, 0, &source_projection, &carried, source_stamp_ns, 0.0, config,
      &carried_prediction);

  ASSERT_EQ(cold_status, f1tenth_mpc::MpcControlTimeStatus::kOk);
  ASSERT_EQ(carried_status, f1tenth_mpc::MpcControlTimeStatus::kOk);
  EXPECT_GT(std::abs(cold_prediction.state.yaw_rate -
                     carried_prediction.state.yaw_rate), 0.1);
}

TEST(MpcControlTimePredictor, ReturnsLatentPlantAtControlTime)
{
  constexpr int64_t source_stamp_ns = 3000000000;
  constexpr int64_t target_stamp_ns = source_stamp_ns + 29000000;
  constexpr double wheel_radius_m = 0.059;
  double lap_length = 0.0;
  const auto trajectory = square_trajectory(&lap_length);

  f1tenth_mpc::MpcSynchronizedState source{};
  source.source_stamp_ns = source_stamp_ns;
  source.map_x = 0.0;
  source.map_y = 0.0;
  source.map_yaw = 0.0;
  source.u = 2.0;

  f1tenth_mpc::MpcCommandHistory history;
  ASSERT_TRUE(history.push({source_stamp_ns - 25000000, 0.0, 2.0}));
  ASSERT_TRUE(history.push({source_stamp_ns, 0.0, 2.0}));
  MpcPathProjection_t source_projection{};
  source_projection.s = 0.0;
  source_projection.segment = 0;

  MpcModelState_t source_plant{};
  source_plant.u = 2.0f;
  source_plant.target_speed = 2.0f;
  source_plant.delayed_throttle_command_1 = 0.25f;
  source_plant.delayed_throttle_command_2 = 0.25f;
  source_plant.source_throttle_command = 0.25f;
  source_plant.source_target_initialized = 1;
  source_plant.source_last_target_speed = 2.0f;
  source_plant.source_actuator_speed_mps = 2.0f;
  source_plant.source_actuator_speed_initialized = 1;
  source_plant.source_rear_wheel_omega_feedback_radps[0] =
      2.0 / wheel_radius_m;
  source_plant.source_rear_wheel_omega_feedback_radps[1] =
      2.0 / wheel_radius_m;
  source_plant.source_rear_wheel_omega_feedback_valid = 1;

  f1tenth_mpc::MpcControlTimePredictorConfig config;
  f1tenth_mpc::MpcControlTimePrediction prediction{};
  const auto status = f1tenth_mpc::predict_to_control_time(
      f1tenth_mpc::MpcControlTimeMode::kAcceptedModelCommandHistory,
      source, target_stamp_ns, history, trajectory.data(), trajectory.size(),
      lap_length, 0, &source_projection, &source_plant, source_stamp_ns, 0.0,
      config, &prediction);

  ASSERT_EQ(status, f1tenth_mpc::MpcControlTimeStatus::kOk);
  ASSERT_TRUE(prediction.plant_state_valid);
  EXPECT_TRUE(prediction.plant_state.source_plant_initialized);
  EXPECT_NEAR(prediction.plant_state.source_plant_state[UNITY_SOURCE_U],
              prediction.state.u, 1.0e-5);
  EXPECT_GT(prediction.plant_state.source_plant_state[
                UNITY_SOURCE_WHEEL_OMEGA_RL],
            2.0 / wheel_radius_m - 1.0);
  EXPECT_NEAR(prediction.age_s, 0.029, 1.0e-12);
}

TEST(MpcControlTimePredictor, PropagatesThenFusesAtObservedSensorEpoch)
{
  constexpr int64_t plant_stamp_ns = 4000000000;
  constexpr int64_t sensor_stamp_ns = plant_stamp_ns + 15000000;
  constexpr int64_t control_stamp_ns = sensor_stamp_ns + 10000000;
  double lap_length = 0.0;
  const auto trajectory = square_trajectory(&lap_length);

  f1tenth_mpc::MpcSynchronizedState observed{};
  observed.source_stamp_ns = sensor_stamp_ns;
  observed.map_x = 0.0;
  observed.map_y = 0.0;
  observed.map_yaw = 0.0;
  observed.u = 3.0;
  observed.v = 0.2;
  observed.yaw_rate = 0.4;

  f1tenth_mpc::MpcCommandHistory history;
  ASSERT_TRUE(history.push({plant_stamp_ns, 0.0, 2.0}));
  MpcPathProjection_t observed_projection{};
  observed_projection.s = 0.02;
  observed_projection.lateral_error = 0.01;
  observed_projection.heading_error = -0.02;
  observed_projection.segment = 0;

  MpcModelState_t carried{};
  carried.u = 2.0f;
  carried.target_speed = 2.0f;
  carried.source_throttle_command = 0.25f;
  carried.source_target_initialized = 1;
  carried.source_last_target_speed = 2.0f;
  carried.source_actuator_speed_mps = 2.0f;
  carried.source_actuator_speed_initialized = 1;
  carried.source_plant_initialized = 1;
  carried.source_suspension_history_initialized = 1;
  carried.source_plant_state[UNITY_SOURCE_U] = 2.0;
  carried.source_plant_state[UNITY_SOURCE_BODY_HEIGHT] = 0.00464804098;
  for (int wheel = 0; wheel < 4; ++wheel)
    carried.source_plant_state[UNITY_SOURCE_WHEEL_OMEGA_FL + wheel] =
        2.0 / 0.059;
  carried.source_rear_wheel_omega_feedback_radps[0] = 50.0f;
  carried.source_rear_wheel_omega_feedback_radps[1] = 51.0f;
  carried.source_rear_wheel_omega_feedback_valid = 1;

  f1tenth_mpc::MpcControlTimePredictorConfig config;
  f1tenth_mpc::MpcControlTimePrediction prediction{};
  const auto status = f1tenth_mpc::predict_to_control_time(
      f1tenth_mpc::MpcControlTimeMode::kAcceptedModelCommandHistory,
      observed, control_stamp_ns, history, trajectory.data(), trajectory.size(),
      lap_length, 0, &observed_projection, &carried, plant_stamp_ns, 0.0,
      config, &prediction);

  ASSERT_EQ(status, f1tenth_mpc::MpcControlTimeStatus::kOk);
  ASSERT_TRUE(prediction.plant_state_valid);
  EXPECT_TRUE(prediction.sensor_state_fused);
  EXPECT_EQ(prediction.plant_source_stamp_ns, plant_stamp_ns);
  EXPECT_NEAR(prediction.plant_to_sensor_s, 0.015, 1.0e-12);
  ASSERT_TRUE(prediction.plant_handoff_state_valid);
  EXPECT_EQ(prediction.plant_handoff_stamp_ns, sensor_stamp_ns);
  EXPECT_NEAR(prediction.plant_handoff_progress_m, 0.02, 1.0e-6);
  EXPECT_NEAR(prediction.plant_handoff_state.u, observed.u, 1.0e-6);
  EXPECT_TRUE(std::isfinite(prediction.plant_state.u));
  EXPECT_NEAR(prediction.plant_handoff_state.v, observed.v, 1.0e-6);
  EXPECT_NEAR(prediction.plant_handoff_state.r, observed.yaw_rate, 1.0e-6);
  EXPECT_NEAR(prediction.plant_handoff_state.e_y,
              observed_projection.lateral_error, 1.0e-6);
  EXPECT_NEAR(prediction.plant_handoff_state.e_psi,
              observed_projection.heading_error, 1.0e-6);
  EXPECT_NEAR(prediction.plant_handoff_state.source_plant_state[
                  UNITY_SOURCE_WHEEL_OMEGA_RL], 50.0, 1.0e-6);
  EXPECT_NEAR(prediction.plant_handoff_state.source_plant_state[
                  UNITY_SOURCE_WHEEL_OMEGA_RR], 51.0, 1.0e-6);

  /* The next sensor sample should be newer than the carried sensor-epoch
   * snapshot, even though it is older than the preceding control-time state. */
  auto next_observed = observed;
  next_observed.source_stamp_ns = sensor_stamp_ns + 25000000;
  next_observed.u = 3.1;
  MpcPathProjection_t next_projection = observed_projection;
  next_projection.s = 0.08;
  f1tenth_mpc::MpcControlTimePrediction next_prediction{};
  const auto next_status = f1tenth_mpc::predict_to_control_time(
      f1tenth_mpc::MpcControlTimeMode::kAcceptedModelCommandHistory,
      next_observed, sensor_stamp_ns + 35000000, history,
      trajectory.data(), trajectory.size(), lap_length, 0, &next_projection,
      &prediction.plant_handoff_state, prediction.plant_handoff_stamp_ns,
      prediction.plant_handoff_progress_m, config, &next_prediction);

  ASSERT_EQ(next_status, f1tenth_mpc::MpcControlTimeStatus::kOk);
  EXPECT_TRUE(next_prediction.sensor_state_fused);
  EXPECT_NEAR(next_prediction.plant_to_sensor_s, 0.025, 1.0e-12);
  EXPECT_EQ(next_prediction.plant_handoff_stamp_ns,
      next_observed.source_stamp_ns);
  EXPECT_NEAR(next_prediction.plant_handoff_state.u, next_observed.u, 1.0e-6);
}

TEST(MpcControlTimePredictor, DoesNotFuseObservationOlderThanCarriedPlant)
{
  constexpr int64_t sensor_stamp_ns = 5000000000;
  constexpr int64_t plant_stamp_ns = sensor_stamp_ns + 5000000;
  constexpr int64_t target_stamp_ns = plant_stamp_ns + 10000000;
  double lap_length = 0.0;
  const auto trajectory = square_trajectory(&lap_length);

  f1tenth_mpc::MpcSynchronizedState observed{};
  observed.source_stamp_ns = sensor_stamp_ns;
  observed.map_x = 0.0;
  observed.map_y = 0.0;
  observed.map_yaw = 0.0;
  observed.u = 0.5;

  f1tenth_mpc::MpcCommandHistory history;
  ASSERT_TRUE(history.push({plant_stamp_ns, 0.0, 2.0}));
  MpcPathProjection_t stale_projection{};
  stale_projection.s = 0.0;
  stale_projection.segment = 0;

  MpcModelState_t carried{};
  carried.u = 2.0f;
  carried.target_speed = 2.0f;
  carried.source_rear_wheel_omega_feedback_radps[0] = 100.0f;
  carried.source_rear_wheel_omega_feedback_radps[1] = 100.0f;
  carried.source_rear_wheel_omega_feedback_valid = 1;

  f1tenth_mpc::MpcControlTimePredictorConfig config;
  f1tenth_mpc::MpcControlTimePrediction prediction{};
  const auto status = f1tenth_mpc::predict_to_control_time(
      f1tenth_mpc::MpcControlTimeMode::kAcceptedModelCommandHistory,
      observed, target_stamp_ns, history, trajectory.data(), trajectory.size(),
      lap_length, 0, &stale_projection, &carried, plant_stamp_ns, 0.0,
      config, &prediction);

  ASSERT_EQ(status, f1tenth_mpc::MpcControlTimeStatus::kOk);
  ASSERT_TRUE(prediction.plant_state_valid);
  EXPECT_FALSE(prediction.sensor_state_fused);
  ASSERT_TRUE(prediction.plant_handoff_state_valid);
  EXPECT_EQ(prediction.plant_handoff_stamp_ns, plant_stamp_ns);
  EXPECT_NEAR(prediction.plant_handoff_progress_m, 0.0, 1.0e-6);
  EXPECT_GT(prediction.plant_state.u, 1.5);
  EXPECT_GT(std::abs(prediction.plant_state.u - observed.u), 1.0);
  EXPECT_NE(prediction.plant_state.source_plant_state[
                UNITY_SOURCE_WHEEL_OMEGA_RL], 100.0);
}

}  // namespace
