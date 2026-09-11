#include "f1tenth_mpc/vehicle_model.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>

extern "C" {
#include "vehicle_model.h"
}

using namespace f1tenth_mpc;

TEST(VehicleModel, StationaryZeroInputIsDeterministic)
{
  VehicleModel model;
  const ModelState state{};
  const auto first = model.step(state, ModelInput{}, 0.05);
  const auto second = model.step(state, ModelInput{}, 0.05);
  EXPECT_DOUBLE_EQ(first.x, second.x);
  EXPECT_DOUBLE_EQ(first.y, second.y);
  EXPECT_DOUBLE_EQ(first.u, model.config().min_speed_mps);
  EXPECT_DOUBLE_EQ(first.steering, 0.0);
}

TEST(VehicleModel, SteeringAndSpeedAreBounded)
{
  VehicleModel model;
  ModelState state{};
  for (int i = 0; i < 200; ++i) {
    state = model.step(state, ModelInput{100.0, 100.0}, 0.005);
    EXPECT_GE(state.u, model.config().min_speed_mps);
    EXPECT_LE(state.u, model.config().max_speed_mps);
    EXPECT_LE(std::abs(state.steering), model.config().max_steering_rad + 1e-12);
  }
  EXPECT_GT(state.u, 1.0);
  EXPECT_GT(state.steering, 0.0);
  EXPECT_TRUE(state.steering_valid);
}

TEST(VehicleModel, DirectInputsProduceDynamicLateralState)
{
  VehicleModel model;
  ModelState state;
  state.u = 8.0;
  for (int i = 0; i < 40; ++i) {
    state = model.step(state, ModelInput{0.12, 0.0}, 0.01);
  }
  EXPECT_GT(std::abs(state.v), 1e-4);
  EXPECT_GT(std::abs(state.r), 1e-4);
  EXPECT_TRUE(std::isfinite(state.x));
  EXPECT_TRUE(std::isfinite(state.y));
}

TEST(VehicleModel, BaselineMatchesBachelorProjectPlant)
{
  VehicleModel model;
  ModelState state;
  state.x = 1.2;
  state.y = -0.7;
  state.yaw = 0.4;
  state.u = 6.0;
  state.v = 0.15;
  state.r = 0.8;
  const ModelInput input{0.12, 1.7};
  constexpr double dt = 0.01;

  const auto actual = model.step(state, input, dt);
  VehicleState_t expected_state{
    static_cast<float>(state.x), static_cast<float>(state.y),
    static_cast<float>(state.yaw), static_cast<float>(state.u),
    static_cast<float>(state.v), static_cast<float>(state.r)};
  const ControlInput_t expected_input{
    static_cast<float>(input.steering_rad),
    static_cast<float>(input.acceleration_mps2)};
  const auto expected = vehicle_model_predict_next_state(
    &expected_state, &expected_input, static_cast<float>(dt));

  EXPECT_NEAR(actual.x, expected.pos_x, 1e-5);
  EXPECT_NEAR(actual.y, expected.pos_y, 1e-5);
  EXPECT_NEAR(actual.yaw, expected.heading, 1e-5);
  EXPECT_NEAR(actual.u, expected.long_vel, 1e-5);
  EXPECT_NEAR(actual.v, expected.lat_vel, 1e-5);
  EXPECT_NEAR(actual.r, expected.yaw_rate, 1e-5);
  EXPECT_NEAR(actual.steering, input.steering_rad, 1e-12);
  EXPECT_TRUE(actual.steering_valid);
}

TEST(VehicleModel, RolloutRequiresMatchingTimeVector)
{
  VehicleModel model;
  EXPECT_THROW(model.rollout(ModelState{}, {{1.0, 0.0}}, {}), std::invalid_argument);
}
