#include "f1tenth_mpc/bachelor_mpc_controller.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <fstream>

using namespace f1tenth_mpc;

TEST(BachelorMpcController, ProducesFiniteAccelerationAndSteering)
{
  const std::string path = "/tmp/f1tenth_bachelor_mpc_circle.csv";
  std::ofstream stream(path);
  ASSERT_TRUE(stream.good());
  constexpr int kPoints = 100;
  constexpr double kRadius = 5.0;
  constexpr double kPi = 3.14159265358979323846;
  for (int i = 0; i < kPoints; ++i) {
    const double theta = 2.0 * kPi * static_cast<double>(i) / kPoints;
    const double s = kRadius * theta;
    stream << s << ',' << kRadius * std::cos(theta) << ',' << kRadius * std::sin(theta)
           << ',' << theta + kPi / 2.0 << ',' << 1.0 / kRadius << ",3.0,0.0,2.0,2.0\n";
  }
  stream.close();

  TrackModel track;
  ASSERT_TRUE(track.load_csv(path));
  BachelorMpcConfig config;
  config.max_solver_iterations = 50;
  BachelorMpcController controller(track, config);

  ModelState state;
  state.x = kRadius;
  state.y = 0.0;
  state.yaw = kPi / 2.0;
  state.u = 2.0;
  state.r = state.u / kRadius;
  const auto result = controller.solve(state);

  EXPECT_TRUE(result.valid) << result.status_text;
  EXPECT_TRUE(std::isfinite(result.steering_rad));
  EXPECT_TRUE(std::isfinite(result.acceleration_mps2));
  EXPECT_TRUE(std::isfinite(result.predicted_lateral_error_m));
  EXPECT_LE(std::abs(result.steering_rad), 0.39 + 1e-4);
  EXPECT_LE(result.acceleration_mps2, 0.72 * 9.82 + 1e-4);
  EXPECT_GE(result.acceleration_mps2, -0.72 * 9.82 - 1e-4);
  EXPECT_EQ(result.horizon_steps, 20U);
}
