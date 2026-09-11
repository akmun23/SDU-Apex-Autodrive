#include "f1tenth_mpc/track_model.hpp"

#include <gtest/gtest.h>

#include <fstream>

using namespace f1tenth_mpc;

TEST(TrackModel, LoadsAndProjectsClosedPolyline)
{
  const std::string path = "/tmp/f1tenth_mpc_test_track.csv";
  std::ofstream stream(path);
  stream << "# s,x,y,yaw,kappa,vx\n"
         << "0,0,0,0,0,4\n"
         << "1,1,0,0,0,4\n"
         << "2,1,1,1.57079632679,0,4\n"
         << "3,0,1,3.14159265359,0,4\n";
  stream.close();

  TrackModel track;
  std::string error;
  ASSERT_TRUE(track.load_csv(path, &error)) << error;
  EXPECT_EQ(track.size(), 4U);
  EXPECT_GT(track.length(), 3.0);
  const auto projection = track.project(0.5, 0.2, 0.0);
  ASSERT_TRUE(projection.valid);
  EXPECT_NEAR(projection.s, 0.5, 0.1);
  EXPECT_NEAR(projection.lateral_error, 0.2, 0.05);
  EXPECT_NEAR(track.sample(1.5).x, 1.0, 1e-9);
}
