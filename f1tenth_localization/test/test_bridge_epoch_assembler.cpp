#include "f1tenth_localization/bridge_epoch_assembler.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>

using f1tenth_localization::BridgeEpochAssembler;
using f1tenth_localization::SensorPacket;

TEST(BridgeEpochAssembler, EveryCallbackOrderProducesSameEpoch)
{
  std::array<int, 3> order{0, 1, 2};
  do {
    BridgeEpochAssembler join;
    SensorPacket input{};
    input.stamp_ns = 1'000'000'000;
    input.have_left = input.have_right = input.have_imu = true;
    SensorPacket packet{};
    BridgeEpochAssembler::Command command{}, feedback{};
    for (int i = 0; i < 3; ++i) {
      switch (order[i]) {
        case 0: join.push_packet(input); break;
        case 1: ASSERT_TRUE(join.push_command(input.stamp_ns, .4, .5)); break;
        case 2: ASSERT_TRUE(join.push_feedback(input.stamp_ns, .2, .3)); break;
      }
      EXPECT_EQ(join.pop_ready(&packet, &command, &feedback), i == 2);
    }
    EXPECT_EQ(packet.stamp_ns, input.stamp_ns);
    EXPECT_EQ(command.stamp_ns, packet.stamp_ns);
    EXPECT_EQ(feedback.stamp_ns, packet.stamp_ns);
    EXPECT_DOUBLE_EQ(command.steering_normalized, .4);
    EXPECT_DOUBLE_EQ(feedback.steering_normalized, .2);
    EXPECT_FALSE(join.pop_ready(&packet, &command, &feedback));
  } while (std::next_permutation(order.begin(), order.end()));
}

TEST(BridgeEpochAssembler, NewerFeedbackCannotRepairMissingOlderFeedback)
{
  BridgeEpochAssembler join;
  SensorPacket input{};
  input.stamp_ns = 1'000'000'000;
  input.have_left = input.have_right = input.have_imu = true;
  join.push_packet(input);
  ASSERT_TRUE(join.push_command(input.stamp_ns, .4, .5));
  ASSERT_TRUE(join.push_feedback(input.stamp_ns + 25'000'000, -.4, .6));
  SensorPacket packet{};
  BridgeEpochAssembler::Command command{}, feedback{};
  EXPECT_FALSE(join.pop_ready(&packet, &command, &feedback));
  ASSERT_TRUE(join.push_feedback(input.stamp_ns, .2, .3));
  ASSERT_TRUE(join.pop_ready(&packet, &command, &feedback));
  EXPECT_DOUBLE_EQ(feedback.steering_normalized, .2);
  EXPECT_THROW(join.push_packet(input), std::runtime_error);
}

TEST(BridgeEpochAssembler, BacklogExhaustionNeverDropsAnIntegrationInterval)
{
  BridgeEpochAssembler join;
  SensorPacket input{};
  input.have_left = input.have_right = input.have_imu = true;
  for (std::size_t i = 0; i < 64; ++i) {
    input.stamp_ns = 1'000'000'000 + i * 25'000'000;
    join.push_packet(input);
  }
  input.stamp_ns += 25'000'000;
  EXPECT_THROW(join.push_packet(input), std::runtime_error);
  EXPECT_FALSE(join.push_feedback(input.stamp_ns, .2, -.1));
}
