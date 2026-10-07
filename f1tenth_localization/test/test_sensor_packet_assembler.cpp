#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "f1tenth_localization/sensor_packet_assembler.hpp"

namespace f1tenth_localization
{
namespace
{

void add_complete_packet(SensorPacketAssembler & assembler, int64_t stamp_ns)
{
  assembler.add_encoder_sample(stamp_ns, 0.1, true);
  assembler.add_encoder_sample(stamp_ns, 0.2, false);
  assembler.add_imu_sample(stamp_ns, 0.0, 0.0, 0.0, 0.0);
}

TEST(SensorPacketAssembler, RecordsNormalSourceGapWithoutReordering)
{
  SensorPacketAssembler assembler;
  std::vector<int64_t> emitted;
  assembler.set_packet_callback([&emitted](const SensorPacket & packet) {
    emitted.push_back(packet.stamp_ns);
  });

  add_complete_packet(assembler, 1'000'000'000);
  add_complete_packet(assembler, 1'100'000'000);

  EXPECT_EQ(emitted, (std::vector<int64_t>{1'000'000'000, 1'100'000'000}));
  EXPECT_EQ(assembler.diagnostics().source_reversal_count, 0U);
  EXPECT_EQ(assembler.diagnostics().duplicate_source_count, 0U);
  EXPECT_EQ(assembler.diagnostics().late_completed_packet_count, 0U);
  EXPECT_EQ(assembler.diagnostics().maximum_reorder_depth, 0U);
}

TEST(SensorPacketAssembler, RecordsLateCompletionAfterBypassingIncompletePacket)
{
  SensorPacketAssembler assembler;
  std::vector<int64_t> emitted;
  assembler.set_packet_callback([&emitted](const SensorPacket & packet) {
    emitted.push_back(packet.stamp_ns);
  });

  constexpr int64_t earlier = 1'025'000'000;
  constexpr int64_t later = 1'050'000'000;
  assembler.add_encoder_sample(earlier, 0.1, true);
  add_complete_packet(assembler, later);

  EXPECT_EQ(emitted, (std::vector<int64_t>{later}));
  EXPECT_EQ(assembler.diagnostics().maximum_reorder_depth, 1U);
  EXPECT_EQ(assembler.diagnostics().maximum_reorder_time_ns, 25'000'000);

  assembler.add_encoder_sample(earlier, 0.2, false);
  assembler.add_imu_sample(earlier, 0.0, 0.0, 0.0, 0.0);

  EXPECT_EQ(emitted, (std::vector<int64_t>{later, earlier}));
  EXPECT_EQ(assembler.diagnostics().source_reversal_count, 1U);
  EXPECT_EQ(assembler.diagnostics().late_completed_packet_count, 1U);
  EXPECT_EQ(assembler.diagnostics().last_emitted_source_stamp_ns, earlier);
}

TEST(SensorPacketAssembler, CountsConsecutiveDuplicateCompletedSourceStamp)
{
  SensorPacketAssembler assembler;
  std::size_t emitted = 0;
  assembler.set_packet_callback([&emitted](const SensorPacket &) {++emitted;});

  add_complete_packet(assembler, 1'050'000'000);
  add_complete_packet(assembler, 1'050'000'000);

  EXPECT_EQ(emitted, 2U);
  EXPECT_EQ(assembler.diagnostics().duplicate_source_count, 1U);
  EXPECT_EQ(assembler.diagnostics().source_reversal_count, 0U);
}

}  // namespace
}  // namespace f1tenth_localization
