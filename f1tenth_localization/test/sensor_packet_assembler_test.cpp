#include <cassert>
#include <cstdint>
#include <vector>

#include "f1tenth_localization/sensor_packet_assembler.hpp"

int main()
{
  using f1tenth_localization::SensorPacket;
  using f1tenth_localization::SensorPacketAssembler;

  SensorPacketAssembler assembler;
  std::vector<SensorPacket> packets;
  assembler.set_packet_callback([&packets](const SensorPacket & packet) {
    packets.push_back(packet);
  });

  // Callback order is intentionally permuted. The completed packet must still
  // contain exactly the three values sharing source timestamp 100.
  assembler.add_imu_sample(100, 1.0, 2.0, 3.0, 4.0);
  assembler.add_encoder_sample(100, 5.0, false);
  assembler.add_encoder_sample(100, 6.0, true);
  assert(packets.size() == 1);
  assert(packets[0].stamp_ns == 100);
  assert(packets[0].left_angle_rad == 6.0);
  assert(packets[0].right_angle_rad == 5.0);
  assert(packets[0].ax_mps2 == 1.0);

  // An incomplete older timestamp is discarded when a newer source timestamp
  // arrives; it is never completed from a later callback.
  assembler.add_encoder_sample(200, 7.0, true);
  assembler.add_imu_sample(200, 0.0, 0.0, 0.0, 0.0);
  assembler.add_encoder_sample(300, 8.0, false);
  assert(assembler.packet_drop_count() == 1);
  assert(assembler.packet_coherence_fault_count() == 1);
  assembler.add_encoder_sample(300, 9.0, true);
  assembler.add_imu_sample(300, 10.0, 11.0, 12.0, 13.0);
  assert(packets.size() == 2);
  assert(packets[1].stamp_ns == 300);

  // Source-time regression is rejected rather than inserted into the stream.
  assembler.add_imu_sample(250, 0.0, 0.0, 0.0, 0.0);
  assert(assembler.packet_drop_count() == 2);
  assert(assembler.packet_coherence_fault_count() == 2);

  assembler.reset();
  assert(assembler.packet_drop_count() == 0);
  assert(assembler.packet_coherence_fault_count() == 0);
  return 0;
}
