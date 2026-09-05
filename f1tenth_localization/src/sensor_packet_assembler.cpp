#include "f1tenth_localization/sensor_packet_assembler.hpp"

#include <algorithm>

namespace f1tenth_localization
{

SensorPacketAssembler::SensorPacketAssembler(std::size_t max_pending_packets)
{
  set_max_pending_packets(max_pending_packets);
}

void SensorPacketAssembler::set_max_pending_packets(std::size_t max_pending_packets)
{
  max_pending_packets_ = std::clamp<std::size_t>(max_pending_packets, 2, 32);
}

void SensorPacketAssembler::set_packet_callback(PacketCallback callback)
{
  packet_callback_ = std::move(callback);
}

void SensorPacketAssembler::reset()
{
  pending_.clear();
  newest_source_stamp_ns_ = 0;
  has_processed_packet_ = false;
  packet_drop_count_ = 0;
  packet_coherence_fault_count_ = 0;
}

std::uint64_t SensorPacketAssembler::packet_drop_count() const noexcept
{
  return packet_drop_count_;
}

std::uint64_t SensorPacketAssembler::packet_coherence_fault_count() const noexcept
{
  return packet_coherence_fault_count_;
}

void SensorPacketAssembler::discard_incomplete_before(int64_t source_stamp_ns)
{
  auto end = pending_.lower_bound(source_stamp_ns);
  for (auto it = pending_.begin(); it != end;) {
    if (!it->second.complete()) {
      if (has_processed_packet_) {
        ++packet_coherence_fault_count_;
        ++packet_drop_count_;
      }
      it = pending_.erase(it);
    } else {
      ++it;
    }
  }
}

template<typename Update>
void SensorPacketAssembler::update_packet(int64_t source_stamp_ns, Update && update)
{
  if (source_stamp_ns <= 0) {
    return;
  }
  if (newest_source_stamp_ns_ > 0 && source_stamp_ns < newest_source_stamp_ns_) {
    if (has_processed_packet_) {
      ++packet_coherence_fault_count_;
      ++packet_drop_count_;
    }
    return;
  }
  discard_incomplete_before(source_stamp_ns);
  newest_source_stamp_ns_ = std::max(newest_source_stamp_ns_, source_stamp_ns);
  auto & packet = pending_[source_stamp_ns];
  packet.stamp_ns = source_stamp_ns;
  update(packet);

  process_ready_packets();
  while (pending_.size() > max_pending_packets_) {
    auto it = pending_.begin();
    if (it->second.complete()) {
      const SensorPacket packet_copy = it->second;
      pending_.erase(it);
      process_packet(packet_copy);
    } else {
      if (has_processed_packet_) {
        ++packet_coherence_fault_count_;
        ++packet_drop_count_;
      }
      pending_.erase(it);
    }
  }
}

void SensorPacketAssembler::add_encoder_sample(
  int64_t source_stamp_ns, double angle, bool left)
{
  update_packet(source_stamp_ns, [angle, left](SensorPacket & packet) {
    if (left) {
      packet.left_angle_rad = angle;
      packet.have_left = true;
    } else {
      packet.right_angle_rad = angle;
      packet.have_right = true;
    }
  });
}

void SensorPacketAssembler::add_imu_sample(
  int64_t source_stamp_ns, double ax, double ay, double yaw_rate, double yaw)
{
  update_packet(source_stamp_ns, [=](SensorPacket & packet) {
    packet.ax_mps2 = ax;
    packet.ay_mps2 = ay;
    packet.yaw_rate_radps = yaw_rate;
    packet.imu_yaw_rad = yaw;
    packet.have_imu = true;
  });
}

void SensorPacketAssembler::process_ready_packets()
{
  while (!pending_.empty() && pending_.begin()->second.complete()) {
    const SensorPacket packet = pending_.begin()->second;
    pending_.erase(pending_.begin());
    process_packet(packet);
  }
}

void SensorPacketAssembler::process_packet(const SensorPacket & packet)
{
  has_processed_packet_ = true;
  if (packet_callback_) {
    packet_callback_(packet);
  }
}

}  // namespace f1tenth_localization
