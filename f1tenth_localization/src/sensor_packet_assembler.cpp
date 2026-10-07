#include "f1tenth_localization/sensor_packet_assembler.hpp"

#include <algorithm>
#include <iterator>

namespace f1tenth_localization
{

void SensorPacketAssembler::set_packet_callback(PacketCallback callback)
{
  packet_callback_ = std::move(callback);
}

void SensorPacketAssembler::reset()
{
  pending_.clear();
  has_processed_packet_ = false;
  packet_drop_count_ = 0;
  packet_coherence_fault_count_ = 0;
  diagnostics_ = {};
  highest_emitted_source_stamp_ns_ = 0;
}

std::uint64_t SensorPacketAssembler::packet_drop_count() const noexcept
{
  return packet_drop_count_;
}

std::uint64_t SensorPacketAssembler::packet_coherence_fault_count() const noexcept
{
  return packet_coherence_fault_count_;
}

const SensorPacketAssemblerDiagnostics & SensorPacketAssembler::diagnostics() const noexcept
{
  return diagnostics_;
}

template<typename Update>
void SensorPacketAssembler::update_packet(int64_t source_stamp_ns, Update && update)
{
  if (source_stamp_ns <= 0) {
    return;
  }
  diagnostics_.latest_seen_source_stamp_ns = std::max(
    diagnostics_.latest_seen_source_stamp_ns, source_stamp_ns);
  auto & packet = pending_[source_stamp_ns];
  packet.stamp_ns = source_stamp_ns;
  update(packet);

  process_ready_packets();
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
  for (auto it = pending_.begin(); it != pending_.end();) {
    if (!it->second.complete()) {
      ++it;
      continue;
    }
    const SensorPacket packet = it->second;

    const auto reorder_depth = static_cast<std::uint64_t>(std::distance(pending_.begin(), it));
    diagnostics_.maximum_reorder_depth = std::max(
      diagnostics_.maximum_reorder_depth, reorder_depth);
    if (reorder_depth > 0) {
      const auto reorder_time_ns = packet.stamp_ns - pending_.begin()->first;
      diagnostics_.maximum_reorder_time_ns = std::max(
        diagnostics_.maximum_reorder_time_ns, reorder_time_ns);
    }

    if (diagnostics_.last_emitted_source_stamp_ns > 0) {
      if (packet.stamp_ns < diagnostics_.last_emitted_source_stamp_ns) {
        ++diagnostics_.source_reversal_count;
      } else if (packet.stamp_ns == diagnostics_.last_emitted_source_stamp_ns) {
        ++diagnostics_.duplicate_source_count;
      }
    }
    if (highest_emitted_source_stamp_ns_ > 0 &&
      packet.stamp_ns < highest_emitted_source_stamp_ns_)
    {
      ++diagnostics_.late_completed_packet_count;
    }
    highest_emitted_source_stamp_ns_ = std::max(
      highest_emitted_source_stamp_ns_, packet.stamp_ns);
    diagnostics_.last_emitted_source_stamp_ns = packet.stamp_ns;

    it = pending_.erase(it);
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
