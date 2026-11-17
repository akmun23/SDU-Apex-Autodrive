#pragma once

#include "f1tenth_localization/bridge_request_command_history.hpp"
#include "f1tenth_localization/sensor_packet_assembler.hpp"

#include <deque>
#include <stdexcept>

namespace f1tenth_localization
{

// DDS topics may arrive in any order. Never replace a missing epoch with the
// most recently received actuator sample. The caller serializes all methods.
class BridgeEpochAssembler
{
public:
  using Command = BridgeRequestCommandHistory::Command;

  bool push_command(int64_t stamp, double steering, double throttle)
  {
    return throttle >= 0.0 && commands_.push(stamp, steering, throttle);
  }

  bool push_feedback(int64_t stamp, double steering, double throttle)
  {
    return throttle >= 0.0 && feedback_.push(stamp, steering, throttle);
  }

  void push_packet(const SensorPacket & packet)
  {
    if (!packet.complete() || packet.stamp_ns <= last_packet_stamp_)
      throw std::runtime_error("non-increasing or incomplete Bridge sensor epoch");
    if (packets_.size() >= BridgeRequestCommandHistory::kCapacity)
      throw std::runtime_error("Bridge actuator/sensor epoch join overflow");
    last_packet_stamp_ = packet.stamp_ns;
    packets_.push_back(packet);
  }

  bool pop_ready(SensorPacket * packet, Command * command, Command * feedback)
  {
    if (!packet || !command || !feedback || packets_.empty()) return false;
    const auto & front = packets_.front();
    if (!commands_.find_exact(front.stamp_ns, command) ||
        !feedback_.find_exact(front.stamp_ns, feedback)) return false;
    *packet = front;
    packets_.pop_front();
    return true;
  }

private:
  BridgeRequestCommandHistory commands_;
  BridgeRequestCommandHistory feedback_;
  std::deque<SensorPacket> packets_;
  int64_t last_packet_stamp_{};
};

}  // namespace f1tenth_localization
