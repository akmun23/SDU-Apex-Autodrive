#ifndef F1TENTH_LOCALIZATION__SENSOR_PACKET_ASSEMBLER_HPP_
#define F1TENTH_LOCALIZATION__SENSOR_PACKET_ASSEMBLER_HPP_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <utility>

namespace f1tenth_localization
{

struct SensorPacket
{
  int64_t stamp_ns{0};
  double left_angle_rad{0.0};
  double right_angle_rad{0.0};
  double ax_mps2{0.0};
  double ay_mps2{0.0};
  double yaw_rate_radps{0.0};
  double imu_yaw_rad{0.0};
  bool have_left{false};
  bool have_right{false};
  bool have_imu{false};

  bool complete() const noexcept
  {
    return have_left && have_right && have_imu;
  }
};

class SensorPacketAssembler final
{
public:
  using PacketCallback = std::function<void(const SensorPacket &)>;

  explicit SensorPacketAssembler(std::size_t max_pending_packets = 8);

  void set_max_pending_packets(std::size_t max_pending_packets);
  void set_packet_callback(PacketCallback callback);

  void add_encoder_sample(int64_t source_stamp_ns, double angle, bool left);
  void add_imu_sample(
    int64_t source_stamp_ns, double ax, double ay, double yaw_rate, double yaw);
  void reset();

  std::uint64_t packet_drop_count() const noexcept;
  std::uint64_t packet_coherence_fault_count() const noexcept;

private:
  void discard_incomplete_before(int64_t source_stamp_ns);
  void process_ready_packets();
  void process_packet(const SensorPacket & packet);

  template<typename Update>
  void update_packet(int64_t source_stamp_ns, Update && update);

  std::map<int64_t, SensorPacket> pending_;
  std::size_t max_pending_packets_{8};
  int64_t newest_source_stamp_ns_{0};
  // Subscription startup can observe one sensor before the other two. Those
  // partial packets are expected until the first complete synchronized
  // packet; faults after that point are real stream-integrity failures.
  bool has_processed_packet_{false};
  std::uint64_t packet_drop_count_{0};
  std::uint64_t packet_coherence_fault_count_{0};
  PacketCallback packet_callback_;
};

}  // namespace f1tenth_localization

#endif  // F1TENTH_LOCALIZATION__SENSOR_PACKET_ASSEMBLER_HPP_
