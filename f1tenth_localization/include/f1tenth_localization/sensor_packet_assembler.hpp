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

struct SensorPacketAssemblerDiagnostics
{
  // latest_seen is the maximum source stamp received on any input stream.
  // last_emitted is the source stamp of the last completed packet callback.
  int64_t latest_seen_source_stamp_ns{0};
  int64_t last_emitted_source_stamp_ns{0};
  // Reversal compares with the preceding emitted packet. Duplicate counts a
  // consecutive equal emitted stamp. Late completion compares with the
  // greatest source stamp emitted so far.
  std::uint64_t source_reversal_count{0};
  std::uint64_t duplicate_source_count{0};
  std::uint64_t late_completed_packet_count{0};
  // The current assembler does not discard incomplete packets; this remains
  // zero until an explicit drop policy exists.
  std::uint64_t incomplete_packet_drop_count{0};
  // Number and source-time span of incomplete earlier packets bypassed while
  // emitting a later complete packet.
  std::uint64_t maximum_reorder_depth{0};
  int64_t maximum_reorder_time_ns{0};
};

class SensorPacketAssembler final
{
public:
  using PacketCallback = std::function<void(const SensorPacket &)>;

  SensorPacketAssembler() = default;
  void set_packet_callback(PacketCallback callback);

  void add_encoder_sample(int64_t source_stamp_ns, double angle, bool left);
  void add_imu_sample(
    int64_t source_stamp_ns, double ax, double ay, double yaw_rate, double yaw);
  void reset();

  std::uint64_t packet_drop_count() const noexcept;
  std::uint64_t packet_coherence_fault_count() const noexcept;
  const SensorPacketAssemblerDiagnostics & diagnostics() const noexcept;

private:
  void process_ready_packets();
  void process_packet(const SensorPacket & packet);

  template<typename Update>
  void update_packet(int64_t source_stamp_ns, Update && update);

  std::map<int64_t, SensorPacket> pending_;
  // Subscription startup can observe one sensor before the other two. Those
  // partial packets are expected until the first complete synchronized
  // packet; faults after that point are real stream-integrity failures.
  bool has_processed_packet_{false};
  std::uint64_t packet_drop_count_{0};
  std::uint64_t packet_coherence_fault_count_{0};
  SensorPacketAssemblerDiagnostics diagnostics_;
  int64_t highest_emitted_source_stamp_ns_{0};
  PacketCallback packet_callback_;
};

}  // namespace f1tenth_localization

#endif  // F1TENTH_LOCALIZATION__SENSOR_PACKET_ASSEMBLER_HPP_
