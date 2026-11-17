#pragma once

#include "f1tenth_localization/odometry_observer.hpp"
#include "f1tenth_localization/plant_odometry_runtime.hpp"
#include "f1tenth_localization/sensor_packet_assembler.hpp"

namespace f1tenth_localization
{

struct OdometryPipelineStep
{
  OdometryEstimate estimate;
  // Source-plant prediction before the observer correction.
  PredictedPlantState raw_plant;
  double plant_advance_us{0.0};
  double observer_update_us{0.0};
  double plant_correction_us{0.0};
};

// One sensor packet in, one corrected estimate out: advance the source plant
// 25 Unity ticks, correct it with the packet's measurements, and re-seed the
// plant with the corrected state. The odometry node and the offline replay
// tool both run exactly this sequence.
class OdometryPipeline final
{
public:
  static constexpr double kPacketPeriodS = 0.025;

  explicit OdometryPipeline(const OdometryObserverConfig & config);

  OdometryPipelineStep step(
    const SensorPacket & packet,
    double steering_command_norm, double throttle_command_norm,
    double steering_feedback_rad, double throttle_feedback_norm);

  // Plant state after correction, as the next packet will consume it.
  const PredictedPlantState & plant() const noexcept {return plant_;}
  const SourceDerivedOdomRuntime & runtime() const noexcept {return runtime_;}

private:
  OdometryObserver observer_;
  SourceDerivedOdomRuntime runtime_;
  PredictedPlantState plant_{};
  bool started_{false};
  double stamp_s_{0.0};
};

}  // namespace f1tenth_localization
