#include "unity_source_plant.hpp"

#include <algorithm>
#include <cmath>

namespace f1tenth_mpc {
namespace {

constexpr double kDt = 0.001;
constexpr double kPi = 3.1415926535897932384626433832795;
constexpr double kMassKg = 3.47;
// PxVehicleUpdate.cpp::integrateBody uses getMassSpaceInvInertiaTensor()
// transformed by the temporary chassis pose. Its vehicle-specific transform
// resets the COM-local rotation to identity, so the applied vehicle-step
// inertia is the mass-space diagonal (Unity right/up/forward mapped to this
// model's roll/pitch/yaw axes). Do not substitute a full Euler rigid-body
// tensor or add omega x (I omega): the source vehicle integrator does neither.
constexpr std::array<double, 3> kVehicleMassSpaceInertiaKgM2{
    0.0228149127, 0.0934206, 0.100088306};
constexpr double kDragPerS = 0.273;
constexpr double kGravity = 9.81;
constexpr double kWheelbaseM = 0.324;
constexpr double kTrackM = 0.236;
constexpr double kWheelRadiusM = 0.059;
// PxVehicleUpdate.cpp::setVehicleToleranceScale sets the lateral-slip
// denominator floor to 1.0 * scale.length; the active PhysX scale length is 1m.
constexpr double kMinLateralSpeedForTireModelMps = 1.0;
constexpr double kSuspensionMaxCompressionM = 0.025;
constexpr double kSuspensionMaxDroopM = 0.025;
constexpr double kWheelInertiaKgM2 = 0.0001897145;
constexpr double kWheelDampingNms = 0.25;
constexpr double kMotorTorqueNm = 428.0;
constexpr double kSteeringRateRadps = 3.2;
constexpr double kMaxSteeringRad = 0.52359877559829887308;
constexpr std::array<double, 4> kWheelX{
    0.17467999458312988, 0.17467999458312988,
    -0.15532000362873077, -0.15532000362873077};
constexpr std::array<double, 4> kWheelY{
    0.11574111133813858, -0.11590110510587692,
    0.11792000383138657, -0.11807999759912491};
// Unity passes this same COM-relative point to PhysX as both the tire-force
// and suspension-force application point. In this plant's forward/left/up
// frame, the point's vertical component is below COM and differs by axle.
constexpr std::array<double, 4> kForceApplicationZ{
    -0.0634351372718811, -0.0634351372718811,
    -0.07333999872207642, -0.07333999872207642};
// WheelCollider transform centers in the candidate's forward/left/up actor
// frame, before the source target-position shift along suspension-down.
constexpr std::array<double, 4> kWheelBaseX{
    0.17, 0.17, -0.16, -0.16};
constexpr std::array<double, 4> kWheelBaseY{
    0.11582110643130854, -0.11582110643130854,
    0.11792, -0.11808};
constexpr std::array<double, 4> kWheelBaseZ{
    0.08490486745229364, 0.08490486745229364, 0.075, 0.075};
constexpr std::array<double, 4> kSprungMassKg{
    0.817194700124942, 0.8160186332083913,
    0.9189924292167562, 0.9177942374499102};
constexpr std::array<double, 5> kForwardCurve{0.15, 0.9, 0.25, 0.58, 0.8};
constexpr std::array<double, 5> kSideCurve{0.01, 1.0, 0.1, 0.5, 1.0};
constexpr double kTireLoadFilterMin = 0.0;
constexpr double kTireLoadFilterMinValue = 0.2308;
constexpr double kTireLoadFilterMax = 3.0;
constexpr double kTireLoadFilterMaxValue = 3.0;

bool finite_state(const UnitySourcePlantState &s)
{
  const double scalar[] = {
    s.u_mps, s.v_mps, s.yaw_rate_radps, s.body_height_m,
    s.body_height_rate_mps, s.roll_rad, s.roll_rate_radps,
    s.pitch_rad, s.pitch_rate_radps, s.physical_steering_rad};
  for (double value : scalar) if (!std::isfinite(value)) return false;
  for (double value : s.wheel_omega_radps)
    if (!std::isfinite(value)) return false;
  for (double value : s.suspension_jounce_m)
    if (!std::isfinite(value)) return false;
  return true;
}

double curve_base_response(double slip, const std::array<double, 5> &curve)
{
  const double magnitude = std::abs(slip);
  const double extremum_slip = curve[0];
  const double extremum_value = curve[1];
  const double asymptote_slip = curve[2];
  const double asymptote_value = curve[3];
  double response;
  if (magnitude < extremum_slip) {
    const double t = magnitude / extremum_slip;
    response = extremum_value * (t + t * t - t * t * t);
  } else if (magnitude < asymptote_slip) {
    const double t = (magnitude - extremum_slip) /
        (asymptote_slip - extremum_slip);
    const double smooth = 3.0 * t * t - 2.0 * t * t * t;
    response = extremum_value + (asymptote_value - extremum_value) * smooth;
  } else {
    response = asymptote_value;
  }
  return std::copysign(response, slip);
}

double tire_load(double suspension_force, double sprung_mass)
{
  const double rest_load = (sprung_mass + 0.109) * kGravity;
  const double normalized = suspension_force / rest_load;
  double filtered;
  if (normalized <= kTireLoadFilterMin) {
    filtered = kTireLoadFilterMinValue;
  } else if (normalized >= kTireLoadFilterMax) {
    filtered = kTireLoadFilterMaxValue;
  } else {
    filtered = kTireLoadFilterMinValue +
        (normalized - kTireLoadFilterMin) *
        (kTireLoadFilterMaxValue - kTireLoadFilterMinValue) /
        (kTireLoadFilterMax - kTireLoadFilterMin);
  }
  return filtered * rest_load;
}

double suspension_jounce(const UnitySourcePlantState &s, std::size_t i)
{
  const double sr = std::sin(s.roll_rad), cr = std::cos(s.roll_rad);
  const double sp = std::sin(s.pitch_rad), cp = std::cos(s.pitch_rad);
  const double x = kWheelBaseX[i], y = kWheelBaseY[i], z = kWheelBaseZ[i];
  // Unity WheelCollider coordinates were converted into the MPC's
  // forward/left/up frame. The vertical component of R_y(pitch)R_x(roll)p
  // is therefore cos(pitch)*(sin(roll)*y + cos(roll)*z) - sin(pitch)*x.
  const double rotated_lateral_height = sr * y + cr * z;
  const double base_height = cp * rotated_lateral_height - sp * x;
  const double denominator = cp * cr;
  if (std::abs(denominator) < 1.0e-5) return 0.0;
  const double numerator = s.body_height_m + base_height;
  const double travel = numerator / denominator - kWheelRadiusM;
  const double raw_jounce = kSuspensionMaxCompressionM - travel;
  return std::clamp(raw_jounce, -kSuspensionMaxDroopM,
                    kSuspensionMaxCompressionM);
}

double suspension_force(const UnitySourcePlantState &s, std::size_t i,
                        double jounce, double jounce_rate)
{
  const double roll = s.roll_rad;
  const double pitch = s.pitch_rad;
  const double denominator = std::cos(pitch) * std::cos(roll);
  const double gravity_along_suspension =
      std::max(0.0, kGravity * denominator);
  const double spring_weight = std::max(
      0.0, kSprungMassKg[i] * gravity_along_suspension + 500.0 * jounce);
  const double force_along_axis = spring_weight + 100.0 * jounce_rate;
  const double gravity_remainder = denominator > 0.0
      ? kGravity * (1.0 - denominator * denominator) : 0.0;
  return denominator * force_along_axis +
      gravity_remainder * kSprungMassKg[i];
}

double wheel_omega_update(double omega, double drive_torque,
                          double brake_torque, double tire_torque,
                          double duration)
{
  const double brake = omega > 0.0 ? -brake_torque
      : omega < 0.0 ? brake_torque : 0.0;
  const double total_torque = drive_torque + brake + tire_torque;
  const double next = (omega + duration * total_torque / kWheelInertiaKgM2) /
      (1.0 + kWheelDampingNms * duration / kWheelInertiaKgM2);
  if (brake_torque > 0.0 && omega * next <= 0.0) return 0.0;
  omega = next;
  return omega;
}

std::array<double, 3> angular_acceleration(
    const std::array<double, 3> &torque)
{
  return {torque[0] / kVehicleMassSpaceInertiaKgM2[0],
          torque[1] / kVehicleMassSpaceInertiaKgM2[1],
          torque[2] / kVehicleMassSpaceInertiaKgM2[2]};
}

bool step(const UnitySourcePlantState &input_state,
          const UnitySourcePlantInput &input, double duration,
          UnitySourcePlantOutput *out)
{
  if (!out || !finite_state(input_state) ||
      !std::isfinite(input.steering_target_rad) ||
      !std::isfinite(input.throttle_normalized) ||
      !(duration > 0.0) || !std::isfinite(duration) ||
      std::abs(input.steering_target_rad) > kMaxSteeringRad ||
      input.throttle_normalized < -1.0 || input.throttle_normalized > 1.0)
    return false;

  UnitySourcePlantState s = input_state;
  const double steering_target = input.steering_target_rad;
  const unsigned int steering_steps = static_cast<unsigned int>(
      std::llround(duration / kDt));
  const double command_norm = std::clamp(
      steering_target / kMaxSteeringRad, -1.0, 1.0);
  for (unsigned int i = 0; i < steering_steps; ++i) {
    // VehicleController.Steer() stores the internal angle with the opposite
    // sign to the ROS/physical steering convention. Preserve its branch and
    // update ordering, rather than substituting a symmetric servo lag.
    double internal_deg = -s.physical_steering_rad * (180.0 / kPi);
    const double last_internal_deg = internal_deg;
    const double threshold = last_internal_deg / 30.0;
    const double increment = 183.345993 * kDt;
    const double target_deg = -command_norm * 30.0;
    if (command_norm > threshold) {
      internal_deg -= increment;
      if (internal_deg <= target_deg) internal_deg = target_deg;
    } else {
      internal_deg += increment;
      if (internal_deg >= target_deg) internal_deg = target_deg;
    }
    s.physical_steering_rad = -internal_deg * (kPi / 180.0);
  }

  const double tangent = std::tan(s.physical_steering_rad);
  const double numerator = 2.0 * kWheelbaseM * tangent;
  const std::array<double, 4> steer{
    // Unity's AutonomousSteering sign is opposite the candidate's
    // forward/left/up physical convention. Transforming the source FL/FR
    // equations therefore swaps the +/- track terms.
    std::atan2(numerator, 2.0 * kWheelbaseM - kTrackM * tangent),
    std::atan2(numerator, 2.0 * kWheelbaseM + kTrackM * tangent), 0.0, 0.0};
  std::array<double, 4> fx_body{}, fy_body{}, fx_tire{}, fy_tire{};
  std::array<double, 4> slip_x{}, slip_y{}, load{}, suspension_force_n{};
  const double drive_torque = input.throttle_normalized == 0.0
      ? 0.0 : kMotorTorqueNm * input.throttle_normalized / 4.0;
  const double brake_torque = input.throttle_normalized == 0.0
      ? kMotorTorqueNm : 0.0;
  const unsigned int substep_count = std::abs(s.u_mps) < 5.0 ? 5U : 1U;
  const double substep_dt = duration / static_cast<double>(substep_count);
  double ax = 0.0, ay = 0.0, yaw_accel = 0.0;
  std::array<double, 4> final_jounce_rate{};
  for (unsigned int substep = 0; substep < substep_count; ++substep) {
    std::array<double, 4> jounce{};
    std::array<double, 4> jounce_rate{};
    double total_fx = 0.0, total_fy = 0.0, yaw_moment = 0.0;
    for (std::size_t i = 0; i < 4; ++i) {
      jounce[i] = suspension_jounce(s, i);
      if (s.suspension_history_initialized) {
        jounce_rate[i] =
            (jounce[i] - s.suspension_jounce_m[i]) / substep_dt;
      }
      const double vx = s.u_mps - s.yaw_rate_radps * kWheelY[i];
      const double vy = s.v_mps + s.yaw_rate_radps * kWheelX[i];
      const double cs = std::cos(steer[i]), sn = std::sin(steer[i]);
      const double wheel_forward = cs * vx + sn * vy;
      const double wheel_lateral = -sn * vx + cs * vy;
      const double wheel_surface = s.wheel_omega_radps[i] * kWheelRadiusM;
      const double denominator = std::max(
          std::max(std::abs(wheel_forward), std::abs(wheel_surface)) + 0.1,
          0.0);
      slip_x[i] = denominator > 0.0
          ? (wheel_surface - wheel_forward) / denominator : 0.0;
      slip_y[i] = std::atan(wheel_lateral /
          (std::abs(wheel_forward) + kMinLateralSpeedForTireModelMps));
      // The tire-load filter belongs to VehicleComputeTireForce. The chassis
      // receives the unfiltered spring/damper support force from PhysX.
      suspension_force_n[i] =
          suspension_force(s, i, jounce[i], jounce_rate[i]);
      load[i] = tire_load(suspension_force_n[i], kSprungMassKg[i]);
      const double response_x = curve_base_response(slip_x[i], kForwardCurve);
      const double response_y = curve_base_response(slip_y[i], kSideCurve);
      const double utilization = std::sqrt(std::max(
          1.0,
          std::pow(response_x / kForwardCurve[1], 2.0) +
          std::pow(response_y / kSideCurve[1], 2.0)));
      // VehicleComputeTireForce normalizes combined slip using the raw curve
      // responses, then applies each WheelFrictionCurve stiffness to its force.
      const double force_wheel_x = load[i] * response_x * kForwardCurve[4] /
          utilization;
      const double force_wheel_y = -load[i] * response_y * kSideCurve[4] /
          utilization;
      fx_tire[i] = force_wheel_x;
      fy_tire[i] = force_wheel_y;
      fx_body[i] = force_wheel_x * cs - force_wheel_y * sn;
      fy_body[i] = force_wheel_x * sn + force_wheel_y * cs;
      total_fx += fx_body[i];
      total_fy += fy_body[i];
      yaw_moment += kWheelX[i] * fy_body[i] - kWheelY[i] * fx_body[i];
    }

    ax = total_fx / kMassKg - kDragPerS * s.u_mps +
        s.yaw_rate_radps * s.v_mps;
    ay = total_fy / kMassKg - kDragPerS * s.v_mps -
        s.yaw_rate_radps * s.u_mps;
    const double heave_accel =
        (suspension_force_n[0] + suspension_force_n[1] +
         suspension_force_n[2] + suspension_force_n[3]) / kMassKg - kGravity -
        kDragPerS * s.body_height_rate_mps;
    // In the transformed axial frame, an upward normal force on the left
    // wheels gives positive roll torque.
    double roll_moment =
        kWheelY[0] * suspension_force_n[0] +
        kWheelY[1] * suspension_force_n[1] +
        kWheelY[2] * suspension_force_n[2] +
        kWheelY[3] * suspension_force_n[3];
    double pitch_moment =
        -(kWheelX[0] * suspension_force_n[0] +
          kWheelX[1] * suspension_force_n[1] +
          kWheelX[2] * suspension_force_n[2] +
          kWheelX[3] * suspension_force_n[3]);
    for (std::size_t i = 0; i < 4; ++i) {
      // PxVehicleUpdate applies tire force at tireForceAppPointOffset, then
      // adds r x F to chassisTorque. Its Unity value is axle-specific and is
      // not the WheelCollider center nor one shared 74.34 mm lever arm.
      roll_moment -= kForceApplicationZ[i] * fy_body[i];
      pitch_moment += kForceApplicationZ[i] * fx_body[i];
    }
    const auto angular_accel = angular_acceleration(
        {roll_moment, pitch_moment, yaw_moment});
    const double roll_accel = angular_accel[0];
    const double pitch_accel = angular_accel[1];
    yaw_accel = angular_accel[2];

    for (std::size_t i = 0; i < 4; ++i) {
      const double tire_torque = -kWheelRadiusM * fx_tire[i];
      s.wheel_omega_radps[i] = wheel_omega_update(
          s.wheel_omega_radps[i], drive_torque, brake_torque,
          tire_torque, substep_dt);
    }
    // PxVehicleUpdates integrates velocities and its internal chassis
    // transform at every configured vehicle substep (5 below 5 m/s, 1 above).
    s.u_mps += substep_dt * ax;
    s.v_mps += substep_dt * ay;
    s.yaw_rate_radps += substep_dt * yaw_accel;
    s.body_height_rate_mps += substep_dt * heave_accel;
    s.body_height_m += substep_dt * s.body_height_rate_mps;
    s.roll_rate_radps += substep_dt * roll_accel;
    s.roll_rad += substep_dt * s.roll_rate_radps;
    s.pitch_rate_radps += substep_dt * pitch_accel;
    s.pitch_rad += substep_dt * s.pitch_rate_radps;
    s.suspension_jounce_m = jounce;
    s.suspension_history_initialized = true;
    final_jounce_rate = jounce_rate;
  }

  // PhysX applies Rigidbody angular damping once in the ordinary 1 ms scene
  // step after PxVehicleUpdates, not inside each vehicle substep.
  const double angular_damping = std::max(1.0 - 0.1 * duration, 0.0);
  s.roll_rate_radps *= angular_damping;
  s.pitch_rate_radps *= angular_damping;
  s.yaw_rate_radps *= angular_damping;

  if (!finite_state(s)) return false;
  out->state = s;
  out->longitudinal_slip = slip_x;
  out->lateral_slip = slip_y;
  out->normal_load_n = load;
  out->suspension_force_n = suspension_force_n;
  out->suspension_jounce_rate_mps = final_jounce_rate;
  out->tire_longitudinal_force_n = fx_tire;
  out->tire_lateral_force_n = fy_tire;
  out->longitudinal_acceleration_mps2 = ax;
  out->lateral_acceleration_mps2 = ay;
  out->yaw_acceleration_radps2 = yaw_accel;
  return true;
}

}  // namespace

bool unity_source_plant_step_1ms(const UnitySourcePlantState &state,
                                const UnitySourcePlantInput &input,
                                UnitySourcePlantOutput *output)
{
  return step(state, input, kDt, output);
}

bool unity_source_plant_step(const UnitySourcePlantState &state,
                             const UnitySourcePlantInput &input,
                             double duration_s,
                             UnitySourcePlantOutput *output)
{
  if (!output || !std::isfinite(duration_s) || duration_s <= 0.0 ||
      duration_s > 0.2500001) return false;
  const auto exact_steps = static_cast<unsigned int>(
      std::llround(duration_s / kDt));
  if (exact_steps == 0 || std::abs(exact_steps * kDt - duration_s) > 1.0e-8)
    return false;
  UnitySourcePlantState current = state;
  UnitySourcePlantOutput result;
  for (unsigned int i = 0; i < exact_steps; ++i) {
    if (!unity_source_plant_step_1ms(current, input, &result)) return false;
    current = result.state;
  }
  *output = result;
  return true;
}

}  // namespace f1tenth_mpc
