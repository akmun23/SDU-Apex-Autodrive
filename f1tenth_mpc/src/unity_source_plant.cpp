#include "unity_source_plant.hpp"
#include "unity_source_plant_c.h"

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
constexpr double kStickyTireSpeedThresholdMps = 0.2;
constexpr double kStickyForwardTimerThresholdS = 1.0;
constexpr double kStickySideTimerThresholdS = 1.0;
constexpr double kStickyForwardTargetRatio = 0.01;
constexpr double kStickySideTargetRatio = 0.1;
constexpr double kStickyConstraintDamping = 1000.0;
constexpr double kSuspensionMaxCompressionM = 0.025;
constexpr double kSuspensionMaxDroopM = 0.025;
constexpr double kWheelInertiaKgM2 = 0.0001897145;
constexpr double kWheelDampingNms = 0.25;
// The active scene overrides the prefab's 85.6 N m default with 428 N m.
// CAWD divides nonzero drive torque by four; CAWB applies full MotorTorque
// independently as brake torque to each wheel when throttle is exactly zero.
constexpr double kMotorTorqueNm = 428.0;
constexpr double kSteeringRateRadps = 3.2;
constexpr double kMaxSteeringRad = UNITY_SOURCE_MAX_STEERING_RAD;
constexpr std::array<double, 4> kWheelX{
    0.17467999458312988, 0.17467999458312988,
    -0.15532000362873077, -0.15532000362873077};
constexpr std::array<double, 4> kWheelY{
    0.11574111133813858, -0.11590110510587692,
    0.11792000383138657, -0.11807999759912491};
// WheelCollider centres relative to the rigid-body COM, transformed from the
// active Unity scene into forward/left/up. The contact ray starts one radius
// plus max-compression above each centre, exactly as PhysX's raycast vehicle
// path does.
constexpr std::array<double, 4> kWheelCenterZCom{
    -0.004435138776898384, -0.004435138776898384,
    -0.014340000227093697, -0.014340000227093697};
// Active Unity prefab Rigidbody.centerOfMass, expressed in the model frame.
constexpr std::array<double, 3> kComOffsetFromActorBody{
    -0.00468, 0.00008, 0.06434};
// Unity passes this same COM-relative point to PhysX as both the tire-force
// and suspension-force application point. In this plant's forward/left/up
// frame, the point's vertical component is below COM and differs by axle.
constexpr std::array<double, 4> kForceApplicationZ{
    -0.0634351372718811, -0.0634351372718811,
    -0.07333999872207642, -0.07333999872207642};
constexpr std::array<double, 4> kSprungMassKg{
    0.817194700124942, 0.8160186332083913,
    0.9189924292167562, 0.9177942374499102};
constexpr std::array<double, 5> kForwardCurve{
    0.15, 0.9, UNITY_SOURCE_FORWARD_ASYMPTOTE_SLIP, 0.58, 0.8};
constexpr std::array<double, 5> kSideCurve{0.01, 1.0, 0.1, 0.5, 1.0};
constexpr double kTireLoadFilterMin = 0.0;
constexpr double kTireLoadFilterMinValue = 0.2308;
constexpr double kTireLoadFilterMax = 3.0;
constexpr double kTireLoadFilterMaxValue = 3.0;

using Vec3 = std::array<double, 3>;

struct AttitudeBasis {
  double sr{};
  double cr{1.0};
  double sp{};
  double cp{1.0};
  Vec3 ground_normal_body{};
  Vec3 tire_longitudinal_unsteered_body{};
  Vec3 tire_lateral_unsteered_body{};
};

Vec3 add(const Vec3 &a, const Vec3 &b)
{
  return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}

Vec3 subtract(const Vec3 &a, const Vec3 &b)
{
  return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

Vec3 scale(const Vec3 &v, double s)
{
  return {v[0] * s, v[1] * s, v[2] * s};
}

double dot(const Vec3 &a, const Vec3 &b)
{
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

Vec3 cross(const Vec3 &a, const Vec3 &b)
{
  return {a[1] * b[2] - a[2] * b[1],
          a[2] * b[0] - a[0] * b[2],
          a[0] * b[1] - a[1] * b[0]};
}

Vec3 rotate_body_to_world(const AttitudeBasis &attitude, const Vec3 &v)
{
  return {attitude.cp * v[0] + attitude.sp *
            (attitude.sr * v[1] + attitude.cr * v[2]),
          attitude.cr * v[1] - attitude.sr * v[2],
          -attitude.sp * v[0] + attitude.cp *
            (attitude.sr * v[1] + attitude.cr * v[2])};
}

Vec3 rotate_world_to_body(const AttitudeBasis &attitude, const Vec3 &v)
{
  return {attitude.cp * v[0] - attitude.sp * v[2],
          attitude.sp * attitude.sr * v[0] + attitude.cr * v[1] +
            attitude.cp * attitude.sr * v[2],
          attitude.sp * attitude.cr * v[0] - attitude.sr * v[1] +
            attitude.cp * attitude.cr * v[2]};
}

bool make_attitude_basis(const UnitySourcePlantState &s,
                         AttitudeBasis *attitude)
{
  if (!attitude) return false;
  attitude->sr = std::sin(s.roll_rad);
  attitude->cr = std::cos(s.roll_rad);
  attitude->sp = std::sin(s.pitch_rad);
  attitude->cp = std::cos(s.pitch_rad);
  attitude->ground_normal_body = {
    -attitude->sp, attitude->cp * attitude->sr,
    attitude->cp * attitude->cr};

  // This tire basis depends only on chassis attitude and the ground plane,
  // not on the wheel. Compute it once per PhysX vehicle substep and reuse it
  // for all four contacts.
  const Vec3 chassis_lateral_world = rotate_body_to_world(
      *attitude, Vec3{0.0, 1.0, 0.0});
  const Vec3 world_up{0.0, 0.0, 1.0};
  Vec3 long_raw_world = cross(chassis_lateral_world, world_up);
  const double long_norm = std::sqrt(dot(long_raw_world, long_raw_world));
  if (!(long_norm > 1.0e-8)) return false;
  long_raw_world = scale(long_raw_world, 1.0 / long_norm);
  const Vec3 lateral_raw_world = cross(world_up, long_raw_world);
  attitude->tire_longitudinal_unsteered_body = rotate_world_to_body(
      *attitude, long_raw_world);
  attitude->tire_lateral_unsteered_body = rotate_world_to_body(
      *attitude, lateral_raw_world);
  return true;
}

struct WheelContactKinematics {
  Vec3 contact_position_body{};
  Vec3 ground_normal_body{};
  Vec3 tire_longitudinal_body{};
  Vec3 tire_lateral_body{};
  double jounce_m{};
  double forward_speed_mps{};
  double lateral_speed_mps{};
};

bool wheel_contact_kinematics(const UnitySourcePlantState &s,
                              const AttitudeBasis &attitude,
                              std::size_t wheel, double steering_rad,
                              WheelContactKinematics *out)
{
  if (!out || wheel >= 4) return false;
  const Vec3 suspension_direction{0.0, 0.0, -1.0};
  const Vec3 wheel_center{kWheelX[wheel], kWheelY[wheel],
                          kWheelCenterZCom[wheel]};
  const Vec3 ray_start = add(wheel_center,
      Vec3{0.0, 0.0, kWheelRadiusM + kSuspensionMaxCompressionM});
  const Vec3 com_offset{kComOffsetFromActorBody[0],
                        kComOffsetFromActorBody[1],
                        kComOffsetFromActorBody[2]};
  const double ray_direction_dot_normal =
      dot(attitude.ground_normal_body, suspension_direction);
  if (std::abs(ray_direction_dot_normal) < 1.0e-8) return false;

  // Intersect PhysX's suspension ray with the flat racing surface. PhysX
  // defines jounce as (2*radius + maxCompression) minus ray travel.
  const double ray_travel = -(s.body_height_m +
      dot(attitude.ground_normal_body, com_offset) +
      dot(attitude.ground_normal_body, ray_start)) /
      ray_direction_dot_normal;
  const double rest_travel =
      2.0 * kWheelRadiusM + kSuspensionMaxCompressionM;
  out->jounce_m = std::clamp(
      rest_travel - ray_travel, -kSuspensionMaxDroopM,
      kSuspensionMaxCompressionM);
  out->contact_position_body = add(ray_start,
      scale(suspension_direction, rest_travel - out->jounce_m));
  out->ground_normal_body = attitude.ground_normal_body;

  // PxVehicleUpdate computes wheel-bottom velocity as v_COM + omega x r,
  // then projects tire directions onto the hit plane. Use all three angular
  // rates and solve the same ground-plane tire basis in the current attitude.
  const double body_vertical_speed =
      (s.body_height_rate_mps + attitude.sp * s.u_mps -
       attitude.cp * attitude.sr * s.v_mps) /
      (attitude.cp * attitude.cr);
  const Vec3 body_velocity{s.u_mps, s.v_mps, body_vertical_speed};
  const Vec3 angular_velocity{
      s.roll_rate_radps, s.pitch_rate_radps, s.yaw_rate_radps};
  const Vec3 wheel_bottom_velocity = add(
      body_velocity, cross(angular_velocity, out->contact_position_body));

  const double cs = std::cos(steering_rad), sn = std::sin(steering_rad);
  out->tire_longitudinal_body = add(
      scale(attitude.tire_longitudinal_unsteered_body, cs),
      scale(attitude.tire_lateral_unsteered_body, sn));
  out->tire_lateral_body = subtract(
      scale(attitude.tire_lateral_unsteered_body, cs),
      scale(attitude.tire_longitudinal_unsteered_body, sn));
  out->forward_speed_mps = dot(wheel_bottom_velocity,
                               out->tire_longitudinal_body);
  out->lateral_speed_mps = dot(wheel_bottom_velocity,
                               out->tire_lateral_body);
  return std::isfinite(out->jounce_m) &&
      std::isfinite(out->forward_speed_mps) &&
      std::isfinite(out->lateral_speed_mps);
}

bool finite_state(const UnitySourcePlantState &s)
{
  const double scalar[] = {
    s.u_mps, s.v_mps, s.yaw_rate_radps, s.body_height_m,
    s.body_height_rate_mps, s.roll_rad, s.roll_rate_radps,
    s.pitch_rad, s.pitch_rate_radps, s.physical_steering_rad};
  for (double value : scalar) if (!std::isfinite(value)) return false;
  for (double value : s.wheel_omega_radps)
    if (!std::isfinite(value)) return false;
  for (double value : s.wheel_rotation_angle_rad)
    if (!std::isfinite(value)) return false;
  for (double value : s.suspension_jounce_m)
    if (!std::isfinite(value)) return false;
  for (double value : s.low_forward_speed_time_s)
    if (!std::isfinite(value) || value < 0.0) return false;
  for (double value : s.low_side_speed_time_s)
    if (!std::isfinite(value) || value < 0.0) return false;
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

std::array<double, 4> wheel_steering_angles(double physical_steering_rad)
{
  const double tangent = std::tan(physical_steering_rad);
  const double numerator = 2.0 * kWheelbaseM * tangent;
  return {
    // Unity's AutonomousSteering sign is opposite the candidate's
    // forward/left/up physical convention. Transforming the source FL/FR
    // equations therefore swaps the +/- track terms.
    std::atan2(numerator, 2.0 * kWheelbaseM - kTrackM * tangent),
    std::atan2(numerator, 2.0 * kWheelbaseM + kTrackM * tangent), 0.0, 0.0};
}

bool source_slip_excess(
    const UnitySourcePlantState &state,
    std::array<double, 4> *slip,
    std::array<double, 4> *excess_mps,
    std::array<double, 4> *derivative_wheel_omega)
{
  if (!slip || !excess_mps || !derivative_wheel_omega ||
      !finite_state(state)) return false;
  AttitudeBasis attitude;
  if (!make_attitude_basis(state, &attitude)) return false;
  const auto steer = wheel_steering_angles(state.physical_steering_rad);
  for (std::size_t i = 0; i < 4; ++i) {
    WheelContactKinematics contact;
    if (!wheel_contact_kinematics(state, attitude, i, steer[i], &contact))
      return false;
    const double wheel_forward = contact.forward_speed_mps;
    const double wheel_surface = state.wheel_omega_radps[i] * kWheelRadiusM;
    const double denominator =
        std::max(std::abs(wheel_forward), std::abs(wheel_surface)) + 0.1;
    const double difference = wheel_surface - wheel_forward;
    (*slip)[i] = difference / denominator;
    const double residual = std::abs(difference) -
        kForwardCurve[0] * denominator;
    (*excess_mps)[i] = std::max(0.0, residual);
    if (residual <= 0.0) {
      (*derivative_wheel_omega)[i] = 0.0;
      continue;
    }
    const double d_denominator = std::abs(wheel_surface) >
        std::abs(wheel_forward)
        ? std::copysign(kWheelRadiusM, wheel_surface) : 0.0;
    (*derivative_wheel_omega)[i] =
        std::copysign(kWheelRadiusM, difference) -
        kForwardCurve[0] * d_denominator;
  }
  return true;
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

struct StickyTireConstraint {
  bool active{};
  Vec3 direction_body{};
  Vec3 angular_direction_body{};
  double target_speed_mps{};
};

void solve_sticky_tire_velocity_rows(
    const std::array<StickyTireConstraint, 8> &rows,
    double duration,
    double *u_mps, double *v_mps, double *body_height_rate_mps,
    double roll_rad, double pitch_rad,
    double *roll_rate_radps, double *pitch_rate_radps,
    double *yaw_rate_radps,
    Vec3 *delta_linear_body, Vec3 *delta_angular_body)
{
  const double cp = std::cos(pitch_rad), sp = std::sin(pitch_rad);
  const double cr = std::cos(roll_rad), sr = std::sin(roll_rad);
  Vec3 linear_velocity{
      *u_mps, *v_mps,
      (*body_height_rate_mps + sp * *u_mps - cp * sr * *v_mps) / (cp * cr)};
  Vec3 angular_velocity{*roll_rate_radps, *pitch_rate_radps,
                        *yaw_rate_radps};
  const Vec3 initial_linear_velocity = linear_velocity;
  const Vec3 initial_angular_velocity = angular_velocity;
  std::array<double, 8> accumulated_impulse{};
  const double inverse_mass = 1.0 / kMassKg;
  const double inverse_inertia[3] = {
      1.0 / kVehicleMassSpaceInertiaKgM2[0],
      1.0 / kVehicleMassSpaceInertiaKgM2[1],
      1.0 / kVehicleMassSpaceInertiaKgM2[2]};
  const double spring_a = duration * kStickyConstraintDamping;
  const double x = 1.0 / (1.0 + spring_a);
  const double impulse_multiplier = 1.0 - x;

  // PxVehicleSuspLimitConstraintShader orders all forward sticky rows before
  // all side rows. The active scene uses one velocity solver iteration.
  for (std::size_t row = 0; row < rows.size(); ++row) {
    if (!rows[row].active) continue;
    const Vec3 &direction = rows[row].direction_body;
    const Vec3 &angular_direction = rows[row].angular_direction_body;
    const double response = inverse_mass +
        angular_direction[0] * angular_direction[0] * inverse_inertia[0] +
        angular_direction[1] * angular_direction[1] * inverse_inertia[1] +
        angular_direction[2] * angular_direction[2] * inverse_inertia[2];
    if (!(response > 0.0) || !std::isfinite(response)) continue;

    const double normal_velocity = dot(linear_velocity, direction) +
        dot(angular_velocity, angular_direction);
    const double constant = x * spring_a *
        rows[row].target_speed_mps / response;
    const double velocity_multiplier = -x * spring_a / response;
    const double new_impulse = impulse_multiplier * accumulated_impulse[row] +
        velocity_multiplier * normal_velocity + constant;
    const double impulse_delta = new_impulse - accumulated_impulse[row];
    accumulated_impulse[row] = new_impulse;
    linear_velocity = add(linear_velocity,
        scale(direction, inverse_mass * impulse_delta));
    for (std::size_t axis = 0; axis < 3; ++axis) {
      angular_velocity[axis] += angular_direction[axis] *
          inverse_inertia[axis] * impulse_delta;
    }
  }

  *u_mps = linear_velocity[0];
  *v_mps = linear_velocity[1];
  *body_height_rate_mps = linear_velocity[2] * cp * cr -
      sp * linear_velocity[0] + cp * sr * linear_velocity[1];
  *roll_rate_radps = angular_velocity[0];
  *pitch_rate_radps = angular_velocity[1];
  *yaw_rate_radps = angular_velocity[2];
  if (delta_linear_body) {
    *delta_linear_body = subtract(linear_velocity, initial_linear_velocity);
  }
  if (delta_angular_body) {
    *delta_angular_body = subtract(angular_velocity, initial_angular_velocity);
  }
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

  const std::array<double, 4> steer =
      wheel_steering_angles(s.physical_steering_rad);
  std::array<double, 4> fx_tire{}, fy_tire{};
  std::array<WheelContactKinematics, 4> contacts{};
  std::array<double, 4> slip_x{}, slip_y{}, load{}, suspension_force_n{};
  std::array<double, 4> maximum_positive_slip{};
  std::array<double, 4> maximum_slip_excess_mps{};
  std::array<StickyTireConstraint, 8> sticky_rows{};
  std::array<bool, 4> sticky_forward_active{};
  std::array<bool, 4> sticky_side_active{};
  const bool intends_to_accelerate = input.throttle_normalized != 0.0;
  const double drive_torque = input.throttle_normalized == 0.0
      ? 0.0 : kMotorTorqueNm * input.throttle_normalized / 4.0;
  const double brake_torque = input.throttle_normalized == 0.0
      ? kMotorTorqueNm : 0.0;
  const unsigned int substep_count = std::abs(s.u_mps) < 5.0 ? 5U : 1U;
  const double substep_dt = duration / static_cast<double>(substep_count);
  double ax = 0.0, ay = 0.0, yaw_accel = 0.0;
  std::array<double, 4> final_jounce_rate{};
  for (unsigned int substep = 0; substep < substep_count; ++substep) {
    AttitudeBasis attitude;
    if (!make_attitude_basis(s, &attitude)) return false;
    std::array<double, 4> jounce{};
    std::array<double, 4> jounce_rate{};
    Vec3 total_force_body{0.0, 0.0, 0.0};
    Vec3 total_torque_body{0.0, 0.0, 0.0};
    for (std::size_t i = 0; i < 4; ++i) {
      if (!wheel_contact_kinematics(s, attitude, i, steer[i], &contacts[i]))
        return false;
      jounce[i] = contacts[i].jounce_m;
      if (s.suspension_history_initialized) {
        jounce_rate[i] =
            (jounce[i] - s.suspension_jounce_m[i]) / substep_dt;
      }
      const double wheel_forward = contacts[i].forward_speed_mps;
      const double wheel_lateral = contacts[i].lateral_speed_mps;
      const double wheel_surface = s.wheel_omega_radps[i] * kWheelRadiusM;
      const double wheel_omega = s.wheel_omega_radps[i];
      if (!intends_to_accelerate &&
          std::abs(wheel_forward) < kStickyTireSpeedThresholdMps &&
          std::abs(wheel_omega) <
              kStickyTireSpeedThresholdMps / kWheelRadiusM) {
        s.low_forward_speed_time_s[i] = std::min(
            s.low_forward_speed_time_s[i] + substep_dt,
            kStickyForwardTimerThresholdS + kDt);
      } else {
        s.low_forward_speed_time_s[i] = 0.0;
      }
      if (!intends_to_accelerate &&
          std::abs(wheel_lateral) < kStickyTireSpeedThresholdMps) {
        s.low_side_speed_time_s[i] = std::min(
            s.low_side_speed_time_s[i] + substep_dt,
            kStickySideTimerThresholdS + kDt);
      } else {
        s.low_side_speed_time_s[i] = 0.0;
      }
      const bool sticky_forward = !intends_to_accelerate &&
          ((std::abs(wheel_forward) < kStickyTireSpeedThresholdMps &&
            wheel_omega == 0.0) ||
           s.low_forward_speed_time_s[i] >
               kStickyForwardTimerThresholdS);
      const bool sticky_side = s.low_forward_speed_time_s[i] > 0.0 &&
          s.low_side_speed_time_s[i] > kStickySideTimerThresholdS;
      sticky_forward_active[i] = sticky_forward;
      sticky_side_active[i] = sticky_side;
      const Vec3 application_point{
          kWheelX[i], kWheelY[i], kForceApplicationZ[i]};
      sticky_rows[i] = StickyTireConstraint{
          sticky_forward, contacts[i].tire_longitudinal_body,
          cross(application_point, contacts[i].tire_longitudinal_body),
          kStickyForwardTargetRatio * wheel_forward};
      sticky_rows[4 + i] = StickyTireConstraint{
          sticky_side, contacts[i].tire_lateral_body,
          cross(application_point, contacts[i].tire_lateral_body),
          kStickySideTargetRatio * wheel_lateral};
      const double denominator = std::max(
          std::max(std::abs(wheel_forward), std::abs(wheel_surface)) + 0.1,
          0.0);
      const double raw_slip_x = denominator > 0.0
          ? (wheel_surface - wheel_forward) / denominator : 0.0;
      slip_x[i] = sticky_forward ? 0.0 : raw_slip_x;
      maximum_positive_slip[i] =
          std::max(maximum_positive_slip[i], slip_x[i]);
      const double slip_excess = std::max(0.0,
          std::abs(wheel_surface - wheel_forward) -
              kForwardCurve[0] * denominator);
      maximum_slip_excess_mps[i] = std::max(
          maximum_slip_excess_mps[i], slip_excess);
      const double raw_slip_y = std::atan(wheel_lateral /
          (std::abs(wheel_forward) + kMinLateralSpeedForTireModelMps));
      slip_y[i] = sticky_side ? 0.0 : raw_slip_y;
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
      const double force_wheel_x = sticky_forward ? 0.0 :
          load[i] * response_x * kForwardCurve[4] / utilization;
      const double force_wheel_y = sticky_side ? 0.0 :
          -load[i] * response_y * kSideCurve[4] / utilization;
      fx_tire[i] = force_wheel_x;
      fy_tire[i] = force_wheel_y;
      const Vec3 tire_force_body = add(
          scale(contacts[i].tire_longitudinal_body, force_wheel_x),
          scale(contacts[i].tire_lateral_body, force_wheel_y));
      const Vec3 suspension_force_body =
          scale(contacts[i].ground_normal_body, suspension_force_n[i]);
      const Vec3 wheel_force_body = add(tire_force_body,
                                        suspension_force_body);
      total_force_body = add(total_force_body, wheel_force_body);

      // PhysX uses the configured COM-relative suspension/tire application
      // point for both forces, then adds r x F to the rigid-body torque.
      total_torque_body = add(total_torque_body,
                              cross(application_point, wheel_force_body));
    }
    const double body_vertical_speed =
        (s.body_height_rate_mps + std::sin(s.pitch_rad) * s.u_mps -
         std::cos(s.pitch_rad) * std::sin(s.roll_rad) * s.v_mps) /
        (std::cos(s.pitch_rad) * std::cos(s.roll_rad));
    const Vec3 body_velocity{s.u_mps, s.v_mps, body_vertical_speed};
    const Vec3 body_angular_velocity{
        s.roll_rate_radps, s.pitch_rate_radps, s.yaw_rate_radps};
    const Vec3 gravity_body = rotate_world_to_body(
        attitude, Vec3{0.0, 0.0, -kGravity});
    // This is inertial acceleration in body coordinates (what the IMU
    // measures); subtract omega x v separately when differentiating the
    // body-frame velocity state.
    const Vec3 inertial_acceleration_body = add(
        subtract(scale(total_force_body, 1.0 / kMassKg),
                 scale(body_velocity, kDragPerS)), gravity_body);
    const Vec3 body_velocity_rate = subtract(
        inertial_acceleration_body,
        cross(body_angular_velocity, body_velocity));
    ax = inertial_acceleration_body[0];
    ay = inertial_acceleration_body[1];
    const double heave_accel =
        rotate_body_to_world(attitude, inertial_acceleration_body)[2];

    const auto angular_accel = angular_acceleration(
        total_torque_body);
    const double roll_accel = angular_accel[0];
    const double pitch_accel = angular_accel[1];
    yaw_accel = angular_accel[2];

    for (std::size_t i = 0; i < 4; ++i) {
      const double tire_torque = -kWheelRadiusM * fx_tire[i];
      s.wheel_omega_radps[i] = wheel_omega_update(
          s.wheel_omega_radps[i], drive_torque, brake_torque,
          tire_torque, substep_dt);
      s.wheel_rotation_angle_rad[i] +=
          s.wheel_omega_radps[i] * substep_dt;
    }
    // PxVehicleUpdates integrates velocities and its internal chassis
    // transform at every configured vehicle substep (5 below 5 m/s, 1 above).
    s.u_mps += substep_dt * body_velocity_rate[0];
    s.v_mps += substep_dt * body_velocity_rate[1];
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

  // PxVehicleUpdates emits these rows; PxScene solves them after the vehicle
  // substeps. Match its one-velocity-iteration acceleration-spring update in
  // DySolverConstraint1D instead of suppressing low-speed reversal by clamp.
  Vec3 sticky_delta_velocity{}, sticky_delta_angular_velocity{};
  solve_sticky_tire_velocity_rows(sticky_rows, duration,
      &s.u_mps, &s.v_mps, &s.body_height_rate_mps,
      s.roll_rad, s.pitch_rad, &s.roll_rate_radps,
      &s.pitch_rate_radps, &s.yaw_rate_radps,
      &sticky_delta_velocity, &sticky_delta_angular_velocity);
  ax += sticky_delta_velocity[0] / duration;
  ay += sticky_delta_velocity[1] / duration;
  yaw_accel += sticky_delta_angular_velocity[2] / duration;

  // PhysX applies Rigidbody angular damping once in the ordinary 1 ms scene
  // step after PxVehicleUpdates, not inside each vehicle substep.
  const double angular_damping = std::max(1.0 - 0.1 * duration, 0.0);
  s.roll_rate_radps *= angular_damping;
  s.pitch_rate_radps *= angular_damping;
  s.yaw_rate_radps *= angular_damping;

  if (!finite_state(s)) return false;
  out->state = s;
  out->longitudinal_slip = slip_x;
  out->maximum_positive_longitudinal_slip = maximum_positive_slip;
  out->maximum_longitudinal_slip_excess_mps = maximum_slip_excess_mps;
  out->lateral_slip = slip_y;
  out->normal_load_n = load;
  out->suspension_force_n = suspension_force_n;
  out->suspension_jounce_rate_mps = final_jounce_rate;
  out->sticky_tire_longitudinal_active = sticky_forward_active;
  out->sticky_tire_lateral_active = sticky_side_active;
  out->tire_longitudinal_force_n = fx_tire;
  out->tire_lateral_force_n = fy_tire;
  out->longitudinal_acceleration_mps2 = ax;
  out->lateral_acceleration_mps2 = ay;
  out->yaw_acceleration_radps2 = yaw_accel;
  return true;
}

}  // namespace

double unity_steering_command_normalized_to_physical_rad(
    double normalized_command, double max_steering_rad)
{
  return normalized_command * max_steering_rad;
}

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
  UnitySourcePlantOutput result{};
  std::array<double, 4> maximum_positive_slip{};
  std::array<double, 4> maximum_slip_excess_mps{};
  for (unsigned int i = 0; i < exact_steps; ++i) {
    if (!unity_source_plant_step_1ms(current, input, &result)) return false;
    for (std::size_t wheel = 0; wheel < maximum_positive_slip.size(); ++wheel) {
      maximum_positive_slip[wheel] = std::max(maximum_positive_slip[wheel],
          result.maximum_positive_longitudinal_slip[wheel]);
      maximum_slip_excess_mps[wheel] = std::max(
          maximum_slip_excess_mps[wheel],
          result.maximum_longitudinal_slip_excess_mps[wheel]);
    }
    current = result.state;
  }
  result.maximum_positive_longitudinal_slip = maximum_positive_slip;
  result.maximum_longitudinal_slip_excess_mps = maximum_slip_excess_mps;
  *output = result;
  return true;
}

bool unity_source_plant_slip_excess(
    const UnitySourcePlantState &state,
    std::array<double, 4> *longitudinal_slip,
    std::array<double, 4> *excess_speed_mps,
    std::array<double, 4> *excess_derivative_wheel_omega)
{
  return source_slip_excess(state, longitudinal_slip, excess_speed_mps,
                            excess_derivative_wheel_omega);
}

}  // namespace f1tenth_mpc
