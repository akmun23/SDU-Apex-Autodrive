#include "unity_source_plant.hpp"
#include "unity_source_plant_c.h"

#include <algorithm>
#include <cmath>

namespace f1tenth_mpc {
namespace {

constexpr double kDt = 0.001;
constexpr double kPi = 3.1415926535897932384626433832795;
constexpr double kMassKg = 3.47;
// The current F1TENTH runtime captures (locked brake, brake pulse, and
// randomized yaw transitions) report inertiaTensor=(0.07804474,0.106500648,
// 0.0399487168) kg m^2 and inertiaTensorRotation=(-0.00117634609,
// 0.00208427128,0.00431931624,0.9999879). The scene constraint solver uses
// this Rigidbody mass frame: R*diag(1/I)*R^T, mapped from Unity
// right/up/forward into plant forward/left/up.
constexpr double kInverseInertiaConstraintBodyKgM2[3][3] = {
    {25.031794528147884, -0.05074067626085555, 0.03695920329073879},
    {-0.05074067626085555, 12.813118838227348, -0.029710480898300164},
    {0.03695920329073879, -0.02971048089830015, 9.389957331844979}};
// PxVehicleUpdate::updateNoDrive resets the temporary chassis COM-pose
// rotation to identity before integrateBody transforms getMassSpaceInvInertiaTensor.
// Consequently its vehicle-force substeps use the captured principal
// mass-space diagonal in the actor frame, not the rotated tensor used by the
// later scene constraint solve. Unity principal moments are right/up/forward;
// map reciprocals into this plant's forward/left/up axes.
constexpr std::array<double, 3> kInverseInertiaVehicleBodyKgM2{
    25.0320931459806, 12.813163321448696, 9.389614230328439};
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
// WheelCollider.center is (0, 0.025, 0) in the active prefab. These are the
// resulting wheel centers relative to the rigid-body COM, including that
// local-center offset and front camber. Sticky activation uses the suspension
// ray's bottom point, not the simplified force-model contact point below.
constexpr std::array<double, 4> kStickyWheelY{
    kWheelY[0], kWheelY[1], kWheelY[2], kWheelY[3]};
constexpr std::array<double, 4> kStickyWheelCenterZCom{
    0.02056485, 0.02056485, 0.01066, 0.01066};
constexpr std::array<std::array<double, 3>, 4> kStickySuspensionDirectionBody{{
    {0.0, 0.08715574274765817, -0.9961946980917455},
    {0.0, -0.08715574274765817, -0.9961946980917455},
    {0.0, 0.0, -1.0},
    {0.0, 0.0, -1.0}}};
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
constexpr double kWheelSubstepThresholdMps = 5.0;
// The shipped WheelCapture UnityPlayer debug symbols identify the active
// WheelCollider backend as PxVehicleNoDrive. Its binary globals are
// gLowLongSpeedSubstepCount=3 and gHighLongSpeedSubstepCount=1; no project
// script calls ConfigureVehicleSubsteps, so PxVehicleWheelsSimData retains
// these defaults. Do not replace this source-backed schedule with a capture-
// fitted effective count.
constexpr unsigned int kWheelSubstepsBelowThreshold = 3;
constexpr unsigned int kWheelSubstepsAboveThreshold = 1;
// Public PhysX PxVehicle sticky-tire thresholds and activation history.
constexpr double kStickyTireSpeedThresholdMps = 0.2;
constexpr double kStickyTireSpeedThresholdTimeS = 1.0;
constexpr double kStickyTireForwardTargetRatio = 0.01;
constexpr double kStickyTireSideTargetRatio = 0.1;
constexpr double kStickyTireConstraintDampingPerS = 1000.0;
// The active Unity Rigidbody reports six position and one velocity solver
// iterations. Sticky rows are ordinary 1D constraints in the normal batch,
// so they are applied in all six position passes and again in the mandatory
// velocity/writeback pass.
constexpr unsigned int kStickyTireSolverIterations = 7;

using Vec3 = std::array<double, 3>;

struct StickyTireConstraint {
  bool active{};
  Vec3 direction_body{};
  Vec3 angular_direction_body{};
  double target_speed_mps{};
};

struct AttitudeBasis {
  double sr{};
  double cr{1.0};
  double sp{};
  double cp{1.0};
  Vec3 ground_normal_body{};
  Vec3 tire_longitudinal_unsteered_body{};
  Vec3 tire_lateral_unsteered_body{};
};

double body_vertical_speed(const UnitySourcePlantState &s,
                           const AttitudeBasis &attitude)
{
  return (s.body_height_rate_mps + attitude.sp * s.u_mps -
          attitude.cp * attitude.sr * s.v_mps) /
      (attitude.cp * attitude.cr);
}

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

struct WheelSteeringBasis {
  double cosine{1.0};
  double sine{};
};

std::array<WheelSteeringBasis, 4> make_wheel_steering_basis(
    const std::array<double, 4> &steer)
{
  std::array<WheelSteeringBasis, 4> basis{};
  for (std::size_t wheel = 0; wheel < basis.size(); ++wheel) {
    basis[wheel] = {std::cos(steer[wheel]), std::sin(steer[wheel])};
  }
  return basis;
}

bool wheel_contact_kinematics(const UnitySourcePlantState &s,
                              const AttitudeBasis &attitude,
                              const Vec3 &body_velocity,
                              const Vec3 &angular_velocity,
                              std::size_t wheel,
                              const WheelSteeringBasis &steering,
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
  const double ray_jounce = rest_travel - ray_travel;
  out->jounce_m = std::clamp(
      ray_jounce, -kSuspensionMaxDroopM,
      kSuspensionMaxCompressionM);
  // PxVehicleUpdate::intersectRayPlane forms wheelBottomPos from the raw
  // ray-plane distance before processSuspWheelTireWheels clamps jounce for
  // suspension force. Keep the contact kinematics on that raw intersection.
  out->contact_position_body = add(
      ray_start, scale(suspension_direction, ray_travel));
  out->ground_normal_body = attitude.ground_normal_body;

  // PxVehicleUpdate computes wheel-bottom velocity as v_COM + omega x r,
  // then projects tire directions onto the hit plane. Use all three angular
  // rates and solve the same ground-plane tire basis in the current attitude.
  const Vec3 wheel_bottom_velocity = add(
      body_velocity, cross(angular_velocity, out->contact_position_body));

  out->tire_longitudinal_body = add(
      scale(attitude.tire_longitudinal_unsteered_body, steering.cosine),
      scale(attitude.tire_lateral_unsteered_body, steering.sine));
  out->tire_lateral_body = subtract(
      scale(attitude.tire_lateral_unsteered_body, steering.cosine),
      scale(attitude.tire_longitudinal_unsteered_body, steering.sine));
  out->forward_speed_mps = dot(wheel_bottom_velocity,
                               out->tire_longitudinal_body);
  out->lateral_speed_mps = dot(wheel_bottom_velocity,
                               out->tire_lateral_body);
  return std::isfinite(out->jounce_m) &&
      std::isfinite(out->forward_speed_mps) &&
      std::isfinite(out->lateral_speed_mps);
}

bool sticky_contact_kinematics(const UnitySourcePlantState &s,
                               const AttitudeBasis &attitude,
                               const Vec3 &body_velocity,
                               const Vec3 &angular_velocity,
                               std::size_t wheel,
                               const WheelSteeringBasis &steering,
                               WheelContactKinematics *out)
{
  if (!out || wheel >= 4) return false;
  const Vec3 suspension_direction{
      kStickySuspensionDirectionBody[wheel][0],
      kStickySuspensionDirectionBody[wheel][1],
      kStickySuspensionDirectionBody[wheel][2]};
  const Vec3 wheel_center{kWheelX[wheel], kStickyWheelY[wheel],
                          kStickyWheelCenterZCom[wheel]};
  const double rest_travel =
      2.0 * kWheelRadiusM + kSuspensionMaxCompressionM;
  const Vec3 ray_start = subtract(wheel_center,
      scale(suspension_direction,
            kWheelRadiusM + kSuspensionMaxCompressionM));
  const double ray_dot_normal =
      dot(attitude.ground_normal_body, suspension_direction);
  if (std::abs(ray_dot_normal) < 1.0e-8) return false;

  const Vec3 com_offset{kComOffsetFromActorBody[0],
                        kComOffsetFromActorBody[1],
                        kComOffsetFromActorBody[2]};
  const double ray_travel = -(s.body_height_m +
      dot(attitude.ground_normal_body, com_offset) +
      dot(attitude.ground_normal_body, ray_start)) / ray_dot_normal;
  const double ray_jounce = rest_travel - ray_travel;
  out->jounce_m = std::clamp(
      ray_jounce, -kSuspensionMaxDroopM,
      kSuspensionMaxCompressionM);
  // Sticky activation and its target speed use PhysX's wheelBottomPos, which
  // is computed from the unclamped ray distance. The later jounce clamp does
  // not move that point.
  out->contact_position_body = add(
      ray_start, scale(suspension_direction, ray_travel));
  out->ground_normal_body = attitude.ground_normal_body;

  const Vec3 wheel_bottom_velocity = add(
      body_velocity, cross(angular_velocity, out->contact_position_body));

  out->tire_longitudinal_body = add(
      scale(attitude.tire_longitudinal_unsteered_body, steering.cosine),
      scale(attitude.tire_lateral_unsteered_body, steering.sine));
  out->tire_lateral_body = subtract(
      scale(attitude.tire_lateral_unsteered_body, steering.cosine),
      scale(attitude.tire_longitudinal_unsteered_body, steering.sine));
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
  const auto steering_basis = make_wheel_steering_basis(steer);
  const Vec3 body_velocity{
      state.u_mps, state.v_mps, body_vertical_speed(state, attitude)};
  const Vec3 angular_velocity{
      state.roll_rate_radps, state.pitch_rate_radps, state.yaw_rate_radps};
  for (std::size_t i = 0; i < 4; ++i) {
    WheelContactKinematics contact;
    if (!wheel_contact_kinematics(
            state, attitude, body_velocity, angular_velocity,
            i, steering_basis[i], &contact))
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

double suspension_force(const AttitudeBasis &attitude, std::size_t i,
                        double jounce, double jounce_rate)
{
  const double denominator = attitude.cp * attitude.cr;
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
  const double applied_torque = drive_torque + tire_torque;
  // WheelCollider brakeTorque can hold a stationary wheel. Treating brake
  // torque as zero at omega == 0 lets tire reaction torque spin a fully
  // braked wheel up from rest, which contradicts the captured Unity encoder
  // response. This is the wheel-level brake complementarity only; it does
  // not impose the unrelated chassis-velocity "sticky tire" constraint.
  if (brake_torque > 0.0 && std::abs(omega) <= 1.0e-12 &&
      std::abs(applied_torque) <= brake_torque)
    return 0.0;
  const double brake = brake_torque > 0.0
      ? -std::copysign(brake_torque, omega != 0.0 ? omega : applied_torque)
      : 0.0;
  const double total_torque = applied_torque + brake;
  const double next = (omega + duration * total_torque / kWheelInertiaKgM2) /
      (1.0 + kWheelDampingNms * duration / kWheelInertiaKgM2);
  if (brake_torque > 0.0 && omega * next <= 0.0) return 0.0;
  omega = next;
  return omega;
}

std::array<double, 3> angular_acceleration(
    const std::array<double, 3> &torque)
{
  // PxVehicleUpdate::updateNoDrive deliberately gives integrateBody a
  // chassis transform whose COM-local rotation is identity. Its vehicle
  // substeps therefore apply the mass-space diagonal in the actor frame.
  // The later scene solver instead uses the Rigidbody mass frame (above).
  // The public vehicle integrator has no separate omega x (I omega) term.
  return {
      kInverseInertiaVehicleBodyKgM2[0] * torque[0],
      kInverseInertiaVehicleBodyKgM2[1] * torque[1],
      kInverseInertiaVehicleBodyKgM2[2] * torque[2]};
}

Vec3 inverse_inertia_times(const Vec3 &value)
{
  return {
      kInverseInertiaConstraintBodyKgM2[0][0] * value[0] +
          kInverseInertiaConstraintBodyKgM2[0][1] * value[1] +
          kInverseInertiaConstraintBodyKgM2[0][2] * value[2],
      kInverseInertiaConstraintBodyKgM2[1][0] * value[0] +
          kInverseInertiaConstraintBodyKgM2[1][1] * value[1] +
          kInverseInertiaConstraintBodyKgM2[1][2] * value[2],
      kInverseInertiaConstraintBodyKgM2[2][0] * value[0] +
          kInverseInertiaConstraintBodyKgM2[2][1] * value[1] +
          kInverseInertiaConstraintBodyKgM2[2][2] * value[2]};
}

void solve_sticky_tire_constraints(
    const std::array<StickyTireConstraint, 8> &rows,
    double duration, const AttitudeBasis &attitude,
    UnitySourcePlantState *state)
{
  if (!state) return;
  std::array<std::size_t, 8> active_rows{};
  std::size_t active_count = 0;
  for (std::size_t row = 0; row < rows.size(); ++row) {
    if (rows[row].active) active_rows[active_count++] = row;
  }
  // With no active PhysX rows, the solver is an identity operation. Avoid
  // converting the velocity basis and running seven empty PGS sweeps on the
  // overwhelmingly common high-speed/no-sticky path.
  if (active_count == 0) return;

  Vec3 linear_velocity{
      state->u_mps, state->v_mps, body_vertical_speed(*state, attitude)};
  Vec3 angular_velocity{
      state->roll_rate_radps, state->pitch_rate_radps,
      state->yaw_rate_radps};
  const double inverse_mass = 1.0 / kMassKg;
  const double spring_a = duration * kStickyTireConstraintDampingPerS;
  const double x = 1.0 / (1.0 + spring_a);
  const double impulse_multiplier = 1.0 - x;
  std::array<double, 8> accumulated_impulse{};
  std::array<Vec3, 8> angular_response{};
  std::array<double, 8> velocity_multiplier{};
  std::array<double, 8> constant{};
  std::size_t valid_count = 0;
  for (std::size_t active_index = 0;
       active_index < active_count; ++active_index) {
    const std::size_t row = active_rows[active_index];
    angular_response[row] =
        inverse_inertia_times(rows[row].angular_direction_body);
    const double response = inverse_mass +
        dot(rows[row].angular_direction_body, angular_response[row]);
    if (!(response > 0.0) || !std::isfinite(response)) continue;
    active_rows[valid_count++] = row;
    velocity_multiplier[row] = -x * spring_a / response;
    constant[row] =
        x * spring_a * rows[row].target_speed_mps / response;
  }
  // PxVehicleSuspLimitConstraintShader emits rows in this exact order:
  // forward FL..RR, followed by side FL..RR. The active Unity body reports
  // six main PGS iterations. Keep PhysX's ordered PGS update; reversing and
  // averaging the rows is not the simulator's algorithm.
  for (unsigned int iteration = 0;
       iteration < kStickyTireSolverIterations; ++iteration) {
    for (std::size_t active_index = 0;
         active_index < valid_count; ++active_index) {
      const std::size_t row = active_rows[active_index];
      const double contact_speed =
          dot(linear_velocity, rows[row].direction_body) +
          dot(angular_velocity, rows[row].angular_direction_body);
      const double new_impulse =
          impulse_multiplier * accumulated_impulse[row] +
          velocity_multiplier[row] * contact_speed + constant[row];
      // The source constraint shader uses +/-FLT_MAX, i.e. no tire-force
      // clamp. Preserve the accumulated impulse update used by PhysX PGS.
      const double impulse_delta = new_impulse - accumulated_impulse[row];
      accumulated_impulse[row] = new_impulse;
      linear_velocity = add(linear_velocity,
          scale(rows[row].direction_body, inverse_mass * impulse_delta));
      angular_velocity = add(angular_velocity,
          scale(angular_response[row], impulse_delta));
    }
  }

  const double cp = std::cos(state->pitch_rad);
  const double sp = std::sin(state->pitch_rad);
  const double cr = std::cos(state->roll_rad);
  const double sr = std::sin(state->roll_rad);
  state->u_mps = linear_velocity[0];
  state->v_mps = linear_velocity[1];
  state->body_height_rate_mps = linear_velocity[2] * cp * cr -
      sp * linear_velocity[0] + cp * sr * linear_velocity[1];
  state->roll_rate_radps = angular_velocity[0];
  state->pitch_rate_radps = angular_velocity[1];
  state->yaw_rate_radps = angular_velocity[2];
}

std::array<double, 2> euler_roll_pitch_rates(
    const AttitudeBasis &attitude, double roll_body_rate,
    double pitch_body_rate, double yaw_body_rate)
{
  // Unity's pose is a quaternion, while this reduced state carries the two
  // tilt angles needed by the flat-ground wheel model. Gyro values are body
  // angular-velocity components p/q/r. For the plant's Z-Y-X Euler
  // convention, the kinematic conversion is:
  //   roll_dot  = p + sin(roll)*tan(pitch)*q + cos(roll)*tan(pitch)*r
  //   pitch_dot = cos(roll)*q - sin(roll)*r
  // Treating p/q as roll_dot/pitch_dot (the old update) loses the yaw/tilt
  // coupling and corrupts suspension-ray orientation during aggressive turns.
  const double tan_pitch = attitude.sp / attitude.cp;
  return {
    roll_body_rate + tan_pitch *
        (attitude.sr * pitch_body_rate + attitude.cr * yaw_body_rate),
    attitude.cr * pitch_body_rate - attitude.sr * yaw_body_rate};
}

bool step(const UnitySourcePlantState &input_state,
          const UnitySourcePlantInput &input, double duration,
          UnitySourcePlantOutput *out, bool complete_output)
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
  const auto steering_basis = make_wheel_steering_basis(steer);
  std::array<double, 4> fx_tire{}, fy_tire{};
  std::array<WheelContactKinematics, 4> contacts{}, sticky_contacts{};
  std::array<double, 4> slip_x{}, slip_y{}, load{}, suspension_force_n{};
  std::array<double, 4> contact_forward_speed{}, contact_lateral_speed{};
  std::array<double, 4> sticky_contact_forward_speed{};
  std::array<double, 4> sticky_contact_lateral_speed{};
  std::array<std::array<double, 3>, 4> sticky_contact_position{};
  std::array<std::array<double, 3>, 4> sticky_tire_longitudinal{};
  std::array<bool, 4> sticky_forward_output{}, sticky_side_output{};
  std::array<double, 4> maximum_positive_slip{};
  std::array<double, 4> maximum_slip_excess_mps{};
  std::array<StickyTireConstraint, 8> sticky_tire_rows{};
  const double drive_torque = input.throttle_normalized == 0.0
      ? 0.0 : kMotorTorqueNm * input.throttle_normalized / 4.0;
  const double brake_torque = input.throttle_normalized == 0.0
      ? kMotorTorqueNm : 0.0;
  const unsigned int substep_count = std::abs(s.u_mps) <
      kWheelSubstepThresholdMps ? kWheelSubstepsBelowThreshold :
                                  kWheelSubstepsAboveThreshold;
  const double substep_dt = duration / static_cast<double>(substep_count);
  double yaw_accel = 0.0;
  std::array<double, 4> final_jounce_rate{};
  for (unsigned int substep = 0; substep < substep_count; ++substep) {
    AttitudeBasis attitude;
    if (!make_attitude_basis(s, &attitude)) return false;
    const Vec3 body_velocity{
        s.u_mps, s.v_mps, body_vertical_speed(s, attitude)};
    const Vec3 body_angular_velocity{
        s.roll_rate_radps, s.pitch_rate_radps, s.yaw_rate_radps};
    std::array<double, 4> jounce{};
    std::array<double, 4> jounce_rate{};
    Vec3 total_force_body{0.0, 0.0, 0.0};
    Vec3 total_torque_body{0.0, 0.0, 0.0};
    for (std::size_t i = 0; i < 4; ++i) {
      if (!wheel_contact_kinematics(
              s, attitude, body_velocity, body_angular_velocity,
              i, steering_basis[i], &contacts[i]))
        return false;
      if (!sticky_contact_kinematics(
              s, attitude, body_velocity, body_angular_velocity,
              i, steering_basis[i], &sticky_contacts[i]))
        return false;
      jounce[i] = contacts[i].jounce_m;
      if (s.suspension_history_initialized) {
        jounce_rate[i] =
            (jounce[i] - s.suspension_jounce_m[i]) / substep_dt;
      }
    }
    for (std::size_t i = 0; i < 4; ++i) {
      const double wheel_forward = contacts[i].forward_speed_mps;
      const double wheel_lateral = contacts[i].lateral_speed_mps;
      const double sticky_wheel_forward =
          sticky_contacts[i].forward_speed_mps;
      const double sticky_wheel_lateral =
          sticky_contacts[i].lateral_speed_mps;
      const double wheel_surface = s.wheel_omega_radps[i] * kWheelRadiusM;
      suspension_force_n[i] = suspension_force(
          attitude, i, jounce[i], jounce_rate[i]);
      load[i] = tire_load(suspension_force_n[i], kSprungMassKg[i]);
      const bool intention_to_accelerate = input.throttle_normalized != 0.0;
      const bool tire_force_branch_active = load[i] > 0.0;
      if (tire_force_branch_active &&
          std::abs(sticky_wheel_forward) < kStickyTireSpeedThresholdMps &&
          std::abs(s.wheel_omega_radps[i]) <
              kStickyTireSpeedThresholdMps / kWheelRadiusM &&
          !intention_to_accelerate) {
        s.low_forward_speed_time_s[i] += substep_dt;
      } else if (tire_force_branch_active) {
        s.low_forward_speed_time_s[i] = 0.0;
      }
      // PxVehicleUpdate.cpp resets both low-speed histories when the driver
      // intends to accelerate. The side-constraint activation later ignores
      // the intent flag itself, but consumes this already-gated timer.
      if (tire_force_branch_active &&
          std::abs(sticky_wheel_lateral) < kStickyTireSpeedThresholdMps &&
          !intention_to_accelerate) {
        s.low_side_speed_time_s[i] += substep_dt;
      } else if (tire_force_branch_active) {
        s.low_side_speed_time_s[i] = 0.0;
      }
      const bool sticky_forward_active = input.apply_sticky_tire_constraints &&
          tire_force_branch_active &&
          !intention_to_accelerate &&
          ((input.apply_immediate_sticky_constraints &&
            std::abs(sticky_wheel_forward) < kStickyTireSpeedThresholdMps &&
            s.wheel_omega_radps[i] == 0.0) ||
           s.low_forward_speed_time_s[i] > kStickyTireSpeedThresholdTimeS);
      const bool sticky_side_active = input.apply_sticky_tire_constraints &&
          tire_force_branch_active &&
          s.low_forward_speed_time_s[i] > 0.0 &&
          s.low_side_speed_time_s[i] > kStickyTireSpeedThresholdTimeS;
      const Vec3 application_point{
          kWheelX[i], kWheelY[i], kForceApplicationZ[i]};
      // Public PxVehicle stores VehicleConstraintData only for internal
      // substep k==0. Timers and tire forces update at every substep, but its
      // solver rows, directions, and targets are frozen from the first one.
      if (substep == 0) {
        contact_forward_speed[i] = wheel_forward;
        contact_lateral_speed[i] = wheel_lateral;
        sticky_contact_forward_speed[i] = sticky_wheel_forward;
        sticky_contact_lateral_speed[i] = sticky_wheel_lateral;
        sticky_contact_position[i] = sticky_contacts[i].contact_position_body;
        sticky_tire_longitudinal[i] =
            sticky_contacts[i].tire_longitudinal_body;
        sticky_forward_output[i] = sticky_forward_active;
        sticky_side_output[i] = sticky_side_active;
        sticky_tire_rows[i] = StickyTireConstraint{
            sticky_forward_active,
            sticky_contacts[i].tire_longitudinal_body,
            cross(application_point,
                  sticky_contacts[i].tire_longitudinal_body),
            kStickyTireForwardTargetRatio * sticky_wheel_forward};
        sticky_tire_rows[4 + i] = StickyTireConstraint{
            sticky_side_active,
            sticky_contacts[i].tire_lateral_body,
            cross(application_point,
                  sticky_contacts[i].tire_lateral_body),
            kStickyTireSideTargetRatio * sticky_wheel_lateral};
      }
      const double denominator = std::max(
          std::max(std::abs(wheel_forward), std::abs(wheel_surface)) + 0.1,
          0.0);
      const double raw_slip_x = denominator > 0.0
          ? (wheel_surface - wheel_forward) / denominator : 0.0;
      slip_x[i] = sticky_forward_active ? 0.0 : raw_slip_x;
      maximum_positive_slip[i] =
          std::max(maximum_positive_slip[i], slip_x[i]);
      const double slip_excess = std::max(0.0,
          std::abs(wheel_surface - wheel_forward) -
              kForwardCurve[0] * denominator);
      maximum_slip_excess_mps[i] = std::max(
          maximum_slip_excess_mps[i], slip_excess);
      // The tire-force curve uses body-left-positive contact velocity here.
      // Unity's WheelHit.sidewaysSlip reports the opposite sign for the same
      // wheel motion; keep this force-model coordinate separate from the
      // externally published WheelHit-compatible diagnostic below.
      const double raw_slip_y = std::atan(wheel_lateral /
          (std::abs(wheel_forward) + kMinLateralSpeedForTireModelMps));
      slip_y[i] = sticky_side_active ? 0.0 : raw_slip_y;
      // The tire-load filter belongs to VehicleComputeTireForce. It is not
      // Unity WheelCollider source; retain it only as the current capture-
      // validated effective load mapping until a WheelCollider-specific
      // mapping is established. The chassis receives the unfiltered support.
      const double response_x = curve_base_response(slip_x[i], kForwardCurve);
      const double response_y = curve_base_response(slip_y[i], kSideCurve);
      // Retain the combined-slip cap only as an empirically supported
      // effective contact-force constraint. Removing it to follow the two
      // separately exposed WheelCollider curves worsened the same capture's
      // 750 ms rollout RMSE (u/v/r: 0.081/0.065/0.068 versus
      // 0.057/0.021/0.034). This is not asserted to be Unity's internal
      // implementation; it remains an explicit model approximation.
      const double normalized_longitudinal =
          response_x / kForwardCurve[1];
      const double normalized_lateral = response_y / kSideCurve[1];
      const double utilization = std::sqrt(std::max(
          1.0, normalized_longitudinal * normalized_longitudinal +
                   normalized_lateral * normalized_lateral));
      const double force_wheel_x = sticky_forward_active ? 0.0 :
          load[i] * response_x * kForwardCurve[4] / utilization;
      const double force_wheel_y = sticky_side_active ? 0.0 :
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
    // PxVehicleUpdate integrates only vehicle tire/suspension forces in its
    // internal substeps. Gravity and Rigidbody damping are applied later by
    // the PhysX scene's once-per-fixed-step unconstrained-velocity pass.
    // Keeping those out of this loop matters at the strict sticky-tire speed
    // boundary: damping each substep can move a contact across 0.2 m/s before
    // PhysX would apply any scene damping.
    const Vec3 inertial_acceleration_body =
        scale(total_force_body, 1.0 / kMassKg);
    const Vec3 body_velocity_rate = subtract(
        inertial_acceleration_body,
        cross(body_angular_velocity, body_velocity));
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
    // transform at every configured vehicle substep.
    s.u_mps += substep_dt * body_velocity_rate[0];
    s.v_mps += substep_dt * body_velocity_rate[1];
    s.yaw_rate_radps += substep_dt * yaw_accel;
    s.body_height_rate_mps += substep_dt * heave_accel;
    s.body_height_m += substep_dt * s.body_height_rate_mps;
    s.roll_rate_radps += substep_dt * roll_accel;
    s.pitch_rate_radps += substep_dt * pitch_accel;
    const auto tilt_rates = euler_roll_pitch_rates(
        attitude, s.roll_rate_radps,
        s.pitch_rate_radps, s.yaw_rate_radps);
    s.roll_rad += substep_dt * tilt_rates[0];
    s.pitch_rad += substep_dt * tilt_rates[1];
    s.suspension_jounce_m = jounce;
    s.suspension_history_initialized = true;
    final_jounce_rate = jounce_rate;
  }

  // Match DyBodyCoreIntegrator.h::bodyCoreComputeUnconstrainedVelocity:
  // scene gravity is added once, then the captured Rigidbody linear damping
  // is applied as max(1 - damping*dt, 0). The vehicle substep transform above
  // remains local to PxVehicleUpdate; the state returned to callers is the
  // actor velocity/height after the scene step.
  AttitudeBasis scene_attitude;
  if (!make_attitude_basis(s, &scene_attitude)) return false;
  Vec3 scene_linear_velocity = rotate_body_to_world(
      scene_attitude,
      Vec3{s.u_mps, s.v_mps, body_vertical_speed(s, scene_attitude)});
  scene_linear_velocity[2] -= kGravity * duration;
  const double linear_damping = std::max(1.0 - kDragPerS * duration, 0.0);
  scene_linear_velocity = scale(scene_linear_velocity, linear_damping);
  const Vec3 scene_body_velocity = rotate_world_to_body(
      scene_attitude, scene_linear_velocity);
  s.u_mps = scene_body_velocity[0];
  s.v_mps = scene_body_velocity[1];
  s.body_height_rate_mps = scene_linear_velocity[2];

  // PxVehicleUpdates first sets the post-vehicle angular velocity. The
  // ordinary PhysX scene step then applies Rigidbody angular damping during
  // unconstrained-velocity pre-integration, before solving its constraint
  // rows (including the public PxVehicle sticky-tire rows).
  const double angular_damping = std::max(1.0 - 0.1 * duration, 0.0);
  s.roll_rate_radps *= angular_damping;
  s.pitch_rate_radps *= angular_damping;
  s.yaw_rate_radps *= angular_damping;

  solve_sticky_tire_constraints(
      sticky_tire_rows, duration, scene_attitude, &s);

  // PxRigidBody pose integration follows the unconstrained-velocity update
  // and sticky-row solve. Preserve the substep height during the vehicle
  // update above, but return the actor height advanced by the final scene
  // velocity, matching the pose that the next suspension raycast observes.
  s.body_height_m = input_state.body_height_m +
      duration * s.body_height_rate_mps;

  if (!finite_state(s)) return false;
  out->state = s;
  out->maximum_positive_longitudinal_slip = maximum_positive_slip;
  out->maximum_longitudinal_slip_excess_mps = maximum_slip_excess_mps;
  if (!complete_output) return true;

  out->contact_forward_speed_mps = contact_forward_speed;
  out->contact_lateral_speed_mps = contact_lateral_speed;
  out->sticky_contact_forward_speed_mps = sticky_contact_forward_speed;
  out->sticky_contact_lateral_speed_mps = sticky_contact_lateral_speed;
  out->sticky_contact_position_body_m = sticky_contact_position;
  out->sticky_tire_longitudinal_body = sticky_tire_longitudinal;
  out->sticky_forward_active = sticky_forward_output;
  out->sticky_side_active = sticky_side_output;
  out->longitudinal_slip = slip_x;
  for (std::size_t i = 0; i < slip_y.size(); ++i)
    out->lateral_slip[i] = -slip_y[i];
  out->normal_load_n = load;
  out->suspension_force_n = suspension_force_n;
  out->suspension_jounce_rate_mps = final_jounce_rate;
  out->tire_longitudinal_force_n = fx_tire;
  out->tire_lateral_force_n = fy_tire;
  const Vec3 final_world_velocity = rotate_body_to_world(
      scene_attitude,
      Vec3{s.u_mps, s.v_mps, body_vertical_speed(s, scene_attitude)});
  const AttitudeBasis initial_attitude{
      std::sin(input_state.roll_rad), std::cos(input_state.roll_rad),
      std::sin(input_state.pitch_rad), std::cos(input_state.pitch_rad)};
  const Vec3 initial_body_velocity{
      input_state.u_mps, input_state.v_mps,
      body_vertical_speed(input_state, initial_attitude)};
  const Vec3 initial_world_velocity = rotate_body_to_world(
      initial_attitude, initial_body_velocity);
  const Vec3 average_world_acceleration = scale(
      subtract(final_world_velocity, initial_world_velocity), 1.0 / duration);
  const Vec3 average_body_acceleration = rotate_world_to_body(
      scene_attitude, average_world_acceleration);
  out->longitudinal_acceleration_mps2 = average_body_acceleration[0];
  out->lateral_acceleration_mps2 = average_body_acceleration[1];
  // Report the effective acceleration of the propagated body-rate state over
  // this call, including scene damping and any sticky-row impulse. `yaw_accel`
  // above is only the unconstrained torque acceleration of the final vehicle
  // substep and is not the derivative of the state returned to callers.
  out->yaw_acceleration_radps2 =
      (s.yaw_rate_radps - input_state.yaw_rate_radps) / duration;
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
  return step(state, input, kDt, output, true);
}

bool unity_source_plant_step_1ms_minimal(
    const UnitySourcePlantState &state,
    const UnitySourcePlantInput &input,
    UnitySourcePlantOutput *output)
{
  return step(state, input, kDt, output, false);
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
    const bool final_step = i + 1 == exact_steps;
    if (!step(current, input, kDt, &result, final_step)) return false;
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
