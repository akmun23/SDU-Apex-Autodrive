// Include the plant to check its private AD arithmetic without a test API.
#include "../src/unity_source_plant.cpp"
#include "unity_source_plant_c.h"
#include "mpc_types.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>

namespace {

f1tenth_mpc::UnitySourcePlantState state_at_speed(double speed_mps)
{
  f1tenth_mpc::UnitySourcePlantState state;
  state.u_mps = speed_mps;
  // Keep the production state default: it is the active Unity actor height.
  // WheelCollider's native target-position shift is applied by the plant.
  for (double &omega : state.wheel_omega_radps)
    omega = speed_mps / 0.059;
  return state;
}

double raw_curve(double slip, const std::array<double, 5> &curve)
{
  const double magnitude = std::abs(slip);
  double response;
  if (magnitude < curve[0]) {
    const double t = magnitude / curve[0];
    response = curve[1] * (t + t * t - t * t * t);
  } else if (magnitude < curve[2]) {
    const double t = (magnitude - curve[0]) / (curve[2] - curve[0]);
    const double smooth = 3.0 * t * t - 2.0 * t * t * t;
    response = curve[1] + (curve[3] - curve[1]) * smooth;
  } else {
    response = curve[3];
  }
  return std::copysign(response, slip);
}

double expected_yaw_acceleration_from_wheel_forces(
    const f1tenth_mpc::UnitySourcePlantOutput &output)
{
  constexpr double wheel_x[4] = {
    0.17467999458312988, 0.17467999458312988,
    -0.15532000362873077, -0.15532000362873077};
  constexpr double wheel_y[4] = {
    0.11574111133813858, -0.11590110510587692,
    0.11792000383138657, -0.11807999759912491};
  double torque_z = 0.0;
  for (std::size_t wheel = 0; wheel < 4; ++wheel) {
    const double fx = output.tire_longitudinal_force_n[wheel];
    const double fy = output.tire_lateral_force_n[wheel];
    torque_z += wheel_x[wheel] * fy - wheel_y[wheel] * fx;
  }

  // PxVehicleUpdate::updateNoDrive makes the temporary COM pose rotation
  // identity before integrateBody. Vehicle substeps therefore use the
  // principal mass-space diagonal, not the rotated scene-solver tensor.
  // The practice player's measured Rigidbody inertiaTensor.y is 0.100282848.
  constexpr double inverse_yaw_inertia = 9.971794977342487;
  return inverse_yaw_inertia * torque_z;
}

}  // namespace

namespace f1tenth_mpc {
namespace {

void expect_scalar_exact(double actual, double expected)
{
  EXPECT_EQ(actual, expected);
  EXPECT_EQ(std::signbit(actual), std::signbit(expected));
}

template <std::size_t N>
void expect_scalar_exact(const SourceDual<N> &actual, const SourceDual<N> &expected)
{
  expect_scalar_exact(actual.value, expected.value);
  for (std::size_t i=0; i<N; ++i)
    expect_scalar_exact(actual.derivative[i], expected.derivative[i]);
}

template <std::size_t K>
void expect_scalar_exact(const SourceHess<K> &actual, const SourceHess<K> &expected)
{
  expect_scalar_exact(actual.value, expected.value);
  EXPECT_EQ(actual.order, expected.order);
  for (std::size_t i=0; i<K; ++i) expect_scalar_exact(actual.grad[i], expected.grad[i]);
  for (std::size_t i=0; i<actual.kPairs; ++i) expect_scalar_exact(actual.hess[i], expected.hess[i]);
}

double explicit_sine(double value) { return std::sin(value); }
double explicit_cosine(double value) { return std::cos(value); }

template <std::size_t N>
SourceDual<N> explicit_sine(const SourceDual<N> &value)
{
  SourceDual<N> out(std::sin(value.value));
  const AdTangent slope = static_cast<AdTangent>(std::cos(value.value));
  for (std::size_t i=0; i<N; ++i) out.derivative[i] = slope * value.derivative[i];
  return out;
}

template <std::size_t N>
SourceDual<N> explicit_cosine(const SourceDual<N> &value)
{
  SourceDual<N> out(std::cos(value.value));
  const AdTangent slope = static_cast<AdTangent>(-std::sin(value.value));
  for (std::size_t i=0; i<N; ++i) out.derivative[i] = slope * value.derivative[i];
  return out;
}

template <std::size_t K>
SourceHess<K> explicit_chain(const SourceHess<K> &a, double f0, double f1, double f2)
{
  SourceHess<K> out(f0);
  if (a.order == 0) return out;
  for (std::size_t i=0; i<K; ++i) out.grad[i] = f1 * a.grad[i];
  std::size_t p=0;
  for (std::size_t i=0; i<K; ++i)
    for (std::size_t j=i; j<K; ++j, ++p)
      out.hess[p] = a.order == 2 ? f1 * a.hess[p] + f2 * a.grad[i] * a.grad[j]
                                : f2 * a.grad[i] * a.grad[j];
  out.order = 2;
  return out;
}

template <std::size_t K>
SourceHess<K> explicit_sine(const SourceHess<K> &a)
{
  const double si=std::sin(a.value), co=std::cos(a.value);
  return explicit_chain(a,si,co,-si);
}

template <std::size_t K>
SourceHess<K> explicit_cosine(const SourceHess<K> &a)
{
  const double si=std::sin(a.value), co=std::cos(a.value);
  return explicit_chain(a,co,-si,-co);
}

TEST(UnitySourcePlant, PairedTrigMatchesExplicitScalarDualAndHessianArithmetic)
{
  for (double angle : {-3.141592653589793, -.17, -0., 0., .23, 1.5707963267948966, 12.3}) {
    const auto scalar = scalar_sincos(angle);
    expect_scalar_exact(scalar.sine,explicit_sine(angle));
    expect_scalar_exact(scalar.cosine,explicit_cosine(angle));
    SourceDual<28> dual(angle);
    const double lanes[] = {-0., 0., -.73, .19, 1.e-200, -1.e-200, 1., -1.};
    for (std::size_t i=0;i<28;++i) dual.derivative[i]=lanes[i%8];
    const auto first = scalar_sincos(dual);
    expect_scalar_exact(first.sine,explicit_sine(dual));
    expect_scalar_exact(first.cosine,explicit_cosine(dual));
    for (unsigned char order : {0,1,2}) {
      SourceHess<5> hess(angle);
      hess.order=order;
      if (order > 0)
        for (std::size_t i=0;i<5;++i) hess.grad[i]=lanes[i];
      if (order == 2)
        for (std::size_t i=0;i<hess.kPairs;++i) hess.hess[i]=lanes[i%8];
      const auto second = scalar_sincos(hess);
      expect_scalar_exact(second.sine,explicit_sine(hess));
      expect_scalar_exact(second.cosine,explicit_cosine(hess));
    }
  }
}

template <typename Scalar>
Scalar interval_seed(double value, int lane)
{
  Scalar out(value);
  if constexpr (requires { out.derivative; }) out.derivative[lane]=1.;
  else if constexpr (requires { out.grad; }) { out.grad[lane]=1.; out.order=1; }
  return out;
}

template <typename Scalar>
void check_interval_trig(double speed)
{
  UnitySourcePlantStateT<Scalar> state;
  state.u_mps=interval_seed<Scalar>(speed,0);
  state.v_mps=interval_seed<Scalar>(.1,1);
  state.yaw_rate_radps=interval_seed<Scalar>(-.4,2);
  state.roll_rad=.02; state.pitch_rad=-.015;
  for (auto &omega : state.wheel_omega_radps) omega=speed/.059;
  const UnitySourcePlantInputT<Scalar> input{interval_seed<Scalar>(.03,3),interval_seed<Scalar>(.2,4)};
  UnitySourcePlantOutputT<Scalar> actual{}, expected{};
  ASSERT_TRUE(step_interval(state,input,.025,&actual));
  Scalar forward=0., lateral=0., yaw=0.;
  std::array<Scalar,4> maximum_slip{}, maximum_grip{};
  for (int tick=0;tick<25;++tick) {
    const auto current=state;
    if (tick==24) { ASSERT_TRUE((step<Scalar,true>(current,input,kDt,&expected,true))); }
    else { ASSERT_TRUE((step<Scalar,false>(current,input,kDt,&expected,true))); }
    const auto &next=expected.state;
    const Scalar u_mid=.5*(current.u_mps+next.u_mps);
    const Scalar r_mid=.5*(current.yaw_rate_radps+next.yaw_rate_radps);
    const Scalar v_rear_mid=.5*(current.v_mps+next.v_mps)-r_mid*kComForwardFromRearAxleM;
    const Scalar roll_mid=.5*(current.roll_rad+next.roll_rad);
    const Scalar pitch_mid=.5*(current.pitch_rad+next.pitch_rad);
    const Scalar pitch_rate_mid=.5*(current.pitch_rate_radps+next.pitch_rate_radps);
    const Scalar cos_pitch=explicit_cosine(pitch_mid);
    ASSERT_GE(std::abs(primal(cos_pitch)),1.e-6);
    const Scalar yaw_rate_euler=(pitch_rate_mid*explicit_sine(roll_mid)+r_mid*explicit_cosine(roll_mid))/cos_pitch;
    const Scalar yaw_mid=yaw+.5*kDt*yaw_rate_euler;
    forward += kDt*(u_mid*explicit_cosine(yaw_mid)-v_rear_mid*explicit_sine(yaw_mid));
    lateral += kDt*(u_mid*explicit_sine(yaw_mid)+v_rear_mid*explicit_cosine(yaw_mid));
    yaw += kDt*yaw_rate_euler;
    for (std::size_t wheel=0;wheel<4;++wheel) {
      maximum_slip[wheel]=scalar_max(maximum_slip[wheel],expected.maximum_positive_longitudinal_slip[wheel]);
      maximum_grip[wheel]=scalar_max(maximum_grip[wheel],expected.maximum_grip_loss[wheel]);
    }
    state=next;
  }
  expect_scalar_exact(actual.integrated_body_forward_displacement_m,forward);
  expect_scalar_exact(actual.integrated_body_lateral_displacement_m,lateral);
  expect_scalar_exact(actual.integrated_yaw_delta_rad,yaw);
  const auto actual_coordinates=unity_source_plant_state_coordinates(actual.state);
  const auto expected_coordinates=unity_source_plant_state_coordinates(expected.state);
  for (std::size_t i=0;i<actual_coordinates.size();++i)
    expect_scalar_exact(*actual_coordinates[i],*expected_coordinates[i]);
  EXPECT_EQ(actual.state.suspension_history_initialized,expected.state.suspension_history_initialized);
  expect_scalar_exact(actual.longitudinal_acceleration_mps2,expected.longitudinal_acceleration_mps2);
  expect_scalar_exact(actual.lateral_acceleration_mps2,expected.lateral_acceleration_mps2);
  expect_scalar_exact(actual.yaw_acceleration_radps2,expected.yaw_acceleration_radps2);
  for (std::size_t wheel=0;wheel<4;++wheel) {
    expect_scalar_exact(actual.maximum_positive_longitudinal_slip[wheel],maximum_slip[wheel]);
    expect_scalar_exact(actual.maximum_grip_loss[wheel],maximum_grip[wheel]);
    expect_scalar_exact(actual.state.wheel_rotation_angle_rad[wheel],expected.state.wheel_rotation_angle_rad[wheel]);
    expect_scalar_exact(actual.contact_forward_speed_mps[wheel],expected.contact_forward_speed_mps[wheel]);
    expect_scalar_exact(actual.contact_lateral_speed_mps[wheel],expected.contact_lateral_speed_mps[wheel]);
    expect_scalar_exact(actual.sticky_contact_forward_speed_mps[wheel],expected.sticky_contact_forward_speed_mps[wheel]);
    expect_scalar_exact(actual.sticky_contact_lateral_speed_mps[wheel],expected.sticky_contact_lateral_speed_mps[wheel]);
    expect_scalar_exact(actual.longitudinal_slip[wheel],expected.longitudinal_slip[wheel]);
    expect_scalar_exact(actual.lateral_slip[wheel],expected.lateral_slip[wheel]);
    expect_scalar_exact(actual.normal_load_n[wheel],expected.normal_load_n[wheel]);
    expect_scalar_exact(actual.suspension_force_n[wheel],expected.suspension_force_n[wheel]);
    expect_scalar_exact(actual.suspension_jounce_rate_mps[wheel],expected.suspension_jounce_rate_mps[wheel]);
    expect_scalar_exact(actual.tire_longitudinal_force_n[wheel],expected.tire_longitudinal_force_n[wheel]);
    expect_scalar_exact(actual.tire_lateral_force_n[wheel],expected.tire_lateral_force_n[wheel]);
    EXPECT_EQ(actual.sticky_forward_active[wheel],expected.sticky_forward_active[wheel]);
    EXPECT_EQ(actual.sticky_side_active[wheel],expected.sticky_side_active[wheel]);
    for (std::size_t axis=0;axis<3;++axis) {
      expect_scalar_exact(actual.sticky_contact_position_body_m[wheel][axis],expected.sticky_contact_position_body_m[wheel][axis]);
      expect_scalar_exact(actual.sticky_tire_longitudinal_body[wheel][axis],expected.sticky_tire_longitudinal_body[wheel][axis]);
    }
  }
}

TEST(UnitySourcePlant, IntervalDisplacementMatchesExplicitRepeatedTrig)
{
  for (double speed : {1.01,4.99,5.01}) {
    check_interval_trig<double>(speed);
    check_interval_trig<SourceDual<28>>(speed);
    check_interval_trig<SourceHess<5>>(speed);
  }
}

}  // namespace
}  // namespace f1tenth_mpc

TEST(UnitySourcePlant, UsesExactPracticePlayerWheelSubstepSchedule)
{
  // The exact-build native initialization is setSubStepCount(5.0f, 5, 1);
  // PhysX uses the one-substep branch at equality.
  EXPECT_EQ(f1tenth_mpc::unity_source_wheel_substep_count(0.0), 5U);
  EXPECT_EQ(f1tenth_mpc::unity_source_wheel_substep_count(4.999), 5U);
  EXPECT_EQ(f1tenth_mpc::unity_source_wheel_substep_count(-4.999), 5U);
  EXPECT_EQ(f1tenth_mpc::unity_source_wheel_substep_count(5.0), 1U);
  EXPECT_EQ(f1tenth_mpc::unity_source_wheel_substep_count(-5.0), 1U);
  EXPECT_EQ(f1tenth_mpc::unity_source_wheel_substep_count(5.001), 1U);
  EXPECT_EQ(f1tenth_mpc::unity_source_wheel_substep_count(-5.001), 1U);
  EXPECT_EQ(f1tenth_mpc::unity_source_wheel_substep_count(12.0), 1U);
  EXPECT_EQ(f1tenth_mpc::unity_source_wheel_substep_count(128.0), 1U);
  EXPECT_EQ(f1tenth_mpc::unity_source_wheel_substep_count(-128.0), 1U);
}

TEST(UnitySourcePlant, PackedTimerDirectionsMatchAllTwentyEightLanes)
{
  using namespace f1tenth_mpc;
  for (double speed : {1.01, 2., 4.99, 5.01, 8.}) {
    auto state = state_at_speed(speed);
    state.v_mps = .1;
    state.yaw_rate_radps = -.4;
    state.roll_rad = .02;
    UnitySourcePlantInput input{.12, .15};
    for (int tick = 0; tick < 4; ++tick) {
      UnitySourcePlantSensitivitySeeds full{}, packed{};
      for (int j = 0; j < 26; ++j) {
        full.state[j][j] = 1.;
        packed.state[j][j < 18 ? j : 20] = 1.;
      }
      full.steering_target[26] = full.throttle[27] = 1.;
      packed.steering_target[18] = packed.throttle[19] = 1.;
      UnitySourcePlantSensitivity a{}, b{};
      ASSERT_TRUE(unity_source_plant_step_with_sensitivities(state,input,.025,full,&a));
      ASSERT_TRUE(unity_source_plant_step_with_sensitivities(state,input,.025,packed,&b));
      for (int i = 0; i < 26; ++i) {
        for (int j = 0; j < 18; ++j) EXPECT_DOUBLE_EQ(a.state[i][j],b.state[i][j]);
        for (int j = 18; j < 26; ++j)
          EXPECT_DOUBLE_EQ(a.state[i][j],i == j ? b.state[i][20] : 0.);
        EXPECT_DOUBLE_EQ(a.state[i][26],b.state[i][18]);
        EXPECT_DOUBLE_EQ(a.state[i][27],b.state[i][19]);
      }
      for (int j = 0; j < 28; ++j) {
        const int lane = j < 18 ? j : (j < 26 ? 20 : j-8);
        const bool timer = j >= 18 && j < 26;
        EXPECT_DOUBLE_EQ(a.body_forward_displacement[j],timer ? 0. : b.body_forward_displacement[lane]);
        EXPECT_DOUBLE_EQ(a.body_lateral_displacement[j],timer ? 0. : b.body_lateral_displacement[lane]);
        EXPECT_DOUBLE_EQ(a.yaw_delta[j],timer ? 0. : b.yaw_delta[lane]);
        for (int wheel = 0; wheel < 4; ++wheel)
          EXPECT_DOUBLE_EQ(a.maximum_grip_loss_derivative[wheel][j],
                           timer ? 0. : b.maximum_grip_loss_derivative[wheel][lane]);
      }
      UnitySourcePlantOutput next;
      ASSERT_TRUE(unity_source_plant_step(state,input,.025,&next));
      state = next.state;
    }
  }
}

namespace f1tenth_mpc {
namespace {

TEST(UnitySourcePlant, ScalarCompoundTangentsMatchExplicitDualArithmetic)
{
  const AdTangent tiny=std::numeric_limits<AdTangent>::denorm_min();
  const AdTangent large=std::numeric_limits<AdTangent>::max()/4.f;
  const AdTangent lanes[]={-0.f,0.f,-.73f,.19f,tiny,-tiny,large,-large};
  for (double value : {-2.3,-0.,0.,1.37})
    for (double scalar : {-3.25,-0.,0.,1.3}) {
      SourceDual<28> input(value);
      for (std::size_t i=0;i<28;++i) input.derivative[i]=lanes[i%8];
      auto sum=input, difference=input, product=input;
      sum+=scalar; difference-=scalar; product*=scalar;
      SourceDual<28> expected_sum(value+scalar), expected_difference(value-scalar), expected_product(value*scalar);
      for (std::size_t i=0;i<28;++i) {
        expected_sum.derivative[i]=input.derivative[i]+AdTangent(0);
        expected_difference.derivative[i]=input.derivative[i]-AdTangent(0);
        expected_product.derivative[i]=input.derivative[i]*static_cast<AdTangent>(scalar)+
            static_cast<AdTangent>(input.value)*AdTangent(0);
      }
      expect_scalar_exact(sum,expected_sum);
      expect_scalar_exact(difference,expected_difference);
      expect_scalar_exact(product,expected_product);
    }
}

TEST(UnitySourcePlant, ScalarAdditionAndMultiplicationHaveDistinctZeroSigns)
{
  SourceDual<1> input(1.);
  input.derivative[0]=-0.;
  auto sum=input, product=input;
  sum+=1.; product*=1.;
  const auto scalar_sum=input+1., scalar_product=input*1.;
  EXPECT_EQ(sum.derivative[0],input.derivative[0]+0.);
  EXPECT_EQ(product.derivative[0],input.derivative[0]*1.+input.value*0.);
  EXPECT_FALSE(std::signbit(sum.derivative[0]));
  EXPECT_FALSE(std::signbit(product.derivative[0]));
  EXPECT_TRUE(std::signbit(scalar_sum.derivative[0]));
  EXPECT_TRUE(std::signbit(scalar_product.derivative[0]));
}

}  // namespace
}  // namespace f1tenth_mpc

// Exercise both signed spline pieces and the plateau through the public
// source map, including saturated combined slip. The exact rollout supplies
// the finite difference; it does not use the specialized spline derivative.
TEST(UnitySourcePlant, ForceSubstepCapLeavesTheScheduleAtOrAboveItAndApproximatesBelow)
{
  using namespace f1tenth_mpc;
  UnitySourcePlantSensitivitySeeds seeds{};
  for (int k = 0; k < 18; ++k) seeds.state[k][k] = 1.;
  seeds.steering_target[18] = 1.;
  seeds.throttle[19] = 1.;
  const UnitySourcePlantInput input{.2, .3};
  const auto same = [](const UnitySourcePlantSensitivity &a, const UnitySourcePlantSensitivity &b) {
    for (int j = 0; j < 26; ++j)
      for (int lane = 0; lane < 20; ++lane)
        if (a.state[j][lane] != b.state[j][lane]) return false;
    return true;
  };
  // Above 5 m/s the plant already uses one substep per tick: any cap is a no-op.
  auto fast = state_at_speed(8.);
  fast.yaw_rate_radps = .4;
  UnitySourcePlantSensitivity schedule{}, capped{};
  ASSERT_TRUE(unity_source_plant_step_with_sensitivities(fast, input, .025, seeds, &schedule, 0));
  ASSERT_TRUE(unity_source_plant_step_with_sensitivities(fast, input, .025, seeds, &capped, 1));
  EXPECT_TRUE(same(schedule, capped));
  // Below 5 m/s a cap of the full schedule or more is also a no-op,
  auto slow = state_at_speed(3.);
  slow.yaw_rate_radps = 1.;
  ASSERT_TRUE(unity_source_plant_step_with_sensitivities(slow, input, .025, seeds, &schedule, 0));
  ASSERT_TRUE(unity_source_plant_step_with_sensitivities(slow, input, .025, seeds, &capped, 5));
  EXPECT_TRUE(same(schedule, capped));
  // and a smaller cap changes the map but stays near it.
  ASSERT_TRUE(unity_source_plant_step_with_sensitivities(slow, input, .025, seeds, &capped, 1));
  EXPECT_FALSE(same(schedule, capped));
  double difference = 0., reference = 0.;
  for (int j = 0; j < 3; ++j)
    for (int lane = 0; lane < 20; ++lane) {
      const double d = capped.state[j][lane] - schedule.state[j][lane];
      difference += d * d;
      reference += schedule.state[j][lane] * schedule.state[j][lane];
    }
  EXPECT_LT(std::sqrt(difference / reference), .15);
}

TEST(UnitySourcePlant, TireSplineTangentsMatchExactRolloutsAcrossRegions)
{
  using namespace f1tenth_mpc;
  for (double ratio : {.5, .9, 1.03, 1.3, 2.}) {
    for (double lateral : {-.2, -.02, .02, .2, 2.}) {
      auto state = state_at_speed(8.);
      state.v_mps = lateral;
      for (double &omega : state.wheel_omega_radps) omega *= ratio;
      const UnitySourcePlantInput input{.01, .12};
      UnitySourcePlantSensitivitySeeds seeds{};
      seeds.state[UNITY_SOURCE_WHEEL_OMEGA_FL][0] = 1.;
      seeds.state[UNITY_SOURCE_V][1] = 1.;
      UnitySourcePlantSensitivity derivative{};
      ASSERT_TRUE(unity_source_plant_step_with_sensitivities(state,input,.001,seeds,&derivative));
      for (int axis = 0; axis < 2; ++axis) {
        constexpr double epsilon = 1.e-6;
        auto plus = state, minus = state;
        if (axis == 0) {
          plus.wheel_omega_radps[0] += epsilon;
          minus.wheel_omega_radps[0] -= epsilon;
        } else {
          plus.v_mps += epsilon;
          minus.v_mps -= epsilon;
        }
        UnitySourcePlantOutput a{}, b{};
        ASSERT_TRUE(unity_source_plant_step(plus,input,.001,&a));
        ASSERT_TRUE(unity_source_plant_step(minus,input,.001,&b));
        const double exact[] = {(a.state.u_mps-b.state.u_mps)/(2.*epsilon),
            (a.state.v_mps-b.state.v_mps)/(2.*epsilon),
            (a.state.yaw_rate_radps-b.state.yaw_rate_radps)/(2.*epsilon),
            (a.state.wheel_omega_radps[0]-b.state.wheel_omega_radps[0])/(2.*epsilon)};
        for (int j = 0; j < 4; ++j)
          EXPECT_NEAR(derivative.state[j][axis],exact[j],2.e-7+2.e-6*std::abs(exact[j]))
              << "ratio " << ratio << " lateral " << lateral << " axis " << axis << " state " << j;
      }
    }
  }
}

TEST(UnitySourcePlant, OriginalFiveMeterSubstepBoundaryHasMeasuredFiniteJump)
{
  // A literal source hybrid switch, unlike the C1 tire-curve interpolation
  // knots. Retain all other physical initial states identical, including
  // wheel angular speeds, and approach 5 m/s from both sides.
  auto s=state_at_speed(5.);
  s.v_mps=.37;
  s.yaw_rate_radps=.8;
  s.physical_steering_rad=.12;
  const f1tenth_mpc::UnitySourcePlantInput command{.2,.3};
  double differences[3]{};
  int index=0;
  for(double eps:{1.e-3,1.e-4,1.e-5}) {
    auto below=s,above=s;
    below.u_mps=5.-eps;above.u_mps=5.+eps;
    f1tenth_mpc::UnitySourcePlantOutput a{},b{};
    ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step_1ms(
        below,command,&a));
    ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step_1ms(
        above,command,&b));
    const double difference=
        std::abs(a.state.yaw_rate_radps-b.state.yaw_rate_radps)+
        std::abs(a.state.v_mps-b.state.v_mps)+
        std::abs(a.state.wheel_omega_radps[0]-b.state.wheel_omega_radps[0]);
    differences[index++]=difference;
  }
  // Exact source-mode discontinuity must not be disguised with a smooth
  // Hessian. If the source build changes, this assertion forces a review
  // of the branch assumptions and their curvature.
  EXPECT_GT(differences[2],1.e-8);
  EXPECT_LT(differences[2],differences[0]*10.);
}

TEST(UnitySourcePlant, ReconcilesJounceAfterBodyStateCorrection)
{
  auto state = state_at_speed(3.2);
  state.body_height_m = 0.0042;
  state.roll_rad = 0.012;
  state.pitch_rad = -0.018;
  state.physical_steering_rad = 0.21;
  state.suspension_history_initialized = true;
  state.suspension_jounce_m = {0.018, -0.018, 0.012, -0.012};
  const auto unchanged_state = state;

  ASSERT_TRUE(f1tenth_mpc::unity_source_plant_reconcile_suspension_jounce(
      &state));

  EXPECT_DOUBLE_EQ(state.u_mps, unchanged_state.u_mps);
  EXPECT_DOUBLE_EQ(state.v_mps, unchanged_state.v_mps);
  EXPECT_DOUBLE_EQ(state.yaw_rate_radps, unchanged_state.yaw_rate_radps);
  EXPECT_DOUBLE_EQ(state.roll_rad, unchanged_state.roll_rad);
  EXPECT_DOUBLE_EQ(state.pitch_rad, unchanged_state.pitch_rad);
  EXPECT_DOUBLE_EQ(state.physical_steering_rad,
                   unchanged_state.physical_steering_rad);
  EXPECT_TRUE(state.suspension_history_initialized);
  EXPECT_NE(state.suspension_jounce_m,
            unchanged_state.suspension_jounce_m);
  for (double jounce : state.suspension_jounce_m) {
    EXPECT_GE(jounce, -0.025);
    EXPECT_LE(jounce, 0.025);
  }
}

TEST(UnitySourcePlant, WheelPoseAngleMatchesNoDriveUpdateWrap)
{
  constexpr double pi = 3.1415926535897932384626433832795;
  constexpr double wrap = 10.0 * pi;
  auto state = state_at_speed(8.0);  // above 5 m/s: one native substep
  state.wheel_rotation_angle_rad.fill(wrap - 0.001);
  f1tenth_mpc::UnitySourcePlantOutput output;
  ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step_1ms(
      state, {0.0, 0.25}, &output));

  for (std::size_t wheel = 0; wheel < 4; ++wheel) {
    const double unwrapped = state.wheel_rotation_angle_rad[wheel] +
        output.state.wheel_omega_radps[wheel] * 0.001;
    ASSERT_GE(unwrapped, wrap);
    EXPECT_DOUBLE_EQ(output.state.wheel_rotation_angle_rad[wheel],
                     unwrapped - wrap);
    EXPECT_LT(output.state.wheel_rotation_angle_rad[wheel], wrap);
    EXPECT_GE(output.state.wheel_rotation_angle_rad[wheel], -wrap);
  }
}

TEST(UnitySourcePlant, ZeroSlipAtMatchedWheelSurfaceSpeed)
{
  // At 8 m/s the exact native 5/5/1 schedule selects one substep. This
  // keeps the reported slip at the initially matched wheel/body speed before
  // the nonzero torque changes wheel spin.
  const auto state = state_at_speed(8.0);
  f1tenth_mpc::UnitySourcePlantOutput output;
  ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step_1ms(
      state, {0.0, 0.25}, &output));
  for (double slip : output.longitudinal_slip)
    EXPECT_NEAR(slip, 0.0, 1.0e-12);
  for (double slip : output.lateral_slip)
    EXPECT_NEAR(slip, 0.0, 1.0e-12);
  EXPECT_TRUE(std::isfinite(output.longitudinal_acceleration_mps2));
  EXPECT_TRUE(std::isfinite(output.lateral_acceleration_mps2));
  EXPECT_TRUE(std::isfinite(output.yaw_acceleration_radps2));
}

TEST(UnitySourcePlant, MinimalStepsMatchFullStateAndMultiStepDiagnostics)
{
  const auto initial = state_at_speed(4.2);
  const f1tenth_mpc::UnitySourcePlantInput input{0.18, 0.35};
  f1tenth_mpc::UnitySourcePlantOutput full_step{}, minimal_step{};
  ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step_1ms(
      initial, input, &full_step));
  ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step_1ms_minimal(
      initial, input, &minimal_step));

  const auto expect_same_state = [](const auto &a, const auto &b) {
    EXPECT_DOUBLE_EQ(a.u_mps, b.u_mps);
    EXPECT_DOUBLE_EQ(a.v_mps, b.v_mps);
    EXPECT_DOUBLE_EQ(a.yaw_rate_radps, b.yaw_rate_radps);
    EXPECT_DOUBLE_EQ(a.body_height_m, b.body_height_m);
    EXPECT_DOUBLE_EQ(a.body_height_rate_mps, b.body_height_rate_mps);
    EXPECT_DOUBLE_EQ(a.roll_rad, b.roll_rad);
    EXPECT_DOUBLE_EQ(a.roll_rate_radps, b.roll_rate_radps);
    EXPECT_DOUBLE_EQ(a.pitch_rad, b.pitch_rad);
    EXPECT_DOUBLE_EQ(a.pitch_rate_radps, b.pitch_rate_radps);
    EXPECT_DOUBLE_EQ(a.physical_steering_rad, b.physical_steering_rad);
    EXPECT_EQ(a.suspension_history_initialized,
              b.suspension_history_initialized);
    for (std::size_t wheel = 0; wheel < 4; ++wheel) {
      EXPECT_DOUBLE_EQ(a.wheel_omega_radps[wheel],
                       b.wheel_omega_radps[wheel]);
      EXPECT_DOUBLE_EQ(a.wheel_rotation_angle_rad[wheel],
                       b.wheel_rotation_angle_rad[wheel]);
      EXPECT_DOUBLE_EQ(a.suspension_jounce_m[wheel],
                       b.suspension_jounce_m[wheel]);
      EXPECT_DOUBLE_EQ(a.low_forward_speed_time_s[wheel],
                       b.low_forward_speed_time_s[wheel]);
      EXPECT_DOUBLE_EQ(a.low_side_speed_time_s[wheel],
                       b.low_side_speed_time_s[wheel]);
    }
  };
  expect_same_state(full_step.state, minimal_step.state);
  EXPECT_EQ(full_step.maximum_positive_longitudinal_slip,
            minimal_step.maximum_positive_longitudinal_slip);
  EXPECT_EQ(full_step.maximum_grip_loss,
            minimal_step.maximum_grip_loss);

  f1tenth_mpc::UnitySourcePlantOutput batched{}, explicit_output{};
  ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step(
      initial, input, 0.025, &batched));
  auto explicit_state = initial;
  std::array<double, 4> maximum_slip{}, maximum_excess{};
  for (int tick = 0; tick < 25; ++tick) {
    ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step_1ms(
        explicit_state, input, &explicit_output));
    for (std::size_t wheel = 0; wheel < 4; ++wheel) {
      maximum_slip[wheel] = std::max(maximum_slip[wheel],
          explicit_output.maximum_positive_longitudinal_slip[wheel]);
      maximum_excess[wheel] = std::max(maximum_excess[wheel],
          explicit_output.maximum_grip_loss[wheel]);
    }
    explicit_state = explicit_output.state;
  }
  expect_same_state(explicit_state, batched.state);
  EXPECT_EQ(maximum_slip, batched.maximum_positive_longitudinal_slip);
  EXPECT_EQ(maximum_excess, batched.maximum_grip_loss);
  EXPECT_EQ(explicit_output.longitudinal_slip, batched.longitudinal_slip);
  EXPECT_EQ(explicit_output.lateral_slip, batched.lateral_slip);
  EXPECT_DOUBLE_EQ(explicit_output.longitudinal_acceleration_mps2,
                   batched.longitudinal_acceleration_mps2);
  EXPECT_DOUBLE_EQ(explicit_output.lateral_acceleration_mps2,
                   batched.lateral_acceleration_mps2);
  EXPECT_DOUBLE_EQ(explicit_output.yaw_acceleration_radps2,
                   batched.yaw_acceleration_radps2);
}

TEST(UnitySourcePlant, ForwardSensitivitiesMatchSmoothBranchDifferences)
{
  auto state = state_at_speed(6.2);
  state.v_mps = 0.14;
  state.yaw_rate_radps = 0.72;
  state.physical_steering_rad = 0.24;
  state.suspension_history_initialized = true;
  for (std::size_t wheel = 0; wheel < 4; ++wheel) {
    state.wheel_omega_radps[wheel] =
        (state.u_mps + (wheel >= 2 ? 0.12 : 0.0)) / 0.059;
    state.suspension_jounce_m[wheel] = 0.0;
  }
  // Keep the target far enough away that the per-1ms steering slew does not
  // land exactly on its clamp in the last integration sample.
  const f1tenth_mpc::UnitySourcePlantInput input{0.50, 0.42};
  constexpr double dt = 0.025;
  constexpr double epsilon[] = {1.0e-3, 1.0e-3, 1.0e-2, 1.0e-4};
  f1tenth_mpc::UnitySourcePlantSensitivitySeeds seeds{};
  seeds.state[UNITY_SOURCE_U][0] = 1.0;
  seeds.state[UNITY_SOURCE_YAW_RATE][1] = 1.0;
  seeds.state[UNITY_SOURCE_WHEEL_OMEGA_RL][2] = 1.0;
  seeds.state[UNITY_SOURCE_PHYSICAL_STEERING][3] = 1.0;
  f1tenth_mpc::UnitySourcePlantSensitivity sensitivity{};
  ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step_with_sensitivities(
      state, input, dt, seeds, &sensitivity));

  for (std::size_t direction = 0; direction < 4; ++direction) {
    auto plus_state = state;
    auto minus_state = state;
    switch (direction) {
      case 0:
        plus_state.u_mps += epsilon[direction];
        minus_state.u_mps -= epsilon[direction];
        break;
      case 1:
        plus_state.yaw_rate_radps += epsilon[direction];
        minus_state.yaw_rate_radps -= epsilon[direction];
        break;
      case 2:
        plus_state.wheel_omega_radps[2] += epsilon[direction];
        minus_state.wheel_omega_radps[2] -= epsilon[direction];
        break;
      case 3:
        plus_state.physical_steering_rad += epsilon[direction];
        minus_state.physical_steering_rad -= epsilon[direction];
        break;
    }
    f1tenth_mpc::UnitySourcePlantOutput plus{}, minus{};
    ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step(
        plus_state, input, dt, &plus));
    ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step(
        minus_state, input, dt, &minus));
    const double denominator = 2.0 * epsilon[direction];
    const auto check = [&](std::size_t output_index,
                           double plus_value, double minus_value) {
      const double finite_difference = (plus_value - minus_value) / denominator;
      EXPECT_NEAR(sensitivity.state[output_index][direction],
                  finite_difference,
                  2.0e-4 + 2.0e-3 * std::abs(finite_difference))
          << "direction=" << direction << " output=" << output_index;
    };
    check(UNITY_SOURCE_U, plus.state.u_mps, minus.state.u_mps);
    check(UNITY_SOURCE_V, plus.state.v_mps, minus.state.v_mps);
    check(UNITY_SOURCE_YAW_RATE,
          plus.state.yaw_rate_radps, minus.state.yaw_rate_radps);
    check(UNITY_SOURCE_WHEEL_OMEGA_RL,
          plus.state.wheel_omega_radps[2], minus.state.wheel_omega_radps[2]);
    const double d_forward =
        (plus.integrated_body_forward_displacement_m -
         minus.integrated_body_forward_displacement_m) / denominator;
    const double d_lateral =
        (plus.integrated_body_lateral_displacement_m -
         minus.integrated_body_lateral_displacement_m) / denominator;
    const double d_yaw = (plus.integrated_yaw_delta_rad -
                          minus.integrated_yaw_delta_rad) / denominator;
    EXPECT_NEAR(sensitivity.body_forward_displacement[direction], d_forward,
                2.0e-5 + 2.0e-3 * std::abs(d_forward));
    EXPECT_NEAR(sensitivity.body_lateral_displacement[direction], d_lateral,
                2.0e-5 + 2.0e-3 * std::abs(d_lateral));
    EXPECT_NEAR(sensitivity.yaw_delta[direction], d_yaw,
                2.0e-5 + 2.0e-3 * std::abs(d_yaw));
    for (std::size_t wheel = 0; wheel < 4; ++wheel) {
      const double derivative = (plus.maximum_grip_loss[wheel] -
          minus.maximum_grip_loss[wheel]) / denominator;
      EXPECT_NEAR(sensitivity.maximum_grip_loss_derivative[wheel][direction],
          derivative, 2.0e-4 + 2.0e-3*std::abs(derivative))
          << "peak slip direction=" << direction << " wheel=" << wheel;
    }
  }
}

TEST(UnitySourcePlant, ActualRaceHighSteerSlipJacobianMatchesLocalSource)
{
  // Recorded source state: 2.43 m/s, -0.426 rad rack, r=-3.13 rad/s,
  // large lateral slip and roll. Attitude/jounce differences of 10^-5
  // can switch tire branches here. The QP needs the local Jacobian;
  // that derivative does not describe a long step through other branches.
  auto state = state_at_speed(2.4328919425940874);
  state.v_mps = -0.46965857369047104;
  state.yaw_rate_radps = -3.1298;
  state.wheel_omega_radps = {54.10617949432098, 54.86846405988446,
                             49.23056577931635, 50.103356173729836};
  state.body_height_m = 0.0042854783209262635;
  state.body_height_rate_mps = -0.0006659972217720152;
  state.roll_rad = -0.07576734860993374;
  state.roll_rate_radps = 0.1122;
  state.pitch_rad = 0.05662826068699716;
  state.pitch_rate_radps = 0.0486;
  state.physical_steering_rad = -0.42568582719828624;
  state.suspension_jounce_m = {0.013184991097413112,
      -0.004451872282625946, 0.004658395210780819,
      -0.013310263141604828};
  state.suspension_history_initialized = true;
  const f1tenth_mpc::UnitySourcePlantInput control{
      -0.4277801996638101, 0.13};
  f1tenth_mpc::UnitySourcePlantSensitivitySeeds seeds{};
  seeds.state[UNITY_SOURCE_PITCH][0] = 1.0;
  seeds.state[UNITY_SOURCE_SUSPENSION_JOUNCE_RL][1] = 1.0;
  seeds.steering_target[2] = 1.0;
  seeds.throttle[3] = 1.0;
  f1tenth_mpc::UnitySourcePlantSensitivity jac{};
  ASSERT_TRUE(unity_source_plant_step_with_sensitivities(
      state, control, 0.025, seeds, &jac));
  // Small, source-representable perturbations: larger changes belong to
  // nonlinear physical shooting / SQP relinearization, not local A/B tests.
  constexpr double eps[] = {1.0e-8, 1.0e-8, 1.0e-7, 1.0e-6};
  for (int lane = 0; lane < 4; ++lane) {
    auto plus = state, minus = state;
    auto control_plus = control, control_minus = control;
    if (lane == 0) {
      plus.pitch_rad += eps[lane];
      minus.pitch_rad -= eps[lane];
    } else if (lane == 1) {
      plus.suspension_jounce_m[2] += eps[lane];
      minus.suspension_jounce_m[2] -= eps[lane];
    } else if (lane == 2) {
      control_plus.steering_target_rad += eps[lane];
      control_minus.steering_target_rad -= eps[lane];
    } else {
      control_plus.throttle_normalized += eps[lane];
      control_minus.throttle_normalized -= eps[lane];
    }
    f1tenth_mpc::UnitySourcePlantOutput high{}, low{};
    ASSERT_TRUE(unity_source_plant_step(
        plus, control_plus, 0.025, &high));
    ASSERT_TRUE(unity_source_plant_step(
        minus, control_minus, 0.025, &low));
    const auto check = [&](double ad, double a, double b, const char *label) {
      const double fd = (a - b) / (2.0 * eps[lane]);
      EXPECT_NEAR(ad, fd, 0.001 + 0.003 * std::abs(fd))
          << "lane=" << lane << " component=" << label
          << " ad=" << ad << " source_fd=" << fd;
    };
    check(jac.state[UNITY_SOURCE_YAW_RATE][lane],
          high.state.yaw_rate_radps, low.state.yaw_rate_radps, "yaw_rate");
    check(jac.state[UNITY_SOURCE_U][lane],
          high.state.u_mps, low.state.u_mps, "u");
    check(jac.state[UNITY_SOURCE_V][lane],
          high.state.v_mps, low.state.v_mps, "v");
    check(jac.state[UNITY_SOURCE_WHEEL_OMEGA_RL][lane],
          high.state.wheel_omega_radps[2],
          low.state.wheel_omega_radps[2], "omega_rl");
    check(jac.yaw_delta[lane],
          high.integrated_yaw_delta_rad,
          low.integrated_yaw_delta_rad, "integrated_yaw");
    check(jac.body_lateral_displacement[lane],
          high.integrated_body_lateral_displacement_m,
          low.integrated_body_lateral_displacement_m, "integrated_lateral");
  }
}

TEST(UnitySourcePlant, ForwardSensitivitiesSupportUpperEightLanes)
{
  auto state = state_at_speed(6.2);
  state.v_mps = 0.08;
  state.yaw_rate_radps = 0.24;
  state.physical_steering_rad = 0.18;
  state.roll_rad = 0.01;
  state.body_height_m = 0.00464804098;
  state.suspension_history_initialized = true;
  for (std::size_t wheel = 0; wheel < 4; ++wheel)
    state.wheel_omega_radps[wheel] = state.u_mps / 0.059;

  constexpr double epsilon[4] = {2.0e-6, 2.0e-6, 1.0e-4, 1.0e-3};
  const f1tenth_mpc::UnitySourcePlantInput input{0.23, 0.42};
  f1tenth_mpc::UnitySourcePlantSensitivitySeeds seeds{};
  seeds.state[UNITY_SOURCE_BODY_HEIGHT][4] = 1.0;
  seeds.state[UNITY_SOURCE_ROLL][5] = 1.0;
  seeds.steering_target[6] = 1.0;
  seeds.throttle[7] = 1.0;
  f1tenth_mpc::UnitySourcePlantSensitivity sensitivity{};
  ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step_with_sensitivities(
      state, input, 0.025, seeds, &sensitivity));

  for (std::size_t lane = 0; lane < 4; ++lane) {
    const std::size_t direction = lane + 4;
    auto plus_state = state;
    auto minus_state = state;
    auto plus_input = input;
    auto minus_input = input;
    if (lane == 0) {
      plus_state.body_height_m += epsilon[lane];
      minus_state.body_height_m -= epsilon[lane];
    } else if (lane == 1) {
      plus_state.roll_rad += epsilon[lane];
      minus_state.roll_rad -= epsilon[lane];
    } else if (lane == 2) {
      plus_input.steering_target_rad += epsilon[lane];
      minus_input.steering_target_rad -= epsilon[lane];
    } else {
      plus_input.throttle_normalized += epsilon[lane];
      minus_input.throttle_normalized -= epsilon[lane];
    }
    f1tenth_mpc::UnitySourcePlantOutput plus{}, minus{};
    ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step(
        plus_state, plus_input, 0.025, &plus));
    ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step(
        minus_state, minus_input, 0.025, &minus));
    const double denominator = 2.0 * epsilon[lane];
    const auto check = [&](std::size_t output_index,
                           double plus_value, double minus_value) {
      const double finite_difference =
          (plus_value - minus_value) / denominator;
      EXPECT_NEAR(sensitivity.state[output_index][direction], finite_difference,
          2.0e-5 + 2.0e-3 * std::abs(finite_difference))
          << "upper lane=" << lane << " output=" << output_index;
    };
    check(UNITY_SOURCE_U, plus.state.u_mps, minus.state.u_mps);
    check(UNITY_SOURCE_V, plus.state.v_mps, minus.state.v_mps);
    check(UNITY_SOURCE_YAW_RATE,
          plus.state.yaw_rate_radps, minus.state.yaw_rate_radps);
    const double forward_difference =
        (plus.integrated_body_forward_displacement_m -
         minus.integrated_body_forward_displacement_m) / denominator;
    EXPECT_NEAR(sensitivity.body_forward_displacement[direction],
        forward_difference, 2.0e-5 + 2.0e-3 * std::abs(forward_difference));
    for (std::size_t wheel = 0; wheel < 4; ++wheel) {
      const double derivative = (plus.maximum_grip_loss[wheel] -
          minus.maximum_grip_loss[wheel]) / denominator;
      EXPECT_NEAR(sensitivity.maximum_grip_loss_derivative[wheel][direction],
          derivative, 2.0e-4 + 2.0e-3*std::abs(derivative))
          << "peak slip upper lane=" << direction << " wheel=" << wheel;
    }
  }
}

TEST(UnitySourcePlant, ForwardSensitivitiesCoverAttitudeAndJounceStates)
{
  auto state = state_at_speed(6.2);
  state.v_mps = 0.08;
  state.yaw_rate_radps = 0.24;
  state.physical_steering_rad = 0.18;
  state.roll_rad = 0.01;
  state.roll_rate_radps = 0.025;
  state.pitch_rad = -0.006;
  state.pitch_rate_radps = 0.018;
  state.body_height_m = 0.00464804098;
  state.suspension_history_initialized = true;
  for (std::size_t wheel = 0; wheel < 4; ++wheel) {
    state.wheel_omega_radps[wheel] = state.u_mps / 0.059;
    state.suspension_jounce_m[wheel] = 0.0;
  }
  const f1tenth_mpc::UnitySourcePlantInput input{0.23, 0.42};
  constexpr double epsilon = 2.0e-6;
  f1tenth_mpc::UnitySourcePlantSensitivitySeeds seeds{};
  seeds.state[UNITY_SOURCE_BODY_HEIGHT][0] = 1.0;
  seeds.state[UNITY_SOURCE_ROLL][1] = 1.0;
  seeds.state[UNITY_SOURCE_PITCH][2] = 1.0;
  seeds.state[UNITY_SOURCE_SUSPENSION_JOUNCE_RL][3] = 1.0;
  f1tenth_mpc::UnitySourcePlantSensitivity sensitivity{};
  ASSERT_TRUE(unity_source_plant_step_with_sensitivities(
      state, input, 0.025, seeds, &sensitivity));

  const std::size_t input_state[] = {
      UNITY_SOURCE_BODY_HEIGHT, UNITY_SOURCE_ROLL, UNITY_SOURCE_PITCH,
      UNITY_SOURCE_SUSPENSION_JOUNCE_RL};
  for (std::size_t lane = 0; lane < 4; ++lane) {
    auto plus_state = state;
    auto minus_state = state;
    const auto perturb = [&](f1tenth_mpc::UnitySourcePlantState *target,
                             double amount) {
      switch (input_state[lane]) {
        case UNITY_SOURCE_BODY_HEIGHT: target->body_height_m += amount; break;
        case UNITY_SOURCE_ROLL: target->roll_rad += amount; break;
        case UNITY_SOURCE_PITCH: target->pitch_rad += amount; break;
        case UNITY_SOURCE_SUSPENSION_JOUNCE_RL:
          target->suspension_jounce_m[2] += amount;
          break;
      }
    };
    perturb(&plus_state, epsilon);
    perturb(&minus_state, -epsilon);
    f1tenth_mpc::UnitySourcePlantOutput plus{}, minus{};
    ASSERT_TRUE(unity_source_plant_step(plus_state, input, 0.025, &plus));
    ASSERT_TRUE(unity_source_plant_step(minus_state, input, 0.025, &minus));
    const auto check = [&](std::size_t output_state,
                           double plus_value, double minus_value) {
      const double finite_difference =
          (plus_value - minus_value) / (2.0 * epsilon);
      EXPECT_NEAR(sensitivity.state[output_state][lane], finite_difference,
          2.0e-5 + 2.0e-4 * std::abs(finite_difference))
          << "lane=" << lane << " output=" << output_state;
    };
    check(UNITY_SOURCE_U, plus.state.u_mps, minus.state.u_mps);
    check(UNITY_SOURCE_V, plus.state.v_mps, minus.state.v_mps);
    check(UNITY_SOURCE_YAW_RATE,
          plus.state.yaw_rate_radps, minus.state.yaw_rate_radps);
    check(UNITY_SOURCE_WHEEL_OMEGA_FL,
          plus.state.wheel_omega_radps[0], minus.state.wheel_omega_radps[0]);
    check(UNITY_SOURCE_ROLL_RATE,
          plus.state.roll_rate_radps, minus.state.roll_rate_radps);
  }
}

TEST(UnitySourcePlant, ForwardSensitivitiesMatchAllFourJounceDirections)
{
  auto state = state_at_speed(6.0);
  state.v_mps = 0.08 + 0.24 * 0.15532;
  state.yaw_rate_radps = 0.24;
  state.physical_steering_rad = 0.18;
  state.body_height_m = 0.00464804098;
  state.roll_rad = 0.01;
  state.roll_rate_radps = 0.025;
  state.pitch_rad = -0.006;
  state.pitch_rate_radps = 0.018;
  state.suspension_history_initialized = true;
  for (std::size_t wheel = 0; wheel < 4; ++wheel) {
    state.wheel_omega_radps[wheel] = state.u_mps / 0.059;
    state.suspension_jounce_m[wheel] = 0.0;
  }
  const f1tenth_mpc::UnitySourcePlantInput input{0.23, 0.42};
  constexpr double epsilon = 2.0e-6;
  f1tenth_mpc::UnitySourcePlantSensitivitySeeds seeds{};
  for (std::size_t wheel = 0; wheel < 4; ++wheel)
    seeds.state[UNITY_SOURCE_SUSPENSION_JOUNCE_FL + wheel][wheel] = 1.0;
  f1tenth_mpc::UnitySourcePlantSensitivity sensitivity{};
  ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step_with_sensitivities(
      state, input, 0.025, seeds, &sensitivity));

  for (std::size_t lane = 0; lane < 4; ++lane) {
    auto plus_state = state;
    auto minus_state = state;
    plus_state.suspension_jounce_m[lane] += epsilon;
    minus_state.suspension_jounce_m[lane] -= epsilon;
    f1tenth_mpc::UnitySourcePlantOutput plus{}, minus{};
    ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step(
        plus_state, input, 0.025, &plus));
    ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step(
        minus_state, input, 0.025, &minus));
    const auto check = [&](std::size_t output_index,
                           double plus_value, double minus_value) {
      const double finite_difference =
          (plus_value - minus_value) / (2.0 * epsilon);
      EXPECT_NEAR(sensitivity.state[output_index][lane], finite_difference,
          2.0e-5 + 2.0e-4 * std::abs(finite_difference))
          << "wheel=" << lane << " output=" << output_index;
    };
    check(UNITY_SOURCE_U, plus.state.u_mps, minus.state.u_mps);
    check(UNITY_SOURCE_V, plus.state.v_mps, minus.state.v_mps);
    check(UNITY_SOURCE_YAW_RATE,
          plus.state.yaw_rate_radps, minus.state.yaw_rate_radps);
    check(UNITY_SOURCE_WHEEL_OMEGA_FL,
          plus.state.wheel_omega_radps[0], minus.state.wheel_omega_radps[0]);
    check(UNITY_SOURCE_WHEEL_OMEGA_RR,
          plus.state.wheel_omega_radps[3], minus.state.wheel_omega_radps[3]);
    check(UNITY_SOURCE_ROLL_RATE,
          plus.state.roll_rate_radps, minus.state.roll_rate_radps);
  }
}

TEST(UnitySourcePlant, SuspensionJounceHistoryInitializesAndAdvances)
{
  auto state = state_at_speed(5.0);
  // Move the collider ray farther into suspension travel after accounting
  // for WheelCollider.center.
  state.body_height_m = -0.025;
  f1tenth_mpc::UnitySourcePlantOutput first;
  ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step_1ms(
      state, {0.0, 0.25}, &first));
  EXPECT_TRUE(first.state.suspension_history_initialized);
  for (double jounce : first.state.suspension_jounce_m)
    EXPECT_TRUE(std::isfinite(jounce));

  auto displaced = first.state;
  displaced.body_height_m += 0.001;
  f1tenth_mpc::UnitySourcePlantOutput second;
  ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step_1ms(
      displaced, {0.0, 0.25}, &second));
  EXPECT_TRUE(second.state.suspension_history_initialized);
  EXPECT_NE(second.state.suspension_jounce_m[0],
            first.state.suspension_jounce_m[0]);
  for (double load : second.normal_load_n)
    EXPECT_TRUE(std::isfinite(load));
}

TEST(UnitySourcePlant, PhysXAppliesNoWheelForceBeyondMaximumDroop)
{
  // This fixture places all four ray intersections beyond max droop. PhysX
  // reports jounce=-maxDroop and leaves the initialized force/slip outputs at
  // zero; a positive spring or filtered tire load here is a model error.
  const auto state = state_at_speed(4.0);
  auto airborne = state;
  airborne.body_height_m = 0.045;
  f1tenth_mpc::UnitySourcePlantOutput output;
  ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step_1ms(
      airborne, {0.0, 0.25}, &output));
  for (std::size_t wheel = 0; wheel < 4; ++wheel) {
    EXPECT_DOUBLE_EQ(output.state.suspension_jounce_m[wheel], -0.025);
    EXPECT_DOUBLE_EQ(output.suspension_force_n[wheel], 0.0);
    EXPECT_DOUBLE_EQ(output.normal_load_n[wheel], 0.0);
    EXPECT_DOUBLE_EQ(output.tire_longitudinal_force_n[wheel], 0.0);
    EXPECT_DOUBLE_EQ(output.tire_lateral_force_n[wheel], 0.0);
    EXPECT_DOUBLE_EQ(output.longitudinal_slip[wheel], 0.0);
    EXPECT_DOUBLE_EQ(output.lateral_slip[wheel], 0.0);
  }
  // No wheel force is generated, but PhysX scene-level Rigidbody drag still
  // applies once per fixed step to an airborne body.
  constexpr double rigidbody_drag_per_s = 0.273;
  EXPECT_NEAR(output.longitudinal_acceleration_mps2,
              -rigidbody_drag_per_s * state.u_mps, 1.0e-9);
  EXPECT_NEAR(output.lateral_acceleration_mps2, 0.0, 1.0e-12);
  EXPECT_NEAR(output.yaw_acceleration_radps2, 0.0, 1.0e-12);
}

TEST(UnitySourcePlant, BrakeLockedWheelReportsZeroForwardSlipAtLowSpeed)
{
  auto state = state_at_speed(0.15);
  state.wheel_omega_radps.fill(0.0);
  state.suspension_history_initialized = true;
  f1tenth_mpc::UnitySourcePlantOutput output;
  ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step_1ms(
      state, {0.0, 0.0}, &output));
  for (std::size_t wheel = 0; wheel < 4; ++wheel) {
    EXPECT_DOUBLE_EQ(output.state.wheel_omega_radps[wheel], 0.0);
    // At this source-defined low speed, locked wheels and sticky-tire
    // handling produce zero longitudinal slip under hard braking.
    EXPECT_DOUBLE_EQ(output.longitudinal_slip[wheel], 0.0);
    EXPECT_DOUBLE_EQ(output.tire_longitudinal_force_n[wheel], 0.0);
  }
}

TEST(UnitySourcePlant, StickyForwardConstraintUsesSourceSpeedBoundary)
{
  auto above = state_at_speed(0.2005);
  above.wheel_omega_radps.fill(0.0);
  above.suspension_history_initialized = true;
  f1tenth_mpc::UnitySourcePlantOutput above_output;
  ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step_1ms(
      above, {0.0, 0.0}, &above_output));
  for (bool active : above_output.sticky_forward_active)
    EXPECT_FALSE(active);

  auto below = state_at_speed(0.1995);
  below.wheel_omega_radps.fill(0.0);
  below.suspension_history_initialized = true;
  f1tenth_mpc::UnitySourcePlantOutput below_output;
  ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step_1ms(
      below, {0.0, 0.0}, &below_output));
  for (bool active : below_output.sticky_forward_active)
    EXPECT_TRUE(active);
}

TEST(UnitySourcePlant, StraightLockedWheelBrakeDoesNotCreateYawImpulse)
{
  auto state = state_at_speed(0.15);
  state.wheel_omega_radps.fill(0.0);
  state.suspension_history_initialized = true;
  f1tenth_mpc::UnitySourcePlantOutput output;
  ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step_1ms(
      state, {0.0, 0.0}, &output));
  EXPECT_LT(std::abs(output.state.yaw_rate_radps), 3.0e-3);
}

TEST(UnitySourcePlant, LowSideTimerResetsWhenDriverIntendsToAccelerate)
{
  auto state = state_at_speed(0.0);
  state.wheel_omega_radps.fill(0.0);
  state.low_side_speed_time_s.fill(0.5);
  f1tenth_mpc::UnitySourcePlantOutput output;
  ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step_1ms(
      state, {0.0, 0.2}, &output));
  for (std::size_t wheel = 0; wheel < 4; ++wheel) {
    EXPECT_DOUBLE_EQ(output.state.low_forward_speed_time_s[wheel], 0.0);
    EXPECT_DOUBLE_EQ(output.state.low_side_speed_time_s[wheel], 0.0);
  }
}

TEST(UnitySourcePlant, StickyTireTimersSurviveMpcInternalStateRoundTrip)
{
  std::array<double, UNITY_SOURCE_PLANT_STATE_SIZE> state{};
  state[UNITY_SOURCE_U] = 0.1;
  state[UNITY_SOURCE_BODY_HEIGHT] = -0.025;
  state[UNITY_SOURCE_LOW_FORWARD_SPEED_TIME_RL] = 0.8;
  state[UNITY_SOURCE_LOW_SIDE_SPEED_TIME_RR] = 0.9;
  UnitySourcePlantInputC input{};
  std::copy(state.begin(), state.end(), input.state);
  input.suspension_history_initialized = 1;
  input.throttle_normalized = 0.0;
  UnitySourcePlantOutputC output{};
  ASSERT_EQ(unity_source_plant_step_c(&input, 0.001, &output), 1);
  EXPECT_GT(output.state[UNITY_SOURCE_LOW_FORWARD_SPEED_TIME_RL], 0.8);
  EXPECT_GT(output.state[UNITY_SOURCE_LOW_SIDE_SPEED_TIME_RR], 0.9);
}

TEST(UnitySourcePlant, LowSpeedTireForceYawMirrorsWithSteeringDirection)
{
  // One 1 ms update only changes steering by 0.0032 rad. Use one captured
  // 25 ms control interval so the source steering actuator has an observable
  // input before comparing the mirrored yaw response.
  auto left = state_at_speed(0.12);
  left.wheel_omega_radps.fill(0.0);
  left.physical_steering_rad = 0.15;
  left.suspension_history_initialized = true;
  auto right = left;
  right.physical_steering_rad = -left.physical_steering_rad;

  f1tenth_mpc::UnitySourcePlantOutput left_output, right_output;
  ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step(
      left, {left.physical_steering_rad, 0.0}, 0.025, &left_output));
  ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step(
      right, {right.physical_steering_rad, 0.0}, 0.025, &right_output));
  EXPECT_GT(std::abs(left_output.state.yaw_rate_radps), 1.0e-4);
  // The practice player's measured inertiaTensorRotation is non-identity;
  // mirrored forces are not expected to have identical magnitudes after the
  // scene solver applies the rotated mass-frame tensor. Steering reversal
  // must still reverse the yaw response.
  EXPECT_LT(left_output.state.yaw_rate_radps *
                right_output.state.yaw_rate_radps,
            0.0);
}

TEST(UnitySourcePlant, LeftWheelSupportProducesPositiveSourceFrameRoll)
{
  auto state = state_at_speed(5.0);
  state.suspension_history_initialized = true;
  const double front_jounce = 0.025 -
      (state.body_height_m + 0.08490486745229364 - 0.059);
  const double rear_jounce = 0.025 -
      (state.body_height_m + 0.075 - 0.059);
  state.suspension_jounce_m = {
      front_jounce - 0.001, front_jounce,
      rear_jounce - 0.001, rear_jounce};
  f1tenth_mpc::UnitySourcePlantOutput output;
  ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step_1ms(
      state, {0.0, 0.2}, &output));
  EXPECT_GT(output.state.roll_rate_radps, state.roll_rate_radps);
}

TEST(UnitySourcePlant, PxVehicleSubstepUsesPrincipalInertiaWithoutGyroTerm)
{
  f1tenth_mpc::UnitySourcePlantState state;
  state.u_mps = 8.0;  // above 5 m/s: native schedule selects one step
  state.roll_rate_radps = 1.0;
  state.v_mps = state.roll_rate_radps *
      -(state.body_height_m + 0.06434);
  for (double &omega : state.wheel_omega_radps) omega = state.u_mps / 0.059;
  f1tenth_mpc::UnitySourcePlantOutput output;
  ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step_1ms(
      state, {0.0, 0.0}, &output));
  const double force_derived_yaw_acceleration =
      expected_yaw_acceleration_from_wheel_forces(output);
  const double expected_rate =
      (state.yaw_rate_radps + 0.001 * force_derived_yaw_acceleration) *
      (1.0 - 0.1 * 0.001);
  EXPECT_NEAR(output.state.yaw_rate_radps, expected_rate, 1.0e-10);
  EXPECT_NEAR(output.yaw_acceleration_radps2,
      (output.state.yaw_rate_radps - state.yaw_rate_radps) / 0.001,
      1.0e-10);
  EXPECT_NEAR(force_derived_yaw_acceleration, 0.0, 1.0e-12);
}

TEST(UnitySourcePlant, RigidbodyAngularUpdateUsesTorqueThenSceneDamping)
{
  auto state = state_at_speed(8.0);  // above 5 m/s: one vehicle substep
  state.yaw_rate_radps = 0.75;
  state.suspension_history_initialized = true;

  f1tenth_mpc::UnitySourcePlantOutput output;
  ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step_1ms(
      state, {0.0, 0.2}, &output));  // drive intent disables sticky rows

  // PxVehicle integrates angular velocity from the wheel-force torque first;
  // PhysX scene pre-integration then applies Rigidbody angular damping before
  // constraints. At 8 m/s the model uses one vehicle substep, and drive intent
  // disables sticky rows, so this checks the exact ordered update.
  EXPECT_NEAR(output.yaw_acceleration_radps2,
      (output.state.yaw_rate_radps - state.yaw_rate_radps) / 0.001,
      1.0e-12);
}

TEST(UnitySourcePlant, GyroBodyRatesAdvanceTiltUsingEulerKinematics)
{
  auto state = state_at_speed(8.0);  // above 5 m/s: one native vehicle substep
  state.roll_rad = 0.20;
  state.pitch_rad = -0.15;
  state.roll_rate_radps = 0.30;
  state.pitch_rate_radps = -0.40;
  state.yaw_rate_radps = 2.20;
  state.physical_steering_rad = 0.10;

  f1tenth_mpc::UnitySourcePlantOutput output;
  ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step_1ms(
      state, {state.physical_steering_rad, 0.0}, &output));

  // The production plant applies Rigidbody angular damping after the
  // substep's pose integration. Undo that final 1 ms damping to recover the
  // body rates used by the semi-implicit pose update.
  constexpr double dt = 0.001;
  const double undamped_p = output.state.roll_rate_radps / (1.0 - 0.1 * dt);
  const double undamped_q = output.state.pitch_rate_radps / (1.0 - 0.1 * dt);
  const double undamped_r = output.state.yaw_rate_radps / (1.0 - 0.1 * dt);
  const double expected_roll_rate = undamped_p +
      std::tan(state.pitch_rad) *
          (std::sin(state.roll_rad) * undamped_q +
           std::cos(state.roll_rad) * undamped_r);
  const double expected_pitch_rate =
      std::cos(state.roll_rad) * undamped_q -
      std::sin(state.roll_rad) * undamped_r;

  EXPECT_NEAR((output.state.roll_rad - state.roll_rad) / dt,
              expected_roll_rate, 2.0e-6);
  EXPECT_NEAR((output.state.pitch_rad - state.pitch_rad) / dt,
              expected_pitch_rate, 2.0e-6);
  EXPECT_GT(std::abs(expected_roll_rate - undamped_p), 0.05);
  EXPECT_GT(std::abs(expected_pitch_rate - undamped_q), 0.05);
}

TEST(UnitySourcePlant, ThrottleChangesWheelSpinAndLongitudinalSlip)
{
  const auto state = state_at_speed(5.0);
  f1tenth_mpc::UnitySourcePlantOutput output;
  ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step(
      state, {0.0, 0.35}, 0.025, &output));
  EXPECT_GT(output.state.wheel_omega_radps[2], state.wheel_omega_radps[2]);
  EXPECT_GT(output.longitudinal_slip[2], 0.0);
  EXPECT_GT(output.longitudinal_acceleration_mps2, -2.0);
}

TEST(UnitySourcePlant, WheelSpinUsesPhysXNoDriveImplicitUpdate)
{
  // At 8 m/s PxVehicleNoDrive selects one vehicle substep per 1 ms update.
  // Check the production wheel state against integrateNoDriveWheelSpeeds:
  // (omega + dt/I * (tireTorque + driveTorque)) /
  // (1 + dampingRate * dt/I).
  constexpr double wheel_radius_m = 0.059;
  constexpr double wheel_mass_kg = 0.109;
  constexpr double wheel_moi_kgm2 =
      0.5 * wheel_mass_kg * wheel_radius_m * wheel_radius_m;
  constexpr double wheel_damping_rate = 0.25;
  constexpr double motor_torque_nm = 428.0;
  constexpr double throttle = 0.2;
  constexpr double dt = 0.001;
  const auto state = state_at_speed(8.0);
  f1tenth_mpc::UnitySourcePlantOutput output;
  ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step_1ms(
      state, {0.0, throttle}, &output));

  const double tire_torque_nm =
      -wheel_radius_m * output.tire_longitudinal_force_n[2];
  const double per_wheel_drive_torque_nm =
      motor_torque_nm * throttle / 4.0;
  const double dt_over_moi = dt / wheel_moi_kgm2;
  const double expected_omega =
      (state.wheel_omega_radps[2] + dt_over_moi *
          (tire_torque_nm + per_wheel_drive_torque_nm)) /
      (1.0 + wheel_damping_rate * dt_over_moi);
  EXPECT_NEAR(output.state.wheel_omega_radps[2], expected_omega, 1.0e-10);
}

TEST(UnitySourcePlant, ZeroThrottleUsesFullBrakeTorque)
{
  auto state = state_at_speed(5.0);
  f1tenth_mpc::UnitySourcePlantOutput output;
  ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step_1ms(
      state, {0.0, 0.0}, &output));
  for (double omega : output.state.wheel_omega_radps)
    EXPECT_NEAR(omega, 0.0, 1.0e-12);
}

TEST(UnitySourcePlant, HardBrakeReverseTransientStaysInsidePlantSweepBound)
{
  constexpr double speeds_mps[] = {0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0};
  double worst_reverse_mps = 0.0;
  double worst_start_speed_mps = 0.0;
  int worst_tick = -1;
  for (double speed : speeds_mps) {
    auto state = state_at_speed(speed);
    for (int tick = 0; tick < 3000; ++tick) {
      f1tenth_mpc::UnitySourcePlantOutput output;
      ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step_1ms(
          state, {0.0, 0.0}, &output));
      state = output.state;
      if (state.u_mps < worst_reverse_mps) {
        worst_reverse_mps = state.u_mps;
        worst_start_speed_mps = speed;
        worst_tick = tick;
      }
      if (tick > 250 && state.u_mps == 0.0) break;
    }
  }
  EXPECT_GE(worst_reverse_mps, -MPC_PHYSX_BRAKE_REVERSE_TRANSIENT_MPS)
      << "native 1 ms PhysX sweep reached -0.00779 m/s over 0.5--12 m/s; "
      << "plant minimum occurred from " << worst_start_speed_mps
      << " m/s at 1 ms tick " << worst_tick;
}

TEST(UnitySourcePlant, SteeringGeneratesLateralForceAndYaw)
{
  const auto state = state_at_speed(5.0);
  f1tenth_mpc::UnitySourcePlantOutput output;
  ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step(
      state, {0.25, 0.2}, 0.025, &output));
  EXPECT_GT(output.lateral_acceleration_mps2, 0.0);
  EXPECT_GT(output.yaw_acceleration_radps2, 0.0);
  // Positive-left steering gives negative lateral contact velocity in the
  // wheel's local right/forward convention. WheelHit reports the opposite
  // sign, while the force still opposes the actual contact velocity.
  EXPECT_GT(output.lateral_slip[0], 0.0);
  EXPECT_GT(output.tire_lateral_force_n[0], 0.0);
  EXPECT_NEAR(output.state.physical_steering_rad,
              0.025 * 3.2, 2.0e-4);

  f1tenth_mpc::UnitySourcePlantOutput reverse_turn;
  ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step(
      state, {-0.25, 0.2}, 0.025, &reverse_turn));
  EXPECT_LT(reverse_turn.lateral_acceleration_mps2, 0.0);
  EXPECT_LT(reverse_turn.yaw_acceleration_radps2, 0.0);
  EXPECT_LT(reverse_turn.lateral_slip[0], 0.0);
  EXPECT_LT(reverse_turn.tire_lateral_force_n[0], 0.0);
  EXPECT_NEAR(reverse_turn.state.physical_steering_rad,
              -0.025 * 3.2, 2.0e-4);
}

TEST(UnitySourcePlant, SteeringPreservesVehicleControllerTargetCrossingClamp)
{
  constexpr double max_steering_rad = 0.52359877559829887308;
  auto state = state_at_speed(0.0);
  state.physical_steering_rad = 0.4;

  f1tenth_mpc::UnitySourcePlantOutput output;
  ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step_1ms(
      state, {0.1 * max_steering_rad, 0.2}, &output));

  // VehicleController.Steer compares AutonomousSteering with the internal
  // angle's opposite-sign normalized value, then clamps after its selected
  // direction step. On this unwind case the source reaches the target in one
  // 1 ms update; replacing it with a symmetric rate limiter is not source
  // equivalent.
  EXPECT_NEAR(output.state.physical_steering_rad,
              0.1 * max_steering_rad, 1.0e-6);
}

TEST(UnitySourcePlant, OppositeSteeringTargetHasFiniteSourceBranchJump)
{
  // Unity Steer()
  // compares command_norm to the opposite-sign internal angle, then clamps
  // against -command_norm*30. These two nearby legal commands therefore
  // select different actuator modes. This is not smooth servo curvature.
  auto state = state_at_speed(2.4122047424316406);
  state.physical_steering_rad = 0.39126089215278625;
  constexpr double lower = -0.39126095175743103;
  constexpr double upper = -0.39126086235046387;
  f1tenth_mpc::UnitySourcePlantOutput a, b;
  ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step_1ms(
      state, {lower, 0.1}, &a));
  ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step_1ms(
      state, {upper, 0.1}, &b));
  EXPECT_LT(upper - lower, 1.e-7);
  EXPECT_GT(a.state.physical_steering_rad, 0.38);
  EXPECT_NEAR(b.state.physical_steering_rad, upper, 1.e-12);
  EXPECT_GT(a.state.physical_steering_rad - b.state.physical_steering_rad, 0.77);
}

TEST(UnitySourcePlant, InertialAccelerationIncludesBodyFrameYawTransport)
{
  auto state = state_at_speed(3.5);
  state.v_mps = -0.35;
  state.yaw_rate_radps = -2.4;
  f1tenth_mpc::UnitySourcePlantOutput output;
  ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step_1ms(
      state, {0.0, 0.15}, &output));

  const double expected_lateral_acceleration =
      (output.state.v_mps - state.v_mps) / 0.001 +
      0.5 * (state.yaw_rate_radps + output.state.yaw_rate_radps) *
          0.5 * (state.u_mps + output.state.u_mps);
  EXPECT_NEAR(output.lateral_acceleration_mps2,
              expected_lateral_acceleration, 1.0e-5);
  const double body_coordinate_derivative =
      (output.state.v_mps - state.v_mps) / 0.001;
  EXPECT_GT(std::abs(output.lateral_acceleration_mps2 -
                     body_coordinate_derivative), 5.0);
}

TEST(UnitySourcePlant, NormalizedCommandUsesPhysicalSteeringSign)
{
  constexpr double max_steering_rad = 0.52359877559829887308;
  const double positive_target =
      f1tenth_mpc::unity_steering_command_normalized_to_physical_rad(
          0.5, max_steering_rad);
  const double negative_target =
      f1tenth_mpc::unity_steering_command_normalized_to_physical_rad(
          -0.5, max_steering_rad);
  EXPECT_GT(positive_target, 0.0);
  EXPECT_LT(negative_target, 0.0);
  EXPECT_DOUBLE_EQ(positive_target, -negative_target);

  const auto initial = state_at_speed(5.0);
  f1tenth_mpc::UnitySourcePlantOutput left, right;
  ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step(
      initial, {positive_target, 0.2}, 0.025, &left));
  ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step(
      initial, {negative_target, 0.2}, 0.025, &right));
  EXPECT_GT(left.state.physical_steering_rad, 0.0);
  EXPECT_LT(right.state.physical_steering_rad, 0.0);
  EXPECT_GT(left.yaw_acceleration_radps2, 0.0);
  EXPECT_LT(right.yaw_acceleration_radps2, 0.0);
}

TEST(UnitySourcePlant, LateralSlipUsesPhysXToleranceScaleFloor)
{
  auto state = state_at_speed(8.0);  // isolate one substep's input slip
  state.v_mps = 0.24;
  state.yaw_rate_radps = 0.55;
  state.physical_steering_rad = 0.18;

  f1tenth_mpc::UnitySourcePlantOutput output;
  ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step_1ms(
      state, {0.18, 0.25}, &output));

  const double wheel_forward = output.contact_forward_speed_mps[0];
  const double wheel_lateral = output.contact_lateral_speed_mps[0];
  const double source_slip = std::atan(
      wheel_lateral / (std::abs(wheel_forward) + 1.0));
  EXPECT_NEAR(output.lateral_slip[0], -source_slip, 1.0e-12);
}

TEST(UnitySourcePlant, WheelSlipUsesFullAngularVelocityAtContactPoint)
{
  constexpr double wheel_radius = 0.059;
  constexpr double com_height_from_actor = 0.06434;
  constexpr double forward_speed = 8.0;  // above 5 m/s: one native substep
  auto state = state_at_speed(forward_speed);
  const double contact_z = -(state.body_height_m + com_height_from_actor);
  state.roll_rate_radps = 0.4;
  f1tenth_mpc::UnitySourcePlantOutput roll_output;
  ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step_1ms(
      state, {0.0, 0.25}, &roll_output));
  const double roll_contact_lateral_speed =
      -state.roll_rate_radps * contact_z;
  const double roll_contact_forward_speed =
      std::abs(roll_output.contact_forward_speed_mps[0]);
  EXPECT_NEAR(roll_output.lateral_slip[0],
      -std::atan(roll_contact_lateral_speed /
                 (roll_contact_forward_speed + 1.0)), 1.0e-10);

  state = state_at_speed(forward_speed);
  const double pitch_contact_z =
      -(state.body_height_m + com_height_from_actor);
  state.pitch_rate_radps = 0.4;
  f1tenth_mpc::UnitySourcePlantOutput pitch_output;
  ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step_1ms(
      state, {0.0, 0.25}, &pitch_output));
  const double pitch_contact_forward_speed =
      forward_speed + state.pitch_rate_radps * pitch_contact_z;
  const double pitch_contact_forward_abs =
      std::abs(pitch_output.contact_forward_speed_mps[0]);
  const double expected_longitudinal_slip =
      (forward_speed - pitch_contact_forward_speed) /
      (std::max(forward_speed, pitch_contact_forward_abs) + 0.1);
  EXPECT_NEAR(pitch_output.longitudinal_slip[0],
              expected_longitudinal_slip, 1.0e-10);
  EXPECT_NEAR(state.wheel_omega_radps[0] * wheel_radius,
              forward_speed, 1.0e-12);
}

TEST(UnitySourcePlant, ContactJounceUsesPhysXRayPlaneIntersection)
{
  auto state = state_at_speed(8.0);  // one native substep; ray is initial
  state.body_height_m = -0.025;
  state.roll_rad = 0.035;
  state.pitch_rad = -0.045;
  const double sr = std::sin(state.roll_rad), cr = std::cos(state.roll_rad);
  const double sp = std::sin(state.pitch_rad), cp = std::cos(state.pitch_rad);
  const std::array<double, 3> normal_body{-sp, cp * sr, cp * cr};
  // UnityPlayer::AddWheelToExistingVehicle sets chassis-local (0,-1,0)
  // suspension travel for every wheel; front camber only transforms center.
  const std::array<double, 3> suspension_direction{0.0, 0.0, -1.0};
  const std::array<double, 3> collider_transform{
      0.17467999458312988, 0.11792,
      -0.00434};
  constexpr double center_y = -0.025 * 0.08715574274765817;
  constexpr double center_z = 0.025 * 0.9961946980917455;
  std::array<double, 3> wheel_center{};
  std::array<double, 3> ray_start{};
  for (std::size_t axis = 0; axis < 3; ++axis) {
    const double center_offset = axis == 1 ? center_y :
        (axis == 2 ? center_z : 0.0);
    // UnityPlayer::ComputeWheelCentreOfMassOffset adds
    // suspensionDir * distance * (1-targetPosition). With target=0.5 this is
    // one max-compression interval along native travel.
    wheel_center[axis] = collider_transform[axis] + center_offset +
        suspension_direction[axis] * 0.025;
    ray_start[axis] = wheel_center[axis] -
        suspension_direction[axis] * (0.059 + 0.025);
  }
  const std::array<double, 3> com_offset{-0.00468, 0.00008, 0.06434};
  const auto dot3 = [](const std::array<double, 3> &a,
                       const std::array<double, 3> &b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  };
  const double ray_travel = -(state.body_height_m +
      dot3(normal_body, com_offset) + dot3(normal_body, ray_start)) /
      dot3(normal_body, suspension_direction);
  const double expected_jounce = std::clamp(
      2.0 * 0.059 + 0.025 - ray_travel, -0.025, 0.025);

  f1tenth_mpc::UnitySourcePlantOutput output;
  ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step_1ms(
      state, {0.0, 0.25}, &output));
  EXPECT_NEAR(output.state.suspension_jounce_m[0], expected_jounce, 1.0e-10);
}

TEST(UnitySourcePlant, TireAndStickyPhysicsUseSameWheelBottomKinematics)
{
  auto state = state_at_speed(3.5);
  state.body_height_m = -0.025;
  state.v_mps = 0.12;
  state.yaw_rate_radps = -0.2;
  state.physical_steering_rad = -0.25;
  f1tenth_mpc::UnitySourcePlantOutput output;
  ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step_1ms(
      state, {0.0, 0.0}, &output));
  for (std::size_t wheel = 0; wheel < 4; ++wheel) {
    EXPECT_DOUBLE_EQ(output.contact_forward_speed_mps[wheel],
                     output.sticky_contact_forward_speed_mps[wheel]);
    EXPECT_DOUBLE_EQ(output.contact_lateral_speed_mps[wheel],
                     output.sticky_contact_lateral_speed_mps[wheel]);
  }
}

TEST(UnitySourcePlant, UnityPlayerTireForceCallbackAppliesCombinedSlipCircle)
{
  auto state = state_at_speed(4.0);
  state.v_mps = 0.7;
  state.yaw_rate_radps = 0.45;
  state.physical_steering_rad = 0.21;
  for (double &omega : state.wheel_omega_radps)
    omega += 2.2;

  f1tenth_mpc::UnitySourcePlantOutput output;
  ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step_1ms(
      state, {0.4, 0.25}, &output));
  const std::array<double, 5> forward{0.15, 0.9, 0.25, 0.58, 0.8};
  const std::array<double, 5> lateral{0.01, 1.0, 0.1, 0.5, 1.0};
  for (std::size_t i = 0; i < 4; ++i) {
    const double raw_x = raw_curve(output.longitudinal_slip[i], forward);
    const double force_model_slip_y = -output.lateral_slip[i];
    const double raw_y = raw_curve(force_model_slip_y, lateral);
    const double utilization = std::sqrt(std::max(
        1.0, std::pow(raw_x / forward[1], 2.0) +
                 std::pow(raw_y / lateral[1], 2.0)));
    EXPECT_NEAR(output.tire_longitudinal_force_n[i],
        output.normal_load_n[i] * raw_x * forward[4] /
            utilization, 1.0e-10);
    EXPECT_NEAR(output.tire_lateral_force_n[i],
        -output.normal_load_n[i] * raw_y * lateral[4] /
            utilization, 1.0e-10);
  }
}

TEST(UnitySourcePlant, SlowCoordinateHessianMatchesWideGradientAndIsSymmetric)
{
  using namespace f1tenth_mpc;
  auto state=state_at_speed(6.2);
  state.v_mps=.14;state.yaw_rate_radps=.72;state.physical_steering_rad=.24;
  const UnitySourcePlantInput input{.4,.3};
  // Hessian inputs: u, v, yaw rate, steering target, throttle. Outputs: the
  // next u, v, yaw rate and rack, body displacement, and the four slip maxima.
  constexpr int input_coordinate[3]={UNITY_SOURCE_U,UNITY_SOURCE_V,UNITY_SOURCE_YAW_RATE};
  constexpr int output_coordinate[4]={UNITY_SOURCE_U,UNITY_SOURCE_V,UNITY_SOURCE_YAW_RATE,UNITY_SOURCE_PHYSICAL_STEERING};
  UnitySourcePlantSensitivitySeeds seeds;
  for(int i=0;i<3;++i)seeds.state[input_coordinate[i]][i]=1.;
  seeds.steering_target[3]=1.;seeds.throttle[4]=1.;
  for(double dt:{.001,.025}) {
    auto h=std::make_unique<UnitySourcePlantHessian>();
    ASSERT_TRUE(unity_source_plant_step_with_hessian(state,input,dt,h.get()));
    UnitySourcePlantSensitivity g;
    ASSERT_TRUE(unity_source_plant_step_with_sensitivities(state,input,dt,seeds,&g));
    for(std::size_t o=0;o<kUnityPlantHessianOutputs;++o)for(std::size_t i=0;i<kUnityPlantHessianInputs;++i) {
      const double reference=o<4?g.state[output_coordinate[o]][i]:o<7?
          (o==4?g.body_forward_displacement[i]:o==5?g.body_lateral_displacement[i]:g.yaw_delta[i]):
          g.maximum_grip_loss_derivative[o-7][i];
      EXPECT_NEAR(h->gradient[o][i],reference,2.e-6*std::max(1.,std::abs(reference)));
      for(std::size_t j=0;j<kUnityPlantHessianInputs;++j)EXPECT_DOUBLE_EQ(h->hessian[o][i][j],h->hessian[o][j][i]);
    }


    for (std::size_t j = 0; j < kUnityPlantHessianInputs; ++j) {
      constexpr double epsilon = 1.e-6;
      auto plus_state = state, minus_state = state;
      auto plus_input = input, minus_input = input;
      if (j < 3) {
        *unity_source_plant_state_coordinates(plus_state)[input_coordinate[j]] += epsilon;
        *unity_source_plant_state_coordinates(minus_state)[input_coordinate[j]] -= epsilon;
      } else if (j == 3) {
        plus_input.steering_target_rad += epsilon;
        minus_input.steering_target_rad -= epsilon;
      } else {
        plus_input.throttle_normalized += epsilon;
        minus_input.throttle_normalized -= epsilon;
      }
      // The Hessian is the derivative of the Hessian type's own double-precision
      // gradient; the single-precision wide gradient is checked above.
      auto plus=std::make_unique<UnitySourcePlantHessian>(), minus=std::make_unique<UnitySourcePlantHessian>();
      ASSERT_TRUE(unity_source_plant_step_with_hessian(plus_state,plus_input,dt,plus.get()));
      ASSERT_TRUE(unity_source_plant_step_with_hessian(minus_state,minus_input,dt,minus.get()));
      for (std::size_t o = 0; o < kUnityPlantHessianOutputs; ++o)
        for (std::size_t i = 0; i < kUnityPlantHessianInputs; ++i) {
          const double reference = (plus->gradient[o][i]-minus->gradient[o][i])/(2.*epsilon);
          EXPECT_NEAR(h->hessian[o][i][j],reference,1.e-5*std::max(1.,std::abs(reference)))
              << "output " << o << " inputs " << i << ',' << j << " dt " << dt;
        }
    }
  }
}

TEST(UnitySourcePlant, ActiveTorqueSlipExpressionsMatchPublicPhysX)
{
  auto state=state_at_speed(6.);
  state.v_mps=.3;
  const std::array<double,4> ratios{.5,.84,1.,1.2};
  for(int w=0;w<4;++w)state.wheel_omega_radps[w]=ratios[w]*state.u_mps/.059;
  f1tenth_mpc::UnitySourcePlantOutput output;
  ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step_1ms(state,{.2,.3},&output));
  // PxVehicleUpdate.cpp::computeTireSlips, non-tank with applied torque;
  // setVehicleToleranceScale sets length=1, hence offsets .1 and 1 m/s.
  for(int w=0;w<4;++w) {
    const double forward=output.contact_forward_speed_mps[w];
    const double surface=state.wheel_omega_radps[w]*.059;
    const double expected=(surface-forward)/(std::max(std::abs(forward),std::abs(surface))+.1);
    EXPECT_NEAR(output.longitudinal_slip[w],expected,1.e-12);
    EXPECT_NEAR(output.lateral_slip[w],
        -std::atan(output.contact_lateral_speed_mps[w]/(std::abs(forward)+1.)),1.e-12);
  }
}

TEST(UnitySourcePlant, TireLoadFilterMatchesPublicPhysXInAllThreeRegions)
{
  // PxVehicleTireLoadFilterData defaults and
  // PxVehicleUpdate.cpp::computeFilteredNormalisedTireLoad.
  constexpr std::array<double,4> sprung{.817194700124942,.8160186332083913,
      .9189924292167562,.9177942374499102};
  std::array<bool,3> covered{};
  for(double prior_jounce:{-.02,0.,.02}) {
    auto state=state_at_speed(6.);
    state.suspension_history_initialized=true;
    state.suspension_jounce_m.fill(prior_jounce);
    if(prior_jounce==0.) {ASSERT_TRUE(f1tenth_mpc::unity_source_plant_reconcile_suspension_jounce(&state));}
    f1tenth_mpc::UnitySourcePlantOutput output;
    ASSERT_TRUE(f1tenth_mpc::unity_source_plant_step_1ms(state,{0.,.3},&output));
    for(int w=0;w<4;++w) {
      const double rest=(sprung[w]+.109)*9.81;
      const double normalized=output.suspension_force_n[w]/rest;
      const int region=normalized<=0.?0:normalized>=3.?2:1;
      covered[region]=true;
      const double filtered=region==0?.2308:region==2?3.:.2308+normalized*(3.-.2308)/3.;
      EXPECT_NEAR(output.normal_load_n[w],rest*filtered,1.e-11);
    }
  }
  for(bool observed:covered)EXPECT_TRUE(observed);
}

namespace f1tenth_mpc {
namespace {

TEST(UnitySourcePlant, HessianDivisionUsesDirectPrimalAndExplicitDerivatives)
{
  EXPECT_EQ((SourceHess<5>(10.)/3.).value,10./3.);
  EXPECT_EQ((SourceHess<5>(10.)/SourceHess<5>(3.)).value,10./3.);
  EXPECT_EQ((10./SourceHess<5>(3.)).value,10./3.);
  for (double numerator : {10.,-2.3,-0.,0.,1.e100})
    for (double denominator : {3.,-3.,.7})
      for (unsigned char numerator_order : {0,1,2})
        for (unsigned char denominator_order : {0,1,2}) {
          SourceHess<5> a(numerator), b(denominator);
          a.order=numerator_order; b.order=denominator_order;
          const double lanes[]={-.73,.19,0.,-0.,.13};
          for (std::size_t i=0;i<5;++i) {
            if (a.order > 0) a.grad[i]=lanes[i];
            if (b.order > 0) b.grad[i]=lanes[4-i];
          }
          for (std::size_t p=0;p<a.kPairs;++p) {
            if (a.order == 2) a.hess[p]=lanes[p%5];
            if (b.order == 2) b.hess[p]=lanes[4-p%5];
          }
          const double inverse=1./b.value;
          SourceHess<5> expected(a.value/b.value);
          if (b.order == 0) {
            expected=a;
            expected.value=a.value/b.value;
            if (a.order > 0) for (auto &g : expected.grad) g*=inverse;
            if (a.order == 2) for (auto &h : expected.hess) h*=inverse;
          } else {
            const auto reciprocal=explicit_chain(b,inverse,-inverse*inverse,2.*inverse*inverse*inverse);
            expected.order=2;
            if (a.order == 0) {
              for (std::size_t i=0;i<5;++i) expected.grad[i]=reciprocal.grad[i]*a.value;
              for (std::size_t p=0;p<a.kPairs;++p) expected.hess[p]=reciprocal.hess[p]*a.value;
            } else {
              for (std::size_t i=0;i<5;++i)
                expected.grad[i]=a.grad[i]*reciprocal.value+a.value*reciprocal.grad[i];
              const double av=a.order==2?reciprocal.value:0., bv=reciprocal.order==2?a.value:0.;
              std::size_t p=0;
              for (std::size_t i=0;i<5;++i) for (std::size_t j=i;j<5;++j,++p)
                expected.hess[p]=(a.order==2?a.hess[p]*av:0.)+
                    (reciprocal.order==2?reciprocal.hess[p]*bv:0.)+
                    a.grad[i]*reciprocal.grad[j]+a.grad[j]*reciprocal.grad[i];
            }
          }
          expect_scalar_exact(a/b,expected);
          expect_scalar_exact((a/b).value,(SourceDual<1>(numerator)/SourceDual<1>(denominator)).value);
          auto scalar_expected=a;
          scalar_expected.value=a.value/denominator;
          if (a.order > 0) for (auto &g : scalar_expected.grad) g*=inverse;
          if (a.order == 2) for (auto &h : scalar_expected.hess) h*=inverse;
          expect_scalar_exact(a/denominator,scalar_expected);
          const auto left_expected=explicit_chain(b,numerator/b.value,
              -numerator*inverse*inverse,2.*numerator*inverse*inverse*inverse);
          expect_scalar_exact(numerator/b,left_expected);
        }
}

TEST(UnitySourcePlant, HessianPrimalMatchesScalarPlantAcrossSubstepBranches)
{
  for (double speed : {1.01,4.99,5.,5.01,8.})
    for (double throttle : {0.,.001,.2,1.}) {
      auto state=state_at_speed(speed);
      state.v_mps=.1; state.yaw_rate_radps=-.4;
      state.roll_rad=.02; state.pitch_rad=-.015;
      const UnitySourcePlantInput input{.12,throttle};
      UnitySourcePlantOutput scalar;
      UnitySourcePlantHessian hessian;
      ASSERT_TRUE(unity_source_plant_step(state,input,.025,&scalar));
      ASSERT_TRUE(unity_source_plant_step_with_hessian(state,input,.025,&hessian));
      const double expected[]={scalar.state.u_mps,scalar.state.v_mps,scalar.state.yaw_rate_radps,
          scalar.state.physical_steering_rad,scalar.integrated_body_forward_displacement_m,
          scalar.integrated_body_lateral_displacement_m,scalar.integrated_yaw_delta_rad,
          scalar.maximum_grip_loss[0],scalar.maximum_grip_loss[1],scalar.maximum_grip_loss[2],scalar.maximum_grip_loss[3]};
      for (std::size_t i=0;i<kUnityPlantHessianOutputs;++i)
        expect_scalar_exact(hessian.value[i],expected[i]);
    }
}

}  // namespace
}  // namespace f1tenth_mpc
