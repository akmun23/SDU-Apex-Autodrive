// CasADi external ABI for the actual production 25 ms plant and its existing
// forward-mode derivatives. No PhysX executable/library or fitted equations.
#include "unity_source_plant_c.h"

#include <cmath>
#include <cstdint>

using casadi_int = long long;
namespace {
constexpr int kNx = 21;  // rear-axle XY/yaw + source state[0..17]
constexpr int kInput = 23;  // kNx + steering target + positive torque
const casadi_int kInputSparsity[]{kInput, 1, 1};
const casadi_int kOutputSparsity[]{kNx, 1, 1};
const casadi_int kJacobianSparsity[]{kNx, kInput, 1};
const casadi_int kPrefixInputSparsity[]{24, 1, 1};
const casadi_int kPrefixJacobianSparsity[]{kNx, 24, 1};

bool step(const double *x, UnitySourcePlantInputC *in,
          UnitySourcePlantOutputC *out, double duration)
{
  if (!(x[22] >= .001 && x[22] <= 1.)) return false;
  *in = {};
  for (int i = 0; i < 18; ++i) in->state[i] = x[i+3];
  // With q>0 the source resets both sticky histories; initialized zero is an
  // invariant. They are not lost physical degrees of freedom in this mode.
  in->steering_target_rad = x[21];
  in->throttle_normalized = x[22];
  in->suspension_history_initialized = 1;
  return unity_source_plant_step_c(in, duration, out) != 0;
}

int evaluate(const double **arg, double **res, double duration)
{
  UnitySourcePlantInputC in{};
  UnitySourcePlantOutputC out{};
  if (!step(arg[0], &in, &out, duration)) return 1;
  if (!res[0]) return 0;
  const double yaw = arg[0][2], c = std::cos(yaw), s = std::sin(yaw);
  const double dx = out.integrated_body_forward_displacement_m;
  const double dy = out.integrated_body_lateral_displacement_m;
  res[0][0] = arg[0][0]+c*dx-s*dy;
  res[0][1] = arg[0][1]+s*dx+c*dy;
  res[0][2] = yaw+out.integrated_yaw_delta_rad;
  for (int i = 0; i < 18; ++i) res[0][i+3] = out.state[i];
  return 0;
}

int jacobian(const double **arg, double **res, double duration, int columns)
{
  UnitySourcePlantInputC in{};
  UnitySourcePlantOutputC out{};
  if (!step(arg[0], &in, &out, duration)) return 1;
  UnitySourcePlantSensitivitySeedsC seeds{};
  UnitySourcePlantSensitivityC sensitivity{};
  for (int i = 0; i < 18; ++i) seeds.state[i][i] = 1.;
  seeds.steering_target[18] = 1.;
  seeds.throttle[19] = 1.;
  if (!unity_source_plant_step_sensitivity_c(&in, duration, &seeds, &sensitivity)) return 1;
  if (!res[0]) return 0;
  double *jac = res[0];
  for (int i = 0; i < kNx*columns; ++i) jac[i] = 0.;
  const double c = std::cos(arg[0][2]), s = std::sin(arg[0][2]);
  jac[0] = jac[kNx+1] = jac[2*kNx+2] = 1.;
  jac[2*kNx] = -s*out.integrated_body_forward_displacement_m
                         -c*out.integrated_body_lateral_displacement_m;
  jac[2*kNx+1] = c*out.integrated_body_forward_displacement_m
                         -s*out.integrated_body_lateral_displacement_m;
  for (int lane = 0; lane < 20; ++lane) {
    const int column = lane+3;
    jac[column*kNx] = c*sensitivity.body_forward_displacement[lane]
                            -s*sensitivity.body_lateral_displacement[lane];
    jac[column*kNx+1] = s*sensitivity.body_forward_displacement[lane]
                            +c*sensitivity.body_lateral_displacement[lane];
    jac[column*kNx+2] = sensitivity.yaw_delta[lane];
    for (int i = 0; i < 18; ++i)
      jac[column*kNx+i+3] = sensitivity.state[i][lane];
  }
  return 0;
}
}

extern "C" {
int sdu_command_step(const double **a, double **r, casadi_int *, double *, int)
{ return evaluate(a, r, .025); }
int jac_sdu_command_step(const double **a, double **r, casadi_int *, double *, int)
{ return jacobian(a, r, .025, kInput); }
casadi_int sdu_command_step_n_in() { return 1; }
casadi_int sdu_command_step_n_out() { return 1; }
const casadi_int *sdu_command_step_sparsity_in(casadi_int) { return kInputSparsity; }
const casadi_int *sdu_command_step_sparsity_out(casadi_int) { return kOutputSparsity; }
int sdu_command_step_work(casadi_int *a, casadi_int *r, casadi_int *i, casadi_int *w)
{ *a=1; *r=1; *i=0; *w=0; return 0; }
casadi_int jac_sdu_command_step_n_in() { return 2; }
casadi_int jac_sdu_command_step_n_out() { return 1; }
const casadi_int *jac_sdu_command_step_sparsity_in(casadi_int i)
{ return i == 0 ? kInputSparsity : kOutputSparsity; }
const casadi_int *jac_sdu_command_step_sparsity_out(casadi_int)
{ return kJacobianSparsity; }
int jac_sdu_command_step_work(casadi_int *a, casadi_int *r, casadi_int *i, casadi_int *w)
{ *a=2; *r=1; *i=0; *w=0; return 0; }

// Last input is a FIXED source-tick prefix, never an optimizer decision.
// Its zero tangent is valid only for callers binding that input to a literal.
int sdu_command_prefix(const double **a, double **r, casadi_int *, double *, int)
{ return evaluate(a, r, a[0][23]); }
int jac_sdu_command_prefix(const double **a, double **r, casadi_int *, double *, int)
{ return jacobian(a, r, a[0][23], 24); }
casadi_int sdu_command_prefix_n_in() { return 1; }
casadi_int sdu_command_prefix_n_out() { return 1; }
const casadi_int *sdu_command_prefix_sparsity_in(casadi_int) { return kPrefixInputSparsity; }
const casadi_int *sdu_command_prefix_sparsity_out(casadi_int) { return kOutputSparsity; }
int sdu_command_prefix_work(casadi_int *a, casadi_int *r, casadi_int *i, casadi_int *w)
{ *a=1; *r=1; *i=0; *w=0; return 0; }
casadi_int jac_sdu_command_prefix_n_in() { return 2; }
casadi_int jac_sdu_command_prefix_n_out() { return 1; }
const casadi_int *jac_sdu_command_prefix_sparsity_in(casadi_int i)
{ return i == 0 ? kPrefixInputSparsity : kOutputSparsity; }
const casadi_int *jac_sdu_command_prefix_sparsity_out(casadi_int) { return kPrefixJacobianSparsity; }
int jac_sdu_command_prefix_work(casadi_int *a, casadi_int *r, casadi_int *i, casadi_int *w)
{ *a=2; *r=1; *i=0; *w=0; return 0; }
}
