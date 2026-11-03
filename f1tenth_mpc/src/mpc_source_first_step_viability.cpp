#include "mpc_rti_internal.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

// Nonlinear physical admissibility of the NEXT published 25ms command.
// This is a local source-informed constraint of the normal Riccati QP,
// not a separate controller or an alternative publication path. Sampling
// evaluates the identical nonlinear source, map and command gate as the
// eventual full 30-stage validator. The sampled rectangle is conservative
// guidance; exact full-horizon validation remains mandatory.
int mpc_rti_source_first_step_feasible_box(
    const MpcRtiState_t *measured, double progress,
    const MpcTrajectorySample_t *trajectory, size_t count, double lap,
    float prediction_dt, const MpcRtiCycleConfiguration_t *cycle,
    const MpcModelControl_t *nominal_first, double deadline_us,
    float decision_lb[2], float decision_ub[2],
    int *source_checks)
{
    if(source_checks)*source_checks=0;
    if(!measured || !trajectory || count<3 || lap<=0. ||
       !cycle || !nominal_first || !decision_lb || !decision_ub ||
       !(prediction_dt>0.f) || !std::isfinite(progress))return 0;
    const auto &model=cycle->model;
    constexpr int ns=7, nt=5;
    constexpr double max_slip=0.10;  // Unity side-curve asymptote.
    constexpr double reserve_m=.012; // above original hard footprint.
    constexpr double activation_slip=.055;
    constexpr double activation_clearance_m=.060;
    // Convert the QP's first-stage bounds to the physical command axis. In
    // increment mode, the previous command is an actual optimizer state.
    const double previous[2]={
        measured->plant.delayed_steering_command_2,
        measured->plant.delayed_throttle_command_2};
    double low[2],high[2];
    for(int a=0;a<2;++a) {
        const double offset=model.command_increment_qp_enabled ? previous[a]:0.;
        low[a]=decision_lb[a]+offset;
        high[a]=decision_ub[a]+offset;
    }
    low[0]=std::max(low[0],-double(model.max_steering_rad));
    high[0]=std::min(high[0],double(model.max_steering_rad));
    low[1]=std::max(low[1],double(MPC_THROTTLE_MIN_DRIVE));
    high[1]=std::min(high[1],double(MPC_THROTTLE_MAX));
    if(low[0]>high[0] || low[1]>high[1])return 0;

    MpcRtiCycleConfiguration_t check_config=*cycle;
    check_config.capture_diagnostic_outputs=1;
    check_config.qp_capture=nullptr;
    check_config.qp_capture_second_pass=nullptr;
    // Prevent unrelated 30-stage terminal conditions from contaminating the
    // physically exact one-stage admissibility evaluation.
    check_config.terminal_viability=0;
    MpcModelControl_t controls[PREDICTION_HORIZON]{};
    MpcRtiNominal_t verified{};
    MpcRtiDiagnosticStageOutput_t outputs[MPC_RTI_DIAGNOSTIC_SAMPLE_COUNT]{};
    int output_count=0;
    int failure=-1;
    const auto check=[&](double steer,double throttle,
                         float *slack,double *front_slip) {
        if(source_checks)++*source_checks;
        controls[0].steering_target_rad=float(steer);
        controls[0].throttle_command=float(throttle);
        output_count=0;
        const auto status=mpc_rti_validate_candidate_plan_with_diagnostics(
            measured,progress,controls,trajectory,count,lap,1,prediction_dt,
            &check_config,&verified,outputs,&output_count,&failure);
        if(status!=MPC_RTI_ROLLOUT_OK || !verified.valid ||
           output_count<1 || outputs[0].stage!=1)return false;
        *slack=verified.validation_minimum_corridor_slack_m;
        *front_slip=std::max(
            std::abs(outputs[0].output.lateral_slip[0]),
            std::abs(outputs[0].output.lateral_slip[1]));
        return std::isfinite(*slack) && std::isfinite(*front_slip);
    };

    float nominal_slack=-INFINITY;
    double nominal_slip=INFINITY;
    const bool nominal_valid=check(
        std::clamp(double(nominal_first->steering_target_rad),low[0],high[0]),
        std::clamp(double(nominal_first->throttle_command),low[1],high[1]),
        &nominal_slack,&nominal_slip);
    if(nominal_valid && nominal_slack>=activation_clearance_m &&
       nominal_slip<=activation_slip)
        return 0; // normal QP is free where nonlinear safety is not at risk.
    if(deadline_us>0. && deadline_us-monotonic_microseconds()<8500.)
        return 0;

    double steering[ns],throttle[nt];
    for(int i=0;i<ns;++i)
        steering[i]=low[0]+(high[0]-low[0])*double(i)/(ns-1);
    for(int j=0;j<nt;++j)
        throttle[j]=low[1]+(high[1]-low[1])*double(j)/(nt-1);

    bool safe[ns][nt]{};
    float clearance[ns][nt]{};
    double slip[ns][nt]{};
    for(int i=0;i<ns;++i) {
        for(int j=0;j<nt;++j) {
            // If the physical oracle itself cannot complete within a
            // bounded budget, retain the unmodified QP; do not publish an
            // incomplete or fictitious safety certificate.
            if(deadline_us>0. &&
               deadline_us-monotonic_microseconds()<5200.)
                return 0;
            safe[i][j]=check(steering[i],throttle[j],
                &clearance[i][j],&slip[i][j]) &&
                clearance[i][j]>=reserve_m && slip[i][j]<=max_slip;
        }
    }
    // The actual source's nonmonotonic post-peak tire response means that
    // feasible control points must NOT be replaced by their convex hull.
    // Restrict to one contiguous, fully sampled neighbouring grid cell,
    // then independently source-check its centre (no extrapolation).
    double best=INFINITY;
    int bi=-1,bj=-1;
    const double span_s=std::max(.01,high[0]-low[0]);
    const double span_t=std::max(.01,high[1]-low[1]);
    for(int i=0;i<ns-1;++i)for(int j=0;j<nt-1;++j) {
        if(!(safe[i][j] && safe[i+1][j] &&
             safe[i][j+1] && safe[i+1][j+1]))continue;
        const double center_s=.5*(steering[i]+steering[i+1]);
        const double center_t=.5*(throttle[j]+throttle[j+1]);
        const double error_s=(center_s-nominal_first->steering_target_rad)/span_s;
        const double error_t=(center_t-nominal_first->throttle_command)/span_t;
        const double score=error_s*error_s+error_t*error_t;
        if(score<best){best=score;bi=i;bj=j;}
    }
    if(bi<0)return 0;
    float center_slack=-INFINITY;
    double center_slip=INFINITY;
    if(!check(.5*(steering[bi]+steering[bi+1]),
              .5*(throttle[bj]+throttle[bj+1]),
              &center_slack,&center_slip) ||
       center_slack<reserve_m || center_slip>max_slip)return 0;
    const double offset_s=model.command_increment_qp_enabled ? previous[0]:0.;
    const double offset_t=model.command_increment_qp_enabled ? previous[1]:0.;
    const float result_lb_s=float(steering[bi]-offset_s);
    const float result_ub_s=float(steering[bi+1]-offset_s);
    const float result_lb_t=float(throttle[bj]-offset_t);
    const float result_ub_t=float(throttle[bj+1]-offset_t);
    if(result_lb_s>result_ub_s || result_lb_t>result_ub_t)return 0;
    decision_lb[0]=result_lb_s;
    decision_ub[0]=result_ub_s;
    decision_lb[1]=result_lb_t;
    decision_ub[1]=result_ub_t;
    return 1;
}
