// Accuracy-only independent audit of the ORIGINAL full physical 1ms plant:
// active source contact/tire/steering/suspension substeps versus one-sided AD.
// Never used as a replacement physical model or online MPC cost.
#include "mpc_qp_failure_context.hpp"
#include "mpc_source_exact_nlp.hpp"
#include "unity_source_plant_c.h"
#include "vehicle_model.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using nlohmann::json;
namespace fs=std::filesystem;
int main(int argc,char **argv) {
    if(argc<2||argc>3) {
        std::cerr<<"usage: mpc_source_physical_one_sided_audit CASE_DIR [DELTA=0.0003]\n";
        return 2;
    }
    const float h=argc>=3?std::stof(argv[2]):.0003f;
    if(!(h>0.) || !std::isfinite(h))return 2;
    auto ctx=std::make_unique<sdu_apex::qp_fixture::FailureReplayContext>();
    if(!sdu_apex::qp_fixture::read_failure_context(
        (fs::path(argv[1])/"source_context.mctx").string(),ctx.get()))
        return 3;
    auto nominal=std::make_unique<MpcRtiNominal_t>();
    MpcRtiReference_t refs[PREDICTION_HORIZON+1]{};
    if(!mpc_rti_build_nominal_same_epoch(
        &ctx->initial_state,ctx->progress_m,nullptr,ctx->r1_nominal,
        PREDICTION_HORIZON,ctx->trajectory.data(),
        ctx->trajectory.size(),ctx->lap_length_m,ctx->prediction_dt,
        PREDICTION_HORIZON,&ctx->model,nominal.get(),refs))return 4;
    json rows=json::array();
    int evaluated=0,discrete_or_contact=0;
    int one_sided_switch_resolutions=0;
    double maximum_abs_body_error=0.,maximum_scaled_body_error=0.;
    double maximum_abs_state_error=0.,maximum_scaled_state_error=0.;
    for(int stage=0;stage<PREDICTION_HORIZON;++stage) {
        UnitySourcePlantInputC input{};
        if(!vehicle_model_pack_source_state(
            &nominal->states[stage].plant,input.state))return 5;
        input.steering_target_rad=std::clamp(
            double(nominal->controls[stage].steering_target_rad),
            -double(UNITY_SOURCE_MAX_STEERING_RAD),
            double(UNITY_SOURCE_MAX_STEERING_RAD));
        input.throttle_normalized=
            nominal->controls[stage].throttle_command;
        input.suspension_history_initialized=
            nominal->states[stage].plant.source_suspension_history_initialized;
        UnitySourcePlantOutputC center{};
        if(!unity_source_plant_step_c(
            &input,ctx->prediction_dt,&center))return 6;
        for(int axis=0;axis<2;++axis)
            for(int sign : {-1,1}) {
                UnitySourcePlantInputC trial=input;
                MpcModelControl_t physical=nominal->controls[stage];
                if(axis==0)physical.steering_target_rad+=sign*h;
                else physical.throttle_command+=sign*h;
                trial.steering_target_rad=std::clamp(
                    double(physical.steering_target_rad),
                    -double(UNITY_SOURCE_MAX_STEERING_RAD),
                    double(UNITY_SOURCE_MAX_STEERING_RAD));
                trial.throttle_normalized=physical.throttle_command;
                if(trial.steering_target_rad<
                        -ctx->model.max_steering_rad ||
                   trial.steering_target_rad>
                        ctx->model.max_steering_rad ||
                   trial.throttle_normalized<MPC_THROTTLE_MIN_DRIVE ||
                   trial.throttle_normalized>MPC_THROTTLE_MAX)continue;
                const double delta=axis==0?
                    trial.steering_target_rad-input.steering_target_rad:
                    trial.throttle_normalized-input.throttle_normalized;
                if(delta==0.)continue;
                double state_direction[UNITY_SOURCE_PLANT_STATE_SIZE]{};
                const double control_direction[2]={
                    axis==0?delta:0.,axis==1?delta:0.};
                UnitySourcePlantDirectionalCurvatureC source_ad{};
                UnitySourceDirectionalBranchReportC branch{};
                if(!unity_source_plant_step_one_sided_directional_c(
                    &input,ctx->prediction_dt,state_direction,
                    control_direction,&source_ad,&branch))
                    return 7;
                UnitySourcePlantOutputC actual{};
                if(!unity_source_plant_step_c(
                    &trial,ctx->prediction_dt,&actual))return 8;
                double max_abs_body=0.,max_scaled_body=0.;
                double max_abs_state=0.,max_scaled_state=0.;
                const double old_body[]{
                    center.integrated_body_forward_displacement_m,
                    center.integrated_body_lateral_displacement_m,
                    center.integrated_yaw_delta_rad};
                const double new_body[]{
                    actual.integrated_body_forward_displacement_m,
                    actual.integrated_body_lateral_displacement_m,
                    actual.integrated_yaw_delta_rad};
                for(int k=0;k<3;++k) {
                    const double physical_change=new_body[k]-old_body[k];
                    const double error=std::abs(physical_change-
                                               source_ad.first_body[k]);
                    max_abs_body=std::max(max_abs_body,error);
                    max_scaled_body=std::max(max_scaled_body,
                        error/std::max({1.e-8,
                            std::abs(physical_change),
                            std::abs(source_ad.first_body[k])}));
                }
                for(int k=0;k<UNITY_SOURCE_PLANT_STATE_SIZE;++k) {
                    const double physical_change=actual.state[k]-center.state[k];
                    const double error=std::abs(physical_change-
                                               source_ad.first_state[k]);
                    max_abs_state=std::max(max_abs_state,error);
                    max_scaled_state=std::max(max_scaled_state,
                        error/std::max({1.e-7,
                            std::abs(physical_change),
                            std::abs(source_ad.first_state[k])}));
                }
                one_sided_switch_resolutions+=
                    branch.derivative_selected_switch_ties;
                if(!branch.valid_one_sided_directional)
                    ++discrete_or_contact;
                else {
                    maximum_abs_body_error=std::max(
                        maximum_abs_body_error,max_abs_body);
                    maximum_scaled_body_error=std::max(
                        maximum_scaled_body_error,max_scaled_body);
                    maximum_abs_state_error=std::max(
                        maximum_abs_state_error,max_abs_state);
                    maximum_scaled_state_error=std::max(
                        maximum_scaled_state_error,max_scaled_state);
                }
                rows.push_back({
                    {"stage",stage},{"axis",axis},{"sign",sign},
                    {"physical_float_control_delta",delta},
                    {"one_sided_source_derivative_valid",
                     branch.valid_one_sided_directional!=0},
                    {"source_switch_ties",
                     branch.exact_physical_switch_ties},
                    {"source_switches_direction_resolved",
                     branch.derivative_selected_switch_ties},
                    {"source_contact_switch_ties",
                     branch.contact_switch_ties},
                    {"source_zero_throttle_brake_switches",
                     branch.zero_throttle_brake_switches},
                    {"source_substep_schedule_switches",
                     branch.discrete_substep_schedule_switches},
                    {"maximum_original_source_body_change_abs_error",
                     max_abs_body},
                    {"maximum_original_source_body_change_scaled_error",
                     max_scaled_body},
                    {"maximum_original_source_26_state_change_abs_error",
                     max_abs_state},
                    {"maximum_original_source_26_state_change_scaled_error",
                     max_scaled_state}});
                ++evaluated;
            }
    }
    std::cout<<json({
        {"schema","original-25ms-physical-one-sided-source-AD/v1"},
        {"case",argv[1]},
        {"physical_float_command_step",h},
        {"evaluated",evaluated},
        {"nondifferentiable_or_discrete_step_count",discrete_or_contact},
        {"direction_resolved_exact_physical_ties",
         one_sided_switch_resolutions},
        {"maximum_valid_source_body_abs_error",maximum_abs_body_error},
        {"maximum_valid_source_body_scaled_error",maximum_scaled_body_error},
        {"maximum_valid_source_state_abs_error",maximum_abs_state_error},
        {"maximum_valid_source_state_scaled_error",maximum_scaled_state_error},
        {"source_and_AD_use_same_original_inner_plant",true},
        {"is_Unity_PhysX_fidelity_certificate",false},
        {"trials",rows}
    }).dump(2)<<'\n';
    return 0;
}
