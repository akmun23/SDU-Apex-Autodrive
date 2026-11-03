/*
 * Counterfactual one-control-epoch source-step experiment.
 *
 * Input is a captured command epoch and a complete 30-stage command plan.
 * Apply ONLY its first action to the EXACT source model, including queue,
 * wheel speeds, suspension, and loaded-path Frenet projection. The remaining
 * plan becomes the shifted warm start at a new (counterfactual) command epoch.
 * Rebuild the objective schedule and hard LTV QP at that new state.
 *
 * This is NOT replaying the logged states, nor controlling Unity. Results
 * are causal only under the checked-in source plant, whose differences
 * from WheelCollider/Unity still require separate validation.
 *
 * Never certify a long trajectory by checking only the QP: the full
 * remaining plan is independently evaluated in source rollout when the
 * next epoch's solver runs.
 */
#include "mpc_rti.h"
#include "mpc_rti_internal.hpp"
#include "mpc_qp_fixture_format.hpp"
#include "mpc_qp_failure_context.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>

static void print_exact_state(const MpcRtiState_t &state,double progress) {
    const auto &s=state.plant;
    std::cout<<"{\"e_y\":"<<s.e_y
        <<",\"e_psi\":"<<s.e_psi
        <<",\"u\":"<<s.u
        <<",\"v\":"<<s.v
        <<",\"r\":"<<s.r
        <<",\"rack\":"<<s.actual_steering_angle
        <<",\"progress\":"<<progress
        <<",\"queue\":["
        <<s.delayed_steering_command_1<<","
        <<s.delayed_steering_command_2<<","
        <<s.delayed_throttle_command_1<<","
        <<s.delayed_throttle_command_2
        <<"],\"source_initialized\":"<<s.source_plant_initialized
        <<",\"internal\":[";
    for(int j=0;j<MPC_PLANT_INTERNAL_STATE_COUNT;++j) {
        if(j)std::cout<<",";
        std::cout<<s.source_plant_internal_state[j];
    }
    std::cout<<"]}";
}

int main(int argc,char**argv) {
    if(argc!=4) {
        std::cerr<<"usage: mpc_offline_advance_source_epoch INPUT_CASE accepted_controls.txt OUTPUT_CASE_DIR\n";
        return 2;
    }
    namespace fs=std::filesystem;
    auto input=std::make_unique<sdu_apex::qp_fixture::FailureReplayContext>();
    const fs::path original(argv[1]),target(argv[3]);
    if(!sdu_apex::qp_fixture::read_failure_context(
           (original/"source_context.mctx").string(),input.get())) {
        std::cerr<<"frozen input context not readable\n";return 3;
    }
    auto template_qp=std::make_unique<MpcRtiQpCapture_t>();
    if(!sdu_apex::qp_fixture::read_capture(
            (original/"r1.fqp").string(),template_qp.get())) {
        std::cerr<<"frozen original QP metadata not readable\n";return 4;
    }
    MpcModelControl_t proposal[PREDICTION_HORIZON]{};
    std::ifstream commands(argv[2]);
    if(!commands.good())return 5;
    for(auto &u:proposal)
        if(!(commands>>u.steering_target_rad>>u.throttle_command) ||
            !std::isfinite(u.steering_target_rad) ||
            !std::isfinite(u.throttle_command))return 5;
    MpcRtiCycleConfiguration_t validation{};
    validation.model=input->model;
    validation.model.world_sweep_check=nullptr;
    validation.model.world_sweep_context=nullptr;
    validation.model.world_map_rejection_only=0;
    validation.terminal_viability=0;
    validation.objective_schedule=nullptr;
    // Validate ONLY the first command: the whole candidate may remain
    // infeasible later; otherwise counterfactual recovery cannot be tested.
    auto one_step=std::make_unique<MpcRtiNominal_t>();
    int failure_stage=-1;
    const auto status=mpc_rti_validate_candidate_plan(
        &input->initial_state,input->progress_m,proposal,
        input->trajectory.data(),input->trajectory.size(),
        input->lap_length_m,1,input->prediction_dt,
        &validation,one_step.get(),&failure_stage);
    if(status!=MPC_RTI_ROLLOUT_OK) {
        std::cout<<"{\"advanced\":false,\"first_control_source_status\":"
                 <<static_cast<int>(status)
                 <<",\"failure_stage\":"<<failure_stage<<"}\n";
        return 6;
    }
    const MpcRtiState_t next=one_step->states[1];
    const double next_progress=one_step->progress[1];
    if(!std::isfinite(next_progress))return 7;
    MpcModelControl_t shifted[PREDICTION_HORIZON]{};
    for(int k=0;k<PREDICTION_HORIZON-1;++k)
        shifted[k]=proposal[k+1];
    shifted[PREDICTION_HORIZON-1]=proposal[PREDICTION_HORIZON-1];
    auto next_nominal=std::make_unique<MpcRtiNominal_t>();
    MpcRtiReference_t references[PREDICTION_HORIZON+1]{};
    if(!mpc_rti_build_nominal_same_epoch(
        &next,next_progress,nullptr,shifted,PREDICTION_HORIZON,
        input->trajectory.data(),input->trajectory.size(),
        input->lap_length_m,input->prediction_dt,PREDICTION_HORIZON,
        &validation.model,next_nominal.get(),references)) {
        std::cerr<<"next source nominal could not be built\n";return 8;
    }
    auto nextctx=std::make_unique<sdu_apex::qp_fixture::FailureReplayContext>(*input);
    nextctx->command_epoch_ns+=(std::int64_t)llround(
         double(input->prediction_dt)*1.e9);
    nextctx->initial_state=next;
    nextctx->progress_m=next_progress;
    // A new epoch has new references and branch-frozen weights. Never
    // evaluate its objective against the prior epoch's stale schedule.
    if(!mpc_rti_build_objective_schedule(
           next_nominal.get(),references,&validation.model,
           &nextctx->objective)) {
        std::cerr<<"counterfactual objective regeneration failed\n";return 9;
    }
    // This new epoch is hypothetical, NOT a historical captured solver run.
    nextctx->r1_captured=nextctx->r2_captured=0;
    nextctx->r1_controls_valid=nextctx->r2_controls_valid=0;
    nextctx->r1_source_rollout_controls_valid=
        nextctx->r2_source_rollout_controls_valid=0;
    std::copy_n(shifted,PREDICTION_HORIZON,nextctx->r1_nominal);
    auto fresh=std::make_unique<MpcRtiQpCapture_t>();
    if(!mpc_rti_build_ltv_qp_with_progress_corridor(
        next_nominal->states,next_nominal->controls,references,
        PREDICTION_HORIZON,input->prediction_dt,&validation.model,
        &nextctx->objective,&fresh->problem,next_nominal->progress,
        nextctx->trajectory.data(),nextctx->trajectory.size(),
        nextctx->lap_length_m,nullptr)) {
        std::cerr<<"counterfactual QP regeneration failed\n";return 10;
    }
    fresh->solver_config=template_qp->solver_config;
    for(int k=0;k<=RICCATI_HORIZON;++k)
        fresh->solver_config.state_polygons[k]=
            fresh->problem.state_polygons[k];
    fresh->solver_config.max_solve_time_us=0.;
    fresh->solver_config.cold_start_each_solve=1;
    fresh->valid=fresh->capture_count=1;
    fs::create_directories(target);
    if(!sdu_apex::qp_fixture::write_failure_context(
           (target/"source_context.mctx").string(),*nextctx) ||
       !sdu_apex::qp_fixture::write_capture(
           (target/"r1.fqp").string(),*fresh)) {
        std::cerr<<"counterfactual snapshot serialization failed\n";
        return 11;
    }
    std::ofstream seed(target/"counterfactual_shifted_seed.txt");
    seed<<std::setprecision(13);
    for(auto &u:shifted)
        seed<<u.steering_target_rad<<" "<<u.throttle_command<<"\n";
    std::cout<<std::setprecision(13)
             <<"{\"advanced\":true,\"source_step_s\":"<<input->prediction_dt
             <<",\"epoch_before_ns\":"<<input->command_epoch_ns
             <<",\"epoch_after_ns\":"<<nextctx->command_epoch_ns
             <<",\"progress_before_m\":"<<input->progress_m
             <<",\"progress_after_m\":"<<next_progress
             <<",\"e_y_before_m\":"<<input->initial_state.plant.e_y
             <<",\"e_y_after_m\":"<<next.plant.e_y
             <<",\"e_psi_after_rad\":"<<next.plant.e_psi
             <<",\"speed_after_mps\":"<<next.plant.u
             <<",\"applied_steering_rad\":"<<proposal[0].steering_target_rad
             <<",\"applied_throttle\":"<<proposal[0].throttle_command
             <<",\"first_step_strict_source_valid\":true"
             <<",\"next_full_horizon_source_feasibility_claim\":false"
             <<",\"exact_state_after\":";
    print_exact_state(next,next_progress);
    std::cout<<"}\n";
    return 0;
}
