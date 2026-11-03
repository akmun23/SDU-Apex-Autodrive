/*
 * Authoritative exact-source single-shooting QP oracle.
 *
 * Evaluates one complete source trajectory at the same saved command epoch:
 * original, progress-interpolated left/right 30-stage physical footprint
 * constraints, same frozen racing objective and full actuator history.
 * No QP relinearization, no fallback, no wall relaxation and no publication.
 *
 * This is intentionally standalone so numeric derivatives in a source
 * control-space SQP need not rebuild Jacobians 120+ times per iteration.
 */
#include "mpc_rti.h"
#include "mpc_rti_internal.hpp"
#include "mpc_qp_failure_context.hpp"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>

int main(int argc,char**argv) {
    if(argc!=3) {
        std::cerr<<"usage: mpc_offline_source_merit CASE controls.txt|--serve\n";
        return 2;
    }
    sdu_apex::qp_fixture::FailureReplayContext ctx{};
    if(!sdu_apex::qp_fixture::read_failure_context(
        std::string(argv[1])+"/source_context.mctx",&ctx))return 3;
    // Daemon mode: parse the frozen source context ONCE, then evaluate many
    // exact source trajectories in the same process. Each line contains
    // exactly 30 pairs of commanded steering/throttle values and receives
    // exactly one JSON line. No cache, approximate physics, or warm-state
    // carryover is permitted across requests.
    const bool serve=std::string(argv[2])=="--serve";
    std::ifstream input_file;
    if(!serve) {
        input_file.open(argv[2]);
        if(!input_file.good())return 4;
    }
    std::istream &input=serve ? static_cast<std::istream&>(std::cin)
                              : static_cast<std::istream&>(input_file);
    do {
    // The optimization base needs the exact source adjoint, whereas the
    // nonlinear line search only needs objective and original constraints.
    // Explicit 'G'/'Q'/'D' requests all use the SAME source-plant
    // integrator. 'D' is a derivative-only query: physical wall/rate values
    // are exact but strict source validation is intentionally not performed.
    // Such outputs must never be interpreted as publishable candidates.
    bool need_gradient=true;
    bool need_strict=true;
    if(serve) {
        char kind='G';
        if(!(input>>kind)) {
            if(input.eof())return 0;
            return 4;
        }
        if(kind!='G' && kind!='Q' && kind!='D')return 4;
        need_gradient=kind=='G';
        need_strict=kind!='D';
    }
    MpcModelControl_t controls[PREDICTION_HORIZON]{};
    for(auto &c:controls) {
        if(!(input>>c.steering_target_rad>>c.throttle_command)) {
            if(serve && input.eof())return 0;
            return 4;
        }
    }
    auto nominal=std::make_unique<MpcRtiNominal_t>();
    MpcRtiReference_t refs[PREDICTION_HORIZON+1]{};
    MpcRtiConfiguration_t model=ctx.model;
    model.world_sweep_check=nullptr;
    model.world_sweep_context=nullptr;
    model.world_map_rejection_only=0;
    if(!mpc_rti_build_nominal_same_epoch(
        &ctx.initial_state,ctx.progress_m,nullptr,controls,
        PREDICTION_HORIZON,ctx.trajectory.data(),ctx.trajectory.size(),
        ctx.lap_length_m,ctx.prediction_dt,PREDICTION_HORIZON,&model,
        nominal.get(),refs)) {
        std::cerr<<"source nominal failed\n";return 5;
    }
    MpcRtiCycleConfiguration_t score_config{};
    score_config.model=model;
    score_config.objective_schedule=&ctx.objective;
    score_config.terminal_viability=0;
    double objective=NAN;
    float minimum_slack=NAN;
    if(!mpc_rti_score_candidate_plan_for_diagnostics(nominal.get(),
        ctx.trajectory.data(),ctx.trajectory.size(),ctx.lap_length_m,
        &score_config,&objective,&minimum_slack))return 6;
    double left[PREDICTION_HORIZON+1]{},right[PREDICTION_HORIZON+1]{};
    for(int k=0;k<=PREDICTION_HORIZON;++k) {
        MpcRtiReference_t reference{};
        if(!mpc_rti_reference_at_progress(ctx.trajectory.data(),
              ctx.trajectory.size(),ctx.lap_length_m,nominal->progress[k],
              &model,&reference))return 7;
        float ls=NAN,rs=NAN;
        if(!mpc_rti_compute_corridor_slack(
              &nominal->states[k].plant,reference.left_bound,
              reference.right_bound,&model,k,&ls,&rs))return 8;
        left[k]=ls;right[k]=rs;
    }
    double source_gradient[PREDICTION_HORIZON][MPC_RTI_NU]{};
    const bool gradient_valid=need_gradient &&
        mpc_rti_source_adjoint_gradient_from_exact_nominal(
            nominal.get(),ctx.trajectory.data(),ctx.trajectory.size(),
            ctx.lap_length_m,ctx.prediction_dt,&model,
            &ctx.objective,source_gradient);
    int failure_stage=-1;
    MpcRtiRolloutStatus_t strict_status=MPC_RTI_ROLLOUT_INVALID_INPUT;
    if(need_strict) {
        auto strict=std::make_unique<MpcRtiNominal_t>();
        strict_status=mpc_rti_validate_candidate_plan(
            &ctx.initial_state,ctx.progress_m,controls,ctx.trajectory.data(),
            ctx.trajectory.size(),ctx.lap_length_m,PREDICTION_HORIZON,
            ctx.prediction_dt,&score_config,strict.get(),&failure_stage);
    }
    std::cout<<std::setprecision(14)
        <<"{\"max_steering_rad\":"<<model.max_steering_rad
        <<",\"max_steering_command_step_rad\":"<<model.max_steering_command_step_rad
        <<",\"max_throttle_command_step\":"<<model.max_throttle_command_step
        <<",\"prediction_dt\":"<<ctx.prediction_dt
        <<",\"objective\":"<<objective
        <<",\"physical_minimum_clearance\":"<<minimum_slack
        <<",\"strict_checked\":"<<(need_strict?"true":"false")
        <<",\"original_valid\":"
        <<(strict_status==MPC_RTI_ROLLOUT_OK?"true":"false")
        <<",\"failure_stage\":"<<failure_stage
        <<",\"failure_reason\":"<<static_cast<int>(strict_status);
    std::cout<<",\"left_clearance_m\":[";
    for(int k=0;k<=PREDICTION_HORIZON;++k) {
        if(k)std::cout<<',';
        std::cout<<left[k];
    }
    std::cout<<"],\"right_clearance_m\":[";
    for(int k=0;k<=PREDICTION_HORIZON;++k) {
        if(k)std::cout<<',';
        std::cout<<right[k];
    }
    std::cout<<"],\"source_gradient_valid\":"
             <<(gradient_valid?"true":"false")
             <<",\"source_gradient_flat\":[";
    if(gradient_valid)
        for(int k=0;k<PREDICTION_HORIZON;++k) {
            if(k)std::cout<<',';
            std::cout<<source_gradient[k][0]<<','<<source_gradient[k][1];
        }
    std::cout<<"],\"rack_steering_rad\":[";
    for(int k=0;k<=PREDICTION_HORIZON;++k) {
        if(k)std::cout<<',';
        std::cout<<nominal->states[k].plant.actual_steering_angle;
    }
    std::cout<<"],\"previous_throttle_command\":[";
    for(int k=0;k<=PREDICTION_HORIZON;++k) {
        if(k)std::cout<<',';
        std::cout<<nominal->states[k].plant.delayed_throttle_command_2;
    }
    std::cout<<"],\"progress_m\":[";
    for(int k=0;k<=PREDICTION_HORIZON;++k) {
        if(k)std::cout<<',';
        std::cout<<nominal->progress[k];
    }
    std::cout<<"]}\n"<<std::flush;
    if(!serve)break;
    } while(true);
    return 0;
}
