/*
 * Frozen-source-controller single epoch executor.
 * Invokes the ACTUAL mpc_rti_solve_cycle entry with native source SQP,
 * then exports a full command plan ONLY when independently source-valid.
 * Source predictions and complete actuator queue are printed so that a
 * separate exact-plant one-step advancement can be checked byte-for-byte.
 * No ROS publishes, no Unity simulator activity, no offline fallback.
 */
#include "mpc_rti.h"
#include "mpc_qp_failure_context.hpp"
#include "mpc_qp_fixture_format.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <type_traits>

static void state_json(const MpcRtiState_t &state,double progress) {
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

static bool nearly_equal_state(const MpcRtiState_t &a,
                               const MpcRtiState_t &b,double *maximum) {
    const auto &x=a.plant, &y=b.plant;
    double max_error=0.;
    const auto metric=[&](double a,double b) {
        max_error=std::max(max_error,std::abs(a-b));
    };
    metric(x.e_y,y.e_y);metric(x.e_psi,y.e_psi);
    metric(x.u,y.u);metric(x.v,y.v);metric(x.r,y.r);
    metric(x.actual_steering_angle,y.actual_steering_angle);
    metric(x.delayed_steering_command_1,y.delayed_steering_command_1);
    metric(x.delayed_steering_command_2,y.delayed_steering_command_2);
    metric(x.delayed_throttle_command_1,y.delayed_throttle_command_1);
    metric(x.delayed_throttle_command_2,y.delayed_throttle_command_2);
    for(int j=0;j<MPC_PLANT_INTERNAL_STATE_COUNT;++j)
        metric(x.source_plant_internal_state[j],
               y.source_plant_internal_state[j]);
    *maximum=max_error;
    return std::isfinite(max_error) && max_error<=1e-8 &&
        x.source_plant_initialized==y.source_plant_initialized &&
        x.source_suspension_history_initialized==
            y.source_suspension_history_initialized &&
        x.source_imu_state_initialized==y.source_imu_state_initialized;
}
static_assert(std::is_trivially_copyable<MpcRtiMemory_t>::value,
    "offline memory serialization requires a trivially copyable optimizer state");
static bool read_persistent_memory(const std::filesystem::path &file,
                                   MpcRtiMemory_t *memory) {
    std::ifstream input(file,std::ios::binary);
    const std::uint32_t magic_expected=0x50535131u;
    std::uint32_t magic=0,size=0;
    input.read(reinterpret_cast<char*>(&magic),sizeof(magic));
    input.read(reinterpret_cast<char*>(&size),sizeof(size));
    if(!input || magic!=magic_expected ||size!=sizeof(*memory))return false;
    input.read(reinterpret_cast<char*>(memory),sizeof(*memory));
    return input.good() && input.peek()==std::char_traits<char>::eof();
}
static bool write_persistent_memory(const std::filesystem::path &file,
                                    const MpcRtiMemory_t *memory) {
    const std::uint32_t magic=0x50535131u;
    const std::uint32_t size=sizeof(*memory);
    std::ofstream output(file,std::ios::binary);
    output.write(reinterpret_cast<const char*>(&magic),sizeof(magic));
    output.write(reinterpret_cast<const char*>(&size),sizeof(size));
    output.write(reinterpret_cast<const char*>(memory),sizeof(*memory));
    return output.good();
}
int main(int argc,char**argv) {
    if(argc<3||argc>12) {
        std::cerr<<"usage: native_controller_epoch CASE output_controls.txt [iterations] [previous_memory|-] [next_memory]\n";
        return 2;
    }
    namespace fs=std::filesystem;
    auto ctx=std::make_unique<sdu_apex::qp_fixture::FailureReplayContext>();
    const auto case_dir=fs::path(argv[1]);
    const fs::path output(argv[2]);
    if(!sdu_apex::qp_fixture::read_failure_context(
        (case_dir/"source_context.mctx").string(),ctx.get()))return 3;
    auto capture=std::make_unique<MpcRtiQpCapture_t>();
    if(!sdu_apex::qp_fixture::read_capture(
        (case_dir/"r1.fqp").string(),capture.get()))return 4;
    const int iters=argc>=4?std::stoi(argv[3]):2;
    bool terminal3=false,opt20=false,suffix_fd=false,adaptive=false;
    bool forward_source=false;
    double hard_deadline_budget_us=0.;
    for(int i=6;i<argc;++i) {
        const std::string flag(argv[i]);
        if(flag=="--terminal3")terminal3=true;
        else if(flag=="--opt20")opt20=true;
        else if(flag=="--suffix-fd")suffix_fd=true;
        else if(flag=="--adaptive")adaptive=true;
        else if(flag=="--full-source-forward")forward_source=true;
        else if(flag.rfind("--deadline-us=",0)==0) {
            try {hard_deadline_budget_us=std::stod(flag.substr(14));}
            catch(...) {return 2;}
            if(hard_deadline_budget_us<1000. ||
               hard_deadline_budget_us>1.e6)return 2;
        }
        else return 2;
    }
    if(iters<1||iters>8)return 5;
    MpcRtiCycleConfiguration_t cfg{};
    cfg.model=ctx->model;
    cfg.model.world_sweep_check=nullptr;
    cfg.model.world_sweep_context=nullptr;
    cfg.model.world_map_rejection_only=0;
    cfg.solver=capture->solver_config;
    cfg.solver.max_solve_time_us=0.;
    cfg.solver.reuse_same_epoch=1;
    cfg.max_sqp_iterations=2;
    cfg.native_source_reduced_sqp=1;
    cfg.native_source_reduced_sqp_iterations=iters;
    cfg.native_source_sqp_terminal_lookahead_stages=
        terminal3?3:0;
    cfg.native_source_sqp_control_decision_stages=opt20?20:30;
    cfg.native_source_sqp_reuse_fd_prefix=suffix_fd?1:0;
    cfg.native_source_sqp_adaptive_stop=adaptive?1:0;
    cfg.native_source_sqp_full_source_forward=forward_source?1:0;
    cfg.command_epoch_ns=ctx->command_epoch_ns;
    cfg.line_search_command_epoch_valid=1;
    cfg.enable_nonlinear_line_search=1;
    cfg.line_search_max_backtracks=4;
    cfg.feedback_rollout=0;
    cfg.progress_coupled_hard_corridor=0;
    cfg.feedback_rollout_gain_scale=1.f;
    auto memory=std::make_unique<MpcRtiMemory_t>();
    auto result=std::make_unique<MpcRtiCycleResult_t>();
    const bool prior_memory=argc>=5&&std::string(argv[4])!="-";
    double memory_prior_state_difference=0.;
    double memory_prior_progress_difference=0.;
    if(prior_memory) {
        if(!read_persistent_memory(argv[4],memory.get())) {
            std::cerr<<"prior source MPC memory is invalid or ABI-mismatched\n";
            return 7;
        }
        if(!memory->nominal.valid || memory->nominal.horizon!=
                PREDICTION_HORIZON ||
            !memory->warm_start_controls_valid ||
            memory->warm_start_control_count!=PREDICTION_HORIZON ||
            mpc_rti_command_epoch_shift(memory->command_epoch_ns,
                ctx->command_epoch_ns,ctx->prediction_dt)!=1 ||
            !nearly_equal_state(memory->nominal.states[1],
                ctx->initial_state,&memory_prior_state_difference) ||
            ((memory_prior_progress_difference=std::abs(
                  memory->nominal.progress[1]-ctx->progress_m))>1e-8)) {
            std::cerr<<"prior certified memory disagrees with actual next source state/epoch\n";
            return 8;
        }
    } else {
        memory->command_epoch_ns=ctx->command_epoch_ns;
        memory->warm_start_controls_valid=1;
        memory->warm_start_control_count=PREDICTION_HORIZON;
        std::copy_n(ctx->r1_nominal,PREDICTION_HORIZON,
                    memory->warm_start_controls);
    }
    if(hard_deadline_budget_us>0.)
        cfg.absolute_deadline_monotonic_us=
            mpc_rti_monotonic_time_us()+hard_deadline_budget_us;
    const auto status=mpc_rti_solve_cycle(
        &ctx->initial_state,ctx->progress_m,ctx->trajectory.data(),
        ctx->trajectory.size(),ctx->lap_length_m,ctx->prediction_dt,
        PREDICTION_HORIZON,&cfg,memory.get(),result.get());
    // Offline-only diagnostic source-seed capture. Important for identifying
    // whether a freshly synthesized last tail command breaks an otherwise
    // valid shifted warm start; never capture on the live ROS path.
    if(const char *dump=std::getenv("MPC_SOURCE_SQP_DEBUG_SEED")) {
        std::ofstream stream(dump);
        stream<<std::setprecision(12);
        for(int k=0;k<result->nominal_control_count;++k)
            stream<<result->nominal_controls[k].steering_target_rad<<" "
                  <<result->nominal_controls[k].throttle_command<<"\n";
    }
    const bool candidate=status==MPC_RTI_CYCLE_ACCEPTED_FEASIBLE &&
        result->candidate_control_count==PREDICTION_HORIZON &&
        memory->nominal.valid;
    auto independently_validated=std::make_unique<MpcRtiNominal_t>();
    int failure=-1;
    MpcRtiCycleConfiguration_t check_cfg=cfg;
    check_cfg.objective_schedule=&result->comparison_objective;
    const auto verified=candidate &&
        mpc_rti_validate_candidate_plan(
            &ctx->initial_state,ctx->progress_m,result->candidate_controls,
            ctx->trajectory.data(),ctx->trajectory.size(),ctx->lap_length_m,
            PREDICTION_HORIZON,ctx->prediction_dt,&check_cfg,
            independently_validated.get(),&failure)==MPC_RTI_ROLLOUT_OK;
    bool memory_matches=false,first_control_matches=false,epoch_matches=false;
    if(verified) {
        first_control_matches=
            result->first_control.steering_target_rad==
                result->candidate_controls[0].steering_target_rad &&
            result->first_control.throttle_command==
                result->candidate_controls[0].throttle_command;
        memory_matches=memory->warm_start_controls_valid &&
            memory->warm_start_control_count==PREDICTION_HORIZON &&
            memory->nominal.valid &&
            memory->nominal.horizon==PREDICTION_HORIZON;
        for(int k=0;k<PREDICTION_HORIZON && memory_matches;++k)
            memory_matches=memory->warm_start_controls[k].steering_target_rad==
                    result->candidate_controls[k].steering_target_rad &&
                memory->warm_start_controls[k].throttle_command==
                    result->candidate_controls[k].throttle_command;
        epoch_matches=memory->command_epoch_ns==ctx->command_epoch_ns;
    }
    const bool exportable=verified&&first_control_matches&&memory_matches&&epoch_matches;
    if(exportable) {
        if(argc>=6) {
            fs::create_directories(fs::path(argv[5]).parent_path());
            if(!write_persistent_memory(argv[5],memory.get()))
                return 9;
        }
        fs::create_directories(output.parent_path());
        std::ofstream file(output);
        file<<std::setprecision(12);
        for(int k=0;k<PREDICTION_HORIZON;++k)
            file<<result->candidate_controls[k].steering_target_rad<<" "
                <<result->candidate_controls[k].throttle_command<<"\n";
        if(!file.good())return 6;
    } else {
        // No partially infeasible selected controls may flow to next epoch.
        std::error_code ignore;
        fs::remove(output,ignore);
    }
    std::cout<<std::setprecision(17)
       <<"{\"controller_entry\":true,\"accepted\":"<<(exportable?"true":"false")
       <<",\"deadline_requested_us\":"<<hard_deadline_budget_us
       <<",\"status\":"<<int(status)
       <<",\"input_failure_reason\":"<<result->input_failure_reason
       <<",\"nominal_failure_stage\":"<<result->nominal_build_failure_stage
       <<",\"nominal_build_us\":"<<result->nominal_build_us
       <<",\"control_decision_stages\":"
       <<cfg.native_source_sqp_control_decision_stages
       <<",\"source_suffix_reuse\":"
       <<(suffix_fd?"true":"false")
       <<",\"full_source_forward_sensitivities\":"
       <<(forward_source?"true":"false")
       <<",\"terminal_lookahead_steps\":"
       <<cfg.native_source_sqp_terminal_lookahead_stages
       <<",\"prior_memory_loaded\":"<<(prior_memory?"true":"false")
       <<",\"prior_memory_next_state_error\":"
       <<memory_prior_state_difference
       <<",\"prior_memory_next_progress_error\":"
       <<memory_prior_progress_difference
       <<",\"epoch_shift_stages\":"
       <<result->command_epoch_shift_stages
       <<",\"nominal_seed_source\":"
       <<result->nominal_control_seed_source
       <<",\"epoch_ns\":"<<ctx->command_epoch_ns
       <<",\"candidate_count\":"<<result->candidate_control_count
       <<",\"strict_revalidation\":"<<(verified?"true":"false")
       <<",\"memory_match\":"<<(memory_matches?"true":"false")
       <<",\"epoch_match\":"<<(epoch_matches?"true":"false")
       <<",\"published_first_control_matches\":"
       <<(first_control_matches?"true":"false")
       <<",\"initial_wall_violation_m2\":"
       <<result->native_source_sqp_initial_wall_violation_m2
       <<",\"final_wall_violation_m2\":"
       <<result->native_source_sqp_final_wall_violation_m2
       <<",\"sqp_iterations\":"<<result->sqp_iterations_completed
       <<",\"qp_solves\":"<<result->native_source_sqp_qp_solves
       <<",\"source_evaluations\":"<<result->native_source_sqp_source_evaluations
       <<",\"initial_source_strict_valid\":"
       <<(result->native_source_sqp_initial_strict_source_valid?"true":"false")
       <<",\"final_source_strict_valid\":"
       <<(result->native_source_sqp_final_strict_source_valid?"true":"false")
       <<",\"initial_rate_violation_m2\":"
       <<result->native_source_sqp_initial_rate_violation_m2
       <<",\"final_rate_violation_m2\":"
       <<result->native_source_sqp_final_rate_violation_m2
       <<",\"last_qp_status\":"
       <<result->native_source_sqp_last_qp_status
       <<",\"source_forward_intervals\":"
       <<result->native_source_sqp_full_forward_intervals
       <<",\"derivative_time_ms\":"<<result->qp_build_us/1000.
       <<",\"qp_time_ms\":"
       <<result->native_source_sqp_qp_us/1000.
       <<",\"exact_source_eval_time_ms\":"
       <<result->native_source_sqp_source_evaluation_us/1000.
       <<",\"solver_time_ms\":"<<result->total_cycle_us/1000.
       <<",\"initial_state\":";
    state_json(ctx->initial_state,ctx->progress_m);
    std::cout<<",\"predicted_next_state\":";
    if(exportable)state_json(independently_validated->states[1],
                                independently_validated->progress[1]);
    else std::cout<<"null";
    std::cout<<",\"command\":";
    if(exportable)
        std::cout<<"["<<result->first_control.steering_target_rad<<","
                 <<result->first_control.throttle_command<<"]";
    else std::cout<<"null";
    std::cout<<"}\n";
    return 0;
}
