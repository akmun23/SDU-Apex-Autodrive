/*
 * Frozen source-SQP numerical comparison driver. Never publishes commands.
 * Runs the identical native optimizer that will be selected by live RTI
 * after mathematical qualification, using a recorded source context/seed.
 */
#include "mpc_source_reduced_sqp.hpp"
#include "mpc_qp_failure_context.hpp"
#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>

int main(int argc,char **argv) {
    if(argc<3 || argc>11) {
        std::cerr << "usage: mpc_native_source_sqp_replay CASE seed.txt [iterations] [--no-soc] [controls_out.txt]; seed=- uses captured r1 nominal\n";
        return 2;
    }
    sdu_apex::qp_fixture::FailureReplayContext ctx{};
    if(!sdu_apex::qp_fixture::read_failure_context(
          std::string(argv[1])+"/source_context.mctx",&ctx))return 3;
    MpcModelControl_t controls[PREDICTION_HORIZON]{};
    if(std::string(argv[2])=="-") {
        std::copy_n(ctx.r1_nominal,PREDICTION_HORIZON,controls);
    } else {
        std::ifstream input(argv[2]);
        for(auto &c:controls)
            if(!(input>>c.steering_target_rad>>c.throttle_command))return 4;
    }
    sdu_apex::source_sqp::Options options{};
    options.max_iterations=argc>3 ? std::clamp(std::atoi(argv[3]),1,12):8;
    // Default matches the native source-SQP runtime; override via
    // --ipm-max=90 for the stored 90-iteration reference.
    options.qp_max_iterations=40;
    std::string control_output;
    for(int i=4;i<argc;++i) {
        const std::string option(argv[i]);
        if(option=="--no-soc")options.second_order_correction=false;
        else if(option=="--terminal3")options.terminal_lookahead_steps=3;
        else if(option=="--suffix-fd")options.reuse_unperturbed_fd_prefix=true;
        else if(option=="--full-source-forward")
            options.use_full_source_forward_sensitivities=true;
        else if(option.rfind("--step=",0)==0) {
            try {options.derivative_step=std::stod(option.substr(7));}
            catch (...) {return 2;}
        }
        else if(option=="--opt20")options.control_decision_stages=20;
        else if(option=="--adaptive")options.source_exact_early_stop=true;
        else if(option.rfind("--ipm-max=",0)==0) {
            try { options.qp_max_iterations=std::stoi(option.substr(10)); }
            catch(...) {return 2;}
            if(options.qp_max_iterations<1 || 
               options.qp_max_iterations>200)return 2;
        }
        else if(control_output.empty())control_output=option;
        else return 2;
    }
    // No large derivative capture or file I/O in an ordinary replay.
    const bool dump_derivatives=std::getenv("MPC_SOURCE_SQP_DUMP_JACOBIAN")!=nullptr;
    std::unique_ptr<sdu_apex::source_sqp::DerivativeSnapshot> derivative_snapshot;
    if(dump_derivatives) {
        derivative_snapshot=std::make_unique<
            sdu_apex::source_sqp::DerivativeSnapshot>();
        options.first_derivative_snapshot=derivative_snapshot.get();
    }
    MpcRtiCycleConfiguration_t cfg{};
    cfg.model=ctx.model;
    cfg.model.world_sweep_check=nullptr;
    cfg.model.world_sweep_context=nullptr;
    cfg.model.world_map_rejection_only=0;
    cfg.objective_schedule=&ctx.objective;
    sdu_apex::source_sqp::Statistics stats{};
    auto certified=std::make_unique<MpcRtiNominal_t>();
    MpcModelControl_t selected[PREDICTION_HORIZON]{};
    const bool completed=sdu_apex::source_sqp::optimize(
        &ctx.initial_state,ctx.progress_m,ctx.trajectory.data(),
        ctx.trajectory.size(),ctx.lap_length_m,ctx.prediction_dt,
        &cfg,controls,options,&stats,selected,certified.get());
    if(dump_derivatives) {
        std::ofstream matrix_file("/tmp/mpc_native_source_sqp_first_jacobian.txt");
        matrix_file<<std::setprecision(17);
        for(int row=0;row<60;++row) {
            matrix_file<<"wall "<<row<<" "
                       <<derivative_snapshot->wall[row];
            for(double v:derivative_snapshot->wall_jacobian[row])
                matrix_file<<" "<<v;
            matrix_file<<"\n";
        }
        for(int row=0;row<120;++row) {
            matrix_file<<"rate "<<row<<" "
                       <<derivative_snapshot->rate[row];
            for(double v:derivative_snapshot->rate_jacobian[row])
                matrix_file<<" "<<v;
            matrix_file<<"\n";
        }
        std::ofstream gradient_file("/tmp/mpc_native_source_sqp_first_gradient.txt");
        gradient_file<<std::setprecision(17);
        for(int col=0;col<60;++col)
            gradient_file<<col<<" "<<derivative_snapshot->gradient[col]<<"\n";
    }
    if(!control_output.empty()) {
        std::ofstream output(control_output);
        output<<std::setprecision(12);
        for(const auto &c:selected)
            output<<c.steering_target_rad<<" "<<c.throttle_command<<"\n";
    }
    std::cout<<std::setprecision(14)
       <<"{\"native_source_sqp\":true,\"complete\":"<<(completed?"true":"false")
       <<",\"feasible\":"<<(stats.feasible?"true":"false")
       <<",\"certified_valid\":"<<(certified->valid?"true":"false")
       <<",\"final_certificate_reused\":"
       <<(stats.final_source_certificate_reused?"true":"false")
       <<",\"iterations\":"<<stats.iterations
       <<",\"accepted\":"<<stats.accepted_iterations
       <<",\"first_feasible_iteration\":"<<stats.first_feasible_iteration
       <<",\"source_evaluations\":"<<stats.source_evaluations
       <<",\"source_fd_columns\":"<<stats.source_finite_difference_columns
       <<",\"source_forward_sensitivity_intervals\":"
       <<stats.source_forward_sensitivity_intervals
       <<",\"source_tangent_objective_evaluations\":"
       <<stats.source_tangent_objective_evaluations
       <<",\"full_forward_dual_plant_ms\":"
       <<stats.full_forward_dual_plant_us/1000.
       <<",\"full_forward_tangent_scoring_ms\":"
       <<stats.full_forward_tangent_scoring_us/1000.
       <<",\"qp_solves\":"<<stats.qp_solves
       <<",\"qp_iterations\":"<<stats.qp_iterations
       <<",\"qp_status\":"<<stats.last_qp_status
       <<",\"max_affine_violation\":"<<stats.max_affine_violation
       <<",\"initial_cost\":"<<stats.initial_cost
       <<",\"final_cost\":"<<stats.final_cost
       <<",\"source_exact_early_stop_selected\":"
       <<(stats.source_exact_early_stop_selected?"true":"false")
       <<",\"control_decision_stages\":"
       <<stats.control_decision_stages
       <<",\"terminal_lookahead_steps\":"
       <<stats.terminal_lookahead_steps
       <<",\"initial_tail_wall_V\":"
       <<stats.initial_tail_violation_m2
       <<",\"final_tail_wall_V\":"
       <<stats.final_tail_violation_m2
       <<",\"initial_wall_V\":"<<stats.initial_wall_violation_m2
       <<",\"final_wall_V\":"<<stats.final_wall_violation_m2
       <<",\"initial_rate_V\":"<<stats.initial_rate_violation_m2
       <<",\"final_rate_V\":"<<stats.final_rate_violation_m2
       <<",\"initial_strict_source_valid\":"
       <<(stats.initial_strict_source_valid?"true":"false")
       <<",\"final_strict_source_valid\":"
       <<(stats.final_strict_source_valid?"true":"false")
       <<",\"final_source_failure_stage\":"
       <<stats.final_source_failure_stage
       <<",\"initial_first_failure_stage\":";
    int first=-1,last=-1;
    for(int k=1;k<=PREDICTION_HORIZON;++k) {
        if(first<0 && stats.initial_stage_min_clearance_m[k]<0.)first=k;
        if(last<0 && stats.final_stage_min_clearance_m[k]<0.)last=k;
    }
    std::cout<<first<<",\"final_first_failure_stage\":"<<last
       <<",\"total_elapsed_ms\":"<<stats.total_us/1000.
       <<",\"source_evaluation_ms\":"<<stats.exact_source_evaluation_us/1000.
       <<",\"derivative_ms\":"<<stats.derivative_us/1000.
       <<",\"qp_ms\":"<<stats.qp_us/1000.
       <<",\"cost_prediction_error\":"<<stats.first_cost_prediction_error
       <<",\"accepted_history\":["; 
    for(int k=0;k<stats.iterations;++k) {
        if(k)std::cout<<',';
        std::cout<<"{\"cost\":"<<stats.accepted_cost[k]
             <<",\"wall_v\":"<<stats.accepted_wall_violation_m2[k]
             <<",\"alpha\":"<<stats.accepted_alpha[k]
             <<",\"phase1_slack\":"<<stats.phase1_slack_sum[k]
             <<",\"qp_direction_norm\":"<<stats.first_qp_direction_norm[k]
             <<",\"max_wall_model_defect_m\":"
             <<stats.full_step_max_wall_defect_m[k]
             <<",\"max_rate_model_defect\":"
             <<stats.full_step_max_rate_defect[k]
             <<",\"qp_predicted_wall_V\":"
             <<stats.full_step_local_wall_violation_m2[k]
             <<",\"actual_source_wall_V\":"
             <<stats.full_step_source_wall_violation_m2[k]
             <<"}";
    }
    std::cout<<"],\"initial_stage_clearance_m\":[";
    for(int k=0;k<=PREDICTION_HORIZON;++k) {
        if(k)std::cout<<',';
        std::cout<<stats.initial_stage_min_clearance_m[k];
    }
    std::cout<<"],\"final_stage_clearance_m\":[";
    for(int k=0;k<=PREDICTION_HORIZON;++k) {
        if(k)std::cout<<',';
        std::cout<<stats.final_stage_min_clearance_m[k];
    }
    std::cout<<"]}\n";
    return completed?0:1;
}
