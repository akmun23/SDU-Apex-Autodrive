/*
 * Frozen source-SQP numerical comparison driver. Never publishes commands.
 * Runs the identical native optimizer that will be selected by live RTI
 * after mathematical qualification, using a recorded source context/seed.
 */
#include "mpc_source_reduced_sqp.hpp"
#include "mpc_source_sensitivity.hpp"
#include "mpc_qp_failure_context.hpp"
#include <array>
#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>

int main(int argc,char **argv) {
    if(argc<3 || argc>13) {
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
    bool oracle_only=false;
    bool oracle_details=false;
    std::string source_slip_direction_file;
    bool source_slip_reuse_cache=false;
    for(int i=4;i<argc;++i) {
        const std::string option(argv[i]);
        if(option=="--oracle-only")oracle_only=true;
        else if(option=="--oracle-details") {
            oracle_only=true;oracle_details=true;
        }
        else if(option=="--no-soc")options.second_order_correction=false;
        else if(option=="--gn-hessian")options.use_gauss_newton_hessian=true;
        else if(option.rfind("--source-slip-direction=",0)==0) {
            source_slip_direction_file=option.substr(24);
            oracle_only=true;
        }
        else if(option=="--source-slip-reuse-stage-cache") {
            source_slip_reuse_cache=true;
            oracle_only=true;
        }
        else if(option.rfind("--phase1-gn-damping=",0)==0)
            options.phase1_gn_damping=std::stod(option.substr(20));
        else if(option.rfind("--phase1-gn-band=",0)==0)
            options.phase1_gn_margin_band_m=std::stod(option.substr(17));
        else if(option.rfind("--phase1-inexact-qp=",0)==0)
            options.phase1_inexact_relative_dual=
                std::stod(option.substr(sizeof("--phase1-inexact-qp=")-1));
        else if(option=="--branch-source-two-cut-qp")
            options.use_two_source_branch_qp=true;
        else if(option=="--two-branch-prune-box-dominated")
            options.use_two_branch_box_presolve=true;
        else if(option=="--stage-jacobians")
            options.use_stage_local_jacobians=true;
        else if(option=="--phase1-corridor-only")
            options.use_phase1_corridor_only_scoring=true;
        else if(option=="--exact-stage-corridor-fd") {
            options.use_phase1_corridor_only_scoring=true;
            options.use_exact_stage_corridor_fd=true;
        }
        else if(option=="--compress-positive-drive-timers") {
            options.use_stage_local_jacobians=true;
            options.compress_positive_drive_passive_timer_sensitivities=true;
        }
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
    const char *const dump_prefix=std::getenv("MPC_SOURCE_SQP_DUMP_PREFIX");
    const std::string dump_base=(dump_prefix && *dump_prefix)
        ?std::string(dump_prefix):std::string("/tmp/mpc_native_source_sqp_first");

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
    if(oracle_only) {
        // Independent exact 30-stage SOURCE objective/corridor oracle
        // on arbitrary user-supplied physical steering/throttle controls.
        // This mode DOES NOT call the source SQP or publish any commands.
        auto nominal=std::make_unique<MpcRtiNominal_t>();
        MpcRtiReference_t refs[PREDICTION_HORIZON+1]{};
        if(!mpc_rti_build_nominal_same_epoch(
                &ctx.initial_state,ctx.progress_m,nullptr,controls,
                PREDICTION_HORIZON,ctx.trajectory.data(),
                ctx.trajectory.size(),ctx.lap_length_m,ctx.prediction_dt,
                PREDICTION_HORIZON,&cfg.model,nominal.get(),refs))
            return 7;
        double cost=NAN;
        float clearance=NAN,side[PREDICTION_HORIZON+1][2]{};
        if(!mpc_rti_score_candidate_plan_and_corridor_for_diagnostics(
                nominal.get(),ctx.trajectory.data(),ctx.trajectory.size(),
                ctx.lap_length_m,&cfg,&cost,&clearance,side))
            return 8;
        double violation=0.;
        double minimum=INFINITY;
        for(int stage=1;stage<=PREDICTION_HORIZON;++stage)
            for(int j=0;j<2;++j) {
                const double c=side[stage][j];
                if(!std::isfinite(c))return 9;
                minimum=std::min(minimum,c);
                violation+=std::min(0.,c)*std::min(0.,c);
            }
        double source_slip_cost=0.;
        for(int stage=1;stage<=PREDICTION_HORIZON;++stage) {
            const auto &ref=cfg.objective_schedule->references[stage];
            const double weight=cfg.model.weight_longitudinal_slip*
                (1.+cfg.model.longitudinal_slip_curvature_gain_m*
                    std::abs(ref.path_curvature))*
                (stage==PREDICTION_HORIZON
                    ?cfg.model.terminal_multiplier:1.);
            for(double slip:nominal->states[stage].incoming_wheel_slip_excess_mps)
                source_slip_cost+=weight*slip*slip;
        }
        std::cout<<std::setprecision(17)
            <<"{\"source_exact_nominal\":true,\"cost\":"<<cost
            <<",\"source_longitudinal_slip_cost\":"<<source_slip_cost
            <<",\"wall_violation_m2\":"<<violation
            <<",\"min_wall_clearance_m\":"<<minimum
            <<",\"walls\":[";
        for(int stage=1;stage<=PREDICTION_HORIZON;++stage) {
            for(int side_index=0;side_index<2;++side_index) {
                if(stage!=1||side_index!=0)std::cout<<",";
                std::cout<<side[stage][side_index];
            }
        }
        std::cout<<"]";
        if(!source_slip_direction_file.empty()) {
            std::array<double,60> direction{};
            std::ifstream input(source_slip_direction_file);
            if(!input)return 11;
            for(int col=0;col<60;++col) {
                int index=-1;
                if(!(input>>index>>direction[col]) || index!=col ||
                   !std::isfinite(direction[col]))return 12;
            }
            if(source_slip_reuse_cache) {
                auto precomputed=std::make_unique<
                    sdu_apex::source_sqp::FullSourceDerivatives>();
                if(!sdu_apex::source_sqp::full_source_derivatives(
                    nominal.get(),ctx.trajectory.data(),ctx.trajectory.size(),
                    ctx.lap_length_m,ctx.prediction_dt,&cfg,60,.002,cost,
                    0.,precomputed.get(),false,true,true,false,false))return 14;
            }
            sdu_apex::source_sqp::SourceSlipDirectionalCurvature curvature{};
            if(!sdu_apex::source_sqp::source_slip_directional_curvature(
                nominal.get(),&cfg,ctx.prediction_dt,direction.data(),
                &curvature,source_slip_reuse_cache))return 13;
            std::cout<<",\"source_slip_lagrangian_curvature\":{"
                <<"\"reused_stage_jacobians\":"
                    <<(source_slip_reuse_cache?"true":"false")<<","
                <<"\"gauss_newton\":"<<curvature.gauss_newton
                <<",\"residual_weighted\":"<<curvature.residual_weighted
                <<",\"front_lateral_gauss_newton\":"
                    <<curvature.front_lateral_gauss_newton
                <<",\"front_lateral_residual_weighted\":"
                    <<curvature.front_lateral_residual_weighted
                <<",\"total\":"<<curvature.total
                <<",\"first_source_parity_max_error\":"
                    <<curvature.max_first_source_mismatch
                <<",\"stages\":"<<curvature.stages
                <<",\"elapsed_us\":"<<curvature.elapsed_us<<"}";
        }
        if(oracle_details) {
            std::cout<<",\"source_states\":[";
            for(int stage=0;stage<=PREDICTION_HORIZON;++stage) {
                if(stage)std::cout<<",";
                const auto &state=nominal->states[stage];
                const auto &x=state.plant;
                std::cout<<"{\"stage\":"<<stage
                         <<",\"progress_m\":"<<nominal->progress[stage]
                         <<",\"lateral_error_m\":"<<x.e_y
                         <<",\"heading_error_rad\":"<<x.e_psi
                         <<",\"u_mps\":"<<x.u
                         <<",\"v_mps\":"<<x.v
                         <<",\"yaw_rate\":"<<x.r
                         <<",\"rack_rad\":"<<x.actual_steering_angle
                         <<",\"incoming_branch_flags\":"
                         <<state.incoming_branch_flags
                         <<",\"source_plant_internal_state\":[";
                for(int i=0;i<MPC_PLANT_INTERNAL_STATE_COUNT;++i) {
                    if(i)std::cout<<",";
                    std::cout<<x.source_plant_internal_state[i];
                }
                std::cout<<"],\"wheel_slip_excess\":[";
                for(int i=0;i<4;++i) {
                    if(i)std::cout<<",";
                    std::cout<<state.incoming_wheel_slip_excess_mps[i];
                }
                std::cout<<"]";
                if(stage>0) {
                    MpcRtiCorridorDifferential_t d{};
                    const bool ok=mpc_rti_corridor_differential_at_progress(
                        &x,ctx.trajectory.data(),ctx.trajectory.size(),
                        ctx.lap_length_m,nominal->progress[stage],
                        &cfg.model,stage,&d)!=0;
                    std::cout<<",\"corridor_branch_derivative\":";
                    if(ok) {
                        std::cout<<"{\"lower_d_ey\":"<<d.lower_d_ey
                            <<",\"upper_d_ey\":"<<d.upper_d_ey
                            <<",\"lower_d_epsi\":"<<d.lower_d_epsi
                            <<",\"upper_d_epsi\":"<<d.upper_d_epsi
                            <<",\"lower_d_progress_forward\":"<<d.lower_d_progress_forward
                            <<",\"lower_d_progress_backward\":"<<d.lower_d_progress_backward
                            <<",\"upper_d_progress_forward\":"<<d.upper_d_progress_forward
                            <<",\"upper_d_progress_backward\":"<<d.upper_d_progress_backward
                            <<"}";
                    } else std::cout<<"null";
                }
                std::cout<<"}";
            }
            std::cout<<"]";
        }
        std::cout<<"}"<<std::endl;
        return 0;
    }
    sdu_apex::source_sqp::Statistics stats{};
    auto certified=std::make_unique<MpcRtiNominal_t>();
    MpcModelControl_t selected[PREDICTION_HORIZON]{};
    const bool completed=sdu_apex::source_sqp::optimize(
        &ctx.initial_state,ctx.progress_m,ctx.trajectory.data(),
        ctx.trajectory.size(),ctx.lap_length_m,ctx.prediction_dt,
        &cfg,controls,options,&stats,selected,certified.get());
    if(dump_derivatives) {
        std::ofstream matrix_file(dump_base+"_jacobian.txt");
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
        std::ofstream gradient_file(dump_base+"_gradient.txt");
        gradient_file<<std::setprecision(17);
        for(int col=0;col<60;++col)
            gradient_file<<col<<" "<<derivative_snapshot->gradient[col]<<"\n";
        if(derivative_snapshot->source_cost_hessian_valid) {
            // Export the nonlinear cost model even if the QP solver
            // fails to produce a direction. No extra source evaluation.
            std::ofstream source_hessian(dump_base+"_source_cost_hessian.txt");
            source_hessian<<std::setprecision(17);
            for(int row=0;row<60;++row) {
                for(int col=0;col<60;++col)
                    source_hessian<<(col?" ":"")
                        <<derivative_snapshot->source_cost_hessian[row][col];
                source_hessian<<"\n";
            }
        }
        if(derivative_snapshot->qp_direction_valid) {
            std::ofstream direction_file(dump_base+"_qp_direction.txt");
            direction_file<<std::setprecision(17)
                          <<"# normalized_delta_y steering_scale=.15 throttle_scale=.25 "
                          <<"phase1="<<(derivative_snapshot->qp_was_phase1?1:0)<<"\n";
            for(int col=0;col<60;++col)
                direction_file<<col<<" "
                              <<derivative_snapshot->qp_direction[col]<<"\n";
            std::ofstream hessian_file(dump_base+"_qp_hessian.txt");
            hessian_file<<std::setprecision(17);
            for(int row=0;row<60;++row) {
                for(int col=0;col<60;++col)
                    hessian_file<<(col?" ":"")
                                <<derivative_snapshot->qp_hessian[row][col];
                hessian_file<<"\n";
            }
            if(derivative_snapshot->qp_wall_duals_valid) {
                std::ofstream dual_file(dump_base+"_qp_wall_duals.txt");
                dual_file<<std::setprecision(17)
                    <<"# QP wall constraint duals lambda>=0; "
                    <<"elastic="<<int(derivative_snapshot->qp_wall_duals_from_elastic)
                    <<" phase1="<<int(derivative_snapshot->qp_was_phase1)
                    <<" (NOT duals from source-nonlinear optimum)\n";
                for(int row=0;row<60;++row)
                    dual_file<<row<<" "<<derivative_snapshot->qp_wall_duals[row]
                             <<"\n";
            }
        }
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
       <<",\"two_branch_source_probes\":"<<stats.two_branch_source_probes
       <<",\"two_branch_relinearizations\":"<<stats.two_branch_relinearizations
       <<",\"two_branch_qp_solves\":"<<stats.two_branch_qp_solves
       <<",\"two_branch_box_redundant_wall_rows\":"<<stats.two_branch_box_redundant_wall_rows
       <<",\"two_branch_selected_directions\":"<<stats.two_branch_selected_directions
       <<",\"two_branch_initial_max_wall_defect_m\":"<<stats.two_branch_initial_max_wall_defect_m
       <<",\"two_branch_selected_wall_v\":"<<stats.two_branch_selected_wall_v
       <<",\"near_feasible_projection_trials\":"<<stats.near_feasible_source_projection_trials
       <<",\"near_feasible_projection_accepts\":"<<stats.near_feasible_source_projection_accepts
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
             <<",\"accepted_qp_cost_model_valid\":"
             <<(stats.accepted_cost_model_valid[k]?"true":"false")
             <<",\"accepted_qp_cost_predicted_delta\":"
             <<stats.accepted_cost_predicted_change[k]
             <<",\"accepted_source_cost_actual_delta\":"
             <<stats.accepted_cost_actual_change[k]
             <<",\"accepted_qp_cost_model_error\":"
             <<stats.accepted_cost_prediction_error[k]
             <<",\"qp_cost_model_valid\":"
             <<(stats.full_step_cost_model_valid[k]?"true":"false")
             <<",\"qp_cost_predicted_delta\":"
             <<stats.full_step_cost_predicted_change[k]
             <<",\"source_cost_actual_delta\":"
             <<stats.full_step_cost_actual_change[k]
             <<",\"source_cost_model_error\":"
             <<stats.full_step_cost_prediction_error[k]
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
