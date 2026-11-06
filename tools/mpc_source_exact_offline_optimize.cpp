// Offline qualification of the same full-step source nonlinear SQP
// used by the ROS controller. One original NLP, no alternate optimizers.
#include "mpc_source_exact_offline_solver.hpp"
#include "mpc_qp_failure_context.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <memory>
#include <string>

using nlohmann::json;
int main(int argc,char **argv)
{
    if(argc<3 || argc>5) {
        std::cerr<<"usage: mpc_source_exact_offline_optimize "
            "CASE_DIR CONTROLS.txt|- [MAX_SQP=4] [OUTPUT_CONTROLS.txt]\n";
        return 2;
    }
    namespace fs=std::filesystem;
    auto ctx=std::make_unique<sdu_apex::qp_fixture::FailureReplayContext>();
    if(!sdu_apex::qp_fixture::read_failure_context(
        (fs::path(argv[1])/"source_context.mctx").string(),ctx.get()))
        return 3;
    MpcRtiCycleConfiguration_t cfg{};
    cfg.model=ctx->model;
    cfg.model.world_sweep_check=nullptr;
    cfg.model.world_sweep_context=nullptr;
    cfg.model.world_map_rejection_only=0;
    cfg.objective_schedule=&ctx->objective;
    MpcModelControl_t controls[PREDICTION_HORIZON]{};
    if(std::string(argv[2])=="-")
        std::copy_n(ctx->r1_nominal,PREDICTION_HORIZON,controls);
    else {
        std::ifstream in(argv[2]);
        if(!in)return 4;
        for(auto &u:controls)
            if(!(in>>u.steering_target_rad>>u.throttle_command))
                return 5;
    }
    sdu_apex::source_nlp::OfflineOptions options{};
    options.maximum_iterations=argc>=4?std::stoi(argv[3]):3;
    if(const char *p=std::getenv("MPC_SOURCE_NLP_QP_REGULARIZATION"))
        options.source_qp_curvature_regularization=std::stod(p);
    if(const char *p=std::getenv("MPC_SOURCE_NLP_QP_SOFT_WEIGHT"))
        options.source_qp_soft_physical_weight=std::stod(p);
    if(const char *p=std::getenv("MPC_SOURCE_NLP_MIN_CLEARANCE_M"))
        options.minimum_source_clearance_m=std::stod(p);
    if(const char *p=std::getenv("MPC_SOURCE_NLP_COMPACT_QP"))
        options.use_source_compact_qp_linearization=std::string(p)!="0";
    options.record_primal_trace=std::getenv("MPC_SOURCE_NLP_RECORD_PRIMAL_TRACE")!=nullptr;
    options.source_slip_epigraph=std::getenv("MPC_SOURCE_NLP_SLIP_EPIGRAPH")!=nullptr;
    options.exact_lagrangian_hessian=std::getenv("MPC_SOURCE_NLP_FULL_HESSIAN")!=nullptr;
    options.strict_accepted_source_branch_curvature_guard=
        std::getenv("MPC_SOURCE_NLP_STRICT_BRANCH_CURVATURE_GUARD")!=nullptr;
    options.source_discontinuous_secant_guard=
        std::getenv("MPC_SOURCE_NLP_DISCONTINUOUS_SECANT_GUARD")!=nullptr;
    options.source_infeasible_second_order_correction=
        std::getenv("MPC_SOURCE_NLP_INFEASIBLE_SOURCE_SOC")!=nullptr;
    options.source_trial_relinearized_correction=
        std::getenv("MPC_SOURCE_NLP_TRIAL_RELINEARIZATION")!=nullptr;
    if(const char *p=std::getenv("MPC_SOURCE_NLP_QP_MAX_ITERATIONS"))
        options.source_qp_max_iterations=std::stoi(p);
    auto result=std::make_unique<sdu_apex::source_nlp::OfflineResult>();
    if(!sdu_apex::source_nlp::optimize_offline_exact_source(
        ctx->initial_state,ctx->progress_m,ctx->trajectory.data(),
        ctx->trajectory.size(),ctx->lap_length_m,ctx->prediction_dt,
        cfg,controls,options,result.get()))
        return 6;
    json report={
        {"schema","source-exact-single-nlp-sqp/v1"},
        {"case",argv[1]},
        {"single_full_step_per_sqp",false},
        {"second_order_correction_globalization",true},
        {"alpha_search",true},
        {"fixed_alpha_grid",false},
        {"multiple_shooting",false},
        {"stationarity_coordinates","steering/0.15rad, throttle/0.25"},
        {"preferred_reserve_in_original_kkt",false},
        {"nonlinear_acceptance_objective","original_racing_cost"},
        {"fallback_controller",false},
        {"strict_accepted_source_branch_curvature_guard",
         options.strict_accepted_source_branch_curvature_guard},
        {"source_discontinuous_secant_guard",
         options.source_discontinuous_secant_guard},
        {"source_infeasible_second_order_correction",
         options.source_infeasible_second_order_correction},
        {"source_trial_relinearized_correction",
         options.source_trial_relinearized_correction},
        {"source_feasible_initial",result->initial_source_feasible},
        {"solver_status",static_cast<int>(result->solver_status)},
        {"linearization_failed",result->linearization_failed},
        {"qp_failed",result->qp_failed},
        {"deadline_exhausted",result->deadline_exhausted},
        {"iteration_admission_stopped",result->iteration_admission_stopped},
        {"source_feasible_final",result->final_source_feasible},
        {"full_step_stalled",result->full_step_sqp_stalled},
        {"numerical_failure",result->numerical_failure},
        {"termination_reason",result->termination_reason},
        {"original_nlp_kkt_converged",result->source_nlp_kkt_converged},
        {"clearance_requirement_m",options.minimum_source_clearance_m},
        {"final_corridor_margin_m",result->final_minimum_corridor_slack},
        {"initial_cost",result->initial_source_cost},
        {"initial_corridor_margin_m",result->initial_minimum_corridor_slack},
        {"final_cost",result->final_source_cost},
        {"initial_constraint_violation",result->initial_exact_constraint_violation},
        {"final_constraint_violation",result->final_exact_constraint_violation},
        {"accepted_sqp_iterations",result->accepted_iterations},
        {"source_constraint_layout_resets",result->source_constraint_layout_resets},
        {"accepted_source_branch_curvature_resets",
         result->accepted_source_branch_curvature_resets},
        {"accepted_source_discontinuous_secant_skips",
         result->accepted_source_discontinuous_secant_skips},
        {"exact_source_full_step_evaluations",result->source_trial_evaluations},
        {"exact_source_evaluations",result->exact_source_evaluations},
        {"history",json::array()}};
    for(const auto &it:result->history)
        report["history"].push_back({
            {"iteration",it.iteration},
            {"accepted",it.accepted},
            {"full_step_evaluated",it.full_step_evaluated},
            {"full_step_accepted",it.full_step_accepted},
            {"normal_step_computed",it.normal_step_computed},
            {"normal_only_step",it.normal_only_step},
            {"normal_primal_residual",it.normal_primal_residual},
            {"normal_dual_residual",it.normal_dual_residual},
            {"normal_step_accepted",it.normal_step_accepted},
            {"normal_linearized_violation",it.normal_linearized_violation},
            {"linearized_physical_violation_inf",it.linearized_physical_violation_inf},
            {"original_duals_trustworthy",it.original_duals_trustworthy},
            {"linearization_failed",it.linearization_failed},
            {"qp_failed",it.qp_failed},
            {"qp_attempted",it.qp_attempted},
            {"full_step_rejected",it.full_step_rejected},
            {"source_eval_failed",it.nonlinear_evaluation_failed},
            {"second_order_correction_attempted",it.second_order_correction_attempted},
            {"second_order_correction_accepted",it.second_order_correction_accepted},
            {"safeguarded_step_length",it.safeguarded_step_length},
            {"safeguarded_evaluations",it.safeguarded_evaluations},
            {"lagrangian_hessian_wall_ms",it.lagrangian_hessian_wall_ms},
            {"lagrangian_hessian_valid",it.lagrangian_hessian_valid},
            {"hessian_source_directional_evaluations",it.hessian_source_directional_evaluations},
            {"hessian_quadratic_consistency_relative",it.hessian_quadratic_consistency_relative},
            {"rejected_gradient_bfgs_wall_ms",
             it.rejected_gradient_bfgs_wall_ms},
            {"rejected_gradient_bfgs_updated",
             it.rejected_gradient_bfgs_updated},
            {"rejected_gradient_source_branch_changes",
             it.rejected_gradient_source_branch_changes},
            {"rejected_gradient_bfgs_damping",
             it.rejected_gradient_bfgs_damping},
            {"lagrangian_hessian_min_eigenvalue",it.lagrangian_hessian_min_eigenvalue},
            {"second_order_correction_cost",it.second_order_correction_cost},
            {"second_order_correction_clearance",it.second_order_correction_clearance},
            {"qp_iterations",it.qp_iterations},
            {"qp_converged",it.qp_converged},
            {"invalid_qp_duals_discarded",it.invalid_qp_duals_discarded},
            {"qp_primal",it.qp_primal_residual},
            {"qp_dual",it.qp_dual_residual},
            {"qp_complementarity",it.qp_complementarity_residual},
            {"qp_hessian_min_eigenvalue",it.qp_hessian_min_eigenvalue},
            {"qp_hessian_max_eigenvalue",it.qp_hessian_max_eigenvalue},
            {"physical_violation_before",it.physical_violation_before},
            {"physical_violation_trial",it.physical_violation_trial},
            {"physical_violation_after",it.physical_violation_after},
            {"maximum_hard_violation_before",it.maximum_hard_violation_before},
            {"maximum_hard_violation_after",it.maximum_hard_violation_after},
            {"accepted_step_inf_scaled",it.accepted_step_inf_scaled},
            {"exact_source_evaluations",it.exact_source_evaluations},
            {"qp_tolerance_used",it.qp_tolerance_used},
            {"lagrangian_bfgs_updated",it.lagrangian_bfgs_updated},
            {"lagrangian_bfgs_damping",it.lagrangian_bfgs_damping},
            {"original_nlp_stationarity_inf",
             it.source_nlp_stationarity_inf},
            {"original_nlp_inequality_residual_inf",
             it.source_nlp_inequality_residual_inf},
            {"original_nlp_complementarity_inf",
             it.source_nlp_complementarity_inf},
            {"original_nlp_dual_feasibility_inf",it.source_nlp_dual_feasibility_inf},
            {"predicted_cost_change",it.predicted_cost_change},
            {"actual_cost_change",it.actual_cost_change},
            {"cost_before",it.cost_before},
            {"trial_cost",it.cost_trial},
            {"cost_after",it.cost_after},
            {"violation_before",it.violation_before},
            {"trial_violation",it.violation_trial},
            {"violation_after",it.violation_after},
            {"minimum_source_clearance_after",it.corridor_slack_after},
            {"linearization_ms",it.linearization_wall_ms},
            {"qp_ms",it.qp_wall_ms},
            {"qp_regularization_used",it.qp_regularization_used},
            {"model_predicted_hard_violation",
             it.model_predicted_hard_violation},
            {"model_actual_over_predicted_reduction",
             std::isfinite(it.model_actual_over_predicted_reduction)?
             it.model_actual_over_predicted_reduction:-1.e100},
            {"model_maximum_hard_constraint_error",
             it.model_maximum_hard_constraint_error},
            {"source_constraint_path_model_valid",it.source_constraint_path_model_valid},
            {"source_constraint_path_first_invalid_stage",it.source_constraint_path_first_invalid_stage},
            {"physical_violation_directional_first",it.physical_violation_directional_first},
            {"physical_violation_directional_second",it.physical_violation_directional_second},
            {"one_sided_constraint_direction_disagreement_inf",it.one_sided_constraint_direction_disagreement_inf},
            {"model_maximum_quadratic_footprint_error_m",it.model_maximum_quadratic_footprint_error_m},
            {"calibrated_source_secants",it.calibrated_source_secants},
            {"secant_curvature_added",it.secant_curvature_added},
            {"reused_exact_source_linearization",
             it.reused_exact_source_linearization},
            {"source_constraint_layout_reset",
             it.source_constraint_layout_reset},
            {"accepted_source_branch_curvature_reset",
             it.accepted_source_branch_curvature_reset},
            {"accepted_source_discontinuous_secant_skipped",
             it.accepted_source_discontinuous_secant_skipped},
            {"accepted_source_physical_branch_changes",
             it.accepted_source_physical_branch_changes},
            {"accepted_source_peak_branch_changes",
             it.accepted_source_peak_branch_changes},
            {"accepted_source_reference_branch_changes",
             it.accepted_source_reference_branch_changes},
            {"source_rollout_ms",it.source_rollout_wall_ms},
            {"iteration_ms",it.iteration_wall_ms}});
    if(options.record_primal_trace) {
        report["final_physical_multipliers"]=result->final_physical_multipliers;
        report["primal_trace_origin"]="deterministic_offline_reconstruction";
        report["primal_trace"]=json::array();
        const bool controls_only_trace=std::getenv("MPC_SOURCE_NLP_TRACE_CONTROLS_ONLY")!=nullptr;
        report["primal_trace_state_storage"]=controls_only_trace?
            "condensed_controls_only":"source_nodes_and_controls";
        const auto primal=[controls_only_trace](const sdu_apex::source_nlp::Decision &d) {
            json j={{"controls",json::array()}};
            if(!controls_only_trace)j["nodes"]=json::array();
            for(int k=0;k<PREDICTION_HORIZON;++k) {
                j["controls"].push_back({d.controls[k].steering_target_rad,
                                        d.controls[k].throttle_command});
                if(!controls_only_trace)j["nodes"].push_back(d.nodes[k].z);
            }
            return j;
        };
        for(const auto &t:result->primal_trace) {
            json safeguarded=json::array();
            for(size_t i=0;i<t.safeguarded_trials.size();++i)
                safeguarded.push_back({{"alpha",t.safeguarded_step_lengths[i]},
                    {"source_valid",t.safeguarded_valid[i]},{"primal",primal(t.safeguarded_trials[i])}});
            report["primal_trace"].push_back({{"iteration",t.iteration},
                {"accepted",t.accepted},{"trial_evaluated",t.trial_evaluated},
                {"warm_start",primal(t.warm_start)},{"physical_multipliers",t.physical_multipliers},
                {"trial",primal(t.trial)},
                {"normal_evaluated",t.normal_evaluated},{"normal_accepted",t.normal_accepted},
                {"normal",primal(t.normal)},
                {"correction_evaluated",t.correction_evaluated},
                {"correction_accepted",t.correction_accepted},
                {"correction",primal(t.correction)},
                {"safeguarded_trials",safeguarded},{"safeguarded_accepted",t.safeguarded_accepted}});
        }
    }
    if(argc>=5) {
        std::ofstream out(argv[4]);
        if(!out)return 7;
        out<<std::setprecision(10);
        for(const auto &u:result->optimized.controls)
            out<<u.steering_target_rad<<' '<<u.throttle_command<<'\n';
    }
    std::cout<<report.dump(2)<<'\n';
    return 0;
}
