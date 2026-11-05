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
    auto result=std::make_unique<sdu_apex::source_nlp::OfflineResult>();
    if(!sdu_apex::source_nlp::optimize_offline_exact_source(
        ctx->initial_state,ctx->progress_m,ctx->trajectory.data(),
        ctx->trajectory.size(),ctx->lap_length_m,ctx->prediction_dt,
        cfg,controls,options,result.get()))
        return 6;
    json report={
        {"schema","source-exact-single-nlp-sqp/v1"},
        {"case",argv[1]},
        {"single_full_step_per_sqp",true},
        {"alpha_search",false},
        {"multiple_shooting",false},
        {"fallback_controller",false},
        {"source_feasible_initial",result->initial_source_feasible},
        {"source_feasible_final",result->final_source_feasible},
        {"full_step_stalled",result->full_step_sqp_stalled},
        {"original_nlp_kkt_converged",result->source_nlp_kkt_converged},
        {"clearance_requirement_m",options.minimum_source_clearance_m},
        {"final_corridor_margin_m",result->final_minimum_corridor_slack},
        {"initial_cost",result->initial_source_cost},
        {"final_cost",result->final_source_cost},
        {"initial_constraint_violation",result->initial_exact_constraint_violation},
        {"final_constraint_violation",result->final_exact_constraint_violation},
        {"accepted_sqp_iterations",result->accepted_iterations},
        {"exact_source_full_step_evaluations",result->source_trial_evaluations},
        {"history",json::array()}};
    for(const auto &it:result->history)
        report["history"].push_back({
            {"iteration",it.iteration},
            {"accepted",it.accepted},
            {"full_step_rejected",it.full_step_rejected},
            {"source_eval_failed",it.nonlinear_evaluation_failed},
            {"qp_iterations",it.qp_iterations},
            {"qp_converged",it.qp_converged},
            {"qp_primal",it.qp_primal_residual},
            {"qp_dual",it.qp_dual_residual},
            {"original_nlp_stationarity_inf",
             it.source_nlp_stationarity_inf},
            {"original_nlp_inequality_residual_inf",
             it.source_nlp_inequality_residual_inf},
            {"original_nlp_complementarity_inf",
             it.source_nlp_complementarity_inf},
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
            {"calibrated_source_secants",it.calibrated_source_secants},
            {"secant_curvature_added",it.secant_curvature_added},
            {"reused_exact_source_linearization",
             it.reused_exact_source_linearization},
            {"source_rollout_ms",it.source_rollout_wall_ms},
            {"iteration_ms",it.iteration_wall_ms}});
    if(argc>=5) {
        std::ofstream out(argv[4]);
        if(!out)return 7;
        for(const auto &u:result->optimized.controls)
            out<<u.steering_target_rad<<' '<<u.throttle_command<<'\n';
    }
    std::cout<<report.dump(2)<<'\n';
    return 0;
}
