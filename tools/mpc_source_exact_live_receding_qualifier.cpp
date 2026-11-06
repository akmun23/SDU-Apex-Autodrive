// Offline, exact-source receding qualification of the REAL mpc_rti_solve_cycle
// entry point. Includes nominal preparation, two-stage constrained SQP,
// original physical validation, memory shift and a strict 25ms deadline.
// Does not publish ROS controls or claim Unity racing qualification.
#include "mpc_source_exact_nlp.hpp"
#include "mpc_qp_failure_context.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

int main(int argc,char **argv)
{
    if(argc<4 || argc>9) {
        std::cerr<<"usage: mpc_source_exact_live_receding_qualifier "
                 <<"CASE_DIR CONTROLS.txt CYCLES [ITERATIONS=2] "
                 <<"[TRIALS=1] [DEADLINE_MS=22] [EXTRA_CLEARANCE_M=0.03] [PARALLEL_TRIALS=0]\n";
        return 2;
    }
    using nlohmann::json;
    const std::filesystem::path folder=argv[1];
    const int cycles=std::stoi(argv[3]);
    const int iterations=argc>=5?std::stoi(argv[4]):3;
    const int trials=argc>=6?std::stoi(argv[5]):1;
    const double deadline_ms=argc>=7?std::stod(argv[6]):22.;
    const double extra_clearance=argc>=8?std::stod(argv[7]):0.03;
    const int parallel_trials=argc>=9?std::stoi(argv[8]):0;
    if(cycles<1 || cycles>400 || iterations<1 || iterations>3 ||
       trials<1 || trials>32 || deadline_ms<1. || deadline_ms>1000. ||
       extra_clearance<0. || extra_clearance>.1 ||
       !std::isfinite(extra_clearance) ||
       (parallel_trials!=0 && parallel_trials!=1))
        return 2;
    auto ctx=std::make_unique<sdu_apex::qp_fixture::FailureReplayContext>();
    // One original source-NLP fixture. No dependency on the previous
    // controller's QP captures, Riccati solver state or linearized plant.
    if(!sdu_apex::qp_fixture::read_failure_context(
        (folder/"source_context.mctx").string(),ctx.get()))return 3;
    auto memory=std::make_unique<MpcRtiMemory_t>();
    memory->command_epoch_ns=ctx->command_epoch_ns;
    memory->warm_start_controls_valid=1;
    memory->warm_start_control_count=PREDICTION_HORIZON;
    std::ifstream input(argv[2]);
    if(!input)return 4;
    for(auto &u:memory->warm_start_controls)
        if(!(input>>u.steering_target_rad>>u.throttle_command))return 4;
    MpcRtiCycleConfiguration_t cfg{};
    cfg.model=ctx->model;
    cfg.model.world_sweep_check=nullptr;
    cfg.model.world_sweep_context=nullptr;
    cfg.model.world_map_rejection_only=0;
    cfg.source_exact_nlp_live=1;
    cfg.source_exact_nlp_constrained_qp=1;
    const bool full_step=true; // the only runtime nonlinear program
    cfg.source_exact_nlp_full_step_sqp=full_step?1:0;
    cfg.source_exact_nlp_compact_qp=1;
    (void)parallel_trials;
    cfg.source_exact_nlp_qp_regularization=
        std::getenv("MPC_SOURCE_NLP_QP_REGULARIZATION")?
            std::stof(std::getenv("MPC_SOURCE_NLP_QP_REGULARIZATION")):1000.f;
    cfg.source_exact_nlp_qp_soft_weight=
        std::getenv("MPC_SOURCE_NLP_QP_SOFT_WEIGHT")?
            std::stof(std::getenv("MPC_SOURCE_NLP_QP_SOFT_WEIGHT")):1000000.f;
    cfg.source_exact_nlp_iterations=iterations;
    (void)trials;
    cfg.source_exact_nlp_extra_clearance_m=
        static_cast<float>(extra_clearance);
    auto state=ctx->initial_state;
    double progress=ctx->progress_m;
    json observations=json::array();
    double max_cycle_ms=0.,sum_cycle_ms=0.;
    int accepted_count=0,source_parity_failures=0,deadline_misses=0;
    const auto epoch_step=static_cast<std::int64_t>(
        std::llround(1.e9*ctx->prediction_dt));
    for(int tick=0;tick<cycles;++tick) {
        cfg.command_epoch_ns=ctx->command_epoch_ns+epoch_step*tick;
        cfg.absolute_deadline_monotonic_us=
            mpc_rti_monotonic_time_us()+deadline_ms*1000.;
        auto result=std::make_unique<MpcRtiCycleResult_t>();
        const auto status=mpc_rti_solve_cycle(
            &state,progress,ctx->trajectory.data(),
            ctx->trajectory.size(),ctx->lap_length_m,
            ctx->prediction_dt,PREDICTION_HORIZON,
            &cfg,memory.get(),result.get());
        const double cycle_ms=result->total_cycle_us/1000.;
        max_cycle_ms=std::max(max_cycle_ms,cycle_ms);
        sum_cycle_ms+=cycle_ms;
        // Exhausted SQP *iteration admission* is not a missed control
        // deadline. The live solver is allowed to publish its already
        // exact-source-feasible iterate before the hard 25ms deadline.
        // Count only an actually late or deadline-rejected publication.
        const bool missed_deadline=cycle_ms>=deadline_ms ||
            status==MPC_RTI_CYCLE_WAITING_FOR_SOLVER_BUDGET;
        if(missed_deadline)++deadline_misses;
        auto oracle=std::make_unique<MpcRtiNominal_t>();
        MpcRtiCycleConfiguration_t oracle_cfg=cfg;
        oracle_cfg.objective_schedule=&result->comparison_objective;
        int failure_stage=-1;
        const bool accepted=status==MPC_RTI_CYCLE_ACCEPTED_FEASIBLE &&
            result->candidate_control_count==PREDICTION_HORIZON;
        const bool source_valid=accepted &&
            mpc_rti_validate_candidate_plan(
                &state,progress,result->candidate_controls,
                ctx->trajectory.data(),ctx->trajectory.size(),
                ctx->lap_length_m,PREDICTION_HORIZON,
                ctx->prediction_dt,&oracle_cfg,oracle.get(),
                &failure_stage)==MPC_RTI_ROLLOUT_OK && oracle->valid;
        double independently_scored_cost=NAN;
        double max_dynamics_residual=NAN;
        if(source_valid) {
            auto decision=std::make_unique<sdu_apex::source_nlp::Decision>();
            auto score=std::make_unique<sdu_apex::source_nlp::Evaluation>();
            if(!sdu_apex::source_nlp::decision_from_nominal(
                *oracle,decision.get()) ||
               !sdu_apex::source_nlp::evaluate_existing_source_nominal(
                state,progress,ctx->trajectory.data(),
                ctx->trajectory.size(),ctx->lap_length_m,
                ctx->prediction_dt,oracle_cfg,*oracle,*decision,
                score.get()))return 5;
            independently_scored_cost=score->objective;
            max_dynamics_residual=score->max_dynamics_abs;
            if(independently_scored_cost!=
                result->source_exact_nlp_final_source_cost ||
               max_dynamics_residual>1.e-8 ||
               score->minimum_command_rate_margin < -1.e-7 ||
               score->minimum_command_box_margin < -1.e-7)
                ++source_parity_failures;
        }
        const bool publishable=source_valid &&
            !missed_deadline && source_parity_failures==0;
        // Capture a complete same-epoch original-source NLP case at the
        // first rejected receding state. The failing state, reference
        // schedule and the EXACT nominal SQP seed can then be optimized
        // independently without synthesizing or guessing a corner state.
        if(!publishable) {
            if(const char *dump=std::getenv("MPC_SOURCE_SQP_REJECT_DUMP")) {
                std::filesystem::create_directories(dump);
                auto replay=std::make_unique<
                    sdu_apex::qp_fixture::FailureReplayContext>();
                replay->initial_state=state;
                replay->progress_m=progress;
                replay->lap_length_m=ctx->lap_length_m;
                replay->prediction_dt=ctx->prediction_dt;
                replay->model=cfg.model;
                replay->objective=result->comparison_objective;
                replay->trajectory=ctx->trajectory;
                replay->command_epoch_ns=cfg.command_epoch_ns;
                replay->r1_captured=1;
                replay->r1_controls_valid=1;
                std::copy_n(result->nominal_controls,
                    PREDICTION_HORIZON,replay->r1_nominal);
                if(!sdu_apex::qp_fixture::write_failure_context(
                        (std::filesystem::path(dump)/
                         "source_context.mctx").string(),*replay))
                    return 6;
                std::ofstream controls(
                    std::filesystem::path(dump)/"initial.controls");
                controls<<std::setprecision(10);
                for(const auto &u:result->nominal_controls)
                    controls<<u.steering_target_rad<<' '<<
                        u.throttle_command<<'\n';
            }
        }
        observations.push_back({
            {"tick",tick},{"cycle_ms",cycle_ms},
            {"status",int(status)},
            {"candidate_control_count",result->candidate_control_count},
            {"nominal_control_count",result->nominal_control_count},
            {"first_command_steering",result->first_control.steering_target_rad},
            {"first_command_throttle",result->first_control.throttle_command},
            {"source_active",result->source_exact_nlp_active},
            {"publishable",publishable},
            {"exact_source_feasible",result->source_exact_nlp_final_source_feasible!=0},
            {"deadline_exhausted",result->source_exact_nlp_deadline_exhausted!=0},
            {"source_trials",result->source_exact_nlp_source_trial_evaluations},
            {"accepted_sqp_iterations",result->source_exact_nlp_accepted_iterations},
            {"exact_original_cost",result->source_exact_nlp_final_source_cost},
            {"independent_original_cost",std::isfinite(independently_scored_cost)?
                independently_scored_cost:0.},
            {"minimum_corridor_m",result->source_exact_nlp_final_original_margin_m},
            {"source_parity_residual",std::isfinite(max_dynamics_residual)?
                max_dynamics_residual:0.}});
        if(!publishable)break;
        ++accepted_count;
        state=oracle->states[1];
        progress=oracle->progress[1];
    }
    std::cout<<json({
        {"schema","source-exact-live-cycle-receding-qualification/v1"},
        {"source_model_only",true},{"unity_live_qualified",false},
        {"requested_cycles",cycles},{"published_cycles",accepted_count},
        {"required_original_source_extra_clearance_m",extra_clearance},
        {"parallel_exact_source_trials",!full_step && parallel_trials!=0},
        {"full_step_single_nlp_sqp",full_step},
        {"qp_regularization",cfg.source_exact_nlp_qp_regularization},
        {"qp_physical_slack_weight",cfg.source_exact_nlp_qp_soft_weight},
        {"source_cost_parity_failures",source_parity_failures},
        {"deadlines_missed",deadline_misses},
        {"max_cycle_ms",max_cycle_ms},
        {"mean_cycle_ms",sum_cycle_ms/std::max(1,int(observations.size()))},
        {"all_qualified",accepted_count==cycles &&
            source_parity_failures==0 && deadline_misses==0},
        {"epochs",std::move(observations)}
    }).dump(2)<<'\n';
    return 0;
}
