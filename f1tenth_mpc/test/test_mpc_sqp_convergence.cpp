#include "mpc_rti.h"
#include "mpc_source_reduced_sqp.hpp"
#include "mpc_source_sensitivity.hpp"
#include "../src/mpc_rti_internal.hpp"
#include "mpc_qp_convexity.hpp"
#include <gtest/gtest.h>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <memory>

namespace {
MpcRtiCycleConfiguration_t configuration() {
    MpcRtiCycleConfiguration_t c{};
    c.model.weight_e_y=4500.f;c.model.weight_e_psi=1450.f;
    c.model.weight_u=45.f;c.model.weight_u_overspeed=20.f;
    c.model.weight_v=5.f;c.model.weight_r=7.8125f;
    c.model.weight_steering_target=5.f;c.model.weight_throttle_command=.1f;
    c.model.weight_steering_slew=1500.f;c.model.weight_throttle_slew=50.f;
    c.model.terminal_multiplier=3.f;
    c.model.max_speed_mps=c.model.active_speed_ceiling_mps=16.f;
    c.model.max_steering_rad=SOURCE_MAX_STEERING_RAD;
    c.model.corridor_margin_m=c.model.first_prediction_corridor_margin_m=.05f;
    c.solver.rho=c.solver.rho_u=7.f;c.solver.rho_max=32.f;
    c.solver.tolerance=.001f;c.solver.max_iterations=150;
    c.solver.over_relaxation=1.6f;
    c.enable_nonlinear_line_search=1;c.line_search_max_backtracks=4;
    c.line_search_command_epoch_valid=1;c.max_sqp_iterations=2;
    return c;
}
struct Straight {
    MpcTrajectorySample_t track[3]={
        {0.,0.,0.,0.,0.,2.,4.,4.,0.},
        {10.,10.,0.,0.,0.,2.,4.,4.,0.},
        {20.,20.,0.,0.,0.,2.,4.,4.,0.}};
    size_t count=3;double lap{};
    MpcRtiState_t initial{};
    Straight() {
        EXPECT_TRUE(mpc_trajectory_prepare(track,&count,&lap));
        initial.plant.e_y=.1f;initial.plant.u=2.f;
        initial.plant.delayed_throttle_command_1=.1f;
        initial.plant.delayed_throttle_command_2=.1f;
        EXPECT_TRUE(vehicle_model_initialize_plant_state(&initial.plant));
    }
};
}


TEST(MpcSourceReducedSqp, NativeCycleUsesFullSourceAndIndependentStrictValidator)
{
    Straight straight;
    auto config=configuration();
    config.native_source_reduced_sqp=1;
    config.native_source_reduced_sqp_iterations=2;
    auto memory=std::make_unique<MpcRtiMemory_t>();
    auto result=std::make_unique<MpcRtiCycleResult_t>();
    const auto status=mpc_rti_solve_cycle(
        &straight.initial,0.,straight.track,straight.count,straight.lap,
        PREDICTION_DT_SECONDS,PREDICTION_HORIZON,&config,memory.get(),
        result.get());
    EXPECT_EQ(result->native_source_sqp_active,1);
    EXPECT_EQ(result->native_source_sqp_fd_columns,
              60*result->sqp_iterations_completed);
    EXPECT_EQ(result->jacobian_source_sensitivity_calls,0)
        << "native source SQP must bypass old Riccati R1/R2 Jacobians";
    EXPECT_GT(result->native_source_sqp_source_evaluations,0);
    EXPECT_GT(result->native_source_sqp_qp_solves,0);
    ASSERT_EQ(status,MPC_RTI_CYCLE_ACCEPTED_FEASIBLE)
        << "result status " << result->status
        << ", initial wall V "
        << result->native_source_sqp_initial_wall_violation_m2
        << ", final wall V "
        << result->native_source_sqp_final_wall_violation_m2;
    ASSERT_EQ(result->candidate_control_count,PREDICTION_HORIZON);
    ASSERT_TRUE(memory->nominal.valid);
    auto independent=std::make_unique<MpcRtiNominal_t>();
    int failure=-1;
    EXPECT_EQ(mpc_rti_validate_candidate_plan(&straight.initial,0.,
        result->candidate_controls,straight.track,straight.count,
        straight.lap,PREDICTION_HORIZON,PREDICTION_DT_SECONDS,
        &config,independent.get(),&failure),MPC_RTI_ROLLOUT_OK);
    EXPECT_NEAR(independent->validation_objective,
        result->nonlinear_objective,1.e-3);
    EXPECT_GE(independent->validation_minimum_corridor_slack_m,-1.e-6f);
    EXPECT_FLOAT_EQ(result->first_control.steering_target_rad,
        result->candidate_controls[0].steering_target_rad);
    EXPECT_FLOAT_EQ(result->first_control.throttle_command,
        result->candidate_controls[0].throttle_command);
    // The actuator target is not constrained to the rack angle within one
    // source stage. The source steering plant already respects its physical
    // slew internally: verify rack evolution, NOT an artificial input trust
    // region. Full legal target +/-0.5236 rad is always available.
    const double physical_rack_step=3.2*double(PREDICTION_DT_SECONDS);
    for(int k=0;k<PREDICTION_HORIZON;++k) {
        const auto &plant=independent->states[k].plant;
        const auto &next=independent->states[k+1].plant;
        const auto &u=result->candidate_controls[k];
        EXPECT_LE(std::abs(double(u.steering_target_rad)),
            config.model.max_steering_rad+1e-5) << k;
        EXPECT_LE(std::abs(double(next.actual_steering_angle)-
            plant.actual_steering_angle),physical_rack_step+1e-5) << k;
        EXPECT_LE(std::abs(double(u.throttle_command)-
            plant.delayed_throttle_command_2),
            config.model.max_throttle_command_step+1e-5) << k;
    }
}

TEST(MpcSourceReducedSqp, FeasibleOnePassSkipsUnneededSocAndPreservesSourceCertificate)
{
    Straight straight;
    auto config=configuration();
    MpcModelControl_t seed[PREDICTION_HORIZON]{};
    for(int k=0;k<PREDICTION_HORIZON;++k)
        seed[k].throttle_command=.1f;
    auto initial=std::make_unique<MpcRtiNominal_t>();
    MpcRtiReference_t refs[PREDICTION_HORIZON+1]{};
    ASSERT_TRUE(mpc_rti_build_nominal_same_epoch(
        &straight.initial,0.,nullptr,seed,PREDICTION_HORIZON,
        straight.track,straight.count,straight.lap,
        PREDICTION_DT_SECONDS,PREDICTION_HORIZON,
        &config.model,initial.get(),refs));
    auto schedule=std::make_unique<MpcRtiObjectiveSchedule_t>();
    ASSERT_TRUE(mpc_rti_build_objective_schedule(initial.get(),refs,
        &config.model,schedule.get()));
    config.objective_schedule=schedule.get();
    MpcModelControl_t first[PREDICTION_HORIZON]{};
    MpcModelControl_t second[PREDICTION_HORIZON]{};
    auto certified_a=std::make_unique<MpcRtiNominal_t>();
    auto certified_b=std::make_unique<MpcRtiNominal_t>();
    sdu_apex::source_sqp::Statistics a{},b{};
    sdu_apex::source_sqp::Options opts{};
    opts.max_iterations=1;
    opts.use_full_source_forward_sensitivities=true;
    opts.second_order_correction=true;
    ASSERT_TRUE(sdu_apex::source_sqp::optimize(&straight.initial,0.,
        straight.track,straight.count,straight.lap,
        PREDICTION_DT_SECONDS,&config,seed,opts,&a,first,
        certified_a.get()));
    ASSERT_TRUE(a.initial_strict_source_valid);
    EXPECT_EQ(a.qp_solves,1);
    opts.second_order_correction=false;
    ASSERT_TRUE(sdu_apex::source_sqp::optimize(&straight.initial,0.,
        straight.track,straight.count,straight.lap,
        PREDICTION_DT_SECONDS,&config,seed,opts,&b,second,
        certified_b.get()));
    ASSERT_TRUE(a.feasible && b.feasible);
    ASSERT_TRUE(certified_a->valid && certified_b->valid);
    EXPECT_EQ(a.qp_solves,b.qp_solves);
    EXPECT_DOUBLE_EQ(a.final_cost,b.final_cost);
    EXPECT_EQ(a.final_wall_violation_m2,0.);
    EXPECT_EQ(b.final_wall_violation_m2,0.);
    for(int k=0;k<PREDICTION_HORIZON;++k) {
        EXPECT_FLOAT_EQ(first[k].steering_target_rad,
                        second[k].steering_target_rad);
        EXPECT_FLOAT_EQ(first[k].throttle_command,
                        second[k].throttle_command);
    }
}

TEST(MpcSourceReducedSqp, TwoSourcePassesRelinearizeAndRefreshGnHessian)
{
    Straight track;
    auto config=configuration();
    MpcModelControl_t seed[PREDICTION_HORIZON]{};
    for(int k=0;k<PREDICTION_HORIZON;++k) {
        seed[k].steering_target_rad=.02f;
        seed[k].throttle_command=.12f;
    }
    auto nominal=std::make_unique<MpcRtiNominal_t>();
    MpcRtiReference_t refs[PREDICTION_HORIZON+1]{};
    ASSERT_TRUE(mpc_rti_build_nominal_same_epoch(
        &track.initial,0.,nullptr,seed,PREDICTION_HORIZON,
        track.track,track.count,track.lap,PREDICTION_DT_SECONDS,
        PREDICTION_HORIZON,&config.model,nominal.get(),refs));
    auto schedule=std::make_unique<MpcRtiObjectiveSchedule_t>();
    ASSERT_TRUE(mpc_rti_build_objective_schedule(
        nominal.get(),refs,&config.model,schedule.get()));
    config.objective_schedule=schedule.get();
    sdu_apex::source_sqp::Options options{};
    options.max_iterations=2;
    options.use_full_source_forward_sensitivities=true;
    options.use_stage_local_jacobians=true;
    options.use_gauss_newton_hessian=true;
    options.source_exact_early_stop=false;
    auto stats=std::make_unique<sdu_apex::source_sqp::Statistics>();
    auto certified=std::make_unique<MpcRtiNominal_t>();
    MpcModelControl_t output[PREDICTION_HORIZON]{};
    ASSERT_TRUE(sdu_apex::source_sqp::optimize(
        &track.initial,0.,track.track,track.count,track.lap,
        PREDICTION_DT_SECONDS,&config,seed,options,stats.get(),
        output,certified.get()));
    EXPECT_EQ(stats->iterations,2)
        << "do not skip the second SQP relinearization on an easy start";
    EXPECT_EQ(stats->source_forward_sensitivity_intervals,120);
    EXPECT_EQ(stats->source_tangent_objective_evaluations,240);
    EXPECT_GT(stats->qp_solves,0);
    ASSERT_TRUE(stats->feasible);
    ASSERT_TRUE(certified->valid);
    EXPECT_GE(certified->validation_minimum_corridor_slack_m,0.);
}

TEST(MpcSourceReducedSqp, TwentyDecisionStagesPreserveThirtySourceStages)
{
    Straight straight;
    auto config=configuration();
    MpcModelControl_t seed[PREDICTION_HORIZON]{};
    for(int k=0;k<PREDICTION_HORIZON;++k) {
        seed[k].steering_target_rad=0.f;
        seed[k].throttle_command=.1f;
    }
    auto baseline=std::make_unique<MpcRtiNominal_t>();
    MpcRtiReference_t refs[PREDICTION_HORIZON+1]{};
    ASSERT_TRUE(mpc_rti_build_nominal_same_epoch(&straight.initial,0.,
        nullptr,seed,PREDICTION_HORIZON,straight.track,straight.count,
        straight.lap,PREDICTION_DT_SECONDS,PREDICTION_HORIZON,
        &config.model,baseline.get(),refs));
    auto schedule=std::make_unique<MpcRtiObjectiveSchedule_t>();
    ASSERT_TRUE(mpc_rti_build_objective_schedule(baseline.get(),refs,
        &config.model,schedule.get()));
    config.objective_schedule=schedule.get();

    sdu_apex::source_sqp::Options options{};
    options.max_iterations=2;
    options.control_decision_stages=20;
    options.reuse_unperturbed_fd_prefix=true;
    sdu_apex::source_sqp::Statistics stats{};
    auto certified=std::make_unique<MpcRtiNominal_t>();
    MpcModelControl_t selected[PREDICTION_HORIZON]{};
    ASSERT_TRUE(sdu_apex::source_sqp::optimize(&straight.initial,0.,
        straight.track,straight.count,straight.lap,
        PREDICTION_DT_SECONDS,&config,seed,options,&stats,selected,
        certified.get()));
    EXPECT_EQ(stats.control_decision_stages,20);
    EXPECT_EQ(stats.source_finite_difference_columns,40*stats.iterations);
    for(int k=20;k<PREDICTION_HORIZON;++k) {
        EXPECT_FLOAT_EQ(selected[k].steering_target_rad,
                        seed[k].steering_target_rad);
        EXPECT_FLOAT_EQ(selected[k].throttle_command,
                        seed[k].throttle_command);
    }
    ASSERT_TRUE(stats.feasible);
    ASSERT_TRUE(certified->valid);
    EXPECT_EQ(certified->horizon,PREDICTION_HORIZON);
    auto independent=std::make_unique<MpcRtiNominal_t>();
    int failure=-1;
    EXPECT_EQ(mpc_rti_validate_candidate_plan(&straight.initial,0.,selected,
        straight.track,straight.count,straight.lap,PREDICTION_HORIZON,
        PREDICTION_DT_SECONDS,&config,independent.get(),&failure),
        MPC_RTI_ROLLOUT_OK);
    EXPECT_NEAR(independent->validation_objective,stats.final_cost,1e-3);
}

TEST(MpcSourceReducedSqp, FullSourcePhysicalForwardDerivativesMatchFiniteDifference)
{
    Straight straight;
    auto config=configuration();
    MpcModelControl_t seed[PREDICTION_HORIZON]{};
    for(int k=0;k<PREDICTION_HORIZON;++k)
        seed[k].throttle_command=.1f;
    auto nominal=std::make_unique<MpcRtiNominal_t>();
    MpcRtiReference_t references[PREDICTION_HORIZON+1]{};
    ASSERT_TRUE(mpc_rti_build_nominal_same_epoch(
        &straight.initial,0.,nullptr,seed,PREDICTION_HORIZON,
        straight.track,straight.count,straight.lap,PREDICTION_DT_SECONDS,
        PREDICTION_HORIZON,&config.model,nominal.get(),references));
    auto schedule=std::make_unique<MpcRtiObjectiveSchedule_t>();
    ASSERT_TRUE(mpc_rti_build_objective_schedule(
        nominal.get(),references,&config.model,schedule.get()));
    config.objective_schedule=schedule.get();

    auto fd=std::make_unique<sdu_apex::source_sqp::DerivativeSnapshot>();
    auto forward=std::make_unique<sdu_apex::source_sqp::DerivativeSnapshot>();
    auto run=[&](bool use_forward,
                 sdu_apex::source_sqp::DerivativeSnapshot *snapshot,
                 sdu_apex::source_sqp::Statistics *stats) {
        sdu_apex::source_sqp::Options opts{};
        opts.max_iterations=1;
        opts.control_decision_stages=20;
        opts.reuse_unperturbed_fd_prefix=!use_forward;
        opts.use_full_source_forward_sensitivities=use_forward;
        opts.first_derivative_snapshot=snapshot;
        MpcModelControl_t selected[PREDICTION_HORIZON]{};
        auto certified=std::make_unique<MpcRtiNominal_t>();
        const bool ok=sdu_apex::source_sqp::optimize(
            &straight.initial,0.,straight.track,straight.count,
            straight.lap,PREDICTION_DT_SECONDS,&config,seed,
            opts,stats,selected,certified.get());
        EXPECT_TRUE(ok);
        EXPECT_TRUE(certified->valid);
        EXPECT_TRUE(stats->feasible);
        return ok;
    };
    sdu_apex::source_sqp::Statistics old_stats{},new_stats{};
    ASSERT_TRUE(run(false,fd.get(),&old_stats));
    ASSERT_TRUE(run(true,forward.get(),&new_stats));
    EXPECT_EQ(old_stats.source_finite_difference_columns,40);
    EXPECT_EQ(new_stats.source_finite_difference_columns,0);
    EXPECT_EQ(new_stats.source_forward_sensitivity_intervals,50);
    EXPECT_EQ(new_stats.source_tangent_objective_evaluations,80);
    for(int j=0;j<40;++j) {
        EXPECT_NEAR(forward->gradient[j],fd->gradient[j],
            0.07*std::max(1.,std::abs(fd->gradient[j]))+0.15) << j;
        for(int row=0;row<60;++row)
            EXPECT_NEAR(forward->wall_jacobian[row][j],
                fd->wall_jacobian[row][j],0.012) << row << " " << j;
        for(int row=0;row<120;++row)
            EXPECT_NEAR(forward->rate_jacobian[row][j],
                fd->rate_jacobian[row][j],1.e-3) << row << " " << j;
    }
}

TEST(MpcSourceReducedSqp, FullSourceGaussNewtonHasCrossStageCouplingsAndIsPsd)
{
    Straight straight;
    auto config=configuration();
    MpcModelControl_t seed[PREDICTION_HORIZON]{};
    for(int k=0;k<PREDICTION_HORIZON;++k)
        seed[k].throttle_command=.1f;
    auto nominal=std::make_unique<MpcRtiNominal_t>();
    MpcRtiReference_t references[PREDICTION_HORIZON+1]{};
    ASSERT_TRUE(mpc_rti_build_nominal_same_epoch(
        &straight.initial,0.,nullptr,seed,PREDICTION_HORIZON,
        straight.track,straight.count,straight.lap,PREDICTION_DT_SECONDS,
        PREDICTION_HORIZON,&config.model,nominal.get(),references));
    auto schedule=std::make_unique<MpcRtiObjectiveSchedule_t>();
    ASSERT_TRUE(mpc_rti_build_objective_schedule(
        nominal.get(),references,&config.model,schedule.get()));
    config.objective_schedule=schedule.get();
    double objective=NAN;
    float minimum_slack=NAN;
    float margin[PREDICTION_HORIZON+1][2]{};
    ASSERT_TRUE(mpc_rti_score_candidate_plan_and_corridor_for_diagnostics(
        nominal.get(),straight.track,straight.count,straight.lap,
        &config,&objective,&minimum_slack,margin));
    auto output=std::make_unique<sdu_apex::source_sqp::FullSourceDerivatives>();
    ASSERT_TRUE(sdu_apex::source_sqp::full_source_derivatives(
        nominal.get(),straight.track,straight.count,straight.lap,
        PREDICTION_DT_SECONDS,&config,60,.002,objective,0.,output.get(),
        true));
    EXPECT_EQ(output->dual_plant_intervals,60);
    EXPECT_EQ(output->score_evaluations,120);
    Eigen::Matrix<double,60,60> H;
    for(int i=0;i<60;++i)for(int j=0;j<60;++j) {
        H(i,j)=output->gn_hessian[i][j];
        EXPECT_TRUE(std::isfinite(H(i,j))) << i << "," << j;
        EXPECT_NEAR(H(i,j),output->gn_hessian[j][i],1e-8);
    }
    EXPECT_GT(std::abs(H(1,3)),1e-4)
        << "consecutive throttle controls influence overlapping source states";
    EXPECT_GT(std::abs(H(0,2)),1e-5)
        << "steering commands influence common future heading residuals";
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double,60,60>>
        eigen(H,Eigen::EigenvaluesOnly);
    ASSERT_EQ(eigen.info(),Eigen::Success);
    EXPECT_GT(eigen.eigenvalues().minCoeff(),-1.e-6)
        << "GN Hessian must be PSD, independent of tire branch curvature";
}

TEST(MpcSourceReducedSqp, LocalSourceTransitionJacobianMatchesFullHorizonTangent)
{
    Straight straight;
    auto config=configuration();
    MpcModelControl_t seed[PREDICTION_HORIZON]{};
    for(int k=0;k<PREDICTION_HORIZON;++k)
        seed[k].throttle_command=.1f;
    auto nominal=std::make_unique<MpcRtiNominal_t>();
    MpcRtiReference_t refs[PREDICTION_HORIZON+1]{};
    ASSERT_TRUE(mpc_rti_build_nominal_same_epoch(
        &straight.initial,0.,nullptr,seed,PREDICTION_HORIZON,
        straight.track,straight.count,straight.lap,PREDICTION_DT_SECONDS,
        PREDICTION_HORIZON,&config.model,nominal.get(),refs));
    auto schedule=std::make_unique<MpcRtiObjectiveSchedule_t>();
    ASSERT_TRUE(mpc_rti_build_objective_schedule(
        nominal.get(),refs,&config.model,schedule.get()));
    config.objective_schedule=schedule.get();
    double objective=NAN;
    float min_slack=NAN;
    float margins[PREDICTION_HORIZON+1][2]{};
    ASSERT_TRUE(mpc_rti_score_candidate_plan_and_corridor_for_diagnostics(
        nominal.get(),straight.track,straight.count,straight.lap,
        &config,&objective,&min_slack,margins));
    auto direct=std::make_unique<sdu_apex::source_sqp::FullSourceDerivatives>();
    auto local=std::make_unique<sdu_apex::source_sqp::FullSourceDerivatives>();
    ASSERT_TRUE(sdu_apex::source_sqp::full_source_derivatives(
        nominal.get(),straight.track,straight.count,straight.lap,
        PREDICTION_DT_SECONDS,&config,60,.002,objective,0.,
        direct.get(),true,false));
    ASSERT_TRUE(sdu_apex::source_sqp::full_source_derivatives(
        nominal.get(),straight.track,straight.count,straight.lap,
        PREDICTION_DT_SECONDS,&config,60,.002,objective,0.,
        local.get(),true,true));
    ASSERT_EQ(local->dual_plant_intervals,60);
    EXPECT_EQ(local->score_evaluations,120);
    for(int j=0;j<60;++j) {
        EXPECT_NEAR(local->gradient[j],direct->gradient[j],
            0.003*std::max(1.,std::abs(direct->gradient[j]))+.01) << j;
        for(int row=0;row<60;++row) {
            EXPECT_NEAR(local->footprint[row][j],
                direct->footprint[row][j],5.e-4) << row << ":" << j;
            EXPECT_NEAR(local->gn_hessian[row][j],
                direct->gn_hessian[row][j],
                .003*std::max(1.,std::abs(direct->gn_hessian[row][j]))+
                .015) << row << ":" << j;
        }
        for(int row=0;row<120;++row)
            EXPECT_NEAR(local->actuator_rate[row][j],
                direct->actuator_rate[row][j],2.e-5) << row << ":" << j;
    }
}

TEST(MpcSourceReducedSqp, StageLocalFullSourceMatchesCausalDirectTangents)
{
    Straight straight;
    auto config=configuration();
    MpcModelControl_t seed[PREDICTION_HORIZON]{};
    for(int k=0;k<PREDICTION_HORIZON;++k) {
        seed[k].steering_target_rad=.02f*std::sin(.3f*k);
        seed[k].throttle_command=.1f;
    }
    auto nominal=std::make_unique<MpcRtiNominal_t>();
    MpcRtiReference_t refs[PREDICTION_HORIZON+1]{};
    ASSERT_TRUE(mpc_rti_build_nominal_same_epoch(
        &straight.initial,0.,nullptr,seed,PREDICTION_HORIZON,
        straight.track,straight.count,straight.lap,PREDICTION_DT_SECONDS,
        PREDICTION_HORIZON,&config.model,nominal.get(),refs));
    auto schedule=std::make_unique<MpcRtiObjectiveSchedule_t>();
    ASSERT_TRUE(mpc_rti_build_objective_schedule(
        nominal.get(),refs,&config.model,schedule.get()));
    config.objective_schedule=schedule.get();
    double objective=NAN;
    float minimum_slack=NAN;
    float margins[PREDICTION_HORIZON+1][2]{};
    ASSERT_TRUE(mpc_rti_score_candidate_plan_and_corridor_for_diagnostics(
        nominal.get(),straight.track,straight.count,straight.lap,
        &config,&objective,&minimum_slack,margins));
    auto direct=std::make_unique<sdu_apex::source_sqp::FullSourceDerivatives>();
    auto local=std::make_unique<sdu_apex::source_sqp::FullSourceDerivatives>();
    const auto call=[&](bool stage_local,
                        sdu_apex::source_sqp::FullSourceDerivatives *dest) {
        return sdu_apex::source_sqp::full_source_derivatives(
            nominal.get(),straight.track,straight.count,straight.lap,
            PREDICTION_DT_SECONDS,&config,60,.002,objective,0.,dest,
            true,stage_local);
    };
    ASSERT_TRUE(call(false,direct.get()));
    ASSERT_TRUE(call(true,local.get()));
    EXPECT_EQ(direct->dual_plant_intervals,60);
    EXPECT_EQ(local->dual_plant_intervals,60);
    EXPECT_EQ(local->score_evaluations,120);
    for(int col=0;col<60;++col) {
        EXPECT_NEAR(direct->gradient[col],local->gradient[col],
                    .001*std::max(1.,std::abs(direct->gradient[col]))) << col;
        for(int row=0;row<60;++row) {
            EXPECT_NEAR(direct->footprint[row][col],local->footprint[row][col],
                        2.e-4+5.e-5*std::abs(direct->footprint[row][col]))
                << row << "," << col;
            EXPECT_NEAR(direct->gn_hessian[row][col],
                        local->gn_hessian[row][col],
                        .002*std::max(1.,std::abs(direct->gn_hessian[row][col])))
                << row << "," << col;
        }
        for(int row=0;row<120;++row)
            EXPECT_NEAR(direct->actuator_rate[row][col],
                        local->actuator_rate[row][col],1.e-4)
                << row << "," << col;
    }
}

TEST(MpcSourceReducedSqp, ForwardSourceDeadlineDoesNotPublish)
{
    Straight straight;
    auto config=configuration();
    config.native_source_reduced_sqp=1;
    config.native_source_reduced_sqp_iterations=1;
    config.native_source_sqp_control_decision_stages=30;
    config.native_source_sqp_full_source_forward=1;
    config.absolute_deadline_monotonic_us=1.;
    auto memory=std::make_unique<MpcRtiMemory_t>();
    auto result=std::make_unique<MpcRtiCycleResult_t>();
    const auto status=mpc_rti_solve_cycle(
        &straight.initial,0.,straight.track,straight.count,straight.lap,
        PREDICTION_DT_SECONDS,PREDICTION_HORIZON,&config,memory.get(),
        result.get());
    EXPECT_EQ(status,MPC_RTI_CYCLE_WAITING_FOR_SOLVER_BUDGET);
    EXPECT_EQ(result->native_source_sqp_active,1);
    EXPECT_EQ(result->candidate_control_count,0);
    EXPECT_FALSE(memory->nominal.valid);
}

TEST(MpcSourceReducedSqp, PhysicallyImpossibleFutureCorridorNeverCertified)
{
    Straight straight;
    // A width smaller than the car envelope after the first ~0.4 metres
    // cannot be "solved" by a numerically feasible local QP. This test
    // verifies Phase I may improve its violation without accidentally
    // turning that intermediate into a publishable control.
    straight.track[1].s=straight.track[1].x=.4;
    straight.track[1].left_bound=straight.track[1].right_bound=.05;
    straight.track[2].left_bound=straight.track[2].right_bound=.05;
    ASSERT_TRUE(mpc_trajectory_prepare(straight.track,
        &straight.count,&straight.lap));
    auto config=configuration();
    MpcModelControl_t seed[PREDICTION_HORIZON]{};
    for(auto &u:seed)u.throttle_command=.1f;
    auto nominal=std::make_unique<MpcRtiNominal_t>();
    MpcRtiReference_t refs[PREDICTION_HORIZON+1]{};
    ASSERT_TRUE(mpc_rti_build_nominal_same_epoch(&straight.initial,0.,
        nullptr,seed,PREDICTION_HORIZON,straight.track,straight.count,
        straight.lap,PREDICTION_DT_SECONDS,PREDICTION_HORIZON,
        &config.model,nominal.get(),refs));
    MpcRtiObjectiveSchedule_t schedule{};
    ASSERT_TRUE(mpc_rti_build_objective_schedule(nominal.get(),refs,
        &config.model,&schedule));
    config.objective_schedule=&schedule;
    sdu_apex::source_sqp::Options options{};
    options.max_iterations=2;
    auto stats=sdu_apex::source_sqp::Statistics{};
    auto certified=std::make_unique<MpcRtiNominal_t>();
    MpcModelControl_t selected[PREDICTION_HORIZON]{};
    const bool numerically_evaluable=sdu_apex::source_sqp::optimize(
        &straight.initial,0.,straight.track,straight.count,
        straight.lap,PREDICTION_DT_SECONDS,&config,seed,options,
        &stats,selected,certified.get());
    // For a geometrically impossible track, rejection may be at the
    // model-evaluation gate rather than after a meaningful SQP Phase I.
    // Neither case is permission to certify a control. This must remain
    // conservative even if the invalid map geometry cannot be scored.
    EXPECT_FALSE(stats.feasible);
    EXPECT_FALSE(certified->valid);
    if(numerically_evaluable) {
        EXPECT_GT(stats.initial_wall_violation_m2,0.);
        EXPECT_LE(stats.final_wall_violation_m2,
            stats.initial_wall_violation_m2+1e-10);
    }
}

TEST(MpcSourceReducedSqp, ExpiredSourceDeadlineCannotPublishUnverifiedCandidate)
{
    Straight straight;
    auto config=configuration();
    config.native_source_reduced_sqp=1;
    config.native_source_reduced_sqp_iterations=2;
    // Already expired: deterministic refusal, not a heuristic timeout.
    config.absolute_deadline_monotonic_us=1.;
    auto memory=std::make_unique<MpcRtiMemory_t>();
    auto result=std::make_unique<MpcRtiCycleResult_t>();
    const auto status=mpc_rti_solve_cycle(
        &straight.initial,0.,straight.track,straight.count,straight.lap,
        PREDICTION_DT_SECONDS,PREDICTION_HORIZON,&config,memory.get(),
        result.get());
    EXPECT_EQ(status,MPC_RTI_CYCLE_WAITING_FOR_SOLVER_BUDGET);
    EXPECT_EQ(result->native_source_sqp_active,1);
    EXPECT_EQ(result->candidate_control_count,0);
    EXPECT_FALSE(memory->nominal.valid);
}

TEST(MpcSqpSuccessorRecourse, FutureNarrowingChangesNormalQpTerminalDirection)
{
    Straight straight;
    auto config=configuration();
    config.successor_recourse_weight=10000.f;
    MpcModelControl_t controls[PREDICTION_HORIZON]{};
    for(auto &u:controls)u.throttle_command=.12f;
    auto nominal=std::make_unique<MpcRtiNominal_t>();
    MpcRtiReference_t refs[PREDICTION_HORIZON+1]{};
    ASSERT_TRUE(mpc_rti_build_nominal_same_epoch(&straight.initial,0.,nullptr,
        controls,PREDICTION_HORIZON,straight.track,straight.count,
        straight.lap,PREDICTION_DT_SECONDS,PREDICTION_HORIZON,
        &config.model,nominal.get(),refs));
    const double end=nominal->progress[PREDICTION_HORIZON];
    MpcTrajectorySample_t narrowed[]={
        {0.,0.,0.,0.,0.,2.,4.,4.,0.},
        {end,end,0.,0.,0.,2.,4.,4.,0.},
        {end+.08,end+.08,0.,0.,0.,2.,.23,4.,0.},
        {end+2.,end+2.,0.,0.,0.,2.,.23,4.,0.},
        {20.,20.,0.,0.,0.,2.,4.,4.,0.}};
    size_t count=5;double lap=0.;
    ASSERT_TRUE(mpc_trajectory_prepare(narrowed,&count,&lap));
    ASSERT_TRUE(mpc_rti_build_nominal_same_epoch(&straight.initial,0.,nullptr,
        controls,PREDICTION_HORIZON,narrowed,count,lap,
        PREDICTION_DT_SECONDS,PREDICTION_HORIZON,&config.model,
        nominal.get(),refs));
    MpcRtiObjectiveSchedule_t frozen{};
    ASSERT_TRUE(mpc_rti_build_objective_schedule(nominal.get(),refs,
        &config.model,&frozen));
    ASSERT_TRUE(mpc_rti_initialize_successor_recourse(nominal.get(),
        narrowed,count,lap,PREDICTION_DT_SECONDS,&config,&frozen));
    double future_cost=NAN;
    ASSERT_TRUE(mpc_rti_successor_recourse_cost(nominal.get(),narrowed,
        count,lap,&config.model,&frozen,&future_cost));
    EXPECT_GT(future_cost,1.);
    auto qp=std::make_unique<MpcRtiProblem_t>();
    ASSERT_TRUE(mpc_rti_build_ltv_qp_with_objective_schedule(
        nominal->states,nominal->controls,refs,PREDICTION_HORIZON,
        PREDICTION_DT_SECONDS,&config.model,&frozen,qp.get()));
    const float ey_q=qp->terminal_q[MPC_OPT_IDX_EY];
    const float ey_Q=qp->terminal_Q[MPC_OPT_IDX_EY];
    ASSERT_TRUE(mpc_rti_add_successor_recourse_qp(qp.get(),nominal.get(),
        narrowed,count,lap,&config.model,&frozen));
    const float x=nominal->states[PREDICTION_HORIZON].plant.e_y;
    const double gradient=(qp->terminal_Q[MPC_OPT_IDX_EY]-ey_Q)*x+
        qp->terminal_q[MPC_OPT_IDX_EY]-ey_q;
    EXPECT_GT(gradient,0.) << "tightening LEFT wall should steer right";
    EXPECT_GT(qp->terminal_Q[MPC_OPT_IDX_EY],ey_Q);
}

TEST(MpcSqpSuccessorRecourse, FrozenPolicyHasMatchingExactSourceQpGradient)
{
    Straight track;
    auto config=configuration();
    config.successor_recourse_weight=12000.f;
    track.initial.plant.e_y=.14f;
    track.initial.plant.e_psi=.04f;
    for(auto &sample:track.track) {
        sample.left_bound=.55f;
        sample.right_bound=.55f;
    }
    MpcModelControl_t controls[PREDICTION_HORIZON]{};
    for(int k=0;k<PREDICTION_HORIZON;++k) {
        controls[k].steering_target_rad=.014f;
        controls[k].throttle_command=.15f;
    }
    auto nominal=std::make_unique<MpcRtiNominal_t>();
    MpcRtiReference_t references[PREDICTION_HORIZON+1]{};
    ASSERT_TRUE(mpc_rti_build_nominal_same_epoch(&track.initial,0.,nullptr,
        controls,PREDICTION_HORIZON,track.track,track.count,track.lap,
        PREDICTION_DT_SECONDS,PREDICTION_HORIZON,&config.model,
        nominal.get(),references));
    MpcRtiObjectiveSchedule_t frozen{};
    ASSERT_TRUE(mpc_rti_build_objective_schedule(nominal.get(),references,
        &config.model,&frozen));
    ASSERT_TRUE(mpc_rti_initialize_successor_recourse(nominal.get(),
        track.track,track.count,track.lap,PREDICTION_DT_SECONDS,
        &config,&frozen));
    frozen.successor_recourse_reserve_m=.3f;
    ASSERT_TRUE(frozen.successor_recourse_valid);
    double original_cost=NAN;
    ASSERT_TRUE(mpc_rti_successor_recourse_cost(nominal.get(),track.track,
        track.count,track.lap,&config.model,&frozen,&original_cost));
    EXPECT_GT(original_cost,0.);
    auto qp=std::make_unique<MpcRtiProblem_t>();
    ASSERT_TRUE(mpc_rti_build_ltv_qp_with_objective_schedule(
        nominal->states,nominal->controls,references,PREDICTION_HORIZON,
        PREDICTION_DT_SECONDS,&config.model,&frozen,qp.get()));
    float baseline_Q[MPC_RTI_NX]{},baseline_q[MPC_RTI_NX]{};
    std::copy_n(qp->terminal_Q,MPC_RTI_NX,baseline_Q);
    std::copy_n(qp->terminal_q,MPC_RTI_NX,baseline_q);
    ASSERT_TRUE(mpc_rti_add_successor_recourse_qp(qp.get(),nominal.get(),
        track.track,track.count,track.lap,&config.model,&frozen));
    float x[MPC_RTI_NX]{};
    serialize_state(&nominal->states[PREDICTION_HORIZON],x);
    x[MPC_OPT_IDX_PREVIOUS_STEERING_COMMAND]=
        nominal->controls[PREDICTION_HORIZON-1].steering_target_rad;
    x[MPC_OPT_IDX_PREVIOUS_THROTTLE_COMMAND]=
        nominal->controls[PREDICTION_HORIZON-1].throttle_command;
    int checked=0;
    for(int index:{0,1,2,3,4,5,20,21,22}) {
        auto plus=std::make_unique<MpcRtiNominal_t>(*nominal);
        auto minus=std::make_unique<MpcRtiNominal_t>(*nominal);
        constexpr float h=1.e-3f;
        const auto shift=[&](MpcRtiNominal_t &plan,float delta) {
            auto &state=plan.states[PREDICTION_HORIZON].plant;
            switch(index) {
            case 0:state.e_y+=delta;break;
            case 1:state.e_psi+=delta;break;
            case 2:state.u+=delta;break;
            case 3:state.v+=delta;break;
            case 4:state.r+=delta;break;
            case 5:state.actual_steering_angle+=delta;break;
            case 20:plan.progress[PREDICTION_HORIZON]+=delta;break;
            case 21:plan.controls[PREDICTION_HORIZON-1].steering_target_rad+=delta;break;
            case 22:plan.controls[PREDICTION_HORIZON-1].throttle_command+=delta;break;
            }
        };
        shift(*plus,h);shift(*minus,-h);
        double cp=NAN,cm=NAN;
        ASSERT_TRUE(mpc_rti_successor_recourse_cost(plus.get(),track.track,
            track.count,track.lap,&config.model,&frozen,&cp))<<index;
        ASSERT_TRUE(mpc_rti_successor_recourse_cost(minus.get(),track.track,
            track.count,track.lap,&config.model,&frozen,&cm))<<index;
        const double numerical=(cp-cm)/(2*h);
        const double analytical=(qp->terminal_Q[index]-baseline_Q[index])*
            x[index]+qp->terminal_q[index]-baseline_q[index];
        EXPECT_NEAR(analytical,numerical,
            std::max(3.,.17*std::abs(numerical)))<<index;
        ++checked;
    }
    EXPECT_EQ(checked,9);
}

TEST(MpcSqpSuccessor, ControlledEightStageSourceContinuationIsStrictlyCertified)
{
    Straight track;
    auto config=configuration();
    const MpcModelControl_t prior{.05f,.20f};
    auto found=std::make_unique<MpcRtiSuccessorProbe_t>();
    ASSERT_TRUE(mpc_rti_source_successor_probe(&track.initial,0.,prior,
        track.track,track.count,track.lap,PREDICTION_DT_SECONDS,
        &config,found.get()));
    EXPECT_EQ(found->candidates_evaluated,9);
    EXPECT_GT(found->feasible_candidates,0);
    ASSERT_TRUE(found->feasible);
    EXPECT_GE(found->minimum_clearance_m,0.f);
    EXPECT_GT(found->progress_m,.02);
    auto exact=std::make_unique<MpcRtiNominal_t>();
    MpcModelControl_t controls[PREDICTION_HORIZON]{};
    std::copy_n(found->controls,found->horizon,controls);
    int failure=-1;
    EXPECT_EQ(mpc_rti_validate_candidate_plan(&track.initial,0.,controls,
        track.track,track.count,track.lap,found->horizon,
        PREDICTION_DT_SECONDS,&config,exact.get(),&failure),
        MPC_RTI_ROLLOUT_OK);
    EXPECT_NEAR(exact->validation_minimum_corridor_slack_m,
                found->minimum_clearance_m,1.e-7);
    EXPECT_NEAR(exact->progress[found->horizon],
                found->progress_m,1.e-7);
    // These queued commands are a causal part of the full physical state.
    // An uninitialized or replaced queue must not be silently substituted
    // when constructing the exact source successor.
    EXPECT_FLOAT_EQ(track.initial.plant.delayed_throttle_command_2,.1f);
    EXPECT_EQ(exact->horizon,8);
}

TEST(MpcSqpSuccessor, SixteenStepContinuationMatchesStrictSourceDynamics)
{
    Straight track;
    auto cfg=configuration();
    const MpcModelControl_t previous{.05f,.20f};
    auto result=std::make_unique<MpcRtiSuccessorProbe_t>();
    ASSERT_TRUE(mpc_rti_source_successor_probe(&track.initial,0.,previous,
        track.track,track.count,track.lap,PREDICTION_DT_SECONDS,
        &cfg,result.get(),16));
    EXPECT_EQ(result->horizon,16);
    EXPECT_EQ(result->candidates_evaluated,9);
    ASSERT_TRUE(result->feasible);
    MpcModelControl_t controls[PREDICTION_HORIZON]{};
    std::copy_n(result->controls,result->horizon,controls);
    auto source=std::make_unique<MpcRtiNominal_t>();
    int failure=-1;
    ASSERT_EQ(mpc_rti_validate_candidate_plan(&track.initial,0.,controls,
        track.track,track.count,track.lap,16,PREDICTION_DT_SECONDS,
        &cfg,source.get(),&failure),MPC_RTI_ROLLOUT_OK);
    EXPECT_NEAR(source->validation_minimum_corridor_slack_m,
        result->minimum_clearance_m,1.e-7);
    EXPECT_NEAR(source->progress[16],result->progress_m,1.e-7);
    EXPECT_EQ(failure,-1);
}

TEST(MpcSqpSuccessor, DoesNotClaimControlInvariantFromInvalidFutureState)
{
    Straight track;
    auto config=configuration();
    // Vehicle is outside the original 4m corridor. A short safe-looking
    // steering target is never a valid 200ms source successor here.
    track.initial.plant.e_y=4.5f;
    MpcRtiSuccessorProbe_t future{};
    const MpcModelControl_t prior{0.f,.2f};
    ASSERT_TRUE(mpc_rti_source_successor_probe(&track.initial,0.,prior,
        track.track,track.count,track.lap,PREDICTION_DT_SECONDS,
        &config,&future));
    EXPECT_EQ(future.candidates_evaluated,9);
    EXPECT_EQ(future.feasible_candidates,0);
    EXPECT_FALSE(future.feasible);
}

TEST(MpcSqpSourcePhaseOne, ProgressGradientMatchesExactChangingWidth)
{
    Straight track;
    auto cfg=configuration();
    // Left wall contracts with progress, whereas the right wall is wide.
    // A locally fixed source pose still becomes less feasible as s grows.
    for(auto &point:track.track) {
        point.left_bound=.22f-.005f*point.s;
        point.right_bound=4.f;
    }
    const MpcModelState_t state=[]{
        MpcModelState_t result{};
        result.e_y=.15f;
        result.e_psi=.08f;
        return result;
    }();
    constexpr int stage=8;
    const double station=5.;
    const auto strict_merit=[&](double s) {
        MpcRtiReference_t ref{};
        EXPECT_TRUE(mpc_rti_reference_at_progress(track.track,track.count,
            track.lap,s,&cfg.model,&ref));
        float left=0.f,right=0.f;
        EXPECT_TRUE(mpc_rti_compute_corridor_slack(&state,
            ref.left_bound,ref.right_bound,&cfg.model,stage,&left,&right));
        return std::pow(std::min(0.,double(left)),2.)+
               std::pow(std::min(0.,double(right)),2.);
    };
    const double eps=.0005;
    const double numerical=(strict_merit(station+eps)-
                            strict_merit(station-eps))/(2.*eps);
    const double analytical=mpc_rti_source_corridor_progress_derivative(
        state,station,stage,track.track,track.count,track.lap,&cfg.model);
    ASSERT_TRUE(std::isfinite(analytical));
    ASSERT_GT(std::abs(analytical),1.e-4);
    EXPECT_NEAR(analytical,numerical,1.e-5);
}

TEST(MpcSqpSourcePhaseOne, RepairsTinyLateHorizonViolationWithStrictPlant)
{
    Straight s;
    auto cfg=configuration();
    cfg.model.command_increment_qp_enabled=1;
    cfg.model.max_steering_command_step_rad=2.f*SOURCE_MAX_STEERING_RAD;
    cfg.model.max_throttle_command_step=1.f;
    MpcModelControl_t seed[PREDICTION_HORIZON]{};
    for (auto &command:seed) {
        command.steering_target_rad=.12f;
        command.throttle_command=.15f;
    }
    auto wide=std::make_unique<MpcRtiNominal_t>();
    MpcRtiReference_t references[PREDICTION_HORIZON+1]{};
    ASSERT_TRUE(mpc_rti_build_nominal_same_epoch(&s.initial,0.,nullptr,
        seed,PREDICTION_HORIZON,s.track,s.count,s.lap,
        PREDICTION_DT_SECONDS,PREDICTION_HORIZON,&cfg.model,
        wide.get(),references));
    float required_halfwidth=0.f;
    for(int k=1;k<=PREDICTION_HORIZON;++k) {
        float left=0.f,right=0.f;
        ASSERT_TRUE(mpc_rti_compute_corridor_slack(&wide->states[k].plant,
            100.f,100.f,&cfg.model,k,&left,&right));
        required_halfwidth=std::max(required_halfwidth,
            100.f-std::min(left,right));
    }
    // The original 30-stage source nominal violates by 3mm somewhere in
    // the horizon; physical command history and first-step geometry are safe.
    const float bound=required_halfwidth-.003f;
    ASSERT_GT(bound,.1f);
    for(auto &point:s.track) {
        point.left_bound=bound;
        point.right_bound=bound;
    }
    auto baseline=std::make_unique<MpcRtiNominal_t>();
    ASSERT_TRUE(mpc_rti_build_nominal_same_epoch(&s.initial,0.,nullptr,
        seed,PREDICTION_HORIZON,s.track,s.count,s.lap,
        PREDICTION_DT_SECONDS,PREDICTION_HORIZON,&cfg.model,
        baseline.get(),references));
    MpcRtiObjectiveSchedule_t objective{};
    ASSERT_TRUE(mpc_rti_build_objective_schedule(
        baseline.get(),references,&cfg.model,&objective));
    cfg.objective_schedule=&objective;
    int failure=-1;
    EXPECT_EQ(mpc_rti_validate_candidate_plan(
        &s.initial,0.,seed,s.track,s.count,s.lap,
        PREDICTION_HORIZON,PREDICTION_DT_SECONDS,&cfg,
        baseline.get(),&failure),MPC_RTI_ROLLOUT_CORRIDOR);
    EXPECT_GT(failure,0);
    MpcRtiSourceNonlinearOptions_t options{};
    options.max_iterations=1;
    options.max_exact_trials_per_iteration=1;
    options.initial_step_scale=1.f;
    options.minimum_step_scale=.001f;
    options.armijo_fraction=.0001f;
    options.steering_direction_norm_rad=.2f;
    options.throttle_direction_norm=.35f;
    auto repaired=std::make_unique<MpcRtiSourceNonlinearResult_t>();
    auto cached=std::make_unique<MpcRtiNominal_t>();
    ASSERT_TRUE(mpc_rti_source_repair_infeasible_nominal(
        &s.initial,0.,s.track,s.count,s.lap,PREDICTION_DT_SECONDS,
        &cfg,&objective,seed,&options,repaired.get(),cached.get()));
    EXPECT_TRUE(cached->valid);
    EXPECT_EQ(cached->horizon,PREDICTION_HORIZON);
    EXPECT_EQ(repaired->feasibility_iterations_attempted,1);
    EXPECT_EQ(repaired->feasible_seed_recovered,1);
    EXPECT_GT(repaired->minimum_corridor_clearance_m,0.);
    auto certified=std::make_unique<MpcRtiNominal_t>();
    EXPECT_EQ(mpc_rti_validate_candidate_plan(
        &s.initial,0.,repaired->controls,s.track,s.count,s.lap,
        PREDICTION_HORIZON,PREDICTION_DT_SECONDS,&cfg,
        certified.get(),&failure),MPC_RTI_ROLLOUT_OK);
    for (int k=0;k<=PREDICTION_HORIZON;++k) {
        EXPECT_NEAR(cached->progress[k],certified->progress[k],1.e-7)<<k;
        EXPECT_NEAR(cached->states[k].plant.e_y,
                    certified->states[k].plant.e_y,1.e-7)<<k;
        EXPECT_NEAR(cached->states[k].plant.r,
                    certified->states[k].plant.r,1.e-7)<<k;
    }
}

TEST(MpcSqpClearanceMerit, ProgressAwareReserveDerivativeMatchesExactMargin)
{
    Straight track;
    auto cfg=configuration();
    cfg.model.weight_source_corridor_reserve=20000.f;
    cfg.model.near_wall_tracking_reserve_m=.04f;
    MpcModelControl_t controls[PREDICTION_HORIZON]{};
    for(auto &u:controls)u.throttle_command=.12f;
    auto nominal=std::make_unique<MpcRtiNominal_t>();
    MpcRtiReference_t refs[PREDICTION_HORIZON+1]{};
    ASSERT_TRUE(mpc_rti_build_nominal_same_epoch(&track.initial,0.,nullptr,
        controls,PREDICTION_HORIZON,track.track,track.count,track.lap,
        PREDICTION_DT_SECONDS,PREDICTION_HORIZON,&cfg.model,
        nominal.get(),refs));
    float halfwidth=0.f;
    for(int k=1;k<=PREDICTION_HORIZON;++k) {
        float left=0.f,right=0.f;
        ASSERT_TRUE(mpc_rti_compute_corridor_slack(&nominal->states[k].plant,
            100.f,100.f,&cfg.model,k,&left,&right));
        halfwidth=std::max(halfwidth,100.f-std::min(left,right));
    }
    for(auto &point:track.track) {
        point.left_bound=halfwidth+.025f-.001f*point.s;
        point.right_bound=halfwidth+.025f;
    }
    ASSERT_TRUE(mpc_rti_build_nominal_same_epoch(&track.initial,0.,nullptr,
        controls,PREDICTION_HORIZON,track.track,track.count,track.lap,
        PREDICTION_DT_SECONDS,PREDICTION_HORIZON,&cfg.model,
        nominal.get(),refs));
    MpcRtiObjectiveSchedule_t objective{};
    ASSERT_TRUE(mpc_rti_build_objective_schedule(nominal.get(),refs,
        &cfg.model,&objective));
    auto qp=std::make_unique<MpcRtiProblem_t>();
    ASSERT_TRUE(mpc_rti_build_ltv_qp_with_objective_schedule(
        nominal->states,nominal->controls,refs,PREDICTION_HORIZON,
        PREDICTION_DT_SECONDS,&cfg.model,&objective,qp.get()));
    float before[PREDICTION_HORIZON+1]{};
    for(int k=1;k<PREDICTION_HORIZON;++k)
       before[k]=qp->steps[k].q[MPC_OPT_IDX_PROGRESS_DEVIATION];
    before[PREDICTION_HORIZON]=qp->terminal_q[MPC_OPT_IDX_PROGRESS_DEVIATION];
    ASSERT_TRUE(mpc_rti_add_progress_aware_reserve_qp(qp.get(),nominal.get(),
        track.track,track.count,track.lap,&cfg.model));
    int validated=0;
    for(int k:{1,4,8,12,18,24,30}) {
        const auto &plant=nominal->states[k].plant;
        const double station=nominal->progress[k];
        auto objective_at=[&](double progress) {
            MpcRtiReference_t ref{};
            EXPECT_TRUE(mpc_rti_reference_at_progress(track.track,
                track.count,track.lap,progress,&cfg.model,&ref));
            float left=0.f,right=0.f;
            EXPECT_TRUE(mpc_rti_compute_corridor_slack(&plant,ref.left_bound,
                ref.right_bound,&cfg.model,k,&left,&right));
            const double reserve=cfg.model.near_wall_tracking_reserve_m;
            const double l=std::max(0.,reserve-double(left));
            const double r=std::max(0.,reserve-double(right));
            return double(cfg.model.weight_source_corridor_reserve)*(l*l+r*r);
        };
        const double eps=.002;
        const double numerical=(objective_at(station+eps)-objective_at(station-eps))/(2*eps);
        const double q=(k==PREDICTION_HORIZON
            ? qp->terminal_q[MPC_OPT_IDX_PROGRESS_DEVIATION]
            : qp->steps[k].q[MPC_OPT_IDX_PROGRESS_DEVIATION])-before[k];
        if(std::abs(numerical)<.05)continue;
        EXPECT_NEAR(q,numerical,std::max(.2,.03*std::abs(numerical)))<<k;
        ++validated;
    }
    EXPECT_GE(validated,3);
}

TEST(MpcSqpClearanceMerit, SourceAndQpShareFirstDerivative)
{
    Straight s;
    auto config0=configuration();
    auto config1=config0;
    s.initial.plant.e_y=.13f;
    s.initial.plant.e_psi=.055f;
    config1.model.near_wall_tracking_reserve_m=.04f;
    config0.model.near_wall_tracking_reserve_m=.04f;
    config1.model.weight_source_corridor_reserve=20000.f;
    config0.model.weight_source_corridor_reserve=0.f;
    MpcModelControl_t controls[PREDICTION_HORIZON]{};
    for(int k=0;k<PREDICTION_HORIZON;++k) {
        controls[k].steering_target_rad=.012f+.00018f*k;
        controls[k].throttle_command=.12f;
    }
    auto nominal=std::make_unique<MpcRtiNominal_t>();
    MpcRtiReference_t refs[PREDICTION_HORIZON+1]{};
    ASSERT_TRUE(mpc_rti_build_nominal_same_epoch(&s.initial,0.,nullptr,
        controls,PREDICTION_HORIZON,s.track,s.count,s.lap,
        PREDICTION_DT_SECONDS,PREDICTION_HORIZON,&config0.model,
        nominal.get(),refs));
    float required_halfwidth=0.f;
    for(int k=1;k<=PREDICTION_HORIZON;++k) {
        float left=0.f,right=0.f;
        ASSERT_TRUE(mpc_rti_compute_corridor_slack(&nominal->states[k].plant,
            100.f,100.f,&config0.model,k,&left,&right));
        required_halfwidth=std::max(required_halfwidth,
            100.f-std::min(left,right));
    }
    for(auto &point:s.track) {
        point.left_bound=required_halfwidth+.025f;
        point.right_bound=required_halfwidth+.025f;
    }
    ASSERT_TRUE(mpc_rti_build_nominal_same_epoch(&s.initial,0.,nullptr,
        controls,PREDICTION_HORIZON,s.track,s.count,s.lap,
        PREDICTION_DT_SECONDS,PREDICTION_HORIZON,&config1.model,
        nominal.get(),refs));
    MpcRtiObjectiveSchedule_t frozen{};
    ASSERT_TRUE(mpc_rti_build_objective_schedule(nominal.get(),refs,
        &config1.model,&frozen));
    config0.objective_schedule=&frozen;
    config1.objective_schedule=&frozen;
    auto qp0=std::make_unique<MpcRtiProblem_t>();
    auto qp1=std::make_unique<MpcRtiProblem_t>();
    ASSERT_TRUE(mpc_rti_build_ltv_qp_with_objective_schedule(
        nominal->states,nominal->controls,refs,PREDICTION_HORIZON,
        PREDICTION_DT_SECONDS,&config0.model,&frozen,qp0.get()));
    ASSERT_TRUE(mpc_rti_build_ltv_qp_with_objective_schedule(
        nominal->states,nominal->controls,refs,PREDICTION_HORIZON,
        PREDICTION_DT_SECONDS,&config1.model,&frozen,qp1.get()));
    double gradient0[PREDICTION_HORIZON][MPC_RTI_NU]{};
    double gradient1[PREDICTION_HORIZON][MPC_RTI_NU]{};
    ASSERT_TRUE(mpc_rti_qp_source_gradient(qp0.get(),controls,
        config0.model.command_increment_qp_enabled,gradient0));
    ASSERT_TRUE(mpc_rti_qp_source_gradient(qp1.get(),controls,
        config1.model.command_increment_qp_enabled,gradient1));
    int comparisons=0;
    for(int k:{0,3,7,11}) {
        const double analytical=gradient1[k][0]-gradient0[k][0];
        if(std::abs(analytical)<.01)continue;
        MpcModelControl_t plus[PREDICTION_HORIZON]{};
        MpcModelControl_t minus[PREDICTION_HORIZON]{};
        std::copy_n(controls,PREDICTION_HORIZON,plus);
        std::copy_n(controls,PREDICTION_HORIZON,minus);
        constexpr float h=1.e-3f;
        plus[k].steering_target_rad+=h;
        minus[k].steering_target_rad-=h;
        const auto incremental_cost=[&](const MpcModelControl_t *seq) {
            auto accepted0=std::make_unique<MpcRtiNominal_t>();
            auto accepted1=std::make_unique<MpcRtiNominal_t>();
            int failure=-1;
            EXPECT_EQ(mpc_rti_validate_candidate_plan(&s.initial,0.,seq,
                s.track,s.count,s.lap,PREDICTION_HORIZON,
                PREDICTION_DT_SECONDS,&config0,accepted0.get(),&failure),
                MPC_RTI_ROLLOUT_OK);
            EXPECT_EQ(mpc_rti_validate_candidate_plan(&s.initial,0.,seq,
                s.track,s.count,s.lap,PREDICTION_HORIZON,
                PREDICTION_DT_SECONDS,&config1,accepted1.get(),&failure),
                MPC_RTI_ROLLOUT_OK);
            return accepted1->validation_objective-
                accepted0->validation_objective;
        };
        const double numeric=(incremental_cost(plus)-
            incremental_cost(minus))/(2*h);
        EXPECT_NEAR(analytical,numeric,
            std::max(4.,.28*std::abs(numeric)))<<k;
        ++comparisons;
    }
    EXPECT_GE(comparisons,1);
}

TEST(MpcSqpClearanceAdjoint, MatchesExactSourceFootprintMarginOnStraight)
{
    Straight s;
    auto cfg=configuration();
    s.initial.plant.e_y=.17f;
    s.initial.plant.e_psi=.065f;
    MpcModelControl_t controls[PREDICTION_HORIZON]{};
    for (int k=0;k<PREDICTION_HORIZON;++k) {
        controls[k].steering_target_rad=.02f+.0002f*k;
        controls[k].throttle_command=.12f;
    }
    auto nominal=std::make_unique<MpcRtiNominal_t>();
    MpcRtiReference_t refs[PREDICTION_HORIZON+1]{};
    ASSERT_TRUE(mpc_rti_build_nominal_same_epoch(&s.initial,0.,nullptr,
        controls,PREDICTION_HORIZON,s.track,s.count,s.lap,
        PREDICTION_DT_SECONDS,PREDICTION_HORIZON,&cfg.model,
        nominal.get(),refs));
    MpcRtiObjectiveSchedule_t objective{};
    ASSERT_TRUE(mpc_rti_build_objective_schedule(nominal.get(),refs,
        &cfg.model,&objective));
    cfg.objective_schedule=&objective;
    auto problem=std::make_unique<MpcRtiProblem_t>();
    ASSERT_TRUE(mpc_rti_build_ltv_qp_with_objective_schedule(
        nominal->states,nominal->controls,refs,PREDICTION_HORIZON,
        PREDICTION_DT_SECONDS,&cfg.model,&objective,problem.get()));
    double gradient[PREDICTION_HORIZON][MPC_RTI_NU]{};
    float nominal_slack=0.f;
    int limiting_stage=-1;
    ASSERT_TRUE(mpc_rti_qp_clearance_gradient(problem.get(),nominal.get(),
        cfg.model.command_increment_qp_enabled,gradient,&nominal_slack,
        &limiting_stage));
    EXPECT_GE(limiting_stage,1);
    auto checked=std::make_unique<MpcRtiNominal_t>();
    int failure=-1;
    ASSERT_EQ(mpc_rti_validate_candidate_plan(&s.initial,0.,controls,
        s.track,s.count,s.lap,PREDICTION_HORIZON,
        PREDICTION_DT_SECONDS,&cfg,checked.get(),&failure),
        MPC_RTI_ROLLOUT_OK);
    EXPECT_NEAR(nominal_slack,
        checked->validation_minimum_corridor_slack_m,2.e-4f);
    int derivative_checks=0;
    for (int k:{0,2,5,9}) {
        if(std::abs(gradient[k][0])<1.e-4)continue;
        MpcModelControl_t plus[PREDICTION_HORIZON]{};
        MpcModelControl_t minus[PREDICTION_HORIZON]{};
        std::copy_n(controls,PREDICTION_HORIZON,plus);
        std::copy_n(controls,PREDICTION_HORIZON,minus);
        constexpr float h=1.e-3f;
        plus[k].steering_target_rad+=h;
        minus[k].steering_target_rad-=h;
        auto higher=std::make_unique<MpcRtiNominal_t>();
        auto lower=std::make_unique<MpcRtiNominal_t>();
        ASSERT_EQ(mpc_rti_validate_candidate_plan(&s.initial,0.,plus,
            s.track,s.count,s.lap,PREDICTION_HORIZON,
            PREDICTION_DT_SECONDS,&cfg,higher.get(),&failure),
            MPC_RTI_ROLLOUT_OK);
        ASSERT_EQ(mpc_rti_validate_candidate_plan(&s.initial,0.,minus,
            s.track,s.count,s.lap,PREDICTION_HORIZON,
            PREDICTION_DT_SECONDS,&cfg,lower.get(),&failure),
            MPC_RTI_ROLLOUT_OK);
        const double numerical=(higher->validation_minimum_corridor_slack_m-
            lower->validation_minimum_corridor_slack_m)/(2*h);
        EXPECT_NEAR(gradient[k][0],numerical,
            std::max(.025,.22*std::abs(numerical)))<<k;
        ++derivative_checks;
    }
    EXPECT_GE(derivative_checks,1);
}

TEST(MpcSqpGlobalization, AgreementFilterDoesNotReintroduceScalarTrustRegion)
{
    Straight s;
    auto cfg=configuration();
    cfg.minimum_sqp_model_agreement_ratio=.2f;
    cfg.max_sqp_iterations=1;
    auto memory=std::make_unique<MpcRtiMemory_t>();
    auto result=std::make_unique<MpcRtiCycleResult_t>();
    auto captured=std::make_unique<MpcRtiQpCapture_t>();
    cfg.qp_capture=captured.get();
    // This field exists for legacy same-build fixture layout only; neither
    // its contents nor a positive optional agreement ratio can scale the QP.
    memory->sqp_trust_region_scale=.0625f;
    mpc_rti_solve_cycle(&s.initial,0.,s.track,s.count,s.lap,
        PREDICTION_DT_SECONDS,PREDICTION_HORIZON,
        &cfg,memory.get(),result.get());
    ASSERT_TRUE(captured->valid);
    ASSERT_TRUE(captured->proposal_controls_valid);
    for(int k=0;k<PREDICTION_HORIZON;++k) {
        EXPECT_FLOAT_EQ(captured->unscaled_proposal_controls[k].steering_target_rad,
                        captured->proposal_controls[k].steering_target_rad)<<k;
        EXPECT_FLOAT_EQ(captured->unscaled_proposal_controls[k].throttle_command,
                        captured->proposal_controls[k].throttle_command)<<k;
    }
    EXPECT_FLOAT_EQ(memory->sqp_trust_region_scale,.0625f);
}

TEST(MpcSqpConvergence, RelinearizationPreservesSameEpochCommandAndProgressSeed) {
    Straight s;auto c=configuration();
    MpcModelControl_t controls[PREDICTION_HORIZON]{};
    for (int k=0;k<PREDICTION_HORIZON;++k) {
        controls[k].steering_target_rad=.001f*k;
        controls[k].throttle_command=.10f+.001f*k;
    }
    auto previous=std::make_unique<MpcRtiNominal_t>();
    auto repeated=std::make_unique<MpcRtiNominal_t>();
    MpcRtiReference_t refs[PREDICTION_HORIZON+1]{};
    ASSERT_TRUE(mpc_rti_build_nominal_same_epoch(&s.initial,0.,nullptr,
        controls,PREDICTION_HORIZON,s.track,s.count,s.lap,PREDICTION_DT_SECONDS,
        PREDICTION_HORIZON,&c.model,previous.get(),refs));
    // No separately supplied controls: a same-epoch relinearization must
    // preserve the full nominal, including its unconsumed first and last input.
    ASSERT_TRUE(mpc_rti_build_nominal_same_epoch(&s.initial,0.,previous.get(),
        nullptr,0,s.track,s.count,s.lap,PREDICTION_DT_SECONDS,
        PREDICTION_HORIZON,&c.model,repeated.get(),refs));
    for (int k=0;k<PREDICTION_HORIZON;++k) {
        EXPECT_FLOAT_EQ(repeated->controls[k].steering_target_rad,
            previous->controls[k].steering_target_rad) << k;
        EXPECT_FLOAT_EQ(repeated->controls[k].throttle_command,
            previous->controls[k].throttle_command) << k;
        EXPECT_NEAR(repeated->progress[k],previous->progress[k],1.e-7) << k;
    }
    EXPECT_NEAR(repeated->progress[PREDICTION_HORIZON],
        previous->progress[PREDICTION_HORIZON],1.e-7);
}

TEST(MpcSqpConvergence, WarmControlsAdvanceByElapsedCommandEpochs) {
    Straight s;auto c=configuration();c.max_sqp_iterations=1;
    constexpr int64_t first=10'000'000'000;
    MpcModelControl_t controls[PREDICTION_HORIZON]{};
    for(int k=0;k<PREDICTION_HORIZON;++k)
        controls[k]={.001f*k,.1f+.001f*k};
    for(int stages:{0,1,3}) {
        auto memory=std::make_unique<MpcRtiMemory_t>();
        auto result=std::make_unique<MpcRtiCycleResult_t>();
        ASSERT_TRUE(mpc_rti_memory_set_warm_start_controls_at_epoch(
            memory.get(),controls,PREDICTION_HORIZON,first));
        c.command_epoch_ns=first+stages*25'000'000;
        mpc_rti_solve_cycle(&s.initial,0.,s.track,s.count,s.lap,
            PREDICTION_DT_SECONDS,PREDICTION_HORIZON,&c,memory.get(),result.get());
        ASSERT_EQ(result->nominal_control_count,PREDICTION_HORIZON);
        EXPECT_EQ(result->command_epoch_shift_stages,stages);
        EXPECT_FLOAT_EQ(result->nominal_controls[0].steering_target_rad,
            controls[stages].steering_target_rad);
        EXPECT_FLOAT_EQ(result->nominal_controls[0].throttle_command,
            controls[stages].throttle_command);
    }
    EXPECT_EQ(mpc_rti_command_epoch_shift(first,first-1,PREDICTION_DT_SECONDS),-1);
    EXPECT_EQ(mpc_rti_command_epoch_shift(first,first+750'000'000,PREDICTION_DT_SECONDS),-1);
    EXPECT_EQ(mpc_rti_command_epoch_shift(0,first,PREDICTION_DT_SECONDS),-1);
    EXPECT_EQ(mpc_rti_command_epoch_shift(first,first+24'100'000,PREDICTION_DT_SECONDS),1);
}

TEST(MpcSqpConvergence, QpFootprintAgreesWithExactSourceAtNominalAndExplicitRestoration)
{
    Straight track;
    auto cfg=configuration();
    cfg.model.nonlinear_corridor_tolerance_m=.005f;
    track.initial.plant.e_psi=.11f;
    MpcModelControl_t controls[PREDICTION_HORIZON]{};
    for(auto &u:controls)u.throttle_command=.12f;
    auto nominal=std::make_unique<MpcRtiNominal_t>();
    auto qp=std::make_unique<MpcRtiProblem_t>();
    MpcRtiReference_t refs[PREDICTION_HORIZON+1]{};
    ASSERT_TRUE(mpc_rti_build_nominal_same_epoch(&track.initial,0.,nullptr,
        controls,PREDICTION_HORIZON,track.track,track.count,track.lap,
        PREDICTION_DT_SECONDS,PREDICTION_HORIZON,&cfg.model,
        nominal.get(),refs));
    ASSERT_TRUE(mpc_rti_build_ltv_qp(nominal->states,nominal->controls,refs,
        PREDICTION_HORIZON,PREDICTION_DT_SECONDS,&cfg.model,qp.get()));
    const auto polygon_margin=[&](int k) {
        const auto &p=qp->state_polygons[k];
        const auto &plant=nominal->states[k].plant;
        double minimum=INFINITY;
        for(int i=0;i<p.count;++i) {
            // Plane gradients mix metres with heading radians; the hard QP
            // constrains THIS lateral residual, not norm(normal) distance.
            minimum=std::min(minimum,double(p.upper[i])-
                double(p.normal[i][0])*plant.e_y-
                double(p.normal[i][1])*plant.e_psi);
        }
        return minimum;
    };
    double strict_source[PREDICTION_HORIZON+1]{};
    for(int k:{1,5,10,15,25,30}) {
        float right=0.f,left=0.f;
        ASSERT_TRUE(mpc_rti_compute_corridor_slack(&nominal->states[k].plant,
            refs[k].left_bound,refs[k].right_bound,
            &cfg.model,k,&right,&left));
        strict_source[k]=std::min(right,left);
        EXPECT_NEAR(polygon_margin(k),
            strict_source[k]-cfg.model.nonlinear_corridor_tolerance_m,1.e-4)
            <<"stage="<<k;
    }
    constexpr float relaxation=.18f;
    ASSERT_TRUE(mpc_rti_prepare_feasibility_restoration_qp(qp.get(),
        nominal.get(),relaxation,8000.f));
    for(int k:{1,5,10,15,25,30})
        EXPECT_NEAR(polygon_margin(k),
            strict_source[k]-cfg.model.nonlinear_corridor_tolerance_m+
            relaxation,1.e-4)<<"stage="<<k;
}

TEST(MpcSqpConvergence, RestorationAcceptsPhysicalHeadingOutsideLegacyFunnel) {
    Straight s;auto c=configuration();s.initial.plant.e_psi=.4f;
    c.model.weight_e_y=c.model.weight_e_psi=0.f;
    c.enable_feasibility_restoration=c.prefer_feasibility_first=1;
    c.feasibility_restoration_qp_relaxation_m=.18f;
    c.feasibility_restoration_wall_weight=8000.f;
    c.max_sqp_iterations=1;
    auto memory=std::make_unique<MpcRtiMemory_t>();
    auto result=std::make_unique<MpcRtiCycleResult_t>();
    const auto status=mpc_rti_solve_cycle(&s.initial,0.,s.track,s.count,s.lap,
        PREDICTION_DT_SECONDS,PREDICTION_HORIZON,&c,memory.get(),result.get());
    ASSERT_TRUE(status==MPC_RTI_CYCLE_ACCEPTED_OPTIMAL ||
        status==MPC_RTI_CYCLE_ACCEPTED_FEASIBLE || status==MPC_RTI_CYCLE_ACCEPTED_LINE_SEARCH);
    ASSERT_TRUE(memory->nominal.valid);
    EXPECT_GT(memory->nominal.states[PREDICTION_HORIZON].plant.e_psi,.35f);
    EXPECT_GE(memory->nominal.validation_minimum_corridor_slack_m,0.f);
}

TEST(MpcSqpConvergence, QpDoesNotInventPhysicalWheelSpeedCeilings) {
    Straight s;auto c=configuration();
    MpcModelControl_t controls[PREDICTION_HORIZON]{};
    for(auto &control:controls)control.throttle_command=.1f;
    auto nominal=std::make_unique<MpcRtiNominal_t>();
    auto problem=std::make_unique<MpcRtiProblem_t>();
    MpcRtiReference_t refs[PREDICTION_HORIZON+1]{};
    ASSERT_TRUE(mpc_rti_build_nominal_same_epoch(&s.initial,0.,nullptr,controls,
        PREDICTION_HORIZON,s.track,s.count,s.lap,PREDICTION_DT_SECONDS,
        PREDICTION_HORIZON,&c.model,nominal.get(),refs));
    ASSERT_TRUE(mpc_rti_build_ltv_qp(nominal->states,nominal->controls,refs,
        PREDICTION_HORIZON,PREDICTION_DT_SECONDS,&c.model,problem.get()));
    // This source-valid high-spin initial state is not a motor command or
    // actuator saturation. Future QP state bounds must not invent a limit.
    auto spin=s.initial.plant;
    for(int wheel=0;wheel<4;++wheel)
        spin.source_plant_internal_state[MPC_PLANT_STATE_WHEEL_OMEGA_FL+wheel]=400.;
    ASSERT_TRUE(mpc_vehicle_model_step(&spin,&controls[0],PREDICTION_DT_SECONDS,0.f).valid);
    for(int k=1;k<PREDICTION_HORIZON;++k)for(int wheel=0;wheel<4;++wheel) {
        // +/-BIG_BOUND denotes infinity to Riccati, scaling and OSQP.
        EXPECT_GE(problem->steps[k].x_ub[MPC_OPT_IDX_WHEEL_OMEGA_FL+wheel],BIG_BOUND);
        EXPECT_LE(problem->steps[k].x_lb[MPC_OPT_IDX_WHEEL_OMEGA_FL+wheel],-BIG_BOUND);
    }
}

TEST(MpcTerminalViability, CostGradientMatchesIndependentExactContinuation) {
    Straight s;auto c=configuration();
    MpcModelControl_t controls[PREDICTION_HORIZON]{};
    for(auto &u:controls)u.throttle_command=.1f;
    auto nominal=std::make_unique<MpcRtiNominal_t>();
    MpcRtiReference_t refs[PREDICTION_HORIZON+1]{};
    ASSERT_TRUE(mpc_rti_build_nominal_same_epoch(&s.initial,0.,nullptr,
        controls,PREDICTION_HORIZON,s.track,s.count,s.lap,PREDICTION_DT_SECONDS,
        PREDICTION_HORIZON,&c.model,nominal.get(),refs));
    MpcRtiObjectiveSchedule_t frozen{};
    ASSERT_TRUE(mpc_rti_build_objective_schedule(nominal.get(),refs,&c.model,&frozen));
    ASSERT_TRUE(mpc_rti_initialize_terminal_targets(nominal.get(),s.track,s.count,
        s.lap,PREDICTION_DT_SECONDS,&c.model,&frozen));
    auto problem=std::make_unique<MpcRtiProblem_t>();
    ASSERT_TRUE(mpc_rti_add_terminal_viability(problem.get(),nominal.get(),
        s.track,s.count,s.lap,&c.model,&frozen));
    EXPECT_EQ(problem->terminal_constraints.count,10);
    for(int coordinate:{0,1,2,3,4,5,8,20,21,22}) {
        auto plus=*nominal,minus=*nominal;
        constexpr double h=1.e-4;
        auto shift=[&](MpcRtiNominal_t &p,double delta) {
            auto &v=p.states[PREDICTION_HORIZON].plant;
            switch(coordinate) {
            case 0:v.e_y+=delta;break;case 1:v.e_psi+=delta;break;
            case 2:v.u+=delta;break;case 3:v.v+=delta;break;
            case 4:v.r+=delta;break;case 5:v.actual_steering_angle+=delta;break;
            case 8:v.source_plant_internal_state[2]+=delta;break;
            case 20:p.progress[PREDICTION_HORIZON]+=delta;break;
            case 21:p.controls[PREDICTION_HORIZON-1].steering_target_rad+=delta;break;
            case 22:p.controls[PREDICTION_HORIZON-1].throttle_command+=delta;break;
            }
        };
        shift(plus,h);shift(minus,-h);
        double a=0.,b=0.;
        ASSERT_TRUE(mpc_rti_terminal_value(&plus,s.track,s.count,s.lap,&c.model,&frozen,&a,0));
        ASSERT_TRUE(mpc_rti_terminal_value(&minus,s.track,s.count,s.lap,&c.model,&frozen,&b,0));
        float x[MPC_RTI_NX]{};serialize_state(&nominal->states[PREDICTION_HORIZON],x);
        x[21]=nominal->controls[PREDICTION_HORIZON-1].steering_target_rad;
        x[22]=nominal->controls[PREDICTION_HORIZON-1].throttle_command;
        const double gradient=problem->terminal_Q[coordinate]*x[coordinate]+
            problem->terminal_q[coordinate];
        EXPECT_NEAR(gradient,(a-b)/(2.*h),std::max(.1,.02*std::abs(gradient)))<<coordinate;
    }
    EXPECT_NEAR(problem->terminal_Q[0]*nominal->states[PREDICTION_HORIZON].plant.e_y+
        problem->terminal_q[0],6.*c.model.weight_e_y*
        nominal->states[PREDICTION_HORIZON].plant.e_y,.5);
    // The 750ms path remains wide. A narrowing immediately beyond its
    // endpoint must enter both the terminal QP and exact continuation gate.
    const double end=nominal->progress[PREDICTION_HORIZON];
    MpcTrajectorySample_t narrow[]={
        {0.,0.,0.,0.,0.,2.,4.,4.,0.},
        {end,end,0.,0.,0.,2.,4.,4.,0.},
        {end+.02,end+.02,0.,0.,0.,2.,.2,.2,0.},
        {10.,10.,0.,0.,0.,2.,.2,.2,0.},
        {20.,20.,0.,0.,0.,2.,4.,4.,0.}};
    size_t narrow_count=5;double narrow_lap=0.;
    ASSERT_TRUE(mpc_trajectory_prepare(narrow,&narrow_count,&narrow_lap));
    auto narrow_problem=std::make_unique<MpcRtiProblem_t>();
    ASSERT_TRUE(mpc_rti_add_terminal_viability(narrow_problem.get(),nominal.get(),
        narrow,narrow_count,narrow_lap,&c.model,&frozen));
    float terminal_x[MPC_RTI_NX]{};
    serialize_state(&nominal->states[PREDICTION_HORIZON],terminal_x);
    terminal_x[21]=nominal->controls[PREDICTION_HORIZON-1].steering_target_rad;
    terminal_x[22]=nominal->controls[PREDICTION_HORIZON-1].throttle_command;
    double violation=-INFINITY;
    for(int f=0;f<6;++f) {
        double r=-narrow_problem->terminal_constraints.upper[f];
        for(int i=0;i<MPC_RTI_NX;++i)
            r+=narrow_problem->terminal_constraints.normal[f][i]*terminal_x[i];
        violation=std::max(violation,r);
    }
    EXPECT_GT(violation,0.);
    double narrow_cost=0.;
    EXPECT_FALSE(mpc_rti_terminal_value(nominal.get(),narrow,narrow_count,narrow_lap,
        &c.model,&frozen,&narrow_cost,1));
    EXPECT_GT(narrow_problem->terminal_Q[0],problem->terminal_Q[0]);
    double cost=0.;
    ASSERT_TRUE(mpc_rti_terminal_value(nominal.get(),s.track,s.count,s.lap,
        &c.model,&frozen,&cost,1));
    c.model.world_sweep_check=[](void*,double,double,double,double,double,double,
        double,float*slack)->int{*slack=-.01f;return 0;};
    EXPECT_FALSE(mpc_rti_terminal_value(nominal.get(),s.track,s.count,s.lap,
        &c.model,&frozen,&cost,1));
}

TEST(RiccatiTerminalPlanes, CoupledConstraintChangesOptimalDirectionWithScaling) {
    auto steps=std::make_unique<RiccatiStepData_t[]>(RICCATI_HORIZON);
    float Q[RICCATI_MAX_NX]{},q[RICCATI_MAX_NX]{},lo[RICCATI_MAX_NX],hi[RICCATI_MAX_NX],x0[RICCATI_MAX_NX]{};
    std::fill_n(lo,RICCATI_MAX_NX,-BIG_BOUND);std::fill_n(hi,RICCATI_MAX_NX,BIG_BOUND);
    for(int k=0;k<RICCATI_HORIZON;++k) {
        auto &step=steps[k];
        for(int i=0;i<RICCATI_MAX_NX;++i) {
            step.A[i][i]=1.f;step.x_lb[i]=-BIG_BOUND;step.x_ub[i]=BIG_BOUND;
        }
        step.B[0][0]=step.B[1][1]=1.f;
        for(int a=0;a<2;++a){step.R_diag[a]=2.f;step.u_lb[a]=-1.f;step.u_ub[a]=1.f;}
    }
    Q[0]=Q[1]=2.f;q[0]=q[1]=-2.f;
    for(int scaled:{0,1}) {
        RiccatiAdmmConfig_t c{};c.rho=c.rho_u=7.f;c.max_iterations=300;c.tolerance=1.e-4f;
        c.use_scaling=scaled;
        std::fill_n(c.state_scale,RICCATI_MAX_NX,1.f);
        std::fill_n(c.input_scale,RICCATI_MAX_NU,1.f);
        c.state_scale[0]=.2f;c.state_scale[1]=.5f;
        c.terminal_constraints.count=1;c.terminal_constraints.rho=7.f;
        c.terminal_constraints.normal[0][0]=1.f;c.terminal_constraints.normal[0][1]=1.f;
        c.terminal_constraints.upper[0]=.2f;
        RiccatiAdmmState_t state{};RiccatiSolution_t result{};
        ASSERT_EQ(riccati_admm_solve(steps.get(),Q,q,lo,hi,x0,&c,&state,&result),RICCATI_STATUS_OPTIMAL);
        EXPECT_NEAR(result.x[RICCATI_HORIZON][0],.1f,.001f);
        EXPECT_NEAR(result.x[RICCATI_HORIZON][1],.1f,.001f);
        EXPECT_LT(result.primal_residual,.001f);
    }
}

TEST(MpcTerminalViability, InfeasibleTailStillGetsNormalGlobalization) {
    EXPECT_TRUE(mpc_rti_line_search_should_backtrack(
        MPC_RTI_ROLLOUT_TERMINAL_VIABILITY,0,0,0));
    EXPECT_TRUE(mpc_rti_line_search_should_continue(
        MPC_RTI_ROLLOUT_TERMINAL_VIABILITY,0));
}

TEST(MpcSqpWarmStart, ProgressRebasePreservesPhysicalStationAcrossEpochsAndScaling) {
    for(int shift:{0,1,3})for(int scaled:{0,1}) {
        auto memory=std::make_unique<MpcRtiMemory_t>();
        memory->solver_state.initialized=1;
        memory->solver_state.scaling_enabled=scaled;
        std::fill_n(memory->solver_state.state_scale,MPC_RTI_NX,1.f);
        memory->solver_state.state_scale[MPC_OPT_IDX_PROGRESS_DEVIATION]=2.f;
        memory->solver_nominal_progress_valid=1;
        MpcRtiNominal_t nominal{};
        for(int k=0;k<=RICCATI_HORIZON;++k) {
            memory->solver_nominal_progress[k]=100.+.2*k;
            memory->solver_state.z_x[k][MPC_OPT_IDX_PROGRESS_DEVIATION]=.1f+.01f*k;
            nominal.progress[k]=100.+.2*k+.03;
        }
        for(int k=0;k<RICCATI_HORIZON;++k)memory->solver_state.z_u[k][0]=.01f*k;
        const auto old=*memory;
        RiccatiAdmmConfig_t config{};config.reuse_same_epoch=!shift;
        float x0[MPC_RTI_NX]{};
        mpc_rti_rebase_solver_warm_start(memory.get(),&nominal,x0,&config,shift);
        EXPECT_EQ(config.reuse_same_epoch,1);
        const double scale=scaled?2.:1.;
        for(int k=1;k<=RICCATI_HORIZON;++k) {
            int from=std::min(k+shift,RICCATI_HORIZON);
            const double before=old.solver_nominal_progress[from]+scale*
                old.solver_state.z_x[from][MPC_OPT_IDX_PROGRESS_DEVIATION];
            const double after=nominal.progress[k]+scale*
                memory->solver_state.z_x[k][MPC_OPT_IDX_PROGRESS_DEVIATION];
            EXPECT_NEAR(before,after,1.e-6);
        }
        EXPECT_FLOAT_EQ(memory->solver_state.z_u[0][0],old.solver_state.z_u[shift][0]);
        EXPECT_DOUBLE_EQ(memory->solver_nominal_progress[5],nominal.progress[5]);
    }
}

TEST(RiccatiWarmStartPenalty, ShiftPrecedesChannelPenaltyNormalization) {
    auto steps=std::make_unique<RiccatiStepData_t[]>(RICCATI_HORIZON);
    for(int k=0;k<RICCATI_HORIZON;++k) {
        for(int i=0;i<MPC_RTI_NX;++i) {
            steps[k].A[i][i]=1.f;steps[k].x_lb[i]=-BIG_BOUND;steps[k].x_ub[i]=BIG_BOUND;
        }
        for(int a=0;a<2;++a){steps[k].u_lb[a]=-1.f;steps[k].u_ub[a]=1.f;}
    }
    RiccatiAdmmState_t memory{};memory.initialized=1;memory.rho=memory.rho_u=1.f;
    for(int k=0;k<=RICCATI_HORIZON;++k)memory.rho_state[k][0]=1.f+k;
    RiccatiAdmmConfig_t config{};config.rho=config.rho_u=1.f;
    config.rho_x_by_state[0]=1.f;config.max_iterations=1;config.tolerance=.01f;
    float Q[MPC_RTI_NX]{},q[MPC_RTI_NX]{},lo[MPC_RTI_NX],hi[MPC_RTI_NX],x0[MPC_RTI_NX]{};
    std::fill_n(lo,MPC_RTI_NX,-BIG_BOUND);std::fill_n(hi,MPC_RTI_NX,BIG_BOUND);
    RiccatiSolution_t solution{};
    ASSERT_NE(riccati_admm_solve(steps.get(),Q,q,lo,hi,x0,&config,&memory,&solution),RICCATI_STATUS_ERROR);
    EXPECT_FLOAT_EQ(memory.rho_state[1][0],2.f);
}

TEST(MpcTerminalViability, StationaryBackupHasNoArtificialHeadingOrSlipFunnel) {
    Straight s;auto c=configuration();s.initial.plant.u=0.f;
    s.initial.plant.delayed_throttle_command_1=0.f;
    s.initial.plant.delayed_throttle_command_2=0.f;
    s.initial.plant.source_plant_initialized=0;
    ASSERT_TRUE(vehicle_model_initialize_plant_state(&s.initial.plant));
    MpcModelControl_t controls[PREDICTION_HORIZON]{};
    auto nominal=std::make_unique<MpcRtiNominal_t>();
    MpcRtiReference_t refs[PREDICTION_HORIZON+1]{};
    ASSERT_TRUE(mpc_rti_build_nominal_same_epoch(&s.initial,0.,nullptr,controls,
        PREDICTION_HORIZON,s.track,s.count,s.lap,PREDICTION_DT_SECONDS,
        PREDICTION_HORIZON,&c.model,nominal.get(),refs));
    MpcRtiObjectiveSchedule_t frozen{};
    ASSERT_TRUE(mpc_rti_build_objective_schedule(nominal.get(),refs,&c.model,&frozen));
    ASSERT_TRUE(mpc_rti_initialize_terminal_targets(nominal.get(),s.track,s.count,
        s.lap,PREDICTION_DT_SECONDS,&c.model,&frozen));
    auto problem=std::make_unique<MpcRtiProblem_t>();
    ASSERT_TRUE(mpc_rti_add_terminal_viability(problem.get(),nominal.get(),
        s.track,s.count,s.lap,&c.model,&frozen));
    for(int f=0;f<problem->terminal_constraints.count;++f) {
        double norm=0.;
        for(float v:problem->terminal_constraints.normal[f])norm+=(double)v*v;
        EXPECT_GT(norm,0.);
    }
    double cost=0.;
    EXPECT_TRUE(mpc_rti_terminal_value(nominal.get(),s.track,s.count,s.lap,
        &c.model,&frozen,&cost,1));
}

TEST(MpcSqpSourceFirstStep, NearWallHighSlipBoxIsSourceCertified)
{
    Straight track;
    auto config=configuration();
    // Same source wheel/suspension state as a failed high-slip Unity corner,
    // with a standalone fixed-width reference to make this regression
    // independent of gitignored competition racelines and ROS bags.
    for(auto &p:track.track) {
        p.left_bound=1.10;
        p.right_bound=.40;
    }
    auto &x=track.initial.plant;
    x.e_y=-.109221704f;x.e_psi=.188616931f;
    x.u=4.53607178f;x.v=-.0278971419f;x.r=1.46069324f;
    x.actual_steering_angle=.111526541f;
    x.delayed_steering_command_1=.159697622f;
    x.delayed_steering_command_2=.159697622f;
    x.delayed_throttle_command_1=.270000011f;
    x.delayed_throttle_command_2=.275999993f;
    constexpr double hidden[22]={
        117.724031,116.928086,117.248094,116.372337,
        .00407802641,-1.06493075e-06,.0618434765,.0385258321,
        .0133695471,.0664072751,-.00988773326,.00445341637,
        -.00454001014,.0100709336,0,0,0,0,0,0,0,0};
    std::copy_n(hidden,22,x.source_plant_internal_state);
    x.source_plant_initialized=x.source_suspension_history_initialized=
        x.source_imu_state_initialized=1;
    config.model.nonlinear_corridor_tolerance_m=0.f;
    MpcModelControl_t nominal{.27f,.28f};
    float lb[2]={-.5235987f,MPC_THROTTLE_MIN_DRIVE};
    float ub[2]={.5235987f,1.f};
    int checks=0;
    ASSERT_TRUE(mpc_rti_source_first_step_feasible_box(
        &track.initial,5.,track.track,track.count,track.lap,
        PREDICTION_DT_SECONDS,&config,&nominal,0.,lb,ub,&checks));
    EXPECT_GE(checks,35);
    EXPECT_GT(ub[0],lb[0]);
    EXPECT_GT(ub[1],lb[1]);
    EXPECT_LE(ub[0]-lb[0],.175f);
    config.capture_diagnostic_outputs=1;
    MpcRtiNominal_t verified{};
    MpcRtiDiagnosticStageOutput_t diagnostics[MPC_RTI_DIAGNOSTIC_SAMPLE_COUNT]{};
    for(auto steering:{lb[0],.5f*(lb[0]+ub[0]),ub[0]})
      for(auto throttle:{lb[1],.5f*(lb[1]+ub[1]),ub[1]}) {
        MpcModelControl_t controls[PREDICTION_HORIZON]{};
        controls[0]={steering,throttle};
        int count=0,failed=-1;
        ASSERT_EQ(mpc_rti_validate_candidate_plan_with_diagnostics(
            &track.initial,5.,controls,track.track,track.count,
            track.lap,1,PREDICTION_DT_SECONDS,&config,&verified,
            diagnostics,&count,&failed),MPC_RTI_ROLLOUT_OK)
            <<"steering="<<steering<<" throttle="<<throttle;
        ASSERT_EQ(count,1);
        EXPECT_GE(verified.validation_minimum_corridor_slack_m,.0119f);
        EXPECT_LE(std::abs(diagnostics[0].output.lateral_slip[0]),.10001);
        EXPECT_LE(std::abs(diagnostics[0].output.lateral_slip[1]),.10001);
    }
}

TEST(MpcSqpLateralSlip, NativeSourceAndQpMeritShareExactGradient)
{
    Straight s;
    auto config=configuration();
    // Deliberately put the steered front wheels past the 0.1rad
    // Unity side-curve asymptote, where a bicycle tangent is unreliable.
    auto &x=s.initial.plant;
    x.actual_steering_angle=.45f;
    x.delayed_steering_command_1=.45f;
    x.delayed_steering_command_2=.45f;
    x.source_plant_initialized=0;
    ASSERT_TRUE(vehicle_model_initialize_plant_state(&x));
    MpcModelControl_t controls[PREDICTION_HORIZON]{};
    for(auto &control:controls){
        control.steering_target_rad=.45f;
        control.throttle_command=.08f;
    }
    auto nominal=std::make_unique<MpcRtiNominal_t>();
    MpcRtiReference_t refs[PREDICTION_HORIZON+1]{};
    ASSERT_TRUE(mpc_rti_build_nominal_same_epoch(
        &s.initial,0.,nullptr,controls,PREDICTION_HORIZON,
        s.track,s.count,s.lap,PREDICTION_DT_SECONDS,PREDICTION_HORIZON,
        &config.model,nominal.get(),refs));
    auto baseline=std::make_unique<MpcRtiProblem_t>();
    auto penalized=std::make_unique<MpcRtiProblem_t>();
    ASSERT_TRUE(mpc_rti_build_ltv_qp(nominal->states,controls,refs,
        PREDICTION_HORIZON,PREDICTION_DT_SECONDS,&config.model,baseline.get()));
    constexpr float weight=3500.f;
    config.model.weight_source_lateral_slip=weight;
    ASSERT_TRUE(mpc_rti_build_ltv_qp(nominal->states,controls,refs,
        PREDICTION_HORIZON,PREDICTION_DT_SECONDS,&config.model,penalized.get()));
    constexpr int k=1;
    double slip[4]{},J[4][MPC_OPT_PHYSICAL_NX]{};
    ASSERT_TRUE(vehicle_model_source_lateral_slip(
        &nominal->states[k].plant,slip,J));
    EXPECT_GT(std::max(std::abs(slip[0]),std::abs(slip[1])),.10);
    float state[MPC_RTI_NX]{};
    serialize_state(&nominal->states[k],state);
    const auto &prev=baseline->steps[k];
    const auto &cur=penalized->steps[k];
    for(int i=0;i<MPC_OPT_PHYSICAL_NX;++i) {
        double expected=0.;
        for(int front=0;front<2;++front) {
            if(std::abs(slip[front])<=
                kMpcSourceFrontLateralSlipReserveRad)continue;
            const double r=slip[front]-std::copysign(
                kMpcSourceFrontLateralSlipReserveRad,slip[front]);
            expected+=2.*weight*r*J[front][i];
        }
        double gradient=cur.q[i]-prev.q[i]+
            (cur.Q_diag[i]-prev.Q_diag[i])*state[i];
        for(int j=0;j<MPC_OPT_PHYSICAL_NX;++j)
            if(i!=j)
                gradient+=(cur.Q_offdiag[i][j]-prev.Q_offdiag[i][j])*state[j];
        EXPECT_NEAR(gradient,expected,std::max(.15,.003*std::abs(expected)))
            <<"coordinate "<<i;
    }
    // QP Hessian must be positive semidefinite in the physical subspace
    // for ANY direction, not just directions near the nominal.
    for(int seed=0;seed<10;++seed) {
        double v[MPC_OPT_PHYSICAL_NX]{};
        for(int i=0;i<MPC_OPT_PHYSICAL_NX;++i)
            v[i]=std::sin(1.+seed*2.1+i*.67);
        double quadratic=0.;
        for(int i=0;i<MPC_OPT_PHYSICAL_NX;++i) {
            quadratic+=v[i]*v[i]*(cur.Q_diag[i]-prev.Q_diag[i]);
            for(int j=0;j<MPC_OPT_PHYSICAL_NX;++j)
                if(i!=j)
                    quadratic+=v[i]*v[j]*
                        (cur.Q_offdiag[i][j]-prev.Q_offdiag[i][j]);
        }
        EXPECT_GE(quadratic,-.01);
    }
}

TEST(MpcSqpProgressHardCorridor, HardFootprintPlanesFollowExactTrackWidth)
{
    Straight track;
    track.track[0].left_bound=1.f;
    track.track[1].left_bound=.75f;
    track.track[2].left_bound=.5f;
    track.track[0].right_bound=.8f;
    track.track[1].right_bound=1.f;
    track.track[2].right_bound=1.2f;
    auto config=configuration();
    config.progress_coupled_hard_corridor=1;
    MpcModelControl_t controls[PREDICTION_HORIZON]{};
    for(auto &control:controls)control.throttle_command=.12f;
    auto plan=std::make_unique<MpcRtiNominal_t>();
    MpcRtiReference_t refs[PREDICTION_HORIZON+1]{};
    ASSERT_TRUE(mpc_rti_build_nominal_same_epoch(&track.initial,0.,nullptr,
        controls,PREDICTION_HORIZON,track.track,track.count,track.lap,
        PREDICTION_DT_SECONDS,PREDICTION_HORIZON,&config.model,plan.get(),refs));
    auto qp=std::make_unique<MpcRtiProblem_t>();
    ASSERT_TRUE(mpc_rti_build_ltv_qp_with_progress_corridor(
        plan->states,plan->controls,refs,PREDICTION_HORIZON,
        PREDICTION_DT_SECONDS,&config.model,nullptr,qp.get(),
        plan->progress,track.track,track.count,track.lap));
    constexpr int stage=4;
    const auto &plane=qp->state_polygons[stage];
    ASSERT_EQ(plane.count,8);
    EXPECT_NEAR(plane.progress_normal[0],.025f,1.e-4);
    EXPECT_NEAR(plane.progress_normal[1],-.02f,1.e-4);
    EXPECT_EQ(qp->steps[stage].x_lb[MPC_OPT_IDX_EY],-BIG_BOUND+1.f);
    EXPECT_EQ(qp->steps[stage].x_lb[MPC_OPT_IDX_PROGRESS_DEVIATION],
              -BIG_BOUND+1.f);
    const double ds=.15;
    MpcRtiReference_t shifted{};
    ASSERT_TRUE(mpc_rti_reference_at_progress(track.track,track.count,track.lap,
        plan->progress[stage]+ds,&config.model,&shifted));
    const auto &state=plan->states[stage].plant;
    float physical_right=0.f,physical_left=0.f;
    ASSERT_TRUE(mpc_rti_compute_corridor_slack(&state,shifted.left_bound,
        shifted.right_bound,&config.model,stage,&physical_right,&physical_left));
    float linear_left=INFINITY,linear_right=INFINITY;
    for(int face=0;face<plane.count;++face) {
        const float slack=plane.upper[face]-
            plane.normal[face][0]*state.e_y-
            plane.normal[face][1]*state.e_psi-
            plane.progress_normal[face]*ds;
        if(face%2==0)linear_left=std::min(linear_left,slack);
        else linear_right=std::min(linear_right,slack);
    }
    EXPECT_NEAR(linear_left,physical_left,1.e-4);
    EXPECT_NEAR(linear_right,physical_right,1.e-4);
}

TEST(MpcSqpSuccessorSourceGradient, ExactEightStepSourceTailControlDerivative)
{
    Straight straight;
    auto cfg=configuration();
    straight.initial.plant.e_y=.16f;
    straight.initial.plant.e_psi=.03f;
    for(auto &point:straight.track) {
        point.left_bound=.38f;
        point.right_bound=1.5f;
    }
    MpcModelControl_t controls[PREDICTION_HORIZON]{};
    for(int k=0;k<8;++k) {
        controls[k].steering_target_rad=.017f+.002f*k;
        controls[k].throttle_command=.12f+.004f*k;
    }
    MpcModelControl_t anchor[PREDICTION_HORIZON]{};
    std::copy_n(controls,8,anchor);
    double analytic[16][2]{},value=0.;
    ASSERT_TRUE(mpc_rti_source_successor_merit_gradient(
        &straight.initial,0.,controls,anchor,straight.track,straight.count,
        straight.lap,PREDICTION_DT_SECONDS,8,&cfg.model,&value,analytic));
    int nonzero=0;
    for(int k=0;k<8;++k)for(int channel=0;channel<2;++channel) {
        auto plus=std::make_unique<MpcModelControl_t[]>(PREDICTION_HORIZON);
        auto minus=std::make_unique<MpcModelControl_t[]>(PREDICTION_HORIZON);
        std::copy_n(controls,PREDICTION_HORIZON,plus.get());
        std::copy_n(controls,PREDICTION_HORIZON,minus.get());
        const float h=channel==0?.00005f:.0002f;
        if(channel==0) {
            plus[k].steering_target_rad+=h;
            minus[k].steering_target_rad-=h;
        } else {
            plus[k].throttle_command+=h;
            minus[k].throttle_command-=h;
        }
        double yp=NAN,ym=NAN;
        ASSERT_TRUE(mpc_rti_source_successor_merit_gradient(
            &straight.initial,0.,plus.get(),anchor,straight.track,
            straight.count,straight.lap,PREDICTION_DT_SECONDS,8,
            &cfg.model,&yp,nullptr))<<k<<":"<<channel;
        ASSERT_TRUE(mpc_rti_source_successor_merit_gradient(
            &straight.initial,0.,minus.get(),anchor,straight.track,
            straight.count,straight.lap,PREDICTION_DT_SECONDS,8,
            &cfg.model,&ym,nullptr))<<k<<":"<<channel;
        const double numerical=(yp-ym)/(2.*h);
        EXPECT_NEAR(analytic[k][channel],numerical,
            std::max(2.,.1*std::abs(numerical)))<<k<<":"<<channel;
        nonzero+=std::abs(numerical)>1.;
    }
    EXPECT_GT(nonzero,3);
}

TEST(MpcSqpReachableSteering, ActuatorRateCoordinatesReproduceFullSourceNominal)
{
    Straight track;
    auto cfg=configuration();
    cfg.model.command_increment_qp_enabled=1;
    cfg.model.source_reachable_steering_qp=1;
    cfg.model.max_steering_command_step_rad=2*SOURCE_MAX_STEERING_RAD;
    cfg.model.max_throttle_command_step=1.f;
    MpcModelControl_t controls[PREDICTION_HORIZON]{};
    for(int k=0;k<PREDICTION_HORIZON;++k) {
        controls[k].steering_target_rad=.035f+
            .08f*std::sin(.16f*k);
        controls[k].throttle_command=.12f+.005f*k;
    }
    auto nominal=std::make_unique<MpcRtiNominal_t>();
    MpcRtiReference_t refs[PREDICTION_HORIZON+1]{};
    ASSERT_TRUE(mpc_rti_build_nominal_same_epoch(&track.initial,0.,nullptr,
        controls,PREDICTION_HORIZON,track.track,track.count,track.lap,
        PREDICTION_DT_SECONDS,PREDICTION_HORIZON,&cfg.model,
        nominal.get(),refs));
    const float rate_limit=mpc_rti_steering_qp_step_limit(
        &cfg.model,PREDICTION_DT_SECONDS);
    EXPECT_LT(rate_limit,.081f);
    float prior=track.initial.plant.actual_steering_angle;
    for(int k=0;k<PREDICTION_HORIZON;++k) {
        const float requested=nominal->controls[k].steering_target_rad;
        EXPECT_LE(std::abs(requested-prior),rate_limit+1.e-5f)<<k;
        const float physical=nominal->states[k+1].plant.actual_steering_angle;
        EXPECT_NEAR(physical,requested,3.e-5)<<k;
        prior=physical;
    }
    auto qp=std::make_unique<MpcRtiProblem_t>();
    ASSERT_TRUE(mpc_rti_build_ltv_qp_with_objective_schedule(
        nominal->states,nominal->controls,refs,PREDICTION_HORIZON,
        PREDICTION_DT_SECONDS,&cfg.model,nullptr,qp.get()));
    double x[RICCATI_MAX_NX]{};
    std::copy_n(qp->x0,RICCATI_MAX_NX,x);
    for(int k=0;k<RICCATI_HORIZON;++k) {
        const auto &step=qp->steps[k];
        const double ds=nominal->controls[k].steering_target_rad-
            x[MPC_OPT_IDX_ACTUAL_STEERING_ANGLE];
        const double dt=nominal->controls[k].throttle_command-
            x[MPC_OPT_IDX_PREVIOUS_THROTTLE_COMMAND];
        EXPECT_LE(std::abs(ds),step.u_ub[0]+2.e-5)<<k;
        double next[RICCATI_MAX_NX]{};
        for(int i=0;i<RICCATI_MAX_NX;++i) {
            next[i]=step.d[i]+step.B[i][0]*ds+step.B[i][1]*dt;
            for(int j=0;j<RICCATI_MAX_NX;++j)
                next[i]+=double(step.A[i][j])*x[j];
        }
        float actual[RICCATI_MAX_NX]{};
        serialize_state(&nominal->states[k+1],actual);
        for(int i=0;i<MPC_OPT_PHYSICAL_NX;++i)
            EXPECT_NEAR(next[i],actual[i],.015)<<k<<":"<<i;
        EXPECT_NEAR(next[MPC_OPT_IDX_PROGRESS_DEVIATION],0.,2.e-4)<<k;
        EXPECT_NEAR(next[MPC_OPT_IDX_PREVIOUS_STEERING_COMMAND],
            nominal->controls[k].steering_target_rad,1.e-4)<<k;
        EXPECT_NEAR(next[MPC_OPT_IDX_PREVIOUS_THROTTLE_COMMAND],
            nominal->controls[k].throttle_command,1.e-4)<<k;
        std::copy_n(next,RICCATI_MAX_NX,x);
    }
}

TEST(MpcSqpReachableSteering, RealCycleReturnsPhysicalSteeringCommands)
{
    Straight track;
    auto cfg=configuration();
    cfg.model.command_increment_qp_enabled=1;
    cfg.model.source_reachable_steering_qp=1;
    cfg.model.max_steering_command_step_rad=2*SOURCE_MAX_STEERING_RAD;
    cfg.model.max_throttle_command_step=1.f;
    cfg.max_sqp_iterations=2;
    auto memory=std::make_unique<MpcRtiMemory_t>();
    auto result=std::make_unique<MpcRtiCycleResult_t>();
    const auto status=mpc_rti_solve_cycle(&track.initial,0.,track.track,
        track.count,track.lap,PREDICTION_DT_SECONDS,PREDICTION_HORIZON,
        &cfg,memory.get(),result.get());
    ASSERT_TRUE(status==MPC_RTI_CYCLE_ACCEPTED_OPTIMAL ||
        status==MPC_RTI_CYCLE_ACCEPTED_FEASIBLE ||
        status==MPC_RTI_CYCLE_ACCEPTED_LINE_SEARCH);
    ASSERT_TRUE(result->candidate_control_count==PREDICTION_HORIZON);
    float previous=track.initial.plant.actual_steering_angle;
    for(int k=0;k<PREDICTION_HORIZON;++k) {
        const float target=result->candidate_controls[k].steering_target_rad;
        EXPECT_LE(std::abs(target-previous),
            mpc_rti_steering_qp_step_limit(&cfg.model,PREDICTION_DT_SECONDS)+2.e-5)
            <<k;
        previous=target;
    }
}

TEST(MpcSqpReachableSteering, ExactSourceGradientMatchesFiniteControlDifference)
{
    Straight track;
    auto cfg=configuration();
    cfg.model.command_increment_qp_enabled=1;
    cfg.model.source_reachable_steering_qp=1;
    cfg.model.max_steering_command_step_rad=2*SOURCE_MAX_STEERING_RAD;
    cfg.model.max_throttle_command_step=1.f;
    MpcModelControl_t controls[PREDICTION_HORIZON]{};
    for(int k=0;k<PREDICTION_HORIZON;++k) {
        controls[k].steering_target_rad=.02f+.015f*std::sin(.22f*k);
        controls[k].throttle_command=.15f;
    }
    auto nominal=std::make_unique<MpcRtiNominal_t>();
    MpcRtiReference_t refs[PREDICTION_HORIZON+1]{};
    ASSERT_TRUE(mpc_rti_build_nominal_same_epoch(&track.initial,0.,nullptr,
        controls,PREDICTION_HORIZON,track.track,track.count,track.lap,
        PREDICTION_DT_SECONDS,PREDICTION_HORIZON,&cfg.model,
        nominal.get(),refs));
    MpcRtiObjectiveSchedule_t frozen{};
    ASSERT_TRUE(mpc_rti_build_objective_schedule(
        nominal.get(),refs,&cfg.model,&frozen));
    cfg.objective_schedule=&frozen;
    double derivative[PREDICTION_HORIZON][MPC_RTI_NU]{};
    ASSERT_TRUE(mpc_rti_source_adjoint_gradient_from_exact_nominal(
        nominal.get(),track.track,track.count,track.lap,
        PREDICTION_DT_SECONDS,&cfg.model,&frozen,derivative));
    const auto cost=[&](const MpcModelControl_t *input) {
        auto candidate=std::make_unique<MpcRtiNominal_t>();
        int failed=-1;
        EXPECT_EQ(mpc_rti_validate_candidate_plan(&track.initial,0.,input,
            track.track,track.count,track.lap,PREDICTION_HORIZON,
            PREDICTION_DT_SECONDS,&cfg,candidate.get(),&failed),
            MPC_RTI_ROLLOUT_OK);
        return candidate->validation_objective;
    };
    for(int stage:{0,3,12,24}) for(int axis=0;axis<2;++axis) {
        MpcModelControl_t plus[PREDICTION_HORIZON]{};
        MpcModelControl_t minus[PREDICTION_HORIZON]{};
        std::copy_n(nominal->controls,PREDICTION_HORIZON,plus);
        std::copy_n(nominal->controls,PREDICTION_HORIZON,minus);
        constexpr float h=1.e-3f;
        if(axis==0) {
            plus[stage].steering_target_rad+=h;
            minus[stage].steering_target_rad-=h;
        } else {
            plus[stage].throttle_command+=h;
            minus[stage].throttle_command-=h;
        }
        const double finite=(cost(plus)-cost(minus))/(2.*h);
        EXPECT_NEAR(derivative[stage][axis],finite,
            std::max(4.,.03*std::abs(finite)))
            <<stage<<":"<<axis;
    }
}

TEST(MpcQpFinitePrecision, RepairsOnlyRoundoffIndefinitenessAndPreservesNominalGradient)
{
    // Float32 casts of a large, nearly rank-1 source-suspension GN
    // quadratic can produce an indefinite state/control QP despite PSD
    // mathematics. Test the ACTUAL 25-coordinate Hessian after the
    // command-increment conversion, not a diagonal-Q surrogate.
    RiccatiStepData_t stage{};
    constexpr int i=MPC_OPT_IDX_BODY_HEIGHT;
    constexpr int j=MPC_OPT_IDX_SUSPENSION_JOUNCE_RR;
    stage.Q_diag[i]=100000.f;
    stage.Q_diag[j]=2.f;
    stage.Q_offdiag[i][j]=447.5f;
    stage.Q_offdiag[j][i]=447.5f;
    stage.Q_diag[MPC_OPT_IDX_PROGRESS_DEVIATION]=3.f;
    stage.R_diag[0]=6.f;
    stage.R_diag[1]=2.f;
    stage.N[MPC_OPT_IDX_EY][0]=.03f;
    stage.Q_diag[MPC_OPT_IDX_EY]=4.f;
    stage.q[i]=7.f;stage.q[j]=-11.f;
    stage.q[MPC_OPT_IDX_EY]=9.f;
    stage.r[0]=-8.f;stage.r[1]=.7f;
    float nominal_x[MPC_RTI_NX]{};
    nominal_x[i]=.026f;nominal_x[j]=.2f;
    nominal_x[MPC_OPT_IDX_EY]=.13f;
    const float nominal_u[MPC_RTI_NU]={.08f,-.02f};
    auto build_hessian=[](const RiccatiStepData_t &s) {
        Eigen::Matrix<double,MPC_RTI_NX+MPC_RTI_NU,
                      MPC_RTI_NX+MPC_RTI_NU> h;
        h.setZero();
        for(int a=0;a<MPC_RTI_NX;++a) {
            h(a,a)=double(s.Q_diag[a])+s.Q_offdiag[a][a];
            for(int b=a+1;b<MPC_RTI_NX;++b)
                h(a,b)=h(b,a)=.5*(double(s.Q_offdiag[a][b])+
                                                 s.Q_offdiag[b][a]);
            for(int b=0;b<MPC_RTI_NU;++b)
                h(a,MPC_RTI_NX+b)=h(MPC_RTI_NX+b,a)=s.N[a][b];
        }
        h(MPC_RTI_NX,MPC_RTI_NX)=s.R_diag[0];
        h(MPC_RTI_NX+1,MPC_RTI_NX+1)=s.R_diag[1];
        h(MPC_RTI_NX,MPC_RTI_NX+1)=
            h(MPC_RTI_NX+1,MPC_RTI_NX)=s.R_cross;
        return h;
    };
    auto nominal_gradient=[&](const RiccatiStepData_t &s) {
        Eigen::Matrix<double,MPC_RTI_NX+MPC_RTI_NU,1> z,g;
        for(int a=0;a<MPC_RTI_NX;++a)z[a]=nominal_x[a];
        for(int a=0;a<MPC_RTI_NU;++a)z[MPC_RTI_NX+a]=nominal_u[a];
        g=build_hessian(s)*z;
        for(int a=0;a<MPC_RTI_NX;++a)g[a]+=s.q[a];
        for(int a=0;a<MPC_RTI_NU;++a)g[MPC_RTI_NX+a]+=s.r[a];
        return g;
    };
    const auto before=nominal_gradient(stage);
    Eigen::SelfAdjointEigenSolver<
        Eigen::Matrix<double,MPC_RTI_NX+MPC_RTI_NU,
                      MPC_RTI_NX+MPC_RTI_NU>> initial(build_hessian(stage));
    ASSERT_EQ(initial.info(),Eigen::Success);
    EXPECT_LT(initial.eigenvalues()[0],-1.e-5);
    float applied=0.f;
    ASSERT_TRUE(mpc_rti_convexify_float_source_hessian(
        &stage,nominal_x,nominal_u,&applied));
    Eigen::SelfAdjointEigenSolver<
        Eigen::Matrix<double,MPC_RTI_NX+MPC_RTI_NU,
                      MPC_RTI_NX+MPC_RTI_NU>> after(build_hessian(stage));
    ASSERT_EQ(after.info(),Eigen::Success);
    EXPECT_GE(after.eigenvalues()[0],-1.e-7);
    EXPECT_GT(applied,0.f);
    EXPECT_LT(applied,.06f); // within the float32 roundoff allowance
    const Eigen::Matrix<double,MPC_RTI_NX+MPC_RTI_NU,1> gradient_error=
        (nominal_gradient(stage)-before).cwiseAbs();
    EXPECT_LT(gradient_error.maxCoeff(),.002);
    // A materially indefinite, deliberately nonconvex quadratic may not
    // be disguised by roundoff regularization.
    stage.Q_offdiag[i][j]=5000.f;
    stage.Q_offdiag[j][i]=5000.f;
    EXPECT_FALSE(mpc_rti_convexify_float_source_hessian(
        &stage,nominal_x,nominal_u,nullptr));
}

TEST(MpcSqpViolationFilter, InfeasibleIntermediateReducesExactSourceWallErrorButIsNeverCertified)
{
    Straight track;
    auto config=configuration();
    track.initial.plant.e_y=.48f; // already beyond the strict first-step wall
    for(auto &sample:track.track) {
        sample.left_bound=.55f;
        sample.right_bound=.55f;
    }
    MpcModelControl_t options[3][PREDICTION_HORIZON]{};
    for(int j=0;j<3;++j)
        for(int k=0;k<PREDICTION_HORIZON;++k) {
            options[j][k].throttle_command=.12f;
            options[j][k].steering_target_rad=
                j==0?-.42f:j==1?0.f:.42f;
        }
    double violation[3]{},minimum[3]{};
    for(int i=0;i<3;++i)
        ASSERT_TRUE(mpc_rti_exact_source_corridor_violation(&track.initial,
            0.,options[i],track.track,track.count,track.lap,
            PREDICTION_DT_SECONDS,&config.model,&violation[i],&minimum[i]));
    const auto least=std::min_element(violation,violation+3)-violation;
    const auto worst=std::max_element(violation,violation+3)-violation;
    ASSERT_GT(violation[worst],violation[least]+.0001);
    ASSERT_LT(minimum[worst],0.);
    MpcModelControl_t selected[PREDICTION_HORIZON]{};
    double initial_v=NAN,selected_v=NAN;
    float alpha=0.f;
    ASSERT_TRUE(mpc_rti_select_source_violation_decreasing_step(
        &track.initial,0.,options[worst],options[least],
        track.track,track.count,track.lap,PREDICTION_DT_SECONDS,
        &config.model,selected,&initial_v,&selected_v,&alpha));
    EXPECT_GT(alpha,0.f);
    EXPECT_LE(alpha,1.f);
    EXPECT_NEAR(initial_v,violation[worst],1.e-8);
    EXPECT_LT(selected_v,initial_v);
    // This is an SQP initializer, not an allowed published control.
    // In particular, it cannot undo the fixed measured stage-0 violation.
    auto plan=std::make_unique<MpcRtiNominal_t>();
    int failure_stage=-1;
    EXPECT_NE(mpc_rti_validate_candidate_plan(&track.initial,0.,selected,
        track.track,track.count,track.lap,PREDICTION_HORIZON,
        PREDICTION_DT_SECONDS,&config,plan.get(),&failure_stage),
        MPC_RTI_ROLLOUT_OK);
    // A strictly valid source nominal is left untouched: the filter must
    // not turn a safe source trajectory into an infeasible SQP iterate.
    Straight open_track;
    MpcModelControl_t safe[PREDICTION_HORIZON]{};
    MpcModelControl_t unsafe[PREDICTION_HORIZON]{};
    for(int k=0;k<PREDICTION_HORIZON;++k) {
        safe[k].throttle_command=unsafe[k].throttle_command=.12f;
        unsafe[k].steering_target_rad=.42f;
    }
    ASSERT_TRUE(mpc_rti_select_source_violation_decreasing_step(
        &open_track.initial,0.,safe,unsafe,open_track.track,
        open_track.count,open_track.lap,PREDICTION_DT_SECONDS,
        &config.model,selected,&initial_v,&selected_v,&alpha));
    EXPECT_EQ(alpha,0.f);
    EXPECT_EQ(initial_v,0.);
    EXPECT_EQ(selected_v,0.);
}

TEST(MpcSqpElasticCorridor, InfeasibleSourceNominalHasAWellDefinedElasticQp)
{
    Straight track;
    auto cfg=configuration();
    track.initial.plant.e_y=.46f;
    for(auto &point:track.track) {
        point.left_bound=.55f;
        point.right_bound=.55f;
    }
    MpcModelControl_t controls[PREDICTION_HORIZON]{};
    for(auto &u:controls)u.throttle_command=.12f;
    auto nominal=std::make_unique<MpcRtiNominal_t>();
    auto qp=std::make_unique<MpcRtiProblem_t>();
    MpcRtiReference_t references[PREDICTION_HORIZON+1]{};
    ASSERT_TRUE(mpc_rti_build_nominal_same_epoch(&track.initial,0.,nullptr,
        controls,PREDICTION_HORIZON,track.track,track.count,track.lap,
        PREDICTION_DT_SECONDS,PREDICTION_HORIZON,&cfg.model,nominal.get(),
        references));
    ASSERT_TRUE(mpc_rti_build_ltv_qp_with_objective_schedule(
        nominal->states,nominal->controls,references,PREDICTION_HORIZON,
        PREDICTION_DT_SECONDS,&cfg.model,nullptr,qp.get()));
    EXPECT_EQ(qp->state_polygons[1].elastic_weight,0.f);
    int active=0;
    ASSERT_TRUE(mpc_rti_prepare_elastic_corridor_qp(
        qp.get(),nominal.get(),references,&cfg.model,8000.f,&active));
    ASSERT_EQ(active,1);
    for(int k=1;k<=RICCATI_HORIZON;++k) {
        const auto &polygon=qp->state_polygons[k];
        EXPECT_FLOAT_EQ(polygon.elastic_weight,8000.f);
        ASSERT_EQ(polygon.count,8);
        for(int face=0;face<polygon.count;++face)
            EXPECT_EQ(polygon.upper[face],
                      qp->state_polygons[k].upper[face]);
        const float lb=k<RICCATI_HORIZON
            ?qp->steps[k].x_lb[MPC_OPT_IDX_EY]
            :qp->terminal_x_lb[MPC_OPT_IDX_EY];
        EXPECT_FLOAT_EQ(lb,-BIG_BOUND+1.f);
    }
    // The physical nominal and physical strict source validation remain
    // unchanged by the penalized mathematical QP construction.
    int failure=-1;
    auto physical=std::make_unique<MpcRtiNominal_t>();
    EXPECT_NE(mpc_rti_validate_candidate_plan(&track.initial,0.,controls,
        track.track,track.count,track.lap,PREDICTION_HORIZON,
        PREDICTION_DT_SECONDS,&cfg,physical.get(),&failure),
        MPC_RTI_ROLLOUT_OK);
}

TEST(MpcSqpElasticCorridor, SourceCertifiedNominalKeepsHardPolygons)
{
    Straight track;
    auto cfg=configuration();
    MpcModelControl_t controls[PREDICTION_HORIZON]{};
    for(auto &u:controls)u.throttle_command=.12f;
    auto nominal=std::make_unique<MpcRtiNominal_t>();
    auto qp=std::make_unique<MpcRtiProblem_t>();
    MpcRtiReference_t references[PREDICTION_HORIZON+1]{};
    ASSERT_TRUE(mpc_rti_build_nominal_same_epoch(&track.initial,0.,nullptr,
        controls,PREDICTION_HORIZON,track.track,track.count,track.lap,
        PREDICTION_DT_SECONDS,PREDICTION_HORIZON,&cfg.model,nominal.get(),
        references));
    ASSERT_TRUE(mpc_rti_build_ltv_qp_with_objective_schedule(
        nominal->states,nominal->controls,references,PREDICTION_HORIZON,
        PREDICTION_DT_SECONDS,&cfg.model,nullptr,qp.get()));
    int active=-1;
    ASSERT_TRUE(mpc_rti_prepare_elastic_corridor_qp(
        qp.get(),nominal.get(),references,&cfg.model,8000.f,&active));
    ASSERT_EQ(active,0);
    for(int k=1;k<=RICCATI_HORIZON;++k)
        EXPECT_FLOAT_EQ(qp->state_polygons[k].elastic_weight,0.f);
}

TEST(MpcSqpElasticAdmm, ReconditionedWarmStartPreservesUnscaledDualsAcrossRhoTransitions)
{
    auto memory=std::make_unique<RiccatiAdmmState_t>();
    riccati_admm_state_init(memory.get());
    memory->initialized=1;
    memory->rho=7.f;memory->rho_u=11.f;
    const int index0=MPC_OPT_IDX_EY;
    const int index1=MPC_OPT_IDX_PROGRESS_DEVIATION;
    for(int k=0;k<=RICCATI_HORIZON;++k) {
        for(int i=0;i<RICCATI_MAX_NX;++i) {
            memory->rho_state[k][i]=7.f;
            memory->y_x[k][i]=.4f + .001f*i + .002f*k;
        }
    }
    for(int k=0;k<RICCATI_HORIZON;++k) {
        for(int j=0;j<RICCATI_MAX_NU;++j) {
            memory->rho_input[k][j]=11.f;
            memory->y_u[k][j]=.14f + .003f*k+.001f*j;
        }
    }
    auto previous=*memory;
    RiccatiAdmmConfig_t elastic{};
    elastic.rho=elastic.rho_u=127.f;
    for(int i=0;i<RICCATI_MAX_NX;++i)
        elastic.rho_x_by_state[i]=127.f;
    for(int j=0;j<RICCATI_MAX_NU;++j)
        elastic.rho_u_by_input[j]=127.f;
    ASSERT_TRUE(mpc_rti_recondition_admm_penalties(memory.get(),&elastic));
    for(int k:{0,12,30}) {
        for(int i:{index0,index1,RICCATI_MAX_NX-1}) {
            EXPECT_FLOAT_EQ(memory->rho_state[k][i],127.f);
            EXPECT_NEAR(127.*memory->y_x[k][i],
                        7.*previous.y_x[k][i],2.e-5);
            EXPECT_FLOAT_EQ(memory->z_x[k][i],previous.z_x[k][i]);
        }
        if(k<RICCATI_HORIZON)
            for(int j=0;j<RICCATI_MAX_NU;++j)
                EXPECT_NEAR(127.*memory->y_u[k][j],
                            11.*previous.y_u[k][j],2.e-5);
    }
    RiccatiAdmmConfig_t hard{};
    hard.rho=7.f;hard.rho_u=11.f;
    ASSERT_TRUE(mpc_rti_recondition_admm_penalties(memory.get(),&hard));
    EXPECT_FLOAT_EQ(memory->rho,7.f);
    EXPECT_FLOAT_EQ(memory->rho_u,11.f);
    for(int k:{0,12,30}) {
        for(int i:{index0,index1,RICCATI_MAX_NX-1})
            EXPECT_NEAR(memory->y_x[k][i],previous.y_x[k][i],2.e-6);
        if(k<RICCATI_HORIZON)
            for(int j=0;j<RICCATI_MAX_NU;++j)
                EXPECT_NEAR(memory->y_u[k][j],previous.y_u[k][j],2.e-6);
    }
}

TEST(MpcSqpElasticAdmm, InfeasibleNominalBuildsElasticR1AndKeepsStrictValidation)
{
    Straight track;
    auto cfg=configuration();
    cfg.model.command_increment_qp_enabled=1;
    cfg.model.source_reachable_steering_qp=1;
    cfg.model.max_steering_command_step_rad=2.f*SOURCE_MAX_STEERING_RAD;
    cfg.model.max_throttle_command_step=1.f;
    cfg.source_violation_filter_sqp=1;
    cfg.source_elastic_wall_weight=500.f;
    cfg.solver.tolerance=.02f;
    cfg.solver.max_iterations=150;
    cfg.solver.rho=cfg.solver.rho_u=7.f;
    cfg.solver.rho_max=32.f;
    track.initial.plant.e_y=.48f;
    for(auto &sample:track.track) {
        sample.left_bound=.55f;
        sample.right_bound=.55f;
    }
    auto capture=std::make_unique<MpcRtiQpCapture_t>();
    cfg.qp_capture=capture.get();
    auto memory=std::make_unique<MpcRtiMemory_t>();
    auto result=std::make_unique<MpcRtiCycleResult_t>();
    const auto status=mpc_rti_solve_cycle(&track.initial,0.,track.track,
         track.count,track.lap,PREDICTION_DT_SECONDS,
         PREDICTION_HORIZON,&cfg,memory.get(),result.get());
    ASSERT_TRUE(capture->valid);
    EXPECT_FLOAT_EQ(capture->solver_config.rho,127.f);
    EXPECT_FLOAT_EQ(capture->solver_config.rho_u,127.f);
    EXPECT_FLOAT_EQ(capture->solver_config.rho_max,127.f);
    EXPECT_FLOAT_EQ(capture->solver_config.tolerance,.002f);
    EXPECT_GE(capture->solver_config.max_iterations,300);
    EXPECT_FLOAT_EQ(capture->problem.state_polygons[1].elastic_weight,500.f);
    EXPECT_FLOAT_EQ(capture->solver_config.state_polygons[1].elastic_weight,500.f);
    // A soft/elastic QP cannot turn an already-missed stage-0 physical
    // clearance into a legal published trajectory.
    EXPECT_NE(status,MPC_RTI_CYCLE_ACCEPTED_FEASIBLE);
    EXPECT_NE(status,MPC_RTI_CYCLE_ACCEPTED_LINE_SEARCH);
}

TEST(MpcSqpElasticAdmm, RawQpCertificateRejectsProjectedOnlyHardFeasibility)
{
    auto qp=std::make_unique<MpcRtiProblem_t>();
    qp->horizon=RICCATI_HORIZON;
    for(int k=0;k<RICCATI_HORIZON;++k) {
        for(int i=0;i<RICCATI_MAX_NX;++i) {
            qp->steps[k].x_lb[i]=-BIG_BOUND;
            qp->steps[k].x_ub[i]= BIG_BOUND;
        }
        for(int a=0;a<RICCATI_MAX_NU;++a) {
            qp->steps[k].u_lb[a]=-BIG_BOUND;
            qp->steps[k].u_ub[a]= BIG_BOUND;
        }
    }
    for(int i=0;i<RICCATI_MAX_NX;++i) {
        qp->terminal_x_lb[i]=-BIG_BOUND;
        qp->terminal_x_ub[i]= BIG_BOUND;
    }
    auto &poly=qp->state_polygons[1];
    poly.count=1;
    poly.normal[0][0]=1.f;
    poly.upper[0]=.1f;
    auto solved=std::make_unique<RiccatiSolution_t>();
    auto warm=std::make_unique<RiccatiAdmmState_t>();
    riccati_admm_state_init(warm.get());
    solved->x[1][MPC_OPT_IDX_EY]=.12f;
    warm->z_x[1][MPC_OPT_IDX_EY]=.1f;
    qp->steps[0].d[MPC_OPT_IDX_EY]=.12f;
    float raw_primal_u[RICCATI_HORIZON][RICCATI_MAX_NU]{};
    MpcRtiQpPrimalCertificate_t cert{};
    ASSERT_TRUE(mpc_rti_qp_primal_certificate(
        qp.get(),solved.get(),warm.get(),raw_primal_u,&cert));
    EXPECT_NEAR(cert.max_hard_constraint_error,.02,2.e-8);
    EXPECT_NEAR(cert.max_physical_split_error,.02,2.e-8);
    // The SAME nominal is legal only in the elastic-QP sense. It must still
    // be classified physically illegal by the exact-source publisher.
    poly.elastic_weight=500.f;
    ASSERT_TRUE(mpc_rti_qp_primal_certificate(
        qp.get(),solved.get(),warm.get(),raw_primal_u,&cert));
    EXPECT_EQ(cert.max_hard_constraint_error,0.);
    EXPECT_NEAR(cert.max_elastic_wall_violation_m,.02,2.e-8);
    EXPECT_NEAR(cert.max_physical_split_error,.02,2.e-8);
    // In contrast an exact primal/split match on a hard wall is certified.
    solved->x[1][MPC_OPT_IDX_EY]=.10f;
    qp->steps[0].d[MPC_OPT_IDX_EY]=.10f;
    poly.elastic_weight=0.f;
    ASSERT_TRUE(mpc_rti_qp_primal_certificate(
        qp.get(),solved.get(),warm.get(),raw_primal_u,&cert));
    EXPECT_EQ(cert.max_hard_constraint_error,0.);
    EXPECT_EQ(cert.max_physical_split_error,0.);
    EXPECT_LT(cert.max_dynamics_equality_error,1.e-7);
    // A fake low-residual QP with a nonlinear/dynamics-inconsistent
    // successor is rejected even when its projected wall state is legal.
    qp->steps[0].d[MPC_OPT_IDX_EY]=.11f;
    ASSERT_TRUE(mpc_rti_qp_primal_certificate(
        qp.get(),solved.get(),warm.get(),raw_primal_u,&cert));
    EXPECT_NEAR(cert.max_dynamics_equality_error,.01,2.e-8);
}
