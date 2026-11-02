#include "mpc_rti.h"
#include "../src/mpc_rti_internal.hpp"
#include <gtest/gtest.h>
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

TEST(MpcSqpConvergence, SourceInformedModeHonorsTwoPassLimit) {
    Straight s;auto c=configuration();c.source_informed_sqp=1;
    auto memory=std::make_unique<MpcRtiMemory_t>();
    auto result=std::make_unique<MpcRtiCycleResult_t>();
    mpc_rti_solve_cycle(&s.initial,0.,s.track,s.count,s.lap,
        PREDICTION_DT_SECONDS,PREDICTION_HORIZON,&c,memory.get(),result.get());
    ASSERT_TRUE(memory->nominal.valid);
    EXPECT_EQ(result->sqp_iterations_completed,2);
    EXPECT_TRUE(std::isfinite(result->first_pass_objective));
    EXPECT_LE(result->nonlinear_objective,result->first_pass_objective);
    EXPECT_GE(result->exact_objective_improvement,0.);
    EXPECT_EQ(result->source_nonlinear_exact_evaluations,0);
    EXPECT_EQ(result->experimental_source_nonlinear_selected,0);
    EXPECT_TRUE(std::isfinite(result->source_directional_derivative));
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
