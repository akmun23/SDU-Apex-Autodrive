#include "mpc_rti.h"
#include <gtest/gtest.h>
#include <cmath>
#include <memory>

TEST(MpcFeasibilityRestoration, QpOnlyRelaxationCreatesConvexBoundaryRecoveryPressure) {
    auto qp = std::make_unique<MpcRtiProblem_t>();
    auto nominal = std::make_unique<MpcRtiNominal_t>();
    nominal->valid = 1;
    nominal->horizon = PREDICTION_HORIZON;
    MpcRtiConfiguration_t cfg{};
    cfg.max_steering_rad = .523598775f;
    for (int k=0;k<=RICCATI_HORIZON;++k) {
        const int j = k==RICCATI_HORIZON ? PREDICTION_HORIZON : k+MPC_COMMAND_DELAY_STEPS;
        nominal->states[j].plant.e_y=.35f;
        nominal->states[j].plant.e_psi=0.f;
        auto &poly=qp->state_polygons[k];
        poly.count=1;
        poly.normal[0][0]=1.f;
        poly.normal[0][1]=.25f;
        poly.upper[0]=.30f;
        float *lb=k==RICCATI_HORIZON ? qp->terminal_x_lb : qp->steps[k].x_lb;
        float *ub=k==RICCATI_HORIZON ? qp->terminal_x_ub : qp->steps[k].x_ub;
        lb[MPC_OPT_IDX_EY]=-.30f;
        ub[MPC_OPT_IDX_EY]=.30f;
        if(k<RICCATI_HORIZON) {
            auto &stage=qp->steps[k];
            stage.Q_diag[MPC_OPT_IDX_U]=40.f;
            stage.q[MPC_OPT_IDX_U]=-12.f;
            stage.u_lb[0]=-cfg.max_steering_rad;
            stage.u_ub[0]=cfg.max_steering_rad;
            stage.u_lb[1]=MPC_THROTTLE_MIN_DRIVE;
            stage.u_ub[1]=MPC_THROTTLE_MAX;
            nominal->controls[j].steering_target_rad=0.f;
            nominal->controls[j].throttle_command=.3f;
        }
    }
    EXPECT_EQ(mpc_rti_prepare_feasibility_restoration_qp(qp.get(), nominal.get(),
        .18f,8000.f),1);
    EXPECT_NEAR(qp->state_polygons[0].upper[0],.48f,1e-5f);
    EXPECT_NEAR(qp->steps[0].x_ub[MPC_OPT_IDX_EY],.48f,1e-5f);
    EXPECT_GT(qp->steps[0].Q_diag[MPC_OPT_IDX_EY],0.f);
    EXPECT_GT(qp->steps[0].Q_diag[MPC_OPT_IDX_EPSI],0.f);
    // q is negative because Q*x+q produces a positive gradient away from
    // the violated +e_y wall; a positive q would push in the WRONG direction.
    const float derivative = qp->steps[0].Q_diag[MPC_OPT_IDX_EY]*.35f +
        qp->steps[0].q[MPC_OPT_IDX_EY];
    EXPECT_GT(derivative,100.f);
    EXPECT_NEAR(qp->steps[0].Q_diag[MPC_OPT_IDX_U],40.f,1e-5f);
    EXPECT_NEAR(qp->steps[0].u_lb[0],-cfg.max_steering_rad,1e-5f);
    EXPECT_NEAR(qp->steps[0].u_ub[0],cfg.max_steering_rad,1e-5f);
    EXPECT_NEAR(qp->steps[0].u_lb[1],MPC_THROTTLE_MIN_DRIVE,1e-5f);
    EXPECT_NEAR(qp->steps[0].u_ub[1],MPC_THROTTLE_MAX,1e-5f);
}

TEST(MpcFeasibilityRestoration, DisabledOrInvalidParametersCannotChangeQp) {
    auto qp=std::make_unique<MpcRtiProblem_t>();
    auto nominal=std::make_unique<MpcRtiNominal_t>();
    nominal->valid=1;
    EXPECT_EQ(mpc_rti_prepare_feasibility_restoration_qp(qp.get(),nominal.get(),
        0.f,8000.f),0);
    EXPECT_EQ(mpc_rti_prepare_feasibility_restoration_qp(qp.get(),nominal.get(),
        .18f,0.f),0);
    EXPECT_FLOAT_EQ(qp->steps[0].Q_diag[MPC_OPT_IDX_EY],0.f);
}

TEST(MpcFeasibilityRestoration, IncrementQpPreservesSignedCommandChanges) {
    auto qp = std::make_unique<MpcRtiProblem_t>();
    auto nominal = std::make_unique<MpcRtiNominal_t>();
    nominal->valid = 1;
    nominal->horizon = PREDICTION_HORIZON;
    // Regression: the restoration objective must not overwrite signed
    // command-difference limits or the absolute-command state boxes.
    for (int k = 0; k < RICCATI_HORIZON; ++k) {
        auto &stage = qp->steps[k];
        stage.u_lb[0] = -.12f;
        stage.u_ub[0] = .12f;
        stage.u_lb[1] = -.25f;
        stage.u_ub[1] = .25f;
        stage.x_lb[MPC_OPT_IDX_PREVIOUS_THROTTLE_COMMAND] =
            MPC_THROTTLE_MIN_DRIVE;
        stage.x_ub[MPC_OPT_IDX_PREVIOUS_THROTTLE_COMMAND] =
            MPC_THROTTLE_MAX;
    }
    EXPECT_EQ(mpc_rti_prepare_feasibility_restoration_qp(
        qp.get(), nominal.get(), .18f, 8000.f), 1);
    for (int k = 0; k < RICCATI_HORIZON; ++k) {
        const auto &stage = qp->steps[k];
        EXPECT_FLOAT_EQ(stage.u_lb[0], -.12f);
        EXPECT_FLOAT_EQ(stage.u_ub[0], .12f);
        EXPECT_FLOAT_EQ(stage.u_lb[1], -.25f);
        EXPECT_FLOAT_EQ(stage.u_ub[1], .25f);
        EXPECT_FLOAT_EQ(stage.x_lb[MPC_OPT_IDX_PREVIOUS_THROTTLE_COMMAND],
            MPC_THROTTLE_MIN_DRIVE);
        EXPECT_FLOAT_EQ(stage.x_ub[MPC_OPT_IDX_PREVIOUS_THROTTLE_COMMAND],
            MPC_THROTTLE_MAX);
    }
}

TEST(MpcFeasibilityRestoration, PreservesCoupledSlipConvexity) {
    auto qp=std::make_unique<MpcRtiProblem_t>();
    auto nominal=std::make_unique<MpcRtiNominal_t>();
    nominal->valid=1;nominal->horizon=PREDICTION_HORIZON;
    // Exact squared slip residual (u-wheel)^2. Reducing only Q_uu by
    // 90% used to make this joint Hessian indefinite (determinant -3.6).
    auto &s=qp->steps[5];
    s.Q_diag[MPC_OPT_IDX_U]=s.Q_diag[MPC_OPT_IDX_WHEEL_OMEGA_RL]=2.f;
    s.Q_offdiag[MPC_OPT_IDX_U][MPC_OPT_IDX_WHEEL_OMEGA_RL]=-2.f;
    s.Q_offdiag[MPC_OPT_IDX_WHEEL_OMEGA_RL][MPC_OPT_IDX_U]=-2.f;
    ASSERT_TRUE(mpc_rti_prepare_feasibility_restoration_qp(qp.get(),nominal.get(),.18f,8000.f));
    const double determinant=s.Q_diag[MPC_OPT_IDX_U]*s.Q_diag[MPC_OPT_IDX_WHEEL_OMEGA_RL]-
        s.Q_offdiag[MPC_OPT_IDX_U][MPC_OPT_IDX_WHEEL_OMEGA_RL]*
        s.Q_offdiag[MPC_OPT_IDX_WHEEL_OMEGA_RL][MPC_OPT_IDX_U];
    EXPECT_GE(determinant,0.);
    EXPECT_FLOAT_EQ(s.Q_diag[MPC_OPT_IDX_U],2.f);
}
