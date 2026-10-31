#include "mpc_viability_recovery.hpp"
#include <gtest/gtest.h>
#include <cmath>

TEST(MpcViabilityRecovery, TerminalScreenRejectsSlipAndHeading)
{
  MpcModelState_t s{};
  s.u=4.f; s.v=.1f; s.e_psi=.01f;
  EXPECT_FALSE(f1tenth_mpc::terminal_instability_predicted(s));
  EXPECT_TRUE(f1tenth_mpc::terminal_recovered(s));
  s.v=2.5f;
  EXPECT_TRUE(f1tenth_mpc::terminal_instability_predicted(s));
  EXPECT_FALSE(f1tenth_mpc::terminal_recovered(s));
  s.v=.1f; s.e_psi=-.5f;
  EXPECT_TRUE(f1tenth_mpc::terminal_instability_predicted(s));
  EXPECT_FALSE(f1tenth_mpc::terminal_recovered(s));
  s.e_psi=std::nanf("");
  EXPECT_TRUE(f1tenth_mpc::terminal_instability_predicted(s));
}

TEST(MpcViabilityRecovery, TaperPreservesFullNominalTailAndActuatorBounds)
{
  MpcModelControl_t baseline[30]{};
  MpcModelControl_t repaired[30]{};
  for (int k=0;k<30;++k) {
    baseline[k].steering_target_rad=-.40f;
    baseline[k].throttle_command=.15f;
  }
  f1tenth_mpc::taper_viability_controls(
      baseline,repaired,30,12,.18f,-.08f,.5235988f);
  EXPECT_NEAR(repaired[0].steering_target_rad,-.22f,1e-6);
  EXPECT_NEAR(repaired[0].throttle_command,.07f,1e-6);
  EXPECT_NEAR(repaired[11].steering_target_rad,-.385f,1e-6);
  EXPECT_FLOAT_EQ(repaired[12].steering_target_rad,baseline[12].steering_target_rad);
  EXPECT_FLOAT_EQ(repaired[29].throttle_command,baseline[29].throttle_command);
  f1tenth_mpc::taper_viability_controls(
      baseline,repaired,30,16,-.18f,-.80f,.5235988f);
  EXPECT_GE(repaired[0].steering_target_rad,-.5235988f);
  EXPECT_GE(repaired[0].throttle_command,MPC_THROTTLE_MIN_DRIVE);
}

TEST(MpcViabilityRecovery, SteadyBridgeUsesPhysicalCommandBounds)
{
  MpcModelControl_t commands[30]{};
  f1tenth_mpc::steady_viability_controls(
      commands,30,0.25f,0.10f,0.5235988f);
  for (int k=0;k<30;++k) {
    EXPECT_FLOAT_EQ(commands[k].steering_target_rad,0.25f);
    EXPECT_FLOAT_EQ(commands[k].throttle_command,0.10f);
  }
  f1tenth_mpc::steady_viability_controls(
      commands,30,1.f,-1.f,0.5235988f);
  for (int k=0;k<30;++k) {
    EXPECT_FLOAT_EQ(commands[k].steering_target_rad,0.5235988f);
    EXPECT_GE(commands[k].throttle_command,MPC_THROTTLE_MIN_DRIVE);
  }
}

TEST(MpcViabilityRecovery, TargetedRepairPreservesEarlyAndLateControls)
{
    MpcModelControl_t base[30]{};
    MpcModelControl_t corrected[30]{};
    for (int k=0;k<30;++k) {
        base[k].steering_target_rad = 0.1f;
        base[k].throttle_command = 0.20f;
    }
    f1tenth_mpc::targeted_viability_controls(
        base, corrected, 30, 6, 21, 0.2f, -0.12f, 0.5235988f);
    for (int k=0;k<6;++k) {
        EXPECT_FLOAT_EQ(corrected[k].steering_target_rad,base[k].steering_target_rad);
        EXPECT_FLOAT_EQ(corrected[k].throttle_command,base[k].throttle_command);
    }
    EXPECT_GT(corrected[12].steering_target_rad,base[12].steering_target_rad);
    EXPECT_LT(corrected[12].throttle_command,base[12].throttle_command);
    for (int k=21;k<30;++k) {
        EXPECT_FLOAT_EQ(corrected[k].steering_target_rad,base[k].steering_target_rad);
        EXPECT_FLOAT_EQ(corrected[k].throttle_command,base[k].throttle_command);
    }
    f1tenth_mpc::targeted_viability_controls(
        base, corrected, 30, 6, 21, -3.f, -3.f, 0.5235988f);
    for (int k=0;k<30;++k) {
        EXPECT_GE(corrected[k].steering_target_rad,-0.5235988f);
        EXPECT_LE(corrected[k].steering_target_rad,0.5235988f);
        EXPECT_GE(corrected[k].throttle_command,MPC_THROTTLE_MIN_DRIVE);
    }
}

TEST(MpcViabilityRecovery, DirectedEarlyHazardRepairIsFrontLoadedAndBounded)
{
    MpcModelControl_t source[30]{},corrected[30]{};
    for(int i=0;i<30;++i) {
        source[i].throttle_command=.45f;
        source[i].steering_target_rad=.05f;
    }
    f1tenth_mpc::early_hazard_viability_controls(
        source,corrected,30,20,.3f,-.12f,.5235988f);
    EXPECT_GT(corrected[0].steering_target_rad,.18f);
    EXPECT_GT(corrected[1].steering_target_rad,.34f);
    EXPECT_FLOAT_EQ(corrected[18].steering_target_rad,.05f);
    EXPECT_FLOAT_EQ(corrected[29].throttle_command,.45f);
    EXPECT_FLOAT_EQ(f1tenth_mpc::corridor_corrective_sign(-.5f,-.3f,.2f),1.f);
    EXPECT_FLOAT_EQ(f1tenth_mpc::corridor_corrective_sign(.4f,-.3f,.2f),-1.f);
    EXPECT_FLOAT_EQ(f1tenth_mpc::corridor_corrective_sign(0.f,-.3f,.2f),0.f);
}

TEST(MpcViabilityRecovery, RareDampedWallWarningIgnoresNormalFastSolve)
{
    using f1tenth_mpc::line_search_wall_early_warning;
    EXPECT_TRUE(line_search_wall_early_warning(
        MPC_RTI_CYCLE_ACCEPTED_LINE_SEARCH,MPC_RTI_ROLLOUT_CORRIDOR,
        .125f,.018f,.25f,.025f));
    EXPECT_FALSE(line_search_wall_early_warning(
        MPC_RTI_CYCLE_ACCEPTED_LINE_SEARCH,MPC_RTI_ROLLOUT_CORRIDOR,
        .5f,.018f,.25f,.025f));
    EXPECT_FALSE(line_search_wall_early_warning(
        MPC_RTI_CYCLE_ACCEPTED_LINE_SEARCH,MPC_RTI_ROLLOUT_CORRIDOR,
        .125f,.08f,.25f,.025f));
    EXPECT_FALSE(line_search_wall_early_warning(
        MPC_RTI_CYCLE_ACCEPTED_OPTIMAL,MPC_RTI_ROLLOUT_CORRIDOR,
        .125f,.018f,.25f,.025f));
}

TEST(MpcViabilityRecovery, GeometricSeedInterpolatesAndEnforcesLegalInputs)
{
    float feedforward[10]{};
    for(int j=0;j<10;++j) feedforward[j]=.01f*float(j);
    MpcModelControl_t commands[PREDICTION_HORIZON]{};
    ASSERT_TRUE(f1tenth_mpc::geometric_feasibility_controls(
        feedforward,1.2f,.12f,.5235987f,commands));
    EXPECT_NEAR(commands[0].steering_target_rad,0.f,1.e-7f);
    EXPECT_NEAR(commands[29].steering_target_rad,.108f,1.e-6f);
    EXPECT_NEAR(commands[14].steering_target_rad,
        1.2f*.01f*14.f*9.f/29.f,1.e-6f);
    EXPECT_FLOAT_EQ(commands[17].throttle_command,.12f);
    EXPECT_FALSE(f1tenth_mpc::geometric_feasibility_controls(
        feedforward,1.2f,0.f,.5235987f,commands));
    feedforward[3]=INFINITY;
    EXPECT_FALSE(f1tenth_mpc::geometric_feasibility_controls(
        feedforward,1.2f,.12f,.5235987f,commands));
}

TEST(MpcViabilityRecovery, BlendRetainsFastNominalWithoutBypassingBounds)
{
    MpcModelControl_t geo[PREDICTION_HORIZON]{},
        nominal[PREDICTION_HORIZON]{},mixed[PREDICTION_HORIZON]{};
    for(int i=0;i<PREDICTION_HORIZON;++i) {
        geo[i].steering_target_rad=.1f;
        geo[i].throttle_command=.06f;
        nominal[i].steering_target_rad=.3f;
        nominal[i].throttle_command=.32f;
    }
    ASSERT_TRUE(f1tenth_mpc::blended_viability_controls(
        geo,nominal,PREDICTION_HORIZON,.75f,.5235987f,mixed));
    EXPECT_NEAR(mixed[0].steering_target_rad,.25f,1.e-6f);
    EXPECT_NEAR(mixed[15].throttle_command,.255f,1.e-6f);
    EXPECT_FALSE(f1tenth_mpc::blended_viability_controls(
        geo,nominal,PREDICTION_HORIZON,1.1f,.5235987f,mixed));
    nominal[0].throttle_command=1.05f;
    EXPECT_FALSE(f1tenth_mpc::blended_viability_controls(
        geo,nominal,PREDICTION_HORIZON,.75f,.5235987f,mixed));
}
