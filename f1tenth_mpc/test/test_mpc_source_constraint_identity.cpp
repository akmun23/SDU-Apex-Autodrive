#include "mpc_source_exact_nlp.hpp"
#include "vehicle_model.h"
#include <gtest/gtest.h>
#include <cmath>
#include <memory>

namespace sdu_apex::source_nlp {
namespace {
TEST(SourceConstraintIdentity, SameRowCountDifferentPhysicalRowMustReset) {
    const std::vector<SparseNonlinearRow> a{
        {NonlinearRowType::BodyCorner,10,1,-1,-1,.05},
        {NonlinearRowType::OriginalSlipSubstep,11,2,8,1,.03},
        {NonlinearRowType::PhysicalCommandRate,12,0,-1,-1,.2}
    };
    auto same=a;
    // The exact source inequality changes with the current state, so
    // changing only its margin must NOT invalidate the dual row identity.
    same[0].value=-.03;
    same[1].value=.11;
    EXPECT_TRUE(same_original_source_constraint_layout(a,same));
    // Same number of rows, but the active source contact/substep differs.
    auto branch=a;
    branch[1].source_fixed_tick=9;
    EXPECT_FALSE(same_original_source_constraint_layout(a,branch));
    branch=a;
    branch[1].source_vehicle_substep=2;
    EXPECT_FALSE(same_original_source_constraint_layout(a,branch));
    // Same count, entirely different physics row.
    branch=a;
    branch[2].type=NonlinearRowType::PhysicalCommandBox;
    EXPECT_FALSE(same_original_source_constraint_layout(a,branch));
    branch=a;
    branch[0].index=2;
    EXPECT_FALSE(same_original_source_constraint_layout(a,branch));
}
TEST(SourceConstraintIdentity, ExtraOrMissingOneSidedReferenceCutMustReset) {
    std::vector<SparseNonlinearRow> nominal{
        {NonlinearRowType::TerminalSpeedViability,29,0,-1,-1,.01}
    };
    auto both_sides=nominal;
    both_sides.push_back(
        {NonlinearRowType::TerminalSpeedViability,29,1,-1,-1,.01});
    EXPECT_FALSE(same_original_source_constraint_layout(nominal,both_sides));
    EXPECT_FALSE(same_original_source_constraint_layout(both_sides,nominal));
    EXPECT_TRUE(same_original_source_constraint_layout(nominal,nominal));
}
TEST(SourceTrackingCurvature, MatchesImplicitProjectionFiniteDifferences) {
    MpcTrajectorySample_t track[4]{};
    for(int i=0;i<4;++i) {
        track[i].s=i;track[i].x=i;track[i].y=.05*i*i;
        track[i].heading=.2*i;track[i].left_bound=track[i].right_bound=1.;
    }
    const double first[6]={.02,-.1,.3,-.04,.08,.2};
    const double second[6]={.01,.03,-.02,.015,-.005,.04};
    const float body[3]={.1f,.005f,.02f};
    double progress=0.;float ey=0.,epsi=0.;float jacobian[3][6]{};
    ASSERT_TRUE(mpc_trajectory_advance_tracking_with_progress(track,4,5.,.4,.03f,.08f,
        body[0],body[1],body[2],.5,&progress,&ey,&epsi,jacobian));
    double d[3]{},dd[3]{};int knots=0;
    ASSERT_TRUE(mpc_trajectory_tracking_path_curvature(track,4,5.,.4,.03f,.08f,body,
        progress,ey,first,second,d,dd,&knots));
    EXPECT_EQ(knots,0);
    for(int row=0;row<3;++row) {
        double expected=0.;for(int j=0;j<6;++j)expected+=jacobian[row][j]*first[j];
        EXPECT_NEAR(d[row],expected,1.e-6);
    }
    const double h=.02;
    double values[2][3]{};
    for(int side=0;side<2;++side) {
        const double t=side?h:-h;
        double args[6]={.03f,.08f,body[0],body[1],body[2],.4};
        for(int j=0;j<6;++j)args[j]+=t*first[j]+.5*t*t*second[j];
        double s=0.;float y=0.,psi=0.;
        ASSERT_TRUE(mpc_trajectory_advance_tracking_with_progress(track,4,5.,args[5],args[0],args[1],
            args[2],args[3],args[4],.5,&s,&y,&psi,nullptr));
        values[side][0]=y;values[side][1]=psi;values[side][2]=s;
    }
    const double base[3]={ey,epsi,progress};
    for(int row=0;row<3;++row) {
        EXPECT_NEAR((values[1][row]-values[0][row])/(2.*h),d[row],2.e-5);
        EXPECT_NEAR((values[1][row]-2.*base[row]+values[0][row])/(h*h),dd[row],2.e-3);
    }
}
TEST(SourceTrackingCurvature, ReferenceKnotUsesDirectionOfProgress) {
    MpcTrajectorySample_t track[4]{};
    for(int i=0;i<4;++i) {
        track[i].s=i;track[i].x=i;track[i].heading=i>1?.5*(i-1):0.;
        track[i].left_bound=track[i].right_bound=1.;
    }
    const float body[3]={.125f,0.f,0.f};
    double first[6]{},second[6]{},d[3]{},dd[3]{};int knots=0;
    first[2]=-.2;
    ASSERT_TRUE(mpc_trajectory_tracking_path_curvature(track,4,5.,.875,0.f,0.f,body,
        1.,0.f,first,second,d,dd,&knots));
    EXPECT_NEAR(d[1],0.,1.e-12);EXPECT_NEAR(d[2],-.2,1.e-12);
    EXPECT_EQ(knots,1);
    first[2]=.2;
    ASSERT_TRUE(mpc_trajectory_tracking_path_curvature(track,4,5.,.875,0.f,0.f,body,
        1.,0.f,first,second,d,dd,&knots));
    EXPECT_NEAR(d[1],-.1,1.e-12);EXPECT_NEAR(d[2],.2,1.e-12);
}
TEST(SourceTrackingCurvature, ConstantWidthsCannotHideTrackingSegmentSwitch) {
    MpcTrajectorySample_t track[4]{};
    for(int i=0;i<4;++i) {
        track[i].s=i;track[i].x=i;track[i].heading=.2*i;
        track[i].left_bound=track[i].right_bound=1.;
    }
    auto base=std::make_unique<MpcRtiNominal_t>();
    base->valid=1;base->horizon=kStages;
    MpcModelState_t plant{};plant.u=3.f;
    ASSERT_TRUE(vehicle_model_initialize_plant_state(&plant));
    for(int k=0;k<=kStages;++k) {
        base->states[k].plant=plant;base->progress[k]=.5;
        if(k<kStages)base->controls[k]={0.f,.2f};
    }
    auto candidate=std::make_unique<MpcRtiNominal_t>(*base);
    Evaluation score{};score.valid=1;
    SourceCurvatureBranches same{},switched{};
    ASSERT_TRUE(compare_source_curvature_branches(*base,score,*candidate,score,track,4,5.,.025f,&same));
    EXPECT_TRUE(same.same);
    candidate->progress[10]=1.5;
    ASSERT_TRUE(compare_source_curvature_branches(*base,score,*candidate,score,track,4,5.,.025f,&switched));
    EXPECT_FALSE(switched.same);
    EXPECT_EQ(switched.tracking_reference_changes,1);
    EXPECT_EQ(switched.reference_changes,1);
    candidate->progress[10]=5.5;
    ASSERT_TRUE(compare_source_curvature_branches(*base,score,*candidate,score,track,4,5.,.025f,&same));
    EXPECT_TRUE(same.same); // Periodic wrap itself is not a segment change.
}
} // namespace
} // namespace sdu_apex::source_nlp
