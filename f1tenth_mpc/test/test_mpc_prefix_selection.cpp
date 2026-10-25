#include "mpc_prefix_selection.hpp"

#include <gtest/gtest.h>

namespace {

using f1tenth_mpc::select_validated_prefix;
using f1tenth_mpc::select_validated_recovery_prefix;
using f1tenth_mpc::iteration_limited_prefix_meets_residual_quality;
using f1tenth_mpc::compose_candidate_prefix_warm_start;
using f1tenth_mpc::candidate_prefix_yaw_direction_conflicts;

struct TestControl {
  int id{};
};

TEST(MpcPrefixSelection, CandidatePrefixNeverSeedsRejectedCandidateTail)
{
  const TestControl candidate[] = {{100}, {101}, {102}};
  const TestControl retained[] = {
      {200}, {201}, {202}, {203}, {204}, {205}, {206}, {207}};
  TestControl next[6]{};

  const int count = compose_candidate_prefix_warm_start(
      candidate, 3, retained, 8, 2, 6, next);

  ASSERT_EQ(count, 6);
  EXPECT_EQ(next[0].id, 100);
  EXPECT_EQ(next[1].id, 101);
  EXPECT_EQ(next[2].id, 102);
  EXPECT_EQ(next[3].id, 205);
  EXPECT_EQ(next[4].id, 206);
  EXPECT_EQ(next[5].id, 207);
}

TEST(MpcPrefixSelection, CandidatePrefixWithoutCertifiedTailIsPrefixOnly)
{
  const TestControl candidate[] = {{100}, {101}, {102}};
  TestControl next[6]{};

  const int count = compose_candidate_prefix_warm_start(
      candidate, 3, static_cast<const TestControl *>(nullptr), 0, 0, 6, next);

  EXPECT_EQ(count, 3);
  EXPECT_EQ(next[0].id, 100);
  EXPECT_EQ(next[1].id, 101);
  EXPECT_EQ(next[2].id, 102);
}

TEST(MpcPrefixSelection, UsesObjectiveWhenClearanceReserveIsNotThreatened)
{
  const auto choice = select_validated_prefix(
      true, 10.0, 0.30, true, 11.0, 0.32, 0.05, 0.005);
  EXPECT_TRUE(choice.choose_candidate);
  EXPECT_FALSE(choice.clearance_override);
}

TEST(MpcPrefixSelection, ReservesClearanceWhenCandidateImprovesIt)
{
  const auto choice = select_validated_prefix(
      true, 20.0, 0.0597, true, 10.0, 0.0499, 0.05, 0.005);
  EXPECT_TRUE(choice.choose_candidate);
  EXPECT_TRUE(choice.clearance_override);
}

TEST(MpcPrefixSelection, DoesNotOverrideForNumericalClearanceGain)
{
  const auto choice = select_validated_prefix(
      true, 12.0, 0.053, true, 10.0, 0.049, 0.05, 0.005);
  EXPECT_FALSE(choice.choose_candidate);
  EXPECT_FALSE(choice.clearance_override);
}

TEST(MpcPrefixSelection, KeepsObjectiveChoiceIfCandidateDoesNotImproveSlack)
{
  const auto choice = select_validated_prefix(
      true, 12.0, 0.025, true, 11.0, 0.030, 0.05, 0.005);
  EXPECT_FALSE(choice.choose_candidate);
  EXPECT_FALSE(choice.clearance_override);
}

TEST(MpcPrefixSelection, SelectsOnlyExactValidPrefix)
{
  EXPECT_TRUE(select_validated_prefix(
      true, 100.0, 0.01, false, 0.0, 0.0, 0.05, 0.005).choose_candidate);
  EXPECT_FALSE(select_validated_prefix(
      false, 0.0, 0.0, true, 100.0, 0.01, 0.05, 0.005).choose_candidate);
}

TEST(MpcPrefixSelection, StrongTurnKeepsPrefixThatRotatesIntoTurn)
{
  /* Captured at 110.403 m: both prefixes passed exact plant/corridor checks.
   * The candidate's terminal r=+0.453 rad/s opposed the -2.75 rad/s path
   * reference; the retained prefix ended at r=-1.907 rad/s. The next solve
   * rejected the candidate branch and accepted the retained branch. */
  EXPECT_TRUE(candidate_prefix_yaw_direction_conflicts(
      -2.75, 0.453, -1.907));
}

TEST(MpcPrefixSelection, DoesNotGuardWeakTurnsOrAlignedRetainedPrefix)
{
  EXPECT_FALSE(candidate_prefix_yaw_direction_conflicts(
      -0.8, 0.453, -1.907));
  EXPECT_FALSE(candidate_prefix_yaw_direction_conflicts(
      -2.75, 0.453, 0.0));
  EXPECT_FALSE(candidate_prefix_yaw_direction_conflicts(
      -2.75, -0.453, -1.907));
  EXPECT_FALSE(candidate_prefix_yaw_direction_conflicts(
      NAN, 0.453, -1.907));
}

TEST(MpcPrefixSelection, NonFiniteObjectivesUseSlackOutsideReserve)
{
  const auto choice = select_validated_prefix(
      true, NAN, 0.20, true, NAN, 0.10, 0.05, 0.005);
  EXPECT_TRUE(choice.choose_candidate);
  EXPECT_FALSE(choice.clearance_override);
}

TEST(MpcPrefixSelection, IterationLimitedCandidateMustBeatRecoveryMeaningfully)
{
  const auto choice = select_validated_recovery_prefix(
      true, 5159.8, 0.0575, false, true,
      true, 5209.2, 0.0600, 0.02, 0.05, 0.005);
  EXPECT_FALSE(choice.choose_candidate);
}

TEST(MpcPrefixSelection, IterationLimitedCandidateMayWinByClearance)
{
  const auto choice = select_validated_recovery_prefix(
      true, 110.0, 0.061, false, true,
      true, 100.0, 0.049, 0.02, 0.05, 0.005);
  EXPECT_TRUE(choice.choose_candidate);
  EXPECT_TRUE(choice.clearance_override);
}

TEST(MpcPrefixSelection, ConvergedCandidateUsesNormalObjectiveOrdering)
{
  const auto choice = select_validated_recovery_prefix(
      true, 100.0, 0.01, true, true,
      true, 101.0, 0.20, 0.02, 0.05, 0.005);
  EXPECT_TRUE(choice.choose_candidate);
  EXPECT_FALSE(choice.clearance_override);
}

TEST(MpcPrefixSelection, IterationLimitedCandidateMayWinWithMeaningfulCostGain)
{
  const auto choice = select_validated_recovery_prefix(
      true, 95.0, 0.20, false, true,
      true, 100.0, 0.20, 0.02, 0.05, 0.005);
  EXPECT_TRUE(choice.choose_candidate);
  EXPECT_FALSE(choice.clearance_override);
}

TEST(MpcPrefixSelection, ExactValidatedCandidateCanStandAloneWhenPriorIsInvalid)
{
  /* The solver residual belongs to the full QP iterate. The three-command
   * prefix is independently validated through the nonlinear plant and is the
   * only exact-valid action when no retained prefix remains. */
  EXPECT_TRUE(select_validated_recovery_prefix(
      true, 100.0, 0.10, true, true,
      false, 0.0, 0.0, 0.02, 0.05, 0.005).choose_candidate);
  EXPECT_TRUE(select_validated_recovery_prefix(
      true, 100.0, 0.10, false, true,
      false, 0.0, 0.0, 0.02, 0.05, 0.005).choose_candidate);
  EXPECT_TRUE(select_validated_recovery_prefix(
      true, 15577.2881, 0.00243711937, false, false,
      false, 0.0, 0.0, 0.02, 0.05, 0.005).choose_candidate);
}

TEST(MpcPrefixSelection, StandaloneExactPrefixRequiresFiniteCostAndSlack)
{
  EXPECT_FALSE(select_validated_recovery_prefix(
      true, NAN, 0.10, false, false,
      false, 0.0, 0.0, 0.02, 0.05, 0.005).choose_candidate);
  EXPECT_FALSE(select_validated_recovery_prefix(
      true, 100.0, -1.0e-6, false, false,
      false, 0.0, 0.0, 0.02, 0.05, 0.005).choose_candidate);
}

TEST(MpcPrefixSelection, V300HighResidualThrottleSpikeKeepsExactRecoveryPrefix)
{
  /* Captured at 31.917 m: the candidate asked for full throttle while the
   * car was already above reference. Both 75 ms rollouts passed the corridor,
   * but the source-plant objective was 67,875 versus 4,520 for the retained
   * converged prefix, with only 0.3 mm clearance difference. */
  const auto choice = select_validated_recovery_prefix(
      true, 67875.4, 0.0230, false, false,
      true, 4520.4, 0.0228, 0.02, 0.05, 0.005);
  EXPECT_FALSE(choice.choose_candidate);
  EXPECT_FALSE(choice.clearance_override);

  EXPECT_FALSE(iteration_limited_prefix_meets_residual_quality(
      0.70468688, 0.135472074, 0.02, 2.0));
}

TEST(MpcPrefixSelection, PoorResidualCannotReplacePriorOnSmallClearanceGain)
{
  /* Captured at 40.787 m in the live v300 attempt: the 100-iteration
   * candidate had primal residual 0.233 (> 2x 0.02 tolerance), scored 7,121
   * vs the retained plan's 7,733, and improved 75 ms clearance by only 9.9 mm
   * (2.62 cm vs 1.64 cm). It requested -0.524 rad steering versus -0.050 rad
   * on the retained sequence; the next cycle had no valid prefix and collided.
   * Exact short-horizon corridor validity alone must not authorize it.
   */
  const auto choice = select_validated_recovery_prefix(
      true, 7121.38916, 0.0262431242, false, false,
      true, 7732.78662, 0.0163606443, 0.02, 0.05, 0.005);
  EXPECT_FALSE(choice.choose_candidate);
  EXPECT_FALSE(choice.clearance_override);
}

TEST(MpcPrefixSelection, PoorResidualMayRefreshPlanOnLargeExactCostGain)
{
  /* Captured on the practice-track refinement-reserve run near 154.6 m:
   * both 75 ms source-plant prefixes were exact-valid; the candidate's exact
   * cost was 3,039 vs 9,719 for the retained prefix, while losing only 1 mm
   * of corridor slack. The old residual-only gate kept consuming the retained
   * plan until its final action, then had no prefix left to publish. */
  const auto choice = select_validated_recovery_prefix(
      true, 3038.7, 0.1744, false, false,
      true, 9719.1, 0.1755, 0.02, 0.05, 0.005);
  EXPECT_TRUE(choice.choose_candidate);
  EXPECT_FALSE(choice.clearance_override);
}

TEST(MpcPrefixSelection, StaleRefreshCannotReplacePriorWithMuchWorseCandidate)
{
  /* Captured v323 failure: five exact-valid candidate prefixes were selected
   * despite costing 5.5-8x as much as the retained prefix; their controls
   * reversed steering and cross-track error grew. Corridor validity alone is
   * not sufficient to refresh an iteration-limited plan. */
  const auto choice = select_validated_recovery_prefix(
      true, 4949.40918, 0.334675789, false, false,
      true, 1048.42334, 0.347337842, 0.02, 0.05, 0.005,
      3, 3, 0.02);
  EXPECT_FALSE(choice.choose_candidate);
  EXPECT_FALSE(choice.clearance_override);
  EXPECT_FALSE(choice.stale_prior_refresh);
}

TEST(MpcPrefixSelection, StaleRefreshAllowsNonWorseExactObjective)
{
  const auto choice = select_validated_recovery_prefix(
      true, 950.0, 0.30, false, false,
      true, 1000.0, 0.31, 0.02, 0.05, 0.005,
      3, 3, 0.02);
  EXPECT_TRUE(choice.choose_candidate);
  EXPECT_FALSE(choice.clearance_override);
  EXPECT_TRUE(choice.stale_prior_refresh);
}

TEST(MpcPrefixSelection, StalePriorRefreshRequiresChainAndClearance)
{
  const auto chain_too_short = select_validated_recovery_prefix(
      true, 950.0, 0.30, false, false,
      true, 1000.0, 0.31, 0.02, 0.05, 0.005,
      2, 3, 0.02);
  EXPECT_FALSE(chain_too_short.choose_candidate);
  EXPECT_FALSE(chain_too_short.stale_prior_refresh);

  const auto candidate_too_close_to_wall = select_validated_recovery_prefix(
      true, 950.0, 0.049, false, false,
      true, 1000.0, 0.31, 0.02, 0.05, 0.005,
      4, 3, 0.02);
  EXPECT_FALSE(candidate_too_close_to_wall.choose_candidate);
  EXPECT_FALSE(candidate_too_close_to_wall.stale_prior_refresh);

  const auto clearance_loss_too_large = select_validated_recovery_prefix(
      true, 950.0, 0.20, false, false,
      true, 1000.0, 0.31, 0.02, 0.05, 0.005,
      4, 3, 0.02);
  EXPECT_FALSE(clearance_loss_too_large.choose_candidate);
  EXPECT_FALSE(clearance_loss_too_large.stale_prior_refresh);
}

TEST(MpcPrefixSelection, PoorResidualStillRejectsSmallCostWinOrClearanceLoss)
{
  /* The prior V300 failure case improved exact cost by only 7.9%; it remains
   * rejected even though its short prefix had slightly more slack. */
  EXPECT_FALSE(select_validated_recovery_prefix(
      true, 7121.4, 0.0262, false, false,
      true, 7732.8, 0.0164, 0.02, 0.05, 0.005).choose_candidate);

  /* A >10% cost improvement is not enough if it materially erodes the
   * already-certified prefix's corridor clearance. */
  EXPECT_FALSE(select_validated_recovery_prefix(
      true, 8000.0, 0.155, false, false,
      true, 10000.0, 0.200, 0.02, 0.05, 0.005).choose_candidate);
}

TEST(MpcPrefixSelection, ResidualGateUsesConfiguredTolerance)
{
  EXPECT_TRUE(iteration_limited_prefix_meets_residual_quality(
      0.035, 0.039, 0.02, 2.0));
  EXPECT_FALSE(iteration_limited_prefix_meets_residual_quality(
      0.041, 0.02, 0.02, 2.0));
  EXPECT_FALSE(iteration_limited_prefix_meets_residual_quality(
      NAN, 0.02, 0.02, 2.0));
}

}  // namespace
