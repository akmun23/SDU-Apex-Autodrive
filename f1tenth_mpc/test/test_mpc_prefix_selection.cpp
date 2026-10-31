#include "mpc_prefix_selection.hpp"

#include <gtest/gtest.h>

namespace {

using f1tenth_mpc::select_validated_prefix;
using f1tenth_mpc::select_validated_recovery_prefix;
using f1tenth_mpc::iteration_limited_prefix_meets_residual_quality;
using f1tenth_mpc::compose_candidate_prefix_warm_start;
using f1tenth_mpc::copy_retained_plan_warm_start;
using f1tenth_mpc::candidate_prefix_yaw_direction_conflicts;
using f1tenth_mpc::prefix_control_changes_bounded;

struct TestControl {
  int id{};
  double steering_target_rad{};
  double throttle_command{};
};

TEST(MpcPrefixSelection, MaterialImprovementSupportsSignedObjectives)
{
  using f1tenth_mpc::objective_improves_materially;
  EXPECT_TRUE(objective_improves_materially(90.0, 100.0, 0.1));
  EXPECT_FALSE(objective_improves_materially(95.0, 100.0, 0.1));
  EXPECT_TRUE(objective_improves_materially(-110.0, -100.0, 0.1));
  EXPECT_FALSE(objective_improves_materially(-105.0, -100.0, 0.1));
  EXPECT_FALSE(objective_improves_materially(-95.0, -100.0, 0.1));
  EXPECT_TRUE(objective_improves_materially(-1.0, 0.0, 0.1));
  EXPECT_FALSE(objective_improves_materially(0.0, 0.0, 0.1));
  EXPECT_FALSE(objective_improves_materially(NAN, -100.0, 0.1));
  EXPECT_TRUE(select_validated_recovery_prefix(
      true, -115.0, 0.1, false, true, true, -100.0, 0.1,
      0.1, 0.05, 0.005).choose_candidate);
}

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

TEST(MpcPrefixSelection, SingleExactValidatedActionCanSeedNextReplan)
{
  const TestControl candidate[] = {{100}};
  TestControl next[6]{};

  const int count = compose_candidate_prefix_warm_start(
      candidate, 1, static_cast<const TestControl *>(nullptr), 0, 0, 6, next);

  ASSERT_EQ(count, 1);
  EXPECT_EQ(next[0].id, 100);
}

TEST(MpcPrefixSelection, ExactValidatedShortCandidateCanRecoverWithoutPrior)
{
  /* A finite one-step source-plant-valid action may be issued after both
   * longer candidate and retained prefixes fail. The next control tick must
   * replan; this does not certify the rejected horizon tail. */
  const auto choice = select_validated_recovery_prefix(
      true, 4.0, 0.012, false, false,
      false, 0.0, 0.0, 0.02, 0.05, 0.005);
  EXPECT_TRUE(choice.choose_candidate);
}

TEST(MpcPrefixSelection, RetainedSuffixIncludesIssuedActionWithoutPadding)
{
  const TestControl plan[] = {{0}, {1}, {2}, {3}};
  TestControl next[6]{};
  ASSERT_EQ(copy_retained_plan_warm_start(plan, 4, 2, 6, next), 2);
  EXPECT_EQ(next[0].id, 2);  // Already issued: next cycle shifts it once.
  EXPECT_EQ(next[1].id, 3);
  EXPECT_EQ(next[2].id, 0);  // Not a fabricated repeat of action 3.
  EXPECT_EQ(copy_retained_plan_warm_start(plan, 4, 4, 6, next), 0);
}

TEST(MpcPrefixSelection, CandidateSuffixDoesNotFabricateExpiredPlanActions)
{
  const TestControl prefix[] = {{100}, {101}, {102}};
  const TestControl plan[] = {{0}, {1}, {2}, {3}, {4}, {5}};
  TestControl next[8]{};
  ASSERT_EQ(compose_candidate_prefix_warm_start(
      prefix, 3, plan, 6, 2, 8, next), 4);
  EXPECT_EQ(next[2].id, 102);
  EXPECT_EQ(next[3].id, 5);
  EXPECT_EQ(next[4].id, 0);
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

TEST(MpcPrefixSelection, PoorResidualCannotSuppressExactValidatedClearanceGain)
{
  /* Reproduced from the IROS 7.125 s trial: both 75 ms prefixes pass the exact
   * source-plant corridor check, but the prior has only 4.42 cm slack while
   * the candidate has 7.99 cm. The candidate's cost is higher and ADMM did not
   * meet residual quality; choosing the prior starts a long stale-plan chain. */
  const auto choice = select_validated_recovery_prefix(
      true, 123.178444, 0.0798761, false, false,
      true, 120.830353, 0.0442098, 0.02, 0.05, 0.005);
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

TEST(MpcPrefixSelection, InvalidCandidateLeavesValidatedPriorEligible)
{
  /* A rejected/poor-residual iterate is not a reason to discard a retained
   * plan. The controller must exact-revalidate that prior prefix separately;
   * selector=false means keep the prior when prior_valid is true. */
  const auto choice = select_validated_recovery_prefix(
      false, NAN, NAN, false, false,
      true, 100.0, 0.04, 0.02, 0.05, 0.005);
  EXPECT_FALSE(choice.choose_candidate);
  EXPECT_FALSE(choice.clearance_override);
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
      true, 7732.78662, 0.0163606443, 0.02, 0.05, 0.02);
  EXPECT_FALSE(choice.choose_candidate);
  EXPECT_FALSE(choice.clearance_override);
}

TEST(MpcPrefixSelection, NearBoundaryAcceptsSmallExactGainWithSmoothControls)
{
  const TestControl candidate[] = {
      {0, 0.13904, 0.05504}, {1, 0.14406, 0.24890},
      {2, 0.15997, 0.33613}};
  const TestControl prior[] = {
      {3, 0.13904, 0.23879}, {4, 0.14406, 0.25629},
      {5, 0.16171, 0.24159}};
  ASSERT_TRUE(prefix_control_changes_bounded(
      candidate, prior, 3, 0.10, 0.20));

  /* Captured at IROS s=76.534 m in a17: both exact prefixes were valid, but
   * the prior had only 0.25 mm extra corridor while the candidate had 5.91 mm.
   * Candidate cost was 1.45x the retained prefix; its steering sequence was
   * essentially unchanged and it reduced throttle rather than reversing rack.
   */
  const auto choice = select_validated_recovery_prefix(
      true, 19.1675014, 0.00590585172, false, false,
      true, 13.2032776, 0.000246345997, 0.02, 0.05, 0.02,
      0, 3, 0.02, 0.30, 0.005, 1.50, true);
  EXPECT_TRUE(choice.choose_candidate);
  EXPECT_TRUE(choice.clearance_override);
  EXPECT_TRUE(choice.near_boundary_override);
}

TEST(MpcPrefixSelection, ExactSaferPrefixAllowsMeasuredThrottleDifference)
{
  /* IROS a63 at s=204.286 m: both 75 ms prefixes passed exact nonlinear
   * validation. The candidate improved slack by 14.3 mm and objective by
   * 8.4%, but its throttle differed by 0.2145, just above the old 0.20
   * selector-only cap. A 0.22 cap admits this measured deceleration while
   * retaining the exact corridor gate and a bounded command-change check. */
  const TestControl candidate[] = {
      {0, 0.185937822, 0.125826895},
      {1, 0.175750256, 0.143172622},
      {2, 0.0990049466, 0.207327500}};
  const TestControl prior[] = {
      {3, 0.145937830, 0.340312451},
      {4, 0.0957502648, 0.343172610},
      {5, 0.139004946, 0.283786476}};
  EXPECT_FALSE(prefix_control_changes_bounded(
      candidate, prior, 3, 0.10, 0.20));
  ASSERT_TRUE(prefix_control_changes_bounded(
      candidate, prior, 3, 0.10, 0.22));

  const auto choice = select_validated_recovery_prefix(
      true, 66.9457474, 0.0371378511, false, false,
      true, 73.1083221, 0.0228424817, 0.02, 0.10, 0.02,
      0, 3, 0.02, 0.30, 0.002, 1.50, true);
  EXPECT_TRUE(choice.choose_candidate);
  EXPECT_TRUE(choice.near_boundary_override);
}

TEST(MpcPrefixSelection, NearBoundaryOverrideAlsoAppliesToConvergedSolve)
{
  /* The a17 capture was a converged SQP/ADMM result. The former early return
   * for converged candidates bypassed this policy, silently disabling the
   * configured near-boundary reserve for the normal case. */
  const auto choice = select_validated_recovery_prefix(
      true, 19.1675014, 0.00590585172, true, true,
      true, 13.2032776, 0.000246345997, 0.02, 0.05, 0.02,
      0, 3, 0.02, 0.30, 0.005, 1.50, true);
  EXPECT_TRUE(choice.choose_candidate);
  EXPECT_TRUE(choice.near_boundary_override);
  EXPECT_FALSE(choice.stale_prior_refresh);
}

TEST(MpcPrefixSelection, A65SoftReserveEnablesExactValidatedSaferPrefix)
{
  /* Captured at IROS progress 209.062 m in a65. Both three-command prefixes
   * passed exact plant/corridor validation. The candidate reduced exact
   * rollout cost (94.81 vs 96.73), gained 12.5 mm of corridor slack
   * (127.9 vs 115.4 mm), and its first controls stayed within configured
   * bounds. With the selector reserve accidentally set to zero, the
   * near-boundary override could never activate and the controller retained
   * the less-safe plan. A 0.15 m soft selector reserve enables this measured
   * choice; the hard nonlinear corridor gate remains unchanged. */
  const auto choice = select_validated_recovery_prefix(
      true, 94.8087387, 0.127883807, false, false,
      true, 96.726799, 0.115420997, 0.02, 0.15, 0.02,
      3, 3, 0.02, 0.30, 0.002, 1.50, true);
  EXPECT_TRUE(choice.choose_candidate);
  EXPECT_TRUE(choice.near_boundary_override);
  EXPECT_TRUE(choice.clearance_override);
}

TEST(MpcPrefixSelection, A69SoftReserveRefreshesDriftingExactPrior)
{
  /* At IROS progress 117.876 m in a69, both 75 ms prefixes were exact-valid
   * and their first three controls stayed within the selector's change box.
   * The fresh candidate cost 19.496 vs 17.014 for the retained plan, but
   * improved minimum wall slack from 0.2414 to 0.2499 m. The 0.15 m reserve
   * did not activate because the retained prefix still had >0.15 m slack;
   * increasing the validation-only reserve to 0.30 m should prefer this
   * modestly higher-cost, safer replan before the retained-plan chain grows.
   * Exact hard-corridor checks remain mandatory for both choices. */
  const auto choice = select_validated_recovery_prefix(
      true, 19.4965038, 0.249897897, false, false,
      true, 17.014307, 0.241380274, 0.02, 0.30, 0.02,
      4, 3, 0.02, 0.30, 0.002, 1.50, true);
  EXPECT_TRUE(choice.choose_candidate);
  EXPECT_TRUE(choice.near_boundary_override);
  EXPECT_TRUE(choice.clearance_override);
}

TEST(MpcPrefixSelection, NearBoundaryDoesNotAuthorizeSteeringReversal)
{
  const TestControl candidate[] = {
      {0, -0.524, 0.30}, {1, -0.400, 0.32}, {2, -0.300, 0.35}};
  const TestControl prior[] = {
      {3, -0.050, 0.30}, {4, -0.060, 0.31}, {5, -0.070, 0.32}};
  EXPECT_FALSE(prefix_control_changes_bounded(
      candidate, prior, 3, 0.10, 0.20));

  const auto choice = select_validated_recovery_prefix(
      true, 7121.38916, 0.0262431242, false, false,
      true, 7732.78662, 0.0163606443, 0.02, 0.05, 0.02,
      0, 3, 0.02, 0.30, 0.005, 1.50, false);
  EXPECT_FALSE(choice.choose_candidate);
  EXPECT_FALSE(choice.near_boundary_override);
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
      3, 3, 0.02, 0.30, 0.0, 1.5, true);
  EXPECT_TRUE(choice.choose_candidate);
  EXPECT_FALSE(choice.clearance_override);
  EXPECT_TRUE(choice.stale_prior_refresh);
}

TEST(MpcPrefixSelection, StaleChainMayRefreshWithBoundedCostAndClearanceLoss)
{
  /* a10 at 117.27 m: after 8 retained actions, exact prefixes had near-equal
   * clearance and cost (2.32 vs 2.26). This selector may refresh and break
   * the chain without needing a large clearance gain. */
  const auto choice = select_validated_recovery_prefix(
      true, 2.32, 0.076, false, false,
      true, 2.26, 0.076, 0.02, 0.05, 0.02,
      8, 3, 0.02, 0.30, 0.0, 1.5, true);
  EXPECT_TRUE(choice.choose_candidate);
  EXPECT_FALSE(choice.clearance_override);
  EXPECT_TRUE(choice.stale_prior_refresh);
}

TEST(MpcPrefixSelection, StaleChainCanTradeClearanceWithinBoundedLoss)
{
  const auto choice = select_validated_recovery_prefix(
      true, 42.43, 0.321, false, false,
      true, 33.56, 0.337, 0.02, 0.05, 0.02,
      17, 3, 0.02, 0.30, 0.0, 1.5, true);
  EXPECT_TRUE(choice.choose_candidate);
  EXPECT_TRUE(choice.stale_prior_refresh);
}

TEST(MpcPrefixSelection, ExactValidatedStaleRefreshUsesOnlyHardCorridorReserve)
{
  /* IROS 7.125 s run a50 at 289.751 m: candidate/prior slack was
   * 6.77/13.63 mm, objective 50.33/41.70, and the retained-plan chain was 15.
   * Both exact prefixes remained valid under the hard corridor, which already
   * includes the mandated 5 cm vehicle-to-wall margin. Requiring an additional
   * 5 cm quality reserve retained the old sequence until it became invalid. */
  const auto choice = select_validated_recovery_prefix(
      true, 50.3341484, 0.0067718327, false, false,
      true, 41.696476, 0.0136312544, 0.02, 0.0, 0.002,
      15, 3, 0.02, 0.30, 0.002, 1.5, true);
  EXPECT_TRUE(choice.choose_candidate);
  EXPECT_TRUE(choice.stale_prior_refresh);
  EXPECT_FALSE(choice.clearance_override);
}

TEST(MpcPrefixSelection, IrosNearWallCandidateWithFourteenMillimeterGainWins)
{
  /* IROS 2026-10-30, witness 7a9e6afbf50d666f at 161.46 m:
   * candidate/prior 75 ms slacks were 33.03/19.08 mm, with exact objectives
   * 44.93/49.24 and six prior-plan actions. Both prefixes passed the hard
   * nonlinear corridor check, but the 20 mm preference threshold retained
   * the less-safe plan; it then became invalid on the next cycle. */
  const auto choice = select_validated_recovery_prefix(
      true, 44.933342, 0.0330274552, false, false,
      true, 49.2412224, 0.0190804303, 0.02, 0.05, 0.01,
      6, 3, 0.02, 0.30);
  EXPECT_TRUE(choice.choose_candidate);
  EXPECT_TRUE(choice.clearance_override);
  EXPECT_FALSE(choice.stale_prior_refresh);
}

TEST(MpcPrefixSelection, StaleChainRejectsLargeCostIncrease)
{
  const auto choice = select_validated_recovery_prefix(
      true, 4949.40918, 0.334675789, false, false,
      true, 1048.42334, 0.347337842, 0.02, 0.05, 0.02,
      8, 3, 0.02, 0.30);
  EXPECT_FALSE(choice.choose_candidate);
  EXPECT_FALSE(choice.stale_prior_refresh);
}

TEST(MpcPrefixSelection, StaleChainRefreshesForFiveMillimeterGain)
{
  /* a08 accumulated 9 prior actions before an exact-valid candidate improved
   * 75 ms slack by 7.5 mm at a 29.8% exact-cost increase. */
  const auto choice = select_validated_recovery_prefix(
      true, 16.3531685, 0.203892231, false, false,
      true, 12.6028471, 0.196427286, 0.02, 0.05, 0.02,
      9, 3, 0.02, 0.30, 0.0, 1.5, true);
  EXPECT_TRUE(choice.choose_candidate);
  EXPECT_TRUE(choice.stale_prior_refresh);
}

TEST(MpcPrefixSelection, ConvergedSolveUsesNormalExactObjectiveOrdering)
{
  /* A converged candidate is compared by its exact-prefix objective and
   * corridor rule; the stale-plan allowance is only for iteration-limited
   * candidates, whose solver result did not converge. */
  const auto choice = select_validated_recovery_prefix(
      true, 8.37733459, 0.500848472, true, true,
      true, 7.99494505, 0.490914971, 0.02, 0.05, 0.02,
      4, 3, 0.02, 0.30, 0.0, 1.5, true);
  EXPECT_FALSE(choice.choose_candidate);
  EXPECT_FALSE(choice.clearance_override);
  EXPECT_FALSE(choice.stale_prior_refresh);
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
      true, 7732.8, 0.0164, 0.02, 0.05, 0.02).choose_candidate);

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
