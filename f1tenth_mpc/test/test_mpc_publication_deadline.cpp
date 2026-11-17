#include "mpc_publication_deadline.hpp"

#include <gtest/gtest.h>

#include <limits>

TEST(MpcPublicationDeadline, ContinuationUsesElapsedStagesAndNeverExtendsThePlan)
{
  using f1tenth_mpc::mpc_plan_continuation_stage;
  constexpr int64_t plan = 1'000'000'000, period = 25'000'000;
  EXPECT_EQ(mpc_plan_continuation_stage(plan, plan, plan + period, period, 30, period - 1), -1);
  EXPECT_EQ(mpc_plan_continuation_stage(plan, plan, plan + period, period, 30, period), 1);
  EXPECT_EQ(mpc_plan_continuation_stage(plan, plan, plan + 4 * period, period, 30, period), 4);
  EXPECT_EQ(mpc_plan_continuation_stage(plan, plan, plan + period + 600'000, period, 30, period), 1);
  EXPECT_EQ(mpc_plan_continuation_stage(plan, plan + period, plan + period + 600'000, period, 30, period), -1);
  EXPECT_EQ(mpc_plan_continuation_stage(plan, plan, plan + 29 * period, period, 30, period), 29);
  EXPECT_EQ(mpc_plan_continuation_stage(plan, plan, plan + 30 * period, period, 30, period), -1);
  EXPECT_EQ(mpc_plan_continuation_stage(plan, plan - 1, plan + period, period, 30, period), -1);
  EXPECT_EQ(mpc_plan_continuation_stage(0, plan, plan + period, period, 30, period), -1);
  EXPECT_EQ(mpc_plan_continuation_stage(plan, plan, plan + period, 0, 30, period), -1);
  EXPECT_EQ(mpc_plan_continuation_stage(plan, plan + period, plan, period, 30, period), -1);
}

TEST(MpcPublicationDeadline, SolveBudgetEndsBeforeItsRequestWindowCloses)
{
  // A callback at phase 28 ms has only 20 ms, rather than 22 ms, to
  // complete validation and publication for the 50 ms request.
  EXPECT_DOUBLE_EQ(f1tenth_mpc::mpc_command_publication_deadline_us(
    22'000., 0., 1'028'000'000, 1'050'000'000, 2'000'000), 20'000.);
  EXPECT_DOUBLE_EQ(f1tenth_mpc::mpc_command_publication_deadline_us(
    22'000., 0., 1'032'000'000, 1'075'000'000, 2'000'000), 22'000.);
}

TEST(MpcPublicationDeadline, CommandMustEnterItsIntendedRequestWindow)
{
  using Gate = f1tenth_mpc::MpcCommandEpochGate;
  const auto gate = [](int64_t last, int64_t now) {
    return f1tenth_mpc::mpc_command_epoch_gate(
      1'075'000'000, last, now, 25'000'000, 2'000'000);
  };
  // A fast solve must wait rather than applying the 75 ms plan at 50 ms.
  EXPECT_EQ(gate(1'025'000'000, 1'047'000'000), Gate::Wait);
  EXPECT_EQ(gate(1'050'000'000, 1'051'000'000), Gate::Ready);
  // Preserve the window across measured sub-millisecond bridge jitter.
  EXPECT_EQ(gate(1'050'600'000, 1'051'000'000), Gate::Ready);
  // A late solve must defer rather than applying the 75 ms plan at 100 ms.
  EXPECT_EQ(gate(1'050'000'000, 1'073'000'000), Gate::Missed);
  EXPECT_EQ(gate(1'075'000'000, 1'076'000'000), Gate::Missed);
}

TEST(MpcPublicationDeadline, ChoosesAWindowReachableBeforeTheDeadline)
{
  int64_t epoch{};
  const auto choose = [&](int64_t callback, int64_t predicted) {
    EXPECT_TRUE(f1tenth_mpc::mpc_next_bridge_request_epoch(
      1'000'000'000, predicted, callback + 22'000'000,
      25'000'000, 2'000'000, &epoch));
    return epoch;
  };
  EXPECT_EQ(choose(1'032'000'000, 1'050'000'000), 1'075'000'000);
  // Waiting for the 50 ms request cannot fit a deadline at 50 ms. Use the
  // earlier window instead of starving every cycle with the same phase.
  EXPECT_EQ(choose(1'028'000'000, 1'050'000'000), 1'050'000'000);
  EXPECT_EQ(choose(1'027'000'000, 1'045'000'000), 1'050'000'000);
}

TEST(MpcPublicationDeadline, AcceptsOnlyStrictlyUnderBudget)
{
  EXPECT_TRUE(f1tenth_mpc::mpc_cycle_within_publication_deadline(19.999, 20.0));
  EXPECT_FALSE(f1tenth_mpc::mpc_cycle_within_publication_deadline(20.0, 20.0));
  EXPECT_FALSE(f1tenth_mpc::mpc_cycle_within_publication_deadline(20.001, 20.0));
}

TEST(MpcPublicationDeadline, AcceptsTwentyTwoMsButNotTheTwentyFiveMsControlPeriod)
{
  EXPECT_TRUE(f1tenth_mpc::mpc_cycle_within_publication_deadline(21.999, 22.0));
  EXPECT_FALSE(f1tenth_mpc::mpc_cycle_within_publication_deadline(22.0, 22.0));
  EXPECT_FALSE(f1tenth_mpc::mpc_cycle_within_publication_deadline(22.001, 22.0));
  EXPECT_FALSE(f1tenth_mpc::mpc_publication_budget_valid(25.0));
  int publications = 0;
  constexpr int64_t start_ns = 1'000'000'000;
  EXPECT_TRUE(f1tenth_mpc::mpc_publish_before_deadline(
    start_ns, 22.0, nullptr,
    [] { return int64_t{1'021'999'000}; },
    [&] { ++publications; }));
  EXPECT_FALSE(f1tenth_mpc::mpc_publish_before_deadline(
    start_ns, 22.0, nullptr,
    [] { return int64_t{1'022'000'000}; },
    [&] { ++publications; }));
  EXPECT_EQ(publications, 1);
}

TEST(MpcPublicationDeadline, PriorPlanCannotBypassTheSameGate)
{
  // The production publish wrapper is shared by normal, candidate-prefix,
  // prior-plan and open-loop commands. Exact nonlinear validity cannot waive
  // the callback-to-publication computation deadline.
  int publications = 0;
  double elapsed_ms = -1.0;
  constexpr int64_t callback_start_ns = 1'000'000'000;
  const bool published = f1tenth_mpc::mpc_publish_before_deadline(
    callback_start_ns, 20.0, &elapsed_ms,
    [] { return int64_t{1'020'000'000}; },
    [&] { ++publications; });
  EXPECT_FALSE(published);
  EXPECT_DOUBLE_EQ(elapsed_ms, 20.0);
  EXPECT_EQ(publications, 0);

  const bool late_published = f1tenth_mpc::mpc_publish_before_deadline(
    callback_start_ns, 20.0, &elapsed_ms,
    [] { return int64_t{1'020'001'000}; },
    [&] { ++publications; });
  EXPECT_FALSE(late_published);
  EXPECT_NEAR(elapsed_ms, 20.001, 1.0e-12);
  EXPECT_EQ(publications, 0);
}

TEST(MpcPublicationDeadline, InvokesPublisherOnlyAfterPassingBoundaryCheck)
{
  int publications = 0;
  double elapsed_ms = -1.0;
  constexpr int64_t callback_start_ns = 1'000'000'000;
  const bool published = f1tenth_mpc::mpc_publish_before_deadline(
    callback_start_ns, 20.0, &elapsed_ms,
    [] { return int64_t{1'019'999'000}; },
    [&] { ++publications; });
  EXPECT_TRUE(published);
  EXPECT_NEAR(elapsed_ms, 19.999, 1.0e-12);
  EXPECT_EQ(publications, 1);
}

TEST(MpcPublicationDeadline, RejectsInvalidMeasurementsAndBudgets)
{
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  EXPECT_FALSE(f1tenth_mpc::mpc_cycle_within_publication_deadline(nan, 20.0));
  EXPECT_FALSE(f1tenth_mpc::mpc_cycle_within_publication_deadline(inf, 20.0));
  EXPECT_FALSE(f1tenth_mpc::mpc_cycle_within_publication_deadline(-0.1, 20.0));
  EXPECT_FALSE(f1tenth_mpc::mpc_cycle_within_publication_deadline(1.0, 0.0));
  EXPECT_FALSE(f1tenth_mpc::mpc_cycle_within_publication_deadline(1.0, nan));
  EXPECT_TRUE(f1tenth_mpc::mpc_publication_budget_valid(20.0));
  EXPECT_TRUE(f1tenth_mpc::mpc_publication_budget_valid(5.0));
  EXPECT_FALSE(f1tenth_mpc::mpc_publication_budget_valid(0.0));
  EXPECT_TRUE(f1tenth_mpc::mpc_publication_budget_valid(20.001));
  EXPECT_TRUE(f1tenth_mpc::mpc_publication_budget_valid(22.0));
  EXPECT_FALSE(f1tenth_mpc::mpc_publication_budget_valid(22.001));
  EXPECT_FALSE(f1tenth_mpc::mpc_publication_budget_valid(inf));
  EXPECT_FALSE(f1tenth_mpc::mpc_publication_budget_valid(nan));
  EXPECT_FALSE(f1tenth_mpc::mpc_cycle_within_publication_deadline(
    1.0, 22.001));
}
