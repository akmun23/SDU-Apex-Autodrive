#include "mpc_publication_deadline.hpp"

#include <gtest/gtest.h>

#include <limits>

TEST(MpcPublicationDeadline, AcceptsOnlyStrictlyUnderBudget)
{
  EXPECT_TRUE(f1tenth_mpc::mpc_cycle_within_publication_deadline(19.999, 20.0));
  EXPECT_FALSE(f1tenth_mpc::mpc_cycle_within_publication_deadline(20.0, 20.0));
  EXPECT_FALSE(f1tenth_mpc::mpc_cycle_within_publication_deadline(20.001, 20.0));
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
  EXPECT_FALSE(f1tenth_mpc::mpc_publication_budget_valid(20.001));
  EXPECT_FALSE(f1tenth_mpc::mpc_publication_budget_valid(inf));
  EXPECT_FALSE(f1tenth_mpc::mpc_publication_budget_valid(nan));
  EXPECT_FALSE(f1tenth_mpc::mpc_cycle_within_publication_deadline(
    1.0, 20.001));
}
