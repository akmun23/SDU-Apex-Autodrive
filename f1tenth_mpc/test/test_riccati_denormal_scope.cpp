#include "riccati_solver.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <memory>
#if defined(__SSE__) && (defined(__x86_64__) || defined(__i386__))
#include <xmmintrin.h>

TEST(RiccatiDenormalScope, RestoresThreadMxcsrOnSuccessAndEarlyFailure)
{
    constexpr unsigned kFlushDaz = (1u << 15) | (1u << 6);
    struct RestoreOriginalMxcsr {
        unsigned original = _mm_getcsr();
        ~RestoreOriginalMxcsr() { _mm_setcsr(original); }
    } restore;
    for (unsigned desired : {restore.original & ~kFlushDaz,
                             restore.original | kFlushDaz}) {
        _mm_setcsr(desired);
        auto steps = std::make_unique<RiccatiStepData_t[]>(RICCATI_HORIZON);
        auto state = std::make_unique<RiccatiAdmmState_t>();
        auto solution = std::make_unique<RiccatiSolution_t>();
        float terminal_Q[RICCATI_MAX_NX]{};
        float terminal_q[RICCATI_MAX_NX]{};
        float lower[RICCATI_MAX_NX]{};
        float upper[RICCATI_MAX_NX]{};
        float x0[RICCATI_MAX_NX]{};
        for (int i = 0; i < RICCATI_MAX_NX; ++i) {
            terminal_Q[i] = 1.0f;
            lower[i] = -BIG_BOUND;
            upper[i] = BIG_BOUND;
        }
        for (int k = 0; k < RICCATI_HORIZON; ++k) {
            auto &stage = steps[k];
            for (int i = 0; i < RICCATI_MAX_NX; ++i) {
                stage.Q_diag[i] = 1.0f;
                stage.A[i][i] = 1.0f;
                stage.x_lb[i] = -BIG_BOUND;
                stage.x_ub[i] = BIG_BOUND;
            }
            for (int a = 0; a < RICCATI_MAX_NU; ++a) {
                stage.R_diag[a] = 1.0f;
                stage.u_lb[a] = -1.0f;
                stage.u_ub[a] = 1.0f;
            }
        }
        RiccatiAdmmConfig_t config{};
        config.rho = config.rho_u = 7.0f;
        config.rho_max = 32.0f;
        config.tolerance = 0.01f;
        config.max_iterations = 12;
        config.over_relaxation = 1.0f;
        const RiccatiStatus_t status = riccati_admm_solve(
            steps.get(), terminal_Q, terminal_q, lower, upper, x0,
            &config, state.get(), solution.get());
        EXPECT_NE(status, RICCATI_STATUS_ERROR);
        EXPECT_EQ(_mm_getcsr(), desired)
            << "successful Riccati solve changed caller's floating-point mode";
        const auto invalid = riccati_admm_solve(
            nullptr, terminal_Q, terminal_q, lower, upper, x0,
            &config, state.get(), solution.get());
        EXPECT_EQ(invalid, RICCATI_STATUS_ERROR);
        EXPECT_EQ(_mm_getcsr(), desired)
            << "failed Riccati solve did not restore caller's MXCSR";
    }
}
#endif
