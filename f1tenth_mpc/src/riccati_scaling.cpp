// Explicit physical<->scaled QP boundary; no independent ADMM solver here.
// The unscaled and scaled call paths use the same Riccati/ADMM core.
#include "riccati_solver_internal.hpp"
#include "riccati_state_polygon.h"
#include <math.h>
#include <string.h>

#if defined(__SSE__) && (defined(__x86_64__) || defined(__i386__))
#include <xmmintrin.h>
#endif

// Very small PhysX/Jacobian terms can create subnormal Riccati matrix
// intermediates. On x86 these trigger large, state-dependent floating-point
// assists, causing millisecond-scale deadline outliers. Only the solver's
// calling thread uses FTZ/DAZ; all original control/status logic is retained.
// Restore the original MXCSR on EVERY return, including error and timeout.
class ScopedRiccatiDenormalMode final {
public:
    ScopedRiccatiDenormalMode() noexcept
    {
#if defined(__SSE__) && (defined(__x86_64__) || defined(__i386__))
        previous_ = _mm_getcsr();
        if ((previous_ & kFlushMask) != kFlushMask)
            _mm_setcsr(previous_ | kFlushMask);
#endif
    }
    ~ScopedRiccatiDenormalMode() noexcept
    {
#if defined(__SSE__) && (defined(__x86_64__) || defined(__i386__))
        if ((previous_ & kFlushMask) != kFlushMask)
            _mm_setcsr(previous_);
#endif
    }
    ScopedRiccatiDenormalMode(const ScopedRiccatiDenormalMode &) = delete;
    ScopedRiccatiDenormalMode &operator=(const ScopedRiccatiDenormalMode &) = delete;
private:
#if defined(__SSE__) && (defined(__x86_64__) || defined(__i386__))
    static constexpr unsigned kFlushMask = (1u << 15) | (1u << 6);
    unsigned previous_{};
#endif
};

static int riccati_scaling_signature_matches(
    const RiccatiAdmmState_t *state,
    const RiccatiAdmmConfig_t *config)
{
    enum { nx = RICCATI_MAX_NX, nu = RICCATI_MAX_NU };
    if (!state || !config || !state->initialized ||
        !state->scaling_enabled)
        return 0;
    for (int i = 0; i < nx; ++i)
        if (state->state_scale[i] != config->state_scale[i]) return 0;
    for (int a = 0; a < nu; ++a)
        if (state->input_scale[a] != config->input_scale[a]) return 0;
    return 1;
}

static float scale_state_lower_bound(float bound, float scale)
{
    /* BIG_BOUND is a semantic unconstrained marker, not a physical limit.
     * Dividing it by D can move it inside the core solver's BIG_BOUND test
     * and accidentally constrain scaled wheel/body states. */
    return bound <= -BIG_BOUND ? -BIG_BOUND : bound / scale;
}

static float scale_state_upper_bound(float bound, float scale)
{
    return bound >= BIG_BOUND ? BIG_BOUND : bound / scale;
}

static RiccatiStatus_t riccati_admm_solve_impl(
    const RiccatiStepData_t *step_data,
    const float *terminal_Q,
    const float *terminal_q,
    const float *terminal_x_lb,
    const float *terminal_x_ub,
    const float *x0,
    const RiccatiAdmmConfig_t *config,
    RiccatiAdmmState_t *admm_state,
    RiccatiSolution_t *solution,
    float primal_u_out[RICCATI_HORIZON][RICCATI_MAX_NU],
    float feedback_out[RICCATI_HORIZON][RICCATI_MAX_NU][RICCATI_MAX_NX])
{
    const ScopedRiccatiDenormalMode denormal_mode;
    enum { nx = RICCATI_MAX_NX, nu = RICCATI_MAX_NU,
           N = RICCATI_HORIZON };
    if (!admm_state || !solution || !step_data || !terminal_Q ||
        !terminal_q || !terminal_x_lb || !terminal_x_ub || !x0) {
        if (solution) {
            solution->status = RICCATI_STATUS_ERROR;
            solution->time_budget_exhausted = 0;
        }
        return RICCATI_STATUS_ERROR;
    }
    solution->time_budget_exhausted = 0;
    if(config && (config->terminal_constraints.count<0 ||
        config->terminal_constraints.count>RICCATI_TERMINAL_MAX_PLANES)) {
        solution->status=RICCATI_STATUS_ERROR;
        return RICCATI_STATUS_ERROR;
    }
    for (int k = 1; config && k <= N; ++k) {
        const RiccatiStatePolygon_t *polygon = &config->state_polygons[k];
        if (polygon->count < 0 || polygon->count > RICCATI_STATE_POLYGON_MAX_PLANES) {
            solution->status = RICCATI_STATUS_ERROR;
            return RICCATI_STATUS_ERROR;
        }
        if (polygon->count == 0) continue;
        for (int axis = 0; axis < 2; ++axis) {
            const float lo = k < N ? step_data[k].x_lb[axis] : terminal_x_lb[axis];
            const float hi = k < N ? step_data[k].x_ub[axis] : terminal_x_ub[axis];
            if (!(lo > -BIG_BOUND || hi < BIG_BOUND)) {
                solution->status = RICCATI_STATUS_ERROR;
                return RICCATI_STATUS_ERROR;
            }
        }
    }
    for (int k = 0; k < RICCATI_HORIZON; ++k)
        for (int i = 0; i < RICCATI_MAX_NX; ++i)
            for (int a = 0; a < RICCATI_MAX_NU; ++a)
                if (!isfinite(step_data[k].N[i][a])) {
                    solution->status = RICCATI_STATUS_ERROR;
                    return RICCATI_STATUS_ERROR;
                }

    const int use_scaling = config ? config->use_scaling : 0;
    if ((use_scaling != 0 && use_scaling != 1) ||
        (config && config->reuse_same_epoch != 0 && config->reuse_same_epoch != 1)) {
        solution->status = RICCATI_STATUS_ERROR;
        return RICCATI_STATUS_ERROR;
    }
    if (!use_scaling) {
        if (admm_state->initialized && admm_state->scaling_enabled)
            riccati_admm_state_init(admm_state);
        return riccati_admm_solve_core(
            step_data, terminal_Q, terminal_q, terminal_x_lb, terminal_x_ub,
            x0, config, admm_state, solution, primal_u_out, feedback_out);
    }

    if (!step_data || !terminal_Q || !terminal_q || !terminal_x_lb ||
        !terminal_x_ub || !x0) {
        solution->status = RICCATI_STATUS_ERROR;
        return RICCATI_STATUS_ERROR;
    }
    for (int i = 0; i < nx; ++i) {
        if (!isfinite(config->state_scale[i]) ||
            config->state_scale[i] <= 0.0f) {
            solution->status = RICCATI_STATUS_ERROR;
            return RICCATI_STATUS_ERROR;
        }
    }
    for (int a = 0; a < nu; ++a) {
        if (!isfinite(config->input_scale[a]) ||
            config->input_scale[a] <= 0.0f) {
            solution->status = RICCATI_STATUS_ERROR;
            return RICCATI_STATUS_ERROR;
        }
    }

    if (admm_state->initialized &&
        !riccati_scaling_signature_matches(admm_state, config))
        riccati_admm_state_init(admm_state);

    /* x = Dx*xhat and u = Du*uhat.  Scale dynamics, all diagonal and
     * affine costs, and hard bounds consistently. */
#ifdef MPC_ENABLE_RICCATI_PROFILE
    const uint64_t scaling_start_ns = riccati_profile_now_ns();
#endif
    RiccatiStepData_t scaled_steps[RICCATI_HORIZON] = {};
    float scaled_terminal_Q[RICCATI_MAX_NX] = {};
    float scaled_terminal_q[RICCATI_MAX_NX] = {};
    float scaled_terminal_lb[RICCATI_MAX_NX] = {};
    float scaled_terminal_ub[RICCATI_MAX_NX] = {};
    float scaled_x0[RICCATI_MAX_NX] = {};
    for (int i = 0; i < nx; ++i) {
        const float scale = config->state_scale[i];
        scaled_terminal_Q[i] = terminal_Q[i] * scale * scale;
        scaled_terminal_q[i] = terminal_q[i] * scale;
        scaled_terminal_lb[i] = scale_state_lower_bound(
            terminal_x_lb[i], scale);
        scaled_terminal_ub[i] = scale_state_upper_bound(
            terminal_x_ub[i], scale);
        scaled_x0[i] = x0[i] / scale;
    }
    memcpy(scaled_steps, step_data, (size_t)N * sizeof(scaled_steps[0]));
    for (int k = 0; k < N; ++k) {
        RiccatiStepData_t *scaled = &scaled_steps[k];
        const RiccatiStepData_t *physical = &step_data[k];
        for (int i = 0; i < nx; ++i) {
            const float state_scale = config->state_scale[i];
            scaled->d[i] = physical->d[i] / state_scale;
            scaled->Q_diag[i] = physical->Q_diag[i] *
                state_scale * state_scale;
            scaled->q[i] = physical->q[i] * state_scale;
            for(int j=0;j<nx;++j)
                scaled->Q_offdiag[i][j] =
                    physical->Q_offdiag[i][j]*state_scale*config->state_scale[j];
            for (int a = 0; a < nu; ++a) {
                scaled->N[i][a] = physical->N[i][a] * state_scale *
                    config->input_scale[a];
            }
            scaled->x_lb[i] = scale_state_lower_bound(
                physical->x_lb[i], state_scale);
            scaled->x_ub[i] = scale_state_upper_bound(
                physical->x_ub[i], state_scale);
            for (int j = 0; j < nx; ++j) {
                scaled->A[i][j] = physical->A[i][j] *
                    config->state_scale[j] / state_scale;
            }
            for (int a = 0; a < nu; ++a) {
                scaled->B[i][a] = physical->B[i][a] *
                    config->input_scale[a] / state_scale;
            }
        }
        scaled->R_cross = physical->R_cross *
            config->input_scale[0]*config->input_scale[1];
        for (int a = 0; a < nu; ++a) {
            const float input_scale = config->input_scale[a];
            scaled->R_diag[a] = physical->R_diag[a] *
                input_scale * input_scale;
            scaled->r[a] = physical->r[a] * input_scale;
            scaled->u_lb[a] = physical->u_lb[a] / input_scale;
            scaled->u_ub[a] = physical->u_ub[a] / input_scale;
        }
    }

    RiccatiAdmmConfig_t scaled_config = *config;
    scaled_config.use_scaling = 0;
    for (int k = 1; k <= N; ++k)
        for (int face = 0; face < scaled_config.state_polygons[k].count; ++face)
            for (int axis = 0; axis < 2; ++axis)
                scaled_config.state_polygons[k].normal[face][axis] *=
                    config->state_scale[axis];
    for (int f=0;f<scaled_config.terminal_constraints.count;++f)
        for(int i=0;i<nx;++i)
            scaled_config.terminal_constraints.normal[f][i] *= config->state_scale[i];
    RiccatiSolution_t scaled_solution = {};
    float scaled_primal_u[RICCATI_HORIZON][RICCATI_MAX_NU];
#ifdef MPC_ENABLE_RICCATI_PROFILE
    const uint64_t scaling_finish_ns = riccati_profile_now_ns();
#endif
    const RiccatiStatus_t status = riccati_admm_solve_core(
        scaled_steps, scaled_terminal_Q, scaled_terminal_q,
        scaled_terminal_lb, scaled_terminal_ub, scaled_x0,
        &scaled_config, admm_state, &scaled_solution,
        primal_u_out ? scaled_primal_u : NULL, feedback_out);
    *solution = scaled_solution;
    /* The core may fail before producing a factorization or primal controls.
     * Do not read/scale those optional output buffers on an error path. */
    if (status != RICCATI_STATUS_OPTIMAL &&
        status != RICCATI_STATUS_MAX_ITERATIONS)
        return status;
#ifdef MPC_ENABLE_RICCATI_PROFILE
    const uint64_t output_unscale_start_ns = riccati_profile_now_ns();
#endif
    for (int k = 0; k <= N; ++k)
        for (int i = 0; i < nx; ++i)
            solution->x[k][i] *= config->state_scale[i];
    for (int k = 0; k < N; ++k)
        for (int a = 0; a < nu; ++a)
            solution->u[k][a] *= config->input_scale[a];
    if (primal_u_out)
        for (int k = 0; k < N; ++k)
            for (int a = 0; a < nu; ++a)
                primal_u_out[k][a] =
                    scaled_primal_u[k][a] * config->input_scale[a];
    if (feedback_out)
        for (int k = 0; k < N; ++k)
            for (int a = 0; a < nu; ++a)
                for (int i = 0; i < nx; ++i)
                    feedback_out[k][a][i] *=
                        config->input_scale[a] / config->state_scale[i];

    if (admm_state->initialized) {
        admm_state->scaling_enabled = 1;
        memcpy(admm_state->state_scale, config->state_scale,
               (size_t)nx * sizeof(config->state_scale[0]));
        memcpy(admm_state->input_scale, config->input_scale,
               (size_t)nu * sizeof(config->input_scale[0]));
    }
#ifdef MPC_ENABLE_RICCATI_PROFILE
    const uint64_t output_unscale_finish_ns = riccati_profile_now_ns();
    if (scaling_finish_ns >= scaling_start_ns)
        riccati_debug_last_mutable().scaling_transform_time_ns +=
            scaling_finish_ns - scaling_start_ns;
    if (output_unscale_finish_ns >= output_unscale_start_ns)
        riccati_debug_last_mutable().output_unscale_time_ns +=
            output_unscale_finish_ns - output_unscale_start_ns;
#endif
    return status;
}

RiccatiStatus_t riccati_admm_solve(
    const RiccatiStepData_t *step_data,
    const float *terminal_Q,
    const float *terminal_q,
    const float *terminal_x_lb,
    const float *terminal_x_ub,
    const float *x0,
    const RiccatiAdmmConfig_t *config,
    RiccatiAdmmState_t *admm_state,
    RiccatiSolution_t *solution)
{
    return riccati_admm_solve_impl(step_data, terminal_Q, terminal_q,
        terminal_x_lb, terminal_x_ub, x0, config, admm_state, solution, NULL, NULL);
}

RiccatiStatus_t riccati_admm_solve_with_primal_controls(
    const RiccatiStepData_t *step_data,
    const float *terminal_Q,
    const float *terminal_q,
    const float *terminal_x_lb,
    const float *terminal_x_ub,
    const float *x0,
    const RiccatiAdmmConfig_t *config,
    RiccatiAdmmState_t *admm_state,
    RiccatiSolution_t *solution,
    float primal_u_out[RICCATI_HORIZON][RICCATI_MAX_NU])
{
    return riccati_admm_solve_impl(step_data, terminal_Q, terminal_q,
        terminal_x_lb, terminal_x_ub, x0, config, admm_state, solution,
        primal_u_out, NULL);
}

RiccatiStatus_t riccati_admm_solve_with_feedback(
    const RiccatiStepData_t *step_data, const float *terminal_Q,
    const float *terminal_q, const float *terminal_x_lb,
    const float *terminal_x_ub, const float *x0,
    const RiccatiAdmmConfig_t *config, RiccatiAdmmState_t *admm_state,
    RiccatiSolution_t *solution,
    float primal_u_out[RICCATI_HORIZON][RICCATI_MAX_NU],
    float feedback_out[RICCATI_HORIZON][RICCATI_MAX_NU][RICCATI_MAX_NX])
{
    return riccati_admm_solve_impl(step_data, terminal_Q, terminal_q,
        terminal_x_lb, terminal_x_ub, x0, config, admm_state, solution,
        primal_u_out, feedback_out);
}
