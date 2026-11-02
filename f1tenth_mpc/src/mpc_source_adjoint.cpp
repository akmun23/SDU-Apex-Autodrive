#include "mpc_rti_internal.hpp"

#include <cmath>
#include <cstring>

/*
 * Accuracy-first source nonlinear-shooting derivative.
 *
 * The trajectory is integrated with the complete source plant at the SAME
 * physical command epoch. Source-derived A/B and the stage cost's first
 * derivatives are evaluated ONLY at that exact nonlinear nominal; the
 * QP's finite-step state prediction is not used as the control trajectory.
 *
 * This is currently an experimental mathematical primitive, not a selected
 * production control policy. In particular, the local stage cost derivative
 * must be separately certified against exact-merit finite differences in
 * every tire regime and at changing reference progress.
 */
namespace {
/* Both public variants use this ONE exact adjoint. Only the source rollout
 * and reference acquisition differ; QP Hessians and costate are identical. */
int source_adjoint_at_nominal(
    const MpcRtiNominal_t *nominal,
    const MpcRtiReference_t references[PREDICTION_HORIZON+1],
    float prediction_dt, const MpcRtiConfiguration_t *model,
    const MpcRtiObjectiveSchedule_t *frozen_objective,
    double gradient_out[PREDICTION_HORIZON][MPC_RTI_NU])
{
    if (!nominal || !nominal->valid || nominal->horizon!=PREDICTION_HORIZON ||
        !references || !model || !frozen_objective ||
        !frozen_objective->valid ||
        frozen_objective->horizon!=PREDICTION_HORIZON ||
        !gradient_out || !std::isfinite(prediction_dt) || prediction_dt<=0.)
        return 0;
    /* The frozen objective and live geometry must be passed separately.
     * Rebuild physical derivatives against this full exact-source nominal;
     * do not substitute stale reference curvature/corridor geometry or
     * recalculate speed-weight branches from candidate speed. */
    MpcRtiProblem_t problem{};
    if (!mpc_rti_build_ltv_qp_with_objective_schedule(
            nominal->states, nominal->controls, references, PREDICTION_HORIZON,
            prediction_dt, model, frozen_objective, &problem))
        return 0;

    return mpc_rti_qp_source_gradient(&problem,nominal->controls,
        model->command_increment_qp_enabled,gradient_out);
}
} // namespace

int mpc_rti_qp_source_gradient(
    const MpcRtiProblem_t *problem,
    const MpcModelControl_t controls[PREDICTION_HORIZON],
    int increment_qp,
    double gradient_out[PREDICTION_HORIZON][MPC_RTI_NU])
{
    if(!problem || !controls || !gradient_out)return 0;
    constexpr int nx = RICCATI_MAX_NX;
    constexpr int nu = RICCATI_MAX_NU;
    constexpr int N = RICCATI_HORIZON;
    double state[N + 1][nx]{};
    double decision[N][nu]{};
    for (int i = 0; i < nx; ++i) state[0][i] = problem->x0[i];

    for (int k = 0; k < N; ++k) {
        const auto &stage = problem->steps[k];
        const float previous_steer = k
            ? controls[k - 1].steering_target_rad
            : problem->x0[MPC_OPT_IDX_PREVIOUS_STEERING_COMMAND];
        const float previous_throttle = k
            ? controls[k - 1].throttle_command
            : problem->x0[MPC_OPT_IDX_PREVIOUS_THROTTLE_COMMAND];
        decision[k][0] = controls[k].steering_target_rad -
            (increment_qp ? previous_steer : 0.f);
        decision[k][1] = controls[k].throttle_command -
            (increment_qp ? previous_throttle : 0.f);

        // The affine LTV transition exactly reproduces this *nominal* source
        // trajectory, up to floating-point rounding. Never interpret this
        // identity as accurate prediction of a distant candidate sequence.
        for (int i = 0; i < nx; ++i) {
            double value = stage.d[i];
            for (int j = 0; j < nx; ++j)
                value += stage.A[i][j] * state[k][j];
            for (int a = 0; a < nu; ++a)
                value += stage.B[i][a] * decision[k][a];
            if (!std::isfinite(value)) return 0;
            state[k + 1][i] = value;
        }
    }

    double costate[N + 1][nx]{};
    for (int i = 0; i < nx; ++i)
        costate[N][i] =
            problem->terminal_Q[i] * state[N][i] + problem->terminal_q[i];

    double delta_gradient[N][nu]{};
    for (int k = N - 1; k >= 0; --k) {
        const auto &stage = problem->steps[k];
        for (int a = 0; a < nu; ++a) {
            double g = stage.R_diag[a] * decision[k][a] + stage.r[a];
            g += stage.R_cross * decision[k][1 - a];
            for (int i = 0; i < nx; ++i) {
                g += stage.N[i][a] * state[k][i];
                g += stage.B[i][a] * costate[k + 1][i];
            }
            delta_gradient[k][a] = g;
        }
        for (int i = 0; i < nx; ++i) {
            double g = stage.Q_diag[i] * state[k][i] + stage.q[i];
            for (int j = 0; j < nx; ++j)
                g += stage.Q_offdiag[i][j] * state[k][j];
            for (int a = 0; a < nu; ++a)
                g += stage.N[i][a] * decision[k][a];
            for (int j = 0; j < nx; ++j)
                g += stage.A[j][i] * costate[k + 1][j];
            costate[k][i] = g;
        }
    }

    // With decision v[k]=u_abs[k]-u_abs[k-1], chain rule gives
    // dJ/du_abs[k]=dJ/dv[k]-dJ/dv[k+1]. The measured command before stage 0
    // is held fixed, not optimized.
    for (int k = 0; k < N; ++k) {
        for (int a = 0; a < nu; ++a) {
            const double gradient = delta_gradient[k][a] -
                (increment_qp && k + 1 < N
                    ? delta_gradient[k + 1][a] : 0.0);
            if (!std::isfinite(gradient)) return 0;
            gradient_out[k][a] = gradient;
        }
    }
    return 1;
}

extern "C" int mpc_rti_source_adjoint_gradient_from_exact_nominal(
    const MpcRtiNominal_t *validated_nominal,
    const MpcTrajectorySample_t *trajectory, size_t trajectory_count,
    double lap_length, float prediction_dt,
    const MpcRtiConfiguration_t *model,
    const MpcRtiObjectiveSchedule_t *frozen_objective,
    double gradient_out[PREDICTION_HORIZON][MPC_RTI_NU])
{
    MpcRtiReference_t references[PREDICTION_HORIZON+1]{};
    if (!mpc_rti_references_from_exact_nominal(
            validated_nominal,trajectory,trajectory_count,lap_length,
            model,references))
        return 0;
    return source_adjoint_at_nominal(validated_nominal,references,
        prediction_dt,model,frozen_objective,gradient_out);
}

extern "C" int mpc_rti_source_adjoint_gradient(
    const MpcRtiState_t *initial, double progress,
    const MpcTrajectorySample_t *trajectory, size_t trajectory_count,
    double lap_length, float prediction_dt,
    const MpcRtiConfiguration_t *model,
    const MpcModelControl_t controls[PREDICTION_HORIZON],
    const MpcRtiObjectiveSchedule_t *frozen_objective,
    double gradient_out[PREDICTION_HORIZON][MPC_RTI_NU])
{
    if (!initial || !trajectory || trajectory_count < 3 || !model ||
        !controls || !frozen_objective || !frozen_objective->valid ||
        frozen_objective->horizon!=PREDICTION_HORIZON || !gradient_out)
        return 0;
    MpcRtiNominal_t nominal{};
    MpcRtiReference_t references[PREDICTION_HORIZON+1]{};
    if (!mpc_rti_build_nominal_same_epoch(initial,progress,nullptr,
        controls,PREDICTION_HORIZON,trajectory,trajectory_count,lap_length,
        prediction_dt,PREDICTION_HORIZON,model,&nominal,references))
        return 0;
    return source_adjoint_at_nominal(&nominal,references,
        prediction_dt,model,frozen_objective,gradient_out);
}
