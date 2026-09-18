#define _POSIX_C_SOURCE 200809L

#include "mpc_rti.h"

#include <math.h>
#include <string.h>
#include <time.h>

static int finite_nonnegative(float value)
{
    return isfinite(value) && value >= 0.0f;
}

static int finite_rti_state(const MpcRtiState_t *state)
{
    return state && isfinite(state->plant.e_y) &&
        isfinite(state->plant.e_psi) && isfinite(state->plant.u) &&
        isfinite(state->plant.v) && isfinite(state->plant.r) &&
        isfinite(state->plant.target_speed) &&
        isfinite(state->plant.steering_command) &&
        isfinite(state->previous_steering_rate) &&
        isfinite(state->previous_target_speed_rate);
}

static int finite_reference(const MpcRtiReference_t *reference)
{
    return reference && isfinite(reference->e_y) &&
        isfinite(reference->e_psi) && isfinite(reference->u) &&
        isfinite(reference->v) && isfinite(reference->r) &&
        isfinite(reference->steering_command) &&
        isfinite(reference->path_curvature) &&
        isfinite(reference->left_bound) && reference->left_bound > 0.0f &&
        isfinite(reference->right_bound) && reference->right_bound > 0.0f;
}

static int valid_configuration(const MpcRtiConfiguration_t *configuration)
{
    if (!configuration) return 0;
    if (configuration->use_fd_jacobian_oracle != 0 &&
        configuration->use_fd_jacobian_oracle != 1) return 0;
#ifndef MPC_ENABLE_FD_ORACLE
    if (configuration->use_fd_jacobian_oracle) return 0;
#endif
    const VehicleParameters_t vehicle = vehicle_model_get_parameters();
    const float weights[] = {
        configuration->weight_e_y,
        configuration->weight_e_psi,
        configuration->weight_u,
        configuration->weight_v,
        configuration->weight_r,
        configuration->weight_steering_command,
        configuration->weight_steering_rate,
        configuration->weight_target_speed_rate,
        configuration->weight_steering_rate_change,
        configuration->weight_target_speed_rate_change,
        configuration->terminal_multiplier,
        configuration->corridor_margin_m,
        configuration->corridor_preview_halfwidth_m,
        configuration->nonlinear_corridor_tolerance_m};
    for (unsigned int i = 0; i < sizeof(weights) / sizeof(weights[0]); ++i)
        if (!finite_nonnegative(weights[i])) return 0;

    return isfinite(configuration->max_speed_mps) &&
        configuration->max_speed_mps > 0.0f &&
        configuration->max_speed_mps <=
            vehicle.maximum_command_speed_mps &&
        isfinite(configuration->active_speed_ceiling_mps) &&
        configuration->active_speed_ceiling_mps > 0.0f &&
        configuration->active_speed_ceiling_mps <=
            configuration->max_speed_mps &&
        isfinite(configuration->max_steering_rad) &&
        configuration->max_steering_rad > 0.0f &&
        configuration->max_steering_rad <= vehicle.max_steering_angle &&
        isfinite(configuration->max_steering_rate_radps) &&
        configuration->max_steering_rate_radps > 0.0f &&
        configuration->max_steering_rate_radps <=
            vehicle.steering_rate_radps &&
        isfinite(configuration->max_target_speed_rate_increase_mps2) &&
        configuration->max_target_speed_rate_increase_mps2 > 0.0f &&
        configuration->max_target_speed_rate_increase_mps2 <=
            vehicle.maximum_target_speed_rate_increase_mps2 &&
        isfinite(configuration->max_target_speed_rate_reduction_mps2) &&
        configuration->max_target_speed_rate_reduction_mps2 > 0.0f &&
        configuration->max_target_speed_rate_reduction_mps2 <=
            vehicle.maximum_target_speed_rate_reduction_mps2;
}

static void serialize_state(const MpcRtiState_t *state, float x[MPC_RTI_NX])
{
    x[MPC_RTI_IDX_EY] = state->plant.e_y;
    x[MPC_RTI_IDX_EPSI] = state->plant.e_psi;
    x[MPC_RTI_IDX_U] = state->plant.u;
    x[MPC_RTI_IDX_V] = state->plant.v;
    x[MPC_RTI_IDX_R] = state->plant.r;
    x[MPC_RTI_IDX_TARGET_SPEED] = state->plant.target_speed;
    x[MPC_RTI_IDX_STEERING_COMMAND] = state->plant.steering_command;
    x[MPC_RTI_IDX_PREVIOUS_STEERING_RATE] = state->previous_steering_rate;
    x[MPC_RTI_IDX_PREVIOUS_TARGET_SPEED_RATE] =
        state->previous_target_speed_rate;
}

static float clampf_rti(float value, float lower, float upper)
{
    return fmaxf(lower, fminf(upper, value));
}

static void include_bound_knots(
    const MpcTrajectorySample_t *trajectory,
    size_t trajectory_count,
    double lower_s,
    double upper_s,
    float *left_bound,
    float *right_bound)
{
    size_t lower = 0;
    size_t upper = trajectory_count;
    while (lower < upper) {
        const size_t middle = lower + (upper - lower) / 2;
        if (trajectory[middle].s < lower_s) lower = middle + 1;
        else upper = middle;
    }
    for (size_t i = lower; i < trajectory_count &&
            trajectory[i].s <= upper_s; ++i) {
        *left_bound = fminf(*left_bound, (float)trajectory[i].left_bound);
        *right_bound = fminf(*right_bound, (float)trajectory[i].right_bound);
    }
}

static int reference_at_progress(
    const MpcTrajectorySample_t *trajectory,
    size_t trajectory_count,
    double lap_length,
    double progress,
    const MpcRtiConfiguration_t *configuration,
    MpcRtiReference_t *reference)
{
    MpcTrajectorySample_t sample;
    if (!reference || !mpc_trajectory_sample(trajectory, trajectory_count,
            lap_length, progress, &sample)) return 0;
    const float speed = fminf((float)sample.speed,
        configuration->active_speed_ceiling_mps);
    const float feedforward = clampf_rti(
        atanf((float)sample.curvature /
              MPC_YAW_RATE_STEERING_GAIN_PER_M),
        -configuration->max_steering_rad,
        configuration->max_steering_rad);
    float left_bound = (float)sample.left_bound;
    float right_bound = (float)sample.right_bound;
    if (configuration->corridor_preview_halfwidth_m > 0.0f) {
        const double halfwidth =
            configuration->corridor_preview_halfwidth_m;
        MpcTrajectorySample_t endpoint;
        if (!mpc_trajectory_sample(trajectory, trajectory_count, lap_length,
                progress - halfwidth, &endpoint)) return 0;
        left_bound = fminf(left_bound, (float)endpoint.left_bound);
        right_bound = fminf(right_bound, (float)endpoint.right_bound);
        if (!mpc_trajectory_sample(trajectory, trajectory_count, lap_length,
                progress + halfwidth, &endpoint)) return 0;
        left_bound = fminf(left_bound, (float)endpoint.left_bound);
        right_bound = fminf(right_bound, (float)endpoint.right_bound);

        /* Piecewise-linear bounds attain their interval minimum at an
         * endpoint or a trajectory knot. Binary-search only the exact meter
         * window; scanning the full closed trajectory per reference dominated
         * the N30 callback without adding information. */
        double center = fmod(progress - trajectory[0].s, lap_length);
        if (center < 0.0) center += lap_length;
        center += trajectory[0].s;
        const double lap_end = trajectory[0].s + lap_length;
        const double window_start = center - halfwidth;
        const double window_end = center + halfwidth;
        if (window_start < trajectory[0].s) {
            include_bound_knots(trajectory, trajectory_count,
                window_start + lap_length, lap_end, &left_bound, &right_bound);
            include_bound_knots(trajectory, trajectory_count,
                trajectory[0].s, window_end, &left_bound, &right_bound);
        } else if (window_end >= lap_end) {
            include_bound_knots(trajectory, trajectory_count,
                window_start, lap_end, &left_bound, &right_bound);
            include_bound_knots(trajectory, trajectory_count,
                trajectory[0].s, window_end - lap_length,
                &left_bound, &right_bound);
        } else {
            include_bound_knots(trajectory, trajectory_count,
                window_start, window_end, &left_bound, &right_bound);
        }
    }
    *reference = (MpcRtiReference_t){
        .e_y = 0.0f,
        .e_psi = 0.0f,
        .u = speed,
        .v = 0.0f,
        .r = (float)sample.curvature * speed,
        .steering_command = feedforward,
        .path_curvature = (float)sample.curvature,
        .left_bound = left_bound,
        .right_bound = right_bound};
    return finite_reference(reference);
}

static int rollout_nominal_controls(
    const MpcRtiState_t *initial_state,
    double initial_progress,
    const MpcModelControl_t controls[PREDICTION_HORIZON],
    const MpcRtiReference_t references[PREDICTION_HORIZON + 1],
    int horizon,
    float prediction_dt,
    MpcRtiState_t states[PREDICTION_HORIZON + 1],
    double progress[PREDICTION_HORIZON + 1])
{
    states[0] = *initial_state;
    progress[0] = initial_progress;
    for (int k = 0; k < horizon; ++k) {
        const MpcStageResult_t step = mpc_vehicle_model_step(
            &states[k].plant, &controls[k], prediction_dt,
            references[k].path_curvature);
        if (!step.valid || !isfinite(progress[k] + step.delta_s_m)) return 0;
        states[k + 1].plant = step.next;
        states[k + 1].previous_steering_rate = controls[k].steering_rate;
        states[k + 1].previous_target_speed_rate =
            controls[k].target_speed_rate;
        progress[k + 1] = progress[k] + step.delta_s_m;
    }
    return 1;
}

static int corridor_contains(
    const MpcRtiState_t *state,
    const MpcRtiReference_t *reference,
    const MpcRtiConfiguration_t *configuration)
{
    const float lower = configuration->corridor_margin_m -
        reference->right_bound -
        configuration->nonlinear_corridor_tolerance_m;
    const float upper = reference->left_bound -
        configuration->corridor_margin_m +
        configuration->nonlinear_corridor_tolerance_m;
    return isfinite(lower) && isfinite(upper) && lower <= upper &&
        state->plant.e_y >= lower && state->plant.e_y <= upper;
}

static int state_inside_command_envelope(
    const MpcRtiState_t *state,
    const MpcRtiConfiguration_t *configuration)
{
    /* The two history states are observed prior commands, not new actions.
     * Keep them finite, but allow them outside today's optimizer input box;
     * QP inputs themselves remain bounded at every stage. */
    return finite_rti_state(state) && state->plant.u >= 0.0f &&
        state->plant.u <= configuration->max_speed_mps &&
        state->plant.target_speed >= 0.0f &&
        state->plant.target_speed <= configuration->active_speed_ceiling_mps &&
        fabsf(state->plant.steering_command) <=
            configuration->max_steering_rad;
}

static int valid_previous_nominal(
    const MpcRtiNominal_t *nominal,
    int horizon)
{
    if (!nominal || !nominal->valid || nominal->horizon != horizon)
        return 0;
    for (int k = 0; k <= horizon; ++k) {
        if (!finite_rti_state(&nominal->states[k]) ||
            !isfinite(nominal->progress[k])) return 0;
    }
    for (int k = 0; k < horizon; ++k)
        if (!isfinite(nominal->controls[k].steering_rate) ||
            !isfinite(nominal->controls[k].target_speed_rate)) return 0;
    return 1;
}

static void set_unconstrained_bounds(RiccatiStepData_t *stage)
{
    for (int i = 0; i < MPC_RTI_NX; ++i) {
        stage->x_lb[i] = -BIG_BOUND;
        stage->x_ub[i] = BIG_BOUND;
    }
    for (int i = 0; i < MPC_RTI_NU; ++i) {
        stage->u_lb[i] = -BIG_BOUND;
        stage->u_ub[i] = BIG_BOUND;
    }
}

static int set_state_bounds(
    float lower[MPC_RTI_NX],
    float upper[MPC_RTI_NX],
    const MpcRtiReference_t *reference,
    const MpcRtiConfiguration_t *configuration)
{
    const float ey_lower = configuration->corridor_margin_m -
        reference->right_bound;
    const float ey_upper = reference->left_bound -
        configuration->corridor_margin_m;
    if (!isfinite(ey_lower) || !isfinite(ey_upper) || ey_lower > ey_upper)
        return 0;

    for (int i = 0; i < MPC_RTI_NX; ++i) {
        lower[i] = -BIG_BOUND;
        upper[i] = BIG_BOUND;
    }
    lower[MPC_RTI_IDX_EY] = ey_lower;
    upper[MPC_RTI_IDX_EY] = ey_upper;
    lower[MPC_RTI_IDX_U] = 0.0f;
    upper[MPC_RTI_IDX_U] = configuration->max_speed_mps;
    lower[MPC_RTI_IDX_TARGET_SPEED] = 0.0f;
    upper[MPC_RTI_IDX_TARGET_SPEED] =
        configuration->active_speed_ceiling_mps;
    lower[MPC_RTI_IDX_STEERING_COMMAND] =
        -configuration->max_steering_rad;
    upper[MPC_RTI_IDX_STEERING_COMMAND] =
        configuration->max_steering_rad;
    return 1;
}

static void add_tracking_cost(
    float Q[MPC_RTI_NX],
    float q[MPC_RTI_NX],
    const MpcRtiReference_t *reference,
    const MpcRtiConfiguration_t *configuration,
    float multiplier)
{
    const int indices[] = {
        MPC_RTI_IDX_EY, MPC_RTI_IDX_EPSI, MPC_RTI_IDX_U,
        MPC_RTI_IDX_V, MPC_RTI_IDX_R,
        MPC_RTI_IDX_STEERING_COMMAND};
    const float targets[] = {
        reference->e_y, reference->e_psi, reference->u,
        reference->v, reference->r, reference->steering_command};
    const float weights[] = {
        configuration->weight_e_y, configuration->weight_e_psi,
        configuration->weight_u, configuration->weight_v,
        configuration->weight_r, configuration->weight_steering_command};
    for (int i = 0; i < 6; ++i) {
        const int index = indices[i];
        const float scaled_weight = multiplier * weights[i];
        Q[index] += 2.0f * scaled_weight;
        q[index] -= 2.0f * scaled_weight * targets[i];
    }
    /* Intentionally no direct target-speed-state tracking cost. */
}

int mpc_rti_build_ltv_qp(
    const MpcRtiState_t nominal_states[PREDICTION_HORIZON + 1],
    const MpcModelControl_t nominal_controls[PREDICTION_HORIZON],
    const MpcRtiReference_t references[PREDICTION_HORIZON + 1],
    int horizon,
    float prediction_dt,
    const MpcRtiConfiguration_t *configuration,
    MpcRtiProblem_t *problem)
{
    if (!nominal_states || !nominal_controls || !references || !problem ||
        !valid_configuration(configuration) || horizon < 1 ||
        horizon > PREDICTION_HORIZON || !isfinite(prediction_dt) ||
        prediction_dt <= 0.0f) {
        return 0;
    }
    for (int k = 0; k <= horizon; ++k) {
        if (!finite_rti_state(&nominal_states[k]) ||
            !finite_reference(&references[k])) return 0;
    }
    for (int k = 0; k < horizon; ++k) {
        if (!isfinite(nominal_controls[k].steering_rate) ||
            !isfinite(nominal_controls[k].target_speed_rate)) return 0;
    }

    memset(problem, 0, sizeof(*problem));
    problem->horizon = horizon;
    serialize_state(&nominal_states[0], problem->x0);

    for (int k = 0; k < horizon; ++k) {
        RiccatiStepData_t *stage = &problem->steps[k];
        MpcStageLinearization_t plant_linearization;
#ifdef MPC_ENABLE_FD_ORACLE
        const int linearized = configuration->use_fd_jacobian_oracle
            ? mpc_model_linearize_fd_oracle(&nominal_states[k].plant,
                &nominal_controls[k], prediction_dt,
                references[k].path_curvature, &plant_linearization)
            : mpc_model_linearize(&nominal_states[k].plant,
                &nominal_controls[k], prediction_dt,
                references[k].path_curvature, &plant_linearization);
#else
        const int linearized = mpc_model_linearize(&nominal_states[k].plant,
            &nominal_controls[k], prediction_dt,
            references[k].path_curvature, &plant_linearization);
#endif
        if (!linearized) {
            return 0;
        }
        problem->nonsmooth_jacobian_columns +=
            plant_linearization.nonsmooth_column_count;
        set_unconstrained_bounds(stage);

        for (int row = 0; row < MPC_RTI_PLANT_NX; ++row) {
            stage->d[row] = plant_linearization.d[row];
            for (int col = 0; col < MPC_RTI_PLANT_NX; ++col)
                stage->A[row][col] = plant_linearization.A[row][col];
            for (int input = 0; input < MPC_RTI_NU; ++input)
                stage->B[row][input] = plant_linearization.B[row][input];
        }
        /* Previous-input memory has the exact dynamics p[k+1] = w[k]. */
        for (int input = 0; input < MPC_RTI_NU; ++input) {
            const int memory_row = MPC_RTI_IDX_PREVIOUS_STEERING_RATE + input;
            stage->B[memory_row][input] = 1.0f;
        }

        add_tracking_cost(stage->Q_diag, stage->q, &references[k],
                          configuration, 1.0f);
        stage->R_diag[0] = 2.0f * configuration->weight_steering_rate +
            2.0f * configuration->weight_steering_rate_change;
        stage->R_diag[1] = 2.0f * configuration->weight_target_speed_rate +
            2.0f * configuration->weight_target_speed_rate_change;
        stage->Q_diag[MPC_RTI_IDX_PREVIOUS_STEERING_RATE] +=
            2.0f * configuration->weight_steering_rate_change;
        stage->Q_diag[MPC_RTI_IDX_PREVIOUS_TARGET_SPEED_RATE] +=
            2.0f * configuration->weight_target_speed_rate_change;
        stage->N[MPC_RTI_IDX_PREVIOUS_STEERING_RATE][0] =
            -2.0f * configuration->weight_steering_rate_change;
        stage->N[MPC_RTI_IDX_PREVIOUS_TARGET_SPEED_RATE][1] =
            -2.0f * configuration->weight_target_speed_rate_change;

        if (!set_state_bounds(stage->x_lb, stage->x_ub, &references[k],
                              configuration)) return 0;
        stage->u_lb[0] = -configuration->max_steering_rate_radps;
        stage->u_ub[0] = configuration->max_steering_rate_radps;
        stage->u_lb[1] = -configuration->max_target_speed_rate_reduction_mps2;
        stage->u_ub[1] = configuration->max_target_speed_rate_increase_mps2;
    }

    add_tracking_cost(problem->terminal_Q, problem->terminal_q,
                      &references[horizon], configuration,
                      configuration->terminal_multiplier);
    if (!set_state_bounds(problem->terminal_x_lb, problem->terminal_x_ub,
                          &references[horizon], configuration)) return 0;
    return 1;
}

int mpc_rti_build_nominal(
    const MpcRtiState_t *current_state,
    double current_progress,
    const MpcRtiNominal_t *previous_nominal,
    const MpcTrajectorySample_t *trajectory,
    size_t trajectory_count,
    double lap_length,
    float prediction_dt,
    int horizon,
    const MpcRtiConfiguration_t *configuration,
    MpcRtiNominal_t *nominal,
    MpcRtiReference_t references[PREDICTION_HORIZON + 1])
{
    if (!finite_rti_state(current_state) || !isfinite(current_progress) ||
        !trajectory || !references || !nominal || horizon < 1 ||
        horizon > PREDICTION_HORIZON || !isfinite(prediction_dt) ||
        prediction_dt <= 0.0f || !valid_configuration(configuration)) return 0;

    MpcRtiNominal_t previous_snapshot;
    const int warm = valid_previous_nominal(previous_nominal, horizon);
    if (warm) {
        previous_snapshot = *previous_nominal;
        previous_nominal = &previous_snapshot;
    }
    memset(nominal, 0, sizeof(*nominal));
    nominal->horizon = horizon;
    nominal->states[0] = *current_state;

    double progress_seed[PREDICTION_HORIZON + 1] = {0};
    if (warm) {
        progress_seed[0] = current_progress;
        const double previous_origin = previous_nominal->progress[1];
        for (int k = 1; k < horizon; ++k) {
            progress_seed[k] = current_progress +
                previous_nominal->progress[k + 1] - previous_origin;
        }
        const double last_increment = previous_nominal->progress[horizon] -
            previous_nominal->progress[horizon - 1];
        progress_seed[horizon] = current_progress +
            previous_nominal->progress[horizon] - previous_origin +
            last_increment;
        for (int k = 0; k < horizon; ++k) {
            const int shifted_index = k + 1 < horizon ? k + 1 : horizon - 1;
            nominal->controls[k] = previous_nominal->controls[shifted_index];
            nominal->controls[k].steering_rate = clampf_rti(
                nominal->controls[k].steering_rate,
                -configuration->max_steering_rate_radps,
                configuration->max_steering_rate_radps);
            nominal->controls[k].target_speed_rate = clampf_rti(
                nominal->controls[k].target_speed_rate,
                -configuration->max_target_speed_rate_reduction_mps2,
                configuration->max_target_speed_rate_increase_mps2);
        }
    } else {
        TrajectoryReferencePoint_t seed_references[PREDICTION_HORIZON + 1];
        if (!mpc_reference_build_speed_seed(
                trajectory, trajectory_count, lap_length, current_progress,
                configuration->active_speed_ceiling_mps, prediction_dt,
                horizon, seed_references, progress_seed)) return 0;
        for (int k = 0; k <= horizon; ++k)
            if (!reference_at_progress(trajectory, trajectory_count,
                    lap_length, progress_seed[k], configuration,
                    &references[k])) return 0;

        MpcModelState_t plant = current_state->plant;
        for (int k = 0; k < horizon; ++k) {
            const float desired_delta = references[k + 1].steering_command;
            const float desired_target = references[k + 1].u;
            nominal->controls[k].steering_rate = clampf_rti(
                (desired_delta - plant.steering_command) / prediction_dt,
                -configuration->max_steering_rate_radps,
                configuration->max_steering_rate_radps);
            nominal->controls[k].target_speed_rate = clampf_rti(
                (desired_target - plant.target_speed) / prediction_dt,
                -configuration->max_target_speed_rate_reduction_mps2,
                configuration->max_target_speed_rate_increase_mps2);
            const MpcStageResult_t step = mpc_vehicle_model_step(
                &plant, &nominal->controls[k], prediction_dt,
                references[k].path_curvature);
            if (!step.valid) return 0;
            plant = step.next;
        }
    }

    MpcRtiReference_t pass_a_references[PREDICTION_HORIZON + 1];
    for (int k = 0; k <= horizon; ++k) {
        if (!reference_at_progress(trajectory, trajectory_count, lap_length,
                progress_seed[k], configuration, &pass_a_references[k]))
            return 0;
    }
    MpcRtiState_t pass_a_states[PREDICTION_HORIZON + 1];
    double pass_a_progress[PREDICTION_HORIZON + 1];
    if (!rollout_nominal_controls(current_state, current_progress,
            nominal->controls, pass_a_references, horizon, prediction_dt,
            pass_a_states, pass_a_progress)) return 0;

    /* Pass B resamples path data from Pass A's exact nonlinear delta_s and
     * rerolls once. This is deterministic and does not iterate to convergence. */
    for (int k = 0; k <= horizon; ++k) {
        if (!reference_at_progress(trajectory, trajectory_count, lap_length,
                pass_a_progress[k], configuration, &references[k])) return 0;
    }
    if (!rollout_nominal_controls(current_state, current_progress,
            nominal->controls, references, horizon, prediction_dt,
            nominal->states, nominal->progress)) return 0;
    for (int k = 0; k <= horizon; ++k)
        if (!state_inside_command_envelope(&nominal->states[k],
                configuration)) return 0;

    nominal->valid = 1;
    return 1;
}

MpcRtiRolloutStatus_t mpc_rti_rollout_candidate(
    const MpcRtiState_t *initial_state,
    double initial_progress,
    const MpcModelControl_t controls[PREDICTION_HORIZON],
    const MpcTrajectorySample_t *trajectory,
    size_t trajectory_count,
    double lap_length,
    const MpcRtiReference_t nominal_references[PREDICTION_HORIZON + 1],
    const double nominal_progress[PREDICTION_HORIZON + 1],
    MpcRtiCandidatePathDelta_t *path_delta,
    int horizon,
    float prediction_dt,
    const MpcRtiConfiguration_t *configuration,
    MpcRtiState_t states[PREDICTION_HORIZON + 1],
    double progress[PREDICTION_HORIZON + 1],
    int *failure_stage)
{
    if (failure_stage) *failure_stage = -1;
    if (path_delta) memset(path_delta, 0, sizeof(*path_delta));
    if (!finite_rti_state(initial_state) || !isfinite(initial_progress) ||
        !controls || !trajectory || trajectory_count < 3 ||
        !isfinite(lap_length) || !(lap_length > 0.0) ||
        ((nominal_references == NULL) != (nominal_progress == NULL)) ||
        (path_delta && (!nominal_references || !nominal_progress)) ||
        !states || !valid_configuration(configuration) || horizon < 1 ||
        horizon > PREDICTION_HORIZON || !isfinite(prediction_dt) ||
        prediction_dt <= 0.0f) return MPC_RTI_ROLLOUT_INVALID_INPUT;
    states[0] = *initial_state;
    if (!state_inside_command_envelope(&states[0], configuration))
        return MPC_RTI_ROLLOUT_INVALID_INPUT;
    double candidate_progress[PREDICTION_HORIZON + 1] = {0.0};
    candidate_progress[0] = initial_progress;
    MpcRtiReference_t candidate_reference = {0};
    if (!reference_at_progress(trajectory, trajectory_count, lap_length,
            candidate_progress[0], configuration, &candidate_reference))
        return MPC_RTI_ROLLOUT_INVALID_INPUT;
    if (!corridor_contains(&states[0], &candidate_reference, configuration))
        return MPC_RTI_ROLLOUT_CORRIDOR;
    if (path_delta) {
        if (!finite_reference(&nominal_references[0]) ||
            !isfinite(nominal_progress[0]))
            return MPC_RTI_ROLLOUT_INVALID_INPUT;
        path_delta->progress_error_m[0] =
            fabs(candidate_progress[0] - nominal_progress[0]);
        path_delta->curvature_error_per_m[0] = fabsf(
            candidate_reference.path_curvature -
            nominal_references[0].path_curvature);
        path_delta->left_bound_error_m[0] = fabsf(
            candidate_reference.left_bound - nominal_references[0].left_bound);
        path_delta->right_bound_error_m[0] = fabsf(
            candidate_reference.right_bound - nominal_references[0].right_bound);
        path_delta->sample_count = 1;
    }
    if (progress) progress[0] = initial_progress;

    for (int k = 0; k < horizon; ++k) {
        const MpcModelControl_t *control = &controls[k];
        if (!isfinite(control->steering_rate) ||
            !isfinite(control->target_speed_rate)) {
            if (failure_stage) *failure_stage = k;
            return MPC_RTI_ROLLOUT_INVALID_INPUT;
        }
        if (control->steering_rate <
                -configuration->max_steering_rate_radps - 1.0e-5f ||
            control->steering_rate >
                configuration->max_steering_rate_radps + 1.0e-5f ||
            control->target_speed_rate <
                -configuration->max_target_speed_rate_reduction_mps2 - 1.0e-5f ||
            control->target_speed_rate >
                configuration->max_target_speed_rate_increase_mps2 + 1.0e-5f) {
            if (failure_stage) *failure_stage = k;
            return MPC_RTI_ROLLOUT_COMMAND_LIMIT;
        }
        if (!reference_at_progress(trajectory, trajectory_count, lap_length,
                candidate_progress[k], configuration, &candidate_reference)) {
            if (failure_stage) *failure_stage = k;
            return MPC_RTI_ROLLOUT_INVALID_INPUT;
        }
        const MpcStageResult_t step = mpc_vehicle_model_step(
            &states[k].plant, control, prediction_dt,
            candidate_reference.path_curvature);
        if (!step.valid) {
            if (failure_stage) *failure_stage = k;
            return MPC_RTI_ROLLOUT_INVALID_MODEL;
        }
        states[k + 1].plant = step.next;
        states[k + 1].previous_steering_rate = control->steering_rate;
        states[k + 1].previous_target_speed_rate =
            control->target_speed_rate;
        candidate_progress[k + 1] = candidate_progress[k] + step.delta_s_m;
        if (progress) progress[k + 1] = candidate_progress[k + 1];
        if (!state_inside_command_envelope(&states[k + 1], configuration)) {
            if (failure_stage) *failure_stage = k;
            return MPC_RTI_ROLLOUT_STATE_LIMIT;
        }
        if (!reference_at_progress(trajectory, trajectory_count, lap_length,
                candidate_progress[k + 1], configuration, &candidate_reference) ||
            !finite_reference(&candidate_reference)) {
            if (failure_stage) *failure_stage = k;
            return MPC_RTI_ROLLOUT_INVALID_INPUT;
        }
        if (path_delta) {
            if (!finite_reference(&nominal_references[k + 1]) ||
                !isfinite(nominal_progress[k + 1])) {
                if (failure_stage) *failure_stage = k;
                return MPC_RTI_ROLLOUT_INVALID_INPUT;
            }
            path_delta->progress_error_m[k + 1] = fabs(
                candidate_progress[k + 1] - nominal_progress[k + 1]);
            path_delta->curvature_error_per_m[k + 1] = fabsf(
                candidate_reference.path_curvature -
                nominal_references[k + 1].path_curvature);
            path_delta->left_bound_error_m[k + 1] = fabsf(
                candidate_reference.left_bound -
                nominal_references[k + 1].left_bound);
            path_delta->right_bound_error_m[k + 1] = fabsf(
                candidate_reference.right_bound -
                nominal_references[k + 1].right_bound);
            path_delta->sample_count = k + 2;
        }
        if (!corridor_contains(&states[k + 1], &candidate_reference,
                               configuration)) {
            if (failure_stage) *failure_stage = k;
            return MPC_RTI_ROLLOUT_CORRIDOR;
        }
    }
    return MPC_RTI_ROLLOUT_OK;
}

void mpc_rti_memory_reset(MpcRtiMemory_t *memory)
{
    if (!memory) return;
    memset(memory, 0, sizeof(*memory));
    riccati_admm_state_init(&memory->solver_state);
}

static int valid_cycle_configuration(
    const MpcRtiCycleConfiguration_t *configuration)
{
    const int valid_refinement_mode = configuration &&
        (configuration->refinement_mode == MPC_RTI_REFINEMENT_R1 ||
         configuration->refinement_mode == MPC_RTI_REFINEMENT_R2 ||
         configuration->refinement_mode == MPC_RTI_REFINEMENT_ADAPTIVE);
    return configuration && valid_configuration(&configuration->model) &&
        valid_refinement_mode &&
        finite_nonnegative(configuration->rti2_progress_error_trigger_m) &&
        finite_nonnegative(configuration->rti2_curvature_error_trigger_per_m) &&
        finite_nonnegative(configuration->rti2_bound_error_trigger_m) &&
        finite_nonnegative(configuration->rti2_min_corridor_slack_trigger_m) &&
        finite_nonnegative(
            configuration->rti2_steering_rate_correction_trigger_radps) &&
        finite_nonnegative(
            configuration->rti2_target_speed_rate_correction_trigger_mps2) &&
        (configuration->refinement_mode == MPC_RTI_REFINEMENT_R1 ||
         (isfinite(configuration->rti2_residual_imbalance_trigger) &&
          configuration->rti2_residual_imbalance_trigger >= 1.0f)) &&
        finite_nonnegative(configuration->rti2_lateral_load_trigger_mps2) &&
        configuration->rti2_nonsmooth_columns_trigger >= 0 &&
        isfinite(configuration->solver.rho) &&
        configuration->solver.rho > 0.0f &&
        isfinite(configuration->solver.rho_u) &&
        configuration->solver.rho_u > 0.0f &&
        isfinite(configuration->solver.tolerance) &&
        configuration->solver.tolerance > 0.0f &&
        configuration->solver.max_iterations > 0 &&
        finite_nonnegative(configuration->degraded_residual_limit) &&
        finite_nonnegative(configuration->maximum_regularization) &&
        configuration->maximum_regularization <= 1.0e-2f &&
        configuration->max_consecutive_degraded_solves >= 0;
}

typedef struct
{
    MpcRtiCycleStatus_t status;
    MpcRtiNominal_t candidate;
    MpcRtiCandidatePathDelta_t path_delta;
    MpcModelControl_t first_action;
    int iterations;
    float primal_residual;
    float dual_residual;
    float maximum_regularization;
    int regularization_count;
    int nonsmooth_columns;
    int nonlinear_failure_stage;
    float nonlinear_objective;
    float minimum_corridor_slack;
    float lateral_accel_proxy;
    float rho_start;
    float rho_u_start;
    float rho_final;
    float rho_u_final;
    int rho_change_count;
    int factorization_count;
    uint64_t factorization_time_ns;
    double solve_us;
} MpcRtiPassResult_t;

static double monotonic_microseconds(void)
{
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) return 0.0;
    return (double)now.tv_sec * 1.0e6 + (double)now.tv_nsec * 1.0e-3;
}

static int evaluate_nonlinear_candidate(
    const MpcRtiNominal_t *candidate,
    const MpcTrajectorySample_t *trajectory,
    size_t trajectory_count,
    double lap_length,
    const MpcRtiCycleConfiguration_t *configuration,
    float *objective,
    float *minimum_corridor_slack,
    float *lateral_accel_proxy)
{
    if (!candidate || !trajectory || !configuration || !objective ||
        !minimum_corridor_slack || !lateral_accel_proxy ||
        !candidate->valid || candidate->horizon < 1 ||
        candidate->horizon > PREDICTION_HORIZON) return 0;

    double cost = 0.0;
    float min_slack = INFINITY;
    float max_lateral_accel = 0.0f;
    const MpcRtiConfiguration_t *model = &configuration->model;
    for (int k = 0; k <= candidate->horizon; ++k) {
        MpcRtiReference_t reference;
        if (!reference_at_progress(trajectory, trajectory_count, lap_length,
                candidate->progress[k], model, &reference) ||
            !corridor_contains(&candidate->states[k], &reference, model))
            return 0;
        const MpcModelState_t *state = &candidate->states[k].plant;
        const double terminal_scale = k == candidate->horizon
            ? model->terminal_multiplier : 1.0;
        const double errors[] = {
            state->e_y - reference.e_y,
            state->e_psi - reference.e_psi,
            state->u - reference.u,
            state->v - reference.v,
            state->r - reference.r,
            state->steering_command - reference.steering_command};
        const double weights[] = {
            model->weight_e_y, model->weight_e_psi, model->weight_u,
            model->weight_v, model->weight_r,
            model->weight_steering_command};
        for (size_t i = 0; i < sizeof(errors) / sizeof(errors[0]); ++i)
            cost += terminal_scale * weights[i] * errors[i] * errors[i];

        const float left_slack = reference.left_bound -
            model->corridor_margin_m - state->e_y;
        const float right_slack = reference.right_bound -
            model->corridor_margin_m + state->e_y;
        min_slack = fminf(min_slack, fminf(left_slack, right_slack));
        max_lateral_accel = fmaxf(max_lateral_accel,
            fabsf(state->u * state->r));
        if (k == candidate->horizon) continue;

        const MpcModelControl_t *control = &candidate->controls[k];
        const MpcRtiState_t *previous_state = &candidate->states[k];
        const double steer_change = control->steering_rate -
            previous_state->previous_steering_rate;
        const double speed_change = control->target_speed_rate -
            previous_state->previous_target_speed_rate;
        cost += model->weight_steering_rate *
                control->steering_rate * control->steering_rate +
            model->weight_target_speed_rate *
                control->target_speed_rate * control->target_speed_rate +
            model->weight_steering_rate_change * steer_change * steer_change +
            model->weight_target_speed_rate_change * speed_change * speed_change;
    }
    if (!isfinite(cost) || !isfinite(min_slack) ||
        !isfinite(max_lateral_accel)) return 0;
    *objective = (float)cost;
    *minimum_corridor_slack = min_slack;
    *lateral_accel_proxy = max_lateral_accel;
    return 1;
}

static MpcRtiCycleStatus_t solve_rti_pass(
    const MpcRtiState_t *current_state,
    double current_progress,
    const MpcRtiNominal_t *nominal,
    const MpcRtiReference_t references[PREDICTION_HORIZON + 1],
    const MpcTrajectorySample_t *trajectory,
    size_t trajectory_count,
    double lap_length,
    float prediction_dt,
    int horizon,
    const MpcRtiCycleConfiguration_t *configuration,
    MpcRtiMemory_t *memory,
    MpcRtiPassResult_t *pass)
{
    memset(pass, 0, sizeof(*pass));
    pass->status = MPC_RTI_CYCLE_REJECTED_INPUT;
    pass->nonlinear_failure_stage = -1;
    pass->nonlinear_objective = INFINITY;
    pass->minimum_corridor_slack = -INFINITY;
    const double start_us = monotonic_microseconds();
    MpcRtiProblem_t problem;
    if (!mpc_rti_build_ltv_qp(nominal->states, nominal->controls, references,
            horizon, prediction_dt, &configuration->model, &problem)) {
        pass->solve_us = monotonic_microseconds() - start_us;
        return pass->status;
    }
    pass->nonsmooth_columns = problem.nonsmooth_jacobian_columns;

    RiccatiSolution_t solution = {0};
    const float previous_rho = memory->solver_state.initialized
        ? memory->solver_state.rho : configuration->solver.rho;
    const float previous_rho_u = memory->solver_state.initialized
        ? memory->solver_state.rho_u : configuration->solver.rho_u;
    const RiccatiStatus_t solver_status = riccati_admm_solve(
        problem.steps, problem.terminal_Q, problem.terminal_q,
        problem.terminal_x_lb, problem.terminal_x_ub, problem.x0,
        MPC_RTI_NX, MPC_RTI_NU, horizon, &configuration->solver,
        &memory->solver_state, &solution);
    pass->iterations = solution.iterations;
    pass->primal_residual = solution.primal_residual;
    pass->dual_residual = solution.dual_residual;
    RiccatiDebugInfo_t debug = {0};
    riccati_debug_get_last(&debug);
    pass->maximum_regularization = debug.max_control_hessian_regularization;
    pass->regularization_count = debug.control_hessian_regularization_count;
    pass->rho_start = previous_rho;
    pass->rho_u_start = previous_rho_u;
    pass->rho_final = debug.rho;
    pass->rho_u_final = debug.rho_u;
    pass->rho_change_count = debug.rho_change_count;
    pass->factorization_count = debug.quadratic_factorization_count;
    pass->factorization_time_ns = debug.quadratic_factorization_time_ns;
    if (!isfinite(solution.primal_residual) ||
        !isfinite(solution.dual_residual) || solver_status == RICCATI_STATUS_ERROR) {
        pass->status = MPC_RTI_CYCLE_REJECTED_SOLVER;
        pass->solve_us = monotonic_microseconds() - start_us;
        return pass->status;
    }
    if (!isfinite(pass->maximum_regularization) ||
        pass->maximum_regularization > configuration->maximum_regularization) {
        pass->status = MPC_RTI_CYCLE_REJECTED_REGULARIZATION;
        pass->solve_us = monotonic_microseconds() - start_us;
        return pass->status;
    }

    int accepted_degraded = 0;
    if (solver_status == RICCATI_STATUS_MAX_ITERATIONS) {
        const float max_residual = fmaxf(pass->primal_residual,
                                          pass->dual_residual);
        if (max_residual > configuration->degraded_residual_limit ||
            memory->consecutive_degraded_solves + 1 >
                configuration->max_consecutive_degraded_solves) {
            pass->status = MPC_RTI_CYCLE_REJECTED_RESIDUAL;
            pass->solve_us = monotonic_microseconds() - start_us;
            return pass->status;
        }
        accepted_degraded = 1;
    } else if (solver_status != RICCATI_STATUS_OPTIMAL) {
        pass->status = MPC_RTI_CYCLE_REJECTED_SOLVER;
        pass->solve_us = monotonic_microseconds() - start_us;
        return pass->status;
    }

    MpcModelControl_t candidate_controls[PREDICTION_HORIZON] = {0};
    for (int k = 0; k < horizon; ++k) {
        candidate_controls[k].steering_rate = solution.u[k][0];
        candidate_controls[k].target_speed_rate = solution.u[k][1];
    }
    MpcRtiState_t candidate_states[PREDICTION_HORIZON + 1];
    double candidate_progress[PREDICTION_HORIZON + 1];
    const MpcRtiRolloutStatus_t rollout_status = mpc_rti_rollout_candidate(
        current_state, current_progress, candidate_controls, trajectory,
        trajectory_count, lap_length, references, nominal->progress,
        &pass->path_delta, horizon, prediction_dt, &configuration->model,
        candidate_states, candidate_progress, &pass->nonlinear_failure_stage);
    if (rollout_status != MPC_RTI_ROLLOUT_OK) {
        pass->status = MPC_RTI_CYCLE_REJECTED_NONLINEAR_ROLLOUT;
        pass->solve_us = monotonic_microseconds() - start_us;
        return pass->status;
    }

    pass->candidate.valid = 1;
    pass->candidate.horizon = horizon;
    memcpy(pass->candidate.states, candidate_states,
           (size_t)(horizon + 1) * sizeof(candidate_states[0]));
    memcpy(pass->candidate.controls, candidate_controls,
           (size_t)horizon * sizeof(candidate_controls[0]));
    memcpy(pass->candidate.progress, candidate_progress,
           (size_t)(horizon + 1) * sizeof(candidate_progress[0]));
    pass->first_action = candidate_controls[0];
    if (!evaluate_nonlinear_candidate(&pass->candidate, trajectory,
            trajectory_count, lap_length, configuration,
            &pass->nonlinear_objective, &pass->minimum_corridor_slack,
            &pass->lateral_accel_proxy)) {
        pass->status = MPC_RTI_CYCLE_REJECTED_NONLINEAR_ROLLOUT;
        pass->solve_us = monotonic_microseconds() - start_us;
        return pass->status;
    }
    pass->status = accepted_degraded
        ? MPC_RTI_CYCLE_ACCEPTED_DEGRADED
        : MPC_RTI_CYCLE_ACCEPTED_OPTIMAL;
    pass->solve_us = monotonic_microseconds() - start_us;
    return pass->status;
}

static float max_path_delta_double(
    const double values[PREDICTION_HORIZON + 1], int count)
{
    float result = 0.0f;
    for (int i = 0; i < count; ++i)
        result = fmaxf(result, (float)fabs(values[i]));
    return result;
}

static float max_path_delta_float(
    const float values[PREDICTION_HORIZON + 1], int count)
{
    float result = 0.0f;
    for (int i = 0; i < count; ++i)
        result = fmaxf(result, fabsf(values[i]));
    return result;
}

static unsigned int adaptive_rti2_trigger_mask(
    const MpcRtiCycleConfiguration_t *configuration,
    const MpcRtiNominal_t *nominal,
    const MpcRtiPassResult_t *r1)
{
    const int count = r1->path_delta.sample_count;
    const float progress_error = max_path_delta_double(
        r1->path_delta.progress_error_m, count);
    const float curvature_error = max_path_delta_float(
        r1->path_delta.curvature_error_per_m, count);
    const float left_error = max_path_delta_float(
        r1->path_delta.left_bound_error_m, count);
    const float right_error = max_path_delta_float(
        r1->path_delta.right_bound_error_m, count);
    unsigned int mask = 0;
    if (progress_error >= configuration->rti2_progress_error_trigger_m)
        mask |= MPC_RTI2_TRIGGER_PROGRESS;
    if (curvature_error >= configuration->rti2_curvature_error_trigger_per_m)
        mask |= MPC_RTI2_TRIGGER_CURVATURE;
    if (left_error >= configuration->rti2_bound_error_trigger_m)
        mask |= MPC_RTI2_TRIGGER_LEFT_BOUND;
    if (right_error >= configuration->rti2_bound_error_trigger_m)
        mask |= MPC_RTI2_TRIGGER_RIGHT_BOUND;
    if (r1->minimum_corridor_slack <=
        configuration->rti2_min_corridor_slack_trigger_m)
        mask |= MPC_RTI2_TRIGGER_LOW_SLACK;
    if (r1->nonsmooth_columns >=
        configuration->rti2_nonsmooth_columns_trigger)
        mask |= MPC_RTI2_TRIGGER_NONSMOOTH;
    if (fabsf(r1->first_action.steering_rate -
              nominal->controls[0].steering_rate) >=
            configuration->rti2_steering_rate_correction_trigger_radps ||
        fabsf(r1->first_action.target_speed_rate -
              nominal->controls[0].target_speed_rate) >=
            configuration->rti2_target_speed_rate_correction_trigger_mps2)
        mask |= MPC_RTI2_TRIGGER_ACTION_CORRECTION;
    if (r1->status == MPC_RTI_CYCLE_ACCEPTED_DEGRADED)
        mask |= MPC_RTI2_TRIGGER_DEGRADED_SOLVE;
    const float residual_min = fminf(r1->primal_residual, r1->dual_residual);
    const float residual_max = fmaxf(r1->primal_residual, r1->dual_residual);
    if (residual_max / fmaxf(residual_min, 1.0e-6f) >=
        configuration->rti2_residual_imbalance_trigger)
        mask |= MPC_RTI2_TRIGGER_RESIDUAL_IMBALANCE;
    if (nominal->controls[0].steering_rate *
            r1->first_action.steering_rate < 0.0f &&
        fabsf(nominal->controls[0].steering_rate) > 0.1f &&
        fabsf(r1->first_action.steering_rate) > 0.1f)
        mask |= MPC_RTI2_TRIGGER_STEERING_REVERSAL;
    if (r1->lateral_accel_proxy >=
        configuration->rti2_lateral_load_trigger_mps2)
        mask |= MPC_RTI2_TRIGGER_LATERAL_LOAD;
    return mask;
}

static MpcRtiCycleStatus_t reject_cycle(
    MpcRtiMemory_t *memory,
    MpcRtiCycleResult_t *result,
    MpcRtiCycleStatus_t status)
{
    if (memory) mpc_rti_memory_reset(memory);
    if (result) result->status = status;
    return status;
}

MpcRtiCycleStatus_t mpc_rti_solve_cycle(
    const MpcRtiState_t *current_state,
    double current_progress,
    const MpcTrajectorySample_t *trajectory,
    size_t trajectory_count,
    double lap_length,
    float prediction_dt,
    int horizon,
    const MpcRtiCycleConfiguration_t *configuration,
    MpcRtiMemory_t *memory,
    MpcRtiCycleResult_t *result)
{
    if (result) {
        memset(result, 0, sizeof(*result));
        result->status = MPC_RTI_CYCLE_REJECTED_INPUT;
        result->nonlinear_failure_stage = -1;
    }
    if (!result || !memory || !finite_rti_state(current_state) ||
        !isfinite(current_progress) || !trajectory || !trajectory_count ||
        !isfinite(lap_length) || !valid_cycle_configuration(configuration) ||
        horizon < 1 || horizon > PREDICTION_HORIZON ||
        !isfinite(prediction_dt) || prediction_dt <= 0.0f) {
        return reject_cycle(memory, result, MPC_RTI_CYCLE_REJECTED_INPUT);
    }

    MpcRtiNominal_t nominal;
    MpcRtiReference_t references[PREDICTION_HORIZON + 1];
    if (!mpc_rti_build_nominal(current_state, current_progress,
            &memory->nominal, trajectory, trajectory_count, lap_length,
            prediction_dt, horizon, &configuration->model, &nominal,
            references)) {
        return reject_cycle(memory, result, MPC_RTI_CYCLE_REJECTED_INPUT);
    }
    result->nominal_first_control = nominal.controls[0];

    MpcRtiProblem_t problem;
    if (!mpc_rti_build_ltv_qp(nominal.states, nominal.controls, references,
            horizon, prediction_dt, &configuration->model, &problem)) {
        return reject_cycle(memory, result, MPC_RTI_CYCLE_REJECTED_INPUT);
    }
    result->nonsmooth_jacobian_columns = problem.nonsmooth_jacobian_columns;

    RiccatiSolution_t solution = {0};
    const RiccatiStatus_t solver_status = riccati_admm_solve(
        problem.steps, problem.terminal_Q, problem.terminal_q,
        problem.terminal_x_lb, problem.terminal_x_ub, problem.x0,
        MPC_RTI_NX, MPC_RTI_NU, horizon, &configuration->solver,
        &memory->solver_state, &solution);
    result->solver_iterations = solution.iterations;
    result->primal_residual = solution.primal_residual;
    result->dual_residual = solution.dual_residual;
    if (!isfinite(solution.primal_residual) ||
        !isfinite(solution.dual_residual) || solver_status == RICCATI_STATUS_ERROR) {
        return reject_cycle(memory, result, MPC_RTI_CYCLE_REJECTED_SOLVER);
    }

    RiccatiDebugInfo_t debug = {0};
    riccati_debug_get_last(&debug);
    result->maximum_regularization =
        debug.max_control_hessian_regularization;
    result->regularization_count =
        debug.control_hessian_regularization_count;
    if (!isfinite(result->maximum_regularization) ||
        result->maximum_regularization > configuration->maximum_regularization) {
        return reject_cycle(memory, result,
                            MPC_RTI_CYCLE_REJECTED_REGULARIZATION);
    }

    int accepted_degraded = 0;
    if (solver_status == RICCATI_STATUS_MAX_ITERATIONS) {
        const float max_residual = fmaxf(result->primal_residual,
                                          result->dual_residual);
        if (max_residual > configuration->degraded_residual_limit ||
            memory->consecutive_degraded_solves + 1 >
                configuration->max_consecutive_degraded_solves) {
            return reject_cycle(memory, result,
                                MPC_RTI_CYCLE_REJECTED_RESIDUAL);
        }
        accepted_degraded = 1;
    } else if (solver_status != RICCATI_STATUS_OPTIMAL) {
        return reject_cycle(memory, result, MPC_RTI_CYCLE_REJECTED_SOLVER);
    }

    MpcModelControl_t candidate_controls[PREDICTION_HORIZON] = {0};
    for (int k = 0; k < horizon; ++k) {
        candidate_controls[k].steering_rate = solution.u[k][0];
        candidate_controls[k].target_speed_rate = solution.u[k][1];
    }
    MpcRtiState_t candidate_states[PREDICTION_HORIZON + 1];
    double candidate_progress[PREDICTION_HORIZON + 1];
    int nonlinear_failure_stage = -1;
    const MpcRtiRolloutStatus_t rollout_status = mpc_rti_rollout_candidate(
        current_state, current_progress, candidate_controls, trajectory,
        trajectory_count, lap_length, references, nominal.progress,
        &result->candidate_path_delta, horizon, prediction_dt,
        &configuration->model, candidate_states, candidate_progress,
        &nonlinear_failure_stage);
    result->nonlinear_failure_stage = nonlinear_failure_stage;
    if (rollout_status != MPC_RTI_ROLLOUT_OK) {
        return reject_cycle(memory, result,
                            MPC_RTI_CYCLE_REJECTED_NONLINEAR_ROLLOUT);
    }

    /* Command targets come from the same nonlinear first-stage update used
     * for prediction, never from callback/source arrival spacing. */
    result->first_control = candidate_controls[0];
    result->published_steering_command =
        candidate_states[1].plant.steering_command;
    result->published_target_speed = candidate_states[1].plant.target_speed;
    result->status = accepted_degraded
        ? MPC_RTI_CYCLE_ACCEPTED_DEGRADED
        : MPC_RTI_CYCLE_ACCEPTED_OPTIMAL;

    memory->nominal.valid = 1;
    memory->nominal.horizon = horizon;
    memcpy(memory->nominal.states, candidate_states,
           (size_t)(horizon + 1) * sizeof(candidate_states[0]));
    memcpy(memory->nominal.controls, candidate_controls,
           (size_t)horizon * sizeof(candidate_controls[0]));
    memcpy(memory->nominal.progress, candidate_progress,
           (size_t)(horizon + 1) * sizeof(candidate_progress[0]));
    memory->consecutive_degraded_solves = accepted_degraded
        ? memory->consecutive_degraded_solves + 1 : 0;
    return result->status;
}

float mpc_rti_matrix_stage_cost(
    const RiccatiStepData_t *stage,
    const float state[MPC_RTI_NX],
    const float control[MPC_RTI_NU])
{
    if (!stage || !state || !control) return NAN;
    float cost = 0.0f;
    for (int i = 0; i < MPC_RTI_NX; ++i) {
        cost += 0.5f * stage->Q_diag[i] * state[i] * state[i] +
            stage->q[i] * state[i];
        for (int input = 0; input < MPC_RTI_NU; ++input)
            cost += state[i] * stage->N[i][input] * control[input];
    }
    for (int input = 0; input < MPC_RTI_NU; ++input)
        cost += 0.5f * stage->R_diag[input] * control[input] * control[input] +
            stage->r[input] * control[input];
    return cost;
}

float mpc_rti_scalar_stage_cost(
    const MpcRtiConfiguration_t *configuration,
    const MpcRtiReference_t *reference,
    const MpcRtiState_t *state,
    const MpcModelControl_t *control)
{
    if (!valid_configuration(configuration) || !finite_reference(reference) ||
        !finite_rti_state(state) || !control ||
        !isfinite(control->steering_rate) ||
        !isfinite(control->target_speed_rate)) return NAN;
    const float ey = state->plant.e_y - reference->e_y;
    const float epsi = state->plant.e_psi - reference->e_psi;
    const float eu = state->plant.u - reference->u;
    const float ev = state->plant.v - reference->v;
    const float er = state->plant.r - reference->r;
    const float edelta = state->plant.steering_command -
        reference->steering_command;
    const float dq_delta = control->steering_rate -
        state->previous_steering_rate;
    const float dq_speed = control->target_speed_rate -
        state->previous_target_speed_rate;
    return configuration->weight_e_y * ey * ey +
        configuration->weight_e_psi * epsi * epsi +
        configuration->weight_u * eu * eu +
        configuration->weight_v * ev * ev +
        configuration->weight_r * er * er +
        configuration->weight_steering_command * edelta * edelta +
        configuration->weight_steering_rate *
            control->steering_rate * control->steering_rate +
        configuration->weight_target_speed_rate *
            control->target_speed_rate * control->target_speed_rate +
        configuration->weight_steering_rate_change * dq_delta * dq_delta +
        configuration->weight_target_speed_rate_change * dq_speed * dq_speed;
}
