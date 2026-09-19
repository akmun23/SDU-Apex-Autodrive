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
        isfinite(reference->target_speed) &&
        isfinite(reference->target_speed_rate) &&
        isfinite(reference->path_curvature) &&
        isfinite(reference->left_bound) && reference->left_bound > 0.0f &&
        isfinite(reference->right_bound) && reference->right_bound > 0.0f;
}

static int valid_configuration(const MpcRtiConfiguration_t *configuration)
{
    if (!configuration) return 0;
    if (configuration->recovery_seed_policy <
            MPC_RTI_RECOVERY_SEED_NOMINAL ||
        configuration->recovery_seed_policy >
            MPC_RTI_RECOVERY_SEED_BRAKE_HEADING_FEEDBACK ||
        !finite_nonnegative(configuration->recovery_steering_k_e_y) ||
        !finite_nonnegative(configuration->recovery_steering_k_e_psi) ||
        !finite_nonnegative(configuration->recovery_steering_k_r)) return 0;
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
        configuration->weight_target_speed_state,
        configuration->weight_v,
        configuration->weight_r,
        configuration->weight_steering_command,
        configuration->weight_steering_rate,
        configuration->weight_target_speed_rate,
        configuration->weight_steering_rate_change,
        configuration->weight_target_speed_rate_change,
        configuration->terminal_multiplier,
        configuration->corridor_margin_m,
        configuration->first_prediction_corridor_margin_m,
        configuration->planning_half_width_m,
        configuration->vehicle_half_width_m,
        configuration->vehicle_longitudinal_extent_m,
        configuration->wall_clearance_m,
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
            vehicle.maximum_target_speed_rate_reduction_mps2 &&
        configuration->first_prediction_corridor_margin_m <=
            configuration->corridor_margin_m &&
        configuration->planning_half_width_m > 0.0f &&
        configuration->vehicle_half_width_m > 0.0f &&
        configuration->vehicle_longitudinal_extent_m > 0.0f &&
        configuration->wall_clearance_m >= 0.0f;
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

static float target_speed_feedforward(
    const MpcTrajectorySample_t *sample,
    const MpcRtiConfiguration_t *configuration,
    float prediction_dt)
{
    /* The Unity adapter consumes a target speed, while the identified plant
     * responds to target-speed error and target-speed slew.  Invert that
     * source-command model locally instead of allowing the target state to
     * float toward the global command ceiling. */
    const float reference_speed = (float)fmin(
        sample->speed, configuration->active_speed_ceiling_mps);
    const float reference_rate = clampf_rti(
        (float)sample->acceleration,
        -configuration->max_target_speed_rate_reduction_mps2,
        configuration->max_target_speed_rate_increase_mps2);
    const float target_speed = reference_speed +
        (reference_rate - MPC_LONGITUDINAL_RESPONSE_BIAS_MPS2 -
         MPC_LONGITUDINAL_SPEED_COEFF_PER_S * reference_speed -
         MPC_LONGITUDINAL_TARGET_RATE_COEFF * reference_rate) /
        MPC_LONGITUDINAL_TARGET_ERROR_GAIN_PER_S -
        0.5f * reference_rate * prediction_dt;
    return clampf_rti(target_speed, 0.0f,
        configuration->active_speed_ceiling_mps);
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
        .target_speed = target_speed_feedforward(
            &sample, configuration, PREDICTION_DT_SECONDS),
        .target_speed_rate = clampf_rti(
            (float)sample.acceleration,
            -configuration->max_target_speed_rate_reduction_mps2,
            configuration->max_target_speed_rate_increase_mps2),
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
    const MpcRtiConfiguration_t *configuration,
    float corridor_margin_m)
{
    const float lower = corridor_margin_m -
        reference->right_bound -
        configuration->nonlinear_corridor_tolerance_m;
    const float upper = reference->left_bound -
        corridor_margin_m +
        configuration->nonlinear_corridor_tolerance_m;
    return isfinite(lower) && isfinite(upper) && lower <= upper &&
        state->plant.e_y >= lower && state->plant.e_y <= upper;
}

static float corridor_margin_at_prediction(
    const MpcRtiConfiguration_t *configuration,
    int prediction_index)
{
    return prediction_index == 1
        ? configuration->first_prediction_corridor_margin_m
        : configuration->corridor_margin_m;
}

static float heading_aware_corridor_margin(
    const MpcRtiConfiguration_t *configuration,
    float e_psi,
    int prediction_index)
{
    if (!configuration || !isfinite(e_psi)) return INFINITY;
    const float physical_half_extent =
        configuration->vehicle_half_width_m * fabsf(cosf(e_psi)) +
        configuration->vehicle_longitudinal_extent_m * fabsf(sinf(e_psi));
    const float lateral_extent = fmaxf(
        configuration->planning_half_width_m, physical_half_extent);
    const float footprint_margin =
        configuration->wall_clearance_m + lateral_extent;
    return fmaxf(
        corridor_margin_at_prediction(configuration, prediction_index),
        footprint_margin);
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
    const MpcRtiConfiguration_t *configuration,
    float corridor_margin_m)
{
    const float ey_lower = corridor_margin_m - reference->right_bound;
    const float ey_upper = reference->left_bound - corridor_margin_m;
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

static int set_state_bounds_with_ey(
    float lower[MPC_RTI_NX],
    float upper[MPC_RTI_NX],
    const MpcRtiConfiguration_t *configuration,
    float ey_lower,
    float ey_upper)
{
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
    const float target_speed = reference->target_speed;
    const float weights[] = {
        configuration->weight_e_y, configuration->weight_e_psi,
        configuration->weight_u, configuration->weight_v,
        configuration->weight_r, configuration->weight_steering_command,
    };
    for (int i = 0; i < 6; ++i) {
        const int index = indices[i];
        const float scaled_weight = multiplier * weights[i];
        Q[index] += 2.0f * scaled_weight;
        q[index] -= 2.0f * scaled_weight * targets[i];
    }
    const float target_speed_weight =
        multiplier * configuration->weight_target_speed_state;
    Q[MPC_RTI_IDX_TARGET_SPEED] += 2.0f * target_speed_weight;
    q[MPC_RTI_IDX_TARGET_SPEED] -= 2.0f * target_speed_weight * target_speed;
}

int mpc_rti_build_ltv_qp_with_schedule(
    const MpcRtiState_t nominal_states[PREDICTION_HORIZON + 1],
    const MpcModelControl_t nominal_controls[PREDICTION_HORIZON],
    const MpcRtiReference_t references[PREDICTION_HORIZON + 1],
    int horizon,
    float prediction_dt,
    const MpcRtiConfiguration_t *configuration,
    const MpcRtiCorridorSchedule_t *schedule,
    MpcRtiProblem_t *problem)
{
    if (!nominal_states || !nominal_controls || !references || !problem ||
        !valid_configuration(configuration) || horizon < 1 ||
        horizon > PREDICTION_HORIZON || !isfinite(prediction_dt) ||
        prediction_dt <= 0.0f) {
        return 0;
    }
    if (!vehicle_model_set_active_target_speed_ceiling(
            configuration->active_speed_ceiling_mps)) return 0;
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

        if (schedule && schedule->horizon == horizon) {
            if (!set_state_bounds_with_ey(stage->x_lb, stage->x_ub,
                    configuration,
                    schedule->active_lower[k], schedule->active_upper[k]))
                return 0;
        } else if (!set_state_bounds(stage->x_lb, stage->x_ub, &references[k],
                configuration, corridor_margin_at_prediction(configuration, k)))
            return 0;
        stage->u_lb[0] = -configuration->max_steering_rate_radps;
        stage->u_ub[0] = configuration->max_steering_rate_radps;
        stage->u_lb[1] = -configuration->max_target_speed_rate_reduction_mps2;
        stage->u_ub[1] = configuration->max_target_speed_rate_increase_mps2;
    }

    add_tracking_cost(problem->terminal_Q, problem->terminal_q,
                      &references[horizon], configuration,
                      configuration->terminal_multiplier);
    if (schedule && schedule->horizon == horizon) {
        if (!set_state_bounds_with_ey(problem->terminal_x_lb,
                problem->terminal_x_ub, configuration,
                schedule->active_lower[horizon],
                schedule->active_upper[horizon])) return 0;
    } else {
        const float terminal_margin = horizon == 1
            ? configuration->first_prediction_corridor_margin_m
            : configuration->corridor_margin_m;
        if (!set_state_bounds(problem->terminal_x_lb, problem->terminal_x_ub,
                &references[horizon], configuration, terminal_margin)) return 0;
    }
    return 1;
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
    return mpc_rti_build_ltv_qp_with_schedule(
        nominal_states, nominal_controls, references, horizon, prediction_dt,
        configuration, NULL, problem);
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
    if (!vehicle_model_set_active_target_speed_ceiling(
            configuration->active_speed_ceiling_mps)) return 0;

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
            const float desired_target = references[k + 1].target_speed;
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

static float corridor_bound_violation(float value, float lower, float upper)
{
    if (!isfinite(value) || !isfinite(lower) || !isfinite(upper) ||
        lower > upper) return INFINITY;
    return fmaxf(fmaxf(lower - value, value - upper), 0.0f);
}

static int fill_corridor_schedule_for_seed(
    const MpcRtiNominal_t *seed,
    const MpcTrajectorySample_t *trajectory,
    size_t trajectory_count,
    double lap_length,
    const MpcRtiConfiguration_t *configuration,
    MpcRtiCorridorSchedule_t *schedule)
{
    if (!seed || !seed->valid || !trajectory || !configuration || !schedule)
        return 0;

    schedule->horizon = seed->horizon;
    schedule->first_normal_feasible_stage = -1;
    schedule->initial_normal_violation = 0.0f;
    schedule->max_seed_violation = 0.0f;
    for (int k = 0; k <= seed->horizon; ++k) {
        MpcRtiReference_t reference;
        if (!reference_at_progress(trajectory, trajectory_count, lap_length,
                seed->progress[k], configuration, &reference)) return 0;
        const float margin = heading_aware_corridor_margin(
            configuration, seed->states[k].plant.e_psi, k);
        schedule->normal_lower[k] = margin - reference.right_bound;
        schedule->normal_upper[k] = reference.left_bound - margin;
        if (!isfinite(schedule->normal_lower[k]) ||
            !isfinite(schedule->normal_upper[k]) ||
            schedule->normal_lower[k] > schedule->normal_upper[k]) return 0;
        schedule->seed_e_y[k] = seed->states[k].plant.e_y;
        schedule->seed_violation[k] = corridor_bound_violation(
            schedule->seed_e_y[k], schedule->normal_lower[k],
            schedule->normal_upper[k]);
        schedule->max_seed_violation = fmaxf(schedule->max_seed_violation,
            schedule->seed_violation[k]);
        schedule->active_lower[k] = schedule->normal_lower[k];
        schedule->active_upper[k] = schedule->normal_upper[k];
        schedule->recovery_stage[k] = 0;
    }
    schedule->initial_normal_violation = schedule->seed_violation[0];

    int any_violation = 0;
    for (int k = 0; k <= seed->horizon; ++k) {
        if (schedule->seed_violation[k] > 1.0e-6f) any_violation = 1;
        if (k > 0 && schedule->seed_violation[k] <= 1.0e-6f &&
            schedule->first_normal_feasible_stage < 0)
            schedule->first_normal_feasible_stage = k;
    }
    if (!any_violation) return 1;
    if (schedule->first_normal_feasible_stage < 0) {
        schedule->recovery_not_found = 1;
        return 1;
    }

    const float epsilon = 1.0e-3f;
    for (int k = 0; k <= seed->horizon; ++k) {
        const int temporary = k < schedule->first_normal_feasible_stage &&
            schedule->seed_violation[k] > 1.0e-6f;
        schedule->recovery_stage[k] = temporary ? 1 : 0;
        if (!temporary) continue;
        if (schedule->seed_e_y[k] < schedule->normal_lower[k]) {
            schedule->active_lower[k] = schedule->seed_e_y[k] - epsilon;
            schedule->active_upper[k] = schedule->normal_upper[k];
        } else if (schedule->seed_e_y[k] > schedule->normal_upper[k]) {
            schedule->active_lower[k] = schedule->normal_lower[k];
            schedule->active_upper[k] = schedule->seed_e_y[k] + epsilon;
        }
        if (!isfinite(schedule->active_lower[k]) ||
            !isfinite(schedule->active_upper[k]) ||
            schedule->active_lower[k] > schedule->active_upper[k]) return 0;
    }
    schedule->recovery_active = 1;
    return 1;
}

/* Evaluate one bounded recovery policy with the same exact virtual-car
 * model used by the nominal rollout.  It is intentionally only called after
 * nominal feasibility has failed; ordinary operation therefore retains exact
 * nominal parity. */
static int build_recovery_policy_seed(
    const MpcRtiState_t *current_state,
    const MpcRtiNominal_t *nominal,
    const MpcTrajectorySample_t *trajectory,
    size_t trajectory_count,
    double lap_length,
    float prediction_dt,
    const MpcRtiConfiguration_t *configuration,
    MpcRtiNominal_t *seed)
{
    if (!current_state || !nominal || !trajectory || !configuration || !seed ||
        configuration->recovery_seed_policy == MPC_RTI_RECOVERY_SEED_NOMINAL)
        return 0;

    *seed = *nominal;
    seed->states[0] = *current_state;
    seed->progress[0] = nominal->progress[0];
    for (int k = 0; k < nominal->horizon; ++k) {
        MpcRtiReference_t reference;
        if (!reference_at_progress(trajectory, trajectory_count, lap_length,
                seed->progress[k], configuration, &reference)) return 0;

        MpcModelControl_t control = nominal->controls[k];
        const float desired_steering = clampf_rti(
            reference.steering_command -
                configuration->recovery_steering_k_e_y *
                    seed->states[k].plant.e_y -
                configuration->recovery_steering_k_e_psi *
                    seed->states[k].plant.e_psi -
                configuration->recovery_steering_k_r *
                    (seed->states[k].plant.r - reference.r),
            -configuration->max_steering_rad,
            configuration->max_steering_rad);
        control.steering_rate = clampf_rti(
            (desired_steering - seed->states[k].plant.steering_command) /
                prediction_dt,
            -configuration->max_steering_rate_radps,
            configuration->max_steering_rate_radps);
        if (configuration->recovery_seed_policy ==
            MPC_RTI_RECOVERY_SEED_BRAKE_HEADING_FEEDBACK) {
            // This policy models the simulator's discrete no-throttle action:
            // full commanded braking is the bounded recovery alternative.
            control.target_speed_rate =
                -configuration->max_target_speed_rate_reduction_mps2;
        } else {
            control.target_speed_rate = clampf_rti(
                control.target_speed_rate,
                -configuration->max_target_speed_rate_reduction_mps2,
                configuration->max_target_speed_rate_increase_mps2);
        }

        const MpcStageResult_t step = mpc_vehicle_model_step(
            &seed->states[k].plant, &control, prediction_dt,
            reference.path_curvature);
        if (!step.valid || !isfinite(seed->progress[k] + step.delta_s_m))
            return 0;
        seed->controls[k] = control;
        seed->states[k + 1].plant = step.next;
        seed->states[k + 1].previous_steering_rate = control.steering_rate;
        seed->states[k + 1].previous_target_speed_rate =
            control.target_speed_rate;
        seed->progress[k + 1] = seed->progress[k] + step.delta_s_m;
        if (!state_inside_command_envelope(&seed->states[k + 1], configuration))
            return 0;
    }
    seed->valid = 1;
    return 1;
}

/* Build the one schedule consumed by both the QP and the exact rollout.
 * A nominal rollout is the default deterministic bounded recovery seed.  If
 * configured, one bounded heading-feedback or braking-feedback policy is
 * evaluated only after nominal feasibility fails. This is intentionally a
 * single O(N) policy evaluation, not an online beam search. If that seed
 * cannot re-enter the normal corridor within N, no temporary envelope is
 * enabled and the caller reports recovery_not_found. */
int mpc_rti_build_corridor_schedule(
    const MpcRtiState_t *current_state,
    const MpcRtiNominal_t *nominal,
    const MpcTrajectorySample_t *trajectory,
    size_t trajectory_count,
    double lap_length,
    float prediction_dt,
    const MpcRtiConfiguration_t *configuration,
    MpcRtiCorridorSchedule_t *schedule)
{
    if (!finite_rti_state(current_state) || !nominal || !trajectory ||
        !configuration || !schedule || !nominal->valid || nominal->horizon < 1 ||
        nominal->horizon > PREDICTION_HORIZON || !isfinite(prediction_dt) ||
        prediction_dt <= 0.0f) return 0;

    memset(schedule, 0, sizeof(*schedule));
    MpcRtiNominal_t seed = *nominal;
    if (!fill_corridor_schedule_for_seed(&seed, trajectory, trajectory_count,
            lap_length, configuration, schedule)) return 0;

    if (schedule->max_seed_violation <= 1.0e-6f ||
        configuration->recovery_seed_policy == MPC_RTI_RECOVERY_SEED_NOMINAL)
        return 1;

    MpcRtiNominal_t policy_seed;
    if (build_recovery_policy_seed(current_state, nominal, trajectory,
            trajectory_count, lap_length, prediction_dt, configuration,
            &policy_seed)) {
        memset(schedule, 0, sizeof(*schedule));
        seed = policy_seed;
        if (!fill_corridor_schedule_for_seed(&seed, trajectory,
                trajectory_count, lap_length, configuration, schedule)) return 0;
    }
    return 1;
}

MpcRtiRolloutStatus_t mpc_rti_rollout_candidate_with_schedule(
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
    const MpcRtiCorridorSchedule_t *schedule,
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
    if (!vehicle_model_set_active_target_speed_ceiling(
            configuration->active_speed_ceiling_mps))
        return MPC_RTI_ROLLOUT_INVALID_INPUT;
    states[0] = *initial_state;
    if (!state_inside_command_envelope(&states[0], configuration))
        return MPC_RTI_ROLLOUT_INVALID_INPUT;
    double candidate_progress[PREDICTION_HORIZON + 1] = {0.0};
    candidate_progress[0] = initial_progress;
    MpcRtiReference_t candidate_reference = {0};
    if (!reference_at_progress(trajectory, trajectory_count, lap_length,
            candidate_progress[0], configuration, &candidate_reference))
        return MPC_RTI_ROLLOUT_INVALID_INPUT;
    /* x0 is measured and immutable. The QP applies corridor constraints only
     * to x1..xN; rejecting an already-off-center x0 here would disagree with
     * the solved feasible set and can publish no recovery action. */
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
        int in_corridor = 0;
        if (schedule && schedule->recovery_active &&
            schedule->horizon == horizon) {
            const float lower = schedule->active_lower[k + 1];
            const float upper = schedule->active_upper[k + 1];
            in_corridor = isfinite(lower) && isfinite(upper) &&
                lower <= upper && states[k + 1].plant.e_y >= lower &&
                states[k + 1].plant.e_y <= upper;
        } else {
            in_corridor = corridor_contains(&states[k + 1],
                &candidate_reference, configuration,
                corridor_margin_at_prediction(configuration, k + 1));
        }
        if (!in_corridor) {
            if (failure_stage) *failure_stage = k;
            return MPC_RTI_ROLLOUT_CORRIDOR;
        }
    }
    return MPC_RTI_ROLLOUT_OK;
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
    return mpc_rti_rollout_candidate_with_schedule(
        initial_state, initial_progress, controls, trajectory,
        trajectory_count, lap_length, nominal_references, nominal_progress,
        path_delta, horizon, prediction_dt, configuration, NULL, states,
        progress, failure_stage);
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
        (configuration->refinement_mode != MPC_RTI_REFINEMENT_ADAPTIVE ||
         (isfinite(configuration->rti2_residual_imbalance_trigger) &&
          configuration->rti2_residual_imbalance_trigger >= 1.0f)) &&
        finite_nonnegative(configuration->rti2_lateral_load_trigger_mps2) &&
        configuration->rti2_nonsmooth_columns_trigger >= 0 &&
        finite_nonnegative(configuration->rti2_residual_recovery_limit) &&
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
    int nonlinear_failure_reason;
    float nonlinear_objective;
    float minimum_corridor_slack;
    float lateral_accel_proxy;
    int lateral_accel_proxy_stage;
    float lateral_accel_proxy_by_stage[PREDICTION_HORIZON + 1];
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
    const MpcRtiCorridorSchedule_t *schedule,
    float *objective,
    float *minimum_corridor_slack,
    float *lateral_accel_proxy,
    int *lateral_accel_proxy_stage,
    float lateral_accel_proxy_by_stage[PREDICTION_HORIZON + 1])
{
    if (!candidate || !trajectory || !configuration || !objective ||
        !minimum_corridor_slack || !lateral_accel_proxy ||
        !lateral_accel_proxy_stage || !lateral_accel_proxy_by_stage ||
        !candidate->valid || candidate->horizon < 1 ||
        candidate->horizon > PREDICTION_HORIZON) return 0;

    double cost = 0.0;
    float min_slack = INFINITY;
    float max_lateral_accel = 0.0f;
    const MpcRtiConfiguration_t *model = &configuration->model;
    for (int k = 0; k <= candidate->horizon; ++k) {
        MpcRtiReference_t reference;
        if (!reference_at_progress(trajectory, trajectory_count, lap_length,
                candidate->progress[k], model, &reference)) return 0;
        if (k > 0) {
            int in_corridor = 0;
            if (schedule && schedule->recovery_active &&
                schedule->horizon == candidate->horizon) {
                const float lower = schedule->active_lower[k];
                const float upper = schedule->active_upper[k];
                in_corridor = isfinite(lower) && isfinite(upper) &&
                    lower <= upper && candidate->states[k].plant.e_y >= lower &&
                    candidate->states[k].plant.e_y <= upper;
            } else {
                in_corridor = corridor_contains(&candidate->states[k],
                    &reference, model, corridor_margin_at_prediction(model, k));
            }
            if (!in_corridor) return 0;
        }
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

        if (k > 0) {
            float left_slack;
            float right_slack;
            if (schedule && schedule->recovery_active &&
                schedule->horizon == candidate->horizon) {
                left_slack = schedule->active_upper[k] - state->e_y;
                right_slack = state->e_y - schedule->active_lower[k];
            } else {
                left_slack = reference.left_bound -
                    corridor_margin_at_prediction(model, k) - state->e_y;
                right_slack = reference.right_bound -
                    corridor_margin_at_prediction(model, k) + state->e_y;
            }
            min_slack = fminf(min_slack, fminf(left_slack, right_slack));
        }
        lateral_accel_proxy_by_stage[k] = fabsf(state->u * state->r);
        if (lateral_accel_proxy_by_stage[k] > max_lateral_accel) {
            max_lateral_accel = lateral_accel_proxy_by_stage[k];
            *lateral_accel_proxy_stage = k;
        }
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
    const MpcRtiCorridorSchedule_t *schedule,
    MpcRtiMemory_t *memory,
    MpcRtiPassResult_t *pass)
{
    memset(pass, 0, sizeof(*pass));
    pass->status = MPC_RTI_CYCLE_REJECTED_INPUT;
    pass->nonlinear_failure_stage = -1;
    pass->nonlinear_failure_reason = -1;
    pass->nonlinear_objective = INFINITY;
    pass->minimum_corridor_slack = -INFINITY;
    const double start_us = monotonic_microseconds();
    MpcRtiProblem_t problem;
    if (!mpc_rti_build_ltv_qp_with_schedule(nominal->states,
            nominal->controls, references, horizon, prediction_dt,
            &configuration->model, schedule, &problem)) {
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
    int residual_recovery_candidate = 0;
    if (solver_status == RICCATI_STATUS_MAX_ITERATIONS) {
        const float max_residual = fmaxf(pass->primal_residual,
                                          pass->dual_residual);
        if (max_residual > configuration->degraded_residual_limit ||
            memory->consecutive_degraded_solves + 1 >
                configuration->max_consecutive_degraded_solves) {
            /* In adaptive/R2 mode, keep a bounded projected solution long
             * enough to validate its exact nonlinear rollout and use it as a
             * same-cycle R2 seed. It is never accepted or published as R1.
             * R1-only mode keeps the strict residual gate. */
            if (configuration->refinement_mode == MPC_RTI_REFINEMENT_R1 ||
                configuration->rti2_residual_recovery_limit <= 0.0f ||
                max_residual > configuration->rti2_residual_recovery_limit) {
                pass->status = MPC_RTI_CYCLE_REJECTED_RESIDUAL;
                pass->solve_us = monotonic_microseconds() - start_us;
                return pass->status;
            }
            residual_recovery_candidate = 1;
        } else {
            accepted_degraded = 1;
        }
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
    pass->candidate.horizon = horizon;
    memcpy(pass->candidate.controls, candidate_controls,
           (size_t)horizon * sizeof(candidate_controls[0]));
    pass->first_action = candidate_controls[0];
    MpcRtiState_t candidate_states[PREDICTION_HORIZON + 1];
    double candidate_progress[PREDICTION_HORIZON + 1];
    const MpcRtiRolloutStatus_t rollout_status =
        mpc_rti_rollout_candidate_with_schedule(
        current_state, current_progress, candidate_controls, trajectory,
        trajectory_count, lap_length, references, nominal->progress,
        &pass->path_delta, horizon, prediction_dt, &configuration->model,
        schedule, candidate_states, candidate_progress,
        &pass->nonlinear_failure_stage);
    if (rollout_status != MPC_RTI_ROLLOUT_OK) {
        pass->status = MPC_RTI_CYCLE_REJECTED_NONLINEAR_ROLLOUT;
        pass->nonlinear_failure_reason = (int)rollout_status;
        pass->solve_us = monotonic_microseconds() - start_us;
        return pass->status;
    }

    pass->candidate.valid = 1;
    memcpy(pass->candidate.states, candidate_states,
           (size_t)(horizon + 1) * sizeof(candidate_states[0]));
    memcpy(pass->candidate.progress, candidate_progress,
           (size_t)(horizon + 1) * sizeof(candidate_progress[0]));
    if (!evaluate_nonlinear_candidate(&pass->candidate, trajectory,
            trajectory_count, lap_length, configuration, schedule,
            &pass->nonlinear_objective, &pass->minimum_corridor_slack,
            &pass->lateral_accel_proxy, &pass->lateral_accel_proxy_stage,
            pass->lateral_accel_proxy_by_stage)) {
        pass->status = MPC_RTI_CYCLE_REJECTED_NONLINEAR_ROLLOUT;
        pass->nonlinear_failure_reason = MPC_RTI_ROLLOUT_INVALID_MODEL;
        pass->solve_us = monotonic_microseconds() - start_us;
        return pass->status;
    }
    pass->status = residual_recovery_candidate
        ? MPC_RTI_CYCLE_REJECTED_RESIDUAL
        : accepted_degraded
        ? MPC_RTI_CYCLE_ACCEPTED_DEGRADED
        : MPC_RTI_CYCLE_ACCEPTED_OPTIMAL;
    pass->solve_us = monotonic_microseconds() - start_us;
    return pass->status;
}

/* A first-pass nonlinear rollout can leave the corridor at a predicted
 * stage even though the measured state is still inside it. In that one case,
 * use the exact model rollout of R1's bounded QP controls as an R2
 * linearization seed. This seed is deliberately not accepted or published:
 * R2 must still pass the normal exact nonlinear corridor gate. */
static int build_rti_corridor_repair_seed(
    const MpcRtiState_t *current_state,
    double current_progress,
    const MpcModelControl_t controls[PREDICTION_HORIZON],
    const MpcTrajectorySample_t *trajectory,
    size_t trajectory_count,
    double lap_length,
    float prediction_dt,
    int horizon,
    const MpcRtiConfiguration_t *configuration,
    MpcRtiNominal_t *seed)
{
    if (!current_state || !controls || !trajectory || !configuration ||
        !seed || !finite_rti_state(current_state) ||
        !isfinite(current_progress) || horizon < 1 ||
        horizon > PREDICTION_HORIZON || !isfinite(prediction_dt) ||
        prediction_dt <= 0.0f || !state_inside_command_envelope(
            current_state, configuration)) return 0;

    memset(seed, 0, sizeof(*seed));
    seed->horizon = horizon;
    seed->states[0] = *current_state;
    seed->progress[0] = current_progress;
    MpcRtiReference_t reference;
    if (!reference_at_progress(trajectory, trajectory_count, lap_length,
            current_progress, configuration, &reference))
        return 0;

    for (int k = 0; k < horizon; ++k) {
        const MpcModelControl_t *control = &controls[k];
        if (!isfinite(control->steering_rate) ||
            !isfinite(control->target_speed_rate) ||
            control->steering_rate <
                -configuration->max_steering_rate_radps - 1.0e-5f ||
            control->steering_rate >
                configuration->max_steering_rate_radps + 1.0e-5f ||
            control->target_speed_rate <
                -configuration->max_target_speed_rate_reduction_mps2 -
                    1.0e-5f ||
            control->target_speed_rate >
                configuration->max_target_speed_rate_increase_mps2 +
                    1.0e-5f ||
            !reference_at_progress(trajectory, trajectory_count, lap_length,
                seed->progress[k], configuration, &reference))
            return 0;

        const MpcStageResult_t step = mpc_vehicle_model_step(
            &seed->states[k].plant, control, prediction_dt,
            reference.path_curvature);
        if (!step.valid) return 0;
        seed->states[k + 1].plant = step.next;
        seed->states[k + 1].previous_steering_rate =
            control->steering_rate;
        seed->states[k + 1].previous_target_speed_rate =
            control->target_speed_rate;
        seed->progress[k + 1] = seed->progress[k] + step.delta_s_m;
        if (!isfinite(seed->progress[k + 1]) ||
            !state_inside_command_envelope(&seed->states[k + 1],
                configuration) ||
            !reference_at_progress(trajectory, trajectory_count, lap_length,
                seed->progress[k + 1], configuration, &reference))
            return 0;
        seed->controls[k] = *control;
    }
    seed->valid = 1;
    return 1;
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
        result->r1_status = -1;
        result->r2_status = -1;
        result->nonlinear_failure_reason = -1;
        result->r1_nonlinear_failure_reason = -1;
        result->r2_nonlinear_failure_reason = -1;
        result->r1_nonlinear_failure_stage = -1;
        result->r2_nonlinear_failure_stage = -1;
        result->r1_nonlinear_objective = NAN;
        result->r2_nonlinear_objective = NAN;
        result->r1_min_corridor_slack = NAN;
        result->r2_min_corridor_slack = NAN;
        result->minimum_predicted_corridor_slack_m = NAN;
        result->recovery_reentry_stage = -1;
        result->recovery_initial_normal_violation_m = NAN;
        result->recovery_max_seed_violation_m = NAN;
        result->lateral_accel_proxy_mps2 = NAN;
        result->lateral_accel_proxy_stage = -1;
        for (int k = 0; k <= PREDICTION_HORIZON; ++k)
            result->lateral_accel_proxy_by_stage_mps2[k] = NAN;
    }
    if (!result || !memory || !finite_rti_state(current_state) ||
        !isfinite(current_progress) || !trajectory || !trajectory_count ||
        !isfinite(lap_length) || !valid_cycle_configuration(configuration) ||
        horizon < 1 || horizon > PREDICTION_HORIZON ||
        !isfinite(prediction_dt) || prediction_dt <= 0.0f) {
        return reject_cycle(memory, result, MPC_RTI_CYCLE_REJECTED_INPUT);
    }

    const double cycle_start_us = monotonic_microseconds();
    MpcRtiNominal_t nominal;
    MpcRtiReference_t references[PREDICTION_HORIZON + 1];
    if (!mpc_rti_build_nominal(current_state, current_progress,
            &memory->nominal, trajectory, trajectory_count, lap_length,
            prediction_dt, horizon, &configuration->model, &nominal,
            references)) {
        result->r1_status = MPC_RTI_CYCLE_REJECTED_INPUT;
        result->total_rti_us = monotonic_microseconds() - cycle_start_us;
        return reject_cycle(memory, result, MPC_RTI_CYCLE_REJECTED_INPUT);
    }
    result->nominal_first_control = nominal.controls[0];
    MpcRtiCorridorSchedule_t corridor_schedule;
    if (!mpc_rti_build_corridor_schedule(current_state, &nominal, trajectory,
            trajectory_count, lap_length, prediction_dt, &configuration->model,
            &corridor_schedule)) {
        result->r1_status = MPC_RTI_CYCLE_REJECTED_INPUT;
        result->total_rti_us = monotonic_microseconds() - cycle_start_us;
        return reject_cycle(memory, result, MPC_RTI_CYCLE_REJECTED_INPUT);
    }
    result->recovery_active = corridor_schedule.recovery_active;
    result->recovery_not_found = corridor_schedule.recovery_not_found;
    result->recovery_reentry_stage =
        corridor_schedule.first_normal_feasible_stage;
    result->recovery_initial_normal_violation_m =
        corridor_schedule.initial_normal_violation;
    result->recovery_max_seed_violation_m =
        corridor_schedule.max_seed_violation;
    for (int k = 0; k <= horizon; ++k) {
        result->recovery_normal_lower_m[k] =
            corridor_schedule.normal_lower[k];
        result->recovery_normal_upper_m[k] =
            corridor_schedule.normal_upper[k];
        result->recovery_active_lower_m[k] =
            corridor_schedule.active_lower[k];
        result->recovery_active_upper_m[k] =
            corridor_schedule.active_upper[k];
        result->recovery_seed_e_y_m[k] = corridor_schedule.seed_e_y[k];
        result->recovery_seed_violation_m[k] =
            corridor_schedule.seed_violation[k];
        result->recovery_stage[k] = corridor_schedule.recovery_stage[k];
    }
    MpcRtiPassResult_t r1;
    const MpcRtiCycleStatus_t r1_status = solve_rti_pass(
        current_state, current_progress, &nominal, references, trajectory,
        trajectory_count, lap_length, prediction_dt, horizon, configuration,
        &corridor_schedule, memory, &r1);
    result->rti_iterations_used = 1;
    result->r1_status = (int)r1_status;
    result->r1_nonlinear_failure_reason = r1.nonlinear_failure_reason;
    result->r1_nonlinear_failure_stage = r1.nonlinear_failure_stage;
    result->r1_first_action = r1.first_action;
    result->r1_solver_iterations = r1.iterations;
    result->r1_solve_us = r1.solve_us;
    result->r1_nonlinear_objective = r1.nonlinear_objective;
    result->r1_min_corridor_slack = r1.minimum_corridor_slack;
    result->rho_start = r1.rho_start;
    result->rho_u_start = r1.rho_u_start;
    result->rho_change_count = r1.rho_change_count;
    result->factorization_count = r1.factorization_count;
    result->factorization_time_ns = r1.factorization_time_ns;
    const int r1_accepted = r1_status == MPC_RTI_CYCLE_ACCEPTED_OPTIMAL ||
        r1_status == MPC_RTI_CYCLE_ACCEPTED_DEGRADED;
    const MpcRtiMemory_t r1_memory = *memory;
    const int repairable_corridor_failure = !r1_accepted &&
        configuration->refinement_mode != MPC_RTI_REFINEMENT_R1 &&
        r1_status == MPC_RTI_CYCLE_REJECTED_NONLINEAR_ROLLOUT &&
        r1.nonlinear_failure_reason == MPC_RTI_ROLLOUT_CORRIDOR &&
        r1.nonlinear_failure_stage >= 0;
    const int repairable_residual_failure = !r1_accepted &&
        configuration->refinement_mode != MPC_RTI_REFINEMENT_R1 &&
        r1_status == MPC_RTI_CYCLE_REJECTED_RESIDUAL &&
        r1.candidate.valid;
    if (!r1_accepted && !repairable_corridor_failure &&
        !repairable_residual_failure) {
        result->solver_iterations = r1.iterations;
        result->primal_residual = r1.primal_residual;
        result->dual_residual = r1.dual_residual;
        result->maximum_regularization = r1.maximum_regularization;
        result->regularization_count = r1.regularization_count;
        result->nonsmooth_jacobian_columns = r1.nonsmooth_columns;
        result->nonlinear_failure_stage = r1.nonlinear_failure_stage;
        result->nonlinear_failure_reason = r1.nonlinear_failure_reason;
        result->total_rti_us = monotonic_microseconds() - cycle_start_us;
        return reject_cycle(memory, result, r1_status);
    }

    MpcRtiPassResult_t selected = r1;
    int selected_candidate = r1_accepted ? 1 : 0;
    unsigned int trigger_mask = 0;
    if (configuration->refinement_mode == MPC_RTI_REFINEMENT_ADAPTIVE &&
        r1_accepted) {
        trigger_mask = adaptive_rti2_trigger_mask(
            configuration, &nominal, &r1);
    } else if (configuration->refinement_mode == MPC_RTI_REFINEMENT_R2) {
        trigger_mask = 1u << 31; /* Explicit R2 request, not a data trigger. */
    }
    if (repairable_corridor_failure)
        trigger_mask |= MPC_RTI2_TRIGGER_R1_CORRIDOR_REPAIR;
    if (repairable_residual_failure)
        trigger_mask |= MPC_RTI2_TRIGGER_R1_RESIDUAL_RECOVERY;
    const int run_r2 = configuration->refinement_mode ==
            MPC_RTI_REFINEMENT_R2 ||
        (configuration->refinement_mode == MPC_RTI_REFINEMENT_ADAPTIVE &&
         trigger_mask != 0u) || repairable_corridor_failure;
    result->rti2_trigger_reason_mask = trigger_mask;
    result->rti2_triggered = run_r2;

    MpcRtiPassResult_t r2;
    memset(&r2, 0, sizeof(r2));
    r2.status = MPC_RTI_CYCLE_REJECTED_INPUT;
    r2.nonlinear_objective = INFINITY;
    r2.minimum_corridor_slack = -INFINITY;
    r2.nonlinear_failure_stage = -1;
    if (run_r2) {
        MpcRtiNominal_t repair_seed;
        const MpcRtiNominal_t *r2_seed = &r1.candidate;
        if (repairable_corridor_failure) {
            if (build_rti_corridor_repair_seed(current_state,
                    current_progress, r1.candidate.controls, trajectory,
                    trajectory_count, lap_length, prediction_dt, horizon,
                    &configuration->model, &repair_seed)) {
                r2_seed = &repair_seed;
            } else {
                result->r2_status = MPC_RTI_CYCLE_REJECTED_INPUT;
                result->rti2_triggered = 0;
                result->solver_iterations = r1.iterations;
                result->primal_residual = r1.primal_residual;
                result->dual_residual = r1.dual_residual;
                result->maximum_regularization = r1.maximum_regularization;
                result->regularization_count = r1.regularization_count;
                result->nonsmooth_jacobian_columns = r1.nonsmooth_columns;
                result->nonlinear_failure_stage = r1.nonlinear_failure_stage;
                result->nonlinear_failure_reason =
                    r1.nonlinear_failure_reason;
                result->total_rti_us =
                    monotonic_microseconds() - cycle_start_us;
                return reject_cycle(memory, result, r1_status);
            }
        }
        MpcRtiReference_t r2_references[PREDICTION_HORIZON + 1];
        int references_valid = 1;
        for (int k = 0; k <= horizon; ++k) {
            if (!reference_at_progress(trajectory, trajectory_count,
                    lap_length, r2_seed->progress[k],
                    &configuration->model, &r2_references[k])) {
                references_valid = 0;
                break;
            }
        }
        if (references_valid) {
            /* R2 is the same sample: seed directly from X1/U1/progress. Never
             * call mpc_rti_build_nominal() here (that shifts the horizon). */
            MpcRtiNominal_t r2_nominal = *r2_seed;
            const MpcRtiCycleStatus_t r2_status = solve_rti_pass(
                current_state, current_progress, &r2_nominal, r2_references,
                trajectory, trajectory_count, lap_length, prediction_dt,
                horizon, configuration, &corridor_schedule, memory, &r2);
            result->r2_status = (int)r2_status;
            result->r2_nonlinear_failure_reason =
                r2.nonlinear_failure_reason;
            result->r2_nonlinear_failure_stage = r2.nonlinear_failure_stage;
            result->rti_iterations_used = 2;
            result->r2_first_action = r2.first_action;
            result->r2_solver_iterations = r2.iterations;
            result->r2_solve_us = r2.solve_us;
            result->r2_nonlinear_objective = r2.nonlinear_objective;
            result->r2_min_corridor_slack = r2.minimum_corridor_slack;
            result->rho_change_count += r2.rho_change_count;
            result->factorization_count += r2.factorization_count;
            result->factorization_time_ns += r2.factorization_time_ns;
            if (r2_status == MPC_RTI_CYCLE_ACCEPTED_OPTIMAL ||
                r2_status == MPC_RTI_CYCLE_ACCEPTED_DEGRADED) {
                int select_r2 = !r1_accepted;
                if (r1_accepted) {
                    const float objective_scale = fmaxf(1.0f,
                        fmaxf(fabsf(r1.nonlinear_objective),
                              fabsf(r2.nonlinear_objective)));
                    const float objective_tie = 1.0e-4f * objective_scale;
                    select_r2 = r2.nonlinear_objective <
                            r1.nonlinear_objective - objective_tie ||
                        (fabsf(r2.nonlinear_objective -
                               r1.nonlinear_objective) <= objective_tie &&
                         r2.minimum_corridor_slack >
                             r1.minimum_corridor_slack);
                }
                if (select_r2) {
                    selected = r2;
                    selected_candidate = 2;
                }
            } else {
                if (r1_accepted) {
                    /* A failed R2 cannot destroy the already feasible R1
                     * state or its ADMM warm-start. */
                    *memory = r1_memory;
                }
            }
        } else {
            result->r2_status = MPC_RTI_CYCLE_REJECTED_INPUT;
            if (r1_accepted) *memory = r1_memory;
        }
        if (!r1_accepted &&
            result->r2_status != MPC_RTI_CYCLE_ACCEPTED_OPTIMAL &&
            result->r2_status != MPC_RTI_CYCLE_ACCEPTED_DEGRADED) {
            result->solver_iterations = r2.iterations;
            result->primal_residual = r2.primal_residual;
            result->dual_residual = r2.dual_residual;
            result->maximum_regularization = r2.maximum_regularization;
            result->regularization_count = r2.regularization_count;
            result->nonsmooth_jacobian_columns = r2.nonsmooth_columns;
            result->nonlinear_failure_stage = r2.nonlinear_failure_stage;
            result->nonlinear_failure_reason = r2.nonlinear_failure_reason;
            result->total_rti_us = monotonic_microseconds() - cycle_start_us;
            return reject_cycle(memory, result,
                (MpcRtiCycleStatus_t)result->r2_status);
        }
    }

    if (run_r2 && selected_candidate == 1) *memory = r1_memory;
    result->selected_candidate = selected_candidate;
    result->r2_status = run_r2 ? result->r2_status : -1;
    result->candidate_path_delta = selected.path_delta;
    result->max_candidate_progress_error_m = max_path_delta_double(
        selected.path_delta.progress_error_m,
        selected.path_delta.sample_count);
    result->max_candidate_curvature_error_per_m = max_path_delta_float(
        selected.path_delta.curvature_error_per_m,
        selected.path_delta.sample_count);
    result->max_candidate_left_bound_error_m = max_path_delta_float(
        selected.path_delta.left_bound_error_m,
        selected.path_delta.sample_count);
    result->max_candidate_right_bound_error_m = max_path_delta_float(
        selected.path_delta.right_bound_error_m,
        selected.path_delta.sample_count);
    result->minimum_predicted_corridor_slack_m =
        selected.minimum_corridor_slack;
    result->lateral_accel_proxy_mps2 = selected.lateral_accel_proxy;
    result->lateral_accel_proxy_stage = selected.lateral_accel_proxy_stage;
    memcpy(result->lateral_accel_proxy_by_stage_mps2,
        selected.lateral_accel_proxy_by_stage,
        (size_t)(horizon + 1) * sizeof(float));
    result->solver_iterations = selected.iterations;
    result->primal_residual = selected.primal_residual;
    result->dual_residual = selected.dual_residual;
    result->maximum_regularization = selected.maximum_regularization;
    result->regularization_count = selected.regularization_count;
    result->nonsmooth_jacobian_columns = selected.nonsmooth_columns;
    result->nonlinear_failure_stage = selected.nonlinear_failure_stage;
    result->nonlinear_failure_reason = selected.nonlinear_failure_reason;
    result->first_control = selected.first_action;
    result->published_steering_command =
        selected.candidate.states[1].plant.steering_command;
    result->published_target_speed =
        selected.candidate.states[1].plant.target_speed;
    result->status = selected.status;

    memory->nominal = selected.candidate;
    memory->consecutive_degraded_solves =
        selected.status == MPC_RTI_CYCLE_ACCEPTED_DEGRADED
            ? r1_memory.consecutive_degraded_solves + 1 : 0;
    result->rho_final = memory->solver_state.rho;
    result->rho_u_final = memory->solver_state.rho_u;
    result->total_rti_us = monotonic_microseconds() - cycle_start_us;
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
    const float etarget = state->plant.target_speed - reference->target_speed;
    const float edelta = state->plant.steering_command -
        reference->steering_command;
    const float dq_delta = control->steering_rate -
        state->previous_steering_rate;
    const float dq_speed = control->target_speed_rate -
        state->previous_target_speed_rate;
    return configuration->weight_e_y * ey * ey +
        configuration->weight_e_psi * epsi * epsi +
        configuration->weight_u * eu * eu +
        configuration->weight_target_speed_state * etarget * etarget +
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
