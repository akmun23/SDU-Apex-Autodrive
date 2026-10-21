#define _POSIX_C_SOURCE 200809L

#include "mpc_rti.h"
#include "vehicle_model.h"

#include <math.h>
#include <string.h>
#include <time.h>

static double monotonic_microseconds(void);
static float corridor_margin_for_state(
    const MpcRtiConfiguration_t *configuration,
    const MpcModelState_t *state,
    int prediction_index);

static int finite_nonnegative(float value)
{
    return isfinite(value) && value >= 0.0f;
}

static int finite_rti_state(const MpcRtiState_t *state)
{
    if (!(state && isfinite(state->plant.e_y) &&
        isfinite(state->plant.e_psi) && isfinite(state->plant.u) &&
        isfinite(state->plant.v) && isfinite(state->plant.r) &&
        isfinite(state->plant.actual_steering_angle) &&
        isfinite(state->plant.delayed_throttle_command_1) &&
        isfinite(state->plant.delayed_throttle_command_2))) return 0;
    if (state->plant.delayed_throttle_command_1 < 0.0f ||
        state->plant.delayed_throttle_command_1 > 1.0f ||
        state->plant.delayed_throttle_command_2 < 0.0f ||
        state->plant.delayed_throttle_command_2 > 1.0f) return 0;
    if (state->plant.source_plant_initialized ||
        state->plant.source_imu_state_initialized) {
        for (int index = 0; index < MPC_PLANT_INTERNAL_STATE_COUNT; ++index)
            if (!isfinite(
                    state->plant.source_plant_internal_state[index])) return 0;
    }
    return 1;
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
        configuration->weight_u_overspeed,
        configuration->weight_longitudinal_slip,
        configuration->weight_v,
        configuration->weight_r,
        configuration->weight_steering_command,
        configuration->weight_steering_target,
        configuration->weight_throttle_command,
        configuration->weight_steering_slew,
        configuration->weight_throttle_slew,
        configuration->startup_nominal_throttle_seed,
        configuration->terminal_multiplier,
        configuration->corridor_margin_m,
        configuration->first_prediction_corridor_margin_m,
        configuration->corridor_preview_halfwidth_m,
        configuration->nonlinear_corridor_tolerance_m};
    for (unsigned int i = 0; i < sizeof(weights) / sizeof(weights[0]); ++i)
        if (!finite_nonnegative(weights[i])) return 0;

    return configuration->jacobian_threads >= 0 &&
        configuration->jacobian_threads <= 4 &&
        isfinite(configuration->max_speed_mps) &&
        configuration->max_speed_mps > 0.0f &&
        configuration->max_speed_mps <=
            vehicle.maximum_command_speed_mps &&
        isfinite(configuration->active_speed_ceiling_mps) &&
        configuration->active_speed_ceiling_mps > 0.0f &&
        configuration->active_speed_ceiling_mps <=
            configuration->max_speed_mps &&
        configuration->startup_nominal_throttle_seed <= MPC_THROTTLE_MAX &&
        isfinite(configuration->max_steering_rad) &&
        configuration->max_steering_rad > 0.0f &&
        configuration->max_steering_rad <= vehicle.max_steering_angle &&
        configuration->first_prediction_corridor_margin_m <=
            configuration->corridor_margin_m;
}

static int optimizer_index_from_model_state(int model_index)
{
    if (model_index >= MPC_STATE_IDX_EY &&
        model_index <= MPC_STATE_IDX_R) return model_index;
    if (model_index == MPC_STATE_IDX_ACTUAL_STEERING_ANGLE)
        return MPC_OPT_IDX_ACTUAL_STEERING_ANGLE;
    if (model_index >= MPC_STATE_IDX_WHEEL_OMEGA_FL &&
        model_index <= MPC_STATE_IDX_SUSPENSION_JOUNCE_RR) {
        return MPC_OPT_IDX_WHEEL_OMEGA_FL +
            model_index - MPC_STATE_IDX_WHEEL_OMEGA_FL;
    }
    return -1;
}

static void serialize_state(const MpcRtiState_t *state, float x[MPC_RTI_NX])
{
    MpcModelState_t prepared_plant = state->plant;
    (void)vehicle_model_initialize_plant_state(&prepared_plant);
    float full[MPC_MODEL_NX] = {0};
    full[MPC_STATE_IDX_EY] = state->plant.e_y;
    full[MPC_STATE_IDX_EPSI] = state->plant.e_psi;
    full[MPC_STATE_IDX_U] = state->plant.u;
    full[MPC_STATE_IDX_V] = state->plant.v;
    full[MPC_STATE_IDX_R] = state->plant.r;
    full[MPC_STATE_IDX_ACTUAL_STEERING_ANGLE] =
        state->plant.actual_steering_angle;
    for (int wheel = 0; wheel < 4; ++wheel) {
        full[MPC_STATE_IDX_WHEEL_OMEGA_FL + wheel] = (float)
            prepared_plant.source_plant_internal_state[
                MPC_PLANT_STATE_WHEEL_OMEGA_FL + wheel];
        full[MPC_STATE_IDX_SUSPENSION_JOUNCE_FL + wheel] = (float)
            prepared_plant.source_plant_internal_state[
                MPC_PLANT_STATE_SUSPENSION_JOUNCE_FL + wheel];
    }
    full[MPC_STATE_IDX_BODY_HEIGHT] = (float)
        prepared_plant.source_plant_internal_state[MPC_PLANT_STATE_BODY_HEIGHT];
    full[MPC_STATE_IDX_BODY_HEIGHT_RATE] = (float)
        prepared_plant.source_plant_internal_state[
            MPC_PLANT_STATE_BODY_HEIGHT_RATE];
    full[MPC_STATE_IDX_ROLL] = (float)
        prepared_plant.source_plant_internal_state[MPC_PLANT_STATE_ROLL];
    full[MPC_STATE_IDX_ROLL_RATE] = (float)
        prepared_plant.source_plant_internal_state[MPC_PLANT_STATE_ROLL_RATE];
    full[MPC_STATE_IDX_PITCH] = (float)
        prepared_plant.source_plant_internal_state[MPC_PLANT_STATE_PITCH];
    full[MPC_STATE_IDX_PITCH_RATE] = (float)
        prepared_plant.source_plant_internal_state[MPC_PLANT_STATE_PITCH_RATE];
    for (int i = 0; i < MPC_MODEL_NX; ++i) {
        const int optimizer_index = optimizer_index_from_model_state(i);
        if (optimizer_index >= 0) x[optimizer_index] = full[i];
    }
}

static void store_slip_metrics(MpcRtiState_t *state,
                               const MpcStageResult_t *step)
{
    state->incoming_max_positive_wheel_slip = 0.0f;
    for (int wheel = 0; wheel < 4; ++wheel) {
        state->incoming_max_positive_wheel_slip = fmaxf(
            state->incoming_max_positive_wheel_slip,
            (float)step->output.maximum_positive_longitudinal_slip[wheel]);
        state->incoming_wheel_slip_excess_mps[wheel] =
            step->output.maximum_longitudinal_slip_excess_mps[wheel];
    }
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
        vehicle_model_steering_for_curvature((float)sample.curvature),
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
        states[k + 1].incoming_branch_flags = step.branch_flags;
        states[k + 1].incoming_body_forward_delta_m =
            step.body_forward_delta_m;
        states[k + 1].incoming_body_lateral_delta_m =
            step.body_lateral_delta_m;
        states[k + 1].incoming_body_yaw_delta_rad =
            step.body_yaw_delta_rad;
        store_slip_metrics(&states[k + 1], &step);
        states[k + 1].incoming_transition_valid = 1;
        progress[k + 1] = progress[k] + step.delta_s_m;
    }
    return 1;
}

static void align_yaw_rate_references(
    MpcRtiReference_t references[PREDICTION_HORIZON + 1],
    int horizon)
{
    for (int k = 0; k < horizon; ++k) {
        references[k].r = references[k].path_curvature * references[k].u;
    }
    references[horizon].r = references[horizon].path_curvature *
        references[horizon].u;
}

static int corridor_contains(
    const MpcRtiState_t *state,
    const MpcRtiReference_t *reference,
    const MpcRtiConfiguration_t *configuration,
    float corridor_margin_m)
{
    const float lower = corridor_margin_m - reference->right_bound -
        configuration->nonlinear_corridor_tolerance_m;
    const float upper = reference->left_bound - corridor_margin_m +
        configuration->nonlinear_corridor_tolerance_m;
    return isfinite(lower) && isfinite(upper) && lower <= upper &&
        state->plant.e_y >= lower && state->plant.e_y <= upper;
}

static void record_rollout_failure(
    MpcRtiRolloutFailure_t *failure,
    int stage,
    const MpcRtiState_t *state,
    const MpcRtiReference_t *reference,
    double progress,
    float margin_m,
    float lower_bound_m,
    float upper_bound_m)
{
    if (!failure || !state || !reference) return;
    failure->valid = 1;
    failure->stage = stage;
    failure->state = *state;
    failure->reference = *reference;
    failure->progress = progress;
    failure->margin_m = margin_m;
    failure->lower_bound_m = lower_bound_m;
    failure->upper_bound_m = upper_bound_m;
}

static float corridor_margin_at_prediction(
    const MpcRtiConfiguration_t *configuration,
    int prediction_index)
{
    return prediction_index == 1
        ? configuration->first_prediction_corridor_margin_m
        : configuration->corridor_margin_m;
}

static float corridor_margin_for_state(
    const MpcRtiConfiguration_t *configuration,
    const MpcModelState_t *state,
    int prediction_index)
{
    if (!configuration || !state || !isfinite(state->e_psi))
        return NAN;

    /* Require the configured wall clearance plus the vehicle rectangle's
     * lateral projection about the rear axle. This keeps the full physical
     * clearance at every heading without padding the aligned car to the
     * wider planning footprint. This is a geometric feasibility rule, not a
     * change to Unity physics. */
    const float abs_cos = fabsf(cosf(state->e_psi));
    const float abs_sin = fabsf(sinf(state->e_psi));
    const float physical_extent =
        0.5f * MPC_CAR_WIDTH_M * abs_cos +
        fmaxf(MPC_REAR_AXLE_TO_FRONT_BUMPER_M,
              MPC_REAR_OVERHANG_M) * abs_sin;
    return corridor_margin_at_prediction(configuration, prediction_index) +
        physical_extent;
}

static MpcRtiCommandEnvelopeFailure_t command_envelope_failure(
    const MpcRtiState_t *state,
    const MpcRtiConfiguration_t *configuration)
{
    if (!state || !configuration)
        return MPC_RTI_COMMAND_ENVELOPE_NONFINITE_STATE;
    if (!isfinite(state->plant.delayed_throttle_command_1) ||
        !isfinite(state->plant.delayed_throttle_command_2))
        return MPC_RTI_COMMAND_ENVELOPE_NONFINITE_STATE;
    if (state->plant.delayed_throttle_command_1 < 0.0f ||
        state->plant.delayed_throttle_command_1 > 1.0f ||
        state->plant.delayed_throttle_command_2 < 0.0f ||
        state->plant.delayed_throttle_command_2 > 1.0f)
        return MPC_RTI_COMMAND_ENVELOPE_THROTTLE_QUEUE;
    if (!finite_rti_state(state))
        return MPC_RTI_COMMAND_ENVELOPE_NONFINITE_STATE;
    /* Native PxVehicleNoDrive hard braking can briefly reverse body-u by up
     * to 7.79 mm/s across the tested 0.5--12 m/s stop envelope. This is a
     * measured source-solver transient, not a reverse-drive command. */
    if (state->plant.u < -MPC_PHYSX_BRAKE_REVERSE_TRANSIENT_MPS)
        return MPC_RTI_COMMAND_ENVELOPE_SPEED_BELOW_ZERO;
    const VehicleParameters_t vehicle = vehicle_model_get_parameters();
    if (state->plant.u > vehicle.maximum_command_speed_mps)
        return MPC_RTI_COMMAND_ENVELOPE_SPEED_ABOVE_MAX;
    if (fabsf(state->plant.delayed_steering_command_1) >
            configuration->max_steering_rad ||
        fabsf(state->plant.delayed_steering_command_2) >
            configuration->max_steering_rad)
        return MPC_RTI_COMMAND_ENVELOPE_STEERING_QUEUE;
    if (fabsf(state->plant.actual_steering_angle) >
        configuration->max_steering_rad)
        return MPC_RTI_COMMAND_ENVELOPE_ACTUAL_STEERING;
    return MPC_RTI_COMMAND_ENVELOPE_OK;
}

static int state_inside_command_envelope(
    const MpcRtiState_t *state,
    const MpcRtiConfiguration_t *configuration)
{
    /* Queued actuator commands are already in flight, not new decisions. */
    return command_envelope_failure(state, configuration) ==
        MPC_RTI_COMMAND_ENVELOPE_OK;
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
        if (!isfinite(nominal->controls[k].steering_target_rad) ||
            !isfinite(nominal->controls[k].throttle_command)) return 0;
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

static int set_state_bounds_with_corridor(
    float lower[MPC_RTI_NX],
    float upper[MPC_RTI_NX],
    float ey_lower,
    float ey_upper,
    const MpcRtiConfiguration_t *configuration)
{
    if (!isfinite(ey_lower) || !isfinite(ey_upper) || ey_lower > ey_upper)
        return 0;

    for (int i = 0; i < MPC_RTI_NX; ++i) {
        lower[i] = -BIG_BOUND;
        upper[i] = BIG_BOUND;
    }
    lower[MPC_STATE_IDX_EY] = ey_lower;
    upper[MPC_STATE_IDX_EY] = ey_upper;
    lower[MPC_STATE_IDX_U] = -MPC_PHYSX_BRAKE_REVERSE_TRANSIENT_MPS;
    upper[MPC_STATE_IDX_U] =
        vehicle_model_get_parameters().maximum_command_speed_mps;
    lower[5] =
        -configuration->max_steering_rad;
    upper[5] =
        configuration->max_steering_rad;
    for (int wheel = 0; wheel < 4; ++wheel) {
        lower[6 + wheel] = -350.0f;
        upper[6 + wheel] = 350.0f;
    }
    return 1;
}

static void add_tracking_term(
    float *q_diag,
    float *q_linear,
    int index,
    float weight,
    float target,
    float multiplier)
{
    /* Zero-weight channels are common in the active profile (for example
     * lateral velocity). Avoid constructing/multiplying a term that cannot
     * affect the QP. Keeping this helper scalar also lets the compiler fold
     * the fixed factor without creating per-stage temporary arrays. */
    if (weight <= 0.0f) return;
    const float scaled_weight = multiplier * weight;
    const float doubled_weight = 2.0f * scaled_weight;
    q_diag[index] += doubled_weight;
    q_linear[index] -= doubled_weight * target;
}

static void add_tracking_cost(
    float Q[MPC_RTI_NX],
    float q[MPC_RTI_NX],
    const MpcRtiReference_t *reference,
    const MpcRtiConfiguration_t *configuration,
    float multiplier,
    float nominal_u)
{
    const float speed_weight = nominal_u > reference->u
        ? configuration->weight_u_overspeed : configuration->weight_u;
    add_tracking_term(Q, q, MPC_STATE_IDX_EY,
        configuration->weight_e_y, reference->e_y, multiplier);
    add_tracking_term(Q, q, MPC_STATE_IDX_EPSI,
        configuration->weight_e_psi, reference->e_psi, multiplier);
    add_tracking_term(Q, q, MPC_STATE_IDX_U,
        speed_weight, reference->u, multiplier);
    add_tracking_term(Q, q, MPC_STATE_IDX_V,
        configuration->weight_v, reference->v, multiplier);
    add_tracking_term(Q, q, MPC_STATE_IDX_R,
        configuration->weight_r, reference->r, multiplier);
    add_tracking_term(Q, q, 5,
        configuration->weight_steering_command,
        reference->steering_command, multiplier);
}

/* Gauss-Newton local model of the source tire curve's force-loss region.
 * The residual is wheel-surface-speed mismatch beyond the source curve's
 * extremum. Its Jacobian is taken with respect to each wheel omega while the
 * other plant coordinates are held at the current RTI linearization point. */
static void add_source_wheel_slip_cost(
    float Q[MPC_RTI_NX], float q[MPC_RTI_NX],
    const MpcModelState_t *plant, float weight, float multiplier)
{
    if (weight <= 0.0f) return;
    double excess[4] = {0}, derivative[4] = {0};
    if (!vehicle_model_longitudinal_slip_excess(
            plant, excess, derivative)) return;
    MpcModelState_t prepared = *plant;
    if (!vehicle_model_initialize_plant_state(&prepared)) return;
    for (int wheel = 0; wheel < 4; ++wheel) {
        if (excess[wheel] <= 0.0 || fabs(derivative[wheel]) < 1.0e-9)
            continue;
        const int index = 6 + wheel;
        const float omega = (float)prepared.source_plant_internal_state[
            MPC_PLANT_STATE_WHEEL_OMEGA_FL + wheel];
        const float local_weight = weight *
            (float)(derivative[wheel] * derivative[wheel]);
        const float local_target = omega -
            (float)(excess[wheel] / derivative[wheel]);
        add_tracking_term(Q, q, index, local_weight, local_target, multiplier);
    }
}

static double source_slip_cost(
    const MpcRtiConfiguration_t *configuration,
    const MpcModelState_t *plant)
{
    if (!configuration || !plant ||
        configuration->weight_longitudinal_slip <= 0.0f) return 0.0;
    double excess[4] = {0}, derivative[4] = {0};
    if (!vehicle_model_longitudinal_slip_excess(
            plant, excess, derivative)) return INFINITY;
    double cost = 0.0;
    for (int wheel = 0; wheel < 4; ++wheel)
        cost += configuration->weight_longitudinal_slip *
            excess[wheel] * excess[wheel];
    return cost;
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
        !valid_configuration(configuration) ||
        horizon != PREDICTION_HORIZON || !isfinite(prediction_dt) ||
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
        if (!isfinite(nominal_controls[k].steering_target_rad) ||
            !isfinite(nominal_controls[k].throttle_command)) return 0;
    }

    memset(problem, 0, sizeof(*problem));
    problem->horizon = horizon;
    problem->linearization_failure_stage = -1;
    problem->linearization_failure_index = -1;
    /* The newest queued command is already in flight and cannot be optimized.
     * Start at its exact nonlinear endpoint; linearize only transitions that
     * new throttle/steering decisions can affect inside this horizon. */
    serialize_state(&nominal_states[MPC_COMMAND_DELAY_STEPS], problem->x0);
    /* Match evaluate_nonlinear_candidate(): the first optimized input's
     * slew cost is measured from the command already present at the incoming
     * state. Seeding these auxiliary coordinates from the post-delay nominal
     * state instead compares the decision against a newly generated nominal
     * command, weakening/misaligning the first-action continuity penalty. */
    problem->x0[MPC_OPT_IDX_PREVIOUS_STEERING_COMMAND] =
        nominal_states[0].plant.delayed_steering_command_2;
    problem->x0[MPC_OPT_IDX_PREVIOUS_THROTTLE_COMMAND] =
        nominal_states[0].plant.delayed_throttle_command_2;

    MpcStageLinearization_t plant_linearizations[RICCATI_HORIZON];
    int linearization_ok[RICCATI_HORIZON] = {0};
    const int jacobian_threads = configuration->jacobian_threads > 0
        ? configuration->jacobian_threads : 1;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(jacobian_threads) \
    if(horizon >= 8 && jacobian_threads > 1)
#endif
    for (int k = 0; k < RICCATI_HORIZON; ++k) {
        const int model_stage = k + MPC_COMMAND_DELAY_STEPS;
        MpcStageLinearization_t *plant_linearization =
            &plant_linearizations[k];
        MpcStageResult_t known_step = {0};
        known_step.next = nominal_states[model_stage + 1].plant;
        known_step.branch_flags =
            nominal_states[model_stage + 1].incoming_branch_flags;
        known_step.body_forward_delta_m =
            nominal_states[model_stage + 1].incoming_body_forward_delta_m;
        known_step.body_lateral_delta_m =
            nominal_states[model_stage + 1].incoming_body_lateral_delta_m;
        known_step.body_yaw_delta_rad =
            nominal_states[model_stage + 1].incoming_body_yaw_delta_rad;
        known_step.valid =
            nominal_states[model_stage + 1].incoming_transition_valid;
#ifdef MPC_ENABLE_FD_ORACLE
        const int linearized = configuration->use_fd_jacobian_oracle
            ? mpc_model_linearize_fd_oracle(&nominal_states[model_stage].plant,
                &nominal_controls[model_stage], prediction_dt,
                references[model_stage].path_curvature, plant_linearization)
            : (known_step.valid
                ? mpc_model_linearize_from_known_step(
                    &nominal_states[model_stage].plant,
                    &nominal_controls[model_stage], prediction_dt,
                    references[model_stage].path_curvature,
                    &known_step, plant_linearization)
                : mpc_model_linearize(&nominal_states[model_stage].plant,
                    &nominal_controls[model_stage], prediction_dt,
                    references[model_stage].path_curvature,
                    plant_linearization));
#else
        const int linearized = known_step.valid
            ? mpc_model_linearize_from_known_step(
                &nominal_states[model_stage].plant,
                &nominal_controls[model_stage], prediction_dt,
                references[model_stage].path_curvature,
                &known_step, plant_linearization)
            : mpc_model_linearize(&nominal_states[model_stage].plant,
                &nominal_controls[model_stage], prediction_dt,
                references[model_stage].path_curvature, plant_linearization);
#endif
        linearization_ok[k] = linearized;
    }

    /* The four FIFO coordinates remain in the nonlinear plant, but neither
     * queued commands nor post-horizon commands are QP decisions. The first
     * decision now directly controls the first transition after the known
     * delay prefix. */
    float nominal_optimizer_states[PREDICTION_HORIZON + 1][MPC_RTI_NX] = {{0}};
    for (int k = 0; k <= horizon; ++k) {
        serialize_state(&nominal_states[k], nominal_optimizer_states[k]);
        if (k <= MPC_COMMAND_DELAY_STEPS) {
            nominal_optimizer_states[k][
                MPC_OPT_IDX_PREVIOUS_STEERING_COMMAND] =
                nominal_states[k].plant.delayed_steering_command_2;
            nominal_optimizer_states[k][
                MPC_OPT_IDX_PREVIOUS_THROTTLE_COMMAND] =
                nominal_states[k].plant.delayed_throttle_command_2;
        } else {
            nominal_optimizer_states[k][
                MPC_OPT_IDX_PREVIOUS_STEERING_COMMAND] =
                nominal_controls[k - MPC_COMMAND_DELAY_STEPS - 1]
                    .steering_target_rad;
            nominal_optimizer_states[k][
                MPC_OPT_IDX_PREVIOUS_THROTTLE_COMMAND] =
                nominal_controls[k - MPC_COMMAND_DELAY_STEPS - 1]
                    .throttle_command;
        }
    }

    for (int solver_stage = 0; solver_stage < RICCATI_HORIZON;
         ++solver_stage) {
        const int model_stage = solver_stage + MPC_COMMAND_DELAY_STEPS;
        RiccatiStepData_t *stage = &problem->steps[solver_stage];
        set_unconstrained_bounds(stage);
        const MpcStageLinearization_t *lin =
            &plant_linearizations[solver_stage];
        if (!linearization_ok[solver_stage]) {
            problem->linearization_failure_stage = model_stage;
            problem->linearization_failure_reason = lin->failure_reason;
            problem->linearization_failure_index = lin->failure_index;
            return 0;
        }
        problem->nonsmooth_jacobian_columns +=
            lin->nonsmooth_column_count;
        const float *next_optimizer_state =
            nominal_optimizer_states[model_stage + 1];

        for (int row = 0; row < MPC_OPT_PHYSICAL_NX; ++row) {
            const int model_row = row < 5 ? row : row == 5 ? 7 : row + 4;
            stage->B[row][0] =
                lin->A[model_row][MPC_STATE_IDX_DELAYED_STEERING_COMMAND_2];
            stage->B[row][1] =
                lin->A[model_row][MPC_STATE_IDX_DELAYED_THROTTLE_COMMAND_2];
            for (int col = 0; col < MPC_OPT_PHYSICAL_NX; ++col) {
                const int model_col = col < 5 ? col : col == 5 ? 7 : col + 4;
                stage->A[row][col] = lin->A[model_row][model_col];
            }
            const float applied_steering =
                nominal_states[model_stage].plant.delayed_steering_command_2;
            const float applied_throttle =
                nominal_states[model_stage].plant.delayed_throttle_command_2;
            /* Preserve the nominal-point identity while condensing the known
             * prefix; accumulate cancellation-sensitive rows in double. */
            double affine_offset = next_optimizer_state[row];
            for (int col = 0; col < MPC_OPT_PHYSICAL_NX; ++col)
                affine_offset -= (double)stage->A[row][col] *
                    nominal_optimizer_states[model_stage][col];
            affine_offset -= (double)stage->B[row][0] * applied_steering;
            affine_offset -= (double)stage->B[row][1] * applied_throttle;
            stage->d[row] = (float)affine_offset;
        }

        /* Exact command-slew augmentation. These two auxiliary coordinates
         * are not vehicle states: they carry the previous optimized input so
         * adjacent steering/throttle increments are quadratic costs while
         * the physical plant states and controls stay unchanged. */
        stage->B[MPC_OPT_IDX_PREVIOUS_STEERING_COMMAND][0] = 1.0f;
        stage->B[MPC_OPT_IDX_PREVIOUS_THROTTLE_COMMAND][1] = 1.0f;

        /* The state at solver stage 0 is fixed by the queued commands, so its
         * state cost is constant and omitted. All remaining in-horizon costs
         * are unchanged; the endpoint cost is transferred to terminal_Q/q. */
        if (model_stage > MPC_COMMAND_DELAY_STEPS) {
            add_tracking_cost(stage->Q_diag, stage->q,
                &references[model_stage], configuration, 1.0f,
                nominal_states[model_stage].plant.u);
            add_source_wheel_slip_cost(stage->Q_diag, stage->q,
                &nominal_states[model_stage].plant,
                configuration->weight_longitudinal_slip, 1.0f);
        }

        /* The future command becomes delayed-steering state one sample after
         * issue. Keep its existing reference alignment, now directly on q[k]. */
        {
            const int reference_index = solver_stage + 1;
            const float doubled_weight = 2.0f *
                configuration->weight_steering_command;
            stage->R_diag[0] += doubled_weight;
            stage->r[0] -= doubled_weight *
                references[reference_index].steering_command;
        }

        stage->R_diag[0] +=
            2.0f * configuration->weight_steering_target;
        stage->R_diag[1] +=
            2.0f * configuration->weight_throttle_command;
        const float steering_slew_hessian =
            2.0f * configuration->weight_steering_slew;
        const float throttle_slew_hessian =
            2.0f * configuration->weight_throttle_slew;
        stage->Q_diag[MPC_OPT_IDX_PREVIOUS_STEERING_COMMAND] +=
            steering_slew_hessian;
        stage->R_diag[0] += steering_slew_hessian;
        stage->N[MPC_OPT_IDX_PREVIOUS_STEERING_COMMAND][0] -=
            steering_slew_hessian;
        stage->Q_diag[MPC_OPT_IDX_PREVIOUS_THROTTLE_COMMAND] +=
            throttle_slew_hessian;
        stage->R_diag[1] += throttle_slew_hessian;
        stage->N[MPC_OPT_IDX_PREVIOUS_THROTTLE_COMMAND][1] -=
            throttle_slew_hessian;

        const int bound_index = model_stage;
        const float margin = corridor_margin_for_state(configuration,
            &nominal_states[bound_index].plant, bound_index);
        const float ey_lower = margin - references[bound_index].right_bound;
        const float ey_upper = references[bound_index].left_bound - margin;
        if (!set_state_bounds_with_corridor(stage->x_lb, stage->x_ub,
                ey_lower, ey_upper, configuration)) return 0;

        stage->u_lb[0] = -configuration->max_steering_rad;
        stage->u_ub[0] = configuration->max_steering_rad;
        stage->u_lb[1] = MPC_THROTTLE_MIN_DRIVE;
        stage->u_ub[1] = MPC_THROTTLE_MAX;
    }

    /* Terminal state costs were previously carried by a held-state tail stage.
     * Put them in the solver's actual terminal cost and omit that dead stage. */
    add_tracking_cost(problem->terminal_Q, problem->terminal_q,
        &references[horizon], configuration,
        configuration->terminal_multiplier, nominal_states[horizon].plant.u);
    add_source_wheel_slip_cost(problem->terminal_Q, problem->terminal_q,
        &nominal_states[horizon].plant,
        configuration->weight_longitudinal_slip,
        configuration->terminal_multiplier);

    const float terminal_margin = corridor_margin_for_state(
        configuration, &nominal_states[horizon].plant, horizon);
    const float terminal_lower = terminal_margin - references[horizon].right_bound;
    const float terminal_upper = references[horizon].left_bound - terminal_margin;
    if (!set_state_bounds_with_corridor(problem->terminal_x_lb,
            problem->terminal_x_ub,
            terminal_lower, terminal_upper,
            configuration)) return 0;
    return 1;
}

static int fail_nominal_build(
    MpcRtiNominal_t *nominal,
    MpcRtiNominalBuildFailureStage_t stage)
{
    if (nominal) {
        nominal->build_failure_stage = stage;
        nominal->valid = 0;
    }
    return 0;
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
    if (!nominal) return 0;
    memset(nominal, 0, sizeof(*nominal));
    nominal->build_failure_index = -1;
    if (!finite_rti_state(current_state) || !isfinite(current_progress) ||
        !trajectory || !references || horizon < 1 ||
        horizon > PREDICTION_HORIZON || !isfinite(prediction_dt) ||
        prediction_dt <= 0.0f || !valid_configuration(configuration))
        return fail_nominal_build(
            nominal, MPC_RTI_NOMINAL_BUILD_INVALID_INPUT);
    if (!vehicle_model_set_active_target_speed_ceiling(
            configuration->active_speed_ceiling_mps))
        return fail_nominal_build(
            nominal, MPC_RTI_NOMINAL_BUILD_SPEED_CEILING);

    MpcRtiNominal_t previous_snapshot;
    const int warm = valid_previous_nominal(previous_nominal, horizon);
    if (warm) {
        previous_snapshot = *previous_nominal;
        previous_nominal = &previous_snapshot;
    }
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
            nominal->controls[k].steering_target_rad = clampf_rti(
                nominal->controls[k].steering_target_rad,
                -configuration->max_steering_rad,
                configuration->max_steering_rad);
            nominal->controls[k].throttle_command = clampf_rti(
                nominal->controls[k].throttle_command,
                MPC_THROTTLE_MIN_DRIVE, MPC_THROTTLE_MAX);
        }
    } else {
        TrajectoryReferencePoint_t seed_references[PREDICTION_HORIZON + 1];
        if (!mpc_reference_build_speed_seed(
                trajectory, trajectory_count, lap_length, current_progress,
                configuration->active_speed_ceiling_mps, prediction_dt,
                horizon, seed_references, progress_seed))
            return fail_nominal_build(
                nominal, MPC_RTI_NOMINAL_BUILD_SPEED_SEED);
        for (int k = 0; k <= horizon; ++k)
            if (!reference_at_progress(trajectory, trajectory_count,
                    lap_length, progress_seed[k], configuration,
                    &references[k]))
                return fail_nominal_build(
                    nominal, MPC_RTI_NOMINAL_BUILD_SEED_REFERENCE);

        MpcModelState_t plant = current_state->plant;
        for (int k = 0; k < horizon; ++k) {
            const float desired_delta = references[k + 1].steering_command;
            nominal->controls[k].steering_target_rad = clampf_rti(
                desired_delta,
                -configuration->max_steering_rad,
                configuration->max_steering_rad);
            nominal->controls[k].throttle_command = clampf_rti(
                fmaxf(plant.delayed_throttle_command_2,
                    MPC_THROTTLE_MIN_DRIVE),
                MPC_THROTTLE_MIN_DRIVE, MPC_THROTTLE_MAX);
            const MpcStageResult_t step = mpc_vehicle_model_step(
                &plant, &nominal->controls[k], prediction_dt,
                references[k].path_curvature);
            if (!step.valid)
                return fail_nominal_build(
                    nominal, MPC_RTI_NOMINAL_BUILD_SEED_STEP);
            plant = step.next;
        }
    }

    /* A shifted nominal can retain minimum-drive throttle after a rejected
     * or heavily repaired cycle, leaving the car effectively stopped even
     * though the trajectory still requests acceleration. Re-seed both cold
     * and warm nominals from rest; this is only a linearization seed. ADMM
     * may change it, and the full nonlinear rollout/corridor validation
     * remains the only path to a published command. */
    MpcRtiReference_t launch_reference = {0};
    if (current_state->plant.u < 0.5f &&
        configuration->startup_nominal_throttle_seed >
            MPC_THROTTLE_MIN &&
        reference_at_progress(trajectory, trajectory_count, lap_length,
            progress_seed[1], configuration, &launch_reference) &&
        launch_reference.u > current_state->plant.u + 1.0f) {
        for (int k = 0; k < horizon; ++k)
            nominal->controls[k].throttle_command =
                configuration->startup_nominal_throttle_seed;
    }

    for (int k = 0; k < horizon; ++k)
        nominal->controls[k].throttle_command = clampf_rti(
            nominal->controls[k].throttle_command,
            MPC_THROTTLE_MIN_DRIVE, MPC_THROTTLE_MAX);

    MpcRtiReference_t pass_a_references[PREDICTION_HORIZON + 1];
    for (int k = 0; k <= horizon; ++k) {
        if (!reference_at_progress(trajectory, trajectory_count, lap_length,
                progress_seed[k], configuration, &pass_a_references[k]))
            return fail_nominal_build(
                nominal, MPC_RTI_NOMINAL_BUILD_PASS_A_REFERENCE);
    }
    MpcRtiState_t pass_a_states[PREDICTION_HORIZON + 1];
    double pass_a_progress[PREDICTION_HORIZON + 1];
    if (!rollout_nominal_controls(current_state, current_progress,
            nominal->controls, pass_a_references, horizon, prediction_dt,
            pass_a_states, pass_a_progress))
        return fail_nominal_build(
            nominal, MPC_RTI_NOMINAL_BUILD_PASS_A_ROLLOUT);

    /* Pass B resamples path data from Pass A's exact nonlinear delta_s. */
    for (int k = 0; k <= horizon; ++k) {
        if (!reference_at_progress(trajectory, trajectory_count, lap_length,
                pass_a_progress[k], configuration, &references[k]))
            return fail_nominal_build(
                nominal, MPC_RTI_NOMINAL_BUILD_PASS_B_REFERENCE);
    }
    /* The initial condition is fixed by odometry; only future samples track
     * the requested trajectory speed. */
    references[0].u = current_state->plant.u;
    /* Keep the trajectory's requested speed as the optimization target.
     * Clipping it to a short source-plant reachable envelope made the target
     * collapse toward the current speed while stopped/slow, so the optimizer
     * saw little speed error and emitted tiny throttle.  The nonlinear source
     * rollout itself remains the feasibility check; it already accounts for
     * command delay, wheel spin, and the reachable body speed. */
    align_yaw_rate_references(references, horizon);
    /* Normal MPC decisions are positive motor commands only. The physical
     * plant still models q=0 as Unity hard braking for already queued inputs. */
    if (!rollout_nominal_controls(current_state, current_progress,
            nominal->controls, references, horizon, prediction_dt,
            nominal->states, nominal->progress))
        return fail_nominal_build(
            nominal, MPC_RTI_NOMINAL_BUILD_FINAL_ROLLOUT);
    for (int k = 0; k <= horizon; ++k) {
        const MpcRtiCommandEnvelopeFailure_t failure =
            command_envelope_failure(&nominal->states[k], configuration);
        if (failure != MPC_RTI_COMMAND_ENVELOPE_OK) {
            nominal->build_failure_constraint = failure;
            nominal->build_failure_index = k;
            return fail_nominal_build(
                nominal, MPC_RTI_NOMINAL_BUILD_COMMAND_ENVELOPE);
        }
    }
    for (int k = 0; k <= horizon; ++k)
        nominal->speed_reference[k] = references[k].u;
    nominal->speed_reference_valid = 1;

    nominal->valid = 1;
    return 1;
}

static MpcRtiRolloutStatus_t rollout_candidate_impl(
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
    int *failure_stage,
    MpcRtiRolloutFailure_t *failure)
{
    if (failure_stage) *failure_stage = -1;
    if (failure) memset(failure, 0, sizeof(*failure));
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
    if (nominal_references)
        candidate_reference.u = nominal_references[0].u;
    /* x0 is measured and immutable. The QP applies corridor constraints only
     * to x1..xN, so exact validation follows that same convention. */
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
        if (!isfinite(control->steering_target_rad) ||
            !isfinite(control->throttle_command)) {
            if (failure_stage) *failure_stage = k;
            return MPC_RTI_ROLLOUT_INVALID_INPUT;
        }
        if (control->steering_target_rad <
                -configuration->max_steering_rad - 1.0e-5f ||
            control->steering_target_rad >
                configuration->max_steering_rad + 1.0e-5f ||
            control->throttle_command <
                MPC_THROTTLE_MIN - 1.0e-5f ||
            control->throttle_command >
                MPC_THROTTLE_MAX + 1.0e-5f) {
            if (failure_stage) *failure_stage = k;
            return MPC_RTI_ROLLOUT_COMMAND_LIMIT;
        }
        if (!reference_at_progress(trajectory, trajectory_count, lap_length,
                candidate_progress[k], configuration, &candidate_reference)) {
            if (failure_stage) *failure_stage = k;
            return MPC_RTI_ROLLOUT_INVALID_INPUT;
        }
        if (nominal_references)
            candidate_reference.u = nominal_references[k].u;
        const MpcStageResult_t step = mpc_vehicle_model_step(
            &states[k].plant, control, prediction_dt,
            candidate_reference.path_curvature);
        if (!step.valid) {
            if (failure_stage) *failure_stage = k;
            return MPC_RTI_ROLLOUT_INVALID_MODEL;
        }
        states[k + 1].plant = step.next;
        states[k + 1].incoming_branch_flags = step.branch_flags;
        states[k + 1].incoming_body_forward_delta_m =
            step.body_forward_delta_m;
        states[k + 1].incoming_body_lateral_delta_m =
            step.body_lateral_delta_m;
        states[k + 1].incoming_body_yaw_delta_rad =
            step.body_yaw_delta_rad;
        store_slip_metrics(&states[k + 1], &step);
        states[k + 1].incoming_transition_valid = 1;
        candidate_progress[k + 1] = candidate_progress[k] + step.delta_s_m;
        if (progress) progress[k + 1] = candidate_progress[k + 1];
        if (!state_inside_command_envelope(&states[k + 1], configuration)) {
            if (failure_stage) *failure_stage = k;
            return MPC_RTI_ROLLOUT_STATE_LIMIT;
        }
        if (!reference_at_progress(trajectory, trajectory_count, lap_length,
                candidate_progress[k + 1], configuration, &candidate_reference)) {
            if (failure_stage) *failure_stage = k;
            return MPC_RTI_ROLLOUT_INVALID_INPUT;
        }
        if (nominal_references)
            candidate_reference.u = nominal_references[k + 1].u;
        if (!finite_reference(&candidate_reference)) {
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
        const float margin = corridor_margin_for_state(configuration,
            &states[k + 1].plant, k + 1);
        const float lower_bound = margin - candidate_reference.right_bound -
            configuration->nonlinear_corridor_tolerance_m;
        const float upper_bound = candidate_reference.left_bound - margin +
            configuration->nonlinear_corridor_tolerance_m;
        const int in_corridor = corridor_contains(&states[k + 1],
            &candidate_reference, configuration, margin);
        if (!in_corridor) {
            if (failure_stage) *failure_stage = k;
            record_rollout_failure(failure, k, &states[k + 1],
                &candidate_reference, candidate_progress[k + 1],
                margin, lower_bound, upper_bound);
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
    return rollout_candidate_impl(
        initial_state, initial_progress, controls, trajectory,
        trajectory_count, lap_length, nominal_references, nominal_progress,
        path_delta, horizon, prediction_dt, configuration, states,
        progress, failure_stage, NULL);
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
    if (!configuration) return 0;
    for (int input = 0; input < RICCATI_MAX_NU; ++input) {
        const float rho_input = configuration->solver.rho_u_by_input[input];
        if (!finite_nonnegative(rho_input) ||
            (rho_input > 0.0f && (rho_input < 1.0f || rho_input > 127.0f)))
            return 0;
    }
    return valid_configuration(&configuration->model) &&
        isfinite(configuration->solver.rho) &&
        configuration->solver.rho > 0.0f &&
        isfinite(configuration->solver.rho_u) &&
        configuration->solver.rho_u > 0.0f &&
        finite_nonnegative(configuration->solver.rho_max) &&
        (configuration->solver.rho_max == 0.0f ||
         (configuration->solver.rho_max >= 1.0f &&
          configuration->solver.rho_max <= 127.0f)) &&
        isfinite(configuration->solver.tolerance) &&
        configuration->solver.tolerance > 0.0f &&
        configuration->solver.max_iterations > 0 &&
        finite_nonnegative(configuration->max_cycle_time_us);
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
    int admm_status;
    float state_primal_residual;
    float state_dual_residual;
    float input_primal_residual;
    float input_dual_residual;
    int state_primal_stage;
    int state_primal_index;
    int state_dual_stage;
    int state_dual_index;
    int input_primal_stage;
    int input_primal_index;
    int input_dual_stage;
    int input_dual_index;
    float maximum_regularization;
    int regularization_count;
    int nonsmooth_columns;
    int linearization_failure_stage;
    int linearization_failure_reason;
    int linearization_failure_index;
    int nonlinear_failure_stage;
    int nonlinear_failure_reason;
    MpcRtiRolloutFailure_t nonlinear_failure;
    int first_action_valid;
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
    double qp_build_us;
    double admm_us;
    double exact_rollout_us;
    double nonlinear_validation_us;
    double solve_us;
} MpcRtiPassResult_t;

static double monotonic_microseconds(void)
{
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) return 0.0;
    return (double)now.tv_sec * 1.0e6 + (double)now.tv_nsec * 1.0e-3;
}

/* The live frozen-plan profile measured exact rollout at <=2.675 ms and
 * nonlinear acceptance at <=0.026 ms. Reserve 3.5 ms for those operations
 * plus candidate bookkeeping. A 6 ms reserve made the 20 ms production
 * budget reject the first startup cycle before ADMM, despite the complete
 * command cycle remaining inside the outer 25 ms deadline. */
enum { MPC_RTI_POST_SOLVE_RESERVE_US = 3500 };

static int evaluate_nonlinear_candidate(
    const MpcRtiNominal_t *candidate,
    const MpcTrajectorySample_t *trajectory,
    size_t trajectory_count,
    double lap_length,
    const MpcRtiCycleConfiguration_t *configuration,
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
    double previous_steering =
        candidate->states[0].plant.delayed_steering_command_2;
    double previous_throttle =
        candidate->states[0].plant.delayed_throttle_command_2;
    float min_slack = INFINITY;
    float max_lateral_accel = 0.0f;
    const MpcRtiConfiguration_t *model = &configuration->model;
    for (int k = 0; k <= candidate->horizon; ++k) {
        MpcRtiReference_t reference;
        if (!reference_at_progress(trajectory, trajectory_count, lap_length,
                candidate->progress[k], model, &reference)) return 0;
        if (candidate->speed_reference_valid)
            reference.u = candidate->speed_reference[k];
        if (k > 0 && !corridor_contains(&candidate->states[k], &reference,
                model, corridor_margin_for_state(model,
                    &candidate->states[k].plant, k))) return 0;
        const MpcModelState_t *state = &candidate->states[k].plant;
        const double terminal_scale = k == candidate->horizon
            ? model->terminal_multiplier : 1.0;
        const double errors[] = {
            state->e_y - reference.e_y,
            state->e_psi - reference.e_psi,
            state->u - reference.u,
            state->v - reference.v,
            state->r - reference.r,
            state->delayed_steering_command_2 - reference.steering_command,
            state->actual_steering_angle - reference.steering_command};
        const double speed_weight = state->u > reference.u
            ? model->weight_u_overspeed : model->weight_u;
        const double weights[] = {
            model->weight_e_y, model->weight_e_psi, speed_weight,
            model->weight_v, model->weight_r,
            model->weight_steering_command,
            model->weight_steering_command};
        for (size_t i = 0; i < sizeof(errors) / sizeof(errors[0]); ++i)
            cost += terminal_scale * weights[i] * errors[i] * errors[i];

        if (k > 0) {
            for (int wheel = 0; wheel < 4; ++wheel) {
                const double excess =
                    candidate->states[k].incoming_wheel_slip_excess_mps[wheel];
                cost += terminal_scale * model->weight_longitudinal_slip *
                    excess * excess;
            }
        }

        if (k > 0) {
            const float margin = corridor_margin_for_state(model, state, k);
            const float left_slack = reference.left_bound - margin -
                state->e_y + model->nonlinear_corridor_tolerance_m;
            const float right_slack = reference.right_bound - margin +
                state->e_y + model->nonlinear_corridor_tolerance_m;
            min_slack = fminf(min_slack, fminf(left_slack, right_slack));
        }
        lateral_accel_proxy_by_stage[k] = fabsf(state->u * state->r);
        if (lateral_accel_proxy_by_stage[k] > max_lateral_accel) {
            max_lateral_accel = lateral_accel_proxy_by_stage[k];
            *lateral_accel_proxy_stage = k;
        }
        if (k == candidate->horizon) continue;

        const MpcModelControl_t *control = &candidate->controls[k];
        cost += model->weight_steering_target *
                control->steering_target_rad * control->steering_target_rad +
            model->weight_throttle_command *
                control->throttle_command * control->throttle_command;
        const double steering_delta =
            (double)control->steering_target_rad - previous_steering;
        const double throttle_delta =
            (double)control->throttle_command - previous_throttle;
        cost += model->weight_steering_slew *
                steering_delta * steering_delta +
            model->weight_throttle_slew *
                throttle_delta * throttle_delta;
        previous_steering = control->steering_target_rad;
        previous_throttle = control->throttle_command;
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
    double cycle_deadline_us,
    MpcRtiPassResult_t *pass)
{
    memset(pass, 0, sizeof(*pass));
    pass->status = MPC_RTI_CYCLE_REJECTED_INPUT;
    pass->nonlinear_failure_stage = -1;
    pass->nonlinear_failure_reason = -1;
    pass->linearization_failure_stage = -1;
    pass->linearization_failure_index = -1;
    pass->admm_status = -1;
    pass->nonlinear_objective = INFINITY;
    pass->minimum_corridor_slack = -INFINITY;
    const double start_us = monotonic_microseconds();
    MpcRtiProblem_t problem = {0};
    RiccatiSolution_t solution = {0};
    RiccatiStatus_t solver_status = RICCATI_STATUS_ERROR;
    const float previous_rho = memory->solver_state.initialized
        ? memory->solver_state.rho : configuration->solver.rho;
    const float previous_rho_u = memory->solver_state.initialized
        ? memory->solver_state.rho_u : configuration->solver.rho_u;
    RiccatiDebugInfo_t debug = {0};
    problem.linearization_failure_stage = -1;
    problem.linearization_failure_index = -1;
    double phase_start_us = monotonic_microseconds();
    if (!mpc_rti_build_ltv_qp(nominal->states, nominal->controls,
            references, horizon, prediction_dt, &configuration->model,
            &problem)) {
        pass->linearization_failure_stage =
            problem.linearization_failure_stage;
        pass->linearization_failure_reason =
            problem.linearization_failure_reason;
        pass->linearization_failure_index =
            problem.linearization_failure_index;
        pass->solve_us = monotonic_microseconds() - start_us;
        return pass->status;
    }
    pass->qp_build_us = monotonic_microseconds() - phase_start_us;
    pass->nonsmooth_columns = problem.nonsmooth_jacobian_columns;

    RiccatiAdmmConfig_t solver_config = configuration->solver;
    if (cycle_deadline_us > 0.0) {
        const double solver_budget_us = cycle_deadline_us -
            monotonic_microseconds() - (double)MPC_RTI_POST_SOLVE_RESERVE_US;
        if (solver_budget_us <= 0.0) {
            pass->status = MPC_RTI_CYCLE_REJECTED_SOLVER;
            pass->solve_us = monotonic_microseconds() - start_us;
            return pass->status;
        }
        if (solver_config.max_solve_time_us <= 0.0 ||
            solver_config.max_solve_time_us > solver_budget_us)
            solver_config.max_solve_time_us = solver_budget_us;
    }

    phase_start_us = monotonic_microseconds();
    solver_status = riccati_admm_solve(
        problem.steps, problem.terminal_Q, problem.terminal_q,
        problem.terminal_x_lb, problem.terminal_x_ub, problem.x0,
        &solver_config, &memory->solver_state, &solution);
    pass->admm_us = monotonic_microseconds() - phase_start_us;
    riccati_debug_get_last(&debug);
    pass->rho_final = debug.rho;
    pass->rho_u_final = debug.rho_u;
    pass->maximum_regularization = debug.max_control_hessian_regularization;
    pass->regularization_count = debug.control_hessian_regularization_count;
    pass->rho_change_count = debug.rho_change_count;
    pass->factorization_count = debug.quadratic_factorization_count;
    pass->factorization_time_ns = debug.quadratic_factorization_time_ns;

    pass->nonsmooth_columns = problem.nonsmooth_jacobian_columns;
    pass->admm_status = solver_status;
    pass->iterations = solution.iterations;
    pass->primal_residual = solution.primal_residual;
    pass->dual_residual = solution.dual_residual;
    pass->state_primal_residual = solution.state_primal_residual;
    pass->state_dual_residual = solution.state_dual_residual;
    pass->input_primal_residual = solution.input_primal_residual;
    pass->input_dual_residual = solution.input_dual_residual;
    pass->state_primal_stage = solution.state_primal_stage;
    pass->state_primal_index = solution.state_primal_index;
    pass->state_dual_stage = solution.state_dual_stage;
    pass->state_dual_index = solution.state_dual_index;
    pass->input_primal_stage = solution.input_primal_stage;
    pass->input_primal_index = solution.input_primal_index;
    pass->input_dual_stage = solution.input_dual_stage;
    pass->input_dual_index = solution.input_dual_index;
    pass->rho_start = previous_rho;
    pass->rho_u_start = previous_rho_u;
    if (!isfinite(solution.primal_residual) ||
        !isfinite(solution.dual_residual) || solver_status == RICCATI_STATUS_ERROR) {
        pass->status = MPC_RTI_CYCLE_REJECTED_SOLVER;
        pass->solve_us = monotonic_microseconds() - start_us;
        return pass->status;
    }
    MpcModelControl_t candidate_controls[PREDICTION_HORIZON];
    memcpy(candidate_controls, nominal->controls,
        sizeof(candidate_controls));
    for (int k = 0; k < RICCATI_HORIZON; ++k) {
        /* The condensed QP starts at the first command that can affect the
         * plant. Project its direct q[k] decision into that stage's actuator
         * box; the last two beyond-horizon commands stay nominal and cannot
         * affect this rollout or its first action. */
        const RiccatiStepData_t *command_stage = &problem.steps[k];
        candidate_controls[k].steering_target_rad = clampf_rti(
            solution.u[k][0], command_stage->u_lb[0],
            command_stage->u_ub[0]);
        candidate_controls[k].throttle_command = clampf_rti(
            solution.u[k][1], command_stage->u_lb[1],
            command_stage->u_ub[1]);
    }
    /* Do not overwrite the solved command: the exact candidate rollout must
     * validate the action actually selected by the drive-only QP. */
    pass->first_action = candidate_controls[0];
    pass->first_action_valid = isfinite(pass->first_action.steering_target_rad) &&
        isfinite(pass->first_action.throttle_command) &&
        pass->first_action.steering_target_rad >=
            -configuration->model.max_steering_rad - 1.0e-5f &&
        pass->first_action.steering_target_rad <=
            configuration->model.max_steering_rad + 1.0e-5f &&
        pass->first_action.throttle_command >=
            MPC_THROTTLE_MIN_DRIVE - 1.0e-5f &&
        pass->first_action.throttle_command <=
            MPC_THROTTLE_MAX + 1.0e-5f;

    if (solver_status != RICCATI_STATUS_OPTIMAL) {
        pass->status = MPC_RTI_CYCLE_REJECTED_SOLVER;
        pass->solve_us = monotonic_microseconds() - start_us;
        return pass->status;
    }

    pass->candidate.horizon = horizon;
    pass->candidate.speed_reference_valid = nominal->speed_reference_valid;
    memcpy(pass->candidate.speed_reference, nominal->speed_reference,
        sizeof(pass->candidate.speed_reference));
    memcpy(pass->candidate.controls, candidate_controls,
           (size_t)horizon * sizeof(candidate_controls[0]));
    pass->first_action = candidate_controls[0];
    MpcRtiState_t candidate_states[PREDICTION_HORIZON + 1];
    double candidate_progress[PREDICTION_HORIZON + 1];
    phase_start_us = monotonic_microseconds();
    const MpcRtiRolloutStatus_t rollout_status =
        rollout_candidate_impl(
        current_state, current_progress, candidate_controls, trajectory,
        trajectory_count, lap_length, references, nominal->progress,
        &pass->path_delta, horizon, prediction_dt, &configuration->model,
        candidate_states, candidate_progress,
        &pass->nonlinear_failure_stage, &pass->nonlinear_failure);
    pass->exact_rollout_us = monotonic_microseconds() - phase_start_us;
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
    phase_start_us = monotonic_microseconds();
    const int candidate_valid = evaluate_nonlinear_candidate(&pass->candidate, trajectory,
        trajectory_count, lap_length, configuration,
            &pass->nonlinear_objective, &pass->minimum_corridor_slack,
            &pass->lateral_accel_proxy, &pass->lateral_accel_proxy_stage,
            pass->lateral_accel_proxy_by_stage);
    pass->nonlinear_validation_us = monotonic_microseconds() - phase_start_us;
    if (!candidate_valid) {
        /* `candidate.valid` means that the candidate passed the complete
         * nonlinear acceptance gate, not merely that the rollout produced
         * finite state arrays.  Keeping it set here lets the caller mistake
         * a corridor-invalid candidate for a publishable residual candidate
         * and can also select states[1] instead of the bounded first action. */
        pass->candidate.valid = 0;
        pass->status = MPC_RTI_CYCLE_REJECTED_NONLINEAR_ROLLOUT;
        pass->nonlinear_failure_reason = MPC_RTI_ROLLOUT_INVALID_MODEL;
        pass->solve_us = monotonic_microseconds() - start_us;
        return pass->status;
    }
    pass->status = MPC_RTI_CYCLE_ACCEPTED_OPTIMAL;
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
    const double cycle_start_us = monotonic_microseconds();
    if (result) {
        memset(result, 0, sizeof(*result));
        result->status = MPC_RTI_CYCLE_REJECTED_INPUT;
        result->admm_status = -1;
        result->nonlinear_failure_reason = -1;
        result->nonlinear_failure_stage = -1;
        result->linearization_failure_stage = -1;
        result->linearization_failure_index = -1;
        result->minimum_predicted_corridor_slack_m = NAN;
        result->lateral_accel_proxy_mps2 = NAN;
        result->lateral_accel_proxy_stage = -1;
        result->nonlinear_objective = NAN;
        result->total_cycle_us = 0.0;
    }
    if (!result || !memory || !finite_rti_state(current_state) ||
        !isfinite(current_progress) || !trajectory || trajectory_count == 0 ||
        !isfinite(lap_length) || lap_length <= 0.0 ||
        !valid_cycle_configuration(configuration) ||
        horizon != PREDICTION_HORIZON || !isfinite(prediction_dt) ||
        prediction_dt <= 0.0f) {
        if (result)
            result->input_failure_reason = MPC_RTI_INPUT_FAILURE_CYCLE_ARGUMENTS;
        if (result) result->total_cycle_us = monotonic_microseconds() - cycle_start_us;
        return MPC_RTI_CYCLE_REJECTED_INPUT;
    }

    MpcRtiState_t prepared_state = *current_state;
    if (!vehicle_model_initialize_plant_state(&prepared_state.plant)) {
        result->input_failure_reason = MPC_RTI_INPUT_FAILURE_PLANT_STATE;
        result->total_cycle_us = monotonic_microseconds() - cycle_start_us;
        return MPC_RTI_CYCLE_REJECTED_INPUT;
    }

    const MpcRtiMemory_t memory_before = *memory;
    const double deadline_us = configuration->max_cycle_time_us > 0.0
        ? cycle_start_us + configuration->max_cycle_time_us : 0.0;
    MpcRtiNominal_t nominal;
    MpcRtiReference_t references[PREDICTION_HORIZON + 1];
    if (!mpc_rti_build_nominal(&prepared_state, current_progress,
            &memory->nominal, trajectory, trajectory_count, lap_length,
            prediction_dt, horizon, &configuration->model, &nominal,
            references)) {
        result->input_failure_reason = MPC_RTI_INPUT_FAILURE_NOMINAL_BUILD;
        result->nominal_build_failure_stage = nominal.build_failure_stage;
        result->nominal_build_failure_constraint =
            nominal.build_failure_constraint;
        result->nominal_build_failure_index = nominal.build_failure_index;
        result->total_cycle_us = monotonic_microseconds() - cycle_start_us;
        return MPC_RTI_CYCLE_REJECTED_INPUT;
    }
    result->nominal_first_control = nominal.controls[0];

    MpcRtiPassResult_t pass;
    memset(&pass, 0, sizeof(pass));
    pass.status = MPC_RTI_CYCLE_REJECTED_INPUT;
    pass.nonlinear_failure_stage = -1;
    pass.nonlinear_failure_reason = -1;
    pass.linearization_failure_stage = -1;
    pass.linearization_failure_index = -1;
    pass.admm_status = -1;
    const MpcRtiCycleStatus_t status = solve_rti_pass(
        &prepared_state, current_progress, &nominal, references,
        trajectory, trajectory_count, lap_length, prediction_dt, horizon,
        configuration, memory, deadline_us, &pass);

    result->solver_iterations = pass.iterations;
    result->admm_status = pass.admm_status;
    result->primal_residual = pass.primal_residual;
    result->dual_residual = pass.dual_residual;
    result->maximum_regularization = pass.maximum_regularization;
    result->regularization_count = pass.regularization_count;
    result->nonsmooth_jacobian_columns = pass.nonsmooth_columns;
    result->linearization_failure_stage = pass.linearization_failure_stage;
    result->linearization_failure_reason = pass.linearization_failure_reason;
    result->linearization_failure_index = pass.linearization_failure_index;
    result->nonlinear_failure_stage = pass.nonlinear_failure_stage;
    result->nonlinear_failure_reason = pass.nonlinear_failure_reason;
    result->nonlinear_failure = pass.nonlinear_failure;
    if (pass.candidate.horizon == horizon && pass.first_action_valid) {
        result->candidate_control_count = horizon;
        memcpy(result->candidate_controls, pass.candidate.controls,
            (size_t)horizon * sizeof(result->candidate_controls[0]));
    }
    result->nonlinear_objective = pass.nonlinear_objective;
    result->candidate_path_delta = pass.path_delta;
    result->max_candidate_progress_error_m = max_path_delta_double(
        pass.path_delta.progress_error_m, pass.path_delta.sample_count);
    result->max_candidate_curvature_error_per_m = max_path_delta_float(
        pass.path_delta.curvature_error_per_m, pass.path_delta.sample_count);
    result->max_candidate_left_bound_error_m = max_path_delta_float(
        pass.path_delta.left_bound_error_m, pass.path_delta.sample_count);
    result->max_candidate_right_bound_error_m = max_path_delta_float(
        pass.path_delta.right_bound_error_m, pass.path_delta.sample_count);
    result->minimum_predicted_corridor_slack_m = pass.minimum_corridor_slack;
    result->lateral_accel_proxy_mps2 = pass.lateral_accel_proxy;
    result->lateral_accel_proxy_stage = pass.lateral_accel_proxy_stage;
    memcpy(result->lateral_accel_proxy_by_stage_mps2,
        pass.lateral_accel_proxy_by_stage,
        sizeof(result->lateral_accel_proxy_by_stage_mps2));
    result->rho_start = pass.rho_start;
    result->rho_u_start = pass.rho_u_start;
    result->rho_final = pass.rho_final;
    result->rho_u_final = pass.rho_u_final;
    result->rho_change_count = pass.rho_change_count;
    result->factorization_count = pass.factorization_count;
    result->factorization_time_ns = pass.factorization_time_ns;
    result->qp_build_us = pass.qp_build_us;
    result->admm_us = pass.admm_us;
    result->exact_rollout_us = pass.exact_rollout_us;
    result->nonlinear_validation_us = pass.nonlinear_validation_us;
    result->total_solve_us = pass.solve_us;
    result->total_cycle_us = monotonic_microseconds() - cycle_start_us;

    if (status != MPC_RTI_CYCLE_ACCEPTED_OPTIMAL) {
        *memory = memory_before;
        result->status = status;
        return status;
    }

    result->status = MPC_RTI_CYCLE_ACCEPTED_OPTIMAL;
    result->first_control = pass.first_action;
    result->published_steering_command = pass.first_action.steering_target_rad;
    result->published_throttle = pass.first_action.throttle_command;
    result->minimum_predicted_corridor_slack_m = pass.minimum_corridor_slack;
    memory->nominal = pass.candidate;
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
    }
    for (int input = 0; input < MPC_RTI_NU; ++input)
        cost += 0.5f * stage->R_diag[input] * control[input] * control[input] +
            stage->r[input] * control[input];
    for (int state_index = 0; state_index < MPC_RTI_NX; ++state_index)
        for (int input = 0; input < MPC_RTI_NU; ++input)
            cost += state[state_index] * stage->N[state_index][input] *
                control[input];
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
        !isfinite(control->steering_target_rad) ||
        !isfinite(control->throttle_command)) return NAN;
    const float ey = state->plant.e_y - reference->e_y;
    const float epsi = state->plant.e_psi - reference->e_psi;
    const float eu = state->plant.u - reference->u;
    const float ev = state->plant.v - reference->v;
    const float er = state->plant.r - reference->r;
    const float edelta = state->plant.delayed_steering_command_2 -
        reference->steering_command;
    const float eactual_delta = state->plant.actual_steering_angle -
        reference->steering_command;
    return configuration->weight_e_y * ey * ey +
        configuration->weight_e_psi * epsi * epsi +
        (state->plant.u > reference->u
            ? configuration->weight_u_overspeed : configuration->weight_u) *
            eu * eu +
        (float)source_slip_cost(configuration, &state->plant) +
        configuration->weight_v * ev * ev +
        configuration->weight_r * er * er +
        configuration->weight_steering_command * edelta * edelta +
        configuration->weight_steering_command * eactual_delta * eactual_delta +
        configuration->weight_steering_target *
            control->steering_target_rad * control->steering_target_rad +
        configuration->weight_throttle_command *
            control->throttle_command * control->throttle_command;
}
