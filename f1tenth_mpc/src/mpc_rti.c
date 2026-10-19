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
        configuration->weight_u_overspeed,
        configuration->weight_longitudinal_slip,
        configuration->weight_v,
        configuration->weight_r,
        configuration->weight_steering_command,
        configuration->weight_steering_target,
        configuration->weight_throttle_command,
        configuration->terminal_multiplier,
        configuration->corridor_margin_m,
        configuration->first_prediction_corridor_margin_m,
        configuration->corridor_preview_halfwidth_m,
        configuration->nonlinear_corridor_tolerance_m};
    for (unsigned int i = 0; i < sizeof(weights) / sizeof(weights[0]); ++i)
        if (!finite_nonnegative(weights[i])) return 0;

    return configuration->jacobian_threads >= 0 &&
        configuration->jacobian_threads <= 16 &&
        isfinite(configuration->max_speed_mps) &&
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

int mpc_rti_select_throttle_modes(
    MpcModelControl_t controls[PREDICTION_HORIZON],
    const MpcRtiState_t predicted_states[PREDICTION_HORIZON + 1],
    const MpcRtiReference_t references[PREDICTION_HORIZON + 1],
    int horizon)
{
    if (!controls || !predicted_states || !references || horizon < 1 ||
        horizon > PREDICTION_HORIZON) return 0;
    /* Float roundoff can leave a nominally flat launch segment with a few
     * 1e-11 m/s of apparent overspeed. That must not select Unity's full-brake
     * branch at standstill. */
    const float speed_mode_epsilon_mps = 1.0e-3f;
    for (int k = 0; k < horizon; ++k) {
        /* q[k] enters queue slot 2 in state k+1, slot 1 in state k+2, and is
         * consumed by the source transition k+2 -> k+3. Select its drive or
         * brake branch against that resulting state, including wheel spin;
         * comparing state k+2 is one packet early and can brake just before
         * a positive target acceleration becomes reachable. */
        const int effect_stage = k + 3 <= horizon ? k + 3 : horizon;
        const MpcRtiReference_t *target = &references[effect_stage];
        const MpcRtiState_t *predicted = &predicted_states[effect_stage];
        /* A falling reference-speed derivative is not itself a brake request.
         * q=0 selects Unity's full-brake branch, so applying it while the
         * predicted vehicle is still below the speed target drives it farther
         * from the reference. The delayed effect stage already compares the
         * predicted speed with the speed target at the time this command acts. */
        const int brake =
            predicted->plant.u > target->u + speed_mode_epsilon_mps;
        /* Select only Unity's discrete drive/brake branch here. In drive mode
         * throttle magnitude remains a continuous QP decision in
         * [MPC_THROTTLE_MIN_DRIVE, 1]; never replace it with a fixed
         * full-throttle heuristic. */
        controls[k].throttle_command = brake ? MPC_THROTTLE_MIN :
            clampf_rti(fmaxf(controls[k].throttle_command,
                MPC_THROTTLE_MIN_DRIVE), MPC_THROTTLE_MIN_DRIVE,
                MPC_THROTTLE_MAX);
    }
    return 1;
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

static int corridor_contains_with_schedule(
    const MpcRtiState_t *state,
    const MpcRtiReference_t *reference,
    const MpcRtiConfiguration_t *configuration,
    const MpcRtiCorridorSchedule_t *schedule,
    int prediction_index)
{
    if (corridor_contains(state, reference, configuration,
            corridor_margin_for_state(configuration, &state->plant,
                prediction_index)))
        return 1;
    if (!schedule || schedule->initial_normal_violation <= 1.0e-6f ||
        schedule->horizon < prediction_index ||
        prediction_index <= 0 ||
        !schedule->recovery_stage[prediction_index]) return 0;

    /* During recovery, allow the exact rollout to temporarily use the
     * configured-clearance reserve it needs to turn back toward the nominal
     * corridor. The horizon endpoint is not a recovery stage, so it must
     * re-enter the normal corridor. At every intermediate stage, retain the
     * full heading-dependent footprint and at least 5 cm of wall clearance. */
    MpcRtiConfiguration_t physical = *configuration;
    const float recovery_clearance = 0.05f;
    physical.corridor_margin_m = recovery_clearance;
    physical.first_prediction_corridor_margin_m = recovery_clearance;
    physical.nonlinear_corridor_tolerance_m = 0.0f;
    const float physical_margin = corridor_margin_for_state(
        &physical, &state->plant, prediction_index);
    const float physical_lower = physical_margin - reference->right_bound;
    const float physical_upper = reference->left_bound - physical_margin;
    return isfinite(physical_lower) && isfinite(physical_upper) &&
        physical_lower <= physical_upper &&
        state->plant.e_y >= physical_lower &&
        state->plant.e_y <= physical_upper;
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
    /* The in-flight two-command prefix is known and cannot be optimized.
     * Start at its exact nonlinear endpoint; only linearize the 28 transitions
     * that new throttle/steering decisions can affect inside this horizon. */
    serialize_state(&nominal_states[MPC_COMMAND_DELAY_STEPS], problem->x0);

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
    float nominal_optimizer_states[PREDICTION_HORIZON + 1][MPC_RTI_NX];
    for (int k = 0; k <= horizon; ++k) {
        serialize_state(&nominal_states[k], nominal_optimizer_states[k]);
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
                lin->A[model_row][MPC_STATE_IDX_DELAYED_STEERING_COMMAND_1];
            stage->B[row][1] =
                lin->A[model_row][MPC_STATE_IDX_DELAYED_THROTTLE_COMMAND_1];
            for (int col = 0; col < MPC_OPT_PHYSICAL_NX; ++col) {
                const int model_col = col < 5 ? col : col == 5 ? 7 : col + 4;
                stage->A[row][col] = lin->A[model_row][model_col];
            }
            const float applied_steering =
                nominal_states[model_stage].plant.delayed_steering_command_1;
            const float applied_throttle =
                nominal_states[model_stage].plant.delayed_throttle_command_1;
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

        const int bound_index = model_stage;
        const float margin = corridor_margin_for_state(configuration,
            &nominal_states[bound_index].plant, bound_index);
        const int measured_state_recovery = schedule &&
            schedule->horizon == horizon && schedule->recovery_active &&
            schedule->initial_normal_violation > 1.0e-6f;
        const float ey_lower = measured_state_recovery
            ? schedule->active_lower[bound_index]
            : margin - references[bound_index].right_bound;
        const float ey_upper = measured_state_recovery
            ? schedule->active_upper[bound_index]
            : references[bound_index].left_bound - margin;
        if (!set_state_bounds_with_corridor(stage->x_lb, stage->x_ub,
                ey_lower, ey_upper, configuration)) return 0;

        stage->u_lb[0] = -configuration->max_steering_rad;
        stage->u_ub[0] = configuration->max_steering_rad;
        const MpcModelControl_t *command = &nominal_controls[solver_stage];
        const int brake_mode = command->throttle_command == MPC_THROTTLE_MIN;
        stage->u_lb[1] = brake_mode ? 0.0f : MPC_THROTTLE_MIN_DRIVE;
        stage->u_ub[1] = brake_mode ? 0.0f : MPC_THROTTLE_MAX;
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
    const int measured_state_recovery = schedule &&
        schedule->horizon == horizon && schedule->recovery_active &&
        schedule->initial_normal_violation > 1.0e-6f;
    const float terminal_lower = measured_state_recovery
        ? schedule->active_lower[horizon]
        : terminal_margin - references[horizon].right_bound;
    const float terminal_upper = measured_state_recovery
        ? schedule->active_upper[horizon]
        : references[horizon].left_bound - terminal_margin;
    if (!set_state_bounds_with_corridor(problem->terminal_x_lb,
            problem->terminal_x_ub, terminal_lower, terminal_upper,
            configuration)) return 0;
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
                MPC_THROTTLE_MIN, MPC_THROTTLE_MAX);
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
                plant.delayed_throttle_command_2,
                MPC_THROTTLE_MIN, MPC_THROTTLE_MAX);
            const MpcStageResult_t step = mpc_vehicle_model_step(
                &plant, &nominal->controls[k], prediction_dt,
                references[k].path_curvature);
            if (!step.valid)
                return fail_nominal_build(
                    nominal, MPC_RTI_NOMINAL_BUILD_SEED_STEP);
            plant = step.next;
        }
    }

    for (int k = 0; k < horizon; ++k) {
        if (nominal->controls[k].throttle_command > MPC_THROTTLE_MIN &&
            nominal->controls[k].throttle_command <
                MPC_THROTTLE_MIN_DRIVE)
            nominal->controls[k].throttle_command = MPC_THROTTLE_MIN_DRIVE;
    }

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
    if (!mpc_rti_select_throttle_modes(
            nominal->controls, pass_a_states, references,
            horizon))
        return fail_nominal_build(
            nominal, MPC_RTI_NOMINAL_BUILD_THROTTLE_MODE);

    /* Throttle mode remains source-specific: positive q drives; exact zero
     * engages Unity's hard-brake branch. */
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
        const float margin = corridor_margin_for_state(configuration,
            &seed->states[k].plant, k);
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

    /* A measured state can begin just outside the configured clearance even
     * though the vehicle remains physically inside the track. Requiring x1
     * to satisfy the normal corridor makes that state unrecoverable whenever
     * lateral position cannot change in one 25 ms packet (notably at rest).
     * Keep the immutable normal bounds, but give the QP/exact rollout the
     * same one-sided allowance, tapering the measured violation to zero by
     * the horizon. Exact validation separately enforces the physical car
     * footprint with at least 5 cm of wall clearance. */
    if (schedule->initial_normal_violation > 1.0e-6f) {
        const int recovery_end =
            schedule->first_normal_feasible_stage > 0
            ? schedule->first_normal_feasible_stage : seed->horizon;
        if (schedule->first_normal_feasible_stage < 0)
            schedule->recovery_not_found = 1;
        const float initial_allowance =
            schedule->initial_normal_violation +
            configuration->nonlinear_corridor_tolerance_m;
        const int recover_above = seed->states[0].plant.e_y >
            schedule->normal_upper[0];
        for (int k = 0; k <= seed->horizon; ++k) {
            schedule->active_lower[k] = schedule->normal_lower[k];
            schedule->active_upper[k] = schedule->normal_upper[k];
            const float fraction = k >= recovery_end ? 0.0f :
                (float)(recovery_end - k) / (float)recovery_end;
            const float allowance = initial_allowance * fraction;
            if (recover_above)
                schedule->active_upper[k] += allowance;
            else
                schedule->active_lower[k] -= allowance;
            schedule->recovery_stage[k] =
                (k > 0 && k < recovery_end) ? 1 : 0;
        }
        schedule->recovery_active = 1;
        return 1;
    }
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
        control.steering_target_rad = clampf_rti(
            desired_steering,
            -configuration->max_steering_rad,
            configuration->max_steering_rad);
        if (configuration->recovery_seed_policy ==
            MPC_RTI_RECOVERY_SEED_BRAKE_HEADING_FEEDBACK) {
            // This policy models the simulator's discrete no-throttle action:
            // full commanded braking is the bounded recovery alternative.
            control.throttle_command = MPC_THROTTLE_MIN;
        } else {
            control.throttle_command = clampf_rti(
                control.throttle_command,
                MPC_THROTTLE_MIN,
                MPC_THROTTLE_MAX);
        }

        const MpcStageResult_t step = mpc_vehicle_model_step(
            &seed->states[k].plant, &control, prediction_dt,
            reference.path_curvature);
        if (!step.valid || !isfinite(seed->progress[k] + step.delta_s_m))
            return 0;
        seed->controls[k] = control;
        seed->states[k + 1].plant = step.next;
        seed->states[k + 1].incoming_branch_flags = step.branch_flags;
        seed->states[k + 1].incoming_body_forward_delta_m =
            step.body_forward_delta_m;
        seed->states[k + 1].incoming_body_lateral_delta_m =
            step.body_lateral_delta_m;
        seed->states[k + 1].incoming_body_yaw_delta_rad =
            step.body_yaw_delta_rad;
        seed->states[k + 1].incoming_transition_valid = 1;
        seed->progress[k + 1] = seed->progress[k] + step.delta_s_m;
        if (!state_inside_command_envelope(&seed->states[k + 1], configuration))
            return 0;
    }
    seed->valid = 1;
    return 1;
}

/* Build the one schedule consumed by both the QP and the exact rollout.
 * A nominal rollout is the default deterministic bounded recovery seed. If
 * configured, one bounded heading-feedback or braking-feedback policy is
 * evaluated only after nominal feasibility fails. This is intentionally a
 * single O(N) policy evaluation, not an online beam search. If measured x0
 * starts outside the requested clearance, a taper is retained to the horizon
 * endpoint even when this seed does not re-enter the nominal corridor; exact
 * validation still enforces the 5 cm physical wall-clearance floor. */
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
        if (control->throttle_command > MPC_THROTTLE_MIN &&
            control->throttle_command < MPC_THROTTLE_MIN_DRIVE) {
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
        const int in_corridor = corridor_contains_with_schedule(
            &states[k + 1], &candidate_reference, configuration, schedule,
            k + 1);
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
    return mpc_rti_rollout_candidate_with_schedule(
        initial_state, initial_progress, controls, trajectory,
        trajectory_count, lap_length, nominal_references, nominal_progress,
        path_delta, horizon, prediction_dt, configuration, NULL, states,
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
            configuration->rti2_steering_target_correction_trigger_rad) &&
        finite_nonnegative(
            configuration->rti2_throttle_correction_trigger) &&
        (configuration->refinement_mode != MPC_RTI_REFINEMENT_ADAPTIVE ||
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
        configuration->max_consecutive_degraded_solves >= 0 &&
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
    int throttle_mode_switches;
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

enum
{
    /* Reserve time after ADMM for exact nonlinear rollout/validation and at
     * least one bounded corridor-repair rollout if R1 is infeasible. */
    MPC_RTI_POST_SOLVE_RESERVE_US = 6000,
    /* A measured exact plant horizon rollout costs about 0.6 ms at the
     * median and 1.2 ms at p99. Corridor repair is cheaper than another QP,
     * so allow a trial with this reserve instead of requiring R2's 8 ms. */
    MPC_RTI_CORRIDOR_REPAIR_TRIAL_RESERVE_US = 2000,
    MPC_RTI2_MIN_REMAINING_US = 8000
};

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
        if (candidate->speed_reference_valid)
            reference.u = candidate->speed_reference[k];
        if (k > 0) {
            if (!corridor_contains_with_schedule(&candidate->states[k],
                    &reference, model, schedule, k)) return 0;
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
    }
    if (!isfinite(cost) || !isfinite(min_slack) ||
        !isfinite(max_lateral_accel)) return 0;
    *objective = (float)cost;
    *minimum_corridor_slack = min_slack;
    *lateral_accel_proxy = max_lateral_accel;
    return 1;
}

/* A single RTI linearization can produce a QP step whose exact nonlinear
 * rollout leaves the corridor even though the previous exact nominal is
 * feasible.  Do not widen the corridor or accept that rollout.  Instead,
 * backtrack the QP control increment toward the exact nominal and validate
 * every trial with the same nonlinear model and corridor schedule.  This is
 * an ordinary trust-region/line-search repair: the returned candidate is
 * publishable only after the existing exact rollout and objective checks
 * pass. */
static int build_exact_corridor_repair_candidate(
    const MpcRtiState_t *current_state,
    double current_progress,
    const MpcRtiNominal_t *nominal,
    const MpcRtiPassResult_t *failed_pass,
    const MpcRtiReference_t references[PREDICTION_HORIZON + 1],
    const MpcTrajectorySample_t *trajectory,
    size_t trajectory_count,
    double lap_length,
    float prediction_dt,
    int horizon,
    const MpcRtiCycleConfiguration_t *configuration,
    const MpcRtiCorridorSchedule_t *schedule,
    MpcRtiNominal_t *candidate,
    MpcRtiCandidatePathDelta_t *path_delta,
    float *alpha_out,
    double cycle_deadline_us,
    int *budget_skipped,
    unsigned int *trials_attempted,
    double *repair_time_us)
{
    if (!current_state || !nominal || !nominal->valid || !failed_pass ||
        !references || !trajectory ||
        !configuration || !candidate || !path_delta || !alpha_out ||
        horizon < 1 || horizon > PREDICTION_HORIZON) return 0;
    if (budget_skipped) *budget_skipped = 0;
    if (trials_attempted) *trials_attempted = 0;
    if (repair_time_us) *repair_time_us = 0.0;
    const double repair_start_us = monotonic_microseconds();

    /* The failed pass stores its QP controls even when its nonlinear rollout
     * is rejected.  The invalid candidate flag is expected in this path, so
     * only the controls are copied below. */
    const float alphas[] = {0.75f, 0.50f, 0.25f, 0.125f, 0.0625f, 0.0f};
    for (size_t trial = 0; trial < sizeof(alphas) / sizeof(alphas[0]);
         ++trial) {
        if (cycle_deadline_us > 0.0 &&
            cycle_deadline_us - monotonic_microseconds() <
                (double)MPC_RTI_CORRIDOR_REPAIR_TRIAL_RESERVE_US) {
            if (budget_skipped) *budget_skipped = 1;
            break;
        }
        if (trials_attempted) ++*trials_attempted;
        const float alpha = alphas[trial];
        MpcModelControl_t controls[PREDICTION_HORIZON] = {0};
        int controls_valid = 1;
        for (int k = 0; k < horizon; ++k) {
            const MpcModelControl_t nominal_control = nominal->controls[k];
            const MpcModelControl_t qp_control = failed_pass->candidate.controls[k];
            controls[k].steering_target_rad = nominal_control.steering_target_rad +
                alpha * (qp_control.steering_target_rad - nominal_control.steering_target_rad);
            const int nominal_brakes = nominal_control.throttle_command ==
                MPC_THROTTLE_MIN;
            const int qp_brakes = qp_control.throttle_command ==
                MPC_THROTTLE_MIN;
            const int selected_brake_mode = nominal_brakes == qp_brakes
                ? nominal_brakes : (alpha >= 0.5f ? qp_brakes : nominal_brakes);
            if (selected_brake_mode) {
                /* q=0 is a discrete Unity hard-brake mode, not the low end
                 * of a continuous throttle interpolation. Never turn a
                 * drive/brake line-search blend into a small positive drive
                 * command and call it braking. */
                controls[k].throttle_command = MPC_THROTTLE_MIN;
            } else {
                const float nominal_drive = nominal_brakes
                    ? MPC_THROTTLE_MIN_DRIVE
                    : nominal_control.throttle_command;
                const float qp_drive = qp_brakes
                    ? MPC_THROTTLE_MIN_DRIVE
                    : qp_control.throttle_command;
                controls[k].throttle_command = fmaxf(
                    MPC_THROTTLE_MIN_DRIVE,
                    nominal_drive + alpha * (qp_drive - nominal_drive));
            }
            if (!isfinite(controls[k].steering_target_rad) ||
                !isfinite(controls[k].throttle_command)) {
                controls_valid = 0;
                break;
            }
        }
        if (!controls_valid) continue;

        MpcRtiState_t states[PREDICTION_HORIZON + 1];
        double progress[PREDICTION_HORIZON + 1];
        MpcRtiCandidatePathDelta_t trial_delta;
        int failure_stage = -1;
        const MpcRtiRolloutStatus_t rollout_status =
            mpc_rti_rollout_candidate_with_schedule(
                current_state, current_progress, controls, trajectory,
                trajectory_count, lap_length, references, nominal->progress,
                &trial_delta, horizon, prediction_dt,
                &configuration->model, schedule, states, progress,
                &failure_stage, NULL);
        if (rollout_status != MPC_RTI_ROLLOUT_OK) continue;

        MpcRtiNominal_t trial_candidate = {0};
        trial_candidate.horizon = horizon;
        trial_candidate.valid = 1;
        trial_candidate.speed_reference_valid = nominal->speed_reference_valid;
        memcpy(trial_candidate.speed_reference, nominal->speed_reference,
            sizeof(trial_candidate.speed_reference));
        memcpy(trial_candidate.controls, controls,
            (size_t)horizon * sizeof(controls[0]));
        memcpy(trial_candidate.states, states,
            (size_t)(horizon + 1) * sizeof(states[0]));
        memcpy(trial_candidate.progress, progress,
            (size_t)(horizon + 1) * sizeof(progress[0]));
        float objective;
        float minimum_corridor_slack;
        float lateral_accel_proxy;
        int lateral_accel_proxy_stage = -1;
        float lateral_accel_proxy_by_stage[PREDICTION_HORIZON + 1] = {0};
        if (!evaluate_nonlinear_candidate(&trial_candidate, trajectory,
                trajectory_count, lap_length, configuration, schedule,
                &objective, &minimum_corridor_slack, &lateral_accel_proxy,
                &lateral_accel_proxy_stage,
                lateral_accel_proxy_by_stage)) continue;

        *candidate = trial_candidate;
        *path_delta = trial_delta;
        *alpha_out = alpha;
        if (repair_time_us)
            *repair_time_us = monotonic_microseconds() - repair_start_us;
        return 1;
    }
    if (repair_time_us)
        *repair_time_us = monotonic_microseconds() - repair_start_us;
    return 0;
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
    double cycle_deadline_us,
    MpcRtiPassResult_t *pass)
{
    memset(pass, 0, sizeof(*pass));
    pass->status = MPC_RTI_CYCLE_REJECTED_INPUT;
    pass->nonlinear_failure_stage = -1;
    pass->nonlinear_failure_reason = -1;
    pass->linearization_failure_stage = -1;
    pass->linearization_failure_index = -1;
    pass->nonlinear_objective = INFINITY;
    pass->minimum_corridor_slack = -INFINITY;
    const double start_us = monotonic_microseconds();
    MpcRtiProblem_t problem = {0};
    problem.linearization_failure_stage = -1;
    problem.linearization_failure_index = -1;
    double phase_start_us = monotonic_microseconds();
    const int qp_valid = mpc_rti_build_ltv_qp_with_schedule(nominal->states,
            nominal->controls, references, horizon, prediction_dt,
            &configuration->model, schedule, &problem);
    pass->qp_build_us = monotonic_microseconds() - phase_start_us;
    if (!qp_valid) {
        pass->linearization_failure_stage =
            problem.linearization_failure_stage;
        pass->linearization_failure_reason =
            problem.linearization_failure_reason;
        pass->linearization_failure_index =
            problem.linearization_failure_index;
        pass->solve_us = monotonic_microseconds() - start_us;
        return pass->status;
    }
    pass->nonsmooth_columns = problem.nonsmooth_jacobian_columns;

    RiccatiSolution_t solution = {0};
    RiccatiAdmmConfig_t solver_config = configuration->solver;
    if (cycle_deadline_us > 0.0) {
        const double remaining_us = cycle_deadline_us -
            monotonic_microseconds();
        const double solver_budget_us = remaining_us -
            (double)MPC_RTI_POST_SOLVE_RESERVE_US;
        if (solver_budget_us <= 0.0) {
            pass->status = MPC_RTI_CYCLE_REJECTED_SOLVER;
            pass->solve_us = monotonic_microseconds() - start_us;
            return pass->status;
        }
        if (solver_config.max_solve_time_us <= 0.0 ||
            solver_config.max_solve_time_us > solver_budget_us) {
            solver_config.max_solve_time_us = solver_budget_us;
        }
    }
    const float previous_rho = memory->solver_state.initialized
        ? memory->solver_state.rho : configuration->solver.rho;
    const float previous_rho_u = memory->solver_state.initialized
        ? memory->solver_state.rho_u : configuration->solver.rho_u;
    phase_start_us = monotonic_microseconds();
    const RiccatiStatus_t solver_status = riccati_admm_solve(
        problem.steps, problem.terminal_Q, problem.terminal_q,
        problem.terminal_x_lb, problem.terminal_x_ub, problem.x0,
        &solver_config,
        &memory->solver_state, &solution);
    pass->admm_us = monotonic_microseconds() - phase_start_us;
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
    /* Do not overwrite the solved throttle here. The QP selected a continuous
     * drive command or the exact-zero brake mode; the candidate is validated
     * with that same command below. */
    pass->first_action = candidate_controls[0];
    pass->first_action_valid = isfinite(pass->first_action.steering_target_rad) &&
        isfinite(pass->first_action.throttle_command) &&
        pass->first_action.steering_target_rad >=
            -configuration->model.max_steering_rad - 1.0e-5f &&
        pass->first_action.steering_target_rad <=
            configuration->model.max_steering_rad + 1.0e-5f &&
        pass->first_action.throttle_command >=
            MPC_THROTTLE_MIN - 1.0e-5f &&
        pass->first_action.throttle_command <=
            MPC_THROTTLE_MAX + 1.0e-5f;

    int accepted_degraded = 0;
    int residual_recovery_candidate = 0;
    if (solver_status == RICCATI_STATUS_MAX_ITERATIONS) {
        const float max_residual = fmaxf(pass->primal_residual,
                                          pass->dual_residual);
        if (max_residual > configuration->degraded_residual_limit ||
            memory->consecutive_degraded_solves + 1 >
                configuration->max_consecutive_degraded_solves) {
            /* Residual magnitude is solver-coordinate dependent and is not
             * itself a physical safety test. Do not reject a finite iterate
             * at an arbitrary second threshold: project its controls into
             * the QP actuator box, then accept it only if the complete
             * source-plant rollout and nonlinear corridor checks pass. */
            residual_recovery_candidate = 1;
        } else {
            accepted_degraded = 1;
        }
    } else if (solver_status != RICCATI_STATUS_OPTIMAL) {
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
        mpc_rti_rollout_candidate_with_schedule(
        current_state, current_progress, candidate_controls, trajectory,
        trajectory_count, lap_length, references, nominal->progress,
        &pass->path_delta, horizon, prediction_dt, &configuration->model,
        schedule, candidate_states, candidate_progress,
        &pass->nonlinear_failure_stage, &pass->nonlinear_failure);
    pass->exact_rollout_us = monotonic_microseconds() - phase_start_us;
    if (rollout_status != MPC_RTI_ROLLOUT_OK) {
        pass->status = MPC_RTI_CYCLE_REJECTED_NONLINEAR_ROLLOUT;
        pass->nonlinear_failure_reason = (int)rollout_status;
        pass->solve_us = monotonic_microseconds() - start_us;
        return pass->status;
    }

    /* The QP uses a smooth local model, but Unity's throttle is hybrid:
     * exact zero selects the hard-brake branch and any positive command
     * selects motor drive. Reclassify each command from the QP candidate's
     * exact predicted speed, not only from the stale nominal used to build
     * the QP. Then replay the whole horizon through the exact source-derived
     * plant before nonlinear acceptance. This prevents a candidate that is
     * already predicted to overspeed from being trapped in positive drive
     * merely because its previous nominal was in drive mode.
     */
    MpcModelControl_t mode_checked_controls[PREDICTION_HORIZON];
    memcpy(mode_checked_controls, candidate_controls,
        (size_t)horizon * sizeof(candidate_controls[0]));
    if (!mpc_rti_select_throttle_modes(mode_checked_controls,
            candidate_states, references, horizon)) {
        pass->status = MPC_RTI_CYCLE_REJECTED_INPUT;
        pass->solve_us = monotonic_microseconds() - start_us;
        return pass->status;
    }
    for (int k = 0; k < horizon; ++k) {
        if (mode_checked_controls[k].throttle_command !=
                candidate_controls[k].throttle_command)
            ++pass->throttle_mode_switches;
    }
    if (pass->throttle_mode_switches > 0) {
        memcpy(candidate_controls, mode_checked_controls,
            (size_t)horizon * sizeof(candidate_controls[0]));
        phase_start_us = monotonic_microseconds();
        const MpcRtiRolloutStatus_t mode_rollout_status =
            mpc_rti_rollout_candidate_with_schedule(
                current_state, current_progress, candidate_controls,
                trajectory, trajectory_count, lap_length, references,
                nominal->progress, &pass->path_delta, horizon,
                prediction_dt, &configuration->model, schedule,
                candidate_states, candidate_progress,
                &pass->nonlinear_failure_stage, &pass->nonlinear_failure);
        pass->exact_rollout_us += monotonic_microseconds() - phase_start_us;
        if (mode_rollout_status != MPC_RTI_ROLLOUT_OK) {
            pass->status = MPC_RTI_CYCLE_REJECTED_NONLINEAR_ROLLOUT;
            pass->nonlinear_failure_reason = (int)mode_rollout_status;
            pass->solve_us = monotonic_microseconds() - start_us;
            return pass->status;
        }
        memcpy(pass->candidate.controls, candidate_controls,
            (size_t)horizon * sizeof(candidate_controls[0]));
        pass->first_action = candidate_controls[0];
    }

    pass->candidate.valid = 1;
    memcpy(pass->candidate.states, candidate_states,
           (size_t)(horizon + 1) * sizeof(candidate_states[0]));
    memcpy(pass->candidate.progress, candidate_progress,
           (size_t)(horizon + 1) * sizeof(candidate_progress[0]));
    phase_start_us = monotonic_microseconds();
    const int candidate_valid = evaluate_nonlinear_candidate(&pass->candidate, trajectory,
            trajectory_count, lap_length, configuration, schedule,
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
        if (!isfinite(control->steering_target_rad) ||
            !isfinite(control->throttle_command) ||
            control->steering_target_rad <
                -configuration->max_steering_rad - 1.0e-5f ||
            control->steering_target_rad >
                configuration->max_steering_rad + 1.0e-5f ||
            control->throttle_command <
                MPC_THROTTLE_MIN -
                    1.0e-5f ||
            control->throttle_command >
                MPC_THROTTLE_MAX +
                    1.0e-5f ||
            !reference_at_progress(trajectory, trajectory_count, lap_length,
                seed->progress[k], configuration, &reference))
            return 0;

        const MpcStageResult_t step = mpc_vehicle_model_step(
            &seed->states[k].plant, control, prediction_dt,
            reference.path_curvature);
        if (!step.valid) return 0;
        seed->states[k + 1].plant = step.next;
        seed->states[k + 1].incoming_branch_flags = step.branch_flags;
        seed->states[k + 1].incoming_body_forward_delta_m =
            step.body_forward_delta_m;
        seed->states[k + 1].incoming_body_lateral_delta_m =
            step.body_lateral_delta_m;
        seed->states[k + 1].incoming_body_yaw_delta_rad =
            step.body_yaw_delta_rad;
        store_slip_metrics(&seed->states[k + 1], &step);
        seed->states[k + 1].incoming_transition_valid = 1;
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
    if (fabsf(r1->first_action.steering_target_rad -
              nominal->controls[0].steering_target_rad) >=
            configuration->rti2_steering_target_correction_trigger_rad ||
        fabsf(r1->first_action.throttle_command -
              nominal->controls[0].throttle_command) >=
            configuration->rti2_throttle_correction_trigger)
        mask |= MPC_RTI2_TRIGGER_ACTION_CORRECTION;
    if (r1->status == MPC_RTI_CYCLE_ACCEPTED_DEGRADED)
        mask |= MPC_RTI2_TRIGGER_DEGRADED_SOLVE;
    const float residual_min = fminf(r1->primal_residual, r1->dual_residual);
    const float residual_max = fmaxf(r1->primal_residual, r1->dual_residual);
    if (residual_max / fmaxf(residual_min, 1.0e-6f) >=
        configuration->rti2_residual_imbalance_trigger)
        mask |= MPC_RTI2_TRIGGER_RESIDUAL_IMBALANCE;
    if (nominal->controls[0].steering_target_rad *
            r1->first_action.steering_target_rad < 0.0f &&
        fabsf(nominal->controls[0].steering_target_rad) > 0.1f &&
        fabsf(r1->first_action.steering_target_rad) > 0.1f)
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
    /* A residual rejection is a numerical convergence failure, not a
     * feasibility failure. Keep the finite ADMM z/y/rho iterate so the next
     * 40 Hz cycle can continue the same constrained solve. If its exact
     * nonlinear candidate is usable, the caller promotes it to degraded
     * authority; hard input, solver, regularity, and nonlinear failures retain
     * the cold-start behavior. */
    if (memory && status != MPC_RTI_CYCLE_REJECTED_RESIDUAL)
        mpc_rti_memory_reset(memory);
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
        result->linearization_failure_stage = -1;
        result->linearization_failure_index = -1;
        result->r1_nonlinear_objective = NAN;
        result->r2_nonlinear_objective = NAN;
        result->r1_min_corridor_slack = NAN;
        result->r2_min_corridor_slack = NAN;
        result->minimum_predicted_corridor_slack_m = NAN;
        result->recovery_reentry_stage = -1;
        result->nominal_build_failure_index = -1;
        result->recovery_initial_normal_violation_m = NAN;
        result->recovery_max_seed_violation_m = NAN;
        result->corridor_repair_alpha = NAN;
        result->lateral_accel_proxy_mps2 = NAN;
        result->lateral_accel_proxy_stage = -1;
        for (int k = 0; k <= PREDICTION_HORIZON; ++k)
            result->lateral_accel_proxy_by_stage_mps2[k] = NAN;
    }
    if (!result || !memory || !finite_rti_state(current_state) ||
        !isfinite(current_progress) || !trajectory || !trajectory_count ||
        !isfinite(lap_length) || !valid_cycle_configuration(configuration) ||
        horizon != PREDICTION_HORIZON ||
        !isfinite(prediction_dt) || prediction_dt <= 0.0f) {
        if (result)
            result->input_failure_reason = MPC_RTI_INPUT_FAILURE_CYCLE_ARGUMENTS;
        return reject_cycle(memory, result, MPC_RTI_CYCLE_REJECTED_INPUT);
    }

    MpcRtiState_t prepared_current_state = *current_state;
    if (!vehicle_model_initialize_plant_state(
            &prepared_current_state.plant)) {
        result->input_failure_reason = MPC_RTI_INPUT_FAILURE_PLANT_STATE;
        return reject_cycle(memory, result, MPC_RTI_CYCLE_REJECTED_INPUT);
    }
    current_state = &prepared_current_state;

    const double cycle_start_us = monotonic_microseconds();
    const double cycle_deadline_us =
        configuration->max_cycle_time_us > 0.0
        ? cycle_start_us + configuration->max_cycle_time_us : 0.0;
    MpcRtiNominal_t nominal;
    MpcRtiReference_t references[PREDICTION_HORIZON + 1];
    if (!mpc_rti_build_nominal(current_state, current_progress,
            &memory->nominal, trajectory, trajectory_count, lap_length,
            prediction_dt, horizon, &configuration->model, &nominal,
            references)) {
        result->input_failure_reason = MPC_RTI_INPUT_FAILURE_NOMINAL_BUILD;
        result->nominal_build_failure_stage = nominal.build_failure_stage;
        result->nominal_build_failure_constraint =
            nominal.build_failure_constraint;
        result->nominal_build_failure_index = nominal.build_failure_index;
        result->r1_status = MPC_RTI_CYCLE_REJECTED_INPUT;
        result->total_rti_us = monotonic_microseconds() - cycle_start_us;
        return reject_cycle(memory, result, MPC_RTI_CYCLE_REJECTED_INPUT);
    }
    result->nominal_first_control = nominal.controls[0];
    MpcRtiCorridorSchedule_t corridor_schedule;
    if (!mpc_rti_build_corridor_schedule(current_state, &nominal, trajectory,
            trajectory_count, lap_length, prediction_dt, &configuration->model,
            &corridor_schedule)) {
        result->input_failure_reason =
            MPC_RTI_INPUT_FAILURE_CORRIDOR_SCHEDULE;
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
    const MpcRtiMemory_t memory_before_r1 = *memory;
    MpcRtiPassResult_t r1;
    const MpcRtiCycleStatus_t r1_status = solve_rti_pass(
        current_state, current_progress, &nominal, references, trajectory,
        trajectory_count, lap_length, prediction_dt, horizon, configuration,
        &corridor_schedule, memory, cycle_deadline_us, &r1);
    result->rti_iterations_used = 1;
    result->r1_status = (int)r1_status;
    result->linearization_failure_stage = r1.linearization_failure_stage;
    result->linearization_failure_reason = r1.linearization_failure_reason;
    result->linearization_failure_index = r1.linearization_failure_index;
    result->r1_nonlinear_failure_reason = r1.nonlinear_failure_reason;
    result->r1_nonlinear_failure_stage = r1.nonlinear_failure_stage;
    result->r1_nonlinear_failure = r1.nonlinear_failure;
    result->r1_first_action = r1.first_action;
    result->r1_solver_iterations = r1.iterations;
    result->r1_solve_us = r1.solve_us;
    result->r1_qp_build_us = r1.qp_build_us;
    result->r1_admm_us = r1.admm_us;
    result->r1_exact_rollout_us = r1.exact_rollout_us;
    result->r1_nonlinear_validation_us = r1.nonlinear_validation_us;
    result->r1_nonlinear_objective = r1.nonlinear_objective;
    result->r1_min_corridor_slack = r1.minimum_corridor_slack;
    result->rho_start = r1.rho_start;
    result->rho_u_start = r1.rho_u_start;
    result->rho_final = r1.rho_final;
    result->rho_u_final = r1.rho_u_final;
    result->rho_change_count = r1.rho_change_count;
    result->factorization_count = r1.factorization_count;
    result->factorization_time_ns = r1.factorization_time_ns;
    const int r1_accepted = r1_status == MPC_RTI_CYCLE_ACCEPTED_OPTIMAL ||
        r1_status == MPC_RTI_CYCLE_ACCEPTED_DEGRADED;
    const MpcRtiMemory_t r1_memory = *memory;
    const int repairable_corridor_failure = !r1_accepted &&
        r1_status == MPC_RTI_CYCLE_REJECTED_NONLINEAR_ROLLOUT &&
        r1.nonlinear_failure_reason == MPC_RTI_ROLLOUT_CORRIDOR &&
        r1.nonlinear_failure_stage >= 0;
    const int repairable_residual_failure = !r1_accepted &&
        r1_status == MPC_RTI_CYCLE_REJECTED_RESIDUAL &&
        r1.candidate.valid;
    MpcRtiPassResult_t corridor_repair = r1;
    int corridor_repair_available = 0;
    float corridor_repair_alpha = NAN;
    if (repairable_corridor_failure) {
        int repair_budget_skipped = 0;
        corridor_repair_available = build_exact_corridor_repair_candidate(
            current_state, current_progress, &nominal, &r1, references,
            trajectory, trajectory_count, lap_length, prediction_dt,
            horizon, configuration, &corridor_schedule,
            &corridor_repair.candidate, &corridor_repair.path_delta,
            &corridor_repair_alpha, cycle_deadline_us,
            &repair_budget_skipped, &result->corridor_repair_trials,
            &result->corridor_repair_us);
        if (repair_budget_skipped) result->rti2_budget_skipped = 1;
        if (corridor_repair_available) {
            corridor_repair.status = MPC_RTI_CYCLE_ACCEPTED_DEGRADED;
            corridor_repair.first_action =
                corridor_repair.candidate.controls[0];
            corridor_repair.nonlinear_failure_reason = -1;
            corridor_repair.nonlinear_failure_stage = -1;
            corridor_repair.nonlinear_objective = INFINITY;
            corridor_repair.minimum_corridor_slack = -INFINITY;
            corridor_repair.lateral_accel_proxy_stage = -1;
            float repair_lateral_accel_by_stage[PREDICTION_HORIZON + 1] = {0};
            if (!evaluate_nonlinear_candidate(
                    &corridor_repair.candidate, trajectory,
                    trajectory_count, lap_length, configuration,
                    &corridor_schedule, &corridor_repair.nonlinear_objective,
                    &corridor_repair.minimum_corridor_slack,
                    &corridor_repair.lateral_accel_proxy,
                    &corridor_repair.lateral_accel_proxy_stage,
                    repair_lateral_accel_by_stage)) {
                corridor_repair_available = 0;
            } else {
                memcpy(corridor_repair.lateral_accel_proxy_by_stage,
                    repair_lateral_accel_by_stage,
                    sizeof(repair_lateral_accel_by_stage));
            }
        }
    }
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
    int run_r2 = configuration->refinement_mode == MPC_RTI_REFINEMENT_R2 ||
        (configuration->refinement_mode == MPC_RTI_REFINEMENT_ADAPTIVE &&
         trigger_mask != 0u) ||
        (repairable_corridor_failure &&
         configuration->refinement_mode != MPC_RTI_REFINEMENT_R1);
    /* A residual-limited R1 candidate has already passed the exact nonlinear
     * rollout and hard corridor gate. In R1-only mode it is the requested
     * result; in adaptive mode, retain R2 for low-clearance candidates where
     * refinement could materially improve the reserve. */
    if (repairable_residual_failure &&
        (configuration->refinement_mode == MPC_RTI_REFINEMENT_R1 ||
         (configuration->refinement_mode == MPC_RTI_REFINEMENT_ADAPTIVE &&
          r1.minimum_corridor_slack >
              configuration->rti2_min_corridor_slack_trigger_m))) {
        selected = r1;
        selected.status = MPC_RTI_CYCLE_ACCEPTED_DEGRADED;
        selected_candidate = 1;
        result->residual_candidate_published = 1;
        run_r2 = 0;
    }
    /* R1-only remains one QP solve. If its exact plant rollout misses the
     * corridor, use the existing bounded blend search rather than turning a
     * distant, small corridor miss into q=0 hard braking. Each trial is
     * replayed through the nonlinear source plant over the full horizon and
     * is published only after the same hard corridor validator succeeds. */
    if (repairable_corridor_failure && corridor_repair_available &&
        configuration->refinement_mode == MPC_RTI_REFINEMENT_R1) {
        *memory = memory_before_r1;
        selected = corridor_repair;
        selected_candidate = 3;
        result->corridor_repair_used = 1;
        result->corridor_repair_alpha = corridor_repair_alpha;
        run_r2 = 0;
    }
    if (run_r2 && cycle_deadline_us > 0.0 &&
        cycle_deadline_us - monotonic_microseconds() <
            (double)MPC_RTI2_MIN_REMAINING_US) {
        /* R1 has already passed exact nonlinear validation when accepted.
         * Skip optional refinement rather than overrun; rejected R1 still
         * falls through the existing bounded-recovery/fallback path. */
        run_r2 = 0;
        result->rti2_budget_skipped = 1;
    }
    if (!run_r2 && result->rti2_budget_skipped &&
        repairable_residual_failure && r1.candidate.valid) {
        /* R1's residual-limited candidate has already passed the configured
         * residual-recovery bound and exact nonlinear/corridor validation.
         * R2 is optional refinement; if the 40 Hz deadline prevents it from
         * running, retain that verified R1 candidate as degraded authority
         * instead of converting a compute-budget miss into Unity's full-brake
         * command. This uses the same gate as the existing R2-failure path. */
        *memory = r1_memory;
        selected = r1;
        selected.status = MPC_RTI_CYCLE_ACCEPTED_DEGRADED;
        selected_candidate = 1;
        result->residual_candidate_published = 1;
    }
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
                if (corridor_repair_available) {
                    *memory = memory_before_r1;
                    selected = corridor_repair;
                    selected_candidate = 3;
                    result->corridor_repair_used = 1;
                    result->corridor_repair_alpha = corridor_repair_alpha;
                    result->rti2_triggered = 1;
                } else {
                    result->total_rti_us =
                        monotonic_microseconds() - cycle_start_us;
                    return reject_cycle(memory, result, r1_status);
                }
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
                horizon, configuration, &corridor_schedule, memory,
                cycle_deadline_us, &r2);
            result->r2_status = (int)r2_status;
            result->r2_nonlinear_failure_reason =
                r2.nonlinear_failure_reason;
            result->r2_nonlinear_failure_stage = r2.nonlinear_failure_stage;
            result->r2_nonlinear_failure = r2.nonlinear_failure;
            result->rti_iterations_used = 2;
            result->r2_first_action = r2.first_action;
            result->r2_solver_iterations = r2.iterations;
            result->r2_solve_us = r2.solve_us;
            result->r2_qp_build_us = r2.qp_build_us;
            result->r2_admm_us = r2.admm_us;
            result->r2_exact_rollout_us = r2.exact_rollout_us;
            result->r2_nonlinear_validation_us =
                r2.nonlinear_validation_us;
            result->r2_nonlinear_objective = r2.nonlinear_objective;
            result->r2_min_corridor_slack = r2.minimum_corridor_slack;
            result->rho_final = r2.rho_final;
            result->rho_u_final = r2.rho_u_final;
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
            /* A residual-recovery R1 candidate may have passed the exact
             * nonlinear rollout, but it has not passed the convergence gate.
             * Do not promote it to a command when R2 also fails: otherwise a
             * residual failure is hidden as an accepted degraded cycle. */
            if (corridor_repair_available) {
                *memory = memory_before_r1;
                selected = corridor_repair;
                selected_candidate = 3;
                result->corridor_repair_used = 1;
                result->corridor_repair_alpha = corridor_repair_alpha;
            } else if (result->r2_status == MPC_RTI_CYCLE_REJECTED_RESIDUAL &&
                       r2.candidate.valid) {
                /* R2 may fail only the numerical residual gate while its
                 * bounded candidate has already passed the exact nonlinear
                 * rollout and corridor checks. Keep that best found control
                 * rather than withdrawing authority or braking. */
                selected = r2;
                selected.status = MPC_RTI_CYCLE_ACCEPTED_DEGRADED;
                selected_candidate = 2;
                result->residual_candidate_published = 1;
            } else if (r1_status == MPC_RTI_CYCLE_REJECTED_RESIDUAL &&
                       r1.candidate.valid) {
                /* If R2 did not produce a usable candidate, retain R1's
                 * exact-feasible residual-limited iterate. */
                *memory = r1_memory;
                selected = r1;
                selected.status = MPC_RTI_CYCLE_ACCEPTED_DEGRADED;
                selected_candidate = 1;
                result->residual_candidate_published = 1;
            } else {
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
    result->throttle_mode_switches = selected.throttle_mode_switches;
    result->primal_residual = selected.primal_residual;
    result->dual_residual = selected.dual_residual;
    result->maximum_regularization = selected.maximum_regularization;
    result->regularization_count = selected.regularization_count;
    result->nonsmooth_jacobian_columns = selected.nonsmooth_columns;
    result->nonlinear_failure_stage = selected.nonlinear_failure_stage;
    result->nonlinear_failure_reason = selected.nonlinear_failure_reason;
    result->first_control = selected.first_action;
    result->published_throttle = selected.first_action.throttle_command;
    if (selected.candidate.valid) {
        result->published_steering_command =
            selected.candidate.states[1].plant.delayed_steering_command_2;
    }
    result->status = selected.status;

    if (!selected.candidate.valid &&
        selected.status != MPC_RTI_CYCLE_REJECTED_RESIDUAL) {
        /* Match the original rejection boundary: a nonlinear-invalid
         * horizon must not poison either the nominal trajectory or the ADMM
         * warm start. Preserve the previously accepted nominal while still
         * publishing its finite first action. */
        const MpcRtiNominal_t previous_nominal = memory_before_r1.nominal;
        mpc_rti_memory_reset(memory);
        memory->nominal = previous_nominal;
    } else if (selected.candidate.valid) {
        memory->nominal = selected.candidate;
    }
    memory->consecutive_degraded_solves =
        selected.status == MPC_RTI_CYCLE_ACCEPTED_DEGRADED
            ? (selected_candidate == 3
                ? memory_before_r1.consecutive_degraded_solves
                : r1_memory.consecutive_degraded_solves) + 1 : 0;
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
