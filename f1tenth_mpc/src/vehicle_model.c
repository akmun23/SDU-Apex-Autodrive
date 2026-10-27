/**
 * @file vehicle_model.c
 * @brief MPC transition backed by the checked-in 1 ms four-wheel plant.
 *
 * Steering response, wheel spin, tire slip, suspension, and body motion all
 * come from the same recursive source-derived plant. No fitted yaw response
 * is layered on top of those dynamics.
 */

#include "vehicle_model.h"
#include "mpc_linearization.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

/* Unity's Rigidbody velocity is measured at its COM. The MPC state and
 * /odom twist are referenced to base_link at the rear axle. The source
 * vehicle geometry puts COM 0.15532 m forward of that axle (see
 * docs/development/CURRENT_RUNTIME_PROGRESS_20261017.md). */
static const double kComForwardFromRearAxleM = 0.15532;

static VehicleParameters_t active_parameters = {
    .steering_wheelbase_m = SOURCE_STEERING_WHEELBASE_M,
    .max_steering_angle = SOURCE_MAX_STEERING_RAD,
    .steering_rate_radps = SOURCE_STEERING_RATE_RADPS,
    .maximum_command_speed_mps = MPC_MAX_COMMAND_SPEED_MPS,
};
static float active_target_speed_ceiling_mps = MPC_MAX_COMMAND_SPEED_MPS;

static float clampf_local(float value, float lower, float upper)
{
    if (value < lower)
        return lower;
    if (value > upper)
        return upper;
    return value;
}

static double clamp_source_steering_target(float target_rad)
{
    const double target = (double)target_rad;
    const double limit = UNITY_SOURCE_MAX_STEERING_RAD;
    if (target > limit) return limit;
    if (target < -limit) return -limit;
    return target;
}

static int finite_positive(float value)
{
    return isfinite(value) && value > 0.0f;
}

float vehicle_model_steering_for_curvature(float curvature_radpm)
{
    if (!isfinite(curvature_radpm)) return NAN;
    const float steering = atanf(
        active_parameters.steering_wheelbase_m * curvature_radpm);
    return clampf_local(steering, -SOURCE_MAX_STEERING_RAD,
        SOURCE_MAX_STEERING_RAD);
}

VehicleParameters_t vehicle_model_default_parameters(void)
{
    return (VehicleParameters_t){
        .steering_wheelbase_m = SOURCE_STEERING_WHEELBASE_M,
        .max_steering_angle = SOURCE_MAX_STEERING_RAD,
        .steering_rate_radps = SOURCE_STEERING_RATE_RADPS,
        .maximum_command_speed_mps = MPC_MAX_COMMAND_SPEED_MPS,
    };
}

VehicleParameters_t vehicle_model_get_parameters(void)
{
    return active_parameters;
}

int vehicle_model_set_parameters(const VehicleParameters_t *parameters)
{
    if (parameters == NULL ||
        !finite_positive(parameters->steering_wheelbase_m) ||
        !finite_positive(parameters->max_steering_angle) ||
        !finite_positive(parameters->steering_rate_radps) ||
        !finite_positive(parameters->maximum_command_speed_mps)) {
        return 0;
    }
    active_parameters = *parameters;
    if (active_target_speed_ceiling_mps > active_parameters.maximum_command_speed_mps)
        active_target_speed_ceiling_mps = active_parameters.maximum_command_speed_mps;
    return 1;
}

float vehicle_model_get_active_target_speed_ceiling(void)
{
    return active_target_speed_ceiling_mps;
}

int vehicle_model_set_active_target_speed_ceiling(float ceiling_mps)
{
    if (!isfinite(ceiling_mps) || ceiling_mps <= 0.0f ||
        ceiling_mps > active_parameters.maximum_command_speed_mps) {
        return 0;
    }
    active_target_speed_ceiling_mps = ceiling_mps;
    return 1;
}

static void seed_source_physics(
    const MpcModelState_t *state, UnitySourcePlantInputC *input)
{
    memset(input, 0, sizeof(*input));
    input->state[UNITY_SOURCE_U] = state->u;
    input->state[UNITY_SOURCE_V] =
        state->v + state->r * kComForwardFromRearAxleM;
    input->state[UNITY_SOURCE_YAW_RATE] = state->r;
    input->state[UNITY_SOURCE_BODY_HEIGHT] = 0.00464804098;
    input->state[UNITY_SOURCE_PHYSICAL_STEERING] = state->actual_steering_angle;

    const double wheel_x[4] = {0.17468, 0.17468, -0.15532, -0.15532};
    const double wheel_y[4] = {0.11582, -0.11582, 0.11792, -0.11808};
    const double wheel_radius = 0.059;
    for (int wheel = 0; wheel < 4; ++wheel) {
        const double vx = state->u - state->r * wheel_y[wheel];
        const double vy = state->v + state->r * wheel_x[wheel];
        const double angle = wheel < 2 ? state->actual_steering_angle : 0.0;
        input->state[UNITY_SOURCE_WHEEL_OMEGA_FL + wheel] =
            (vx * cos(angle) + vy * sin(angle)) / wheel_radius;
    }
    if (state->source_imu_state_initialized) {
        input->state[UNITY_SOURCE_ROLL] =
            state->source_plant_internal_state[MPC_PLANT_STATE_ROLL];
        input->state[UNITY_SOURCE_ROLL_RATE] =
            state->source_plant_internal_state[MPC_PLANT_STATE_ROLL_RATE];
        input->state[UNITY_SOURCE_PITCH] =
            state->source_plant_internal_state[MPC_PLANT_STATE_PITCH];
        input->state[UNITY_SOURCE_PITCH_RATE] =
            state->source_plant_internal_state[MPC_PLANT_STATE_PITCH_RATE];
    }
    input->suspension_history_initialized = 0;
}

static void source_state_to_internal(
    MpcModelState_t *state,
    const double source_state[UNITY_SOURCE_PLANT_STATE_SIZE])
{
    memcpy(&state->source_plant_internal_state[
               MPC_PLANT_STATE_WHEEL_OMEGA_FL],
           &source_state[UNITY_SOURCE_WHEEL_OMEGA_FL],
           4 * sizeof(source_state[0]));
    state->source_plant_internal_state[MPC_PLANT_STATE_BODY_HEIGHT] =
        source_state[UNITY_SOURCE_BODY_HEIGHT];
    state->source_plant_internal_state[MPC_PLANT_STATE_BODY_HEIGHT_RATE] =
        source_state[UNITY_SOURCE_BODY_HEIGHT_RATE];
    state->source_plant_internal_state[MPC_PLANT_STATE_ROLL] =
        source_state[UNITY_SOURCE_ROLL];
    state->source_plant_internal_state[MPC_PLANT_STATE_ROLL_RATE] =
        source_state[UNITY_SOURCE_ROLL_RATE];
    state->source_plant_internal_state[MPC_PLANT_STATE_PITCH] =
        source_state[UNITY_SOURCE_PITCH];
    state->source_plant_internal_state[MPC_PLANT_STATE_PITCH_RATE] =
        source_state[UNITY_SOURCE_PITCH_RATE];
    memcpy(&state->source_plant_internal_state[
               MPC_PLANT_STATE_SUSPENSION_JOUNCE_FL],
           &source_state[UNITY_SOURCE_SUSPENSION_JOUNCE_FL],
           4 * sizeof(source_state[0]));
    memcpy(&state->source_plant_internal_state[
               MPC_PLANT_STATE_LOW_FORWARD_SPEED_TIME_FL],
           &source_state[UNITY_SOURCE_LOW_FORWARD_SPEED_TIME_FL],
           4 * sizeof(source_state[0]));
    memcpy(&state->source_plant_internal_state[
               MPC_PLANT_STATE_LOW_SIDE_SPEED_TIME_FL],
           &source_state[UNITY_SOURCE_LOW_SIDE_SPEED_TIME_FL],
           4 * sizeof(source_state[0]));
}

static void write_source_plant_state(
    const MpcModelState_t *state,
    double source_state[UNITY_SOURCE_PLANT_STATE_SIZE])
{
    source_state[UNITY_SOURCE_U] = state->u;
    source_state[UNITY_SOURCE_V] =
        state->v + state->r * kComForwardFromRearAxleM;
    source_state[UNITY_SOURCE_YAW_RATE] = state->r;
    memcpy(&source_state[UNITY_SOURCE_WHEEL_OMEGA_FL],
           &state->source_plant_internal_state[
               MPC_PLANT_STATE_WHEEL_OMEGA_FL],
           4 * sizeof(source_state[0]));
    source_state[UNITY_SOURCE_BODY_HEIGHT] =
        state->source_plant_internal_state[MPC_PLANT_STATE_BODY_HEIGHT];
    source_state[UNITY_SOURCE_BODY_HEIGHT_RATE] =
        state->source_plant_internal_state[MPC_PLANT_STATE_BODY_HEIGHT_RATE];
    source_state[UNITY_SOURCE_ROLL] =
        state->source_plant_internal_state[MPC_PLANT_STATE_ROLL];
    source_state[UNITY_SOURCE_ROLL_RATE] =
        state->source_plant_internal_state[MPC_PLANT_STATE_ROLL_RATE];
    source_state[UNITY_SOURCE_PITCH] =
        state->source_plant_internal_state[MPC_PLANT_STATE_PITCH];
    source_state[UNITY_SOURCE_PITCH_RATE] =
        state->source_plant_internal_state[MPC_PLANT_STATE_PITCH_RATE];
    source_state[UNITY_SOURCE_PHYSICAL_STEERING] =
        state->actual_steering_angle;
    memcpy(&source_state[UNITY_SOURCE_SUSPENSION_JOUNCE_FL],
           &state->source_plant_internal_state[
               MPC_PLANT_STATE_SUSPENSION_JOUNCE_FL],
           4 * sizeof(source_state[0]));
    memcpy(&source_state[UNITY_SOURCE_LOW_FORWARD_SPEED_TIME_FL],
           &state->source_plant_internal_state[
               MPC_PLANT_STATE_LOW_FORWARD_SPEED_TIME_FL],
           4 * sizeof(source_state[0]));
    memcpy(&source_state[UNITY_SOURCE_LOW_SIDE_SPEED_TIME_FL],
           &state->source_plant_internal_state[
               MPC_PLANT_STATE_LOW_SIDE_SPEED_TIME_FL],
           4 * sizeof(source_state[0]));
}

int vehicle_model_longitudinal_slip_excess(
    const MpcModelState_t *state,
    double excess_speed_mps[4],
    double excess_derivative_wheel_omega[4])
{
    if (!state || !excess_speed_mps || !excess_derivative_wheel_omega)
        return 0;
    MpcModelState_t prepared = *state;
    if (!vehicle_model_initialize_plant_state(&prepared)) return 0;
    double source_state[UNITY_SOURCE_PLANT_STATE_SIZE] = {0};
    double slip[4] = {0};
    write_source_plant_state(&prepared, source_state);
    return unity_source_plant_slip_excess_c(
        source_state, slip, excess_speed_mps,
        excess_derivative_wheel_omega);
}

int vehicle_model_initialize_plant_state(MpcModelState_t *state)
{
    if (!state || !isfinite(state->u) || !isfinite(state->v) ||
        !isfinite(state->r) || !isfinite(state->actual_steering_angle)) {
        return 0;
    }

    if (!state->source_plant_initialized) {
        UnitySourcePlantInputC initial;
        seed_source_physics(state, &initial);
        source_state_to_internal(state, initial.state);
        state->source_suspension_history_initialized = 0;
        state->source_plant_initialized = 1;
    }
    return 1;
}

static int finite_mpc_state(const MpcModelState_t *state)
{
    if (!state || !isfinite(state->e_y) || !isfinite(state->e_psi) ||
        !isfinite(state->u) || !isfinite(state->v) || !isfinite(state->r) ||
        !isfinite(state->actual_steering_angle) ||
        !isfinite(state->delayed_throttle_command_1) ||
        !isfinite(state->delayed_throttle_command_2)) return 0;
    if (state->delayed_throttle_command_1 < 0.0f ||
        state->delayed_throttle_command_1 > 1.0f ||
        state->delayed_throttle_command_2 < 0.0f ||
        state->delayed_throttle_command_2 > 1.0f) return 0;
    if (state->source_plant_initialized || state->source_imu_state_initialized) {
        for (int index = 0; index < MPC_PLANT_INTERNAL_STATE_COUNT; ++index)
            if (!isfinite(state->source_plant_internal_state[index])) return 0;
    }
    return 1;
}

static unsigned int mask_popcount(uint64_t mask)
{
    unsigned int count = 0;
    while (mask != 0u) {
        count += mask & 1u;
        mask >>= 1u;
    }
    return count;
}

/* The plant has already integrated body displacement and Euler yaw at its
 * internal 1 ms rate. Project that actual interval motion into Frenet
 * coordinates rather than reconstructing it from 25 ms endpoint twist. */
static int vehicle_model_tracking_step_integrated(
    const MpcModelState_t *state,
    float body_forward_delta_m,
    float body_lateral_delta_m,
    float body_yaw_delta_rad,
    float path_curvature,
    float *e_y_next,
    float *e_psi_next,
    float *delta_s)
{
    if (!state || !e_y_next || !e_psi_next || !delta_s) return 0;
    const float c = cosf(state->e_psi);
    const float s = sinf(state->e_psi);
    const float path_forward_delta =
        body_forward_delta_m * c - body_lateral_delta_m * s;
    const float path_lateral_delta =
        body_forward_delta_m * s + body_lateral_delta_m * c;
    const float denominator = 1.0f - path_curvature * state->e_y;
    if (fabsf(denominator) < 0.05f) return 0;
    /* Project the integrated displacement onto the local reference circle.
     * The former Euler progress / second-order lateral correction accumulated
     * coordinate error even for a perfect constant-curvature motion. This
     * is a coordinate transform; the Unity plant displacement is unchanged.
     * Rationalize (1-radius)/k to avoid cancellation near a straight. */
    const float y = state->e_y + path_lateral_delta;
    const float normal = 1.0f - path_curvature * y;
    const float tangent = path_curvature * path_forward_delta;
    const float radius = hypotf(normal, tangent);
    if (!(radius > 0.05f)) return 0;
    const float path_angle = atan2f(tangent, normal);
    *delta_s = fabsf(path_curvature) > 1.0e-6f
        ? path_angle / path_curvature : path_forward_delta / normal;
    *e_y_next = (2.0f * y - path_curvature *
        (path_forward_delta * path_forward_delta + y * y)) /
        (1.0f + radius);
    const float e_psi_raw = state->e_psi + body_yaw_delta_rad -
        path_angle;
    *e_psi_next = fabsf(e_psi_raw) > 3.14159265358979323846f
        ? remainderf(e_psi_raw, 6.28318530717958647692f) : e_psi_raw;
    return isfinite(*e_y_next) && isfinite(*e_psi_next) &&
        isfinite(*delta_s);
}

/* MPC transitions use the stage's direct command at the command epoch;
 * control-time propagation explicitly selects the last accepted queue entry.
 * Both paths use the same 1 ms source-derived four-wheel plant. The compact
 * RTI state tracks path/body/controller variables, while this carried state
 * preserves wheel spin, sprung-body motion, and suspension history across
 * every stage of the horizon. */
static int vehicle_model_step_scalar(
    const MpcModelState_t *state,
    const MpcModelControl_t *control,
    float dt,
    float path_curvature,
    int use_queued_command,
    int advance_command_queues,
    MpcStageResult_t *stage)
{
    if (stage == NULL) return 0;
    *stage = (MpcStageResult_t){0};
    if (!finite_mpc_state(state) || control == NULL ||
        !isfinite(control->steering_target_rad) ||
        !isfinite(control->throttle_command) ||
        !(dt > 0.0f) || !isfinite(dt) || !isfinite(path_curvature)) {
        return 0;
    }

    const VehicleParameters_t parameters = vehicle_model_get_parameters();
    const float steering_target = clampf_local(
        control->steering_target_rad,
        -parameters.max_steering_angle, parameters.max_steering_angle);
    const float q_throttle = clampf_local(control->throttle_command,
        MPC_THROTTLE_MIN, MPC_THROTTLE_MAX);
    if (steering_target != control->steering_target_rad)
        stage->branch_flags |= MPC_STAGE_CLIPPED_STEERING_TARGET;
    if (q_throttle != control->throttle_command)
        stage->branch_flags |= MPC_STAGE_CLIPPED_THROTTLE;

    const float delta_next = steering_target;

    MpcModelState_t next = *state;
    UnitySourcePlantInputC source_input;
    if (state->source_plant_initialized) {
        memset(&source_input, 0, sizeof(source_input));
        write_source_plant_state(state, source_input.state);
        source_input.suspension_history_initialized =
            state->source_suspension_history_initialized;
    } else {
        seed_source_physics(state, &source_input);
    }
    /* The legal float command for 30 deg rounds 1.4e-8 rad above the exact
     * source double limit. Clip only that representation error at the plant
     * boundary; this preserves the simulator's physical steering limit. */
    source_input.steering_target_rad = clamp_source_steering_target(
        use_queued_command ? state->delayed_steering_command_2
                           : steering_target);
    source_input.throttle_normalized = use_queued_command
        ? state->delayed_throttle_command_2 : q_throttle;
    if (source_input.throttle_normalized == 0.0f)
        stage->branch_flags |= MPC_STAGE_PLANT_BRAKE;
    UnitySourcePlantOutputC source_output;
    if (!unity_source_plant_step_c(
            &source_input, dt, &source_output)) return 0;
    memcpy(stage->output.longitudinal_slip, source_output.longitudinal_slip,
           sizeof(stage->output.longitudinal_slip));
    memcpy(stage->output.maximum_positive_longitudinal_slip,
           source_output.maximum_positive_longitudinal_slip,
           sizeof(stage->output.maximum_positive_longitudinal_slip));
    memcpy(stage->output.maximum_longitudinal_slip_excess_mps,
           source_output.maximum_longitudinal_slip_excess_mps,
           sizeof(stage->output.maximum_longitudinal_slip_excess_mps));
    memcpy(stage->output.lateral_slip, source_output.lateral_slip,
           sizeof(stage->output.lateral_slip));
    memcpy(stage->output.normal_load_n, source_output.normal_load_n,
           sizeof(stage->output.normal_load_n));
    source_state_to_internal(&next, source_output.state);
    next.source_plant_initialized = 1;
    next.source_suspension_history_initialized =
        source_output.suspension_history_initialized;

    const float u_next = (float)source_output.state[UNITY_SOURCE_U];
    const float r_next =
        (float)source_output.state[UNITY_SOURCE_YAW_RATE];
    const float body_yaw_delta = (float)
        source_output.integrated_yaw_delta_rad;
    const float body_forward_delta = (float)
        source_output.integrated_body_forward_displacement_m;
    const float body_lateral_delta = (float)
        source_output.integrated_body_lateral_displacement_m;
    const float v_next = (float)(
        source_output.state[UNITY_SOURCE_V] -
        r_next * kComForwardFromRearAxleM);
    const float actual_next = (float)
        source_output.state[UNITY_SOURCE_PHYSICAL_STEERING];
    if (!isfinite(u_next) || !isfinite(v_next) || !isfinite(r_next) ||
        !isfinite(actual_next) ||
        fabsf(u_next) > parameters.maximum_command_speed_mps) return 0;

    if (!vehicle_model_tracking_step_integrated(
            state, body_forward_delta, body_lateral_delta,
            body_yaw_delta, path_curvature,
            &next.e_y, &next.e_psi, &stage->delta_s_m)) return 0;
    stage->body_forward_delta_m = body_forward_delta;
    stage->body_lateral_delta_m = body_lateral_delta;
    stage->body_yaw_delta_rad = body_yaw_delta;
    next.u = u_next;
    next.v = v_next;
    next.r = r_next;
    if (advance_command_queues) {
        /* Keep the last command as history, then append the new decision as
         * the next in-flight command (slot 2). */
        next.delayed_steering_command_1 = state->delayed_steering_command_2;
        next.delayed_steering_command_2 = delta_next;
        next.delayed_throttle_command_1 = state->delayed_throttle_command_2;
        next.delayed_throttle_command_2 = q_throttle;
    }
    next.actual_steering_angle = actual_next;
    stage->next = next;
    stage->valid = finite_mpc_state(&stage->next) &&
        isfinite(stage->delta_s_m);
    return stage->valid;
}

static float mpc_state_component(const MpcModelState_t *state, int index)
{
    switch (index) {
    case MPC_STATE_IDX_EY: return state->e_y;
    case MPC_STATE_IDX_EPSI: return state->e_psi;
    case MPC_STATE_IDX_U: return state->u;
    case MPC_STATE_IDX_V: return state->v;
    case MPC_STATE_IDX_R: return state->r;
    case MPC_STATE_IDX_DELAYED_STEERING_COMMAND_1:
        return state->delayed_steering_command_1;
    case MPC_STATE_IDX_DELAYED_STEERING_COMMAND_2:
        return state->delayed_steering_command_2;
    case MPC_STATE_IDX_ACTUAL_STEERING_ANGLE:
        return state->actual_steering_angle;
    case MPC_STATE_IDX_DELAYED_THROTTLE_COMMAND_1:
        return state->delayed_throttle_command_1;
    case MPC_STATE_IDX_DELAYED_THROTTLE_COMMAND_2:
        return state->delayed_throttle_command_2;
    case MPC_STATE_IDX_WHEEL_OMEGA_FL:
        return (float)state->source_plant_internal_state[
            MPC_PLANT_STATE_WHEEL_OMEGA_FL];
    case MPC_STATE_IDX_WHEEL_OMEGA_FR:
        return (float)state->source_plant_internal_state[
            MPC_PLANT_STATE_WHEEL_OMEGA_FR];
    case MPC_STATE_IDX_WHEEL_OMEGA_RL:
        return (float)state->source_plant_internal_state[
            MPC_PLANT_STATE_WHEEL_OMEGA_RL];
    case MPC_STATE_IDX_WHEEL_OMEGA_RR:
        return (float)state->source_plant_internal_state[
            MPC_PLANT_STATE_WHEEL_OMEGA_RR];
    case MPC_STATE_IDX_BODY_HEIGHT:
        return (float)state->source_plant_internal_state[
            MPC_PLANT_STATE_BODY_HEIGHT];
    case MPC_STATE_IDX_BODY_HEIGHT_RATE:
        return (float)state->source_plant_internal_state[
            MPC_PLANT_STATE_BODY_HEIGHT_RATE];
    case MPC_STATE_IDX_ROLL:
        return (float)state->source_plant_internal_state[MPC_PLANT_STATE_ROLL];
    case MPC_STATE_IDX_ROLL_RATE:
        return (float)state->source_plant_internal_state[
            MPC_PLANT_STATE_ROLL_RATE];
    case MPC_STATE_IDX_PITCH:
        return (float)state->source_plant_internal_state[MPC_PLANT_STATE_PITCH];
    case MPC_STATE_IDX_PITCH_RATE:
        return (float)state->source_plant_internal_state[
            MPC_PLANT_STATE_PITCH_RATE];
    case MPC_STATE_IDX_SUSPENSION_JOUNCE_FL:
        return (float)state->source_plant_internal_state[
            MPC_PLANT_STATE_SUSPENSION_JOUNCE_FL];
    case MPC_STATE_IDX_SUSPENSION_JOUNCE_FR:
        return (float)state->source_plant_internal_state[
            MPC_PLANT_STATE_SUSPENSION_JOUNCE_FR];
    case MPC_STATE_IDX_SUSPENSION_JOUNCE_RL:
        return (float)state->source_plant_internal_state[
            MPC_PLANT_STATE_SUSPENSION_JOUNCE_RL];
    case MPC_STATE_IDX_SUSPENSION_JOUNCE_RR:
        return (float)state->source_plant_internal_state[
            MPC_PLANT_STATE_SUSPENSION_JOUNCE_RR];
    default: return NAN;
    }
}

static void mpc_state_perturb(MpcModelState_t *state, int index, float amount)
{
    switch (index) {
    case MPC_STATE_IDX_EY: state->e_y += amount; break;
    case MPC_STATE_IDX_EPSI: state->e_psi += amount; break;
    case MPC_STATE_IDX_U: state->u += amount; break;
    case MPC_STATE_IDX_V: state->v += amount; break;
    case MPC_STATE_IDX_R: state->r += amount; break;
    case MPC_STATE_IDX_DELAYED_STEERING_COMMAND_1:
        state->delayed_steering_command_1 += amount; break;
    case MPC_STATE_IDX_DELAYED_STEERING_COMMAND_2:
        state->delayed_steering_command_2 += amount; break;
    case MPC_STATE_IDX_ACTUAL_STEERING_ANGLE:
        state->actual_steering_angle += amount; break;
    case MPC_STATE_IDX_DELAYED_THROTTLE_COMMAND_1:
        state->delayed_throttle_command_1 += amount; break;
    case MPC_STATE_IDX_DELAYED_THROTTLE_COMMAND_2:
        state->delayed_throttle_command_2 += amount; break;
    case MPC_STATE_IDX_WHEEL_OMEGA_FL:
        state->source_plant_internal_state[MPC_PLANT_STATE_WHEEL_OMEGA_FL] += amount; break;
    case MPC_STATE_IDX_WHEEL_OMEGA_FR:
        state->source_plant_internal_state[MPC_PLANT_STATE_WHEEL_OMEGA_FR] += amount; break;
    case MPC_STATE_IDX_WHEEL_OMEGA_RL:
        state->source_plant_internal_state[MPC_PLANT_STATE_WHEEL_OMEGA_RL] += amount; break;
    case MPC_STATE_IDX_WHEEL_OMEGA_RR:
        state->source_plant_internal_state[MPC_PLANT_STATE_WHEEL_OMEGA_RR] += amount; break;
    case MPC_STATE_IDX_BODY_HEIGHT:
        state->source_plant_internal_state[MPC_PLANT_STATE_BODY_HEIGHT] += amount; break;
    case MPC_STATE_IDX_BODY_HEIGHT_RATE:
        state->source_plant_internal_state[MPC_PLANT_STATE_BODY_HEIGHT_RATE] += amount; break;
    case MPC_STATE_IDX_ROLL:
        state->source_plant_internal_state[MPC_PLANT_STATE_ROLL] += amount; break;
    case MPC_STATE_IDX_ROLL_RATE:
        state->source_plant_internal_state[MPC_PLANT_STATE_ROLL_RATE] += amount; break;
    case MPC_STATE_IDX_PITCH:
        state->source_plant_internal_state[MPC_PLANT_STATE_PITCH] += amount; break;
    case MPC_STATE_IDX_PITCH_RATE:
        state->source_plant_internal_state[MPC_PLANT_STATE_PITCH_RATE] += amount; break;
    case MPC_STATE_IDX_SUSPENSION_JOUNCE_FL:
        state->source_plant_internal_state[MPC_PLANT_STATE_SUSPENSION_JOUNCE_FL] += amount; break;
    case MPC_STATE_IDX_SUSPENSION_JOUNCE_FR:
        state->source_plant_internal_state[MPC_PLANT_STATE_SUSPENSION_JOUNCE_FR] += amount; break;
    case MPC_STATE_IDX_SUSPENSION_JOUNCE_RL:
        state->source_plant_internal_state[MPC_PLANT_STATE_SUSPENSION_JOUNCE_RL] += amount; break;
    case MPC_STATE_IDX_SUSPENSION_JOUNCE_RR:
        state->source_plant_internal_state[MPC_PLANT_STATE_SUSPENSION_JOUNCE_RR] += amount; break;
    default: break;
    }
}

static float mpc_stage_component(const MpcStageResult_t *stage, int index)
{
    return mpc_state_component(&stage->next, index);
}

static float mpc_component_delta(int row, float lhs, float rhs)
{
    return row == 1 ? remainderf(lhs - rhs, 6.2831853071795864769f)
                    : lhs - rhs;
}

enum { MPC_FORWARD_SENSITIVITY_COLUMNS = 20 };

static const int kPlantSensitiveModelColumns[
    MPC_FORWARD_SENSITIVITY_COLUMNS] = {
    MPC_STATE_IDX_U, MPC_STATE_IDX_V, MPC_STATE_IDX_R,
    MPC_STATE_IDX_ACTUAL_STEERING_ANGLE,
    MPC_MODEL_NX + 0, MPC_MODEL_NX + 1,
    MPC_STATE_IDX_WHEEL_OMEGA_FL, MPC_STATE_IDX_WHEEL_OMEGA_FR,
    MPC_STATE_IDX_WHEEL_OMEGA_RL, MPC_STATE_IDX_WHEEL_OMEGA_RR,
    MPC_STATE_IDX_BODY_HEIGHT, MPC_STATE_IDX_BODY_HEIGHT_RATE,
    MPC_STATE_IDX_ROLL, MPC_STATE_IDX_ROLL_RATE,
    MPC_STATE_IDX_PITCH, MPC_STATE_IDX_PITCH_RATE,
    MPC_STATE_IDX_SUSPENSION_JOUNCE_FL,
    MPC_STATE_IDX_SUSPENSION_JOUNCE_FR,
    MPC_STATE_IDX_SUSPENSION_JOUNCE_RL,
    MPC_STATE_IDX_SUSPENSION_JOUNCE_RR};

static int model_column_seed_source(int model_column, int *source_index,
                                   int *second_source_index,
                                   double *second_source_scale,
                                   int *command_kind)
{
    *source_index = -1;
    *second_source_index = -1;
    *second_source_scale = 0.0;
    *command_kind = 0;
    switch (model_column) {
    case MPC_STATE_IDX_U: *source_index = UNITY_SOURCE_U; return 1;
    case MPC_STATE_IDX_V: *source_index = UNITY_SOURCE_V; return 1;
    case MPC_STATE_IDX_R:
        *source_index = UNITY_SOURCE_YAW_RATE;
        *second_source_index = UNITY_SOURCE_V;
        *second_source_scale = kComForwardFromRearAxleM;
        return 1;
    case MPC_MODEL_NX + 0:
        *command_kind = 1;
        return 1;
    case MPC_STATE_IDX_ACTUAL_STEERING_ANGLE:
        *source_index = UNITY_SOURCE_PHYSICAL_STEERING;
        return 1;
    case MPC_MODEL_NX + 1:
        *command_kind = 2;
        return 1;
    default: break;
    }
    if (model_column >= MPC_STATE_IDX_WHEEL_OMEGA_FL &&
        model_column <= MPC_STATE_IDX_WHEEL_OMEGA_RR) {
        *source_index = UNITY_SOURCE_WHEEL_OMEGA_FL +
            model_column - MPC_STATE_IDX_WHEEL_OMEGA_FL;
        return 1;
    }
    switch (model_column) {
    case MPC_STATE_IDX_BODY_HEIGHT: *source_index = UNITY_SOURCE_BODY_HEIGHT; return 1;
    case MPC_STATE_IDX_BODY_HEIGHT_RATE:
        *source_index = UNITY_SOURCE_BODY_HEIGHT_RATE; return 1;
    case MPC_STATE_IDX_ROLL: *source_index = UNITY_SOURCE_ROLL; return 1;
    case MPC_STATE_IDX_ROLL_RATE:
        *source_index = UNITY_SOURCE_ROLL_RATE; return 1;
    case MPC_STATE_IDX_PITCH: *source_index = UNITY_SOURCE_PITCH; return 1;
    case MPC_STATE_IDX_PITCH_RATE:
        *source_index = UNITY_SOURCE_PITCH_RATE; return 1;
    default: break;
    }
    if (model_column >= MPC_STATE_IDX_SUSPENSION_JOUNCE_FL &&
        model_column <= MPC_STATE_IDX_SUSPENSION_JOUNCE_RR) {
        *source_index = UNITY_SOURCE_SUSPENSION_JOUNCE_FL +
            model_column - MPC_STATE_IDX_SUSPENSION_JOUNCE_FL;
        return 1;
    }
    return 0;
}

static float source_output_derivative_to_model(
    const UnitySourcePlantSensitivityC *sensitivity,
    int model_row, int lane)
{
    switch (model_row) {
    case MPC_STATE_IDX_U:
        return (float)sensitivity->state[UNITY_SOURCE_U][lane];
    case MPC_STATE_IDX_V:
        return (float)(sensitivity->state[UNITY_SOURCE_V][lane] -
            kComForwardFromRearAxleM *
                sensitivity->state[UNITY_SOURCE_YAW_RATE][lane]);
    case MPC_STATE_IDX_R:
        return (float)sensitivity->state[UNITY_SOURCE_YAW_RATE][lane];
    case MPC_STATE_IDX_ACTUAL_STEERING_ANGLE:
        return (float)sensitivity->state[
            UNITY_SOURCE_PHYSICAL_STEERING][lane];
    default: break;
    }
    if (model_row >= MPC_STATE_IDX_WHEEL_OMEGA_FL &&
        model_row <= MPC_STATE_IDX_WHEEL_OMEGA_RR) {
        const int source = UNITY_SOURCE_WHEEL_OMEGA_FL +
            model_row - MPC_STATE_IDX_WHEEL_OMEGA_FL;
        return (float)sensitivity->state[source][lane];
    }
    if (model_row >= MPC_STATE_IDX_BODY_HEIGHT &&
        model_row <= MPC_STATE_IDX_PITCH_RATE) {
        static const int source_state[] = {
            UNITY_SOURCE_BODY_HEIGHT, UNITY_SOURCE_BODY_HEIGHT_RATE,
            UNITY_SOURCE_ROLL, UNITY_SOURCE_ROLL_RATE,
            UNITY_SOURCE_PITCH, UNITY_SOURCE_PITCH_RATE};
        return (float)sensitivity->state[
            source_state[model_row - MPC_STATE_IDX_BODY_HEIGHT]][lane];
    }
    if (model_row >= MPC_STATE_IDX_SUSPENSION_JOUNCE_FL &&
        model_row <= MPC_STATE_IDX_SUSPENSION_JOUNCE_RR) {
        const int source = UNITY_SOURCE_SUSPENSION_JOUNCE_FL +
            model_row - MPC_STATE_IDX_SUSPENSION_JOUNCE_FL;
        return (float)sensitivity->state[source][lane];
    }
    return 0.0f;
}

static int vehicle_model_forward_plant_jacobian(
    const MpcModelState_t *state, const MpcModelControl_t *control,
    float dt, float path_curvature,
    const MpcStageResult_t *nominal_stage,
    MpcStageLinearization_t *linearization,
    uint32_t *covered_columns)
{
    if (!state || !control || !nominal_stage || !linearization ||
        !covered_columns)
        return 0;
    *covered_columns = 0;
    UnitySourcePlantInputC source_input;
    memset(&source_input, 0, sizeof(source_input));
    write_source_plant_state(state, source_input.state);
    source_input.steering_target_rad = clamp_source_steering_target(
        control->steering_target_rad);
    source_input.throttle_normalized = clampf_local(
        control->throttle_command, MPC_THROTTLE_MIN, MPC_THROTTLE_MAX);
    source_input.suspension_history_initialized =
        state->source_suspension_history_initialized;
    /* Shared geometry for every sensitivity lane and source-state batch. */
    const float tracking_c = cosf(state->e_psi);
    const float tracking_s = sinf(state->e_psi);
    const float tracking_x =
        nominal_stage->body_forward_delta_m * tracking_c -
        nominal_stage->body_lateral_delta_m * tracking_s;
    const float tracking_y = state->e_y +
        nominal_stage->body_forward_delta_m * tracking_s +
        nominal_stage->body_lateral_delta_m * tracking_c;
    const float tracking_normal = 1.0f - path_curvature * tracking_y;
    const float tracking_tangent = path_curvature * tracking_x;
    const float tracking_radius_squared =
        tracking_normal * tracking_normal +
        tracking_tangent * tracking_tangent;
    if (!(tracking_radius_squared > 0.0025f)) return 0;
    const float inverse_tracking_radius = 1.0f / sqrtf(tracking_radius_squared);
    const float inverse_tracking_radius_squared =
        1.0f / tracking_radius_squared;
    for (int first_column = 0;
         first_column < MPC_FORWARD_SENSITIVITY_COLUMNS;
         first_column += UNITY_SOURCE_PLANT_SENSITIVITY_WIDTH) {
        UnitySourcePlantSensitivitySeedsC seeds = {0};
        UnitySourcePlantSensitivityC sensitivity = {0};
        const int lane_count =
            MPC_FORWARD_SENSITIVITY_COLUMNS - first_column <
                    UNITY_SOURCE_PLANT_SENSITIVITY_WIDTH
                ? MPC_FORWARD_SENSITIVITY_COLUMNS - first_column
                : UNITY_SOURCE_PLANT_SENSITIVITY_WIDTH;
        for (int lane = 0; lane < lane_count; ++lane) {
            const int column_index = first_column + lane;
            const int model_column =
                kPlantSensitiveModelColumns[column_index];
            int source_index, second_source_index, command_kind;
            double second_source_scale;
            if (!model_column_seed_source(model_column, &source_index,
                    &second_source_index, &second_source_scale,
                    &command_kind)) return 0;
            if (source_index >= 0)
                seeds.state[source_index][lane] = 1.0;
            if (second_source_index >= 0)
                seeds.state[second_source_index][lane] =
                    second_source_scale;
            if (command_kind == 1) {
                const double target = control->steering_target_rad;
                seeds.steering_target[lane] =
                    target > -UNITY_SOURCE_MAX_STEERING_RAD &&
                    target < UNITY_SOURCE_MAX_STEERING_RAD ? 1.0 : 0.0;
            }
            if (command_kind == 2 &&
                control->throttle_command > MPC_THROTTLE_MIN &&
                control->throttle_command < MPC_THROTTLE_MAX)
                seeds.throttle[lane] = 1.0;
        }
        if (!unity_source_plant_step_sensitivity_c(
                &source_input, dt, &seeds, &sensitivity)) return 0;
        ++linearization->source_sensitivity_call_count;
        for (int wheel = 0; wheel < 4; ++wheel)
            linearization->maximum_slip_excess_mps[wheel] = (float)
                sensitivity.maximum_longitudinal_slip_excess_mps[wheel];
        for (int lane = 0; lane < lane_count; ++lane) {
            const int model_column =
                kPlantSensitiveModelColumns[first_column + lane];
            for (int wheel = 0; wheel < 4; ++wheel) {
                const float derivative = (float)
                    sensitivity.maximum_longitudinal_slip_excess_derivative[wheel][lane];
                if (!isfinite(derivative)) return 0;
                if (model_column < MPC_MODEL_NX)
                    linearization->maximum_slip_excess_A[wheel][model_column] = derivative;
                else
                    linearization->maximum_slip_excess_B[wheel][
                        model_column - MPC_MODEL_NX] = derivative;
            }
            for (int row = 2; row < MPC_MODEL_NX; ++row) {
                /* Command-history rows are exact shift/append equations,
                 * not Unity plant outputs. Preserve their analytic Jacobian. */
                if (row == MPC_STATE_IDX_DELAYED_STEERING_COMMAND_1 ||
                    row == MPC_STATE_IDX_DELAYED_STEERING_COMMAND_2 ||
                    row == MPC_STATE_IDX_DELAYED_THROTTLE_COMMAND_1 ||
                    row == MPC_STATE_IDX_DELAYED_THROTTLE_COMMAND_2) {
                    continue;
                }
                const float derivative =
                    source_output_derivative_to_model(
                        &sensitivity, row, lane);
                if (!isfinite(derivative)) return 0;
                if (model_column < MPC_MODEL_NX)
                    linearization->A[row][model_column] = derivative;
                else
                    linearization->B[row][model_column - MPC_MODEL_NX] =
                        derivative;
            }

            const float d_forward = (float)
                sensitivity.body_forward_displacement[lane];
            const float d_lateral = (float)
                sensitivity.body_lateral_displacement[lane];
            const float d_yaw = (float)sensitivity.yaw_delta[lane];
            const float motion_derivatives[3] = {d_forward, d_lateral, d_yaw};
            for (int motion = 0; motion < 3; ++motion) {
                if (model_column < MPC_MODEL_NX)
                    linearization->body_motion_A[motion][model_column] =
                        motion_derivatives[motion];
                else
                    linearization->body_motion_B[motion][
                        model_column - MPC_MODEL_NX] = motion_derivatives[motion];
            }
            const float d_path_forward =
                d_forward * tracking_c - d_lateral * tracking_s;
            const float d_path_lateral =
                d_forward * tracking_s + d_lateral * tracking_c;
            const float d_delta_s =
                (tracking_normal * d_path_forward +
                 tracking_tangent * d_path_lateral) *
                inverse_tracking_radius_squared;
            const float d_ey =
                (tracking_normal * d_path_lateral -
                 tracking_tangent * d_path_forward) *
                inverse_tracking_radius;
            const float d_epsi = d_yaw - path_curvature * d_delta_s;
            /* Derivative of the exact integrated-motion Frenet map. The path
             * coordinate columns e_y/e_psi remain on their cheap dedicated
             * path above; this maps plant-state and direct-input sensitivities. */
            if (!isfinite(d_ey) || !isfinite(d_epsi)) return 0;
            if (model_column < MPC_MODEL_NX) {
                linearization->A[MPC_STATE_IDX_EY][model_column] = d_ey;
                linearization->A[MPC_STATE_IDX_EPSI][model_column] = d_epsi;
            } else {
                const int input = model_column - MPC_MODEL_NX;
                linearization->B[MPC_STATE_IDX_EY][input] = d_ey;
                linearization->B[MPC_STATE_IDX_EPSI][input] = d_epsi;
            }
            const int command_derivative_is_local =
                model_column < MPC_MODEL_NX ||
                (model_column == MPC_MODEL_NX &&
                    control->steering_target_rad >
                        -UNITY_SOURCE_MAX_STEERING_RAD &&
                    control->steering_target_rad <
                        UNITY_SOURCE_MAX_STEERING_RAD) ||
                (model_column == MPC_MODEL_NX + 1 &&
                    control->throttle_command > MPC_THROTTLE_MIN &&
                    control->throttle_command < MPC_THROTTLE_MAX);
            if (command_derivative_is_local)
                *covered_columns |= UINT32_C(1) << model_column;
            else
                *covered_columns &= ~(UINT32_C(1) << model_column);
        }
    }
    return 1;
}

static int vehicle_model_linearize_with_known_step(
    const MpcModelState_t *state,
    const MpcModelControl_t *control,
    float dt,
    float path_curvature,
    const MpcStageResult_t *known_step,
    MpcStageResult_t *stage_out,
    MpcStageLinearization_t *linearization)
{
    if (!state || !control || !stage_out || !linearization) return 0;
    memset(linearization, 0, sizeof(*linearization));
    linearization->failure_index = -1;
    MpcModelState_t prepared_state = *state;
    if (!vehicle_model_initialize_plant_state(&prepared_state)) {
        linearization->failure_reason = 1;
        return 0;
    }
    state = &prepared_state;
    MpcStageResult_t known_storage = {0};
    if (known_step) {
        if (!known_step->valid ||
            !finite_mpc_state(&known_step->next) ||
            !isfinite(known_step->delta_s_m)) {
            linearization->failure_reason = 2;
            return 0;
        }
        known_storage = *known_step;
    } else if (!vehicle_model_step_scalar(
                   state, control, dt, path_curvature, 0, 1,
                   &known_storage)) {
        linearization->failure_reason = 2;
        return 0;
    }
    MpcStageResult_t *stage = &known_storage;
    *stage_out = known_storage;
    linearization->nominal_branch_flags = stage->branch_flags;

    static const float state_epsilon[MPC_MODEL_NX] = {
        1.0e-4f, 1.0e-3f, 1.0e-3f, 1.0e-3f, 1.0e-3f,
        1.0e-4f, 1.0e-4f, 1.0e-4f, 1.0e-3f, 1.0e-3f,
        1.0e-2f, 1.0e-2f, 1.0e-2f, 1.0e-2f,
        1.0e-3f, 1.0e-3f, 1.0e-3f, 1.0e-3f,
        1.0e-3f, 1.0e-3f,
        1.0e-4f, 1.0e-4f, 1.0e-4f, 1.0e-4f};
    static const float input_epsilon[MPC_MODEL_NU] = {1.0e-3f, 1.0e-2f};

    /* These columns have exact deterministic dependencies:
     * - e_y/e_psi enter only the cheap Frenet integration below;
     * - command history advances by shift and append and does not drive the
     *   plant inside the command-time-aligned MPC horizon.
     * Fill these exact dependencies directly rather than finite-differencing
     * the costly 1 ms tire/contact equations for every bookkeeping state. */
    const VehicleParameters_t parameters = vehicle_model_get_parameters();
    const int steering_unsaturated =
        control->steering_target_rad > -parameters.max_steering_angle &&
        control->steering_target_rad < parameters.max_steering_angle;
    const int throttle_unsaturated =
        control->throttle_command > MPC_THROTTLE_MIN &&
        control->throttle_command < MPC_THROTTLE_MAX;
    linearization->A[MPC_STATE_IDX_DELAYED_STEERING_COMMAND_1]
                     [MPC_STATE_IDX_DELAYED_STEERING_COMMAND_2] = 1.0f;
    /* The source plant receives the QP decision directly. These exact
     * dependencies only advance the actuator-history fields carried beside
     * the plant; they are not the physical input Jacobian. */
    linearization->B[MPC_STATE_IDX_DELAYED_STEERING_COMMAND_2][0] = 1.0f;
    linearization->A[MPC_STATE_IDX_DELAYED_THROTTLE_COMMAND_1]
                     [MPC_STATE_IDX_DELAYED_THROTTLE_COMMAND_2] = 1.0f;
    linearization->B[MPC_STATE_IDX_DELAYED_THROTTLE_COMMAND_2][1] = 1.0f;
    if (!steering_unsaturated) {
        linearization->nonsmooth_column_mask |= UINT64_C(1) <<
            (MPC_MODEL_NX + 0);
        ++linearization->nonsmooth_column_count;
    }
    if (!throttle_unsaturated) {
        linearization->nonsmooth_column_mask |= UINT64_C(1) <<
            (MPC_MODEL_NX + 1);
        ++linearization->nonsmooth_column_count;
    }

    /* Path-coordinate columns need only the already-computed body transition,
     * so use symmetric differences of the exact same midpoint equations. */
    for (int column = 0; column < 2; ++column) {
        const float epsilon = state_epsilon[column];
        MpcModelState_t state_plus = *state;
        MpcModelState_t state_minus = *state;
        mpc_state_perturb(&state_plus, column, epsilon);
        mpc_state_perturb(&state_minus, column, -epsilon);
        float plus_ey = 0.0f, plus_epsi = 0.0f, plus_ds = 0.0f;
        float minus_ey = 0.0f, minus_epsi = 0.0f, minus_ds = 0.0f;
        const int plus_valid = vehicle_model_tracking_step_integrated(
            &state_plus, stage->body_forward_delta_m,
            stage->body_lateral_delta_m, stage->body_yaw_delta_rad,
            path_curvature, &plus_ey, &plus_epsi, &plus_ds);
        const int minus_valid = vehicle_model_tracking_step_integrated(
            &state_minus, stage->body_forward_delta_m,
            stage->body_lateral_delta_m, stage->body_yaw_delta_rad,
            path_curvature, &minus_ey, &minus_epsi, &minus_ds);
        if (plus_valid && minus_valid) {
            linearization->A[0][column] =
                (plus_ey - minus_ey) / (2.0f * epsilon);
            linearization->A[1][column] = mpc_component_delta(
                1, plus_epsi, minus_epsi) / (2.0f * epsilon);
        } else if (plus_valid) {
            linearization->A[0][column] =
                (plus_ey - stage->next.e_y) / epsilon;
            linearization->A[1][column] = mpc_component_delta(
                1, plus_epsi, stage->next.e_psi) / epsilon;
            linearization->nonsmooth_column_mask |= UINT64_C(1) << column;
            ++linearization->nonsmooth_column_count;
        } else if (minus_valid) {
            linearization->A[0][column] =
                (stage->next.e_y - minus_ey) / epsilon;
            linearization->A[1][column] = mpc_component_delta(
                1, stage->next.e_psi, minus_epsi) / epsilon;
            linearization->nonsmooth_column_mask |= UINT64_C(1) << column;
            ++linearization->nonsmooth_column_count;
        } else {
            linearization->failure_reason = 3;
            linearization->failure_index = column;
            return 0;
        }
    }

    uint32_t sensitivity_covered_columns = 0;
    const int sensitivities_valid = vehicle_model_forward_plant_jacobian(
        state, control, dt, path_curvature, stage, linearization,
        &sensitivity_covered_columns);

    for (int column = 0; column < MPC_MODEL_NX + MPC_MODEL_NU; ++column) {
        const int is_state = column < MPC_MODEL_NX;
        const int index = is_state ? column : column - MPC_MODEL_NX;
        const int analytic_column = is_state
            ? (index == MPC_STATE_IDX_EY || index == MPC_STATE_IDX_EPSI ||
               /* Queue coordinates affect only exact shift/overwrite rows.
                * They are bookkeeping in this direct-input horizon and must
                * not trigger a finite-difference plant run. */
               index == MPC_STATE_IDX_DELAYED_STEERING_COMMAND_1 ||
               index == MPC_STATE_IDX_DELAYED_STEERING_COMMAND_2 ||
               index == MPC_STATE_IDX_DELAYED_THROTTLE_COMMAND_1 ||
               index == MPC_STATE_IDX_DELAYED_THROTTLE_COMMAND_2 ||
               (sensitivities_valid &&
                (sensitivity_covered_columns & (UINT32_C(1) << index)) != 0))
            : (sensitivities_valid &&
               (sensitivity_covered_columns &
                    (UINT32_C(1) << (MPC_MODEL_NX + index))) != 0);
        if (analytic_column) continue;
        ++linearization->finite_difference_column_count;
        const float epsilon = is_state
            ? state_epsilon[index] : input_epsilon[index];
        MpcModelState_t state_plus = *state;
        MpcModelControl_t control_plus = *control;
        if (is_state) {
            mpc_state_perturb(&state_plus, index, epsilon);
        } else if (index == 0) {
            control_plus.steering_target_rad += epsilon;
        } else {
            control_plus.throttle_command += epsilon;
        }
        MpcStageResult_t plus = {0};
        MpcStageResult_t minus = {0};
        (void)vehicle_model_step_scalar(
            &state_plus, &control_plus, dt, path_curvature, 0, 1, &plus);
        const int plus_same_branch = plus.valid &&
            plus.branch_flags == stage->branch_flags;

        /* Use a forward difference in smooth regions; pay for the opposite
         * side only when perturbation crosses a plant/model branch. */
        int nonsmooth = !plus_same_branch;
        if (!plus_same_branch) {
            MpcModelState_t state_minus = *state;
            MpcModelControl_t control_minus = *control;
            if (is_state) {
                mpc_state_perturb(&state_minus, index, -epsilon);
            } else if (index == 0) {
                control_minus.steering_target_rad -= epsilon;
            } else {
                control_minus.throttle_command -= epsilon;
            }
            (void)vehicle_model_step_scalar(
                &state_minus, &control_minus, dt, path_curvature, 0, 1,
                &minus);
        }
        const int minus_same_branch = minus.valid &&
            minus.branch_flags == stage->branch_flags;
        for (int row = 0; row < MPC_MODEL_NX; ++row) {
            const float base_value = mpc_stage_component(stage, row);
            const float plus_value = mpc_stage_component(&plus, row);
            const float minus_value = mpc_stage_component(&minus, row);
            float derivative;
            if (plus_same_branch) {
                derivative = mpc_component_delta(
                    row, plus_value, base_value) / epsilon;
            } else if (minus_same_branch) {
                derivative = mpc_component_delta(
                    row, base_value, minus_value) / epsilon;
            } else if (plus.valid && minus.valid) {
                const unsigned int plus_distance = mask_popcount(
                    (uint16_t)(plus.branch_flags ^ stage->branch_flags));
                const unsigned int minus_distance = mask_popcount(
                    (uint16_t)(minus.branch_flags ^ stage->branch_flags));
                derivative = plus_distance <= minus_distance
                    ? mpc_component_delta(row, plus_value, base_value) / epsilon
                    : mpc_component_delta(row, base_value, minus_value) / epsilon;
            } else if (plus.valid) {
                derivative = mpc_component_delta(
                    row, plus_value, base_value) / epsilon;
            } else if (minus.valid) {
                derivative = mpc_component_delta(
                    row, base_value, minus_value) / epsilon;
            } else {
                linearization->failure_reason = 4;
                linearization->failure_index = column;
                return 0;
            }
            if (!isfinite(derivative)) {
                linearization->failure_reason = 5;
                linearization->failure_index = column;
                return 0;
            }
            /* Queue append is exact even when the physical command is at a
             * bound and its plant derivative needs a one-sided difference. */
            if (!is_state &&
                ((index == 0 && row ==
                    MPC_STATE_IDX_DELAYED_STEERING_COMMAND_2) ||
                 (index == 1 && row ==
                    MPC_STATE_IDX_DELAYED_THROTTLE_COMMAND_2))) {
                continue;
            }
            if (is_state) linearization->A[row][index] = derivative;
            else linearization->B[row][index] = derivative;
        }
        /* Preserve the same one-sided branch choice for physical integrated
         * displacement, so the loaded-path transform can apply its own exact
         * geometric derivative without rerunning the source plant. */
        const int use_plus = plus_same_branch ||
            (!minus_same_branch && plus.valid && (!minus.valid ||
             mask_popcount((uint16_t)(plus.branch_flags ^ stage->branch_flags)) <=
             mask_popcount((uint16_t)(minus.branch_flags ^ stage->branch_flags))));
        const MpcStageResult_t *offset = use_plus ? &plus : &minus;
        const float scale = (use_plus ? 1.0f : -1.0f) / epsilon;
        const float motion_derivatives[3] = {
            (offset->body_forward_delta_m - stage->body_forward_delta_m) * scale,
            (offset->body_lateral_delta_m - stage->body_lateral_delta_m) * scale,
            (offset->body_yaw_delta_rad - stage->body_yaw_delta_rad) * scale};
        for (int motion = 0; motion < 3; ++motion) {
            if (is_state)
                linearization->body_motion_A[motion][index] =
                    motion_derivatives[motion];
            else
                linearization->body_motion_B[motion][index] =
                    motion_derivatives[motion];
        }
        if (nonsmooth) {
                linearization->nonsmooth_column_mask |= UINT64_C(1) << column;
            ++linearization->nonsmooth_column_count;
        }
    }
    linearization->valid = 1;
    return 1;
}

int mpc_vehicle_model_step_with_jacobian(
    const MpcModelState_t *state,
    const MpcModelControl_t *control,
    float dt,
    float path_curvature,
    MpcStageResult_t *stage,
    MpcStageLinearization_t *linearization)
{
    return vehicle_model_linearize_with_known_step(
        state, control, dt, path_curvature, NULL, stage, linearization);
}

int mpc_vehicle_model_linearize_from_known_step(
    const MpcModelState_t *state,
    const MpcModelControl_t *control,
    float dt,
    float path_curvature,
    const MpcStageResult_t *known_step,
    MpcStageResult_t *stage,
    MpcStageLinearization_t *linearization)
{
    return vehicle_model_linearize_with_known_step(
        state, control, dt, path_curvature, known_step, stage,
        linearization);
}

MpcStageResult_t mpc_vehicle_model_step(
    const MpcModelState_t *state,
    const MpcModelControl_t *control,
    float dt,
    float path_curvature)
{
    MpcStageResult_t stage = {0};
    vehicle_model_step_scalar(state, control, dt, path_curvature, 0, 1,
        &stage);
    return stage;
}

int vehicle_model_reproject_tracking(
    const MpcModelState_t *state,
    const MpcStageResult_t *physical_step,
    float path_curvature,
    MpcStageResult_t *reprojected_step)
{
    if (!state || !physical_step || !reprojected_step ||
        !physical_step->valid || !finite_mpc_state(state) ||
        !finite_mpc_state(&physical_step->next) ||
        !isfinite(physical_step->body_forward_delta_m) ||
        !isfinite(physical_step->body_lateral_delta_m) ||
        !isfinite(physical_step->body_yaw_delta_rad) ||
        !isfinite(path_curvature)) {
        return 0;
    }

    *reprojected_step = *physical_step;
    if (!vehicle_model_tracking_step_integrated(
            state, physical_step->body_forward_delta_m,
            physical_step->body_lateral_delta_m,
            physical_step->body_yaw_delta_rad, path_curvature,
            &reprojected_step->next.e_y,
            &reprojected_step->next.e_psi,
            &reprojected_step->delta_s_m)) {
        reprojected_step->valid = 0;
        return 0;
    }
    reprojected_step->valid = finite_mpc_state(&reprojected_step->next) &&
        isfinite(reprojected_step->delta_s_m);
    return reprojected_step->valid;
}

MpcStageResult_t mpc_vehicle_model_propagate(
    const MpcModelState_t *state,
    float dt,
    float path_curvature)
{
    MpcStageResult_t stage = {0};
    if (!state) return stage;
    const MpcModelControl_t queued_command = {
        state->delayed_steering_command_2,
        state->delayed_throttle_command_2};
    vehicle_model_step_scalar(
        state, &queued_command, dt, path_curvature, 1, 0, &stage);
    return stage;
}

int vehicle_model_queue_command(
    MpcModelState_t *state,
    float steering_target_rad,
    float throttle_command,
    int throttle_valid)
{
    if (!state || !isfinite(steering_target_rad) ||
        fabsf(steering_target_rad) >
            vehicle_model_get_parameters().max_steering_angle ||
        (throttle_valid && (!isfinite(throttle_command) ||
            throttle_command < MPC_THROTTLE_MIN ||
            throttle_command > MPC_THROTTLE_MAX))) return 0;
    state->delayed_steering_command_1 = state->delayed_steering_command_2;
    state->delayed_steering_command_2 = steering_target_rad;
    if (throttle_valid) {
        state->delayed_throttle_command_1 =
            state->delayed_throttle_command_2;
        state->delayed_throttle_command_2 = throttle_command;
    }
    return 1;
}
