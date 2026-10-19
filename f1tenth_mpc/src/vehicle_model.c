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

/* Conservative residual use selected by cross-run replay. This is an
 * observer correction, not a PhysX parameter; the source plant remains the
 * physical transition and the correction is disabled outside its support. */
static const float kYawResidualCorrectionGain = 0.5f;
static const float kYawResidualMinSpeedMps = 2.8f;
static const float kYawResidualMaxSpeedMps = 5.0f;
static const float kYawResidualMinSteeringRad = 0.10f;
static const float kYawResidualMaxSteeringRad = 0.31f;

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
        !isfinite(state->yaw_rate_prediction_residual_radps) ||
        (state->yaw_rate_residual_correction_enabled != 0 &&
         state->yaw_rate_residual_correction_enabled != 1) ||
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
    *delta_s = path_forward_delta / denominator;
    *e_y_next = state->e_y + path_lateral_delta -
        0.5f * path_curvature * (*delta_s) * (*delta_s);
    const float e_psi_raw = state->e_psi + body_yaw_delta_rad -
        path_curvature * (*delta_s);
    *e_psi_next = atan2f(sinf(e_psi_raw), cosf(e_psi_raw));
    return isfinite(*e_y_next) && isfinite(*e_psi_next) &&
        isfinite(*delta_s);
}

/* Nonlinear rollouts use the 1 ms source-derived four-wheel plant. The compact
 * RTI state tracks path/body/controller variables, while this carried state
 * preserves wheel spin, sprung-body motion, and suspension history across
 * every stage of the horizon. */
static int vehicle_model_step_scalar(
    const MpcModelState_t *state,
    const MpcModelControl_t *control,
    float dt,
    float path_curvature,
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
        state->delayed_steering_command_1);
    source_input.throttle_normalized = state->delayed_throttle_command_1;
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
    const float raw_r_next =
        (float)source_output.state[UNITY_SOURCE_YAW_RATE];
    const float predicted_speed = fabsf(
        (float)source_output.state[UNITY_SOURCE_U]);
    const float predicted_steering = fabsf(
        (float)source_output.state[UNITY_SOURCE_PHYSICAL_STEERING]);
    const float yaw_residual = state->yaw_rate_prediction_residual_radps;
    const int residual_supported =
        state->yaw_rate_residual_correction_enabled &&
        predicted_speed >= kYawResidualMinSpeedMps &&
        predicted_speed <= kYawResidualMaxSpeedMps &&
        predicted_steering >= kYawResidualMinSteeringRad &&
        predicted_steering <= kYawResidualMaxSteeringRad &&
        source_output.state[UNITY_SOURCE_PHYSICAL_STEERING] *
            state->actual_steering_angle > 0.0 &&
        raw_r_next * yaw_residual < 0.0f;
    const float r_next = raw_r_next - (residual_supported ?
        kYawResidualCorrectionGain * yaw_residual : 0.0f);
    const float yaw_rate_correction = r_next - raw_r_next;
    const float body_yaw_delta = (float)
        source_output.integrated_yaw_delta_rad + yaw_rate_correction * dt;
    const float body_forward_delta = (float)
        source_output.integrated_body_forward_displacement_m;
    const float body_lateral_delta = (float)
        (source_output.integrated_body_lateral_displacement_m -
         yaw_rate_correction * kComForwardFromRearAxleM * dt);
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
        /* Match odometry's two-slot packet FIFO: the plant used slot 1 for
         * this interval; now advance slot 2 and append this interval's new
         * decision. */
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
                   state, control, dt, path_curvature, 1, &known_storage)) {
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
     * - each two-slot actuator FIFO advances by shift and append.
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
    /* These are exact command-queue dependencies. At a control bound the
     * QP's feasible direction is one-sided (inward), but the queue still
     * changes one-for-one with that feasible control change. A zero
     * derivative at the lower throttle bound makes a stationary vehicle
     * unable to discover positive throttle at all. */
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

    for (int column = 0; column < MPC_MODEL_NX + MPC_MODEL_NU; ++column) {
        const int is_state = column < MPC_MODEL_NX;
        const int index = is_state ? column : column - MPC_MODEL_NX;
        const int analytic_column = is_state
            ? (index == MPC_STATE_IDX_EY || index == MPC_STATE_IDX_EPSI ||
               index == MPC_STATE_IDX_DELAYED_STEERING_COMMAND_2 ||
               index == MPC_STATE_IDX_DELAYED_THROTTLE_COMMAND_2)
            : 1;
        if (analytic_column) continue;
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
            &state_plus, &control_plus, dt, path_curvature, 1, &plus);
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
                &state_minus, &control_minus, dt, path_curvature, 1, &minus);
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
            if (is_state) linearization->A[row][index] = derivative;
            else linearization->B[row][index] = derivative;
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
    vehicle_model_step_scalar(state, control, dt, path_curvature, 1, &stage);
    return stage;
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
        state, &queued_command, dt, path_curvature, 0, &stage);
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
