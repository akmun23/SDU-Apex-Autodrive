/**
 * @file vehicle_model.cpp
 * @brief MPC transition backed by the checked-in 1 ms four-wheel plant.
 *
 * Steering response, wheel spin, tire slip, suspension, and body motion all
 * come from the same recursive source-derived plant. No fitted yaw response
 * is layered on top of those dynamics.
 */

#include <cstdint>
#include "vehicle_model.h"

#include <algorithm>
#include <math.h>
#include <stddef.h>
#include <string.h>

/* Unity's Rigidbody velocity is measured at its COM. The MPC state and
 * /odom twist are referenced to base_link at the rear axle. The source
 * vehicle geometry puts COM 0.15532 m forward of that axle. */
static const double kComForwardFromRearAxleM = 0.15532;

static VehicleParameters_t active_parameters = {
    .steering_wheelbase_m = SOURCE_STEERING_WHEELBASE_M,
    .max_steering_angle = SOURCE_MAX_STEERING_RAD,
    .maximum_command_speed_mps = MPC_MAX_COMMAND_SPEED_MPS,
};

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
    return VehicleParameters_t{
        .steering_wheelbase_m = SOURCE_STEERING_WHEELBASE_M,
        .max_steering_angle = SOURCE_MAX_STEERING_RAD,
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
        !finite_positive(parameters->maximum_command_speed_mps)) {
        return 0;
    }
    active_parameters = *parameters;
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

int vehicle_model_pack_source_state(
    const MpcModelState_t *state,
    double source[UNITY_SOURCE_PLANT_STATE_SIZE])
{
    if(!state||!source||!state->source_plant_initialized)
        return 0;
    write_source_plant_state(state,source);
    for(int i=0;i<UNITY_SOURCE_PLANT_STATE_SIZE;++i)
        if(!std::isfinite(source[i]))return 0;
    return 1;
}

int vehicle_model_inject_source_state(
    MpcModelState_t *state,
    const double source[UNITY_SOURCE_PLANT_STATE_SIZE])
{
    if(!state||!source||!state->source_plant_initialized)
        return 0;
    for(int i=0;i<UNITY_SOURCE_PLANT_STATE_SIZE;++i)
        if(!std::isfinite(source[i]))return 0;
    state->u=static_cast<float>(source[UNITY_SOURCE_U]);
    state->r=static_cast<float>(source[UNITY_SOURCE_YAW_RATE]);
    state->v=static_cast<float>(
        source[UNITY_SOURCE_V]-
        kComForwardFromRearAxleM*source[UNITY_SOURCE_YAW_RATE]);
    state->actual_steering_angle=static_cast<float>(
        source[UNITY_SOURCE_PHYSICAL_STEERING]);
    source_state_to_internal(state,source);
    return std::isfinite(state->u) && std::isfinite(state->v) &&
           std::isfinite(state->r) &&
           std::isfinite(state->actual_steering_angle);
}

int vehicle_model_source_interval_hessian(
    const MpcModelState_t *state,const MpcModelControl_t *control,float dt,
    unsigned int force_substep_cap,UnitySourcePlantHessianC *hessian)
{
    if(!state||!control||!hessian||!state->source_plant_initialized||
       !std::isfinite(state->u)||!std::isfinite(state->v)||
       !std::isfinite(state->r)||
       !std::isfinite(state->actual_steering_angle)||
       !std::isfinite(dt)||dt<=0.f||
       !std::isfinite(control->steering_target_rad)||
       !std::isfinite(control->throttle_command))return 0;
    UnitySourcePlantInputC input{};
    write_source_plant_state(state,input.state);
    input.steering_target_rad=clamp_source_steering_target(
        clampf_local(control->steering_target_rad,
            -active_parameters.max_steering_angle,
            active_parameters.max_steering_angle));
    input.throttle_normalized=clampf_local(
        control->throttle_command,MPC_THROTTLE_MIN,MPC_THROTTLE_MAX);
    input.suspension_history_initialized=
        state->source_suspension_history_initialized;
    if(!unity_source_plant_step_hessian_c(&input,dt,force_substep_cap,hessian))return 0;
    const bool steer_interior=
        control->steering_target_rad>= -active_parameters.max_steering_angle &&
        control->steering_target_rad<= active_parameters.max_steering_angle;
    const bool throttle_interior=
        control->throttle_command>MPC_THROTTLE_MIN &&
        control->throttle_command<=MPC_THROTTLE_MAX;
    for(int o=0;o<UNITY_SOURCE_HESSIAN_OUTPUTS;++o)
        for(int i=0;i<UNITY_SOURCE_HESSIAN_INPUTS;++i) {
            const bool active_i=i<3 || (i==3?steer_interior:throttle_interior);
            if(!active_i)hessian->gradient[o][i]=0.;
            if(!std::isfinite(hessian->gradient[o][i]))return 0;
            for(int j=0;j<UNITY_SOURCE_HESSIAN_INPUTS;++j) {
                const bool active_j=j<3 || (j==3?steer_interior:throttle_interior);
                if(!active_i||!active_j)hessian->hessian[o][i][j]=0.;
                if(!std::isfinite(hessian->hessian[o][i][j]))return 0;
            }
        }
    return 1;
}

int vehicle_model_full_source_interval_sensitivities(
    const MpcModelState_t *state,const MpcModelControl_t *control,float dt,
    const double seed_state[26][UNITY_SOURCE_PLANT_SENSITIVITY_WIDTH],
    const double seed_control[2][UNITY_SOURCE_PLANT_SENSITIVITY_WIDTH],
    unsigned int force_substep_cap,
    double next_state[26][UNITY_SOURCE_PLANT_SENSITIVITY_WIDTH],
    double body_motion[3][UNITY_SOURCE_PLANT_SENSITIVITY_WIDTH],
    double grip_loss[4],
    double grip_loss_derivative[4][UNITY_SOURCE_PLANT_SENSITIVITY_WIDTH])
{
    static_assert(UNITY_SOURCE_PLANT_STATE_SIZE==26);
    static_assert(UNITY_SOURCE_PLANT_SENSITIVITY_WIDTH==28);
    if(!state||!control||!seed_state||!seed_control||!next_state||
       !body_motion||!grip_loss||!grip_loss_derivative||
       !state->source_plant_initialized||
       !std::isfinite(state->u)||!std::isfinite(state->v)||
       !std::isfinite(state->r)||
       !std::isfinite(state->actual_steering_angle)||
       !std::isfinite(dt)||dt<=0.f||
       !std::isfinite(control->steering_target_rad)||
       !std::isfinite(control->throttle_command))return 0;
    UnitySourcePlantInputC input{};
    write_source_plant_state(state,input.state);
    input.steering_target_rad=clamp_source_steering_target(
        clampf_local(control->steering_target_rad,
            -active_parameters.max_steering_angle,
            active_parameters.max_steering_angle));
    input.throttle_normalized=clampf_local(
        control->throttle_command,MPC_THROTTLE_MIN,MPC_THROTTLE_MAX);
    input.suspension_history_initialized=
        state->source_suspension_history_initialized;
    UnitySourcePlantSensitivitySeedsC seeds;
    for(int k=0;k<26;++k)
        for(int lane=0;lane<UNITY_SOURCE_PLANT_SENSITIVITY_WIDTH;++lane)
            seeds.state[k][lane]=seed_state[k][lane];
    // At a legal actuator bound the QP can only move inward. The
    // source steering clamp uses strict (<,>) branch tests, so its dual
    // derivative at equality is the physically correct inward one-sided
    // sensitivity. At zero throttle the source switches to full braking and
    // has no classical derivative; keep that distinct.
    const bool steer_interior=
        control->steering_target_rad>= -active_parameters.max_steering_angle &&
        control->steering_target_rad<= active_parameters.max_steering_angle;
    const bool throttle_interior=
        control->throttle_command>MPC_THROTTLE_MIN &&
        control->throttle_command<=MPC_THROTTLE_MAX;
    for(int lane=0;lane<UNITY_SOURCE_PLANT_SENSITIVITY_WIDTH;++lane) {
        seeds.steering_target[lane]=steer_interior?
            seed_control[0][lane]:0.;
        seeds.throttle[lane]=throttle_interior?
            seed_control[1][lane]:0.;
    }
    // The C sensitivity API fills every consumed result coordinate on success.
    UnitySourcePlantSensitivityC result;
    if(!unity_source_plant_step_sensitivity_c(&input,dt,force_substep_cap,&seeds,&result))
        return 0;
    for(int k=0;k<26;++k)
        for(int lane=0;lane<UNITY_SOURCE_PLANT_SENSITIVITY_WIDTH;++lane) {
            const double value=result.state[k][lane];
            if(!std::isfinite(value))return 0;
            next_state[k][lane]=value;
        }
    for(int lane=0;lane<UNITY_SOURCE_PLANT_SENSITIVITY_WIDTH;++lane) {
        body_motion[0][lane]=result.body_forward_displacement[lane];
        body_motion[1][lane]=result.body_lateral_displacement[lane];
        body_motion[2][lane]=result.yaw_delta[lane];
        for(int m=0;m<3;++m)
            if(!std::isfinite(body_motion[m][lane]))return 0;
        for(int w=0;w<4;++w) {
            grip_loss_derivative[w][lane]=
                result.maximum_grip_loss_derivative[w][lane];
            if(!std::isfinite(grip_loss_derivative[w][lane]))return 0;
        }
    }
    for(int w=0;w<4;++w) {
        grip_loss[w]=result.maximum_grip_loss[w];
        if(!std::isfinite(grip_loss[w]))return 0;
    }
    return 1;
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
     * This coordinate transform preserves constant-curvature motion and
     * the Unity plant displacement.
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
 * The cycle state tracks path/body/controller variables, while this carried state
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
    *stage = MpcStageResult_t{0};
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
    UnitySourcePlantOutputC source_output;
    if (!unity_source_plant_step_c(
            &source_input, dt, &source_output)) return 0;
    memcpy(stage->output.maximum_grip_loss,
           source_output.maximum_grip_loss,
           sizeof(stage->output.maximum_grip_loss));
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

MpcStageResult_t mpc_vehicle_model_step(
    const MpcModelState_t *state,
    const MpcModelControl_t *control,
    float dt,
    float path_curvature)
{
    MpcStageResult_t stage = {};
    vehicle_model_step_scalar(state, control, dt, path_curvature, 0, 1,
        &stage);
    return stage;
}

MpcStageResult_t mpc_vehicle_model_propagate(
    const MpcModelState_t *state,
    float dt,
    float path_curvature)
{
    MpcStageResult_t stage = {};
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
