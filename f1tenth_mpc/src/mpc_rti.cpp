#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define _POSIX_C_SOURCE 200809L

#include "mpc_rti_internal.hpp"
#include "mpc_source_exact_offline_solver.hpp"
#include "mpc_steering_variation.hpp"
#include "mpc_tracking_tolerance.h"
#include "vehicle_model.h"

#include <algorithm>
#include <memory>
#include <cmath>
#include <math.h>
#include <string.h>
#include <vector>
#ifdef __linux__
#include <sys/syscall.h>
#include <unistd.h>
#endif
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

double monotonic_microseconds(void);

double thread_cpu_microseconds(void)
{
#ifdef CLOCK_THREAD_CPUTIME_ID
    struct timespec now;
    if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &now) == 0)
        return (double)now.tv_sec * 1.0e6 + (double)now.tv_nsec * 1.0e-3;
#endif
    return NAN;
}

typedef struct { float left; float right; } MpcCorridorMargins;

static MpcCorridorMargins corridor_margin_for_state(
    const MpcRtiConfiguration_t *configuration,
    const MpcModelState_t *state,
    int prediction_index);

static int finite_nonnegative(float value)
{
    return isfinite(value) && value >= 0.0f;
}

int finite_rti_state(const MpcRtiState_t *state)
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

int finite_reference(const MpcRtiReference_t *reference)
{
    return reference && isfinite(reference->e_y) &&
        isfinite(reference->e_psi) && isfinite(reference->u) &&
        isfinite(reference->v) && isfinite(reference->r) &&
        isfinite(reference->steering_command) &&
        isfinite(reference->path_curvature) &&
        isfinite(reference->left_bound) && reference->left_bound > 0.0f &&
        isfinite(reference->right_bound) && reference->right_bound > 0.0f &&
        (!reference->source_witness_valid ||
         (isfinite(reference->source_steering_target) &&
          isfinite(reference->source_throttle) &&
          reference->source_throttle >= MPC_THROTTLE_MIN_DRIVE &&
          reference->source_throttle <= MPC_THROTTLE_MAX));
}

int valid_configuration(const MpcRtiConfiguration_t *configuration)
{
    if (!configuration) return 0;
    const VehicleParameters_t vehicle = vehicle_model_get_parameters();
    const float weights[] = {
        configuration->weight_e_y,
        configuration->weight_e_psi,
        configuration->tracking_lateral_tolerance_m,
        configuration->tracking_heading_tolerance_rad,
        configuration->weight_u,
        configuration->weight_u_overspeed,
        configuration->overspeed_curvature_gain_m,
        configuration->underspeed_curvature_gain_m,
        configuration->weight_longitudinal_slip,
        configuration->longitudinal_slip_curvature_gain_m,
        configuration->weight_v,
        configuration->weight_r,
        configuration->weight_steering_command,
        configuration->weight_steering_target,
        configuration->weight_throttle_command,
        configuration->weight_steering_slew,
        configuration->weight_steering_correction_variation,
        configuration->weight_throttle_slew,
        configuration->max_steering_command_step_rad,
        configuration->max_throttle_command_step,
        configuration->startup_nominal_throttle_seed,
        configuration->terminal_multiplier,
        configuration->corridor_margin_m,
        configuration->first_prediction_corridor_margin_m,
        configuration->corridor_preview_halfwidth_m,
        configuration->reference_clearance_reserve_m,
        configuration->near_wall_tracking_reserve_m,
        configuration->weight_source_corridor_reserve,
        configuration->weight_source_lateral_slip,
        configuration->nonlinear_corridor_tolerance_m};
    for (unsigned int i = 0; i < sizeof(weights) / sizeof(weights[0]); ++i)
        if (!finite_nonnegative(weights[i])) return 0;

    return configuration->tracking_lateral_tolerance_m <= 0.15f &&
        configuration->tracking_heading_tolerance_rad <= 0.15f &&
        configuration->near_wall_tracking_reserve_m <= 0.15f &&
        configuration->overspeed_curvature_gain_m <= 10.0f &&
        configuration->underspeed_curvature_gain_m <= 10.0f &&
        configuration->jacobian_threads >= 0 &&
        configuration->jacobian_threads <= MPC_RTI_MAX_JACOBIAN_THREADS &&
        isfinite(configuration->max_speed_mps) &&
        configuration->max_speed_mps > 0.0f &&
        configuration->max_speed_mps <=
            vehicle.maximum_command_speed_mps &&
        isfinite(configuration->active_speed_ceiling_mps) &&
        configuration->active_speed_ceiling_mps > 0.0f &&
        configuration->active_speed_ceiling_mps <=
            configuration->max_speed_mps &&
        configuration->startup_nominal_throttle_seed <=
            MPC_THROTTLE_MAX &&
        // These constraints belong to the ORIGINAL source NLP, even
        // without an incremental QP. A zero (leftover from removing the
        // old parser) silently prohibits both steering and throttle.
        configuration->max_steering_command_step_rad > 0.f &&
        configuration->max_throttle_command_step > 0.f &&
        (configuration->command_increment_qp_enabled == 0 ||
         (configuration->command_increment_qp_enabled == 1 &&
          configuration->max_steering_command_step_rad > 0.f &&
          configuration->max_steering_command_step_rad <=
              2.f * configuration->max_steering_rad &&
          configuration->max_throttle_command_step > 0.f &&
          configuration->max_throttle_command_step <= 1.f)) &&
        (configuration->source_reachable_steering_qp == 0 ||
         (configuration->source_reachable_steering_qp == 1 &&
          configuration->command_increment_qp_enabled == 1)) &&
        (configuration->weight_steering_correction_variation == 0.f ||
         (isfinite(configuration->steering_correction_variation_epsilon_rad) &&
          configuration->steering_correction_variation_epsilon_rad >= 0.005f &&
          configuration->steering_correction_variation_epsilon_rad <= 0.10f)) &&
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

void serialize_state(const MpcRtiState_t *state, float x[MPC_RTI_NX])
{
    x[MPC_OPT_IDX_PROGRESS_DEVIATION] = 0.0f;
    MpcModelState_t prepared_plant = state->plant;
    (void)vehicle_model_initialize_plant_state(&prepared_plant);
    float full[MPC_MODEL_NX] = {};
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

static int reference_at_progress_with_sample(
    const MpcTrajectorySample_t *trajectory,
    size_t trajectory_count,
    double lap_length,
    double progress,
    const MpcRtiConfiguration_t *configuration,
    MpcRtiReference_t *reference,
    MpcTrajectorySample_t *world_sample,
    const MpcTrajectorySample_t *projected_sample = nullptr)
{
    MpcTrajectorySample_t sample;
    if (!reference ||
        (!projected_sample && !mpc_trajectory_sample(trajectory,
            trajectory_count, lap_length, progress, &sample))) return 0;
    if (projected_sample) sample = *projected_sample;
    /* Keep the ORIGINAL interpolated geometry, before adjusting witness
     * targets/speed for the current optimization phase. The same sample
     * is used to reconstruct physical world-frame rear-axle poses. */
    if (world_sample) *world_sample = sample;
    const float speed = fminf((float)sample.speed,
        configuration->active_speed_ceiling_mps);
    /* A witness certifies ONE speed profile. During an explicitly requested
     * warmup/capped-speed phase use geometry-only tracking references instead;
     * neither its lateral state nor its torque is valid at the other speed.
     * The same full nonlinear source plant still predicts every candidate.
     * Once uncapped, the original witness references are restored unchanged. */
    if (sample.source_witness_valid && speed < (float)sample.speed-1.e-4f)
        sample.source_witness_valid = 0;
    // The smooth robust contour has its own tangent and curvature.
    // Steering/yaw/heading targets must follow THAT curve, or the
    // strong heading cost resists every intentional inboard maneuver.
    const float contour_steering_correction =
        (float)atan(.324*sample.robust_contour_curvature_delta_radpm);
    const float feedforward = sample.source_witness_valid ?
        clampf_rti((float)sample.source_physical_steering +
            contour_steering_correction,
            -configuration->max_steering_rad, configuration->max_steering_rad)
        : clampf_rti(
            vehicle_model_steering_for_curvature((float)(
                sample.curvature+sample.robust_contour_curvature_delta_radpm)),
            -configuration->max_steering_rad, configuration->max_steering_rad);
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
    *reference = MpcRtiReference_t{
        .e_y = (float)sample.robust_contour_lateral_offset_m,
        .e_psi = (sample.source_witness_valid ?
                     (float)sample.source_heading_error : 0.f) +
                 (float)sample.robust_contour_heading_offset_rad,
        .u = speed,
        .v = sample.source_witness_valid ? (float)sample.source_v_rear : 0.0f,
        .r = (sample.source_witness_valid
             ? (float)sample.source_yaw_rate : (float)sample.curvature*speed)
             + speed*(float)sample.robust_contour_curvature_delta_radpm,
        .steering_command = feedforward,
        .path_curvature = (float)sample.curvature,
        .left_bound = left_bound,
        .right_bound = right_bound,
        .source_witness_valid = sample.source_witness_valid,
        .source_steering_target = clampf_rti(
            (float)sample.source_steering_target +
                contour_steering_correction,
            -configuration->max_steering_rad,configuration->max_steering_rad),
        .source_throttle = (float)sample.source_throttle};
    if (configuration->reference_clearance_reserve_m > 0.f) {
        MpcModelState_t target_body{};
        target_body.e_psi = reference->e_psi;
        const MpcCorridorMargins body = corridor_margin_for_state(configuration, &target_body, 2);
        const float lower = body.right - right_bound;
        const float upper = left_bound - body.left;
        if (lower < upper) {
            const float reserve = fminf(configuration->reference_clearance_reserve_m, .5f*(upper-lower));
            reference->e_y = clampf_rti(
                reference->e_y, lower+reserve, upper-reserve);
        }
    }
    return finite_reference(reference);
}

/* Standard reference callers do not need a copy of world-frame geometry. */
int mpc_rti_reference_at_progress(
    const MpcTrajectorySample_t *trajectory, size_t trajectory_count,
    double lap_length, double progress,
    const MpcRtiConfiguration_t *configuration,
    MpcRtiReference_t *reference)
{
    return reference_at_progress_with_sample(
        trajectory, trajectory_count, lap_length, progress,
        configuration, reference, NULL);
}

int mpc_rti_source_command_rate_margins(
    const MpcModelState_t *start,const MpcModelControl_t *command,
    const MpcRtiConfiguration_t *configuration,double out[4])
{
    if(!start||!command||!configuration||!out ||
       !isfinite(start->actual_steering_angle) ||
       !isfinite(start->delayed_throttle_command_2) ||
       !isfinite(command->steering_target_rad) ||
       !isfinite(command->throttle_command) ||
       !isfinite(configuration->max_steering_command_step_rad) ||
       !isfinite(configuration->max_throttle_command_step))
        return 0;
    const double steering_difference=
        double(command->steering_target_rad)-
        double(start->actual_steering_angle);
    const double throttle_difference=
        double(command->throttle_command)-
        double(start->delayed_throttle_command_2);
    const double physical_steering_limit=
        configuration->max_steering_command_step_rad;
    const double throttle_limit=configuration->max_throttle_command_step;
    out[0]=physical_steering_limit-steering_difference;
    out[1]=physical_steering_limit+steering_difference;
    out[2]=throttle_limit-throttle_difference;
    out[3]=throttle_limit+throttle_difference;
    return 1;
}

int mpc_rti_source_exact_stage(
    const MpcRtiState_t *current,double current_progress,
    const MpcModelControl_t *command,
    const MpcTrajectorySample_t *trajectory,size_t trajectory_count,
    double lap_length,float dt,MpcRtiState_t *next,
    double *next_progress)
{
    if(!current||!command||!trajectory||!next||!next_progress||
       !isfinite(current_progress)||!isfinite(lap_length)||lap_length<=0.||
       !isfinite(dt)||dt<=0.f)return 0;
    MpcStageResult_t physical=mpc_vehicle_model_step(
        &current->plant,command,dt,0.0f);
    if(!physical.valid||
       !isfinite(current_progress+physical.delta_s_m))return 0;
    MpcRtiState_t updated={};
    if(!mpc_trajectory_advance_tracking_with_progress(
            trajectory,trajectory_count,lap_length,current_progress,
            current->plant.e_y,current->plant.e_psi,
            physical.body_forward_delta_m,
            physical.body_lateral_delta_m,
            physical.body_yaw_delta_rad,
            current_progress+physical.delta_s_m,
            next_progress,&physical.next.e_y,&physical.next.e_psi,
            updated.incoming_tracking_jacobian))return 0;
    updated.incoming_tracking_jacobian_valid=1;
    updated.plant=physical.next;
    updated.incoming_branch_flags=physical.branch_flags;
    updated.incoming_body_forward_delta_m=
        physical.body_forward_delta_m;
    updated.incoming_body_lateral_delta_m=
        physical.body_lateral_delta_m;
    updated.incoming_body_yaw_delta_rad=
        physical.body_yaw_delta_rad;
    store_slip_metrics(&updated,&physical);
    updated.incoming_transition_valid=1;
    *next=updated;
    return 1;
}

static int rollout_nominal_controls(
    const MpcRtiState_t *initial_state,
    double initial_progress,
    const MpcModelControl_t controls[PREDICTION_HORIZON],
    const MpcTrajectorySample_t *trajectory, size_t trajectory_count,
    double lap_length,
    int horizon,
    float prediction_dt,
    MpcRtiState_t states[PREDICTION_HORIZON + 1],
    double progress[PREDICTION_HORIZON + 1], int *failure_index)
{
    states[0] = *initial_state;
    progress[0] = initial_progress;
    for (int k=0;k<horizon;++k) {
        if(!mpc_rti_source_exact_stage(&states[k],progress[k],
                &controls[k],trajectory,trajectory_count,lap_length,
                prediction_dt,&states[k+1],&progress[k+1])) {
            *failure_index=k+1;
            return 0;
        }
    }
    return 1;
}

static void align_yaw_rate_references(
    MpcRtiReference_t references[PREDICTION_HORIZON + 1],
    int horizon)
{
    for (int k = 0; k < horizon; ++k) {
        if (!references[k].source_witness_valid)
            references[k].r = references[k].path_curvature * references[k].u;
    }
    if (!references[horizon].source_witness_valid)
        references[horizon].r = references[horizon].path_curvature *
            references[horizon].u;
}

int mpc_rti_references_from_exact_nominal(
    const MpcRtiNominal_t *nominal,
    const MpcTrajectorySample_t *trajectory, size_t trajectory_count,
    double lap_length, const MpcRtiConfiguration_t *model,
    MpcRtiReference_t references[PREDICTION_HORIZON + 1])
{
    if(!nominal || !nominal->valid ||
       (nominal->horizon<1 || nominal->horizon>PREDICTION_HORIZON) ||
       !trajectory || trajectory_count < 3 || !model || !references ||
       !isfinite(lap_length) || lap_length <= 0.)
        return 0;
    for(int k=0;k<=nominal->horizon;++k) {
        if(!isfinite(nominal->progress[k]) ||
           !mpc_rti_reference_at_progress(trajectory,trajectory_count,lap_length,
               nominal->progress[k],model,&references[k]))
            return 0;
    }
    references[0].u = nominal->states[0].plant.u;
    align_yaw_rate_references(references,nominal->horizon);
    return 1;
}

/* Experimental independently mapped swept-body authority. This is a
 * *second* hard source-step checker when opt-in, superseding approximate
 * Frenet corridor bounds only after exact map-cell validation. The reference
 * frame is the physical rear axle, not the center of the vehicle rectangle.
 * No map: callback returns false and the rollout fails closed. */
/* Reuse the source trajectory samples ALREADY calculated by the exact
 * nonlinear rollout. Two redundant binary searches per checked movement
 * were previously performed here. The independent scoring path retains
 * its own sampling wrapper for callers without live rollout samples. */
static int world_sweep_clearance_from_samples(
    const MpcRtiConfiguration_t *model,
    const MpcTrajectorySample_t *r0, const MpcModelState_t *s0,
    const MpcTrajectorySample_t *r1, const MpcModelState_t *s1,
    int prediction_index, float *slack_m)
{
    if (!model || !model->world_sweep_check || !r0 || !r1 ||
        !s0 || !s1 || !slack_m) return 0;
    const double yaw0=r0->heading+s0->e_psi;
    const double yaw1=r1->heading+s1->e_psi;
    const double x0=r0->x-sin(r0->heading)*s0->e_y;
    const double y0=r0->y+cos(r0->heading)*s0->e_y;
    const double x1=r1->x-sin(r1->heading)*s1->e_y;
    const double y1=r1->y+cos(r1->heading)*s1->e_y;
    const double required_margin=prediction_index==1
        ? model->first_prediction_corridor_margin_m
        : model->corridor_margin_m;
    return model->world_sweep_check(model->world_sweep_context,
        x0,y0,yaw0,x1,y1,yaw1,required_margin,slack_m)==1 &&
        isfinite(*slack_m) && *slack_m>=0.0f;
}

static int world_sweep_clearance(
    const MpcRtiConfiguration_t *model,
    const MpcTrajectorySample_t *trajectory,size_t count,double lap_length,
    double p0,const MpcModelState_t *s0,
    double p1,const MpcModelState_t *s1,
    int prediction_index,float *slack_m)
{
    if (!model || !model->world_sweep_check || !s0 || !s1)
        return 0;
    MpcTrajectorySample_t r0{}, r1{};
    if (!mpc_trajectory_sample(trajectory,count,lap_length,p0,&r0) ||
        !mpc_trajectory_sample(trajectory,count,lap_length,p1,&r1))
        return 0;
    return world_sweep_clearance_from_samples(
        model,&r0,s0,&r1,s1,prediction_index,slack_m);
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

float corridor_margin_at_prediction(
    const MpcRtiConfiguration_t *configuration,
    int prediction_index)
{
    return prediction_index == 1
        ? configuration->first_prediction_corridor_margin_m
        : configuration->corridor_margin_m;
}

static MpcCorridorMargins corridor_margin_for_state(
    const MpcRtiConfiguration_t *configuration,
    const MpcModelState_t *state,
    int prediction_index)
{
    if (!configuration || !state || !isfinite(state->e_psi))
        return MpcCorridorMargins{NAN, NAN};

    /* Require the configured wall clearance plus the vehicle rectangle's
     * lateral projection about the rear axle. This keeps the full physical
     * clearance at every heading without padding the aligned car to the
     * wider planning footprint. This is a geometric feasibility rule, not a
     * change to Unity physics. */
    const float abs_cos = fabsf(cosf(state->e_psi));
    const float sine = sinf(state->e_psi);
    const float common = corridor_margin_at_prediction(configuration,
        prediction_index) + 0.5f * MPC_CAR_WIDTH_M * abs_cos;
    // The pose is at the REAR AXLE, not the rectangle centre. Project both
    // longitudinal anchors separately: using the front overhang on both
    // sides invents (front-rear)*abs(sin(epsi)) of extra vehicle on one side.
    return MpcCorridorMargins{
        common + fmaxf(MPC_REAR_AXLE_TO_FRONT_BUMPER_M * sine,
                      -MPC_REAR_OVERHANG_M * sine),
        common + fmaxf(-MPC_REAR_AXLE_TO_FRONT_BUMPER_M * sine,
                      MPC_REAR_OVERHANG_M * sine)};
}


// Differentiation of the SAME original preview-width envelope that
// reference_at_progress_with_sample uses: min of the current interpolated
// width, its +/- preview endpoints and every loaded knot in the window.
// Each candidate is affine in progress inside its current loaded segment.
// A knot strictly inside the window is constant with respect to progress.
// This is the derivative of the implemented piecewise-linear reference,
// not a fitted cubic curve or a finite-difference estimate of float widths.
static int corridor_reference_width_slopes(
    const MpcTrajectorySample_t *track,size_t count,double lap,
    double progress,const MpcRtiConfiguration_t *cfg,
    const MpcRtiReference_t *reference,
    double *left_forward,double *left_backward,
    double *right_forward,double *right_backward)
{
    if(!track || count<3 || !cfg || !reference || !left_forward ||
       !left_backward || !right_forward || !right_backward)return 0;
    struct ActiveWidth {
        float value{};
        double forward{},backward{};
    };
    const auto update=[](ActiveWidth &active,float value,
                         double forward,double backward) {
        if(value<active.value) {
            active.value=value;
            active.forward=forward;
            active.backward=backward;
        } else if(value==active.value) {
            // For f(s)=min_i f_i(s): right derivative is min of slopes;
            // left derivative is max of slopes at a tied active knot.
            active.forward=std::min(active.forward,forward);
            active.backward=std::max(active.backward,backward);
        }
    };
    MpcTrajectorySample_t center{};
    MpcTrajectoryProgressDifferential_t cs{};
    if(!mpc_trajectory_sample(track,count,lap,progress,&center) ||
       !mpc_trajectory_sample_progress_differential(
            track,count,lap,progress,&cs))return 0;
    ActiveWidth left{float(center.left_bound),
                     cs.forward.left_bound,cs.backward.left_bound};
    ActiveWidth right{float(center.right_bound),
                      cs.forward.right_bound,cs.backward.right_bound};
    const double half=cfg->corridor_preview_halfwidth_m;
    if(half>0.) {
        for(double q : {progress-half,progress+half}) {
            MpcTrajectorySample_t p{};
            MpcTrajectoryProgressDifferential_t dp{};
            if(!mpc_trajectory_sample(track,count,lap,q,&p) ||
               !mpc_trajectory_sample_progress_differential(
                    track,count,lap,q,&dp))return 0;
            update(left,float(p.left_bound),
                   dp.forward.left_bound,dp.backward.left_bound);
            update(right,float(p.right_bound),
                   dp.forward.right_bound,dp.backward.right_bound);
        }
        double wrapped=fmod(progress-track[0].s,lap);
        if(wrapped<0.)wrapped+=lap;
        wrapped+=track[0].s;
        const double lap_end=track[0].s+lap;
        const double first=track[0].s;
        const double window_start=wrapped-half;
        const double window_end=wrapped+half;
        bool ambiguous_window_edge=false;
        const auto knots=[&](double lo,double hi) {
            size_t first_index=0,last_index=count;
            while(first_index<last_index) {
                const size_t mid=first_index+(last_index-first_index)/2;
                if(track[mid].s<lo)first_index=mid+1;
                else last_index=mid;
            }
            for(size_t i=first_index;i<count && track[i].s<=hi;++i) {
                const double knot_s=track[i].s;
                // At the EXACT event where an interior candidate enters
                // or exits preview, the active constraint set changes;
                // refuse a falsely smooth Jacobian at that event.
                const double edge_tolerance=1.e-11*std::max(1.,lap);
                // The periodic lap seam is an artificial split of the
                // preview window, NOT a moving preview endpoint.
                const bool moving_lower=
                    std::abs(lo-window_start)<=edge_tolerance ||
                    std::abs(lo-(window_start+lap))<=edge_tolerance;
                const bool moving_upper=
                    std::abs(hi-window_end)<=edge_tolerance ||
                    std::abs(hi-(window_end-lap))<=edge_tolerance;
                if((moving_lower &&
                    std::abs(knot_s-lo)<=edge_tolerance) ||
                   (moving_upper &&
                    std::abs(knot_s-hi)<=edge_tolerance))
                    ambiguous_window_edge=true;
                update(left,float(track[i].left_bound),0.,0.);
                update(right,float(track[i].right_bound),0.,0.);
            }
        };
        if(window_start<first) {
            knots(window_start+lap,lap_end);
            knots(first,window_end);
        } else if(window_end>=lap_end) {
            knots(window_start,lap_end);
            knots(first,window_end-lap);
        } else {
            knots(window_start,window_end);
        }
        if(ambiguous_window_edge)return 0;
    }
    // Exact float min parity with the source scorer is mandatory.
    if(left.value!=reference->left_bound ||
       right.value!=reference->right_bound)return 0;
    *left_forward=left.forward;*left_backward=left.backward;
    *right_forward=right.forward;*right_backward=right.backward;
    return std::isfinite(*left_forward)&&std::isfinite(*left_backward)&&
           std::isfinite(*right_forward)&&std::isfinite(*right_backward);
}

int mpc_rti_corridor_differential_at_progress(
    const MpcModelState_t *state,
    const MpcTrajectorySample_t *trajectory,
    size_t trajectory_count,double lap_length,double progress_m,
    const MpcRtiConfiguration_t *configuration,int prediction_index,
    MpcRtiCorridorDifferential_t *out)
{
    if(!state || !trajectory || !out || !configuration ||
       !std::isfinite(progress_m) || !std::isfinite(lap_length) ||
       lap_length<=0. || prediction_index<1 ||
       prediction_index>PREDICTION_HORIZON ||
       !std::isfinite(state->e_y) ||
       !std::isfinite(state->e_psi))
        return 0;

    // The original corridor is a physical rear-axle rectangle:
    //   lower = e_y + right_bound + tolerance - margin_right(e_psi)
    //   upper = left_bound + tolerance - e_y - margin_left(e_psi).
    // The active front/rear bumper changes at sin(e_psi)=0, while
    // the lateral width has an |cos(e_psi)| switch at cos=0.
    // Never pretend a classical derivative exists at those corners.
    const double psi=state->e_psi;
    const double sine=std::sin(psi),cosine=std::cos(psi);
    if(std::abs(sine)<1.e-6 || std::abs(cosine)<1.e-6)
        return 0;
    MpcRtiReference_t center{};
    if(!mpc_rti_reference_at_progress(trajectory,trajectory_count,
          lap_length,progress_m,configuration,&center))
        return 0;
    MpcRtiCorridorDifferential_t result{};
    float lower=NAN,upper=NAN;
    if(!mpc_rti_compute_corridor_slack(
          state,center.left_bound,center.right_bound,
          configuration,prediction_index,&lower,&upper))
        return 0;
    result.margin_lower_m=lower;
    result.margin_upper_m=upper;
    result.lower_d_ey=1.;
    result.upper_d_ey=-1.;
    const double half_width=.5*double(MPC_CAR_WIDTH_M);
    const double dcommon=-half_width*sine*(cosine>0.?1.:-1.);
    const double front=MPC_REAR_AXLE_TO_FRONT_BUMPER_M;
    const double rear=MPC_REAR_OVERHANG_M;
    const double dleft=(front*sine > -rear*sine)
        ?front*cosine:-rear*cosine;
    const double dright=(-front*sine > rear*sine)
        ?-front*cosine:rear*cosine;
    result.lower_d_epsi=-dcommon-dright;
    result.upper_d_epsi=-dcommon-dleft;

    // Exact one-sided derivatives of the ORIGINAL loaded reference and
    // preview-width min: the interpolation segment, knot extrema, heading
    // footprint and physical source progress must be the same as the
    // original nonlinear scorer. No arbitrary FD step or spline fitting.
    if(!corridor_reference_width_slopes(
          trajectory,trajectory_count,lap_length,progress_m,configuration,
          &center,&result.upper_d_progress_forward,
          &result.upper_d_progress_backward,
          &result.lower_d_progress_forward,
          &result.lower_d_progress_backward))
        return 0;
    result.progress_derivative_step_m=0.;
    result.progress_slope_convergence_error=0.;
    // Do not classify a true loaded-reference knot as smooth merely
    // because the two valid segment slopes happen to differ by <0.05.
    // This flag describes differentiability of the ORIGINAL interpolation,
    // not an arbitrary engineering threshold for a large cost change.
    const double lower_jump=std::abs(result.lower_d_progress_forward-
                                      result.lower_d_progress_backward);
    const double upper_jump=std::abs(result.upper_d_progress_forward-
                                      result.upper_d_progress_backward);
    const double lower_scale=std::max({1.,
        std::abs(result.lower_d_progress_forward),
        std::abs(result.lower_d_progress_backward)});
    const double upper_scale=std::max({1.,
        std::abs(result.upper_d_progress_forward),
        std::abs(result.upper_d_progress_backward)});
    result.progress_reference_kink=
        lower_jump>1.e-10*lower_scale ||
        upper_jump>1.e-10*upper_scale;
    result.source_heading_active_branch=sine>0.?1:-1;
    if(!std::isfinite(result.lower_d_epsi) ||
       !std::isfinite(result.upper_d_epsi) ||
       !std::isfinite(result.lower_d_progress_forward) ||
       !std::isfinite(result.upper_d_progress_forward) ||
       !std::isfinite(result.lower_d_progress_backward) ||
       !std::isfinite(result.upper_d_progress_backward))
        return 0;
    *out=result;
    return 1;
}

int mpc_rti_source_exact_corridor_corners(
    const MpcModelState_t *state,
    const MpcTrajectorySample_t *trajectory,size_t trajectory_count,
    double lap_length,double progress_m,
    const MpcRtiConfiguration_t *configuration,int prediction_index,
    MpcRtiSourceCorridorCorners_t *out)
{
    if(!state||!trajectory||!configuration||!out||
       prediction_index<1||prediction_index>PREDICTION_HORIZON||
       !std::isfinite(state->e_y)||!std::isfinite(state->e_psi)||
       !std::isfinite(progress_m))return 0;
    MpcRtiReference_t ref{};
    if(!mpc_rti_reference_at_progress(trajectory,trajectory_count,
            lap_length,progress_m,configuration,&ref))return 0;
    float source_right=NAN,source_left=NAN;
    if(!mpc_rti_compute_corridor_slack(state,ref.left_bound,
            ref.right_bound,configuration,prediction_index,
            &source_right,&source_left))return 0;
    MpcRtiSourceCorridorCorners_t r{};
    const double psi=state->e_psi;
    const double sine=std::sin(psi);
    const double cosine=std::cos(psi);
    const double x[2]={-double(MPC_REAR_OVERHANG_M),
                        double(MPC_REAR_AXLE_TO_FRONT_BUMPER_M)};
    const double y[2]={-.5*double(MPC_CAR_WIDTH_M),
                         .5*double(MPC_CAR_WIDTH_M)};
    double lat[4]{},derivative[4]{};
    double minimum=INFINITY,maximum=-INFINITY;
    for(int along=0;along<2;++along)
        for(int across=0;across<2;++across) {
            const int index=2*along+across;
            lat[index]=x[along]*sine+y[across]*cosine;
            derivative[index]=x[along]*cosine-y[across]*sine;
            minimum=std::min(minimum,lat[index]);
            maximum=std::max(maximum,lat[index]);
        }
    // Express all eight original constraints relative to their ORIGINAL
    // float-evaluated source min-slacks. This preserves source-parity at
    // the active support corner without subtracting two differently
    // rounded trigonometric approximations.
    for(int j=0;j<4;++j) {
        r.left[j]=double(source_left)+(maximum-lat[j]);
        r.right[j]=double(source_right)+(lat[j]-minimum);
        r.left_d_psi[j]=-derivative[j];
        r.right_d_psi[j]=derivative[j];
    }
    // The original preview-reference minimum is piecewise affine and may
    // change at a loaded knot. Keep BOTH ORIGINAL directional slopes.
    if(corridor_reference_width_slopes(
            trajectory,trajectory_count,lap_length,progress_m,
            configuration,&ref,
            &r.left_d_progress_forward,
            &r.left_d_progress_backward,
            &r.right_d_progress_forward,
            &r.right_d_progress_backward)) {
        r.reference_derivatives_valid=1;
        const double ls=std::max({1.,
            std::abs(r.left_d_progress_forward),
            std::abs(r.left_d_progress_backward)});
        const double rs=std::max({1.,
            std::abs(r.right_d_progress_forward),
            std::abs(r.right_d_progress_backward)});
        r.reference_progress_kink=
            (std::abs(r.left_d_progress_forward-
                      r.left_d_progress_backward)>1.e-10*ls)||
            (std::abs(r.right_d_progress_forward-
                      r.right_d_progress_backward)>1.e-10*rs);
    }
    *out=r;
    return 1;
}

int mpc_rti_compute_corridor_slack(
    const MpcModelState_t *state,
    float left_bound_m,
    float right_bound_m,
    const MpcRtiConfiguration_t *configuration,
    int prediction_index,
    float *lower_slack_m,
    float *upper_slack_m)
{
    if (!state || !configuration || !lower_slack_m || !upper_slack_m ||
        prediction_index < 0 || prediction_index > PREDICTION_HORIZON ||
        !isfinite(left_bound_m) || !isfinite(right_bound_m) ||
        !isfinite(configuration->nonlinear_corridor_tolerance_m)) return 0;
    const MpcCorridorMargins margin = corridor_margin_for_state(configuration, state,
        prediction_index);
    const float lower = margin.right - right_bound_m -
        configuration->nonlinear_corridor_tolerance_m;
    const float upper = left_bound_m - margin.left +
        configuration->nonlinear_corridor_tolerance_m;
    if (!isfinite(lower) || !isfinite(upper) ||
        lower > upper) return 0;
    *lower_slack_m = state->e_y - lower;
    *upper_slack_m = upper - state->e_y;
    return isfinite(*lower_slack_m) && isfinite(*upper_slack_m);
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
    if (!nominal || !nominal->valid || nominal->horizon < 1 ||
        nominal->horizon > horizon)
        return 0;
    for (int k = 0; k <= nominal->horizon; ++k) {
        if (!finite_rti_state(&nominal->states[k]) ||
            !isfinite(nominal->progress[k])) return 0;
    }
    for (int k = 0; k < nominal->horizon; ++k)
        if (!isfinite(nominal->controls[k].steering_target_rad) ||
            !isfinite(nominal->controls[k].throttle_command)) return 0;
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

static int seed_unsolved_tail_control(
    const MpcTrajectorySample_t *trajectory, size_t count, double length,
    double station, double next_station, float last_throttle,
    const MpcRtiConfiguration_t *configuration, MpcModelControl_t *control)
{
    MpcTrajectorySample_t sample;
    if (!mpc_trajectory_sample(trajectory, count, length, station, &sample))
        return 0;
    if (sample.source_witness_valid &&
        sample.speed <= configuration->active_speed_ceiling_mps + 1.e-4f) {
        control->steering_target_rad = clampf_rti((float)sample.source_steering_target,
            -configuration->max_steering_rad, configuration->max_steering_rad);
        control->throttle_command = clampf_rti((float)sample.source_throttle,
            MPC_THROTTLE_MIN_DRIVE, MPC_THROTTLE_MAX);
        return 1;
    }
    if (!mpc_trajectory_sample(trajectory, count, length, next_station, &sample)) return 0;
    control->steering_target_rad = clampf_rti(
        vehicle_model_steering_for_curvature((float)sample.curvature),
        -configuration->max_steering_rad, configuration->max_steering_rad);
    control->throttle_command = clampf_rti(last_throttle,
        MPC_THROTTLE_MIN_DRIVE, MPC_THROTTLE_MAX);
    return 1;
}

static int mpc_rti_build_nominal_impl(
    const MpcRtiState_t *current_state,
    double current_progress,
    const MpcRtiNominal_t *previous_nominal,
    const MpcModelControl_t *warm_start_controls,
    int warm_start_control_count,
    int stage_shift,
    int hold_unsolved_warm_tail,
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
    // No whole-horizon copy on the normal path: the previous iterate lives in
    // persistent memory, independently of the output. Preserve alias safety
    // only for a caller explicitly building into its own previous nominal.
    MpcRtiNominal_t aliased_previous;
    if (previous_nominal == nominal) {
        aliased_previous = *previous_nominal;
        previous_nominal = &aliased_previous;
    }
    *nominal = {};
    nominal->build_failure_index = -1;
    if (!finite_rti_state(current_state) || !isfinite(current_progress) ||
        !trajectory || !references || horizon < 1 ||
        horizon > PREDICTION_HORIZON || !isfinite(prediction_dt) ||
        prediction_dt <= 0.0f || !valid_configuration(configuration) ||
        ((warm_start_controls == NULL) != (warm_start_control_count == 0)) ||
        warm_start_control_count < 0 ||
        warm_start_control_count > PREDICTION_HORIZON ||
        (warm_start_controls && warm_start_control_count < 1))
        return fail_nominal_build(
            nominal, MPC_RTI_NOMINAL_BUILD_INVALID_INPUT);
    for (int k = 0; warm_start_controls &&
            k < warm_start_control_count; ++k) {
        if (!isfinite(warm_start_controls[k].steering_target_rad) ||
            !isfinite(warm_start_controls[k].throttle_command))
            return fail_nominal_build(
                nominal, MPC_RTI_NOMINAL_BUILD_INVALID_INPUT);
    }
    if (!vehicle_model_set_active_target_speed_ceiling(
            configuration->active_speed_ceiling_mps))
        return fail_nominal_build(
            nominal, MPC_RTI_NOMINAL_BUILD_SPEED_CEILING);

    const int warm = valid_previous_nominal(previous_nominal, horizon);
    nominal->horizon = horizon;
    nominal->states[0] = *current_state;
    /* x0 is measured and cannot be repaired by optimization. A finite future
     * warm-start rollout, however, is only a linearization point: imposing
     * candidate bounds here can abort before the QP can repair its controls.
     * The QP and exact nonlinear acceptance still enforce future bounds. */
    const MpcRtiCommandEnvelopeFailure_t initial_failure =
        command_envelope_failure(current_state, configuration);
    if (initial_failure != MPC_RTI_COMMAND_ENVELOPE_OK) {
        nominal->build_failure_constraint = initial_failure;
        nominal->build_failure_index = 0;
        return fail_nominal_build(
            nominal, MPC_RTI_NOMINAL_BUILD_COMMAND_ENVELOPE);
    }

    double progress_seed[PREDICTION_HORIZON + 1] = {};
    if (warm) {
        const int previous_horizon = previous_nominal->horizon;
        progress_seed[0] = current_progress;
        const int shift = stage_shift;
        const double previous_origin = previous_nominal->progress[shift];
        const double previous_last_increment =
            previous_nominal->progress[previous_horizon] -
            previous_nominal->progress[previous_horizon - 1];
        for (int k = 1; k < horizon; ++k) {
            const int previous_stage = k + shift;
            const double previous_progress = previous_stage <= previous_horizon
                ? previous_nominal->progress[previous_stage]
                : previous_nominal->progress[previous_horizon] +
                    (previous_stage - previous_horizon) *
                        previous_last_increment;
            progress_seed[k] = current_progress +
                previous_progress - previous_origin;
        }
        const int terminal_stage = horizon + shift;
        const double previous_terminal_progress =
            terminal_stage <= previous_horizon
                ? previous_nominal->progress[terminal_stage]
                : previous_nominal->progress[previous_horizon] +
                    (terminal_stage - previous_horizon) *
                        previous_last_increment;
        progress_seed[horizon] = current_progress +
            previous_terminal_progress - previous_origin;
        for (int k = 0; k < horizon; ++k) {
            const int shifted_index = k + shift < previous_horizon
                ? k + shift : previous_horizon - 1;
            nominal->controls[k] = previous_nominal->controls[shifted_index];
            nominal->controls[k].steering_target_rad = clampf_rti(
                nominal->controls[k].steering_target_rad,
                -configuration->max_steering_rad,
                configuration->max_steering_rad);
            nominal->controls[k].throttle_command = clampf_rti(
                nominal->controls[k].throttle_command,
                MPC_THROTTLE_MIN_DRIVE,
                MPC_THROTTLE_MAX);
            if (k + shift >= previous_horizon &&
                !seed_unsolved_tail_control(trajectory, trajectory_count,
                    lap_length, progress_seed[k], progress_seed[k + 1],
                    nominal->controls[k].throttle_command, configuration,
                    &nominal->controls[k]))
                return fail_nominal_build(
                    nominal, MPC_RTI_NOMINAL_BUILD_SEED_REFERENCE);
        }
    } else {
        double phase_start_us = monotonic_microseconds();
        TrajectoryReferencePoint_t seed_references[PREDICTION_HORIZON + 1];
        if (!mpc_reference_build_speed_seed(
                trajectory, trajectory_count, lap_length, current_progress,
                configuration->active_speed_ceiling_mps, prediction_dt,
                horizon, seed_references, progress_seed))
            return fail_nominal_build(
                nominal, MPC_RTI_NOMINAL_BUILD_SPEED_SEED);
        for (int k = 0; k <= horizon; ++k)
            if (!mpc_rti_reference_at_progress(trajectory, trajectory_count,
                    lap_length, progress_seed[k], configuration,
                    &references[k]))
                return fail_nominal_build(
                    nominal, MPC_RTI_NOMINAL_BUILD_SEED_REFERENCE);
        nominal->reference_seed_us +=
            monotonic_microseconds() - phase_start_us;

        /* Seed throttle deterministically from the last accepted value; the
         * optimized input is applied directly at each command-time stage.
         * Avoid plant transitions solely to construct this seed. */
        phase_start_us = monotonic_microseconds();
        float queued_throttle = current_state->plant.delayed_throttle_command_2;
        for (int k = 0; k < horizon; ++k) {
            const float desired_delta = references[k].source_witness_valid ?
                references[k].source_steering_target : references[k + 1].steering_command;
            nominal->controls[k].steering_target_rad = clampf_rti(
                desired_delta,
                -configuration->max_steering_rad,
                configuration->max_steering_rad);
            nominal->controls[k].throttle_command = clampf_rti(
                references[k].source_witness_valid ? references[k].source_throttle :
                    fmaxf(queued_throttle, MPC_THROTTLE_MIN_DRIVE),
                MPC_THROTTLE_MIN_DRIVE,
                MPC_THROTTLE_MAX);
            /* Carry the seed value forward as the next stage's initial guess;
             * the nonlinear plant consumes each stage's direct input. */
            queued_throttle = nominal->controls[k].throttle_command;
        }
        nominal->cold_start_queue_seed_us =
            monotonic_microseconds() - phase_start_us;
    }

    /* A shifted nominal can retain minimum-drive throttle after a rejected
     * or heavily repaired cycle, leaving the car effectively stopped even
     * though the trajectory still requests acceleration. Re-seed both cold
     * and warm nominals from rest; this is only a linearization seed. ADMM
     * may change it, and the full nonlinear rollout/corridor validation
     * remains the only path to a published command. */
    MpcRtiReference_t launch_reference = {};
    if (current_state->plant.u < 0.5f &&
        configuration->startup_nominal_throttle_seed >
            MPC_THROTTLE_MIN &&
        mpc_rti_reference_at_progress(trajectory, trajectory_count, lap_length,
            progress_seed[1], configuration, &launch_reference) &&
        launch_reference.u > current_state->plant.u + 1.0f) {
        for (int k = 0; k < horizon; ++k)
            nominal->controls[k].throttle_command =
                configuration->startup_nominal_throttle_seed;
    }

    /* This seed represents the commands in the actuator timeline, not merely
     * the previous converged iterate. Index zero is already in flight; shift
     * it out so the next optimization starts with the action that follows it.
     * Explicit issued-command history takes precedence over launch heuristics. */
    if (warm_start_controls) {
        for (int k = 0; k < horizon; ++k) {
            const int requested_index = k + stage_shift;
            if (requested_index < warm_start_control_count) {
                nominal->controls[k] = warm_start_controls[requested_index];
            } else if(hold_unsolved_warm_tail) {
                // The final ten stages of reduced source-SQP are fixed by
                // construction. Do NOT synthesize a discontinuous witness
                // command inside those frozen variables: the QP cannot fix
                // the artificial actuator-rate violation, although the
                // previous strict-source-validated plan remains feasible.
                // Hold the actual last previously issued command instead.
                nominal->controls[k]=k>0?nominal->controls[k-1]:
                    warm_start_controls[warm_start_control_count-1];
            } else if (!seed_unsolved_tail_control(trajectory, trajectory_count,
                    lap_length, progress_seed[k], progress_seed[k + 1],
                    warm_start_controls[warm_start_control_count - 1].
                        throttle_command,
                    configuration, &nominal->controls[k])) {
                return fail_nominal_build(
                    nominal, MPC_RTI_NOMINAL_BUILD_SEED_REFERENCE);
            }
            nominal->controls[k].steering_target_rad = clampf_rti(
                nominal->controls[k].steering_target_rad,
                -configuration->max_steering_rad,
                configuration->max_steering_rad);
            nominal->controls[k].throttle_command = clampf_rti(
                nominal->controls[k].throttle_command,
                MPC_THROTTLE_MIN_DRIVE,
                MPC_THROTTLE_MAX);
        }
    }

    for (int k = 0; k < horizon; ++k)
        nominal->controls[k].throttle_command = clampf_rti(
            nominal->controls[k].throttle_command,
            MPC_THROTTLE_MIN_DRIVE,
            MPC_THROTTLE_MAX);

    // The normal delta-input SQP must start from a physically rate-feasible
    // command trajectory. Otherwise the nonlinear line search mixes an
    // unbounded warm/geometry nominal with a rate-limited QP solution, and
    // the accepted trajectory can still contain full-lock command reversals.
    // Project only the NOMINAL command seed, then evaluate it with the SAME
    // exact nonlinear plant as the candidate. No extra rollout or fallback.
    if (configuration->command_increment_qp_enabled) {
        float prev_s=configuration->source_reachable_steering_qp
            ? current_state->plant.actual_steering_angle
            : current_state->plant.delayed_steering_command_2;
        float prev_t=current_state->plant.delayed_throttle_command_2;
        const float steering_limit=mpc_source_nominal_steering_step_limit(
            configuration,prediction_dt);
        for (int k=0;k<horizon;++k) {
            auto &u=nominal->controls[k];
            u.steering_target_rad=clampf_rti(
                u.steering_target_rad,
                fmaxf(-configuration->max_steering_rad,
                      prev_s-steering_limit),
                fminf(configuration->max_steering_rad,
                      prev_s+steering_limit));
            u.throttle_command=clampf_rti(
                u.throttle_command,
                fmaxf(MPC_THROTTLE_MIN_DRIVE,
                      prev_t-configuration->max_throttle_command_step),
                fminf(MPC_THROTTLE_MAX,
                      prev_t+configuration->max_throttle_command_step));
            prev_s=u.steering_target_rad;
            prev_t=u.throttle_command;
        }
    }

    /* Physical motion no longer depends on an approximate curvature pass.
     * One exact loaded-frame rollout gives both the nominal states and their
     * final progress, so the old reference/second-rollout pass is redundant. */
    double phase_start_us = monotonic_microseconds();
    if (!rollout_nominal_controls(current_state, current_progress,
            nominal->controls,
            trajectory, trajectory_count, lap_length, horizon, prediction_dt,
            nominal->states, nominal->progress, &nominal->build_failure_index))
        return fail_nominal_build(
            nominal, MPC_RTI_NOMINAL_BUILD_PASS_A_ROLLOUT);
    nominal->nominal_rollout_a_us = monotonic_microseconds() - phase_start_us;

    phase_start_us = monotonic_microseconds();
    for (int k = 0; k <= horizon; ++k) {
        if (!mpc_rti_reference_at_progress(trajectory, trajectory_count, lap_length,
                nominal->progress[k], configuration, &references[k]))
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
    nominal->progress_reference_resampling_us +=
        monotonic_microseconds() - phase_start_us;
    for (int k = 0; k <= horizon; ++k)
        nominal->speed_reference[k] = references[k].u;
    nominal->speed_reference_valid = 1;

    nominal->valid = 1;
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
    return mpc_rti_build_nominal_impl(current_state, current_progress,
        previous_nominal, NULL, 0, 1, 0, trajectory, trajectory_count, lap_length,
        prediction_dt, horizon, configuration, nominal, references);
}

int mpc_rti_build_nominal_with_warm_start(
    const MpcRtiState_t *current_state,
    double current_progress,
    const MpcRtiNominal_t *previous_nominal,
    const MpcModelControl_t *warm_start_controls,
    int warm_start_control_count,
    const MpcTrajectorySample_t *trajectory,
    size_t trajectory_count,
    double lap_length,
    float prediction_dt,
    int horizon,
    const MpcRtiConfiguration_t *configuration,
    MpcRtiNominal_t *nominal,
    MpcRtiReference_t references[PREDICTION_HORIZON + 1])
{
    return mpc_rti_build_nominal_impl(current_state, current_progress,
        previous_nominal, warm_start_controls, warm_start_control_count, 1, 0,
        trajectory, trajectory_count, lap_length, prediction_dt, horizon,
        configuration, nominal, references);
}

int mpc_rti_build_nominal_same_epoch(
    const MpcRtiState_t *current_state,
    double current_progress,
    const MpcRtiNominal_t *previous_nominal,
    const MpcModelControl_t *same_epoch_controls,
    int same_epoch_control_count,
    const MpcTrajectorySample_t *trajectory,
    size_t trajectory_count,
    double lap_length,
    float prediction_dt,
    int horizon,
    const MpcRtiConfiguration_t *configuration,
    MpcRtiNominal_t *nominal,
    MpcRtiReference_t references[PREDICTION_HORIZON + 1])
{
    return mpc_rti_build_nominal_impl(current_state,current_progress,
        previous_nominal,same_epoch_controls,same_epoch_control_count,0,0,
        trajectory,trajectory_count,lap_length,prediction_dt,horizon,
        configuration,nominal,references);
}

static int diagnostic_sample_index_for_stage(int stage)
{
    static const int sample_stages[MPC_RTI_DIAGNOSTIC_SAMPLE_COUNT] = {
        1, 2, 3, 5, 10, 20, 30};
    for (int i = 0; i < MPC_RTI_DIAGNOSTIC_SAMPLE_COUNT; ++i)
        if (sample_stages[i] == stage) return i;
    return -1;
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
    MpcRtiDiagnosticStageOutput_t diagnostic_outputs[
        MPC_RTI_DIAGNOSTIC_SAMPLE_COUNT],
    int *diagnostic_output_count,
    int *failure_stage,
    MpcRtiRolloutFailure_t *failure,
    MpcModelControl_t evaluated_controls[PREDICTION_HORIZON])
{
    if (failure_stage) *failure_stage = -1;
    if (failure) memset(failure, 0, sizeof(*failure));
    if (path_delta) {
        memset(path_delta, 0, sizeof(*path_delta));
        /* The exact rollout owns these per-stage physical clearance
         * measurements regardless of whether heavy debug output is enabled.
         * Scoring may reuse them, but only immediately for this validated
         * unchanged candidate. */
        {
            for (int k = 0; k <= PREDICTION_HORIZON; ++k) {
                path_delta->lower_corridor_slack_m[k] = NAN;
                path_delta->upper_corridor_slack_m[k] = NAN;
            }
        }
    }
    if (diagnostic_output_count) *diagnostic_output_count = 0;
    if (diagnostic_outputs)
        memset(diagnostic_outputs, 0,
            MPC_RTI_DIAGNOSTIC_SAMPLE_COUNT * sizeof(diagnostic_outputs[0]));
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
    MpcRtiReference_t candidate_reference = {};
    MpcTrajectorySample_t current_world_sample = {};
    if (!reference_at_progress_with_sample(
            trajectory, trajectory_count, lap_length,
            candidate_progress[0], configuration, &candidate_reference,
            configuration->world_sweep_check ? &current_world_sample : NULL))
        return MPC_RTI_ROLLOUT_INVALID_INPUT;
    if (nominal_references)
        candidate_reference.u = nominal_references[0].u;
    if (path_delta) {
        path_delta->sampled_reference[0] = candidate_reference;
        path_delta->sampled_progress[0] = candidate_progress[0];
    }
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
        MpcModelControl_t adjusted = controls[k];
        if (evaluated_controls) evaluated_controls[k] = adjusted;
        const MpcModelControl_t *control = &adjusted;
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
        /* The candidate reference at stage k was sampled at the end of
         * stage k-1 (or before the loop for k=0), at the same exact
         * candidate_progress[k]. Re-interpolating it here performs one
         * redundant trajectory lookup for every source-plant step of every
         * exact nonlinear validation/backtracking rollout. Its speed override
         * was also applied at that boundary. Carry the sampled reference
         * forward unchanged; hard corridor checks still sample each newly
         * reached state below and are NEVER skipped. */
        MpcStageResult_t step = mpc_vehicle_model_step(
            &states[k].plant, control, prediction_dt,
            0.0f);
        if (!step.valid) {
            if (failure_stage) *failure_stage = k;
            return MPC_RTI_ROLLOUT_INVALID_MODEL;
        }
        double next_progress;
        MpcTrajectorySample_t projected_sample{};
        if (!mpc_trajectory_advance_tracking_with_progress_sampled(
                trajectory, trajectory_count,
                lap_length, candidate_progress[k], states[k].plant.e_y,
                states[k].plant.e_psi, step.body_forward_delta_m,
                step.body_lateral_delta_m, step.body_yaw_delta_rad,
                candidate_progress[k] + step.delta_s_m, &next_progress,
                &step.next.e_y, &step.next.e_psi,
                states[k + 1].incoming_tracking_jacobian, &projected_sample)) {
            if (failure_stage) *failure_stage = k;
            return MPC_RTI_ROLLOUT_INVALID_MODEL;
        }
        states[k + 1].incoming_tracking_jacobian_valid = 1;
        if (diagnostic_outputs) {
            const int sample_index = diagnostic_sample_index_for_stage(k + 1);
            if (sample_index >= 0) {
                diagnostic_outputs[sample_index].stage = k + 1;
                diagnostic_outputs[sample_index].valid = 1;
                diagnostic_outputs[sample_index].output = step.output;
                if (diagnostic_output_count)
                    *diagnostic_output_count = sample_index + 1;
            }
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
        candidate_progress[k + 1] = next_progress;
        if (progress) progress[k + 1] = candidate_progress[k + 1];
        if (!state_inside_command_envelope(&states[k + 1], configuration)) {
            if (failure_stage) *failure_stage = k;
            return MPC_RTI_ROLLOUT_STATE_LIMIT;
        }
        MpcTrajectorySample_t next_world_sample = {};
        if (!reference_at_progress_with_sample(
                trajectory, trajectory_count, lap_length,
                candidate_progress[k + 1], configuration, &candidate_reference,
                configuration->world_sweep_check ? &next_world_sample : NULL,
                &projected_sample)) {
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
            path_delta->sampled_reference[k + 1] = candidate_reference;
            path_delta->sampled_progress[k + 1] = candidate_progress[k + 1];
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
        const MpcCorridorMargins margin = corridor_margin_for_state(configuration,
            &states[k + 1].plant, k + 1);
        const float lower_bound = margin.right - candidate_reference.right_bound -
            configuration->nonlinear_corridor_tolerance_m;
        const float upper_bound = candidate_reference.left_bound - margin.left +
            configuration->nonlinear_corridor_tolerance_m;
        if (path_delta) {
            path_delta->lower_corridor_slack_m[k + 1] =
                states[k + 1].plant.e_y - lower_bound;
            path_delta->upper_corridor_slack_m[k + 1] =
                upper_bound - states[k + 1].plant.e_y;
        }
        int in_corridor = isfinite(lower_bound) && isfinite(upper_bound) &&
            lower_bound <= upper_bound &&
            states[k + 1].plant.e_y >= lower_bound &&
            states[k + 1].plant.e_y <= upper_bound;
        if (configuration->world_sweep_check &&
            (!configuration->world_map_rejection_only || !in_corridor)) {
            float world_slack=-INFINITY;
            in_corridor = world_sweep_clearance_from_samples(
                configuration,&current_world_sample,&states[k].plant,
                &next_world_sample,&states[k+1].plant,
                k+1,&world_slack);
            if (path_delta)
                path_delta->lower_corridor_slack_m[k+1]=
                    path_delta->upper_corridor_slack_m[k+1]=world_slack;
        }
        if (!in_corridor) {
            if (failure_stage) *failure_stage = k;
            record_rollout_failure(failure, k, &states[k + 1],
                &candidate_reference, candidate_progress[k + 1],
                fmaxf(margin.left, margin.right), lower_bound, upper_bound);
            return MPC_RTI_ROLLOUT_CORRIDOR;
        }
        /* Preserve the exact interpolated geometry for the next movement.
         * This copy is small and occurs once per source-plant step. */
        if (configuration->world_sweep_check)
            current_world_sample = next_world_sample;
    }
    if (path_delta) path_delta->sampled_references_valid = 1;
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
        progress, NULL, NULL, failure_stage, NULL, NULL);
}

void mpc_rti_memory_reset(MpcRtiMemory_t *memory)
{
    if (!memory) return;
    memset(memory, 0, sizeof(*memory));
}

int mpc_rti_memory_set_warm_start_controls(
    MpcRtiMemory_t *memory,
    const MpcModelControl_t *controls,
    int control_count)
{
    if (!memory || !controls || control_count < 1 ||
        control_count > PREDICTION_HORIZON) return 0;
    for (int k = 0; k < control_count; ++k) {
        if (!isfinite(controls[k].steering_target_rad) ||
            !isfinite(controls[k].throttle_command)) return 0;
    }
    memcpy(memory->warm_start_controls, controls,
        (size_t)control_count * sizeof(memory->warm_start_controls[0]));
    memory->warm_start_control_count = control_count;
    memory->warm_start_controls_valid = 1;
    memory->command_epoch_ns = 0;
    return 1;
}

int mpc_rti_memory_set_warm_start_controls_at_epoch(
    MpcRtiMemory_t *memory,const MpcModelControl_t *controls,
    int control_count,int64_t command_epoch_ns)
{
    if(command_epoch_ns<0 || !mpc_rti_memory_set_warm_start_controls(
            memory,controls,control_count)) return 0;
    memory->command_epoch_ns=command_epoch_ns;
    return 1;
}

int mpc_rti_command_epoch_shift(int64_t previous_ns,int64_t current_ns,
    float prediction_dt)
{
    if(previous_ns<=0 || current_ns<=0 ||
       !isfinite(prediction_dt) || prediction_dt<=0.f) return -1;
    const double stages=(double)(current_ns-previous_ns)/
        ((double)prediction_dt*1.e9);
    // Forecasts of the same future Bridge request can move backward by
    // sub-ms host jitter. Quantize the signed difference to a control stage;
    // a revision inside the same stage must not suppress a fresh-state solve.
    if(stages<=-.5 || stages>=PREDICTION_HORIZON-.5) return -1;
    return std::max(0,(int)llround(stages));
}

static int valid_cycle_configuration(
    const MpcRtiCycleConfiguration_t *cfg)
{
    if(!cfg || cfg->command_epoch_ns<0 ||
       !valid_configuration(&cfg->model) ||
       !cfg->source_exact_nlp_live ||
       !cfg->source_exact_nlp_constrained_qp ||
       !cfg->source_exact_nlp_full_step_sqp ||
       cfg->source_exact_nlp_iterations<1 ||
       cfg->source_exact_nlp_iterations>8 ||
       !isfinite(cfg->source_exact_nlp_qp_regularization) ||
       cfg->source_exact_nlp_qp_regularization<=0.f ||
       cfg->source_exact_nlp_qp_regularization>100000.f ||
       !isfinite(cfg->source_exact_nlp_qp_soft_weight) ||
       cfg->source_exact_nlp_qp_soft_weight<=0.f ||
       cfg->source_exact_nlp_qp_soft_weight>1.e9f ||
       !isfinite(cfg->source_exact_nlp_extra_clearance_m) ||
       cfg->source_exact_nlp_extra_clearance_m<0.f ||
       cfg->source_exact_nlp_extra_clearance_m>.1f ||
       !finite_nonnegative(cfg->max_cycle_time_us) ||
       !finite_nonnegative(cfg->absolute_deadline_monotonic_us))
        return 0;
    return 1;
}

double monotonic_microseconds(void)
{
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) return 0.0;
    return (double)now.tv_sec * 1.0e6 + (double)now.tv_nsec * 1.0e-3;
}

double mpc_rti_monotonic_time_us(void)
{
    return monotonic_microseconds();
}

/* The production cycle still has an absolute 20 ms publication deadline.
 * Reserve 3 ms for exact nonlinear validation, retained-plan checks, and
 * command publication. This margin is retained until a paired replay/live
 * test demonstrates that a smaller reserve improves solve quality without
 * reducing deadline reliability. */
static constexpr double MPC_RTI_POST_SOLVE_RESERVE_US = 3000.0;
/* Historical live exact rollouts stayed below 2.675 ms. Keep one 3.5 ms
 * allowance per optional trial, plus the ordinary post-solve reserve. */
static constexpr double MPC_RTI_LINE_SEARCH_TRIAL_RESERVE_US = 3500.0;
/* Refinement must reserve enough for one exact nonlinear backtrack in
 * addition to the ordinary post-solve/publication reserve. Without this,
 * a time-limited second QP can consume its budget and reach line search with
 * less than the one-trial completion reserve. */
static constexpr double MPC_RTI_REFINEMENT_SOLVE_RESERVE_US =
    MPC_RTI_POST_SOLVE_RESERVE_US + MPC_RTI_LINE_SEARCH_TRIAL_RESERVE_US;

int mpc_rti_build_objective_schedule(
    const MpcRtiNominal_t *nominal,
    const MpcRtiReference_t references[PREDICTION_HORIZON + 1],
    const MpcRtiConfiguration_t *configuration,
    MpcRtiObjectiveSchedule_t *schedule)
{
    if (!nominal || !references || !configuration || !schedule ||
        !nominal->valid || nominal->horizon < 1 ||
        nominal->horizon > PREDICTION_HORIZON) return 0;
    memset(schedule, 0, sizeof(*schedule));
    schedule->horizon = nominal->horizon;
    for (int k = 0; k <= nominal->horizon; ++k) {
        schedule->references[k] = references[k];
        if (k > 0 && configuration->near_wall_tracking_reserve_m > 0.f) {
            float lower_slack = NAN, upper_slack = NAN;
            if (!mpc_rti_compute_corridor_slack(
                    &nominal->states[k].plant, references[k].left_bound,
                    references[k].right_bound, configuration, k,
                    &lower_slack, &upper_slack)) return 0;
            /* Compare with the hard body footprint, NOT the optional
             * nonlinear-rollout acceptance tolerance. This provides a
             * continuous inward target as the nominal enters the reserve,
             * without creating a new hard feasibility boundary. The target
             * is frozen for R1, R2, line search and source adjoint scoring. */
            lower_slack -= configuration->nonlinear_corridor_tolerance_m;
            upper_slack -= configuration->nonlinear_corridor_tolerance_m;
            const float reserve = configuration->near_wall_tracking_reserve_m;
            const float lower_deficit = fmaxf(0.f, reserve - lower_slack);
            const float upper_deficit = fmaxf(0.f, reserve - upper_slack);
            schedule->references[k].e_y += clampf_rti(
                lower_deficit - upper_deficit, -reserve, reserve);
        }
        schedule->speed_weights[k] = nominal->states[k].plant.u > references[k].u
            ? overspeed_weight_for_reference(configuration, &references[k])
            : underspeed_weight_for_reference(configuration, &references[k]);
    }
    schedule->valid = 1;
    return 1;
}

static int evaluate_nonlinear_candidate(
    const MpcRtiNominal_t *candidate,
    const MpcTrajectorySample_t *trajectory,
    size_t trajectory_count,
    double lap_length,
    const MpcRtiCycleConfiguration_t *configuration,
    double *objective,
    float *minimum_corridor_slack,
    float *lateral_accel_proxy,
    int *lateral_accel_proxy_stage,
    float lateral_accel_proxy_by_stage[PREDICTION_HORIZON + 1],
    const MpcRtiCandidatePathDelta_t *verified_path_delta,
    MpcRtiRolloutStatus_t *evaluation_failure=nullptr,
    int allow_hard_constraint_violations_for_metrics=0,
    float (*source_corridor_by_stage)[2]=nullptr)
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
    const MpcRtiObjectiveSchedule_t *schedule = configuration->objective_schedule;
    if (schedule && (!schedule->valid || schedule->horizon < candidate->horizon))
        return 0;
    MpcRtiReference_t references[PREDICTION_HORIZON + 1];
    int identical_references = verified_path_delta &&
        verified_path_delta->sampled_references_valid &&
        verified_path_delta->sample_count == candidate->horizon + 1 &&
        candidate->speed_reference_valid;
    if (identical_references) {
        for (int k = 0; k <= candidate->horizon; ++k) {
            if (candidate->progress[k] !=
                    verified_path_delta->sampled_progress[k] ||
                candidate->speed_reference[k] !=
                    verified_path_delta->sampled_reference[k].u) {
                identical_references = 0;
                break;
            }
        }
    }
    if (identical_references) {
        memcpy(references,verified_path_delta->sampled_reference,
            (size_t)(candidate->horizon+1)*sizeof(references[0]));
    } else {
        /* May be a caller-modified trajectory or an unvalidated
         * objective input. Never use stale progress, speed or geometry. */
        for (int k = 0; k <= candidate->horizon; ++k) {
            if (!mpc_rti_reference_at_progress(trajectory, trajectory_count, lap_length,
                    candidate->progress[k], model, &references[k])) return 0;
            if (candidate->speed_reference_valid)
                references[k].u = candidate->speed_reference[k];
        }
    }
    align_yaw_rate_references(references, candidate->horizon);
    const MpcRtiReference_t *objective_references = schedule
        ? schedule->references : references;
    for (int k = 0; k <= candidate->horizon; ++k) {
        const MpcRtiReference_t *reference = &objective_references[k];
        float stage_corridor_slack = INFINITY;
        if (k > 0 && verified_path_delta) {
            if (verified_path_delta->sample_count != candidate->horizon + 1)
                return 0;
            const float left=verified_path_delta->lower_corridor_slack_m[k];
            const float right=verified_path_delta->upper_corridor_slack_m[k];
            if (!isfinite(left) || !isfinite(right) ||
                left < 0.f || right < 0.f) return 0;
            if(source_corridor_by_stage) {
                source_corridor_by_stage[k][0]=left;
                source_corridor_by_stage[k][1]=right;
            }
            stage_corridor_slack=fminf(left,right);
            min_slack=fminf(min_slack,stage_corridor_slack);
        } else if (k > 0) {
            float lower_slack, upper_slack;
            const int bounds_ok=mpc_rti_compute_corridor_slack(
                &candidate->states[k].plant,
                references[k].left_bound,references[k].right_bound,model,k,
                &lower_slack,&upper_slack);
            if(source_corridor_by_stage) {
                // With the native source SQP consuming these EXACT physical
                // rows, an invalid footprint query may NEVER be replaced by
                // zero-filled/ambiguous geometry. Even the diagnostic-only
                // objective is unusable for a physical QP when no true
                // left/right margin was computed.
                if(!bounds_ok || !std::isfinite(lower_slack) ||
                   !std::isfinite(upper_slack))return 0;
                source_corridor_by_stage[k][0]=lower_slack;
                source_corridor_by_stage[k][1]=upper_slack;
            }
            const int frenet_ok=bounds_ok &&
                lower_slack>=0.0f && upper_slack>=0.0f;
            if (model->world_sweep_check &&
                (!model->world_map_rejection_only || !frenet_ok)) {
                float world_slack=-INFINITY;
                if (!world_sweep_clearance(model,trajectory,trajectory_count,
                        lap_length,candidate->progress[k-1],
                        &candidate->states[k-1].plant,candidate->progress[k],
                        &candidate->states[k].plant,k,&world_slack) &&
                    !allow_hard_constraint_violations_for_metrics)
                    return 0;
                stage_corridor_slack=world_slack;
                min_slack=fminf(min_slack,world_slack);
            } else {
                if (!frenet_ok && !allow_hard_constraint_violations_for_metrics)
                    return 0;
                stage_corridor_slack=fminf(lower_slack,upper_slack);
                min_slack=fminf(min_slack,stage_corridor_slack);
            }
        }
        if(k>0 && model->weight_source_corridor_reserve>0.f &&
           model->near_wall_tracking_reserve_m>0.f) {
            /* Continuous physical viability pressure: zero inside the
             * desired reserve, positive on the approach to the original hard
             * footprint. This same active-plane hinge is linearized as PSD
             * Gauss-Newton curvature in the normal source SQP. */
            const double deficit=fmax(0.,
                double(model->near_wall_tracking_reserve_m)-
                stage_corridor_slack);
            cost+=model->weight_source_corridor_reserve*deficit*deficit;
        }
        const MpcModelState_t *state = &candidate->states[k].plant;
        const double terminal_scale = k == candidate->horizon
            ? model->terminal_multiplier : 1.0;
        const double errors[] = {
            mpc_tracking_excess(state->e_y - reference->e_y,
                model->tracking_lateral_tolerance_m),
            mpc_tracking_excess(state->e_psi - reference->e_psi,
                model->tracking_heading_tolerance_rad),
            state->u - reference->u,
            state->v - reference->v,
            state->r - reference->r,
            state->delayed_steering_command_2 - reference->steering_command,
            state->actual_steering_angle - reference->steering_command};
        const double speed_weight = schedule ? schedule->speed_weights[k] :
            (state->u > reference->u
                ? overspeed_weight_for_reference(model, reference)
                : underspeed_weight_for_reference(model, reference));
        const double weights[] = {
            model->weight_e_y, model->weight_e_psi, speed_weight,
            model->weight_v, model->weight_r,
            model->weight_steering_command,
            model->weight_steering_command};
        for (size_t i = 0; i < sizeof(errors) / sizeof(errors[0]); ++i)
            cost += terminal_scale * weights[i] * errors[i] * errors[i];

        if (k > 0 && model->weight_source_lateral_slip>0.f) {
            double lateral_slip[4]{};
            if(!vehicle_model_source_lateral_slip(
                   state,lateral_slip,nullptr))return 0;
            for(int front=0;front<2;++front) {
                const double excess=std::max(0.,
                    std::abs(lateral_slip[front])-
                    kMpcSourceFrontLateralSlipReserveRad);
                cost+=terminal_scale*
                    model->weight_source_lateral_slip*excess*excess;
            }
        }
        if (k > 0) {
            for (int wheel = 0; wheel < 4; ++wheel) {
                const double excess =
                    candidate->states[k].incoming_wheel_slip_excess_mps[wheel];
                cost += terminal_scale * longitudinal_slip_weight_for_reference(
                    model, reference) *
                    excess * excess;
            }
        }

        lateral_accel_proxy_by_stage[k] = fabsf(state->u * state->r);
        if (lateral_accel_proxy_by_stage[k] > max_lateral_accel) {
            max_lateral_accel = lateral_accel_proxy_by_stage[k];
            *lateral_accel_proxy_stage = k;
        }
        if (k == candidate->horizon) continue;

        const MpcModelControl_t *control = &candidate->controls[k];
        const double target_error = control->steering_target_rad -
            reference->steering_command;
        cost += model->weight_steering_target *
                target_error * target_error +
            model->weight_throttle_command *
                control->throttle_command * control->throttle_command;
        const double steering_delta =
            (double)control->steering_target_rad - previous_steering;
        const double throttle_delta =
            (double)control->throttle_command - previous_throttle;
        const double steering_correction_delta = steering_delta -
            steering_reference_increment(&objective_references[k],
                &objective_references[k + 1]);
        cost += model->weight_steering_slew *
                steering_correction_delta * steering_correction_delta +
            mpc_steering_variation_exact_cost(
                model->weight_steering_correction_variation,
                model->steering_correction_variation_epsilon_rad,
                steering_correction_delta) +
            model->weight_throttle_slew *
                throttle_delta * throttle_delta;
        previous_steering = control->steering_target_rad;
        previous_throttle = control->throttle_command;
    }
    if (!isfinite(cost) || !isfinite(min_slack) ||
        !isfinite(max_lateral_accel)) return 0;
    *objective = cost;
    *minimum_corridor_slack = min_slack;
    *lateral_accel_proxy = max_lateral_accel;
    return 1;
}

int mpc_rti_score_candidate_plan(
    MpcRtiNominal_t *candidate,
    const MpcTrajectorySample_t *trajectory, size_t trajectory_count,
    double lap_length, const MpcRtiCycleConfiguration_t *configuration)
{
    float accel = 0.0f;
    int stage = 0;
    float by_stage[PREDICTION_HORIZON + 1] = {};
    if (!candidate) return 0;
    return evaluate_nonlinear_candidate(candidate, trajectory, trajectory_count,
        lap_length, configuration, &candidate->validation_objective,
        &candidate->validation_minimum_corridor_slack_m, &accel, &stage,
        by_stage, NULL);
}

int mpc_rti_score_candidate_plan_for_diagnostics(
    const MpcRtiNominal_t *candidate,
    const MpcTrajectorySample_t *trajectory, size_t trajectory_count,
    double lap_length, const MpcRtiCycleConfiguration_t *configuration,
    double *objective, float *minimum_corridor_slack_m)
{
    if (!candidate || !candidate->valid || !configuration || !objective ||
        !minimum_corridor_slack_m) return 0;
    float lateral_accel_proxy = 0.0f;
    int lateral_accel_proxy_stage = -1;
    float lateral_accel_proxy_by_stage[PREDICTION_HORIZON + 1] = {};
    return evaluate_nonlinear_candidate(candidate, trajectory,
        trajectory_count, lap_length, configuration, objective,
        minimum_corridor_slack_m, &lateral_accel_proxy,
        &lateral_accel_proxy_stage, lateral_accel_proxy_by_stage,
        NULL, NULL, 1);
}

int mpc_rti_score_candidate_plan_and_corridor_for_diagnostics(
    const MpcRtiNominal_t *candidate,
    const MpcTrajectorySample_t *trajectory, size_t trajectory_count,
    double lap_length, const MpcRtiCycleConfiguration_t *configuration,
    double *objective, float *minimum_corridor_slack_m,
    float corridor_by_stage[PREDICTION_HORIZON+1][2])
{
    if(!candidate || !candidate->valid || !configuration || !objective ||
       !minimum_corridor_slack_m || !corridor_by_stage)return 0;
    float lateral_accel_proxy=0.f;
    int accel_stage=-1;
    float by_stage[PREDICTION_HORIZON+1]{};
    return evaluate_nonlinear_candidate(candidate,trajectory,trajectory_count,
        lap_length,configuration,objective,minimum_corridor_slack_m,
        &lateral_accel_proxy,&accel_stage,by_stage,nullptr,nullptr,1,
        corridor_by_stage);
}

MpcRtiRolloutStatus_t mpc_rti_validate_candidate_plan(
    const MpcRtiState_t *initial_state,
    double initial_progress,
    const MpcModelControl_t controls[PREDICTION_HORIZON],
    const MpcTrajectorySample_t *trajectory,
    size_t trajectory_count,
    double lap_length,
    int horizon,
    float prediction_dt,
    const MpcRtiCycleConfiguration_t *configuration,
    MpcRtiNominal_t *validated_plan,
    int *failure_stage)
{
    return mpc_rti_validate_candidate_plan_with_diagnostics(
        initial_state, initial_progress, controls, trajectory,
        trajectory_count, lap_length, horizon, prediction_dt,
        configuration, validated_plan, NULL, NULL, failure_stage);
}

MpcRtiRolloutStatus_t mpc_rti_validate_candidate_plan_with_diagnostics(
    const MpcRtiState_t *initial_state,
    double initial_progress,
    const MpcModelControl_t controls[PREDICTION_HORIZON],
    const MpcTrajectorySample_t *trajectory,
    size_t trajectory_count,
    double lap_length,
    int horizon,
    float prediction_dt,
    const MpcRtiCycleConfiguration_t *configuration,
    MpcRtiNominal_t *validated_plan,
    MpcRtiDiagnosticStageOutput_t diagnostic_outputs[
        MPC_RTI_DIAGNOSTIC_SAMPLE_COUNT],
    int *diagnostic_output_count,
    int *failure_stage)
{
    if (!validated_plan || !configuration)
        return MPC_RTI_ROLLOUT_INVALID_INPUT;
    memset(validated_plan, 0, sizeof(*validated_plan));
    validated_plan->horizon = horizon;
    validated_plan->speed_reference_valid = 0;
    MpcRtiRolloutStatus_t status = rollout_candidate_impl(
        initial_state, initial_progress, controls, trajectory,
        trajectory_count, lap_length, NULL, NULL, NULL, horizon,
        prediction_dt, &configuration->model, validated_plan->states,
        validated_plan->progress,
        configuration->capture_diagnostic_outputs ? diagnostic_outputs : NULL,
        diagnostic_output_count, failure_stage, NULL, NULL);
    if (status != MPC_RTI_ROLLOUT_OK) return status;
    memcpy(validated_plan->controls, controls,
        (size_t)horizon * sizeof(validated_plan->controls[0]));
    validated_plan->valid = 1;

    // Revalidate the actual original-source plan under the single
    // nonlinear program. No recourse search, separate terminal controller
    // or emergency feasibility optimizer is called here.
    MpcRtiCycleConfiguration_t local_configuration=*configuration;
    double objective = 0.0;
    float minimum_corridor_slack = 0.0f;
    float lateral_accel_proxy = 0.0f;
    int lateral_accel_proxy_stage = 0;
    float lateral_accel_proxy_by_stage[PREDICTION_HORIZON + 1] = {0.0f};
    if (!evaluate_nonlinear_candidate(validated_plan, trajectory,
            trajectory_count, lap_length, &local_configuration, &objective,
            &minimum_corridor_slack, &lateral_accel_proxy,
            &lateral_accel_proxy_stage, lateral_accel_proxy_by_stage,
            NULL)) {
        validated_plan->valid = 0;
        return MPC_RTI_ROLLOUT_INVALID_MODEL;
    }
    validated_plan->validation_objective = objective;
    validated_plan->validation_minimum_corridor_slack_m =
        minimum_corridor_slack;
    return MPC_RTI_ROLLOUT_OK;
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
        result->source_solver_status = MPC_SOURCE_INVALID_MODEL;
        result->admm_status = -1;
        result->nonlinear_failure_reason = -1;
        result->nonlinear_failure_stage = -1;
        result->line_search_full_step_failure_reason = -1;
        result->line_search_full_step_failure_stage = -1;
        result->line_search_last_failure_reason = -1;
        result->line_search_last_failure_stage = -1;
        result->line_search_nominal_objective = NAN;
        result->linearization_failure_stage = -1;
        result->linearization_failure_index = -1;
        result->state_primal_stage = -1;
        result->state_primal_index = -1;
        result->state_dual_stage = -1;
        result->state_dual_index = -1;
        result->input_primal_stage = -1;
        result->input_primal_index = -1;
        result->input_dual_stage = -1;
        result->input_dual_index = -1;
        result->minimum_predicted_corridor_slack_m = NAN;
        result->lateral_accel_proxy_mps2 = NAN;
        result->lateral_accel_proxy_stage = -1;
        result->nonlinear_objective = NAN;
        result->first_pass_objective=NAN;
        result->second_pass_objective=NAN;
        result->refinement_first_prefix_objective = NAN;
        result->refinement_second_prefix_objective = NAN;
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
    const int epoch_shift=configuration->command_epoch_ns>0
        ? mpc_rti_command_epoch_shift(memory->command_epoch_ns,
            configuration->command_epoch_ns,prediction_dt) : 1;
    if(epoch_shift<0) mpc_rti_memory_reset(memory);
    result->command_epoch_shift_stages=epoch_shift;
    MpcRtiCycleConfiguration_t aligned_configuration=*configuration;
    configuration=&aligned_configuration;
    const double deadline_us =
        configuration->absolute_deadline_monotonic_us > 0.0
        ? configuration->absolute_deadline_monotonic_us
        : (configuration->max_cycle_time_us > 0.0
            ? cycle_start_us + configuration->max_cycle_time_us : 0.0);
    MpcRtiNominal_t nominal;
    MpcRtiReference_t references[PREDICTION_HORIZON + 1];
    double nominal_start_us = monotonic_microseconds();
    const int warm_controls_valid = memory->warm_start_controls_valid &&
        memory->warm_start_control_count >= 1 &&
        memory->warm_start_control_count <= PREDICTION_HORIZON;
    const int nominal_built=mpc_rti_build_nominal_impl(&prepared_state,
        current_progress,&memory->nominal,
        warm_controls_valid ? memory->warm_start_controls : nullptr,
        warm_controls_valid ? memory->warm_start_control_count : 0,
        epoch_shift<0 ? 0 : epoch_shift,
        0,
        trajectory,trajectory_count,lap_length,prediction_dt,horizon,
        &configuration->model,&nominal,references);
    // Record these common measurements for both success and failure; only
    // failure diagnostics need an early return below.
    result->nominal_build_us = monotonic_microseconds() - nominal_start_us;
    result->reference_seed_us = nominal.reference_seed_us;
    result->cold_start_queue_seed_us = nominal.cold_start_queue_seed_us;
    result->nominal_rollout_a_us = nominal.nominal_rollout_a_us;
    result->progress_reference_resampling_us =
        nominal.progress_reference_resampling_us;
    if (!nominal_built) {
        result->input_failure_reason = MPC_RTI_INPUT_FAILURE_NOMINAL_BUILD;
        result->nominal_build_failure_stage = nominal.build_failure_stage;
        result->nominal_build_failure_constraint =
            nominal.build_failure_constraint;
        result->nominal_build_failure_index = nominal.build_failure_index;
        result->total_cycle_us = monotonic_microseconds() - cycle_start_us;
        return MPC_RTI_CYCLE_REJECTED_INPUT;
    }
    result->nominal_first_control = nominal.controls[0];
    result->nominal_control_count = horizon;
    result->nominal_control_seed_source = warm_controls_valid ? 2 :
        (memory->nominal.valid ? 1 : 0);
    memcpy(result->nominal_controls, nominal.controls,
        (size_t)horizon * sizeof(result->nominal_controls[0]));

    if (!mpc_rti_build_objective_schedule(&nominal, references,
            &configuration->model, &result->comparison_objective))
        return MPC_RTI_CYCLE_REJECTED_INPUT;
    MpcRtiCycleConfiguration_t comparison_configuration = *configuration;
    comparison_configuration.objective_schedule = &result->comparison_objective;

    {
        // One isolated source-exact optimization cycle. Source dynamics,
        // objective, active tire/contact branches and ORIGINAL physical
        // footprint/actuator inequalities come from the same sparse
        // 29-state NLP. Unlike the older native reduced source QP, no
        // quadratic racing cost determines the accepted command.
        //
        // This is explicitly opt-in and is NOT qualified for 40Hz yet.
        // If a 25ms deadline is missed, publish no stale candidate and
        // restore the previous warm-start memory without modifying the
        // established practice/competition controller.
        const double solve_begin=monotonic_microseconds();
        result->source_exact_nlp_active=1;
        sdu_apex::source_nlp::OfflineOptions opts{};
        // The canonical source MPC preparation already integrated the
        // identical 30-stage nominal in this command epoch; the exact
        // optimizer can reuse its verified original physical prefix.
        opts.initial_same_epoch_source_nominal=&nominal;
        opts.record_primal_trace=configuration->capture_source_benchmark!=0;
        opts.maximum_iterations=
            configuration->source_exact_nlp_iterations;
        opts.use_source_compact_qp_linearization=
            configuration->source_exact_nlp_compact_qp!=0;
        opts.source_qp_curvature_regularization=
            configuration->source_exact_nlp_qp_regularization>0.f?
            configuration->source_exact_nlp_qp_regularization:10000.;
        opts.source_qp_soft_physical_weight=
            configuration->source_exact_nlp_qp_soft_weight>0.f?
            configuration->source_exact_nlp_qp_soft_weight:1.e6;
        opts.minimum_source_clearance_m=
            configuration->source_exact_nlp_extra_clearance_m;
        // Exact source physics still enforces its ORIGINAL nonnegative
        // body-corner corridor. The optional EXTRA clearance is stored
        // in float geometry; ~0.2 micron rounding produced a spurious
        // complete-corner rejection in a real Unity race. Only the extra
        // reserve admits a submicron (0.5 um) numerical tolerance.
        opts.allowable_corridor_violation_m=
            opts.minimum_source_clearance_m>0.?5.e-7:0.;
        // Do not start a complete source-AD iteration unless its normal
        // execution budget remains. An exact-source-feasible current
        // iterate is retained as the same NLP's latest verified solution.
        opts.minimum_remaining_us_for_new_iteration=
            deadline_us>0.?9000.:0.;
        // Reserve 1.5ms for independent physical validation, memory
        // update and publication; never silently use stale solver data.
        opts.absolute_deadline_monotonic_us=
            deadline_us>0. ? std::max(0.,deadline_us-1500.):0.;
        auto optimized=std::make_unique<sdu_apex::source_nlp::OfflineResult>();
        const bool completed=
            sdu_apex::source_nlp::optimize_offline_exact_source(
                prepared_state,current_progress,trajectory,
                trajectory_count,lap_length,prediction_dt,
                comparison_configuration,nominal.controls,opts,
                optimized.get());
        result->total_solve_us=monotonic_microseconds()-solve_begin;
        result->source_exact_nlp_deadline_exhausted=
            optimized->deadline_exhausted;
        result->source_solver_status=optimized->solver_status;
        result->source_linearization_failed=optimized->linearization_failed;
        result->source_qp_failed=optimized->qp_failed;
        result->source_iteration_admission_stopped=optimized->iteration_admission_stopped;
        result->source_exact_nlp_source_trial_evaluations=
            optimized->source_trial_evaluations;
        result->source_exact_nlp_accepted_iterations=
            optimized->accepted_iterations;
        result->source_exact_nlp_initial_source_feasible=
            optimized->initial_source_feasible;
        result->source_exact_nlp_final_source_feasible=
            optimized->final_source_feasible;
        result->source_exact_nlp_kkt_converged=
            optimized->source_nlp_kkt_converged;
        result->source_exact_nlp_full_step_stalled=
            optimized->full_step_sqp_stalled;
        result->source_exact_nlp_initial_source_cost=
            optimized->initial_source_cost;
        result->source_exact_nlp_final_source_cost=
            optimized->final_source_cost;
        result->source_exact_nlp_final_original_margin_m=
            optimized->final_minimum_corridor_slack;
        result->source_exact_nlp_final_original_violation=
            optimized->final_exact_constraint_violation;
        result->source_iteration_metrics_count=static_cast<int>(std::min(
            optimized->history.size(),size_t(MPC_RTI_SOURCE_TRACE_CAPACITY)));
        for(int i=0;i<result->source_iteration_metrics_count;++i) {
            const auto &in=optimized->history[i];
            auto &out=result->source_iteration_metrics[i];
            out.accepted=in.accepted;
            out.full_step_evaluated=in.full_step_evaluated;
            out.full_step_accepted=in.full_step_accepted;
            out.qp_converged=in.qp_converged;
            out.original_duals_trustworthy=in.original_duals_trustworthy;
            out.qp_iterations=in.qp_iterations;
            out.linearization_ms=in.linearization_wall_ms;
            out.qp_ms=in.qp_wall_ms;
            out.source_rollout_ms=in.source_rollout_wall_ms;
            out.total_ms=in.iteration_wall_ms;
            out.cost_before=in.cost_before;
            out.predicted_cost_change=in.predicted_cost_change;
            out.full_step_cost=in.cost_trial;
            out.cost_after=in.cost_after;
            out.accepted_step_length=in.safeguarded_step_length;
            out.qp_primal_residual=in.qp_primal_residual;
            out.qp_dual_residual=in.qp_dual_residual;
            out.stationarity_inf=in.source_nlp_stationarity_inf;
            out.complementarity_inf=in.source_nlp_complementarity_inf;
            out.maximum_footprint_prediction_error_m=in.model_maximum_hard_constraint_error;
        }
        if(completed) {
            result->source_optimized_control_count=horizon;
            std::copy_n(optimized->optimized.controls,horizon,
                        result->source_optimized_controls);
        }
        result->source_primal_trace_count=static_cast<int>(std::min(
            optimized->primal_trace.size(),size_t(MPC_RTI_SOURCE_TRACE_CAPACITY)));
        result->source_primal_trace_truncated=optimized->primal_trace.size()>
            MPC_RTI_SOURCE_TRACE_CAPACITY;
        for(int i=0;i<result->source_primal_trace_count;++i) {
            const auto &in=optimized->primal_trace[i];
            auto &out=result->source_primal_trace[i];
            out.iteration=in.iteration;out.accepted=in.accepted;
            out.trial_evaluated=in.trial_evaluated;
            out.correction_evaluated=in.correction_evaluated;
            out.correction_accepted=in.correction_accepted;
            out.normal_accepted=in.normal_accepted;out.filter_accepted=in.safeguarded_accepted;
            const auto &accepted=in.normal_accepted?in.normal:
                in.safeguarded_accepted?in.safeguarded_trials.back():
                in.correction_accepted?in.correction:in.trial;
            if(in.accepted)std::copy_n(accepted.controls,horizon,out.accepted_controls);
            std::copy_n(in.warm_start.controls,horizon,out.warm_start);
            std::copy_n(in.trial.controls,horizon,out.trial);
            std::copy_n(in.correction.controls,horizon,out.correction);
        }
        result->first_pass_objective=optimized->initial_source_cost;
        result->second_pass_objective=optimized->final_source_cost;
        result->nonlinear_objective=optimized->final_source_cost;
        result->exact_objective_improvement=
            optimized->initial_source_cost-optimized->final_source_cost;
        result->sqp_iterations_attempted=
            static_cast<int>(optimized->history.size());
        result->sqp_iterations_completed=
            result->sqp_iterations_attempted;
        result->sqp_iterations_accepted=
            optimized->accepted_iterations;
        result->exact_rollout_us=result->total_solve_us;
        result->minimum_predicted_corridor_slack_m=static_cast<float>(
            optimized->final_minimum_corridor_slack);
        result->total_cycle_us=monotonic_microseconds()-cycle_start_us;
        if(!completed || !optimized->final_source_feasible) {
            *memory=memory_before;
            result->status=
                !completed ? MPC_RTI_CYCLE_REJECTED_SOLVER :
                optimized->deadline_exhausted
                  ? MPC_RTI_CYCLE_WAITING_FOR_SOLVER_BUDGET :
                    MPC_RTI_CYCLE_REJECTED_NONLINEAR_ROLLOUT;
            return result->status;
        }
        auto checked=std::make_unique<MpcRtiNominal_t>();
        int failure_stage=-1;
        const auto original_validation=mpc_rti_validate_candidate_plan(
            &prepared_state,current_progress,optimized->optimized.controls,
            trajectory,trajectory_count,lap_length,horizon,prediction_dt,
            &comparison_configuration,checked.get(),&failure_stage);
        result->nonlinear_failure_stage=failure_stage;
        result->total_cycle_us=monotonic_microseconds()-cycle_start_us;
        if(original_validation!=MPC_RTI_ROLLOUT_OK || !checked->valid) {
            *memory=memory_before;
            result->status=MPC_RTI_CYCLE_REJECTED_NONLINEAR_ROLLOUT;
            return result->status;
        }
        if(deadline_us>0. &&
           monotonic_microseconds()>=deadline_us) {
            // An accurate but late control is NOT the control actually
            // applied in the corresponding original source epoch.
            *memory=memory_before;
            result->source_exact_nlp_deadline_exhausted=1;
            result->source_solver_status=MPC_SOURCE_DEADLINE_EXCEEDED;
            result->status=MPC_RTI_CYCLE_WAITING_FOR_SOLVER_BUDGET;
            return result->status;
        }
        result->candidate_control_count=horizon;
        std::copy_n(optimized->optimized.controls,horizon,
                    result->candidate_controls);
        result->first_control=result->candidate_controls[0];
        result->published_steering_command=
            result->first_control.steering_target_rad;
        result->published_throttle=
            result->first_control.throttle_command;
        memory->nominal=*checked;
            memory->solver_nominal_progress_valid=0;
        if(!mpc_rti_memory_set_warm_start_controls_at_epoch(
            memory,result->candidate_controls,horizon,
            configuration->command_epoch_ns)) {
            *memory=memory_before;
            result->candidate_control_count=0;
            result->status=MPC_RTI_CYCLE_REJECTED_INPUT;
            return result->status;
        }
        result->total_cycle_us=monotonic_microseconds()-cycle_start_us;
        result->status=MPC_RTI_CYCLE_ACCEPTED_FEASIBLE;
        return result->status;
    }

}
