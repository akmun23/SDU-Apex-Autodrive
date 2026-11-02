#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define _POSIX_C_SOURCE 200809L

#include "mpc_rti_internal.hpp"
#include "mpc_steering_variation.hpp"
#include "mpc_tracking_tolerance.h"
#include "vehicle_model.h"

#include <algorithm>
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

int mpc_rti_should_keep_first_pass_after_refinement_deadline_skip(
    int first_pass_admm_status,
    int refinement_admm_status,
    int refinement_skipped_for_deadline)
{
    const int first_pass_has_iterate =
        first_pass_admm_status == RICCATI_STATUS_OPTIMAL ||
        first_pass_admm_status == RICCATI_STATUS_MAX_ITERATIONS;
    return first_pass_has_iterate && refinement_skipped_for_deadline &&
        refinement_admm_status < 0;
}

int mpc_rti_line_search_budget_available(double remaining_us)
{
    /* The live v323 hairpin solve had 3.05 ms left at the publication gate;
     * its two exact backtracks took 1.51 ms offline and found a valid alpha
     * 0.25 action. A 7 ms precondition suppressed both useful trials. Keep
     * 2.2 ms as the minimum per-trial reserve; the final publication gate
     * remains authoritative if a trial takes longer under live load. */
    const double kMinimumRemainingUs = 2200.0;
    return isfinite(remaining_us) && remaining_us >= kMinimumRemainingUs;
}

int mpc_rti_line_search_should_backtrack(
    MpcRtiRolloutStatus_t candidate_rollout_status,
    int candidate_exact_valid, int nominal_exact_valid,
    int exact_merit_regression)
{
    if (exact_merit_regression) return 1;
    if (candidate_exact_valid) return 0;
    if (candidate_rollout_status == MPC_RTI_ROLLOUT_CORRIDOR ||
        candidate_rollout_status == MPC_RTI_ROLLOUT_TERMINAL_VIABILITY) return 1;
    /* A successful source step followed by a failed exact objective or
     * horizon-domain check is a nonlinear INVALID_MODEL candidate, not
     * permission to suppress every intermediate globalization trial.
     * A physically invalid source step may also become valid for a smaller
     * correction, provided the exact nominal is already certified. */
    return nominal_exact_valid &&
        (candidate_rollout_status == MPC_RTI_ROLLOUT_OK ||
         candidate_rollout_status == MPC_RTI_ROLLOUT_INVALID_MODEL);
}

int mpc_rti_line_search_should_continue(
    MpcRtiRolloutStatus_t trial_status, int nominal_exact_valid)
{
    return trial_status == MPC_RTI_ROLLOUT_OK ||
        trial_status == MPC_RTI_ROLLOUT_CORRIDOR ||
        trial_status == MPC_RTI_ROLLOUT_TERMINAL_VIABILITY ||
        (trial_status == MPC_RTI_ROLLOUT_INVALID_MODEL &&
         nominal_exact_valid);
}

int mpc_rti_refinement_improves_action(
    int first_valid, double first_objective,
    int refined_valid, double refined_objective)
{
    if (!refined_valid || !isfinite(refined_objective)) return 0;
    return !first_valid || !isfinite(first_objective) ||
        refined_objective < first_objective;
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
        (configuration->command_increment_qp_enabled == 0 ||
         (configuration->command_increment_qp_enabled == 1 &&
          configuration->max_steering_command_step_rad > 0.f &&
          configuration->max_steering_command_step_rad <=
              2.f * configuration->max_steering_rad &&
          configuration->max_throttle_command_step > 0.f &&
          configuration->max_throttle_command_step <= 1.f)) &&
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
    for (int k = 0; k < horizon; ++k) {
        MpcStageResult_t step = mpc_vehicle_model_step(
            &states[k].plant, &controls[k], prediction_dt, 0.0f);
        if (!step.valid || !isfinite(progress[k] + step.delta_s_m)) {
            *failure_index = k + 1;
            return 0;
        }
        double next_progress;
        if (!mpc_trajectory_advance_tracking_with_progress(trajectory, trajectory_count,
                lap_length, progress[k], states[k].plant.e_y,
                states[k].plant.e_psi, step.body_forward_delta_m,
                step.body_lateral_delta_m, step.body_yaw_delta_rad,
                progress[k] + step.delta_s_m, &next_progress,
                &step.next.e_y, &step.next.e_psi,
                states[k + 1].incoming_tracking_jacobian)) {
            *failure_index = k + 1;
            return 0;
        }
        step.delta_s_m = (float)(next_progress - progress[k]);
        states[k + 1].incoming_tracking_jacobian_valid = 1;
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
        progress[k + 1] = next_progress;
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

int mpc_rti_build_corridor_polygon(
    const MpcModelState_t *state, float left_bound_m, float right_bound_m,
    const MpcRtiConfiguration_t *configuration, int prediction_index,
    RiccatiStatePolygon_t *polygon)
{
    if (!state || !configuration || !polygon || !isfinite(state->e_psi) ||
        !isfinite(left_bound_m) || !isfinite(right_bound_m) ||
        prediction_index < 0 || prediction_index > PREDICTION_HORIZON) return 0;
    const float margin = corridor_margin_at_prediction(configuration, prediction_index);
    const float sine = sinf(state->e_psi), cosine = cosf(state->e_psi);
    const float x[2] = {-MPC_REAR_OVERHANG_M, MPC_REAR_AXLE_TO_FRONT_BUMPER_M};
    const float y[2] = {-.5f*MPC_CAR_WIDTH_M, .5f*MPC_CAR_WIDTH_M};
    polygon->count = 0;
    for (int along = 0; along < 2; ++along) {
        for (int side = 0; side < 2; ++side) {
            const float lateral = x[along]*sine + y[side]*cosine;
            const float derivative = x[along]*cosine - y[side]*sine;
            const int left = polygon->count++, right = polygon->count++;
            polygon->normal[left][0] = 1.f;
            polygon->normal[left][1] = derivative;
            polygon->upper[left] = left_bound_m - margin - lateral + derivative*state->e_psi;
            polygon->normal[right][0] = -1.f;
            polygon->normal[right][1] = -derivative;
            polygon->upper[right] = right_bound_m - margin + lateral - derivative*state->e_psi;
        }
    }
    return 1;
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
        float prev_s=current_state->plant.delayed_steering_command_2;
        float prev_t=current_state->plant.delayed_throttle_command_2;
        for (int k=0;k<horizon;++k) {
            auto &u=nominal->controls[k];
            u.steering_target_rad=clampf_rti(
                u.steering_target_rad,
                fmaxf(-configuration->max_steering_rad,
                      prev_s-configuration->max_steering_command_step_rad),
                fminf(configuration->max_steering_rad,
                      prev_s+configuration->max_steering_command_step_rad));
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
        previous_nominal, NULL, 0, 1, trajectory, trajectory_count, lap_length,
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
        previous_nominal, warm_start_controls, warm_start_control_count, 1,
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
        previous_nominal,same_epoch_controls,same_epoch_control_count,0,
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

typedef struct {
    const MpcRtiProblem_t *problem;
    const MpcRtiNominal_t *nominal;
    const float (*gain)[RICCATI_MAX_NU][RICCATI_MAX_NX];
    float linear_x[RICCATI_HORIZON + 1][RICCATI_MAX_NX];
    float alpha;
    float gain_scale;
} MpcRtiRolloutPolicy_t;

static void initialize_rollout_policy(MpcRtiRolloutPolicy_t *policy,
    const MpcRtiProblem_t *problem, const MpcRtiNominal_t *nominal,
    const MpcModelControl_t controls[PREDICTION_HORIZON],
    const float gain[RICCATI_HORIZON][RICCATI_MAX_NU][RICCATI_MAX_NX],
    int command_increment_qp_enabled)
{
    policy->problem = problem;
    policy->nominal = nominal;
    policy->gain = gain;
    policy->alpha = 1.f;
    policy->gain_scale = 1.f;
    memcpy(policy->linear_x[0], problem->x0, sizeof(problem->x0));
    /* Use the linear trajectory of PROJECTED commands, not the unconstrained
     * ADMM primal state paired with a different set of projected controls.
     * IMPORTANT: transformed QPs require increments (absolute minus the
     * carried previous-command state). Passing absolute commands here
     * double-counts the physical control in A*p+B*du and corrupts Riccati
     * feedback and backtracking when the increment QP is enabled. */
    for (int k = 0; k < RICCATI_HORIZON; ++k) {
        const RiccatiStepData_t *stage = &problem->steps[k];
        const float input_s = mpc_rti_qp_decision_from_absolute_command(
            controls[k].steering_target_rad,
            policy->linear_x[k][MPC_OPT_IDX_PREVIOUS_STEERING_COMMAND],
            command_increment_qp_enabled);
        const float input_t = mpc_rti_qp_decision_from_absolute_command(
            controls[k].throttle_command,
            policy->linear_x[k][MPC_OPT_IDX_PREVIOUS_THROTTLE_COMMAND],
            command_increment_qp_enabled);
        for (int i = 0; i < RICCATI_MAX_NX; ++i) {
            double next = stage->d[i] +
                stage->B[i][0] * input_s +
                stage->B[i][1] * input_t;
            for (int j = 0; j < RICCATI_MAX_NX; ++j)
                next += stage->A[i][j] * policy->linear_x[k][j];
            policy->linear_x[k+1][i] = (float)next;
        }
    }
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
    const MpcRtiRolloutPolicy_t *policy,
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
        if (policy) {
            float actual[MPC_RTI_NX] = {0}, base[MPC_RTI_NX] = {};
            serialize_state(&states[k], actual);
            serialize_state(&policy->nominal->states[k], base);
            actual[MPC_OPT_IDX_PROGRESS_DEVIATION] = (float)
                (candidate_progress[k] - policy->nominal->progress[k]);
            actual[MPC_OPT_IDX_PREVIOUS_STEERING_COMMAND] = states[k].plant.delayed_steering_command_2;
            actual[MPC_OPT_IDX_PREVIOUS_THROTTLE_COMMAND] = states[k].plant.delayed_throttle_command_2;
            base[MPC_OPT_IDX_PREVIOUS_STEERING_COMMAND] = policy->nominal->states[k].plant.delayed_steering_command_2;
            base[MPC_OPT_IDX_PREVIOUS_THROTTLE_COMMAND] = policy->nominal->states[k].plant.delayed_throttle_command_2;
            float correction[2] = {};
            for (int j = 0; j < MPC_RTI_NX; ++j) {
                const float reference = base[j] + policy->alpha * (policy->linear_x[k][j]-base[j]);
                const float dx = actual[j]-reference;
                correction[0] += policy->gain[k][0][j]*dx;
                correction[1] += policy->gain[k][1][j]*dx;
            }
            const RiccatiStepData_t *stage = &policy->problem->steps[k];
            if (configuration->command_increment_qp_enabled) {
                // The QP inputs are command differences, whereas the
                // nonlinear plant always consumes ABSOLUTE commands.
                // Never clamp a physical command to a delta-input box.
                const float prev_s=states[k].plant.delayed_steering_command_2;
                const float prev_t=states[k].plant.delayed_throttle_command_2;
                adjusted.steering_target_rad=clampf_rti(
                    adjusted.steering_target_rad+policy->gain_scale*correction[0],
                    fmaxf(-configuration->max_steering_rad,
                          prev_s-configuration->max_steering_command_step_rad),
                    fminf(configuration->max_steering_rad,
                          prev_s+configuration->max_steering_command_step_rad));
                adjusted.throttle_command=clampf_rti(
                    adjusted.throttle_command+policy->gain_scale*correction[1],
                    fmaxf(MPC_THROTTLE_MIN_DRIVE,
                          prev_t-configuration->max_throttle_command_step),
                    fminf(MPC_THROTTLE_MAX,
                          prev_t+configuration->max_throttle_command_step));
            } else {
                adjusted.steering_target_rad = clampf_rti(
                    adjusted.steering_target_rad + policy->gain_scale*correction[0],
                    stage->u_lb[0], stage->u_ub[0]);
                adjusted.throttle_command = clampf_rti(
                    adjusted.throttle_command + policy->gain_scale*correction[1],
                    stage->u_lb[1], stage->u_ub[1]);
            }
        }
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
        progress, NULL, NULL, failure_stage, NULL, NULL, NULL);
}

void mpc_rti_memory_reset(MpcRtiMemory_t *memory)
{
    if (!memory) return;
    memset(memory, 0, sizeof(*memory));
    riccati_admm_state_init(&memory->solver_state);
}

void mpc_rti_memory_reset_solver_state(MpcRtiMemory_t *memory)
{
    if (!memory) return;
    riccati_admm_state_init(&memory->solver_state);
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
    if(previous_ns<=0 || current_ns<previous_ns ||
       !isfinite(prediction_dt) || prediction_dt<=0.f) return -1;
    const double stages=(double)(current_ns-previous_ns)/
        ((double)prediction_dt*1.e9);
    if(stages>=PREDICTION_HORIZON-.5) return -1;
    return (int)llround(stages);
}

static int terminal_dynamic_instability(const MpcRtiNominal_t *p)
{
    if(!p || !p->valid || p->horizon<1)return 1;
    const MpcModelState_t *s=&p->states[p->horizon].plant;
    return !isfinite(s->e_psi) || !isfinite(s->u) || !isfinite(s->v) ||
        fabsf(s->e_psi)>.35f ||
        fabsf(s->v)>fmaxf(.75f,.25f*fabsf(s->u));
}

static int valid_cycle_configuration(
    const MpcRtiCycleConfiguration_t *configuration)
{
    if (!configuration || configuration->command_epoch_ns<0) return 0;
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
        finite_nonnegative(configuration->max_cycle_time_us) &&
        finite_nonnegative(configuration->absolute_deadline_monotonic_us) &&
        (configuration->enable_nonlinear_line_search == 0 ||
         configuration->enable_nonlinear_line_search == 1) &&
        (configuration->feedback_rollout == 0 || configuration->feedback_rollout == 1) &&
        isfinite(configuration->feedback_rollout_gain_scale) &&
        configuration->feedback_rollout_gain_scale >= 0.f &&
        configuration->feedback_rollout_gain_scale <= 1.f &&
        configuration->line_search_max_backtracks >= 0 &&
        configuration->line_search_max_backtracks <= 4 &&
        configuration->max_sqp_iterations >= 0 &&
        configuration->max_sqp_iterations <= 2 &&
        (configuration->source_informed_sqp == 0 ||
         configuration->source_informed_sqp == 1) &&
        (configuration->terminal_viability == 0 || configuration->terminal_viability == 1) &&
        (configuration->prefer_terminal_continuable_refinement == 0 ||
         configuration->prefer_terminal_continuable_refinement == 1) &&
        isfinite(configuration->minimum_sqp_model_agreement_ratio) &&
        configuration->minimum_sqp_model_agreement_ratio >= 0.f &&
        configuration->minimum_sqp_model_agreement_ratio <= 1.f &&
        (configuration->capture_sqp_prediction_metrics == 0 ||
         configuration->capture_sqp_prediction_metrics == 1) &&
        (!configuration->enable_nonlinear_line_search ||
         configuration->line_search_max_backtracks > 0) &&
        (configuration->line_search_command_epoch_valid == 0 ||
         configuration->line_search_command_epoch_valid == 1) &&
        (configuration->capture_diagnostic_outputs == 0 ||
         configuration->capture_diagnostic_outputs == 1);
}

typedef struct
{
    MpcRtiCycleStatus_t status;
    int sqp_attempted;
    int sqp_completed;
    int sqp_accepted;
    MpcRtiNominal_t candidate;
    MpcModelControl_t proposed_controls[PREDICTION_HORIZON];
    MpcRtiCandidatePathDelta_t path_delta;
    MpcModelControl_t first_action;
    int iterations;
    float primal_residual;
    float dual_residual;
    int admm_status;
    int solver_skipped_for_deadline;
    int solver_time_budget_exhausted;
    int restoration_attempted;
    int restoration_accepted;
    int restoration_iterations;
    float restoration_primal_residual;
    double restoration_us;
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
    int line_search_attempted;
    int line_search_trials;
    int line_search_accepted;
    int line_search_skipped_budget;
    int line_search_skipped_command_epoch;
    int line_search_full_step_failure_reason;
    int line_search_full_step_failure_stage;
    int line_search_last_failure_reason;
    int line_search_last_failure_stage;
    int line_search_merit_triggered;
    int line_search_nominal_exact_valid;
    double line_search_nominal_objective;
    MpcRtiSqpStepMetrics_t sqp_step_metrics;
    float line_search_selected_alpha;
    double line_search_us;
    int diagnostic_output_count;
    MpcRtiDiagnosticStageOutput_t
        diagnostic_outputs[MPC_RTI_DIAGNOSTIC_SAMPLE_COUNT];
    int first_action_valid;
    double nonlinear_objective;
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
    uint64_t solver_cold_start_initialization_time_ns;
    uint64_t riccati_rhs_time_ns;
    uint64_t admm_projection_time_ns;
    uint64_t rho_update_time_ns;
    uint64_t solver_scaling_time_ns;
    uint64_t solver_output_unscale_time_ns;
    int jacobian_stages_completed;
    int jacobian_threads_used;
    int jacobian_source_sensitivity_calls;
    int jacobian_finite_difference_columns;
    double jacobian_wall_us;
    double jacobian_thread_cpu_us;
    double jacobian_slowest_stage_us;
    double jacobian_worker_cpu_us[MPC_RTI_MAX_JACOBIAN_THREADS];
    double jacobian_worker_stage_wall_us[MPC_RTI_MAX_JACOBIAN_THREADS];
    int jacobian_worker_stage_count[MPC_RTI_MAX_JACOBIAN_THREADS];
    uint64_t jacobian_worker_os_tid[MPC_RTI_MAX_JACOBIAN_THREADS];
    double qp_build_us;
    double admm_us;
    double admm_thread_cpu_us;
    double exact_rollout_us;
    double nonlinear_validation_us;
    double solve_us;
} MpcRtiPassResult_t;

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
    int allow_hard_constraint_violations_for_metrics=0)
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
        if (k > 0 && verified_path_delta) {
            if (verified_path_delta->sample_count != candidate->horizon + 1)
                return 0;
            const float left=verified_path_delta->lower_corridor_slack_m[k];
            const float right=verified_path_delta->upper_corridor_slack_m[k];
            if (!isfinite(left) || !isfinite(right) ||
                left < 0.f || right < 0.f) return 0;
            min_slack=fminf(min_slack,fminf(left,right));
        } else if (k > 0) {
            float lower_slack, upper_slack;
            const int bounds_ok=mpc_rti_compute_corridor_slack(
                &candidate->states[k].plant,
                references[k].left_bound,references[k].right_bound,model,k,
                &lower_slack,&upper_slack);
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
                min_slack=fminf(min_slack,world_slack);
            } else {
                if (!frenet_ok && !allow_hard_constraint_violations_for_metrics)
                    return 0;
                min_slack=fminf(min_slack,fminf(lower_slack,upper_slack));
            }
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
    if(configuration->terminal_viability && candidate->horizon==PREDICTION_HORIZON &&
       !allow_hard_constraint_violations_for_metrics) {
        double terminal_cost=0.;
        if(!mpc_rti_terminal_value(candidate,trajectory,trajectory_count,lap_length,
            model,schedule,&terminal_cost,1)) {
            if(evaluation_failure)*evaluation_failure=MPC_RTI_ROLLOUT_TERMINAL_VIABILITY;
            return 0;
        }
        cost+=terminal_cost;
    }
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
        diagnostic_output_count, failure_stage, NULL, NULL, NULL);
    if (status != MPC_RTI_ROLLOUT_OK) return status;
    memcpy(validated_plan->controls, controls,
        (size_t)horizon * sizeof(validated_plan->controls[0]));
    validated_plan->valid = 1;

    MpcRtiCycleConfiguration_t local_configuration=*configuration;
    MpcRtiObjectiveSchedule_t local_schedule{};
    if(configuration->terminal_viability && horizon==PREDICTION_HORIZON &&
        !configuration->objective_schedule) {
        MpcRtiReference_t refs[PREDICTION_HORIZON+1]{};
        if(!mpc_rti_references_from_exact_nominal(validated_plan,trajectory,
            trajectory_count,lap_length,&configuration->model,refs) ||
           !mpc_rti_build_objective_schedule(validated_plan,refs,&configuration->model,&local_schedule) ||
           !mpc_rti_initialize_terminal_targets(validated_plan,trajectory,
            trajectory_count,lap_length,prediction_dt,&configuration->model,&local_schedule)) {
            validated_plan->valid=0;return MPC_RTI_ROLLOUT_TERMINAL_VIABILITY;
        }
        local_configuration.objective_schedule=&local_schedule;
    }
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

int mpc_rti_terminal_extension_probe(
    const MpcRtiNominal_t *plan,
    const MpcTrajectorySample_t *trajectory,
    size_t trajectory_count, double lap_length,
    float prediction_dt,
    const MpcRtiCycleConfiguration_t *configuration)
{
    if (!plan || !plan->valid ||
        plan->horizon!=PREDICTION_HORIZON ||
        !trajectory || trajectory_count<3 ||
        !isfinite(lap_length) || lap_length<=0. ||
        !isfinite(prediction_dt) || prediction_dt<=0.f ||
        !configuration) return 0;

    const int horizon=plan->horizon;
    MpcTrajectorySample_t sample={};
    if(!mpc_trajectory_sample(trajectory,trajectory_count,
            lap_length,plan->progress[horizon],&sample))return 0;
    const MpcRtiConfiguration_t *model=&configuration->model;
    MpcModelControl_t source_command={
        .steering_target_rad=clampf_rti(
            (float)(sample.source_witness_valid
                ? sample.source_steering_target
                : vehicle_model_steering_for_curvature(
                    (float)sample.curvature)),
            -model->max_steering_rad,model->max_steering_rad),
        .throttle_command=clampf_rti(
            sample.source_witness_valid
                ? (float)sample.source_throttle
                : plan->controls[horizon-1].throttle_command,
            MPC_THROTTLE_MIN_DRIVE,MPC_THROTTLE_MAX)};
    MpcRtiCycleConfiguration_t validation=*configuration;
    validation.objective_schedule=NULL;
    for(int option=0;option<2;++option) {
        const MpcModelControl_t command=
            option==0 ? source_command : plan->controls[horizon-1];
        MpcModelControl_t controls[PREDICTION_HORIZON]{};
        for(int k=0;k<MPC_RECOVERY_PREFIX_HORIZON;++k)
            controls[k]=command;
        MpcRtiNominal_t extended={};
        int failed=-1;
        if(mpc_rti_validate_candidate_plan(
                &plan->states[horizon],plan->progress[horizon],
                controls,trajectory,trajectory_count,lap_length,
                MPC_RECOVERY_PREFIX_HORIZON,prediction_dt,
                &validation,&extended,&failed)!=MPC_RTI_ROLLOUT_OK)
            continue;
        const MpcModelState_t *s=
            &extended.states[MPC_RECOVERY_PREFIX_HORIZON].plant;
        if(isfinite(s->e_y) && isfinite(s->e_psi) &&
           isfinite(s->u) && isfinite(s->v) &&
           fabsf(s->e_psi)<.35f &&
           fabsf(s->v)<fmaxf(.60f,.25f*fabsf(s->u)))
            return 1;
    }
    return 0;
}

/* Reuse the first QP's exact source Jacobians for a feasibility-biased
 * alternative. QP-only polygon/box relaxation is a SEARCH device, never a
 * safety constraint relaxation. The native nonlinear validator below remains
 * authoritative, with no epsilon widening and no acceptance of violations.
 *
 * Form the exact rank-one PSD Gauss-Newton Hessian for each violated
 * polygon plane, including lateral/heading cross terms at the nominal.
 * This is not a mathematical slack-variable Phase-I; it is a cheap, convex
 * feasibility-restoration direction that can be explored without rebuilding
 * 30 expensive physical Jacobians. */
int mpc_rti_prepare_feasibility_restoration_qp(
    MpcRtiProblem_t *problem, const MpcRtiNominal_t *nominal,
    float relaxation, float wall_weight)
{
    if (!problem || !nominal || !nominal->valid ||
        !isfinite(relaxation) || relaxation <= 0.f || relaxation > .5f ||
        !isfinite(wall_weight) || wall_weight <= 0.f) return 0;
    /* Add PSD wall curvature without reducing individual entries of the
     * coupled source-slip Hessian. Preserve all physical input boxes. */
    for (int k = 0; k <= RICCATI_HORIZON; ++k) {
        const int model_stage = k == RICCATI_HORIZON
            ? PREDICTION_HORIZON : k + MPC_COMMAND_DELAY_STEPS;
        const MpcModelState_t *x = &nominal->states[model_stage].plant;
        RiccatiStatePolygon_t *polygon = &problem->state_polygons[k];
        float *lb = k == RICCATI_HORIZON
            ? problem->terminal_x_lb : problem->steps[k].x_lb;
        float *ub = k == RICCATI_HORIZON
            ? problem->terminal_x_ub : problem->steps[k].x_ub;
        float *Q = k == RICCATI_HORIZON
            ? problem->terminal_Q : problem->steps[k].Q_diag;
        float *q = k == RICCATI_HORIZON
            ? problem->terminal_q : problem->steps[k].q;
        lb[MPC_STATE_IDX_EY] -= relaxation;
        ub[MPC_STATE_IDX_EY] += relaxation;
        const float xy[2] = {x->e_y, x->e_psi};
        for (int plane = 0; plane < polygon->count; ++plane) {
            const float upper = polygon->upper[plane];
            const float n0 = polygon->normal[plane][0];
            const float n1 = polygon->normal[plane][1];
            const float violation = n0*xy[0] + n1*xy[1] - upper + 0.02f;
            polygon->upper[plane] = upper + relaxation;
            if (violation <= 0.0f) continue;
            const float normals[2] = {n0,n1};
            for(int i=0;i<2;++i) {
                float affine=2.f*wall_weight*violation*normals[i];
                for(int j=0;j<2;++j) {
                    const float h=2.f*wall_weight*normals[i]*normals[j];
                    if(k<RICCATI_HORIZON || i==j)affine-=h*xy[j];
                    if(i==j)Q[i]+=h;
                    else if(k<RICCATI_HORIZON)
                        problem->steps[k].Q_offdiag[i][j]+=h;
                }
                q[i]+=affine;
            }
        }
    }
    return 1;
}

void mpc_rti_rebase_solver_warm_start(MpcRtiMemory_t *memory,
    const MpcRtiNominal_t *nominal,const float *x0,
    RiccatiAdmmConfig_t *solver_config,int stage_shift)
{
    /* The progress coordinate is a deviation from THIS QP's nominal,
     * unlike the absolute physical states. Shift once at a new epoch, then
     * rebase the persistent primal before same-epoch SQP reuse. */
    if (memory->solver_state.initialized) {
        const int shift=solver_config->reuse_same_epoch ? 0 : stage_shift;
        for(int i=0;i<shift;++i)
            riccati_admm_shift_warm_start(&memory->solver_state,x0);
        if (memory->solver_nominal_progress_valid) {
            const float scale=memory->solver_state.scaling_enabled
                ? memory->solver_state.state_scale[MPC_OPT_IDX_PROGRESS_DEVIATION] : 1.f;
            for(int k=1;k<=RICCATI_HORIZON;++k) {
                const int source=k+shift<=RICCATI_HORIZON?k+shift:RICCATI_HORIZON;
                memory->solver_state.z_x[k][MPC_OPT_IDX_PROGRESS_DEVIATION] +=
                    (float)((memory->solver_nominal_progress[source]-nominal->progress[k])/scale);
            }
        }
        solver_config->reuse_same_epoch=1;
    }
    memcpy(memory->solver_nominal_progress,nominal->progress,
        sizeof(memory->solver_nominal_progress));
    memory->solver_nominal_progress_valid=1;

}

struct MpcRtiSqpCurvature {
    MpcRtiProblem_t problem{};
    MpcModelControl_t controls[PREDICTION_HORIZON]{};
    int valid{};
};

static void add_source_curvature(MpcRtiProblem_t *problem,
    const MpcRtiNominal_t *nominal, int increment_qp,
    const double regularization[2])
{
    constexpr int previous[2]={MPC_OPT_IDX_PREVIOUS_STEERING_COMMAND,
        MPC_OPT_IDX_PREVIOUS_THROTTLE_COMMAND};
    for(int k=0;k<RICCATI_HORIZON;++k) {
        auto &s=problem->steps[k];
        const double command[2]={nominal->controls[k].steering_target_rad,
            nominal->controls[k].throttle_command};
        for(int a=0;a<2;++a) {
            const float h=(float)regularization[a];
            s.R_diag[a]+=h;
            s.r[a]-=h*command[a];
            if(increment_qp) {
                s.Q_diag[previous[a]]+=h;
                s.N[previous[a]][a]+=h;
                s.q[previous[a]]-=h*command[a];
            }
        }
    }
}

static double qp_objective_for_controls(
    const MpcRtiProblem_t &problem,
    const MpcModelControl_t controls[PREDICTION_HORIZON],
    int command_increments)
{
    double x[RICCATI_MAX_NX]{};
    std::copy_n(problem.x0, RICCATI_MAX_NX, x);
    double cost = 0.;
    for (int k = 0; k < RICCATI_HORIZON; ++k) {
        const RiccatiStepData_t &stage = problem.steps[k];
        const double u[2] = {
            controls[k].steering_target_rad - (command_increments
                ? x[MPC_OPT_IDX_PREVIOUS_STEERING_COMMAND] : 0.),
            controls[k].throttle_command - (command_increments
                ? x[MPC_OPT_IDX_PREVIOUS_THROTTLE_COMMAND] : 0.)};
        for (int i = 0; i < RICCATI_MAX_NX; ++i) {
            cost += .5 * stage.Q_diag[i] * x[i] * x[i] + stage.q[i] * x[i];
            for (int j = 0; j < RICCATI_MAX_NX; ++j)
                cost += .5 * stage.Q_offdiag[i][j] * x[i] * x[j];
            for (int a = 0; a < RICCATI_MAX_NU; ++a)
                cost += x[i] * stage.N[i][a] * u[a];
        }
        for (int a = 0; a < RICCATI_MAX_NU; ++a)
            cost += .5 * stage.R_diag[a] * u[a] * u[a] +
                stage.r[a] * u[a];
        cost += stage.R_cross * u[0] * u[1];

        double next[RICCATI_MAX_NX]{};
        for (int i = 0; i < RICCATI_MAX_NX; ++i) {
            next[i] = stage.d[i];
            for (int j = 0; j < RICCATI_MAX_NX; ++j)
                next[i] += stage.A[i][j] * x[j];
            for (int a = 0; a < RICCATI_MAX_NU; ++a)
                next[i] += stage.B[i][a] * u[a];
        }
        std::copy_n(next, RICCATI_MAX_NX, x);
    }
    for (int i = 0; i < RICCATI_MAX_NX; ++i)
        cost += .5 * problem.terminal_Q[i] * x[i] * x[i] +
            problem.terminal_q[i] * x[i];
    return cost;
}

static double agreement_ratio(double predicted, double actual)
{
    if (predicted > 1.e-9) return actual / predicted;
    return actual > 1.e-9 ? 1. : -1.;
}

static void initialize_step_metrics(MpcRtiSqpStepMetrics_t *metrics)
{
    if (!metrics) return;
    memset(metrics, 0, sizeof(*metrics));
    metrics->selected_alpha = NAN;
    metrics->source_exact_rollout_status = -1;
    metrics->source_exact_failure_stage = -1;
    metrics->maximum_state_error_stage = -1;
    metrics->maximum_state_error_index = -1;
    metrics->maximum_footprint_error_stage = -1;
    metrics->maximum_source_footprint_violation_stage = -1;
    metrics->qp_predicted_cost_reduction = NAN;
    metrics->unscaled_qp_predicted_cost_reduction = NAN;
    metrics->source_exact_cost_reduction = NAN;
    metrics->realized_source_exact_cost_reduction = NAN;
    metrics->selected_qp_predicted_cost_reduction = NAN;
    metrics->selected_source_exact_cost_reduction = NAN;
    metrics->model_agreement_ratio = NAN;
    metrics->realized_model_agreement_ratio = NAN;
    metrics->selected_model_agreement_ratio = NAN;
    metrics->trust_region_scale_before = NAN;
    metrics->trust_region_scale_after = NAN;
    metrics->maximum_state_error = NAN;
    metrics->maximum_footprint_constraint_error_m = NAN;
    metrics->maximum_source_footprint_violation_m = NAN;
    for (int k = 0; k <= PREDICTION_HORIZON; ++k) {
        metrics->state_error_by_stage[k] = NAN;
        metrics->state_error_index_by_stage[k] = -1;
        metrics->footprint_constraint_error_by_stage_m[k] = NAN;
        metrics->source_footprint_violation_by_stage_m[k] = NAN;
    }
}

static void finalize_step_metrics(MpcRtiPassResult_t *pass,
    MpcRtiMemory_t *memory,
    const MpcRtiCycleConfiguration_t *configuration)
{
    if (!pass || !pass->sqp_step_metrics.valid) return;
    auto &metrics = pass->sqp_step_metrics;
    metrics.accepted = pass->sqp_accepted;
    metrics.line_search_attempted = pass->line_search_attempted;
    metrics.line_search_trials = pass->line_search_trials;
    metrics.line_search_accepted = pass->line_search_accepted;
    metrics.line_search_merit_triggered = pass->line_search_merit_triggered;
    if (isfinite(pass->line_search_selected_alpha))
        metrics.selected_alpha = pass->line_search_selected_alpha;

    /* The agreement-controlled radius is an upper bound on the next SQP
     * correction. It is adjusted only when the caller enables the ratio gate;
     * zero-initialized memory therefore preserves the historical full step. */
    if (!memory || !configuration ||
        configuration->minimum_sqp_model_agreement_ratio <= 0.f)
        return;
    const float before = isfinite(metrics.trust_region_scale_before) &&
        metrics.trust_region_scale_before > 0.f
        ? metrics.trust_region_scale_before
        : (memory->sqp_trust_region_scale > 0.f
            ? memory->sqp_trust_region_scale : 1.f);
    float after = before;
    if (!pass->sqp_accepted) {
        after = fmaxf(.0625f, .5f * before);
    } else {
        const float alpha = isfinite(metrics.selected_alpha)
            ? metrics.selected_alpha : 1.f;
        if (alpha <= 0.f) {
            after = fmaxf(.0625f, .5f * before);
        } else if (alpha < 1.f) {
            after = fminf(after, alpha);
        }
        if (isfinite(metrics.selected_model_agreement_ratio)) {
            if (metrics.selected_model_agreement_ratio <
                    configuration->minimum_sqp_model_agreement_ratio) {
                after = fmaxf(.0625f, .5f * after);
            } else if (metrics.selected_model_agreement_ratio >= .75) {
                after = fminf(1.f, 1.5f * after);
            }
        }
    }
    memory->sqp_trust_region_scale = after;
    metrics.trust_region_scale_before = before;
    metrics.trust_region_scale_after = after;
}

static double polygon_signed_slack_m(
    const RiccatiStatePolygon_t &polygon, double ey, double epsi)
{
    if (polygon.count <= 0 || polygon.count >
            RICCATI_STATE_POLYGON_MAX_PLANES) return NAN;
    double slack = INFINITY;
    for (int i = 0; i < polygon.count; ++i) {
        const double n0 = polygon.normal[i][0];
        const double n1 = polygon.normal[i][1];
        const double norm = std::hypot(n0, n1);
        if (!(norm > 0.) || !std::isfinite(norm)) return NAN;
        slack = std::min(slack, (polygon.upper[i] - n0 * ey - n1 * epsi) / norm);
    }
    return slack;
}

static void measure_qp_source_prediction(
    const MpcRtiProblem_t &problem,
    const MpcRtiNominal_t &linearization,
    const MpcRtiNominal_t &source_plan,
    int source_stage_count,
    const MpcTrajectorySample_t *trajectory,
    size_t trajectory_count,
    double lap_length,
    const MpcRtiConfiguration_t &model,
    MpcRtiSqpStepMetrics_t *metrics)
{
    if (!metrics || !trajectory || source_stage_count < 0) return;
    source_stage_count = std::min(source_stage_count, PREDICTION_HORIZON);
    double x[RICCATI_MAX_NX]{};
    std::copy_n(problem.x0, RICCATI_MAX_NX, x);
    for (int k = 0; k <= source_stage_count; ++k) {
        float exact_state[MPC_RTI_NX]{};
        serialize_state(&source_plan.states[k], exact_state);
        exact_state[MPC_OPT_IDX_PROGRESS_DEVIATION] = static_cast<float>(
            source_plan.progress[k] - linearization.progress[k]);
        exact_state[MPC_OPT_IDX_PREVIOUS_STEERING_COMMAND] =
            source_plan.states[k].plant.delayed_steering_command_2;
        exact_state[MPC_OPT_IDX_PREVIOUS_THROTTLE_COMMAND] =
            source_plan.states[k].plant.delayed_throttle_command_2;
        double maximum = -1.;
        int index = -1;
        for (int i = 0; i < MPC_RTI_NX; ++i) {
            const double error = std::abs(x[i] - exact_state[i]);
            if (error > maximum) { maximum = error; index = i; }
        }
        metrics->state_error_by_stage[k] = static_cast<float>(maximum);
        metrics->state_error_index_by_stage[k] = index;
        ++metrics->state_sample_count;
        if (maximum > metrics->maximum_state_error ||
            !std::isfinite(metrics->maximum_state_error)) {
            metrics->maximum_state_error = maximum;
            metrics->maximum_state_error_stage = k;
            metrics->maximum_state_error_index = index;
        }

        if (k > 0) {
            MpcRtiReference_t source_reference{}, qp_reference{};
            const double qp_progress = linearization.progress[k] +
                x[MPC_OPT_IDX_PROGRESS_DEVIATION];
            const int source_reference_ok = mpc_rti_reference_at_progress(
                trajectory, trajectory_count, lap_length,
                source_plan.progress[k], &model, &source_reference);
            const int qp_reference_ok = mpc_rti_reference_at_progress(
                trajectory, trajectory_count, lap_length,
                qp_progress, &model, &qp_reference);
            MpcModelState_t qp_state{};
            qp_state.e_y = static_cast<float>(x[MPC_OPT_IDX_EY]);
            qp_state.e_psi = static_cast<float>(x[MPC_OPT_IDX_EPSI]);
            float source_lower = NAN, source_upper = NAN;
            double linearized_slack = polygon_signed_slack_m(
                problem.state_polygons[k], x[MPC_OPT_IDX_EY],
                x[MPC_OPT_IDX_EPSI]);
            const int source_slack_ok = source_reference_ok &&
                mpc_rti_compute_corridor_slack(
                    &source_plan.states[k].plant,
                    source_reference.left_bound, source_reference.right_bound,
                    &model, k, &source_lower, &source_upper);
            float qp_lower = NAN, qp_upper = NAN;
            const int qp_slack_ok = qp_reference_ok &&
                mpc_rti_compute_corridor_slack(&qp_state,
                    qp_reference.left_bound, qp_reference.right_bound,
                    &model, k, &qp_lower, &qp_upper);
            if (source_slack_ok) {
                const double source_slack = std::min(source_lower, source_upper);
                const double violation = std::max(0., -source_slack);
                metrics->source_footprint_violation_by_stage_m[k] =
                    static_cast<float>(violation);
                ++metrics->footprint_sample_count;
                if (violation > metrics->maximum_source_footprint_violation_m ||
                    !std::isfinite(metrics->maximum_source_footprint_violation_m)) {
                    metrics->maximum_source_footprint_violation_m = violation;
                    metrics->maximum_source_footprint_violation_stage = k;
                }
                if (qp_slack_ok && std::isfinite(linearized_slack)) {
                    const double exact_qp_slack = std::min(qp_lower, qp_upper);
                    const double error = std::max(
                        std::abs(linearized_slack - exact_qp_slack),
                        std::abs(exact_qp_slack - source_slack));
                    metrics->footprint_constraint_error_by_stage_m[k] =
                        static_cast<float>(error);
                    if (error > metrics->maximum_footprint_constraint_error_m ||
                        !std::isfinite(metrics->maximum_footprint_constraint_error_m)) {
                        metrics->maximum_footprint_constraint_error_m = error;
                        metrics->maximum_footprint_error_stage = k;
                    }
                }
            }
        }

        if (k == source_stage_count) break;
        const RiccatiStepData_t &stage = problem.steps[k];
        const double u[2] = {
            source_plan.controls[k].steering_target_rad -
                (model.command_increment_qp_enabled
                    ? x[MPC_OPT_IDX_PREVIOUS_STEERING_COMMAND] : 0.),
            source_plan.controls[k].throttle_command -
                (model.command_increment_qp_enabled
                    ? x[MPC_OPT_IDX_PREVIOUS_THROTTLE_COMMAND] : 0.)};
        double next[RICCATI_MAX_NX]{};
        for (int i = 0; i < RICCATI_MAX_NX; ++i) {
            next[i] = stage.d[i];
            for (int j = 0; j < RICCATI_MAX_NX; ++j)
                next[i] += stage.A[i][j] * x[j];
            for (int a = 0; a < RICCATI_MAX_NU; ++a)
                next[i] += stage.B[i][a] * u[a];
        }
        std::copy_n(next, RICCATI_MAX_NX, x);
    }
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
    double post_solve_reserve_us,
    MpcRtiSqpCurvature *curvature,
    MpcRtiCycleResult_t *cycle_result,
    MpcRtiPassResult_t *pass)
{
    *pass = {};
    initialize_step_metrics(&pass->sqp_step_metrics);
    /* Every SQP pass must share the cycle's frozen exact-source merit.
     * The caller constructs it once before R1; R2 relinearizes only physical
     * dynamics and geometry. Removing the per-pass schedule-rebuild branch
     * prevents a future caller from silently changing optimization objectives
     * mid-cycle. */
    if (!configuration->objective_schedule ||
        !configuration->objective_schedule->valid ||
        configuration->objective_schedule->horizon != horizon)
        return MPC_RTI_CYCLE_REJECTED_INPUT;
    if (configuration->qp_capture)
        configuration->qp_capture->valid = 0;
    pass->status = MPC_RTI_CYCLE_REJECTED_INPUT;
    pass->sqp_attempted = 1;
    pass->nonlinear_failure_stage = -1;
    pass->nonlinear_failure_reason = -1;
    pass->line_search_full_step_failure_reason = -1;
    pass->line_search_full_step_failure_stage = -1;
    pass->line_search_last_failure_reason = -1;
    pass->line_search_last_failure_stage = -1;
    pass->line_search_nominal_objective = NAN;
    pass->linearization_failure_stage = -1;
    pass->linearization_failure_index = -1;
    pass->admm_status = -1;
    pass->nonlinear_objective = INFINITY;
    pass->minimum_corridor_slack = -INFINITY;
    const double start_us = monotonic_microseconds();
    MpcRtiProblem_t problem = {};
    RiccatiSolution_t solution = {};
    RiccatiStatus_t solver_status = RICCATI_STATUS_ERROR;
    const float previous_rho = memory->solver_state.initialized
        ? memory->solver_state.rho : configuration->solver.rho;
    const float previous_rho_u = memory->solver_state.initialized
        ? memory->solver_state.rho_u : configuration->solver.rho_u;
    RiccatiDebugInfo_t debug = {};
    problem.linearization_failure_stage = -1;
    problem.linearization_failure_index = -1;
    double phase_start_us = monotonic_microseconds();
    const int qp_built = mpc_rti_build_ltv_qp_with_objective_schedule(
        nominal->states, nominal->controls, references, horizon,
        prediction_dt, &configuration->model,
        configuration->objective_schedule, &problem);
    // Preserve Jacobian diagnostics on BOTH successful and failed QP builds.
    // The former implementation copied this entire block twice.
    pass->jacobian_stages_completed = problem.jacobian_stages_completed;
    pass->jacobian_threads_used = problem.jacobian_threads_used;
    pass->jacobian_source_sensitivity_calls =
        problem.jacobian_source_sensitivity_calls;
    pass->jacobian_finite_difference_columns =
        problem.jacobian_finite_difference_columns;
    pass->jacobian_wall_us = problem.jacobian_wall_us;
    pass->jacobian_thread_cpu_us = problem.jacobian_thread_cpu_us;
    pass->jacobian_slowest_stage_us = problem.jacobian_slowest_stage_us;
    memcpy(pass->jacobian_worker_cpu_us, problem.jacobian_worker_cpu_us,
           sizeof(pass->jacobian_worker_cpu_us));
    memcpy(pass->jacobian_worker_stage_wall_us,
           problem.jacobian_worker_stage_wall_us,
           sizeof(pass->jacobian_worker_stage_wall_us));
    memcpy(pass->jacobian_worker_stage_count,
           problem.jacobian_worker_stage_count,
           sizeof(pass->jacobian_worker_stage_count));
    memcpy(pass->jacobian_worker_os_tid,
           problem.jacobian_worker_os_tid,
           sizeof(pass->jacobian_worker_os_tid));
    if (!qp_built) {
        pass->linearization_failure_stage = problem.linearization_failure_stage;
        pass->linearization_failure_reason = problem.linearization_failure_reason;
        pass->linearization_failure_index = problem.linearization_failure_index;
        pass->solve_us = monotonic_microseconds() - start_us;
        return pass->status;
    }
    if(configuration->terminal_viability &&
       !mpc_rti_add_terminal_viability(&problem,nominal,trajectory,
            trajectory_count,lap_length,&configuration->model,
            configuration->objective_schedule)) {
        pass->solve_us=monotonic_microseconds()-start_us;
        return pass->status;
    }
    double source_gradient[PREDICTION_HORIZON][2]{};
    if (!mpc_rti_qp_source_gradient(&problem,nominal->controls,
            configuration->model.command_increment_qp_enabled,source_gradient)) {
        pass->solve_us=monotonic_microseconds()-start_us;
        return pass->status;
    }
    if (configuration->source_informed_sqp && curvature) {
        if (curvature->valid) {
            double predicted[PREDICTION_HORIZON][2]{};
            if(!mpc_rti_qp_source_gradient(&curvature->problem,nominal->controls,
                configuration->model.command_increment_qp_enabled,predicted))
                return pass->status;
            constexpr double scale[2]={.15,.25};
            double step_norm=0.,curvature_residual=0.;
            for(int k=0;k<PREDICTION_HORIZON;++k) {
                const double delta[2]={nominal->controls[k].steering_target_rad-
                    curvature->controls[k].steering_target_rad,
                    nominal->controls[k].throttle_command-
                    curvature->controls[k].throttle_command};
                for(int a=0;a<2;++a) {
                    step_norm+=delta[a]*delta[a]/(scale[a]*scale[a]);
                    curvature_residual+=delta[a]*(source_gradient[k][a]-predicted[k][a]);
                }
            }
            if(step_norm>1.e-12) {
                /* Only missing POSITIVE curvature along the accepted source
                 * step calls for extra damping. The norm of the gradient
                 * error also damps negative/perpendicular changes and was
                 * rejected by the first live source-informed trial. */
                const double secant=fmax(0.,curvature_residual/step_norm);
                double regularization[2];
                for(int a=0;a<2;++a) {
                    regularization[a]=fmin(1.e7,secant/(scale[a]*scale[a]));
                    cycle_result->source_curvature_regularization[a]=regularization[a];
                }
                add_source_curvature(&problem,nominal,
                    configuration->model.command_increment_qp_enabled,regularization);
            }
        } else {
            curvature->problem=problem;
            memcpy(curvature->controls,nominal->controls,sizeof(curvature->controls));
            curvature->valid=1;
        }
    }
    pass->qp_build_us = monotonic_microseconds() - phase_start_us;
    pass->nonsmooth_columns = problem.nonsmooth_jacobian_columns;

    /* After an actual short-prefix fallback, do not expend the remaining
     * deadline on an unsafe racing-reference QP and postpone feasibility
     * restoration to an optional later step. Solve a recovery objective FIRST
     * on the SAME exact source-model Jacobians, without expensive rebuild.
     * QP-only wall slack is NEVER permitted in nonlinear acceptance. */
    const int first_is_restoration =
        configuration->enable_feasibility_restoration &&
        configuration->prefer_feasibility_first;
    if (first_is_restoration) {
        if (!mpc_rti_prepare_feasibility_restoration_qp(&problem, nominal,
                configuration->feasibility_restoration_qp_relaxation_m,
                configuration->feasibility_restoration_wall_weight)) {
            pass->status = MPC_RTI_CYCLE_REJECTED_INPUT;
            pass->solve_us = monotonic_microseconds() - start_us;
            return pass->status;
        }
        pass->restoration_attempted = 1;
    }
    RiccatiAdmmConfig_t solver_config = configuration->solver;
    solver_config.terminal_constraints=problem.terminal_constraints;
    memcpy(solver_config.state_polygons, problem.state_polygons,
        sizeof(problem.state_polygons));
    if (cycle_deadline_us > 0.0) {
        const double solver_budget_us = cycle_deadline_us -
            monotonic_microseconds() - post_solve_reserve_us;
        if (solver_budget_us <= 0.0) {
            pass->status = MPC_RTI_CYCLE_WAITING_FOR_SOLVER_BUDGET;
            pass->solver_skipped_for_deadline = 1;
            pass->solve_us = monotonic_microseconds() - start_us;
            return pass->status;
        }
        if (solver_config.max_solve_time_us <= 0.0 ||
            solver_config.max_solve_time_us > solver_budget_us)
            solver_config.max_solve_time_us = solver_budget_us;
    }

    const int epoch_shift=configuration->command_epoch_ns>0
        ? mpc_rti_command_epoch_shift(memory->command_epoch_ns,
            configuration->command_epoch_ns,prediction_dt) : 1;
    mpc_rti_rebase_solver_warm_start(memory,nominal,problem.x0,&solver_config,
        epoch_shift<0 ? 0 : epoch_shift);

    MpcRtiQpCapture_t * const qp_capture = configuration->qp_capture;
    if (qp_capture) {
        /* This is deliberately a fixed-size memory snapshot only. The
         * production solver never performs serialization or file I/O. */
        qp_capture->problem = problem;
        qp_capture->solver_config = solver_config;
        qp_capture->initial_solver_state = memory->solver_state;
        qp_capture->capture_count += 1;
    }

    phase_start_us = monotonic_microseconds();
    const double admm_thread_cpu_start_us = thread_cpu_microseconds();
    float feedback[RICCATI_HORIZON][RICCATI_MAX_NU][RICCATI_MAX_NX];
    solver_status = riccati_admm_solve_with_feedback(
        problem.steps, problem.terminal_Q, problem.terminal_q,
        problem.terminal_x_lb, problem.terminal_x_ub,
        problem.x0, &solver_config, &memory->solver_state, &solution,
        qp_capture ? qp_capture->raw_primal_u : NULL,
        configuration->feedback_rollout ? feedback : NULL);
    if (qp_capture) {
        qp_capture->final_solver_state = memory->solver_state;
        qp_capture->raw_solution = solution;
        qp_capture->raw_status = solver_status;
        qp_capture->valid = 1;
    }
    pass->admm_us = monotonic_microseconds() - phase_start_us;
    const double admm_thread_cpu_finish_us = thread_cpu_microseconds();
    pass->admm_thread_cpu_us =
        isfinite(admm_thread_cpu_start_us) &&
        isfinite(admm_thread_cpu_finish_us)
        ? fmax(0.0,admm_thread_cpu_finish_us-admm_thread_cpu_start_us)
        : NAN;
    if (first_is_restoration) {
        pass->restoration_us = pass->admm_us;
        pass->restoration_iterations = solution.iterations;
        pass->restoration_primal_residual = solution.primal_residual;
    }
    riccati_debug_get_last(&debug);
    if (qp_capture) qp_capture->debug_info = debug;
    pass->rho_final = debug.rho;
    pass->rho_u_final = debug.rho_u;
    pass->maximum_regularization = debug.max_control_hessian_regularization;
    pass->regularization_count = debug.control_hessian_regularization_count;
    pass->rho_change_count = debug.rho_change_count;
    pass->factorization_count = debug.quadratic_factorization_count;
    pass->factorization_time_ns = debug.quadratic_factorization_time_ns;
    pass->solver_cold_start_initialization_time_ns =
        debug.cold_start_initialization_time_ns;
    pass->riccati_rhs_time_ns = debug.quadratic_rhs_time_ns;
    pass->admm_projection_time_ns = debug.projection_residual_time_ns;
    pass->rho_update_time_ns = debug.rho_update_time_ns;
    pass->solver_scaling_time_ns = debug.scaling_transform_time_ns;
    pass->solver_output_unscale_time_ns = debug.output_unscale_time_ns;

    pass->nonsmooth_columns = problem.nonsmooth_jacobian_columns;
    pass->admm_status = solver_status;
    pass->solver_time_budget_exhausted = solution.time_budget_exhausted;
    pass->sqp_completed =
        solver_status == RICCATI_STATUS_OPTIMAL ||
        solver_status == RICCATI_STATUS_MAX_ITERATIONS;
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
        !isfinite(solution.dual_residual) ||
        (solver_status != RICCATI_STATUS_OPTIMAL &&
         solver_status != RICCATI_STATUS_MAX_ITERATIONS)) {
        pass->status = MPC_RTI_CYCLE_REJECTED_SOLVER;
        pass->solve_us = monotonic_microseconds() - start_us;
        return pass->status;
    }
    MpcModelControl_t candidate_controls[PREDICTION_HORIZON];
    memcpy(candidate_controls, nominal->controls,
        sizeof(candidate_controls));
    // Map Riccati's decision coordinates to the native Unity controls.
    // In increment coordinates, the QP solves for physically meaningful
    // local changes, while the source plant always receives ABSOLUTE
    // commands. Reconstruct sequentially from the measured command history;
    // do not trust an iteration-limited ADMM auxiliary x to exactly equal
    // the preceding command before nonlinear validation.
    float previous_steering=problem.x0[MPC_OPT_IDX_PREVIOUS_STEERING_COMMAND];
    float previous_throttle=problem.x0[MPC_OPT_IDX_PREVIOUS_THROTTLE_COMMAND];
    for (int k = 0; k < RICCATI_HORIZON; ++k) {
        const RiccatiStepData_t *command_stage = &problem.steps[k];
        if (configuration->model.command_increment_qp_enabled) {
            const float ds=clampf_rti(solution.u[k][0],
                command_stage->u_lb[0],command_stage->u_ub[0]);
            const float dt=clampf_rti(solution.u[k][1],
                command_stage->u_lb[1],command_stage->u_ub[1]);
            candidate_controls[k].steering_target_rad=clampf_rti(
                previous_steering+ds,
                -configuration->model.max_steering_rad,
                configuration->model.max_steering_rad);
            candidate_controls[k].throttle_command=clampf_rti(
                previous_throttle+dt,MPC_THROTTLE_MIN_DRIVE,MPC_THROTTLE_MAX);
            previous_steering=candidate_controls[k].steering_target_rad;
            previous_throttle=candidate_controls[k].throttle_command;
        } else {
            candidate_controls[k].steering_target_rad=clampf_rti(
                solution.u[k][0],command_stage->u_lb[0],command_stage->u_ub[0]);
            candidate_controls[k].throttle_command=clampf_rti(
                solution.u[k][1],command_stage->u_lb[1],command_stage->u_ub[1]);
        }
    }
    MpcModelControl_t unscaled_qp_controls[PREDICTION_HORIZON];
    memcpy(unscaled_qp_controls, candidate_controls,
        sizeof(unscaled_qp_controls));
    const double nominal_qp_objective = qp_objective_for_controls(
        problem, nominal->controls,
        configuration->model.command_increment_qp_enabled);
    const double unscaled_proposed_qp_objective = qp_objective_for_controls(
        problem, unscaled_qp_controls,
        configuration->model.command_increment_qp_enabled);
    float trust_region_scale = 1.f;
    if (configuration->minimum_sqp_model_agreement_ratio > 0.f && memory &&
        memory->sqp_trust_region_scale > 0.f &&
        isfinite(memory->sqp_trust_region_scale))
        trust_region_scale = clampf_rti(memory->sqp_trust_region_scale,
            .0625f, 1.f);
    if (configuration->minimum_sqp_model_agreement_ratio > 0.f) {
        for (int k = 0; k < RICCATI_HORIZON; ++k) {
            candidate_controls[k].steering_target_rad = clampf_rti(
                nominal->controls[k].steering_target_rad + trust_region_scale *
                    (unscaled_qp_controls[k].steering_target_rad -
                     nominal->controls[k].steering_target_rad),
                -configuration->model.max_steering_rad,
                configuration->model.max_steering_rad);
            candidate_controls[k].throttle_command = clampf_rti(
                nominal->controls[k].throttle_command + trust_region_scale *
                    (unscaled_qp_controls[k].throttle_command -
                     nominal->controls[k].throttle_command),
                MPC_THROTTLE_MIN_DRIVE, MPC_THROTTLE_MAX);
        }
    }
    MpcModelControl_t qp_controls[PREDICTION_HORIZON];
    memcpy(qp_controls, candidate_controls, sizeof(qp_controls));
    const double proposed_qp_objective = qp_objective_for_controls(
        problem, qp_controls,
        configuration->model.command_increment_qp_enabled);
    cycle_result->source_directional_derivative=0.;
    for(int k=0;k<PREDICTION_HORIZON;++k) {
        cycle_result->source_directional_derivative += source_gradient[k][0]*
            (candidate_controls[k].steering_target_rad-nominal->controls[k].steering_target_rad)+
            source_gradient[k][1]*(candidate_controls[k].throttle_command-
                nominal->controls[k].throttle_command);
    }
    pass->sqp_step_metrics.valid = 1;
    pass->sqp_step_metrics.qp_predicted_cost_reduction =
        nominal_qp_objective - proposed_qp_objective;
    pass->sqp_step_metrics.unscaled_qp_predicted_cost_reduction =
        nominal_qp_objective - unscaled_proposed_qp_objective;
    pass->sqp_step_metrics.trust_region_scale_before = trust_region_scale;
    pass->sqp_step_metrics.comparison_includes_feedback =
        configuration->feedback_rollout &&
        !configuration->capture_sqp_prediction_metrics;
    MpcRtiRolloutPolicy_t policy;
    if (configuration->feedback_rollout) {
        if (configuration->model.command_increment_qp_enabled &&
            configuration->feedback_rollout_gain_scale>0.f) {
            // v = command increment, u_absolute = p + v.
            // The sensitivity to the measured previous absolute command
            // therefore includes +1 even with arbitrary feedback gain scale.
            for (int k=0;k<RICCATI_HORIZON;++k) {
                feedback[k][0][MPC_OPT_IDX_PREVIOUS_STEERING_COMMAND]+=
                    1.f/configuration->feedback_rollout_gain_scale;
                feedback[k][1][MPC_OPT_IDX_PREVIOUS_THROTTLE_COMMAND]+=
                    1.f/configuration->feedback_rollout_gain_scale;
            }
        }
        initialize_rollout_policy(&policy, &problem, nominal, candidate_controls,
            (const float (*)[RICCATI_MAX_NU][RICCATI_MAX_NX])feedback,
            configuration->model.command_increment_qp_enabled);
        policy.gain_scale = configuration->feedback_rollout_gain_scale;
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

    pass->candidate.horizon = horizon;
    pass->candidate.speed_reference_valid = nominal->speed_reference_valid;
    memcpy(pass->candidate.speed_reference, nominal->speed_reference,
        sizeof(pass->candidate.speed_reference));
    MpcRtiState_t candidate_states[PREDICTION_HORIZON + 1];
    double candidate_progress[PREDICTION_HORIZON + 1];
    phase_start_us = monotonic_microseconds();
    MpcRtiRolloutFailure_t full_step_failure = {};
    const MpcRtiRolloutStatus_t rollout_status =
        rollout_candidate_impl(
        current_state, current_progress, candidate_controls, trajectory,
        trajectory_count, lap_length, references, nominal->progress,
        &pass->path_delta, horizon, prediction_dt, &configuration->model,
        candidate_states, candidate_progress,
        configuration->capture_diagnostic_outputs
            ? pass->diagnostic_outputs : NULL,
        configuration->capture_diagnostic_outputs
            ? &pass->diagnostic_output_count : NULL,
        &pass->nonlinear_failure_stage, &full_step_failure,
        configuration->feedback_rollout ? &policy : NULL, candidate_controls);
    memcpy(pass->proposed_controls, candidate_controls,
        (size_t)horizon * sizeof(candidate_controls[0]));
    memcpy(pass->candidate.controls, candidate_controls,
        (size_t)horizon * sizeof(candidate_controls[0]));
    pass->first_action = candidate_controls[0];
    pass->exact_rollout_us = monotonic_microseconds() - phase_start_us;
    MpcRtiRolloutStatus_t candidate_evaluation_failure=MPC_RTI_ROLLOUT_INVALID_MODEL;
    int candidate_valid = rollout_status == MPC_RTI_ROLLOUT_OK;
    double nominal_objective = NAN;
    float nominal_minimum_corridor_slack = NAN;
    float nominal_lateral_accel_proxy = NAN;
    int nominal_lateral_accel_proxy_stage = -1;
    float nominal_lateral_accel_proxy_by_stage[PREDICTION_HORIZON + 1] = {};
    if (candidate_valid) {
        pass->candidate.valid = 1;
        memcpy(pass->candidate.states, candidate_states,
            (size_t)(horizon + 1) * sizeof(candidate_states[0]));
        memcpy(pass->candidate.progress, candidate_progress,
            (size_t)(horizon + 1) * sizeof(candidate_progress[0]));
        phase_start_us = monotonic_microseconds();
        candidate_valid = evaluate_nonlinear_candidate(&pass->candidate,
            trajectory, trajectory_count, lap_length, configuration,
            &pass->nonlinear_objective, &pass->minimum_corridor_slack,
            &pass->lateral_accel_proxy, &pass->lateral_accel_proxy_stage,
            pass->lateral_accel_proxy_by_stage,
            &pass->path_delta,&candidate_evaluation_failure);
        pass->nonlinear_validation_us =
            monotonic_microseconds() - phase_start_us;
    }
    pass->sqp_step_metrics.source_exact_rollout_status =
        rollout_status == MPC_RTI_ROLLOUT_OK
            ? (candidate_valid ? MPC_RTI_ROLLOUT_OK
                               : candidate_evaluation_failure)
            : rollout_status;
    pass->sqp_step_metrics.source_exact_failure_stage =
        pass->nonlinear_failure_stage;
    pass->sqp_step_metrics.source_exact_rollout_complete = candidate_valid;
    int merit_regression = 0;
    phase_start_us = monotonic_microseconds();
    pass->line_search_nominal_exact_valid =
        evaluate_nonlinear_candidate(nominal, trajectory,
            trajectory_count, lap_length, configuration,
            &nominal_objective, &nominal_minimum_corridor_slack,
            &nominal_lateral_accel_proxy,
            &nominal_lateral_accel_proxy_stage,
            nominal_lateral_accel_proxy_by_stage,NULL);
    pass->nonlinear_validation_us +=
        monotonic_microseconds() - phase_start_us;
    if (pass->line_search_nominal_exact_valid) {
        pass->line_search_nominal_objective = nominal_objective;
        if (candidate_valid) {
            const double actual_reduction = nominal_objective -
                pass->nonlinear_objective;
            const double applied_ratio = agreement_ratio(
                pass->sqp_step_metrics.qp_predicted_cost_reduction,
                actual_reduction);
            pass->sqp_step_metrics.realized_source_exact_cost_reduction =
                actual_reduction;
            pass->sqp_step_metrics.realized_model_agreement_ratio = applied_ratio;
            merit_regression = !first_is_restoration &&
                (pass->nonlinear_objective > nominal_objective ||
                 (configuration->minimum_sqp_model_agreement_ratio > 0.f &&
                  applied_ratio < configuration->minimum_sqp_model_agreement_ratio));
            pass->line_search_merit_triggered = merit_regression;
        }
    }

    /* The ordinary rollout is the exact trajectory of the command that the
     * controller would use (including its optional feedback correction). In
     * diagnostic mode, separately roll the raw QP proposal through the exact
     * source plant with wide map bounds so a corridor rejection still yields
     * complete cost/state evidence. The hard original-corridor result above
     * remains the only acceptance authority. */
    MpcRtiNominal_t raw_source_plan{};
    const MpcRtiNominal_t *measured_source_plan = nullptr;
    int measured_source_stage_count = candidate_valid ? horizon :
        (pass->path_delta.sample_count > 0
            ? pass->path_delta.sample_count - 1 : 0);
    if (configuration->capture_sqp_prediction_metrics) {
        std::vector<MpcTrajectorySample_t> wide_trajectory(
            trajectory, trajectory + trajectory_count);
        for (auto &sample : wide_trajectory)
            sample.left_bound = sample.right_bound = 100.f;
        MpcRtiCycleConfiguration_t metric_configuration = *configuration;
        metric_configuration.model.world_sweep_check = nullptr;
        metric_configuration.model.world_sweep_context = nullptr;
        metric_configuration.model.world_map_rejection_only = 0;
        MpcRtiRolloutStatus_t source_status = MPC_RTI_ROLLOUT_INVALID_INPUT;
        int source_failure_stage = -1;
        source_status = mpc_rti_validate_candidate_plan(
            current_state, current_progress, qp_controls,
            wide_trajectory.data(), wide_trajectory.size(), lap_length,
            horizon, prediction_dt, &metric_configuration,
            &raw_source_plan, &source_failure_stage);
        memcpy(raw_source_plan.controls, qp_controls,
            (size_t)horizon * sizeof(qp_controls[0]));
        pass->sqp_step_metrics.source_exact_rollout_status = source_status;
        pass->sqp_step_metrics.source_exact_failure_stage = source_failure_stage;
        pass->sqp_step_metrics.source_exact_rollout_complete =
            source_status == MPC_RTI_ROLLOUT_OK;
        measured_source_stage_count = source_status == MPC_RTI_ROLLOUT_OK
            ? horizon : std::clamp(source_failure_stage + 1, 0, horizon);
        if (measured_source_stage_count > 0 || source_status == MPC_RTI_ROLLOUT_OK)
            measured_source_plan = &raw_source_plan;

        double source_nominal_objective = NAN;
        double source_proposal_objective = NAN;
        float metric_clearance = NAN;
        float metric_lateral = NAN;
        int metric_lateral_stage = -1;
        float metric_lateral_by_stage[PREDICTION_HORIZON + 1]{};
        /* Keep the original frozen objective and corridor references for the
         * cost comparison. The widened trajectory is used only to continue
         * the physical rollout after a rejected footprint proposal. The
         * metric evaluator may score that rejected point outside the hard
         * corridor, but the production acceptance path remains unchanged. */
        const int source_nominal_scored = evaluate_nonlinear_candidate(
            nominal, trajectory, trajectory_count, lap_length,
            &metric_configuration, &source_nominal_objective,
            &metric_clearance, &metric_lateral, &metric_lateral_stage,
            metric_lateral_by_stage, nullptr, nullptr, 1);
        const int source_proposal_scored = source_status == MPC_RTI_ROLLOUT_OK &&
            evaluate_nonlinear_candidate(&raw_source_plan, trajectory,
                trajectory_count, lap_length, &metric_configuration,
                &source_proposal_objective, &metric_clearance, &metric_lateral,
                &metric_lateral_stage, metric_lateral_by_stage, nullptr,
                nullptr, 1);
        if (source_nominal_scored && source_proposal_scored) {
            pass->sqp_step_metrics.source_exact_cost_reduction =
                source_nominal_objective - source_proposal_objective;
            pass->sqp_step_metrics.model_agreement_ratio = agreement_ratio(
                pass->sqp_step_metrics.qp_predicted_cost_reduction,
                pass->sqp_step_metrics.source_exact_cost_reduction);
        }
    } else if (pass->line_search_nominal_exact_valid && candidate_valid) {
        pass->sqp_step_metrics.source_exact_rollout_complete = 1;
        pass->sqp_step_metrics.source_exact_rollout_status = MPC_RTI_ROLLOUT_OK;
        pass->sqp_step_metrics.source_exact_cost_reduction =
            pass->sqp_step_metrics.realized_source_exact_cost_reduction;
        pass->sqp_step_metrics.model_agreement_ratio =
            pass->sqp_step_metrics.realized_model_agreement_ratio;
        measured_source_plan = &pass->candidate;
        measured_source_stage_count = horizon;
    } else {
        /* Preserve stagewise model-error evidence even when the full proposal
         * exits the source plant's hard corridor before reaching the horizon.
         * rollout_candidate_impl has filled states/progress through the first
         * failing transition; these partial diagnostics never authorize the
         * action and do not pretend to have a complete-horizon cost. */
        MpcRtiNominal_t partial_source_plan{};
        partial_source_plan.valid = 1;
        partial_source_plan.horizon = horizon;
        memcpy(partial_source_plan.controls, candidate_controls,
            (size_t)horizon * sizeof(candidate_controls[0]));
        memcpy(partial_source_plan.states, candidate_states,
            (size_t)(horizon + 1) * sizeof(candidate_states[0]));
        memcpy(partial_source_plan.progress, candidate_progress,
            (size_t)(horizon + 1) * sizeof(candidate_progress[0]));
        measured_source_stage_count = rollout_status == MPC_RTI_ROLLOUT_OK
            ? horizon
            : std::clamp(pass->nonlinear_failure_stage + 1, 0, horizon);
        measured_source_plan = &partial_source_plan;
        measure_qp_source_prediction(problem, *nominal,
            *measured_source_plan, measured_source_stage_count,
            trajectory, trajectory_count, lap_length,
            configuration->model, &pass->sqp_step_metrics);
        /* The partial plan is stack-backed, so metrics are copied above and
         * must not be dereferenced later. */
        measured_source_plan = nullptr;
    }
    if (measured_source_plan) {
        measure_qp_source_prediction(problem, *nominal,
            *measured_source_plan, measured_source_stage_count,
            trajectory, trajectory_count, lap_length,
            configuration->model, &pass->sqp_step_metrics);
    } else {
        pass->sqp_step_metrics.state_sample_count = 0;
        pass->sqp_step_metrics.footprint_sample_count = 0;
    }
    if (candidate_valid && !merit_regression) {
        pass->line_search_selected_alpha = 1.0f;
        if (pass->line_search_nominal_exact_valid) {
            pass->sqp_step_metrics.selected_qp_predicted_cost_reduction =
                pass->sqp_step_metrics.qp_predicted_cost_reduction;
            pass->sqp_step_metrics.selected_source_exact_cost_reduction =
                nominal_objective - pass->nonlinear_objective;
            pass->sqp_step_metrics.selected_model_agreement_ratio =
                pass->sqp_step_metrics.realized_model_agreement_ratio;
        }
        pass->status = solver_status == RICCATI_STATUS_OPTIMAL
            ? MPC_RTI_CYCLE_ACCEPTED_OPTIMAL
            : MPC_RTI_CYCLE_ACCEPTED_FEASIBLE;
        pass->sqp_accepted = 1;
        if (first_is_restoration) {
            pass->restoration_accepted = 1;
            riccati_admm_state_init(&memory->solver_state);
        }
        finalize_step_metrics(pass, memory, configuration);
        pass->solve_us = monotonic_microseconds() - start_us;
        return pass->status;
    }

    pass->candidate.valid = 0;
    if (!candidate_valid) {
        pass->nonlinear_failure = full_step_failure;
        pass->nonlinear_failure_reason = rollout_status == MPC_RTI_ROLLOUT_OK
            ? candidate_evaluation_failure : (int)rollout_status;
        pass->line_search_full_step_failure_reason =
            pass->nonlinear_failure_reason;
        pass->line_search_full_step_failure_stage =
            pass->nonlinear_failure_stage;
    }

    /* Exact-source globalization must recover from a full-QP proposal that
     * leaves the simulator's differentiable/valid plant domain. This is NOT
     * permission to publish an invalid command: every damped candidate goes
     * through the full nonlinear source and hard corridor validation before
     * acceptance. A valid nominal provides a trustworthy physical endpoint
     * of the interpolation. Previously INVALID_MODEL caused an immediate
     * rejected cycle without trying any intermediate, potentially valid
     * physical trajectory, even when ADMM had converged correctly. */
    const int corridor_backtrack = !candidate_valid &&
        rollout_status == MPC_RTI_ROLLOUT_CORRIDOR;
    const int should_backtrack = mpc_rti_line_search_should_backtrack(
        rollout_status==MPC_RTI_ROLLOUT_OK && !candidate_valid
            ? candidate_evaluation_failure : rollout_status, candidate_valid,
            pass->line_search_nominal_exact_valid, merit_regression);
    if (!should_backtrack || !configuration->enable_nonlinear_line_search) {
        pass->status = MPC_RTI_CYCLE_REJECTED_NONLINEAR_ROLLOUT;
        finalize_step_metrics(pass, memory, configuration);
        pass->solve_us = monotonic_microseconds() - start_us;
        return pass->status;
    }

    if (cycle_deadline_us > 0.0 &&
        !configuration->line_search_command_epoch_valid) {
        pass->line_search_skipped_command_epoch = 1;
        pass->status = MPC_RTI_CYCLE_REJECTED_NONLINEAR_ROLLOUT;
        finalize_step_metrics(pass, memory, configuration);
        pass->solve_us = monotonic_microseconds() - start_us;
        return pass->status;
    }

    /* Keep the bounded 4-step exact-validated globalization as an
     * interim safety net while replacing the underlying first-order
     * finite-step model mismatch. 6/8-step experiments reduced literal
     * alpha=0 but worsened tracking and slow-lap tails; do not restore them
     * as a substitute for genuinely improving the nonlinear SQP direction. */
    static constexpr float backtrack_alpha[4] = {
        0.5f, 0.25f, 0.125f, 0.0625f};
    const double line_search_start_us = monotonic_microseconds();
    MpcRtiNominal_t best_trial_candidate = {};
    MpcRtiCandidatePathDelta_t best_trial_path_delta = {};
    MpcRtiDiagnosticStageOutput_t best_trial_diagnostic_outputs[
        MPC_RTI_DIAGNOSTIC_SAMPLE_COUNT]{};
    int best_trial_diagnostic_output_count = 0;
    double best_trial_objective = INFINITY;
    float best_trial_minimum_corridor_slack = -INFINITY;
    float best_trial_lateral_accel_proxy = 0.0f;
    int best_trial_lateral_accel_proxy_stage = -1;
    float best_trial_lateral_accel_proxy_by_stage[
        PREDICTION_HORIZON + 1] = {};
    float best_trial_alpha = 0.0f;
    double best_trial_predicted_reduction = NAN;
    double best_trial_actual_reduction = NAN;
    double best_trial_agreement = NAN;
    int best_trial_valid = 0;
    /* If BOTH endpoints of the backtracking segment are physically outside
     * the corridor, spending several exact source rollouts on smaller alphas
     * consumes the time needed to find a different, feasible steering and
     * torque direction. In opt-in restoration mode, go straight to the
     * same-Jacobian feasibility-biased solve; the default policy is untouched.
     * This is intentionally conservative: we skip backtracking ONLY when the
     * original candidate and nominal are both physically invalid. */
    const int restoration_priority =
        configuration->enable_feasibility_restoration &&
        !first_is_restoration && corridor_backtrack &&
        !pass->line_search_nominal_exact_valid;
    /* For a feasibility-FIRST solve, both endpoints can be physically bad
     * while a damped step between them is valid (nonlinear tire dynamics,
     * yaw-dependent footprint). Test intermediate alphas through the exact
     * 750ms physical gate instead of discarding the converged direction. */
    for (int trial_index = 0;
         trial_index < configuration->line_search_max_backtracks &&
         !restoration_priority;
         ++trial_index) {
        if (cycle_deadline_us > 0.0) {
            /* Backtracking is incremental. Reserve 2.2 ms for one trial and
             * completion, then re-check before every later trial. Do not
             * multiply the per-trial reserve: that needlessly skips valid
             * recovery steps when one or two measured backtracks fit before
             * the hard publication gate. */
            const double remaining_us = cycle_deadline_us -
                monotonic_microseconds();
            /* With feasibility restoration enabled, do not begin another
             * full source-plant backtrack when only ~2.2ms remain. Captured
             * 10-lap practice cycles spent up to 14.25ms across exact
             * rollouts and missed the 20ms publication gate by 0.24-0.54ms.
             * Reserve 3.5ms for unpredictable native-plant branch cost and
             * command publication; normal baseline thresholds are unchanged. */
            if (!mpc_rti_line_search_budget_available(remaining_us) ||
                (configuration->enable_feasibility_restoration &&
                 remaining_us < 3500.0)) {
                pass->line_search_skipped_budget = 1;
                break;
            }
        }

        const float alpha = backtrack_alpha[trial_index];
        if (configuration->feedback_rollout) policy.alpha = alpha;
        MpcModelControl_t trial_controls[PREDICTION_HORIZON];
        memcpy(trial_controls, nominal->controls,
            (size_t)horizon * sizeof(trial_controls[0]));
        for (int k = 0; k < RICCATI_HORIZON && k < horizon; ++k) {
            const float nominal_steering = nominal->controls[k].steering_target_rad;
            const float nominal_throttle = nominal->controls[k].throttle_command;
            trial_controls[k].steering_target_rad = clampf_rti(
                nominal_steering + alpha *
                    (qp_controls[k].steering_target_rad -
                     nominal_steering),
                -configuration->model.max_steering_rad,
                configuration->model.max_steering_rad);
            trial_controls[k].throttle_command = clampf_rti(
                nominal_throttle + alpha *
                    (qp_controls[k].throttle_command -
                     nominal_throttle),
                MPC_THROTTLE_MIN_DRIVE, MPC_THROTTLE_MAX);
        }
        MpcModelControl_t qp_trial_controls[PREDICTION_HORIZON];
        memcpy(qp_trial_controls, trial_controls, sizeof(qp_trial_controls));

        MpcRtiCandidatePathDelta_t trial_path_delta = {};
        MpcRtiRolloutFailure_t trial_failure = {};
        int trial_failure_stage = -1;
        pass->candidate.valid = 0;
        phase_start_us = monotonic_microseconds();
        const MpcRtiRolloutStatus_t trial_rollout_status =
            rollout_candidate_impl(current_state, current_progress,
                trial_controls, trajectory, trajectory_count, lap_length,
                references, nominal->progress, &trial_path_delta, horizon,
                prediction_dt, &configuration->model, candidate_states,
                candidate_progress,
                configuration->capture_diagnostic_outputs
                    ? pass->diagnostic_outputs : NULL,
                configuration->capture_diagnostic_outputs
                    ? &pass->diagnostic_output_count : NULL,
                &trial_failure_stage, &trial_failure,
                configuration->feedback_rollout ? &policy : NULL, trial_controls);
        const double trial_rollout_us =
            monotonic_microseconds() - phase_start_us;
        pass->exact_rollout_us += trial_rollout_us;
        pass->line_search_attempted = 1;
        ++pass->line_search_trials;
        pass->line_search_last_failure_reason = (int)trial_rollout_status;
        pass->line_search_last_failure_stage = trial_failure_stage;

        int trial_exact_valid = trial_rollout_status == MPC_RTI_ROLLOUT_OK;
        if (trial_exact_valid) {
            pass->candidate.horizon = horizon;
            pass->candidate.speed_reference_valid =
                nominal->speed_reference_valid;
            memcpy(pass->candidate.speed_reference,
                nominal->speed_reference, sizeof(pass->candidate.speed_reference));
            memcpy(pass->candidate.controls, trial_controls,
                (size_t)horizon * sizeof(trial_controls[0]));
            memcpy(pass->candidate.states, candidate_states,
                (size_t)(horizon + 1) * sizeof(candidate_states[0]));
            memcpy(pass->candidate.progress, candidate_progress,
                (size_t)(horizon + 1) * sizeof(candidate_progress[0]));
            pass->candidate.valid = 1;
            phase_start_us = monotonic_microseconds();
            trial_exact_valid = evaluate_nonlinear_candidate(&pass->candidate,
                trajectory, trajectory_count, lap_length, configuration,
                &pass->nonlinear_objective,
                &pass->minimum_corridor_slack,
                &pass->lateral_accel_proxy,
                &pass->lateral_accel_proxy_stage,
                pass->lateral_accel_proxy_by_stage,
                &trial_path_delta);
            pass->nonlinear_validation_us +=
                monotonic_microseconds() - phase_start_us;
            if (!trial_exact_valid) {
                pass->candidate.valid = 0;
                pass->line_search_last_failure_reason =
                    MPC_RTI_ROLLOUT_INVALID_MODEL;
                pass->line_search_last_failure_stage = -1;
            }
        }

        const double trial_actual_reduction =
            pass->line_search_nominal_exact_valid && trial_exact_valid
                ? nominal_objective - pass->nonlinear_objective : NAN;
        const double trial_predicted_reduction =
            nominal_qp_objective - qp_objective_for_controls(
                problem, qp_trial_controls,
                configuration->model.command_increment_qp_enabled);
        const double trial_agreement = agreement_ratio(
            trial_predicted_reduction, trial_actual_reduction);
        const int trial_agreement_ok =
            configuration->minimum_sqp_model_agreement_ratio <= 0.f ||
            trial_agreement >=
                configuration->minimum_sqp_model_agreement_ratio;
        if (mpc_rti_refinement_improves_action(
                pass->line_search_nominal_exact_valid, nominal_objective,
                trial_exact_valid, pass->nonlinear_objective) &&
            trial_agreement_ok) {
            best_trial_candidate = pass->candidate;
            best_trial_candidate.validation_objective =
                pass->nonlinear_objective;
            best_trial_candidate.validation_minimum_corridor_slack_m =
                pass->minimum_corridor_slack;
            best_trial_path_delta = trial_path_delta;
            best_trial_objective = pass->nonlinear_objective;
            best_trial_minimum_corridor_slack =
                pass->minimum_corridor_slack;
            best_trial_lateral_accel_proxy = pass->lateral_accel_proxy;
            best_trial_lateral_accel_proxy_stage =
                pass->lateral_accel_proxy_stage;
            memcpy(best_trial_lateral_accel_proxy_by_stage,
                pass->lateral_accel_proxy_by_stage,
                sizeof(best_trial_lateral_accel_proxy_by_stage));
            best_trial_alpha = alpha;
            best_trial_predicted_reduction = trial_predicted_reduction;
            best_trial_actual_reduction = trial_actual_reduction;
            best_trial_agreement = trial_agreement;
            best_trial_valid = 1;
            best_trial_diagnostic_output_count =
                pass->diagnostic_output_count;
            if (configuration->capture_diagnostic_outputs)
                memcpy(best_trial_diagnostic_outputs,
                    pass->diagnostic_outputs,
                    sizeof(best_trial_diagnostic_outputs));
            /* Stop at the first decreasing exact-valid backtrack. Feasibility
             * alone must not accept an increase that triggered merit search.
             * If no such trial fits the budget, the independently validated
             * alpha=0 nominal remains available below. */
            break;
        }

        if (!mpc_rti_line_search_should_continue(
                trial_rollout_status,
                pass->line_search_nominal_exact_valid))
            break;
    }

    if (best_trial_valid) {
        pass->candidate = best_trial_candidate;
        pass->path_delta = best_trial_path_delta;
        pass->nonlinear_objective = best_trial_objective;
        pass->minimum_corridor_slack = best_trial_minimum_corridor_slack;
        pass->lateral_accel_proxy = best_trial_lateral_accel_proxy;
        pass->lateral_accel_proxy_stage =
            best_trial_lateral_accel_proxy_stage;
        memcpy(pass->lateral_accel_proxy_by_stage,
            best_trial_lateral_accel_proxy_by_stage,
            sizeof(pass->lateral_accel_proxy_by_stage));
        pass->first_action = best_trial_candidate.controls[0];
        pass->first_action_valid = 1;
        pass->line_search_accepted = 1;
        pass->line_search_selected_alpha = best_trial_alpha;
        pass->sqp_step_metrics.selected_qp_predicted_cost_reduction =
            best_trial_predicted_reduction;
        pass->sqp_step_metrics.selected_source_exact_cost_reduction =
            best_trial_actual_reduction;
        pass->sqp_step_metrics.selected_model_agreement_ratio =
            best_trial_agreement;
        pass->line_search_last_failure_reason = -1;
        pass->line_search_last_failure_stage = -1;
        pass->diagnostic_output_count = best_trial_diagnostic_output_count;
        if (configuration->capture_diagnostic_outputs)
            memcpy(pass->diagnostic_outputs, best_trial_diagnostic_outputs,
                sizeof(pass->diagnostic_outputs));
        pass->line_search_us = monotonic_microseconds() -
            line_search_start_us;
        pass->sqp_accepted = 1;
        if (first_is_restoration) {
            pass->restoration_accepted = 1;
            riccati_admm_state_init(&memory->solver_state);
        }
        pass->status = MPC_RTI_CYCLE_ACCEPTED_LINE_SEARCH;
        finalize_step_metrics(pass, memory, configuration);
        pass->solve_us = monotonic_microseconds() - start_us;
        return pass->status;
    }

    /* The nominal sequence is itself a safe action only after this exact
     * objective/corridor evaluation. Retaining it is alpha=0 backtracking,
     * not acceptance of an unchecked QP suffix. */
    if (pass->line_search_nominal_exact_valid) {
        pass->candidate = *nominal;
        pass->candidate.validation_objective = nominal_objective;
        pass->candidate.validation_minimum_corridor_slack_m =
            nominal_minimum_corridor_slack;
        pass->candidate.valid = 1;
        pass->path_delta = MpcRtiCandidatePathDelta_t{};
        pass->first_action = nominal->controls[0];
        pass->first_action_valid =
            isfinite(pass->first_action.steering_target_rad) &&
            isfinite(pass->first_action.throttle_command) &&
            pass->first_action.steering_target_rad >=
                -configuration->model.max_steering_rad - 1.0e-5f &&
            pass->first_action.steering_target_rad <=
                configuration->model.max_steering_rad + 1.0e-5f &&
            pass->first_action.throttle_command >=
                MPC_THROTTLE_MIN_DRIVE - 1.0e-5f &&
            pass->first_action.throttle_command <=
                MPC_THROTTLE_MAX + 1.0e-5f;
        if (pass->first_action_valid) {
            pass->nonlinear_objective = nominal_objective;
            pass->minimum_corridor_slack =
                nominal_minimum_corridor_slack;
            pass->lateral_accel_proxy = nominal_lateral_accel_proxy;
            pass->lateral_accel_proxy_stage =
                nominal_lateral_accel_proxy_stage;
            memcpy(pass->lateral_accel_proxy_by_stage,
                nominal_lateral_accel_proxy_by_stage,
                sizeof(pass->lateral_accel_proxy_by_stage));
            pass->diagnostic_output_count = 0;
            pass->line_search_accepted = 1;
            pass->line_search_selected_alpha = 0.0f;
            pass->sqp_step_metrics.selected_qp_predicted_cost_reduction = 0.0;
            pass->sqp_step_metrics.selected_source_exact_cost_reduction = 0.0;
            pass->sqp_accepted = 1;
            pass->status = MPC_RTI_CYCLE_ACCEPTED_LINE_SEARCH;
            finalize_step_metrics(pass, memory, configuration);
            pass->solve_us = monotonic_microseconds() - start_us;
            return pass->status;
        }
        pass->candidate.valid = 0;
    }

    /* Optional feasibility-biased ADMM retry on the SAME 30 source-plant
     * linearizations. Invoke only after the ordinary full candidate and all
     * eligible backtracks failed, and only when the 20ms command gate leaves
     * enough CPU for another solve + an independent exact rollout. */
    const double restoration_scope_start_us = monotonic_microseconds();
    if (configuration->enable_feasibility_restoration &&
        !first_is_restoration &&
        cycle_deadline_us > 0.0 &&
        cycle_deadline_us - monotonic_microseconds() > 6400.0 &&
        mpc_rti_prepare_feasibility_restoration_qp(&problem, nominal,
            configuration->feasibility_restoration_qp_relaxation_m,
            configuration->feasibility_restoration_wall_weight)) {
        pass->restoration_attempted = 1;
        const double restoration_start_us = monotonic_microseconds();
        RiccatiAdmmConfig_t restoration_solver = configuration->solver;
        restoration_solver.terminal_constraints=problem.terminal_constraints;
        memcpy(restoration_solver.state_polygons, problem.state_polygons,
               sizeof(problem.state_polygons));
        const double restore_budget_us = cycle_deadline_us -
            monotonic_microseconds() - 3500.0;
        if (restore_budget_us > 1000.0) {
            restoration_solver.max_solve_time_us = restore_budget_us;
            restoration_solver.reuse_same_epoch = 0;
            RiccatiAdmmState_t restore_memory;
            riccati_admm_state_init(&restore_memory);
            RiccatiSolution_t restore_solution = {};
            const double restore_start_us = monotonic_microseconds();
            const double restore_start_cpu_us = thread_cpu_microseconds();
            const RiccatiStatus_t restore_status =
                riccati_admm_solve_with_feedback(problem.steps,
                    problem.terminal_Q, problem.terminal_q,
                    problem.terminal_x_lb, problem.terminal_x_ub,
                    problem.x0, &restoration_solver, &restore_memory,
                    &restore_solution, NULL, NULL);
            pass->admm_us += monotonic_microseconds() - restore_start_us;
            const double restore_finish_cpu_us = thread_cpu_microseconds();
            if (isfinite(pass->admm_thread_cpu_us) &&
                isfinite(restore_start_cpu_us) &&
                isfinite(restore_finish_cpu_us))
                pass->admm_thread_cpu_us += fmax(0.0,
                    restore_finish_cpu_us-restore_start_cpu_us);
            pass->iterations += restore_solution.iterations;
            pass->restoration_iterations = restore_solution.iterations;
            pass->restoration_primal_residual =
                restore_solution.primal_residual;
            if ((restore_status == RICCATI_STATUS_OPTIMAL ||
                 restore_status == RICCATI_STATUS_MAX_ITERATIONS) &&
                isfinite(restore_solution.primal_residual) &&
                restore_solution.primal_residual <= 0.15f &&
                cycle_deadline_us - monotonic_microseconds() > 1500.0) {
                MpcModelControl_t restore_controls[PREDICTION_HORIZON];
                memcpy(restore_controls, nominal->controls,
                       sizeof(restore_controls));
                float previous_restore_s =
                    problem.x0[MPC_OPT_IDX_PREVIOUS_STEERING_COMMAND];
                float previous_restore_t =
                    problem.x0[MPC_OPT_IDX_PREVIOUS_THROTTLE_COMMAND];
                for (int k = 0; k < RICCATI_HORIZON; ++k) {
                    const RiccatiStepData_t *stage = &problem.steps[k];
                    const int increments =
                        configuration->model.command_increment_qp_enabled;
                    const float decision_s = clampf_rti(
                        restore_solution.u[k][0], stage->u_lb[0],
                        stage->u_ub[0]);
                    const float decision_t = clampf_rti(
                        restore_solution.u[k][1], stage->u_lb[1],
                        stage->u_ub[1]);
                    restore_controls[k].steering_target_rad = clampf_rti(
                        mpc_rti_absolute_command_from_qp_decision(
                            decision_s, previous_restore_s, increments),
                        -configuration->model.max_steering_rad,
                        configuration->model.max_steering_rad);
                    restore_controls[k].throttle_command = clampf_rti(
                        mpc_rti_absolute_command_from_qp_decision(
                            decision_t, previous_restore_t, increments),
                        MPC_THROTTLE_MIN_DRIVE, MPC_THROTTLE_MAX);
                    previous_restore_s = restore_controls[k].steering_target_rad;
                    previous_restore_t = restore_controls[k].throttle_command;
                }
                MpcRtiNominal_t restored = {};
                int failed_stage = -1;
                const double exact_start_us = monotonic_microseconds();
                const MpcRtiRolloutStatus_t exact_status =
                    mpc_rti_validate_candidate_plan(current_state,
                        current_progress, restore_controls,
                        trajectory, trajectory_count, lap_length,
                        horizon, prediction_dt, configuration,
                        &restored, &failed_stage);
                pass->exact_rollout_us +=
                    monotonic_microseconds() - exact_start_us;
                if (exact_status == MPC_RTI_ROLLOUT_OK) {
                    pass->candidate = restored;
                    pass->first_action = restored.controls[0];
                    pass->first_action_valid = 1;
                    pass->nonlinear_objective = restored.validation_objective;
                    pass->minimum_corridor_slack =
                        restored.validation_minimum_corridor_slack_m;
                    pass->sqp_accepted = 1;
                    pass->status = MPC_RTI_CYCLE_ACCEPTED_FEASIBLE;
                    pass->restoration_accepted = 1;
                    pass->restoration_us =
                        monotonic_microseconds() - restoration_start_us;
                    pass->line_search_selected_alpha = 1.0f;
                    /* Do not carry ADMM duals from a temporarily relaxed
                     * projection into the next hard-constrained MPC cycle. */
                    riccati_admm_state_init(&memory->solver_state);
                    finalize_step_metrics(pass, memory, configuration);
                    pass->solve_us = monotonic_microseconds() - start_us;
                    return pass->status;
                }
            }
        }
    }
    if (pass->restoration_attempted && !pass->restoration_accepted)
        pass->restoration_us =
            monotonic_microseconds() - restoration_scope_start_us;
    pass->candidate.valid = 0;
    memcpy(pass->candidate.controls, candidate_controls,
        (size_t)horizon * sizeof(candidate_controls[0]));
    pass->line_search_us = monotonic_microseconds() - line_search_start_us;
    pass->status = MPC_RTI_CYCLE_REJECTED_NONLINEAR_ROLLOUT;
    finalize_step_metrics(pass, memory, configuration);
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
        initialize_step_metrics(&result->r1_step_metrics);
        initialize_step_metrics(&result->r2_step_metrics);
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
    if(configuration->command_epoch_ns>0)
        aligned_configuration.solver.reuse_same_epoch=epoch_shift==0;
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
    if(configuration->terminal_viability &&
       !mpc_rti_initialize_terminal_targets(&nominal,trajectory,trajectory_count,
            lap_length,prediction_dt,&configuration->model,
            &result->comparison_objective)) {
        result->total_cycle_us=monotonic_microseconds()-cycle_start_us;
        return result->status;
    }
    comparison_configuration.objective_schedule = &result->comparison_objective;

    MpcRtiSqpCurvature curvature{};
    MpcRtiPassResult_t pass;
    memset(&pass, 0, sizeof(pass));
    pass.status = MPC_RTI_CYCLE_REJECTED_INPUT;
    pass.nonlinear_failure_stage = -1;
    pass.nonlinear_failure_reason = -1;
    pass.linearization_failure_stage = -1;
    pass.linearization_failure_index = -1;
    pass.admm_status = -1;
    /* R1 is the safety-critical, executable candidate; R2 is optional
     * refinement. A prior implementation subtracted an estimated R2 budget
     * from R1's deadline BEFORE building R1's Jacobian. When a Jacobian
     * worker was descheduled, R1 would skip ADMM at ~13 ms despite the
     * absolute 20 ms publication window still having several milliseconds.
     * Never let speculative R2 reservation starve the normal primary solve.
     * R2 is still attempted whenever the measured remaining budget AFTER
     * R1 pays for a complete relinearization, solve and exact validation.
     * The solve_rti_pass post-solve reserve remains mandatory for R1. */
    const double first_deadline=deadline_us;
    MpcRtiCycleStatus_t status = solve_rti_pass(
        &prepared_state, current_progress, &nominal, references,
        trajectory, trajectory_count, lap_length, prediction_dt, horizon,
        &comparison_configuration, memory, first_deadline,
        (double)MPC_RTI_POST_SOLVE_RESERVE_US, &curvature, result, &pass);
    result->r1_step_metrics = pass.sqp_step_metrics;
    result->first_pass_objective=pass.nonlinear_objective;
    result->second_pass_objective=NAN;
    /* Normal SQP iteration: when R1 provides an exact, physically feasible
     * trajectory, R2 relinearizes about THAT trajectory at the SAME command
     * epoch. R1/R2 are scored with a shared nonlinear objective, so an R2
     * that does not improve physical motion never replaces the safe iterate.
     * No extra solve is started unless the time budget permits one complete
     * model build, QP and nonlinear validation. */
    /* The configured SQP limit applies to every QP mode. Extra source
     * work may use spare time, but must never silently disable R2. */
    const int sqp_limit = configuration->max_sqp_iterations > 0
        ? configuration->max_sqp_iterations : 1;
    const int needs_refinement =
        (pass.sqp_accepted && pass.candidate.valid &&
         pass.first_action_valid && !configuration->prefer_feasibility_first) ||
        (status == MPC_RTI_CYCLE_REJECTED_NONLINEAR_ROLLOUT &&
         (pass.line_search_full_step_failure_reason ==
            MPC_RTI_ROLLOUT_CORRIDOR || pass.line_search_full_step_failure_reason ==
            MPC_RTI_ROLLOUT_TERMINAL_VIABILITY));
    const double refinement_remaining_us = deadline_us > 0.0
        ? deadline_us - monotonic_microseconds() : INFINITY;
    /* A second nominal + Jacobian is not free: the symmetric-speed trial
     * built 92 second QPs with no remaining ADMM budget. Reserve the measured
     * preparation cost, 1 ms of solver work and nonlinear completion before
     * starting refinement, rather than spending those milliseconds to skip. */
    const double refinement_minimum_us =
        (double)MPC_RTI_REFINEMENT_SOLVE_RESERVE_US +
        fmax(1000.0, pass.qp_build_us) + result->nominal_build_us + 1000.0;
    if (sqp_limit > 1 && needs_refinement &&
        pass.candidate.horizon == horizon &&
        refinement_remaining_us >= refinement_minimum_us) {
        const MpcRtiPassResult_t first_pass = pass;
        const MpcRtiCycleStatus_t first_status = status;
        const MpcRtiMemory_t memory_after_first = *memory;
        const int first_action_valid = first_pass.sqp_accepted &&
            first_pass.first_action_valid;
        const MpcModelControl_t *refinement_seed = first_action_valid
            ? first_pass.candidate.controls : first_pass.proposed_controls;
        /* One nonlinear SQP cycle has ONE objective. R2 must linearize at
         * its newly integrated physical source trajectory but still minimize
         * the same frozen reference targets and speed-weight branches as R1.
         * Rebuilding an independent objective here changes the optimization
         * problem mid-iteration and invalidates its curvature/merit checks. */
        MpcRtiCycleConfiguration_t refinement_configuration =
            comparison_configuration;
        /* No command has elapsed between SQP passes. The ADMM split/dual
         * iterate must stay at the same stages, just like the nominal seed.
         * The first solve of the next control cycle still shifts once. */
        refinement_configuration.solver.reuse_same_epoch = 1;
        if (comparison_configuration.qp_capture_second_pass)
            refinement_configuration.qp_capture =
                comparison_configuration.qp_capture_second_pass;

        MpcRtiNominal_t refinement_nominal;
        MpcRtiReference_t refinement_references[PREDICTION_HORIZON + 1];
        const double refinement_nominal_start_us = monotonic_microseconds();
        // Rebuild R2 from its own command seed. The old optional shortcut
        // reused R1's accepted rollout, but failed live closed-loop races and
        // was never enabled in production. Keeping it only added an unproven
        // alternative path to the safety-critical solver.
        const int refinement_nominal_built = mpc_rti_build_nominal_same_epoch(
            &prepared_state, current_progress, &nominal,
            refinement_seed, horizon, trajectory, trajectory_count,
            lap_length, prediction_dt, horizon, &configuration->model,
            &refinement_nominal, refinement_references);
        result->refinement_nominal_build_us =
            monotonic_microseconds() - refinement_nominal_start_us;
        if (refinement_nominal_built) {
            for (int k = 0; k <= horizon; ++k) {
                refinement_references[k].u = nominal.speed_reference[k];
                refinement_nominal.speed_reference[k] =
                    nominal.speed_reference[k];
            }
            /* No separate R2 yaw-reference alignment: every cost target,
             * including yaw, comes from the cycle's frozen objective.
             * Physical path-frame geometry was already reconstructed by
             * mpc_rti_build_nominal_same_epoch above. */
            MpcRtiPassResult_t refined_pass;
            memset(&refined_pass, 0, sizeof(refined_pass));
            const MpcRtiCycleStatus_t refined_status = solve_rti_pass(
                &prepared_state, current_progress, &refinement_nominal,
                refinement_references, trajectory, trajectory_count,
                lap_length, prediction_dt, horizon, &refinement_configuration, memory,
                deadline_us,
                (double)MPC_RTI_REFINEMENT_SOLVE_RESERVE_US,
                &curvature, result, &refined_pass);
            result->r2_step_metrics = refined_pass.sqp_step_metrics;
            result->second_pass_objective=refined_pass.nonlinear_objective;
            const int refined_action_valid = refined_pass.sqp_accepted &&
                refined_pass.first_action_valid;
            const int keep_first_pass =
                mpc_rti_should_keep_first_pass_after_refinement_deadline_skip(
                    first_pass.admm_status, refined_pass.admm_status,
                    refined_pass.solver_skipped_for_deadline);
            int refined_prefix_better = 0;
            if (!first_action_valid && !refined_action_valid &&
                !keep_first_pass &&
                (deadline_us <= 0.0 || deadline_us - monotonic_microseconds()
                    >= MPC_RTI_POST_SOLVE_RESERVE_US + 1000.0)) {
                /* Neither full horizon is certified. Do not blindly replace
                 * the first proposal by R2: compare equal 75 ms exact prefixes
                 * on the same state, inputs in flight, references and costs.
                 * The node still revalidates any prefix before publication.
                 * A short prefix cannot certify either rejected full tail. */
                const double prefix_start_us = monotonic_microseconds();
                MpcRtiNominal_t prefix;
                int failure_stage;
                const int prefix_horizon = MPC_RECOVERY_PREFIX_HORIZON;
                const int first_prefix_valid = first_pass.first_action_valid &&
                    mpc_rti_validate_candidate_plan(&prepared_state,
                        current_progress, first_pass.proposed_controls,
                        trajectory, trajectory_count, lap_length,
                        prefix_horizon, prediction_dt, &comparison_configuration,
                        &prefix, &failure_stage) == MPC_RTI_ROLLOUT_OK;
                if (first_prefix_valid)
                    result->refinement_first_prefix_objective =
                        prefix.validation_objective;
                const int second_prefix_valid = refined_pass.first_action_valid &&
                    mpc_rti_validate_candidate_plan(&prepared_state,
                        current_progress, refined_pass.proposed_controls,
                        trajectory, trajectory_count, lap_length,
                        prefix_horizon, prediction_dt, &comparison_configuration,
                        &prefix, &failure_stage) == MPC_RTI_ROLLOUT_OK;
                if (second_prefix_valid)
                    result->refinement_second_prefix_objective =
                        prefix.validation_objective;
                refined_prefix_better = mpc_rti_refinement_improves_action(
                    first_prefix_valid,
                    result->refinement_first_prefix_objective,
                    second_prefix_valid,
                    result->refinement_second_prefix_objective);
                result->refinement_prefix_selected_pass =
                    refined_prefix_better ? 2 : 1;
                result->refinement_prefix_validation_us =
                    monotonic_microseconds() - prefix_start_us;
            }
            /* Both passes use the same objective schedule, but different
             * exact nonlinear nominal trajectories and source Jacobians.
             * Compare them after full source validation under that objective. */
            MpcRtiNominal_t first_comparison = first_pass.candidate;
            MpcRtiNominal_t refined_comparison = refined_pass.candidate;
            const int first_comparison_valid = first_action_valid &&
                mpc_rti_score_candidate_plan(&first_comparison, trajectory,
                    trajectory_count, lap_length, &comparison_configuration);
            const int refined_comparison_valid = refined_action_valid &&
                mpc_rti_score_candidate_plan(&refined_comparison, trajectory,
                    trajectory_count, lap_length, &comparison_configuration);
            int chose_refinement = 0;
            int terminal_preferred = 0;
            /* A fully source-valid 750ms horizon does not prove its final
             * physical state has a safe successor at 775-825ms. Only when
             * BOTH R1/R2 are already valid, probe two legal causal source
             * continuations from each terminal state. A positive finding
             * is an actionable certificate for those two tested controls,
             * not an infinite-horizon guarantee. Never spend the reserved
             * publication time on this optional comparison. */
            if (configuration->prefer_terminal_continuable_refinement &&
                !configuration->terminal_viability &&
                first_comparison_valid && refined_comparison_valid &&
                (first_comparison.validation_minimum_corridor_slack_m<.04f ||
                 refined_comparison.validation_minimum_corridor_slack_m<.04f ||
                 terminal_dynamic_instability(&first_comparison) ||
                 terminal_dynamic_instability(&refined_comparison)) &&
                (deadline_us<=0. ||
                 deadline_us-monotonic_microseconds()>=
                     MPC_RTI_POST_SOLVE_RESERVE_US+3000.)) {
                const double extension_start_us=monotonic_microseconds();
                const int viable_first=mpc_rti_terminal_extension_probe(
                    &first_comparison,trajectory,trajectory_count,lap_length,
                    prediction_dt,configuration);
                ++result->refinement_terminal_probe_count;
                /* Both alternatives must be evaluated with the same timing
                 * policy; if the budget is unexpectedly consumed, do not
                 * favor a plan merely because it was tested first. */
                if (deadline_us<=0. ||
                    deadline_us-monotonic_microseconds()>=
                        MPC_RTI_POST_SOLVE_RESERVE_US+1200.) {
                    const int viable_refined=mpc_rti_terminal_extension_probe(
                        &refined_comparison,trajectory,trajectory_count,
                        lap_length,prediction_dt,configuration);
                    ++result->refinement_terminal_probe_count;
                    if(viable_first!=viable_refined)
                        terminal_preferred=viable_refined ? 2:1;
                }
                result->refinement_terminal_probe_us=
                    monotonic_microseconds()-extension_start_us;
                result->refinement_terminal_selected_pass=terminal_preferred;
            }
            if (terminal_preferred==2 ||
                (terminal_preferred==0 &&
                 mpc_rti_refinement_improves_action(first_comparison_valid,
                    first_comparison.validation_objective, refined_comparison_valid,
                    refined_comparison.validation_objective))) {
                pass = refined_pass;
                status = refined_status;
                chose_refinement = 1;
            } else if (terminal_preferred==1) {
                pass=first_pass;
                status=first_status;
                *memory=memory_after_first;
            } else if (keep_first_pass) {
                /* A refinement that never reached ADMM has no candidate to
                 * supersede the completed first pass. Keep that finite
                 * iterate available to the controller's exact prefix and
                 * certified-plan validation paths instead of erasing it. */
                pass = first_pass;
                status = first_status;
                *memory = memory_after_first;
            } else if (refined_prefix_better) {
                pass = refined_pass;
                status = refined_status;
                chose_refinement = 1;
            } else {
                pass = first_pass;
                status = first_status;
                *memory = memory_after_first;
            }
            if (first_comparison_valid || refined_comparison_valid)
                result->refinement_full_horizon_selected_pass =
                    chose_refinement ? 2 : (first_comparison_valid ? 1 : 0);
            if (chose_refinement && refined_comparison_valid)
                pass.nonlinear_objective = refined_comparison.validation_objective;
            else if (!chose_refinement && first_comparison_valid)
                pass.nonlinear_objective = first_comparison.validation_objective;
            const MpcRtiPassResult_t *additional_pass = chose_refinement
                ? &first_pass : &refined_pass;
            pass.sqp_attempted += additional_pass->sqp_attempted;
            pass.sqp_completed += additional_pass->sqp_completed;
            pass.sqp_accepted += additional_pass->sqp_accepted;
            pass.iterations += additional_pass->iterations;
            pass.restoration_attempted += additional_pass->restoration_attempted;
            pass.restoration_accepted += additional_pass->restoration_accepted;
            pass.restoration_iterations += additional_pass->restoration_iterations;
            pass.restoration_us += additional_pass->restoration_us;
            pass.solve_us += additional_pass->solve_us;
            pass.qp_build_us += additional_pass->qp_build_us;
            pass.admm_us += additional_pass->admm_us;
            pass.admm_thread_cpu_us += additional_pass->admm_thread_cpu_us;
            pass.exact_rollout_us += additional_pass->exact_rollout_us;
            pass.nonlinear_validation_us +=
                additional_pass->nonlinear_validation_us;
            pass.jacobian_wall_us += additional_pass->jacobian_wall_us;
            pass.jacobian_thread_cpu_us +=
                additional_pass->jacobian_thread_cpu_us;
            pass.jacobian_slowest_stage_us = fmax(
                pass.jacobian_slowest_stage_us,
                additional_pass->jacobian_slowest_stage_us);
            pass.jacobian_stages_completed +=
                additional_pass->jacobian_stages_completed;
            pass.jacobian_source_sensitivity_calls +=
                additional_pass->jacobian_source_sensitivity_calls;
            pass.jacobian_finite_difference_columns +=
                additional_pass->jacobian_finite_difference_columns;
            for (int worker = 0;
                 worker < MPC_RTI_MAX_JACOBIAN_THREADS; ++worker) {
                pass.jacobian_worker_cpu_us[worker] +=
                    additional_pass->jacobian_worker_cpu_us[worker];
                pass.jacobian_worker_stage_wall_us[worker] +=
                    additional_pass->jacobian_worker_stage_wall_us[worker];
                pass.jacobian_worker_stage_count[worker] +=
                    additional_pass->jacobian_worker_stage_count[worker];
                if (pass.jacobian_worker_os_tid[worker] == 0)
                    pass.jacobian_worker_os_tid[worker] =
                        additional_pass->jacobian_worker_os_tid[worker];
            }
            if (additional_pass->jacobian_threads_used >
                pass.jacobian_threads_used)
                pass.jacobian_threads_used =
                    additional_pass->jacobian_threads_used;
            pass.solver_time_budget_exhausted |=
                additional_pass->solver_time_budget_exhausted;
            pass.solver_skipped_for_deadline |=
                additional_pass->solver_skipped_for_deadline;
        }
    }

    result->exact_objective_improvement=isfinite(result->first_pass_objective) &&
        isfinite(pass.nonlinear_objective)
        ? result->first_pass_objective-pass.nonlinear_objective : 0.;
    result->solver_iterations = pass.iterations;
    result->solver_skipped_for_deadline = pass.solver_skipped_for_deadline;
    result->solver_time_budget_exhausted =
        pass.solver_time_budget_exhausted;
    result->sqp_iterations_attempted = pass.sqp_attempted;
    result->sqp_iterations_completed = pass.sqp_completed;
    result->sqp_iterations_accepted = pass.sqp_accepted;
    result->admm_status = pass.admm_status;
    result->primal_residual = pass.primal_residual;
    result->dual_residual = pass.dual_residual;
    result->state_primal_residual = pass.state_primal_residual;
    result->state_dual_residual = pass.state_dual_residual;
    result->input_primal_residual = pass.input_primal_residual;
    result->input_dual_residual = pass.input_dual_residual;
    result->state_primal_stage = pass.state_primal_stage;
    result->state_primal_index = pass.state_primal_index;
    result->state_dual_stage = pass.state_dual_stage;
    result->state_dual_index = pass.state_dual_index;
    result->input_primal_stage = pass.input_primal_stage;
    result->input_primal_index = pass.input_primal_index;
    result->input_dual_stage = pass.input_dual_stage;
    result->input_dual_index = pass.input_dual_index;
    result->maximum_regularization = pass.maximum_regularization;
    result->regularization_count = pass.regularization_count;
    result->nonsmooth_jacobian_columns = pass.nonsmooth_columns;
    result->linearization_failure_stage = pass.linearization_failure_stage;
    result->linearization_failure_reason = pass.linearization_failure_reason;
    result->linearization_failure_index = pass.linearization_failure_index;
    result->jacobian_stages_completed = pass.jacobian_stages_completed;
    result->jacobian_threads_used = pass.jacobian_threads_used;
    result->jacobian_source_sensitivity_calls =
        pass.jacobian_source_sensitivity_calls;
    result->jacobian_finite_difference_columns =
        pass.jacobian_finite_difference_columns;
    result->jacobian_wall_us = pass.jacobian_wall_us;
    result->jacobian_thread_cpu_us = pass.jacobian_thread_cpu_us;
    result->jacobian_slowest_stage_us = pass.jacobian_slowest_stage_us;
    memcpy(result->jacobian_worker_cpu_us, pass.jacobian_worker_cpu_us,
        sizeof(result->jacobian_worker_cpu_us));
    memcpy(result->jacobian_worker_stage_wall_us,
        pass.jacobian_worker_stage_wall_us,
        sizeof(result->jacobian_worker_stage_wall_us));
    memcpy(result->jacobian_worker_stage_count,
        pass.jacobian_worker_stage_count,
        sizeof(result->jacobian_worker_stage_count));
    memcpy(result->jacobian_worker_os_tid, pass.jacobian_worker_os_tid,
        sizeof(result->jacobian_worker_os_tid));
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
    result->solver_cold_start_initialization_time_ns =
        pass.solver_cold_start_initialization_time_ns;
    result->riccati_rhs_time_ns = pass.riccati_rhs_time_ns;
    result->admm_projection_time_ns = pass.admm_projection_time_ns;
    result->rho_update_time_ns = pass.rho_update_time_ns;
    result->solver_scaling_time_ns = pass.solver_scaling_time_ns;
    result->solver_output_unscale_time_ns =
        pass.solver_output_unscale_time_ns;
    result->feasibility_restoration_attempted = pass.restoration_attempted;
    result->feasibility_restoration_accepted = pass.restoration_accepted;
    result->feasibility_restoration_iterations = pass.restoration_iterations;
    result->feasibility_restoration_primal_residual =
        pass.restoration_primal_residual;
    result->feasibility_restoration_us = pass.restoration_us;
    result->qp_build_us = pass.qp_build_us;
    result->admm_us = pass.admm_us;
    result->admm_thread_cpu_us = pass.admm_thread_cpu_us;
    result->exact_rollout_us = pass.exact_rollout_us;
    result->nonlinear_validation_us = pass.nonlinear_validation_us;
    result->total_solve_us = pass.solve_us;
    result->total_cycle_us = monotonic_microseconds() - cycle_start_us;

    result->line_search_attempted = pass.line_search_attempted;
    result->line_search_trials = pass.line_search_trials;
    result->line_search_accepted = pass.line_search_accepted;
    result->line_search_skipped_budget = pass.line_search_skipped_budget;
    result->line_search_skipped_command_epoch =
        pass.line_search_skipped_command_epoch;
    result->line_search_full_step_failure_reason =
        pass.line_search_full_step_failure_reason;
    result->line_search_full_step_failure_stage =
        pass.line_search_full_step_failure_stage;
    result->line_search_last_failure_reason =
        pass.line_search_last_failure_reason;
    result->line_search_last_failure_stage =
        pass.line_search_last_failure_stage;
    result->line_search_merit_triggered =
        pass.line_search_merit_triggered;
    result->line_search_nominal_exact_valid =
        pass.line_search_nominal_exact_valid;
    result->line_search_nominal_objective =
        pass.line_search_nominal_objective;
    result->line_search_selected_alpha = pass.line_search_selected_alpha;
    result->line_search_us = pass.line_search_us;
    result->diagnostic_output_count = pass.diagnostic_output_count;
    if (pass.diagnostic_output_count > 0)
        memcpy(result->diagnostic_outputs, pass.diagnostic_outputs,
            (size_t)pass.diagnostic_output_count *
                sizeof(result->diagnostic_outputs[0]));

    if (status != MPC_RTI_CYCLE_ACCEPTED_OPTIMAL &&
        status != MPC_RTI_CYCLE_ACCEPTED_FEASIBLE &&
        status != MPC_RTI_CYCLE_ACCEPTED_LINE_SEARCH) {
        *memory = memory_before;
        result->status = status;
        return status;
    }

    result->status = status;
    result->first_control = pass.first_action;
    result->published_steering_command = pass.first_action.steering_target_rad;
    result->published_throttle = pass.first_action.throttle_command;
    result->minimum_predicted_corridor_slack_m = pass.minimum_corridor_slack;
    memory->nominal = pass.candidate;
    if (!mpc_rti_memory_set_warm_start_controls_at_epoch(memory,
            pass.candidate.controls, pass.candidate.horizon,
            configuration->command_epoch_ns)) {
        /* A cycle cannot be considered committed if its command sequence
         * cannot be represented as the next control warm start. */
        *memory = memory_before;
        result->status = MPC_RTI_CYCLE_REJECTED_INPUT;
        result->input_failure_reason = MPC_RTI_INPUT_FAILURE_NOMINAL_BUILD;
        return result->status;
    }
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
        for(int j=0;j<MPC_RTI_NX;++j)
            cost+=0.5f*stage->Q_offdiag[i][j]*state[i]*state[j];
    }
    for (int input = 0; input < MPC_RTI_NU; ++input)
        cost += 0.5f * stage->R_diag[input] * control[input] * control[input] +
            stage->r[input] * control[input];
    cost+=stage->R_cross*control[0]*control[1];
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
            ? overspeed_weight_for_reference(configuration, reference)
            : underspeed_weight_for_reference(configuration, reference)) *
            eu * eu +
        (float)source_slip_cost(configuration, &state->plant, reference) +
        configuration->weight_v * ev * ev +
        configuration->weight_r * er * er +
        configuration->weight_steering_command * edelta * edelta +
        configuration->weight_steering_command * eactual_delta * eactual_delta +
        configuration->weight_steering_target *
            (control->steering_target_rad -
                reference->steering_command) *
            (control->steering_target_rad -
                reference->steering_command) +
        configuration->weight_throttle_command *
            control->throttle_command * control->throttle_command;
}
