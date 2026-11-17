#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define _POSIX_C_SOURCE 200809L

#include "mpc_cycle_internal.hpp"
#include "mpc_source_sqp.hpp"
#include "vehicle_model.h"

#include <algorithm>
#include <memory>
#include <cmath>
#include <math.h>
#include <string.h>
#include <time.h>

typedef struct { float left; float right; } MpcCorridorMargins;

float corridor_margin_at_prediction(
    const MpcSolverConfiguration_t *configuration,
    int prediction_index)
{
    return prediction_index == 1
        ? configuration->first_prediction_corridor_margin_m
        : configuration->corridor_margin_m;
}

float corridor_tolerance_at_prediction(
    const MpcSolverConfiguration_t *configuration,
    int prediction_index)
{
    const int start = configuration->far_stage_tolerance_start_stage;
    float tolerance = configuration->nonlinear_corridor_tolerance_m;
    if (configuration->far_stage_corridor_tolerance_m > 0.f &&
        prediction_index > start) {
        const float ramp = fminf(1.f, (float)(prediction_index - start) /
                                      (float)(PREDICTION_HORIZON - start));
        tolerance += configuration->far_stage_corridor_tolerance_m * ramp;
    }
    return tolerance;
}

static MpcCorridorMargins corridor_margin_for_state(
    const MpcSolverConfiguration_t *configuration,
    const MpcModelState_t *state,
    int prediction_index);

static int finite_nonnegative(float value)
{
    return isfinite(value) && value >= 0.0f;
}

int finite_cycle_state(const MpcCycleState_t *state)
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

int finite_reference(const MpcCycleReference_t *reference)
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

int valid_configuration(const MpcSolverConfiguration_t *configuration)
{
    if (!configuration) return 0;
    const VehicleParameters_t vehicle = vehicle_model_get_parameters();
    const float weights[] = {
        configuration->weight_e_y,
        configuration->weight_e_psi,
        configuration->weight_u,
        configuration->weight_u_overspeed,
        configuration->overspeed_curvature_gain_m,
        configuration->underspeed_curvature_gain_m,
        configuration->lateral_curvature_gain_m,
        configuration->heading_curvature_gain_m,
        configuration->weight_grip_loss,
        configuration->grip_loss_tolerance,
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
        configuration->nonlinear_corridor_tolerance_m,
        configuration->far_stage_corridor_tolerance_m};
    for (unsigned int i = 0; i < sizeof(weights) / sizeof(weights[0]); ++i)
        if (!finite_nonnegative(weights[i])) return 0;

    return configuration->overspeed_curvature_gain_m <= 10.0f &&
        configuration->underspeed_curvature_gain_m <= 10.0f &&
        configuration->jacobian_threads >= 0 &&
        configuration->jacobian_threads <= MPC_CYCLE_MAX_JACOBIAN_THREADS &&
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
        isfinite(configuration->max_steering_rad) &&
        configuration->max_steering_rad > 0.0f &&
        configuration->max_steering_rad <= vehicle.max_steering_angle &&
        configuration->first_prediction_corridor_margin_m <=
            configuration->corridor_margin_m;
}

static void store_grip_metrics(MpcCycleState_t *state,
                               const MpcStageResult_t *step)
{
    for (int wheel = 0; wheel < 4; ++wheel) {
        state->incoming_wheel_grip_loss[wheel] =
            step->output.maximum_grip_loss[wheel];
    }
}

static float clampf_cycle(float value, float lower, float upper)
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
    const MpcSolverConfiguration_t *configuration,
    MpcCycleReference_t *reference,
    MpcTrajectorySample_t *world_sample,
    const MpcTrajectorySample_t *projected_sample = nullptr,
    MpcCycleReferenceSamples *corridor_samples = nullptr)
{
    MpcTrajectorySample_t sample;
    if (!reference ||
        (!projected_sample && !mpc_trajectory_sample(trajectory,
            trajectory_count, lap_length, progress, &sample))) return 0;
    if (projected_sample) sample = *projected_sample;
    /* Keep the interpolated geometry before adjusting witness
     * targets/speed for the current optimization phase. The same sample
     * is used to reconstruct physical world-frame rear-axle poses. */
    if (world_sample) *world_sample = sample;
    const float speed = fminf((float)sample.speed,
        configuration->active_speed_ceiling_mps);
    /* A witness certifies one speed profile. During an explicitly requested
     * warmup/capped-speed phase use geometry-only tracking references instead;
     * neither its lateral state nor its torque is valid at the other speed.
     * The same full nonlinear source plant still predicts every candidate.
     * Once uncapped, the original witness references are restored unchanged. */
    if (sample.source_witness_valid && speed < (float)sample.speed-1.e-4f)
        sample.source_witness_valid = 0;
    // The smooth inboard contour has its own tangent and curvature.
    // Steering/yaw/heading targets follow that curve, or the
    // strong heading cost resists every intentional inboard maneuver.
    const float contour_steering_correction =
        (float)atan(.324*sample.robust_contour_curvature_delta_radpm);
    const float feedforward = sample.source_witness_valid ?
        clampf_cycle((float)sample.source_physical_steering +
            contour_steering_correction,
            -configuration->max_steering_rad, configuration->max_steering_rad)
        : clampf_cycle(
            vehicle_model_steering_for_curvature((float)(
                sample.curvature+sample.robust_contour_curvature_delta_radpm)),
            -configuration->max_steering_rad, configuration->max_steering_rad);
    float left_bound = (float)sample.left_bound;
    float right_bound = (float)sample.right_bound;
    if (corridor_samples) corridor_samples->center = {left_bound, right_bound};
    if (configuration->corridor_preview_halfwidth_m > 0.0f) {
        const double halfwidth =
            configuration->corridor_preview_halfwidth_m;
        MpcTrajectorySample_t endpoint;
        if (!mpc_trajectory_sample(trajectory, trajectory_count, lap_length,
                progress - halfwidth, &endpoint)) return 0;
        if (corridor_samples) corridor_samples->endpoints[0] = {
            (float)endpoint.left_bound, (float)endpoint.right_bound};
        left_bound = fminf(left_bound, (float)endpoint.left_bound);
        right_bound = fminf(right_bound, (float)endpoint.right_bound);
        if (!mpc_trajectory_sample(trajectory, trajectory_count, lap_length,
                progress + halfwidth, &endpoint)) return 0;
        if (corridor_samples) corridor_samples->endpoints[1] = {
            (float)endpoint.left_bound, (float)endpoint.right_bound};
        left_bound = fminf(left_bound, (float)endpoint.left_bound);
        right_bound = fminf(right_bound, (float)endpoint.right_bound);

        /* Piecewise-linear bounds attain their interval minimum at an
         * endpoint or a trajectory knot. Binary-search only the exact meter
         * window to avoid a full-track scan per reference. */
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
    *reference = MpcCycleReference_t{
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
        .source_steering_target = clampf_cycle(
            (float)sample.source_steering_target +
                contour_steering_correction,
            -configuration->max_steering_rad,configuration->max_steering_rad),
        .source_throttle = (float)sample.source_throttle};
    return finite_reference(reference);
}

int mpc_cycle_sample_stage_reference(
    const MpcTrajectorySample_t *trajectory, size_t trajectory_count,
    double lap_length, double progress,
    const MpcSolverConfiguration_t *configuration, MpcCycleStageReference *stage)
{
    if (!configuration || !stage) return 0;
    return reference_at_progress_with_sample(
        trajectory, trajectory_count, lap_length, progress, configuration,
        &stage->reference, nullptr, nullptr, &stage->samples);
}

/* Standard reference callers do not need a copy of world-frame geometry. */
int mpc_cycle_reference_at_progress(
    const MpcTrajectorySample_t *trajectory, size_t trajectory_count,
    double lap_length, double progress,
    const MpcSolverConfiguration_t *configuration,
    MpcCycleReference_t *reference)
{
    return reference_at_progress_with_sample(
        trajectory, trajectory_count, lap_length, progress,
        configuration, reference, NULL);
}

int mpc_cycle_source_exact_stage(
    const MpcCycleState_t *current,double current_progress,
    const MpcModelControl_t *command,
    const MpcTrajectorySample_t *trajectory,size_t trajectory_count,
    double lap_length,float dt,MpcCycleState_t *next,
    double *next_progress)
{
    if(!current||!command||!trajectory||!next||!next_progress||
       !isfinite(current_progress)||!isfinite(lap_length)||lap_length<=0.||
       !isfinite(dt)||dt<=0.f)return 0;
    MpcStageResult_t physical=mpc_vehicle_model_step(
        &current->plant,command,dt,0.0f);
    if(!physical.valid||
       !isfinite(current_progress+physical.delta_s_m))return 0;
    MpcCycleState_t updated={};
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
    updated.incoming_body_forward_delta_m=
        physical.body_forward_delta_m;
    updated.incoming_body_lateral_delta_m=
        physical.body_lateral_delta_m;
    updated.incoming_body_yaw_delta_rad=
        physical.body_yaw_delta_rad;
    store_grip_metrics(&updated,&physical);
    updated.incoming_transition_valid=1;
    *next=updated;
    return 1;
}

static int rollout_nominal_controls(
    const MpcCycleState_t *initial_state,
    double initial_progress,
    const MpcModelControl_t controls[PREDICTION_HORIZON],
    const MpcTrajectorySample_t *trajectory, size_t trajectory_count,
    double lap_length,
    int horizon,
    float prediction_dt,
    MpcCycleState_t states[PREDICTION_HORIZON + 1],
    double progress[PREDICTION_HORIZON + 1])
{
    states[0] = *initial_state;
    progress[0] = initial_progress;
    for (int k=0;k<horizon;++k) {
        if(!mpc_cycle_source_exact_stage(&states[k],progress[k],
                &controls[k],trajectory,trajectory_count,lap_length,
                prediction_dt,&states[k+1],&progress[k+1])) {
            return 0;
        }
    }
    return 1;
}

static void align_yaw_rate_references(
    MpcCycleReference_t references[PREDICTION_HORIZON + 1],
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

int mpc_cycle_references_from_exact_nominal(
    const MpcCycleNominal_t *nominal,
    const MpcTrajectorySample_t *trajectory, size_t trajectory_count,
    double lap_length, const MpcSolverConfiguration_t *model,
    MpcCycleReference_t references[PREDICTION_HORIZON + 1])
{
    if(!nominal || !nominal->valid ||
       (nominal->horizon<1 || nominal->horizon>PREDICTION_HORIZON) ||
       !trajectory || trajectory_count < 3 || !model || !references ||
       !isfinite(lap_length) || lap_length <= 0.)
        return 0;
    for(int k=0;k<=nominal->horizon;++k) {
        if(!isfinite(nominal->progress[k]) ||
           !mpc_cycle_reference_at_progress(trajectory,trajectory_count,lap_length,
               nominal->progress[k],model,&references[k]))
            return 0;
    }
    references[0].u = nominal->states[0].plant.u;
    align_yaw_rate_references(references,nominal->horizon);
    return 1;
}

static MpcCorridorMargins corridor_margin_for_state(
    const MpcSolverConfiguration_t *configuration,
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
    // The pose is at the rear axle, not the rectangle centre. Project both
    // longitudinal anchors separately: using the front overhang on both
    // sides invents (front-rear)*abs(sin(epsi)) of extra vehicle on one side.
    return MpcCorridorMargins{
        common + fmaxf(MPC_REAR_AXLE_TO_FRONT_BUMPER_M * sine,
                      -MPC_REAR_OVERHANG_M * sine),
        common + fmaxf(-MPC_REAR_AXLE_TO_FRONT_BUMPER_M * sine,
                      MPC_REAR_OVERHANG_M * sine)};
}

// Differentiation of the preview-width envelope that
// reference_at_progress_with_sample uses: min of the current interpolated
// width, its +/- preview endpoints and every loaded knot in the window.
// Each candidate is affine in progress inside its current loaded segment.
// A knot strictly inside the window is constant with respect to progress.
// This is the derivative of the implemented piecewise-linear reference,
// not a fitted cubic curve or a finite-difference estimate of float widths.
static int corridor_reference_width_slopes(
    const MpcTrajectorySample_t *track,size_t count,double lap,
    double progress,const MpcSolverConfiguration_t *cfg,
    const MpcCycleReference_t *reference,
    const MpcCycleReferenceSamples &samples,
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
    const auto &center=samples.center;
    MpcTrajectoryProgressDifferential_t cs{};
    if(!mpc_trajectory_sample_progress_differential(
            track,count,lap,progress,&cs))return 0;
    ActiveWidth left{float(center.left_bound),
                     cs.forward.left_bound,cs.backward.left_bound};
    ActiveWidth right{float(center.right_bound),
                      cs.forward.right_bound,cs.backward.right_bound};
    const double half=cfg->corridor_preview_halfwidth_m;
    if(half>0.) {
        for(int endpoint=0;endpoint<2;++endpoint) {
            const double q=endpoint==0?progress-half:progress+half;
            const auto &p=samples.endpoints[endpoint];
            MpcTrajectoryProgressDifferential_t dp{};
            if(!mpc_trajectory_sample_progress_differential(
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
        const auto knots=[&](double lo,double hi) {
            size_t first_index=0,last_index=count;
            while(first_index<last_index) {
                const size_t mid=first_index+(last_index-first_index)/2;
                if(track[mid].s<lo)first_index=mid+1;
                else last_index=mid;
            }
            for(size_t i=first_index;i<count && track[i].s<=hi;++i) {
                const double knot_s=track[i].s;
                // A knot exactly at a moving preview endpoint is the endpoint:
                // the bound is continuous there, so the knot adds its value to the
                // minimum (the scorer's window is closed) and no slope of its own
                // (a slope of zero would be wrong in the direction the knot leaves).
                // The periodic lap seam is an artificial split of the
                // preview window, not a moving preview endpoint.
                const double edge_tolerance=1.e-11*std::max(1.,lap);
                const bool moving_lower=
                    std::abs(lo-window_start)<=edge_tolerance ||
                    std::abs(lo-(window_start+lap))<=edge_tolerance;
                const bool moving_upper=
                    std::abs(hi-window_end)<=edge_tolerance ||
                    std::abs(hi-(window_end-lap))<=edge_tolerance;
                if((moving_lower && std::abs(knot_s-lo)<=edge_tolerance) ||
                   (moving_upper && std::abs(knot_s-hi)<=edge_tolerance)) {
                    left.value=std::min(left.value,float(track[i].left_bound));
                    right.value=std::min(right.value,float(track[i].right_bound));
                    continue;
                }
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
    }
    // Exact float min parity with the source scorer is mandatory.
    if(left.value!=reference->left_bound ||
       right.value!=reference->right_bound)return 0;
    *left_forward=left.forward;*left_backward=left.backward;
    *right_forward=right.forward;*right_backward=right.backward;
    return std::isfinite(*left_forward)&&std::isfinite(*left_backward)&&
           std::isfinite(*right_forward)&&std::isfinite(*right_backward);
}

static int corridor_corners_with_reference(
    const MpcModelState_t *state,
    const MpcTrajectorySample_t *trajectory,size_t trajectory_count,
    double lap_length,double progress_m,
    const MpcSolverConfiguration_t *configuration,int prediction_index,
    const MpcCycleStageReference &stage, float source_right, float source_left,
    MpcCycleSourceCorridorCorners_t *out)
{
    if(!state||!trajectory||!configuration||!out||
       prediction_index<1||prediction_index>PREDICTION_HORIZON||
       !std::isfinite(state->e_y)||!std::isfinite(state->e_psi)||
       !std::isfinite(progress_m))return 0;
    const auto &ref=stage.reference;
    auto physical = *configuration;
    physical.corridor_margin_m = physical.first_prediction_corridor_margin_m = 0.f;
    physical.nonlinear_corridor_tolerance_m = 0.f;
    float physical_right, physical_left;
    if(!mpc_cycle_compute_corridor_slack(state,ref.left_bound,
            ref.right_bound,&physical,prediction_index,
            &physical_right,&physical_left))return 0;
    MpcCycleSourceCorridorCorners_t r{};
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
    // Express all eight corner constraints relative to their source
    // float-evaluated source min-slacks. This preserves source-parity at
    // the active support corner without subtracting two differently
    // rounded trigonometric approximations.
    for(int j=0;j<4;++j) {
        r.left[j]=double(source_left)+(maximum-lat[j]);
        r.right[j]=double(source_right)+(lat[j]-minimum);
        r.physical_left[j]=double(physical_left)+(maximum-lat[j]);
        r.physical_right[j]=double(physical_right)+(lat[j]-minimum);
        r.left_d_psi[j]=-derivative[j];
        r.right_d_psi[j]=derivative[j];
    }
    // The preview-reference minimum is piecewise affine and may change at
    // a loaded knot. Both directional slopes describe its active pieces.
    if(corridor_reference_width_slopes(
            trajectory,trajectory_count,lap_length,progress_m,
            configuration,&ref,stage.samples,
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

int mpc_cycle_source_exact_corridor_corners(
    const MpcModelState_t *state,
    const MpcTrajectorySample_t *trajectory,size_t trajectory_count,
    double lap_length,double progress_m,
    const MpcSolverConfiguration_t *configuration,int prediction_index,
    MpcCycleSourceCorridorCorners_t *out)
{
    if(!state||!trajectory||!configuration||!out||
       prediction_index<1||prediction_index>PREDICTION_HORIZON||
       !std::isfinite(state->e_y)||!std::isfinite(state->e_psi)||
       !std::isfinite(progress_m))return 0;
    MpcCycleStageReference stage{};
    if(!mpc_cycle_sample_stage_reference(trajectory,trajectory_count,
            lap_length,progress_m,configuration,&stage))return 0;
    float right=NAN,left=NAN;
    if(!mpc_cycle_compute_corridor_slack(state,stage.reference.left_bound,
            stage.reference.right_bound,configuration,prediction_index,
            &right,&left))return 0;
    return corridor_corners_with_reference(state,trajectory,trajectory_count,
        lap_length,progress_m,configuration,prediction_index,stage,right,left,out);
}

int mpc_cycle_compute_corridor_slack(
    const MpcModelState_t *state,
    float left_bound_m,
    float right_bound_m,
    const MpcSolverConfiguration_t *configuration,
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
    const float tolerance = corridor_tolerance_at_prediction(configuration, prediction_index);
    const float lower = margin.right - right_bound_m - tolerance;
    const float upper = left_bound_m - margin.left + tolerance;
    if (!isfinite(lower) || !isfinite(upper)) return 0;
    *lower_slack_m = state->e_y - lower;
    *upper_slack_m = upper - state->e_y;
    return isfinite(*lower_slack_m) && isfinite(*upper_slack_m);
}

static MpcCycleCommandEnvelopeFailure_t command_envelope_failure(
    const MpcCycleState_t *state,
    const MpcSolverConfiguration_t *configuration)
{
    if (!state || !configuration)
        return MPC_CYCLE_COMMAND_ENVELOPE_NONFINITE_STATE;
    if (!isfinite(state->plant.delayed_throttle_command_1) ||
        !isfinite(state->plant.delayed_throttle_command_2))
        return MPC_CYCLE_COMMAND_ENVELOPE_NONFINITE_STATE;
    if (state->plant.delayed_throttle_command_1 < 0.0f ||
        state->plant.delayed_throttle_command_1 > 1.0f ||
        state->plant.delayed_throttle_command_2 < 0.0f ||
        state->plant.delayed_throttle_command_2 > 1.0f)
        return MPC_CYCLE_COMMAND_ENVELOPE_THROTTLE_QUEUE;
    if (!finite_cycle_state(state))
        return MPC_CYCLE_COMMAND_ENVELOPE_NONFINITE_STATE;
    /* Native PxVehicleNoDrive hard braking can briefly reverse body-u by up
     * to 7.79 mm/s across the tested 0.5--12 m/s stop envelope. This is a
     * measured source-solver transient, not a reverse-drive command. */
    if (state->plant.u < -MPC_PHYSX_BRAKE_REVERSE_TRANSIENT_MPS)
        return MPC_CYCLE_COMMAND_ENVELOPE_SPEED_BELOW_ZERO;
    const VehicleParameters_t vehicle = vehicle_model_get_parameters();
    if (state->plant.u > vehicle.maximum_command_speed_mps)
        return MPC_CYCLE_COMMAND_ENVELOPE_SPEED_ABOVE_MAX;
    if (fabsf(state->plant.delayed_steering_command_1) >
            configuration->max_steering_rad ||
        fabsf(state->plant.delayed_steering_command_2) >
            configuration->max_steering_rad)
        return MPC_CYCLE_COMMAND_ENVELOPE_STEERING_QUEUE;
    if (fabsf(state->plant.actual_steering_angle) >
        configuration->max_steering_rad)
        return MPC_CYCLE_COMMAND_ENVELOPE_ACTUAL_STEERING;
    return MPC_CYCLE_COMMAND_ENVELOPE_OK;
}

static int state_inside_command_envelope(
    const MpcCycleState_t *state,
    const MpcSolverConfiguration_t *configuration)
{
    /* Queued actuator commands are already in flight, not new decisions. */
    return command_envelope_failure(state, configuration) ==
        MPC_CYCLE_COMMAND_ENVELOPE_OK;
}

static int valid_previous_nominal(
    const MpcCycleNominal_t *nominal,
    int horizon)
{
    if (!nominal || !nominal->valid || nominal->horizon < 1 ||
        nominal->horizon > horizon)
        return 0;
    for (int k = 0; k <= nominal->horizon; ++k) {
        if (!finite_cycle_state(&nominal->states[k]) ||
            !isfinite(nominal->progress[k])) return 0;
    }
    for (int k = 0; k < nominal->horizon; ++k)
        if (!isfinite(nominal->controls[k].steering_target_rad) ||
            !isfinite(nominal->controls[k].throttle_command)) return 0;
    return 1;
}

static int seed_unsolved_tail_control(
    const MpcTrajectorySample_t *trajectory, size_t count, double length,
    double station, double next_station, float last_throttle,
    const MpcSolverConfiguration_t *configuration, MpcModelControl_t *control)
{
    MpcTrajectorySample_t sample;
    if (!mpc_trajectory_sample(trajectory, count, length, station, &sample))
        return 0;
    if (sample.source_witness_valid &&
        sample.speed <= configuration->active_speed_ceiling_mps + 1.e-4f) {
        control->steering_target_rad = clampf_cycle((float)sample.source_steering_target,
            -configuration->max_steering_rad, configuration->max_steering_rad);
        control->throttle_command = clampf_cycle((float)sample.source_throttle,
            MPC_THROTTLE_MIN_DRIVE, MPC_THROTTLE_MAX);
        return 1;
    }
    if (!mpc_trajectory_sample(trajectory, count, length, next_station, &sample)) return 0;
    control->steering_target_rad = clampf_cycle(
        vehicle_model_steering_for_curvature((float)sample.curvature),
        -configuration->max_steering_rad, configuration->max_steering_rad);
    control->throttle_command = clampf_cycle(last_throttle,
        MPC_THROTTLE_MIN_DRIVE, MPC_THROTTLE_MAX);
    return 1;
}

static int mpc_cycle_build_nominal_impl(
    const MpcCycleState_t *current_state,
    double current_progress,
    const MpcCycleNominal_t *previous_nominal,
    const MpcModelControl_t *warm_start_controls,
    int warm_start_control_count,
    int stage_shift,
    const MpcTrajectorySample_t *trajectory,
    size_t trajectory_count,
    double lap_length,
    float prediction_dt,
    int horizon,
    const MpcSolverConfiguration_t *configuration,
    MpcCycleNominal_t *nominal,
    MpcCycleReference_t references[PREDICTION_HORIZON + 1])
{
    if (!nominal) return 0;
    // No whole-horizon copy on the normal path: the previous iterate lives in
    // persistent memory, independently of the output. Preserve alias safety
    // only for a caller explicitly building into its own previous nominal.
    MpcCycleNominal_t aliased_previous;
    if (previous_nominal == nominal) {
        aliased_previous = *previous_nominal;
        previous_nominal = &aliased_previous;
    }
    *nominal = {};
    if (!finite_cycle_state(current_state) || !isfinite(current_progress) ||
        !trajectory || !references || horizon < 1 ||
        horizon > PREDICTION_HORIZON || !isfinite(prediction_dt) ||
        prediction_dt <= 0.0f || !valid_configuration(configuration) ||
        ((warm_start_controls == NULL) != (warm_start_control_count == 0)) ||
        warm_start_control_count < 0 ||
        warm_start_control_count > PREDICTION_HORIZON ||
        (warm_start_controls && warm_start_control_count < 1))
        return 0;
    for (int k = 0; warm_start_controls &&
            k < warm_start_control_count; ++k) {
        if (!isfinite(warm_start_controls[k].steering_target_rad) ||
            !isfinite(warm_start_controls[k].throttle_command))
            return 0;
    }

    const int warm = valid_previous_nominal(previous_nominal, horizon);
    nominal->horizon = horizon;
    nominal->states[0] = *current_state;
    /* x0 is measured and cannot be repaired by optimization. A finite future
     * warm-start rollout, however, is only a linearization point: imposing
     * candidate bounds here can abort before the QP can repair its controls.
     * The QP and exact nonlinear acceptance still enforce future bounds. */
    const MpcCycleCommandEnvelopeFailure_t initial_failure =
        command_envelope_failure(current_state, configuration);
    if (initial_failure != MPC_CYCLE_COMMAND_ENVELOPE_OK) {
        return 0;
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
            nominal->controls[k].steering_target_rad = clampf_cycle(
                nominal->controls[k].steering_target_rad,
                -configuration->max_steering_rad,
                configuration->max_steering_rad);
            nominal->controls[k].throttle_command = clampf_cycle(
                nominal->controls[k].throttle_command,
                MPC_THROTTLE_MIN_DRIVE,
                MPC_THROTTLE_MAX);
            if (k + shift >= previous_horizon &&
                !seed_unsolved_tail_control(trajectory, trajectory_count,
                    lap_length, progress_seed[k], progress_seed[k + 1],
                    nominal->controls[k].throttle_command, configuration,
                    &nominal->controls[k]))
                return 0;
        }
    } else {
        TrajectoryReferencePoint_t seed_references[PREDICTION_HORIZON + 1];
        if (!mpc_reference_build_speed_seed(
                trajectory, trajectory_count, lap_length, current_progress,
                configuration->active_speed_ceiling_mps, prediction_dt,
                horizon, seed_references, progress_seed))
            return 0;
        for (int k = 0; k <= horizon; ++k)
            if (!mpc_cycle_reference_at_progress(trajectory, trajectory_count,
                    lap_length, progress_seed[k], configuration,
                    &references[k]))
                return 0;

        /* Seed throttle deterministically from the last accepted value; the
         * optimized input is applied directly at each command-time stage.
         * Avoid plant transitions solely to construct this seed. */
        float queued_throttle = current_state->plant.delayed_throttle_command_2;
        for (int k = 0; k < horizon; ++k) {
            const float desired_delta = references[k].source_witness_valid ?
                references[k].source_steering_target : references[k + 1].steering_command;
            nominal->controls[k].steering_target_rad = clampf_cycle(
                desired_delta,
                -configuration->max_steering_rad,
                configuration->max_steering_rad);
            nominal->controls[k].throttle_command = clampf_cycle(
                references[k].source_witness_valid ? references[k].source_throttle :
                    fmaxf(queued_throttle, MPC_THROTTLE_MIN_DRIVE),
                MPC_THROTTLE_MIN_DRIVE,
                MPC_THROTTLE_MAX);
            /* Carry the seed value forward as the next stage's initial guess;
             * the nonlinear plant consumes each stage's direct input. */
            queued_throttle = nominal->controls[k].throttle_command;
        }
    }

    /* A shifted nominal can retain minimum-drive throttle after a rejected
     * or heavily repaired cycle, leaving the car effectively stopped even
     * though the trajectory still requests acceleration. Re-seed both cold
     * and warm nominals from rest; this is only a linearization seed. SQP
     * may change it, and the full nonlinear rollout/corridor validation
     * remains the only path to a published command. */
    MpcCycleReference_t launch_reference = {};
    if (current_state->plant.u < 0.5f &&
        configuration->startup_nominal_throttle_seed >
            MPC_THROTTLE_MIN &&
        mpc_cycle_reference_at_progress(trajectory, trajectory_count, lap_length,
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
                return 0;
            }
            nominal->controls[k].steering_target_rad = clampf_cycle(
                nominal->controls[k].steering_target_rad,
                -configuration->max_steering_rad,
                configuration->max_steering_rad);
            nominal->controls[k].throttle_command = clampf_cycle(
                nominal->controls[k].throttle_command,
                MPC_THROTTLE_MIN_DRIVE,
                MPC_THROTTLE_MAX);
        }
    }

    for (int k = 0; k < horizon; ++k)
        nominal->controls[k].throttle_command = clampf_cycle(
            nominal->controls[k].throttle_command,
            MPC_THROTTLE_MIN_DRIVE,
            MPC_THROTTLE_MAX);

    // Rate-limit the nominal command seed before rollout. Mixing an
    // unbounded seed with a rate-limited QP solution can leave full-lock
    // reversals in the accepted line-search trajectory. Seed and candidate
    // use the same nonlinear plant.
    /* One exact loaded-frame rollout gives both the nominal states and their
     * final progress for reference sampling. */
    if (!rollout_nominal_controls(current_state, current_progress,
            nominal->controls,
            trajectory, trajectory_count, lap_length, horizon, prediction_dt,
            nominal->states, nominal->progress))
        return 0;

    for (int k = 0; k <= horizon; ++k) {
        if (!mpc_cycle_reference_at_progress(trajectory, trajectory_count, lap_length,
                nominal->progress[k], configuration, &references[k]))
            return 0;
    }
    /* The initial condition is fixed by odometry; only future samples track
     * the requested trajectory speed. */
    references[0].u = current_state->plant.u;
    /* Keep the requested speed as the target. A short reachable-speed envelope
     * collapses toward the measured speed at rest and suppresses acceleration.
     * The nonlinear rollout checks reachability, command delay and wheel spin. */
    align_yaw_rate_references(references, horizon);
    for (int k = 0; k <= horizon; ++k)
        nominal->speed_reference[k] = references[k].u;
    nominal->speed_reference_valid = 1;

    nominal->valid = 1;
    return 1;
}

int mpc_cycle_build_nominal_same_epoch(
    const MpcCycleState_t *current_state,
    double current_progress,
    const MpcCycleNominal_t *previous_nominal,
    const MpcModelControl_t *same_epoch_controls,
    int same_epoch_control_count,
    const MpcTrajectorySample_t *trajectory,
    size_t trajectory_count,
    double lap_length,
    float prediction_dt,
    int horizon,
    const MpcSolverConfiguration_t *configuration,
    MpcCycleNominal_t *nominal,
    MpcCycleReference_t references[PREDICTION_HORIZON + 1])
{
    return mpc_cycle_build_nominal_impl(current_state,current_progress,
        previous_nominal,same_epoch_controls,same_epoch_control_count,0,
        trajectory,trajectory_count,lap_length,prediction_dt,horizon,
        configuration,nominal,references);
}

static MpcCycleRolloutStatus_t rollout_candidate_impl(
    const MpcCycleState_t *initial_state,
    double initial_progress,
    const MpcModelControl_t controls[PREDICTION_HORIZON],
    const MpcTrajectorySample_t *trajectory,
    size_t trajectory_count,
    double lap_length,
    int horizon,
    float prediction_dt,
    const MpcSolverConfiguration_t *configuration,
    MpcCycleState_t states[PREDICTION_HORIZON + 1],
    double progress[PREDICTION_HORIZON + 1],
    int *failure_stage)
{
    if (failure_stage) *failure_stage = -1;

    if (!finite_cycle_state(initial_state) || !isfinite(initial_progress) ||
        !controls || !trajectory || trajectory_count < 3 ||
        !isfinite(lap_length) || !(lap_length > 0.0) ||
        !states || !valid_configuration(configuration) || horizon < 1 ||
        horizon > PREDICTION_HORIZON || !isfinite(prediction_dt) ||
        prediction_dt <= 0.0f) return MPC_CYCLE_ROLLOUT_INVALID_INPUT;
    states[0] = *initial_state;
    if (!state_inside_command_envelope(&states[0], configuration))
        return MPC_CYCLE_ROLLOUT_INVALID_INPUT;
    double candidate_progress[PREDICTION_HORIZON + 1] = {0.0};
    candidate_progress[0] = initial_progress;
    MpcCycleReference_t candidate_reference = {};
    if (!reference_at_progress_with_sample(
            trajectory, trajectory_count, lap_length,
            candidate_progress[0], configuration, &candidate_reference,
            NULL))
        return MPC_CYCLE_ROLLOUT_INVALID_INPUT;

    /* x0 is measured and immutable. The QP applies corridor constraints only
     * to x1..xN, so exact validation follows that same convention. */

    if (progress) progress[0] = initial_progress;

    for (int k = 0; k < horizon; ++k) {
        MpcModelControl_t adjusted = controls[k];
        const MpcModelControl_t *control = &adjusted;
        if (!isfinite(control->steering_target_rad) ||
            !isfinite(control->throttle_command)) {
            if (failure_stage) *failure_stage = k;
            return MPC_CYCLE_ROLLOUT_INVALID_INPUT;
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
            return MPC_CYCLE_ROLLOUT_COMMAND_LIMIT;
        }
        /* Reuse the reference and speed override sampled at candidate_progress[k]
         * at the preceding stage boundary (before the loop for k=0).
         * Corridor checks below sample each newly reached state. */
        MpcStageResult_t step = mpc_vehicle_model_step(
            &states[k].plant, control, prediction_dt,
            0.0f);
        if (!step.valid) {
            if (failure_stage) *failure_stage = k;
            return MPC_CYCLE_ROLLOUT_INVALID_MODEL;
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
            return MPC_CYCLE_ROLLOUT_INVALID_MODEL;
        }
        states[k + 1].incoming_tracking_jacobian_valid = 1;
        states[k + 1].plant = step.next;
        states[k + 1].incoming_body_forward_delta_m =
            step.body_forward_delta_m;
        states[k + 1].incoming_body_lateral_delta_m =
            step.body_lateral_delta_m;
        states[k + 1].incoming_body_yaw_delta_rad =
            step.body_yaw_delta_rad;
        store_grip_metrics(&states[k + 1], &step);
        states[k + 1].incoming_transition_valid = 1;
        candidate_progress[k + 1] = next_progress;
        if (progress) progress[k + 1] = candidate_progress[k + 1];
        if (!state_inside_command_envelope(&states[k + 1], configuration)) {
            if (failure_stage) *failure_stage = k;
            return MPC_CYCLE_ROLLOUT_STATE_LIMIT;
        }
        if (!reference_at_progress_with_sample(
                trajectory, trajectory_count, lap_length,
                candidate_progress[k + 1], configuration, &candidate_reference,
                NULL, &projected_sample)) {
            if (failure_stage) *failure_stage = k;
            return MPC_CYCLE_ROLLOUT_INVALID_INPUT;
        }
        if (!finite_reference(&candidate_reference)) {
            if (failure_stage) *failure_stage = k;
            return MPC_CYCLE_ROLLOUT_INVALID_INPUT;
        }

        const MpcCorridorMargins margin = corridor_margin_for_state(configuration,
            &states[k + 1].plant, k + 1);
        const float tolerance = corridor_tolerance_at_prediction(configuration, k + 1);
        const float lower_bound = margin.right - candidate_reference.right_bound -
            tolerance;
        const float upper_bound = candidate_reference.left_bound - margin.left +
            tolerance;

        int in_corridor = isfinite(lower_bound) && isfinite(upper_bound) &&
            lower_bound <= upper_bound &&
            states[k + 1].plant.e_y >= lower_bound &&
            states[k + 1].plant.e_y <= upper_bound;
        if (!in_corridor) {
            if (failure_stage) *failure_stage = k;
            return MPC_CYCLE_ROLLOUT_CORRIDOR;
        }
    }
    return MPC_CYCLE_ROLLOUT_OK;
}

void mpc_cycle_memory_reset(MpcCycleMemory_t *memory)
{
    if (!memory) return;
    memset(memory, 0, sizeof(*memory));
}

int mpc_cycle_memory_set_warm_start_controls(
    MpcCycleMemory_t *memory,
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

int mpc_cycle_memory_set_warm_start_controls_at_epoch(
    MpcCycleMemory_t *memory,const MpcModelControl_t *controls,
    int control_count,int64_t command_epoch_ns)
{
    if(command_epoch_ns<0 || !mpc_cycle_memory_set_warm_start_controls(
            memory,controls,control_count)) return 0;
    memory->command_epoch_ns=command_epoch_ns;
    return 1;
}

int mpc_cycle_command_epoch_shift(int64_t previous_ns,int64_t current_ns,
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
    const MpcCycleConfiguration_t *cfg)
{
    if(!cfg || cfg->command_epoch_ns<0 ||
       !valid_configuration(&cfg->model) ||
       cfg->sqp_iterations<1 || cfg->sqp_iterations>MPC_CYCLE_SOURCE_TRACE_CAPACITY ||
       !isfinite(cfg->sqp_violation_penalty) ||
       cfg->sqp_violation_penalty<=0.f ||
       cfg->sqp_violation_penalty>1.e9f ||
       !isfinite(cfg->sqp_accuracy_tolerance) ||
       cfg->sqp_accuracy_tolerance<=0.f ||
       cfg->sqp_accuracy_tolerance>1.f ||
       cfg->derivative_force_substeps<0 || cfg->derivative_force_substeps>5 ||
       !finite_nonnegative(cfg->publication_far_tolerance_m) ||
       cfg->publication_tolerance_start_stage<0 ||
       cfg->publication_tolerance_start_stage>=PREDICTION_HORIZON ||
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

double mpc_cycle_monotonic_time_us(void)
{
    return monotonic_microseconds();
}

int mpc_cycle_build_objective_schedule(
    const MpcCycleNominal_t *nominal,
    const MpcCycleReference_t references[PREDICTION_HORIZON + 1],
    const MpcSolverConfiguration_t *configuration,
    MpcCycleObjectiveSchedule_t *schedule)
{
    if (!nominal || !references || !configuration || !schedule ||
        !nominal->valid || nominal->horizon < 1 ||
        nominal->horizon > PREDICTION_HORIZON) return 0;
    memset(schedule, 0, sizeof(*schedule));
    schedule->horizon = nominal->horizon;
    for (int k = 0; k <= nominal->horizon; ++k) {
        schedule->references[k] = references[k];
        schedule->e_y_weights[k] = lateral_weight_for_reference(configuration, &references[k]);
        schedule->e_psi_weights[k] = heading_weight_for_reference(configuration, &references[k]);
        schedule->speed_weights[k] = nominal->states[k].plant.u > references[k].u
            ? overspeed_weight_for_reference(configuration, &references[k])
            : underspeed_weight_for_reference(configuration, &references[k]);
    }
    schedule->valid = 1;
    return 1;
}

int mpc_cycle_score_candidate_with_references(
    const MpcCycleNominal_t *candidate,
    const MpcTrajectorySample_t *trajectory,
    size_t trajectory_count,
    double lap_length,
    const MpcCycleConfiguration_t *configuration,
    const MpcCycleStageReference stages[PREDICTION_HORIZON + 1],
    double *objective,
    float *minimum_corridor_slack,
    int allow_hard_constraint_violations_for_metrics,
    float (*source_corridor_by_stage)[2],
    MpcCycleGripMetrics_t *grip_metrics,
    MpcCycleSourceCorridorCorners_t *corners)
{
    if (!candidate || !trajectory || !configuration || !objective ||
        !minimum_corridor_slack || !stages ||
        !candidate->valid || candidate->horizon < 1 ||
        candidate->horizon > PREDICTION_HORIZON) return 0;

    if (grip_metrics) *grip_metrics = {};
    double cost = 0.0;
    double previous_steering =
        candidate->states[0].plant.delayed_steering_command_2;
    double previous_throttle =
        candidate->states[0].plant.delayed_throttle_command_2;
    float min_slack = INFINITY;
    const MpcSolverConfiguration_t *model = &configuration->model;
    const MpcCycleObjectiveSchedule_t *schedule = configuration->objective_schedule;
    if (schedule && (!schedule->valid || schedule->horizon < candidate->horizon))
        return 0;
    const auto reference_at = [&](int k) -> const MpcCycleReference_t & {
        return schedule ? schedule->references[k] : stages[k].reference;
    };
    for (int k = 0; k <= candidate->horizon; ++k) {
        const MpcCycleReference_t *reference = &reference_at(k);
        float stage_corridor_slack = INFINITY;
        if (k > 0) {
            float lower_slack, upper_slack;
            const int bounds_ok=mpc_cycle_compute_corridor_slack(
                &candidate->states[k].plant,
                stages[k].reference.left_bound,stages[k].reference.right_bound,model,k,
                &lower_slack,&upper_slack);
            if(source_corridor_by_stage) {
                // An invalid footprint query leaves no physical left/right margins.
                // Reject it rather than supplying zero-filled geometry to the QP.
                if(!bounds_ok || !std::isfinite(lower_slack) ||
                   !std::isfinite(upper_slack))return 0;
                source_corridor_by_stage[k][0]=lower_slack;
                source_corridor_by_stage[k][1]=upper_slack;
            }
            if(corners && (!bounds_ok || !corridor_corners_with_reference(
                    &candidate->states[k].plant,trajectory,trajectory_count,
                    lap_length,candidate->progress[k],model,k,stages[k],
                    lower_slack,upper_slack,&corners[k-1])))return 0;
            const int frenet_ok=bounds_ok &&
                lower_slack>=0.0f && upper_slack>=0.0f;
            if (!frenet_ok && !allow_hard_constraint_violations_for_metrics)
                return 0;
            stage_corridor_slack = fminf(lower_slack, upper_slack);
            min_slack = fminf(min_slack, stage_corridor_slack);
        }
        const MpcModelState_t *state = &candidate->states[k].plant;
        const double terminal_scale = k == candidate->horizon
            ? model->terminal_multiplier : 1.0;
        const double errors[] = {
            state->e_y - reference->e_y,
            state->e_psi - reference->e_psi,
            state->u - reference->u,
            state->v - reference->v,
            state->r - reference->r,
            state->delayed_steering_command_2 - reference->steering_command,
            state->actual_steering_angle - reference->steering_command};
        const double speed_weight = schedule ? schedule->speed_weights[k] :
            (state->u > reference->u
                ? overspeed_weight_for_reference(model, reference)
                : underspeed_weight_for_reference(model, reference));
        const double lateral_weight = schedule ? schedule->e_y_weights[k] :
            lateral_weight_for_reference(model, reference);
        const double heading_weight = schedule ? schedule->e_psi_weights[k] :
            heading_weight_for_reference(model, reference);
        const double weights[] = {
            lateral_weight, heading_weight, speed_weight,
            model->weight_v, model->weight_r,
            model->weight_steering_command,
            model->weight_steering_command};
        for (size_t i = 0; i < sizeof(errors) / sizeof(errors[0]); ++i)
            cost += terminal_scale * weights[i] * errors[i] * errors[i];

        if (k > 0) {
            for (int wheel = 0; wheel < 4; ++wheel) {
                const double excess = fmax(0.0,
                    candidate->states[k].incoming_wheel_grip_loss[wheel] -
                    (double)model->grip_loss_tolerance * model->grip_loss_tolerance);
                const double wheel_cost = terminal_scale * model->weight_grip_loss * excess * excess;
                cost += wheel_cost;
                if (grip_metrics) {
                    grip_metrics->excess[k - 1][wheel] = excess;
                    grip_metrics->cost += wheel_cost;
                }
            }
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
            steering_reference_increment(&reference_at(k), &reference_at(k + 1));
        cost += model->weight_steering_slew *
                steering_correction_delta * steering_correction_delta +
            model->weight_throttle_slew *
                throttle_delta * throttle_delta;
        previous_steering = control->steering_target_rad;
        previous_throttle = control->throttle_command;
    }
    if (!isfinite(cost) || !isfinite(min_slack)) return 0;
    *objective = cost;
    *minimum_corridor_slack = min_slack;
    return 1;
}

static int evaluate_nonlinear_candidate(
    const MpcCycleNominal_t *candidate,
    const MpcTrajectorySample_t *trajectory, size_t trajectory_count,
    double lap_length, const MpcCycleConfiguration_t *configuration,
    double *objective, float *minimum_corridor_slack,
    int allow_hard_constraint_violations_for_metrics=0,
    float (*source_corridor_by_stage)[2]=nullptr,
    MpcCycleGripMetrics_t *grip_metrics=nullptr)
{
    if(!candidate || !trajectory || !configuration || !objective ||
       !minimum_corridor_slack || !candidate->valid ||
       candidate->horizon<1 || candidate->horizon>PREDICTION_HORIZON)return 0;
    if(grip_metrics)*grip_metrics={};
    const auto *schedule=configuration->objective_schedule;
    if(schedule && (!schedule->valid || schedule->horizon<candidate->horizon))return 0;
    MpcCycleStageReference stages[PREDICTION_HORIZON + 1];
    for(int k=0;k<=candidate->horizon;++k) {
        if(!mpc_cycle_sample_stage_reference(trajectory,trajectory_count,lap_length,
                candidate->progress[k],&configuration->model,&stages[k]))return 0;
        auto &ref=stages[k].reference;
        if(candidate->speed_reference_valid)ref.u=candidate->speed_reference[k];
        if(!ref.source_witness_valid)ref.r=ref.path_curvature*ref.u;
    }
    return mpc_cycle_score_candidate_with_references(candidate,trajectory,
        trajectory_count,lap_length,configuration,stages,objective,
        minimum_corridor_slack,allow_hard_constraint_violations_for_metrics,
        source_corridor_by_stage,grip_metrics,nullptr);
}

int minimum_candidate_corridor_slack(
    const MpcCycleNominal_t *candidate,
    const MpcTrajectorySample_t *trajectory, size_t trajectory_count,
    double lap_length, const MpcSolverConfiguration_t *configuration,
    float *minimum_slack)
{
    if (!candidate || !candidate->valid || !trajectory || !configuration ||
        !minimum_slack || candidate->horizon < 1 ||
        candidate->horizon > PREDICTION_HORIZON) return 0;
    float minimum = INFINITY;
    for (int k = 0; k <= candidate->horizon; ++k) {
        MpcCycleReference_t reference;
        if (!mpc_cycle_reference_at_progress(trajectory, trajectory_count, lap_length,
                candidate->progress[k], configuration, &reference)) return 0;
        if (k == 0) continue;
        float lower, upper;
        if (!mpc_cycle_compute_corridor_slack(&candidate->states[k].plant,
                reference.left_bound, reference.right_bound, configuration, k,
                &lower, &upper) || !std::isfinite(lower) || !std::isfinite(upper)) return 0;
        minimum = fminf(minimum, fminf(lower, upper));
    }
    if (!std::isfinite(minimum)) return 0;
    *minimum_slack = minimum;
    return 1;
}

int mpc_cycle_score_candidate_plan_and_corridor_for_diagnostics(
    const MpcCycleNominal_t *candidate,
    const MpcTrajectorySample_t *trajectory, size_t trajectory_count,
    double lap_length, const MpcCycleConfiguration_t *configuration,
    double *objective, float *minimum_corridor_slack_m,
    float corridor_by_stage[PREDICTION_HORIZON+1][2],
    MpcCycleGripMetrics_t *grip_metrics)
{
    if(!candidate || !candidate->valid || !configuration || !objective ||
       !minimum_corridor_slack_m || !corridor_by_stage)return 0;
    return evaluate_nonlinear_candidate(candidate,trajectory,trajectory_count,
        lap_length,configuration,objective,minimum_corridor_slack_m,
        1,
        corridor_by_stage,grip_metrics);
}

MpcCycleRolloutStatus_t mpc_cycle_validate_candidate_plan(
    const MpcCycleState_t *initial_state,
    double initial_progress,
    const MpcModelControl_t controls[PREDICTION_HORIZON],
    const MpcTrajectorySample_t *trajectory,
    size_t trajectory_count,
    double lap_length,
    int horizon,
    float prediction_dt,
    const MpcCycleConfiguration_t *configuration,
    MpcCycleNominal_t *validated_plan,
    MpcCyclePlanMetrics_t *metrics,
    int *failure_stage)
{
    if (!validated_plan || !configuration)
        return MPC_CYCLE_ROLLOUT_INVALID_INPUT;
    memset(validated_plan, 0, sizeof(*validated_plan));
    validated_plan->horizon = horizon;
    validated_plan->speed_reference_valid = 0;
    MpcCycleRolloutStatus_t status = rollout_candidate_impl(
        initial_state, initial_progress, controls, trajectory,
        trajectory_count, lap_length, horizon,
        prediction_dt, &configuration->model, validated_plan->states,
        validated_plan->progress, failure_stage);
    if (status != MPC_CYCLE_ROLLOUT_OK) return status;
    memcpy(validated_plan->controls, controls,
        (size_t)horizon * sizeof(validated_plan->controls[0]));
    validated_plan->valid = 1;

    // Revalidate the plan against the nonlinear program.
    MpcCycleConfiguration_t local_configuration=*configuration;
    double objective = 0.0;
    float minimum_corridor_slack = 0.0f;
    if (!evaluate_nonlinear_candidate(validated_plan, trajectory,
            trajectory_count, lap_length, &local_configuration, &objective,
            &minimum_corridor_slack)) {
        validated_plan->valid = 0;
        return MPC_CYCLE_ROLLOUT_INVALID_MODEL;
    }
    if (metrics) *metrics = {objective, minimum_corridor_slack};
    return MPC_CYCLE_ROLLOUT_OK;
}

MpcCycleCycleStatus_t mpc_cycle_solve(
    const MpcCycleState_t *current_state,
    double current_progress,
    const MpcTrajectorySample_t *trajectory,
    size_t trajectory_count,
    double lap_length,
    float prediction_dt,
    int horizon,
    const MpcCycleConfiguration_t *configuration,
    MpcCycleMemory_t *memory,
    MpcCycleCycleResult_t *result)
{
    const double cycle_start_us = monotonic_microseconds();
    if (result) {
        memset(result, 0, sizeof(*result));
        result->nonlinear_failure_stage = -1;
        result->status = MPC_CYCLE_CYCLE_REJECTED_INPUT;
        result->source_solver_status = MPC_SOURCE_INVALID_MODEL;

        result->total_cycle_us = 0.0;
    }
    if (!result || !memory || !finite_cycle_state(current_state) ||
        !isfinite(current_progress) || !trajectory || trajectory_count == 0 ||
        !isfinite(lap_length) || lap_length <= 0.0 ||
        !valid_cycle_configuration(configuration) ||
        horizon != PREDICTION_HORIZON || !isfinite(prediction_dt) ||
        prediction_dt <= 0.0f) {
        if (result) result->total_cycle_us = monotonic_microseconds() - cycle_start_us;
        return MPC_CYCLE_CYCLE_REJECTED_INPUT;
    }

    MpcCycleState_t prepared_state = *current_state;
    if (!vehicle_model_initialize_plant_state(&prepared_state.plant)) {

        result->total_cycle_us = monotonic_microseconds() - cycle_start_us;
        return MPC_CYCLE_CYCLE_REJECTED_INPUT;
    }

    const MpcCycleMemory_t memory_before = *memory;
    const int epoch_shift=configuration->command_epoch_ns>0
        ? mpc_cycle_command_epoch_shift(memory->command_epoch_ns,
            configuration->command_epoch_ns,prediction_dt) : 1;
    if(epoch_shift<0) mpc_cycle_memory_reset(memory);

    MpcCycleConfiguration_t aligned_configuration=*configuration;
    configuration=&aligned_configuration;
    const double deadline_us =
        configuration->absolute_deadline_monotonic_us > 0.0
        ? configuration->absolute_deadline_monotonic_us
        : (configuration->max_cycle_time_us > 0.0
            ? cycle_start_us + configuration->max_cycle_time_us : 0.0);
    MpcCycleNominal_t nominal;
    MpcCycleReference_t references[PREDICTION_HORIZON + 1];
    double nominal_start_us = monotonic_microseconds();
    const int warm_controls_valid = memory->warm_start_controls_valid &&
        memory->warm_start_control_count >= 1 &&
        memory->warm_start_control_count <= PREDICTION_HORIZON;
    const int nominal_built=mpc_cycle_build_nominal_impl(&prepared_state,
        current_progress,&memory->nominal,
        warm_controls_valid ? memory->warm_start_controls : nullptr,
        warm_controls_valid ? memory->warm_start_control_count : 0,
        epoch_shift<0 ? 0 : epoch_shift,
        trajectory,trajectory_count,lap_length,prediction_dt,horizon,
        &configuration->model,&nominal,references);
    // Record these common measurements for both success and failure; only
    // failure diagnostics need an early return below.
    result->nominal_build_us = monotonic_microseconds() - nominal_start_us;

    if (!nominal_built) {

        result->total_cycle_us = monotonic_microseconds() - cycle_start_us;
        return MPC_CYCLE_CYCLE_REJECTED_INPUT;
    }
    result->nominal_first_control = nominal.controls[0];
    result->nominal_control_count = horizon;

    memcpy(result->nominal_controls, nominal.controls,
        (size_t)horizon * sizeof(result->nominal_controls[0]));

    if (!mpc_cycle_build_objective_schedule(&nominal, references,
            &configuration->model, &result->comparison_objective))
        return MPC_CYCLE_CYCLE_REJECTED_INPUT;
    MpcCycleConfiguration_t comparison_configuration = *configuration;
    comparison_configuration.objective_schedule = &result->comparison_objective;

    // One anytime trust-region SQP over the exact source rollout. The best
    // exactly verified iterate is always returned; the cycle publishes it
    // when feasible and the publication deadline still holds.
    constexpr double kPublishReserveUs = 3500.;
    constexpr int kExactDerivativeCycles = 40;
    const double solve_begin = monotonic_microseconds();

    sdu_apex::source_sqp::Options options{};
    options.maximum_iterations = configuration->sqp_iterations;
    options.same_epoch_nominal = &nominal;
    options.violation_penalty = configuration->sqp_violation_penalty;
    options.accuracy_tolerance = configuration->sqp_accuracy_tolerance;
    options.deadline_us = deadline_us > 0.
        ? std::max(0., deadline_us - kPublishReserveUs) : 0.;
    if (memory->sqp_radius > 0.) options.radius = memory->sqp_radius;
    options.exact_derivatives = memory->sqp_exact_derivative_cycles > 0;
    options.curvature_cache = &memory->curvature_cache;
    options.command_epoch_ns = configuration->command_epoch_ns;
    auto solved = std::make_unique<sdu_apex::source_sqp::Result>();
    const bool seeded = sdu_apex::source_sqp::solve(
        prepared_state, current_progress, trajectory, trajectory_count,
        lap_length, prediction_dt, comparison_configuration,
        nominal.controls, options, solved.get());
    result->total_solve_us = monotonic_microseconds() - solve_begin;
    result->source_exact_nlp_deadline_exhausted =
        solved->termination == sdu_apex::source_sqp::Termination::Deadline;
    result->source_solver_status = !seeded ? MPC_SOURCE_INVALID_MODEL :
        solved->termination == sdu_apex::source_sqp::Termination::Deadline
            ? MPC_SOURCE_DEADLINE_EXCEEDED :
        solved->final_feasible ? (solved->converged
            ? MPC_SOURCE_FEASIBLE_SOLUTION_AVAILABLE : MPC_SOURCE_KKT_NOT_CONVERGED)
            : MPC_SOURCE_PRIMAL_INFEASIBLE;
    result->source_exact_nlp_source_trial_evaluations = solved->rollouts;
    result->source_exact_nlp_accepted_iterations = solved->accepted_iterations;
    result->source_exact_nlp_initial_source_feasible = solved->initial_feasible;
    result->source_exact_nlp_final_source_feasible = solved->final_feasible;
    result->source_exact_nlp_kkt_converged = solved->converged;
    result->source_exact_nlp_initial_source_cost = solved->initial_cost;
    result->source_exact_nlp_final_source_cost = solved->final_cost;
    result->source_exact_nlp_final_original_margin_m = solved->final_minimum_corridor_m;
    result->source_exact_nlp_final_original_violation = solved->final_violation;
    result->source_iteration_metrics_count = static_cast<int>(std::min(
        solved->history.size(), size_t(MPC_CYCLE_SOURCE_TRACE_CAPACITY)));
    for (int i = 0; i < result->source_iteration_metrics_count; ++i) {
        const auto &in = solved->history[i];
        auto &out = result->source_iteration_metrics[i];
        out.feasible_before = in.feasible_before;
        out.accurate = in.accurate;
        out.accepted = in.accepted;
        out.trial_evaluated = in.cost_trial > 0.;
        out.rejected_trials = in.rejected_trials;
        out.qp_rounds = in.qp_rounds;
        out.qp_iterations = in.qp_iterations;
        out.newly_activated_grip_wheels = in.newly_active_grip;
        out.radius = in.radius;
        out.radius_active = in.radius_active;
        out.linearization_ms = in.linearization_ms;
        out.qp_ms = in.qp_ms;
        out.rollout_ms = in.rollout_ms;
        out.total_ms = in.total_ms;
        out.cost_before = in.cost_before;
        out.cost_trial = in.cost_trial;
        out.cost_after = in.accepted ? in.cost_trial : in.cost_before;
        out.predicted_reduction = in.predicted_reduction;
        out.actual_over_predicted = in.ratio;
        out.grip_cost_before = in.grip_cost_before;
        out.grip_cost_trial = in.grip_cost_trial;
    }

    result->sqp_iterations_attempted = static_cast<int>(solved->history.size());
    result->sqp_restarted = solved->restarted ? 1 : 0;
    result->sqp_seed_failure = static_cast<int>(solved->seed_failure);

    result->total_cycle_us = monotonic_microseconds() - cycle_start_us;
    if (!seeded) {
        *memory = memory_before;
        result->status = MPC_CYCLE_CYCLE_REJECTED_SOLVER;
        return result->status;
    }
    // Clearance is a reserve to restore, not permission to cross a physical
    // wall. Independently validate every publication against all zero-reserve,
    // physical rows (zero tolerance up to the far-stage ramp), including a
    // reserve-restoration iterate.
    auto physical_configuration = comparison_configuration;
    physical_configuration.model.corridor_margin_m = 0.f;
    physical_configuration.model.first_prediction_corridor_margin_m = 0.f;
    physical_configuration.model.nonlinear_corridor_tolerance_m = 0.f;
    physical_configuration.model.far_stage_corridor_tolerance_m =
        configuration->publication_far_tolerance_m;
    physical_configuration.model.far_stage_tolerance_start_stage =
        configuration->publication_tolerance_start_stage;
    float initial_physical_margin;
    if (!minimum_candidate_corridor_slack(
            &nominal, trajectory, trajectory_count, lap_length,
            &physical_configuration.model, &initial_physical_margin)) {
        *memory = memory_before;
        result->status = MPC_CYCLE_CYCLE_REJECTED_INPUT;
        result->total_cycle_us = monotonic_microseconds() - cycle_start_us;
        return MPC_CYCLE_CYCLE_REJECTED_INPUT;
    }
    result->source_exact_nlp_initial_source_feasible = initial_physical_margin >= 0.f;
    result->source_exact_nlp_final_reserve_satisfied = solved->final_feasible;
    const auto &final_controls = solved->optimal.controls;
    result->source_optimized_control_count = horizon;
    std::copy_n(final_controls, horizon, result->source_optimized_controls);
    auto checked = std::make_unique<MpcCycleNominal_t>();
    MpcCyclePlanMetrics_t validation_metrics{};
    int failure_stage = -1;
    const double validation_begin = monotonic_microseconds();
    const auto physical_validation = mpc_cycle_validate_candidate_plan(
        &prepared_state, current_progress, final_controls, trajectory,
        trajectory_count, lap_length, horizon, prediction_dt,
        &physical_configuration, checked.get(), &validation_metrics, &failure_stage);
    result->validation_us = monotonic_microseconds() - validation_begin;
    result->nonlinear_rollout_status = physical_validation;
    result->nonlinear_failure_stage = failure_stage;
    const bool feasible = physical_validation == MPC_CYCLE_ROLLOUT_OK && checked->valid;
    float physical_margin = validation_metrics.minimum_corridor_slack_m;
    if (!feasible) {
        *checked = solved->optimal;
        double checked_cost;
        float physical_by_stage[PREDICTION_HORIZON + 1][2]{};
        if (!mpc_cycle_score_candidate_plan_and_corridor_for_diagnostics(
                checked.get(), trajectory, trajectory_count, lap_length,
                &physical_configuration, &checked_cost, &physical_margin, physical_by_stage, nullptr)) {
            *memory = memory_before;
            result->status = MPC_CYCLE_CYCLE_REJECTED_INPUT;
            result->total_cycle_us = monotonic_microseconds() - cycle_start_us;
            return MPC_CYCLE_CYCLE_REJECTED_INPUT;
        }
    }
    result->source_exact_nlp_final_physical_margin_m = physical_margin;
    result->source_exact_nlp_final_source_feasible = feasible && physical_margin >= 0.f;
    if (!feasible || physical_margin < 0.f) {
        *memory = memory_before;
        result->status = MPC_CYCLE_CYCLE_REJECTED_NONLINEAR_ROLLOUT;
        result->total_cycle_us = monotonic_microseconds() - cycle_start_us;
        return result->status;
    }
    if (feasible && result->source_solver_status == MPC_SOURCE_PRIMAL_INFEASIBLE)
        result->source_solver_status = MPC_SOURCE_KKT_NOT_CONVERGED;
    result->total_cycle_us = monotonic_microseconds() - cycle_start_us;
    if (deadline_us > 0. && monotonic_microseconds() >= deadline_us) {
        // A late control is not the control applied in its source epoch.
        *memory = memory_before;
        result->source_exact_nlp_deadline_exhausted = 1;
        result->source_solver_status = MPC_SOURCE_DEADLINE_EXCEEDED;
        result->status = MPC_CYCLE_CYCLE_WAITING_FOR_SOLVER_BUDGET;
        return result->status;
    }
    result->candidate_control_count = horizon;
    std::copy_n(final_controls, horizon, result->candidate_controls);
    result->first_control = result->candidate_controls[0];
    result->published_steering_command = result->first_control.steering_target_rad;
    result->published_throttle = result->first_control.throttle_command;
    memory->nominal = *checked;
    memory->sqp_radius = solved->radius;
    // A plan the capped derivatives cannot describe (a coast at low speed) stays on exact
    // derivatives for a while, instead of wasting an iteration on rejected trials every cycle.
    memory->sqp_exact_derivative_cycles = solved->exact_derivatives ? kExactDerivativeCycles :
        std::max(0, memory->sqp_exact_derivative_cycles - 1);
    if (!mpc_cycle_memory_set_warm_start_controls_at_epoch(
            memory, result->candidate_controls, horizon,
            configuration->command_epoch_ns)) {
        *memory = memory_before;
        result->candidate_control_count = 0;
        result->status = MPC_CYCLE_CYCLE_REJECTED_INPUT;
        return result->status;
    }
    result->total_cycle_us = monotonic_microseconds() - cycle_start_us;
    result->status = MPC_CYCLE_CYCLE_ACCEPTED_FEASIBLE;
    return result->status;
}
