/**
 * @file vehicle_model.c
 * @brief Dynamic nonlinear bicycle model implementation.
 * @dependencies vehicle_model.h, <stdio.h>, <string.h>, <stdlib.h>
 */

#include "vehicle_model.h"

static VehicleParameters_t active_parameters = {
    .wheelbase_meters = VP_WHEELBASE_M,
    .distance_cg_to_front_axle = VP_CG_TO_FRONT_AXLE_M,
    .distance_cg_to_rear_axle = VP_CG_TO_REAR_AXLE_M,
    .height_cg_to_ground = VP_CG_HEIGHT_M,
    .vehicle_mass = VP_MASS_KG,
    .yaw_moment_of_inertia = VP_YAW_INERTIA_KGM2,
    .front_cornering_stiffness = VP_FRONT_CORNERING_STIFFNESS,
    .rear_cornering_stiffness = VP_REAR_CORNERING_STIFFNESS,
    .max_steering_angle = VP_MAX_STEERING_RAD,
    .max_velocity = VP_MAX_VELOCITY_MPS,
    .min_velocity = 0.0f,
    .max_acceleration = VP_MAX_ACCEL_MPS2,
    .min_acceleration = VP_MIN_ACCEL_MPS2,
    .friction_coefficient = VP_FRICTION_COEFF,
    .gravity_mps2 = GRAVITY_MPS2,
    .tire_shape_factor = VP_C_SHAPE,
    .minimum_slip_velocity = MIN_SLIP_VELOCITY,
    .minimum_stiffness_scale = MIN_STIFF_SCALE,
    .steering_time_constant_seconds = STEERING_EFFECTIVE_TIME_CONSTANT_SECONDS,
};

VehicleParameters_t vehicle_model_default_parameters(void)
{
    const VehicleParameters_t defaults = {
        .wheelbase_meters = VP_WHEELBASE_M,
        .distance_cg_to_front_axle = VP_CG_TO_FRONT_AXLE_M,
        .distance_cg_to_rear_axle = VP_CG_TO_REAR_AXLE_M,
        .height_cg_to_ground = VP_CG_HEIGHT_M,
        .vehicle_mass = VP_MASS_KG,
        .yaw_moment_of_inertia = VP_YAW_INERTIA_KGM2,
        .front_cornering_stiffness = VP_FRONT_CORNERING_STIFFNESS,
        .rear_cornering_stiffness = VP_REAR_CORNERING_STIFFNESS,
        .max_steering_angle = VP_MAX_STEERING_RAD,
        .max_velocity = VP_MAX_VELOCITY_MPS,
        .min_velocity = 0.0f,
        .max_acceleration = VP_MAX_ACCEL_MPS2,
        .min_acceleration = VP_MIN_ACCEL_MPS2,
        .friction_coefficient = VP_FRICTION_COEFF,
        .gravity_mps2 = GRAVITY_MPS2,
        .tire_shape_factor = VP_C_SHAPE,
        .minimum_slip_velocity = MIN_SLIP_VELOCITY,
        .minimum_stiffness_scale = MIN_STIFF_SCALE,
        .steering_time_constant_seconds = STEERING_EFFECTIVE_TIME_CONSTANT_SECONDS,
    };
    return defaults;
}

VehicleParameters_t vehicle_model_get_parameters(void)
{
    return active_parameters;
}

int vehicle_model_set_parameters(const VehicleParameters_t *parameters)
{
    if (parameters == NULL ||
        !isfinite(parameters->wheelbase_meters) ||
        !isfinite(parameters->distance_cg_to_front_axle) ||
        !isfinite(parameters->distance_cg_to_rear_axle) ||
        !isfinite(parameters->height_cg_to_ground) ||
        !isfinite(parameters->vehicle_mass) ||
        !isfinite(parameters->yaw_moment_of_inertia) ||
        !isfinite(parameters->front_cornering_stiffness) ||
        !isfinite(parameters->rear_cornering_stiffness) ||
        !isfinite(parameters->max_steering_angle) ||
        !isfinite(parameters->max_velocity) ||
        !isfinite(parameters->min_velocity) ||
        !isfinite(parameters->max_acceleration) ||
        !isfinite(parameters->min_acceleration) ||
        !isfinite(parameters->friction_coefficient) ||
        !isfinite(parameters->gravity_mps2) ||
        !isfinite(parameters->tire_shape_factor) ||
        !isfinite(parameters->minimum_slip_velocity) ||
        !isfinite(parameters->minimum_stiffness_scale) ||
        !isfinite(parameters->steering_time_constant_seconds) ||
        parameters->wheelbase_meters <= 0.0f ||
        parameters->distance_cg_to_front_axle <= 0.0f ||
        parameters->distance_cg_to_rear_axle <= 0.0f ||
        parameters->vehicle_mass <= 0.0f ||
        parameters->yaw_moment_of_inertia <= 0.0f ||
        parameters->front_cornering_stiffness <= 0.0f ||
        parameters->rear_cornering_stiffness <= 0.0f ||
        parameters->max_steering_angle <= 0.0f ||
        parameters->max_velocity <= 0.0f ||
        parameters->min_velocity < 0.0f ||
        parameters->max_acceleration <= 0.0f ||
        parameters->min_acceleration >= 0.0f ||
        parameters->friction_coefficient <= 0.0f ||
        parameters->gravity_mps2 <= 0.0f ||
        parameters->tire_shape_factor <= 0.0f ||
        parameters->minimum_slip_velocity <= 0.0f ||
        parameters->minimum_stiffness_scale <= 0.0f ||
        parameters->minimum_stiffness_scale > 1.0f ||
        parameters->steering_time_constant_seconds <= 0.0f ||
        parameters->wheelbase_meters <
            parameters->distance_cg_to_front_axle +
            parameters->distance_cg_to_rear_axle - 1e-4f ||
        parameters->wheelbase_meters >
            parameters->distance_cg_to_front_axle +
            parameters->distance_cg_to_rear_axle + 1e-4f) {
        return 0;
    }

    active_parameters = *parameters;
    return 1;
}

/*===========================================================================
 * Shared Helper Math
 *===========================================================================*/

void vehicle_model_compute_slip_terms(
    float vx,
    float vy,
    float omega,
    float delta,
    SlipTerms_t *slip_terms)
{
    const VehicleParameters_t *p = &active_parameters;
    /*
     * Bicycle slip-angle definitions:
     *   alpha_f = delta - atan((v_y + l_f*omega)/v_x)
     *   alpha_r =       -atan((v_y - l_r*omega)/v_x)
     * The floor belongs only to this denominator. It must not be reused as a
     * physical state-speed clamp by either rollout or Frenet kinematics.
     */
    if (fabsf(vx) < p->minimum_slip_velocity)
        slip_terms->vx_safe = (vx < 0.0f) ? -p->minimum_slip_velocity : p->minimum_slip_velocity;
    else
        slip_terms->vx_safe = vx;
    slip_terms->inv_vx_safe = util_recip(slip_terms->vx_safe);

    slip_terms->front_num = vy + p->distance_cg_to_front_axle * omega;
    slip_terms->rear_num = vy - p->distance_cg_to_rear_axle * omega;

    slip_terms->front_ratio = slip_terms->front_num * slip_terms->inv_vx_safe;
    slip_terms->rear_ratio = slip_terms->rear_num * slip_terms->inv_vx_safe;

    slip_terms->alpha_front = delta - atanf(slip_terms->front_ratio);
    slip_terms->alpha_rear = -atanf(slip_terms->rear_ratio);
}

void vehicle_model_compute_normal_loads(
    float longitudinal_force,
    float *front_normal_load,
    float *rear_normal_load)
{
    const VehicleParameters_t *p = &active_parameters;
    /*
     *  Quasi-static longitudinal load transfer:
     *  acceleration/braking shifts load by (F_x * h / L) between front/rear axles.
     */
    *front_normal_load = util_div(
        p->vehicle_mass * p->gravity_mps2 * p->distance_cg_to_rear_axle -
            longitudinal_force * p->height_cg_to_ground,
        p->wheelbase_meters);

    *rear_normal_load = util_div(
        p->vehicle_mass * p->gravity_mps2 * p->distance_cg_to_front_axle +
            longitudinal_force * p->height_cg_to_ground,
        p->wheelbase_meters);
}

void vehicle_model_compute_effective_lateral_stiffness(
    uint8_t use_front_axle,
    float normal_load,
    float slip_angle,
    float *effective_stiffness,
    float *lateral_force)
{
    const VehicleParameters_t *p = &active_parameters;
    float cornering_stiffness =
        use_front_axle ? p->front_cornering_stiffness : p->rear_cornering_stiffness;
    float linear_stiffness = p->friction_coefficient * cornering_stiffness * normal_load;
    float peak_force = p->friction_coefficient * normal_load;
    float B_term = cornering_stiffness / p->tire_shape_factor;
    float B_alpha = B_term * slip_angle;
    float inner_angle = p->tire_shape_factor * atanf(B_alpha);
    float cos_inner = cosf(inner_angle);
    float inv_denom = util_recip(1.0f + B_alpha * B_alpha);

    /*
     * Local linearization slope for a Pacejka-like tire law, with a floor to
     * prevent near-saturated operating points from collapsing Jacobian gains.
     */
    float effective_slope = peak_force * p->tire_shape_factor * B_term * cos_inner * inv_denom;
    float minimum_slope = linear_stiffness * p->minimum_stiffness_scale;
    if (effective_stiffness != NULL) {
        *effective_stiffness =
            (effective_slope > minimum_slope) ? effective_slope : minimum_slope;
    }

    if (lateral_force != NULL)
    {
        *lateral_force = peak_force * sinf(inner_angle);
    }
}

void vehicle_model_compute_body_dynamics(
    const VehicleState_t *current_state,
    const ControlInput_t *control_input,
    VehicleBodyDynamics_t *dynamics)
{
    if (current_state == NULL || control_input == NULL || dynamics == NULL)
        return;

    const ControlInput_t saturated_control =
        vehicle_model_saturate_control(control_input);
    const float vx = current_state->long_vel;
    const float vy = current_state->lat_vel;
    const float omega = current_state->yaw_rate;
    const float delta = saturated_control.steer_ang;
    const float longitudinal_acceleration = saturated_control.long_acc;
    const float cos_delta = cosf(delta);
    const float sin_delta = sinf(delta);
    const VehicleParameters_t *p = &active_parameters;
    const float longitudinal_force = p->vehicle_mass * longitudinal_acceleration;

    vehicle_model_compute_slip_terms(
        vx, vy, omega, delta, &dynamics->slip_terms);
    vehicle_model_compute_normal_loads(
        longitudinal_force,
        &dynamics->front_normal_load,
        &dynamics->rear_normal_load);
    vehicle_model_compute_effective_lateral_stiffness(
        1u,
        dynamics->front_normal_load,
        dynamics->slip_terms.alpha_front,
        NULL,
        &dynamics->front_lateral_force);
    vehicle_model_compute_effective_lateral_stiffness(
        0u,
        dynamics->rear_normal_load,
        dynamics->slip_terms.alpha_rear,
        NULL,
        &dynamics->rear_lateral_force);

    dynamics->longitudinal_force = longitudinal_force;
    dynamics->long_acceleration =
        (longitudinal_force - dynamics->front_lateral_force * sin_delta) *
            (1.0f / p->vehicle_mass) + vy * omega;
    dynamics->lateral_acceleration =
        (dynamics->front_lateral_force * cos_delta +
         dynamics->rear_lateral_force) * (1.0f / p->vehicle_mass) - vx * omega;
    dynamics->yaw_acceleration =
        (p->distance_cg_to_front_axle * dynamics->front_lateral_force * cos_delta -
         p->distance_cg_to_rear_axle * dynamics->rear_lateral_force) *
        (1.0f / p->yaw_moment_of_inertia);
}

/*===========================================================================
 * Control Saturation
 *===========================================================================*/

ControlInput_t vehicle_model_saturate_control(
    const ControlInput_t *raw_control)
{
    const VehicleParameters_t *p = &active_parameters;
    ControlInput_t saturated_control;

    /* Clamp steering angle to physical limits */
    saturated_control.steer_ang = util_clamp(
        raw_control->steer_ang,
        -p->max_steering_angle,
        p->max_steering_angle);

    /* Clamp acceleration to [min, max] */
    saturated_control.long_acc = util_clamp(
        raw_control->long_acc,
        p->min_acceleration,
        p->max_acceleration);

    return saturated_control;
}

/*===========================================================================
 * Single-Step State Prediction
 *===========================================================================*/

VehicleState_t vehicle_model_predict_next_state(
    const VehicleState_t *current_state,
    const ControlInput_t *control_input,
    float time_step)
{
    const VehicleParameters_t *p = &active_parameters;
    VehicleState_t next_state;

    // Saturate control inputs to ensure predictions respect physical limits.
    ControlInput_t saturated_control = vehicle_model_saturate_control(control_input);

    // Extract current state variables for readability
    float psi = current_state->heading;
    float vx  = current_state->long_vel;
    float vy  = current_state->lat_vel;
    float omega = current_state->yaw_rate;

    VehicleBodyDynamics_t dynamics = {0};
    vehicle_model_compute_body_dynamics(
        current_state, &saturated_control, &dynamics);

    /*
     * Analytical pose integration on SE(2) with constant body twist over dt.
     * Falls back to straight-line integration when yaw rate is near zero.
     */
    {
        float psi_end = psi + time_step * omega;

        if (fabsf(omega) < 1e-6f)
        {
            float cos_psi = cosf(psi);
            float sin_psi = sinf(psi);
            next_state.pos_x = current_state->pos_x + time_step * (vx * cos_psi - vy * sin_psi);
            next_state.pos_y = current_state->pos_y + time_step * (vx * sin_psi + vy * cos_psi);
        }
        else
        {
            float inv_omega = util_recip(omega);
            float sin_psi_start = sinf(psi);
            float cos_psi_start = cosf(psi);
            float sin_psi_end = sinf(psi_end);
            float cos_psi_end = cosf(psi_end);

            next_state.pos_x = current_state->pos_x +
                (vx * (sin_psi_end - sin_psi_start) + vy * (cos_psi_end - cos_psi_start)) * inv_omega;
            next_state.pos_y = current_state->pos_y +
                (vx * (cos_psi_start - cos_psi_end) + vy * (sin_psi_end - sin_psi_start)) * inv_omega;
        }

        next_state.heading = psi_end;
    }

    // Forward Euler integration for body-dynamic states
    next_state.long_vel = vx + time_step * dynamics.long_acceleration;
    next_state.lat_vel = vy + time_step * dynamics.lateral_acceleration;
    next_state.yaw_rate = omega + time_step * dynamics.yaw_acceleration;

    // Apply state constraints and normalization

    /* Keep the physical forward-speed domain [0, max].  The slip-angle
     * denominator floor above is numerical regularization only; it must not
     * create artificial forward motion when the vehicle is stopped. */
    next_state.long_vel = util_clamp(
        next_state.long_vel,
        0.0f,
        p->max_velocity);

    /* Normalize heading to principal angle domain. */
    next_state.heading = util_normalize_angle(next_state.heading);

    return next_state;
}

FrenetState_t vehicle_model_predict_next_frenet_state(
    const FrenetState_t *state,
    const ControlInput_t *control,
    float time_step,
    float path_curvature)
{
    const VehicleParameters_t *p = &active_parameters;
    FrenetState_t next = *state;
    VehicleState_t body_state = {0.0f, 0.0f, 0.0f,
                                 state->flong_vel, state->flat_vel,
                                 state->fyaw_rate};
    VehicleBodyDynamics_t dynamics = {0};
    vehicle_model_compute_body_dynamics(&body_state, control, &dynamics);

    float ey_denom = 1.0f - path_curvature * state->flat_error;
    if (fabsf(ey_denom) < 1e-3f)
        ey_denom = (ey_denom >= 0.0f) ? 1e-3f : -1e-3f;

    /* These are geometric identities in the physical current velocity.
     * A reference-speed floor would create motion at standstill and make the
     * plant model disagree with the measured state. */
    const float vx = state->flong_vel;
    next.flat_error = state->flat_error + time_step * (
        vx * sinf(state->fhead_error) +
        state->flat_vel * cosf(state->fhead_error));
    next.fhead_error = util_normalize_angle(state->fhead_error + time_step * (
        state->fyaw_rate - path_curvature * vx *
        cosf(state->fhead_error) / ey_denom));
    next.flong_vel = util_clamp(
        vx + time_step * dynamics.long_acceleration,
        0.0f,
        p->max_velocity);
    next.flat_vel = state->flat_vel + time_step * dynamics.lateral_acceleration;
    next.fyaw_rate = state->fyaw_rate + time_step * dynamics.yaw_acceleration;
    return next;
}

/*===========================================================================
 * Multi-Step Trajectory Prediction
 *===========================================================================*/

void vehicle_model_predict_trajectory(
    const VehicleState_t *initial_state,
    const ControlInput_t *control_sequence,
    float time_step,
    uint16_t step_count,
    VehicleState_t *predicted_trajectory)
{
    /* First element of trajectory is the initial state */
    predicted_trajectory[0] = *initial_state;

    /* Predict each subsequent state */
    for (uint16_t step_index = 0; step_index < step_count; step_index++)
    {
        predicted_trajectory[step_index + 1] = vehicle_model_predict_next_state(
            &predicted_trajectory[step_index],
            &control_sequence[step_index],
            time_step);
    }
}

/*===========================================================================
 * Frenet Frame Linearization (Direct Computation)
 *===========================================================================
 *
 * Computes the 5×5 discrete state-space matrices for the Frenet frame
 * directly, without building the full 6×6 global matrices first.
 *
 * Frenet state: [e_y, e_psi, v_x, v_y, omega]
 *
 * Rows 0-1: Frenet kinematic relations (path-relative)
 *   e_y_dot   ≈ v_x * e_psi + v_y           (linearized at e_psi=0)
 *   e_psi_dot ≈ omega - kappa * v_x          (linearized at e_y=0, e_psi=0)
 *
 * Rows 2-4: Body-frame dynamics (identical to global model rows 3-5)
 *   v_x_dot, v_y_dot, omega_dot
 *   Computed inline — avoids the 6×6 global A/B allocation and the
 *   wasted work on global rows 0-2 (X, Y, heading kinematics).
 */
void vehicle_model_compute_frenet_linearization(
    const FrenetState_t *frenet_state,
    const ControlInput_t *operating_control,
    float time_step,
    float path_curvature,
    float reference_velocity,
    float state_matrix_A[NX_FRENET][NX_FRENET],
    float input_matrix_B[NX_FRENET][2])
{
    const VehicleParameters_t *p = &active_parameters;

    // Extract Frenet operating point variables
    float vx    = frenet_state->flong_vel;
    float vy    = frenet_state->flat_vel;
    float omega = frenet_state->fyaw_rate;
    float delta = operating_control->steer_ang;

    /* Steering trig values at operating point (accuracy-first path). */
    float cos_delta = cosf(delta);
    float sin_delta = sinf(delta);

    /* Shared slip-angle kinematics used by prediction and linearization. */
    SlipTerms_t slip_terms;
    vehicle_model_compute_slip_terms(vx, vy, omega, delta, &slip_terms);

    /*
     * Longitudinal force (for load transfer only).
     * Fx = m * a_cmd — independent of state, so dFx/dvx = 0.
     */
    float Fx = p->vehicle_mass * operating_control->long_acc;
    const float dFx_dvx = 0.0f;

    // Compute slip angle intermediates for Jacobian
    float front_num = slip_terms.front_num;
    float rear_num = slip_terms.rear_num;

    /*
     * Analytic slip-angle derivatives from atan(n/d):
     *   d(atan(n/d))/dx = (d*dn/dx - n*dd/dx) / (d^2 + n^2)
     * Using closed-form derivatives keeps the Jacobian stable and low-noise.
     */
    float front_num2 = front_num * front_num;
    float rear_num2 = rear_num * rear_num;
    float vx2 = slip_terms.vx_safe * slip_terms.vx_safe;

    float D_f = vx2 + front_num2;
    float D_r = vx2 + rear_num2;
    if (D_f == 0) D_f = 1.0f;
    if (D_r == 0) D_r = 1.0f;

    float inv_D_f = util_recip(D_f);
    float inv_D_r = util_recip(D_r);

    float daf_dvx = front_num * inv_D_f;
    float daf_dvy = -slip_terms.vx_safe * inv_D_f;
    float daf_domega = -(p->distance_cg_to_front_axle * slip_terms.vx_safe) * inv_D_f;

    float dar_dvx = rear_num * inv_D_r;
    float dar_dvy = -slip_terms.vx_safe * inv_D_r;
    float dar_domega = p->distance_cg_to_rear_axle * slip_terms.vx_safe * inv_D_r;

    /* Below the slip-velocity floor, alpha depends on vx_safe (constant),
     * so d(alpha)/d(vx) must be zero. */
    if (fabsf(vx) < p->minimum_slip_velocity) {
        daf_dvx = 0.0f;
        dar_dvx = 0.0f;
    }

    /*
     * Normal forces with load transfer
     *   F_zf = (m*g*l_r - F_x*h) / L
     *   F_zr = (m*g*l_f + F_x*h) / L
     */
    float F_zf;
    float F_zr;
    vehicle_model_compute_normal_loads(Fx, &F_zf, &F_zr);

    /*
     * Pacejka-like local linearization at the operating point:
     * Keep nonlinear saturation behavior, then convert to effective
     * cornering stiffness dF_y/dalpha for Jacobian entries.
     */
    float C_Sf_Fzf;
    float C_Sr_Fzr;
    float F_yf;
    vehicle_model_compute_effective_lateral_stiffness(
        1u,
        F_zf,
        slip_terms.alpha_front,
        &C_Sf_Fzf,
        &F_yf);
    vehicle_model_compute_effective_lateral_stiffness(
        0u,
        F_zr,
        slip_terms.alpha_rear,
        &C_Sr_Fzr,
        NULL);

    /*
     * Tire force Jacobians w.r.t. body states
     */
    float dFyf_dvx = C_Sf_Fzf * daf_dvx;
    float dFyf_dvy = C_Sf_Fzf * daf_dvy;
    float dFyf_domega = C_Sf_Fzf * daf_domega;
    float dFyf_ddelta = C_Sf_Fzf;

    float dFyr_dvx = C_Sr_Fzr * dar_dvx;
    float dFyr_dvy = C_Sr_Fzr * dar_dvy;
    float dFyr_domega = C_Sr_Fzr * dar_domega;

    /* Initialize Frenet matrices to zero. */
    memset(state_matrix_A, 0, sizeof(float) * NX_FRENET * NX_FRENET);
    memset(input_matrix_B, 0, sizeof(float) * NX_FRENET * 2);

    /*
     * Frenet error-kinematics Jacobian (rows 0,1).
     *
     * These rows are geometric identities in the ACTUAL longitudinal speed,
     * so the linearization must use vx, not the reference speed — substituting
     * reference_velocity here biases the Riccati lateral gain on every
     * accel/decel transient. Must stay consistent with the nonlinear rollout
     * in mpc_predict_frenet_next_state(), which uses the same v_eff.
     *
     * At physical standstill, the path-relative translational derivatives are
     * zero. Only the slip-angle denominator is regularized; no artificial
     * forward speed is introduced into the state transition.
     *
     * Row 0: e_y_dot   = v_eff*sin(e_psi) + v_y*cos(e_psi)
     * Row 1: e_psi_dot = omega - kappa*v_eff*cos(e_psi)/(1 - kappa*e_y)
     */
    (void)reference_velocity;
    {
        const float v_eff = vx;
        float epsi = frenet_state->fhead_error;
        float ey = frenet_state->flat_error;
        float cp = cosf(epsi);
        float sp = sinf(epsi);
        float denom = 1.0f - path_curvature * ey;
        if (fabsf(denom) < 1e-3f)
            denom = (denom >= 0.0f) ? 1e-3f : -1e-3f;
        float inv_denom = util_recip(denom);
        float inv_denom2 = inv_denom * inv_denom;

        state_matrix_A[0][0] = 1.0f;
        state_matrix_A[0][1] = time_step * (v_eff * cp - vy * sp);
        state_matrix_A[0][2] = time_step * sp;
        state_matrix_A[0][3] = time_step * cp;

        state_matrix_A[1][0] = -time_step * (path_curvature * path_curvature) * v_eff * cp * inv_denom2;
        state_matrix_A[1][1] = 1.0f + time_step * path_curvature * v_eff * sp * inv_denom;
        state_matrix_A[1][2] = -time_step * path_curvature * cp * inv_denom;
        state_matrix_A[1][4] = time_step;
    }

    /* Row 1 already assigned in the exact Frenet Jacobian block above. */

    /*
     * Rows 2-4: Body-frame dynamics (computed directly — avoids the
     * 6×6 global matrix allocation and discarded rows 0-2).
     *
     * These are the Jacobians of [v_x_dot, v_y_dot, omega_dot] w.r.t.
     * body states [v_x, v_y, omega], discretized via Forward Euler.
     */

    /* Full model with cos(δ)/sin(δ) force resolution */

    /* Row 2: dvx/dt = (Fx - Fyf*sin(δ) + m*vy*ω) / m */
    const float inv_mass = 1.0f / p->vehicle_mass;
    const float inv_yaw_inertia = 1.0f / p->yaw_moment_of_inertia;
    state_matrix_A[2][2] = 1.0f + time_step * ((dFx_dvx - dFyf_dvx * sin_delta) * inv_mass);
    state_matrix_A[2][3] = time_step * (-dFyf_dvy * sin_delta * inv_mass + omega);
    state_matrix_A[2][4] = time_step * (-dFyf_domega * sin_delta * inv_mass + vy);

    /* Row 3: dvy/dt = (Fyf*cos(δ) + Fyr - m*vx*ω) / m */
    state_matrix_A[3][2] = time_step * ((dFyf_dvx * cos_delta + dFyr_dvx - p->vehicle_mass * omega) * inv_mass);
    state_matrix_A[3][3] = 1.0f + time_step * ((dFyf_dvy * cos_delta + dFyr_dvy) * inv_mass);
    state_matrix_A[3][4] = time_step * ((dFyf_domega * cos_delta + dFyr_domega - p->vehicle_mass * vx) * inv_mass);

    /* Row 4: dω/dt = (lf*Fyf*cos(δ) - lr*Fyr) / Iz */
    state_matrix_A[4][2] = time_step * ((p->distance_cg_to_front_axle * dFyf_dvx * cos_delta - p->distance_cg_to_rear_axle * dFyr_dvx) * inv_yaw_inertia);
    state_matrix_A[4][3] = time_step * ((p->distance_cg_to_front_axle * dFyf_dvy * cos_delta - p->distance_cg_to_rear_axle * dFyr_dvy) * inv_yaw_inertia);
    state_matrix_A[4][4] = 1.0f + time_step * ((p->distance_cg_to_front_axle * dFyf_domega * cos_delta - p->distance_cg_to_rear_axle * dFyr_domega) * inv_yaw_inertia);

    /* B matrix — steering column with cos(δ)/sin(δ) force resolution */
    float dFyf_dd_sin = dFyf_ddelta * sin_delta;
    float Fyf_cos = F_yf * cos_delta;
    float dFyf_dd_cos = dFyf_ddelta * cos_delta;
    float Fyf_sin = F_yf * sin_delta;

    /* B[2][0]: d(dvx/dt)/dδ = (-dFyf_dd*sin(δ) - Fyf*cos(δ)) / m */
    input_matrix_B[2][0] = time_step * ((-dFyf_dd_sin - Fyf_cos) * inv_mass);

    /* B[3][0]: d(dvy/dt)/dδ = (dFyf_dd*cos(δ) - Fyf*sin(δ)) / m */
    input_matrix_B[3][0] = time_step * ((dFyf_dd_cos - Fyf_sin) * inv_mass);

    /* B[4][0]: d(dω/dt)/dδ = lf*(dFyf_dd*cos(δ) - Fyf*sin(δ)) / Iz */
    input_matrix_B[4][0] = time_step * (p->distance_cg_to_front_axle * (dFyf_dd_cos - Fyf_sin) * inv_yaw_inertia);

    /* B[2][1]: acceleration → vx directly (dt * 1) */
    input_matrix_B[2][1] = time_step;

    /* B[3][1], B[4][1]: acceleration affects lateral dynamics via load transfer. */
    {
        float inv_Fzf = util_recip(F_zf);
        float inv_Fzr = util_recip(F_zr);
        float C_Sf_norm = C_Sf_Fzf * inv_Fzf;
        float C_Sr_norm = C_Sr_Fzr * inv_Fzr;
        float dFzf_da = -(p->vehicle_mass * p->height_cg_to_ground) / p->wheelbase_meters;
        float dFzr_da = -dFzf_da;
        float dFyf_da = C_Sf_norm * slip_terms.alpha_front * dFzf_da;
        float dFyr_da = C_Sr_norm * slip_terms.alpha_rear * dFzr_da;

        input_matrix_B[3][1] = time_step * ((dFyf_da * cos_delta + dFyr_da) * inv_mass);
        input_matrix_B[4][1] = time_step * ((p->distance_cg_to_front_axle * dFyf_da * cos_delta
                                            - p->distance_cg_to_rear_axle * dFyr_da)
                                           * inv_yaw_inertia);
    }

    /*
     * B rows 0-1 are all zero:
     * Steering and torque don't directly change e_y or e_psi.
     * Their effect propagates through omega (row 4) and v_y (row 3),
     * which then affect e_y and e_psi through the A matrix coupling.
     */
    /* input_matrix_B[0][0..1] = 0 (already zeroed) */
    /* input_matrix_B[1][0..1] = 0 (already zeroed) */
}
