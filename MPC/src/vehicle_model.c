/**
 * @file vehicle_model.c
 * @brief Dynamic nonlinear bicycle model implementation.
 * @dependencies vehicle_model.h, <stdio.h>, <string.h>, <stdlib.h>
 */

#include "vehicle_model.h"

static const float kAccelerationEnvelopeSpeedMps[] = {
    0.0f, 2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f,
    14.0f, 16.0f, 18.0f, 20.0f, 22.0f, 23.0f};
static const float kAccelerationEnvelopeMps2[] = {
    5.5f, 4.4f, 4.4f, 3.568f, 3.175f, 2.562f, 2.043f,
    1.565f, 0.956f, 0.529f, 0.529f, 0.529f, 0.086f};
static const size_t kAccelerationEnvelopeCount =
    sizeof(kAccelerationEnvelopeSpeedMps) /
    sizeof(kAccelerationEnvelopeSpeedMps[0]);

float vehicle_model_max_forward_acceleration(float speed_mps)
{
    float speed = isfinite(speed_mps) ? speed_mps : 0.0f;
    if (speed <= kAccelerationEnvelopeSpeedMps[0])
        return kAccelerationEnvelopeMps2[0];
    if (speed >= kAccelerationEnvelopeSpeedMps[kAccelerationEnvelopeCount - 1U])
        return kAccelerationEnvelopeMps2[kAccelerationEnvelopeCount - 1U];

    for (size_t index = 1U; index < kAccelerationEnvelopeCount; ++index) {
        if (speed <= kAccelerationEnvelopeSpeedMps[index]) {
            const float lower_speed = kAccelerationEnvelopeSpeedMps[index - 1U];
            const float upper_speed = kAccelerationEnvelopeSpeedMps[index];
            const float ratio = (speed - lower_speed) /
                (upper_speed - lower_speed);
            return kAccelerationEnvelopeMps2[index - 1U] + ratio *
                (kAccelerationEnvelopeMps2[index] -
                 kAccelerationEnvelopeMps2[index - 1U]);
        }
    }
    return kAccelerationEnvelopeMps2[kAccelerationEnvelopeCount - 1U];
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
    /*
     * Bicycle slip-angle definitions:
     *   alpha_f = delta - atan((v_y + l_f*omega)/v_x)
     *   alpha_r =       -atan((v_y - l_r*omega)/v_x)
     * A minimum longitudinal velocity floor keeps the ratio well-conditioned.
     */
    slip_terms->vx_safe = (vx > MIN_SLIP_VELOCITY) ? vx : MIN_SLIP_VELOCITY;
    slip_terms->inv_vx_safe = util_recip(slip_terms->vx_safe);

    slip_terms->front_num = vy + VP_CG_TO_FRONT_AXLE_M * omega;
    slip_terms->rear_num = vy - VP_CG_TO_REAR_AXLE_M * omega;

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
    /*
     *  Quasi-static longitudinal load transfer:
     *  acceleration/braking shifts load by (F_x * h / L) between front/rear axles.
     */
    *front_normal_load = util_div(
        VP_MASS_TIMES_GRAVITY_N * VP_CG_TO_REAR_AXLE_M - longitudinal_force * VP_CG_HEIGHT_M,
        VP_WHEELBASE_M);

    *rear_normal_load = util_div(
        VP_MASS_TIMES_GRAVITY_N * VP_CG_TO_FRONT_AXLE_M + longitudinal_force * VP_CG_HEIGHT_M,
        VP_WHEELBASE_M);
}

void vehicle_model_compute_effective_lateral_stiffness(
    uint8_t use_front_axle,
    float normal_load,
    float slip_angle,
    float *effective_stiffness,
    float *lateral_force)
{
    const float load = fmaxf(0.0f, normal_load);
    const float initial_physical_slope = use_front_axle ?
        VP_C_ALPHA_F : VP_C_ALPHA_R;
    const float initial_normalized_slope = util_div(
        initial_physical_slope, fmaxf(load, 1.0e-6f));
    const float signed_slip = tanf(slip_angle);
    const float slip = fabsf(signed_slip);
    float normalized_force;
    float force_slope_per_slip;

    /*
     * AutoDRIVE documents a two-piece cubic spline in normalized lateral
     * slip S_y = tan(alpha):
     *
     *   (0, 0) -> (0.01, 1.00) -> (0.10, 0.50)
     *
     * The exact coefficients are not published.  Hermite segments provide
     * the documented knots, a zero slope at the extremum/asymptote, and the
     * measured small-slip slope at S_y=0.  Prediction and linearization use
     * this same evaluator, so the optimizer does not solve a different tire
     * model than the rollout.
     */
    if (slip < VP_LATERAL_EXTREMUM_SLIP)
    {
        const float t = util_div(slip, VP_LATERAL_EXTREMUM_SLIP);
        const float h = VP_LATERAL_EXTREMUM_SLIP;
        const float y0 = 0.0f;
        const float y1 = VP_LATERAL_EXTREMUM_VALUE;
        const float m0 = initial_normalized_slope;
        const float m1 = 0.0f;
        const float t2 = t * t;
        const float t3 = t2 * t;
        const float h00 = 2.0f * t3 - 3.0f * t2 + 1.0f;
        const float h10 = t3 - 2.0f * t2 + t;
        const float h01 = -2.0f * t3 + 3.0f * t2;
        const float h11 = t3 - t2;
        normalized_force = h00 * y0 + h10 * h * m0 + h01 * y1 + h11 * m1;
        force_slope_per_slip = util_div(
            (6.0f * t2 - 6.0f * t) * y0 +
            (3.0f * t2 - 4.0f * t + 1.0f) * h * m0 +
            (-6.0f * t2 + 6.0f * t) * y1 +
            (3.0f * t2 - 2.0f * t) * h * m1,
            h);
    }
    else if (slip < VP_LATERAL_ASYMPTOTE_SLIP)
    {
        const float h = VP_LATERAL_ASYMPTOTE_SLIP -
            VP_LATERAL_EXTREMUM_SLIP;
        const float t = util_div(
            slip - VP_LATERAL_EXTREMUM_SLIP, h);
        const float t2 = t * t;
        const float t3 = t2 * t;
        const float y0 = VP_LATERAL_EXTREMUM_VALUE;
        const float y1 = VP_LATERAL_ASYMPTOTE_VALUE;
        const float h00 = 2.0f * t3 - 3.0f * t2 + 1.0f;
        const float h01 = -2.0f * t3 + 3.0f * t2;
        const float h10 = t3 - 2.0f * t2 + t;
        const float h11 = t3 - t2;
        normalized_force = h00 * y0 + h01 * y1;
        force_slope_per_slip = util_div(
            (-6.0f * t2 + 6.0f * t) * y0 +
            (6.0f * t2 - 6.0f * t) * y1,
            h);
        (void)h10;
        (void)h11;
    }
    else
    {
        normalized_force = VP_LATERAL_ASYMPTOTE_VALUE;
        force_slope_per_slip = 0.0f;
    }

    /* S_y = tan(alpha), so dS_y/dalpha = sec^2(alpha). */
    const float slip_angle_slope = 1.0f + signed_slip * signed_slip;
    const float effective_slope = load * force_slope_per_slip *
        slip_angle_slope;
    const float minimum_slope = load * initial_normalized_slope * MIN_STIFF_SCALE;
    *effective_stiffness = fmaxf(effective_slope, minimum_slope);

    if (lateral_force != NULL)
    {
        *lateral_force = load * copysignf(normalized_force, signed_slip);
    }
}

/*===========================================================================
 * Control Saturation
 *===========================================================================*/

ControlInput_t vehicle_model_saturate_control(
    const ControlInput_t *raw_control)
{
    ControlInput_t saturated_control;

    /* Clamp steering angle to physical limits */
    saturated_control.steer_ang = util_clamp(
        raw_control->steer_ang,
        -VP_MAX_STEERING_RAD,
        VP_MAX_STEERING_RAD);

    /* Clamp acceleration to [min, max] */
    saturated_control.long_acc = util_clamp(
        raw_control->long_acc,
        VP_MIN_ACCEL_MPS2,
        VP_MAX_ACCEL_MPS2);

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
    VehicleState_t next_state;

    // Saturate control inputs to ensure predictions respect physical limits.
    ControlInput_t saturated_control = vehicle_model_saturate_control(control_input);
    /* The direct acceleration command is only physically available up to the
     * measured full-throttle envelope at the current speed. Keep the
     * nonlinear prediction consistent with the per-stage MPC bounds. */
    saturated_control.long_acc = util_clamp(
        saturated_control.long_acc,
        VP_MIN_ACCEL_MPS2,
        vehicle_model_max_forward_acceleration(current_state->long_vel));

    // Extract current state variables for readability
    float psi = current_state->heading;
    float vx  = current_state->long_vel;
    float vy  = current_state->lat_vel;
    float omega = current_state->yaw_rate;

    // Extract control inputs
    float delta   = saturated_control.steer_ang;
    float a_cmd  = saturated_control.long_acc;

    // Precompute steering trigonometric terms for force resolution
    float cos_delta = cosf(delta);
    float sin_delta = sinf(delta);

    /*
     * Compute longitudinal force from acceleration command
     *
     * F_x = m * a_cmd
     *
     * The MPC outputs an acceleration command directly.
     * The longitudinal force is simply mass × acceleration.
     */
    float Fx = VP_MASS_KG * a_cmd;

    /*
     * Compute tire slip angles
     *
     * Minimum velocity floor to prevent division by zero at standstill.
     * Below this speed, the dynamic model degenerates — slip angles are
     * undefined when v_x ≈ 0.
     */
    SlipTerms_t slip_terms;
    vehicle_model_compute_slip_terms(vx, vy, omega, delta, &slip_terms);

    /*
     * Compute normal forces (load transfer under acceleration)
     *
     * Static weight distribution plus longitudinal load transfer:
     *   F_zf = (m * g * l_r - F_x * h) / (l_f + l_r)
     *   F_zr = (m * g * l_f + F_x * h) / (l_f + l_r)
     */
    float F_zf;
    float F_zr;
    vehicle_model_compute_normal_loads(Fx, &F_zf, &F_zr);

    /* Use the same asymptotic tire law as the linearization. */
    float unused_stiffness;
    float F_yf;
    float F_yr;
    vehicle_model_compute_effective_lateral_stiffness(
        1u, F_zf, slip_terms.alpha_front, &unused_stiffness, &F_yf);
    vehicle_model_compute_effective_lateral_stiffness(
        0u, F_zr, slip_terms.alpha_rear, &unused_stiffness, &F_yr);

    /* Full model: cos(δ)/sin(δ) force resolution for real-world accuracy */
    /* dv_x/dt = (F_x - F_yf * sin(delta) + m * v_y * omega) / m */
    float dvx_dt = (Fx - F_yf * sin_delta) * VP_INV_MASS_1_PER_KG + vy * omega;

    /* dv_y/dt = (F_yf * cos(delta) + F_yr - m * v_x * omega) / m */
    float dvy_dt = (F_yf * cos_delta + F_yr) * VP_INV_MASS_1_PER_KG - vx * omega;

    /* domega/dt = (l_f * F_yf * cos(delta) - l_r * F_yr) / I_z */
    float domega_dt =
        (VP_CG_TO_FRONT_AXLE_M * F_yf * cos_delta - VP_CG_TO_REAR_AXLE_M * F_yr) *
        VP_INV_YAW_INERTIA_1_PER_KGM2;

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
    next_state.long_vel = vx + time_step * dvx_dt;
    next_state.lat_vel = vy + time_step * dvy_dt;
    next_state.yaw_rate = omega + time_step * domega_dt;

    // Apply state constraints and normalization

    /* Clamp longitudinal velocity to [min, max] */
    next_state.long_vel = util_clamp(
        next_state.long_vel,
        VP_MIN_VELOCITY_MPS,
        VP_MAX_VELOCITY_MPS);

    /* Normalize heading to principal angle domain. */
    next_state.heading = util_normalize_angle(next_state.heading);

    return next_state;
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
    float Fx = VP_MASS_KG * operating_control->long_acc;
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
    float daf_domega = -(VP_CG_TO_FRONT_AXLE_M * slip_terms.vx_safe) * inv_D_f;

    float dar_dvx = rear_num * inv_D_r;
    float dar_dvy = -slip_terms.vx_safe * inv_D_r;
    float dar_domega = VP_CG_TO_REAR_AXLE_M * slip_terms.vx_safe * inv_D_r;

    /* Below the slip-velocity floor, alpha depends on vx_safe (constant),
     * so d(alpha)/d(vx) must be zero. */
    if (vx <= MIN_SLIP_VELOCITY) {
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
     * Asymptotic simulator-tire local linearization at the operating point:
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
     * Loss-of-rank safeguard: as vx -> 0 the (e_y, e_psi) subsystem becomes
     * uncontrollable and A/B degenerate. Flooring at MIN_LINEARIZATION_VELOCITY
     * keeps it controllable at standstill (the regularization the old
     * reference_velocity substitution was really providing). Floor value
     * matches the FPGA kernel (MIN_LIN_VEL = 0.5 m/s) for CPU/FPGA parity.
     *
     * Row 0: e_y_dot   = v_eff*sin(e_psi) + v_y*cos(e_psi)
     * Row 1: e_psi_dot = omega - kappa*v_eff*cos(e_psi)/(1 - kappa*e_y)
     */
    (void)reference_velocity;
    {
        const float v_eff = (vx > MIN_LINEARIZATION_VELOCITY)
                                ? vx : MIN_LINEARIZATION_VELOCITY;
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
    state_matrix_A[2][2] = 1.0f + time_step * ((dFx_dvx - dFyf_dvx * sin_delta) * VP_INV_MASS_1_PER_KG);
    state_matrix_A[2][3] = time_step * (-dFyf_dvy * sin_delta * VP_INV_MASS_1_PER_KG + omega);
    state_matrix_A[2][4] = time_step * (-dFyf_domega * sin_delta * VP_INV_MASS_1_PER_KG + vy);

    /* Row 3: dvy/dt = (Fyf*cos(δ) + Fyr - m*vx*ω) / m */
    state_matrix_A[3][2] = time_step * ((dFyf_dvx * cos_delta + dFyr_dvx - VP_MASS_KG * omega) * VP_INV_MASS_1_PER_KG);
    state_matrix_A[3][3] = 1.0f + time_step * ((dFyf_dvy * cos_delta + dFyr_dvy) * VP_INV_MASS_1_PER_KG);
    state_matrix_A[3][4] = time_step * ((dFyf_domega * cos_delta + dFyr_domega - VP_MASS_KG * vx) * VP_INV_MASS_1_PER_KG);

    /* Row 4: dω/dt = (lf*Fyf*cos(δ) - lr*Fyr) / Iz */
    state_matrix_A[4][2] = time_step * ((VP_CG_TO_FRONT_AXLE_M * dFyf_dvx * cos_delta - VP_CG_TO_REAR_AXLE_M * dFyr_dvx) * VP_INV_YAW_INERTIA_1_PER_KGM2);
    state_matrix_A[4][3] = time_step * ((VP_CG_TO_FRONT_AXLE_M * dFyf_dvy * cos_delta - VP_CG_TO_REAR_AXLE_M * dFyr_dvy) * VP_INV_YAW_INERTIA_1_PER_KGM2);
    state_matrix_A[4][4] = 1.0f + time_step * ((VP_CG_TO_FRONT_AXLE_M * dFyf_domega * cos_delta - VP_CG_TO_REAR_AXLE_M * dFyr_domega) * VP_INV_YAW_INERTIA_1_PER_KGM2);

    /* B matrix — steering column with cos(δ)/sin(δ) force resolution */
    float dFyf_dd_sin = dFyf_ddelta * sin_delta;
    float Fyf_cos = F_yf * cos_delta;
    float dFyf_dd_cos = dFyf_ddelta * cos_delta;
    float Fyf_sin = F_yf * sin_delta;

    /* B[2][0]: d(dvx/dt)/dδ = (-dFyf_dd*sin(δ) - Fyf*cos(δ)) / m */
    input_matrix_B[2][0] = time_step * ((-dFyf_dd_sin - Fyf_cos) * VP_INV_MASS_1_PER_KG);

    /* B[3][0]: d(dvy/dt)/dδ = (dFyf_dd*cos(δ) - Fyf*sin(δ)) / m */
    input_matrix_B[3][0] = time_step * ((dFyf_dd_cos - Fyf_sin) * VP_INV_MASS_1_PER_KG);

    /* B[4][0]: d(dω/dt)/dδ = lf*(dFyf_dd*cos(δ) - Fyf*sin(δ)) / Iz */
    input_matrix_B[4][0] = time_step * (VP_CG_TO_FRONT_AXLE_M * (dFyf_dd_cos - Fyf_sin) * VP_INV_YAW_INERTIA_1_PER_KGM2);

    /* B[2][1]: acceleration → vx directly (dt * 1) */
    input_matrix_B[2][1] = time_step;

    /* B[3][1], B[4][1]: acceleration affects lateral dynamics via load transfer. */
    {
        float inv_Fzf = util_recip(F_zf);
        float inv_Fzr = util_recip(F_zr);
        float C_Sf_norm = C_Sf_Fzf * inv_Fzf;
        float C_Sr_norm = C_Sr_Fzr * inv_Fzr;
        float dFzf_da = -(VP_MASS_KG * VP_CG_HEIGHT_M) / VP_WHEELBASE_M;
        float dFzr_da = -dFzf_da;
        float dFyf_da = C_Sf_norm * slip_terms.alpha_front * dFzf_da;
        float dFyr_da = C_Sr_norm * slip_terms.alpha_rear * dFzr_da;

        input_matrix_B[3][1] = time_step * ((dFyf_da * cos_delta + dFyr_da) * VP_INV_MASS_1_PER_KG);
        input_matrix_B[4][1] = time_step * ((VP_CG_TO_FRONT_AXLE_M * dFyf_da * cos_delta
                                            - VP_CG_TO_REAR_AXLE_M * dFyr_da)
                                           * VP_INV_YAW_INERTIA_1_PER_KGM2);
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
