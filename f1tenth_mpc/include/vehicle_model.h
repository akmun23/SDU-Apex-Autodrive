/**
 * @file vehicle_model.h
 * @brief Source-command prediction interface for the AutoDRIVE Unity object.
 *
 * This API deliberately has no Pacejka, friction-coefficient, cornering
 * stiffness, load-transfer, or direct-acceleration vehicle contract.  The
 * nominal lateral relation is the current Unity controller's Ackermann
 * geometry.  WheelCollider/contact effects are handled by an observable
 * response map before promotion; until then this is a bounded, conservative
 * controller-stage baseline rather than a claim of full Unity parity.
 */

#ifndef VEHICLE_MODEL_H
#define VEHICLE_MODEL_H

#include "mpc_types.h"

typedef struct
{
    float e_y;
    float e_psi;
    float u;
    float v;
    float r;
    float target_speed;
    float steering_command;
    /* Unity's bridge delivers the steering target through a two-sample
     * command queue before the physical steering angle follows it. These
     * states are reconstructed from our own past published commands. */
    float delayed_steering_command_1;
    float delayed_steering_command_2;
    /* Unity keeps a separate physical wheel angle and slews it toward the
     * delayed commanded target. This state is estimated causally by the
     * controller; it is not a simulator-only runtime input. */
    float actual_steering_angle;
} MpcModelState_t;

typedef struct
{
    float steering_rate;
    float target_speed_rate;
} MpcModelControl_t;

/*
 * The Unity wheel/contact response is represented here as an identified
 * command-to-yaw map.  These are not real-car tire parameters.  The optional
 * curvature term is an empirical correction for the observed reduction in
 * yaw response during the simulator's high-curvature transitions.
 */
typedef struct
{
    float response_time_constant_s;
    float steering_gain_per_m;
    float curvature_gain_reduction_per_m;
    float curvature_gain_start_per_m;
    float curvature_gain_end_per_m;
    /* Optional empirical low-speed response extension.  A value of zero
     * leaves the identified constant-time-constant model unchanged. */
    float low_speed_response_time_constant_s;
    float low_speed_transition_speed_mps;
} MpcYawRateModelParameters_t;

enum
{
    MPC_STAGE_CLIPPED_STEERING_RATE = 1u << 0,
    MPC_STAGE_CLIPPED_SPEED_RATE = 1u << 1,
    MPC_STAGE_CLIPPED_STEERING_COMMAND = 1u << 2,
    MPC_STAGE_CLIPPED_TARGET_SPEED = 1u << 3,
    MPC_STAGE_CLIPPED_ACCELERATION = 1u << 4,
    MPC_STAGE_CLIPPED_BODY_SPEED = 1u << 5,
    MPC_STAGE_CLIPPED_ACTUAL_STEERING = 1u << 6,
};

typedef struct
{
    MpcModelState_t next;
    float delta_s_m;
    float body_accel_mps2;
    unsigned int branch_flags;
    int valid;
} MpcStageResult_t;

#ifdef __cplusplus
extern "C" {
#endif

/* Authoritative 8-state accepted AutoDRIVE model stage. The control-rate
 * decisions update command states before this same interval's response. */
MpcStageResult_t mpc_vehicle_model_step(
    const MpcModelState_t *state,
    const MpcModelControl_t *control,
    float dt,
    float path_curvature);

VehicleParameters_t vehicle_model_default_parameters(void);
VehicleParameters_t vehicle_model_get_parameters(void);
int vehicle_model_set_parameters(const VehicleParameters_t *parameters);

MpcYawRateModelParameters_t vehicle_model_default_yaw_rate_parameters(void);
MpcYawRateModelParameters_t vehicle_model_get_yaw_rate_parameters(void);
int vehicle_model_set_yaw_rate_parameters(
    const MpcYawRateModelParameters_t *parameters);

/* Identified Unity steering relation used by both prediction and feed-forward. */
float vehicle_model_yaw_rate_gain(float steering_rad);
float vehicle_model_yaw_rate_gain_for_curvature(float curvature_radpm);
float vehicle_model_steering_for_curvature(float curvature_radpm);

/* Apply the controller's temporary target-speed ceiling to every nonlinear
 * and linearized model step.  This is a command-policy limit, not a change to
 * the Unity vehicle physics or the project-wide 16 m/s model envelope. */
float vehicle_model_get_active_target_speed_ceiling(void);
int vehicle_model_set_active_target_speed_ceiling(float ceiling_mps);

#ifdef __cplusplus
}
#endif

#endif /* VEHICLE_MODEL_H */
