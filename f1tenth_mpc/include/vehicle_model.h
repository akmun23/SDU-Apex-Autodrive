/**
 * @file vehicle_model.h
 * @brief Source-command prediction interface for the AutoDRIVE Unity object.
 *
 * This API deliberately has no Pacejka, friction-coefficient, cornering
 * stiffness, load-transfer, or direct-acceleration vehicle contract.  The
 * nonlinear vehicle transition is the checked-in source-derived four-wheel
 * plant advanced at 1 ms. This is not the native patched WheelCollider
 * contact solver and is not claimed to be a bit-exact Unity replica.
 */

#ifndef VEHICLE_MODEL_H
#define VEHICLE_MODEL_H

#include "mpc_types.h"
#include "unity_source_plant_c.h"

enum
{
    MPC_PLANT_STATE_WHEEL_OMEGA_FL = 0,
    MPC_PLANT_STATE_WHEEL_OMEGA_FR = 1,
    MPC_PLANT_STATE_WHEEL_OMEGA_RL = 2,
    MPC_PLANT_STATE_WHEEL_OMEGA_RR = 3,
    MPC_PLANT_STATE_BODY_HEIGHT = 4,
    MPC_PLANT_STATE_BODY_HEIGHT_RATE = 5,
    MPC_PLANT_STATE_ROLL = 6,
    MPC_PLANT_STATE_ROLL_RATE = 7,
    MPC_PLANT_STATE_PITCH = 8,
    MPC_PLANT_STATE_PITCH_RATE = 9,
    MPC_PLANT_STATE_SUSPENSION_JOUNCE_FL = 10,
    MPC_PLANT_STATE_SUSPENSION_JOUNCE_FR = 11,
    MPC_PLANT_STATE_SUSPENSION_JOUNCE_RL = 12,
    MPC_PLANT_STATE_SUSPENSION_JOUNCE_RR = 13,
    MPC_PLANT_STATE_LOW_FORWARD_SPEED_TIME_FL = 14,
    MPC_PLANT_STATE_LOW_FORWARD_SPEED_TIME_FR = 15,
    MPC_PLANT_STATE_LOW_FORWARD_SPEED_TIME_RL = 16,
    MPC_PLANT_STATE_LOW_FORWARD_SPEED_TIME_RR = 17,
    MPC_PLANT_STATE_LOW_SIDE_SPEED_TIME_FL = 18,
    MPC_PLANT_STATE_LOW_SIDE_SPEED_TIME_FR = 19,
    MPC_PLANT_STATE_LOW_SIDE_SPEED_TIME_RL = 20,
    MPC_PLANT_STATE_LOW_SIDE_SPEED_TIME_RR = 21,
    /* Wheel rotation angle is accumulated for odometry's encoder output but
     * does not affect wheel forces or body motion. Keep it in the shared
     * odometry plant only; it is not part of MPC model state. */
    MPC_PLANT_INTERNAL_STATE_COUNT = 22
};

typedef struct
{
    float e_y;
    float e_psi;
    float u;
    float v;
    float r;
    /* Two command samples published by odometry: slot 1 is the preceding
     * command, slot 2 is the newest command associated with this epoch. */
    float delayed_steering_command_1;
    float delayed_steering_command_2;
    /* Unity keeps a physical wheel angle, slewed by the source plant. */
    float actual_steering_angle;
    /* Odom-published throttle history; slot 2 is the newest in-flight
     * command. Neither sample is an optimizer cost/state coordinate. */
    float delayed_throttle_command_1;
    float delayed_throttle_command_2;
    /* Four wheel rates, six sprung-body values and four suspension jounces.
     * Encoder wheel-angle accumulation is deliberately omitted: it is an
     * odometry output accumulator and has no effect on MPC plant dynamics. */
    double source_plant_internal_state[MPC_PLANT_INTERNAL_STATE_COUNT];
    int source_plant_initialized;
    int source_suspension_history_initialized;
    int source_imu_state_initialized;
} MpcModelState_t;

/* Non-recursive per-wheel quantities emitted by a plant transition. Keeping
 * these outside MpcModelState_t avoids copying diagnostics through every
 * finite-difference perturbation and state handoff. */
typedef struct
{
    double longitudinal_slip[4];
    double maximum_positive_longitudinal_slip[4];
    double maximum_longitudinal_slip_excess_mps[4];
    double lateral_slip[4];
    double normal_load_n[4];
} MpcModelOutput_t;

typedef struct
{
    /* Absolute actuator targets. Unity applies its own physical steering slew
     * and the control-rate command queues below this interface. */
    float steering_target_rad;
    /* Normalized simulator throttle; zero is Unity's brake branch. */
    float throttle_command;
} MpcModelControl_t;

enum
{
    MPC_STAGE_CLIPPED_STEERING_TARGET = 1u << 0,
    MPC_STAGE_CLIPPED_THROTTLE = 1u << 1,
    MPC_STAGE_PLANT_BRAKE = 1u << 2,
};

typedef struct
{
    MpcModelState_t next;
    MpcModelOutput_t output;
    float delta_s_m;
    float body_forward_delta_m;
    float body_lateral_delta_m;
    float body_yaw_delta_rad;
    unsigned int branch_flags;
    int valid;
} MpcStageResult_t;

#ifdef __cplusplus
extern "C" {
#endif

/* Seed missing wheel/suspension state for explicit cold-start use. */
int vehicle_model_initialize_plant_state(MpcModelState_t *state);

/* Unity-source tire slip metrics for the current MPC plant state. */
int vehicle_model_longitudinal_slip_excess(
    const MpcModelState_t *state,
    double excess_speed_mps[4],
    double excess_derivative_wheel_omega[4]);

/* One MPC horizon stage at the command-application epoch: drive the source
 * plant with this decision, then shift it into the carried command history. */
MpcStageResult_t mpc_vehicle_model_step(
    const MpcModelState_t *state,
    const MpcModelControl_t *control,
    float dt,
    float path_curvature);

/* Recompute only the Frenet/path-coordinate part of a known physical step.
 * The full plant transition is independent of path curvature; callers may
 * reuse that transition when references are resampled, provided the input
 * physical state and command sequence are unchanged. */
int vehicle_model_reproject_tracking(
    const MpcModelState_t *state,
    const MpcStageResult_t *physical_step,
    float path_curvature,
    MpcStageResult_t *reprojected_step);

/* Pre-command-time propagation uses newest accepted command in slot 2 without
 * advancing either command-history slot. */
MpcStageResult_t mpc_vehicle_model_propagate(
    const MpcModelState_t *state,
    float dt,
    float path_curvature);

/* Shift command history and append one newly accepted command. */
int vehicle_model_queue_command(
    MpcModelState_t *state,
    float steering_target_rad,
    float throttle_command,
    int throttle_valid);

VehicleParameters_t vehicle_model_default_parameters(void);
VehicleParameters_t vehicle_model_get_parameters(void);
int vehicle_model_set_parameters(const VehicleParameters_t *parameters);

/* Geometric initial steering seed. The four-wheel plant computes yaw response. */
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
