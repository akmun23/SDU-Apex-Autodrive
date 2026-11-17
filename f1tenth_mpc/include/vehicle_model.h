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

/* Interval grip losses emitted by a plant transition for the cycle scorer. */
typedef struct
{
    double maximum_grip_loss[4];
} MpcModelOutput_t;

typedef struct
{
    /* Absolute actuator targets. Unity applies its own physical steering slew
     * and the control-rate command queues below this interface. */
    float steering_target_rad;
    /* Normalized simulator throttle; zero is Unity's brake branch. */
    float throttle_command;
} MpcModelControl_t;

typedef struct
{
    MpcModelState_t next;
    MpcModelOutput_t output;
    float delta_s_m;
    float body_forward_delta_m;
    float body_lateral_delta_m;
    float body_yaw_delta_rad;
    int valid;
} MpcStageResult_t;

#ifdef __cplusplus
extern "C" {
#endif

/* Canonical physical MPC <-> 26-coordinate source conversion.
 * The vehicle dynamics use this same conversion. Import
 * changes only physical state coordinates; path errors, actuator history
 * and source-initialized flags remain owned by the caller.
 */
int vehicle_model_pack_source_state(
    const MpcModelState_t *state,
    double source_state[UNITY_SOURCE_PLANT_STATE_SIZE]);
int vehicle_model_inject_source_state(
    MpcModelState_t *state,
    const double source_state[UNITY_SOURCE_PLANT_STATE_SIZE]);

/* Full 26-channel source-plant tangent integration in 28 independent
 * directions. The caller seeds and receives all wheel/suspension/sticky-tire
 * history coordinates plus body displacement. grip_residual[w] is the signed
 * maximum over the interval's substeps of |wheel surface - ground speed| minus
 * the tire curve's first extremum (positive part = the objective's slip
 * excess); grip_loss_derivative is its tangent. Requires a
 * source-initialized state. No finite differences, no physics substitution. */
int vehicle_model_full_source_interval_sensitivities(
    const MpcModelState_t *state,const MpcModelControl_t *control,float dt,
    const double seed_state[26][UNITY_SOURCE_PLANT_SENSITIVITY_WIDTH],
    const double seed_control[2][UNITY_SOURCE_PLANT_SENSITIVITY_WIDTH],
    unsigned int force_substep_cap,
    double next_state[26][UNITY_SOURCE_PLANT_SENSITIVITY_WIDTH],
    double body_motion[3][UNITY_SOURCE_PLANT_SENSITIVITY_WIDTH],
    double grip_residual[4],
    double grip_loss_derivative[4][UNITY_SOURCE_PLANT_SENSITIVITY_WIDTH]);
/* Exact slow-coordinate expansion of one interval (see
 * UnitySourcePlantHessianC): value, gradient and Hessian with respect to
 * {u, v, yaw rate, steering target, throttle}. A command
 * held on a legal actuator bound can only move inward, so its row and column
 * are zeroed exactly as in the first-order sensitivities. */
int vehicle_model_source_interval_hessian(
    const MpcModelState_t *state,const MpcModelControl_t *control,float dt,
    unsigned int force_substep_cap,UnitySourcePlantHessianC *hessian);

/* Seed missing wheel/suspension state for explicit cold-start use. */
int vehicle_model_initialize_plant_state(MpcModelState_t *state);

/* One MPC horizon stage at the command-application epoch: drive the source
 * plant with this decision, then shift it into the carried command history. */
MpcStageResult_t mpc_vehicle_model_step(
    const MpcModelState_t *state,
    const MpcModelControl_t *control,
    float dt,
    float path_curvature);

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

#ifdef __cplusplus
}
#endif

#endif /* VEHICLE_MODEL_H */
