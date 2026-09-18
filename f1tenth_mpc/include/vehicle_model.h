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
} MpcModelState_t;

typedef struct
{
    float steering_rate;
    float target_speed_rate;
} MpcModelControl_t;

enum
{
    MPC_STAGE_CLIPPED_STEERING_RATE = 1u << 0,
    MPC_STAGE_CLIPPED_SPEED_RATE = 1u << 1,
    MPC_STAGE_CLIPPED_STEERING_COMMAND = 1u << 2,
    MPC_STAGE_CLIPPED_TARGET_SPEED = 1u << 3,
    MPC_STAGE_CLIPPED_ACCELERATION = 1u << 4,
    MPC_STAGE_CLIPPED_BODY_SPEED = 1u << 5,
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

/* Authoritative 7-state accepted AutoDRIVE model stage. The control-rate
 * decisions update command states before this same interval's response. */
MpcStageResult_t mpc_vehicle_model_step(
    const MpcModelState_t *state,
    const MpcModelControl_t *control,
    float dt,
    float path_curvature);

VehicleParameters_t vehicle_model_default_parameters(void);
VehicleParameters_t vehicle_model_get_parameters(void);
int vehicle_model_set_parameters(const VehicleParameters_t *parameters);

/* Clamp MPC command-space output to the source steering and project limits. */
ControlInput_t vehicle_model_saturate_control(const ControlInput_t *raw_control);

VehicleState_t vehicle_model_predict_next_state(
    const VehicleState_t *current_state,
    const ControlInput_t *control_input,
    float time_step);

FrenetState_t vehicle_model_predict_next_frenet_state(
    const FrenetState_t *state,
    const ControlInput_t *control,
    float time_step,
    float path_curvature);

void vehicle_model_predict_trajectory(
    const VehicleState_t *initial_state,
    const ControlInput_t *control_sequence,
    float time_step,
    uint16_t step_count,
    VehicleState_t *predicted_trajectory);

/* Numerical linearization is intentional: it guarantees that the Riccati
 * stage uses the same source-command map as recursive scoring. */
void vehicle_model_compute_frenet_linearization(
    const FrenetState_t *frenet_state,
    const ControlInput_t *operating_control,
    float time_step,
    float path_curvature,
    float reference_velocity,
    float state_matrix_A[NX_FRENET][NX_FRENET],
    float input_matrix_B[NX_FRENET][NU]);

#ifdef __cplusplus
}
#endif

#endif /* VEHICLE_MODEL_H */
