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

#endif /* VEHICLE_MODEL_H */
