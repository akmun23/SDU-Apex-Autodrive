/**
 * @file mpc.h
 * @brief Riccati-ADMM Model Predictive Control public API.
 * @details Main interface for the F1/10th MPC Riccati-ADMM implementation.
 * @dependencies mpc_types.h, vehicle_model.h, util_math.h
 */

#ifndef MPC_H
#define MPC_H

#include "mpc_types.h"
#include "vehicle_model.h"
#include "util_math.h"

/*===========================================================================
 * Configuration Helpers
 *===========================================================================*/

/**
 * @brief Read a float environment variable with fallback.
 * @param name Environment variable name to query.
 * @param default_val Fallback value when the variable is absent or invalid.
 * @return Parsed float value, or default_val when not available.
 */
float get_env_float(const char *name, float default_val);

/**
 * @brief Read an integer environment variable with fallback.
 * @param name Environment variable name to query.
 * @param default_val Fallback value when the variable is absent or invalid.
 * @return Parsed integer value, or default_val when not available.
 */
int get_env_int(const char *name, int default_val);

/**
 * @brief Build the default MPC configuration and apply environment overrides.
 * @return MpcConfiguration_t populated with runtime-ready default parameters.
 */
MpcConfiguration_t get_default_configuration(void);

/*===========================================================================
 * MPC Initialization
 *===========================================================================*/

/**
 * @brief Initialize the MPC system with default F1/10th configuration.
 * @return None.
 */
void mpc_initialize(void);

/**
 * @brief Initialize the MPC system with a caller-supplied configuration.
 * @param configuration Pointer to custom MPC configuration; NULL selects defaults.
 * @return None.
 */
void mpc_initialize_with_configuration(const MpcConfiguration_t *configuration);

/*===========================================================================
 * Control Computation
 *===========================================================================*/

/**
 * @brief Compute optimal control for current vehicle state (Frenet frame).
 * @param current_frenet_state Current state in Frenet frame.
 * @param reference_trajectory Array of reference points (length >= horizon).
 * @param result Output: optimal control and solver status.
 * @return Solver status code indicating convergence result.
 *         MPC_STATUS_SUCCESS when an optimal solution is found;
 *         MPC_STATUS_MAXIMUM_ITERATIONS_REACHED when iteration budget is exhausted
 *         but result->optimal_control still contains the best available control;
 *         MPC_STATUS_ERROR on NULL input or unrecoverable failure.
 */
MpcSolverStatus_t mpc_compute_optimal_control(
    const FrenetState_t *current_frenet_state,
    const TrajectoryReferencePoint_t *reference_trajectory,
    MpcSolverResult_t *result);

/*===========================================================================
 * Configuration Access
 *===========================================================================*/

/**
 * @brief Retrieve a copy of the active MPC configuration.
 * @return Copy of the current MpcConfiguration_t; all fields reflect the
 *         most recently applied configuration.
 */
MpcConfiguration_t mpc_get_configuration(void);

/**
 * @brief Replace the active MPC configuration.
 * @param configuration Pointer to new configuration; call is a no-op if NULL.
 * @return None.
 */
void mpc_set_configuration(const MpcConfiguration_t *configuration);

/**
 * @brief Clear solver warm-start state and reset previous-control memory.
 * @details Forces the next solve to start from a cold initial condition.
 *          Does not change the active configuration or vehicle parameters.
 * @return None.
 */
void mpc_reset(void);

/**
 * @brief Supply the previous command using the nominal control interval.
 * @details Compatibility entry point for callers without source-time timing
 *          or steering feedback. Runtime callers should use
 *          mpc_set_previous_command_with_dt().
 * @param command Pointer to the previous command; no-op if NULL.
 * @return None.
 */
void mpc_set_previous_command(const ControlInput_t *command);

/**
 * @brief Supply the previous command and the source-time state interval.
 * @details The interval is used for steering-command rate and effective-
 *          steering propagation. It must come from source/state stamps, not
 *          callback arrival time. A valid measured steering value initializes
 *          the effective/actual steering state at the current source epoch.
 */
void mpc_set_previous_command_with_dt(
    const ControlInput_t *command,
    float source_dt_seconds,
    float measured_steering_rad,
    int measured_steering_valid);

/*===========================================================================
 * Diagnostics
 *===========================================================================*/

/**
 * @brief Copy the last solved projected ADMM plan for diagnostics/visualization.
 * @details Exposes the feasible projected state/control horizon (`z_x`, `z_u`)
 *          from the most recent solve so offline tools can visualize the MPC's
 *          planned trajectory. Returns 0 when no plan is available yet.
 * @param x_out Optional output buffer for the augmented state plan
 *              [PREDICTION_HORIZON + 1][RICCATI_MAX_NX].
 * @param u_out Optional output buffer for the control plan
 *              [PREDICTION_HORIZON][RICCATI_MAX_NU].
 * @return 1 when a plan was copied, 0 otherwise.
 */
int mpc_debug_copy_last_plan(
    float x_out[PREDICTION_HORIZON + 1][RICCATI_MAX_NX],
    float u_out[PREDICTION_HORIZON][RICCATI_MAX_NU]);

#endif /* MPC_H */
