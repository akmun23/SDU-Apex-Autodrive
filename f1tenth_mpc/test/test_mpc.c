#include "mpc.h"

#include <math.h>
#include <stdio.h>

static int failures = 0;

static void check_true(int condition, const char *message)
{
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", message);
        ++failures;
    }
}

static float first_speed_slew_for_configuration(
    const MpcConfiguration_t *configuration,
    const FrenetState_t *state,
    const TrajectoryReferencePoint_t reference[PREDICTION_HORIZON])
{
    mpc_initialize_with_configuration(configuration);
    const ControlInput_t zero_command = {0.0f, 0.0f};
    mpc_set_previous_command_with_dt(&zero_command, TIME_STEP_SECONDS, 0.0f, 0);
    MpcSolverResult_t result = {0};
    const MpcSolverStatus_t status = mpc_compute_optimal_control(
        state, reference, &result);
    if (status != MPC_STATUS_SUCCESS &&
        status != MPC_STATUS_MAXIMUM_ITERATIONS_REACHED) {
        return NAN;
    }
    return result.optimal_control.target_speed_rate;
}

int main(void)
{
    const MpcConfiguration_t configuration = get_default_configuration();
    const float state_weights[NX_AUG] = {
        configuration.weight_lateral_error,
        configuration.weight_heading_error,
        configuration.weight_velocity,
        configuration.weight_lateral_velocity,
        configuration.weight_yaw_rate,
        configuration.weight_target_speed_state,
        configuration.weight_commanded_steering,
        configuration.weight_effective_steering,
        configuration.weight_steering_rate,
        configuration.weight_target_speed_rate_change,
    };
    check_true(NX_FRENET == 6 && NX_AUG == 10 && NU == 2,
        "production MPC uses six Frenet states and ten augmented states");
    for (int state_index = 0; state_index < NX_AUG; ++state_index) {
        check_true(isfinite(state_weights[state_index]) &&
                   state_weights[state_index] > 0.0f,
            "every augmented state has a positive default quadratic weight");
    }

    TrajectoryReferencePoint_t reference[PREDICTION_HORIZON];
    for (int index = 0; index < PREDICTION_HORIZON; ++index) {
        reference[index] = (TrajectoryReferencePoint_t){
            .reference_velocity = 4.0f,
            .path_curvature = 0.05f,
            .left_wall_bound = 1.0f,
            .right_wall_bound = 1.0f,
        };
    }

    mpc_initialize_with_configuration(&configuration);
    const FrenetState_t state = {
        .flat_error = 0.05f,
        .fhead_error = 0.02f,
        .flong_vel = 3.5f,
        .flat_vel = 0.0f,
        .fyaw_rate = 0.0f,
        .ftarget_speed_mps = 3.5f,
    };
    MpcSolverResult_t result = {0};
    const MpcSolverStatus_t status = mpc_compute_optimal_control(
        &state, reference, &result);

    check_true(status == MPC_STATUS_SUCCESS,
               "source-command MPC converges within the ADMM iteration budget");
    check_true(isfinite(result.optimal_control.steer_ang) &&
               isfinite(result.optimal_control.target_speed_rate),
               "MPC output is finite");
    check_true(fabsf(result.optimal_control.steer_ang) <= 0.523599f,
               "MPC steering respects the Unity source limit");
    check_true(result.optimal_control.target_speed_rate <= 3.0001f &&
               result.optimal_control.target_speed_rate >= -8.0001f,
               "MPC longitudinal output respects target-speed policy bounds");
    check_true(result.iterations_used > 0 &&
               result.iterations_used <= configuration.max_solver_iterations,
               "10-state solve uses the configured ADMM iteration budget");
    check_true(isfinite(result.primal_residual) && isfinite(result.dual_residual),
               "ADMM returns finite primal and dual residuals");

    /* A QP rate decision belongs to the future nominal interval, not the
     * previous observed source interval. */
    mpc_initialize_with_configuration(&configuration);
    const ControlInput_t zero_command = {0.0f, 0.0f};
    const float measured_dt = 0.050f;
    mpc_set_previous_command_with_dt(&zero_command, measured_dt, 0.0f, 0);
    MpcSolverResult_t variable_dt_result = {0};
    const MpcSolverStatus_t variable_dt_status = mpc_compute_optimal_control(
        &state, reference, &variable_dt_result);
    float plan_x[PREDICTION_HORIZON + 1][RICCATI_MAX_NX];
    float plan_u[PREDICTION_HORIZON][RICCATI_MAX_NU];
    const int has_plan = mpc_debug_copy_last_plan(plan_x, plan_u);
    check_true(variable_dt_status == MPC_STATUS_SUCCESS,
               "MPC converges for a measured non-nominal control interval");
    check_true(has_plan, "debug plan exposes the optimized first control");
    if (has_plan) {
        const float expected_steering_target =
            TIME_STEP_SECONDS * plan_u[0][0];
        check_true(fabsf(variable_dt_result.optimal_control.steer_ang -
                         expected_steering_target) < 1.0e-4f,
            "steering angle target integrates optimized rate over prediction dt");
    }

    MpcConfiguration_t no_target_state_cost = configuration;
    MpcConfiguration_t high_target_state_cost = configuration;
    no_target_state_cost.weight_target_speed_state = 0.0f;
    high_target_state_cost.weight_target_speed_state = 1000.0f;
    const float low_weight_rate = first_speed_slew_for_configuration(
        &no_target_state_cost, &state, reference);
    const float high_weight_rate = first_speed_slew_for_configuration(
        &high_target_state_cost, &state, reference);
    check_true(isfinite(low_weight_rate) && isfinite(high_weight_rate) &&
               fabsf(high_weight_rate - low_weight_rate) > 0.01f,
        "target-speed-state weight participates in the optimized control");

    if (failures != 0) {
        fprintf(stderr, "%d MPC core test(s) failed\n", failures);
        return 1;
    }
    puts("MPC core tests passed");
    return 0;
}
