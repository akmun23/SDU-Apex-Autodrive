#include "mpc_rti.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static int failures = 0;

static void check_true(int condition, const char *message)
{
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", message);
        ++failures;
    }
}

static void check_close(float actual, float expected, float tolerance,
                        const char *message)
{
    check_true(isfinite(actual) && fabsf(actual - expected) <= tolerance,
               message);
}

static MpcRtiConfiguration_t test_configuration(void)
{
    return (MpcRtiConfiguration_t){
        .weight_e_y = 1500.0f,
        .weight_e_psi = 50.0f,
        .weight_u = 200.0f,
        .weight_target_speed_state = 20.0f,
        .weight_v = 0.0f,
        .weight_r = 1.5f,
        .weight_steering_command = 1.0f,
        .weight_steering_rate = 2.0f,
        .weight_target_speed_rate = 0.5f,
        .weight_steering_rate_change = 5.0f,
        .weight_target_speed_rate_change = 5.0f,
        .terminal_multiplier = 3.0f,
        .max_speed_mps = 16.0f,
        .active_speed_ceiling_mps = 8.0f,
        .max_steering_rad = SOURCE_MAX_STEERING_RAD,
        .max_steering_rate_radps = SOURCE_STEERING_RATE_RADPS,
        .max_target_speed_rate_increase_mps2 = 3.0f,
        .max_target_speed_rate_reduction_mps2 = 8.0f,
        .corridor_margin_m = 0.05f,
        .first_prediction_corridor_margin_m = 0.05f,
        .corridor_preview_halfwidth_m = 0.0f,
        .nonlinear_corridor_tolerance_m = 0.0f,
    };
}

static void make_nominal(
    MpcRtiState_t states[PREDICTION_HORIZON + 1],
    MpcModelControl_t controls[PREDICTION_HORIZON],
    MpcRtiReference_t refs[PREDICTION_HORIZON + 1])
{
    states[0] = (MpcRtiState_t){
        .plant = {.e_y = 0.02f, .e_psi = -0.01f, .u = 5.0f,
                  .v = 0.04f, .r = 0.21f, .target_speed = 5.2f,
                  .steering_command = 0.03f},
        .previous_steering_rate = 0.4f,
        .previous_target_speed_rate = -0.8f};
    controls[0] = (MpcModelControl_t){
        .steering_rate = -0.2f, .target_speed_rate = 0.3f};
    const MpcStageResult_t predicted = mpc_vehicle_model_step(
        &states[0].plant, &controls[0], 0.025f, 0.04f);
    states[1].plant = predicted.next;
    states[1].previous_steering_rate = controls[0].steering_rate;
    states[1].previous_target_speed_rate = controls[0].target_speed_rate;
    refs[0] = (MpcRtiReference_t){
        .e_y = 0.0f, .e_psi = 0.0f, .u = 5.4f, .v = 0.0f,
        .r = 0.216f, .steering_command = atanf(0.04f /
            MPC_YAW_RATE_STEERING_GAIN_PER_M),
        .target_speed = 5.1f, .target_speed_rate = 0.0f,
        .path_curvature = 0.04f, .left_bound = 0.45f,
        .right_bound = 0.40f};
    refs[1] = refs[0];
    refs[1].u = 5.3f;
    refs[1].r = 0.212f;
}

static float matrix_cost_zero_reference(
    const RiccatiStepData_t *stage,
    const float control[MPC_RTI_NU])
{
    const float zero_state[MPC_RTI_NX] = {0};
    return mpc_rti_matrix_stage_cost(stage, zero_state, control);
}

static void test_absolute_nine_state_affine_ltv_build(void)
{
    MpcRtiState_t states[PREDICTION_HORIZON + 1] = {0};
    MpcModelControl_t controls[PREDICTION_HORIZON] = {0};
    MpcRtiReference_t refs[PREDICTION_HORIZON + 1] = {0};
    MpcRtiProblem_t problem;
    MpcRtiConfiguration_t config = test_configuration();
    make_nominal(states, controls, refs);

    check_true(mpc_rti_build_ltv_qp(states, controls, refs, 1, 0.025f,
                                   &config, &problem),
               "one-stage nine-state LTV QP builds");
    check_true(problem.horizon == 1,
               "QP preserves the requested horizon length");
    MpcStageLinearization_t expected;
    check_true(mpc_model_linearize(&states[0].plant, &controls[0],
        0.025f, refs[0].path_curvature, &expected),
        "independent plant linearization for augmentation test succeeds");
    for (int row = 0; row < MPC_RTI_PLANT_NX; ++row) {
        check_close(problem.steps[0].d[row], expected.d[row], 1.0e-7f,
                    "augmented dynamics preserve plant affine offset");
        for (int col = 0; col < MPC_RTI_PLANT_NX; ++col)
            check_close(problem.steps[0].A[row][col], expected.A[row][col],
                        1.0e-7f,
                        "augmented dynamics preserve plant A block");
        for (int input = 0; input < MPC_RTI_NU; ++input)
            check_close(problem.steps[0].B[row][input],
                        expected.B[row][input], 1.0e-7f,
                        "augmented dynamics preserve plant B block");
    }
    for (int memory_row = MPC_RTI_IDX_PREVIOUS_STEERING_RATE;
         memory_row < MPC_RTI_NX; ++memory_row) {
        check_close(problem.steps[0].d[memory_row], 0.0f, 1.0e-7f,
                    "previous-control memory has zero affine offset");
        for (int col = 0; col < MPC_RTI_NX; ++col)
            check_close(problem.steps[0].A[memory_row][col], 0.0f,
                        1.0e-7f,
                        "previous-control memory has zero A rows");
        for (int input = 0; input < MPC_RTI_NU; ++input)
            check_close(problem.steps[0].B[memory_row][input],
                memory_row == MPC_RTI_IDX_PREVIOUS_STEERING_RATE + input
                    ? 1.0f : 0.0f,
                1.0e-7f,
                "previous-control memory advances as p_next equals input");
    }
    check_close(problem.x0[MPC_RTI_IDX_PREVIOUS_STEERING_RATE], 0.4f,
                1.0e-7f, "QP x0 carries prior steering rate");
    check_close(problem.x0[MPC_RTI_IDX_PREVIOUS_TARGET_SPEED_RATE], -0.8f,
                1.0e-7f, "QP x0 carries prior target-speed rate");

    check_close(problem.steps[0].x_lb[MPC_RTI_IDX_EY], -0.35f, 1.0e-7f,
                "right corridor and robust margin form e_y lower bound");
    check_close(problem.steps[0].x_ub[MPC_RTI_IDX_EY], 0.40f, 1.0e-7f,
                "left corridor and robust margin form e_y upper bound");
    check_close(problem.steps[0].x_ub[MPC_RTI_IDX_U], 16.0f, 1.0e-7f,
                "body-speed state has project maximum bound");
    check_close(problem.steps[0].x_ub[MPC_RTI_IDX_TARGET_SPEED], 8.0f,
                1.0e-7f, "target-speed state uses active ceiling");
    check_close(problem.steps[0].x_lb[MPC_RTI_IDX_STEERING_COMMAND],
                -SOURCE_MAX_STEERING_RAD, 1.0e-7f,
                "steering command has source lower bound");
    check_close(problem.steps[0].u_lb[0], -3.2f, 1.0e-7f,
                "steering-rate input uses source bound");
    check_close(problem.steps[0].u_lb[1], -8.0f, 1.0e-7f,
                "target-speed-rate input uses braking bound");
    check_close(problem.steps[0].u_ub[1], 3.0f, 1.0e-7f,
                "target-speed-rate input uses acceleration bound");
}

static void test_previous_control_penalty_matrix_algebra(void)
{
    MpcRtiState_t states[PREDICTION_HORIZON + 1] = {0};
    MpcModelControl_t controls[PREDICTION_HORIZON] = {0};
    MpcRtiReference_t refs[PREDICTION_HORIZON + 1] = {0};
    MpcRtiProblem_t problem;
    MpcRtiConfiguration_t config = test_configuration();
    config.weight_e_y = 3.5f;
    config.weight_e_psi = 0.7f;
    config.weight_u = 2.0f;
    config.weight_v = 0.4f;
    config.weight_r = 1.5f;
    config.weight_steering_command = 0.8f;
    config.weight_steering_rate = 1.1f;
    config.weight_target_speed_rate = 0.6f;
    config.weight_steering_rate_change = 2.3f;
    config.weight_target_speed_rate_change = 1.4f;
    make_nominal(states, controls, refs);
    refs[0].e_y = -0.01f;
    refs[0].e_psi = 0.015f;
    refs[0].u = 0.54f;
    refs[0].v = -0.02f;
    refs[0].r = -0.2f;
    refs[0].steering_command = 0.025f;
    check_true(mpc_rti_build_ltv_qp(states, controls, refs, 1, 0.025f,
                                   &config, &problem),
               "build rate-change cost test QP");

    MpcRtiState_t sample = states[0];
    sample.plant.e_y = 0.09f;
    sample.plant.e_psi = 0.035f;
    sample.plant.u = 0.47f;
    sample.plant.v = -0.12f;
    sample.plant.r = 0.48f;
    sample.plant.target_speed = 6.0f;
    sample.plant.steering_command = 0.07f;
    sample.previous_steering_rate = -0.055f;
    sample.previous_target_speed_rate = -0.11f;
    const MpcModelControl_t input = {
        .steering_rate = 0.075f, .target_speed_rate = 0.09f};
    const float x[MPC_RTI_NX] = {
        sample.plant.e_y, sample.plant.e_psi, sample.plant.u,
        sample.plant.v, sample.plant.r, sample.plant.target_speed,
        sample.plant.steering_command, sample.previous_steering_rate,
        sample.previous_target_speed_rate};
    const float u[MPC_RTI_NU] = {
        input.steering_rate, input.target_speed_rate};
    const float matrix_delta =
        mpc_rti_matrix_stage_cost(&problem.steps[0], x, u) -
        matrix_cost_zero_reference(&problem.steps[0], (float[2]){0.0f, 0.0f});
    MpcRtiState_t zero_state = {0};
    const MpcModelControl_t zero_input = {0};
    const float scalar_delta =
        mpc_rti_scalar_stage_cost(&config, &refs[0], &sample, &input) -
        mpc_rti_scalar_stage_cost(&config, &refs[0], &zero_state,
                                  &zero_input);
    check_close(matrix_delta, scalar_delta, 2.0e-5f,
                "Q/R/N expansion equals scalar tracking and delta-input costs");

    check_close(problem.steps[0].Q_diag[MPC_RTI_IDX_PREVIOUS_STEERING_RATE],
                2.0f * config.weight_steering_rate_change, 1.0e-7f,
                "rate-change penalty contributes Q_pp = 2w");
    check_close(problem.steps[0].R_diag[0],
                2.0f * (config.weight_steering_rate +
                        config.weight_steering_rate_change), 1.0e-7f,
                "rate-change penalty contributes R_uu = 2w plus effort");
    check_close(problem.steps[0].N[MPC_RTI_IDX_PREVIOUS_STEERING_RATE][0],
                -2.0f * config.weight_steering_rate_change, 1.0e-7f,
                "rate-change penalty contributes N_pu = -2w");
    check_close(problem.steps[0].Q_diag[MPC_RTI_IDX_TARGET_SPEED],
                2.0f * config.weight_target_speed_state, 1.0e-7f,
                "target-speed state has a direct tracking weight");
    check_close(problem.steps[0].q[MPC_RTI_IDX_TARGET_SPEED],
                -2.0f * config.weight_target_speed_state * refs[0].target_speed,
                1.0e-5f, "target-speed state has a direct linear cost");

    MpcRtiConfiguration_t zero_config = config;
    zero_config.weight_v = 4.0f;
    MpcRtiReference_t zero_ref = {0};
    zero_ref.left_bound = zero_ref.right_bound = 1.0f;
    MpcRtiState_t zero_nominal[PREDICTION_HORIZON + 1] = {0};
    MpcModelControl_t zero_nominal_controls[PREDICTION_HORIZON] = {0};
    MpcRtiReference_t zero_refs[PREDICTION_HORIZON + 1] = {0};
    zero_refs[0] = zero_ref;
    zero_refs[1] = zero_ref;
    check_true(mpc_rti_build_ltv_qp(zero_nominal,
        zero_nominal_controls, zero_refs, 1, 0.025f, &zero_config,
        &problem), "zero-reference QP builds");
    check_close(mpc_rti_scalar_stage_cost(&zero_config, &zero_ref,
        &zero_nominal[0], &zero_nominal_controls[0]), 0.0f, 1.0e-7f,
        "zero reference, zero state and zero input have zero scalar cost");
    check_close(mpc_rti_matrix_stage_cost(&problem.steps[0],
        (float[MPC_RTI_NX]){0}, (float[MPC_RTI_NU]){0}), 0.0f, 1.0e-7f,
        "zero reference, zero state and zero input have zero matrix cost");
    zero_nominal[0].plant.target_speed = 6.0f;
    check_true(mpc_rti_build_ltv_qp(zero_nominal,
        zero_nominal_controls, zero_refs, 1, 0.025f, &zero_config,
        &problem), "QP tolerates target-speed state away from reference");
    check_close(problem.steps[0].Q_diag[MPC_RTI_IDX_TARGET_SPEED],
                2.0f * zero_config.weight_target_speed_state, 1.0e-7f,
                "target-speed state is directly tracked to its reference");
}

static void test_invalid_problem_rejected(void)
{
    MpcRtiState_t states[PREDICTION_HORIZON + 1] = {0};
    MpcModelControl_t controls[PREDICTION_HORIZON] = {0};
    MpcRtiReference_t refs[PREDICTION_HORIZON + 1] = {0};
    MpcRtiProblem_t problem;
    MpcRtiConfiguration_t config = test_configuration();
    refs[0].left_bound = refs[0].right_bound = 0.04f;
    refs[1] = refs[0];
    check_true(!mpc_rti_build_ltv_qp(states, controls, refs, 1, 0.025f,
                                    &config, &problem),
               "infeasible corridor margin is rejected, not collapsed");
    refs[0].left_bound = refs[0].right_bound = 1.0f;
    refs[1] = refs[0];
    config.active_speed_ceiling_mps = config.max_speed_mps + 1.0f;
    check_true(!mpc_rti_build_ltv_qp(states, controls, refs, 1, 0.025f,
                                    &config, &problem),
               "active speed ceiling above project max is rejected");
}

static void make_circle_trajectory(
    MpcTrajectorySample_t points[80], size_t *count, double *lap_length)
{
    const double radius = 10.0;
    const size_t n = 80;
    for (size_t i = 0; i < n; ++i) {
        const double angle = 2.0 * 3.14159265358979323846 *
            (double)i / (double)n;
        points[i] = (MpcTrajectorySample_t){
            .s = radius * angle,
            .x = radius * cos(angle),
            .y = radius * sin(angle),
            .heading = angle + 0.5 * 3.14159265358979323846,
            .curvature = 1.0 / radius,
            .speed = 4.0 -
                (MPC_LONGITUDINAL_RESPONSE_BIAS_MPS2 +
                 MPC_LONGITUDINAL_SPEED_COEFF_PER_S * 4.0f) /
                    MPC_LONGITUDINAL_TARGET_ERROR_GAIN_PER_S,
            .left_bound = 0.60,
            .right_bound = 0.60,
            .acceleration = 1.5};
    }
    *count = n;
    (void)mpc_trajectory_prepare(points, count, lap_length);
}

static void test_two_pass_nominal_qp_and_nonlinear_candidate(void)
{
    MpcTrajectorySample_t trajectory[80];
    size_t trajectory_count = 0;
    double lap_length = 0.0;
    make_circle_trajectory(trajectory, &trajectory_count, &lap_length);
    check_true(trajectory_count == 80 && lap_length > 60.0,
               "synthetic continuous loop is prepared");

    const float speed = 4.0f;
    const float target_speed = speed -
        (MPC_LONGITUDINAL_RESPONSE_BIAS_MPS2 +
         MPC_LONGITUDINAL_SPEED_COEFF_PER_S * speed) /
            MPC_LONGITUDINAL_TARGET_ERROR_GAIN_PER_S;
    const float curvature = 0.1f;
    const MpcRtiState_t current = {
        .plant = {.e_y = 0.0f, .e_psi = 0.0f, .u = speed, .v = 0.0f,
                  .r = curvature * speed, .target_speed = target_speed,
                  .steering_command = atanf(curvature /
                      MPC_YAW_RATE_STEERING_GAIN_PER_M)},
        .previous_steering_rate = 0.0f,
        .previous_target_speed_rate = 0.0f};
    MpcRtiConfiguration_t config = test_configuration();
    config.active_speed_ceiling_mps = 8.0f;
    MpcRtiNominal_t nominal;
    MpcRtiReference_t cold_references[PREDICTION_HORIZON + 1];
    const int horizon = 4;
    check_true(mpc_rti_build_nominal(&current, 0.0, NULL, trajectory,
        trajectory_count, lap_length, 0.025f, horizon, &config, &nominal,
        cold_references), "cold two-pass nonlinear nominal builds");
    check_true(nominal.valid && nominal.horizon == horizon,
               "cold nominal retains N+1 states and N controls");
    check_true(nominal.progress[1] > 0.0 && nominal.progress[1] < 0.2,
               "nominal progress follows model delta_s, not a fixed waypoint jump");
    const float reference_speed = (float)trajectory[0].speed;
    const float expected_target_speed = reference_speed +
        (1.5f - MPC_LONGITUDINAL_RESPONSE_BIAS_MPS2 -
         MPC_LONGITUDINAL_SPEED_COEFF_PER_S * reference_speed -
         MPC_LONGITUDINAL_TARGET_RATE_COEFF * 1.5f) /
        MPC_LONGITUDINAL_TARGET_ERROR_GAIN_PER_S - 0.5f * 1.5f * 0.025f;
    check_close(cold_references[0].target_speed, expected_target_speed,
                1.0e-5f, "target-speed reference inverts raceline acceleration");
    check_close(cold_references[0].target_speed_rate, 1.5f, 1.0e-7f,
                "target-speed reference carries raceline acceleration");
    check_close((float)nominal.progress[1], 0.025f * speed, 2.0e-4f,
                "steady constant-curvature nominal advances approximately u*dt");

    MpcRtiState_t prior_rate_outside_current_input_box = current;
    prior_rate_outside_current_input_box.previous_target_speed_rate = -8.05f;
    MpcRtiNominal_t prior_rate_nominal;
    MpcRtiReference_t prior_rate_references[PREDICTION_HORIZON + 1];
    check_true(mpc_rti_build_nominal(&prior_rate_outside_current_input_box,
        0.0, NULL, trajectory, trajectory_count, lap_length, 0.025f,
        horizon, &config, &prior_rate_nominal, prior_rate_references),
        "causal previous-input memory may lie outside the new action box");
    check_true(prior_rate_nominal.controls[0].target_speed_rate >= -8.0f &&
               prior_rate_nominal.controls[0].target_speed_rate <= 3.0f,
        "new optimizer actions remain inside their target-rate box");

    MpcTrajectorySample_t preview_trajectory[80];
    memcpy(preview_trajectory, trajectory, sizeof(trajectory));
    preview_trajectory[1].left_bound = 0.20;
    size_t preview_count = trajectory_count;
    double preview_lap_length = 0.0;
    check_true(mpc_trajectory_prepare(preview_trajectory, &preview_count,
        &preview_lap_length), "corridor-preview trajectory remains valid");
    MpcRtiConfiguration_t preview_config = config;
    preview_config.corridor_preview_halfwidth_m = 0.10f;
    MpcRtiNominal_t preview_nominal;
    MpcRtiReference_t preview_references[PREDICTION_HORIZON + 1];
    const double preview_s = preview_trajectory[1].s - 0.05;
    check_true(mpc_rti_build_nominal(&current, preview_s, NULL,
        preview_trajectory, preview_count, preview_lap_length, 0.025f,
        horizon, &preview_config, &preview_nominal, preview_references),
        "fixed-meter corridor preview nominal builds");
    check_true(preview_references[0].left_bound <= 0.2001f,
        "corridor preview uses the narrow bound inside its physical window");

    MpcRtiNominal_t broad_nominal;
    MpcRtiReference_t broad_references[PREDICTION_HORIZON + 1];
    check_true(mpc_rti_build_nominal(&current, 0.0, NULL,
        preview_trajectory, preview_count, preview_lap_length, 0.025f,
        horizon, &preview_config, &broad_nominal, broad_references),
        "broad-path nominal is built for candidate-progress regression");
    MpcRtiState_t shifted_candidate = current;
    shifted_candidate.plant.e_y = 0.30f;
    MpcModelControl_t zero_candidate_controls[PREDICTION_HORIZON] = {{0}};
    MpcRtiState_t shifted_candidate_states[PREDICTION_HORIZON + 1];
    check_true(mpc_rti_rollout_candidate(&shifted_candidate, preview_s,
        zero_candidate_controls, preview_trajectory, preview_count,
        preview_lap_length, NULL, NULL, NULL, 1, 0.025f, &preview_config,
        shifted_candidate_states, NULL, NULL) == MPC_RTI_ROLLOUT_CORRIDOR,
        "candidate rollout checks corridor at its own progress, not nominal schedule");

    MpcRtiNominal_t warm_seed = nominal;
    for (int k = 0; k <= horizon; ++k)
        warm_seed.progress[k] = (double)k * 0.10;
    for (int k = 0; k < horizon; ++k) {
        warm_seed.controls[k].steering_rate = 0.1f * (float)(k + 1);
        warm_seed.controls[k].target_speed_rate = 0.1f * (float)(k + 1);
    }
    MpcRtiNominal_t warm_nominal;
    MpcRtiReference_t warm_references[PREDICTION_HORIZON + 1];
    check_true(mpc_rti_build_nominal(&current, 0.10, &warm_seed,
        trajectory, trajectory_count, lap_length, 0.025f, horizon, &config,
        &warm_nominal, warm_references), "warm RTI nominal shift and two-pass rollout");
    check_close(warm_nominal.controls[0].steering_rate,
                warm_seed.controls[1].steering_rate, 1.0e-7f,
                "warm nominal controls shift one stage toward the present");
    check_close(warm_nominal.controls[horizon - 1].target_speed_rate,
                warm_seed.controls[horizon - 1].target_speed_rate, 1.0e-7f,
                "warm nominal repeats the final control at the horizon");

    MpcRtiProblem_t problem;
    check_true(mpc_rti_build_ltv_qp(nominal.states, nominal.controls,
        cold_references, horizon, 0.025f, &config, &problem),
        "two-pass nominal builds a complete 9-state LTV QP");
    MpcRtiConfiguration_t first_step_relaxed = config;
    first_step_relaxed.first_prediction_corridor_margin_m = 0.0f;
    MpcRtiProblem_t first_step_relaxed_problem;
    check_true(mpc_rti_build_ltv_qp(nominal.states, nominal.controls,
        cold_references, horizon, 0.025f, &first_step_relaxed,
        &first_step_relaxed_problem),
        "first-prediction corridor envelope can be evaluated independently");
    check_close(first_step_relaxed_problem.steps[1].x_lb[MPC_RTI_IDX_EY],
        -cold_references[1].right_bound, 1.0e-7f,
        "first predicted state uses its configured corridor margin");
    check_close(first_step_relaxed_problem.steps[1].x_ub[MPC_RTI_IDX_EY],
        cold_references[1].left_bound, 1.0e-7f,
        "first predicted upper bound uses its configured corridor margin");
    check_close(first_step_relaxed_problem.steps[2].x_lb[MPC_RTI_IDX_EY],
        config.corridor_margin_m - cold_references[2].right_bound,
        1.0e-7f,
        "normal hard corridor margin remains active after the first prediction");
    RiccatiAdmmConfig_t solver_config = {
        .rho = 7.0f, .rho_u = 7.0f, .tolerance = 1.0e-4f,
        .max_iterations = 500, .adaptive_rho = 0, .shared_rho = 0};
    RiccatiAdmmState_t admm_state;
    RiccatiSolution_t solution = {0};
    riccati_admm_state_init(&admm_state);
    const RiccatiStatus_t solve_status = riccati_admm_solve(
        problem.steps, problem.terminal_Q, problem.terminal_q,
        problem.terminal_x_lb, problem.terminal_x_ub, problem.x0,
        MPC_RTI_NX, MPC_RTI_NU, horizon, &solver_config, &admm_state,
        &solution);
    check_true(solve_status == RICCATI_STATUS_OPTIMAL,
               "9-state RTI test QP solves to declared tolerance");
    if (solve_status != RICCATI_STATUS_ERROR) {
        MpcModelControl_t candidate_controls[PREDICTION_HORIZON] = {0};
        for (int k = 0; k < horizon; ++k) {
            candidate_controls[k].steering_rate = solution.u[k][0];
            candidate_controls[k].target_speed_rate = solution.u[k][1];
        }
        MpcRtiState_t candidate_states[PREDICTION_HORIZON + 1];
        double candidate_progress[PREDICTION_HORIZON + 1];
        MpcRtiCandidatePathDelta_t candidate_path_delta;
        int failure_stage = -1;
        const MpcRtiRolloutStatus_t rollout_status =
            mpc_rti_rollout_candidate(&current, 0.0, candidate_controls,
                trajectory, trajectory_count, lap_length, cold_references,
                nominal.progress, &candidate_path_delta, horizon, 0.025f,
                &config, candidate_states, candidate_progress, &failure_stage);
        check_true(rollout_status == MPC_RTI_ROLLOUT_OK,
                   "QP controls pass exact nonlinear rollout and corridor gate");
        if (rollout_status == MPC_RTI_ROLLOUT_OK) {
            check_true(candidate_path_delta.sample_count == horizon + 1,
                       "candidate-vs-nominal path diagnostics cover the horizon");
            check_true(candidate_path_delta.progress_error_m[0] < 1.0e-9,
                       "candidate and nominal begin at the same progress");
            const MpcStageResult_t expected_step = mpc_vehicle_model_step(
                &current.plant, &candidate_controls[0], 0.025f,
                cold_references[0].path_curvature);
            check_close((float)candidate_progress[1],
                (float)expected_step.delta_s_m, 1.0e-7f,
                "candidate progress uses the nonlinear stage delta_s");
        }
    }

    MpcRtiState_t outside = current;
    outside.plant.e_y = 0.60f;
    MpcRtiState_t rejected_states[PREDICTION_HORIZON + 1];
    int failure_stage = -2;
    check_true(mpc_rti_rollout_candidate(&outside, 0.0, nominal.controls,
        trajectory, trajectory_count, lap_length, NULL, NULL, NULL, horizon,
        0.025f, &config, rejected_states, NULL, &failure_stage) ==
            MPC_RTI_ROLLOUT_CORRIDOR,
        "predicted state outside hard corridor is rejected");
    check_true(failure_stage >= 0,
        "measured x0 is not mistaken for a controllable predicted corridor state");

    MpcRtiState_t near_edge = current;
    near_edge.plant.e_y = 0.56f;
    MpcModelControl_t zero_control[PREDICTION_HORIZON] = {{0}};
    check_true(mpc_rti_rollout_candidate(&near_edge, 0.0, zero_control,
        trajectory, trajectory_count, lap_length, NULL, NULL, NULL, 1,
        0.025f, &config, rejected_states, NULL, &failure_stage) ==
            MPC_RTI_ROLLOUT_CORRIDOR,
        "default inset rejects a first prediction closer than its margin");
    check_true(mpc_rti_rollout_candidate(&near_edge, 0.0, zero_control,
        trajectory, trajectory_count, lap_length, NULL, NULL, NULL, 1,
        0.025f, &first_step_relaxed, rejected_states, NULL, &failure_stage) ==
            MPC_RTI_ROLLOUT_OK,
        "diagnostic first-step margin permits safe raw-bound clearance only at x1");

    MpcRtiCycleConfiguration_t cycle_config = {
        .model = config,
        .solver = solver_config,
        .rti2_residual_recovery_limit = 0.25f,
        .degraded_residual_limit = 0.01f,
        .maximum_regularization = 1.0e-2f,
        .max_consecutive_degraded_solves = 1};
    MpcRtiMemory_t memory = {0};
    MpcRtiCycleResult_t cycle_result;
    const MpcRtiCycleStatus_t cycle_status = mpc_rti_solve_cycle(
        &current, 0.0, trajectory, trajectory_count, lap_length, 0.025f,
        horizon, &cycle_config, &memory, &cycle_result);
    check_true(cycle_status == MPC_RTI_CYCLE_ACCEPTED_OPTIMAL ||
               cycle_status == MPC_RTI_CYCLE_ACCEPTED_DEGRADED,
               "complete RTI cycle accepts only a solved, safe nonlinear candidate");
    if (cycle_status == MPC_RTI_CYCLE_ACCEPTED_OPTIMAL ||
        cycle_status == MPC_RTI_CYCLE_ACCEPTED_DEGRADED) {
        check_true(memory.nominal.valid && memory.nominal.horizon == horizon,
                   "accepted nonlinear candidate becomes next RTI nominal");
        const MpcStageResult_t first_command_step = mpc_vehicle_model_step(
            &current.plant, &cycle_result.first_control, 0.025f,
            cold_references[0].path_curvature);
        check_close(cycle_result.published_steering_command,
                    first_command_step.next.steering_command, 1.0e-7f,
                    "published steering is exactly the model's first-stage command");
        check_close(cycle_result.published_target_speed,
                    first_command_step.next.target_speed, 1.0e-7f,
                    "published target speed is exactly the model's first-stage command");
        check_true(cycle_result.rti_iterations_used == 1 &&
                   !cycle_result.rti2_triggered &&
                   cycle_result.selected_candidate == 1,
                   "R1 mode remains a single-pass baseline by default");

        MpcRtiCycleConfiguration_t forced_r2_config = cycle_config;
        forced_r2_config.refinement_mode = MPC_RTI_REFINEMENT_R2;
        MpcRtiMemory_t forced_r2_memory = {0};
        MpcRtiCycleResult_t forced_r2_result;
        const MpcRtiCycleStatus_t forced_r2_status = mpc_rti_solve_cycle(
            &current, 0.0, trajectory, trajectory_count, lap_length, 0.025f,
            horizon, &forced_r2_config, &forced_r2_memory,
            &forced_r2_result);
        if (forced_r2_status != MPC_RTI_CYCLE_ACCEPTED_OPTIMAL &&
            forced_r2_status != MPC_RTI_CYCLE_ACCEPTED_DEGRADED) {
            fprintf(stderr,
                "R2 debug: cycle=%d R1=%d R2=%d passes=%d iter=%d "
                "primal=%g dual=%g failure_stage=%d\n",
                forced_r2_status, forced_r2_result.r1_status,
                forced_r2_result.r2_status,
                forced_r2_result.rti_iterations_used,
                forced_r2_result.r2_solver_iterations,
                forced_r2_result.primal_residual,
                forced_r2_result.dual_residual,
                forced_r2_result.nonlinear_failure_stage);
        }
        check_true(forced_r2_status == MPC_RTI_CYCLE_ACCEPTED_OPTIMAL ||
                   forced_r2_status == MPC_RTI_CYCLE_ACCEPTED_DEGRADED,
                   "forced R2 cycle retains a valid feasible output");
        if (forced_r2_status == MPC_RTI_CYCLE_ACCEPTED_OPTIMAL ||
            forced_r2_status == MPC_RTI_CYCLE_ACCEPTED_DEGRADED) {
            check_true(forced_r2_result.rti_iterations_used == 2 &&
                       forced_r2_result.rti2_triggered &&
                       (forced_r2_result.r2_status ==
                            MPC_RTI_CYCLE_ACCEPTED_OPTIMAL ||
                        forced_r2_result.r2_status ==
                            MPC_RTI_CYCLE_ACCEPTED_DEGRADED),
                       "forced R2 executes a second accepted pass in the same cycle");
            check_true(forced_r2_result.rti2_trigger_reason_mask ==
                           (1u << 31),
                       "forced R2 is distinguishable from adaptive trigger reasons");
            check_true(forced_r2_result.selected_candidate == 1 ||
                       forced_r2_result.selected_candidate == 2,
                       "R1/R2 selector reports exactly one feasible candidate");
            check_close((float)forced_r2_memory.nominal.progress[0], 0.0f,
                        1.0e-8f,
                        "same-sample R2 does not shift the cycle start progress");
            check_close(forced_r2_result.published_steering_command,
                forced_r2_memory.nominal.states[1].plant.steering_command,
                1.0e-7f,
                "published command matches the selected R1/R2 nominal");
        }

        MpcRtiCycleConfiguration_t adaptive_config = cycle_config;
        adaptive_config.refinement_mode = MPC_RTI_REFINEMENT_ADAPTIVE;
        adaptive_config.rti2_progress_error_trigger_m = 1.0e6f;
        adaptive_config.rti2_curvature_error_trigger_per_m = 1.0e6f;
        adaptive_config.rti2_bound_error_trigger_m = 1.0e6f;
        adaptive_config.rti2_min_corridor_slack_trigger_m = 0.0f;
        adaptive_config.rti2_steering_rate_correction_trigger_radps = 1.0e6f;
        adaptive_config.rti2_target_speed_rate_correction_trigger_mps2 = 1.0e6f;
        adaptive_config.rti2_residual_imbalance_trigger = 1.0e6f;
        adaptive_config.rti2_lateral_load_trigger_mps2 = 0.0f;
        adaptive_config.rti2_nonsmooth_columns_trigger = 1000000;
        MpcRtiMemory_t adaptive_memory = {0};
        MpcRtiCycleResult_t adaptive_result;
        const MpcRtiCycleStatus_t adaptive_status = mpc_rti_solve_cycle(
            &current, 0.0, trajectory, trajectory_count, lap_length,
            0.025f, horizon, &adaptive_config, &adaptive_memory,
            &adaptive_result);
        check_true(adaptive_status == MPC_RTI_CYCLE_ACCEPTED_OPTIMAL ||
                   adaptive_status == MPC_RTI_CYCLE_ACCEPTED_DEGRADED,
                   "adaptive RTI cycle returns a feasible output");
        if (adaptive_status == MPC_RTI_CYCLE_ACCEPTED_OPTIMAL ||
            adaptive_status == MPC_RTI_CYCLE_ACCEPTED_DEGRADED) {
            check_true(adaptive_result.rti_iterations_used == 2 &&
                       adaptive_result.rti2_triggered,
                       "adaptive mode runs R2 when a configured trigger fires");
            check_true((adaptive_result.rti2_trigger_reason_mask &
                           MPC_RTI2_TRIGGER_LATERAL_LOAD) != 0,
                       "adaptive trigger telemetry identifies lateral load");
        }
    }
    mpc_rti_memory_reset(&memory);
    check_true(!memory.nominal.valid && !memory.solver_state.initialized,
               "RTI reset clears nominal and ADMM warm-start memory");
}

static void test_directional_corridor_schedule(void)
{
    MpcTrajectorySample_t trajectory[80];
    size_t trajectory_count = 0;
    double lap_length = 0.0;
    make_circle_trajectory(trajectory, &trajectory_count, &lap_length);
    MpcRtiConfiguration_t config = test_configuration();
    config.corridor_preview_halfwidth_m = 0.0f;

    const MpcRtiState_t current = {
        .plant = {.e_y = 0.60f, .e_psi = 0.0f, .u = 4.0f, .v = 0.0f,
                  .r = 0.4f, .target_speed = 3.9f,
                  .steering_command = atanf(0.1f /
                      MPC_YAW_RATE_STEERING_GAIN_PER_M)},
        .previous_steering_rate = 0.0f,
        .previous_target_speed_rate = 0.0f};
    MpcRtiNominal_t nominal;
    MpcRtiReference_t references[PREDICTION_HORIZON + 1];
    check_true(mpc_rti_build_nominal(&current, 0.0, NULL, trajectory,
        trajectory_count, lap_length, 0.025f, 4, &config, &nominal,
        references), "recovery schedule receives a valid nominal");

    /* Isolate the schedule policy from model fitting: this is a deterministic
     * bounded seed that is outside the inset for one stage and re-enters at
     * the next stage. */
    nominal.states[0].plant.e_y = 0.60f;
    nominal.states[1].plant.e_y = 0.58f;
    nominal.states[2].plant.e_y = 0.54f;
    nominal.states[3].plant.e_y = 0.20f;
    nominal.states[4].plant.e_y = 0.10f;
    for (int k = 0; k <= 4; ++k) nominal.progress[k] = 0.10 * k;

    MpcRtiCorridorSchedule_t schedule;
    check_true(mpc_rti_build_corridor_schedule(&current, &nominal,
        trajectory, trajectory_count, lap_length, 0.025f, &config, &schedule),
        "directional recovery schedule builds");
    check_true(schedule.recovery_active && !schedule.recovery_not_found,
        "schedule activates only for a recoverable seed");
    check_true(schedule.first_normal_feasible_stage == 2,
        "schedule records the first normal-feasible stage");
    check_true(schedule.active_upper[1] > schedule.normal_upper[1] &&
               schedule.active_lower[1] == schedule.normal_lower[1],
        "schedule widens only the violated directional side");
    check_close(schedule.active_upper[2], schedule.normal_upper[2], 1.0e-7f,
        "schedule restores the normal upper bound at re-entry");
    check_close(schedule.active_lower[2], schedule.normal_lower[2], 1.0e-7f,
        "schedule restores the normal lower bound at re-entry");
    check_true(schedule.active_upper[1] >= schedule.seed_e_y[1] &&
               schedule.active_lower[1] <= schedule.seed_e_y[1],
        "temporary envelope contains the bounded recovery seed");

    MpcRtiConfiguration_t feedback_config = config;
    feedback_config.recovery_seed_policy =
        MPC_RTI_RECOVERY_SEED_HEADING_FEEDBACK;
    feedback_config.recovery_steering_k_e_y = 1.0f;
    feedback_config.recovery_steering_k_e_psi = 1.0f;
    feedback_config.recovery_steering_k_r = 0.1f;
    MpcRtiCorridorSchedule_t feedback_schedule;
    check_true(mpc_rti_build_corridor_schedule(&current, &nominal,
        trajectory, trajectory_count, lap_length, 0.025f, &feedback_config,
        &feedback_schedule),
        "heading-feedback recovery seed builds deterministically");
    check_true(feedback_schedule.horizon == nominal.horizon &&
               (feedback_schedule.recovery_active ||
                feedback_schedule.recovery_not_found ||
                feedback_schedule.max_seed_violation <= 1.0e-6f),
        "heading-feedback recovery reports a bounded schedule outcome");
    check_true(fabsf(feedback_schedule.seed_e_y[1] - schedule.seed_e_y[1]) >
                   1.0e-5f,
               "heading-feedback policy changes only the recovery seed");

    MpcRtiProblem_t normal_problem;
    MpcRtiProblem_t scheduled_problem;
    check_true(mpc_rti_build_ltv_qp(nominal.states, nominal.controls,
        references, 4, 0.025f, &config, &normal_problem),
        "normal parity QP builds");
    check_true(mpc_rti_build_ltv_qp_with_schedule(nominal.states,
        nominal.controls, references, 4, 0.025f, &config, &schedule,
        &scheduled_problem), "scheduled QP builds");
    check_close(scheduled_problem.steps[1].x_ub[MPC_RTI_IDX_EY],
        schedule.active_upper[1], 1.0e-7f,
        "QP consumes the scheduled active upper bound");
    check_close(scheduled_problem.steps[2].x_ub[MPC_RTI_IDX_EY],
        normal_problem.steps[2].x_ub[MPC_RTI_IDX_EY], 1.0e-7f,
        "QP restores normal bounds after re-entry");
    check_close(scheduled_problem.steps[1].u_lb[0],
        normal_problem.steps[1].u_lb[0], 1.0e-7f,
        "recovery does not relax the steering-rate actuator limit");
    check_close(scheduled_problem.steps[1].u_ub[1],
        normal_problem.steps[1].u_ub[1], 1.0e-7f,
        "recovery does not relax the speed-rate actuator limit");
    check_close(scheduled_problem.x0[MPC_RTI_IDX_EY], current.plant.e_y,
        1.0e-7f, "measured x0 remains fixed and is never projected");

    MpcRtiNominal_t normal_nominal = nominal;
    for (int k = 0; k <= 4; ++k) normal_nominal.states[k].plant.e_y = 0.0f;
    MpcRtiCorridorSchedule_t normal_schedule;
    check_true(mpc_rti_build_corridor_schedule(&current, &normal_nominal,
        trajectory, trajectory_count, lap_length, 0.025f, &config,
        &normal_schedule),
        "normal schedule builds");
    check_true(!normal_schedule.recovery_active &&
               !normal_schedule.recovery_not_found,
        "normal schedule is a no-op");
    check_close(normal_schedule.active_upper[1],
        normal_schedule.normal_upper[1], 1.0e-7f,
        "normal schedule preserves exact normal upper bound");
}

int main(void)
{
    test_absolute_nine_state_affine_ltv_build();
    test_previous_control_penalty_matrix_algebra();
    test_invalid_problem_rejected();
    test_two_pass_nominal_qp_and_nonlinear_candidate();
    test_directional_corridor_schedule();
    if (failures) {
        fprintf(stderr, "%d MPC RTI QP checks failed\n", failures);
        return 1;
    }
    puts("MPC RTI QP tests passed");
    return 0;
}
