/*
 * Native AutoDRIVE MPC benchmark.
 *
 * This is an offline, deterministic testbench. It intentionally uses only
 * the public MPC API and a small synthetic Frenet plant; no simulator or
 * ground-truth ROS topic is involved. Its purpose is to compare weight
 * profiles at the actual 10 Hz control interval before any profile is used in
 * the competition controller.
 */

#include "mpc.h"
#include "mpc_types.h"
#include "util_math.h"

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

typedef struct
{
    const char *name;
    float curvature;
    float velocity_start;
    float velocity_end;
    float initial_ey;
    float initial_epsi;
    float initial_vx;
    float left_wall;
    float right_wall;
} BenchmarkScenario;

typedef struct
{
    const char *name;
    float lateral_scale;
    float heading_scale;
    float velocity_scale;
    float steering_rate_scale;
    float acceleration_rate_scale;
} WeightProfile;

typedef struct
{
    double score;
    double mean_abs_ey;
    double max_abs_ey;
    double mean_abs_speed_error;
    double max_abs_speed_error;
    double max_solve_us;
    unsigned long solver_failures;
    unsigned long max_iteration_calls;
    unsigned long wall_violations;
} BenchmarkResult;

static double elapsed_us(const struct timespec *start, const struct timespec *end)
{
    return (double)(end->tv_sec - start->tv_sec) * 1.0e6 +
           (double)(end->tv_nsec - start->tv_nsec) / 1.0e3;
}

static void make_reference(
    TrajectoryReferencePoint_t reference[PREDICTION_HORIZON],
    float curvature,
    float velocity,
    float left_wall,
    float right_wall)
{
    for (int i = 0; i < PREDICTION_HORIZON; ++i) {
        memset(&reference[i], 0, sizeof(reference[i]));
        reference[i].reference_velocity = velocity;
        reference[i].reference_yaw_rate = curvature * velocity;
        reference[i].path_curvature = curvature;
        reference[i].left_wall_bound = left_wall;
        reference[i].right_wall_bound = right_wall;
    }
}

static void synthetic_frenet_step(
    FrenetState_t *state,
    const ControlInput_t *control,
    double *effective_steering,
    double *effective_acceleration,
    float curvature,
    float velocity_reference)
{
    const double dt = (double)CONTROL_DT_SECONDS;
    const double wheelbase = (double)VP_WHEELBASE_M;
    const double tau_steering = 0.060;
    const double tau_yaw = 0.080;
    const double vx = fmax((double)state->flong_vel, (double)VP_MIN_VELOCITY_MPS);
    const double vy = (double)state->flat_vel;
    const double epsi = (double)state->fhead_error;
    const double ey = (double)state->flat_error;
    const double denom = fmax(0.5, 1.0 - (double)curvature * ey);

    /* A deliberately simple plant captures command delay and yaw response;
     * the controller's full dynamic bicycle model remains the object under
     * test. */
    const double steering_alpha = fmin(1.0, dt / tau_steering);
    *effective_steering += steering_alpha *
        ((double)control->steer_ang - *effective_steering);

    const double yaw_target = vx * tan(*effective_steering) / wheelbase;
    const double yaw_alpha = fmin(1.0, dt / tau_yaw);
    const double yaw_rate = (double)state->fyaw_rate +
        yaw_alpha * (yaw_target - (double)state->fyaw_rate);
    const double ey_dot = vx * sin(epsi) + vy * cos(epsi);
    const double epsi_dot = yaw_rate -
        (double)curvature * vx * cos(epsi) / denom;
    const double acceleration_tau =
        (double)LONGITUDINAL_ACCEL_EFFECTIVE_TIME_CONSTANT_SECONDS;
    const double acceleration_retention = exp(-dt / acceleration_tau);
    const double acceleration_command_gain = 1.0 - acceleration_retention;
    const double acceleration_average_effective_gain =
        acceleration_tau * acceleration_command_gain / dt;
    const double accel = util_clamp(
        acceleration_average_effective_gain * (*effective_acceleration) +
        (1.0 - acceleration_average_effective_gain) *
            (double)control->long_acc,
        (double)VP_MIN_ACCEL_MPS2,
        (double)VP_MAX_ACCEL_MPS2);
    *effective_acceleration = acceleration_retention * (*effective_acceleration) +
        acceleration_command_gain * (double)control->long_acc;
    const double next_vx = util_clamp(
        vx + dt * accel,
        (double)VP_MIN_VELOCITY_MPS,
        fmax((double)velocity_reference + 1.0, 6.0));

    state->flat_error = (float)(ey + dt * ey_dot);
    state->fhead_error = util_normalize_angle((float)(epsi + dt * epsi_dot));
    state->flong_vel = (float)next_vx;
    state->flat_vel = (float)(0.85 * (double)state->flat_vel);
    state->fyaw_rate = (float)yaw_rate;
}

static BenchmarkResult run_case(const MpcConfiguration_t *configuration)
{
    BenchmarkResult result;
    memset(&result, 0, sizeof(result));

    const BenchmarkScenario scenarios[] = {
        {"straight",     0.00f, 2.5f, 3.5f,  0.28f,  0.08f, 2.0f, 0.85f, 0.85f},
        {"left_curve",   0.12f, 3.0f, 3.5f,  0.18f,  0.06f, 2.5f, 0.75f, 0.75f},
        {"right_curve", -0.16f, 3.0f, 3.3f, -0.18f, -0.06f, 2.5f, 0.75f, 0.75f},
        {"speed_step",   0.04f, 1.5f, 3.5f,  0.15f,  0.04f, 1.5f, 0.90f, 0.90f},
    };
    const unsigned long steps_per_case = 180UL;
    const unsigned long warmup_steps = 20UL;
    unsigned long valid_samples = 0UL;

    for (size_t scenario = 0; scenario < sizeof(scenarios) / sizeof(scenarios[0]); ++scenario) {
        mpc_initialize_with_configuration(configuration);

        FrenetState_t state = {
            .flat_error = scenarios[scenario].initial_ey,
            .fhead_error = scenarios[scenario].initial_epsi,
            .flong_vel = scenarios[scenario].initial_vx,
            .flat_vel = 0.0f,
            .fyaw_rate = 0.0f,
        };
        ControlInput_t previous = {0.0f, 0.0f};
        double effective_steering = 0.0;
        double effective_acceleration = 0.0;
        TrajectoryReferencePoint_t reference[PREDICTION_HORIZON];

        for (unsigned long step = 0; step < steps_per_case; ++step) {
            const float transition = fminf(1.0f, (float)step / 60.0f);
            const float velocity = scenarios[scenario].velocity_start +
                transition * (scenarios[scenario].velocity_end -
                              scenarios[scenario].velocity_start);
            make_reference(reference,
                           scenarios[scenario].curvature,
                           velocity,
                           scenarios[scenario].left_wall,
                           scenarios[scenario].right_wall);

            MpcSolverResult_t solve;
            struct timespec start;
            struct timespec end;
            mpc_set_previous_command(&previous);
            clock_gettime(CLOCK_MONOTONIC, &start);
            const MpcSolverStatus_t status = mpc_compute_optimal_control(
                &state, reference, &solve);
            clock_gettime(CLOCK_MONOTONIC, &end);
            const double solve_time = elapsed_us(&start, &end);
            if (solve_time > result.max_solve_us) result.max_solve_us = solve_time;

            if ((status != MPC_STATUS_SUCCESS &&
                 status != MPC_STATUS_MAXIMUM_ITERATIONS_REACHED) ||
                !isfinite(solve.optimal_control.steer_ang) ||
                !isfinite(solve.optimal_control.long_acc)) {
                result.solver_failures++;
                break;
            }
            if (status == MPC_STATUS_MAXIMUM_ITERATIONS_REACHED) {
                result.max_iteration_calls++;
            }

            if (step >= warmup_steps) {
                const double speed_error =
                    (double)state.flong_vel - (double)velocity;
                const double abs_ey = fabs((double)state.flat_error);
                const double abs_speed_error = fabs(speed_error);
                result.score += 100.0 * abs_ey * abs_ey +
                    8.0 * (double)state.fhead_error * (double)state.fhead_error +
                    abs_speed_error * abs_speed_error +
                    0.02 * (double)solve.optimal_control.long_acc *
                    (double)solve.optimal_control.long_acc;
                if (abs_ey > fmin((double)scenarios[scenario].left_wall,
                                  (double)scenarios[scenario].right_wall)) {
                    result.wall_violations++;
                    result.score += 10000.0;
                }
                result.mean_abs_ey += abs_ey;
                result.mean_abs_speed_error += abs_speed_error;
                if (abs_ey > result.max_abs_ey) result.max_abs_ey = abs_ey;
                if (abs_speed_error > result.max_abs_speed_error)
                    result.max_abs_speed_error = abs_speed_error;
                ++valid_samples;
            }

            previous = solve.optimal_control;
            synthetic_frenet_step(
                &state, &solve.optimal_control, &effective_steering,
                &effective_acceleration,
                scenarios[scenario].curvature, velocity);
        }
    }

    if (valid_samples > 0UL) {
        result.mean_abs_ey /= (double)valid_samples;
        result.mean_abs_speed_error /= (double)valid_samples;
        result.score /= (double)valid_samples;
    } else {
        result.score = HUGE_VAL;
    }
    return result;
}

int main(void)
{
    const WeightProfile profiles[] = {
        {"baseline", 1.00f, 1.00f, 1.00f, 1.00f, 1.00f},
        {"fast", 0.75f, 0.75f, 1.25f, 0.50f, 0.75f},
        {"precise", 1.50f, 1.50f, 0.85f, 0.80f, 1.00f},
        {"smooth", 1.25f, 1.25f, 0.80f, 2.00f, 2.00f},
        {"speed", 1.00f, 1.00f, 1.50f, 1.00f, 0.75f},
    };
    const size_t profile_count = sizeof(profiles) / sizeof(profiles[0]);

    double best_score = HUGE_VAL;
    size_t best_profile = 0U;
    unsigned long total_failures = 0UL;

    const int single_candidate = getenv("MPC_BENCHMARK_SINGLE") != NULL &&
        atoi(getenv("MPC_BENCHMARK_SINGLE")) != 0;

    printf("profile,score,mean_abs_ey_m,max_abs_ey_m,mean_abs_speed_error_mps,"
           "max_abs_speed_error_mps,max_solve_us,solver_failures,max_iteration_calls,wall_violations\n");

    if (single_candidate) {
        MpcConfiguration_t configuration = get_default_configuration();
        BenchmarkResult result = run_case(&configuration);
        printf("candidate,%.6f,%.6f,%.6f,%.6f,%.6f,%.2f,%lu,%lu,%lu\n",
               result.score, result.mean_abs_ey, result.max_abs_ey,
               result.mean_abs_speed_error, result.max_abs_speed_error,
               result.max_solve_us, result.solver_failures,
               result.max_iteration_calls, result.wall_violations);
        return (result.solver_failures == 0UL && result.wall_violations == 0UL) ? 0 : 1;
    }

    for (size_t i = 0; i < profile_count; ++i) {
        MpcConfiguration_t configuration = get_default_configuration();
        configuration.weight_lateral_error *= profiles[i].lateral_scale;
        configuration.weight_heading_error *= profiles[i].heading_scale;
        configuration.weight_velocity *= profiles[i].velocity_scale;
        configuration.weight_steering_rate *= profiles[i].steering_rate_scale;
        configuration.weight_acceleration_rate *= profiles[i].acceleration_rate_scale;
        BenchmarkResult result = run_case(&configuration);
        total_failures += result.solver_failures;
        if (result.solver_failures == 0UL && result.score < best_score) {
            best_score = result.score;
            best_profile = i;
        }
        printf("%s,%.6f,%.6f,%.6f,%.6f,%.6f,%.2f,%lu,%lu,%lu\n",
               profiles[i].name, result.score, result.mean_abs_ey,
               result.max_abs_ey, result.mean_abs_speed_error,
               result.max_abs_speed_error, result.max_solve_us,
               result.solver_failures, result.max_iteration_calls,
               result.wall_violations);
    }

    if (total_failures != 0UL || best_score == HUGE_VAL) {
        fprintf(stderr, "MPC benchmark failed: no valid profile\n");
        return 1;
    }

    printf("best_profile=%s score=%.6f control_rate_hz=%.1f prediction_dt_s=%.3f horizon=%d\n",
           profiles[best_profile].name, best_score,
           (double)CONTROL_RATE_HZ, (double)TIME_STEP_SECONDS,
           PREDICTION_HORIZON);
    return 0;
}
