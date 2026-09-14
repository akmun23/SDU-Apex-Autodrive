#include "mpc_observer.h"

#include <math.h>
#include <stdio.h>

static int failures = 0;

static void check_true(int condition, const char *message)
{
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", message);
        failures++;
    }
}

static MpcObserverMeasurement_t measurement(float stamp, float u, float r,
                                             float ax, float ay, float steering,
                                             float throttle)
{
    return (MpcObserverMeasurement_t){
        .stamp_s = stamp,
        .u_mps = u,
        .yaw_rate_radps = r,
        .longitudinal_accel_mps2 = ax,
        .lateral_accel_mps2 = ay,
        .steering_rad = steering,
        .applied_throttle_norm = throttle,
    };
}

static void test_observer_update_and_propagation(void)
{
    mpc_observer_initialize(NULL);
    MpcObserverState_t state;
    MpcObserverMeasurement_t first =
        measurement(1.000f, 5.0f, 0.5f, 0.4f, 2.0f, 0.10f, 0.40f);
    MpcObserverMeasurement_t second =
        measurement(1.025f, 5.2f, 0.6f, 0.8f, 2.2f, 0.12f, 0.60f);
    check_true(mpc_observer_update(
        &first, &state),
        "first source sample initializes observer");
    check_true(state.valid && isfinite(state.v_mps),
               "initialized state is finite");
    check_true(mpc_observer_update(
        &second, &state),
        "25 ms source sample is accepted");
    check_true(state.u_mps > 4.0f && state.u_mps < 5.5f &&
               state.r_radps == 0.6f,
               "legal measured u is corrected causally and r is source aligned");
    check_true(fabsf(state.v_mps) < 1.5f && state.q_drive > 0.4f,
               "dynamic states remain bounded and drive state responds");
    check_true(mpc_observer_propagate_to(1.050f, &state),
               "state propagates to future solve epoch");
    check_true(fabsf(state.state_age_s - 0.025f) < 1e-6f,
               "source state age is explicit");
}

static void test_timing_gate_and_cg_height_contract(void)
{
    mpc_observer_initialize(NULL);
    MpcObserverState_t state;
    MpcObserverMeasurement_t first =
        measurement(2.000f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    MpcObserverMeasurement_t delayed =
        measurement(2.060f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    mpc_observer_update(
        &first, &state);
    check_true(!mpc_observer_update(
        &delayed, &state),
        "source interval outside observer contract is flagged");
    check_true(state.timing_degraded, "timing degradation is retained in state");

    const MpcObserverConfiguration_t config =
        mpc_observer_default_configuration();
    check_true(config.max_abs_v_mps > 0.0f,
               "observer configuration has a finite lateral bound");
}

int main(void)
{
    test_observer_update_and_propagation();
    test_timing_gate_and_cg_height_contract();
    if (failures != 0) {
        fprintf(stderr, "%d MPC observer test(s) failed\n", failures);
        return 1;
    }
    puts("MPC observer tests passed");
    return 0;
}
