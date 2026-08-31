/* Regression tests for the AutoDRIVE asymptotic lateral tire model. */

#include "vehicle_model.h"

#include <assert.h>
#include <math.h>

int main(void)
{
    const float normal_load = 20.0f;
    const float force_limit = VP_LATERAL_ASYMPTOTE_VALUE * normal_load;
    float stiffness_zero = 0.0f;
    float force_zero = 0.0f;
    vehicle_model_compute_effective_lateral_stiffness(
        1u, normal_load, 0.0f, &stiffness_zero, &force_zero);

    assert(fabsf(force_zero) < 1.0e-6f);
    assert(stiffness_zero > 0.0f);

    float stiffness_small = 0.0f;
    float force_small = 0.0f;
    vehicle_model_compute_effective_lateral_stiffness(
        1u, normal_load, 0.001f, &stiffness_small, &force_small);
    assert(force_small > 0.0f);
    assert(stiffness_small <= stiffness_zero * 1.01f);

    float stiffness_large = 0.0f;
    float force_large = 0.0f;
    vehicle_model_compute_effective_lateral_stiffness(
        1u, normal_load, 1.0f, &stiffness_large, &force_large);
    assert(force_large > 0.0f);
    if (!(fabsf(force_large - force_limit) < 1.0e-5f)) return 1;
    assert(stiffness_large > 0.0f);

    float negative_force = 0.0f;
    vehicle_model_compute_effective_lateral_stiffness(
        1u, normal_load, -1.0f, &stiffness_large, &negative_force);
    assert(negative_force < 0.0f);
    if (!(fabsf(negative_force) - force_limit < 1.0e-5f)) return 1;

    return 0;
}
