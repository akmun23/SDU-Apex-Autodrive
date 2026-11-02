#ifndef MPC_STEERING_VARIATION_HPP
#define MPC_STEERING_VARIATION_HPP

#include <cmath>

/* Convex smooth absolute-value variation φ(z)=λ(√(ε²+z²)-ε).
 * Its tangent quadratic majorizer at z0 has coefficient
 * w=λ/(2√(ε²+z0²)), and therefore exactly the same local derivative.
 * The constant part is irrelevant to the QP, but not to exact merit.
 * Ordinary quadratic slew remains available independently. */
inline float mpc_steering_variation_quadratic_weight(
    float lambda, float epsilon, float nominal_correction_delta)
{
    if (lambda<=0.f)return 0.f;
    return lambda/(2.f*std::hypot(epsilon,nominal_correction_delta));
}
inline double mpc_steering_variation_exact_cost(
    double lambda, double epsilon, double correction_delta)
{
    if (lambda<=0.)return 0.;
    return lambda*correction_delta*correction_delta/
        (std::hypot(epsilon,correction_delta)+epsilon);
}
#endif
