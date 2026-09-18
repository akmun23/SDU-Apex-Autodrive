# Phase 6 — fixed-rho Riccati prefactorization

The Riccati recursion now separates invariant quadratic setup from changing
ADMM linear terms. With `adaptive_rho=0`, the constrained quadratic terms are
factored once per QP; every later ADMM iteration updates only the linear
backward/forward recursion and the usual projection/dual residuals. The cold
unconstrained warm-seed pass deliberately stays on the reference implementation
so it does not create a throwaway factorization at `rho=0`.

Validation passed on both the development and untouched holdout traces:

- all 2,630 solves retained their status, iteration count, residuals, first
  actions, nonlinear trajectories, and minimum corridor clearance exactly;
- the N=30 full ADMM unit parity tests match the reference within `1e-5`, and
  the fixed-rho production path reports exactly one quadratic factorization per
  QP;
- the ROS package passes all 9 tests, and a separate `BUILD_TESTING=OFF`
  production build completes cleanly without finite-difference-oracle or
  profiling dependencies;
- paired 10,000-cycle O3 benchmarks show 0.4125 ms vs 0.1516 ms p95 for the
  Riccati-plus-projection solver core (2.72x faster). Whole RTI p95 was 0.541 ms
  vs 0.278 ms (1.95x faster).

The model coefficients were not changed. This is offline solver evidence, not
live controller acceptance. The optimization is enabled in the ROS MPC node
by default, with a `use_riccati_prefactorization` parameter for reference A/B.
Live shadow remains failed at the current acceptance gate; no MPC command
authority was enabled.
