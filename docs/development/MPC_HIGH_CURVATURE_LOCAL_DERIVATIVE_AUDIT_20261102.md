# High-curvature QP/source derivative audit — 2026-11-02

## Scope

This is a **read-only** inspection of the existing simulator trace and source-plant sensitivity implementation. No racing parameters, QP formulation, or live controller were modified. Checkout had substantial unrelated in-progress uncommitted changes; preserve them. Fixture: `live_runs/comp_gradient_nearwall_capture_20261102_01/run` (17 nearby captured QPs; this test samples 18 bag states).

## Commands

From repository root:

```bash
python3 tools/audit_mpc_source_corner_jacobians.py live_runs/comp_gradient_nearwall_capture_20261102_01/run --output /tmp/mpc_current_corner_jacobian_audit.csv --max-rows-per-bag 18
python3 tools/audit_mpc_source_corner_jacobians.py live_runs/comp_gradient_nearwall_capture_20261102_01/run --output /tmp/mpc_current_corner_jacobian_small.csv --max-rows-per-bag 18 --epsilon-scale 0.1
python3 tools/audit_mpc_source_corner_jacobians.py live_runs/comp_gradient_nearwall_capture_20261102_01/run --output /tmp/mpc_current_corner_jacobian_tiny.csv --max-rows-per-bag 18 --epsilon-scale 0.01
build/f1tenth_mpc/test_mpc_sqp_convergence --gtest_brief=1
build/f1tenth_mpc/test_mpc_qp_fixture_roundtrip --gtest_brief=1
```

## Results

All scales ran 2,772 analytic-vs-finite-difference source sensitivity comparisons (18 corner states). **The finite-difference perturbation size is not a source-vs-Unity test**: both paths use the same local source plant; it tests local differentiation and perturbation-scale nonlinearity only.

| FD epsilon multiplier | Maximum absolute derivative difference (mixed derivative units) | Maximum scaled-relative difference | Checks >10% relative | Checks >1% relative |
|---|---:|---:|---:|---:|
| 1 | 28.9417 | 4.6179 | 51 | 105 |
| 0.1 | 0.21575 | 0.0700 | 0 | 3 |
| 0.01 | 0.007556 | 0.002773 | 0 | 0 |

Largest baseline deviations are concentrated in the rear-left suspension jounce and pitch sensitivities, at steering roughly 0.38–0.49 rad and forward speeds about 2.4–3.4 m/s. Decreasing epsilon removes the overwhelming majority of the discrepancy: this is strong evidence of a **large-perturbation finite-step nonlinear response**, not proof that the infinitesimal source Jacobian is incorrect. Under the smallest tested scale, no tested derivative exceeded 1% scaled-relative error. This does **not** prove that QP finite steps, source/Unity prediction or QP Hessian assembly are correct.

Focused existing C++ tests: 22/22 SQP convergence tests and 2/2 QP fixture tests passed. The tests were run against existing build artifacts, not a freshly rebuilt binary after every uncommitted source modification.

## Priority next checks before modifying runtime

1. For each failed/rejected captured QP, independently compare the *same* QP against OSQP, including KKT residuals and solver iteration budgets, and break out state errors by **named state and units**. An existing stored analysis `/tmp/mpc_osqp_source_fidelity_final.jsonl` reports OSQP-solved QPs whose external controls still fail physical source rollout; this does not establish an ADMM implementation bug. The current default Python interpreter lacks OSQP, so an independent fresh OSQP run was not completed here.
2. Compute source nonlinear `f(x+dx,u+du)` versus first- and second-order finite-step predictions for **actual QP directions** at the corner captures; report lateral/heading/yaw/slip/wheel/suspension channels and physical footprint errors separately as functions of SQP alpha. Do not replace full nonlinear checks with a Jacobian agreement test.
3. Extract actual Unity transitions at matched command epochs and compare with full source transitions (not only their derivatives). Explicitly account for delayed actuator queue, corrected initial physical state, tire slip regime and time integration before attributing a residual to tire physics.
4. Examine whether the QP promotes controls that irreversibly exhaust feasible source successor states before `corridor@0`. Keep the hard physical validator; prioritize proactive successor feasibility in the normal constrained formulation once steps are mathematically verified.
5. Rebuild and repeat tests on the intended candidate commit and perform identical-state closed-loop A/B tests before enabling experimental changes.

**Disposition:** Evidence demonstrates source high-curvature *finite-step* nonlinearity; no conclusive source-vs-Unity mismatch or mathematical QP bug has yet been isolated. Do not claim the failures are solved.
