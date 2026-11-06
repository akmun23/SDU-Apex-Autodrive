# Original-source curvature and accepted-SQP branch audit — 5 Nov 2026

**Status: not qualified as a new MPC formulation.** This report follows
the recently added `source_constraint_path_curvature` source-Jet2 physical,
track-projection and footprint derivative and compares it to the ORIGINAL
30-stage source integration. It does not change Unity/PhysX physics,
raceline, original racing cost, or physical hard inequalities.

## Validated implementation

- `source_constraint_path_curvature` chains second derivatives through
  source physical state, source body integration, implicit track projection,
  eight body-corner constraints and the 120 command-rate and 120 box rows.
- The independent source oracle can report the exact 480-row first and
  second derivatives along an explicitly supplied command direction.
- `tools/audit_mpc_source_linearization_domain.py` now reports the
  **first-order** and **second-order** predicted margins alongside exact
  source margins, and separately checks physical, peak and reference branch
  identity from the same original source model.
- Source-Jet2 first derivatives agree with the existing condensed source
  Jacobian: maximum error in the epoch-775 initial trial is
  **9.11e-8**. This agreement does not independently certify second-order
  accuracy or full nonlinear convergence.
- `tools/test_audit_mpc_source_linearization_domain.py` guards against
  certifying a second-order Taylor model when the original source model
  switches branches. `tools/test_source_sqp_regression.py` checks that
  accepted final primal states do not retain stale pre-step dual certificates.

## Quantitative example: recorded epoch 775, first full SQP candidate

| Fraction of recorded step | Actual minimum clearance | Affine minimum | Quadratic minimum | Worst affine error | Worst quadratic error |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.001 | -0.90497 m | -0.90498 m | -0.90497 m | 0.00001 m | ~0 m |
| 0.1 | -0.79486 m | -0.80966 m | -0.80302 m | 0.02700 m | 0.08654 m |
| 0.5 | -0.50890 m | -0.42450 m | -0.70677 m | 0.26438 m | 2.18991 m |
| 1.0 | -0.55368 m | +0.01036 m | -7.15552 m | 1.13934 m | 8.84117 m |

**The quadratic model is branch-local, not globally valid.** Source
substep schedules, combined-slip saturation, tire curve regions and
reference interpolation change during the candidate. The full-step
quadratic prediction is much worse than the affine model, even though
the differential is valid in an infinitesimal neighborhood of its
nominal trajectory. This must NOT be added unconditionally to the QP.

Other exact-source trials at a normalized initial model radius of 0.05
confirm the pattern: at epoch 765 the quadratic model can greatly
increase prediction error; at epoch 769 it sometimes improves it;
at epoch 775 the relative quality depends strongly on the current
branch and SQP iteration.

## Persistent BFGS curvature across accepted branch changes

Original source branch identity changes in many accepted SQP steps.
A strict reset was implemented and tested as an *offline-only* audit
option `MPC_SOURCE_NLP_STRICT_BRANCH_CURVATURE_GUARD=1`.
It is **disabled by default** after the regression comparison:

| 60-iteration recorded warm start | Baseline feasibility/clearance | Strict branch reset |
| --- | --- | --- |
| 765 | Feasible, +0.051606 m | Infeasible, -0.105688 m |
| 769 | Feasible, +0.000019 m | Infeasible, -0.091855 m |
| 775 | Infeasible, -0.273559 m | Infeasible, -0.169504 m |

The strict reset discards BFGS memory at *every* accepted epoch-765
step. A change of spline interpolation region alone need not imply
a discontinuity in the physical function or its first derivative.
Consequently a strict equality of branch labels is too strong a
criterion for persistent quasi-Newton curvature. It remains useful
only as a scientific diagnostic, not as a live-controller improvement.

Independently fixed a genuine KKT **certification** defect: when the
source optimizer accepts a new primal, it now clears dual multipliers
calculated at the *previous* primal. Fresh multipliers from a new
original-source linearization are required before an NLP KKT claim.
This does not by itself produce a better control trajectory.

## Remaining mathematical work

1. Classify the actual physical switching surfaces: truly discontinuous
   substep schedules and contact/actuator modes versus continuous but
   nonsmooth tire curve and reference interpolation branches. Numerically
   validate source values and one-sided derivatives **at each surface**.
2. Develop a hybrid/semismooth constraint derivative treatment for source
   switching. Do not silently smooth or substitute a different physics
   plant. A smooth Hessian at the initial branch is not enough for a
   finite step that traverses several different branches.
3. Keep the exact source nonlinear rollout as the independent safety and
   cost authority, and only use derivative models within their verified
   domain. A physically feasible candidate must still pass all original
   hard inequalities.
4. Requalify all 22 competition states and original constrained-NLP KKT
   conditions. Current independent IPOPT evidence remains 16/22
   physically feasible, **0/22 KKT-qualified**. No runtime solver
   substitution is justified.
5. Keep the source-only live controller unchanged until improvements pass
   the corpus and repeated Unity races. Timing is deferred by design.

## Reproduction

```bash
cmake --build build/f1tenth_mpc --target \
  mpc_source_nlp_oracle mpc_source_exact_offline_optimize mpc_controller_node -j4

python3 -m tools.audit_mpc_source_linearization_domain \
  --case /tmp/apex_phase1_full_compact/epoch_775 \
  --iteration 0 \
  --output artifacts/source_branch_guard_20261105/epoch775_curvature_domain.json

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m unittest \
  tools.test_audit_mpc_source_linearization_domain \
  tools.test_source_sqp_regression -v

ctest --test-dir build/f1tenth_mpc --output-on-failure
```

The `/tmp` fixture was produced from the frozen competition corpus
on the current development machine. The audit requires an actual
original-source context and a recorded QP primal trace. It never
constructs an alternative command output.
