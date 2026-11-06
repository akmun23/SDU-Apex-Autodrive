# Infeasible-start elastic SQP: source and independent QP audit — 2026-11-03

## Purpose and release status

The project goal is a correctly solved QP whose proposed command sequence is **exactly feasible under the nonlinear source plant**, not a controller that hides QP failures behind fallback, wall expansion, or full braking. These are distinct requirements. This document records development experiments, not a qualified race controller. The experimental `source_violation_filter_sqp` and `source_directional_second_order_sqp` remain **disabled by default**. Do not delete established safety validation or promote the experimental configuration.

## Implemented this session

1. **Exact convex elastic corridor QP:** `mpc_qp_elastic_polygon.hpp` eliminates nonnegative wall slacks in the ADMM proximal objective as `(w/2) max(0, a*x-b)^2`. Original corridor halfplanes are unchanged. The independently reconstructed OSQP V7 fixture uses an *explicit nonnegative slack for each face at each stage*. The captured 30-stage full problem has 23 physical QP states and two controls.
2. **Corrected ADMM stage split and frozen capture:** updated native polygon projection, scalar/coordinate conditioning, float32 finite-difference Hessian safeguards, objective and source-gradient penalty accounting, and V7 QP serialization. Source/rollout hard validation is unchanged.
3. **Exact-source R1-to-R2 infeasible filter:** an otherwise nonpublishable R1 proposal may become the R2 nominal only when its 30-stage source-footprint squared violation decreases. This is a candidate *initializer*, never a certified command.
4. **Source-derived directional second-order R2 cost correction:** tests actual previous SQP direction using ±0.01 and ±0.02 full source rollouts, rejects inconsistent curvature and only inserts a convex positive missing-curvature correction. The effect on frozen failure cases is mixed; do not promote without a stronger accepted-state success measure.
5. **Experimental ADMM conditioning:** an elastic QP with weight 500 uses rho 127 and strict tolerance <=0.002, at least 300 ADMM iterations, and rescales warm-start dual variables to preserve unscaled Lagrange multipliers when moving between hard/elastic QPs. The legacy hard-QP settings are unchanged.
6. **Raw primal QP certificate:** for the opt-in mode, mathematical acceptance now checks raw model-coordinate x-z agreement, raw affine equality `x[k+1]=A*x[k]+B*u[k]+d`, hard state/input/terminal constraints and hard polygon planes. Elastic-wall shortfall is reported separately and is NOT a physical feasibility certificate. The threshold is provisional (0.002 on split/bounds, 0.001 dynamics) and is distinct from the untouched nonlinear source validation.

## Reproducible numerical evidence

- Full-horizon artificial elastic QP: native solver at tolerance 1e-5 took 154 iterations. Independent OSQP at 1e-9 absolute and relative tolerance produced max native versus OSQP state error **2.60423e-5** (various units), max control error **2.93522e-5**. Native/OSQP agree in this fully specified test; it does not measure nonlinear fidelity.
- For a *real* frozen competition R1 QP (case 6), explicit-slack OSQP solved the weight=100/500/1000/8000 versions, each with tiny numerical residuals. Nevertheless, the physical wall violation of the *soft optimum* was respectively **0.16526/0.15903/0.15212/0.10054 m**. Penalty alone lets the racing objective buy illegal clearance.
- A separate hard-wall feasibility LP with all affine dynamics and non-wall actuator/state constraints fixed found **minimum necessary wall slack = 0** on that frozen R1 problem. That means the *affine local QP* has a hard-wall-feasible solution, but the elastic racing objective selects an illegal optimum. It does **not** mean the exact nonlinear source plant has a feasible 750ms plan.
- Strict independent OSQP solved the original hard-wall QP for the same snapshot in 7,350 iterations at ~344ms (offline), with original polygon violation ~2e-12. Native hard Riccati-ADMM needed more than 1,200 iterations on tested rho values (7,32,64,127) and was not numerically certified; a better hard-QP method is required before promoting it.
- Frozen source/SQP comparison: **59/85** recorded cases were exactly parity-replayable; **26** failed historical raw-iterate replay parity under the modified working tree and must not be silently counted as successes or failures of this A/B. Source failure and Unity fidelity are separate, harder tests.
- In the 59 matched cases, ordinary R2 had status distributions: accepted optimal 1, accepted line-search 11, rejected nonlinear rollout 45, accepted feasible 2. The experimental default elastic R2 had accepted optimal 1, accepted line-search 10, rejected nonlinear rollout 41, rejected solver 7. It is not a reliable improvement.
- Of the 59 cases, the exact-source infeasible-intermediate filter selected a reduced-violation R1 iterate in **39**; median fractional decrease of the squared source-corridor violation across selected cases **85.65%**. Of ten cases with final exact nonlinear objective comparable between the two configurations, **2** improved and **8** worsened.
- The complete native build and **22/22 CTest suites** passed immediately *before* the final raw-primal equality-certificate addition. The three targeted \`MpcSqpElasticAdmm\` regression tests passed after that addition. A fresh full build and 22-suite run after the final certificate change remain necessary. These are NOT Unity closed-loop tests.

## Verified regression commands

```bash
cd ~/Documents/GitHub/SDU-Apex-Autodrive
cmake --build build/f1tenth_mpc -j8
(cd build/f1tenth_mpc && ctest --output-on-failure -j8)
python3 tools/verify_mpc_elastic_qp_full_horizon.py \
  --osqp-vendor /tmp/sdu-mpc-osqp-v104
python3 tools/audit_mpc_elastic_wall_feasibility.py \
  /tmp/mpc_frozen_elastic_qps_20261103/epoch_1793641756087085412_mode_14_r1.fqp
```

## Next correctness work (gates; no racing or timing optimization yet)

**P0 / Priority 1 — True hard-feasibility first.** The LP proves a feasible *affine* solution may exist even when the elastic *cost* optimum violates walls. Replace a single weighted racing+violation objective with a lexicographic/feasibility filter or properly constrained elastic Phase-I: minimize physical corridor slack first (retain all actuator bounds), then switch to original hard geometry once slack can reach zero. Do not simply raise w; this makes Riccati-ADMM poorly conditioned and does not guarantee zero slack. Recheck source feasibility after every integrated iterate.

**P0 / Priority 2 — Reconcile hard QP numerical convergence.** Diagnose why the existing hard original-corridor QP reaches exact OSQP optimum only after thousands of iterations and why finite-precision ADMM exhibits raw primal-versus-z mismatch. Test improved 3D polyhedral prox, active-set warm starts or an alternative sparse QP solver *offline* without committing to a runtime dependency. The new raw mathematical certificate must continue to reject invalid proposals.

**P0 / Priority 3 — Source-accurate second SQP.** Keep R2. Compare the exact source residual after each proposed first-order QP direction and use defect-corrected R2 relinearization/second-order correction that decreases **actual nonlinear V** and reaches **V=0**, instead of merely improving a local cost Taylor series. Do not force a smooth global Hessian through tire/contact switching branches.

**P0 / Priority 4 — Recursive feasibility.** Stage-0 infeasibility cannot be fixed by an optimization at that same epoch. Extend source-exact successor viability and terminal continuation so the first applied command is not allowed to drive into a no-recovery frontier at the next epoch. Correct source-versus-Unity yaw/wheel/suspension uncertainty before claiming small-margin robust feasibility.

**Release gates.** Exact objective/constraint parity in all frozen QPs, source-feasible full-horizon selected commands from each *recoverable* frozen failure, model versus Unity state alignment, both-track complete races, then five independent ten-lap runs. Only after correctness, revisit 20ms callback budget, fallback/search deletion and weights. No hard-wall validator may be replaced by elastic slack acceptance.

## Reproducibility cautions

- All source code changes are in the existing uncommitted working tree and should be reviewed before a new commit. Many earlier uncommitted changes are also present; do not overwrite them blindly.
- Two archived /tmp datasets and the OSQP vendor installation are local temporary diagnostics, not durable repository assets. The test script itself remains in `tools/`.
- This phase implemented a **correct soft mathematical QP**, not a proven robust nonlinear MPC. The major unsolved task is converting the reduced nonlinear violation into strict source-feasible solutions that can be used directly without fallback.
