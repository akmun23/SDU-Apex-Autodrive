# Phase 1: one source nonlinear problem and a reproducible SQP gate

Date: 5 November 2026. Base commit: `a63e30608`.

Phase 1 is **not qualified**. A passing regression suite or a verified feasible
witness does not establish recovery from a recorded warm start, and physical
feasibility does not establish KKT convergence. Do not count the historical
replay tool's `passed` field as Phase 1 completion: it certifies replay parity,
determinism and preservation of known feasible results.

This plan follows the current controller implementation. The shared workspace
already contained the constraint-identity guard and rejected-gradient BFGS
experiment described in `MPC_POST_A63E_SQP_CURVATURE_20261105.md`; those changes
are preserved. No new numerical mode, alternate controller, witness seed or
multiple-shooting solver is introduced by this qualification work.

## 1.1 Authoritative problem and ownership

For one command epoch, fix the measured source state, actuator queues, complete
raceline, model configuration and original objective schedule. Optimize exactly
30 steering/throttle pairs. For `k=0,...,29`, eliminate states by

\[
X_{k+1}=f_{\rm source}(X_k,U_k),\qquad
\min_{U\in\mathbb R^{60}} J_{\rm race}(X(U),U).
\]

The original inequalities are 240 individual left/right body-corner clearances,
120 actuator-rate margins and 120 command-box margins. All margins use the
`g >= 0` convention. These constraints must be evaluated on the unmodified
source-integrated controls, with the original vehicle dimensions and source
collision margin. The additional preferred 0.03 m clearance is not the
physical wall. A negative physical clearance cannot qualify, even when its
magnitude is below a residual tolerance.

| Responsibility | Existing owner |
| --- | --- |
| Original nonlinear transition | `mpc_rti_source_exact_stage`, `vehicle_model.cpp`, `unity_source_plant.cpp` |
| Original nonlinear racing score | `mpc_rti_score_candidate_plan_and_corridor_for_diagnostics` |
| Original objective derivative | `mpc_source_exact_nlp_objective.cpp`, compact source adjoint |
| Original physical rows | `evaluate_existing_source_nominal`, `mpc_source_exact_compact.cpp` |
| Exact reduced control sensitivities | `StageJacobian`, condensed propagation in `mpc_source_exact_qp.cpp` |
| One convex local subproblem | `mpc_source_convex_qp.cpp` |
| Accepted nonlinear primal and publication eligibility | `mpc_source_exact_offline_solver.cpp`, existing controller validation |
| Independent scoring | standalone `mpc_source_nlp_oracle` |

The frozen footprint is 0.273 m wide, extends 0.080 m behind the rear axle
and 0.430 m ahead of it. Its source boundary includes the original 0.050 m
collision margin. Preferred extra clearance is measured beyond that boundary.

Actuator semantics also remain source-defined. `mpc_vehicle_model_step` applies
the current held stage command to the source plant and advances command-history
slots; it does not insert an additional queue delay into the 30-stage rollout.
The initial state is already aligned to the command epoch by the controller.
Steering-rate rows compare target to actual rack angle, while throttle-rate
rows compare current command to `delayed_throttle_command_2`. The legal driving
throttle interval is `[0.001, 1]`; exact zero selects a separate source brake
mode and cannot be used to create a Phase 1 feasibility witness.

The `Decision.nodes` arrays and sparse equality placeholders are oracle and
sensitivity storage. The runtime QP already eliminates their sensitivities;
they must not become independent optimization variables.

**Verified invariants:** source rollout bypasses the nominal seed builder's
rate projection; the exact supplied float controls are integrated. Physical
feasibility already requires `corridor >= 0`. Rate/box roundoff tolerances do
not soften that wall check. Frozen corpus reconstruction retains the initial
state, queues, original controls and all 31 objective reference samples.

**Open objective mismatch:** nonlinear acceptance and the QP still include an
additional finite preferred-clearance penalty absent from the frozen original
`J_race`. Source configuration may itself contain an original reserve cost;
that original term must remain. Remove only the added numerical reserve term
from the QP gradient, curvature, nonlinear acceptance and stopping logic as one
coherent change. Retain the preferred reserve as a separately reported quantity.
Do not silently relabel the augmented objective as the original racing cost.

This patch fixes the narrower diagnostic error: the added preferred reserve
no longer enters the **original NLP stationarity certificate**. That correction
does not make the remaining augmented numerical objective authoritative.

Acceptance proof for this step: independent cost/gradient parity on the frozen
corpus and legal inward control perturbations; identical original constraint
rows, geometry and actuator queues; no output with negative physical clearance.

## 1.2 Permanent qualification harness

`tools/qualify_mpc_source_phase1.py` builds on the existing portable 22-state
replay. It uses the production SQP with its research modes disabled. Primary
runs start from the recorded controls, have no wall-clock deadline and retain
the native iteration termination guard. Iteration exhaustion is not convergence.
At least two runs must agree after removing wall-clock fields.

Qualification traces retain controls and the frozen source context instead of
repeating every condensed state in JSON. The independent oracle reconstructs
those states; standard historical replay still retains its original node format.
This affects trace storage only, never numerical decisions.

Every iteration retains its exact warm start, trial/correction history and QP
dual estimate in original physical row units. A persistent independent source
oracle re-integrates accepted primals and evaluates the **original** objective
gradient and all 480 physical margins. It checks the complete accepted-primal
chain, including preservation of an already-feasible trajectory. Derivative or
QP failures are retained as iteration records rather than disappearing at an
early `break`.

Recorded quantities include physical violation `V`, maximum hard violation,
minimum physical clearance, original racing cost, QP primal/dual/complementarity
residuals, independent original NLP stationarity/complementarity/dual feasibility,
minimum/maximum stabilized QP Hessian eigenvalues, condition number, actual
accepted step norm and cumulative exact nonlinear trajectory evaluations.
Source AD calls are not relabeled as additional full primal trajectories.

Final source feasibility and final KKT are independently checked at the returned
controls; the last iteration's pre-step KKT residual is not reused as a final
certificate. Unavailable, nonfinite or negative dual estimates cannot establish
convergence. Results distinguish `feasibility_qualified`, `kkt_qualified` and
full `passed`. False solver convergence, missing evidence, nondeterminism or a
previously feasible case becoming infeasible prevents qualification.

The default run also tests two deterministic bounded perturbations around
recorded starts 765, 769, 773, 774 and 775. The perturbed float controls are
stored explicitly. They supplement the primary runs; they cannot replace a
failed recorded start. A complete pass requires those perturbations as well.

Comparison defaults to the immutable original frozen replay. A prior Phase 1
report may be supplied explicitly; its checksum is recorded and it must reside
outside the output directory. Solver, oracle and corpus checksums identify the
actual binaries and inputs. A failed result cannot freeze a new passing baseline.

## 1.3 QP conditioning

Control normalization already exists:

\[
\Delta U=D_u\Delta z,\qquad
D_u=\operatorname{diag}(0.15\ {\rm rad},0.25,\ldots).
\]

Therefore `g_z = D_u^T g_U`, `H_z = D_u^T H_U D_u`, and
`A_z = A_U D_u`. With positive row scale `R` and objective divisor `s`, the
scaled QP multiplier `mu` maps back as `lambda = R mu s`. Complementarity and
published multiplier estimates remain in original row/cost units. Do not
normalize elastic rows without transforming their epigraph variables and costs.

The existing predictor/corrector eliminated elastic variables by subtracting
large Schur terms. This patch instead forms a positive weighted covariance.
For one shared epigraph group, let

\[
q=\sum_i d_i,\quad \bar c=q^{-1}\sum_i d_i c_i,\quad
r=w+d_t.
\]

Then its Schur contribution is evaluated as

\[
\sum_i d_i(c_i-\bar c)(c_i-\bar c)^T
+\frac{qr}{q+r}\bar c\bar c^T,
\]

rather than `sum(d_i c_i c_i^T) - v v^T/(q+r)`. The right-hand side uses the
same decomposition. Symmetric diagonal equilibration and two residual-based
refinement steps improve the same Cholesky solve. No constraint is removed or
relaxed and the original residual termination remains authoritative.

Focused proofs cover independent stiff elastics, a shared epigraph with cross
control curvature, physical multiplier scaling, hard-wall dominance and
inconsistent hard rows. This algebraic improvement **does not solve an
inconsistent affine hard problem**, and it is not epoch 775 convergence proof.

Next: capture the failing 775 local QP, distinguish genuine affine
inconsistency from factorization error, inspect singular values of active
physical rows and verify a scaled KKT factorization against the original
residuals. Only remove a row after proving its physical redundancy. Do not
accept enormous finite multipliers as reliable source dual estimates.

## 1.4 Internal infeasible-start SQP

Use the original physical violation only:

\[
V(U)=\tfrac12\|\min(0,g_{\rm physical}(U))\|_2^2.
\]

The current default already accepts strict `V` reduction while infeasible;
it can nevertheless stagnate because the affine physical problem is
inconsistent or the local source branch prediction is inaccurate.

Implement an internal normal/elastic restoration step when the local hard QP
cannot be certified, followed by the tangential racing-objective step of the
same SQP. Elastic variables belong to that local subproblem; publication still
uses unrelaxed physical margins. During restoration, distinguish restoration
duals from estimates of original racing KKT multipliers. Do not feed arbitrary
uncertified QP multipliers into the Lagrangian memory.

Use a deterministic filter or a precisely specified restoration acceptance
rule with predicted and actual physical violation reduction. A racing-cost
increase must not veto a necessary feasibility improvement. Bound penalty
updates and distinguish a failed QP from an inaccurate nonlinear prediction.

Two temporary trials were discarded during this investigation: always using a
squared-elastic normal step while infeasible stalled 775 near -0.167 m and
regressed 768; attempting it only after an uncertified hard QP also regressed
770. Neither implementation is retained as a runtime mode. These failed trials
are not evidence of physical unrecoverability.

Required proof remains `V=0` with nonnegative physical clearance from the
original recorded 775 warm start, without a witness initialization.

## 1.5 Source-derived Lagrangian curvature

Retain the existing expensive full-Hessian calculation as an offline reference,
with fixed physical multiplier identities. Before optimizing it, verify
Hessian-vector products against differences of the original Lagrangian
gradient at legal, source-consistent controls. Record antisymmetry, raw
negative eigenvalues and the actual convexification separately.

The structured implementation belongs beside the source stage derivatives:
propagate first/second source tangents, accumulate adjoint-weighted transition
curvature, then include objective feature and corner-heading/progress curvature.
Use the existing source second-directional machinery where valid. Compare
resulting local QP directions and nonlinear recovery with the reference.

The original longitudinal-slip cost contains a maximum over source substeps.
At a maximum tie, a single selected peak derivative need not provide the
stationary subgradient even at a valid nonsmooth optimum. Consolidate the
existing exact local slip epigraph into the same SQP and return its active
substep multipliers. Independently verify their nonnegativity, zero weight on
inactive branches and correct sum against the original peak-cost derivative.
Do not substitute a smoothed maximum or require an arbitrary selected branch
gradient to vanish at a tie.

Source switching is a prerequisite, not a later optimization detail. Track
wheel-substep schedule changes at 5 m/s, exact-zero throttle's brake branch,
contact/sticky-tire and tire-slip branches, steering clamps, slip-maximum ties,
and progress/reference knot changes. Existing coarse source flags alone are
not proof that all those branches match. Select compatible one-sided source
derivatives or invalidate smooth curvature at a switch. Preserve the existing
constraint-identity guard for carried multipliers and Hessian memory.

At the discarded stalled 775 trial, selected throttle derivatives of `V`
differed materially from centered finite differences. That observation requires
branch/step-size investigation; it does not by itself prove an AD formula bug.

## 1.6 Step acceptance and stabilization

Always try the full certified SQP step first. Evaluate actual nonlinear source
constraints before acceptance. Use an adaptive, deterministic globalization
rule when needed; do not restore a fixed alpha grid. A second-order correction
must remain second order in the primary step and use its exact source defect.

Classify rejection explicitly: uncertified QP, failed source evaluation,
source branch change, inaccurate cost model, inaccurate constraint model or
true physical infeasibility. Reuse exact derivatives only while the accepted
primal and source problem remain identical. Keep accepted feasibility progress.
Update stabilization from actual/predicted reduction; bounded growth without
progress must report stagnation rather than consume iterations indefinitely.
A rounded-away control correction cannot establish KKT convergence.

## 1.7 Warm-start alignment

Preserve recorded initial queues and controls exactly. The controller already
rejects older command epochs and shifts warm starts through the command-epoch
contract. `mpc_rti_command_epoch_shift` rounds the signed elapsed epoch difference
to the nearest prediction stage; revisions within the same stage retain shift
zero, while differences at or below minus half a stage invalidate the shift.
Do not replace this with an unconditional one-stage shift. Extend tests around fractional epochs, queue handoff and source
suspension initialization before changing initialization policy. Compare the
first exact rollout with the independent oracle; report every initialization
transformation explicitly. Perturbed-start tests are now part of qualification.
A feasible witness may diagnose sensitivity, never seed the primary run.

## 1.8 Genuine nonlinear KKT

The normalized stationarity certificate is now

\[
\|D_u^T(\nabla_U J_{\rm race}-A_U^T\lambda)\|_\infty.
\]

One normalized steering unit is 0.15 rad and one throttle unit is 0.25.
The existing absolute threshold `1e-2` therefore bounds the local original
racing-cost derivative per those physical perturbation units; it is not
rescaled by the current cost or selected to force convergence. The independent
report also retains physical-coordinate stationarity for interpretation.
Complementarity uses original physical margins and multipliers. Physical wall
clearance must remain nonnegative regardless of the algebraic primal tolerance.

The harness verifies stationarity, complementarity, primal and dual feasibility
using the returned dual estimate at the actual final primal. It does not fit
new multipliers using an independent optimizer. Certification needs both the
independent residual checks and the native solver's KKT claim. Correcting the
certificate is distinct from making the SQP converge to it. The current
independent certificate uses the source-selected peak gradient. It is
conservative at a nonsmooth peak tie; the active-substep multiplier protocol
above remains necessary before claiming complete nonsmooth KKT qualification.

## 1.9 Release gate and remaining work

The complete gate requires all 22 primary recorded starts and the difficult
perturbed starts to recover strict source feasibility and original KKT, preserve
already-feasible accepted iterates, and pass independent physical checks. Keep
known witnesses as oracle regression inputs only. Archive original racing-cost
comparisons and per-iteration evidence for every change. Never infer physical
unrecoverability from an unsuccessful finite optimization run.

Remove superseded numerical research paths only after the single replacement
qualifies, while preserving source/oracle regression coverage. Do not change
live iteration/deadline settings, claim Unity safety or enable research modes
on the basis of an offline improvement. No new real-time qualification has
been performed here.

## Evidence from this implementation

- All six physical model/configuration source files match the frozen corpus
  SHA-256 checksums, including the footprint constants in `mpc_types.h`.
- The full build and all **15 CTest suites pass**, including the new fail-closed
  qualification tests, source/controller regressions and QP stress proofs.
- The complete 2,000-iteration gate reproduces **22/22** primary cases and
  **10/10** perturbed cases deterministically over two runs. **14/22** primary
  cases and **2/10** perturbations return independently source-feasible controls;
  **0/22** primary cases establish original nonlinear KKT. All previously
  validated feasible cases remain feasible. The 60-iteration primary run also
  found 14/22 feasible cases, so the larger budget did not resolve recovery.
- Epoch 775 ends with -0.302734911 m physical clearance and
  `V = 0.935710405`; it remains a failed recovery, not an unrecoverable state.
  The new oracle also exposes unstable stationarity estimates where uncertified
  QPs have enormous cancelling multipliers.
- The exact failed summary is archived in
  [`MPC_PHASE1_QUALIFICATION_20261105.json`](MPC_PHASE1_QUALIFICATION_20261105.json).
  Its `passed`, `feasibility_qualified` and `perturbations_qualified` fields are
  all false. Per-iteration traces from this run are in
  `/tmp/apex_phase1_full_compact/`; reproduce them with the command below.
- The additional preferred-reserve objective mismatch, reliable 775 recovery,
  branch-aware source curvature and full KKT convergence remain open.

## Reproduction

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
cmake -S f1tenth_mpc -B build/f1tenth_mpc
cmake --build build/f1tenth_mpc -j4
ctest --test-dir build/f1tenth_mpc --output-on-failure

# Full source qualification; exit 1 means solver qualification failed.
python3 -m tools.qualify_mpc_source_phase1 \
  --output artifacts/mpc_phase1 --iterations 2000 --repeat 2

# Compare a later build against that exact report without overwriting it.
python3 -m tools.qualify_mpc_source_phase1 \
  --output artifacts/mpc_phase1_next --iterations 2000 --repeat 2 \
  --baseline-report artifacts/mpc_phase1/phase1_report.json
```

A short smoke test may use `--iterations 1 --perturbations 0`. It exercises
trace/oracle correctness and is deliberately insufficient for Phase 1 passing.
