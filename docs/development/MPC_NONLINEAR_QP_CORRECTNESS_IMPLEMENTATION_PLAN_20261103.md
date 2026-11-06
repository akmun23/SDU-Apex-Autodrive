# MPC nonlinear QP correctness — implementation and release plan
Date: 2026-11-03
Repository: akmun23/SDU-Apex-Autodrive
Priority: correct the *mathematical QP, nonlinear SQP and source-model
successor state* before further Unity racing or performance tuning.

## Fundamental distinction / invariants

The real optimization problem is nonlinear:
    x[k+1] = F_source(x[k], u[k]; wheel/slip/actuator FIFO)
    c_footprint(x[k], s[k]) >= 0
    u[k] satisfies the real steering/throttle and slew limits.
SQP can still use a **quadratic** subproblem: its dynamics and corridor
constraints are *local approximations*. Finite-difference Jacobian parity,
small ADMM residuals and positive QP wall margin DO NOT certify a 750 ms
physically safe or indefinitely recoverable trajectory.

Never publish a command from an infeasible intermediate SQP iterate.
Never equate soft/elastic QP slack with physical clearance. The original
full-source rollout and actual Unity model/geometry remain the acceptance
authorities. Keep two SQP passes. Preserve existing stable practice configs.
Never hide unrecoverable source states with brake/shutdown counts.

## Baseline evidence to preserve

- 2026-11-02 high-curvature first-order derivative audit: 2,772 checks per
  perturbation scale, max scaled-relative discrepancy 0.277% at the smallest
  perturbation. Local derivatives plausible; finite-step validity unproven.
- Full-scale strict OSQP QP in source-captured case 6: e_y error 1.702 m,
  e_psi 0.529 rad, progress 0.455 m, yaw rate 9.27 rad/s; mathematically
  solved local QPs can be source-corridor invalid.
- Twelve as-captured/strict OSQP solved QPs from case 2/6/10 all failed
  exact 750 ms corridor; cannot fix by ADMM-only iteration count.
- Strict competition and practice live failures sometimes happen at stage 0
  AFTER accepted future horizons lose clearance over preceding epochs.
  This is recursive feasibility and source/Unity uncertainty as well as SQP.
- Novel source-gradient/curvature and progress-sensitive hard geometry are
  currently opt-in and **not race qualified**. Latest full package tests
  previously: 336 passed / 0 failed. No complete repeatable race evidence.
- Source/Unity yaw and rear-wheel-speed differences at some high-slip epochs
  mean source-only optimization is not a Unity fidelity certificate.
- Never replace a proper original source corridor with one diagnostic Python
  approximation: Python and C++ had a documented ~19.7mm difference at one
  near-wall failure.

## Phase 0: establish reproducible correctness harness (first deliverable)

Record exact command epoch, all source hidden wheel/suspension states and
actuator queues, QP objective / affine dynamics / hard footprint planes,
ADMM warm state, solver stop reason, first failure stage and physical-unit
error. Reconstruct *identical* raw QP in independent OSQP (state/input
coordinate transforms and float precision included). Exercise all known
failure fixtures, not one selected "good" case.

Measure three distinct things:
(1) raw QP mathematical feasibility, station-dependent footprint and KKT;
(2) nonlinear full-source rollout versus the QP-predicted rollout;
(3) physical source versus synchronized Unity transition and corridor.
Report stagewise, channelwise absolute errors (m, rad, rad/s, m/s) and
QP-predicted vs **same frozen objective** exact source reduction. Track
source violation V = sum_i,k max(0, -clearance_i,k)^2 *separately* from
racing cost. Compare both invalid and valid iterations.

GATE 0: QP/OSQP parity is reproducible for frozen captures, and every
recorded failure is classified (may have multiple causes). Do not claim
mathematical feasibility merely because a projection variable z is feasible
when the raw dynamics x violates its constraints.

## Phase 1: finite-step source/geometry model agreement (no race tests)

Use full source transition F, source sensitivities and the ACTUAL SQP
directions (x_qp[k]-x_nominal[k], u_qp[k]-u_nominal[k]), not only
one-at-a-time perturbations. Compare first-order, directional second-order
finite differences, exact finite-step and source-corrected/relinearized
predictions. Sweep alpha in {1,0.5,0.25,0.125,0.0625}.
Measure physical state errors, nonlinear *constraint violation* and
same-objective cost agreement. Identify contact/tire branch transitions
individually. Reuse the existing single-step finite-difference audit but
do NOT substitute it for actual-direction agreement.

GATE 1: a selected curvature treatment demonstrably decreases source
prediction and corridor error on failing proposed directions without
unbounded outliers; otherwise retain source-exact relinearization in R2.

## Phase 2: implement a unified feasible-start AND infeasible-start SQP

Introduce explicit elastic variables / mathematically equivalent exact
positive-part squared-slack penalties for the *local linearized* corridor
constraints, without silently shifting strict physical walls by a fixed
0.18m. Use a filter/merit strategy: if starting source trajectory is
invalid, accept a nonpublishable intermediate ONLY if exact nonlinear
V decreases; once it is valid, prefer strictly feasible objective
improvement. Keep all physical actuator FIFO and command constraints.
Reintegrate the accepted source iterate and recompute the QP Jacobians
for R2. Rejected intermediate plans may NEVER be published.

Do not overwrite the hard physical validation. Prove the embedded
elastic objective, solver projection and independent OSQP slack oracle
match. Avoid escalating weights until the true SQP termination conditions
are correct.

GATE 2: from every known *recoverable* frozen failure, decrease exact
source V and obtain a strictly source-feasible full horizon via the normal
SQP, WITHOUT a fallback policy. For unrecoverable command-start states,
diagnose infeasibility correctly, never claim a feasible QP publication.

## Phase 3: source-informed second-order R1/R2 model corrections

Retain two SQP/relinearization passes. Investigate regularized Lagrangian
Hessian versus source-derived second-order correction (SOC):
r_k = F_source(x_trial[k],u_trial[k]) -
      (A_k*x_trial[k]+B_k*u_trial[k]+d_k).
Introduce defect-corrected second subproblem or source-exact relinearized
R2 that actually reduces this measured mismatch. Use PSD stabilization
where necessary; keep non-smooth tire/contact transitions exact-source
rather than forcing a global polynomial. Never turn off hard validator.

GATE 3: smaller finite-step model/cost error and more exactly feasible
selected trajectories on both source practice/competition frozen corpora,
including previously alpha-zero and no-R2 failures. Record outliers.

## Phase 4: recursive feasibility / first action validity

After accepting a 750ms plan, use the *actual queued first action* and
exact 33-state model to verify at least one viable full-horizon successor
at the next controller epoch. Extend the constrained terminal region with
a source-optimized continuation, not a 3-policy held-command proxy. Use
measured source-versus-Unity discrepancy as uncertainty allowance
particularly near <1cm wall clearance. A nominal accepted plan is not
itself a robust predecessor certificate.

GATE 4: recorded progressive clearance-collapse sequences no longer
reach a stage-0 no-continuation frontier from a previously accepted normal
plan. This is an empirical source/Unity validation, not formal proof
of infinite-horizon recursive feasibility.

## Phase 5: consolidation and deployment gates

Only after Phases 0–4 pass, remove obsolete independent Phase-I, gradient
search, repeated restoration and fallback *normal operating paths*.
Preserve a narrow nonpublishing safety mechanism and actual collision
detection. Rebuild and run the entire numerical test suite after each
replacement. First source fixture corpus, then synchronized Unity closed
loop on practice and competition maps; finally 5 independent runs with
10 scored laps each, zero collisions, no unexplained full-brake or
unsafe commands, and a source/Unity fidelity report. Only then enforce
hard 20ms callback deadline and target 15–17ms maximum.

## Implementation files

- f1tenth_mpc/src/mpc_rti_qp.cpp + mpc_rti.cpp:
  mathematical objective, hard/elastic footprints, SQP filter/SOC, R1/R2.
- f1tenth_mpc/src/mpc_source_nonlinear_optimizer.cpp,
  mpc_source_adjoint.cpp: exact source violation and gradient.
- f1tenth_mpc/src/mpc_successor_viability.cpp:
  feasible successor with actuator FIFO.
- f1tenth_mpc/src/riccati_*.cpp:
  full QP constraint projection, KKT and scaling.
- tools/mpc_live_failure_replay.cpp,
  tools/mpc_qp_oracle.py, tools/audit_mpc_osqp_source_fidelity.py:
  raw QP, source defect, constraint/cost agreement and failure corpus.
- test/test_mpc_sqp_convergence.cpp,
  test/test_mpc_corridor_polygon.cpp,
  test/test_mpc_qp_fixture_roundtrip.cpp:
  deterministic numerical and strict nonpublication regressions.

## Status as of plan creation

Phase 0 partially implemented; many independent QP/OSQP and physical
source-corner audits exist. Phase 1 has **individual-direction** second-order
finite-difference evidence, NOT QP-direction correction. Phases 2–4 have
experimental components but NONE satisfies the gates above. Production
configurations must remain unpromoted until these gates pass.
