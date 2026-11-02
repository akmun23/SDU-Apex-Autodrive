# Full source fidelity in production RTI QP — 2026-11-01

## Source geometry vs frozen exact objective

The Bridge odometry fix in MPC_BRIDGE_COMMAND_EPOCH_ACCURACY_20261101.md makes source actuator-command history accurate. The QP now uses this accurate full source state while also differentiating the same cost used by exact candidate evaluation.

In f1tenth_mpc/src/mpc_rti_qp.cpp, mpc_rti_build_ltv_qp_with_objective_schedule explicitly distinguishes:
- physical source nonlinear nominal and stage sensitivities, current path-frame Jacobians, current progress and physical corridor geometry;
- frozen cycle objective targets, steering correction increment references, slip penalties and speed-weight branches.

This entrypoint is now used by production solve_rti_pass and experimental mpc_rti_source_adjoint_gradient. Existing unfrozen wrapper is retained for independent QP callers. The prior source-adjoint cost mismatch from rebuilding targets at changed progress is corrected.

New regression tests prove: (a) when a frozen schedule comes from the same nominal, the complete 30-stage QP matrices are bit-for-bit unchanged, (b) deliberately altered frozen speed weights shift QP Hessian/gradient by the analytically expected values without changing A or geometric corridor bounds, and (c) source-adjoint gradients agree with independent exact-source finite differences under deliberately altered frozen targets/weights. All 21 MPC CTest targets pass.

Three difficult practice source-oracle cases, four iterations, exact final costs:
progress 23.86m: adjoint 103.26, finite difference 103.27;
progress 39.81m: adjoint 192.74, finite difference 192.45;
progress 50.86m: adjoint 279.96, finite difference 280.12.
All original hard-corridor valid in these offline exact-source optimization tests.

## Source-model branch boundary and globalization

In the new alternate-frozen-target gradient test, stage 21 throttle source adjoint is 1595.83, while exact finite differences yield 1595.62 at step +/-0.0001, but 1288.02 at +/-0.0003. This demonstrates a nonsmooth source tire/slip branch crossed by the wider finite difference, not a consistently wrong infinitesimal derivative. The regression explicitly checks the local derivative. Large QP steps must still be physically integrated and validated.

The first live run after the QP cost change, phase13_qp_frozen_cost_accuracy_20261101_01, aborted at progress 23.40m with nonlinear INVALID_MODEL despite a converged QP. The original code automatically skipped line search for INVALID_MODEL, even though the source nominal was fully valid.

The normal SQP globalization was fixed: if a full candidate is INVALID_MODEL but the nominal exact trajectory was valid, test the existing bounded intermediate steps using exact source integration, full hard corridor checks and the frozen merit. No invalid command/trajectory is accepted; other fault statuses keep their original behavior. This is not a new fallback policy and does not alter physics or MPC weights.

A subsequent unchanged-profile run, phase13_qp_model_domain_backtracking_20261101_01, completed 8 laps with zero reported collision or MPC rejection. This is a **single run only**, not proof of repeatability.

## Unresolved cases

At 227.063m (see live_runs/mpc_phase13_accuracy_qp_replay_20261101/corner227), the previous source nominal itself was corridor-invalid; ADMM reached 150 iterations with residual ~0.20 and its proposed trajectory crossed the physical corridor near stage 15. A backtrack from a physically invalid nominal may fail. This needs better SQP feasibility handling and nonlinear relinearization at a physically viable plan, not relaxation of the physical collision boundary.

No source-model timing/real-time worst-case, repeated full-race robustness or competition success is certified by these changes. Keep the exact-source safety checks and existing verified production configuration.

## 2026-11-01: same nonlinear objective across both SQP passes

A production error was confirmed: the second SQP pass built its own target and
speed-weight objective even though the controller is trying to improve the
same physical objective across consecutive nonlinear relinearizations.
About 90% of production cycles use two passes, so this was consequential.

R2 now uses the same immutable exact-source objective schedule as R1,
while independently recomputing its physical source nominal, source Jacobians,
progress, and live geometry. The obsolete per-pass objective-rebuild branch
and R2-only yaw-reference reweighting are DELETED, with a mandatory frozen
schedule invariant in the internal solve function. The public no-schedule
QP entrypoint remains for actual third-party test/replay callers.

The new regression proves the first-pass matrix is unchanged when the
schedule comes from the same nominal, and proves the independent nonlinear R2
state/Jacobians can change without changing the frozen target and speed Hessian.
All 21 CTests pass; the source-adjoint-vs-finite-difference 3-case corpus passes.

First 10-scored-lap race with frozen shared objective:
phase14_shared_merit_two_sqp_20261101_01: 49.104s, no reported collision
or solver rejection, one viability recovery, 2214/2214 exact Bridge command
matches, worst controller gate 18.330ms, cross-track RMSE 0.03837m.

## Failed five-race qualification and Riccati feedback finding

Prior series had multiple full 10-scored-lap completions but the fifth
phase13_qp_model_domain_backtracking_20261101_06 failed after five physical
laps at unwrapped progress 169.579m: exact rollout INVALID_MODEL.
The nominal itself was physically invalid, ADMM reached the 150-iteration
limit with primal residual ~0.084. Thus no five-consecutive-race qualification.

At this exact state, an offline reconstructed raw-QP control sequence
passes original full nonlinear corridor checks, but the feedback-modified
candidate differs by as much as 0.307rad steering / 0.296 throttle and
fails. Replay does not contain full original ADMM warm-state: mechanism
suggestive, not proof of identical live action.

Disable-feedback A/B completed all 10 scored laps in 49.862s but produced
seven viability recoveries, cross-track RMSE 0.04838m, max gate 18.263ms.
That is worse than the 49.089-49.153s feedback-enabled races. Do NOT remove
active Riccati feedback merely to simplify code; remove it only when a tested
physically consistent replacement retains useful performance.

## Sharp source yaw/slip mismatch (not merely stale state)

On the 49.104s race, 7 of 2184 750ms source/Unity windows exceed 5cm,
all overlapping the high-steering corner at progress ~240m; peak 0.08451m.
The first 250ms source yaw-rate differs by ~0.20-0.27rad/s from Unity
during sharp tire slip even though the final yaw-rate error is ~0.00045rad/s.
Integrated heading error at the endpoint is ~0.042rad. One-step errors peak
near 0.070rad/s in this corner when steering ~-0.46rad and speed ~2.7m/s.

Same-bag counterfactual diagnostic replay (never used for runtime control):
- causal initial state: 7 windows >5cm, peak 0.08451m
- Unity oracle initial u only: 5 windows, peak 0.08204m
- Unity oracle initial u/v/r: 5 windows, peak 0.08458m
- geometric jounce initialization: 10 windows, peak 0.06275m
- causal steering boundary correction: 7 windows, peak 0.08451m
- Bridge request-stamped command reconstruction: 7 windows over 2182
  valid starts, peak 0.08451m

These outliers persist after correcting command timing and start motion.
Do NOT add an empirical yaw correction or claim source parity based on
terminal u/r. Investigate actual PhysX/Unity lateral tire load, wheel contact,
body yaw impulse and suspension integration during high-slip transients,
and require validation over multiple independent corners/races.

## 2026-11-01: observer rear-wheel injection audited; a falsified root-cause hypothesis

The UnityPlantState published to MPC contains post-odometry-observer
*estimated* rear wheel angular velocities, not independent Unity ground-truth
rear wheel speeds. In odometry_observer.cpp the encoder-integral residual is
converted to rear omega innovation with 2*max(|residual|-encoder_quantum,0)/dt,
and that corrected result is injected into the source plant at the subsequent
sensor epoch. Front wheel omega has no such correction. This explains why
independent-source-step minus published observer rear omega occasionally
exceeds 40rad/s and both rear errors have the same sign. These discrepancies
must NEVER be represented as direct Unity rear-wheel-physics errors.

Causal counterfactual implemented in tools/evaluate_mpc_bag.py with option
--source-initial-rear-omega-from-prior-step. For each scoring window, it
reconstructs the raw pre-observer rear-wheel omega using the immediately
preceding source state and the VERIFIED request slot-2 command (no Unity truth,
no future measurements). All 30 subsequent physics steps are otherwise
unchanged. A/B over three fully finished practice races:

                       baseline >5cm  raw-rear >5cm    baseline peak -> raw peak
phase14...two_sqp_01         7/2184        6/2183        0.08451 -> 0.08548m
phase14...clean_02          11/2186       11/2185        0.11242 -> 0.11244m
phase14...clean_03          10/2189       10/2188        0.07007 -> 0.06758m

This does NOT reliably improve rollout accuracy. Encoder-estimated rear-wheel
velocity is not the demonstrated cause of source/Unity trajectory mismatch;
do not remove the correction or introduce an unvalidated plant-state override.

Cleanup in tools/evaluate_mpc_bag.py: one cached, shared prior-source 25ms
transition is now used by both encoder diagnostic and 750ms counterfactual;
the old test alternative which wrongly used request queue slot 1 to reconstruct
pre-observer wheel state is removed and replaced by the source-grounded slot-2
alternative. The older explicit slot-1 scenario remains for a labeled
counterfactual phase comparison; it is NOT the production input. Identical
0.085478857m peak and 6/2183 failures before/after this code cleanup,
verifying parity. No production physics/weights/control code changed by it.

tools/analyze_source_plant_one_step.py separately reports raw-source versus
next-PUBLISHED odometry wheel/spin/jounce/steering/body telemetry and normal
loads, and must be interpreted as observer innovation, not independent Unity
wheel-truth evidence.

## Independent truth-reference inconsistency is a second error mechanism

The source/world-pose >5cm failures are not all the same mechanism:
- tight steering at 2.3-2.8m/s: yaw residual builds for ~250-300ms and leaves
  an accumulated heading and trajectory offset even after terminal r matches;
- fast 7-9m/s intervals: endpoint speed/yaw errors can be small, but Unity
  published pose disagrees with its own twist integral by as much as
  ~4-6cm over some 750ms windows. This is not, by itself, proof of incorrect
  tire dynamics.

The separate source-independent --truth-pose-twist-phase-only diagnostic on
phase14...clean_03 confirms the pose/twist reference discrepancy (per 25ms
RMSE 0.493cm, worst 5.281cm), with the lowest overall RMSE at zero
timestamp offset (+0ms). The extreme individual intervals occur at
~6.8-8.2m/s; a global 25ms queue shift or arbitrary extra sensor lag is
NOT supported by these experiments.

Physics accuracy remains NOT certified to a 5cm maximum across full races;
do not remove source validation, Riccati feedback, or physical safety
checks merely because 10-lap totals are about 49.1s.

## 2026-11-01: high-slip Jacobian radius and physically meaningful SQP benchmark

Added reproducible offline tool:
tools/audit_mpc_source_corner_jacobians.py. It loads three complete practice
races and seeds the exact production 25ms source forward sensitivities at
36 recorded steering >0.3rad, speed 2-4.5m/s epochs with the verified
request queue slot 2. Compares 5,544 derivatives (11 controls/state
perturbations x 14 physical and integrated-motion outputs x 36 samples)
against independently reevaluated exact nonlinear source differences.
Machine-readable reports:
live_runs/mpc_phase14_high_slip_source_jacobians_20261101.csv
live_runs/mpc_phase14_high_slip_source_local_jacobians_20261101.csv

At larger physically meaningful finite perturbations, 420/5,544
directional comparisons differ by >0.01 in raw derivative units, max
17.784 (jounce-to-yaw); at 1/1000 of those perturbations, NONE exceed
0.01, maximum 0.001217, p99 absolute difference 0.000229.
This establishes locally consistent source AD while exposing severe
finite-step curvature and tire/contact branch sensitivity. It is not
evidence of a wrong C++ Jacobian in the infinitesimal limit, and the
physical plant still must be matched independently to Unity.

A permanent regression gtest now tests the source sensitivity versus
independent exact plant finite differences at the exact recorded high-slip
corner state (stamp 1793554724271802658, speed 2.43m/s, rack -0.426rad,
yaw -3.13rad/s), including pitch, jounce, steering and throttle directions,
u/v/r, rear wheel spin and 25ms integrated displacement. All 21 CTests
pass. NO production physics approximation or artificial yaw correction
was introduced.

Corrected the experimental *multi-iteration shooting benchmark* in
tools/mpc_bag_solver_replay.cpp:
1. The QP now uses the same frozen reference and speed-weight objective
   for every physical relinearization (previously rebuilt its own QP targets).
2. The QP includes the actual trajectory corridor polygons rather than
   disabling them as a wide-corridor diagnostic.
3. Every proposed full nonlinear source candidate is scored against the
   original track geometry; the old wide-corridor-only acceptance is removed.

Corrected full-step QP shooting outcomes across saved cases:
23.8618m 310.53 -> best 305.59 exact objective;
39.8147m 466.72 -> no improvement;
50.8604m 372.90 -> first full step fails original hard corridor
(stage 27). The constrained ADMM may say optimal in one iteration even
when its linearized trajectory changes by >1m laterally against source.
No claim of real-time improvement or robust QP certification is warranted.
The separate direct-source adjoint experiment previously improved the
same three objectives substantially: a better nonlinear optimization
direction DOES exist.

Added separate, additive endpoint classifications to the source/Unity
case index, retaining the original maximum-full-path >5cm gate. Across
the three complete practice races:
  run01: 7 full-path failures; 7 endpoint source-motion dominated;
  run02: 11 full-path failures; 11 endpoint source-motion dominated;
  run03: 10 full-path failures; 1 endpoint source-motion dominated,
         1 reference-pose dominated, 7 mixed or underdetermined.
The classifications score independent integrated Unity pose versus Unity
twist and source versus integrated Unity twist. They do not mistake
a noisy ground-truth pose sample for a source-physics correction.
Seven focused Python case-index tests pass.

Remaining implementation objective: correctly represent local dynamic
curvature or use source-exact constrained shooting with a physically
meaningful step selection, WITHOUT introducing a second incompatible
target schedule or hiding invalid source states behind a wide QP corridor.
A 3rd/4th unrestricted frozen-QP pass is not justified by the evidence.
Retire runtime fallback paths only after the normal physically accurate
formulation qualifies across repeated complete races.

## 2026-11-01: five-run qualification failed; cold-QP initialization audit

Following the initial three full 10-scored-lap completions, fourth race
live_runs/phase14_shared_merit_clean_20261101_04
ABORTED after 7 recorded laps at unwrapped progress ~228.008m.
The captured MPC diagnostics identify status=rejected_solver,
admm_status=2, zero ADMM iterations, zero completed Riccati
factorizations, no source Jacobian/nominal build failure, no deadline
exhaustion, all 30 Jacobian stages built. The captured controller had a
prior certified-plan snapshot, but this cycle published no command.
This invalidates the five-consecutive-race reliability criterion.
Do not misclassify this as a collision or physical rollout failure.

riccati_solver.cpp originally initialized a fresh ADMM solution by solving
a completely unconstrained zero-rho Riccati problem BEFORE the actual
positive-rho constrained iterations. That auxiliary solve can fail on its
2x2 control Schur complement without establishing that the actual ADMM
system is impossible. This is a suspected direct cause of the recorded
zero-iteration rejection, not yet proven by a bit-identical live-QP replay.

An experiment replacing ALL cold starts with a positive-rho initial solve
passed tests but then aborted at lap 0 (nonlinear corridor stage 23):
phase15_regularized_cold_init_20261101_01. It is NOT retained.

The current narrowly scoped numerical fix leaves the previously verified
zero-rho cold seed unchanged on success and invokes the actual positively
regularized Riccati initialization ONLY if the auxiliary zero-rho pass
fails; the original QP ADMM constrained iteration, nonlinear source
validation, convergence checks and output policy are unchanged. Added
RiccatiSolverColdStart.AuxiliaryIndefiniteSchurDoesNotRejectAugmentedQp
regression: deliberately non-SPD zero-rho control Schur complement fails
on the auxiliary factorization, while positive-rho ADMM can execute
ordinary iterations. All 21 CTests pass. This is an internal QP numerical
initialization repair, NOT a new behavioral fallback controller. A new
10-scored-lap race must validate it before qualification. Do not claim
the source/Unity slip error resolved.

## 2026-11-01: qualified five race completions under conditional internal cold seed

The fixed always-positive-rho cold-init experiment was REJECTED following
its immediate race-zero physical corridor failure. The retained repair in
f1tenth_mpc/src/riccati_solver.cpp keeps the previously validated zero-rho
cold initialization unchanged on success, but if that auxiliary
factorization fails it retries initialization against the actual
positive-rho ADMM system. The original projection, QP iterations,
convergence gates, nonlinear physical validation, and publication
policies remain unchanged. This is numerical QP initialization, not a
recovery controller. A GTest demonstrates that an auxiliary
indefinite zero-rho 2x2 Schur complement cannot prevent valid augmented
iterations. This confirms the mechanism is possible; original live
factorization state was not captured, so the precise previous failure
cause has NOT been bit-identically replayed.

Five independent consecutive 11-lap trials, one warmup lap plus 10 scored,
all finished with no crashes and no rejected MPC solver cycles:
    run                       ten scored lap total      gate max     recovery
    phase15..._01             49.129s                   18.044ms      0
    phase15..._02             49.353s                   18.104ms      0
    phase15..._03             49.226s                   18.091ms      0
    phase15..._04             49.236s                   19.895ms      1
    phase15..._05             49.147s                   18.157ms      0
Mean 49.2182s; all <=49.8s; no 20ms deadline miss. Bridge command-
epoch parity is exact for every observed epoch in all five runs.
Machine-readable summary:
live_runs/mpc_phase15_cold_init_five_race_20261101_summary.json

IMPORTANT: A five-successful-RACE gate passed, but the higher goal of
robust FALLBACK-FREE QP did NOT pass:
- Race 02 cross-track RMSE 0.09530m, peak |e_y|=0.72689m near unwrapped
  progress 66.31m. This was not one QP rejection. Repeated accepted
  line searches chose alphas 0.0625/0.125 with ADMM 150 or 300 iteration
  caps and nontrivial primal/dual residuals. A corridor slack ~0.415m
  shows the current physical state was not a wall collision.
- Race 04 accepted one viability recovery near progress 293.257m,
  the nonlinear full candidate failed the 30-stage horizon at stage26.
  ADMM took 288 iterations, primal residual0.2457, dual0.3451 and
  stopped on time budget. Retaining source-certified feasibility is
  mandatory until normal QP can handle that branch independently.
- Worst-case 19.895ms in race04 leaves very little publication margin.

Independent exact source/Unity 750ms audits for the two hard runs:
- Run02: 3/2196 overlapping full windows above 5cm, peak 5.45cm,
  none clearly source-dominated at the endpoint.
- Run04: 9/2193 overlapping full windows above 5cm, peak 6.13cm;
  eight endpoint source-motion dominated, sharp yaw transient around
  progress 27.47m. Physics fidelity is NOT fully solved.

Do not remove the existing Riccati feedback, source collision/footprint
validator or rare control recovery while this evidence remains.
Next work MUST improve normal QP physical trajectory consistency and
finite-step nonlinear correction choices at saved cases (progress
66.31m and 293.257m). Keep immutable objective schedule between
relinearizations, and avoid false full-step LTV feasibility claims.
