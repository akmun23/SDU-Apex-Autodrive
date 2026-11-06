# Source MPC restoration and curvature verification

Status: **not qualified**. 6 November 2026.

The controller must recover all 22 recorded problems from their recorded
controls, preserve physical feasibility, and converge on the original racing
objective. Neither passing tests nor a positive wall margin establishes this.
The separate IPOPT reference recovered 16/22, with 0/22 KKT qualification; it
has not justified replacing the runtime algorithm.

## One mathematical problem

The runtime still optimizes 30 steering/throttle pairs, with all states
eliminated by the source transition. Original inequalities are 240 body-corner
clearances, 120 actuator-rate margins and 120 legal command bounds. A negative
physical clearance cannot qualify, however small. The independent oracle now
always scores the original racing objective and rejects requests to add the
discarded extra 3 cm reserve penalty. The genuine original reserve cost remains.

The original source formulas include the Unity steering slew, Ackermann wheel
angles, nonzero-throttle motor torque and zero-throttle brake branch. The
source model retains wheel, suspension and body dynamics, including the
recovered five-versus-one vehicle substep rule at 5 m/s. It remains a source
translation, not a bit-exact reproduction of Unity's private WheelCollider
solver. Offline source qualification cannot establish Unity prediction fidelity
or performance across all tracks.

No trust radius, additional output envelope, alternative controller, or
multiple-shooting runtime path is used. QP bounds are the legal control bounds.
The same original-source evaluation decides nonlinear acceptance and eligibility
for publication. Elastic residuals never certify physical feasibility.

## Restoration and accepted progress

The hard QP retains its direction and duals when certified. Failure of that
solve triggers the internal normal/tangential construction. Independently,
the permanent qualification harness uses HiGHS LPs to classify affine
feasibility within legal command bounds. An inconsistent linearization is not
evidence of a poor Hessian or a physically unrecoverable state.

Physical restoration minimizes

\[
V(U)=\tfrac12\|[g(U)]_-\|^2.
\]

While infeasible, acceptance requires exact nonlinear violation reduction;
becoming physically feasible may increase racing cost. Once feasible,
acceptance preserves feasibility and decreases the original racing cost.
Backtracking uses the original source one-sided directional slope. It is an
adaptive line search, not a fixed grid of command alternatives. Second-order
corrections are checked against the same source trajectory.

A saturated regularization parameter, rounded-zero command change or repeated
precision-limited accepted progress produces explicit numerical failure unless
the original KKT conditions pass. Source derivatives are cached at unchanged
controls. These failures do not become convergence claims.

An additional defect was isolated in the normal problem: rejected racing steps
multiplied its Tikhonov damping from 1e-6 to 1e6. This made its own regularized
stationarity compatible with negligible physical violation reduction. At the
failed 773/775 iterates, physical violation gradients were still approximately
8.75e-5 and 7.29e-5 in normalized coordinates. Removing damping alone exposed
another issue: the affine least-squares minimizer is generally nonunique, and
barrier centring can move irrelevant controls. An exact lexicographic
minimum-norm refinement was investigated, but the native barrier solver failed
on its degenerate feasible face. An independent HiGHS QP solved the same local
problem. That refinement is **not retained in the runtime implementation**.

The runtime retains the regularized elastic normal problem with a fixed small
coefficient `mu=1e-6` in normalized controls. It cannot grow after racing-step
rejections. Its exact optimum satisfies

\[
V_{\rm affine}(n)+\tfrac\mu2\|n\|^2\le V_{\rm affine}(0),
\]

so a nonzero normal predicts violation descent. This norm term belongs only
to the internal normal model, never the original racing objective. Exact source
acceptance remains authoritative. Negative normal residual rows become
equalities in an SVD nullspace; other physical rows remain inequalities.
Equality row equilibration preserves independent physical equations across
large unit scales, with multipliers mapped back to original units. Restoration
duals are not original NLP multipliers. Regression checks verify the known
scalar normal solution, its invariance to racing damping, physical bounds and
correctly scaled equality multipliers.

## Source curvature and switching

Complete path derivatives propagate both first and second state tangents,
including incoming `A*x_second`, through the source plant, implicit progress
projection, rectangular footprint, rack-rate rows and original racing features.
The objective derivatives include longitudinal peak slip, front lateral slip,
steering/throttle cross coupling, actuator queues and genuine reserve cost.

At genuine source ties, derivatives follow the side selected by the path.
Contact, brake and discrete integration ties invalidate a classical local
Hessian where appropriate. The physical forward model is never smoothed to make
derivatives easier. Tracking interpolation segment changes are now compared
explicitly: constant corridor widths previously hid centreline/heading changes.

An expensive offline Hessian reference polarizes exact source directional
curvature of `J - lambda*g`. Three independent directions check quadratic
consistency. Only certified original hard-QP multipliers contribute constraint
curvature; otherwise zero multipliers retain exact objective curvature during
restoration. This is a curvature reference for the same SQP, not another solver.

Symmetry and quadratic consistency are insufficient validation. The independent
audit also compares vector products against branch-preserving differences of
the source Lagrangian gradient, accounting for realized float command changes.
Its report keeps contaminated finite-difference columns separate from the
analytical reference. The three directional probes are not an all-entry or
global smoothness certificate. Epoch 769 passed the analytical HVP checks;
765 and 775 did not pass every tight check. They must not be called fully
validated Hessians.

Full reference curvature costs seconds per iteration on this host and is not
enabled for production. In an 80-iteration diagnostic, epochs 765/769 became
feasible, while 775 remained infeasible. All three failed KKT qualification.

Original optimality has a separate nonsmooth issue. Near the feasible epoch-754
iterate, a 8.94e-8 physical command change switched the selected peak-slip
substep and changed one normalized objective-gradient component from +4.18 to
-10.90, with no other observed physical branch change. More smooth Hessian
tuning cannot establish a classical stationary gradient at this max-feature
switch. The existing exact peak epigraph experiment reduced the residual but
did not achieve the required original first-order certificate. It remains
diagnostic and is not promoted to the runtime path.

## Reproduction and evidence

The immutable corpus is `f1tenth_mpc/test/data/competition_corner_20261105`.
The permanent qualification command is:

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python3 -m tools.qualify_mpc_source_phase1 \
  --tools build_task/source_curvature \
  --output artifacts/source_phase1_new_run \
  --iterations 2000 --repeat 2 --jobs 4 --perturbations 2 \
  --baseline-report docs/development/MPC_PHASE1_QUALIFICATION_20261105.json
```

Use an empty output directory. The harness pins native executables before work,
records checksums and preserves recorded/perturbed controls. Independent jobs
preserve corpus and RNG order; each deterministic repeat remains serial.
Ten perturbations supplement, and never replace, recorded starts. The generic
replay report's `passed` field is only replay consistency. The actual gate is
`phase1_report.json` with separate feasibility and KKT qualification.

Local detailed traces and superseded numerical comparisons are under
`artifacts/source_path_recovery_20261106/`. The final measured report and source
checksums are archived alongside this document when qualification finishes.

Unused multiple-shooting solver/oracle tools and obsolete extra-reserve
comparison scripts were removed. They cannot become an alternative runtime
path or a misleading objective reference.

The remaining gates are 22/22 recovery, trustworthy first-order convergence,
10/10 perturbation robustness, and then independent real-time and Unity
validation. No existing result establishes a robust racing controller.
