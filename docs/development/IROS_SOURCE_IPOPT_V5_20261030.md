# IROS 2026 source IPOPT v5: sector timing, physical seed preflight and rigid near-miss recovery
Date: 2026-10-30

## Certified result recovered

An independently sourced production witness at **7.1000000000000005 s** has passed the
unchanged independent source dynamics/periodicity check and the full 1 ms
vehicle-perimeter check. The prior 7.100 s near-miss only violated numerical
clearance by -4.66048033498e-12 m. A bounded rigid SE(2) search identified a
translation (-3.31955e-10, -5.12154e-11) m and yaw rotation -4.64714e-11 rad.
Independent source validation of the transformed commands passed with
1 ms minimum extra clearance slack **2.99e-11 m** and maximum recursive XY
difference **1.1492e-08 m**.

- Automatically reproduced certificate:
  `live_runs/iros_new_basin_test_20261030/automated_7p100_rigid/certified/`
- Strict physical source contract and 5 cm vehicle clearance remain unchanged.
- This is a **numerically marginal source certificate**, not a robust geometric
  margin, proof of global optimality, or achieved Unity/MPC ten-lap performance.
- The rigid repair is now called automatically for extremely near-feasible
  shorter candidate reports (negative slack no worse than 1 µm, scaled residual
  <=1e-5, XY rollout mismatch <=0.1 mm); the recovered candidate must pass a
  full independent source check and the campaign's second certification.

## New capabilities

1. `sector_time.py` provides 1–32 periodic sector-local time decision variables.
   Timing is linearly interpolated between cyclic sector knots to avoid hard
   boundaries. An N-interval 24/25 ms *source-map chord* is still explicitly
   an **uncertified time initializer**, not the production vehicle dynamics.
2. `command_ocp.py` now supports 12 timing sectors in the actual CasADi NLP,
   with a time-minimizing objective and a smoothness regularizer over sector
   fractions. The existing scalar time predictor is retained as version 1.
   Time proposals resample by cumulative per-packet time and preserve held
   packet ownership, not by uniform fractional lap progress.
3. `causal_seed.py` creates alternative steering-command profiles and
   propagates all 21 physical source states through the production source
   transition, settling hidden memory before distributed periodic projection.
   The projection is **uncertified** and remains subject to IPOPT source repair.
4. A new map preflight skips source-propagated initializers with >10 mm
   25 ms node clearance deficit. These obviously invalid starts must not
   consume an entire 1200-iteration IPOPT solve. Preflight is not certification:
   even a preflight-passing seed must pass 1 ms full-perimeter replay later.
5. `rigid_pose_polish.py` optimizes bounded global translation/rotation of
   almost source-feasible witnesses via a linearized near-contact clearance
   problem and evaluates the **actual** clearance before revalidation.
   Translation/rotation preserve the source dynamics mathematically.
6. `command_campaign.py` includes the automatic near-miss recovery, the
   causal-alternative archive branch, separate hybrid sector-time search
   contract, independent certifications and previous equal-time quality
   objectives. It never promotes a simulated-only time prediction.
7. `track_optimizer.py` has `--time-sectors 12` CLI support, retains the
   certified seed/source fingerprints and includes all new implementation
   modules in its code hash.

## Real-world diagnostics

- Sector-time predictor (285 intervals, 12 sectors, 24 IPOPT iterations)
  initialized at the independently certified 7.125 s witness and explored a
  slightly shorter time, but **did not produce a shorter whole-25ms interval
  proposal** within that deliberately bounded trial. Its output was labeled
  `no_time_proposal`.
- A source-propagated +0.002 rad steering-harmonic seed had initial scaled
  dynamic defect ~0.006, but its XY geometry left the map free-space boundary
  (node clearance -0.07275 m) and the trial made no useful progress. That
  exploration was stopped; the map preflight now rejects similar seeds rather
  than waste solver cycles. Alternative geometry is not yet solved in general.
- Automated 7.100 s SE(2) recovery was re-run against the source physics and
  reported `source_witness_passed`. The entire 25ms steering/throttle command
  sequence is unchanged by rigid transformation.
- 85 focused tests passed, including CasADi sector timing dimensions,
  independent resampling, source seed generation, archive integration,
  rigid correction and campaign promotion. This does **not** establish
  Unity closed-loop drivability.

## Next acceptance milestones

A. Produce a second independently certified 7.100 s or faster witness with
   materially greater than numerical extra clearance, ideally >=1 mm.
B. Independently validate candidate source witnesses in Unity with MPC for a
   warmup and ten scored laps; report actual achieved lap time and variance.
C. If the sector predictor remains trapped, introduce map-constrained geometry
   basins using bounded lateral offsets at selected corner entry/apex/exit
   landmarks instead of blind global sinusoidal steering perturbations.
D. Keep exactly one heavy optimization job active. The dashboard should show
   the active new workspace and distinguish certified improvements from
   infeasible IPOPT progress.


## 30 October: substantial equal-time quality improvement and early acceptance

The running equal-time clearance objective produced a separate, near-feasible
geometry at 7.100 seconds. Its repair phase saved an iterate with scaled
constraint residual 9.622098e-06, but IPOPT had not converged by ~700
iterations. Stopping the solve at a durable checkpoint and running
`certify_witness` against the unchanged production source model produced a
**source-certified 7.100 s witness with 0.016890176 m extra 1 ms footprint
clearance**, beyond the same 0.05 m required margin. The steering-command
step RMS also improved from ~0.05218 rad to ~0.04237 rad.

This is an equal-time **quality** gain, not a faster lap and not a Unity
ten-lap driving result. The checkpoint meets the strict scaled dynamics
acceptance threshold of 1e-05, but does not prove an exact mathematical
solution or performance on the physical controller.

A new optional `solve_witness(early_clearance_target_m=...)` monitors the
*restoration objective only*. Every 40 iterations, if scaled NLP residual and
periodic state defect are <=1e-05, it runs the original full 1 ms source replay.
It stops IPOPT at a saved candidate only if that replay passes the full XY
mismatch gate and yields at least the required positive clearance reserve.
The normal final source rollout and independent `certify_witness` still run
before archiving or promoting the witness. The elite restoration caller sets
the acceptance reserve to the incumbent slack plus 2 mm. In a full-source
replay using the saved high-clearance iterate, the early acceptance triggered
at iteration 0 and still produced `source_witness_passed`.

The GUI now correctly counts all equal-time alternatives, not only margin
and geometry variations. It reports the *published* source clearance separately
from archived alternative clearance and names restoration phases explicitly.


## 30 October: prioritize shorter feasible duration steps

A subsequent campaign from the high-clearance 7.100 s source witness first
attempted **6.400 s** with a globally resampled initializer; the large
dynamic infeasibility made this a poor initial use of the solver budget.
The campaign search-starters ordering was corrected to try the **7.075 s**
one-packet removal first, then 7.075 s resampling, and only then successively
larger duration reductions (7.050 s, etc.). Large coarse jumps remain
available rather than being discarded. The order is versioned as
near-time-before-coarse/drop-first-v2 so an interrupted older campaign
cannot silently change its execution history on resume.

The 7.100 s / +16.89 mm clearance model-certified witness remains the
launch seed. It has not been validated as a completed Unity/MPC 10-lap race.
