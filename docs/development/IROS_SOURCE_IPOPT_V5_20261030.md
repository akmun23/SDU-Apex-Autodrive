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
