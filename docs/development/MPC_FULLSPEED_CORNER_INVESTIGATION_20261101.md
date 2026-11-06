# Full-speed competition-corner investigation — 1 November 2026

## Comparison conditions

All tests used the unchanged latest source-validated 6.925-second IPOPT
competition witness in f1tenth_planning/trajectories/autodrive_iros_2026_compete/
ipopt_best_6p925_source_witness.csv. This is an offline source time,
not an achieved closed-loop Unity lap. The mandatory 50mm body corridor,
actuator command bounds and 20ms publication fault gate were unchanged.
The user-requested IPOPT service remains STOPPED.

## New full-speed Unity results

| Experiment | Completed laps, including warmup | Fatal outcome |
| --- | ---: | --- |
| Steering SQP trust radius 0.08 -> 0.20 rad | 0 | corridor stage 16 |
| Steering-slew penalty 2000 -> 750 | 0 | invalid model / nonlinear |
| Trust and slew both adjusted | 0 | corridor stage 23 |
| Progress-aligned fractional warm start | 0 | corridor stage 25 |
| Overspeed penalty 20 -> 80 | 0 | corridor stage 21 |
| Overspeed penalty 80, curvature scale 2 | 0 | corridor stage 13 |
| Local soft interior reference reserve 80mm | 0 | corridor stage 22 |
| Local soft reserve 150mm | 0 | corridor stage 27 |
| Conservative occupancy-map sweep as Frenet alternative | 0 | corridor stage 22 |
| Witness steering recovery with 100% and 95% source throttle | 0 | corridor stage 8 |

All results are actual Unity watcher outcomes stored in corresponding
live_runs/comp_*_20261101_01 folders. None demonstrates native-speed
race completion. Unsuccessful new C++ algorithms were REMOVED rather
than accumulating redundant toggles. The historic recovery algorithm
remains opt-in and production practice behavior unchanged.

## Technical evidence

- The controller callback timer is 5ms; MPC prediction stages are
  25ms. Previous warm-control sequences are shifted by one complete
  prediction stage on each accepted cycle. This is a meaningful timing
  concern, but an initial progress-fraction alignment FAILED its full
  speed Unity trial. Future changes must align true command epochs,
  physical state time and actuator queues together, not just position.
- The correct 750ms SQP QP already includes coupled e_y/e_psi oriented
  footprint polygon constraints. A heading-frozen duplicate wall
  penalty is not the missing feature.
- In a full-speed rejection at s=66.018m, the controller had dwindled
  from 14.6mm to 2.4mm extra predicted corridor slack over preceding
  control cycles. Measured speed was 6.73m/s vs 7.13m/s target.
  Another failed attempt at s=56.49m was about 0.65m/s ABOVE the
  rapidly decreasing source speed. These are distinct corner mechanisms.
- The fastest source witness requires up to 13.35m/s2 in v^2*abs(kappa)
  (roughly 9.86m/s2 at 90th percentile). Minimum source speed is
  2.476m/s in tight hairpins. Low corner speed in the SOURCE optimum
  must not be conflated with the MPC's extra fallback-induced halts.
- Source-plant certification reports only 1.85mm sampled EXTRA
  clearance beyond the mandatory 50mm. A simple independent
  orientation-aware Frenet clearance computation finds about 8.4mm
  additional margin around s=32.485m, despite about 1.375m free
  room on the opposite side. The fastest line skims a wall, but a
  much safer adjacent contour is geometrically possible in places.
- Once source validation fails at corridor stage zero, queued steering
  and throttle commands have made that transition unavoidable by
  the new command. Additional SQP iterations there cannot rescue it.
- With approximately 18ms P99 control callback and a hard 20ms
  publication gate, indiscriminately adding RTI passes is unsafe.

## Protected baseline and nearest complete competition evidence

Practice retains its independent qualified YAML profile
f1tenth_mpc/config/mpc_practice.yaml with lateral=4500,
heading=1450, nonlinear feedback gain=.75. Five qualified races
averaged 49.162 seconds. Four newer clean ten-lap practice regression
races were 49.158, 49.123, 49.064 and 49.528 seconds: the last misses
the stricter sub-49.5-second target, so there was NO promotion.

On the latest IPOPT geometry a deliberately capped 5m/s controller
completed 10/10 scored laps in 112.266 seconds without hard missed
publication deadlines. A 6m/s, 5m/s2 lateral envelope completed one
98.872-second race with a deadline miss, and FAILED an independent
follow-up. Neither is competitive or repeatability-qualified.

## Next engineering implementation: not another broad weight sweep

1. Precompute a *smooth anticipatory robust contour offset* from the
   existing IPOPT raceline at trajectory load time. One-sided body
   clearance can be improved 3-8cm toward available corridor width
   before the approaching hairpin. The existing single-sample local
   clearance clamp caused earlier Unity failures at 40-150mm; do not
   repeat that abrupt shift. Smooth spatial reference changes and
   obey physical curvature/actuator viability.
2. Add a feasible-successor certificate through the queued actuator
   timeline. A valid 750ms current horizon is NOT a recursively safe
   next MPC epoch. Select candidate plans with enough lookahead body
   reserve while retaining a full native-source-validated backup.
3. Improve nonlinear SQP globalization only where measured future
   corridor slack and QP residuals justify more iterations. First
   reduce Jacobian/QP overhead so two SQP passes fit the 20ms budget.
4. Validate any shared algorithm first on full-speed Unity competition,
   and run the unchanged practice baseline through five independent
   clean ten-scored-lap races before promoting. Keep track-specific
   weights and references separate, never separate the MPC algorithms.

Raw tests and configuration overlays remain archived under live_runs.
The user-requested IPOPT optimizer must not restart automatically.
