# Practice-map trust-region-free MPC trials — 2026-11-01

## Goal and preserved baseline

Phase 1 / Phase 3 validation on the practice map using the same source-command_4p800 raceline, original Unity/PhysX source model, full physical footprint corridor and 20 ms publication gate. Existing f1tenth_mpc/config/mpc_practice.yaml and competition production configurations are unchanged. The prior qualified practice five-race average was 49.1622 s. A concrete canonical individual ten-lap comparison run, practice_canonical_profile_20261101_01, completed in 49.041 s (3.03% zero-alpha, max gate 18.316 ms, no collision).

Only per-run overlay files under live_runs/mpc_phase13_practice_development_20261101 were used. No model, QP weight or geometric corridor constraints were relaxed.

## Isolated trials

| Test | Outcome |
|---|---|
| Direct-input QP, zero steering/throttle SQP trust radii | phase13_practice_direct_no_trust_20261101_01: no lap, corridor@0, progress 27.177 m; QP/restoration unable to produce exact-valid successor |
| Increment-input QP, zero trust, feedback rollout gain 0.75 | phase13_practice_increment_no_trust_20261101_01: no lap; invalid_model@-1 near progress 22.851 m, speed 7.37m/s; ADMM 150 iterations, primal .088, dual .569 |
| Increment-input QP, zero trust, feedback OFF | phase13_practice_increment_no_feedback_20261101_01: warmup plus 2 scored laps, 5.126/4.969s; zero collisions, 21.1% alpha=0; 4 viability-recovery actions |
| Increment-input QP, zero trust, feedback gain 0.25 | phase13_practice_increment_gain025_20261101_01: warmup plus 2 scored laps, 5.104/4.972s, no collision; 16.5% alpha=0, 0 recoveries |
| Same, first full race | phase13_practice_increment_gain025_tenlap_20261101_01: TEN scored laps 49.210s; max gate 18.253ms, 0 deadlines/collisions, 1 recovery, 10.50% alpha=0 |
| Same, independent second full race | phase13_practice_increment_gain025_tenlap_20261101_02: TEN scored laps 49.728s; max gate 18.098ms, 0 deadlines/collisions, 0 recoveries, 12.01% alpha=0 |

Both full races are under the specified 1–2% practice regression guard relative to the specific historical canonical 49.041s run (+0.34% and +1.40%). Their mean is 49.469s, compared with the established five-race mean 49.1622s. This is TWO independent completed races, not a five-trial robustness certification. Slower scored laps still occur without recovery (5.129 and 5.115s in second race).

The experimental leader is live_runs/mpc_phase13_practice_development_20261101/practice_increment_gain025.yaml. It has zero steering/throttle SQP proximal trust radii, increment-QP enabled, feedback_rollout=true at gain 0.25. The 1.04719755rad steering-command increment and 1.0 throttle increment limits span the full physically possible command ranges; they are NOT local SQP trust regions around the previous iterate. Actual steering and throttle absolute bounds remain enforced along with source dynamics and nonlinear collision checks.

## Where the normal solver still stagnates

Using progress modulo the practice raceline length 26.5929521m, alpha=0 in ten-lap full race 01 clusters at:
- 10–12m: 28.4% experimental vs 10.3% canonical; primarily merit rejection.
- 22–24m: 21.4% experimental vs 15.2% canonical; merit and future corridor rejection.
- 24–26m: 42.0% experimental vs 8.0% canonical; future corridor and merit rejection.
- 26–28m (wrap): 51.8% experimental vs 6.5% canonical; predominantly corridor rejection.

Those stations should be compared against baseline at identical physical state and actuator epoch, not optimized via track-specific weights.

## Next engineering gates

1. Save/replay hard first-hairpin and late-corner states: measured x0, delayed command queue, nominal warm-start, initial QP residuals, nonlinear source rollout, predicted corridor slack at each future stage, and exact merit at alpha=1,1/2,1/4,1/8,1/16,0.
2. Fix normal SQP globalization to produce useful nonzero accepted directions even with infeasible nominal forecasts. Explore feasibility-oriented residual merit and separate steering/throttle acceptance where physically valid; retain exact source-plant/footprint authority.
3. Analyze Riccati feedback as a function of x0 yaw rate/lateral velocity/slip/steering-actuator state and the 25ms command epoch. High gain 0.75 failed and gain 0.25 completed two races; this is not proof that 0.25 is globally optimal.
4. Reduce alpha=0 toward canonical 3% and eliminate slow-lap tails. Verify 5 independent full races under 20ms before promoting any production configuration.
5. Only then transfer changes to the competition controller. Maintain IPOPT as geometry/speed/wall reference, never command/state replay; validate full-speed competition first one safe lap, then ten scored laps and repeatability.

Each run folder retains sealed manifest, ROS bag, physical collision watcher and reconstructible solver/cycle diagnostics. No legacy experimental solver branch was added to production.
