# Source-only SQP — two-track Unity qualification (2026-11-05)

All numbers below are measured from ROS/Unity bag exports rather than
offline source-only simulations. The current SQP uses one original-source
nonlinear MPC, full SQP control corrections, exact nonlinear candidate
validation and no separate fallback controller or nonlinear alpha grid.

## Practice: complete 10-lap race

- Run: `live_runs/source_sqp_practice_10lap_20261105_05`
- Setup: `mpc_practice.yaml`, overlay
  `mpc_practice_source_viability_trial.yaml`, lateral acceleration
  8 m/s², braking-reference limit 5 m/s².
- 10 complete laps, 53.046999 s total, 5.304700 s mean.
- Per-lap seconds: 5.9068, 5.2218, 5.2238, 5.2518, 5.2358,
  5.2288, 5.2528, 5.2528, 5.2388, 5.2338.
- 2,115 accepted original-source-feasible cycles,
  13 `waiting_for_solver_budget` cycles, no reported
  `rejected_nonlinear_rollout`, no reported collision.
- 30 pre-localization cycles excluded from source solving.
- Max observed source solver 33.976852 ms; p99 19.566207 ms;
  11/2,128 active source solves exceed 22 ms.
- Cross-track RMSE 0.017626 m; heading RMSE 0.023858 rad.
- **Not hard-real-time qualified**. Not yet below target <49 s.

## IROS competition: two complete laps, then source-NLP rejection

- Run: `live_runs/source_sqp_iros_viability_20261105_02`
- Map: `autodrive_iros_2026_compete.yaml`
- Trajectory: `ipopt_best_6p925_source_witness.csv`.
- Setup: `mpc_competition.yaml`, overlay
  `mpc_competition_source_viability_trial.yaml`, same reference
  speed-envelope configuration as practice, original track-specific weights.
- Supervised target 3 laps. Completed 2 laps, with 8.8690 s and
  7.6136 s measured. No collision reported.
- 683 accepted source-feasible cycles; 56 waiting-for-budget cycles;
  1 rejected nonlinear trajectory.
- Final failure: progress=143.847508 m, body speed=2.902865 m/s,
  lateral error=-0.170375 m, heading error=-0.261090 rad,
  final original source minimum footprint clearance=-0.431197 m;
  3 SQP iterations, 1 accepted, cost improved from
  38,976.73 to 19,461.24 but remained source-infeasible.
- Max observed source solver 26.769179 ms; p99 22.681038 ms;
  14/746 active solver calls exceed 22 ms.
- Cross-track RMSE 0.087785 m; heading RMSE 0.071142 rad.
- **Not qualified**: 56 unqualified-budget cycles, 1 genuine
  nonlinear infeasibility. The controller withholds infeasible commands.

## Engineering conclusions

1. The hard-wall versus soft EXTRA 3cm reserve separation eliminated
   erroneous reserve-only solver rejection during practice and enabled
   3-lap and 10-lap successful race runs.
2. Source-calibrated positive-semidefinite curvature correction improves
   predictions of nonlinear full SQP corrections but has not solved all
   competition corner failures.
3. Offline exact-source recoverability audit of the earlier practice
   high-speed failure (`source_sqp_fullstep_ratefix_practice_20261105_02`)
   tested 4,111 admissible command trajectories. None were source-feasible;
   the best still violated the corridor by 0.688 m. This is **not a
   mathematical infeasibility proof**, but supports proactive viability.
4. An additional experimental terminal-speed operating-inequality
   implementation is being developed. It was NOT active in these race
   configurations (not a reason to claim successor viability).
5. Next: reproduce competition failure with a cost/geometry parity-certified
   original-source fixture, distinguish recoverability from SQP
   nonconvergence, improve predecessor-state feasibility and terminal
   viability in the original problem, and reduce integrated >22ms latency.
6. The release Docker build initially failed because offline tools are
   excluded by `.dockerignore` but unconditionally referenced by CMake.
   The offline-only CMake target declarations are now conditional; the
   current competition image built successfully. The IROS runner also
   supports explicit Linux CPU affinity.

Source evidence lives in each run's `report.json`, `cycles.csv`,
`cycles_diagnostics.jsonl`, `watcher.log`, controller logs and ROS bag.
No legacy fallback, additional source-model approximation, or alpha-scale
search was introduced for these races.
