# MPC full-physics performance / multi-SQP progress — 2026-10-21

This progress log follows `SDU_Apex_Full_Physics_MPC_Performance_and_Multi_SQP_Handoff_2026-10-21.md` from reviewed head `0949ce3012a72848510405c3db98af21d1fec77e`. The handoff is a plan, not proof that its proposed work has already been done.

## Scope preserved

- No Unity/source-plant equations, tire/contact behavior, 1 ms plant integration, 30 x 25 ms horizon, drive-only MPC throttle bounds, map/reference, or nonlinear footprint/corridor gate were changed.
- No second SQP iteration or MPPI was introduced. An optional, bounded nonlinear backtracking implementation was added for offline comparison, but it is disabled in production because the runtime cannot certify the command-effect epoch required by the handoff. Earlier supervised practice runs ended on controller faults or collision; no capture in this performance branch completed a lap.
- The replay fixture is practice-track data, but its measured vehicle speed is limited. This is not high-speed/high-slip validation.

## Implemented

1. **Repeatable saved-input benchmark and phase instrumentation.** The working tree records nominal seed/rollout-A/reference-resample/rollout-B, 29-stage Jacobian wall and worker CPU, slowest stage, sensitivity and finite-difference counts, ADMM/Riccati subphases, candidate rollout, and exact validation. `tools/benchmark_mpc_replay.py` preserves raw CSV and machine/build/fixture metadata as JSON. The node now records publication-gate elapsed time separately from later diagnostic-emission time. The replay benchmark itself remains offline core timing.
2. **Remove exact-zero historical-command Jacobian work.** The plant consumes delay slot 2; slot 1 is only overwritten by slot 2 during the exact queue shift. The slot-1 derivatives are therefore exactly zero. Those two columns are now marked analytic/zero instead of running full-plant finite differences. A regression perturbs both old command slots and verifies no predicted state or physical output changes.
3. **Exact physical-transition reuse between nominal passes.** Pass A and Pass B use the same initial physical state and command sequence. The source-derived physical plant does not depend on Frenet curvature; only path projection/progress does. Pass A now retains its full stage transition. Pass B reprojects the cached body displacement/yaw through the newly sampled path curvature, avoiding a duplicate 30-stage 1 ms plant rollout. No physical state or validation is skipped. A 30-stage test compares cached reprojection to a fresh complete-plant pass B.
4. **Remove cold-start plant work used only to shift queued throttle.** The cold nominal seed previously simulated all 30 stages and 25 plant substeps per stage solely to read `delayed_throttle_command_2` after each queue shift. The queue recurrence is deterministic and independent of tire/body state; the seed now computes that exact recurrence directly. The actual full-physics nominal rollouts remain unchanged. A regression compares the new seed sequence against the previous source-plant queue evolution for initial queued throttle `0`, `0.2`, and `0.8`.
5. **Align the forward-mode tangent storage.** `Dual20` now begins with its 20 float tangent lanes in a 16-byte-aligned object, followed by the double primal. This preserves double primal arithmetic and all 20 derivatives while presenting tangent vectors contiguously to the compiler. The matching 409-row replay has bit-identical non-timing outputs and the full MPC package tests pass. Two portable replay samples improved cycle p50 from the queue-seed reference `11.841 ms` to `10.034/10.456 ms`; p95 was `12.538/14.097 ms` versus `14.089 ms`, while max varied (`18.938/16.754 ms` versus `18.088 ms`). Treat this as a promising exact-layout optimization, not yet a proven tail-latency improvement; the live gate miss below is decisive for real-time acceptance.
6. **Make failure-path evidence explicit.** After analyzing the live timeout, MPC diagnostics now record whether a previous-plan suffix was attempted, passed exact rollout/corridor validation, and then failed the timing gate, including validation duration. They also distinguish a published command from a valid-but-unpublished candidate. The evaluator includes these fields in its slowest-cycle report. This source change built and passed tests after the captured run; its new JSON fields still need confirmation in a subsequent bag.
7. **Add bounded, rejection-only nonlinear backtracking for offline study.** After a full-step exact corridor rejection only, the core can test alpha `0.5` and then `0.25`, each with a fresh source-plant rollout and the unchanged full corridor/objective checks. It does not rebuild Jacobians or solve another QP. It is off by default, and the ROS runtime does not assert that its predicted command epoch is causally safe for optional trials.

## Replay evidence

Fixture: `live_runs/mpc_performance_20261021/practice_runtime_priorplan_solver_inputs.csv` (409 saved cycles), paired with the practice map’s configured mintime reference. Same replay settings and input rows for each run; four Jacobian workers; 100 ADMM cap. Raw artifacts:

- Initial measured baseline: `live_runs/mpc_performance_20261021/practice_runtime_priorplan_replay.csv` and `practice_runtime_priorplan_benchmark.json`.
- Zero historical-command finite differences: `live_runs/mpc_performance_20261021/practice_runtime_priorplan_zero_history_fd_replay.csv` and `practice_runtime_priorplan_zero_history_fd_benchmark.json`.
- Cached Pass-B physical transition: `live_runs/mpc_performance_20261021/practice_runtime_priorplan_physics_cache_replay.csv` and `practice_runtime_priorplan_physics_cache_benchmark.json`.
- Exact cold-start queue recurrence: `live_runs/mpc_performance_20261021/practice_runtime_priorplan_queue_seed_replay2.csv` and `practice_runtime_priorplan_queue_seed_benchmark2.json`.

| Metric | Initial | Zero-FD columns | Pass-B physical cache | Exact queue seed |
|---|---:|---:|---:|---:|
| Core replay cycle p50 | 14.570 ms | 13.825 ms | 12.212 ms | 11.841 ms |
| Core replay cycle p95 | 17.320 ms | 16.529 ms | 14.588 ms | 14.089 ms |
| Core replay cycle p99 | 19.702 ms | 18.706 ms | 17.010 ms | 16.451 ms |
| Core replay cycle max | 21.157 ms | 22.859 ms | 19.505 ms | 18.088 ms |
| Nominal Pass B p50 | 1.823 ms | 1.819 ms | 0.0073 ms | 0.0067 ms |
| Finite-difference columns, total | 23,722 | 0 | 0 | 0 |
| Exact-accepted cycles | 394/409 | 394/409 | 394/409 | 394/409 |

Comparisons across all 409 rows found **no status changes and bit-identical first steering/throttle commands**; all non-timing fields except the deliberately changed finite-difference counter were identical. The nominal Pass-B timing directly confirms the cached path is doing only the inexpensive reprojection. Max cycle time is noisy across these offline runs; the replay is not deadline-gated and cannot establish a live 20 ms maximum.

The exact queue seed was also compared to the physical-cache replay on all 409
rows: solver status, iteration count, acceptance/failure fields, first
steering/throttle and full-plan deltas were identical. On the first cold
fixture row, solver cycle fell `14.358 -> 12.162 ms`; seed-only work fell
`1.796 ms -> 0.00073 ms`. Across all rows, cycle p50/p95/max changed
`12.212/14.588/19.505 -> 11.841/14.089/18.088 ms`. This removes computation
without changing vehicle dynamics or the nominal throttle policy.

The recorded operating points are modest: forward speed p95 2.87 m/s (max 3.08 m/s), absolute steering p95 0.290 rad (max 0.370 rad). The 15 nonlinear rejections and 4 iteration-limited cases remain unchanged. This performance work did not claim to fix convergence.

GCC 13.3's `-O3` vectorization report for the current `unity_source_plant.cpp` shows the tangent loops are already vectorized using 16-byte vectors. No SIMD source rewrite or fast-math flag has been added.

An offline worker-count sweep on the same fixture originally showed four workers fastest among 1/2/4. In response to the proposed eight-worker setting, the code and replay-tool limits were raised to eight. Four paired, interleaved replays of the same 409 saved practice states compared 4 versus 8 workers. All four pairs had identical status and first steering/throttle for every row (394/409 exact-accepted; same status counts). Median of per-run cycle quantiles:

| Workers | p50 | p95 | p99 | median of run maxima |
|---|---:|---:|---:|---:|
| 4 | 7.467 ms | 9.909 ms | 11.592 ms | 13.129 ms |
| 8 | 4.269 ms | 6.100 ms | 7.600 ms | 9.120 ms |

Eight workers were faster in each paired replay. There was substantial host-frequency variability between runs: the 4-worker per-run p50 ranged 5.18–9.80 ms, and 8-worker p50 ranged 4.03–8.45 ms. The largest 8-worker offline cycle was 17.48 ms; this alone was not proof of an under-20-ms live maximum. Raw artifacts `practice_runtime_priorplan_worker_sweep_{4a,4b,4c,4d,8a,8b,8c,8d}_{replay.csv,benchmark.json}` are in `live_runs/mpc_performance_20261021/`.

The native-ISA experiment was then isolated. `-march=native` with normal contraction was faster but changed 14/409 solver statuses and produced steering differences up to `0.489 rad` and throttle differences up to `0.328`; it fails parity and is rejected. With `-ffp-contract=off`, the 409 statuses and first steering/throttle commands exactly matched portable `-O3`. The explicit CMake opt-in `SDU_APEX_MPC_NATIVE_TUNING` now enables only `-march=native -ffp-contract=off`; it defaults OFF so the ordinary build remains portable. The opt-in replay measured cycle p50/p95 `11.508/13.569 ms` versus portable `12.212/14.588 ms` (about 5.8%/7.0% faster) and p99/max `15.880/17.661 ms` versus `17.010/19.505 ms`. The native-mode full package test suite passed 4/4. These results are specific to this CPU/compiler/corpus; the option is not enabled in the production default and should only be used when build and deployment CPU are the same compatible host.

## Supervised live timing check

Capture: `live_runs/mpc_performance_cache_live_20261021/run`. This used the
established batch-mode practice simulator and mounted current workspace build,
portable default flags, 3 m/s configured reference ceiling, full bag recording,
and immediate stop on collision or MPC rejection. No collision or lap occurred;
the car reached `2.144 m/s` before shutdown.

- The bag contains 252 optimal and 23 iteration-limited exact-valid solver
  candidates, but only 274 published MPC drive-command messages. One computed
  candidate was not published because the callback crossed the strict `<20 ms`
  gate.
- Solver-reported internal cycle p50/p95/max was
  `14.586/18.585/19.441 ms`; this is not the end-to-end publication gate.
- The terminal diagnostic explicitly reports `MPC cycle reached or exceeded
  its 20 ms budget`. The preceding candidate diagnostic records
  `20.647 ms` callback-to-diagnostic-emission and `18.736 ms` solver cycle.
  The original bag did not preserve the exact elapsed value at the gate, and
  successful-cycle diagnostic serialization can add time after publication.
- The node now records `publication_gate_elapsed_ms` at the actual check, and
  the evaluator reports it separately. This instrumentation rebuild passed
  and all four MPC package tests passed, but a fresh supervised capture is still
  needed to validate that new field live.
- This is a real live timing miss, not a pass. It confirms `<20 ms` is not yet
  guaranteed under simulator/ROS load. Do not enable another SQP iteration.
  The old-bag evaluator now labels legacy callback-to-diagnostic timing
  accurately and reports the explicit budget-rejection event and command
  publication count separately.

## Fresh live test of aligned tangent layout and current mounted source

Capture: `live_runs/mpc_performance_aligned_dual20_live_20261021/run`. The
established **practice** simulator was launched in batch mode with Xvfb/Vulkan
(no GUI and no `-no-graphics`), on
`autodrive_practice_20260924_b` with its configured mintime raceline. The
development container rebuilt the mounted current workspace source using the
portable build. The topic-policy guard passed. The 3 m/s setting is a
reference ceiling, not a hard speed limit. The supervisor stopped the run as
soon as collision count became one.

- The run collided, completed zero laps, and peaked at `2.207 m/s`. It is not
  a successful-driving or racing-performance result. There were 396 truth,
  odometry, and plant-state samples; the bag decoded schema-11
  `UnityPlantState`, `DriveCommand`, and the development actuator-timing
  stream directly. All 303 forwarded actuator events were sequence-contiguous;
  interface-receive-to-actuator-publish delay was
  `0.129/0.186/0.241 ms` median/p95/max. The compliance guard and decoder
  therefore worked on this exact run.
- The publication-gate elapsed time was `9.991/13.825/32.771 ms`
  p50/p95/max against the strict `<20 ms` gate; `1/304` cycles was at or over
  budget. Solver cycle was `9.781/13.612/23.938 ms`; QP/Jacobian wall was
  `5.086/6.584/21.044 ms` / `5.045/6.542/20.967 ms`. The single tail event
  cannot be dismissed as diagnostic serialization: the gate value is captured
  before publication. It also shows this layout alone does not establish the
  required maximum. Jacobian worker CPU is summed across workers and is not
  comparable to wall time.
- There were 303 exact-valid solver candidates among 304 gate records and 303
  published drive commands; one explicit publication-budget rejection was
  recorded. ADMM converged in 267 cycles; 36 iteration-limited candidates
  passed exact nonlinear validation. This confirms the valid-but-iteration-
  limited path is being used, not that the run is reliable at all operating
  points.
- The new evaluator outlier report identifies the worst row as
  `rejected_solver`: node solve `23.943 ms`, QP build/Jacobian `21.044/20.967
  ms`, ADMM `0 ms` (not started), and control-time prediction `0.164 ms`. The
  publication-gate check for this rejected solve occurs only after attempting
  exact revalidation of the shifted prior plan; that check recorded `32.771
  ms`, so the prior suffix could not be published within budget. The
  `8.828 ms` callback-minus-node-solve residual includes work outside the
  numerical solve; it is not all attributable to one function. The immediate
  optimization target is the long Jacobian/QP build, with a separate bounded
  budget path needed so late prior-plan validation cannot consume another
  over-budget interval.
- Fixed-input source-plant replay scored 366 windows. At 750 ms, `357/366`
  endpoints and complete paths stayed within 5 cm; nine exceeded. The
  collision counter first became nonzero at `1792567623.803 s`, and the
  extreme rollout errors occur in windows crossing that contact event. The
  maximum 750 ms position error was `1.216 m`; it must not be interpreted as a
  free-space tire-model error or hidden by the 5 cm pass fraction. Separately,
  the closed-loop receding-horizon score below is not a fixed-input rollout.
- The bag's *receding-horizon* comparison at 750 ms was `0.0703 m` XY RMSE,
  `0.1348 m` p95, `0.2708 m` max, and speed RMSE `0.3839 m/s`. These are
  successive replanned closed-loop forecasts, not a fixed-input 750 ms plant
  test. They do not meet the 5 cm endpoint target and must not be blended with
  the separate fixed-input rollout score.
- Production odometry on this short run had forward-speed RMSE `0.1784 m/s`,
  lateral-speed RMSE `0.0224 m/s`, and `1.266 m` relative-position endpoint
  error. Its yaw-rate residual is exactly zero because this output uses the
  gyro; that is an observation, not independent yaw-prediction evidence.
  AMCL scan-pose error before collision was `0.071 m` XY RMSE. The collision
  and short span make these metrics a failure diagnosis, not a clean odometry
  regression.
- The controller records a control fault and publishes no replacement when
  a candidate misses the timing gate. The development watcher terminates the
  simulator on collision/fault; no hard-brake command is sent. The saved-plan
  suffix path is implemented for solver rejection, but this run did not
  demonstrate a timely prior-plan action after a timing overrun. A late
  candidate cannot be made valid by publishing it after the deadline.

This is the requested fresh verification of the timing instrumentation and
current source on the practice simulator. The result is **not** a timing pass:
one cycle reached `32.771 ms`. The captured gate value and phase breakdown
make the tail measurable; more optimization is still required before another
SQP step is considered.

## Supervised practice run with eight Jacobian workers

Capture: `live_runs/mpc_8workers_live_20261021/run`. The current mounted
workspace was rebuilt in the development container; the MPC startup log
confirmed `Jacobian workers=8`. It used the established Xvfb/Vulkan practice
batch simulator, practice map/raceline, 3 m/s reference ceiling, diagnostics,
focused bag, runtime topic-policy guard, and collision/MPC-fault watcher.

- The MPC ran 530 cycles over 13.6 s of sensor time. It published 529
  commands: 480 `accepted_optimal`, 37 `accepted_feasible`, and 12
  `accepted_prior_plan`; five initial cycles waited for localization. There was
  no collision and no completed lap. The peak truth speed was `3.013 m/s`.
- For all 529 published-command gate records, elapsed time was
  `6.392/10.158/19.730 ms` p50/p95/max; **0/529** reached the 20 ms limit.
  Solver cycle was `6.207/9.259/18.953 ms` p50/p95/max. This is a live timing
  pass for this supervised, capped run, but the maximum had only `0.270 ms`
  of margin and does not establish a robust timing guarantee at racing speed.
- Odom arrived at a mean `25.03 ms` interval (39.95 Hz). All 529 actuator
  publication events were forwarded with zero sequence gaps; interface-receive
  to actuator-publish delay was `0.114/0.161/0.311 ms` median/p95/max. Bridge
  request/response latency had a `34.75 ms` maximum, so occasional response
  jitter remains even though no forwarded actuator sequence was lost.
- The watcher stopped on a terminal `rejected_nonlinear_rollout`, not a timing
  overrun or collision. The candidate failed the exact corridor check at stage
  16; the shifted prior plan also failed exact corridor validation at stage
  27. The controller correctly latched a fault and published no invalid
  replacement. This is evidence that MPC executes at eight workers, not that
  it can yet complete a lap. The resulting Socket.IO disconnect occurred
  during supervised shutdown, after the watcher stopped the simulator.
- The 8-worker competition YAML is retained on this 8-physical-core host.
  Repeat the live timing/cadence check at full operating speeds before treating
  the 20 ms maximum as robust; do not add another SQP pass based on this short
  run.

## Verification completed

- `colcon build --packages-select f1tenth_mpc --parallel-workers 4` passed.
- Explicit `mpc_bag_solver_replay` target rebuild passed.
- Focused `test_unity_source_plant` and `test_vehicle_model_source_rollout` passed (2/2), including new queue-column and full-horizon cache parity tests.
- Paired replay output parity passed for all 409 rows.
- `git diff --check` passed after the code and documentation edits.
- Latest incremental MPC build and full package test run passed:
  `colcon build --packages-select f1tenth_mpc --parallel-workers 4`; CTest 4/4.
- `tools/evaluate_mpc_bag.py` decodes the live bag; its legacy timing fields
  are explicitly identified as diagnostic-emission time, not gate time. It
  now prints the three slowest publication-gate rows with status, action
  source, node/QP/Jacobian/ADMM phases, and control-time prediction cost; the
  new live bag reproduces the `32.771 ms` outlier and its phase breakdown.

## Still required; do not infer completion

1. Build separate replay groups for accepted, iteration-limited, corridor-rejected, steering-bound, low-speed-launch, queued-brake, and high-slip/high-roll practice states. Current fixture does not cover the high-speed/high-slip cases.
2. The exact `publication_gate_elapsed_ms` field is verified in live bags. The
   8-worker capped practice run had no gate overrun, but its max was
   `19.730 ms`; demonstrate durable reserve in longer/full-speed practice
   runs and include ROS/Unity contention plus command-start lateness.
3. Current and iteration-limited candidates are published only after exact
   nonlinear validation. A shifted prior converged plan is also revalidated
   against the current state before use when the solve fails. The latest run
   exposed the remaining timing-fault case: no replacement is published after
   the gate is missed, and the retained actuator command persists until the
   development supervisor stops the simulator. Define and verify the bounded
   fault lifecycle before any unsupervised or multi-SQP live run; never publish
   an expired command.
4. The 4/8 worker comparison is complete offline and the first supervised
   8-worker practice run passed the `<20 ms` publication gate for all published
   commands. Continue checking that eight workers do not harm Unity/bridge
   cadence at higher speed and longer duration.
5. Continue handoff P1 with measured Jacobian/AD and Riccati kernels. Keep the current 22-coordinate QP and all plant states; do not coarsen physics. The tangent layout remains a parity-preserving candidate, not a validated SIMD/tail-latency win.
6. Only after full-cycle timing reserve and fault handling are established, prototype bounded line search / trust region, then optional second true SQP iteration (new nominal, fresh full-plant linearization, new QP, exact validation). Baseline remains one SQP per cycle; do not equate ADMM iterations with SQP iterations.
7. The new live capture decoded all 396 schema-11 plant-state samples and 303
   drive commands without generated Python message classes. Keep this capture
   as a reproducible decoder regression; investigate the older
   capture/topic discrepancy separately rather than generalizing it to the
   current decoder.

## Current decision

Keep the exact queue-seed and pass-transition optimizations: they preserve the plant and solver outputs while removing redundant work. Keep the aligned tangent layout as a parity-preserving performance candidate. Eight workers are configured on the current 8-core host: one capped live practice run measured a `19.730 ms` maximum with no publication-gate overrun, but then ended on exact corridor rejection before completing a lap. Do not enable multi-SQP yet. High-speed timing, lap completion, and production fault lifecycle remain unverified.

## Review handoffs: solver-quality and line-search follow-through — 2026-10-21

The two external review documents were followed as recommendations, not treated
as instructions to blindly raise iteration counts or publish unverified
controls. Replay data and the fresh supervised practice capture give these
results:

- **More ADMM iterations are not a win.** On the identical 528-state practice
  fixture, with eight Jacobian workers, warm sequence state and unchanged
  weights/rho, caps 100/150/200 produced exact-valid counts `509/506/504`.
  Core-cycle p50/p95/max were `4.224/6.140/7.785 ms`,
  `5.014/8.702/14.734 ms`, and `4.790/9.309/12.013 ms`. ADMM p95 rose from
  `2.444 ms` at cap 100 to `3.357/4.945 ms`; p99 was `3.698/5.969/7.791 ms`.
  Keep the production cap at 100. The replay fixture's speed max was only
  `3.23 m/s`, so these numbers do not cover racing-speed states.
- **Backtracking is a small offline recovery, not a live fix yet.** Sequence
  replay recovered `0/19` original exact-corridor rejects with alpha 0.5 and
  `1/19` when alpha 0.25 was also permitted (`509 -> 510` exact-valid rows
  overall). Independent-row replay recovered `3/18` with alpha 0.5, but its
  cycle p95 rose `6.488 -> 6.693 ms`; alpha 0.25 added no further recovery
  and p95 rose to `7.204 ms`. The two-alpha sequence replay added up to
  `2.109 ms` for a trial. The benchmark now reports `line_search_ms` so that
  extra cost cannot be hidden in total timing. Keep backtracking disabled in
  production: no runtime signal proves the relevant Unity command-effect
  epoch, and the observed benefit is modest and replay-regime-dependent.
- **The latest live fault was a deadline miss, not wholesale nonlinear
  rejection.** Capture:
  `live_runs/mpc_absolute_deadline_practice_20261021/run`. It contained 221
  MPC solve diagnostics and 220 published commands. There were 203
  `accepted_optimal`, 18 `accepted_feasible`, and one rejected cycle. One
  candidate passed exact nonlinear validation as `accepted_feasible` but was
  not published because callback-to-publication-gate time was `20.448 ms`
  against the strict `<20 ms` limit. The fault diagnostic reason was exactly
  `MPC cycle reached or exceeded its 20 ms budget`. The bag's gate p50/p95/max
  was `6.824/10.829/20.448 ms`; the overrun was one cycle, not “everything
  rejected.”
- The late candidate's phase data were node solve `19.965 ms`, QP/Jacobian
  `5.907/5.793 ms`, ADMM `6.786 ms`, exact candidate rollout max across the
  run `3.797 ms`, nonlinear validation max `0.054 ms`, and callback remainder
  `0.484 ms`. These measured tails sum to about `4.335 ms` after the expensive
  solve/ADMM work. The prior 3.5 ms post-solve reserve was based on older
  rollout maxima (2.675 ms); it is now raised to 5.0 ms. This is a measured
  compromise below the 6 ms setting that had deferred the first cold-start
  solve, **not yet a verified timing fix**. The 20 ms publication gate remains
  in force.
- This live attempt did not complete a lap. It had 566 truth samples over
  14.20 s, max truth speed `1.999 m/s`, and collision count `0 -> 9`. The
  watcher was armed after the simulator had already accumulated eight
  collisions, then stopped on the next observed collision. Therefore this
  bag cannot be used to claim immediate first-collision stop behavior or
  successful driving. It does establish the actual deadline failure and
  provides a decoder-valid diagnostic capture.

### Current decision and next gate

Keep ADMM cap 100; do not increase it. Keep nonlinear backtracking as a
disabled-by-default offline-capable feature until the live command-effect
epoch can be established. The 5 ms reserve change requires one supervised
practice verification with the watcher subscribed before simulator motion;
accept only if every published cycle remains below 20 ms and logs distinguish
an exactly feasible candidate, a revalidated prior plan, and an actual fault.
No multi-SQP change or raceline/physics change is justified by these results.
The controller still cannot promise to continue when neither its current
candidate nor a revalidated prior-plan prefix is feasible and on-time; that
case is an actual control fault, not a solver-iteration tuning issue.

### Follow-through: reserve verification and hard-state ADMM probe — 2026-10-21

This section supersedes the pending-reserve statement above. The 5.0 ms
post-solve reserve was built into the current development image and checked
in a new supervised **practice** capture,
`live_runs/mpc_reserve5ms_practice_20261021/run`. The watcher was armed before
simulator motion. Across 536 published commands, callback-to-publication gate
time was `8.765/12.077/17.998 ms` p50/p95/max, with no cycle reaching the
strict 20 ms gate. This verifies the reserve on a short, speed-capped run;
it is not a full-speed or full-lap timing guarantee.

The run did not complete a lap. After repeated exactly validated 75 ms
candidate/prior-plan prefixes, both available prefixes failed the exact
heading-aware corridor check at stage 2. The watcher stopped the simulator on
that MPC fault; the final bag records one collision at shutdown, so collision
ordering is ambiguous and this is not a collision-free result. The bag and
`solver_inputs.csv` decode and replay successfully. At the terminal state,
`e_y=0.8603 m`, `e_psi=0.6062 rad`, and `u=2.6097 m/s`. The cap-100 rejected
candidate's first exact failure was `e_y=0.9848 m` against an upper bound of
`0.9278 m` at stage 2. This identifies a near-boundary closed-loop
feasibility/recovery case; it does not by itself prove that no other legal
action could recover the state.

The same saved-state corpus was replayed while varying solver cap and weights.
The cap-100 baseline exactly accepted `515/537` rows, with core-cycle
`5.518/8.450/10.515 ms` p50/p95/max. Raising the cap to 150/200 yielded
`516/513` exact accepts; a temporary offline-only cap-500 probe yielded
`513/537`, `5.302/8.581/22.417 ms` p50/p95/max, and `18.944 ms` maximum ADMM
time. At the terminal stage-2 failure, cap 500 still returned an
iteration-limited candidate and missed the exact upper corridor bound by
`3.2 cm` (`e_y=0.9768 m`, upper bound `0.9444 m`). The temporary compile-time
probe ceiling was restored to 200; the competition profile remains at 100.
The added iterations therefore cost time without resolving this hard sample.

Doubling or quadrupling lateral/heading weights likewise did not resolve the
stage-2 failure and reduced exact acceptance on the replay corpus. The
offline alpha-0.5/0.25 backtracking variants also did not recover this sample;
their limited recoveries elsewhere do not justify live enablement. Keep the
exact corridor and command gates unchanged. The evidence points next to the
nominal/feasibility formulation and the closed-loop state approaching the
boundary, not to a generally inadequate iteration cap. Do not claim the
current controller can complete laps yet.

The Docker build had exposed that the dev-only replay source is omitted from
the Docker build context. `f1tenth_mpc/CMakeLists.txt` now adds that optional
target only when its source exists, so the normal package/image build no longer
fails for a missing diagnostic-only file. The 5 ms reserve and this packaging
fix were built into the development image; package tests and source checks are
now rerun after restoring the temporary cap probe and pass, as summarized
below.

### Path-tracking / weight diagnosis from the same capture — 2026-10-21

The controller-state cross-track error was small through most of the capture,
then grew rapidly at the end: over samples 0–401, maximum `|e_y|` was `0.142 m`;
over the last 22 samples it rose from `0.264 m` to `0.860 m`, while `|e_psi|`
rose from `0.202` to `0.606 rad`. These 22 states were in a sustained tight
negative-curvature section (roughly `-0.78` to `-0.56 1/m`) at `2.23–2.96 m/s`,
with reference speeds around `3.3–3.6 m/s`. In the baseline fixed-state replay,
21/22 current solves were exact-corridor rejected; all 11 samples already above
`|e_y|=0.5 m` were rejected. This is a real loss of tracking, not an isolated
single solver status.

To test the claim that speed tracking is crowding out path correction, the
offline replay tool now exposes a speed-tracking cost multiplier; it changes
only the two speed objective coefficients and does not alter the production
configuration. Against the same 537 saved states:

- Baseline exact accepts: `515/537`.
- Lateral-position weight ×8: `507/537`.
- Lateral-position and heading weights ×8: `512/537`.
- Both speed weights ×0.1: `517/537` (a gain of only two rows).
- Path weights ×8 and speed weights ×0.1: `516/537`.
- Path weights ×8 and both speed weights ×0.01: `524/537`; 10/22 of the
  late states with `|e_y| > 0.25 m` became exact-valid, including 1/11 above
  `0.5 m`. None of the late states above `0.75 m` became valid; the terminal
  `e_y=0.860 m` state still fails at stage 2.

This is stronger evidence that objective priority contributes: at the
terminal state, the one-stage speed-undershoot term is about `536` (weight
`u=500`), versus about `92` from lateral error and `12` from heading error
(weights `e_y=125`, `e_psi=31.25`). The extreme path-first replay also chose
stronger steering and generally less throttle in the late high-error region.
However, it still fails 12/22 late states, including the terminal state; exact
acceptance on fixed saved states does not prove better closed-loop tracking or
lap time. Its offline replay timing reached `16.5 ms` p95 / `26.9 ms` max on
the host, so this intentionally extreme setting is not a production candidate
as-is. Each row starts from its recorded state instead of rolling the tested
weights forward. Do not promote these exploratory weights to production from
these results alone.

The evidence supports the user's concern that the current objective is too
speed-dominant to preserve the line: lowering speed priority and substantially
raising path priority recovers some moderate/large-error fixed states. It does
not recover the terminal excursion, so weighting is not the only demonstrated
cause. The next test must compare closed-loop cross-track response with a
normalized path/speed priority change introduced before the excursion, and
inspect tight-turn nominal/path feasibility separately. Keep the 5 cm exact
corridor, 100-iteration production cap, and live weights unchanged until that
comparison is available.

### Uncapped native-speed practice trials — 2026-10-21

These development trials used the practice simulator, the existing mintime
trajectory `autodrive_practice_20260924_b_mintime_raceline.csv`, a 16 m/s
controller ceiling (above the trajectory's own maximum), and no startup speed
ramp. Thus the trajectory profile was not globally capped.

- `mpc_full_speed_overspeed_candidate_20261021`: `e_y/e_psi=3000/750`,
  `weight_u/overspeed=7.5/15`, no input-slew change. It failed at progress
  20.78 m, with `e_y=-0.186 m`, `e_psi=-0.306 rad`, and `u=4.413 m/s`; the
  next feasible controls alternated between steering limits and failed the
  exact corridor at stage 2. This established that overspeed cost alone did
  not prevent late-turn loss of line.
- `mpc_full_speed_lateral_weight_candidate_20261021`: `e_y=4000`,
  `e_psi=750`, with the same elevated overspeed cost and slew penalties as the
  4000/750 retry below. The first solve exceeded the 20 ms budget
  (`26.20 ms`) despite an exactly valid prior-plan prefix; the controller
  published nothing. This was a deadline/fault-path failure before a meaningful
  weight comparison, not evidence against the weights.
- `mpc_full_speed_rate_weight_candidate_20261021`: weights
  `e_y/e_psi/u/overspeed/steering-slew/throttle-slew = 3000/750/7.5/15/2000/100`.
  It reached progress 34.74 m and failed exact corridor validation at stage 2
  by about 5 cm. The state at the failure was already near the corridor edge.
  It had zero completed laps; the bag reports a collision at shutdown, so this
  is not a collision-free result.
- `mpc_full_speed_lateral_weight_retry_candidate_20261021`: changed only
  `e_y` to 4000. It failed at stage 1 by about 6 mm after 22 consecutive
  exactly validated command prefixes. At the failing prediction, `e_y=0.414 m`,
  `e_psi=0.271 rad`, `u=3.304 m/s`, with steering saturated at the left limit.
  Peak truth speed was 6.92 m/s; publication-gate p50/p95/max was
  `8.45/13.19/16.40 ms`. Zero official laps; the bag records a collision at
  shutdown, so collision ordering remains ambiguous.
- `mpc_full_speed_heading_weight_candidate_20261021`: changed only `e_psi`
  from 750 to 1500 while retaining `e_y=4000`. It advanced to progress 43.96 m,
  then failed at stage 2 with `e_y=1.236 m`, `e_psi=0.474 rad`, `u=4.453 m/s`
  against a `1.162 m` upper corridor bound. Peak speed was 7.42 m/s; p95 truth
  speed was 5.99 m/s. The controller logged 197 optimal, 22 feasible, 35
  exact-validated-prefix, and one prior-plan action; max publication-gate time
  was `14.51 ms`. It completed zero official laps and the bag records a
  collision at shutdown.

The terminal heading-weight failure maps to trajectory progress about 15.55 m,
where the native target is approximately 3.42 m/s and acceleration is
approximately `-2.90 m/s²`; the predicted vehicle speed was about 4.45 m/s.
The rejected candidate drove throttle down to the legal floor `q=0.01`, but the
MPC's optimization bounds prohibit `q=0`. Unity maps exactly zero to its hard-
brake branch; every positive q is the motor-drive branch. This explained why
that particular horizon could not reproduce the abrupt deceleration request,
but it was not a proof that the native-speed line cannot be driven without
q=0: a later q>0-only profile completed five uncapped laps, described below.
Do not treat q=0 removal as the sole cause of the earlier tracking failures.

An additional guarded code change lets a prior action continue past the
20 ms computation budget only after exact 75 ms current-state nonlinear
validation and only if its command-start epoch remains within the lateness
limit. It was built in the heading-weight run, whose maximum gate time stayed
below 20 ms, so this overrun-specific branch has not yet been exercised in a
live trial. Keep that distinction explicit.

The MPC `q=0` exclusion remains in force. A high overspeed penalty was first
tested to request the legal minimum `q=0.01` earlier, then a source-slip cost
and throttle-slew adjustment was tested to reduce powered-turn wheel spin.
Both retained the native speed profile and 16 m/s ceiling; neither adds a
speed cap or makes q=0 available. Results follow below.

The first `weight_u_overspeed=60` run is now recorded at
`live_runs/mpc_native_speed_tracking_overspeed60_20261021`. The reference
trajectory speed was left untouched; its sampled maximum is 8.070 m/s and
`integral(ds/v)` estimates 5.972 s/lap. The car completed two laps in 7.119 s
and 6.149 s (mean 6.634 s), reached 7.987 m/s, then collided on the next lap.
This is the first uncapped multi-lap progress in this sequence, but it is not a
successful five-lap result. At the final turn (`progress=87.52 m`, curvature
`-0.702 1/m`), the current state was `e_y=0.367 m`, `e_psi=0.393 rad`,
`u=3.742 m/s`; the reference was 3.415 m/s. The exact rollout rejected at
stage 2 after prior-plan prefixes shortened from 19 to 2 steps. The last
validated prefix used full-left steering and q=0.345. Its predicted rear-wheel
longitudinal slip was about 0.33–0.41, while yaw rate `-1.884 rad/s` lagged
the `-2.400 rad/s` reference. Thus this loss of line occurred during a
high-curvature, powered exit, not because it hit the native maximum speed.

The source-slip/throttle-slew candidate kept the path,
heading, speed objective, and q lower bound fixed; it raised source-derived
wheel-spin cost from 5 to 25 and reduced throttle-slew cost from 100 to 10.
This targets earlier drive reduction and less torque-induced wheel spin in
powered turns, without making q=0 available. The five-lap watcher continued
through occasional rejected full horizons while retaining exact-validated
candidate/prior prefixes; it stopped only on collision, stall, or target laps.

That candidate completed five uncapped laps without collision at the native
trajectory profile in `live_runs/mpc_native_speed_slip25_throttle10_20261021`.
Lap times were `6.849, 6.587, 6.410, 6.181, 6.261 s` (mean `6.457 s`), versus
`integral(ds/v)=5.972 s/lap`; peak speed was `8.022 m/s` against the native
`8.070 m/s` maximum and commanded q stayed at or above 0.01. MPC statuses were
1223 optimal, 63 feasible, 21 exact-validated candidate-prefix, and 4
exact-validated prior-plan actions; no fault ended the run. Publication-gate
p50/p95/max was `7.324/10.592/22.278 ms`; two cycles exceeded 20 ms but
published timely, exact-validated prior-plan actions.

An offline projection of `/current_map_pose` onto the trajectory measured
cross-track `|e_y|` RMSE/p95/max of `0.083/0.154/0.448 m`. By truth-speed bin,
2–4 m/s was `0.212/0.443 m` p95/max, 4–6 m/s was `0.063/0.109 m`, and 6–9
m/s was `0.022/0.040 m`. This is a map-pose path-tracking metric, not a Unity
truth pose score. The first two lap-count intervals contained the largest
tail; later intervals tightened. This establishes full-profile multi-lap
operation with q=0 still excluded, but it is not yet sub-5 s or within the
desired 5 cm tail.

The isolated lateral-weight-6000 trial used the same uncapped trajectory in
`live_runs/mpc_native_speed_lateral6000_20261021`; it completed two laps then
collided. Its path-error RMSE/p95/max was `0.185/0.183/1.271 m` versus
`0.083/0.154/0.448 m` for the successful 4000-weight predecessor, so the
increase is rejected. At the final exact-corridor failure (progress `98.951 m`,
curvature `-0.483 1/m`), state was `e_y=0.658 m`, `e_psi=0.713 rad`,
`u=3.535 m/s` against `3.789 m/s` reference; reference acceleration was
`-4.10 m/s²`. The valid prior action was `steering=-0.141 rad, q=0.131`, while
the new nominal wanted `steering=-0.331 rad, q=0.165`. Thus the immediate
failure was insufficient turn-in / growing heading error, not overspeed.

The steering-slew-1000 trial returned `weight_e_y` to 4000 and reduced only
`weight_steering_slew` from 2000 to 1000.
This tests whether the earlier rate penalty delayed steering response. It
retains nonzero steering-rate regularization, all other settings from the
successful five-lap profile, the original speed profile, and q>=0.01. Run
uncapped for five laps with collision monitoring.

The steering-slew-1000 trial was run in
`live_runs/mpc_native_speed_steering_slew1000_20261021`; it collided before
the first lap. At failure (`progress=43.182 m`), state was
`e_y=0.952 m`, `e_psi=0.907 rad`, `u=3.447 m/s`, and the new nominal requested
full-left steering `-0.524 rad` while the prior-plan prefix no longer passed
the corridor. This large direction change was not a timing problem (gate
max `10.804 ms`) and shows that simply halving steering-rate cost makes the
transient less stable. Reject this value; keep the successful 2000 steering
slew cost.

Increasing heading weight from 1500 to 2000 while restoring steering slew to
2000 was also tested in `live_runs/mpc_native_speed_heading2000_20261021`.
It collided after one completed lap; the final state was `e_y=0.325 m`,
`e_psi=0.549 rad`, `u=2.848 m/s` against 3.307 m/s reference. The last
candidate failed corridor validation at stage 2 after exact prior-plan prefixes
shortened, so this heading-weight increase is rejected too. The 4000/1500
profile with 2000 steering-slew cost is the current best measured setting and
has been copied into `mpc_competition.yaml`. A later production-default
confirmation did not repeat the five-lap result: it collided after two
completed laps. Treat this as the best observed profile, not a repeatably
validated one. q=0 remains unavailable to MPC.

The temporary iteration-ceiling experiment has been fully reverted. The
replay tool was rebuilt with its production ceiling, the MPC package rebuilt,
CTest passed `4/4`, the three touched Python tools passed `py_compile`, and
`git diff --check` passed. A direct `colcon build` attempt could not use the
build directory's stale container-absolute dependency prefix; direct CMake
build/test succeeded against the current workspace dependencies.

### Native-speed tracking weight follow-up — 2026-10-21

The goal of these trials was to improve tracking on the trajectory's unmodified
speed profile without making `q=0` selectable. MPC remains constrained to
`q>=0.01`; the 16 m/s ceiling is above the trajectory's 8.070 m/s maximum, so
it is not a speed cap on this track.

The production-default 10-lap attempt used the exact 4000/1500/2000 baseline
that had completed five laps in the overlay run. It collided after two
completed laps (reported lap times 7.031 and 6.185 s). At the terminal
diagnostic, progress was `87.125 m`, with `e_y=0.315 m`, `e_psi=0.599 rad`,
`u=3.694 m/s` against `3.240 m/s`, and `r=-0.552 rad/s`. The exact candidate
failed the corridor at stage 2; its predicted state at progress `87.306 m` was
`e_y=0.489 m`, `e_psi=0.719 rad`, `u=4.014 m/s`, and `r=+0.143 rad/s` in a
negative-curvature section (`kappa=-0.748 1/m`). The nominal requested
left-steering `-0.166 rad`, `q=0.185`, but the preceding cycles had accepted
short prefixes while their commanded steering/throttle diverged from that
nominal. Thus the same weight values did not reproduce a robust five-lap run;
run-to-run solver/initial-state variation remains unresolved.

Two isolated weight candidates were then tested on the same practice map,
trajectory, batch simulator, and no-speed-cap setup:

| Candidate | Outcome | Evidence | Decision |
|---|---|---|---|
| Steering slew `2000 -> 1500` | One completed lap, then collision near the beginning of the next | `live_runs/mpc_native_speed_steering_slew1500_multilap_20261021`; final progress `58.83 m`, `e_y=0.597 m`, `e_psi=0.412 rad`, `u=1.216 m/s`, `r=-0.389 rad/s`, while reference speed was about `3.3 m/s`. The nominal requested full-left steer and `q≈0.04`; the solve had reached 100 iterations and the exact candidate failed corridor at stage 1 after 29 consecutive short candidate-prefix actions. Gate timing max `16.153 ms`. | Reject. Lowering slew did not fix tracking and lost more than the 2000 baseline. |
| Yaw-rate weight `7.8125 -> 50` | Collision before first lap | `live_runs/mpc_native_speed_yawrate50_20261021`; terminal progress `42.400 m`, `e_y=0.623 m`, `e_psi=0.831 rad`, `u=5.592 m/s` vs `3.514 m/s`, `r=+1.153 rad/s` in negative curvature (`kappa=-0.676 1/m`). The nominal asked for left steer `-0.171 rad`, `q=0.158`; the output was a repeated candidate prefix at full-right `+0.524 rad`, `q=0.217`. ADMM hit 100 iterations with primal residual `1.22`; the exact corridor check failed at stage 2. Gate timing max `17.057 ms`. | Reject. Larger yaw-rate cost did not overcome a nonconverged candidate-prefix sequence and made the run substantially worse. |

The first 1500-slew launch attempt has an empty bag because the recorder had
not reached its recording state; it is not scored. The properly recorded
1500-slew retry first stopped at an MPC fault under the strict `mpc-lap`
watcher, then a separate run with the established collision-only lap watcher
was used to assess actual driving through intermittent solver failures. That
second run produced the collision result above. All runs kept `q>=0.01`; none
requested a full-brake command.

**Conclusion:** removing the full-brake decision did not make the native-speed
raceline inherently undriveable: the 2000-slew, `weight_r=7.8125` profile
already completed five laps once, with q always positive. But the failed
production repeat and the two weight candidates do not establish robust
multi-lap tracking. More weight escalation is not justified by this evidence:
the failures are dominated by 100-iteration/nonconverged solves followed by
repeated three-step candidate-prefix commands, including commands opposite to
the new nominal. Keep the production values unchanged for now; next fix and
validate the iteration-limited command/prefix behavior and its command-epoch
alignment, then re-run the same native-speed profile before another weight
sweep. No speed cap or q=0 option should be added.
