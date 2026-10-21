# MPC full-physics performance / multi-SQP progress — 2026-10-21

This progress log follows `SDU_Apex_Full_Physics_MPC_Performance_and_Multi_SQP_Handoff_2026-10-21.md` from reviewed head `0949ce3012a72848510405c3db98af21d1fec77e`. The handoff is a plan, not proof that its proposed work has already been done.

## Scope preserved

- No Unity/source-plant equations, tire/contact behavior, 1 ms plant integration, 30 x 25 ms horizon, drive-only MPC throttle bounds, map/reference, or nonlinear footprint/corridor gate were changed.
- No second SQP iteration, line search, or MPPI was introduced. Earlier supervised practice runs ended on controller faults without collision; the newest aligned-tangent practice capture ended on a collision and was stopped by the watcher. No capture in this performance branch completed a lap.
- The replay fixture is practice-track data, but its measured vehicle speed is limited. This is not high-speed/high-slip validation.

## Implemented

1. **Repeatable saved-input benchmark and phase instrumentation.** The working tree records nominal seed/rollout-A/reference-resample/rollout-B, 29-stage Jacobian wall and worker CPU, slowest stage, sensitivity and finite-difference counts, ADMM/Riccati subphases, candidate rollout, and exact validation. `tools/benchmark_mpc_replay.py` preserves raw CSV and machine/build/fixture metadata as JSON. The node now records publication-gate elapsed time separately from later diagnostic-emission time. The replay benchmark itself remains offline core timing.
2. **Remove exact-zero historical-command Jacobian work.** The plant consumes delay slot 2; slot 1 is only overwritten by slot 2 during the exact queue shift. The slot-1 derivatives are therefore exactly zero. Those two columns are now marked analytic/zero instead of running full-plant finite differences. A regression perturbs both old command slots and verifies no predicted state or physical output changes.
3. **Exact physical-transition reuse between nominal passes.** Pass A and Pass B use the same initial physical state and command sequence. The source-derived physical plant does not depend on Frenet curvature; only path projection/progress does. Pass A now retains its full stage transition. Pass B reprojects the cached body displacement/yaw through the newly sampled path curvature, avoiding a duplicate 30-stage 1 ms plant rollout. No physical state or validation is skipped. A 30-stage test compares cached reprojection to a fresh complete-plant pass B.
4. **Remove cold-start plant work used only to shift queued throttle.** The cold nominal seed previously simulated all 30 stages and 25 plant substeps per stage solely to read `delayed_throttle_command_2` after each queue shift. The queue recurrence is deterministic and independent of tire/body state; the seed now computes that exact recurrence directly. The actual full-physics nominal rollouts remain unchanged. A regression compares the new seed sequence against the previous source-plant queue evolution for initial queued throttle `0`, `0.2`, and `0.8`.
5. **Align the forward-mode tangent storage.** `Dual20` now begins with its 20 float tangent lanes in a 16-byte-aligned object, followed by the double primal. This preserves double primal arithmetic and all 20 derivatives while presenting tangent vectors contiguously to the compiler. The matching 409-row replay has bit-identical non-timing outputs and the full MPC package tests pass. Two portable replay samples improved cycle p50 from the queue-seed reference `11.841 ms` to `10.034/10.456 ms`; p95 was `12.538/14.097 ms` versus `14.089 ms`, while max varied (`18.938/16.754 ms` versus `18.088 ms`). Treat this as a promising exact-layout optimization, not yet a proven tail-latency improvement; the live gate miss below is decisive for real-time acceptance.
6. **Make failure-path evidence explicit.** After analyzing the live timeout, MPC diagnostics now record whether a previous-plan suffix was attempted, passed exact rollout/corridor validation, and then failed the timing gate, including validation duration. They also distinguish a published command from a valid-but-unpublished candidate. The evaluator includes these fields in its slowest-cycle report. This source change built and passed tests after the captured run; its new JSON fields still need confirmation in a subsequent bag.

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

An offline worker-count sweep on the same fixture, after the exact optimizations, preserved statuses and all first actions across 1/2/4 workers. The 4-worker run was fastest on this Ryzen 9 5900HS replay (cycle p50/p95 12.21/14.59 ms); 2 workers measured 18.29/20.67 ms and 1 worker 18.07/33.03 ms. This supports keeping four workers for this machine and fixture, not a universal deployment claim; ROS-load scheduling remains untested. Raw worker artifacts are `practice_runtime_priorplan_workers{1,2}_replay.csv` with matching `*_benchmark.json` files in `live_runs/mpc_performance_20261021/`.

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
2. The exact `publication_gate_elapsed_ms` field is verified in a fresh live
   bag. Reduce the live tail so **every** cycle is below `20 ms` with reserve;
   include realistic ROS/Unity load and track command-start lateness. One
   passing p95 is not sufficient.
3. Current and iteration-limited candidates are published only after exact
   nonlinear validation. A shifted prior converged plan is also revalidated
   against the current state before use when the solve fails. The latest run
   exposed the remaining timing-fault case: no replacement is published after
   the gate is missed, and the retained actuator command persists until the
   development supervisor stops the simulator. Define and verify the bounded
   fault lifecycle before any unsupervised or multi-SQP live run; never publish
   an expired command.
4. Benchmark 1/2/4 Jacobian workers under realistic concurrent workload. The current corpus has only been replayed with four workers.
5. Continue handoff P1 with measured Jacobian/AD and Riccati kernels. Keep the current 22-coordinate QP and all plant states; do not coarsen physics. The tangent layout remains a parity-preserving candidate, not a validated SIMD/tail-latency win.
6. Only after full-cycle timing reserve and fault handling are established, prototype bounded line search / trust region, then optional second true SQP iteration (new nominal, fresh full-plant linearization, new QP, exact validation). Baseline remains one SQP per cycle; do not equate ADMM iterations with SQP iterations.
7. The new live capture decoded all 396 schema-11 plant-state samples and 303
   drive commands without generated Python message classes. Keep this capture
   as a reproducible decoder regression; investigate the older
   capture/topic discrepancy separately rather than generalizing it to the
   current decoder.

## Current decision

Keep the exact queue-seed and pass-transition optimizations: they preserve the plant and solver outputs while removing redundant work. Keep the aligned tangent layout as a parity-preserving performance candidate, but do not count it as a reliable tail-latency win until paired timing confirms it. Do not enable multi-SQP yet. The live timing gate missed once at `32.771 ms`; no high-speed convergence, lap completion, or safe production fault lifecycle has been demonstrated.
