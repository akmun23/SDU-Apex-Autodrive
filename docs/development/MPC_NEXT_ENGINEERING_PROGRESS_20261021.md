# MPC Next Engineering Handoff — Progress

**Handoff:** `/home/akselmo/Downloads/SDU_Apex_MPC_Next_Engineering_Handoff_2026-10-21.md`  
**Baseline:** `56e2a3e679b92af70575a599c13bf75dae64593d`  
**Working branch:** `mpc/next-engineering-handoff-20261021`  
**Status:** implementation and evidence work is in progress; the handoff is
**not complete and the MPC is not release-ready**. Two supervised runs of the
requested IPOPT baseline trajectory produced no completed laps.

This log tracks the handoff in its prescribed PR-A through PR-J order. A test
or report is recorded as complete only for the exact scope it exercised. No
physics, runtime weights, or raceline has been altered in this work. Fine-
grained Riccati timing is an explicit build option. The strict `<20 ms`
publication rule was tightened after audit: the prior-plan over-budget
exception is removed; normal, candidate-prefix, prior-plan and open-loop
publications share one final monotonic deadline wrapper.

## Current outcome

PR-A has an offline-only frozen-QP capture/replay path and a saved-state
corpus representing two five-lap successes and three failure runs. The corpus
contains 125 exact 22-state, 2-input, 29-decision-stage QP fixtures. All
125/125 fixtures round-trip byte-for-byte and replay the recorded solver result
exactly on the same binary. An additional A/B against the exact handoff commit
replayed all 3,222 saved states with the fixture sink disabled; all 39
non-timing output fields matched exactly. This is strong saved-state output
parity, but **not** a live production timing test and does not recover
historical live ADMM memory.

The first frozen-QP comparison is mixed. Changing ADMM initialization changes
iteration counts and controls on several samples; some policies hit the
100-iteration cap, and there is no across-case win. No solver policy is
promoted. The independent OSQP/HiGHS oracle has useful converged/infeasible
results, but its 0.2-second per-QP budget is insufficient to classify the full
corpus. The later 5-second refinement has now run on all 71 fixtures that were
not already converged or proven infeasible: 59 solved and 12 still reached the
5-second limit. The long-budget output does not identify a production ADMM
policy win; see the machine-readable summary before drawing conclusions.

PR-B has a source-plant replay ladder and diagnostic-only ablations. The
historical **0.235 m** number is the controller's receding-horizon
logged-prediction comparison, not a single universal plant error. On an
earlier five-lap bag, fixed-input source replay was **0.05115 m XY RMSE** at
750 ms (p95 **0.12447 m**, max **0.23417 m**); 269/1,293 endpoints and 292/1,293
paths exceeded 5 cm. On two other full five-lap bags, using the current task
library and the same timestamp-reconstructed command-queue method, endpoint XY
RMSE was **0.06988 m** (native-speed run) and **0.06576 m** (8-worker priority
run), with p95 **0.15188/0.16398 m** and max **0.47613/0.30014 m**. Starting
each window with GT u/v/r barely changed the *aggregate* result on those two
long captures. By contrast, on the short IPOPT run, the 750 ms endpoint error
was **0.2867 m** from the production odom state but **0.0114 m** when each
window was initialized with GT u/v/r. That is a strong run-specific
initial-forward-speed signature, not a general explanation of the full-lap
tail. All queue epochs are still reconstructed from ROS/bridge timestamps;
Unity per-FixedUpdate applied-command IDs are absent. No source-plant
equation change is justified yet.

PR-C has one replay-only reference-conditioned nominal experiment. It does not
improve consistently across all five groups, so it remains diagnostic and is
not connected to production. No live candidate-prefix or prior-certified
suffix change is promoted.

PR-E now has a compile-time `SDU_APEX_MPC_RICCATI_PROFILE` switch, default OFF.
The profile-on and profile-off binaries replayed the same 3,222 saved states
from five success/failure groups. All 39 non-timing output fields matched
exactly on every row. Offline wall-time differences were mixed (profile-off
was faster in three cases and slower in two), so this is not evidence of a
timing win; it only establishes a behavior-neutral way to omit fine-grained
Riccati clock reads from a default build. Full live diagnostic-on/off A/B is
still required before deciding whether to keep the production default OFF.

## PR tracker

| PR | Scope | Status | Exit/remaining work |
|---|---|---|---|
| A | Deterministic evidence and frozen-QP fixtures | **Implemented; gate partial** | 125 fixtures, round-trip/replay parity, 5 groups. Exact baseline-vs-current non-timing output parity on 3,222 states. Run 5 was sealed before launch with source/image/map/raceline/overlay hashes. Remaining: paired live/load timing evidence. |
| B | Isolate 750 ms source-plant/Unity discrepancy | **Partial; source replay and failed live trials analyzed, cause unresolved** | Full-lap current-library endpoint XY RMSE is 0.051–0.070 m across three captures, with 0.124–0.164 m p95 and up to 0.476 m max. Two successes show GT-start only modestly changes aggregate errors; the short IPOPT run improves 0.2867→0.0114 m with GT u/v/r. This run dependence prevents claiming one universal odom-state cause. Exact Unity command-effect epochs remain unknown; no physics change is promoted. |
| C | Rejected-candidate suffix/recovery quality | **Offline multi-run chain analysis done; no promotion** | Five saved-state suffix comparisons plus seven closed-loop captures show chain length alone is not causal: a five-lap success had an 8-prefix chain; a failure had a 38-prefix chain with low residual. The second IPOPT standard-MPC probe ends after both candidate and prior-plan 75 ms checks fail at corridor stage 2. No validated suffix policy has passed held-out and live gates. |
| D | ADMM numerical convergence on identical QPs | **Expanded oracle done; no policy promotion** | 125 frozen QPs; all 71 unresolved/non-infeasible fixtures received a 5 s/250k OSQP attempt: 59 solved, 12 time-limited. Earlier seed/rho screen remains mixed; no solver policy promotion. |
| E | Performance profiling/redundant work and diagnostics A/B | **Partial; no timing win claimed** | Fine Riccati profiling is optional/default OFF. 3,222-row profile ON/OFF replay preserved all 39 stable fields; elapsed timing is mixed. No 3-pair diagnostic A/B or live-load WCET evidence. |
| F | Strict deterministic `<20 ms` callback-to-publication | **Strict pre-publication admission gate implemented; live release evidence partial** | Removed prior-plan over-budget exception and reject configured end-to-end budgets above 20 ms. A shared final wrapper samples monotonic time immediately before the publisher callback and is used by normal, candidate-prefix, prior-plan, open-loop stage 0 and every open-loop timer output. The timer measures from callback entry. Tests verify the publish callback is suppressed at/over 20 ms and invoked below it, and that the configured budget cannot exceed 20 ms. This does not measure completion/transport time after the ROS `publish()` invocation. Two earlier short practice trials had every published callback below 20 ms (max 11.56 and 12.06 ms), but neither completed a lap. No deliberate live overrun/fault-path test, scheduler/load campaign, deterministic/WCET proof, or production independent-stop verification yet. |
| G | Full-source-plant nominal feedforward/feasible speed profile | **Not started** | PR-B has not isolated a source-local plant residual from command/feedback timing; no feedforward or speed-reference change. |
| H | QP/nonlinear objective alignment and tracking | **Not started** | Freeze production weights until model/solver evidence; then controlled, held-out closed-loop tests. |
| I | Dead-code inventory and safe cleanup | **Inventory complete; deletion not justified** | Added `docs/development/MPC_CODE_INVENTORY.md` after checking CMake, Docker context, launch files, entrypoints and tooling. Every replay/evidence path is needed by A-H; no source branch has been proven unreachable, so no deletion-only change is safe yet. |
| J | 10 independent sessions × 5 clean full-speed laps | **Not eligible; no new lap completed** | Two supervised standard-MPC runs on the requested IPOPT baseline line ended before a lap: first collided; second faulted on exact corridor validation and was stopped by the watcher. PR-B/C/D/F/G/H acceptance and command-effect epoch requirements remain open. The 10×5 release campaign has not started. |

## Requested IPOPT baseline line: standard-MPC closed-loop probes

These are separate from the earlier one-shot 30-command frozen-horizon
experiment above. The current MPC—not a path follower or replay controller—ran
on the requested `ipopt_v1_baseline_raceline.csv`, practice map
`autodrive_practice_20260924_b.yaml`, diagnostics enabled, 8 Jacobian workers,
100 ADMM maximum iterations, no speed cap, and the established batchmode
simulator startup (graphics enabled; no GUI and no `-no-graphics`). Both
manifests were sealed before launch and record the same requested raceline
SHA-256 `35bed562a8b0611c48cbf2df8520aff091a3f212adf0eacd5f59e6b9f527fb73`,
map YAML SHA-256 `fcc4416fbab3c84aed6394e23d792f5533c914a777b743556b5f2b568807985d`,
dev image `sha256:770c97399f3af4efeb4d9faaa4894fc158d09d6a275ca60d4fd834ebd1be1ac5`,
and practice simulator image
`sha256:b4bbda41fdb1da7a2eadba4350ed3a0cb5020e7eb783454e78399ef5852dbd76`.
The checked-in baseline SHA is recorded in the manifests; the dirty mounted
source hashes are sealed there as well.

| Run | Useful evidence | Outcome |
|---|---|---|
| `live_runs/mpc_ipopt_v1_mpc_probe_20261021/` | 250 diagnostic rows; 244 published commands; 172 optimal full-horizon, 37 feasible full-horizon, 35 valid 75 ms candidate prefixes, 1 final rejection. Published-command gate p50/p95/max `6.617/10.047/11.556 ms`; 0/244 at or above 20 ms; no callback interval over 25 ms. A 23-command terminal prefix chain rose from `e_y=0.054` to `1.036 m`, speed `6.42→3.60 m/s`, had three steering sign reversals, max recorded residual `2.679`, and minimum prefix corridor slack `0.0227 m`. | **0 laps; collision reported at 6.385 s.** The detailed terminal failure stage was overwritten by the then-existing duplicate diagnostic publication bug, so it cannot be reconstructed from the final row. Do not infer its exact validation stage. Bag/report/manifest retained. |
| `live_runs/mpc_ipopt_v1_mpc_probe_20261021_retry2/` | 142 diagnostic rows; 137 published commands; 88 optimal full-horizon, 21 feasible full-horizon, 28 valid 75 ms prefixes, 1 final rejection. Gate p50/p95/max `6.903/10.192/12.055 ms`; 0/137 at or above 20 ms; no callback interval over 25 ms. Terminal detail survived after the diagnostic fix. | **0 laps; no collision recorded.** At 3.538 s, current state was `e_y=0.6308 m`, `e_psi=0.8263 rad`, `u=0.2214 m/s` versus `2.9494 m/s` reference. Candidate and prior-plan 75 ms exact checks both failed corridor at stage 2. Predicted `e_y=0.8052 m` exceeded the stage-2 upper corridor `0.6407 m` by `0.1645 m`. The watcher stopped Unity after the controller fault; a later bridge socket-loss report followed the intentional stop. This was not a collision-triggered abort. Bag/report/manifest retained. |

The second run therefore does **not** demonstrate that the controller can
continue safely through this state. It also does not justify weakening exact
corridor validation: neither available action was certified even for the
75 ms prefix. The failure occurs after an eight-prefix chain, with steering
near the negative limit and the current state already close to the right-side
corridor boundary. The earlier collision run and this fault run are two
failures, not a five-lap or repeatability result. The short-run timing samples
are useful and below 20 ms, but do not prove deterministic WCET.

### State/plant attribution on the retry-2 bag

Replayed the second probe's `run/` bag through the current source evaluator library
`build_task/f1tenth_mpc/libunity_source_plant_eval.so` (SHA-256
`bac0ca3657e7c3db7a290e40a91acc30dbf163e411d112b86b096b531294eb8d`). The
library under the default older `build/` path had a different SHA, so the
current library was passed explicitly. On this particular bag, the two
libraries produced identical reported metrics; the stale default did not
explain this run's error. Continue to pin evaluator-library hashes in future
reports.

The production odometry's forward-speed error on this short bag is substantial
(`u` bias `+0.1923 m/s`, RMSE `0.4103 m/s`, p95 absolute `0.9290 m/s`, max
`0.9823 m/s`); lateral-speed RMSE is `0.0423 m/s`, while yaw rate is gyro
passthrough and therefore has zero same-signal score error. With the timestamp-
reconstructed command queue and production odometry initial state, 119 valid
750 ms windows have endpoint XY RMSE `0.2867 m`, p95 `0.5808 m`, max `0.5927 m`;
54/119 exceed 5 cm. Reinitializing each window with **offline Unity-truth
u/v/r** while keeping the logged hidden plant state and future recorded
commands gives endpoint XY RMSE `0.0114 m`, p95 `0.0202 m`; the model's 750 ms
u RMSE is `0.0182 m/s`, `v` RMSE `0.0026 m/s`, and yaw-rate RMSE
`0.0179 rad/s`. This oracle result is diagnostic only and cannot be used online.
It shows the source plant is highly accurate on this capture conditional on
correct initial motion state; it does not by itself establish a production
odom repair.

That attribution does not generalize unchanged to the longer clean runs. On
the native-speed five-lap bag, source-plant endpoint XY RMSE is `0.06988 m`
(p95 `0.15188`, max `0.47613`) and GT u/v/r starts improve the aggregate only
modestly. On the 8-worker five-lap bag, it is `0.06576 m` (p95 `0.16398`, max
`0.30014`), again with only a small aggregate GT-start change. Therefore the
short capture's large forward-speed initialization contribution is
run-specific. The full-lap 5 cm target is not met, and the remaining error
cannot honestly be assigned solely to odometry initialization, Unity command
timing, or source-plant equations.

On the second probe, command-epoch offsets reconstructed from the bridge stream also do
not resolve the error: 750 ms XY RMSE is `0.3653 m` at offset `-1`, `0.3381 m`
at `0`, and `0.3181 m` at `+1` (118–120 windows). The logged-command-stamp
reconstruction yields `0.2867 m`. These are alternative timestamp-based
associations, not proof of Unity `FixedUpdate` application. No choice is close
to the 5 cm criterion, so no convenient offset or parameter change is selected.

The second probe also repeats the candidate-prefix analysis with detailed final
diagnostics. Across successful and failed sessions, prefix-chain length alone
is not a causal failure rule: a five-lap success contains an eight-prefix
chain, while a different failed session has a 38-prefix chain with low
residual. In the second probe, its final eight-prefix chain had minimum
validated-prefix slack `0.0337 m`; the *next* candidate and the certified
prior-plan prefix both fail at stage 2. This is evidence to retain the exact
gate and improve earlier full-horizon planning, not evidence for an unchecked
suffix or heuristic steering override.

### IMU-integrated forward speed: existing-data check, not promoted

The evaluator's causal midpoint-IMU candidate was scored from the first
production odom state without later truth correction. It improves forward
speed RMSE on the short IPOPT probe (`0.184 m/s` versus production odom's
`0.410 m/s`) but fails badly when recursively integrated over the successful
full-lap bags: `1.610 m/s` RMSE on the native-speed success and `1.757 m/s` on
the 8-worker success (versus production odom `0.0977/0.0542 m/s`). The
recovery/failure bag is also worse (`0.514 m/s` versus `0.234 m/s`). One-step
IMU increments initialized from each prior *GT* state are substantially more
accurate, but that is an acausal oracle condition. Therefore replacing the
source-plant forward-speed propagation with uncorrected IMU integration is
rejected; no blend gain or correction constant is promoted from these bags.
This negative result is retained to avoid repeating an attractive but
non-transferable short-run result.

### PR-F strict deadline correction

The independent audit first found that the old `cycle_time_is_valid()` path
allowed an exactly revalidated prior-plan action to publish after the 20 ms
callback budget if its predicted command-start lateness was ≤2 ms. That
contradicted the handoff's strict release rule. Removed the
`exact_prior_plan` exception. A second audit found that the gate preceded
some message/setup work and that the open-loop timer lacked a final gate. The
final `mpc_publish_before_deadline()` wrapper now samples the monotonic clock
immediately before invoking the ROS publisher and suppresses that invocation
unless elapsed callback time is strictly `< budget`. Normal, candidate-prefix,
prior-plan, open-loop initial and each timer output all use `publish_command()`
and thus this wrapper; the timer's start timestamp is taken at callback entry.
The runtime parameter rejects budgets above 20 ms, preventing an override from
relaxing the hard ceiling. This is a pre-publication admission check; it does
not measure ROS publish-return or DDS transport latency. Command-start
lateness is still diagnostic and is checked after publication,
so it cannot grant an over-budget command. The controller and focused test
target build. Tests exercise the publish callback itself at the exact 20 ms
boundary, above it, and below it. This establishes the code policy, **not** a
measured live overrun/fault lifecycle or deterministic WCET. No over-budget
command has been induced in a simulator after this change; the earlier two
captures had no published command at/over 20 ms.

## PR-A implementation details

### Code and evidence

- `f1tenth_mpc/include/mpc_rti.h`: fixed-size offline `MpcRtiQpCapture_t` and
  optional capture sink. Production callers leave the pointer null; no file I/O
  was added to the controller path.
- `f1tenth_mpc/src/mpc_rti.c`: copies QP, effective solver configuration and
  pre-solve ADMM state at the QP/solver boundary only when the optional sink is
  non-null; saves post-solve state, raw primal controls and solver diagnostics.
- `f1tenth_mpc/src/riccati_solver.c` and `include/riccati_solver.h`: a shared
  implementation supports an optional offline raw-primal-control output; the
  original API calls the same implementation with that output null.
- `tools/mpc_qp_fixture_format.hpp`, `tools/mpc_qp_fixture_replay.cpp`,
  `tools/mpc_qp_fixture.py`: explicit little-endian schema, finite/bounds/
  dimensions/EOF checks, SHA-256 sidecars, exact round-trip and solver replay.
- `tools/capture_mpc_qp_regression_corpus.py`: reproducibly rebuilds the
  corpus from five archived practice sessions, samples 25 saved states per
  case and links source run reports, full saved-state rows, command queues,
  trajectory/bag hashes, and current source ABI hashes.
- `tools/mpc_qp_oracle.py`: independent OSQP 1.0.4 sparse-QP oracle plus HiGHS
  LP feasibility check; pinned dependency is
  `tools/requirements_mpc_qp_oracle.txt`.
- `tools/validate_mpc_qp_corpus.py` and
  `tools/refine_mpc_qp_oracle_cases.py`: round-trip, parity, same-QP variants,
  objective/constraint metrics and longer oracle refinements.
- The `MpcRtiProblem_t.horizon` diagnostic field now reports 29 decision
  stages; the 30 physical samples and plant horizon are unchanged.

### Corpus composition

Persistent artifacts are under
`live_runs/mpc_qp_frozen_corpus_20261021/` (approximately 89 MB):

1. `success_priority_worker_tid_5lap`: five clean laps.
2. `success_native_slip25_5lap`: five clean laps at the native speed profile.
3. `failure_workers8_terminal_collision`: terminal wall approach/collision.
4. `failure_recovery_policy_collision`: long candidate-prefix chain ending in
   collision.
5. `failure_slew500_mpc_rejection`: MPC rejection/bridge fault case, no
   collision recorded.

Each is explicitly a **saved-state reconstruction**. Historical bags did not
record live ADMM z/y/rho memory or a sealed before-run working-tree/image
manifest. The fixture records the exact solver memory generated by this
deterministic saved-state replay, not the unavailable historical live memory.
The exporter was run after the fact and cannot retroactively seal the original
runtime. Command application to individual Unity `FixedUpdate` epochs remains
unknown where the source run did not expose those IDs.

### Validation run

Commands run from repository root:

```bash
cmake --build build_task/f1tenth_mpc --target \
  mpc_offline_qp_tools test_vehicle_model_source_rollout -j4
ctest --test-dir build_task/f1tenth_mpc --output-on-failure \
  -R 'vehicle_model_source_rollout|mpc_qp_fixture'
python3 -m unittest tools.test_mpc_qp_fixture -v
python3 tools/capture_mpc_qp_regression_corpus.py \
  --replay-binary build_task/f1tenth_mpc/mpc_bag_solver_replay
python3 tools/validate_mpc_qp_corpus.py \
  --corpus live_runs/mpc_qp_frozen_corpus_20261021 \
  --replay-binary build_task/f1tenth_mpc/mpc_qp_fixture_replay
```

The focused build and CTest passed (2/2), and Python schema tests passed (5/5).
The validation summary currently reports:

- 125 fixtures; 125 byte-identical round trips; 125 exact replay parity.
- OSQP statuses: 50 `solved`, 14 `solved inaccurate`, 57 time-limit,
  3 primal-infeasible and 1 primal-infeasible-inaccurate.
- HiGHS confirmed four QPs infeasible. These are feasibility findings, not
  failures of the vehicle model.
- For the 50 OSQP `solved` QPs, median/p95/max primal residuals are
  `9.43e-8 / 1.21e-6 / 1.52e-6`; dual residuals are
  `3.75e-7 / 2.27e-6 / 6.67e-6`.
- Only 20 QPs meet both oracle convergence and native-primal feasibility
  criteria for direct objective comparison. Their absolute native-vs-oracle
  objective gap is at most `0.0495`; this is not enough to certify the other
  105 QPs.
- Selected seed/rho tests cover 15 QPs across the five groups. Across variants,
  iteration-cap cases remain, residual and first-control differences vary by
  case, and no single policy dominates. Do not infer improvement from lower
  median iterations alone.

The summary and all per-fixture records are in
`live_runs/mpc_qp_frozen_corpus_20261021/validation_summary.json`.
`oracle_refinement.json` records the longer 5-second/250k-iteration refinement
subset if present; its results must be summarized separately from the
0.2-second screening pass.

### PR-A capture-disabled baseline A/B

Built `mpc_bag_solver_replay` from exact baseline `56e2a3e` in
`build_task/baseline-build`, then replayed baseline and current branch against
the identical five archived saved-state CSVs. The current executable had no
`MPC_REPLAY_QP_CAPTURE_DIR` set and `MpcRtiCycleConfiguration_t.qp_capture`
was null for every cycle. All **3,222/3,222 rows × 39 non-timing output
fields** matched exactly across these groups:

| Group | Rows | Baseline cycle p50/p95/max (µs) | Current cycle p50/p95/max (µs) |
|---|---:|---:|---:|
| recovery-policy failure | 197 | 5,350 / 7,609 / 17,382 | 5,411 / 8,466 / 9,460 |
| slew-500 rejection | 129 | 4,685 / 6,298 / 9,038 | 3,896 / 6,012 / 7,078 |
| workers-8 terminal failure | 40 | 6,632 / 8,231 / 8,605 | 6,273 / 7,424 / 7,520 |
| native-speed five-lap success | 1,311 | 3,954 / 6,464 / 9,748 | 4,005 / 6,430 / 10,888 |
| priority-worker five-lap success | 1,545 | 3,938 / 6,763 / 9,347 | 4,264 / 7,268 / 9,753 |

Timing is one paired offline replay per binary/case, and the table is sensitive
to order and host scheduling. It is **not** evidence of a production timing
win or strict `<20 ms` live compliance. The stable outputs prove the optional
capture changes do not change decisions when disabled for these saved-state
sequences. Machine-readable output:
`live_runs/mpc_qp_frozen_corpus_20261021/capture_disabled_baseline_ab.json`.

### PR-B source-plant diagnostic results

Replayed the latest practice five-lap capture with current
`unity_source_plant_eval` and added 50/75/125 ms score checkpoints to the
existing evaluator (the plant still advances at 1 ms and scores on 25 ms
epochs). It uses the logged Unity plant state and queued commands; no future
truth is consumed in the baseline rollout.

| Horizon | u RMSE m/s | v RMSE m/s | yaw-rate RMSE rad/s | yaw RMSE rad | XY RMSE / p95 m |
|---|---:|---:|---:|---:|---:|
| 25 ms | 0.08262 | 0.01632 | 0.12238 | 0.00254 | 0.00428 / 0.00724 |
| 50 ms | 0.07007 | 0.01468 | 0.11656 | 0.00416 | 0.00559 / 0.01007 |
| 75 ms | 0.07461 | 0.01452 | 0.11523 | 0.00562 | 0.00664 / 0.01312 |
| 100 ms | 0.06911 | 0.01472 | 0.11705 | 0.00687 | 0.00792 / 0.01600 |
| 125 ms | 0.07320 | 0.01438 | 0.11412 | 0.00800 | 0.00909 / 0.01916 |
| 250 ms | 0.06473 | 0.01418 | 0.11289 | 0.01200 | 0.01566 / 0.03593 |
| 500 ms | 0.06378 | 0.01405 | 0.11225 | 0.01784 | 0.03118 / 0.07680 |
| 750 ms | 0.06440 | 0.01420 | 0.11384 | 0.02196 | 0.05115 / 0.12447 |

At 750 ms, max absolute residuals are `u=0.23354 m/s`, `v=0.12631 m/s`,
`r=0.93942 rad/s`, yaw `0.11825 rad`, XY `0.23417 m`. 1,024/1,293 window
endpoints are within 5 cm; 1,001/1,293 complete paths never exceed 5 cm.
Therefore **269 endpoints and 292 paths fail** the requested 5 cm criterion.
This does not establish a source-equation defect: Unity application epochs for
individual commands are absent from the bag.

The timestamp-reconstructed queue gives virtually the same 750 ms XY RMSE
(0.05117 m). Bridge request-offset variants are worse (XY RMSE 0.20259 m for
−1, 0.1298 m for 0, and 0.1069 m for +1); none identifies a trustworthy
per-physics-step application schedule. Correct initial truth u/v/r changes XY
RMSE only 0.05115→0.04939 m. In contrast, supplying the *next recorded actual
steering feedback* as the interval command—an acausal oracle, never deployable—
changes 750 ms XY RMSE to 0.03127 m (p95 0.07638, max 0.13455), r RMSE to
0.05496 rad/s, and yaw RMSE to 0.01622 rad. This implicates steering
command/feedback phase as a major testable factor, but does not prove the
online model can know that future feedback. Future-heading re-anchoring also
shrinks XY RMSE to 0.0139 m, but is likewise an oracle, not a candidate.

Unity source cross-check: the practice prefab sets `SteeringRate=183.346`
deg/s; `TimeManager.asset` sets the fixed timestep to 1 ms. `FixedUpdate()`
sets `currentS` from the rack angle **before** calling `Steer()`, and
`Steer()` rate-limits the rack before assigning Ackermann wheel angles. The
source plant uses the same 183.345993 deg/s rate and 1 ms step/order. Thus the
bag's published `CurrentSteeringAngle` is a pre-advance rack sample, while
wheel state is subsequently advanced; the remaining concern is command
arrival/application association, not an identified incorrect steering rate or
missing steering substep. No physics parameters were altered.

Raw outputs: `source_plant_checkpoints_replay.txt` and `prb_*.txt` under
`live_runs/mpc_qp_frozen_corpus_20261021/`.

### PR-C offline reference-seed experiment

Added a `reference-seed` mode to `tools/mpc_bag_solver_replay.cpp`. It clears
only the saved nominal/control suffix before each solve, preserving the
recorded current state/actuator queue and ADMM z/y/rho memory; the existing
`sequence` mode remains the baseline. It is explicitly offline-only and does
not claim to reconstruct a certified previous plan or reproduce live command
publication. Replayed all five groups (3,222 saved-state rows):

| Group | Sequence full-horizon accepted | Reference-seed accepted | Sequence → reference-seed nonlinear rejects |
|---|---:|---:|---:|
| recovery-policy failure | 177/197 | 165/197 | 20 → 32 |
| slew-500 rejection | 129/129 | 127/129 | 0 → 2 |
| workers-8 terminal failure | 18/40 | 27/40 | 22 → 13 |
| native-speed five-lap success | 1,276/1,311 | 1,250/1,311 | 35 → 61 |
| priority-worker five-lap success | 1,484/1,545 | 1,444/1,545 | 61 → 101 |

Reference-seeding helped the terminal workers-8 replay and some residual tails,
but regressed both successful groups and two other failure outcomes. It is not
a global recovery fix and is **not enabled in production**. First controls can
also differ substantially (absolute steering delta p95 reaches the steering
limit in some cases). The saved-state sequence does not record enough of the
historical candidate/certified control plans to test the other two exact PR-C
builders. Full result:
`live_runs/mpc_qp_frozen_corpus_20261021/prc_reference_seed_replay.json`.

## PR-E fine-grained profiling toggle

`f1tenth_mpc/CMakeLists.txt` now exposes
`SDU_APEX_MPC_RICCATI_PROFILE` (default `OFF`). Enabling it compiles
`MPC_ENABLE_RICCATI_PROFILE` into `mpc_core`; disabling it removes only the
fine-grained `clock_gettime` measurements around Riccati factorization, RHS,
projection, rho updates, scaling and output-unscale. Solver deadline clocks,
status counters, safety checks and equations remain intact.

Profile-on (existing build, explicit `ON`) and profile-off builds replayed the
same five saved-state CSVs in sequence mode, with the same 8-Jacobian-worker
arguments. All **3,222 rows × 39 non-timing CSV fields** were identical. The
profile-off replay was faster in three groups and slower in two; neither
single-pass timing nor this low-overhead instrumentation difference warrants
a speed claim. Raw outputs and metadata:
`live_runs/mpc_qp_frozen_corpus_20261021/profile_toggle/`.

The manifest helper now selects the repository's rootless Docker daemon using
the same socket precedence as `tools/docker_env.sh`; it preserves an explicit
`DOCKER_HOST`. Unit tests cover explicit and discovered sockets and ensure
both image inspection and running-container checks use that daemon. A CLI
smoke seal successfully recorded image IDs/digests and the requested
practice-map/raceline/config hashes without starting containers. This smoke
artifact is under `build_task/` and is not a real-run manifest.

## Follow-up implementation and supervised practice attempts

### PR-D: expand long-budget oracle coverage

The original `refine_mpc_qp_oracle_cases.py` selected at most two unresolved
fixtures per case. The corpus validator has **71** fixtures that were not
converged by the short oracle screen and were not already proven infeasible
(this includes the `solved inaccurate` results). Added `--all-unresolved`,
which refines all such fixtures in stable `(case_id, fixture)` order while
preserving the original representative-screen behavior as the default. Added
unit coverage for both selection modes and registered it with CTest. The
full-corpus run is saved separately as
`live_runs/mpc_qp_frozen_corpus_20261021/oracle_refinement_all_unresolved.json`;
it does not overwrite the earlier 10-fixture representative result. It is
still an independent-QP numerical oracle, not a live plant or MPC improvement.

### Retry history for the requested IPOPT baseline raceline

The intended tests use the practice track; retry 3 is explicitly invalid
because it loaded defaults. No GUI and no `-no-graphics` simulator mode was
used. Each attempted launch was stopped and the Docker process list was
checked before continuing.

| Attempt | What actually ran | Result / interpretation |
|---|---|---|
| Initial capture | Recorder and Unity startup investigation only | No controller/simulator connection; retained small bag, not scored. |
| Retry 2 | Development stack, no verified simulator motion | Startup observation stalled before valid control; not scored as a driving run. |
| Retry 3 | Simulator and controller ran, but launch used misspelled environment-variable names | Wrong default map/trajectory/overlay loaded; collision. Invalid as a test of the requested IPOPT baseline. Manifest and bag retained; no retroactive relabeling. |
| Retry 4 | Correct practice map, `ipopt_v1_baseline_raceline.csv`, open-loop test overlay and diagnostics; all startup tools ready first | Rejected at stage 4 after 3 planned actions because open-loop envelope validation used a stricter float comparison than production publication. Difference was ~`1.44e-8 rad` at the steering bound, not a plant mismatch. |
| Retry 5 | Same requested assets and overlay; manifest sealed before launch; mounted source rebuilt by the development entrypoint; recording and collision supervisor active before batchmode Unity | Fixed envelope path executed **30/30 consecutive planned commands**. Collision was reported 0.713 s after the plan began, just before the 750 ms fixed horizon ended. No lap completed. This is a failed supervised one-shot probe, not full-speed MPC tracking or a clean rollout acceptance. Bag/report/manifest are retained at `live_runs/mpc_ipopt_v1_frozen_horizon_20261021_retry5/`. |

Retry 5's manifest confirms practice map
`autodrive_practice_20260924_b.yaml`, the requested IPOPT baseline trajectory,
`mpc_open_loop_horizon_test.yaml`, diagnostics enabled, four Jacobian workers,
batchmode simulator and supervisor. `AUTODRIVE_REBUILD=0` skips the image
rebuild, but `tools/start_dev.sh` sets `SDU_APEX_BUILD_MOUNTED_SOURCE=1` in that
mode; the mounted workspace was rebuilt in the development entrypoint before
launch. This verifies the tested source tree was compiled, while the sealed
image digest still identifies the devkit base image used for that build.

The shared drive-command envelope helper now applies the same existing
`1e-6` numerical tolerance and exact-bound clamp to production and the
one-shot test publisher. It does not relax physical steering/throttle limits
or permit `q=0`. The dedicated test covers tiny floating-point overshoot,
material out-of-range commands, zero throttle, NaN/Inf and invalid limits.
Retry 5 verifies the test publisher now emits its whole sequence instead of
false-rejecting the fourth stage.

Retry 5 has 45 actual MPC diagnostic cycles. All 45 command publication gates
were below 20 ms (p50 9.951, p95 12.540, p99 14.729, max 15.226 ms), with no
callback interval over 25 ms. This was a **test-specific** four-worker,
50-iteration open-loop overlay run; it is not proof of the production
configuration's strict timing gate or WCET. Solver status was iteration-cap
for all 45 solver cycles; 17 outputs were full-horizon accepted and 28 were
candidate prefixes. The full plan had 11 steering sign reversals and 26.7%
steering saturation. No lap-time or repeatability result exists.

The first collision report is at 1.952729 s from bag origin; the open-loop
plan began later and the supervisor observed collision 0.713 s after that
plan start. Unity's physical contact instant can precede the collision topic
receipt. Therefore the 0–650 ms scoring subset (ending about 38 ms before the
report) is **not guaranteed collision-free**. Its errors are: XY RMSE
0.07092 m / p95 0.12100 / max 0.13658 m; yaw RMSE 0.05566 rad / max 0.08011;
forward-speed RMSE 0.13901 m/s / max 0.18481; lateral-speed RMSE 0.00155 m/s;
yaw-rate RMSE 0.14896 rad/s / max 0.30126. This subset fails the 5 cm goal and
cannot isolate a clean source-plant residual. The complete horizon's
collision-contaminated score is not used to assess plant accuracy.

The bag reports **zero known Unity command-effect epochs**. Exact ROS/bridge
stamp matches prove which message was published/received, not which
`FixedUpdate` consumed it. No new plant coefficient, model, race weight or
trajectory was promoted from these runs.

### PR-I deliverable

`docs/development/MPC_CODE_INVENTORY.md` maps production CMake targets to
launch/image paths, identifies development-only fixtures/replayers/oracles,
and records the result of the reachability check. Cleanup is staged: no file
was deleted because the analysis/replay tools are live dependencies for
unpassed PR-A–H exit gates. This meets the inventory step; it does not claim
that dead-code deletion or a diagnostics-node split has been completed.

## Current exact next steps

1. PR-A focused tests pass; the IPOPT MPC manifests are sealed before launch.
   Remaining acceptance is a paired live/load timing comparison and full
   frozen-QP solver-quality evidence. Offline parity does not satisfy live
   timing or numerical-quality gates.
2. Continue PR-B only with a causal attribution target: existing command
   timestamp/bridge offset sweeps did not remove the error, and initial-state
   truth helps substantially only on the short second probe, not across the clean five-lap
   captures. A per-Unity-FixedUpdate applied-command ID is not present in the
   captured topic contract. Either acquire it through a development-only
   instrumented simulator build, or bound timing uncertainty from captured
   request/response data; do not claim a specific effect epoch or edit plant
   parameters to compensate.
3. PR-C: retain reference-seed as an offline comparator only. New captures
   must record full rejected QP suffix, published float32 prefix, current
   certified plan/cursor and command-effect timing to compare the other
   builders. No recovery change is justified by the current saved states.
4. PR-D's 71-fixture long refinement completed (59 solved, 12 time-limited).
   Analyze the 12 hard cases with the exact same QP and independent feasibility
   checks if proposing a solver-policy candidate; do not promote settings from
   the mixed existing seed/rho screen.
5. PR-E/F still require paired diagnostic/live-load experiments, scheduler
   evidence, a live over-budget fault-path check, and a verified production
   independent actuator-stop lifecycle. Strict `<20 ms` rejection is now
   implemented and unit-tested. The two short probes passed observed timing
   but failed to complete a lap; this is not timing-release evidence.
6. G/H depend on PR-B and C/D evidence. The PR-I dependency inventory is
   complete, but no exact deletion candidate is proven dead; deleting the
   retained evidence path would damage open PR-A–H validation. J requires all
   previous exit criteria and the 10×5 session release campaign; it remains
   not started because current MPC failed both requested-raceline probes and
   the upstream release gates are open.

## Known caveats and test hygiene

- The independent oracle's short budget was intentionally chosen for screening;
  status `run time limit reached` is not a proof of infeasibility. The 5 s
  refinement still timed out on 12 difficult QPs; no solver policy can be
  promoted from the incomplete oracle set.
- Source-plant-vs-Unity 750 ms error has now been measured on multiple bags,
  but no result establishes the desired ≤5 cm full-lap bound or exact Unity
  command-effect epochs. There is no demonstrated closed-loop repeatability
  on the requested line and no faster lap from this work.
- The task-created `build_task/`, `install_task/`, and `log_task/` validation
  directories (about 334 MB total) were removed after the final build/tests;
  their pass counts and outcomes are retained below. They can be regenerated.
- The code inventory is complete; source deletion remains unjustified because
  retained replay/evidence paths are still required by open PR-A–H gates. The
  independent sub-agent's first audit found the configurable deadline-budget
  bypass and stale wording. The budget bypass and wording are corrected; the
  sub-agent re-audit confirmed the hard cap, shared publisher path, and that
  PR-A–J are accurately marked implemented/partial/not started. It also
  confirmed no live deadline/fault-stop evidence or release campaign exists.
  Its final check confirmed the PR-I inventory/audit wording and configured-
  budget fault messages are now accurate.

## Verification of the current handoff state

After the strict-deadline change, the current `build_task/f1tenth_mpc` CTest
suite passed **9/9**, including the publisher-dispatch deadline tests. The
focused Python reporter/experiment/fixture/oracle/manifest suite passed
**28/28**, and `tools/verify_runtime_topic_policy.py` returned `PASS`.
`git diff --check` is clean. These verify the current code and evidence tools;
they do not convert the two failed practice probes into a passing closed-loop
result. `docker ps` was empty after the probes and all recording and
supervisor processes were closed.

## 2026-10-22 — throttle epoch correction and five-lap practice validation

This section supersedes the earlier statement that no clean five-lap source-
plant rollout had passed the local 5 cm horizon gate. It does **not** claim a
continuous five-lap open-loop prediction or a completed MPC/racing validation.

### Root cause and production correction

The production odometry plant had propagated the just-completed 25 ms interval
using the latest throttle feedback callback. That callback is a response
snapshot at the interval endpoint and can already describe the next throttle
request. At a command transition this retroactively applies the wrong torque
to the interval that just elapsed. Steering propagation already used the
delayed command queue. The correction makes throttle consistent: propagate
the interval with `delayed_throttle_commands_norm_[1]`, and do not use
headerless endpoint feedback as the interval input. Feedback remains subject
to the existing finite/freshness checks; this change does not remove it from
the observation contract or alter the vehicle physics.

Evidence from the five-lap bag, reconstructed with Bridge-sent request values:

| Same-state one-step command phase | Forward-speed RMSE | Lateral-speed RMSE | Yaw-rate RMSE |
| --- | ---: | ---: | ---: |
| Older queue slot 1 | 0.07089 m/s | 0.01553 m/s | 0.12736 rad/s |
| Correct interval command, queue slot 2 | 0.01216 m/s | 0.00066 m/s | 0.00464 rad/s |

The command-phase change reduces one-step yaw-rate RMSE by about 96% and
forward-speed RMSE by about 83% in this capture. It is therefore not accurate
to attribute the previously large rollout error solely to tire/contact
equations: a command/response phase mismatch was a major cause. A second
same-bag diagnostic that substitutes retrospective endpoint throttle
feedback gives worse 750 ms results (XY RMSE 0.01039 m and 7/1551 windows
exceeding 5 cm at an intermediate point), versus command-driven propagation
(XY RMSE 0.01008 m and 0/1551 exceeding 5 cm). This supports using the queued
command, not endpoint feedback, for propagation.

### Code and regression coverage

- `f1tenth_localization/src/unity_source_odom_runtime.cpp`: elapsed throttle
  now comes from delayed queue slot 2; comments explain why endpoint feedback
  is retrospective.
- `f1tenth_localization/test/test_source_derived_odom_runtime.cpp`: the
  regression test asserts that changing only endpoint throttle feedback does
  not change any state propagated over the completed interval.
- The development CSV follower needs Unity's `q=0` hard-brake mode. The
  actuator interface now has a validated configurable minimum throttle,
  defaulting to the production/MPC lower bound `0.01`; the launch overrides
  it to `0.0` only for development CSV following with MPC disabled. This was
  necessary because an earlier capture rejected all 34 zero-throttle brake
  commands. It does not enable zero throttle as an MPC decision or change the
  competition default.
- Production odometry target and localization tests were built in the ROS
  Humble development image. The focused C++ runtime binary passed **8/8**;
  actuator-interface tests passed **5/5** in the same ROS environment.
  `tools/verify_runtime_topic_policy.py` returned `PASS`; the real validation
  run used the dev-only CSV/ground-truth follower, with MPC disabled. No
  ground-truth input was added to odometry.

### Real practice capture

Bag: `live_runs/odom_throttle_epoch_fix_practice_20261022_run1/run`.
The established practice batch-mode simulator ran with the updated development
container, CSV path follower (not MPC), and full diagnostic recording. Five
laps completed with zero collisions; all **1584/1584** actuator publications
were forwarded, with no sequence gaps. Lap-time topics contained no finite
completed-lap times, so this is not a racing-time result. The bag is about
17 MB and retained for reproducibility.

Production odometry over the run (1583 truth matches):

- forward speed: RMSE 0.0126 m/s, p95 absolute 0.0244, max 0.1356 m/s;
- lateral speed: RMSE 0.0007 m/s, p95 absolute 0.0010, max 0.0112 m/s;
- yaw rate is read from the gyro, so its zero comparison error is not an
  independent prediction result;
- integrated relative pose drift: 5.5 cm at run end, 7.9 cm p95, 8.2 cm max.

Source-plant rollout, each window initialized from its logged
`UnityPlantState`, then propagated for 750 ms using the reconstructed queued
commands with no future state correction (1551 valid windows):

- forward-speed RMSE 0.0110 m/s (p95 absolute 0.0236, max 0.0480);
- lateral-speed RMSE 0.00066 m/s (p95 0.00104, max 0.00656);
- yaw-rate RMSE 0.00455 rad/s (p95 absolute 0.00801, max 0.04314);
- yaw RMSE 0.00229 rad (p95 absolute 0.00480, max 0.00943);
- XY endpoint RMSE 1.01 cm, p95 2.24 cm, max 3.89 cm;
- the maximum position error anywhere along each 750 ms path was at most
  4.45 cm: **1551/1551** paths passed the 5 cm gate.

The remaining XY discrepancy is not all plant dynamics: comparison against
the integrated recorded Unity twist attributes 0.54 cm XY RMSE / 1.07 cm p95
to disagreement between 40 Hz pose increments and twist integration, and
0.71 cm RMSE / 1.51 cm p95 to plant-vs-twist-integral differences. Yaw-rate
tail errors are larger when wheel/body speed mismatch is high (p95 0.00942
rad/s versus 0.00483 below 0.5 m/s mismatch), so wheel-spin/tire-contact
residuals remain a plausible contributor; the logs have no independent
per-wheel Unity angular-speed truth to prove that attribution.

### What this establishes—and what it does not

Established on this practice profile and capture: the interval command
association was wrong when endpoint throttle feedback was used; the corrected
production propagation materially improves one-step state prediction, and
the source plant's 750 ms rolling-window XY predictions stay under 5 cm in
all recorded windows, up to 7.13 m/s. This is evidence of useful rollout
accuracy in the tested envelope, not proof across all steering/speed regimes.

Not established: a continuous 750 ms+ forecast from one lap-start state (the
scored windows are reinitialized from logged plant state), under-5-cm
multi-lap odometry drift, MPC convergence/tracking, maximum-speed or
independent-raceline behavior, or a 1:1 reproduction of PhysX's native private
WheelCollider/contact solver. The one-run odometry pose maximum remains above
5 cm, despite low velocity errors. Continue investigating that drift and
high-wheel-mismatch yaw tails before declaring odometry solved. No new tire
coefficient was fitted or promoted.

The current simulator/controller containers and recorder were stopped by the
five-lap watcher. The run produced no collision and is preserved above.
