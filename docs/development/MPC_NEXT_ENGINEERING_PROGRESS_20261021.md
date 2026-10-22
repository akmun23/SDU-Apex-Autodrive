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

## 2026-10-22 — production MPC five-lap validation after odometry correction

This is the first five-lap run in this continuation with the production MPC
enabled after the throttle-epoch odometry change. It supersedes the prior
statement that no new lap had been completed. It does **not** pass the
requested high-accuracy criteria.

### Run setup and completion

- Bag: `live_runs/mpc_odom_epoch_fix_practice_20261022_run1/run`.
- Practice map and default practice mintime trajectory were loaded from the
  normal launch configuration; no raceline, MPC weights, or physics parameters
  were overridden.
- `sdu-apex-autodrive:dev` rebuilt mounted workspace source before launch;
  logs confirm production odometry and MPC started. MPC was enabled with the
  standard 8 Jacobian workers, 100-iteration cap, and diagnostics enabled.
- Established practice batch-mode simulator and collision/MPC-fault watcher
  were used. Five laps completed, collision count stayed zero, and the watcher
  observed no MPC fault/rejection status. All containers and recorder stopped
  after the target-lap guard.
- Simulator lap times: **6.842, 6.245, 6.079, 6.045, 6.250 s**; mean **6.292
  s**, best **6.045 s**. This is a successful five-lap run, not a sub-5-second
  result.
- Ground-truth speed reached **7.989 m/s** (p95 7.197 m/s).

### MPC solve and output behavior

- Statuses: 1237 `accepted_optimal`, 29 `accepted_feasible`, 10
  `accepted_candidate_prefix`, 6 startup `waiting_for_localization`; no
  rejected/fault cycle was recorded.
- Exact-valid candidates: **1266/1276**. ADMM converged on 1246 solves; 30
  reached the iteration cap. Converged iterations had p50/p95/max **9/17/100**.
- End-to-end publication gate: p50/p95/max **6.138/7.979/14.642 ms**;
  **0/1276** at or above the 20 ms budget. Solver-reported cycle p50/p95/max
  was **5.900/7.675/14.087 ms**.
- This confirms that the configured controller ran on time in this capture;
  it does not establish worst-case timing under other host loads.

### Accuracy: odometry and source-plant rollout

Production odometry versus Unity truth (1283 matched samples):

- forward speed RMSE **0.0329 m/s**, p95 absolute **0.0451**, max **0.1830**;
- lateral speed RMSE **0.0060 m/s**, p95 absolute **0.0076**, max **0.0670**;
- yaw rate comes directly from the gyro, so its zero error is not an
  independent model-accuracy measurement;
- integrated XY error relative to the initial pose: **27.0 cm** at the end,
  **26.9 cm** p95, **27.3 cm** maximum. This fails the 5 cm full-run pose
  objective.

Independent fixed-input source-plant windows (each starts from its logged
`UnityPlantState`; Bridge-reconstructed interval commands are held as the
recorded sequence for 750 ms; no future state correction; 1252 valid windows):

- forward speed RMSE **0.01742 m/s**, p95 absolute **0.02759**, max **0.11930**;
- lateral speed RMSE **0.00081 m/s**, p95 **0.00165**, max **0.00726**;
- yaw-rate RMSE **0.00546 rad/s**, p95 absolute **0.01134**, max **0.05363**;
- yaw RMSE **0.00261 rad**, p95 absolute **0.00515**, max **0.01845**;
- 750 ms XY endpoint RMSE **2.124 cm**, p95 **3.400 cm**, max **13.247 cm**;
- maximum error along the whole 750 ms path exceeded 5 cm in **52/1252**
  windows. By starting steering magnitude, failures were 50/607 below 0.10
  rad, 2/309 from 0.10–0.20 rad, and 0/336 at or above 0.20 rad. The worst
  windows start near straight steering and show about 0.10–0.12 m/s forward
  speed underprediction by 750 ms. This points toward a remaining
  longitudinal/throttle-transient mismatch in this run; it is not proof of a
  specific hidden PhysX effect.

The previous five-lap CSV-follower run had much lower errors, but it was a
different command profile and speed distribution. It is not a paired
comparison. The new MPC run shows that accuracy does not automatically
transfer from that follower profile to the MPC's throttle/steering sequence.

### Interpreting MPC's logged predicted horizon

The MPC diagnostic predictions compared to later Unity truth report 25 ms
errors of **0.0637 m XY RMSE**, **0.1594 m/s forward-speed RMSE**, **0.2433 m/s
lateral-speed RMSE**, and **0.0932 rad/s yaw-rate RMSE**. Pairing uncertainty
is p95 **9.59 ms**. At 750 ms, logged-plan comparison is 0.1778 m XY RMSE,
0.3545 m/s forward-speed RMSE, and 0.4614 rad/s yaw-rate RMSE. These are not
clean fixed-input plant scores beyond the already queued commands: the MPC
replans every 25 ms, so subsequent real inputs differ from a frozen predicted
horizon. Use the preceding source-plant replay for fixed-input plant accuracy;
use these diagnostic figures as closed-loop plan-versus-later-truth evidence,
with the time-alignment caveat.

### Conclusion

The MPC **can complete five practice laps** on its default raceline without a
collision and its publication timing passed the 20 ms gate in this run. It is
not yet accurate enough to claim a high-precision odometry/rollout pipeline:
full-run odometry drift is 27 cm and the fixed-input rollout has 52/1252
750 ms paths above 5 cm, mainly during near-straight throttle transients.
The throttle-epoch correction remains supported by the command-phase evidence,
but it does not remove this additional speed-dependent/transient error. No
weights or plant coefficients were changed from this test. Bag and complete
MPC diagnostics are retained for the next causal analysis.

## 2026-10-22 — faster IPOPT raceline attempt

The faster candidate was tested with the same practice simulator and production
MPC configuration as the five-lap default-line run. No MPC weights, plant
parameters, or trajectory points were changed for this trial.

- Candidate: `f1tenth_planning/trajectories/autodrive_practice_20260924_b/ipopt_v1_baseline_raceline.csv`
  (1024 points, 27.551 m). Its nominal `integral(ds / v)` estimate is about
  **5.367 s**; this is an optimizer estimate, not a measured lap time.
- Runtime log confirmed that both AMCL and MPC loaded this candidate. The
  production stack used the dev image and the established practice batch-mode
  simulator. The run was recorded at
  `live_runs/mpc_ipopt_v1_fastline_practice_20261022_run1/run`.
- The supervised run ended after **5.91 s** with **zero completed laps**. The
  MPC-fault watcher tripped when the candidate and prior plan both failed exact
  nonlinear corridor validation at stage 2. The recorded collision counter
  reached **1** during the termination window, so this is a collision-terminated
  failure, not a clean abort or a successful line validation.
- Before termination, ground-truth speed peaked at **7.627 m/s**. MPC recorded
  139 `accepted_optimal`, 38 `accepted_feasible`, 46
  `accepted_candidate_prefix`, and one `rejected_nonlinear_rollout` status.
  The final candidate-prefix chain lasted **22 actions**; its ADMM residual
  p50/p95/max was **0.605/1.808/7.021**, with minimum validated-prefix
  corridor slack **3.07 cm**. Thus the solver was publishing progressively
  shortened prefixes while losing solution quality before the hard rejection.
- Publication-gate time was **7.246/12.347/17.031 ms** p50/p95/max; no sample
  exceeded the 20 ms gate. Timing was not the cause of this run ending.
- The short, collision-terminated bag is not suitable for lap-time comparison.
  Its fixed-input source-plant replay had a 750 ms endpoint 5 cm pass rate of
  **198/204** and whole-path pass rate of **192/204**; the maximum whole-path
  error was **1.461 m** at a near-straight start. These outliers include the
  terminal collision/stop interval and do not establish the candidate plant
  model's steady driving accuracy. No lap-time or high-accuracy claim is made.

### Conclusion

The faster line is **not currently usable with this MPC**: this repeat did not
complete even one lap and recorded a collision. It did confirm that the MPC
can execute at the 20 ms timing target while driving this candidate, but the
22-action prefix chain, poor ADMM residuals, and subsequent stage-2 corridor
failure are the immediate control limitation. The default-line five-lap result
remains the only successful multi-lap result in this comparison. Preserve this
bag as the failure case; do not promote the faster line based on its nominal
5.367 s estimate.

## 2026-10-22 — PR-03 feasibility audit and PR-04 saved-state initialization A/B

This follow-up is offline only. No simulator was launched, and no production
plant, odometry, MPC weights, recovery policy, or trajectory was changed. The
only C++ change is to the offline `mpc_bag_solver_replay` CSV: it now exports
the already-computed exact nonlinear validation objective. The replay target
was rebuilt.

### PR-03 — what the faster path does and does not prove

The audit reads the active competition profile rather than using the old
15 cm evaluator default. Current profile hash:
`2bba4666f4e8910d1a0017dd919519e885056d4ba685c8d220826483ec3c9230`; the
configured corridor and first-prediction margins are both 0.05 m, and the
nonlinear tolerance is 0.005 m.

- IPOPT path: 27.55113 m / 1024 points; peak speed 8.4875 m/s; maximum
  absolute curvature 1.1175 1/m and lateral demand 8.400 m/s². Its minimum
  static centerline slack after 5 cm wall margin and heading-aware footprint
  is 0.1342 m. Geometric steering seed peaks at 0.3483 rad, below the 0.5236
  rad limit, with no seed slew-limit exceedances. Static geometry is therefore
  not the demonstrated failure; the speed schedule is not thereby certified.
- Four captured full-horizon-accepted plans immediately before the terminal
  prefix chain replay through the current source plant with minimum corridor
  slack 0.1282–0.1591 m. Replay versus recorded plant predictions differs by
  at most 2.35e-7 in the reported channels. This is local source-replay parity,
  not proof of Unity future truth or whole-lap feasibility. Maximum speed
  reference error on these plans is 0.476–1.057 m/s; maximum longitudinal
  slip is 0.586–0.924.
- At the last accepted prefix state, its three-step prefix has 3.07 cm minimum
  slack. The *unpublished* full candidate suffix crosses the corridor at stage
  4 (about -1.84 m slack); the next live candidate and retained prior-plan
  prefix both fail exact validation at stage 2. Do not score that rejected
  suffix as executed vehicle motion.
- In the successful default-line bag, 77 snapshots are within 0.05 1/m of the
  IPOPT failure curvature magnitude. Their speed range is 2.63–3.51 m/s
  (median 3.08 m/s), versus about 4.72 m/s at |curvature|=0.8375 1/m in the
  IPOPT terminal state. This is an unsupported speed/curvature combination
  in that successful capture, not proof that the combination is impossible.
- The IPOPT path seam has a 2.69 cm closing segment, 4.85 m/s² first/last
  acceleration discontinuity, and 0.0159 rad end-to-start heading delta.
  These defects are measured, but no evidence yet ties them causally to the
  terminal failure.

**PR-03 conclusion:** geometry is not the established cause. Evidence localizes
the failure to the high-speed/high-curvature closed-loop state and loss of a
feasible full horizon; global dynamic infeasibility is not proven. No global
speed cap or path edit is justified. Detailed artifacts:
`live_runs/mpc_ipopt_v1_fastline_practice_20261022_run1/trajectory_feasibility/`.

### PR-04 — command-consistent warm start and alternate initialization

Production warm-start logic was not changed. The existing code seeds the next
nominal with the action actually published, then shifts it once. Bag checks
found exact candidate-to-issued first actions on all 223 published fastline
commands, and exact previous candidate stage-1 to next nominal stage-0 on all
46 adjacent prefix transitions. The suspected command off-by-one is not
present in these captures; the remaining warm suffix can still be a poor
seed after the state diverges.

Six offline initialization modes used identical saved inputs and current
solver settings (100 iteration cap, 8 Jacobian workers, 5 cm margins, current
weights and `q >= 0.01`):

| Saved-state window | Sequence/warm | Sequence/cold ADMM | Nominal-cold/warm ADMM | Independent/cold | Reference-seed/warm ADMM | Dual-cold |
|---|---:|---:|---:|---:|---:|---:|
| Fastline terminal, 24 consecutive snapshots (diagnostics 205–228) | 3/24 accepted | 4/24 | 3/24 | 7/24 | 7/24 | 3/24 |
| Successful default line, 31 consecutive high-curvature snapshots (diagnostics 590–620) | 30/31 | 25/31 | 27/31 | 21/31 | 22/31 | 30/31 |

Acceptance means a reconstructed full-horizon candidate passed the exact
nonlinear plant/corridor check. It is not live publication, closed-loop, or
lap-success evidence. These are saved-state reconstructions: the bags do not
contain historical live ADMM z/y/rho or the full certified-plan suffix.

Findings:

- On the IPOPT tail, reference-seed and independent cold each add four
  accepted snapshots (diagnostics 210, 212, 213, 215); neither rescues the
  final stage-2 failure. The newly accepted plan at diagnostic 212 has only
  5.1 mm minimum corridor slack. Resetting only the nominal while preserving
  the issued-control warm sequence and ADMM memory adds no accepts; resetting
  only ADMM duals adds none.
- The evidence points more specifically to the **warm control suffix** being
  a poor seed in some fastline states, although its first action is aligned.
  Reference-seed changes the first action by up to 0.515 on this chain.
- On the held-out successful default-line segment, the existing sequence/warm
  replay passes 30/31, versus 22/31 for reference-seed, 21/31 for independent
  cold, 27/31 for nominal-cold and 25/31 for cold ADMM. Dual-cold is exactly
  identical to sequence/warm in status, first actions and exact nonlinear
  objectives. Several cold/reseed cases fail exact rollout at stages 18–29
  despite very small ADMM residuals: QP convergence alone is not enough.
- The live 22-prefix chain has five steering sign changes. Its per-cycle QP
  objective ranges 396–116,808 (median 13,772), but those objectives belong
  to different QPs and are not directly rankable. `e_y` grows from 0.0015 to
  0.894 m, `e_psi` from 0.018 to 0.819 rad, and speed over-reference from
  0.087 to 1.660 m/s. The replay now exports exact nonlinear objective for
  paired, same-snapshot comparisons.
- The bag has 224 exact bridge-request timestamp matches and 223 exact
  published-command timestamp matches. Command-start lateness is p95 0 ms,
  max 0.91 ms; Unity command-effect epoch IDs are absent. These observations
  neither prove nor exclude a Unity fixed-step application-delay mismatch.

**PR-04 conclusion:** do not globally reset ADMM or replace the production
warm start. Reference seeding helps several fastline snapshots, but regresses
the default-line held-out segment and one recovered case is nearly on the
corridor boundary. No production recovery-policy change is supported.

### Reproducibility limit and next action

The Oct-22 bags contain current hidden state/queue, issued float commands,
nominal/candidate control suffixes, outcome, residuals, objectives, and prefix
slack. They do not contain the pre-solve live ADMM z/y/rho snapshot or
certified-plan controls/cursor contents needed for exact same-memory A/B/C/D
replay. Run reports also lack the historical commit/build/image, map, loaded
YAML and supervisor manifest fields; these cannot be reconstructed
retroactively. Before promoting a solver policy, add opt-in, nonblocking
development capture for those missing fields, report dropped records and
capture overhead, then run a supervised fastline attempt and default-line
comparison without changing production behavior or the <20 ms gate.

The faster candidate still has **0 completed laps**; 5.367 s is an optimizer
estimate only. Keep the default line as the only proven multi-lap result.

Reproduction identifiers:

- Bag hashes: fastline
  `192b9f36f2b2a393506553c74d17379530f85ea14b7e159df4d39cf4564904bf`;
  default success
  `a9f80c1fa28c9b706e39294fca73fbefde4b2a059bc54c9f709ed52b39776d17`.
- Trajectory hashes: fastline
  `35bed562a8b0611c48cbf2df8520aff091a3f212adf0eacd5f59e6b9f527fb73`;
  default
  `09b0c3e5b36b781b8709eb08a98cd8c36eb7951391f269ed761409a64938a253`.
- Source plant hash: `21466f30fced578185fc4941d9a53d95feabe40702265c6fe5b06befbf4ade76`.
- Solver-input hashes: fastline
  `d569513fc7f8a3483e4d67f3728e3eb77f310cf8ad703856026d8209933fe05f`;
  default high-curvature
  `930d69c478e53163fda3392bf31b1a2bf4d9640699797f5f86bf77eb7df7b0bf`.
- Offline replay binary hash:
  `d265902ae4306bcc0c8683666b7c4fce3a431f5718779bded099e4a7f4898be4`.

Verification: replay target and two missing CTest binaries built; all 9 CTests
passed. All 25 focused Python tests passed (trajectory audit, saved-state
export, odometry drift, one-step source analysis, and rollout case index). No
simulator test was run in this follow-up.

## 2026-10-22 — Source-exact steering epoch diagnostic (PR-02)

This follow-up corrects the offline evaluator's steering-actuator diagnostic;
it does not change the production plant, odometry, MPC or trajectory. No
simulator was launched.

### Source behavior and evaluator correction

`VehicleController.CurrentSteeringAngle` returns the `currentS` snapshot,
which `FixedUpdate()` assigns from the internal rack angle before calling
`Steer()`. `Steer()` is not a symmetric rate limiter: its direction branch
compares normalized autonomous steering with the sign-flipped internal rack
angle and clamps to target after one rate step. On an unwind where that step
has already crossed the target, the source snaps to the target in that
physics update. The production `unity_source_plant.cpp` already ports this
branch, including the practice prefab's 183.345993 deg/s steering-rate value.

The prior `evaluate_mpc_bag.py` feedback diagnostic instead used a symmetric
rate-limited interpolation. That approximation can manufacture a large
residual exactly at target crossing. It now replays the source branch for
exactly 25 × 1 ms per nominal request epoch (not a rounded ROS packet gap),
clamps the normalized command as the source setter does, and labels the result
as feedback-versus-feedback rather than vehicle-motion accuracy. A new C++
regression test and four Python tests cover normal rack motion, the source
target-crossing snap, command clamping, and invalid timing.

### Practice-bag results using the corrected diagnostic

| Capture | Same request epoch `C[i+0]` | Best epoch counts | Interpretation |
|---|---:|---:|---|
| Default-line, five-lap success | 0.00251 rad RMSE; 0.00005 rad p95; 1202/1281 transitions choose `C[i+0]` | `i−2:40`, `i−1:24`, `i:1202`, `i+1:13`, `i+2:1`, `i+3:1` | Source steering response is consistent for nearly all transitions; the residual tail is sparse. |
| IPOPT fastline, failed trial | 0.03043 rad RMSE; 0.00640 rad p95; 185/230 transitions choose `C[i+0]` | `i−2:28`, `i−1:14`, `i:185`, `i+1:3` | Most intervals follow the same source branch, but one command-boundary event dominates RMSE. |

The retained fastline bag's last 22 candidate-prefix actions begin at progress
38.610 m with MPC state `e_y=0.001 m`, `e_psi=0.018 rad`, `u=6.654 m/s`,
path curvature −0.175 1/m and reference speed 6.57 m/s. Over the prefix chain,
the reference falls to 3.06 m/s by progress 41.159 m while curvature reaches
−0.837 1/m; the MPC state is then `e_y=0.894 m`, `e_psi=0.819 rad`,
`u=4.723 m/s`, with only 3.07 cm predicted prefix slack. The next diagnostic
is at progress 41.193 m (−0.885 1/m, 3.00 m/s reference), with
`e_y=0.978 m`, `e_psi=0.828 rad`, `u=4.767 m/s`; the new candidate and prior
plan both fail at validation stage 2. Published actions are ordered
`[steering_rad, q]`; the chain's issued q values range from 0.01 to 0.604.
This records the failure progression but does not by itself distinguish a
speed-profile/weight issue from source-plant or actuator-epoch mismatch, and
does not prove that the path is globally infeasible.

The fastline outlier is the interval ending at ROS stamp
`1792623140442672939 ns` (25.839 ms after its prior sample). Reported
physical steering changes from approximately −0.3933 rad to +0.0105 rad.
The preceding request's normalized command is −0.859 (source target about
−0.4498 rad); the next request is +0.020 (target +0.01047 rad), and the
endpoint feedback matches that next target to about 0.00003 rad. The default
capture has one analogous, smaller boundary residual (about 0.0795 rad max).
The source recurrence predicts the fastline endpoint with 0.4603 rad residual
when forced to use the prior request, but with the next request it is within
measurement rounding. This indicates a request/physics-step association
ambiguity at a command boundary; it does **not** identify which Unity
`FixedUpdate` consumed the command.

The best neighboring-epoch results are diagnostics using adjacent logged
commands, not causal evidence for an online delay change. Bridge request to
response latency is about 28 ms in both bags, but this measures transport and
callback response timing, not Unity's applied-command physics-step ID. Neither
bag records the applied sequence ID or fixed-step counter needed to settle the
rare boundary case. Therefore no command delay, steering-rate parameter, or
production state correction was changed.

### Instrumentation feasibility and verification

The current `tools/start_simulator.sh` trace environment has no matching
consumer in the checked-in Unity subset; setting it cannot currently produce
the requested fixed-step trace. The practice Docker image contains an IL2CPP
player (`GameAssembly.so`, no `Assembly-CSharp.dll`), so editing managed
scripts beside the executable would not instrument the running image. The
public upstream AutoDRIVE Simulator source branch is available and declares
Unity 2022.3.52f1, matching the local source subset's project version, but it
is a separate, large source tree and does not by itself establish identity
with the competition practice player. A trace-enabled player must be built
from a verified matching project/player before using its step IDs as evidence.

Verification performed:

- `cmake --build build/f1tenth_mpc --target test_unity_source_plant -j2`
- `ctest --test-dir build/f1tenth_mpc -R '^test_unity_source_plant$' --output-on-failure`
  — passed.
- `python3 -m unittest tools.test_evaluate_mpc_bag_steering -v` — 4 passed.
- `python3 -m py_compile tools/evaluate_mpc_bag.py tools/test_evaluate_mpc_bag_steering.py`
  — passed.
- Re-scored both retained practice captures with
  `--source-plant-rollout --source-reconstruct-command-queue-from-bridge`;
  the evaluator now reports the exact `VehicleController.Steer` branch.
- `git diff --check` — passed before this documentation update.

**PR-02 status:** the source-equation mismatch in the diagnostic has been
fixed and production dynamics were confirmed to already use the source
steering branch. The remaining rare actuator epoch discrepancy is not
identifiable from the current bag schema. The IPOPT fastline still has zero
completed laps and one collision; this diagnostic does not improve its
feasibility or justify another unchanged live attempt. Continue with a
trace-enabled, version-matched practice player or keep the boundary residual
explicitly unresolved; do not tune the MPC/plant to this single sample.

## 2026-10-22 — PR-00/01/02 evidence continuation

No simulator was launched. This pass reused the retained practice captures;
the only code change is to the offline one-step analysis. No production
odometry, MPC, weight, plant, or trajectory behavior changed.

### Provenance reconciliation (PR-00)

The three relevant raw SQLite bags are present and their bytes were hashed.
The report sidecars record the captured HEAD (`cfefebcfe18aee16a7e55f95764db61dd0eca59f`)
and report-generation build environment, but **none has a sealed pre-launch
run manifest**. Their `metadata.yaml` files are ROS bag metadata, not the
handoff's run manifest. In particular, simulator/controller image digests,
exact loaded YAML hashes, map hash, initial pose, supervisor version, and full
runtime limits cannot be reconstructed after the fact. Do not claim those
properties for these historic runs.

| Regime | Raw bag SHA-256 | Trajectory SHA-256 | Recorded outcome |
|---|---|---|---|
| CSV follower, five-lap development capture (`live_runs/csv_v251_gt_5lap_run1_20261020`) | `1c466a4ba68c8aa840c937dd2c9a051eb29ae8f9ecff0c52596394e2aac01126` | Not sealed in its run report; do not infer it from the bag name | Comparator only; no MPC claim |
| Default mintime, production MPC (`live_runs/mpc_odom_epoch_fix_practice_20261022_run1`) | `a9f80c1fa28c9b706e39294fca73fbefde4b2a059bc54c9f709ed52b39776d17` | `09b0c3e5b36b781b8709eb08a98cd8c36eb7951391f269ed761409a64938a253` | 5 clean laps; mean 6.292 s, best 6.045 s; no collision |
| IPOPT fastline, production MPC (`live_runs/mpc_ipopt_v1_fastline_practice_20261022_run1`) | `192b9f36f2b2a393506553c74d17379530f85ea14b7e159df4d39cf4564904bf` | `35bed562a8b0611c48cbf2df8520aff091a3f212adf0eacd5f59e6b9f527fb73` | 0 laps; MPC fault at 5.733 s; collision at 5.871 s; nominal `∫ds/v=5.367 s` is not an achieved lap |

Existing `report.json`, `cycles.csv`, source one-step outputs, source-rollout
outputs, feasibility audits, and saved-state replay CSVs remain beside their
bags. They are separately labeled by regime; none is a substitute for the
missing launch manifest. A future launch must use
`tools/seal_mpc_run_manifest.py` before starting either container.

### Drift attribution (PR-01)

The independent exact-epoch report was reproduced from 1,283 same-stamp truth
and odometry rows, all at the rear-axle reference point. The final relative
pose error is 0.2697 m, almost entirely forward (−0.2697 m; lateral +0.0018 m);
relative yaw is numerically identical because this simulator's reported IMU
quaternion matches its truth orientation. Forward-velocity error is bias
−0.00834 m/s, RMSE 0.03287 m/s, p95 0.04607 m/s, max 0.18296 m/s. Trapezoid
integration of the measured odometry-u residual accounts for −0.26785 m of
forward displacement, matching the observed final forward error within about
2 mm.

The pose integrator is not the main source: integrating production odom twist
reproduces its own recorded pose increments with 1.85 mm XY RMSE. Integrating
Unity's measured twist reproduces Unity pose with 7.5 mm XY RMSE over the run;
midpoint and interpolated 1 ms integration differ by only about 3 mm at the
endpoint. The large 27 cm discrepancy is therefore a velocity-estimation
error, not a frame/reference-point conversion or midpoint-integration defect.
About 21 cm of the forward displacement error accumulates in the first two
seconds; subsequent per-lap speed bias is near zero. The yaw/position and
velocity error cannot be summarized as a single rollout-accuracy number.

### First divergence and command/slip separation (PR-02)

`tools/analyze_source_plant_one_step.py` now reports elapsed-time bins and the
**incremental one-step error**
`(predicted next u − next truth u) − (initial source u − initial truth u)`.
This distinguishes error already inherited by the initial state from new
error added by that interval. It also reports wheel/body-speed mismatch proxy,
source-reported rear slip, queue throttle, and next-feedback/queue difference.
The distinction is diagnostic only; it does not inject truth into production.

On the default-MPC capture:

- At 0.25–0.50 s, mean initial/next u error is −0.0315/−0.0340 m/s; mean
  source rear-slip magnitude is 0.602 and the model-derived wheel/body
  mismatch proxy averages 7.62 m/s.
- At 0.50–0.75 s, the error is already −0.1355 m/s at the start and
  −0.1417 m/s after one step; the added error averages −0.0062 m/s per
  interval. At 0.75–1.00 s it is nearly flat (−0.1699 to −0.1698 m/s), then
  recovers as slip and mismatch decline. This is a transient concentrated in
  the initial high-spin acceleration, not a constant per-lap speed scale
  error.
- At the 0.50 s interval, q changes 0.552→0.086; one-step u error grows from
  −0.075 to −0.106 m/s while source-reported rear slip is 0.887 and the
  model-derived wheel/body mismatch proxy is 13.35 m/s. These latter two are
  not independent Unity wheel truth.
- Command-slot A/B on identical recorded start states does **not** support a
  one-packet shift: slot 2 has 0.0325 m/s u RMSE over all 1,282 intervals,
  versus 0.1318 m/s for slot 1. During |Δq|>0.2 the scores are 0.1348 vs
  0.1554 m/s; for falling q transitions, 0.1359 vs 0.1926 m/s. Substituting
  next-epoch throttle feedback (an offline-only future oracle) gives 0.0362
  m/s overall and 0.1341 m/s for |Δq|>0.2—no meaningful fix to the startup
  discrepancy. The bag confirms feedback/command association, not the exact
  `FixedUpdate` where torque was applied.
- As an operating-profile comparator, the CSV-follower bag's one-step u error
  is 0.00604 m/s RMSE overall (bias +0.00041, p95 0.01089); its first two
  seconds use q near 0.12 with much lower source-reported slip. The fastline
  MPC bag has 0.03691 m/s one-step u RMSE before its collision/fault cutoff,
  but only 233 intervals and no completed lap. These are different command
  profiles, not a randomized causal comparison.

**Causal ranking from retained evidence:** (1) ordinary pose integration and
rear-axle convention are ruled out as the 27 cm cause; (2) a simple one-packet
command-phase error is disfavored by the paired slot test and encoder-angle
replay; (3) the remaining clear association is high-slip, rapid torque-change
acceleration, but available wheel-speed/slip channels are model/encoder
proxies rather than per-wheel Unity contact truth; (4) the applied command's
per-`FixedUpdate` identity remains unknown. Do not tune torque/tire
coefficients or production odometry from this single run yet.

### Verification and next gate

- `python3 -m unittest tools.test_analyze_source_plant_one_step -v` — 6 passed.
- `python3 -m py_compile tools/analyze_source_plant_one_step.py tools/test_analyze_source_plant_one_step.py` — passed.
- The new analyzer was run in-memory against the default-MPC, CSV-follower,
  and fastline practice bags; no duplicate data artifacts were written.
- Added `docs/development/MPC_HANDOFF_BASELINE_COMPARISON_20261022.json` as
  the PR-00 machine-readable comparison index. It separates the CSV follower,
  pre-fix IPOPT probes, post-fix default-MPC success, post-fix IPOPT failure,
  and saved-state initialization replays. It carries raw DB3 and trajectory
  hashes where verified and marks the missing post-fix launch manifests and
  unavailable CSV trajectory identity explicitly. This is an index of existing
  evidence, not a causal A/B claim.
- Extended `tools/capture_mpc_qp_regression_corpus.py` with an explicit
  `--post-odom-epoch-fix-only` mode. It captured 31 post-fix default-line
  high-curvature states and 24 post-fix IPOPT terminal-chain states into the
  separate `live_runs/mpc_qp_post_odom_epoch_fix_20261022/` delta corpus. The
  existing 125-fixture corpus was not modified. These remain saved-state
  reconstructions; they do not restore historical live ADMM memory.
- Validated all 55 delta fixtures: sidecars valid, roundtrip bytes identical,
  and recorded solver output replayed with exact parity. Added grouped
  minimum-bound-relaxation diagnostics to `tools/mpc_qp_oracle.py`, then
  independently classified the reconstructed QPs with OSQP and an LP
  feasibility check:
  - Default high-curvature segment: 19 solved, 6 solved-inaccurate, 6 oracle
    time-limited; native ADMM reached its cap on 1/31 snapshots.
  - IPOPT terminal chain: 22/24 oracle time-limited and 2/24 LP-confirmed
    infeasible. Both confirmed-infeasible fixtures are the final two rows
    (source diagnostics 227–228). Relaxing only their linearized `e_y`
    corridor bounds by 0.2632 m and 0.5077 m respectively restores LP
    feasibility; relaxing control bounds alone or other state bounds alone
    does not. This is a local QP diagnosis at the already-diverged terminal
    states, not proof that the physical path is globally infeasible. No
    production corridor or action bound was weakened.
- `python3 -m unittest tools.test_mpc_qp_oracle -v` — 2 passed. The independent
  QP validation recorded exact parity for all 55 fixtures; its oracle
  classifications and bound-relaxation diagnosis are retained in
  `live_runs/mpc_qp_post_odom_epoch_fix_20261022/validation_summary.json`.
- `git diff --check` — passed.
- No simulator/controller run was active; no simulator was launched.

PR-01's pose-drift source is localized to early forward-velocity estimation,
but its causal model component is not yet identified. PR-02 establishes that
one-packet shifting is not the fix; source-exact torque application timing is
still unavailable. PR-00's comparison index and post-fix QP delta are present,
but its gate remains partial because pre-launch manifests for the Oct. 22
captures are absent. The faster IPOPT trajectory remains unproven in
closed loop; do not rerun it unchanged. The terminal local-QP evidence points
to a corridor-feasibility loss after the car has already diverged substantially,
not a control-bound-only failure. The next handoff step remains an opt-in,
bounded development capture of the live solver/plan state for PR-04. Any
subsequent faster-line attempt must change the upstream speed/path plan based
on source-plant feasibility, not widen the corridor or repeat this same run.
Production selection, plant, weights and corridor remain unchanged.

## 2026-10-22 — targeted higher-budget fastline QP follow-up

Continued the handoff's numerical classification of the post-odom-fix IPOPT
terminal chain. This was an offline identical-QP oracle experiment only: no
simulator was started, and no production solver setting, weights, plant,
corridor, or trajectory was changed.

- Extended `tools/refine_mpc_qp_oracle_cases.py` with repeatable `--case-id`
  and `--fixture` filters, so a second budget can be spent only on the exact
  unresolved fixtures rather than rerunning already-classified rows.
- The 5 s screen returned `solved` for 13 of the 22 fixtures not already
  proven infeasible; the remaining 9 reached the time limit. The two terminal
  fixtures independently confirmed infeasible by LP were excluded.
- Re-ran only those 9 at up to 20 s / 1,000,000 iterations. One returned
  `solved` (`qp_000012.fqp`, 452,750 iterations; primal/dual residual
  `3.47e-7 / 1.62e-6`). One returned `solved inaccurate` at the time limit
  (`qp_000018.fqp`; residuals `4.76e-9 / 2.68e-7`). Seven still reached the
  time limit, with large residuals on several. Time-limited means unresolved,
  not infeasible.
- Across the 24 fastline terminal fixtures, 15 have an OSQP solver-returned
  result after the 5/20 s screens, 7 remain time-limited, and 2 are LP-proven
  infeasible under the reconstructed local box-constrained QPs. This does not
  prove that the path or vehicle is globally infeasible: these are frozen
  local linearizations at states reached after the live trajectory had
  already diverged. No corridor relaxation is proposed.
- Outputs are
  `live_runs/mpc_qp_post_odom_epoch_fix_20261022/ipopt_terminal_oracle_refinement_5s.json`
  and `.../ipopt_terminal_oracle_refinement_20s.json`; the comparison index
  `docs/development/MPC_HANDOFF_BASELINE_COMPARISON_20261022.json` records the
  counts separately from the original 0.2 s screen.

This narrows the numerical question but does not create a faster usable
raceline. The next consequential step remains bounded development capture of
the *live* solver/plan state and rejected full suffix, then distinguishing an
upstream nominal/reference problem from numerical QP trouble before another
fastline simulator run. The default-line result and fastline failure evidence
remain unchanged.

### Live fastline failure-context telemetry (diagnostic-only)

Added opt-in diagnostic fields to capture the controller state that the
existing fastline bag did not preserve. This does **not** change candidate
selection, the command envelope, plant equations, corridor acceptance,
trajectory, or MPC weights:

- `f1tenth_mpc/src/mpc_controller_node.cpp` emits diagnostics schema version 3
  with the actually issued action (only when this cycle published one), the
  causal two-command steering/throttle queue, source/control/candidate epoch
  stamps, and—on a rejection, candidate/prior-plan prefix, or continuing
  prefix chain—the certified-plan validity, cursor, remaining horizon,
  complete controls, stage states, hidden source-plant state and stage-wise
  corridor slack. Snapshot capture is limited to those diagnostic contexts;
  it does not add file I/O to the solver callback.
- `f1tenth_mpc/src/mpc_rti.c` exposes the exact nonlinear footprint clearance
  calculation through `mpc_rti_compute_corridor_slack()`. The stage-wise
  candidate slacks come from the same bounds and footprint used by the
  acceptance check; diagnostic slack does not relax or replace that check.
- `f1tenth_mpc/test/test_vehicle_model_source_rollout.cpp` checks the shared
  slack calculation at the first-stage margin and later-stage margin. The
  source-plant regression test also checks the Unity steering-target crossing
  clamp exercised by the current workspace implementation.

Verification on the current working tree:

```bash
cmake --build build/f1tenth_mpc --target \
  test_vehicle_model_source_rollout mpc_controller_node -j2
build/f1tenth_mpc/test_vehicle_model_source_rollout
```

Both targets built; all 32 source-rollout tests passed. This proves the code
builds and the shared corridor-slack semantics match the test expectations;
it does **not** yet prove the ROS diagnostic message decodes from a live bag,
that telemetry load is negligible, or that the fastline is driveable. The next
step is one supervised practice capture on the unchanged IPOPT fastline with
diagnostics enabled, bagging the `mpc_performance` topics, and stopping at the
first MPC fault or collision. Its purpose is to inspect the newly captured
certified-plan/candidate context, not to claim a lap-time trial. No capture has
yet been launched under this schema.

### 2026-10-22 — fastline diagnostic decode and retained-plan experiment setup

The diagnostic-only practice run on the unchanged IPOPT fastline completed
its intended fault capture. It did **not** complete a lap and is not a racing
success:

- Bag: `live_runs/mpc_ipopt_v1_fastline_diagnostic_20261022_run1/run/`
- Report and cycles: `report.json`, `cycles.csv` in the same run directory.
- First unpublished MPC rejection: 3.279684 s. Collision telemetry followed
  at 3.392061 s (112 ms later); the bridge fault arrived at 3.465790 s. The
  collision is retained in the run outcome and is not hidden by the earlier
  score cutoff. No lap was completed.
- 131 MPC diagnostics: 83 full-horizon accepts, 42 exact-validated 75 ms
  candidate prefixes, one rejected nonlinear rollout, and five localization
  wait cycles. The controller tracking diagnostics reached cross-track RMSE
  0.184 m and heading RMSE 0.178 rad before failure. The solver hit its
  100-iteration limit on all 42 prefix cycles; the final failed cycle had
  large residuals. Publication-gate latency max was 11.203 ms with no sample
  at or above 20 ms, and no callback interval exceeded 25 ms. Timing therefore
  does not explain this failure.
- The exact source replay reproduced logged states for previously accepted
  plans. The audit had initially compared center-sample corridor bounds; that
  was incorrect because runtime uses a minimum-bound preview over ±0.10 m.
  The audit now uses the same periodic preview and stops at the first
  corridor violation. This was a diagnostic mismatch, not a production
  acceptance defect.
- Same-state source-plant counterfactuals from the new capture compare the
  rejected candidate suffix with the retained certified suffix. Across 43
  eligible snapshots, the candidate suffix crossed the corridor in all 43;
  the prior suffix survived more stages in 29 and passed all of its remaining
  actions in 26. On the 21-snapshot terminal chain, the prior suffix passed
  its complete remaining sequence in 5, survived longer in 12, tied in 7 and
  was worse in 2. This is one captured chain, not closed-loop evidence or a
  guarantee that replaying the prior plan will complete a lap.
- Added default-off parameter
  `prefer_full_validated_prior_plan_on_rejection`. When explicitly enabled,
  it first revalidates the *entire remaining* certified plan suffix from the
  current state and queue; it publishes only if that exact source-plant and
  corridor check passes within the existing publication deadline. Otherwise
  the existing candidate-prefix/short-prior checks remain in their original
  order. The competition default remains `false`; no plant, path, weight,
  corridor, command envelope or accepted-plan semantics changed. Corrected
  the prefix log message so it no longer implies a prior-plan check happened
  when the default policy did not perform one.
- Added a separate `full_prior_plan_suffix_revalidation` diagnostic record
  and exporter columns so a later 75 ms prior-plan retry cannot overwrite the
  result of the full-suffix attempt. The bag decoder test verifies both
  records remain separately available.
- Two prelaunch manifests (`..._run1` and `..._run2`) were superseded before
  launch as telemetry/schema changes were made. The current sealed follow-up
  manifest is
  `live_runs/mpc_ipopt_v1_prior_full_suffix_20261022_run3/run_manifest.json`.
  It uses the same practice map, IPOPT trajectory SHA
  `35bed562a8b0611c48cbf2df8520aff091a3f212adf0eacd5f59e6b9f527fb73`,
  initial pose, 8 Jacobian workers and 4 ROS executor threads; only the
  run-local parameter overlay enables the experiment.

Verification: MPC controller and source-rollout targets build; all 32
source-rollout tests passed before the full-suffix case was added. The
expanded `MpcCandidate.PriorPlanSuffixRequiresExactCurrentStateValidation`
test passed for a 29-action suffix in the complete 32-test rerun. The
trajectory-audit and run-report Python suites pass 19/19, runtime-topic policy
passes, and `git diff --check` passes.

The follow-up live practice experiment has **not** been launched. At the time
of preparation, an unrelated `sdu_apex_sim_compete` container was running the
competition image, reconnecting to port 4567 with no controller or recorder
container. It was left untouched. The repository simulator and bridge use
that fixed port, so the practice test must wait until that external container
is no longer active; do not connect the test controller to the wrong track or
stop the unrelated session without confirming ownership. Once the port is
free, run the sealed test under the established Xvfb batch-mode path and the
fault/collision supervisor, then evaluate whether retained-plan certification
actually improves the fastline outcome. No faster lap or usable fastline has
yet been demonstrated.

### 2026-10-22 — handoff confirmation and odometry drift attribution

Confirmed the active plan is `SDU_Apex_MPC_Odometry_Fix_Next_Steps_Handoff_2026-10-22.md`.
Continued PR-01/PR-02 from the retained practice captures, without launching a
simulator or changing production odometry, plant, MPC, weights, or racelines.

**Default-line MPC capture**
(`live_runs/mpc_odom_epoch_fix_practice_20261022_run1`):

- Exact-stamp comparison has 1,283 truth/odom/IMU/source rows, no unmatched
  rows, fixed initial pose alignment only, and rear-axle reference conversion.
  Relative yaw from the IMU quaternion matches truth to floating-point
  precision; integrating gyro instead has 0.0592 rad RMSE. Yaw and the
  midpoint-versus-1 ms pose integrator are therefore not the source of the
  27 cm final odometry displacement error.
- Recorded odometry ends 0.2697 m from truth. Replacing only its `u` series
  with same-epoch truth reduces the endpoint to 0.0276 m; replacing only rear
  lateral velocity leaves 0.2512 m; replacing both leaves 0.0058 m. The main
  measured contribution is forward-velocity error, not yaw or lateral
  integration.
- Most of the displacement accumulates during launch: at 0.60 s odometry `u`
  is 0.138 m/s low; it reaches 0.18 m/s low near 1.0 s and remains about
  0.14 m/s low through much of the first lap. The first lap ends about 0.23 m
  behind in the track-forward direction; later laps add substantially less.
- The one-step source-model replay's overall forward error is 0.0325 m/s
  RMSE, versus 0.00745 m/s when only same-epoch truth `u/v/r` replace the
  initial visible motion state and hidden states/commands stay unchanged. In
  the 0.175–1.7 s launch interval the corresponding RMSEs are 0.1396 and
  0.0062 m/s. This isolates a large carried motion-state discrepancy; it
  does not yet identify whether the initial discrepancy is caused by
  acceleration-state evolution, unobserved wheel/contact state, or exact
  Unity command-effect timing.
- Shifting the replay to the older queued throttle slot worsens overall
  forward RMSE from 0.0325 to 0.1318 m/s; substituting next-epoch throttle
  feedback gives 0.0362 m/s. Neither supports an extra-packet-shift fix.
  A 25 ms trapezoidal IMU-acceleration increment is also not a valid
  replacement: its incremental RMSE is 0.0591 m/s versus 0.0087 for the
  source-plant increment, and recursive full-IMU propagation reaches
  0.6275 m/s RMSE. No IMU fusion change is promoted.
- The early launch includes sharp throttle changes and a wheel/body-speed
  mismatch proxy above 10 m/s. The capture has no Unity physics-step ID, so
  this is a regime association, not proof that the command was applied on a
  particular FixedUpdate. PR-02's exact 1 ms event trace is still needed
  before attributing this to torque/slip or changing the plant.

**CSV-follower practice capture**
(`live_runs/csv_v251_gt_5lap_run1_20261020`):

- Recorded odometry ends 0.1828 m from truth; its `u/v` errors are much
  smaller than in the MPC launch capture. Replacing both velocity components
  leaves 0.0802 m endpoint discrepancy between integrated Unity truth twist
  and Unity truth pose, so the complete 18 cm cannot be assigned to odometry.
- At elapsed 28.929963 s, one truth-pose interval is 0.3451 m over 24.746 ms,
  while truth speed is about 3.27 m/s; the truth pose then repeats for three
  samples. Bridge request-to-response latency is 110.4 ms at the jump and
  remains 88.3/86.0 ms in the following samples. Progress is unavailable at
  the jump. This is strong evidence of a pose/twist epoch or burst-delivery
  anomaly, but not proof of the Unity physics epoch because the bag lacks its
  step ID. The samples are flagged as epoch-ambiguous, not discarded or used
  to fit physics.
- For the rest of this capture, exact-stamp source one-step forward RMSE is
  0.00604 m/s, but there are no throttle-rise/fall samples. It cannot validate
  the transient regimes that dominate the default MPC launch discrepancy.

The reduced IPOPT optimizer and its README were also checked against the
handoff: this fastline is generated by a spatial acceleration/lateral-limit
surrogate, not the production 33-state source plant. Its 5.367 s value is an
integral estimate, not a Unity or MPC lap time. The geometry audit alone does
not certify dynamic feasibility. Do not repeat the unchanged fastline or
promote it; PR-03 still requires source-plant feasibility evidence and a
source-consistent speed schedule.

Added synthetic PR-01 coverage to `tools/test_analyze_mpc_odom_drift.py` for
reordered/missing exact epochs, reversing yaw, yaw seam crossing, high lateral
slip/reference-point conversion, and jittered intervals with a missed packet.
`python3 -m unittest tools.test_analyze_mpc_odom_drift -v` passes 10/10;
`git diff --check` passes. These tests validate offline diagnostic semantics,
not odometry accuracy.

**Next in sequence:** resolve PR-02 with a development-only Unity FixedUpdate
trace of request/command receipt, applied actuator targets, physics-step ID,
wheel speeds/slip, and body state; compare it against clean launch windows
before changing any production equation. The active simulator port is still
occupied by an unrelated competition-image container with no visible
controller/recorder; it remains untouched. The sealed practice experiment is
not launched until the established practice simulator can be started without
connecting to that other session.

### 2026-10-22 — PR-03 local source-plant throttle feasibility

Continued the exact Oct. 22 odometry/MPC handoff after the PR-01/PR-02
diagnostics. No live simulator was launched: `sdu_apex_sim_compete` is still
running, while neither bridge port 4567 nor 4568 has a listener. It was not
stopped or repurposed. The PR-02 causal attribution remains **insufficient**:
the bags do not contain a Unity `FixedUpdate` command-application ID, so ROS
epoch reconstruction cannot establish the exact physics-step input. No plant
coefficient or odometry change is justified by these captures.

**Pinned torque branch.** The active practice scene overrides the F1TENTH
prefab `MotorTorque` from 85.6 to 428 N m. The prefab selects `driveType=5`
(CAWD) and `brakeType=0` (CAWB). `VehicleController.Drive()` therefore
distributes a positive command `q` across four wheels and clears their brakes;
`q=0` instead applies 428 N m brake torque to each wheel. At the production
MPC minimum `q=0.01`, drive torque is 4.28 N m total, 1.07 N m/wheel, and no
brake torque. Added this explicit source/scene mapping beside the production
plant constant so the minimum legal MPC command is not later misread as a
brake. The edit is comments only; dynamics and legal command bounds are
unchanged.

**New offline audit implementation.** Extended
`tools/audit_mpc_trajectory_feasibility.py` with `--fixed-throttle-grid`.
It replays each requested positive throttle through the same 1 ms source
plant, preserving the captured initial hidden state, in-flight two-command
queue, and captured steering sequence. It now also runs this grid against the
terminal rejected full candidate suffix. `q=0` and non-finite/out-of-range
values are rejected. This is explicitly a local counterfactual—not a searched
policy, global infeasibility test, or Unity-native contact-solver proof.

Artifacts are under
`live_runs/mpc_ipopt_v1_fastline_diagnostic_20261022_run1/trajectory_feasibility/q_grid_terminal/`:

- `ipopt_q_grid_terminal.json`
- `ipopt_q_grid_terminal_source_witness.csv`
- `ipopt_q_grid_terminal_bins.csv`
- `ipopt_q_grid_terminal_candidate_prior_ab.csv`

The evaluator library is `build/f1tenth_mpc/libunity_source_plant_eval.so`
(SHA-256 `bac0ca3657e7c3db7a290e40a91acc30dbf163e411d112b86b096b531294eb8d`);
the IPOPT trajectory remains unchanged (SHA-256
`35bed562a8b0611c48cbf2df8520aff091a3f212adf0eacd5f59e6b9f527fb73`).
Geometric audit: lap 27.5511 m, peak CSV speed 8.4875 m/s, idealized peak
lateral demand 8.4000 m/s², and 0.1342 m minimum aligned footprint slack at
the configured wall buffer. These are geometry/reference facts, not dynamic
feasibility results.

**Measured local source-plant result.** The last two captured full-horizon
states before the terminal prefix chain are at progress 25.6686 and 25.8410 m.
Their source-state forward speeds are 6.576 and 6.347 m/s, but rear-wheel
surface speeds are only about 3.56 and 2.96 m/s, respectively. This is a
large *model-state* wheel/body mismatch; the bag has no independent Unity
per-wheel physical-speed truth, so it must not be presented as confirmed
physical tire slip.

Holding each captured steering plan and command queue fixed, constant `q=0.125`
source-plant rollouts pass the full 30 stages (750 ms) for both states with
minimum corridor slack 0.183 and 0.158 m. End speeds are 3.060 and 3.060 m/s
against terminal references 2.799 and 2.848 m/s; maximum within-horizon speed
errors are 0.580 and 0.338 m/s. Constant `q=0.20` instead violates the exact
corridor at stages 28 and 26. This establishes a local source-model witness
for the captured steering suffix; it does **not** certify the whole lap or
show what Unity would do at the exact same substeps.

At the terminal rejected candidate (progress 28.2605 m, initial speed
4.580 m/s, `e_y=0.724 m`, `e_psi=0.754 rad`, reference about 2.85 m/s), every
tested constant-q continuation
`0.05–0.50` violates the corridor by stage 4 or 5. The first two commands
remain the recorded in-flight queue, so changing future throttle cannot
repair the already-late state immediately. This is a local failure of the
captured steering/queue/state combination; other steering or earlier
throttle sequences have not been disproved. The source replay reproduces the
logged candidate prediction to below `2e-7` in the common states, verifying
the audit binary against the captured MPC model, not against native Unity.

**Decision:** the data do not establish that the IPOPT speed profile is
globally infeasible. The geometry is clear, and at least two pre-terminal
states admit a 750 ms source-plant witness at q=0.125. Conversely, the actual
terminal state is already too late for the tested fixed-q continuations. No
speed schedule, path, weight, MPC behavior, or odometry was changed. PR-03 is
still **partial**: only the captured progress interval has reachable-state
witnesses; a complete source-plant-derived schedule over the lap and a
different-seed feasibility search remain outstanding. Do not advertise the
5.367 s integral estimate as an achieved or validated lap time.

**Verification:** `python3 -m unittest tools.test_audit_mpc_trajectory_feasibility -v`
passes 11/11; Python byte-compilation and `git diff --check` pass. No production
behavior or test bag was modified.

**Next:** finish PR-03's per-progress feasibility/witness coverage without
inventing a global speed cap. Then use the recorded pre-terminal state and
source-plant witness as the fixed-state entry point for PR-04 candidate vs.
retained-plan vs. feasible-continuation analysis. Live use of the faster line
still waits for the established practice simulator/bridge endpoint to be
available without interfering with the unrelated active competition image.

### 2026-10-22 — PR-02 wheel-epoch check and PR-03/04 evidence extension

The exact Oct. 22 handoff was confirmed as the active plan. Work remained
offline: a `sdu_apex_sim_compete` container was still active with no listener
on ports 4567/4568. It was not stopped or reused, and no new simulator run was
started.

**PR-02 control-time wheel comparison.** Added the reusable diagnostic
`tools/analyze_mpc_control_time_wheels.py` and four synthetic tests in
`tools/test_analyze_mpc_control_time_wheels.py`. It pairs MPC's rear-wheel
hidden state at `candidate_command_stamp_ns` with encoder angle differences
over the preceding fixed 25 ms epoch, then scans ±1 and ±2 epochs. It labels
the comparison as an interval-average/point-state diagnostic—not independent
wheel-speed truth—and does not feed the result into production.

- Default-line five-lap bag: 1,275 matched control epochs / 2,550 wheel
  comparisons; point-vs-interval wheel-omega RMSE 4.185 rad/s, p95 absolute
  7.536 rad/s. Candidate-time pairing is markedly better than ±25 ms (44.6–
  44.9 rad/s RMSE) or ±50 ms (20.86–20.90 rad/s). The corresponding wheel
  surface-speed RMSE is 0.136 m/s.
- Fast-line diagnostic bag: 126 epochs / 252 comparisons; RMSE 11.946 rad/s,
  p95 absolute 24.474 rad/s, max 40.010 rad/s. Candidate-time pairing is
  markedly better than ±25 ms (151–154 rad/s RMSE) or ±50 ms (115.1 rad/s).
  Wheel surface-speed RMSE is 0.388 m/s.
- The encoder sample was within 5 ms of the candidate stamp for included
  pairs. The predicted plant state is 50 ms beyond the sensor-source stamp in
  these captures. Comparing its predicted wheel speeds with source-stamp
  observer state would be an epoch mismatch; the diagnostic instead compares
  at the candidate time. The bags still lack Unity `FixedUpdate`/applied-
  command IDs, and encoder angle is quantized. Thus the scan rejects a simple
  adjacent-packet shift as the explanation; it does not identify the exact
  native physics-step command effect or prove wheel-contact error.

Artifacts: `live_runs/mpc_odom_epoch_fix_practice_20261022_run1/`
`control_time_wheels.{json,csv}` and
`live_runs/mpc_ipopt_v1_fastline_diagnostic_20261022_run1/`
`trajectory_feasibility/control_time_wheels.{json,csv}`.

**PR-03 source-plant feasibility coverage.** Replayed all 80 usable
full-horizon accepted snapshots in the fast-line diagnostic bag, spanning
unwrapped progress 17.601–25.841 m, with the same 1 ms four-wheel source plant
and exact orientation-aware corridor check. All 80 captured candidate plans
ran 30 stages (750 ms) without exceeding the configured corridor tolerance;
minimum reported slack was −2.56 mm, within the 5 mm nonlinear tolerance.
This is limited evidence for that captured portion and this model—not a
whole-lap Unity feasibility claim.

Holding each snapshot's steering sequence and two queued commands fixed, the
constant-throttle source counterfactuals stayed within the 750 ms corridor in
80/80 cases at q=0.01, 75/80 at q=0.125, 72/80 at q=0.15, 71/80 at q=0.20,
63/80 at q=0.30, and 60/80 at q=1.0. These counts concern corridor only;
they do not establish appropriate speed tracking or a useful fastest control
schedule. The fixed-q probes are not an optimizer and do not show that the
IPOPT speed profile is globally feasible/infeasible. The separate terminal
state remains unrecoverable by the tested constant-q + captured-steering
continuations, which only establish failure of that local family.

Geometry audit remains: 27.5511 m lap, peak CSV speed 8.4875 m/s, minimum
centerline corridor slack after the configured 5 cm margin and aligned
footprint 0.1342 m. This rules out centerline geometry alone as the observed
failure in this audit; it does not certify tracking or dynamic feasibility at
every progress point.

Artifact bundle:
`live_runs/mpc_ipopt_v1_fastline_diagnostic_20261022_run1/trajectory_feasibility/`
`all_accepted_preterminal_q_sweep.{json,_bins.csv,_source_witness.csv,` plus
`_candidate_prior_ab.csv}`. The shorter multi-regime probe is retained as
`nearest_regime_q_sweep.*`; neither replaces the original trajectory.

**PR-04 candidate-prefix chain evidence.** Added
`tools/analyze_mpc_prefix_chains.py` with three synthetic tests and generated
per-action/per-chain reports for the post-odom-fix five-lap baseline and both
fast-line captures.

- Successful default-line five-lap bag: 10 candidate-prefix actions in 10
  separate one-action chains; no long chain occurred in that session.
- Fast-line practice bag: 46 prefix actions in 20 chains; longest chain 22
  consecutive actions, ending at unwrapped progress 41.159 m immediately
  before a nonlinear-rollout rejection. This older bag does **not** contain
  certified-plan snapshots, so same-state prior-plan A/B cannot be recreated
  from it.
- Fast-line diagnostic bag: 42 prefix actions in 17 chains; longest chain 21
  actions, from progress 25.998 to 28.261 m, followed by nonlinear rejection.
  In that chain steering sign reversed four times; primal residual p50/p95
  was 0.464/1.227, dual residual p50/p95 0.230/2.104, and logged predicted
  minimum corridor slack declined to 0.118 m at the last prefix. The entire
  chain's predicted slack stayed positive because each emitted action was
  only certified for its local 75 ms; the following 750 ms solve still
  failed.
- On 43 same-state candidate-vs-retained-plan source replays in this
  diagnostic bag, the candidate full suffix failed corridor in 43/43;
  the retained certified-plan suffix passed all remaining recorded controls
  in 22/43 and survived more stages in 29/43. The earlier 26/43 count used
  an off-by-one plan cursor and is superseded. This is offline support
  for testing prior-plan-first recovery when a certified plan exists, but it
  is not a closed-loop benefit or timing result. The five-lap default bag
  lacks snapshots, so its isolated prefixes are not comparable with this A/B.

**Warm-start priority check.** Inspection confirms the publication path stores
the actual issued candidate sequence, and
`mpc_rti_build_nominal_impl()` shifts its already-issued first control out
before constructing the next nominal. The existing test
`MpcReference.IssuedCandidateSequenceSeedsNextNominalAfterShift` directly
checks this mapping. The targeted C++ test target rebuilt and that test passed
(1/1). No extra warm-start code change was justified. The retained certified
plan remains a separate cursor-owned object in the node.

This is an association, not causality: the comparison set contains one
successful default session and two failed fast-line captures with different
states/path context. It cannot establish that chain length causes loss of
control. The second fast-line bag lacks the snapshots needed to evaluate the
prior plan, and neither bag has recorded live ADMM memory from the historical
solve.

**New diagnostic tests:**
`python3 -m unittest tools.test_analyze_mpc_control_time_wheels
tools.test_analyze_mpc_prefix_chains tools.test_audit_mpc_trajectory_feasibility -v`
passes 18/18; Python byte-compilation and `git diff --check` pass. The build
and targeted warm-start test above pass. No production controller, odometry,
weights, raceline or vehicle dynamics were changed.

**Next handoff work:** PR-03 cannot justify a path/speed change from the
current evidence; first collect a source-consistent speed feasibility
witness across the missing progress range (or state the precise unsupported
coverage). PR-04 should next implement and replay a full-source-plant-derived
continuation seed and compare it with candidate-prefix and retained-plan
suffixes on saved terminal states. The opt-in full-prior-suffix policy
remains OFF until both its quality advantage and its complete 20 ms callback
cost are measured. A supervised faster-line live run remains blocked by the
unavailable practice simulator endpoint; the existing failed runs do not
count as a usable lap.

#### PR-03 coverage extension from the second fast-line capture

The independent fast-line practice bag contains a wider set of usable full
horizons than the diagnostic bag. Replaying all 176 complete accepted plans
with the exact source plant gives full 750 ms corridor-valid witnesses over
unwrapped progress **17.585–38.433 m** (lap length 27.551 m). All 176 captured
plans pass the configured nonlinear corridor tolerance; minimum slack is
9.0 mm. Together with the diagnostic bag's 80 witnesses over 17.601–25.841 m,
the captured state coverage is substantially denser, but it still does not
cover the critical reference interval **10.88–17.59 m modulo the lap**.

Fixed-throttle counterfactuals with captured steering and the exact queued
commands stayed within the corridor for:

| Constant legal q | Full 750 ms corridor pass | Interpretation |
|---:|---:|---|
| 0.01 | 167/176 | Most robust corridor result in this sampled set; speed tracking is not certified by this count |
| 0.125 | 141/176 | Local corridor response only |
| 0.15 | 136/176 | Local corridor response only |
| 0.20 | 124/176 | Local corridor response only |
| 0.30 | 116/176 | Local corridor response only |
| 0.50–1.00 | 113/176 | These commands coincide in portions of the source-model response; not a monotonic throttle ranking |

By arc range, q=0.30 remains corridor-valid in 11/11 snapshots at 23–25 m,
but 0/11 at 25–27 m; q=0.01 passes 11/11 at both. At 27–29 m, q=0.01
passes 25/25 while q=0.125 passes only 4/25, q=0.15 2/25, and q≥0.20 0/25.
These are state- and steering-conditioned corridor outcomes, not a race-speed
schedule: q=0.01 may remain inside the corridor while badly under-target, and
the input is still positive drive rather than the unavailable hard-brake
mode.

The two fast-line failure points fall in the uncovered progress range, after
long local-prefix chains:

- Fast-line practice run: unwrapped progress 41.193 m = 13.642 m modulo the
  lap; reference speed 3.003 m/s, curvature −0.885 1/m; logged state
  `e_y=0.978 m`, `e_psi=0.828 rad`, `u=4.767 m/s`. The full-horizon candidate
  rejected at corridor stage 2 after a 22-action prefix chain.
- Fast-line diagnostic run: unwrapped progress 28.328 m = 0.777 m modulo the
  lap; reference speed 2.844 m/s, curvature −1.037 1/m; `e_y=0.812 m`,
  `e_psi=0.770 rad`, `u=4.860 m/s`, `r=−0.933 rad/s`. It rejected at corridor
  stage 2 after a 21-action prefix chain.

This localizes where more evidence is needed: high-speed/high-curvature
tracking and allowed-q deceleration in the 10.88–17.59 m interval, including
the first run's failure corner. It does **not** show the geometric line is
impossible, nor does the accepted-plan evidence show the actual vehicle can
track it. The two failure states have already accumulated 0.8–1.0 m lateral
error and ~0.77–0.83 rad heading error; changing the reference speed after
that point alone would not explain the preceding loss. No speed schedule is
promoted from this incomplete envelope.

Expanded artifacts are at
`live_runs/mpc_ipopt_v1_fastline_practice_20261022_run1/trajectory_feasibility/all_accepted_preterminal_source_q_sweep.*`.

### 2026-10-22 — full-prior-suffix live probe and corrected failure interpretation

Run `live_runs/mpc_ipopt_v1_prior_full_suffix_20261022_run4/` used the faster
IPOPT trajectory, the run-3 opt-in full-validated-prior policy, eight
Jacobian workers, and the practice batch-mode simulator. Its sealed manifest
records the loaded trajectory/config/image hashes. The experiment completed
**no lap** and had a collision, so it is not a faster-line success.

The retained diagnostics narrow what this run can establish:

- Before the first rejected cycle (bag time 1.7317 s, progress 24.008 m),
  tracking remained close to the reference: `|e_y|=0.0092 m`,
  `|e_psi|=0.0157 rad`, and measured forward speed was 7.865 m/s against a
  7.974 m/s target. The controller had published 60 full-horizon exact-valid
  commands and two exact-valid prior-plan actions; there were no candidate
  prefixes.
- The rejected callback took 34.316 ms to publication admission, over the
  strict 20 ms bound. It did not publish a new actuator command. The prior
  suffix was source-plant/corridor-valid, but its 2.429 ms validation finished
  after the deadline, so it was correctly not published. The callback's
  Jacobian stage consumed 29.332 ms wall time; the slowest worker reported
  about 3.24 ms CPU but 22.50 ms wall, consistent with a tail scheduling or
  worker-wait event rather than an expensive 22 ms single-thread calculation.
- The run supervisor was started only after the simulator was already
  running. It did not stop the simulator at the first rejected MPC diagnostic;
  the collision counter was first observed 0.656 s later, followed by bridge
  loss. This is a supervision-startup race in the experiment procedure, not
  evidence that the MPC intentionally continued issuing rejected outputs.
  The last published throttle, `q=0.3658`, remained the last actuator command
  after the rejection. The subsequent acceleration/collision interval is
  therefore **invalid for judging the IPOPT speed profile or source-plant
  tracking**. No trajectory change is justified from it.
- The captured source-plant fixed-input windows are poor over 750 ms
  (`u` RMSE 2.136 m/s; endpoint XY RMSE 0.599 m, p95 1.336 m), despite small
  one-step errors. These windows do not have enough matched, clean controls to
  attribute the long-horizon error, and the terminal interval is
  collision-contaminated. They are not evidence that the plant is accurate on
  this faster line.

The supervisor implementation already aborts on `rejected_solver`; the
failure here was that it was armed too late. Future supervised trials must
start `tools/watch_sim_run.sh ... mpc-lap-N ...` **before** launching Unity,
wait for its `WATCHER_READY` line, and only then start the recorder and
simulator. This preserves the required fault-on-rejection semantics and
prevents a missed diagnostic from turning a solver fault into an uncontrolled
drive interval. Do not change the controller fault policy to mask this
orchestration defect.

For the deadline event, the same 63 saved controller states were replayed
through the current offline RTI binary with the established solver profile at
four and eight Jacobian workers. Both replays produced identical statuses and
first commands. Four-worker cycle p50/p95/max was 5.00/7.10/7.80 ms; eight-
worker replay was 4.47/6.54/7.97 ms. This CPU-only replay mildly favors eight
workers at p50/p95 and shows no useful basis for changing the production
setting; it cannot reproduce Unity/ROS contention or establish a live worst
case. No production parameter or raceline was changed.

Artifacts added after sealing the run manifest:
`solver_inputs.csv`, `solver_replay_4workers.csv`, and
`solver_replay_8workers.csv`. The exported inputs explicitly identify
themselves as saved-state reconstructions and do not contain live ADMM memory.

**Next action:** resume PR-03/04 offline from clean pre-fault captured states;
do not repeat this unchanged run. Before any new live trial, arm and verify
the supervisor before Unity. A candidate speed/path adjustment still needs a
clean source-plant feasibility witness and a single-factor controlled
comparison; this run supplies neither.

### 2026-10-22 — corrected same-state prefix quality A/B (supersedes earlier counts)

Followed PR-04 of
`/home/akselmo/Downloads/SDU_Apex_MPC_Odometry_Fix_Next_Steps_Handoff_2026-10-22.md`.
This update is offline analysis of sealed practice captures; **no simulator was
launched and no production MPC/odom, weights, or trajectory behavior changed**.

The same-state audit cursor fix is now covered by
`test_candidate_prefix_snapshot_cursor_is_post_publication`. On a published
`accepted_candidate_prefix`, the diagnostic snapshot cursor has already
advanced, so the replaced prior action is at `cursor`; on a rejected cycle,
it is at `cursor + 1`. The previous 26/43 prior-full-suffix-valid count is
therefore invalid; corrected count is 22/43. The separate 29/43
“prior survives longer” result is unchanged.

Extended `tools/audit_mpc_trajectory_feasibility.py` to replay, from each
identical state and queued-command pair, both candidate and retained-plan
three-action/75 ms prefixes. It reports source-plant/footprint-corridor
validity, minimum slack, speed error, and the scalar nonlinear objective with
the runtime YAML weights. The objective calculation was cross-checked against
logged production nonlinear objectives for accepted full-horizon plans:

- Faster-line diagnostic capture: two accepted plans, objective relative
  differences 0.28% and 0.59%; their sampled replay state differences were
  below `2.6e-7` in `e_y/e_psi/u/v/r/steering`.
- Default-line five-lap capture: four accepted plans, relative differences
  0.002%, 0.25%, 0.002%, and 2.86%. The last is retained as a known parity
  outlier, not hidden; it does not affect the three-step comparison because
  that bag lacks retained-plan snapshots.

| Practice capture | Same-state prefix pairs | Both pass offline source/footprint screen | Lower 75 ms nonlinear cost | Median cost, candidate / prior | Median speed-error RMSE, candidate / prior |
|---|---:|---:|---:|---:|---:|
| Faster-line diagnostic run | 43 | 42 | Prior 28/42; candidate 14/42 | 9564.5 / 3030.8 | 2.072 / 1.961 m/s |
| Supervised faster-line run5 | 15 | 14 | Prior 9/14; candidate 5/14 | 10314.8 / 9294.5 | 1.510 / 1.502 m/s |
| Successful default-line five-lap run | 10 | 10 | Prior 8/10; candidate 2/10 | 24197.3 / 819.6 | 3.621 / 3.585 m/s |

The prior-prefix median minimum corridor slack was 0.2944 m versus 0.2944 m
for the candidate in the diagnostic capture, and 0.2606 m versus 0.2575 m in
run5. In the default-line capture they were 0.3354 m versus 0.3274 m
(candidate/prior). Thus, over the sampled states, the prior prefix often has
lower three-step objective and similar screened clearance. The successful
default-line bag did not record explicit retained-plan snapshots, but its
retained sequence can be reconstructed from the accepted full-plan and
published-control status history: all 10 reconstructed comparisons have
complete publication flags; the reconstruction algorithm also matches all
28 explicit snapshots in the faster-line run5 bag and all 43 in the earlier
fastline diagnostic bag. The reconstructed default-line comparison is still
conditional on the logged statuses and causality; it is not a live counterfactual.

The same-state full-suffix checks strengthen, but do not prove, the policy
hypothesis. The candidate's rejected full suffix fails all 10 default-line
cases, while the retained suffix passes 9/10. For the faster-line diagnostic
the candidate fails 43/43 and retained suffix passes 22/43; in supervised
fastline run5 the candidate fails 15/15 and retained suffix passes 0/15.
These are source-plant replays from the same captured states, not a simulated
closed-loop branch: each alternative would change the next state and future
solver outputs. The run5 full-prior policy was enabled, but no retained suffix
passed at its 15 candidate-prefix events, so it could not rescue that run.
The default-line and earlier fastline captures had that policy disabled.
Therefore the existing full-suffix switch is insufficient for the faster line;
the evidence motivates evaluating the handoff's quality-aware exact-validated
75 ms candidate/prior choice, not blindly enabling the switch or promoting a
new production default.

The supervised faster-line run remains **zero laps**. It was observed with
the watcher armed before Unity; 68 optimal, 4 feasible, 13 prior-plan, and 14
candidate-prefix commands were accepted before a nonlinear rejection. The
first collision count was observed 116 ms after rejection during shutdown, so
the run is not labeled collision-free. Its MPC timing stayed below 20 ms
(p50/p95/p99/max 6.592/10.351/11.456/11.479 ms); timing did not explain this
failure. The faster line is still unproven in closed loop.

Artifacts:

- `live_runs/mpc_ipopt_v1_prior_full_suffix_20261022_run5_supervised/recovery_ab_prefix_quality*`
- `live_runs/mpc_ipopt_v1_fastline_diagnostic_20261022_run1/trajectory_feasibility/recovery_ab_prefix_quality*`
- `live_runs/mpc_odom_epoch_fix_practice_20261022_run1/trajectory_feasibility/default_prefix_quality*`

Validation: `python3 -m unittest tools.test_audit_mpc_trajectory_feasibility
-v` passes **14/14**; Python compilation and `git diff --check` pass. Added a
CSV regression assertion so column ordering cannot silently shift
`prior_snapshot_source`, chain length, progress, and command values. All three
audits were regenerated with matching source/library/config hashes and their
candidate/prior CSV headers and row widths agree. The quality-aware policy is
not yet implemented or promoted; the next step is an offline held-out
decision replay covering default, IPOPT and steering-reversal states, then a
feature-off-by-default live trial only if it has a defined expected benefit.

**Next:** complete the faster-line source-plant feasibility/continuation
analysis on clean saved states and the uncovered corner interval; then obtain
the missing default-line same-state comparison before any recovery-policy
change. Any eventual policy trial must retain the 750 ms exact gate, 75 ms
recovery validation, q/steering envelope, and strict 20 ms admission rule.

### 2026-10-22 — quality-aware prefix policy on the faster IPOPT line

Followed PR-04 of
`/home/akselmo/Downloads/SDU_Apex_MPC_Odometry_Fix_Next_Steps_Handoff_2026-10-22.md`
with one supervised live trial. The policy was enabled only in a run overlay;
`quality_aware_prefix_selection_on_rejection` remains `false` in the
competition configuration. The full-prior-suffix preference remained off.
The established practice batch-mode launch and pre-armed watcher were used;
the faster IPOPT trajectory and physics were not edited.

Run: `live_runs/mpc_ipopt_v1_quality_prefix_20261022_run1/`.

- The controller loaded the 1,024-point, 27.55 m IPOPT trajectory. It issued
  109 commands and scored 115 diagnostics (including five localization-wait
  records). There were **0 completed laps**.
- Outcomes: 65 optimal full-horizon accepts, 10 feasible full-horizon
  accepts, 5 accepted candidate prefixes, 29 accepted retained-plan prefixes,
  and one unpublished nonlinear rejection. Across the 35 rejection-policy
  decisions, it accepted a retained-plan prefix 29 times, a candidate prefix
  5 times, then rejected when neither prefix passed exact validation.
- The prefix selector found the chosen prefixes locally exact-valid and
  selected between them by their runtime nonlinear cost; that does **not**
  certify a viable remaining 750 ms continuation. Tracking deteriorated
  during the run: diagnostic cross-track RMSE 0.185 m (max absolute 0.677 m),
  heading RMSE 0.165 rad (max absolute 0.516 rad), and speed-error RMSE
  2.641 m/s. The terminal rejection was a corridor failure at stage 2, at
  progress 28.763 m.
- The first collision message was received 110 ms after the unpublished MPC
  rejection, during termination; the run is therefore **not** described as
  collision-free. No evidence establishes that the collision caused the MPC
  rejection.
- Command-publication admission timing was p50/p95/p99/max
  **6.893/10.472/10.931/12.001 ms**, with zero of 109 published commands at
  or over 20 ms. This trial did not expose a timing overrun.
- Bag, metadata, manifest, overlay, `report.json`, and `cycles.csv` are
  retained in the run directory. The live evidence is a failed attempt to
  drive the faster line, not proof that the line is physically infeasible or
  that the quality-aware policy improves reliability.

Decision: **do not promote** the quality-aware policy or faster line. Its
75 ms exact-prefix comparison was insufficient to prevent cumulative loss of
tracking, and the policy selected the retained-plan prefix on most recovery
decisions. This is evidence for the handoff's next question—whether a local
prefix choice restores full-horizon viability—not evidence that the retained
plan alone caused the failure. Before another live attempt, replay the exact
selected controls and state chain from this run alongside the held-out default
and IPOPT captures; score the next-state/full-suffix feasibility and
continuation quality, not only the 75 ms prefix objective. Do not repeat the
same run unchanged.

The quality-aware branch is still default-off and all its published actions
passed its exact 75 ms source-plant/corridor gate and the existing 20 ms
admission gate. The run's terminal cycle published no replacement command;
the independent watcher terminated the run. Runtime logs and the bag should
be retained together for the next causal analysis.

### 2026-10-22 — full-prior-suffix policy on the faster IPOPT line

The follow-up changed only the run overlay setting
`prefer_full_validated_prior_plan_on_rejection: true`, retaining
`quality_aware_prefix_selection_on_rejection: true`; both remain disabled in
the competition YAML. Trajectory, map, physics and weights were unchanged.
An initial exploratory launch was not recorded because the bag recorder was
omitted; it is excluded as evidence and retained without deletion. The repeat
below had the watcher and bag recorder armed before batch-mode Unity started.

Run: `live_runs/mpc_ipopt_v1_quality_full_suffix_20261022_run2/`.

- The faster-line run completed **0 laps**. It issued 103 commands: 68
  optimal full-horizon, 1 feasible full-horizon, 1 candidate-prefix and 33
  prior-plan commands, followed by one unpublished exact nonlinear corridor
  rejection. The terminal rejection occurred at progress 28.547 m,
  `e_y=0.750 m`, `e_psi=0.487 rad`, `u=3.318 m/s`, target speed 2.801 m/s,
  curvature −1.065 m⁻¹; the candidate failed at stage 2. The 100-iteration
  terminal solve had primal/dual residuals 1.114/2.253.
- The full-prior-suffix check ran 35 times; 32 checks were exact-valid. This
  means the **remaining** suffix of the stored plan was valid, not that a new
  750 ms plan was available each cycle. As that plan was consumed, the checked
  horizon decreased from 29 controls to 3. Across the final eight valid
  checks, cross-track error rose from 0.404 m to 0.616 m while heading error
  rose from 0.213 rad to 0.407 rad. The next candidate-prefix attempt was
  accepted once, then the next cycle had no exact-valid option. Thus retaining
  a valid old suffix did not prevent accumulating tracking error or eventual
  loss of a certified command.
- The quality selector was exercised once, earlier in the run: both local
  75 ms prefixes passed exact validation and it selected the retained plan
  (prior objective 1,208 vs candidate 34,143; both minimum slacks 0.346 m).
  This does not demonstrate a quality or lap-completion benefit.
- Controller diagnostic tracking RMSE was 0.213 m cross-track, 0.138 rad
  heading and 2.550 m/s speed error. No lap time was recorded. The first
  collision message arrived 140 ms after the MPC rejection during termination;
  neither the rejection nor its cause is attributed to that later collision.
- Command-publication admission timing was p50/p95/p99/max
  **6.357/11.079/11.960/13.641 ms**, with 0/103 at or above 20 ms. Timing
  was not the observed failure.
- Fixed-input source-plant replay used 84 rolling 750 ms windows, each
  initialized from its logged plant state and propagated with recorded
  queue-slot-2 commands; future truth was scoring-only. No windows were
  excluded by the pose/twist closure screen. Endpoint XY RMSE was **3.916 cm**,
  p95 **5.965 cm**, max **6.272 cm**; 24/84 endpoints exceeded 5 cm. Maximum
  path error over each 750 ms was >5 cm in 28/84 windows (path-max RMSE
  4.04 cm, p95 6.08 cm, max 6.27 cm). At 750 ms, `u` RMSE was 0.0351 m/s,
  `v` 0.0018 m/s, yaw-rate 0.0132 rad/s and yaw 0.0031 rad. These are
  short-run, reinitialized fixed-input results, not a continuous lap-start
  forecast or a guarantee of MPC tracking.
- The separate logged-MPC-horizon comparison gave 75 ms XY RMSE 1.16 cm
  (max 3.41 cm) and 750 ms RMSE 0.302 m (max 0.588 m). Since the controller
  replans and publishes new commands every 25 ms, this comparison is **not**
  fixed-input plant accuracy; it includes planned-versus-later-executed
  command divergence.
- Bag, report, cycles CSV, trajectory audit, source-rollout case index,
  overlay and sealed manifest are retained in the run directory. The source
  rollout was recomputed from the bag with the current
  `libunity_source_plant_eval.so`.
- The run-specific trajectory audit found a 27.551 m, 1,024-point line with
  minimum aligned geometric footprint slack 0.134 m after the requested 5 cm
  clearance; its geometry-only test passes. Peak reference speed is 8.487 m/s,
  peak |curvature| 1.117 m⁻¹ and peak lateral-acceleration demand 8.40 m/s².
  This does **not** establish throttle/steering dynamic feasibility: the audit
  selected no preceding full-horizon plans for source replay, and explicitly
  makes no global infeasibility claim.
- Its two terminal same-state comparisons show the limitation of the prior
  plan: at progress 28.500 m the new 30-control candidate first violates the
  corridor at stage 5, while the retained plan is valid only through its
  remaining 2 controls; at 28.547 m the candidate fails at stage 3 and the
  retained plan has only 1 valid remaining control. These are source-plant
  counterfactuals, not closed-loop outcomes. Fixed-steering q=0.01 and q=1.0
  counterfactuals both first violate at stage 5; neither is an optimized or
  accepted plan, so this does not show that all legal throttle schedules fail.

Decision: **do not promote either recovery flag or the faster line.** This
second supervised trial still reached no lap and ended in a corridor rejection
despite the full-prior-suffix check. The measured short-window plant replay is
near 5 cm on average but has a tail above the target, while controller tracking
error was already large. The next step is offline same-state/continuation
analysis of the captured final steering-reversal chain and the IPOPT
feasibility audit—not another unchanged live run. Determine whether the loss
is dominated by solver quality, reference speed/input feasibility, or
steering/actuator state mismatch before changing weights or launching again.
The candidate and faster-line trial remain development-only; the trusted
reference and runtime physics were not edited.

Verification after this analysis: `python3 -m unittest
tools.test_export_mpc_run_report tools.test_audit_mpc_trajectory_feasibility
-v` passes 24/24; `cmake --build build/f1tenth_mpc --target
mpc_controller_node -j4` succeeds; `git diff --check` passes. No simulator,
MPC, recorder or watcher process/container remained running after the test.

## 2026-10-22 — handoff clarification and sealed faster-line follow-up

The active task brief is
`/home/akselmo/Downloads/SDU_Apex_MPC_Odometry_Fix_Next_Steps_Handoff_2026-10-22.md`.
This follows its PR-00 evidence, PR-03 feasibility, and PR-04 continuation
requirements; it does **not** treat a nominal trajectory-time estimate as a
successful lap. The earlier default-line five-lap capture and two IPOPT
attempts remain separate evidence classes.

### Fresh full-speed practice attempt

Run: `live_runs/mpc_fastline_5lap_20261022/`. The pre-launch manifest is sealed
and records the current workspace revision, dirty-source hashes, current dev
controller image, practice simulator image, practice map and requested IPOPT
trajectory. This was the current production MPC, batchmode with graphics, no
speed cap, no parameter overlay, eight Jacobian workers, and the strict 20 ms
publication-admission limit. Runtime physics, controller weights and the
trajectory file were not edited for this attempt.

- Outcome: **0 laps**. There were 240 published commands, 212 full-horizon
  exact accepts, 28 candidate-prefix accepts, then one unpublished exact
  corridor rejection at 6.1576 s (stage 2). The first collision message was
  138 ms later, during termination; collision is not identified as the cause
  of the MPC rejection.
- Publication-admission timing p50/p95/max was
  **6.764/10.366/12.673 ms**, with no sample at/above 20 ms and no callback
  interval over 25 ms. Timing did not explain the observed failure.
- Diagnostic tracking RMSE was 0.181 m cross-track, 0.207 rad heading, and
  2.384 m/s speed error. Near progress 39.48 m the state was still close to
  the path (`e_y=0.030 m`, `e_psi=-0.003 rad`). The first subsequent full
  horizon candidate failed at stage 25. In the ensuing prefix sequence,
  cross-track and heading errors grew to 1.040 m and 1.119 rad. At progress
  39.903 m, path curvature was −0.297 m⁻¹, measured yaw rate −1.169 rad/s,
  current physical steering −0.198 rad, while the next command was +0.305 rad;
  later physical steering/yaw rate became positive while the reference
  curvature remained negative. Residuals were poor and every cycle in the
  terminal 14-action chain hit the 100-iteration cap (chain maximum primal
  residual 6.926, dual 1.671). This is direct evidence that the published
  short-prefix sequence degraded tracking under a difficult, iteration-limited
  solve. It is **not yet proof** whether the initiating cause is the QP
  nominal/linearization, objective/constraints, command effect epoch, or an
  interaction. The bag lacks a Unity FixedUpdate command-effect ID.
- Warm-start audit for this run matched the next nominal first control to the
  issued prefix's next control exactly within 1e-5 on all 28 prefix
  transitions; this particular warm-start alignment is not the observed
  mismatch.

### PR-03/04 offline evidence from this capture

`trajectory_feasibility_latest.json` and its CSV witnesses were generated
with the current `libunity_source_plant_eval.so` and the exact bag. The IPOPT
line has 1,024 points over 27.551 m, maximum reference speed 8.487 m/s and
maximum |curvature| 1.117 m⁻¹. The geometry-only heading-aware footprint audit
passes the requested 5 cm clearance, with minimum remaining corridor slack
0.134 m. Geometry passing does not imply dynamic trackability.

One captured, exact-full-horizon-accepted plan beginning at progress 40.330 m
replayed through the source plant for all 30 stages with maximum discrepancy
from the controller's logged prediction below 4.4e-7 in the compared Frenet
states. It remained corridor-valid with 0.201 m minimum slack, but its speed
tracking RMSE was 0.594 m/s. This verifies source replay parity for that saved
plan—not future Unity accuracy or the candidate's global feasibility.

The candidate/prior counterfactual covers 29 saved states around the run:
both local 3-step prefixes were source-plant/footprint-valid on 28 states;
the prior prefix had lower local objective on 20/28; its remaining suffix was
exact-valid through all recorded remaining controls on 16 states and survived
more stages on 15. The candidate suffix failed its full captured continuation
screen on all 29. These are same-state offline comparisons; they do not
establish a closed-loop benefit from switching to the prior prefix.

The fixed-input source-plant score has 221 clean 750 ms windows: endpoint XY
RMSE 3.742 cm, p95 8.006 cm, max 9.182 cm; 45 endpoints and 46 full paths
exceed 5 cm. `u/v/yaw-rate/yaw` RMSE are 0.03385 m/s, 0.00348 m/s,
0.02429 rad/s, and 0.00655 rad. Separately, the recorded MPC-plan versus later
truth comparison has 750 ms XY RMSE 0.3985 m; because the controller replans,
that is not a fixed-input plant score. Neither result establishes a complete
lap-level prediction.

A saved-state weight/iteration probe on 28 turn states adds no promotion:
100 vs 200 iterations both yielded 9 exact-full-horizon accepts; multiplying
heading weight by 3 yielded 8; multiplying overspeed weight by 4 yielded 10.
The overspeed result is only one additional accepted saved state. It does not
justify a global weight change. These probes are offline reconstructions and
do not restore live ADMM memory.

### PR-00 evidence additions and decision

The machine-readable run comparison at
`docs/development/MPC_HANDOFF_BASELINE_COMPARISON_20261022.json` now includes
this sealed run, bag/report/cycle hashes, timing, separated fixed-input vs
logged-MPC rollout metrics, and the latest feasibility/counterfactual report.
The report previously named `source_rollout.json` was plain text, so it is
retained as `source_rollout.txt` to match its actual format.

Added 28 post-fix saved-state QP fixtures under
`live_runs/mpc_qp_post_odom_epoch_fix_20261022_fastline_run3/fixtures/`,
without overwriting the existing 125-fixture historical corpus or the prior
55-fixture post-fix delta. All 28 sidecars validate, round-trip byte-identically,
and reproduce the captured replay solver result exactly. These are
saved-state QPs, not historical live ADMM-memory snapshots.

#### PR-04 — controlled cold-start source-reference-seed check

To avoid conflating a changed nominal with prior-row ADMM memory, replayed the
same 28 `turn_states.csv` rows twice with per-row solver-memory reset, cold
ADMM initialization, identical queue/state, eight Jacobian workers, 100
iterations and identical weights: ordinary cold start (`independent`) versus
`reference-seed`. The fixture/state input SHA-256 is
`36c51056d4719cb8611a62768cad6a3a14829b1ffcfaa48d1d247a4c0c7b5185`; replay
binary SHA-256 is
`da882518c6d78f60b4cc486c269f9e6c96df16ddfc4bdbf4920ebdf02ef673ab`.

- Both arms produced exactly 3 optimal full-horizon accepts, 6 feasible
  full-horizon accepts, and 19 nonlinear-rollout rejections (9/28 exact
  full-horizon accepts each); no row changed status.
- On jointly accepted rows, first commands, nonlinear objective and corridor
  slack were identical; the median minimum slack was 0.166328 m in both arms.
  Maximum measured replay cycle was 7.890 ms vs 9.086 ms, respectively; this
  is offline host timing, not a live 20 ms claim.

This resolves an important experiment-design trap: the existing
`reference-seed` mode is not a distinct feasible-continuation method for a
cold solve. Ordinary cold nominal construction already generates the same
reference/speed seed and rolls it through the source plant. Therefore this A/B
shows no benefit and **does not satisfy PR-04(C)**, which requires a genuinely
different full-horizon source-plant-derived feasible continuation. No
production policy or behavior was changed. The next coding item is to build
that distinct offline continuation candidate, then compare it against the
captured candidate and certified-plan suffix at the same saved states; do not
repeat this equivalent cold-seed A/B.

#### PR-04(C) — source-plant-feasible blended continuation, replayed through MPC

Implemented a development-only same-state experiment. For every candidate /
retained-plan pair, formed three deterministic full-horizon control sequences
with candidate weights 0.25, 0.50, and 0.75; a short prior suffix is extended
by holding its last action. The current source plant replayed each sequence
from the saved plant state and two-slot command queue. For MPC replay, selected
the lowest logged nonlinear objective among source-valid blends with a
strictly nonnegative physical corridor margin. This is a seed-generation
search from available current-cycle data, not a hand-tuned production
controller. The replay executable accepts the sequence only through the
development-only `MPC_REPLAY_NOMINAL_SEED_CSV` environment variable. Production
launches/code do not read it.

| Capture | Source-plant blend trials | States with a selected positive-slack seed | Ordinary cold full-horizon accepts | Seeded exact full-horizon accepts | Interpretation |
|---|---:|---:|---:|---:|---|
| IPOPT fastline run 3 terminal turn | 87 | 5 | 3/5 | 5/5 | Two cold rejects became exact-valid. Among the 3 jointly accepted states, median nonlinear objective changed 23,188 → 34,101; median slack 0.1776 → 0.1818 m. |
| Successful default mintime run | 30 | 10 | 3/10 | 10/10 | Seven cold rejects became exact-valid, with no acceptance regression in this selected set. On the 3 jointly accepted states, median objective 3,200.9 → 3,194.0; median slack 0.3416 → 0.3426 m. |
| IPOPT fastline run 1, transfer capture | 141 | 24 | 17/24 | 17/24 | Four cold rejects were rescued, but four previously accepted states became rejected; net acceptance is unchanged. On 13 jointly accepted states, median objective worsened 19,207 → 32,602 and median slack fell 0.2074 → 0.1609 m. |

Every seeded accepted result passed the MPC replay's own exact 30-stage
nonlinear source-plant/corridor validator; these counts are not based on the
blend seed simply being valid. Runs reset ADMM memory at each row, use cold
initialization, eight Jacobian workers and the same 100-iteration solver
configuration. They do not reconstruct live ADMM state. The state sets are
selected states with available positive-slack candidate/prior blends, not
random or independent samples. Offline replay maxima were below 12 ms, but
that is not live timing under Unity/ROS load.

**PR-04(C) conclusion:** the continuation seed is implementable and can rescue
specific difficult states, including states on both the failed fastline and
successful default reference. Its transfer capture also demonstrates equal
net acceptance with material per-state regressions and worse objective/margin.
Therefore **do not promote it as a general recovery policy or run it in
production**. It is evidence that local full-horizon feasible control
continuations exist where a cold solve can fail—not evidence that this blend
selector is safe or faster closed-loop. Retain it as an offline comparison
tool while the next step examines the four rescue/four regression transitions
and whether their first-action/quality differences can be predicted from
causal current-state diagnostics without another solve.

**Decision:** do not promote the IPOPT trajectory, change production weights,
or rerun the same fast-line experiment unchanged. The best-supported next
engineering step is PR-04's rescue-versus-regression analysis of the
source-derived feasible-continuation seed on these captured states, followed
by PR-03's per-progress dynamic feasibility analysis from the last clean
full-horizon state. The key unresolved fact is
why the optimizer emits a direction-reversing, poor-residual candidate after
its full horizon first fails while a short prefix still passes its local
75 ms corridor check. Continue to distinguish solver-quality failure from
reference-speed/input feasibility and command-effect timing before making a
production change.

#### PR-04(C) follow-up — inspect the held-out rescue/regression transitions

Compared row-level solver diagnostics and initial controls for all changed
acceptance decisions in the 24-state fastline transfer capture. Four rows
changed rejected → exact full-horizon accepted (progress 18.0469–18.4759 m);
four changed exact full-horizon accepted → rejected (28.2254, 28.8713,
29.0836 and 29.4693 m). The states are in distinct portions of the captured
run, so this comparison is descriptive rather than a randomized paired-policy
result.

| Outcome under source seed | Progress m | Input (u,v,r) at saved state | Seed first action ((\delta,q)) | Solver diagnostic |
|---|---:|---|---|---|
| Reject → accept | 18.0469 | (-0.197, 0.044, 1.457) | (-0.090, 0.120) | Seeded residuals became small (primal 0.0178→1.91e-6; dual 0.0141→2.67e-5). |
| Reject → accept | 18.2333 | (-0.194, 0.078, 1.874) | (-0.026, 0.051) | Seeded solve accepted with 0.315 m minimum margin; primal rose to 0.0206 but remained accepted. |
| Reject → accept | 18.3645 | (-0.130, 0.039, 2.174) | (-0.075, 0.264) | Both solves had low residuals; the cold arm failed nonlinear validation, while the seeded arm passed with 0.071 m margin. |
| Reject → accept | 18.4759 | (-0.124, 0.052, 2.338) | (-0.074, 0.118) | Seeded residuals worsened to 0.0131/0.0182 but the exact nonlinear rollout passed. |
| Accept → reject | 28.2254 | (0.159, 0.229, 3.255) | (-0.469, 0.094) | Cold arm already had poor residuals (0.292/0.352) but passed exact validation; seeded arm failed solve/validation. |
| Accept → reject | 28.8713 | (0.442, 0.084, 1.873) | (-0.471, 0.161) | Seed worsened residuals 0.0113/0.0197→0.141/0.218. |
| Accept → reject | 29.0836 | (0.443, -0.142, 2.327) | (-0.512, 0.139) | Seed worsened residuals 3.81e-6/7.34e-5→0.0196/0.0189. |
| Accept → reject | 29.4693 | (0.330, -0.290, 3.019) | (-0.179, 0.195) | Seed worsened residuals 4.29e-6/7.68e-5→0.0199/0.0110. |

**Finding:** yaw rate is elevated in both rescued and regressed rows (about
1.46–3.26 rad/s); the intervals overlap substantially. A simple speed, yaw,
or corridor-margin threshold is therefore not supported by this tiny selected
set. In some rescued cases the seed helped numerical convergence; in others
it changed nonlinear acceptance despite similarly small residuals. The
regression rows show the seed is not reliably aligned with the solver's local
basin even when its own 30-step source rollout has positive corridor margin.
This fails PR-04's quality-aware-selection requirement. Do not add a heuristic
filter or production branch based on these rows. Next, compare the actual
candidate-prefix and retained-plan sequences at the same saved control epochs
where bag instrumentation contains both; if the necessary full suffixes or
post-publication cursor are absent, mark that arm unreconstructable and add
bounded nonblocking development capture before a new supervised run.

#### PR-03 extension — dynamic feasibility witnesses by progress bin

The PR-03 report already emitted geometric corridor bins, but dynamic blend
feasibility was summarized only over the whole bag. Added
`summarize_dynamic_feasibility_by_progress()` to the existing audit tool and
now emit a separate `*_dynamic_feasibility_bins.csv` plus per-bin JSON summary.
Each bin counts sampled saved states, tested candidate/prior blends, exact
30-stage source-valid sequences and strictly positive-slack witnesses. A bin
with no positive witness is explicitly labeled as a finite-search result,
never as physical infeasibility. Original audits, bags, and trajectory files
were left untouched; reports were regenerated to separate `pr03_*` paths.

| Capture | Unwrapped bins sampled | Bins with a positive-slack witness | 30-step source-valid blends / tested | Uncovered bins |
|---|---:|---:|---:|---|
| Fastline run 3 (terminal chain) | 8 | 6 | 30/87 | 40.5–41.5 m (9 saved states) |
| Default mintime five-lap capture | 4 | 3 | 23/30 | 86.0–86.5 m (2 saved states) |
| Fastline run 1 transfer capture | 17 | 11 | 56/141 | 33.0–33.5 m and 39.0–41.5 m |

For the failed run-3 terminal chain, the finite candidate/prior blend set has
no positive-margin full-horizon witness in the final 40.5–41.5 m interval,
although witnesses exist in earlier bins, including 40.0–40.5 m. This
localizes where these tested continuations stop working; it does not establish
that no other legal sequence can traverse those states. Fastline run 1 shows
the same pattern later in its run, while the successful default capture has
tested witnesses at its sampled high-curvature states. The run progress is
unwrapped across laps and should not be compared as a common lap-relative
coordinate without explicit modulo mapping.

Validation after the change: 22 focused Python tests pass, Python compilation,
JSON parsing and `git diff --check` pass. No production source, MPC behavior,
weights, or trajectory changed; no simulator was launched. The dynamic-bin
summaries and source witnesses are diagnostic and offline.

#### PR-04 follow-up — recheck next-epoch viability with the third backtrack

The first branch-viability replay accidentally configured only two line-search
backtracks, so it did not test the configured `alpha=0.125` trial. Corrected
`tools/replay_mpc_prefix_branch_viability.py` to use three backtracks and added
a unit test that pins this setting. Added an explicit `--state-row` selector
to `tools/search_mpc_source_plant_continuation.py` so candidate/prior rows with
nearly identical progress can be tested without silently selecting the wrong
branch endpoint.

Replayed all 14 candidate/prior pairs from the same run-5 bag using three
backtracks. All 28 prefixes remained exactly corridor-valid over 75 ms, but
all 28 subsequent 750 ms cold MPC solves still rejected at nonlinear corridor
validation (status 6), even after exercising the `alpha=0.125` trial. Failure
stages ranged from 0 to 18. Thus the earlier “all fail” conclusion is now
confirmed for the tested three-backtrack policy rather than the incomplete
two-backtrack configuration.

At the first failure state (`s=27.034 m`), the prior-prefix endpoint has a
source-plant continuation witness with 30/30 stages and +0.122 m minimum
corridor slack. It reduces speed from 5.56 to 2.34 m/s over 750 ms, with
1.02 m/s speed-reference RMSE; the production objective is 32,925 and is
dominated by cross-track (25,220), heading (4,345), and yaw-rate (336) terms.
This is a feasible but very slow, poor-tracking continuation—not a fast-line
recovery. The previous certified-plan suffix after the three-action prefix,
extended by holding its last remaining action, violates the corridor from
both branch endpoints (first violation at stages 16/17; minimum slack −1.306/
−1.264 m).

The prior-endpoint source witness was then tested as an MPC nominal seed with
three backtracks. It transferred to the nearby candidate-prefix endpoint and
remained exact source-plant valid (+0.127 m slack). From both endpoints MPC
accepted a full 30-stage exact rollout at `alpha=0.125` (status 8): prior
endpoint slack +0.100 m/objective 35,683; candidate endpoint slack +0.167
m/objective 31,617. The unseeded candidate endpoint still rejected at stage
18. Offline cycle measurements were 11.95 and 16.86 ms respectively; they are
not live timing evidence. The objective improved versus the source seed on the
candidate branch but regressed on the prior branch, so this two-state result
does not establish a quality-safe general recovery policy.

Artifacts are under
`live_runs/mpc_ipopt_v1_prior_full_suffix_20261022_run5_supervised/alpha_seed_search/`:
`prefix_branch_viability_v3/`, `branch_continuations/prior_27_034/`, and
`branch_continuations/candidate_27_034_seeded_with_prior_witness/` (plus the
explicit failed finite-start searches). These are offline counterfactuals on
the in-process source plant; they are not Unity live validation. No production
controller, weights, reference trajectory, or vehicle physics changed, and
no simulator was launched. PR-04's production/release gate therefore remains
open: broader held-out quality transfer and successful-default-line no-
regression evidence are still required before any live recovery change.

#### PR-04(C) — branch-set source-seed transfer and full-stage refinement

Ran the exact production-objective source-plant continuation search on all 28
predicted branch endpoints (14 candidate/prior pairs), with the same two
bounded feed-forward initializations. Only one endpoint (the prior branch at
the first pair, `s=27.448 m`) produced a full 30-step positive-slack witness;
the other 27 had no witness in this finite 10-knot search. This is a bounded
search result, not evidence that those states are physically infeasible.
Seeding the corresponding MPC replay from that witness produced one exact
full-horizon acceptance (`alpha=0.125`, +0.100 m slack); the unseeded replay
rejected at stage 16.

Transferred that same witness to its paired candidate endpoint. Its independent
source rollout remained corridor-valid (+0.127 m slack), and the exact MPC
replay accepted at `alpha=0.125` (+0.167 m slack). Thus the same control seed
rescued both sides of this one matched pair; it did not transfer to the later
branch states.

Added a development-only 30-knot parameterization to the source-continuation
search to refine the 10-knot feasible witness with a steering/throttle decision
at every 25 ms stage. From the prior endpoint, refinement lowered the exact
production objective 32,925 → 31,868 (3.2%), reduced speed-reference RMSE
1.018 → 0.912 m/s, increased terminal speed 2.34 → 2.69 m/s, and retained
+0.114 m source-plant corridor slack. SLSQP nevertheless reported
`Inequality constraints incompatible` after 8 iterations; the returned point
was exactly re-evaluated and was feasible, but this is not a robust optimizer
convergence result. The maximum absolute cross-track error remained about
0.67 m.

MPC replay from that refined seed accepted both paired endpoints, but the prior
branch had only +0.0039 m remaining corridor slack and residuals 0.215/0.102;
the candidate branch had +0.105 m slack. These are exact source-plant replay
results, not Unity runs. The near-zero margin and residual quality fail the
recovery quality gate, despite full-horizon acceptance. No policy is promoted.

Artifacts are under
`live_runs/mpc_ipopt_v1_prior_full_suffix_20261022_run5_supervised/alpha_seed_search/branch_seed_transfer_14pairs/`
and `branch_continuations/prior_27_034_fullgrid_refinement/`. The diagnostic
tool now supports `--decision-knots 10|15|30`; production code and the loaded
IPOPT trajectory remain unchanged. PR-04 still lacks broad accepted-quality
transfer, safe corridor margin across the difficult-state corpus, and a
no-regression comparison against the successful default line. Therefore the
handoff gate for a supervised fast-line run is not met yet.

#### PR-04 quality check — accepted nonlinear step can worsen exact cost

Compared the accepted MPC candidates against the exact production objective
of the same 30-stage feasible nominal seed, after applying the replay's actual
one-action warm-start shift and hold-last-action convention. At the prior
endpoint, exact cost rose 31,867.76 → 39,939.88 (+25.3%) while remaining
corridor-valid with only +3.94 mm slack. At the candidate endpoint, cost rose
34,305.76 → 36,213.57 (+5.6%) with +0.105 m slack. Both results are
`ACCEPTED_LINE_SEARCH` solely because the exact rollout/corridor validator
passed; `mpc_rti.c` does not require nonlinear-cost improvement relative to the
valid nominal. This is a concrete quality gap, not evidence of a physics-model
mismatch.

Next offline test: compare an objective-aware acceptance rule against current
acceptance while retaining the exact-valid nominal when the update worsens
cost. Count only cases where that nominal is itself full-horizon exact-valid;
do not turn a rejected candidate into an assumed-safe action. Check both the
fastline failure corpus and the successful default-line states before any
production edit. No production policy was changed in this step.

#### PR-04(D) — exact-objective lower-envelope screen on existing saved states

Using the existing blend-seed replays, compared three same-state candidates:
the positive-slack source-plant seed, the cold MPC result if exact-valid, and
the source-seeded MPC result if exact-valid. For the 39 preselected states
with a valid source seed (5 terminal-fastline, 10 default-line, 24 fastline
transfer), the lowest-cost exact-valid candidate covered all 39 states. The
winner counts were source-seed / cold-MPC / source-seeded-MPC = 1/1/3, 2/1/7,
and 13/7/4 for those three captures. Minimum chosen source-model slack was
0.173 m, 0.238 m, and 0.123 m respectively. In the 24-state fastline transfer,
the source-seeded MPC itself accepted 17; the lower-envelope selector chose
the source seed on 13 states, the cold MPC on 7, and the seeded MPC on 4.

This supports the *mechanism* of a cost-aware choice among already computed
and exactly validated candidates: it could avoid a worse nonlinear MPC update
and retain a feasible source-derived sequence on selected states. It is an
offline oracle/lower envelope, not a causal live-policy result: the cold and
seeded MPC solves were performed separately, these states were selected
because a valid blend seed existed, and no online candidate-generation cost
was measured. The evidence does not authorize production integration or a
fastline Unity run. Machine-readable rowwise results and input hashes are in
`live_runs/mpc_ipopt_v1_prior_full_suffix_20261022_run5_supervised/alpha_seed_search/quality_envelope/`.

A stricter same-solve comparison then used only the source-validated blend seed
and its seeded MPC result: keep the MPC candidate only if it is exact-valid and
its nonlinear objective is no greater than the already-valid seed. This
counterfactual selected an exact-valid full-horizon sequence for all 39
preselected states. It retained the seed / selected the MPC candidate on
1/4, 2/8, and 19/5 states across the terminal-fastline, default, and fastline
transfer captures. It avoided 1/5, 2/10, and 12/24 cost-increasing solver
outputs; the fastline transfer's minimum selected slack was +0.124 m. This is
the clearest current evidence for an objective-aware feasible-nominal guard.
It is still an offline replay of selected states: seed construction cost and
closed-loop behavior have not been measured. The next implementation remains
development-only—record/compare the nominal and candidate costs in the replay
harness first, then consider a default-OFF live experiment if that evaluation
passes the strict timing and default-line checks.

Next: implement the quality selector only in the offline replay harness and
evaluate it without oracle leakage on every saved state for which all actions
are causally available; compare cost, exact slack, accepted fraction and
runtime against current candidate-first behavior. If that passes both fastline
and default-line cases, then consider a default-OFF production experiment.

#### 2026-10-22 follow-up — current-build selector replay and speed-feasibility probe

Extended the offline continuation search so it can initialize from an exact
30-stage source-plant witness, apply explicitly requested throttle offsets,
and require additional corridor reserve. This keeps the refinement anchored to
a known-feasible sequence instead of restarting from geometric steering. Added
tests for seed loading, legal bounds, full-stage conversion, and quality
selection across seeded and unseeded rows. The focused suite now passes 41
tests. No production MPC, plant, weights, or trajectory changed.

The same saved IPOPT terminal state (`s=27.44789 m`, `u=5.555 m/s`) was used
for the new source-plant probe. The IPOPT speed reference there is 3.874 m/s;
within the next roughly 1.3 m it falls toward 2.75 m/s while curvature reaches
about `|kappa|=1.11 1/m`. The state therefore enters this braking/turning zone
about 1.68 m/s above its local reference. This is a measured state/reference
mismatch, not by itself proof that the path is globally infeasible.

With the exact steering sequence from the valid source witness held fixed,
constant-throttle source rollouts gave:

| Throttle | Full 750 ms corridor-valid | Minimum additional corridor slack | End speed |
|---:|:---:|---:|---:|
| 0.01 | yes | 0.147 m | 0.87 m/s |
| 0.05 | yes | 0.142 m | 1.23 m/s |
| 0.10 | yes | 0.078 m | 2.43 m/s |
| 0.15 | no | −0.327 m | 3.62 m/s |
| 0.20 | no | −1.344 m | 4.89 m/s |
| 0.25 | no | −2.195 m | 6.09 m/s |

This one fixed-steering sweep shows that a slower, legal source-plant
continuation exists from the state, while larger throttle values on that same
steering sequence leave the corridor. It does not identify the fastest
possible sequence or prove the IPOPT speed profile globally infeasible.

The witness-seeded, 30-knot optimization with a required extra 5 cm corridor
reserve returned a full 30-stage source-valid sequence with 11.72 cm minimum
slack, objective 32,205.7, progress gain 1.866 m, and end speed 2.547 m/s.
The original seed objective was 32,925.4, progress gain 1.857 m, and end speed
2.343 m/s. A +0.02 throttle-offset initialization also remained source-valid
(6.06 cm slack) and ended at 2.781 m/s, but its objective was higher at
35,725.3. The +0.05 and +0.10 offsets did not find a valid sequence. Thus the
small objective improvement is real in the source evaluator, but remains a
slow local continuation with substantial cross-track error; it is not a faster
raceline solution.

### Reproducibility check against the currently built replay executable

The earlier PR-04 report’s accepted status-8 replay used solver executable
SHA-256 `5e023718…`, whereas the current `build/f1tenth_mpc/mpc_bag_solver_replay`
is `b6f8d9f8…`. Replaying the identical `s=27.44789 m` state and identical
source-feasible seed through the current binary twice returned status 6 both
times (exact nonlinear corridor failure at stage 15/16); the source seed
itself still replays valid with +12.17 cm source corridor slack. The old
accepted result is therefore not reproducible on the currently built solver
and is not evidence for a current-build recovery. The binary, state, seed,
trajectory, and source-plant hashes are retained in the run artifact. This
does not identify whether the difference is solver-version behavior or the
candidate update moving away from the valid nominal; it demonstrates that
solver acceptance is binary/build-specific and must be rechecked.

Added a reproducible replay/report path that records every saved state, not
only states with an available source-valid seed. Current-build results:

| Capture | All saved states | Source-valid seed coverage | Ordinary cold exact accepts | Selector full-horizon valid | Selected-vs-cold cost increases on jointly valid states |
|---|---:|---:|---:|---:|---:|
| Fastline run-3 terminal subset | 28 | 5 | 9 | 11 | 1/9; worst +47.1% |
| Default mintime success | 1,276 | 10 | 1,194 | 1,201 | 1/1,194; worst +0.17% |
| Fastline run-1 transfer | 224 | 24 | 179 | 186 | 7/179; worst 17.57× at progress 38.610 m |

Selector composition on those full saved-state populations was, respectively:
6 cold-valid / 4 seeded-MPC candidate / 1 retained source seed / 17 no valid
action; 1,191 cold-valid / 8 seeded-MPC candidate / 2 retained seed / 75 no
valid action; and 162 cold-valid / 5 seeded-MPC candidate / 19 retained seed /
38 no valid action. All selected actions were exact-valid in the in-process
source plant. However, the fastline transfer contains a severe objective
regression versus the cold replay, and the default set contains a small one.
The selector also used only 5/28, 10/1,276, and 24/224 states with tested
source-valid seeds. Seed-generation and extra exact-plant-validation cost are
excluded from replay timing. This is still a counterfactual offline selector,
not a causal, timing-qualified live policy.

### Decision / gate

This produced concrete implementation and data: a reusable witness-seeded
feasibility refinement, complete-population offline objective-selector
reporting, and a throttle/corridor response measurement from the fastline
failure state. It does **not** make the faster line ready for the simulator.
PR-04 fails its quality/no-regression gate (including the 17.57× transfer
outlier), and the current replay binary does not reproduce the prior claimed
seeded acceptance. PR-03 also still lacks a source-plant-feasible, connected
full-lap control/speed schedule. No simulator was launched; no fastline lap was
achieved. The original IPOPT CSV and default mintime trajectory remain
unchanged.

Next: diagnose the progress-38.610 objective outlier and the current-build
stage-15/16 rejection from the exact state/seed; then determine whether a
source-valid low-speed continuation can be connected to the next speed-profile
segment without losing corridor margin. Do not promote the selector or launch
the fastline until that comparison passes on the difficult set and default
no-regression gate.

#### 2026-10-22 correction — quality selector must include the cold candidate

The v2 full-population selector had a comparison omission: on states with a
source-valid seed it compared only that seed against the seeded MPC result,
even though an ordinary cold replay had also been computed for the same saved
state. Consequently it could select a valid but much worse seeded result over
an already available valid cold result. The previously reported 17.57x
fastline-transfer “selector regression” at `s=38.6095073 m` was real for the
v2 selector, but was an offline selector defect—not evidence that cold MPC
itself selected that action.

Corrected the *development-only replay selector* to choose the lowest exact
nonlinear objective among all exact-valid candidates available offline:
ordinary cold MPC, source-plant seed, and source-seeded MPC. Reran all three
current-build saved-state populations into separate `..._v3` artifact
directories. This does not alter production code, and the selector is an
offline lower envelope: computing both cold and seeded solves plus source
validation is not a measured or feasible live-cycle policy.

| Capture | States | Seed coverage | Cold exact-valid | Seeded MPC exact-valid | Selector exact-valid | Selection counts (cold / seed / seeded / none) | Improvement vs cold | Min selected slack |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Fastline terminal subset | 28 | 5 | 9 | 5 | 11 | 7 / 1 / 3 / 17 | 2 lower-cost, 0 higher-cost | 14.87 cm |
| Default mintime capture | 1,276 | 10 | 1,194 | 10 | 1,201 | 1,192 / 2 / 7 / 75 | 2 lower-cost, 0 higher-cost | 0.40 cm |
| Fastline transfer | 224 | 24 | 179 | 17 | 186 | 169 / 13 / 4 / 38 | 10 lower-cost, 0 higher-cost | 0.51 cm |

The key outlier is now explained more specifically. At `s=38.6095073 m`, cold
MPC is status 0 after one iteration, with primal/dual residuals
`5.72e-6 / 1.03e-4`, objective 3,614.21, and 17.56 cm slack. The exact-valid
source seed has objective 92,606.80; seeding MPC from it reaches its 100-step
limit with residuals `0.135 / 0.0282` and objective 67,108.57. The corrected
selector therefore chooses cold MPC. This points to a poor source-seed basin
and solver convergence sensitivity at this state, not a source-plant/Unity
discrepancy. Across the fastline transfer set, 12/17 exact-valid seeded MPC
outputs are still costlier than their source seed, despite exact corridor
validity.

The fastline remains gated. The corrected lower envelope still finds no tested
action at 38/224 transfer states, and its minimum selected transfer slack is
only 5.1 mm. The terminal subset has only 11/28 states with a selected exact-
valid action. The default set's 1,201/1,276 result is a counterfactual
upper-bound over sparse source seeds, not a production acceptance rate or
closed-loop improvement. PR-03 still lacks a connected full-lap feasible
control/speed witness. No simulator was launched, no production behavior was
changed, and neither line's lap time changed.

Validation after this correction: focused replay/feasibility pytest set: **43
passed**; Python byte-compile passed. Reports and rowwise CSVs are in the three
`mpc_seed_replay_currentbuild_v3/` folders listed in
`docs/development/MPC_HANDOFF_BASELINE_COMPARISON_20261022.json`.

Next useful step is not to promote the lower envelope. Investigate the
source-seed construction at the 12 costly fastline states and the 38 states
without any tested valid action, using their matched cold plans as starting
points for source-plant continuation. Then evaluate connected continuation
across adjacent epochs and preserve objective/slack. Only a broad, causal
recovery result with default-line no-regression can unlock a supervised
fastline run.

#### 2026-10-22 follow-up — source-nominal merit guard and cold-plan continuation

Extended the offline replay report to score a distinct, potentially causal
same-cycle rule: compare the exact-valid source nominal against the exact-valid
MPC update and retain the nominal if the update's exact nonlinear objective is
higher. This is separate from the three-candidate offline lower envelope,
which also knows the result of an independent cold solve and is therefore not
a live one-solve policy. Focused replay tests now cover both comparisons.

Across current-build seeded-state replays the same-cycle rule retained the
source seed / accepted the update on 19/5 fastline-transfer states, 2/8 default
states, and 1/4 terminal-subset states. It guarantees no objective increase
relative to the supplied source seed by construction. That does **not** imply
no regression relative to an independently computed cold solution: on the
fastline transfer, 7 of the 17 states where both cold and guarded actions were
valid remained higher cost than cold, worst at progress 38.6095 m (67,108.57
versus 3,614.21, +17.57×). The prior/seed quality itself matters; merely adding
a seed-relative merit check cannot recover a better cold basin without paying
for another solve or having a better causal nominal already available.

For the progress-38.6095 outlier, exported the accepted cold plan from the
current replay executable and used it as the initial sequence for the exact
source-plant continuation search. The cold plan is exact corridor-valid with
17.56 cm slack; source optimization refined it to a 30-stage witness with
17.06 cm slack, 3.096 m progress gain, 2.782 m/s terminal speed, 0.229 m/s
speed-error RMSE, and production nonlinear objective 3,517.18 (versus 3,614.21
for the cold plan). The seed's 5 cm extra reserve requirement was met.

Replaying MPC from that source witness produced status 0 after one ADMM
iteration, with small residuals (`0.0156` primal, `0.00208` dual) and an exact
corridor-valid result (17.03 cm slack)—but the exact objective rose to
40,154.52, **11.41× the source seed**. An independent source replay of the
candidate's diagnostic control list attributes most of the cost to slip
(22,285), cross-track (10,646), heading (5,796) and speed (1,649), with early
throttle pulses up to 0.655 and 0.808 m/s mean speed overshoot. The diagnostic
prints controls at six significant digits, so replaying that rounded sequence
gives total cost 40,752.52 rather than the solver's 40,154.52 (1.49%
difference); the component split is indicative, not exact attribution. The
source seed itself has full-precision objective terms dominated by cross-track
1,873, heading 915 and slip 531. This is direct evidence that “optimal” QP
convergence plus exact corridor validity can still produce a very poor
nonlinear merit step. It is not, by itself, evidence of a rollout-vs-Unity
mismatch.

Saved-state artifacts for the corrected all-candidate lower envelope and the
same-cycle guard are in the three `mpc_seed_replay_currentbuild_v4/` folders;
the detailed cold-plan/source-seed comparison is in
`.../mpc_seed_replay_currentbuild_v3/cold_plan_diag/source_seed_merit_diagnostic.json`.
The machine-readable ledger includes report hashes and metric summaries.

Decision: this strengthens the case for a **development-only nonlinear
objective merit/backtracking experiment**, not a live-policy change. Before
production use, evaluate candidate-vs-current-nominal exact objective on a
broad held-out set with the actual causal retained-plan state, preserve exact
corridor gates, and test acceptance/quality on default and fastline sets. The
fastline remains unapproved: only 186/224 transfer saved states and 11/28
terminal-subset states have any selected exact-valid 750 ms action; transfer
minimum selected slack is 5.1 mm; source seeds cover only 24/224 states; and
there is no connected full-lap control/speed witness. No simulator was
launched and no production source, weights, plant, or trajectory changed.

#### 2026-10-22 — exact nonlinear objective merit guard: complete saved-state replay

Implemented an opt-in exact-merit check in `mpc_rti.c`. When enabled, the RTI
candidate is compared against the current nominal only after both are scored
with the exact 30-stage source-plant rollout, objective, and corridor test. If
an exact-valid QP update is worse than an exact-valid nominal, bounded
backtracking fractions (0.5, 0.25, 0.125) are tried. The solver accepts the
first exact-valid strict objective improvement; if none improves, it keeps the
exact-validated nominal. The option remains disabled in the ROS production
node; the replay executable enables it only for this offline experiment.

Replayed every saved cold-start state in the existing three PR-04 populations
using each corpus's recorded trajectory and v4 baseline. Also replayed the
matched source-seeded subsets. The rowwise outputs are in the
`merit_backtracking_currentbuild_v5/` subdirectories next to each v4 corpus.

| Saved-state corpus | States | Exact-valid before → after | Recovered states | Validity regressions | Objective worse / better / equal on jointly valid rows | Merit-triggered rows | Selected alpha (0 / .125 / .25 / .5) |
|---|---:|---:|---:|---:|---:|---:|---|
| Default mintime | 1,276 | 1,194 → 1,252 | 58 | 0 | 0 / 363 / 831 | 363 | 3 / 23 / 51 / 286 |
| Fastline transfer | 224 | 179 → 202 | 23 | 0 | 0 / 50 / 129 | 50 | 26 / 11 / 39 / 19 |
| Fastline terminal subset | 28 | 9 → 12 | 3 | 0 | 0 / 1 / 8 | 1 | 0 / 0 / 1 / 0 |

The other 45 fastline-transfer and 82 default-line line-search attempts were
corridor-repair attempts, not objective-merit triggers. The matched
source-seeded replays remained exact-valid on 24/24 fastline-transfer, 10/10
default, and 5/5 terminal states. Where their nominal was exact-valid, the
selected result never exceeded its exact nominal objective. This is
saved-state solver evidence, not closed-loop lap evidence or proof of actual
command-effect epoch alignment.

Replay timing is diagnostic only. On the 1,276 default states, maximum replay
cycle time was 19.36 ms, with no row at/above 20 ms and no backtrack-budget
skips. That gives insufficient timing margin/evidence to enable it live:
offline replays omit the concurrent Unity/ROS scheduling environment and do
not establish callback-to-publication timing. The online command-epoch-valid
signal is also deliberately unset. No simulator was started, no trajectory or
weights changed, and no lap-time result changed.

Validation: rebuilt `mpc_controller_node`, `mpc_bag_solver_replay`, and
`test_vehicle_model_source_rollout`; the focused CTest passed. Focused replay,
prefix-chain, and feasibility pytest selection: **15 passed**.

**Decision:** this is a measured offline improvement to exact-valid
full-horizon solver output on all three saved-state populations, with no
measured objective or validity regressions. It does not yet satisfy all of
PR-04: candidate/prior prefix-chain policy, command-epoch safety, and live
timing remain unresolved. It therefore stays default-off and does not
authorize a fastline simulator attempt. The fastline also still has no
connected full-lap source-plant-feasible schedule; the IPOPT reference and its
estimated 5.367 s time remain unchanged and unachieved.

Next: inspect the newly recovered fastline states by stage/progress and
determine whether their exact-valid full plans can be connected to subsequent
captured epochs without discontinuous first commands or loss of corridor
slack. Keep the IPOPT terminal-prefix run separate from clean pre-fault
windows. Only after this and the command-epoch gate pass should the feature be
exposed behind a default-OFF live parameter and tested on the default line
before the fastline.

#### 2026-10-22 — fastline chain: retained-plan availability and source-seeded continuation

Continued PR-03/04 offline on the exact captured fastline chain. No simulator
was launched. The chain is rows 201–223 in
`live_runs/mpc_ipopt_v1_fastline_practice_20261022_run1/cycles.csv`, from
progress 38.6095 m through the terminal rejection at 41.1929 m.

The capture does **not** contain a retained certified-plan branch to replay at
these epochs: `prior_plan_chain_length=0` and `prior_plan_exact_valid=false`
for all 22 published candidate-prefix cycles. The actions therefore cannot
be compared against an actual historical certified plan. The offline
`candidate_prior_ab.csv` uses a prior sequence reconstructed from published
control history; that is a counterfactual comparator, not the retained
full-horizon plan required by PR-04 B/D. On the actual chain, every published
action was only three-stage validated; the solver reached its 100-iteration
cap with large residuals, and the candidate full-horizon failure moved from
stage 28 at the first row to stage 4 by chain row 22, then stage 2 for both
candidate and unavailable prior at the final rejection.

Checked source-plant continuation at the first and seventh chain states using
the saved 22-state/internal-state snapshots, the exact fastline, the current
`libunity_source_plant_eval.so`, legal `q >= 0.01`, 30 stages, and a required
5 cm additional corridor reserve:

| Progress | Source-plant witness | MPC replay seeded with that witness | Same-state cold replay |
|---:|---|---|---|
| 38.6095 m | Valid, 17.67 cm minimum slack, +3.026 m progress, terminal speed 3.057 m/s, source objective 1,131.56 | Exact full horizon valid; status 8 after one iteration, line-search alpha 0 retained the seed; objective 1,121.75, 18.17 cm slack | Exact full horizon valid; objective 3,614.21, 17.56 cm slack |
| 39.5799 m | Valid, 22.84 cm minimum slack, +2.302 m progress, terminal speed 2.711 m/s, source objective 40,388.15 | Exact full horizon valid; status 8 after one iteration, alpha 0.25; objective 33,797.69, 23.23 cm slack | Exact full horizon valid; objective 53,507.38, 23.01 cm slack |

At 40.1702 m the previously retained source-search witness also passes the
source corridor, but its endpoint speed is only about 2.02 m/s and the
source-seeded MPC objective is about 88,865 with poor ADMM residuals. At
40.3089 m, the 10-knot source search found no witness (best slack −0.426 m)
and SLSQP exited early; this is **not** proof of physical infeasibility.
Search results and their exact inputs are under
`live_runs/mpc_ipopt_v1_fastline_practice_20261022_run1/continuation_blend_search/pr04_chain_row201_currentplant/`,
`.../pr04_chain_row207_currentplant/`, and
`.../merit_backtracking_currentbuild_v5/source_continuation_terminal/`.

Interpretation: source-seeded full-horizon plans materially improve exact
validity/objective at these two saved states, so the line is not shown to be
geometrically impossible there. But the witnesses do so by shedding speed;
their low terminal speeds and large tracking costs do not validate the
IPOPT line's 5.367 s estimate. They are isolated 750 ms witnesses, not a
connected schedule or a production recovery policy. Together with the absent
retained-plan snapshot and weak solver residuals, this does **not** pass PR-04
or authorize another fastline simulator run. The current merit/backtracking
change remains replay-only/default-off; no production policy, plant, weights,
or trajectory changed in this experiment.

Next PR-04 data requirement: capture the actual certified full-horizon plan
and cursor (if one exists) alongside each rejected candidate and its nominal
suffix, plus the exact float32 issued command and decision epoch. On the
existing bag, continue only with source-seeded counterfactuals and label the
retained-plan arm unavailable. Do not infer a causal prior-plan benefit from
the reconstructed history comparator.

#### 2026-10-22 — preserve full per-cycle solver sequences in bag exports

Inspected the fastline SQLite bag's `/mpc/diagnostics` CDR directly, then
re-exported all 229 diagnostic messages through the ROS bag decoder. The raw
messages contain the 30-entry `candidate_controls` and `nominal_controls`,
decision/source/issued command stamps, plant state, ADMM diagnostics, and
candidate/prior revalidation results. They contain **no**
`certified_plan_snapshot`; `prior_plan_actions_remaining` is zero throughout
the chain. Thus the absence of a prior plan is in the recorded run state, not
merely an omission by the previous CSV exporter.

Updated `tools/export_mpc_run_report.py` to emit a sibling JSONL sidecar with
the complete decoded diagnostic payload for every MPC cycle, in receipt-time
order. It records exact command epochs and preserves candidate/nominal/retained
plan data when available; it never reconstructs a missing certified suffix.
The cycle summary CSV remains compact. The report schema is now
`sdu_apex_mpc_run_report/v4` and links the sidecar path/count. Actual fastline
re-export: 229 JSONL diagnostic records, with the original bag, report, and
cycle CSV retained unchanged under
`live_runs/mpc_ipopt_v1_fastline_practice_20261022_run1/continuation_blend_search/diagnostic_export/`.

Validation: `python3 -m pytest -q tools/test_export_mpc_run_report.py` —
**11 passed**. The exporter decoded both custom plant-state and drive-command
CDR messages directly; this did not require generated message Python packages.
No simulator was started and no production MPC behavior changed.

The row-201 source witness also gives a useful counterfactual through the
failure corner. At the nearest progress samples (within 6.3 cm), its
source-predicted state versus the subsequently observed candidate-prefix run
was:

| Witness progress / captured progress | Witness e_y / e_psi / u / r | Captured run e_y / e_psi / u / r |
|---:|---|---|
| 39.52 / 39.58 m | +0.025 m / +0.031 rad / 5.45 m/s / −1.00 rad/s | +0.078 m / +0.083 rad / 6.16 m/s / −0.77 rad/s |
| 40.13 / 40.17 m | +0.051 m / +0.035 rad / 4.51 m/s / −1.48 rad/s | +0.177 m / +0.189 rad / 5.58 m/s / +0.69 rad/s |
| 40.35 / 40.31 m | +0.059 m / +0.031 rad / 4.18 m/s / −1.63 rad/s | +0.218 m / +0.285 rad / 5.75 m/s / +1.00 rad/s |

This is an **offline source-model counterfactual**, not a matched live or Unity
validation: the two paths use different controls, and the witness is slower.
It does show that, from the same initial captured full state, the exact source
plant evaluator can find a corridor-valid trajectory that keeps heading/yaw
with the negative-curvature section, while the live candidate-prefix sequence
diverges. Combined with 100-iteration caps and the high residuals, this
supports prioritizing full-horizon nominal/solver quality over changing
source-plant coefficients. It does not justify using this one saved witness as
a runtime policy or claim a faster lap.
