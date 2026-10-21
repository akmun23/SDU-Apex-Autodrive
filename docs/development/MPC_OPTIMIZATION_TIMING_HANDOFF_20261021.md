# MPC optimization, timing and racing handoff — execution record

**Handoff:** `SDU_Apex_MPC_Optimization_Timing_and_Racing_Performance_Handoff_2026-10-21.md`  
**Baseline inspected:** `bdca33ac97aebc784dee3ca4b49449789dbbde94`  
**Updated:** 2026-10-21  
**Status:** partial execution, with remaining roadmap work and blockers explicitly listed. This is not a claim that the handoff's final acceptance criteria have passed.

## Outcome

Implemented and live-verified the first observability/performance slice without changing vehicle physics, MPC weights, the raceline, the throttle envelope, the hard corridor, or the solver's recovery policy:

1. Exact 1 ms source-plant output samples are now cached from the already evaluated nonlinear rollout for diagnostic stages 1, 2, 3, 5, 10, 20 and 30; diagnostics no longer replay those plant steps. A short accepted prefix publishes only the stages it actually validated.
2. High-rate diagnostics use a bounded, replaceable one-frame queue and a lower-priority worker for final JSON stream materialization and ROS publication. No diagnostic frame was dropped in the live capture. **JSON field construction still runs in the callback**; it is measured, not claimed moved off-thread.
3. Added a versioned, compact JSON + exact-stamp cycle-CSV report with progress bins, outcome-specific convergence/timing statistics, worker CPU/wall/TID data, rho/factorization/regularization counters, first-fault scoring cutoff, run hashes and explicit measurement-regime labels. Fixed cutoff, timing-unit, worker selection, lap-counting and report-size defects found during validation.
4. Repaired stale defaults in the temporary Jacobian replay launch so it inherits the production YAML and only applies explicitly requested overrides.
5. Added the `mpc_performance` bag profile, retaining plant/control/truth/controller timing channels without recording high-volume lidar.

The live practice run completed five uncapped laps with no collision and no publication gate at or above 20 ms. It did **not** meet the handoff's 10 independent five-lap campaign, sub-5-second racing objective, low tracking-error goal, or stable-pace requirement. No solver/physics/weight/raceline change is promoted from this instrumentation run.

## Fresh live evidence

- Bag: `live_runs/mpc_diag_async_worker_20261021/run`
- Metadata: `live_runs/mpc_diag_async_worker_20261021/run_metadata.json`
- Compact summary report: `live_runs/mpc_handoff_optimization_20261021/reports/diag_async_worker.json` (about 37 KB)
- Per-cycle CSV: `live_runs/mpc_handoff_optimization_20261021/reports/diag_async_worker.csv` (about 1.7 MB)
- The JSON contains summary/progress bins; the paired CSV carries all **1,352 decoded diagnostic cycles** (1,347 command publications plus five localization-wait cycles). The raw bag remains the authoritative full diagnostic record.
- Simulator: pinned practice image, batch mode under Xvfb with graphics; development stack built from the working tree; original practice map and mintime trajectory; no speed cap; diagnostics enabled.
- 1,347 published control cycles; all 1,347 bridge request stamps and DriveCommand stamps matched, with zero reported command mismatches after float32-aware comparison. 0 Unity FixedUpdate/effect epochs are observable from the permitted bag interfaces, so command receipt is not mislabeled as application.
- Outcomes: 1,228 `accepted_optimal`, 88 `accepted_feasible` (iteration-limited but exact-valid full rollout), 31 `accepted_candidate_prefix`, 0 prior-plan actions; 103 solver iteration-cap cycles. The prefix status remains distinct from full-horizon acceptance.
- Publication gate p50/p95/p99/p99.9/max: **6.19 / 9.18 / 10.40 / 14.04 / 16.22 ms**; 0/1,347 at or above 20 ms; no callback interval exceeded 25 ms. This is one run, not a worst-case guarantee.
- Bridge request-to-response p50/p95/p99/max was **2.89/28.70/29.03/38.16 ms**. This is bridge response telemetry, not the Unity command-effect phase.
- Jacobian wall p50/p95/p99/p99.9/max: **2.70 / 3.24 / 4.70 / 7.01 / 10.38 ms**. Summed worker CPU was 16.61 ms median and 22.03 ms max. Eight workers were reported on each solve; 32 distinct OS TIDs appeared during the run, consistent with the already observed multiple executor-owned OpenMP teams. This capture did not include a fresh full-run host scheduler/frequency trace.
- Diagnostic producer field-build p50/p95/p99/max: **0.206 / 0.284 / 0.325 / 1.073 ms**. Queue enqueue max 0.036 ms; asynchronous worker queue-wait p99.9/max 0.513/0.870 ms; stream materialization p99.9/max 0.021/0.055 ms; publish max 1.130 ms; 0 replaced frames; 0 additional plant steps. Worker “last” timings are sampled before enqueuing the current diagnostic, lack a completion-frame ID, and are **not frame-correlated**; treat them as coarse worker-overhead distributions only.
- Five completed laps, 0 collisions. Distinct lap times from the bag's lap-completion events were **6.892, 6.169, 6.774, 6.444, 6.329 s** (mean **6.521 s**). Pace is not consistently improving and is not sub-5.
- Tracking diagnostics: cross-track absolute error p50/p95/max **0.033/0.241/0.520 m**; heading absolute error p95/max **0.204/0.375 rad**; speed absolute error p95/max **1.422/5.030 m/s**. These are controller diagnostic state/reference errors, not a substitute for truth-scored vehicle-model rollout accuracy.
- Controller-state cross-track RMSE was **0.110 m**, heading RMSE **0.082 rad**, speed-reference error RMSE **0.717 m/s**. The estimated duration with `|e_y|>5 cm` was **11.26 s**, integrated over observed diagnostic intervals ≤100 ms; it is an MPC-state estimate, not truth-scored off-path time.
- A separate evaluator replay of this bag (not the run-report JSON) scored odom and 750 ms command-input rollout against Unity truth. Production odom had `u` RMSE `0.0459 m/s`, `v` RMSE `0.0224 m/s`, and gyro-derived yaw rate; integrated relative-XY endpoint error was `0.176 m` (p95 `0.225`, max `0.251 m`). The un-reanchored source-rollout replay had 750 ms XY endpoint RMSE `0.235 m` (p95 `0.456`, max `1.080 m`), forward-speed RMSE `0.528 m/s`, yaw-rate RMSE `0.537 rad/s`, and yaw RMSE `0.137 rad`. This remains materially above the requested few-centimeter rollout target; the new run is not a precision win.

Prior comparison artifacts remain under `live_runs/mpc_handoff_optimization_20261021/reports/`. They include successful five-lap runs and failed repeats, and must not be averaged together as if they were paired trials. Prior progress is in `MPC_PERFORMANCE_MULTI_SQP_PROGRESS_20261021.md`.

## Checklist against the handoff

Status key: **Done** = implemented and evidence exists; **Partial** = concrete work/evidence exists but exit criterion remains; **Open** = not implemented or not validated. Deferred items are not represented as successes.

### Measurement contract C1–C3

| Item | Status | Evidence / remaining work |
|---|---|---|
| C1: pin code/build/CPU/runtime/map/trajectory/config/start metadata | **Partial** | `run_metadata.json` records base commit, compiler/build mode, CPU topology, executor/Jacobian workers, image IDs, map/trajectory/config hashes, startup pose and supervisor. The development workspace was bind-mounted. A runtime-source manifest (`3ce8c851…2781c18`) was reconstructed after the run from source files whose mtimes precede bag start; `runtime_hash_sealed_at_launch=false`. The report-generation diff hash is not evidence of runtime source identity. Host scheduler/frequency trace and supervisor content hash remain missing; container limits are recorded as unspecified, not measured. A named manifest covering every required success/failure/terminal/held-out replay group is also incomplete. |
| C1: command/effect epoch joins | **Partial** | Report joins exact bridge request stamp and exact DriveCommand stamp, never nearest-arrival matches. Unity application/effect epoch remains unknown because neither interface exposes a FixedUpdate command ID; runtime receipt must not be called effect. |
| C1: fault cutoff and separate map/truth scoring | **Partial** | Report cuts at first collision, bridge fault or unpublished production rejection statuses and keeps truth/odom streams separate. New run has no cutoff event. No in-process final-two-second fault ring buffer. |
| C2: status, residual hotspot, prefix, slack, timing and worker metrics | **Partial** | Compact JSON/CSV include status percentages, normalized distinct solver termination causes (optimal, iteration cap, ADMM time budget, pre-ADMM QP deadline skip, numerical/solver error), residual stage/coordinate, prefix chains, failure stage, corridor slack, exact stamp matches, gate/command-start/bridge request-response timing, per-outcome/progress bins, worker CPU/wall/TID, tracking RMSE and estimated off-path duration. Runtime does not expose QP objective improvement per iteration or a distinct QP-build-timeout cause; this run has no QP-build-timeout category, scheduler delays/migrations/context switches or per-core frequency. |
| C2: deterministic / versioned report | **Done for present fields** | `tools/export_mpc_run_report.py` and focused tests provide schema v3; absent measurements remain null/unknown. Compact JSON summary links the companion per-cycle CSV, exact lap-completion values and distributions are recorded, lap events are deduplicated, cutoff behavior and units are tested. |
| C3: identical-QP replay | **Open** | Current solver replay rebuilds QPs and cannot load frozen `A/B/d`, costs and bounds. Report schema explicitly records this as unsupported; it must not be inferred from an independent-state replay. Implementing requires a QP fixture exporter/loader and validation hashes. |
| C3: identical source-state replay | **Partial** | Existing saved-state replay with `independent` or `nominal-cold` reset modes runs a fixed input fixture, rebuilds from the provided source state and is now labeled `identical_source_state`; ensure A/B use identical fixture hashes and record the seed semantics. |
| C3: sequence replay | **Done as an explicit offline regime** | `sequence` replay is tagged `sequence_replay`; it remains an offline solver replay, not Unity closed-loop. |
| C3: live closed loop | **Done as an explicit measurement regime** | Fresh bag report is tagged `live_closed_loop`. Only live runs count for racing completion/reliability. |
| C3: single command to produce all four regime reports and replay ≥3 failure + ≥2 success bags from archived metadata | **Open** | The tools label one selected regime per replay/report and `identical_qp` has no fixture loader. The required multi-regime orchestration and five-case archived replay set were not completed. Existing one-run results do not imply this exit criterion passed. |

### PR-01 through PR-11

| PR | Status | Execution result / why remaining work is open |
|---|---|---|
| PR-01 Baseline/replay observability | **Partial** | Exact stamps, solver outcome/residual hotspots, candidate/prefix controls, command deltas, slack, failure stage, diagnostic timing and outcome/progress reports are present. Exporter tests now cover all three production `rejected_*` statuses, bridge-fault/collision cutoffs, and post-cutoff lap exclusion. Fault ring buffer, every requested low-level field, and three failure plus two success replay artifacts tied to this *new* schema remain outstanding. Existing five-lap vs failed-run comparisons are documented in the continuation progress log. |
| PR-02 Diagnostic overhead/copies | **Partial** | Exact rollout samples cached; extra diagnostic plant replays = 0; bounded asynchronous publisher live-tested. JSON field assembly remains synchronous (p95 0.284 ms). Async worker samples are “latest completed” values sampled before enqueue, lack completion-frame IDs, and are not paired per frame. There was no diagnostics-on/off timing experiment or stack/copy/memory-bandwidth audit. The measured work stayed bounded in this one run; that does not meet the handoff's comparative p99 exit criterion. |
| PR-03 Dead/legacy/build cleanup | **Partial** | `mpc_jacobian_replay_benchmark.launch.py` no longer silently injects historical weights/iteration cap/zero budget and now inherits production YAML. No broad deletion occurred: a call-graph/launch/CI/reference inventory and standalone removal PR are still required before deleting other tools/flags. This avoids deleting replay/evaluation needed to establish physics and numerical parity. |
| PR-04 Deterministic scheduling | **Open** | Fresh per-worker CPU/wall/TID reporting is available and one run met the gate. It does not solve known >20 ms tails in prior captures. No coordinator refactor, placement/affinity A/B, per-TID scheduler accounting, paired load trials, or ten-run invariant has been done. No thread/affinity policy was changed without controlled evidence. |
| PR-05 Riccati/ADMM numerical quality | **Open** | Existing warm/cold/hybrid corpus results are recorded in the progress log; they did not justify a production change. A true identical-QP comparison, trusted independent QP reference, objective/KKT parity, factorization reuse study and held-out win remain absent. No cap/rho/tolerance change was made. |
| PR-06 Nominal continuation/recovery | **Partial investigation only** | Issued-prefix warm-start alignment and prior prefix-chain evidence are recorded in the progress log. This turn did not change solver/recovery behavior. Alternative suffixes, source-plant viability anchors, trust-region bounds and held-out recovery efficacy are not established; changing them without those comparisons could worsen tracking or conceal faults. |
| PR-07 Full-physics feedforward/speed feasibility | **Open** | No source-plant-derived steering/throttle feedforward or speed-envelope table was generated. This run kept the same trajectory. Requires offline full-plant operating-point/rollout validation over both turns and requested speed bands, then held-out validation. |
| PR-08 Objective consistency/structured tuning | **Open** | No weights or objectives were changed. Need measured progress-retiming mismatch and paired held-out objective/tracking evidence first; one performance capture cannot identify a safe weight change. |
| PR-09 Trust region/line search | **Open** | No live or offline trust-region/line-search candidate promoted. Existing sequence-replay line-search evidence is small and documented in the progress log. Command-effect epoch and timing exit remain unresolved. |
| PR-10 Runtime/development split | **Partial** | Diagnostics remain opt-in and a reduced `mpc_performance` capture profile exists. The temporary replay launch now inherits production YAML. A fully separate production target/image and verified runtime/dev cleanup inventory/parity suite were not created. |
| PR-11 10 × 5-lap campaign | **Not passed** | One new uncapped five-lap run succeeded, but rollout/track error remains high (750 ms XY RMSE 0.235 m; controller cross-track max 0.520 m), the mean is 6.521 s/lap, and prior repeats include collisions and >20 ms gates. It is not justified to treat this as a release-quality repeatability campaign. Ten independent clean sessions, consistent pace and campaign-wide <20 ms remain unmet. |

### Critical investigations E1–E5 and risk matrix F

| Item | Status | Evidence / disposition |
|---|---|---|
| E1 rejected suffix poisoning | **Open** | Causal hypothesis remains: published prefix is matched to issued command, but rejected QP suffix may seed later nominal. No controlled A/B across current corpus. Keep previous evidence and do not change policy yet. |
| E2 ADMM projection vs bad physical action | **Open** | Residuals and exact candidate outcomes are reported; a same-QP projection-vs-candidate audit with alternative solver reference was not run. |
| E3 steering response at difficult turns | **Partial** | Prior practice evidence in progress log separated the negative-hairpin sign issue from a bridge sequence-gap explanation. Exact per-FixedUpdate apply ID remains unavailable. This new run is not a dedicated hairpin timing experiment. |
| E4 post-publication telemetry contention | **Partial** | New run measures callback JSON field build, queue, worker serialization/publish, and reports zero plant replay and zero replaced frames. Worker timings are unpaired latest-completion snapshots, not attributable to the corresponding diagnostic frame. It does not compare against diagnostics-off or prove no worker overlaps every next callback under all loads. |
| E5 physical nominal/feedforward | **Open** | No full-plant feedforward operating point generated; preserve existing nominal and profile. |
| F diagnostic replay removal | **Partial** | Implemented and measured as above; paired overhead exit remains. |
| F stale launch defaults | **Done** | Fixed in the temporary launch; `py_compile` passes. |
| F Riccati profiling on/off A/B | **Open** | The handoff's explicit `MPC_ENABLE_RICCATI_PROFILE` on/off comparison was not run; no claim is made that profiling overhead has been eliminated. |
| F copy/zero/stack optimization | **Open** | Not profiled; transaction/rollback memory semantics left unchanged. |
| F CPU placement/coordinator | **Open** | No policy promoted; prior one-executor comparison was negative and is retained. |
| F specialized Riccati/ADMM/seed/feedforward/trust-region | **Open** | No production changes without controlled parity/held-out evidence. |
| F mandatory second SQP iteration | **Deferred / Open** | Not enabled: fresh Jacobian/QP cost and the observed live deadline tail do not leave demonstrated timing reserve. Revisit only after the <20 ms maximum is durable under full-speed contention. |
| F prohibited shortcuts (global fast-math, reduced/bicycle plant, loosened footprint, `q=0` fallback) | **Not used; prohibited** | No such shortcut was introduced. The handoff's no-weakened-physics/constraint rules remain intact. |

### Remaining handoff sections A, G, H, I and J

| Handoff section / decision | Status | Evidence / disposition |
|---|---|---|
| A1 freeze best native-speed weights as experimental baseline | **Done** | No weight/config changes; current competition YAML hash is in run metadata. Previous five-lap successes remain experimental, not a validated racing controller. |
| A2 stop destabilizing bad recovery chains | **Open** | Prefix status/chain/slack/residual is observable, but recovery behavior has not been changed because suffix alternatives have not passed same-state held-out replay. |
| A3 strict deadline as measured invariant | **Partial** | Fresh run has 0/1,347 gate violations; earlier captures include >20 ms, so the invariant is not established. |
| A4 improve operating point/QP quality | **Open** | No feedforward, ADMM policy or nominal continuation change promoted. |
| A5 remove only proven redundant work and dev/runtime coupling | **Partial** | Diagnostic plant recomputation removed and stale launch defaults corrected; no broad call-graph or build-target cleanup. |
| A6 pursue speed only after repeated safe laps | **Not passed** | One fresh five-lap success, but other captured repeats failed and pace remains 6.169–6.892 s/lap. |
| B full source physics, 1 ms steps, 30 × 25 ms horizon, full footprint, q≥0.01, no weakened corridor | **Preserved** | This work changed diagnostics/reporting and a replay launch's defaults only. No motion/model/constraint change was made. |
| G1 change scope, invariants, enable/disable and rollback | **Partial** | This change set states its behavior-neutral scope and left physics/weights/path/constraints/recovery unchanged. Diagnostics are opt-in. No per-PR rollback artifact or clean baseline restore proof has been produced. |
| G2 pin commit/config/corpus; identical-QP/source-state comparison | **Partial** | The live run metadata pins maps/configs and post-run source manifest with the documented launch-sealing caveat. Paired replay fixture hashes and identical-QP inputs are not enforced; same-QP replay is still unavailable. |
| G3 build, all four CTests, focused regressions | **Done for this change set** | The mounted release build was used for the live run; four configured CTests pass, plus five report-exporter and two replay-regime tests. Compiler warning audit and sanitizer runs remain open. |
| G4 parity for performance refactors / exact rollout and feasibility for solver changes | **Partial** | Cached diagnostic plant outputs are covered by C++ regression and existing 409-state performance refactors had parity evidence. No new solver algorithm was promoted; there is no consolidated all-PR parity/KKT/75-ms feasibility package. |
| G5 timing distributions, deadline misses, coexistence and diagnostics overhead | **Partial** | Gate and bridge distributions are captured live. No scheduler trace, paired diagnostics-on/off test or full contention matrix exists. |
| G6 no supervisor-independent stale-command run; stop lifecycle | **Done for this live test only** | The run used the supervisor and completed five laps without collision; the wider stale-command fault lifecycle remains an open engineering item in PR-06. |
| G7 machine-readable evidence, runnable command and archived plots/bags | **Partial** | Bag, metadata, JSON and CSV are archived; no single reproducibility script/plot bundle covers all requested replay regimes/cases. |
| H1 fastest speed profile / physical feasibility | **Open** | No source-plant feasible-speed optimization was run. The current native profile is unchanged; measured lap performance is reported above. |
| H2 raise ADMM iterations? | **No change** | Kept the existing 100-iteration cap; existing comparisons did not justify raising it. New run had 103 iteration-limited samples, but 88 still passed exact full nonlinear validation. |
| H3 enable line search? | **No change** | Existing sequence replay showed limited recovery and no command-epoch guarantee; not enabled/promoted. |
| H4 change worker/executor counts? | **No change** | Kept production 4 executors / 8 Jacobian workers; prior single-executor live comparison was negative, and the new run is not a paired scheduling experiment. |
| H5 is issued first-action alignment sufficient? | **No** | Existing progress evidence confirms warm-start prefix alignment, but future suffix quality remains unresolved; E1 stays open. |
| H6 are weights dominant? | **Unknown; no sweep** | No causal attribution from this diagnostics-only run. Weights frozen. |
| I1 source/launch/CI cleanup inventory complete | **Not met** | Only one stale launch default was repaired; no full cleanup inventory exists. |
| I2 source physics/hard constraints preserved | **Met for this change set** | No plant, hard corridor, drive throttle range, horizon or fallback policy changes. Existing source-plant tests pass. |
| I3 worst-case timing under 20 ms for full campaign | **Not met** | One new run passed; older runs contain violations. |
| I4 recovery chains no longer degrade steering/slack | **Not met** | Measured but not corrected or proven. |
| I5 ten clean five-lap sessions with improved tracking and pace | **Not met** | Only one fresh successful run; tracking tails remain high. |
| I6 reproducible evidence package and external replay | **Partial** | Bag, metadata and JSON/CSV exist; no ten-session package and no second-developer reproduction yet. |
| J source navigation | **Documented** | Handoff Section J remains the authoritative source map. Files changed for this continuation are listed below. |

### Files changed and purpose

| File | Purpose of this change |
|---|---|
| `f1tenth_mpc/include/mpc_rti.h`, `src/mpc_rti.c` | Fixed-size optional diagnostic outputs captured from the exact validated source-plant rollout; existing non-diagnostic API preserved. |
| `f1tenth_mpc/src/mpc_controller_node.cpp` | No diagnostic plant replay; bounded asynchronous diagnostic publisher, worker timing and per-cycle source-sample reporting. JSON field assembly remains synchronous. |
| `f1tenth_mpc/test/test_vehicle_model_source_rollout.cpp` | Regression: cached diagnostic samples match actual production plant outputs and are absent when not requested. |
| `tools/evaluate_mpc_bag.py` | Preserve integer nanosecond stamp on decoded DriveCommand for exact report joins. |
| `tools/export_mpc_run_report.py`, `tools/test_export_mpc_run_report.py` | Exact-stamp versioned JSON/CSV, cutoffs, worker/solver summaries and regression tests; corrected units/lap dedup/float32 comparison tolerance. |
| `tools/benchmark_mpc_replay.py`, `tools/test_mpc_experiment_regimes.py` | Explicitly distinguish saved-state/sequence offline replay; record identical-QP as unsupported rather than imply equivalence. |
| `tools/record_debug_bag.sh` | Add reduced-size `mpc_performance` capture profile. |
| `sdu_apex_autodrive/launch/mpc_jacobian_replay_benchmark.launch.py` | Remove stale implicit race weights/iteration/deadline defaults; inherit production YAML unless environment override is explicit. |
| `docs/development/MPC_PERFORMANCE_MULTI_SQP_PROGRESS_20261021.md` | Append dated measured results and outstanding priorities. |
| `docs/development/MPC_OPTIMIZATION_TIMING_HANDOFF_20261021.md` | This line-item execution/defer matrix. |

## Handoff non-negotiables and acceptance

No reduced predictor, simplified plant, q=0 MPC action, weakened corridor, raised speed cap, or unchecked fallback was introduced. Full source-derived dynamics, 1 ms plant integration, 30 × 25 ms horizon, footprint constraints, current 0.01 drive throttle floor, and the existing practice trajectory/config were retained for the live run.

The handoff's final completion definition is **not met**: strict worst-case timing across the full campaign, ten collision-free five-lap sessions, improved tracking tails and improved pace are unproven. The new run has a maximum 0.520 m cross-track diagnostic error and five-lap mean 6.521 s. Keep the existing best configuration as an experimental baseline; do not call it validated reliable racing.

## Verification performed

- Fresh ROS Jazzy / `colcon` package build and all four configured CTests passed before the live run (details in the session progress log); same development container rebuilt and launched successfully.
- Live simulator/dev/recorder run: five laps, zero collisions, then clean shutdown; afterward `docker ps` and process checks showed no simulator/controller/recorder/watcher process remaining.
- Python syntax checks passed for the report tools and launch file.
- `tools/test_export_mpc_run_report.py`: 6 tests pass, including exact stamp joins, distinct solver termination categories, known not-run localization cycles, all production rejection statuses, bridge/collision cutoffs, nonzero-epoch conversion, post-cutoff lap exclusion, unit conversion, worker summaries and lap-event deduplication.
- `tools/test_mpc_experiment_regimes.py`: 2 tests pass; unknown regimes are not silently relabeled.
- The report's `score_cutoff_reason` now distinguishes collision, bridge fault, and unpublished MPC rejection; the generic `scored_through_first_fault_*` fields do not mislabel an MPC rejection as a collision/bridge fault.
- `git diff --check` passes.
- Four configured MPC CTests were rerun after the Python-only reporting changes and passed 4/4; no C/C++ edits followed the successful build/live verification recorded in the progress log.

## Independent report-audit corrections

The first sub-agent review found concrete reporting defects; these were corrected
before requesting its final audit:

- Production `rejected_input`, `rejected_solver`, and
  `rejected_nonlinear_rollout` events now all trigger the first-fault score
  cutoff when no command was published. Lap-time samples after that cutoff are
  excluded. Regression tests cover each production rejection class, bridge
  fault and collision.
- Worker maximum CPU time and maximum wall time now come from their own
  per-worker arrays/TIDs; they are not conflated. Asynchronous publisher
  counters are explicitly named `async_last_*` and documented as unpaired
  latest-completion snapshots without a diagnostic-frame ID.
- The JSON report is compact summary/progress output; per-cycle data exists
  only in the companion CSV. This avoids embedding the raw diagnostic payload
  and keeps the JSON at about 37 KB for this capture (CSV about 1.7 MB).
- The report tests were expanded to six and now check production rejection
  cutoffs, post-cutoff lap exclusion, explicit cutoff reason, and conversion
  from a nonzero bag epoch to elapsed run time. The compact JSON now carries
  the exact deduplicated lap completion values as well as their distribution.

Final sub-agent audit (2026-10-21): no remaining concrete inconsistencies were
found in the checklist, progress log, exporter, tests or regenerated report.
The reviewer confirmed six exporter tests, the exact five-lap list, known
`not_run` localization cycles, and 1,352 diagnostic rows = 1,347 published
commands + five localization waits (1,353 CSV lines including the header).
This verifies report/checklist consistency; it does not turn the still-open
engineering and racing acceptance gates above into passes.

## Next work, in evidence order

1. Finish PR-01/02 with a diagnostic-on/off paired measurement, callback-vs-worker overlap accounting, and a bounded fault-buffer design; retain field-build only if measured callback cost remains negligible.
2. Add/export frozen QP fixtures so same-QP solver comparisons are genuinely controlled; label each saved-state and sequence replay with fixture hash and memory/seed semantics.
3. Use the already archived successful and failed practice bags to compare suffix policies and quantify corridor slack/steering authority. Do not run another broad weight sweep.
4. Capture a full-run host scheduler trace on the next already-justified supervised live trial; use it to decide if coordinator/placement changes have evidence.
5. Only then choose one isolated algorithmic branch (PR-05 or PR-06 first), replay held-out captures, and perform live validation. Do not stack solver, objective and feedforward changes together.
6. After each change is independently validated, progress toward PR-11; report every failed session and do not redefine the strict gate.
