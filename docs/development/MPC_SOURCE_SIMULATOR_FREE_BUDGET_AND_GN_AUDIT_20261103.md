# Source-SQP no-simulator timing and cost-fidelity qualification — 2026-11-03

## Decision

**Status: source-exact MPC is much closer to live real-time, but it is not yet fully qualified.** Stage-local 26-state/2-control source transition Jacobians reduced the *measured cold-start* source derivative median from ~16 ms to ~4.5 ms with six P cores, without simplifying the physics or the 30-stage/60-control horizon. The full optimizer now reaches and solves a QP in the no-simulator cold-start replay. However, shared-core CPU contention can still consume the strict 18 ms solver gate, and one difficult saved high-slip corner remains unaccepted even offline.

Do not claim that source-only numerical fidelity implies Unity/PhysX agreement. Do not introduce fallback controllers, artificial actuator trust regions, automatic brake-on-solver-error, or deadline extension to conceal unresolved convergence problems.

## What was added

1. **Full-source Gauss–Newton Hessian (experimental, opt-in):** full 60x60 PSD J'WJ for differentiable frozen racing-cost residuals; steering/throttle and cross-stage terms, smooth steering-variation curvature, original exact first derivative, 60 real command variables, 26 physical source channels and exact nonlinear acceptance. Neither source dynamics nor physical footprint/rate inequalities are relaxed. Enabled with `native_source_sqp_gauss_newton_hessian=true`.
2. **Stage-local source transitions (experimental, opt-in):** 30 independent stage Jacobians each with the exact 26-state source plant and two command inputs. Evaluate local 26+2 derivatives on separate CPU cores; causally multiply to obtain all 60-control source gradients and wall/rate Jacobians. The original full direct-source sensitivities remain as a reference until **live** correctness and deadline qualification. Enabled with `native_source_sqp_stage_local_jacobians=true` and `native_source_sqp_full_source_forward=true`. Existing global source interval physics is unchanged.
3. **Exact scorer work reuse:** when the objective schedule is frozen, a source-tangent candidate no longer separately interpolates progress-speed targets already interpolated by the full scorer. Verified identical command/status/objective/clearance on selected cold/moving/high-slip cases.
4. **Traceable source-SQP diagnostics:** even when the derivative builder runs out of deadline, record its elapsed time, summed source worker time, stage interval count and tangent-scoring count. An actual cost-optimizing first SQP iteration now also records **the quadratic QP predicted racing-cost change, exact source-model actual change, and their signed difference**. Infeasible Phase-I steps are explicitly marked *not comparable* because their QP optimizes wall violation, not racing cost.
5. **One-source-controller live-mode rejection path:** native source mode refuses to fall through into the older prior-plan, short-prefix, reacquisition or viability-recovery publishers. If no complete source-certified MPC plan exists, it emits no alternative command; this is a visible solver failure, not yet successful driving.
6. **Simulator-free native timing harness:** `tools/mpc_source_live_fixture.cpp`, `tools/prepare_mpc_source_budget_fixtures.py`, `tools/benchmark_mpc_source_budget.py`. They reconstruct complete cold source states from recorded Unity ROS diagnostics, reuse exact practice raceline and real practice weights, then time the true native `mpc_rti_solve_cycle` path, with optional hard deadlines, CPU affinity, synthetic contention and JSON p50/p95/p99/worst distributions.
7. **Isolated pilot config:** `f1tenth_mpc/config/mpc_source_gn_live_pilot.yaml` enables full-source GN + stage-local source Jacobians, retains the *existing 20 ms end-to-end ROS gate* and does not enable alternative controller fallbacks. The previous Unity pilot ran *before* this stage-local optimization and did not complete startup.

## What is timed and what is not

The C++ native controller-epoch benchmark includes normal source nominal construction, every source tangent, Gauss–Newton Hessian, QP solve, nonlinear source acceptance, state/optimizer memory update and the solver's own certificate. It reports elapsed wall time **inside the real production `mpc_rti_solve_cycle`**. The standalone verification tool also checks its selected command against the independent full physical source validator, *after* the reported solver timer; that extra verifier is for test evidence and is not an additional ROS publication requirement. Process startup/file parsing are excluded by the C++ timer. The test uses actual captured source hidden state and command history, but not the Unity/PhysX process, live ROS executors, odometry synchronization, queue contention, command transport or actuator discrepancy.

**The relevant deadline is 20 ms end-to-end in this controller, within the 25 ms 40 Hz period.** A practical source-solver goal is <=17–18 ms *including scheduling tails*, to reserve time for the rest of the ROS callback.

The baseline competition source context supplies the solver/model ABI and a base QP capture, while the recorded practice parameters override the racing weights and constraints and the actual practice source raceline replaces the competition path. This reconstruction is much more representative of the actual Unity cold start than a generic straight track, but it is **not a byte-perfect reproduction of the complete live callback configuration**. The objective schedule is freshly built through the normal native controller cycle, not copied from the competition context.

## Cold-start C++ replay: 40 repetitions each, same six separate P cores

CPU mask `0,2,4,6,8,10` on i5-13400F, six OpenMP source workers, Release -O3, GN Hessian, exact 30-stage source, no Unity running. Internal C++ solver wall clock; no artificial deadline in this timing experiment.

| Captured case | Derivatives | Median solve | P95 | Observed max | Median derivative | >=20 ms | Source certified |
|---|---|---:|---:|---:|---:|---:|---:|
| Unity startup, early | Direct full source | 22.169 ms | 25.381 ms | 39.646 ms | 16.206 ms | 39/40 | 40/40 |
| Unity startup, early | **Stage-local exact source** | **10.511 ms** | **15.818 ms** | **16.253 ms** | **4.458 ms** | **0/40** | **40/40** |
| Unity startup, stable | Direct full source | 21.954 ms | 23.370 ms | 30.451 ms | 16.080 ms | 40/40 | 40/40 |
| Unity startup, stable | **Stage-local exact source** | **10.402 ms** | **15.075 ms** | **17.118 ms** | **4.478 ms** | **0/40** | **40/40** |

A separate **18 ms absolute C++ solver deadline**, same six cores and no synthetic contention, succeeded for **35/35** in each stationary case with stage-local derivatives. Direct source derivatives yielded **0/35** certified plans in each case under that strict gate.

This is the first measured demonstration of adequate startup compute on the actual high-fidelity C++ QP **without** Unity contention, but it is not live deadline certification.

## Scheduling contention matters

35 repetitions per cold case, six P cores available to MPC source workers:

| Test environment | Cold early stage-local median / P95 | Cold stable stage-local median / P95 | Strict 18 ms source-certified |
|---|---|---|---|
| No added synthetic CPU load | ~10.5 / ~15.8 ms (40 trials) | ~10.4 / ~15.1 ms (40 trials) | 35/35 early; 35/35 stable |
| Three competing synthetic workers pinned onto **same six P cores** | **17.65 / 19.66 ms** | **16.97 / 19.70 ms** | **26/35 early; 27/35 stable** |
| Three competing synthetic workers on E-cores **12,13,14**, leaving P cores for MPC | **9.93 / 10.91 ms** | **9.93 / 10.81 ms** | Not separately tested |

Even the non-deadline shared-core stress trials certified all 35 horizons per case, but their latency is too close to the 20 ms full callback deadline once ROS overhead is included. This strongly supports careful source-worker CPU placement (six P cores), avoiding uncontrolled competition with Unity, ROS and other work. Synthetic load is not interchangeable with native Unity contention and says nothing about realistic tire contact events.

**A one-core limit, or OpenMP six-worker oversubscription, is not an acceptable production timing configuration.** Source workers need enough genuinely distinct cores. Do not let the ROS control-thread affinity restrict them to one CPU; a separate worker placement policy must be validated live.

## Moving-state and high-slip replay

25 repetitions with six P cores, GN Hessian, one SQP iteration:

| Recorded condition | Direct source median | Stage-local median | Stage-local P95 | Source-certified plans |
|---|---:|---:|---:|---:|
| Feasible moving competition state | 12.77 ms | **10.09 ms** | 11.24 ms | **25/25** |
| Difficult high-slip / corridor-infeasible initial state | 20.55 ms | **16.64 ms** | 17.16 ms | **0/25** |

The high-slip case completes QP computation but does not produce a certified full horizon. It is a **nonlinear convergence/feasibility problem**, not something solved by a faster linear algebra implementation. Do not publish that candidate and do not count it as a live MPC success.

60 consecutive full source-model epochs were also run using GN + stage-local derivatives and a 25 ms absolute C++ deadline, with persisted warm memory and exact source one-step state advancement: **60/60 source-valid, median 8.86 ms, observed maximum 20.61 ms**, no alternative controller. The identical *18 ms* strict gate **failed at epoch 0** on one run, confirming that tail latency is still too large to declare deterministic real-time qualification.

## Fidelity and exact cost agreement

- Six recorded difficult competition states: stage-local vs original direct-source derivatives agree with maximum **relative gradient error 3.77e-5** and maximum **absolute footprint/rate Jacobian discrepancy 1.2e-4**. Same six QP statuses and acceptance outcomes; source cost differs by at most about 1.7e-6 on the six tests. The local transition multiplication introduces ordinary finite-precision differences; it does **not** replace the source model.
- Full 60x60 GN Hessian has an explicit positive-semidefiniteness and cross-control/cross-stage C++ regression test. Original direct-vs-local gradients/constraints now have another dedicated straight-track parity test.
- In an extended eleven-state source-controller objective-prediction audit, the median **absolute first-step QP predicted-minus-realized cost change** was **18.01** for the previous diagonal Hessian (10 valid comparisons) and **5.31** with full-source GN (11 valid comparisons). The sets are *not* perfectly matched, so interpret directionally.
- The former underpredicts physically achievable reductions in some cases and substantially overpredicts others. **GN is not universally accurate.** At source epoch 037, GN predicted a ~6.94 cost reduction but the exact source cost improved by only ~0.01; this model-agreement outlier must be investigated before racing qualification. At one cold start, GN predicted -10,873.88 vs actual -12,594.87 (signed error -1,720.99), compared with old predicted -10,588.70 vs actual -7,840.98 (signed error +2,747.71). Absolute cost scales strongly by state.
- Finite-difference agreement with source forward-mode gradients still degrades at high-slip nonsmooth regimes. Example worst normalized mismatch at one high-slip capture: ~0.66 using h=0.0005, ~1.70 using h=0.002. Very small h=0.00005 is **worse**, consistent with limited float precision and/or branch crossing; this alone does not prove the analytic tangent is wrong. Further source branch stability and independent directional derivatives are needed.

No high-slip model discrepancy was hidden by tuning a trust radius or switching controllers.

## Reproduce without simulator

Build the native test tools once (the replay targets are deliberately `EXCLUDE_FROM_ALL`):

```bash
cmake -S f1tenth_mpc -B build/f1tenth_mpc \
  -DSDU_APEX_MPC_PARALLEL_SOURCE_SENSITIVITIES=ON
cmake --build build/f1tenth_mpc \
  --target mpc_source_live_fixture mpc_native_source_sqp_controller_epoch \
           mpc_native_source_sqp_replay mpc_offline_advance_source_epoch -j6
(cd build/f1tenth_mpc && ctest --output-on-failure)
```

For the recorded Unity practice startup, first generate the JSONL diagnostics from its already recorded bag if they are not available:

```bash
python3 tools/export_mpc_run_report.py \
  live_runs/mpc_source_gn_pilot_20261103_02/run \
  --experiment-regime live_closed_loop \
  --output-json /tmp/mpc_live02_report.json \
  --output-csv /tmp/mpc_live02_report.csv
```

Create the reconstructed physical source fixtures once; substitute any compatible recorded same-build source case for `BASE`:

```bash
BASE=live_runs/comp_gradient_nearwall_capture_20261102_01/qp_captures/epoch_1793692853295529252_case_6
python3 tools/prepare_mpc_source_budget_fixtures.py \
  --diagnostics /tmp/mpc_live02_report_diagnostics.jsonl \
  --parameters live_runs/mpc_source_gn_pilot_20261103_02/mpc_parameters.yaml \
  --trajectory f1tenth_planning/trajectories/autodrive_practice_20260924_b/source_command_4p800.csv \
  --base-case "$BASE" \
  --out /tmp/mpc_source_budget_fixtures
```

Benchmark **compute needed**, with no artificial solver cutoff:

```bash
python3 tools/benchmark_mpc_source_budget.py \
  --cases cold_early=/tmp/mpc_source_budget_fixtures/cold_early \
  --cases cold_stable=/tmp/mpc_source_budget_fixtures/cold_stable \
  --cpus 0,2,4,6,8,10 --workers 6 \
  --repeats 100 --out /tmp/mpc_source_budget_baseline
```

Benchmark **actual acceptance under an 18 ms absolute solver deadline**, and optionally 3 competing CPU workers on those same cores:

```bash
python3 tools/benchmark_mpc_source_budget.py \
  --cases cold_early=/tmp/mpc_source_budget_fixtures/cold_early \
  --cases cold_stable=/tmp/mpc_source_budget_fixtures/cold_stable \
  --cpus 0,2,4,6,8,10 --workers 6 \
  --repeats 100 --deadline-us 18000 \
  --load-workers 3 --out /tmp/mpc_source_budget_stress
```

Read `summary.json` for per-case median/P95/P99/worst, derivative/nominal/QP/source integration breakdown, 17 and 20 ms misses and **source-certified acceptance count**. `samples.json` contains each native C++ result including actual and QP-predicted cost changes.

The optional pilot overlay is `f1tenth_mpc/config/mpc_source_gn_live_pilot.yaml`. It is **not** enabled by default. This simulator-free benchmark does not itself establish Unity qualification. The separate, concurrently updated `MPC_SOURCE_GN_STAGE_LOCAL_UNITY_AUDIT_20261103.md` documents subsequent Unity stage-local pilots, including a five-lap completion in run 08, seven deadline deferrals, and loss of full-horizon feasibility during the attempted sixth lap. The standalone benchmark results here should be interpreted alongside, not as a replacement for, those live results.

## Remaining exit criteria

1. Replay **many captured cold and high-slip cases** (including branch transitions) and investigate outliers in predicted-vs-exact actual merit; make the source QP direction reliably useful without artificial trust regions.
2. Establish a realistic CPU placement plan: six P-core sensitivity workers, no excessive OpenMP/Eigen teams, clear separation from ROS executor and Unity CPU-heavy tasks. Profile full callback and its p99/worst under actual Unity load.
3. Achieve repeated real **<=20 ms end-to-end** publication including synchronization and command validation with zero misses; simulator collision remains independent, not a fallback logic event.
4. Qualify one normal source-SQP controller on cold starts, live corner transitions and complete practice/competition races. Zero source-rejected plan publications and zero legacy fallback controls.
5. Only after repeated successful live runs, remove the old direct-derivative implementation and unnecessary parallel experimental variants; keep maintainable one-owner production C++.

**No live qualification is claimed in this report.**
