# Piecewise-source SQP precision and feasibility audit — 2026-11-04

## Decision

The correct approach is a **single nonlinear SQP with the original 26-state Unity-derived source plant, branch-correct derivatives and exact-source feasibility acceptance**, not a separate controller for each tire/geometry regime. A local QP is not globally exact across finite control corrections or nondifferentiable transitions.

**Production status:** No branch-aware multicut QP has been enabled in any live launch profile. The original source physics, full physical steering range, positive drive throttle [0.001,1], exact source certificate, and 22ms live deadline remain unchanged. No artificial command trust region, zero-throttle fallback, or substitute vehicle model has been introduced.

## Actual source-model and constraint switching surfaces

| Original physical or mathematical threshold | Code source | Regularity and treatment |
| --- | --- | --- |
| Longitudinal tire slip magnitude 0.15 and 0.25 | unity_source_plant.cpp curve_base_response | C1 but generally not C2. Use source selected-branch curvature; crossing demands exact new evaluation |
| Lateral tire slip magnitude 0.01rad and 0.1rad | same tire spline | C1 but not generally C2; especially severe second-derivative jump at 0.01rad |
| Combined-slip squared normalized force demand crosses 1 | source tire-force utilization sqrt(max(1,z)) | Continuous, generally not C1. One-sided or generalized first derivative required |
| Ray travel equals zero or suspension jounce crosses negative max droop | wheel_contact_kinematics, grounded check | Contact active-set switch, possibly nonsmooth force jump. Source rollout and stage Jacobians must use actual active set |
| Jounce reaches compression/droop limits | suspension clamp | Active saturation and derivative switch |
| Raw load/rest load crosses 0 or 3 | tire_load | Source piecewise-linear load filtering, derivative switch |
| Magnitudes of wheel-ground and wheel-surface forward speed become equal | longitudinal slip denominator | max/abs gradient switch; exact slip denominator required |
| Steering actuator reaches target, changes sign/rate, or snaps | source steering controller | Real physical slew/snap must be part of dynamics, never a command trust band |
| Absolute forward speed crosses 5m/s | unity_source_wheel_substep_count | **Discrete source numerical integration switch**: 5 internal wheel substeps per 1ms below 5m/s, 1 above. Not a globally smooth physical map |
| Zero throttle, sticky-tire speed/time thresholds | drive torque/brake torque, sticky constraints | The source switches to brake at exactly zero and changes sticky constraints at low speed. Racing domain positive drive avoids this mode |
| Physical footprint heading selects another corner or changes cos sign | mpc_rti.cpp corridor_margin_for_state | max/abs active geometry changes. Need source-active vertex Jacobians |
| Frenet geometry approaches singular denominators or heading wrap | vehicle_model.cpp tracking integration | Nonlinear coordinate singularity/representation branch; cannot use one global quadratic |
| Raceline reference interpolation changes segments | reference-at-progress | Piecewise reference/width/progress derivatives; source objective schedule may freeze speed reference but must never freeze actual wall geometry |
| Speed weighting, clearance and slip reserve hinge | nonlinear source racing cost | Piecewise C1/non-C2 objective terms; GN is PSD but not exact Lagrangian curvature |

The trace now records original contact, jounce saturation, longitudinal-slip denominator selection and speed signs, steering target capture, brake mode, substep counts, existing tire/load/combined-slip and sticky decisions. It reuses original source force evaluations; no alternative plant is calculated. Tests verify trace output neutrality and the original 4-versus-20 wheel events per 1ms across 5m/s.

Actual native-QP high-slip probes at normalized epsilon 0.00005:

- Failure 15, stage 15: combined-slip demand 0.9999907539 crosses 1.0 at ~15.0ms; lateral tire slip 0.009995702rad crosses 0.01rad at ~16.4ms, hidden in original 1ms-only tracing.
- Failure 16, stage 29: two lateral-curve branch crossings around 17–18ms.
- No additional switches detected at this epsilon in these two specific stages. Other states may differ.

Paths: live_runs/mpc_source_second_order_20261104_01/case_15/full_branch_stage15_20261104.json and case_16/full_branch_stage29_20261104.json. Reproduce with mpc_source_stage_second_order_probe and MPC_SOURCE_INNER_TRACE_STAGE=15/29.

## Corrected exact-source two-branch QP comparison

From 270 frozen real Unity capture/state/iteration combinations, original 60-control physical domain, full 30-stage source rollout, two SQP passes, six pinned cores:

| Variant | Strictly source-feasible | Newly recovered | Previously feasible lost | Median offline time | P95 offline time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original hard Phase-I QP | 103/270 | — | — | 9.47ms | 21.38ms |
| **Two truly distinct source Jacobian cuts** (experimental OFF) | **118/270** | **17** | **2** | 12.94ms | 24.61ms |
| Earlier accidentally blended-second-Jacobian variant, REJECTED | 123/270 | 21 | 1 | 12.81ms | 25.27ms |

The earlier claim of 123 feasible results **does not belong to the mathematically correct distinct-branch implementation**; the actual distinct branch relinearization was retested and gives 118. A later full-regression rerun with independent post-QP affine certification, under concurrent build/test machine load, still gave 118 source-feasible results, with ~14.3ms median and ~26.9ms P95. Timings are CPU-load dependent, not hard realtime bounds. The correct implementation still fails the key case_live_fast_17 within two SQP iterations. It loses two previously feasible cases and exceeds the 22ms time budget at P95. It remains OFF. A first/second-branch independent physical affine constraint and vehicle command-bound check was added after candidate QP solving; no QP surrogate certificate is accepted as nonlinear source feasibility.

An optional exact physical-box presolve removes affine rows proved redundant across **all** legal steering and throttle values, not only a trust region around previous controls. Median 23 two-branch wall rows were removable in a 270-case corpus; feasibility stayed 118, but median solve speed improved negligibly. The option stays OFF live.

Rejected and removed from source: single elastic Phase I, one-column source secant, trapezoidal blended Jacobian, optimistic wall-margin compensation, two-source GN wall residual QP, and one/two-branch maximin clearance QPs. These either failed on critical cases or traded feasibility for worse compute and lost previously feasible outcomes.

## Validated computational reductions

- Source stage-local AD: positive-drive passive low-speed sticky histories have identically zero influence on the racing physical transition. Eight inactive sensitivity seeds were omitted, **all 26 physical state values continue**. Source AD interval count halves from 120 to 60 per two-pass solve; 536/536 comparisons had identical published-command candidates and source feasibility.
- Inexact physical hard Phase-I QP: stop when original affine constraints, complementarity and bounded dual residual pass independent checks, then retain full nonlinear source certification. No newly infeasible outcomes in repeated 268-distinct-case comparisons; median speed saving small, severe previous 40-iteration failures much better.
- Phase-I corridor-only differentiation: the infeasible feasibility QP does not use racing cost derivatives. It now optionally differentiates exact source footprint/progress alone using original physical geometry. 270/270 selected commands unchanged; all 60x60 wall and 120x60 rate Jacobian elements match full-source scorer on three critical captures. Median saved ~0.03ms, not a breakthrough.

## Actual mathematical route to a high-precision feasible MPC

1. Preserve the complete original source state and exact original internal integration. Enumerating all wheel/substep branch combinations as separate control modes would be combinatorial. Instead evaluate active branches during each exact rollout and use correct selected-branch first derivatives and source-derived one-sided/generalized derivatives at nonsmooth surfaces. At discrete 5m/s integration switches do not pretend a smooth derivative exists.
2. Derive the **discrete full-source Lagrangian second-order information**, including costate-weighted f_xx/f_xu/f_uu, physical footprint curvature, terminal continuation and source objective. Compare each Hessian-vector product against independent frozen-dual finite differences away from and on both sides of switching surfaces. GN remains a positive-semidefinite approximate Hessian, not a globally exact SQP quadratic.
3. Replace the Phase-I zero-affine-wall-violation obsession with a consistent **nonlinear physical feasibility restoration and filter**, not arbitrary clearance reserves or a fallback throttle. Exact objective/violation acceptance must update the **same** reference/progress source state after every accepted iteration; reject tiny ineffective steps as convergence. Do not publish a merely QP-feasible but physically wall-violating plan.
4. Benchmark structured **full-state multiple-shooting SQP** as an alternative to the 60-command condensed single-shooting QP. Retain 26 states at each stage and source dynamics equality constraints; use sparse/block-banded factorization/condensing to meet realtime. Do not assume a naive 800-variable dense QP can fit 22ms.
5. Check viability at each successive horizon/terminal: strict source feasibility at the current horizon does not guarantee recovery from all possible later states; do not claim an always-feasible solution when the vehicle is already physically unrecoverable. The captured case_live_fast_17 **does** have a strict source-feasible offline witness with about 0.138m minimum wall clearance, so its live miss is a genuine current solver problem.
6. Time the entire source rollout, Jacobians, Hessian, QP, nonlinear line search, strict source and native Unity transfer, and ROS publication. Gate on <=22ms worst case with a practical 15–17ms typical goal; qualify 5+ repeated full 10-lap practice races and full competition races before any default replacement.

**Result:** original source tire switching and several additional physical/geometry switching surfaces are now observable and testable. Accurate *local* derivatives and a faster source plant have improved, but the current QP is NOT yet globally accurate across nonlinear branch changes and cannot guarantee recoverable-corner feasibility at 22ms. Production remains unchanged. The next major solver implementation is a source-derived, feasibility-aware semismooth/multiple-shooting SQP rather than another hard-coded band or branch-specific fallback.

## Exact nonlinear scoring, source/Unity fidelity and 40 Hz qualification update (2026-11-04)

### Implementation correction: never score a projected optimizer decision

`mpc_source_exact_offline_solver.cpp::exact_source_rollout` now advances all
30 original physical stages with the **unmodified float steering/throttle**
supplied by the optimizer, not the seed-building helper
`mpc_rti_build_nominal_same_epoch`. The latter projects actuator-rate
inequalities when increment-QP is active, which could silently change a
nonlinear candidate before cost/constraint evaluation. Violated source
rate inequalities are now visible in the original cost/feasibility filter.
Identical control prefixes reuse their previously validated source states,
while changed suffixes are reintegrated. Frozen references are rebuilt from
the exact resulting source progress; original source cost is authoritative.

The new `OriginalSourceTrialsNeverProjectActuatorRateViolations` C++ test
proposes invalid raw throttle (0.8 with a 0.01 rate step), proves the
normal seed builder would project it, checks the nonlinear optimizer retains
it and the negative original physical rate margin, and compares exact
source cost against independently propagated original source stages.
22/22 CTest passed, including this regression.

All three **pre-feasible witness** tests pass the strict independent
source scorer. All three **original infeasible failed-seed** recovery tests
also pass with two repeats and oracle parity. With genuine unprojected
controls, case_15 first restores at nonlinear iteration index 56 and
requires 60 offline iterations for the regression (previously 55). Final
case_15 minimum source clearance is only +0.002734m. This is a correctness
gate, NOT a real-time convergence achievement. The result remains too close
to a wall to claim reliable racing.

### Unmodified source physics vs recorded Unity

`tools/qualify_source_unity_horizon.py` now retains **all** 30-stage
source-vs-Unity predictions and separately reports windows containing a
kinematic impulse-like truth transition (>1m/s forward-speed or >0.25rad
heading change per 20–30ms interval). This is an observational flag, not
a contact oracle; it cannot establish no collision without physical data.

- `source_bridge_epoch_validation_20261104_01`: 42/42 ordinary 0.75s
  windows; RMSE of per-window *maximum* position error **0.012223m**,
  worst **0.025245m**. No impact-like windows.
- `source_exact_nlp_live_sim_20261104_03`: 17/21 windows without a
  flagged discontinuity; maximum-path-error RMSE **0.010038m**, worst
  **0.021470m**. Four impact-like windows retain a worst error of
  **2.421548m** in the **unfiltered** report.
- Recorded physical speed drops ~4.387m/s and yaw moves ~0.74rad in
  a single epoch at timestamp 1793823406824322729. The first
  collision-count=1 receipt is timestamp 1793823407343329592, about
  0.519s later. It would be incorrect to classify every earlier source
  prediction as collision-free based on collision-count receipt time.
  Direct stage-by-stage free-motion prediction is within ~1cm before
  that abrupt transition; missing wall-impact dynamics must not be
  advertised as ordinary free-flight modeling accuracy.

Audits: `build/source_unity_horizon_clean_split_20261104.json`,
`build/source_unity_horizon_source_exact_live_split_20261104.json`.
Three `tools.test_qualify_source_unity_horizon` Python regression tests pass.

### Remaining blocking failures

- Source directional derivative qualification **still fails** in
  high-slip switching regions; case_15 has a wrong-sign one-sided
  control prediction. Exact nonlinear acceptance remains mandatory.
- An eight-thread, one-iteration, at-most-eight-source-trial 40-cycle
  source-only receding simulation from case_16 stays feasible for only
  **3/40** cycles; case_15 **39/40** (worst ~26.79ms), case_17
  **38/40** (worst ~26.32ms). These are *solver timings alone*, not ROS
  end-to-end cycle certifications.
- A 12-stage independently controlled source tail at case_16 still
  needs hundreds of milliseconds on difficult entry cases (recorded
  first epoch ~533ms in the bounded experiment). It cannot be enabled
  on a 25ms loop simply by increasing nonlinear iterations.
- Recorded experimental Unity run 03 crashed and had late/no-publish
  cycles; the default controller has **not** been replaced. Historical
  source/Unity fidelity outside impacts does not fix recursive feasibility.

**Required next engineering milestone:** derive a sparse, structurally
factorized feasibility-first source multiple-shooting RTI step coupling
terminal decisions to horizon controls; prove full original cost/geometry
and active-source branch derivatives on physical finite steps; demonstrate
40 consecutive feasible receding epochs with positive clearance on every
hard corner and no overrun of the 22ms controller budget. Then perform
new Unity closed-loop races. Do not mask infeasibility by projecting
decisions during nonlinear cost evaluation.

### Live 22 ms source-constrained MPC qualification (4 Nov, continuation)

**Preserved mathematical authority:** exact original source stages, original
nonlinear racing cost, source footprint/command-rate/command-box inequalities,
independent accepted-plan validation; the QP only proposes a direction. Normal
feasible cost-optimization step ordering remains unchanged. For an infeasible
source horizon, try full QP steps before fractional steps, rather than
prematurely searching only the 1/16 and 1/32 directions. The latter produced
~4 mm original physical wall penetration in the live captured predecessor.

- The captured *real Unity physical source state*, including all 22 hidden
  plant-memory values, delayed actuator commands, 30 seed commands, and
  track geometry, is in
  `build/source_qp_unity_live_failure_20261104_05/fixture`.
  Under the exact physical source optimization oracle, two nonlinear SQP
  iterations with larger feasibility-first trial steps produced **+10.568 mm**
  minimum original corner slack with the configured **10 mm extra reserve**;
  the previous small-step trial order gave **-4.032 mm** after two iterations.
  This matches the original live wall margin and is a targeted physical
  feasibility regression, NOT a general full-track guarantee.
- `SDU_APEX_MPC_LTO=true` allows host-native link-time optimization without
  changing mathematical acceptance. Six source workers use explicit
  `OMP_PLACES={0},{2},{4},{6},{8},{10}` and `OMP_PROC_BIND=TRUE`.
  Use `SDU_APEX_MPC_RUNTIME_AFFINITY_CPUS=0-11` for ROS and
  `SDU_APEX_SIM_RUNTIME_AFFINITY_CPUS=12-15` for the Unity executable.
  Docker `--cpuset-cpus` was silently discarded on this host, causing
  catastrophic live PhysX/MPC CPU contention (152 ms–846 ms solve spikes).
  This explicit Unity process affinity was directly checked within its
  Docker image using `taskset`/`/proc/self/status`.
- Successful **actual Unity practice** run:
  `live_runs/source_qp_affinity_isolated_practice_20261104_07`.
  One warmup plus **five scored laps**, six official counter transitions;
  no collisions and no unpublished solver rejections. Scored laps:
  4.9558, 4.9508, 4.9508, 4.9358, 4.9518 s;
  scored total 24.7450 s, mean 4.9490 s, population SD 0.00685 s.
  1,251/1,251 accepted-feasible source controls in health counter;
  report captured 1,238 diagnostic publishes (excluding startup intervals),
  max callback-to-gate 15.426 ms, P99 14.270 ms, and no callback period
  above 25 ms. Maximum **solver** time 15.278 ms in the periodic health log.
  Note that the post-simulator-shutdown health record includes one transport
  teardown fault after the watcher had completed; there was no collision
  or MPC rejection during scoring.
- Limitations: one completed five-scored-lap Unity session does not establish
  reproducibility across successive races, target machines, or competition
  corners. Full ten-scored-lap testing and multiple independent sessions
  remain necessary. The normal, established production profiles remain
  unchanged; all runtime settings above are opt-in simulator experiments.

### Ten-scored-lap race and submicron extra-margin correction

- An **11-official-lap** attempt using the same fast exact-source controller
  `live_runs/source_qp_affinity_isolated_10scored_20261104_08` exposed
  a spurious rejection at progress 34.373 m: original-source minimum
  corner margin **+0.00999976508 m**, requested **+0.01000000000 m**;
  squared extra-reserve violation only **2.75e-14**. No physics-wall
  penetration and no solve deadline miss (17.22 ms maximum). It correctly
  reproduced in the captured same-epoch 26-physical-state fixture under
  `build/source_qp_unity_live_failure_20261104_08/fixture`.
- Fixed exact-source feasibility certification: *always* require
  `original_source_corner_margin >= 0`; when the simulator explicitly
  requests an **additional** clearance, allow only **0.5 micrometre**
  tolerance to accommodate source float geometry and reference rounding.
  The original source-valued margin, exact cost and unmodified physical
  nonlinear constraint violation remain reported. The independent full
  physical-source candidate plan validator still executes before commands
  are published. Verified the original failed state via the actual
  `mpc_rti_solve_cycle` entry: accepted 30-stage source-valid plan with
  +32.76 mm actual clearance, 9.29 ms full cycle, original-source exact
  objective, matched memory and command epoch.
- **Complete 10-scored-lap Unity practice race**, one warm-up + ten scored,
  `live_runs/source_qp_numerical_clearance_10scored_20261104_09`:
  `SCREEN_COMPLETE: lap limit`; 11 official lap transitions, zero collisions,
  zero unpublished MPC rejections, and 2230 independently full-plan checked
  and published source-exact feasible controls (30 startup localization waits
  excluded); 2244/2244 accepted feasible in final controller health counter.
  The startup/post-shutdown timing distinction matters: final health
  reports one transport teardown control fault **after race completion**.
  Ten scored official lap times: 4.9378, 4.9828, 4.9708, 5.0208, 4.9918,
  4.9558, 4.9358, 5.0038, 4.9808, 4.9608 seconds. Total **49.7410 s**;
  mean 4.9741 s, population SD 0.02608 s. Publication gate median
  **7.954 ms**, P99 **14.423 ms**, worst **20.130 ms**; no 22 ms gate
  budget violations, no callback interval above 25 ms. Worst controller
  solve in health log **19.874 ms**. Real Unity race is no longer merely
  a source-only offline result; further independent repeat races and the
  competition map remain outstanding.
- Runtime process isolation is essential on this host. Docker cpuset was
  discarded by the kernel. Explicit `taskset` applied inside Unity on CPUs
  12-15 while ROS/source MPC uses 0-11 with 6 explicit P-core OMP places.
  The prior `live_runs/source_qp_feasibility_order_practice_20261104_06`
  collision followed 152ms/846ms no-publish stalls when PhysX 16 workers
  shared those MPC cores. The fixed affinity is *not* a replacement for
  profiling on other machines; keep it an opt-in simulator runtime option.
