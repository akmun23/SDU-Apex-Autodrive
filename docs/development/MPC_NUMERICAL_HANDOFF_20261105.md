# Source MPC numerical handoff — 5 November 2026

## Current result

The worktree contains implemented numerical/publication fixes, a reproducible
failure benchmark and **unqualified offline experiments**. The nine-phase task
is unfinished. No Unity race has been run or qualified with these changes.
Runtime qualification and speed tuning remain secondary to mathematical
correctness, source cost fidelity and feasible output.

All **22 recorded competition states have independently checked source-feasible
recovery witnesses**. The live optimizer still fails some late seeds, including
record 775. No tested method has met the complete original-NLP KKT exit condition.
Existence of a witness is not evidence that the live solver can find it.

Throttle remains **[0.001, 1]**. Exact-zero throttle is not controller-admissible.
Keep original collision constraints hard, independently validate publication
against source physics and the real deadline, and retain separate optimality
reporting. Do not add a controller fallback or fixed alpha grid.

The repository was already dirty before this work. Pre-existing experimental
terminal-speed code, viability profiles and the two-track report are retained.
Those profiles are not validated competition configurations.

## Phase status

| Phase | Status |
|---|---|
| 1: failure benchmark | Infrastructure implemented/tested. Some exact historical data were never logged and cannot be recovered retroactively. |
| 2: feasibility/convergence | Active. Important QP/KKT fixes implemented; late-state recovery and final stationarity unresolved. |
| 3: recursive feasibility | Incomplete. Existing speed-only terminal condition remains experimental. |
| 4: publication continuity | Retention, statuses, rollback, epoch and observability fixes implemented; no new race qualification. |
| 5: timing | Two worker-related issues fixed; the new solver has no qualified timing distribution. |
| 6: source/Unity fidelity | Recorded competition data analyzed; some horizon errors exceed the nominal safety margin. |
| 7: race qualification | Not started for this worktree. |
| 8: speed recovery | Not started. Practice and competition objectives remain separate. |
| 9: cleanup | Deferred until the formulation is stable. Research options still need evaluation/removal. |

## Implemented changes

### Benchmark, exact source scoring and capture

- Checked-in bundle:
  `f1tenth_mpc/test/data/competition_corner_20261105/`.
  Manifest/hash checks, recorded sequence, raceline, physical model, original
  oracle baseline and investigation are retained, mostly as deterministic gzip.
- `tools/competition_failure_benchmark.py` extracts/replays the episode and
  independently scores returned controls. Frozen model/objective PODs use
  explicit same-build ABI guards. The baseline has not been replaced with
  improved optimizer outputs.
- There are 22 epochs in receipt order, progress 142.680016–144.154231 m.
  Last publication: record 756 at 142.841566887 m. First infeasible: 757 at
  142.992144702 m. Reported terminal failure: 769 at 143.847507964 m.
  Last recorded solve: 775 at 144.154231245 m.
- Historical streams used 12 digits and omitted live SQP iteration primals,
  some initialization flags, frozen schedules and discarded final controls.
  Reconstructed iterations are labeled as reconstruction. Initial cost parity
  is checked across all 22 records; the actual published final primals can
  also be independently rescored. Initial and final primals are never mixed
  in a parity assertion.
- New opt-in `capture_source_benchmark` records 17-digit states, state flags,
  frozen problem data and primary/SOC iteration primals. Rejected cycles also
  retain their final controls. Capture-on/off numerical equivalence is tested.
- The replay regression independently checks **25 stored feasible witnesses**;
  no recorded state's source reachability remains unresolved. Earlier failed
  searches remain preserved as failed searches, not infeasibility proofs.

Record 775 was resolved by independent SLSQP, then the original native source
evaluator verified clearance **0.00008550286293029785 m** with admissible
positive throttle. Local files are
`artifacts/phase2_compare_775/slsqp.controls` and
`artifacts/phase2_baseline/witness775_verified.json`. Search required 300
iterations, 2007 oracle calls and about 21.7 s; KKT was not converged. This
tiny-margin witness is neither a Unity nor a real-time qualification.

Original chronology includes 19 consecutive no-publication epochs. Twelve
rate-limited older-epoch warnings were observed; that is not a complete count
of skipped callbacks. See
`MPC_COMPETITION_FAILURE_BENCHMARK_20261105.md` for the detailed original evidence.

### SQP and convex KKT solve

- Added `mpc_source_convex_qp.{hpp,cpp}`: infeasible-start primal-dual
  predictor/corrector, eliminating slack/elastic variables into the control
  dimension. Original source body/rate/box inequalities are hard. Preferred
  additional clearance uses a finite quadratic elastic epigraph.
- Preferred-clearance multipliers no longer accumulate as an AL condition
  that inadvertently makes the extra 3 cm hard.
- Original-NLP stationarity is reported in physical command coordinates.
  Primal feasibility, dual feasibility and complementarity are checked
  separately. A feasible rollout or small correction does not imply optimality.
- Added omitted source wheel peak-slip GN features and the resulting
  steering/throttle coupling. Source A/B and condensed physical derivatives
  come from the same exact 29-state source transitions.
- Racing-cost predictions exclude numerical stabilization and calibrated
  constraint-merit secants.
- Added damped Lagrangian BFGS with frozen multipliers across accepted source
  steps. Known GN curvature refreshes after the original source primal becomes
  feasible; the earlier metric is retained during infeasible recovery. This
  preserves useful early recovery but is not a complete convergence fix.
- Added one bounded Maratos SOC from the exact nonlinear constraint defect,
  using the same objective/Jacobian. Its primal is independently integrated
  and recorded.
- Rejected-step retries reuse the current exact source linearization. If a
  terminal reference kink changes row count, dual/Hessian memory resets rather
  than assigning multipliers to different rows or rejecting a valid primal.
- Removed fixed default acceptance floors of `1e-5` cost and `1e-10` squared
  violation. They blocked useful small improvements near a solution. Default
  acceptance now requires strict reduction above a double-roundoff threshold.
- Inner QP iterations check the absolute deadline. Regularization ceiling is
  1e12 rather than 1e5 because measured scaled source curvature can exceed the
  old ceiling. A higher ceiling alone does not fix infeasible linearizations.

### Publication, synchronization and timing fixes

- A later derivative/QP failure finalizes the retained source primal rather
  than discarding an earlier feasible iterate. Initial invalid-model failure
  still fails the solve. Publication independently checks physical feasibility
  and the callback deadline, while optimality remains separate.
- Solver diagnostics distinguish `deadline_exceeded`, `primal_infeasible`,
  `nonlinear_step_rejected`, `KKT_not_converged`, `invalid_model` and
  `feasible_solution_available`; availability is also a separate boolean.
  Admission stopping before the deadline is not labeled an actual overrun.
- Fixed a rollback bug: every callback cleared
  `last_published_command_stamp_ns_` before
  `SolverMemoryTransaction::begin`, disabling saving the issued warm state.
  The stamp now persists; per-cycle publication remains a separate flag.
  Native checks exercise the rollback transaction.
- Epoch shifting quantizes a signed forecast difference. Sub-ms backward
  revisions of the same future Bridge request map to stage 0; a true older
  stage is rejected. Regression covers the observed -595814 ns revision and
  a full older 25 ms stage. Current source state is still revalidated.
- Added consecutive/max no-publication counters, last-issued header age,
  older-epoch rejection/revision counts, and unchanged host Bridge-payload age.
  Payload age survives history overflow and resets on value/clock changes.
  It is explicitly not a Unity FixedUpdate application timestamp.
- Final publication failures preserve actual deadline status; an invalid
  command is distinguished from an expired deadline.
- Disabled Eigen's implicit OpenMP teams for small KKT products, which caused
  experimental tails around 170 ms. Explicit source-stage AD workers remain.
  That path now honors configured `model.jacobian_threads`, previously ignored.
  These changes do not establish the 22 ms timing exit condition.

## Offline numerical experiments, not live configuration

The following options default to false and have no enabling ROS parameter:

| Offline executable environment flag | Purpose |
|---|---|
| `MPC_SOURCE_NLP_SLIP_EPIGRAPH=1` | Shared wheel-peak epigraph; constraint generation checks all original positive substep cuts. |
| `MPC_SOURCE_NLP_ELASTIC_GLOBALIZATION=1` | Exact-L1 regularization of an infeasible local linearization. Original physical feasibility is still required for publication. |
| `MPC_SOURCE_NLP_MERIT_GLOBALIZATION=1` | Safeguarded interpolated Armijo operation after rejected full/SOC steps; no fixed alpha grid. |
| `MPC_SOURCE_NLP_FULL_HESSIAN=1` | FD of exact source Lagrangian gradients with frozen duals, symmetry and spectral regularization. |
| `MPC_SOURCE_NLP_ABSOLUTE_CURVATURE=1` | Absolute-eigenvalue variant of the offline Hessian experiment. |

The FD-Hessian path currently applies only with the slip epigraph flag **off**.
Research iteration limit is 2000; live configuration remains capped at 8 and
the competition profile requests 3. Offline traces retain safeguarded trials,
step lengths, validity and acceptance. The live C trace supports primary/SOC
paths; it must be extended before enabling safeguarded globalization live.

An additional research bug was fixed at handoff: Armijo equality at floating
roundoff could count zero-progress accepted iterations. The main code now
requires representable command changes and strict merit reduction. Some older
experiment logs and copied experimental binaries predate this correction.

Representative results from their named builds, **not** a current qualification:

| Artifact under `artifacts/phase2_baseline/` | Result |
|---|---|
| `hybrid757.json` | First feasible iterate at index 2. At 100 steps: cost 91.7605, margin +0.0298749 m, stationarity 27.53; KKT false. |
| `hybrid769.json`, `hybrid775.json` | Default method still fails these seeds. |
| `armijo769.json` | Source-feasible recovery, cost 8197.68, margin +0.001493 m; stationarity about 204691 after 100 steps. Globalization was frequent. |
| `fullH769.json` | At 30 steps: source-feasible, cost 10484.77, margin +0.030736 m, stationarity 36208.85. |
| `fullH_abs769.json` | At 100 steps: source-feasible, cost 8228.89, margin +0.0257914 m, stationarity 2415.33. |
| `fullH_abs775.json` | Still infeasible, margin about -0.209 m. |

The 16/22-feasible report in `artifacts/phase2_current_20/report.json` predates
later BFGS changes; it is not the current solver's result. These experiments
show useful curvature effects but no complete phase-2 exit.

Independent tools retained:

- `tools/compare_source_nlp_methods.py`: native SQP versus SciPy SLSQP, using
  the same exact source oracle and physical bounds. This found the 775 witness.
- `tools/compare_source_ipopt.py`: CasADi/IPOPT interface to the native oracle,
  not a symbolic surrogate plant. `--oracle` selects a research build. IPOPT
  also failed late cold seeds. A solver's infeasibility status is not a physical
  infeasibility proof; independent scoring even found a feasible final primal
  in one run that reported infeasibility.
- `tools/mpc_source_curvature_audit.cpp`: frozen-multiplier source Lagrangian
  secants in steering/throttle/coupled/SQP directions. Crossing source branches
  is not a smooth-Hessian certificate.

Primary references:
[primal-dual KKT methods](https://web.stanford.edu/~boyd/cvxbook/),
[stabilized SQP](https://pages.cs.wisc.edu/~solodov/izmsol09sSQP.pdf),
[CasADi callbacks](https://web.casadi.org/docs/),
[IPOPT curvature options](https://coin-or.github.io/Ipopt/SPECIALS).

## New accuracy finding: precision through wrappers

Visible motion/tracking states, commands, interval displacement and projection
Jacobians pass through float wrappers, although native source physics and the
22 hidden states use doubles. Tiny-step source secants become noisy near a
low-cost iterate. Genuine tire/contact/steering switches remain a separate issue.

At `refresh757.controls`, original cost is 55.7216140; a double-wrapper
experiment gives 55.7216002 using the original frozen samples/weights. For
physical throttle coordinate 13, AD is about -49.11. Original float-wrapper
secants at a 2.5e-6 command change were -9.64 and -53.18. Double wrappers give
about -49.10/-49.12 at 2.5e-7 and -49.109/-49.110 at 2.5e-8.

**Precision promotion is not merged into runtime.** Local isolated copies are:

- `artifacts/phase2_precision_source/f1tenth_mpc/` and `tools/`
- `artifacts/phase2_precision_build/`
- `artifacts/phase2_precision_cases/`

The experiment broadly replaces float declarations with doubles and float math
calls with their double equivalents in a copy of the package/tools. It is not
a minimal production patch. POD layouts change. New fixtures import the
original frozen references/speed weights through portable JSON; original
terminal/successor objectives were asserted disabled. Input visible states and
nominal commands were first converted to their original float values, and
numeric profile values were float-rounded before promotion.

The copied fixture builder explicitly reports original versus promoted cost;
initial differences were about -0.04436 at 757 and -0.01094 at 769. These exceed
strict historical parity and must not be concealed by relaxing the old
baseline. Original bundle and original baseline remain intact. The native
oracle now accepts `{"metadata":true}` to export frozen references/weights.

Evidence under `artifacts/phase2_baseline/`:

- `gradient757_near_stationary.json`: original AD/one-sided secants.
- `constraint_gradient757.json`, `constraint_gradient775.json`: body-Jacobian
  comparisons. At scaled step 1e-4, relative Frobenius errors are approximately
  0.00317/0.00451; larger 1e-3 steps cross more branches.
- `objective757.json`, `objective769.json`, `objective775.json`: portable frozen
  schedule exports used by the promoted fixture builder.

Results under `artifacts/phase2_precision_cases/` do not establish convergence.
The later `fullH_small_steps*.json` experiment uses Hessian FD step 1e-5 and
removes fixed acceptance floors. At 100 steps, 757 stayed feasible with cost
55.72034 and stationarity 6.62; 769 remained infeasible by 0.02040 m and 775
by 0.15866 m. Those binaries still have the older Armijo equality behavior.

Ignored artifacts are local workspace evidence, not part of a fresh Git checkout.
To recreate the experiment, copy the package/tools, promote float types/math
calls in the copy, build its three native tools, export each original schedule
using the original oracle, and import those numeric references into the copied
fixture builder instead of its incompatible raw objective POD. Explicitly log
the cost difference rather than altering historical parity assertions. A future
production patch needs minimal precision changes and versioned legacy replay.

## Source-to-Unity recorded-data check

Reports: `artifacts/phase6_competition_20261105/`.

- `one_step.json`/`.csv`: 780 exact-epoch comparisons, zero failed source
  steps and zero recorded collisions. Signed speed/lateral-velocity/yaw-rate
  errors are retained. This is the old failed race, not a new race.
- `horizon.json`: 30-stage windows initialized once then propagated with the
  unmodified native source. Of 144 Bridge-command windows, 9 exceed 5 cm maximum
  path error. P95 maximum error is 0.05436 m; maximum is 0.08295 m. Queued-command
  association produces nearly identical results.
- `command_epochs.json`: 780/782 exact odometry/request stamp matches, two
  missing early associations, zero matched command-value mismatches. Its
  `certified=false` reflects missing epochs. This is host-request evidence,
  not proof of Unity apply ticks.

The 8.3 cm outlier exceeds the nominal 5 cm physical margin. Fidelity has not
passed. Complete signed lateral-position/heading/yaw-rate/speed outlier analysis,
separate initialization from physics/timing error, and fix proven discrepancies.
Do not fit tire parameters or invent command delay from aggregate RMSE alone.

## Validation and restart

Main source/controller targets built; all **13 CTest suites passed** in
`artifacts/phase2_baseline/tests_handoff.log`. This includes updated convex QP
tests, 25-witness replay, native capture/rollback, history-age overflow and the
existing source/physical tests. A final controller-only status-classification
correction was rebuilt in `build_handoff_final.log` in the same directory.
No race or timing qualification follows from these checks.

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
cmake -S f1tenth_mpc -B build/f1tenth_mpc
cmake --build build/f1tenth_mpc --target \
  mpc_controller_node mpc_source_exact_offline_optimize mpc_source_nlp_oracle \
  mpc_source_live_fixture mpc_source_benchmark_capture_check \
  mpc_source_feasibility_witness_probe test_mpc_source_convex_qp \
  test_mpc_control_time_predictor -j4
ctest --test-dir build/f1tenth_mpc --output-on-failure

python3 tools/competition_failure_benchmark.py replay \
  --output artifacts/competition_resume_20261105 --iterations 20 --repeat 2

build/f1tenth_mpc/mpc_source_exact_offline_optimize \
  artifacts/phase2_current_20/epoch_757 \
  artifacts/phase2_current_20/epoch_757/initial.controls 100 \
  artifacts/resume757.controls > artifacts/resume757.json

MPC_SOURCE_NLP_FULL_HESSIAN=1 \
MPC_SOURCE_NLP_ELASTIC_GLOBALIZATION=1 \
MPC_SOURCE_NLP_MERIT_GLOBALIZATION=1 \
build/f1tenth_mpc/mpc_source_exact_offline_optimize \
  artifacts/phase2_current_20/epoch_769 \
  artifacts/phase2_current_20/epoch_769/initial.controls 100 \
  artifacts/resume769.controls > artifacts/resume769.json
```

For a fresh checkout, use the generated `competition_resume_20261105/epoch_*`
paths instead of the older local `phase2_current_20` paths. Build environment
is host ROS Jazzy; Docker competition images use Humble. Replay needs no running
ROS graph or Unity. No new simulation was launched in this work.

## Next work

1. Resolve precision/derivative consistency through a minimal double solver
   path and explicit legacy replay conversion/validation. Preserve source
   equations, original weights/samples and physical limits. Independently
   validate command rounding and original source publication feasibility.
2. Finish nonlinear curvature and semismooth peak/physical-branch treatment.
   Check multiplier identities when source substep schedules change. The
   existing full-curvature experiments help 769 but do not solve 775. Do not
   weaken KKT tolerances or increase iterations and call a stall convergence.
3. Re-run every recorded problem and the receding-horizon sequence, independently
   score every returned primal, and retain earlier feasible iterates. Extend
   live capture before making new globalization operations live.
4. Add source-validated terminal viability including heading, lateral position,
   yaw/rack state and actuator history/delay; speed-only viability is insufficient.
5. Then optimize/qualify timing under Unity/ROS contention, complete real race
   qualification, restore speed, and remove experiments that failed the benchmark.
