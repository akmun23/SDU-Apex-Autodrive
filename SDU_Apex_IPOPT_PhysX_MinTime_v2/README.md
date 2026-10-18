> **v2.4.1 contact-arm correction:** see [CHANGELOG_v2_4_1.md](CHANGELOG_v2_4_1.md). Run `bash run_convergence_campaign_v240.sh` after installing; its internal v240 names are retained. No IPOPT/native feasibility is claimed.

## v2.3.16 — Native full contact-point wheel slip

The generated working CasADi wheel model now evaluates the velocity at each
wheel tire-force application point using chassis body-forward/body-left speed,
yaw, **roll rate** and **pitch rate**. Source COM-relative contact arm = wheel
centre minus one radius. The original historical Python source is unchanged.

Source-independent baseline checks using the saved 2.4 m/s native flying-lap
wheel telemetry and 143-station warm start reduce lateral-slip RMS residual
from 5.47e-4 to 5.90e-5 rad and lateral tire-force RMS residual from
0.577 to 0.068 N. These comparisons use interpolated body angular rates and
are not an exact 1 ms native simulator equivalence proof.

After installing the in-place ZIP and activating the existing `.venv`, run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/
bash diagnose_contact_v2316.sh
bash diagnose_phase1_v2316.sh
```

The force parity diagnostic refuses incomplete native contact traces and
requires the saved 143-station native seed to phase-align to the 481 contact
observations. The Phase-I run has **0 brake zones for diagnostics only**. Full
CAWB braking, strict zero-defect feasibility, native replay, and Unity parity
remain required for racing-lap acceptance. No optimizer minimum-time lap has
been validated by this update.

## Latest: v2.3.15 — Convert native callback lateral tire FORCE to model chassis axes

The v2.3.14 native-slip correction is right-positive as PhysX expects. Its raw
callback lateral force is ALSO right-positive; the IPOPT chassis and Ackermann
geometry are forward/LEFT/up. Applying right-positive `fy` as left-positive
in the body-force sum caused the user's v2.3.14 initial Phase-I defect to grow
to **8.7611**, despite strong tire-callback and slip parity. Using the recorded
481-sample native tire forces, transforming the force with the wheel's native
right unit vector lowers the difference between reconstructed lateral chassis
acceleration and 25ms native `dv/dt + u*r` from approximately **4.028 to
0.201 m/s² RMS**, with the timestamp/scene-contact caveats below.

Only the **generated working wheel model** changes: the native callback raw
force `fy` is preserved for diagnostic outputs, but the chassis rotations use
`fx_body = fx*cos(steer) + fy*sin(steer)` and
`fy_body = fx*sin(steer) - fy*cos(steer)`. No friction values fitted; original
source, strict zero-defect gate, and native validation remain mandatory.

Run `bash diagnose_force_frame_v2315.sh` (saved v2.3.14 native slip CSV;
**no IPOPT and no native replay**) before `bash diagnose_phase1_v2315.sh`
(bounded **40 Phase-I iterations**, no brake zones only for diagnosis).
No feasible racing lap or validated sub-4.9-second lap is claimed.

## Latest: v2.3.14 — Correct native PhysX lateral-slip frame in generated model

The v2.3.13 481-time-step, 1,924-wheel sample native PhysX audit exposed reversed lateral-slip sign: model wheel-axis contact lateral speed uses forward/LEFT/up while native PhysX/Unity lateral slip uses the opposite right-positive sign. At identical recorded wheel speeds and native suspension forces, reversing the sign reduces lateral-slip RMS from **0.003828 to 0.000547 rad**, and lateral tire-force RMS from **3.762 to 0.577 N**. The native tire callback's longitudinal forces remain unchanged. No source history or tire coefficients are fitted/modified.

Run `bash diagnose_slip_v2314.sh` to compare the **actual generated CasADi OCP function** to the saved v2.3.13 native CSV without another native replay or IPOPT. After passing this check, `bash diagnose_phase1_v2314.sh` runs a bounded 40-iteration elastic feasibility diagnostic using the saved original native warm start. Any residual > 0 means NO valid periodic lap; --brake-zones 0 is diagnostic only. Final racing solve requires CAWB braking, strict exact dynamics, 1ms native/Unity checks.

## Latest: v2.3.13 — Audited native vs CasADi wheel-slip reconstruction

The v2.3.12 1,924-wheel callback parity check passed, but IPOPT initial/max defect **regressed** relative to v2.3.10. Before modifying tire friction or running further IPOPT iterations, run `bash diagnose_slip_v2313.sh`. It generates a same-trajectory, per-wheel comparison of native PhysX slip and the generated OCP CasADi slip/contact force calculations using the recorded wheel omega and body state. The native 1 ms update/scene-step telemetry staggering is documented and **not** claimed to be exact zero-offset contact state. No OCP physics or historical source is changed.

## Latest: v2.3.10 — Correct source suspension roll/mount coordinate basis

Native per-wheel loads confirmed the prior symbolic suspension load transfer had
**opposite lateral sign** at nearly every warm-start track node. The source
geometry was Unity x-right, whereas the historical suspension roll angle is
in the model left-positive frame. Convert only the mount lateral coordinate
before evaluating the same PhysX-derived load formula. The measured load RMS
error decreased from 2.174 N to 0.992 N in the v2.3.9 audit, but lateral
acceleration RMS *increased* from 0.410 to 0.486 m/s². This is a real
coordinate correction, **not** a solution to the native/CasADi model gap.

Run `bash diagnose_model_v2310.sh` and optionally
`bash diagnose_phase1_v2310.sh` in this existing optimizer directory. The
latter performs at most 40 IPOPT feasibility iterations with the unprojected
native warm start and no native replay. A fully feasible zero-defect IPOPT lap,
4.9 s target or native validation is **not** established.

## Latest: v2.3.8 — Scalar tire-force forensic audit (17 October 2026)

The user's v2.3.7 IPOPT report shows only one accepted improvement in 50 iterations:
strict max dynamics defect 1.064776 -> 1.062793, still far from the
full-lap feasibility requirement. The largest seed errors are lateral force
balance (1.064776), suspension loads (/10-scaled 0.658572) and roll-rate
collocation (0.604459). More IPOPT iterations will not establish whether
this is a native-versus-symbolic model mismatch or a seed projector error.

v2.3.8 evaluates tire forces **per node with scalar CasADi arguments**, exactly
as the original IPOPT problem constructs them. This prevents using an
unverified 1xN batched calculation to initialize scalar source equations.
The projection audit reports the difference between the old batched form and
new scalar form. It does not modify physical equations, solver overlay,
measured motion, or native validation requirements.

Run **`bash diagnose_force_v238.sh`** from the existing optimizer directory.
It uses the saved native PhysX warm start, restores your existing source wheel
model and produces `force_consistency_v238/summary.json` and
`force_consistency_v238/per_node.csv`. It does **not** launch IPOPT or native
PhysX. The per-node file lists the lateral-force residual with native loads,
with symbolic loads, and all four wheel loads/forces/slips. The key check is
whether its raw native-load lateral defect reproduces the existing 1.064776
IPOPT seed error and whether scalar vs batched projected forces differ.

No zero-defect IPOPT lap, minimum-time solution, or sub-4.9-second native/Unity
lap has been demonstrated. Continue with a physical-model correction only
once the mismatch is traced to the specific source operation and state.

## Latest: v2.3.7 (17 October 2026)

User data show projected native warm-start worsened the initial maximum strict
dynamics defect (1.064776 unprojected vs 4.055590 projected), while the old
checkpoint logic accepted a lower weighted L2 objective even as the max defect
increased to 12.311628. The new checkpoint accepts a step only when both
metrics improve; diagnostic JSON includes per-equation-class max and RMS.
The actual vehicle model and strict zero-slack acceptance are unchanged.

From the optimizer directory, run `bash diagnose_phase1_v237.sh` after the
in-place ZIP extraction. It uses `--no-seed-force-projection`, 50 IPOPT
feasibility iterations, drive-only mode, and skips native replay. It is a
model-disagreement diagnostic only. **No feasible IPOPT minimum-time lap or
sub-4.9s native/Unity result is certified by this release.**

## v2.3.1 update: measured native seed reconciliation

See **CHANGELOG_v2_3_1.md**. The v2.3.0 native flying lap completed in the
user's environment; its measured finite-difference accelerations and spring
loads do not satisfy the *different* analytic CasADi model. v2.3.1 projects
**only those algebraic states** onto CasADi's own tire/spring formulas and
allows IPOPT to retain projected elastic intermediate candidates satisfying
all hard nonelastic constraints. The final zero-defect dynamic constraints and
native/Unity acceptance requirements are unchanged.

For a mode-matched diagnostic **reuse the saved 143-node native lap** (requires
the parent output directory with `provenance.json` and matching map/centerline):

```bash
python3 run_ipopt.py --repo-root .. --mesh-spacing 0.20 \
  --seed-file ./optimized_practice_v2_3_0/warmstart_physx/native_warmstart.npz \
  --brake-zones 0 --phase1-hessian limited-memory \
  --feasibility-iterations 500 --iterations 1200 \
  --physx-binary ./native_physx_vehicle_step \
  --output ./optimized_practice_v2_3_1_drive_bootstrap --allow-history-fetch
```

**Why zero initial brake zones?** The actual native warm-start uses positive
throttle throughout. The requested three IPOPT brake zones impose full
zero-throttle wheel braking not observed in that lap. First solve a trajectory
with a *matching* driving mode, then progressively introduce braking to
pursue the sub-4.9-second minimum-time target. Zero-brake mode is a
bootstrap diagnostic, **not** the final racing strategy.

Check `warmstart_physx/seed_projection_audit.json` to quantify symbolic
versus measured load/acceleration differences. No v2.3.1 convergence or lap
time is guaranteed from unit tests.

---

> **v2.3.0 (current):** The historical IPOPT solver now receives an initial
> **native PhysX flying-lap wheel/body trajectory** by default. Previously it
> always began with a synthetic curvature-derived state whose defect stalled
> at 0.466543 across many IPOPT runs. The new native seed is a required
> preflight: if the cautious follower cannot complete a **wall-clear lap**,
> the runner writes warm-start failure diagnostics and **does not run IPOPT**.
> This is not a minimum-time trajectory, nor is successful convergence
> established. Use `--seed-backend synthetic` to restore old behavior or
> `--seed-backend source` to use the source-derived C++ vehicle plant.
> See `CHANGELOG_v2_3_0.md`.

> **Update v2.2.9:** CasADi simple-bound detection and off-domain-safe Frenet divisions in phase I; braking-topology warning. These changes preserve the exact equations on feasible states, but no successful minimum-time lap has been demonstrated. See `CHANGELOG_v2_2_9.md`.

> **Latest packaging / source-compatibility update: v2.2.8.** The source loader now accepts native PhysX runtime revisions that do not declare `kSteeringLimitDeg`, using recognizable implemented steering update equations from the current checked-in C++ plant while warning when compile-time alias parity cannot be verified. This fixes a startup exception, **not** IPOPT feasibility. See `CHANGELOG_v2_2_8.md`.

> **Latest numerical update: v2.2.7.** Phase I now solves an elastic dynamic-defect projection, with all track/actuator bounds hard and a mandatory zero-defect audit before time optimization. It replaces the stalled tolerance continuation of v2.2.6. See CHANGELOG_v2_2_7.md.

## v2.2.5 in-place update — synthetic-seed feasibility fix

This version replaces the Phase-I objective for the restored IPOPT OCP:
**minimize the squared, scaled vehicle-dynamics equation defects** plus a
small `0.001 ×` initial-reference regularizer. Previously Phase I minimized
only the distance to a synthetic trajectory; consequently stage 1 (tolerance
0.5) was accepted without any effort to reduce the physical defects, and
stage 2 (tolerance 0.35) stalled at a 0.163 residual. v2.2.5 optimizes the
actual physics residual merit **even at initially feasible relaxed stages**.
It preserves the best feasible seed if a failed IPOPT attempt increases the
residual, and refuses to enter minimum-time optimization without the full
zero-tolerance equality constraints (existing 1e-3 acceptance threshold).

There is NO change to the 1 ms native PhysX validator, the Unity-derived tire
curves, footprint clearance, actuator limits, or the IPOPT minimum-time
objective. The historic source in `_restored_git_history` remains untouched;
the known v2.2.4 working solver is safely upgraded. Existing `.venv`, native
binary and all run outputs remain in the same folder.

**Install:** extract the v2.2.5 ZIP over the same repository root with
`unzip -o ... -d .`, then run from `SDU_Apex_IPOPT_PhysX_MinTime_v2/`.
Use a new `--output` directory to keep diagnostics from earlier runs.

**Not yet demonstrated:** A converged periodic IPOPT lap or sub-4.9 s native or
Unity run. The numerical correction should be judged by whether the reported
`normalized_physics_defect_energy` decreases and whether subsequent stage
violations approach zero. If the same physical constraints remain infeasible,
that suggests a deeper initial-state/dynamics-model inconsistency rather than
a shortage of IPOPT iterations.

## v2.2.4 update

IPOPT Phase-I homotopy now skips relaxed stages already feasible at their initial seed and stops as soon as a relaxed problem cannot be solved; it no longer passes failed or NaN points into tighter stages. Minimum-time optimization remains gated on the final exact zero-defect equations. See CHANGELOG_v2_2_4.md.

# SDU Apex — Restored IPOPT / PhysX minimum-time raceline optimizer

**Target:** Porto / practice map (`autodrive_practice_20260924_b.pgm`), beat the **other teams'** 4.9 s record. The competition map is a *different* course (other teams' record 11 s).

## What this package actually does

This replaces the unsuccessful random-parameter search with the **original recovered CasADi/IPOPT minimum-lap-time nonlinear program**. Specifically it restores the pre-cleanup `optimizer_unity_wheels.py` (not the earlier scalar/speed-control OCP), which optimizes:

- Lateral path offset and heading at every spatial node (whole-lap free geometry);
- Longitudinal and lateral velocity, yaw rate and four independent wheel angular velocities;
- Heave/roll/pitch and their rates, wheel normal loads, suspension and forces;
- Physical steering rate and steering limit, direct normalized throttle (not target speed);
- Contiguous **exact-zero** throttle/braking phase lengths;
- Periodic full-lap states, track footprint/wall clearance and the **minimum-lap-time integral**.

The symbolic CasADi tire/force constraints are from **the simulated Unity car's source** (wheel slip equations, friction-curve response, combined slip and spring/damper), *not* generic Pacejka/real-car assumptions. The wrapper reads the **current** checked-in `unity_source_plant.cpp` and `physx_odom_reset_initial.json` rather than depending on the old deleted training captures. The precise historical IPOPT solver and its supporting source files are recovered **byte for byte from Git commit `ca120e0cde7ae60e70082d38f86c5931f79f148f`** on first run, into `_restored_git_history/`.

The solver's resulting steering and throttle sequence is then **actually driven through the current native PhysX 4.1 executable**, which steps 1 ms through your existing recovered wheel/shape/tire configuration. The 25-tick batched native API (40 Hz) is used: no arbitrary generic physics substitute is used in the validator. The validator checks the swept vehicle perimeter against the provided PGM at every control stage, direction at the lap gate, the warm-up lap, and the scored lap. **No MPC raceline is exported on failure.** One optional outer iteration feeds measured native lateral tracking discrepancy into IPOPT's track-boundary constraints and re-solves.

### Accuracy boundary — important

The **IPOPT dynamics are not an exact 1 ms discrete PhysX scene**. They are the original symbolic four-wheel source-derived model using spatial collocation and simplified contact integration. The native PhysX validator *is* the actual existing repository executable, but its flat-plane reconstructed colliders/callback are not proven byte-for-byte identical to Unity's WheelCollider and scene. The wrapper does not magically turn the old IPOPT model into a proven 1:1 Unity model. **Sub-4.9 s has NOT been demonstrated by this ZIP.** Any prediction must pass native-PhysX replay and, eventually, actual Unity closed-loop validation before being considered a race-capable result.

## Install / upgrade in place (Linux)

Extract over the SAME existing optimizer folder; no second environment or
new optimizer directory is required:

```bash
cd ~/Documents/GitHub/SDU-Apex-Autodrive
unzip -o ~/Downloads/SDU_Apex_IPOPT_PhysX_MinTime_v2_2_8_IN_PLACE_UPDATE.zip -d .
cd SDU_Apex_IPOPT_PhysX_MinTime_v2
source .venv/bin/activate
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/
python3 run_ipopt.py --repo-root .. --setup-check
```

Preserves: `.venv/`, `native_physx_vehicle_step`, `_restored_git_history/`,
`_working_solver_limited_memory/` (safely upgraded if unchanged by the user),
and existing `optimized_*` runs.

**v2.2.3:** IPOPT Phase I now uses progressive temporary tolerance on the
wheel/body dynamics equations (0.5 → 0.2 → 0.07 → 0.015 → 0), with all
vehicle and track limits hard. The original synthetic-seed distance cap is
inactive during feasibility; it never represented a physical constraint.
**Only the zero-defect final solve is eligible for minimum-time optimization.**
No candidate can be exported before native PhysX checks. This is a numerical
method, **not a guarantee of convergence or a lap below 4.9 s**.
See `CHANGELOG_v2_2_3.md`.

### Check restoration and map *without* running IPOPT or PhysX

```bash
python3 run_ipopt.py --repo-root .. --allow-history-fetch --setup-check
```

`--allow-history-fetch` only fetches the **historical source commit** from the repo's GitHub origin if it is missing locally; it is otherwise offline. It **does not start Unity** or replay a bag. You may inspect the retrieved original implementation in `_restored_git_history/SDU_Apex_Autodrive_Exact_MinTime_Optimizer_v1_3/f1tenth_planning/autodrive_mintime/`.

### Build your native PhysX process (skip if a working binary is already present)

```bash
./build_native_physx.sh .. ./native_physx_vehicle_step
```

Requires the PhysX 4.1 source libraries already present in your repository's `external/physx-4.1-source/.../bin/linux.clang/release` directory. If missing, use the existing repository localization build first. The optimizer never fabricates or silently substitutes a binary.

### Run the *real IPOPT* optimizer on Porto / practice

```bash
python3 run_ipopt.py \
  --repo-root .. \
  --map ../f1tenth_planning/maps/autodrive_practice_20260924_b.pgm \
  --physx-binary ./native_physx_vehicle_step \
  --mesh-spacing 0.20 \
  --brake-zones 0 \
  --phase1-hessian limited-memory \
  --feasibility-iterations 1200 \
  --iterations 1200 \
  --native-max-seconds 18 \
  --warmup-laps 1 \
  --native-correction-passes 1 \
  --output ./optimized_practice_ipopt \
  --allow-history-fetch
```

This performs a nonlinear *minimum-time solve*, **not 1,200 random native PhysX evaluations**. In particular:

- `--iterations` refers to IPOPT's optimization iterations after the feasibility stage.
- `--mesh-spacing 0.20` is deliberately below the original optimizer's internal 0.25 m maximum node span.
- Phase I no longer imposes the historical captured-motion trust-radius inequality on the synthetic centerline seed. This was an artificial distance-to-measured-Unity-capture cap; the Phase-II speed-optimization trust regions still apply. Phase I progressively tightens dynamic equality defect bounds and requires strict convergence before minimizing time. Inspect `ipopt_report_pass_00.json` for each stage and run `python3 diagnose_report.py optimized_practice_ipopt/ipopt_report_pass_00.json` if it fails.
- `--brake-zones 3` seeds the **count/order** of discrete zero-throttle brake phases from curvature, while IPOPT optimizes their lengths. Try `--brake-zones 0`, `2` or `4` in separate **new** output directories for other mode topologies: an NLP solver cannot continuously invent a brand-new discrete brake phase.
- `--native-max-seconds 18` covers **both** the warm-up lap and scored lap.
- The chosen map defaults to **practice**. The longer competition map is only used if explicitly selected with the **matching** centerline.
- `--native-correction-passes 1` attempts one more IPOPT solve only when the first native replay hits the wall. The observed signed lateral tracking residual is inserted into the next corridor constraints; this does **not** fix the tire model itself.
- The output is **not** automatically installed into any ROS/MPC runtime folder. Review the results and validate in Unity before deployment.

### Tests (ROS Jazzy environments)

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/
```

### Outputs

```text
optimized_practice_ipopt/
  provenance.json                    map and vehicle source SHAs and limitations
  ipopt_report_pass_00.json          feasibility and minimum-time solver report
  ipopt_candidate_pass_00/
    optimizer_report.json
    unity_wheel_hybrid_states.csv    full OCP state, wheel slip and throttle results
    unity_wheel_hybrid_raceline.csv   original OCP research export
  native_pass_00/
    native_trace.csv                 PhysX rollout and tire slip
    commands_40hz.csv               applied steering/throttle controls
    native_report.json              PhysX lap time, completion and safety
  [optional pass_01 files...]         correction iteration
  report.json                        acceptance/rejection and benchmark result
  autodrive_mintime_raceline.csv     ONLY ON COMPLETED, MAP-SAFE NATIVE LAP
```

Exit codes: `0` native validated with MPC CSV, `2` IPOPT failed to establish a feasible lap, `3` deliberately skipped native validation, `4` native validation failed, `1` package/configuration/runtime error. **A valid sub-4.9 s result is possible but is not guaranteed.**

### Compare with Unity *offline*

If you have **an authorized existing Unity ground-truth capture** with aligned time/pose/velocity columns, use:

```bash
python3 compare_truth.py \
  --truth /path/to/unity_truth.csv \
  --trace ./optimized_practice_ipopt/native_pass_00/native_trace.csv \
  --output ./optimized_practice_ipopt/unity_parity_report.json
```

Inspect `python3 compare_truth.py --help` for the exact required columns. **The ZIP does not launch or control Unity, cannot claim Unity parity without observed data, and runs no new simulator experiments automatically.**

## Source/provenance

- Original source (restored locally, unchanged): `ca120e0cde7ae60e70082d38f86c5931f79f148f` before the October 17 cleanup.
- Live source parameters: current checkout's `f1tenth_mpc/src/unity_source_plant.cpp`, `f1tenth_localization/config/physx_odom_reset_initial.json`.
- Native validation: current checkout's `f1tenth_localization/src/physx_native_vehicle_step.cpp` and recovered Unity shapes, 1 ms stepping.
- Tire-source model is **not** exact contact scene; wheel/spin spatial integration uses the historical CasADi approximations; native replay identifies divergence and rejects the result.
- No real-car friction, Pacejka coefficients or arbitrary longitudinal acceleration envelope was added.

## v2.1: IPOPT Phase-I numerical trust-region fix

The original Phase-I NLP included an artificial `feasibility_objective <= 25` constraint. This was tuned for warm-starting a *measured captured Unity lap*, but the restored wrapper has no capture and synthesizes its state seed from the map. A failed run reported objective 31.7362664, constraint violation 6.7362661—the difference is the outdated 25-unit cap. v2.1 defaults to **200** and allows `--feasibility-trust-region` to override it. All dynamics, periodicity, wall, steering, tire and suspension constraints remain enforced. **This does not guarantee solver feasibility or a sub-4.9 s lap.**

When Phase I fails, `run_ipopt.py` now prints the largest named constraint violations and keeps the full report. Run `python3 diagnose_report.py <output>/ipopt_report_pass_00.json` on a saved failure report for more detail. An optional `--diagnose-initial-only` constructs and reports initial residuals without calling IPOPT or PhysX.

To repair an already-extracted v2 installation without changing packages, add `feasibility_trust_region=200.,` alongside `wall_clearance_m=args.wall_margin,` in the `kwargs=dict(...)` passed to `solve_bottom_up`; the packaged v2.1 runner does this automatically. Always use a fresh `--output` directory.

## v2.2: response to the 237-node practice-map feasibility failure

The user's `ipopt_report_pass_00.json` shows **physical dynamics defects**
(max = 1.10059; primarily yaw-rate and lateral-velocity collocation) after
1,000 Phase-I IPOPT iterations, *not* a trust-bound issue. A predicted time
shown in an infeasible iterate is **not a valid lap time**.

v2.2 keeps the original physical NLP and all clearance checks, but allows a
limited-memory Hessian in the **Phase-I feasibility stage**. The original
`optimizer_unity_wheels.py` is preserved byte-for-byte in `_restored_git_history`;
a single IPOPT numerical option is changed in a **separate copy** named
`_working_solver_limited_memory/`. Both source checksums are saved in the
run's `provenance.json` and `ipopt_report_pass_00.json`.

Try a coarse, drive-only feasibility-first solve to remove discontinuous
exact-zero braking phases from the *initial* periodic-lap search:

```bash
python3 run_ipopt.py \
  --repo-root .. \
  --map ../f1tenth_planning/maps/autodrive_practice_20260924_b.pgm \
  --physx-binary ./native_physx_vehicle_step \
  --mesh-spacing 0.20 \
  --brake-zones 0 \
  --phase1-hessian limited-memory \
  --feasibility-trust-region 200 \
  --feasibility-iterations 1200 \
  --iterations 1200 \
  --native-max-seconds 18 \
  --warmup-laps 1 \
  --native-correction-passes 1 \
  --print-level 5 \
  --output ./optimized_practice_phase1_v2_2 \
  --allow-history-fetch
```

If that feasible lap succeeds, add exact-zero braking with a **fresh** run
using `--brake-zones 1`, `2` or `3`; the chosen topology is fixed for each IPOPT
solve, while individual phase lengths remain variable. No successful drive-only
lap guarantees a competitive time, and a brake-phase run can still fail.

To reproduce the previous numerical method for comparison, use
`--phase1-hessian exact`. Run tests with:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/
```


## v2.3.3 Phase-I performance fix

For the previous v2.3.2 pre-IPOPT stall inside `opti.initial()` / `opti.value()`, run `bash diagnose_phase1_v233.sh` after installing this in-place update. It uses the saved native seed from `optimized_practice_v2_3_0`, skips native replay, records `ipopt_v233_debug.log` and stops after 180 seconds. The historical full-lap CasADi/IPOPT formulation and its strict zero-defect acceptance are unchanged; no convergence or lap-time result is claimed.

### v2.3.4 strict feasibility failure audit

`bash diagnose_phase1_v234.sh` runs a bounded, no-PhysX-replay 24-iteration
feasibility diagnostic from `optimized_practice_v2_3_0/warmstart_physx/native_warmstart.npz`.
If IPOPT does not find strict zero-elastic-slack feasible dynamics, the returned
`ipopt_report_pass_00.json` now includes labeled residuals in
`nlp_diagnostics.worst_constraints` and the Phase-I merit in `solve_stages`.
An IPOPT iteration table or low relaxed `inf_pr` alone is not a valid racing
solution; no raceline is exported without an accepted strict solution and
native validation.

## v2.3.6 diagnostic (native algebraic projected seed and wheel-speed substep correction)

```bash
bash diagnose_phase1_v236.sh
```

Uses the existing native warm-start file, 50 IPOPT feasibility iterations and a 240-second timeout, with `--brake-zones 0` and `--skip-native` ONLY to isolate mathematical model feasibility. Unlike v2.3.5, it does **not** use `--no-seed-force-projection`. The generated (not historical) OCP solver now selects PhysX 5/1 wheel substeps using actual longitudinal velocity `u=x[2,k]`, not Frenet offset `ey=x[0,k]`. Strict physics feasibility, full-brake optimization and native vehicle validation remain mandatory before exporting any controller path.

### v2.3.9 suspension coordinate diagnostic

`bash diagnose_suspension_v239.sh` performs a **no-solve, no-PhysX-replay**
comparison of the measured four-wheel native suspension loads and the scalar
model under the original and reflection candidate coordinate conventions.
Results are written to `suspension_frame_v239/`. The original IPOPT wheel
model is **not** modified by this audit.


## v2.3.12 source-matched tire force callback

See `CHANGELOG_v2_3_12.md`. Run `bash diagnose_callback_v2312.sh` (uses existing v2.3.11 native-contact CSV; no simulation/IPOPT), then `bash diagnose_phase1_v2312.sh` for a bounded feasibility-only comparison against v2.3.10. Race lap still requires CAWB braking, strict IPOPT dynamics convergence, native PhysX replay, and Unity verification.
