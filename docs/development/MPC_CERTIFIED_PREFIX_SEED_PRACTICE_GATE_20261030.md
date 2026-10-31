# MPC recovery-seed fix: verified practice robustness and IROS investigation

**Date:** 2026-10-30. **Branch/source:** `be72ca234` with pre-existing dirty/uncommitted working tree. **Status:** practice five-run strict gate **PASSED**. IROS full 7.050 s source witness **NOT VALIDATED** (no completed laps). All changes are opt-in; protected production configuration profiles have not been promoted.

## The concrete recoverability defect

Previously, after a *published and exactly 75ms-validated* short fallback action, `MpcControllerNode::control_timer_callback()` forcibly discarded the entire `rti_memory_.warm_start_controls` sequence on the next cycle. The solver thus forgot the next two verified actuator commands and generated a fresh source-witness cold nominal from a vehicle that had already deviated into the short fallback. The code also discarded a fully accepted nominal when an *older* fallback remained present in the 16-action history. That creates a discontinuity between the issued command/queue and the linearization's next plan.

The opt-in change (`fallback_retain_validated_seed_enabled`, **default FALSE**):
- Retains only the three commands from an exactly 75ms certified/published recovery; the solver shifts out the already issued first command when seeding the next cycle.
- Discards the rejected *unvalidated* 750ms tail and resets its ADMM state, never publishing a control on the basis of only a good nominal.
- If the last issued action was already accepted as a **new fully exact-valid 750ms** plan, preserves that good nominal despite old fallback activity still in the 16-action diagnostic window.
- Preserves physical source-plant rollout/actuator delay, full corridor constraint, native source raceline/speed, and 20ms publication gate.
- Files: `f1tenth_mpc/include/mpc_fallback_seed.hpp`, `f1tenth_mpc/src/mpc_controller_node.cpp`, `f1tenth_mpc/test/test_mpc_fallback_seed.cpp`, `f1tenth_mpc/CMakeLists.txt`.
- Three specific unit tests passed; final full MPC suite **205 tests, 0 errors/failures**.

## Five independent practice validations: all PASS

Frozen overlay: `live_runs/mpc_collision_margin_experiment_20261030/practice_guarded_prefixseed_20261030_01.yaml`. Source: `f1tenth_planning/trajectories/autodrive_practice_20260924_b/source_command_4p800.csv`. Main weights e_y=5000, e_psi=1700, u=45, steering_command=4.5, original 0.08rad SQP steering trust, 6 Jacobian workers. Single change relative to prior 4/5 guarded profile is prefix-seed correction, not altered racing speed. Each Unity run: 1 warmup + 10 scored laps, 40 Hz physical actuator interface, MPC 200 Hz callbacks. No production YAML edited for promotion.

| Run suffix `practice_prefixseed_fivegate_20261030_XX` | 10 scored lap total [s] | Recovery actions | MPC deadline misses | Simulation collisions | Result |
|---|---:|---:|---:|---:|---|
| 01 | 49.340001 | 1 | 0 | 0 | PASS |
| 02 | 49.374000 | 11 | 0 | 0 | PASS |
| 03 | 49.486000 | 6 | 0 | 0 | PASS |
| 04 | 49.395999 | 2 | 0 | 0 | PASS |
| 05 | 49.257999 | 3 | 0 | 0 | PASS |

**Gate:** `prefixseed_practice_fivegate.json`: `passed=true, errors=[]`. Five out of five full races, 50 scored laps. Mean total **49.3708s**, mean scored lap **4.9371s**. Fastest ten-lap run 49.258s; slowest 49.486s (both below strict 49.5s). Highest measured callback gate time **18.832ms**, no >=20ms deadline miss. Total 23 recovery actions across the five full races. Although all are clean, minimum reported Frenet corridor slack was sometimes <1mm; this is **not** the same as independently certified true physical footprint clearance, and no continuous swept-body certificate is established.

Previous guarded-profile five-run baseline completed only **4/5**; previous steering weight 4.5 unguarded completed **3/5**; earlier steering weight 3 baseline only **2/5**. These are finite observational samples, not statistical proof of indefinite reliability.

Raw output: `live_runs/mpc_collision_margin_experiment_20261030/prefixseed_practice_repeat_results.json`, `prefixseed_practice_fivegate.json`, each `live_runs/practice_prefixseed_fivegate_20261030_XX/` (full ROS bag, report, CSV and diagnostics). Never override this profile with an unvalidated IROS-specific tuning.

## IROS full 7.050s high-clearance reference: not robust

Source `live_runs/optimizer_iros_adaptive_regions_v6_20261030/exports/83d2ebd3f02d1be7/source_command_raceline_with_witness.csv`: **7.050s native source-plant, not Unity lap time**, peak requested speed ~11.87m/s, extra sampled body clearance +0.03471m beyond hard 5cm margin. Each IROS test requested 11 total simulator laps, no artificial speed limit, but stopped before the first lap from invalid future corridor.

| Experiment | One-run IROS progress [m], loop ~43.06m | Full laps | Publication deadline misses | Note |
|---|---:|---:|---:|---|
| Prior guarded baseline (initial run) | 34.92 | 0 | 0 | Later identical baseline repeats failed at 18.73m / 23.27m |
| **Prefix-seed fix, default trust 0.08rad** | **35.66** | 0 | 0 | Initial single run; no completion |
| Prefix seed + early front-loaded signed steering search | 35.46 | 0 | 0 | No benefit; practice deadline miss |
| Prefix seed + SQP steering trust 0.14rad | 35.49 | 0 | 0 | No benefit |
| Prefix seed + SQP steering trust 0.22rad | 38.25 | 0 | 0 | **Single-run outlier** |
| Same trust 0.22rad independent repeats | 18.30 / 18.89 | 0 | 0 | Not repeatable; practice 49.564s, slower than gate |
| SQP steering trust 0.30 / 0.40rad | 18.61 / 18.69 | 0 | 0 | Worse |
| Prefix seed + strongly damped line-search wall warning | 23.26 | 0 | 0 | Worse; practice 49.627s |
| **Prefix seed + conditional 75ms exact terminal-extension check** | **35.80** | 0 | 0 | One run; essentially same as prefixseed alone. Practice 49.289s, zero deadline miss |

The earlier **7.075s native** source reference permitted one **8.1267s measured Unity competition lap** before a second-lap corridor rejection. No experiment yet demonstrates ten consecutive competition laps, let alone a 7.05s real Unity lap.

## Additional experiments and flags (DO NOT promote)

`viability_early_hazard_recovery_enabled` (default OFF) generates front-loaded signed corrective steering toward the center of the Frenet corridor. A ten-lap practice run was 49.310s but incurred one **20.491ms deadline miss**; reject.

`viability_line_search_wall_warning_enabled` (default OFF) triggers extra recovery only when a heavily damped valid line search (alpha<=0.25) followed a rejected corridor-crossing full step and minimum exact-valid slack<2.5cm. Warn-event frequency in one practice run: 4/2221 accepted cycles; in one IROS failure: 7/261. However actual IROS progress fell to 23.26m and practice was 49.627s. Reject.

`viability_terminal_extension_select_enabled` (default OFF) checks an additional 3×25ms exact source-plant rollout from the fully validated 750ms terminal physical state including queued actuators. It tries source-witness commands and held final commands, requiring exact world/corridor validation and terminal dynamics. It does **not** establish a recursively controlled invariant backup safe set nor guarantee safety indefinitely. Practice one run 49.289s; IROS one run ~35.80m, still fatal. Not promoted.

`world_map_authority_enabled` / `world_map_rejection_only` still default OFF. The experimental map-world-body sweep implementation compiles and passes tests but full-map validation once cost 20.78ms and rejection-only did not cure IROS; not a continuous Unity/PhysX swept substep certificate. Separate source `docs/development/IROS_7050_TEN_LAP_ROBUSTNESS_20261030.md`.

## Quantitative diagnosis: reference command interface and failure state

The native 7.050s witness contains 7050 ~1ms source-state samples but its **steering and throttle commands are already 40 Hz zero-order held**. All significant command changes fall at indices divisible by 25. Thus the optimizer is NOT fundamentally commanding at 1kHz; the concern is **path progress/time phase synchronization** and how the MPC handles its physical 2×25ms in-flight command queue and imminent nonlinear wall constraint.

Independent read-only audit `live_runs/mpc_collision_margin_experiment_20261030/iros_7050_command_phase_audit.json` compared actual published steering/throttle commands with the source witness at the same path station:
- Last 30 issued actions, baseline IROS: steering RMS deviation **0.05747rad**; throttle RMS **0.08987**.
- Prefix-seed correction: last 30 steering RMS deviation **0.00820rad**; throttle RMS **0.03053**.
- This strongly supports fixing the fallback command-sequence discontinuity, but **does NOT prove causality** by itself and does not demonstrate physical-state trajectory convergence. The later failure persists despite far closer commands.

Last-six-cycle diagnostics near IROS failure show normally accepted ADMM/line-search actions but shrinking nonlinear wall reserve. In some cycles the QP converges to residual ~1e-5, while its requested full steering step is *physically* invalid at stages ~18–28, requiring alpha~0.125/0.0625; the current state can look safe while the 750ms future is not. The fallback episode limit stops issuing uncapped 75ms prefixes after its finite limit to avoid unrecoverable collision. Merely raising that limit or relaxing walls was already tested and rejected.

## Required next step to improve IROS without sacrificing practice

1. Use a **deterministic saved-state full source-plant candidate oracle** starting at last accepted IROS cycles, with the full internal model state, physical steering rack and **2 delayed command slots**. Evaluate controls on a full 750ms *and successor* horizon with true 5cm clearance, an exact safe-prefix reserve, and a time-optimal progress objective. Compare whether a physically viable continuation **exists** in the feasible command envelope before changing solver weights. If not, MPC must anticipate earlier (recursive feasibility), not continue issuing short-lived successful commands.
2. Investigate source witness **40Hz command-phase alignment** vs the controller's progress-indexed ZOH command reference; quantify phase jitter, epoch offset, state and model mismatch, particularly steering edges near IROS station ~15–33m. Do not conflate phase mismatch with missing commanded sample rate.
3. If any robust fix is found, rerun **five separate complete practice races** under unchanged `source_command_4p800.csv`, strict <49.5s per 10 scored laps, zero collisions/deadline misses, *then* full-speed IROS attempts with independent repeats. Do not alter the successful practice profile while experimenting.
4. In parallel, the optimizer's Pareto portfolio could seek an equally fast 40Hz physically feasible reference with extra hard clearance and less sensitivity to latency. Current high-clearance 7.050s witness has large 40Hz command steps (up to 0.229rad steering), while the much smoother 7.050s witness has only 0.000389m extra margin; a new Pareto point with BOTH is desirable and cannot be invented by adjusting MPC weights alone.

### Reproducibility and ownership
- No unrelated staged/dirty worktree files were reset or committed.
- The existing optimizer is paused only during Unity resource ownership and resumed afterward.
- The five-run practice overlay is preserved frozen at `live_runs/mpc_collision_margin_experiment_20261030/practice_guarded_prefixseed_20261030_01.yaml`; all later flags are opt-in in separate files.
- Check hashes/run manifests before claiming future strict cross-version improvements.
