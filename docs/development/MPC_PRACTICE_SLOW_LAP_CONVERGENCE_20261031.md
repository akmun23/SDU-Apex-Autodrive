# Practice MPC slow-lap and graceful-recovery investigation — 2026-10-31

## Baseline and actual configuration

Five full independent ten-scored-lap runs: `live_runs/practice_ftz_previous_fivegate_config_20261031_0{1..5}`. 50/50 scored laps, zero collisions and zero deadline misses, ten-lap times 49.370 / 49.350 / 49.253 / 49.854 / 49.594 s. Goal is **5/5 under 49.5 s**, then sub-49.0 s without any hard source-physics or publication gate relaxation. Practice runner loads `f1tenth_mpc/config/mpc_practice_fastline.yaml` then `live_runs/mpc_collision_margin_experiment_20261030/practice_guarded_prefixseed_20261030_01.yaml`, **not** competition YAML. Effective practice knobs: lateral 5000, heading 1700, speed 45, steering command 4.5, steering slew 1500, steering trust 0.04 rad, throttle trust 0.05, world corridor margin 0.02 m, 30×25 ms native-source horizon and two SQP passes.

## Fifty-lap evidence

Comparing 16 fast individual scored laps <=4.92 s to 10 slow scored laps >=5.00 s:

| Metric | Fast <=4.92 s | Slow >=5.00 s |
|---|---:|---:|
| Mean time | 4.898 s | 5.037 s |
| Lateral tracking RMSE | 0.033 m | 0.071 m |
| Heading tracking RMSE | 0.028 rad | 0.051 rad |
| Speed tracking RMSE | 0.129 m/s | 0.237 m/s |

**Six of ten slow laps have zero accepted viability recovery actions**. Across 50 laps, simple Pearson correlations with lap duration are lateral RMSE r=0.738, heading RMSE r=0.833, speed RMSE r=0.862 and recovery count r=0.442. These are descriptive associations, not causal parameter sensitivities: steering/speed errors may be a consequence of earlier feasibility problems.

## Repeatable hazard at raceline station 2–3 m

The raceline ends at station approximately 26.593 m; the MPC diagnostic `progress_m` is cumulative. Analyze `progress_m % 26.593`, not raw cumulative progress, to identify recurring local corners. Recovery events around station 1.6–3.2 m occurred in **four of five races**. Other less frequent clusters occur around station 7–8 m and ~15 m.

The 5.177 s outlier (baseline run 04, scored lap 9) compared to run 03 scored lap 9 (4.953 s) at identical local stations:

| Station | Fast actual speed | Slow actual speed | Slow speed deficit relative to reference |
|---|---:|---:|---:|
| 1.5–3.5 m | 4.04 m/s | 3.41 m/s | 0.594 m/s |
| 3.5–6.0 m | 6.26 m/s | 5.27 m/s | 1.030 m/s |
| 6.0–10.0 m | 7.49 m/s | 7.28 m/s | 0.288 m/s |

Three accepted recoveries occur around station 2.26–3.01 m on the slow lap. The following acceleration zone stays under reference. The same run also had a 5.077 s lap **without any recovery**, exhibiting large e_y and e_psi near the hairpin and 10–15 m region. Therefore improve ordinary two-pass SQP tracking/convergence as well as fallback; never treat recovery suppression alone as sufficient.

## Recoveries are discrete corrections and cause cold-start numerical work

Across five baseline races, 21 recoveries have adjacent published-command comparisons. Median absolute steering change is **0.092 rad** on recovery versus **0.008 rad** on 7,344 line-search-accepted changes; 95th percentiles 0.1251 versus 0.0367 rad. Most selected recovery controls differ from the nominal first control by exactly **0.12 rad steering**, frequently with a **-0.08 throttle** correction. These are bounded, exactly validated steering tapers, but noticeably less smooth than ordinary SQP actions. **18 of 21** accepted recoveries followed a rejected nonlinear corridor stage >=23 of the 30-stage horizon; many are late-horizon viability problems, not imminent stage-0 collisions.

After recovery, `mpc_rti_memory_reset_solver_state` deliberately discards incompatible ADMM state but `mpc_rti_memory_set_warm_start_controls` retains the new exact-validated 30-stage control sequence. Across 21 recoveries the **first** subsequent callback has median ADMM iteration count **150**, versus **22** across all 11,125 measured solver callbacks; second subsequent callback median 163. The recovery reset itself cannot be called incorrect or the sole cause of the cold-start iteration count; a safe consistent ADMM primal/dual warm-start from the changed nonlinear trajectory should be researched rather than naively keeping stale multipliers.

## Controlled experiments and acceptance

`weight_u=47` alone: 49.653 s, four recoveries, clean, no deadline miss; not promoted. Single-variable screening launched with lateral 5300, heading 1750, steering-command weight 4.3, steering-slew 1450, throttle-slew 45. Each screen must preserve exact source checks, both SQP passes, legal commands, collision avoidance and <20 ms publication. An isolated fast run does not meet five-repeat qualification. Preserve the original validated policy unless all gates pass.

A separate **opt-in (default OFF)** experimental bounded recovery search (`viability_fine_taper_recovery_enabled`) was staged while trials ran but not deployed into their runtime: propose gentler +/-0.08 rad steering with -0.04 throttle plus original +/-0.12 rad (with -0.08 or zero throttle), retaining larger +/-0.18 rad/braking proposals within **the same eight candidate validations**. All candidate trajectories must still satisfy the entire 750 ms source model and terminal dynamics. Existing baseline remains unchanged with flag off. Full compiler/unit tests and new Unity races are required **before promotion**. If the experiment is not better and reliably safe, remove code/option and rebuild the original.

## Next engineering targets

1. Find state-aware, map-independent weight policy reducing off-raceline speed deficits in the hairpin instead of blindly increasing heading penalty; preserve physical clearance.
2. Improve SQP terminal viability earlier (stage 23–29) so valid fast horizons do not repeatedly require full emergency re-seeding. Keep R2; recent five-run baseline chose 5,203 of 6,119 R2 attempts.
3. Use exact full-source rollout comparisons to select minimally disruptive, forward-viable rescue seeds within the existing eight-trial deadline budget; quantify actual published steering and throttle jumps, post-recovery speed deficit, lap-time outliers and safe terminal continuation.
4. Longer-term: initialize a consistent ADMM primal/dual guess for the new validated recovery nominal (not stale solver memory), and measure post-recovery iterations and next-cycle SQP acceptance on identical saved states.

## Completed single-variable weight screen (native Unity, one full 10-scored-lap run each)

The following are diagnostic **one-run observations, not five-repeat qualifications**:

| Change vs effective practice baseline | 10-lap seconds | Accepted recoveries | Max callback ms | Outcome |
|---|---:|---:|---:|---|
| Speed-tracking `weight_u` 45→47 | 49.653 | 4 | 18.145 | clean, slower than best base run |
| Lateral tracking `weight_e_y` 5000→5300 | 49.501 | 1 | 18.087 | clean, narrowly above 49.5 |
| Heading tracking `weight_e_psi` 1700→1750 | 50.197 | 8 | 18.659 | clean, slower |
| Steering command penalty 4.5→4.3 | 49.449 | 6 | 18.511 | clean, insufficient consistency evidence |
| Steering slew penalty 1500→1450 | 49.545 | 7 | 18.218 | clean, slower |
| Throttle slew penalty 50→45 | 49.448 | 11 | 18.707 | clean, too many recoveries |

Every screened candidate retained full source-plant/corridor validation and had zero publication misses in its individual run; **none was promoted**. An already-running separate steering-slew=1400 series, five clean races, produced 49.299/49.739/49.654/49.607/49.563 s, also not consistently below 49.5.

## Experimental fine-taper rescue results: removed after regression

The default-off fine taper candidate bank was compiled with 215/215 C++ tests passing and run once via `live_runs/practice_ftz_fine_taper_20261031_01`. It completed all ten scored laps safely in **49.704 s**, with **17** accepted recovery interventions (vs baseline 1–8) and max callback **18.818 ms**. Four interventions used the new gentler -0.08 rad/-0.04 throttle correction, nine used -0.12 rad/-0.08 throttle. Although some individual steering jumps became smaller (median absolute recovery transition 0.072 rad, vs 0.092 across baseline 21 recoveries), the policy increased the frequency of source-viability interventions substantially, and it failed the lap-time target.

**The experimental option and unit test were fully removed, original `mpc_controller_node_impl.hpp` and `mpc_controller_recovery.cpp` restored byte-for-byte from private pre-test snapshots, rebuilt with the default portable numerical flags, and 214/214 tests passed.** There is no unqualified fine taper logic in production C++ now. Preserve the unmodified validated practice overlay.

## Targeted terminal behavior study

Independent `terminal_multiplier` 3.0→3.5 test `live_runs/practice_ftz_terminalstudy_terminal3p5_20261031_01`: complete ten scored laps, **49.328 s**, six recoveries, max callback 18.207 ms, zero misses/collisions. The final six scored laps were 4.879/4.903/4.902/4.902/4.903/4.903 s; two early scored laps took 5.051 and 5.027 s with recoveries. This suggests terminal objective may help *steady-state* tracking but needs repeatability validation; no promotion yet. Independent steering-target and steering-trust screening follows.


## Terminal/steering-target/trust individual trials and disqualification

All trials were performed on the **restored original controller**, with exact PhysX source-model validation and no safety-limit changes. Independent single-variable controls:

| Change vs practice baseline | Ten-lap time | Recoveries | Max callback | Result |
|---|---:|---:|---:|---|
| Terminal multiplier 3.0→3.5 | 49.328 s | 6 | 18.207 ms | 10/10 clean; first trial only |
| Steering-target penalty 5.0→4.5 | 49.421 s | 5 | 18.105 ms | 10/10 clean; first trial only |
| Steering SQP trust 0.040→0.035 rad | 49.293 s | 2 | 17.823 ms | 10/10 clean **first trial only** |

**The steering trust variant failed immediate repeatability:** the second independent race `practice_ftz_terminalstudy_trust0035_20261031_02` stopped after **one scored lap** (runner RC=5, no collision/deadline miss) due to `rejected_nonlinear_rollout` / `invalid_model@-1` while crossing the recurrent first hairpin. Saved solver diagnostics: QP reached iteration cap 150, primal residual 0.6577, attempted 150-iteration feasibility restoration also failed, and the certified short-prefix allowance had been exhausted. Strong caution: the narrower trust boundary can trap an already-difficult next-cycle feasibility problem. This is a failure of the hard 10-lap completion criterion regardless of the preceding 49.293 s race; **do not promote trust0.035**. Full failing controller log/ROS bag saved in its run directory. The immediate trust repeated gate automatically stopped on this failure.

The terminal multiplier=3.5 is separately undergoing full repeated Unity practice qualification, results pending. Never claim one fast 49.328 s run proves 5/5 sub49.5. Do not mix this parameter with discarded finer correction bank, narrow 0.035 steering trust or failed standalone weight changes without another complete gated experiment.


## Final repeatability findings — terminal cost, trust, and ADMM cap

Five independent ten-scored-lap races were completed using only `terminal_multiplier=3.5` (steering trust retained at 0.04 rad, all original safety gates):

| Race | Time (s) | Accepted recoveries | Max callback (ms) |
|---|---:|---:|---:|
| 01 | 49.328 | 6 | 18.207 |
| 02 | 49.541 | 8 | 18.367 |
| 03 | 49.586 | 8 | 18.362 |
| 04 | 49.546 | 0 | 18.268 |
| 05 | 49.467 | 9 | 18.924 |

**All five completed safely:** 50/50 scored laps, zero collisions and zero missed 20 ms publication deadlines; five-run mean **49.494 s** but only **2/5 individual race times below 49.5 s**. This is inferior to the original fast profile (mean 49.484 s, 3/5 below 49.5 s). Do not promote the stronger terminal cost despite its isolated clean steady-state sequence of 4.903 s laps. Artifact `live_runs/mpc_collision_margin_experiment_20261030/ftz_terminal3p5_full_five_race_summary_20261031.json`.

An independent adaptive stage-local trust-expansion experiment, implemented concurrently in the same checkout, completed an initial practice race in **49.625 s**, 4 recoveries, worst callback 18.208 ms, zero deadline misses, and was reverted with the original 214 C++ tests passing. This one-shot is not a performance promotion. It is distinct from the failed globally tightened 0.035-rad trust experiment.

**More ADMM iterations did not pass the first safety/completion gate.** The original fast policy with only `max_solver_iterations=180` (baseline 150, runtime confirmed through ROS log) was tested in `live_runs/practice_ftz_solveriter180_20261031_01`. The trial stopped in the warm-up with **0 scored laps**, `rejected_nonlinear_rollout` at physical corridor stage 0, four accepted recoveries, maximum callback 18.294 ms, **zero collision and zero publication deadline misses**. The controller correctly failed closed. Because this is only one run and another proposed path can be numerically different, the observation alone does not prove the iteration cap caused the rejection; however it **unambiguously fails qualification**, and increasing iterations unconditionally is rejected. No C++ code changed for this weight/iteration cap experiment. The runtime remains configured with the previous 150-iteration baseline.

### Decisions and more consequential next engineering direction

- **Retain the five-race clean scoped-Riccati FTZ implementation and original fast practice overlay** `practice_guarded_prefixseed_20261030_01.yaml` as the benchmark (49.484 s average, 3/5 individually below 49.5 s, worst callback 18.254 ms). Do not enable failed overrides in any production YAML.
- **Do not simply spend CPU on more ADMM iterations**, shrink steering trust globally, or apply gentler fixed steering perturbations. These did not qualify in the saved full Unity race tests and can make recovery worse or invalidate the full-horizon plan.
- The recurrent stage 23–29 viability failures near track station 2–3 m should be addressed *inside the optimization* with source-plant-verified terminal continuation and a consistent warm start for the post-recovery SQP. Preserve the complete 750 ms horizon, original hard footprint/corridor, per-candidate exact validation and 20 ms publication gate. Consider measuring whether existing `prefer_terminal_continuable_refinement` changes R2 selection on the saved hard states, then limit live trials to cases with exact physical continuity and timing headroom.
- The worst slow laps do not always include recovery. Alongside post-recovery recovery-state numerical work, trace lateral/heading/velocity tracking and candidate stage-merit at the precise track station, rather than tuning weights based solely on ten-lap totals. Keep 5 clean ten-lap races **each** under 49.5 seconds as the acceptance gate, then target below 49.0 seconds.

This audit deliberately reports negative experiments and the fact that **5/5 individually below 49.5 s has not yet been achieved**. It is not legitimate to interpret a five-run mean below 49.5 seconds as proof of that strict goal.
