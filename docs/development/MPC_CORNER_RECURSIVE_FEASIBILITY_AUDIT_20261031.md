# MPC practice-corner recursive-feasibility audit — 2026-10-31

## Targets and qualified control

All comparisons are closed-loop Unity/PhysX practice-map runs with full native 30-stage × 25 ms source rollout, 200 Hz controller, physical swept-world/corridor validation and hard 20 ms publication gate. A clean ten-scored-lap race (plus warmup lap) has no collision or missed deadline. The additional goal is **5/5 independent clean races under 49.5 s**, eventually **under 49.0 s**; a single fast race is never sufficient.

Baseline fast profile: `live_runs/mpc_collision_margin_experiment_20261030/practice_guarded_prefixseed_20261030_01.yaml` applied after `f1tenth_mpc/config/mpc_practice_fastline.yaml`, with native scoped Riccati FTZ/DAZ. Five independent ten-lap races `live_runs/practice_ftz_previous_fivegate_config_20261031_01`–`_05`: 49.370, 49.350, 49.253, 49.854, 49.594 s. All five physically safe with zero deadline misses, **3/5 under 49.5**, mean 49.484 s. Actual effective practice weights: lateral 5000, heading 1700, speed 45, steering-command 4.5, steering-slew 1500, throttle-slew 50; steering SQP trust radius 0.040 rad, throttle trust 0.050; physical corridor margin 0.020 m. Competition YAML is NOT the practice base.

## Empirical corner failure modes

In the 50 baseline scored laps, 16 laps <=4.92 s show mean lateral/heading/speed RMSE 0.033 m / 0.028 rad / 0.129 m/s; 10 laps >=5.00 s show 0.071 m / 0.051 rad / 0.237 m/s. Six of ten slow laps had NO accepted full-horizon viability recovery. The local raceline length is ~26.593 m; `progress_m` in diagnostics is cumulative, so reduce modulo length. Most recovery clusters occur near station **1.6–3.2 m**, with further episodes near 7–8 and ~15 m.

Run04 scored lap9 (5.177 s) had three recoveries near station 2.26–3.01 m. Compared to an ordinary 4.953 s lap at local station 1.5–3.5 m, speed 3.41 vs 4.04 m/s; in the following 3.5–6 m acceleration zone, 5.27 vs 6.26 m/s with 1.03 m/s mean underspeed. The disturbance propagates beyond the corner even after tracking recovers.

Across five baseline races, 21 accepted recoveries show median consecutive published steering jump **0.092 rad** versus **0.008 rad** for 7,344 accepted-line-search commands. The existing controller's tapered candidate bank often selects +/-0.12 rad and +/-0.08 throttle corrections. Most recoveries follow exact nonlinear corridor failure around stages 23–29, so the issue is commonly **future full-horizon feasibility**, not an immediate wall collision. The first callback after recovery has median ADMM iteration count 150 vs overall median 22, consistent with costly recovery-induced relinearization and cold numerical state; never retain incompatible ADMM multipliers to fix this.

### Failure evidence: shortening steering trust and increasing iteration cap

`practice_ftz_terminalstudy_trust0035_20261031_01`: reduced steering trust 0.04→0.035, ten scored laps in 49.293 s, 0 deadline misses. **Its immediate second independent race** `_02` aborted after one scored lap with `rejected_nonlinear_rollout`, `invalid_model@-1` and attempted 150-iteration feasibility restoration. The QP hit iteration cap, primal residual 0.658; no safe full-horizon continuation remained. Source path and full failing diagnostic preserved. The first fast run was **disqualified**.

`practice_ftz_solveriter180_20261031_01`: original fast profile with ADMM cap 150→180, production code unchanged, aborted in warmup, **zero completed scored laps**, no collision/deadline miss. The rejected nonlinear prediction was `corridor@0` (first source-plant transition, before the car could be recovered using the tried candidates). Two SQP passes and infeasibility restoration failed; aggregate 315 ADMM iterations, 0.046 primal residual, ~14.013 ms callback. In the prior certified plan two short prefix actions were used and `prior_plan_actions_remaining=0`; no full-horizon feasible successor was available. The approximate offending state had e_y=-0.060 m and e_psi=0.187 rad, path curvature=-0.921 1/m; first predicted transition reached e_y=-0.046 m / e_psi=0.225 rad and exceeded the footprint corridor lower bound by ~0.017 m. `failure_stage=0` is the source **transition index**, not a check that the immutable state x0 is in violation. No tested command achieved a certifiable continuation. Do not solve this by relaxing hard corridor or increasing the short-prefix allowance.

These failures show that some subsequent states are **locally unreachable from the chosen nominal/control queue** even though an earlier short continuation was physically valid. This calls for recursive-feasibility-aware candidate selection, stage-local QP feasibility and an uncertainty buffer for successive measurement/actuation epochs; the existing 750 ms native rollout only certifies its predicted initial state and control history, not all future replanning disturbances.

## Experimental results: no changes promoted

All following are first-screened as independent single-variable or algorithm experiments; every C++ experiment was opt-in and disabled by default. After regression it was **fully removed**, source restored byte-for-byte and 214/214 ROS C++ tests passed. These outcomes MUST NOT be described as improvements:

| Experiment | 10 scored laps | Recoveries | Max callback | Decision |
|---|---:|---:|---:|---|
| Speed weight 45→47 | 49.653 s | 4 | 18.145 ms | reject |
| Lateral 5000→5300 | 49.501 s | 1 | 18.087 ms | reject |
| Heading 1700→1750 | 50.197 s | 8 | 18.659 ms | reject |
| Steering-command 4.5→4.3 | 49.449 s | 6 | 18.511 ms | no repeatability proof |
| Steering slew 1500→1450 | 49.545 s | 7 | 18.218 ms | reject |
| Throttle slew 50→45 | 49.448 s | 11 | 18.707 ms | reject |
| Fine source-validated recovery taper (smaller steering +/-0.08 rad) | 49.704 s | 17 | 18.818 ms | regressed; code removed |
| Adaptive stage-specific QP steering trust (up to 0.06 rad on predicted misalignment in high curvature) | 49.625 s | 4 | 18.208 ms | regressed; code removed |
| Exact-footprint geometry-aware soft wall-approach penalty (0.05 m, 2000 objective weight) | 49.564 s | 7 | 18.202 ms | regressed; code removed |
| Geometric clearance-reference reserve 0.03 m, using existing code | 49.539 s | 4 | 18.107 ms | not promoted |

Terminal multiplier 3.0→3.5 (baseline QP, five complete races): 49.328, 49.541, 49.586, 49.546, 49.467 s; **2/5 under 49.5**, mean ~49.494 s; 0 collisions or missed deadlines, worst callback 18.924 ms. Stronger terminal cost is not repeatably faster; leave baseline 3.0. Steering-target penalty 5.0→4.5 produced 49.421 s, five recoveries on only one run; no promotion. Static steering trust 0.035 and ADMM 180 suffered complete failures as documented above.

The earlier steering slew=1400 series produced five clean races 49.299/49.739/49.654/49.607/49.563 s, only 1/5 under target. Never combine unqualified single-variable changes as though gains were additive.

## Correct next engineering work — avoid re-running losing parameter sweeps

1. **Source-valid multi-tick viability rather than just short-prefix feasibility.** In the actual `try_publish_quality_aware_prefix` / certified-plan handoff, track the next 2–4 control epochs, in-flight actuator queue and a certifiable *full-horizon continuation*. Avoid accepting a short prefix whose successor is likely to be unsolvable. A new candidate must pass the same exact 750 ms PhysX/corridor/terminal limits as production; one/two short prefixes alone do NOT establish recursive feasibility.
2. **Escape a bad SQP nominal before the current first transition becomes unavoidable.** Investigate a bounded, source-witness/geometry-seeded feasibility QP or alternate local linearization, preferably reusing the existing first physical Jacobian and under the existing deadline reserve. Current QP often runs to 150 iterations with a large terminal/state residual after recovery; merely adding iterations or widening static trust did not help. Test identical failing live states OFFLINE before new closed-loop races.
3. **Robustness to next measurement epoch** rather than static wall penalty. Compare accepted predicted step 1 pose/slack with actual odometry/map-corrected pose on the next callback. Derive an empirical disturbance bound (meters and radians, conditional on turn/slip/sensor age). If justified, build a state-dependent uncertainty tube/soft corridor reserve into the optimization while preserving the existing hard wall constraint. This guards the missing *replanning stability* property without slowing all straights.
4. **Track-section speed and tracking analysis** must be per-local-station, not just lap aggregates. Quantify early-hairpin e_y, e_psi, commanded slew, QP step damping, 30-stage corridor slack and the post-corner acceleration deficit. Do not optimize for lower recovery count alone; six of ten slow baseline laps have no recoveries.
5. **Verification gates:** unit tests; exact-off parity/replay on difficult first-hairpin saved states; zero collisions, zero deadline misses and *five complete ten-scored-lap races all under 49.5 s*; then competition-map safety. No accepted 20 ms deadline misses and no physics/geometry shortcuts. Stop and remove any regressing option; preserve unrelated optimizer working-tree changes.

Follow-up raw evidence: `live_runs/practice_ftz_*_20261031_01` matching variants, `live_runs/mpc_collision_margin_experiment_20261030/{ftz_weight_screen_20261031.json,ftz_terminalstudy_20261031.json,ftz_terminal3p5_four_more_summary_20261031.json}` and `docs/development/MPC_PRACTICE_SLOW_LAP_CONVERGENCE_20261031.md`. An independent feedback-stabilized nonlinear rollout flag is under separate assessment.

## Feedback-stabilized nonlinear source rollout and actual reliability gate

The *existing* `feedback_rollout: true` option reuses Riccati LTV gains to correct the candidate controls using mismatch between the exact PhysX candidate rollout and the QP's linear trajectory. Existing hard source/corridor/deadline gates remain authoritative. It is not an open-loop friction-model shortcut. Tested using practice overlay `live_runs/mpc_collision_margin_experiment_20261030/practice_ftz_feedback_rollout_20261031.yaml`, with no other tuning relative to the qualified fast baseline:

| Independent trial | Official 10-scored-lap time | Accepted recovery cycles | Maximum callback | Result |
|---|---:|---:|---:|---|
| 01 | 49.389 s | 5 | 17.879 ms | complete, safe, below 49.5 |
| 02 | **49.226 s** | 1 | 17.932 ms | complete, safe, below 49.5, fastest new full-race time |
| 03 | 49.283 s | 0 | 18.213 ms | complete, safe, below 49.5 |
| 04 | 49.277 s | 0 | 18.204 ms | complete, safe, below 49.5 |
| 05 | 1 scored lap (4.936 s) | 4 | 18.068 ms | **MPC fault `rejected_nonlinear_rollout`, `corridor@12`, unqualified** |

**4/5 completed; no collision or publication deadline miss; NOT QUALIFIED FOR PRODUCTION** because the fifth run aborted. Baseline remained untouched, `feedback_rollout` default OFF.

Race05 failure was localized to raceline station ~3.36 m, already experiencing e_y=-0.0038 m, heading error e_psi=-0.403 rad, speed3.634m/s, lateral speed .902m/s; latest full nonlinear candidate failed at transition stage12 with e_y=-.472m, e_psi=-1.453rad. The solver hit 150 ADMM iterations, primal residual1.441, dual residual0.298; one feasibility restoration failed; two short prior-plan actions had been consumed and no full-horizon suffix remained. In the preceding ~0.36 s, steering commands swung from -.444 to -.211 to -.033 to +.066 to -.306 to +.234 to -.242rad, while near-horizon certified clearance briefly fell to ~0.0165m. This suggests aggressive Riccati feedback corrections plus local source-state disturbances can steer into unrecoverable corner-entry states. It does not prove gain magnitude is the unique cause; a controlled damping experiment follows.

For this work, **do not count the first four successes as a five-run pass**. Furthermore the strong full-gain feedback first race contained a 5.178 s outlier lap with accepted recoveries near station2.5m; race03 contained a 5.079 s lap with *no* accepted recovery, dominated by lateral/heading error at station0–3.5m. Thus the mechanism helps average pace but does not yet ensure consistent 4.9s per lap.

## New retained offline corner attribution tool

Added `tools/analyze_mpc_corner_viability.py`, driven solely by post-run `mpc_run_report.json` and `mpc_cycle_detail_diagnostics.jsonl`. It computes ten scored-lap sectional speed deficit vs witness, lateral/heading RMSE, nonlinear four-backtrack counts, steering jump events >.08rad and accepted recovery stage/reason, locating repeated episodes at `progress_m % track_length_m`. Uses **bag diagnostic timestamps** for per-lap attribution; those totals can differ from the Unity official scoreboard and must not replace authoritative official `tools/audit_mpc_race_quality.py` times. Unit test `tools/test_analyze_mpc_corner_viability.py` covers sector wrap and independent lap command continuity; `python3 -m unittest -q tools.test_analyze_mpc_corner_viability tools.test_export_mpc_run_report` completed 18 tests without failure. Example:

```bash
python3 tools/analyze_mpc_corner_viability.py \
  live_runs/practice_ftz_feedback_rollout_20261031_02 \
  --track-length-m 26.5929520925 \
  --output live_runs/mpc_collision_margin_experiment_20261030/feedback_run02_corner_viability.json
```

No new instrumentation was inserted into the 200 Hz controller. The optimizer's unrelated dirty working tree was not reset or committed.

## Closed-loop Riccati-feedback follow-up: full five-race gates

The already-implemented `feedback_rollout=true` evaluates an explicit projected-input LTV Riccati feedback policy during the 750ms exact native PhysX nonlinear rollout. Every stage's actual feedback-corrected command is clamped to the unchanged actuator/QP stage bounds, and the full validated command sequence is used for publication and warm-start; the source-plant corridor gate and 20ms publication deadline remain authoritative. No claim of robust performance follows from one fast trial.

**Full gain 1.0** (`live_runs/practice_ftz_feedback_rollout_20261031_0{1..5}`), five independent 11-lap Unity executions (warmup+10 scored): scored race times 49.389, 49.226, 49.283, 49.277, then **race 5 aborted after one scored lap** (`rejected_nonlinear_rollout` corridor@12). Zero deadline misses or collision events in all starts; the 4/5 completion result **disqualifies full gain 1.0**. At the failing state, track station 3.360m, vehicle e_y=-0.0038m, heading error=-0.403rad, u=3.634m/s, lateral velocity=+0.902m/s, yaw rate=-2.739rad/s; after two prior-plan commands the exact full source rollout rejected at stage 12. Feasibility restoration attempted 150 ADMM iterations but reached 1.441 input-primal residual. No safe full continuation could be published. Source evidence `live_runs/practice_ftz_feedback_rollout_20261031_05/mpc_cycle_detail_diagnostics.jsonl`; full reporting preserved.

An opt-in (default `feedback_rollout=false`) physical feedback gain scalar 0..1 was added to existing policy evaluation (`feedback_rollout_gain_scale`, default 1.0 when enabled). It scales only Riccati state-error corrections; base open-loop candidate and QP, state/command limits, 750ms native rollout, hard geometry, line search, and publication deadline remain unchanged. Code compiled, 214 ROS C++ tests passed and full practice runs used exact ROS parameter overlays. Preserve this until a fully qualified configuration exists; **the gain scalar is not itself proof of robustness**.

**Damped gain 0.75** (`live_runs/practice_ftz_feedback_gain075_20261031_0{1..5}`) completed all five independent ten-scored-lap races (50/50 laps, 0 collisions, **0 missed deadlines**):

| Race | Time (s) | Accepted recoveries | Max callback (ms) |
|---|---:|---:|---:|
| 01 | 49.355 | 0 | 18.186 |
| 02 | 49.653 | 4 | 18.252 |
| 03 | 49.169 | 0 | 17.974 |
| 04 | 49.211 | 2 | 18.089 |
| 05 | 49.207 | 1 | 18.280 |

Mean **49.319s** (vs previous static fast profile mean 49.484s); best **49.169s**; **4/5, not 5/5, races below 49.5s**. This passes full-distance safety but *not* the stricter speed-consistency gate; do not silently claim full qualification. Gain 0.75 remains an **experimental candidate**, not a forced production setting.

Offline position-normalized diagnostics built in `tools/analyze_mpc_corner_viability.py` (corresponding one-unit-test `tools/test_analyze_mpc_corner_viability.py`, 18/18 combined report+analyzer tests): saves per-scored-lap, per-track-sector speed deficits, e_y/e_psi RMSE, accepted recoveries, four-backtrack occurrences and steering-command jumps without adding any control-callback work. Five independent damped runs had three scored laps >=5.0s. Trial 02's 49.653s was dominated by scored-lap1 **5.251s** (one recovery at station 13.950m, ~1.129m/s speed deficit in the 15–21m sector), plus lap10 **5.079s** (no accepted recovery, ~0.590m/s deficit at station 1.5–3.5m). The failure mode is therefore a *multiregion tracking/convergence tail*, not just the known 2–3m hairpin fallback. Preserve failures and prioritize reducing worst-lap speed deficits as well as mean time.

An isolated one-variable **experimental heading penalty screen** on exactly the 0.75 gain profile lowers heading weight from 1700 to 1600 while keeping lateral at 5000 and every native physics/safety constraint unchanged. File `live_runs/mpc_collision_margin_experiment_20261030/practice_ftz_feedback_gain075_heading1600_20261031.yaml`. **No positive or negative result should be inferred before the complete closed-loop race and, if promising, the five-repeat gate.** Target after reliable <49.5s is consistently <49.0s across independent ten-scored-lap races; the current best 49.169s requires ~0.17s improvement across ten laps, but tail reduction is as important as nominal speed.

## Three independent weight screens on gain 0.75: no promotion

After all five damped-feedback races completed without collision/deadline miss (but only 4/5 <49.5s), the user requested eventual sub-49s tuning. Three one-variable, **non-cumulative** practice screen races tested weights against the unmodified gain-0.75 profile. Every screen used the exact same 30×25ms native PhysX rollout, hard footprint clearance, two RTI passes, ADMM cap and recovery policy:

| Single change | Ten scored laps | Accepted recoveries | Max callback | Decision |
|---|---:|---:|---:|---|
| Heading weight 1700→1600, lateral stays 5000 | 49.419s | 1 | 18.332ms | No improvement evidenced |
| Lateral weight 5000→4800, heading stays 1700 | 49.465s | 0 | 18.145ms | No improvement evidenced |
| Heading weight 1700→1500, lateral stays 5000 | 49.443s | 2 | 17.787ms | No improvement evidenced |

All three were individually clean with zero missed deadlines, but **none provided evidence for speed promotion** relative to the fixed gain-0.75 five-race 49.319s mean and 49.169s best. Single screens are diagnostic, not statistical proof that a weight is harmful; do not combine the three or claim that 49.419/49.465/49.443 predict a new five-race mean. All changes existed only in isolated `live_runs/mpc_collision_margin_experiment_20261030/practice_ftz_feedback_gain075_{heading1600,lateral4800,heading1500}_20261031.yaml` overlays; the actual production practice profile and controller parameter defaults remain unchanged. The saved full bag/ROS diagnostics for each run are available in `live_runs/practice_ftz_gain075_{heading1600,lateral4800,heading1500}_20261031_01/`.

**Current decision**: preserve the 0.75 gain as an opt-in candidate (not an automatically enabled default), retain the tested original 5000/1700 weights, and do not promote any single-screen lower weights. Speed target 49.0s is NOT met; next engineering work should correlate the exact physical corner-entry state/queued steering and local speed debt to initial nominal construction and recursively viable control sequence across updates. Only after a safe, repeatable viable controller is established should further weights be tuned and subjected to five independent full-distance races.
