# FTZ MPC practice: sub-49.5 repeatability, recovery and convergence tuning

Date: 2026-10-31. Goal: **five independent complete, collision-free ten-scored-lap practice Unity races, EACH under 49.5 s, zero missed 20 ms publication deadlines**; next objective <49.0 s while retaining complete 30 ×25 ms native Unity/PhysX-derived rollout, hard body/corridor validation, two SQP passes and safe recovery. Never promote a single unusually fast run. Never use the off-line 4.8 s source witness as an achieved closed-loop lap time.

## Correct runtime configuration layering

`tools/run_mpc_practice_trial.sh` uses `f1tenth_mpc/config/mpc_practice_fastline.yaml` as the practice base, **not** `mpc_competition.yaml`, then applies experiment overlay from `SDU_APEX_MPC_PARAMETER_OVERLAY`. The checked baseline `live_runs/mpc_collision_margin_experiment_20261030/practice_guarded_prefixseed_20261030_01.yaml` therefore has actual steering trust **0.04 rad**, throttle trust **0.05**, steering slew **1500**, lateral weight **5000**, heading weight **1700**, speed weight **45**, steering-command weight **4.5**, physical practice wall margin **0.02 m**, and a full independent nonlinear check. An earlier report incorrectly compared alternatives against competition-YAML 0.08-rad trust, 2000 slew and 5-cm margin; those were **not** the practice starting values. The earlier audit is corrected. The IROS competition configuration/5-cm margin is separate and none of these experiments establishes IROS qualification.

## Existing baseline: safe but performance not five-of-five

FTZ and the original fast prefix-seed overlay: ten-lap times **49.370, 49.350, 49.253, 49.854, 49.594 s**, mean 49.4842 s, **5/5 clean** no collision/no missed publication deadlines, **3/5 below 49.5**, 23 accepted recovery actions, 18.254 ms worst callback and 7.999 ms worst ADMM. R2 refinement selected in **5203 of 6119 attempts (85%)**, so never skip R2 to save time. Up to four separate candidate nonlinear backtracks remain necessary because their control streams change immediately; do not reuse invalid future physical states. Evidence: `live_runs/mpc_collision_margin_experiment_20261030/ftz_fastpolicy_five_race_aggregate_20261031.json`.

Across **50 scored baseline laps**, per-lap count of ADMM iteration-cap stops correlated with slower laps (Pearson **r ≈ 0.801**) and deadline-limited ADMM stops also correlated (**r ≈ 0.514**). These are observational associations, not causal evidence: physically difficult corner states may both slow the car and challenge convergence. One 5.168-s lap contained 72 iteration-cap events and three source-certified rescue actions. Laps with zero and one recovery both averaged approximately 4.94 s; **clusters** of multiple rescues coincide with particularly bad laps. Recovery taper can alter steering by ~0.12 rad and throttle by ~0.08; this is source-validated emergency correction, not a free normal racing action. Even recovery-free laps can reach >5.08 s when lateral and heading tracking error increase.

## Completed independent Unity tests

All rows use the same native model, exact 750 ms constraints, and full two-pass SQP with no unvalidated publication. Max callback is the observed publication-gate time, **not** a theoretical Linux worst-case bound.

| Change from baseline | Ten scored laps (s) | Accepted rescues | Worst callback (ms) | Result |
|---|---:|---:|---:|---|
| Speed weight 45 → 50 | 49.477 | 2 | 17.985 | Clean; insufficient evidence of speed gain |
| Heading weight 1700 → 1850 | only 5 laps | 8 | 18.816 | **Rejected** first-stage exact corridor infeasibility |
| Lateral weight 5000 → 5500 | 49.442 | 7 | 18.652 | Clean; no demonstrated gain |
| Terminal extension viability screening | 49.667 | 7 | 18.486 | Clean but slower |
| More ADMM iterations 150 → 180 | 49.336 | 3 | 18.192 | Clean, promising, **one-run only** |
| Steering slew 1500 → 1450 *and* 180 ADMM iterations | 49.627 | 13 | 18.226 | Clean but slower; do not combine blindly |
| Recovery search 8 → 12 exact-valid candidates | 50.024 | 1 | 18.256 | Clean but substantially slower; rescue count is not lap-time objective |
| Steering slew 1500 → 1400 | 49.299 first run | 3 | 18.253 | **Five-run series now complete: 5/5 safe but just 1/5 below 49.5 s** |

The 180-iteration standalone trial had 202 iteration-cap stops vs 284 in one historical 150-iteration race, but maximum ADMM increased to 8.576 ms and the outcome requires repeat qualification. An increased recovery trial budget had **395 capped solves and 655 four-backtrack cycles**, explaining why one rescue does not necessarily mean fast tracking. These results are not paired identical-state counterfactuals.

## Five Unity races with steering slew weight 1450

| Run | Ten-lap time | Rescue actions | Worst callback |
|---|---:|---:|---:|
| 01 | 49.230 s | 2 | 18.283 ms |
| 02 | 49.553 s | 5 | 18.279 ms |
| 03 | 49.440 s | 7 | 18.303 ms |
| 04 | 49.349 s | 1 | 18.040 ms |
| 05 | 49.351 s | 3 | 18.134 ms |

**5/5 complete clean races, 4/5 below 49.5 s**, mean **49.3846 s** (about 0.10 s faster than five-run 1500 baseline), best **49.230 s**. This variant is a useful candidate but has **not** achieved strict 5/5 race-time goal, so keep original fast overlay as protected reference; no production YAML/solver/source-model changes. The apparent first-run gain correlated with lower lateral RMSE, heading RMSE and number of four-backtrack cycles, but the second run remained 49.553 s.

## Status and next experimental gates

1. **1400 steering-slew qualification completed and rejected for racing speed:** five clean ten-lap runs 49.299 / 49.739 / 49.654 / 49.607 / 49.563 s, mean **49.572 s**, **1/5** below 49.5, despite some 4.862-4.867 s individual scored laps. Worst callback 18.457 ms, zero collisions or missed deadlines. Run 03 49.654 s with only one recovery but lateral RMSE 0.117 m, 324 solver iteration-cap stops, 65 deadline-limited ADMM stops; further proof that lower recovery count alone does not guarantee fast lap time. Aggregate `live_runs/mpc_collision_margin_experiment_20261030/ftz_slew1400_five_run_aggregate_20261031.json`. Keep 1450 the stronger of the tested tuning alternatives on five-run mean and 4/5 time threshold, but **neither** achieves strict 5/5.
2. The standalone **180-iteration ADMM solver** originally completed one 49.336 s race without missed deadlines and reduced iteration-cap stops, so additional four-run repeatability qualification was launched. Continue to prioritize **near-corridor continuation and SQP numerical conditioning** rather than merely raising more weights, reducing wall margins, or increasing recovery-count limit. Evaluate R2 selected full horizons and speed/heading errors at each slow lap; avoid resetting warm state on unneeded recovery. New recovery proposals must remain physically full-horizon valid and have objectively lower speed cost, with the previous certified escape sequence retained.
3. Sub-49.0 s means <4.9 s per lap across all ten scored laps. Isolated **4.862–4.867 s** laps are now observed even with the 1400 trial, but no credible repeatable sub-49.0 s full-race qualification exists. A 4.8-s native source-reference lap remains a target, not demonstrated closed-loop performance. The 4.8 s offline reference is not yet an achieved full closed-loop guarantee.
4. After a new single candidate demonstrates a repeatable improvement on five independent practice races, perform **separate IROS competition-map qualification**. Never generalize practice-specific 2-cm margins or dynamics/track speed from this alone.

All experiment overlays, runner logs, sealed run manifests, full ROS bags and exported CSV/JSON diagnostics are under `live_runs/mpc_collision_margin_experiment_20261030/` and `live_runs/practice_ftz_*_20261031_*`. Compacted results index: `live_runs/mpc_collision_margin_experiment_20261030/ftz_practice_new_tuning_trials_20261031.json`. The working tree was already dirty from unrelated C++ structural MPC and IPOPT work; no unqualified C++ or controller parameter changes were made in the currently deployed production configuration.
