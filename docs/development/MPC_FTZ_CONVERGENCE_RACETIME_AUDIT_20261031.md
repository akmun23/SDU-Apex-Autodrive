# MPC convergence / native rollout budget and practice race-time audit

Date: 2026-10-31. Production controller remains the previously validated scoped-FTZ Riccati solver; no physical model, C++ MPC controller, safety margins, terminal viability or fallback algorithms were changed in this audit.

## Why more computation must be useful

- The normal accepted RTI source-model rollout is **not** repeated by the ROS publication path. A complete validation of a different control vector is necessary; almost all line-search candidates change their first control, so their 750 ms exact PhysX trajectories do not have reusable physical suffixes. Node recovery paths do additional fresh-current-state validation only when needed.
- On five new full-distance practice races, **6,119 R2 refinements were attempted and 5,203 R2 plans were selected** (85.03%). A one-pass simplification would usually discard the better certified solution. Keep R2, the exact source rollout, hard body/corridor checks and current publication gate.
- **2,852 cycles used all four nonlinear backtracks** in those same races. Trial-order changes would choose different, potentially smaller controls and cannot be justified as numerically identical caching. The previous alpha repeated in only ~52% of adjacent accepted line-search episodes in a representative bag. An adaptive proposal policy is possible but must remain experimental and validate every proposed action exactly.

## One-variable closed-loop practice sweep

All profiles retain the original 30 × 25 ms source horizon. **Important effective-parameter correction:** `run_mpc_practice_trial.sh` loads `mpc_practice_fastline.yaml` first, then the experiment overlay. The actual baseline trust region was **0.04 rad steering / 0.05 throttle**, the steering slew weight was **1500**, and the physical corridor margin was **0.02 m** (not the 0.08/2000/0.05 competition-YAML values). The `0.06` trust experiment **widened** the steering region from 0.04 to 0.06 rad, and the `1600` slew experiment **increased** the penalty from 1500 to 1600. Native rollout and all safety checks were always enabled.

| Variant vs fast practice profile | Ten scored laps | Outcome | Recoveries | Max callback |
|---|---:|---|---:|---:|
| Previously verified prefix-seed profile; FTZ retained | 49.370 s | clean | 8 | 18.028 ms |
| Steering SQP trust 0.04 → 0.06 rad | 49.843 s | clean, slower | 10 | 18.043 ms |
| Heading tracking weight 1700 → 1650 | 49.618 s | clean, slower | 5 | 18.036 ms |
| Steering slew weight 1500 → 1600 | incomplete (9/10 scored laps) | **rejected nonlinear corridor at stage 15** | 7 | 18.477 ms |

All trial reports and overlays are saved under `live_runs/mpc_collision_margin_experiment_20261030/ftz_singlevar_convergence_sweep_20261031.json` and `practice_ftz_tune_*.yaml`. The failed candidate and slower parameters were **not retained**. The independent baseline profile is `live_runs/mpc_collision_margin_experiment_20261030/practice_guarded_prefixseed_20261030_01.yaml`. The previous, newer guarded practice profile averaged 50.072 s over five races, also with no collisions or publication deadline misses; the fast profile recovers ~0.588 s over ten scored laps in the new five-race sample.

## Five independent full-distance repeats of fast profile on same FTZ build

| Run | Ten scored lap time | Accepted recovery cycles | Maximum callback |
|---|---:|---:|---:|
| 01 | 49.370 s | 8 | 18.028 ms |
| 02 | 49.350 s | 4 | 18.161 ms |
| 03 | **49.253 s** | 1 | 18.012 ms |
| 04 | 49.854 s | 7 | 18.254 ms |
| 05 | 49.594 s | 3 | 17.993 ms |

**Qualified for safe practice completion:** 5/5 complete ten-scored-lap races (50 scored laps), zero collisions and zero missed 20 ms publication deadlines; worst callback 18.254 ms, worst ADMM 7.999 ms. **Not qualified for every run faster than 49.5 s:** only 3/5 meet that time individually. Five-run mean 49.4842 s, fastest 49.253 s, slowest 49.854 s. Earlier five-race performance under a different historical build is not a replacement for these new-run results.

The two slowest races had more accepted recoveries; in run 04, multiple recovery actions clustered near 268 m of cumulative progress and a scored lap took approximately 5.168 s. This is a correlation rather than proof of causation. Future experiments should identify the precise near-corridor state, compare successful R2 and recovery seed rollouts, and improve subsequent exact-valid continuation without reducing clearance or hard constraint checks. The IROS competition map remains separately unqualified with this practice-specific profile.

## Source and evidence

- Full five-run aggregate including exact R2 selected and ADMM figures: `live_runs/mpc_collision_margin_experiment_20261030/ftz_fastpolicy_five_race_aggregate_20261031.json`.
- Five full ROS bags and diagnostics: `live_runs/practice_ftz_previous_fivegate_config_20261031_01` through `_05`.
- Offline `tools/export_mpc_run_report.py` gained two exports for pre-existing solver telemetry (`refinement_full_horizon_selected_pass`, `refinement_nominal_build_us`). This corrects misleading zero-R2 counts in older CSV-based sweep reports; no 200 Hz controller callback work was added. The existing `line_search_trials` and `line_search_us` exports from the earlier nonlinear audit remain.
- C++ production binary/code and safety configuration were not modified for the new races. A previously validated package C++ suite contained 214 passing tests; offline report regression + fixture/provenance suite passes 39 tests and `git diff --check` passes. Do not commit preexisting unrelated optimizer changes in this dirty checkout.
