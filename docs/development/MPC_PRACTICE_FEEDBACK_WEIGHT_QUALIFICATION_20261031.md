# Practice MPC: first 5/5 sub-49.5 s qualification — 2026-10-31

## Qualification and production candidate

**The 5/5 individual race-time gate is satisfied** on the exact Unity/PhysX practice track with the same previously certified 750 ms horizon, two RTI SQP passes, 1 ms native source-plant integration, 200 Hz controller, conservative world-map/footprint/corridor checks, physical constraints, and hard 20 ms publication deadline. Nothing is disabled or relaxed to pass the gate.

Load `f1tenth_mpc/config/mpc_practice_fastline.yaml` followed by the new trackable overlay `f1tenth_mpc/config/mpc_practice_feedback_qualified_20261031.yaml`. It exactly matches the tested overlay `live_runs/mpc_collision_margin_experiment_20261030/practice_ftz_gain075_screen_combined4500_1450_20261031.yaml`. The only changes from the previous fastest FTZ/DAZ practice profile are **Riccati source-validated feedback rollout enabled with gain scale 0.75, lateral state tracking weight 4500 (from 5000), and heading state tracking weight 1450 (from 1700)**. Steering-command weight 4.5, steering slew 1500, speed weight 45, ADMM iteration cap 150, steering SQP trust 0.04 rad, throttle trust 0.05, margin 0.02 m and original fallback remain unchanged.

| Race ID suffix | Ten scored laps, seconds | Accepted viability recoveries | Max callback, ms | Collisions or 20ms deadline misses |
|---|---:|---:|---:|---|
| `_01` | 49.162 | 0 | 18.121 | none |
| `_02` | 49.091 | 0 | 18.126 | none |
| `_03` | 49.175 | 1 | 18.088 | none |
| `_04` | 49.291 | 0 | 18.134 | none |
| `_05` | 49.092 | 0 | 17.964 | none |

**Result:** 50 of 50 scored laps completed; **5/5 ten-lap race times strictly below 49.5 s**, mean **49.1622 s**, fastest **49.0910 s**, slowest **49.2910 s**, total **one** accepted viability recovery, zero collisions, zero publication deadline misses, worst callback **18.1345 ms**. Compared with historical fastest profile (five-race mean 49.484 s, 3/5 sub-49.5), improvement is **0.322 s per ten scored laps** and better repeatability. Compared with source-validated feedback gain 0.75 and the old lateral/heading weights (49.319 s five-race mean, 4/5 sub-49.5), improvement is **0.157 s per ten scored laps**. Individual race starts remain independent; no tests were concatenated or cherry-picked after the fact.

Canonical raw evidence: `live_runs/practice_ftz_gain075_screen_combined4500_1450_20261031_01` through `_05`, each containing ROS bag, run manifest, `mpc_run_report.json`, `mpc_cycle_detail.csv` and `mpc_cycle_detail_diagnostics.jsonl`. Machine-checked aggregate `live_runs/mpc_collision_margin_experiment_20261030/ftz_gain075_combined4500_1450_fivegate_20261031.json` and actual overlay. Exact `git diff --check` clean, and 18/18 offline corner-analysis/reporting Python tests passed at qualification. Existing 214 ROS C++ tests had passed on the Riccati gain-scaled controller binary. The IROS competition map has **not** been qualified using this practice-only overlay; retain the separate competition profile.

## Evidence driving the new weight choice

The user requested **reducing lateral and heading penalties** rather than focusing only on raw solver speed. On scored-lap diagnostic samples from the previous 49.169 s full race at gain 0.75, the mean scalar weights multiplied by squared *current* state errors were approximately lateral **11.219**, heading **1.581**, and speed **0.919**. Crucially, those figures exclude warm-up, where huge speed transients distorted earlier measurements. These are **not** the whole QP objective (they exclude input/slip/terminal and horizon costs), but they show substantial lateral tracking authority during actual racing. State units prevent comparing the bare numbers 5000, 1700, 45, etc. without their errors.

Single-variable preliminary screens at gain 0.75: lateral 5000→4800 **49.465s** (clean, zero recoveries), heading 1700→1600 **49.419s** (clean, one recovery), heading 1700→1500 **49.443s** (clean, two recoveries). Larger lateral 5000→4200 **49.242s**, zero recoveries; more aggressive heading 1700→1350 **aborted after three scored laps** with exact nonlinear `corridor@0`, four recoveries, no collision or deadline miss. The combined moderate weights 4500/1450 initially achieved 49.162s, then **passed four more full-distance repeats** as reported. Separately, feedback gain 0.60 on unchanged 5000/1700 weights achieved only 49.420s with two recoveries. Therefore the gain 0.75 and moderate combined weights are empirically stronger; do not overwrite the qualified values with any losing one-shot.

### Dynamic-state reliability and why unlimited weight reduction is unsafe

The heading-1350 first-hairpin failure at local track station **2.704 m** had current state e_y=0.0064m, e_psi=-0.101rad, u=3.868m/s, **lateral velocity v=0.649m/s, yaw rate r=-3.104rad/s**. Normal successful 2.6–2.9m passages on the gain-0.75 profile had **median lateral velocity v~0.026–0.027m/s** and yaw rate r~-1.75rad/s (different instants within the same segment; descriptive, not a causal model). It was too late after **two short prior-plan actions** and 150 QP iterations (`primal_residual~0.695`) to admit a footprint-valid first transition. Hard native source check failed closed. This motivates investigating the *existing* QP tracking penalties for physical lateral velocity and yaw (current defaults `weight_v=5`, `weight_r=7.8125`, both relative to native reference), and recursive feasibility of short fallback prefixes. It does **not** justify changing physical limits, skipping exact rollouts, or making every corner slower.

Future tests may evaluate slight increases in `weight_v` and `weight_r` independently, and a separately staged lateral4500→4200 change with heading fixed at 1450, but **those are not qualified** and must not be combined or promoted without full source-validated Unity race gates. Target next is repeatedly **below 49.0 s** (not merely a fastest single lap below 4.9 s) while preserving all five full-distance completions and the 20 ms publication threshold.

## Additional sub-49 s screens (not promoted)

The 5/5 qualified practice overlay 4500/1450/gain0.75 is preserved unchanged. Subsequent isolated full-distance tests evaluated further lateral reduction **with heading held at 1450**:

| Lateral weight | Ten-lap seconds | Accepted recoveries | Max callback (ms) | Result |
|---|---:|---:|---:|---|
| 4200 | 49.395 | 1 | 17.782 | clean, slower than qualified mean |
| 4000 | **49.071** | 1 | 17.936 | clean fastest individual race so far; repeatability unqualified |
| 3700 | 49.155 | 0 | 18.172 | clean, slower than 4000 |
| 3300 | 49.140 | 0 | 18.081 | clean, slower than 4000 |

These are not matched-seed A/B experiments; small one-run differences are not grounds for automatic production promotion. At 4000, all ten individual scored laps were below 4.953s. The sole accepted recovery occurred at station 15.137m and resulted from predicted corridor stage 24. At 3700, the controller achieved less speed undershoot (0.146 vs 0.152 m/s on the 4000 test), but lateral RMSE increased (0.0417 vs 0.0365m) and odometry travel was ~0.52m longer over the full 11-lap session. These observations suggest that the extra trajectory distance or disturbed corner tracking can offset a small speed gain; they do not establish causation across one independent trial each. At 3300, speed undershoot improved further to 0.131m/s but time was still 49.140s.

Raw records: `live_runs/practice_fixed_heading1450_lateral{4000,3700,3300}_20261031_01/`, `live_runs/practice_ftz_gain075_combined_lateral4200_20261031_01/` and the independent screen summary `live_runs/mpc_collision_margin_experiment_20261030/fixed_heading1450_weight_screen_summary_20261031.json`. An independently owned sequence of other isolated weight tests is running; **do not launch a second Unity test or forcibly replace its optimizer pause owner**. Because none of these individual screens has been repeated five times, the qualified profile remains 4500/1450 with mean 49.1622 seconds, 5/5 under 49.5, worst callback 18.1345ms.
