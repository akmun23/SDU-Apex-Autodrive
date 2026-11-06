# MPC practice: fixed heading 1450, working baseline 4000 (31 October 2026)

- Previous 4500/1450 remains the five-race qualified rollback: 49.162, 49.091, 49.175, 49.291, 49.092 s, no collisions or deadline misses.
- New 4000/1450 with feedback 0.75 is the experimental working baseline: one complete 49.071 s ten-lap race, one recovery, zero missed deadlines, 17.936 ms max callback.
- New tuning file: f1tenth_mpc/config/mpc_practice_feedback_tuning_baseline_20261031.yaml.
- Keep heading weight fixed at 1450, both SQP passes, exact 750 ms source rollout, 1 ms internal integration, map/corridor safety and deadline enforcement.
- Changes must move forward from the fastest qualifying experimental configuration; do not modify the independently five-race qualified rollback.

## Individual screens

| Parameter | Lateral | Heading | Ten-lap seconds | Recoveries | Max callback ms |
|---|---:|---:|---:|---:|---:|
| weight_e_y=4000.0 | 4000.0 | 1450 | 49.071 | 1 | 17.936 |
| weight_e_y=3700.0 | 3700.0 | 1450 | 49.155 | 0 | 18.172 |
| weight_e_y=3300.0 | 3300.0 | 1450 | 49.140 | 0 | 18.081 |
| weight_steering_slew=1450.0 | 4500.0 | 1450 | 49.141 | 1 | 18.067 |

Note: steering-slew=1450 started before the baseline promotion and used lateral 4500. It is not a direct comparative trial on the 4000 baseline. Remaining subsequent screens use lateral 4000.

## Reproducibility and algorithmic investigation

- Per-run full ROS bags and diagnostic CSV/JSONL stored under live_runs/practice_fixed_heading1450_*_20261031_01.
- Screening log: /tmp/sdu_mpc_weight_screens_fixed_heading_20261031.log. Aggregate: live_runs/mpc_collision_margin_experiment_20261030/fixed_heading1450_weight_screen_summary_20261031.json.
- Candidate independent five-race repeatability run: /tmp/sdu_mpc_fixed_heading1450_repeat_best_20261031.py; outputs in the same experiment directory.
- Fixed-heading profile: live_runs/mpc_collision_margin_experiment_20261030/fixed_heading1450_winner_line_search_sectors_20261031.json.
- Offline checks on 60 recorded states: live_runs/mpc_collision_margin_experiment_20261030/offline_fixed_heading1450_rti_policies_20261031/comparison.json. Those are not live timing certification.
- Next algorithmic priorities: improve SQP step acceptance, fewer physically validated full backtracks, feasibility restoration in hard corners, and proactive terminal viability. Do not parallelize native plant until shared mutable state is made thread safe.
