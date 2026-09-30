# Open-plane throttle sweep — pause handoff (2026-09-30)

## Status

The first Explore-simulator run was stopped cleanly at the user's request. Its experiment container closed the rosbag, ran the partial-data analyzer, and exited; the simulator was stopped. That data is preserved and must not be deleted. A seeded supplement is now running in a separate directory, resuming only the unusable/unstarted conditions.

- Run: `live_runs/openplane_throttle_5pct_5deg_20260930_r04/`
- Bag: `live_runs/openplane_throttle_5pct_5deg_20260930_r04/run/run_0.db3` (1,679,384,576 bytes)
- Analysis: `live_runs/openplane_throttle_5pct_5deg_20260930_r04/throttle_transition_analysis.json`
- Logs: `experiment.log`, `recorder.log`, `bridge.log`, and `analysis.log` in the run directory.
- Seed: `20260930`
- Supplement run: `live_runs/openplane_throttle_5pct_5deg_20260930_r05_resume/` (active; latest heartbeat reached condition 40/738, with its reset in progress)

## Test design

This is a randomized throttle **step-response surface**, not a gradual-ramp-versus-step comparison. It uses 13 steering commands from -30° to +30° in 5° increments. It covers 58 throttle transitions per steering angle, with two replicates (1,508 conditions total). The design includes 0%-start targets at 5% increments, plus transitions from 10% throttle baselines using 5/10/20/40 percentage-point steps and a full-throttle endpoint. Each condition has a 4 s baseline dwell and 8 s response observation, then a simulator reset. There is no speed or distance cap.

## Partial-run results

Stopped during phase index 770 (`steer=-0.3490667 rad`, throttle `0.00 -> 0.90`). The phase was correctly recorded as interrupted and is not fit-usable.

- 770 completed conditions; **all 770 are marked usable for response fitting** by the closed-bag analyzer.
- One incomplete condition; zero duplicate conditions; 737 not-yet-started conditions. To finish the planned surface without duplication, rerun the one interrupted condition plus those 737 absent conditions (738 conditions), after implementing a resume/supplement mode.
- All nine analyzed active streams passed the 40 Hz gate. Measured rates were approximately 39.987–40.004 Hz; p95 inter-message gaps were 25.09–25.82 ms; maximum gaps were 27.5–57.4 ms (the gate is 60 ms).
- Command tracking passed for 770/770 completed conditions; throttle-feedback tracking passed for 770/770; steering tracking passed for all 771 observed phases. Throttle endpoint-error p95 was about `2.38e-8` normalized.
- 771/771 requested resets recovered. Maximum reset position error was 1.6 mm; maximum condition-start spread from spawn was 0.1 mm.
- No collisions and no bridge timing-fault messages. Observed maximum speed was 21.7605 m/s and maximum displacement from spawn was 173.9957 m; these do not violate the requested uncapped test design.
- Across varied conditions, response speed change median was 1.213 m/s and p95 was 6.123 m/s. These are descriptive surface statistics, not yet a learned-model result.

`passed_integrity_gates` is `false` because the 1,508-condition sweep was intentionally interrupted, leaving 737 unstarted conditions and one incomplete condition. The completed data did **not** fail its stream, command/feedback, reset, collision, or timing checks. This resolves the earlier concern that the run might simply be collecting unusable data: the analyzed 770 completed conditions are usable under the current gates.

## Preliminary response fit

`tools/fit_open_plane_throttle_surface.py` fits the closed analysis JSON and writes `live_runs/openplane_throttle_5pct_5deg_20260930_r04/preliminary_fit.json`. It compares a mean baseline, quadratic ridge, and ExtraTrees using grouped out-of-fold predictions. Replicates of one steering/throttle condition stay together; additional evaluations hold out whole throttle transitions and whole steering angles.

- For 8-second speed change, ExtraTrees with commands plus measured baseline body state (`vx`, `vy`, yaw rate, speed, tilt) achieved RMSE **1.01 m/s**, MAE **0.52 m/s**, R² **0.857** on held-out replicate groups; RMSE **1.14 m/s**, R² **0.820** on held-out throttle transitions.
- Commands plus speed alone achieved RMSE **1.38 m/s**, R² **0.736** on held-out throttle transitions. Adding measured body state reduced that error by about 18%, evidence that the starting dynamic state carries useful information beyond throttle and steering commands.
- Feature ablation on held-out throttle transitions gives RMSE **1.19 m/s** with baseline lateral velocity+yaw added, and **1.18 m/s** with tilt magnitude added (both R² about **0.80**), versus 1.38 m/s without them. The tilt value is an unsigned magnitude derived from the odometry quaternion—not signed IMU roll—and this is predictive association, not proof that roll causes wheel slip.
- On a stricter whole-steering-angle holdout, the body-state ExtraTrees fit was near baseline (RMSE **2.66 m/s**, R² **0.015**). Do not claim generalization to unseen steering regimes or integrate this model into MPC/odometry.
- The measured throttle-feedback half-response time (`t50`) was not predicted better than the mean baseline (R² approximately zero or negative). The current sweep is step-only; it does not compare matched gradual ramps against steps.

This is preliminary evidence of a learnable response surface within sampled support, not proof of a useful full vehicle plant. The supplement should improve replication coverage; after it closes, refit with the combined unique conditions and preserve whole steering levels as holdouts.

## Resume guidance

Do not rerun all 1,508 conditions or overwrite either bag. Supplement support is implemented in `tools/open_plane_throttle_transition_surface.py` and `tools/run_open_plane_experiment.sh` through `SDU_APEX_EXPERIMENT_RESUME_ANALYSIS`. It validates the seed/design, skips conditions marked fit-usable in the source analysis, and records the exact scheduled subset so the analyzer can validate a partial-design bag. Preflight confirmed it skipped 770 conditions and scheduled the interrupted condition plus the 737 absent conditions. The active supplement is `live_runs/openplane_throttle_5pct_5deg_20260930_r05_resume/`; its latest logged position was condition 40/738, with a reset in progress. Keep the r04 bag immutable. Once r05 closes, verify its keys do not overlap the 770 usable r04 keys and that the union covers the complete 1,508-condition design with the requested replicate counts and quality gates.

If the next question is specifically the effect of throttle slew, add matched gradual-ramp and rapid-step conditions at the same starting state, steering, and final throttle. This step-response sweep alone cannot isolate that comparison.

## Relevant artifacts

- Analyzer: `tools/analyze_open_plane_throttle_transitions.py`
- Experiment: `tools/open_plane_throttle_transition_surface.py`
- Entrypoint and graceful recorder shutdown: `tools/run_open_plane_experiment.sh`
- Full per-condition results and missing-condition set: `live_runs/openplane_throttle_5pct_5deg_20260930_r04/throttle_transition_analysis.json`
