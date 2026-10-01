# Offline plant: next steps — 2026-09-30

This plan follows
`SDU_APEX_COMPLETE_OFFLINE_PLANT_IMPLEMENTATION_HANDOFF_2026-09-30.md` and
the measured state of the workspace. It is a modeling plan, not a claim that
the current surrogate is an accurate lap simulator.

## Current evidence

- The canonical mixed dataset remains unchanged: 93 clean whole-run records,
  491,519 aligned samples, split by complete run. Its training support ends at
  8.409 m/s.
- Phase-level salvage from the 24 rejected captures produced three separate
  archives under
  `live_runs/derived_dynamics_learning_20260928/boundary_phase_salvage_dataset_20260930/`:
  - `verified`: 1,804 sequences / 111,480 samples; 908 train and 896 test
    sequences across 15 runs. Only the phases marked valid by their experiment
    and passing local 40 Hz, collision, timing, alignment, and length gates
    belong here.
  - `speed_target_mismatch`: 45 sequences / 2,449 samples. Their telemetry
    passed local data-quality gates, but the speed-matching experiment failed;
    they are training-only diagnostics, not validation evidence.
  - `unverified`: 18 sequences / 9,372 samples, including unscored phases and
    bounded segments from older captures without phase markers. Original
    whole-run train/test grouping is preserved. Keep them out of the baseline
    until an explicit ablation shows value.
- The verified salvage reaches 8.34 m/s, below the canonical training maximum.
  The auxiliary groups reach about 8.47 m/s. This salvage increases clean
  sequence volume and coverage, but does **not** close the 9–10 m/s support gap
  or establish a prediction-accuracy gain.
- Current free-running mixed-GRU practice replays still have meter-scale drift
  (1.52–3.52 m radial RMSE over 12 laps). No model is ready for MPC or odometry.
- The r04/r05 throttle transition surface is now fully captured and audited:
  1,508/1,508 expected replicate keys, 754 conditions with two valid repeats.
  The closed r05 bag passes quality gates, all nine streams measured about
  39.981–40.004 Hz, and no collision or timing fault was recorded. Detailed
  metrics are in
  [`OPENPLANE_THROTTLE_SURFACE_RESULTS_20260930.md`](OPENPLANE_THROTTLE_SURFACE_RESULTS_20260930.md).
- A first grouped response-surface fit predicts 8-second speed change well
  when throttle transitions are held out (0.827 m/s RMSE), but fails to
  generalize to a held-out steering angle (2.829 m/s RMSE, R² -0.023). It is
  not a recursive plant model and does not close the high-steering gap.

## Ordered work

1. Export the full 40 Hz response sequences and a separate condition table.
   Each reset-delimited response is its own sequence; never train across a
   reset. Retain starting state, steering, throttle transition, replicate,
   reset ID, and quality evidence. Do not reduce the sweep to endpoint speed
   changes or a steering/throttle lookup.
2. Resolve whether `/ips` is an independent truth source and align it against
   `/odom` by frame and timestamp. The existing canonical labels are bridge
   `/odom`; do not call them simulator truth until that check is done.
3. Keep the canonical dataset and GRU checkpoint frozen as the baseline.
   Add only the `verified` salvage **training runs** to a new boundary-training
   view; keep its 896 sequence holdout untouched. Use the speed-mismatch and
   unverified archives only as explicitly named training ablations, never as
   clean validation. Keep all sequences from one source run in one fold.
4. Run the narrow model comparison from the handoff, not a broad architecture
   search: frozen plain GRU; explicit body-acceleration model without latent
   state; latent dimensions 2/4/8; and nominal dynamics plus learned residual.
   Use five whole-run grouped folds where data supports it. Keep actuator
   response, wheel-speed evolution, and history-dependent traction inside the
   recursive plant state. After initialization, inputs are commands and `dt`
   plus the model's own predicted state only—never future recorded sensors,
   feedback, or truth.
5. Keep rigid-body coupling and pose integration explicit. With the plant
   state expressed at the rear axle and COM offset `L=0.15532 m`, use
   `u_rear_dot=a_x_eff+r*(v_rear+L*r)`,
   `v_rear_dot=a_y_eff-r*u_rear-L*alpha_z_eff`, and
   `r_dot=alpha_z_eff`. Pose remains rear-axle referenced, verified from
   measured pose differences and twist.
   Treat wheel/tire quantities as effective learned dynamics unless the
   available labels identify actual per-wheel forces. Test Euler/Heun/midpoint
   only after choosing the model family, so integration and architecture
   changes are not confounded.
6. Score full free-running trajectories on held-out complete runs, including
   velocity/yaw error, position/heading drift, and error versus rollout time.
   Short-horizon improvement is not evidence of a trustworthy lap simulation.
   Freeze the model and thresholds before collecting fresh continuous
   open-plane and practice runs.
7. Only after free-running prediction is credible, build the closed-loop
   surrogate around the **current production** odometry/localization/MPC and
   actuator stack. The learned sensor observer is a separate experiment, not a
   prerequisite and not a runtime input to the plant teacher.

Do not start another broad factorial capture before the completed throttle
sweep is ingested and these held-out model evaluations identify a specific
missing feasible region. Do not integrate either GRU candidate into MPC or
odometry on the basis of short-window scores.
