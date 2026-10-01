# Offline plant training update — 2026-09-30

## Newer 3D teacher update — 2026-10-01

The schema-6 3D rigid-body data audit, frame/acceleration conventions, and
latest command-only teacher results are now recorded in
[`RIGID_BODY_TEACHER_PROGRESS_20261001.md`](RIGID_BODY_TEACHER_PROGRESS_20261001.md).
That later update supersedes this document for the 3D teacher status. The
existing 2D GRU remains the stronger full-run comparator; no new model is
integrated into MPC or odometry.

Current data layout and run status supersede the in-progress details below:
see [`VEHICLE_DYNAMICS_DATA_CATALOG_20260930.md`](VEHICLE_DYNAMICS_DATA_CATALOG_20260930.md).

## Bottom line

There is enough data to demonstrate useful local state prediction, but not to
claim an accurate long-horizon offline simulator. The next captures should be
longer and deliberately cover new *feasible* trajectories. More repeated
short probes or indiscriminate speed/steering Cartesian products will not
close the current gap.

The mixed dataset manifest at
`live_runs/derived_dynamics_learning_20260928/plant_teacher_mixed_dataset_20260930/manifest.json`
is the authoritative run inventory: 118 discovered bag records, 93 clean
records assigned to train/validation/test/final-test, one additional clean
record excluded because it came from a different simulator player, and 24
records rejected by quality/sequence gates. Two more bags could not be
imported because they lack steering/throttle command topics. Within the
learning splits, counts are 63 train, 3 validation, 25 test, and 2 final-test
records. These are bag counts, not independent long trajectories: many are
short or fragmented. The derived archive is about 30 MB; local training
environments are not datasets and should not be mistaken for run files.

The dataset exporter excludes rejected/aborted/stream-fault records from its
training and evaluation arrays, while retaining their quality dispositions in
the manifest. Some aborted captures contain complete pre-abort phases near
the high-speed boundary; those are being kept separate for phase-level
salvage analysis, not mixed into the clean archive. See the data catalog for
the current retained/discarded set.

## What the completed evaluation established

- The merged training archive has 491,519 samples. Its maximum training speed
  is 8.409 m/s; there is no empirical support at 9–10 m/s. Grid diagnostics
  show 517 of 660 nominal speed/signed-steering/throttle cells observed, but
  only 239 meet the coarse threshold of at least 100 samples from at least
  two bags. These nominal cells are not all physically feasible combinations.
- Existing open-plane probe phases are generally only 1.2–2.0 seconds. A new
  `SDU_APEX_EXPERIMENT_PROBE_DWELL_S` option lengthens validated probe phases
  only; default zero preserves old schedules. Large factorial
  `full_input_excitation` and `grid` profiles reject this override so they do
  not silently exceed the experiment timeout.
- The plain mixed-domain GRU remains the best current teacher candidate for
  unanchored full-run position. Its four held-out 12-lap practice replays have
  1.52–3.52 m radial position RMSE, 2.54–9.88 m endpoint error, and
  0.34–1.17 rad absolute final heading error. This is not negligible drift
  and is not acceptable as a trusted lap simulator yet.
- A structured model that predicts effective longitudinal/lateral/yaw
  accelerations had lower pointwise body-state error in some strata but
  worse whole-run heading and position behavior. Do not select it from
  pointwise RMSE alone.
- A first mixed-GRU candidate trained with integrated-heading loss (weight
  0.05, 5-second supervised rollouts) reduced absolute final heading error on
  all four held-out practice runs. However, its longitudinal-speed and
  yaw-rate RMSE roughly doubled and its position radial RMSE worsened on all
  four runs: 3.52 to 5.61 m, 2.75 to 4.80 m, 3.43 to 5.56 m, and 1.52 to
  6.27 m. Reject this checkpoint for full-run plant use. Correcting net
  heading alone can distort local dynamics and still worsen global position.
- Throttle has one useful but narrow confirmatory study that should not be
  repeated unchanged: 10 clean whole bags (5 at 4.5 m/s, 5 at 6.5 m/s), 24
  matched ramp/step pairs per bag, both turn directions, `|steer|=0.30/0.42`
  rad, and a `0.08` normalized-throttle change around bases `0.18` and `0.27`.
  The ramps last 0.30 s, the step/ramp probes last 1.8 s, and every one of
  the 480 phases passed command-profile tracking, throttle-feedback tracking,
  and the nominal 40 Hz receive gate. Median measured throttle half-response
  was about 0.09 s for steps and 0.21 s for ramps.
- That paired study found a short-lived response: step-minus-ramp change in
  mean absolute rear-wheel/kinematic-speed residual was `+0.596 m/s`
  (run-cluster 95% CI `+0.531` to `+0.663`) in 0.10–0.35 s, but `+0.0043 m/s`
  (CI `-0.0128` to `+0.0193`) in 0.35–0.65 s. By 0.65–1.00 s the estimate
  remained near zero (CI also spans zero). This is an encoder/kinematic
  residual proxy, not a direct tire-slip or force measurement. The paired
  lateral-acceleration estimate in the first window was also inconclusive
  (CI spans zero). See
  `live_runs/throttle_slew_pair_study_20260928/confirmatory_analysis.json`.
- Throttle coverage remains incomplete for the teacher: the usable mixed
  dataset contains no command above `0.50` normalized, despite the actuator
  interface allowing a higher command ceiling. The current open-plane
  excitation safety clamp is `0.50`. Training feedback coverage is concentrated
  in the 0.10–0.20 range; the 0.40–0.50 and 0.50+ feedback bins each contain
  under 9,000 training samples, and the 0.50+ bin here represents the ceiling,
  not evidence above 0.50. The existing ramp/step captures therefore establish
  local transient behavior, not a 0–100% throttle response surface.
- There is also coarse 0–0.50 throttle excitation crossed with many signed
  steering settings in the randomized full-input bags (throttle levels
  0, .08, .14, .20, .28, .36, .44, .50). Those 1.3–2.5 s phases let speed
  evolve freely, so they do not give a matched-start steady-state map at each
  throttle/steering/speed point. They are broad screening data, not dense
  independent evidence for every combination.
- The plain mixed-GRU teacher was scored on 11 held-out throttle-slew bags.
  This was 48 separate ~1.4 s probe sequences per run, each initialized with
  a true 16-frame history—not one continuous free-running rollout across the
  whole experiment. Mean across runs of per-probe `u` RMSE was about
  0.165 m/s at 4.5 m/s and 0.259 m/s at 6.5 m/s; radial position RMSE was
  about 0.17 m and 0.24 m, respectively. Worst single-probe position error
  approached 1.2 m. This says the model can track some local throttle pulses,
  but the tests do not establish whole-trajectory throttle accuracy.
- The four full-run practice captures are more informative than overlapping
  short windows, but remain a small run-level evaluation set. Do not tune
  against them repeatedly; candidate selection belongs on train/validation
  runs, with newly captured final runs held back.

No model is connected to MPC, odometry, AMCL, or EKF. No simulator physics or
competition runtime topic policy was changed. The open-plane runner is
development-only and uses truth only for experiment safety and offline
labels.

## Recommended next captures

1. **Longer matched open-plane probes, first in an established safe band.**
   Use a 4-second probe dwell at a time, both steering signs, randomized
   condition order, and existing matched-start, speed, and steering-feedback
   validation. Start with the 4.5 m/s transition profile. Follow with a
   separate 6.5 m/s run focused on its measured transition band and nearby
   support points. These should be separate bags. Do not combine every speed
   into one run or test full lock at high speed. Preserve existing tilt,
   collision, odometry-timeout, and bridge-timing aborts.
2. **Broaden throttle/steering transitions efficiently.** The refined
   development-only `throttle_transition_surface` design samples targets every
   5% from zero, plus 5%, 10%, 20%, 40% and full-throttle increases from each
   10% throttle baseline. It spans 13 steering commands at 5° increments from
   -30° to +30°, with two independently reset and randomized replicates. This
   is 1,508 conditions: denser than a 10% throttle grid while adding much finer
   steering coverage, without spending most of the capture on a single
   steering slice. Each condition begins at the same built-in-reset spawn,
   records a 4-second baseline at the start throttle/steering, then an 8-second
   response after the throttle step, and resets. Do not require body-motion
   equilibrium: high-steering telemetry was strongly oscillatory, so a steady
   gate could strand the test indefinitely. The bounded observation window
   captures that transient; it is not a speed or distance cap.
   Rollover marks a condition incomplete and triggers reset; the first
   collision or bridge timing fault aborts the capture. Actual throttle
   feedback is logged alongside commands and its tracking error is scored
   separately, preserving deadband and quantization observations. The capture
   also includes odometry, encoders, IMU, steering, collision and timing data.
   The first exhaustive straight-only attempt is retained as an incomplete
   pilot, not training data. The refined multi-steering capture is the current
   next run. Every capture needs closed-bag cadence,
   collision, reset-count, feedback-tracking and per-condition analysis before
   it is considered training data. The older narrow ramp/step study should
   not be repeated unchanged.
3. **Capture complete racing trajectories on multiple feasible racelines.**
   Keep vehicle stack and map fixed while varying raceline; pair each
   candidate with a known-safe baseline at the same speed cap. Capture enough
   continuous laps to include sustained corner entry, steady cornering, exit,
   and straights. Increase the cap by 0.25–0.5 m/s only after every run at
   the previous cap passes the existing no-collision and data-quality gates.
   Stop immediately on the first collision or tilt-guard trip. Use separate
   bags and independent seeds/runs; repetitions within one bag are not
   independent run-level evidence.
4. **Use the observed feasible envelope, not a rectangular grid.** The
   existing OpenPlane ceiling is 8.409 m/s; earlier high-speed steering
   probes encountered a tilt-limit abort. Do not infer that 10 m/s at large
   steering is attainable. Test new speed/curvature points only where current
   data and completed safe runs support them; report unsupported regions
   explicitly.
5. **Make each run count.** Record odometry/truth labels, actuator feedback
   and commands, encoders, IMU, collision count, bridge packet timing, and
   phase/lap markers. Require complete command/state alignment and acceptable
   nominal 40 Hz stream quality before a run enters training. Assign entire
   runs—never windows from a run—to train, validation, or test. Hold out at
   least one complete raceline/speed combination until model selection is
   finished.

## Reset-isolated throttle matrix: current run

The earlier 0–100% straight setpoint draft was removed; it did not satisfy the
requested pairwise `i% -> j%` design. The refined surface covers 58 selected
strictly increasing transitions at each of 13 steering angles, with two
replicates (1,508 conditions). Each condition uses a 4-second baseline and
8-second step-response observation, then resets to the same spawn. There is
no driving speed or distance cap; speed and displacement are recorded as
outcomes. Rollover invalidates only that condition and requests reset; any
collision or bridge timing fault aborts the whole capture. Competition launch
still forces the development reset input off; the runtime topic-policy check
passes.

Real-simulator reset preflight completed in Explore batch mode at full throttle
for 10 seconds: maximum speed was 18.4232 m/s, measured travel was 129.2301 m,
the car returned to within 0.0002 m of its initial spawn, no collision or
timing-fault abort occurred, and active odometry measured 39.9969 Hz with a
25.4549 ms p95 gap. This confirms the reset path after a high-speed, long
displacement probe; it does not impose a distance limit.

The exhaustive straight-only attempt was stopped at 237 valid pairs after the
estimated full sweep approached 17 hours. Its 454 MB bag is preserved at
`live_runs/openplane_throttle_transitions_straight_20260930_r01/`; it is
incomplete and must not be used as a full-surface training capture. The bag
closed, but the first manual-stop path exposed a redundant `rclpy.shutdown()`
traceback and the post-run audit lacked the workspace on `PYTHONPATH`; the
signal and analyzer paths have since been corrected. The reset
preflight remains clean: full throttle for 10 seconds reached 18.4232 m/s and
129.2301 m travel, then returned within 0.0002 m of spawn, with no collision or
timing fault and 39.9969 Hz active odometry.

The initial `r01` and `r02` throttle-transition pilots were discarded after
they produced no fit-usable response conditions. `r03` remains separate and
pending condition-level review because its first response was prolonged and
did not complete normally. The replacement r04 capture completed 770
fit-usable conditions and one interrupted condition; its overall design gate
is false because the r04 capture was partial. The r05 supplement completed
the remaining 738 conditions. The combined reports now cover all 1,508
expected replicate keys, with two fit-usable repeats for each of 754
steering/throttle conditions. r05 passed its closed-bag checks: active streams
were about 39.981–40.004 Hz, all scheduled conditions tracked their commands
and actuator feedback, resets returned within 1.6 mm, and there were no
collisions or timing faults. The analyzer's scheduled-condition post-run path
had a `NameError`; it was fixed and rerun successfully against the closed bag.
The first combined grouped fit predicts 8-second speed change well on held-out
throttle transitions (0.827 m/s RMSE) but does not generalize to a held-out
steering angle (2.829 m/s RMSE, R² -0.023). This is not yet a recursive plant
model or a solution to high-angle prediction. Full results and artifacts are
in [`OPENPLANE_THROTTLE_SURFACE_RESULTS_20260930.md`](OPENPLANE_THROTTLE_SURFACE_RESULTS_20260930.md).

Restarts after the exhaustive pilot exposed a reset edge case: the
old process had been stopped while the car was approximately 573 m from its
actual reset spawn and moving at 29.5 m/s. The next process mistakenly treated
that pre-reset pose as the spawn and a 5-second reset wait expired. The harness
now learns the spawn from stable post-reset telemetry and allows up to 30
seconds for reset recovery; this is not a driving timeout or distance limit.
The runner's background driver also now explicitly handles SIGINT/SIGTERM so
stops close the bag and write a completion/abort event.

Longer probes improve estimation of slower transients and make history
dependence observable; racing trajectories supply the correlated combinations
the optimizer actually requests. Both are needed. Open-plane excitation alone
cannot teach the plant every closed-loop racing trajectory, and track-only
logs may not isolate the cause of a mismatch.

## Reproduction notes

Use the normal Explore batch-mode simulator workflow and then the development
recorder/bridge experiment runner. Do not start a GUI or use `-no-graphics`.
Example for one matched, moderate-speed transition run:

```bash
SDU_APEX_EXPERIMENT_PROFILE=isolated_transition_45mps \
SDU_APEX_EXPERIMENT_PROBE_DWELL_S=4 \
SDU_APEX_EXPERIMENT_TIMEOUT_S=1200 \
./tools/run_open_plane_experiment.sh
```

For a separate 6.5 m/s transition-band run, use
`SDU_APEX_EXPERIMENT_PROFILE=isolated_transition_65mps` with a new run ID.
Each profile has randomized signed-steering order and three within-run
repetitions. The dwell override applies only to phases tagged for validation.
A run must pass the existing recorded quality gate before dataset
construction; failed and aborted runs are preserved but are not training
inputs.

Training artifacts and held-out replay comparisons live under
`live_runs/derived_dynamics_learning_20260928/`, in directories named
`plant_teacher_mixed_gru_5s_20260930`,
`plant_teacher_mixed_structured_euler_5s_20260930`, and
`plant_teacher_mixed_gru_heading_5s_20260930`.
